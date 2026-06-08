import warp as wp


MATL_PLBPLASTICINE = wp.constant(0)
MATL_PLASTICINE = wp.constant(1)
MATL_WATER = wp.constant(2)
MATL_SAND = wp.constant(3)
MATL_NEOHOOKEAN = wp.constant(4)
MATL_COROTATED = wp.constant(5)


@wp.struct
class MPMMaterial:
    name: int = 0
    E: float = None  # Young's modulus
    nu: float = None  # Poisson's ratio

    mu: float = None
    lam: float = None

    yield_stress: float = None
    cohesion: float = None
    alpha: float = None


def get_material(
        name: str,
        E: float = None,
        nu: float = None,
        yield_stress: float = None,
        cohesion: float = None,
        alpha: float = None,
) -> MPMMaterial:
    material = MPMMaterial()
    if name in ('plb_plasticine', 'plasticine'):
        material.name = MATL_PLASTICINE
        material.E = E
        material.nu = nu
        material.mu, material.lam = get_lame(E, nu)

        material.yield_stress = yield_stress
    elif name == 'water':
        material.name = MATL_WATER
        material.E = E
        material.nu = nu
        material.mu, material.lam = get_lame(E, nu)
    elif name == 'sand':
        material.name = MATL_SAND
        material.E = E
        material.nu = nu
        material.mu, material.lam = get_lame(E, nu)

        material.cohesion = cohesion
        material.alpha = alpha
    elif name == 'neohookean':
        material.name = MATL_NEOHOOKEAN
        material.E = E
        material.nu = nu
        material.mu, material.lam = get_lame(E, nu)
    elif name == 'corotated':
        material.name = MATL_COROTATED
        material.E = E
        material.nu = nu
        material.mu, material.lam = get_lame(E, nu)
    else:
        raise ValueError(type)
    return material


def get_lame(E, nu):
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return mu, lam


@wp.func
def svd(F: wp.mat33):
    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sigma = wp.vec3(0.0)
    wp.svd3(F, U, sigma, V)

    U_det = wp.determinant(U)
    V_det = wp.determinant(V)

    if U_det < 0.0:
        U = wp.mat33(
            U[0, 0], U[0, 1], -U[0, 2],
            U[1, 0], U[1, 1], -U[1, 2],
            U[2, 0], U[2, 1], -U[2, 2],
        )
        sigma = wp.vec3(sigma[0], sigma[1], -sigma[2])
    if V_det < 0.0:
        V = wp.mat33(
            V[0, 0], V[0, 1], -V[0, 2],
            V[1, 0], V[1, 1], -V[1, 2],
            V[2, 0], V[2, 1], -V[2, 2],
        )
        sigma = wp.vec3(sigma[0], sigma[1], -sigma[2])

    Vh = wp.transpose(V)
    return U, sigma, Vh


@wp.func
def identity_deformation(F_trial: wp.mat33, material: MPMMaterial):
    return F_trial


@wp.func
def plasticine_deformation(F_trial: wp.mat33, material: MPMMaterial):  # von mises
    U, sigma, Vh = svd(F_trial)

    threshold = 0.01
    sigma = wp.vec3(wp.max(sigma[0], threshold), wp.max(sigma[1], threshold), wp.max(sigma[2], threshold))

    epsilon = wp.vec3(wp.log(sigma[0]), wp.log(sigma[1]), wp.log(sigma[2]))
    epsilon_trace = epsilon[0] + epsilon[1] + epsilon[2]
    epsilon_bar = epsilon - wp.vec3(epsilon_trace / 3.0, epsilon_trace / 3.0, epsilon_trace / 3.0)
    epsilon_bar_norm = wp.length(epsilon_bar) + 1e-5

    delta_gamma = epsilon_bar_norm - material.yield_stress / (2.0 * material.mu)

    if delta_gamma > 0.0:
        yield_epsilon = epsilon - (delta_gamma / epsilon_bar_norm) * epsilon_bar
        yield_sigma = wp.mat33(
            wp.exp(yield_epsilon[0]), 0.0, 0.0,
            0.0, wp.exp(yield_epsilon[1]), 0.0,
            0.0, 0.0, wp.exp(yield_epsilon[2]),
        )
        F_corrected = U * yield_sigma * Vh
        return F_corrected
    else:
        return F_trial


@wp.func
def water_deformation(F_trial: wp.mat33, material: MPMMaterial):
    J = wp.determinant(F_trial)
    Je_1_3 = wp.pow(J, 1.0 / 3.0)
    F_corrected = wp.diag(wp.vec3(Je_1_3, Je_1_3, Je_1_3))
    return F_corrected


@wp.func
def sand_deformation(F_trial: wp.mat33, material: MPMMaterial):
    U, sigma, Vh = svd(F_trial)

    threshold = 0.05
    sigma = wp.vec3(wp.max(sigma[0], threshold), wp.max(sigma[1], threshold), wp.max(sigma[2], threshold))

    epsilon = wp.vec3(wp.log(sigma[0]), wp.log(sigma[1]), wp.log(sigma[2]))
    trace = epsilon[0] + epsilon[1] + epsilon[2]
    epsilon_hat = epsilon - wp.vec3(trace / 3.0, trace / 3.0, trace / 3.0)
    epsilon_hat_norm = wp.length(epsilon_hat)

    # drucker prager
    shifted_trace = trace - material.cohesion * 3.0
    if shifted_trace < 0.0:
        shift = (3.0 * material.lam + 2.0 * material.mu) / (2.0 * material.mu) * shifted_trace * material.alpha
        delta_gamma = epsilon_hat_norm + shift
        inv_n = (wp.max(delta_gamma, 0.0) / epsilon_hat_norm)
        compress_epsilon = epsilon - wp.vec3(inv_n * epsilon_hat[0], inv_n * epsilon_hat[1], inv_n * epsilon_hat[2])
        eps = wp.vec3(wp.exp(compress_epsilon[0]), wp.exp(compress_epsilon[1]), wp.exp(compress_epsilon[2]))
    else:
        expand_epsilon = wp.vec3(material.cohesion, material.cohesion, material.cohesion)
        eps = wp.vec3(wp.exp(expand_epsilon[0]), wp.exp(expand_epsilon[1]), wp.exp(expand_epsilon[2]))

    F_corrected = U * wp.diag(eps) * Vh
    return F_corrected



@wp.func
def sigma_elasticity(F: wp.mat33, material: MPMMaterial):
    U, sigma, Vh = svd(F)

    threshold = 1e-5
    sigma = wp.vec3(wp.max(sigma[0], threshold), wp.max(sigma[1], threshold), wp.max(sigma[2], threshold))

    epsilon = wp.vec3(wp.log(sigma[0]), wp.log(sigma[1]), wp.log(sigma[2]))
    trace = epsilon[0] + epsilon[1] + epsilon[2]
    tau = 2.0 * material.mu * epsilon + wp.vec3(material.lam * trace)

    stress = U * wp.diag(tau) * wp.transpose(U)
    return stress


@wp.func
def volume_elasticity(F: wp.mat33, material: MPMMaterial):
    J = wp.determinant(F)
    I = wp.identity(n=3, dtype=float)
    stress = material.lam * J * (J - 1.0) * I
    return stress


@wp.func
def neohookean_elasticity(F: wp.mat33, material: MPMMaterial):
    I = wp.identity(n=3, dtype=float)
    Ft = wp.transpose(F)
    FFt = F * Ft
    J = wp.determinant(F)

    kirchhoff_stress = material.mu * (FFt - I) + material.lam * wp.log(J) * I
    return kirchhoff_stress


@wp.func
def corotated_elasticity(F: wp.mat33, material: MPMMaterial):
    I = wp.identity(n=3, dtype=float)
    U, sigma, Vh = svd(F)
    Ft = wp.transpose(F)
    R = U * Vh
    J = sigma[0] * sigma[1] * sigma[2]

    corotated_stress = 2.0 * material.mu * (F - R) * Ft
    volume_stress = material.lam * (J - 1.0) * J * I
    kirchhoff_stress = corotated_stress + volume_stress
    return kirchhoff_stress
