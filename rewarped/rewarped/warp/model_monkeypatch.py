import types

import warp as wp
from warp.sim.model import Control, Model, State


def get_copy_fn(copy):
    if copy == "clone":
        return wp.clone  # NOTE: will copy array.grad, https://github.com/NVIDIA/warp/issues/272#issuecomment-2239791350
    elif copy == "zeros":
        return wp.zeros_like
    elif copy == "empty":
        return wp.empty_like
    else:
        raise ValueError(copy)


def Model_state(self: Model, requires_grad=None, copy="clone", integrator_type=None) -> State:
    s = State()
    if requires_grad is None:
        requires_grad = self.requires_grad

    # s.requires_grad = requires_grad
    # s.particle_count = self.particle_count
    # s.body_count = self.body_count
    # s.joint_count = self.joint_count

    copy_fn = get_copy_fn(copy)

    # particles
    if self.particle_count:
        s.particle_q = copy_fn(self.particle_q, requires_grad=requires_grad)
        s.particle_qd = copy_fn(self.particle_qd, requires_grad=requires_grad)
        if copy == "clone":
            s.particle_f = wp.zeros_like(self.particle_qd, requires_grad=requires_grad)
        else:
            s.particle_f = copy_fn(self.particle_qd, requires_grad=requires_grad)

    # articulations
    if self.body_count:
        s.body_q = copy_fn(self.body_q, requires_grad=requires_grad)
        s.body_qd = copy_fn(self.body_qd, requires_grad=requires_grad)
        if copy == "clone":
            s.body_f = wp.zeros_like(self.body_qd, requires_grad=requires_grad)
        else:
            s.body_f = copy_fn(self.body_qd, requires_grad=requires_grad)

    # joints
    if self.joint_count:
        # joint state
        s.joint_q = copy_fn(self.joint_q, requires_grad=requires_grad)
        s.joint_qd = copy_fn(self.joint_qd, requires_grad=requires_grad)

    if integrator_type == "featherstone":  # FeatherstoneIntegrator.allocate_state_aux_vars
        # allocate auxiliary variables that vary with state
        if self.body_count:
            # joints
            s.joint_qdd = wp.zeros_like(self.joint_qd, requires_grad=requires_grad)
            s.joint_tau = wp.zeros_like(self.joint_qd, requires_grad=requires_grad)
            if requires_grad:
                # used in the custom grad implementation of eval_dense_solve_batched
                s.joint_solve_tmp = wp.zeros_like(self.joint_qd, requires_grad=True)
            else:
                s.joint_solve_tmp = None
            s.joint_S_s = wp.empty(
                (self.joint_dof_count,),
                dtype=wp.spatial_vector,
                device=self.device,
                requires_grad=requires_grad,
            )

            # derived rigid body data (maximal coordinates)
            B = self.body_count
            s.body_q_com = wp.empty((B,), dtype=wp.transform, device=self.device, requires_grad=requires_grad)
            s.body_I_s = wp.empty((B,), dtype=wp.spatial_matrix, device=self.device, requires_grad=requires_grad)
            s.body_v_s = wp.empty((B,), dtype=wp.spatial_vector, device=self.device, requires_grad=requires_grad)
            s.body_a_s = wp.empty((B,), dtype=wp.spatial_vector, device=self.device, requires_grad=requires_grad)
            s.body_f_s = wp.zeros((B,), dtype=wp.spatial_vector, device=self.device, requires_grad=requires_grad)
            s.body_ft_s = wp.zeros((B,), dtype=wp.spatial_vector, device=self.device, requires_grad=requires_grad)

        # allocate mass, Jacobian matrices, and other auxiliary variables pertaining to the model
        if self.joint_count:
            # system matrices
            s.M = wp.zeros((self.fs_M_size,), dtype=wp.float32, device=self.device, requires_grad=requires_grad)
            s.J = wp.zeros((self.fs_J_size,), dtype=wp.float32, device=self.device, requires_grad=requires_grad)
            s.P = wp.empty_like(s.J, requires_grad=requires_grad)
            s.H = wp.empty((self.fs_H_size,), dtype=wp.float32, device=self.device, requires_grad=requires_grad)

            # zero since only upper triangle is set which can trigger NaN detection
            s.L = wp.zeros_like(s.H)

        s._featherstone_augmented = True

    if integrator_type == "mpm":
        if copy == "clone":
            s.mpm_particle = self.mpm_state.particle.clone(requires_grad)
            s.mpm_grid = self.mpm_model.grid.clone(requires_grad)
        elif copy == "zeros":
            s.mpm_particle = self.mpm_state.particle.zeros(requires_grad)
            s.mpm_grid = self.mpm_model.grid.zeros(requires_grad)
        elif copy == "empty":
            s.mpm_particle = self.mpm_state.particle.empty(requires_grad)
            s.mpm_grid = self.mpm_model.grid.empty(requires_grad)
        else:
            raise ValueError(copy)

        # add references
        s.mpm_x = s.mpm_particle.x
        s.mpm_v = s.mpm_particle.v
        s.mpm_C = s.mpm_particle.C
        s.mpm_F_trial = s.mpm_particle.F_trial
        s.mpm_F = s.mpm_particle.F
        s.mpm_stress = s.mpm_particle.stress

        s.mpm_grid_v = s.mpm_grid.v
        s.mpm_grid_mv = s.mpm_grid.mv
        s.mpm_grid_m = s.mpm_grid.m

    return s


def Control_clear_acts(self: Control):
    if self.joint_act:
        self.joint_act.zero_()
    if self.tri_activations:
        self.tri_activations.zero_()
    if self.tet_activations:
        self.tet_activations.zero_()
    if self.muscle_activations:
        self.muscle_activations.zero_()


def Model_control(self: Model, requires_grad=None, copy="clone") -> Control:
    c = Control(self)
    if requires_grad is None:
        requires_grad = self.requires_grad
    if copy is not None:
        copy_fn = get_copy_fn(copy)

        if self.joint_count:
            c.joint_act = copy_fn(self.joint_act, requires_grad=requires_grad)
        if self.tri_count:
            c.tri_activations = copy_fn(self.tri_activations, requires_grad=requires_grad)
        if self.tet_count:
            c.tet_activations = copy_fn(self.tet_activations, requires_grad=requires_grad)
        if self.muscle_count:
            c.muscle_activations = copy_fn(self.muscle_activations, requires_grad=requires_grad)
    else:
        c.joint_act = self.joint_act
        c.tri_activations = self.tri_activations
        c.tet_activations = self.tet_activations
        c.muscle_activations = self.muscle_activations

    # monkeypatch add clear_acts() method
    c.clear_acts = types.MethodType(Control_clear_acts, c)
    return c
