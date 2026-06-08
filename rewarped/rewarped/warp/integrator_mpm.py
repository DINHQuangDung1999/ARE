import warp as wp
from warp.sim.integrator import Integrator
from warp.sim.model import Control, Model, State


class MPMIntegrator(Integrator):
    def __init__(self, model):
        pass

    def simulate(self, model: Model, state_in: State, state_out: State, dt: float, control: Control = None):
        device = model.device

        if control is None:
            control = model.control(clone_variables=False)

        num_envs = model.num_envs
        mpm_model = model.mpm_model
        statics = mpm_model.statics
        constant = mpm_model.constant
        p_curr = state_in.mpm_particle
        p_next = state_out.mpm_particle
        grid = state_in.mpm_grid

        body_curr, body_next = state_in.body_q, state_out.body_q

        num_grids = constant.num_grids
        num_particles = p_curr.x.shape[0]
        grid_op_name = model.mpm_model.grid_op_name

        # zero grid
        grid.clear()
        # grid.zero_grad()

        wp.launch(mpm_model.eval_stress, dim=num_particles, inputs=[constant, statics, p_curr], device=device)
        wp.launch(mpm_model.p2g, dim=num_particles, inputs=[constant, statics, p_curr, grid], device=device)
        if grid_op_name == "dexdeform":
            wp.launch(
                kernel=mpm_model.grid_op,
                dim=[num_envs] + [num_grids] * 3,
                inputs=[
                    constant,
                    grid,
                    model.shape_transform,
                    model.shape_body,
                    model.shape_geo,
                    (model.shape_count - 1) // num_envs,
                    body_curr,
                    body_next,
                ],
                device=device,
            )
        elif grid_op_name in ("freeslip", "noslip"):
            raise NotImplementedError
            wp.launch(mpm_model.grid_op, dim=[num_grids] * 3, inputs=[constant, grid], device=device)
        else:
            raise ValueError(grid_op_name)
        wp.launch(mpm_model.g2p, dim=num_particles, inputs=[constant, statics, p_curr, p_next, grid], device=device)

        return state_out
