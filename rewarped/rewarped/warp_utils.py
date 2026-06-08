import warp as wp


@wp.kernel
def eval_kinematic_joints(
    joint_q: wp.array(dtype=float),
    joint_act: wp.array(dtype=float),
    p: float,
    # outputs
    joint_q_next: wp.array(dtype=float),
):
    tid = wp.tid()
    joint_q_next[tid] = joint_q[tid] + p * joint_act[tid]  # relative


def sim_eval_kinematic_fk(model, state_in, state_out, sim_dt, sim_substeps, control):
    wp.launch(
        kernel=eval_kinematic_joints,
        dim=model.joint_axis_count,
        inputs=[state_in.joint_q, control.joint_act, float(1.0 / sim_substeps)],
        outputs=[state_out.joint_q],
        device=model.device,
    )
    wp.sim.eval_fk(model, state_out.joint_q, state_out.joint_qd, None, state_out)


def sim_update(sim_params, model, states, control):
    integrator, sim_substeps, sim_dt, eval_kinematic_fk, eval_ik = sim_params
    state_in, states_mid, state_out = states

    state_0 = state_in
    for i in range(sim_substeps):
        if i == sim_substeps - 1:
            state_1 = state_out
        else:
            state_1 = states_mid[i] if states_mid is not None else model.state(copy="zeros")

        if eval_kinematic_fk:
            sim_eval_kinematic_fk(model, state_0, state_1, sim_dt, sim_substeps, control)

        state_0.clear_forces()
        wp.sim.collide(model, state_0)
        integrator.simulate(model, state_0, state_1, sim_dt, control=control)
        state_0 = state_1

    if eval_ik:
        wp.sim.eval_ik(model, state_out, state_out.joint_q, state_out.joint_qd)


def sim_update_inplace(sim_params, model, states, control):
    integrator, sim_substeps, sim_dt, eval_kinematic_fk, eval_ik = sim_params
    state_0, state_1 = states

    for i in range(sim_substeps):
        if eval_kinematic_fk:
            sim_eval_kinematic_fk(model, state_0, state_1, sim_dt, sim_substeps, control)

        state_0.clear_forces()
        wp.sim.collide(model, state_0)
        integrator.simulate(model, state_0, state_1, sim_dt, control=control)
        state_0, state_1 = state_1, state_0

    if eval_ik:
        wp.sim.eval_ik(model, state_0, state_0.joint_q, state_0.joint_qd)

    return (state_0, state_1)
