import torch

import warp as wp

from .warp_utils import sim_update


# for checkpointing method
def assign_tensors(x, x_out, names, tensors):
    # need to assign b/c state_0, state_1 cannot be swapped
    # TODO: Add fn to get wp.array attributes instead of vars(..)
    for name in vars(x):
        if name in names:
            continue
        attr = getattr(x, name)
        if isinstance(attr, wp.array):
            wp_array = getattr(x_out, name)
            wp_array.assign(attr)
    for name, tensor in zip(names, tensors):
        # assert not torch.isnan(tensor).any(), print("NaN tensor", name)
        wp_array = getattr(x_out, name)
        wp_array.assign(wp.from_torch(tensor, dtype=wp_array.dtype))


def assign_adjoints(x, names, adj_tensors):
    # register outputs with tape
    for name, adj_tensor in zip(names, adj_tensors):
        # assert not torch.isnan(adj_tensor).any(), print("NaN adj", name)
        wp_array = getattr(x, name)
        # wp_array.grad = wp.from_torch(adj_tensor, dtype=wp_array.dtype)
        wp_array.grad.assign(wp.from_torch(adj_tensor, dtype=wp_array.dtype))


class UpdateFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        update_params,
        sim_params,
        states,
        control,
        states_bwd,
        control_bwd,
        state_tensors_names,
        control_tensors_names,
        *tensors,
    ):
        tape, integrator, model, use_graph_capture, synchronize = update_params
        sim_substeps, sim_dt, kinematic_fk, eval_ik = sim_params
        state_in, states_mid, state_out = states
        state_in_bwd, states_mid_bwd, state_out_bwd = states_bwd

        num_state = len(state_tensors_names)
        state_tensors, control_tensors = tensors[:num_state], tensors[num_state:]

        if synchronize:
            # ensure Torch operations complete before running Warp
            wp.synchronize_device()

        if tape is None:
            tape = wp.Tape()
            update_params = (tape, *update_params[1:])

        # for name in vars(model):
        #     attr = getattr(model, name)
        #     if isinstance(attr, wp.array):
        #         # print(name)
        #         attr.requires_grad = True

        ctx.update_params = update_params
        ctx.sim_params = sim_params
        ctx.states = states
        ctx.control = control
        ctx.states_bwd = states_bwd
        ctx.control_bwd = control_bwd
        ctx.state_tensors_names = state_tensors_names
        ctx.control_tensors_names = control_tensors_names
        ctx.state_tensors = state_tensors
        ctx.control_tensors = control_tensors

        if use_graph_capture:
            if getattr(integrator, "update_graph", None) is None:
                assert getattr(integrator, "bwd_update_graph", None) is None

                device = wp.get_device()
                # make torch use the warp stream from the given device
                torch_stream = wp.stream_to_torch(device)

                # capture graph
                with wp.ScopedDevice(device), torch.cuda.stream(torch_stream):
                    wp.capture_begin(force_module_load=False)
                    try:
                        with tape:
                            sim_update(update_params, sim_params, states_bwd, control_bwd)
                    finally:
                        integrator.update_graph = wp.capture_end()

                    wp.capture_begin(force_module_load=False)
                    try:
                        tape.backward()
                    finally:
                        integrator.bwd_update_graph = wp.capture_end()

            assign_tensors(state_in, state_in_bwd, state_tensors_names, state_tensors)
            assign_tensors(control, control_bwd, control_tensors_names, control_tensors)
            wp.capture_launch(integrator.update_graph)
            assign_tensors(state_out_bwd, state_out, [], [])  # write to state_out
        else:
            with tape:
                sim_update(update_params, sim_params, states, control)

        if synchronize:
            # ensure Warp operations complete before returning data to Torch
            wp.synchronize_device()

        outputs = []
        for name in state_tensors_names:
            out_tensor = wp.to_torch(getattr(state_out, name))
            # assert not torch.isnan(out_tensor).any(), print("NaN fwd", name)
            if use_graph_capture:
                out_tensor = out_tensor.clone()
            outputs.append(out_tensor)
        for name in control_tensors_names:
            out_tensor = wp.to_torch(getattr(control, name))
            # assert not torch.isnan(out_tensor).any(), print("NaN fwd", name)
            if use_graph_capture:
                out_tensor = out_tensor.clone()
            outputs.append(out_tensor)

        return tuple(outputs)

    @staticmethod
    def backward(ctx, *adj_tensors):
        update_params = ctx.update_params
        sim_params = ctx.sim_params
        states = ctx.states
        control = ctx.control
        states_bwd = ctx.states_bwd
        control_bwd = ctx.control_bwd
        state_tensors_names = ctx.state_tensors_names
        control_tensors_names = ctx.control_tensors_names
        state_tensors = ctx.state_tensors
        control_tensors = ctx.control_tensors

        tape, integrator, model, use_graph_capture, synchronize = update_params
        sim_substeps, sim_dt, kinematic_fk, eval_ik = sim_params
        state_in, states_mid, state_out = states
        state_in_bwd, states_mid_bwd, state_out_bwd = states_bwd

        rescale_grad = None
        clip_grad = None
        zero_nans = False
        # rescale_grad = sim_substeps
        # clip_grad = 1.0
        # zero_nans = True

        # ensure grads are contiguous in memory
        adj_tensors = [adj_tensor.contiguous() for adj_tensor in adj_tensors]

        num_state = len(state_tensors_names)
        adj_state_tensors, adj_control_tensors = adj_tensors[:num_state], adj_tensors[num_state:]

        if synchronize:
            # ensure Torch operations complete before running Warp
            wp.synchronize_device()

        if use_graph_capture:
            # checkpointing method
            assign_tensors(state_in, state_in_bwd, state_tensors_names, state_tensors)
            assign_tensors(control, control_bwd, control_tensors_names, control_tensors)
            wp.capture_launch(integrator.update_graph)

            assign_adjoints(state_out_bwd, state_tensors_names, adj_state_tensors)
            assign_adjoints(control_bwd, control_tensors_names, adj_control_tensors)
            wp.capture_launch(integrator.bwd_update_graph)
            assert len(tape.gradients) > 0
        else:
            assign_adjoints(state_out, state_tensors_names, adj_state_tensors)
            assign_adjoints(control, control_tensors_names, adj_control_tensors)
            tape.backward()

        if use_graph_capture:
            state_in, state_out = state_in_bwd, state_out_bwd
            control = control_bwd

        if synchronize:
            # ensure Warp operations complete before returning data to Torch
            wp.synchronize_device()

        try:
            adj_inputs = []
            for name in state_tensors_names:
                grad = tape.gradients[getattr(state_in, name)]
                # adj_tensor = wp.to_torch(wp.clone(grad))
                adj_tensor = wp.to_torch(grad).clone()

                if rescale_grad is not None:
                    adj_tensor /= rescale_grad
                if clip_grad is not None:
                    adj_tensor = torch.nan_to_num(adj_tensor, nan=0.0, posinf=-clip_grad, neginf=clip_grad)
                    adj_tensor = torch.clamp(adj_tensor, -clip_grad, clip_grad)

                # print(name, adj_tensor.norm(), adj_tensor)
                adj_inputs.append(adj_tensor)
            for name in control_tensors_names:
                grad = tape.gradients[getattr(control, name)]
                # adj_tensor = wp.to_torch(wp.clone(grad))
                adj_tensor = wp.to_torch(grad).clone()

                if rescale_grad is not None:
                    adj_tensor /= rescale_grad
                if clip_grad is not None:
                    adj_tensor = torch.nan_to_num(adj_tensor, nan=0.0, posinf=-clip_grad, neginf=clip_grad)
                    adj_tensor = torch.clamp(adj_tensor, -clip_grad, clip_grad)

                # print(name, adj_tensor.norm(), adj_tensor)
                adj_inputs.append(adj_tensor)
        except KeyError as e:
            print(f"Missing gradient for {name}")
            raise e

        # zero gradients
        tape.zero()

        if zero_nans:
            adj_inputs = [torch.nan_to_num_(adj_input, 0.0, 0.0, 0.0) for adj_input in adj_inputs]

        # return adjoint w.r.t inputs
        # None for each arg of forward() that is not ctx or *tensors
        return (
            None,  # update_params,
            None,  # sim_params,
            None,  # states,
            None,  # control,
            None,  # states_bwd,
            None,  # control_bwd,
            None,  # state_tensors_names,
            None,  # control_tensors_names,
        ) + tuple(adj_inputs)
