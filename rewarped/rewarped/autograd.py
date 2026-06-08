import torch

import warp as wp

from .warp_utils import sim_update


# for checkpointing method
def assign_tensors(x, x_out, names, tensors, view=False):
    # need to assign b/c state_0, state_1 cannot be swapped
    # if view=True, then x == x_out except for tensors given by names, so we can skip assigning some
    # TODO: Add fn to get wp.array attributes instead of vars(..)
    if not view:
        for name in vars(x):
            if name in names:
                continue
            attr = getattr(x, name)
            if isinstance(attr, wp.array):
                wp_array = getattr(x_out, name)
                wp_array.assign(attr)
    for name, tensor in zip(names, tensors, strict=True):
        # assert not torch.isnan(tensor).any(), print("NaN tensor", name)
        wp_array = getattr(x_out, name)
        wp_array.assign(wp.from_torch(tensor, dtype=wp_array.dtype))


def assign_adjoints(x, names, adj_tensors):
    # register outputs with tape
    for name, adj_tensor in zip(names, adj_tensors, strict=True):
        # assert not torch.isnan(adj_tensor).any(), print("NaN adj", name)
        wp_array = getattr(x, name)
        # wp_array.grad = wp.from_torch(adj_tensor, dtype=wp_array.dtype)
        wp_array.grad.assign(wp.from_torch(adj_tensor, dtype=wp_array.dtype))


class UpdateFunction(torch.autograd.Function):
    """Custom torch autograd fn for `warp.sim.integrator.simulate()`."""

    @staticmethod
    def forward(
        ctx,
        autograd_params,
        sim_params,
        model,
        states,
        control,
        model_bwd,
        states_bwd,
        control_bwd,
        model_tensors_names,
        state_tensors_names,
        control_tensors_names,
        *tensors,
    ):
        tape, use_graph_capture, synchronize = autograd_params
        integrator, sim_substeps, sim_dt, eval_kinematic_fk, eval_ik = sim_params
        state_in, states_mid, state_out = states
        state_in_bwd, states_mid_bwd, state_out_bwd = states_bwd

        M = len(model_tensors_names)
        S = len(state_tensors_names)
        C = len(control_tensors_names)
        i, j, k = M, M + S, M + S + C
        model_tensors = tensors[:i]
        state_tensors = tensors[i:j]
        control_tensors = tensors[j:k]

        if synchronize:
            # ensure Torch operations complete before running Warp
            wp.synchronize_device()

        if tape is None:
            tape = wp.Tape()
            autograd_params = (tape, *autograd_params[1:])

        # for name in vars(model):
        #     attr = getattr(model, name)
        #     if isinstance(attr, wp.array):
        #         # print(name)
        #         attr.requires_grad = True

        ctx.autograd_params = autograd_params
        ctx.sim_params = sim_params
        ctx.model = model
        ctx.states = states
        ctx.control = control
        ctx.model_bwd = model_bwd
        ctx.states_bwd = states_bwd
        ctx.control_bwd = control_bwd
        ctx.model_tensors_names = model_tensors_names
        ctx.state_tensors_names = state_tensors_names
        ctx.control_tensors_names = control_tensors_names
        ctx.model_tensors = model_tensors
        ctx.state_tensors = state_tensors
        ctx.control_tensors = control_tensors

        if use_graph_capture:
            if getattr(tape, "update_graph", None) is None:
                assert getattr(tape, "bwd_update_graph", None) is None

                device = wp.get_device()
                # make torch use the warp stream from the given device
                torch_stream = wp.stream_to_torch(device)

                # capture graph
                with wp.ScopedDevice(device), torch.cuda.stream(torch_stream):
                    wp.capture_begin(force_module_load=False)
                    try:
                        with tape:
                            sim_update(sim_params, model_bwd, states_bwd, control_bwd)
                    finally:
                        tape.update_graph = wp.capture_end()

                    wp.capture_begin(force_module_load=False)
                    try:
                        tape.backward()
                    finally:
                        tape.bwd_update_graph = wp.capture_end()

            assign_tensors(model, model_bwd, model_tensors_names, model_tensors, view=True)
            assign_tensors(state_in, state_in_bwd, state_tensors_names, state_tensors)
            assign_tensors(control, control_bwd, control_tensors_names, control_tensors)
            wp.capture_launch(tape.update_graph)
            assign_tensors(state_out_bwd, state_out, [], [])  # write to state_out
        else:
            with tape:
                sim_update(sim_params, model, states, control)

        if synchronize:
            # ensure Warp operations complete before returning data to Torch
            wp.synchronize_device()

        # TODO: not clong right now, since these should be static?
        outputs = []
        for name in model_tensors_names:
            out_tensor = wp.to_torch(getattr(model, name))
            # assert not torch.isnan(out_tensor).any(), print("NaN fwd", name)
            # if use_graph_capture:
            #     out_tensor = out_tensor.clone()
            outputs.append(out_tensor)

        for data, tensors_names in zip(
            (state_out, control),
            (state_tensors_names, control_tensors_names),
            strict=True,
        ):
            for name in tensors_names:
                out_tensor = wp.to_torch(getattr(data, name))
                # assert not torch.isnan(out_tensor).any(), print("NaN fwd", name)
                if use_graph_capture:
                    out_tensor = out_tensor.clone()
                outputs.append(out_tensor)

        return tuple(outputs)

    @staticmethod
    def backward(ctx, *adj_tensors):
        autograd_params = ctx.autograd_params
        sim_params = ctx.sim_params
        model = ctx.model
        states = ctx.states
        control = ctx.control
        model_bwd = ctx.model_bwd
        states_bwd = ctx.states_bwd
        control_bwd = ctx.control_bwd
        model_tensors_names = ctx.model_tensors_names
        state_tensors_names = ctx.state_tensors_names
        control_tensors_names = ctx.control_tensors_names
        model_tensors = ctx.model_tensors
        state_tensors = ctx.state_tensors
        control_tensors = ctx.control_tensors

        tape, use_graph_capture, synchronize = autograd_params
        integrator, sim_substeps, sim_dt, eval_kinematic_fk, eval_ik = sim_params
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

        M = len(model_tensors_names)
        S = len(state_tensors_names)
        C = len(control_tensors_names)
        i, j, k = M, M + S, M + S + C
        adj_model_tensors = adj_tensors[:i]
        adj_state_tensors = adj_tensors[i:j]
        adj_control_tensors = adj_tensors[j:k]

        if synchronize:
            # ensure Torch operations complete before running Warp
            wp.synchronize_device()

        if use_graph_capture:
            # checkpointing method
            assign_tensors(model, model_bwd, model_tensors_names, model_tensors, view=True)
            assign_tensors(state_in, state_in_bwd, state_tensors_names, state_tensors)
            assign_tensors(control, control_bwd, control_tensors_names, control_tensors)
            wp.capture_launch(tape.update_graph)

            assign_adjoints(model_bwd, model_tensors_names, adj_model_tensors)
            assign_adjoints(state_out_bwd, state_tensors_names, adj_state_tensors)
            assign_adjoints(control_bwd, control_tensors_names, adj_control_tensors)
            wp.capture_launch(tape.bwd_update_graph)
            assert len(tape.gradients) > 0
        else:
            assign_adjoints(model, model_tensors_names, adj_model_tensors)
            assign_adjoints(state_out, state_tensors_names, adj_state_tensors)
            assign_adjoints(control, control_tensors_names, adj_control_tensors)
            tape.backward()

        if use_graph_capture:
            model = model_bwd
            state_in, state_out = state_in_bwd, state_out_bwd
            control = control_bwd

        if synchronize:
            # ensure Warp operations complete before returning data to Torch
            wp.synchronize_device()

        try:
            adj_inputs = []
            for data, tensors_names in zip(
                (model, state_in, control),
                (model_tensors_names, state_tensors_names, control_tensors_names),
                strict=True,
            ):
                for name in tensors_names:
                    grad = tape.gradients[getattr(data, name)]
                    # adj_tensor = wp.to_torch(wp.clone(grad))
                    adj_tensor = wp.to_torch(grad).clone()

                    if rescale_grad is not None:
                        adj_tensor /= rescale_grad
                    if clip_grad is not None:
                        adj_tensor = torch.nan_to_num(adj_tensor, nan=0.0, neginf=-clip_grad, posinf=clip_grad)
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
            None,  # autograd_params,
            None,  # sim_params,
            None,  # model,
            None,  # states,
            None,  # control,
            None,  # model_bwd,
            None,  # states_bwd,
            None,  # control_bwd,
            None,  # model_tensors_names,
            None,  # state_tensors_names,
            None,  # control_tensors_names,
        ) + tuple(adj_inputs)


class WarpKernelsFunction(torch.autograd.Function):
    """Custom torch autograd fn for arbitrary warp kernels.
    Assumes only one fn call, unlike UpdateFunction `sim_update()` which loops over `Integrator.simulate()`.
    """

    @staticmethod
    def forward(
        ctx,
        autograd_params,
        fn,
        fn_kwargs,
        x,
        y,
        x_tensors_names,
        y_tensors_names,
        *tensors,
    ):
        tape, use_graph_capture, graph_params = autograd_params
        x_bwd, y_bwd = graph_params

        num_x = len(x_tensors_names)
        x_tensors = tensors[:num_x]

        if tape is None:
            tape = wp.Tape()
            autograd_params = (tape, *autograd_params[1:])

        ctx.autograd_params = autograd_params
        ctx.fn = fn
        ctx.fn_kwargs = fn_kwargs
        ctx.x = x
        ctx.y = y
        ctx.x_tensors_names = x_tensors_names
        ctx.y_tensors_names = y_tensors_names
        ctx.x_tensors = x_tensors

        if use_graph_capture:
            if getattr(tape, "update_graph", None) is None:
                assert getattr(tape, "bwd_update_graph", None) is None

                device = wp.get_device()
                # make torch use the warp stream from the given device
                torch_stream = wp.stream_to_torch(device)

                # capture graph
                with wp.ScopedDevice(device), torch.cuda.stream(torch_stream):
                    wp.capture_begin(force_module_load=False)
                    try:
                        with tape:
                            fn(x_bwd, y_bwd, **fn_kwargs)
                    finally:
                        tape.update_graph = wp.capture_end()

                    wp.capture_begin(force_module_load=False)
                    try:
                        tape.backward()
                    finally:
                        tape.bwd_update_graph = wp.capture_end()

            assign_tensors(x, x_bwd, x_tensors_names, x_tensors)
            wp.capture_launch(tape.update_graph)
            assign_tensors(y_bwd, y, [], [])  # write to state_out
        else:
            with tape:
                fn(x, y, **fn_kwargs)

        outputs = []
        for name in y_tensors_names:
            out_tensor = wp.to_torch(getattr(y, name))
            # assert not torch.isnan(out_tensor).any(), print("NaN fwd", name)
            if use_graph_capture:
                out_tensor = out_tensor.clone()
            outputs.append(out_tensor)
        return tuple(outputs)

    @staticmethod
    def backward(ctx, *adj_tensors):
        autograd_params = ctx.autograd_params
        fn = ctx.fn
        fn_kwargs = ctx.fn_kwargs
        x = ctx.x
        y = ctx.y
        x_tensors_names = ctx.x_tensors_names
        y_tensors_names = ctx.y_tensors_names
        x_tensors = ctx.x_tensors

        tape, use_graph_capture, graph_params = autograd_params
        x_bwd, y_bwd = graph_params

        # ensure grads are contiguous in memory
        adj_tensors = [adj_tensor.contiguous() for adj_tensor in adj_tensors]

        if use_graph_capture:
            # checkpointing method
            assign_tensors(x, x_bwd, x_tensors_names, x_tensors)
            wp.capture_launch(tape.update_graph)

            assign_adjoints(y_bwd, y_tensors_names, adj_tensors)
            wp.capture_launch(tape.bwd_update_graph)
            assert len(tape.gradients) > 0
        else:
            assign_adjoints(y, y_tensors_names, adj_tensors)
            tape.backward()

        if use_graph_capture:
            x, y = x_bwd, y_bwd

        adj_inputs = []
        try:
            for name in x_tensors_names:
                grad = tape.gradients[getattr(x, name)]
                # adj_tensor = wp.to_torch(wp.clone(grad))
                adj_tensor = wp.to_torch(grad).clone()

                # print(name, adj_tensor.norm(), adj_tensor)
                adj_inputs.append(adj_tensor)

            for name in y_tensors_names:
                grad = tape.gradients[getattr(y, name)]
                # adj_tensor = wp.to_torch(wp.clone(grad))
                adj_tensor = wp.to_torch(grad).clone()

                # print(name, adj_tensor.norm(), adj_tensor)
                adj_inputs.append(adj_tensor)

        except KeyError as e:
            print(f"Missing gradient for {name}")
            raise e

        # zero gradients
        tape.zero()

        # return adjoint w.r.t inputs
        # None for each arg of forward() that is not ctx or *tensors
        return (
            None,  # autograd_params,
            None,  # fn,
            None,  # fn_kwargs,
            None,  # x,
            None,  # y,
            None,  # x_tensors_names,
            None,  # y_tensors_names,
        ) + tuple(adj_inputs)
