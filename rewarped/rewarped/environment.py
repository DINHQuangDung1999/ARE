# from https://github.com/NVIDIA/warp/blob/release-0.13/examples/env/environment.py

# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import os
from enum import Enum
from typing import Tuple

import numpy as np
import torch

import warp as wp
import warp.sim
import warp.sim.render

from .warp_utils import sim_update_inplace


class RenderMode(Enum):
    NONE = "none"
    OPENGL = "opengl"
    USD = "usd"

    def __str__(self):
        return self.value


class IntegratorType(Enum):
    EULER = "euler"
    FEATHERSTONE = "featherstone"
    XPBD = "xpbd"
    VBD = "vbd"
    MPM = "mpm"

    def __str__(self):
        return self.value


def compute_env_offsets(num_envs, env_offset=(5.0, 0.0, 5.0), env_offset_correction=True, up_axis="Y"):
    # compute positional offsets per environment
    env_offset = np.array(env_offset)
    nonzeros = np.nonzero(env_offset)[0]
    num_dim = nonzeros.shape[0]
    if num_dim > 0:
        side_length = int(np.ceil(num_envs ** (1.0 / num_dim)))
        env_offsets = []
    else:
        env_offsets = np.zeros((num_envs, 3))
    if num_dim == 1:
        for i in range(num_envs):
            env_offsets.append(i * env_offset)
    elif num_dim == 2:
        for i in range(num_envs):
            d0 = i // side_length
            d1 = i % side_length
            offset = np.zeros(3)
            offset[nonzeros[0]] = d0 * env_offset[nonzeros[0]]
            offset[nonzeros[1]] = d1 * env_offset[nonzeros[1]]
            env_offsets.append(offset)
    elif num_dim == 3:
        for i in range(num_envs):
            d0 = i // (side_length * side_length)
            d1 = (i // side_length) % side_length
            d2 = i % side_length
            offset = np.zeros(3)
            offset[0] = d0 * env_offset[0]
            offset[1] = d1 * env_offset[1]
            offset[2] = d2 * env_offset[2]
            env_offsets.append(offset)
    env_offsets = np.array(env_offsets)
    if not env_offset_correction:
        return env_offsets
    min_offsets = np.min(env_offsets, axis=0)
    correction = min_offsets + (np.max(env_offsets, axis=0) - min_offsets) / 2.0
    if isinstance(up_axis, str):
        up_axis = "XYZ".index(up_axis.upper())
    correction[up_axis] = 0.0  # ensure the envs are not shifted below the ground plane
    env_offsets -= correction
    return env_offsets


def compute_up_vector(up_axis):
    if isinstance(up_axis, str):
        up_axis = "XYZ".index(up_axis.upper())
    if up_axis == 0:
        return (1.0, 0.0, 0.0)
    elif up_axis == 1:
        return (0.0, 1.0, 0.0)
    elif up_axis == 2:
        return (0.0, 0.0, 1.0)
    else:
        raise ValueError(up_axis)


class Environment:
    sim_name: str = "Environment"

    frame_dt = 1.0 / 60.0
    episode_duration = 5.0  # seconds

    integrator_type: IntegratorType = IntegratorType.EULER

    sim_substeps_euler: int = 16
    sim_substeps_featherstone: int = 16
    sim_substeps_xpbd: int = 5
    sim_substeps_vbd: int = 5
    sim_substeps_mpm: int = 16

    euler_settings = dict(angular_damping=0.05, friction_smoothing=1.0)
    featherstone_settings = dict(
        angular_damping=0.05,
        update_mass_matrix_every=sim_substeps_featherstone,
        friction_smoothing=1.0,
    )
    xpbd_settings = dict(
        iterations=2,
        soft_body_relaxation=0.9,
        soft_contact_relaxation=0.9,
        joint_linear_relaxation=0.7,
        joint_angular_relaxation=0.4,
        rigid_contact_relaxation=0.8,
        rigid_contact_con_weighting=True,
        angular_damping=0.0,
        enable_restitution=False,
    )
    vbd_settings = dict(
        iterations=10,
        handle_self_contact=False,
        penetration_free_conservative_bound_relaxation=0.42,
        friction_epsilon=1e-2,
        body_particle_contact_buffer_pre_alloc=4,
        vertex_collision_buffer_pre_alloc=32,
        edge_collision_buffer_pre_alloc=64,
        triangle_collision_buffer_pre_alloc=32,
        edge_edge_parallel_epsilon=1e-5,
    )
    mpm_settings = dict()

    # stiffness and damping for joint attachment dynamics used by Euler
    joint_attach_ke: float = 32000.0
    joint_attach_kd: float = 50.0

    # friction coefficients for rigid body contacts used by XPBD
    rigid_contact_torsional_friction: float = 0.5
    rigid_contact_rolling_friction: float = 0.001

    # whether to apply model.joint_q, joint_qd to bodies before simulating
    eval_fk: bool = True
    # whether to directly set state.body_q based on state.joint_q
    eval_kinematic_fk: bool = False
    # whether to update state.joint_q, state.joint_qd
    eval_ik: bool = False

    render_mode: RenderMode = RenderMode.NONE
    opengl_render_settings = dict(scaling=1.0)
    usd_render_settings = dict(scaling=100.0)

    show_rigid_contact_points = False
    contact_points_radius = 1e-3
    show_joints = False
    # whether OpenGLRenderer should render each environment in a separate tile
    use_tiled_rendering = False
    # whether to play the simulation indefinitely when using the OpenGL renderer
    continuous_opengl_render: bool = True

    use_graph_capture: bool = wp.get_preferred_device().is_cuda
    synchronize: bool = True
    requires_grad: bool = False
    num_envs: int = 8

    ground_plane: bool = True
    ground_plane_settings = {}
    up_axis: str = "Y"
    gravity: float = -9.81
    env_offset: Tuple[float, float, float] = (1.0, 0.0, 1.0)
    env_offset_correction: bool = True  # if False, then env_offsets are only + (not centered at origin)

    # whether each environment should have its own collision group
    # to avoid collisions between environments
    separate_collision_group_per_env: bool = True

    asset_dir = os.path.join(os.path.dirname(__file__), "assets")
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    render_dir = None

    def __init__(self):
        self.profile = False
        self.plot_body_coords = False
        self.plot_joint_coords = False

    def init(self):
        if self.use_tiled_rendering and self.render_mode == RenderMode.OPENGL:
            # no environment offset when using tiled rendering
            self.env_offset = (0.0, 0.0, 0.0)

        self.builder = self.create_builder()
        assert self.builder.num_envs == self.num_envs
        self.model = self.create_model()

        # self.device = self.model.device
        if not self.model.device.is_cuda:
            self.use_graph_capture = False

        self.sim_substeps, self.integrator_settings, self.integrator = self.create_integrator(self.model)

        self.episode_frames = int(self.episode_duration / self.frame_dt)
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_steps = int(self.episode_duration / self.sim_dt)

        # set up current and next state to be used by the integrator
        self.state_0 = None
        self.state_1 = None
        self.control_0 = None

        self.renderer = self.create_renderer()

    def create_modelbuilder(self):
        builder = wp.sim.ModelBuilder(up_vector=compute_up_vector(self.up_axis), gravity=self.gravity)

        default_modelbuilder_settings = dict(
            # Default particle settings
            default_particle_radius=0.1,
            # particle_max_velocity=1e5,  # set in create_model instead
            # Default triangle soft mesh settings
            default_tri_ke=100.0,
            default_tri_ka=100.0,
            default_tri_kd=10.0,
            default_tri_drag=0.0,
            default_tri_lift=0.0,
            # Default distance constraint properties
            default_spring_ke=100.0,
            default_spring_kd=0.0,
            # Default edge bending properties
            default_edge_ke=100.0,
            default_edge_kd=0.0,
            # Default rigid shape contact material properties
            default_shape_ke=1.0e5,
            default_shape_kd=1000.0,
            default_shape_kf=1000.0,
            default_shape_ka=0.0,
            default_shape_mu=0.5,
            default_shape_restitution=0.0,
            default_shape_density=1000.0,
            default_shape_thickness=1e-5,
            # Default joint settings
            default_joint_limit_ke=100.0,
            default_joint_limit_kd=1.0,
            # Maximum number of soft contacts that can be registered
            soft_contact_max=64 * 1024,
            # maximum number of contact points to generate per mesh shape
            rigid_mesh_contact_max=0,  # 0 = unlimited
            # contacts to be generated within the given distance margin to be generated at
            # every simulation substep (can be 0 if only one PBD solver iteration is used)
            rigid_contact_margin=0.1,
            # number of rigid contact points to allocate in the model during self.finalize() per environment
            # if setting is None, the number of worst-case number of contacts will be calculated in self.finalize()
            num_rigid_contacts_per_env=None,
        )

        for k, v in default_modelbuilder_settings.items():
            setattr(builder, k, v)

        return builder

    def create_builder(self):
        builder = self.create_modelbuilder()
        env_builder = self.create_modelbuilder()
        self.create_env(env_builder)
        self.env_offsets = compute_env_offsets(self.num_envs, self.env_offset, self.env_offset_correction, self.up_axis)
        for i in range(self.num_envs):
            xform = wp.transform(self.env_offsets[i], wp.quat_identity())
            builder.add_builder(
                env_builder,
                xform,
                update_num_env_count=True,
                separate_collision_group=self.separate_collision_group_per_env,
            )
        return builder

    def create_env(self, builder):
        self.create_articulation(builder)

    def create_articulation(self, builder):
        raise NotImplementedError

    def create_model(self):
        self.builder.set_ground_plane(**self.ground_plane_settings)  # recreate builder._ground_params
        model = self.builder.finalize(device=self.device, requires_grad=self.requires_grad)
        model.ground = self.ground_plane

        if self.integrator_type == IntegratorType.EULER:
            model.joint_attach_ke = self.joint_attach_ke
            model.joint_attach_kd = self.joint_attach_kd
        if self.integrator_type == IntegratorType.XPBD:
            model.rigid_contact_torsional_friction = self.rigid_contact_torsional_friction
            model.rigid_contact_rolling_friction = self.rigid_contact_rolling_friction

        default_model_settings = dict(
            # Default particle settings
            particle_ke=1.0e3,
            particle_kd=1.0e2,
            particle_kf=1.0e2,
            particle_mu=0.5,
            particle_cohesion=0.0,
            particle_adhesion=0.0,
            particle_max_velocity=1e5,
            # Default soft contact settings
            soft_contact_radius=0.2,
            soft_contact_margin=0.2,
            soft_contact_ke=1.0e3,
            soft_contact_kd=10.0,
            soft_contact_kf=1.0e3,
            soft_contact_mu=0.5,
            soft_contact_restitution=0.0,
        )

        for k, v in default_model_settings.items():
            setattr(model, k, v)

        return model

    def create_integrator(self, model):
        if self.integrator_type == IntegratorType.EULER:
            sim_substeps = self.sim_substeps_euler
            integrator_settings = self.euler_settings
            integrator = wp.sim.SemiImplicitIntegrator(**integrator_settings)
        elif self.integrator_type == IntegratorType.FEATHERSTONE:
            sim_substeps = self.sim_substeps_featherstone
            integrator_settings = self.featherstone_settings
            integrator = wp.sim.FeatherstoneIntegrator(model, **integrator_settings)
        elif self.integrator_type == IntegratorType.XPBD:
            sim_substeps = self.sim_substeps_xpbd
            integrator_settings = self.xpbd_settings
            integrator = wp.sim.XPBDIntegrator(**integrator_settings)
        elif self.integrator_type == IntegratorType.VBD:
            sim_substeps = self.sim_substeps_vbd
            integrator_settings = self.vbd_settings
            integrator = wp.sim.VBDIntegrator(model, **integrator_settings)
        elif self.integrator_type == IntegratorType.MPM:
            sim_substeps = self.sim_substeps_mpm
            integrator_settings = self.mpm_settings
            integrator = wp.sim.MPMIntegrator(model, **integrator_settings)
        else:
            raise NotImplementedError(self.integrator_type)
        return sim_substeps, integrator_settings, integrator

    def create_renderer(self):
        if self.render_dir is not None:
            self.render_dir = os.path.realpath(self.render_dir)
        elif "WARP_RENDER_DIR" in os.environ:
            self.render_dir = os.path.realpath(os.environ.get("WARP_RENDER_DIR"))
        else:
            self.render_dir = os.path.join(wp.config.kernel_cache_dir, "../outputs")

        if self.render_mode == RenderMode.NONE:
            renderer = None
        elif self.render_mode == RenderMode.OPENGL:
            renderer = wp.sim.render.SimRendererOpenGL(
                self.model,
                self.sim_name,
                up_axis=self.up_axis,
                show_rigid_contact_points=self.show_rigid_contact_points,
                contact_points_radius=self.contact_points_radius,
                show_joints=self.show_joints,
                **self.opengl_render_settings,
            )
            if self.use_tiled_rendering and self.num_envs > 1:
                floor_id = self.model.shape_count - 1
                # all shapes except the floor
                instance_ids = np.arange(floor_id, dtype=np.int32).tolist()
                shapes_per_env = floor_id // self.num_envs
                additional_instances = []
                if self.activate_ground_plane:
                    additional_instances.append(floor_id)
                renderer.setup_tiled_rendering(
                    instances=[
                        instance_ids[i * shapes_per_env : (i + 1) * shapes_per_env] + additional_instances
                        for i in range(self.num_envs)
                    ]
                )
        elif self.render_mode == RenderMode.USD:
            filename = os.path.join(self.render_dir, self.sim_name + f"_{self.num_envs}.usd")
            renderer = wp.sim.render.SimRendererUsd(
                self.model,
                filename,
                up_axis=self.up_axis,
                show_rigid_contact_points=self.show_rigid_contact_points,
                **self.usd_render_settings,
            )
        else:
            raise NotImplementedError(self.render_mode)

        return renderer

    @property
    def state(self):
        # shortcut to current state
        return self.state_0

    @property
    def control(self):
        return self.control_0

    def update(self):
        sim_params = (self.integrator, self.sim_substeps, self.sim_dt, self.eval_kinematic_fk, self.eval_ik)
        self.state_0, self.state_1 = sim_update_inplace(
            sim_params,
            self.model,
            (self.state_0, self.state_1),
            self.control_0,
        )

    def render(self, state=None):
        if self.renderer is not None:
            with wp.ScopedTimer("render", False):
                self.render_time += self.frame_dt
                self.renderer.begin_frame(self.render_time)
                self.renderer.render(state or self.state_0)
                self.renderer.end_frame()

    def print_model_info(self):
        print("--- Model Info ---")
        print("num_envs", self.num_envs)
        print("frame_dt", self.frame_dt)
        print("sim_dt", self.sim_dt)
        print("sim_substeps", self.sim_substeps)

        if self.model.joint_count > 0:
            print("--- joint ---")
            print("joint_parent", len(self.model.joint_parent), self.model.joint_parent.numpy())
            print("joint_child", len(self.model.joint_child), self.model.joint_child.numpy())
            print("joint_name", len(self.model.joint_name), self.model.joint_name)
            if len(self.model.joint_q) > 0:
                print("joint_q", len(self.model.joint_q), self.model.joint_q.numpy())
                print("joint_qd", len(self.model.joint_qd), self.model.joint_qd.numpy())
                print("joint_armature", len(self.model.joint_armature), self.model.joint_armature.numpy())
            if len(self.model.joint_axis) > 0:
                # print("joint_axis", self.model.joint_axis.numpy())
                print("joint_axis_mode", len(self.model.joint_axis_mode), self.model.joint_axis_mode.numpy())
                print("joint_act", len(self.model.joint_act), self.model.joint_act.numpy())
                print("joint_target_ke", len(self.model.joint_target_ke), self.model.joint_target_ke.numpy())
                print("joint_target_kd", len(self.model.joint_target_kd), self.model.joint_target_kd.numpy())
                print("joint_limit_ke", len(self.model.joint_limit_ke), self.model.joint_limit_ke.numpy())
                print("joint_limit_kd", len(self.model.joint_limit_kd), self.model.joint_limit_kd.numpy())
                print("joint_limit_lower", len(self.model.joint_limit_lower), self.model.joint_limit_lower.numpy())
                print("joint_limit_upper", len(self.model.joint_limit_upper), self.model.joint_limit_upper.numpy())
            # print("joint_X_p", self.model.joint_X_p.numpy())
            # print("joint_X_c", self.model.joint_X_c.numpy())

        if self.model.body_count > 0:
            print("--- body ---")
            print("body_name", len(self.model.body_name), self.model.body_name)
            print("body_q", self.state_0.body_q.numpy())
            print("body_com", self.model.body_com.numpy())
            print("body_mass", self.model.body_mass.numpy())
            print("body_inertia", self.model.body_inertia.numpy())

            print("rigid_contact_max", self.model.rigid_contact_max)
            print("rigid_contact_max_limited", self.model.rigid_contact_max_limited)
            print("rigid_mesh_contact_max", self.model.rigid_mesh_contact_max)
            print("rigid_contact_thickness", self.model.rigid_contact_thickness)

        if self.model.shape_count > 0:
            print("--- shape ---")
            print("shape_geo_type", self.model.shape_geo.type.numpy())
            # print("shape_geo_scale", self.model.shape_geo.scale.numpy())
            print("shape_geo_is_solid", self.model.shape_geo.is_solid.numpy())
            print("shape_geo_thickness", self.model.shape_geo.thickness.numpy())

            print("shape_material_ke", self.model.shape_materials.ke.numpy())
            print("shape_material_kd", self.model.shape_materials.kd.numpy())
            print("shape_material_kf", self.model.shape_materials.kf.numpy())
            print("shape_material_ka", self.model.shape_materials.ka.numpy())
            print("shape_material_mu", self.model.shape_materials.mu.numpy())
            print("shape_material_restitution", self.model.shape_materials.restitution.numpy())

            print("shape_collision_radius", self.model.shape_collision_radius.numpy())
            # print("shape_transform", self.model.shape_transform.numpy())

        if self.model.particle_count > 0:
            print("--- particle ---")
            print("particle_count", self.model.particle_count)
            print("particle_max_radius", self.model.particle_max_radius)
            print("particle_max_velocity", self.model.particle_max_velocity)
            # print("particle_q", self.model.particle_q.numpy())
            # print("particle_qd", self.model.particle_qd.numpy())
            # print("particle_mass", self.model.particle_mass.numpy())
            # print("particle_radius", self.model.particle_radius.numpy())
            print("edge_count", self.model.edge_count)
            print("tri_count", self.model.tri_count)
            print("spring_count", self.model.spring_count)
        print()

    def parse_args(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument(
            "--integrator",
            help="Type of integrator",
            type=IntegratorType,
            choices=list(IntegratorType),
            default=self.integrator_type.value,
        )
        self.parser.add_argument(
            "--renderer",
            help="Type of renderer",
            type=RenderMode,
            choices=list(RenderMode),
            # default=self.render_mode.value,
            default=RenderMode.USD,
        )
        self.parser.add_argument(
            "--render_dir", help="Render output directory", type=str, default="outputs/", required=False
        )
        self.parser.add_argument(
            "--num_envs", help="Number of environments to simulate", type=int, default=self.num_envs
        )
        self.parser.add_argument("--profile", help="Enable profiling", type=bool, default=self.profile)

        args = self.parser.parse_args()
        self.integrator_type = args.integrator
        self.render_mode = args.renderer
        self.num_envs = args.num_envs
        self.profile = args.profile
        self.render_dir = args.render_dir

    def run(self):
        # ---------------
        # run simulation

        self.init()

        self.sim_time = 0.0
        self.render_time = 0.0
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        if self.eval_fk:
            wp.sim.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, None, self.state_0)

        if self.renderer is not None:
            self.render(self.state_0)

            if self.render_mode == RenderMode.OPENGL:
                self.renderer.paused = True

        profiler = {}

        if self.use_graph_capture:
            # create update graph
            wp.capture_begin()
            try:
                self.update()
            finally:
                graph = wp.capture_end()

        if self.plot_body_coords:
            q_history = []
            q_history.append(self.state_0.body_q.numpy().copy())
            qd_history = []
            qd_history.append(self.state_0.body_qd.numpy().copy())
            delta_history = []
            delta_history.append(self.state_0.body_deltas.numpy().copy())
            num_con_history = []
            num_con_history.append(self.model.rigid_contact_inv_weight.numpy().copy())
        if self.plot_joint_coords:
            joint_q_history = []
            joint_q = wp.zeros_like(self.model.joint_q)
            joint_qd = wp.zeros_like(self.model.joint_qd)

        # simulate
        with wp.ScopedTimer("simulate", detailed=False, print=False, active=True, dict=profiler):
            running = True
            while running:
                for f in range(self.episode_frames):
                    if self.use_graph_capture:
                        wp.capture_launch(graph)
                        self.sim_time += self.frame_dt
                    else:
                        self.update()
                        self.sim_time += self.frame_dt

                        if not self.profile:
                            if self.plot_body_coords:
                                q_history.append(self.state_0.body_q.numpy().copy())
                                qd_history.append(self.state_0.body_qd.numpy().copy())
                                delta_history.append(self.state_0.body_deltas.numpy().copy())
                                num_con_history.append(self.model.rigid_contact_inv_weight.numpy().copy())

                            if self.plot_joint_coords:
                                wp.sim.eval_ik(self.model, self.state_0, joint_q, joint_qd)
                                joint_q_history.append(joint_q.numpy().copy())

                    self.render()
                    if self.render_mode == RenderMode.OPENGL and self.renderer.has_exit:
                        running = False
                        break

                if not self.continuous_opengl_render or self.render_mode != RenderMode.OPENGL:
                    break

            if self.synchronize:
                # wp.synchronize()
                wp.synchronize_device()

        avg_time = np.array(profiler["simulate"]).mean() / self.episode_frames
        avg_steps_second = 1000.0 * float(self.num_envs) / avg_time

        print(f"envs: {self.num_envs} steps/second {avg_steps_second} avg_time {avg_time}")

        if self.renderer is not None:
            self.renderer.save()

        self.run_plots(
            q_history,
            qd_history,
            delta_history,
            num_con_history,
            joint_q_history,
        )

        return 1000.0 * float(self.num_envs) / avg_time

    def run_plots(self, q_history, qd_history, delta_history, num_con_history, joint_q_history):
        if self.plot_body_coords:
            import matplotlib.pyplot as plt

            q_history = np.array(q_history)
            qd_history = np.array(qd_history)
            delta_history = np.array(delta_history)
            num_con_history = np.array(num_con_history)

            # find bodies with non-zero mass
            body_indices = np.where(self.model.body_mass.numpy() > 0)[0]
            body_indices = body_indices[:5]  # limit number of bodies to plot

            fig, ax = plt.subplots(len(body_indices), 7, figsize=(10, 10), squeeze=False)
            fig.subplots_adjust(hspace=0.2, wspace=0.2)
            for i, j in enumerate(body_indices):
                ax[i, 0].set_title(f"Body {j} Position")
                ax[i, 0].grid()
                ax[i, 1].set_title(f"Body {j} Orientation")
                ax[i, 1].grid()
                ax[i, 2].set_title(f"Body {j} Linear Velocity")
                ax[i, 2].grid()
                ax[i, 3].set_title(f"Body {j} Angular Velocity")
                ax[i, 3].grid()
                ax[i, 4].set_title(f"Body {j} Linear Delta")
                ax[i, 4].grid()
                ax[i, 5].set_title(f"Body {j} Angular Delta")
                ax[i, 5].grid()
                ax[i, 6].set_title(f"Body {j} Num Contacts")
                ax[i, 6].grid()
                ax[i, 0].plot(q_history[:, j, :3])
                ax[i, 1].plot(q_history[:, j, 3:])
                ax[i, 2].plot(qd_history[:, j, 3:])
                ax[i, 3].plot(qd_history[:, j, :3])
                ax[i, 4].plot(delta_history[:, j, 3:])
                ax[i, 5].plot(delta_history[:, j, :3])
                ax[i, 6].plot(num_con_history[:, j])
                ax[i, 0].set_xlim(0, self.sim_steps)
                ax[i, 1].set_xlim(0, self.sim_steps)
                ax[i, 2].set_xlim(0, self.sim_steps)
                ax[i, 3].set_xlim(0, self.sim_steps)
                ax[i, 4].set_xlim(0, self.sim_steps)
                ax[i, 5].set_xlim(0, self.sim_steps)
                ax[i, 6].set_xlim(0, self.sim_steps)
                ax[i, 6].yaxis.get_major_locator().set_params(integer=True)
            plt.show()

        if self.plot_joint_coords:
            import matplotlib.pyplot as plt

            joint_q_history = np.array(joint_q_history)
            dof_q = joint_q_history.shape[1]
            ncols = int(np.ceil(np.sqrt(dof_q)))
            nrows = int(np.ceil(dof_q / float(ncols)))
            fig, axes = plt.subplots(
                ncols=ncols,
                nrows=nrows,
                constrained_layout=True,
                figsize=(ncols * 3.5, nrows * 3.5),
                squeeze=False,
                sharex=True,
            )

            joint_id = 0
            joint_type_names = {
                wp.sim.JOINT_BALL: "ball",
                wp.sim.JOINT_REVOLUTE: "hinge",
                wp.sim.JOINT_PRISMATIC: "slide",
                wp.sim.JOINT_UNIVERSAL: "universal",
                wp.sim.JOINT_COMPOUND: "compound",
                wp.sim.JOINT_FREE: "free",
                wp.sim.JOINT_FIXED: "fixed",
                wp.sim.JOINT_DISTANCE: "distance",
                wp.sim.JOINT_D6: "D6",
            }
            joint_lower = self.model.joint_limit_lower.numpy()
            joint_upper = self.model.joint_limit_upper.numpy()
            joint_type = self.model.joint_type.numpy()
            while joint_id < len(joint_type) - 1 and joint_type[joint_id] == wp.sim.JOINT_FIXED:
                # skip fixed joints
                joint_id += 1
            q_start = self.model.joint_q_start.numpy()
            qd_start = self.model.joint_qd_start.numpy()
            qd_i = qd_start[joint_id]
            for dim in range(ncols * nrows):
                ax = axes[dim // ncols, dim % ncols]
                if dim >= dof_q:
                    ax.axis("off")
                    continue
                ax.grid()
                ax.plot(joint_q_history[:, dim])
                if joint_type[joint_id] != wp.sim.JOINT_FREE:
                    lower = joint_lower[qd_i]
                    if abs(lower) < 2 * np.pi:
                        ax.axhline(lower, color="red")
                    upper = joint_upper[qd_i]
                    if abs(upper) < 2 * np.pi:
                        ax.axhline(upper, color="red")
                joint_name = joint_type_names[joint_type[joint_id]]
                ax.set_title(f"$\\mathbf{{q_{{{dim}}}}}$ ({self.model.joint_name[joint_id]} / {joint_name} {joint_id})")
                if joint_id < self.model.joint_count - 1 and q_start[joint_id + 1] == dim + 1:
                    joint_id += 1
                    qd_i = qd_start[joint_id]
                else:
                    qd_i += 1
            plt.tight_layout()
            plt.show()


def run_env(Demo, **kwargs):
    demo = Demo(device="cuda" if torch.cuda.is_available() else "cpu", **kwargs)
    demo.parse_args()
    if demo.profile:
        import matplotlib.pyplot as plt

        env_count = 2
        env_times = []
        env_size = []

        for i in range(15):
            demo.num_envs = env_count
            demo.initialized = False  # force re-initialization for WarpEnv
            steps_per_second = demo.run()

            env_size.append(env_count)
            env_times.append(steps_per_second)

            env_count *= 2

        # dump times
        for i in range(len(env_times)):
            print(f"envs: {env_size[i]} | steps/second: {env_times[i]}")

        # plot
        plt.figure(1)
        plt.plot(env_size, env_times)
        plt.xscale("log")
        plt.xlabel("Number of Envs")
        plt.yscale("log")
        plt.ylabel("Steps/Second")
        plt.show()
    else:
        try:
            steps_per_second = demo.run()
            print(f"envs: {demo.num_envs} | steps/second: {steps_per_second}")
        except KeyboardInterrupt:
            if demo.renderer is not None:
                demo.renderer.save()
            return -1
