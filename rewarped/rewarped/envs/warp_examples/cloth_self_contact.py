import os
import math

import numpy as np
import torch
from pxr import Usd, UsdGeom

import warp as wp
import warp.examples
import warp.sim
from warp.sim.model import PARTICLE_FLAG_ACTIVE

from ...environment import IntegratorType, run_env
from ...warp_env import WarpEnv


@wp.kernel
def initialize_rotation(
    # input
    vertex_indices_to_rot: wp.array(dtype=wp.int32),
    pos: wp.array(dtype=wp.vec3),
    rot_centers: wp.array(dtype=wp.vec3),
    rot_axes: wp.array(dtype=wp.vec3),
    t: wp.array(dtype=float),
    # output
    roots: wp.array(dtype=wp.vec3),
    roots_to_ps: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    v_index = vertex_indices_to_rot[wp.tid()]

    p = pos[v_index]
    rot_center = rot_centers[tid]
    rot_axis = rot_axes[tid]
    op = p - rot_center

    root = wp.dot(op, rot_axis) * rot_axis

    root_to_p = p - root

    roots[tid] = root
    roots_to_ps[tid] = root_to_p

    if tid == 0:
        t[0] = 0.0


@wp.kernel
def apply_rotation(
    # input
    vertex_indices_to_rot: wp.array(dtype=wp.int32),
    rot_axes: wp.array(dtype=wp.vec3),
    roots: wp.array(dtype=wp.vec3),
    roots_to_ps: wp.array(dtype=wp.vec3),
    t: wp.array(dtype=float),
    angular_velocity: float,
    dt: float,
    end_time: float,
    # output
    pos_0: wp.array(dtype=wp.vec3),
    pos_1: wp.array(dtype=wp.vec3),
):
    cur_t = t[0]
    if cur_t > end_time:
        return

    tid = wp.tid()
    v_index = vertex_indices_to_rot[wp.tid()]

    rot_axis = rot_axes[tid]

    ux = rot_axis[0]
    uy = rot_axis[1]
    uz = rot_axis[2]

    theta = cur_t * angular_velocity

    R = wp.mat33(
        wp.cos(theta) + ux * ux * (1.0 - wp.cos(theta)),
        ux * uy * (1.0 - wp.cos(theta)) - uz * wp.sin(theta),
        ux * uz * (1.0 - wp.cos(theta)) + uy * wp.sin(theta),
        uy * ux * (1.0 - wp.cos(theta)) + uz * wp.sin(theta),
        wp.cos(theta) + uy * uy * (1.0 - wp.cos(theta)),
        uy * uz * (1.0 - wp.cos(theta)) - ux * wp.sin(theta),
        uz * ux * (1.0 - wp.cos(theta)) - uy * wp.sin(theta),
        uz * uy * (1.0 - wp.cos(theta)) + ux * wp.sin(theta),
        wp.cos(theta) + uz * uz * (1.0 - wp.cos(theta)),
    )

    root = roots[tid]
    root_to_p = roots_to_ps[tid]
    root_to_p_rot = R * root_to_p
    p_rot = root + root_to_p_rot

    pos_0[v_index] = p_rot
    pos_1[v_index] = p_rot

    if tid == 0:
        t[0] = cur_t + dt


class ClothSelfContact(WarpEnv):
    sim_name = "ClothSelfContact" + "WarpExamples"
    env_offset = (5.0, 0.0, 0.0)

    integrator_type = IntegratorType.VBD
    sim_substeps_vbd = 10
    vbd_settings = dict(
        iterations=4,
        handle_self_contact=True,
    )

    eval_fk = True
    eval_ik = False

    frame_dt = 1.0 / 60.0
    up_axis = "Y"
    ground_plane = False

    state_tensors_names = ("particle_q", "particle_qd")

    def __init__(self, num_envs=1, episode_length=300, early_termination=False, **kwargs):
        num_obs = 0
        num_act = 0
        super().__init__(num_envs, num_obs, num_act, episode_length, early_termination, **kwargs)

        # TODO: support parallel envs in apply_rotation
        if self.num_envs != 1:
            raise NotImplementedError

        self.action_scale = 1.0
        self.rot_angular_velocity = math.pi / 3
        self.rot_end_time = 10

    def create_env(self, builder):
        self.create_cloth(builder)

    def create_cloth(self, builder):
        usd_stage = Usd.Stage.Open(os.path.join(warp.examples.get_asset_directory(), "square_cloth.usd"))
        usd_geom = UsdGeom.Mesh(usd_stage.GetPrimAtPath("/root/cloth/cloth"))

        mesh_points = np.array(usd_geom.GetPointsAttr().Get())
        mesh_indices = np.array(usd_geom.GetFaceVertexIndicesAttr().Get())

        input_scale_factor = 1.0
        renderer_scale_factor = 0.01

        vertices = [wp.vec3(v) * input_scale_factor for v in mesh_points]
        faces = mesh_indices.reshape(-1, 3)

        builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vertices=vertices,
            indices=mesh_indices,
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=0.02,
            tri_ke=1.0e5,
            tri_ka=1.0e5,
            tri_kd=2.0e-6,
            edge_ke=10,
        )
        builder.color()

    def create_model(self):
        model = super().create_model()
        model.soft_contact_ke = 1.0e5
        model.soft_contact_kd = 1.0e-6
        model.soft_contact_mu = 0.2

        # set up contact query and contact detection distances
        model.soft_contact_radius = 0.2
        model.soft_contact_margin = 0.35

        return model

    def init_sim(self):
        super().init_sim()
        # self.print_model_info()

        # Create the CUDA graph. We first manually load the necessary
        # modules to avoid the capture to load all the modules that are
        # registered and possibly not relevant.
        wp.load_module(device=self.device)
        wp.set_module_options({"block_dim": 256}, warp.sim.integrator_vbd)
        wp.load_module(module=warp.sim, device=self.device, recursive=True)

        with torch.no_grad():
            cloth_size = 50
            left_side = [cloth_size - 1 + i * cloth_size for i in range(cloth_size)]
            right_side = [i * cloth_size for i in range(cloth_size)]
            rot_point_indices = left_side + right_side

            if len(rot_point_indices):
                flags = self.model.particle_flags.numpy()
                for fixed_vertex_id in rot_point_indices:
                    flags[fixed_vertex_id] = wp.uint32(int(flags[fixed_vertex_id]) & ~int(PARTICLE_FLAG_ACTIVE))

                self.model.particle_flags = wp.array(flags)

            rot_axes = [[1, 0, 0]] * len(right_side) + [[-1, 0, 0]] * len(left_side)

            self.rot_point_indices = wp.array(rot_point_indices, dtype=int)
            self.t = wp.zeros((1,), dtype=float)
            self.rot_centers = wp.zeros(len(rot_point_indices), dtype=wp.vec3)
            self.rot_axes = wp.array(rot_axes, dtype=wp.vec3)

            self.roots = wp.zeros_like(self.rot_centers)
            self.roots_to_ps = wp.zeros_like(self.rot_centers)

            wp.launch(
                kernel=initialize_rotation,
                dim=self.rot_point_indices.shape[0],
                inputs=[
                    self.rot_point_indices,
                    self.state_0.particle_q,
                    self.rot_centers,
                    self.rot_axes,
                    self.t,
                ],
                outputs=[
                    self.roots,
                    self.roots_to_ps,
                ],
            )

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)

    @torch.no_grad()
    def randomize_init(self, env_ids):
        pass

    def pre_physics_step(self, actions):
        actions = actions.view(self.num_envs, -1)
        actions = torch.clip(actions, -1.0, 1.0)
        self.actions = actions
        acts = self.action_scale * actions

    def do_physics_step(self):
        # TODO: graph capture
        for i in range(self.sim_substeps):
            wp.launch(
                kernel=apply_rotation,
                dim=self.rot_point_indices.shape[0],
                inputs=[
                    self.rot_point_indices,
                    self.rot_axes,
                    self.roots,
                    self.roots_to_ps,
                    self.t,
                    self.rot_angular_velocity,
                    self.sim_dt,
                    self.rot_end_time,
                ],
                outputs=[
                    self.state_0.particle_q,
                    self.state_1.particle_q,
                ],
            )

            self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def compute_observations(self):
        self.obs_buf = {}

    def compute_reward(self):
        rew = torch.zeros(self.num_envs, device=self.device)

        reset_buf, progress_buf = self.reset_buf, self.progress_buf
        max_episode_steps, early_termination = self.episode_length, self.early_termination
        truncated = progress_buf > max_episode_steps - 1
        reset = torch.where(truncated, torch.ones_like(reset_buf), reset_buf)
        if early_termination:
            raise NotImplementedError
        else:
            terminated = torch.where(torch.zeros_like(reset), torch.ones_like(reset), reset)
        self.rew_buf, self.reset_buf, self.terminated_buf, self.truncated_buf = rew, reset, terminated, truncated


if __name__ == "__main__":
    run_env(ClothSelfContact)
