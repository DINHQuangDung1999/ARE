# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import math
import os

import torch

import warp as wp

from ...environment import IntegratorType, run_env
from ...warp_env import WarpEnv
from .utils.torch_utils import quat_conjugate, quat_from_angle_axis, quat_mul, quat_rotate


class Go2VelTrack(WarpEnv):
    sim_name = "Go2VelTrack" + "DFlex"
    env_offset = (2.5, 0.0, 2.5)

    integrator_type = IntegratorType.FEATHERSTONE
    # Keep the 50 Hz control/update rate, but use finer internal substeps so
    # rigid contacts do not tunnel through the plane at reset or early rollout.
    sim_substeps_featherstone = 16
    featherstone_settings = dict(angular_damping=0.0, update_mass_matrix_every=sim_substeps_featherstone)

    eval_fk = True
    eval_ik = False if integrator_type == IntegratorType.FEATHERSTONE else True

    # Match Isaac Lab control timing: dt=0.005 with decimation=4 => 0.02 s/action.
    frame_dt = 1.0 / 50.0
    up_axis = "Y"
    ground_plane = True

    state_tensors_names = ("joint_q", "joint_qd")
    control_tensors_names = ("joint_act",)

    def __init__(self, num_envs=1, episode_length=1000, early_termination=True, **kwargs):
        num_obs = 45
        num_act = 12
        super().__init__(num_envs, num_obs, num_act, episode_length, early_termination, **kwargs)

        self.action_scale = 0.25
        self.joint_vel_obs_scaling = 1.0
        self.command_resample_interval = max(int(round(10.0 / self.frame_dt)), 1)
        self.command_threshold = 0.1

        self.base_height_target = 0.4
        self.base_contact_height = 0.18
        self.base_contact_gravity_threshold = 0.2

        self.reward_scales = {
            "track_lin_vel_xy_exp": 1.0,
            "track_ang_vel_z_exp": 0.5,
            "lin_vel_z_l2": -2.0,
            "ang_vel_xy_l2": -0.01,
            "action_rate_l2": -0.02,
            "dof_acc_l2": -2.5e-7,
            "base_height_l2": -1.0,
            "flat_orientation_l2": -0.2,
        }
        self.tracking_std = math.sqrt(0.25)

    def create_modelbuilder(self):
        builder = super().create_modelbuilder()
        builder.rigid_contact_margin = 0.05
        return builder

    def create_articulation(self, builder):
        wp.sim.parse_urdf(
            os.path.join(self.asset_dir, "dflex/go2/urdf/go2_description_simplified_locomotion.urdf"),
            builder,
            floating=True,
            density=1000.0,
            # stiffness=0.0,
            stiffness=25.0, # This works with action_scale = 10.0 and and mode target position
            damping=0.5,
            contact_ke=5.0e2,
            contact_kd=3.0e3,
            contact_kf=1.0e2,
            contact_mu=1.0,
            contact_restitution=0.0,
            limit_ke=1.0e3,
            limit_kd=1.0e1,
            armature=0.05,
            # armature_scale=5,
            enable_self_collisions=False,
            # up_axis="y",
        )


        builder.joint_axis_mode = [wp.sim.JOINT_MODE_TARGET_POSITION] * len(builder.joint_axis_mode)
        builder.joint_act[:] = [0.0] * len(builder.joint_act)
        builder.joint_q[7:] = [-0.1, 0.8, -1.5, 0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 0.8, -1.5]
        builder.joint_q[:7] = [0.0, 0.0, 0.0, *wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -math.pi * 0.5)]
        builder.joint_q[1] = 0.5

    def init_sim(self):
        super().init_sim()

        with torch.no_grad():
            self.joint_act = wp.to_torch(self.model.joint_act).view(self.num_envs, -1).clone()
            self.joint_act_indices = ...

            self.start_joint_q = self.state.joint_q.view(self.num_envs, -1).clone()
            self.start_joint_qd = self.state.joint_qd.view(self.num_envs, -1).clone()
            self.default_dof_pos = self.start_joint_q[:, 7:].clone()

            self.start_pos = self.start_joint_q[:, :3]
            self.start_rot = list(wp.quat_from_axis_angle((1.0, 0.0, 0.0), -math.pi * 0.5))
            self.start_rotation = torch.tensor(self.start_rot, device=self.device)

            self.x_unit_tensor = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self.device).repeat((self.num_envs, 1))
            self.y_unit_tensor = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=self.device).repeat((self.num_envs, 1))
            self.z_unit_tensor = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=self.device).repeat((self.num_envs, 1))

            self.up_vec = self.y_unit_tensor.clone()
            self.inv_start_rot = quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))

            self.commands = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
            self.prev_actions = torch.zeros((self.num_envs, self.num_act), dtype=torch.float32, device=self.device)
            self.prev_joint_vel = torch.zeros((self.num_envs, self.num_act), dtype=torch.float32, device=self.device)
            self.feet_air_time = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
            self.foot_contact = torch.zeros((self.num_envs, 4), dtype=torch.bool, device=self.device)

            self.resample_commands(torch.arange(self.num_envs, device=self.device))

    @torch.no_grad()
    def resample_commands(self, env_ids):
        if len(env_ids) == 0:
            return
        n = len(env_ids)
        self.commands[env_ids, 0] = 2.0 * torch.rand(n, device=self.device) - 1.0
        self.commands[env_ids, 1] = 2.0 * torch.rand(n, device=self.device) - 1.0
        self.commands[env_ids, 2] = 2.0 * torch.rand(n, device=self.device) - 1.0

    @torch.no_grad()
    def randomize_init(self, env_ids):
        joint_q = self.state.joint_q.view(self.num_envs, -1)
        joint_qd = self.state.joint_qd.view(self.num_envs, -1)

        n = len(env_ids)
        num_joint_q = 19
        num_joint_qd = 18

        joint_q[env_ids] = self.start_joint_q[env_ids].clone()
        joint_qd[env_ids] = self.start_joint_qd[env_ids].clone()
        joint_q[env_ids, 3:7] = self.start_rotation.clone()

        joint_q[env_ids, 0:3] += 0.05 * (torch.rand(size=(n, 3), device=self.device) - 0.5) * 2.0
        angle = (torch.rand(n, device=self.device) - 0.5) * math.pi / 12.0
        axis = torch.nn.functional.normalize(torch.rand((n, 3), device=self.device) - 0.5, dim=-1)
        joint_q[env_ids, 3:7] = quat_mul(joint_q[env_ids, 3:7], quat_from_angle_axis(angle, axis))
        joint_q[env_ids, 7:] += 0.2 * (torch.rand(size=(n, num_joint_q - 7), device=self.device) - 0.5) * 2.0
        joint_qd[env_ids, :] = 0.5 * (torch.rand(size=(n, num_joint_qd), device=self.device) - 0.5)

        ang_vel, lin_vel = joint_qd[env_ids, 0:3], joint_qd[env_ids, 3:6]
        joint_qd[env_ids, 3:6] = lin_vel + torch.cross(joint_q[env_ids, 0:3], ang_vel, dim=-1)

        self.prev_actions[env_ids] = 0.0
        self.prev_joint_vel[env_ids] = joint_qd[env_ids, 6:18]
        self.feet_air_time[env_ids] = 0.0
        self.foot_contact[env_ids] = False
        self.resample_commands(env_ids)

    def pre_physics_step(self, actions):
        actions = actions.view(self.num_envs, -1)
        actions = torch.clip(actions, -1.0, 1.0)
        self.actions = actions
        joint_targets = self.default_dof_pos + self.action_scale * actions

        if self.joint_act_indices is ...:
            self.control.assign("joint_act", joint_targets.flatten())
        else:
            joint_act = self.scatter_actions(self.joint_act, self.joint_act_indices, joint_targets)
            self.control.assign("joint_act", joint_act.flatten())

    def _compute_kinematics(self):
        joint_q = self.state.joint_q.clone().view(self.num_envs, -1)
        joint_qd = self.state.joint_qd.clone().view(self.num_envs, -1)

        root_pos = joint_q[:, 0:3] - self.env_offsets
        root_quat = joint_q[:, 3:7]
        ang_vel_w = joint_qd[:, 0:3]
        lin_vel_w = joint_qd[:, 3:6] - torch.cross(joint_q[:, 0:3], ang_vel_w, dim=-1)
        joint_pos = joint_q[:, 7:]
        joint_vel = joint_qd[:, 6:18]

        root_quat_body = quat_mul(root_quat, self.inv_start_rot)
        root_quat_body = root_quat_body / torch.linalg.norm(root_quat_body, dim=-1, keepdim=True).clamp_min(1e-6)
        root_quat_conj = quat_conjugate(root_quat_body)

        lin_vel_b = quat_rotate(root_quat_conj, lin_vel_w)
        ang_vel_b = quat_rotate(root_quat_conj, ang_vel_w)
        projected_gravity = quat_rotate(root_quat_conj, -self.up_vec)

        return joint_pos, joint_vel, root_pos, lin_vel_w, lin_vel_b, ang_vel_w, ang_vel_b, projected_gravity

    def compute_observations(self):
        if (self.progress_buf % self.command_resample_interval == 0).any():
            env_ids = torch.nonzero(self.progress_buf % self.command_resample_interval == 0, as_tuple=False).squeeze(-1)
            self.resample_commands(env_ids)

        joint_pos, joint_vel, _, _, _, _, ang_vel_b, projected_gravity = self._compute_kinematics()
        obs_buf = [
            joint_pos - self.default_dof_pos,
            self.joint_vel_obs_scaling * joint_vel,
            ang_vel_b,
            projected_gravity,
            self.commands,
            self.actions.clone(),
        ]
        self.obs_buf = torch.cat(obs_buf, dim=-1)

    def compute_reward(self):
        joint_pos, joint_vel, root_pos, lin_vel_w, lin_vel_b, _, ang_vel_b, projected_gravity = self._compute_kinematics()

        lin_vel_error = torch.sum((self.commands[:, :2] - lin_vel_b[:, :2]) ** 2, dim=-1)
        ang_vel_error = (self.commands[:, 2] - ang_vel_b[:, 2]) ** 2
        track_lin_vel_xy_exp = torch.exp(-lin_vel_error / (self.tracking_std ** 2))
        track_ang_vel_z_exp = torch.exp(-ang_vel_error / (self.tracking_std ** 2))

        lin_vel_z_l2 = lin_vel_w[:, 1] ** 2
        ang_vel_xy_l2 = torch.sum(ang_vel_b[:, :2] ** 2, dim=-1)
        action_rate_l2 = torch.sum((self.actions - self.prev_actions) ** 2, dim=-1)
        dof_acc_l2 = torch.sum(((joint_vel - self.prev_joint_vel) / self.frame_dt) ** 2, dim=-1)
        base_height_l2 = (root_pos[:, 1] - self.base_height_target) ** 2
        flat_orientation_l2 = torch.sum(projected_gravity[:, [0, 2]] ** 2, dim=-1)

        rew = (
            self.reward_scales["track_lin_vel_xy_exp"] * track_lin_vel_xy_exp
            + self.reward_scales["track_ang_vel_z_exp"] * track_ang_vel_z_exp
            + self.reward_scales["lin_vel_z_l2"] * lin_vel_z_l2
            + self.reward_scales["ang_vel_xy_l2"] * ang_vel_xy_l2
            + self.reward_scales["action_rate_l2"] * action_rate_l2
            + self.reward_scales["dof_acc_l2"] * dof_acc_l2
            + self.reward_scales["base_height_l2"] * base_height_l2
            + self.reward_scales["flat_orientation_l2"] * flat_orientation_l2
        )

        reset_buf, progress_buf = self.reset_buf, self.progress_buf
        truncated = progress_buf > self.episode_length - 1
        reset = torch.where(truncated, torch.ones_like(reset_buf), reset_buf)

        if self.early_termination:
            base_contact_like = (root_pos[:, 1] < self.base_contact_height) | (projected_gravity[:, 1] > -self.base_contact_gravity_threshold)
            terminated = base_contact_like
            reset = torch.where(terminated, torch.ones_like(reset), reset)
        else:
            terminated = torch.zeros_like(reset, dtype=torch.bool)

        self.prev_actions.copy_(self.actions)
        self.prev_joint_vel.copy_(joint_vel)
        self.rew_buf, self.reset_buf, self.terminated_buf, self.truncated_buf = rew, reset, terminated, truncated
        self.extras["reward_terms"] = {
            "track_lin_vel_xy_exp": track_lin_vel_xy_exp.mean(),
            "track_ang_vel_z_exp": track_ang_vel_z_exp.mean(),
            "lin_vel_z_l2": lin_vel_z_l2.mean(),
            "ang_vel_xy_l2": ang_vel_xy_l2.mean(),
            "action_rate_l2": action_rate_l2.mean(),
            "dof_acc_l2": dof_acc_l2.mean(),
            "base_height_l2": base_height_l2.mean(),
            "flat_orientation_l2": flat_orientation_l2.mean(),
        }


if __name__ == "__main__":
    run_env(Go2VelTrack, 
            episode_length=200, 
            early_termination=True, 
            render=True, 
            render_mode='usd')
