# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os

import hydra
import torch
from omegaconf import OmegaConf

import warp as wp

from ...environment import IntegratorType, run_env
from ...warp_env import WarpEnv
from .utils.torch_jit_utils import (
    compute_heading_and_up,
    compute_rot,
    quat_conjugate,
    tensor_clamp,
    torch_rand_float,
    unscale,
)


class Ant(WarpEnv):
    sim_name = "Ant" + "IsaacGymEnvs"
    env_offset = (0.0, 2.5, 0.0)

    integrator_type = IntegratorType.FEATHERSTONE
    sim_substeps_featherstone = 16
    featherstone_settings = dict(angular_damping=0.0, update_mass_matrix_every=sim_substeps_featherstone)

    eval_fk = True
    eval_ik = False if integrator_type == IntegratorType.FEATHERSTONE else True

    frame_dt = 1.0 / 60.0
    up_axis = "Z"
    ground_plane = True

    state_tensors_names = ("joint_q", "joint_qd", "body_f")
    control_tensors_names = ("joint_act",)

    def __init__(self, num_envs=8, episode_length=-1, early_termination=True, with_force=False, **kwargs):
        cfg = self.create_cfg_ig()
        cfg = self.set_params_from_ig_cfg(cfg)

        if episode_length == -1:
            episode_length = cfg["env"]["episodeLength"]

        num_obs = cfg["env"]["numObservations"]
        if not with_force:
            num_obs -= 24
        num_act = cfg["env"]["numActions"]
        super().__init__(num_envs, num_obs, num_act, episode_length, early_termination, **kwargs)

        self.with_force = with_force
        self.up_axis_idx = 2  # index of up axis: Y=1, Z=2
        self.joint_gears = 15.0
        self.action_scale = self.joint_gears * self.power_scale
        self.action_scale = self.action_scale

    def create_cfg_ig(self):
        cfg_path = os.path.join(os.path.dirname(__file__), "cfg/tasks")
        with hydra.initialize(version_base=None, config_path=os.path.relpath(cfg_path)):
            ig_cfg = OmegaConf.load(os.path.join(cfg_path, "Ant.yaml"))
        return ig_cfg

    def set_params_from_ig_cfg(self, cfg):
        self.randomization_params = cfg["task"]["randomization_params"]
        self.randomize = cfg["task"]["randomize"]
        self.dof_vel_scale = cfg["env"]["dofVelocityScale"]
        self.contact_force_scale = cfg["env"]["contactForceScale"]
        self.power_scale = cfg["env"]["powerScale"]
        self.heading_weight = cfg["env"]["headingWeight"]
        self.up_weight = cfg["env"]["upWeight"]
        self.actions_cost_scale = cfg["env"]["actionsCost"]
        self.energy_cost_scale = cfg["env"]["energyCost"]
        self.joints_at_limit_cost_scale = cfg["env"]["jointsAtLimitCost"]
        self.death_cost = cfg["env"]["deathCost"]
        self.termination_height = cfg["env"]["terminationHeight"]

        self.debug_viz = cfg["env"]["enableDebugVis"]
        self.plane_static_friction = cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = cfg["env"]["plane"]["restitution"]

        cfg["env"]["numObservations"] = 60
        cfg["env"]["numActions"] = 8
        return cfg

    def create_modelbuilder(self):
        builder = super().create_modelbuilder()
        builder.rigid_contact_margin = 0.02
        return builder

    def create_articulation(self, builder):
        asset_root = os.path.join(self.asset_dir, "isaacgymenvs")
        asset_file = "mjcf/nv_ant.xml"

        # if "asset" in cfg["env"]:
        #     asset_file = cfg["env"]["asset"].get("assetFileName", asset_file)

        asset_path = os.path.join(asset_root, asset_file)
        # asset_root = os.path.dirname(asset_path)
        # asset_file = os.path.basename(asset_path)

        p = [0.0, 0.0, 0.0]
        # p[self.up_axis_idx] = 0.44
        p[self.up_axis_idx] = 0.7
        q = wp.quat_identity()

        wp.sim.parse_mjcf(
            asset_path,
            builder,
            floating=True,
            density=1000.0,
            stiffness=0.0,
            damping=0.1,
            # stiffness=1.0,
            # damping=1.0,
            # contact_thickness=0.02,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            ignore_names=("floor",),
            up_axis="Y",
        )
        builder.joint_q[:3] = p
        builder.joint_q[3:7] = q
        builder.joint_qd[:6] = [0.0] * 6
        builder.joint_act[:] = [0.0] * len(builder.joint_act)
        # builder.joint_q[7:] = [0.0, 0.5236, 0.0, -0.5236, 0.0, -0.5236, 0.0, 0.5236]
        builder.joint_q[7:] = [0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0]
        builder.joint_axis_mode = [wp.sim.JOINT_MODE_FORCE] * len(builder.joint_axis_mode)

        self.extremity_names = [s for s in builder.body_name if "foot" in s]
        self.extremities_index = [builder.body_name.index(s) for s in self.extremity_names]

    def init_sim(self):
        super().init_sim()
        self.print_model_info()

        with torch.no_grad():
            self.joint_act = wp.to_torch(self.model.joint_act).view(self.num_envs, -1).clone()
            self.joint_act_indices = ...

            self.start_joint_q = self.state.joint_q.view(self.num_envs, -1).clone()
            self.start_joint_qd = self.state.joint_qd.view(self.num_envs, -1).clone()

            self.joint_limit_lower = wp.to_torch(self.model.joint_limit_lower).view(self.num_envs, -1).clone()
            self.joint_limit_upper = wp.to_torch(self.model.joint_limit_upper).view(self.num_envs, -1).clone()

            # --- From Ant._create_env()

            self.num_dof = self.start_joint_q.shape[1] - 7

            # --- From Ant.__init__()

            # initialize some data used later on
            self.up_vec = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat((self.num_envs, 1))
            self.heading_vec = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat((self.num_envs, 1))
            start_rotation = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)
            self.inv_start_rot = quat_conjugate(start_rotation).repeat((self.num_envs, 1))

            self.basis_vec0 = self.heading_vec.clone()
            self.basis_vec1 = self.up_vec.clone()

            self.targets = torch.tensor([1000.0, 0.0, 0.0], device=self.device).repeat((self.num_envs, 1))
            self.target_dirs = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat((self.num_envs, 1))

            dt = self.frame_dt
            self.potentials = torch.tensor([-1000.0 / dt], device=self.device).repeat(self.num_envs)
            self.prev_potentials = self.potentials.clone()

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)

        with torch.no_grad():
            initial_root_states = self.start_joint_q[:, :7]
            to_target = self.targets[env_ids] - (initial_root_states[env_ids, 0:3] - self.env_offsets[env_ids])
            to_target[:, 2] = 0.0
            dt = self.frame_dt
            self.prev_potentials[env_ids] = -torch.norm(to_target, p=2, dim=-1) / dt
            self.potentials[env_ids] = self.prev_potentials[env_ids].clone()

    @torch.no_grad()
    def randomize_init(self, env_ids):
        joint_q = self.state.joint_q.view(self.num_envs, -1)
        joint_qd = self.state.joint_qd.view(self.num_envs, -1)

        dof_pos = joint_q[:, 7:]
        dof_vel = joint_qd[:, 6:]

        positions = torch_rand_float(-0.2, 0.2, (len(env_ids), self.num_dof), device=self.device)
        velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)

        dof_limits_lower, dof_limits_upper = self.joint_limit_lower[env_ids], self.joint_limit_upper[env_ids]
        initial_dof_pos = self.start_joint_q[:, 7:]
        dof_pos[env_ids] = tensor_clamp(initial_dof_pos[env_ids] + positions, dof_limits_lower, dof_limits_upper)
        dof_vel[env_ids] = velocities

    def pre_physics_step(self, actions):
        actions = actions.view(self.num_envs, -1)
        actions = torch.clip(actions, -1.0, 1.0)
        self.actions = actions
        acts = self.action_scale * actions

        if self.joint_act_indices is ...:
            self.control.assign("joint_act", acts.flatten())
        else:
            joint_act = self.scatter_actions(self.joint_act, self.joint_act_indices, acts)
            self.control.assign("joint_act", joint_act.flatten())

    def compute_observations(self):
        joint_q = self.state.joint_q.clone().view(self.num_envs, -1)
        joint_qd = self.state.joint_qd.clone().view(self.num_envs, -1)
        body_f = self.state.body_f.clone().view(self.num_envs, -1, 6)
        vec_sensor_tensor = body_f[:, self.extremities_index, :]

        self.obs_buf, self.potentials, self.prev_potentials, self.up_vec, self.heading_vec = compute_ant_observations(
            self.env_offsets,
            joint_q,
            joint_qd,
            self.joint_limit_lower,
            self.joint_limit_upper,
            self.targets,
            self.potentials,
            self.inv_start_rot,
            self.dof_vel_scale,
            self.with_force,
            vec_sensor_tensor,
            self.actions,
            self.frame_dt,
            self.contact_force_scale,
            self.basis_vec0,
            self.basis_vec1,
            self.up_axis_idx,
        )

    def compute_reward(self):
        self.rew_buf, self.reset_buf, self.terminated_buf, self.truncated_buf = compute_ant_reward(
            self.obs_buf,
            self.reset_buf,
            self.progress_buf,
            self.actions,
            self.up_weight,
            self.heading_weight,
            self.potentials,
            self.prev_potentials,
            self.actions_cost_scale,
            self.energy_cost_scale,
            self.joints_at_limit_cost_scale,
            self.termination_height,
            self.death_cost,
            self.episode_length,
            self.early_termination,
        )


# @torch.jit.script
def compute_ant_observations(
    env_offsets,
    joint_q,
    joint_qd,
    joint_limit_lower,
    joint_limit_upper,
    targets,
    potentials,
    inv_start_rot,
    dof_vel_scale,
    with_force,
    sensor_force_torques,
    actions,
    dt,
    contact_force_scale,
    basis_vec0,
    basis_vec1,
    up_axis_idx,
):
    dof_pos = joint_q[:, 7:]
    dof_vel = joint_qd[:, 6:]
    dof_limits_lower, dof_limits_upper = joint_limit_lower, joint_limit_upper

    torso_position = joint_q[:, 0:3] - env_offsets
    torso_rotation = joint_q[:, 3:7]
    velocity = joint_qd[:, 3:6]
    ang_velocity = joint_qd[:, 0:3]

    to_target = targets - torso_position
    to_target[:, 2] = 0.0

    prev_potentials_new = potentials.clone()
    potentials = -torch.norm(to_target, p=2, dim=-1) / dt

    torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
        torso_rotation, inv_start_rot, to_target, basis_vec0, basis_vec1, 2
    )

    vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = compute_rot(
        torso_quat, velocity, ang_velocity, targets, torso_position
    )

    dof_pos_scaled = unscale(dof_pos, dof_limits_lower, dof_limits_upper)

    if with_force:
        force = [sensor_force_torques.view(-1, 24) * contact_force_scale]
    else:
        force = []

    # obs_buf shapes: 1, 3, 3, 1, 1, 1, 1, 1, num_dofs(8), num_dofs(8), 24, num_dofs(8)
    obs = (
        torso_position[:, up_axis_idx].view(-1, 1),
        vel_loc,
        angvel_loc,
        yaw.unsqueeze(-1),
        roll.unsqueeze(-1),
        angle_to_target.unsqueeze(-1),
        up_proj.unsqueeze(-1),
        heading_proj.unsqueeze(-1),
        dof_pos_scaled,
        dof_vel * dof_vel_scale,
        *force,
        actions,
    )
    obs = torch.cat(obs, dim=-1)

    return obs, potentials, prev_potentials_new, up_vec, heading_vec


# @torch.jit.script
def compute_ant_reward(
    obs_buf,
    reset_buf,
    progress_buf,
    actions,
    up_weight,
    heading_weight,
    potentials,
    prev_potentials,
    actions_cost_scale,
    energy_cost_scale,
    joints_at_limit_cost_scale,
    termination_height,
    death_cost,
    max_episode_length,
    early_termination,
):
    # reward from direction headed
    heading_weight_tensor = torch.ones_like(obs_buf[:, 11]) * heading_weight
    heading_reward = torch.where(obs_buf[:, 11] > 0.8, heading_weight_tensor, heading_weight * obs_buf[:, 11] / 0.8)

    # aligning up axis of ant and environment
    up_reward = torch.zeros_like(heading_reward)
    up_reward = torch.where(obs_buf[:, 10] > 0.93, up_reward + up_weight, up_reward)

    # energy penalty for movement
    actions_cost = torch.sum(actions**2, dim=-1)
    electricity_cost = torch.sum(torch.abs(actions * obs_buf[:, 20:28]), dim=-1)
    dof_at_limit_cost = torch.sum(obs_buf[:, 12:20] > 0.99, dim=-1)

    # reward for duration of staying alive
    alive_reward = torch.ones_like(potentials) * 0.5
    progress_reward = potentials - prev_potentials

    total_reward = (
        progress_reward
        + alive_reward
        + up_reward
        + heading_reward
        - actions_cost_scale * actions_cost
        - energy_cost_scale * electricity_cost
        - dof_at_limit_cost * joints_at_limit_cost_scale
    )

    # adjust reward for fallen agents
    total_reward = torch.where(
        obs_buf[:, 0] < termination_height, torch.ones_like(total_reward) * death_cost, total_reward
    )

    # reset agents
    truncated = progress_buf > max_episode_length - 1
    reset = torch.where(truncated, torch.ones_like(reset_buf), reset_buf)
    if early_termination:
        terminated = obs_buf[:, 0] < termination_height
        reset = torch.where(terminated, torch.ones_like(reset), reset)
    else:
        terminated = torch.where(torch.zeros_like(reset), torch.ones_like(reset), reset)

    return total_reward, reset, terminated, truncated


if __name__ == "__main__":
    run_env(Ant)
