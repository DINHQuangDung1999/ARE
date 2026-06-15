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
import numpy as np
import torch
from gym import spaces
from omegaconf import OmegaConf

import warp as wp

from ...environment import IntegratorType, run_env
from ...warp_env import WarpEnv
from .utils.torch_jit_utils import (
    get_euler_xyz,
    quat_conjugate,
    quat_from_angle_axis,
    quat_from_euler_xyz,
    quat_mul,
    scale,
    torch_rand_float,
    unscale,
)


class ShadowHand(WarpEnv):
    sim_name = "ShadowHand" + "IsaacGymEnvs"
    env_offset = (1.0, 1.0, 0.0)

    integrator_type = IntegratorType.FEATHERSTONE
    sim_substeps_featherstone = 32
    featherstone_settings = dict(angular_damping=0.01, update_mass_matrix_every=sim_substeps_featherstone)

    eval_fk = True
    eval_ik = False if integrator_type == IntegratorType.FEATHERSTONE else True

    frame_dt = 1.0 / 60.0
    # frame_dt = 1.0 / 120.0
    up_axis = "Z"
    ground_plane = False

    state_tensors_names = ("joint_q", "joint_qd")
    control_tensors_names = ("joint_act",)

    def __init__(self, num_envs=8, episode_length=-1, early_termination=True, with_joint_tau=False, **kwargs):
        cfg = self.create_cfg_ig()
        cfg = self.set_params_from_ig_cfg(cfg)

        self.control_freq_inv = cfg["env"].get("controlFrequencyInv", 1)
        if self.control_freq_inv > 1:
            self.frame_dt = self.frame_dt * self.control_freq_inv
            self.sim_substeps_featherstone = self.sim_substeps_featherstone * self.control_freq_inv
            self.featherstone_settings["update_mass_matrix_every"] = self.sim_substeps_featherstone

        if episode_length == -1:
            episode_length = cfg["env"]["episodeLength"]
        num_obs = cfg["env"]["numObservations"]
        if not with_joint_tau:
            num_obs -= 16
        num_state = cfg["env"]["numStates"]
        num_act = cfg["env"]["numActions"]

        super().__init__(num_envs, num_obs, num_act, episode_length, early_termination, **kwargs)

        self.ig_cfg = cfg
        self.use_pd_control = False
        self.use_torque_control = True
        self.use_quat = True
        self.with_joint_tau = with_joint_tau
        self.num_state = num_state
        self.up_axis_idx = 2  # index of up axis: Y=1, Z=2
        self.action_scale = 1.0

    @property
    def observation_space(self):
        d = {
            "obs": spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_obs,), dtype=np.float32),
        }
        if self.asymmetric_obs:
            d["state"] = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_state,), dtype=np.float32)
        d = spaces.Dict(d)
        return d

    def create_cfg_ig(self):
        cfg_path = os.path.join(os.path.dirname(__file__), "cfg/tasks")
        with hydra.initialize(version_base=None, config_path=os.path.relpath(cfg_path)):
            ig_cfg = OmegaConf.load(os.path.join(cfg_path, "ShadowHand.yaml"))
        return ig_cfg

    def set_params_from_ig_cfg(self, cfg):
        # self.aggregate_mode = cfg["env"]["aggregateMode"]

        self.dist_reward_scale = cfg["env"]["distRewardScale"]
        self.rot_reward_scale = cfg["env"]["rotRewardScale"]
        self.action_penalty_scale = cfg["env"]["actionPenaltyScale"]
        self.success_tolerance = cfg["env"]["successTolerance"]
        self.reach_goal_bonus = cfg["env"]["reachGoalBonus"]
        self.fall_dist = cfg["env"]["fallDistance"]
        self.fall_penalty = cfg["env"]["fallPenalty"]
        self.rot_eps = cfg["env"]["rotEps"]

        self.vel_obs_scale = 0.2  # scale factor of velocity based observations
        self.force_torque_obs_scale = 10.0  # scale factor of velocity based observations

        self.reset_position_noise = cfg["env"]["resetPositionNoise"]
        self.reset_rotation_noise = cfg["env"]["resetRotationNoise"]
        self.reset_dof_pos_noise = cfg["env"]["resetDofPosRandomInterval"]
        self.reset_dof_vel_noise = cfg["env"]["resetDofVelRandomInterval"]

        self.force_scale = cfg["env"].get("forceScale", 0.0)
        self.force_prob_range = cfg["env"].get("forceProbRange", [0.001, 0.1])
        self.force_decay = cfg["env"].get("forceDecay", 0.99)
        self.force_decay_interval = cfg["env"].get("forceDecayInterval", 0.08)

        self.hand_dof_speed_scale = cfg["env"]["dofSpeedScale"]
        self.use_relative_control = cfg["env"]["useRelativeControl"]
        self.act_moving_average = cfg["env"]["actionsMovingAverage"]

        self.debug_viz = cfg["env"]["enableDebugVis"]

        # self.max_episode_length = cfg["env"]["episodeLength"]
        self.reset_time = cfg["env"].get("resetTime", -1.0)
        self.print_success_stat = cfg["env"]["printNumSuccesses"]
        self.max_consecutive_successes = cfg["env"]["maxConsecutiveSuccesses"]
        self.av_factor = cfg["env"].get("averFactor", 0.1)

        self.object_type = cfg["env"]["objectType"]
        assert self.object_type in ["block", "egg", "pen"]

        self.ignore_z = self.object_type == "pen"

        self.asset_files_dict = {
            "block": "urdf/objects/cube_multicolor.urdf",
            "egg": "mjcf/open_ai_assets/hand/egg.xml",
            "pen": "mjcf/open_ai_assets/hand/pen.xml",
        }

        if "asset" in cfg["env"]:
            asset = cfg["env"]["asset"]
            self.asset_files_dict["block"] = asset.get("assetFileNameBlock", self.asset_files_dict["block"])
            self.asset_files_dict["egg"] = asset.get("assetFileNameEgg", self.asset_files_dict["egg"])
            self.asset_files_dict["pen"] = asset.get("assetFileNamePen", self.asset_files_dict["pen"])

        # can be "openai", "full_no_vel", "full", "full_state"
        self.obs_type = cfg["env"]["observationType"]
        obs_types = ("openai", "full_no_vel", "full", "full_state")
        if self.obs_type not in obs_types:
            raise Exception(f"Unknown type of observations!\nobservationType should be one of: {obs_types}")
        print("Obs type:", self.obs_type)

        self.num_obs_dict = {
            "openai": 42,
            "full_no_vel": 77,
            "full": 157,
            "full_state": 211,
        }

        # self.up_axis = "Z"

        self.fingertips = [
            "robot0:ffdistal",
            "robot0:mfdistal",
            "robot0:rfdistal",
            "robot0:lfdistal",
            "robot0:thdistal",
        ]
        self.num_fingertips = len(self.fingertips)

        self.use_vel_obs = False
        self.fingertip_obs = True
        self.asymmetric_obs = cfg["env"]["asymmetric_observations"]

        num_states = 0
        if self.asymmetric_obs:
            num_states = 211

        cfg["env"]["numObservations"] = self.num_obs_dict[self.obs_type]
        cfg["env"]["numStates"] = num_states
        cfg["env"]["numActions"] = 24

        return cfg

    def create_modelbuilder(self):
        builder = super().create_modelbuilder()
        # builder.default_shape_thickness = 0.002
        # builder.rigid_contact_margin = 0.001
        builder.rigid_contact_margin = 0.05
        return builder

    def create_env(self, builder):
        self.create_shadow_hand(builder)

        # load manipulated object and goal assets
        self.create_object(builder)

    def create_shadow_hand(self, builder):
        p = [0.0, 0.0, 0.0]
        p[self.up_axis_idx] = 0.5
        q = wp.quat_identity()
        q = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), np.pi)
        self.hand_start_pose = wp.transform(p, q)

        wp.sim.parse_mjcf(
            os.path.join(self.asset_dir, "shadow/right_hand.xml"),
            builder,
            xform=self.hand_start_pose,
            floating=False,
            density=1000.0,
            stiffness=0.0,
            damping=1.0,
            armature=0.1,
            # parse_meshes=False,
            # ignore_names=["C_forearm"],
            visual_classes=["D_Vizual"],
            collider_classes=["DC_Hand"],
            enable_self_collisions=True,
            ignore_inertial_definitions=False,
            reduce_capsule_height_by_radius=True,
            collapse_fixed_joints=True,
            force_show_colliders=True,
            hide_visuals=True,
            up_axis=self.up_axis,
        )

        # self.num_hand_bodies = builder.body_count
        # self.num_hand_shapes = builder.shape_count
        self.num_hand_dofs = builder.joint_count
        # print("Num dofs: ", self.num_hand_dofs)
        # self.num_hand_actuators = self.num_hand_dofs

        self.actuated_dof_indices = list(range(self.num_hand_dofs))

        # self.hand_dof_lower_limits = []
        # self.hand_dof_upper_limits = []
        # self.hand_dof_default_pos = []
        # self.hand_dof_default_vel = []
        # self.sensors = []

        for i in range(self.num_hand_dofs):
            # self.hand_dof_lower_limits.append(builder.joint_limit_lower[i])
            # self.hand_dof_upper_limits.append(builder.joint_limit_upper[i])
            # self.hand_dof_default_pos.append(0.0)
            # self.hand_dof_default_vel.append(0.0)

            stiffness = 1.0
            damping = 0.1
            armature = 0.001

            if self.use_torque_control:
                stiffness = 1.0
                damping = 1.0

            builder.joint_target_ke[i] = stiffness
            builder.joint_target_kd[i] = damping
            builder.joint_armature[i] = armature
        builder.joint_limit_ke[:] = builder.joint_target_ke[:]
        builder.joint_limit_kd[:] = builder.joint_target_kd[:]
        if self.use_pd_control or self.use_torque_control:
            builder.joint_axis_mode[:] = [wp.sim.JOINT_MODE_FORCE] * len(builder.joint_axis_mode)
        else:
            builder.joint_axis_mode[:] = [wp.sim.JOINT_MODE_TARGET_POSITION] * len(builder.joint_axis_mode)
        builder.joint_q[:] = [0.0] * len(builder.joint_q)
        builder.joint_act[:] = [0.0] * len(builder.joint_act)

    def create_object(self, builder):
        if self.object_type == "block":
            self.create_object_urdf(builder)
        else:
            raise NotImplementedError

    def create_object_urdf(self, builder):
        object_asset_file = self.asset_files_dict[self.object_type]

        density = 1000.0
        if self.object_type == "block":
            # density = 400.0
            density = 3641.33  # mass=1.0
            # density = 3641.33 * 2
        else:
            raise NotImplementedError

        p = [0.0, 0.0, 0.0]
        p[0] = self.hand_start_pose.p[0]
        pose_dy, pose_dz = -0.39, 0.10
        pose_dy += 0.256
        p[1] = self.hand_start_pose.p[1] + pose_dy
        p[2] = self.hand_start_pose.p[2] + pose_dz
        if self.object_type == "pen":
            p[2] = self.hand_start_pose.p[2] + 0.02
        q = wp.quat_identity()
        object_start_pose = wp.transform(p, q)

        self.goal_displacement = [-0.2, -0.06, 0.12]
        # self.goal_displacement_tensor = None
        # goal_p = (np.array(object_start_pose.p) + np.array(self.goal_displacement)).tolist()
        # goal_p[2] -= 0.04
        # self.goal_start_pose = wp.transform(goal_p, wp.quat_identity())

        wp.sim.parse_urdf(
            os.path.join(
                self.asset_dir,
                f"isaacgymenvs/{object_asset_file}",
            ),
            builder,
            xform=object_start_pose,
            density=density,
            contact_ke=100000.0,
            contact_kd=1000.0,
            contact_kf=1000.0,
            contact_mu=0.5,
            floating=True,
            # contact_thickness=0.02,
            # contact_thickness=0.001,
            enable_self_collisions=False,
            ignore_inertial_definitions=True,
            collapse_fixed_joints=True,
            #
            force_show_colliders=True,
            hide_visuals=True,
        )

    def init_sim(self):
        super().init_sim()
        # self.print_model_info()

        with torch.no_grad():
            self.joint_act = wp.to_torch(self.model.joint_act).view(self.num_envs, -1).clone()
            self.joint_act_indices = ...

            self.start_joint_q = self.state.joint_q.view(self.num_envs, -1).clone()
            self.start_joint_qd = self.state.joint_qd.view(self.num_envs, -1).clone()

            M = self.num_hand_dofs

            # --- From _create_envs()

            joint_limit_lower = wp.to_torch(self.model.joint_limit_lower).view(self.num_envs, -1).clone()
            joint_limit_upper = wp.to_torch(self.model.joint_limit_upper).view(self.num_envs, -1).clone()
            self.hand_dof_lower_limits = joint_limit_lower[:, :M]
            self.hand_dof_upper_limits = joint_limit_upper[:, :M]

            self.hand_dof_default_pos = wp.to_torch(self.model.joint_q).view(self.num_envs, -1).clone()
            self.hand_dof_default_pos = self.hand_dof_default_pos[:, :M]

            self.object_rb_masses = None

            # only pos, no vel
            self.object_init_state = self.start_joint_q[:, M:]
            self.goal_states = self.object_init_state.clone()
            self.goal_states[:, self.up_axis_idx] -= 0.04
            self.goal_init_state = self.goal_states.clone()
            self.hand_start_states = None

            self.hand_indices = None
            self.object_indices = None
            self.goal_object_indices = None

            if self.obs_type == "full_state" or self.asymmetric_obs:
                self.dof_force_tensor = None

            # create some wrapper tensors for different slices
            self.hand_default_dof_pos = self.start_joint_q[:, :M]
            self.dof_state = None
            self.hand_dof_state = None
            self.hand_dof_pos = None
            self.hand_dof_vel = None

            self.rigid_body_states = None
            self.num_bodies = None

            self.root_state_tensor = None

            # self.num_dofs = self.joint_act.shape[-1]
            # print("Num dofs: ", self.num_dofs)

            self.prev_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
            # self.cur_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)

            # self.global_indices = torch.arange(self.num_envs * 3, dtype=torch.int32, device=self.device).view(self.num_envs, -1)
            self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device)
            self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device)
            self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device)
            self.x_unit_tensor = self.x_unit_tensor.repeat((self.num_envs, 1))
            self.y_unit_tensor = self.y_unit_tensor.repeat((self.num_envs, 1))
            self.z_unit_tensor = self.z_unit_tensor.repeat((self.num_envs, 1))

            self.goal_reset_buf = self.reset_buf.clone()
            self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

            # self.av_factor = torch.tensor(self.av_factor, dtype=torch.float, device=self.device)

            self.total_successes = 0
            self.total_resets = 0

            # object apply random forces parameters
            self.force_decay = torch.tensor(self.force_decay, dtype=torch.float, device=self.device)
            self.force_prob_range = torch.tensor(self.force_prob_range, dtype=torch.float, device=self.device)
            self.random_force_prob = torch.exp(
                (torch.log(self.force_prob_range[0]) - torch.log(self.force_prob_range[1]))
                * torch.rand(self.num_envs, device=self.device)
                + torch.log(self.force_prob_range[1])
            )

            # self.rb_forces = torch.zeros((self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device)

    def reset_idx(self, env_ids):
        if self.early_termination:
            old_prev_targets = self.prev_targets
            # old_cur_targets = self.cur_targets
        self.prev_targets = torch.zeros_like(self.prev_targets)
        # self.cur_targets = torch.zeros_like(self.cur_targets)
        super().reset_idx(env_ids)

        self.successes[env_ids] = 0.0

        if self.early_termination:
            reset_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            reset_mask[env_ids] = True
            prev_env_ids = torch.where(~reset_mask)[0]

            self.prev_targets.index_copy_(0, prev_env_ids, old_prev_targets[prev_env_ids])
            # self.cur_targets.index_copy_(0, prev_env_ids, old_cur_targets[prev_env_ids])

    @torch.no_grad()
    def randomize_init(self, env_ids):
        joint_q = self.state.joint_q.view(self.num_envs, -1)
        joint_qd = self.state.joint_qd.view(self.num_envs, -1)

        M = self.num_hand_dofs
        hand_dof_pos = joint_q[:, :M]
        hand_dof_vel = joint_qd[:, :M]
        object_pose = joint_q[:, M:]
        object_vel = joint_qd[:, M:]

        # generate random values
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_hand_dofs * 2 + 5), device=self.device)

        # randomize start object poses
        self.reset_target_pose(env_ids)

        # # reset rigid body forces
        # self.rb_forces[env_ids, :, :] = 0.0

        # reset object
        object_pose[env_ids, 0:2] += self.reset_position_noise * rand_floats[:, 0:2]
        object_pose[env_ids, self.up_axis_idx] += self.reset_position_noise * rand_floats[:, self.up_axis_idx]

        new_object_rot = randomize_rotation(
            rand_floats[:, 3],
            rand_floats[:, 4],
            self.x_unit_tensor[env_ids],
            self.y_unit_tensor[env_ids],
        )
        if self.object_type == "pen":
            rand_angle_y = torch.tensor(0.3)
            new_object_rot = randomize_rotation_pen(
                rand_floats[:, 3],
                rand_floats[:, 4],
                rand_angle_y,
                self.x_unit_tensor[env_ids],
                self.y_unit_tensor[env_ids],
                self.z_unit_tensor[env_ids],
            )

        if self.use_quat:
            object_pose[env_ids, 3:7] = new_object_rot
        else:
            rx, ry, rz = get_euler_xyz(new_object_rot)
            object_pose[env_ids, 3] = rx
            object_pose[env_ids, 4] = ry
            object_pose[env_ids, 5] = rz

        object_vel[env_ids, 0:6] = 0.0

        # reset random force probabilities
        self.random_force_prob[env_ids] = torch.exp(
            (torch.log(self.force_prob_range[0]) - torch.log(self.force_prob_range[1]))
            * torch.rand(len(env_ids), device=self.device)
            + torch.log(self.force_prob_range[1])
        )

        # reset hand
        delta_max = (self.hand_dof_upper_limits - self.hand_dof_default_pos)[env_ids, :]
        delta_min = (self.hand_dof_lower_limits - self.hand_dof_default_pos)[env_ids, :]
        rand_delta = delta_min + (delta_max - delta_min) * 0.5 * (rand_floats[:, 5 : 5 + M] + 1)

        pos = self.hand_default_dof_pos[env_ids, :] + self.reset_dof_pos_noise * rand_delta
        hand_dof_pos[env_ids, :] = pos
        hand_dof_vel[env_ids, :] += self.reset_dof_vel_noise * rand_floats[:, 5 + M : 5 + M * 2]

        self.prev_targets[env_ids, :M] = pos
        # self.cur_targets[env_ids, :M] = pos

    @torch.no_grad()
    def reset_target_pose(self, env_ids, apply_reset=False):
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), 4), device=self.device)

        new_rot = randomize_rotation(
            rand_floats[:, 0],
            rand_floats[:, 1],
            self.x_unit_tensor[env_ids],
            self.y_unit_tensor[env_ids],
        )

        self.goal_states[env_ids, 0:3] = self.goal_init_state[env_ids, 0:3]
        if self.use_quat:
            self.goal_states[env_ids, 3:7] = new_rot
        else:
            rx, ry, rz = get_euler_xyz(new_rot)
            self.goal_states[env_ids, 3] = rx
            self.goal_states[env_ids, 4] = ry
            self.goal_states[env_ids, 5] = rz

        if apply_reset:
            pass
        self.goal_reset_buf[env_ids] = 0

    def pre_physics_step(self, actions):
        actions = actions.view(self.num_envs, -1)
        actions = torch.clip(actions, -1.0, 1.0)
        self.actions = actions
        acts = self.action_scale * actions

        prev_targets = self.prev_targets
        if self.use_pd_control:
            # https://github.com/HaozhiQi/hora/blob/aa4d654d17eedf53104c317aa5262088bf2d825c/hora/tasks/allegro_hand_hora.py#L439
            raise NotImplementedError
        elif self.use_torque_control:
            cur_targets = acts
        elif self.use_relative_control:
            dt = self.frame_dt
            targets = prev_targets + self.hand_dof_speed_scale * dt * acts
            # cur_targets = tensor_clamp(targets, self.hand_dof_lower_limits, self.hand_dof_upper_limits)
            cur_targets = torch.clamp(targets, self.hand_dof_lower_limits, self.hand_dof_upper_limits)
        else:
            cur_targets = scale(acts, self.hand_dof_lower_limits, self.hand_dof_upper_limits)
            cur_targets = self.act_moving_average * cur_targets + (1.0 - self.act_moving_average) * prev_targets
            # cur_targets = tensor_clamp(cur_targets, self.hand_dof_lower_limits, self.hand_dof_upper_limits)
            cur_targets = torch.clamp(cur_targets, self.hand_dof_lower_limits, self.hand_dof_upper_limits)

        acts = cur_targets
        self.prev_targets = cur_targets.detach().clone()

        if self.joint_act_indices is ...:
            self.control.assign("joint_act", acts.flatten())
        else:
            joint_act = self.scatter_actions(self.joint_act, self.joint_act_indices, acts)
            self.control.assign("joint_act", joint_act.flatten())

        if self.force_scale > 0.0:
            raise NotImplementedError

    def compute_observations(self):
        joint_q = self.state.joint_q.clone().view(self.num_envs, -1)
        joint_qd = self.state.joint_qd.clone().view(self.num_envs, -1)

        M = self.num_hand_dofs
        object_q = joint_q[:, M:]
        object_qd = joint_qd[:, M:]  # ang_vel, lin_vel

        # object_pose = object_q[:, 0:7]
        _object_pos = object_q[:, 0:3]
        object_pos = object_q[:, 0:3] - self.env_offsets
        if self.use_quat:
            object_rot = object_q[:, 3:7]
        else:
            object_rot = object_q[:, 3:6]
            object_rot = quat_from_euler_xyz(object_rot[:, 0], object_rot[:, 1], object_rot[:, 2])
        object_linvel = object_qd[:, 3:6]
        object_angvel = object_qd[:, 0:3]
        _object_pose = (object_pos, object_rot)

        # twist -> com velocity
        object_linvel = object_linvel - torch.cross(_object_pos, object_angvel, dim=-1)

        # goal_pose = self.goal_states[:, 0:7]
        goal_pos = self.goal_states[:, 0:3] - self.env_offsets
        if self.use_quat:
            goal_rot = self.goal_states[:, 3:7]
        else:
            goal_rot = self.goal_states[:, 3:6]
            goal_rot = quat_from_euler_xyz(goal_rot[:, 0], goal_rot[:, 1], goal_rot[:, 2])
        _goal_pose = (goal_pos, goal_rot)

        object_goal_rot_diff = quat_mul(object_rot, quat_conjugate(goal_rot))

        hand_dof_pos = joint_q[:, :M]
        hand_dof_pos = unscale(hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits)

        hand_dof_vel = joint_qd[:, :M]
        hand_dof_vel = self.vel_obs_scale * hand_dof_vel

        if self.with_joint_tau:
            # body_f = self.state.body_f.clone().view(self.num_envs, -1, 6)
            joint_tau = self.state.joint_tau.clone().view(self.num_envs, -1)

            # hand_f = body_f[:, :M]  # 0:3 torque, 3:6 force
            hand_dof_force = self.force_torque_obs_scale * joint_tau[:, :M]
            _hand_dof_force = [hand_dof_force]
        else:
            _hand_dof_force = []

        if self.obs_type == "full_no_vel":
            obs_buf = [
                hand_dof_pos,
                *_object_pose,
                *_goal_pose,
                object_goal_rot_diff,
                self.actions.clone(),
            ]
        elif self.obs_type == "full":
            obs_buf = [
                hand_dof_pos,
                hand_dof_vel,
                *_object_pose,
                object_linvel,
                self.vel_obs_scale * object_angvel,
                *_goal_pose,
                object_goal_rot_diff,
                self.actions.clone(),
            ]
        elif self.obs_type == "full_state":
            obs_buf = [
                hand_dof_pos,
                hand_dof_vel,
                *_hand_dof_force,
                # 48
                *_object_pose,
                object_linvel,
                self.vel_obs_scale * object_angvel,
                # 61
                *_goal_pose,
                object_goal_rot_diff,
                # 72
                self.actions.clone(),
                # 88
            ]
        else:
            print("Unknown observations type!")

        self.obs_buf = {"obs": torch.cat(obs_buf, dim=-1)}
        if self.asymmetric_obs:
            state_buf = [
                hand_dof_pos,
                hand_dof_vel,
                *_hand_dof_force,
                # 48
                *_object_pose,
                object_linvel,
                self.vel_obs_scale * object_angvel,
                # 61
                *_goal_pose,
                object_goal_rot_diff,
                # 72
                self.actions.clone(),
                # 88
            ]
            self.obs_buf["state"] = torch.cat(state_buf, dim=-1)

        # for computing rewards
        self._obs_buf = {
            "object_pos": object_pos,
            "object_rot": object_rot,
            "goal_pos": goal_pos,
            "goal_rot": goal_rot,
        }

    def compute_reward(self):
        (
            self.rew_buf,
            self.reset_buf,
            self.progress_buf,
            self.terminated_buf,
            self.truncated_buf,
            self.goal_reset_buf,
            self.successes,
            self.consecutive_successes,
        ) = compute_hand_reward(
            self.rew_buf,
            self.reset_buf,
            self.progress_buf,
            self.terminated_buf,
            self.truncated_buf,
            self.episode_length,
            self.early_termination,
            #
            self.goal_reset_buf,
            self.successes,
            self.consecutive_successes,
            #
            self._obs_buf["object_pos"],
            self._obs_buf["object_rot"],
            self._obs_buf["goal_pos"],
            self._obs_buf["goal_rot"],
            self.actions,
            #
            self.dist_reward_scale,
            self.rot_reward_scale,
            self.rot_eps,
            self.action_penalty_scale,
            self.success_tolerance,
            self.reach_goal_bonus,
            self.fall_dist,
            self.fall_penalty,
            self.max_consecutive_successes,
            self.av_factor,
            self.ignore_z,
        )

    def render(self, state=None):
        if self.renderer is not None:
            with wp.ScopedTimer("render", False):
                self.render_time += self.frame_dt
                self.renderer.begin_frame(self.render_time)
                # render state 1 (swapped with state 0 just before)
                self.renderer.render(state or self.state_1)
                self.render_transforms(state=state)
                self.render_goal(state=state)
                self.renderer.end_frame()

    def render_transforms(self, state=None):
        state = state or self.state_1

        # TODO: render arrows

    def render_goal(self, state=None):
        state = state or self.state_1

        # TODO: render goal object

    def run_cfg(self):
        iters = 4
        actions = [
            torch.rand((self.num_envs, self.num_actions), device=self.device) * 2.0 - 1.0
            # torch.zeros((self.num_envs, self.num_actions), device=self.device)
            for _ in range(self.episode_length)
        ]
        actions = [a.requires_grad_() for a in actions]

        opt = torch.optim.Adam(actions, lr=0.1)

        policy = actions

        return iters, opt, policy


# @torch.jit.script
def compute_hand_reward(
    rew_buf,
    reset_buf,
    progress_buf,
    terminated_buf,
    truncated_buf,
    max_episode_length: float,
    early_termination: bool,
    #
    goal_reset_buf,
    successes,
    consecutive_successes,
    #
    object_pos,
    object_rot,
    target_pos,
    target_rot,
    actions,
    #
    dist_reward_scale: float,
    rot_reward_scale: float,
    rot_eps: float,
    action_penalty_scale: float,
    success_tolerance: float,
    reach_goal_bonus: float,
    fall_dist: float,
    fall_penalty: float,
    max_consecutive_successes: int,
    av_factor: float,
    ignore_z_rot: bool,
):
    # Distance from the hand to the object
    goal_dist = torch.norm(object_pos - target_pos, p=2, dim=-1)

    if ignore_z_rot:
        success_tolerance = 2.0 * success_tolerance

    # Orientation alignment for the cube in hand and goal cube
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

    dist_rew = goal_dist * dist_reward_scale
    rot_rew = 1.0 / (torch.abs(rot_dist) + rot_eps) * rot_reward_scale

    action_penalty = torch.sum(actions**2, dim=-1)

    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
    reward = dist_rew + rot_rew + action_penalty * action_penalty_scale

    # Find out which envs hit the goal and update successes count
    goal_resets = torch.where(torch.abs(rot_dist) <= success_tolerance, torch.ones_like(goal_reset_buf), goal_reset_buf)
    successes = successes + goal_resets

    # Success bonus: orientation is within `success_tolerance` of goal orientation
    reward = torch.where(goal_resets == 1, reward + reach_goal_bonus, reward)

    # Fall penalty: distance to the goal is larger than a threshold
    reward = torch.where(goal_dist >= fall_dist, reward + fall_penalty, reward)

    # Check env termination conditions, including maximum success number
    if early_termination:
        terminated = torch.where(goal_dist >= fall_dist, torch.ones_like(reset_buf), reset_buf)

        if max_consecutive_successes > 0:
            # Reset progress buffer on goal envs if max_consecutive_successes > 0
            progress_buf = torch.where(
                torch.abs(rot_dist) <= success_tolerance, torch.zeros_like(progress_buf), progress_buf
            )
            terminated = torch.where(successes >= max_consecutive_successes, torch.ones_like(reset_buf), reset_buf)
    else:
        terminated = torch.where(torch.zeros_like(reset_buf), torch.ones_like(reset_buf), reset_buf)

    # timed_out = progress_buf >= max_episode_length - 1
    truncated = progress_buf > max_episode_length - 1
    resets = torch.where(truncated, torch.ones_like(reset_buf), reset_buf)
    if early_termination:
        resets = torch.where(terminated, torch.ones_like(resets), resets)

    # Apply penalty for not reaching the goal
    if max_consecutive_successes > 0:
        reward = torch.where(truncated, reward + 0.5 * fall_penalty, reward)

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    cons_successes = torch.where(
        num_resets > 0,
        av_factor * finished_cons_successes / num_resets + (1.0 - av_factor) * consecutive_successes,
        consecutive_successes,
    )

    return reward, resets, progress_buf, terminated, truncated, goal_resets, successes, cons_successes


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(
        quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
        quat_from_angle_axis(rand1 * np.pi, y_unit_tensor),
    )


@torch.jit.script
def randomize_rotation_pen(rand0, rand1, max_angle, x_unit_tensor, y_unit_tensor, z_unit_tensor):
    rot = quat_mul(
        quat_from_angle_axis(0.5 * np.pi + rand0 * max_angle, x_unit_tensor),
        quat_from_angle_axis(rand0 * np.pi, z_unit_tensor),
    )
    return rot


if __name__ == "__main__":
    run_env(ShadowHand)
