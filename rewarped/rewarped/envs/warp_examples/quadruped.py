# from https://github.com/NVIDIA/warp/blob/release-1.3/warp/examples/sim/example_quadruped.py

# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Sim Quadruped
#
# Shows how to set up a simulation of a rigid-body quadruped articulation
# from a URDF using the wp.sim.ModelBuilder().
# Note this example does not include a trained policy.
#
###########################################################################

import math
import os

import torch

import warp as wp
import warp.examples

from ...environment import IntegratorType, run_env
from ...warp_env import WarpEnv


class Quadruped(WarpEnv):
    sim_name = "Quadruped" + "WarpExamples"
    env_offset = (0.0, 0.0, 1.0)

    # integrator_type = IntegratorType.EULER
    # sim_substeps_euler = 16
    # euler_settings = dict(angular_damping=0.0)
    # joint_attach_ke = 16000.0
    # joint_attach_kd = 200.0

    integrator_type = IntegratorType.FEATHERSTONE
    sim_substeps_featherstone = 16
    featherstone_settings = dict(angular_damping=0.0, update_mass_matrix_every=sim_substeps_featherstone)

    eval_fk = True
    eval_ik = False

    frame_dt = 1.0 / 100.0
    up_axis = "Y"
    ground_plane = True

    state_tensors_names = ("joint_q", "joint_qd")
    control_tensors_names = ("joint_act",)

    def __init__(self, num_envs=8, episode_length=300, early_termination=False, control_type="pos", **kwargs):
        num_obs = 0
        num_act = 12
        super().__init__(num_envs, num_obs, num_act, episode_length, early_termination, **kwargs)

        self.control_type = control_type
        if control_type == "pos":
            self.action_scale = 1.0
        elif control_type == "force":
            self.action_scale = 50.0
        else:
            raise ValueError(control_type)

    def create_modelbuilder(self):
        builder = super().create_modelbuilder()
        builder.rigid_contact_margin = 0.05
        return builder

    def create_articulation(self, builder):
        wp.sim.parse_urdf(
            os.path.join(warp.examples.get_asset_directory(), "quadruped.urdf"),
            builder,
            xform=wp.transform([0.0, 0.7, 0.0], wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -math.pi * 0.5)),
            floating=True,
            density=1000,
            armature=0.01,
            stiffness=200,
            damping=1,
            contact_ke=1.0e4,
            contact_kd=1.0e2,
            contact_kf=1.0e2,
            contact_mu=1.0,
            limit_ke=1.0e4,
            limit_kd=1.0e1,
        )

        builder.joint_q[-12:] = [0.2, 0.4, -0.6, -0.2, -0.4, 0.6, -0.2, 0.4, -0.6, 0.2, -0.4, 0.6]

        if self.control_type == "pos":
            builder.joint_act[-12:] = [0.2, 0.4, -0.6, -0.2, -0.4, 0.6, -0.2, 0.4, -0.6, 0.2, -0.4, 0.6]
            builder.joint_axis_mode = [wp.sim.JOINT_MODE_TARGET_POSITION] * len(builder.joint_axis_mode)
        elif self.control_type == "force":
            builder.joint_act[-12:] = [0.0] * 12
            builder.joint_axis_mode = [wp.sim.JOINT_MODE_FORCE] * len(builder.joint_axis_mode)
        else:
            raise ValueError(self.control_type)

    def init_sim(self):
        super().init_sim()
        # self.print_model_info()

        with torch.no_grad():
            self.joint_act = wp.to_torch(self.model.joint_act).view(self.num_envs, -1).clone()
            self.joint_act_indices = ...

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

        if self.control_type == "pos":
            self.control.assign("joint_act", wp.to_torch(self.model.joint_act).clone().flatten())
        elif self.control_type == "force":
            if self.joint_act_indices is ...:
                self.control.assign("joint_act", acts.flatten())
            else:
                joint_act = self.scatter_actions(self.joint_act, self.joint_act_indices, acts)
                self.control.assign("joint_act", joint_act.flatten())
        else:
            raise ValueError(self.control_type)

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
    run_env(Quadruped)
