# from https://github.com/NVIDIA/warp/blob/release-1.3/warp/examples/sim/example_granular.py

# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Sim Granular
#
# Shows how to set up a particle-based granular material model using the
# wp.sim.ModelBuilder().
#
###########################################################################

import torch

import warp as wp

from ...environment import IntegratorType, run_env
from ...warp_env import WarpEnv


class Granular(WarpEnv):
    sim_name = "Granular" + "WarpExamples"
    env_offset = (0.0, 0.0, 10.0)

    # integrator_type = IntegratorType.EULER
    # sim_substeps_euler = 64
    # euler_settings = dict(angular_damping=0.0)

    integrator_type = IntegratorType.FEATHERSTONE
    sim_substeps_featherstone = 64
    featherstone_settings = dict(angular_damping=0.0, update_mass_matrix_every=sim_substeps_featherstone)

    eval_fk = False
    eval_ik = False

    frame_dt = 1.0 / 60.0
    episode_duration = 6.67  # seconds
    up_axis = "Y"
    ground_plane = True

    state_tensors_names = ("particle_q", "particle_qd")

    def __init__(self, num_envs=8, episode_length=-1, early_termination=False, **kwargs):
        num_obs = 0
        num_act = 0

        episode_length = int(self.episode_duration / self.frame_dt)
        super().__init__(num_envs, num_obs, num_act, episode_length, early_termination, **kwargs)

        self.action_scale = 1.0
        self.radius = 0.1

    def create_modelbuilder(self):
        builder = super().create_modelbuilder()
        builder.default_particle_radius = self.radius
        return builder

    def create_env(self, builder):
        self.create_granular(builder)

    def create_granular(self, builder):
        builder.add_particle_grid(
            dim_x=16,
            dim_y=32,
            dim_z=16,
            cell_x=self.radius * 2.0,
            cell_y=self.radius * 2.0,
            cell_z=self.radius * 2.0,
            pos=wp.vec3(0.0, 1.0, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(5.0, 0.0, 0.0),
            mass=0.1,
            jitter=self.radius * 0.1,
        )

    def create_model(self):
        model = super().create_model()

        model.particle_kf = 25.0
        model.soft_contact_kd = 100.0
        model.soft_contact_kf *= 2.0

        return model

    def init_sim(self):
        super().init_sim()
        # self.print_model_info()

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)

    @torch.no_grad()
    def randomize_init(self, env_ids):
        pass

    def pre_physics_step(self, actions):
        pass

    def do_physics_step(self):
        if self.state_0.particle_q is not None:
            self.model.particle_grid.build(self.state_0.particle_q, self.radius * 2.0)

        return super().do_physics_step()

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
    run_env(Granular)
