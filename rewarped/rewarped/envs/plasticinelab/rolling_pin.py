import math

import numpy as np
import torch
from gym import spaces

import warp as wp

from ...environment import IntegratorType, run_env
from ...mpm_warp_env_mixin import MPMWarpEnvMixin
from ...warp_env import WarpEnv


class RollingPin(MPMWarpEnvMixin, WarpEnv):
    sim_name = "RollingPin" + "PlasticineLab"
    env_offset = (2.0, 0.0, 2.0)

    integrator_type = IntegratorType.MPM
    sim_substeps_mpm = 40

    eval_fk = True
    kinematic_fk = True
    eval_ik = False

    up_axis = "Y"
    ground_plane = True

    state_tensors_names = ("joint_q", "body_q") + ("mpm_x", "mpm_v", "mpm_C", "mpm_F_trial", "mpm_F", "mpm_stress")
    control_tensors_names = ("joint_act",)

    def __init__(self, num_envs=2, episode_length=300, early_termination=False, **kwargs):
        num_obs = 0
        num_act = 3
        super().__init__(num_envs, num_obs, num_act, episode_length, early_termination, **kwargs)

        self.action_scale = [0.01, 0.01, math.pi / 60.0]  # dx, dy, ry
        self.action_scale = torch.tensor(self.action_scale, device=self.device)
        self.downsample_particle = 250
        self.roller_size = (0.4, 0.03)  # (length, radius)

        self.h = 0.125
        self.mpm_bound = 3
        self.mpm_num_grids = 48
        self.ground_y = self.mpm_bound / self.mpm_num_grids

    @property
    def observation_space(self):
        d = {
            "particle_q": spaces.Box(low=-np.inf, high=np.inf, shape=(self.downsample_particle, 3), dtype=np.float32),
            "com_q": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "joint_q": spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_act,), dtype=np.float32),
        }
        d = spaces.Dict(d)
        return d

    def create_modelbuilder(self):
        builder = super().create_modelbuilder()
        builder.rigid_contact_margin = 0.05
        return builder

    def create_builder(self):
        builder = super().create_builder()
        self.create_builder_mpm(builder)
        return builder

    def create_env(self, builder):
        self.create_roller(builder)

    def create_roller(self, builder):
        l, r = self.roller_size

        builder.add_articulation()
        b = builder.add_body(
            # origin=wp.transform((0.0, 0.0, 0.0), wp.quat_identity()),
            name="body_roller",
        )

        builder.add_shape_capsule(
            body=b,
            radius=r,
            half_height=l / 2,
            pos=wp.vec3(0.0, 0.0, 0.5),
            rot=wp.quat_from_axis_angle((1.0, 0.0, 0.0), -math.pi * 0.5),
            density=1000.0,
            has_ground_collision=True,
            has_shape_collision=True,
        )
        builder.add_shape_capsule(
            body=b,
            radius=r * 0.4,
            half_height=l / 2 * 1.5,
            pos=wp.vec3(0.0, 0.0, 0.5),
            rot=wp.quat_from_axis_angle((1.0, 0.0, 0.0), -math.pi * 0.5),
            density=1000.0,
            has_ground_collision=True,
            has_shape_collision=True,
        )

        ly = 2 * r + self.ground_y
        hy = 0.25 + ly

        lin_axes = [
            wp.sim.JointAxis([1.0, 0.0, 0.0], limit_lower=0.0 + 4 * r, limit_upper=1.0 - 4 * r),
            wp.sim.JointAxis([0.0, 1.0, 0.0], limit_lower=ly, limit_upper=hy),
        ]
        ang_axes = [
            wp.sim.JointAxis([0.0, 1.0, 0.0], limit_lower=-0.66 * math.pi, limit_upper=0.66 * math.pi),
        ]
        builder.add_joint_d6(
            parent=-1,
            child=b,
            linear_axes=lin_axes,
            angular_axes=ang_axes,
            name="joint_roller",
        )
        builder.joint_q[:3] = [0.5, 0.66 * (hy - ly) + ly, 0.0]

    def create_cfg_mpm(self, mpm_cfg):
        mpm_cfg.update(["--physics.sim", "dexdeform"])

        E, nu = 5000.0, 0.2
        yield_stress = 50.0
        rho = 1.0

        num_grids = 48
        body_friction = 0.9

        bound = self.mpm_bound
        num_grids = self.mpm_num_grids
        ground_y = self.ground_y

        h = self.h

        center = (0.5, 0.7 * h + ground_y, 0.5)
        size = (0.3, h, 0.3)
        resolution = 18

        mpm_cfg.update(["--physics.env.physics", "plb_plasticine"])
        mpm_cfg.update(["--physics.env.physics.E", str(E)])
        mpm_cfg.update(["--physics.env.physics.nu", str(nu)])
        mpm_cfg.update(["--physics.env.physics.yield_stress", str(yield_stress)])
        mpm_cfg.update(["--physics.env.rho", str(rho)])

        mpm_cfg.update(["--physics.sim.bound", str(bound)])
        mpm_cfg.update(["--physics.sim.num_grids", str(num_grids)])
        mpm_cfg.update(["--physics.sim.body_friction", str(body_friction)])

        mpm_cfg.update(["--physics.env.shape", "cube_hd"])
        mpm_cfg.update(["--physics.env.shape.center", str(tuple(center))])
        mpm_cfg.update(["--physics.env.shape.size", str(tuple(size))])
        mpm_cfg.update(["--physics.env.shape.resolution", str(resolution)])

        print(mpm_cfg)
        return mpm_cfg

    def create_model(self):
        model = super().create_model()
        self.create_model_mpm(model)
        return model

    def init_sim(self):
        super().init_sim()
        self.print_model_info()

        with torch.no_grad():
            self.joint_act = wp.to_torch(self.model.joint_act).view(self.num_envs, -1).clone()
            self.joint_act_indices = ...

            self.joint_limit_lower = wp.to_torch(self.model.joint_limit_lower).view(self.num_envs, -1).clone()
            self.joint_limit_upper = wp.to_torch(self.model.joint_limit_upper).view(self.num_envs, -1).clone()

    @torch.no_grad()
    def randomize_init(self, env_ids):
        mpm_x = self.state.mpm_x.view(self.num_envs, -1, 3)
        mpm_x -= self.env_offsets.view(self.num_envs, 1, 3)

        bounds = torch.tensor([0.1, 0.05, 0.1], device=self.device)
        rand_pos = (torch.rand(size=(len(env_ids), 1, 3), device=self.device) - 0.5) * 2.0
        rand_pos[:, :, 1] = rand_pos[:, :, 1] / 2.0 + 0.5  # +height only
        rand_pos = torch.round(rand_pos, decimals=3)
        mpm_x[env_ids, :, :] += bounds * rand_pos

        scale = 0.1
        rand_scale = (torch.rand(size=(len(env_ids), 1, 1), device=self.device) - 0.5) * 2.0
        rand_scale = torch.round(rand_scale, decimals=3)
        mpm_x[env_ids, :, :] *= 1.0 + (scale * rand_scale)

        mpm_x += self.env_offsets.view(self.num_envs, 1, 3)

    def pre_physics_step(self, actions):
        actions = actions.view(self.num_envs, -1)
        actions = torch.clip(actions, -1.0, 1.0)
        self.actions = actions
        acts = self.action_scale * actions

        if self.kinematic_fk:
            # ensure joint limit
            joint_q = self.state.joint_q.clone().view(self.num_envs, -1)
            joint_q = joint_q.detach()
            acts = torch.clip(acts, self.joint_limit_lower - joint_q, self.joint_limit_upper - joint_q)

        if self.joint_act_indices is ...:
            self.control.assign("joint_act", acts.flatten())
        else:
            joint_act = self.scatter_actions(self.joint_act, self.joint_act_indices, acts)
            self.control.assign("joint_act", joint_act.flatten())

    def compute_observations(self):
        joint_q = self.state.joint_q.clone().view(self.num_envs, -1)
        particle_q = self.state.mpm_x.clone().view(self.num_envs, -1, 3)

        particle_q -= self.env_offsets.view(self.num_envs, 1, 3)

        # TODO: fix joint creation so joint_q also includes env_offsets

        if self.downsample_particle is not None:
            num_full, N = particle_q.shape[1], self.downsample_particle
            downsample = num_full // N
            particle_q = particle_q[:, ::downsample, :]
            particle_q = particle_q[:, :N, :]
            # assert particle_q.shape[1] == N

        com_q = particle_q.mean(1)

        self.obs_buf = {
            "joint_q": joint_q,
            "particle_q": particle_q,
            "com_q": com_q,
        }

    def compute_reward(self):
        # joint_q = self.obs_buf["joint_q"]
        particle_q = self.obs_buf["particle_q"]
        # com_q = self.obs_buf["com_q"]

        h = particle_q[:, :, 1]
        d = h.mean(-1) / self.h
        dist_rew = 1.0 / (1.0 + d)
        dist_rew = dist_rew.square()
        dist_rew = torch.where(d <= 0.33, dist_rew * 2, dist_rew)

        flatten_rew = -h.var(-1)

        rew = dist_rew + flatten_rew

        reset_buf, progress_buf = self.reset_buf, self.progress_buf
        max_episode_steps, early_termination = self.episode_length, self.early_termination
        truncated = progress_buf > max_episode_steps - 1
        reset = torch.where(truncated, torch.ones_like(reset_buf), reset_buf)
        if early_termination:
            raise NotImplementedError
        else:
            terminated = torch.where(torch.zeros_like(reset), torch.ones_like(reset), reset)
        self.rew_buf, self.reset_buf, self.terminated_buf, self.truncated_buf = rew, reset, terminated, truncated

    def render(self, state=None):
        if self.renderer is not None:
            with wp.ScopedTimer("render", False):
                self.render_time += self.frame_dt
                self.renderer.begin_frame(self.render_time)
                # render state 1 (swapped with state 0 just before)
                self.renderer.render(state or self.state_1)
                self.render_mpm(state=state)
                self.renderer.end_frame()

    def run_cfg(self):
        actions = [
            torch.rand((self.num_envs, self.num_actions), device=self.device) * 2.0 - 1.0
            # torch.zeros((self.num_envs, self.num_actions), device=self.device)
            for _ in range(self.episode_length)
        ]

        iters = 10
        train_rate = 0.01
        actions = [a.requires_grad_() for a in actions]
        opt = torch.optim.Adam(actions, lr=train_rate)

        policy = actions

        return iters, opt, policy


if __name__ == "__main__":
    run_env(RollingPin, no_grad=False)
