# from https://github.com/NVIDIA/warp/blob/release-1.3/warp/examples/optim/example_walker.py

# Copyright (c) 2024 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Walker
#
# Trains a tetrahedral mesh quadruped to run. Feeds 8 time-varying input
# phases as inputs into a single layer fully connected network with a tanh
# activation function. Interprets the output of the network as tet
# activations, which are fed into the wp.sim soft mesh model. This is
# simulated forward in time and then evaluated based on the center of mass
# momentum of the mesh.
#
###########################################################################

import math
import os

import numpy as np
import torch
import torch.nn as nn
from gym import spaces
from pxr import Usd, UsdGeom

import warp as wp
import warp.examples

from ...environment import IntegratorType, run_env
from ...warp_env import WarpEnv


class Jumper(WarpEnv):
    sim_name = "Jumper" + "GradSim"
    env_offset = (0.0, 0.0, 7.5)

    # integrator_type = IntegratorType.EULER
    # sim_substeps_euler = 80
    # euler_settings = dict(angular_damping=0.05)

    integrator_type = IntegratorType.FEATHERSTONE
    sim_substeps_featherstone = 80
    featherstone_settings = dict(angular_damping=0.05, update_mass_matrix_every=sim_substeps_featherstone)

    eval_fk = False
    eval_ik = False

    frame_dt = 1.0 / 60.0
    up_axis = "Y"
    ground_plane = True

    state_tensors_names = ("particle_q", "particle_qd")
    control_tensors_names = ("tet_activations",)

    def __init__(self, which="table", num_envs=8, episode_length=300, early_termination=False, **kwargs):
        if which == "bear":
            full_num_obs = 1986  # model.particle_count
            full_num_act = 5354  # model.tet_count
            action_scale = 0.3  # activation_strength
            particle_q_downsample = 6
            act_downsample = 17  # 17 | 35 | 45 | 51 ... (full_num_act // N + 1) * N - full_num_act == 1
            act_padding = 1
        elif which == "ostrich":
            full_num_obs = 646
            full_num_act = 1582
            action_scale = 0.3
            particle_q_downsample = 1
            act_downsample = 1
            act_padding = 0
        elif which == "gear":
            full_num_obs = 480
            full_num_act = 1152
            action_scale = 0.3
            particle_q_downsample = 1
            act_downsample = 1
            act_padding = 0
        elif which == "table":
            full_num_obs = 204
            full_num_act = 444
            action_scale = 0.3
            particle_q_downsample = 1
            act_downsample = 2
            act_padding = 0
        else:
            raise ValueError(which)

        num_obs = full_num_obs // particle_q_downsample
        num_act = full_num_act // act_downsample
        super().__init__(num_envs, num_obs, num_act, episode_length, early_termination, **kwargs)

        self.which = which
        self.particle_q_downsample = particle_q_downsample
        self.action_scale = action_scale
        self.act_downsample = act_downsample
        self.act_padding = act_padding

        self.phase_count = 8
        self.phase_step = 2.0 * math.pi / self.phase_count
        self.phase_freq = 5.0

    @property
    def observation_space(self):
        d = {
            "phase": spaces.Box(low=-np.inf, high=np.inf, shape=(self.phase_count,), dtype=np.float32),
            "loss": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "com_q": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "com_qd": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "particle_q": spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_obs, 3), dtype=np.float32),
            "particle_qd": spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_obs, 3), dtype=np.float32),
            "actions": spaces.Box(low=-1.0, high=1.0, shape=(self.num_act,), dtype=np.float32),
        }
        d = spaces.Dict(d)
        return d

    def create_modelbuilder(self):
        builder = super().create_modelbuilder()
        builder.default_particle_radius = 0.05
        return builder

    def create_env(self, builder):
        if self.which == "bear":
            self.create_bear(builder)
        elif self.which == "ostrich":
            self.create_ostrich(builder)
        elif self.which == "gear":
            self.create_gear(builder)
        elif self.which == "table":
            self.create_table(builder)
        else:
            raise ValueError(self.which)

    def create_bear(self, builder):
        asset_stage = Usd.Stage.Open(os.path.join(warp.examples.get_asset_directory(), "bear.usd"))
        geom = UsdGeom.Mesh(asset_stage.GetPrimAtPath("/root/bear"))
        points = geom.GetPointsAttr().Get()

        xform = geom.ComputeLocalToWorldTransform(0.0)
        for i in range(len(points)):
            points[i] = xform.Transform(points[i])

        points = [wp.vec3(point) for point in points]
        tet_indices = geom.GetPrim().GetAttribute("tetraIndices").Get()

        builder.add_soft_mesh(
            pos=wp.vec3(0.0, 0.2, 0.0),
            rot=wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), math.pi * 0.5),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=points,
            indices=tet_indices,
            density=1.0,
            k_mu=2000.0,
            k_lambda=2000.0,
            k_damp=2.0,
            tri_ke=0.0,
            tri_ka=1e-8,
            tri_kd=0.0,
            tri_drag=0.0,
            tri_lift=0.0,
        )

    def create_ostrich(self, builder):
        asset_stage = Usd.Stage.Open(os.path.join(self.asset_dir, "dflex/usd", "ostrich.usda"))
        geom = UsdGeom.Mesh(asset_stage.GetPrimAtPath("/mesh"))
        points = geom.GetPointsAttr().Get()

        xform = geom.ComputeLocalToWorldTransform(0.0)
        for i in range(len(points)):
            points[i] = xform.Transform(points[i])

        points = [wp.vec3(point) for point in points]
        tet_indices = geom.GetPrim().GetAttribute("tetraIndices").Get()

        q = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -math.pi * 0.5)
        q = q * wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), math.pi)

        builder.add_soft_mesh(
            pos=wp.vec3(0.0, 0.2, 0.0),
            rot=q,
            scale=2.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=points,
            indices=tet_indices,
            density=1.0,
            k_mu=2000.0,
            k_lambda=2000.0,
            k_damp=1.0,
            tri_ke=0.0,
            tri_ka=1e-8,
            tri_kd=0.0,
            tri_drag=0.0,
            tri_lift=0.0,
        )

    def create_gear(self, builder):
        asset_stage = Usd.Stage.Open(os.path.join(self.asset_dir, "dflex/usd", "prop.usda"))
        geom = UsdGeom.Mesh(asset_stage.GetPrimAtPath("/mesh"))
        points = geom.GetPointsAttr().Get()

        xform = geom.ComputeLocalToWorldTransform(0.0)
        for i in range(len(points)):
            points[i] = xform.Transform(points[i])

        points = [wp.vec3(point) for point in points]
        tet_indices = geom.GetPrim().GetAttribute("tetraIndices").Get()

        builder.add_soft_mesh(
            pos=wp.vec3(0.0, 2.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            # vel=wp.vec3(0.0, 0.0, 0.0),
            vel=wp.vec3(1.5, 0.0, 0.0),
            vertices=points,
            indices=tet_indices,
            density=1.0,
            k_mu=1000.0,
            k_lambda=1000.0,
            k_damp=1.0,
            tri_ke=0.0,
            tri_ka=1e-8,
            tri_kd=0.0,
            tri_drag=0.0,
            tri_lift=0.0,
        )

    def create_table(self, builder):
        asset_stage = Usd.Stage.Open(os.path.join(self.asset_dir, "dflex/usd", "table.usda"))
        geom = UsdGeom.Mesh(asset_stage.GetPrimAtPath("/mesh"))
        points = geom.GetPointsAttr().Get()

        xform = geom.ComputeLocalToWorldTransform(0.0)
        for i in range(len(points)):
            points[i] = xform.Transform(points[i])

        points = [wp.vec3(point) for point in points]
        tet_indices = geom.GetPrim().GetAttribute("tetraIndices").Get()

        builder.add_soft_mesh(
            pos=wp.vec3(0.0, 0.2, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=points,
            indices=tet_indices,
            density=1.0,
            k_mu=1000.0,
            k_lambda=1000.0,
            k_damp=1.0,
            tri_ke=0.0,
            tri_ka=1e-8,
            tri_kd=0.0,
            tri_drag=0.0,
            tri_lift=0.0,
        )

    def create_model(self):
        model = super().create_model()

        if self.which == "bear":
            model.soft_contact_ke = 2.0e3
            model.soft_contact_kd = 0.1
            model.soft_contact_kf = 10.0
            model.soft_contact_mu = 0.7
        elif self.which == "ostrich":
            model.soft_contact_ke = 2.0e3
            model.soft_contact_kd = 0.1
            model.soft_contact_kf = 10.0
            model.soft_contact_mu = 0.7
        elif self.which == "gear":
            model.soft_contact_ke = 1.0e4
            model.soft_contact_kd = 1.0
            model.soft_contact_kf = 10.0
            model.soft_contact_mu = 0.5
        elif self.which == "table":
            model.soft_contact_ke = 2.0e3
            model.soft_contact_kd = 0.1
            model.soft_contact_kf = 10.0
            model.soft_contact_mu = 0.7
        else:
            raise ValueError(self.which)

        return model

    def init_sim(self):
        super().init_sim()
        self.print_model_info()

        with torch.no_grad():
            self.tet_activations = wp.to_torch(self.model.tet_activations).view(self.num_envs, -1).clone()
            self.tet_activations_indices = ...

            self.start_pos = wp.to_torch(self.model.particle_q).view(self.num_envs, -1, 3).clone().mean(1)

            self.targets = torch.tensor([10.0, 0.0, 0.0], device=self.device).unsqueeze(0)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if self.early_termination:
            raise NotImplementedError
        else:
            # self.start_pos = self.state.particle_q.detach().clone().view(self.num_envs, -1, 3).mean(1)
            self.last_actions = torch.zeros(self.num_envs, self.num_act, device=self.device)

    @torch.no_grad()
    def randomize_init(self, env_ids):
        particle_q = self.state.particle_q.view(self.num_envs, -1, 3)
        particle_q -= self.env_offsets.view(self.num_envs, 1, 3)

        bounds = torch.tensor([0.8, 0.4, 0.8], device=self.device)
        rand_pos = (torch.rand(size=(len(env_ids), 1, 3), device=self.device) - 0.5) * 2.0
        rand_pos[:, :, 1] = rand_pos[:, :, 1] / 2.0 + 0.5  # +height only
        rand_pos = torch.round(rand_pos, decimals=3)
        particle_q[env_ids, :, :] += bounds * rand_pos

        scale = 0.1
        rand_scale = (torch.rand(size=(len(env_ids), 1, 1), device=self.device) - 0.5) * 2.0
        rand_scale = torch.round(rand_scale, decimals=3)
        particle_q[env_ids, :, :] *= 1.0 + (scale * rand_scale)

        particle_q += self.env_offsets.view(self.num_envs, 1, 3)

    def pre_physics_step(self, actions):
        actions = actions.view(self.num_envs, -1)
        actions = torch.clip(actions, -1.0, 1.0)
        self.last_actions = self.actions.clone()
        self.actions = actions
        acts = self.action_scale * actions

        if self.act_downsample > 1:
            N = self.model.tet_count // self.num_envs
            if self.act_padding > 0:
                acts = torch.cat([acts, torch.zeros(self.num_envs, self.act_padding, device=self.device)], dim=-1)
            acts = torch.repeat_interleave(acts, self.act_downsample, dim=-1)
            acts = acts[:, :N]

        if self.tet_activations_indices is ...:
            self.control.assign("tet_activations", acts.flatten())
        else:
            tet_activations = self.scatter_actions(self.tet_activations, self.tet_activations_indices, acts)
            self.control.assign("tet_activations", tet_activations.flatten())

    def compute_observations(self):
        particle_q = self.state.particle_q.clone().view(self.num_envs, -1, 3)
        particle_qd = self.state.particle_qd.clone().view(self.num_envs, -1, 3)

        particle_q -= self.env_offsets.view(self.num_envs, 1, 3)

        # rescaling
        particle_q, particle_qd = 0.1 * particle_q, 0.1 * particle_qd

        com_q = particle_q.mean(1)
        # compute center of mass velocity
        com_qd = particle_qd.mean(1)

        loss = com_qd[:, 2].pow(2).sqrt() + com_qd[:, 1].pow(2).sqrt() - com_qd[:, 0]  # sqrt(vz**2) + sqrt(vy**2) - vx
        loss = loss.reshape(self.num_envs, 1)

        phase_count, phase_step, phase_freq = (
            self.phase_count,
            self.phase_step,
            self.phase_freq,
        )
        sim_time = self.progress_buf * self.frame_dt
        phase_counts = (
            torch.arange(phase_count, dtype=torch.float, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        )
        phase = torch.sin(phase_freq * sim_time.unsqueeze(-1) + phase_counts * phase_step)

        if self.particle_q_downsample > 1:
            particle_q = particle_q[:, :: self.particle_q_downsample, :]
            particle_qd = particle_qd[:, :: self.particle_q_downsample, :]

        self.obs_buf = {
            "phase": phase,
            "loss": loss,
            "com_q": com_q,
            "com_qd": com_qd,
            "particle_q": particle_q,
            "particle_qd": particle_qd,
            "actions": self.actions.clone(),
        }

    def compute_reward(self):
        com_q, com_qd = self.obs_buf["com_q"], self.obs_buf["com_qd"]
        progress_rew = com_qd[:, 0]
        start_pos = self.start_pos - self.env_offsets
        start_pos = 0.1 * start_pos  # rescaling
        up_rew = com_q - start_pos
        up_rew = up_rew[:, 1]
        action_cost = -self.actions.pow(2).sum(1)
        rew = (
            10.0 * progress_rew
            + 30.0 * up_rew
            + 0.001 * action_cost
        )

        reset_buf, progress_buf = self.reset_buf, self.progress_buf
        max_episode_steps, early_termination = self.episode_length, self.early_termination
        truncated = progress_buf > max_episode_steps - 1
        reset = torch.where(truncated, torch.ones_like(reset_buf), reset_buf)
        if early_termination:
            raise NotImplementedError
        else:
            terminated = torch.where(torch.zeros_like(reset), torch.ones_like(reset), reset)
        self.rew_buf, self.reset_buf, self.terminated_buf, self.truncated_buf = rew, reset, terminated, truncated

    def run(self):
        train_iters = 30
        train_rate = 0.025

        tet_count = self.num_act
        phase_count, phase_step, phase_freq = (
            self.phase_count,
            self.phase_step,
            self.phase_freq,
        )

        phases = []
        for i in range(self.episode_length):
            phase = torch.zeros(phase_count, dtype=torch.float, device=self.device, requires_grad=True)
            phases.append(phase)

        # single layer linear network
        k = 1.0 / phase_count
        net = nn.Sequential(
            nn.Linear(phase_count, tet_count, bias=True),
            nn.Tanh(),
        )
        weights, bias = net[0].weight, net[0].bias
        rng = np.random.default_rng(42)
        _weights = rng.uniform(-np.sqrt(k), np.sqrt(k), (tet_count, phase_count))
        weights.data = torch.tensor(_weights, dtype=torch.float, device=self.device, requires_grad=True)
        # nn.init.uniform_(weights, -np.sqrt(k), np.sqrt(k))
        nn.init.zeros_(bias)

        net.to(self.device)
        net.train()
        opt = torch.optim.Adam(net.parameters(), lr=train_rate)
        print(net)

        for iter in range(train_iters):
            obs = self.reset(clear_grad=True)

            profiler = {}
            with wp.ScopedTimer("episode", detailed=False, print=False, active=True, dict=profiler):
                obses, actions, rewards, dones, infos = [obs], [], [], [], []
                for i in range(self.episode_length):
                    sim_time = i * self.frame_dt
                    phase_counts = torch.arange(
                        phase_count,
                        dtype=torch.float,
                        device=self.device,
                        requires_grad=True,
                    )
                    phases[i] = torch.sin(phase_freq * sim_time + phase_counts * phase_step)
                    phases[i] = phases[i].repeat(self.num_envs, 1)
                    action = net(phases[i])

                    obs, reward, done, info = self.step(action)

                    obses.append(obs)
                    actions.append(action)
                    rewards.append(reward)
                    dones.append(done)
                    infos.append(info)

                actions = torch.stack(actions)
                rewards = torch.stack(rewards)

                # loss = torch.stack([obs["loss"] for obs in obses[1:]])  # skip the first observation
                # loss = loss.sum(0)  # sum over time
                # # loss /= self.num_envs

                loss = -rewards.sum(0)

                # Optimization step
                opt.zero_grad()
                loss.sum().backward()
                opt.step()

                grad_norm = weights.grad.norm()

                print(f"Iter: {iter} Loss: {loss.tolist()}")
                print(f"Grad W: {grad_norm.item()}")
                print(
                    "Traj actions:",
                    actions.mean().item(),
                    actions.std().item(),
                    actions.min().item(),
                    actions.max().item(),
                )

            avg_time = np.array(profiler["episode"]).mean() / self.episode_length
            avg_steps_second = 1000.0 * float(self.num_envs) / avg_time
            total_time_second = np.array(profiler["episode"]).sum() / 1000.0

            print(
                f"num_envs: {self.num_envs} |",
                f"steps/second: {avg_steps_second:.4} |",
                f"milliseconds/step: {avg_time:.4f} |",
                f"total_seconds: {total_time_second:.4f} |",
            )
            print()

        if self.renderer is not None:
            self.renderer.save()

        return 1000.0 * float(self.num_envs) / avg_time


@torch.jit.script
def normalize(x, eps: float = 1e-9):
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)


if __name__ == "__main__":
    run_env(Jumper, no_grad=False)
