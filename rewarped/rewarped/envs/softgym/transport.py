import numpy as np
import torch
from gym import spaces

import warp as wp

from ...environment import IntegratorType, run_env
from ...mpm_warp_env_mixin import MPMWarpEnvMixin
from ...warp_env import WarpEnv


class Transport(MPMWarpEnvMixin, WarpEnv):
    sim_name = "Transport" + "SoftGym"
    env_offset = (1.5, 0.0, 1.5)
    env_offset_correction = False

    integrator_type = IntegratorType.MPM
    sim_substeps_mpm = 40

    eval_fk = True
    eval_kinematic_fk = True
    eval_ik = False

    up_axis = "Y"
    ground_plane = True

    state_tensors_names = ("joint_q", "body_q") + ("mpm_x", "mpm_v", "mpm_C", "mpm_F_trial", "mpm_F", "mpm_stress")
    control_tensors_names = ("joint_act",)

    def __init__(self, num_envs=2, episode_length=300, early_termination=False, **kwargs):
        num_obs = 0
        num_act = 3
        super().__init__(num_envs, num_obs, num_act, episode_length, early_termination, **kwargs)

        self.action_scale = 0.01
        self.downsample_particle = 250

        self.box_p = (0.5, 0.15, 0.5)
        self.box_size = (0.18, 0.16, 0.22)  # l, h, w
        self.box_thickness = 0.05

    @property
    def observation_space(self):
        d = {
            "particle_q": spaces.Box(low=-np.inf, high=np.inf, shape=(self.downsample_particle, 3), dtype=np.float32),
            "com_q": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "joint_q": spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_act,), dtype=np.float32),
            "target_q": spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_act,), dtype=np.float32),
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
        self.create_box(builder)

    def create_box(self, builder):
        l, h, w = self.box_size
        t = self.box_thickness

        boxes = [
            # bottom
            [l / 2 + t, t / 2, w / 2 + t],
            # left
            [t / 2, (h + t) / 2, w / 2 + t],
            # right
            [t / 2, (h + t) / 2, w / 2 + t],
            # back
            [l / 2 + t, (h + t) / 2, t / 2],
            # front
            [l / 2 + t, (h + t) / 2, t / 2],
        ]
        positions = [
            [0.0, -(h + t) / 2, 0.0],
            [-(l + t) / 2, -t / 2, 0.0],
            [(l + t) / 2, -t / 2, 0.0],
            [0.0, -t / 2, -(w + t) / 2],
            [0.0, -t / 2, (w + t) / 2],
        ]

        builder.add_articulation()
        b = builder.add_body(
            # origin=wp.transform((0.0, 0.0, 0.0), wp.quat_identity()),
            name="body_box",
        )

        for box, pos in zip(boxes, positions):
            hx, hy, hz = box
            builder.add_shape_box(
                body=b,
                pos=wp.vec3(pos),
                hx=hx,
                hy=hy,
                hz=hz,
                density=1000.0,
                has_ground_collision=True,
                has_shape_collision=True,
            )

        # prevent hitting mpm env bounds
        lx, hx = 0.0 + self.box_size[0], 1.0 - self.box_size[0]
        ly, hy = 0.0 + self.box_size[1], 1.0 - self.box_size[1]
        lz, hz = 0.0 + self.box_size[2], 1.0 - self.box_size[2]
        axes = [
            wp.sim.JointAxis([1.0, 0.0, 0.0], limit_lower=lx, limit_upper=hx),
            wp.sim.JointAxis([0.0, 1.0, 0.0], limit_lower=ly, limit_upper=hy),
            wp.sim.JointAxis([0.0, 0.0, 1.0], limit_lower=lz, limit_upper=hz),
        ]
        builder.add_joint_d6(
            parent=-1,
            child=b,
            linear_axes=axes,
            name="joint_box",
        )
        builder.joint_q[0:3] = [*self.box_p]

    def create_cfg_mpm(self, mpm_cfg):
        mpm_cfg.update(["--physics.sim", "regular"])
        mpm_cfg.update(["--physics.env", "water"])
        # mpm_cfg.update(["--physics.env", "sand"])

        num_grids = 48
        center = self.box_p
        margin = 0.02
        size = [self.box_size[0] - margin, (self.box_size[1] - self.box_thickness) * 0.85, self.box_size[2] - margin]
        size = [np.round(x, 3) for x in size]
        size = np.array(size).tolist()
        resolution = 20

        # num_grids = 96
        # d = 0.04, 0.2, 0.04
        # center = [0.5 + d[0] / 2.0, 0.15, 0.5 + d[2] / 2.0]
        # size = [0.2 - d[0], 0.08, 0.3 - d[2]]
        # resolution = 16

        mpm_cfg.update(["--physics.sim.num_grids", str(num_grids)])
        mpm_cfg.update(["--physics.env.shape.center", str(tuple(center))])
        mpm_cfg.update(["--physics.env.shape.size", str(tuple(size))])
        mpm_cfg.update(["--physics.env.shape.resolution", str(resolution)])
        mpm_cfg.update(["--physics.env.rho", str(1e3)])
        mpm_cfg.update(["--physics.sim.ground_friction", str(0.0)])

        print(mpm_cfg)
        return mpm_cfg

    def create_model(self):
        model = super().create_model()
        self.create_model_mpm(model)
        return model

    def init_sim(self):
        self.init_sim_mpm()
        super().init_sim()
        self.print_model_info()

        with torch.no_grad():
            self.joint_act = wp.to_torch(self.model.joint_act).view(self.num_envs, -1).clone()
            self.joint_act_indices = ...
            # joint_act_indices = [0, 2]
            # joint_act_indices = torch.tensor(joint_act_indices, device=self.device)
            # self.joint_act_indices = joint_act_indices.unsqueeze(0).expand(self.num_envs, -1)

            self.target_q = wp.to_torch(self.model.joint_q).view(self.num_envs, -1).clone()

            self.joint_limit_lower = wp.to_torch(self.model.joint_limit_lower).view(self.num_envs, -1).clone()
            self.joint_limit_upper = wp.to_torch(self.model.joint_limit_upper).view(self.num_envs, -1).clone()

    @torch.no_grad()
    def randomize_init(self, env_ids):
        joint_q = self.state.joint_q.clone().view(self.num_envs, -1)

        bounds = torch.tensor([0.2, 0.3, 0.2], device=self.device)
        rand_pos = (torch.rand(size=(len(env_ids), 3), device=self.device) - 0.5) * 2.0
        rand_pos[:, 1] = rand_pos[:, 1] / 2.0 + 0.5  # +height only
        rand_pos = torch.round(rand_pos, decimals=3)
        self.target_q = self.target_q.clone()
        self.target_q[env_ids, :] = joint_q[env_ids, :].clone()
        self.target_q[env_ids, :] += bounds * rand_pos

    def pre_physics_step(self, actions):
        actions = actions.view(self.num_envs, -1)
        actions = torch.clip(actions, -1.0, 1.0)
        self.actions = actions
        acts = self.action_scale * actions

        if self.eval_kinematic_fk:
            # ensure joint limit
            joint_q = self.state.joint_q.clone().view(self.num_envs, -1)
            joint_q = joint_q.detach()
            l, h = self.joint_limit_lower - joint_q, self.joint_limit_upper - joint_q
            if self.joint_act_indices is not ...:
                l, h = torch.gather(l, 1, self.joint_act_indices), torch.gather(h, 1, self.joint_act_indices)
            acts = torch.clip(acts, l, h)

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
            "target_q": self.target_q.clone(),
            "particle_q": particle_q,
            "com_q": com_q,
        }

    def compute_reward(self):
        joint_q = self.obs_buf["joint_q"]
        target_q = self.obs_buf["target_q"]
        particle_q = self.obs_buf["particle_q"]
        # com_q = self.obs_buf["com_q"]

        rew = compute_reward(joint_q, target_q, particle_q, self.box_size, self.box_thickness)

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
                self.renderer.render(state or self.state_0)
                self.render_mpm(state)

                # render target
                target_q = self.target_q + self.env_offsets
                target_q = target_q.cpu().numpy()
                particle_radius = 0.01
                particle_color = (1.0, 0.0, 0.0)
                self.renderer.render_points("target_q", target_q, radius=particle_radius, colors=particle_color)

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


@torch.jit.script
def sigmoid_compare(a, b, eps: float = 1e-5, side: str = ">"):
    d = a - b
    if side == ">":
        out = torch.sigmoid(d / eps)
    elif side == "<":
        out = 1.0 - torch.sigmoid(d / eps)
    else:
        raise ValueError
    return out


@torch.jit.script
def compute_reward(joint_q, target_q, particle_q, box_size: tuple[float, float, float], box_thickness: float):
    # to target
    d = torch.norm(joint_q - target_q, p=2, dim=-1)
    dist_reward = 1.0 / (1.0 + d.square())
    dist_reward = dist_reward.square()
    dist_reward = torch.where(d <= 0.02, dist_reward * 2, dist_reward)

    # outside_box
    joint_q = joint_q.unsqueeze(1)
    l, h, w = box_size
    t = box_thickness
    bounds = [l / 2 + t, h / 2 + t / 2, w / 2 + t]
    bounds = torch.tensor([bounds], device=joint_q.device)
    bounds = bounds.unsqueeze(0)

    # # NOTE: not differentiable
    # out = torch.abs(particle_q - joint_q) > bounds
    # out = out.to(dtype=torch.float)
    # out_particles = out.max(dim=-1).values

    diff = torch.abs(particle_q - joint_q)
    out = sigmoid_compare(diff, bounds)
    out_particles = out.max(dim=-1).values

    spill_cost = -1.0 * torch.mean(out_particles, dim=-1)

    rew = 0.1 * dist_reward + 1.0 * spill_cost

    return rew


if __name__ == "__main__":
    run_env(Transport, no_grad=False)
