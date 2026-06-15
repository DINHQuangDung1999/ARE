import torch

import warp as wp

from .warp_mpm_sga.config import EvalConfig
from .warp_mpm_sga.sim import MPMInitData, MPMModelBuilder, MPMStateInitializer, MPMStaticsInitializer
from .warp_mpm_sga.warp import replace_torch_cbrt, replace_torch_polar, replace_torch_svd, replace_torch_trace


class MPMWarpEnvMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.up_axis == "Y"

    def create_cfg_mpm(self, mpm_cfg):
        return mpm_cfg

    def create_modelbuilder_mpm(self, builder):
        mpm_builder = MPMModelBuilder()
        self.mpm_cfg = self.create_cfg_mpm(EvalConfig(path=None))
        mpm_builder.parse_cfg(self.mpm_cfg.physics.sim, num_envs=self.num_envs, env_offsets=self.env_offsets)
        mpm_builder.model = mpm_builder.finalize(self.device, requires_grad=self.requires_grad)
        mpm_builder.state_initializer = MPMStateInitializer(mpm_builder.model)
        mpm_builder.statics_initializer = MPMStaticsInitializer(mpm_builder.model)
        builder.mpm_builder = mpm_builder
        return mpm_builder

    def create_builder_mpm(self, builder):
        mpm_builder = self.create_modelbuilder_mpm(builder)
        for env_id, offset in enumerate(self.env_offsets):
            self.create_env_mpm(builder, mpm_builder, env_id, offset)
        return mpm_builder

    def create_env_mpm(self, builder, mpm_builder, env_id, offset):
        init_data = MPMInitData.get(self.mpm_cfg.physics.env)
        center = self.mpm_cfg.physics.env.shape.center + offset
        lin_vel = self.mpm_cfg.physics.env.vel.lin_vel
        ang_vel = self.mpm_cfg.physics.env.vel.ang_vel
        self.create_obj_mpm(mpm_builder, env_id, init_data, center, lin_vel, ang_vel)

    def create_obj_mpm(self, mpm_builder, env_id, init_data, center, lin_vel, ang_vel):
        init_data.set_env_id(env_id)
        init_data.set_center(center)
        init_data.set_lin_vel(lin_vel)
        init_data.set_ang_vel(ang_vel)
        mpm_builder.state_initializer.add_group(init_data)
        mpm_builder.statics_initializer.add_group(init_data)

    def create_model_mpm(self, model):
        model.mpm_model = self.builder.mpm_builder.model
        model.mpm_state, _ = self.builder.mpm_builder.state_initializer.finalize(requires_grad=self.requires_grad)
        model.mpm_model.statics = self.builder.mpm_builder.statics_initializer.finalize()
        return model

    def print_model_info(self):
        super().print_model_info()
        print("--- mpm ---")
        print("mpm_dt", self.model.mpm_model.constant.dt)
        print("mpm_particle_count", self.model.mpm_state.particle.x.shape[0])
        print("mpm_x", self.model.mpm_state.particle.x)
        print("mpm_v", self.model.mpm_state.particle.v)
        print("mpm_vol", self.model.mpm_model.statics.vol)
        print("mpm_rho", self.model.mpm_model.statics.rho)
        print("mpm_material", self.model.mpm_model.statics.material)
        print()

    def init_sim_mpm(self):
        wp.set_module_options({"fast_math": True})
        torch.backends.cudnn.benchmark = True
        replace_torch_svd()
        replace_torch_polar()
        replace_torch_trace()
        replace_torch_cbrt()

    def render_mpm(self, state=None):
        state = state or self.state_1

        # render mpm particles
        particle_q = state.mpm_x
        particle_q = particle_q.numpy()
        particle_radius = 7.5e-3
        particle_color = (0.875, 0.451, 1.0)  # 0xdf73ff
        self.renderer.render_points("particle_q", particle_q, radius=particle_radius, colors=particle_color)
