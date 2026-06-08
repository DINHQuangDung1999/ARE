from dataclasses import dataclass, field
from .base import BaseEnvConfig
from .physics import SandPhysicsConfig
from .shape import CubeShapeConfig
from .vel import ZeroVelConfig

@dataclass(kw_only=True)
class SandEnvConfig(BaseEnvConfig, name='sand'):
    physics: SandPhysicsConfig = field(default_factory=SandPhysicsConfig)
    shape: CubeShapeConfig = field(default_factory=CubeShapeConfig)
    vel: ZeroVelConfig = field(default_factory=ZeroVelConfig)

    rho: float = 1e3
