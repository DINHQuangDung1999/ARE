from dataclasses import dataclass, field
from .base import BaseEnvConfig
from .physics import CorotatedPhysicsConfig
from .shape import CubeShapeConfig
from .vel import ZeroVelConfig

@dataclass(kw_only=True)
class JellyEnvConfig(BaseEnvConfig, name='jelly'):
    physics: CorotatedPhysicsConfig = field(default_factory=CorotatedPhysicsConfig)
    shape: CubeShapeConfig = field(default_factory=CubeShapeConfig)
    vel: ZeroVelConfig = field(default_factory=ZeroVelConfig)

    rho: float = 1e3
