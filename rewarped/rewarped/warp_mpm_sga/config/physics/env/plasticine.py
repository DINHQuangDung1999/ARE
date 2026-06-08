from dataclasses import dataclass, field
from .base import BaseEnvConfig
from .physics import PlasticinePhysicsConfig
from .shape import CubeShapeConfig
from .vel import ZeroVelConfig

@dataclass(kw_only=True)
class PlasticineEnvConfig(BaseEnvConfig, name='plasticine'):
    physics: PlasticinePhysicsConfig = field(default_factory=PlasticinePhysicsConfig)
    shape: CubeShapeConfig = field(default_factory=CubeShapeConfig)
    vel: ZeroVelConfig = field(default_factory=ZeroVelConfig)

    rho: float = 1e3
