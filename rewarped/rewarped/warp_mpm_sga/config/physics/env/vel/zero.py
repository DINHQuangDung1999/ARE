from dataclasses import dataclass
from .base import BaseVelConfig

@dataclass(kw_only=True)
class ZeroVelConfig(BaseVelConfig, name='zero'):
    random: bool = False
    lin_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ang_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
