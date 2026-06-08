from dataclasses import dataclass
from .base import BaseShapeConfig

@dataclass(kw_only=True)
class MeshShapeConfig(BaseShapeConfig, name='mesh'):
    filepath: str = None
    center: tuple[float, float, float] = (0.5, 0.5, 0.6)
    size: tuple[float, float, float] = (0.2, 0.2, 0.2)
    resolution: int = 40
    mode: str = 'uniform'
