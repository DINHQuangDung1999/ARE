from dataclasses import dataclass, field
from pathlib import Path
import math
from .base import BasePhysicsConfig


@dataclass(kw_only=True)
class PlasticinePhysicsConfig(BasePhysicsConfig, name='plasticine'):
    path: str = str(Path(__file__).parent.resolve() / 'templates' / 'plasticine.py')
    material: str = 'plasticine'
    elasticity: str = 'sigma'

    # # NClaw
    # E: float = 3e5
    # nu: float = 0.25
    # yield_stress: float = 5e3

    youngs_modulus: float = field(init=False)
    youngs_modulus_log: float = 13.0
    poissons_ratio: float = 0.25
    yield_stress: float = 3e4

    def __post_init__(self):
        self.youngs_modulus = math.exp(self.youngs_modulus_log)
