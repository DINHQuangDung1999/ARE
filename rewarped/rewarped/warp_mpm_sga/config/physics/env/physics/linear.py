from dataclasses import dataclass, field
from pathlib import Path
import math
from .base import BasePhysicsConfig


@dataclass(kw_only=True)
class LinearPhysicsConfig(BasePhysicsConfig, name='linear'):
    path: str = str(Path(__file__).parent.resolve() / 'templates' / 'linear.py')

    youngs_modulus_log: float = 10.0
    poissons_ratio_sigmoid: float = -1.0
    youngs_modulus: float = field(init=False)
    poissons_ratio: float = field(init=False)

    def __post_init__(self):
        sigmoid = lambda x: 1.0 / (1.0 + math.exp(-x))
        self.youngs_modulus = math.exp(self.youngs_modulus_log)
        self.poissons_ratio = sigmoid(self.poissons_ratio_sigmoid) * 0.49
