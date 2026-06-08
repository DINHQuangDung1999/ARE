from dataclasses import dataclass
from pathlib import Path
from .base import BasePhysicsConfig


@dataclass(kw_only=True)
class PlbPlasticinePhysicsConfig(BasePhysicsConfig, name='plb_plasticine'):
    path: str = str(Path(__file__).parent.resolve() / 'templates' / 'plasticine.py')
    material: str = 'plb_plasticine'

    # PlasticineLab
    E: float = 5e3
    nu: float = 0.2
    yield_stress: float = 50.0
