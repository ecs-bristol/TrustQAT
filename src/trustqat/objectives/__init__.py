"""Registry of trustworthiness objectives."""

from .base import TrustObjective, get_objective, list_objectives, objectives_from_config, register_objective
from .soft_ece import SoftECEObjective

__all__ = [
    'SoftECEObjective',
    'TrustObjective',
    'get_objective',
    'list_objectives',
    'objectives_from_config',
    'register_objective',
]
