"""
RTD Simulator package.

A simulator for Resonant Tunneling Diode (RTD) devices, providing both
physical modeling and visualization capabilities.
"""

from .models import (
    RTDModel,
    ConcreteRTDModel,
    PerturbationType,
    get_rtd_model,
    register_rtd_model,
    CurveAnalyzer,
)

__version__ = "0.1.0"
__all__ = [
    'RTDModel',
    'ConcreteRTDModel',
    'PerturbationType',
    'get_rtd_model',
    'register_rtd_model',
    'CurveAnalyzer',
] 