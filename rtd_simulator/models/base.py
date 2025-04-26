"""
Base classes and plugin registry for RTD models.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Union, Dict, Any
import numpy as np
from numpy.typing import NDArray
from enum import Enum

class PerturbationType(Enum):
    """Types of perturbations that can be applied to the RTD."""
    SQUARE = "square"
    SINE = "sine"
    TRIANGLE = "triangle"
    SAWTOOTH = "sawtooth"

class RTDModel(ABC):
    """
    Abstract base class for RTD models.
    All RTD models must implement these methods to ensure compatibility.
    """
    @abstractmethod
    def iv_characteristic(self, voltage: NDArray[np.float64]) -> NDArray[np.float64]:
        pass

    @abstractmethod
    def step(self, dt: float, vbias: float, perturbation: Optional[float] = None) -> Tuple[float, float]:
        pass

    @abstractmethod
    def simulate(self, t_end: float, dt: float, vbias: Union[float, NDArray[np.float64]]) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        pass

    @abstractmethod
    def get_parameter_ranges(self) -> Dict[str, Tuple[float, float]]:
        pass

    def create_perturbation(self, 
                          t: NDArray[np.float64],
                          perturbation_type: PerturbationType,
                          amplitude: float,
                          frequency: float = 1.0,
                          start_time: float = 0.0,
                          duration: Optional[float] = None) -> NDArray[np.float64]:
        """
        Create a perturbation signal of the specified type.
        """
        perturbation = np.zeros_like(t)
        mask = t >= start_time
        if duration is not None:
            mask = mask & (t <= start_time + duration)
        if perturbation_type == PerturbationType.SQUARE:
            perturbation[mask] = amplitude
        elif perturbation_type == PerturbationType.SINE:
            perturbation[mask] = amplitude * np.sin(2 * np.pi * frequency * (t[mask] - start_time))
        elif perturbation_type == PerturbationType.TRIANGLE:
            period = 1.0 / frequency
            phase = (t[mask] - start_time) % period
            perturbation[mask] = amplitude * (2 * np.abs(2 * phase / period - 1) - 1)
        elif perturbation_type == PerturbationType.SAWTOOTH:
            period = 1.0 / frequency
            phase = (t[mask] - start_time) % period
            perturbation[mask] = amplitude * (2 * phase / period - 1)
        return perturbation

# Plugin registry for RTD models
RTD_MODEL_REGISTRY: Dict[str, Any] = {}

def register_rtd_model(name: str, model_cls: Any) -> None:
    """Register an RTD model class for plugin/extensibility."""
    RTD_MODEL_REGISTRY[name] = model_cls

def get_rtd_model(name: str) -> Any:
    """Retrieve a registered RTD model class by name."""
    return RTD_MODEL_REGISTRY.get(name) 