"""
Base classes and interfaces for RTD models.

This module defines the abstract base class and interfaces for implementing
Resonant Tunneling Diode (RTD) models.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Type, TypeVar, Protocol, Optional, Callable, Union, List
import numpy as np
from numpy.typing import NDArray
from enum import Enum
from .cache import cached_iv_characteristic, iv_cache
from dataclasses import dataclass

# Type aliases
FloatArray = NDArray[np.float64]
ParameterRanges = Dict[str, Tuple[float, float]]
StabilityResult = Tuple[bool, str]
SimulationResult = Tuple[FloatArray, FloatArray, FloatArray]

@dataclass
class StabilityResult:
    """Result of stability check for RTD model."""
    is_stable: bool
    oscillation_amplitude: float
    max_voltage: float
    max_current: float
    error: Optional[str] = None

class RTDModelProtocol(Protocol):
    """Protocol for runtime type checking of RTD models."""
    def iv_characteristic(self, voltage: FloatArray) -> FloatArray: ...
    def step(self, dt: float, vbias: float, v: float, i: float) -> Tuple[float, float]: ...
    def simulate(self, t_end: float, dt: float, vbias: float, v0: float, i0: float) -> SimulationResult: ...
    def simulate_vectorized(self, t_end: float, dt: float, vbias: Union[float, FloatArray], v0: float, i0: float) -> SimulationResult: ...
    def get_parameter_ranges(self) -> ParameterRanges: ...

RTDModelT = TypeVar('RTDModelT', bound='RTDModel')

class RTDModel(ABC):
    """
    Abstract base class for RTD models.
    
    This class defines the interface that all RTD models must implement.
    It provides common functionality and enforces consistent behavior
    across different model implementations.
    """
    
    def __init__(self, **parameters):
        """
        Initialize the RTD model with given parameters.
        
        Args:
            **parameters: Model-specific parameters
        """
        self._parameters = parameters
        self._cache_enabled = True
    
    @property
    def parameters(self) -> Dict[str, float]:
        """Get the current model parameters."""
        return self._parameters.copy()
    
    @parameters.setter
    def parameters(self, new_parameters: Dict[str, float]) -> None:
        """Set new model parameters and clear the cache."""
        self._parameters = new_parameters.copy()
        # Clear cache for this model when parameters change
        iv_cache.clear(self.__class__.__name__)
    
    @abstractmethod
    def iv_characteristic(self, voltage: FloatArray) -> FloatArray:
        """
        Calculate the IV characteristic for given voltage values.
        
        Args:
            voltage: Array of voltage values
            
        Returns:
            Array of current values
        """
        pass
    
    def _cached_iv_characteristic(self, voltage: FloatArray) -> FloatArray:
        """
        Calculate or retrieve from cache the IV characteristic.
        
        Args:
            voltage: Array of voltage values
            
        Returns:
            Array of current values
        """
        if not self._cache_enabled:
            return self.iv_characteristic(voltage)
            
        # Convert voltage array to tuple for caching
        v_tuple = tuple(voltage)
        
        # Get cached result or calculate new
        current_tuple = cached_iv_characteristic(
            self.__class__.__name__,
            v_tuple,
            **self._parameters
        )
        
        return np.array(current_tuple, dtype=np.float64)
    
    def enable_cache(self) -> None:
        """Enable IV-curve caching."""
        self._cache_enabled = True
    
    def disable_cache(self) -> None:
        """Disable IV-curve caching."""
        self._cache_enabled = False
    
    @abstractmethod
    def step(self, dt: float, vbias: float, v: float, i: float) -> Tuple[float, float]:
        """
        Perform one integration step.
        
        Args:
            dt: Time step
            vbias: Bias voltage
            v: Current voltage
            i: Current current
            
        Returns:
            Tuple of (new voltage, new current)
        """
        pass
    
    @abstractmethod
    def simulate(self, t_end: float, dt: float, vbias: float, v0: float, i0: float) -> SimulationResult:
        """
        Run a complete simulation.
        
        Args:
            t_end: End time
            dt: Time step
            vbias: Bias voltage
            v0: Initial voltage
            i0: Initial current
            
        Returns:
            Tuple of (time array, voltage array, current array)
        """
        pass
    
    @abstractmethod
    def simulate_vectorized(self, t_end: float, dt: float, vbias: Union[float, FloatArray], v0: float, i0: float) -> SimulationResult:
        """
        Run a complete simulation using vectorized operations.
        
        Args:
            t_end: End time
            dt: Time step
            vbias: Bias voltage (can be a scalar or array)
            v0: Initial voltage
            i0: Initial current
            
        Returns:
            Tuple of (time array, voltage array, current array)
        """
        pass
    
    @abstractmethod
    def get_parameter_ranges(self) -> ParameterRanges:
        """
        Get valid parameter ranges.
        
        Returns:
            Dictionary mapping parameter names to (min, max) tuples
        """
        pass

# Model registry
_model_registry: Dict[str, Type[RTDModel]] = {}

def register_model(name: str) -> Callable[[Type[RTDModel]], Type[RTDModel]]:
    """
    Register a model class.
    
    Args:
        name: Name to register the model under
        
    Returns:
        Decorator function that registers the model class
    """
    def decorator(model_class: Type[RTDModel]) -> Type[RTDModel]:
        _model_registry[name] = model_class
        return model_class
    return decorator

def get_model(name: str) -> Type[RTDModel]:
    """
    Get a registered model class.
    
    Args:
        name: Name of the model
        
    Returns:
        The model class
        
    Raises:
        KeyError: If model is not registered
    """
    return _model_registry[name]

class PerturbationType(Enum):
    """Types of perturbations that can be applied to the RTD."""
    SQUARE = "square"
    SINE = "sine"
    TRIANGLE = "triangle"
    SAWTOOTH = "sawtooth"

def register_rtd_model(name: str, model_cls: Type[RTDModel]) -> None:
    """
    Register an RTD model class for plugin/extensibility.
    
    Args:
        name: Name to register the model under.
        model_cls: RTD model class to register.
    """
    _model_registry[name] = model_cls

def get_rtd_model(name: str) -> Optional[Type[RTDModel]]:
    """
    Retrieve a registered RTD model class by name.
    
    Args:
        name: Name of the model to retrieve.
        
    Returns:
        The registered model class, or None if not found.
    """
    return _model_registry.get(name)

def get_rtd_model_by_name(name: str) -> Optional[Type[RTDModel]]:
    """
    Retrieve a registered RTD model class by name.
    
    Args:
        name: Name of the model to retrieve.
        
    Returns:
        The registered model class, or None if not found.
    """
    return _model_registry.get(name)

def create_perturbation(
    t: NDArray[np.float64],
    perturbation_type: PerturbationType,
    amplitude: float,
    frequency: float,
    phase: float = 0.0
) -> NDArray[np.float64]:
    """
    Create a perturbation signal.
    
    Args:
        t: Time array
        perturbation_type: Type of perturbation
        amplitude: Amplitude of perturbation
        frequency: Frequency of perturbation
        phase: Phase offset
        
    Returns:
        Array of perturbation values
    """
    if perturbation_type == PerturbationType.SQUARE:
        return amplitude * np.sign(np.sin(2 * np.pi * frequency * t + phase))
    elif perturbation_type == PerturbationType.SINE:
        return amplitude * np.sin(2 * np.pi * frequency * t + phase)
    elif perturbation_type == PerturbationType.TRIANGLE:
        return amplitude * (2 * np.abs(2 * (frequency * t + phase) - np.floor(2 * (frequency * t + phase) + 0.5)) - 1)
    elif perturbation_type == PerturbationType.SAWTOOTH:
        return amplitude * (2 * (frequency * t + phase) - np.floor(2 * (frequency * t + phase) + 0.5))
    else:
        raise ValueError(f"Unknown perturbation type: {perturbation_type}") 