"""
IV-curve caching utilities for RTD models.

This module provides caching functionality for IV characteristic calculations
to improve simulation performance.
"""

from typing import Dict, Tuple, Optional, List, Callable
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from functools import lru_cache

@dataclass
class CacheEntry:
    """Represents a cached IV curve calculation."""
    voltage_range: NDArray[np.float64]
    current_values: NDArray[np.float64]
    parameters: Dict[str, float]
    timestamp: float

class IVCache:
    """
    LRU cache for IV characteristic calculations.
    
    This class provides caching functionality for IV curve calculations
    with automatic cache invalidation when model parameters change.
    """
    
    def __init__(self, max_size: int = 100):
        """
        Initialize the cache.
        
        Args:
            max_size: Maximum number of cache entries to keep
        """
        self.max_size = max_size
        self.cache: Dict[str, CacheEntry] = {}
        self.parameter_history: Dict[str, List[Dict[str, float]]] = {}
        
    def get(self, 
            model_name: str, 
            voltage_range: NDArray[np.float64],
            parameters: Dict[str, float]) -> Optional[NDArray[np.float64]]:
        """
        Get cached IV curve if available and valid.
        
        Args:
            model_name: Name of the RTD model
            voltage_range: Array of voltage values
            parameters: Current model parameters
            
        Returns:
            Cached current values if available and valid, None otherwise
        """
        cache_key = self._generate_cache_key(model_name, voltage_range, parameters)
        
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            if self._is_valid(entry, parameters):
                return entry.current_values
        
        return None
    
    def put(self,
            model_name: str,
            voltage_range: NDArray[np.float64],
            current_values: NDArray[np.float64],
            parameters: Dict[str, float]) -> None:
        """
        Store IV curve in cache.
        
        Args:
            model_name: Name of the RTD model
            voltage_range: Array of voltage values
            current_values: Array of current values
            parameters: Current model parameters
        """
        cache_key = self._generate_cache_key(model_name, voltage_range, parameters)
        
        # Update parameter history
        if model_name not in self.parameter_history:
            self.parameter_history[model_name] = []
        self.parameter_history[model_name].append(parameters)
        
        # Create new cache entry
        entry = CacheEntry(
            voltage_range=voltage_range.copy(),
            current_values=current_values.copy(),
            parameters=parameters.copy(),
            timestamp=float(len(self.parameter_history[model_name]))
        )
        
        # Update cache
        self.cache[cache_key] = entry
        
        # Remove oldest entry if cache is full
        if len(self.cache) > self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k].timestamp)
            del self.cache[oldest_key]
    
    def clear(self, model_name: Optional[str] = None) -> None:
        """
        Clear the cache for a specific model or all models.
        
        Args:
            model_name: Name of the model to clear cache for, or None to clear all
        """
        if model_name is None:
            self.cache.clear()
            self.parameter_history.clear()
        else:
            # Remove all entries for the specified model
            keys_to_remove = [k for k in self.cache.keys() if k.startswith(f"{model_name}_")]
            for key in keys_to_remove:
                del self.cache[key]
            if model_name in self.parameter_history:
                del self.parameter_history[model_name]
    
    def _generate_cache_key(self,
                          model_name: str,
                          voltage_range: NDArray[np.float64],
                          parameters: Dict[str, float]) -> str:
        """Generate a unique cache key for the given parameters."""
        # Create a hashable representation of the voltage range
        v_hash = hash(voltage_range.tobytes())
        # Create a hashable representation of the parameters
        p_hash = hash(frozenset(parameters.items()))
        return f"{model_name}_{v_hash}_{p_hash}"
    
    def _is_valid(self, entry: CacheEntry, current_parameters: Dict[str, float]) -> bool:
        """Check if a cache entry is still valid."""
        return entry.parameters == current_parameters

# Global cache instance
iv_cache = IVCache()

@lru_cache(maxsize=100)
def cached_iv_characteristic(model_name: str,
                           voltage_range: Tuple[float, ...],
                           **parameters) -> Tuple[float, ...]:
    """
    Calculate and cache IV characteristic.
    
    Args:
        model_name: Name of the RTD model
        voltage_range: Tuple of voltage values
        **parameters: Model parameters
        
    Returns:
        Tuple of current values
    """
    # Convert voltage range to numpy array
    v_array = np.array(voltage_range, dtype=np.float64)
    
    # Check cache
    cached_result = iv_cache.get(model_name, v_array, parameters)
    if cached_result is not None:
        return tuple(cached_result)
    
    # Calculate IV characteristic based on model type
    if model_name == "SimplifiedRTDModel":
        k = parameters.get('k', 5.92)
        h = parameters.get('h', 15.76)
        w = parameters.get('w', 2.259)
        current_values = k * v_array - h * np.arctan(v_array / w)
    elif model_name == "SchulmanRTDModel":
        a = parameters.get('a', 1.0e-3)
        b = parameters.get('b', 0.1)
        c = parameters.get('c', 0.2)
        d = parameters.get('d', 0.01)
        h = parameters.get('h', 1.0e-6)
        n1 = parameters.get('n1', 1.0)
        n2 = parameters.get('n2', 1.0)
        T = parameters.get('T', 300.0)
        k_b = 1.380649e-23  # Boltzmann constant [J/K]
        qe = 1.602176634e-19  # Electron charge [C]
        
        # Calculate thermal voltage
        VT = k_b * T / qe
        
        # Calculate exponential terms with overflow protection
        exp_arg_up = (b - c + n1 * v_array) / VT
        exp_arg_down = (b - c - n1 * v_array) / VT
        
        # Clip arguments to avoid overflow
        exp_arg_up = np.minimum(exp_arg_up, 700.0)  # np.exp(700) is near float64 max
        exp_arg_down = np.minimum(exp_arg_down, 700.0)
        
        term_up = np.exp(exp_arg_up)
        term_down = np.exp(exp_arg_down)
        
        # Calculate components with protection against division by zero
        eps = 1e-30  # Small constant to prevent division by zero
        J1 = a * np.log((1.0 + term_up) / (1.0 + term_down + eps))
        J2 = np.pi/2 + np.arctan((c - n1 * v_array) / (d + eps))
        
        # Protect against overflow in the final exponential
        exp_arg_final = n2 * v_array / VT
        exp_arg_final = np.minimum(exp_arg_final, 700.0)
        J3 = h * (np.exp(exp_arg_final) - 1.0)
        
        current_values = J1 * J2 + J3
    else:
        raise ValueError(f"Unknown model type: {model_name}")
    
    # Store in cache
    iv_cache.put(model_name, v_array, current_values, parameters)
    
    return tuple(current_values) 