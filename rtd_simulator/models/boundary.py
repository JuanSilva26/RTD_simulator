"""
Simulation boundary validation for RTD models.

This module provides tools for validating simulation parameters
and detecting stability boundaries in RTD models.
"""

from typing import Tuple, Optional, Dict
import numpy as np
from numpy.typing import NDArray
from .base import RTDModel, FloatArray, StabilityResult

class BoundaryValidator:
    """
    Validates simulation boundaries and detects stability limits.
    
    This class provides methods to check if simulation parameters
    are within safe operating ranges and to detect stability boundaries.
    """
    
    def __init__(self, model: RTDModel):
        """
        Initialize the validator.
        
        Args:
            model: The RTD model to validate
        """
        self.model = model
        self._default_voltage_range = (-3.0, 3.0)  # Default safe voltage range
        self._default_current_limit = 1.0  # Default current limit in mA
    
    def validate_voltage_range(self, v_min: float, v_max: float) -> Tuple[bool, str]:
        """
        Validate voltage range for simulation.
        
        Args:
            v_min: Minimum voltage
            v_max: Maximum voltage
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if v_min >= v_max:
            return False, "Minimum voltage must be less than maximum voltage"
            
        safe_min, safe_max = self._default_voltage_range
        if v_min < safe_min or v_max > safe_max:
            return False, f"Voltage range ({v_min}, {v_max}) exceeds safe limits ({safe_min}, {safe_max})"
            
        return True, ""
    
    def validate_current_limit(self, i_max: float) -> Tuple[bool, str]:
        """
        Validate maximum current limit.
        
        Args:
            i_max: Maximum allowed current in mA
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if i_max <= 0:
            return False, "Current limit must be positive"
            
        if i_max > self._default_current_limit:
            return False, f"Current limit {i_max} mA exceeds safe limit {self._default_current_limit} mA"
            
        return True, ""
    
    def check_stability_boundary(self, 
                               vbias: float,
                               v0: float = 0.0,
                               i0: float = 0.0,
                               dt: float = 0.01,
                               steps: int = 1000) -> StabilityResult:
        """
        Check if simulation parameters are near stability boundary.
        
        Args:
            vbias: Bias voltage
            v0: Initial voltage
            i0: Initial current
            dt: Time step
            steps: Number of steps to simulate
            
        Returns:
            StabilityResult indicating if parameters are stable
        """
        try:
            # Run short simulation
            t, v, i = self.model.simulate(dt * steps, dt, vbias, v0, i0)
            
            # Check for oscillations
            v_std = float(np.std(v[-100:]))  # Check last 100 points
            i_std = float(np.std(i[-100:]))
            
            # Check for divergence
            v_max = float(np.max(np.abs(v)))
            i_max = float(np.max(np.abs(i)))
            
            # Define stability criteria
            is_stable = (v_std < 0.1 and  # Small oscillations
                        i_std < 0.1 and
                        v_max < 10.0 and  # Reasonable voltage
                        i_max < 10.0)     # Reasonable current
            
            return StabilityResult(
                is_stable=is_stable,
                oscillation_amplitude=max(v_std, i_std),
                max_voltage=v_max,
                max_current=i_max
            )
            
        except Exception as e:
            return StabilityResult(
                is_stable=False,
                oscillation_amplitude=float('inf'),
                max_voltage=float('inf'),
                max_current=float('inf'),
                error=str(e)
            )
    
    def find_stability_boundary(self,
                              v_start: float,
                              v_end: float,
                              v_step: float = 0.1,
                              **kwargs) -> Dict[float, StabilityResult]:
        """
        Find stability boundary by sweeping voltage.
        
        Args:
            v_start: Starting voltage
            v_end: Ending voltage
            v_step: Voltage step size
            **kwargs: Additional arguments for check_stability_boundary
            
        Returns:
            Dictionary mapping voltages to stability results
        """
        results = {}
        voltages = np.arange(v_start, v_end + v_step, v_step)
        
        for vbias in voltages:
            results[vbias] = self.check_stability_boundary(vbias, **kwargs)
            
        return results

def validate_simulation_parameters(model: RTDModel,
                                v_min: float,
                                v_max: float,
                                i_max: float) -> Tuple[bool, str]:
    """
    Validate simulation parameters for an RTD model.
    
    Args:
        model: The RTD model
        v_min: Minimum voltage
        v_max: Maximum voltage
        i_max: Maximum current limit
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    validator = BoundaryValidator(model)
    
    # Check voltage range
    is_valid, msg = validator.validate_voltage_range(v_min, v_max)
    if not is_valid:
        return False, msg
        
    # Check current limit
    is_valid, msg = validator.validate_current_limit(i_max)
    if not is_valid:
        return False, msg
        
    return True, "" 