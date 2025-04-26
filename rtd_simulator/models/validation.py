"""
Validation framework for RTD model plugins.

This module provides tools for validating RTD model implementations,
ensuring they meet the required interface and behave correctly.
"""

from typing import Dict, Tuple, Type, Optional, List
import numpy as np
from numpy.typing import NDArray
from .base import RTDModel, ParameterRanges, SimulationResult
from .numerical import rk4_step, rk4_simulate

# Type aliases
FloatArray = NDArray[np.float64]
ValidationResult = Tuple[bool, List[str]]

class ModelValidator:
    """
    Validates RTD model implementations.
    
    This class provides methods to verify that a model implementation
    meets all requirements and behaves correctly.
    """
    
    def __init__(self, model_class: Type[RTDModel]):
        """
        Initialize the validator.
        
        Args:
            model_class: The RTD model class to validate
        """
        self.model_class = model_class
        self.model = model_class()  # Create instance with default parameters
        self.errors: List[str] = []
    
    def validate(self) -> ValidationResult:
        """
        Run all validation checks.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        self.errors = []
        
        # Run all validation checks
        self._check_required_methods()
        self._check_parameter_validation()
        self._check_iv_characteristic()
        self._check_simulation_methods()
        self._check_boundary_conditions()
        
        return len(self.errors) == 0, self.errors
    
    def _check_required_methods(self) -> None:
        """Check that all required methods are implemented."""
        required_methods = [
            'iv_characteristic',
            'step',
            'simulate',
            'simulate_vectorized',
            'get_parameter_ranges'
        ]
        
        for method in required_methods:
            if not hasattr(self.model, method):
                self.errors.append(f"Missing required method: {method}")
    
    def _check_parameter_validation(self) -> None:
        """Check parameter validation behavior."""
        try:
            # Get parameter ranges
            ranges = self.model.get_parameter_ranges()
            
            # Test parameter validation
            for param, (min_val, max_val) in ranges.items():
                # Test below minimum
                test_params = {param: min_val - 1}
                try:
                    self.model_class(**test_params)
                    self.errors.append(f"Parameter {param} validation failed: accepted value below minimum")
                except ValueError:
                    pass
                
                # Test above maximum
                test_params = {param: max_val + 1}
                try:
                    self.model_class(**test_params)
                    self.errors.append(f"Parameter {param} validation failed: accepted value above maximum")
                except ValueError:
                    pass
                
                # Test valid value
                test_params = {param: (min_val + max_val) / 2}
                try:
                    self.model_class(**test_params)
                except ValueError:
                    self.errors.append(f"Parameter {param} validation failed: rejected valid value")
                    
        except Exception as e:
            self.errors.append(f"Parameter validation check failed: {str(e)}")
    
    def _check_iv_characteristic(self) -> None:
        """Check IV characteristic calculation."""
        try:
            # Test with various voltage ranges
            test_voltages = [
                np.linspace(-3.0, 3.0, 100),  # Standard range
                np.array([0.0]),              # Single point
                np.linspace(-10.0, 10.0, 100) # Extended range
            ]
            
            for voltages in test_voltages:
                try:
                    currents = self.model.iv_characteristic(voltages)
                    
                    # Check output type and shape
                    if not isinstance(currents, np.ndarray):
                        self.errors.append("IV characteristic must return numpy array")
                    if currents.shape != voltages.shape:
                        self.errors.append("IV characteristic output shape must match input shape")
                    
                    # Check for NaN or Inf
                    if np.any(np.isnan(currents)) or np.any(np.isinf(currents)):
                        self.errors.append("IV characteristic produced NaN or Inf values")
                        
                except Exception as e:
                    self.errors.append(f"IV characteristic calculation failed: {str(e)}")
                    
        except Exception as e:
            self.errors.append(f"IV characteristic check failed: {str(e)}")
    
    def _check_simulation_methods(self) -> None:
        """Check simulation methods."""
        try:
            # Test parameters
            t_end = 1.0
            dt = 0.01
            vbias = 1.0
            v0 = 0.0
            i0 = 0.0
            
            # Test step method
            try:
                v, i = self.model.step(dt, vbias, v0, i0)
                if not isinstance(v, float) or not isinstance(i, float):
                    self.errors.append("step method must return float values")
            except Exception as e:
                self.errors.append(f"step method failed: {str(e)}")
            
            # Test simulate method
            try:
                t, v, i = self.model.simulate(t_end, dt, vbias, v0, i0)
                if not isinstance(t, np.ndarray) or not isinstance(v, np.ndarray) or not isinstance(i, np.ndarray):
                    self.errors.append("simulate method must return numpy arrays")
                if len(t) != len(v) or len(t) != len(i):
                    self.errors.append("simulate method output arrays must have same length")
            except Exception as e:
                self.errors.append(f"simulate method failed: {str(e)}")
            
            # Test simulate_vectorized method
            try:
                t, v, i = self.model.simulate_vectorized(t_end, dt, vbias, v0, i0)
                if not isinstance(t, np.ndarray) or not isinstance(v, np.ndarray) or not isinstance(i, np.ndarray):
                    self.errors.append("simulate_vectorized method must return numpy arrays")
                if len(t) != len(v) or len(t) != len(i):
                    self.errors.append("simulate_vectorized method output arrays must have same length")
            except Exception as e:
                self.errors.append(f"simulate_vectorized method failed: {str(e)}")
                
        except Exception as e:
            self.errors.append(f"Simulation methods check failed: {str(e)}")
    
    def _check_boundary_conditions(self) -> None:
        """Check behavior at boundary conditions."""
        try:
            # Test extreme parameter values
            ranges = self.model.get_parameter_ranges()
            for param, (min_val, max_val) in ranges.items():
                # Test at minimum
                test_params = {param: min_val}
                try:
                    model = self.model_class(**test_params)
                    # Run short simulation
                    model.simulate(0.1, 0.01, 1.0, 0.0, 0.0)
                except Exception as e:
                    self.errors.append(f"Model failed at minimum {param}: {str(e)}")
                
                # Test at maximum
                test_params = {param: max_val}
                try:
                    model = self.model_class(**test_params)
                    # Run short simulation
                    model.simulate(0.1, 0.01, 1.0, 0.0, 0.0)
                except Exception as e:
                    self.errors.append(f"Model failed at maximum {param}: {str(e)}")
                    
        except Exception as e:
            self.errors.append(f"Boundary conditions check failed: {str(e)}")

def validate_model(model_class: Type[RTDModel]) -> ValidationResult:
    """
    Validate an RTD model implementation.
    
    Args:
        model_class: The RTD model class to validate
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    validator = ModelValidator(model_class)
    return validator.validate() 