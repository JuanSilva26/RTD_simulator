# RTD Model Plugin Development Guide

## Overview

The RTD Simulator supports a plugin system that allows developers to create and integrate new RTD models. This guide explains how to create, validate, and register custom RTD models.

## Model Interface

All RTD models must implement the `RTDModel` abstract base class, which defines the following required methods:

```python
class RTDModel(ABC):
    @abstractmethod
    def iv_characteristic(self, voltage: FloatArray) -> FloatArray:
        """Calculate the IV characteristic for given voltage values."""
        pass

    @abstractmethod
    def step(self, dt: float, vbias: float, v: float, i: float) -> Tuple[float, float]:
        """Perform one integration step."""
        pass

    @abstractmethod
    def simulate(self, t_end: float, dt: float, vbias: float, v0: float, i0: float) -> SimulationResult:
        """Run a complete simulation."""
        pass

    @abstractmethod
    def simulate_vectorized(self, t_end: float, dt: float, vbias: Union[float, FloatArray], v0: float, i0: float) -> SimulationResult:
        """Run a complete simulation using vectorized operations."""
        pass

    @abstractmethod
    def get_parameter_ranges(self) -> ParameterRanges:
        """Get valid parameter ranges."""
        pass
```

## Creating a New Model

1. **Define Model Parameters**

   - Declare all model parameters in the constructor
   - Implement parameter validation in `_validate_parameters`
   - Define valid parameter ranges in `get_parameter_ranges`

2. **Implement IV Characteristic**

   - Override `iv_characteristic` to calculate current for given voltages
   - Consider using the built-in caching system for performance

3. **Implement Simulation Methods**
   - Override `step` for single-step integration
   - Override `simulate` for complete simulations
   - Override `simulate_vectorized` for optimized performance

## Example Model

```python
from rtd_simulator.models.base import RTDModel, register_model
import numpy as np
from numpy.typing import NDArray

@register_model("MyModel")
class MyRTDModel(RTDModel):
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        self._validate_parameters(alpha, beta)
        super().__init__(alpha=alpha, beta=beta)

    def _validate_parameters(self, alpha: float, beta: float) -> None:
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if beta <= 0:
            raise ValueError("beta must be positive")

    def get_parameter_ranges(self) -> ParameterRanges:
        return {
            'alpha': (0.1, 10.0),
            'beta': (0.1, 10.0)
        }

    def iv_characteristic(self, voltage: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.alpha * np.tanh(self.beta * voltage)

    def step(self, dt: float, vbias: float, v: float, i: float) -> Tuple[float, float]:
        # Implement single-step integration
        pass

    def simulate(self, t_end: float, dt: float, vbias: float, v0: float, i0: float) -> SimulationResult:
        # Implement complete simulation
        pass

    def simulate_vectorized(self, t_end: float, dt: float, vbias: Union[float, NDArray[np.float64]], v0: float, i0: float) -> SimulationResult:
        # Implement vectorized simulation
        pass
```

## Model Registration

Models can be registered in two ways:

1. **Using the Decorator**

   ```python
   @register_model("ModelName")
   class MyModel(RTDModel):
       pass
   ```

2. **Manual Registration**
   ```python
   register_rtd_model("ModelName", MyModel)
   ```

## Best Practices

1. **Parameter Validation**

   - Always validate parameters in `_validate_parameters`
   - Define reasonable parameter ranges in `get_parameter_ranges`
   - Use type hints for all methods and parameters

2. **Performance Optimization**

   - Use vectorized operations where possible
   - Implement `simulate_vectorized` for better performance
   - Leverage the built-in caching system

3. **Error Handling**

   - Raise descriptive exceptions for invalid parameters
   - Handle numerical instabilities gracefully
   - Document expected parameter ranges and behaviors

4. **Testing**
   - Write unit tests for all model methods
   - Test parameter validation and error cases
   - Verify simulation results against known solutions

## Using Registered Models

Registered models can be retrieved using:

```python
from rtd_simulator.models import get_rtd_model

# Get model class
model_class = get_rtd_model("ModelName")

# Create instance
model = model_class(alpha=1.0, beta=1.0)
```

## Troubleshooting

Common issues and solutions:

1. **Model Not Found**

   - Verify model registration
   - Check for typos in model name
   - Ensure model is imported before use

2. **Parameter Validation Errors**

   - Check parameter ranges
   - Verify parameter types
   - Ensure all required parameters are provided

3. **Performance Issues**
   - Use vectorized operations
   - Enable caching
   - Profile simulation methods

## Next Steps

1. Implement model validation
2. Add simulation boundary checks
3. Create example plugins
4. Add performance benchmarks
