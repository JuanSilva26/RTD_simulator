"""
Basic RTD model example.

This module provides a simple RTD model implementation that demonstrates
the plugin system and follows all required interfaces.
"""

from typing import Dict, Tuple
import numpy as np
from numpy.typing import NDArray
from ..base import RTDModel, register_model, FloatArray, ParameterRanges

@register_model("basic_rtd")
class BasicRTDModel(RTDModel):
    """
    A basic RTD model implementation.
    
    This model implements a simple RTD with:
    - Linear resistance in series with a tunnel diode
    - Basic NDR (Negative Differential Resistance) region
    - Temperature-dependent parameters
    """
    
    def __init__(self,
                 r_series: float = 1.0,      # Series resistance (ohms)
                 v_peak: float = 0.5,        # Peak voltage (V)
                 i_peak: float = 1.0,        # Peak current (mA)
                 v_valley: float = 1.0,      # Valley voltage (V)
                 i_valley: float = 0.1,      # Valley current (mA)
                 temp: float = 300.0):       # Temperature (K)
        """
        Initialize the basic RTD model.
        
        Args:
            r_series: Series resistance in ohms
            v_peak: Peak voltage in volts
            i_peak: Peak current in milliamps
            v_valley: Valley voltage in volts
            i_valley: Valley current in milliamps
            temp: Temperature in kelvin
        """
        super().__init__()
        self.r_series = r_series
        self.v_peak = v_peak
        self.i_peak = i_peak
        self.v_valley = v_valley
        self.i_valley = i_valley
        self.temp = temp
    
    def get_parameter_ranges(self) -> ParameterRanges:
        """
        Get valid parameter ranges.
        
        Returns:
            Dictionary mapping parameter names to (min, max) tuples
        """
        return {
            'r_series': (0.1, 10.0),     # Reasonable series resistance
            'v_peak': (0.1, 1.0),        # Typical peak voltage
            'i_peak': (0.1, 10.0),       # Typical peak current
            'v_valley': (0.5, 2.0),      # Typical valley voltage
            'i_valley': (0.01, 1.0),     # Typical valley current
            'temp': (77.0, 400.0)        # Cryogenic to elevated temp
        }
    
    def iv_characteristic(self, voltage: FloatArray) -> FloatArray:
        """
        Calculate the IV characteristic.
        
        Args:
            voltage: Array of voltage values
            
        Returns:
            Array of current values
        """
        # Convert to numpy array if needed
        v = np.asarray(voltage)
        
        # Calculate current through series resistance
        i_series = v / self.r_series
        
        # Calculate tunnel diode current
        # Simple piecewise linear approximation
        i_tunnel = np.zeros_like(v)
        
        # Forward bias region
        mask = v > 0
        v_fwd = v[mask]
        
        # Peak region
        peak_mask = v_fwd <= self.v_peak
        i_tunnel[mask][peak_mask] = self.i_peak * v_fwd[peak_mask] / self.v_peak
        
        # Valley region
        valley_mask = (v_fwd > self.v_peak) & (v_fwd <= self.v_valley)
        slope = (self.i_valley - self.i_peak) / (self.v_valley - self.v_peak)
        i_tunnel[mask][valley_mask] = self.i_peak + slope * (v_fwd[valley_mask] - self.v_peak)
        
        # Beyond valley
        beyond_mask = v_fwd > self.v_valley
        i_tunnel[mask][beyond_mask] = self.i_valley + 0.1 * (v_fwd[beyond_mask] - self.v_valley)
        
        # Temperature scaling
        temp_scale = np.exp(-0.01 * (self.temp - 300.0))
        i_tunnel *= temp_scale
        
        # Total current is minimum of series and tunnel currents
        return np.minimum(i_series, i_tunnel)
    
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
        # Simple RC circuit dynamics
        tau = self.r_series * 1e-3  # Time constant (ms)
        v_new = v + (vbias - v) * dt / tau
        i_new = self.iv_characteristic(np.array([v_new]))[0]
        
        return v_new, i_new
    
    def simulate(self, t_end: float, dt: float, vbias: float, v0: float, i0: float) -> Tuple[FloatArray, FloatArray, FloatArray]:
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
        # Create time array
        t = np.arange(0, t_end + dt, dt)
        
        # Initialize arrays
        v = np.zeros_like(t)
        i = np.zeros_like(t)
        
        # Set initial conditions
        v[0] = v0
        i[0] = i0
        
        # Integrate
        for n in range(1, len(t)):
            v[n], i[n] = self.step(dt, vbias, v[n-1], i[n-1])
        
        return t, v, i
    
    def simulate_vectorized(self, t_end: float, dt: float, vbias: float, v0: float, i0: float) -> Tuple[FloatArray, FloatArray, FloatArray]:
        """
        Run a complete simulation using vectorized operations.
        
        Args:
            t_end: End time
            dt: Time step
            vbias: Bias voltage
            v0: Initial voltage
            i0: Initial current
            
        Returns:
            Tuple of (time array, voltage array, current array)
        """
        # For this simple model, we can use the same implementation
        return self.simulate(t_end, dt, vbias, v0, i0) 