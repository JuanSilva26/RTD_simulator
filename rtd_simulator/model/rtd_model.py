"""
RTD Model Implementation

This module implements the core Resonant Tunneling Diode (RTD) models.
"""

import numpy as np
import torch
from typing import Tuple, Optional, Union, List, Dict, Any
from numpy.typing import NDArray
from abc import ABC, abstractmethod
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
    
    All RTD models must implement these methods to ensure compatibility
    with the simulation platform.
    """
    
    @abstractmethod
    def iv_characteristic(self, voltage: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Calculate the current-voltage (IV) characteristic for given voltage values.
        
        Args:
            voltage: Array of voltage values
            
        Returns:
            Array of current values according to the model's IV characteristic
        """
        pass
    
    @abstractmethod
    def step(self, dt: float, vbias: float, perturbation: Optional[float] = None) -> Tuple[float, float]:
        """
        Perform one integration step.
        
        Args:
            dt: Time step
            vbias: Bias voltage
            perturbation: Optional perturbation to add to the system
            
        Returns:
            Tuple of (voltage, current) after the step
        """
        pass
    
    @abstractmethod
    def simulate(self, t_end: float, dt: float, vbias: Union[float, NDArray[np.float64]]) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        Simulate the RTD dynamics.
        
        Args:
            t_end: End time for simulation
            dt: Time step
            vbias: Bias voltage (can be a scalar or array)
            
        Returns:
            Tuple of (time array, voltage array, current array)
        """
        pass
    
    @abstractmethod
    def get_parameter_ranges(self) -> Dict[str, Tuple[float, float]]:
        """
        Return valid ranges for all model parameters.
        
        Returns:
            Dictionary mapping parameter names to (min, max) tuples
        """
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
        
        Args:
            t: Time array
            perturbation_type: Type of perturbation
            amplitude: Amplitude of the perturbation
            frequency: Frequency of the perturbation
            start_time: Start time of the perturbation
            duration: Duration of the perturbation (None for continuous)
            
        Returns:
            Array of perturbation values
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

class SimplifiedRTDModel(RTDModel):
    """
    Simplified dimensionless RTD model implementing the differential equations:
    dv/dt = (i - f(v))/m
    di/dt = m(vbias - v - ri)
    
    where f(v) = kv - h*arctan(v/w) is a simplified IV characteristic
    """
    
    def __init__(self, 
                 k: float = 5.92,
                 h: float = 15.76,
                 w: float = 2.259,
                 m: float = 0.078,
                 r: float = 1.0,
                 initial_v: float = -1.1,
                 initial_i: Optional[float] = None):
        """
        Initialize the simplified RTD model with default or custom parameters.
        
        Args:
            k: Fitting parameter for linear term
            h: Fitting parameter for arctan term
            w: Fitting parameter for arctan term
            m: Stiffness parameter
            r: Resistance
            initial_v: Initial voltage
            initial_i: Initial current (if None, will be calculated from IV curve)
        """
        # Validate parameters
        self._validate_parameters(k, h, w, m, r)
        
        self.k = k
        self.h = h
        self.w = w
        self.m = m
        self.r = r
        
        # Initialize device state
        self.v = initial_v
        # Calculate initial current if not provided
        if initial_i is None:
            self.i = float(self.simplified_iv(np.array([initial_v]))[0])
        else:
            self.i = initial_i
        
    def _validate_parameters(self, k: float, h: float, w: float, m: float, r: float) -> None:
        """Validate that all parameters are within reasonable ranges."""
        if k <= 0:
            raise ValueError("k must be positive")
        if h <= 0:
            raise ValueError("h must be positive")
        if w <= 0:
            raise ValueError("w must be positive")
        if m <= 0:
            raise ValueError("m must be positive")
        if r <= 0:
            raise ValueError("r must be positive")
    
    def get_parameter_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Return valid ranges for all model parameters."""
        return {
            'k': (0.1, 10.0),
            'h': (1.0, 30.0),
            'w': (0.1, 5.0),
            'm': (0.01, 1.0),
            'r': (0.1, 10.0)
        }
        
    def simplified_iv(self, voltage: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Calculate the simplified IV characteristic f(v) = kv - h*arctan(v/w)
        This is a dimensionless approximation of the RTD's IV curve.
        
        Args:
            voltage: Array of voltage values
            
        Returns:
            Array of current values
        """
        return self.k * voltage - self.h * np.arctan(voltage / self.w)
    
    # Implement the abstract method using our simplified IV
    def iv_characteristic(self, voltage: NDArray[np.float64]) -> NDArray[np.float64]:
        """Implementation of the abstract method using simplified IV characteristic."""
        return self.simplified_iv(voltage)

    def step(self, 
             dt: float, 
             vbias: float, 
             perturbation: Optional[float] = None) -> Tuple[float, float]:
        """
        Perform one integration step using Runge-Kutta 4th order method.
        
        Args:
            dt: Time step
            vbias: Bias voltage
            perturbation: Optional perturbation to add to the system
            
        Returns:
            Tuple of (voltage, current) after the step
        """
        # Add perturbation if provided
        if perturbation is not None:
            vbias += perturbation
            
        # RK4 integration
        k1v = (self.i - float(self.simplified_iv(np.array([self.v]))[0])) / self.m
        k1i = self.m * (vbias - self.v - self.r * self.i)
        
        k2v = (self.i + 0.5 * dt * k1i - float(self.simplified_iv(np.array([self.v + 0.5 * dt * k1v]))[0])) / self.m
        k2i = self.m * (vbias - (self.v + 0.5 * dt * k1v) - self.r * (self.i + 0.5 * dt * k1i))
        
        k3v = (self.i + 0.5 * dt * k2i - float(self.simplified_iv(np.array([self.v + 0.5 * dt * k2v]))[0])) / self.m
        k3i = self.m * (vbias - (self.v + 0.5 * dt * k2v) - self.r * (self.i + 0.5 * dt * k2i))
        
        k4v = (self.i + dt * k3i - float(self.simplified_iv(np.array([self.v + dt * k3v]))[0])) / self.m
        k4i = self.m * (vbias - (self.v + dt * k3v) - self.r * (self.i + dt * k3i))
        
        # Update state
        self.v += (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
        self.i += (dt / 6.0) * (k1i + 2 * k2i + 2 * k3i + k4i)
        
        return float(self.v), float(self.i)
    
    def simulate(self, t_end: float, dt: float, vbias: Union[float, NDArray[np.float64]]) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        Simulate the RTD dynamics.
        
        Args:
            t_end: End time for simulation
            dt: Time step
            vbias: Bias voltage (can be a scalar or array)
            
        Returns:
            Tuple of (time array, voltage array, current array)
        """
        # Create time array
        t = np.arange(0, t_end, dt)
        n = len(t)
        
        # Initialize arrays
        v = np.zeros(n, dtype=np.float64)
        i = np.zeros(n, dtype=np.float64)
        
        # Set initial conditions
        v[0] = float(self.v)
        i[0] = float(self.i)
        
        # Convert vbias to array if scalar
        if isinstance(vbias, (float, int)):
            vbias = np.full(n, float(vbias), dtype=np.float64)
        
        # Ensure vbias is float64
        vbias = np.asarray(vbias, dtype=np.float64)
            
        # Simulation loop using Euler method
        for j in range(1, n):
            # Calculate derivatives
            dv = (1.0/self.m) * (i[j-1] - self.simplified_iv(np.array([v[j-1]])))
            di = self.m * (vbias[j-1] - v[j-1] - self.r * i[j-1])
            
            # Update state
            v[j] = v[j-1] + dv * dt
            i[j] = i[j-1] + di * dt
            
        return t, v, i 

class SchulmanRTDModel(RTDModel):
    """
    Physical RTD model based on Schulman's equations.
    
    Device dynamics:
    dV/dt = 1/C * (I - F(V))
    dI/dt = 1/L * (Vbias - V - R*I)
    
    where F(V) is Schulman's IV characteristic:
    F(V) = J1 * J2 + J3
    J1 = a * ln((1 + term_up)/(1 + term_down))
    J2 = π/2 + arctan((c - n1*V)/d)
    J3 = h * (exp(qe/(k*T) * (n2*V)) - 1)
    term_up = exp(qe/(k*T) * (b - c + n1*V))
    term_down = exp(qe/(k*T) * (b - c - n1*V))
    """
    
    # Physical constants
    _k = 1.380649e-23    # Boltzmann constant [J/K]
    _qe = 1.602176634e-19  # Electron charge [C]
    
    def __init__(self,
                 # Fitting parameters
                 a: float = 6.715e-4,    # Current scale [A]
                 b: float = 6.499e-2,    # Voltage offset [V]
                 c: float = 9.709e-2,    # Peak voltage [V]
                 d: float = 2.213e-2,    # Width parameter [V]
                 h: float = 1.664e-4,    # Current scale [A]
                 n1: float = 3.106e-2,   # Voltage coefficient
                 n2: float = 1.721e-2,   # Voltage coefficient
                 # Circuit parameters
                 capacitance: float = 1.0e-12,  # Device capacitance [F]
                 inductance: float = 1.0e-9,    # Circuit inductance [H]
                 resistance: float = 50.0,      # Circuit resistance [Ω]
                 temperature: float = 300.0,    # Temperature [K]
                 # Initial conditions
                 initial_v: float = 0.0,
                 initial_i: Optional[float] = None):
        """
        Initialize the Schulman RTD model with default or custom parameters.
        
        Args:
            a: Current scale [A]
            b: Voltage offset [V]
            c: Peak voltage [V]
            d: Width parameter [V]
            h: Current scale [A]
            n1: Voltage coefficient
            n2: Voltage coefficient
            capacitance: Device capacitance [F]
            inductance: Circuit inductance [H]
            resistance: Circuit resistance [Ω]
            temperature: Temperature [K]
            initial_v: Initial voltage [V]
            initial_i: Initial current [A] (if None, will be calculated from IV curve)
        """
        # Validate parameters
        self._validate_parameters(a, b, c, d, h, n1, n2, capacitance, inductance, resistance, temperature)
        
        # Fitting parameters
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.h = h
        self.n1 = n1
        self.n2 = n2
        
        # Circuit parameters
        self.C = capacitance
        self.L = inductance
        self.R = resistance
        self.T = temperature
        
        # Initialize device state
        self.v = initial_v
        if initial_i is None:
            self.i = float(self.schulman_iv(np.array([initial_v]))[0])
        else:
            self.i = initial_i
            
    def _validate_parameters(self, a: float, b: float, c: float, d: float, h: float,
                           n1: float, n2: float, C: float, L: float, R: float, T: float) -> None:
        """Validate that all parameters are within reasonable ranges."""
        if a <= 0:
            raise ValueError("a must be positive")
        if d <= 0:
            raise ValueError("d must be positive")
        if h <= 0:
            raise ValueError("h must be positive")
        if C <= 0:
            raise ValueError("Capacitance must be positive")
        if L <= 0:
            raise ValueError("Inductance must be positive")
        if R <= 0:
            raise ValueError("Resistance must be positive")
        if T <= 0:
            raise ValueError("Temperature must be positive")
            
    def get_parameter_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Return valid ranges for all model parameters."""
        return {
            'a': (1e-6, 1e-2),    # [A]
            'b': (-1.0, 1.0),     # [V]
            'c': (-1.0, 1.0),     # [V]
            'd': (1e-3, 1e-1),    # [V]
            'h': (1e-6, 1e-2),    # [A]
            'n1': (1e-3, 1e-1),   # dimensionless
            'n2': (1e-3, 1e-1),   # dimensionless
            'C': (1e-15, 1e-9),   # [F]
            'L': (1e-12, 1e-6),   # [H]
            'R': (1.0, 1e3),      # [Ω]
            'T': (4.0, 400.0)     # [K]
        }
        
    def schulman_iv(self, voltage: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Calculate Schulman's IV characteristic F(V).
        
        Args:
            voltage: Array of voltage values [V]
            
        Returns:
            Array of current values [A]
        """
        # Calculate thermal voltage
        VT = self._k * self.T / self._qe
        
        # Calculate exponential terms
        term_up = np.exp((self.b - self.c + self.n1 * voltage) / VT)
        term_down = np.exp((self.b - self.c - self.n1 * voltage) / VT)
        
        # Calculate components
        J1 = self.a * np.log((1 + term_up) / (1 + term_down))
        J2 = np.pi/2 + np.arctan((self.c - self.n1 * voltage) / self.d)
        J3 = self.h * (np.exp(self.n2 * voltage / VT) - 1)
        
        return J1 * J2 + J3
    
    def iv_characteristic(self, voltage: NDArray[np.float64]) -> NDArray[np.float64]:
        """Implementation of the abstract method using Schulman's IV characteristic."""
        return self.schulman_iv(voltage)
    
    def step(self, 
             dt: float, 
             vbias: float, 
             perturbation: Optional[float] = None) -> Tuple[float, float]:
        """
        Perform one integration step using Runge-Kutta 4th order method.
        
        Args:
            dt: Time step [s]
            vbias: Bias voltage [V]
            perturbation: Optional perturbation to add to the system [V]
            
        Returns:
            Tuple of (voltage [V], current [A]) after the step
        """
        # Add perturbation if provided
        if perturbation is not None:
            vbias += perturbation
            
        # RK4 integration
        k1v = (1.0/self.C) * (self.i - float(self.schulman_iv(np.array([self.v]))[0]))
        k1i = (1.0/self.L) * (vbias - self.v - self.R * self.i)
        
        k2v = (1.0/self.C) * ((self.i + 0.5 * dt * k1i) - float(self.schulman_iv(np.array([self.v + 0.5 * dt * k1v]))[0]))
        k2i = (1.0/self.L) * (vbias - (self.v + 0.5 * dt * k1v) - self.R * (self.i + 0.5 * dt * k1i))
        
        k3v = (1.0/self.C) * ((self.i + 0.5 * dt * k2i) - float(self.schulman_iv(np.array([self.v + 0.5 * dt * k2v]))[0]))
        k3i = (1.0/self.L) * (vbias - (self.v + 0.5 * dt * k2v) - self.R * (self.i + 0.5 * dt * k2i))
        
        k4v = (1.0/self.C) * ((self.i + dt * k3i) - float(self.schulman_iv(np.array([self.v + dt * k3v]))[0]))
        k4i = (1.0/self.L) * (vbias - (self.v + dt * k3v) - self.R * (self.i + dt * k3i))
        
        # Update state
        self.v += (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
        self.i += (dt / 6.0) * (k1i + 2 * k2i + 2 * k3i + k4i)
        
        return float(self.v), float(self.i)
    
    def simulate(self, t_end: float, dt: float, vbias: Union[float, NDArray[np.float64]]) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        Simulate the RTD dynamics.
        
        Args:
            t_end: End time for simulation [s]
            dt: Time step [s]
            vbias: Bias voltage [V] (can be a scalar or array)
            
        Returns:
            Tuple of (time array [s], voltage array [V], current array [A])
        """
        # Create time array
        t = np.arange(0, t_end, dt)
        n = len(t)
        
        # Initialize arrays
        v = np.zeros(n, dtype=np.float64)
        i = np.zeros(n, dtype=np.float64)
        
        # Set initial conditions
        v[0] = float(self.v)
        i[0] = float(self.i)
        
        # Convert vbias to array if scalar
        if isinstance(vbias, (float, int)):
            vbias = np.full(n, float(vbias), dtype=np.float64)
        
        # Ensure vbias is float64
        vbias = np.asarray(vbias, dtype=np.float64)
            
        # Simulation loop using Euler method
        for j in range(1, n):
            # Calculate derivatives
            dv = (1.0/self.C) * (i[j-1] - self.schulman_iv(np.array([v[j-1]])))
            di = (1.0/self.L) * (vbias[j-1] - v[j-1] - self.R * i[j-1])
            
            # Update state
            v[j] = v[j-1] + dv * dt
            i[j] = i[j-1] + di * dt
            
        return t, v, i 