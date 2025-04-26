"""
RTD Model Implementation

This module implements the core Resonant Tunneling Diode (RTD) models.

Numerical Methods:
    - Runge-Kutta 4th order (RK4) integration for all simulations
    - Adaptive time stepping with error control
    - Stability analysis through eigenvalue computation
    - Numba-accelerated implementations for performance

Stability Features:
    - Automatic time step adjustment based on error estimates
    - Stability checks through Jacobian eigenvalue analysis
    - Protection against numerical instabilities
    - Configurable error tolerances and step size limits

The models implement the following differential equations:
    dV/dt = 1/C * (I - F(V))  # Simplified model: C = m
    dI/dt = 1/L * (Vbias - V - R*I)  # Simplified model: L = 1/m, R = r

where F(V) is the IV characteristic of the device.
"""

from typing import (
    Tuple, Optional, Union, Dict, Any, List, TypeVar, Type,
    Protocol, runtime_checkable, overload
)
import numpy as np
from numpy.typing import NDArray
import torch
from abc import ABC, abstractmethod
from enum import Enum
from .base import PerturbationType, register_rtd_model, RTDModel as BaseRTDModel
from .numerical import rk4_step, rk4_simulate, rk4_adaptive
from numba import jit

# Type aliases for better readability
FloatArray = NDArray[np.float64]
SimulationResult = Tuple[FloatArray, FloatArray, FloatArray]
ParameterRanges = Dict[str, Tuple[float, float]]
StabilityResult = Tuple[bool, str]

class RTDModel(BaseRTDModel):
    """
    Abstract base class for RTD models.
    
    All RTD models must implement these methods to ensure compatibility
    with the simulation platform.
    
    Numerical Integration:
        - All simulations use Runge-Kutta 4th order (RK4) method
        - Adaptive time stepping available through simulate_adaptive
        - Stability analysis through check_stability method
    
    Stability:
        - Time step selection based on system eigenvalues
        - Error control through relative and absolute tolerances
        - Protection against numerical instabilities
    """
    
    @abstractmethod
    def iv_characteristic(self, voltage: FloatArray) -> FloatArray:
        """
        Calculate the current-voltage (IV) characteristic for given voltages.
        
        Args:
            voltage: Array of voltage values to calculate current for.
            
        Returns:
            Array of current values corresponding to the input voltages.
        """
        pass

    @abstractmethod
    def step(
        self,
        dt: float,
        vbias: float,
        perturbation: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Perform a single simulation step.
        
        Args:
            dt: Time step size.
            vbias: Bias voltage.
            perturbation: Optional perturbation value.
            
        Returns:
            Tuple of (voltage, current) at the end of the step.
        """
        pass

    @abstractmethod
    def simulate(
        self,
        t_end: float,
        dt: float,
        vbias: Union[float, FloatArray]
    ) -> SimulationResult:
        """
        Run a complete simulation.
        
        Args:
            t_end: End time of the simulation.
            dt: Time step size.
            vbias: Bias voltage (constant or time-varying).
            
        Returns:
            Tuple of (time, voltage, current) arrays.
        """
        pass

    @abstractmethod
    def get_parameter_ranges(self) -> ParameterRanges:
        """
        Get the valid parameter ranges for this model.
        
        Returns:
            Dictionary mapping parameter names to (min, max) tuples.
        """
        pass

    @abstractmethod
    def check_stability(self, dt: float, vbias: float) -> StabilityResult:
        """
        Check the stability of the simulation with given parameters.
        
        Args:
            dt: Time step size to check.
            vbias: Bias voltage to check.
            
        Returns:
            Tuple of (is_stable, reason) where is_stable is a boolean
            indicating stability and reason is a string explaining the result.
        """
        pass

    def create_perturbation(
        self,
        t: FloatArray,
        perturbation_type: PerturbationType,
        amplitude: float,
        frequency: float = 1.0,
        start_time: float = 0.0,
        duration: Optional[float] = None
    ) -> FloatArray:
        """
        Create a perturbation signal of the specified type.
        
        Args:
            t: Time array.
            perturbation_type: Type of perturbation to create.
            amplitude: Amplitude of the perturbation.
            frequency: Frequency of the perturbation (default: 1.0).
            start_time: Start time of the perturbation (default: 0.0).
            duration: Optional duration of the perturbation.
            
        Returns:
            Array containing the perturbation signal.
        """
        return super().create_perturbation(
            t, perturbation_type, amplitude, frequency, start_time, duration
        )

class SimplifiedRTDModel(RTDModel):
    """
    Simplified dimensionless RTD model implementing the differential equations:
    dv/dt = (i - f(v))/m
    di/dt = m(vbias - v - ri)
    
    where f(v) = kv - h*arctan(v/w) is a simplified IV characteristic.
    
    Numerical Methods:
        - RK4 integration with fixed or adaptive time stepping
        - Stability analysis through eigenvalue computation
        - Numba-accelerated implementation available
    
    Stability:
        - Time step selection based on system eigenvalues
        - Error control through relative and absolute tolerances
        - Protection against numerical instabilities
    """
    
    def __init__(self, 
                 k: float = 5.92,
                 h: float = 15.76,
                 w: float = 2.259,
                 m: float = 0.078,
                 r: float = 1.0,
                 initial_v: float = -1.1,
                 initial_i: Optional[float] = None) -> None:
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
            
        # Define derivative functions
        self._f_v = lambda i, v: (i - float(self.simplified_iv(np.array([v]))[0])) / self.m
        self._f_i = lambda vbias, v, i: self.m * (vbias - v - self.r * i)
        
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
    
    def get_parameter_ranges(self) -> ParameterRanges:
        """Return valid ranges for all model parameters."""
        return {
            'k': (0.1, 10.0),
            'h': (1.0, 30.0),
            'w': (0.1, 5.0),
            'm': (0.01, 1.0),
            'r': (0.1, 10.0)
        }
        
    def simplified_iv(self, voltage: FloatArray) -> FloatArray:
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
    def iv_characteristic(self, voltage: FloatArray) -> FloatArray:
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
        # Use centralized RK4 implementation
        new_v, new_i = rk4_step(dt, vbias, self.v, self.i, self._f_v, self._f_i, perturbation)
        
        # Update state
        self.v = new_v
        self.i = new_i
        
        return float(self.v), float(self.i)
    
    def simulate(self, t_end: float, dt: float, vbias: Union[float, FloatArray]) -> SimulationResult:
        """
        Simulate the RTD dynamics using Runge-Kutta 4th order method.
        
        Args:
            t_end: End time for simulation [s]
            dt: Time step [s]
            vbias: Bias voltage [V] (can be a scalar or array)
            
        Returns:
            Tuple of (time array [s], voltage array [V], current array [A])
        """
        # Use centralized RK4 implementation
        return rk4_simulate(t_end, dt, vbias, self.v, self.i, self._f_v, self._f_i)

    def simulate_vectorized(self, t_end: float, dt: float, vbias: Union[float, FloatArray]) -> SimulationResult:
        """
        Simulate the RTD dynamics using Numba-accelerated implementation.
        
        Args:
            t_end: End time for simulation
            dt: Time step
            vbias: Bias voltage (can be a scalar or array)
            
        Returns:
            Tuple of (time array, voltage array, current array)
        """
        # Create time array for size calculation
        t = np.arange(0, t_end, dt)
        n = len(t)
        
        # Convert vbias to array if scalar
        if isinstance(vbias, (float, int)):
            vbias = np.full(n, float(vbias), dtype=np.float64)
        
        # Ensure vbias is float64
        vbias = np.asarray(vbias, dtype=np.float64)
        
        # Call the Numba-optimized simulation core
        return _simulate_rtd_numba(t_end, dt, vbias,
                                 self.k, self.h, self.w, self.m, self.r,
                                 float(self.v), float(self.i))

    def check_stability(self, dt: float, vbias: float) -> StabilityResult:
        """
        Check if the current model configuration is stable for the given time step and bias voltage.
        
        Args:
            dt: Time step to check
            vbias: Bias voltage to check
            
        Returns:
            Tuple of (is_stable, message)
        """
        # Get current state
        v = self.v
        i = self.i
        
        # Calculate dF/dv numerically
        dv = 1e-6
        F_v = float(self.simplified_iv(np.array([v]))[0])
        F_v_plus = float(self.simplified_iv(np.array([v + dv]))[0])
        dF_dv = (F_v_plus - F_v) / dv
        
        # Construct Jacobian
        J = np.array([
            [-1.0/self.m * dF_dv, 1.0/self.m],
            [-self.m, -self.m * self.r]
        ])
        
        # Calculate eigenvalues
        eigvals = np.linalg.eigvals(J)
        
        # Check stability
        max_real_eigval = np.max(np.real(eigvals))
        if max_real_eigval > 0:
            return False, f"System is unstable: maximum eigenvalue real part is {max_real_eigval:.2e}"
        
        # Check time step stability
        max_eigval_mag = np.max(np.abs(eigvals))
        stable_dt = 2.0 / max_eigval_mag
        
        if dt > stable_dt:
            return False, f"Time step {dt:.2e} is too large for stability. Maximum stable time step is {stable_dt:.2e}"
        
        return True, "System is stable for the given configuration"

    def simulate_adaptive(self, t_end: float, dt_initial: float, vbias: Union[float, FloatArray],
                        rel_tol: float = 1e-6, abs_tol: float = 1e-8,
                        max_dt: float = 1e-9, min_dt: float = 1e-12) -> SimulationResult:
        """
        Simulate the RTD dynamics using adaptive time stepping with Runge-Kutta 4th order method.
        
        Args:
            t_end: End time for simulation [s]
            dt_initial: Initial time step [s]
            vbias: Bias voltage [V] (can be a scalar or array)
            rel_tol: Relative error tolerance
            abs_tol: Absolute error tolerance
            max_dt: Maximum allowed time step [s]
            min_dt: Minimum allowed time step [s]
            
        Returns:
            Tuple of (time array [s], voltage array [V], current array [A])
        """
        # Use centralized adaptive RK4 implementation
        return rk4_adaptive(t_end, dt_initial, vbias, self.v, self.i, 
                          self._f_v, self._f_i, rel_tol, abs_tol, max_dt, min_dt)

@jit(nopython=True)
def _simplified_iv_numba(v: float, k: float, h: float, w: float) -> float:
    """Numba-optimized version of the simplified IV characteristic."""
    return k * v - h * np.arctan(v / w)

@jit(nopython=True)
def _simulate_rtd_numba(t_end: float, dt: float, vbias: np.ndarray,
                       k: float, h: float, w: float, m: float, r: float,
                       initial_v: float, initial_i: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba-optimized RTD simulation core using Runge-Kutta 4th order method.
    """
    # Create time array
    t = np.arange(0, t_end, dt)
    n = len(t)
    
    # Initialize arrays
    v = np.zeros(n, dtype=np.float64)
    i = np.zeros(n, dtype=np.float64)
    
    # Set initial conditions
    v[0] = initial_v
    i[0] = initial_i
    
    # Pre-calculate constants
    dt_over_m = dt / m
    m_dt = m * dt
    
    # Simulation loop using RK4
    for j in range(1, n):
        # RK4 integration
        # k1
        k1v = dt_over_m * (i[j-1] - _simplified_iv_numba(v[j-1], k, h, w))
        k1i = m_dt * (vbias[j-1] - v[j-1] - r * i[j-1])
        
        # k2
        v_temp = v[j-1] + 0.5 * k1v
        i_temp = i[j-1] + 0.5 * k1i
        k2v = dt_over_m * (i_temp - _simplified_iv_numba(v_temp, k, h, w))
        k2i = m_dt * (vbias[j-1] - v_temp - r * i_temp)
        
        # k3
        v_temp = v[j-1] + 0.5 * k2v
        i_temp = i[j-1] + 0.5 * k2i
        k3v = dt_over_m * (i_temp - _simplified_iv_numba(v_temp, k, h, w))
        k3i = m_dt * (vbias[j-1] - v_temp - r * i_temp)
        
        # k4
        v_temp = v[j-1] + k3v
        i_temp = i[j-1] + k3i
        k4v = dt_over_m * (i_temp - _simplified_iv_numba(v_temp, k, h, w))
        k4i = m_dt * (vbias[j-1] - v_temp - r * i_temp)
        
        # Update state
        v[j] = v[j-1] + (k1v + 2 * k2v + 2 * k3v + k4v) / 6.0
        i[j] = i[j-1] + (k1i + 2 * k2i + 2 * k3i + k4i) / 6.0
        
    return t, v, i

@jit(nopython=True)
def _schulman_iv_numba(voltage: float, a: float, b: float, c: float, d: float, h: float,
                      n1: float, n2: float, T: float, k_b: float, qe: float) -> float:
    """
    Numba-optimized version of Schulman's IV characteristic with numerical stability improvements.
    """
    # Calculate thermal voltage
    VT = k_b * T / qe
    
    # Calculate exponential terms with overflow protection
    exp_arg_up = (b - c + n1 * voltage) / VT
    exp_arg_down = (b - c - n1 * voltage) / VT
    
    # Clip arguments to avoid overflow
    exp_arg_up = min(exp_arg_up, 700.0)  # np.exp(700) is near float64 max
    exp_arg_down = min(exp_arg_down, 700.0)
    
    term_up = np.exp(exp_arg_up)
    term_down = np.exp(exp_arg_down)
    
    # Calculate components with protection against division by zero
    eps = 1e-30  # Small constant to prevent division by zero
    J1 = a * np.log((1.0 + term_up) / (1.0 + term_down + eps))
    J2 = np.pi/2 + np.arctan((c - n1 * voltage) / (d + eps))
    
    # Protect against overflow in the final exponential
    exp_arg_final = n2 * voltage / VT
    exp_arg_final = min(exp_arg_final, 700.0)
    J3 = h * (np.exp(exp_arg_final) - 1.0)
    
    return J1 * J2 + J3

@jit(nopython=True)
def _simulate_schulman_rtd_numba(t_end: float, dt: float, vbias: np.ndarray,
                                a: float, b: float, c: float, d: float, h: float,
                                n1: float, n2: float, C: float, L: float, R: float, T: float,
                                k_b: float, qe: float, initial_v: float, initial_i: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba-optimized Schulman RTD simulation core using Runge-Kutta 4th order method.
    """
    # Create time array
    t = np.arange(0, t_end, dt)
    n = len(t)
    
    # Initialize arrays
    v = np.zeros(n, dtype=np.float64)
    i = np.zeros(n, dtype=np.float64)
    
    # Set initial conditions
    v[0] = initial_v
    i[0] = initial_i
    
    # Pre-calculate constants
    dt_over_C = dt / C
    dt_over_L = dt / L
    
    # Simulation loop using RK4
    for j in range(1, n):
        # RK4 integration
        # k1
        k1v = dt_over_C * (i[j-1] - _schulman_iv_numba(v[j-1], a, b, c, d, h, n1, n2, T, k_b, qe))
        k1i = dt_over_L * (vbias[j-1] - v[j-1] - R * i[j-1])
        
        # k2
        v_temp = v[j-1] + 0.5 * k1v
        i_temp = i[j-1] + 0.5 * k1i
        k2v = dt_over_C * (i_temp - _schulman_iv_numba(v_temp, a, b, c, d, h, n1, n2, T, k_b, qe))
        k2i = dt_over_L * (vbias[j-1] - v_temp - R * i_temp)
        
        # k3
        v_temp = v[j-1] + 0.5 * k2v
        i_temp = i[j-1] + 0.5 * k2i
        k3v = dt_over_C * (i_temp - _schulman_iv_numba(v_temp, a, b, c, d, h, n1, n2, T, k_b, qe))
        k3i = dt_over_L * (vbias[j-1] - v_temp - R * i_temp)
        
        # k4
        v_temp = v[j-1] + k3v
        i_temp = i[j-1] + k3i
        k4v = dt_over_C * (i_temp - _schulman_iv_numba(v_temp, a, b, c, d, h, n1, n2, T, k_b, qe))
        k4i = dt_over_L * (vbias[j-1] - v_temp - R * i_temp)
        
        # Update state
        v[j] = v[j-1] + (k1v + 2 * k2v + 2 * k3v + k4v) / 6.0
        i[j] = i[j-1] + (k1i + 2 * k2i + 2 * k3i + k4i) / 6.0
        
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
    
    Numerical Methods:
        - RK4 integration with fixed or adaptive time stepping
        - Stability analysis through eigenvalue computation
        - Numba-accelerated implementation available
    
    Stability:
        - Time step selection based on system eigenvalues
        - Error control through relative and absolute tolerances
        - Protection against numerical instabilities
        - Special handling for exponential terms to prevent overflow
    """
    
    # Physical constants
    _k: float = 1.380649e-23    # Boltzmann constant [J/K]
    _qe: float = 1.602176634e-19  # Elementary charge [C]
    
    def __init__(self,
                 a: float = 1.0e-3,
                 b: float = 0.1,
                 c: float = 0.2,
                 d: float = 0.01,
                 h: float = 1.0e-6,
                 n1: float = 1.0,
                 n2: float = 1.0,
                 C: float = 1.0e-12,
                 L: float = 1.0e-9,
                 R: float = 50.0,
                 T: float = 300.0,
                 initial_v: float = 0.0,
                 initial_i: Optional[float] = None) -> None:
        """
        Initialize the Schulman RTD model with default or custom parameters.
        
        Args:
            a: Fitting parameter for J1 term
            b: Fitting parameter for J1 term
            c: Fitting parameter for J2 term
            d: Fitting parameter for J2 term
            h: Fitting parameter for J3 term
            n1: Fitting parameter for J1 and J2 terms
            n2: Fitting parameter for J3 term
            C: Capacitance [F]
            L: Inductance [H]
            R: Resistance [Ω]
            T: Temperature [K]
            initial_v: Initial voltage [V]
            initial_i: Initial current [A] (if None, will be calculated from IV curve)
        """
        # Validate parameters
        self._validate_parameters(a, b, c, d, h, n1, n2, C, L, R, T)
        
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.h = h
        self.n1 = n1
        self.n2 = n2
        self.C = C
        self.L = L
        self.R = R
        self.T = T
        
        # Initialize device state
        self.v = initial_v
        # Calculate initial current if not provided
        if initial_i is None:
            self.i = float(self.schulman_iv(np.array([initial_v]))[0])
        else:
            self.i = initial_i
            
        # Define derivative functions
        self._f_v = lambda i, v: (i - float(self.schulman_iv(np.array([v]))[0])) / self.C
        self._f_i = lambda vbias, v, i: (vbias - v - self.R * i) / self.L
        
    def _validate_parameters(self, a: float, b: float, c: float, d: float, h: float,
                           n1: float, n2: float, C: float, L: float, R: float, T: float) -> None:
        """Validate that all parameters are within reasonable ranges."""
        if a <= 0:
            raise ValueError("a must be positive")
        if b <= 0:
            raise ValueError("b must be positive")
        if c <= 0:
            raise ValueError("c must be positive")
        if d <= 0:
            raise ValueError("d must be positive")
        if h <= 0:
            raise ValueError("h must be positive")
        if n1 <= 0:
            raise ValueError("n1 must be positive")
        if n2 <= 0:
            raise ValueError("n2 must be positive")
        if C <= 0:
            raise ValueError("C must be positive")
        if L <= 0:
            raise ValueError("L must be positive")
        if R <= 0:
            raise ValueError("R must be positive")
        if T <= 0:
            raise ValueError("T must be positive")
    
    def get_parameter_ranges(self) -> ParameterRanges:
        """Return valid ranges for all model parameters."""
        return {
            'a': (1e-6, 1e-2),
            'b': (0.01, 1.0),
            'c': (0.01, 1.0),
            'd': (0.001, 0.1),
            'h': (1e-8, 1e-4),
            'n1': (0.1, 10.0),
            'n2': (0.1, 10.0),
            'C': (1e-15, 1e-9),
            'L': (1e-12, 1e-6),
            'R': (1.0, 1000.0),
            'T': (4.0, 400.0)
        }
        
    def schulman_iv(self, voltage: FloatArray) -> FloatArray:
        """
        Calculate Schulman's IV characteristic.
        
        Args:
            voltage: Array of voltage values [V]
            
        Returns:
            Array of current values [A]
        """
        # Pre-calculate common terms
        kT = self._k * self.T
        qe = self._qe
        
        # Calculate J1 term
        term_up = np.exp(qe/kT * (self.b - self.c + self.n1 * voltage))
        term_down = np.exp(qe/kT * (self.b - self.c - self.n1 * voltage))
        J1 = self.a * np.log((1 + term_up)/(1 + term_down))
        
        # Calculate J2 term
        J2 = np.pi/2 + np.arctan((self.c - self.n1 * voltage)/self.d)
        
        # Calculate J3 term
        J3 = self.h * (np.exp(qe/kT * self.n2 * voltage) - 1)
        
        # Combine terms
        return J1 * J2 + J3
    
    # Implement the abstract method using Schulman's IV
    def iv_characteristic(self, voltage: FloatArray) -> FloatArray:
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
        # Use centralized RK4 implementation
        new_v, new_i = rk4_step(dt, vbias, self.v, self.i, self._f_v, self._f_i, perturbation)
        
        # Update state
        self.v = new_v
        self.i = new_i
        
        return float(self.v), float(self.i)
    
    def simulate(self, t_end: float, dt: float, vbias: Union[float, FloatArray]) -> SimulationResult:
        """
        Simulate the RTD dynamics using Runge-Kutta 4th order method.
        
        Args:
            t_end: End time for simulation [s]
            dt: Time step [s]
            vbias: Bias voltage [V] (can be a scalar or array)
            
        Returns:
            Tuple of (time array [s], voltage array [V], current array [A])
        """
        # Use centralized RK4 implementation
        return rk4_simulate(t_end, dt, vbias, self.v, self.i, self._f_v, self._f_i)

    def simulate_vectorized(self, t_end: float, dt: float, vbias: Union[float, FloatArray]) -> SimulationResult:
        """
        Simulate the RTD dynamics using Numba-accelerated implementation.
        
        Args:
            t_end: End time for simulation
            dt: Time step
            vbias: Bias voltage (can be a scalar or array)
            
        Returns:
            Tuple of (time array, voltage array, current array)
        """
        # Create time array for size calculation
        t = np.arange(0, t_end, dt)
        n = len(t)
        
        # Convert vbias to array if scalar
        if isinstance(vbias, (float, int)):
            vbias = np.full(n, float(vbias), dtype=np.float64)
        
        # Ensure vbias is float64
        vbias = np.asarray(vbias, dtype=np.float64)
        
        # Call the Numba-optimized simulation core
        return _simulate_schulman_rtd_numba(t_end, dt, vbias,
                                          self.a, self.b, self.c, self.d, self.h,
                                          self.n1, self.n2, self.C, self.L, self.R, self.T,
                                          self._k, self._qe, float(self.v), float(self.i))

    def check_stability(self, dt: float, vbias: float) -> StabilityResult:
        """
        Check if the current model configuration is stable for the given time step and bias voltage.
        
        Args:
            dt: Time step to check
            vbias: Bias voltage to check
            
        Returns:
            Tuple of (is_stable, message)
        """
        # Get current state
        v = self.v
        i = self.i
        
        # Calculate dF/dv numerically
        dv = 1e-6
        F_v = float(self.schulman_iv(np.array([v]))[0])
        F_v_plus = float(self.schulman_iv(np.array([v + dv]))[0])
        dF_dv = (F_v_plus - F_v) / dv
        
        # Construct Jacobian
        J = np.array([
            [-1.0/self.C * dF_dv, 1.0/self.C],
            [-1.0/self.L, -self.R/self.L]
        ])
        
        # Calculate eigenvalues
        eigvals = np.linalg.eigvals(J)
        
        # Check stability
        max_real_eigval = np.max(np.real(eigvals))
        if max_real_eigval > 0:
            return False, f"System is unstable: maximum eigenvalue real part is {max_real_eigval:.2e}"
        
        # Check time step stability
        max_eigval_mag = np.max(np.abs(eigvals))
        stable_dt = 2.0 / max_eigval_mag
        
        if dt > stable_dt:
            return False, f"Time step {dt:.2e} is too large for stability. Maximum stable time step is {stable_dt:.2e}"
        
        return True, "System is stable for the given configuration"

    def simulate_adaptive(self, t_end: float, dt_initial: float, vbias: Union[float, FloatArray],
                        rel_tol: float = 1e-6, abs_tol: float = 1e-8,
                        max_dt: float = 1e-9, min_dt: float = 1e-12) -> SimulationResult:
        """
        Simulate the RTD dynamics using adaptive time stepping with Runge-Kutta 4th order method.
        
        Args:
            t_end: End time for simulation [s]
            dt_initial: Initial time step [s]
            vbias: Bias voltage [V] (can be a scalar or array)
            rel_tol: Relative error tolerance
            abs_tol: Absolute error tolerance
            max_dt: Maximum allowed time step [s]
            min_dt: Minimum allowed time step [s]
            
        Returns:
            Tuple of (time array [s], voltage array [V], current array [A])
        """
        # Use centralized adaptive RK4 implementation
        return rk4_adaptive(t_end, dt_initial, vbias, self.v, self.i, 
                          self._f_v, self._f_i, rel_tol, abs_tol, max_dt, min_dt)

# At the end of the file, register the models
register_rtd_model("Simplified", SimplifiedRTDModel)
register_rtd_model("Schulman", SchulmanRTDModel) 