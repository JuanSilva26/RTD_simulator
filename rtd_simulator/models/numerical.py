"""
Numerical methods for RTD simulation.

This module provides centralized implementations of numerical methods used in RTD simulations,
including Runge-Kutta integration and adaptive time stepping.
"""

from typing import Tuple, Callable, Optional, Union
import numpy as np
from numpy.typing import NDArray

# Type aliases
FloatArray = NDArray[np.float64]
StateFunction = Callable[[float, float, float], Tuple[float, float]]
SimulationResult = Tuple[FloatArray, FloatArray, FloatArray]

def rk4_step(
    dt: float,
    vbias: float,
    v: float,
    i: float,
    f_v: Callable[[float, float], float],
    f_i: Callable[[float, float, float], float],
    perturbation: Optional[float] = None
) -> Tuple[float, float]:
    """
    Perform one step of Runge-Kutta 4th order integration.
    
    Args:
        dt: Time step
        vbias: Bias voltage
        v: Current voltage
        i: Current current
        f_v: Function for dV/dt (takes i, v as arguments)
        f_i: Function for dI/dt (takes vbias, v, i as arguments)
        perturbation: Optional perturbation to add to vbias
        
    Returns:
        Tuple of (new_voltage, new_current)
    """
    # Add perturbation if provided
    if perturbation is not None:
        vbias += perturbation
    
    # RK4 integration
    k1v = f_v(i, v)
    k1i = f_i(vbias, v, i)
    
    k2v = f_v(i + 0.5 * dt * k1i, v + 0.5 * dt * k1v)
    k2i = f_i(vbias, v + 0.5 * dt * k1v, i + 0.5 * dt * k1i)
    
    k3v = f_v(i + 0.5 * dt * k2i, v + 0.5 * dt * k2v)
    k3i = f_i(vbias, v + 0.5 * dt * k2v, i + 0.5 * dt * k2i)
    
    k4v = f_v(i + dt * k3i, v + dt * k3v)
    k4i = f_i(vbias, v + dt * k3v, i + dt * k3i)
    
    # Update state
    new_v = v + (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
    new_i = i + (dt / 6.0) * (k1i + 2 * k2i + 2 * k3i + k4i)
    
    return float(new_v), float(new_i)

def rk4_simulate(
    t_end: float,
    dt: float,
    vbias: Union[float, FloatArray],
    initial_v: float,
    initial_i: float,
    f_v: Callable[[float, float], float],
    f_i: Callable[[float, float, float], float]
) -> SimulationResult:
    """
    Run a complete simulation using Runge-Kutta 4th order method.
    
    Args:
        t_end: End time of the simulation
        dt: Time step size
        vbias: Bias voltage (constant or time-varying)
        initial_v: Initial voltage
        initial_i: Initial current
        f_v: Function for dV/dt (takes i, v as arguments)
        f_i: Function for dI/dt (takes vbias, v, i as arguments)
        
    Returns:
        Tuple of (time, voltage, current) arrays
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
    
    # Convert vbias to array if scalar
    if isinstance(vbias, (float, int)):
        vbias = np.full(n, float(vbias), dtype=np.float64)
    
    # Ensure vbias is float64
    vbias = np.asarray(vbias, dtype=np.float64)
    
    # Simulation loop
    for j in range(1, n):
        v[j], i[j] = rk4_step(dt, vbias[j-1], v[j-1], i[j-1], f_v, f_i)
    
    return t, v, i

def rk4_adaptive(
    t_end: float,
    dt_initial: float,
    vbias: Union[float, FloatArray],
    initial_v: float,
    initial_i: float,
    f_v: Callable[[float, float], float],
    f_i: Callable[[float, float, float], float],
    rel_tol: float = 1e-6,
    abs_tol: float = 1e-8,
    max_dt: float = 1e-9,
    min_dt: float = 1e-12
) -> SimulationResult:
    """
    Run a simulation with adaptive time stepping using Runge-Kutta 4th order method.
    
    Args:
        t_end: End time of the simulation
        dt_initial: Initial time step
        vbias: Bias voltage (constant or time-varying)
        initial_v: Initial voltage
        initial_i: Initial current
        f_v: Function for dV/dt (takes i, v as arguments)
        f_i: Function for dI/dt (takes vbias, v, i as arguments)
        rel_tol: Relative error tolerance
        abs_tol: Absolute error tolerance
        max_dt: Maximum allowed time step
        min_dt: Minimum allowed time step
        
    Returns:
        Tuple of (time, voltage, current) arrays
    """
    # Initialize arrays with reasonable size
    initial_size = int(t_end / dt_initial) * 2
    t = np.zeros(initial_size, dtype=np.float64)
    v = np.zeros(initial_size, dtype=np.float64)
    i = np.zeros(initial_size, dtype=np.float64)
    
    # Set initial conditions
    t[0] = 0.0
    v[0] = initial_v
    i[0] = initial_i
    
    # Convert vbias to array if scalar
    if isinstance(vbias, (float, int)):
        vbias = np.full(initial_size, float(vbias), dtype=np.float64)
    
    # Ensure vbias is float64
    vbias = np.asarray(vbias, dtype=np.float64)
    
    # Initialize time stepping
    current_t = 0.0
    dt = dt_initial
    idx = 1
    
    while current_t < t_end:
        # Store current state
        v_prev = v[idx-1]
        i_prev = i[idx-1]
        
        # Take two half steps
        v_half, i_half = rk4_step(dt/2, vbias[idx-1], v_prev, i_prev, f_v, f_i)
        v_full, i_full = rk4_step(dt, vbias[idx-1], v_prev, i_prev, f_v, f_i)
        
        # Calculate error estimate
        v_error = abs(v_full - v_prev)
        i_error = abs(i_full - i_prev)
        
        # Check if step is acceptable
        if (v_error <= rel_tol * abs(v_prev) + abs_tol and 
            i_error <= rel_tol * abs(i_prev) + abs_tol):
            # Accept step
            t[idx] = current_t + dt
            v[idx] = v_full
            i[idx] = i_full
            current_t += dt
            idx += 1
            
            # Adjust time step
            if v_error > 0 and i_error > 0:
                scale = min(0.9 * (rel_tol * abs(v_prev) + abs_tol) / v_error,
                          0.9 * (rel_tol * abs(i_prev) + abs_tol) / i_error)
                dt = min(max_dt, max(min_dt, dt * scale))
        else:
            # Reject step and try smaller dt
            dt = max(min_dt, dt * 0.5)
        
        # Ensure we don't exceed t_end
        if current_t + dt > t_end:
            dt = t_end - current_t
    
    # Trim arrays to actual size
    t = t[:idx]
    v = v[:idx]
    i = i[:idx]
    
    return t, v, i 