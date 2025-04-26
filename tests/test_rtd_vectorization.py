import numpy as np
import pytest
from rtd_simulator.model.rtd_model import SimplifiedRTDModel, SchulmanRTDModel

def test_vectorized_simulation_matches_original():
    """Test that vectorized simulation produces identical results to original implementation."""
    # Create model with default parameters
    model = SimplifiedRTDModel()
    
    # Simulation parameters
    t_end = 1.0
    dt = 0.01
    vbias = 1.0
    
    # Run original simulation
    t_orig, v_orig, i_orig = model.simulate(t_end, dt, vbias)
    
    # Run vectorized simulation
    t_vec, v_vec, i_vec = model.simulate_vectorized(t_end, dt, vbias)
    
    # Compare results
    np.testing.assert_allclose(t_orig, t_vec, rtol=1e-10)
    np.testing.assert_allclose(v_orig, v_vec, rtol=1e-10)
    np.testing.assert_allclose(i_orig, i_vec, rtol=1e-10)

def test_vectorized_simulation_with_array_vbias():
    """Test vectorized simulation with time-varying bias voltage."""
    model = SimplifiedRTDModel()
    
    t_end = 1.0
    dt = 0.01
    t = np.arange(0, t_end, dt)
    vbias = np.sin(2 * np.pi * t)  # Sinusoidal bias voltage
    
    # Run simulations
    t_orig, v_orig, i_orig = model.simulate(t_end, dt, vbias)
    t_vec, v_vec, i_vec = model.simulate_vectorized(t_end, dt, vbias)
    
    # Compare results
    np.testing.assert_allclose(t_orig, t_vec, rtol=1e-10)
    np.testing.assert_allclose(v_orig, v_vec, rtol=1e-10)
    np.testing.assert_allclose(i_orig, i_vec, rtol=1e-10)

def test_vectorized_simulation_performance():
    """Test that vectorized simulation is faster than original."""
    import time
    
    model = SimplifiedRTDModel()
    t_end = 1.0
    dt = 0.001  # Smaller dt for more steps
    vbias = 1.0
    
    # Time original implementation
    start = time.time()
    model.simulate(t_end, dt, vbias)
    original_time = time.time() - start
    
    # Time vectorized implementation
    start = time.time()
    model.simulate_vectorized(t_end, dt, vbias)
    vectorized_time = time.time() - start
    
    # Vectorized should be significantly faster
    assert vectorized_time < original_time * 0.5  # At least 2x faster

def test_schulman_vectorized_simulation_matches_original():
    """Test that Schulman vectorized simulation produces identical results to original implementation."""
    # Create model with default parameters
    model = SchulmanRTDModel()
    
    # Simulation parameters (use smaller time scales for Schulman model)
    t_end = 1e-6
    dt = 1e-9
    vbias = 0.1
    
    # Run original simulation
    t_orig, v_orig, i_orig = model.simulate(t_end, dt, vbias)
    
    # Run vectorized simulation
    t_vec, v_vec, i_vec = model.simulate_vectorized(t_end, dt, vbias)
    
    # Compare results
    np.testing.assert_allclose(t_orig, t_vec, rtol=1e-10)
    np.testing.assert_allclose(v_orig, v_vec, rtol=1e-10)
    np.testing.assert_allclose(i_orig, i_vec, rtol=1e-10)

def test_schulman_vectorized_simulation_with_array_vbias():
    """Test Schulman vectorized simulation with time-varying bias voltage."""
    model = SchulmanRTDModel()
    
    t_end = 1e-6
    dt = 1e-9
    t = np.arange(0, t_end, dt)
    vbias = 0.1 * np.sin(2 * np.pi * 1e9 * t)  # 1 GHz sinusoidal bias voltage
    
    # Run simulations
    t_orig, v_orig, i_orig = model.simulate(t_end, dt, vbias)
    t_vec, v_vec, i_vec = model.simulate_vectorized(t_end, dt, vbias)
    
    # Compare results
    np.testing.assert_allclose(t_orig, t_vec, rtol=1e-10)
    np.testing.assert_allclose(v_orig, v_vec, rtol=1e-10)
    np.testing.assert_allclose(i_orig, i_vec, rtol=1e-10)

def test_schulman_vectorized_simulation_performance():
    """Test that Schulman vectorized simulation is faster than original."""
    import time
    
    model = SchulmanRTDModel()
    t_end = 1e-6
    dt = 1e-9
    vbias = 0.1
    
    # Time original implementation
    start = time.time()
    model.simulate(t_end, dt, vbias)
    original_time = time.time() - start
    
    # Time vectorized implementation
    start = time.time()
    model.simulate_vectorized(t_end, dt, vbias)
    vectorized_time = time.time() - start
    
    # Vectorized should be significantly faster
    assert vectorized_time < original_time * 0.5  # At least 2x faster 