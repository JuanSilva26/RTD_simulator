"""
Tests for RTD model implementations.
"""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose
from rtd_simulator.models.base import PerturbationType
from rtd_simulator.models.rtd_model import SimplifiedRTDModel, SchulmanRTDModel

def test_simplified_model_initialization(simplified_model):
    """Test SimplifiedRTDModel initialization with default parameters."""
    assert simplified_model.k == 5.92
    assert simplified_model.h == 15.76
    assert simplified_model.w == 2.259
    assert simplified_model.m == 0.078
    assert simplified_model.r == 1.0
    assert simplified_model.v == -1.1
    
def test_schulman_model_initialization(schulman_model):
    """Test SchulmanRTDModel initialization with default parameters."""
    assert schulman_model.a == 6.715e-4
    assert schulman_model.b == 6.499e-2
    assert schulman_model.c == 9.709e-2
    assert schulman_model.d == 2.213e-2
    assert schulman_model.h == 1.664e-4
    assert schulman_model.n1 == 3.106e-2
    assert schulman_model.n2 == 1.721e-2
    assert schulman_model.C == 1.0e-12
    assert schulman_model.L == 1.0e-9
    assert schulman_model.R == 50.0
    assert schulman_model.T == 300.0
    assert schulman_model.v == 0.0

def test_simplified_iv_characteristic(simplified_model, voltage_array):
    """Test simplified IV characteristic calculation."""
    current = simplified_model.iv_characteristic(voltage_array)
    
    # Test shape and type
    assert isinstance(current, np.ndarray)
    assert current.shape == voltage_array.shape
    
    # Test specific points
    assert current[voltage_array == 0].item() == pytest.approx(0.0, abs=1e-10)
    
    # Test monotonicity in linear regions
    linear_region = voltage_array > 1.0
    if any(linear_region):
        assert np.all(np.diff(current[linear_region]) > 0)

def test_schulman_iv_characteristic(schulman_model, voltage_array):
    """Test Schulman IV characteristic calculation."""
    current = schulman_model.iv_characteristic(voltage_array)
    
    # Test shape and type
    assert isinstance(current, np.ndarray)
    assert current.shape == voltage_array.shape
    
    # Test specific points
    assert current[voltage_array == 0].item() == pytest.approx(0.0, abs=1e-10)
    
    # Test physical constraints
    assert np.all(np.isfinite(current))  # No infinities or NaNs

def test_simplified_model_step(simplified_model, simulation_params):
    """Test single step integration for simplified model."""
    dt = simulation_params['dt']
    vbias = simulation_params['vbias']
    
    # Take one step
    v_new, i_new = simplified_model.step(dt, vbias)
    
    # Test output types
    assert isinstance(v_new, float)
    assert isinstance(i_new, float)
    
    # Test physical constraints
    assert np.isfinite(v_new)
    assert np.isfinite(i_new)

def test_schulman_model_step(schulman_model, simulation_params):
    """Test single step integration for Schulman model."""
    dt = simulation_params['dt']
    vbias = simulation_params['vbias']
    
    # Take one step
 