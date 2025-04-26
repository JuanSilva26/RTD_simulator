"""
Common test fixtures for RTD simulator tests.
"""

import pytest
import numpy as np
from rtd_simulator.models.rtd_model import SimplifiedRTDModel, SchulmanRTDModel

@pytest.fixture
def simplified_model():
    """Create a SimplifiedRTDModel instance with default parameters."""
    return SimplifiedRTDModel()

@pytest.fixture
def schulman_model():
    """Create a SchulmanRTDModel instance with default parameters."""
    return SchulmanRTDModel()

@pytest.fixture
def simulation_params():
    """Common simulation parameters."""
    return {
        't_end': 1e-7,  # 100 ns
        'dt': 1e-10,    # 0.1 ns
        'vbias': 0.5    # 0.5 V
    }

@pytest.fixture
def voltage_array():
    """Create a test voltage array."""
    return np.linspace(-1, 1, 100)

@pytest.fixture
def perturbation_params():
    """Common perturbation parameters."""
    return {
        'perturbation_type': 'sine',
        'amplitude': 0.1,
        'frequency': 1e9,
        'start_time': 0.0,
        'duration': None
    }

@pytest.fixture
def mock_qapp(qtbot):
    """Create a Qt application instance for widget tests."""
    return qtbot 