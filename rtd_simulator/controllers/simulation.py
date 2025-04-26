"""
Simulation controller for RTD models.
Handles simulation execution in background threads and coordinates between model and view.
"""

from typing import Optional, Dict, Any, Tuple, Union
import numpy as np
from numpy.typing import NDArray
from PyQt6.QtCore import QObject, pyqtSignal, QThread
from ..models.base import RTDModel, PerturbationType

class SimulationWorker(QThread):
    """Worker thread for running RTD simulations."""
    
    # Signals for progress and results
    progress = pyqtSignal(int)  # Progress percentage
    result = pyqtSignal(tuple)  # (time, voltage, current) arrays
    error = pyqtSignal(str)     # Error message if simulation fails
    
    def __init__(self, model: RTDModel, t_end: float, dt: float, vbias: float,
                 perturbation_params: Optional[Dict[str, Any]] = None):
        """
        Initialize simulation worker.
        
        Args:
            model: RTD model instance
            t_end: End time for simulation
            dt: Time step
            vbias: Bias voltage
            perturbation_params: Optional parameters for perturbation signal
        """
        super().__init__()
        self.model = model
        self.t_end = t_end
        self.dt = dt
        self.vbias = vbias
        self.perturbation_params = perturbation_params
        self._is_running = True
        
    def run(self):
        """Execute simulation in background thread."""
        try:
            # Create time array for progress calculation
            n_steps = int(self.t_end / self.dt)
            last_progress = 0
            
            # Add perturbation if specified
            if self.perturbation_params:
                t = np.arange(0, self.t_end, self.dt)
                perturbation = self.model.create_perturbation(t, **self.perturbation_params)
                vbias_array = self.vbias + perturbation
            else:
                vbias_array = self.vbias
            
            # Run simulation with progress updates
            t, v, i = [], [], []
            current_t = 0.0
            current_idx = 0
            
            while current_t < self.t_end and self._is_running:
                # Get current vbias value
                current_vbias = float(vbias_array[current_idx] if isinstance(vbias_array, np.ndarray) else vbias_array)
                
                # Perform one step
                new_v, new_i = self.model.step(self.dt, current_vbias)
                
                # Store results
                t.append(current_t)
                v.append(new_v)
                i.append(new_i)
                
                # Update progress
                current_t += self.dt
                current_idx += 1
                progress = int(100 * current_t / self.t_end)
                if progress > last_progress:
                    self.progress.emit(progress)
                    last_progress = progress
            
            # Convert to numpy arrays and emit result
            if self._is_running:  # Only emit if not cancelled
                result = (np.array(t), np.array(v), np.array(i))
                self.result.emit(result)
                
        except Exception as e:
            self.error.emit(str(e))
    
    def stop(self):
        """Stop the simulation."""
        self._is_running = False

class SimulationController(QObject):
    """
    Controller for RTD simulations.
    Manages simulation execution and coordinates between model and view.
    """
    
    # Signals for view updates
    simulation_started = pyqtSignal()
    simulation_finished = pyqtSignal()
    simulation_error = pyqtSignal(str)
    simulation_progress = pyqtSignal(int)
    new_results = pyqtSignal(tuple)  # (time, voltage, current) arrays
    
    def __init__(self):
        """Initialize simulation controller."""
        super().__init__()
        self.current_worker: Optional[SimulationWorker] = None
        
    def start_simulation(self, model: RTDModel, t_end: float, dt: float, vbias: float,
                        perturbation_params: Optional[Dict[str, Any]] = None):
        """
        Start a new simulation in a background thread.
        
        Args:
            model: RTD model instance
            t_end: End time for simulation
            dt: Time step
            vbias: Bias voltage
            perturbation_params: Optional parameters for perturbation signal
        """
        # Stop any running simulation
        self.stop_simulation()
        
        # Create and configure worker
        self.current_worker = SimulationWorker(model, t_end, dt, vbias, perturbation_params)
        self.current_worker.progress.connect(self.simulation_progress.emit)
        self.current_worker.result.connect(self._handle_results)
        self.current_worker.error.connect(self.simulation_error.emit)
        self.current_worker.finished.connect(self.simulation_finished.emit)
        
        # Start simulation
        self.simulation_started.emit()
        self.current_worker.start()
        
    def stop_simulation(self):
        """Stop the current simulation if running."""
        if self.current_worker and self.current_worker.isRunning():
            self.current_worker.stop()
            self.current_worker.wait()
            self.current_worker = None
            
    def _handle_results(self, results: Tuple[NDArray[np.float64], ...]):
        """Process and emit simulation results."""
        self.new_results.emit(results) 