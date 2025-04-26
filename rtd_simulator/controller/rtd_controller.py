"""
RTD Controller module for managing Model-View interaction.
"""

from typing import Optional, Tuple, Dict, Any, cast
import os
import numpy as np
import pandas as pd
from datetime import datetime
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QWidget, QFileDialog,
                            QToolBar, QMainWindow, QPushButton)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from ..model.rtd_model import RTDModel, SimplifiedRTDModel, SchulmanRTDModel, PerturbationType
from ..model.curve_analysis import CurveAnalyzer
from ..view.plotting import RTDPlotter

class RTDController:
    """Controller class for managing RTD simulation and visualization."""
    
    def __init__(self, plotter: RTDPlotter, parent: Optional[QWidget] = None):
        """
        Initialize the controller with model and view components.
        
        Args:
            plotter: RTDPlotter instance
            parent: Parent widget (optional)
        """
        self.model: RTDModel = SimplifiedRTDModel()  # Start with simplified model
        self.plotter = plotter
        self.parent = parent
        self.canvas = None
        self.toolbar = None
        self.analyzer = CurveAnalyzer()
        self._current_simulation_data: Optional[Dict[str, np.ndarray]] = None
                
    def update_parameters(self, **kwargs) -> None:
        """
        Update RTD model parameters.
        
        Args:
            **kwargs: Parameter names and values to update
        """
        # Handle model switching
        if 'model_type' in kwargs:
            model_type = kwargs.pop('model_type')
            if model_type == "Simplified" and not isinstance(self.model, SimplifiedRTDModel):
                self.model = SimplifiedRTDModel()
            elif model_type == "Schulman" and not isinstance(self.model, SchulmanRTDModel):
                self.model = SchulmanRTDModel()
        
        # Update remaining parameters
        for param, value in kwargs.items():
            if hasattr(self.model, param):
                setattr(self.model, param, value)
                
    def run_simulation(self, t_end: float, dt: float, vbias: float,
                     pulse_amplitude: float = 1.0,
                     pulse_frequency: float = 0.04,
                     duty_cycle: float = 0.5,
                     pulse_type: str = "square",
                     offset: float = 0.0) -> None:
        """
        Run simulation and store results.
        
        Args:
            t_end: Simulation end time [s]
            dt: Time step [s]
            vbias: Bias voltage [V]
            pulse_amplitude: Amplitude of the pulse signal [V]
            pulse_frequency: Frequency of the pulse signal [Hz]
            duty_cycle: Duty cycle of the pulse (0-1)
            pulse_type: Type of pulse signal ("square", "sine", "triangle", "sawtooth")
            offset: DC offset of the pulse signal [V]
        """
        # Create time array
        t = np.arange(0, t_end, dt)
        
        # Create pulse signal based on type
        period = 1.0 / pulse_frequency
        pulse = np.zeros_like(t)
        
        if pulse_type == "square":
            # Calculate pulse timing
            for i in range(len(t)):
                cycle_time = t[i] % period
                if cycle_time < period * duty_cycle:
                    pulse[i] = pulse_amplitude
                    
        elif pulse_type == "sine":
            pulse = pulse_amplitude * np.sin(2 * np.pi * pulse_frequency * t)
            
        elif pulse_type == "triangle":
            # Normalized time within period
            t_norm = (t % period) / period
            # Triangle wave generation
            pulse = pulse_amplitude * (2 * np.abs(2 * t_norm - 1) - 1)
            
        elif pulse_type == "sawtooth":
            # Normalized time within period
            t_norm = (t % period) / period
            # Sawtooth wave generation
            pulse = pulse_amplitude * (2 * t_norm - 1)
            
        # Add offset
        pulse += offset
        
        # Run simulation with pulse perturbation
        t, v, i = self.model.simulate(t_end=t_end, dt=dt, vbias=vbias + pulse)
        
        # Store results
        self._current_simulation_data = {
            'time': t,
            'voltage': v,
            'current': i,
            'pulse': pulse
        }
        
    def get_simulation_data(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Get current simulation data.
        
        Returns:
            Dictionary containing time, voltage, and current arrays
        """
        return self._current_simulation_data
        
    def plot_main_view(self, analysis_options: Dict[str, bool]) -> None:
        """
        Plot the main view with IV curve and dynamics.
        
        Args:
            analysis_options: Dictionary of analysis options to show/hide
        """
        if self._current_simulation_data is None:
            raise ValueError("No simulation data available. Run simulation first.")
            
        # Get data
        t = self._current_simulation_data['time']
        v = self._current_simulation_data['voltage']
        i = self._current_simulation_data['current']
        pulse = self._current_simulation_data['pulse']
        
        # Create IV curve data with appropriate range based on model type
        if isinstance(self.model, SchulmanRTDModel):
            v_range = np.linspace(0.0, 4.5, 1000)  # 0 to 4.5V for Schulman
        else:
            v_range = np.linspace(-3.0, 3.0, 1000)  # -3 to 3V for Simplified
            
        i_values = self.model.iv_characteristic(v_range)
        
        # Plot everything
        self.plotter.plot_main_view(
            v_range=v_range,
            i_values=i_values,
            t=t,
            v=v,
            i=i,
            pulse=pulse,
            show_iv=analysis_options.get('show_iv', True),
            show_phase_space=analysis_options.get('show_phase_space', False),
            show_current=analysis_options.get('show_current', True)
        )
        
    def _setup_export_plot_style(self, ax):
        """Set up high-quality plot style for exports."""
        ax.set_facecolor('none')  # Transparent background
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(direction='out', length=6, width=1.5)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        ax.legend(loc='upper right', frameon=False)
        
    def export_data_and_plots(self, options: Dict[str, Any]) -> None:
        """
        Export data and plots based on user options.
        
        Args:
            options: Dictionary containing export options
        """
        if self._current_simulation_data is None:
            raise ValueError("No simulation data available. Run simulation first.")
            
        # Get save location from user
        file_dialog = QFileDialog()
        save_dir = file_dialog.getExistingDirectory(
            None,
            "Select Export Directory",
            "",
            QFileDialog.Option.ShowDirsOnly
        )
        
        if not save_dir:
            return
            
        # Create timestamp for file names
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export data if selected
        if options['export_data']:
            data_path = os.path.join(save_dir, f"rtd_data_{timestamp}.csv")
            df = pd.DataFrame({
                'Time (s)': self._current_simulation_data['time'],
                'Voltage (V)': self._current_simulation_data['voltage'],
                'Current (A)': self._current_simulation_data['current'],
                'Pulse (V)': self._current_simulation_data['pulse']
            })
            df.to_csv(data_path, index=False)
            
        # Export plots if selected
        if options['export_plots']:
            plot_format = options['plot_format'].split('(')[1].strip(').*')
            
            # Get current figure state
            current_fig = self.plotter.figure
            
            # Create IV curve data
            v_range = np.linspace(-3.0, 3.0, 1000)
            i_values = self.model.iv_characteristic(v_range)
            
            # Set up figure style for exports
            plt_params = {
                'figsize': (10, 8),
                'dpi': 600,
                'facecolor': 'none',
                'edgecolor': 'none'
            }
            
            if "All Plots" in options['selected_plots'] or not options['selected_plots']:
                # Save current view with high quality
                plot_path = os.path.join(save_dir, f"rtd_plots_{timestamp}{plot_format}")
                current_fig.savefig(plot_path, 
                                  dpi=600,
                                  bbox_inches='tight',
                                  facecolor='none',
                                  edgecolor='none',
                                  pad_inches=0.1)
            else:
                # Save individual plots
                if "IV Characteristics" in options['selected_plots']:
                    fig = Figure(**plt_params)
                    ax = fig.add_subplot(111)
                    ax.plot(v_range, i_values, 'b-', label='IV Curve', linewidth=2)
                    ax.set_xlabel('Voltage (V)', fontsize=12)
                    ax.set_ylabel('Current (A)', fontsize=12)
                    ax.set_title('IV Characteristics', fontsize=14, pad=20)
                    self._setup_export_plot_style(ax)
                    plot_path = os.path.join(save_dir, f"iv_curve_{timestamp}{plot_format}")
                    fig.savefig(plot_path, bbox_inches='tight', pad_inches=0.1)
                    
                if "Pulse Wave" in options['selected_plots']:
                    fig = Figure(**plt_params)
                    ax = fig.add_subplot(111)
                    ax.plot(self._current_simulation_data['time'],
                           self._current_simulation_data['pulse'],
                           'g-', label='Pulse', linewidth=2)
                    ax.set_xlabel('Time (s)', fontsize=12)
                    ax.set_ylabel('Amplitude (V)', fontsize=12)
                    ax.set_title('Pulse Wave', fontsize=14, pad=20)
                    self._setup_export_plot_style(ax)
                    plot_path = os.path.join(save_dir, f"pulse_wave_{timestamp}{plot_format}")
                    fig.savefig(plot_path, bbox_inches='tight', pad_inches=0.1)
                    
                if "Voltage vs Time" in options['selected_plots']:
                    fig = Figure(**plt_params)
                    ax = fig.add_subplot(111)
                    ax.plot(self._current_simulation_data['time'],
                           self._current_simulation_data['voltage'],
                           'b-', label='Voltage', linewidth=2)
                    ax.set_xlabel('Time (s)', fontsize=12)
                    ax.set_ylabel('Voltage (V)', fontsize=12)
                    ax.set_title('Voltage vs Time', fontsize=14, pad=20)
                    self._setup_export_plot_style(ax)
                    plot_path = os.path.join(save_dir, f"voltage_time_{timestamp}{plot_format}")
                    fig.savefig(plot_path, bbox_inches='tight', pad_inches=0.1)
                    
                if "Current vs Time" in options['selected_plots']:
                    fig = Figure(**plt_params)
                    ax = fig.add_subplot(111)
                    ax.plot(self._current_simulation_data['time'],
                           self._current_simulation_data['current'],
                           'r-', label='Current', linewidth=2)
                    ax.set_xlabel('Time (s)', fontsize=12)
                    ax.set_ylabel('Current (A)', fontsize=12)
                    ax.set_title('Current vs Time', fontsize=14, pad=20)
                    self._setup_export_plot_style(ax)
                    plot_path = os.path.join(save_dir, f"current_time_{timestamp}{plot_format}")
                    fig.savefig(plot_path, bbox_inches='tight', pad_inches=0.1)
            
            # Restore main view
            self.plot_main_view({
                'show_iv': True,
                'show_phase_space': True,
                'show_current': True
            })
            
    def show_iv_analysis(self) -> None:
        """Show IV curve analysis in a new window."""
        if self._current_simulation_data is None:
            raise ValueError("No simulation data available. Run simulation first.")
            
        # Create analysis window
        dialog = QDialog()
        dialog.setWindowTitle("IV Curve Analysis")
        dialog.setGeometry(200, 200, 800, 600)
        
        # Create layout
        layout = QVBoxLayout(dialog)
        
        # Create figure for analysis plots
        fig = Figure(figsize=(8, 6))
        canvas = FigureCanvasQTAgg(fig)
        layout.addWidget(canvas)
        
        # Create plotter for analysis
        analysis_plotter = RTDPlotter(fig)
        
        # Create IV curve data
        v_range = np.linspace(-3.0, 3.0, 1000)
        i_values = self.model.iv_characteristic(v_range)
        
        # Get analysis results
        i_fit, params = self.analyzer.fit_iv_curve(v_range, i_values)
        peaks = self.analyzer.find_peaks_and_valleys(v_range, i_values)
        
        # Plot analysis
        analysis_data = {
            'iv_fit': (v_range, i_fit, params),
            'peaks': peaks
        }
        
        analysis_plotter.plot_iv_analysis(v_range, i_values, analysis_data)
        
        # Show dialog
        dialog.exec()
        
    def show_dynamics_analysis(self) -> None:
        """Show detailed dynamics analysis in a new window."""
        if self._current_simulation_data is None:
            raise ValueError("No simulation data available. Run simulation first.")
            
        # Create analysis window
        dialog = QDialog()
        dialog.setWindowTitle("Dynamics Analysis")
        dialog.setGeometry(200, 200, 800, 600)
        
        # Create layout
        layout = QVBoxLayout(dialog)
        
        # Create figure for analysis plots
        fig = Figure(figsize=(8, 6))
        canvas = FigureCanvasQTAgg(fig)
        layout.addWidget(canvas)
        
        # Create plotter for analysis
        analysis_plotter = RTDPlotter(fig)
        
        # Get data
        t = self._current_simulation_data['time']
        v = self._current_simulation_data['voltage']
        i = self._current_simulation_data['current']
        
        # Get frequency analysis
        v_freq = self.analyzer.analyze_frequency(t, v)
        i_freq = self.analyzer.analyze_frequency(t, i)
        
        # Plot analysis
        analysis_plotter.plot_dynamics_analysis(t, v, i, v_freq, i_freq)
        
        # Show dialog
        dialog.exec() 