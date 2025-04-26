"""
RTD simulation controller.
"""

from typing import Optional, Tuple, Dict, Any, cast, Type
import os
import numpy as np
import pandas as pd
from datetime import datetime
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QWidget, QFileDialog,
                            QToolBar, QMainWindow, QPushButton, QHBoxLayout, QLabel, QDoubleSpinBox, QSpinBox,
                            QMessageBox)
from PyQt6.QtCore import Qt, QObject, pyqtSignal
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from ..models.base import get_rtd_model, RTDModel, PerturbationType
from ..models.curve_analysis import CurveAnalyzer
from ..views.plotting import RTDPlotter

class RTDController(QObject):
    """Controller for RTD simulation and analysis."""
    
    # Signals
    simulation_started = pyqtSignal()
    simulation_finished = pyqtSignal()
    simulation_error = pyqtSignal(str)
    progress_updated = pyqtSignal(int)
    
    def __init__(self, plotter: RTDPlotter, parent: Optional[QObject] = None):
        """
        Initialize the controller.
        
        Args:
            plotter: RTDPlotter instance for visualization
            parent: Parent QObject (optional)
        """
        super().__init__(parent)
        self.plotter = plotter
        
        # Initialize with a default model (Simplified)
        simplified_model = get_rtd_model("Simplified")
        self.model = simplified_model() if simplified_model else None
        
        self._is_running = False
        self.canvas = None
        self.toolbar = None
        self.analyzer = CurveAnalyzer()
        self._current_simulation_data: Optional[Dict[str, np.ndarray]] = None
                
    def start_simulation(self, model: RTDModel, params: Dict[str, Any]) -> None:
        """
        Start a new simulation.
        
        Args:
            model: RTD model to simulate
            params: Simulation parameters
        """
        self.model = model
        self._is_running = True
        self.simulation_started.emit()
        
        try:
            # Run simulation
            t, v, i = model.simulate(
                t_end=params['t_end'],
                dt=params['dt'],
                vbias=params['vbias']
            )
            
            # Update plot
            self.plotter.plot_main_view(
                v_range=params['v_range'],
                i_values=model.iv_characteristic(params['v_range']),
                t=t, v=v, i=i,
                pulse=params.get('pulse', None)
            )
            
            self.simulation_finished.emit()
            
        except Exception as e:
            self.simulation_error.emit(str(e))
            self._is_running = False
            
    def stop_simulation(self) -> None:
        """Stop the current simulation."""
        self._is_running = False
        
    def is_running(self) -> bool:
        """Check if a simulation is currently running."""
        return self._is_running
                
    def update_parameters(self, **kwargs) -> None:
        """
        Update RTD model parameters.
        
        Args:
            **kwargs: Parameter names and values to update
        """
        # Handle model switching
        if 'model_type' in kwargs:
            model_type = kwargs.pop('model_type')
            model_cls = get_rtd_model(model_type)
            if model_cls is not None and not isinstance(self.model, model_cls):
                self.model = model_cls()
        
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
        if isinstance(self.model, get_rtd_model("Schulman")):
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
            
        if self.model is None:
            raise ValueError("No RTD model available. Initialize a model first.")
            
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
        fit_result = self.analyzer.fit_iv_curve(v_range, i_values)
        poly = fit_result.get('polynomial')
        if poly is not None:
            i_fit = poly(v_range)
            
            # Calculate R^2 (coefficient of determination)
            ss_tot = np.sum((i_values - np.mean(i_values))**2)
            ss_res = np.sum((i_values - i_fit)**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # Add R^2 to the fit parameters
            fit_params = fit_result.copy()
            fit_params['r_squared'] = r_squared
        else:
            i_fit = i_values  # fallback if polynomial fit fails
            fit_params = {}
            
        peaks = self.analyzer.find_peaks_and_valleys(v_range, i_values)
        
        # Create peak analysis structure
        peak_voltages = [p[0] for p in peaks['peaks']]
        peak_currents = [p[1] for p in peaks['peaks']]
        valley_voltages = [v[0] for v in peaks['valleys']]
        valley_currents = [v[1] for v in peaks['valleys']]
        
        # Plot analysis
        analysis_data = {
            'iv_fit': (v_range, i_fit, fit_params),
            'peaks': {
                'peak_voltages': peak_voltages,
                'peak_currents': peak_currents,
                'valley_voltages': valley_voltages,
                'valley_currents': valley_currents
            }
        }
        
        analysis_plotter.plot_iv_analysis(v_range, i_values, analysis_data)
        
        # Show dialog
        dialog.exec()
        
    def show_advanced_iv_analysis(self) -> None:
        """Show advanced IV curve analysis with detailed peak detection."""
        if self._current_simulation_data is None:
            raise ValueError("No simulation data available. Run simulation first.")
            
        if self.model is None:
            raise ValueError("No RTD model available. Initialize a model first.")
            
        # Create analysis window
        dialog = QDialog()
        dialog.setWindowTitle("IV Analysis")
        dialog.setGeometry(100, 100, 1000, 800)  # Larger window for detailed analysis
        
        # Create layout
        layout = QVBoxLayout(dialog)
        
        # Create control panel for peak detection parameters
        control_panel = QWidget()
        control_layout = QHBoxLayout(control_panel)
        
        # Add widgets for adjusting peak detection parameters
        height_label = QLabel("Height:")
        height_spin = QDoubleSpinBox()
        height_spin.setRange(0.0, 100.0)
        height_spin.setValue(0.0)  # Auto
        height_spin.setSpecialValueText("Auto")
        height_spin.setSuffix("%")
        height_spin.setToolTip("Minimum peak height (percentage of range)")
        
        distance_label = QLabel("Distance:")
        distance_spin = QSpinBox()
        distance_spin.setRange(0, 200)
        distance_spin.setValue(0)  # Auto
        distance_spin.setSpecialValueText("Auto")
        distance_spin.setToolTip("Minimum samples between peaks")
        
        prominence_label = QLabel("Prominence:")
        prominence_spin = QDoubleSpinBox()
        prominence_spin.setRange(0.0, 100.0)
        prominence_spin.setValue(0.0)  # Auto
        prominence_spin.setSpecialValueText("Auto")
        prominence_spin.setSuffix("%")
        prominence_spin.setToolTip("Minimum prominence (percentage of range)")
        
        rel_height_label = QLabel("Rel. Height:")
        rel_height_spin = QDoubleSpinBox()
        rel_height_spin.setRange(0.1, 1.0)
        rel_height_spin.setValue(0.5)
        rel_height_spin.setSingleStep(0.1)
        rel_height_spin.setToolTip("Relative height at which peak width is measured")
        
        # Add button to update analysis
        update_btn = QPushButton("Update Analysis")
        
        # Add widgets to layout
        control_layout.addWidget(height_label)
        control_layout.addWidget(height_spin)
        control_layout.addWidget(distance_label)
        control_layout.addWidget(distance_spin)
        control_layout.addWidget(prominence_label)
        control_layout.addWidget(prominence_spin)
        control_layout.addWidget(rel_height_label)
        control_layout.addWidget(rel_height_spin)
        control_layout.addWidget(update_btn)
        control_layout.addStretch()
        
        # Create toolbar section
        toolbar_panel = QWidget()
        toolbar_layout = QHBoxLayout(toolbar_panel)
        
        # Add export button
        export_btn = QPushButton("Export Results")
        export_btn.setToolTip("Export analysis results to CSV")
        toolbar_layout.addWidget(export_btn)
        
        # Add help button
        help_btn = QPushButton("Help")
        help_btn.setToolTip("Show information about peak detection parameters")
        toolbar_layout.addWidget(help_btn)
        
        toolbar_layout.addStretch()
        
        # Create figure for analysis plots
        fig = Figure(figsize=(10, 8), tight_layout=True)
        canvas = FigureCanvasQTAgg(fig)
        
        # Create navigation toolbar
        nav_toolbar = NavigationToolbar(canvas, dialog)
        
        # Add all components to main layout
        layout.addWidget(control_panel)
        layout.addWidget(toolbar_panel)
        layout.addWidget(nav_toolbar)
        layout.addWidget(canvas)
        
        # Create plotter for analysis
        analysis_plotter = RTDPlotter(fig)
        
        # Generate IV curve data - check the model type safely
        schulman_model = get_rtd_model("Schulman")
        is_schulman = schulman_model is not None and isinstance(self.model, schulman_model)
        
        if is_schulman:
            v_range = np.linspace(0.0, 4.5, 1000)  # 0 to 4.5V for Schulman, matches main view
        else:
            v_range = np.linspace(-3.0, 3.0, 1000)  # Default range for Simplified
            
        i_values = self.model.iv_characteristic(v_range)
        
        # Function to update the analysis with current parameters
        def update_analysis():
            # Get parameter values (or None for auto)
            height = None
            if height_spin.value() > 0:
                height = height_spin.value() / 100.0 * (np.max(i_values) - np.min(i_values))
                
            distance = None
            if distance_spin.value() > 0:
                distance = distance_spin.value()
                
            prominence = None
            if prominence_spin.value() > 0:
                prominence = prominence_spin.value() / 100.0 * (np.max(i_values) - np.min(i_values))
                
            # Create parameters dict
            peak_params = {
                'height': height,
                'distance': distance,
                'prominence': prominence,
                'rel_height': rel_height_spin.value(),
            }
            
            # Plot analysis with updated parameters
            analysis_plotter.plot_advanced_iv_analysis(v_range, i_values, peak_params)
            canvas.draw()
        
        # Function to export results
        def export_results():
            # Get save location from user
            file_dialog = QFileDialog()
            save_path, _ = file_dialog.getSaveFileName(
                dialog,
                "Save Analysis Results",
                "",
                "CSV Files (*.csv);;All Files (*)"
            )
            
            if not save_path:
                return
                
            # Ensure file has .csv extension
            if not save_path.lower().endswith('.csv'):
                save_path += '.csv'
                
            # Get current peak detection parameters
            height = None
            if height_spin.value() > 0:
                height = height_spin.value() / 100.0 * (np.max(i_values) - np.min(i_values))
                
            distance = None
            if distance_spin.value() > 0:
                distance = distance_spin.value()
                
            prominence = None
            if prominence_spin.value() > 0:
                prominence = prominence_spin.value() / 100.0 * (np.max(i_values) - np.min(i_values))
                
            # Create parameters dict and run analysis
            peak_params = {
                'height': height,
                'distance': distance,
                'prominence': prominence,
                'rel_height': rel_height_spin.value(),
            }
            
            # Run analysis to get results for export
            results = self.analyzer.advanced_peak_detection(v_range, i_values, **peak_params)
            
            # Create DataFrames for peaks and valleys
            if results['peaks']:
                peaks_df = pd.DataFrame(results['peaks'])
                peaks_df.to_csv(save_path, index=False)
                
                # If we have valleys, save them to a separate file
                if results['valleys']:
                    valley_path = save_path.replace('.csv', '_valleys.csv')
                    valleys_df = pd.DataFrame(results['valleys'])
                    valleys_df.to_csv(valley_path, index=False)
        
        # Function to show help dialog
        def show_help():
            help_text = """
            <h3>Peak Detection Parameters</h3>
            <p><b>Height:</b> Minimum peak height required. Peaks below this height will be ignored.</p>
            <p><b>Distance:</b> Minimum number of samples between neighboring peaks.</p>
            <p><b>Prominence:</b> Prominence of a peak measures how much it stands out due to its height and position relative to other peaks.</p>
            <p><b>Rel. Height:</b> Relative height at which peak width is measured (between 0-1).</p>
            
            <h3>Understanding the Analysis</h3>
            <p>The top plot shows the IV curve with detected peaks and valleys. Peak colors indicate sharpness (red = sharp, blue = rounded).</p>
            <p>Width markers show the width of each peak at the specified relative height.</p>
            <p>The middle plots show the first and second derivatives, which help in understanding peak characteristics.</p>
            <p>The bottom section provides detailed information about each detected peak and valley.</p>
            """
            
            msg = QMessageBox()
            msg.setWindowTitle("Peak Detection Help")
            msg.setTextFormat(Qt.TextFormat.RichText)
            msg.setText(help_text)
            msg.exec()
        
        # Connect signals
        update_btn.clicked.connect(update_analysis)
        export_btn.clicked.connect(export_results)
        help_btn.clicked.connect(show_help)
        
        # Initial plot
        update_analysis()
        
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