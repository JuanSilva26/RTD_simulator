"""
Main window for RTD simulation GUI.
"""

from typing import Optional, Dict, Any, cast
import numpy as np
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                            QPushButton, QLabel, QFrame, QToolBar, QComboBox, QFileDialog)
from PyQt6.QtCore import QTimer, Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ..controller.rtd_controller import RTDController
from ..model.rtd_model import SimplifiedRTDModel
from ..model.curve_analysis import CurveAnalyzer
from .plotting import RTDPlotter
from .custom_widgets import StatusBar, PresetManager, ModernSliderSpinBox, CollapsibleSection
from .dialogs import ExportDialog
from .parameter_sections import (RTDParameterSection, SimulationParameterSection,
                               PulseParameterSection)
from .animation_window import AnimationWindow

class MainWindow(QMainWindow):
    """Main window for RTD simulation application."""
    
    def __init__(self, parent: Optional[QWidget] = None):
        """
        Initialize the main window.
        
        Args:
            parent: Parent widget (optional)
        """
        super().__init__(parent)
        
        # Create figure first
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        # Initialize plotter and controller
        self.plotter = RTDPlotter(figure=self.figure)
        self.controller = RTDController(plotter=self.plotter, parent=self)
        
        # Setup real-time update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_plot)
        self.update_timer.setInterval(100)  # Update every 100ms
        
        # Initialize UI
        self.init_ui()
        
        # Run initial simulation with default parameters
        self._on_parameters_changed()
        
        # Start timer after initial simulation
        self.update_timer.start()
        
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle('RTD Simulation Platform')
        self.setGeometry(100, 100, 1400, 800)
        
        # Create menubar
        self.create_menu_bar()
        
        # Create toolbar
        self.create_toolbar()
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)
        layout.setSpacing(10)
        
        # Create collapsible sidebar
        sidebar = QWidget()
        sidebar.setMaximumWidth(350)
        sidebar.setMinimumWidth(250)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(5, 5, 5, 5)
        sidebar_layout.setSpacing(5)
        
        # Add parameter sections
        self.create_parameter_sections(sidebar_layout)
        
        # Add stretch to push sections to top
        sidebar_layout.addStretch()
        
        # Create plot area
        plot_area = self.create_plot_area()
        
        # Add sidebar and plot area to main layout
        layout.addWidget(sidebar)
        
        # Add separator line
        line = QFrame()
        line.setFrameShape(QFrame.Shape.VLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(line)
        
        layout.addWidget(plot_area)
        
        # Set layout proportions
        layout.setStretch(0, 0)  # Sidebar (no stretch)
        layout.setStretch(1, 0)  # Line (no stretch)
        layout.setStretch(2, 1)  # Plot area (stretch)
        
        # Create status bar
        self.status_bar = StatusBar()
        self.setStatusBar(self.status_bar)
        
    def create_menu_bar(self):
        """Create the menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        export_action = file_menu.addAction('Export...')
        export_action.triggered.connect(self.show_export_dialog)
        
        # View menu
        view_menu = menubar.addMenu('View')
        view_menu.addAction('Reset Layout')
        
        # Analysis menu
        analysis_menu = menubar.addMenu('Analysis')
        analysis_menu.addAction('IV Analysis').triggered.connect(self.show_iv_analysis)
        analysis_menu.addAction('Dynamics Analysis').triggered.connect(self.show_dynamics_analysis)
        
    def create_toolbar(self):
        """Create the toolbar."""
        # Remove any existing toolbars
        for toolbar in self.findChildren(QToolBar):
            self.removeToolBar(toolbar)
        
        self.toolbar = QToolBar()
        self.addToolBar(self.toolbar)
        
        # Add export button
        export_btn = QPushButton("Export...")
        export_btn.setToolTip("Export simulation data and plots")
        export_btn.clicked.connect(self.show_export_dialog)
        self.toolbar.addWidget(export_btn)
        
        self.toolbar.addSeparator()
        
        # Add analysis buttons
        iv_analysis_btn = QPushButton("IV Analysis")
        iv_analysis_btn.setToolTip("Show detailed IV curve analysis")
        iv_analysis_btn.clicked.connect(self.show_iv_analysis)
        self.toolbar.addWidget(iv_analysis_btn)
        
        dynamics_btn = QPushButton("Dynamics Analysis")
        dynamics_btn.setToolTip("Show detailed dynamics analysis")
        dynamics_btn.clicked.connect(self.show_dynamics_analysis)
        self.toolbar.addWidget(dynamics_btn)
        
        # Add animation button
        animation_btn = QPushButton("Animation")
        animation_btn.clicked.connect(self.show_animation)
        self.toolbar.addWidget(animation_btn)
        
    def create_parameter_sections(self, layout: QVBoxLayout):
        """Create parameter sections in the sidebar."""
        # Add preset manager at the top
        self.preset_manager = PresetManager(self)
        self.preset_manager.presetLoaded.connect(self._load_preset)
        layout.addWidget(self.preset_manager)
        
        # Add separator line
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(line)
        
        # RTD Parameters section
        self.rtd_section = RTDParameterSection()
        self.rtd_section.modelChanged.connect(self._on_model_changed)
        self.rtd_section.model_combo.currentTextChanged.connect(
            lambda text: self._on_parameters_changed()
        )
        self.rtd_section.m_spin.valueChanged.connect(
            lambda value: self._on_parameters_changed()
        )
        self.rtd_section.r_spin.valueChanged.connect(
            lambda value: self._on_parameters_changed()
        )
        layout.addWidget(self.rtd_section)
        
        # Simulation Parameters section
        self.sim_section = SimulationParameterSection()
        self.sim_section.t_end_spin.valueChanged.connect(
            lambda value: self._on_parameters_changed()
        )
        self.sim_section.dt_spin.valueChanged.connect(
            lambda value: self._on_parameters_changed()
        )
        self.sim_section.vbias_spin.valueChanged.connect(
            lambda value: self._on_parameters_changed()
        )
        layout.addWidget(self.sim_section)
        
        # Pulse Parameters section
        self.pulse_section = PulseParameterSection()
        self.pulse_section.pulse_type_combo.currentTextChanged.connect(
            lambda text: self._on_parameters_changed()
        )
        self.pulse_section.pulse_amplitude_spin.valueChanged.connect(
            lambda value: self._on_parameters_changed()
        )
        self.pulse_section.pulse_freq_spin.valueChanged.connect(
            lambda value: self._on_parameters_changed()
        )
        self.pulse_section.duty_cycle_spin.valueChanged.connect(
            lambda value: self._on_parameters_changed()
        )
        self.pulse_section.offset_spin.valueChanged.connect(
            lambda value: self._on_parameters_changed()
        )
        layout.addWidget(self.pulse_section)
        
    def create_plot_area(self) -> QWidget:
        """Create the plot area widget."""
        plot_widget = QWidget()
        layout = QVBoxLayout(plot_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Add plot toolbar
        layout.addWidget(self.toolbar)
        
        # Add canvas
        layout.addWidget(self.canvas)
        return plot_widget
        
    def _on_parameters_changed(self) -> None:
        """Handle parameter changes."""
        try:
            # Get current parameters
            rtd_params = self.rtd_section.get_parameters()
            sim_params = self.sim_section.get_parameters()
            pulse_params = self.pulse_section.get_parameters()
            
            # Update status
            self.status_bar.set_status("Updating simulation...")
            
            # Update controller parameters
            self.controller.update_parameters(**rtd_params)
            
            # Convert parameters to proper types
            t_end = float(sim_params['t_end'])
            dt = float(sim_params['dt'])
            vbias = float(sim_params['vbias'])
            pulse_amplitude = float(pulse_params['pulse_amplitude'])
            pulse_frequency = float(pulse_params['pulse_frequency'])
            duty_cycle = float(pulse_params['duty_cycle']) / 100.0  # Convert from percentage
            pulse_type = str(pulse_params['pulse_type']).lower()
            offset = float(pulse_params['offset'])
            
            # Run simulation with new parameters
            self.controller.run_simulation(
                t_end=t_end,
                dt=dt,
                vbias=vbias,
                pulse_amplitude=pulse_amplitude,
                pulse_frequency=pulse_frequency,
                duty_cycle=duty_cycle,
                pulse_type=pulse_type,
                offset=offset
            )
            
            # Update status
            self.status_bar.set_status("Ready")
            
        except Exception as e:
            print(f"Error updating parameters: {e}")
            self.status_bar.set_status(f"Error: {str(e)}")
        
    def _update_plot(self) -> None:
        """Update the plot with current simulation data."""
        try:
            if self.controller.get_simulation_data() is not None:
                self.controller.plot_main_view({
                    'show_iv': True,
                    'show_phase_space': False,
                    'show_current': True
                })
                self.canvas.draw()
        except Exception as e:
            print(f"Error updating plot: {e}")
            self.status_bar.set_status(f"Error: {str(e)}")
        
    def show_export_dialog(self):
        """Show export options dialog."""
        dialog = ExportDialog(self)
        if dialog.exec() == ExportDialog.DialogCode.Accepted:
            options = dialog.get_export_options()
            self.controller.export_data_and_plots(options)
            
    def show_iv_analysis(self):
        """Show IV curve analysis window."""
        self.controller.show_iv_analysis()
        
    def show_dynamics_analysis(self):
        """Show dynamics analysis window."""
        self.controller.show_dynamics_analysis()
        
    def show_animation(self):
        """Show the animation window."""
        if self.controller._current_simulation_data is not None:
            t = self.controller._current_simulation_data['time']
            pulse = self.controller._current_simulation_data['pulse']
            voltage = self.controller._current_simulation_data['voltage']
            
            animation_window = AnimationWindow(t, pulse, voltage, self)
            animation_window.exec()
            
    def get_parameters(self) -> dict:
        """Get current parameter values."""
        params = {}
        
        # RTD parameters
        params.update(self.rtd_section.get_parameters())
        
        # Simulation parameters
        params.update(self.sim_section.get_parameters())
        
        # Pulse parameters
        params.update(self.pulse_section.get_parameters())
        
        return params
        
    def _load_preset(self, params: dict):
        """Load parameters from a preset."""
        # RTD parameters
        if 'm' in params:
            self.rtd_section.m_spin.setValue(params['m'])
        if 'r' in params:
            self.rtd_section.r_spin.setValue(params['r'])
            
        # Simulation parameters
        if 't_end' in params:
            self.sim_section.t_end_spin.setValue(params['t_end'])
        if 'dt' in params:
            self.sim_section.dt_spin.setValue(params['dt'])
        if 'vbias' in params:
            self.sim_section.vbias_spin.setValue(params['vbias'])
            
        # Pulse parameters
        if 'pulse_type' in params:
            self.pulse_section.pulse_type_combo.setCurrentText(params['pulse_type'])
        if 'pulse_amplitude' in params:
            self.pulse_section.pulse_amplitude_spin.setValue(params['pulse_amplitude'])
        if 'pulse_frequency' in params:
            self.pulse_section.pulse_freq_spin.setValue(params['pulse_frequency'])
        if 'duty_cycle' in params:
            self.pulse_section.duty_cycle_spin.setValue(params['duty_cycle'])
        if 'offset' in params:
            self.pulse_section.offset_spin.setValue(params['offset'])
            
        # Update simulation
        self._update_plot() 

    def closeEvent(self, event) -> None:
        """Handle window close event."""
        self.update_timer.stop()
        super().closeEvent(event) 

    def _on_model_changed(self, model_type: str) -> None:
        """Handle RTD model type changes."""
        # Update simulation parameter ranges
        self.sim_section.update_ranges(model_type)
        # Update simulation
        self._on_parameters_changed() 