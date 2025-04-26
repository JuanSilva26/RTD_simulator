"""
RTD plotting utilities.

This module provides plotting functionality for RTD simulations, including
IV curves, time series plots, and phase space visualizations.
"""

from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from ..models.curve_analysis import CurveAnalyzer
from numpy.typing import NDArray
from PyQt6.QtWidgets import (QToolBar, QComboBox, QSpinBox,
                           QLabel, QWidget, QHBoxLayout, QPushButton,
                           QButtonGroup, QCheckBox, QLineEdit)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QIcon, QAction
from matplotlib.backends.backend_qt import NavigationToolbar2QT

class PlotToolbar(QToolBar):
    """Enhanced toolbar for plot controls."""
    
    gridChanged = pyqtSignal(bool, bool)  # (major_grid, minor_grid)
    scaleChanged = pyqtSignal(str)  # 'linear' or 'log'
    titleChanged = pyqtSignal(str, str)  # (axis_id, new_title)
    layoutChanged = pyqtSignal(str)  # layout_type
    
    def __init__(self, canvas, parent=None):
        """Initialize the toolbar."""
        super().__init__(parent)
        self.canvas = canvas
        
        # Add matplotlib's built-in navigation toolbar
        self.nav_toolbar = NavigationToolbar2QT(canvas, self)
        self.addWidget(self.nav_toolbar)
        self.addSeparator()
        
        # Layout controls
        layout_widget = QWidget()
        layout_control = QHBoxLayout(layout_widget)
        layout_control.setContentsMargins(0, 0, 0, 0)
        layout_control.setSpacing(4)
        
        layout_label = QLabel("Layout:")
        layout_label.setStyleSheet("color: #e1e1e1;")
        layout_control.addWidget(layout_label)
        
        self.layout_combo = QComboBox()
        self.layout_combo.addItems([
            "Standard",
            "IV Focus",
            "Time Series Focus",
            "Equal Grid"
        ])
        self.layout_combo.setStyleSheet("""
            QComboBox {
                background: #2a2a2a;
                border: 1px solid #404040;
                border-radius: 3px;
                color: #e1e1e1;
                padding: 2px 8px;
                min-width: 120px;
                min-height: 24px;
            }
            QComboBox:hover {
                border-color: #2196F3;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cGF0aCBkPSJNOCwxMkwxMiw4TDQsOEw4LDEyWiIgZmlsbD0iI2UxZTFlMSIvPjwvc3ZnPg==);
            }
            QComboBox QAbstractItemView {
                background: #2a2a2a;
                border: 1px solid #404040;
                selection-background-color: #2196F3;
                selection-color: white;
            }
        """)
        self.layout_combo.currentTextChanged.connect(
            lambda x: self.layoutChanged.emit(x.lower().replace(" ", "_")))
        layout_control.addWidget(self.layout_combo)
        
        self.addWidget(layout_widget)
        self.addSeparator()
        
        # Title editing controls
        title_widget = QWidget()
        title_layout = QHBoxLayout(title_widget)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(4)
        
        title_label = QLabel("Title:")
        title_label.setStyleSheet("color: #e1e1e1;")
        title_layout.addWidget(title_label)
        
        self.plot_selector = QComboBox()
        self.plot_selector.addItems(["IV Curve", "Pulse Wave", "Voltage", "Current"])
        self.plot_selector.setStyleSheet("""
            QComboBox {
                background: #2a2a2a;
                border: 1px solid #404040;
                border-radius: 3px;
                color: #e1e1e1;
                padding: 2px 8px;
                min-width: 100px;
                min-height: 24px;
            }
            QComboBox:hover {
                border-color: #2196F3;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cGF0aCBkPSJNOCwxMkwxMiw4TDQsOEw4LDEyWiIgZmlsbD0iI2UxZTFlMSIvPjwvc3ZnPg==);
            }
            QComboBox QAbstractItemView {
                background: #2a2a2a;
                border: 1px solid #404040;
                selection-background-color: #2196F3;
                selection-color: white;
            }
        """)
        title_layout.addWidget(self.plot_selector)
        
        self.title_edit = QLineEdit()
        self.title_edit.setPlaceholderText("Enter plot title...")
        self.title_edit.setStyleSheet("""
            QLineEdit {
                background: #2a2a2a;
                border: 1px solid #404040;
                border-radius: 3px;
                color: #e1e1e1;
                padding: 2px 8px;
                min-width: 150px;
                min-height: 24px;
            }
            QLineEdit:hover {
                border-color: #2196F3;
            }
            QLineEdit:focus {
                border-color: #2196F3;
                background: #323232;
            }
        """)
        self.title_edit.returnPressed.connect(self._update_plot_title)
        title_layout.addWidget(self.title_edit)
        
        apply_title_btn = QPushButton("Apply")
        apply_title_btn.setStyleSheet("""
            QPushButton {
                background: #2196F3;
                border: none;
                border-radius: 3px;
                color: white;
                padding: 4px 12px;
                min-height: 24px;
            }
            QPushButton:hover {
                background: #42A5F5;
            }
            QPushButton:pressed {
                background: #1976D2;
            }
        """)
        apply_title_btn.clicked.connect(self._update_plot_title)
        title_layout.addWidget(apply_title_btn)
        
        self.addWidget(title_widget)
        self.addSeparator()
        
        # Grid controls
        grid_widget = QWidget()
        grid_layout = QHBoxLayout(grid_widget)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.setSpacing(4)
        
        grid_label = QLabel("Grid:")
        grid_label.setStyleSheet("color: #e1e1e1;")
        grid_layout.addWidget(grid_label)
        
        self.major_grid = QCheckBox("Major")
        self.major_grid.setChecked(True)
        self.major_grid.setStyleSheet("""
            QCheckBox {
                color: #e1e1e1;
                spacing: 5px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                background: #2a2a2a;
                border: 1px solid #404040;
                border-radius: 3px;
            }
            QCheckBox::indicator:checked {
                background: #2196F3;
                border-color: #2196F3;
            }
            QCheckBox::indicator:hover {
                border-color: #42A5F5;
            }
        """)
        self.major_grid.toggled.connect(self._on_grid_changed)
        grid_layout.addWidget(self.major_grid)
        
        self.minor_grid = QCheckBox("Minor")
        self.minor_grid.setStyleSheet(self.major_grid.styleSheet())
        self.minor_grid.toggled.connect(self._on_grid_changed)
        grid_layout.addWidget(self.minor_grid)
        
        self.addWidget(grid_widget)
        self.addSeparator()
        
        # Scale controls
        scale_widget = QWidget()
        scale_layout = QHBoxLayout(scale_widget)
        scale_layout.setContentsMargins(0, 0, 0, 0)
        scale_layout.setSpacing(4)
        
        scale_label = QLabel("Scale:")
        scale_label.setStyleSheet("color: #e1e1e1;")
        scale_layout.addWidget(scale_label)
        
        self.scale_combo = QComboBox()
        self.scale_combo.addItems(["Linear", "Log"])
        self.scale_combo.setStyleSheet("""
            QComboBox {
                background: #2a2a2a;
                border: 1px solid #404040;
                border-radius: 3px;
                color: #e1e1e1;
                padding: 2px 8px;
                min-width: 80px;
                min-height: 24px;
            }
            QComboBox:hover {
                border-color: #2196F3;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cGF0aCBkPSJNOCwxMkwxMiw4TDQsOEw4LDEyWiIgZmlsbD0iI2UxZTFlMSIvPjwvc3ZnPg==);
            }
            QComboBox QAbstractItemView {
                background: #2a2a2a;
                border: 1px solid #404040;
                selection-background-color: #2196F3;
                selection-color: white;
            }
        """)
        self.scale_combo.currentTextChanged.connect(
            lambda x: self.scaleChanged.emit(x.lower()))
        scale_layout.addWidget(self.scale_combo)
        
        self.addWidget(scale_widget)
        
    def _on_grid_changed(self):
        """Handle grid checkbox changes."""
        self.gridChanged.emit(
            self.major_grid.isChecked(),
            self.minor_grid.isChecked()
        )

    def _update_plot_title(self):
        """Update the title of the selected plot."""
        plot_type = self.plot_selector.currentText()
        new_title = self.title_edit.text()
        
        # Map plot type to axis ID
        axis_map = {
            "IV Curve": "iv",
            "Pulse Wave": "pulse",
            "Voltage": "voltage",
            "Current": "current"
        }
        
        axis_id = axis_map.get(plot_type)
        if axis_id:
            self.titleChanged.emit(axis_id, new_title)
            self.title_edit.clear()

class RTDPlotter:
    """Plotting utilities for RTD simulation."""
    
    def __init__(self, figure: Optional[Figure] = None):
        """
        Initialize the plotter.
        
        Args:
            figure: Matplotlib figure to use for plotting
        """
        self.figure = figure or plt.figure(figsize=(12, 8))
        self.analyzer = CurveAnalyzer()
        plt.style.use('default')
        self.toolbar = None
        self.current_layout = "standard"
        self.last_plot_data = None  # Store last plot data for layout updates

    def _create_layout(self, layout_type: str, show_iv: bool = True,
                      show_phase_space: bool = False,
                      show_current: bool = True) -> Dict[str, Any]:
        """Create subplot layout based on selected type."""
        self.figure.clear()
        axes = {}
        
        if layout_type == "standard":
            if show_current:
                gs = GridSpec(3, 2, figure=self.figure)
                axes['iv'] = self.figure.add_subplot(gs[:, 0])
                axes['pulse'] = self.figure.add_subplot(gs[0, 1])
                axes['voltage'] = self.figure.add_subplot(gs[1, 1])
                axes['current'] = self.figure.add_subplot(gs[2, 1])
            else:
                gs = GridSpec(2, 2, figure=self.figure)
                axes['iv'] = self.figure.add_subplot(gs[:, 0])
                axes['pulse'] = self.figure.add_subplot(gs[0, 1])
                axes['voltage'] = self.figure.add_subplot(gs[1, 1])
                    
        elif layout_type == "iv_focus":
            # Show only the IV curve
            axes['iv'] = self.figure.add_subplot(111)
                
        elif layout_type == "time_series_focus":
            if show_current:
                # Create a grid for time series plots only
                gs = GridSpec(3, 1, figure=self.figure, height_ratios=[1, 1, 1])
                axes['pulse'] = self.figure.add_subplot(gs[0])
                axes['voltage'] = self.figure.add_subplot(gs[1])
                axes['current'] = self.figure.add_subplot(gs[2])
            else:
                gs = GridSpec(2, 1, figure=self.figure, height_ratios=[1, 1])
                axes['pulse'] = self.figure.add_subplot(gs[0])
                axes['voltage'] = self.figure.add_subplot(gs[1])
            
            # Adjust spacing between subplots
            self.figure.subplots_adjust(hspace=0.4)
                
        else:  # equal_grid
            if show_current:
                gs = GridSpec(2, 2, figure=self.figure)
                axes['iv'] = self.figure.add_subplot(gs[0, 0])
                axes['pulse'] = self.figure.add_subplot(gs[0, 1])
                axes['voltage'] = self.figure.add_subplot(gs[1, 0])
                axes['current'] = self.figure.add_subplot(gs[1, 1])
            else:
                gs = GridSpec(2, 2, figure=self.figure)
                axes['iv'] = self.figure.add_subplot(gs[0, 0])
                axes['pulse'] = self.figure.add_subplot(gs[0, 1])
                axes['voltage'] = self.figure.add_subplot(gs[1, :])
        
        return axes
        
    def plot_main_view(self, v_range: NDArray, i_values: NDArray,
                      t: NDArray, v: NDArray, i: NDArray,
                      pulse: NDArray,
                      show_iv: bool = True,
                      show_phase_space: bool = False,
                      show_current: bool = True) -> None:
        """
        Plot the main view with IV curve and dynamics.
        
        Args:
            v_range: Voltage array for IV curve
            i_values: Current array for IV curve
            t: Time array for dynamics
            v: Voltage array for dynamics
            i: Current array for dynamics
            pulse: Pulse signal array
            show_iv: Whether to show IV curve plot
            show_phase_space: Whether to show phase space overlaid on IV curve
            show_current: Whether to show current plot
        """
        # Store current plot data
        self.last_plot_data = {
            'v_range': v_range,
            'i_values': i_values,
            't': t,
            'v': v,
            'i': i,
            'pulse': pulse,
            'show_iv': show_iv,
            'show_phase_space': show_phase_space,
            'show_current': show_current
        }
        
        # Create layout based on current layout type
        self.axes = self._create_layout(
            self.current_layout,
            show_iv,
            show_phase_space,
            show_current
        )
        
        # Plot IV curve if enabled
        if show_iv and 'iv' in self.axes:
            ax_iv = self.axes['iv']
            ax_iv.plot(v_range, i_values, 'b-', label='IV Curve')
            
            # Overlay phase space if enabled
            if show_phase_space:
                ax_iv.plot(v, i, 'k--', label='Phase Space', alpha=0.7)
                
            ax_iv.set_xlabel('Voltage (V)')
            ax_iv.set_ylabel('Current (A)')
            ax_iv.set_title('IV Characteristics')
            ax_iv.grid(True)
            ax_iv.legend(loc='upper right')
        
        # Plot pulse
        if 'pulse' in self.axes:
            ax_pulse = self.axes['pulse']
            ax_pulse.plot(t, pulse, 'g-', label='Pulse')
            ax_pulse.set_xlabel('Time (s)')
            ax_pulse.set_ylabel('Amplitude (V)')
            ax_pulse.set_title('Pulse Wave')
            ax_pulse.grid(True)
            ax_pulse.legend(loc='upper right')
        
        # Plot voltage
        if 'voltage' in self.axes:
            ax_v = self.axes['voltage']
            ax_v.plot(t, v, 'b-', label='Voltage')
            ax_v.set_xlabel('Time (s)')
            ax_v.set_ylabel('Voltage (V)')
            ax_v.set_title('Voltage vs Time')
            ax_v.grid(True)
            ax_v.legend(loc='upper right')
        
        # Plot current if enabled
        if show_current and 'current' in self.axes:
            ax_i = self.axes['current']
            ax_i.plot(t, i, 'r-', label='Current')
            ax_i.set_xlabel('Time (s)')
            ax_i.set_ylabel('Current (A)')
            ax_i.set_title('Current vs Time')
            ax_i.grid(True)
            ax_i.legend(loc='upper right')
        
        # Adjust layout with more space
        self.figure.tight_layout(pad=2.0)
        
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
        
    def save_plot(self, filename: str, dpi: int = 1200):
        """
        Save current plot with high quality settings.
        
        Args:
            filename: Path to save the plot
            dpi: Resolution for the saved plot
        """
        self.figure.savefig(filename,
                           dpi=dpi,
                           bbox_inches='tight',
                           facecolor='none',
                           edgecolor='none',
                           pad_inches=0.1)
        
    def plot_dynamics_analysis(self, t: NDArray, v: NDArray, i: NDArray,
                             v_freq: Dict[str, NDArray],
                             i_freq: Dict[str, NDArray]) -> None:
        """
        Plot detailed dynamics analysis.
        
        Args:
            t: Time array
            v: Voltage array
            i: Current array
            v_freq: Voltage frequency analysis results
            i_freq: Current frequency analysis results
        """
        self.figure.clear()
        
        # Create subplots
        gs = GridSpec(2, 2, figure=self.figure)
        
        # Plot voltage analysis
        ax_v = self.figure.add_subplot(gs[0, 0])
        ax_v.plot(t, v, 'b-')
        ax_v.set_xlabel('Time (s)')
        ax_v.set_ylabel('Voltage (V)')
        ax_v.set_title('Voltage Time Series')
        ax_v.grid(True)
        
        ax_v_freq = self.figure.add_subplot(gs[0, 1])
        ax_v_freq.semilogy(v_freq['frequencies'], v_freq['magnitudes'])
        ax_v_freq.set_xlabel('Frequency (Hz)')
        ax_v_freq.set_ylabel('Magnitude')
        ax_v_freq.set_title(f'Voltage Spectrum\nDominant Freq: {v_freq["dominant_frequency"]:.2f} Hz')
        ax_v_freq.grid(True)
        
        # Plot current analysis
        ax_i = self.figure.add_subplot(gs[1, 0])
        ax_i.plot(t, i, 'r-')
        ax_i.set_xlabel('Time (s)')
        ax_i.set_ylabel('Current (A)')
        ax_i.set_title('Current Time Series')
        ax_i.grid(True)
        
        ax_i_freq = self.figure.add_subplot(gs[1, 1])
        ax_i_freq.semilogy(i_freq['frequencies'], i_freq['magnitudes'])
        ax_i_freq.set_xlabel('Frequency (Hz)')
        ax_i_freq.set_ylabel('Magnitude')
        ax_i_freq.set_title(f'Current Spectrum\nDominant Freq: {i_freq["dominant_frequency"]:.2f} Hz')
        ax_i_freq.grid(True)
        
        self.figure.tight_layout()
        
    def create_iv_curve(self, v_range: NDArray, i_values: NDArray) -> None:
        """
        Create IV curve plot with analysis.
        
        Args:
            v_range: Voltage array
            i_values: Current array
        """
        self.figure.clear()
        
        # Create main plot and analysis subplot
        ax1 = self.figure.add_subplot(211)  # IV curve
        ax2 = self.figure.add_subplot(212)  # FFT analysis
        
        # Plot original IV curve
        ax1.plot(v_range, i_values, 'b-', label='IV Curve', alpha=0.7)
        
        # Fit curve and plot
        i_fit, params = self.analyzer.fit_iv_curve(v_range, i_values)
        ax1.plot(v_range, i_fit, 'r--', label=f'Fit (R² = {params["r_squared"]:.3f})')
        
        # Find and plot peaks/valleys
        peaks = self.analyzer.find_peaks_and_valleys(v_range, i_values)
        ax1.plot(peaks['peak_voltages'], peaks['peak_currents'], 'go', label='Peaks')
        ax1.plot(peaks['valley_voltages'], peaks['valley_currents'], 'mo', label='Valleys')
        
        # Calculate and display statistics
        stats = self.analyzer.calculate_statistics(v_range, i_values)
        stats_text = (f"Mean V: {stats['v_mean']:.2f}V\n"
                     f"Mean I: {stats['i_mean']:.2f}A\n"
                     f"Peak-to-Peak V: {stats['peak_to_peak_v']:.2f}V\n"
                     f"Peak-to-Peak I: {stats['peak_to_peak_i']:.2f}A")
        ax1.text(0.02, 0.98, stats_text,
                transform=ax1.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax1.set_xlabel('Voltage (V)')
        ax1.set_ylabel('Current (A)')
        ax1.set_title('IV Curve Analysis')
        ax1.grid(True)
        ax1.legend()
        
        # Plot frequency content
        freq_data = self.analyzer.analyze_frequency(v_range, i_values)
        ax2.semilogy(freq_data['frequencies'], freq_data['magnitudes'])
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Magnitude')
        ax2.set_title('Frequency Analysis')
        ax2.grid(True)
        
        self.figure.tight_layout()
        
    def create_time_series(self, t: NDArray, v: NDArray, i: NDArray) -> None:
        """
        Create time series plots with analysis.
        
        Args:
            t: Time array
            v: Voltage array
            i: Current array
        """
        self.figure.clear()
        
        # Create subplots
        gs = self.figure.add_gridspec(3, 2)
        ax1 = self.figure.add_subplot(gs[0, :])  # Pulse wave
        ax2 = self.figure.add_subplot(gs[1, 0])  # Voltage
        ax3 = self.figure.add_subplot(gs[1, 1])  # Voltage FFT
        ax4 = self.figure.add_subplot(gs[2, 0])  # Current
        ax5 = self.figure.add_subplot(gs[2, 1])  # Current FFT
        
        # Plot pulse wave
        pulse = np.zeros_like(t)
        pulse[t < t[-1]/2] = 1.0
        ax1.plot(t, pulse, 'g-', label='Pulse')
        ax1.set_ylabel('Amplitude')
        ax1.set_title('Pulse Wave')
        ax1.grid(True)
        ax1.legend()
        
        # Plot voltage and its FFT
        ax2.plot(t, v, 'b-', label='Voltage')
        ax2.set_ylabel('Voltage (V)')
        ax2.set_title('Voltage vs Time')
        ax2.grid(True)
        ax2.legend()
        
        v_freq = self.analyzer.analyze_frequency(t, v)
        ax3.semilogy(v_freq['frequencies'], v_freq['magnitudes'])
        ax3.set_xlabel('Frequency (Hz)')
        ax3.set_title('Voltage Spectrum')
        ax3.grid(True)
        
        # Plot current and its FFT
        ax4.plot(t, i, 'r-', label='Current')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Current (A)')
        ax4.set_title('Current vs Time')
        ax4.grid(True)
        ax4.legend()
        
        i_freq = self.analyzer.analyze_frequency(t, i)
        ax5.semilogy(i_freq['frequencies'], i_freq['magnitudes'])
        ax5.set_xlabel('Frequency (Hz)')
        ax5.set_title('Current Spectrum')
        ax5.grid(True)
        
        self.figure.tight_layout()
        
    def create_phase_space(self, v: NDArray, i: NDArray) -> None:
        """
        Create phase space plot with analysis.
        
        Args:
            v: Voltage array
            i: Current array
        """
        self.figure.clear()
        
        # Create main plot and analysis subplot
        ax1 = self.figure.add_subplot(211)  # Phase space
        ax2 = self.figure.add_subplot(212)  # Statistics
        
        # Plot phase space
        ax1.plot(v, i, 'g-', label='Phase Space')
        ax1.set_xlabel('Voltage (V)')
        ax1.set_ylabel('Current (A)')
        ax1.set_title('Phase Space Analysis')
        ax1.grid(True)
        ax1.legend()
        
        # Calculate and display statistics
        stats = self.analyzer.calculate_statistics(v, i)
        stats_text = (f"Voltage Statistics:\n"
                     f"  Mean: {stats['v_mean']:.2f}V\n"
                     f"  Std Dev: {stats['v_std']:.2f}V\n"
                     f"  Range: [{stats['v_min']:.2f}, {stats['v_max']:.2f}]V\n\n"
                     f"Current Statistics:\n"
                     f"  Mean: {stats['i_mean']:.2f}A\n"
                     f"  Std Dev: {stats['i_std']:.2f}A\n"
                     f"  Range: [{stats['i_min']:.2f}, {stats['i_max']:.2f}]A")
        
        ax2.text(0.5, 0.5, stats_text,
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax2.transAxes,
                bbox=dict(boxstyle='round', facecolor='white'))
        ax2.set_xticks([])
        ax2.set_yticks([])
        
        self.figure.tight_layout()
        
    def show(self) -> None:
        """Display the plot."""
        self.figure.canvas.draw()
        
    def plot_iv_analysis(self, v_range: NDArray, i_values: NDArray, 
                        analysis_data: Dict[str, Any]) -> None:
        """
        Plot IV curve analysis in a separate window.
        
        Args:
            v_range: Voltage array
            i_values: Current array
            analysis_data: Dictionary containing analysis results
        """
        self.figure.clear()
        
        # Create grid layout
        gs = GridSpec(2, 2, figure=self.figure)
        
        # IV curve with fit
        ax_iv = self.figure.add_subplot(gs[0, :])
        ax_iv.plot(v_range, i_values, 'b-', label='IV Curve')
        
        if 'iv_fit' in analysis_data:
            v_fit, i_fit, params = analysis_data['iv_fit']
            ax_iv.plot(v_fit, i_fit, 'r--', 
                      label=f'Fit (R² = {params["r_squared"]:.3f})')
            
        if 'peaks' in analysis_data:
            peaks = analysis_data['peaks']
            ax_iv.plot(peaks['peak_voltages'], peaks['peak_currents'], 
                      'go', label='Peaks')
            ax_iv.plot(peaks['valley_voltages'], peaks['valley_currents'], 
                      'mo', label='Valleys')
            
        ax_iv.set_xlabel('Voltage (V)')
        ax_iv.set_ylabel('Current (A)')
        ax_iv.set_title('IV Curve Analysis')
        ax_iv.grid(True)
        ax_iv.legend(loc='upper right')
        
        # Statistics
        ax_stats = self.figure.add_subplot(gs[1, 0])
        stats = self.analyzer.calculate_statistics(v_range, i_values)
        stats_text = (f"Voltage Statistics:\n"
                     f"  Mean: {stats['v_mean']:.2f}V\n"
                     f"  Std Dev: {stats['v_std']:.2f}V\n"
                     f"  Range: [{stats['v_min']:.2f}, {stats['v_max']:.2f}]V\n\n"
                     f"Current Statistics:\n"
                     f"  Mean: {stats['i_mean']:.2f}A\n"
                     f"  Std Dev: {stats['i_std']:.2f}A\n"
                     f"  Range: [{stats['i_min']:.2f}, {stats['i_max']:.2f}]A")
        
        ax_stats.text(0.5, 0.5, stats_text,
                     horizontalalignment='center',
                     verticalalignment='center',
                     transform=ax_stats.transAxes,
                     bbox=dict(boxstyle='round', facecolor='white'))
        ax_stats.set_xticks([])
        ax_stats.set_yticks([])
        ax_stats.set_title('Statistics')
        
        # Frequency analysis
        ax_freq = self.figure.add_subplot(gs[1, 1])
        freq_data = self.analyzer.analyze_frequency(v_range, i_values)
        ax_freq.semilogy(freq_data['frequencies'], freq_data['magnitudes'])
        ax_freq.set_xlabel('Frequency (Hz)')
        ax_freq.set_ylabel('Magnitude')
        ax_freq.set_title('Frequency Analysis')
        ax_freq.grid(True)
        
        self.figure.tight_layout()
        
    def setup_toolbar(self, canvas, parent=None):
        """Create and setup the plot toolbar."""
        self.toolbar = PlotToolbar(canvas, parent)
        self.toolbar.gridChanged.connect(self._update_grid)
        self.toolbar.scaleChanged.connect(self._update_scale)
        self.toolbar.titleChanged.connect(self._update_title)
        self.toolbar.layoutChanged.connect(self._update_layout)
        return self.toolbar
        
    def _update_grid(self, major: bool, minor: bool):
        """Update grid visibility."""
        for ax in self.figure.get_axes():
            ax.grid(major, which='major', linestyle='-', alpha=0.2)
            ax.grid(minor, which='minor', linestyle=':', alpha=0.1)
            self.figure.canvas.draw_idle()
            
    def _update_scale(self, scale: str):
        """Update axis scale."""
        for ax in self.figure.get_axes():
            if scale == 'log':
                ax.set_yscale('log')
                # Ensure positive values for log scale
                ylim = ax.get_ylim()
                if ylim[0] <= 0:
                    ax.set_ylim(1e-6, ylim[1])
            else:
                ax.set_yscale('linear')
            self.figure.canvas.draw_idle()
            
    def _update_title(self, axis_id: str, new_title: str):
        """Update the title of the specified axis."""
        ax = self.axes.get(axis_id)
        if ax:
            ax.set_title(new_title)
            self.figure.canvas.draw_idle()
            
    def _update_layout(self, layout_type: str):
        """Update the plot layout."""
        self.current_layout = layout_type
        # Re-plot with current data
        if self.last_plot_data is not None:
            self.plot_main_view(
                v_range=self.last_plot_data['v_range'],
                i_values=self.last_plot_data['i_values'],
                t=self.last_plot_data['t'],
                v=self.last_plot_data['v'],
                i=self.last_plot_data['i'],
                pulse=self.last_plot_data['pulse'],
                show_iv=self.last_plot_data['show_iv'],
                show_phase_space=self.last_plot_data['show_phase_space'],
                show_current=self.last_plot_data['show_current']
            )
            self.figure.canvas.draw_idle()
            
    def _handle_export(self, filename: str, format_type: str):
        """Handle export requests."""
        self._setup_export_plot_style(self.figure.gca())
        self.save_plot(filename, dpi=1200) 