"""
Parameter section components for the RTD simulation platform.

This module contains the parameter sections used in the sidebar of the main window.
"""

from typing import Callable, Dict, Optional
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QFormLayout, QHBoxLayout,
                            QComboBox, QFrame, QLabel, QGridLayout)
from PyQt6.QtCore import Qt, pyqtSignal
from .custom_widgets import ModernSliderSpinBox, CollapsibleSection, InfoButton

class RTDParameterSection(CollapsibleSection):
    """Section for RTD model parameters."""
    
    # Add signal for model changes
    modelChanged = pyqtSignal(str)
    
    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize the RTD parameter section.
        
        Args:
            parent: Optional parent widget
        """
        super().__init__("RTD Parameters", parent)
        self._init_ui()
        
    def _init_ui(self) -> None:
        """Initialize the section's user interface."""
        # Model selection
        model_layout = QFormLayout()
        
        # Create a horizontal layout for combo box and info button
        model_row = QWidget()
        model_row_layout = QHBoxLayout(model_row)
        model_row_layout.setContentsMargins(0, 0, 0, 0)
        model_row_layout.setSpacing(4)
        
        self.model_combo = QComboBox()
        self.model_combo.addItems(["Simplified", "Schulman"])
        self.model_combo.setToolTip("Select RTD model type")
        model_row_layout.addWidget(self.model_combo)
        
        # Create info button for fixed parameters
        fixed_params_content = (
            "Fixed Parameters:\n"
            "a = 6.715e-4 A\n"
            "b = 6.499e-2 V\n"
            "c = 9.709e-2 V\n"
            "d = 2.213e-2 V\n"
            "h = 1.664e-4 A\n"
            "n1 = 3.106e-2\n"
            "n2 = 1.721e-2\n"
            "T = 300.0 K"
        )
        self.fixed_params_button = InfoButton("Schulman Model Fixed Parameters", fixed_params_content)
        model_row_layout.addWidget(self.fixed_params_button)
        self.fixed_params_button.hide()  # Hide initially since Simplified is default
        
        model_layout.addRow("Model Type:", model_row)
        
        model_widget = QWidget()
        model_widget.setLayout(model_layout)
        self.addWidget(model_widget)
        
        # Add separator line
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        self.addWidget(line)
        
        # Simplified model parameters
        self.simplified_params = QWidget()
        simplified_layout = QVBoxLayout(self.simplified_params)
        simplified_layout.setContentsMargins(0, 0, 0, 0)
        
        self.m_spin = ModernSliderSpinBox(
            "Stiffness", "m",
            "Stiffness parameter of the RTD model",
            0.01, 1.0, 0.001, 3
        )
        self.m_spin.setValue(0.078)
        simplified_layout.addWidget(self.m_spin)
        
        self.r_spin = ModernSliderSpinBox(
            "Resistance", "r",
            "Resistance parameter of the RTD model",
            0.1, 10.0, 0.1, 2
        )
        self.r_spin.setValue(1.0)
        simplified_layout.addWidget(self.r_spin)
        
        self.addWidget(self.simplified_params)
        
        # Schulman model parameters
        self.schulman_params = QWidget()
        schulman_layout = QVBoxLayout(self.schulman_params)
        schulman_layout.setContentsMargins(0, 0, 0, 0)
        
        # Circuit parameters
        circuit_group = QWidget()
        circuit_layout = QVBoxLayout(circuit_group)
        circuit_layout.setContentsMargins(0, 0, 0, 0)
        
        # Add a label for circuit parameters
        circuit_label = QLabel("Circuit Parameters")
        circuit_label.setStyleSheet("font-weight: bold;")
        circuit_layout.addWidget(circuit_label)
        
        # Capacitance with unit selector
        cap_widget = QWidget()
        cap_layout = QHBoxLayout(cap_widget)
        cap_layout.setContentsMargins(0, 0, 0, 0)
        cap_layout.setSpacing(8)
        
        self.C_spin = ModernSliderSpinBox(
            "Capacitance", "",  # Empty unit as we'll use the combo box
            "Device capacitance",
            0.01, 1000.0, 0.01, 2
        )
        self.C_spin.setValue(1.0)
        cap_layout.addWidget(self.C_spin, stretch=1)
        
        self.C_unit_combo = QComboBox()
        self.C_unit_combo.addItems(["pF", "nF", "µF", "mF"])
        self.C_unit_combo.setCurrentText("pF")
        self.C_unit_combo.setFixedWidth(60)
        cap_layout.addWidget(self.C_unit_combo)
        
        circuit_layout.addWidget(cap_widget)
        
        # Inductance with unit selector
        ind_widget = QWidget()
        ind_layout = QHBoxLayout(ind_widget)
        ind_layout.setContentsMargins(0, 0, 0, 0)
        ind_layout.setSpacing(8)
        
        self.L_spin = ModernSliderSpinBox(
            "Inductance", "",  # Empty unit as we'll use the combo box
            "Circuit inductance",
            0.01, 1000.0, 0.01, 2
        )
        self.L_spin.setValue(1.0)
        ind_layout.addWidget(self.L_spin, stretch=1)
        
        self.L_unit_combo = QComboBox()
        self.L_unit_combo.addItems(["pH", "nH", "µH", "mH"])
        self.L_unit_combo.setCurrentText("nH")
        self.L_unit_combo.setFixedWidth(60)
        ind_layout.addWidget(self.L_unit_combo)
        
        circuit_layout.addWidget(ind_widget)
        
        # Resistance (updated range)
        self.R_spin = ModernSliderSpinBox(
            "Resistance", "Ω",
            "Circuit resistance",
            0.01, 50.0, 0.01, 2
        )
        self.R_spin.setValue(1.0)
        circuit_layout.addWidget(self.R_spin)
        
        schulman_layout.addWidget(circuit_group)
        
        self.addWidget(self.schulman_params)
        self.schulman_params.hide()  # Hide Schulman parameters initially
        
        # Connect model selection signal
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        
    def _on_model_changed(self, model_type: str) -> None:
        """Handle model type changes.
        
        Args:
            model_type: Selected model type
        """
        if model_type == "Simplified":
            self.simplified_params.show()
            self.schulman_params.hide()
            self.fixed_params_button.hide()
        else:
            self.simplified_params.hide()
            self.schulman_params.show()
            self.fixed_params_button.show()
        
        # Emit signal for model change
        self.modelChanged.emit(model_type)
            
    def get_parameters(self) -> Dict[str, float | str]:
        """Get current parameter values.
        
        Returns:
            Dictionary containing model parameters
        """
        params = {
            'model_type': self.model_combo.currentText()
        }
        
        if params['model_type'] == "Simplified":
            params.update({
                'm': self.m_spin.value(),
                'r': self.r_spin.value()
            })
        else:
            # Get capacitance with unit conversion
            c_value = self.C_spin.value()
            c_unit = self.C_unit_combo.currentText()
            c_scale = {
                'pF': 1e-12,
                'nF': 1e-9,
                'µF': 1e-6,
                'mF': 1e-3
            }[c_unit]
            
            # Get inductance with unit conversion
            l_value = self.L_spin.value()
            l_unit = self.L_unit_combo.currentText()
            l_scale = {
                'pH': 1e-12,
                'nH': 1e-9,
                'µH': 1e-6,
                'mH': 1e-3
            }[l_unit]
            
            # Fixed parameters for Schulman model
            params.update({
                'a': 6.715e-4,
                'b': 6.499e-2,
                'c': 9.709e-2,
                'd': 2.213e-2,
                'h': 1.664e-4,
                'n1': 3.106e-2,
                'n2': 1.721e-2,
                'T': 300.0,
                # Adjustable circuit parameters with proper unit conversion
                'C': c_value * c_scale,  # Convert to F
                'L': l_value * l_scale,  # Convert to H
                'R': self.R_spin.value()
            })
            
        return params

class SimulationParameterSection(CollapsibleSection):
    """Section for simulation parameters."""
    
    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize the simulation parameter section.
        
        Args:
            parent: Optional parent widget
        """
        super().__init__("Simulation Parameters", parent)
        self._init_ui()
        
    def _init_ui(self) -> None:
        """Initialize the section's user interface."""
        # Time unit selection for Schulman model
        self.time_unit_widget = QWidget()
        time_unit_layout = QFormLayout(self.time_unit_widget)
        self.time_unit_combo = QComboBox()
        self.time_unit_combo.addItems(["Nanoseconds", "Microseconds", "Milliseconds"])
        self.time_unit_combo.setToolTip("Select time unit for simulation")
        time_unit_layout.addRow("Time Unit:", self.time_unit_combo)
        self.time_unit_widget.hide()  # Hide initially
        self.addWidget(self.time_unit_widget)
        
        # End Time control
        self.t_end_spin = ModernSliderSpinBox(
            "End Time", "s",
            "Simulation end time",
            10.0, 10000.0, 10.0, 1
        )
        self.t_end_spin.setValue(100.0)
        self.addWidget(self.t_end_spin)
        
        # Time Step control
        self.dt_spin = ModernSliderSpinBox(
            "Time Step", "s",
            "Simulation time step",
            0.001, 1.0, 0.001, 3
        )
        self.dt_spin.setValue(0.01)
        self.addWidget(self.dt_spin)
        
        # Bias Voltage control
        self.vbias_spin = ModernSliderSpinBox(
            "Bias Voltage", "V",
            "DC bias voltage",
            -3.0, 3.0, 0.1, 2
        )
        self.vbias_spin.setValue(0.0)
        self.addWidget(self.vbias_spin)
        
        # Connect time unit change signal
        self.time_unit_combo.currentTextChanged.connect(self._on_time_unit_changed)
        
    def _on_time_unit_changed(self, unit: str) -> None:
        """Handle time unit changes.
        
        Args:
            unit: Selected time unit
        """
        if unit == "Nanoseconds":
            # End Time: 0.01ns to 1000ns
            self.t_end_spin.setRange(0.01, 1000.0)
            self.t_end_spin.setDecimals(2)
            self.t_end_spin.setStep(0.01)
            self.t_end_spin.setValue(1.0)
            self.t_end_spin.unit = "ns"
            
        elif unit == "Microseconds":
            # End Time: 0.01µs to 1000µs
            self.t_end_spin.setRange(0.01, 1000.0)
            self.t_end_spin.setDecimals(2)
            self.t_end_spin.setStep(0.01)
            self.t_end_spin.setValue(1.0)
            self.t_end_spin.unit = "µs"
            
        else:  # Milliseconds
            # End Time: 0.01ms to 1000ms
            self.t_end_spin.setRange(0.01, 1000.0)
            self.t_end_spin.setDecimals(2)
            self.t_end_spin.setStep(0.01)
            self.t_end_spin.setValue(1.0)
            self.t_end_spin.unit = "ms"
        
    def update_ranges(self, model_type: str) -> None:
        """Update parameter ranges based on model type.
        
        Args:
            model_type: Selected model type ("Simplified" or "Schulman")
        """
        if model_type == "Schulman":
            self.time_unit_widget.show()
            # Set initial time unit to Nanoseconds
            self.time_unit_combo.setCurrentText("Nanoseconds")
            self._on_time_unit_changed("Nanoseconds")
            
            # Time Step: Fixed in nanoseconds (0.01ns to 100ns)
            self.dt_spin.setRange(0.01, 100.0)
            self.dt_spin.setDecimals(2)
            self.dt_spin.setStep(0.01)
            self.dt_spin.setValue(0.1)
            self.dt_spin.unit = "ns"
            
            # Bias Voltage: 0V to 4.5V (matching IV characteristic range)
            self.vbias_spin.setRange(0.0, 4.5)
            self.vbias_spin.setDecimals(3)
            self.vbias_spin.setStep(0.001)
            self.vbias_spin.setValue(3.0)  # Default to 3V
        else:
            self.time_unit_widget.hide()
            self.t_end_spin.unit = "s"
            self.dt_spin.unit = "s"
            
            self.t_end_spin.setRange(10.0, 10000.0)
            self.t_end_spin.setDecimals(1)
            self.t_end_spin.setStep(10.0)
            self.t_end_spin.setValue(100.0)
            
            self.dt_spin.setRange(0.001, 1.0)
            self.dt_spin.setDecimals(3)
            self.dt_spin.setStep(0.001)
            self.dt_spin.setValue(0.01)
            
            self.vbias_spin.setRange(-3.0, 3.0)
            self.vbias_spin.setDecimals(2)
            self.vbias_spin.setStep(0.1)
            self.vbias_spin.setValue(0.0)
            
    def get_parameters(self) -> Dict[str, float]:
        """Get current parameter values.
        
        Returns:
            Dictionary containing simulation parameters
        """
        # Convert time values based on selected unit for Schulman model
        if self.time_unit_widget.isVisible():
            unit = self.time_unit_combo.currentText()
            end_time_scale_factor = {
                "Nanoseconds": 1e-9,
                "Microseconds": 1e-6,
                "Milliseconds": 1e-3
            }[unit]
            
            return {
                't_end': self.t_end_spin.value() * end_time_scale_factor,
                'dt': self.dt_spin.value() * 1e-9,  # Always convert from ns to seconds
                'vbias': self.vbias_spin.value()
            }
        else:
            return {
                't_end': self.t_end_spin.value(),
                'dt': self.dt_spin.value(),
                'vbias': self.vbias_spin.value()
            }

class PulseParameterSection(CollapsibleSection):
    """Section for pulse signal parameters."""
    
    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize the pulse parameter section.
        
        Args:
            parent: Optional parent widget
        """
        super().__init__("Pulse Parameters", parent)
        self._init_ui()
        
    def _init_ui(self) -> None:
        """Initialize the section's user interface."""
        # Pulse type selection
        pulse_type_layout = QFormLayout()
        self.pulse_type_combo = QComboBox()
        self.pulse_type_combo.addItems(["Square", "Sine", "Triangle", "Sawtooth"])
        self.pulse_type_combo.setToolTip("Type of pulse signal")
        pulse_type_layout.addRow("Pulse Type:", self.pulse_type_combo)
        
        pulse_type_widget = QWidget()
        pulse_type_widget.setLayout(pulse_type_layout)
        self.addWidget(pulse_type_widget)
        
        # Amplitude control
        self.pulse_amplitude_spin = ModernSliderSpinBox(
            "Amplitude", "V",
            "Pulse signal amplitude",
            0.1, 5.0, 0.1, 2
        )
        self.pulse_amplitude_spin.setValue(1.0)
        self.addWidget(self.pulse_amplitude_spin)
        
        # Frequency and cycle time controls
        self.pulse_freq_spin = ModernSliderSpinBox(
            "Frequency", "Hz",
            "Pulse signal frequency",
            0.001, 1.0, 0.001, 3
        )
        self.pulse_freq_spin.setValue(0.04)
        self.addWidget(self.pulse_freq_spin)
        
        # Duty cycle control
        self.duty_cycle_spin = ModernSliderSpinBox(
            "Duty Cycle", "%",
            "Pulse signal duty cycle",
            1.0, 99.0, 1.0, 1
        )
        self.duty_cycle_spin.setValue(50.0)
        self.addWidget(self.duty_cycle_spin)
        
        # Offset control
        self.offset_spin = ModernSliderSpinBox(
            "Offset", "V",
            "DC offset voltage",
            -5.0, 5.0, 0.1, 2
        )
        self.offset_spin.setValue(0.0)
        self.addWidget(self.offset_spin)
        
    def get_parameters(self) -> Dict[str, float | str]:
        """Get current parameter values.
        
        Returns:
            Dictionary containing pulse parameters
        """
        return {
            'pulse_type': self.pulse_type_combo.currentText(),
            'pulse_amplitude': self.pulse_amplitude_spin.value(),
            'pulse_frequency': self.pulse_freq_spin.value(),
            'duty_cycle': self.duty_cycle_spin.value(),
            'offset': self.offset_spin.value()
        } 