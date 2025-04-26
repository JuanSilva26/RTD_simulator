"""
Custom widgets for the RTD Simulation Platform.
"""

from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QLabel,
                           QDoubleSpinBox, QSlider, QFrame, QPushButton,
                           QSizePolicy, QStatusBar, QComboBox, QMenu,
                           QInputDialog, QMessageBox)
from PyQt6.QtCore import Qt, pyqtSignal, pyqtProperty
from PyQt6.QtGui import QPalette, QColor, QPainter, QFont
import json
from pathlib import Path
from typing import Dict, Any

class ModernSliderSpinBox(QWidget):
    """A modern combination of slider and spinbox with unit display."""
    
    valueChanged = pyqtSignal(float)
    
    def __init__(self, name: str, unit: str = "", tooltip: str = "",
                 min_val: float = 0.0, max_val: float = 1.0,
                 step: float = 0.1, decimals: int = 2,
                 parent: QWidget = None):
        """
        Initialize the widget.
        
        Args:
            name: Parameter name
            unit: Unit to display
            tooltip: Tooltip text
            min_val: Minimum value
            max_val: Maximum value
            step: Step size
            decimals: Number of decimal places
            parent: Parent widget
        """
        super().__init__(parent)
        self.name = name
        self._unit = unit
        self._scale_factor = 1.0  # Internal scaling factor for slider values
        
        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        
        # Create header with name and unit
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        
        name_label = QLabel(name)
        name_label.setStyleSheet("""
            QLabel {
                color: #e1e1e1;
                font-size: 11px;
                font-weight: 500;
            }
        """)
        header_layout.addWidget(name_label)
        
        self.unit_label = QLabel(f"({unit})" if unit else "")
        self.unit_label.setStyleSheet("""
            QLabel {
                color: #808080;
                font-size: 10px;
            }
        """)
        header_layout.addWidget(self.unit_label)
            
        header_layout.addStretch()
        layout.addWidget(header)
        
        # Create control layout
        controls = QWidget()
        control_layout = QHBoxLayout(controls)
        control_layout.setContentsMargins(0, 0, 0, 0)
        control_layout.setSpacing(8)
        
        # Create slider
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self._update_slider_range(min_val, max_val, step)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(10)
        self.slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: none;
                height: 4px;
                background: #404040;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #2196F3;
                border: none;
                width: 12px;
                margin: -4px 0;
                border-radius: 6px;
            }
            QSlider::handle:horizontal:hover {
                background: #42A5F5;
            }
            QSlider::sub-page:horizontal {
                background: #2196F3;
                border-radius: 2px;
            }
        """)
        control_layout.addWidget(self.slider)
        
        # Create spinbox
        self.spinbox = QDoubleSpinBox()
        self.spinbox.setRange(min_val, max_val)
        self.spinbox.setSingleStep(step)
        self.spinbox.setDecimals(decimals)
        self.spinbox.setFixedWidth(85)
        self.spinbox.setStyleSheet("""
            QDoubleSpinBox {
                border: 1px solid #404040;
                border-radius: 3px;
                padding: 2px 4px;
                background: #2a2a2a;
                color: #e1e1e1;
                selection-background-color: #2196F3;
                min-height: 24px;
            }
            QDoubleSpinBox:hover {
                border-color: #2196F3;
            }
            QDoubleSpinBox:focus {
                border-color: #42A5F5;
            }
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
                width: 20px;
                border: none;
                background: #404040;
            }
            QDoubleSpinBox::up-button {
                border-top-right-radius: 2px;
            }
            QDoubleSpinBox::down-button {
                border-bottom-right-radius: 2px;
            }
            QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {
                background: #505050;
            }
            QDoubleSpinBox::up-arrow {
                image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cGF0aCBkPSJNOCw0TDQsOEwxMiw4TDgsNFoiIGZpbGw9IiNlMWUxZTEiLz48L3N2Zz4=);
                width: 12px;
                height: 12px;
            }
            QDoubleSpinBox::down-arrow {
                image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cGF0aCBkPSJNOCwxMkwxMiw4TDQsOEw4LDEyWiIgZmlsbD0iI2UxZTFlMSIvPjwvc3ZnPg==);
                width: 12px;
                height: 12px;
            }
            QDoubleSpinBox::up-arrow:hover, QDoubleSpinBox::down-arrow:hover {
                opacity: 0.8;
            }
        """)
        control_layout.addWidget(self.spinbox)
        
        layout.addWidget(controls)
        
        # Connect signals
        self.slider.valueChanged.connect(self._on_slider_value_changed)
        self.spinbox.valueChanged.connect(self._on_spinbox_value_changed)
        self.spinbox.valueChanged.connect(self.valueChanged.emit)
        
        # Set tooltip
        if tooltip:
            self.setToolTip(tooltip)
            
    def _update_slider_range(self, min_val: float, max_val: float, step: float):
        """Update slider range with appropriate scaling to avoid integer overflow.
        
        Args:
            min_val: Minimum value
            max_val: Maximum value
            step: Step size
        """
        # Calculate appropriate scale factor to prevent overflow
        range_size = (max_val - min_val) / step
        if range_size > 1e9:  # If range would be too large
            self._scale_factor = step * (range_size / 1e6)  # Scale to reasonable range
        else:
            self._scale_factor = step
            
        # Set slider range using scaled values
        self.slider.setMinimum(int(min_val / self._scale_factor))
        self.slider.setMaximum(int(max_val / self._scale_factor))

    def _on_slider_value_changed(self, value: int):
        """Handle slider value changes."""
        self.spinbox.setValue(value * self._scale_factor)

    def _on_spinbox_value_changed(self, value: float):
        """Handle spinbox value changes."""
        # Prevent recursive signal emission
        self.slider.blockSignals(True)
        self.slider.setValue(int(value / self._scale_factor))
        self.slider.blockSignals(False)

    def value(self) -> float:
        """Get current value."""
        return self.spinbox.value()
        
    def setValue(self, value: float):
        """Set current value."""
        self.spinbox.setValue(value)

    def setRange(self, min_val: float, max_val: float):
        """Set the range for both slider and spinbox.
        
        Args:
            min_val: Minimum value
            max_val: Maximum value
        """
        step = self.spinbox.singleStep()
        self._update_slider_range(min_val, max_val, step)
        self.spinbox.setRange(min_val, max_val)

    def setDecimals(self, decimals: int):
        """Set number of decimal places.
        
        Args:
            decimals: Number of decimal places
        """
        self.spinbox.setDecimals(decimals)

    def setStep(self, step: float):
        """Set step size for both slider and spinbox.
        
        Args:
            step: Step size
        """
        self.spinbox.setSingleStep(step)
        current_min = self.spinbox.minimum()
        current_max = self.spinbox.maximum()
        self._update_slider_range(current_min, current_max, step)
        
    @property
    def unit(self) -> str:
        """Get the current unit."""
        return self._unit
        
    @unit.setter
    def unit(self, value: str):
        """Set the unit and update the display.
        
        Args:
            value: New unit string
        """
        self._unit = value
        self.unit_label.setText(f"({value})" if value else "")

class InfoButton(QPushButton):
    """A circular info button that shows a tooltip on hover and displays a message box on click."""
    
    def __init__(self, title: str, content: str, parent: QWidget = None):
        """
        Initialize the info button.
        
        Args:
            title: Title for the info dialog
            content: Content to display in the info dialog
            parent: Parent widget
        """
        super().__init__(parent)
        self.title = title
        self.content = content
        
        # Set button properties
        self.setFixedSize(20, 20)
        self.setStyleSheet("""
            QPushButton {
                background-color: #2a2a2a;
                border: 1px solid #404040;
                border-radius: 10px;
                color: #e1e1e1;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #404040;
                border-color: #505050;
            }
        """)
        self.setText("i")
        
        # Set tooltip
        self.setToolTip("Click for more information")
        
        # Connect click event
        self.clicked.connect(self.show_info)
        
    def show_info(self):
        """Show the information dialog."""
        QMessageBox.information(self, self.title, self.content)

class CollapsibleSection(QWidget):
    """A collapsible section for grouping parameters."""
    
    def __init__(self, title: str, parent: QWidget = None):
        """
        Initialize the section.
        
        Args:
            title: Section title
            parent: Parent widget
        """
        super().__init__(parent)
        
        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 8)
        layout.setSpacing(1)
        
        # Create header
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(12, 8, 12, 8)
        
        self.toggle_btn = QPushButton("▼")
        self.toggle_btn.setFixedWidth(16)
        self.toggle_btn.setStyleSheet("""
            QPushButton {
                border: none;
                color: #e1e1e1;
                background: transparent;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                color: #2196F3;
            }
        """)
        header_layout.addWidget(self.toggle_btn)
        
        title_label = QLabel(title)
        title_label.setStyleSheet("""
            QLabel {
                color: #e1e1e1;
                font-size: 12px;
                font-weight: bold;
                background: transparent;
            }
        """)
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        
        # Create content widget
        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(16, 4, 12, 8)
        self.content_layout.setSpacing(8)
        
        # Add widgets to main layout
        layout.addWidget(header)
        layout.addWidget(self.content)
        
        # Connect toggle button
        self.toggle_btn.clicked.connect(self.toggle_content)
        
        # Style
        self.setStyleSheet("""
            CollapsibleSection {
                background-color: #2a2a2a;
                border: 1px solid #404040;
                border-radius: 6px;
                margin: 2px;
            }
            CollapsibleSection:hover {
                border-color: #505050;
            }
            QWidget#content {
                background-color: #1e1e1e;
            }
        """)
        self.content.setObjectName("content")
        
    def toggle_content(self):
        """Toggle content visibility."""
        if self.content.isVisible():
            self.content.hide()
            self.toggle_btn.setText("▶")
        else:
            self.content.show()
            self.toggle_btn.setText("▼")
            
    def addWidget(self, widget: QWidget):
        """Add widget to content layout."""
        self.content_layout.addWidget(widget)
        
class StatusBar(QStatusBar):
    """Custom status bar with parameter display."""
    
    def __init__(self, parent: QWidget = None):
        """Initialize the status bar."""
        super().__init__(parent)
        
        # Create widgets
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #e1e1e1;")
        self.addWidget(self.status_label)
        
        # Add separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.VLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        self.addWidget(separator)
        
        # Create parameter display
        self.param_label = QLabel()
        self.param_label.setStyleSheet("color: #b0b0b0;")
        self.addWidget(self.param_label)
        
        # Add stretch
        self.addPermanentWidget(QLabel(), 1)  # Stretch
        
        # Style
        self.setStyleSheet("""
            QStatusBar {
                background-color: #2a2a2a;
                border-top: 1px solid #404040;
            }
        """)
        
    def set_status(self, text: str):
        """Set status text."""
        self.status_label.setText(text)
        
    def set_parameters(self, params: dict):
        """Set parameter display."""
        param_text = ", ".join(f"{k}: {v:.3g}" for k, v in params.items())
        self.param_label.setText(param_text)

class PresetManager(QWidget):
    """Manages parameter presets with save/load functionality."""
    
    presetLoaded = pyqtSignal(dict)  # Emits parameter dict when preset is loaded
    
    def __init__(self, parent: QWidget = None):
        """Initialize the preset manager."""
        super().__init__(parent)
        
        # Create layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        
        # Create preset combo box
        self.preset_combo = QComboBox()
        self.preset_combo.setMinimumWidth(150)
        self.preset_combo.currentTextChanged.connect(self._on_preset_selected)
        layout.addWidget(self.preset_combo)
        
        # Create buttons
        save_btn = QPushButton("Save")
        save_btn.setToolTip("Save current parameters as preset")
        save_btn.clicked.connect(self.save_preset)
        layout.addWidget(save_btn)
        
        delete_btn = QPushButton("Delete")
        delete_btn.setToolTip("Delete selected preset")
        delete_btn.clicked.connect(self.delete_preset)
        layout.addWidget(delete_btn)
        
        # Initialize presets
        self.presets: Dict[str, Dict[str, Any]] = {}
        self.preset_file = Path("presets.json")
        self.load_presets()
        
    def load_presets(self):
        """Load presets from file."""
        try:
            if self.preset_file.exists():
                with open(self.preset_file, 'r') as f:
                    self.presets = json.load(f)
                self._update_combo_box()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load presets: {str(e)}")
            
    def save_presets(self):
        """Save presets to file."""
        try:
            with open(self.preset_file, 'w') as f:
                json.dump(self.presets, f, indent=2)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to save presets: {str(e)}")
            
    def _update_combo_box(self):
        """Update the combo box with current presets."""
        self.preset_combo.clear()
        self.preset_combo.addItem("Select Preset...")
        self.preset_combo.addItems(sorted(self.presets.keys()))
        
    def _on_preset_selected(self, name: str):
        """Handle preset selection."""
        if name and name != "Select Preset...":
            self.presetLoaded.emit(self.presets[name])
            
    def save_preset(self):
        """Save current parameters as a new preset."""
        name, ok = QInputDialog.getText(
            self, "Save Preset",
            "Enter preset name:",
            text=self.preset_combo.currentText()
        )
        
        if ok and name:
            # Get current parameters from parent
            params = self.parent().get_parameters() if hasattr(self.parent(), 'get_parameters') else {}
            
            self.presets[name] = params
            self.save_presets()
            self._update_combo_box()
            self.preset_combo.setCurrentText(name)
            
    def delete_preset(self):
        """Delete the selected preset."""
        name = self.preset_combo.currentText()
        if name and name != "Select Preset...":
            if QMessageBox.question(
                self, "Confirm Delete",
                f"Delete preset '{name}'?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            ) == QMessageBox.StandardButton.Yes:
                del self.presets[name]
                self.save_presets()
                self._update_combo_box() 