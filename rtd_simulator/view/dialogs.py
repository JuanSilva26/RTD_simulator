"""
Dialog components for the RTD simulation platform.

This module contains various dialog windows used in the application.
"""

from typing import Dict, List, Optional
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QGroupBox, 
                            QCheckBox, QListWidget, QDialogButtonBox)
from PyQt6.QtCore import Qt

class ExportDialog(QDialog):
    """Dialog for configuring export options.
    
    This dialog allows users to select what data to export (simulation data
    and/or plots) and in what format.
    """
    
    def __init__(self, parent=None):
        """Initialize the export dialog.
        
        Args:
            parent: Optional parent widget
        """
        super().__init__(parent)
        self.setWindowTitle("Export Options")
        self.setModal(True)
        
        self._init_ui()
        
    def _init_ui(self) -> None:
        """Initialize the dialog's user interface."""
        layout = QVBoxLayout(self)
        
        # Export type selection
        type_group = QGroupBox("Export Type")
        type_layout = QVBoxLayout()
        self.data_check = QCheckBox("Simulation Data (CSV)")
        self.plots_check = QCheckBox("Plots")
        self.data_check.setChecked(True)
        type_layout.addWidget(self.data_check)
        type_layout.addWidget(self.plots_check)
        type_group.setLayout(type_layout)
        layout.addWidget(type_group)
        
        # Plot selection
        self.plot_group = QGroupBox("Select Plots")
        plot_layout = QVBoxLayout()
        self.plot_list = QListWidget()
        self.plot_list.addItems([
            "All Plots",
            "IV Characteristics",
            "Pulse Wave",
            "Voltage vs Time",
            "Current vs Time"
        ])
        self.plot_list.setEnabled(False)
        plot_layout.addWidget(self.plot_list)
        self.plot_group.setLayout(plot_layout)
        layout.addWidget(self.plot_group)
        
        # Format selection for plots
        self.format_group = QGroupBox("Plot Format")
        format_layout = QVBoxLayout()
        self.format_list = QListWidget()
        self.format_list.addItems([
            "PNG (*.png)",
            "JPEG (*.jpg)",
            "SVG (*.svg)",
            "PDF (*.pdf)"
        ])
        self.format_list.setEnabled(False)
        format_layout.addWidget(self.format_list)
        self.format_group.setLayout(format_layout)
        layout.addWidget(self.format_group)
        
        # Connect checkboxes to enable/disable options
        self.plots_check.stateChanged.connect(self._toggle_plot_options)
        
        # Add standard buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
    def _toggle_plot_options(self, state: int) -> None:
        """Enable/disable plot options based on checkbox state.
        
        Args:
            state: Qt checkbox state
        """
        enabled = state == Qt.CheckState.Checked
        self.plot_list.setEnabled(enabled)
        self.format_list.setEnabled(enabled)
        
    def get_export_options(self) -> Dict[str, bool | List[str] | Optional[str]]:
        """Get the selected export options.
        
        Returns:
            Dictionary containing export configuration:
            - export_data: Whether to export simulation data
            - export_plots: Whether to export plots
            - selected_plots: List of plot types to export
            - plot_format: Selected plot format
        """
        options = {
            'export_data': self.data_check.isChecked(),
            'export_plots': self.plots_check.isChecked(),
            'selected_plots': [],
            'plot_format': None
        }
        
        if options['export_plots']:
            selected_items = self.plot_list.selectedItems()
            options['selected_plots'] = [item.text() for item in selected_items]
            format_items = self.format_list.selectedItems()
            if format_items:
                options['plot_format'] = format_items[0].text()
                
        return options 