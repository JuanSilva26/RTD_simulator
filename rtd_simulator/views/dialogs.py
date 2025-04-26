"""
Dialog components for the RTD simulation platform.

This module contains various dialog windows used in the application.
"""

from typing import Dict, List, Optional, Any
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QGroupBox, 
                            QCheckBox, QListWidget, QDialogButtonBox,
                            QComboBox, QHBoxLayout, QPushButton, QLabel,
                            QLineEdit, QMessageBox)
from PyQt6.QtCore import Qt
import json
import os

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
        self.setMinimumWidth(400)
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Add export options
        self.export_data = QCheckBox("Export Data (CSV)")
        self.export_plots = QCheckBox("Export Plots")
        self.plot_format = QComboBox()
        self.plot_format.addItems(["PNG (*.png)", "PDF (*.pdf)", "SVG (*.svg)"])
        
        # Add to layout
        layout.addWidget(self.export_data)
        layout.addWidget(self.export_plots)
        layout.addWidget(QLabel("Plot Format:"))
        layout.addWidget(self.plot_format)
        
        # Add buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
    def get_export_options(self) -> Dict[str, Any]:
        """Get selected export options."""
        return {
            'export_data': self.export_data.isChecked(),
            'export_plots': self.export_plots.isChecked(),
            'plot_format': self.plot_format.currentText()
        }

class PresetDialog(QDialog):
    """Dialog for managing simulation presets."""
    
    def __init__(self, parent=None):
        """Initialize the preset dialog."""
        super().__init__(parent)
        self.setWindowTitle("Manage Presets")
        self.setMinimumWidth(400)
        self.setMinimumHeight(300)
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Create preset list
        self.preset_list = QListWidget()
        self.preset_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        layout.addWidget(self.preset_list)
        
        # Create input area for new presets
        input_layout = QHBoxLayout()
        self.preset_name = QLineEdit()
        self.preset_name.setPlaceholderText("Enter preset name...")
        input_layout.addWidget(self.preset_name)
        
        # Create buttons
        button_layout = QHBoxLayout()
        
        self.save_btn = QPushButton("Save")
        self.save_btn.clicked.connect(self._save_preset)
        button_layout.addWidget(self.save_btn)
        
        self.delete_btn = QPushButton("Delete")
        self.delete_btn.clicked.connect(self._delete_preset)
        button_layout.addWidget(self.delete_btn)
        
        input_layout.addLayout(button_layout)
        layout.addLayout(input_layout)
        
        # Add dialog buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        # Load existing presets
        self._load_presets()
        
    def _load_presets(self):
        """Load presets from file."""
        try:
            if os.path.exists('presets.json'):
                with open('presets.json', 'r') as f:
                    self.presets = json.load(f)
                    self.preset_list.clear()
                    self.preset_list.addItems(self.presets.keys())
            else:
                self.presets = {}
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load presets: {str(e)}")
            self.presets = {}
            
    def _save_presets(self):
        """Save presets to file."""
        try:
            with open('presets.json', 'w') as f:
                json.dump(self.presets, f, indent=4)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to save presets: {str(e)}")
            
    def _save_preset(self):
        """Save current parameters as a preset."""
        name = self.preset_name.text().strip()
        if not name:
            QMessageBox.warning(self, "Error", "Please enter a preset name")
            return
            
        # Get current parameters from parent window
        params = self.parent().get_parameters()
        
        # Save preset
        self.presets[name] = params
        self._save_presets()
        
        # Update list
        self.preset_list.clear()
        self.preset_list.addItems(self.presets.keys())
        
        # Clear input
        self.preset_name.clear()
        
    def _delete_preset(self):
        """Delete selected preset."""
        selected = self.preset_list.currentItem()
        if not selected:
            QMessageBox.warning(self, "Error", "Please select a preset to delete")
            return
            
        name = selected.text()
        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Are you sure you want to delete preset '{name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            del self.presets[name]
            self._save_presets()
            self.preset_list.clear()
            self.preset_list.addItems(self.presets.keys())
            
    def get_selected_preset(self) -> Optional[Dict[str, Any]]:
        """Get the selected preset parameters."""
        selected = self.preset_list.currentItem()
        if selected:
            return self.presets[selected.text()]
        return None 