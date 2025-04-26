"""
Animation window for RTD dynamics visualization.

This module provides a window for animating the construction of pulse and voltage waveforms
over time, helping visualize the RTD's response to input signals.
"""

from typing import Optional, Dict, Any
import numpy as np
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QPushButton, QHBoxLayout,
                           QSlider, QLabel)
from PyQt6.QtCore import Qt, QTimer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

class AnimationWindow(QDialog):
    """Window for animating RTD dynamics."""
    
    def __init__(self, t: np.ndarray, pulse: np.ndarray, voltage: np.ndarray,
                 parent: Optional[Any] = None):
        """
        Initialize the animation window.
        
        Args:
            t: Time array
            pulse: Pulse signal array
            voltage: Voltage response array
            parent: Parent widget
        """
        super().__init__(parent)
        self.setWindowTitle("RTD Dynamics Animation")
        self.setGeometry(100, 100, 1000, 800)
        
        # Store data
        self.t = t
        self.pulse = pulse
        self.voltage = voltage
        self.current_index = 0
        
        # Base step size (will be modified by speed multiplier)
        self.base_step_size = max(1, len(t) // 200)
        self.speed_multiplier = 1
        
        # Create layout
        layout = QVBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Create figure and canvas
        self.figure = Figure(figsize=(10, 8), dpi=120)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.figure.set_facecolor('#2b2b2b')
        layout.addWidget(self.canvas)
        
        # Create subplots with increased spacing
        gs = self.figure.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.5)
        self.ax_pulse = self.figure.add_subplot(gs[0])
        self.ax_voltage = self.figure.add_subplot(gs[1])
        
        # Initial setup of plots
        self._setup_static_plot_elements()
        
        # Create control buttons
        control_layout = QHBoxLayout()
        control_layout.setContentsMargins(0, 10, 0, 0)
        
        self.play_button = QPushButton("Play")
        self.play_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 5px 15px;
                border-radius: 3px;
                min-width: 80px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        self.play_button.clicked.connect(self.toggle_animation)
        control_layout.addWidget(self.play_button)
        
        self.reset_button = QPushButton("Reset")
        self.reset_button.setStyleSheet("""
            QPushButton {
                background-color: #424242;
                color: white;
                border: none;
                padding: 5px 15px;
                border-radius: 3px;
                min-width: 80px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #616161;
            }
        """)
        self.reset_button.clicked.connect(self.reset_animation)
        control_layout.addWidget(self.reset_button)
        
        speed_label = QLabel("Speed:")
        speed_label.setStyleSheet("color: white; font-size: 14px; font-weight: bold;")
        control_layout.addWidget(speed_label)
        
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(100)
        self.speed_slider.setValue(1)  # Start at minimum speed
        self.speed_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                background: #424242;
                height: 8px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #2196F3;
                width: 16px;
                margin: -4px 0;
                border-radius: 8px;
            }
        """)
        self.speed_slider.valueChanged.connect(self.update_speed)
        control_layout.addWidget(self.speed_slider)
        
        layout.addLayout(control_layout)
        
        # Setup animation timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_animation)
        self.is_playing = False
        self.update_interval = 20  # Start with moderate speed (20ms)
        
        # Initialize plot lines
        self._init_plot_lines()
        
    def _setup_static_plot_elements(self):
        """Setup the static elements of the plots that don't change during animation."""
        for ax in [self.ax_pulse, self.ax_voltage]:
            ax.set_facecolor('white')
            ax.grid(True, color='lightgray', linestyle='-', linewidth=0.5)
            ax.tick_params(which='major', length=6, width=1.0, labelsize=10, colors='black')
            for spine in ax.spines.values():
                spine.set_color('black')
                spine.set_linewidth(1.0)
        
        # Set titles and labels
        self.ax_pulse.set_title("Pulse Wave", color='black', pad=15, fontsize=12, fontweight='bold')
        self.ax_pulse.set_xlabel("Time (s)", color='black', fontsize=10)
        self.ax_pulse.set_ylabel("Amplitude (V)", color='black', fontsize=10)
        
        self.ax_voltage.set_title("Voltage Response", color='black', pad=15, fontsize=12, fontweight='bold')
        self.ax_voltage.set_xlabel("Time (s)", color='black', fontsize=10)
        self.ax_voltage.set_ylabel("Voltage (V)", color='black', fontsize=10)
        
        # Set axis limits with padding
        pulse_padding = 0.15 * (max(self.pulse) - min(self.pulse))
        voltage_padding = 0.15 * (max(self.voltage) - min(self.voltage))
        
        self.ax_pulse.set_xlim(0, self.t[-1])
        self.ax_pulse.set_ylim(min(self.pulse) - pulse_padding, max(self.pulse) + pulse_padding)
        
        self.ax_voltage.set_xlim(0, self.t[-1])
        self.ax_voltage.set_ylim(min(self.voltage) - voltage_padding, max(self.voltage) + voltage_padding)
        
        # Adjust layout with more space between subplots
        self.figure.tight_layout(h_pad=3.0)
        
    def _init_plot_lines(self):
        """Initialize the plot lines for animation."""
        # Remove any existing lines
        for line in self.ax_pulse.lines + self.ax_voltage.lines:
            line.remove()
        
        # Create new lines
        self.pulse_line, = self.ax_pulse.plot([], [], color='green', linewidth=2.0, label='Pulse')
        self.voltage_line, = self.ax_voltage.plot([], [], color='blue', linewidth=2.0, label='Voltage')
        
        # Set up legends
        legend_style = {
            'facecolor': 'white',
            'labelcolor': 'black',
            'fontsize': 10,
            'framealpha': 1.0,
            'edgecolor': 'gray',
            'loc': 'upper right'
        }
        
        self.ax_pulse.legend(**legend_style)
        self.ax_voltage.legend(**legend_style)
        
        # Draw the canvas once
        self.canvas.draw()
        
    def update_speed(self, value: int):
        """Update animation speed based on slider value."""
        # Exponential mapping for speed multiplier (1 to 50)
        self.speed_multiplier = np.exp(np.log(50) * (value - 1) / 99)
        
        # Update interval goes from 20ms down to 1ms exponentially
        self.update_interval = max(1, int(20 * np.exp(-np.log(20) * (value - 1) / 99)))
        
        if self.is_playing:
            self.timer.setInterval(self.update_interval)
        
    def toggle_animation(self):
        """Toggle animation play/pause."""
        if self.is_playing:
            self.timer.stop()
            self.play_button.setText("Play")
        else:
            if self.current_index >= len(self.t):
                self.reset_animation()
            self.timer.setInterval(self.update_interval)
            self.timer.start()
            self.play_button.setText("Pause")
        self.is_playing = not self.is_playing
        
    def update_animation(self):
        """Update the animation frame."""
        if self.current_index < len(self.t):
            # Calculate effective step size based on speed multiplier
            effective_step = int(self.base_step_size * self.speed_multiplier)
            
            # Update data
            current_t = self.t[:self.current_index]
            current_pulse = self.pulse[:self.current_index]
            current_voltage = self.voltage[:self.current_index]
            
            # Update line data
            self.pulse_line.set_data(current_t, current_pulse)
            self.voltage_line.set_data(current_t, current_voltage)
            
            # Draw everything
            self.canvas.draw()
            
            # Update index with speed-adjusted step size
            self.current_index += effective_step
        else:
            self.timer.stop()
            self.is_playing = False
            self.play_button.setText("Play")
        
    def reset_animation(self):
        """Reset the animation to the beginning."""
        self.timer.stop()
        self.is_playing = False
        self.play_button.setText("Play")
        self.current_index = 0
        
        # Reset plot lines
        self._init_plot_lines()
        
    def closeEvent(self, event):
        """Clean up when window is closed."""
        self.timer.stop()
        super().closeEvent(event) 