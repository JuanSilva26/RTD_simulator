"""
Real-time plotting view with blitting optimization for RTD simulation results.
"""

from typing import Optional, Tuple, List
import numpy as np
from numpy.typing import NDArray
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.axes import Axes

class RTDPlotView(QWidget):
    """
    Real-time plotting widget using Matplotlib with blitting optimization.
    Displays voltage, current, and IV characteristics of RTD simulations.
    """
    
    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize the plot view."""
        super().__init__(parent)
        
        # Create figure and subplots
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        
        # Create subplots
        self.ax_v = self.figure.add_subplot(221)  # Voltage vs time
        self.ax_i = self.figure.add_subplot(222)  # Current vs time
        self.ax_iv = self.figure.add_subplot(212)  # IV characteristic
        
        # Initialize line artists
        self.line_v = Line2D([], [], color='b', label='Voltage')
        self.line_i = Line2D([], [], color='r', label='Current')
        self.line_iv = Line2D([], [], color='g', label='IV Curve')
        
        # Add lines to axes
        self.ax_v.add_line(self.line_v)
        self.ax_i.add_line(self.line_i)
        self.ax_iv.add_line(self.line_iv)
        
        # Configure axes
        self._setup_axes()
        
        # Set up layout
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
        # Initialize background cache for blitting
        self.backgrounds: List[Optional[object]] = [None, None, None]
        self.canvas.draw()
        self._cache_backgrounds()
        
    def _setup_axes(self):
        """Configure plot axes with labels and legends."""
        # Voltage plot
        self.ax_v.set_xlabel('Time')
        self.ax_v.set_ylabel('Voltage')
        self.ax_v.grid(True)
        self.ax_v.legend()
        
        # Current plot
        self.ax_i.set_xlabel('Time')
        self.ax_i.set_ylabel('Current')
        self.ax_i.grid(True)
        self.ax_i.legend()
        
        # IV characteristic
        self.ax_iv.set_xlabel('Voltage')
        self.ax_iv.set_ylabel('Current')
        self.ax_iv.grid(True)
        self.ax_iv.legend()
        
    def _cache_backgrounds(self):
        """Cache the background for each subplot for blitting."""
        self.backgrounds[0] = self.canvas.copy_from_bbox(self.ax_v.bbox)
        self.backgrounds[1] = self.canvas.copy_from_bbox(self.ax_i.bbox)
        self.backgrounds[2] = self.canvas.copy_from_bbox(self.ax_iv.bbox)
        
    def update_plots(self, t: NDArray[np.float64], v: NDArray[np.float64], 
                    i: NDArray[np.float64], redraw: bool = False):
        """
        Update all plots with new data using blitting for efficiency.
        
        Args:
            t: Time array
            v: Voltage array
            i: Current array
            redraw: Whether to force a full redraw
        """
        if redraw:
            # Update data ranges
            self.ax_v.set_xlim(t.min(), t.max())
            self.ax_v.set_ylim(v.min(), v.max())
            self.ax_i.set_xlim(t.min(), t.max())
            self.ax_i.set_ylim(i.min(), i.max())
            self.ax_iv.set_xlim(v.min(), v.max())
            self.ax_iv.set_ylim(i.min(), i.max())
            
            # Force full redraw
            self.canvas.draw()
            self._cache_backgrounds()
        
        # Restore backgrounds
        for ax, bg in zip([self.ax_v, self.ax_i, self.ax_iv], self.backgrounds):
            self.canvas.restore_region(bg)
        
        # Update line data
        self.line_v.set_data(t, v)
        self.line_i.set_data(t, i)
        self.line_iv.set_data(v, i)
        
        # Redraw just the lines
        self.ax_v.draw_artist(self.line_v)
        self.ax_i.draw_artist(self.line_i)
        self.ax_iv.draw_artist(self.line_iv)
        
        # Blit each axes region
        self.canvas.blit(self.ax_v.bbox)
        self.canvas.blit(self.ax_i.bbox)
        self.canvas.blit(self.ax_iv.bbox)
        
    def clear_plots(self):
        """Clear all plots."""
        self.line_v.set_data([], [])
        self.line_i.set_data([], [])
        self.line_iv.set_data([], [])
        self.canvas.draw()
        self._cache_backgrounds() 