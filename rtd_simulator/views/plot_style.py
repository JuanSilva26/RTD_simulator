"""
Plot style manager for RTD visualization.
"""

from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

class PlotStyle:
    """Manages plot styling and themes."""
    
    # Default color schemes
    COLOR_SCHEMES = {
        'default': {
            'background': 'white',
            'grid': '#e0e0e0',
            'text': 'black',
            'line': '#1f77b4',
            'measurement': 'red',
            'cursor': 'blue'
        }
    }
    
    # Default line styles
    LINE_STYLES = {
        'solid': '-',
        'dashed': '--',
        'dotted': ':',
        'dashdot': '-.'
    }
    
    # Default marker styles
    MARKER_STYLES = {
        'point': '.',
        'circle': 'o',
        'square': 's',
        'triangle': '^',
        'diamond': 'D'
    }
    
    def __init__(self, fig: Figure, ax: Axes, theme: str = 'default'):
        """
        Initialize plot style.
        
        Args:
            fig: Matplotlib figure
            ax: Matplotlib axes
            theme: Color scheme name ('default')
        """
        self.fig = fig
        self.ax = ax
        self.set_theme(theme)
        
    def set_theme(self, theme: str) -> None:
        """
        Set the color theme for the plot.
        
        Args:
            theme: Color scheme name
        """
        if theme not in self.COLOR_SCHEMES:
            raise ValueError(f"Unknown theme: {theme}. Available themes: {list(self.COLOR_SCHEMES.keys())}")
            
        self.colors = self.COLOR_SCHEMES[theme]
        
        # Apply theme
        self.fig.patch.set_facecolor(self.colors['background'])
        self.ax.set_facecolor(self.colors['background'])
        self.ax.grid(color=self.colors['grid'], linestyle='--', alpha=0.7)
        
        # Update text colors
        for text in self.ax.texts:
            text.set_color(self.colors['text'])
            
        self.ax.xaxis.label.set_color(self.colors['text'])
        self.ax.yaxis.label.set_color(self.colors['text'])
        self.ax.title.set_color(self.colors['text'])
        
        # Update tick colors
        self.ax.tick_params(colors=self.colors['text'])
        
        self.fig.canvas.draw()
        
    def set_line_style(self, line: Any, style: str) -> None:
        """
        Set the style for a line.
        
        Args:
            line: Matplotlib line object
            style: Line style name
        """
        if style not in self.LINE_STYLES:
            raise ValueError(f"Unknown line style: {style}. Available styles: {list(self.LINE_STYLES.keys())}")
            
        line.set_linestyle(self.LINE_STYLES[style])
        self.fig.canvas.draw()
        
    def set_marker_style(self, line: Any, style: str) -> None:
        """
        Set the marker style for a line.
        
        Args:
            line: Matplotlib line object
            style: Marker style name
        """
        if style not in self.MARKER_STYLES:
            raise ValueError(f"Unknown marker style: {style}. Available styles: {list(self.MARKER_STYLES.keys())}")
            
        line.set_marker(self.MARKER_STYLES[style])
        self.fig.canvas.draw()
        
    def set_line_color(self, line: Any, color: str) -> None:
        """
        Set the color for a line.
        
        Args:
            line: Matplotlib line object
            color: Color name or hex code
        """
        line.set_color(color)
        self.fig.canvas.draw()
        
    def set_line_width(self, line: Any, width: float) -> None:
        """
        Set the width for a line.
        
        Args:
            line: Matplotlib line object
            width: Line width in points
        """
        line.set_linewidth(width)
        self.fig.canvas.draw()
        
    def set_font_size(self, size: float) -> None:
        """
        Set the font size for all text elements.
        
        Args:
            size: Font size in points
        """
        self.ax.xaxis.label.set_fontsize(size)
        self.ax.yaxis.label.set_fontsize(size)
        self.ax.title.set_fontsize(size)
        self.ax.tick_params(labelsize=size)
        self.fig.canvas.draw() 