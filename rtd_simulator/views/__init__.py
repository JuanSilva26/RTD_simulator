"""
RTD view package.

This package contains all the UI components for the RTD simulator,
including the main window, plotting tools, and custom widgets.
"""

from .plotting import RTDPlotter
from .main_window import MainWindow
from .plot_style import PlotStyle

__all__ = [
    'RTDPlotter',
    'MainWindow',
    'PlotStyle',
] 