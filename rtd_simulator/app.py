"""
Main entry point for the RTD simulator application.
"""

import sys
from PyQt6.QtWidgets import QApplication
from .views.main_window import MainWindow

def main():
    """Initialize and run the RTD simulator application."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 