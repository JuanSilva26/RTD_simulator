"""
Main application entry point for RTD simulation platform.
"""

import sys
from PyQt6.QtWidgets import QApplication
from .view.main_window import MainWindow

def main():
    """Run the RTD simulation application."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 