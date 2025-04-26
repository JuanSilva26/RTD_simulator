"""
Main entry point for the RTD simulator application.
"""

import sys
from PyQt6.QtWidgets import QApplication
from .views.main_window import MainWindow
from .utils.logging import setup_logging, get_logger

# Set up logging
setup_logging()
logger = get_logger(__name__)

def main():
    """Initialize and run the RTD simulator application."""
    try:
        logger.info("Starting RTD simulator application")
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        logger.info("Application window shown")
        sys.exit(app.exec())
    except Exception as e:
        logger.critical(f"Application failed to start: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 