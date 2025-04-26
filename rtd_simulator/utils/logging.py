"""
Logging configuration for the RTD simulator.

This module provides centralized logging configuration and utilities
for the entire application.
"""

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional

# Create logs directory if it doesn't exist
LOGS_DIR = Path(__file__).parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Default log file path
DEFAULT_LOG_FILE = LOGS_DIR / "rtd_simulator.log"

def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    console_output: bool = True
) -> None:
    """
    Set up logging configuration for the application.
    
    Args:
        level: Logging level (default: INFO)
        log_file: Path to log file (default: logs/rtd_simulator.log)
        console_output: Whether to output logs to console (default: True)
    """
    # Use default log file if none specified
    if log_file is None:
        log_file = DEFAULT_LOG_FILE
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Add file handler
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # Log startup message
    logger.info("Logging system initialized")

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Name of the logger (typically __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name) 