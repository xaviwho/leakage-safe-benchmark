"""
Logging utility for the project.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "digital_twin",
    level: str = "INFO",
    log_file: Optional[str] = None,
    console_output: bool = True
) -> logging.Logger:
    """
    Set up logger with file and console handlers.

    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file. If None, no file handler added.
        console_output: Whether to output to console

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
