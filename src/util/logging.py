"""
Logging utilities for SAE training.

Provides consistent logging with timestamps across all scripts.
"""

import logging
import sys
from datetime import datetime


def setup_logger(name: str = "SAE", level: int = logging.INFO) -> logging.Logger:
    """
    Create a logger with datetime formatting.

    Args:
        name: Logger name (typically __name__ of the calling module)
        level: Logging level (default: INFO)

    Returns:
        Configured logger instance

    Example:
        >>> from util.logging import setup_logger
        >>> logger = setup_logger(__name__)
        >>> logger.info("Training started")
        [2024-01-15 14:23:01] INFO - Training started
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers if logger already exists
    if logger.handlers:
        return logger

    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    # Format: [2024-01-15 14:23:01] INFO - Message
    formatter = logging.Formatter(
        fmt='[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted string like "2m 30s" or "1h 5m 23s"

    Examples:
        >>> format_time(45)
        "45s"
        >>> format_time(125)
        "2m 5s"
        >>> format_time(3665)
        "1h 1m 5s"
    """
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {minutes}m {secs}s"