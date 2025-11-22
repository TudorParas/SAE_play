"""
Logging utilities for SAE experiments.

Provides non-intrusive stdout capture for experiment logging.
"""

import sys
from io import StringIO
from typing import Optional


class TeeLogger:
    """
    Captures stdout while still printing to console.

    This allows experiments to capture all print output for saving to log files
    while maintaining real-time console output during training.

    Usage:
        logger = TeeLogger()
        logger.start()

        print("This goes to both console AND is captured")
        print("Training epoch 1...")

        logger.stop()
        captured_log = logger.get_log()

        # Save to file
        with open("log.txt", "w") as f:
            f.write(captured_log)
    """

    def __init__(self):
        self._buffer = StringIO()
        self._original_stdout: Optional[object] = None
        self._tee: Optional['_TeeStream'] = None
        self._is_capturing = False

    def start(self):
        """
        Start capturing stdout. All prints still go to console.

        Safe to call multiple times (only first call has effect).
        """
        if self._is_capturing:
            return

        self._original_stdout = sys.stdout
        self._tee = _TeeStream(self._original_stdout, self._buffer)
        sys.stdout = self._tee
        self._is_capturing = True

    def stop(self):
        """
        Stop capturing stdout. Restores original stdout.

        Safe to call multiple times (only first call has effect).
        """
        if not self._is_capturing:
            return

        sys.stdout = self._original_stdout
        self._is_capturing = False

    def get_log(self) -> str:
        """
        Get all captured output so far.

        Can be called while still capturing or after stop().

        Returns:
            All captured stdout as a string
        """
        return self._buffer.getvalue()

    def clear(self):
        """Clear the captured buffer (keeps capturing if active)."""
        self._buffer = StringIO()
        if self._is_capturing and self._tee:
            self._tee._buffer = self._buffer

    def __enter__(self):
        """Context manager support: start on enter."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager support: stop on exit."""
        self.stop()
        return False  # Don't suppress exceptions


class _TeeStream:
    """
    Internal stream that writes to two destinations.

    Writes to both the original stream (console) and a buffer (for capture).
    """

    def __init__(self, original_stream, buffer: StringIO):
        self._original = original_stream
        self._buffer = buffer

    def write(self, data: str):
        """Write to both streams."""
        self._original.write(data)
        self._buffer.write(data)

    def flush(self):
        """Flush both streams."""
        self._original.flush()
        # StringIO doesn't need flushing, but be safe
        if hasattr(self._buffer, 'flush'):
            self._buffer.flush()

    # Forward other attributes to original stream (for compatibility)
    def __getattr__(self, name):
        return getattr(self._original, name)