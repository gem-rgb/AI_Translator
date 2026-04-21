"""
capture.py — Screen capture and clipboard monitoring.

Provides multiple text-capture strategies:
  - Full-screen or region-based screen capture (for OCR)
  - Clipboard monitoring (instant, perfect accuracy)
"""

import threading
import time
import ctypes
from ctypes import wintypes

import mss
import numpy as np
import pyperclip

from config import config


class ScreenCapture:
    """Fast screen capture using mss."""

    def __init__(self):
        self._sct = None
        self._source_hwnd = None

    def _get_sct(self):
        """Lazy-init mss (must be created in the thread that uses it)."""
        if self._sct is None:
            self._sct = mss.mss()
        return self._sct

    def capture_full_screen(self, monitor_index=None):
        """Capture the entire screen as a numpy array (BGR format).

        Args:
            monitor_index: Which monitor to capture (1-based). None uses config default.

        Returns:
            numpy.ndarray: Screenshot in BGR format.
        """
        sct = self._get_sct()
        idx = monitor_index or config["monitor_index"]
        monitor = sct.monitors[idx]
        img = np.array(sct.grab(monitor))
        # mss returns BGRA, drop alpha channel
        return img[:, :, :3]

    def capture_region(self, x, y, width, height):
        """Capture a specific screen region.

        Args:
            x, y: Top-left corner coordinates.
            width, height: Region dimensions.

        Returns:
            numpy.ndarray: Screenshot region in BGR format.
        """
        sct = self._get_sct()
        region = {"left": x, "top": y, "width": width, "height": height}
        img = np.array(sct.grab(region))
        return img[:, :, :3]

    def set_source_window(self, hwnd):
        """Remember the source window that should be captured."""
        self._source_hwnd = int(hwnd) if hwnd else None

    @staticmethod
    def _get_window_rect(hwnd):
        """Return a visible window rectangle as (x, y, w, h) or None."""
        if not hwnd:
            return None

        user32 = ctypes.windll.user32
        if not user32.IsWindow(hwnd) or not user32.IsWindowVisible(hwnd) or user32.IsIconic(hwnd):
            return None

        rect = wintypes.RECT()
        if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return None

        x = rect.left
        y = rect.top
        width = rect.right - rect.left
        height = rect.bottom - rect.top

        if width < 80 or height < 80:
            return None

        return (x, y, width, height)

    def capture_active_window(self):
        """Capture the selected source window or current foreground window."""
        user32 = ctypes.windll.user32
        hwnd = self._source_hwnd or user32.GetForegroundWindow()
        rect = self._get_window_rect(hwnd)
        if rect is None:
            return None
        return self.capture_region(*rect)

    def capture_around_cursor(self, radius=200):
        """Capture a region around the current mouse cursor.

        Args:
            radius: Half-size of the capture region in pixels.

        Returns:
            numpy.ndarray: Screenshot region in BGR format.
        """
        import ctypes

        class POINT(ctypes.Structure):
            _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

        pt = POINT()
        ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))

        x = max(0, pt.x - radius)
        y = max(0, pt.y - radius)
        return self.capture_region(x, y, radius * 2, radius * 2)

    def capture(self):
        """Capture based on current config settings.

        Returns:
            numpy.ndarray: Screenshot in BGR format.
        """
        region = config["capture_region"]
        if region:
            return self.capture_region(*region)

        if config.get("capture_active_window_only", True):
            active_window = self.capture_active_window()
            if active_window is not None:
                return active_window

        return self.capture_full_screen()

    def close(self):
        if self._sct:
            self._sct.close()
            self._sct = None


class ClipboardMonitor:
    """Monitor clipboard for text changes in a background thread."""

    def __init__(self, callback, poll_interval=0.5):
        """
        Args:
            callback: Function to call with new clipboard text.
            poll_interval: Seconds between clipboard checks.
        """
        self._callback = callback
        self._interval = poll_interval
        self._last_text = ""
        self._running = False
        self._thread = None

    def start(self):
        """Start monitoring clipboard in a background thread."""
        if self._running:
            return
        self._running = True
        # Initialize with current clipboard content so we don't
        # immediately trigger on existing content
        try:
            self._last_text = pyperclip.paste() or ""
        except Exception:
            self._last_text = ""
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None

    def _poll_loop(self):
        """Internal polling loop."""
        while self._running:
            try:
                current = pyperclip.paste() or ""
                if current and current != self._last_text:
                    self._last_text = current
                    self._callback(current)
            except Exception:
                pass
            time.sleep(self._interval)

    @property
    def is_running(self):
        return self._running
