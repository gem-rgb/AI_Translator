"""
config.py — Centralized configuration management for the translation layer.

Stores user preferences in a JSON file and provides defaults for all settings.
"""

import json
import os

CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".translator")
CONFIG_FILE = os.path.join(CONFIG_DIR, "settings.json")

DEFAULTS = {
    # Capture settings
    "capture_mode": "clipboard",       # "ocr", "clipboard", "both"
    "capture_interval_sec": 2.0,       # seconds between OCR captures
    "capture_region": None,            # [x, y, w, h] or None for full screen
    "monitor_index": 1,                # which monitor to capture
    "capture_active_window_only": True,  # prefer the selected/foreground app window

    # OCR settings
    "tesseract_path": r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    "ocr_language": "eng",             # Tesseract language pack

    # Translation settings
    "target_language": "en",           # translate TO this language
    "translation_mode": "online",      # "online" or "offline"
    "cache_size": 256,                 # max cached translations
    "scratchpad_history": 8,           # frames of hidden screenshot memory
    "scratchpad_min_score": 0.48,      # decision threshold for candidate selection
    "scratchpad_track_ttl": 6,         # stale track eviction in frames

    # Overlay settings
    "overlay_font_size": 16,
    "overlay_opacity": 0.85,
    "overlay_position": [100, 100],    # [x, y] on screen
    "overlay_max_width": 600,
    "overlay_bg_color": [30, 30, 30],  # RGB
    "overlay_text_color": [255, 255, 255],  # RGB
    "overlay_width_ratio": 0.42,       # docked translation panel width
    "overlay_dock_right": True,        # snap translation panel to right side

    # Hotkey
    "toggle_hotkey": "ctrl+shift+t",

    # General
    "auto_start": False,
    "minimize_to_tray": True,
}


class Config:
    """Manage application settings with JSON persistence."""

    def __init__(self):
        self._settings = dict(DEFAULTS)
        self._load()

    def _load(self):
        """Load settings from disk, merging with defaults."""
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                self._settings.update(saved)
            except (json.JSONDecodeError, IOError):
                pass  # use defaults on corrupt file

    def save(self):
        """Persist current settings to disk."""
        os.makedirs(CONFIG_DIR, exist_ok=True)
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(self._settings, f, indent=2)

    def get(self, key, default=None):
        return self._settings.get(key, default)

    def set(self, key, value):
        self._settings[key] = value
        self.save()

    def reset(self):
        """Reset all settings to defaults."""
        self._settings = dict(DEFAULTS)
        self.save()

    def __getitem__(self, key):
        return self._settings[key]

    def __setitem__(self, key, value):
        self.set(key, value)

    def __repr__(self):
        return f"Config({json.dumps(self._settings, indent=2)})"


# Singleton instance
config = Config()
