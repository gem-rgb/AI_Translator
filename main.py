"""
main.py - Pipeline orchestrator for live chat translation.

Pipeline:
  1. Capture screenshot (full screen or selected region)
  2. OCR to extract text with bounding boxes
  3. Group words into lines -> lines into text blocks
  4. Detect and translate only meaningful foreign-language content
  5. Show translated messages in a docked companion panel
"""

import atexit
import logging
import sys
import threading

from PyQt5 import QtCore, QtGui, QtWidgets

from capture import ScreenCapture
from config import config
from ocr import OCREngine
from overlay import RegionSelector, TranslationOverlay
from renderer import filter_blocks_by_quality, group_lines_into_blocks, group_words_into_lines
from scratchpad import ScreenScratchpad
from translator import Translator


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("translator")


class TranslationWorker(QtCore.QThread):
    """Background worker that captures, OCRs, and emits translated entries."""

    translations_ready = QtCore.pyqtSignal(object, str)
    status_update = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = False
        self._capture_requested = False
        self._continuous = False
        self._screen_capture = ScreenCapture()
        self._ocr = OCREngine()
        self._scratchpad = ScreenScratchpad()
        self._translator = Translator()
        self._last_emitted_signature = None
        self._consecutive_unchanged = 0

    def run(self):
        """Main worker loop."""
        self._running = True

        # Lower this thread's priority so it doesn't starve user apps
        try:
            import ctypes
            BELOW_NORMAL = 0x00004000
            handle = ctypes.windll.kernel32.GetCurrentThread()
            ctypes.windll.kernel32.SetThreadPriority(handle, -1)  # THREAD_PRIORITY_BELOW_NORMAL
        except Exception:
            pass

        self.status_update.emit("Worker started - press Ctrl+Shift+T to capture")

        while self._running:
            if self._capture_requested or self._continuous:
                self._capture_requested = False
                self._process_screenshot()

                if self._continuous:
                    self._sleep_until_next_capture()
                    continue

            self.msleep(120)

        self._screen_capture.close()

    def request_capture(self):
        """Trigger a one-shot capture."""
        self._capture_requested = True
        self.status_update.emit("Capturing...")

    def set_source_window(self, hwnd):
        """Remember which app window should be used as the capture source."""
        self._screen_capture.set_source_window(hwnd)

    def toggle_continuous(self):
        """Toggle continuous capture mode."""
        self._continuous = not self._continuous
        self._consecutive_unchanged = 0
        state = "ON" if self._continuous else "OFF"
        self.status_update.emit(f"Continuous capture: {state}")
        return self._continuous

    @property
    def is_continuous(self):
        return self._continuous

    def _sleep_until_next_capture(self):
        """Sleep in small chunks so stop requests stay responsive.

        Uses adaptive backoff: if the screen hasn't changed for several
        consecutive frames, increase the sleep interval to reduce CPU usage.
        """
        base_sec = float(config["capture_interval_sec"])
        adaptive = config.get("adaptive_interval", True)
        backoff_sec = float(config.get("adaptive_backoff_sec", 5.0))

        if adaptive and self._consecutive_unchanged >= 3:
            interval_sec = max(base_sec, backoff_sec)
        else:
            interval_sec = base_sec

        remaining_ms = int(max(0.1, interval_sec) * 1000)
        while self._running and self._continuous and remaining_ms > 0:
            chunk = min(150, remaining_ms)
            self.msleep(chunk)
            remaining_ms -= chunk

    @staticmethod
    def _entry_signature(entries):
        """Create a stable signature for deduplicating panel updates."""
        return tuple(
            (
                entry.get("original_text", "").strip(),
                entry.get("translated_text", "").strip(),
                entry.get("source_lang", "auto"),
            )
            for entry in entries
        )

    def _emit_translations(self, entries, info):
        """Emit panel data only when it changes."""
        signature = self._entry_signature(entries)
        if self._continuous and signature == self._last_emitted_signature:
            self._consecutive_unchanged += 1
            return

        self._consecutive_unchanged = 0
        self._last_emitted_signature = signature
        self.translations_ready.emit(entries, info)

    def reset_state(self):
        """Reset hidden decision memory after major context changes."""
        self._scratchpad.reset()
        self._last_emitted_signature = None
        self._consecutive_unchanged = 0

    def _process_screenshot(self):
        """Full pipeline: capture -> OCR -> translate -> panel update."""
        try:
            image = self._screen_capture.capture()

            word_results = self._ocr.extract_text_with_boxes(image)
            if not word_results:
                self._emit_translations([], "No text detected in current view")
                self.status_update.emit("No text detected in capture")
                return

            lines = group_words_into_lines(word_results)
            blocks = group_lines_into_blocks(lines)
            if not blocks:
                self._emit_translations([], "No message-like text blocks found")
                self.status_update.emit("No text blocks found")
                return

            original_count = len(blocks)
            blocks = filter_blocks_by_quality(
                blocks,
                image_shape=image.shape,
                min_quality=0.45,
                max_blocks=15,
            )
            if not blocks:
                self._emit_translations([], "No translatable chat content found")
                self.status_update.emit("No quality text blocks found after filtering")
                return

            scratchpad_result = self._scratchpad.observe(image, blocks)
            candidate_blocks = scratchpad_result["selected_blocks"]
            self.status_update.emit(scratchpad_result["summary"])

            if not candidate_blocks:
                info = "Scratchpad found no stable foreign-language chat candidates"
                self._emit_translations([], info)
                self.status_update.emit(info)
                return

            self.status_update.emit(
                f"Found {original_count} blocks, filtered to {len(blocks)} quality blocks, scratchpad kept {len(candidate_blocks)} candidate(s)"
            )

            translated_entries = []
            filtered_count = 0

            for idx, block in enumerate(candidate_blocks):
                text = block["text"].strip()
                if not text:
                    continue

                result = self._translator.process(text)
                if result["filter_reason"]:
                    filtered_count += 1
                    logger.debug(f"Filtered block {idx}: {result['filter_reason']}")
                    continue

                if result["was_translated"]:
                    self.status_update.emit(
                        f"Translating candidate {idx + 1}/{len(candidate_blocks)} ({result['source_lang']})..."
                    )
                    translated_entries.append({
                        "translated_text": result["translated"] or text,
                        "original_text": text,
                        "source_lang": result["source_lang"],
                        "target_lang": result["target_lang"],
                        "quality_score": block.get("quality_score", 0.0),
                        "scratchpad_score": block.get("scratchpad_score", 0.0),
                        "y": block.get("y", 0),
                    })

            translated_entries.sort(key=lambda entry: entry.get("y", 0))

            filter_info = f" ({filtered_count} filtered)" if filtered_count > 0 else ""
            if not translated_entries:
                info = f"No foreign chat detected in current view{filter_info}"
                self._emit_translations([], info)
                self.status_update.emit(info)
                return

            info = f"Translated {len(translated_entries)} message(s){filter_info}"
            self._emit_translations(translated_entries, info)
            self.status_update.emit(info)

        except Exception as exc:
            logger.error(f"Pipeline error: {exc}", exc_info=True)
            self.status_update.emit(f"Error: {exc}")

    def stop(self):
        """Stop the worker cleanly without blocking the GUI thread too long."""
        self._running = False
        self._continuous = False
        self.requestInterruption()
        # Wait briefly — if the thread is stuck in OCR, don't block forever
        if not self.wait(1500):
            logger.warning("Worker thread did not stop in time, terminating")
            self.terminate()
            self.wait(500)


class MainApp(QtWidgets.QApplication):
    """Main application with tray controls and a docked translation panel."""

    def __init__(self, argv):
        super().__init__(argv)
        self.setQuitOnLastWindowClosed(False)

        logger.info("=" * 50)
        logger.info("  Visual Chat Translation Layer")
        logger.info("=" * 50)

        self._overlay = TranslationOverlay()
        self._worker = TranslationWorker()
        self._region_selector = None
        self._keyboard = None
        self._shutting_down = False

        self._worker.translations_ready.connect(self._overlay.update_translation)
        self._worker.status_update.connect(self._on_status_update)

        self._setup_tray()
        self._worker.start()
        self._register_hotkeys()

        # Safety-net shutdown handlers
        self.aboutToQuit.connect(self._cleanup)
        atexit.register(self._cleanup_atexit)

        logger.info(f"Target language: {config['target_language']}")
        logger.info(f"Capture hotkey: {config['toggle_hotkey']}")
        logger.info(f"Press {config['toggle_hotkey']} to capture and translate")
        logger.info("=" * 50)

    def _setup_tray(self):
        """Create system tray icon and menu."""
        self._tray = QtWidgets.QSystemTrayIcon(self)

        pixmap = QtGui.QPixmap(64, 64)
        pixmap.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setBrush(QtGui.QColor(0, 120, 220))
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawEllipse(4, 4, 56, 56)
        painter.setPen(QtGui.QColor(255, 255, 255))
        painter.setFont(QtGui.QFont("Segoe UI", 28, QtGui.QFont.Bold))
        painter.drawText(pixmap.rect(), QtCore.Qt.AlignCenter, "T")
        painter.end()

        self._tray.setIcon(QtGui.QIcon(pixmap))
        self._tray.setToolTip("Visual Chat Translator")

        menu = QtWidgets.QMenu()
        menu.setStyleSheet("""
            QMenu {
                background-color: #1e1e2e;
                color: #ffffff;
                border: 1px solid #333;
                padding: 4px;
            }
            QMenu::item {
                padding: 8px 24px;
            }
            QMenu::item:selected {
                background-color: #0088ff;
            }
            QMenu::separator {
                height: 1px;
                background: #333;
                margin: 4px 12px;
            }
        """)

        capture_action = menu.addAction("Capture & Translate (Ctrl+Shift+T)")
        capture_action.triggered.connect(self._trigger_capture)

        menu.addSeparator()

        self._continuous_action = menu.addAction("Continuous Mode: OFF")
        self._continuous_action.triggered.connect(self._toggle_continuous)

        menu.addSeparator()

        region_action = menu.addAction("Select Capture Region")
        region_action.triggered.connect(self._select_region)

        clear_region = menu.addAction("Reset to Full Screen")
        clear_region.triggered.connect(self._clear_region)

        menu.addSeparator()

        lang_menu = menu.addMenu("Target Language")
        self._lang_group = QtWidgets.QActionGroup(self)
        for lang_name, lang_code in [
            ("English", "en"),
            ("Arabic", "ar"),
            ("French", "fr"),
            ("German", "de"),
            ("Spanish", "es"),
            ("Russian", "ru"),
            ("Chinese", "zh-CN"),
            ("Japanese", "ja"),
            ("Korean", "ko"),
            ("Turkish", "tr"),
            ("Portuguese", "pt"),
            ("Hindi", "hi"),
        ]:
            action = lang_menu.addAction(lang_name)
            action.setCheckable(True)
            action.setChecked(config["target_language"] == lang_code)
            action.setData(lang_code)
            action.triggered.connect(self._change_target_language)
            self._lang_group.addAction(action)

        speed_menu = menu.addMenu("Capture Speed")
        self._speed_group = QtWidgets.QActionGroup(self)
        for label, sec in [
            ("Fast (1s)", 1.0),
            ("Normal (3s)", 3.0),
            ("Slow (5s)", 5.0),
            ("Very Slow (8s)", 8.0),
        ]:
            action = speed_menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(abs(config["capture_interval_sec"] - sec) < 0.1)
            action.setData(sec)
            action.triggered.connect(self._change_speed)
            self._speed_group.addAction(action)

        menu.addSeparator()

        # View mode toggle
        self._compact_action = menu.addAction("Compact View")
        self._compact_action.setCheckable(True)
        self._compact_action.setChecked(config.get("overlay_compact_mode", True))
        self._compact_action.triggered.connect(self._toggle_compact_mode)

        menu.addSeparator()

        show_action = menu.addAction("Show Translation Panel")
        show_action.triggered.connect(self._overlay.show_panel)

        dock_action = menu.addAction("Dock Panel Right")
        dock_action.triggered.connect(self._overlay.dock_right_half)

        menu.addSeparator()

        quit_action = menu.addAction("Quit")
        quit_action.triggered.connect(self._quit)

        self._tray.setContextMenu(menu)
        self._tray.activated.connect(self._on_tray_activated)
        self._tray.show()

    def _register_hotkeys(self):
        """Register global hotkeys without blocking shutdown."""
        try:
            import keyboard

            self._keyboard = keyboard
            keyboard.add_hotkey(config["toggle_hotkey"], self._trigger_capture)
            keyboard.add_hotkey("ctrl+shift+c", self._toggle_continuous)

            logger.info(f"Hotkey '{config['toggle_hotkey']}' registered for capture")
            logger.info("Hotkey 'ctrl+shift+c' registered for continuous mode")
        except ImportError:
            logger.warning("'keyboard' package not found - hotkeys disabled")
        except Exception as exc:
            logger.warning(f"Hotkey registration failed: {exc}")

    def _cleanup_hotkeys(self):
        """Unregister hotkeys on a separate thread with a timeout to avoid deadlock."""
        if self._keyboard is None:
            return

        kb = self._keyboard
        self._keyboard = None

        def _do_unhook():
            try:
                kb.unhook_all_hotkeys()
                kb.unhook_all()
            except Exception:
                pass

        t = threading.Thread(target=_do_unhook, daemon=True)
        t.start()
        t.join(timeout=1.0)

    def _cleanup(self):
        """Shared cleanup logic — safe to call multiple times."""
        if self._shutting_down:
            return
        self._shutting_down = True

        logger.info("Running cleanup...")
        self._cleanup_hotkeys()
        self._worker.stop()

    def _cleanup_atexit(self):
        """Last-resort cleanup registered with atexit."""
        if not self._shutting_down:
            self._cleanup()

    def _trigger_capture(self):
        """Trigger a one-shot capture and show the panel."""
        self._remember_source_window()
        self._overlay.show_panel()
        self._worker.request_capture()

    def _remember_source_window(self):
        """Record the current foreground app as the capture source."""
        hwnd = self._get_foreground_app_hwnd()
        self._worker.set_source_window(hwnd)

    def _toggle_continuous(self):
        """Toggle continuous capture mode."""
        is_on = self._worker.toggle_continuous()
        label = "ON" if is_on else "OFF"
        self._continuous_action.setText(f"Continuous Mode: {label}")

        if is_on:
            self._remember_source_window()
            self._overlay.show_panel()
            self._tray.showMessage(
                "Continuous Mode ON",
                f"Capturing every {config['capture_interval_sec']}s",
                QtWidgets.QSystemTrayIcon.Information,
                2000,
            )
        else:
            self._tray.showMessage(
                "Continuous Mode OFF",
                "Press Ctrl+Shift+T for manual capture",
                QtWidgets.QSystemTrayIcon.Information,
                2000,
            )

    def _get_foreground_app_hwnd(self):
        """Return the current foreground app window, excluding the panel."""
        try:
            import ctypes

            user32 = ctypes.windll.user32
            hwnd = user32.GetForegroundWindow()
            if not hwnd or hwnd == int(self._overlay.winId()):
                return None
            if not user32.IsWindowVisible(hwnd):
                return None
            return hwnd
        except Exception as exc:
            logger.debug(f"Foreground window lookup skipped: {exc}")
            return None

    def _change_target_language(self):
        action = self._lang_group.checkedAction()
        if action:
            lang = action.data()
            config.set("target_language", lang)
            logger.info(f"Target language: {lang}")
            self._worker.reset_state()
            self._worker._translator.clear_cache()

    def _change_speed(self):
        action = self._speed_group.checkedAction()
        if action:
            sec = action.data()
            config.set("capture_interval_sec", sec)
            logger.info(f"Capture interval: {sec}s")

    def _toggle_compact_mode(self):
        compact = self._compact_action.isChecked()
        config.set("overlay_compact_mode", compact)
        # Force re-render with new mode
        if self._worker._last_emitted_signature:
            self._worker._last_emitted_signature = None
            self._worker.request_capture()

    def _select_region(self):
        self._region_selector = RegionSelector()
        self._region_selector.region_selected.connect(self._on_region_selected)
        self._region_selector.show()

    def _on_region_selected(self, x, y, w, h):
        config.set("capture_region", [x, y, w, h])
        self._worker.reset_state()
        logger.info(f"Capture region set: {x},{y} {w}x{h}")
        self._tray.showMessage(
            "Region Set",
            f"Capture region: {w}x{h} at ({x}, {y})\nPress Ctrl+Shift+T to capture",
            QtWidgets.QSystemTrayIcon.Information,
            2000,
        )

    def _clear_region(self):
        config.set("capture_region", None)
        self._worker.reset_state()
        logger.info("Capture region cleared - full screen")

    def _on_status_update(self, message):
        logger.info(message)

    def _on_tray_activated(self, reason):
        if reason == QtWidgets.QSystemTrayIcon.DoubleClick:
            self._trigger_capture()

    def _quit(self):
        logger.info("Shutting down...")
        self._cleanup()
        self._tray.hide()
        self._overlay.close()
        # Deferred quit to let pending events flush
        QtCore.QTimer.singleShot(100, self.quit)


def main():
    """Entry point."""
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

    app = MainApp(sys.argv)
    app._tray.showMessage(
        "Visual Chat Translator",
        f"Press {config['toggle_hotkey']} to capture & translate.\n"
        "Right-click tray icon -> Select Capture Region for the chat area.\n"
        "Ctrl+Shift+C for continuous mode.",
        QtWidgets.QSystemTrayIcon.Information,
        4000,
    )

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
