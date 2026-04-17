"""
main.py — Pipeline orchestrator for visual chat translation.

New pipeline:
  1. Capture screenshot (full screen or selected region)
  2. OCR to extract text with bounding boxes
  3. Group words into lines → lines into text blocks (chat bubbles)
  4. Detect language of each block
  5. Translate non-English blocks
  6. Paint translations over original text on the screenshot
  7. Display the translated image in the overlay

Runs as a system tray application with hotkey activation.
"""

import sys
import os
import logging
import threading
import time

from PyQt5 import QtWidgets, QtCore, QtGui

from config import config
from capture import ScreenCapture, ClipboardMonitor
from ocr import OCREngine
from translator import Translator
from renderer import group_words_into_lines, group_lines_into_blocks, render_translations
from overlay import TranslationOverlay, RegionSelector

# ── Logging setup ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("translator")


class TranslationWorker(QtCore.QThread):
    """Background worker that runs the visual translation pipeline."""

    # Signals
    image_ready = QtCore.pyqtSignal(object, object, str)   # translated_img, original_img, info
    status_update = QtCore.pyqtSignal(str)                  # status message

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = False
        self._capture_requested = False  # one-shot capture trigger
        self._continuous = False         # continuous capture mode
        self._screen_capture = ScreenCapture()
        self._ocr = OCREngine()
        self._translator = Translator()

    def run(self):
        """Main worker loop."""
        self._running = True
        self.status_update.emit("Worker started — press Ctrl+Shift+T to capture")

        while self._running:
            if self._capture_requested or self._continuous:
                self._capture_requested = False
                self._process_screenshot()

                if self._continuous:
                    interval = int(config["capture_interval_sec"] * 1000)
                    self.msleep(interval)
                    continue

            self.msleep(200)  # idle wait

    def request_capture(self):
        """Trigger a one-shot capture."""
        self._capture_requested = True
        self.status_update.emit("📸 Capturing...")

    def toggle_continuous(self):
        """Toggle continuous capture mode."""
        self._continuous = not self._continuous
        state = "ON" if self._continuous else "OFF"
        self.status_update.emit(f"Continuous capture: {state}")
        return self._continuous

    @property
    def is_continuous(self):
        return self._continuous

    def _process_screenshot(self):
        """Full pipeline: capture → OCR → translate → render."""
        try:
            # 1. Capture screenshot
            self.status_update.emit("📸 Capturing screen...")
            image = self._screen_capture.capture()
            original = image.copy()

            # 2. OCR with bounding boxes
            self.status_update.emit("🔍 Extracting text (OCR)...")
            word_results = self._ocr.extract_text_with_boxes(image)

            if not word_results:
                self.status_update.emit("⚠ No text detected in capture")
                return

            # 3. Group words → lines → blocks
            lines = group_words_into_lines(word_results)
            blocks = group_lines_into_blocks(lines)

            if not blocks:
                self.status_update.emit("⚠ No text blocks found")
                return

            self.status_update.emit(
                f"📝 Found {len(blocks)} text blocks — detecting languages..."
            )

            # 4. Detect language & translate non-English blocks
            translations = {}
            translated_count = 0

            for idx, block in enumerate(blocks):
                text = block["text"].strip()
                if not text or len(text) < 3:
                    continue

                # Detect language
                needs, source_lang = self._translator.needs_translation(text)

                if needs:
                    self.status_update.emit(
                        f"🌐 Translating block {idx + 1}/{len(blocks)} "
                        f"({source_lang})..."
                    )
                    translated_text = self._translator.translate(text, source_lang=source_lang)
                    translations[idx] = {
                        "translated_text": translated_text,
                        "source_lang": source_lang,
                    }
                    translated_count += 1

            if not translations:
                self.status_update.emit("✅ All text is already in English — nothing to translate")
                # Still show the original image
                self.image_ready.emit(original, original,
                                       f"All {len(blocks)} blocks are English")
                return

            # 5. Render translations onto image
            self.status_update.emit("🎨 Rendering translations...")
            translated_img = render_translations(image, blocks, translations)

            # 6. Emit result
            info = f"Translated {translated_count}/{len(blocks)} blocks"
            self.image_ready.emit(translated_img, original, info)
            self.status_update.emit(f"✅ {info}")

        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            self.status_update.emit(f"❌ Error: {e}")

    def stop(self):
        self._running = False
        self._screen_capture.close()
        self.wait(3000)


class MainApp(QtWidgets.QApplication):
    """Main application with system tray and hotkey support."""

    def __init__(self, argv):
        super().__init__(argv)
        self.setQuitOnLastWindowClosed(False)

        logger.info("═" * 50)
        logger.info("  Visual Chat Translation Layer")
        logger.info("═" * 50)

        # ── Components ──
        self._overlay = TranslationOverlay()
        self._worker = TranslationWorker()
        self._region_selector = None
        self._hotkey_thread = None

        # ── Connect signals ──
        self._worker.image_ready.connect(self._overlay.update_translation)
        self._worker.status_update.connect(self._on_status_update)
        self._worker.status_update.connect(self._overlay.set_status)

        # ── System Tray ──
        self._setup_tray()

        # ── Start worker thread ──
        self._worker.start()

        # ── Register global hotkey ──
        self._register_hotkeys()

        logger.info(f"Target language: {config['target_language']}")
        logger.info(f"Capture hotkey: {config['toggle_hotkey']}")
        logger.info(f"Press {config['toggle_hotkey']} to capture & translate")
        logger.info("═" * 50)

    def _setup_tray(self):
        """Create system tray icon and menu."""
        self._tray = QtWidgets.QSystemTrayIcon(self)

        # Create icon
        pixmap = QtGui.QPixmap(64, 64)
        pixmap.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setBrush(QtGui.QColor(0, 120, 220))
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawEllipse(4, 4, 56, 56)
        painter.setPen(QtGui.QColor(255, 255, 255))
        font = QtGui.QFont("Segoe UI", 28, QtGui.QFont.Bold)
        painter.setFont(font)
        painter.drawText(pixmap.rect(), QtCore.Qt.AlignCenter, "T")
        painter.end()

        self._tray.setIcon(QtGui.QIcon(pixmap))
        self._tray.setToolTip("Visual Chat Translator")

        # ── Tray Menu ──
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

        # Capture now
        capture_action = menu.addAction("📸  Capture & Translate (Ctrl+Shift+T)")
        capture_action.triggered.connect(self._trigger_capture)

        menu.addSeparator()

        # Continuous mode
        self._continuous_action = menu.addAction("🔄  Continuous Mode: OFF")
        self._continuous_action.triggered.connect(self._toggle_continuous)

        menu.addSeparator()

        # Select region
        region_action = menu.addAction("🔲  Select Capture Region")
        region_action.triggered.connect(self._select_region)

        clear_region = menu.addAction("🖥  Reset to Full Screen")
        clear_region.triggered.connect(self._clear_region)

        menu.addSeparator()

        # Target language
        lang_menu = menu.addMenu("🌍  Target Language")
        self._lang_group = QtWidgets.QActionGroup(self)
        for lang_name, lang_code in [("English", "en"), ("Arabic", "ar"),
                                      ("French", "fr"), ("German", "de"),
                                      ("Spanish", "es"), ("Russian", "ru"),
                                      ("Chinese", "zh-CN"), ("Japanese", "ja"),
                                      ("Korean", "ko"), ("Turkish", "tr"),
                                      ("Portuguese", "pt"), ("Hindi", "hi")]:
            action = lang_menu.addAction(lang_name)
            action.setCheckable(True)
            action.setChecked(config["target_language"] == lang_code)
            action.setData(lang_code)
            action.triggered.connect(self._change_target_language)
            self._lang_group.addAction(action)

        # Capture speed
        speed_menu = menu.addMenu("⏱  Capture Speed (continuous)")
        self._speed_group = QtWidgets.QActionGroup(self)
        for label, sec in [("Fast (1s)", 1.0), ("Normal (2s)", 2.0),
                           ("Slow (3s)", 3.0), ("Very Slow (5s)", 5.0)]:
            action = speed_menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(abs(config["capture_interval_sec"] - sec) < 0.1)
            action.setData(sec)
            action.triggered.connect(self._change_speed)
            self._speed_group.addAction(action)

        menu.addSeparator()

        # Show/hide overlay
        show_action = menu.addAction("👁  Show Overlay")
        show_action.triggered.connect(self._overlay.fade_in)

        menu.addSeparator()

        # Quit
        quit_action = menu.addAction("❌  Quit")
        quit_action.triggered.connect(self._quit)

        self._tray.setContextMenu(menu)
        self._tray.activated.connect(self._on_tray_activated)
        self._tray.show()

    def _register_hotkeys(self):
        """Register global hotkeys in a background thread."""
        def _hotkey_listener():
            try:
                import keyboard

                # Ctrl+Shift+T → capture & translate
                hotkey = config["toggle_hotkey"]
                keyboard.add_hotkey(hotkey, self._trigger_capture)
                logger.info(f"Hotkey '{hotkey}' registered for capture")

                # Ctrl+Shift+C → toggle continuous mode
                keyboard.add_hotkey("ctrl+shift+c", self._toggle_continuous)
                logger.info("Hotkey 'ctrl+shift+c' registered for continuous mode")

                keyboard.wait()
            except ImportError:
                logger.warning("'keyboard' package not found — hotkeys disabled")
            except Exception as e:
                logger.warning(f"Hotkey registration failed: {e}")

        self._hotkey_thread = threading.Thread(target=_hotkey_listener, daemon=True)
        self._hotkey_thread.start()

    # ── Actions ──

    def _trigger_capture(self):
        """Trigger a one-shot capture & translate."""
        self._worker.request_capture()
        self._overlay.fade_in()
        self._overlay.set_status("📸 Capturing & translating...")

    def _toggle_continuous(self):
        """Toggle continuous capture mode."""
        is_on = self._worker.toggle_continuous()
        label = "ON" if is_on else "OFF"
        self._continuous_action.setText(f"🔄  Continuous Mode: {label}")

        if is_on:
            self._overlay.fade_in()
            self._tray.showMessage(
                "Continuous Mode ON",
                f"Capturing every {config['capture_interval_sec']}s",
                QtWidgets.QSystemTrayIcon.Information, 2000,
            )
        else:
            self._tray.showMessage(
                "Continuous Mode OFF",
                "Press Ctrl+Shift+T for manual capture",
                QtWidgets.QSystemTrayIcon.Information, 2000,
            )

    def _change_target_language(self):
        action = self._lang_group.checkedAction()
        if action:
            lang = action.data()
            config.set("target_language", lang)
            logger.info(f"Target language: {lang}")
            self._worker._translator.clear_cache()

    def _change_speed(self):
        action = self._speed_group.checkedAction()
        if action:
            sec = action.data()
            config.set("capture_interval_sec", sec)
            logger.info(f"Capture interval: {sec}s")

    def _select_region(self):
        self._region_selector = RegionSelector()
        self._region_selector.region_selected.connect(self._on_region_selected)
        self._region_selector.show()

    def _on_region_selected(self, x, y, w, h):
        config.set("capture_region", [x, y, w, h])
        logger.info(f"Capture region set: {x},{y} {w}×{h}")
        self._tray.showMessage(
            "Region Set",
            f"Capture region: {w}×{h} at ({x}, {y})\nPress Ctrl+Shift+T to capture",
            QtWidgets.QSystemTrayIcon.Information, 2000,
        )

    def _clear_region(self):
        config.set("capture_region", None)
        logger.info("Capture region cleared — full screen")

    def _on_status_update(self, message):
        logger.info(message)

    def _on_tray_activated(self, reason):
        if reason == QtWidgets.QSystemTrayIcon.DoubleClick:
            self._trigger_capture()

    def _quit(self):
        logger.info("Shutting down...")
        self._worker.stop()
        self._tray.hide()
        self.quit()


def main():
    """Entry point."""
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

    app = MainApp(sys.argv)

    app._tray.showMessage(
        "Visual Chat Translator",
        f"Press {config['toggle_hotkey']} to capture & translate.\n"
        f"Right-click tray icon → Select Capture Region for chat area.\n"
        f"Ctrl+Shift+C for continuous mode.",
        QtWidgets.QSystemTrayIcon.Information, 4000,
    )

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
