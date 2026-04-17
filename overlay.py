"""
overlay.py — Image-based overlay for displaying translated screenshots.

Shows the translated screenshot in a floating, resizable window.
Non-English chat messages are painted over with their English translations
directly on the captured image.
"""

import sys
import cv2
import numpy as np

from PyQt5 import QtWidgets, QtCore, QtGui

from config import config


class TranslationOverlay(QtWidgets.QWidget):
    """Floating window that displays the translated screenshot image."""

    # Signal to update image from non-GUI threads
    image_signal = QtCore.pyqtSignal(object, object, str)  # translated_img, original_img, info

    def __init__(self, parent=None):
        super().__init__(parent)

        self._drag_pos = None
        self._resize_edge = None
        self._is_visible = False
        self._current_translated = None
        self._current_original = None
        self._show_original = False  # toggle between original/translated

        self._setup_window()
        self._setup_ui()
        self._setup_animations()

        # Connect signal for thread-safe image updates
        self.image_signal.connect(self._on_image_received)

    def _setup_window(self):
        """Configure window flags for overlay behavior."""
        self.setWindowFlags(
            QtCore.Qt.WindowStaysOnTopHint
            | QtCore.Qt.FramelessWindowHint
            | QtCore.Qt.Tool
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setAttribute(QtCore.Qt.WA_ShowWithoutActivating)

        # Load saved position
        pos = config.get("overlay_position", [100, 100])
        self.move(pos[0], pos[1])
        self.resize(700, 500)
        self.setMinimumSize(300, 200)

    def _setup_ui(self):
        """Build the overlay UI."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Container with dark background
        self._container = QtWidgets.QFrame(self)
        self._container.setStyleSheet("""
            QFrame {
                background-color: rgba(20, 20, 30, 240);
                border-radius: 12px;
                border: 1px solid rgba(0, 150, 255, 100);
            }
        """)
        container_layout = QtWidgets.QVBoxLayout(self._container)
        container_layout.setContentsMargins(2, 2, 2, 2)
        container_layout.setSpacing(0)

        # ── Title bar ──
        title_bar = QtWidgets.QFrame(self._container)
        title_bar.setFixedHeight(36)
        title_bar.setStyleSheet("""
            QFrame {
                background-color: rgba(0, 120, 220, 180);
                border-top-left-radius: 12px;
                border-top-right-radius: 12px;
                border-bottom-left-radius: 0px;
                border-bottom-right-radius: 0px;
                border: none;
            }
        """)
        title_layout = QtWidgets.QHBoxLayout(title_bar)
        title_layout.setContentsMargins(12, 0, 8, 0)

        title_label = QtWidgets.QLabel("🌐 Translated Chat", title_bar)
        title_label.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 13px;
                font-weight: bold;
                font-family: 'Segoe UI', Arial;
                border: none;
                background: transparent;
            }
        """)
        title_layout.addWidget(title_label)

        self._info_label = QtWidgets.QLabel("", title_bar)
        self._info_label.setStyleSheet("""
            QLabel {
                color: rgba(255, 255, 255, 180);
                font-size: 11px;
                font-family: 'Segoe UI', Arial;
                border: none;
                background: transparent;
            }
        """)
        title_layout.addWidget(self._info_label)
        title_layout.addStretch()

        # Toggle button (original / translated)
        self._toggle_btn = QtWidgets.QPushButton("👁 Original", title_bar)
        self._toggle_btn.setFixedSize(90, 26)
        self._toggle_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self._toggle_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(255, 255, 255, 30);
                color: white;
                border: 1px solid rgba(255,255,255,60);
                border-radius: 4px;
                font-size: 11px;
                font-family: 'Segoe UI', Arial;
                padding: 2px 8px;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 60);
            }
        """)
        self._toggle_btn.clicked.connect(self._toggle_view)
        title_layout.addWidget(self._toggle_btn)

        # Close button
        close_btn = QtWidgets.QPushButton("✕", title_bar)
        close_btn.setFixedSize(26, 26)
        close_btn.setCursor(QtCore.Qt.PointingHandCursor)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(255, 60, 60, 100);
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgba(255, 60, 60, 200);
            }
        """)
        close_btn.clicked.connect(self.fade_out)
        title_layout.addWidget(close_btn)

        container_layout.addWidget(title_bar)

        # ── Image display area ──
        self._scroll = QtWidgets.QScrollArea(self._container)
        self._scroll.setWidgetResizable(True)
        self._scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: rgba(30, 30, 40, 255);
                border-bottom-left-radius: 12px;
                border-bottom-right-radius: 12px;
            }
            QScrollBar:vertical {
                background: rgba(40, 40, 50, 200);
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: rgba(0, 150, 255, 150);
                border-radius: 4px;
                min-height: 30px;
            }
            QScrollBar:horizontal {
                background: rgba(40, 40, 50, 200);
                height: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:horizontal {
                background: rgba(0, 150, 255, 150);
                border-radius: 4px;
                min-width: 30px;
            }
        """)

        self._image_label = QtWidgets.QLabel()
        self._image_label.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        self._image_label.setStyleSheet("background: transparent; border: none;")
        self._scroll.setWidget(self._image_label)

        container_layout.addWidget(self._scroll)

        # ── Status bar ──
        self._status_bar = QtWidgets.QLabel("Ready — Press Ctrl+Shift+T to capture", self._container)
        self._status_bar.setFixedHeight(24)
        self._status_bar.setStyleSheet("""
            QLabel {
                color: rgba(255, 255, 255, 130);
                font-size: 10px;
                font-family: 'Segoe UI', Arial;
                padding-left: 12px;
                background-color: rgba(15, 15, 20, 200);
                border-bottom-left-radius: 12px;
                border-bottom-right-radius: 12px;
                border: none;
            }
        """)
        container_layout.addWidget(self._status_bar)

        layout.addWidget(self._container)

    def _setup_animations(self):
        """Setup fade-in / fade-out animations."""
        self._opacity_effect = QtWidgets.QGraphicsOpacityEffect(self)
        self._opacity_effect.setOpacity(0.0)
        self.setGraphicsEffect(self._opacity_effect)

        self._fade_anim = QtCore.QPropertyAnimation(self._opacity_effect, b"opacity")
        self._fade_anim.setDuration(300)
        self._fade_anim.setEasingCurve(QtCore.QEasingCurve.InOutQuad)

    def update_translation(self, translated_img, original_img, info=""):
        """Thread-safe method to update the displayed image.

        Args:
            translated_img: numpy BGR image with translations painted on.
            original_img: numpy BGR original screenshot.
            info: Status info string.
        """
        self.image_signal.emit(translated_img, original_img, info)

    def _on_image_received(self, translated_img, original_img, info):
        """Handle image update on the GUI thread."""
        self._current_translated = translated_img
        self._current_original = original_img
        self._show_original = False
        self._toggle_btn.setText("👁 Original")

        self._display_image(translated_img)
        self._info_label.setText(info)
        self._status_bar.setText(f"✅ {info}")

        self.fade_in()

    def _display_image(self, img):
        """Display a numpy BGR image in the image label."""
        if img is None:
            return

        # Scale image to fit window width while maintaining aspect ratio
        display_width = self.width() - 20  # padding
        h, w = img.shape[:2]
        scale = display_width / w if w > display_width else 1.0
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Convert BGR to RGB
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w

        qt_img = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qt_img)
        self._image_label.setPixmap(pixmap)
        self._image_label.adjustSize()

    def _toggle_view(self):
        """Toggle between translated and original view."""
        self._show_original = not self._show_original
        if self._show_original and self._current_original is not None:
            self._display_image(self._current_original)
            self._toggle_btn.setText("🌐 Translated")
            self._status_bar.setText("📷 Showing original — click to see translation")
        elif self._current_translated is not None:
            self._display_image(self._current_translated)
            self._toggle_btn.setText("👁 Original")
            self._status_bar.setText("✅ Showing translated view")

    def set_status(self, text):
        """Update status bar text (thread-safe via QMetaObject)."""
        QtCore.QMetaObject.invokeMethod(
            self._status_bar, "setText",
            QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(str, text),
        )

    def fade_in(self):
        """Fade the overlay in."""
        self._fade_anim.stop()
        self._fade_anim.setStartValue(self._opacity_effect.opacity())
        self._fade_anim.setEndValue(1.0)
        self._fade_anim.start()
        self.show()
        self._is_visible = True

    def fade_out(self):
        """Fade the overlay out."""
        self._fade_anim.stop()
        self._fade_anim.setStartValue(self._opacity_effect.opacity())
        self._fade_anim.setEndValue(0.0)
        self._fade_anim.finished.connect(self._on_fade_out_done)
        self._fade_anim.start()

    def _on_fade_out_done(self):
        """Hide window after fade out completes."""
        try:
            self._fade_anim.finished.disconnect(self._on_fade_out_done)
        except TypeError:
            pass
        if self._opacity_effect.opacity() < 0.1:
            self.hide()
            self._is_visible = False

    # ── Dragging support (title bar area) ──

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and event.pos().y() < 40:
            self._drag_pos = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if self._drag_pos and event.buttons() == QtCore.Qt.LeftButton:
            self.move(event.globalPos() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and self._drag_pos:
            self._drag_pos = None
            pos = self.pos()
            config.set("overlay_position", [pos.x(), pos.y()])
            event.accept()

    def resizeEvent(self, event):
        """Re-render image when window is resized."""
        super().resizeEvent(event)
        if self._show_original and self._current_original is not None:
            self._display_image(self._current_original)
        elif self._current_translated is not None:
            self._display_image(self._current_translated)


class RegionSelector(QtWidgets.QWidget):
    """Full-screen overlay for selecting a capture region via drag."""

    region_selected = QtCore.pyqtSignal(int, int, int, int)  # x, y, w, h

    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            QtCore.Qt.WindowStaysOnTopHint
            | QtCore.Qt.FramelessWindowHint
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setStyleSheet("background-color: rgba(0, 0, 0, 80);")

        screen = QtWidgets.QApplication.primaryScreen().geometry()
        self.setGeometry(screen)

        self._origin = None
        self._current = None
        self.setCursor(QtCore.Qt.CrossCursor)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor(0, 0, 0, 80))

        if self._origin and self._current:
            rect = QtCore.QRect(self._origin, self._current).normalized()
            painter.setCompositionMode(QtGui.QPainter.CompositionMode_Clear)
            painter.fillRect(rect, QtCore.Qt.transparent)

            painter.setCompositionMode(QtGui.QPainter.CompositionMode_SourceOver)
            pen = QtGui.QPen(QtGui.QColor(0, 168, 255), 2, QtCore.Qt.SolidLine)
            painter.setPen(pen)
            painter.drawRect(rect)

            w = rect.width()
            h = rect.height()
            painter.setPen(QtGui.QColor(255, 255, 255))
            painter.setFont(QtGui.QFont("Segoe UI", 10))
            painter.drawText(rect.topLeft() + QtCore.QPoint(4, -6), f"{w} × {h}")

    def mousePressEvent(self, event):
        self._origin = event.pos()
        self._current = event.pos()
        self.update()

    def mouseMoveEvent(self, event):
        self._current = event.pos()
        self.update()

    def mouseReleaseEvent(self, event):
        if self._origin and self._current:
            rect = QtCore.QRect(self._origin, self._current).normalized()
            if rect.width() > 20 and rect.height() > 20:
                self.region_selected.emit(rect.x(), rect.y(), rect.width(), rect.height())
        self.close()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            self.close()
