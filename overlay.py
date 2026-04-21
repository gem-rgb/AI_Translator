"""
overlay.py - Companion panel for live translated text only.

This window is a normal dockable companion panel, not a screenshot overlay.
"""

import ctypes

from PyQt5 import QtCore, QtGui, QtWidgets

from config import config


class TranslationOverlay(QtWidgets.QWidget):
    """Dockable companion panel that shows translated text only."""

    translations_signal = QtCore.pyqtSignal(object, str)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._drag_pos = None
        self._entries = []
        self._hidden_for_capture = False
        self._capture_exclusion_applied = False

        self._setup_window()
        self._setup_ui()
        self.translations_signal.connect(self._on_translations_received)

    def _setup_window(self):
        """Configure the panel as a normal minimizable window."""
        self.setWindowFlags(
            QtCore.Qt.Window
            | QtCore.Qt.FramelessWindowHint
        )

        if config.get("overlay_dock_right", True):
            self.dock_right_half(save_position=False)
        else:
            pos = config.get("overlay_position", [100, 100])
            self.move(pos[0], pos[1])
            self.resize(640, 820)

        self.setMinimumSize(360, 360)

    def _setup_ui(self):
        """Build the translation panel UI."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._container = QtWidgets.QFrame(self)
        self._container.setStyleSheet("""
            QFrame {
                background-color: #f5f7fa;
                border: 1px solid #d6dde7;
                border-radius: 14px;
            }
        """)
        container_layout = QtWidgets.QVBoxLayout(self._container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)

        title_bar = QtWidgets.QFrame(self._container)
        title_bar.setFixedHeight(48)
        title_bar.setStyleSheet("""
            QFrame {
                background-color: #16324f;
                border-top-left-radius: 14px;
                border-top-right-radius: 14px;
                border-bottom-left-radius: 0;
                border-bottom-right-radius: 0;
                border: none;
            }
        """)
        title_layout = QtWidgets.QHBoxLayout(title_bar)
        title_layout.setContentsMargins(14, 0, 10, 0)

        title_label = QtWidgets.QLabel("Live Translation Panel", title_bar)
        title_label.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 14px;
                font-weight: 700;
                font-family: 'Segoe UI';
                background: transparent;
            }
        """)
        title_layout.addWidget(title_label)

        self._info_label = QtWidgets.QLabel("Waiting for capture", title_bar)
        self._info_label.setStyleSheet("""
            QLabel {
                color: rgba(255, 255, 255, 170);
                font-size: 11px;
                font-family: 'Segoe UI';
                background: transparent;
            }
        """)
        title_layout.addWidget(self._info_label)
        title_layout.addStretch()

        dock_btn = QtWidgets.QPushButton("Dock Right", title_bar)
        dock_btn.setCursor(QtCore.Qt.PointingHandCursor)
        dock_btn.setFixedHeight(28)
        dock_btn.setStyleSheet(self._button_style())
        dock_btn.clicked.connect(self.dock_right_half)
        title_layout.addWidget(dock_btn)

        minimize_btn = QtWidgets.QPushButton("Minimize", title_bar)
        minimize_btn.setCursor(QtCore.Qt.PointingHandCursor)
        minimize_btn.setFixedHeight(28)
        minimize_btn.setStyleSheet(self._button_style())
        minimize_btn.clicked.connect(self.showMinimized)
        title_layout.addWidget(minimize_btn)

        hide_btn = QtWidgets.QPushButton("Hide", title_bar)
        hide_btn.setCursor(QtCore.Qt.PointingHandCursor)
        hide_btn.setFixedHeight(28)
        hide_btn.setStyleSheet(self._button_style(soft=True))
        hide_btn.clicked.connect(self.hide)
        title_layout.addWidget(hide_btn)

        container_layout.addWidget(title_bar)

        self._summary_bar = QtWidgets.QLabel(
            "Select a region or start continuous mode to watch translated chat here.",
            self._container,
        )
        self._summary_bar.setWordWrap(True)
        self._summary_bar.setContentsMargins(16, 12, 16, 12)
        self._summary_bar.setStyleSheet("""
            QLabel {
                color: #35516d;
                background-color: #e7f0fa;
                border: none;
                border-bottom: 1px solid #d6dde7;
                font-size: 11px;
                font-family: 'Segoe UI';
            }
        """)
        container_layout.addWidget(self._summary_bar)

        self._scroll = QtWidgets.QScrollArea(self._container)
        self._scroll.setWidgetResizable(True)
        self._scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background: #f5f7fa;
            }
            QScrollBar:vertical {
                width: 8px;
                background: #edf2f7;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: #b7c6d8;
                border-radius: 4px;
                min-height: 30px;
            }
        """)

        self._cards_widget = QtWidgets.QWidget()
        self._cards_layout = QtWidgets.QVBoxLayout(self._cards_widget)
        self._cards_layout.setContentsMargins(16, 16, 16, 16)
        self._cards_layout.setSpacing(12)

        self._cards_layout.addWidget(self._build_empty_state())
        self._cards_layout.addStretch()

        self._scroll.setWidget(self._cards_widget)
        container_layout.addWidget(self._scroll)

        self._status_bar = QtWidgets.QLabel(
            "Ready",
            self._container,
        )
        self._status_bar.setFixedHeight(28)
        self._status_bar.setStyleSheet("""
            QLabel {
                color: #50657d;
                background-color: #eef3f7;
                border: none;
                border-top: 1px solid #d6dde7;
                border-bottom-left-radius: 14px;
                border-bottom-right-radius: 14px;
                font-size: 11px;
                font-family: 'Segoe UI';
                padding-left: 14px;
            }
        """)
        container_layout.addWidget(self._status_bar)

        layout.addWidget(self._container)

    @staticmethod
    def _button_style(soft=False):
        """Shared title-bar button styling."""
        if soft:
            bg = "rgba(255, 255, 255, 0.12)"
            hover = "rgba(255, 255, 255, 0.22)"
        else:
            bg = "#2f6ea6"
            hover = "#3e82bf"

        return f"""
            QPushButton {{
                background-color: {bg};
                color: white;
                border: none;
                border-radius: 6px;
                font-size: 11px;
                font-family: 'Segoe UI';
                font-weight: 600;
                padding: 0 12px;
            }}
            QPushButton:hover {{
                background-color: {hover};
            }}
        """

    def showEvent(self, event):
        """Re-apply capture exclusion when the panel becomes visible."""
        super().showEvent(event)
        self._apply_capture_exclusion()

    def _apply_capture_exclusion(self):
        """Best-effort request to exclude this panel from Windows capture."""
        if self._capture_exclusion_applied:
            return

        try:
            user32 = ctypes.windll.user32
            WDA_EXCLUDEFROMCAPTURE = 0x11
            user32.SetWindowDisplayAffinity(int(self.winId()), WDA_EXCLUDEFROMCAPTURE)
            self._capture_exclusion_applied = True
        except Exception:
            self._capture_exclusion_applied = False

    def update_translation(self, entries, info=""):
        """Thread-safe update for the panel contents."""
        self.translations_signal.emit(entries, info)

    def _on_translations_received(self, entries, info):
        """Render translated entries on the GUI thread."""
        self._entries = entries or []
        self._info_label.setText(info or "Updated")

        count = len(self._entries)
        if count:
            self._summary_bar.setText(
                f"Showing {count} translated item{'s' if count != 1 else ''} from the current app view."
            )
        else:
            self._summary_bar.setText(
                "No foreign-language chat detected in the current app view."
            )

        self._rebuild_cards()
        self._status_bar.setText(info or "Ready")
        self.show_panel()

    def _rebuild_cards(self):
        """Rebuild the list of translated message cards."""
        while self._cards_layout.count():
            item = self._cards_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        if not self._entries:
            self._cards_layout.addWidget(self._build_empty_state())
            self._cards_layout.addStretch()
            return

        for index, entry in enumerate(self._entries, start=1):
            self._cards_layout.addWidget(self._build_entry_card(index, entry))

        self._cards_layout.addStretch()
        self._scroll.verticalScrollBar().setValue(0)

    def _build_empty_state(self):
        """Create the placeholder shown when nothing is translated yet."""
        empty_state = QtWidgets.QLabel(
            "No translated messages yet.\n\nThe panel will show only the text that looks like real foreign-language chat content.",
            self._cards_widget,
        )
        empty_state.setAlignment(QtCore.Qt.AlignCenter)
        empty_state.setWordWrap(True)
        empty_state.setStyleSheet("""
            QLabel {
                color: #60758f;
                background: white;
                border: 1px dashed #c7d3e0;
                border-radius: 12px;
                padding: 28px 18px;
                font-size: 12px;
                font-family: 'Segoe UI';
            }
        """)
        return empty_state

    def _build_entry_card(self, index, entry):
        """Create one translated message card."""
        frame = QtWidgets.QFrame(self._cards_widget)
        frame.setStyleSheet("""
            QFrame {
                background: white;
                border: 1px solid #dce4ee;
                border-radius: 12px;
            }
        """)
        layout = QtWidgets.QVBoxLayout(frame)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        meta = QtWidgets.QLabel(
            f"Message {index:02d}  |  {entry.get('source_lang', 'auto').upper()} -> {entry.get('target_lang', 'EN').upper()}",
            frame,
        )
        meta.setStyleSheet("""
            QLabel {
                color: #5d7087;
                font-size: 10px;
                font-weight: 700;
                font-family: 'Segoe UI';
                letter-spacing: 0.4px;
                border: none;
            }
        """)
        layout.addWidget(meta)

        translated = QtWidgets.QLabel(entry.get("translated_text", ""), frame)
        translated.setWordWrap(True)
        translated.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        translated.setStyleSheet("""
            QLabel {
                color: #162636;
                font-size: 14px;
                font-weight: 600;
                font-family: 'Segoe UI';
                border: none;
            }
        """)
        layout.addWidget(translated)

        original_text = entry.get("original_text", "").strip()
        if original_text:
            original = QtWidgets.QLabel(original_text, frame)
            original.setWordWrap(True)
            original.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            original.setStyleSheet("""
                QLabel {
                    color: #6b7d91;
                    font-size: 11px;
                    font-family: 'Segoe UI';
                    border: none;
                }
            """)
            layout.addWidget(original)

        return frame

    def set_status(self, text):
        """Update the summary status quietly."""
        QtCore.QMetaObject.invokeMethod(
            self._status_bar,
            "setText",
            QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(str, text),
        )

    def show_panel(self):
        """Show or restore the panel without an animation."""
        if self.isMinimized():
            self.showNormal()
        else:
            self.show()
        self.raise_()
        self._apply_capture_exclusion()

    def prepare_for_capture(self):
        """Temporarily hide the panel from view if needed."""
        if self.isVisible() and not self.isMinimized():
            self.hide()
            self._hidden_for_capture = True

    def restore_after_capture(self):
        """Restore the panel after background capture completes."""
        if self._hidden_for_capture:
            self._hidden_for_capture = False
            if self._entries:
                self.show_panel()

    def dock_right_half(self, save_position=True):
        """Snap the panel to the right side of the primary screen."""
        screen = QtWidgets.QApplication.primaryScreen()
        if screen is None:
            return

        available = screen.availableGeometry()
        width_ratio = float(config.get("overlay_width_ratio", 0.42))
        width_ratio = min(0.6, max(0.25, width_ratio))
        panel_width = int(available.width() * width_ratio)

        self.setGeometry(
            available.right() - panel_width + 1,
            available.y(),
            panel_width,
            available.height(),
        )

        if save_position:
            config.set("overlay_position", [self.x(), self.y()])

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and event.pos().y() < 52:
            self._drag_pos = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if self._drag_pos and event.buttons() == QtCore.Qt.LeftButton:
            self.move(event.globalPos() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and self._drag_pos:
            self._drag_pos = None
            config.set("overlay_position", [self.x(), self.y()])
            event.accept()


class RegionSelector(QtWidgets.QWidget):
    """Full-screen overlay for selecting a capture region via drag."""

    region_selected = QtCore.pyqtSignal(int, int, int, int)

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

            painter.setPen(QtGui.QColor(255, 255, 255))
            painter.setFont(QtGui.QFont("Segoe UI", 10))
            painter.drawText(
                rect.topLeft() + QtCore.QPoint(4, -6),
                f"{rect.width()} x {rect.height()}",
            )

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
