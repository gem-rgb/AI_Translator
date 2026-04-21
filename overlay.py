"""
overlay.py - Compact floating translation panel.

This window is a lightweight, semi-transparent tool window that shows only
translated text.  It stays on top but does not steal focus, block clicks
on the area beneath it, or prevent minimization.
"""

import ctypes

from PyQt5 import QtCore, QtGui, QtWidgets

from config import config


class TranslationOverlay(QtWidgets.QWidget):
    """Compact floating panel that shows translated text only."""

    translations_signal = QtCore.pyqtSignal(object, str)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._drag_pos = None
        self._resize_edge = None
        self._entries = []
        self._hidden_for_capture = False
        self._capture_exclusion_applied = False
        self._first_show = True

        self._setup_window()
        self._setup_ui()
        self.translations_signal.connect(self._on_translations_received)

    # ------------------------------------------------------------------
    # Window setup
    # ------------------------------------------------------------------

    def _setup_window(self):
        """Configure the panel as a lightweight tool window."""
        self.setWindowFlags(
            QtCore.Qt.Tool                   # no taskbar entry, lightweight
            | QtCore.Qt.FramelessWindowHint  # custom chrome
            | QtCore.Qt.WindowStaysOnTopHint # always visible
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)

        # Apply opacity from config
        opacity = float(config.get("overlay_opacity", 0.92))
        self.setWindowOpacity(min(1.0, max(0.3, opacity)))

        if config.get("overlay_dock_right", True):
            self.dock_right_half(save_position=False)
        else:
            pos = config.get("overlay_position", [100, 100])
            self.move(pos[0], pos[1])
            self.resize(340, 600)

        self.setMinimumSize(260, 200)

    def _setup_ui(self):
        """Build the clean translation panel UI."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._container = QtWidgets.QFrame(self)
        self._container.setObjectName("panelContainer")
        self._container.setStyleSheet("""
            QFrame#panelContainer {
                background-color: rgba(22, 27, 38, 235);
                border: 1px solid rgba(70, 85, 110, 180);
                border-radius: 12px;
            }
        """)
        container_layout = QtWidgets.QVBoxLayout(self._container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)

        # ---- Compact title bar ----
        title_bar = QtWidgets.QFrame(self._container)
        title_bar.setObjectName("titleBar")
        title_bar.setFixedHeight(34)
        title_bar.setStyleSheet("""
            QFrame#titleBar {
                background-color: rgba(16, 20, 30, 240);
                border-top-left-radius: 12px;
                border-top-right-radius: 12px;
                border-bottom-left-radius: 0;
                border-bottom-right-radius: 0;
                border: none;
            }
        """)
        title_layout = QtWidgets.QHBoxLayout(title_bar)
        title_layout.setContentsMargins(12, 0, 6, 0)

        # Small indicator dot
        dot = QtWidgets.QLabel("●", title_bar)
        dot.setStyleSheet("QLabel { color: #00c896; font-size: 8px; background: transparent; border: none; }")
        title_layout.addWidget(dot)

        title_label = QtWidgets.QLabel("Translations", title_bar)
        title_label.setStyleSheet("""
            QLabel {
                color: rgba(255, 255, 255, 210);
                font-size: 12px;
                font-weight: 600;
                font-family: 'Segoe UI';
                background: transparent;
                border: none;
            }
        """)
        title_layout.addWidget(title_label)

        self._info_label = QtWidgets.QLabel("", title_bar)
        self._info_label.setStyleSheet("""
            QLabel {
                color: rgba(255, 255, 255, 100);
                font-size: 10px;
                font-family: 'Segoe UI';
                background: transparent;
                border: none;
            }
        """)
        title_layout.addWidget(self._info_label)
        title_layout.addStretch()

        minimize_btn = self._make_btn("—", title_bar)
        minimize_btn.clicked.connect(self.showMinimized)
        title_layout.addWidget(minimize_btn)

        hide_btn = self._make_btn("✕", title_bar)
        hide_btn.clicked.connect(self.hide)
        title_layout.addWidget(hide_btn)

        container_layout.addWidget(title_bar)

        # ---- Scroll area for translated text ----
        self._scroll = QtWidgets.QScrollArea(self._container)
        self._scroll.setWidgetResizable(True)
        self._scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background: transparent;
            }
            QScrollBar:vertical {
                width: 5px;
                background: transparent;
                border-radius: 2px;
            }
            QScrollBar::handle:vertical {
                background: rgba(255, 255, 255, 40);
                border-radius: 2px;
                min-height: 24px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0;
            }
        """)

        self._cards_widget = QtWidgets.QWidget()
        self._cards_widget.setStyleSheet("background: transparent;")
        self._cards_layout = QtWidgets.QVBoxLayout(self._cards_widget)
        self._cards_layout.setContentsMargins(10, 10, 10, 10)
        self._cards_layout.setSpacing(8)

        self._cards_layout.addWidget(self._build_empty_state())
        self._cards_layout.addStretch()

        self._scroll.setWidget(self._cards_widget)
        container_layout.addWidget(self._scroll)

        layout.addWidget(self._container)

    @staticmethod
    def _make_btn(text, parent):
        btn = QtWidgets.QPushButton(text, parent)
        btn.setCursor(QtCore.Qt.PointingHandCursor)
        btn.setFixedSize(24, 24)
        btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: rgba(255, 255, 255, 120);
                border: none;
                border-radius: 4px;
                font-size: 12px;
                font-family: 'Segoe UI';
                font-weight: 600;
            }
            QPushButton:hover {
                background: rgba(255, 255, 255, 25);
                color: rgba(255, 255, 255, 220);
            }
        """)
        return btn

    # ------------------------------------------------------------------
    # Capture exclusion
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Translation updates
    # ------------------------------------------------------------------

    def update_translation(self, entries, info=""):
        """Thread-safe update for the panel contents."""
        self.translations_signal.emit(entries, info)

    def _on_translations_received(self, entries, info):
        """Render translated entries on the GUI thread."""
        self._entries = entries or []
        count = len(self._entries)
        self._info_label.setText(f"{count} msg" if count else "")
        self._rebuild_cards()
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

        compact = config.get("overlay_compact_mode", True)

        for index, entry in enumerate(self._entries, start=1):
            self._cards_layout.addWidget(
                self._build_entry_card(index, entry, compact=compact)
            )

        self._cards_layout.addStretch()
        self._scroll.verticalScrollBar().setValue(0)

    def _build_empty_state(self):
        """Create the placeholder shown when nothing is translated yet."""
        empty_state = QtWidgets.QLabel(
            "Waiting for foreign-language\nchat content…",
            self._cards_widget,
        )
        empty_state.setAlignment(QtCore.Qt.AlignCenter)
        empty_state.setWordWrap(True)
        empty_state.setStyleSheet("""
            QLabel {
                color: rgba(255, 255, 255, 60);
                background: transparent;
                border: 1px dashed rgba(255, 255, 255, 20);
                border-radius: 8px;
                padding: 20px 14px;
                font-size: 12px;
                font-family: 'Segoe UI';
            }
        """)
        return empty_state

    def _build_entry_card(self, index, entry, compact=True):
        """Create one translated message card.

        In compact mode (default), only the translated text is shown.
        In detailed mode, metadata and original text are also rendered.
        """
        frame = QtWidgets.QFrame(self._cards_widget)
        frame.setObjectName("entryCard")
        frame.setStyleSheet("""
            QFrame#entryCard {
                background: rgba(255, 255, 255, 8);
                border: 1px solid rgba(255, 255, 255, 10);
                border-radius: 8px;
            }
        """)
        layout = QtWidgets.QVBoxLayout(frame)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)

        if not compact:
            meta = QtWidgets.QLabel(
                f"{entry.get('source_lang', 'auto').upper()} → {entry.get('target_lang', 'EN').upper()}",
                frame,
            )
            meta.setStyleSheet("""
                QLabel {
                    color: rgba(0, 200, 150, 180);
                    font-size: 9px;
                    font-weight: 700;
                    font-family: 'Segoe UI';
                    letter-spacing: 0.5px;
                    background: transparent;
                    border: none;
                }
            """)
            layout.addWidget(meta)

        translated = QtWidgets.QLabel(entry.get("translated_text", ""), frame)
        translated.setWordWrap(True)
        translated.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        translated.setStyleSheet("""
            QLabel {
                color: rgba(255, 255, 255, 230);
                font-size: 13px;
                font-weight: 500;
                font-family: 'Segoe UI';
                background: transparent;
                border: none;
                line-height: 1.35;
            }
        """)
        layout.addWidget(translated)

        if not compact:
            original_text = entry.get("original_text", "").strip()
            if original_text:
                original = QtWidgets.QLabel(original_text, frame)
                original.setWordWrap(True)
                original.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
                original.setStyleSheet("""
                    QLabel {
                        color: rgba(255, 255, 255, 60);
                        font-size: 10px;
                        font-family: 'Segoe UI';
                        background: transparent;
                        border: none;
                    }
                """)
                layout.addWidget(original)

        return frame

    # ------------------------------------------------------------------
    # Panel visibility
    # ------------------------------------------------------------------

    def set_status(self, text):
        """Update the info label quietly."""
        QtCore.QMetaObject.invokeMethod(
            self._info_label,
            "setText",
            QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(str, text),
        )

    def show_panel(self):
        """Show or restore the panel without stealing focus."""
        if self.isMinimized():
            self.showNormal()
        else:
            self.show()

        # Only raise on first show — avoid stealing focus on every update
        if self._first_show:
            self.raise_()
            self._first_show = False

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
        width_ratio = float(config.get("overlay_width_ratio", 0.22))
        width_ratio = min(0.45, max(0.15, width_ratio))
        panel_width = max(280, int(available.width() * width_ratio))

        self.setGeometry(
            available.right() - panel_width + 1,
            available.y(),
            panel_width,
            available.height(),
        )

        if save_position:
            config.set("overlay_position", [self.x(), self.y()])

    # ------------------------------------------------------------------
    # Drag & resize
    # ------------------------------------------------------------------

    _EDGE_MARGIN = 6

    def _hit_test_edge(self, pos):
        """Return which edge/corner the position is near, or None."""
        x, y = pos.x(), pos.y()
        w, h = self.width(), self.height()
        m = self._EDGE_MARGIN

        left = x < m
        right = x > w - m
        top = y < m
        bottom = y > h - m

        if left and top:
            return "top-left"
        if right and top:
            return "top-right"
        if left and bottom:
            return "bottom-left"
        if right and bottom:
            return "bottom-right"
        if left:
            return "left"
        if right:
            return "right"
        if top:
            return "top"
        if bottom:
            return "bottom"
        return None

    def _edge_cursor(self, edge):
        cursors = {
            "left": QtCore.Qt.SizeHorCursor,
            "right": QtCore.Qt.SizeHorCursor,
            "top": QtCore.Qt.SizeVerCursor,
            "bottom": QtCore.Qt.SizeVerCursor,
            "top-left": QtCore.Qt.SizeFDiagCursor,
            "bottom-right": QtCore.Qt.SizeFDiagCursor,
            "top-right": QtCore.Qt.SizeBDiagCursor,
            "bottom-left": QtCore.Qt.SizeBDiagCursor,
        }
        return cursors.get(edge, QtCore.Qt.ArrowCursor)

    def mousePressEvent(self, event):
        if event.button() != QtCore.Qt.LeftButton:
            return

        edge = self._hit_test_edge(event.pos())
        if edge:
            self._resize_edge = edge
            self._drag_pos = event.globalPos()
            event.accept()
            return

        # Title bar drag (top 34px)
        if event.pos().y() < 38:
            self._drag_pos = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        # Resize in progress
        if self._resize_edge and self._drag_pos:
            delta = event.globalPos() - self._drag_pos
            self._drag_pos = event.globalPos()
            geo = self.geometry()

            if "right" in self._resize_edge:
                geo.setRight(geo.right() + delta.x())
            if "left" in self._resize_edge:
                geo.setLeft(geo.left() + delta.x())
            if "bottom" in self._resize_edge:
                geo.setBottom(geo.bottom() + delta.y())
            if "top" in self._resize_edge:
                geo.setTop(geo.top() + delta.y())

            if geo.width() >= self.minimumWidth() and geo.height() >= self.minimumHeight():
                self.setGeometry(geo)
            event.accept()
            return

        # Title-bar drag in progress
        if self._drag_pos and event.buttons() == QtCore.Qt.LeftButton and self._resize_edge is None:
            self.move(event.globalPos() - self._drag_pos)
            event.accept()
            return

        # Update cursor shape when hovering edges
        edge = self._hit_test_edge(event.pos())
        if edge:
            self.setCursor(self._edge_cursor(edge))
        else:
            self.unsetCursor()

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            if self._drag_pos and not self._resize_edge:
                config.set("overlay_position", [self.x(), self.y()])
            self._drag_pos = None
            self._resize_edge = None
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
