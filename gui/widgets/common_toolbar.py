"""
Common toolbar widget per a tots els panels de processament.
Estructura unificada: [⚠ Avisos] [📝 Notes] [< Anterior] [Següent >]

G01-G06: Principis generals GUI
- Mateixa estructura sempre (nou o JSON)
- Avisos centralitzats en un botó
- Traçabilitat (qui confirma)
- Botó notes sempre visible
"""

from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QPushButton, QDialog, QVBoxLayout,
    QLabel, QTextEdit, QDialogButtonBox, QListWidget, QListWidgetItem,
    QLineEdit, QMessageBox
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QColor
from datetime import datetime
import getpass


class WarningsDialog(QDialog):
    """Diàleg per mostrar i confirmar avisos."""

    def __init__(self, parent, warnings, confirmed_by=None):
        super().__init__(parent)
        self.setWindowTitle("Avisos")
        self.setMinimumWidth(500)
        self.setMinimumHeight(300)

        self.confirmed_by = confirmed_by
        self.result_confirmed_by = None

        layout = QVBoxLayout(self)

        # Llista d'avisos
        self.list_widget = QListWidget()
        for w in warnings:
            item = QListWidgetItem(w)
            # Color segons severitat
            if "CRÍTIC" in w.upper() or "ERROR" in w.upper():
                item.setBackground(QColor("#FADBD8"))  # Rosa
            elif "AVÍS" in w.upper() or "WARNING" in w.upper():
                item.setBackground(QColor("#FCF3CF"))  # Groc
            else:
                item.setBackground(QColor("#D5F5E3"))  # Verd
            self.list_widget.addItem(item)
        layout.addWidget(self.list_widget)

        # Confirmat per
        confirm_layout = QHBoxLayout()
        confirm_layout.addWidget(QLabel("Revisat per:"))
        self.initials_edit = QLineEdit()
        self.initials_edit.setPlaceholderText("Inicials (ex: JM)")
        self.initials_edit.setMaximumWidth(100)
        if confirmed_by:
            self.initials_edit.setText(confirmed_by)
            self.initials_edit.setEnabled(False)
        confirm_layout.addWidget(self.initials_edit)
        confirm_layout.addStretch()
        layout.addLayout(confirm_layout)

        # Botons
        buttons = QDialogButtonBox()
        if not confirmed_by:
            self.ok_btn = QPushButton("✓ Confirmar revisat")
            self.ok_btn.clicked.connect(self._confirm)
            buttons.addButton(self.ok_btn, QDialogButtonBox.AcceptRole)

        close_btn = QPushButton("Tancar")
        close_btn.clicked.connect(self.reject)
        buttons.addButton(close_btn, QDialogButtonBox.RejectRole)
        layout.addWidget(buttons)

    def _confirm(self):
        initials = self.initials_edit.text().strip()
        if not initials:
            QMessageBox.warning(self, "Inicials requerides",
                              "Cal introduir les inicials de qui revisa.")
            return
        self.result_confirmed_by = initials
        self.accept()


class NotesDialog(QDialog):
    """Diàleg per afegir/veure notes."""

    def __init__(self, parent, existing_notes=None):
        super().__init__(parent)
        self.setWindowTitle("Notes")
        self.setMinimumWidth(400)
        self.setMinimumHeight(300)

        layout = QVBoxLayout(self)

        # Text area
        self.text_edit = QTextEdit()
        if existing_notes:
            self.text_edit.setPlainText(existing_notes)
        self.text_edit.setPlaceholderText("Escriu les teves notes aquí...")
        layout.addWidget(self.text_edit)

        # Info
        user = getpass.getuser()
        date = datetime.now().strftime("%Y-%m-%d %H:%M")
        info_label = QLabel(f"Usuari: {user} | Data: {date}")
        info_label.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(info_label)

        # Botons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_notes(self):
        return self.text_edit.toPlainText()


class CommonToolbar(QWidget):
    """
    Barra inferior comuna per a tots els panels.

    Signals:
        warnings_confirmed: Emès quan es confirmen avisos (amb inicials)
        notes_changed: Emès quan canvien les notes
        previous_clicked: Emès quan es prem Anterior
        next_clicked: Emès quan es prem Següent
    """

    warnings_confirmed = Signal(str)  # inicials
    notes_changed = Signal(str)  # text notes
    previous_clicked = Signal()
    next_clicked = Signal()

    def __init__(self, parent=None, show_previous=True, show_next=True,
                 next_text="Següent →", previous_text="← Anterior"):
        super().__init__(parent)

        self._warnings = []
        self._warnings_confirmed_by = None
        self._notes = ""

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 5, 0, 0)

        # Botó avisos
        self.warnings_btn = QPushButton("⚠ 0 Avisos")
        self.warnings_btn.setToolTip("Veure i confirmar avisos")
        self.warnings_btn.clicked.connect(self._show_warnings)
        self.warnings_btn.setVisible(False)  # Invisible si no hi ha avisos
        layout.addWidget(self.warnings_btn)

        # Botó notes
        self.notes_btn = QPushButton("📝 Notes")
        self.notes_btn.setToolTip("Afegir notes")
        self.notes_btn.clicked.connect(self._show_notes)
        layout.addWidget(self.notes_btn)

        layout.addStretch()

        # Navegació
        self.prev_btn = QPushButton(previous_text)
        self.prev_btn.clicked.connect(self.previous_clicked.emit)
        self.prev_btn.setVisible(show_previous)
        layout.addWidget(self.prev_btn)

        self.next_btn = QPushButton(next_text)
        self.next_btn.clicked.connect(self.next_clicked.emit)
        self.next_btn.setVisible(show_next)
        layout.addWidget(self.next_btn)

        self._update_warnings_button()

    def set_warnings(self, warnings: list, confirmed_by: str = None):
        """Estableix els avisos i l'estat de confirmació."""
        self._warnings = warnings or []
        self._warnings_confirmed_by = confirmed_by
        self._update_warnings_button()

    def add_warning(self, warning: str):
        """Afegeix un avís."""
        if warning not in self._warnings:
            self._warnings.append(warning)
            self._update_warnings_button()

    def clear_warnings(self):
        """Esborra tots els avisos."""
        self._warnings = []
        self._warnings_confirmed_by = None
        self._update_warnings_button()

    def get_warnings(self):
        """Retorna la llista d'avisos."""
        return self._warnings

    def get_confirmed_by(self):
        """Retorna qui ha confirmat els avisos."""
        return self._warnings_confirmed_by

    def is_confirmed(self):
        """Retorna si els avisos estan confirmats."""
        return self._warnings_confirmed_by is not None

    def set_notes(self, notes: str):
        """Estableix les notes."""
        self._notes = notes or ""
        self._update_notes_button()

    def get_notes(self):
        """Retorna les notes."""
        return self._notes

    def set_next_enabled(self, enabled: bool):
        """Habilita/deshabilita el botó Següent."""
        self.next_btn.setEnabled(enabled)

    def set_previous_enabled(self, enabled: bool):
        """Habilita/deshabilita el botó Anterior."""
        self.prev_btn.setEnabled(enabled)

    def set_next_text(self, text: str):
        """Canvia el text del botó Següent."""
        self.next_btn.setText(text)

    def _update_warnings_button(self):
        """Actualitza l'aparença del botó d'avisos."""
        n = len(self._warnings)

        if n == 0:
            self.warnings_btn.setVisible(False)
            return

        self.warnings_btn.setVisible(True)

        if self._warnings_confirmed_by:
            self.warnings_btn.setText(f"✓ {n} Avisos ({self._warnings_confirmed_by})")
            self.warnings_btn.setStyleSheet(
                "background-color: #D5F5E3; color: #1E8449; font-weight: bold;"
            )
        else:
            self.warnings_btn.setText(f"⚠ {n} Avisos")
            self.warnings_btn.setStyleSheet(
                "background-color: #FCF3CF; color: #9A7B0A; font-weight: bold;"
            )

    def _update_notes_button(self):
        """Actualitza l'aparença del botó de notes."""
        if self._notes:
            self.notes_btn.setText("📝 Notes ●")
            self.notes_btn.setStyleSheet("color: #2874A6;")
        else:
            self.notes_btn.setText("📝 Notes")
            self.notes_btn.setStyleSheet("")

    def _show_warnings(self):
        """Mostra el diàleg d'avisos."""
        dialog = WarningsDialog(self, self._warnings, self._warnings_confirmed_by)
        if dialog.exec_() == QDialog.Accepted and dialog.result_confirmed_by:
            self._warnings_confirmed_by = dialog.result_confirmed_by
            self._update_warnings_button()
            self.warnings_confirmed.emit(self._warnings_confirmed_by)

    def _show_notes(self):
        """Mostra el diàleg de notes."""
        dialog = NotesDialog(self, self._notes)
        if dialog.exec_() == QDialog.Accepted:
            self._notes = dialog.get_notes()
            self._update_notes_button()
            self.notes_changed.emit(self._notes)
