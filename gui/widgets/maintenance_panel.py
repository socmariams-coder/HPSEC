"""
HPSEC Suite - Maintenance Panel
================================

Panel per registrar i consultar events de manteniment.
Fora de la pipeline de processament de SEQ.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGroupBox,
    QGridLayout, QFrame, QTableWidget, QTableWidgetItem, QHeaderView,
    QDateEdit, QComboBox, QTextEdit, QMessageBox, QSplitter,
    QScrollArea, QSizePolicy
)
from PySide6.QtCore import Qt, QDate
from PySide6.QtGui import QFont, QColor

from pathlib import Path
from datetime import datetime
import json


class MaintenancePanel(QWidget):
    """Panel per gestionar registres de manteniment."""

    # Tipus d'events predefinits
    EVENT_TYPES = [
        ("column_change", "Canvi de columna", "#E74C3C"),
        ("column_clean", "Neteja columna", "#F39C12"),
        ("intensive_clean", "Neteja intensiva sistema", "#E67E22"),
        ("filter_change", "Canvi de filtres", "#3498DB"),
        ("reagent_change", "Canvi de reactius", "#9B59B6"),
        ("long_stop", "Aturada llarga (>24h)", "#95A5A6"),
        ("calibration_issue", "Incidència calibració", "#E74C3C"),
        ("detector_issue", "Incidència detector", "#C0392B"),
        ("other", "Altres", "#7F8C8D"),
    ]

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.events = []
        self._setup_ui()
        self._load_events()

    def _setup_ui(self):
        """Configura la interfície."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        # Títol
        title = QLabel("Registre de Manteniment")
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        layout.addWidget(title)

        # Info
        info = QLabel(
            "Registra events de manteniment per correlacionar amb canvis en les dades. "
            "Els events es guarden localment i són visibles des de qualsevol seqüència."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #666;")
        layout.addWidget(info)

        # Splitter: Formulari a dalt, Taula a baix
        splitter = QSplitter(Qt.Vertical)

        # === Formulari per afegir event ===
        form_group = QGroupBox("Nou Event")
        form_layout = QGridLayout(form_group)
        form_layout.setSpacing(12)

        # Data
        form_layout.addWidget(QLabel("Data:"), 0, 0)
        self.date_edit = QDateEdit()
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setDate(QDate.currentDate())
        self.date_edit.setDisplayFormat("dd/MM/yyyy")
        form_layout.addWidget(self.date_edit, 0, 1)

        # Tipus
        form_layout.addWidget(QLabel("Tipus:"), 0, 2)
        self.type_combo = QComboBox()
        for type_id, type_name, color in self.EVENT_TYPES:
            self.type_combo.addItem(type_name, type_id)
        self.type_combo.setMinimumWidth(200)
        form_layout.addWidget(self.type_combo, 0, 3)

        # Descripció
        form_layout.addWidget(QLabel("Descripció:"), 1, 0, Qt.AlignTop)
        self.desc_edit = QTextEdit()
        self.desc_edit.setMaximumHeight(80)
        self.desc_edit.setPlaceholderText(
            "Detalls de l'event (opcional):\n"
            "- Columna: marca, lot, nº sèrie\n"
            "- Reactius: lot, data caducitat\n"
            "- Motiu de l'aturada/neteja"
        )
        form_layout.addWidget(self.desc_edit, 1, 1, 1, 3)

        # Botons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.add_btn = QPushButton("Afegir Event")
        self.add_btn.clicked.connect(self._add_event)
        self.add_btn.setStyleSheet("QPushButton { padding: 8px 16px; }")
        btn_layout.addWidget(self.add_btn)

        form_layout.addLayout(btn_layout, 2, 0, 1, 4)

        splitter.addWidget(form_group)

        # === Taula d'events ===
        history_group = QGroupBox("Històric d'Events")
        history_layout = QVBoxLayout(history_group)

        # Filtres
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Filtrar per tipus:"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItem("Tots", None)
        for type_id, type_name, color in self.EVENT_TYPES:
            self.filter_combo.addItem(type_name, type_id)
        self.filter_combo.currentIndexChanged.connect(self._filter_events)
        filter_layout.addWidget(self.filter_combo)
        filter_layout.addStretch()

        self.delete_btn = QPushButton("Eliminar Seleccionat")
        self.delete_btn.setEnabled(False)
        self.delete_btn.clicked.connect(self._delete_event)
        filter_layout.addWidget(self.delete_btn)

        history_layout.addLayout(filter_layout)

        # Taula
        self.events_table = QTableWidget()
        self.events_table.setColumnCount(4)
        self.events_table.setHorizontalHeaderLabels(["Data", "Tipus", "Descripció", "ID"])
        self.events_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.events_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.events_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.events_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.events_table.setColumnHidden(3, True)  # ID ocult
        self.events_table.setAlternatingRowColors(True)
        self.events_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.events_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.events_table.itemSelectionChanged.connect(self._on_selection_changed)
        history_layout.addWidget(self.events_table)

        # Resum
        self.summary_label = QLabel()
        self.summary_label.setStyleSheet("color: #666; font-size: 11px;")
        history_layout.addWidget(self.summary_label)

        splitter.addWidget(history_group)
        splitter.setSizes([200, 400])

        layout.addWidget(splitter)

    def _get_events_file(self):
        """Retorna el path del fitxer d'events."""
        # Guardar a la carpeta d'usuari/HPSEC
        user_dir = Path.home() / "HPSEC_Data"
        user_dir.mkdir(exist_ok=True)
        return user_dir / "maintenance_events.json"

    def _load_events(self):
        """Carrega els events del fitxer."""
        events_file = self._get_events_file()
        if events_file.exists():
            try:
                with open(events_file, 'r', encoding='utf-8') as f:
                    self.events = json.load(f)
            except Exception as e:
                print(f"[WARNING] Error carregant events: {e}")
                self.events = []
        else:
            self.events = []

        self._refresh_table()

    def _save_events(self):
        """Guarda els events al fitxer."""
        events_file = self._get_events_file()
        try:
            with open(events_file, 'w', encoding='utf-8') as f:
                json.dump(self.events, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[WARNING] Error guardant events: {e}")

    def _add_event(self):
        """Afegeix un nou event."""
        date = self.date_edit.date().toString("yyyy-MM-dd")
        type_id = self.type_combo.currentData()
        type_name = self.type_combo.currentText()
        description = self.desc_edit.toPlainText().strip()

        event = {
            "id": datetime.now().strftime("%Y%m%d%H%M%S%f"),
            "date": date,
            "type_id": type_id,
            "type_name": type_name,
            "description": description,
            "created_at": datetime.now().isoformat(),
        }

        self.events.append(event)
        self.events.sort(key=lambda x: x['date'], reverse=True)
        self._save_events()
        self._refresh_table()

        # Netejar formulari
        self.desc_edit.clear()
        self.main_window.set_status(f"Event afegit: {type_name} ({date})", 3000)

    def _delete_event(self):
        """Elimina l'event seleccionat."""
        row = self.events_table.currentRow()
        if row < 0:
            return

        event_id = self.events_table.item(row, 3).text()
        event_type = self.events_table.item(row, 1).text()
        event_date = self.events_table.item(row, 0).text()

        reply = QMessageBox.question(
            self, "Confirmar Eliminació",
            f"Vols eliminar l'event?\n\n{event_date}: {event_type}",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.events = [e for e in self.events if e['id'] != event_id]
            self._save_events()
            self._refresh_table()
            self.main_window.set_status("Event eliminat", 3000)

    def _filter_events(self):
        """Filtra els events per tipus."""
        self._refresh_table()

    def _refresh_table(self):
        """Actualitza la taula d'events."""
        self.events_table.setRowCount(0)

        filter_type = self.filter_combo.currentData()

        # Crear diccionari de colors
        type_colors = {t[0]: t[2] for t in self.EVENT_TYPES}

        filtered = self.events
        if filter_type:
            filtered = [e for e in self.events if e.get('type_id') == filter_type]

        for event in filtered:
            row = self.events_table.rowCount()
            self.events_table.insertRow(row)

            # Data (format dd/MM/yyyy)
            date_str = event.get('date', '')
            try:
                date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                date_display = date_obj.strftime("%d/%m/%Y")
            except:
                date_display = date_str
            self.events_table.setItem(row, 0, QTableWidgetItem(date_display))

            # Tipus amb color
            type_id = event.get('type_id', 'other')
            type_name = event.get('type_name', type_id)
            type_item = QTableWidgetItem(type_name)
            color = type_colors.get(type_id, '#7F8C8D')
            type_item.setForeground(QColor(color))
            type_item.setFont(QFont("Segoe UI", 9, QFont.Bold))
            self.events_table.setItem(row, 1, type_item)

            # Descripció
            desc = event.get('description', '')
            desc_short = desc[:100] + "..." if len(desc) > 100 else desc
            desc_item = QTableWidgetItem(desc_short)
            desc_item.setToolTip(desc)
            self.events_table.setItem(row, 2, desc_item)

            # ID (ocult)
            self.events_table.setItem(row, 3, QTableWidgetItem(event.get('id', '')))

        # Resum
        n_total = len(self.events)
        n_filtered = len(filtered)
        if filter_type:
            self.summary_label.setText(f"Mostrant {n_filtered} de {n_total} events")
        else:
            self.summary_label.setText(f"Total: {n_total} events")

    def _on_selection_changed(self):
        """Handler quan canvia la selecció."""
        self.delete_btn.setEnabled(self.events_table.currentRow() >= 0)

    def get_events_in_range(self, start_date, end_date):
        """
        Retorna events dins d'un rang de dates.

        Útil per correlacionar amb dades de seqüències.

        Args:
            start_date: Data inicial (string YYYY-MM-DD o datetime)
            end_date: Data final (string YYYY-MM-DD o datetime)

        Returns:
            Llista d'events dins del rang
        """
        if isinstance(start_date, datetime):
            start_date = start_date.strftime("%Y-%m-%d")
        if isinstance(end_date, datetime):
            end_date = end_date.strftime("%Y-%m-%d")

        return [
            e for e in self.events
            if start_date <= e.get('date', '') <= end_date
        ]

    def get_recent_events(self, days=30):
        """Retorna events dels últims N dies."""
        from datetime import timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        return self.get_events_in_range(start_date, end_date)
