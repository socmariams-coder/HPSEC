"""
HPSEC Suite - Maintenance Panel
================================

Panel per visualitzar events de manteniment des de l'Excel centralitzat.
Llegeix directament del fitxer configurat a hpsec_config.json.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGroupBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QComboBox, QMessageBox,
    QFileDialog, QSizePolicy
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QColor

from pathlib import Path
from datetime import datetime
import os

# Intentar importar pandas
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


class MaintenancePanel(QWidget):
    """Panel per visualitzar registres de manteniment des d'Excel."""

    # Mapeig de tasques a categories i colors
    TASK_CATEGORIES = {
        "neteja amb azida": ("Neteja azida sodica", "#F39C12"),
        "neteja columna": ("Neteja columna", "#3498DB"),
        "canvi cartutx": ("Canvi cartutx", "#9B59B6"),
        "cartutx oxidant": ("Canvi cartutx oxidant", "#9B59B6"),
        "cartutx d'acid": ("Canvi cartutx acid", "#9B59B6"),
        "visita tecnic": ("Visita tecnic", "#E74C3C"),
        "canvi columna": ("Canvi columna", "#E74C3C"),
        "canvi lampada": ("Canvi lampada", "#E67E22"),
        "canvi filtres": ("Canvi filtres", "#27AE60"),
    }

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.events = []
        self.excel_path = None
        self._setup_ui()

        # Carregar dades despres de mostrar UI
        QTimer.singleShot(100, self._load_from_config)

    def _setup_ui(self):
        """Configura la interficie."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)

        # === Capçalera compacta: Titol + Info + Botons ===
        header_layout = QHBoxLayout()
        header_layout.setSpacing(16)

        # Titol i info
        title_layout = QVBoxLayout()
        title_layout.setSpacing(2)
        title = QLabel("Manteniment de l'Equip")
        title.setFont(QFont("Segoe UI", 14, QFont.Bold))
        title_layout.addWidget(title)

        self.info_label = QLabel("Carregant...")
        self.info_label.setStyleSheet("color: #666; font-size: 11px;")
        title_layout.addWidget(self.info_label)

        header_layout.addLayout(title_layout, 1)

        # Botons compactes
        self.reload_btn = QPushButton("Recarregar")
        self.reload_btn.setFixedWidth(80)
        self.reload_btn.clicked.connect(self._reload_data)
        header_layout.addWidget(self.reload_btn)

        self.change_btn = QPushButton("Canviar...")
        self.change_btn.setFixedWidth(80)
        self.change_btn.clicked.connect(self._change_file)
        header_layout.addWidget(self.change_btn)

        layout.addLayout(header_layout)

        # Path (una linia)
        self.path_label = QLabel("No configurat")
        self.path_label.setStyleSheet("color: #888; font-family: monospace; font-size: 10px;")
        layout.addWidget(self.path_label)

        # === Resum compacte en una linia ===
        summary_layout = QHBoxLayout()
        summary_layout.setSpacing(24)

        self.summary_labels = {}
        for cat_name, color in [("Netejes", "#F39C12"), ("Cartutxos", "#9B59B6"),
                                 ("Tecnics", "#E74C3C"), ("Total", "#2E86AB")]:
            item_layout = QHBoxLayout()
            item_layout.setSpacing(4)

            count_label = QLabel("0")
            count_label.setFont(QFont("Segoe UI", 12, QFont.Bold))
            count_label.setStyleSheet(f"color: {color};")

            name_label = QLabel(cat_name)
            name_label.setStyleSheet("color: #666; font-size: 11px;")

            item_layout.addWidget(count_label)
            item_layout.addWidget(name_label)
            summary_layout.addLayout(item_layout)
            self.summary_labels[cat_name] = count_label

        summary_layout.addStretch()
        layout.addLayout(summary_layout)

        # === Filtres en linia ===
        filter_layout = QHBoxLayout()
        filter_layout.setSpacing(8)

        filter_layout.addWidget(QLabel("Any:"))
        self.year_combo = QComboBox()
        self.year_combo.addItem("Tots", None)
        self.year_combo.setFixedWidth(70)
        self.year_combo.currentIndexChanged.connect(self._filter_events)
        filter_layout.addWidget(self.year_combo)

        filter_layout.addWidget(QLabel("Tipus:"))
        self.type_combo = QComboBox()
        self.type_combo.addItem("Tots", None)
        self.type_combo.setMinimumWidth(140)
        self.type_combo.currentIndexChanged.connect(self._filter_events)
        filter_layout.addWidget(self.type_combo)

        filter_layout.addStretch()

        self.count_label = QLabel("")
        self.count_label.setStyleSheet("color: #666; font-size: 11px;")
        filter_layout.addWidget(self.count_label)

        layout.addLayout(filter_layout)

        # === Taula (ocupa tot l'espai) ===
        self.events_table = QTableWidget()
        self.events_table.setColumnCount(5)
        self.events_table.setHorizontalHeaderLabels(["Data", "Tipus", "Hores", "Usuari", "Detalls"])
        self.events_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
        self.events_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Fixed)
        self.events_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Fixed)
        self.events_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Fixed)
        self.events_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.events_table.setColumnWidth(0, 85)   # Data
        self.events_table.setColumnWidth(1, 140)  # Tipus
        self.events_table.setColumnWidth(2, 50)   # Hores
        self.events_table.setColumnWidth(3, 100)  # Usuari
        self.events_table.setAlternatingRowColors(True)
        self.events_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.events_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.events_table.verticalHeader().setDefaultSectionSize(24)  # Files mes compactes
        layout.addWidget(self.events_table, 1)  # stretch=1 per ocupar tot

    def _load_from_config(self):
        """Carrega el path des de la configuracio."""
        try:
            from hpsec_config import get_config
            config = get_config()
            self.excel_path = config.get("paths", {}).get("maintenance_excel", "")

            if self.excel_path and os.path.exists(self.excel_path):
                self.path_label.setText(self.excel_path)
                self._load_excel()
            else:
                self.info_label.setText(
                    "Fitxer Excel de manteniment no configurat o no trobat.\n"
                    "Configura el path a hpsec_config.json > paths > maintenance_excel"
                )
                self.path_label.setText("No trobat: " + (self.excel_path or "(buit)"))
                self.path_label.setStyleSheet("color: #E74C3C; font-family: monospace;")
        except Exception as e:
            self.info_label.setText(f"Error carregant configuracio: {e}")

    def _change_file(self):
        """Permet canviar el fitxer Excel."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Selecciona fitxer Excel de manteniment",
            "",
            "Excel Files (*.xlsx *.xls);;All Files (*)"
        )

        if file_path:
            self.excel_path = file_path
            self.path_label.setText(file_path)
            self.path_label.setStyleSheet("color: #666; font-family: monospace;")
            self._load_excel()

            # Suggerir guardar a config
            QMessageBox.information(
                self, "Nota",
                f"Per fer permanent aquest canvi, actualitza hpsec_config.json:\n\n"
                f'"maintenance_excel": "{file_path.replace(chr(92), "/")}"'
            )

    def _reload_data(self):
        """Recarrega les dades."""
        if self.excel_path and os.path.exists(self.excel_path):
            self._load_excel()
            self.main_window.set_status("Dades recarregades", 2000)
        else:
            QMessageBox.warning(self, "Error", "Fitxer no trobat")

    def _load_excel(self):
        """Carrega les dades de l'Excel."""
        if not HAS_PANDAS:
            self.info_label.setText("Error: Cal instal-lar pandas (pip install pandas openpyxl)")
            return

        try:
            df = pd.read_excel(self.excel_path, engine='openpyxl')

            # Processar dades
            self.events = []

            for _, row in df.iterrows():
                # Data d'execucio
                date_val = row.get('Data Execució')
                if pd.isna(date_val):
                    continue

                if hasattr(date_val, 'strftime'):
                    date_str = date_val.strftime("%Y-%m-%d")
                else:
                    date_str = str(date_val)[:10]

                # Tasca
                tasca = str(row.get('tasca', '')).strip()
                if pd.isna(tasca) or not tasca or tasca == 'nan':
                    continue

                # Categoritzar
                category, color = self._categorize_task(tasca)

                # Altres camps
                hores = row.get('Unitats', 0)
                if pd.isna(hores):
                    hores = 0

                usuari = str(row.get('Usuari registre', ''))
                if pd.isna(usuari) or usuari == 'nan':
                    usuari = ''
                # Simplificar nom usuari (primer nom + primer cognom)
                if ',' in usuari:
                    parts = usuari.split(',')
                    usuari = f"{parts[1].strip().split()[0]} {parts[0].strip()}"

                event = {
                    "date": date_str,
                    "tasca": tasca,
                    "category": category,
                    "color": color,
                    "hores": float(hores) if hores else 0,
                    "usuari": usuari,
                }
                self.events.append(event)

            # Ordenar per data desc
            self.events.sort(key=lambda x: x['date'], reverse=True)

            # Actualitzar UI
            self._update_filters()
            self._update_summary()
            self._refresh_table()

            self.info_label.setText(
                f"Carregats {len(self.events)} events de manteniment. "
                f"Ultim: {self.events[0]['date'] if self.events else 'N/A'}"
            )

        except Exception as e:
            self.info_label.setText(f"Error llegint Excel: {e}")
            import traceback
            traceback.print_exc()

    def _categorize_task(self, tasca):
        """Categoritza una tasca segons el text."""
        tasca_lower = tasca.lower()

        for pattern, (category, color) in self.TASK_CATEGORIES.items():
            if pattern in tasca_lower:
                return category, color

        return tasca, "#7F8C8D"  # Default gris

    def _update_filters(self):
        """Actualitza els filtres amb les dades disponibles."""
        # Anys
        current_year = self.year_combo.currentData()
        self.year_combo.blockSignals(True)
        self.year_combo.clear()
        self.year_combo.addItem("Tots", None)

        years = set()
        for e in self.events:
            try:
                year = e['date'][:4]
                years.add(year)
            except:
                pass

        for year in sorted(years, reverse=True):
            self.year_combo.addItem(year, year)

        if current_year:
            idx = self.year_combo.findData(current_year)
            if idx >= 0:
                self.year_combo.setCurrentIndex(idx)

        self.year_combo.blockSignals(False)

        # Tipus
        current_type = self.type_combo.currentData()
        self.type_combo.blockSignals(True)
        self.type_combo.clear()
        self.type_combo.addItem("Tots", None)

        categories = set(e['category'] for e in self.events)
        for cat in sorted(categories):
            self.type_combo.addItem(cat, cat)

        if current_type:
            idx = self.type_combo.findData(current_type)
            if idx >= 0:
                self.type_combo.setCurrentIndex(idx)

        self.type_combo.blockSignals(False)

    def _update_summary(self):
        """Actualitza el resum."""
        netejes = sum(1 for e in self.events if 'neteja' in e['category'].lower())
        cartutxos = sum(1 for e in self.events if 'cartutx' in e['category'].lower())
        visites = sum(1 for e in self.events if 'tecnic' in e['category'].lower())
        total = len(self.events)

        self.summary_labels["Netejes"].setText(str(netejes))
        self.summary_labels["Cartutxos"].setText(str(cartutxos))
        self.summary_labels["Tecnics"].setText(str(visites))
        self.summary_labels["Total"].setText(str(total))

    def _filter_events(self):
        """Aplica filtres."""
        self._refresh_table()

    def _refresh_table(self):
        """Actualitza la taula."""
        self.events_table.setRowCount(0)

        filter_year = self.year_combo.currentData()
        filter_type = self.type_combo.currentData()

        filtered = self.events

        if filter_year:
            filtered = [e for e in filtered if e['date'].startswith(filter_year)]

        if filter_type:
            filtered = [e for e in filtered if e['category'] == filter_type]

        for event in filtered:
            row = self.events_table.rowCount()
            self.events_table.insertRow(row)

            # Data (dd/mm/yyyy)
            try:
                date_obj = datetime.strptime(event['date'], "%Y-%m-%d")
                date_display = date_obj.strftime("%d/%m/%Y")
            except:
                date_display = event['date']
            self.events_table.setItem(row, 0, QTableWidgetItem(date_display))

            # Categoria amb color
            cat_item = QTableWidgetItem(event['category'])
            cat_item.setForeground(QColor(event['color']))
            cat_item.setFont(QFont("Segoe UI", 9, QFont.Bold))
            self.events_table.setItem(row, 1, cat_item)

            # Hores
            hores = event.get('hores', 0)
            hores_str = f"{hores:.1f}h" if hores else "-"
            self.events_table.setItem(row, 2, QTableWidgetItem(hores_str))

            # Usuari
            self.events_table.setItem(row, 3, QTableWidgetItem(event.get('usuari', '')))

            # Tasca original (detalls)
            tasca_item = QTableWidgetItem(event['tasca'])
            tasca_item.setToolTip(event['tasca'])
            self.events_table.setItem(row, 4, tasca_item)

        self.count_label.setText(f"Mostrant {len(filtered)} de {len(self.events)}")

    def get_events_in_range(self, start_date, end_date):
        """
        Retorna events dins d'un rang de dates.
        Util per correlacionar amb dades de sequencies.
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
        """Retorna events dels ultims N dies."""
        from datetime import timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        return self.get_events_in_range(start_date, end_date)
