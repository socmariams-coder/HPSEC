"""
HPSEC Suite - Maintenance Panel
================================

Panel per visualitzar:
1. Events de manteniment des de l'Excel centralitzat del tècnic
2. Canvis metodològics (protocol, volum, sensibilitat...) que el tècnic no registra
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGroupBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QComboBox, QMessageBox,
    QFileDialog, QSizePolicy, QDialog, QFormLayout, QLineEdit, QTextEdit,
    QDateEdit, QSplitter, QCheckBox,
)
from PySide6.QtCore import Qt, QTimer, QDate
from PySide6.QtGui import QFont, QColor

from pathlib import Path
from datetime import datetime
import json as _json
import os

# Path per defecte del registre metodològic
METHOD_LOG_PATH = Path(os.environ.get(
    "HPSEC_REGISTRY", Path(__file__).resolve().parent.parent.parent / "REGISTRY"
)) / "method_log.json"

METHOD_CATEGORIES = [
    "Canvi columna",
    "Canvi detector/guany",
    "Canvi protocol preparació",
    "Canvi volum injecció",
    "Canvi sensibilitat UIB",
    "Canvi mètode processament",
    "Canvi reactiu/consumible",
    "Observació",
]

METHOD_COLORS = {
    "Canvi columna": "#C0392B",
    "Canvi detector/guany": "#B03A8E",
    "Canvi protocol preparació": "#8E44AD",
    "Canvi volum injecció": "#2980B9",
    "Canvi sensibilitat UIB": "#16A085",
    "Canvi mètode processament": "#D35400",
    "Canvi reactiu/consumible": "#EF4444",
    "Observació": "#7F8C8D",
}

# Categories que fan SOSPITAR un canvi de règim instrumental. No l'obren:
# registren un candidat que el següent KHP confirmarà o descartarà
# (un canvi de columna amb resposta idèntica NO parteix el bloc — cas 305).
REGIME_SUSPECT_CATEGORIES = {
    "Canvi columna",
    "Canvi detector/guany",
    "Canvi sensibilitat UIB",
}

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
        self.method_changes = []
        self.excel_path = None
        self._setup_ui()

        # Carregar dades despres de mostrar UI
        QTimer.singleShot(100, self._load_all)

    def _setup_ui(self):
        """Configura la interfície amb dues seccions: equip + canvis metodològics."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)

        splitter = QSplitter(Qt.Vertical)

        # =================================================================
        # SECCIÓ 1: MANTENIMENT EQUIP (des d'Excel tècnic)
        # =================================================================
        equip_widget = QWidget()
        equip_layout = QVBoxLayout(equip_widget)
        equip_layout.setContentsMargins(0, 0, 0, 0)
        equip_layout.setSpacing(6)

        # Capçalera
        header_layout = QHBoxLayout()
        header_layout.setSpacing(16)

        title_layout = QVBoxLayout()
        title_layout.setSpacing(2)
        title = QLabel("Manteniment de l'Equip")
        title.setFont(QFont("Segoe UI", 13, QFont.Bold))
        title_layout.addWidget(title)

        self.info_label = QLabel("Carregant...")
        self.info_label.setStyleSheet("color: #666; font-size: 11px;")
        title_layout.addWidget(self.info_label)

        header_layout.addLayout(title_layout, 1)

        self.reload_btn = QPushButton("Recarregar")
        self.reload_btn.setFixedWidth(80)
        self.reload_btn.clicked.connect(self._reload_data)
        header_layout.addWidget(self.reload_btn)

        self.change_btn = QPushButton("Canviar...")
        self.change_btn.setFixedWidth(80)
        self.change_btn.clicked.connect(self._change_file)
        header_layout.addWidget(self.change_btn)

        equip_layout.addLayout(header_layout)

        # Path
        self.path_label = QLabel("No configurat")
        self.path_label.setStyleSheet("color: #888; font-family: monospace; font-size: 10px;")
        equip_layout.addWidget(self.path_label)

        # Resum compacte
        summary_layout = QHBoxLayout()
        summary_layout.setSpacing(24)

        self.summary_labels = {}
        for cat_name, color in [("Netejes", "#F39C12"), ("Cartutxos", "#9B59B6"),
                                 ("Tecnics", "#E74C3C"), ("Total", "#2563EB")]:
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
        equip_layout.addLayout(summary_layout)

        # Filtres
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

        equip_layout.addLayout(filter_layout)

        # Taula manteniment
        self.events_table = QTableWidget()
        self.events_table.setColumnCount(5)
        self.events_table.setHorizontalHeaderLabels(["Data", "Tipus", "Hores", "Usuari", "Detalls"])
        self.events_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
        self.events_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Fixed)
        self.events_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Fixed)
        self.events_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Fixed)
        self.events_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.events_table.setColumnWidth(0, 85)
        self.events_table.setColumnWidth(1, 140)
        self.events_table.setColumnWidth(2, 50)
        self.events_table.setColumnWidth(3, 100)
        self.events_table.setAlternatingRowColors(True)
        self.events_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.events_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.events_table.verticalHeader().setDefaultSectionSize(24)
        equip_layout.addWidget(self.events_table, 1)

        splitter.addWidget(equip_widget)

        # =================================================================
        # SECCIÓ 2: CANVIS METODOLÒGICS (JSON intern)
        # =================================================================
        method_widget = QWidget()
        method_layout = QVBoxLayout(method_widget)
        method_layout.setContentsMargins(0, 8, 0, 0)
        method_layout.setSpacing(6)

        # Capçalera
        method_header = QHBoxLayout()
        method_title = QLabel("Canvis Metodològics")
        method_title.setFont(QFont("Segoe UI", 13, QFont.Bold))
        method_header.addWidget(method_title)

        self.method_info = QLabel("")
        self.method_info.setStyleSheet("color: #666; font-size: 11px;")
        method_header.addWidget(self.method_info, 1)

        self.method_add_btn = QPushButton("+ Afegir")
        self.method_add_btn.setFixedWidth(80)
        self.method_add_btn.setStyleSheet(
            "font-weight: bold; color: #2980B9; border: 1px solid #2980B9;"
            "border-radius: 4px; padding: 3px 8px;"
        )
        self.method_add_btn.clicked.connect(self._add_method_change)
        method_header.addWidget(self.method_add_btn)

        self.method_del_btn = QPushButton("Eliminar")
        self.method_del_btn.setFixedWidth(70)
        self.method_del_btn.setEnabled(False)
        self.method_del_btn.clicked.connect(self._delete_method_change)
        method_header.addWidget(self.method_del_btn)

        method_layout.addLayout(method_header)

        # Taula canvis metodològics
        self.method_table = QTableWidget()
        self.method_table.setColumnCount(4)
        self.method_table.setHorizontalHeaderLabels(["Data", "Categoria", "Descripció", "SEQ ref."])
        self.method_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
        self.method_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Fixed)
        self.method_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.method_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Fixed)
        self.method_table.setColumnWidth(0, 85)
        self.method_table.setColumnWidth(1, 180)
        self.method_table.setColumnWidth(3, 100)
        self.method_table.setAlternatingRowColors(True)
        self.method_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.method_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.method_table.verticalHeader().setDefaultSectionSize(24)
        self.method_table.itemSelectionChanged.connect(
            lambda: self.method_del_btn.setEnabled(
                len(self.method_table.selectedItems()) > 0
            )
        )
        method_layout.addWidget(self.method_table, 1)

        splitter.addWidget(method_widget)

        # Proporcions splitter: 60% equip, 40% metodològic
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        layout.addWidget(splitter, 1)

    def _load_from_config(self):
        """Carrega el path des de la configuracio."""
        try:
            from hpsec_config import get_config
            config = get_config()
            self.excel_path = config.get("paths", "maintenance_excel", default="")

            if self.excel_path and os.path.exists(self.excel_path):
                self.path_label.setText(self.excel_path)
                self.path_label.setStyleSheet("color: #888; font-family: monospace; font-size: 10px;")
                self._load_excel()
            else:
                self.info_label.setText(
                    "Fitxer no trobat. Clica 'Canviar...' per seleccionar-lo."
                )
                self.path_label.setText("No trobat: " + (self.excel_path or "(buit)"))
                self.path_label.setStyleSheet("color: #E74C3C; font-family: monospace; font-size: 10px;")
        except Exception as e:
            self.info_label.setText(f"Error carregant configuració: {e}")

    def _change_file(self):
        """Permet canviar el fitxer Excel i guarda el path a config."""
        start_dir = str(Path(self.excel_path).parent) if self.excel_path else ""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Selecciona fitxer Excel de manteniment",
            start_dir,
            "Excel Files (*.xlsx *.xls);;All Files (*)"
        )

        if file_path:
            self.excel_path = file_path
            self.path_label.setText(file_path)
            self.path_label.setStyleSheet("color: #666; font-family: monospace; font-size: 10px;")
            self._load_excel()

            # Guardar a config automàticament
            try:
                from hpsec_config import get_config
                config = get_config()
                config.set("paths", "maintenance_excel", file_path.replace("\\", "/"))
                config.save()
                self.info_label.setText(
                    f"Carregats {len(self.events)} events. Path guardat a config."
                )
            except Exception as e:
                QMessageBox.warning(
                    self, "Avís",
                    f"Dades carregades però no s'ha pogut guardar el path a config:\n{e}"
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

    # ------------------------------------------------------------------
    # Canvis metodològics
    # ------------------------------------------------------------------

    def _load_all(self):
        """Carrega manteniment (Excel) + canvis metodològics (JSON)."""
        self._load_from_config()
        self._load_method_log()

    def _load_method_log(self):
        """Carrega el registre de canvis metodològics des de JSON."""
        self.method_changes = []
        try:
            if METHOD_LOG_PATH.exists():
                with open(METHOD_LOG_PATH, 'r', encoding='utf-8') as f:
                    self.method_changes = _json.load(f)
                # Ordenar per data desc
                self.method_changes.sort(
                    key=lambda x: x.get('date', ''), reverse=True
                )
        except Exception as e:
            self.method_info.setText(f"Error llegint registre: {e}")

        self._refresh_method_table()

    def _save_method_log(self):
        """Guarda el registre de canvis metodològics a JSON."""
        try:
            METHOD_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            # Ordenar per data desc abans de guardar
            self.method_changes.sort(
                key=lambda x: x.get('date', ''), reverse=True
            )
            with open(METHOD_LOG_PATH, 'w', encoding='utf-8') as f:
                _json.dump(self.method_changes, f, indent=2, ensure_ascii=False)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"No s'ha pogut guardar:\n{e}")

    def _refresh_method_table(self):
        """Actualitza la taula de canvis metodològics."""
        self.method_table.setRowCount(0)

        for entry in self.method_changes:
            row = self.method_table.rowCount()
            self.method_table.insertRow(row)

            # Data
            date_str = entry.get('date', '')
            try:
                date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                date_display = date_obj.strftime("%d/%m/%Y")
            except Exception:
                date_display = date_str
            self.method_table.setItem(row, 0, QTableWidgetItem(date_display))

            # Categoria amb color
            cat = entry.get('category', '')
            cat_item = QTableWidgetItem(cat)
            color = METHOD_COLORS.get(cat, "#7F8C8D")
            cat_item.setForeground(QColor(color))
            cat_item.setFont(QFont("Segoe UI", 9, QFont.Bold))
            self.method_table.setItem(row, 1, cat_item)

            # Descripció
            desc = entry.get('description', '')
            desc_item = QTableWidgetItem(desc)
            desc_item.setToolTip(desc)
            self.method_table.setItem(row, 2, desc_item)

            # SEQ referència
            seq_ref = entry.get('seq_ref', '')
            self.method_table.setItem(row, 3, QTableWidgetItem(seq_ref))

        n = len(self.method_changes)
        self.method_info.setText(f"{n} canvi{'s' if n != 1 else ''} registrat{'s' if n != 1 else ''}")

    def _add_method_change(self):
        """Diàleg per afegir un canvi metodològic."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Afegir canvi metodològic")
        dlg.setMinimumWidth(450)

        form = QFormLayout(dlg)
        form.setSpacing(10)

        date_edit = QDateEdit()
        date_edit.setDate(QDate.currentDate())
        date_edit.setCalendarPopup(True)
        date_edit.setDisplayFormat("dd/MM/yyyy")
        form.addRow("Data:", date_edit)

        cat_combo = QComboBox()
        for cat in METHOD_CATEGORIES:
            cat_combo.addItem(cat)
        form.addRow("Categoria:", cat_combo)

        desc_edit = QTextEdit()
        desc_edit.setMaximumHeight(80)
        desc_edit.setPlaceholderText(
            "Ex: KHP amb pipetes Pasteur en lloc de micropipeta"
        )
        form.addRow("Descripció:", desc_edit)

        seq_edit = QLineEdit()
        seq_edit.setPlaceholderText("Ex: 111_SEQ (opcional)")
        form.addRow("SEQ referència:", seq_edit)

        regime_chk = QCheckBox(
            "Candidat a règim nou de calibració\n"
            "(el següent KHP el confirmarà o descartarà — no parteix cap bloc ara)"
        )
        regime_chk.setToolTip(
            "Marca-ho si el canvi pot alterar la resposta de l'instrument (columna,\n"
            "detector, guany...). El primer KHP/SEQ_CAL posterior farà el test\n"
            "d'equivalència: si el RF segueix dins tolerància, el candidat es descarta\n"
            "i el bloc continua; si trenca, s'obre règim nou amb frontera en aquesta data."
        )
        form.addRow("", regime_chk)
        cat_combo.currentTextChanged.connect(
            lambda cat: regime_chk.setChecked(cat in REGIME_SUSPECT_CATEGORIES)
        )
        regime_chk.setChecked(cat_combo.currentText() in REGIME_SUSPECT_CATEGORIES)

        # Botons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        cancel_btn = QPushButton("Cancel·lar")
        cancel_btn.clicked.connect(dlg.reject)
        btn_layout.addWidget(cancel_btn)
        save_btn = QPushButton("Guardar")
        save_btn.setStyleSheet(
            "background-color: #2980B9; color: white; border: none;"
            "border-radius: 4px; padding: 6px 16px; font-weight: bold;"
        )
        save_btn.clicked.connect(dlg.accept)
        btn_layout.addWidget(save_btn)
        form.addRow("", btn_layout)

        if dlg.exec() == QDialog.Accepted:
            desc = desc_edit.toPlainText().strip()
            if not desc:
                QMessageBox.warning(self, "Avís", "Cal una descripció.")
                return

            entry = {
                "date": date_edit.date().toString("yyyy-MM-dd"),
                "category": cat_combo.currentText(),
                "description": desc,
                "seq_ref": seq_edit.text().strip(),
                "added_at": datetime.now().isoformat(),
                "regime_candidate": regime_chk.isChecked(),
            }
            self.method_changes.append(entry)
            self._save_method_log()
            self._refresh_method_table()

            if regime_chk.isChecked():
                try:
                    from hpsec_calibrate import add_pending_regime_event
                    add_pending_regime_event(
                        entry["date"], desc,
                        seq_ref=entry["seq_ref"], source="event",
                    )
                except Exception as e:
                    QMessageBox.warning(
                        self, "Avís",
                        "El canvi s'ha desat, però no s'ha pogut registrar el "
                        f"candidat a règim:\n{e}"
                    )

    def _delete_method_change(self):
        """Elimina el canvi metodològic seleccionat."""
        rows = set(item.row() for item in self.method_table.selectedItems())
        if not rows:
            return

        resp = QMessageBox.question(
            self, "Confirmar",
            f"Eliminar {len(rows)} entrada{'es' if len(rows) > 1 else ''}?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if resp != QMessageBox.Yes:
            return

        # Eliminar en ordre invers per no desplaçar índexs
        for row in sorted(rows, reverse=True):
            if 0 <= row < len(self.method_changes):
                self.method_changes.pop(row)

        self._save_method_log()
        self._refresh_method_table()

    def get_method_changes_in_range(self, start_date, end_date):
        """Retorna canvis metodològics dins un rang de dates.

        Útil per mostrar marcadors als gràfics d'Històric KHP.
        """
        if isinstance(start_date, datetime):
            start_date = start_date.strftime("%Y-%m-%d")
        if isinstance(end_date, datetime):
            end_date = end_date.strftime("%Y-%m-%d")

        return [
            e for e in self.method_changes
            if start_date <= e.get('date', '') <= end_date
        ]
