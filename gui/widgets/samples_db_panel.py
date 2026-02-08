"""
HPSEC Suite - Samples Database Panel
=====================================

Panel per visualitzar la base de dades de mostres:
- Llistat de totes les mostres
- Historial d'aparicions
- Tendències temporals
- Comparació de mostres
- Configuració d'àlies
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGroupBox,
    QGridLayout, QFrame, QTableWidget, QTableWidgetItem, QHeaderView,
    QComboBox, QMessageBox, QSplitter, QTabWidget, QScrollArea,
    QSizePolicy, QCheckBox, QLineEdit, QDialog, QDialogButtonBox,
    QFormLayout, QFileDialog
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QColor

from pathlib import Path
from datetime import datetime
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from hpsec_samples_db import (
    load_samples_index, rebuild_samples_index, search_samples,
    get_sample_history, get_sample_trends, compare_samples,
    load_sample_aliases, get_sample_alias, get_excel_columns,
    get_excel_preview, configure_alias_mapping
)

# Matplotlib
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class SamplesDBPanel(QWidget):
    """Panel de base de dades de mostres."""

    sample_selected = Signal(str)  # Emès quan es selecciona una mostra

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self._all_samples = []
        self._selected_sample = None
        self._setup_ui()

    def _setup_ui(self):
        """Configura la interfície."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        # Títol
        title = QLabel("Base de Dades de Mostres")
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        layout.addWidget(title)

        # Info
        info = QLabel(
            "Visualitza totes les mostres analitzades. "
            "Pots cercar per nom o àlies, filtrar per tipus, i veure l'historial de cada mostra."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #666;")
        layout.addWidget(info)

        # Toolbar amb filtres
        toolbar = QHBoxLayout()

        # Cerca
        toolbar.addWidget(QLabel("Cercar:"))
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Nom o àlies...")
        self.search_edit.setMaximumWidth(200)
        self.search_edit.textChanged.connect(self._apply_filters)
        toolbar.addWidget(self.search_edit)

        # Filtre tipus
        toolbar.addWidget(QLabel("Tipus:"))
        self.type_filter = QComboBox()
        self.type_filter.addItem("Tots", None)
        self.type_filter.addItem("Mostres", "MOSTRA")
        self.type_filter.addItem("Controls", "CONTROL")
        self.type_filter.addItem("Patrons", "PATRÓ_CAL")
        self.type_filter.currentIndexChanged.connect(self._apply_filters)
        toolbar.addWidget(self.type_filter)

        # Filtre mode
        toolbar.addWidget(QLabel("Mode:"))
        self.mode_filter = QComboBox()
        self.mode_filter.addItem("Tots", None)
        self.mode_filter.addItem("COLUMN", "COLUMN")
        self.mode_filter.addItem("BP", "BP")
        self.mode_filter.currentIndexChanged.connect(self._apply_filters)
        toolbar.addWidget(self.mode_filter)

        # Filtre BP aparellat
        toolbar.addWidget(QLabel("BP:"))
        self.bp_filter = QComboBox()
        self.bp_filter.addItem("Tots", None)
        self.bp_filter.addItem("Amb BP", True)
        self.bp_filter.addItem("Sense BP", False)
        self.bp_filter.currentIndexChanged.connect(self._apply_filters)
        toolbar.addWidget(self.bp_filter)

        toolbar.addStretch()

        # Botó configurar àlies
        self.alias_btn = QPushButton("Configurar àlies...")
        self.alias_btn.clicked.connect(self._show_alias_config)
        toolbar.addWidget(self.alias_btn)

        # Botó refrescar
        self.refresh_btn = QPushButton("Actualitzar")
        self.refresh_btn.clicked.connect(self._load_samples)
        toolbar.addWidget(self.refresh_btn)

        # Botó reconstruir índex
        self.rebuild_btn = QPushButton("Reconstruir")
        self.rebuild_btn.setToolTip("Reconstrueix l'índex escanejant totes les SEQs")
        self.rebuild_btn.clicked.connect(self._rebuild_index)
        toolbar.addWidget(self.rebuild_btn)

        layout.addLayout(toolbar)

        # Contingut amb tabs
        self.content_tabs = QTabWidget()

        # === TAB 1: Llistat ===
        list_widget = QWidget()
        list_layout = QVBoxLayout(list_widget)
        list_layout.setContentsMargins(0, 8, 0, 0)

        self.samples_table = QTableWidget()
        self.samples_table.setColumnCount(8)
        self.samples_table.setHorizontalHeaderLabels([
            "Mostra", "Àlies", "Tipus", "Primera", "Última", "N", "Àrea", "BP?"
        ])

        headers = self.samples_table.horizontalHeader()
        headers.setSectionResizeMode(QHeaderView.ResizeToContents)
        headers.setSectionResizeMode(0, QHeaderView.Stretch)  # Mostra expandeix
        headers.setSectionResizeMode(1, QHeaderView.Stretch)  # Àlies expandeix
        self.samples_table.setAlternatingRowColors(True)
        self.samples_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.samples_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.samples_table.setSortingEnabled(True)
        self.samples_table.verticalHeader().setVisible(False)
        self.samples_table.itemSelectionChanged.connect(self._on_sample_selected)
        self.samples_table.doubleClicked.connect(self._on_sample_double_click)

        list_layout.addWidget(self.samples_table)
        self.content_tabs.addTab(list_widget, "Llistat")

        # === TAB 2: Detall ===
        detail_widget = QWidget()
        detail_layout = QVBoxLayout(detail_widget)
        detail_layout.setContentsMargins(0, 8, 0, 0)

        # Info mostra
        self.detail_info = QGroupBox("Informació")
        info_layout = QGridLayout(self.detail_info)

        self.detail_name = QLabel("-")
        self.detail_name.setFont(QFont("Segoe UI", 12, QFont.Bold))
        info_layout.addWidget(QLabel("Nom:"), 0, 0)
        info_layout.addWidget(self.detail_name, 0, 1)

        self.detail_alias = QLabel("-")
        info_layout.addWidget(QLabel("Àlies:"), 0, 2)
        info_layout.addWidget(self.detail_alias, 0, 3)

        self.detail_type = QLabel("-")
        info_layout.addWidget(QLabel("Tipus:"), 1, 0)
        info_layout.addWidget(self.detail_type, 1, 1)

        self.detail_appearances = QLabel("-")
        info_layout.addWidget(QLabel("Aparicions:"), 1, 2)
        info_layout.addWidget(self.detail_appearances, 1, 3)

        detail_layout.addWidget(self.detail_info)

        # Taula d'aparicions
        detail_layout.addWidget(QLabel("Historial d'aparicions:"))
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(7)
        self.history_table.setHorizontalHeaderLabels([
            "Data", "SEQ", "Mode", "Rèpliques", "Àrea", "[ppm]", "BP?"
        ])
        headers2 = self.history_table.horizontalHeader()
        headers2.setSectionResizeMode(QHeaderView.ResizeToContents)
        headers2.setSectionResizeMode(1, QHeaderView.Stretch)
        self.history_table.setAlternatingRowColors(True)
        self.history_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.history_table.verticalHeader().setVisible(False)
        detail_layout.addWidget(self.history_table)

        self.content_tabs.addTab(detail_widget, "Detall")

        # === TAB 3: Tendències ===
        trends_widget = QWidget()
        trends_layout = QVBoxLayout(trends_widget)
        trends_layout.setContentsMargins(0, 8, 0, 0)

        # Selector de mètrica
        metric_layout = QHBoxLayout()
        metric_layout.addWidget(QLabel("Mètrica:"))
        self.metric_combo = QComboBox()
        self.metric_combo.addItem("Àrea Total", "area_total")
        self.metric_combo.addItem("Concentració (ppm)", "concentration_ppm")
        self.metric_combo.addItem("Fraccions (%)", "fractions")
        self.metric_combo.currentIndexChanged.connect(self._update_trend_chart)
        metric_layout.addWidget(self.metric_combo)
        metric_layout.addStretch()
        trends_layout.addLayout(metric_layout)

        # Gràfic
        self.trend_figure = Figure(figsize=(10, 5), dpi=100)
        self.trend_canvas = FigureCanvas(self.trend_figure)
        trends_layout.addWidget(self.trend_canvas)

        self.content_tabs.addTab(trends_widget, "Tendències")

        # === TAB 4: Comparar ===
        compare_widget = QWidget()
        compare_layout = QVBoxLayout(compare_widget)
        compare_layout.setContentsMargins(0, 8, 0, 0)

        # Instruccions
        compare_layout.addWidget(QLabel(
            "Selecciona múltiples mostres al llistat (Ctrl+clic) per comparar-les."
        ))

        # Taula comparativa
        self.compare_table = QTableWidget()
        self.compare_table.setColumnCount(7)
        self.compare_table.setHorizontalHeaderLabels([
            "Mostra", "N", "Última Àrea", "Àrea Mitja", "Última [ppm]", "[ppm] Mitjà", "HS%"
        ])
        headers3 = self.compare_table.horizontalHeader()
        headers3.setSectionResizeMode(QHeaderView.ResizeToContents)
        headers3.setSectionResizeMode(0, QHeaderView.Stretch)
        self.compare_table.setAlternatingRowColors(True)
        self.compare_table.setEditTriggers(QTableWidget.NoEditTriggers)
        compare_layout.addWidget(self.compare_table)

        # Gràfic comparatiu
        self.compare_figure = Figure(figsize=(10, 4), dpi=100)
        self.compare_canvas = FigureCanvas(self.compare_figure)
        compare_layout.addWidget(self.compare_canvas)

        self.content_tabs.addTab(compare_widget, "Comparar")

        layout.addWidget(self.content_tabs)

        # Resum
        summary_layout = QHBoxLayout()
        self.summary_label = QLabel()
        self.summary_label.setStyleSheet("color: #666;")
        summary_layout.addWidget(self.summary_label)
        summary_layout.addStretch()
        layout.addLayout(summary_layout)

        # Permetre selecció múltiple
        self.samples_table.setSelectionMode(QTableWidget.ExtendedSelection)

    def showEvent(self, event):
        """Carrega les dades quan es mostra el panel."""
        super().showEvent(event)
        self._load_samples()

    def _load_samples(self):
        """Carrega totes les mostres de l'índex."""
        self._all_samples = search_samples()
        self._apply_filters()
        self._update_summary()

    def _apply_filters(self):
        """Aplica els filtres a la taula."""
        query = self.search_edit.text().strip() or None
        sample_type = self.type_filter.currentData()
        seq_type = self.mode_filter.currentData()
        has_bp = self.bp_filter.currentData()

        # Cercar amb filtres
        filtered = search_samples(
            query=query,
            sample_type=sample_type,
            seq_type=seq_type,
            has_bp=has_bp
        )

        # Actualitzar taula
        self.samples_table.setSortingEnabled(False)
        self.samples_table.setRowCount(len(filtered))

        for row, sample in enumerate(filtered):
            # Mostra
            self.samples_table.setItem(row, 0,
                QTableWidgetItem(sample.get("sample_name", "")))

            # Àlies
            alias = sample.get("alias", "") or "-"
            self.samples_table.setItem(row, 1, QTableWidgetItem(alias))

            # Tipus
            self.samples_table.setItem(row, 2,
                QTableWidgetItem(sample.get("sample_type", "")))

            # Primera
            self.samples_table.setItem(row, 3,
                QTableWidgetItem(sample.get("first_seen", "")[:10]))

            # Última
            self.samples_table.setItem(row, 4,
                QTableWidgetItem(sample.get("last_seen", "")[:10]))

            # N aparicions
            n_item = QTableWidgetItem()
            n_item.setData(Qt.DisplayRole, sample.get("n_appearances", 0))
            self.samples_table.setItem(row, 5, n_item)

            # Àrea (última)
            appearances = sample.get("appearances", [])
            last_area = ""
            has_bp_txt = "-"
            if appearances:
                last = appearances[-1]
                area = last.get("area_total")
                if area is not None:
                    last_area = f"{area:.1f}"
                if last.get("has_bp_paired"):
                    has_bp_txt = "Sí"
                else:
                    has_bp_txt = "No"

            self.samples_table.setItem(row, 6, QTableWidgetItem(last_area))
            self.samples_table.setItem(row, 7, QTableWidgetItem(has_bp_txt))

            # Guardar key per referència
            item = self.samples_table.item(row, 0)
            item.setData(Qt.UserRole, sample.get("key"))

        self.samples_table.setSortingEnabled(True)
        self._update_summary()

    def _update_summary(self):
        """Actualitza el resum."""
        total = len(self._all_samples)
        visible = self.samples_table.rowCount()
        self.summary_label.setText(
            f"Mostrant {visible} de {total} mostres"
        )

    def _on_sample_selected(self):
        """Gestiona selecció de mostra."""
        selected = self.samples_table.selectionModel().selectedRows()

        if len(selected) == 1:
            # Una sola mostra: mostrar detall
            row = selected[0].row()
            item = self.samples_table.item(row, 0)
            sample_name = item.text()
            self._selected_sample = sample_name
            self._update_detail(sample_name)
            self._update_trend_chart()

        elif len(selected) > 1:
            # Múltiples: mostrar comparació
            names = []
            for idx in selected:
                item = self.samples_table.item(idx.row(), 0)
                names.append(item.text())
            self._update_comparison(names)

    def _on_sample_double_click(self):
        """Doble clic: anar a tab detall."""
        self.content_tabs.setCurrentIndex(1)

    def _update_detail(self, sample_name: str):
        """Actualitza la vista de detall."""
        # Info bàsica
        self.detail_name.setText(sample_name)
        alias = get_sample_alias(sample_name)
        self.detail_alias.setText(alias if alias else "-")

        # Cercar mostra
        samples = search_samples(query=sample_name)
        sample = samples[0] if samples else None

        if not sample:
            self.detail_type.setText("-")
            self.detail_appearances.setText("-")
            self.history_table.setRowCount(0)
            return

        self.detail_type.setText(sample.get("sample_type", "-"))
        self.detail_appearances.setText(str(sample.get("n_appearances", 0)))

        # Historial
        history = get_sample_history(sample_name)
        self.history_table.setRowCount(len(history))

        for row, app in enumerate(history):
            # Data
            date = app.get("date_processed", "")[:10]
            self.history_table.setItem(row, 0, QTableWidgetItem(date))

            # SEQ
            self.history_table.setItem(row, 1,
                QTableWidgetItem(app.get("seq_name", "")))

            # Mode
            self.history_table.setItem(row, 2,
                QTableWidgetItem(app.get("seq_type", "")))

            # Rèpliques
            n_rep = app.get("n_replicas", 0)
            self.history_table.setItem(row, 3,
                QTableWidgetItem(str(n_rep)))

            # Àrea
            area = app.get("area_total")
            area_txt = f"{area:.1f}" if area else "-"
            self.history_table.setItem(row, 4, QTableWidgetItem(area_txt))

            # Concentració
            conc = app.get("concentration_ppm")
            conc_txt = f"{conc:.2f}" if conc else "-"
            self.history_table.setItem(row, 5, QTableWidgetItem(conc_txt))

            # BP
            bp_txt = "Sí" if app.get("has_bp_paired") else "No"
            self.history_table.setItem(row, 6, QTableWidgetItem(bp_txt))

    def _update_trend_chart(self):
        """Actualitza el gràfic de tendències."""
        if not self._selected_sample:
            return

        metric = self.metric_combo.currentData()
        trends = get_sample_trends(self._selected_sample)

        if not trends or not trends.get("dates"):
            self.trend_figure.clear()
            self.trend_canvas.draw()
            return

        self.trend_figure.clear()
        ax = self.trend_figure.add_subplot(111)

        dates = trends["dates"]
        x = range(len(dates))

        if metric == "area_total":
            values = trends.get("area_total", [])
            ax.plot(x, values, 'b-o', linewidth=2, markersize=6)
            ax.set_ylabel("Àrea Total")
            ax.set_title(f"Evolució Àrea - {self._selected_sample}")

        elif metric == "concentration_ppm":
            values = trends.get("concentration_ppm", [])
            ax.plot(x, values, 'g-o', linewidth=2, markersize=6)
            ax.set_ylabel("Concentració (ppm)")
            ax.set_title(f"Evolució Concentració - {self._selected_sample}")

        elif metric == "fractions":
            fracs = trends.get("fractions", {})
            colors = ['#E63946', '#F6AE2D', '#2A9D8F', '#457B9D', '#1D3557']
            for i, (frac, values) in enumerate(fracs.items()):
                if any(v is not None for v in values):
                    ax.plot(x, values, '-o', linewidth=2, markersize=4,
                           label=frac, color=colors[i % len(colors)])
            ax.set_ylabel("Fracció (%)")
            ax.set_title(f"Evolució Fraccions - {self._selected_sample}")
            ax.legend(loc='upper right')

        ax.set_xlabel("Anàlisi")
        ax.set_xticks(x)
        ax.set_xticklabels(dates, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)

        self.trend_figure.tight_layout()
        self.trend_canvas.draw()

    def _update_comparison(self, sample_names: list):
        """Actualitza la comparació de mostres."""
        comparison = compare_samples(sample_names)
        samples = comparison.get("samples", [])

        # Taula
        self.compare_table.setRowCount(len(samples))

        for row, s in enumerate(samples):
            self.compare_table.setItem(row, 0,
                QTableWidgetItem(s.get("name", "")))

            n_item = QTableWidgetItem()
            n_item.setData(Qt.DisplayRole, s.get("n_appearances", 0))
            self.compare_table.setItem(row, 1, n_item)

            last_area = s.get("last_area")
            self.compare_table.setItem(row, 2,
                QTableWidgetItem(f"{last_area:.1f}" if last_area else "-"))

            avg_area = s.get("avg_area")
            self.compare_table.setItem(row, 3,
                QTableWidgetItem(f"{avg_area:.1f}" if avg_area else "-"))

            last_conc = s.get("last_conc")
            self.compare_table.setItem(row, 4,
                QTableWidgetItem(f"{last_conc:.2f}" if last_conc else "-"))

            avg_conc = s.get("avg_conc")
            self.compare_table.setItem(row, 5,
                QTableWidgetItem(f"{avg_conc:.2f}" if avg_conc else "-"))

            fracs = s.get("last_fractions", {})
            hs = fracs.get("HS")
            self.compare_table.setItem(row, 6,
                QTableWidgetItem(f"{hs:.1f}" if hs else "-"))

        # Gràfic
        self._update_compare_chart(samples)

        # Anar a tab comparar
        self.content_tabs.setCurrentIndex(3)

    def _update_compare_chart(self, samples: list):
        """Actualitza el gràfic comparatiu."""
        self.compare_figure.clear()
        ax = self.compare_figure.add_subplot(111)

        if not samples:
            self.compare_canvas.draw()
            return

        names = [s.get("name", "")[:15] for s in samples]
        areas = [s.get("avg_area") or 0 for s in samples]

        x = range(len(names))
        bars = ax.bar(x, areas, color='#2E86AB', alpha=0.8)

        ax.set_ylabel("Àrea Mitjana")
        ax.set_title("Comparació d'Àrees")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')

        self.compare_figure.tight_layout()
        self.compare_canvas.draw()

    def _show_alias_config(self):
        """Mostra el diàleg de configuració d'àlies."""
        dialog = AliasConfigDialog(self)
        if dialog.exec() == QDialog.Accepted:
            # Refrescar per mostrar nous àlies
            self._load_samples()

    def _rebuild_index(self):
        """Reconstrueix l'índex de mostres."""
        reply = QMessageBox.question(
            self,
            "Reconstruir índex",
            "Això escanejará totes les SEQs i pot trigar uns moments.\n\nContinuar?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        # Reconstruir
        self.main_window.set_status("Reconstruint índex de mostres...")
        self.main_window.show_progress(0, 100)

        try:
            def progress(current, total, msg):
                pct = int(100 * current / total) if total > 0 else 0
                self.main_window.show_progress(pct, 100)
                self.main_window.set_status(msg)

            rebuild_samples_index(progress_callback=progress)

            self.main_window.show_progress(-1)
            self.main_window.set_status("Índex reconstruït", 3000)

            # Refrescar
            self._load_samples()

        except Exception as e:
            self.main_window.show_progress(-1)
            QMessageBox.critical(
                self,
                "Error",
                f"Error reconstruint índex:\n{str(e)}"
            )


class AliasConfigDialog(QDialog):
    """Diàleg per configurar el fitxer d'àlies."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configurar fitxer d'àlies")
        self.setMinimumWidth(500)
        self._columns = []
        self._setup_ui()
        self._load_current_config()

    def _setup_ui(self):
        """Configura la interfície."""
        layout = QVBoxLayout(self)

        # Selector de fitxer
        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("Fitxer Excel:"))
        self.file_edit = QLineEdit()
        self.file_edit.setReadOnly(True)
        file_layout.addWidget(self.file_edit)
        self.browse_btn = QPushButton("Examinar...")
        self.browse_btn.clicked.connect(self._browse_file)
        file_layout.addWidget(self.browse_btn)
        layout.addLayout(file_layout)

        # Selectors de columnes
        cols_group = QGroupBox("Columnes")
        cols_layout = QFormLayout(cols_group)

        self.id_combo = QComboBox()
        self.id_combo.currentIndexChanged.connect(self._update_preview)
        cols_layout.addRow("Columna ID mostra:", self.id_combo)

        self.alias_combo = QComboBox()
        self.alias_combo.currentIndexChanged.connect(self._update_preview)
        cols_layout.addRow("Columna àlies:", self.alias_combo)

        layout.addWidget(cols_group)

        # Vista prèvia
        preview_group = QGroupBox("Vista prèvia")
        preview_layout = QVBoxLayout(preview_group)

        self.preview_table = QTableWidget()
        self.preview_table.setColumnCount(2)
        self.preview_table.setHorizontalHeaderLabels(["ID", "Àlies"])
        self.preview_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.preview_table.setMaximumHeight(150)
        preview_layout.addWidget(self.preview_table)

        layout.addWidget(preview_group)

        # Botons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self._save_config)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _load_current_config(self):
        """Carrega la configuració actual."""
        from hpsec_config import get_config
        cfg = get_config()

        excel_path = cfg.get("paths", "samples_alias_excel", "")
        if excel_path:
            self.file_edit.setText(excel_path)
            self._load_columns(excel_path)

            samples_db = cfg.get_section("samples_db") or {}
            id_col = samples_db.get("alias_column_id", "")
            alias_col = samples_db.get("alias_column_alias", "")

            if id_col:
                idx = self.id_combo.findText(id_col)
                if idx >= 0:
                    self.id_combo.setCurrentIndex(idx)

            if alias_col:
                idx = self.alias_combo.findText(alias_col)
                if idx >= 0:
                    self.alias_combo.setCurrentIndex(idx)

    def _browse_file(self):
        """Obre diàleg per seleccionar fitxer."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Selecciona fitxer Excel",
            "",
            "Excel files (*.xlsx *.xls);;All files (*.*)"
        )
        if path:
            self.file_edit.setText(path)
            self._load_columns(path)

    def _load_columns(self, path: str):
        """Carrega les columnes del fitxer."""
        self._columns = get_excel_columns(path)

        self.id_combo.clear()
        self.alias_combo.clear()

        for col in self._columns:
            self.id_combo.addItem(col)
            self.alias_combo.addItem(col)

        # Seleccionar segona columna per àlies si hi ha múltiples
        if len(self._columns) > 1:
            self.alias_combo.setCurrentIndex(1)

        self._update_preview()

    def _update_preview(self):
        """Actualitza la vista prèvia."""
        path = self.file_edit.text()
        id_col = self.id_combo.currentText()
        alias_col = self.alias_combo.currentText()

        if not path or not id_col or not alias_col:
            self.preview_table.setRowCount(0)
            return

        preview = get_excel_preview(path, id_col, alias_col, n_rows=5)

        self.preview_table.setRowCount(len(preview))
        for row, item in enumerate(preview):
            self.preview_table.setItem(row, 0,
                QTableWidgetItem(item.get("id", "")))
            self.preview_table.setItem(row, 1,
                QTableWidgetItem(item.get("alias", "")))

    def _save_config(self):
        """Guarda la configuració."""
        path = self.file_edit.text()
        id_col = self.id_combo.currentText()
        alias_col = self.alias_combo.currentText()

        if not path:
            QMessageBox.warning(self, "Avís", "Cal seleccionar un fitxer Excel.")
            return

        if not id_col or not alias_col:
            QMessageBox.warning(self, "Avís", "Cal seleccionar les columnes.")
            return

        if id_col == alias_col:
            QMessageBox.warning(self, "Avís", "Les columnes han de ser diferents.")
            return

        if configure_alias_mapping(path, id_col, alias_col):
            QMessageBox.information(
                self, "Configurat",
                "La configuració d'àlies s'ha guardat correctament."
            )
            self.accept()
        else:
            QMessageBox.critical(
                self, "Error",
                "No s'ha pogut guardar la configuració."
            )
