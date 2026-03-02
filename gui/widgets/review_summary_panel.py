"""
HPSEC Suite - Review Summary Panel (Pas 4: Revisar)
=====================================================

Panel de revisió i validació abans de generar resultats.

Contingut:
- 3 cards informatives: Seqüència, Timeouts TOC, Control Qualitat
- Gràfic DOC ppm stacked per fraccions (BioP|HS|BB|SB|LMW)
- Gràfic A254 àrea per mostra
- Botó "Generar Resultats" → Excels a SEQ/RESULTATS/, SUMMARY a SEQ/CHECK/
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QScrollArea, QMessageBox, QSizePolicy, QProgressBar,
    QTableWidget, QTableWidgetItem, QComboBox, QHeaderView,
    QGroupBox, QCheckBox, QGridLayout,
)
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QFont, QColor

import json as _json
import logging
import os

import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from gui.widgets.styles import (
    COLOR_SUCCESS, COLOR_WARNING, COLOR_ERROR,
    apply_panel_layout
)

# Fraction colors (consistent palette)
FRACTION_COLORS = {
    "BioP": "#3498DB",  # Blue
    "HS":   "#E74C3C",  # Red
    "BB":   "#F39C12",  # Orange
    "SB":   "#2ECC71",  # Green
    "LMW":  "#9B59B6",  # Purple
}
FRACTION_ORDER = ["BioP", "HS", "BB", "SB", "LMW"]


class BPDiscoveryWorker(QThread):
    """Worker per cercar dades BP en background."""
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, seq_path, sample_names, data_folder=None):
        super().__init__()
        self.seq_path = seq_path
        self.sample_names = sample_names
        self.data_folder = data_folder

    def run(self):
        try:
            from hpsec_consolidate import find_bp_for_samples
            result = find_bp_for_samples(
                self.seq_path, self.sample_names, self.data_folder
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class _BPReloadWorker(QThread):
    """Worker per recarregar dades BP des d'una BP diferent (canvi dropdown)."""
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, bp_path, sample_names, available_bps, column_seq_path):
        super().__init__()
        self.bp_path = bp_path
        self.sample_names = sample_names
        self.available_bps = available_bps
        self.column_seq_path = column_seq_path

    def run(self):
        try:
            from hpsec_consolidate import load_bp_data_for_sample
            result = {
                "primary_bp": {
                    "path": self.bp_path,
                    "name": Path(self.bp_path).name,
                },
                "available_bps": self.available_bps,
                "samples": {},
            }
            for name in self.sample_names:
                bp_data = load_bp_data_for_sample(self.bp_path, name)
                if bp_data:
                    result["samples"][name] = {
                        "bp_seq": Path(self.bp_path).name,
                        "bp_data": bp_data,
                        "source": "manual",
                    }
                else:
                    result["samples"][name] = {
                        "bp_seq": None,
                        "bp_data": None,
                        "source": None,
                    }
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class GenerateWorker(QThread):
    """Worker per generar resultats en background."""
    progress = Signal(int, str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, samples_grouped, seq_path, calibration_data, mode, config,
                 bp_resolved=None):
        super().__init__()
        self.samples_grouped = samples_grouped
        self.seq_path = seq_path
        self.calibration_data = calibration_data
        self.mode = mode
        self.config = config
        self.bp_resolved = bp_resolved

    def run(self):
        try:
            from hpsec_export import export_sequence, generate_summary_excel, DEFAULT_EXPORT_CONFIG
            config = self.config or DEFAULT_EXPORT_CONFIG
            results = {"excel_files": None, "summary": None, "errors": []}

            def progress_cb(pct, msg):
                self.progress.emit(pct, msg)

            # Excels individuals → SEQ/RESULTATS/
            resultats_path = str(Path(self.seq_path) / "RESULTATS")
            self.progress.emit(0, "Generant Excels individuals...")
            excel_result = export_sequence(
                self.samples_grouped,
                resultats_path,
                self.calibration_data,
                self.mode,
                config,
                progress_cb,
                seq_path=self.seq_path,
                bp_resolved=self.bp_resolved,
            )
            results["excel_files"] = excel_result
            results["errors"].extend(excel_result.get("errors", []))

            # SUMMARY.xlsx → SEQ/CHECK/
            self.progress.emit(90, "Generant SUMMARY.xlsx...")
            check_path = Path(self.seq_path) / "CHECK"
            check_path.mkdir(parents=True, exist_ok=True)
            summary_path = str(check_path / "SUMMARY.xlsx")
            summary_result = generate_summary_excel(
                self.samples_grouped,
                summary_path,
                self.calibration_data,
                self.mode,
                config,
            )
            results["summary"] = summary_result

            self.progress.emit(100, "Completat")
            self.finished.emit(results)

        except Exception as e:
            self.error.emit(str(e))


class ReviewSummaryPanel(QWidget):
    """Panel de revisió i generació de resultats (Pas 4)."""

    review_completed = Signal(dict)

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.worker = None
        self._bp_worker = None
        self._bp_resolved = None
        self._bp_available = []  # BPs disponibles per dropdown
        self._current_method = "COLUMN"
        self._current_seq_path = ""
        self._current_sample_names = []
        self._auto_generated = False  # Evitar doble generació
        self._populated_seq = ""  # Evitar doble populate
        self._sample_checkboxes = []
        self._chart_regular = {}
        self._chart_light = {}
        self._chart_khp = {}
        self._chart_is_bp = False
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(0)

        # Scrollable content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        content = QWidget()
        self.content_layout = QVBoxLayout(content)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(12)
        scroll.setWidget(content)

        # === HEADER COMPACTE ===
        self.header_label = QLabel()
        self.header_label.setTextFormat(Qt.RichText)
        self.header_label.setWordWrap(True)
        self.header_label.setStyleSheet(
            "font-size: 12px; color: #2c3e50; padding: 6px 10px;"
            "background: #fff; border: 1px solid #e0e0e0; border-radius: 6px;"
        )
        self.content_layout.addWidget(self.header_label)

        # === GRÀFICS ===
        if HAS_MATPLOTLIB:
            # Timeout timeline
            self.timeout_figure = Figure(figsize=(10, 1.2), dpi=100)
            self.timeout_figure.set_facecolor("#FAFAFA")
            self.timeout_canvas = FigureCanvas(self.timeout_figure)
            self.timeout_canvas.setMinimumHeight(80)
            self.timeout_canvas.setMaximumHeight(100)
            self.content_layout.addWidget(self.timeout_canvas)

            # === SELECCIÓ MOSTRES (checkboxes) ===
            sel_frame = QFrame()
            sel_frame.setStyleSheet(
                "QFrame { background: #fff; border: 1px solid #e0e0e0;"
                " border-radius: 6px; }"
            )
            sel_layout = QVBoxLayout(sel_frame)
            sel_layout.setContentsMargins(8, 6, 8, 6)
            sel_layout.setSpacing(4)

            sel_header = QHBoxLayout()
            sel_header.addWidget(QLabel(
                "<b style='font-size:11px;color:#555'>Mostres a visualitzar</b>"
            ))
            self.btn_sel_all = QPushButton("Tot")
            self.btn_sel_samples = QPushButton("Mostres")
            self.btn_sel_none = QPushButton("Cap")
            for btn in (self.btn_sel_all, self.btn_sel_samples, self.btn_sel_none):
                btn.setFixedHeight(22)
                btn.setStyleSheet(
                    "QPushButton { font-size: 10px; padding: 2px 8px;"
                    " border: 1px solid #ccc; border-radius: 3px; background: #f8f8f8; }"
                    "QPushButton:hover { background: #e8e8e8; }"
                )
                sel_header.addWidget(btn)
            sel_header.addStretch()
            sel_layout.addLayout(sel_header)

            self.samples_check_scroll = QScrollArea()
            self.samples_check_scroll.setWidgetResizable(True)
            self.samples_check_scroll.setMaximumHeight(80)
            self.samples_check_scroll.setFrameShape(QFrame.NoFrame)
            self.samples_check_widget = QWidget()
            self.samples_check_grid = QGridLayout(self.samples_check_widget)
            self.samples_check_grid.setContentsMargins(0, 0, 0, 0)
            self.samples_check_grid.setSpacing(2)
            self.samples_check_scroll.setWidget(self.samples_check_widget)
            sel_layout.addWidget(self.samples_check_scroll)

            self.content_layout.addWidget(sel_frame)

            self.btn_sel_all.clicked.connect(lambda: self._set_all_checks(True))
            self.btn_sel_samples.clicked.connect(self._check_only_samples)
            self.btn_sel_none.clicked.connect(lambda: self._set_all_checks(False))

            self._sample_checkboxes = []  # list of (QCheckBox, name, category)

            # DOC stacked bar
            self.doc_figure = Figure(figsize=(10, 3.5), dpi=100)
            self.doc_figure.set_facecolor("#FAFAFA")
            self.doc_canvas = FigureCanvas(self.doc_figure)
            self.doc_canvas.setMinimumHeight(220)
            self.content_layout.addWidget(self.doc_canvas)

            # DOC overlay (chromatograms)
            self.doc_overlay_figure = Figure(figsize=(10, 4), dpi=100)
            self.doc_overlay_figure.set_facecolor("#FAFAFA")
            self.doc_overlay_canvas = FigureCanvas(self.doc_overlay_figure)
            self.doc_overlay_canvas.setMinimumHeight(250)
            self.content_layout.addWidget(self.doc_overlay_canvas)

            # A254 bar
            self.dad_figure = Figure(figsize=(10, 2.5), dpi=100)
            self.dad_figure.set_facecolor("#FAFAFA")
            self.dad_canvas = FigureCanvas(self.dad_figure)
            self.dad_canvas.setMinimumHeight(180)
            self.content_layout.addWidget(self.dad_canvas)

            # DAD overlay (254nm chromatograms)
            self.dad_overlay_figure = Figure(figsize=(10, 4), dpi=100)
            self.dad_overlay_figure.set_facecolor("#FAFAFA")
            self.dad_overlay_canvas = FigureCanvas(self.dad_overlay_figure)
            self.dad_overlay_canvas.setMinimumHeight(250)
            self.content_layout.addWidget(self.dad_overlay_canvas)

        # === SECCIÓ CONSOLIDACIÓ BP ===
        self.bp_group = QGroupBox("CONSOLIDACIÓ BP")
        self.bp_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold; font-size: 11px; color: #2c3e50;
                border: 1px solid #d5dbdb; border-radius: 6px;
                margin-top: 8px; padding-top: 18px;
                background-color: #fafafa;
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 12px; padding: 0 4px;
            }
        """)
        bp_layout = QVBoxLayout(self.bp_group)
        bp_layout.setContentsMargins(12, 8, 12, 8)
        bp_layout.setSpacing(6)

        # Dropdown per seleccionar BP
        bp_selector_row = QHBoxLayout()
        bp_selector_row.addWidget(QLabel("SEQ BP:"))
        self.bp_combo = QComboBox()
        self.bp_combo.setMinimumWidth(220)
        self.bp_combo.currentIndexChanged.connect(self._on_bp_combo_changed)
        bp_selector_row.addWidget(self.bp_combo)
        self.bp_status_label = QLabel("")
        self.bp_status_label.setStyleSheet("color: #7f8c8d; font-size: 11px;")
        bp_selector_row.addWidget(self.bp_status_label, 1)
        bp_layout.addLayout(bp_selector_row)

        # Taula de mostres BP
        self.bp_table = QTableWidget()
        self.bp_table.setColumnCount(5)
        self.bp_table.setHorizontalHeaderLabels(
            ["Mostra", "BP", "Rèplica", "ppm", "SNR"]
        )
        self.bp_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for col in range(1, 5):
            self.bp_table.horizontalHeader().setSectionResizeMode(
                col, QHeaderView.ResizeToContents
            )
        self.bp_table.verticalHeader().setVisible(False)
        self.bp_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.bp_table.setAlternatingRowColors(True)
        self.bp_table.setMaximumHeight(200)
        self.bp_table.setStyleSheet("""
            QTableWidget {
                font-size: 11px; border: 1px solid #e0e0e0;
                gridline-color: #f0f0f0;
            }
            QTableWidget::item { padding: 2px 6px; }
        """)
        bp_layout.addWidget(self.bp_table)

        # Info label (mostres sense BP, etc.)
        self.bp_info_label = QLabel("")
        self.bp_info_label.setStyleSheet("color: #7f8c8d; font-size: 10px; font-style: italic;")
        bp_layout.addWidget(self.bp_info_label)

        self.bp_group.setVisible(False)  # Oculta per defecte (només per COLUMN)
        self.content_layout.addWidget(self.bp_group)

        # === BOTÓ GENERAR + PROGRÉS ===
        gen_frame = QFrame()
        gen_frame.setStyleSheet("""
            QFrame {
                background-color: #f0f7ff;
                border: 1px solid #b8d4f0;
                border-radius: 8px;
            }
        """)
        gen_layout = QVBoxLayout(gen_frame)
        gen_layout.setContentsMargins(20, 16, 20, 16)
        gen_layout.setSpacing(8)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.generate_btn = QPushButton("Generar Resultats")
        self.generate_btn.setStyleSheet("""
            QPushButton {
                background-color: #2980B9; color: white;
                border: none; border-radius: 6px;
                padding: 12px 32px; font-size: 14px; font-weight: bold;
            }
            QPushButton:hover { background-color: #3498DB; }
            QPushButton:disabled { background-color: #BDC3C7; }
        """)
        self.generate_btn.clicked.connect(self._run_generate)
        btn_row.addWidget(self.generate_btn)
        btn_row.addStretch()
        gen_layout.addLayout(btn_row)

        self.paths_label = QLabel("")
        self.paths_label.setAlignment(Qt.AlignCenter)
        self.paths_label.setStyleSheet("color: #7f8c8d; font-size: 11px; border: none;")
        gen_layout.addWidget(self.paths_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximum(100)
        gen_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #2c3e50; font-size: 11px; border: none;")
        self.status_label.setVisible(False)
        gen_layout.addWidget(self.status_label)

        self.content_layout.addWidget(gen_frame)
        self.content_layout.addStretch()

        main_layout.addWidget(scroll)

    # ------------------------------------------------------------------
    # Populate from data
    # ------------------------------------------------------------------

    def populate(self, processed_data):
        """Omple el panel amb les dades processades."""
        if not processed_data:
            return

        samples_grouped = processed_data.get("samples_grouped", {})
        summary = processed_data.get("summary", {})
        method = processed_data.get("method", "COLUMN")
        seq_name = processed_data.get("seq_name", "")
        seq_path = processed_data.get("seq_path", "")
        is_bp = method.upper() == "BP"

        # Marcar seqüència populada (evitar re-populate a showEvent)
        self._populated_seq = seq_path or seq_name
        self._auto_generated = False

        # Separate regular (samples), light (blancs/controls), and KHP
        regular = {}  # mostres reals
        light = {}    # blancs / controls (analysis light O sample_type BLANK/CONTROL)
        khp = {}      # KHP standards
        for name, data in samples_grouped.items():
            if data.get("analysis_type") == "khp":
                khp[name] = data
            elif (data.get("analysis_type") == "light"
                  or data.get("sample_type") in ("BLANK", "CONTROL")):
                light[name] = data
            else:
                regular[name] = data

        # Count injections (replicas)
        n_injections = sum(
            len(d.get("replicas", {})) for d in samples_grouped.values()
        )

        # --- HEADER COMPACTE ---
        self._populate_compact_header(
            seq_name, method, summary, regular, light, khp, n_injections, is_bp,
            processed_data,
        )

        # --- GRÀFICS ---
        self._chart_regular = regular
        self._chart_light = light
        self._chart_khp = khp
        self._chart_is_bp = is_bp
        if HAS_MATPLOTLIB:
            try:
                self._plot_timeout_chart(processed_data, is_bp)
            except Exception as e:
                logger.error(f"Error plotting timeout chart: {e}")
            self._build_sample_checkboxes(regular, light, khp)
            self._redraw_charts()

        # --- CONSOLIDACIÓ BP ---
        self._current_method = method
        self._current_seq_path = seq_path
        self._bp_resolved = None

        if method.upper() == "COLUMN" and seq_path:
            # Llançar cerca BP en background — auto-generate al callback
            self._current_sample_names = [
                name for name, d in regular.items()
                if d.get("analysis_type") != "khp"
            ]
            self.bp_group.setVisible(True)
            self.bp_status_label.setText("Cercant BP...")
            self.bp_table.setRowCount(0)
            self._launch_bp_discovery(seq_path, self._current_sample_names)
        else:
            self.bp_group.setVisible(False)
            # BP o mode sense consolidació: auto-generar directament
            self._auto_generate()

        # --- PATHS ---
        if seq_path:
            self.paths_label.setText(
                f"Excels individuals → {seq_path}/RESULTATS/ &nbsp;&nbsp;|&nbsp;&nbsp; "
                f"SUMMARY.xlsx → {seq_path}/CHECK/"
            )

    def _populate_compact_header(self, seq_name, method, summary, regular,
                                 light, khp, n_injections, is_bp, processed_data):
        """Header compacte de 2 línies amb tota la info."""
        n_blancs = sum(1 for d in light.values() if d.get("sample_type") == "BLANK")
        n_controls = sum(1 for d in light.values() if d.get("sample_type") == "CONTROL")
        n_invalid = sum(1 for d in regular.values() if d.get("sample_valid") is False)
        n_errors = summary.get("with_anomalies", 0)

        # Timeout verdict
        zone_totals = {}
        for sample in processed_data.get("samples", []):
            ti = sample.get("timeout_info") or {}
            for zone, count in (ti.get("zone_summary") or {}).items():
                zone_totals[zone] = zone_totals.get(zone, 0) + count
        critical_zone = "BP_PEAK" if is_bp else "HS"
        critical_count = zone_totals.get(critical_zone, 0)
        if critical_count == 0:
            to_verdict = f"<span style='color:{COLOR_SUCCESS}'>TO {critical_zone} lliure</span>"
        else:
            to_verdict = (f"<span style='color:{COLOR_ERROR}'>"
                          f"TO {critical_zone} afectat ({critical_count})</span>")

        line1 = (
            f"<b>{seq_name}</b> &nbsp;&middot;&nbsp; {method}"
            f" &nbsp;&middot;&nbsp; {n_injections} inj"
        )
        parts2 = [f"<b>{len(regular)}</b> mostres"]
        if n_blancs:
            parts2.append(f"{n_blancs} blancs")
        if n_controls:
            parts2.append(f"{n_controls} controls")
        if khp:
            parts2.append(f"{len(khp)} KHP")
        if n_invalid:
            parts2.append(f"<span style='color:{COLOR_ERROR}'>{n_invalid} no vàlides</span>")
        if n_errors:
            parts2.append(f"<span style='color:{COLOR_WARNING}'>{n_errors} anomalies</span>")
        parts2.append(to_verdict)
        line2 = " &middot; ".join(parts2)

        self.header_label.setText(f"{line1}<br>{line2}")


    # ------------------------------------------------------------------
    # BP Consolidation
    # ------------------------------------------------------------------

    def _launch_bp_discovery(self, seq_path, sample_names):
        """Llança BPDiscoveryWorker per cercar dades BP."""
        if self._bp_worker and self._bp_worker.isRunning():
            self._bp_worker.wait(2000)

        data_folder = str(Path(seq_path).parent)
        self._bp_worker = BPDiscoveryWorker(seq_path, sample_names, data_folder)
        self._bp_worker.finished.connect(self._on_bp_discovery_finished)
        self._bp_worker.error.connect(self._on_bp_discovery_error)
        self._bp_worker.start()

    def _on_bp_discovery_finished(self, result):
        """Gestiona el resultat de la cerca BP."""
        self._bp_resolved = result
        self._bp_available = result.get("available_bps", [])

        # Omplir dropdown
        self.bp_combo.blockSignals(True)
        self.bp_combo.clear()
        primary = result.get("primary_bp")
        selected_idx = 0

        if not self._bp_available:
            self.bp_combo.addItem("Cap BP trobada", None)
        else:
            for i, bp in enumerate(self._bp_available):
                bp_name = bp.get("name", "?")
                self.bp_combo.addItem(bp_name, bp.get("path"))
                if primary and bp.get("path") == primary.get("path"):
                    selected_idx = i

        self.bp_combo.setCurrentIndex(selected_idx)
        self.bp_combo.blockSignals(False)

        # Omplir taula
        self._populate_bp_table(result)

        # Auto-generar resultats ara que tenim BP resolt
        self._auto_generate()

    def _on_bp_discovery_error(self, error_msg):
        """Error durant la cerca BP."""
        logger.error(f"Error BP discovery: {error_msg}")
        self.bp_status_label.setText(f"Error: {error_msg}")
        self.bp_group.setVisible(False)

    def _populate_bp_table(self, bp_result):
        """Omple la taula de mostres BP amb el resultat de la cerca."""
        samples = bp_result.get("samples", {})
        n_linked = sum(1 for s in samples.values() if s.get("bp_data"))
        n_total = len(samples)

        self.bp_status_label.setText(f"({n_linked}/{n_total} mostres vinculades)")

        self.bp_table.setRowCount(n_total)

        # Ordenar: vinculades primer, després sense match
        sorted_names = sorted(
            samples.keys(),
            key=lambda n: (0 if samples[n].get("bp_data") else 1, n)
        )

        for row, name in enumerate(sorted_names):
            sdata = samples[name]
            bp_data = sdata.get("bp_data")

            # Col 0: Nom mostra
            item_name = QTableWidgetItem(name)
            self.bp_table.setItem(row, 0, item_name)

            if bp_data:
                # Col 1: Estat (✔ + font)
                source = sdata.get("source", "")
                source_tag = " *" if source == "name_search" else ""
                item_status = QTableWidgetItem(f"✔{source_tag}")
                item_status.setForeground(QColor(COLOR_SUCCESS))
                item_status.setTextAlignment(Qt.AlignCenter)
                if source == "name_search":
                    item_status.setToolTip(
                        f"Trobat per nom a {bp_data.get('seq_name', '?')}"
                    )
                self.bp_table.setItem(row, 1, item_status)

                # Col 2: Rèplica
                replica = bp_data.get("replica", "?")
                item_rep = QTableWidgetItem(f"R{replica}")
                item_rep.setTextAlignment(Qt.AlignCenter)
                self.bp_table.setItem(row, 2, item_rep)

                # Col 3: ppm
                ppm = bp_data.get("concentration_ppm")
                ppm_text = f"{ppm:.2f}" if ppm else "—"
                item_ppm = QTableWidgetItem(ppm_text)
                item_ppm.setTextAlignment(Qt.AlignCenter)
                self.bp_table.setItem(row, 3, item_ppm)

                # Col 4: SNR
                snr = bp_data.get("snr_direct")
                snr_text = f"{snr:.0f}" if snr else "—"
                item_snr = QTableWidgetItem(snr_text)
                item_snr.setTextAlignment(Qt.AlignCenter)
                self.bp_table.setItem(row, 4, item_snr)
            else:
                # Sense match — fila gris
                item_status = QTableWidgetItem("✘")
                item_status.setForeground(QColor("#bdc3c7"))
                item_status.setTextAlignment(Qt.AlignCenter)
                self.bp_table.setItem(row, 1, item_status)

                for col in range(2, 5):
                    item = QTableWidgetItem("—")
                    item.setForeground(QColor("#bdc3c7"))
                    item.setTextAlignment(Qt.AlignCenter)
                    self.bp_table.setItem(row, col, item)

                # Fons gris per la fila sencera
                for col in range(5):
                    it = self.bp_table.item(row, col)
                    if it:
                        it.setBackground(QColor("#f5f5f5"))

        # Info label
        n_missing = n_total - n_linked
        if n_missing > 0:
            self.bp_info_label.setText(
                f"{n_missing} mostr{'a' if n_missing == 1 else 'es'} sense dades BP"
            )
        else:
            self.bp_info_label.setText("Totes les mostres tenen dades BP vinculades")

    def _on_bp_combo_changed(self, index):
        """Quan l'usuari canvia la BP al dropdown, relança la cerca."""
        if index < 0 or not self._bp_available:
            return

        bp_path = self.bp_combo.currentData()
        if not bp_path:
            return

        # Relançar cerca amb la BP seleccionada com a primària
        self.bp_status_label.setText("Actualitzant...")

        if self._bp_worker and self._bp_worker.isRunning():
            self._bp_worker.wait(2000)

        # Carregar dades per cada mostra des de la BP seleccionada
        self._bp_worker = _BPReloadWorker(
            bp_path, self._current_sample_names,
            self._bp_available, self._current_seq_path
        )
        self._bp_worker.finished.connect(self._on_bp_reload_finished)
        self._bp_worker.error.connect(self._on_bp_discovery_error)
        self._bp_worker.start()

    def _on_bp_reload_finished(self, result):
        """Gestiona el resultat de la recàrrega BP (canvi de dropdown)."""
        self._bp_resolved = result
        self._populate_bp_table(result)

    def _on_generate_cal_report(self):
        """Genera informe PDF de la calibració activa."""
        try:
            from hpsec_reports import generate_calibration_report
            from hpsec_calibrate import get_active_global_calibration

            cal = get_active_global_calibration()
            if not cal:
                QMessageBox.warning(self, "Avís", "No hi ha calibració activa.")
                return

            if not cal.get('regression_data'):
                QMessageBox.information(
                    self, "Info",
                    "La calibració activa no té dades de regressió emmagatzemades.\n"
                    "Les calibracions aplicades abans d'aquesta actualització no inclouen\n"
                    "les dades de regressió necessàries per l'informe complet."
                )
                return

            pdf_path = generate_calibration_report(cal)
            if pdf_path and os.path.exists(pdf_path):
                QMessageBox.information(
                    self, "Informe generat",
                    f"Informe de calibració generat:\n{pdf_path}"
                )
                # Obrir el PDF
                import subprocess
                try:
                    os.startfile(pdf_path)
                except AttributeError:
                    subprocess.Popen(['xdg-open', pdf_path])
            else:
                QMessageBox.warning(self, "Error", "No s'ha pogut generar l'informe.")
        except Exception as e:
            logger.error(f"Error generant informe calibració: {e}")
            QMessageBox.critical(self, "Error", f"Error generant informe:\n{e}")

    # ------------------------------------------------------------------
    # Sample checkboxes
    # ------------------------------------------------------------------

    def _build_sample_checkboxes(self, regular, light, khp):
        """Crea checkboxes per cada mostra dins el grid."""
        # Netejar grid anterior
        for cb, _n, _c in self._sample_checkboxes:
            self.samples_check_grid.removeWidget(cb)
            cb.deleteLater()
        self._sample_checkboxes = []

        all_items = []
        for name in sorted(regular.keys()):
            all_items.append((name, "sample", regular[name]))
        for name in sorted(light.keys()):
            all_items.append((name, "light", light[name]))
        for name in sorted(khp.keys()):
            all_items.append((name, "khp", khp[name]))

        n_cols = max(4, min(6, len(all_items) // 3 + 1))
        for idx, (name, cat, _data) in enumerate(all_items):
            cb = QCheckBox(name)
            cb.setChecked(cat == "sample")  # Mostres checked, resta no
            cb.setStyleSheet(
                "QCheckBox { font-size: 10px; }"
                + ("" if cat == "sample"
                   else " QCheckBox { color: #888; }")
            )
            if cat == "light":
                cb.setToolTip("Blanc / Control")
            elif cat == "khp":
                cb.setToolTip("KHP (calibració)")
            cb.stateChanged.connect(self._on_sample_selection_changed)
            row, col = divmod(idx, n_cols)
            self.samples_check_grid.addWidget(cb, row, col)
            self._sample_checkboxes.append((cb, name, cat))

    def _set_all_checks(self, state):
        for cb, _n, _c in self._sample_checkboxes:
            cb.blockSignals(True)
            cb.setChecked(state)
            cb.blockSignals(False)
        self._redraw_charts()

    def _check_only_samples(self):
        for cb, _n, cat in self._sample_checkboxes:
            cb.blockSignals(True)
            cb.setChecked(cat == "sample")
            cb.blockSignals(False)
        self._redraw_charts()

    def _get_checked_samples(self):
        """Retorna dict {name: data} de mostres seleccionades."""
        all_data = {}
        all_data.update(self._chart_regular)
        all_data.update(self._chart_light)
        all_data.update(getattr(self, '_chart_khp', {}))
        checked = {}
        for cb, name, _cat in self._sample_checkboxes:
            if cb.isChecked() and name in all_data:
                checked[name] = all_data[name]
        return checked

    def _on_sample_selection_changed(self, _state=None):
        self._redraw_charts()

    def _redraw_charts(self):
        """Redibuixa els 4 gràfics amb les mostres seleccionades."""
        if not HAS_MATPLOTLIB:
            return
        checked = self._get_checked_samples()
        reg = {k: v for k, v in checked.items()
               if v.get("analysis_type") not in ("light",)
               and v.get("sample_type") not in ("BLANK", "CONTROL")}
        light = {k: v for k, v in checked.items()
                 if v.get("analysis_type") == "light"
                 or v.get("sample_type") in ("BLANK", "CONTROL")}
        is_bp = getattr(self, '_chart_is_bp', False)
        try:
            self._plot_doc_chart(reg, light, is_bp)
            self._plot_dad_chart(reg, light)
            self._plot_doc_overlay(reg, light, is_bp)
            self._plot_dad_overlay(reg, light)
        except Exception as e:
            logger.error(f"Error redrawing charts: {e}")

    # ------------------------------------------------------------------
    # Timeout timeline
    # ------------------------------------------------------------------

    def _plot_timeout_chart(self, processed_data, is_bp):
        """Diagrama fraccions (eix temps) amb comptador timeouts per zona."""
        self.timeout_figure.clear()
        ax = self.timeout_figure.add_subplot(111)

        zone_totals = {}
        for sample in processed_data.get("samples", []):
            ti = sample.get("timeout_info") or {}
            for zone, count in (ti.get("zone_summary") or {}).items():
                zone_totals[zone] = zone_totals.get(zone, 0) + count

        if is_bp:
            zones = [
                ("BP_PEAK", 0, 8, "#E74C3C"),
                ("BP_TAIL", 8, 20, "#F39C12"),
                ("POST_RUN", 20, 70, "#95a5a6"),
            ]
        else:
            zones = [
                ("RUN_START", 0, 10.8, "#95a5a6"),
                ("BioP", 10.8, 18, FRACTION_COLORS["BioP"]),
                ("HS", 18, 23, FRACTION_COLORS["HS"]),
                ("BB", 23, 30, FRACTION_COLORS["BB"]),
                ("SB", 30, 37, FRACTION_COLORS["SB"]),
                ("LMW", 37, 48, FRACTION_COLORS["LMW"]),
                ("POST_RUN", 48, 70, "#95a5a6"),
            ]

        for zone_name, t0, t1, color in zones:
            count = zone_totals.get(zone_name, 0)
            alpha = 0.8 if count > 0 else 0.3
            ax.barh(0, t1 - t0, left=t0, height=0.6, color=color,
                    alpha=alpha, edgecolor='white', linewidth=0.5)
            mid = (t0 + t1) / 2
            label = f"{zone_name}\n{count}" if count > 0 else zone_name
            fw = 'bold' if count > 0 else 'normal'
            fc = '#c0392b' if count > 0 else '#555'
            ax.text(mid, 0, label, ha='center', va='center',
                    fontsize=6, fontweight=fw, color=fc)

        ax.set_xlim(0, 70)
        ax.set_yticks([])
        ax.set_xlabel("Temps (min)", fontsize=7)
        ax.tick_params(axis='x', labelsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        self.timeout_figure.tight_layout(pad=0.3)
        self.timeout_canvas.draw()

    # ------------------------------------------------------------------
    # Charts
    # ------------------------------------------------------------------

    def _plot_doc_chart(self, regular, light, is_bp):
        """Gràfic DOC: barres stacked per fraccions (COLUMN) o simples (BP)."""
        self.doc_figure.clear()
        ax = self.doc_figure.add_subplot(111)

        names = []
        fractions_data = {f: [] for f in FRACTION_ORDER}
        ppm_values = []

        # Regular samples
        for name in sorted(regular.keys()):
            data = regular[name]
            sel = (data.get("selected") or {}).get("doc", "1")
            rep = (data.get("replicas") or {}).get(sel, {})
            areas = (rep.get("areas") or {}).get("DOC", {})
            quant = data.get("quantification") or {}

            names.append(name)
            for frac in FRACTION_ORDER:
                fractions_data[frac].append(areas.get(frac, 0))
            ppm_values.append(quant.get("concentration_ppm") or 0)

        # Light samples (simple area_total, grey)
        light_start = len(names)
        for name in sorted(light.keys()):
            data = light[name]
            sel = (data.get("selected") or {}).get("doc", "1")
            rep = (data.get("replicas") or {}).get(sel, {})
            area = rep.get("area_total", 0)

            names.append(name)
            for frac in FRACTION_ORDER:
                fractions_data[frac].append(0)
            ppm_values.append(0)
            # Store area for light bar overlay
            fractions_data["BioP"][-1] = area  # Use BioP slot for total area

        if not names:
            ax.text(0.5, 0.5, "Sense dades", ha='center', va='center', transform=ax.transAxes)
            self.doc_canvas.draw()
            return

        x = np.arange(len(names))
        bar_width = 0.7

        if is_bp:
            # BP mode: simple bars (total area)
            totals = [sum(fractions_data[f][i] for f in FRACTION_ORDER) for i in range(len(names))]
            colors = ['#95a5a6' if i >= light_start else '#3498DB' for i in range(len(names))]
            ax.bar(x, totals, bar_width, color=colors, edgecolor='white', linewidth=0.5)
        else:
            # COLUMN mode: stacked fractions
            bottom = np.zeros(len(names))
            for frac in FRACTION_ORDER:
                values = np.array(fractions_data[frac], dtype=float)
                colors = []
                for i in range(len(names)):
                    if i >= light_start:
                        colors.append('#B0B0B0')  # Grey for light
                    else:
                        colors.append(FRACTION_COLORS[frac])
                ax.bar(x, values, bar_width, bottom=bottom, color=colors,
                       edgecolor='white', linewidth=0.3, label=frac if light_start > 0 or True else None)
                bottom += values

            # Legend (only fraction labels, no duplicates)
            handles = [
                ax.bar(0, 0, color=FRACTION_COLORS[f], label=f)[0]
                for f in FRACTION_ORDER
            ]
            ax.legend(handles=handles, labels=FRACTION_ORDER,
                      loc='upper right', fontsize=8, framealpha=0.8, ncol=len(FRACTION_ORDER))

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel("Àrea DOC", fontsize=9)
        ax.set_title("DOC per mostra (fraccions)", fontsize=10, fontweight='bold')
        ax.tick_params(axis='y', labelsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        self.doc_figure.tight_layout()
        self.doc_canvas.draw()

    def _plot_dad_chart(self, regular, light):
        """Gràfic A254 per mostra (barres simples)."""
        self.dad_figure.clear()
        ax = self.dad_figure.add_subplot(111)

        names = []
        areas_254 = []
        colors = []

        # Regular
        for name in sorted(regular.keys()):
            data = regular[name]
            selected = data.get("selected") or {}
            sel = selected.get("dad", selected.get("doc", "1"))
            rep = (data.get("replicas") or {}).get(sel, {})
            a254 = (rep.get("areas") or {}).get("A254", {}).get("total", 0)
            names.append(name)
            areas_254.append(a254)
            colors.append('#E74C3C')

        # Light
        for name in sorted(light.keys()):
            data = light[name]
            sel = (data.get("selected") or {}).get("doc", "1")
            rep = (data.get("replicas") or {}).get(sel, {})
            a254 = rep.get("area_254", 0)
            names.append(name)
            areas_254.append(a254)
            colors.append('#B0B0B0')

        if not names:
            ax.text(0.5, 0.5, "Sense dades", ha='center', va='center', transform=ax.transAxes)
            self.dad_canvas.draw()
            return

        x = np.arange(len(names))
        ax.bar(x, areas_254, 0.7, color=colors, edgecolor='white', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel("Àrea A254", fontsize=9)
        ax.set_title("A254 per mostra", fontsize=10, fontweight='bold')
        ax.tick_params(axis='y', labelsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        self.dad_figure.tight_layout()
        self.dad_canvas.draw()

    def _plot_doc_overlay(self, regular, light, is_bp):
        """Cromatogrames DOC superposats."""
        self.doc_overlay_figure.clear()
        ax = self.doc_overlay_figure.add_subplot(111)

        all_samples = {}
        all_samples.update(regular)
        all_samples.update(light)

        if not all_samples:
            ax.text(0.5, 0.5, "Sense dades", ha='center', va='center',
                    transform=ax.transAxes, color='#999')
            self.doc_overlay_canvas.draw()
            return

        import matplotlib.cm as cm
        n = len(all_samples)
        cmap = cm.get_cmap('tab20', max(n, 1))

        for i, (name, data) in enumerate(sorted(all_samples.items())):
            sel = (data.get("selected") or {}).get("doc", "1")
            rep = (data.get("replicas") or {}).get(sel, {})
            t = rep.get("t_doc")
            y = rep.get("y_doc_net")
            if t is not None and y is not None and len(t) > 0:
                ax.plot(t, y, label=name, linewidth=0.8, alpha=0.7, color=cmap(i))

        ax.set_xlabel("Temps (min)", fontsize=9)
        ax.set_ylabel("DOC (ppb)", fontsize=9)
        ax.set_title("Cromatogrames DOC superposats", fontsize=10, fontweight='bold')
        ax.tick_params(axis='both', labelsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            if len(handles) > 12:
                ax.legend(handles, labels, loc='upper left',
                          bbox_to_anchor=(1.01, 1), fontsize=6, framealpha=0.8,
                          ncol=1, handlelength=1.5, borderaxespad=0)
            else:
                ax.legend(loc='upper right', fontsize=7, framealpha=0.8)

        self.doc_overlay_figure.tight_layout()
        self.doc_overlay_canvas.draw()

    def _plot_dad_overlay(self, regular, light):
        """Cromatogrames A254 superposats."""
        self.dad_overlay_figure.clear()
        ax = self.dad_overlay_figure.add_subplot(111)

        all_samples = {}
        all_samples.update(regular)
        all_samples.update(light)

        if not all_samples:
            ax.text(0.5, 0.5, "Sense dades", ha='center', va='center',
                    transform=ax.transAxes, color='#999')
            self.dad_overlay_canvas.draw()
            return

        import matplotlib.cm as cm
        n = len(all_samples)
        cmap = cm.get_cmap('tab20', max(n, 1))

        for i, (name, data) in enumerate(sorted(all_samples.items())):
            selected = data.get("selected") or {}
            sel = selected.get("dad", selected.get("doc", "1"))
            rep = (data.get("replicas") or {}).get(sel, {})
            df_dad = rep.get("df_dad")
            if df_dad is None:
                continue
            try:
                if df_dad.empty:
                    continue
            except AttributeError:
                continue

            # Find time and 254nm columns
            t_col = None
            for c in df_dad.columns:
                if 'time' in str(c).lower():
                    t_col = c
                    break
            col_254 = None
            for c in df_dad.columns:
                if '254' in str(c):
                    col_254 = c
                    break

            if t_col is not None and col_254 is not None:
                ax.plot(df_dad[t_col], df_dad[col_254], label=name,
                        linewidth=0.8, alpha=0.7, color=cmap(i))

        ax.set_xlabel("Temps (min)", fontsize=9)
        ax.set_ylabel("A254 (mAU)", fontsize=9)
        ax.set_title("Cromatogrames A254 superposats", fontsize=10, fontweight='bold')
        ax.tick_params(axis='both', labelsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            if len(handles) > 12:
                ax.legend(handles, labels, loc='upper left',
                          bbox_to_anchor=(1.01, 1), fontsize=6, framealpha=0.8,
                          ncol=1, handlelength=1.5, borderaxespad=0)
            else:
                ax.legend(loc='upper right', fontsize=7, framealpha=0.8)

        self.dad_overlay_figure.tight_layout()
        self.dad_overlay_canvas.draw()

    def _save_charts(self, seq_path):
        """Guarda els 4 gràfics a SEQ/CHECK/plots/."""
        if not HAS_MATPLOTLIB or not seq_path:
            return
        try:
            plots_dir = Path(seq_path) / "CHECK" / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)

            for name, fig in [
                ("timeout_zones.png", self.timeout_figure),
                ("doc_areas.png", self.doc_figure),
                ("doc_overlay.png", self.doc_overlay_figure),
                ("dad_areas.png", self.dad_figure),
                ("dad_overlay.png", self.dad_overlay_figure),
            ]:
                fig.savefig(
                    str(plots_dir / name), dpi=150, bbox_inches='tight',
                    facecolor='#FAFAFA', edgecolor='none',
                )

            logger.info(f"Charts saved to {plots_dir}")
        except Exception as e:
            logger.error(f"Error saving charts: {e}")

    # ------------------------------------------------------------------
    # Generate results
    # ------------------------------------------------------------------

    def _auto_generate(self):
        """Auto-genera resultats si encara no s'ha fet per aquesta seqüència."""
        if self._auto_generated:
            return
        self._auto_generated = True
        self._run_generate(silent=True)

    def _run_generate(self, silent=False):
        """Genera Excels individuals + SUMMARY."""
        processed_data = self.main_window.processed_data
        if not processed_data:
            if not silent:
                QMessageBox.warning(self, "Avís", "No hi ha dades processades.")
            return

        samples_grouped = processed_data.get("samples_grouped", {})
        if not samples_grouped:
            if not silent:
                QMessageBox.warning(self, "Avís", "No hi ha mostres per exportar.")
            return

        seq_path = self.main_window.seq_path or processed_data.get("seq_path", "")
        if not seq_path:
            if not silent:
                QMessageBox.warning(self, "Avís", "No s'ha trobat el path de la seqüència.")
            return

        method = processed_data.get("method", "COLUMN")
        calibration_data = self.main_window.calibration_data

        self.generate_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setVisible(True)
        self.status_label.setText("Generant...")

        self.worker = GenerateWorker(
            samples_grouped, seq_path, calibration_data, method, None,
            bp_resolved=self._bp_resolved,
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_progress(self, pct, msg):
        self.progress_bar.setValue(pct)
        self.status_label.setText(msg)

    def _on_finished(self, results):
        self.generate_btn.setEnabled(True)
        self.generate_btn.setText("Regenerar Resultats")
        self.progress_bar.setVisible(False)

        errors = results.get("errors", [])
        excel_result = results.get("excel_files", {})
        summary_result = results.get("summary", {})
        n_exported = excel_result.get("n_exported", 0) if excel_result else 0

        if errors:
            self.status_label.setText(f"Completat amb {len(errors)} errors")
            self.status_label.setVisible(True)
            # Només mostrar diàleg si generació manual (no auto)
            if not self._auto_generated:
                QMessageBox.warning(self, "Avisos", f"Errors durant la generació:\n" + "\n".join(errors[:5]))
        else:
            self.status_label.setText(f"{n_exported} Excels + SUMMARY generats correctament")
            self.status_label.setVisible(True)

        # Guardar gràfics a CHECK/plots/
        self._save_charts(self._current_seq_path)

        # Escriure review_result.json
        self._write_review_result(results)

        self.review_completed.emit(results)

    def _on_error(self, error_msg):
        self.generate_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Error")
        QMessageBox.critical(self, "Error", f"Error durant la generació:\n{error_msg}")

    # ------------------------------------------------------------------
    # review_result.json
    # ------------------------------------------------------------------

    def _write_review_result(self, results):
        """Persisteix l'estat de la revisió a review_result.json."""
        try:
            from datetime import datetime

            seq_path = self._current_seq_path
            if not seq_path:
                return

            data_dir = Path(seq_path) / "CHECK" / "data"
            data_dir.mkdir(parents=True, exist_ok=True)

            excel_result = results.get("excel_files", {}) or {}
            n_exported = excel_result.get("n_exported", 0)
            n_skipped = excel_result.get("n_skipped", 0)

            # BP info
            bp_info = {}
            if self._bp_resolved:
                primary = self._bp_resolved.get("primary_bp")
                if primary:
                    bp_info["bp_seq_name"] = primary.get("name")
                    bp_info["bp_seq_path"] = primary.get("path")
                    bp_samples = self._bp_resolved.get("samples", {})
                    bp_info["n_linked"] = sum(
                        1 for s in bp_samples.values() if s.get("bp_data")
                    )
                    # mtime del analysis_result.json de la BP
                    bp_analysis = Path(primary["path"]) / "CHECK" / "data" / "analysis_result.json"
                    if bp_analysis.exists():
                        bp_info["bp_analysis_mtime"] = os.path.getmtime(str(bp_analysis))

            # Mostres descartades (sample_valid=False)
            processed_data = self.main_window.processed_data or {}
            samples_grouped = processed_data.get("samples_grouped", {})
            discarded = [
                name for name, d in samples_grouped.items()
                if d.get("sample_valid") is False
            ]

            review_data = {
                "success": not results.get("errors"),
                "timestamp": datetime.now().isoformat(),
                "version": "1.0",
                "seq_name": Path(seq_path).name,
                "method": self._current_method,
                "n_exported": n_exported,
                "n_skipped": n_skipped,
                "discarded_samples": discarded,
                "bp_info": bp_info,
                "summary_path": str(Path(seq_path) / "CHECK" / "SUMMARY.xlsx"),
            }

            review_path = data_dir / "review_result.json"
            with open(review_path, 'w', encoding='utf-8') as f:
                _json.dump(review_data, f, indent=2, ensure_ascii=False)

            logger.info(f"review_result.json escrit: {review_path}")
        except Exception as e:
            logger.error(f"Error escrivint review_result.json: {e}")

    # ------------------------------------------------------------------
    # Reset / showEvent
    # ------------------------------------------------------------------

    def reset(self):
        """Reseteja el panel."""
        self.header_label.setText("")
        if HAS_MATPLOTLIB:
            self.timeout_figure.clear()
            self.timeout_canvas.draw()
            self.doc_figure.clear()
            self.doc_canvas.draw()
            self.doc_overlay_figure.clear()
            self.doc_overlay_canvas.draw()
            self.dad_figure.clear()
            self.dad_canvas.draw()
            self.dad_overlay_figure.clear()
            self.dad_overlay_canvas.draw()
            # Netejar checkboxes
            for cb, _n, _c in self._sample_checkboxes:
                self.samples_check_grid.removeWidget(cb)
                cb.deleteLater()
            self._sample_checkboxes = []
        self._chart_regular = {}
        self._chart_light = {}
        self._chart_khp = {}
        self._chart_is_bp = False
        self.progress_bar.setVisible(False)
        self.status_label.setVisible(False)
        self.status_label.setText("")
        self.paths_label.setText("")
        self.generate_btn.setEnabled(True)
        self.generate_btn.setText("Generar Resultats")
        # BP state
        self._bp_resolved = None
        self._bp_available = []
        self.bp_group.setVisible(False)
        self.bp_table.setRowCount(0)
        self.bp_combo.clear()
        self.bp_status_label.setText("")
        self.bp_info_label.setText("")
        # Auto-generate state
        self._auto_generated = False
        self._populated_seq = ""

    def showEvent(self, event):
        """Quan es mostra el panel, omplir amb dades actuals."""
        super().showEvent(event)
        processed_data = self.main_window.processed_data
        if not processed_data or not processed_data.get("success"):
            return
        # Evitar re-populate si és la mateixa seqüència
        seq_id = processed_data.get("seq_path", "") or processed_data.get("seq_name", "")
        if seq_id and seq_id == self._populated_seq:
            return
        self.populate(processed_data)
