"""
HPSEC Suite - Export Panel (Pas 4: Exportar)
=============================================

Panel d'exportació de resultats amb principis FAIR.

Contingut:
- Resum pre-exportació: comptadors (N mostres, M excloses, K avisos)
- Opcions exportació: checkboxes (Excels individuals, SUMMARY, PDF report)
- Secció FAIR (placeholder): metadades traçabilitat
- Consolidació BP (COLUMN mode)
- Botó "Generar Resultats" + barra progrés
- Escriu review_result.json per al dashboard
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QScrollArea, QMessageBox, QProgressBar,
    QTableWidget, QTableWidgetItem, QComboBox, QHeaderView,
    QGroupBox, QCheckBox, QGridLayout,
)
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QFont, QColor

import json as _json
import logging
import os

from pathlib import Path

logger = logging.getLogger(__name__)

from hpsec_export import export_sequence, generate_summary_excel, DEFAULT_EXPORT_CONFIG
from gui.widgets.styles import (
    COLOR_SUCCESS, COLOR_WARNING, COLOR_ERROR,
    apply_panel_layout
)


class BPDiscoveryWorker(QThread):
    """Worker per cercar dades BP en background."""
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, seq_path, sample_names, data_folder):
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
    """Worker per rellegir dades BP des d'una BP diferent."""
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, bp_path, sample_names, available_bps, seq_path):
        super().__init__()
        self.bp_path = bp_path
        self.sample_names = sample_names
        self.available_bps = available_bps
        self.seq_path = seq_path

    def run(self):
        try:
            from hpsec_consolidate import find_bp_for_samples
            result = find_bp_for_samples(
                self.seq_path, self.sample_names,
                str(Path(self.seq_path).parent),
                preferred_bp_path=self.bp_path
            )
            result["available_bps"] = self.available_bps
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class GenerateWorker(QThread):
    """Worker per generar resultats en background."""
    progress = Signal(int, str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, samples_grouped, seq_path, calibration_data, mode, config,
                 bp_resolved=None, generate_pdf=False):
        super().__init__()
        self.samples_grouped = samples_grouped
        self.seq_path = seq_path
        self.calibration_data = calibration_data
        self.mode = mode
        self.config = config
        self.bp_resolved = bp_resolved
        self.generate_pdf = generate_pdf

    def run(self):
        try:
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
            self.progress.emit(85, "Generant SUMMARY.xlsx...")
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

            # PDF analysis report
            if self.generate_pdf:
                self.progress.emit(92, "Generant PDF anàlisi...")
                try:
                    from generate_analysis_report import generate_analysis_report
                    report_data = {
                        "samples_grouped": self.samples_grouped,
                        "method": self.mode,
                        "seq_path": self.seq_path,
                        "seq_name": Path(self.seq_path).name,
                        "success": True,
                    }
                    pdf_path = generate_analysis_report(
                        self.seq_path, analysis_data=report_data
                    )
                    results["pdf_report"] = pdf_path
                except Exception as e:
                    results["errors"].append(f"PDF report: {e}")

            self.progress.emit(100, "Completat")
            self.finished.emit(results)

        except Exception as e:
            self.error.emit(str(e))


class ExportPanel(QWidget):
    """Panel d'exportació de resultats (Pas 4: Exportar)."""

    export_completed = Signal(dict)

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.worker = None
        self._bp_worker = None
        self._bp_resolved = None
        self._bp_available = []
        self._current_method = "COLUMN"
        self._current_seq_path = ""
        self._current_sample_names = []
        self._auto_generated = False
        self._populated_seq = ""

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

        # === HEADER / SUMMARY ===
        self.header_label = QLabel()
        self.header_label.setTextFormat(Qt.RichText)
        self.header_label.setWordWrap(True)
        self.header_label.setStyleSheet(
            "font-size: 12px; color: #2c3e50; padding: 6px 10px;"
            "background: #fff; border: 1px solid #e0e0e0; border-radius: 6px;"
        )
        self.content_layout.addWidget(self.header_label)

        # === OPCIONS EXPORTACIÓ ===
        options_group = QGroupBox("Opcions d'Exportació")
        options_group.setStyleSheet("""
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
        options_layout = QVBoxLayout(options_group)
        options_layout.setContentsMargins(12, 8, 12, 8)

        self.individual_check = QCheckBox("Excels individuals (un per mostra)")
        self.individual_check.setChecked(True)
        self.individual_check.setToolTip(
            "Excel per mostra: ID + DOC + DAD + RESULTS"
        )
        options_layout.addWidget(self.individual_check)

        self.summary_check = QCheckBox("SUMMARY.xlsx (taula resum)")
        self.summary_check.setChecked(True)
        self.summary_check.setToolTip(
            "SUMMARY.xlsx a SEQ/CHECK/ amb una fila per mostra"
        )
        options_layout.addWidget(self.summary_check)

        self.pdf_check = QCheckBox("Informe PDF d'anàlisi")
        self.pdf_check.setChecked(False)
        self.pdf_check.setToolTip(
            "REPORT_Analysis_*.pdf amb cromatogrames i estadístiques"
        )
        options_layout.addWidget(self.pdf_check)

        self.content_layout.addWidget(options_group)

        # === SECCIÓ FAIR (placeholder) ===
        fair_group = QGroupBox("Traçabilitat (FAIR)")
        fair_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold; font-size: 11px; color: #2c3e50;
                border: 1px solid #d5dbdb; border-radius: 6px;
                margin-top: 8px; padding-top: 18px;
                background-color: #f8f9fa;
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 12px; padding: 0 4px;
            }
        """)
        fair_layout = QVBoxLayout(fair_group)
        fair_layout.setContentsMargins(12, 8, 12, 8)

        self.fair_info = QLabel()
        self.fair_info.setWordWrap(True)
        self.fair_info.setStyleSheet("color: #5D6D7E; font-size: 11px;")
        fair_layout.addWidget(self.fair_info)

        self.content_layout.addWidget(fair_group)

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

        self.bp_info_label = QLabel("")
        self.bp_info_label.setStyleSheet("color: #7f8c8d; font-size: 10px; font-style: italic;")
        bp_layout.addWidget(self.bp_info_label)

        self.bp_group.setVisible(False)
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

        self._populated_seq = seq_path or seq_name
        self._auto_generated = False

        # Separate categories
        regular = {}
        light = {}
        khp = {}
        for name, data in samples_grouped.items():
            if data.get("analysis_type") == "khp":
                khp[name] = data
            elif (data.get("analysis_type") == "light"
                  or data.get("sample_type") in ("BLANK", "CONTROL")):
                light[name] = data
            else:
                regular[name] = data

        n_injections = sum(
            len(d.get("replicas", {})) for d in samples_grouped.values()
        )

        # --- HEADER ---
        self._populate_header(
            seq_name, method, summary, regular, light, khp, n_injections, is_bp,
            processed_data,
        )

        # --- FAIR INFO ---
        self._update_fair_info(processed_data)

        # --- CONSOLIDACIÓ BP ---
        self._current_method = method
        self._current_seq_path = seq_path
        self._bp_resolved = None

        if method.upper() == "COLUMN" and seq_path:
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
            self._auto_generate()

        # --- PATHS ---
        if seq_path:
            self.paths_label.setText(
                f"Excels individuals \u2192 {seq_path}/RESULTATS/ &nbsp;&nbsp;|&nbsp;&nbsp; "
                f"SUMMARY.xlsx \u2192 {seq_path}/CHECK/"
            )

    def _populate_header(self, seq_name, method, summary, regular,
                         light, khp, n_injections, is_bp, processed_data):
        """Header compacte de 2 línies."""
        n_blancs = sum(1 for d in light.values() if d.get("sample_type") == "BLANK")
        n_controls = sum(1 for d in light.values() if d.get("sample_type") == "CONTROL")
        n_invalid = sum(1 for d in regular.values() if d.get("sample_valid") is False)
        n_errors = summary.get("with_anomalies", 0)

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

    def _update_fair_info(self, processed_data):
        """Actualitza la secció FAIR amb metadades de traçabilitat."""
        parts = []
        method = processed_data.get("method", "?")
        seq_name = processed_data.get("seq_name", "?")
        parts.append(f"Seqüència: <b>{seq_name}</b> ({method})")

        # Calibration info
        cal_data = self.main_window.calibration_data
        if cal_data and cal_data.get("success"):
            khp_source = cal_data.get("khp_source", "?")
            parts.append(f"Calibració: {khp_source}")

        # Config fingerprint
        try:
            from hpsec_config import Config
            cfg = Config()
            fp = cfg.compute_fingerprint()
            parts.append(f"Config fingerprint: <code>{fp}</code>")
        except Exception:
            pass

        # Calibration fingerprint
        try:
            from hpsec_calibrate import compute_calibration_fingerprint
            cal_fp = compute_calibration_fingerprint()
            parts.append(f"Calibració fingerprint: <code>{cal_fp}</code>")
        except Exception:
            pass

        # Suite version
        try:
            from hpsec_version import __version__
            parts.append(f"HPSEC Suite v{__version__}")
        except Exception:
            parts.append("HPSEC Suite")

        self.fair_info.setText("<br>".join(parts))

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

        self._populate_bp_table(result)
        self._auto_generate()

    def _on_bp_discovery_error(self, error_msg):
        """Error durant la cerca BP."""
        logger.error(f"Error BP discovery: {error_msg}")
        self.bp_status_label.setText(f"Error: {error_msg}")
        self.bp_group.setVisible(False)
        self._auto_generate()

    def _populate_bp_table(self, bp_result):
        """Omple la taula de mostres BP."""
        samples = bp_result.get("samples", {})
        n_linked = sum(1 for s in samples.values() if s.get("bp_data"))
        n_total = len(samples)

        self.bp_status_label.setText(f"({n_linked}/{n_total} mostres vinculades)")

        self.bp_table.setRowCount(n_total)

        sorted_names = sorted(
            samples.keys(),
            key=lambda n: (0 if samples[n].get("bp_data") else 1, n)
        )

        for row, name in enumerate(sorted_names):
            sdata = samples[name]
            bp_data = sdata.get("bp_data")

            item_name = QTableWidgetItem(name)
            self.bp_table.setItem(row, 0, item_name)

            if bp_data:
                source = sdata.get("source", "")
                source_tag = " *" if source == "name_search" else ""
                item_status = QTableWidgetItem(f"\u2714{source_tag}")
                item_status.setForeground(QColor(COLOR_SUCCESS))
                item_status.setTextAlignment(Qt.AlignCenter)
                if source == "name_search":
                    item_status.setToolTip(
                        f"Trobat per nom a {bp_data.get('seq_name', '?')}"
                    )
                self.bp_table.setItem(row, 1, item_status)

                replica = bp_data.get("replica", "?")
                item_rep = QTableWidgetItem(f"R{replica}")
                item_rep.setTextAlignment(Qt.AlignCenter)
                self.bp_table.setItem(row, 2, item_rep)

                ppm = bp_data.get("concentration_ppm")
                ppm_text = f"{ppm:.2f}" if ppm else "\u2014"
                item_ppm = QTableWidgetItem(ppm_text)
                item_ppm.setTextAlignment(Qt.AlignCenter)
                self.bp_table.setItem(row, 3, item_ppm)

                snr = bp_data.get("snr_direct")
                snr_text = f"{snr:.0f}" if snr else "\u2014"
                item_snr = QTableWidgetItem(snr_text)
                item_snr.setTextAlignment(Qt.AlignCenter)
                self.bp_table.setItem(row, 4, item_snr)
            else:
                item_status = QTableWidgetItem("\u2718")
                item_status.setForeground(QColor("#bdc3c7"))
                item_status.setTextAlignment(Qt.AlignCenter)
                self.bp_table.setItem(row, 1, item_status)

                for col in range(2, 5):
                    item = QTableWidgetItem("\u2014")
                    item.setForeground(QColor("#bdc3c7"))
                    item.setTextAlignment(Qt.AlignCenter)
                    self.bp_table.setItem(row, col, item)

                for col in range(5):
                    it = self.bp_table.item(row, col)
                    if it:
                        it.setBackground(QColor("#f5f5f5"))

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

        self.bp_status_label.setText("Actualitzant...")

        if self._bp_worker and self._bp_worker.isRunning():
            self._bp_worker.wait(2000)

        self._bp_worker = _BPReloadWorker(
            bp_path, self._current_sample_names,
            self._bp_available, self._current_seq_path
        )
        self._bp_worker.finished.connect(self._on_bp_reload_finished)
        self._bp_worker.error.connect(self._on_bp_discovery_error)
        self._bp_worker.start()

    def _on_bp_reload_finished(self, result):
        """Gestiona el resultat de la recàrrega BP."""
        self._bp_resolved = result
        self._populate_bp_table(result)

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

        if self.worker is not None:
            self.worker.wait()
        self.worker = GenerateWorker(
            samples_grouped, seq_path, calibration_data, method, None,
            bp_resolved=self._bp_resolved,
            generate_pdf=self.pdf_check.isChecked(),
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_progress(self, pct, msg):
        self.progress_bar.setValue(pct)
        self.status_label.setText(msg)

    def _on_finished(self, results):
        if self.worker is not None:
            self.worker.wait()
        self.generate_btn.setEnabled(True)
        self.generate_btn.setText("Regenerar Resultats")
        self.progress_bar.setVisible(False)

        errors = results.get("errors", [])
        excel_result = results.get("excel_files", {})
        n_exported = excel_result.get("n_exported", 0) if excel_result else 0

        if errors:
            self.status_label.setText(f"Completat amb {len(errors)} errors")
            self.status_label.setVisible(True)
            if not self._auto_generated:
                QMessageBox.warning(self, "Avisos",
                                    f"Errors durant la generació:\n" + "\n".join(errors[:5]))
        else:
            pdf_info = ""
            if results.get("pdf_report"):
                pdf_info = " + PDF"
            self.status_label.setText(f"{n_exported} Excels + SUMMARY{pdf_info} generats correctament")
            self.status_label.setVisible(True)

        # Save charts from analyze panel
        try:
            analyze_panel = getattr(self.main_window, '_wizard_analyze_panel', None)
            if analyze_panel and hasattr(analyze_panel, 'save_charts'):
                analyze_panel.save_charts(self._current_seq_path)
        except Exception as e:
            logger.error(f"Error saving charts: {e}")

        # Write review_result.json
        self._write_review_result(results)

        self.export_completed.emit(results)

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
                    bp_analysis = Path(primary["path"]) / "CHECK" / "data" / "analysis_result.json"
                    if bp_analysis.exists():
                        bp_info["bp_analysis_mtime"] = os.path.getmtime(str(bp_analysis))

            # Discarded samples
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
        self.fair_info.setText("")
        self.progress_bar.setVisible(False)
        self.status_label.setVisible(False)
        self.status_label.setText("")
        self.paths_label.setText("")
        self.generate_btn.setEnabled(True)
        self.generate_btn.setText("Generar Resultats")
        self._bp_resolved = None
        self._bp_available = []
        self.bp_group.setVisible(False)
        self.bp_table.setRowCount(0)
        self.bp_combo.clear()
        self.bp_status_label.setText("")
        self.bp_info_label.setText("")
        self._auto_generated = False
        self._populated_seq = ""
        self.individual_check.setChecked(True)
        self.summary_check.setChecked(True)
        self.pdf_check.setChecked(False)

    def showEvent(self, event):
        """Quan es mostra el panel, omplir amb dades actuals."""
        super().showEvent(event)
        processed_data = self.main_window.processed_data
        if not processed_data or not processed_data.get("success"):
            return
        seq_id = processed_data.get("seq_path", "") or processed_data.get("seq_name", "")
        if seq_id and seq_id == self._populated_seq:
            return
        self.populate(processed_data)
