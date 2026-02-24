"""
HPSEC Suite - Analyze Panel (Fase 3) — Taula Unificada DOC+DAD
================================================================

Panel per la fase 3: Anàlisi de mostres.
- Taula unificada (1 fila per mostra) amb DOC + DAD principals
- Panel de fraccions visible al seleccionar mostra
- Classificació d'anomalies per severitat (no falsos avisos)
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QComboBox,
    QFrame, QAbstractItemView, QProgressBar, QMessageBox, QDialog,
    QGroupBox, QGridLayout, QCheckBox, QScrollArea
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QBrush, QFont

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

from hpsec_analyze import analyze_sequence, save_analysis_result, load_analysis_result
from gui.widgets.styles import (
    COLOR_SUCCESS, COLOR_WARNING, COLOR_ERROR,
    apply_panel_layout, create_empty_state_widget
)
from .worker import AnalyzeWorker
from .dialogs import SampleDetailDialog
from ._constants import (
    CRITICAL_ANOMALIES, WARNING_ANOMALIES,
    DAD_WL_MAIN, SIGNAL_KEYS_MAIN,
)
from hpsec_warnings import (
    has_anomaly, get_anomaly_codes, classify_anomalies,
    ANOMALY_CATALOG,
)
from ._helpers import (
    configure_table_style, populate_signal_summary, populate_fractions_table
)


class AnalyzePanel(QWidget):
    """Panel d'anàlisi de mostres (Fase 3) — Taula unificada."""

    analyze_completed = Signal(dict)

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.samples_grouped = {}
        self.worker = None
        self._warnings_confirmed = False
        self._warnings_confirmed_by = ""
        self._selected_sample = None
        self._sample_row_map = {}       # P3: sample_name → row index (O(1) lookup)
        self._status_initialized = False  # B3: avoid redundant showEvent work

        self._setup_ui()

    # ------------------------------------------------------------------
    # UI Setup
    # ------------------------------------------------------------------

    def _setup_ui(self):
        """Configura la interfície — Taula unificada + panel fraccions."""
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        # Botó analitzar (amagat - l'acció es dispara des del wizard header)
        self.analyze_btn = QPushButton()
        self.analyze_btn.setVisible(False)
        self.analyze_btn.clicked.connect(self._run_analyze)

        # === SCROLL AREA per contenir tot el contingut ===
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setStyleSheet("QScrollArea { border: none; }")

        scroll_content = QWidget()
        layout = QVBoxLayout(scroll_content)
        apply_panel_layout(layout)

        # === INFO PANEL ===
        self.info_frame = QFrame()
        self.info_frame.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border: 1px solid #e9ecef;
                border-radius: 6px;
            }
        """)
        info_layout = QHBoxLayout(self.info_frame)
        info_layout.setContentsMargins(16, 12, 16, 12)
        info_layout.setSpacing(24)

        self.import_info = QLabel()
        self.import_info.setStyleSheet("border: none;")
        info_layout.addWidget(self.import_info)

        sep1 = QFrame()
        sep1.setFrameShape(QFrame.VLine)
        sep1.setStyleSheet("background-color: #dee2e6; border: none; max-width: 1px;")
        info_layout.addWidget(sep1)

        self.cal_info = QLabel()
        self.cal_info.setStyleSheet("border: none;")
        info_layout.addWidget(self.cal_info)

        info_layout.addStretch()

        self.status_indicator = QLabel()
        self.status_indicator.setStyleSheet("border: none;")
        info_layout.addWidget(self.status_indicator)

        layout.addWidget(self.info_frame)

        # Empty state
        self.empty_state = create_empty_state_widget(
            "🔬", "Preparant anàlisi...",
            "Carregant dades de la seqüència."
        )
        self.empty_state.setVisible(False)
        layout.addWidget(self.empty_state)

        # Status frame (error messages)
        self.status_frame = QFrame()
        self.status_frame.setVisible(False)
        status_layout = QVBoxLayout(self.status_frame)
        status_layout.setContentsMargins(12, 8, 12, 8)
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        status_layout.addWidget(self.status_label)
        layout.addWidget(self.status_frame)

        # === PROGRESS ===
        self.progress_frame = QFrame()
        self.progress_frame.setVisible(False)
        progress_layout = QVBoxLayout(self.progress_frame)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        self.progress_bar = QProgressBar()
        self.progress_label = QLabel("Preparant...")
        progress_layout.addWidget(self.progress_label)
        progress_layout.addWidget(self.progress_bar)
        layout.addWidget(self.progress_frame)

        # === RESULTS FRAME ===
        self.results_frame = QFrame()
        self.results_frame.setVisible(False)
        results_layout = QVBoxLayout(self.results_frame)
        results_layout.setContentsMargins(0, 0, 0, 0)
        results_layout.setSpacing(8)

        # === LEGEND BAR ===
        legend_frame = QFrame()
        legend_frame.setStyleSheet("QFrame { background-color: #f8f9fa; border-radius: 4px; }")
        legend_layout = QHBoxLayout(legend_frame)
        legend_layout.setContentsMargins(12, 6, 12, 6)
        legend = QLabel(
            "<span style='color:#27AE60'>●</span> OK &nbsp;"
            "<span style='color:#F39C12'>●</span> Warning &nbsp;"
            "<span style='color:#E74C3C'>●</span> Error &nbsp;&nbsp;|&nbsp;&nbsp;"
            "<span style='color:#666; font-size:10px;'>Doble-clic = detall complet</span>"
        )
        legend.setStyleSheet("color: #666;")
        legend_layout.addWidget(legend)
        legend_layout.addStretch()

        # Botó Generar Report PDF (amagat per SEQ_CAL)
        self.report_btn = QPushButton("Generar Report PDF")
        self.report_btn.setStyleSheet(
            "QPushButton { background-color: #2E86AB; color: white; "
            "font-weight: bold; padding: 4px 14px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #1A5276; }"
        )
        self.report_btn.clicked.connect(self._generate_report)
        legend_layout.addWidget(self.report_btn)

        results_layout.addWidget(legend_frame)

        # === UNIFIED TABLE ===
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(14)
        self.results_table.setHorizontalHeaderLabels([
            "Mostra", "Sel DOC", "Sel DAD", "A_DOC", "ppm",
            "A_UIB", "ppm_U", "SNR", "A_254", "SNR_254",
            "R²_DOC", "R²_DAD", "HCI", "Estat"
        ])
        configure_table_style(self.results_table)
        self._configure_unified_columns()
        results_layout.addWidget(self.results_table)

        # Connect table signals — clic obre detall directament
        self.results_table.doubleClicked.connect(self._on_table_double_click)
        self.results_table.setToolTip("Doble-clic per detall complet")

        # === STATS BAR ===
        self.stats_frame = QFrame()
        self.stats_frame.setStyleSheet("background-color: #f8f9fa; border-radius: 4px; padding: 8px;")
        stats_layout = QHBoxLayout(self.stats_frame)
        stats_layout.setContentsMargins(12, 8, 12, 8)

        self.stats_label = QLabel()
        self.stats_label.setFont(QFont("Segoe UI", 10))
        stats_layout.addWidget(self.stats_label)
        stats_layout.addStretch()

        results_layout.addWidget(self.stats_frame)

        layout.addWidget(self.results_frame, 1)

        # Completar scroll area
        scroll_area.setWidget(scroll_content)
        outer_layout.addWidget(scroll_area, 1)

    def _configure_unified_columns(self):
        """Configura columnes de la taula unificada."""
        header = self.results_table.horizontalHeader()
        for i in range(self.results_table.columnCount()):
            if i == 13:  # Estat — much wider
                header.setSectionResizeMode(i, QHeaderView.Stretch)
            else:
                header.setSectionResizeMode(i, QHeaderView.ResizeToContents)

    # ------------------------------------------------------------------
    # Show / Reset / Check existing
    # ------------------------------------------------------------------

    def showEvent(self, event):
        super().showEvent(event)
        if not self._status_initialized or not self.samples_grouped:
            self._check_existing_analysis()
            self._update_status()
            self._status_initialized = True

    def reset(self):
        """Reinicia el panel al seu estat inicial."""
        self.samples_grouped = {}
        self.worker = None
        self._warnings_confirmed = False
        self._warnings_confirmed_by = ""
        self._selected_sample = None
        self._sample_row_map = {}
        self._status_initialized = False

        self.results_table.setRowCount(0)

        self.empty_state.setVisible(True)
        self.info_frame.setVisible(False)
        self.status_frame.setVisible(False)
        self.progress_frame.setVisible(False)
        self.progress_bar.setValue(0)
        self.results_frame.setVisible(False)
        self.analyze_btn.setEnabled(True)
        self.stats_label.setText("")

    def _check_existing_analysis(self):
        """Comprova si existeix anàlisi prèvia i la carrega automàticament."""
        seq_path = self.main_window.seq_path
        if not seq_path:
            return
        if self.samples_grouped:
            return
        try:
            existing_analysis = load_analysis_result(seq_path)
            if existing_analysis and existing_analysis.get("success"):
                self._load_existing_analysis(existing_analysis)
        except Exception as e:
            logger.warning(f"Error comprovant anàlisi existent: {e}")

    def _load_existing_analysis(self, result):
        """Carrega una anàlisi existent."""
        self.samples_grouped = (result.get("samples_grouped")
                                or result.get("samples_analyzed", {}))
        if self.samples_grouped:
            self.main_window.processed_data = result  # B1: needed for method/seq_path
            self._populate_table()
            self.empty_state.setVisible(False)
            self.info_frame.setVisible(False)
            self.status_frame.setVisible(False)
            self.results_frame.setVisible(True)
            self.main_window.set_status("Anàlisi carregada des de fitxer existent", 3000)
            self.analyze_completed.emit(result)

    def _update_status(self):
        """Actualitza l'indicador d'estat amb format professional."""
        imported_data = self.main_window.imported_data
        calibration_data = self.main_window.calibration_data

        if not imported_data:
            self.info_frame.setVisible(False)
            self.status_frame.setVisible(False)
            self.empty_state.setVisible(True)
            self.analyze_btn.setEnabled(False)
            return

        self.empty_state.setVisible(False)
        # Si ja hi ha resultats carregats, amagar info_frame (redundant)
        self.info_frame.setVisible(not bool(self.samples_grouped))
        self.status_frame.setVisible(False)

        # Use analyzed sample count if available, else imported injections
        if self.samples_grouped:
            n_items = len(self.samples_grouped)
            item_label = "mostres"
        else:
            samples = imported_data.get("samples", {})
            n_items = len(samples)
            item_label = "injeccions"
        method = imported_data.get("method", "-")
        data_mode = imported_data.get("data_mode", "-")

        self.import_info.setText(
            f"<span style='color: #6c757d; font-size: 10px;'>DADES</span><br>"
            f"<b style='font-size: 13px;'>{n_items}</b> <span style='color: #495057;'>{item_label}</span><br>"
            f"<span style='color: #6c757d; font-size: 10px;'>{method} / {data_mode}</span>"
        )

        if calibration_data and calibration_data.get("success"):
            # Get rf_mass_cal + intercept GLOBAL (what quantify_sample actually uses)
            rf_mass_global = None
            intercept_global = 0
            try:
                from hpsec_calibrate import get_rf_mass_cal, get_calibration_intercept
                seq_method = imported_data.get("method", "COLUMN")
                mode_key = "bp" if seq_method.upper() == "BP" else "column"
                rf_mass_global = get_rf_mass_cal(signal='direct', mode=mode_key)
                intercept_global = get_calibration_intercept(signal='direct', mode=mode_key) or 0
            except Exception:
                pass

            # Display: prioritzar calibració global (el que realment usa quantify_sample)
            if rf_mass_global and rf_mass_global > 0:
                rf_display = rf_mass_global
                intercept_display = intercept_global
                cal_note = "Global"
            else:
                rf_mass_local = calibration_data.get("rf_mass", 0)
                rf_direct = calibration_data.get("rf_direct", 0) or calibration_data.get("rf", 0)
                rf_display = rf_mass_local if rf_mass_local > 0 else rf_direct
                intercept_display = 0
                cal_note = "Local"

            # Build regression line text
            if intercept_display and abs(intercept_display) > 0.01:
                recta_str = f"RF=<b>{rf_display:.1f}</b> · b=<b>{intercept_display:.1f}</b>"
            else:
                recta_str = f"RF=<b>{rf_display:.1f}</b> · origen"

            self.cal_info.setText(
                f"<span style='color: #6c757d; font-size: 10px;'>CALIBRACIÓ</span><br>"
                f"<span style='color: #27AE60;'>✓</span> <b style='font-size: 13px;'>{cal_note}</b><br>"
                f"<span style='color: #6c757d; font-size: 10px;'>{recta_str}</span>"
            )

            # Tooltip: detalls complets per si vol aprofundir
            khp_conc = calibration_data.get("khp_conc", 0)
            shift = calibration_data.get("shift_direct", 0) or calibration_data.get("shift", 0)
            shift_sec = shift * 60 if shift else 0
            khp_source = calibration_data.get("khp_source", "LOCAL")
            rf_global_str = f"{rf_mass_global:.2f}" if rf_mass_global else "N/A"
            self.cal_info.setToolTip(
                f"Font: {khp_source}\n"
                f"Quantificació: {cal_note}\n"
                f"Recta: ppm = (A - {intercept_display:.1f}) × 1000 / (RF × V)\n"
                f"RF_mass_cal (global): {rf_global_str}\n"
                f"Intercept (global): {intercept_global:.2f}\n"
                f"KHP SEQ: {khp_conc:g}ppm, shift={shift_sec:.1f}s"
            )
        else:
            self.cal_info.setText(
                f"<span style='color: #6c757d; font-size: 10px;'>CALIBRACIÓ</span><br>"
                f"<span style='color: #E67E22;'>⚠</span> <span style='color: #856404;'>No disponible</span><br>"
                f"<span style='color: #6c757d; font-size: 10px;'>S'usaran valors per defecte</span>"
            )
            self.cal_info.setToolTip("No hi ha calibració disponible")

        if self.samples_grouped:
            n = len(self.samples_grouped)
            self.status_indicator.setText(
                f"<span style='background-color: #d4edda; color: #155724; "
                f"padding: 4px 12px; border-radius: 12px; font-size: 11px;'>"
                f"Anàlisi completada ({n} mostres)</span>"
            )
        else:
            self.status_indicator.setText(
                f"<span style='background-color: #d4edda; color: #155724; "
                f"padding: 4px 12px; border-radius: 12px; font-size: 11px;'>"
                f"Llest per analitzar</span>"
            )
        self.analyze_btn.setEnabled(True)

    # ------------------------------------------------------------------
    # Run analysis
    # ------------------------------------------------------------------

    def _run_analyze(self):
        """Executa l'anàlisi."""
        imported_data = self.main_window.imported_data
        calibration_data = self.main_window.calibration_data
        seq_path = self.main_window.seq_path

        # Auto-load if not in memory
        if not imported_data and seq_path:
            from hpsec_import import import_from_manifest
            self.main_window.set_status("Carregant dades d'importació...")
            try:
                imported_data = import_from_manifest(seq_path)
            except Exception as e:
                logger.warning(f"Error carregant import: {e}")
                imported_data = None
            if imported_data and imported_data.get('success'):
                self.main_window.imported_data = imported_data

        # ensure_data_loaded() es fa dins del AnalyzeWorker (thread)
        # per no bloquejar la UI si cal llegir MasterFile + CSV + Export3D

        if not calibration_data and seq_path:
            import json
            cal_path = Path(seq_path) / "CHECK" / "data" / "calibration_result.json"
            if cal_path.exists():
                self.main_window.set_status("Carregant dades de calibració...")
                with open(cal_path, 'r', encoding='utf-8') as f:
                    cal_file = json.load(f)
                calibrations = cal_file.get("calibrations", [])
                if calibrations:
                    active_cal = None
                    for cal in calibrations:
                        if cal.get("is_active", False):
                            active_cal = cal
                            break
                    if not active_cal:
                        active_cal = calibrations[0]

                    # Reconstruct full calibration_data matching calibrate_panel output
                    area = active_cal.get("area", 0)
                    conc = active_cal.get("conc_ppm", 5)
                    volume = active_cal.get("volume_uL", 0)
                    rf = active_cal.get("rf", 0)
                    if rf == 0 and conc > 0 and area > 0:
                        rf = area / conc
                    rf_direct = active_cal.get("rf_direct", rf)
                    rf_uib = active_cal.get("rf_uib", 0)
                    rf_mass = active_cal.get("rf_mass", 0)

                    calibration_data = {
                        "success": True,
                        "mode": active_cal.get("mode", "DUAL"),
                        "rf_direct": rf_direct,
                        "rf_uib": rf_uib,
                        "rf": rf,
                        "rf_mass": rf_mass,
                        "shift_direct": active_cal.get("shift_direct") or active_cal.get("shift_min", 0),
                        "shift_uib": active_cal.get("shift_uib") or active_cal.get("shift_min_u", 0),
                        "khp_area_direct": area,
                        "khp_area_uib": active_cal.get("area_u", 0),
                        "khp_area": area,
                        "khp_conc": conc,
                        "volume_uL": volume,
                        "khp_source": active_cal.get("khp_source", "LOCAL"),
                        "calibration": active_cal,
                        "errors": [],
                        "loaded_from_json": True,
                    }
                    self.main_window.calibration_data = calibration_data

        if not imported_data:
            QMessageBox.warning(self, "Avís", "No s'han trobat dades d'importació.")
            self.analyze_completed.emit({'success': False, 'errors': ["No hi ha dades d'importació"]})
            return

        samples = imported_data.get("samples", {})
        if not samples:
            QMessageBox.warning(self, "Avís", "No s'han trobat mostres a les dades importades.")
            self.analyze_completed.emit({'success': False, 'errors': ["No hi ha mostres a les dades"]})
            return

        self.analyze_btn.setEnabled(False)
        self.empty_state.setVisible(False)
        self.status_frame.setVisible(False)
        self.progress_frame.setVisible(True)
        self.progress_bar.setValue(0)
        self.results_frame.setVisible(False)

        self.worker = AnalyzeWorker(imported_data, calibration_data)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_progress(self, msg, pct):
        self.progress_label.setText(msg)
        self.progress_bar.setValue(pct)

    def _on_finished(self, result):
        """Gestiona la finalització de l'anàlisi."""
        self.progress_frame.setVisible(False)
        self.analyze_btn.setEnabled(True)

        if not result or not result.get("success"):
            error_msg = result.get("error", "Error desconegut") if result else "Resultat buit"
            # Mostrar error inline (visible i persistent)
            self._show_inline_message(error_msg, level="error")
            self._update_status()
            self.analyze_completed.emit(result or {"success": False, "error": error_msg})
            return

        self.main_window.processed_data = result
        self.samples_grouped = result.get("samples_grouped", {})

        save_analysis_result(result)

        self._populate_table()
        self.results_frame.setVisible(True)

        # Mostrar warnings inline si n'hi ha
        warnings = result.get("warnings_structured") or result.get("warnings", [])
        if warnings:
            if isinstance(warnings, dict):
                parts = []
                for cat, items in warnings.items():
                    if items:
                        parts.append(f"<b>{cat}:</b> {len(items)} avís(os)")
                warn_text = "<br>".join(parts)
            elif isinstance(warnings, list) and warnings:
                warn_text = f"{len(warnings)} avís(os) detectats"
            else:
                warn_text = ""
            if warn_text:
                self._show_inline_message(warn_text, level="warning")
            else:
                self.status_frame.setVisible(False)
        else:
            self.status_frame.setVisible(False)

        self.analyze_completed.emit(result)

    def _on_error(self, error_msg):
        logger.error(f"Error durant anàlisi: {error_msg}")
        self.progress_frame.setVisible(False)
        self.analyze_btn.setEnabled(True)
        # Mostrar error inline en lloc de QMessageBox
        self._show_inline_message(str(error_msg), level="error")
        self.analyze_completed.emit({"success": False, "error": error_msg})

    def _show_inline_message(self, message, level="info"):
        """Mostra un missatge inline al panell (error/warning/info)."""
        colors = {
            "error": ("background: #FADBD8; border: 1px solid #E74C3C; "
                      "border-radius: 6px; padding: 10px;",
                      "#922B21"),
            "warning": ("background: #FCF3CF; border: 1px solid #F39C12; "
                        "border-radius: 6px; padding: 10px;",
                        "#7D6608"),
            "info": ("background: #D6EAF8; border: 1px solid #2980B9; "
                     "border-radius: 6px; padding: 10px;",
                     "#1A5276"),
        }
        frame_style, text_color = colors.get(level, colors["info"])
        icon = {"error": "\u274c", "warning": "\u26a0\ufe0f", "info": "\u2139\ufe0f"}.get(level, "")
        self.status_frame.setStyleSheet(f"QFrame {{ {frame_style} }}")
        self.status_label.setStyleSheet(f"color: {text_color}; font-size: 12px;")
        self.status_label.setText(f"{icon} {message}")
        self.status_frame.setVisible(True)

    # ------------------------------------------------------------------
    # (SEQ_CAL regression removed — ara a GlobalCalibrationPanel, tab 5)
    # ------------------------------------------------------------------

    # === REMOVED: 14 mètodes SEQ_CAL (~950 línies) ===
    # _build_seq_cal_regression_section, _on_calibration_data_updated,
    # _check_and_show_seq_cal, _check_uib_sensitivity_mixing,
    # _run_seq_cal_regression, _update_seq_cal_ui, _populate_seq_cal_table,
    # _on_seq_cal_row_clicked, _on_seq_cal_point_toggled,
    # _on_seq_cal_signal_changed, _on_seq_cal_recalculate,
    # _on_seq_cal_repair_toggled, _update_seq_cal_comparison,
    # _update_seq_cal_graph
    # ==================================================


    # ------------------------------------------------------------------
    # Populate unified table
    # ------------------------------------------------------------------

    def _populate_table(self):
        """Omple la taula unificada amb els resultats (13 cols, selectors DOC/DAD independents)."""
        self.results_table.setRowCount(0)
        self._sample_row_map = {}
        n_ok, n_warning, n_error, n_light, n_khp = 0, 0, 0, 0, 0

        # Separar mostres regulars, KHP i light
        regular_names = []
        khp_names = []
        light_names = []
        for name in sorted(self.samples_grouped.keys()):
            sd = self.samples_grouped[name]
            if sd.get("analysis_type") == "khp":
                khp_names.append(name)
            elif sd.get("analysis_type") == "light":
                light_names.append(name)
            else:
                regular_names.append(name)

        # --- Regular samples ---
        for sample_name in regular_names:
            sample_data = self.samples_grouped[sample_name]
            row = self.results_table.rowCount()
            self.results_table.insertRow(row)
            self._sample_row_map[sample_name] = row

            replicas = sample_data.get("replicas") or {}
            comparison = sample_data.get("comparison") or {}
            recommendation = sample_data.get("recommendation") or {}
            selected = sample_data.get("selected") or {"doc": "1", "dad": "1"}
            quantification = sample_data.get("quantification") or {}

            # Recommended replicas (may differ for DOC vs DAD)
            doc_rec = (recommendation.get("doc") or {}).get("replica", "1")
            dad_rec = (recommendation.get("dad") or {}).get("replica", "1")
            doc_sel = selected.get("doc", doc_rec)
            dad_sel = selected.get("dad", dad_rec)
            doc_rep = replicas.get(doc_sel, {})
            dad_rep = replicas.get(dad_sel, {})

            # Col 0: Sample name
            item_name = QTableWidgetItem(sample_name)
            item_name.setData(Qt.UserRole, sample_name)
            inj_indices = []
            for rk, rd in sorted(replicas.items()):
                idx = rd.get("injection_index")
                if idx is not None:
                    inj_indices.append(f"R{rk}: inj #{idx}")
            if inj_indices:
                item_name.setToolTip("Ordre injecció: " + ", ".join(inj_indices))
            self.results_table.setItem(row, 0, item_name)

            # Col 1: Sel DOC — replica selector with (s) for suggested + "Cap" option
            doc_combo = QComboBox()
            doc_combo.setStyleSheet("QComboBox { border: none; background: transparent; padding: 2px; }")
            for rep_num in sorted(replicas.keys()):
                label = f"R{rep_num} (s)" if rep_num == doc_rec else f"R{rep_num}"
                doc_combo.addItem(label, rep_num)
                if rep_num == doc_sel:
                    doc_combo.setCurrentIndex(doc_combo.count() - 1)
            doc_combo.addItem("Cap", "none")
            if doc_sel == "none":
                doc_combo.setCurrentIndex(doc_combo.count() - 1)
            doc_combo.currentIndexChanged.connect(
                lambda idx, name=sample_name: self._on_doc_replica_changed(name)
            )
            self.results_table.setCellWidget(row, 1, doc_combo)

            # Col 2: Sel DAD — replica selector with (s) for suggested + "Cap" option
            dad_combo = QComboBox()
            dad_combo.setStyleSheet("QComboBox { border: none; background: transparent; padding: 2px; }")
            for rep_num in sorted(replicas.keys()):
                label = f"R{rep_num} (s)" if rep_num == dad_rec else f"R{rep_num}"
                dad_combo.addItem(label, rep_num)
                if rep_num == dad_sel:
                    dad_combo.setCurrentIndex(dad_combo.count() - 1)
            dad_combo.addItem("Cap", "none")
            if dad_sel == "none":
                dad_combo.setCurrentIndex(dad_combo.count() - 1)
            dad_combo.currentIndexChanged.connect(
                lambda idx, name=sample_name: self._on_dad_replica_changed(name)
            )
            self.results_table.setCellWidget(row, 2, dad_combo)

            # --- DOC columns (from DOC replica) ---

            # Col 3: A_DOC
            areas = doc_rep.get("areas") or {}
            doc_areas = areas.get("DOC") or {}
            area_direct = doc_areas.get("total", 0)
            self.results_table.setItem(row, 3, QTableWidgetItem(
                f"{area_direct:.0f}" if area_direct else "-"))

            # Col 4: ppm
            ppm_direct = quantification.get("concentration_ppm_direct") or quantification.get("concentration_ppm")
            self.results_table.setItem(row, 4, QTableWidgetItem(
                f"{ppm_direct:.2f}" if ppm_direct else "-"))

            # Col 5: A_UIB
            areas_uib = doc_rep.get("areas_uib") or {}
            area_uib = areas_uib.get("total", 0)
            self.results_table.setItem(row, 5, QTableWidgetItem(
                f"{area_uib:.0f}" if area_uib else "-"))

            # Col 6: ppm_U
            ppm_uib = quantification.get("concentration_ppm_uib")
            self.results_table.setItem(row, 6, QTableWidgetItem(
                f"{ppm_uib:.2f}" if ppm_uib else "-"))

            # Col 7: SNR (DOC Direct)
            snr_info = doc_rep.get("snr_info") or {}
            snr_direct = snr_info.get("snr_direct", 0)
            snr_item = QTableWidgetItem(f"{snr_direct:.0f}" if snr_direct else "-")
            if snr_direct and snr_direct < 10:
                snr_item.setForeground(QBrush(QColor(COLOR_ERROR)))
            elif snr_direct and snr_direct < 50:
                snr_item.setForeground(QBrush(QColor(COLOR_WARNING)))
            snr_uib = snr_info.get("snr_uib", 0)
            if snr_uib:
                snr_item.setToolTip(f"SNR UIB: {snr_uib:.0f}")
            self.results_table.setItem(row, 7, snr_item)

            # --- DAD columns (from DAD replica) ---

            # Col 8: A_254
            dad_areas = (dad_rep.get("areas") or {})
            area_254 = dad_areas.get("A254", {}).get("total", 0)
            self.results_table.setItem(row, 8, QTableWidgetItem(
                f"{area_254:.1f}" if area_254 else "-"))

            # Col 9: SNR_254
            snr_info_dad = dad_rep.get("snr_info_dad") or {}
            snr_254 = snr_info_dad.get("A254", {}).get("snr", 0)
            snr_254_item = QTableWidgetItem(f"{snr_254:.0f}" if snr_254 else "-")
            if snr_254 and snr_254 < 10:
                snr_254_item.setForeground(QBrush(QColor(COLOR_ERROR)))
            elif snr_254 and snr_254 < 50:
                snr_254_item.setForeground(QBrush(QColor(COLOR_WARNING)))
            self.results_table.setItem(row, 9, snr_254_item)

            # --- Correlation columns (sample-level, not replica-specific) ---

            # Col 10: R²_DOC
            r2_doc = comparison.get("doc", {}).get("pearson", 0) if comparison else 0
            r2_doc_item = QTableWidgetItem(f"{r2_doc:.4f}" if r2_doc > 0 else "-")
            if 0 < r2_doc < 0.990:
                r2_doc_item.setForeground(QBrush(QColor(COLOR_WARNING)))
            self.results_table.setItem(row, 10, r2_doc_item)

            # Col 11: R²_DAD (min across wavelengths)
            dad_comp = comparison.get("dad", {}) if comparison else {}
            r2_dad_min = dad_comp.get("pearson_min", 0)
            wl_min = dad_comp.get("wavelength_min", "")
            if r2_dad_min > 0:
                cell_text = f"{r2_dad_min:.4f}"
                if 0 < r2_dad_min < 0.990 and wl_min:
                    cell_text += f" (A{wl_min})"
            else:
                cell_text = "-"
            r2_dad_item = QTableWidgetItem(cell_text)
            if 0 < r2_dad_min < 0.990:
                r2_dad_item.setForeground(QBrush(QColor(COLOR_WARNING)))
            pearson_per_wl = dad_comp.get("pearson_per_wavelength", {})
            if pearson_per_wl:
                tip_lines = []
                for wl, val in sorted(pearson_per_wl.items()):
                    marker = " ← min" if str(wl) == str(wl_min) else ""
                    warn = " ⚠" if val < 0.990 else ""
                    tip_lines.append(f"A{wl}: {val:.4f}{warn}{marker}")
                r2_dad_item.setToolTip("\n".join(tip_lines))
            self.results_table.setItem(row, 11, r2_dad_item)

            # Col 12: HCI (Humic Character Index)
            hci_val = quantification.get("hci")
            if hci_val is not None:
                hci_char = quantification.get("hci_character", "")
                abbrev = "HA" if "HA" in hci_char else "FA" if "FA" in hci_char else "Mix"
                hci_item = QTableWidgetItem(f"{hci_val:.1f} {abbrev}")
                # Color de fons segons caràcter
                if hci_val > 60:
                    hci_item.setBackground(QBrush(QColor("#FADBD8")))
                elif hci_val < 40:
                    hci_item.setBackground(QBrush(QColor("#D6EAF8")))
                else:
                    hci_item.setBackground(QBrush(QColor("#D5F5E3")))
                hci_item.setToolTip(
                    f"Humic Character Index: {hci_val:.1f} ({hci_char})\n"
                    f"Model PCA+LDA v2.0")
            else:
                hci_item = QTableWidgetItem("-")
            self.results_table.setItem(row, 12, hci_item)

            # Col 13: Estat (considers both DOC and DAD replicas)
            status_color, status_text, tooltip = self._classify_sample_status(
                doc_rep, dad_rep, comparison, sample_data=sample_data)
            status_item = QTableWidgetItem(status_text)
            status_item.setForeground(QBrush(QColor(status_color)))
            status_item.setToolTip(tooltip)
            self.results_table.setItem(row, 13, status_item)

            # Count stats
            if status_color == COLOR_ERROR:
                n_error += 1
            elif status_color == COLOR_WARNING:
                n_warning += 1
            else:
                n_ok += 1

        # --- Separator + KHP STANDARDS ---
        # Per SEQ_CAL, la taula de regressió superior ja mostra tota la info KHP
        cal_data = self.main_window.calibration_data or {}
        is_seq_cal = cal_data.get('is_seq_cal', False)

        if khp_names and not is_seq_cal:
            n_cols = self.results_table.columnCount()

            # Títol separator
            sep_title = "--- KHP STANDARDS ---"
            sep_row = self.results_table.rowCount()
            self.results_table.insertRow(sep_row)
            sep_item = QTableWidgetItem(sep_title)
            sep_item.setFlags(Qt.ItemIsEnabled)
            sep_font = QFont()
            sep_font.setBold(True)
            sep_item.setFont(sep_font)
            sep_item.setForeground(QBrush(QColor("#1565C0")))
            self.results_table.setItem(sep_row, 0, sep_item)
            self.results_table.setSpan(sep_row, 0, 1, n_cols)
            sep_bg = QBrush(QColor("#E3F2FD"))
            for c in range(n_cols):
                item = self.results_table.item(sep_row, c)
                if item is None:
                    item = QTableWidgetItem("")
                    self.results_table.setItem(sep_row, c, item)
                item.setBackground(sep_bg)

            for sample_name in khp_names:
                sample_data = self.samples_grouped[sample_name]
                row = self.results_table.rowCount()
                self.results_table.insertRow(row)
                self._sample_row_map[sample_name] = row

                replicas = sample_data.get("replicas") or {}
                selected = sample_data.get("selected") or {}
                doc_sel = selected.get("doc", sorted(replicas.keys())[0] if replicas else "1")
                doc_rep = replicas.get(doc_sel, {})

                # Col 0: Sample name
                item_name = QTableWidgetItem(sample_name)
                item_name.setData(Qt.UserRole, sample_name)
                inj_indices = []
                for rk, rd in sorted(replicas.items()):
                    idx = rd.get("injection_index")
                    if idx is not None:
                        inj_indices.append(f"R{rk}: inj #{idx}")
                if inj_indices:
                    item_name.setToolTip("Ordre injecció: " + ", ".join(inj_indices))
                self.results_table.setItem(row, 0, item_name)

                # Col 1-2: No selectors for KHP
                self.results_table.setItem(row, 1, QTableWidgetItem("-"))
                self.results_table.setItem(row, 2, QTableWidgetItem("-"))

                # Col 3: A_DOC
                areas = doc_rep.get("areas", {})
                area_doc = areas.get("DOC", {}).get("total", 0) if areas else 0
                area_item = QTableWidgetItem(f"{area_doc:.0f}" if area_doc else "-")
                self.results_table.setItem(row, 3, area_item)

                # Col 4-6: no aplica per KHP en mode normal
                for c in (4, 5, 6):
                    self.results_table.setItem(row, c, QTableWidgetItem("-"))

                # Col 7: SNR
                snr_info = doc_rep.get("snr_info", {})
                snr = snr_info.get("snr_direct", 0) if snr_info else 0
                snr_item = QTableWidgetItem(f"{snr:.0f}" if snr else "-")
                if snr and snr < 10:
                    snr_item.setForeground(QBrush(QColor(COLOR_ERROR)))
                elif snr and snr < 50:
                    snr_item.setForeground(QBrush(QColor(COLOR_WARNING)))
                self.results_table.setItem(row, 7, snr_item)

                # Col 8: A_254
                a254 = 0
                if areas:
                    a254 = (areas.get("254nm", {}).get("total", 0) or
                            areas.get("A254", {}).get("total", 0))
                a254_item = QTableWidgetItem(f"{a254:.0f}" if a254 else "-")
                self.results_table.setItem(row, 8, a254_item)

                # Col 9: SNR_254
                snr_254 = snr_info.get("snr_254", 0) if snr_info else 0
                snr254_item = QTableWidgetItem(f"{snr_254:.0f}" if snr_254 else "-")
                if snr_254 and snr_254 < 10:
                    snr254_item.setForeground(QBrush(QColor(COLOR_ERROR)))
                elif snr_254 and snr_254 < 50:
                    snr254_item.setForeground(QBrush(QColor(COLOR_WARNING)))
                self.results_table.setItem(row, 9, snr254_item)

                # Col 10-11: No R² comparison
                self.results_table.setItem(row, 10, QTableWidgetItem("-"))
                self.results_table.setItem(row, 11, QTableWidgetItem("-"))

                # Col 12: HCI (not applicable for KHP)
                self.results_table.setItem(row, 12, QTableWidgetItem("-"))

                # Col 13: KHP type
                type_item = QTableWidgetItem("KHP")
                type_item.setForeground(QBrush(QColor("#1565C0")))
                self.results_table.setItem(row, 13, type_item)

                # Blue-tinted background
                khp_bg = QBrush(QColor("#E8F4FD"))
                for c in range(n_cols):
                    item = self.results_table.item(row, c)
                    if item:
                        item.setBackground(khp_bg)

                n_khp += 1

        # --- Separator + Light samples (BLANC / CONTROL) ---
        if light_names:
            n_cols = self.results_table.columnCount()
            sep_row = self.results_table.rowCount()
            self.results_table.insertRow(sep_row)
            sep_item = QTableWidgetItem("--- BLANCS / CONTROLS ---")
            sep_item.setFlags(Qt.ItemIsEnabled)  # Non-selectable
            sep_font = QFont()
            sep_font.setBold(True)
            sep_item.setFont(sep_font)
            sep_item.setForeground(QBrush(QColor("#888888")))
            self.results_table.setItem(sep_row, 0, sep_item)
            self.results_table.setSpan(sep_row, 0, 1, n_cols)
            sep_bg = QBrush(QColor("#E8E8E8"))
            for c in range(n_cols):
                item = self.results_table.item(sep_row, c)
                if item is None:
                    item = QTableWidgetItem("")
                    self.results_table.setItem(sep_row, c, item)
                item.setBackground(sep_bg)

            for sample_name in light_names:
                sample_data = self.samples_grouped[sample_name]
                row = self.results_table.rowCount()
                self.results_table.insertRow(row)
                self._sample_row_map[sample_name] = row

                replicas = sample_data.get("replicas") or {}
                selected = sample_data.get("selected") or {}
                doc_sel = selected.get("doc", sorted(replicas.keys())[0] if replicas else "1")
                doc_rep = replicas.get(doc_sel, {})
                sample_type = sample_data.get("sample_type", "BLANK")

                # Col 0: Sample name
                item_name = QTableWidgetItem(sample_name)
                item_name.setData(Qt.UserRole, sample_name)
                inj_indices = []
                for rk, rd in sorted(replicas.items()):
                    idx = rd.get("injection_index")
                    if idx is not None:
                        inj_indices.append(f"R{rk}: inj #{idx}")
                if inj_indices:
                    item_name.setToolTip("Ordre injecció: " + ", ".join(inj_indices))
                self.results_table.setItem(row, 0, item_name)

                # Col 1-2: No selectors for light samples
                self.results_table.setItem(row, 1, QTableWidgetItem("-"))
                self.results_table.setItem(row, 2, QTableWidgetItem("-"))

                # Col 3: A_DOC (area_total from light analysis)
                area_total = doc_rep.get("area_total", 0)
                self.results_table.setItem(row, 3, QTableWidgetItem(
                    f"{area_total:.0f}" if area_total else "-"))

                # Col 4-6: No ppm, no UIB
                for c in (4, 5, 6):
                    self.results_table.setItem(row, c, QTableWidgetItem("-"))

                # Col 7: SNR (color-coded same as regular)
                snr = doc_rep.get("snr", 0)
                snr_item = QTableWidgetItem(f"{snr:.0f}" if snr else "-")
                if snr and snr < 10:
                    snr_item.setForeground(QBrush(QColor(COLOR_ERROR)))
                elif snr and snr < 50:
                    snr_item.setForeground(QBrush(QColor(COLOR_WARNING)))
                self.results_table.setItem(row, 7, snr_item)

                # Col 8-11: No DAD, no R²
                for c in (8, 9, 10, 11):
                    self.results_table.setItem(row, c, QTableWidgetItem("-"))

                # Col 12: HCI (not applicable for light samples)
                self.results_table.setItem(row, 12, QTableWidgetItem("-"))

                # Col 13: sample_type text
                type_item = QTableWidgetItem(sample_type)
                type_item.setForeground(QBrush(QColor("#888888")))
                self.results_table.setItem(row, 13, type_item)

                # Light grey background
                light_bg = QBrush(QColor("#F0F0F0"))
                for c in range(n_cols):
                    item = self.results_table.item(row, c)
                    if item:
                        item.setBackground(light_bg)

                n_light += 1

        # Update stats bar
        total = n_ok + n_warning + n_error
        stats_text = (
            f"<b>Total:</b> {total} mostres &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"<span style='color:#27AE60'>●</span> OK: {n_ok} &nbsp;&nbsp;"
            f"<span style='color:#F39C12'>●</span> Warning: {n_warning} &nbsp;&nbsp;"
            f"<span style='color:#E74C3C'>●</span> Error: {n_error}"
        )
        if n_khp > 0:
            stats_text += f" &nbsp;&nbsp;|&nbsp;&nbsp; <span style='color:#1565C0'>●</span> KHP: {n_khp}"
        if n_light > 0:
            stats_text += f" &nbsp;&nbsp;|&nbsp;&nbsp; <span style='color:#888888'>●</span> Blancs/Controls: {n_light}"
        self.stats_label.setText(stats_text)

    # ------------------------------------------------------------------
    # Anomaly severity classification
    # ------------------------------------------------------------------

    def _classify_sample_status(self, doc_rep_data, dad_rep_data, comparison,
                                sample_data=None):
        """Classifica l'estat d'una mostra considerant ambdues rèpliques (DOC + DAD).

        Usa ANOMALY_CATALOG com a font de veritat per severitat, icones i labels.

        Args:
            doc_rep_data: Dades de la rèplica DOC seleccionada
            dad_rep_data: Dades de la rèplica DAD seleccionada
            comparison: Comparació entre rèpliques
            sample_data: Dict complet del sample_group (per accedir a sample_valid, repaired)

        Returns (color, status_text, tooltip).
        """
        # Comprovar si l'usuari ha seleccionat "Cap"
        if sample_data:
            selected = sample_data.get("selected", {})
            if selected.get("doc") == "none":
                return COLOR_ERROR, "NO VÀL", "Usuari ha seleccionat 'Cap' — No es quantificarà ni exportarà"
            # Comprovar mostra no vàlida (ambdues rèpliques amb anomalies no reparables)
            if sample_data.get("sample_valid") is False and not sample_data.get("repaired"):
                reason = (sample_data.get("recommendation", {})
                          .get("doc", {}).get("reason", "Ambdues rèpliques amb anomalies crítiques"))
                return COLOR_ERROR, "NO VÀL", f"Mostra no vàlida — {reason}\nSeleccionar 'Cap' o generar noves dades"

        # Merge anomalies from both replicas (deduplicate by code)
        doc_anomalies = doc_rep_data.get("anomalies", [])
        dad_anomalies = dad_rep_data.get("anomalies", [])
        all_anomalies = list(doc_anomalies)
        existing_codes = get_anomaly_codes(all_anomalies)
        for a in dad_anomalies:
            code = a.get("code") if isinstance(a, dict) else str(a).replace("_REPAIRED", "")
            if code not in existing_codes:
                all_anomalies.append(a)
                existing_codes.add(code)

        classified = classify_anomalies(all_anomalies)
        timeout_info = doc_rep_data.get("timeout_info", {})
        timeout_severity = timeout_info.get("severity", "OK")
        n_timeouts = timeout_info.get("n_timeouts", 0)
        replica_warnings = comparison.get("doc", {}).get("warnings", []) if comparison else []

        # Build status icons from catalog
        status_parts = []
        seen_icons = set()
        for a in all_anomalies:
            if isinstance(a, dict):
                code = a.get("code", "")
                repaired = a.get("repaired", False)
            else:
                repaired = "_REPAIRED" in str(a)
                code = str(a).replace("_REPAIRED", "")
            entry = ANOMALY_CATALOG.get(code, {})
            icon = entry.get("icon", "")
            if icon and icon not in seen_icons:
                seen_icons.add(icon)
                status_parts.append(f"{icon}*" if repaired else icon)
        if n_timeouts > 0 and "T!" not in seen_icons:
            status_parts.append(f"T({n_timeouts})")

        # Determine color
        has_blocker = bool(classified["blocker"])
        has_warn = bool(classified["warning"] or classified["repaired"]
                        or (timeout_severity in ("WARNING", "CRITICAL"))
                        or replica_warnings)
        if has_blocker:
            status_color = COLOR_ERROR
        elif has_warn:
            status_color = COLOR_WARNING
        else:
            status_color = COLOR_SUCCESS

        status_text = " ".join(status_parts) if status_parts else "\u2713"

        # Build tooltip with catalog labels + action hints
        tooltip_parts = []
        for key, label_prefix in [("blocker", "CRÍTIC"), ("repaired", "REPARAT"),
                                    ("warning", "Avís"), ("info", "Info")]:
            items = classified[key]
            if items:
                for a in items:
                    code = a.get("code") if isinstance(a, dict) else str(a).replace("_REPAIRED", "")
                    entry = ANOMALY_CATALOG.get(code, {})
                    lbl = entry.get("label", code)
                    det = a.get("details", {}) if isinstance(a, dict) else {}
                    if det.get("snr"):
                        lbl += f" (SNR={det['snr']:.1f})"
                    line = f"{label_prefix}: {lbl}"
                    action = entry.get("action", "")
                    if action:
                        line += f"\n   \u2192 {action}"
                    tooltip_parts.append(line)

        if n_timeouts > 0:
            zones = timeout_info.get("zones", [])
            tooltip_parts.append(
                f"Timeouts: {n_timeouts} ({timeout_severity}) — zones: {', '.join(zones) if zones else '?'}"
            )
        if replica_warnings:
            for rw in replica_warnings:
                if isinstance(rw, dict):
                    tooltip_parts.append(rw.get("label", rw.get("code", str(rw))))
                else:
                    tooltip_parts.append(str(rw))

        # Repairable hint
        if sample_data and sample_data.get("repairable") and not sample_data.get("repaired"):
            tooltip_parts.append("Pic amb cim irregular reparable — Doble-clic per opcions de reparació")

        tooltip = "\n".join(tooltip_parts) if tooltip_parts else "OK"
        return status_color, status_text, tooltip

    # ------------------------------------------------------------------
    # Replica change (separate DOC / DAD handlers)
    # ------------------------------------------------------------------

    def _on_doc_replica_changed(self, sample_name):
        """Gestiona el canvi de rèplica DOC (inclou opció 'Cap')."""
        if sample_name not in self.samples_grouped:
            return
        row = self._sample_row_map.get(sample_name)
        if row is None:
            return
        combo = self.results_table.cellWidget(row, 1)
        if combo:
            new_replica = combo.currentData()
            self.samples_grouped[sample_name]["selected"]["doc"] = new_replica
            if new_replica == "none":
                # Marcar mostra com no vàlida per DOC
                self.samples_grouped[sample_name]["sample_valid"] = False
                self.samples_grouped[sample_name]["quantification"] = {
                    "concentration_ppm": None,
                    "concentration_ppm_direct": None,
                    "concentration_ppm_uib": None,
                    "area_total": None,
                    "valid": False,
                    "reason": "Usuari ha seleccionat 'Cap' per DOC"
                }
            else:
                # Restaurar validesa si era "none" abans
                self.samples_grouped[sample_name]["sample_valid"] = True
                self._update_quantification(sample_name)
            self._update_doc_columns(row, sample_name)
            self._update_estat_column(row, sample_name)

    def _on_dad_replica_changed(self, sample_name):
        """Gestiona el canvi de rèplica DAD (inclou opció 'Cap')."""
        if sample_name not in self.samples_grouped:
            return
        row = self._sample_row_map.get(sample_name)
        if row is None:
            return
        combo = self.results_table.cellWidget(row, 2)
        if combo:
            new_replica = combo.currentData()
            self.samples_grouped[sample_name]["selected"]["dad"] = new_replica
            self._update_dad_columns(row, sample_name)
            self._update_estat_column(row, sample_name)

    def _update_doc_columns(self, row, sample_name):
        """Actualitza columnes DOC (3-7) quan canvia la rèplica DOC."""
        sample_data = self.samples_grouped[sample_name]
        selected = sample_data.get("selected", {})
        doc_sel = selected.get("doc", "1")
        replicas = sample_data.get("replicas", {})

        # "Cap" seleccionat → buidar columnes
        if doc_sel == "none":
            for col in (3, 4, 5, 6, 7, 12):
                item = self.results_table.item(row, col)
                if item:
                    item.setText("-")
                    if col == 12:
                        item.setBackground(QBrush(QColor("#FFFFFF")))
                        item.setToolTip("")
            return

        doc_rep = replicas.get(doc_sel, {})
        quantification = sample_data.get("quantification", {})

        areas = doc_rep.get("areas") or {}
        doc_areas = areas.get("DOC") or {}
        areas_uib = doc_rep.get("areas_uib") or {}

        # Col 3: A_DOC
        area_direct = doc_areas.get("total", 0)
        self.results_table.item(row, 3).setText(f"{area_direct:.0f}" if area_direct else "-")

        # Col 4: ppm
        ppm_direct = quantification.get("concentration_ppm_direct") or quantification.get("concentration_ppm")
        self.results_table.item(row, 4).setText(f"{ppm_direct:.2f}" if ppm_direct else "-")

        # Col 5: A_UIB
        area_uib = areas_uib.get("total", 0)
        self.results_table.item(row, 5).setText(f"{area_uib:.0f}" if area_uib else "-")

        # Col 6: ppm_U
        ppm_uib = quantification.get("concentration_ppm_uib")
        self.results_table.item(row, 6).setText(f"{ppm_uib:.2f}" if ppm_uib else "-")

        # Col 7: SNR (DOC Direct)
        snr_info = doc_rep.get("snr_info") or {}
        snr_direct = snr_info.get("snr_direct", 0)
        snr_item = self.results_table.item(row, 7)
        if snr_item:
            snr_item.setText(f"{snr_direct:.0f}" if snr_direct else "-")
            if snr_direct and snr_direct < 10:
                snr_item.setForeground(QBrush(QColor(COLOR_ERROR)))
            elif snr_direct and snr_direct < 50:
                snr_item.setForeground(QBrush(QColor(COLOR_WARNING)))
            else:
                snr_item.setForeground(QBrush(QColor("#000000")))
            snr_uib = snr_info.get("snr_uib", 0)
            snr_item.setToolTip(f"SNR UIB: {snr_uib:.0f}" if snr_uib else "")

        # Col 12: HCI (update from new quantification)
        hci_item = self.results_table.item(row, 12)
        if hci_item:
            hci_val = quantification.get("hci")
            if hci_val is not None:
                hci_char = quantification.get("hci_character", "")
                abbrev = "HA" if "HA" in hci_char else "FA" if "FA" in hci_char else "Mix"
                hci_item.setText(f"{hci_val:.1f} {abbrev}")
                if hci_val > 60:
                    hci_item.setBackground(QBrush(QColor("#FADBD8")))
                elif hci_val < 40:
                    hci_item.setBackground(QBrush(QColor("#D6EAF8")))
                else:
                    hci_item.setBackground(QBrush(QColor("#D5F5E3")))
                hci_item.setToolTip(
                    f"Humic Character Index: {hci_val:.1f} ({hci_char})\n"
                    f"Model PCA+LDA v2.0")
            else:
                hci_item.setText("-")
                hci_item.setBackground(QBrush(QColor("#FFFFFF")))
                hci_item.setToolTip("")

    def _update_dad_columns(self, row, sample_name):
        """Actualitza columnes DAD (8-9) quan canvia la rèplica DAD."""
        sample_data = self.samples_grouped[sample_name]
        selected = sample_data.get("selected", {})
        dad_sel = selected.get("dad", "1")
        replicas = sample_data.get("replicas", {})
        dad_rep = replicas.get(dad_sel, {})

        # Col 8: A_254
        dad_areas = (dad_rep.get("areas") or {})
        area_254 = dad_areas.get("A254", {}).get("total", 0)
        item_8 = self.results_table.item(row, 8)
        if item_8:
            item_8.setText(f"{area_254:.1f}" if area_254 else "-")

        # Col 9: SNR_254
        snr_info_dad = dad_rep.get("snr_info_dad") or {}
        snr_254 = snr_info_dad.get("A254", {}).get("snr", 0)
        snr_254_item = self.results_table.item(row, 9)
        if snr_254_item:
            snr_254_item.setText(f"{snr_254:.0f}" if snr_254 else "-")
            if snr_254 and snr_254 < 10:
                snr_254_item.setForeground(QBrush(QColor(COLOR_ERROR)))
            elif snr_254 and snr_254 < 50:
                snr_254_item.setForeground(QBrush(QColor(COLOR_WARNING)))
            else:
                snr_254_item.setForeground(QBrush(QColor("#000000")))

    def _update_estat_column(self, row, sample_name):
        """Actualitza la columna Estat (col 13) considerant ambdues rèpliques."""
        sample_data = self.samples_grouped[sample_name]
        selected = sample_data.get("selected", {})
        replicas = sample_data.get("replicas", {})
        comparison = sample_data.get("comparison") or {}
        doc_rep = replicas.get(selected.get("doc", "1"), {})
        dad_rep = replicas.get(selected.get("dad", "1"), {})

        status_color, status_text, tooltip = self._classify_sample_status(
            doc_rep, dad_rep, comparison, sample_data=sample_data)
        status_item = self.results_table.item(row, 13)
        if status_item:
            status_item.setText(status_text)
            status_item.setForeground(QBrush(QColor(status_color)))
            status_item.setToolTip(tooltip)

    # ------------------------------------------------------------------
    # Quantification recalculation
    # ------------------------------------------------------------------

    def _update_quantification(self, sample_name):
        """Recalcula la quantificació per una mostra."""
        try:
            from hpsec_analyze import quantify_sample
            from hpsec_calibrate import get_all_active_calibrations

            sample_data = self.samples_grouped[sample_name]
            selected_doc = sample_data["selected"]["doc"]
            selected_replica = sample_data["replicas"].get(selected_doc)

            if selected_replica:
                processed = self.main_window.processed_data or {}
                method = processed.get("method", "COLUMN")
                mode = "BP" if method == "BP" else "COLUMN"
                seq_date = processed.get("seq_date")

                calibration_data = None
                inj_volume = selected_replica.get("inj_volume")
                seq_path = processed.get("seq_path", "")

                if seq_path and inj_volume:
                    try:
                        active_cals = get_all_active_calibrations(seq_path, mode)
                        for cal in active_cals:
                            cal_vol = cal.get("volume_uL", 0)
                            if cal_vol > 0 and abs(cal_vol - inj_volume) < 0.1:
                                calibration_data = cal
                                break
                        if not calibration_data and active_cals:
                            calibration_data = active_cals[0]
                    except Exception as e:
                        logger.warning(f"Error carregant calibracions: {e}")

                if not calibration_data:
                    calibration_data = self.main_window.calibration_data
                    # Ensure volume context from replica is available
                    if calibration_data and inj_volume:
                        if not calibration_data.get("volume_uL"):
                            calibration_data = dict(calibration_data)
                            calibration_data["volume_uL"] = inj_volume

                quantification = quantify_sample(
                    selected_replica, calibration_data,
                    mode=mode, seq_date=seq_date
                )
                # Propagar HCI de la rèplica seleccionada
                hci = selected_replica.get("hci")
                if hci is not None:
                    quantification["hci"] = hci
                    quantification["hci_character"] = selected_replica.get("hci_character", "")
                sample_data["quantification"] = quantification
        except Exception as e:
            logger.error(f"Error recalculant quantificació: {e}")
            self.main_window.set_status(f"Error quantificació: {e}", 5000)

    # ------------------------------------------------------------------
    # Table interaction
    # ------------------------------------------------------------------

    def _on_table_double_click(self, index):
        """Handler per doble clic — obre SampleDetailDialog per totes les tipologies."""
        row = index.row()
        item = self.results_table.item(row, 0)
        if item:
            sample_name = item.data(Qt.UserRole)
            if not sample_name:
                return  # Separator row
            sample_data = self.samples_grouped.get(sample_name)
            if not sample_data:
                return
            self._show_detail(sample_name)

    def _show_detail(self, sample_name):
        """Mostra el diàleg de detall (no-modal per evitar pèrdua de finestra)."""
        if sample_name not in self.samples_grouped:
            return
        # Tancar diàleg anterior si existeix
        if hasattr(self, '_detail_dialog') and self._detail_dialog is not None:
            try:
                self._detail_dialog.close()
            except RuntimeError:
                pass
        method = "COLUMN"
        if self.main_window.processed_data:
            method = self.main_window.processed_data.get("method", "COLUMN")
        dialog = SampleDetailDialog(
            sample_name,
            self.samples_grouped[sample_name],
            method,
            parent=self
        )
        self._detail_dialog = dialog  # Mantenir referència
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.finished.connect(lambda: self._on_detail_closed(sample_name))
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _on_detail_closed(self, sample_name):
        """Actualitza taula després de tancar el diàleg de detall."""
        row = self._sample_row_map.get(sample_name)
        if row is not None:
            sample_data = self.samples_grouped[sample_name]
            if sample_data.get("repaired"):
                self._update_quantification(sample_name)
                self._update_doc_columns(row, sample_name)
                self._update_estat_column(row, sample_name)
        self._detail_dialog = None

    # ------------------------------------------------------------------
    # Report PDF generation
    # ------------------------------------------------------------------

    def _generate_report(self):
        """Genera el report PDF d'anàlisi."""
        processed_data = self.main_window.processed_data
        if not processed_data:
            QMessageBox.warning(self, "Avís", "No hi ha dades processades.")
            return

        seq_path = processed_data.get("seq_path", "")
        if not seq_path:
            QMessageBox.warning(self, "Avís", "No s'ha trobat el path de la seqüència.")
            return

        try:
            self.report_btn.setEnabled(False)
            self.report_btn.setText("Generant...")

            from generate_analysis_report import generate_analysis_report

            # Passar dades en memòria (inclou seleccions actuals de l'usuari)
            report_data = dict(processed_data)
            report_data["samples_grouped"] = self.samples_grouped

            result = generate_analysis_report(
                seq_path, analysis_data=report_data
            )

            self.report_btn.setEnabled(True)
            self.report_btn.setText("Generar Report PDF")

            if result:
                QMessageBox.information(
                    self, "Report generat",
                    f"PDF generat correctament:\n{result}"
                )
                import os
                os.startfile(str(Path(result).parent))
            else:
                QMessageBox.warning(
                    self, "Error",
                    "No s'ha pogut generar el report PDF."
                )

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.report_btn.setEnabled(True)
            self.report_btn.setText("Generar Report PDF")
            QMessageBox.critical(
                self, "Error",
                f"Error generant el report:\n{str(e)}"
            )

