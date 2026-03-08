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

import numpy as np

logger = logging.getLogger(__name__)

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

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

# Fraction colors (consistent palette)
FRACTION_COLORS = {
    "BioP": "#3498DB",  # Blue
    "HS":   "#E74C3C",  # Red
    "BB":   "#F39C12",  # Orange
    "SB":   "#2ECC71",  # Green
    "LMW":  "#9B59B6",  # Purple
}
FRACTION_ORDER = ["BioP", "HS", "BB", "SB", "LMW"]


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
        # Chart data
        self._chart_regular = {}
        self._chart_blank = {}
        self._chart_control = {}
        self._chart_khp = {}
        self._chart_is_bp = False

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

        # === INFO PANEL (amagat — info ja visible a la taula) ===
        self.info_frame = QFrame()
        self.info_frame.setVisible(False)
        info_layout = QHBoxLayout(self.info_frame)
        self.import_info = QLabel()
        info_layout.addWidget(self.import_info)
        self.cal_info = QLabel()
        info_layout.addWidget(self.cal_info)
        self.status_indicator = QLabel()
        info_layout.addWidget(self.status_indicator)

        # Empty state
        self.empty_state = create_empty_state_widget(
            "🔬", "Preparant anàlisi...",
            "Carregant dades de la seqüència."
        )
        self.empty_state.setVisible(False)
        layout.addWidget(self.empty_state)

        # Status frame (mantingut per backward compat, sempre amagat)
        self.status_frame = QFrame()
        self.status_frame.setVisible(False)
        status_layout = QVBoxLayout(self.status_frame)
        self.status_label = QLabel()
        status_layout.addWidget(self.status_label)

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

        # === F0: SELECTOR BAR (primer element — filtra taula + gràfics) ===
        sel_frame = QFrame()
        sel_frame.setStyleSheet(
            "QFrame { background: #fff; border: 1px solid #e0e0e0;"
            " border-radius: 6px; }"
        )
        sel_layout = QHBoxLayout(sel_frame)
        sel_layout.setContentsMargins(10, 6, 10, 6)
        sel_layout.setSpacing(6)

        self._cat_buttons = {}
        self._cat_counts = {}
        self._sample_checkboxes = []

        for cat_key, label, color, checked in [
            ("sample", "Mostres", "#2E86AB", True),
            ("blank", "Blancs", "#95a5a6", False),
            ("control", "Control", "#888", False),
            ("khp", "KHP", "#1565C0", False),
        ]:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setChecked(checked)
            btn.clicked.connect(self._on_cat_toggle)
            self._cat_buttons[cat_key] = btn
            sel_layout.addWidget(btn)

        self._update_cat_btn_styles()

        sel_layout.addWidget(QLabel(
            "<span style='color:#ccc'>|</span>"
        ))

        # Botons agrupació (toggle)
        self._group_mode = 0  # 0=injecció, 1=tipus
        self._group_btns = []
        for i, label in enumerate(["Injecció", "Tipus"]):
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setChecked(i == 0)
            btn.clicked.connect(lambda _checked, idx=i: self._on_group_btn(idx))
            self._group_btns.append(btn)
            sel_layout.addWidget(btn)
        self._style_group_btns()

        sel_layout.addWidget(QLabel(
            "<span style='color:#ccc'>|</span>"
        ))

        sel_layout.addWidget(QLabel(
            "<b style='font-size:11px;color:#555'>DAD:</b>"
        ))
        self._wl_combo = QComboBox()
        self._wl_combo.setStyleSheet(
            "QComboBox { font-size: 11px; padding: 2px 6px;"
            " border: 1px solid #ccc; border-radius: 3px; }"
        )
        for wl in ["254", "220", "252", "272", "290", "362"]:
            self._wl_combo.addItem(f"A{wl}", wl)
        self._wl_combo.currentIndexChanged.connect(self._on_wl_changed)
        sel_layout.addWidget(self._wl_combo)

        sel_layout.addStretch()
        results_layout.addWidget(sel_frame)

        # === UNIFIED TABLE ===
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(14)
        self.results_table.setHorizontalHeaderLabels([
            "Mostra", "Sel DOC", "Sel DAD", "A_DOC", "ppm",
            "A_UIB", "ppm_U", "SNR", "A_254", "SNR_254",
            "R²_DOC", "R²_DAD", "HCI", "Estat"
        ])
        self.results_table.setMinimumHeight(180)
        configure_table_style(self.results_table)
        self._configure_unified_columns()
        results_layout.addWidget(self.results_table)

        # Connect table signals — clic obre detall directament
        self.results_table.doubleClicked.connect(self._on_table_double_click)
        self.results_table.setToolTip("Doble-clic per detall complet")

        layout.addWidget(self.results_frame, 1)

        # === CHARTS SECTION (sempre visible, sense collapsible) ===
        self._charts_visible = True
        self._charts_initialized = False
        self.charts_section = QFrame()
        self.charts_section.setVisible(False)
        charts_outer = QVBoxLayout(self.charts_section)
        charts_outer.setContentsMargins(0, 8, 0, 0)
        charts_outer.setSpacing(4)

        # Charts content (sempre visible)
        self._charts_content = QWidget()
        self._charts_content.setVisible(True)
        self._charts_content_layout = QVBoxLayout(self._charts_content)
        self._charts_content_layout.setContentsMargins(0, 4, 0, 0)
        self._charts_content_layout.setSpacing(4)

        if HAS_MATPLOTLIB:
            # F1+F2: DOC stacked + DOC overlay (costat)
            doc_row = QHBoxLayout()
            doc_row.setSpacing(4)

            self.doc_figure = Figure(figsize=(5, 3), dpi=100)
            self.doc_figure.set_facecolor("#FAFAFA")
            self.doc_canvas = FigureCanvas(self.doc_figure)
            self.doc_canvas.setMinimumHeight(180)
            doc_row.addWidget(self.doc_canvas)

            self.doc_overlay_figure = Figure(figsize=(5, 3), dpi=100)
            self.doc_overlay_figure.set_facecolor("#FAFAFA")
            self.doc_overlay_canvas = FigureCanvas(self.doc_overlay_figure)
            self.doc_overlay_canvas.setMinimumHeight(200)
            doc_row.addWidget(self.doc_overlay_canvas)

            self._charts_content_layout.addLayout(doc_row)

            # F3+F4: DAD stacked + DAD overlay (costat)
            dad_row = QHBoxLayout()
            dad_row.setSpacing(4)

            self.dad_figure = Figure(figsize=(5, 2.5), dpi=100)
            self.dad_figure.set_facecolor("#FAFAFA")
            self.dad_canvas = FigureCanvas(self.dad_figure)
            self.dad_canvas.setMinimumHeight(150)
            dad_row.addWidget(self.dad_canvas)

            self.dad_overlay_figure = Figure(figsize=(5, 3), dpi=100)
            self.dad_overlay_figure.set_facecolor("#FAFAFA")
            self.dad_overlay_canvas = FigureCanvas(self.dad_overlay_figure)
            self.dad_overlay_canvas.setMinimumHeight(200)
            dad_row.addWidget(self.dad_overlay_canvas)

            self._charts_content_layout.addLayout(dad_row)

            # F5: Timeout timeline (full width, al final)
            timeout_label = QLabel(
                "<b style='color:#555; font-size:11px;'>"
                "Distribució de timeouts TOC per mostra</b>"
            )
            self._charts_content_layout.addWidget(timeout_label)
            self.timeout_figure = Figure(figsize=(10, 1.2), dpi=100)
            self.timeout_figure.set_facecolor("#FAFAFA")
            self.timeout_canvas = FigureCanvas(self.timeout_figure)
            self.timeout_canvas.setMinimumHeight(80)
            self.timeout_canvas.setMaximumHeight(100)
            self._charts_content_layout.addWidget(self.timeout_canvas)

        charts_outer.addWidget(self._charts_content)
        layout.addWidget(self.charts_section)

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
        self._chart_regular = {}
        self._chart_blank = {}
        self._chart_control = {}
        self._chart_khp = {}
        self._chart_is_bp = False
        self._charts_initialized = False

        self.results_table.setRowCount(0)

        self.empty_state.setVisible(True)
        self.info_frame.setVisible(False)
        self.status_frame.setVisible(False)
        self.progress_frame.setVisible(False)
        self.progress_bar.setValue(0)
        self.results_frame.setVisible(False)
        self.charts_section.setVisible(False)
        self._charts_content.setVisible(True)
        self.analyze_btn.setEnabled(True)
        self.status_indicator.setText("")

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
            self._populate_charts(result)
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
        self.info_frame.setVisible(False)
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

        if not self.samples_grouped:
            self.status_indicator.setText(
                f"<span style='background-color: #d4edda; color: #155724; "
                f"padding: 4px 12px; border-radius: 12px; font-size: 11px;'>"
                f"Llest per analitzar</span>"
            )
        # Si ja hi ha resultats, status_indicator s'actualitza des de _populate_table
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

        if self.worker is not None:
            self.worker.wait()
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
        if self.worker is not None:
            self.worker.wait()
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
        self._populate_charts(result)

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
        """Omple la taula unificada amb els resultats (13 cols, selectors DOC/DAD independents).

        Filtra per F0 toggles:
        - Mostres (SAMPLE, PR): sempre visibles
        - Control (BLANK, CONTROL): segueix toggle "Control"
        - KHP: segueix toggle "KHP" (amagat per SEQ_CAL)
        """
        self.results_table.setRowCount(0)
        self._sample_row_map = {}
        n_ok, n_warning, n_error, n_light, n_khp, n_blank = 0, 0, 0, 0, 0, 0

        # F0 toggle state
        show_blank = (hasattr(self, '_cat_buttons')
                      and self._cat_buttons.get("blank")
                      and self._cat_buttons["blank"].isChecked())
        show_control = (hasattr(self, '_cat_buttons')
                        and self._cat_buttons.get("control")
                        and self._cat_buttons["control"].isChecked())
        show_khp = (hasattr(self, '_cat_buttons')
                    and self._cat_buttons.get("khp")
                    and self._cat_buttons["khp"].isChecked())
        is_seq_cal = getattr(self.main_window, '_is_seq_cal', False)

        # Separar mostres per tipologia
        sample_names = []   # SAMPLE (mostres reals)
        pr_names = []       # Patrons (PR_*)
        khp_names = []      # KHP (calibració)
        light_names = []    # CONTROL / Neteja (anàlisi lleugera)
        blank_names = []    # BLANK / MQ

        for name in self.samples_grouped.keys():
            sd = self.samples_grouped[name]
            st = sd.get("sample_type", "SAMPLE")
            if sd.get("analysis_type") == "khp":
                khp_names.append(name)
            elif st == "BLANK":
                blank_names.append(name)
            elif sd.get("analysis_type") == "light" or st == "CONTROL":
                light_names.append(name)
            elif st.startswith("PR"):
                pr_names.append(name)
            else:
                sample_names.append(name)

        # Ordenar per índex d'injecció (ordre cronològic al MasterFile)
        def _min_inj_index(name):
            reps = self.samples_grouped[name].get("replicas", {})
            indices = [r.get("injection_index", 999) for r in reps.values()
                       if r.get("injection_index") is not None]
            return min(indices) if indices else 999

        for lst in (sample_names, pr_names, blank_names, khp_names, light_names):
            lst.sort(key=_min_inj_index)

        # Decidir ordre segons botons agrupació
        by_type = getattr(self, '_group_mode', 0) == 1

        if by_type:
            # "Per tipus": separadors entre grups
            regular_names = []
            self._type_groups = []  # [(label, names)]
            if sample_names:
                self._type_groups.append(("MOSTRES", sample_names))
            if pr_names:
                self._type_groups.append(("PATRONS REFERÈNCIA", pr_names))
            for _, names in self._type_groups:
                regular_names.extend(names)
        else:
            # "Ordre injecció": tot barrejat per injection_index
            regular_names = sample_names + pr_names
            self._type_groups = None

        # --- Regular samples ---
        _type_group_idx = 0  # Tracking per separadors "Per tipus"
        _type_group_offset = 0
        for sample_name in regular_names:
            # Inserir separador si mode "Per tipus"
            if by_type and self._type_groups:
                while (_type_group_idx < len(self._type_groups) and
                       _type_group_offset >= len(self._type_groups[_type_group_idx][1])):
                    _type_group_idx += 1
                    _type_group_offset = 0
                if (_type_group_idx < len(self._type_groups) and
                        _type_group_offset == 0):
                    label = self._type_groups[_type_group_idx][0]
                    n_cols = self.results_table.columnCount()
                    sep_row = self.results_table.rowCount()
                    self.results_table.insertRow(sep_row)
                    sep_item = QTableWidgetItem(f"--- {label} ---")
                    sep_item.setFlags(Qt.ItemIsEnabled)
                    sep_font = QFont()
                    sep_font.setBold(True)
                    sep_item.setFont(sep_font)
                    sep_item.setForeground(QBrush(QColor("#2E86AB")))
                    self.results_table.setItem(sep_row, 0, sep_item)
                    self.results_table.setSpan(sep_row, 0, 1, n_cols)
                    sep_bg = QBrush(QColor("#EBF5FB"))
                    for c in range(n_cols):
                        item = self.results_table.item(sep_row, c)
                        if item is None:
                            item = QTableWidgetItem("")
                            self.results_table.setItem(sep_row, c, item)
                        item.setBackground(sep_bg)
                _type_group_offset += 1
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
            area_254 = (dad_areas.get("A254") or {}).get("total", 0)
            self.results_table.setItem(row, 8, QTableWidgetItem(
                f"{area_254:.1f}" if area_254 else "-"))

            # Col 9: SNR_254
            snr_info_dad = dad_rep.get("snr_info_dad") or {}
            snr_254 = (snr_info_dad.get("A254") or {}).get("snr", 0)
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

            # Count stats (blancs apart)
            sample_data_st = sample_data.get("sample_type", "SAMPLE")
            if sample_data_st == "BLANK":
                n_blank += 1
            elif status_color == COLOR_ERROR:
                n_error += 1
            elif status_color == COLOR_WARNING:
                n_warning += 1
            else:
                n_ok += 1

        # --- Separator + KHP STANDARDS (only if toggle active and not SEQ_CAL) ---
        if khp_names and show_khp and not is_seq_cal:
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
                area_doc = (areas.get("DOC") or {}).get("total", 0) if areas else 0
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
                    a254 = ((areas.get("254nm") or {}).get("total", 0) or
                            (areas.get("A254") or {}).get("total", 0))
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

        # --- Separator + BLANCS (MQ, H2O...) — only if toggle active ---
        if blank_names and show_blank:
            n_cols = self.results_table.columnCount()

            sep_row = self.results_table.rowCount()
            self.results_table.insertRow(sep_row)
            sep_item = QTableWidgetItem("--- BLANCS / MQ ---")
            sep_item.setFlags(Qt.ItemIsEnabled)
            sep_font = QFont()
            sep_font.setBold(True)
            sep_item.setFont(sep_font)
            sep_item.setForeground(QBrush(QColor("#7f8c8d")))
            self.results_table.setItem(sep_row, 0, sep_item)
            self.results_table.setSpan(sep_row, 0, 1, n_cols)
            sep_bg = QBrush(QColor("#EAECEE"))
            for c in range(n_cols):
                item = self.results_table.item(sep_row, c)
                if item is None:
                    item = QTableWidgetItem("")
                    self.results_table.setItem(sep_row, c, item)
                item.setBackground(sep_bg)

            for sample_name in blank_names:
                sample_data = self.samples_grouped[sample_name]
                row = self.results_table.rowCount()
                self.results_table.insertRow(row)
                self._sample_row_map[sample_name] = row

                replicas = sample_data.get("replicas") or {}
                comparison = sample_data.get("comparison") or {}
                recommendation = sample_data.get("recommendation") or {}
                selected = sample_data.get("selected") or {"doc": "1", "dad": "1"}
                quantification = sample_data.get("quantification") or {}

                doc_sel = selected.get("doc", sorted(replicas.keys())[0] if replicas else "1")
                dad_sel = selected.get("dad", doc_sel)
                doc_rep = replicas.get(doc_sel, {})
                dad_rep = replicas.get(dad_sel, {})

                # Col 0: Sample name
                item_name = QTableWidgetItem(sample_name)
                item_name.setData(Qt.UserRole, sample_name)
                self.results_table.setItem(row, 0, item_name)

                # Col 1-2: No selectors for blancs
                self.results_table.setItem(row, 1, QTableWidgetItem("-"))
                self.results_table.setItem(row, 2, QTableWidgetItem("-"))

                # Col 3: A_DOC
                areas = doc_rep.get("areas") or {}
                doc_areas = areas.get("DOC") or {}
                area_direct = doc_areas.get("total", 0) or doc_rep.get("area_total", 0)
                self.results_table.setItem(row, 3, QTableWidgetItem(
                    f"{area_direct:.0f}" if area_direct else "-"))

                # Col 4: ppm
                ppm = quantification.get("concentration_ppm")
                self.results_table.setItem(row, 4, QTableWidgetItem(
                    f"{ppm:.2f}" if ppm else "-"))

                # Col 5-6: UIB
                for c in (5, 6):
                    self.results_table.setItem(row, c, QTableWidgetItem("-"))

                # Col 7: SNR
                snr_info = doc_rep.get("snr_info") or {}
                snr = snr_info.get("snr_direct", 0) or doc_rep.get("snr", 0)
                snr_item = QTableWidgetItem(f"{snr:.0f}" if snr else "-")
                if snr and snr < 10:
                    snr_item.setForeground(QBrush(QColor(COLOR_ERROR)))
                elif snr and snr < 50:
                    snr_item.setForeground(QBrush(QColor(COLOR_WARNING)))
                self.results_table.setItem(row, 7, snr_item)

                # Col 8-12: No DAD, no R², no HCI
                for c in (8, 9, 10, 11, 12):
                    self.results_table.setItem(row, c, QTableWidgetItem("-"))

                # Col 13: Tipus
                type_item = QTableWidgetItem("Blanc")
                type_item.setForeground(QBrush(QColor("#7f8c8d")))
                self.results_table.setItem(row, 13, type_item)

                # Light grey background
                blank_bg = QBrush(QColor("#F4F6F6"))
                for c in range(n_cols):
                    item = self.results_table.item(row, c)
                    if item:
                        item.setBackground(blank_bg)

                n_blank += 1

        # --- Separator + Light samples (CONTROL) — only if toggle active ---
        if light_names and show_control:
            n_cols = self.results_table.columnCount()
            sep_row = self.results_table.rowCount()
            self.results_table.insertRow(sep_row)
            sep_item = QTableWidgetItem("--- NETEJA ---")
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

                # Col 13: Neteja
                type_item = QTableWidgetItem("Neteja")
                type_item.setForeground(QBrush(QColor("#888888")))
                self.results_table.setItem(row, 13, type_item)

                # Light grey background
                light_bg = QBrush(QColor("#F0F0F0"))
                for c in range(n_cols):
                    item = self.results_table.item(row, c)
                    if item:
                        item.setBackground(light_bg)

                n_light += 1

        # Update stats (unified at top, in status_indicator)
        total = n_ok + n_warning + n_error
        parts = [f"<b>{total}</b> mostres"]
        if n_blank > 0:
            parts.append(f"{n_blank} blancs")
        if n_khp > 0:
            parts.append(f"{n_khp} KHP")
        if n_light > 0:
            parts.append(f"{n_light} neteja")
        counts = " &middot; ".join(parts)

        status_parts = []
        status_parts.append(f"<span style='color:#27AE60'>\u25cf</span>&nbsp;{n_ok}")
        if n_warning > 0:
            status_parts.append(f"<span style='color:#F39C12'>\u25cf</span>&nbsp;{n_warning}")
        if n_error > 0:
            status_parts.append(f"<span style='color:#E74C3C'>\u25cf</span>&nbsp;{n_error}")
        status_str = " &nbsp;".join(status_parts)

        self.status_indicator.setText(
            f"<span style='background-color: #f8f9fa; color: #2c3e50; "
            f"padding: 4px 12px; border-radius: 12px; font-size: 11px;'>"
            f"{counts} &nbsp;|&nbsp; {status_str}</span>"
        )

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
        replica_warnings = []
        if comparison:
            for domain in ("doc", "dad"):
                replica_warnings.extend((comparison.get(domain) or {}).get("warnings", []))

        # Determine severity
        has_blocker = bool(classified["blocker"])
        has_warn = bool(classified["warning"] or classified["repaired"]
                        or (timeout_severity in ("WARNING", "CRITICAL"))
                        or replica_warnings)

        # Build concise status text
        n_blocker = len(classified["blocker"])
        n_warn = len(classified["warning"])
        n_repaired = len(classified["repaired"])
        # Repairable indicator
        can_repair = (sample_data and sample_data.get("repairable")
                      and not sample_data.get("repaired"))
        repair_icon = " 🔧" if can_repair else ""

        if has_blocker:
            status_color = COLOR_ERROR
            # Mostrar motiu concret del primer blocker
            first_blocker = classified["blocker"][0]
            b_code = first_blocker.get("code") if isinstance(first_blocker, dict) else str(first_blocker)
            b_entry = ANOMALY_CATALOG.get(b_code, {})
            b_label = b_entry.get("label", b_code)
            if n_blocker == 1:
                status_text = b_label + repair_icon
            else:
                status_text = f"{b_label} +{n_blocker - 1}" + repair_icon
        elif has_warn or n_repaired:
            status_color = COLOR_WARNING
            parts = []
            if n_warn:
                parts.append(f"{n_warn} avís" if n_warn == 1 else f"{n_warn} avisos")
            if n_repaired:
                parts.append(f"{n_repaired} reparat" if n_repaired == 1 else f"{n_repaired} reparats")
            if n_timeouts > 0:
                parts.append(f"{n_timeouts} timeout")
            status_text = " \u00b7 ".join(parts) + repair_icon
        else:
            status_color = COLOR_SUCCESS
            status_text = "OK"

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
                f"Timeouts Direct: {n_timeouts} ({timeout_severity}) — zones: {', '.join(zones) if zones else '?'}"
            )
            # UIB timeout propagat
            uib_ti = doc_rep_data.get("timeout_info_uib") or {}
            if uib_ti.get("n_timeouts", 0) > 0:
                uib_zones = uib_ti.get("zones", [])
                uib_in_peak = doc_rep_data.get("timeout_in_peak_uib", False)
                uib_tip = f"Timeouts UIB: {uib_ti['n_timeouts']} — zones: {', '.join(uib_zones) if uib_zones else '?'}"
                if uib_in_peak:
                    uib_tip += " — DINS DEL PIC UIB!"
                tooltip_parts.append(uib_tip)
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
    # Group mode change
    # ------------------------------------------------------------------

    def _style_group_btns(self):
        """Aplica estil als botons toggle d'agrupació."""
        for i, btn in enumerate(self._group_btns):
            if i == self._group_mode:
                btn.setStyleSheet(
                    "QPushButton { background-color: #2E86AB; color: white; "
                    "font-weight: bold; padding: 3px 12px; border-radius: 3px; "
                    "border: none; font-size: 10px; }"
                )
            else:
                btn.setStyleSheet(
                    "QPushButton { background-color: #e9ecef; color: #495057; "
                    "padding: 3px 12px; border-radius: 3px; "
                    "border: 1px solid #ced4da; font-size: 10px; }"
                    "QPushButton:hover { background-color: #dee2e6; }"
                )

    def _on_group_btn(self, idx):
        """Canvia el mode d'agrupació i reomple la taula."""
        self._group_mode = idx
        for i, btn in enumerate(self._group_btns):
            btn.setChecked(i == idx)
        self._style_group_btns()
        if self.samples_grouped:
            self._populate_table()

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
        """Actualitza columnes DOC (3-7) quan canvia la r\u00e8plica DOC."""
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
        area_254 = (dad_areas.get("A254") or {}).get("total", 0)
        item_8 = self.results_table.item(row, 8)
        if item_8:
            item_8.setText(f"{area_254:.1f}" if area_254 else "-")

        # Col 9: SNR_254
        snr_info_dad = dad_rep.get("snr_info_dad") or {}
        snr_254 = (snr_info_dad.get("A254") or {}).get("snr", 0)
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

            # Respectar exclusió de quantificació
            if sample_data.get("skip_quantification"):
                sample_data["quantification"] = {
                    "concentration_ppm": None,
                    "concentration_ppm_direct": None,
                    "concentration_ppm_uib": None,
                    "valid": False,
                    "reason": sample_data["quantification"].get("reason",
                              "Exclosa de quantificació") if sample_data.get("quantification") else
                              "Exclosa de quantificació"
                }
                return

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
        """Genera el report PDF d'anàlisi (cridat des del wizard header)."""
        processed_data = self.main_window.processed_data
        if not processed_data:
            QMessageBox.warning(self, "Avís", "No hi ha dades processades.")
            return

        seq_path = processed_data.get("seq_path", "")
        if not seq_path:
            QMessageBox.warning(self, "Avís", "No s'ha trobat el path de la seqüència.")
            return

        try:
            from generate_analysis_report import generate_analysis_report

            # Passar dades en memòria (inclou seleccions actuals de l'usuari)
            report_data = dict(processed_data)
            report_data["samples_grouped"] = self.samples_grouped

            result = generate_analysis_report(
                seq_path, analysis_data=report_data
            )

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
            QMessageBox.critical(
                self, "Error",
                f"Error generant el report:\n{str(e)}"
            )

    # ------------------------------------------------------------------
    # Charts section (migrated from ReviewSummaryPanel)
    # ------------------------------------------------------------------

    def _populate_charts(self, processed_data):
        """Prepara dades pels gràfics i mostra la secció."""
        if not HAS_MATPLOTLIB or not processed_data:
            return

        samples_grouped = processed_data.get("samples_grouped", {})
        method = processed_data.get("method", "COLUMN")
        is_bp = method.upper() == "BP"

        regular = {}
        blank = {}
        control = {}
        khp = {}
        for name, data in samples_grouped.items():
            if data.get("analysis_type") == "khp":
                khp[name] = data
            elif data.get("sample_type") == "BLANK":
                blank[name] = data
            elif (data.get("sample_type") == "CONTROL"
                  or data.get("analysis_type") == "light"):
                control[name] = data
            else:
                regular[name] = data

        self._chart_regular = regular
        self._chart_blank = blank
        self._chart_control = control
        self._chart_khp = khp
        self._chart_is_bp = is_bp

        try:
            self._plot_timeout_chart(processed_data, is_bp)
        except Exception as e:
            logger.error(f"Error plotting timeout chart: {e}")

        self._build_sample_checkboxes(regular, blank, control, khp)
        self.charts_section.setVisible(True)

        # Charts sempre visibles — dibuixar directament
        self._charts_initialized = True
        self._redraw_charts()

    def _build_sample_checkboxes(self, regular, blank, control, khp):
        """Registra mostres per categoria (sense checkboxes individuals)."""
        self._sample_checkboxes = []
        for name in sorted(regular.keys()):
            self._sample_checkboxes.append((None, name, "sample"))
        for name in sorted(blank.keys()):
            self._sample_checkboxes.append((None, name, "blank"))
        for name in sorted(control.keys()):
            self._sample_checkboxes.append((None, name, "control"))
        for name in sorted(khp.keys()):
            self._sample_checkboxes.append((None, name, "khp"))

        self._cat_counts = {
            "sample": len(regular),
            "blank": len(blank),
            "control": len(control),
            "khp": len(khp),
        }
        self._update_cat_btn_styles()

    def _update_cat_btn_styles(self):
        """Actualitza estil dels botons toggle de categoria."""
        if not hasattr(self, '_cat_buttons'):
            return
        STYLES = {
            "sample":  ("#2E86AB", "#fff"),
            "blank":   ("#95a5a6", "#fff"),
            "control": ("#888",    "#fff"),
            "khp":     ("#1565C0", "#fff"),
        }
        for cat_key, btn in self._cat_buttons.items():
            color, fg = STYLES.get(cat_key, ("#666", "#fff"))
            count = self._cat_counts.get(cat_key, 0)
            base_label = btn.text().split('(')[0].strip()
            btn.setText(f"{base_label} ({count})" if count else base_label)
            if btn.isChecked():
                btn.setStyleSheet(
                    f"QPushButton {{ background: {color}; color: {fg};"
                    f" font-size: 10px; font-weight: bold; padding: 3px 10px;"
                    f" border: none; border-radius: 3px; }}"
                    f"QPushButton:hover {{ opacity: 0.9; }}"
                )
            else:
                btn.setStyleSheet(
                    f"QPushButton {{ background: #f0f0f0; color: {color};"
                    f" font-size: 10px; padding: 3px 10px;"
                    f" border: 1px solid #ddd; border-radius: 3px; }}"
                    f"QPushButton:hover {{ background: #e4e4e4; }}"
                )

    def _on_cat_toggle(self):
        """Un botó de categoria ha canviat — actualitza taula i gràfics."""
        self._update_cat_btn_styles()
        if self.samples_grouped:
            self._populate_table()
        self._redraw_charts()

    def _on_wl_changed(self):
        """Longitud d'ona DAD seleccionada ha canviat — redibuixar gràfics DAD."""
        if self._charts_initialized:
            checked = self._get_checked_samples()
            reg = {k: v for k, v in checked.items()
                   if v.get("sample_type") not in ("BLANK", "CONTROL")
                   and v.get("analysis_type") not in ("light",)}
            light = {k: v for k, v in checked.items()
                     if v.get("sample_type") in ("BLANK", "CONTROL")
                     or v.get("analysis_type") == "light"}
            try:
                self._plot_dad_chart(reg, light)
                self._plot_dad_overlay(reg, light)
            except Exception as e:
                logger.error(f"Error redrawing DAD charts: {e}")

    def _get_selected_wl(self):
        """Retorna la longitud d'ona DAD seleccionada (ex: '254')."""
        if hasattr(self, '_wl_combo'):
            return self._wl_combo.currentData() or "254"
        return "254"

    def _get_checked_samples(self):
        """Retorna dict {name: data} de mostres de categories actives."""
        all_data = {}
        all_data.update(self._chart_regular)
        all_data.update(self._chart_blank)
        all_data.update(self._chart_control)
        all_data.update(getattr(self, '_chart_khp', {}))

        active_cats = {cat for cat, btn in self._cat_buttons.items() if btn.isChecked()}
        checked = {}
        for _cb, name, cat in self._sample_checkboxes:
            if cat in active_cats and name in all_data:
                checked[name] = all_data[name]
        return checked

    def _redraw_charts(self):
        """Redibuixa els 4 gràfics amb les mostres seleccionades."""
        if not HAS_MATPLOTLIB:
            return
        checked = self._get_checked_samples()
        reg = {k: v for k, v in checked.items()
               if v.get("sample_type") not in ("BLANK", "CONTROL")
               and v.get("analysis_type") not in ("light",)}
        light = {k: v for k, v in checked.items()
                 if v.get("sample_type") in ("BLANK", "CONTROL")
                 or v.get("analysis_type") == "light"}
        is_bp = getattr(self, '_chart_is_bp', False)
        try:
            self._plot_doc_chart(reg, light, is_bp)
            self._plot_dad_chart(reg, light)
            self._plot_doc_overlay(reg, light, is_bp)
            self._plot_dad_overlay(reg, light)
        except Exception as e:
            logger.error(f"Error redrawing charts: {e}")

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
                ("BP_PEAK", 0, 5, "#E74C3C"),
                ("BP_TAIL", 5, 10, "#F39C12"),
            ]
            x_max = 12
        else:
            from hpsec_config import ConfigManager
            cfg = ConfigManager()
            fractions = cfg.get_all_fractions(mode="COLUMN")
            max_dur = cfg.get("chromatogram", "max_duration_min", default=78.65)
            zones = []
            first_start = 10.8
            if fractions:
                first_start = fractions[0][1]["start"]
            if first_start > 0:
                zones.append(("RUN_START", 0, first_start, "#95a5a6"))
            for name, frac in fractions:
                color = FRACTION_COLORS.get(name, "#95a5a6")
                zones.append((name, frac["start"], frac["end"], color))
            last_end = zones[-1][2] if zones else 70
            if max_dur > last_end:
                zones.append(("POST_RUN", last_end, max_dur, "#d5dbdb"))
            x_max = max_dur

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

        ax.set_xlim(0, x_max)
        ax.set_yticks([])
        ax.set_xlabel("Temps (min)", fontsize=7)
        ax.tick_params(axis='x', labelsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        self.timeout_figure.tight_layout(pad=0.3)
        self.timeout_canvas.draw()

    def _plot_doc_chart(self, regular, light, is_bp):
        """Gràfic DOC: barres stacked per fraccions (COLUMN) o simples (BP)."""
        self.doc_figure.clear()
        ax = self.doc_figure.add_subplot(111)

        names = []
        fractions_data = {f: [] for f in FRACTION_ORDER}
        ppm_values = []

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
            fractions_data["BioP"][-1] = area

        if not names:
            ax.text(0.5, 0.5, "Sense dades", ha='center', va='center', transform=ax.transAxes)
            self.doc_canvas.draw()
            return

        x = np.arange(len(names))
        bar_width = 0.7

        if is_bp:
            totals = [sum(fractions_data[f][i] for f in FRACTION_ORDER) for i in range(len(names))]
            colors = ['#95a5a6' if i >= light_start else '#3498DB' for i in range(len(names))]
            ax.bar(x, totals, bar_width, color=colors, edgecolor='white', linewidth=0.5)
        else:
            bottom = np.zeros(len(names))
            for frac in FRACTION_ORDER:
                values = np.array(fractions_data[frac], dtype=float)
                colors = []
                for i in range(len(names)):
                    if i >= light_start:
                        colors.append('#B0B0B0')
                    else:
                        colors.append(FRACTION_COLORS[frac])
                ax.bar(x, values, bar_width, bottom=bottom, color=colors,
                       edgecolor='white', linewidth=0.3, label=frac)
                bottom += values

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
        """Gràfic DAD per mostra: barres stacked per fraccions (COLUMN) o simples (BP)."""
        self.dad_figure.clear()
        ax = self.dad_figure.add_subplot(111)
        wl = self._get_selected_wl()
        wl_key = f"A{wl}"
        is_bp = getattr(self, '_chart_is_bp', False)

        names = []
        fractions_data = {f: [] for f in FRACTION_ORDER}

        for name in sorted(regular.keys()):
            data = regular[name]
            selected = data.get("selected") or {}
            sel = selected.get("dad", selected.get("doc", "1"))
            rep = (data.get("replicas") or {}).get(sel, {})
            areas = (rep.get("areas") or {}).get(wl_key, {})
            names.append(name)
            for frac in FRACTION_ORDER:
                fractions_data[frac].append(areas.get(frac, 0))

        light_start = len(names)
        for name in sorted(light.keys()):
            data = light[name]
            sel = (data.get("selected") or {}).get("doc", "1")
            rep = (data.get("replicas") or {}).get(sel, {})
            area = ((rep.get("areas") or {}).get(wl_key) or {}).get("total", 0)
            names.append(name)
            for frac in FRACTION_ORDER:
                fractions_data[frac].append(0)
            fractions_data["BioP"][-1] = area

        if not names:
            ax.text(0.5, 0.5, "Sense dades", ha='center', va='center', transform=ax.transAxes)
            self.dad_canvas.draw()
            return

        x = np.arange(len(names))
        bar_width = 0.7

        if is_bp:
            totals = [sum(fractions_data[f][i] for f in FRACTION_ORDER) for i in range(len(names))]
            colors = ['#95a5a6' if i >= light_start else '#E74C3C' for i in range(len(names))]
            ax.bar(x, totals, bar_width, color=colors, edgecolor='white', linewidth=0.5)
        else:
            bottom = np.zeros(len(names))
            for frac in FRACTION_ORDER:
                values = np.array(fractions_data[frac], dtype=float)
                colors = []
                for i in range(len(names)):
                    if i >= light_start:
                        colors.append('#B0B0B0')
                    else:
                        colors.append(FRACTION_COLORS[frac])
                ax.bar(x, values, bar_width, bottom=bottom, color=colors,
                       edgecolor='white', linewidth=0.3, label=frac)
                bottom += values

            handles = [
                ax.bar(0, 0, color=FRACTION_COLORS[f], label=f)[0]
                for f in FRACTION_ORDER
            ]
            ax.legend(handles=handles, labels=FRACTION_ORDER,
                      loc='upper right', fontsize=8, framealpha=0.8, ncol=len(FRACTION_ORDER))

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel(f"Àrea {wl_key}", fontsize=9)
        ax.set_title(f"{wl_key} per mostra (fraccions)", fontsize=10, fontweight='bold')
        ax.tick_params(axis='y', labelsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        self.dad_figure.tight_layout()
        self.dad_canvas.draw()

    @staticmethod
    def _get_line_style(data):
        """Retorna l'estil de línia segons el tipus de mostra."""
        st = data.get("sample_type", "SAMPLE")
        at = data.get("analysis_type", "")
        if at == "khp":
            return '--'  # KHP: discontínua
        elif st == "BLANK" or at == "light":
            return ':'   # Blanc/Control: punts
        elif st.startswith("PR") or st == "CONTROL":
            return '-.'  # PR/Control: punt-ratlla
        return '-'       # Mostres: sòlida

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
            ls = self._get_line_style(data)
            if t is not None and y is not None and len(t) > 0:
                ax.plot(t, y, label=name, linewidth=1.4, alpha=0.7,
                        color=cmap(i), linestyle=ls)

        ax.set_xlim(0, 12 if is_bp else 70)
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
        """Cromatogrames DAD superposats (longitud d'ona seleccionada)."""
        self.dad_overlay_figure.clear()
        ax = self.dad_overlay_figure.add_subplot(111)
        wl = self._get_selected_wl()

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

        is_bp = getattr(self, '_chart_is_bp', False)

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

            t_col = None
            for c in df_dad.columns:
                if 'time' in str(c).lower():
                    t_col = c
                    break
            wl_col = None
            for c in df_dad.columns:
                if wl in str(c):
                    wl_col = c
                    break

            ls = self._get_line_style(data)
            if t_col is not None and wl_col is not None:
                ax.plot(df_dad[t_col], df_dad[wl_col], label=name,
                        linewidth=1.4, alpha=0.7, color=cmap(i), linestyle=ls)

        ax.set_xlim(0, 12 if is_bp else 70)
        ax.set_xlabel("Temps (min)", fontsize=9)
        ax.set_ylabel(f"A{wl} (mAU)", fontsize=9)
        ax.set_title(f"Cromatogrames A{wl} superposats", fontsize=10, fontweight='bold')
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

    def save_charts(self, seq_path):
        """Guarda els 5 gràfics a SEQ/CHECK/plots/."""
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

