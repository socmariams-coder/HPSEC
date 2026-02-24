"""
HPSEC Suite - Global Calibration Panel
========================================

Panell únic de calibració: una sola vista sense sub-pestanyes.

Contingut:
- Toolbar compacte: mode/senyal/model/repair/PDF
- Selector SEQ_CAL amb checkboxes
- Splitter horitzontal: taula (esquerra) + scatter+comparació (dreta)
- Secció "Aplicar calibració" collapsable
- Barra de progrés per CalSeqWorker

Les SEQ_CAL arriben des del Dashboard (sense passar pel wizard).
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QComboBox, QMessageBox, QSplitter, QRadioButton, QButtonGroup,
    QSizePolicy, QCheckBox, QTabWidget, QListWidget,
    QListWidgetItem, QFrame, QProgressBar, QDateEdit, QScrollArea,
    QGridLayout,
)
from PySide6.QtCore import Qt, Signal, QThread, QDate
from PySide6.QtGui import QFont, QColor

from pathlib import Path
import sys
import os
import json
import logging

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from hpsec_calibrate import (
    get_active_global_calibration,
    load_khp_history,
    fit_calibration_from_history,
    load_calibration_reference,
    compute_calibration_fingerprint,
)

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# WORKER: Processament SEQ_CAL (import + calibrate en thread)
# =============================================================================

class CalSeqWorker(QThread):
    """Worker per processar una SEQ_CAL nova (import + calibrate)."""
    progress = Signal(int, str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, seq_path, config=None):
        super().__init__()
        self.seq_path = seq_path
        self.config = config

    def run(self):
        try:
            from hpsec_import import (
                load_manifest, import_from_manifest, import_sequence,
                ensure_data_loaded,
            )
            from hpsec_calibrate import calibrate_from_import

            def progress_cb(pct, msg):
                self.progress.emit(int(pct), msg)

            seq_name = os.path.basename(self.seq_path)
            progress_cb(0, f"Processant {seq_name}...")

            manifest = load_manifest(self.seq_path)
            if manifest:
                progress_cb(5, "Importat des del manifest...")
                imported_data = import_from_manifest(
                    self.seq_path, manifest=manifest,
                    config=self.config,
                    progress_callback=lambda p, m: progress_cb(5 + int(p * 0.3), m),
                    load_data=True,
                )
            else:
                progress_cb(5, "Importat des del sistema de fitxers...")
                imported_data = import_sequence(
                    self.seq_path,
                    config=self.config,
                    progress_callback=lambda p, m: progress_cb(5 + int(p * 0.3), m),
                )

            if not imported_data or not imported_data.get("success"):
                self.error.emit(
                    f"Error importat {seq_name}: "
                    + str(imported_data.get("errors", ["Desconegut"]))
                )
                return

            if imported_data.get("data_deferred"):
                progress_cb(35, "Carregant senyals des del disc...")
                ensure_data_loaded(
                    imported_data,
                    config=self.config,
                    progress_callback=lambda p, m: progress_cb(35 + int(p * 0.15), m),
                )

            progress_cb(50, "Calibrant KHP...")
            calib_result = calibrate_from_import(
                imported_data,
                config=self.config,
                progress_callback=lambda p, m: progress_cb(50 + int(p * 0.45), m),
            )

            progress_cb(95, "Finalitzant...")
            result = {
                "success": True,
                "seq_name": seq_name,
                "seq_path": self.seq_path,
                "imported_data": imported_data,
                "calib_result": calib_result,
            }
            self.finished.emit(result)

        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n{traceback.format_exc()}")


# =============================================================================
# PANELL PRINCIPAL
# =============================================================================

class GlobalCalibrationPanel(QWidget):
    """Panell de calibració global — vista única.

    Layout:
      [toolbar: mode | senyal | model | repair | PDF]
      [SEQ selector: checkboxes compactes, max-height 70]
      [splitter horitzontal]
        [ESQUERRA: taula punts calibració]
        [DRETA: scatter + comparació + resultats]
      [apply section + progress]
    """

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self._all_calibrations = []
        self._cal_entries = []       # Només _CAL
        self._grouped_by_seq = {}
        self._filtered_entries = []
        self._last_result = None
        self._loading = False
        self._active_seq_path = None
        self._result_cache = {}
        self._calib_results = {}     # {seq_name: calib_result} per popup cromatograma
        self._setup_ui()

    # =====================================================================
    # UI SETUP
    # =====================================================================

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(6)

        # --- Títol ---
        header = QHBoxLayout()
        title = QLabel("Calibració Global")
        title.setFont(QFont("Segoe UI", 15, QFont.Bold))
        header.addWidget(title)
        header.addStretch()
        self._report_btn = QPushButton("📄 PDF")
        self._report_btn.setToolTip("Generar informe calibració (PDF)")
        self._report_btn.setStyleSheet(
            "QPushButton { background: #2980B9; color: white; border: none; "
            "border-radius: 4px; padding: 5px 14px; font-weight: bold; }"
            "QPushButton:hover { background: #3498DB; }"
        )
        self._report_btn.clicked.connect(self._on_generate_report)
        header.addWidget(self._report_btn)
        layout.addLayout(header)

        # --- Toolbar: mode / senyal / model / repair ---
        toolbar = QHBoxLayout()
        toolbar.setSpacing(10)

        toolbar.addWidget(QLabel("Mode:"))
        self._mode_group = QButtonGroup(self)
        self._radio_column = QRadioButton("COLUMN")
        self._radio_bp = QRadioButton("BP")
        self._radio_column.setChecked(True)
        self._mode_group.addButton(self._radio_column, 0)
        self._mode_group.addButton(self._radio_bp, 1)
        toolbar.addWidget(self._radio_column)
        toolbar.addWidget(self._radio_bp)

        sep1 = QFrame()
        sep1.setFrameShape(QFrame.VLine)
        sep1.setStyleSheet("color: #CCC;")
        toolbar.addWidget(sep1)

        toolbar.addWidget(QLabel("Senyal:"))
        self._signal_combo = QComboBox()
        self._signal_combo.addItems(["direct", "uib", "254"])
        self._signal_combo.setFixedWidth(80)
        toolbar.addWidget(self._signal_combo)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.VLine)
        sep2.setStyleSheet("color: #CCC;")
        toolbar.addWidget(sep2)

        toolbar.addWidget(QLabel("Model:"))
        self._model_group = QButtonGroup(self)
        self._radio_intercept = QRadioButton("Intercept")
        self._radio_origin = QRadioButton("Origen")
        self._radio_intercept.setChecked(True)
        self._model_group.addButton(self._radio_intercept, 0)
        self._model_group.addButton(self._radio_origin, 1)
        toolbar.addWidget(self._radio_intercept)
        toolbar.addWidget(self._radio_origin)

        sep3 = QFrame()
        sep3.setFrameShape(QFrame.VLine)
        sep3.setStyleSheet("color: #CCC;")
        toolbar.addWidget(sep3)

        self._repair_chk = QCheckBox("Àrea reparada")
        self._repair_chk.setToolTip(
            "Usar àrea reparada (paràbola) en lloc de l'original\n"
            "per entrades amb cim irregular detectat"
        )
        toolbar.addWidget(self._repair_chk)

        toolbar.addStretch()

        # Calibració vigent (compacte)
        self._cur_cal_label = QLabel("")
        self._cur_cal_label.setStyleSheet("color: #555; font-size: 11px;")
        toolbar.addWidget(self._cur_cal_label)

        layout.addLayout(toolbar)

        # --- SEQ selector ---
        seq_row = QHBoxLayout()
        seq_row.addWidget(QLabel("SEQ_CAL:"))
        self._seq_list = QListWidget()
        self._seq_list.setFlow(QListWidget.LeftToRight)
        self._seq_list.setWrapping(True)
        self._seq_list.setMaximumHeight(50)
        self._seq_list.setSpacing(2)
        self._seq_list.setStyleSheet(
            "QListWidget { border: 1px solid #DEE2E6; border-radius: 3px; background: #FAFAFA; }"
            "QListWidget::item { padding: 2px 6px; }"
        )
        self._seq_list.itemChanged.connect(self._on_seq_selection_changed)
        seq_row.addWidget(self._seq_list, 1)

        self._btn_reprocess = QPushButton("↻")
        self._btn_reprocess.setToolTip("Re-importar i calibrar la SEQ seleccionada")
        self._btn_reprocess.setFixedSize(28, 28)
        self._btn_reprocess.clicked.connect(self._on_reprocess_seq)
        seq_row.addWidget(self._btn_reprocess)
        layout.addLayout(seq_row)

        # --- Connexions selectors ---
        self._mode_group.buttonClicked.connect(self._on_params_changed)
        self._signal_combo.currentIndexChanged.connect(self._on_params_changed)
        self._model_group.buttonClicked.connect(self._on_params_changed)
        self._repair_chk.toggled.connect(self._on_params_changed)

        # --- Splitter principal ---
        self._splitter = QSplitter(Qt.Horizontal)

        # ESQUERRA: Taula de punts
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 4, 0)
        left_layout.setSpacing(4)

        self._points_table = QTableWidget()
        self._pt_cols = ["☑", "SEQ", "ppm", "Vol", "µg", "Àrea", "RF", "SNR", "t_ret", "Estat"]
        self._points_table.setColumnCount(len(self._pt_cols))
        self._points_table.setHorizontalHeaderLabels(self._pt_cols)

        hdr = self._points_table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.Fixed)
        self._points_table.setColumnWidth(0, 32)
        hdr.setSectionResizeMode(1, QHeaderView.Stretch)
        for i in range(2, len(self._pt_cols)):
            hdr.setSectionResizeMode(i, QHeaderView.ResizeToContents)

        self._points_table.setAlternatingRowColors(True)
        self._points_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._points_table.setSelectionMode(QTableWidget.SingleSelection)
        self._points_table.verticalHeader().setVisible(False)
        self._points_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._points_table.doubleClicked.connect(self._on_table_double_click)
        self._points_table.setStyleSheet(
            "QTableWidget { font-size: 12px; gridline-color: #EEE; }"
            "QHeaderView::section { background: #F5F6FA; font-weight: bold; "
            "border: none; border-bottom: 2px solid #DEE2E6; padding: 4px; }"
        )
        left_layout.addWidget(self._points_table)

        self._splitter.addWidget(left)

        # DRETA: Scatter + Comparació
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 0, 0, 0)
        right_layout.setSpacing(4)

        self._figure = Figure(figsize=(6, 4.5), dpi=100)
        self._canvas = FigureCanvas(self._figure)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_layout.addWidget(self._canvas, 3)

        # Resultats compactes
        self._results_label = QLabel("")
        self._results_label.setWordWrap(True)
        self._results_label.setTextFormat(Qt.RichText)
        self._results_label.setStyleSheet(
            "QLabel { background: #F0F4F8; border: 1px solid #D0D5DD; "
            "border-radius: 4px; padding: 6px 10px; font-size: 12px; }"
        )
        right_layout.addWidget(self._results_label)

        # Comparació
        self._comparison_label = QLabel("")
        self._comparison_label.setWordWrap(True)
        self._comparison_label.setTextFormat(Qt.RichText)
        self._comparison_label.setStyleSheet(
            "QLabel { background: #FAFAFA; border: 1px solid #DEE2E6; "
            "border-radius: 4px; padding: 6px 10px; font-size: 11px; }"
        )
        right_layout.addWidget(self._comparison_label)

        self._splitter.addWidget(right)
        self._splitter.setStretchFactor(0, 2)
        self._splitter.setStretchFactor(1, 3)
        layout.addWidget(self._splitter, 1)

        # --- Secció Aplicar ---
        self._apply_frame = QFrame()
        self._apply_frame.setStyleSheet(
            "QFrame#applyFrame { background: #F0FFF4; border: 1px solid #A3D9A5; "
            "border-radius: 6px; }"
        )
        self._apply_frame.setObjectName("applyFrame")
        self._apply_frame.setVisible(False)
        apply_layout = QHBoxLayout(self._apply_frame)
        apply_layout.setContentsMargins(12, 8, 12, 8)
        apply_layout.setSpacing(12)

        apply_layout.addWidget(QLabel("Vigent des de:"))
        self._apply_valid_from = QDateEdit()
        self._apply_valid_from.setCalendarPopup(True)
        self._apply_valid_from.setDate(QDate.currentDate())
        self._apply_valid_from.setDisplayFormat("yyyy-MM-dd")
        apply_layout.addWidget(self._apply_valid_from)

        self._apply_retroactive_chk = QCheckBox("Retroactiu")
        self._apply_retroactive_chk.setToolTip(
            "Requantifica SEQs processades amb els nous RF/intercept"
        )
        self._apply_retroactive_chk.toggled.connect(self._on_retroactive_toggled)
        apply_layout.addWidget(self._apply_retroactive_chk)

        self._retro_info = QLabel("")
        self._retro_info.setStyleSheet("color: #666; font-size: 10px;")
        apply_layout.addWidget(self._retro_info)

        apply_layout.addStretch()

        self._apply_btn = QPushButton("✓ Aplicar Nova Calibració")
        self._apply_btn.setStyleSheet(
            "QPushButton { background: #27AE60; color: white; border: none; "
            "border-radius: 5px; padding: 8px 20px; font-weight: bold; font-size: 12px; }"
            "QPushButton:hover { background: #2ECC71; }"
            "QPushButton:disabled { background: #BDC3C7; }"
        )
        self._apply_btn.clicked.connect(self._on_apply_calibration)
        apply_layout.addWidget(self._apply_btn)

        self._apply_status = QLabel("")
        self._apply_status.setTextFormat(Qt.RichText)
        apply_layout.addWidget(self._apply_status)

        layout.addWidget(self._apply_frame)

        # --- Progress ---
        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setFixedHeight(18)
        self._progress_bar.setStyleSheet(
            "QProgressBar { border: 1px solid #bdc3c7; border-radius: 3px; text-align: center; }"
            "QProgressBar::chunk { background: #2980B9; border-radius: 2px; }"
        )
        layout.addWidget(self._progress_bar)

        self._progress_label = QLabel("")
        self._progress_label.setVisible(False)
        self._progress_label.setStyleSheet("color: #2980B9; font-style: italic; font-size: 11px;")
        layout.addWidget(self._progress_label)

        # Worker
        self._cal_worker = None

        # Llista per retroactiu
        self._retro_seq_checkboxes = []
        self._retro_popup = None

    # =====================================================================
    # HELPERS
    # =====================================================================

    def _get_mode(self):
        return "COLUMN" if self._radio_column.isChecked() else "BP"

    def _get_signal(self):
        return self._signal_combo.currentText()

    def _get_model(self):
        return "intercept" if self._radio_intercept.isChecked() else "origin"

    def _use_repaired(self):
        return self._repair_chk.isChecked()

    def _get_area(self, cal, signal=None):
        """Retorna àrea segons senyal i toggle reparació."""
        sig = (signal or self._get_signal()).lower()
        use_rep = self._use_repaired()

        if sig == 'uib':
            return cal.get('area_u', 0) or 0
        elif sig == '254':
            return cal.get('area_254', 0) or cal.get('a254_area', 0) or 0
        else:
            area = cal.get('area', 0) or 0
            if use_rep:
                area_rep = cal.get('area_repaired')
                if area_rep and area_rep > 0:
                    return area_rep
            return area

    # =====================================================================
    # DATA LOADING
    # =====================================================================

    def showEvent(self, event):
        super().showEvent(event)
        self._load_all_data()

    def _load_all_data(self):
        """Carrega KHP_History i filtra _CAL."""
        self._all_calibrations = load_khp_history(None)
        self._cal_entries = [
            e for e in self._all_calibrations
            if "_CAL" in e.get("seq_name", "").upper()
        ]
        self._update_current_cal_label()
        self._loading = True
        self._populate_seq_list()
        self._loading = False
        self._refresh_points_and_recalculate()

    def _update_current_cal_label(self):
        """Mostra calibració vigent al toolbar."""
        cal = get_active_global_calibration()
        if not cal:
            self._cur_cal_label.setText("Cap calibració activa")
            return
        mode = self._get_mode().lower()
        signal = self._get_signal().lower()
        rf_data = cal.get('rf_mass_cal', {})
        rf_val = None
        if isinstance(rf_data, dict):
            sig_rf = rf_data.get(signal, {})
            if isinstance(sig_rf, dict):
                rf_val = sig_rf.get(mode)
        int_data = cal.get('intercept', 0)
        int_val = 0
        if isinstance(int_data, dict):
            sig_int = int_data.get(signal, {})
            if isinstance(sig_int, dict):
                int_val = sig_int.get(mode, 0)
        elif isinstance(int_data, (int, float)):
            int_val = int_data
        if rf_val is not None:
            self._cur_cal_label.setText(f"Vigent: RF={rf_val:.0f}  int={int_val:.0f}")
        else:
            self._cur_cal_label.setText("—")

    # =====================================================================
    # SEQ_CAL DES DE DASHBOARD
    # =====================================================================

    def load_seq_cal(self, seq_path):
        """Carrega una SEQ_CAL des del Dashboard."""
        self._active_seq_path = seq_path
        seq_name = os.path.basename(seq_path)
        logger.info(f"load_seq_cal: {seq_name}")

        if seq_path in self._result_cache:
            logger.info(f"  Reutilitzant resultat cachejat per {seq_name}")
            self._on_worker_finished(self._result_cache[seq_path], from_cache=True)
            return

        self._load_all_data()
        logger.info(f"  Processant SEQ_CAL '{seq_name}'...")
        self._start_cal_worker(seq_path)

    def _start_cal_worker(self, seq_path):
        """Llança CalSeqWorker."""
        if self._cal_worker and self._cal_worker.isRunning():
            logger.warning("CalSeqWorker ja en execució")
            return
        seq_name = os.path.basename(seq_path)
        self._progress_bar.setVisible(True)
        self._progress_bar.setValue(0)
        self._progress_label.setVisible(True)
        self._progress_label.setText(f"Processant {seq_name}...")

        self._cal_worker = CalSeqWorker(seq_path)
        self._cal_worker.progress.connect(self._on_worker_progress)
        self._cal_worker.finished.connect(self._on_worker_finished)
        self._cal_worker.error.connect(self._on_worker_error)
        self._cal_worker.start()

    def _on_worker_progress(self, pct, msg):
        self._progress_bar.setValue(pct)
        self._progress_label.setText(msg)

    def _on_worker_finished(self, result, from_cache=False):
        self._progress_bar.setVisible(False)
        self._progress_label.setVisible(False)

        seq_name = result.get("seq_name", "")
        seq_path = result.get("seq_path", "")
        if not from_cache and seq_path:
            self._result_cache[seq_path] = result

        calib_result = result.get("calib_result")
        if calib_result:
            self._calib_results[seq_name] = calib_result

        if not from_cache:
            self._load_all_data()

        # Pre-seleccionar la SEQ
        self._pre_select_seq(seq_name)

        if self.main_window:
            self.main_window.set_status(f"SEQ_CAL {seq_name} processada", 5000)

    def _on_worker_error(self, error_msg):
        self._progress_bar.setVisible(False)
        self._progress_label.setVisible(False)
        logger.error(f"CalSeqWorker error: {error_msg}")
        self._comparison_label.setText(
            f"<span style='color:#E74C3C'>❌ Error: {error_msg[:200]}</span>"
        )
        QMessageBox.critical(self, "Error", f"Error processant SEQ_CAL:\n\n{error_msg[:500]}")

    # =====================================================================
    # SEQ SELECTOR
    # =====================================================================

    def _populate_seq_list(self):
        self._seq_list.blockSignals(True)
        self._seq_list.clear()

        mode = self._get_mode()
        self._grouped_by_seq = {}
        for entry in self._cal_entries:
            if entry.get('mode', '').upper() != mode.upper():
                continue
            seq_name = entry.get('seq_name', 'Desconegut')
            self._grouped_by_seq.setdefault(seq_name, []).append(entry)

        for seq_name in sorted(self._grouped_by_seq.keys()):
            entries = self._grouped_by_seq[seq_name]
            n = len(entries)
            concs = sorted(set(e.get('conc_ppm', 0) for e in entries))
            tag = f"{min(concs):g}–{max(concs):g}" if concs else "?"
            item = QListWidgetItem(f"{seq_name} ({n}pt, {tag}ppm)")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            item.setData(Qt.UserRole, seq_name)
            self._seq_list.addItem(item)

        self._seq_list.blockSignals(False)

    def _pre_select_seq(self, seq_name):
        """Pre-selecciona una SEQ des del Dashboard."""
        # Auto-detectar mode
        seq_modes = set()
        for entry in self._cal_entries:
            if entry.get("seq_name", "") == seq_name:
                seq_modes.add(entry.get("mode", "").upper())
        if "BP" in seq_modes and "COLUMN" not in seq_modes:
            self._radio_bp.setChecked(True)
        elif "COLUMN" in seq_modes:
            self._radio_column.setChecked(True)

        self._loading = True
        self._populate_seq_list()
        for i in range(self._seq_list.count()):
            item = self._seq_list.item(i)
            item.setCheckState(
                Qt.Checked if item.data(Qt.UserRole) == seq_name else Qt.Unchecked
            )
        self._loading = False
        self._refresh_points_and_recalculate()

    def _get_selected_seq_names(self):
        return [
            self._seq_list.item(i).data(Qt.UserRole)
            for i in range(self._seq_list.count())
            if self._seq_list.item(i).checkState() == Qt.Checked
        ]

    # =====================================================================
    # PARAMS / EVENTS
    # =====================================================================

    def _on_params_changed(self, *args):
        if self._loading:
            return
        self._loading = True
        self._update_current_cal_label()
        self._populate_seq_list()
        self._loading = False
        self._refresh_points_and_recalculate()

    def _on_seq_selection_changed(self, *args):
        if not self._loading:
            self._refresh_points_and_recalculate()

    def _on_point_toggled(self, *args):
        if not self._loading:
            self._recalculate_regression()

    # =====================================================================
    # TAULA DE PUNTS
    # =====================================================================

    def _refresh_points_and_recalculate(self):
        self._refresh_points_table()
        self._recalculate_regression()

    def _refresh_points_table(self):
        """Pobla la taula amb punts de les SEQs seleccionades."""
        signal = self._get_signal()
        selected_seqs = self._get_selected_seq_names()

        self._points_table.setRowCount(0)
        self._points_table.blockSignals(True)

        self._filtered_entries = []
        for seq_name in selected_seqs:
            for entry in self._grouped_by_seq.get(seq_name, []):
                self._filtered_entries.append(entry)

        self._points_table.setRowCount(len(self._filtered_entries))

        for row, cal in enumerate(self._filtered_entries):
            conc = cal.get('conc_ppm', 0)
            vol = cal.get('volume_uL', 0)
            ug_doc = conc * vol / 1000.0 if conc > 0 and vol > 0 else 0
            area = self._get_area(cal, signal)
            rf_mass = area / ug_doc if ug_doc > 0 else 0
            snr = cal.get('snr', 0) or 0
            t_ret = cal.get('t_retention', 0) or 0
            qs = cal.get('quality_score', 0) or 0
            is_outlier = cal.get('is_outlier', False)
            not_valid = not cal.get('valid_for_calibration', True)
            bad = is_outlier or not_valid or conc <= 0 or area <= 0 or qs >= 100

            # Estat
            if not_valid or qs >= 100:
                estat = "INVALID"
            elif qs > 50:
                estat = "CHECK"
            elif qs > 20:
                estat = "INFO"
            else:
                estat = "OK"

            # Col 0: Checkbox
            chk = QCheckBox()
            chk.setChecked(not bad)
            chk.stateChanged.connect(self._on_point_toggled)
            chk_w = QWidget()
            chk_l = QHBoxLayout(chk_w)
            chk_l.addWidget(chk)
            chk_l.setAlignment(Qt.AlignCenter)
            chk_l.setContentsMargins(0, 0, 0, 0)
            self._points_table.setCellWidget(row, 0, chk_w)

            # Cols 1-9
            items = [
                (cal.get('seq_name', '').replace('_SEQ', ''), None),
                (f"{conc:g}", None),
                (f"{vol:.0f}", None),
                (f"{ug_doc:.3f}", None),
                (f"{area:.1f}", "#E67E22" if self._use_repaired() and cal.get('area_repaired') else None),
                (f"{rf_mass:.0f}", None),
                (f"{snr:.0f}" if snr > 0 else "—",
                 "#dc3545" if 0 < snr < 10 else None),
                (f"{t_ret:.1f}" if t_ret > 0 else "—", None),
                (estat,
                 "#dc3545" if estat == "INVALID" else
                 "#ffc107" if estat in ("CHECK", "INFO") else "#28a745"),
            ]

            for col, (text, color) in enumerate(items):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                if bad:
                    item.setForeground(QColor("#CCC"))
                elif color:
                    item.setForeground(QColor(color))
                self._points_table.setItem(row, col + 1, item)

            # Tooltip ric
            tip = self._build_tooltip(cal, signal, ug_doc, area, rf_mass)
            for col in range(1, len(self._pt_cols)):
                it = self._points_table.item(row, col)
                if it:
                    it.setToolTip(tip)

        self._points_table.blockSignals(False)

    def _build_tooltip(self, cal, signal, ug_doc, area, rf_mass):
        """Construeix tooltip complet per una fila."""
        parts = [
            f"SEQ: {cal.get('seq_name', '')}",
            f"Data: {str(cal.get('date', ''))[:10]}",
            f"ppm={cal.get('conc_ppm', 0):g}  vol={cal.get('volume_uL', 0):.0f}µL  µg={ug_doc:.3f}",
            f"Àrea={area:.1f}  RF={rf_mass:.0f}",
        ]
        if cal.get('fwhm_doc'):
            parts.append(f"FWHM={cal['fwhm_doc']:.2f} min")
        if cal.get('symmetry'):
            parts.append(f"Sim={cal['symmetry']:.2f}")
        if cal.get('area_repaired') and cal['area_repaired'] != cal.get('area', 0):
            parts.append(f"Àrea reparada={cal['area_repaired']:.1f}")
        sel = cal.get('selection') or {}
        if sel:
            parts.append(f"Selecció: {sel.get('method', '?')} ({sel.get('reason', '')})")
        issues = cal.get('quality_issues', [])
        if issues:
            parts.append("---")
            for iss in issues[:5]:
                parts.append(f"  • {iss}")
        return "\n".join(parts)

    # =====================================================================
    # POPUP CROMATOGRAMA
    # =====================================================================

    def _on_table_double_click(self, index):
        """Doble clic: obre popup amb cromatograma interactiu."""
        row = index.row()
        if row < 0 or row >= len(self._filtered_entries):
            return
        cal = self._filtered_entries[row]

        # Buscar dades riques (cromatogrames) des de calib_result cachejat
        seq_name = cal.get('seq_name', '')
        calib_result = self._calib_results.get(seq_name)
        entry_with_replicas = self._find_rich_entry(cal, calib_result)

        if entry_with_replicas and entry_with_replicas.get('replicas'):
            try:
                from gui.widgets.seq_cal_regression_widget import KHPDetailDialog
                dlg = KHPDetailDialog(entry_with_replicas, parent=self)
                dlg.exec()
            except Exception as e:
                logger.warning(f"Error obrint popup: {e}")
                self._show_basic_detail(cal)
        else:
            self._show_basic_detail(cal)

    def _find_rich_entry(self, cal, calib_result):
        """Busca l'entrada rica (amb cromatogrames) de calib_result."""
        if not calib_result:
            return None
        conc = cal.get('conc_ppm', 0)
        for key in ('calibrations_direct', 'calibrations_uib'):
            cal_list = calib_result.get(key, [])
            if isinstance(cal_list, list):
                for group in cal_list:
                    if isinstance(group, dict) and abs(group.get('conc_ppm', 0) - conc) < 0.01:
                        return group
            elif isinstance(cal_list, dict):
                for group_name, group in cal_list.items():
                    if isinstance(group, dict) and abs(group.get('conc_ppm', 0) - conc) < 0.01:
                        return group
        return None

    def _show_basic_detail(self, cal):
        """Detall bàsic (sense cromatograma) — per entrades sense calib_result."""
        text = (
            f"<b>{cal.get('seq_name', '')}</b> — "
            f"KHP {cal.get('conc_ppm', 0):g} ppm · {cal.get('volume_uL', 0):.0f} µL<br><br>"
            f"Àrea: {cal.get('area', 0):.1f}<br>"
            f"RF_mass: {cal.get('rf_mass', 0):.1f}<br>"
            f"t_ret: {cal.get('t_retention', 0):.2f} min<br>"
            f"SNR: {cal.get('snr', 0):.0f}<br>"
        )
        if cal.get('area_repaired'):
            text += f"<br>Àrea reparada: {cal['area_repaired']:.1f}"
        issues = cal.get('quality_issues', [])
        if issues:
            text += "<br><br><b>Issues:</b><br>" + "<br>".join(f"• {i}" for i in issues[:5])
        text += "<br><br><i>Doble clic disponible amb cromatogrames després de '↻ Reprocessar'</i>"
        QMessageBox.information(self, "Detall KHP", text)

    # =====================================================================
    # REPROCESS
    # =====================================================================

    def _on_reprocess_seq(self):
        selected = self._get_selected_seq_names()
        if not selected:
            QMessageBox.information(self, "Info", "Selecciona una SEQ_CAL primer.")
            return
        seq_name = selected[0]
        seq_path = None
        for entry in self._cal_entries:
            if entry.get('seq_name') == seq_name:
                seq_path = entry.get('seq_path')
                break
        if not seq_path or not os.path.isdir(seq_path):
            QMessageBox.warning(self, "Error", f"Directori no trobat:\n{seq_path}")
            return
        # Invalidar cache
        if seq_path in self._result_cache:
            del self._result_cache[seq_path]
        self._start_cal_worker(seq_path)

    # =====================================================================
    # REGRESSIÓ
    # =====================================================================

    def _get_selected_calibrations(self):
        """Retorna entrades amb checkbox marcat."""
        filtered = self._filtered_entries
        selected = []
        for row in range(self._points_table.rowCount()):
            chk_w = self._points_table.cellWidget(row, 0)
            if chk_w:
                chk = chk_w.findChild(QCheckBox)
                if chk and chk.isChecked() and row < len(filtered):
                    # Si repair toggle, substituir àrea al vol
                    entry = dict(filtered[row])  # còpia
                    if self._use_repaired() and entry.get('area_repaired'):
                        entry['area'] = entry['area_repaired']
                    selected.append(entry)
        return selected

    def _recalculate_regression(self):
        selected = self._get_selected_calibrations()
        model = self._get_model()
        signal = self._get_signal()
        mode = self._get_mode()

        result = fit_calibration_from_history(
            selected, mode=mode, signal=signal, model=model
        )
        self._last_result = result

        # Actualitzar resultats
        if result.get('success'):
            rf = result['rf_mass_cal']
            intercept = result['intercept']
            r2 = result['r2']
            n = result['n_points']
            rms = result.get('residuals_rms', 0)
            self._results_label.setText(
                f"<b>RF</b>={rf:.1f} · <b>Intercept</b>={intercept:.1f} · "
                f"<b>R²</b>={r2:.4f} · <b>n</b>={n} · "
                f"<b>RMS</b>={rms:.2f}" if rms else
                f"<b>RF</b>={rf:.1f} · <b>Intercept</b>={intercept:.1f} · "
                f"<b>R²</b>={r2:.4f} · <b>n</b>={n}"
            )
        else:
            self._results_label.setText(
                f"<i style='color:#999'>Sense prou punts per regressió "
                f"(n={result.get('n_points', 0)})</i>"
            )

        self._update_preview_graph(result)
        self._update_comparison(result)
        self._update_apply_visibility()

    # =====================================================================
    # GRÀFIC
    # =====================================================================

    def _update_preview_graph(self, result):
        self._figure.clear()
        mode = self._get_mode()
        signal = self._get_signal()

        has_data = result.get('success') and result.get('points')
        if has_data:
            ax_main = self._figure.add_axes([0.12, 0.35, 0.85, 0.60])
            ax_res = self._figure.add_axes([0.12, 0.08, 0.85, 0.22])
        else:
            ax_main = self._figure.add_subplot(111)
            ax_res = None

        # Punts seleccionats vs exclosos
        selected = self._get_selected_calibrations()
        sel_keys = set()
        for c in selected:
            sel_keys.add((c.get('seq_name', ''), c.get('conc_ppm', 0),
                          c.get('volume_uL', 0)))

        all_entries = []
        for sn in self._get_selected_seq_names():
            for e in self._grouped_by_seq.get(sn, []):
                all_entries.append(e)

        x_sel, y_sel, x_exc, y_exc = [], [], [], []
        for cal in all_entries:
            conc = cal.get('conc_ppm', 0)
            vol = cal.get('volume_uL', 0)
            if conc <= 0 or vol <= 0:
                continue
            ug = conc * vol / 1000.0
            area = self._get_area(cal, signal)
            if area <= 0:
                continue
            key = (cal.get('seq_name', ''), conc, vol)
            if key in sel_keys:
                x_sel.append(ug)
                y_sel.append(area)
            else:
                x_exc.append(ug)
                y_exc.append(area)

        if x_sel:
            ax_main.scatter(x_sel, y_sel, c='#2196F3', s=50, zorder=5,
                            label='Inclosos', edgecolors='white', linewidth=0.5)
        if x_exc:
            ax_main.scatter(x_exc, y_exc, c='#CCC', s=35, zorder=4, marker='x',
                            label='Exclosos', linewidths=1.5)

        # Recta nova
        if result.get('success'):
            rf = result['rf_mass_cal']
            intercept = result['intercept']
            r2 = result['r2']
            all_x = x_sel + x_exc
            if all_x:
                x_line = np.linspace(0, max(all_x) * 1.1, 100)
                y_line = rf * x_line + intercept
                eq = f"y = {rf:.0f}x + {intercept:.0f}" if intercept != 0 else f"y = {rf:.0f}x"
                ax_main.plot(x_line, y_line, '#E74C3C', linewidth=2,
                             label=f"Nova ({eq})")

        # Recta vigent
        cal_actual = get_active_global_calibration()
        if cal_actual and (x_sel or x_exc):
            rf_data = cal_actual.get('rf_mass_cal', {})
            int_data = cal_actual.get('intercept', 0)
            cur_rf = None
            cur_int = 0
            if isinstance(rf_data, dict):
                sig_rf = rf_data.get(signal, {})
                if isinstance(sig_rf, dict):
                    cur_rf = sig_rf.get(mode.lower())
            if isinstance(int_data, dict):
                sig_int = int_data.get(signal, {})
                if isinstance(sig_int, dict):
                    cur_int = sig_int.get(mode.lower(), 0)
            elif isinstance(int_data, (int, float)):
                cur_int = int_data
            if cur_rf is not None:
                all_x = x_sel + x_exc
                x_line = np.linspace(0, max(all_x) * 1.1, 100)
                y_cur = cur_rf * x_line + cur_int
                ax_main.plot(x_line, y_cur, '--', color='#999', linewidth=1.5,
                             alpha=0.7, label=f"Vigent (RF={cur_rf:.0f})")

        ax_main.set_ylabel("Àrea", fontsize=10)
        ax_main.set_title(f"Regressió — {mode} {signal}", fontsize=11, fontweight='bold')
        ax_main.legend(fontsize=7, loc='upper left')
        ax_main.grid(True, alpha=0.2)
        if not has_data:
            ax_main.set_xlabel("µg DOC", fontsize=10)

        # Residuals
        if ax_res and has_data:
            points = result['points']
            y_res = [p.get('residual', 0) for p in points]
            colors = ['#dc3545' if abs(r) > 2 * result.get('residuals_rms', 999)
                       else '#2196F3' for r in y_res]
            ax_res.bar(range(len(y_res)), y_res, color=colors, alpha=0.7)
            ax_res.axhline(0, color='black', linewidth=0.5)
            rms = result.get('residuals_rms', 0)
            if rms:
                ax_res.axhline(rms, color='#aaa', linewidth=0.8, linestyle='--')
                ax_res.axhline(-rms, color='#aaa', linewidth=0.8, linestyle='--')
            ax_res.set_ylabel("Res.", fontsize=8)
            ax_res.set_xlabel("Punt", fontsize=8)
            ax_res.tick_params(labelsize=7)
            ax_res.grid(True, alpha=0.15)

        self._figure.tight_layout()
        self._canvas.draw()

    # =====================================================================
    # COMPARACIÓ
    # =====================================================================

    def _update_comparison(self, result):
        if not result.get('success'):
            self._comparison_label.setText("")
            return

        mode = self._get_mode().lower()
        signal = self._get_signal().lower()
        cal = get_active_global_calibration()
        if not cal:
            self._comparison_label.setText(
                f"Nova: RF={result['rf_mass_cal']:.1f}, int={result['intercept']:.1f}, "
                f"R²={result['r2']:.4f}"
            )
            return

        # Valors actuals
        rf_data = cal.get('rf_mass_cal', {})
        cur_rf = None
        if isinstance(rf_data, dict):
            sig_rf = rf_data.get(signal, {})
            if isinstance(sig_rf, dict):
                cur_rf = sig_rf.get(mode)
        int_data = cal.get('intercept', 0)
        cur_int = 0
        if isinstance(int_data, dict):
            sig_int = int_data.get(signal, {})
            if isinstance(sig_int, dict):
                cur_int = sig_int.get(mode, 0)
        elif isinstance(int_data, (int, float)):
            cur_int = int_data

        new_rf = result['rf_mass_cal']
        new_int = result['intercept']

        rows = []
        if cur_rf and cur_rf > 0:
            pct_rf = (new_rf - cur_rf) / cur_rf * 100
            c = "#dc3545" if abs(pct_rf) > 15 else "#28a745" if abs(pct_rf) < 5 else "#ffc107"
            rows.append(f"RF: {cur_rf:.0f} → <b>{new_rf:.0f}</b> "
                        f"(<span style='color:{c}'>{pct_rf:+.1f}%</span>)")
        else:
            rows.append(f"RF: — → <b>{new_rf:.0f}</b>")

        delta_int = new_int - cur_int
        rows.append(f"Int: {cur_int:.0f} → <b>{new_int:.0f}</b> ({delta_int:+.0f})")
        rows.append(f"R²: <b>{result['r2']:.4f}</b>, n={result['n_points']}")

        # Impacte a 1 ppm
        if cur_rf and cur_rf > 0:
            vol_ex = 100 if mode == "bp" else 400
            area_ex = cur_rf * 1.0 * vol_ex / 1000 + cur_int
            ppm_old = max(0, area_ex - cur_int) * 1000 / (cur_rf * vol_ex)
            ppm_new = max(0, area_ex - new_int) * 1000 / (new_rf * vol_ex) if new_rf > 0 else 0
            if ppm_old > 0:
                pct = (ppm_new - ppm_old) / ppm_old * 100
                rows.append(f"<i>Impacte 1ppm ({vol_ex}µL): {ppm_old:.3f}→{ppm_new:.3f} ({pct:+.1f}%)</i>")

        self._comparison_label.setText(" · ".join(rows[:2]) + "<br>" + " · ".join(rows[2:]))

    # =====================================================================
    # APLICAR CALIBRACIÓ
    # =====================================================================

    def _update_apply_visibility(self):
        visible = (
            self._last_result is not None
            and self._last_result.get('success')
            and self._last_result.get('r2', 0) > 0
        )
        self._apply_frame.setVisible(visible)

    def _on_retroactive_toggled(self, checked):
        if checked:
            self._count_retro_seqs()

    def _count_retro_seqs(self):
        """Compta SEQs disponibles per requantificació."""
        mode = self._get_mode().upper()
        from hpsec_config import get_config
        cfg = get_config()
        data_folder = cfg.get("paths", "data_folder")
        if not data_folder or not os.path.isdir(data_folder):
            self._retro_info.setText("Carpeta dades no trobada")
            return

        count = 0
        self._retro_seq_data = []
        for seq_dir in sorted(os.listdir(data_folder)):
            seq_path = os.path.join(data_folder, seq_dir)
            if not os.path.isdir(seq_path) or "_CAL" in seq_dir.upper():
                continue
            json_path = os.path.join(seq_path, "CHECK", "data", "analysis.json")
            if not os.path.exists(json_path):
                continue
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if data.get("method", "COLUMN").upper() == mode:
                    count += 1
                    self._retro_seq_data.append((seq_dir, json_path))
            except Exception:
                pass

        self._retro_info.setText(f"{count} SEQs {mode} disponibles")

    def _on_apply_calibration(self):
        if not self._last_result or not self._last_result.get('success'):
            return

        rf_new = self._last_result['rf_mass_cal']
        intercept_new = self._last_result['intercept']
        r2 = self._last_result['r2']
        n_pts = self._last_result['n_points']
        mode = self._get_mode()
        signal = self._get_signal()
        is_bp = mode.upper() == "BP"
        retroactive = self._apply_retroactive_chk.isChecked()
        retro_data = getattr(self, '_retro_seq_data', []) if retroactive else []

        # Validacions
        if r2 < 0.95:
            resp = QMessageBox.warning(
                self, "R² baix",
                f"R² = {r2:.4f} < 0.95. Continuar?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if resp != QMessageBox.Yes:
                return

        valid_from = self._apply_valid_from.date().toString("yyyy-MM-dd")
        msg = (
            f"Aplicar nova calibració:\n\n"
            f"  Mode: {mode}, Senyal: {signal}\n"
            f"  RF: {rf_new:.1f}, Intercept: {intercept_new:.1f}\n"
            f"  R²: {r2:.6f}, n: {n_pts}\n"
            f"  Vigent des de: {valid_from}\n"
        )
        if retro_data:
            msg += f"\n  Retroactiu: {len(retro_data)} SEQs\n"
        msg += "\nConfirmar?"

        if QMessageBox.question(self, "Confirmar", msg,
                                QMessageBox.Yes | QMessageBox.No,
                                QMessageBox.No) != QMessageBox.Yes:
            return

        self._apply_btn.setEnabled(False)
        self._apply_status.setText("Aplicant...")

        try:
            import copy
            from hpsec_calibrate import (
                add_calibration, get_active_global_calibration,
                get_rf_mass_cal, get_calibration_intercept,
                requantify_analysis_json, compute_calibration_fingerprint
            )

            current_cal = get_active_global_calibration()
            if current_cal:
                rf_values = copy.deepcopy(dict(current_cal.get('rf_mass_cal', {})))
                intercept_values = current_cal.get('intercept', {})
                if isinstance(intercept_values, dict):
                    intercept_values = copy.deepcopy(dict(intercept_values))
                else:
                    intercept_values = {"direct": {"column": 0, "bp": 0}}
            else:
                rf_values = {"direct": {"column": 0, "bp": 0}, "uib": {"column": 0, "bp": 0}}
                intercept_values = {"direct": {"column": 0, "bp": 0}, "uib": {"column": 0, "bp": 0}}

            mode_key = "bp" if is_bp else "column"
            if isinstance(rf_values.get(signal), dict):
                rf_values[signal][mode_key] = rf_new
            else:
                rf_values[signal] = {mode_key: rf_new}
            if isinstance(intercept_values.get(signal), dict):
                intercept_values[signal][mode_key] = intercept_new
            else:
                intercept_values[signal] = {mode_key: intercept_new}

            selected_seqs = self._get_selected_seq_names()
            source = {
                "type": "SEQ_CAL",
                "description": f"Regressió from {', '.join(selected_seqs)}",
                "seq_references": selected_seqs,
                "mode": mode,
            }
            reg_data = dict(self._last_result)
            reg_data['mode'] = mode
            reg_data['signal'] = signal
            reg_data['model'] = self._get_model()

            cal_id = add_calibration(
                rf_mass_cal_values=rf_values,
                source=source,
                valid_from=valid_from,
                r2=r2,
                n_points=n_pts,
                reason=f"SEQ_CAL panell: {', '.join(selected_seqs)}",
                intercept_values=intercept_values,
                regression_data=reg_data,
            )

            if not cal_id:
                raise RuntimeError("add_calibration ha retornat None")

            # Retroactiu
            retro_ok = 0
            if retro_data:
                self._apply_status.setText(f"Requantificant {len(retro_data)} SEQs...")
                new_cal = get_active_global_calibration()
                rf_col = get_rf_mass_cal(new_cal, signal=signal, mode="column")
                int_col = get_calibration_intercept(new_cal, signal=signal, mode="column")
                rf_bp = get_rf_mass_cal(new_cal, signal=signal, mode="bp")
                int_bp = get_calibration_intercept(new_cal, signal=signal, mode="bp")

                for seq_dir, json_path in retro_data:
                    try:
                        rq = requantify_analysis_json(
                            json_path,
                            new_rf_direct=rf_col, new_intercept_direct=int_col,
                            new_rf_bp=rf_bp, new_intercept_bp=int_bp,
                        )
                        if rq.get('success'):
                            retro_ok += 1
                    except Exception:
                        pass

            status = f"<span style='color:#27AE60'>✓ Calibració {cal_id} aplicada</span>"
            if retro_data:
                status += f" · Requantificades: {retro_ok}/{len(retro_data)}"
            self._apply_status.setText(status)
            self._apply_btn.setEnabled(False)

            self._update_current_cal_label()

            # Refrescar dashboard
            if hasattr(self.main_window, 'dashboard_panel') and self.main_window.dashboard_panel:
                try:
                    self.main_window.dashboard_panel.refresh_sequences()
                except Exception:
                    pass

        except Exception as e:
            logger.error(f"Error aplicant calibració: {e}")
            self._apply_status.setText(f"<span style='color:#E74C3C'>Error: {e}</span>")
            self._apply_btn.setEnabled(True)

    # =====================================================================
    # INFORME PDF
    # =====================================================================

    def _on_generate_report(self):
        try:
            from hpsec_reports import generate_calibration_report
            cal = get_active_global_calibration()
            if not cal:
                QMessageBox.warning(self, "Avís", "No hi ha calibració activa.")
                return
            pdf_path = generate_calibration_report(cal)
            if pdf_path and os.path.exists(pdf_path):
                QMessageBox.information(self, "PDF generat", f"Informe:\n{pdf_path}")
                try:
                    os.startfile(pdf_path)
                except AttributeError:
                    import subprocess
                    subprocess.Popen(['xdg-open', pdf_path])
            else:
                QMessageBox.warning(self, "Error", "No s'ha pogut generar l'informe.")
        except Exception as e:
            logger.error(f"Error informe: {e}")
            QMessageBox.critical(self, "Error", f"Error:\n{e}")
