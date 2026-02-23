"""
HPSEC Suite - Global Calibration Panel
========================================

Panell de calibració global amb dues vistes:
- Tab 0: Recta de Calibració — des de SEQ_CAL dedicades (inclou aplicar)
- Tab 1: Control de Qualitat — Levey-Jennings per KHP de producció

Les SEQ_CAL arriben directament des del Dashboard (sense passar pel wizard).
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGroupBox,
    QGridLayout, QTableWidget, QTableWidgetItem, QHeaderView,
    QComboBox, QMessageBox, QSplitter, QRadioButton, QButtonGroup,
    QSizePolicy, QCheckBox, QTabWidget, QListWidget,
    QListWidgetItem, QFrame, QProgressBar, QDateEdit, QScrollArea
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
    """Worker per processar una SEQ_CAL nova (import + calibrate).

    Reutilitza el pipeline existent:
    1. Carrega manifest o importa des de zero
    2. ensure_data_loaded (si deferred)
    3. calibrate_from_import → register_calibration a KHP_History
    """
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

            # Pas 1: Import
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

            # Pas 1b: ensure_data_loaded si deferred
            if imported_data.get("data_deferred"):
                progress_cb(35, "Carregant senyals des del disc...")
                ensure_data_loaded(
                    imported_data,
                    config=self.config,
                    progress_callback=lambda p, m: progress_cb(35 + int(p * 0.15), m),
                )

            # Pas 2: Calibrate (internament registra a KHP_History)
            progress_cb(50, "Calibrant KHP...")
            calib_result = calibrate_from_import(
                imported_data,
                config=self.config,
                progress_callback=lambda p, m: progress_cb(50 + int(p * 0.45), m),
            )

            progress_cb(95, "Finalitzant...")

            # Preparar resultat
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


class GlobalCalibrationPanel(QWidget):
    """Panell de calibració global: consulta de calibracions vigents i historial.

    Les accions (aplicar nova calibració) es fan des del wizard (CalibratePanel).
    Aquest panell és de consulta i previsualització.
    """

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self._all_calibrations = []
        self._active_seq_path = None  # SEQ_CAL activa (des de Dashboard)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)

        # Títol
        title = QLabel("Calibració Global")
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        layout.addWidget(title)

        subtitle = QLabel(
            "Gestió de rf_mass_cal i intercept — "
            "Recta des de SEQ_CAL + Control de Qualitat de producció"
        )
        subtitle.setFont(QFont("Segoe UI", 9))
        subtitle.setStyleSheet("color: #666;")
        layout.addWidget(subtitle)

        # Botó generar informe PDF
        report_row = QHBoxLayout()
        report_row.addStretch()
        self._report_btn = QPushButton("📄 Generar Informe Calibració (PDF)")
        self._report_btn.setStyleSheet("""
            QPushButton {
                background-color: #2980B9; color: white;
                border: none; border-radius: 6px;
                padding: 8px 20px; font-size: 12px; font-weight: bold;
            }
            QPushButton:hover { background-color: #3498DB; }
        """)
        self._report_btn.clicked.connect(self._on_generate_report)
        report_row.addWidget(self._report_btn)
        report_row.addStretch()
        layout.addLayout(report_row)

        # Barra de progrés (per CalSeqWorker)
        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setStyleSheet("""
            QProgressBar { border: 1px solid #bdc3c7; border-radius: 4px;
                           text-align: center; height: 22px; }
            QProgressBar::chunk { background-color: #2980B9; border-radius: 3px; }
        """)
        layout.addWidget(self._progress_bar)

        self._progress_label = QLabel("")
        self._progress_label.setVisible(False)
        self._progress_label.setStyleSheet("color: #2980B9; font-style: italic;")
        layout.addWidget(self._progress_label)

        # Tabs
        self.tabs = QTabWidget()
        self.cal_view = CalibrationLineView(self)
        self.qc_view = QCMonitorView(self)
        self.tabs.addTab(self.cal_view, "📐 Recta de Calibració")
        self.tabs.addTab(self.qc_view, "📊 Control de Qualitat")
        layout.addWidget(self.tabs, 1)

        # Worker (un sol actiu)
        self._cal_worker = None

    def showEvent(self, event):
        super().showEvent(event)
        self._load_all_data()

    def _load_all_data(self):
        """Carrega KHP_History i distribueix a les dues vistes."""
        self._all_calibrations = load_khp_history(None)

        # Separar CAL vs producció per convenció _CAL al nom
        cal_entries = []
        prod_entries = []
        for entry in self._all_calibrations:
            seq_name = entry.get("seq_name", "")
            if "_CAL" in seq_name.upper():
                cal_entries.append(entry)
            else:
                prod_entries.append(entry)

        self.cal_view.set_data(cal_entries)
        self.qc_view.set_data(prod_entries)

    def load_seq_cal(self, seq_path):
        """Carrega una SEQ_CAL des del Dashboard.

        Comprova si KHP_History ja té entrades per aquesta SEQ.
        Si SÍ: carrega directament i pre-selecciona.
        Si NO: llança worker per processar (Commit 2).
        """
        self._active_seq_path = seq_path
        seq_name = os.path.basename(seq_path)

        logger.info(f"load_seq_cal: {seq_name}")

        # Recarregar totes les dades de KHP_History
        self._load_all_data()

        # Comprovar si ja hi ha entrades per aquesta SEQ a KHP_History
        has_entries = any(
            entry.get("seq_name", "") == seq_name
            for entry in self._all_calibrations
        )

        if has_entries:
            # Ja processada: pre-seleccionar la SEQ al CalibrationLineView
            logger.info(f"  SEQ_CAL '{seq_name}' ja processada, pre-seleccionant")
            self.tabs.setCurrentIndex(0)  # Tab Recta de Calibració
            self.cal_view.pre_select_seq(seq_name)
        else:
            # No processada: llançar worker per importar + calibrar
            logger.info(f"  SEQ_CAL '{seq_name}' NO processada, processant...")
            self.tabs.setCurrentIndex(0)
            self.cal_view.show_processing_message(seq_name)
            self._start_cal_worker(seq_path)

    def _start_cal_worker(self, seq_path):
        """Llança CalSeqWorker per importar i calibrar una SEQ_CAL."""
        if self._cal_worker and self._cal_worker.isRunning():
            logger.warning("CalSeqWorker ja en execució, ignorant nova petició")
            return

        seq_name = os.path.basename(seq_path)

        # Mostrar progrés
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
        """Actualitza barra de progrés."""
        self._progress_bar.setValue(pct)
        self._progress_label.setText(msg)

    def _on_worker_finished(self, result):
        """Worker completat: recarregar dades i pre-seleccionar."""
        self._progress_bar.setVisible(False)
        self._progress_label.setVisible(False)

        seq_name = result.get("seq_name", "")
        logger.info(f"CalSeqWorker completat per {seq_name}")

        # Guardar calib_result per diagnòstic (cromatogrames, mètriques riques)
        calib_result = result.get("calib_result")
        if calib_result:
            self.cal_view.set_active_calib_result(seq_name, calib_result)

        # Recarregar KHP_History (ara tindrà les noves entrades)
        self._load_all_data()

        # Pre-seleccionar la SEQ processada
        self.cal_view.pre_select_seq(seq_name)

        # Notificació
        if self.main_window:
            self.main_window.set_status(f"SEQ_CAL {seq_name} processada", 5000)

    def _on_worker_error(self, error_msg):
        """Error al processar la SEQ_CAL."""
        self._progress_bar.setVisible(False)
        self._progress_label.setVisible(False)

        logger.error(f"CalSeqWorker error: {error_msg}")

        # Mostrar error a la comparació
        self.cal_view.comparison_label.setText(
            f"<div style='text-align:center; padding:20px;'>"
            f"<span style='font-size:14px; color:#E74C3C;'>"
            f"❌ Error processant SEQ_CAL</span><br><br>"
            f"<span style='color:#666;'>{error_msg[:200]}</span></div>"
        )

        QMessageBox.critical(
            self, "Error processant SEQ_CAL",
            f"Error al processar la seqüència de calibració:\n\n{error_msg[:500]}"
        )

    def _on_generate_report(self):
        """Genera informe PDF de la calibració activa."""
        try:
            from hpsec_reports import generate_calibration_report

            cal = get_active_global_calibration()
            if not cal:
                QMessageBox.warning(self, "Avís", "No hi ha calibració activa.")
                return

            if not cal.get('regression_data'):
                QMessageBox.information(
                    self, "Info",
                    "La calibració activa no té dades de regressió emmagatzemades.\n"
                    "L'informe es generarà amb les dades disponibles\n"
                    "(pàgines 1, 3, 4 i 5 — sense scatter de regressió)."
                )

            pdf_path = generate_calibration_report(cal)
            if pdf_path and os.path.exists(pdf_path):
                QMessageBox.information(
                    self, "Informe generat",
                    f"Informe de calibració generat:\n{pdf_path}"
                )
                # Obrir el PDF
                try:
                    os.startfile(pdf_path)
                except AttributeError:
                    import subprocess
                    subprocess.Popen(['xdg-open', pdf_path])
            else:
                QMessageBox.warning(self, "Error", "No s'ha pogut generar l'informe.")
        except Exception as e:
            logger.error(f"Error generant informe calibració: {e}")
            QMessageBox.critical(self, "Error", f"Error generant informe:\n{e}")


# =============================================================================
# VISTA 1: RECTA DE CALIBRACIÓ (des de SEQ_CAL)
# =============================================================================

class CalibrationLineView(QWidget):
    """Vista per construir recta de calibració des de SEQ_CAL dedicades."""

    def __init__(self, parent_panel):
        super().__init__()
        self.parent_panel = parent_panel
        self._cal_entries = []
        self._grouped_by_seq = {}
        self._filtered_entries = []
        self._last_result = None
        self._loading = False
        self._active_calib_results = {}  # {seq_name: calib_result}
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(8)

        splitter = QSplitter(Qt.Horizontal)

        # === ESQUERRA: Controls ===
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 4, 0)
        left_layout.setSpacing(6)

        # Calibració actual
        left_layout.addWidget(self._create_current_cal_group())

        # Selectors mode/senyal/model
        left_layout.addWidget(self._create_selectors_group())

        # Selector de SEQ_CAL
        left_layout.addWidget(self._create_seq_selector())

        # Taula punts (protagonista — stretch=3)
        left_layout.addWidget(self._create_points_table(), 3)

        # Detall del punt seleccionat
        left_layout.addWidget(self._create_detail_group(), 2)

        # Resultats regressió + botons
        left_layout.addWidget(self._create_results_group())
        left_layout.addWidget(self._create_buttons())

        splitter.addWidget(left)

        # === DRETA: Visualització ===
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 0, 0, 0)
        right_layout.setSpacing(6)

        # Gràfic scatter + regressió
        self.figure = Figure(figsize=(6, 5), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_layout.addWidget(self.canvas, 1)

        # Comparació
        self.comparison_label = QLabel("")
        self.comparison_label.setWordWrap(True)
        self.comparison_label.setTextFormat(Qt.RichText)
        self.comparison_label.setStyleSheet(
            "QLabel { background: #f8f9fa; border: 1px solid #dee2e6; "
            "border-radius: 4px; padding: 8px; }"
        )
        right_layout.addWidget(self.comparison_label)

        # === Secció APLICAR CALIBRACIÓ ===
        right_layout.addWidget(self._create_apply_section())

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)

        layout.addWidget(splitter, 1)

    # ---- UI Creation ----

    def _create_current_cal_group(self):
        group = QGroupBox("Calibració Actual")
        grid = QGridLayout(group)
        grid.setContentsMargins(8, 6, 8, 6)

        self.cur_rf_label = QLabel("—")
        self.cur_rf_label.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.cur_intercept_label = QLabel("—")
        self.cur_r2_label = QLabel("—")
        self.cur_npoints_label = QLabel("—")

        grid.addWidget(QLabel("RF_mass_cal:"), 0, 0)
        grid.addWidget(self.cur_rf_label, 0, 1)
        grid.addWidget(QLabel("Intercept:"), 0, 2)
        grid.addWidget(self.cur_intercept_label, 0, 3)
        grid.addWidget(QLabel("R²:"), 1, 0)
        grid.addWidget(self.cur_r2_label, 1, 1)
        grid.addWidget(QLabel("n_punts:"), 1, 2)
        grid.addWidget(self.cur_npoints_label, 1, 3)

        return group

    def _create_selectors_group(self):
        group = QGroupBox("Paràmetres regressió")
        layout = QHBoxLayout(group)
        layout.setContentsMargins(8, 6, 8, 6)

        # Mode: COLUMN / BP
        layout.addWidget(QLabel("Mode:"))
        self.mode_group = QButtonGroup(self)
        self.radio_column = QRadioButton("COLUMN")
        self.radio_bp = QRadioButton("BP")
        self.radio_column.setChecked(True)
        self.mode_group.addButton(self.radio_column, 0)
        self.mode_group.addButton(self.radio_bp, 1)
        layout.addWidget(self.radio_column)
        layout.addWidget(self.radio_bp)

        layout.addSpacing(12)

        # Senyal: Direct / UIB / 254
        layout.addWidget(QLabel("Senyal:"))
        self.signal_combo = QComboBox()
        self.signal_combo.addItems(["direct", "uib", "254"])
        self.signal_combo.setFixedWidth(80)
        layout.addWidget(self.signal_combo)

        layout.addSpacing(12)

        # Model: Intercept / Origen
        layout.addWidget(QLabel("Model:"))
        self.model_group = QButtonGroup(self)
        self.radio_intercept = QRadioButton("Intercept")
        self.radio_origin = QRadioButton("Origen")
        self.radio_intercept.setChecked(True)
        self.model_group.addButton(self.radio_intercept, 0)
        self.model_group.addButton(self.radio_origin, 1)
        layout.addWidget(self.radio_intercept)
        layout.addWidget(self.radio_origin)

        layout.addStretch()

        # Connexions
        self.mode_group.buttonClicked.connect(self._on_params_changed)
        self.signal_combo.currentIndexChanged.connect(self._on_params_changed)
        self.model_group.buttonClicked.connect(self._on_params_changed)

        return group

    def _create_seq_selector(self):
        group = QGroupBox("SEQs de Calibració (_CAL)")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(4, 4, 4, 4)

        self.seq_list = QListWidget()
        self.seq_list.setMaximumHeight(120)
        self.seq_list.itemChanged.connect(self._on_seq_selection_changed)
        layout.addWidget(self.seq_list)

        return group

    def _create_points_table(self):
        group = QGroupBox("Punts de calibració")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(4, 4, 4, 4)

        self.points_table = QTableWidget()
        self._pt_cols = [
            "Usar", "SEQ", "Conc", "Vol", "µg",
            "Àrea", "RF", "SNR", "FWHM", "t_ret",
            "Sim", "R²bg", "QS", "Estat"
        ]
        self.points_table.setColumnCount(len(self._pt_cols))
        self.points_table.setHorizontalHeaderLabels(self._pt_cols)

        header = self.points_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # checkbox
        header.setSectionResizeMode(1, QHeaderView.Stretch)  # SEQ
        for i in range(2, len(self._pt_cols)):
            header.setSectionResizeMode(i, QHeaderView.ResizeToContents)

        self.points_table.setAlternatingRowColors(True)
        self.points_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.points_table.verticalHeader().setVisible(False)
        self.points_table.setSelectionMode(QTableWidget.SingleSelection)
        self.points_table.currentCellChanged.connect(self._on_point_selected)

        layout.addWidget(self.points_table)
        return group

    def _create_detail_group(self):
        """Secció detall del punt seleccionat a la taula."""
        self._detail_group = QGroupBox("Detall punt seleccionat")
        self._detail_group.setStyleSheet("""
            QGroupBox {
                font-size: 11px; border: 1px solid #dee2e6;
                border-radius: 4px; margin-top: 8px; padding-top: 16px;
                background-color: #f8f9fa;
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 8px; padding: 0 4px;
            }
        """)
        layout = QVBoxLayout(self._detail_group)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(2)

        self._detail_label = QLabel(
            "<i style='color:#999;'>Selecciona un punt per veure el detall</i>"
        )
        self._detail_label.setWordWrap(True)
        self._detail_label.setTextFormat(Qt.RichText)
        self._detail_label.setStyleSheet("font-size: 10px; border: none; background: transparent;")
        layout.addWidget(self._detail_label)

        return self._detail_group

    def _create_results_group(self):
        group = QGroupBox("Resultat regressió")
        grid = QGridLayout(group)
        grid.setContentsMargins(8, 6, 8, 6)

        self.res_rf_label = QLabel("—")
        self.res_rf_label.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.res_intercept_label = QLabel("—")
        self.res_r2_label = QLabel("—")
        self.res_npoints_label = QLabel("—")
        self.res_rms_label = QLabel("—")

        grid.addWidget(QLabel("RF_mass_cal:"), 0, 0)
        grid.addWidget(self.res_rf_label, 0, 1)
        grid.addWidget(QLabel("Intercept:"), 0, 2)
        grid.addWidget(self.res_intercept_label, 0, 3)
        grid.addWidget(QLabel("R²:"), 1, 0)
        grid.addWidget(self.res_r2_label, 1, 1)
        grid.addWidget(QLabel("n_punts:"), 1, 2)
        grid.addWidget(self.res_npoints_label, 1, 3)
        grid.addWidget(QLabel("RMS residuals:"), 2, 0)
        grid.addWidget(self.res_rms_label, 2, 1)

        return group

    def _create_buttons(self):
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        self.btn_recalculate = QPushButton("Recalcular")
        self.btn_recalculate.setToolTip("Recalcular regressió amb els punts seleccionats")
        self.btn_recalculate.clicked.connect(self._recalculate_regression)
        layout.addWidget(self.btn_recalculate)

        layout.addStretch()

        self.btn_diagnostic = QPushButton("📊 Diagnòstic")
        self.btn_diagnostic.setToolTip(
            "Obre diagnòstic complet amb cromatogrames per rèplica\n"
            "(disponible després de processar una SEQ_CAL)"
        )
        self.btn_diagnostic.setEnabled(False)
        self.btn_diagnostic.clicked.connect(self._on_show_diagnostic)
        layout.addWidget(self.btn_diagnostic)

        self.btn_reprocess = QPushButton("↻ Reprocessar")
        self.btn_reprocess.setToolTip(
            "Re-importa i re-calibra la SEQ seleccionada\n"
            "per obtenir dades de diagnòstic actualitzades"
        )
        self.btn_reprocess.clicked.connect(self._on_reprocess_seq)
        layout.addWidget(self.btn_reprocess)

        return widget

    # ---- Calib result / Diagnòstic ----

    def set_active_calib_result(self, seq_name, calib_result):
        """Guarda el resultat de calibració ric per diagnòstic posterior."""
        self._active_calib_results[seq_name] = calib_result
        self.btn_diagnostic.setEnabled(True)
        logger.info(f"Calib result guardat per {seq_name}")

    def _on_show_diagnostic(self):
        """Obre diàleg amb diagnòstic complet (cromatogrames, mètriques)."""
        # Trobar la SEQ seleccionada
        selected_seqs = self._get_selected_seq_names()
        calib_result = None
        seq_name = None
        for s in selected_seqs:
            if s in self._active_calib_results:
                calib_result = self._active_calib_results[s]
                seq_name = s
                break

        if not calib_result:
            QMessageBox.information(
                self, "Info",
                "No hi ha dades de diagnòstic disponibles.\n\n"
                "Prem '↻ Reprocessar' per obtenir les dades de cromatograma."
            )
            return

        # Obrir diàleg diagnòstic
        from gui.widgets.calibrate_panel.graph_widgets import KHPReplicaGraphWidget
        dlg = QMessageBox(self)
        dlg.setWindowTitle(f"Diagnòstic — {seq_name}")
        dlg.setIcon(QMessageBox.Information)

        # Construir text amb mètriques riques
        khp_d = calib_result.get('khp_data_direct') or {}
        khp_u = calib_result.get('khp_data_uib') or {}
        lines = [f"<h3>Diagnòstic {seq_name}</h3>"]

        for label, khp in [("DOC Direct", khp_d), ("UIB", khp_u)]:
            if not khp:
                continue
            area = khp.get('area', 0) or 0
            snr = khp.get('snr', 0) or 0
            fwhm = khp.get('fwhm_doc', 0) or 0
            rf = khp.get('rf_mass', 0) or 0
            n_rep = khp.get('n_replicas', 0)
            rsd = khp.get('rsd', 0) or 0
            lines.append(f"<b>{label}</b>: Àrea={area:.1f}, RF={rf:.0f}, "
                         f"SNR={snr:.0f}, FWHM={fwhm:.2f}, "
                         f"n_rep={n_rep}, RSD={rsd:.1f}%")

        warnings = calib_result.get('warnings_structured', [])
        if warnings:
            lines.append(f"<br><b>Avisos ({len(warnings)})</b>:")
            for w in warnings[:10]:
                msg = w.get('message', str(w)) if isinstance(w, dict) else str(w)
                lines.append(f"  • {msg}")

        dlg.setText("<br>".join(lines))
        dlg.setTextFormat(Qt.RichText)
        dlg.exec()

    def _on_reprocess_seq(self):
        """Re-processa la SEQ seleccionada amb CalSeqWorker."""
        selected_seqs = self._get_selected_seq_names()
        if not selected_seqs:
            QMessageBox.information(self, "Info", "Selecciona una SEQ_CAL primer.")
            return

        seq_name = selected_seqs[0]

        # Buscar el path de la SEQ
        seq_path = None
        for entry in self._cal_entries:
            if entry.get('seq_name') == seq_name:
                seq_path = entry.get('seq_path')
                break

        if not seq_path or not os.path.isdir(seq_path):
            QMessageBox.warning(
                self, "Error",
                f"No s'ha trobat el directori de la SEQ:\n{seq_path}"
            )
            return

        # Llançar worker via parent_panel
        self.show_processing_message(seq_name)
        self.parent_panel._start_cal_worker(seq_path)

    def _create_apply_section(self):
        """Secció per aplicar la calibració calculada."""
        self._apply_group = QGroupBox("Aplicar Calibració")
        self._apply_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold; font-size: 12px;
                border: 2px solid #27AE60;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 20px;
                background-color: #f0fff4;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
                color: #27AE60;
            }
        """)
        self._apply_group.setVisible(False)

        apply_layout = QVBoxLayout(self._apply_group)
        apply_layout.setContentsMargins(12, 8, 12, 12)
        apply_layout.setSpacing(8)

        # valid_from DateEdit
        opts_row = QHBoxLayout()
        opts_row.addWidget(QLabel("Vigent des de:"))
        self._apply_valid_from = QDateEdit()
        self._apply_valid_from.setCalendarPopup(True)
        self._apply_valid_from.setDate(QDate.currentDate())
        self._apply_valid_from.setDisplayFormat("yyyy-MM-dd")
        opts_row.addWidget(self._apply_valid_from)
        opts_row.addStretch()
        apply_layout.addLayout(opts_row)

        # Checkbox retroactiu
        self._apply_retroactive_chk = QCheckBox("Aplicar retroactivament")
        self._apply_retroactive_chk.setToolTip(
            "Requantifica SEQs processades amb els nous RF/intercept\n"
            "(les àrees no canvien, només ppm)"
        )
        self._apply_retroactive_chk.toggled.connect(self._on_retroactive_toggled)
        apply_layout.addWidget(self._apply_retroactive_chk)

        # Frame llista SEQs retroactives
        self._retro_frame = QFrame()
        self._retro_frame.setStyleSheet("""
            QFrame {
                background-color: #fff3e0;
                border: 1px solid #ffcc80;
                border-radius: 6px;
            }
        """)
        self._retro_frame.setVisible(False)
        retro_layout = QVBoxLayout(self._retro_frame)
        retro_layout.setContentsMargins(10, 8, 10, 8)
        retro_layout.setSpacing(4)

        self._retro_info_label = QLabel("")
        self._retro_info_label.setWordWrap(True)
        self._retro_info_label.setStyleSheet("font-size: 11px; border: none;")
        retro_layout.addWidget(self._retro_info_label)

        self._retro_scroll = QScrollArea()
        self._retro_scroll.setWidgetResizable(True)
        self._retro_scroll.setMaximumHeight(120)
        self._retro_scroll.setFrameShape(QFrame.NoFrame)
        self._retro_content = QWidget()
        self._retro_content_layout = QVBoxLayout(self._retro_content)
        self._retro_content_layout.setContentsMargins(0, 0, 0, 0)
        self._retro_content_layout.setSpacing(2)
        self._retro_scroll.setWidget(self._retro_content)
        retro_layout.addWidget(self._retro_scroll)

        # Select all / none
        sel_row = QHBoxLayout()
        btn_sel_all = QPushButton("Seleccionar totes")
        btn_sel_all.setFixedHeight(24)
        btn_sel_all.setStyleSheet("font-size: 10px; border: none; color: #2980B9;")
        btn_sel_all.clicked.connect(lambda: self._select_all_retro(True))
        btn_sel_none = QPushButton("Cap")
        btn_sel_none.setFixedHeight(24)
        btn_sel_none.setStyleSheet("font-size: 10px; border: none; color: #2980B9;")
        btn_sel_none.clicked.connect(lambda: self._select_all_retro(False))
        sel_row.addWidget(btn_sel_all)
        sel_row.addWidget(btn_sel_none)
        sel_row.addStretch()
        retro_layout.addLayout(sel_row)

        apply_layout.addWidget(self._retro_frame)

        # Botó aplicar
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._apply_btn = QPushButton("Aplicar com a Nova Calibració")
        self._apply_btn.setStyleSheet("""
            QPushButton {
                background-color: #27AE60; color: white;
                border: none; border-radius: 6px;
                padding: 10px 24px; font-size: 13px; font-weight: bold;
            }
            QPushButton:hover { background-color: #2ECC71; }
            QPushButton:disabled { background-color: #BDC3C7; }
        """)
        self._apply_btn.clicked.connect(self._on_apply_calibration)
        btn_row.addWidget(self._apply_btn)
        btn_row.addStretch()
        apply_layout.addLayout(btn_row)

        # Estat
        self._apply_status = QLabel("")
        self._apply_status.setAlignment(Qt.AlignCenter)
        self._apply_status.setStyleSheet("font-size: 11px; border: none;")
        self._apply_status.setTextFormat(Qt.RichText)
        apply_layout.addWidget(self._apply_status)

        # Llista checkboxes retroactives
        self._retro_seq_checkboxes = []

        return self._apply_group

    # ---- Data & Refresh ----

    def set_data(self, cal_entries):
        """Rep les entrades filtrades _CAL."""
        self._cal_entries = cal_entries
        self._loading = True
        self._load_current_calibration()
        self._populate_seq_list()
        self._loading = False
        self._refresh_points_and_recalculate()

    def pre_select_seq(self, seq_name):
        """Pre-selecciona una SEQ específica (des de Dashboard).

        Detecta automàticament el mode (COLUMN/BP) de la SEQ i ajusta el selector.
        Desmarca les altres SEQs i marca només la indicada.
        """
        # Detectar mode de la SEQ a partir de les entrades
        seq_modes = set()
        for entry in self._cal_entries:
            if entry.get("seq_name", "") == seq_name:
                seq_modes.add(entry.get("mode", "").upper())

        if "BP" in seq_modes and "COLUMN" not in seq_modes:
            self.radio_bp.setChecked(True)
        elif "COLUMN" in seq_modes:
            self.radio_column.setChecked(True)

        # Repoblar llista amb el mode detectat
        self._loading = True
        self._populate_seq_list()

        # Desmarcar totes i marcar només la SEQ indicada
        for i in range(self.seq_list.count()):
            item = self.seq_list.item(i)
            item_seq = item.data(Qt.UserRole)
            item.setCheckState(Qt.Checked if item_seq == seq_name else Qt.Unchecked)

        self._loading = False
        self._refresh_points_and_recalculate()

        logger.info(f"  Pre-seleccionat {seq_name} (modes: {seq_modes})")

    def show_processing_message(self, seq_name):
        """Mostra missatge 'Processant...' mentre s'importa/calibra una SEQ_CAL nova."""
        # Netejar gràfic i taula
        self.points_table.setRowCount(0)
        self.figure.clear()
        self.canvas.draw()

        # Mostrar missatge a la comparació
        self.comparison_label.setText(
            f"<div style='text-align:center; padding:20px;'>"
            f"<span style='font-size:14px; color:#2980B9;'>"
            f"⏳ Processant <b>{seq_name}</b>...</span><br><br>"
            f"<span style='color:#666;'>Important dades i calibrant.<br>"
            f"Això pot trigar uns segons.</span></div>"
        )

        # Netejar resultats
        for lbl in [self.res_rf_label, self.res_intercept_label,
                     self.res_r2_label, self.res_npoints_label, self.res_rms_label]:
            lbl.setText("⏳")

    def _get_mode(self):
        return "COLUMN" if self.radio_column.isChecked() else "BP"

    def _get_signal(self):
        return self.signal_combo.currentText()

    def _get_model(self):
        return "intercept" if self.radio_intercept.isChecked() else "origin"

    def _load_current_calibration(self):
        """Mostra la calibració global activa."""
        cal = get_active_global_calibration()
        if not cal:
            self.cur_rf_label.setText("No disponible")
            return

        mode = self._get_mode().lower()
        signal = self._get_signal().lower()

        # RF
        rf_data = cal.get('rf_mass_cal', {})
        rf_val = None
        if isinstance(rf_data, dict):
            signal_rf = rf_data.get(signal, {})
            if isinstance(signal_rf, dict):
                rf_val = signal_rf.get(mode)
        self.cur_rf_label.setText(f"{rf_val:.1f}" if rf_val is not None else "—")

        # Intercept
        intercept_data = cal.get('intercept', 0)
        int_val = 0
        if isinstance(intercept_data, dict):
            signal_int = intercept_data.get(signal, {})
            if isinstance(signal_int, dict):
                int_val = signal_int.get(mode, 0)
        elif isinstance(intercept_data, (int, float)):
            int_val = intercept_data
        self.cur_intercept_label.setText(f"{int_val:.1f}")

        # R²
        r2_data = cal.get('r2')
        r2_val = r2_data.get(mode) if isinstance(r2_data, dict) else r2_data
        self.cur_r2_label.setText(f"{r2_val:.4f}" if r2_val is not None else "—")

        # n_points
        np_data = cal.get('n_points')
        np_val = np_data.get(mode) if isinstance(np_data, dict) else np_data
        self.cur_npoints_label.setText(str(np_val) if np_val is not None else "—")

    def _populate_seq_list(self):
        """Omple la llista de SEQs _CAL disponibles."""
        self.seq_list.blockSignals(True)
        self.seq_list.clear()

        mode = self._get_mode()

        # Agrupar entrades per SEQ
        self._grouped_by_seq = {}
        for entry in self._cal_entries:
            if entry.get('mode', '').upper() != mode.upper():
                continue
            seq_name = entry.get('seq_name', 'Desconegut')
            self._grouped_by_seq.setdefault(seq_name, []).append(entry)

        # Crear items amb checkbox, ordenats per nom
        for seq_name in sorted(self._grouped_by_seq.keys()):
            entries = self._grouped_by_seq[seq_name]
            concs = sorted(set(e.get('conc_ppm', 0) for e in entries))
            n = len(entries)
            conc_range = f"{min(concs):g}–{max(concs):g}" if concs else "?"

            item = QListWidgetItem(f"{seq_name}  ({n} punts, {conc_range} ppm)")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            item.setData(Qt.UserRole, seq_name)
            self.seq_list.addItem(item)

        self.seq_list.blockSignals(False)

    def _get_selected_seq_names(self):
        """Retorna noms de SEQs seleccionades."""
        selected = []
        for i in range(self.seq_list.count()):
            item = self.seq_list.item(i)
            if item.checkState() == Qt.Checked:
                selected.append(item.data(Qt.UserRole))
        return selected

    def _refresh_points_and_recalculate(self):
        """Refresca taula de punts i recalcula regressió."""
        self._refresh_points_table()
        self._recalculate_regression()

    def _refresh_points_table(self):
        """Pobla la taula amb punts de les SEQs seleccionades (14 columnes)."""
        mode = self._get_mode()
        signal = self._get_signal()
        selected_seqs = self._get_selected_seq_names()

        self.points_table.setRowCount(0)
        self.points_table.blockSignals(True)

        # Recollir punts de les SEQs seleccionades
        self._filtered_entries = []
        for seq_name in selected_seqs:
            for entry in self._grouped_by_seq.get(seq_name, []):
                self._filtered_entries.append(entry)

        self.points_table.setRowCount(len(self._filtered_entries))

        for row, cal in enumerate(self._filtered_entries):
            conc = cal.get('conc_ppm', 0)
            vol = cal.get('volume_uL', 0)
            ug_doc = conc * vol / 1000.0 if conc > 0 and vol > 0 else 0

            # Àrea i mètriques segons senyal seleccionat
            sig = signal.lower()
            if sig == 'uib':
                area = cal.get('area_u', 0) or 0
                snr = cal.get('snr_u', 0) or 0
                fwhm = cal.get('fwhm_u', 0) or 0
                t_ret = cal.get('t_retention_u', 0) or 0
                sym = cal.get('symmetry_u', 0) or 0
                bg = cal.get('bigaussian_uib') or {}
            elif sig == '254':
                area = cal.get('area_254', 0) or cal.get('a254_area', 0) or 0
                snr = 0  # No disponible per 254
                fwhm = cal.get('fwhm_254', 0) or 0
                t_ret = cal.get('t_dad_max', 0) or 0
                sym = 0
                bg = cal.get('bigaussian_254') or {}
            else:
                area = cal.get('area', 0) or 0
                snr = cal.get('snr', 0) or 0
                fwhm = cal.get('fwhm_doc', 0) or 0
                t_ret = cal.get('t_retention', 0) or 0
                sym = cal.get('symmetry', 0) or 0
                bg = cal.get('bigaussian_doc') or {}

            rf_mass = area / ug_doc if ug_doc > 0 else 0
            bg_r2 = bg.get('r2', 0) if isinstance(bg, dict) else 0
            qs = cal.get('quality_score', 0) or 0
            is_outlier = cal.get('is_outlier', False)
            not_valid = not cal.get('valid_for_calibration', True)
            bad_point = is_outlier or not_valid or conc <= 0 or area <= 0 or qs >= 100

            # Estat derivat
            q_issues = cal.get('quality_issues', [])
            c_issues = cal.get('calibration_issues', [])
            all_issues = list(q_issues) + [str(i) for i in c_issues if str(i) not in [str(q) for q in q_issues]]
            if not_valid or qs >= 100:
                estat = "INVALID"
            elif qs > 50:
                estat = "CHECK"
            elif qs > 20 or len(all_issues) > 0:
                estat = f"INFO"
            else:
                estat = "OK"

            # Col 0: Checkbox
            chk = QCheckBox()
            chk.setChecked(not bad_point)
            chk.stateChanged.connect(self._on_point_toggled)
            chk_widget = QWidget()
            chk_layout = QHBoxLayout(chk_widget)
            chk_layout.addWidget(chk)
            chk_layout.setAlignment(Qt.AlignCenter)
            chk_layout.setContentsMargins(0, 0, 0, 0)
            self.points_table.setCellWidget(row, 0, chk_widget)

            # Col 1-13: Dades
            items_data = [
                (cal.get('seq_name', ''), None),
                (f"{conc:g}", None),
                (f"{vol:.0f}", None),
                (f"{ug_doc:.3f}", None),
                (f"{area:.1f}", None),
                (f"{rf_mass:.0f}", None),
                (f"{snr:.0f}" if snr > 0 else "—",
                 "#dc3545" if 0 < snr < 10 else "#ffc107" if snr < 30 else None),
                (f"{fwhm:.2f}" if fwhm > 0 else "—",
                 "#ffc107" if fwhm > 1.5 else None),
                (f"{t_ret:.1f}" if t_ret > 0 else "—", None),
                (f"{sym:.2f}" if sym > 0 else "—",
                 "#ffc107" if sym > 0 and (sym < 0.5 or sym > 2.5) else None),
                (f"{bg_r2:.3f}" if bg_r2 > 0 else "—",
                 "#dc3545" if 0 < bg_r2 < 0.90 else "#ffc107" if bg_r2 < 0.95 else "#28a745" if bg_r2 > 0 else None),
                (str(int(qs)),
                 "#dc3545" if qs >= 100 else "#ffc107" if qs > 20 else None),
                (estat,
                 "#dc3545" if estat == "INVALID" else "#ffc107" if estat in ("CHECK", "INFO") else "#28a745"),
            ]

            for col, (text, color) in enumerate(items_data):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                if bad_point:
                    item.setForeground(QColor("#dc3545"))
                elif color:
                    item.setForeground(QColor(color))
                self.points_table.setItem(row, col + 1, item)

            # Tooltip complet
            tip_parts = [
                f"SEQ: {cal.get('seq_name', '')}",
                f"Data: {str(cal.get('date', ''))[:10]}",
                f"Quality Score: {qs}",
            ]
            sel = cal.get('selection') or {}
            if sel:
                tip_parts.append(f"Selecció: {sel.get('method', '?')} ({sel.get('reason', '')})")
                tip_parts.append(f"Rèpliques: {sel.get('selected_replicas', '?')}")
            rsd_val = cal.get('rsd', 0)
            if rsd_val:
                tip_parts.append(f"RSD: {rsd_val:.1f}%")
            if cal.get('has_irregular_top'):
                tip_parts.append("⚠ Pic irregular (Pic_J)")
            if cal.get('has_timeout'):
                tip_parts.append("⚠ Timeout detectat")
            if is_outlier:
                tip_parts.append("❌ OUTLIER")
            if all_issues:
                tip_parts.append("--- Issues ---")
                for iss in all_issues[:8]:
                    tip_parts.append(f"  • {iss}")

            tip = "\n".join(tip_parts)
            for col in range(1, len(self._pt_cols)):
                it = self.points_table.item(row, col)
                if it:
                    it.setToolTip(tip)

        self.points_table.blockSignals(False)

    # ---- Events ----

    def _on_params_changed(self, *args):
        if self._loading:
            return
        self._loading = True
        self._load_current_calibration()
        self._populate_seq_list()
        self._loading = False
        self._refresh_points_and_recalculate()

    def _on_seq_selection_changed(self, *args):
        if not self._loading:
            self._refresh_points_and_recalculate()

    def _on_point_toggled(self, *args):
        if not self._loading:
            self._recalculate_regression()

    # ---- Regressió ----

    def _get_selected_calibrations(self):
        """Retorna llista d'entrades seleccionades (checkbox marcat a la taula)."""
        filtered = getattr(self, '_filtered_entries', [])

        selected = []
        for row in range(self.points_table.rowCount()):
            chk_widget = self.points_table.cellWidget(row, 0)
            if chk_widget:
                chk = chk_widget.findChild(QCheckBox)
                if chk and chk.isChecked() and row < len(filtered):
                    selected.append(filtered[row])

        return selected

    def _recalculate_regression(self):
        """Executa regressió amb punts seleccionats."""
        selected = self._get_selected_calibrations()
        model = self._get_model()
        signal = self._get_signal()
        mode = self._get_mode()

        result = fit_calibration_from_history(
            selected, mode=mode, signal=signal, model=model
        )

        self._last_result = result

        if result['success']:
            self.res_rf_label.setText(f"{result['rf_mass_cal']:.1f}")
            self.res_intercept_label.setText(f"{result['intercept']:.1f}")
            self.res_r2_label.setText(f"{result['r2']:.4f}")
            self.res_npoints_label.setText(str(result['n_points']))
            rms = result.get('residuals_rms')
            self.res_rms_label.setText(f"{rms:.2f}" if rms is not None else "—")
            pass  # Consulta: no s'aplica, només previsualització
        else:
            for lbl in (self.res_rf_label, self.res_intercept_label,
                        self.res_r2_label, self.res_rms_label):
                lbl.setText("—")
            self.res_npoints_label.setText(str(result.get('n_points', 0)))

        self._update_preview_graph(result)
        self._update_comparison(result)
        self._update_apply_visibility()

    # ---- Detall punt seleccionat ----

    def _on_point_selected(self, row, col, prev_row, prev_col):
        """Mostra detall del punt seleccionat a la taula."""
        filtered = getattr(self, '_filtered_entries', [])
        if row < 0 or row >= len(filtered):
            self._detail_label.setText(
                "<i style='color:#999;'>Selecciona un punt per veure el detall</i>"
            )
            return

        cal = filtered[row]
        signal = self._get_signal().lower()
        lines = []

        # Capçalera
        seq = cal.get('seq_name', '?')
        conc = cal.get('conc_ppm', 0)
        vol = cal.get('volume_uL', 0)
        date = str(cal.get('date', ''))[:10]
        lines.append(
            f"<b>{seq}</b> — KHP {conc:g} ppm · {vol:.0f} µL · {date}"
        )

        # Selecció de rèpliques
        sel = cal.get('selection') or {}
        if sel:
            method = sel.get('method', '?')
            reason = sel.get('reason', '')
            reps = sel.get('selected_replicas', [])
            n_avail = sel.get('n_replicas_available', '?')
            rsd = cal.get('rsd', 0)
            lines.append(
                f"<b>Selecció</b>: {method} "
                f"(R{'+R'.join(map(str, reps))} de {n_avail}) — "
                f"<i>{reason}</i>"
                + (f" — RSD={rsd:.1f}%" if rsd else "")
            )

        # Bigaussian per senyal
        bg_keys = [('bigaussian_doc', 'DOC'), ('bigaussian_uib', 'UIB'), ('bigaussian_254', '254')]
        bg_parts = []
        for bg_key, bg_name in bg_keys:
            bg = cal.get(bg_key)
            if isinstance(bg, dict) and bg.get('r2', 0) > 0:
                r2 = bg['r2']
                status = bg.get('status', '?')
                asym = bg.get('asymmetry', 0)
                color = '#28a745' if status == 'VALID' else '#ffc107' if status == 'CHECK' else '#dc3545'
                bg_parts.append(
                    f"<span style='color:{color}'>{bg_name}: R²={r2:.3f} ({status})"
                    + (f" asim={asym:.2f}" if asym else "")
                    + "</span>"
                )
        if bg_parts:
            lines.append(f"<b>Bigaussian</b>: {' · '.join(bg_parts)}")

        # Anomalies
        anomaly_parts = []
        if cal.get('has_irregular_top'):
            repaired = cal.get('irregular_top_repaired', False)
            anomaly_parts.append(
                f"Pic_J {'(reparat)' if repaired else '(!)'}"
            )
        if cal.get('has_timeout'):
            sev = cal.get('timeout_severity', 'OK')
            anomaly_parts.append(f"Timeout ({sev})")
        if anomaly_parts:
            lines.append(
                f"<b>Anomalies</b>: "
                + "<span style='color:#dc3545'>" + " · ".join(anomaly_parts) + "</span>"
            )

        # Comparació rèpliques
        comp = cal.get('replica_comparison') or {}
        if comp:
            comp_parts = []
            if comp.get('diff_area_pct'):
                comp_parts.append(f"ΔÀrea={comp['diff_area_pct']:.1f}%")
            if comp.get('diff_t_max_sec'):
                comp_parts.append(f"Δt_max={comp['diff_t_max_sec']:.0f}s")
            if comp.get('pearson_r2') is not None:
                comp_parts.append(f"Pearson={comp['pearson_r2']:.3f}")
            if comp_parts:
                lines.append(f"<b>Rèpliques</b>: {' · '.join(comp_parts)}")

        # Quality issues
        q_issues = cal.get('quality_issues', [])
        c_issues = cal.get('calibration_issues', [])
        all_iss = list(q_issues) + [str(i) for i in c_issues]
        if all_iss:
            lines.append(f"<b>Issues ({len(all_iss)})</b>:")
            for iss in all_iss[:6]:
                lines.append(f"  <span style='color:#dc3545'>• {iss}</span>")
            if len(all_iss) > 6:
                lines.append(f"  <i>... i {len(all_iss)-6} més</i>")

        self._detail_label.setText("<br>".join(lines))

    # ---- Gràfic ----

    def _update_preview_graph(self, result):
        """Scatter + recta regressió + residuals subplot."""
        self.figure.clear()

        # Dos subplots: principal (scatter) + residuals
        if result.get('success') and result.get('points'):
            ax_main = self.figure.add_axes([0.12, 0.35, 0.85, 0.60])
            ax_res = self.figure.add_axes([0.12, 0.08, 0.85, 0.22])
        else:
            ax_main = self.figure.add_subplot(111)
            ax_res = None

        mode = self._get_mode()
        signal = self._get_signal()

        # Punts seleccionats vs exclosos
        selected = self._get_selected_calibrations()
        selected_keys = set()
        for c in selected:
            key = (c.get('seq_name', ''), c.get('conc_ppm', 0),
                   c.get('volume_uL', 0), c.get('area', 0))
            selected_keys.add(key)

        selected_seqs = self._get_selected_seq_names()
        all_entries = []
        for seq_name in selected_seqs:
            for e in self._grouped_by_seq.get(seq_name, []):
                all_entries.append(e)

        x_sel, y_sel = [], []
        x_exc, y_exc = [], []

        for cal in all_entries:
            conc = cal.get('conc_ppm', 0)
            vol = cal.get('volume_uL', 0)
            if conc <= 0 or vol <= 0:
                continue
            ug = conc * vol / 1000.0
            area = cal.get('area_u', 0) if signal == 'uib' else (
                cal.get('area_254', 0) or 0) if signal == '254' else cal.get('area', 0)
            if area <= 0:
                continue

            key = (cal.get('seq_name', ''), cal.get('conc_ppm', 0),
                   cal.get('volume_uL', 0), cal.get('area', 0))
            if key in selected_keys:
                x_sel.append(ug)
                y_sel.append(area)
            else:
                x_exc.append(ug)
                y_exc.append(area)

        if x_sel:
            ax_main.scatter(x_sel, y_sel, c='#2196F3', s=50, zorder=5,
                            label='Seleccionats', edgecolors='white', linewidth=0.5)
        if x_exc:
            ax_main.scatter(x_exc, y_exc, c='#aaa', s=40, zorder=4, marker='x',
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
                eq = f"y = {rf:.1f}x + {intercept:.1f}" if intercept != 0 else f"y = {rf:.1f}x"
                ax_main.plot(x_line, y_line, 'r-', linewidth=2,
                             label=f"Nova ({eq}, R²={r2:.4f})")

        # Recta actual (discontinua)
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
                ax_main.plot(x_line, y_cur, '--', color='gray', linewidth=1.5,
                             alpha=0.7, label=f"Actual (RF={cur_rf:.0f})")

        ax_main.set_ylabel("Àrea")
        ax_main.set_title(f"Recta calibració — {mode} {signal}")
        ax_main.legend(fontsize=7, loc='upper left')
        ax_main.grid(True, alpha=0.3)

        if ax_res is None:
            ax_main.set_xlabel("µg DOC injectat")

        # Residuals subplot
        if ax_res is not None and result.get('success') and result.get('points'):
            points = result['points']
            x_res = [p['ug_doc'] for p in points]
            y_res = [p.get('residual', 0) for p in points]
            colors = ['#dc3545' if abs(r) > 2 * result.get('residuals_rms', 999)
                       else '#2196F3' for r in y_res]
            ax_res.bar(range(len(y_res)), y_res, color=colors, alpha=0.7)
            ax_res.axhline(0, color='black', linewidth=0.5)
            rms = result.get('residuals_rms', 0)
            if rms:
                ax_res.axhline(rms, color='#aaa', linewidth=0.8, linestyle='--')
                ax_res.axhline(-rms, color='#aaa', linewidth=0.8, linestyle='--')
            ax_res.set_ylabel("Residual")
            ax_res.set_xlabel("Punt #")
            ax_res.grid(True, alpha=0.2)

        self.figure.tight_layout()
        self.canvas.draw()

    def _update_comparison(self, result):
        """Mostra comparació nova vs actual."""
        if not result.get('success'):
            self.comparison_label.setText(
                "<i>No hi ha prou punts per calcular la regressió.</i>"
            )
            return

        mode = self._get_mode().lower()
        signal = self._get_signal().lower()

        cal = get_active_global_calibration()
        if not cal:
            self.comparison_label.setText(
                f"<b>Nova calibració:</b> RF={result['rf_mass_cal']:.1f}, "
                f"Intercept={result['intercept']:.1f}, R²={result['r2']:.4f}"
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

        lines = ["<b>Comparació amb calibració actual:</b><br>"]

        pct_rf = 0
        if cur_rf is not None and cur_rf > 0:
            delta_rf = new_rf - cur_rf
            pct_rf = delta_rf / cur_rf * 100
            color_rf = "#dc3545" if abs(pct_rf) > 15 else "#28a745" if abs(pct_rf) < 5 else "#ffc107"
            lines.append(
                f"RF_mass: {cur_rf:.1f} → <b>{new_rf:.1f}</b> "
                f"(<span style='color:{color_rf}'>{delta_rf:+.1f}, {pct_rf:+.1f}%</span>)<br>"
            )
        else:
            lines.append(f"RF_mass: — → <b>{new_rf:.1f}</b><br>")

        delta_int = new_int - cur_int
        lines.append(
            f"Intercept: {cur_int:.1f} → <b>{new_int:.1f}</b> ({delta_int:+.1f})<br>"
        )
        lines.append(f"R²: <b>{result['r2']:.4f}</b>, n={result['n_points']}")

        # Impacte estimat a 1 ppm (exemple concret)
        if cur_rf and cur_rf > 0:
            # Exemple: mostra a 1 ppm, 400 µL COLUMN / 100 µL BP
            vol_ex = 100 if mode == "bp" else 400
            area_ex = cur_rf * 1.0 * vol_ex / 1000 + cur_int  # àrea esperada a 1 ppm
            ppm_old = max(0, area_ex - cur_int) * 1000 / (cur_rf * vol_ex)
            ppm_new = max(0, area_ex - new_int) * 1000 / (new_rf * vol_ex) if new_rf > 0 else 0
            if ppm_old > 0:
                pct_impact = (ppm_new - ppm_old) / ppm_old * 100
                lines.append(
                    f"<br><i>Impacte estimat a 1 ppm ({vol_ex}µL): "
                    f"{ppm_old:.3f} → {ppm_new:.3f} ppm ({pct_impact:+.1f}%)</i>"
                )

        if cur_rf is not None and cur_rf > 0 and abs(pct_rf) > 15:
            lines.append(
                "<br><span style='color:#dc3545; font-weight:bold;'>"
                "AVÍS: Variació RF > 15%</span>"
            )

        self.comparison_label.setText("".join(lines))

    # ---- Aplicar calibració ----

    def _update_apply_visibility(self):
        """Mostra/amaga la secció d'aplicar segons si hi ha regressió vàlida."""
        visible = (
            self._last_result is not None
            and self._last_result.get('success')
            and self._last_result.get('r2', 0) > 0
        )
        self._apply_group.setVisible(visible)

    def _on_retroactive_toggled(self, checked):
        """Mostra/amaga llista SEQs retroactives."""
        if checked:
            self._populate_retro_list()
        self._retro_frame.setVisible(checked)

    def _populate_retro_list(self):
        """Pobla la llista de SEQs disponibles per requantificació.

        Filtra per mode (COLUMN/BP) — no es poden barrejar.
        """
        # Netejar
        for cb in self._retro_seq_checkboxes:
            cb.deleteLater()
        self._retro_seq_checkboxes = []

        # Mode actual de la calibració
        current_mode = self._get_mode().upper()  # "COLUMN" o "BP"

        # Buscar JSONs d'anàlisi existents
        from hpsec_config import get_config
        cfg = get_config()
        data_folder_root = cfg.get("paths", "data_folder")
        if not data_folder_root or not os.path.isdir(data_folder_root):
            self._retro_info_label.setText("No s'ha trobat la carpeta de dades")
            return

        analysis_jsons = []
        skipped_other_mode = 0
        for seq_dir in sorted(os.listdir(data_folder_root)):
            seq_path = os.path.join(data_folder_root, seq_dir)
            if not os.path.isdir(seq_path):
                continue
            if "_CAL" in seq_dir.upper():
                continue  # No requantificar SEQ_CAL
            json_path = os.path.join(seq_path, "CHECK", "data", "analysis.json")
            if not os.path.exists(json_path):
                continue

            # Llegir method del JSON per filtrar per mode
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                seq_method = data.get("method", "COLUMN").upper()
            except Exception:
                seq_method = "COLUMN"

            if seq_method != current_mode:
                skipped_other_mode += 1
                continue

            analysis_jsons.append((seq_dir, json_path))

        if not analysis_jsons:
            other_mode = "BP" if current_mode == "COLUMN" else "COLUMN"
            msg = f"No hi ha SEQs {current_mode} analitzades per requantificar"
            if skipped_other_mode:
                msg += f" ({skipped_other_mode} SEQs {other_mode} excloses)"
            self._retro_info_label.setText(msg)
            return

        info = f"SEQs {current_mode} analitzades ({len(analysis_jsons)}):"
        if skipped_other_mode:
            other_mode = "BP" if current_mode == "COLUMN" else "COLUMN"
            info += f"  <i>({skipped_other_mode} {other_mode} excloses)</i>"
        self._retro_info_label.setText(info)

        for seq_dir, json_path in analysis_jsons:
            cb = QCheckBox(seq_dir)
            cb.setProperty("json_path", json_path)
            cb.setChecked(True)
            cb.setStyleSheet("border: none;")
            self._retro_content_layout.addWidget(cb)
            self._retro_seq_checkboxes.append(cb)

    def _select_all_retro(self, checked):
        """Selecciona/deselecciona totes les SEQs retroactives."""
        for cb in self._retro_seq_checkboxes:
            cb.setChecked(checked)

    def _on_apply_calibration(self):
        """Aplica la nova calibració (add_calibration + requantificació opcional)."""
        if not self._last_result or not self._last_result.get('success'):
            QMessageBox.warning(self, "Avís", "No hi ha regressió vàlida per aplicar.")
            return

        rf_new = self._last_result.get('rf_mass_cal', 0)
        intercept_new = self._last_result.get('intercept', 0)
        r2 = self._last_result.get('r2', 0)
        n_pts = self._last_result.get('n_points', 0)
        mode = self._get_mode()
        signal = self._get_signal()
        is_bp = mode.upper() == "BP"

        # Validació mínima
        if r2 < 0.95:
            resp = QMessageBox.warning(
                self, "R² baix",
                f"La R² ({r2:.4f}) és inferior a 0.95.\n"
                "Estàs segur que vols aplicar aquesta calibració?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if resp != QMessageBox.Yes:
                return

        if n_pts < 3:
            resp = QMessageBox.warning(
                self, "Pocs punts",
                f"Només {n_pts} punts a la regressió.\n"
                "Es recomanen ≥5 punts. Vols continuar?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if resp != QMessageBox.Yes:
                return

        valid_from = self._apply_valid_from.date().toString("yyyy-MM-dd")
        retroactive = self._apply_retroactive_chk.isChecked()

        # Comptar SEQs retroactives
        retro_count = sum(1 for cb in self._retro_seq_checkboxes if cb.isChecked()) if retroactive else 0

        # Confirmació
        msg = (
            f"S'aplicarà la nova calibració:\n\n"
            f"  Mode: {mode}\n"
            f"  Senyal: {signal}\n"
            f"  RF: {rf_new:.1f}\n"
            f"  Intercept: {intercept_new:.1f}\n"
            f"  R²: {r2:.6f}\n"
            f"  Vigent des de: {valid_from}\n"
        )
        if retroactive and retro_count > 0:
            msg += f"\n  Retroactiu: {retro_count} SEQs es requantificaran\n"
        msg += "\nConfirmar?"

        resp = QMessageBox.question(
            self, "Confirmar aplicació", msg,
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if resp != QMessageBox.Yes:
            return

        # --- Aplicar ---
        self._apply_btn.setEnabled(False)
        self._apply_status.setText("Aplicant...")

        try:
            import copy
            from hpsec_calibrate import (
                add_calibration, get_active_global_calibration,
                get_rf_mass_cal, get_calibration_intercept,
                requantify_analysis_json, compute_calibration_fingerprint
            )

            # Construir rf_mass_cal_values preservant l'altra branca
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

            # Actualitzar branca corresponent
            mode_key = "bp" if is_bp else "column"
            if isinstance(rf_values.get(signal), dict):
                rf_values[signal][mode_key] = rf_new
            else:
                rf_values[signal] = {mode_key: rf_new}

            if isinstance(intercept_values.get(signal), dict):
                intercept_values[signal][mode_key] = intercept_new
            else:
                intercept_values[signal] = {mode_key: intercept_new}

            # SEQs de referència
            selected_seqs = self._get_selected_seq_names()
            source = {
                "type": "SEQ_CAL",
                "description": f"Regressió from {', '.join(selected_seqs)}",
                "seq_references": selected_seqs,
                "mode": mode,
            }

            # regression_data per persistir al JSON
            reg_data = dict(self._last_result)
            reg_data['mode'] = mode
            reg_data['signal'] = signal
            reg_data['model'] = self._get_model()

            # add_calibration
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

            logger.info(f"Nova calibració aplicada: {cal_id} (RF={rf_new:.1f}, mode={mode})")

            # --- Requantificació retroactiva ---
            retro_results = []
            if retroactive and retro_count > 0:
                self._apply_status.setText(f"Requantificant {retro_count} SEQs...")

                new_cal = get_active_global_calibration()
                rf_col = get_rf_mass_cal(new_cal, signal=signal, mode="column")
                int_col = get_calibration_intercept(new_cal, signal=signal, mode="column")
                rf_bp = get_rf_mass_cal(new_cal, signal=signal, mode="bp")
                int_bp = get_calibration_intercept(new_cal, signal=signal, mode="bp")

                for cb in self._retro_seq_checkboxes:
                    if not cb.isChecked():
                        continue
                    json_path = cb.property("json_path")
                    if not json_path or not Path(json_path).exists():
                        continue
                    try:
                        rq_result = requantify_analysis_json(
                            json_path,
                            new_rf_direct=rf_col,
                            new_intercept_direct=int_col,
                            new_rf_bp=rf_bp,
                            new_intercept_bp=int_bp,
                        )
                        retro_results.append({
                            'seq': cb.text(),
                            'success': rq_result.get('success', False),
                            'updated': rq_result.get('samples_updated', 0),
                        })
                    except Exception as e:
                        retro_results.append({
                            'seq': cb.text(),
                            'success': False,
                            'error': str(e),
                        })

            # --- Actualitzar UI ---
            n_ok = sum(1 for r in retro_results if r.get('success'))
            n_fail = len(retro_results) - n_ok

            status_parts = [
                f"<span style='color:#27AE60'>&#10003; Calibració {cal_id} aplicada</span>"
            ]
            if retro_results:
                status_parts.append(f"<br>Requantificades: {n_ok} OK")
                if n_fail:
                    status_parts.append(f", <span style='color:#E74C3C'>{n_fail} errors</span>")

            self._apply_status.setText("".join(status_parts))
            self._apply_btn.setEnabled(False)

            # Refrescar calibració actual mostrada
            self._load_current_calibration()

            # Refrescar dashboard
            main_window = self.parent_panel.main_window
            if hasattr(main_window, 'dashboard_panel') and main_window.dashboard_panel:
                try:
                    main_window.dashboard_panel.refresh_sequences()
                except Exception:
                    pass

        except Exception as e:
            logger.error(f"Error aplicant calibració: {e}")
            self._apply_status.setText(
                f"<span style='color:#E74C3C'>Error: {e}</span>"
            )
            self._apply_btn.setEnabled(True)


# =============================================================================
# VISTA 2: CONTROL DE QUALITAT (Levey-Jennings)
# =============================================================================

class QCMonitorView(QWidget):
    """Vista QC: Levey-Jennings de KHP producció vs recta vigent."""

    def __init__(self, parent_panel):
        super().__init__()
        self.parent_panel = parent_panel
        self._prod_entries = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(8)

        # Selectors
        sel_widget = QWidget()
        sel_layout = QHBoxLayout(sel_widget)
        sel_layout.setContentsMargins(0, 0, 0, 0)

        sel_layout.addWidget(QLabel("Mode:"))
        self.mode_group = QButtonGroup(self)
        self.radio_column = QRadioButton("COLUMN")
        self.radio_bp = QRadioButton("BP")
        self.radio_column.setChecked(True)
        self.mode_group.addButton(self.radio_column, 0)
        self.mode_group.addButton(self.radio_bp, 1)
        sel_layout.addWidget(self.radio_column)
        sel_layout.addWidget(self.radio_bp)

        sel_layout.addSpacing(16)

        sel_layout.addWidget(QLabel("Senyal:"))
        self.signal_combo = QComboBox()
        self.signal_combo.addItems(["direct", "uib", "254"])
        self.signal_combo.setFixedWidth(80)
        sel_layout.addWidget(self.signal_combo)

        sel_layout.addStretch()

        self.mode_group.buttonClicked.connect(self._refresh)
        self.signal_combo.currentIndexChanged.connect(self._refresh)

        layout.addWidget(sel_widget)

        # Gràfic Levey-Jennings
        self.figure = Figure(figsize=(10, 5), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.canvas, 1)

        # Resum estadístic
        self.stats_label = QLabel("")
        self.stats_label.setWordWrap(True)
        self.stats_label.setTextFormat(Qt.RichText)
        self.stats_label.setStyleSheet(
            "QLabel { background: #f8f9fa; border: 1px solid #dee2e6; "
            "border-radius: 4px; padding: 8px; }"
        )
        layout.addWidget(self.stats_label)

    def _get_mode(self):
        return "COLUMN" if self.radio_column.isChecked() else "BP"

    def _get_signal(self):
        return self.signal_combo.currentText()

    def set_data(self, prod_entries):
        """Rep les entrades de producció (no _CAL)."""
        self._prod_entries = prod_entries
        self._refresh()

    def _refresh(self, *args):
        """Actualitza gràfic i estadístiques."""
        mode = self._get_mode()
        signal = self._get_signal()

        # Obtenir calibració activa
        cal = get_active_global_calibration()
        if not cal:
            self.stats_label.setText("<i>No hi ha calibració activa.</i>")
            self.figure.clear()
            self.canvas.draw()
            return

        # RF i intercept actuals
        rf_data = cal.get('rf_mass_cal', {})
        int_data = cal.get('intercept', 0)
        rf = None
        intercept = 0

        if isinstance(rf_data, dict):
            sig_rf = rf_data.get(signal, {})
            if isinstance(sig_rf, dict):
                rf = sig_rf.get(mode.lower())
        if isinstance(int_data, dict):
            sig_int = int_data.get(signal, {})
            if isinstance(sig_int, dict):
                intercept = sig_int.get(mode.lower(), 0)
        elif isinstance(int_data, (int, float)):
            intercept = int_data

        if rf is None or rf <= 0:
            self.stats_label.setText(
                f"<i>No hi ha RF per {mode} {signal}.</i>"
            )
            self.figure.clear()
            self.canvas.draw()
            return

        # Filtrar entrades per mode
        entries = []
        for e in self._prod_entries:
            if e.get('mode', '').upper() != mode.upper():
                continue
            conc = e.get('conc_ppm', 0)
            vol = e.get('volume_uL', 0)
            if conc <= 0 or vol <= 0:
                continue

            if signal.lower() == 'uib':
                area = e.get('area_u', 0)
            elif signal.lower() == '254':
                area = e.get('area_254', 0) or 0
            else:
                area = e.get('area', 0)

            if area <= 0:
                continue

            ug = conc * vol / 1000.0
            area_pred = rf * ug + intercept
            dev_pct = (area - area_pred) / area_pred * 100 if area_pred > 0 else 0

            entries.append({
                'seq_name': e.get('seq_name', ''),
                'date': e.get('date', ''),
                'conc_ppm': conc,
                'area': area,
                'area_pred': area_pred,
                'dev_pct': dev_pct,
                'is_outlier': e.get('is_outlier', False),
            })

        # Ordenar cronològicament
        entries.sort(key=lambda x: x['date'])

        # Gràfic Levey-Jennings
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        if not entries:
            ax.text(0.5, 0.5, "No hi ha dades QC per aquest mode/senyal",
                    ha='center', va='center', fontsize=12, color='#666')
            self.stats_label.setText("<i>No hi ha entrades QC de producció.</i>")
            self.canvas.draw()
            return

        devs = [e['dev_pct'] for e in entries]
        x_pos = range(len(entries))
        colors = []
        for d in devs:
            if abs(d) > 20:
                colors.append('#dc3545')  # vermell
            elif abs(d) > 10:
                colors.append('#ffc107')  # taronja
            else:
                colors.append('#28a745')  # verd

        ax.bar(x_pos, devs, color=colors, alpha=0.7, width=0.8)

        # Línies de referència
        ax.axhline(0, color='black', linewidth=1)
        ax.axhline(10, color='#ffc107', linewidth=0.8, linestyle='--', alpha=0.7, label='±10%')
        ax.axhline(-10, color='#ffc107', linewidth=0.8, linestyle='--', alpha=0.7)
        ax.axhline(20, color='#dc3545', linewidth=0.8, linestyle='--', alpha=0.7, label='±20%')
        ax.axhline(-20, color='#dc3545', linewidth=0.8, linestyle='--', alpha=0.7)

        # Línia tendència
        if len(devs) >= 3:
            x_arr = np.arange(len(devs))
            coeffs = np.polyfit(x_arr, devs, 1)
            trend_line = np.polyval(coeffs, x_arr)
            ax.plot(x_arr, trend_line, 'b-', linewidth=1.5, alpha=0.6,
                    label=f"Tendència ({coeffs[0]:+.2f}%/SEQ)")

        # Etiquetes eix X (noms SEQ cada N)
        n_labels = min(15, len(entries))
        step = max(1, len(entries) // n_labels)
        tick_pos = list(range(0, len(entries), step))
        tick_labels = [entries[i]['seq_name'][:15] for i in tick_pos]
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=7)

        ax.set_ylabel("Desviació vs recta (%)")
        ax.set_title(f"QC Levey-Jennings — {mode} {signal} (RF={rf:.0f}, int={intercept:.0f})")
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.2, axis='y')
        ax.set_ylim(min(min(devs) - 5, -25), max(max(devs) + 5, 25))

        self.figure.tight_layout()
        self.canvas.draw()

        # Estadístiques
        mean_dev = np.mean(devs)
        std_dev = np.std(devs)
        n_total = len(devs)
        n_out_10 = sum(1 for d in devs if abs(d) > 10)
        n_out_20 = sum(1 for d in devs if abs(d) > 20)

        trend_slope = coeffs[0] if len(devs) >= 3 else 0

        # Indicador d'estat global
        if n_out_20 > n_total * 0.1 or abs(mean_dev) > 15:
            status = "<span style='color:#dc3545; font-weight:bold;'>⚠ FORA DE CONTROL</span>"
        elif n_out_10 > n_total * 0.2 or abs(mean_dev) > 10:
            status = "<span style='color:#ffc107; font-weight:bold;'>⚠ ATENCIÓ</span>"
        else:
            status = "<span style='color:#28a745; font-weight:bold;'>✓ EN CONTROL</span>"

        self.stats_label.setText(
            f"{status} — "
            f"n={n_total}, "
            f"Desv. mitjana: <b>{mean_dev:+.1f}%</b>, "
            f"SD: {std_dev:.1f}%, "
            f"Fora ±10%: {n_out_10} ({n_out_10/n_total*100:.0f}%), "
            f"Fora ±20%: {n_out_20} ({n_out_20/n_total*100:.0f}%), "
            f"Tendència: {trend_slope:+.2f}%/SEQ"
        )
