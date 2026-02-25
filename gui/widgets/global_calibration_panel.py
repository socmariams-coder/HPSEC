"""
HPSEC Suite - Global Calibration Panel
========================================

Panell de calibració global — vista única amb regressió des de SEQ_CAL.
El Levey-Jennings QC es mostra al HistoryPanel (Tab 7).
Les SEQ_CAL arriben directament des del Dashboard.

Codi de regressió i aplicació recuperat del wizard (analyze_panel + review_summary_panel).
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGroupBox,
    QGridLayout, QTableWidget, QTableWidgetItem, QHeaderView,
    QComboBox, QMessageBox, QCheckBox,
    QFrame, QProgressBar, QDateEdit, QScrollArea, QDialog
)
from PySide6.QtCore import Qt, Signal, QThread, QDate
from PySide6.QtGui import QFont, QColor, QBrush

from pathlib import Path
import sys
import os
import copy
import json
import logging

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from hpsec_calibrate import (
    get_active_global_calibration,
    load_khp_history,
    fit_calibration_from_history,
    load_calibration_reference,
    compute_calibration_fingerprint,
    detect_seq_cal_data,
)

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

logger = logging.getLogger(__name__)

# Colors (from styles.py)
COLOR_SUCCESS = "#27AE60"
COLOR_WARNING = "#F39C12"
COLOR_ERROR = "#E74C3C"


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

            progress_cb(90, "Detectant SEQ_CAL...")

            # Detectar SEQ_CAL i extreure dades riques
            method = imported_data.get("method", "COLUMN")
            uib_sensitivity = imported_data.get("uib_sensitivity")
            seq_cal_data = detect_seq_cal_data(
                calib_result, self.seq_path,
                method=method, uib_sensitivity=uib_sensitivity,
            )

            progress_cb(95, "Finalitzant...")

            # Preparar resultat
            result = {
                "success": True,
                "seq_name": seq_name,
                "seq_path": self.seq_path,
                "imported_data": imported_data,
                "calib_result": calib_result,
                "seq_cal_data": seq_cal_data,
            }
            self.finished.emit(result)

        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n{traceback.format_exc()}")



class GlobalCalibrationPanel(QWidget):
    """Panell de calibració global — vista única amb regressió des de SEQ_CAL.

    Les SEQ_CAL arriben des del Dashboard. CalSeqWorker importa + calibra,
    i la CalibrationLineView mostra la regressió amb scatter, residuals,
    comparació amb la calibració vigent, i secció d'aplicar.
    """

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self._active_seq_path = None
        self._result_cache = {}
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)

        # Header: títol + botó PDF
        header = QHBoxLayout()
        title = QLabel("Calibració Global")
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        header.addWidget(title)

        subtitle = QLabel(
            "Regressió des de SEQ_CAL · Aplicar calibració"
        )
        subtitle.setFont(QFont("Segoe UI", 9))
        subtitle.setStyleSheet("color: #666;")
        header.addWidget(subtitle)

        header.addStretch()

        self._report_btn = QPushButton("📄 Informe PDF")
        self._report_btn.setStyleSheet("""
            QPushButton {
                background-color: #2980B9; color: white;
                border: none; border-radius: 6px;
                padding: 8px 16px; font-size: 11px; font-weight: bold;
            }
            QPushButton:hover { background-color: #3498DB; }
        """)
        self._report_btn.clicked.connect(self._on_generate_report)
        header.addWidget(self._report_btn)

        layout.addLayout(header)

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

        # Vista principal: CalibrationLineView (sense tabs)
        self.cal_view = CalibrationLineView(self)
        layout.addWidget(self.cal_view, 1)

        # Worker (un sol actiu)
        self._cal_worker = None

    def load_seq_cal(self, seq_path):
        """Carrega una SEQ_CAL des del Dashboard."""
        self._active_seq_path = seq_path
        seq_name = os.path.basename(seq_path)
        logger.info(f"load_seq_cal: {seq_name}")

        # Comprovar cache
        if seq_path in self._result_cache:
            logger.info(f"  Cache hit per {seq_name}")
            self._on_worker_finished(self._result_cache[seq_path], from_cache=True)
            return

        # Processar
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

        self.cal_view.show_processing_message(seq_name)

        self._cal_worker = CalSeqWorker(seq_path)
        self._cal_worker.progress.connect(self._on_worker_progress)
        self._cal_worker.finished.connect(self._on_worker_finished)
        self._cal_worker.error.connect(self._on_worker_error)
        self._cal_worker.start()

    def _on_worker_progress(self, pct, msg):
        """Actualitza barra de progrés."""
        self._progress_bar.setValue(pct)
        self._progress_label.setText(msg)

    def _on_worker_finished(self, result, from_cache=False):
        """Worker completat: passar dades a CalibrationLineView."""
        self._progress_bar.setVisible(False)
        self._progress_label.setVisible(False)

        seq_name = result.get("seq_name", "")
        seq_path = result.get("seq_path", "")
        logger.info(f"CalSeqWorker completat per {seq_name}")

        # Guardar al cache
        if not from_cache and seq_path:
            self._result_cache[seq_path] = result

        # Passar dades a la vista
        seq_cal_data = result.get("seq_cal_data")
        calib_result = result.get("calib_result")
        imported_data = result.get("imported_data")

        if seq_cal_data:
            self.cal_view.load_seq_cal_data(
                seq_name, seq_path, seq_cal_data,
                calib_result=calib_result,
                imported_data=imported_data,
            )
        else:
            self.cal_view.show_error_message(
                f"La seqüència {seq_name} no conté dades SEQ_CAL vàlides."
            )

        # Notificació
        if self.main_window:
            self.main_window.set_status(f"SEQ_CAL {seq_name} processada", 5000)

    def _on_worker_error(self, error_msg):
        """Error al processar la SEQ_CAL."""
        self._progress_bar.setVisible(False)
        self._progress_label.setVisible(False)

        logger.error(f"CalSeqWorker error: {error_msg}")

        self.cal_view.show_error_message(
            f"Error processant SEQ_CAL:\n{error_msg[:200]}"
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
# CALIBRATION LINE VIEW — Regressió + Aplicar (codi del wizard)
# =============================================================================

class CalibrationLineView(QWidget):
    """Vista de regressió i aplicació de calibració des de SEQ_CAL.

    Codi recuperat del wizard (analyze_panel + review_summary_panel).
    Rebre dades de CalSeqWorker via load_seq_cal_data().
    """

    def __init__(self, parent_panel):
        super().__init__()
        self.parent_panel = parent_panel

        # State
        self._seq_cal_regression = None
        self._seq_cal_entries = []
        self._seq_cal_entries_direct = []
        self._seq_cal_entries_uib = []
        self._seq_cal_method = "COLUMN"
        self._seq_cal_signal = "direct"
        self._seq_cal_excluded = set()
        self._seq_cal_sensitivity = None
        self._seq_name = ""
        self._seq_path = ""
        self._imported_data = None
        self._retro_seq_checkboxes = []
        self._cal_applied = False

        self._setup_ui()

    def _setup_ui(self):
        # Scroll area per tot el contingut
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        container = QWidget()
        self._main_layout = QVBoxLayout(container)
        self._main_layout.setContentsMargins(0, 4, 0, 4)
        self._main_layout.setSpacing(10)
        scroll.setWidget(container)
        outer.addWidget(scroll)

        # Missatge inicial / error
        self._message_label = QLabel(
            "<div style='text-align:center; padding:40px; color:#888;'>"
            "<span style='font-size:14px;'>Selecciona una SEQ_CAL al Dashboard</span>"
            "</div>"
        )
        self._message_label.setWordWrap(True)
        self._main_layout.addWidget(self._message_label)

        # Secció regressió (inicialment oculta)
        self._build_regression_section()

        # Secció aplicar (inicialment oculta)
        self._build_apply_section()

        self._main_layout.addStretch()

    # ------------------------------------------------------------------
    # Interface pública (cridada per GlobalCalibrationPanel)
    # ------------------------------------------------------------------

    def show_processing_message(self, seq_name):
        """Mostra missatge de processament."""
        self._message_label.setText(
            f"<div style='text-align:center; padding:40px;'>"
            f"<span style='font-size:14px; color:#2980B9;'>"
            f"⏳ Processant {seq_name}...</span></div>"
        )
        self._message_label.setVisible(True)
        self.seq_cal_group.setVisible(False)
        self.seq_cal_apply_group.setVisible(False)

    def show_error_message(self, msg):
        """Mostra missatge d'error."""
        self._message_label.setText(
            f"<div style='text-align:center; padding:20px;'>"
            f"<span style='font-size:14px; color:#E74C3C;'>❌ {msg}</span></div>"
        )
        self._message_label.setVisible(True)
        self.seq_cal_group.setVisible(False)
        self.seq_cal_apply_group.setVisible(False)

    def load_seq_cal_data(self, seq_name, seq_path, seq_cal_data,
                          calib_result=None, imported_data=None):
        """Punt d'entrada principal: rebre dades de CalSeqWorker i mostrar-les.

        Equivalent a l'antic _check_and_show_seq_cal() del wizard.
        """
        self._seq_name = seq_name
        self._seq_path = seq_path
        self._imported_data = imported_data
        self._message_label.setVisible(False)

        entries = seq_cal_data.get('entries', [])
        method = seq_cal_data.get('method', 'COLUMN')
        concs = seq_cal_data.get('concs', [])

        if not entries:
            self.show_error_message(f"SEQ_CAL {seq_name} sense entrades vàlides.")
            return

        self._seq_cal_entries = entries
        self._seq_cal_entries_direct = seq_cal_data.get('entries_direct', [])
        self._seq_cal_entries_uib = seq_cal_data.get('entries_uib', [])
        self._seq_cal_method = method

        # Auto-excloure punts amb UIB saturada
        self._seq_cal_excluded = set()
        for i, e in enumerate(entries):
            if e.get('uib_saturated'):
                self._seq_cal_excluded.add(i)

        # Configurar selector senyal Direct/UIB
        has_direct = seq_cal_data.get('has_direct', len(self._seq_cal_entries_direct) > 0)
        has_uib = seq_cal_data.get('has_uib', len(self._seq_cal_entries_uib) > 0)

        self.seq_cal_signal_combo.blockSignals(True)
        self.seq_cal_signal_combo.clear()
        if has_direct:
            self.seq_cal_signal_combo.addItem("DOC Direct", "direct")
        if has_uib:
            self.seq_cal_signal_combo.addItem("DOC UIB", "uib")
        # Default: direct si disponible
        self._seq_cal_signal = "direct" if has_direct else "uib"
        self.seq_cal_signal_combo.setCurrentIndex(0)
        self.seq_cal_signal_combo.blockSignals(False)

        # Visibilitat: combo senyal només si hi ha ambdós senyals
        has_both_signals = has_direct and has_uib
        self.seq_cal_signal_label.setVisible(has_both_signals)
        self.seq_cal_signal_combo.setVisible(has_both_signals)
        self.seq_cal_signal_frame.setVisible(has_both_signals)

        # Sensibilitat UIB de la seqüència
        self._seq_cal_sensitivity = None
        if has_uib and self._seq_cal_entries_uib:
            for e in self._seq_cal_entries_uib:
                s = e.get('uib_sensitivity')
                if s:
                    self._seq_cal_sensitivity = s
                    break

        # Checkbox reparació: visible només si hi ha entrades amb àrea reparada
        any_repaired = any(
            e.get('area_original') and e.get('area_original') != e.get('area', 0)
            for entries_list in (self._seq_cal_entries_direct, self._seq_cal_entries_uib)
            for e in entries_list
        )
        self.seq_cal_repair_check.setVisible(any_repaired)
        if any_repaired:
            self.seq_cal_repair_check.setChecked(True)
            self.seq_cal_signal_frame.setVisible(True)

        # Info text
        conc_str = ", ".join(f"{c:g}" for c in concs)
        signals_str = []
        if has_direct:
            signals_str.append(f"Direct ({len(self._seq_cal_entries_direct)})")
        if has_uib:
            sens_info = f", sens={self._seq_cal_sensitivity} ppb" if self._seq_cal_sensitivity else ""
            signals_str.append(f"UIB ({len(self._seq_cal_entries_uib)}{sens_info})")
        self.seq_cal_info.setText(
            f"<b>Seqüència de calibració: {seq_name}</b> — "
            f"{len(entries)} punts, {len(concs)} concentracions "
            f"({conc_str} ppm), mode {method}<br>"
            f"Senyals disponibles: {', '.join(signals_str)}"
        )

        # Auto-excloure per barreja sensibilitats UIB
        self._check_uib_sensitivity_mixing()

        # Executar regressió
        self._run_seq_cal_regression(entries, method)

        self.seq_cal_group.setVisible(True)

        # Populate apply section
        self._populate_apply_section()

    # ------------------------------------------------------------------
    # Backward-compat stubs (cridats per GlobalCalibrationPanel antic)
    # ------------------------------------------------------------------

    def set_data(self, cal_entries):
        """Stub — les dades ara arriben via load_seq_cal_data."""
        pass

    def pre_select_seq(self, seq_name):
        """Stub — la selecció és implícita a load_seq_cal_data."""
        pass

    def set_seq_cal_data(self, seq_name, seq_cal_data):
        """Stub — les dades ara arriben via load_seq_cal_data."""
        pass

    def set_active_calib_result(self, seq_name, calib_result):
        """Stub — les dades ara arriben via load_seq_cal_data."""
        pass

    # ------------------------------------------------------------------
    # SECCIÓ 1: REGRESSIÓ (recuperat de analyze_panel/panel.py)
    # ------------------------------------------------------------------

    def _build_regression_section(self):
        """Construeix la secció de regressió SEQ_CAL."""
        self.seq_cal_group = QGroupBox("Regressió de Calibració (SEQ_CAL)")
        self.seq_cal_group.setVisible(False)
        self.seq_cal_group.setStyleSheet(
            "QGroupBox { font-weight: bold; color: #1A5276; border: 2px solid #27AE60; "
            "border-radius: 6px; margin-top: 8px; padding-top: 14px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; }"
        )
        seq_cal_layout = QVBoxLayout(self.seq_cal_group)
        seq_cal_layout.setSpacing(10)

        # Info detecció
        self.seq_cal_info = QLabel()
        self.seq_cal_info.setWordWrap(True)
        self.seq_cal_info.setStyleSheet(
            "background: #EBF5FB; border-radius: 4px; padding: 8px; "
            "color: #1A5276; font-weight: normal; font-size: 12px;"
        )
        seq_cal_layout.addWidget(self.seq_cal_info)

        # Selector senyal Direct/UIB
        signal_frame = QFrame()
        signal_frame.setStyleSheet(
            "QFrame { background: #EBF5FB; border-radius: 4px; padding: 6px; }"
        )
        signal_layout = QHBoxLayout(signal_frame)
        signal_layout.setContentsMargins(8, 4, 8, 4)

        self.seq_cal_signal_label = QLabel("Senyal de calibració:")
        self.seq_cal_signal_label.setStyleSheet(
            "font-weight: bold; color: #1A5276; font-size: 11px;"
        )
        signal_layout.addWidget(self.seq_cal_signal_label)

        self.seq_cal_signal_combo = QComboBox()
        self.seq_cal_signal_combo.setMaximumWidth(220)
        self.seq_cal_signal_combo.setStyleSheet(
            "QComboBox { background: white; border: 1px solid #BDC3C7; "
            "border-radius: 3px; padding: 4px 8px; font-size: 11px; }"
        )
        self.seq_cal_signal_combo.currentIndexChanged.connect(
            self._on_seq_cal_signal_changed
        )
        signal_layout.addWidget(self.seq_cal_signal_combo)

        signal_layout.addSpacing(20)

        self.seq_cal_repair_check = QCheckBox("Usar àrea reparada")
        self.seq_cal_repair_check.setChecked(True)
        self.seq_cal_repair_check.setToolTip(
            "Quan activat, usa l'àrea corregida (paràbola) per pics amb cim irregular.\n"
            "Desactivar per usar l'àrea original sense reparació."
        )
        self.seq_cal_repair_check.setStyleSheet("font-size: 11px; color: #1A5276;")
        self.seq_cal_repair_check.stateChanged.connect(self._on_seq_cal_repair_toggled)
        signal_layout.addWidget(self.seq_cal_repair_check)

        signal_layout.addStretch()

        self.seq_cal_signal_frame = signal_frame
        seq_cal_layout.addWidget(signal_frame)

        # Warning sensibilitat UIB barrejada
        self.seq_cal_sensitivity_warning = QLabel()
        self.seq_cal_sensitivity_warning.setWordWrap(True)
        self.seq_cal_sensitivity_warning.setStyleSheet(
            "background: #FCF3CF; border: 1px solid #F39C12; border-radius: 4px; "
            "padding: 8px; color: #7D6608; font-size: 11px; font-weight: normal;"
        )
        self.seq_cal_sensitivity_warning.setVisible(False)
        seq_cal_layout.addWidget(self.seq_cal_sensitivity_warning)

        # Taula de punts de calibració
        self.seq_cal_points_table = QTableWidget()
        self.seq_cal_points_table.setColumnCount(11)
        self.seq_cal_points_table.setHorizontalHeaderLabels([
            "Sel", "Condició", "Conc (ppm)", "Vol (µL)", "µg DOC",
            "Àrea", "RF_mass", "A254", "DOC/254", "Anomalies", "Status"
        ])
        self.seq_cal_points_table.horizontalHeaderItem(0).setToolTip("Incloure punt a la regressió")
        self.seq_cal_points_table.horizontalHeaderItem(4).setToolTip("µg DOC injectat = ppm × µL / 1000")
        self.seq_cal_points_table.horizontalHeaderItem(6).setToolTip("RF_mass = Àrea × 1000 / (ppm × µL)")
        self.seq_cal_points_table.horizontalHeaderItem(7).setToolTip("Àrea integrada a 254nm (DAD)")
        self.seq_cal_points_table.horizontalHeaderItem(8).setToolTip("Ratio àrea DOC / àrea 254nm")
        self.seq_cal_points_table.horizontalHeaderItem(9).setToolTip("Indicadors d'anomalies detectades")
        self.seq_cal_points_table.horizontalHeaderItem(10).setToolTip("Selecció rèpliques (R1, R2, Promig)")
        self.seq_cal_points_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.seq_cal_points_table.setAlternatingRowColors(True)
        self.seq_cal_points_table.verticalHeader().setVisible(False)
        self.seq_cal_points_table.setStyleSheet("""
            QTableWidget { font-size: 11px; gridline-color: #ddd; }
            QTableWidget::item { padding: 2px 4px; }
            QHeaderView::section {
                background-color: #f5f5f5; font-weight: bold;
                font-size: 10px; padding: 4px; border: none;
                border-bottom: 2px solid #ddd;
            }
        """)
        seq_cal_layout.addWidget(self.seq_cal_points_table)

        # Connectar click a la taula per preview cromatograma
        self.seq_cal_points_table.cellClicked.connect(self._on_seq_cal_row_clicked)


        # Chromatogram preview via popup (no inline)
        self._has_seq_cal_chrom = True

        # Resultats regressió
        reg_results_frame = QFrame()
        reg_results_frame.setStyleSheet(
            "QFrame { background: #F8F9FA; border: 1px solid #DEE2E6; "
            "border-radius: 4px; padding: 8px; }"
        )
        reg_grid = QGridLayout(reg_results_frame)
        reg_grid.setSpacing(6)

        self.seq_cal_labels = {}
        reg_items = [
            ("rf", "RF (slope):", 0, 0),
            ("intercept", "Intercept:", 0, 2),
            ("r2", "R²:", 1, 0),
            ("n_points", "Punts:", 1, 2),
            ("rms", "RMS residuals:", 2, 0),
            ("model", "Model:", 2, 2),
        ]
        for key, label_text, row, col in reg_items:
            lbl = QLabel(label_text)
            lbl.setStyleSheet("font-weight: bold; color: #2C3E50; font-size: 11px;")
            reg_grid.addWidget(lbl, row, col)
            val = QLabel("-")
            val.setStyleSheet("font-size: 13px;")
            self.seq_cal_labels[key] = val
            reg_grid.addWidget(val, row, col + 1)

        seq_cal_layout.addWidget(reg_results_frame)

        # Comparació amb calibració vigent
        self.seq_cal_comparison = QLabel()
        self.seq_cal_comparison.setWordWrap(True)
        self.seq_cal_comparison.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.seq_cal_comparison.setStyleSheet(
            "padding: 8px; font-size: 12px; background: #FEFEFE; "
            "border: 1px solid #E0E0E0; border-radius: 4px;"
        )
        seq_cal_layout.addWidget(self.seq_cal_comparison)

        # Also expose as comparison_label for backward compat
        self.comparison_label = self.seq_cal_comparison

        # Gràfic scatter + residuals (matplotlib)
        try:
            self._seq_cal_figure = Figure(figsize=(8, 4), dpi=100)
            self._seq_cal_figure.set_facecolor("#FAFAFA")
            self.seq_cal_graph = FigureCanvas(self._seq_cal_figure)
            self.seq_cal_graph.setMinimumHeight(320)
            seq_cal_layout.addWidget(self.seq_cal_graph)
            self._has_seq_cal_mpl = True
        except Exception:
            self._has_seq_cal_mpl = False
            self.seq_cal_graph = QLabel("(Gràfic no disponible — instal·lar matplotlib)")
            seq_cal_layout.addWidget(self.seq_cal_graph)

        # Botó recalcular
        seq_cal_buttons = QHBoxLayout()
        seq_cal_buttons.addStretch()

        self.seq_cal_recalc_btn = QPushButton("Recalcular")
        self.seq_cal_recalc_btn.setToolTip("Recalcular regressió amb els punts seleccionats")
        self.seq_cal_recalc_btn.clicked.connect(self._on_seq_cal_recalculate)
        self.seq_cal_recalc_btn.setStyleSheet(
            "QPushButton { background: #3498DB; color: white; border: none; "
            "border-radius: 4px; padding: 8px 16px; font-weight: bold; }"
            "QPushButton:hover { background: #2980B9; }"
        )
        seq_cal_buttons.addWidget(self.seq_cal_recalc_btn)

        seq_cal_layout.addLayout(seq_cal_buttons)

        self._main_layout.addWidget(self.seq_cal_group)

    # ------------------------------------------------------------------
    # SECCIÓ 2: APLICAR CALIBRACIÓ (recuperat de review_summary_panel.py)
    # ------------------------------------------------------------------

    def _build_apply_section(self):
        """Crea la secció per aplicar calibració (només visible per SEQ_CAL)."""
        self.seq_cal_apply_group = QGroupBox("APLICAR CALIBRACIÓ (SEQ_CAL)")
        self.seq_cal_apply_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold; font-size: 12px;
                border: 2px solid #2980B9;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 24px;
                background-color: #f0f7ff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
                color: #2980B9;
            }
        """)
        self.seq_cal_apply_group.setVisible(False)

        layout = QVBoxLayout(self.seq_cal_apply_group)
        layout.setContentsMargins(12, 8, 12, 12)
        layout.setSpacing(8)

        # --- Equació resum (compacta, sense duplicar la secció de regressió) ---
        self._cal_equation_label = QLabel("")
        self._cal_equation_label.setAlignment(Qt.AlignCenter)
        self._cal_equation_label.setStyleSheet("""
            font-family: Consolas, monospace; font-size: 12px;
            background-color: #e8f4fd; border: 1px solid #b3d7f0;
            border-radius: 4px; padding: 6px 10px;
        """)
        layout.addWidget(self._cal_equation_label)

        # --- Opcions d'aplicació ---
        opts_layout = QGridLayout()
        opts_layout.setSpacing(8)

        opts_layout.addWidget(QLabel("<b>Opcions d'aplicació:</b>"), 0, 0, 1, 3)

        # valid_from DateEdit
        opts_layout.addWidget(QLabel("Vigent des de:"), 1, 0)
        self._cal_valid_from = QDateEdit()
        self._cal_valid_from.setCalendarPopup(True)
        self._cal_valid_from.setDate(QDate.currentDate())
        self._cal_valid_from.setDisplayFormat("yyyy-MM-dd")
        opts_layout.addWidget(self._cal_valid_from, 1, 1)

        # Checkbox retroactiu
        self._cal_retroactive_chk = QCheckBox("Aplicar retroactivament")
        self._cal_retroactive_chk.setToolTip(
            "Requantifica SEQs processades després de la data de vigència\n"
            "amb els nous RF/intercept (les àrees no canvien, només ppm)"
        )
        self._cal_retroactive_chk.toggled.connect(self._on_retroactive_toggled)
        opts_layout.addWidget(self._cal_retroactive_chk, 2, 0, 1, 2)

        layout.addLayout(opts_layout)

        # --- Llista SEQs retroactives ---
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

        # Scroll area per checkboxes SEQs
        self._retro_scroll = QScrollArea()
        self._retro_scroll.setWidgetResizable(True)
        self._retro_scroll.setMaximumHeight(150)
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

        layout.addWidget(self._retro_frame)

        # --- Retro count label ---
        self._retro_count_label = QLabel("")
        self._retro_count_label.setAlignment(Qt.AlignCenter)
        self._retro_count_label.setStyleSheet("font-size: 11px; color: #666; border: none;")
        self._retro_count_label.setVisible(False)
        layout.addWidget(self._retro_count_label)

        # --- Botó aplicar ---
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._cal_apply_btn = QPushButton("Aplicar com a Nova Calibració")
        self._cal_apply_btn.setStyleSheet("""
            QPushButton {
                background-color: #27AE60; color: white;
                border: none; border-radius: 6px;
                padding: 10px 24px; font-size: 13px; font-weight: bold;
            }
            QPushButton:hover { background-color: #2ECC71; }
            QPushButton:disabled { background-color: #BDC3C7; }
        """)
        self._cal_apply_btn.clicked.connect(self._on_apply_calibration)
        btn_row.addWidget(self._cal_apply_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # Estat aplicació
        self._cal_apply_status = QLabel("")
        self._cal_apply_status.setAlignment(Qt.AlignCenter)
        self._cal_apply_status.setStyleSheet("font-size: 11px; border: none;")
        layout.addWidget(self._cal_apply_status)

        # Botó generar informe PDF (visible després d'aplicar)
        report_row = QHBoxLayout()
        report_row.addStretch()
        self._cal_report_btn = QPushButton("📄 Generar Informe Calibració (PDF)")
        self._cal_report_btn.setStyleSheet("""
            QPushButton {
                background-color: #2980B9; color: white;
                border: none; border-radius: 6px;
                padding: 8px 20px; font-size: 12px; font-weight: bold;
            }
            QPushButton:hover { background-color: #3498DB; }
            QPushButton:disabled { background-color: #BDC3C7; }
        """)
        self._cal_report_btn.setVisible(False)
        self._cal_report_btn.clicked.connect(self._on_generate_cal_report)
        report_row.addWidget(self._cal_report_btn)
        report_row.addStretch()
        layout.addLayout(report_row)

        self._main_layout.addWidget(self.seq_cal_apply_group)

    # ------------------------------------------------------------------
    # Lògica regressió (recuperada de analyze_panel/panel.py)
    # ------------------------------------------------------------------

    def _check_uib_sensitivity_mixing(self):
        """Detecta barreja de sensibilitats UIB i auto-exclou la minoria."""
        if self._seq_cal_signal != "uib":
            self.seq_cal_sensitivity_warning.setVisible(False)
            return

        entries = self._seq_cal_entries
        if not entries:
            self.seq_cal_sensitivity_warning.setVisible(False)
            return

        sens_counts = {}
        for i, e in enumerate(entries):
            if i in self._seq_cal_excluded:
                continue
            s = e.get('uib_sensitivity')
            if s and s > 0:
                sens_counts.setdefault(s, []).append(i)

        unique_sens = sorted(sens_counts.keys())

        if len(unique_sens) <= 1:
            self.seq_cal_sensitivity_warning.setVisible(False)
            return

        majority_sens = max(sens_counts, key=lambda s: len(sens_counts[s]))
        minority_count = 0
        for s, indices in sens_counts.items():
            if s != majority_sens:
                for idx in indices:
                    self._seq_cal_excluded.add(idx)
                    minority_count += 1

        sens_str = ", ".join(f"{s:g} ppb ({len(sens_counts[s])} punts)" for s in unique_sens)
        self.seq_cal_sensitivity_warning.setText(
            f"⚠️ <b>Barreja de sensibilitats UIB detectada:</b> {sens_str}<br>"
            f"S'han auto-exclòs {minority_count} punt(s) amb sensibilitat ≠ {majority_sens:g} ppb. "
            f"Una regressió amb sensibilitats barrejades no seria vàlida."
        )
        self.seq_cal_sensitivity_warning.setVisible(True)
        logger.info(f"UIB sensitivity mixing: {sens_str}, majority={majority_sens}, excluded={minority_count}")

    def _run_seq_cal_regression(self, cal_entries, method):
        """Executa la regressió i actualitza la UI."""
        enabled = []
        for i, entry in enumerate(cal_entries):
            if i in self._seq_cal_excluded:
                continue
            enabled.append(entry)

        if len(enabled) < 2:
            self.seq_cal_info.setText(
                f"<b>⚠ Insuficients punts ({len(enabled)})</b> — "
                "Mínim 2 punts per la regressió."
            )
            return

        reg_result = fit_calibration_from_history(
            enabled, mode=method, signal=self._seq_cal_signal, model="intercept"
        )
        reg_result['signal'] = self._seq_cal_signal
        reg_result['uib_sensitivity'] = self._seq_cal_sensitivity

        self._seq_cal_regression = reg_result

        self._update_seq_cal_ui(cal_entries, reg_result, method)

    def _update_seq_cal_ui(self, cal_entries, reg_result, method):
        """Actualitza tots els elements de la secció SEQ_CAL."""
        self._populate_seq_cal_table(cal_entries)

        if reg_result and reg_result.get('success'):
            rf = reg_result['rf_mass_cal']
            intercept = reg_result['intercept']
            r2 = reg_result['r2']
            n_pts = reg_result['n_points']
            rms = reg_result.get('residuals_rms', 0)

            self.seq_cal_labels['rf'].setText(f"<b>{rf:.1f}</b>")
            self.seq_cal_labels['intercept'].setText(f"{intercept:.1f}")
            r2_color = '#27AE60' if r2 >= 0.99 else '#E67E22' if r2 >= 0.95 else '#E74C3C'
            self.seq_cal_labels['r2'].setText(f"<b style='color: {r2_color}'>{r2:.6f}</b>")
            self.seq_cal_labels['n_points'].setText(f"{n_pts}")
            self.seq_cal_labels['rms'].setText(f"{rms:.2f}")
            self.seq_cal_labels['model'].setText("intercept (lliure)")

            self._update_seq_cal_comparison(rf, intercept, r2, method)
            self._update_seq_cal_graph(reg_result, method)
        else:
            error = reg_result.get('error', 'Error desconegut') if reg_result else 'No result'
            for key in self.seq_cal_labels:
                self.seq_cal_labels[key].setText("-")
            self.seq_cal_comparison.setText(
                f"<i style='color: #E74C3C;'>Regressió fallida: {error}</i>"
            )

    def _populate_seq_cal_table(self, cal_entries):
        """Omple la taula de punts de la regressió SEQ_CAL (12 columnes)."""
        # Forçar rebuild complet: clear + set row count
        self.seq_cal_points_table.clearContents()
        self.seq_cal_points_table.setRowCount(len(cal_entries))

        for i, entry in enumerate(cal_entries):
            conc = entry.get('conc_ppm', 0)
            vol = entry.get('volume_uL', 0)
            area = entry.get('area', 0)
            ug_doc = conc * vol / 1000.0
            rf_mass = entry.get('rf_mass', area / ug_doc if ug_doc > 0 else 0)

            issues = entry.get('quality_issues', [])
            has_severe = any('MULTI_PEAK' in str(iss) and 'MILD' not in str(iss) for iss in issues)
            if area <= 0 or conc <= 0 or vol <= 0:
                status_text = "INVALID"
            elif has_severe:
                status_text = "CHECK"
            elif rf_mass > 0 and (rf_mass < 100 or rf_mass > 3000):
                status_text = "CHECK"
            else:
                status_text = "OK"

            # Checkbox (Col 0)
            cb = QCheckBox()
            cb.setChecked(i not in self._seq_cal_excluded)
            cb.stateChanged.connect(lambda state, idx=i: self._on_seq_cal_point_toggled(idx, state))
            cb_widget = QWidget()
            cb_layout = QHBoxLayout(cb_widget)
            cb_layout.addWidget(cb)
            cb_layout.setAlignment(Qt.AlignCenter)
            cb_layout.setContentsMargins(0, 0, 0, 0)
            self.seq_cal_points_table.setCellWidget(i, 0, cb_widget)

            # Cols 1-6: Dades bàsiques
            items = [
                (1, entry.get('name_full', entry.get('condition_key', ''))),
                (2, f"{conc:g}"),
                (3, f"{vol:.0f}"),
                (4, f"{ug_doc:.3f}"),
                (5, f"{area:.1f}"),
                (6, f"{rf_mass:.0f}"),
            ]
            for col, text in items:
                item = QTableWidgetItem(str(text))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.seq_cal_points_table.setItem(i, col, item)

            # Col 7: A254
            a254 = entry.get('a254_area', 0)
            a254_item = QTableWidgetItem(f"{a254:.0f}" if a254 else "-")
            a254_item.setFlags(a254_item.flags() & ~Qt.ItemIsEditable)
            self.seq_cal_points_table.setItem(i, 7, a254_item)

            # Col 8: DOC/254 ratio
            ratio = entry.get('a254_doc_ratio', 0)
            if not ratio and a254 and area:
                ratio = area / a254
            ratio_item = QTableWidgetItem(f"{ratio:.2f}" if ratio else "-")
            ratio_item.setFlags(ratio_item.flags() & ~Qt.ItemIsEditable)
            # Typical DOC/254 range: ~0.3-8 depending on concentration; flag extreme outliers only
            if ratio and (ratio < 0.1 or ratio > 20):
                ratio_item.setForeground(QBrush(QColor("#E67E22")))
            self.seq_cal_points_table.setItem(i, 8, ratio_item)

            # Col 9: Anomalies
            anomaly_parts = []
            if entry.get('uib_saturated'):
                anomaly_parts.append("\u26d4 SAT")
            if entry.get('irregular_top_repaired'):
                anomaly_parts.append("\u2705 reparat")
            elif entry.get('has_irregular_top'):
                anomaly_parts.append("\u26a0 irregular")
            if entry.get('has_timeout') and entry.get('timeout_severity', 'OK') != 'OK':
                anomaly_parts.append("timeout")
            if any('MULTI_PEAK' in str(iss) for iss in issues):
                anomaly_parts.append("multi-peak")
            anomaly_text = ", ".join(anomaly_parts) if anomaly_parts else "-"
            anomaly_item = QTableWidgetItem(anomaly_text)
            anomaly_item.setFlags(anomaly_item.flags() & ~Qt.ItemIsEditable)
            if anomaly_parts:
                if any("SAT" in p for p in anomaly_parts):
                    anomaly_item.setForeground(QBrush(QColor("#E74C3C")))
                elif any("reparat" in p for p in anomaly_parts):
                    anomaly_item.setForeground(QBrush(QColor("#27AE60")))
                else:
                    anomaly_item.setForeground(QBrush(QColor("#E67E22")))
            self.seq_cal_points_table.setItem(i, 9, anomaly_item)

            # Col 10: Status — show replica selection info
            sel_info = entry.get('selection', {})
            status_display = entry.get('status', '')
            if not status_display:
                # Fallback: build from selection info
                sel_method = sel_info.get('method', '')
                sel_replicas = sel_info.get('selected_replicas', [])
                if sel_method == 'average' and len(sel_replicas) > 1:
                    status_display = f"Promig R{'+R'.join(map(str, sel_replicas))}"
                elif sel_method == 'single':
                    status_display = "R1"
                elif sel_method == 'best_quality':
                    status_display = f"Millor R{sel_replicas[0]}" if sel_replicas else "CHECK"
                elif sel_method.startswith('R'):
                    status_display = sel_method
                else:
                    status_display = status_text  # OK/CHECK/INVALID fallback

            # Color based on validity
            if status_text == "INVALID":
                status_fg = "#E74C3C"
                status_bg = "#FADBD8"
            elif status_text == "CHECK":
                status_fg = "#E67E22"
                status_bg = "#FCF3CF"
            else:
                status_fg = "#1A5276"
                status_bg = "#EBF5FB"

            badge = QLabel(f" {status_display} ")
            badge.setAlignment(Qt.AlignCenter)
            badge.setStyleSheet(
                f"background: {status_bg}; color: {status_fg}; font-weight: bold; "
                f"font-size: 10px; border-radius: 3px; padding: 1px 6px;"
            )
            # Tooltip: show details
            rsd = entry.get('rsd', 0)
            n_rep = entry.get('n_replicas', 1)
            std_a = entry.get('std_area', 0)
            tooltip_parts = [f"Rèpliques: {n_rep}"]
            if rsd > 0:
                tooltip_parts.append(f"RSD: {rsd:.1f}%")
            if std_a > 0:
                tooltip_parts.append(f"SD àrea: {std_a:.1f}")
            badge.setToolTip("\n".join(tooltip_parts))

            badge_w = QWidget()
            badge_l = QHBoxLayout(badge_w)
            badge_l.addWidget(badge)
            badge_l.setAlignment(Qt.AlignCenter)
            badge_l.setContentsMargins(2, 1, 2, 1)
            self.seq_cal_points_table.setCellWidget(i, 10, badge_w)

        # Ajustar alçada: totes les files visibles + scroll si n'hi ha moltes
        row_h = self.seq_cal_points_table.verticalHeader().defaultSectionSize()
        header_h = self.seq_cal_points_table.horizontalHeader().height()
        desired = header_h + row_h * len(cal_entries) + 4
        self.seq_cal_points_table.setMinimumHeight(min(desired, 500))
        self.seq_cal_points_table.setMaximumHeight(max(desired, 200))

    def _on_seq_cal_row_clicked(self, row, col):
        """Clic: obre popup amb cromatograma complet + inset zoom al pic."""
        if not getattr(self, '_has_seq_cal_chrom', False):
            return
        if row < 0 or row >= len(self._seq_cal_entries):
            return

        self._last_seq_cal_chrom_row = row
        entry = self._seq_cal_entries[row]
        replicas = entry.get('replicas', [])
        if not replicas:
            return

        use_repaired = self.seq_cal_repair_check.isChecked()

        try:
            # Crear popup dialog
            conc = entry.get('conc_ppm', 0)
            name = entry.get('name_full', entry.get('condition_key', ''))
            dialog = QDialog(self)
            dialog.setWindowTitle(f"Cromatograma — {name} ({conc:g} ppm)")
            dialog.resize(900, 500)
            dlg_layout = QVBoxLayout(dialog)
            dlg_layout.setContentsMargins(4, 4, 4, 4)

            fig = Figure(figsize=(10, 5), dpi=100)
            fig.set_facecolor("#FAFAFA")
            canvas = FigureCanvas(fig)
            dlg_layout.addWidget(canvas)

            ax = fig.add_subplot(111)

            doc_colors = ['#2196F3', '#1565C0']
            doc_styles = ['-', '--']
            fill_colors = ['#2196F3', '#1565C0']
            dad_colors = ['#9B59B6', '#8E44AD']
            dad_styles = ['-', ':']
            ax2 = None
            peak_times = []
            plot_data = []

            for r_idx, rep in enumerate(replicas[:2]):
                r_label = f"R{r_idx + 1}"
                color = doc_colors[r_idx]
                style = doc_styles[r_idx]

                t_doc = rep.get('t_doc')
                y_doc = rep.get('y_doc')
                y_repaired = rep.get('y_doc_repaired')

                if t_doc is not None and y_doc is not None:
                    t_doc = np.asarray(t_doc)
                    y_doc = np.asarray(y_doc)
                    ax.plot(t_doc, y_doc, color=color, linewidth=1.2,
                            linestyle=style, label=f'{r_label} DOC',
                            alpha=0.9 if r_idx == 0 else 0.6)

                    y_rep_arr = None
                    if y_repaired is not None:
                        y_rep_arr = np.asarray(y_repaired)
                        ax.plot(t_doc, y_rep_arr, color='#E74C3C', linewidth=1,
                                linestyle='--', label=f'{r_label} Reparat',
                                alpha=0.7 if r_idx == 0 else 0.4)

                    peak_info = rep.get('peak_info', {})
                    t_start = peak_info.get('t_start')
                    t_end = peak_info.get('t_end')
                    if t_start is not None and t_end is not None:
                        peak_times.extend([t_start, t_end])
                        mask = (t_doc >= t_start) & (t_doc <= t_end)
                        if np.any(mask):
                            if use_repaired and y_rep_arr is not None:
                                y_fill = y_rep_arr[mask]
                            else:
                                y_fill = y_doc[mask]
                            ax.fill_between(t_doc[mask], 0, y_fill,
                                           color=fill_colors[r_idx], alpha=0.12)
                        if r_idx == 0:
                            ax.axvline(t_start, color='gray', linewidth=0.5,
                                      linestyle=':', alpha=0.6)
                            ax.axvline(t_end, color='gray', linewidth=0.5,
                                      linestyle=':', alpha=0.6)

                    plot_data.append({
                        't': t_doc, 'y': y_doc, 'y_rep': y_rep_arr,
                        'color': color, 'style': style, 'fill_color': fill_colors[r_idx],
                        't_start': t_start, 't_end': t_end,
                    })

                t_dad = rep.get('t_dad')
                y_254 = rep.get('y_dad_254')
                if t_dad is not None and y_254 is not None:
                    t_dad = np.asarray(t_dad)
                    y_254 = np.asarray(y_254)
                    if ax2 is None:
                        ax2 = ax.twinx()
                        ax2.set_ylabel('254nm', color='#9B59B6', fontsize=9)
                        ax2.tick_params(axis='y', labelcolor='#9B59B6', labelsize=8)
                    ax2.plot(t_dad, y_254, color=dad_colors[r_idx], linewidth=0.8,
                            linestyle=dad_styles[r_idx],
                            label=f'{r_label} 254nm',
                            alpha=0.6 if r_idx == 0 else 0.35)

            n_rep = min(len(replicas), 2)
            repair_tag = " [reparat]" if use_repaired and entry.get('irregular_top_repaired') else ""
            ax.set_title(f"{name} ({conc:g} ppm) — {n_rep} rèpliques{repair_tag}",
                        fontsize=11, fontweight='bold')
            ax.set_xlabel('Temps (min)', fontsize=10)
            ax.set_ylabel('Senyal DOC', fontsize=10, color='#2196F3')
            ax.tick_params(labelsize=9)

            lines, labels = ax.get_legend_handles_labels()
            if ax2:
                lines2, labels2 = ax2.get_legend_handles_labels()
                lines += lines2
                labels += labels2
            ax.legend(lines, labels, loc='upper left', fontsize=8, ncol=2)

            # Inset zoom al pic principal
            if peak_times and plot_data:
                t_min_pk = min(peak_times)
                t_max_pk = max(peak_times)
                t_center = (t_min_pk + t_max_pk) / 2
                pk_width = t_max_pk - t_min_pk
                margin = max(2.0 if self._seq_cal_method != "BP" else 1.5, pk_width * 0.5)
                inset_left = t_center - margin
                inset_right = t_center + margin

                t_full = plot_data[0]['t']
                t_range = float(t_full[-1] - t_full[0]) if len(t_full) > 1 else 1.0
                peak_rel = (t_center - float(t_full[0])) / t_range if t_range > 0 else 0.5
                if peak_rel > 0.5:
                    inset_pos = [0.08, 0.45, 0.35, 0.45]
                else:
                    inset_pos = [0.55, 0.45, 0.35, 0.45]

                ax_inset = ax.inset_axes(inset_pos)
                ax_inset.set_xlim(inset_left, inset_right)

                y_max_inset = 0
                for pd in plot_data:
                    imask = (pd['t'] >= inset_left) & (pd['t'] <= inset_right)
                    if not np.any(imask):
                        continue
                    ax_inset.plot(pd['t'][imask], pd['y'][imask],
                                 color=pd['color'], linewidth=1.0, linestyle=pd['style'])
                    if pd['y_rep'] is not None:
                        ax_inset.plot(pd['t'][imask], pd['y_rep'][imask],
                                     color='#E74C3C', linewidth=0.8, linestyle='--')
                    y_max_inset = max(y_max_inset, float(np.max(pd['y'][imask])))

                    if pd['t_start'] is not None and pd['t_end'] is not None:
                        fmask = imask & (pd['t'] >= pd['t_start']) & (pd['t'] <= pd['t_end'])
                        if np.any(fmask):
                            if use_repaired and pd['y_rep'] is not None:
                                y_f = pd['y_rep'][fmask]
                            else:
                                y_f = pd['y'][fmask]
                            ax_inset.fill_between(pd['t'][fmask], 0, y_f,
                                                  color=pd['fill_color'], alpha=0.2)

                if y_max_inset > 0:
                    ax_inset.set_ylim(-y_max_inset * 0.05, y_max_inset * 1.15)
                ax_inset.tick_params(labelsize=7)
                ax_inset.set_title('Zoom pic', fontsize=8, pad=2)
                ax_inset.patch.set_alpha(0.9)
                for spine in ax_inset.spines.values():
                    spine.set_edgecolor('#888')
                    spine.set_linewidth(0.5)

                ax.axvspan(inset_left, inset_right, alpha=0.06, color='#3498DB')

            try:
                fig.tight_layout()
            except Exception:
                pass
            canvas.draw()

            dialog.exec()
        except Exception as e:
            logger.warning(f"Error preview cromatograma: {e}", exc_info=True)

    def _on_seq_cal_point_toggled(self, idx, state):
        """Quan l'usuari marca/desmarca un punt de la regressió → recalcula immediatament."""
        if state == 0:
            self._seq_cal_excluded.add(idx)
        else:
            self._seq_cal_excluded.discard(idx)

        # Recalcular regressió immediatament
        if self._seq_cal_entries and self._seq_cal_method:
            self._run_seq_cal_regression(self._seq_cal_entries, self._seq_cal_method)
            self._populate_apply_section()

    def _on_seq_cal_signal_changed(self, index):
        """Quan l'usuari canvia el senyal (Direct/UIB) del selector."""
        if index < 0:
            return
        signal = self.seq_cal_signal_combo.itemData(index)
        if not signal or signal == self._seq_cal_signal:
            return

        self._seq_cal_signal = signal

        if signal == "uib" and self._seq_cal_entries_uib:
            self._seq_cal_entries = self._seq_cal_entries_uib
        elif signal == "direct" and self._seq_cal_entries_direct:
            self._seq_cal_entries = self._seq_cal_entries_direct

        self._seq_cal_excluded = set()
        for i, e in enumerate(self._seq_cal_entries):
            if e.get('uib_saturated'):
                self._seq_cal_excluded.add(i)

        self._check_uib_sensitivity_mixing()

        if self._seq_cal_entries and self._seq_cal_method:
            self._run_seq_cal_regression(self._seq_cal_entries, self._seq_cal_method)
            self._populate_apply_section()

    def _on_seq_cal_recalculate(self):
        """Recalcula la regressió amb els punts seleccionats."""
        if self._seq_cal_entries and self._seq_cal_method:
            self._run_seq_cal_regression(self._seq_cal_entries, self._seq_cal_method)
            # Actualitzar la secció d'aplicar
            self._populate_apply_section()

    def _on_seq_cal_repair_toggled(self, state):
        """Toggle entre àrea reparada i àrea original per la regressió."""
        use_repaired = (state != 0)

        changed = 0
        seen = set()
        for entries in (self._seq_cal_entries_direct, self._seq_cal_entries_uib):
            for entry in entries:
                if id(entry) in seen:
                    continue
                seen.add(id(entry))
                area_orig = entry.get('area_original')
                area_rep = entry.get('area_repaired')
                conc = entry.get('conc_ppm', 0)
                if not area_orig or not area_rep:
                    continue
                old_area = entry['area']
                if use_repaired:
                    entry['area'] = area_rep
                else:
                    entry['area'] = area_orig
                if entry['area'] != old_area:
                    changed += 1
                    logger.info("Repair toggle: conc=%s area %.1f -> %.1f",
                                conc, old_area, entry['area'])
                vol = entry.get('volume_uL', 0)
                if conc > 0 and vol > 0:
                    entry['rf_mass'] = entry['area'] * 1000.0 / (conc * vol)
                if 'area_u' in entry:
                    entry['area_u'] = entry['area']

        logger.info("Repair toggle: use_repaired=%s, changed=%d", use_repaired, changed)

        if self._seq_cal_entries and self._seq_cal_method:
            self._run_seq_cal_regression(self._seq_cal_entries, self._seq_cal_method)
            # Actualitzar TAMBÉ la secció d'aplicar (que té la seva còpia dels valors)
            self._populate_apply_section()

        # El cromatograma ara es mostra en popup (no cal redibuixar)

    def _update_seq_cal_comparison(self, new_rf, new_intercept, new_r2, method):
        """Mostra la comparació entre calibració vigent i la nova."""
        from gui.widgets.analyze_panel._helpers import format_calibration_comparison_html

        current_cal = get_active_global_calibration()
        if not current_cal:
            self.seq_cal_comparison.setText("<i>No hi ha calibració vigent per comparar</i>")
            return

        signal = self._seq_cal_signal

        rf_cal = current_cal.get('rf_mass_cal', {})
        intercept_cal = current_cal.get('intercept', 0)

        if isinstance(rf_cal, dict):
            current_rf = rf_cal.get(signal, {}).get(method.lower(), 0)
        else:
            current_rf = float(rf_cal) if rf_cal else 0

        if isinstance(intercept_cal, dict):
            current_intercept = intercept_cal.get(signal, {}).get(method.lower(), 0)
        else:
            current_intercept = float(intercept_cal) if intercept_cal else 0

        current_r2 = current_cal.get('r2', {})
        if isinstance(current_r2, dict):
            current_r2_val = current_r2.get(method.lower(), 0) or 0
        else:
            current_r2_val = float(current_r2) if current_r2 else 0

        html = format_calibration_comparison_html(
            rf_vigent=current_rf, int_vigent=current_intercept,
            rf_new=new_rf, int_new=new_intercept,
            r2_new=new_r2, r2_vigent=current_r2_val,
            show_equation=True,
        )
        self.seq_cal_comparison.setText(html)

    def _update_seq_cal_graph(self, reg_result, method):
        """Actualitza el gràfic scatter de regressió SEQ_CAL."""
        if not getattr(self, '_has_seq_cal_mpl', False):
            return
        try:
            points = reg_result.get('points', [])
            if not points:
                self._seq_cal_figure.clear()
                self.seq_cal_graph.draw()
                return

            self._seq_cal_figure.clear()
            gs = self._seq_cal_figure.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.35)
            ax_main = self._seq_cal_figure.add_subplot(gs[0])
            ax_res = self._seq_cal_figure.add_subplot(gs[1])

            excluded = self._seq_cal_excluded
            x_all, y_all, x_inc, y_inc, x_exc, y_exc = [], [], [], [], [], []
            labels = []
            for i, p in enumerate(points):
                conc = p.get('conc_ppm', 0)
                vol = p.get('volume_uL', 0)
                x_val = conc * vol / 1000.0
                y_val = p.get('area', 0)
                x_all.append(x_val)
                y_all.append(y_val)
                labels.append(f"{conc:g} ppm")
                if i in excluded:
                    x_exc.append(x_val)
                    y_exc.append(y_val)
                else:
                    x_inc.append(x_val)
                    y_inc.append(y_val)

            # Scatter per concentració amb error bars (SD) si hi ha rèpliques
            conc_groups = {}
            # Build SD list aligned with x_inc/y_inc
            sd_inc = []
            for i, p in enumerate(points):
                if i not in excluded:
                    sd_inc.append(p.get('std_area', 0))
            labels_inc = [l for i, l in enumerate(labels) if i not in excluded]

            for xi, yi, sdi, lbl in zip(x_inc, y_inc, sd_inc, labels_inc):
                conc_groups.setdefault(lbl, ([], [], []))
                conc_groups[lbl][0].append(xi)
                conc_groups[lbl][1].append(yi)
                conc_groups[lbl][2].append(sdi)

            cmap_colors = ['#2980B9', '#27AE60', '#8E44AD', '#E67E22', '#E74C3C', '#1ABC9C',
                           '#34495E', '#F39C12', '#D35400', '#7F8C8D']
            for idx_c, (lbl, (xs, ys, sds)) in enumerate(sorted(conc_groups.items())):
                c = cmap_colors[idx_c % len(cmap_colors)]
                has_sd = any(s > 0 for s in sds)
                if has_sd:
                    ax_main.errorbar(xs, ys, yerr=sds, fmt='o', color=c, markersize=7,
                                     zorder=5, markeredgecolor='white', markeredgewidth=0.8,
                                     ecolor=c, elinewidth=1.2, capsize=3, capthick=1,
                                     label=lbl)
                else:
                    ax_main.scatter(xs, ys, c=c, s=60, zorder=5,
                                    edgecolors='white', linewidths=0.8, label=lbl)

            if x_exc:
                ax_main.scatter(x_exc, y_exc, c='#E74C3C', s=50, marker='x',
                                zorder=4, linewidths=1.5, label='Exclosos')

            new_rf = reg_result.get('rf_mass_cal', 0)
            new_intercept = reg_result.get('intercept', 0)
            r2 = reg_result.get('r2', 0)
            n_pts = reg_result.get('n_points', len(x_inc))
            x_max = max(x_all) * 1.1 if x_all else 1
            x_line = np.linspace(0, x_max, 100)
            y_line = new_rf * x_line + new_intercept
            ax_main.plot(x_line, y_line, '-', color='#27AE60', linewidth=2)

            if abs(new_intercept) > 0.5:
                eq_text = f"A = {new_rf:.1f} × µg + {new_intercept:.1f}"
            else:
                eq_text = f"A = {new_rf:.1f} × µg"
            eq_text += f"   (R²={r2:.4f}, n={n_pts})"
            ax_main.text(0.03, 0.97, eq_text, transform=ax_main.transAxes,
                         fontsize=8, fontfamily='monospace', verticalalignment='top',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                   edgecolor='#ccc', alpha=0.9))

            # Banda predicció 95%
            if len(x_inc) >= 3:
                try:
                    from gui.widgets.analyze_panel._helpers import compute_prediction_band
                    band = compute_prediction_band(x_line, new_rf, new_intercept,
                                                   np.array(x_inc), np.array(y_inc))
                    if band:
                        ax_main.fill_between(x_line, band[0], band[1],
                                            alpha=0.10, color='#27AE60',
                                            label='Predicció 95%')
                except Exception:
                    pass

            # Recta vigent (referència)
            from hpsec_calibrate import get_rf_mass_cal, get_calibration_intercept
            current_rf = get_rf_mass_cal(signal='direct', mode=method.lower()) or 0
            current_intercept = get_calibration_intercept(signal='direct', mode=method.lower()) or 0
            if current_rf > 0:
                y_current = current_rf * x_line + current_intercept
                ax_main.plot(x_line, y_current, '--', color='#E67E22', linewidth=1.5, alpha=0.7,
                             label=f'Vigent: RF={current_rf:.0f}, int={current_intercept:.1f}')

            ax_main.set_xlabel('µg DOC injectat', fontsize=9)
            ax_main.set_ylabel('Àrea DOC', fontsize=9)
            ax_main.set_title(f'Recta de Calibració {method}', fontsize=10, fontweight='bold')
            ax_main.legend(fontsize=7, loc='upper left')
            ax_main.set_xlim(left=0)
            ax_main.set_ylim(bottom=min(0, min(y_all) - 10) if y_all else 0)
            ax_main.grid(True, alpha=0.3)
            ax_main.tick_params(labelsize=8)

            # Residuals
            residuals = []
            for i, p in enumerate(points):
                if i in excluded:
                    continue
                x_val = p.get('conc_ppm', 0) * p.get('volume_uL', 0) / 1000.0
                y_val = p.get('area', 0)
                y_pred = new_rf * x_val + new_intercept
                residuals.append(y_val - y_pred)

            if residuals:
                rms = reg_result.get('residuals_rms', 0)
                colors = ['#27AE60' if abs(r) < rms * 2 else '#E67E22' if abs(r) < rms * 3 else '#E74C3C'
                          for r in residuals]
                ax_res.bar(range(len(residuals)), residuals, color=colors, alpha=0.8, edgecolor='white')
                ax_res.axhline(y=0, color='#333', linewidth=0.8)
                if rms > 0:
                    ax_res.axhline(y=rms, color='#E67E22', linewidth=0.5, linestyle='--', alpha=0.5)
                    ax_res.axhline(y=-rms, color='#E67E22', linewidth=0.5, linestyle='--', alpha=0.5)

                inc_pts = [p for j, p in enumerate(points) if j not in excluded]
                if inc_pts:
                    ax_res.set_xticks(range(len(inc_pts)))
                    ax_res.set_xticklabels([f"{p.get('conc_ppm', 0):g}" for p in inc_pts],
                                           fontsize=6, rotation=45)
                    ax_res.set_xlabel('ppm', fontsize=7)

                ax_res.set_title('Residuals', fontsize=9, fontweight='bold')
                ax_res.set_ylabel('Àrea obs - pred', fontsize=8)
                ax_res.tick_params(labelsize=7)
                ax_res.grid(True, alpha=0.2, axis='y')

            try:
                self._seq_cal_figure.tight_layout()
            except Exception:
                pass  # twinx axes: tight_layout pot fallar
            self.seq_cal_graph.draw()

        except Exception as e:
            logger.warning(f"Error actualitzant gràfic SEQ_CAL: {e}")
            try:
                self._seq_cal_figure.clear()
                self.seq_cal_graph.draw()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Lògica aplicar (recuperada de review_summary_panel.py)
    # ------------------------------------------------------------------

    def _populate_apply_section(self):
        """Omple la secció aplicar amb dades de la regressió actual."""
        if not self._seq_cal_regression or not self._seq_cal_regression.get('success'):
            self.seq_cal_apply_group.setVisible(False)
            return

        reg = self._seq_cal_regression
        rf_new = reg.get('rf_mass_cal', 0)
        intercept_new = reg.get('intercept', 0)
        r2 = reg.get('r2', 0)
        n_pts = reg.get('n_points', 0)

        # Equació resum (la info detallada ja és a la secció de regressió)
        r2_color = COLOR_SUCCESS if r2 >= 0.99 else (COLOR_WARNING if r2 >= 0.95 else COLOR_ERROR)
        self._cal_equation_label.setText(
            f"Àrea = {rf_new:.1f} × µg_DOC + {intercept_new:.1f}   "
            f"(R² = <span style='color:{r2_color}'>{r2:.6f}</span>, n={n_pts})"
        )

        # Date: usar data de la SEQ si disponible
        if self._imported_data:
            seq_date = self._imported_data.get('date')
            if seq_date:
                try:
                    from datetime import datetime as dt
                    d = dt.strptime(str(seq_date)[:10], '%Y-%m-%d')
                    self._cal_valid_from.setDate(QDate(d.year, d.month, d.day))
                except (ValueError, TypeError):
                    pass

        # Check si ja aplicada
        if self._cal_applied:
            self._cal_apply_btn.setEnabled(False)
            self._cal_apply_status.setText(
                f"<span style='color:{COLOR_SUCCESS}'>&#10003; Calibració ja aplicada</span>"
            )
        else:
            self._cal_apply_btn.setEnabled(True)
            self._cal_apply_status.setText("")

        # Llista SEQs retroactives
        self._populate_retro_seq_list()

        self.seq_cal_apply_group.setVisible(True)

    def _on_retroactive_toggled(self, checked):
        """Mostra/amaga la llista de SEQs retroactives."""
        self._retro_frame.setVisible(checked)
        self._retro_count_label.setVisible(checked)
        if checked and not self._retro_seq_checkboxes:
            self._populate_retro_seq_list()
        if checked:
            self._update_retro_count()

    def _populate_retro_seq_list(self):
        """Carrega llista de SEQs processades posteriors a valid_from."""
        for cb in self._retro_seq_checkboxes:
            cb.deleteLater()
        self._retro_seq_checkboxes = []

        from hpsec_config import get_config
        cfg = get_config()
        data_folder = cfg.get("paths", "data_folder", default="")
        if not data_folder or not Path(data_folder).is_dir():
            self._retro_info_label.setText("No s'ha trobat el data_folder.")
            return

        seq_list = []
        current_name = self._seq_name

        for item in sorted(Path(data_folder).iterdir()):
            if not item.is_dir() or '_SEQ' not in item.name.upper():
                continue
            if item.name == current_name:
                continue
            if '_CAL' in item.name.upper():
                continue
            json_path = item / "CHECK" / "data" / "analysis_result.json"
            if not json_path.exists():
                continue
            seq_list.append((item.name, str(json_path)))

        if not seq_list:
            self._retro_info_label.setText("No s'han trobat SEQs processades per requantificar.")
            return

        self._retro_info_label.setText(
            f"<b>{len(seq_list)} SEQs</b> processades trobades. "
            f"Selecciona les que vols requantificar amb la nova calibració:"
        )

        for seq_name, json_path in seq_list:
            cb = QCheckBox(seq_name)
            cb.setChecked(True)
            cb.setProperty("json_path", json_path)
            cb.setStyleSheet("border: none; background: transparent; font-size: 10px;")
            cb.toggled.connect(self._update_retro_count)
            self._retro_content_layout.addWidget(cb)
            self._retro_seq_checkboxes.append(cb)

        self._update_retro_count()

    def _update_retro_count(self, _=None):
        """Actualitza el comptador de SEQs retroactives seleccionades."""
        total = len(self._retro_seq_checkboxes)
        selected = sum(1 for cb in self._retro_seq_checkboxes if cb.isChecked())
        self._retro_count_label.setText(
            f"<b>{selected}/{total}</b> SEQs seleccionades per requantificar"
        )

    def _select_all_retro(self, select):
        """Selecciona o deselecciona totes les SEQs retroactives."""
        for cb in self._retro_seq_checkboxes:
            cb.setChecked(select)

    def _on_apply_calibration(self):
        """Aplica la nova calibració (add_calibration + requantificació retroactiva)."""
        if not self._seq_cal_regression or not self._seq_cal_regression.get('success'):
            QMessageBox.warning(self, "Avís", "No hi ha regressió vàlida per aplicar.")
            return

        rf_new = self._seq_cal_regression.get('rf_mass_cal', 0)
        intercept_new = self._seq_cal_regression.get('intercept', 0)
        r2 = self._seq_cal_regression.get('r2', 0)
        n_pts = self._seq_cal_regression.get('n_points', 0)
        method = (self._seq_cal_method or "COLUMN").upper()
        is_bp = "BP" in method

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

        valid_from = self._cal_valid_from.date().toString("yyyy-MM-dd")
        retroactive = self._cal_retroactive_chk.isChecked()

        retro_count = sum(1 for cb in self._retro_seq_checkboxes if cb.isChecked()) if retroactive else 0
        msg = (
            f"S'aplicarà la nova calibració:\n\n"
            f"  Mode: {method}\n"
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
        self._cal_apply_btn.setEnabled(False)
        self._cal_apply_status.setText("Aplicant...")

        try:
            from hpsec_calibrate import (
                add_calibration,
                get_rf_mass_cal, get_calibration_intercept,
                requantify_analysis_json,
            )

            # Construir rf_mass_cal_values preservant l'altra branca
            current_cal = get_active_global_calibration()
            if current_cal:
                rf_values = copy.deepcopy(current_cal.get('rf_mass_cal', {}))
                intercept_values = current_cal.get('intercept', {})
                if isinstance(intercept_values, dict):
                    intercept_values = copy.deepcopy(intercept_values)
                else:
                    intercept_values = {"direct": {"column": 0, "bp": 0}}
            else:
                rf_values = {"direct": {"column": 0, "bp": 0}, "uib": {"column": 0, "bp": 0}}
                intercept_values = {"direct": {"column": 0, "bp": 0}, "uib": {"column": 0, "bp": 0}}

            cal_signal = (self._seq_cal_regression or {}).get('signal', 'direct')
            mode_key = "bp" if is_bp else "column"

            if isinstance(rf_values, dict) and cal_signal in rf_values:
                if isinstance(rf_values[cal_signal], dict):
                    rf_values[cal_signal][mode_key] = rf_new
                else:
                    rf_values[cal_signal] = {mode_key: rf_new}
            else:
                rf_values[cal_signal] = {mode_key: rf_new}

            if isinstance(intercept_values, dict) and cal_signal in intercept_values:
                if isinstance(intercept_values[cal_signal], dict):
                    intercept_values[cal_signal][mode_key] = intercept_new
                else:
                    intercept_values[cal_signal] = {mode_key: intercept_new}
            else:
                intercept_values[cal_signal] = {mode_key: intercept_new}

            # Source info
            source = {
                "type": "SEQ_CAL",
                "description": f"Regressió from {self._seq_name}",
                "seq_references": [self._seq_name],
                "mode": method,
            }

            reg_data = dict(self._seq_cal_regression) if self._seq_cal_regression else {}
            reg_data['mode'] = method
            reg_data['signal'] = cal_signal
            reg_data['model'] = reg_data.get('model', 'intercept')

            cal_id = add_calibration(
                rf_mass_cal_values=rf_values,
                source=source,
                valid_from=valid_from,
                r2=r2,
                n_points=n_pts,
                reason=f"SEQ_CAL tab5: {self._seq_name}",
                intercept_values=intercept_values,
                regression_data=reg_data,
            )

            if not cal_id:
                raise RuntimeError("add_calibration ha retornat None")

            logger.info(f"Nova calibració aplicada: {cal_id} (RF={rf_new:.1f}, mode={method})")

            self._cal_applied = True

            # --- Requantificació retroactiva ---
            retro_results = []
            if retroactive and retro_count > 0:
                self._cal_apply_status.setText(f"Requantificant {retro_count} SEQs...")

                new_cal = get_active_global_calibration()
                rf_col = get_rf_mass_cal(new_cal, signal=cal_signal, mode="column")
                int_col = get_calibration_intercept(new_cal, signal=cal_signal, mode="column")
                rf_bp = get_rf_mass_cal(new_cal, signal=cal_signal, mode="bp")
                int_bp = get_calibration_intercept(new_cal, signal=cal_signal, mode="bp")

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

            status_parts = [f"<span style='color:{COLOR_SUCCESS}'>&#10003; Calibració {cal_id} aplicada</span>"]
            if retro_results:
                status_parts.append(f"<br>Requantificades: {n_ok} OK")
                if n_fail:
                    status_parts.append(f", <span style='color:{COLOR_ERROR}'>{n_fail} errors</span>")

            self._cal_apply_status.setText("".join(status_parts))
            self._cal_apply_btn.setEnabled(False)
            self._cal_report_btn.setVisible(True)

            # Refrescar dashboard
            mw = self.parent_panel.main_window if self.parent_panel else None
            if mw and hasattr(mw, 'dashboard_panel') and mw.dashboard_panel:
                try:
                    mw.dashboard_panel.refresh_sequences()
                except Exception:
                    pass

        except Exception as e:
            logger.error(f"Error aplicant calibració: {e}")
            self._cal_apply_status.setText(
                f"<span style='color:{COLOR_ERROR}'>Error: {e}</span>"
            )
            self._cal_apply_btn.setEnabled(True)

    def _on_generate_cal_report(self):
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
