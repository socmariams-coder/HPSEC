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
    QComboBox, QMessageBox, QCheckBox, QRadioButton, QButtonGroup,
    QFrame, QProgressBar, QLineEdit, QScrollArea, QDialog, QSpinBox
)
from PySide6.QtCore import Qt, Signal, QThread, QDate
from PySide6.QtGui import QFont, QColor, QBrush

from pathlib import Path
import sys
import os
import re
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
    load_manual_repairs, manual_repair_key,
)

import matplotlib  # noqa: F401
# matplotlib.use('QtAgg') eliminat: forçava el backend interactiu i obria finestres
# fantasma en generar informes. Backend fixat a Agg (hpsec_suite_qt.py); embedding
# via FigureCanvasQTAgg explícit, que no depèn del backend per defecte.
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

    def __init__(self, seq_path, config=None, force_reimport=False):
        super().__init__()
        self.seq_path = seq_path
        self.config = config
        self.force_reimport = force_reimport

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
            if self.force_reimport:
                # Reimportar des de zero (ignora manifest)
                progress_cb(5, "Reimportant des del MasterFile...")
                imported_data = import_sequence(
                    self.seq_path,
                    config=self.config,
                    progress_callback=lambda p, m: progress_cb(5 + int(p * 0.3), m),
                )
            else:
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
                # No calibrar amb senyals buits: avortar si la càrrega ha fallat
                if imported_data.get("load_error") or imported_data.get("data_deferred"):
                    self.error.emit(
                        f"No s'han pogut carregar les dades de {seq_name}: "
                        + str(imported_data.get("load_error", "dades incompletes"))
                        + ". No es calibra.")
                    return

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
    """Panell de calibració global amb mini-dashboard de SEQ_CALs.

    Mini-dashboard a dalt: llista totes les SEQ_CAL, permet processar/reprocessar.
    CalibrationLineView a baix: regressió, scatter, aplicar.
    """

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self._active_seq_path = None
        self._result_cache = {}
        self._batch_queue = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)

        # === MINI-DASHBOARD SEQ_CAL ===
        dash_header = QHBoxLayout()
        dash_header.addWidget(QLabel("<b>Seqüències de calibració</b>"))
        dash_header.addStretch()

        self._btn_refresh = QPushButton("Actualitzar")
        self._btn_refresh.setFixedWidth(80)
        self._btn_refresh.setStyleSheet("font-size: 11px;")
        self._btn_refresh.clicked.connect(self._refresh_cal_list)
        dash_header.addWidget(self._btn_refresh)

        self._btn_process = QPushButton("Processar")
        self._btn_process.setStyleSheet(
            "QPushButton { background-color: #4A90A4; color: white; "
            "font-weight: bold; padding: 3px 10px; border-radius: 3px; font-size: 11px; }"
            "QPushButton:hover { background-color: #3A7A8E; }")
        self._btn_process.setToolTip("Processar la SEQ_CAL seleccionada")
        self._btn_process.clicked.connect(self._on_process_selected)
        dash_header.addWidget(self._btn_process)

        self._btn_reprocess = QPushButton("Reprocessar")
        self._btn_reprocess.setStyleSheet(
            "QPushButton { background-color: #E67E22; color: white; "
            "font-weight: bold; padding: 3px 10px; border-radius: 3px; font-size: 11px; }"
            "QPushButton:hover { background-color: #D35400; }")
        self._btn_reprocess.setToolTip("Reimportar des de zero (MasterFile) i reprocessar")
        self._btn_reprocess.clicked.connect(self._on_reprocess_selected)
        dash_header.addWidget(self._btn_reprocess)

        self._btn_combine = QPushButton("Combinar")
        self._btn_combine.setStyleSheet(
            "QPushButton { background-color: #27AE60; color: white; "
            "font-weight: bold; padding: 3px 10px; border-radius: 3px; font-size: 11px; }"
            "QPushButton:hover { background-color: #1E8449; }")
        self._btn_combine.setToolTip("Combinar SEQ_CALs seleccionades en una sola regressió")
        self._btn_combine.clicked.connect(self._on_combine_selected)
        dash_header.addWidget(self._btn_combine)

        layout.addLayout(dash_header)

        # Taula SEQ_CAL
        self._cal_table = QTableWidget()
        self._cal_table.setColumnCount(8)
        self._cal_table.setHorizontalHeaderLabels([
            "Nom", "Mètode", "Data", "Estat", "Inj", "KHP", "Conc", "Vol"])
        self._cal_table.horizontalHeader().setStretchLastSection(False)
        self._cal_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for i in range(1, 8):
            self._cal_table.horizontalHeader().setSectionResizeMode(
                i, QHeaderView.ResizeMode.ResizeToContents)
        self._cal_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._cal_table.setSelectionMode(QTableWidget.SelectionMode.MultiSelection)
        self._cal_table.setMaximumHeight(140)
        self._cal_table.setAlternatingRowColors(True)
        self._cal_table.verticalHeader().setVisible(False)
        self._cal_table.setStyleSheet("QTableWidget { font-size: 11px; }")
        self._cal_table.doubleClicked.connect(self._on_cal_table_dblclick)
        layout.addWidget(self._cal_table)

        self._cal_seq_list = []  # [{path, name, method, ...}]

        # Barra de progrés
        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setStyleSheet(
            "QProgressBar { border: 1px solid #bdc3c7; border-radius: 3px; "
            "text-align: center; height: 18px; }"
            "QProgressBar::chunk { background-color: #2980B9; border-radius: 2px; }")
        layout.addWidget(self._progress_bar)

        self._progress_label = QLabel("")
        self._progress_label.setVisible(False)
        self._progress_label.setStyleSheet("color: #2980B9; font-style: italic; font-size: 11px;")
        layout.addWidget(self._progress_label)

        # === CALIBRATION LINE VIEW ===
        self.cal_view = CalibrationLineView(self)
        layout.addWidget(self.cal_view, 1)

        # Worker
        self._cal_worker = None
        self._summary_shown = False
        self._force_reimport = False

    # ------------------------------------------------------------------
    # Mini-dashboard
    # ------------------------------------------------------------------

    def _refresh_cal_list(self):
        """Escaneja carpetes de dades i pobla la taula de SEQ_CALs."""
        from hpsec_config import get_data_folders
        from gui.models.sequence_state import get_all_sequences
        import json as _json

        folders = get_data_folders()
        all_seqs = get_all_sequences(folders)

        self._cal_seq_list = []
        for seq in all_seqs:
            if "_CAL" not in seq.seq_name.upper():
                continue
            # Info del MasterFile/manifest
            check_dir = os.path.join(seq.seq_path, "CHECK", "data")
            has_cal = os.path.exists(os.path.join(check_dir, "calibration_result.json"))
            has_manifest = os.path.exists(os.path.join(check_dir, "import_manifest.json"))

            n_khp = seq.n_khp
            n_inj = seq.n_inj_master
            method = seq.method or ("BP" if "BP" in seq.seq_name.upper() else "COLUMN")
            date = seq.seq_date or ""

            # Volum i concentracions del calibration_result si existeix
            vol = ""
            concs = ""
            if has_cal:
                try:
                    with open(os.path.join(check_dir, "calibration_result.json"),
                              'r', encoding='utf-8') as f:
                        cr = _json.load(f)
                    cals = cr.get('calibrations', [])
                    if cals:
                        vols = set(c.get('volume_uL', 0) for c in cals if c.get('volume_uL'))
                        vol = "/".join(str(int(v)) for v in sorted(vols))
                        cs = sorted(set(c.get('conc_ppm', 0) for c in cals if c.get('conc_ppm', 0) > 0))
                        concs = ", ".join(f"{c:g}" for c in cs)
                except Exception:
                    pass

            status = "\u2713" if has_cal else ("\u25d0" if has_manifest else "\u25cb")

            self._cal_seq_list.append({
                'path': seq.seq_path, 'name': seq.seq_name,
                'method': method, 'date': date, 'status': status,
                'n_inj': n_inj, 'n_khp': n_khp, 'concs': concs, 'vol': vol,
                'processed': has_cal,
            })

        self._populate_cal_table()

    def _populate_cal_table(self):
        self._cal_table.setRowCount(len(self._cal_seq_list))
        for i, s in enumerate(self._cal_seq_list):
            self._cal_table.setItem(i, 0, QTableWidgetItem(s['name']))

            m_item = QTableWidgetItem(s['method'])
            m_item.setForeground(QColor('#EF4444' if s['method'] == 'BP' else '#2E86C1'))
            self._cal_table.setItem(i, 1, m_item)

            self._cal_table.setItem(i, 2, QTableWidgetItem(str(s['date'])[:10]))

            st_item = QTableWidgetItem(s['status'])
            if s['status'] == '\u2713':
                st_item.setForeground(QColor('#27AE60'))
            self._cal_table.setItem(i, 3, st_item)

            self._cal_table.setItem(i, 4, QTableWidgetItem(str(s['n_inj']) if s['n_inj'] else ""))
            self._cal_table.setItem(i, 5, QTableWidgetItem(str(s['n_khp']) if s['n_khp'] else ""))
            self._cal_table.setItem(i, 6, QTableWidgetItem(s['concs']))
            self._cal_table.setItem(i, 7, QTableWidgetItem(s['vol']))

    def _get_selected_cal_path(self):
        """Retorna el path de la primera SEQ_CAL seleccionada."""
        rows = self._cal_table.selectionModel().selectedRows()
        if not rows:
            return None
        row = rows[0].row()
        if row < len(self._cal_seq_list):
            return self._cal_seq_list[row]['path']
        return None

    def _get_selected_cal_paths(self):
        """Retorna llista de paths de totes les SEQ_CAL seleccionades."""
        rows = self._cal_table.selectionModel().selectedRows()
        paths = []
        for idx in rows:
            row = idx.row()
            if row < len(self._cal_seq_list):
                paths.append(self._cal_seq_list[row]['path'])
        return paths

    def _on_cal_table_dblclick(self, index):
        """Doble clic: processar la SEQ_CAL."""
        path = self._get_selected_cal_path()
        if path:
            self._force_reimport = False
            self.load_seq_cal(path)

    def _on_process_selected(self):
        """Botó Processar: processa la SEQ_CAL seleccionada (usa cache si disponible)."""
        path = self._get_selected_cal_path()
        if not path:
            QMessageBox.information(self, "Info", "Selecciona una SEQ_CAL a la taula.")
            return
        self._force_reimport = False
        self.load_seq_cal(path)

    def _on_combine_selected(self):
        """Combinar múltiples SEQ_CALs: processar totes i ajuntar punts."""
        paths = self._get_selected_cal_paths()
        if len(paths) < 2:
            QMessageBox.information(self, "Info",
                "Selecciona almenys 2 SEQ_CALs a la taula per combinar.\n"
                "(Usa Ctrl+clic per selecció múltiple)")
            return

        names = [os.path.basename(p) for p in paths]
        logger.info(f"Combinar SEQ_CALs: {names}")

        # Processar les que no estan al cache
        self._combine_paths = paths
        self._combine_pending = [p for p in paths if p not in self._result_cache]
        self._combine_done = []

        if self._combine_pending:
            # Processar la primera pendent
            self._start_cal_worker(self._combine_pending[0])
        else:
            # Totes al cache — combinar directament
            self._do_combine()

    def _do_combine(self):
        """Combina resultats de múltiples SEQ_CALs i passa a CalibrationLineView."""
        if not hasattr(self, '_combine_paths'):
            return

        paths = self._combine_paths
        all_entries_direct = []
        all_entries_uib = []
        methods = set()
        imported_list = []

        for path in paths:
            result = self._result_cache.get(path)
            if not result:
                continue
            seq_cal_data = result.get('seq_cal_data')
            if not seq_cal_data:
                logger.warning(f"_do_combine: {os.path.basename(path)} sense seq_cal_data, ignorant")
                continue
            seq_name = result.get('seq_name', os.path.basename(path))
            method = seq_cal_data.get('method', 'COLUMN')
            methods.add(method)

            # Taguejar cada entry amb la SEQ d'origen
            for entry in seq_cal_data.get('entries_direct', []):
                entry['_source_seq'] = seq_name
                entry['_source_method'] = method
                all_entries_direct.append(entry)

            for entry in seq_cal_data.get('entries_uib', []):
                entry['_source_seq'] = seq_name
                entry['_source_method'] = method
                all_entries_uib.append(entry)

            if result.get('imported_data'):
                imported_list.append(result['imported_data'])

        if not all_entries_direct and not all_entries_uib:
            QMessageBox.warning(self, "Error", "Cap entrada KHP trobada a les SEQs seleccionades.")
            return

        # Construir seq_cal_data combinat
        combined = {
            'entries': all_entries_direct or all_entries_uib,
            'entries_direct': all_entries_direct,
            'entries_uib': all_entries_uib,
            'method': '+'.join(sorted(methods)),
            'concs': sorted(set(e.get('conc_ppm', 0) for e in (all_entries_direct or all_entries_uib))),
            'signal': 'direct' if all_entries_direct else 'uib',
            'has_direct': bool(all_entries_direct),
            'has_uib': bool(all_entries_uib),
            '_combined': True,
            '_source_seqs': [os.path.basename(p) for p in paths],
        }

        names = [os.path.basename(p) for p in paths]
        combo_name = " + ".join(names)

        self.cal_view.load_seq_cal_data(
            combo_name, paths[0], combined,
            imported_data=imported_list[0] if imported_list else None,
        )

        self.main_window.set_status(f"Combinat: {combo_name}", 5000)

        # Netejar
        del self._combine_paths

    def _on_reprocess_selected(self):
        """Botó Reprocessar: reimporta des de zero (ignora manifest i cache)."""
        path = self._get_selected_cal_path()
        if not path:
            QMessageBox.information(self, "Info", "Selecciona una SEQ_CAL a la taula.")
            return
        seq_name = os.path.basename(path)
        reply = QMessageBox.question(
            self, "Reprocessar",
            f"Reimportar {seq_name} des del MasterFile i reprocessar?\n\n"
            "Això ignora el manifest existent i reimporta tot de zero.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        # Invalidar cache
        if path in self._result_cache:
            del self._result_cache[path]
        self._force_reimport = True
        self._start_cal_worker(path)

    # ------------------------------------------------------------------
    # showEvent / load
    # ------------------------------------------------------------------

    def showEvent(self, event):
        """Pobla mini-dashboard i mostra summary."""
        super().showEvent(event)
        self._refresh_cal_list()
        if self._active_seq_path is None and not self._summary_shown:
            self.cal_view.show_summary()
            self._summary_shown = True

    def load_seq_cal(self, seq_path):
        """Carrega una SEQ_CAL."""
        self._active_seq_path = seq_path
        seq_name = os.path.basename(seq_path)
        logger.info(f"load_seq_cal: {seq_name}")

        if not self._force_reimport and seq_path in self._result_cache:
            logger.info(f"  Cache hit per {seq_name}")
            self._on_worker_finished(self._result_cache[seq_path], from_cache=True)
            return

        self._start_cal_worker(seq_path)

    def _start_cal_worker(self, seq_path):
        """Llança CalSeqWorker per importar i calibrar una SEQ_CAL."""
        if self._cal_worker and self._cal_worker.isRunning():
            logger.warning("CalSeqWorker ja en execució, ignorant nova petició")
            return

        seq_name = os.path.basename(seq_path)

        self._progress_bar.setVisible(True)
        self._progress_bar.setValue(0)
        self._progress_label.setVisible(True)
        self._progress_label.setText(f"Processant {seq_name}...")

        self.cal_view.show_processing_message(seq_name)

        self._cal_worker = CalSeqWorker(seq_path, force_reimport=self._force_reimport)
        self._cal_worker.progress.connect(self._on_worker_progress)
        self._cal_worker.finished.connect(self._on_worker_finished)
        self._cal_worker.error.connect(self._on_worker_error)
        self._cal_worker.start()
        self._force_reimport = False

    def _on_worker_progress(self, pct, msg):
        """Actualitza barra de progrés."""
        self._progress_bar.setValue(pct)
        self._progress_label.setText(msg)

    def _on_worker_finished(self, result, from_cache=False):
        """Worker completat: passar dades a CalibrationLineView o continuar batch."""
        self._progress_bar.setVisible(False)
        self._progress_label.setVisible(False)

        seq_name = result.get("seq_name", "")
        seq_path = result.get("seq_path", "")
        logger.info(f"CalSeqWorker completat per {seq_name}")

        # Guardar al cache
        if not from_cache and seq_path:
            self._result_cache[seq_path] = result

        # Si estem en mode combinació, continuar amb la següent o combinar
        if hasattr(self, '_combine_pending') and self._combine_pending:
            # Treure la que acabem de processar
            if seq_path in self._combine_pending:
                self._combine_pending.remove(seq_path)

            if self._combine_pending:
                # Processar la següent
                self._start_cal_worker(self._combine_pending[0])
                return
            else:
                # Totes processades — combinar
                self._do_combine()
                return

        # Mode normal: passar a la vista
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
        self._sel_combos = {}
        self._cal_applied_per_signal = {}   # {'direct': True, 'uib': False}
        self._cal_applied_signals = set()   # senyals ja aplicats exitosament

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

        # Secció resum (sense SEQ_CAL carregada)
        self._build_summary_section()

        # Secció alineació TOC+DAD (inicialment oculta)
        self._build_alignment_section()

        # Secció regressió (inicialment oculta)
        self._build_regression_section()

        # Secció aplicar (inicialment oculta)
        self._build_apply_section()

        self._main_layout.addStretch()

    # ------------------------------------------------------------------
    # Alignment section (TOC+DAD254 continu + delay per injecció)
    # ------------------------------------------------------------------

    def _build_alignment_section(self):
        """Secció alineació: TOC+DAD254 superposats + delay per injecció."""
        self._align_group = QGroupBox("Alineació TOC — DAD 254")
        self._align_group.setVisible(False)
        self._align_group.setStyleSheet(
            "QGroupBox { font-weight: bold; color: #1D4ED8; border: 1px solid #AED6F1; "
            "border-radius: 4px; margin-top: 6px; padding-top: 10px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 3px; }")
        align_layout = QVBoxLayout(self._align_group)

        # Info label
        self._align_info = QLabel()
        self._align_info.setWordWrap(True)
        self._align_info.setStyleSheet("font-size: 11px; padding: 2px;")
        align_layout.addWidget(self._align_info)

        # Chart
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.figure import Figure
            self._align_figure = Figure(figsize=(10, 3.5), dpi=100)
            self._align_figure.set_facecolor('#FAFAFA')
            self._align_canvas = FigureCanvas(self._align_figure)
            self._align_canvas.setMinimumHeight(250)
            align_layout.addWidget(self._align_canvas)
            self._has_align_chart = True
        except ImportError:
            self._has_align_chart = False

        # Delay table
        self._align_delay_table = QLabel()
        self._align_delay_table.setStyleSheet(
            "font-size: 10px; font-family: monospace; padding: 2px;")
        self._align_delay_table.setWordWrap(True)
        align_layout.addWidget(self._align_delay_table)

        self._main_layout.addWidget(self._align_group)

    def _update_alignment(self, imported_data):
        """Actualitza la secció d'alineació amb dades de la SEQ_CAL importada."""
        if not imported_data:
            self._align_group.setVisible(False)
            return

        import numpy as np

        align = imported_data.get("bp_alignment")
        method = imported_data.get("method", "COLUMN")
        is_bp = method.upper() == "BP"

        # Per COLUMN: construir alineació si no existeix
        if not align and not is_bp:
            # Intentar construir TOC+DAD continu per COLUMN
            align = self._build_column_alignment(imported_data)

        if not align:
            self._align_group.setVisible(False)
            return

        self._align_group.setVisible(True)

        per_inj = align.get("per_injection", [])
        n_inj = align.get("n_injections", 0)
        n_matched = align.get("n_matched", 0)
        delay_med = align.get("delay_median", 0)
        delay_min = align.get("delay_min", 0)
        delay_max = align.get("delay_max", 0)
        drift = align.get("delay_drift", 0)
        toc_timeouts = align.get("toc_timeouts", [])
        delay_blocks = align.get("delay_blocks", [])

        info = (f"<b>{n_inj}</b> injeccions, <b>{n_matched}</b> delays calculats, "
                f"<b>{len(toc_timeouts)}</b> timeouts<br>"
                f"Delay: mediana <b>{delay_med:.1f}</b> min "
                f"(rang {delay_min:.1f}–{delay_max:.1f}, drift {drift:+.1f})")
        if abs(drift) > 2:
            info += (f"<br><span style='color:#E67E22'>Drift significatiu: "
                     f"{drift:+.1f} min — els timeouts desplacen el delay</span>")
        self._align_info.setText(info)

        # Dibuixar gràfic: 2 subplots
        if self._has_align_chart:
            toc_data = align.get("toc_continuous", {})
            dad_data = align.get("dad254_continuous", {})
            t_toc = np.array(toc_data.get("t", []))
            y_toc = np.array(toc_data.get("y", []))
            t_dad = np.array(dad_data.get("t", []))
            y_dad = np.array(dad_data.get("y", []))

            # Recollir KHP cromatogrames Direct vs UIB
            samples = imported_data.get("samples", {})
            khp_chroms = []
            for sn, sd in samples.items():
                if 'khp' not in sn.lower():
                    continue
                for rk, rd in sorted(sd.get("replicas", {}).items()):
                    direct = rd.get("direct", {})
                    uib = rd.get("uib", {})
                    t_d = direct.get("t") if isinstance(direct, dict) else None
                    y_d = direct.get("y_net", direct.get("y")) if isinstance(direct, dict) else None
                    t_u = uib.get("t") if isinstance(uib, dict) else None
                    y_u = uib.get("y_net", uib.get("y")) if isinstance(uib, dict) else None
                    if t_d is not None and len(t_d) > 5:
                        khp_chroms.append({
                            "name": sn, "rep": rk,
                            "t_d": np.array(t_d), "y_d": np.array(y_d) if y_d is not None else None,
                            "t_u": np.array(t_u) if t_u is not None else None,
                            "y_u": np.array(y_u) if y_u is not None else None,
                        })

            has_uib = any(k.get("y_u") is not None and len(k["y_u"]) > 5 for k in khp_chroms)
            n_subplots = 1 + (1 if khp_chroms else 0)

            self._align_figure.clear()

            # Subplot 1: TOC + DAD continu
            ax = self._align_figure.add_subplot(n_subplots, 1, 1)

            if len(t_toc) > 0:
                ax.plot(t_toc, y_toc, 'b-', lw=0.3, alpha=0.7)
                ax.set_ylabel('TOC (ppb)', color='blue', fontsize=8)

            if len(t_dad) > 0:
                ax2 = ax.twinx()
                ax2.plot(t_dad, y_dad, 'g-', lw=0.2, alpha=0.5)
                ax2.set_ylabel('DAD 254 (mAU)', color='green', fontsize=8)

            for to in toc_timeouts:
                ax.axvline(to['t_min'], c='red', lw=0.4, ls='--', alpha=0.3)

            for p in per_inj:
                if not p.get('is_control', False) and 'khp' in p.get('name', '').lower():
                    ax.axvline(p['t_hplc'], c='#E74C3C', lw=0.8, ls=':', alpha=0.5)

            mode_str = "BP" if is_bp else "COLUMN"
            ax.set_title(f'TOC (blau) + DAD 254 (verd) | {mode_str} | '
                        f'delay {delay_med:.1f} min, drift {drift:+.1f}',
                        fontsize=8)
            ax.spines['top'].set_visible(False)
            ax.grid(True, alpha=0.1)

            # Subplot 2: KHP cromatogrames Direct (blau) vs UIB (verd)
            if khp_chroms:
                ax_khp = self._align_figure.add_subplot(n_subplots, 1, 2)
                colors_d = ['#1565C0', '#1976D2', '#1E88E5', '#2196F3', '#42A5F5',
                            '#64B5F6', '#90CAF9', '#BBDEFB']
                colors_u = ['#2E7D32', '#388E3C', '#43A047', '#4CAF50', '#66BB6A',
                            '#81C784', '#A5D6A7', '#C8E6C9']

                for i, k in enumerate(khp_chroms):
                    cd = colors_d[i % len(colors_d)]
                    cu = colors_u[i % len(colors_u)]
                    label_d = f'{k["name"][:6]} R{k["rep"]} D'
                    ax_khp.plot(k["t_d"], k["y_d"], color=cd, lw=0.8, alpha=0.7,
                               label=label_d)
                    if k.get("y_u") is not None and len(k["y_u"]) > 5:
                        label_u = f'{k["name"][:6]} R{k["rep"]} U'
                        ax_khp.plot(k["t_u"], k["y_u"], color=cu, lw=0.8,
                                   alpha=0.7, ls='--', label=label_u)

                ax_khp.axhline(0, c='k', lw=0.3, alpha=0.3)
                uib_str = " + UIB (verd --)" if has_uib else ""
                ax_khp.set_title(f'KHP: Direct (blau){uib_str}', fontsize=8)
                ax_khp.set_xlabel('min', fontsize=8)
                ax_khp.set_ylabel('ppb', fontsize=8)
                ax_khp.legend(fontsize=6, loc='upper right', ncol=2)
                ax_khp.grid(True, alpha=0.1)
                if is_bp:
                    ax_khp.set_xlim(0, 10)

            try:
                self._align_figure.tight_layout()
            except Exception:
                pass
            self._align_canvas.draw_idle()

        # Taula delays (3 columnes)
        lines = []
        for p in per_inj:
            if p.get('is_control'):
                continue
            name = p.get('name', '?')[:12]
            d = p.get('delay_real')
            d_str = f"{d:.1f}" if d is not None else "  -"
            lines.append(f"{name:>12} {d_str:>5}")

        if lines:
            n = len(lines)
            cols = 3
            rows = (n + cols - 1) // cols
            table_lines = []
            for r in range(rows):
                parts = []
                for c in range(cols):
                    idx = r + c * rows
                    if idx < n:
                        parts.append(lines[idx])
                table_lines.append("  |  ".join(parts))
            self._align_delay_table.setText("\n".join(table_lines))

    def _build_column_alignment(self, imported_data):
        """Construeix alineació TOC+DAD254 per COLUMN (com build_bp_continuous_dad254)."""
        try:
            from hpsec_import import build_bp_continuous_dad254
            # Necessitem master_data.toc — pot no estar disponible
            md = imported_data.get("master_data", {})
            if md and md.get("toc") is not None:
                n = build_bp_continuous_dad254(imported_data)
                if n > 0:
                    return imported_data.get("bp_alignment")
        except Exception as e:
            logger.debug("_build_column_alignment failed: %s", e)
        return None

    # ------------------------------------------------------------------
    # Summary section (sense SEQ_CAL carregada)
    # ------------------------------------------------------------------

    def _build_summary_section(self):
        """Secció resum calibracions vigents (visible quan no hi ha SEQ_CAL carregada)."""
        self._summary_group = QGroupBox()
        self._summary_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #bdc3c7; border-radius: 6px;
                margin-top: 0px; padding: 12px;
                background-color: #FAFAFA;
            }
        """)
        self._summary_group.setVisible(False)
        layout = QVBoxLayout(self._summary_group)
        layout.setSpacing(10)

        # Header
        header = QLabel(
            "<div style='font-size:14px; font-weight:bold; color:#2C3E50; "
            "background-color:#D6EAF8; padding:8px 12px; border-radius:4px;'>"
            "Calibracions Vigents</div>"
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        # Taula paràmetres actius
        self._summary_params_table = QTableWidget()
        self._summary_params_table.setColumnCount(9)
        self._summary_params_table.setHorizontalHeaderLabels([
            "Àmbit", "Sens.", "Mode", "RF", "Intercept", "R²",
            "Rang (µg)", "Font", "Equació"
        ])
        hdr = self._summary_params_table.horizontalHeader()
        for col in range(9):
            if col in (7, 8):  # Font, Equació — stretch
                hdr.setSectionResizeMode(col, QHeaderView.Stretch)
            else:
                hdr.setSectionResizeMode(col, QHeaderView.ResizeToContents)
        self._summary_params_table.verticalHeader().setVisible(False)
        self._summary_params_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._summary_params_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._summary_params_table.setSelectionMode(QTableWidget.SingleSelection)
        self._summary_params_table.setAlternatingRowColors(True)
        self._summary_params_table.setMaximumHeight(180)
        self._summary_params_table.setStyleSheet("""
            QTableWidget::item:selected {
                background-color: #D6EAF8; color: #2C3E50;
            }
        """)
        self._summary_params_table.cellClicked.connect(self._on_summary_row_clicked)
        layout.addWidget(self._summary_params_table)
        # Dades associades a cada fila (key, cal) per al scatter
        self._summary_row_data = []

        # Scatter regressió (si disponible)
        self._summary_figure = Figure(figsize=(7, 3), dpi=100)
        self._summary_canvas = FigureCanvas(self._summary_figure)
        self._summary_canvas.setMinimumHeight(220)
        self._summary_canvas.setVisible(False)
        layout.addWidget(self._summary_canvas)

        # Taula historial calibracions
        hist_label = QLabel(
            "<div style='font-size:12px; font-weight:bold; color:#2C3E50; "
            "margin-top:8px;'>Historial de Calibracions</div>"
        )
        layout.addWidget(hist_label)

        self._summary_history_table = QTableWidget()
        self._summary_history_table.setColumnCount(8)
        self._summary_history_table.setHorizontalHeaderLabels([
            "ID", "Àmbit", "Sens.", "Vigent des de", "Fins a", "RF (col)", "RF (bp)", "Activa?"
        ])
        hhdr = self._summary_history_table.horizontalHeader()
        for col in range(8):
            if col in (3, 4):  # Vigent des de, Fins a — stretch
                hhdr.setSectionResizeMode(col, QHeaderView.Stretch)
            else:
                hhdr.setSectionResizeMode(col, QHeaderView.ResizeToContents)
        self._summary_history_table.verticalHeader().setVisible(False)
        self._summary_history_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._summary_history_table.setAlternatingRowColors(True)
        self._summary_history_table.setMaximumHeight(250)
        layout.addWidget(self._summary_history_table)

        # Guia per afegir nova calibració
        guide_label = QLabel(
            "<div style='background-color:#FDF2E9; border:1px solid #F0C27B; "
            "border-radius:4px; padding:8px 12px; margin-top:4px;'>"
            "<span style='font-size:11px; color:#7D6608;'>"
            "💡 <b>Per afegir una nova calibració</b>: "
            "processa una seqüència <b>_CAL</b> des del Dashboard "
            "(clic al botó Importar d'una SEQ amb &ge;3 KHP i &ge;2 concentracions). "
            "Al pas 4 (Revisar) podràs aplicar-la com a nova calibració vigent."
            "</span></div>"
        )
        guide_label.setWordWrap(True)
        layout.addWidget(guide_label)

        # Botó PDF (centrat)
        pdf_row = QHBoxLayout()
        pdf_row.addStretch()
        self._summary_pdf_btn = QPushButton("Generar Informe PDF")
        self._summary_pdf_btn.setStyleSheet("""
            QPushButton {
                background-color: #2980B9; color: white;
                border: none; border-radius: 6px;
                padding: 8px 20px; font-size: 11px; font-weight: bold;
            }
            QPushButton:hover { background-color: #3498DB; }
        """)
        self._summary_pdf_btn.clicked.connect(self._on_summary_pdf)
        pdf_row.addWidget(self._summary_pdf_btn)
        pdf_row.addStretch()
        layout.addLayout(pdf_row)

        self._main_layout.addWidget(self._summary_group)

    def _find_conc_range_for_mode(self, ref, signal_scope, mode, n_pts, source_desc):
        """Busca el rang de concentracions per un mode heretat a les calibracions inactives."""
        try:
            for other_cal in ref.get('calibrations', []):
                if other_cal.get('signal_scope') != signal_scope:
                    continue
                other_reg = other_cal.get('regression_data', {})
                other_mode = (other_reg.get('mode', '') or '').lower()
                if other_mode != mode:
                    continue
                pts = other_reg.get('points', [])
                inc = [p for p in pts if not p.get('excluded')]
                if inc:
                    ugs = [p.get('ug_doc', 0) for p in inc]
                    ug_min, ug_max = min(ugs), max(ugs)
                    rang_str = f"{ug_min:.2f} – {ug_max:.1f}"
                    rang_tip = f"{len(inc)} punts: {ug_min:.3f} – {ug_max:.3f} µg\nFont: {source_desc}"
                    return rang_str, rang_tip
        except Exception:
            pass
        rang_str = f"n={n_pts}" if n_pts and n_pts != '?' else "—"
        return rang_str, f"Font: {source_desc}"

    def show_summary(self):
        """Mostra el resum de calibracions vigents (sense SEQ_CAL carregada)."""
        from hpsec_calibrate import (
            load_calibration_reference, _extract_rf_from_cal, _extract_intercept_from_cal
        )

        ref = load_calibration_reference()
        if not ref:
            return

        calibrations = ref.get('calibrations', [])
        active_ids = ref.get('active_calibration_ids', {})

        # -- Taula paràmetres actius --
        # Recollir calibracions actives
        active_cals = []
        for key in ['direct', 'uib', 'uib_700', 'uib_1000']:
            cal_id = active_ids.get(key)
            if not cal_id:
                continue
            for cal in calibrations:
                if cal.get('id') == cal_id:
                    active_cals.append((key, cal))
                    break

        # Si no hi ha active_calibration_ids, provar active_calibration_id antic
        if not active_cals:
            active_id = ref.get('active_calibration_id')
            if active_id:
                for cal in calibrations:
                    if cal.get('id') == active_id:
                        scope = cal.get('signal_scope', 'direct')
                        active_cals.append((scope, cal))
                        break

        # Filas: per cada cal activa x cada mode (column, bp)
        rows = []
        self._summary_row_data = []
        for key, cal in active_cals:
            scope = cal.get('signal_scope', key)
            sens = cal.get('uib_sensitivity')
            ambit = scope.upper()
            sens_str = f"{int(sens)} ppb" if sens else "—"

            for mode in ['column', 'bp']:
                rf = _extract_rf_from_cal(cal, mode, scope)
                intercept = _extract_intercept_from_cal(cal, mode, scope)
                if rf is None or rf == 0:
                    continue

                # Font per mode: usar source.per_mode si disponible
                reg = cal.get('regression_data', {})
                reg_mode = (reg.get('mode', '') or '').lower()
                source_obj = cal.get('source', {})
                source_mode = (source_obj.get('mode', '') or '').lower()
                per_mode = source_obj.get('per_mode', {})

                # Determinar si aquest mode té regressió directa en aquesta entrada
                mode_is_primary = (
                    (reg_mode == mode) or (source_mode == mode)
                    or (not reg_mode and not source_mode)
                )

                eq = f"Area = {rf:.1f} × µg"
                if intercept:
                    eq += f" + {intercept:.1f}"

                if mode_is_primary:
                    # Mode amb regressió pròpia en aquesta entrada
                    r2 = cal.get('r2', 0)
                    if isinstance(r2, dict):
                        r2 = r2.get(mode, 0) or 0
                    r2 = float(r2) if r2 else 0
                    source_desc = source_obj.get('description', '')
                    source_short = source_desc[:35] + "..." if len(source_desc) > 35 else source_desc

                    reg_points = reg.get('points', [])
                    inc_pts = [p for p in reg_points if not p.get('excluded')]
                    if inc_pts:
                        ug_vals = [p.get('ug_doc', 0) for p in inc_pts]
                        ug_min, ug_max = min(ug_vals), max(ug_vals)
                        rang_str = f"{ug_min:.2f} – {ug_max:.1f}"
                        n_pts = cal.get('n_points', len(inc_pts))
                        if isinstance(n_pts, dict):
                            n_pts = n_pts.get(mode, len(inc_pts))
                        rang_tip = f"{n_pts} punts: {ug_min:.3f} – {ug_max:.3f} µg"
                    else:
                        n_pts = cal.get('n_points', '?')
                        if isinstance(n_pts, dict):
                            n_pts = n_pts.get(mode, '?')
                        # Buscar rang a calibracions inactives
                        rang_str, rang_tip = self._find_conc_range_for_mode(
                            ref, cal.get('signal_scope', 'direct'), mode, n_pts, source_desc)
                elif mode in per_mode and per_mode[mode].get('description'):
                    # Mode heretat però amb traçabilitat (source_per_mode)
                    pm = per_mode[mode]
                    r2 = float(pm.get('r2', 0) or 0)
                    source_desc = pm.get('description', '?')
                    source_short = source_desc[:35] + "..." if len(source_desc) > 35 else source_desc
                    n_pts = pm.get('n_points', '?')
                    # Buscar rang concentracions a la calibració font (inactiva)
                    rang_str, rang_tip = self._find_conc_range_for_mode(
                        ref, cal.get('signal_scope', 'direct'), mode, n_pts, source_desc)
                else:
                    # Mode heretat sense traçabilitat (cals antigues)
                    r2 = 0
                    source_desc = "Heretat (cal. anterior)"
                    source_short = source_desc
                    rang_str = "—"
                    rang_tip = f"RF {mode.upper()} preservat de la calibració anterior"

                rows.append({
                    'values': [ambit, sens_str, mode.upper(), f"{rf:.1f}",
                               f"{intercept:.1f}",
                               f"{r2:.4f}" if r2 else "—",
                               rang_str, source_short, eq],
                    'tooltips': [None, None, None, None, None, None, rang_tip,
                                 source_desc if source_desc != source_short else None,
                                 None],
                    'key': key, 'cal': cal, 'mode': mode,
                })

        self._summary_params_table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            self._summary_row_data.append((row['key'], row['cal'], row['mode']))
            for j, val in enumerate(row['values']):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignCenter)
                tip = row['tooltips'][j]
                if tip:
                    item.setToolTip(tip)
                self._summary_params_table.setItem(i, j, item)

        # -- Scatter de la primera calibració amb regression_data --
        scatter_drawn = False
        if self._summary_row_data:
            key, cal, mode = self._summary_row_data[0]
            scope = cal.get('signal_scope', key)
            sens = cal.get('uib_sensitivity')
            self._draw_summary_scatter(cal, scope, sens, mode)
            scatter_drawn = True

        self._summary_canvas.setVisible(scatter_drawn)
        # Seleccionar primera fila
        if rows:
            self._summary_params_table.selectRow(0)

        # -- Taula historial --
        # Ordenar per valid_from descendent (més recent primer)
        sorted_cals = sorted(calibrations,
                             key=lambda c: c.get('valid_from', ''), reverse=True)
        self._summary_history_table.setRowCount(len(sorted_cals))
        for i, cal in enumerate(sorted_cals):
            scope = cal.get('signal_scope', '?')
            sens = cal.get('uib_sensitivity')
            cal_id = cal.get('id', '?')
            rf_col = _extract_rf_from_cal(cal, 'column', scope)
            rf_bp = _extract_rf_from_cal(cal, 'bp', scope)
            is_active = cal.get('is_active', False)

            # "Vigent des de": valid_from + SEQ referència + mode
            vfrom = cal.get('valid_from', '?')
            src_mode = cal.get('source', {}).get('mode', '')
            seq_refs = cal.get('source', {}).get('seq_references', [])
            if seq_refs:
                first_seq = seq_refs[0] if isinstance(seq_refs[0], str) else str(seq_refs[0])
                mode_tag = f" [{src_mode}]" if src_mode else ""
                vfrom_display = f"{vfrom} ({first_seq}{mode_tag})"
            else:
                src_desc = cal.get('source', {}).get('description', '')
                if src_desc:
                    vfrom_display = f"{vfrom} ({src_desc[:25]})"
                else:
                    vfrom_display = vfrom

            # "Fins a": valid_to o "Vigent" si activa
            vto = cal.get('valid_to')
            if is_active and not vto:
                vto_display = "Vigent"
            elif vto:
                vto_display = vto
            else:
                vto_display = "—"

            values = [
                cal_id[-20:] if len(cal_id) > 20 else cal_id,
                scope.upper(),
                str(int(sens)) if sens else "—",
                vfrom_display,
                vto_display,
                f"{rf_col:.1f}" if rf_col else "—",
                f"{rf_bp:.1f}" if rf_bp else "—",
                "✔" if is_active else "",
            ]
            # Tooltip complet per ID i Font
            tooltips = [cal_id, None, None, None, None, None, None,
                        "; ".join(seq_refs) if seq_refs else None]

            for j, val in enumerate(values):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignCenter)
                if tooltips[j]:
                    item.setToolTip(tooltips[j])
                if is_active:
                    item.setForeground(QBrush(QColor("#27AE60")))
                    item.setFont(QFont("Segoe UI", 9, QFont.Bold))
                self._summary_history_table.setItem(i, j, item)

        # Mostrar summary, amagar la resta
        self._message_label.setVisible(False)
        self._summary_group.setVisible(True)
        self.seq_cal_group.setVisible(False)
        self.seq_cal_apply_group.setVisible(False)

    def _on_summary_row_clicked(self, row, col):
        """Quan l'usuari clica una fila de la taula de paràmetres, mostra el scatter corresponent."""
        if row >= len(self._summary_row_data):
            return
        key, cal, mode = self._summary_row_data[row]
        scope = cal.get('signal_scope', key)
        sens = cal.get('uib_sensitivity')
        self._draw_summary_scatter(cal, scope, sens, mode)
        self._summary_canvas.setVisible(True)

    def _draw_summary_scatter(self, cal, scope='direct', sensitivity=None,
                               mode_filter=None):
        """Dibuixa scatter de regressió al summary.

        Usa RF/intercept del mode sol·licitat (cal['rf_mass_cal'][mode]).
        Només mostra punts si regression_data.mode coincideix amb mode_filter.
        """
        self._summary_figure.clear()
        ax = self._summary_figure.add_subplot(111)

        mode_key = (mode_filter or 'column').lower()
        label = scope.upper()
        if sensitivity:
            label = f"UIB {int(sensitivity)}"

        # RF i intercept per al mode sol·licitat (des de la calibració, no des de regression_data)
        rf_dict = cal.get('rf_mass_cal', {})
        int_dict = cal.get('intercept', {})
        if isinstance(rf_dict, dict):
            rf = rf_dict.get(mode_key, 0) or 0
        else:
            rf = float(rf_dict) if rf_dict else 0
        if isinstance(int_dict, dict):
            intercept = int_dict.get(mode_key, 0) or 0
        else:
            intercept = float(int_dict) if int_dict else 0

        if rf <= 0:
            ax.text(0.5, 0.5,
                    f"RF=0 per {label} {mode_key.upper()}",
                    ha='center', va='center', fontsize=11, color='#7F8C8D',
                    transform=ax.transAxes)
            ax.set_axis_off()
            self._summary_figure.tight_layout()
            self._summary_canvas.draw()
            return

        # Punts de regressió — buscar a regression_data d'aquesta cal o d'una altra
        # del mateix àmbit que tingui punts per al mode sol·licitat
        reg = cal.get('regression_data', {})
        reg_mode = (reg.get('mode', '') or '').lower()
        mode_matches = (reg_mode == mode_key) or not reg_mode
        points = reg.get('points', []) if mode_matches else []

        # R² per al mode
        r2 = cal.get('r2', reg.get('r2', 0))
        if isinstance(r2, dict):
            r2 = r2.get(mode_key, 0) or 0
        r2 = float(r2) if r2 else 0

        # Si no hi ha punts, buscar en una altra calibració del mateix signal_scope
        if not points:
            ref = load_calibration_reference()
            cal_scope = cal.get('signal_scope', scope)
            for other_cal in ref.get('calibrations', []):
                if other_cal.get('signal_scope') != cal_scope:
                    continue
                other_reg = other_cal.get('regression_data', {})
                other_mode = (other_reg.get('mode', '') or '').lower()
                if other_mode == mode_key and other_reg.get('points'):
                    points = other_reg['points']
                    if not r2:
                        r2 = float(other_reg.get('r2', 0) or 0)
                    break

        inc = [p for p in points if not p.get('excluded')]
        exc = [p for p in points if p.get('excluded')]

        x_inc = [p['ug_doc'] for p in inc if p.get('ug_doc', 0) > 0]
        y_inc = [p['area'] for p in inc if p.get('ug_doc', 0) > 0]
        x_exc = [p['ug_doc'] for p in exc if p.get('ug_doc', 0) > 0]
        y_exc = [p['area'] for p in exc if p.get('ug_doc', 0) > 0]

        # Rang eix X
        all_x = x_inc + x_exc
        if all_x:
            x_max = max(all_x) * 1.1
        else:
            x_max = 3.0 if mode_key == 'column' else 1.0

        x_line = np.linspace(0, x_max, 100)
        y_line = rf * x_line + intercept

        # Recta
        eq_label = f'RF={rf:.1f}'
        if intercept:
            eq_label += f', int={intercept:.1f}'
        if r2 and mode_matches:
            eq_label += f' (R²={r2:.4f})'
        ax.plot(x_line, y_line, '-', color='#2980B9', linewidth=1.5, label=eq_label)

        # Banda predicció 95% (només si hi ha punts del mode correcte)
        if x_inc and len(x_inc) >= 3:
            try:
                from gui.widgets.analyze_panel._helpers import compute_prediction_band
                y_lower, y_upper = compute_prediction_band(
                    np.array(x_inc), np.array(y_inc), x_line, rf, intercept
                )
                ax.fill_between(x_line, y_lower, y_upper, alpha=0.12, color='#2980B9')
            except Exception:
                pass

        # Punts inclosos
        if x_inc:
            ax.scatter(x_inc, y_inc, c='#2980B9', s=40, zorder=5, label=f'Punts cal. ({len(x_inc)})')

        # Punts exclosos
        if x_exc:
            ax.scatter(x_exc, y_exc, c='#E74C3C', s=25, marker='x', zorder=4,
                       alpha=0.5, label=f'Exclosos ({len(x_exc)})')

        # Nota si no hi ha punts per aquest mode
        if not x_inc and not x_exc:
            source_info = cal.get('source', {})
            per_mode = source_info.get('per_mode', {})
            pm = per_mode.get(mode_key, {})
            if pm and pm.get('description'):
                # Traçabilitat disponible: mostrar font original
                pm_desc = pm['description']
                pm_r2 = pm.get('r2', 0)
                note = f"RF {mode_key.upper()} = {rf:.0f}\n{pm_desc}"
                if pm_r2:
                    note += f"\n(R\u00b2={float(pm_r2):.4f})"
            else:
                source_mode = (source_info.get('mode', '') or '').lower()
                if source_mode and source_mode != mode_key:
                    note = (f"RF {mode_key.upper()} = {rf:.0f}\n"
                            f"(de calibraci\u00f3 anterior)")
                else:
                    note = f"Sense dades de regressi\u00f3 {mode_key.upper()}"
                    if rf > 0:
                        note += f"\nRF={rf:.0f} de refer\u00e8ncia"
            ax.text(0.5, 0.5, note,
                    ha='center', va='center', fontsize=9, color='#7F8C8D',
                    transform=ax.transAxes, style='italic')

        title = f"Recta vigent — {label} {mode_key.upper()}"

        ax.set_xlabel('µg DOC injectat', fontsize=9)
        ax.set_ylabel('Àrea DOC', fontsize=9)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.legend(fontsize=7, loc='lower right')
        ax.set_xlim(left=0)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
        self._summary_figure.tight_layout()
        self._summary_canvas.draw()

    def _on_summary_pdf(self):
        """Genera informe PDF des del summary (dual si ambdós senyals disponibles)."""
        try:
            from hpsec_reports import generate_calibration_report, generate_dual_calibration_report

            cal_direct = get_active_global_calibration(signal='direct')
            cal_uib = get_active_global_calibration(signal='uib')

            if cal_direct and cal_uib:
                pdf_path = generate_dual_calibration_report()
            elif cal_direct:
                pdf_path = generate_calibration_report(cal_direct)
            elif cal_uib:
                pdf_path = generate_calibration_report(cal_uib)
            else:
                QMessageBox.warning(self, "Avís", "No hi ha calibració activa.")
                return

            if pdf_path and os.path.exists(pdf_path):
                QMessageBox.information(self, "Informe generat",
                                        f"Informe de calibració generat:\n{pdf_path}")
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

    # ------------------------------------------------------------------
    # Interface pública (cridada per GlobalCalibrationPanel)
    # ------------------------------------------------------------------

    def show_processing_message(self, seq_name):
        """Mostra missatge de processament."""
        self._message_label.setText(
            f"<div style='text-align:center; padding:40px;'>"
            f"<span style='font-size:14px; color:#2980B9;'>"
            f"Processant {seq_name}...</span></div>"
        )
        self._message_label.setVisible(True)
        self._summary_group.setVisible(False)
        self._align_group.setVisible(False)
        self.seq_cal_group.setVisible(False)
        self.seq_cal_apply_group.setVisible(False)

    def show_error_message(self, msg):
        """Mostra missatge d'error."""
        self._message_label.setText(
            f"<div style='text-align:center; padding:20px;'>"
            f"<span style='font-size:14px; color:#E74C3C;'>{msg}</span></div>"
        )
        self._message_label.setVisible(True)
        self._summary_group.setVisible(False)
        self._align_group.setVisible(False)
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
        self._is_combined = seq_cal_data.get('_combined', False)
        self._source_seqs = seq_cal_data.get('_source_seqs', [seq_name])
        self._message_label.setVisible(False)
        self._summary_group.setVisible(False)
        self.seq_cal_group.setVisible(False)
        self.seq_cal_apply_group.setVisible(False)

        # Reset estat anterior
        self._seq_cal_regression = None
        self._seq_cal_excluded = set()
        self._cal_applied_per_signal = {}
        self._cal_applied_signals = set()

        # Alineació TOC+DAD254
        self._update_alignment(imported_data)

        entries = seq_cal_data.get('entries', [])
        method = seq_cal_data.get('method', 'COLUMN')
        concs = seq_cal_data.get('concs', [])

        self._seq_cal_entries = entries
        self._seq_cal_entries_direct = seq_cal_data.get('entries_direct', [])
        self._seq_cal_entries_uib = seq_cal_data.get('entries_uib', [])
        self._seq_cal_method = method

        # Log per diagnòstic: verificar que cada llista té replicas del senyal correcte
        if self._seq_cal_entries_direct:
            reps = self._seq_cal_entries_direct[0].get('replicas', [])
            src = reps[0].get('doc_source', '?') if reps else 'N/A'
            logger.info(f"  entries_direct: {len(self._seq_cal_entries_direct)} entries, "
                        f"primer doc_source={src}")
        if self._seq_cal_entries_uib:
            reps = self._seq_cal_entries_uib[0].get('replicas', [])
            src = reps[0].get('doc_source', '?') if reps else 'N/A'
            logger.info(f"  entries_uib: {len(self._seq_cal_entries_uib)} entries, "
                        f"primer doc_source={src}")

        # Auto-excloure punts amb UIB saturada (només si senyal = uib)
        self._seq_cal_excluded = set()
        signal_name = seq_cal_data.get('signal', 'direct')
        for i, e in enumerate(entries):
            if e.get('uib_saturated') and signal_name == 'uib':
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

        # Sensibilitat UIB de la seqüència
        self._seq_cal_sensitivity = None
        if has_uib and self._seq_cal_entries_uib:
            for e in self._seq_cal_entries_uib:
                s = e.get('uib_sensitivity')
                if s:
                    self._seq_cal_sensitivity = s
                    break

        # Compact header: SEQ name + n_inj / n_conc + mode [+ sensitivity UIB]
        n_inj = sum(e.get('n_replicas', 1) for e in entries)
        n_conc = len(concs)
        sens_tag = ""
        if self._seq_cal_sensitivity:
            sens_tag = (
                f" &nbsp;&middot;&nbsp; <span style='color:#E67E22;'>"
                f"UIB {self._seq_cal_sensitivity:g} ppb</span>"
            )
        self.seq_cal_info.setText(
            f"<b style='font-size:13px;'>{seq_name}</b> &nbsp; "
            f"<span style='font-size:11px;'>{n_inj} Inj / {n_conc} conc &nbsp;&middot;&nbsp; "
            f"{method}{sens_tag}</span>"
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
            "QGroupBox { font-weight: bold; color: #1D4ED8; border: 2px solid #27AE60; "
            "border-radius: 6px; margin-top: 8px; padding-top: 14px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; }"
        )
        seq_cal_layout = QVBoxLayout(self.seq_cal_group)
        seq_cal_layout.setSpacing(10)

        # --- 1. Header compacte horitzontal (info + selectors) ---
        header_frame = QFrame()
        header_frame.setStyleSheet(
            "QFrame { background: #DBEAFE; border-radius: 4px; padding: 6px; }"
        )
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(10, 6, 10, 6)

        # Esquerra: nom SEQ + stats
        self.seq_cal_info = QLabel()
        self.seq_cal_info.setWordWrap(True)
        self.seq_cal_info.setStyleSheet(
            "color: #1D4ED8; font-weight: normal; font-size: 11px; background: transparent;"
        )
        header_layout.addWidget(self.seq_cal_info, 1)

        # Dreta: selectors (signal combo + repair checkbox)
        selectors_widget = QWidget()
        selectors_widget.setStyleSheet("background: transparent;")
        selectors_layout = QVBoxLayout(selectors_widget)
        selectors_layout.setContentsMargins(0, 0, 0, 0)
        selectors_layout.setSpacing(4)

        # Signal combo row
        signal_row = QHBoxLayout()
        signal_row.setSpacing(6)
        self.seq_cal_signal_label = QLabel("Senyal:")
        self.seq_cal_signal_label.setStyleSheet(
            "font-weight: bold; color: #1D4ED8; font-size: 11px;"
        )
        signal_row.addWidget(self.seq_cal_signal_label)

        self.seq_cal_signal_combo = QComboBox()
        self.seq_cal_signal_combo.setMaximumWidth(160)
        self.seq_cal_signal_combo.setStyleSheet(
            "QComboBox { background: white; border: 1px solid #BDC3C7; "
            "border-radius: 3px; padding: 3px 6px; font-size: 11px; }"
        )
        self.seq_cal_signal_combo.currentIndexChanged.connect(
            self._on_seq_cal_signal_changed
        )
        signal_row.addWidget(self.seq_cal_signal_combo)
        signal_row.addStretch()
        selectors_layout.addLayout(signal_row)

        header_layout.addWidget(selectors_widget)

        self.seq_cal_signal_frame = header_frame  # keep ref for visibility toggle
        seq_cal_layout.addWidget(header_frame)

        # --- 2. Warning sensibilitat UIB barrejada ---
        self.seq_cal_sensitivity_warning = QLabel()
        self.seq_cal_sensitivity_warning.setWordWrap(True)
        self.seq_cal_sensitivity_warning.setStyleSheet(
            "background: #FCF3CF; border: 1px solid #F39C12; border-radius: 4px; "
            "padding: 8px; color: #7D6608; font-size: 11px; font-weight: normal;"
        )
        self.seq_cal_sensitivity_warning.setVisible(False)
        seq_cal_layout.addWidget(self.seq_cal_sensitivity_warning)

        # --- 3. Taula de punts de calibració ---
        self.seq_cal_points_table = QTableWidget()
        self.seq_cal_points_table.setColumnCount(13)
        self.seq_cal_points_table.setHorizontalHeaderLabels([
            "Sel", "Condició", "Conc", "Vol", "µg DOC",
            "Àrea", "RF", "A254", "DOC/254",
            "R²bg", "RSD%", "Anomalies", "Selecció"
        ])
        self.seq_cal_points_table.horizontalHeaderItem(0).setToolTip(
            "Incloure/excloure punt de la regressió.\n"
            "Desmarcar = Outlier (fila gris, exclòs del càlcul)."
        )
        self.seq_cal_points_table.horizontalHeaderItem(1).setToolTip("Nom de la mostra i rèplica")
        self.seq_cal_points_table.horizontalHeaderItem(2).setToolTip("Concentració KHP (ppm)")
        self.seq_cal_points_table.horizontalHeaderItem(3).setToolTip("Volum d'injecció (µL)")
        self.seq_cal_points_table.horizontalHeaderItem(4).setToolTip("µg DOC injectat = ppm × µL / 1000")
        self.seq_cal_points_table.horizontalHeaderItem(5).setToolTip("Àrea integrada del pic DOC")
        self.seq_cal_points_table.horizontalHeaderItem(6).setToolTip("RF_mass = Àrea × 1000 / (ppm × µL)")
        self.seq_cal_points_table.horizontalHeaderItem(7).setToolTip("Àrea integrada a 254nm (DAD)")
        self.seq_cal_points_table.horizontalHeaderItem(8).setToolTip("Ratio àrea DOC / àrea 254nm")
        self.seq_cal_points_table.horizontalHeaderItem(9).setToolTip(
            "R² del fit bigaussià al pic DOC.\n"
            "Valors > 0.98 = pic ben definit.\n"
            "< 0.95 = pic irregular o multi-pic."
        )
        self.seq_cal_points_table.horizontalHeaderItem(10).setToolTip(
            "RSD (%) entre àrees de rèpliques.\n"
            "< 10% = promig; ≥ 10% = millor qualitat."
        )
        self.seq_cal_points_table.horizontalHeaderItem(11).setToolTip("Indicadors d'anomalies detectades")
        self.seq_cal_points_table.horizontalHeaderItem(12).setToolTip(
            "Control de selecció per punt:\n"
            "· Promig: mitjana de totes les rèpliques\n"
            "· R1, R2: rèplica individual\n"
            "· Millor Q: rèplica amb millor qualitat (R²bg)\n"
            "· Original: àrea sense reparació (si pic irregular)\n"
            "· Outlier: exclou de la regressió"
        )
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

        # Hint UX: bombeta amb consells d'ús
        hint_label = QLabel(
            "\U0001F4A1 <i>Clic a una fila per veure el cromatograma</i> &nbsp;·&nbsp; "
            "<i>Columna <b>Selecció</b>: canviar rèplica, excloure (Outlier) o usar àrea original</i>"
        )
        hint_label.setStyleSheet(
            "color: #7F8C8D; font-size: 10px; padding: 2px 4px; "
            "background: transparent;"
        )
        seq_cal_layout.addWidget(hint_label)

        # Chromatogram preview via popup (no inline)
        self._has_seq_cal_chrom = True

        # --- 4. Gràfic scatter + residuals (PRIMER, abans de barres) ---
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

        # --- 5. Comparació vigent vs nova (QLabel HTML) ---
        self._comparison_label = QLabel()
        self._comparison_label.setWordWrap(True)
        self._comparison_label.setTextFormat(Qt.RichText)
        self._comparison_label.setVisible(False)
        self._comparison_label.setStyleSheet(
            "QLabel { background: white; border: 1px solid #E0E0E0; "
            "border-radius: 4px; padding: 8px; margin-top: 6px; }"
        )
        seq_cal_layout.addWidget(self._comparison_label)
        self._has_comparison_mpl = True

        # Canvas ocult (mantingut per compatibilitat — no visible)
        try:
            self._comparison_figure = Figure(figsize=(1, 1), dpi=50)
            self.seq_cal_comparison_canvas = FigureCanvas(self._comparison_figure)
            self.seq_cal_comparison_canvas.setVisible(False)
            self.seq_cal_comparison_canvas.setMaximumHeight(0)
        except Exception:
            self._comparison_figure = None
            self.seq_cal_comparison_canvas = None

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

        # --- Equació resum ---
        self._cal_equation_label = QLabel("")
        self._cal_equation_label.setAlignment(Qt.AlignCenter)
        self._cal_equation_label.setStyleSheet("""
            font-family: Consolas, monospace; font-size: 12px;
            background-color: #e8f4fd; border: 1px solid #b3d7f0;
            border-radius: 4px; padding: 6px 10px;
        """)
        layout.addWidget(self._cal_equation_label)

        # --- Vigent des de (data) ---
        date_row = QHBoxLayout()
        date_row.addWidget(QLabel("Vigent des de:"))
        self._cal_valid_from = QLineEdit()
        self._cal_valid_from.setPlaceholderText("AAAA-MM-DD")
        self._cal_valid_from.setText(QDate.currentDate().toString("yyyy-MM-dd"))
        self._cal_valid_from.setMaximumWidth(120)
        self._cal_valid_from.setStyleSheet(
            "border: 1px solid #ccc; border-radius: 3px; padding: 4px 6px;"
        )
        self._cal_valid_from.editingFinished.connect(self._refresh_retro_list)
        date_row.addWidget(self._cal_valid_from)
        date_row.addStretch()
        layout.addLayout(date_row)

        # --- Retroactiu: radio buttons (per data / per nº SEQ) ---
        self._cal_retroactive_chk = QCheckBox("Aplicar retroactivament")
        self._cal_retroactive_chk.setToolTip(
            "Requantifica SEQs processades amb els nous RF/intercept\n"
            "(les àrees no canvien, només ppm)"
        )
        self._cal_retroactive_chk.toggled.connect(self._on_retroactive_toggled)
        layout.addWidget(self._cal_retroactive_chk)

        # Frame amb opcions retroactives (visible quan checkbox activat)
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
        retro_layout.setSpacing(6)

        # Radio: per data / per número seqüència
        self._retro_radio_group = QButtonGroup(self)
        radio_row = QHBoxLayout()
        self._retro_radio_date = QRadioButton("Per data (des de vigent)")
        self._retro_radio_date.setChecked(True)
        self._retro_radio_date.setStyleSheet("font-size: 11px; border: none;")
        self._retro_radio_seq = QRadioButton("Per número de seqüència")
        self._retro_radio_seq.setStyleSheet("font-size: 11px; border: none;")
        self._retro_radio_group.addButton(self._retro_radio_date, 0)
        self._retro_radio_group.addButton(self._retro_radio_seq, 1)
        self._retro_radio_group.idToggled.connect(self._on_retro_mode_changed)
        radio_row.addWidget(self._retro_radio_date)
        radio_row.addWidget(self._retro_radio_seq)
        radio_row.addStretch()
        retro_layout.addLayout(radio_row)

        # Filtre per número seqüència (visible quan radio_seq actiu)
        self._retro_seq_filter_frame = QFrame()
        self._retro_seq_filter_frame.setStyleSheet("border: none;")
        self._retro_seq_filter_frame.setVisible(False)
        filter_layout = QHBoxLayout(self._retro_seq_filter_frame)
        filter_layout.setContentsMargins(0, 0, 0, 0)
        filter_layout.setSpacing(6)
        filter_layout.addWidget(QLabel("Des de SEQ ≥"))
        self._retro_seq_from = QSpinBox()
        self._retro_seq_from.setRange(1, 999)
        self._retro_seq_from.setValue(1)
        self._retro_seq_from.setStyleSheet("border: 1px solid #ccc; border-radius: 3px; padding: 2px;")
        self._retro_seq_from.valueChanged.connect(self._refresh_retro_list)
        filter_layout.addWidget(self._retro_seq_from)
        self._retro_mode_label = QLabel("")
        self._retro_mode_label.setStyleSheet("color: #666; font-size: 10px; border: none;")
        filter_layout.addWidget(self._retro_mode_label)
        filter_layout.addStretch()
        retro_layout.addWidget(self._retro_seq_filter_frame)

        # Info + llista SEQs
        self._retro_info_label = QLabel("")
        self._retro_info_label.setWordWrap(True)
        self._retro_info_label.setStyleSheet("font-size: 11px; border: none;")
        retro_layout.addWidget(self._retro_info_label)

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
        btn_sel_all = QPushButton("Totes")
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
        self._cal_report_btn = QPushButton("Generar Informe Calibració (PDF)")
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
            self._update_seq_cal_graph(reg_result, method)
            self._update_seq_cal_comparison(reg_result, method)
        else:
            error = reg_result.get('error', 'Error desconegut') if reg_result else 'No result'
            logger.warning(f"Regressió fallida: {error}")

    def _populate_seq_cal_table(self, cal_entries):
        """Omple la taula de punts de la regressió SEQ_CAL (13 columnes).

        Cols: Sel | Condició | Conc | Vol | µg DOC | Àrea | RF |
              A254 | DOC/254 | R²bg | RSD% | Anomalies | Selecció(combo)
        """
        self.seq_cal_points_table.blockSignals(True)
        self.seq_cal_points_table.clearContents()
        self.seq_cal_points_table.setRowCount(len(cal_entries))

        # Guardar referència als combos per accedir-hi
        self._sel_combos = {}

        for i, entry in enumerate(cal_entries):
            conc = entry.get('conc_ppm', 0)
            vol = entry.get('volume_uL', 0)
            area = entry.get('area', 0)
            ug_doc = conc * vol / 1000.0
            rf_mass = entry.get('rf_mass', area / ug_doc if ug_doc > 0 else 0)

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
            ratio_item = QTableWidgetItem(f"{ratio:.1f}" if ratio else "-")
            ratio_item.setFlags(ratio_item.flags() & ~Qt.ItemIsEditable)
            if ratio and (ratio < 5 or ratio > 200):
                ratio_item.setForeground(QBrush(QColor("#E67E22")))
            self.seq_cal_points_table.setItem(i, 8, ratio_item)

            # Col 9: R²bg (bigaussian DOC)
            bg_doc = entry.get('bigaussian_doc') or {}
            bg_r2 = bg_doc.get('r2', 0) if bg_doc.get('status') not in ('ERROR', None, '') else 0
            if bg_r2 > 0:
                r2bg_item = QTableWidgetItem(f"{bg_r2:.3f}")
                if bg_r2 < 0.95:
                    r2bg_item.setForeground(QBrush(QColor("#E74C3C")))
                elif bg_r2 < 0.98:
                    r2bg_item.setForeground(QBrush(QColor("#E67E22")))
                else:
                    r2bg_item.setForeground(QBrush(QColor("#27AE60")))
                # Tooltip amb info bigaussiana addicional
                asym = bg_doc.get('asymmetry', 0)
                bg_tooltip = f"Bigaussian fit DOC\nR² = {bg_r2:.4f}"
                if asym:
                    bg_tooltip += f"\nAsimetria = {asym:.2f}"
                # 254nm bigaussian si disponible
                bg_254 = entry.get('bigaussian_254') or {}
                r2_254 = bg_254.get('r2', 0) if bg_254.get('status') not in ('ERROR', None, '') else 0
                if r2_254 > 0:
                    bg_tooltip += f"\nR² 254nm = {r2_254:.4f}"
                r2bg_item.setToolTip(bg_tooltip)
            else:
                r2bg_item = QTableWidgetItem("-")
            r2bg_item.setFlags(r2bg_item.flags() & ~Qt.ItemIsEditable)
            self.seq_cal_points_table.setItem(i, 9, r2bg_item)

            # Col 10: RSD%
            rsd = entry.get('rsd', 0)
            n_rep = entry.get('n_replicas', 1)
            if n_rep > 1 and rsd > 0:
                rsd_item = QTableWidgetItem(f"{rsd:.1f}")
                if rsd >= 10:
                    rsd_item.setForeground(QBrush(QColor("#E67E22")))
                # Tooltip amb detalls rèpliques
                comp = entry.get('replica_comparison', {})
                rsd_tooltip = f"RSD àrees: {rsd:.1f}%\nSD: {entry.get('std_area', 0):.1f}\n{n_rep} rèpliques"
                pearson = comp.get('pearson_profiles')
                if pearson is not None:
                    rsd_tooltip += f"\nPearson perfils: {pearson:.4f}"
                diff_pct = comp.get('diff_area_pct', 0)
                if diff_pct > 0:
                    rsd_tooltip += f"\nDif. àrea: {diff_pct:.1f}%"
                rsd_item.setToolTip(rsd_tooltip)
            else:
                rsd_item = QTableWidgetItem("-" if n_rep <= 1 else f"{rsd:.1f}")
            rsd_item.setFlags(rsd_item.flags() & ~Qt.ItemIsEditable)
            self.seq_cal_points_table.setItem(i, 10, rsd_item)

            # Col 11: Anomalies (icones compactes)
            cal_anoms = entry.get('calibration_anomalies', [])
            anomaly_parts = []
            if entry.get('uib_saturated') and self._seq_cal_signal == 'uib':
                anomaly_parts.append("\u26d4 SAT")
            if entry.get('irregular_top_repaired'):
                anomaly_parts.append("\u2705 rep")
            elif entry.get('has_irregular_top'):
                anomaly_parts.append("\u26a0 irr")
            if entry.get('has_timeout') and entry.get('timeout_severity', 'OK') != 'OK':
                anomaly_parts.append("TO")
            # Afegir anomalies del catàleg (blockers)
            for anom in cal_anoms:
                if isinstance(anom, dict) and anom.get('severity') == 'blocker':
                    code = anom.get('code', '')
                    if 'TIMEOUT' in code and 'TO' not in anomaly_parts:
                        anomaly_parts.append("TO")
                    elif 'BIGAUSSIAN' in code:
                        anomaly_parts.append("R²")
            anomaly_text = " ".join(anomaly_parts) if anomaly_parts else "-"
            anomaly_item = QTableWidgetItem(anomaly_text)
            anomaly_item.setFlags(anomaly_item.flags() & ~Qt.ItemIsEditable)
            if anomaly_parts:
                if any("SAT" in p for p in anomaly_parts):
                    anomaly_item.setForeground(QBrush(QColor("#E74C3C")))
                elif any("rep" in p for p in anomaly_parts):
                    anomaly_item.setForeground(QBrush(QColor("#27AE60")))
                else:
                    anomaly_item.setForeground(QBrush(QColor("#E67E22")))
            # Tooltip amb detall complet d'anomalies
            if cal_anoms:
                tooltip_lines = [a.get('label', a.get('code', '')) for a in cal_anoms if isinstance(a, dict)]
                anomaly_item.setToolTip("\n".join(tooltip_lines[:5]))
            self.seq_cal_points_table.setItem(i, 11, anomaly_item)

            # Col 12: Selecció (QComboBox) — rèplica + outlier + original
            sel_combo = QComboBox()
            sel_combo.setStyleSheet(
                "QComboBox { font-size: 10px; padding: 1px 4px; "
                "border: 1px solid #BDC3C7; border-radius: 2px; }"
            )
            sel_info = entry.get('selection', {})
            current_method = sel_info.get('method', 'average')
            n_available = sel_info.get('n_replicas_available', n_rep)

            # Opcions: [Promig], R1, R2, ..., [Millor Q], [Original], Outlier
            # Cada opció porta tooltip descriptiu
            _combo_tooltips = {
                "average": "Mitjana de totes les rèpliques disponibles",
                "best_quality": "Rèplica amb millor R² bigaussià",
                "original": "Àrea sense reparació de pic irregular",
                "outlier": "Excloure d'la regressió (no es compta)",
            }
            combo_options = []
            if n_available > 1:
                combo_options.append(("Promig", "average"))
            for r_num in range(1, n_available + 1):
                combo_options.append((f"R{r_num}", f"R{r_num}"))
            if n_available > 1:
                combo_options.append(("Millor Q", "best_quality"))

            # "Original" per entrades amb àrea reparada
            has_repair = (
                entry.get('area_original')
                and entry.get('area_repaired')
                and entry.get('area_original') != entry.get('area_repaired')
            )
            if has_repair:
                combo_options.append(("Original", "original"))

            # "Outlier" sempre disponible
            combo_options.append(("Outlier", "outlier"))

            # Determinar selecció actual
            current_idx = 0
            if i in self._seq_cal_excluded:
                current_method = 'outlier'
            for opt_idx, (label, method_key) in enumerate(combo_options):
                sel_combo.addItem(label, method_key)
                # Tooltip individual per cada opció del dropdown
                tip = _combo_tooltips.get(method_key, f"Usar rèplica {label}")
                sel_combo.setItemData(opt_idx, tip, Qt.ToolTipRole)
                if method_key == current_method:
                    current_idx = opt_idx
                elif current_method == 'single' and method_key == 'R1':
                    current_idx = opt_idx

            sel_combo.setCurrentIndex(current_idx)
            sel_combo.setToolTip("Seleccionar rèplica o excloure com a outlier")
            sel_combo.currentIndexChanged.connect(
                lambda _idx, row=i: self._on_selection_combo_changed(row)
            )
            self._sel_combos[i] = sel_combo

            combo_w = QWidget()
            combo_l = QHBoxLayout(combo_w)
            combo_l.addWidget(sel_combo)
            combo_l.setContentsMargins(2, 1, 2, 1)
            self.seq_cal_points_table.setCellWidget(i, 12, combo_w)

        self.seq_cal_points_table.blockSignals(False)

        # Ajustar alçada
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

        # Si no hi ha rèpliques pròpies i estem en UIB, buscar a _uib_match_for_replicas
        if not replicas and self._seq_cal_signal == 'uib':
            uib_match = entry.get('_uib_match_for_replicas')
            if uib_match:
                replicas = uib_match.get('replicas', [])
                logger.info(f"Chromatogram popup: usant rèpliques de _uib_match_for_replicas")

        if not replicas:
            return

        # Verificar doc_source de les rèpliques
        signal_display = self._seq_cal_signal.upper()
        for r_idx, rep in enumerate(replicas[:2]):
            src = rep.get('doc_source', '?')
            logger.info(f"Chromatogram popup row={row}: rep {r_idx} doc_source={src}, "
                        f"signal_sel={self._seq_cal_signal}, "
                        f"t_doc len={len(rep.get('t_doc', [])) if rep.get('t_doc') is not None else 'None'}")
            # Si el doc_source no coincideix amb el senyal seleccionat, avisar
            if src != '?' and src != self._seq_cal_signal:
                logger.warning(f"  MISMATCH: rep doc_source='{src}' vs signal='{self._seq_cal_signal}'")

        # Determinar si l'entry usa àrea reparada (via dropdown)
        sel_info = entry.get('selection', {})
        use_repaired = sel_info.get('method', 'average') != 'original'

        try:
            # Crear popup dialog
            conc = entry.get('conc_ppm', 0)
            name = entry.get('name_full', entry.get('condition_key', ''))
            dialog = QDialog(self)
            dialog.setWindowTitle(f"Cromatograma {signal_display} — {name} ({conc:g} ppm)")
            dialog.resize(900, 500)
            dlg_layout = QVBoxLayout(dialog)
            dlg_layout.setContentsMargins(4, 4, 4, 4)

            fig = Figure(figsize=(10, 5), dpi=100)
            fig.set_facecolor("#FAFAFA")
            canvas = FigureCanvas(fig)
            dlg_layout.addWidget(canvas)

            ax = fig.add_subplot(111)

            doc_colors = ['#2196F3', '#757575']  # blau / gris
            doc_styles = ['-', '--']
            fill_colors = ['#2196F3', '#9E9E9E']
            dad_colors = ['#9B59B6', '#78909C']  # lila / gris blavós
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
            repair_tag = " [reparat]" if entry.get('irregular_top_repaired') else ""
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

            # --- Barra d'accions: Reparar pic (mateix diàleg que a Analitzar) ---
            btn_bar = QHBoxLayout()
            repair_btn = QPushButton("🔧 Reparar pic")
            repair_btn.setStyleSheet(
                "QPushButton { border: 1px solid #E67E22; border-radius: 3px;"
                " padding: 5px 14px; color: #E67E22; font-weight: bold; }"
                "QPushButton:hover { background: #FEF9E7; }")
            repair_btn.setToolTip(
                "Reparar el cim irregular (batman) del pic d'aquest punt — "
                "mateix diàleg que al pas Analitzar. Reversible.")
            repair_btn.clicked.connect(lambda: self._repair_seq_cal_entry(entry, dialog))
            btn_bar.addWidget(repair_btn)
            btn_bar.addStretch()
            _close_btn = QPushButton("Tancar")
            _close_btn.clicked.connect(dialog.accept)
            btn_bar.addWidget(_close_btn)
            dlg_layout.addLayout(btn_bar)

            dialog.exec()
        except Exception as e:
            logger.warning(f"Error preview cromatograma: {e}", exc_info=True)

    def _repair_seq_cal_entry(self, entry, parent_dialog=None):
        """Obre el diàleg de reparació (el mateix d'Analitzar) per al punt clicat.

        Els overrides existents es carreguen als cards (neixen 'repaired' amb els
        seus ancoratges); al tancar només es persisteixen els cards MODIFICATS en
        la sessió. Navegació ◀▶ entre punts de la recta (_seq_cal_entries); la
        recalibració es fa un sol cop quan es tanca l'últim diàleg de la cadena."""
        from .analyze_panel.repair_dialog import (
            make_calibration_replica_entry, sync_repair_cards_to_overrides)
        seq_path = self._seq_path
        if not seq_path:
            QMessageBox.information(self, "Reparar pic", "No hi ha cap SEQ_CAL carregada.")
            return
        signal = (self._seq_cal_signal or 'direct').lower()
        y_key = 'y_doc_net' if signal == 'direct' else 'y_doc_uib_net'
        is_bp = (self._seq_cal_method == 'BP')

        replicas = entry.get('replicas', [])
        if not replicas and signal == 'uib':
            um = entry.get('_uib_match_for_replicas')
            if um:
                replicas = um.get('replicas', [])

        overrides = load_manual_repairs(seq_path)
        adapter = {}
        name = ''
        for d in replicas:
            t = d.get('t_doc')
            y = d.get('y_doc')
            if t is None or y is None or len(t) == 0:
                continue
            fn = d.get('filename', '') or ''
            m = re.search(r'R(\d+)', fn)
            rk = m.group(1) if m else str(d.get('replica_num', 1))
            if not name:
                name = d.get('name') or (fn.split('_R')[0] if '_R' in fn else fn)
            override = overrides.get(manual_repair_key(name, rk, signal)) if name else None
            adapter[rk] = make_calibration_replica_entry(
                d, y_key, signal, is_bp, override=override)
        if not adapter:
            QMessageBox.information(
                self, "Reparar pic",
                "No hi ha dades de cromatograma per reparar en aquest punt.")
            return

        try:
            from .analyze_panel.repair_dialog import JaggedPeakRepairDialog
        except Exception as e:
            QMessageBox.warning(self, "Reparar pic", f"No s'ha pogut obrir el diàleg: {e}")
            return

        method = 'BP' if is_bp else 'COLUMN'
        conc = entry.get('conc_ppm', 0)
        rdlg = JaggedPeakRepairDialog(
            f"{name} {signal.upper()} ({conc:g} ppm)",
            {'replicas': adapter}, method, force=True, parent=self)

        def _on_navigate(direction):
            rdlg.close()
            entries = self._seq_cal_entries or []
            idx = next((i for i, e in enumerate(entries) if e is entry), None)
            if idx is None:
                return
            new_idx = idx + direction
            if not (0 <= new_idx < len(entries)):
                return
            # Si veníem del preview del punt, tancar-lo: el nou punt és un altre
            if parent_dialog is not None:
                parent_dialog.accept()
            self._repair_seq_cal_entry(entries[new_idx])

        rdlg.navigate_requested.connect(_on_navigate)

        # La navegació obre el diàleg següent de forma niada: la recalibració
        # s'ajorna fins que la cadena de diàlegs es desfà (depth 0).
        self._repair_nav_depth = getattr(self, '_repair_nav_depth', 0) + 1
        try:
            rdlg.exec()
            changed, repaired_reps = sync_repair_cards_to_overrides(
                rdlg, seq_path, name, signal)
            if changed:
                self._repair_recalc_pending = True
                # Avís clar del que es fa (la usuària ha de saber on va a parar)
                sel_method = entry.get('selection', {}).get('method', '?')
                n_total = len(getattr(rdlg, '_cards', []))
                msg = (f"Reparació desada per <b>{name} {conc:g} ppm</b> "
                       f"({len(repaired_reps)} de {n_total} rèpliques).<br><br>")
                if sel_method in ('average', 'promig') and len(repaired_reps) < n_total:
                    msg += ("⚠️ La selecció d'aquest punt és <b>PROMIG</b> de les rèpliques: "
                            "si en repares només una, el punt es mou <b>a la meitat</b>. "
                            "Repara les dues per veure tot l'efecte.<br><br>")
                msg += ("La reparació va a l'àrea de la rèplica → al promig del punt → "
                        "a la recta. Recalculant en tancar…")
                QMessageBox.information(self, "Reparació aplicada", msg)
                if parent_dialog is not None:
                    parent_dialog.accept()
        finally:
            self._repair_nav_depth -= 1
        if self._repair_nav_depth == 0 and getattr(self, '_repair_recalc_pending', False):
            self._repair_recalc_pending = False
            # Invalidar cache i recalcular la SEQ_CAL (els overrides s'hi reapliquen)
            try:
                self.parent_panel._result_cache.pop(seq_path, None)
            except Exception:
                pass
            self.parent_panel.load_seq_cal(seq_path)

    def _on_seq_cal_point_toggled(self, idx, state):
        """Quan l'usuari marca/desmarca un punt de la regressió → sincronitza amb dropdown."""
        combo = self._sel_combos.get(idx)
        if state == 0:
            self._seq_cal_excluded.add(idx)
            # Sincronitzar dropdown → "Outlier"
            if combo:
                combo.blockSignals(True)
                outlier_idx = combo.findData('outlier')
                if outlier_idx >= 0:
                    combo.setCurrentIndex(outlier_idx)
                combo.blockSignals(False)
            # Grey out fila
            for col in range(1, 12):
                item = self.seq_cal_points_table.item(idx, col)
                if item:
                    item.setForeground(QBrush(QColor("#aaa")))
        else:
            self._seq_cal_excluded.discard(idx)
            # Sincronitzar dropdown → "Promig" (o primera opció)
            if combo:
                combo.blockSignals(True)
                avg_idx = combo.findData('average')
                combo.setCurrentIndex(avg_idx if avg_idx >= 0 else 0)
                combo.blockSignals(False)
            # Restaurar color fila
            for col in range(1, 12):
                item = self.seq_cal_points_table.item(idx, col)
                if item:
                    item.setForeground(QBrush(QColor("#000")))

        # Recalcular regressió immediatament
        if self._seq_cal_entries and self._seq_cal_method:
            self._run_seq_cal_regression(self._seq_cal_entries, self._seq_cal_method)
            self._populate_apply_section()

    def _on_selection_combo_changed(self, row):
        """L'usuari canvia la selecció de rèplica per un punt de calibració."""
        combo = self._sel_combos.get(row)
        if not combo or row >= len(self._seq_cal_entries):
            return

        method_key = combo.currentData()
        if not method_key:
            return

        entry = self._seq_cal_entries[row]
        replicas = entry.get('replicas', [])

        # --- Outlier: excloure/incloure de la regressió ---
        if method_key == 'outlier':
            self._seq_cal_excluded.add(row)
            # Actualitzar checkbox (col 0) si existeix
            cb_widget = self.seq_cal_points_table.cellWidget(row, 0)
            if cb_widget:
                cb = cb_widget.findChild(QCheckBox)
                if cb:
                    cb.blockSignals(True)
                    cb.setChecked(False)
                    cb.blockSignals(False)
            # Grey out la fila
            for col in range(1, 12):
                item = self.seq_cal_points_table.item(row, col)
                if item:
                    item.setForeground(QBrush(QColor("#aaa")))
            logger.info(f"Selection row {row}: OUTLIER")
            if self._seq_cal_entries and self._seq_cal_method:
                self._run_seq_cal_regression_no_table(self._seq_cal_entries, self._seq_cal_method)
                self._populate_apply_section()
            return

        # Si venia d'outlier, re-incloure
        was_excluded = row in self._seq_cal_excluded
        if was_excluded:
            self._seq_cal_excluded.discard(row)
            cb_widget = self.seq_cal_points_table.cellWidget(row, 0)
            if cb_widget:
                cb = cb_widget.findChild(QCheckBox)
                if cb:
                    cb.blockSignals(True)
                    cb.setChecked(True)
                    cb.blockSignals(False)
            # Restaurar color de la fila
            for col in range(1, 12):
                item = self.seq_cal_points_table.item(row, col)
                if item:
                    item.setForeground(QBrush(QColor("#000")))

        if not replicas:
            if self._seq_cal_entries and self._seq_cal_method:
                self._run_seq_cal_regression_no_table(self._seq_cal_entries, self._seq_cal_method)
                self._populate_apply_section()
            return

        # --- Original: usar àrea no reparada ---
        if method_key == 'original':
            area_orig = entry.get('area_original', entry.get('area', 0))
            new_area = area_orig
            new_a254 = entry.get('a254_area', 0)
            selected_reps = []
            logger.info(f"Selection row {row}: ORIGINAL area={new_area:.1f}")
        # --- Replica/average/best_quality ---
        elif method_key == 'average':
            areas = [r.get('area', 0) for r in replicas]
            new_area = float(np.mean(areas))
            a254_areas = [(r.get('a254_area') or 0) for r in replicas if (r.get('a254_area') or 0) > 0]
            new_a254 = float(np.mean(a254_areas)) if a254_areas else 0
            selected_reps = [r.get('replica_num', i+1) for i, r in enumerate(replicas)]
        elif method_key == 'best_quality':
            sorted_reps = sorted(replicas, key=lambda x: sum(
                10 if a.get('severity') == 'blocker' else 1
                for a in x.get('calibration_anomalies', []) if isinstance(a, dict)
            ))
            best = sorted_reps[0]
            new_area = best.get('area', 0)
            new_a254 = best.get('a254_area', 0)
            selected_reps = [best.get('replica_num', 1)]
        elif method_key.startswith('R'):
            rep_num = int(method_key[1:])
            rep = next((r for r in replicas if r.get('replica_num') == rep_num), None)
            if not rep:
                return
            new_area = rep.get('area', 0)
            new_a254 = rep.get('a254_area', 0)
            selected_reps = [rep_num]
        else:
            return

        # Actualitzar l'entry amb la nova selecció
        # CRÍTIC: sincronitzar camps específics per senyal —
        # fit_calibration_from_history llegeix area_u (UIB) o area_254 (254nm)
        entry['area'] = new_area
        if self._seq_cal_signal == 'uib':
            entry['area_u'] = new_area
        elif self._seq_cal_signal == '254':
            entry['area_254'] = new_area
        entry['a254_area'] = new_a254
        conc = entry.get('conc_ppm', 0)
        vol = entry.get('volume_uL', 0)
        if conc > 0 and vol > 0:
            from hpsec_core import area_to_rf_mass
            entry['rf_mass'] = area_to_rf_mass(new_area, conc, vol)
        if new_a254 and new_area:
            entry['a254_doc_ratio'] = new_area / new_a254
        entry['selection'] = {
            'method': method_key,
            'reason': 'manual_override',
            'selected_replicas': selected_reps,
            'n_replicas_available': len(replicas),
            'is_manual': True,
        }

        logger.info(f"Selection override row {row}: method={method_key}, area={new_area:.1f}")

        # Actualitzar cel·les de la taula sense reconstruir (evitar destruir combos)
        rf_mass = entry.get('rf_mass', 0)
        self.seq_cal_points_table.item(row, 5).setText(f"{new_area:.1f}")
        self.seq_cal_points_table.item(row, 6).setText(f"{rf_mass:.0f}")
        if self.seq_cal_points_table.item(row, 7):
            self.seq_cal_points_table.item(row, 7).setText(f"{new_a254:.0f}" if new_a254 else "-")
        ratio = entry.get('a254_doc_ratio', 0)
        if self.seq_cal_points_table.item(row, 8):
            self.seq_cal_points_table.item(row, 8).setText(f"{ratio:.2f}" if ratio else "-")

        # Recalcular regressió SENSE reconstruir taula
        if self._seq_cal_entries and self._seq_cal_method:
            self._run_seq_cal_regression_no_table(self._seq_cal_entries, self._seq_cal_method)
            self._populate_apply_section()

    def _run_seq_cal_regression_no_table(self, cal_entries, method):
        """Recalcula regressió i actualitza gràfics/comparació SENSE reconstruir la taula."""
        enabled = []
        for i, entry in enumerate(cal_entries):
            if i in self._seq_cal_excluded:
                continue
            enabled.append(entry)

        if len(enabled) < 2:
            return

        reg_result = fit_calibration_from_history(
            enabled, mode=method, signal=self._seq_cal_signal, model="intercept"
        )
        reg_result['signal'] = self._seq_cal_signal
        reg_result['uib_sensitivity'] = self._seq_cal_sensitivity

        self._seq_cal_regression = reg_result

        if reg_result and reg_result.get('success'):
            self._update_seq_cal_graph(reg_result, method)
            self._update_seq_cal_comparison(reg_result, method)

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
            logger.info(f"Signal canviat a UIB: {len(self._seq_cal_entries_uib)} entries, "
                        f"doc_source primer={self._seq_cal_entries_uib[0].get('replicas', [{}])[0].get('doc_source', '?') if self._seq_cal_entries_uib and self._seq_cal_entries_uib[0].get('replicas') else 'N/A'}")
        elif signal == "direct" and self._seq_cal_entries_direct:
            self._seq_cal_entries = self._seq_cal_entries_direct
        elif signal == "uib" and not self._seq_cal_entries_uib:
            logger.warning("Signal UIB seleccionat però _seq_cal_entries_uib buit — mantenint entries actuals")
        elif signal == "direct" and not self._seq_cal_entries_direct:
            logger.warning("Signal Direct seleccionat però _seq_cal_entries_direct buit — mantenint entries actuals")

        self._seq_cal_excluded = set()
        for i, e in enumerate(self._seq_cal_entries):
            if e.get('uib_saturated') and signal == 'uib':
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
        """Stub — repair ara es controla via dropdown per-mostra (opció 'Original')."""
        pass

    def _update_seq_cal_comparison(self, reg_result, method):
        """Mostra la comparació vigent vs nova: 5 barres uniformes blau+gris."""
        if not getattr(self, '_has_comparison_mpl', False):
            return

        new_rf = reg_result.get('rf_mass_cal', 0)
        new_intercept = reg_result.get('intercept', 0)
        new_r2 = reg_result.get('r2', 0)
        new_n = reg_result.get('n_points', 0)
        new_rms = reg_result.get('residuals_rms', 0)

        from hpsec_calibrate import get_rf_mass_cal, get_calibration_intercept

        signal = self._seq_cal_signal
        sens = self._seq_cal_sensitivity
        current_cal = get_active_global_calibration(signal=signal, sensitivity=sens)

        # Extreure dades vigent
        current_rf = 0
        current_intercept = 0
        current_r2_val = 0
        current_n = 0
        current_rms = 0
        has_vigent = False

        if current_cal:
            has_vigent = True
            current_rf = get_rf_mass_cal(signal=signal, mode=method.lower(),
                                         sensitivity=sens) or 0
            current_intercept = get_calibration_intercept(signal=signal, mode=method.lower(),
                                                          sensitivity=sens) or 0

            current_r2_raw = current_cal.get('r2', {})
            if isinstance(current_r2_raw, dict):
                current_r2_val = current_r2_raw.get(method.lower(), 0) or 0
            else:
                current_r2_val = float(current_r2_raw) if current_r2_raw else 0

            current_n = current_cal.get('n_points', 0) or 0

            rd = current_cal.get('regression_data', {})
            if rd:
                current_rms = rd.get('residuals_rms', 0) or 0

        if not has_vigent:
            self._comparison_figure.clear()
            self.seq_cal_comparison_canvas.draw()
            return

        # --- Comparació: QTableWidget en lloc de matplotlib ---
        self._comparison_figure.clear()
        self.seq_cal_comparison_canvas.setVisible(False)

        # Obtenir font de la calibració vigent
        vigent_src = ""
        if current_cal:
            src = current_cal.get('source', {})
            refs = src.get('seq_references', [])
            if refs:
                vigent_src = ", ".join(refs)

        def _delta_color(val, thres_ok, thres_warn):
            if abs(val) < thres_ok:
                return '#27AE60'
            elif abs(val) < thres_warn:
                return '#E67E22'
            return '#E74C3C'

        # Preparar dades
        d_rf = ((new_rf - current_rf) / current_rf * 100) if current_rf > 0 else 0
        d_int = new_intercept - current_intercept
        d_r2 = (new_r2 - current_r2_val) if current_r2_val else 0
        d_n = new_n - current_n
        d_rms = (new_rms - current_rms) if current_rms > 0 else 0

        rows = [
            ('RF', current_rf, new_rf, '.0f', f'{d_rf:+.1f}%', _delta_color(d_rf, 5, 15)),
            ('Intercept', current_intercept, new_intercept, '.1f', f'{d_int:+.1f}', _delta_color(d_int, 10, 30)),
            ('R\u00b2', current_r2_val, new_r2, '.4f', f'{d_r2:+.4f}',
             _delta_color(-abs(d_r2), -0.01, -0.001) if current_r2_val else '#666'),
            ('n punts', current_n, new_n, '.0f', f'{d_n:+.0f}', '#666'),
            ('RMS', current_rms, new_rms, '.1f',
             f'{d_rms:+.1f}' if current_rms > 0 else '\u2014',
             _delta_color(d_rms, 5, 15) if current_rms > 0 else '#666'),
        ]

        # Construir HTML
        vigent_label = f"Vigent ({vigent_src})" if vigent_src else "Vigent"
        html = (
            "<div style='margin-top:4px;'>"
            "<table cellspacing='0' cellpadding='6' "
            "style='border-collapse:collapse; width:100%; font-family:Segoe UI; font-size:11px;'>"
            "<tr style='background-color:#2C3E50; color:white;'>"
            "<th style='text-align:left; padding:6px 10px; border:1px solid #2C3E50;'>Paràmetre</th>"
            f"<th style='text-align:center; padding:6px 10px; border:1px solid #2C3E50;'>{vigent_label}</th>"
            "<th style='text-align:center; padding:6px 10px; border:1px solid #2C3E50;'>Nova</th>"
            "<th style='text-align:center; padding:6px 10px; border:1px solid #2C3E50;'>\u0394</th>"
            "</tr>"
        )
        for i, (label, v_vig, v_new, fmt, delta_text, delta_clr) in enumerate(rows):
            bg = '#F8F9FA' if i % 2 == 0 else '#FFFFFF'
            v_text = f'{v_vig:{fmt}}' if v_vig else '\u2014'
            html += (
                f"<tr style='background-color:{bg};'>"
                f"<td style='padding:5px 10px; font-weight:bold; color:#2C3E50; "
                f"border:1px solid #E0E0E0;'>{label}</td>"
                f"<td style='text-align:center; padding:5px 10px; color:#666; "
                f"border:1px solid #E0E0E0;'>{v_text}</td>"
                f"<td style='text-align:center; padding:5px 10px; font-weight:bold; "
                f"color:#333; border:1px solid #E0E0E0;'>{v_new:{fmt}}</td>"
                f"<td style='text-align:center; padding:5px 10px; font-weight:bold; "
                f"color:{delta_clr}; border:1px solid #E0E0E0;'>{delta_text}</td>"
                "</tr>"
            )
        html += "</table></div>"

        self._comparison_label.setText(html)
        self._comparison_label.setVisible(True)

    def _update_seq_cal_graph(self, reg_result, method):
        """Actualitza el gràfic scatter de regressió SEQ_CAL."""
        if not getattr(self, '_has_seq_cal_mpl', False):
            return
        try:
            # Usar cal_entries (totes) per tenir índexs correctes amb _seq_cal_excluded
            all_entries = self._seq_cal_entries or []
            if not all_entries:
                self._seq_cal_figure.clear()
                self.seq_cal_graph.draw()
                return

            self._seq_cal_figure.clear()
            gs = self._seq_cal_figure.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.35)
            ax_main = self._seq_cal_figure.add_subplot(gs[0])
            ax_res = self._seq_cal_figure.add_subplot(gs[1])

            excluded = self._seq_cal_excluded
            x_all, y_all = [], []
            x_inc, y_inc, sd_inc, labels_inc = [], [], [], []
            x_exc, y_exc = [], []
            for i, entry in enumerate(all_entries):
                conc = entry.get('conc_ppm', 0)
                vol = entry.get('volume_uL', 0)
                x_val = conc * vol / 1000.0
                y_val = entry.get('area', 0)
                if x_val <= 0 or y_val <= 0:
                    continue
                x_all.append(x_val)
                y_all.append(y_val)
                if i in excluded:
                    x_exc.append(x_val)
                    y_exc.append(y_val)
                else:
                    x_inc.append(x_val)
                    y_inc.append(y_val)
                    # SD només si s'han usat múltiples rèpliques (average)
                    n_selected = len(entry.get('selection', {}).get('selected_replicas', []))
                    sd = entry.get('std_area', 0) if n_selected > 1 else 0
                    sd_inc.append(sd)
                    labels_inc.append(f"{conc:g} ppm")

            # Scatter per concentració amb error bars (SD) si hi ha rèpliques
            conc_groups = {}

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
            cal_signal = getattr(self, '_seq_cal_signal', 'direct') or 'direct'
            cal_sens = getattr(self, '_seq_cal_sensitivity', None)
            current_rf = get_rf_mass_cal(signal=cal_signal, mode=method.lower(),
                                         sensitivity=cal_sens) or 0
            current_intercept = get_calibration_intercept(signal=cal_signal, mode=method.lower(),
                                                          sensitivity=cal_sens) or 0
            if current_rf > 0:
                y_current = current_rf * x_line + current_intercept
                ax_main.plot(x_line, y_current, '--', color='#E67E22', linewidth=1.5, alpha=0.7,
                             label=f'Vigent: RF={current_rf:.0f}, int={current_intercept:.1f}')

            ax_main.set_xlabel('µg DOC injectat', fontsize=9)
            ax_main.set_ylabel('Àrea DOC', fontsize=9)
            ax_main.set_title(f'Recta de Calibració {method}', fontsize=10, fontweight='bold')
            ax_main.legend(fontsize=7, loc='lower right')
            ax_main.set_xlim(left=0)
            ax_main.set_ylim(bottom=min(0, min(y_all) - 10) if y_all else 0)
            ax_main.grid(True, alpha=0.3)
            ax_main.tick_params(labelsize=8)

            # Residuals — usar cal_entries amb excluded (índexs correctes)
            residuals = []
            res_labels = []
            for i, entry in enumerate(all_entries):
                if i in excluded:
                    continue
                conc = entry.get('conc_ppm', 0)
                vol = entry.get('volume_uL', 0)
                x_val = conc * vol / 1000.0
                y_val = entry.get('area', 0)
                if x_val <= 0 or y_val <= 0:
                    continue
                y_pred = new_rf * x_val + new_intercept
                residuals.append(y_val - y_pred)
                res_labels.append(f"{conc:g}")

            if residuals:
                rms = reg_result.get('residuals_rms', 0)
                colors = ['#27AE60' if abs(r) < rms * 2 else '#E67E22' if abs(r) < rms * 3 else '#E74C3C'
                          for r in residuals]
                ax_res.bar(range(len(residuals)), residuals, color=colors, alpha=0.8, edgecolor='white')
                ax_res.axhline(y=0, color='#333', linewidth=0.8)
                if rms > 0:
                    ax_res.axhline(y=rms, color='#E67E22', linewidth=0.5, linestyle='--', alpha=0.5)
                    ax_res.axhline(y=-rms, color='#E67E22', linewidth=0.5, linestyle='--', alpha=0.5)

                if res_labels:
                    ax_res.set_xticks(range(len(res_labels)))
                    ax_res.set_xticklabels(res_labels, fontsize=6, rotation=45)
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
            logger.warning(f"_populate_apply_section: regressio no disponible "
                          f"(reg={self._seq_cal_regression is not None}, "
                          f"success={self._seq_cal_regression.get('success') if self._seq_cal_regression else 'N/A'})")
            self.seq_cal_apply_group.setVisible(False)
            return
        logger.info(f"_populate_apply_section: rf={self._seq_cal_regression.get('rf_mass_cal'):.1f} "
                   f"combined={getattr(self, '_is_combined', False)}")

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
                    self._cal_valid_from.setText(f"{d.year:04d}-{d.month:02d}-{d.day:02d}")
                except (ValueError, TypeError):
                    pass

        # Check si ja aplicada (per senyal)
        already_applied = self._cal_applied_per_signal.get(self._seq_cal_signal, False)
        if already_applied:
            self._cal_apply_btn.setEnabled(False)
            self._cal_apply_btn.setText("✓ Aplicada")
            self._cal_apply_status.setText(
                f"<span style='color:{COLOR_SUCCESS}'>&#10003; Calibració ja aplicada</span>"
            )
        else:
            self._cal_apply_btn.setEnabled(True)
            self._cal_apply_btn.setText("Aplicar com a Nova Calibració")
            self._cal_apply_status.setText("")

        # Llista SEQs retroactives
        self._populate_retro_seq_list()

        self.seq_cal_apply_group.setVisible(True)

    def _on_retroactive_toggled(self, checked):
        """Mostra/amaga la llista de SEQs retroactives."""
        self._retro_frame.setVisible(checked)
        self._retro_count_label.setVisible(checked)
        if checked:
            self._refresh_retro_list()

    def _on_retro_mode_changed(self, btn_id, checked):
        """Canvia entre mode per data i mode per número de seqüència."""
        if not checked:
            return
        by_seq = (btn_id == 1)
        self._retro_seq_filter_frame.setVisible(by_seq)
        if by_seq:
            method = (self._seq_cal_method or "COLUMN").upper()
            self._retro_mode_label.setText(f"(només {method})")
        self._refresh_retro_list()

    def _refresh_retro_list(self, _=None):
        """Refresca la llista de SEQs retroactives segons el mode seleccionat."""
        self._populate_retro_seq_list()
        self._update_retro_count()

    def _populate_retro_seq_list(self):
        """Carrega llista de SEQs processades filtrada pel mode seleccionat."""
        for cb in self._retro_seq_checkboxes:
            cb.deleteLater()
        self._retro_seq_checkboxes = []

        from hpsec_config import get_data_folders
        data_folders = get_data_folders()
        if not data_folders:
            self._retro_info_label.setText("No s'ha trobat cap carpeta de dades.")
            return

        by_seq = self._retro_radio_seq.isChecked()
        method = (self._seq_cal_method or "COLUMN").upper()
        seq_from = self._retro_seq_from.value() if by_seq else 0
        current_name = self._seq_name
        is_bp_cal = 'BP' in method

        # Data de referència per mode "per data"
        valid_from_str = self._cal_valid_from.text().strip() if hasattr(self._cal_valid_from, 'text') else ""

        seq_list = []
        all_items = []
        for df in data_folders:
            if Path(df).is_dir():
                all_items.extend(Path(df).iterdir())
        for item in sorted(all_items, key=lambda p: p.name):
            if not item.is_dir() or '_SEQ' not in item.name.upper():
                continue
            if item.name == current_name:
                continue
            if '_CAL' in item.name.upper():
                continue
            json_path = item / "CHECK" / "data" / "analysis_result.json"
            if not json_path.exists():
                continue

            # Filtrar per mode SEMPRE (BP vs COLUMN)
            is_bp_seq = '_BP' in item.name.upper()
            if is_bp_cal != is_bp_seq:
                continue

            # Extreure número de seqüència del nom (dígits inicials)
            seq_num_match = re.match(r'^(\d+)', item.name)
            seq_num = int(seq_num_match.group(1)) if seq_num_match else 0

            if by_seq:
                # Filtrar per número ≥ seq_from
                if seq_num < seq_from:
                    continue
            else:
                # Filtrar per data d'anàlisi ≥ valid_from
                if valid_from_str:
                    try:
                        with open(json_path, 'r', encoding='utf-8') as f:
                            meta = json.load(f)
                        analysis_date = (meta.get('timestamp') or meta.get('date') or '')[:10]
                        if analysis_date and analysis_date < valid_from_str:
                            continue
                    except (json.JSONDecodeError, OSError):
                        continue

            seq_list.append((item.name, str(json_path), seq_num))

        if not seq_list:
            self._retro_info_label.setText("No s'han trobat SEQs que coincideixin.")
            return

        mode_txt = f" ({method})" if by_seq else ""
        self._retro_info_label.setText(
            f"<b>{len(seq_list)} SEQs</b>{mode_txt}. "
            f"Selecciona les que vols requantificar:"
        )

        for seq_name, json_path, seq_num in seq_list:
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

        valid_from = self._cal_valid_from.text().strip() or QDate.currentDate().toString("yyyy-MM-dd")
        retroactive = self._cal_retroactive_chk.isChecked()
        cal_signal = (self._seq_cal_regression or {}).get('signal', 'direct')

        # --- Test d'equivalència amb la vigent (règims de comparabilitat) ---
        # Una SEQ_CAL que passa el test és una VALIDACIÓ: no substitueix la recta
        # ni parteix el bloc. Només un canvi confirmat per les dades obre règim.
        from hpsec_calibrate import (
            check_calibration_equivalence, resolve_regime_on_calibration,
            register_calibration_validation, get_seq_acquisition_date,
            get_pending_regime_events,
        )
        mode_key_eq = "bp" if is_bp else "column"
        uib_sens_eq = self._seq_cal_sensitivity if cal_signal == 'uib' else None
        seq_acq_date = (get_seq_acquisition_date(self._seq_path)
                        if self._seq_path else None) or valid_from
        equiv = check_calibration_equivalence(
            rf_new, signal_scope=cal_signal,
            uib_sensitivity=uib_sens_eq, mode=mode_key_eq)
        verdict = equiv.get('verdict')
        dev = equiv.get('deviation_pct') or 0.0

        if verdict == 'equivalent':
            resp = QMessageBox.question(
                self, "Equivalent a la calibració vigent",
                f"El RF nou ({rf_new:.1f}) es desvia {dev:+.1f}% de la vigent "
                f"({equiv.get('rf_reference'):.1f}), dins la tolerància del "
                f"{equiv.get('warning_pct'):.0f}%.\n\n"
                "És una VALIDACIÓ: la calibració vigent es manté i el bloc de "
                "comparabilitat continua (els candidats a canvi de règim "
                "pendents es descarten).\n\n"
                "  Sí → registrar com a VALIDACIÓ de la vigent (recomanat)\n"
                "  No → aplicar igualment com a calibració nova",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.Yes,
            )
            if resp == QMessageBox.Cancel:
                return
            resolve_regime_on_calibration(seq_acq_date, 'equivalent',
                                          seq_name=self._seq_name)
            if resp == QMessageBox.Yes:
                ok = register_calibration_validation(
                    self._seq_name, seq_acq_date, rf_new, dev,
                    signal_scope=cal_signal, uib_sensitivity=uib_sens_eq,
                    mode=mode_key_eq)
                self._cal_apply_status.setText(
                    f"Validació registrada sobre la vigent ({dev:+.1f}%)"
                    if ok else "No s'ha pogut registrar la validació")
                return
        elif verdict == 'warning':
            QMessageBox.warning(
                self, "Zona ambigua",
                f"El RF nou ({rf_new:.1f}) es desvia {dev:+.1f}% de la vigent "
                f"({equiv.get('rf_reference'):.1f}) — entre el llindar de warning "
                f"({equiv.get('warning_pct'):.0f}%) i el de trencament "
                f"({equiv.get('fail_pct'):.0f}%).\n\n"
                "Ni valida la vigent ni confirma un canvi de règim: s'aplicarà com a "
                "calibració nova dins el mateix bloc, i els candidats pendents "
                "queden com estan.",
                QMessageBox.Ok,
            )
        elif verdict == 'break':
            n_pending = len(get_pending_regime_events())
            if n_pending:
                pending_txt = (f"Hi ha {n_pending} esdeveniment(s) candidat(s) al "
                               "Manteniment: la frontera serà la data de l'esdeveniment "
                               "més antic (la causa física del canvi).")
            else:
                pending_txt = ("CAP esdeveniment registrat al Manteniment: la frontera "
                               "serà la data d'adquisició d'aquesta SEQ. Val la pena "
                               "investigar què ha canviat a l'instrument.")
            resp = QMessageBox.warning(
                self, "Canvi de règim detectat",
                f"El RF nou ({rf_new:.1f}) es desvia {dev:+.1f}% de la vigent "
                f"({equiv.get('rf_reference'):.1f}) — supera el llindar del "
                f"{equiv.get('fail_pct'):.0f}%.\n\n"
                "Les seqüències noves NO són comparables amb el bloc anterior: "
                f"s'obrirà un RÈGIM NOU.\n{pending_txt}\n\nContinuar?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
            )
            if resp != QMessageBox.Yes:
                return

        retro_count = sum(1 for cb in self._retro_seq_checkboxes if cb.isChecked()) if retroactive else 0
        msg = (
            f"S'aplicarà la nova calibració:\n\n"
            f"  Senyal: {cal_signal.upper()}\n"
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

        # Trencament confirmat: obrir el règim ABANS d'add_calibration perquè la
        # calibració nova quedi estampada amb el regime_id que li correspon.
        if verdict == 'break':
            regime_res = resolve_regime_on_calibration(
                seq_acq_date, 'break', seq_name=self._seq_name,
                detail=(f"RF {rf_new:.1f} vs vigent "
                        f"{equiv.get('rf_reference'):.1f} ({dev:+.1f}%)"))
            logger.info("Règim nou en aplicar %s: %s", self._seq_name, regime_res)

        # --- Aplicar ---
        self._cal_apply_btn.setEnabled(False)
        self._cal_apply_status.setText("Aplicant...")

        try:
            from hpsec_calibrate import (
                add_calibration,
                get_rf_mass_cal, get_calibration_intercept,
                requantify_analysis_json,
                invalidate_quantification_json,
            )

            cal_signal = (self._seq_cal_regression or {}).get('signal', 'direct')
            mode_key = "bp" if is_bp else "column"

            # v3.0: rf_mass_cal i intercept planers per àmbit (signal_scope)
            # Obtenir valors existents de l'altre mode per preservar-los
            current_cal = get_active_global_calibration(signal=cal_signal)
            if current_cal:
                rf_values = copy.deepcopy(current_cal.get('rf_mass_cal', {}))
                intercept_values = current_cal.get('intercept', {})
                if isinstance(intercept_values, dict):
                    intercept_values = copy.deepcopy(intercept_values)
                else:
                    intercept_values = {"column": 0, "bp": 0}
                # Assegurar format planer (no nested)
                if cal_signal in rf_values and isinstance(rf_values.get(cal_signal), dict):
                    rf_values = rf_values[cal_signal]
                if cal_signal in intercept_values and isinstance(intercept_values.get(cal_signal), dict):
                    intercept_values = intercept_values[cal_signal]
            else:
                rf_values = {"column": 0, "bp": 0}
                intercept_values = {"column": 0, "bp": 0}

            rf_values[mode_key] = rf_new
            intercept_values[mode_key] = intercept_new

            # Determinar uib_sensitivity per UIB
            uib_sensitivity = None
            if cal_signal == 'uib' and hasattr(self, '_seq_cal_entries'):
                for e in self._seq_cal_entries:
                    sens = e.get('uib_sensitivity')
                    if sens:
                        uib_sensitivity = sens
                        break

            # Source info — amb traçabilitat per mode
            other_mode = 'column' if mode_key == 'bp' else 'bp'
            this_mode_source = {
                "description": f"Regressió from {self._seq_name}",
                "r2": r2,
                "n_points": n_pts,
            }
            # Font de l'altre mode: heretar de la calibració anterior
            other_mode_source = {}
            if current_cal:
                prev_spm = current_cal.get('source_per_mode', {})
                if other_mode in prev_spm:
                    other_mode_source = prev_spm[other_mode]
                else:
                    # Cal anterior sense source_per_mode — reconstruir des de source
                    prev_source = current_cal.get('source', {})
                    prev_r2 = current_cal.get('r2', 0)
                    if isinstance(prev_r2, dict):
                        prev_r2 = prev_r2.get(other_mode, 0) or 0
                    prev_rd = current_cal.get('regression_data', {})
                    prev_n = len([p for p in prev_rd.get('points', [])
                                  if not p.get('excluded')]) if prev_rd else None
                    other_mode_source = {
                        "description": prev_source.get('description', '?'),
                        "r2": float(prev_r2) if prev_r2 else 0,
                    }
                    if prev_n:
                        other_mode_source["n_points"] = prev_n

            source = {
                "type": "SEQ_CAL",
                "description": f"Regressió from {self._seq_name}",
                "seq_references": [self._seq_name],
                "mode": method,
                "per_mode": {
                    mode_key: this_mode_source,
                    other_mode: other_mode_source,
                },
            }

            reg_data = dict(self._seq_cal_regression) if self._seq_cal_regression else {}
            reg_data['mode'] = method
            reg_data['signal'] = cal_signal
            reg_data['model'] = reg_data.get('model', 'intercept')

            # Propagar chromatogram_plots_dir si disponible
            if self._seq_path:
                chrom_dir = os.path.join(self._seq_path, "CHECK", "data", "khp_plots")
                if os.path.isdir(chrom_dir):
                    reg_data['chromatogram_plots_dir'] = chrom_dir

            cal_id = add_calibration(
                rf_mass_cal_values=rf_values,
                source=source,
                valid_from=valid_from,
                r2=r2,
                n_points=n_pts,
                reason=f"SEQ_CAL tab5: {self._seq_name}",
                intercept_values=intercept_values,
                regression_data=reg_data,
                signal_scope=cal_signal,
                uib_sensitivity=uib_sensitivity,
            )

            if not cal_id:
                raise RuntimeError("add_calibration ha retornat None")

            logger.info(f"Nova calibració aplicada: {cal_id} (RF={rf_new:.1f}, mode={method})")

            self._cal_applied_per_signal[cal_signal] = True
            self._cal_applied_signals.add(cal_signal)

            # --- Requantificació retroactiva ---
            retro_results = []
            if retroactive and retro_count > 0:
                self._cal_apply_status.setText(f"Requantificant {retro_count} SEQs...")

                cal_sens = self._seq_cal_sensitivity
                rf_col = get_rf_mass_cal(signal=cal_signal, mode="column", sensitivity=cal_sens)
                int_col = get_calibration_intercept(signal=cal_signal, mode="column", sensitivity=cal_sens)
                rf_bp = get_rf_mass_cal(signal=cal_signal, mode="bp", sensitivity=cal_sens)
                int_bp = get_calibration_intercept(signal=cal_signal, mode="bp", sensitivity=cal_sens)

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

            # --- Invalidar SEQs no requantificades ---
            requantified_paths = set()
            for r in retro_results:
                if r.get('success'):
                    # Recuperar path des del checkbox
                    for cb in self._retro_seq_checkboxes:
                        if cb.text() == r.get('seq') and cb.isChecked():
                            requantified_paths.add(cb.property("json_path"))

            n_invalidated = 0
            from hpsec_config import get_data_folders
            data_folders = get_data_folders()
            is_bp_cal = "BP" in method
            all_items = []
            for df in data_folders:
                if Path(df).is_dir():
                    all_items.extend(Path(df).iterdir())
            for item in all_items:
                if not item.is_dir() or '_SEQ' not in item.name.upper():
                    continue
                if '_CAL' in item.name.upper():
                    continue
                if item.name == self._seq_name:
                    continue
                # Filtrar per mode
                is_bp_seq = '_BP' in item.name.upper()
                if is_bp_cal != is_bp_seq:
                    continue
                jp = item / "CHECK" / "data" / "analysis_result.json"
                if not jp.exists():
                    continue
                if str(jp) in requantified_paths:
                    continue
                try:
                    inv = invalidate_quantification_json(str(jp))
                    if inv.get('success') and inv.get('samples_invalidated', 0) > 0:
                        n_invalidated += 1
                except Exception:
                    pass

            # --- Actualitzar UI ---
            n_ok = sum(1 for r in retro_results if r.get('success'))
            n_fail = len(retro_results) - n_ok

            status_parts = [f"<span style='color:{COLOR_SUCCESS}'>&#10003; Calibració {cal_id} aplicada</span>"]
            if retro_results:
                status_parts.append(f"<br>Requantificades: {n_ok} OK")
                if n_fail:
                    status_parts.append(f", <span style='color:{COLOR_ERROR}'>{n_fail} errors</span>")
            if n_invalidated > 0:
                status_parts.append(f"<br>Invalidades: {n_invalidated} SEQs (cal reprocessar)")

            self._cal_apply_status.setText("".join(status_parts))
            self._cal_apply_btn.setEnabled(False)
            self._cal_apply_btn.setText("✓ Aplicada")
            self._cal_report_btn.setVisible(True)

            # Refrescar dashboard
            mw = self.parent_panel.main_window if self.parent_panel else None
            if mw and hasattr(mw, 'dashboard_panel') and mw.dashboard_panel:
                try:
                    mw.dashboard_panel.refresh_sequences()
                except Exception:
                    pass

            # Marcar senyal aplicat al combo (prefix ✓)
            current_idx = self.seq_cal_signal_combo.findData(cal_signal)
            if current_idx >= 0:
                old_text = self.seq_cal_signal_combo.itemText(current_idx)
                if not old_text.startswith("✓"):
                    self.seq_cal_signal_combo.setItemText(current_idx, f"✓ {old_text}")

            # --- Auto-flow: canviar al següent senyal o tornar a resum ---
            remaining = self._get_remaining_signals()
            if remaining:
                next_signal = remaining[0]
                next_label = "UIB" if next_signal == "uib" else "DIRECT"
                QMessageBox.information(
                    self, "Calibració aplicada",
                    f"Calibració {cal_signal.upper()} aplicada correctament.\n\n"
                    f"Ara es mostra el senyal {next_label} per revisar i aplicar."
                )
                idx = self.seq_cal_signal_combo.findData(next_signal)
                if idx >= 0:
                    self.seq_cal_signal_combo.setCurrentIndex(idx)
                    # _on_seq_cal_signal_changed() s'executa automàticament
            else:
                QMessageBox.information(
                    self, "Calibració completa",
                    "Tots els senyals disponibles han estat calibrats.\n"
                    "Es mostra la vista resum."
                )
                self._cal_applied_per_signal.clear()
                self._cal_applied_signals.clear()
                self.show_summary()

        except Exception as e:
            logger.error(f"Error aplicant calibració: {e}")
            self._cal_apply_status.setText(
                f"<span style='color:{COLOR_ERROR}'>Error: {e}</span>"
            )
            self._cal_apply_btn.setEnabled(True)

    def _get_remaining_signals(self):
        """Retorna llista de senyals disponibles encara no aplicats."""
        available = []
        if self._seq_cal_entries_direct:
            available.append('direct')
        if self._seq_cal_entries_uib:
            available.append('uib')
        return [s for s in available if s not in self._cal_applied_signals]

    def _on_generate_cal_report(self):
        """Genera informe PDF (dual si ambdós senyals disponibles)."""
        try:
            from hpsec_reports import generate_calibration_report, generate_dual_calibration_report

            # Si ambdós senyals tenen calibració, generar dual
            cal_direct = get_active_global_calibration(signal='direct')
            cal_uib = get_active_global_calibration(signal='uib')

            if cal_direct and cal_uib:
                # Verificar que almenys un té regression_data
                has_reg = (cal_direct.get('regression_data') or
                           cal_uib.get('regression_data'))
                if not has_reg:
                    QMessageBox.information(
                        self, "Info",
                        "Cap calibració activa té dades de regressió emmagatzemades.\n"
                        "Les calibracions aplicades abans d'aquesta actualització no inclouen\n"
                        "les dades de regressió necessàries per l'informe complet."
                    )
                    return
                pdf_path = generate_dual_calibration_report()
            else:
                signal = getattr(self, '_seq_cal_signal', 'direct') or 'direct'
                sens = getattr(self, '_seq_cal_sensitivity', None)
                cal = get_active_global_calibration(signal=signal, sensitivity=sens)
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
