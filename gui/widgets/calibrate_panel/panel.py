"""
HPSEC Suite - QA/QC KHP Panel (v2 — redissenyat)
==================================================

Panel per a la fase 2: QA/QC KHP.
Verifica el KHP mesurat vs la calibració global (rf_mass_cal),
determina el time shift necessari i mostra mètriques i històric.

Redisseny v2:
- Compact header (1 línia) en lloc de summary_group (3 grids)
- Taula mètriques amb anomaly sub-rows i checkbox outlier
- Cromatogrames unificats (DOC+UIB+254nm en 1 gràfic)
- Recta calibració: només mode actiu + etiquetes SEQ
- Històric amb UIB
- Eliminats: replica_selection_group, validation_group visual
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QGroupBox,
    QGridLayout, QFrame, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QSplitter, QScrollArea, QSizePolicy, QComboBox,
    QCheckBox
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QColor

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from hpsec_calibrate import (
    calibrate_from_import, load_khp_history, load_local_calibrations,
    get_all_active_calibrations, get_rf_mass_cal,
    get_active_global_calibration
)
from hpsec_config import get_config

import logging
import numpy as np

logger = logging.getLogger(__name__)

# Importar components del paquet
from .worker import CalibrateWorker
from .graph_widgets import KHPReplicaGraphWidget, HistoryBarWidget, CalibrationLineWidget
# Importar estils compartits
from gui.widgets.styles import (
    PANEL_MARGINS, PANEL_SPACING, STYLE_GROUPBOX,
    COLOR_SUCCESS, COLOR_WARNING, COLOR_ERROR, COLOR_TEXT_SECONDARY,
    create_subtitle_font, apply_panel_layout
)


class CalibratePanel(QWidget):
    """Panel QA/QC KHP: verificació vs calibració global i determinació del shift."""

    calibration_completed = Signal(dict)
    delay_corrected = Signal()  # Delay corregit → wizard reimporta

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.calibration_data = None
        self.worker = None
        self._existing_calibration = None
        self._all_calibrations = []
        self._current_condition_key = None
        self._warnings_confirmed = False
        self._warnings_confirmed_by = None
        self._notes = ""

        self._setup_ui()

    def reset(self):
        """Reinicia el panel al seu estat inicial."""
        self.calibration_data = None
        self.worker = None
        self._existing_calibration = None
        self._all_calibrations = []
        self._current_condition_key = None
        self._warnings_confirmed = False
        self._warnings_confirmed_by = None
        self._notes = ""

        # Reset UI elements
        self.condition_selector_frame.setVisible(False)
        self.condition_combo.clear()
        if hasattr(self, 'placeholder'):
            self.placeholder.setVisible(True)
        if hasattr(self, 'compact_header'):
            self.compact_header.setVisible(False)
        if hasattr(self, 'khp_graph'):
            self.khp_graph.clear()
        if hasattr(self, 'replica_graphs'):
            self.replica_graphs.clear()
        if hasattr(self, 'history_graph'):
            self.history_graph.clear()
        if hasattr(self, 'history_uib_graph'):
            self.history_uib_graph.clear()
        if hasattr(self, 'history_uib254_graph'):
            self.history_uib254_graph.clear()
        if hasattr(self, 'calibration_line_graph'):
            self.calibration_line_graph.clear()
        if hasattr(self, 'cal_line_group'):
            self.cal_line_group.setVisible(False)

    def showEvent(self, event):
        """Quan el panel es mostra, comprovar si hi ha calibració existent."""
        super().showEvent(event)
        self._check_existing_calibration()

    def _check_existing_calibration(self):
        """Comprova si existeix calibració prèvia i la carrega automàticament."""
        import os

        seq_path = self.main_window.seq_path
        if not seq_path:
            self.condition_selector_frame.setVisible(False)
            return

        has_valid_data = self.calibration_data and self.calibration_data.get("success")

        try:
            all_cals = load_local_calibrations(seq_path)
            from_local = bool(all_cals)

            if not all_cals:
                all_cals = load_khp_history(seq_path)
                from_local = False

            for cal in all_cals:
                cal['_from_local'] = from_local

            if not all_cals:
                self.condition_selector_frame.setVisible(False)
                self._run_calibrate()
                return

            seq_name = os.path.basename(seq_path)
            calibrations_by_condition = {}

            for cal in all_cals:
                cal_seq = cal.get('seq_name', '')
                if cal_seq != seq_name and seq_name not in cal_seq:
                    continue
                condition_key = cal.get('condition_key', 'default')
                if condition_key not in calibrations_by_condition:
                    calibrations_by_condition[condition_key] = cal

            if not calibrations_by_condition:
                self.condition_selector_frame.setVisible(False)
                self._run_calibrate()
                return

            self._all_calibrations = list(calibrations_by_condition.values())
            self._populate_condition_combo()

            if has_valid_data:
                if self._current_condition_key:
                    for i in range(self.condition_combo.count()):
                        if self.condition_combo.itemData(i) == self._current_condition_key:
                            self.condition_combo.blockSignals(True)
                            self.condition_combo.setCurrentIndex(i)
                            self.condition_combo.blockSignals(False)
                            break
                return

            active_cal = None
            for cal in self._all_calibrations:
                if cal.get('is_active', False):
                    active_cal = cal
                    break
            if not active_cal:
                active_cal = self._all_calibrations[0]

            self._current_condition_key = active_cal.get('condition_key')
            self._load_existing_calibration(active_cal)

        except Exception as e:
            logger.warning(f"Error comprovant calibració existent: {e}")
            self.condition_selector_frame.setVisible(False)

    def _populate_condition_combo(self):
        """Omple el ComboBox amb les condicions de calibració disponibles."""
        self.condition_combo.blockSignals(True)
        self.condition_combo.clear()

        for cal in self._all_calibrations:
            condition_key = cal.get('condition_key', 'default')
            volume = cal.get('volume_uL', 0)
            conc = cal.get('conc_ppm', 0)
            mode = cal.get('mode', '')

            if volume > 0 and conc > 0:
                label = f"KHP {conc:.0f}ppm @ {volume:.0f}\u00b5L"
                if mode:
                    label = f"{mode}: {label}"
            else:
                label = condition_key

            self.condition_combo.addItem(label, condition_key)

        self.condition_combo.blockSignals(False)
        self.condition_selector_frame.setVisible(len(self._all_calibrations) > 1)

    def _on_condition_changed(self, index):
        """Handler quan l'usuari canvia la condició de calibració."""
        if index < 0 or index >= len(self._all_calibrations):
            return

        condition_key = self.condition_combo.itemData(index)
        if condition_key == self._current_condition_key:
            return

        for cal in self._all_calibrations:
            if cal.get('condition_key') == condition_key:
                self._current_condition_key = condition_key
                self._load_existing_calibration(cal)
                self.main_window.set_status(f"Mostrant calibració: {self.condition_combo.currentText()}", 3000)
                break

    def _try_load_signals_for_replicas(self, cal_enriched):
        """Carrega senyals des de imported_data per als gràfics de calibració."""
        replicas = cal_enriched.get('replicas', [])
        if not replicas:
            return

        if replicas[0].get('t_doc') is not None:
            return

        imported_data = getattr(self.main_window, 'imported_data', None)
        if imported_data and imported_data.get('success'):
            samples = imported_data.get('samples', {})
            khp_names = imported_data.get('khp_samples', [])

            for rep in replicas:
                filename = rep.get('filename', '')
                khp_name = None
                rep_num = None

                for kname in khp_names:
                    if filename.startswith(kname + '_R'):
                        khp_name = kname
                        try:
                            rep_num = filename.split('_R')[-1]
                        except Exception:
                            pass
                        break

                if not khp_name and khp_names:
                    khp_name = khp_names[0]
                    rep_num = str(rep.get('replica_num', 1))

                if not khp_name or not rep_num:
                    continue

                sample = samples.get(khp_name, {})
                sample_reps = sample.get('replicas', {})
                rep_data = sample_reps.get(str(rep_num))

                if not rep_data:
                    continue

                direct = rep_data.get('direct') or {}
                if direct.get('t') is not None and direct.get('y_net') is not None:
                    rep['t_doc'] = direct['t']
                    rep['y_doc'] = direct['y_net']

                dad_data = rep_data.get('dad', {})
                if dad_data:
                    df_dad = dad_data.get('df')
                    if df_dad is not None and hasattr(df_dad, 'columns') and not df_dad.empty:
                        if 'time (min)' in df_dad.columns:
                            col_254 = None
                            for c in df_dad.columns:
                                if '254' in str(c):
                                    col_254 = c
                                    break
                            if col_254:
                                import pandas as pd
                                t_dad = pd.to_numeric(df_dad['time (min)'], errors='coerce').to_numpy()
                                y_254 = pd.to_numeric(df_dad[col_254], errors='coerce').to_numpy()
                                mask = np.isfinite(t_dad) & np.isfinite(y_254)
                                rep['t_dad'] = t_dad[mask]
                                rep['y_dad_254'] = y_254[mask]

        # Propagar camps del top-level a rèpliques que no els tenen
        bigaussian_doc = cal_enriched.get('bigaussian_doc')
        t_retention = cal_enriched.get('t_retention', 0)
        area = cal_enriched.get('area', 0)
        peak_left = cal_enriched.get('peak_left_idx', 0)
        peak_right = cal_enriched.get('peak_right_idx', 0)

        for rep in replicas:
            if bigaussian_doc and not rep.get('bigaussian_doc'):
                rep['bigaussian_doc'] = bigaussian_doc
            if not rep.get('peak_info') and t_retention > 0:
                rep['peak_info'] = {
                    't_max': rep.get('t_max', t_retention),
                    'y_max': rep.get('area', area),
                }
            if 'peak_left_idx' not in rep and peak_left > 0:
                rep['peak_left_idx'] = peak_left
            if 'peak_right_idx' not in rep and peak_right > 0:
                rep['peak_right_idx'] = peak_right

    def _build_uib_replicas_from_import(self, cal_enriched):
        """Construeix replicas UIB des de imported_data per als gràfics."""
        imported_data = getattr(self.main_window, 'imported_data', None)
        if not imported_data or not imported_data.get('success'):
            return None

        samples = imported_data.get('samples', {})
        khp_names = imported_data.get('khp_samples', [])
        if not khp_names:
            return None

        uib_replicas = []
        for khp_name in khp_names:
            sample = samples.get(khp_name, {})
            sample_reps = sample.get('replicas', {})

            for rep_num, rep_data in sample_reps.items():
                uib = rep_data.get('uib') or {}
                if uib.get('t') is None or uib.get('y_net') is None:
                    continue

                uib_rep = {
                    'filename': f"{khp_name}_R{rep_num}",
                    'replica_num': int(rep_num),
                    't_doc': uib['t'],
                    'y_doc': uib['y_net'],
                    'area': cal_enriched.get('area_u', 0),
                    'snr': cal_enriched.get('snr_u', 0),
                    'doc_source': 'uib',
                    'bigaussian_doc': cal_enriched.get('bigaussian_uib'),
                }
                uib_replicas.append(uib_rep)

        if not uib_replicas:
            return None

        return {
            'replicas': uib_replicas,
            'doc_source': 'uib',
        }

    def _load_existing_calibration(self, cal):
        """Carrega una calibració existent de l'històric."""
        area = cal.get('area', 0)
        conc = cal.get('conc_ppm', 5)
        volume = cal.get('volume_uL', 0)
        rf = cal.get('rf', 0)
        if rf == 0 and conc > 0:
            rf = area / conc
        rf_direct = cal.get('rf_direct', rf)
        rf_uib = cal.get('rf_uib', 0)
        rf_mass = cal.get('rf_mass', 0)

        replicas_info = cal.get('replicas_info', [])
        replicas = []
        for rep_info in replicas_info:
            rep = dict(rep_info)
            rep['t_doc_max'] = rep.get('t_max', 0)
            rep['t_retention'] = rep.get('t_max', 0)
            rep_area = rep.get('area', 0)
            if rep_area > 0 and conc > 0 and volume > 0:
                rep['rf_mass_doc'] = rep_area * 1000 / (conc * volume)
            else:
                rep['rf_mass_doc'] = rf_mass
            if 'fwhm_doc' not in rep:
                rep['fwhm_doc'] = cal.get('fwhm_doc', 0)
            if 'shift_sec' not in rep:
                rep['shift_sec'] = cal.get('shift_sec', 0)
            if 'concentration_ratio' not in rep:
                rep['concentration_ratio'] = cal.get('concentration_ratio', 0)
            if 'a254_doc_ratio' not in rep:
                rep['a254_doc_ratio'] = cal.get('a254_doc_ratio', 0)
            if 'is_bp' not in rep:
                rep['is_bp'] = cal.get('is_bp', False)
            replicas.append(rep)

        if not replicas:
            replicas = [{
                'filename': cal.get('khp_name', 'KHP'),
                'area': area,
                't_max': cal.get('t_retention', 0),
                't_doc_max': cal.get('t_retention', 0),
                't_retention': cal.get('t_retention', 0),
                'snr': cal.get('snr', 0),
                'symmetry': cal.get('symmetry', 0),
                'fwhm_doc': cal.get('fwhm_doc', 0),
                'rf_mass_doc': rf_mass,
                'shift_sec': cal.get('shift_sec', 0),
                'concentration_ratio': cal.get('concentration_ratio', 0),
                'a254_doc_ratio': cal.get('a254_doc_ratio', 0),
                'is_bp': cal.get('is_bp', False),
            }]

        cal_enriched = dict(cal)
        cal_enriched['replicas'] = replicas
        cal_enriched['rf_mass_doc'] = rf_mass
        cal_enriched['n_replicas'] = cal.get('n_replicas', len(replicas))

        self._try_load_signals_for_replicas(cal_enriched)

        khp_data_uib = None
        replicas_info_uib = cal.get('replicas_info_uib', [])
        if replicas_info_uib:
            # UIB rèpliques guardades al JSON — carregar directament
            uib_replicas = []
            for rep_info in replicas_info_uib:
                rep_u = dict(rep_info)
                rep_u['doc_source'] = 'uib'
                uib_replicas.append(rep_u)
            khp_data_uib = {'replicas': uib_replicas, 'doc_source': 'uib'}
        elif rf_uib > 0 or cal.get('area_u', 0) > 0:
            # Fallback: intentar des de imported_data
            khp_data_uib = self._build_uib_replicas_from_import(cal_enriched)

        result = {
            "success": True,
            "mode": cal.get('mode', "DUAL" if cal.get('doc_mode') == 'DUAL' else "DIRECT"),
            "rf_direct": rf_direct,
            "rf_uib": rf_uib,
            "rf": rf,
            "rf_mass": rf_mass,
            "shift_direct": cal.get('shift_min', 0),
            "shift_uib": cal.get('shift_min_u', cal.get('shift_min', 0)),
            "khp_area_direct": area,
            "khp_area_uib": cal.get('area_u', 0),
            "khp_area": area,
            "khp_conc": conc,
            "khp_source": f"HIST\u00d2RIC: {cal.get('seq_name', 'N/A')}",
            "khp_data": cal_enriched,
            "khp_data_direct": cal_enriched,
            "khp_data_uib": khp_data_uib,
            "calibration": cal,
            "errors": [],
            "loaded_from_history": True,
        }

        self.calibration_data = result
        self.main_window.calibration_data = result
        self._current_condition_key = cal.get('condition_key')

        if hasattr(self, 'condition_combo') and self._current_condition_key:
            for i in range(self.condition_combo.count()):
                if self.condition_combo.itemData(i) == self._current_condition_key:
                    self.condition_combo.blockSignals(True)
                    self.condition_combo.setCurrentIndex(i)
                    self.condition_combo.blockSignals(False)
                    break

        # Dispatch: ordre nou (sense summary, replica_selection)
        for fn in [self._update_compact_header, self._update_delay_diagnostic,
                   self._update_graphs, self._update_metrics_table,
                   self._update_validation, self._update_history]:
            try:
                fn(result)
            except Exception as e:
                logger.warning(f"Error a {fn.__name__}: {e}")
                import traceback; traceback.print_exc()

        self.main_window.enable_tab(2)

        source = "local" if cal.get('_from_local') else "global"
        self.main_window.set_status(
            f"Calibració carregada ({source}): {cal.get('condition_key', 'N/A')}", 3000
        )

        self.calibration_completed.emit(result)

    def _setup_ui(self):
        """Configura la interfície — redissenyada v2."""
        layout = QVBoxLayout(self)
        apply_panel_layout(layout)

        # Botó calibrar (amagat - l'acció es dispara des del wizard header)
        self.calibrate_btn = QPushButton()
        self.calibrate_btn.setVisible(False)
        self.calibrate_btn.clicked.connect(self._run_calibrate)

        # Selector de condicions
        self.condition_selector_frame = QFrame()
        self.condition_selector_frame.setVisible(False)
        condition_layout = QHBoxLayout(self.condition_selector_frame)
        condition_layout.setContentsMargins(0, 8, 0, 8)

        condition_label = QLabel("Condició:")
        condition_label.setStyleSheet("font-weight: bold;")
        condition_layout.addWidget(condition_label)

        self.condition_combo = QComboBox()
        self.condition_combo.setMinimumWidth(200)
        self.condition_combo.setToolTip("Seleccionar condició QA/QC (volum/concentració)")
        self.condition_combo.currentIndexChanged.connect(self._on_condition_changed)
        condition_layout.addWidget(self.condition_combo)

        condition_layout.addStretch()
        layout.addWidget(self.condition_selector_frame)

        # Contenedor principal amb scroll
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setSpacing(16)

        # === PLACEHOLDER ===
        self.placeholder = QLabel("Preparant QA/QC KHP...")
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.placeholder.setStyleSheet("color: #888; font-size: 14px; padding: 40px;")
        content_layout.addWidget(self.placeholder)

        # === COMPACT HEADER (substitueix summary_group) ===
        self.compact_header = QLabel()
        self.compact_header.setVisible(False)
        self.compact_header.setWordWrap(True)
        self.compact_header.setTextFormat(Qt.RichText)
        self.compact_header.setStyleSheet(
            "QLabel { background-color: #EBF5FB; border: 1px solid #AED6F1; "
            "border-radius: 6px; padding: 10px 14px; font-size: 12px; }"
        )
        content_layout.addWidget(self.compact_header)

        # === DELAY DIAGNOSTIC ===
        self._build_delay_diagnostic_section(content_layout)

        # === CHROMATOGRAMS (pujar — era després de recta) ===
        self.graphs_group = QGroupBox("Cromatogrames KHP")
        self.graphs_group.setVisible(False)
        graphs_layout = QVBoxLayout(self.graphs_group)
        self.replica_graphs = KHPReplicaGraphWidget()
        graphs_layout.addWidget(self.replica_graphs)
        content_layout.addWidget(self.graphs_group)

        # === METRICS TABLE ===
        self.metrics_group = QGroupBox("Rèpliques KHP")
        self.metrics_group.setVisible(False)
        metrics_layout = QVBoxLayout(self.metrics_group)

        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(13)
        self.metrics_table.setHorizontalHeaderLabels([
            "Rep", "Senyal", "\u00c0rea", "Comp.", "A254",
            "RF", "t_max", "FWHM", "SNR", "Shift", "R\u00b2bg",
            "Estat", "Outlier"
        ])
        self.metrics_table.horizontalHeaderItem(2).setToolTip("\u00c0rea DOC integrada (senyal actual)")
        self.metrics_table.horizontalHeaderItem(3).setToolTip("\u00c0rea DOC companion (UIB si Direct, Direct si UIB)")
        self.metrics_table.horizontalHeaderItem(4).setToolTip("\u00c0rea 254nm (DAD)")
        self.metrics_table.horizontalHeaderItem(5).setToolTip("RF_MASS = \u00c0rea\u00d71000/(ppm\u00d7\u00b5L)")
        self.metrics_table.horizontalHeaderItem(6).setToolTip("Temps del pic m\u00e0xim (min)")
        self.metrics_table.horizontalHeaderItem(7).setToolTip("FWHM (min) - Amplada a mitja al\u00e7ada\nNormal: 0.9-1.5 min")
        self.metrics_table.horizontalHeaderItem(9).setToolTip("Shift vs 254nm (segons)")
        self.metrics_table.horizontalHeaderItem(10).setToolTip("R\u00b2 del fit bigaussi\u00e0\n\u22650.95 VALID, \u22650.80 CHECK")
        self.metrics_table.horizontalHeaderItem(11).setToolTip("Estat: \u2714 OK, \u26a0 Warning, \u2718 Blocker")
        self.metrics_table.horizontalHeaderItem(12).setToolTip("Marcar com a outlier (no s'usa per calibrar)")
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.metrics_table.setAlternatingRowColors(True)
        self.metrics_table.setMinimumHeight(150)
        self.metrics_table.setMaximumHeight(350)
        self.metrics_table.setSelectionMode(QTableWidget.ExtendedSelection)
        self.metrics_table.setSelectionBehavior(QTableWidget.SelectItems)
        metrics_layout.addWidget(self.metrics_table)

        content_layout.addWidget(self.metrics_group)

        # === CALIBRATION LINE (baixar — era a dalt) ===
        self.cal_line_group = QGroupBox("Recta de calibració")
        self.cal_line_group.setVisible(False)
        self.cal_line_group.setStyleSheet(
            "QGroupBox { font-weight: bold; color: #1A5276; border: 2px solid #2E86AB; "
            "border-radius: 6px; margin-top: 8px; padding-top: 12px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }"
        )
        cal_line_layout = QVBoxLayout(self.cal_line_group)
        self.prominent_cal_line_graph = CalibrationLineWidget()
        cal_line_layout.addWidget(self.prominent_cal_line_graph)
        self.cal_line_seqs_label = QLabel()
        self.cal_line_seqs_label.setStyleSheet(
            "color: #555; font-size: 10px; font-style: italic; padding: 2px 4px;"
        )
        self.cal_line_seqs_label.setWordWrap(True)
        cal_line_layout.addWidget(self.cal_line_seqs_label)
        content_layout.addWidget(self.cal_line_group)

        # Alias per backward compat
        self.calibration_line_graph = self.prominent_cal_line_graph

        # === HISTORY (amb UIB) ===
        self.history_group = QGroupBox("Hist\u00f2ric QA/QC")
        self.history_group.setVisible(False)
        history_layout = QVBoxLayout(self.history_group)
        history_layout.setSpacing(6)

        # Header amb filtres i botons
        history_header = QHBoxLayout()
        self.history_filters_label = QLabel()
        self.history_filters_label.setStyleSheet("color: #555; font-size: 11px;")
        history_header.addWidget(self.history_filters_label)

        self.show_outliers_cb = QCheckBox("Incloure outliers")
        self.show_outliers_cb.setToolTip("Mostrar tamb\u00e9 les calibracions marcades com a outliers")
        self.show_outliers_cb.stateChanged.connect(self._on_show_outliers_changed)
        history_header.addWidget(self.show_outliers_cb)

        history_header.addStretch()

        self.history_info_btn = QPushButton("?")
        self.history_info_btn.setFixedSize(20, 20)
        self.history_info_btn.setCursor(Qt.WhatsThisCursor)
        self.history_info_btn.setStyleSheet("""
            QPushButton {
                background: #ECF0F1; border: 1px solid #BDC3C7;
                border-radius: 10px; font-size: 11px; font-weight: bold;
                color: #7F8C8D;
            }
            QPushButton:hover { background: #3498DB; color: white; border-color: #2E86AB; }
        """)
        self.history_info_btn.setToolTip("Clic per veure llegenda i detalls")
        self.history_info_btn.clicked.connect(self._show_history_legend)
        history_header.addWidget(self.history_info_btn)
        history_layout.addLayout(history_header)

        # Gràfics: Direct + UIB + DOC/254
        history_content = QHBoxLayout()

        self.history_graph = HistoryBarWidget(ylabel="\u00c0rea Direct", value_key="area")
        self.history_graph.bar_selected.connect(self._on_history_bar_selected)
        history_content.addWidget(self.history_graph)

        self.history_uib_graph = HistoryBarWidget(ylabel="\u00c0rea UIB", value_key="area_u")
        self.history_uib_graph.bar_selected.connect(self._on_history_bar_selected)
        history_content.addWidget(self.history_uib_graph)

        self.history_doc254_graph = HistoryBarWidget(ylabel="DOC/254", value_key="a254_doc_ratio")
        self.history_doc254_graph.bar_selected.connect(self._on_history_bar_selected)
        history_content.addWidget(self.history_doc254_graph)

        self.history_uib254_graph = HistoryBarWidget(ylabel="UIB/254", value_key="d254_u")
        self.history_uib254_graph.bar_selected.connect(self._on_history_bar_selected)
        history_content.addWidget(self.history_uib254_graph)

        history_layout.addLayout(history_content)

        # Resum i botons
        history_footer = QHBoxLayout()
        self.history_summary = QLabel()
        self.history_summary.setStyleSheet("color: #666; font-size: 11px;")
        history_footer.addWidget(self.history_summary)
        history_footer.addStretch()

        self.toggle_outlier_btn = QPushButton("Marcar Outlier")
        self.toggle_outlier_btn.setEnabled(False)
        self.toggle_outlier_btn.setToolTip("Clicar una barra per seleccionar, despr\u00e9s marcar/desmarcar outlier")
        self.toggle_outlier_btn.clicked.connect(self._toggle_outlier)
        self.toggle_outlier_btn.setStyleSheet("QPushButton { padding: 4px 8px; }")
        history_footer.addWidget(self.toggle_outlier_btn)

        history_layout.addLayout(history_footer)

        content_layout.addWidget(self.history_group)

        # Spacer
        content_layout.addStretch()

        scroll.setWidget(content_widget)
        layout.addWidget(scroll, 1)

        # Referència dummy per compatibilitat amb wizard
        self.next_btn = QPushButton()
        self.next_btn.setVisible(False)

    # =========================================================================
    # RUN CALIBRATE
    # =========================================================================

    def _run_calibrate(self):
        """Executa la calibració."""
        imported_data = self.main_window.imported_data

        if not imported_data:
            seq_path = self.main_window.seq_path
            if seq_path:
                from hpsec_import import import_from_manifest
                self.main_window.set_status("Carregant dades d'importació...")
                imported_data = import_from_manifest(seq_path)
                if imported_data and imported_data.get('success'):
                    self.main_window.imported_data = imported_data
                    self.main_window.set_status("Dades carregades", 1000)

        if not imported_data:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self, "No hi ha dades",
                "No s'han trobat dades d'importació.\n\n"
                "Cal importar la seqüència primer."
            )
            return

        self.calibrate_btn.setEnabled(False)
        self.main_window.show_progress(0)

        # Netejar resultats anteriors
        self.compact_header.setVisible(False)
        self.delay_group.setVisible(False)
        self.cal_line_group.setVisible(False)
        self.graphs_group.setVisible(False)
        self.metrics_group.setVisible(False)
        self.history_group.setVisible(False)

        if self.worker is not None:
            self.worker.wait()
        self.worker = CalibrateWorker(imported_data)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_progress(self, pct, msg):
        self.main_window.show_progress(pct)
        self.main_window.set_status(msg)

    def _on_finished(self, result):
        self.main_window.show_progress(-1)
        self.calibrate_btn.setEnabled(True)
        if self.worker is not None:
            self.worker.wait()

        # Copiar rf_mass_direct i rf_mass_uib a nivell superior
        for signal_key, data_key in [("direct", "khp_data_direct"), ("uib", "khp_data_uib")]:
            khp_data = result.get(data_key)
            if khp_data:
                replicas = khp_data.get("replicas", [khp_data])
                rf_vals = [r.get("rf_mass_doc", 0) for r in replicas if r.get("rf_mass_doc", 0) > 0]
                if not rf_vals:
                    rf_vals = [r.get("rf_mass", 0) for r in replicas if r.get("rf_mass", 0) > 0]
                if rf_vals:
                    result[f"rf_mass_{signal_key}"] = float(np.mean(rf_vals))

        self.calibration_data = result
        self.main_window.calibration_data = result

        # Dispatch: ordre nou
        for fn in [self._update_compact_header, self._update_delay_diagnostic,
                   self._update_graphs, self._update_metrics_table,
                   self._update_validation, self._update_history]:
            try:
                fn(result)
            except Exception as e:
                logger.warning(f"Error a {fn.__name__}: {e}")
                import traceback; traceback.print_exc()

        # Auto-generar PDF de QA/QC
        try:
            from hpsec_reports import generate_calibration_report
            pdf = generate_calibration_report()
            if pdf:
                logger.info(f"Report QA/QC: {pdf}")
        except Exception as e:
            logger.warning(f"No s'ha pogut generar report de QA/QC: {e}")

        self._reload_condition_selector()

        self.main_window.enable_tab(2)
        self.main_window.set_status("QA/QC KHP completat", 5000)

        self.calibration_completed.emit(result)

    def _reload_condition_selector(self):
        """Recarrega el selector de condicions després d'una nova calibració."""
        import os
        seq_path = self.main_window.seq_path
        if not seq_path:
            return

        try:
            all_cals = load_local_calibrations(seq_path)
            if not all_cals:
                self.condition_selector_frame.setVisible(False)
                return

            seq_name = os.path.basename(seq_path)
            calibrations_by_condition = {}

            for cal in all_cals:
                if cal.get('seq_name') != seq_name:
                    continue
                condition_key = cal.get('condition_key', 'default')
                if condition_key not in calibrations_by_condition:
                    calibrations_by_condition[condition_key] = cal

            if not calibrations_by_condition:
                self.condition_selector_frame.setVisible(False)
                return

            self._all_calibrations = list(calibrations_by_condition.values())
            self._populate_condition_combo()

            if self._current_condition_key:
                for i in range(self.condition_combo.count()):
                    if self.condition_combo.itemData(i) == self._current_condition_key:
                        self.condition_combo.blockSignals(True)
                        self.condition_combo.setCurrentIndex(i)
                        self.condition_combo.blockSignals(False)
                        break

        except Exception as e:
            logger.warning(f"Error recarregant selector de condicions: {e}")

    def _on_error(self, error_msg):
        self.main_window.show_progress(-1)
        self.calibrate_btn.setEnabled(True)

        is_no_khp = any(kw in error_msg.lower() for kw in [
            "no s'ha trobat khp", "no khp", "sense khp", "khp no v\u00e0lid",
            "no valid khp", "invalid khp", "all khp invalid"
        ])

        shift_direct = 0.0
        shift_uib = 0.0
        khp_source = "SENSE_KHP"

        if is_no_khp:
            shift_direct, shift_uib, khp_source = self._ask_shift_decision()

        self.calibration_data = {
            "success": False,
            "factor_direct": 0,
            "factor_uib": 0,
            "shift_direct": shift_direct,
            "shift_uib": shift_uib,
            "khp_source": khp_source,
            "errors": [error_msg],
            "warnings_structured": [{
                "code": "NO_VALID_KHP",
                "level": "warning",
                "message": f"Sense KHP v\u00e0lid. Shift: {khp_source}",
            }],
        }
        self.main_window.calibration_data = self.calibration_data

        self.placeholder.setVisible(False)
        self.compact_header.setVisible(True)
        self.compact_header.setText(
            '<span style="color: #922B21; font-weight: bold;">'
            'Sense KHP v\u00e0lid \u2014 Mode: Defaults'
            '</span>'
        )

        try:
            self._update_delay_diagnostic(self.calibration_data)
        except Exception as e:
            logger.warning(f"Error a _update_delay_diagnostic (error path): {e}")

        self.main_window.enable_tab(2)
        self.calibration_completed.emit(self.calibration_data)

    def _ask_shift_decision(self):
        """Diàleg per decidir el shift quan no hi ha KHP vàlid."""
        from PySide6.QtWidgets import QDialog, QDialogButtonBox, QRadioButton, QDoubleSpinBox

        dialog = QDialog(self)
        dialog.setWindowTitle("Sense KHP v\u00e0lid \u2014 Decidir Time Shift")
        dialog.setMinimumWidth(450)

        layout = QVBoxLayout(dialog)
        layout.setSpacing(12)

        warning_frame = QFrame()
        warning_frame.setStyleSheet(
            "background-color: #FFF3CD; border: 1px solid #FFEEBA; "
            "border-radius: 6px; padding: 10px;"
        )
        warning_layout = QVBoxLayout(warning_frame)
        warning_layout.addWidget(QLabel(
            "<b>No s'ha trobat KHP v\u00e0lid.</b><br>"
            "Cal decidir quin time shift aplicar per a la quantificaci\u00f3."
        ))
        layout.addWidget(warning_frame)

        radio_zero = QRadioButton("Usar shift = 0 (sense correcci\u00f3 temporal)")
        radio_zero.setChecked(True)
        layout.addWidget(radio_zero)

        historic_shift_d = 0.0
        historic_shift_u = 0.0
        has_historic = False
        try:
            seq_path = self.main_window.seq_path
            if seq_path:
                history = load_khp_history(seq_path)
                if history:
                    for cal in reversed(history):
                        shift_sec = cal.get('shift_sec', 0)
                        if shift_sec != 0:
                            historic_shift_d = shift_sec / 60.0
                            historic_shift_u = cal.get('shift_uib_sec', shift_sec) / 60.0
                            has_historic = True
                            break
        except Exception:
            pass

        radio_historic = QRadioButton(
            f"Usar shift d'hist\u00f2ric: {historic_shift_d * 60:.1f}s"
            if has_historic else "Usar shift d'hist\u00f2ric (no disponible)"
        )
        radio_historic.setEnabled(has_historic)
        layout.addWidget(radio_historic)

        radio_manual = QRadioButton("Introduir shift manualment (segons):")
        layout.addWidget(radio_manual)

        manual_layout = QHBoxLayout()
        manual_layout.addSpacing(24)
        manual_layout.addWidget(QLabel("Direct:"))
        spin_direct = QDoubleSpinBox()
        spin_direct.setRange(-120, 120)
        spin_direct.setSuffix(" s")
        spin_direct.setDecimals(1)
        manual_layout.addWidget(spin_direct)
        manual_layout.addWidget(QLabel("UIB:"))
        spin_uib = QDoubleSpinBox()
        spin_uib.setRange(-120, 120)
        spin_uib.setSuffix(" s")
        spin_uib.setDecimals(1)
        manual_layout.addWidget(spin_uib)
        manual_layout.addStretch()
        layout.addLayout(manual_layout)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec():
            if radio_historic.isChecked() and has_historic:
                return historic_shift_d, historic_shift_u, "HISTORY"
            elif radio_manual.isChecked():
                return spin_direct.value() / 60.0, spin_uib.value() / 60.0, "MANUAL"
            else:
                return 0.0, 0.0, "ZERO (sense KHP)"
        else:
            return 0.0, 0.0, "ZERO (cancel\u00b7lat)"

    # =========================================================================
    # COMPACT HEADER (substitueix _update_summary)
    # =========================================================================

    def _update_compact_header(self, result):
        """Actualitza el header compacte d'1 línia amb tota la info rellevant."""
        import os

        self.placeholder.setVisible(False)
        self.compact_header.setVisible(True)

        seq_path = self.main_window.seq_path or ""
        seq_name = os.path.basename(seq_path) if seq_path else "-"

        mode = result.get("mode", "-") or "-"

        # Concentració
        khp_conc = result.get("khp_conc", 0)
        conc_text = f"KHP {khp_conc:g}ppm" if khp_conc > 0 else "KHP ?"

        # Volum
        khp_data_main = result.get("khp_data_direct") or result.get("khp_data_uib")
        volume = None
        if khp_data_main:
            volume = khp_data_main.get('volume_uL')
            if not volume:
                replicas = khp_data_main.get('replicas') or []
                if replicas:
                    volume = replicas[0].get('volume_uL')
        vol_text = f"{int(volume)}\u00b5L" if volume else "?\u00b5L"

        # Rèpliques
        n_replicas = 0
        n_valid = 0
        if khp_data_main:
            replicas = khp_data_main.get('replicas') or []
            n_replicas = len(replicas) if replicas else khp_data_main.get("n_replicas", 0)
            n_valid = sum(1 for r in replicas if not r.get('is_outlier', False))
        rep_text = f"{n_replicas} rep ({n_valid} v\u00e0l)" if n_replicas > 0 else "0 rep"

        # RF_MASS + desviació vs global
        rf_html = ""
        try:
            rf_mass_measured = 0
            if khp_data_main:
                replicas_qc = khp_data_main.get('replicas') or [khp_data_main]
                rf_vals = [r.get('rf_mass_doc', 0) or r.get('rf_mass', 0) for r in replicas_qc]
                rf_vals = [v for v in rf_vals if v > 0]
                if rf_vals:
                    rf_mass_measured = np.mean(rf_vals)
                elif khp_data_main.get('rf_mass', 0) > 0:
                    rf_mass_measured = khp_data_main['rf_mass']
            if rf_mass_measured <= 0:
                rf_mass_measured = result.get('rf_mass', 0) or result.get('rf_mass_doc', 0) or 0

            if rf_mass_measured > 0:
                mode_str = mode.lower() if mode else 'column'
                # Determinar senyal i sensibilitat del KHP principal
                main_signal = 'direct' if result.get("khp_data_direct") else 'uib'
                main_sensitivity = None
                if main_signal == 'uib':
                    imported = self.main_window.imported_data or {}
                    main_sensitivity = imported.get("uib_sensitivity")
                rf_mass_cal = get_rf_mass_cal(signal=main_signal, mode=mode_str,
                                              sensitivity=main_sensitivity)
                if rf_mass_cal and rf_mass_cal > 0:
                    deviation_pct = (rf_mass_measured - rf_mass_cal) / rf_mass_cal * 100
                    if abs(deviation_pct) < 5:
                        dev_color = "#27AE60"
                    elif abs(deviation_pct) < 10:
                        dev_color = "#F39C12"
                    else:
                        dev_color = "#E74C3C"
                    rf_html = (
                        f' \u00b7 RF={rf_mass_measured:.0f} '
                        f'<span style="color: {dev_color}; font-weight: bold;">'
                        f'(ref {rf_mass_cal:.0f} \u2192 {deviation_pct:+.1f}%)</span>'
                    )
                else:
                    rf_html = f' \u00b7 RF={rf_mass_measured:.0f}'
        except Exception:
            pass

        # Shift
        shift_parts = []
        shift_d = result.get("shift_direct", 0)
        shift_u = result.get("shift_uib", 0)
        if shift_d != 0 or shift_u != 0:
            shift_d_sec = shift_d * 60
            shift_u_sec = shift_u * 60
            mode_upper = str(mode).upper()
            if "DIRECT" in mode_upper or "DUAL" in mode_upper or "COLUMN" in mode_upper:
                shift_parts.append(f"D:{shift_d_sec:+.1f}s")
            if "UIB" in mode_upper or "DUAL" in mode_upper or "BP" in mode_upper:
                shift_parts.append(f"U:{shift_u_sec:+.1f}s")
            if not shift_parts:
                shift_parts.append(f"D:{shift_d_sec:+.1f}s")

        shift_html = ""
        if shift_parts:
            shift_html = f' \u00b7 Shift: {" ".join(shift_parts)}'

        # Badge [CAL]
        cal_badge = ""
        if "_CAL" in seq_name.upper():
            cal_badge = '<span style="color:#1A56DB;font-weight:bold;">[CAL]</span> '

        # UIB sensitivity
        uib_sens_html = ""
        khp_data_uib_info = result.get("khp_data_uib")
        if khp_data_uib_info:
            uib_sens = khp_data_uib_info.get('uib_sensitivity')
            if not uib_sens:
                uib_reps = khp_data_uib_info.get('replicas') or khp_data_uib_info.get('all_khp_data') or []
                if uib_reps and isinstance(uib_reps, list):
                    uib_sens = uib_reps[0].get('uib_sensitivity')
            if uib_sens:
                uib_sens_html = f" \u00b7 UIB {int(uib_sens)}ppb"

        # KHP source
        khp_source = result.get("khp_source", "")
        source_html = ""
        if khp_source:
            if "LOCAL" in str(khp_source).upper() or "HIST" not in str(khp_source).upper():
                source_html = " \u00b7 local"
            elif "SIBLING" in str(khp_source).upper():
                source_html = " \u00b7 sibling"
            elif "HIST" in str(khp_source).upper():
                source_html = " \u00b7 hist\u00f2ric"

        # Assemble
        html = (
            f'{cal_badge}<b>{seq_name}</b> \u00b7 {mode} \u00b7 {conc_text} \u00b7 {vol_text} \u00b7 '
            f'{rep_text}{rf_html}{shift_html}{uib_sens_html}{source_html}'
        )

        self.compact_header.setText(html)

    # =========================================================================
    # GRAPHS
    # =========================================================================

    def _extract_all_replicas(self, khp_data):
        """Extrae todas las réplicas de los datos KHP."""
        if not khp_data:
            return []

        if isinstance(khp_data, list):
            return khp_data

        if isinstance(khp_data, dict):
            replicas = khp_data.get('all_khp_data') or khp_data.get('replicas')
            if replicas and isinstance(replicas, list):
                return replicas
            return [khp_data]

        return []

    def _update_graphs(self, result):
        """Actualiza los gráficos de KHP per rèplica (unificats DOC+UIB+254nm)."""
        khp_data_direct = result.get("khp_data_direct")
        khp_data_uib = result.get("khp_data_uib")

        direct_list = self._extract_all_replicas(khp_data_direct)
        uib_list = self._extract_all_replicas(khp_data_uib)

        has_graphs = len(direct_list) > 0

        if has_graphs:
            self.graphs_group.setVisible(True)
            self.replica_graphs.plot_replicas(direct_list, uib_list if uib_list else None)
        else:
            self.graphs_group.setVisible(False)

    # =========================================================================
    # METRICS TABLE (amb anomaly sub-rows i checkbox outlier)
    # =========================================================================

    def _count_peaks_in_zone(self, khp, zone_min=4.0):
        """Compta pics dins de \u00b1zone_min del pic principal."""
        peak_info = khp.get('peak_info', {})
        t_max = peak_info.get('t_max', 0) or khp.get('t_doc_max', 0) or khp.get('t_retention', 0)
        all_peaks = khp.get('all_peaks', [])

        if t_max <= 0 or not all_peaks:
            return 1

        count = 0
        for peak in all_peaks:
            t_peak = peak.get('t', 0)
            if abs(t_peak - t_max) <= zone_min:
                count += 1

        return max(count, 1)

    def _timeout_affects_peak(self, khp):
        """Determina si el timeout afecta el pic principal."""
        if not khp.get('has_timeout', False):
            return False

        timeout_info = khp.get('timeout_info', {})
        timeouts_list = timeout_info.get('timeouts', [])

        if not timeouts_list:
            return False

        peak_info = khp.get('peak_info', {})
        t_max = peak_info.get('t_max', 0) or khp.get('t_doc_max', 0) or khp.get('t_retention', 0)

        if t_max <= 0:
            return False

        for to in timeouts_list:
            affected_start = to.get('affected_start_min', to.get('t_start_min', 0) - 0.5)
            affected_end = to.get('affected_end_min', to.get('t_end_min', 0) + 1.0)
            if affected_start <= t_max <= affected_end:
                return True

        return False

    def _update_metrics_table(self, result):
        """Actualitza la taula de mètriques amb anomaly sub-rows i checkbox outlier."""
        self.metrics_table.setRowCount(0)

        khp_data_direct = result.get("khp_data_direct")
        khp_data_uib = result.get("khp_data_uib")

        all_data = []

        # Recopilar timeouts de Direct per propagar a UIB
        direct_timeouts = {}

        import re as _re

        # Direct replicas
        direct_list = self._extract_all_replicas(khp_data_direct)
        for d in direct_list:
            d_copy = d.copy()
            d_copy['_signal'] = 'Direct'
            all_data.append(d_copy)
            if d.get('has_timeout'):
                fname = d.get('filename', '')
                match = _re.search(r'R(\d+)', fname)
                rep_num = match.group(1) if match else '1'
                direct_timeouts[rep_num] = d.get('timeout_info', {})

        # UIB replicas (propagant timeouts)
        uib_list = self._extract_all_replicas(khp_data_uib)
        for d in uib_list:
            d_copy = d.copy()
            d_copy['_signal'] = 'UIB'
            fname = d.get('filename', '')
            match = _re.search(r'R(\d+)', fname)
            rep_num = match.group(1) if match else '1'
            if not d_copy.get('has_timeout') and rep_num in direct_timeouts:
                d_copy['has_timeout'] = True
                d_copy['timeout_info'] = direct_timeouts[rep_num]
                d_copy['_timeout_propagated'] = True
            all_data.append(d_copy)

        # Lookup àrees companion: Direct→UIB, UIB→Direct
        uib_area_by_rep = {}
        direct_area_by_rep = {}
        for d in uib_list:
            fname = d.get('filename', '')
            match = _re.search(r'R(\d+)', fname)
            rn = match.group(1) if match else '1'
            uib_area_by_rep[rn] = d.get('area', 0)
        for d in direct_list:
            fname = d.get('filename', '')
            match = _re.search(r'R(\d+)', fname)
            rn = match.group(1) if match else '1'
            direct_area_by_rep[rn] = d.get('area', 0)

        if not all_data:
            self.metrics_group.setVisible(False)
            return

        self.metrics_group.setVisible(True)

        # Thresholds
        FWHM_THRESHOLD = 1.5

        from hpsec_warnings import ANOMALY_CATALOG, classify_anomalies, IGNORED_KHP_CODES

        for khp in all_data:
            row = self.metrics_table.rowCount()
            self.metrics_table.insertRow(row)

            filename = khp.get('filename', '?')
            signal = khp.get('_signal', '?')

            # Col 0: Rep
            display_name = filename
            if '_R' in filename:
                display_name = 'R' + filename.split('_R')[-1].split('.')[0].split('_')[0]
            self.metrics_table.setItem(row, 0, QTableWidgetItem(display_name))

            # Col 1: Senyal
            self.metrics_table.setItem(row, 1, QTableWidgetItem(signal))

            # Col 2: Àrea
            area = khp.get('area', 0)
            self.metrics_table.setItem(row, 2, QTableWidgetItem(f"{area:.0f}"))

            # Col 3: Comp. (companion area)
            match = _re.search(r'R(\d+)', filename)
            rep_key = match.group(1) if match else '1'
            if signal == 'Direct':
                a_uib = uib_area_by_rep.get(rep_key, 0)
                self.metrics_table.setItem(row, 3, QTableWidgetItem(f"{a_uib:.0f}" if a_uib > 0 else "-"))
            else:
                a_direct = direct_area_by_rep.get(rep_key, 0)
                item_ad = QTableWidgetItem(f"{a_direct:.0f}" if a_direct > 0 else "-")
                item_ad.setToolTip("\u00c0rea Direct companion")
                self.metrics_table.setItem(row, 3, item_ad)

            # Col 4: A254 (àrea 254nm)
            a254 = khp.get('a254_area', 0)
            self.metrics_table.setItem(row, 4, QTableWidgetItem(f"{a254:.1f}" if a254 > 0 else "-"))

            # Col 5: RF_MASS
            rf_mass = khp.get('rf_mass_doc', 0) or khp.get('rf_mass', 0)
            self.metrics_table.setItem(row, 5, QTableWidgetItem(f"{rf_mass:.1f}" if rf_mass > 0 else "-"))

            # Col 6: t_max
            peak_info = khp.get('peak_info') or {}
            t_max = khp.get('t_retention', 0) or peak_info.get('t_max', 0) or khp.get('t_doc_max', 0)
            self.metrics_table.setItem(row, 6, QTableWidgetItem(f"{t_max:.2f}" if t_max > 0 else "-"))

            # Col 7: FWHM
            fwhm = khp.get('fwhm_doc', 0)
            item_fwhm = QTableWidgetItem(f"{fwhm:.2f}" if fwhm > 0 else "-")
            if fwhm > FWHM_THRESHOLD:
                item_fwhm.setBackground(QColor(255, 200, 100))
                item_fwhm.setToolTip(f"FWHM elevat (>{FWHM_THRESHOLD} min)")
            self.metrics_table.setItem(row, 7, item_fwhm)

            # Col 8: SNR
            snr = khp.get('snr', 0)
            item_snr = QTableWidgetItem(f"{snr:.0f}" if snr > 0 else "-")
            if 0 < snr < 10:
                item_snr.setBackground(QColor(255, 200, 100))
            self.metrics_table.setItem(row, 8, item_snr)

            # Col 9: Shift (segons)
            shift_sec = khp.get('shift_sec', khp.get('shift_min', 0) * 60)
            self.metrics_table.setItem(row, 9, QTableWidgetItem(f"{shift_sec:+.1f}s" if shift_sec != 0 else "-"))

            # Col 10: R² bigaussian
            bigauss = khp.get('bigaussian_doc')
            if bigauss and isinstance(bigauss, dict):
                r2 = bigauss.get('r2', 0)
                bg_status = bigauss.get('status', 'INVALID')
                item_r2 = QTableWidgetItem(f"{r2:.3f}")
                if bg_status == 'VALID':
                    item_r2.setBackground(QColor(150, 255, 150))
                elif bg_status == 'CHECK':
                    item_r2.setBackground(QColor(255, 255, 150))
                else:
                    item_r2.setBackground(QColor(255, 200, 100))
                asym = bigauss.get('asymmetry', 0)
                sym = khp.get('symmetry', 0)
                tip = f"Fit {bg_status}\nR\u00b2={r2:.4f}\nAsimetria={asym:.2f}"
                if sym > 0:
                    tip += f"\nSimetria={sym:.2f}"
                item_r2.setToolTip(tip)
            else:
                item_r2 = QTableWidgetItem("-")
            self.metrics_table.setItem(row, 10, item_r2)

            # Col 9: Estat badge
            raw_anomalies = khp.get('calibration_anomalies', [])
            cal_anomalies = [
                a for a in raw_anomalies
                if not isinstance(a, dict) or a.get('code', '') not in IGNORED_KHP_CODES
            ]

            if 'calibration_anomalies' not in khp:
                # Dades anteriors al sistema d'anomalies — sense info QA/QC
                item_status = QTableWidgetItem("?")
                item_status.setBackground(QColor(220, 220, 220))
                item_status.setToolTip("Sense info QA/QC \u2014 reimportar per obtenir-la")
            elif not raw_anomalies:
                # calibration_anomalies existeix i és buit → tot OK
                item_status = QTableWidgetItem("\u2714")
                item_status.setBackground(QColor(150, 255, 150))
                item_status.setToolTip("Sense anomalies")
            else:
                classified = classify_anomalies(cal_anomalies)
                has_blockers = len(classified["blocker"]) > 0
                has_warnings = len(classified["warning"]) > 0

                if has_blockers:
                    status_text = "\u2718"
                    color = QColor(255, 150, 150)
                elif has_warnings:
                    status_text = "\u26a0"
                    color = QColor(255, 200, 100)
                elif cal_anomalies:
                    status_text = "\u2139"
                    color = QColor(255, 255, 150)
                else:
                    status_text = "\u2714"
                    color = QColor(150, 255, 150)

                item_status = QTableWidgetItem(status_text)
                item_status.setBackground(color)

                # Tooltip amb accions
                tooltip_lines = []
                for a in cal_anomalies:
                    if isinstance(a, dict):
                        code = a.get("code", "")
                        entry = ANOMALY_CATALOG.get(code, {})
                        label = a.get("label", code)
                        action = entry.get("action", "")
                        sev_icon = {
                            "blocker": "\u2718",
                            "warning": "\u26a0",
                            "info": "\u2139",
                        }.get(a.get("severity", "info"), "")
                        line = f"{sev_icon} {label}"
                        if action:
                            line += f"\n   \u2192 {action}"
                        tooltip_lines.append(line)
                if tooltip_lines:
                    item_status.setToolTip("\n".join(tooltip_lines))
            self.metrics_table.setItem(row, 11, item_status)

            # Col 12: Checkbox outlier
            is_outlier = khp.get('is_outlier', False)
            cb = QCheckBox()
            cb.setChecked(is_outlier)
            cb.setToolTip("Marcar com a outlier (no s'usa per calibrar)")

            cb_widget = QWidget()
            cb_layout = QHBoxLayout(cb_widget)
            cb_layout.addWidget(cb)
            cb_layout.setAlignment(Qt.AlignCenter)
            cb_layout.setContentsMargins(0, 0, 0, 0)

            replica_num = khp.get('replica_num', 0)
            signal_type = signal
            cb.stateChanged.connect(
                lambda state, rn=replica_num, st=signal_type:
                    self._on_metrics_outlier_toggled(rn, st, state)
            )

            self.metrics_table.setCellWidget(row, 12, cb_widget)

            # Apply grey if outlier
            if is_outlier:
                for col in range(12):
                    item = self.metrics_table.item(row, col)
                    if item:
                        item.setBackground(QColor(230, 230, 230))
                        item.setForeground(QColor(160, 160, 160))

            # Sub-rows per anomalies (blocker/warning)
            if raw_anomalies and cal_anomalies:
                classified_sub = classify_anomalies(cal_anomalies)
                vis_anomalies = classified_sub.get("blocker", []) + classified_sub.get("warning", [])
                for anom in vis_anomalies:
                    if not isinstance(anom, dict):
                        continue
                    sub_row = self.metrics_table.rowCount()
                    self.metrics_table.insertRow(sub_row)
                    code = anom.get("code", "")
                    entry = ANOMALY_CATALOG.get(code, {})
                    label = anom.get("label", code)
                    action = entry.get("action", "")
                    sev = anom.get("severity", "info")
                    icon = "\u2718" if sev == "blocker" else "\u26a0"
                    text = f"  \u21b3 {icon} {code} \u2192 {action}" if action else f"  \u21b3 {icon} {label}"
                    # Merged-style: col 0 empty, text at col 1 spanning
                    item_empty = QTableWidgetItem("")
                    item_empty.setBackground(QColor(248, 249, 250))
                    self.metrics_table.setItem(sub_row, 0, item_empty)
                    item_text = QTableWidgetItem(text)
                    item_text.setToolTip(label)
                    sub_font = QFont()
                    sub_font.setPointSize(8)
                    item_text.setFont(sub_font)
                    item_text.setForeground(QColor('#C62828') if sev == "blocker" else QColor('#E65100'))
                    item_text.setBackground(QColor(248, 249, 250))
                    self.metrics_table.setItem(sub_row, 1, item_text)
                    # Fill remaining cells with grey background
                    for c in range(2, 13):
                        filler = QTableWidgetItem("")
                        filler.setBackground(QColor(248, 249, 250))
                        self.metrics_table.setItem(sub_row, c, filler)
                    # Span col 1 across visible area
                    self.metrics_table.setSpan(sub_row, 1, 1, 11)

    def _on_metrics_outlier_toggled(self, replica_num, signal_type, state):
        """Handler quan canvia el checkbox outlier a la taula de mètriques."""
        is_outlier = (state == Qt.Checked.value if hasattr(Qt.Checked, 'value') else state == 2)

        try:
            from hpsec_calibrate import load_local_calibrations, save_local_calibrations
            import os

            seq_path = self.main_window.seq_path
            if not seq_path:
                return

            calibrations = load_local_calibrations(seq_path)
            seq_name = os.path.basename(seq_path)

            updated = False
            for cal in calibrations:
                if cal.get('seq_name') != seq_name:
                    continue

                replicas_info = cal.get('replicas_info', [])
                for rep in replicas_info:
                    if rep.get('replica_num') == replica_num:
                        rep['is_outlier'] = is_outlier
                        updated = True
                        break

                replica_comp = cal.get('replica_comparison', {})
                replica_details = replica_comp.get('replica_details', [])
                for rep in replica_details:
                    if rep.get('replica_num') == replica_num:
                        rep['is_outlier'] = is_outlier
                        updated = True
                        break

            if updated:
                save_local_calibrations(seq_path, calibrations)
                action = "marcada com a Outlier" if is_outlier else "restaurada com a V\u00e0lida"
                self.main_window.set_status(f"R\u00e8plica R{replica_num} ({signal_type}) {action}", 3000)

        except Exception as e:
            logger.error(f"Error canviant estat r\u00e8plica: {e}")

    # =========================================================================
    # VALIDATION (intern — no mostra UI)
    # =========================================================================

    def _update_validation(self, result):
        """Construeix avisos estructurats i els guarda a calibration_data."""
        from hpsec_warnings import get_max_warning_level

        warnings_structured = list(result.get("warnings_structured", []))
        max_level = get_max_warning_level(warnings_structured)

        if self.calibration_data:
            self.calibration_data["warnings_structured"] = warnings_structured
            self.calibration_data["warning_level"] = max_level
            self.main_window.calibration_data = self.calibration_data

    # =========================================================================
    # HISTORY (amb UIB)
    # =========================================================================

    def _update_history(self, result):
        """Actualitza l'històric amb gràfics Direct + UIB + DOC/254."""
        import os
        import re

        seq_path = self.main_window.seq_path or ""
        current_seq = os.path.basename(seq_path).replace('_SEQ', '').replace('_BP', '') if seq_path else ""

        # Determinar mètode
        method = "COLUMN"
        khp_data = result.get("khp_data") or result.get("khp_data_direct") or result.get("khp_data_uib")
        if khp_data and khp_data.get('is_bp', False):
            method = "BP"
        elif self.main_window.imported_data:
            if self.main_window.imported_data.get("method", "").upper() == "BP":
                method = "BP"

        khp_conc = result.get("khp_conc", 5)

        # Volum
        current_volume = None
        if khp_data:
            current_volume = khp_data.get('volume_uL')
            if not current_volume:
                replicas = khp_data.get('replicas') or []
                if replicas:
                    current_volume = replicas[0].get('volume_uL')
        if not current_volume and self.main_window.imported_data:
            current_volume = self.main_window.imported_data.get('injection_volume')
        if not current_volume:
            current_volume = 400 if method == "COLUMN" else 100

        # Sensibilitat UIB (per filtrar històric)
        current_uib_sensitivity = None
        khp_data_uib = result.get("khp_data_uib")
        if khp_data_uib:
            current_uib_sensitivity = khp_data_uib.get('uib_sensitivity')
            if not current_uib_sensitivity:
                uib_reps = khp_data_uib.get('replicas') or khp_data_uib.get('all_khp_data') or []
                if uib_reps and isinstance(uib_reps, list):
                    current_uib_sensitivity = uib_reps[0].get('uib_sensitivity')

        # Inicialitzar
        self._history_data = []
        self._selected_history_idx = -1
        self.toggle_outlier_btn.setEnabled(False)

        try:
            history = load_khp_history(seq_path)
            if not history:
                self.history_graph.clear()
                self.history_uib_graph.clear()
                self.history_doc254_graph.clear()
                self.history_uib254_graph.clear()
                self.calibration_line_graph.clear()
                self.cal_line_group.setVisible(False)
                self.history_group.setVisible(False)
                return

            include_outliers = self.show_outliers_cb.isChecked()

            filtered_history = []
            for cal in history:
                if cal.get('area', 0) <= 0:
                    continue
                if not include_outliers and cal.get('is_outlier', False):
                    continue

                cal_mode = cal.get('mode', 'COLUMN')
                cal_conc = cal.get('conc_ppm', 0)
                cal_vol = cal.get('volume_uL', current_volume)

                if cal_mode != method:
                    continue
                tol = max(0.05, khp_conc * 0.1)
                if abs(cal_conc - khp_conc) >= tol:
                    continue
                if cal_vol and current_volume and cal_vol != current_volume:
                    continue
                # Filtrar per sensibilitat UIB (700/1000 ppb)
                if current_uib_sensitivity:
                    cal_sens = cal.get('uib_sensitivity')
                    if cal_sens and cal_sens != current_uib_sensitivity:
                        continue

                filtered_history.append(cal)

            # Deduplicar
            seen_seqs = {}
            for cal in filtered_history:
                key = cal.get('seq_name', '') + '_' + cal.get('condition_key', '')
                seen_seqs[key] = cal
            filtered_history = list(seen_seqs.values())

            if not filtered_history:
                self.history_graph.clear()
                self.history_uib_graph.clear()
                self.history_doc254_graph.clear()
                self.history_uib254_graph.clear()
                self.calibration_line_graph.clear()
                self.cal_line_group.setVisible(False)
                self.history_group.setVisible(False)
                self.history_filters_label.setText("")
                return

            outlier_text = " (amb outliers)" if include_outliers else ""
            sens_text = f" \u00b7 UIB {int(current_uib_sensitivity)}ppb" if current_uib_sensitivity else ""
            self.history_filters_label.setText(
                f"{method} \u00b7 KHP{khp_conc:g}ppm \u00b7 {int(current_volume)}\u00b5L{sens_text}{outlier_text} ({len(filtered_history)})"
            )

            self._history_data = filtered_history
            self.history_group.setVisible(True)

            # Ordenar per número de SEQ
            def get_seq_num(cal):
                match = re.search(r'(\d+)', cal.get('seq_name', ''))
                return int(match.group(1)) if match else 0
            filtered_history.sort(key=get_seq_num)

            # Índexs vàlids
            valid_indices = set()
            for idx, cal in enumerate(filtered_history):
                cal_seq_raw = cal.get('seq_name', 'N/A').replace('_SEQ', '').replace('_BP', '')
                stored_valid = cal.get('valid_for_calibration', True)
                stored_outlier = cal.get('is_outlier', False)
                is_valid = stored_valid and not stored_outlier
                is_current = (cal_seq_raw == current_seq)
                if is_valid and not is_current:
                    valid_indices.add(idx)

            # Gràfics de barres: Direct + UIB + DOC/254
            self.history_graph.plot_history(filtered_history, current_seq, valid_indices)
            self.history_doc254_graph.plot_history(filtered_history, current_seq, valid_indices)

            # UIB graphs: mostrar només si hi ha dades UIB
            has_uib_data = any(cal.get('area_u', 0) > 0 for cal in filtered_history)
            if has_uib_data:
                self.history_uib_graph.setVisible(True)
                self.history_uib_graph.plot_history(filtered_history, current_seq, valid_indices)
                # UIB/254 graph
                has_uib254 = any(cal.get('d254_u', 0) > 0 for cal in filtered_history)
                if has_uib254:
                    self.history_uib254_graph.setVisible(True)
                    self.history_uib254_graph.plot_history(filtered_history, current_seq, valid_indices)
                else:
                    self.history_uib254_graph.setVisible(False)
            else:
                self.history_uib_graph.setVisible(False)
                self.history_uib254_graph.setVisible(False)

            # Recta de calibració — regressió guardada a Calibration_Reference.json
            try:
                config = get_config()

                cal_direct = get_active_global_calibration(signal='direct')
                imported = self.main_window.imported_data or {}
                uib_sens = imported.get("uib_sensitivity")
                # Només carregar calibració UIB si la SEQ té dades UIB
                cal_uib = None
                if has_uib_data:
                    cal_uib = get_active_global_calibration(signal='uib', sensitivity=uib_sens)

                if not cal_direct and not cal_uib:
                    self.calibration_line_graph.clear()
                    self.cal_line_group.setVisible(False)
                else:
                    # Punt actual Direct
                    current_area_d = result.get('khp_area_direct') or result.get('khp_area', 0)
                    current_direct = None
                    if current_area_d > 0 and khp_conc > 0 and current_volume > 0:
                        current_direct = {
                            'ug_doc': khp_conc * current_volume / 1000,
                            'area': current_area_d,
                        }

                    # Punt actual UIB
                    current_area_u = result.get('khp_area_uib', 0)
                    current_uib = None
                    if current_area_u > 0 and khp_conc > 0 and current_volume > 0:
                        current_uib = {
                            'ug_doc': khp_conc * current_volume / 1000,
                            'area': current_area_u,
                        }

                    # Títol
                    mode_key = method.lower()
                    rf_d = 0
                    if cal_direct:
                        rf_dict = cal_direct.get('rf_mass_cal', {})
                        rf_d = rf_dict.get(mode_key, 0) if isinstance(rf_dict, dict) else 0
                    ref_label = f"RF={rf_d:.0f}" if rf_d > 0 else "N/A"
                    dual_text = " (Direct + UIB)" if cal_uib else ""
                    self.cal_line_group.setTitle(
                        f"Recta de calibraci\u00f3 vigent \u2014 {ref_label} ({method}){dual_text}"
                    )
                    self.cal_line_group.setVisible(True)

                    self.calibration_line_graph.plot_stored_regression(
                        cal_direct=cal_direct,
                        cal_uib=cal_uib if current_uib or (cal_uib and cal_uib.get('regression_data')) else None,
                        current_direct=current_direct,
                        current_uib=current_uib,
                        current_mode=method.lower(),
                        current_seq_name=current_seq,
                        warning_pct=config.get('calibration', 'qc_thresholds', 'warning_pct', default=5.0),
                        fail_pct=config.get('calibration', 'qc_thresholds', 'fail_pct', default=10.0),
                    )

                    # Etiquetes SEQs visibles a la recta
                    seq_names = []
                    for cal in filtered_history:
                        sn = cal.get('seq_name', '')
                        m = re.search(r'(\d+)', sn)
                        short = m.group(1) if m else sn
                        if short and short not in seq_names:
                            seq_names.append(short)
                    if seq_names:
                        current_short = re.search(r'(\d+)', current_seq or '')
                        current_short = current_short.group(1) if current_short else ''
                        parts = []
                        for s in seq_names:
                            if s == current_short:
                                parts.append(f"<b>\u25b8{s}\u25c2</b>")
                            else:
                                parts.append(s)
                        self.cal_line_seqs_label.setText(f"SEQs: {', '.join(parts)}")
                    else:
                        self.cal_line_seqs_label.setText("")
            except Exception as e:
                logger.error(f"Error plotant gr\u00e0fic calibraci\u00f3: {e}")
                import traceback; traceback.print_exc()
                self.calibration_line_graph.clear()
                self.cal_line_group.setVisible(False)

            # Resum ampliat amb UIB
            n_valid = len(valid_indices)
            n_excluded = len(filtered_history) - n_valid

            summary_parts = [f"{n_valid} v\u00e0lides \u00b7 {n_excluded} excloses"]

            if n_valid > 0:
                valid_areas = [filtered_history[i].get('area', 0) for i in valid_indices]
                mean_area = np.mean(valid_areas)
                std_area = np.std(valid_areas) if len(valid_areas) > 1 else 0
                cv_area = (std_area / mean_area * 100) if mean_area > 0 else 0
                summary_parts.append(f"Direct: {mean_area:.0f} \u00b1 {std_area:.0f} ({cv_area:.1f}%)")

                # UIB stats
                if has_uib_data:
                    valid_uib = [filtered_history[i].get('area_u', 0) for i in valid_indices if filtered_history[i].get('area_u', 0) > 0]
                    if valid_uib:
                        mean_uib = np.mean(valid_uib)
                        std_uib = np.std(valid_uib) if len(valid_uib) > 1 else 0
                        cv_uib = (std_uib / mean_uib * 100) if mean_uib > 0 else 0
                        summary_parts.append(f"UIB: {mean_uib:.0f} \u00b1 {std_uib:.0f} ({cv_uib:.1f}%)")

            self.history_summary.setText(" | ".join(summary_parts))

        except Exception as e:
            import traceback
            logger.warning(f"Error carregant hist\u00f2ric: {e}")
            traceback.print_exc()
            self.history_graph.clear()
            self.history_uib_graph.clear()
            self.history_doc254_graph.clear()
            self.history_uib254_graph.clear()
            self.calibration_line_graph.clear()
            self.cal_line_group.setVisible(False)
            self.history_group.setVisible(False)

    def _on_history_bar_selected(self, real_idx):
        """Handler quan es clica una barra del gràfic històric."""
        self._selected_history_idx = real_idx
        self.toggle_outlier_btn.setEnabled(True)

        if hasattr(self, '_history_data') and 0 <= real_idx < len(self._history_data):
            cal = self._history_data[real_idx]
            is_outlier = cal.get('is_outlier', False) or cal.get('manual_outlier', False)
            seq_name = cal.get('seq_name', '').replace('_SEQ', '').replace('_BP', '')
            self.toggle_outlier_btn.setText(
                f"Desmarcar Outlier ({seq_name})" if is_outlier else f"Marcar Outlier ({seq_name})"
            )
        else:
            self.toggle_outlier_btn.setText("Marcar Outlier")

    # =========================================================================
    # DELAY DIAGNOSTIC SECTION
    # =========================================================================

    def _build_delay_diagnostic_section(self, parent_layout):
        """Construeix la secció de diagnòstic delay HPLC\u2194TOC."""
        self.delay_group = QGroupBox("Diagn\u00f2stic Delay HPLC\u2194TOC")
        self.delay_group.setVisible(False)
        self.delay_group.setStyleSheet(
            "QGroupBox { font-weight: bold; color: #1A5276; border: 2px solid #E67E22; "
            "border-radius: 6px; margin-top: 8px; padding-top: 12px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }"
        )
        delay_main = QVBoxLayout(self.delay_group)
        delay_main.setSpacing(8)

        # Info frame
        info_frame = QFrame()
        info_frame.setStyleSheet(
            "QFrame { background-color: #FAFAFA; border: 1px solid #DDD; "
            "border-radius: 4px; padding: 8px; }"
        )
        info_layout = QGridLayout(info_frame)
        info_layout.setSpacing(6)

        info_layout.addWidget(QLabel("<b>Shift KHP (DOC\u2194254nm):</b>"), 0, 0)
        self.delay_shift_label = QLabel("-")
        self.delay_shift_label.setStyleSheet("font-size: 13px;")
        info_layout.addWidget(self.delay_shift_label, 0, 1)

        info_layout.addWidget(QLabel("<b>Net delay actual (MasterFile):</b>"), 0, 2)
        self.delay_current_label = QLabel("-")
        self.delay_current_label.setStyleSheet("font-size: 13px;")
        info_layout.addWidget(self.delay_current_label, 0, 3)

        info_layout.addWidget(QLabel("<b>Injeccions / Files TOC:</b>"), 1, 0)
        self.delay_counts_label = QLabel("-")
        info_layout.addWidget(self.delay_counts_label, 1, 1, 1, 3)

        delay_main.addWidget(info_frame)

        # Quality indicator
        self.delay_quality_frame = QFrame()
        self.delay_quality_frame.setStyleSheet(
            "QFrame { border-radius: 4px; padding: 6px; }"
        )
        quality_layout = QHBoxLayout(self.delay_quality_frame)
        quality_layout.setContentsMargins(8, 4, 8, 4)
        self.delay_quality_text = QLabel("-")
        self.delay_quality_text.setWordWrap(True)
        quality_layout.addWidget(self.delay_quality_text, 1)
        delay_main.addWidget(self.delay_quality_frame)

        # Botó correcció (visible només si shift significatiu)
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.delay_apply_btn = QPushButton("Corregir delay i reimportar")
        self.delay_apply_btn.setToolTip(
            "Afegeix el delay corregit al MasterFile (sense sobreescriure l'original),\n"
            "regenera 4-TOC_CALC i reimporta la seq\u00fc\u00e8ncia."
        )
        self.delay_apply_btn.setStyleSheet(
            "QPushButton { background-color: #E67E22; color: white; "
            "font-weight: bold; padding: 8px 16px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #D35400; }"
            "QPushButton:disabled { background-color: #CCC; color: #999; }"
        )
        self.delay_apply_btn.setVisible(False)
        self.delay_apply_btn.clicked.connect(self._delay_apply_and_reimport)
        btn_layout.addWidget(self.delay_apply_btn)
        delay_main.addLayout(btn_layout)

        # State
        self._delay_original = None
        self._delay_mf_path = None
        self._delay_is_bp = False
        self._delay_shift_min = 0  # Shift detectat del KHP

        parent_layout.addWidget(self.delay_group)

    def _update_delay_diagnostic(self, result):
        """Actualitza la secció de diagnòstic delay."""
        import os

        imported_data = self.main_window.imported_data or {}
        method = result.get("mode") or ""
        if not method or method == "-":
            method = imported_data.get("method") or "COLUMN"
        method = method.upper()
        is_bp = method == "BP"
        self._delay_is_bp = is_bp

        shift_min = result.get("shift_direct", 0)
        shift_abs = abs(shift_min)
        self._delay_shift_min = shift_min

        show_delay = is_bp or shift_abs > 2.0

        if not show_delay:
            self.delay_group.setVisible(False)
            return

        mf_path = imported_data.get("master_file")
        if not mf_path:
            seq_path = self.main_window.seq_path
            if seq_path:
                from pathlib import Path
                candidates = list(Path(seq_path).glob("*MasterFile*.xlsx"))
                candidates = [c for c in candidates if 'backup' not in c.name.lower()]
                if candidates:
                    mf_path = str(candidates[0])

        if not mf_path or not os.path.exists(str(mf_path)):
            self.delay_group.setVisible(False)
            return

        self._delay_mf_path = str(mf_path)

        try:
            from hpsec_delay import read_current_delay
            current_delay = read_current_delay(mf_path)
        except Exception as e:
            logger.warning(f"Error llegint delay: {e}")
            current_delay = None

        if current_delay is None:
            current_delay = 0.0

        self._delay_original = current_delay

        # Comptadors (des de imported_data, sense llegir MasterFile)
        samples = imported_data.get("samples") or {}
        n_injections = len(samples) if isinstance(samples, dict) else 0
        n_toc = imported_data.get("n_toc_rows", 0)

        shift_sec = shift_min * 60
        self.delay_shift_label.setText(f"{shift_sec:.1f}s ({shift_min:.2f} min)")
        current_delay_sec = current_delay * 60
        self.delay_current_label.setText(f"{current_delay_sec:.1f}s ({current_delay:.3f} min)")
        if n_injections > 0:
            self.delay_counts_label.setText(f"{n_injections} injeccions")
        else:
            self.delay_counts_label.setText("-")

        if is_bp:
            if shift_abs < 0.5:
                color = "#27AE60"
                bg = "#E8F8F5"
                icon = "\u2714"
                text = f"{icon} Shift KHP petit \u2014 delay probablement correcte."
            elif shift_abs < 2.0:
                color = "#E67E22"
                bg = "#FEF9E7"
                icon = "\u26a0"
                text = (f"{icon} Shift KHP moderat ({shift_sec:.1f}s). "
                        "Pot indicar un delay imprec\u00eds. Revisar el cromatograma DOC.")
            else:
                color = "#E74C3C"
                bg = "#FDEDEC"
                icon = "\u2718"
                text = (f"{icon} Shift KHP gran ({shift_sec:.1f}s). "
                        "Les files TOC poden estar mal assignades.")
        else:
            color = "#E67E22"
            bg = "#FEF9E7"
            icon = "\u26a0"
            text = (f"{icon} Shift KHP gran per COLUMN ({shift_sec:.1f}s). "
                    "Normalment no afecta l'an\u00e0lisi per\u00f2 pot indicar un problema.")

        self.delay_quality_frame.setStyleSheet(
            f"QFrame {{ background-color: {bg}; border: 1px solid {color}; "
            f"border-radius: 4px; padding: 6px; }}"
        )
        self.delay_quality_text.setText(text)

        # Botó correcció: visible si shift > threshold
        show_btn = (is_bp and shift_abs >= 0.5) or shift_abs > 2.0
        if show_btn:
            new_delay = current_delay + shift_min
            self.delay_apply_btn.setText(
                f"Corregir delay ({shift_min:+.2f} min) i reimportar"
            )
            self.delay_apply_btn.setVisible(True)
            self.delay_apply_btn.setEnabled(True)
        else:
            self.delay_apply_btn.setVisible(False)

        self.delay_group.setVisible(True)

    def _delay_apply_and_reimport(self):
        """Corregeix el delay al MasterFile i emet senyal per reimportar."""
        from PySide6.QtWidgets import QMessageBox

        old_delay = self._delay_original or 0
        shift = self._delay_shift_min
        new_delay = old_delay + shift
        mf_path = self._delay_mf_path

        if mf_path is None:
            QMessageBox.warning(self, "Error", "No s'ha trobat el MasterFile.")
            return

        reply = QMessageBox.question(
            self,
            "Corregir delay",
            f"S'ha detectat un shift de {shift * 60:.1f}s ({shift:+.2f} min).\n\n"
            f"Es corregir\u00e0 el delay al MasterFile:\n"
            f"  Net delay: {old_delay:.3f} \u2192 {new_delay:.3f} min\n\n"
            f"  \u2022 S'afegir\u00e0 'Net delay (Suite)' al MasterFile (l'original es conserva)\n"
            f"  \u2022 Es regenerar\u00e0 4-TOC_CALC\n"
            f"  \u2022 Es reimportar\u00e0 la seq\u00fc\u00e8ncia\n\n"
            f"Continuar?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        self.delay_apply_btn.setEnabled(False)
        self.delay_apply_btn.setText("Aplicant...")
        from PySide6.QtWidgets import QApplication
        QApplication.processEvents()

        try:
            from hpsec_delay import update_masterfile_delay
            update_result = update_masterfile_delay(
                mf_path, new_delay, backup=True
            )

            if not update_result.get('success'):
                error_msg = update_result.get('error', 'Error desconegut')
                QMessageBox.critical(
                    self, "Error",
                    f"Error actualitzant MasterFile:\n{error_msg}"
                )
                return

            backup_path = update_result.get('backup_path', '')
            logger.info(f"Delay corregit: {old_delay:.3f} \u2192 {new_delay:.3f}, "
                        f"backup: {backup_path}")

            # Emetre senyal per reimportar des del wizard (torna a tab 0)
            self.delay_corrected.emit()

        except Exception as e:
            logger.error(f"Error aplicant delay: {e}")
            import traceback; traceback.print_exc()
            QMessageBox.critical(
                self, "Error",
                f"Error durant l'aplicaci\u00f3 del delay:\n{e}"
            )
        finally:
            self.delay_apply_btn.setText(
                f"Corregir delay ({shift:+.2f} min) i reimportar"
            )
            self.delay_apply_btn.setEnabled(True)

    # =========================================================================
    # HISTORY LEGEND + OUTLIER TOGGLE
    # =========================================================================

    def _show_history_legend(self):
        """Mostra diàleg amb llegenda i detalls del gràfic d'històric."""
        from PySide6.QtWidgets import QMessageBox

        legend_html = """
<h3>Llegenda del Gr\u00e0fic QA/QC Hist\u00f2ric</h3>

<p><b>Qu\u00e8 fa el QA/QC KHP:</b></p>
<p>Verifica la mesura del KHP respecte la calibraci\u00f3 global (rf_mass_cal)
i determina el time shift necessari per a la quantificaci\u00f3.</p>

<p><b>Colors de les barres:</b></p>
<ul>
<li><span style='color:#27AE60'>\u25a0 Verd</span> - SEQ actual (oberta)</li>
<li><span style='color:#5DADE2'>\u25a0 Blau</span> - Verificacions v\u00e0lides</li>
<li><span style='color:#E74C3C'>\u25a0 Vermell</span> - Outliers (exclosos de la mitjana)</li>
</ul>

<p><b>L\u00ednies horitzontals:</b></p>
<ul>
<li><span style='color:#27AE60'>\u2501\u2501\u2501</span> Mitjana de verificacions v\u00e0lides</li>
<li><span style='color:#27AE60'>- - -</span> Desviaci\u00f3 est\u00e0ndard (\u00b11\u03c3)</li>
</ul>

<p><i>Nota: Pots marcar/desmarcar outliers manualment amb el bot\u00f3 "Marcar Outlier"</i></p>
"""
        msg = QMessageBox(self)
        msg.setWindowTitle("Llegenda Gr\u00e0fic Hist\u00f2ric")
        msg.setTextFormat(Qt.RichText)
        msg.setText(legend_html)
        msg.setIcon(QMessageBox.Information)
        msg.exec()

    def _on_show_outliers_changed(self, state):
        if self.calibration_data:
            self._update_history(self.calibration_data)

    def _toggle_outlier(self):
        """Marca o desmarca la calibració seleccionada com a outlier."""
        from PySide6.QtWidgets import QMessageBox, QInputDialog
        import json
        from datetime import datetime

        row = getattr(self, '_selected_history_idx', -1)
        if row < 0 or not hasattr(self, '_history_data') or row >= len(self._history_data):
            return

        cal = self._history_data[row]
        seq_name = cal.get('seq_name', 'N/A')
        current_outlier = cal.get('is_outlier', False) or cal.get('manual_outlier', False)

        if current_outlier:
            reply = QMessageBox.question(
                self, "Desmarcar Outlier",
                f"Vols desmarcar '{seq_name}' com a outlier?\n\n"
                f"Tornar\u00e0 a incloure's en la mitjana.",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                cal['manual_outlier'] = False
                cal['is_outlier'] = False
                cal['outlier_reason'] = None
                self._save_outlier_change(cal, False, None)
        else:
            reason, ok = QInputDialog.getText(
                self, "Marcar Outlier",
                f"Motiu per marcar '{seq_name}' com a outlier:",
                text="Manual exclusion"
            )
            if ok and reason:
                cal['manual_outlier'] = True
                cal['is_outlier'] = True
                cal['outlier_reason'] = reason
                cal['outlier_date'] = datetime.now().isoformat()
                self._save_outlier_change(cal, True, reason)

        if self.calibration_data:
            self._update_history(self.calibration_data)

    def _save_outlier_change(self, cal, is_outlier, reason):
        """Guarda el canvi d'outlier a l'històric JSON."""
        import json
        from pathlib import Path
        from datetime import datetime

        seq_path = self.main_window.seq_path
        if not seq_path:
            return

        try:
            seq_dir = Path(seq_path)
            history_file = None

            for parent in [seq_dir.parent, seq_dir.parent.parent]:
                candidate = parent / "khp_calibration_history.json"
                if candidate.exists():
                    history_file = candidate
                    break

            if not history_file:
                logger.warning("No s'ha trobat fitxer d'hist\u00f2ric")
                return

            with open(history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)

            seq_name = cal.get('seq_name')
            updated = False

            for h in history:
                if h.get('seq_name') == seq_name:
                    h['manual_outlier'] = is_outlier
                    h['is_outlier'] = is_outlier
                    h['outlier_reason'] = reason
                    h['outlier_modified'] = datetime.now().isoformat()
                    updated = True
                    break

            if updated:
                with open(history_file, 'w', encoding='utf-8') as f:
                    json.dump(history, f, indent=2, ensure_ascii=False)
                logger.info(f"Outlier actualitzat per {seq_name}: {is_outlier}")

        except Exception as e:
            logger.warning(f"Error guardant outlier: {e}")

    def _use_historical_average(self):
        """Calibra usant la mitjana de les calibracions vàlides AMB CONDICIONS IDÈNTIQUES."""
        from PySide6.QtWidgets import QMessageBox

        if not hasattr(self, '_history_data') or not self._history_data:
            QMessageBox.warning(self, "Error", "No hi ha hist\u00f2ric disponible.")
            return

        result = self.calibration_data or {}
        khp_data = result.get("khp_data") or result.get("khp_data_direct") or result.get("khp_data_uib")

        current_method = "COLUMN"
        if khp_data and khp_data.get('is_bp', False):
            current_method = "BP"
        elif self.main_window.imported_data:
            if self.main_window.imported_data.get("method", "").upper() == "BP":
                current_method = "BP"

        current_conc = result.get("khp_conc", 5)

        current_volume = None
        if khp_data:
            current_volume = khp_data.get('volume_uL')
        if not current_volume and self.main_window.imported_data:
            current_volume = self.main_window.imported_data.get('injection_volume')
        if not current_volume:
            current_volume = 400 if current_method == "COLUMN" else 100

        valid_cals = []
        for cal in self._history_data:
            is_outlier = cal.get('is_outlier', False) or cal.get('manual_outlier', False)
            area = cal.get('area', 0)
            if is_outlier or area <= 0:
                continue

            cal_mode = cal.get('mode', 'COLUMN')
            cal_conc = cal.get('conc_ppm', 0)
            cal_vol = cal.get('volume_uL', current_volume)

            if cal_mode != current_method:
                continue
            if abs(cal_conc - current_conc) >= 1:
                continue
            if cal_vol and current_volume and cal_vol != current_volume:
                continue

            valid_cals.append(cal)

        if not valid_cals:
            QMessageBox.warning(
                self, "Error",
                "No hi ha calibracions v\u00e0lides per calcular la mitjana."
            )
            return

        areas = [c.get('area', 0) for c in valid_cals]
        concs = [c.get('conc_ppm', 5) for c in valid_cals]
        shifts = [c.get('shift_sec', 0) for c in valid_cals]

        mean_area = np.mean(areas)
        std_area = np.std(areas) if len(areas) > 1 else 0
        mean_conc = np.mean(concs)
        mean_shift = np.mean(shifts)

        rf = mean_area / mean_conc if mean_conc > 0 else 0

        reply = QMessageBox.question(
            self, "Usar Mitjana Hist\u00f2rica",
            f"Calibrar amb mitjana de {len(valid_cals)} calibracions v\u00e0lides:\n\n"
            f"Condicions: {current_method} \u00b7 KHP{current_conc:.0f} \u00b7 {int(current_volume)}\u00b5L\n\n"
            f"\u00c0rea mitjana: {mean_area:.0f} \u00b1 {std_area:.0f}\n"
            f"RF (\u00c0rea/ppm): {rf:.0f}\n"
            f"Shift mitj\u00e0: {mean_shift:.1f} s ({mean_shift/60:.3f} min)\n\n"
            f"Vols aplicar aquesta calibraci\u00f3?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        if self.calibration_data:
            self.calibration_data["rf_direct"] = rf
            self.calibration_data["rf"] = rf
            self.calibration_data["khp_source"] = f"MITJANA HIST\u00d2RICA ({len(valid_cals)} calibracions)"
            self.calibration_data["khp_area_direct"] = mean_area
            self.calibration_data["khp_area"] = mean_area
            self.calibration_data["shift_direct"] = mean_shift / 60
            self.calibration_data["shift_uib"] = mean_shift / 60
            self.main_window.calibration_data = self.calibration_data

        QMessageBox.information(
            self, "Calibraci\u00f3 Aplicada",
            f"Aplicada mitjana de {len(valid_cals)} calibracions\n"
            f"\u00c0rea: {mean_area:.0f} \u00b1 {std_area:.0f}\n"
            f"RF: {rf:.0f}"
        )

    def _go_next(self):
        self.main_window.go_to_tab(2)
