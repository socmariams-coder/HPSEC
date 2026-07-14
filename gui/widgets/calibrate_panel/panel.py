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
    QCheckBox, QDoubleSpinBox, QSlider
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QColor

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from hpsec_calibrate import (
    calibrate_from_import, load_khp_history, load_local_calibrations,
    get_all_active_calibrations, get_rf_mass_cal,
    get_active_global_calibration,
    load_manual_repairs, set_manual_repair, remove_manual_repair, manual_repair_key,
)
from hpsec_config import get_config

import logging
import numpy as np

logger = logging.getLogger(__name__)

# Importar components del paquet
from .worker import CalibrateWorker, SiblingCalibrateWorker
from .graph_widgets import KHPReplicaGraphWidget, HistoryBarWidget, CalibrationLineWidget
# Importar estils compartits
from gui.widgets.styles import (
    PANEL_MARGINS, PANEL_SPACING, STYLE_GROUPBOX,
    COLOR_SUCCESS, COLOR_WARNING, COLOR_ERROR, COLOR_TEXT_SECONDARY,
    COLOR_SUCCESS_LIGHT, COLOR_WARNING_LIGHT, COLOR_ERROR_LIGHT,
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
        # Siblings
        self._sibling_worker = None
        self._sibling_results = {}      # {path: cal_result}
        self._sibling_cards = []        # llista de QFrames creats dinàmicament
        self._active_sibling_path = None  # path del sibling mostrat al detall

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
        self._sibling_worker = None
        self._sibling_results = {}
        self._active_sibling_path = None
        # Netejar cards dinàmiques
        self._clear_sibling_cards()

        # Reset UI elements
        self.condition_selector_frame.setVisible(False)
        self.condition_combo.clear()
        if hasattr(self, 'placeholder'):
            self.placeholder.setVisible(True)
        if hasattr(self, 'compact_header'):
            self.compact_header.setVisible(False)
        pass  # Notes gestionades pel wizard header
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
        # Reset TOC alignment (BP)
        if hasattr(self, '_toc_align_group'):
            self._toc_align_group.setVisible(False)
        self._bp_align_cache = None
        self._bp_delay_original = None
        self._bp_delay_current = None
        if hasattr(self, '_toc_align_figure'):
            self._toc_align_figure.clear()
            self._toc_align_canvas.draw_idle()
        if hasattr(self, '_toc_align_table'):
            self._toc_align_table.setText("")
        if hasattr(self, '_toc_align_info'):
            self._toc_align_info.setText("")

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
                   self._update_toc_alignment,
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

        # === SIBLING CARDS (visible només amb siblings) ===
        self._sibling_cards_layout = QVBoxLayout()
        self._sibling_cards_layout.setSpacing(6)
        content_layout.addLayout(self._sibling_cards_layout)

        # === COMPACT HEADER (substitueix summary_group) ===
        header_row = QHBoxLayout()
        self.compact_header = QLabel()
        self.compact_header.setVisible(False)
        self.compact_header.setWordWrap(True)
        self.compact_header.setTextFormat(Qt.RichText)
        self.compact_header.setStyleSheet(
            "QLabel { background-color: #DBEAFE; border: 1px solid #AED6F1; "
            "border-radius: 6px; padding: 10px 14px; font-size: 12px; }"
        )
        header_row.addWidget(self.compact_header, 1)

        # (Notes gestionades pel wizard header — botó 📝)

        content_layout.addLayout(header_row)

        # === DELAY DIAGNOSTIC ===
        self._build_delay_diagnostic_section(content_layout)

        # === TOC ALIGNMENT (BP only) ===
        self._toc_align_group = QGroupBox("Assignació TOC — DAD 254")
        self._toc_align_group.setVisible(False)
        self._toc_align_group.setStyleSheet(
            "QGroupBox { font-weight: bold; color: #1D4ED8; border: 2px solid #2563EB; "
            "border-radius: 6px; margin-top: 8px; padding-top: 12px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }")
        toc_align_layout = QVBoxLayout(self._toc_align_group)

        # Info label
        self._toc_align_info = QLabel()
        self._toc_align_info.setWordWrap(True)
        self._toc_align_info.setStyleSheet("font-size: 11px; padding: 4px;")
        toc_align_layout.addWidget(self._toc_align_info)

        # --- Controls: delay slider compacte (sota el gràfic) ---
        slider_row = QHBoxLayout()
        slider_row.setSpacing(6)
        slider_row.addWidget(QLabel("<b>Delay:</b>"))

        self._toc_delay_spin = QDoubleSpinBox()
        self._toc_delay_spin.setRange(-5.0, 20.0)
        self._toc_delay_spin.setSingleStep(0.1)
        self._toc_delay_spin.setDecimals(1)
        self._toc_delay_spin.setSuffix(" min")
        self._toc_delay_spin.setFixedWidth(90)
        slider_row.addWidget(self._toc_delay_spin)

        self._toc_delay_slider = QSlider(Qt.Horizontal)
        self._toc_delay_slider.setRange(-50, 200)  # -5.0 to 20.0 min in 0.1 steps
        self._toc_delay_slider.setTickPosition(QSlider.TicksBelow)
        self._toc_delay_slider.setTickInterval(10)
        slider_row.addWidget(self._toc_delay_slider, 1)  # stretch

        self._toc_delay_impact = QLabel("")
        self._toc_delay_impact.setStyleSheet("font-size: 11px; color: #555;")
        self._toc_delay_impact.setFixedWidth(180)
        slider_row.addWidget(self._toc_delay_impact)

        self._toc_delay_reset_btn = QPushButton("Reset")
        self._toc_delay_reset_btn.setFixedWidth(50)
        self._toc_delay_reset_btn.setToolTip("Tornar al delay del MasterFile")
        slider_row.addWidget(self._toc_delay_reset_btn)

        self._toc_delay_apply_btn = QPushButton("Reimportar")
        self._toc_delay_apply_btn.setStyleSheet(
            "QPushButton { background-color: #E67E22; color: white; "
            "font-weight: bold; padding: 4px 10px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #D35400; }")
        self._toc_delay_apply_btn.setVisible(False)
        slider_row.addWidget(self._toc_delay_apply_btn)

        # Connect slider controls
        self._toc_delay_slider.valueChanged.connect(self._on_toc_delay_slider_changed)
        self._toc_delay_spin.valueChanged.connect(self._on_toc_delay_spin_changed)
        self._toc_delay_reset_btn.clicked.connect(self._on_toc_delay_reset)
        self._toc_delay_apply_btn.clicked.connect(self._on_toc_delay_apply)

        # Cache for alignment data (populated by _update_toc_alignment)
        self._bp_align_cache = None
        self._bp_delay_original = None  # delay from MasterFile
        self._bp_delay_current = None

        # Chart (primer) + slider (a sota) + taula (final)
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
            from matplotlib.figure import Figure
            self._toc_align_figure = Figure(figsize=(10, 4), dpi=100)
            self._toc_align_figure.set_facecolor('#FAFAFA')
            self._toc_align_canvas = FigureCanvas(self._toc_align_figure)
            self._toc_align_canvas.setMinimumHeight(300)
            self._toc_align_toolbar = NavigationToolbar2QT(self._toc_align_canvas,
                                                            self._toc_align_group)
            toc_align_layout.addWidget(self._toc_align_toolbar)
            toc_align_layout.addWidget(self._toc_align_canvas)
            self._has_toc_align_chart = True
        except ImportError:
            self._has_toc_align_chart = False

        # Slider a sota del gràfic (compacte, mateixa amplada)
        toc_align_layout.addLayout(slider_row)

        # Delay per injection table
        self._toc_align_table = QLabel()
        self._toc_align_table.setStyleSheet("font-size: 10px; font-family: monospace; padding: 4px;")
        self._toc_align_table.setWordWrap(True)
        toc_align_layout.addWidget(self._toc_align_table)

        content_layout.addWidget(self._toc_align_group)

        # === CALIBRATION LINE (primer — referència visual principal) ===
        self.cal_line_group = QGroupBox("Recta de calibració")
        self.cal_line_group.setVisible(False)
        self.cal_line_group.setStyleSheet(
            "QGroupBox { font-weight: bold; color: #1D4ED8; border: 2px solid #2563EB; "
            "border-radius: 6px; margin-top: 8px; padding-top: 12px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }"
        )
        cal_line_layout = QVBoxLayout(self.cal_line_group)
        self.prominent_cal_line_graph = CalibrationLineWidget()
        cal_line_layout.addWidget(self.prominent_cal_line_graph)
        content_layout.addWidget(self.cal_line_group)

        # === CHROMATOGRAMS ===
        self.graphs_group = QGroupBox("Cromatogrames KHP")
        self.graphs_group.setVisible(False)
        graphs_layout = QVBoxLayout(self.graphs_group)
        self.replica_graphs = KHPReplicaGraphWidget()
        self.replica_graphs.baseline_adjusted.connect(self._on_baseline_adjusted)
        graphs_layout.addWidget(self.replica_graphs)
        content_layout.addWidget(self.graphs_group)

        # === METRICS TABLE ===
        self.metrics_group = QGroupBox("Rèpliques KHP")
        self.metrics_group.setVisible(False)
        metrics_layout = QVBoxLayout(self.metrics_group)

        # Barra d'accions — coherent amb el pas Analitzar (botons visibles,
        # mateix diàleg de reparació). Evita el doble-clic amagat.
        metrics_actions = QHBoxLayout()
        metrics_actions.setSpacing(6)
        _sel_hint = QLabel("Selecciona una rèplica:")
        _sel_hint.setStyleSheet("color:#666; font-size:11px;")
        metrics_actions.addWidget(_sel_hint)
        self.metrics_repair_btn = QPushButton("Reparar pic")
        self.metrics_repair_btn.setStyleSheet(
            "QPushButton { border: 1px solid #E67E22; border-radius: 3px;"
            " padding: 4px 10px; font-size: 11px; color: #E67E22; }"
            "QPushButton:hover { background: #FEF9E7; }")
        self.metrics_repair_btn.setToolTip(
            "Reparar el cim irregular (batman) del pic KHP seleccionat — "
            "mateix diàleg que al pas Analitzar")
        self.metrics_repair_btn.clicked.connect(self._on_calib_repair_clicked)
        metrics_actions.addWidget(self.metrics_repair_btn)
        self.metrics_detail_btn = QPushButton("Detall")
        self.metrics_detail_btn.setStyleSheet(
            "QPushButton { border: 1px solid #CED4DA; border-radius: 3px;"
            " padding: 4px 10px; font-size: 11px; }"
            "QPushButton:hover { background: #E9ECEF; }")
        self.metrics_detail_btn.setToolTip("Obrir el detall complet de la rèplica seleccionada")
        self.metrics_detail_btn.clicked.connect(self._on_calib_detail_clicked)
        metrics_actions.addWidget(self.metrics_detail_btn)
        metrics_actions.addStretch()
        metrics_layout.addLayout(metrics_actions)

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
        # Doble-click sobre fila → diàleg detall + reparació
        self.metrics_table.cellDoubleClicked.connect(self._on_metrics_row_double_clicked)

        # Filtre: mostrar només rèpliques amb avisos/errors (Estat ⚠ o ✘)
        self._only_issues_cb = QCheckBox("Només rèpliques amb avisos")
        self._only_issues_cb.setStyleSheet("font-size: 11px; padding: 2px;")
        self._only_issues_cb.toggled.connect(lambda _: self._apply_issue_filter())
        metrics_layout.addWidget(self._only_issues_cb)

        metrics_layout.addWidget(self.metrics_table)

        content_layout.addWidget(self.metrics_group)

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
            QPushButton:hover { background: #3498DB; color: white; border-color: #2563EB; }
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
        # Detectar mode siblings
        sibling_imported = getattr(self.main_window, 'sibling_imported', {})
        has_siblings = len(sibling_imported) > 1

        if has_siblings:
            self._run_calibrate_siblings(sibling_imported)
            return

        imported_data = self.main_window.imported_data

        if not imported_data:
            seq_path = self.main_window.seq_path
            if seq_path:
                from hpsec_import import import_from_manifest
                self.main_window.set_status("Carregant manifest…")
                # v2.2.0+: load_data=False — només metadades (instantani).
                # CalibrateWorker farà ensure_data_loaded() al thread quan
                # detecti data_deferred=True. Abans aquí es llegia
                # MasterFile + CSV + Export3D al UI thread, bloquejant ~10s.
                imported_data = import_from_manifest(seq_path, load_data=False)
                if imported_data and imported_data.get('success'):
                    self.main_window.imported_data = imported_data
                    self.main_window.set_status("Manifest carregat", 1000)

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
        self._hide_all_sections()

        if self.worker is not None:
            self.worker.wait()
        self.worker = CalibrateWorker(imported_data)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _hide_all_sections(self):
        """Amaga totes les seccions de resultat."""
        self.compact_header.setVisible(False)
        self.delay_group.setVisible(False)
        self.cal_line_group.setVisible(False)
        self.graphs_group.setVisible(False)
        self.metrics_group.setVisible(False)
        self.history_group.setVisible(False)

    def _run_calibrate_siblings(self, sibling_imported):
        """Calibra N siblings independentment."""
        import os
        n = len(sibling_imported)
        names = [os.path.basename(p) for p in sibling_imported]
        logger.info("Verificació siblings: %d carpetes (%s)", n, ", ".join(names))

        self.calibrate_btn.setEnabled(False)
        self.main_window.show_progress(0)
        self._hide_all_sections()
        self._clear_sibling_cards()
        self._sibling_results = {}

        if self._sibling_worker is not None:
            self._sibling_worker.wait()

        self._sibling_worker = SiblingCalibrateWorker(sibling_imported)
        self._sibling_worker.progress.connect(self._on_progress)
        self._sibling_worker.sibling_finished.connect(self._on_sibling_cal_finished)
        self._sibling_worker.all_finished.connect(self._on_all_siblings_cal_finished)
        self._sibling_worker.error.connect(self._on_error)
        self._sibling_worker.start()

    def _on_sibling_cal_finished(self, path, result):
        """Callback per cada sibling calibrat."""
        import os
        name = os.path.basename(path)
        self._sibling_results[path] = result
        self.main_window.sibling_calibrated[path] = result
        ok = result.get("success", False)
        logger.info("Sibling calibrat %s: %s", name, "OK" if ok else "ERROR/SENSE_KHP")

    def _on_all_siblings_cal_finished(self, results):
        """Callback quan tots els siblings han estat calibrats."""
        import os
        self.main_window.show_progress(-1)
        self.calibrate_btn.setEnabled(True)
        self.placeholder.setVisible(False)

        if self._sibling_worker is not None:
            self._sibling_worker.wait()

        self._sibling_results = results

        # Construir cards per cada sibling
        self._build_sibling_cards(results)

        # Usar primari com a calibration_data principal (backward compat)
        primary_path = self.main_window.seq_path
        if primary_path in results:
            primary_result = results[primary_path]
        else:
            primary_result = next(iter(results.values()))

        self.calibration_data = primary_result
        self.main_window.calibration_data = primary_result

        # Mostrar detall del primari (o del primer amb KHP vàlid)
        first_ok = next(
            (p for p, r in results.items() if r.get("success")),
            primary_path
        )
        self._show_sibling_detail(first_ok)

        self.main_window.enable_tab(2)
        n = len(results)
        n_ok = sum(1 for r in results.values() if r.get("success"))
        self.main_window.set_status(
            f"Verificació completada: {n_ok}/{n} carpetes amb KHP", 5000
        )

        # Emetre senyal amb resultat combinat
        self.calibration_completed.emit(primary_result)

    def _clear_sibling_cards(self):
        """Elimina tots els cards de siblings del layout."""
        for card in self._sibling_cards:
            card.setParent(None)
            card.deleteLater()
        self._sibling_cards = []

    def _build_sibling_cards(self, results):
        """Construeix un card resum per cada sibling."""
        import os

        self._clear_sibling_cards()

        for path, result in results.items():
            name = os.path.basename(path)
            ok = result.get("success", False)

            # Extreure mètriques clau
            khp_direct = result.get("khp_data_direct", {})
            khp_uib = result.get("khp_data_uib", {})
            shift_d = result.get("shift_direct", 0)
            shift_u = result.get("shift_uib", 0)

            # RF
            rf_parts = []
            if khp_direct:
                reps = khp_direct.get("replicas", [khp_direct])
                rf_vals = [r.get("rf_mass_doc", 0) for r in reps if r.get("rf_mass_doc", 0) > 0]
                if rf_vals:
                    rf_parts.append(f"RF_D={np.mean(rf_vals):.0f}")
            if khp_uib:
                reps = khp_uib.get("replicas", [khp_uib])
                rf_vals = [r.get("rf_mass_doc", 0) for r in reps if r.get("rf_mass_doc", 0) > 0]
                if rf_vals:
                    rf_parts.append(f"RF_U={np.mean(rf_vals):.0f}")

            # Shift
            shift_parts = []
            if shift_d != 0:
                shift_parts.append(f"D:{shift_d*60:+.1f}s")
            if shift_u != 0:
                shift_parts.append(f"U:{shift_u*60:+.1f}s")

            # Anomalies
            anomalies = result.get("warnings_structured", [])
            n_anom = len(anomalies)

            # Construir card
            card = QFrame()
            card.setCursor(Qt.PointingHandCursor)
            card.setProperty("sibling_path", path)

            if ok:
                border_color = "#27AE60" if n_anom == 0 else "#F39C12"
                icon = "✓" if n_anom == 0 else "⚠"
            else:
                border_color = "#E74C3C"
                icon = "✗"

            is_active = (path == self._active_sibling_path)
            bg_color = "#DBEAFE" if is_active else "#FAFAFA"

            card.setStyleSheet(
                f"QFrame {{ background: {bg_color}; border: 2px solid {border_color}; "
                f"border-radius: 6px; padding: 8px; }}"
                f"QFrame:hover {{ background: #E8F6F3; }}"
            )

            card_layout = QHBoxLayout(card)
            card_layout.setContentsMargins(8, 4, 8, 4)
            card_layout.setSpacing(12)

            # Icona + nom
            label_name = QLabel(f"<b>{icon} {name}</b>")
            label_name.setStyleSheet("font-size: 13px; border: none;")
            card_layout.addWidget(label_name)

            # Mètriques
            parts = []
            if rf_parts:
                parts.append(" · ".join(rf_parts))
            if shift_parts:
                parts.append(f"Shift: {' '.join(shift_parts)}")
            if n_anom > 0:
                parts.append(f"{n_anom} anomalies")
            if not ok:
                errors = result.get("errors", [])
                if errors:
                    short_err = errors[0][:60]
                    parts.append(short_err)

            if parts:
                label_info = QLabel(" · ".join(parts))
                label_info.setStyleSheet("color: #555; font-size: 11px; border: none;")
                card_layout.addWidget(label_info)

            card_layout.addStretch()

            # Botó "Detall"
            btn = QPushButton("Detall ▸")
            btn.setStyleSheet(
                "QPushButton { padding: 2px 8px; border: 1px solid #ccc; "
                "border-radius: 3px; font-size: 11px; background: white; }"
                "QPushButton:hover { background: #2563EB; color: white; }"
            )
            btn.setFixedHeight(22)
            _path = path  # captura per closure
            btn.clicked.connect(lambda checked, p=_path: self._show_sibling_detail(p))
            card_layout.addWidget(btn)

            self._sibling_cards_layout.addWidget(card)
            self._sibling_cards.append(card)

    def _show_sibling_detail(self, path):
        """Mostra el detall complet d'un sibling als widgets existents."""
        import os

        result = self._sibling_results.get(path)
        if not result:
            return

        self._active_sibling_path = path
        name = os.path.basename(path)

        # Actualitzar highlight dels cards
        for card in self._sibling_cards:
            card_path = card.property("sibling_path")
            is_active = (card_path == path)
            bg = "#DBEAFE" if is_active else "#FAFAFA"
            # Re-aplicar estil mantenint border original
            style = card.styleSheet()
            if "background:" in style:
                import re
                style = re.sub(r'background:\s*#[A-Fa-f0-9]+', f'background: {bg}', style)
                card.setStyleSheet(style)

        # Copiar rf_mass_direct/uib (mateixa lògica que _on_finished)
        for signal_key, data_key in [("direct", "khp_data_direct"), ("uib", "khp_data_uib")]:
            khp_data = result.get(data_key)
            if khp_data:
                replicas = khp_data.get("replicas", [khp_data])
                rf_vals = [r.get("rf_mass_doc", 0) for r in replicas if r.get("rf_mass_doc", 0) > 0]
                if not rf_vals:
                    rf_vals = [r.get("rf_mass", 0) for r in replicas if r.get("rf_mass", 0) > 0]
                if rf_vals:
                    result[f"rf_mass_{signal_key}"] = float(np.mean(rf_vals))

        # Actualitzar calibration_data i main_window per al sibling seleccionat
        self.calibration_data = result
        self.main_window.calibration_data = result
        # Temporalment canviar seq_path per al delay diagnostic
        orig_seq_path = self.main_window.seq_path
        self.main_window.seq_path = path

        # Actualitzar tots els widgets amb el resultat del sibling
        for fn in [self._update_compact_header, self._update_delay_diagnostic,
                   self._update_graphs, self._update_metrics_table,
                   self._update_validation, self._update_history]:
            try:
                fn(result)
            except Exception as e:
                logger.warning(f"Error a {fn.__name__} per sibling {name}: {e}")

        # Restaurar seq_path
        self.main_window.seq_path = orig_seq_path

        # Afegir nom sibling al header
        if self.compact_header.isVisible():
            current = self.compact_header.text()
            if f"[{name}]" not in current:
                self.compact_header.setText(
                    f'<span style="color: #2563EB; font-size: 11px;">[{name}]</span> {current}'
                )

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
        # (Notes gestionades pel wizard header)
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
        # (Notes gestionades pel wizard header)

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

    def _all_cal_replicas(self, cals):
        """Aplana TOTES les rèpliques de totes les calibracions (una per concentració).
        Cada entrada de `cals` (calibrations_direct/uib) porta la seva llista 'replicas'."""
        if not cals:
            return []
        out = []
        for cal in cals:
            for rep in (cal.get('replicas') or []):
                if isinstance(rep, dict):
                    out.append(rep)
        return out

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

    def _on_baseline_adjusted(self, replica_idx, new_baseline):
        """Recalcula àrea KHP amb la nova baseline i refresca gràfics i mètriques."""
        import numpy as np
        from hpsec_core import find_peak_boundaries

        # Accedir a les dades de la rèplica
        replicas = self.replica_graphs._replicas_data
        if replica_idx >= len(replicas):
            return

        rep = replicas[replica_idx]
        t = np.asarray(rep.get('t_doc', []), dtype=float)
        y_doc = np.asarray(rep.get('y_doc', []), dtype=float)
        old_bl = rep.get('baseline_level', 0)

        if len(t) < 10:
            return

        # Recalcular y_net amb nova baseline
        # y_doc és net (ja restada baseline anterior), recuperar raw
        y_raw = y_doc + old_bl
        y_net_new = y_raw - new_baseline

        # Recalcular àrea amb trapezoid + marge
        pk = int(np.argmax(y_net_new))
        try:
            li, ri = find_peak_boundaries(t, y_net_new, pk)
        except Exception:
            li, ri = 0, len(t) - 1
        width = ri - li
        margin = max(3, int(width * 0.2))
        li_w = max(0, li - margin)
        ri_w = min(len(t) - 1, ri + margin)
        new_area = float(np.trapezoid(np.maximum(y_net_new[li_w:ri_w+1], 0), t[li_w:ri_w+1]))

        # Actualitzar dades de la rèplica
        rep['y_doc'] = y_net_new.tolist()
        rep['baseline_level'] = new_baseline
        rep['area'] = new_area
        rep['peak_left_idx'] = li
        rep['peak_right_idx'] = ri
        rep['_baseline_manual'] = True

        logger.info(f"Baseline ajustada replica {replica_idx}: "
                    f"bl {old_bl:.1f} -> {new_baseline:.1f}, area {new_area:.1f}")

        # Refrescar gràfics
        if hasattr(self, 'calibration_data') and self.calibration_data:
            self._update_graphs(self.calibration_data)

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

        # Direct replicas — TOTES les concentracions (de calibrations_direct),
        # no només la primària. Fallback al primari si no hi ha llista per condició.
        direct_list = (self._all_cal_replicas(result.get("calibrations_direct"))
                       or self._extract_all_replicas(khp_data_direct))
        for d in direct_list:
            d_copy = d.copy()
            d_copy['_signal'] = 'Direct'
            all_data.append(d_copy)
            if d.get('has_timeout'):
                fname = d.get('filename', '')
                match = _re.search(r'R(\d+)', fname)
                rep_num = match.group(1) if match else '1'
                direct_timeouts[(d.get('conc_ppm'), rep_num)] = d.get('timeout_info', {})

        # UIB replicas (propagant timeouts) — també totes les concentracions
        uib_list = (self._all_cal_replicas(result.get("calibrations_uib"))
                    or self._extract_all_replicas(khp_data_uib))
        for d in uib_list:
            d_copy = d.copy()
            d_copy['_signal'] = 'UIB'
            fname = d.get('filename', '')
            match = _re.search(r'R(\d+)', fname)
            rep_num = match.group(1) if match else '1'
            _tkey = (d.get('conc_ppm'), rep_num)
            if not d_copy.get('has_timeout') and _tkey in direct_timeouts:
                d_copy['has_timeout'] = True
                d_copy['timeout_info'] = direct_timeouts[_tkey]
                d_copy['_timeout_propagated'] = True
            all_data.append(d_copy)

        # Lookup àrees companion: Direct→UIB, UIB→Direct
        uib_area_by_rep = {}
        direct_area_by_rep = {}
        for d in uib_list:
            fname = d.get('filename', '')
            match = _re.search(r'R(\d+)', fname)
            rn = match.group(1) if match else '1'
            uib_area_by_rep[(d.get('conc_ppm'), rn)] = d.get('area', 0)
        for d in direct_list:
            fname = d.get('filename', '')
            match = _re.search(r'R(\d+)', fname)
            rn = match.group(1) if match else '1'
            direct_area_by_rep[(d.get('conc_ppm'), rn)] = d.get('area', 0)

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
            # Incloure la concentració perquè es distingeixin les rèpliques de cada punt
            _conc = khp.get('conc_ppm')
            if _conc:
                display_name = f"{_conc:g}ppm {display_name}"
            item_rep = QTableWidgetItem(display_name)
            # Guardar referència al dict khp per doble-click
            item_rep.setData(Qt.UserRole, khp)
            self.metrics_table.setItem(row, 0, item_rep)

            # Col 1: Senyal
            self.metrics_table.setItem(row, 1, QTableWidgetItem(signal))

            # Col 2: Àrea
            area = khp.get('area', 0)
            self.metrics_table.setItem(row, 2, QTableWidgetItem(f"{area:.0f}"))

            # Col 3: Comp. (companion area)
            match = _re.search(r'R(\d+)', filename)
            rep_key = match.group(1) if match else '1'
            _ckey = (khp.get('conc_ppm'), rep_key)
            if signal == 'Direct':
                a_uib = uib_area_by_rep.get(_ckey, 0)
                self.metrics_table.setItem(row, 3, QTableWidgetItem(f"{a_uib:.0f}" if a_uib > 0 else "-"))
            else:
                a_direct = direct_area_by_rep.get(_ckey, 0)
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
                item_fwhm.setBackground(QColor(COLOR_WARNING_LIGHT))
                item_fwhm.setToolTip(f"FWHM elevat (>{FWHM_THRESHOLD} min)")
            self.metrics_table.setItem(row, 7, item_fwhm)

            # Col 8: SNR
            snr = khp.get('snr', 0)
            item_snr = QTableWidgetItem(f"{snr:.0f}" if snr > 0 else "-")
            if 0 < snr < 10:
                item_snr.setBackground(QColor(COLOR_WARNING_LIGHT))
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
                    item_r2.setBackground(QColor(COLOR_SUCCESS_LIGHT))
                elif bg_status == 'CHECK':
                    item_r2.setBackground(QColor(COLOR_WARNING_LIGHT))
                else:
                    item_r2.setBackground(QColor(COLOR_WARNING_LIGHT))
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
                item_status.setBackground(QColor("#E2E8F0"))
                item_status.setToolTip("Sense info QA/QC \u2014 reimportar per obtenir-la")
            elif not raw_anomalies:
                # calibration_anomalies existeix i és buit → tot OK
                item_status = QTableWidgetItem("\u2714")
                item_status.setBackground(QColor(COLOR_SUCCESS_LIGHT))
                item_status.setToolTip("Sense anomalies")
            else:
                classified = classify_anomalies(cal_anomalies)
                has_blockers = len(classified["blocker"]) > 0
                has_warnings = len(classified["warning"]) > 0

                if has_blockers:
                    status_text = "\u2718"
                    color = QColor(COLOR_ERROR_LIGHT)
                elif has_warnings:
                    status_text = "\u26a0"
                    color = QColor(COLOR_WARNING_LIGHT)
                elif cal_anomalies:
                    status_text = "\u2139"
                    color = QColor(COLOR_WARNING_LIGHT)
                else:
                    status_text = "\u2714"
                    color = QColor(COLOR_SUCCESS_LIGHT)

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

            # Marcar la fila com "amb avisos" per al filtre (Estat ⚠ o ✘)
            _row_flagged = item_status.text() in ("⚠", "✘")
            item_rep.setData(Qt.UserRole + 7, _row_flagged)

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
            _cb_conc = khp.get('conc_ppm')
            cb.stateChanged.connect(
                lambda state, rn=replica_num, st=signal_type, cc=_cb_conc:
                    self._on_metrics_outlier_toggled(rn, st, state, cc)
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
                    item_empty.setData(Qt.UserRole + 7, True)  # sub-fila d'anomalia
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

        self._apply_issue_filter()

    # --- Filtre "només rèpliques amb avisos" -----------------------------
    def _apply_issue_filter(self):
        """Amaga/mostra files segons el marcatge d'avisos (Qt.UserRole+7 a col 0)."""
        table = getattr(self, "metrics_table", None)
        if table is None:
            return
        on = getattr(self, "_only_issues_cb", None) is not None and self._only_issues_cb.isChecked()
        for row in range(table.rowCount()):
            if not on:
                table.setRowHidden(row, False)
                continue
            item0 = table.item(row, 0)
            flagged = bool(item0.data(Qt.UserRole + 7)) if item0 is not None else False
            table.setRowHidden(row, not flagged)

    def set_flagged_samples(self, names):
        """API pel wizard (el Verificar marca l'estat per rèplica; sense ús directe)."""
        self._flagged_samples = set(names or [])

    def toggle_issue_filter(self):
        """API pel wizard: activa/desactiva el filtre des del header."""
        if getattr(self, "_only_issues_cb", None) is not None:
            self._only_issues_cb.setChecked(not self._only_issues_cb.isChecked())

    def _on_metrics_row_double_clicked(self, row: int, _col: int):
        """Obre el diàleg de detall + reparació quan es fa doble-click a una fila."""
        # Recuperar khp guardat al col 0
        item_rep = self.metrics_table.item(row, 0)
        if item_rep is None:
            return
        khp = item_rep.data(Qt.UserRole)
        if not isinstance(khp, dict):
            return
        # Doble-clic = "Detall" (coherent amb Analitzar). La reparació té botó propi.
        self._open_khp_detail(khp)

    def _selected_metrics_khp(self):
        """Retorna el dict khp de la fila seleccionada a la taula (o None)."""
        row = self.metrics_table.currentRow()
        if row < 0:
            return None
        item = self.metrics_table.item(row, 0)
        khp = item.data(Qt.UserRole) if item is not None else None
        return khp if isinstance(khp, dict) else None

    def _on_calib_detail_clicked(self):
        """Botó 'Detall' — obre el detall de la rèplica seleccionada."""
        khp = self._selected_metrics_khp()
        if not khp:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(self, "Detall", "Selecciona una rèplica de la taula.")
            return
        self._open_khp_detail(khp)

    def _open_khp_detail(self, khp: dict):
        """Obre el diàleg de detall KHP per a una rèplica."""
        from .khp_detail_dialog import KHPDetailDialog
        name, replica, signal = self._khp_repair_identity(khp)
        # Identitat del KHP obert (modal) — desambigua la clau d'override per concentració
        self._repair_identity = (name, replica, signal)
        self._detail_conc = khp.get('conc_ppm')
        try:
            khp['replica_num'] = int(replica)
        except (TypeError, ValueError):
            pass
        has_manual = False
        seq_path = self.main_window.seq_path
        if seq_path and name:
            repairs = load_manual_repairs(seq_path)
            has_manual = manual_repair_key(name, replica, signal) in repairs
        dialog = KHPDetailDialog(khp, signal=signal, parent=self,
                                 has_manual_repair=has_manual)
        dialog.outlier_toggled.connect(self._on_detail_outlier_toggled)
        dialog.repair_applied.connect(self._on_detail_repair_applied)
        dialog.repair_undone.connect(self._on_detail_repair_undone)
        dialog.exec()

    def _build_repair_adapter(self, conc, signal):
        """Adapta les rèpliques KHP (concentració + senyal) a la forma que espera
        JaggedPeakRepairDialog (replicas amb t_doc + y_doc_net/y_doc_uib_net).

        Llegeix de calibrations_direct/uib (TOTES les concentracions, cada entrada
        porta la seva llista 'replicas'), no del primari khp_data_direct."""
        import re as _re
        res = self.calibration_data or {}
        cals_key = 'calibrations_direct' if signal == 'direct' else 'calibrations_uib'
        y_key = 'y_doc_net' if signal == 'direct' else 'y_doc_uib_net'
        replicas = {}
        is_bp = False
        for cal in (res.get(cals_key) or []):
            if abs((cal.get('conc_ppm') or 0) - conc) > 1e-6:
                continue
            for d in cal.get('replicas', []):
                t = d.get('t_doc')
                y = d.get('y_doc')
                if t is None or y is None or len(t) == 0:
                    continue
                m = _re.search(r'R(\d+)', d.get('filename', '') or '')
                rk = m.group(1) if m else str(d.get('replica_num', 1))
                is_bp = bool(d.get('is_bp', is_bp))
                replicas[rk] = {
                    't_doc': np.asarray(t, dtype=float),
                    y_key: np.asarray(y, dtype=float),
                    'is_bp': bool(d.get('is_bp', False)),
                    'anomalies': [],
                }
        return {'replicas': replicas}, is_bp

    def _on_calib_repair_clicked(self):
        """Botó 'Reparar pic' — obre el MATEIX diàleg que a Analitzar (coherència)
        i desa el resultat com a override persistent reversible."""
        from PySide6.QtWidgets import QMessageBox
        khp = self._selected_metrics_khp()
        if not khp:
            QMessageBox.information(self, "Reparar pic", "Selecciona una rèplica de la taula.")
            return
        seq_path = self.main_window.seq_path
        if not seq_path or not self.calibration_data:
            return
        name, _replica, signal = self._khp_repair_identity(khp)
        conc = khp.get('conc_ppm', 0)
        adapter, is_bp = self._build_repair_adapter(conc, signal)
        if not adapter['replicas']:
            QMessageBox.information(
                self, "Reparar pic",
                "No hi ha dades de cromatograma per reparar.\n"
                "Pot caldre reprocessar (Verificar).")
            return
        try:
            from ..analyze_panel.repair_dialog import JaggedPeakRepairDialog
        except Exception as e:
            QMessageBox.warning(self, "Reparar pic", f"No s'ha pogut obrir el diàleg: {e}")
            return

        method = "BP" if is_bp else "COLUMN"
        title = f"{name} {signal.upper()} ({conc:g} ppm)"
        dialog = JaggedPeakRepairDialog(title, adapter, method, force=True, parent=self)
        dialog.exec()

        # Sincronitzar overrides amb l'estat final dels cards (aplicats / desfets)
        existing = load_manual_repairs(seq_path)
        changed = False
        for card in getattr(dialog, '_cards', []):
            rk = getattr(card, 'rep_key', None)
            sig = getattr(card, 'signal_type', signal)
            if rk is None:
                continue
            key = manual_repair_key(name, rk, sig)
            if getattr(card, 'state', '') == 'repaired':
                set_manual_repair(seq_path, name, rk, sig,
                                  card._anchor_left_spin.value(),
                                  card._anchor_right_spin.value(),
                                  getattr(dialog, '_factor', None))
                changed = True
            elif key in existing:
                remove_manual_repair(seq_path, name, rk, sig)
                changed = True
        if changed:
            self.main_window.set_status("Reparació desada — recalculant calibració…", 3000)
            self._run_calibrate()

    def _khp_repair_identity(self, khp: dict):
        """Retorna (name, replica, signal) per a la clau de reparació manual.

        name = nom del KHP, replica = número de rèplica (str), signal = direct/uib.
        Consistent amb manual_repair_key() usat a calibrate_from_import.
        """
        import re as _re
        filename = khp.get('filename', '') or ''
        name = khp.get('name') or (filename.split('_R')[0] if '_R' in filename else filename)
        m = _re.search(r'R(\d+)', filename)
        replica = m.group(1) if m else str(khp.get('replica_num', 1))
        signal = (khp.get('_signal') or khp.get('doc_source') or 'direct').lower()
        if signal not in ('direct', 'uib'):
            signal = 'direct'
        return name, replica, signal

    def _on_detail_outlier_toggled(self, replica_num: int, signal: str, is_outlier: bool):
        """Sincronitza el checkbox de la taula amb el toggle del diàleg."""
        state = 2 if is_outlier else 0  # Qt.Checked = 2, Qt.Unchecked = 0
        self._on_metrics_outlier_toggled(replica_num, signal, state,
                                         getattr(self, '_detail_conc', None))

    def _on_detail_repair_applied(self, replica_num: int, signal: str, repaired_data: dict):
        """Desa la reparació manual (ancoratges) com a override persistent i recalcula.

        L'override es reaplica de forma determinista a calibrate_from_import, de manera
        que el camí en viu i el persistit són idèntics. Reversible amb desfer.
        """
        from PySide6.QtWidgets import QMessageBox
        try:
            seq_path = self.main_window.seq_path
            identity = getattr(self, '_repair_identity', None)
            if not seq_path or not identity:
                return
            name, replica, sig = identity
            new_area = float(repaired_data.get('new_area', 0))

            ok = set_manual_repair(
                seq_path, name, replica, sig,
                repaired_data.get('anchor_left_t'),
                repaired_data.get('anchor_right_t'))
            if not ok:
                QMessageBox.warning(self, "Reparació",
                                    "No s'ha pogut guardar la reparació manual.")
                return

            QMessageBox.information(
                self, "Reparació guardada",
                f"Reparació manual de {name} R{replica} desada "
                f"(àrea ≈ {new_area:.2f}).\nRecalculant la calibració…")
            # Re-executar la calibració: l'override s'aplica i la recta s'actualitza
            self._run_calibrate()
        except Exception as exc:
            QMessageBox.warning(self, "Reparació",
                                f"No s'ha pogut guardar la reparació: {exc}")

    def _on_detail_repair_undone(self, replica_num: int, signal: str):
        """Esborra la reparació manual desada (desfer) i recalcula."""
        from PySide6.QtWidgets import QMessageBox
        try:
            seq_path = self.main_window.seq_path
            identity = getattr(self, '_repair_identity', None)
            if not seq_path or not identity:
                return
            name, replica, sig = identity
            removed = remove_manual_repair(seq_path, name, replica, sig)
            if removed:
                self.main_window.set_status(
                    f"Reparació manual de {name} R{replica} desfeta", 3000)
                self._run_calibrate()
            else:
                QMessageBox.information(self, "Desfer",
                                        "No hi havia cap reparació manual desada.")
        except Exception as exc:
            QMessageBox.warning(self, "Desfer",
                                f"No s'ha pogut desfer la reparació: {exc}")

    def _on_metrics_outlier_toggled(self, replica_num, signal_type, state, conc=None):
        """Handler quan canvia el checkbox outlier a la taula de mètriques.

        conc: concentració del punt (per no afectar la mateixa rèplica d'altres
        concentracions quan la taula mostra tota la recta)."""
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
                # Només la calibració d'aquesta concentració (si s'ha indicat)
                if conc is not None and abs((cal.get('conc_ppm') or 0) - conc) > 1e-6:
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
            "QGroupBox { font-weight: bold; color: #1D4ED8; border: 2px solid #E67E22; "
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

    def _update_toc_alignment(self, result):
        """Show TOC+DAD254 alignment for BP sequences with interactive slider."""
        imported = self.main_window.imported_data
        if not imported:
            self._toc_align_group.setVisible(False)
            return

        method = imported.get("method", "COLUMN")
        if method != "BP":
            self._toc_align_group.setVisible(False)
            return

        align = imported.get("bp_alignment")
        if not align:
            self._toc_align_group.setVisible(False)
            return

        self._toc_align_group.setVisible(True)

        # Info
        n_inj = align.get("n_injections", 0)
        n_dad = align.get("n_with_dad", 0)
        n_matched = align.get("n_matched", 0)
        delay_med = align.get("delay_median", 0)
        delay_min_val = align.get("delay_min", 0)
        delay_max_val = align.get("delay_max", 0)
        drift = align.get("delay_drift", 0)
        n_timeouts = len(align.get("toc_timeouts", []))

        delay_blocks = align.get('delay_blocks', [])
        n_blocks = len(delay_blocks)

        info_parts = [
            f"<b>{n_inj}</b> injeccions, <b>{n_dad}</b> amb DAD 254, "
            f"<b>{n_matched}</b> delays, "
            f"<b>{n_timeouts}</b> timeouts, <b>{n_blocks}</b> blocs",
            f"Delay mediana: <b>{delay_med:.1f}</b> min "
            f"(rang {delay_min_val:.1f}–{delay_max_val:.1f})",
        ]
        if delay_blocks:
            blk_info = " | ".join(
                f"B{i+1}: {db['delay_median']:.1f}±{db['delay_std']:.2f} ({db['n_inj']}inj)"
                for i, db in enumerate(delay_blocks)
            )
            info_parts.append(f"<span style='font-size:10px'>Blocs: {blk_info}</span>")
        self._toc_align_info.setText("<br>".join(info_parts))

        # Cache alignment data for slider interaction
        import numpy as np
        toc_data = align.get("toc_continuous", {})
        dad_data = align.get("dad254_continuous", {})
        t_toc = np.array(toc_data.get("t", []))
        y_toc = np.array(toc_data.get("y", []))
        t_dad = np.array(dad_data.get("t", []))
        y_dad = np.array(dad_data.get("y", []))

        # Read MasterFile delay for reference
        mf_delay = None
        imported_data = self.main_window.imported_data or {}
        mf_path = imported_data.get("master_file")
        if not mf_path:
            seq_path = self.main_window.seq_path
            if seq_path:
                candidates = list(Path(seq_path).glob("*MasterFile*.xlsx"))
                candidates = [c for c in candidates if 'backup' not in c.name.lower()]
                if candidates:
                    mf_path = str(candidates[0])
        if mf_path:
            try:
                from hpsec_delay import read_current_delay
                mf_delay = read_current_delay(mf_path)
            except Exception:
                pass

        # Build per-injection info sorted by HPLC time
        per_inj = align.get("per_injection", [])
        inj_list = []
        for p in per_inj:
            inj_list.append({
                'name': p.get('name', '?'),
                't_hplc': p.get('t_hplc', 0),
                'is_control': p.get('is_control', False),
                'delay_real': p.get('delay_real'),
                'y_dad_peak': p.get('y_dad_peak', 0),
                't_dad_peak_rel': p.get('t_dad_peak_rel'),
            })
        inj_list.sort(key=lambda x: x['t_hplc'])

        delay_blocks = align.get('delay_blocks', [])

        self._bp_align_cache = {
            't_toc': t_toc,
            'y_toc': y_toc,
            't_dad': t_dad,
            'y_dad': y_dad,
            'per_injection': inj_list,
            'toc_timeouts': align.get('toc_timeouts', []),
            'delay_blocks': delay_blocks,
            'cadence': align.get('cadence', 11.0),
            'mf_path': mf_path,
            'n_matched': n_matched,
            'drift': drift,
        }
        self._bp_delay_original = mf_delay if mf_delay is not None else delay_med
        self._bp_delay_current = 0.0  # offset addicional (0 = alineació automàtica per bloc)
        self._bp_align_cache['_base_delay'] = 0.0

        # Slider com a offset (0 = alineació automàtica ok, ±X per ajustar)
        self._toc_delay_spin.blockSignals(True)
        self._toc_delay_slider.blockSignals(True)
        self._toc_delay_spin.setRange(-5.0, 5.0)
        self._toc_delay_spin.setValue(0.0)
        self._toc_delay_spin.setSuffix(" min offset")
        self._toc_delay_slider.setRange(-50, 50)  # ±5 min
        self._toc_delay_slider.setValue(0)
        self._toc_delay_spin.blockSignals(False)
        self._toc_delay_slider.blockSignals(False)

        # Apply button hidden unless offset ≠ 0
        self._toc_delay_apply_btn.setVisible(False)
        self._update_toc_delay_impact()

        # Draw full chart
        self._draw_toc_alignment_chart()

        # Delay table (compact)
        lines = []
        for p in inj_list:
            if p['is_control']:
                continue
            name = p['name'][:12]
            d = p['delay_real']
            d_str = f"{d:.1f}" if d is not None else "  -"
            lines.append(f"{name:>12}  {d_str:>4}")

        if lines:
            # 3 columnes per aprofitar espai
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
            self._toc_align_table.setText("\n".join(table_lines))

    def _draw_toc_alignment_chart(self):
        """Draw TOC+DAD chart with per-block delay alignment.

        Mode: per-bloc (defecte) — cada segment TOC entre timeouts es desplaça
        pel delay del seu bloc. El slider afegeix un offset global addicional.
        """
        if not self._has_toc_align_chart or self._bp_align_cache is None:
            return

        import numpy as np
        cache = self._bp_align_cache
        t_toc = cache['t_toc']
        y_toc = cache['y_toc']
        t_dad = cache['t_dad']
        y_dad = cache['y_dad']
        global_offset = (self._bp_delay_current or 0) - (cache.get('_base_delay') or self._bp_delay_current or 0)
        inj_list = cache['per_injection']
        cadence = cache['cadence']
        toc_timeouts = cache.get('toc_timeouts', [])
        delay_blocks = cache.get('delay_blocks', [])

        if not inj_list:
            return

        # Window
        first_t = inj_list[0]['t_hplc']
        last_t = inj_list[-1]['t_hplc']
        win_start = max(0, first_t - 2)
        win_end = last_t + cadence + 10

        self._toc_align_figure.clear()
        ax = self._toc_align_figure.add_subplot(111)

        # --- TOC per blocs: cada segment desplaçat pel delay del seu bloc ---
        if len(t_toc) > 0 and delay_blocks:
            # Construir llista de segments TOC amb el delay del bloc
            to_times = sorted([to['t_min'] for to in toc_timeouts])
            # Blocs: definits per timeouts
            seg_edges = [0.0] + to_times + [t_toc[-1] + 1]
            block_idx = 0

            for s in range(len(seg_edges) - 1):
                seg_start = seg_edges[s]
                seg_end = seg_edges[s + 1]

                # Trobar el delay_block per aquest segment
                block_delay = self._bp_delay_current or 0  # fallback
                for db in delay_blocks:
                    db_start = db['t_start']
                    db_end = db['t_end'] if db['t_end'] is not None else 1e9
                    # Si el segment cau dins aquest bloc
                    seg_mid = (seg_start + seg_end) / 2
                    if db_start <= seg_mid < db_end:
                        block_delay = db['delay_median']
                        break

                total_shift = block_delay + global_offset
                mask = (t_toc >= seg_start) & (t_toc < seg_end)
                if not mask.any():
                    continue
                t_shifted = t_toc[mask] - total_shift
                y_seg = y_toc[mask]

                # Filtrar per finestra visible
                vis = (t_shifted >= win_start) & (t_shifted <= win_end)
                if vis.any():
                    ax.plot(t_shifted[vis], y_seg[vis], 'b-', lw=0.5, alpha=0.7)

            ax.set_ylabel('DOC (ppb)', color='blue')
        elif len(t_toc) > 0:
            # Fallback: delay global únic
            delay = self._bp_delay_current or 0
            t_shifted = t_toc - delay
            mask = (t_shifted >= win_start) & (t_shifted <= win_end)
            if mask.any():
                ax.plot(t_shifted[mask], y_toc[mask], 'b-', lw=0.5, alpha=0.7)
            ax.set_ylabel('DOC (ppb)', color='blue')

        # DAD fix (eix HPLC)
        if len(t_dad) > 0:
            ax2 = ax.twinx()
            mask_d = (t_dad >= win_start) & (t_dad <= win_end)
            if mask_d.any():
                ax2.plot(t_dad[mask_d], y_dad[mask_d], 'g-', lw=0.5,
                         alpha=0.6, label='DAD 254 (mAU)')
            ax2.set_ylabel('DAD 254 (mAU)', color='green')

        # Timeouts: línies verticals vermelles (al temps HPLC equivalent)
        for i_to, to in enumerate(toc_timeouts):
            # El timeout passa a t_toc = to['t_min']. Quin bloc?
            block_delay = self._bp_delay_current or 0
            for db in delay_blocks:
                db_start = db['t_start']
                db_end = db['t_end'] if db['t_end'] is not None else 1e9
                if db_start <= to['t_min'] < db_end:
                    block_delay = db['delay_median']
                    break
            t_to_hplc = to['t_min'] - block_delay - global_offset
            if win_start <= t_to_hplc <= win_end:
                ax.axvline(t_to_hplc, color='#E74C3C', ls='--', lw=0.8,
                           alpha=0.5, zorder=5)
                # Etiqueta només cada 3 timeouts per no saturar
                if i_to % 3 == 0:
                    ax.annotate(f"TO",
                                (t_to_hplc + 0.3, ax.get_ylim()[1] * 0.02),
                                fontsize=5, color='#E74C3C', alpha=0.6)

        # Bandes d'injecció HPLC
        cmap = __import__('matplotlib').colormaps['tab20']
        y_lims = ax.get_ylim()
        y_top = y_lims[1] if y_lims[1] > 0 else 100

        for i, inj in enumerate(inj_list):
            t_start = inj['t_hplc']
            t_end = (inj_list[i + 1]['t_hplc'] if i + 1 < len(inj_list)
                     else t_start + cadence)

            if t_end < win_start or t_start > win_end:
                continue

            is_ctrl = inj['is_control']
            is_khp = 'khp' in inj['name'].lower()

            if is_ctrl:
                color, alpha = '#95A5A6', 0.04
            elif is_khp:
                color, alpha = '#E74C3C', 0.12
            else:
                color, alpha = cmap(i % 20), 0.07

            ax.axvspan(max(t_start, win_start), min(t_end, win_end),
                       alpha=alpha, color=color, zorder=0)

            t_mid = (t_start + t_end) / 2
            if win_start <= t_mid <= win_end:
                sname = inj['name'][:6]
                label = f"{i+1}:{sname}"
                lc = ('#E74C3C' if is_khp else '#95A5A6' if is_ctrl else '#333')
                ax.annotate(label, (t_mid, y_top * 0.97),
                            fontsize=5, rotation=90, va='top', ha='center',
                            color=lc, alpha=0.8,
                            fontweight='bold' if is_khp else 'normal')

        ax.set_xlim(win_start, win_end)
        ax.set_xlabel('min (temps HPLC)')

        # Títol amb info de blocs
        n_blocks = len(delay_blocks)
        if delay_blocks:
            d_range = f"{min(db['delay_median'] for db in delay_blocks):.1f}–{max(db['delay_median'] for db in delay_blocks):.1f}"
        else:
            d_range = "?"
        ax.set_title(
            f'Blau=DOC (alineat per bloc) | Verd=DAD 254 | '
            f'{n_blocks} blocs, delay {d_range} min',
            fontsize=8)
        ax.spines['top'].set_visible(False)

        try:
            self._toc_align_figure.tight_layout()
        except Exception:
            pass
        self._toc_align_canvas.draw_idle()

    # --- Slider event handlers ---

    def _on_toc_delay_slider_changed(self, value):
        """Slider moved — update spinbox and redraw."""
        delay = value / 10.0
        self._toc_delay_spin.blockSignals(True)
        self._toc_delay_spin.setValue(delay)
        self._toc_delay_spin.blockSignals(False)
        self._bp_delay_current = delay
        self._draw_toc_alignment_chart()
        self._update_toc_delay_impact()

    def _on_toc_delay_spin_changed(self, value):
        """Spinbox changed — update slider and redraw."""
        self._toc_delay_slider.blockSignals(True)
        self._toc_delay_slider.setValue(int(value * 10))
        self._toc_delay_slider.blockSignals(False)
        self._bp_delay_current = value
        self._draw_toc_alignment_chart()
        self._update_toc_delay_impact()

    def _on_toc_delay_reset(self):
        """Reset slider to MasterFile delay."""
        if self._bp_delay_original is not None:
            self._toc_delay_spin.setValue(self._bp_delay_original)

    def _update_toc_delay_impact(self):
        """Update impact label showing offset state."""
        offset = self._bp_delay_current or 0
        if abs(offset) < 0.05:
            self._toc_delay_impact.setText("Alineació automàtica ✓")
            self._toc_delay_apply_btn.setVisible(False)
        else:
            self._toc_delay_impact.setText(f"Offset manual: {offset:+.1f} min")
            self._toc_delay_apply_btn.setVisible(True)
            self._toc_delay_apply_btn.setEnabled(True)

    def _on_toc_delay_apply(self):
        """Apply the slider delay to the MasterFile and reimport."""
        from PySide6.QtWidgets import QMessageBox
        cache = self._bp_align_cache
        if cache is None or not cache.get('mf_path'):
            return

        old_delay = self._bp_delay_original or 0.0
        # El slider és un OFFSET sobre l'alineació automàtica (0 = auto). El delay
        # absolut a escriure és delay_original + offset, NO l'offset tot sol.
        offset = self._bp_delay_current or 0.0
        new_delay = old_delay + offset

        reply = QMessageBox.question(
            self, "Aplicar delay",
            f"Canviar el delay de {old_delay:.2f} a {new_delay:.2f} min "
            f"(offset {offset:+.2f})?\n\n"
            f"  • S'escriurà 'Net delay (Suite)' al 0-INFO\n"
            f"  • Es regenerarà el 4-TOC_CALC\n"
            f"  • Es reimportarà la seqüència\n\n"
            f"(Es crea backup automàtic del MasterFile)",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        try:
            from hpsec_delay import update_masterfile_delay
            res = update_masterfile_delay(cache['mf_path'], new_delay)
            if res.get('success'):
                # El nou delay absolut passa a ser la base; l'offset torna a 0
                # perquè un segon "Aplicar" no el torni a sumar.
                self._bp_delay_original = new_delay
                self._bp_delay_current = 0.0
                for _w in (self._toc_delay_spin, self._toc_delay_slider):
                    _w.blockSignals(True)
                    _w.setValue(0)
                    _w.blockSignals(False)
                self._update_toc_delay_impact()
                self.delay_corrected.emit()
            else:
                QMessageBox.warning(self, "Error",
                                    f"No s'ha pogut actualitzar: {res.get('error', '?')}")
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))

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
            logger.warning(f"No s'ha pogut determinar el delay del MasterFile: {mf_path}")
            self.delay_shift_label.setText("—")
            self.delay_current_label.setText("No determinat")
            self.delay_info_label.setText(
                "⚠️ No s'ha pogut determinar el delay actual del MasterFile "
                "(falten hores HPLC/TOC al 0-INFO)."
            )
            self.delay_apply_btn.setVisible(False)
            self.delay_group.setVisible(True)
            return

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
            new_delay = current_delay - shift_min
            self.delay_apply_btn.setText(
                f"Corregir delay ({-shift_min:+.2f} min) i reimportar"
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
        new_delay = old_delay - shift
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
                f"Corregir delay ({-shift:+.2f} min) i reimportar"
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

