"""
HPSEC Suite - QA/QC KHP Panel
==============================

Panel per a la fase 2: QA/QC KHP.
Verifica el KHP mesurat vs la calibració global (rf_mass_cal),
determina el time shift necessari i mostra mètriques i històric.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QGroupBox,
    QGridLayout, QFrame, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QSplitter, QScrollArea, QSizePolicy, QComboBox
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QColor

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from hpsec_calibrate import (
    calibrate_from_import, load_khp_history, load_local_calibrations,
    get_all_active_calibrations, load_qc_history, get_rf_mass_cal
)
from hpsec_config import get_config

import numpy as np

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

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.calibration_data = None
        self.worker = None
        self._existing_calibration = None  # Calibració existent carregada
        self._all_calibrations = []  # Totes les calibracions disponibles (múltiples condicions)
        self._current_condition_key = None  # Condició seleccionada
        self._warnings_confirmed = False  # G05: Traçabilitat
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
        if hasattr(self, 'summary_group'):
            self.summary_group.setVisible(False)
        if hasattr(self, 'khp_graph'):
            self.khp_graph.clear()
        if hasattr(self, 'history_graph'):
            self.history_graph.clear()
        if hasattr(self, 'calibration_line_graph'):
            self.calibration_line_graph.clear()

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

        # Sempre actualitzar el selector (pot haver-hi noves calibracions)
        # Però no recarregar si ja tenim dades vàlides
        has_valid_data = self.calibration_data and self.calibration_data.get("success")

        try:
            # Carregar totes les calibracions locals
            all_cals = load_local_calibrations(seq_path)
            from_local = bool(all_cals)

            # Si no hi ha locals, intentar carregar des de l'històric global
            if not all_cals:
                all_cals = load_khp_history(seq_path)
                from_local = False

            # Marcar origen per mostrar a la UI
            for cal in all_cals:
                cal['_from_local'] = from_local

            if not all_cals:
                self.condition_selector_frame.setVisible(False)
                return

            # Filtrar per la SEQ actual i agrupar per condition_key
            seq_name = os.path.basename(seq_path)
            calibrations_by_condition = {}

            for cal in all_cals:
                cal_seq = cal.get('seq_name', '')
                # Acceptar coincidència exacta o si el seq_name està contingut
                if cal_seq != seq_name and seq_name not in cal_seq:
                    continue
                condition_key = cal.get('condition_key', 'default')
                # Guardar la més recent (primera trobada) per cada condició
                if condition_key not in calibrations_by_condition:
                    calibrations_by_condition[condition_key] = cal

            if not calibrations_by_condition:
                self.condition_selector_frame.setVisible(False)
                return

            # Guardar calibracions disponibles
            self._all_calibrations = list(calibrations_by_condition.values())

            # Configurar selector de condicions (sempre actualitzar per mostrar totes)
            self._populate_condition_combo()

            # Si ja tenim dades vàlides, només actualitzar el selector sense recarregar
            if has_valid_data:
                # Seleccionar la condició actual al combo si existeix
                if self._current_condition_key:
                    for i in range(self.condition_combo.count()):
                        if self.condition_combo.itemData(i) == self._current_condition_key:
                            self.condition_combo.blockSignals(True)
                            self.condition_combo.setCurrentIndex(i)
                            self.condition_combo.blockSignals(False)
                            break
                return

            # Carregar la primera calibració activa (o la primera disponible)
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
            print(f"[WARNING] Error comprovant calibració existent: {e}")
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

            # Format llegible: "KHP 2ppm @ 50µL" o "BP_50_2"
            if volume > 0 and conc > 0:
                label = f"KHP {conc:.0f}ppm @ {volume:.0f}µL"
                if mode:
                    label = f"{mode}: {label}"
            else:
                label = condition_key

            self.condition_combo.addItem(label, condition_key)

        self.condition_combo.blockSignals(False)

        # Mostrar selector només si hi ha múltiples condicions
        self.condition_selector_frame.setVisible(len(self._all_calibrations) > 1)

    def _on_condition_changed(self, index):
        """Handler quan l'usuari canvia la condició de calibració."""
        if index < 0 or index >= len(self._all_calibrations):
            return

        condition_key = self.condition_combo.itemData(index)
        if condition_key == self._current_condition_key:
            return

        # Buscar calibració per aquesta condició
        for cal in self._all_calibrations:
            if cal.get('condition_key') == condition_key:
                self._current_condition_key = condition_key
                self._load_existing_calibration(cal)
                self.main_window.set_status(f"Mostrant calibració: {self.condition_combo.currentText()}", 3000)
                break

    def _try_load_signals_for_replicas(self, cal_enriched):
        """Propaga dades de bigaussian fit i peak_info a les rèpliques per als gràfics.

        Les dades de l'històric no contenen arrays de senyal (t_doc, y_doc).
        Propagam el bigaussian fit des del top-level perquè el widget de gràfics
        pugui dibuixar la corba de fit en lloc de "Sense dades".
        """
        replicas = cal_enriched.get('replicas', [])
        if not replicas:
            return

        # Si ja tenen t_doc, no cal fer res
        if replicas[0].get('t_doc') is not None:
            return

        # Propagar bigaussian_doc del top-level a les rèpliques
        bigaussian_doc = cal_enriched.get('bigaussian_doc')
        t_retention = cal_enriched.get('t_retention', 0)
        area = cal_enriched.get('area', 0)

        for rep in replicas:
            # Bigaussian fit (per visualització)
            if bigaussian_doc and not rep.get('bigaussian_doc'):
                rep['bigaussian_doc'] = bigaussian_doc

            # Peak info (per marcar pic al gràfic)
            if not rep.get('peak_info') and t_retention > 0:
                rep['peak_info'] = {
                    't_max': rep.get('t_max', t_retention),
                    'y_max': rep.get('area', area),
                }

    def _load_existing_calibration(self, cal):
        """Carrega una calibració existent de l'històric.

        Enriqueix les dades per compatibilitat amb les funcions de visualització
        que esperen el format de calibrate_from_import (amb replicas, rf_mass_doc, etc.)
        """
        area = cal.get('area', 0)
        conc = cal.get('conc_ppm', 5)
        volume = cal.get('volume_uL', 0)
        rf = cal.get('rf', 0)
        if rf == 0 and conc > 0:
            rf = area / conc
        rf_direct = cal.get('rf_direct', rf)
        rf_uib = cal.get('rf_uib', 0)
        rf_mass = cal.get('rf_mass', 0)

        # Preparar replicas a partir de replicas_info amb camps compatibles
        replicas_info = cal.get('replicas_info', [])
        replicas = []
        for rep_info in replicas_info:
            rep = dict(rep_info)
            # Camps de compatibilitat per _update_summary i _update_metrics_table
            rep['t_doc_max'] = rep.get('t_max', 0)
            rep['t_retention'] = rep.get('t_max', 0)
            # rf_mass_doc per rèplica
            rep_area = rep.get('area', 0)
            if rep_area > 0 and conc > 0 and volume > 0:
                rep['rf_mass_doc'] = rep_area * 1000 / (conc * volume)
            else:
                rep['rf_mass_doc'] = rf_mass
            # Camps no disponibles per rèplica: usar valors top-level
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

        # Si no hi ha replicas_info, crear rèplica virtual des de dades top-level
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

        # Enriquir cal amb replicas compatibles
        cal_enriched = dict(cal)
        cal_enriched['replicas'] = replicas
        cal_enriched['rf_mass_doc'] = rf_mass
        cal_enriched['n_replicas'] = cal.get('n_replicas', len(replicas))

        # Intentar carregar senyals des del manifest (per gràfics)
        self._try_load_signals_for_replicas(cal_enriched)

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
            "khp_source": f"HISTÒRIC: {cal.get('seq_name', 'N/A')}",
            "khp_data": cal_enriched,
            "khp_data_direct": cal_enriched,
            "khp_data_uib": None,
            "calibration": cal,
            "errors": [],
            "loaded_from_history": True,
        }

        self.calibration_data = result
        self.main_window.calibration_data = result
        self._current_condition_key = cal.get('condition_key')

        # Actualitzar selecció del combo
        if hasattr(self, 'condition_combo') and self._current_condition_key:
            for i in range(self.condition_combo.count()):
                if self.condition_combo.itemData(i) == self._current_condition_key:
                    self.condition_combo.blockSignals(True)
                    self.condition_combo.setCurrentIndex(i)
                    self.condition_combo.blockSignals(False)
                    break

        # Mostrar resultats (TOTS els mètodes, igual que _on_finished)
        self._update_summary(result)
        self._update_graphs(result)
        self._update_metrics_table(result)
        self._update_replica_selection(result)
        self._update_validation(result)
        self._update_history(result)

        self.main_window.enable_tab(2)

        # Indicar font de la calibració
        source = "local" if cal.get('_from_local') else "global"
        self.main_window.set_status(f"Calibració carregada ({source}): {cal.get('condition_key', 'N/A')}", 3000)

        # Emetre senyal per notificar al wizard
        self.calibration_completed.emit(result)

    def _setup_ui(self):
        """Configura la interfície - NET i MINIMALISTA.

        Estructura:
        - Selector de condicions (si múltiples calibracions)
        - Resum de calibració
        - Gràfics i mètriques

        Nota: Títol, avisos, notes i navegació són al wizard header.
        """
        layout = QVBoxLayout(self)
        apply_panel_layout(layout)

        # Botó calibrar (amagat - l'acció es dispara des del wizard header)
        self.calibrate_btn = QPushButton()
        self.calibrate_btn.setVisible(False)
        self.calibrate_btn.clicked.connect(self._run_calibrate)

        # Selector de condicions de calibració (visible quan hi ha múltiples condicions)
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

        # Contenedor principal con scroll
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setSpacing(16)

        # === PLACEHOLDER (mentre carrega) ===
        self.placeholder = QLabel("Preparant QA/QC KHP...")
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.placeholder.setStyleSheet("color: #888; font-size: 14px; padding: 40px;")
        content_layout.addWidget(self.placeholder)

        # === SECCIÓN: Resumen de Calibración (reorganitzat per senyals) ===
        self.summary_group = QGroupBox("Resum QA/QC KHP")
        self.summary_group.setVisible(False)
        summary_main_layout = QVBoxLayout(self.summary_group)

        # --- Secció: Informació General ---
        general_group = QGroupBox("Informació General")
        general_group.setStyleSheet(STYLE_GROUPBOX)
        general_layout = QGridLayout(general_group)
        general_layout.setSpacing(8)

        self.result_labels = {}
        general_items = [
            ("seq_name", "SEQ:", 0, 0),
            ("mode", "Mode:", 0, 2),
            ("khp_source", "Font KHP:", 1, 0),
            ("khp_conc", "Concentració:", 1, 2),
            ("volume", "Volum injecció:", 2, 0),
            ("n_replicas", "Rèpliques:", 2, 2),
            ("uib_sensitivity", "Sensibilitat UIB:", 3, 0),
            ("qc_status", "QC Status:", 3, 2),
        ]

        for key, label_text, row, col in general_items:
            lbl = QLabel(label_text)
            lbl.setStyleSheet("font-weight: bold; color: #2C3E50;")
            general_layout.addWidget(lbl, row, col)
            val = QLabel("-")
            self.result_labels[key] = val
            general_layout.addWidget(val, row, col + 1)

        summary_main_layout.addWidget(general_group)

        # --- Secció: DOC Direct ---
        self.direct_group = QGroupBox("DOC Direct")
        self.direct_group.setStyleSheet("QGroupBox { font-weight: bold; color: #1A5276; }")
        direct_layout = QGridLayout(self.direct_group)
        direct_layout.setSpacing(8)

        direct_items = [
            ("rf_direct", "RF (Àrea/ppm):", 0, 0),
            ("rf_mass_direct", "RF_MASS:", 0, 2),
            ("fwhm_direct", "FWHM:", 1, 0),
            ("shift_direct", "Shift (vs 254):", 1, 2),
            ("snr_direct", "SNR:", 2, 0),
            ("tmax_direct", "t_max:", 2, 2),
        ]

        for key, label_text, row, col in direct_items:
            lbl = QLabel(label_text)
            lbl.setStyleSheet("font-weight: bold; color: #2874A6;")
            direct_layout.addWidget(lbl, row, col)
            val = QLabel("-")
            self.result_labels[key] = val
            direct_layout.addWidget(val, row, col + 1)

        summary_main_layout.addWidget(self.direct_group)

        # --- Secció: DOC UIB ---
        self.uib_group = QGroupBox("DOC UIB")
        self.uib_group.setStyleSheet("QGroupBox { font-weight: bold; color: #1A5276; }")
        uib_layout = QGridLayout(self.uib_group)
        uib_layout.setSpacing(8)

        uib_items = [
            ("rf_uib", "RF (Àrea/ppm):", 0, 0),
            ("rf_mass_uib", "RF_MASS:", 0, 2),
            ("fwhm_uib", "FWHM:", 1, 0),
            ("shift_uib", "Shift (vs 254):", 1, 2),
            ("snr_uib", "SNR:", 2, 0),
            ("tmax_uib", "t_max:", 2, 2),
        ]

        for key, label_text, row, col in uib_items:
            lbl = QLabel(label_text)
            lbl.setStyleSheet("font-weight: bold; color: #2874A6;")
            uib_layout.addWidget(lbl, row, col)
            val = QLabel("-")
            self.result_labels[key] = val
            uib_layout.addWidget(val, row, col + 1)

        summary_main_layout.addWidget(self.uib_group)

        content_layout.addWidget(self.summary_group)

        # === SECCIÓN: Gráficos de KHP (per rèplica) ===
        self.graphs_group = QGroupBox("Gràfics KHP (DOC + DAD 254nm)")
        self.graphs_group.setVisible(False)
        graphs_layout = QVBoxLayout(self.graphs_group)

        # Widget únic que mostra totes les rèpliques
        self.replica_graphs = KHPReplicaGraphWidget()
        graphs_layout.addWidget(self.replica_graphs)

        content_layout.addWidget(self.graphs_group)

        # === SECCIÓN: Tabla de Métricas por Réplica ===
        self.metrics_group = QGroupBox("Mètriques per Rèplica")
        self.metrics_group.setVisible(False)
        metrics_layout = QVBoxLayout(self.metrics_group)

        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(17)
        self.metrics_table.setHorizontalHeaderLabels([
            "Rep", "Senyal", "Àrea", "DOC/254", "FWHM", "RF_M", "CR",
            "t_max", "Shift", "SNR", "Sym", "R²", "Pic_J", "TO", "Pics", "Q", "Estat"
        ])
        # Tooltips per les capçaleres de mètriques
        self.metrics_table.horizontalHeaderItem(2).setToolTip("Àrea DOC integrada")
        self.metrics_table.horizontalHeaderItem(3).setToolTip("Ratio DOC/254nm - Consistència entre senyals")
        self.metrics_table.horizontalHeaderItem(4).setToolTip("FWHM (min) - Amplada a mitja alçada\nNormal: 0.9-1.5 min")
        self.metrics_table.horizontalHeaderItem(5).setToolTip("RF_MASS = Àrea×1000/(ppm×µL) - Àrea per µg DOC injectat")
        self.metrics_table.horizontalHeaderItem(6).setToolTip("CR = pic/total - Concentration Ratio\nCOLUMN: ~0.65, BP: ~1.0")
        self.metrics_table.horizontalHeaderItem(7).setToolTip("Temps del pic màxim (min)")
        self.metrics_table.horizontalHeaderItem(8).setToolTip("Shift vs 254nm (segons)")
        self.metrics_table.horizontalHeaderItem(10).setToolTip("Simetria (sigma_left/sigma_right)\nIdeal: 1.0, Rang: 0.5-2.0")
        self.metrics_table.horizontalHeaderItem(11).setToolTip("R² del fit bigaussià\n≥0.95 VALID, ≥0.80 CHECK")
        self.metrics_table.horizontalHeaderItem(12).setToolTip("Pic_J: Pic amb vall (artefacte)\n+100 si detectat")
        self.metrics_table.horizontalHeaderItem(13).setToolTip("Timeout detectat\n+100 si afecta pic, 0 si fora")
        self.metrics_table.horizontalHeaderItem(14).setToolTip("Pics en zona ±4min\n>1 = INVALID (+100)")
        self.metrics_table.horizontalHeaderItem(15).setToolTip("Quality Score (0=perfecte, >=100=invalid)")
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.metrics_table.setAlternatingRowColors(True)
        self.metrics_table.setMinimumHeight(150)
        self.metrics_table.setMaximumHeight(250)
        # Permetre selecció i còpia
        self.metrics_table.setSelectionMode(QTableWidget.ExtendedSelection)
        self.metrics_table.setSelectionBehavior(QTableWidget.SelectItems)
        metrics_layout.addWidget(self.metrics_table)

        content_layout.addWidget(self.metrics_group)

        # === SECCIÓN: Selecció de Rèpliques ===
        self.replica_selection_group = QGroupBox("Selecció de Rèpliques")
        self.replica_selection_group.setVisible(False)
        replica_sel_layout = QVBoxLayout(self.replica_selection_group)
        replica_sel_layout.setSpacing(8)

        # Fila superior: info selecció actual i controls
        replica_header = QHBoxLayout()

        # Etiqueta selecció actual
        self.selection_info_label = QLabel("Selecció: -")
        self.selection_info_label.setStyleSheet("font-weight: bold; color: #2C3E50;")
        replica_header.addWidget(self.selection_info_label)

        replica_header.addStretch()

        # ComboBox per canviar selecció
        replica_header.addWidget(QLabel("Canviar a:"))
        self.replica_selection_combo = QComboBox()
        self.replica_selection_combo.setMinimumWidth(150)
        self.replica_selection_combo.setToolTip("Seleccionar quines rèpliques usar per la calibració")
        replica_header.addWidget(self.replica_selection_combo)

        # Botó aplicar
        self.apply_selection_btn = QPushButton("Aplicar")
        self.apply_selection_btn.setEnabled(False)
        self.apply_selection_btn.clicked.connect(self._on_apply_replica_selection)
        self.apply_selection_btn.setStyleSheet("""
            QPushButton {
                background: #3498DB; color: white; border: none;
                border-radius: 4px; padding: 6px 12px; font-weight: bold;
            }
            QPushButton:hover { background: #2980B9; }
            QPushButton:disabled { background: #BDC3C7; }
        """)
        replica_header.addWidget(self.apply_selection_btn)

        replica_sel_layout.addLayout(replica_header)

        # Taula comparació rèpliques (C09: eliminada columna Outlier, botó separat)
        self.replica_comparison_table = QTableWidget()
        self.replica_comparison_table.setColumnCount(9)
        self.replica_comparison_table.setHorizontalHeaderLabels([
            "Rèplica", "Àrea", "t_max", "SNR", "Sym", "DOC/254", "Shift", "Q", "Status"
        ])
        self.replica_comparison_table.horizontalHeaderItem(1).setToolTip("Àrea DOC integrada")
        self.replica_comparison_table.horizontalHeaderItem(2).setToolTip("Temps del pic màxim (min)")
        self.replica_comparison_table.horizontalHeaderItem(3).setToolTip("Signal-to-Noise Ratio")
        self.replica_comparison_table.horizontalHeaderItem(4).setToolTip("Simetria del pic")
        self.replica_comparison_table.horizontalHeaderItem(5).setToolTip("Ratio DOC/254nm")
        self.replica_comparison_table.horizontalHeaderItem(6).setToolTip("Shift vs 254nm (segons)")
        self.replica_comparison_table.horizontalHeaderItem(7).setToolTip("Quality Score")
        self.replica_comparison_table.horizontalHeaderItem(8).setToolTip("Usada en calibració actual")
        self.replica_comparison_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        # Columna Status (8) amb amplada mínima per no quedar tallada
        self.replica_comparison_table.horizontalHeader().setMinimumSectionSize(90)
        self.replica_comparison_table.setAlternatingRowColors(True)
        self.replica_comparison_table.setMaximumHeight(140)
        self.replica_comparison_table.verticalHeader().setVisible(False)
        self.replica_comparison_table.setSelectionBehavior(QTableWidget.SelectRows)
        replica_sel_layout.addWidget(self.replica_comparison_table)

        # Fila inferior: botó per marcar outlier i estadístiques
        replica_footer = QHBoxLayout()

        # Botó per marcar rèplica com a outlier (C08: amagat - ara usen dropdown a columna Status)
        # self.mark_replica_outlier_btn = QPushButton("Marcar com a Outlier")
        # self.mark_replica_outlier_btn.setToolTip("Marca la rèplica seleccionada com a outlier")
        # self.mark_replica_outlier_btn.clicked.connect(self._on_mark_replica_outlier)
        # replica_footer.addWidget(self.mark_replica_outlier_btn)

        # Estadístiques diferències (sense stretch per no empènyer a la dreta)
        self.replica_diff_label = QLabel()
        self.replica_diff_label.setWordWrap(True)
        self.replica_diff_label.setStyleSheet(
            "color: #2C3E50; font-size: 12px; font-weight: bold; padding: 6px; "
            "background: #EBF5FB; border-radius: 4px;"
        )
        replica_footer.addWidget(self.replica_diff_label, 1)

        replica_sel_layout.addLayout(replica_footer)

        content_layout.addWidget(self.replica_selection_group)

        # === SECCIÓN: Validación y Problemas ===
        self.validation_group = QGroupBox("Validació i Problemes")
        self.validation_group.setVisible(False)
        validation_layout = QVBoxLayout(self.validation_group)

        self.validation_label = QLabel()
        self.validation_label.setWordWrap(True)
        self.validation_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        validation_layout.addWidget(self.validation_label)

        content_layout.addWidget(self.validation_group)

        # === SECCIÓN: Comparación Histórica ===
        self.history_group = QGroupBox("Històric QA/QC")
        self.history_group.setVisible(False)
        history_layout = QVBoxLayout(self.history_group)
        history_layout.setSpacing(6)

        # Header amb filtres i botons
        history_header = QHBoxLayout()
        self.history_filters_label = QLabel()
        self.history_filters_label.setStyleSheet("color: #555; font-size: 11px;")
        history_header.addWidget(self.history_filters_label)

        # Toggle per incloure outliers
        from PySide6.QtWidgets import QCheckBox
        self.show_outliers_cb = QCheckBox("Incloure outliers")
        self.show_outliers_cb.setToolTip("Mostrar també les calibracions marcades com a outliers")
        self.show_outliers_cb.stateChanged.connect(self._on_show_outliers_changed)
        history_header.addWidget(self.show_outliers_cb)

        history_header.addStretch()

        # Botó info per mostrar llegenda (C16: ara mostra diàleg al clicar)
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

        # Layout horitzontal: taula a l'esquerra, gràfic a la dreta
        history_content = QHBoxLayout()

        # Taula d'històric (esquerra)
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(12)
        self.history_table.setHorizontalHeaderLabels([
            "SEQ", "Mode", "Àrea", "t_max", "SNR", "DOC/254", "CR", "Sym", "Shift", "Q", "Estat", "Motiu"
        ])
        # Tooltips per les capçaleres
        self.history_table.horizontalHeaderItem(3).setToolTip("Temps del pic màxim (min)")
        self.history_table.horizontalHeaderItem(4).setToolTip("Signal-to-Noise Ratio")
        self.history_table.horizontalHeaderItem(5).setToolTip("Ratio DOC/254nm")
        self.history_table.horizontalHeaderItem(6).setToolTip("Concentration Ratio (pic/total)")
        self.history_table.horizontalHeaderItem(7).setToolTip("Simetria del pic")
        self.history_table.horizontalHeaderItem(9).setToolTip("Quality Score v2 (0=perfecte, >=100=invalid)")
        self.history_table.horizontalHeaderItem(11).setToolTip("Motiu d'exclusió si aplica")
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.history_table.setAlternatingRowColors(False)
        self.history_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.history_table.setMaximumHeight(180)
        self.history_table.verticalHeader().setVisible(False)
        self.history_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.history_table.setStyleSheet("QTableWidget { font-size: 11px; }")
        history_content.addWidget(self.history_table, 3)

        # Gràfics (dreta): recta calibració + barres
        graphs_layout = QVBoxLayout()
        graphs_layout.setSpacing(5)

        # Gràfic recta de calibració (a sobre)
        self.calibration_line_graph = CalibrationLineWidget()
        graphs_layout.addWidget(self.calibration_line_graph)

        # Gràfic de barres compacte (a sota)
        self.history_graph = HistoryBarWidget()
        graphs_layout.addWidget(self.history_graph)

        history_content.addLayout(graphs_layout, 2)

        history_layout.addLayout(history_content)

        # Resum i botons
        history_footer = QHBoxLayout()
        self.history_summary = QLabel()
        self.history_summary.setStyleSheet("color: #666; font-size: 11px;")
        history_footer.addWidget(self.history_summary)
        history_footer.addStretch()

        # Botó per marcar/desmarcar outlier
        self.toggle_outlier_btn = QPushButton("Marcar Outlier")
        self.toggle_outlier_btn.setEnabled(False)
        self.toggle_outlier_btn.setToolTip("Marcar/desmarcar la calibració seleccionada com a outlier")
        self.toggle_outlier_btn.clicked.connect(self._toggle_outlier)
        self.toggle_outlier_btn.setStyleSheet("QPushButton { padding: 4px 8px; }")
        history_footer.addWidget(self.toggle_outlier_btn)

        # Botó per usar mitjana històrica
        self.use_average_btn = QPushButton("Usar Mitjana")
        self.use_average_btn.setEnabled(False)
        self.use_average_btn.setToolTip("Calibrar usant la mitjana de les calibracions vàlides")
        self.use_average_btn.clicked.connect(self._use_historical_average)
        self.use_average_btn.setStyleSheet("QPushButton { padding: 4px 8px; }")
        history_footer.addWidget(self.use_average_btn)

        # Botó per aplicar seleccionada
        self.select_cal_btn = QPushButton("Aplicar Seleccionada")
        self.select_cal_btn.setEnabled(False)
        self.select_cal_btn.clicked.connect(self._apply_selected_calibration)
        self.select_cal_btn.setStyleSheet("QPushButton { padding: 4px 8px; }")
        history_footer.addWidget(self.select_cal_btn)

        history_layout.addLayout(history_footer)

        # Connectar selecció de taula
        self.history_table.itemSelectionChanged.connect(self._on_history_selection_changed)

        content_layout.addWidget(self.history_group)

        # Spacer
        content_layout.addStretch()

        scroll.setWidget(content_widget)
        layout.addWidget(scroll, 1)

        # Referència dummy per compatibilitat amb wizard (el wizard l'amaga)
        self.next_btn = QPushButton()
        self.next_btn.setVisible(False)

    def _run_calibrate(self):
        """Executa la calibració."""
        imported_data = self.main_window.imported_data

        # Auto-carregar dades d'importació si no estan en memòria
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
                self,
                "No hi ha dades",
                "No s'han trobat dades d'importació.\n\n"
                "Cal importar la seqüència primer."
            )
            return

        self.calibrate_btn.setEnabled(False)
        self.main_window.show_progress(0)

        # Limpiar resultados anteriores
        self.summary_group.setVisible(False)
        self.graphs_group.setVisible(False)
        self.metrics_group.setVisible(False)
        self.validation_group.setVisible(False)
        self.history_group.setVisible(False)

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

        self.calibration_data = result
        self.main_window.calibration_data = result

        # Mostrar resultados
        self._update_summary(result)
        self._update_graphs(result)
        self._update_metrics_table(result)
        self._update_replica_selection(result)
        self._update_validation(result)
        self._update_history(result)

        # Auto-generar PDF de QA/QC
        try:
            from generate_calibration_report import generate_calibration_report
            seq_path = self.main_window.seq_path
            if seq_path:
                pdf = generate_calibration_report(seq_path)
                if pdf:
                    print(f"[INFO] Report QA/QC: {pdf}")
        except Exception as e:
            print(f"[WARNING] No s'ha pogut generar report de QA/QC: {e}")

        # Recarregar el selector de condicions (potser s'han creat noves calibracions)
        self._reload_condition_selector()

        # Nota: Els avisos es gestionen des del wizard header

        self.main_window.enable_tab(2)
        self.main_window.set_status("QA/QC KHP completat", 5000)

        # Emetre senyal per notificar al wizard
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

            # Seleccionar la condició actual al combo
            if self._current_condition_key:
                for i in range(self.condition_combo.count()):
                    if self.condition_combo.itemData(i) == self._current_condition_key:
                        self.condition_combo.blockSignals(True)
                        self.condition_combo.setCurrentIndex(i)
                        self.condition_combo.blockSignals(False)
                        break

        except Exception as e:
            print(f"[WARNING] Error recarregant selector de condicions: {e}")

    def _on_error(self, error_msg):
        self.main_window.show_progress(-1)
        self.calibrate_btn.setEnabled(True)

        # Continuar con defaults
        self.calibration_data = {
            "success": False,
            "factor_direct": 0,
            "factor_uib": 0,
            "shift_uib": 0,
            "errors": [error_msg]
        }
        self.main_window.calibration_data = self.calibration_data

        self.placeholder.setVisible(False)
        self.summary_group.setVisible(True)
        # Reset all labels
        self.result_labels["seq_name"].setText("-")
        self.result_labels["mode"].setText("Defaults (sense KHP)")
        self.result_labels["khp_source"].setText("-")
        self.result_labels["khp_conc"].setText("-")
        self.result_labels["volume"].setText("-")
        self.result_labels["n_replicas"].setText("-")
        self.result_labels["uib_sensitivity"].setText("-")
        # Direct
        self.result_labels["rf_direct"].setText("-")
        self.result_labels["shift_direct"].setText("-")
        self.result_labels["snr_direct"].setText("-")
        self.result_labels["tmax_direct"].setText("-")
        # UIB
        self.result_labels["rf_uib"].setText("-")
        self.result_labels["shift_uib"].setText("-")
        self.result_labels["snr_uib"].setText("-")
        self.result_labels["tmax_uib"].setText("-")
        # Hide signal sections
        self.direct_group.setVisible(False)
        self.uib_group.setVisible(False)

        # Mostrar error en validación
        self.validation_group.setVisible(True)
        self.validation_label.setText(
            f"<span style='color: red;'><b>Error durant el QA/QC:</b></span><br>"
            f"<pre>{error_msg}</pre>"
        )

        self.main_window.enable_tab(2)

    def _update_summary(self, result):
        """Actualiza el resumen de calibración amb format per senyals."""
        import os

        self.placeholder.setVisible(False)
        self.summary_group.setVisible(True)

        # === INFORMACIÓ GENERAL ===
        seq_path = self.main_window.seq_path or ""
        seq_name = os.path.basename(seq_path) if seq_path else "-"
        self.result_labels["seq_name"].setText(seq_name)

        mode = result.get("mode", "-")
        self.result_labels["mode"].setText(mode)
        self.result_labels["khp_source"].setText(result.get("khp_source", "LOCAL"))

        # Concentració KHP
        khp_conc = result.get("khp_conc", 0)
        self.result_labels["khp_conc"].setText(f"{khp_conc:.0f} ppm" if khp_conc > 0 else "-")

        # Volum injecció
        volume = None
        khp_data_main = result.get("khp_data_direct") or result.get("khp_data_uib")
        if khp_data_main:
            volume = khp_data_main.get('volume_uL')
            if not volume:
                replicas = khp_data_main.get('replicas') or []
                if replicas:
                    volume = replicas[0].get('volume_uL')
        self.result_labels["volume"].setText(f"{int(volume)} µL" if volume else "-")

        # Nombre de rèpliques
        n_replicas = khp_data_main.get("n_replicas", 1) if khp_data_main else 0
        self.result_labels["n_replicas"].setText(str(n_replicas) if n_replicas > 0 else "-")

        # Sensibilitat UIB (700 ppb o 1000 ppb)
        uib_sensitivity = None
        if khp_data_main:
            uib_sensitivity = khp_data_main.get('uib_sensitivity')
            if not uib_sensitivity:
                replicas = khp_data_main.get('replicas') or []
                for r in replicas:
                    uib_sensitivity = r.get('uib_sensitivity')
                    if uib_sensitivity:
                        break
        if uib_sensitivity:
            self.result_labels["uib_sensitivity"].setText(f"{uib_sensitivity} ppb")
        else:
            self.result_labels["uib_sensitivity"].setText("-")

        # === DOC DIRECT ===
        khp_data_direct = result.get("khp_data_direct")
        if khp_data_direct:
            self.direct_group.setVisible(True)

            # RF Direct
            area_direct = result.get("khp_area_direct", 0) or khp_data_direct.get('area', 0)
            if area_direct > 0 and khp_conc > 0:
                rf_direct = area_direct / khp_conc
                self.result_labels["rf_direct"].setText(f"{rf_direct:.0f}")
            else:
                self.result_labels["rf_direct"].setText("-")

            # Shift Direct (sempre en segons, amb minuts entre parèntesi)
            shift_direct = result.get("shift_direct", 0)
            shift_direct_sec = shift_direct * 60
            self.result_labels["shift_direct"].setText(f"{shift_direct_sec:.1f}s")

            # SNR, t_max, FWHM, RF_MASS Direct (de les rèpliques)
            replicas_direct = khp_data_direct.get('replicas') or [khp_data_direct]
            if replicas_direct:
                snr_vals = [r.get('snr', 0) for r in replicas_direct if r.get('snr')]
                tmax_vals = [r.get('t_retention', 0) or r.get('t_doc_max', 0) for r in replicas_direct]
                tmax_vals = [t for t in tmax_vals if t > 0]
                fwhm_vals = [r.get('fwhm_doc', 0) for r in replicas_direct if r.get('fwhm_doc')]
                rf_mass_vals = [r.get('rf_mass_doc', 0) or r.get('rf_mass', 0) for r in replicas_direct]
                rf_mass_vals = [v for v in rf_mass_vals if v > 0]
                # Fallback: rf_mass top-level del khp_data
                if not rf_mass_vals:
                    top_rf = khp_data_direct.get('rf_mass', 0) or khp_data_direct.get('rf_mass_doc', 0)
                    if top_rf > 0:
                        rf_mass_vals = [top_rf]

                self.result_labels["snr_direct"].setText(f"{np.mean(snr_vals):.0f}" if snr_vals else "-")
                self.result_labels["tmax_direct"].setText(f"{np.mean(tmax_vals):.2f} min" if tmax_vals else "-")
                self.result_labels["fwhm_direct"].setText(f"{np.mean(fwhm_vals):.2f} min" if fwhm_vals else "-")
                self.result_labels["rf_mass_direct"].setText(f"{np.mean(rf_mass_vals):.1f}" if rf_mass_vals else "-")
        else:
            self.direct_group.setVisible(False)

        # === DOC UIB ===
        khp_data_uib = result.get("khp_data_uib")
        if khp_data_uib:
            self.uib_group.setVisible(True)

            # RF UIB
            area_uib = result.get("khp_area_uib", 0) or khp_data_uib.get('area', 0)
            if area_uib > 0 and khp_conc > 0:
                rf_uib = area_uib / khp_conc
                self.result_labels["rf_uib"].setText(f"{rf_uib:.0f}")
            else:
                self.result_labels["rf_uib"].setText("-")

            # Shift UIB (en segons)
            shift_uib = result.get("shift_uib", 0)
            shift_uib_sec = shift_uib * 60
            self.result_labels["shift_uib"].setText(f"{shift_uib_sec:.1f}s")

            # SNR, t_max, FWHM, RF_MASS UIB (de les rèpliques)
            replicas_uib = khp_data_uib.get('replicas') or [khp_data_uib]
            if replicas_uib:
                snr_vals = [r.get('snr', 0) for r in replicas_uib if r.get('snr')]
                tmax_vals = [r.get('t_retention', 0) or r.get('t_doc_max', 0) for r in replicas_uib]
                tmax_vals = [t for t in tmax_vals if t > 0]
                fwhm_vals = [r.get('fwhm_doc', 0) for r in replicas_uib if r.get('fwhm_doc')]
                rf_mass_vals = [r.get('rf_mass_doc', 0) or r.get('rf_mass', 0) for r in replicas_uib]
                rf_mass_vals = [v for v in rf_mass_vals if v > 0]
                if not rf_mass_vals:
                    top_rf = khp_data_uib.get('rf_mass_u', 0) or khp_data_uib.get('rf_mass', 0)
                    if top_rf > 0:
                        rf_mass_vals = [top_rf]

                self.result_labels["snr_uib"].setText(f"{np.mean(snr_vals):.0f}" if snr_vals else "-")
                self.result_labels["tmax_uib"].setText(f"{np.mean(tmax_vals):.2f} min" if tmax_vals else "-")
                self.result_labels["fwhm_uib"].setText(f"{np.mean(fwhm_vals):.2f} min" if fwhm_vals else "-")
                self.result_labels["rf_mass_uib"].setText(f"{np.mean(rf_mass_vals):.1f}" if rf_mass_vals else "-")
        else:
            self.uib_group.setVisible(False)

        # === QC: Comparació rf_mass vs rf_mass_cal (global) ===
        try:
            khp_data_main = result.get("khp_data_direct") or result.get("khp_data_uib")
            rf_mass_measured = 0
            if khp_data_main:
                replicas_main = khp_data_main.get('replicas') or [khp_data_main]
                rf_vals = [r.get('rf_mass_doc', 0) or r.get('rf_mass', 0) for r in replicas_main]
                rf_vals = [v for v in rf_vals if v > 0]
                if rf_vals:
                    rf_mass_measured = np.mean(rf_vals)
                elif khp_data_main.get('rf_mass', 0) > 0:
                    rf_mass_measured = khp_data_main['rf_mass']

            if rf_mass_measured > 0:
                mode_str = result.get('mode', 'COLUMN').lower()
                rf_mass_cal = get_rf_mass_cal(signal='direct', mode=mode_str)
                if rf_mass_cal and rf_mass_cal > 0:
                    deviation_pct = abs(rf_mass_measured - rf_mass_cal) / rf_mass_cal * 100
                    config = get_config()
                    warn_pct = config.get('calibration', 'qc_thresholds', 'warning_pct', default=5.0)
                    fail_pct = config.get('calibration', 'qc_thresholds', 'fail_pct', default=10.0)

                    if deviation_pct <= warn_pct:
                        qc_text = f"PASS ({deviation_pct:.1f}% vs cal={rf_mass_cal:.0f})"
                        qc_style = "color: #27AE60; font-weight: bold;"
                    elif deviation_pct <= fail_pct:
                        qc_text = f"WARNING ({deviation_pct:.1f}% vs cal={rf_mass_cal:.0f})"
                        qc_style = "color: #F39C12; font-weight: bold;"
                    else:
                        qc_text = f"FAIL ({deviation_pct:.1f}% vs cal={rf_mass_cal:.0f})"
                        qc_style = "color: #E74C3C; font-weight: bold;"

                    self.result_labels["qc_status"].setText(qc_text)
                    self.result_labels["qc_status"].setStyleSheet(qc_style)
                else:
                    self.result_labels["qc_status"].setText("N/A (sense cal global)")
            else:
                self.result_labels["qc_status"].setText("-")
        except Exception as e:
            print(f"[DEBUG] Error calculant QC: {e}")
            self.result_labels["qc_status"].setText("-")

    def _extract_all_replicas(self, khp_data):
        """
        Extrae todas las réplicas de los datos KHP.

        khp_data puede ser:
        - Un dict con 'all_khp_data' o 'replicas' (resultado de select_best_khp)
        - Una lista de réplicas directamente
        - Un dict individual (única réplica)
        """
        if not khp_data:
            return []

        if isinstance(khp_data, list):
            return khp_data

        if isinstance(khp_data, dict):
            # Buscar lista de réplicas en diferentes claves
            replicas = khp_data.get('all_khp_data') or khp_data.get('replicas')
            if replicas and isinstance(replicas, list):
                return replicas
            # Es un dict individual
            return [khp_data]

        return []

    def _update_graphs(self, result):
        """Actualiza los gráficos de KHP per rèplica."""
        khp_data_direct = result.get("khp_data_direct")
        khp_data_uib = result.get("khp_data_uib")

        # Preparar datos para gráficos - extraer todas las réplicas
        direct_list = self._extract_all_replicas(khp_data_direct)
        uib_list = self._extract_all_replicas(khp_data_uib)

        has_graphs = len(direct_list) > 0

        if has_graphs:
            self.graphs_group.setVisible(True)
            # Usar el nou widget que mostra R1, R2 amb DOC i 254nm
            self.replica_graphs.plot_replicas(direct_list, uib_list if uib_list else None)
        else:
            self.graphs_group.setVisible(False)

    def _count_peaks_in_zone(self, khp, zone_min=4.0):
        """
        Compta pics dins de ±zone_min del pic principal.

        Args:
            khp: Dict amb dades de la rèplica
            zone_min: Zona al voltant del pic principal (minuts)

        Returns:
            Nombre de pics en la zona (1 = normal, >1 = múltiples)
        """
        peak_info = khp.get('peak_info', {})
        t_max = peak_info.get('t_max', 0) or khp.get('t_doc_max', 0) or khp.get('t_retention', 0)
        all_peaks = khp.get('all_peaks', [])

        if t_max <= 0 or not all_peaks:
            return 1  # Sense info, assumim OK

        count = 0
        for peak in all_peaks:
            t_peak = peak.get('t', 0)
            if abs(t_peak - t_max) <= zone_min:
                count += 1

        return max(count, 1)

    def _timeout_affects_peak(self, khp):
        """
        Determina si el timeout afecta el pic principal.

        Args:
            khp: Dict amb dades de la rèplica

        Returns:
            True si timeout afecta pic, False si no
        """
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

    def _calculate_quality_score(self, khp, signal='Direct'):
        """
        Calcula quality score amb nova lògica empírica.

        Args:
            khp: Dict amb dades de la rèplica
            signal: 'Direct' o 'UIB'

        Returns:
            (score, issues): Tuple amb puntuació i llista de problemes
        """
        score = 0
        issues = []
        is_bp = khp.get('is_bp', False)

        # === CRITERIS INVALIDANTS (+100) ===

        # Pic_J (Batman)
        if khp.get('has_batman', False):
            score += 100
            issues.append("Pic_J: pic amb vall")

        # Múltiples pics en zona ±4 min
        n_pics = self._count_peaks_in_zone(khp, zone_min=4.0)
        if n_pics > 1:
            score += 100
            issues.append(f"Múltiples pics: {n_pics} en zona ±4min")

        # Timeout afecta pic
        if self._timeout_affects_peak(khp):
            score += 100
            issues.append("Timeout afecta pic principal")

        # === WARNINGS (+20) ===

        fwhm = khp.get('fwhm_doc', 0)
        if fwhm > 1.5:
            score += 20
            issues.append(f"FWHM elevat: {fwhm:.2f} min")

        snr = khp.get('snr', 0)
        if 0 < snr < 10:
            score += 20
            issues.append(f"SNR baix: {snr:.1f}")

        # === INFO (+10) ===
        # NOTA: Shift NO penalitza (només informatiu)

        sym = khp.get('symmetry', 1.0)
        if sym > 0 and (sym < 0.5 or sym > 2.5):
            score += 10
            issues.append(f"Asimetria: {sym:.2f}")

        cr = khp.get('concentration_ratio', khp.get('cr_doc', 0))
        if cr > 0:
            if is_bp and cr < 0.95:
                score += 10
                issues.append(f"CR baix BP: {cr:.2f}")
            elif not is_bp and cr < 0.40:
                score += 10
                issues.append(f"CR baix: {cr:.2f}")

        return score, issues

    def _update_metrics_table(self, result):
        """Actualiza la tabla de métricas por réplica."""
        self.metrics_table.setRowCount(0)

        khp_data_direct = result.get("khp_data_direct")
        khp_data_uib = result.get("khp_data_uib")

        all_data = []

        # Recopilar timeouts de Direct per propagar a UIB
        direct_timeouts = {}  # {replica_num: timeout_info}

        # Recopilar datos Direct - todas las réplicas
        direct_list = self._extract_all_replicas(khp_data_direct)
        for d in direct_list:
            d_copy = d.copy()  # No modificar original
            d_copy['_signal'] = 'Direct'
            all_data.append(d_copy)
            # Guardar timeout per propagar
            if d.get('has_timeout'):
                import re
                fname = d.get('filename', '')
                match = re.search(r'R(\d+)', fname)
                rep_num = match.group(1) if match else '1'
                direct_timeouts[rep_num] = d.get('timeout_info', {})

        # Recopilar datos UIB - todas las réplicas (propagant timeouts de Direct)
        uib_list = self._extract_all_replicas(khp_data_uib)
        for d in uib_list:
            d_copy = d.copy()
            d_copy['_signal'] = 'UIB'
            # Propagar timeout de Direct si UIB no en té
            import re
            fname = d.get('filename', '')
            match = re.search(r'R(\d+)', fname)
            rep_num = match.group(1) if match else '1'
            if not d_copy.get('has_timeout') and rep_num in direct_timeouts:
                d_copy['has_timeout'] = True
                d_copy['timeout_info'] = direct_timeouts[rep_num]
                d_copy['_timeout_propagated'] = True
            all_data.append(d_copy)

        if not all_data:
            self.metrics_group.setVisible(False)
            return

        self.metrics_group.setVisible(True)

        # Thresholds empírics (de 98 rèpliques analitzades)
        FWHM_THRESHOLD = 1.5  # FWHM > 1.5 min = sospitós
        CR_COLUMN_MIN = 0.4   # CR < 0.4 = massa altres pics (COLUMN)
        CR_BP_MIN = 0.95      # CR < 0.95 = no esperat (BP)
        SHIFT_DIRECT_MAX = 50  # Shift > 50s = warning (DIRECT)
        SHIFT_UIB_MAX = 30     # Shift > 30s = warning (UIB)
        SYM_MIN, SYM_MAX = 0.5, 2.5  # Simetria fora rang = asimètric

        for khp in all_data:
            row = self.metrics_table.rowCount()
            self.metrics_table.insertRow(row)

            # Extreure dades
            filename = khp.get('filename', '?')
            signal = khp.get('_signal', '?')
            is_bp = khp.get('is_bp', False)

            # Col 0: Rep (R1, R2...)
            display_name = filename
            if '_R' in filename:
                display_name = 'R' + filename.split('_R')[-1].split('.')[0].split('_')[0]
            self.metrics_table.setItem(row, 0, QTableWidgetItem(display_name))

            # Col 1: Senyal
            self.metrics_table.setItem(row, 1, QTableWidgetItem(signal))

            # Col 2: Àrea DOC
            area = khp.get('area', 0)
            self.metrics_table.setItem(row, 2, QTableWidgetItem(f"{area:.0f}"))

            # Col 3: DOC/254 (ratio àrees)
            a254_area = khp.get('a254_area', 0)
            doc_254_ratio = khp.get('a254_doc_ratio', 0)
            if doc_254_ratio <= 0 and a254_area > 0 and area > 0:
                doc_254_ratio = area / a254_area
            item_doc254 = QTableWidgetItem(f"{doc_254_ratio:.2f}" if doc_254_ratio > 0 else "-")
            self.metrics_table.setItem(row, 3, item_doc254)

            # Col 4: FWHM (amb color si fora rang)
            fwhm = khp.get('fwhm_doc', 0)
            item_fwhm = QTableWidgetItem(f"{fwhm:.2f}" if fwhm > 0 else "-")
            if fwhm > FWHM_THRESHOLD:
                item_fwhm.setBackground(QColor(255, 200, 100))  # Taronja
                item_fwhm.setToolTip(f"FWHM elevat (>{FWHM_THRESHOLD} min)")
            self.metrics_table.setItem(row, 4, item_fwhm)

            # Col 5: RF_MASS (Àrea / µg DOC)
            rf_mass = khp.get('rf_mass_doc', 0) or khp.get('rf_mass', 0)
            self.metrics_table.setItem(row, 5, QTableWidgetItem(f"{rf_mass:.1f}" if rf_mass > 0 else "-"))

            # Col 6: CR (Concentration Ratio amb color segons mode)
            cr = khp.get('concentration_ratio', khp.get('cr_doc', 0))
            item_cr = QTableWidgetItem(f"{cr:.2f}" if cr > 0 else "-")
            if cr > 0:
                if is_bp and cr < CR_BP_MIN:
                    item_cr.setBackground(QColor(255, 200, 100))
                    item_cr.setToolTip(f"CR baix per BP (esperat >{CR_BP_MIN})")
                elif not is_bp and cr < CR_COLUMN_MIN:
                    item_cr.setBackground(QColor(255, 200, 100))
                    item_cr.setToolTip(f"CR baix (esperat >{CR_COLUMN_MIN})")
            self.metrics_table.setItem(row, 6, item_cr)

            # Col 7: t_max
            peak_info = khp.get('peak_info', {})
            t_max = khp.get('t_retention', 0) or peak_info.get('t_max', 0) or khp.get('t_doc_max', 0)
            self.metrics_table.setItem(row, 7, QTableWidgetItem(f"{t_max:.2f}" if t_max > 0 else "-"))

            # Col 8: Shift (informatiu, no penalitza)
            shift_sec = khp.get('shift_sec', khp.get('shift_min', 0) * 60)
            item_shift = QTableWidgetItem(f"{shift_sec:.1f}")
            self.metrics_table.setItem(row, 8, item_shift)

            # Col 9: SNR
            snr = khp.get('snr', 0)
            item_snr = QTableWidgetItem(f"{snr:.0f}" if snr > 0 else "-")
            if 0 < snr < 10:
                item_snr.setBackground(QColor(255, 200, 100))
            self.metrics_table.setItem(row, 9, item_snr)

            # Col 10: Simetria (amb color si fora rang)
            symmetry = khp.get('symmetry', 0)
            item_sym = QTableWidgetItem(f"{symmetry:.2f}" if symmetry > 0 else "-")
            if symmetry > 0 and (symmetry < SYM_MIN or symmetry > SYM_MAX):
                item_sym.setBackground(QColor(255, 200, 100))
                item_sym.setToolTip(f"Asimètric (rang normal: {SYM_MIN}-{SYM_MAX})")
            self.metrics_table.setItem(row, 10, item_sym)

            # Col 11: R² bigaussian fit
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
                item_r2.setToolTip(f"Fit {bg_status}\nR²={r2:.4f}\nAsimetria={asym:.2f}")
            else:
                item_r2 = QTableWidgetItem("-")
            self.metrics_table.setItem(row, 11, item_r2)

            # Col 12: Pic_J (antic Batman)
            has_batman = khp.get('has_batman', False)
            item_picj = QTableWidgetItem("!" if has_batman else "-")
            if has_batman:
                item_picj.setBackground(QColor(255, 150, 150))
                item_picj.setToolTip("Pic_J: pic amb vall (artefacte) - INVALID")
            self.metrics_table.setItem(row, 12, item_picj)

            # Col 13: Timeout (color segons si afecta pic o no)
            has_timeout = khp.get('has_timeout', False)
            timeout_info = khp.get('timeout_info', {})
            timeouts_list = timeout_info.get('timeouts', [])
            affects_peak = self._timeout_affects_peak(khp)

            if has_timeout and timeouts_list:
                first_to = timeouts_list[0]
                t_start = first_to.get('t_start_min', 0)
                item_to = QTableWidgetItem(f"{t_start:.1f}")
                tooltip = f"TO@{t_start:.1f}min"
                if affects_peak:
                    item_to.setBackground(QColor(255, 100, 100))
                    tooltip += " - AFECTA PIC! (+100)"
                else:
                    # Timeout fora pic: color neutre, no penalitza
                    item_to.setBackground(QColor(220, 220, 220))
                    tooltip += " (fora pic, OK)"
                item_to.setToolTip(tooltip)
            else:
                item_to = QTableWidgetItem("-")
            self.metrics_table.setItem(row, 13, item_to)

            # Col 14: Pics en zona ±4 min
            n_pics = self._count_peaks_in_zone(khp, zone_min=4.0)
            item_pics = QTableWidgetItem(str(n_pics))
            if n_pics > 1:
                item_pics.setBackground(QColor(255, 150, 150))
                item_pics.setToolTip(f"Múltiples pics ({n_pics}) en zona ±4min - INVALID")
            else:
                item_pics.setBackground(QColor(150, 255, 150))
            self.metrics_table.setItem(row, 14, item_pics)

            # Calcular Quality Score amb nova lògica
            quality, issues = self._calculate_quality_score(khp, signal)

            # Col 15: Quality Score
            item_q = QTableWidgetItem(str(int(quality)))
            if quality >= 100:
                item_q.setBackground(QColor(255, 150, 150))
            elif quality > 50:
                item_q.setBackground(QColor(255, 200, 100))
            elif quality > 20:
                item_q.setBackground(QColor(255, 255, 150))
            else:
                item_q.setBackground(QColor(150, 255, 150))
            self.metrics_table.setItem(row, 15, item_q)

            # Col 16: Estat
            valid_for_cal = khp.get('valid_for_calibration', True)
            if not valid_for_cal or quality >= 100:
                status = "INVALID"
                color = QColor(255, 150, 150)
            elif quality > 50:
                status = "CHECK"
                color = QColor(255, 200, 100)
            elif quality > 20:
                status = "INFO"
                color = QColor(255, 255, 150)
            else:
                status = "OK"
                color = QColor(150, 255, 150)
            item_status = QTableWidgetItem(status)
            item_status.setBackground(color)
            if issues:
                item_status.setToolTip("\n".join(issues))
            self.metrics_table.setItem(row, 16, item_status)

    def _update_replica_selection(self, result):
        """Actualitza la secció de selecció de rèpliques."""
        # Obtenir dades KHP (Direct prioritari, sinó UIB)
        khp_data = result.get("khp_data_direct") or result.get("khp_data_uib")

        if not khp_data:
            self.replica_selection_group.setVisible(False)
            return

        # Obtenir info de selecció i comparació
        selection = khp_data.get('selection') or {}
        comparison = khp_data.get('replica_comparison') or {}
        replicas = khp_data.get('replicas') or []

        if not replicas or len(replicas) < 1:
            self.replica_selection_group.setVisible(False)
            return

        self.replica_selection_group.setVisible(True)

        # === Actualitzar etiqueta de selecció actual ===
        method = selection.get('method', 'unknown')
        selected = selection.get('selected_replicas', [])
        is_manual = selection.get('is_manual', False)
        reason = selection.get('reason', '')

        if method == 'average':
            sel_text = f"Mitjana de R{'+R'.join(map(str, selected))}"
        elif method == 'single':
            sel_text = "Única rèplica disponible"
        elif method == 'best_quality':
            sel_text = f"Millor qualitat: R{selected[0] if selected else '?'}"
        elif method.startswith('R'):
            sel_text = f"Manual: {method}"
        else:
            sel_text = f"{method} ({selected})"

        if is_manual:
            sel_text += " [MANUAL]"

        if reason and reason not in sel_text:
            sel_text += f" - {reason}"

        self.selection_info_label.setText(f"Selecció: {sel_text}")

        # === Actualitzar combo de selecció ===
        self.replica_selection_combo.blockSignals(True)
        self.replica_selection_combo.clear()

        n_replicas = len(replicas)
        current_method = selection.get('method', 'average')

        # Opcions disponibles
        options = []
        if n_replicas > 1:
            options.append(("Mitjana (automàtic)", "average"))
            options.append(("Millor qualitat (automàtic)", "best_quality"))
        for i in range(n_replicas):
            options.append((f"Només R{i+1}", f"R{i+1}"))

        for label, value in options:
            self.replica_selection_combo.addItem(label, value)

        # Seleccionar l'opció actual
        for i in range(self.replica_selection_combo.count()):
            if self.replica_selection_combo.itemData(i) == current_method:
                self.replica_selection_combo.setCurrentIndex(i)
                break

        self.replica_selection_combo.blockSignals(False)
        self.replica_selection_combo.currentIndexChanged.connect(self._on_selection_combo_changed)
        self.apply_selection_btn.setEnabled(False)

        # === Actualitzar taula de comparació ===
        self.replica_comparison_table.setRowCount(0)

        # Obtenir detalls de rèpliques
        replica_details = comparison.get('replica_details', [])
        if not replica_details:
            # Construir des de replicas si no hi ha replica_details
            replica_details = []
            for i, rep in enumerate(replicas):
                peak_info = rep.get('peak_info', {})
                replica_details.append({
                    'replica_num': i + 1,
                    'area': rep.get('area', 0),
                    't_max': peak_info.get('t_max', 0) or rep.get('t_doc_max', 0),
                    'snr': rep.get('snr', 0),
                    'symmetry': rep.get('symmetry', 0),
                    'a254_doc_ratio': rep.get('a254_doc_ratio', 0),
                    'shift_sec': rep.get('shift_sec', 0),
                    'quality_score': rep.get('quality_score', 0),
                })

        for i, rep in enumerate(replica_details):
            row = self.replica_comparison_table.rowCount()
            self.replica_comparison_table.insertRow(row)

            rep_num = rep.get('replica_num', i + 1)
            is_selected = rep_num in selected

            # Col 0: Rèplica
            item = QTableWidgetItem(f"R{rep_num}")
            if is_selected:
                item.setBackground(QColor('#D5F5E3'))
                item.setFont(QFont("Segoe UI", 9, QFont.Bold))
            self.replica_comparison_table.setItem(row, 0, item)

            # Col 1: Àrea
            area = rep.get('area', 0)
            item = QTableWidgetItem(f"{area:.1f}" if area > 0 else "-")
            if is_selected:
                item.setBackground(QColor('#D5F5E3'))
            self.replica_comparison_table.setItem(row, 1, item)

            # Col 2: t_max
            t_max = rep.get('t_max', 0)
            item = QTableWidgetItem(f"{t_max:.2f}" if t_max > 0 else "-")
            if is_selected:
                item.setBackground(QColor('#D5F5E3'))
            self.replica_comparison_table.setItem(row, 2, item)

            # Col 3: SNR
            snr = rep.get('snr', 0)
            item = QTableWidgetItem(f"{snr:.0f}" if snr > 0 else "-")
            if is_selected:
                item.setBackground(QColor('#D5F5E3'))
            self.replica_comparison_table.setItem(row, 3, item)

            # Col 4: Symmetry
            sym = rep.get('symmetry', 0)
            item = QTableWidgetItem(f"{sym:.2f}" if sym > 0 else "-")
            if is_selected:
                item.setBackground(QColor('#D5F5E3'))
            self.replica_comparison_table.setItem(row, 4, item)

            # Col 5: DOC/254
            ratio = rep.get('a254_doc_ratio', 0)
            item = QTableWidgetItem(f"{ratio:.2f}" if ratio > 0 else "-")
            if is_selected:
                item.setBackground(QColor('#D5F5E3'))
            self.replica_comparison_table.setItem(row, 5, item)

            # Col 6: Shift
            shift = rep.get('shift_sec', 0)
            item = QTableWidgetItem(f"{shift:.1f}s" if shift != 0 else "-")
            if is_selected:
                item.setBackground(QColor('#D5F5E3'))
            self.replica_comparison_table.setItem(row, 6, item)

            # Col 7: Quality Score
            q = rep.get('quality_score', 0)
            item = QTableWidgetItem(f"{q:.0f}")
            if q >= 100:
                item.setBackground(QColor('#FADBD8'))
            elif q > 50:
                item.setBackground(QColor('#FCF3CF'))
            if is_selected:
                item.setBackground(QColor('#D5F5E3'))
            self.replica_comparison_table.setItem(row, 7, item)

            # Col 8: Status amb ComboBox (C08: permetre canviar manualment)
            is_outlier = rep.get('is_outlier', False)
            status_combo = QComboBox()
            status_combo.addItems(["✓ Vàlida", "✗ Outlier"])

            if is_outlier:
                status_combo.setCurrentIndex(1)
                status_combo.setStyleSheet("""
                    QComboBox {
                        color: #C0392B; background: #FADBD8; font-weight: bold;
                        border: 1px solid #E74C3C; border-radius: 3px;
                        padding: 2px 4px; min-width: 90px;
                    }
                """)
            elif is_selected:
                status_combo.setCurrentIndex(0)
                status_combo.setStyleSheet("""
                    QComboBox {
                        color: #27AE60; background: #D5F5E3; font-weight: bold;
                        border: 1px solid #27AE60; border-radius: 3px;
                        padding: 2px 4px; min-width: 90px;
                    }
                """)
            else:
                status_combo.setCurrentIndex(0)
                status_combo.setStyleSheet("""
                    QComboBox {
                        border: 1px solid #BDC3C7; border-radius: 3px;
                        padding: 2px 4px; min-width: 90px;
                    }
                """)

            # Guardar referència a la rèplica per poder-la actualitzar
            status_combo.setProperty("replica_num", rep.get('replica_num', row + 1))
            status_combo.currentIndexChanged.connect(
                lambda idx, r=rep.get('replica_num', row + 1): self._on_replica_status_changed(r, idx)
            )
            self.replica_comparison_table.setCellWidget(row, 8, status_combo)

        # === Actualitzar etiqueta diferències ===
        if comparison.get('comparable') and len(replica_details) >= 2:
            rsd = comparison.get('rsd_area', 0)
            diff_area = comparison.get('diff_area_pct', 0)
            diff_t = comparison.get('diff_t_max_sec', 0)
            diff_shift = comparison.get('diff_shift_sec', 0)
            pearson = comparison.get('pearson_profiles')

            diff_parts = [
                f"RSD àrea: {rsd:.1f}%",
                f"Δ àrea: {diff_area:.1f}%",
                f"Δ t_max: {diff_t:.1f}s",
            ]
            if diff_shift > 0:
                diff_parts.append(f"Δ shift: {diff_shift:.1f}s")
            if pearson is not None:
                diff_parts.append(f"Pearson perfils: {pearson:.4f}")

            self.replica_diff_label.setText("Diferències entre rèpliques: " + " | ".join(diff_parts))
        else:
            self.replica_diff_label.setText("")

    def _on_replica_status_changed(self, replica_num, status_idx):
        """Handler quan canvia l'estat d'una rèplica via dropdown (C08)."""
        is_outlier = (status_idx == 1)  # 0 = Vàlida, 1 = Outlier

        try:
            from hpsec_calibrate import load_local_calibrations, save_local_calibrations
            import os

            seq_path = self.main_window.seq_path
            if not seq_path:
                return

            calibrations = load_local_calibrations(seq_path)
            seq_name = os.path.basename(seq_path)

            # Buscar la calibració actual i actualitzar la rèplica
            updated = False
            for cal in calibrations:
                if cal.get('seq_name') != seq_name:
                    continue

                # Actualitzar replicas_info
                replicas_info = cal.get('replicas_info', [])
                for rep in replicas_info:
                    if rep.get('replica_num') == replica_num:
                        rep['is_outlier'] = is_outlier
                        updated = True
                        break

                # Actualitzar replica_comparison si existeix
                replica_comp = cal.get('replica_comparison', {})
                replica_details = replica_comp.get('replica_details', [])
                for rep in replica_details:
                    if rep.get('replica_num') == replica_num:
                        rep['is_outlier'] = is_outlier
                        updated = True
                        break

            if updated:
                save_local_calibrations(seq_path, calibrations)
                action = "marcada com a Outlier" if is_outlier else "restaurada com a Vàlida"
                self.main_window.set_status(f"Rèplica R{replica_num} {action}", 3000)

        except Exception as e:
            print(f"[ERROR] Error canviant estat rèplica: {e}")

    def _on_selection_combo_changed(self):
        """Handler quan canvia la selecció al combo."""
        self.apply_selection_btn.setEnabled(True)

    def _on_apply_replica_selection(self):
        """Aplica la nova selecció de rèpliques."""
        if not self.calibration_data:
            return

        new_method = self.replica_selection_combo.currentData()
        if not new_method:
            return

        # Importar funció
        from hpsec_calibrate import set_replica_selection

        seq_path = self.main_window.seq_path
        khp_data = self.calibration_data.get("khp_data_direct") or self.calibration_data.get("khp_data_uib")

        if not khp_data:
            QMessageBox.warning(self, "Error", "No hi ha dades KHP per modificar")
            return

        # Obtenir cal_id (de calibration o khp_data)
        calibration = self.calibration_data.get('calibration', {})
        cal_id = calibration.get('cal_id')

        if not cal_id:
            QMessageBox.warning(self, "Error", "No s'ha trobat l'ID de calibració")
            return

        # Aplicar canvi
        result = set_replica_selection(seq_path, cal_id, new_method, user="gui")

        if result.get('success'):
            QMessageBox.information(
                self, "Selecció actualitzada",
                f"{result.get('message')}\n\n"
                f"Nou àrea: {result.get('changes', {}).get('new_area', 0):.1f}\n"
                f"Anterior: {result.get('changes', {}).get('old_area', 0):.1f}"
            )

            # Actualitzar dades i refrescar vista
            updated_entry = result.get('entry', {})
            if updated_entry:
                # Actualitzar khp_data amb nova selecció
                for key in ['area', 'rf', 'shift_sec', 'shift_min', 'a254_doc_ratio', 'selection']:
                    if key in updated_entry:
                        khp_data[key] = updated_entry[key]

                # Refrescar vistes
                self._update_summary(self.calibration_data)
                self._update_replica_selection(self.calibration_data)
                self._update_history(self.calibration_data)

            self.apply_selection_btn.setEnabled(False)
        else:
            QMessageBox.warning(self, "Error", result.get('message', 'Error desconegut'))

    def _on_mark_replica_outlier(self):
        """Marca/desmarca la rèplica seleccionada com a outlier."""
        # Obtenir fila seleccionada
        selected_rows = self.replica_comparison_table.selectedItems()
        if not selected_rows:
            QMessageBox.information(self, "Selecciona rèplica",
                "Selecciona una fila de la taula per marcar-la com a outlier.")
            return

        row = selected_rows[0].row()
        replica_item = self.replica_comparison_table.item(row, 0)
        if not replica_item:
            return

        replica_name = replica_item.text()

        # Obtenir estat actual d'outlier (columna 8 = Status)
        status_item = self.replica_comparison_table.item(row, 8)
        is_currently_outlier = status_item and "Outlier" in status_item.text()

        # Confirmar acció
        action = "desmarcar" if is_currently_outlier else "marcar"
        reply = QMessageBox.question(
            self, f"Confirmar {action} outlier",
            f"Vols {action} la rèplica '{replica_name}' com a outlier?\n\n"
            f"{'Tornarà a ser vàlida per calibrar.' if is_currently_outlier else 'No es farà servir per calibrar.'}",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        # Aplicar canvi
        try:
            from hpsec_calibrate import load_local_calibrations, save_local_calibrations
            import os

            seq_path = self.main_window.seq_path
            if not seq_path:
                return

            calibrations = load_local_calibrations(seq_path)
            seq_name = os.path.basename(seq_path)

            # Buscar la calibració actual i actualitzar la rèplica
            updated = False
            for cal in calibrations:
                if cal.get('seq_name') != seq_name:
                    continue

                # Actualitzar replicas_info
                replicas_info = cal.get('replicas_info', [])
                for rep in replicas_info:
                    if rep.get('filename', '') == replica_name or f"R{replicas_info.index(rep)+1}" == replica_name:
                        rep['is_outlier'] = not is_currently_outlier
                        updated = True
                        break

                # Actualitzar replica_comparison si existeix
                replica_comp = cal.get('replica_comparison', {})
                replica_details = replica_comp.get('replica_details', [])
                for rep in replica_details:
                    rep_num = rep.get('replica_num', 0)
                    if f"R{rep_num}" == replica_name:
                        rep['is_outlier'] = not is_currently_outlier
                        updated = True
                        break

            if updated:
                save_local_calibrations(seq_path, calibrations)

                # Refrescar vista
                self._update_replica_selection(self.calibration_data)

                QMessageBox.information(
                    self, "Actualitzat",
                    f"Rèplica '{replica_name}' {'desmarcada' if is_currently_outlier else 'marcada'} com a outlier."
                )
            else:
                QMessageBox.warning(self, "Error", "No s'ha pogut trobar la rèplica.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error actualitzant: {str(e)}")

    def _update_validation(self, result):
        """Actualiza la sección de validación amb warnings separats per senyal i rèplica."""
        errors = result.get("errors", [])

        # Recopilar quality_issues PER SENYAL I RÈPLICA (no deduplicar)
        issues_by_signal = {"Direct": {}, "UIB": {}}
        direct_timeouts = []  # Per propagar a UIB

        # Processar Direct
        khp_data_direct = result.get("khp_data_direct")
        if khp_data_direct:
            replicas = self._extract_all_replicas(khp_data_direct)
            for d in replicas:
                rep_name = d.get('filename', 'R?')
                # Extreure número de rèplica
                import re
                match = re.search(r'R(\d+)', rep_name)
                rep_num = f"R{match.group(1)}" if match else rep_name

                issues = d.get('quality_issues', [])
                if issues:
                    if rep_num not in issues_by_signal["Direct"]:
                        issues_by_signal["Direct"][rep_num] = []
                    issues_by_signal["Direct"][rep_num].extend(issues)

                # Guardar timeouts de Direct per propagar a UIB
                if d.get('has_timeout'):
                    timeout_info = d.get('timeout_info', {})
                    timeouts = timeout_info.get('timeouts', [])
                    for to in timeouts:
                        direct_timeouts.append({
                            't_start': to.get('t_start_min', 0),
                            't_end': to.get('t_end_min', 0),
                            'replica': rep_num
                        })

        # Processar UIB
        khp_data_uib = result.get("khp_data_uib")
        if khp_data_uib:
            replicas = self._extract_all_replicas(khp_data_uib)
            for d in replicas:
                rep_name = d.get('filename', 'R?')
                import re
                match = re.search(r'R(\d+)', rep_name)
                rep_num = f"R{match.group(1)}" if match else rep_name

                issues = d.get('quality_issues', [])
                if issues:
                    if rep_num not in issues_by_signal["UIB"]:
                        issues_by_signal["UIB"][rep_num] = []
                    issues_by_signal["UIB"][rep_num].extend(issues)

                # Propagar timeouts de Direct a UIB si no ja detectat
                uib_has_timeout = d.get('has_timeout', False)
                for dt in direct_timeouts:
                    if dt['replica'] == rep_num and not uib_has_timeout:
                        if rep_num not in issues_by_signal["UIB"]:
                            issues_by_signal["UIB"][rep_num] = []
                        issues_by_signal["UIB"][rep_num].append(
                            f"⚠ TimeOut ({dt['t_start']:.1f} min)"
                        )

        # Comptar total issues
        total_issues = sum(
            len(issues)
            for signal_issues in issues_by_signal.values()
            for issues in signal_issues.values()
        )

        if not errors and total_issues == 0:
            self.validation_group.setVisible(False)
            return

        self.validation_group.setVisible(True)

        html = ""

        if errors:
            html += "<span style='color: red;'><b>Errors:</b></span><ul>"
            for e in errors:
                html += f"<li>{e}</li>"
            html += "</ul>"

        if total_issues > 0:
            html += "<span style='color: orange;'><b>Problemes de qualitat:</b></span>"

            # Mostrar per senyal
            for signal_name in ["Direct", "UIB"]:
                signal_issues = issues_by_signal[signal_name]
                if signal_issues:
                    html += f"<br><b style='color: #2874A6;'>{signal_name}:</b><ul>"
                    for rep_num, issues in sorted(signal_issues.items()):
                        for issue in issues:
                            html += f"<li><b>{rep_num}:</b> {issue}</li>"
                    html += "</ul>"

        self.validation_label.setText(html)

    def _update_history(self, result):
        """Actualiza la comparación histórica con taula i gràfic."""
        import os
        import re

        seq_path = self.main_window.seq_path or ""
        current_seq = os.path.basename(seq_path).replace('_SEQ', '').replace('_BP', '') if seq_path else ""

        # Determinar mètode (BP o COLUMN)
        method = "COLUMN"
        khp_data = result.get("khp_data") or result.get("khp_data_direct") or result.get("khp_data_uib")
        if khp_data and khp_data.get('is_bp', False):
            method = "BP"
        elif self.main_window.imported_data:
            if self.main_window.imported_data.get("method", "").upper() == "BP":
                method = "BP"

        # Obtenir paràmetres de filtre
        khp_conc = result.get("khp_conc", 5)

        # Obtenir volum d'injecció actual
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

        # Inicialitzar
        self._history_data = []
        self.history_table.setRowCount(0)
        self.select_cal_btn.setEnabled(False)

        try:
            history = load_khp_history(seq_path)
            if not history:
                self.history_graph.clear()
                self.calibration_line_graph.clear()
                self.history_group.setVisible(False)
                return

            # Decidir si incloure outliers
            include_outliers = self.show_outliers_cb.isChecked()

            filtered_history = []
            for cal in history:
                # Sempre excloure calibracions sense àrea
                if cal.get('area', 0) <= 0:
                    continue

                # Excloure outliers si no està marcat el checkbox
                if not include_outliers and cal.get('is_outlier', False):
                    continue

                # Aplicar filtres per condicions iguals (mode/conc/volum)
                cal_mode = cal.get('mode', 'COLUMN')
                cal_conc = cal.get('conc_ppm', 0)
                cal_vol = cal.get('volume_uL', current_volume)

                # Filtres: mode exacte, conc ±1, volum exacte (o si no hi ha volum registrat)
                if cal_mode != method:
                    continue
                if abs(cal_conc - khp_conc) >= 1:
                    continue
                if cal_vol and current_volume and cal_vol != current_volume:
                    continue

                filtered_history.append(cal)

            # Deduplicar: si hi ha múltiples entrades per la mateixa SEQ+condició,
            # mantenir només la més recent (última de la llista)
            seen_seqs = {}
            for cal in filtered_history:
                key = cal.get('seq_name', '') + '_' + cal.get('condition_key', '')
                seen_seqs[key] = cal  # L'última sobreescriu
            filtered_history = list(seen_seqs.values())

            if not filtered_history:
                self.history_graph.clear()
                self.calibration_line_graph.clear()
                self.history_group.setVisible(False)
                self.history_filters_label.setText("")
                return

            # Mostrar filtres aplicats
            outlier_text = " (amb outliers)" if include_outliers else ""
            self.history_filters_label.setText(
                f"<b>Filtres:</b> {method} · KHP{khp_conc:.0f}ppm · {int(current_volume)}µL{outlier_text} ({len(filtered_history)})"
            )

            self._history_data = filtered_history
            self.history_group.setVisible(True)

            # Ordenar per número de SEQ
            def get_seq_num(cal):
                match = re.search(r'(\d+)', cal.get('seq_name', ''))
                return int(match.group(1)) if match else 0
            filtered_history.sort(key=get_seq_num)

            # Identificar índexs vàlids (per gràfic)
            valid_indices = set()

            # Omplir taula
            for idx, cal in enumerate(filtered_history):
                row = self.history_table.rowCount()
                self.history_table.insertRow(row)

                # Dades
                cal_seq_raw = cal.get('seq_name', 'N/A').replace('_SEQ', '').replace('_BP', '')
                # Mostrar nom SEQ simplificat (ja filtrat per condicions iguals)
                cal_seq = cal_seq_raw

                area = cal.get('area', 0)
                conc = cal.get('conc_ppm', 0)
                rf = area / conc if conc > 0 else 0  # RF = Response Factor (Àrea/ppm)

                # t_max (t_retention és l'equivalent guardat)
                t_max = cal.get('t_retention', 0) or cal.get('t_max', 0)
                if not t_max:
                    peak_info = cal.get('peak_info', {})
                    t_max = peak_info.get('t_max', 0)

                shift_sec = cal.get('shift_sec', 0)
                snr = cal.get('snr', 0)
                symmetry = cal.get('symmetry', 0)
                doc_254_ratio = cal.get('a254_doc_ratio', 0)
                cr = cal.get('concentration_ratio', 0)

                # Usar quality_score_v2 si disponible (nova lògica)
                quality_score = cal.get('quality_score_v2', cal.get('quality_score', 0))
                quality_issues = cal.get('quality_issues_v2', cal.get('quality_issues', []))
                status_v2 = cal.get('status_v2', '')

                # Determinar si és vàlid amb nova lògica
                stored_valid = cal.get('valid_for_calibration', True)
                stored_outlier = cal.get('is_outlier', False)

                # Usar status_v2 si disponible, sinó calcular
                if status_v2:
                    is_valid = status_v2 not in ['INVALID', 'CHECK']
                elif quality_score >= 100:
                    is_valid = False
                else:
                    is_valid = stored_valid and not stored_outlier

                is_current = (cal_seq_raw == current_seq)

                if is_valid and not is_current:
                    valid_indices.add(idx)

                # Estat
                if is_current:
                    status_text = "ACTUAL"
                elif status_v2 == 'INVALID':
                    status_text = "INVALID"
                elif not is_valid:
                    status_text = "EXCLÒS"
                elif status_v2:
                    status_text = status_v2
                else:
                    status_text = "OK"

                # Motiu d'exclusió
                motiu = ""
                if quality_issues:
                    motiu = ", ".join(quality_issues) if isinstance(quality_issues, list) else str(quality_issues)
                elif not is_valid:
                    motiu = "Outlier" if stored_outlier else "Invalid"

                # Mode display
                mode_display = "BP" if cal_mode == "BP" else "COL"

                # Cel·les: SEQ, Mode, Àrea, t_max, SNR, DOC/254, CR, Sym, Shift, Q, Estat, Motiu
                items = [
                    cal_seq,
                    mode_display,
                    f"{area:.0f}",
                    f"{t_max:.2f}" if t_max and t_max > 0 else "-",
                    f"{snr:.0f}" if snr and snr > 0 else "-",
                    f"{doc_254_ratio:.2f}" if doc_254_ratio and doc_254_ratio > 0 else "-",
                    f"{cr:.2f}" if cr and cr > 0 else "-",
                    f"{symmetry:.2f}" if symmetry and symmetry > 0 else "-",
                    f"{shift_sec:.1f}" if shift_sec else "-",
                    f"{quality_score:.0f}" if quality_score else "0",
                    status_text,
                    motiu
                ]

                for col, text in enumerate(items):
                    item = QTableWidgetItem(text)
                    item.setTextAlignment(Qt.AlignCenter)

                    # Tooltip amb detalls
                    if col == 0:
                        item.setToolTip(
                            f"SEQ: {cal_seq_raw}\n"
                            f"Mode: {cal_mode}\n"
                            f"KHP: {cal_conc:.0f} ppm\n"
                            f"Volum: {cal_vol} µL"
                        )

                    # Fons segons estat
                    if is_current:
                        item.setBackground(QColor("#D5F5E3"))  # Verd clar = actual
                        item.setFont(QFont("Segoe UI", 9, QFont.Bold))
                    elif status_text == "INVALID":
                        item.setBackground(QColor("#F5B7B1"))  # Vermell clar
                        item.setForeground(QColor("#666666"))
                    elif not is_valid:
                        item.setBackground(QColor("#FADBD8"))  # Vermell clar = exclòs
                        item.setForeground(QColor("#888888"))

                    # Quality score amb colors (col 9)
                    if col == 9:
                        if quality_score >= 100:
                            item.setBackground(QColor("#F5B7B1"))  # Vermell
                        elif quality_score >= 50:
                            item.setBackground(QColor("#FCF3CF"))  # Groc
                        elif quality_score > 20:
                            item.setBackground(QColor("#FEF9E7"))  # Groc clar
                        else:
                            item.setBackground(QColor("#D5F5E3"))  # Verd clar

                    # Columna Estat amb colors (col 10)
                    if col == 10:
                        if status_text == "ACTUAL":
                            item.setBackground(QColor("#85C1E9"))  # Blau
                        elif status_text == "OK":
                            item.setBackground(QColor("#D5F5E3"))  # Verd
                        elif status_text == "INVALID":
                            item.setBackground(QColor("#F5B7B1"))  # Vermell
                        elif status_text == "EXCLÒS":
                            item.setBackground(QColor("#EBEDEF"))  # Gris

                    self.history_table.setItem(row, col, item)

            # Gràfic de barres (passar valid_indices per colors correctes)
            self.history_graph.plot_history(filtered_history, current_seq, valid_indices)

            # Gràfic de recta de calibració (amb QC_History)
            try:
                qc_history = load_qc_history()
                config = get_config()
                rf_mass_cal = get_rf_mass_cal(signal='direct', mode=method.lower())
                self.calibration_line_graph.plot_calibration(
                    qc_history=qc_history,
                    current_seq_name=current_seq,
                    rf_mass_cal=rf_mass_cal or 682,
                    warning_pct=config.get('calibration', 'qc_thresholds', 'warning_pct', default=5.0),
                    fail_pct=config.get('calibration', 'qc_thresholds', 'fail_pct', default=10.0),
                    n_context=config.get('calibration', 'qc_thresholds', 'n_seqs_context', default=2)
                )
            except Exception as e:
                print(f"Error plotant gràfic calibració: {e}")
                self.calibration_line_graph.clear()

            # Resum
            n_valid = len(valid_indices)
            n_excluded = len(filtered_history) - n_valid

            if n_valid > 0:
                valid_areas = [filtered_history[i].get('area', 0) for i in valid_indices]
                mean_area = np.mean(valid_areas)
                std_area = np.std(valid_areas) if len(valid_areas) > 1 else 0
                self.history_summary.setText(
                    f"{n_valid} vàlides · {n_excluded} excloses · "
                    f"Mitjana: {mean_area:.0f} ± {std_area:.0f}"
                )
                # Habilitar botó de mitjana si hi ha més d'una calibració vàlida
                self.use_average_btn.setEnabled(n_valid >= 2)
            else:
                self.history_summary.setText(
                    f"{n_excluded} calibracions (totes excloses de la mitjana)"
                )
                self.use_average_btn.setEnabled(False)

        except Exception as e:
            import traceback
            print(f"[WARNING] Error carregant històric: {e}")
            traceback.print_exc()
            self.history_graph.clear()
            self.calibration_line_graph.clear()
            self.history_group.setVisible(False)

    def _on_history_selection_changed(self):
        """Handler quan es selecciona una fila de l'històric."""
        row = self.history_table.currentRow()
        has_selection = row >= 0

        self.select_cal_btn.setEnabled(has_selection)
        self.toggle_outlier_btn.setEnabled(has_selection)

        # Actualitzar text del botó outlier segons l'estat actual de la fila seleccionada
        if has_selection and hasattr(self, '_history_data') and row < len(self._history_data):
            cal = self._history_data[row]
            # Comprovar si és outlier (manual o automàtic)
            is_manual_outlier = cal.get('manual_outlier', False)
            is_auto_outlier = cal.get('is_outlier', False)
            snr = cal.get('snr', 0)
            quality_score = cal.get('quality_score', 0)

            # Re-avaluar com a _update_history: SNR=0 i quality_score<=40 es considera vàlid
            if snr == 0 and quality_score <= 40:
                is_valid = not is_manual_outlier  # Només manual pot fer-lo outlier
            else:
                is_valid = not is_manual_outlier and not is_auto_outlier

            is_outlier = not is_valid
            self.toggle_outlier_btn.setText("Desmarcar Outlier" if is_outlier else "Marcar Outlier")
        else:
            self.toggle_outlier_btn.setText("Marcar Outlier")

    def _show_history_legend(self):
        """Mostra diàleg amb llegenda i detalls del gràfic d'històric (C16)."""
        from PySide6.QtWidgets import QMessageBox

        legend_html = """
<h3>Llegenda del Gràfic QA/QC Històric</h3>

<p><b>Què fa el QA/QC KHP:</b></p>
<p>Verifica la mesura del KHP respecte la calibració global (rf_mass_cal)
i determina el time shift necessari per a la quantificació.</p>

<p><b>Colors de les barres:</b></p>
<ul>
<li><span style='color:#27AE60'>■ Verd</span> - SEQ actual (oberta)</li>
<li><span style='color:#5DADE2'>■ Blau</span> - Verificacions vàlides</li>
<li><span style='color:#E74C3C'>■ Vermell</span> - Outliers (exclosos de la mitjana)</li>
</ul>

<p><b>Línies horitzontals:</b></p>
<ul>
<li><span style='color:#27AE60'>━━━</span> Mitjana de verificacions vàlides</li>
<li><span style='color:#27AE60'>- - -</span> Desviació estàndard (±1σ)</li>
</ul>

<p><b>Criteris per marcar Outlier automàtic:</b></p>
<ul>
<li>Àrea fora del rang mitjana ± 2σ</li>
<li>Qualitat (Q) > 100 punts</li>
<li>SNR < 50</li>
</ul>

<p><i>Nota: Pots marcar/desmarcar outliers manualment amb el botó "Marcar Outlier"</i></p>
"""
        msg = QMessageBox(self)
        msg.setWindowTitle("Llegenda Gràfic Històric")
        msg.setTextFormat(Qt.RichText)
        msg.setText(legend_html)
        msg.setIcon(QMessageBox.Information)
        msg.exec()

    def _on_show_outliers_changed(self, state):
        """Handler quan canvia el checkbox d'incloure outliers."""
        if self.calibration_data:
            self._update_history(self.calibration_data)

    def _apply_selected_calibration(self):
        """Aplica la calibració seleccionada de la taula i mostra el report complet."""
        from PySide6.QtWidgets import QMessageBox
        import os

        row = self.history_table.currentRow()
        if row < 0 or not hasattr(self, '_history_data') or row >= len(self._history_data):
            return

        cal = self._history_data[row]
        area = cal.get('area', 0)
        conc = cal.get('conc_ppm', 5)
        seq_name = cal.get('seq_name', 'N/A')

        if area <= 0:
            QMessageBox.warning(self, "Error", "Calibració sense àrea vàlida.")
            return

        rf = area / conc  # RF = Response Factor (Àrea/ppm)

        # === ACTUALITZAR INFORMACIÓ GENERAL ===
        seq_path = self.main_window.seq_path or ""
        current_seq = os.path.basename(seq_path) if seq_path else "-"
        self.result_labels["seq_name"].setText(current_seq)
        self.result_labels["mode"].setText(cal.get('doc_mode', cal.get('mode', 'N/A')))
        self.result_labels["khp_source"].setText(f"ALTERNATIU: {seq_name}")
        self.result_labels["khp_conc"].setText(f"{conc:.0f} ppm")
        self.result_labels["volume"].setText(f"{cal.get('volume_uL', '-')} µL" if cal.get('volume_uL') else "-")
        self.result_labels["n_replicas"].setText(str(cal.get('n_replicas', 1)))

        # === ACTUALITZAR SECCIÓ DIRECT (amb dades de l'històric) ===
        # Nota: l'històric pot no tenir separació Direct/UIB, mostrem el que tenim
        self.direct_group.setVisible(True)
        self.result_labels["rf_direct"].setText(f"{rf:.0f}")

        # Shift (en segons, amb minuts entre parèntesi)
        shift_sec = cal.get('shift_sec', 0)
        shift_min = shift_sec / 60 if shift_sec else cal.get('shift_min', 0)
        self.result_labels["shift_direct"].setText(f"{shift_sec:.1f} s ({shift_min:.3f} min)")

        # SNR i t_max
        snr = cal.get('snr', 0)
        t_retention = cal.get('t_retention', 0)
        self.result_labels["snr_direct"].setText(f"{snr:.0f}" if snr else "-")
        self.result_labels["tmax_direct"].setText(f"{t_retention:.2f} min" if t_retention else "-")

        # Amagar UIB si no tenim dades separades
        self.uib_group.setVisible(False)

        # === ACTUALITZAR SECCIÓ DE VALIDACIÓ AMB QUALITY ISSUES ===
        quality_issues = cal.get('quality_issues', []) or cal.get('calibration_issues', [])
        quality_score = cal.get('quality_score', 0)

        if quality_issues or quality_score > 50:
            self.validation_group.setVisible(True)
            html = f"<b>Calibració històrica: {seq_name}</b><br><br>"

            # Mètriques clau
            doc_254 = cal.get('a254_doc_ratio', 0)
            symmetry = cal.get('symmetry', 0)

            html += "<b>Mètriques:</b><ul>"
            if t_retention:
                html += f"<li>t_max: {t_retention:.2f} min</li>"
            if snr:
                html += f"<li>SNR: {snr:.0f}</li>"
            if doc_254:
                html += f"<li>DOC/254: {doc_254:.2f}</li>"
            if symmetry:
                html += f"<li>Simetria: {symmetry:.2f}</li>"
            html += f"<li>Quality Score: {quality_score}</li>"
            html += "</ul>"

            if quality_issues:
                html += "<span style='color: orange;'><b>Issues detectats:</b></span><ul>"
                for issue in quality_issues:
                    html += f"<li>{issue}</li>"
                html += "</ul>"

            self.validation_label.setText(html)
        else:
            self.validation_group.setVisible(False)

        # Actualitzar dades internes (CRÍTIC: RF i shift han de propagar correctament)
        new_rf = area / conc if conc > 0 else 0  # RF = area/conc (Response Factor)

        # Crear calibration_data si no existeix
        if not self.calibration_data:
            self.calibration_data = {"success": True}

        # Actualitzar RF (Response Factor)
        self.calibration_data["rf_direct"] = new_rf
        self.calibration_data["rf_uib"] = cal.get('rf_uib', new_rf)  # Usar rf_uib de l'històric si disponible
        self.calibration_data["rf"] = new_rf  # Compatibilitat

        # Actualitzar SHIFT (IMPORTANT: propagar correctament)
        self.calibration_data["shift_direct"] = shift_min  # En minuts
        self.calibration_data["shift_uib"] = cal.get('shift_uib_min', shift_min)  # En minuts
        self.calibration_data["shift"] = shift_min  # Compatibilitat

        # Altres metadades
        self.calibration_data["khp_source"] = f"ALTERNATIU: {seq_name}"
        self.calibration_data["alternative_cal"] = cal
        self.calibration_data["khp_conc"] = conc
        self.calibration_data["success"] = True

        # Propagar a main_window
        self.main_window.calibration_data = self.calibration_data

        print(f"[DEBUG] Calibració aplicada: RF={new_rf:.0f}, shift_direct={shift_min:.4f} min")

        QMessageBox.information(
            self, "Calibració Aplicada",
            f"Aplicada calibració de {seq_name}\n\n"
            f"Àrea: {area:.0f}\n"
            f"RF (Àrea/ppm): {rf:.0f}\n"
            f"Quality Score: {quality_score}"
        )

    def _toggle_outlier(self):
        """Marca o desmarca la calibració seleccionada com a outlier."""
        from PySide6.QtWidgets import QMessageBox, QInputDialog
        import json
        from datetime import datetime

        row = self.history_table.currentRow()
        if row < 0 or not hasattr(self, '_history_data') or row >= len(self._history_data):
            return

        cal = self._history_data[row]
        seq_name = cal.get('seq_name', 'N/A')
        current_outlier = cal.get('is_outlier', False) or cal.get('manual_outlier', False)

        if current_outlier:
            # Desmarcar outlier
            reply = QMessageBox.question(
                self, "Desmarcar Outlier",
                f"Vols desmarcar '{seq_name}' com a outlier?\n\n"
                f"Tornarà a incloure's en la mitjana.",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                cal['manual_outlier'] = False
                cal['is_outlier'] = False
                cal['outlier_reason'] = None
                self._save_outlier_change(cal, False, None)
        else:
            # Marcar com outlier - demanar motiu
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

        # Actualitzar vista
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
            # Trobar el fitxer d'històric
            seq_dir = Path(seq_path)
            history_file = None

            # Buscar a la carpeta pare (on es guarden els històrics)
            for parent in [seq_dir.parent, seq_dir.parent.parent]:
                candidate = parent / "khp_calibration_history.json"
                if candidate.exists():
                    history_file = candidate
                    break

            if not history_file:
                print(f"[WARNING] No s'ha trobat fitxer d'històric")
                return

            # Llegir i actualitzar
            with open(history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)

            # Trobar la calibració i actualitzar
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
                print(f"[INFO] Outlier actualitzat per {seq_name}: {is_outlier}")

        except Exception as e:
            print(f"[WARNING] Error guardant outlier: {e}")

    def _use_historical_average(self):
        """Calibra usant la mitjana de les calibracions vàlides AMB CONDICIONS IDÈNTIQUES."""
        from PySide6.QtWidgets import QMessageBox

        if not hasattr(self, '_history_data') or not self._history_data:
            QMessageBox.warning(self, "Error", "No hi ha històric disponible.")
            return

        # Obtenir condicions actuals (sempre filtrar per condicions idèntiques!)
        result = self.calibration_data or {}
        khp_data = result.get("khp_data") or result.get("khp_data_direct") or result.get("khp_data_uib")

        # Determinar mètode actual
        current_method = "COLUMN"
        if khp_data and khp_data.get('is_bp', False):
            current_method = "BP"
        elif self.main_window.imported_data:
            if self.main_window.imported_data.get("method", "").upper() == "BP":
                current_method = "BP"

        current_conc = result.get("khp_conc", 5)

        # Obtenir volum actual
        current_volume = None
        if khp_data:
            current_volume = khp_data.get('volume_uL')
        if not current_volume and self.main_window.imported_data:
            current_volume = self.main_window.imported_data.get('injection_volume')
        if not current_volume:
            current_volume = 400 if current_method == "COLUMN" else 100

        # Filtrar calibracions vàlides AMB CONDICIONS IDÈNTIQUES
        # (ignorem _history_data que pot tenir "mostrar tot", filtrem sempre)
        valid_cals = []
        for cal in self._history_data:
            # Primer: excloure outliers
            is_outlier = cal.get('is_outlier', False) or cal.get('manual_outlier', False)
            area = cal.get('area', 0)
            if is_outlier or area <= 0:
                continue

            # Segon: verificar condicions idèntiques
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
                "No hi ha calibracions vàlides per calcular la mitjana."
            )
            return

        # Calcular mitjanes
        areas = [c.get('area', 0) for c in valid_cals]
        concs = [c.get('conc_ppm', 5) for c in valid_cals]
        shifts = [c.get('shift_sec', 0) for c in valid_cals]
        doc_254_ratios = [c.get('a254_doc_ratio', 0) for c in valid_cals if c.get('a254_doc_ratio', 0) > 0]

        mean_area = np.mean(areas)
        std_area = np.std(areas) if len(areas) > 1 else 0
        mean_conc = np.mean(concs)
        mean_shift = np.mean(shifts)
        mean_doc_254 = np.mean(doc_254_ratios) if doc_254_ratios else 0

        rf = mean_area / mean_conc if mean_conc > 0 else 0  # RF = Response Factor (Àrea/ppm)

        # Confirmar (mostrant condicions aplicades)
        reply = QMessageBox.question(
            self, "Usar Mitjana Històrica",
            f"Calibrar amb mitjana de {len(valid_cals)} calibracions vàlides:\n\n"
            f"Condicions: {current_method} · KHP{current_conc:.0f} · {int(current_volume)}µL\n\n"
            f"Àrea mitjana: {mean_area:.0f} ± {std_area:.0f}\n"
            f"RF (Àrea/ppm): {rf:.0f}\n"
            f"Shift mitjà: {mean_shift:.1f} s ({mean_shift/60:.3f} min)\n"
            f"DOC/254 mitjà: {mean_doc_254:.2f}\n\n"
            f"Vols aplicar aquesta calibració?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        # Aplicar als nous labels
        self.result_labels["khp_source"].setText(f"MITJANA ({len(valid_cals)} cals)")
        self.result_labels["rf_direct"].setText(f"{rf:.0f}")
        self.result_labels["shift_direct"].setText(f"{mean_shift:.1f} s ({mean_shift/60:.3f} min)")
        self.direct_group.setVisible(True)
        self.uib_group.setVisible(False)

        if self.calibration_data:
            self.calibration_data["rf_direct"] = rf
            self.calibration_data["rf"] = rf
            self.calibration_data["khp_source"] = f"MITJANA HISTÒRICA ({len(valid_cals)} calibracions)"
            self.calibration_data["khp_area_direct"] = mean_area
            self.calibration_data["khp_area"] = mean_area
            self.calibration_data["shift_uib"] = mean_shift / 60
            self.calibration_data["average_cal"] = {
                "n_calibrations": len(valid_cals),
                "mean_area": mean_area,
                "std_area": std_area,
                "mean_factor": new_factor,
                "mean_shift_sec": mean_shift,
                "mean_doc_254_ratio": mean_doc_254,
                "source_seqs": [c.get('seq_name') for c in valid_cals]
            }
            self.main_window.calibration_data = self.calibration_data

        QMessageBox.information(
            self, "Calibració Aplicada",
            f"Aplicada mitjana de {len(valid_cals)} calibracions\n"
            f"Àrea: {mean_area:.0f} ± {std_area:.0f}\n"
            f"Factor: {new_factor:.6f}"
        )

    def _go_next(self):
        self.main_window.go_to_tab(2)
