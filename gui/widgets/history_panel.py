"""
HPSEC Suite - History Panel
============================

Panel per visualitzar tot l'històric de calibracions KHP i dades.
Fora de la pipeline, sense filtres aplicats.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGroupBox,
    QGridLayout, QFrame, QTableWidget, QTableWidgetItem, QHeaderView,
    QComboBox, QMessageBox, QSplitter, QTabWidget, QScrollArea,
    QSizePolicy, QCheckBox, QLineEdit
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QColor

from pathlib import Path
from datetime import datetime
import json
import os
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from hpsec_calibrate import (
    load_khp_history,
    get_active_global_calibration,
    get_rf_mass_cal,
    get_calibration_intercept,
)

import logging
import numpy as np
import re

logger = logging.getLogger(__name__)

# Font única per extreure el número de SEQ (retorna None si no n'hi ha;
# aquí es fa servir com a clau d'ordenació, per això el fallback a 0)
from hpsec_consolidate import extract_seq_number as _extract_seq_number


def extract_seq_number(seq_name):
    """Número de SEQ del nom, o 0 si no es pot extreure (clau d'ordenació)."""
    return _extract_seq_number(seq_name or "") or 0


# Matplotlib
import matplotlib  # noqa: F401
# matplotlib.use('QtAgg') eliminat: forçava el backend interactiu i obria finestres
# fantasma en generar informes. Backend fixat a Agg (hpsec_suite_qt.py); embedding
# via FigureCanvasQTAgg explícit, que no depèn del backend per defecte.
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class HistoryPanel(QWidget):
    """Panel per visualitzar històric complet de calibracions."""

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self._all_calibrations = []
        self._setup_ui()

    def _setup_ui(self):
        """Configura la interfície."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        # Títol
        title = QLabel("Control de Qualitat — KHP")
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        layout.addWidget(title)

        # Info
        info = QLabel(
            "Històric complet de calibracions KHP. "
            "Filtra per mode, concentració o nom de seqüència. "
            "Clica un punt als gràfics per veure detalls."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #666;")
        layout.addWidget(info)

        # Toolbar amb filtres i accions
        toolbar = QHBoxLayout()

        # Botó refrescar
        self.refresh_btn = QPushButton("🔄 Actualitzar")
        self.refresh_btn.clicked.connect(self._load_history)
        toolbar.addWidget(self.refresh_btn)

        toolbar.addWidget(QLabel("Mode:"))
        self.mode_filter = QComboBox()
        self.mode_filter.addItem("Tots", None)
        self.mode_filter.addItem("COLUMN", "COLUMN")
        self.mode_filter.addItem("BP", "BP")
        self.mode_filter.currentIndexChanged.connect(self._apply_filters)
        toolbar.addWidget(self.mode_filter)

        toolbar.addWidget(QLabel("KHP:"))
        self.conc_filter = QComboBox()
        self.conc_filter.addItem("Totes", None)
        self.conc_filter.addItem("5 ppm", 5)
        self.conc_filter.addItem("10 ppm", 10)
        self.conc_filter.currentIndexChanged.connect(self._apply_filters)
        toolbar.addWidget(self.conc_filter)

        toolbar.addWidget(QLabel("Volum:"))
        self.vol_filter = QComboBox()
        self.vol_filter.addItem("Tots", None)
        self.vol_filter.addItem("100 µL", 100)
        self.vol_filter.addItem("400 µL", 400)
        self.vol_filter.currentIndexChanged.connect(self._apply_filters)
        toolbar.addWidget(self.vol_filter)

        toolbar.addWidget(QLabel("Sens. UIB:"))
        self.uib_sens_filter = QComboBox()
        self.uib_sens_filter.addItem("Totes", None)
        self.uib_sens_filter.addItem("700 ppb", 700)
        self.uib_sens_filter.addItem("1000 ppb", 1000)
        self.uib_sens_filter.currentIndexChanged.connect(self._apply_filters)
        toolbar.addWidget(self.uib_sens_filter)

        toolbar.addStretch()

        toolbar.addWidget(QLabel("Cercar:"))
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Nom SEQ...")
        self.search_edit.setMaximumWidth(150)
        self.search_edit.textChanged.connect(self._apply_filters)
        toolbar.addWidget(self.search_edit)

        # Checkbox mostrar només vàlids
        self.valid_only_cb = QCheckBox("Només vàlides")
        self.valid_only_cb.stateChanged.connect(self._apply_filters)
        toolbar.addWidget(self.valid_only_cb)

        layout.addLayout(toolbar)

        # Tabs: Taula + Gràfics
        self.content_tabs = QTabWidget()

        # === TAB 1: Taula ===
        table_widget = QWidget()
        table_layout = QVBoxLayout(table_widget)
        table_layout.setContentsMargins(0, 8, 0, 0)

        self.history_table = QTableWidget()
        self.history_table.setColumnCount(22)
        self.history_table.setHorizontalHeaderLabels([
            "Data", "SEQ", "Mode", "KHP", "Vol", "Sens",
            "Àrea_D", "Àrea_U", "RF_D", "RF_M",
            "t_max", "FWHM", "Sym", "SNR",
            "DOC/254", "UIB/254", "AR", "nP",
            "Sel", "Q", "Estat", "Motiu"
        ])

        # Tooltips capçaleres (veure docs/PARAMETRES_CALIBRACIO.md)
        # Columnes: Data(0), SEQ(1), Mode(2), KHP(3), Vol(4), Sens(5),
        #           Àrea_D(6), Àrea_U(7), RF_D(8), RF_M(9), t_max(10), FWHM(11),
        #           Sym(12), SNR(13), DOC/254(14), UIB/254(15), AR(16), nP(17),
        #           Sel(18), Q(19), Estat(20), Motiu(21)
        headers = self.history_table.horizontalHeader()
        self.history_table.horizontalHeaderItem(4).setToolTip("Vol: Volum d'injecció (µL)")
        self.history_table.horizontalHeaderItem(5).setToolTip("Sens: Sensibilitat UIB (ppb)")
        self.history_table.horizontalHeaderItem(6).setToolTip("Àrea_D: Àrea DOC Direct")
        self.history_table.horizontalHeaderItem(7).setToolTip("Àrea_U: Àrea DOC UIB")
        self.history_table.horizontalHeaderItem(8).setToolTip("RF_D: Response Factor Direct = Àrea/Conc")
        self.history_table.horizontalHeaderItem(9).setToolTip("RF_MASS: Àrea/µg DOC injectat - CLAU!")
        self.history_table.horizontalHeaderItem(10).setToolTip("t_max: Temps del pic màxim (min)")
        self.history_table.horizontalHeaderItem(11).setToolTip("FWHM: Full Width at Half Maximum (min)")
        self.history_table.horizontalHeaderItem(12).setToolTip("Sym: Simetria del pic")
        self.history_table.horizontalHeaderItem(13).setToolTip("SNR: Signal-to-Noise Ratio")
        self.history_table.horizontalHeaderItem(14).setToolTip("DOC/254: Ratio Àrea_DOC / Àrea_254nm (Direct)")
        self.history_table.horizontalHeaderItem(15).setToolTip("UIB/254: Ratio Àrea_UIB / Àrea_254nm")
        self.history_table.horizontalHeaderItem(16).setToolTip("AR: Area Ratio = Àrea pic / Àrea total")
        self.history_table.horizontalHeaderItem(17).setToolTip("nP: Nombre de pics detectats")
        self.history_table.horizontalHeaderItem(18).setToolTip("Sel: Selecció rèpliques")
        self.history_table.horizontalHeaderItem(19).setToolTip("Q: Quality Score (0=perfecte, ≥100=invàlid)")
        self.history_table.horizontalHeaderItem(20).setToolTip("Estat: OK, CHECK, INVALID o EXCLÒS")
        self.history_table.horizontalHeaderItem(21).setToolTip("Motiu d'exclusió o problemes")

        headers.setSectionResizeMode(QHeaderView.ResizeToContents)
        headers.setSectionResizeMode(1, QHeaderView.Stretch)  # SEQ expandeix
        self.history_table.setAlternatingRowColors(True)
        self.history_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.history_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.history_table.setSortingEnabled(True)
        self.history_table.verticalHeader().setVisible(False)

        table_layout.addWidget(self.history_table)
        self.content_tabs.addTab(table_widget, "📋 Taula")

        # === TAB 2: Gràfic Àrea ===
        area_widget = QWidget()
        area_layout = QVBoxLayout(area_widget)
        area_layout.setContentsMargins(0, 8, 0, 0)

        self.area_figure = Figure(figsize=(10, 5), dpi=100)
        self.area_canvas = FigureCanvas(self.area_figure)
        area_layout.addWidget(self.area_canvas)

        self.content_tabs.addTab(area_widget, "📊 Àrea")

        # === TAB 3: Gràfic t_max (Deriva) ===
        tmax_widget = QWidget()
        tmax_layout = QVBoxLayout(tmax_widget)
        tmax_layout.setContentsMargins(0, 8, 0, 0)

        self.tmax_figure = Figure(figsize=(10, 5), dpi=100)
        self.tmax_canvas = FigureCanvas(self.tmax_figure)
        tmax_layout.addWidget(self.tmax_canvas)

        self.content_tabs.addTab(tmax_widget, "⏱️ t_max (Deriva)")

        # === TAB 4: Gràfic RF_MASS (clau!) ===
        rfmass_widget = QWidget()
        rfmass_layout = QVBoxLayout(rfmass_widget)
        rfmass_layout.setContentsMargins(0, 8, 0, 0)

        self.rfv_figure = Figure(figsize=(10, 5), dpi=100)  # Mantenim nom intern
        self.rfv_canvas = FigureCanvas(self.rfv_figure)
        rfmass_layout.addWidget(self.rfv_canvas)

        self.content_tabs.addTab(rfmass_widget, "⚡ RF_MASS")

        # === TAB 5: Gràfic D/254 ===
        ratio_widget = QWidget()
        ratio_layout = QVBoxLayout(ratio_widget)
        ratio_layout.setContentsMargins(0, 8, 0, 0)

        self.ratio_figure = Figure(figsize=(10, 5), dpi=100)
        self.ratio_canvas = FigureCanvas(self.ratio_figure)
        ratio_layout.addWidget(self.ratio_canvas)

        self.content_tabs.addTab(ratio_widget, "📈 D/254")

        # === TAB 6: Gràfic FWHM (degradació columna) ===
        fwhm_widget = QWidget()
        fwhm_layout = QVBoxLayout(fwhm_widget)
        fwhm_layout.setContentsMargins(0, 8, 0, 0)

        self.fwhm_figure = Figure(figsize=(10, 5), dpi=100)
        self.fwhm_canvas = FigureCanvas(self.fwhm_figure)
        fwhm_layout.addWidget(self.fwhm_canvas)

        self.content_tabs.addTab(fwhm_widget, "📉 FWHM")

        # === TAB 7: Gràfic UIB Ratio (Àrea Direct / Àrea UIB) ===
        uib_widget = QWidget()
        uib_layout = QVBoxLayout(uib_widget)
        uib_layout.setContentsMargins(0, 8, 0, 0)

        self.uib_figure = Figure(figsize=(10, 5), dpi=100)
        self.uib_canvas = FigureCanvas(self.uib_figure)
        uib_layout.addWidget(self.uib_canvas)

        self.content_tabs.addTab(uib_widget, "🔬 UIB Ratio")

        # === TAB 8: Levey-Jennings (QC) ===
        lj_widget = QWidget()
        lj_layout = QVBoxLayout(lj_widget)
        lj_layout.setContentsMargins(0, 8, 0, 0)

        self.lj_figure = Figure(figsize=(10, 5), dpi=100)
        self.lj_canvas = FigureCanvas(self.lj_figure)
        lj_layout.addWidget(self.lj_canvas)

        self.content_tabs.addTab(lj_widget, "📊 Levey-Jennings")

        layout.addWidget(self.content_tabs)

        # Resum i botons d'acció
        summary_layout = QHBoxLayout()
        self.summary_label = QLabel()
        self.summary_label.setStyleSheet("color: #666;")
        summary_layout.addWidget(self.summary_label)
        summary_layout.addStretch()

        # Botó veure detall (requereix selecció)
        self.detail_btn = QPushButton("📊 Veure Detall")
        self.detail_btn.setToolTip("Mostra gràfics i dades detallades de la calibració seleccionada")
        self.detail_btn.setEnabled(False)
        self.detail_btn.clicked.connect(self._view_calibration_detail)
        summary_layout.addWidget(self.detail_btn)

        # Botó exportar
        self.export_btn = QPushButton("Exportar CSV")
        self.export_btn.clicked.connect(self._export_csv)
        summary_layout.addWidget(self.export_btn)

        layout.addLayout(summary_layout)

        # Connectar selecció de taula
        self.history_table.itemSelectionChanged.connect(self._on_table_selection_changed)

    def _on_table_selection_changed(self):
        """Gestiona canvi de selecció a la taula."""
        selected_rows = self.history_table.selectionModel().selectedRows()
        self.detail_btn.setEnabled(len(selected_rows) == 1)

    def _view_calibration_detail(self):
        """Mostra detalls de la calibració seleccionada."""
        selected_rows = self.history_table.selectionModel().selectedRows()
        if not selected_rows:
            return

        row = selected_rows[0].row()
        seq_name = self.history_table.item(row, 1).text()

        # Buscar la calibració a les dades
        cal = None
        for c in self._all_calibrations:
            if c.get('seq_name') == seq_name:
                cal = c
                break

        if not cal:
            QMessageBox.warning(self, "Error", f"No s'ha trobat la calibració per {seq_name}")
            return

        # Mostrar diàleg amb detalls
        self._show_calibration_detail_dialog(cal)

    def showEvent(self, event):
        """Carrega l'històric quan es mostra el panel."""
        super().showEvent(event)
        # Auto-seleccionar sensibilitat UIB de la SEQ actual (si n'hi ha)
        try:
            imported = getattr(self.main_window, 'imported_data', None) or {}
            current_sens = imported.get("uib_sensitivity")
            if current_sens:
                idx = self.uib_sens_filter.findData(int(current_sens))
                if idx >= 0:
                    self.uib_sens_filter.setCurrentIndex(idx)
        except Exception:
            pass
        self._load_history()

    def _load_history(self):
        """Carrega tot l'històric de calibracions des de REGISTRY/KHP_History.json."""
        self._all_calibrations = []

        # load_khp_history usa get_registry_path() que és global (no depèn de seq_path)
        try:
            history = load_khp_history(None)  # seq_path s'ignora, usa REGISTRY global
            if history:
                self._all_calibrations = history
                logger.info(f"Carregades {len(history)} calibracions des de REGISTRY")
        except Exception as e:
            logger.warning(f"Error carregant històric: {e}")

        # Excloure SEQ_CAL (ja analitzades al wizard) + eliminar duplicats
        seen = set()
        unique_cals = []
        for cal in self._all_calibrations:
            sn = cal.get('seq_name', '')
            if '_CAL' in sn.upper():
                continue
            key = (sn, cal.get('date_processed', ''))
            if key not in seen:
                seen.add(key)
                unique_cals.append(cal)
        self._all_calibrations = unique_cals

        # Ordenar per data (més recent primer)
        self._all_calibrations.sort(
            key=lambda x: x.get('date_processed', ''),
            reverse=True
        )

        # Actualitzar filtres dinàmicament
        self._update_dynamic_filters()

        self._apply_filters()

    def _update_dynamic_filters(self):
        """Actualitza els filtres amb els valors reals de les dades."""
        # Recollir valors únics
        concs = set()
        vols = set()
        uib_sens = set()

        for cal in self._all_calibrations:
            conc = cal.get('conc_ppm', 0)
            if conc > 0:
                concs.add(conc)

            vol = cal.get('volume_uL', 0)
            if vol > 0:
                vols.add(int(vol))

            sens = cal.get('uib_sensitivity')
            if sens is not None and sens > 0:
                uib_sens.add(int(sens))

        # Actualitzar combo de concentracions (preservant selecció actual)
        current_conc = self.conc_filter.currentData()
        self.conc_filter.blockSignals(True)
        self.conc_filter.clear()
        self.conc_filter.addItem("Totes", None)
        for c in sorted(concs):
            self.conc_filter.addItem(f"{c:g} ppm", c)
        if current_conc is not None:
            idx = self.conc_filter.findData(current_conc)
            if idx >= 0:
                self.conc_filter.setCurrentIndex(idx)
        self.conc_filter.blockSignals(False)

        # Actualitzar combo de volums (preservant selecció actual)
        current_vol = self.vol_filter.currentData()
        self.vol_filter.blockSignals(True)
        self.vol_filter.clear()
        self.vol_filter.addItem("Tots", None)
        for v in sorted(vols):
            self.vol_filter.addItem(f"{v} µL", v)
        if current_vol is not None:
            idx = self.vol_filter.findData(current_vol)
            if idx >= 0:
                self.vol_filter.setCurrentIndex(idx)
        self.vol_filter.blockSignals(False)

        # Actualitzar combo de sensibilitat UIB (preservant selecció actual)
        current_sens = self.uib_sens_filter.currentData()
        self.uib_sens_filter.blockSignals(True)
        self.uib_sens_filter.clear()
        self.uib_sens_filter.addItem("Totes", None)
        for s in sorted(uib_sens):
            self.uib_sens_filter.addItem(f"{s} ppb", s)
        if current_sens is not None:
            idx = self.uib_sens_filter.findData(current_sens)
            if idx >= 0:
                self.uib_sens_filter.setCurrentIndex(idx)
        self.uib_sens_filter.blockSignals(False)

    def _apply_filters(self):
        """Aplica els filtres i actualitza la taula."""
        mode_filter = self.mode_filter.currentData()
        conc_filter = self.conc_filter.currentData()
        vol_filter = self.vol_filter.currentData()
        uib_sens_filter = self.uib_sens_filter.currentData()
        search_text = self.search_edit.text().strip().lower()
        valid_only = self.valid_only_cb.isChecked()

        filtered = []
        for cal in self._all_calibrations:
            # Filtre mode
            if mode_filter and cal.get('mode', 'COLUMN') != mode_filter:
                continue

            # Filtre concentració (tolerància relativa 10% per agrupar valors similars)
            if conc_filter:
                cal_conc = cal.get('conc_ppm', 0)
                tol = max(0.01, conc_filter * 0.1)
                if abs(cal_conc - conc_filter) > tol:
                    continue

            # Filtre volum
            if vol_filter and cal.get('volume_uL', 0) != vol_filter:
                continue

            # Filtre sensibilitat UIB
            if uib_sens_filter:
                cal_sens = cal.get('uib_sensitivity')
                if cal_sens is None or cal_sens != uib_sens_filter:
                    continue

            # Filtre cerca
            if search_text:
                seq_name = cal.get('seq_name', '').lower()
                if search_text not in seq_name:
                    continue

            # Filtre només vàlides
            if valid_only:
                is_valid = cal.get('valid_for_calibration', True)
                is_outlier = cal.get('is_outlier', False) or cal.get('manual_outlier', False)
                if not is_valid or is_outlier:
                    continue

            filtered.append(cal)

        self._populate_table(filtered)

    def _populate_table(self, calibrations):
        """Omple la taula amb les calibracions filtrades."""
        self.history_table.setSortingEnabled(False)
        self.history_table.setRowCount(0)

        for cal in calibrations:
            row = self.history_table.rowCount()
            self.history_table.insertRow(row)

            # Col 0: Data
            date_str = cal.get('date_processed', '')[:10]
            self.history_table.setItem(row, 0, QTableWidgetItem(date_str))

            # Col 1: SEQ
            seq_name = cal.get('seq_name', 'N/A')
            self.history_table.setItem(row, 1, QTableWidgetItem(seq_name))

            # Col 2: Mode
            mode = cal.get('mode', 'COLUMN')
            mode_item = QTableWidgetItem(mode)
            if mode == 'BP':
                mode_item.setForeground(QColor('#E67E22'))
            else:
                mode_item.setForeground(QColor('#3498DB'))
            self.history_table.setItem(row, 2, mode_item)

            # Col 3: KHP conc
            conc = cal.get('conc_ppm', 0)
            self.history_table.setItem(row, 3, QTableWidgetItem(f"{conc:g}"))

            # Col 4: Volum
            vol = cal.get('volume_uL', 0)
            self.history_table.setItem(row, 4, QTableWidgetItem(f"{vol}" if vol else "-"))

            # Col 5: Sensibilitat UIB
            uib_sens = cal.get('uib_sensitivity')
            self.history_table.setItem(row, 5, QTableWidgetItem(f"{uib_sens}" if uib_sens else "-"))

            # Col 6: Àrea Direct
            area = cal.get('area', 0)
            self.history_table.setItem(row, 6, QTableWidgetItem(f"{area:.0f}"))

            # Col 7: Àrea UIB
            area_u = cal.get('area_u', 0)
            self.history_table.setItem(row, 7, QTableWidgetItem(f"{area_u:.0f}" if area_u > 0 else "-"))

            # Col 8: RF_D (Response Factor Direct)
            rf = cal.get('rf', 0)
            self.history_table.setItem(row, 8, QTableWidgetItem(f"{rf:.1f}" if rf > 0 else "-"))

            # Col 9: RF_MASS (RF normalitzat per massa = àrea/µg DOC)
            rf_mass = cal.get('rf_mass', 0)
            self.history_table.setItem(row, 9, QTableWidgetItem(f"{rf_mass:.1f}" if rf_mass > 0 else "-"))

            # Col 10: t_max
            t_max = cal.get('t_retention', 0)
            self.history_table.setItem(row, 10, QTableWidgetItem(f"{t_max:.2f}" if t_max > 0 else "-"))

            # Col 11: FWHM
            fwhm = cal.get('fwhm_doc', 0)
            fwhm_item = QTableWidgetItem(f"{fwhm:.2f}" if fwhm > 0 else "-")
            if fwhm > 1.5:
                fwhm_item.setBackground(QColor('#FCF3CF'))
            self.history_table.setItem(row, 11, fwhm_item)

            # Col 12: Sym
            sym = cal.get('symmetry', 0)
            self.history_table.setItem(row, 12, QTableWidgetItem(f"{sym:.2f}" if sym > 0 else "-"))

            # Col 13: SNR
            snr = cal.get('snr', 0)
            snr_item = QTableWidgetItem(f"{snr:.0f}" if snr > 0 else "-")
            if snr > 0 and snr < 10:
                snr_item.setBackground(QColor('#FCF3CF'))
            self.history_table.setItem(row, 13, snr_item)

            # Col 14: DOC/254 (Ratio Àrea_DOC_Direct / Àrea_254nm)
            d254_d = cal.get('d254_d', 0)
            self.history_table.setItem(row, 14, QTableWidgetItem(f"{d254_d:.1f}" if d254_d > 0 else "-"))

            # Col 15: UIB/254 (Ratio Àrea_DOC_UIB / Àrea_254nm)
            d254_u = cal.get('d254_u', 0)
            self.history_table.setItem(row, 15, QTableWidgetItem(f"{d254_u:.1f}" if d254_u > 0 else "-"))

            # Col 16: AR (Area Ratio = àrea pic / àrea total)
            ar = cal.get('area_ratio', 0)
            ar_item = QTableWidgetItem(f"{ar:.2f}" if ar > 0 else "-")
            if ar > 0 and ar < 0.7:
                ar_item.setBackground(QColor('#FCF3CF'))
            self.history_table.setItem(row, 16, ar_item)

            # Col 17: nP (nombre de pics)
            n_peaks = cal.get('n_peaks', 1)
            np_item = QTableWidgetItem(f"{n_peaks}")
            if n_peaks > 1:
                np_item.setBackground(QColor('#FCF3CF'))
            self.history_table.setItem(row, 17, np_item)

            # Col 18: Selecció rèpliques
            selection = cal.get('selection', {})
            sel_method = selection.get('method', 'legacy')
            sel_replicas = selection.get('selected_replicas', [])
            is_manual = selection.get('is_manual', False)
            n_reps = selection.get('n_replicas_available', cal.get('n_replicas', 1))

            if sel_method == 'average':
                sel_text = f"Avg({n_reps})"
            elif sel_method == 'single':
                sel_text = "R1"
            elif sel_method == 'best_quality':
                sel_text = f"R{sel_replicas[0] if sel_replicas else '?'}*"
            elif sel_method.startswith('R'):
                sel_text = sel_method
            elif sel_method == 'legacy':
                sel_text = "-"
            else:
                sel_text = f"R{'+'.join(map(str, sel_replicas))}" if sel_replicas else "-"

            if is_manual:
                sel_text += "[M]"

            sel_item = QTableWidgetItem(sel_text)
            sel_item.setToolTip(
                f"Mètode: {sel_method}\n"
                f"Rèpliques: {sel_replicas}\n"
                f"Disponibles: {n_reps}\n"
                f"Manual: {'Sí' if is_manual else 'No'}"
            )
            if is_manual:
                sel_item.setBackground(QColor('#AED6F1'))
            self.history_table.setItem(row, 18, sel_item)

            # Col 19: R² bigaussiana
            bg_doc = cal.get('bigaussian_doc') or {}
            r2_bg = bg_doc.get('r2', 0)
            if r2_bg > 0:
                q_item = QTableWidgetItem(f"{r2_bg:.3f}")
                if r2_bg < 0.95:
                    q_item.setBackground(QColor('#F5B7B1'))
                elif r2_bg < 0.98:
                    q_item.setBackground(QColor('#FCF3CF'))
                else:
                    q_item.setBackground(QColor('#D5F5E3'))
            else:
                q_item = QTableWidgetItem("-")
            self.history_table.setItem(row, 19, q_item)

            # Col 20: Estat (basat en calibration_anomalies)
            is_valid = cal.get('valid_for_calibration', True)
            is_outlier = cal.get('is_outlier', False) or cal.get('manual_outlier', False)
            cal_anoms = cal.get('calibration_anomalies', [])
            has_blocker = any(a.get('severity') == 'blocker' for a in cal_anoms if isinstance(a, dict))

            if has_blocker or not is_valid:
                status = "INVALID"
                status_color = QColor('#F5B7B1')
            elif is_outlier:
                status = "EXCLÒS"
                status_color = QColor('#EBEDEF')
            elif cal_anoms:
                status = "WARNING"
                status_color = QColor('#FCF3CF')
            else:
                status = "OK"
                status_color = QColor('#D5F5E3')

            status_item = QTableWidgetItem(status)
            status_item.setBackground(status_color)
            self.history_table.setItem(row, 20, status_item)

            # Col 21: Motiu (de calibration_anomalies)
            motiu = ""
            if cal_anoms:
                labels = [a.get('label', a.get('code', '')) for a in cal_anoms if isinstance(a, dict)]
                motiu = ", ".join(labels)
            elif is_outlier:
                motiu = "Outlier manual"
            elif not is_valid:
                motiu = "Calibració invàlida"

            motiu_item = QTableWidgetItem(motiu)
            if motiu:
                motiu_item.setToolTip(motiu)
            self.history_table.setItem(row, 21, motiu_item)

        self.history_table.setSortingEnabled(True)

        # Resum
        n_total = len(self._all_calibrations)
        n_filtered = len(calibrations)
        n_valid = sum(1 for c in calibrations
                      if c.get('valid_for_calibration', True)
                      and not c.get('is_outlier', False)
                      and not c.get('manual_outlier', False))

        self.summary_label.setText(
            f"Mostrant {n_filtered} de {n_total} calibracions · "
            f"{n_valid} vàlides"
        )

        # Actualitzar gràfics
        self._update_graphs(calibrations)

    def _get_all_interventions(self):
        """Obté tots els events (manteniment + canvis metodològics) directament dels fitxers."""
        from gui.widgets.maintenance_panel import METHOD_LOG_PATH, METHOD_COLORS, MaintenancePanel
        TASK_CATEGORIES = MaintenancePanel.TASK_CATEGORIES
        import json as _json

        events = []
        try:
            # 1) Canvis metodològics des de JSON
            if METHOD_LOG_PATH.exists():
                with open(METHOD_LOG_PATH, 'r', encoding='utf-8') as f:
                    for mc in _json.load(f):
                        events.append({
                            'date': mc.get('date', ''),
                            'category': mc.get('category', 'Canvi protocol'),
                            'color': METHOD_COLORS.get(mc.get('category'), '#8E44AD'),
                        })
        except Exception as e:
            logger.debug(f"Error llegint method_log: {e}")

        try:
            # 2) Events de manteniment des d'Excel
            from hpsec_config import get_config
            import pandas as pd
            excel_path = get_config().get("paths", "maintenance_excel", default="")
            if excel_path and os.path.exists(excel_path):
                df = pd.read_excel(excel_path, engine='openpyxl')
                for _, row in df.iterrows():
                    date_val = row.get('Data Execució')
                    if pd.isna(date_val):
                        continue
                    date_str = date_val.strftime("%Y-%m-%d") if hasattr(date_val, 'strftime') else str(date_val)[:10]
                    tasca = str(row.get('tasca', '')).strip()
                    if pd.isna(tasca) or not tasca or tasca == 'nan':
                        continue
                    # Categoritzar
                    tasca_lower = tasca.lower()
                    category, color = tasca, "#7F8C8D"
                    for pattern, (cat, col) in TASK_CATEGORIES.items():
                        if pattern in tasca_lower:
                            category, color = cat, col
                            break
                    events.append({'date': date_str, 'category': category, 'color': color})
        except Exception as e:
            logger.debug(f"Error llegint Excel manteniment: {e}")

        return events

    def _add_maintenance_markers(self, ax, x_seq_numbers, dates, y_min, y_max):
        """
        Afegeix marcadors d'intervencions a una gràfica. Retorna llista de (category, color) usats.

        Args:
            ax: Axes de matplotlib
            x_seq_numbers: Array de seq_numbers (eix X real)
            dates: Llista de dates YYYY-MM-DD (alineades amb x_seq_numbers)
            y_min, y_max: Limits de l'eix Y
        Returns:
            List of (category, color) tuples for legend building
        """
        interventions = self._get_all_interventions()
        if not interventions or len(dates) == 0:
            return []

        added = []
        for event in interventions:
            event_date = event.get('date', '')
            if not event_date:
                continue

            category = event['category']
            color = event['color']

            # Interpolar posició X entre seq_numbers basant-se en dates
            for i in range(len(dates)):
                if event_date <= dates[i]:
                    if i > 0 and event_date > dates[i-1]:
                        # Interpolar entre x[i-1] i x[i]
                        x_pos = (x_seq_numbers[i-1] + x_seq_numbers[i]) / 2.0
                        if not any(abs(x_pos - pos) < 1 for pos, _, _ in added):
                            added.append((x_pos, category, color))
                    break

        # Dibuixar
        for x_pos, category, color in added:
            ax.axvline(x=x_pos, color=color, linestyle='--', linewidth=1.2, alpha=0.6, zorder=1)

        # Retornar categories usades (per llegenda)
        seen = set()
        used = []
        for _, cat, col in added:
            if cat not in seen:
                seen.add(cat)
                used.append((cat, col))
        return used

    def _build_legend(self, ax, interv_used, items=None):
        """Construeix llegenda compacta: items específics + intervencions."""
        from matplotlib.lines import Line2D

        elements = list(items or [])

        # Intervencions
        for cat, col in interv_used[:4]:
            short = cat[:12] + '..' if len(cat) > 14 else cat
            elements.append(Line2D([0],[0], linestyle='--', color=col, linewidth=1.2, label=short))

        if elements:
            ax.legend(handles=elements, loc='best', fontsize=7, framealpha=0.8)

    def _setup_pick_tooltips(self, canvas, sorted_cals, x_values, key_fn):
        """Configura tooltip al clicar un punt del scatter."""
        # Guardar referència per evitar GC
        if not hasattr(self, '_tooltip_annots'):
            self._tooltip_annots = {}

        canvas_id = id(canvas)
        self._tooltip_annots[canvas_id] = None

        def on_pick(event):
            # Netejar tooltip anterior d'aquest canvas
            if self._tooltip_annots.get(canvas_id):
                try:
                    self._tooltip_annots[canvas_id].remove()
                except Exception:
                    pass
                self._tooltip_annots[canvas_id] = None

            if not hasattr(event, 'ind') or len(event.ind) == 0:
                canvas.draw_idle()
                return

            ind = event.ind[0]
            artist = event.artist
            offsets = artist.get_offsets()
            if ind >= len(offsets):
                return

            xy = offsets[ind]
            x_val, y_val = float(xy[0]), float(xy[1])

            # Trobar la calibració corresponent: buscar per (x, y) més proper
            best_idx = None
            best_dist = float('inf')
            for ci, xv in enumerate(x_values):
                dist = abs(float(xv) - x_val)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = ci
                elif dist == best_dist and best_idx is not None:
                    # Desempatar per y si x idèntic (mateixa SEQ, diff rèplica)
                    pass

            if best_idx is None or best_idx >= len(sorted_cals):
                return

            cal = sorted_cals[best_idx]
            ax = artist.axes

            # Info d'estat
            is_valid = cal.get('valid_for_calibration', True)
            is_outlier = cal.get('is_outlier', False) or cal.get('manual_outlier', False)
            status = "OK" if (is_valid and not is_outlier) else "EXCLOSA"
            mode = cal.get('mode', '')

            text = (f"{cal.get('seq_name','')} [{mode}]\n"
                    f"{cal.get('date_processed','')[:10]}\n"
                    f"KHP {cal.get('conc_ppm',0):g} ppm · {cal.get('volume_uL',0):.0f} uL\n"
                    f"{key_fn(cal)} · {status}")

            self._tooltip_annots[canvas_id] = ax.annotate(
                text, xy=(xy[0], xy[1]), fontsize=7,
                bbox=dict(boxstyle='round,pad=0.3', fc='#FFFACD', ec='#BDC3C7', alpha=0.95),
                xytext=(15, 15), textcoords='offset points',
                zorder=10)
            canvas.draw_idle()

        def on_click(event):
            # Clicar fora d'un punt → netejar tooltip
            if self._tooltip_annots.get(canvas_id):
                try:
                    self._tooltip_annots[canvas_id].remove()
                except Exception:
                    pass
                self._tooltip_annots[canvas_id] = None
                canvas.draw_idle()

        canvas.mpl_connect('pick_event', on_pick)
        canvas.mpl_connect('button_press_event', on_click)

    def _update_graphs(self, calibrations):
        """
        Actualitza tots els gràfics amb les calibracions filtrades.

        Millores:
        - Eix X ordenat per número de seqüència (ordre cronològic real)
        - Símbols: BP=quadrat ('s'), COLUMN=cercle ('o')
        - Colors per condicions (volum, concentració, sensibilitat)
        - Nou gràfic UIB Ratio
        """
        from matplotlib.lines import Line2D

        # Ordenar per número de seqüència (ordre cronològic)
        sorted_cals = sorted(
            calibrations,
            key=lambda x: extract_seq_number(x.get('seq_name', ''))
        )

        # Preparar dades
        dates = []
        seq_names = []
        seq_numbers = []
        areas_d = []      # Àrea Direct
        areas_u = []      # Àrea UIB
        t_maxs = []
        doc_254s = []
        rf_masses = []    # RF normalitzat per massa
        fwhms = []
        volumes = []
        concs = []        # Concentració KHP
        uib_sens = []     # Sensibilitat UIB
        modes = []
        is_valids = []

        for cal in sorted_cals:
            date_str = cal.get('date_processed', '')[:10]
            dates.append(date_str)

            seq_name = cal.get('seq_name', 'N/A')
            seq_names.append(seq_name.replace('_SEQ', ''))
            seq_numbers.append(extract_seq_number(seq_name))

            areas_d.append(cal.get('area', 0))
            areas_u.append(cal.get('area_u', 0))
            t_maxs.append(cal.get('t_retention', 0))
            doc_254s.append(cal.get('d254_d', 0))

            vol = cal.get('volume_uL', 0)
            volumes.append(vol)
            rf_masses.append(cal.get('rf_mass', 0))
            fwhms.append(cal.get('fwhm_doc', 0))
            concs.append(cal.get('conc_ppm', 0))
            uib_sens.append(cal.get('uib_sensitivity', 0) or 0)
            modes.append(cal.get('mode', 'COLUMN'))

            is_valid = cal.get('valid_for_calibration', True)
            is_outlier = cal.get('is_outlier', False) or cal.get('manual_outlier', False)
            is_valids.append(is_valid and not is_outlier)

        # Convertir a numpy
        areas_d = np.array(areas_d)
        areas_u = np.array(areas_u)
        t_maxs = np.array(t_maxs)
        doc_254s = np.array(doc_254s)
        rf_masses = np.array(rf_masses)
        fwhms = np.array(fwhms)
        volumes = np.array(volumes)
        concs = np.array(concs)
        uib_sens = np.array(uib_sens)

        x = np.array(seq_numbers, dtype=float)

        # =====================================================================
        # HELPER: Obtenir marcador per mode (BP=quadrat, COLUMN=cercle)
        # =====================================================================
        def get_marker(mode):
            return 's' if mode == 'BP' else 'o'

        # =====================================================================
        # HELPER: Obtenir color per condicions
        # =====================================================================
        # Mapa de colors per volum (tots els volums reals)
        VOL_COLORS = {
            50: '#E74C3C',   # Vermell
            100: '#9B59B6',  # Lila
            200: '#E67E22',  # Taronja
            400: '#1ABC9C',  # Verd
        }
        VOL_COLOR_DEFAULT = '#3498DB'  # Blau (altres)
        INVALID_COLOR = '#CCCCCC'

        def get_color_by_volume(vol, valid):
            if not valid:
                return INVALID_COLOR
            return VOL_COLORS.get(int(vol), VOL_COLOR_DEFAULT)

        CONC_COLORS = {1: '#3498DB', 2: '#27AE60', 5: '#E67E22'}
        CONC_COLOR_DEFAULT = '#9B59B6'

        def get_color_by_conc(conc, valid):
            if not valid:
                return INVALID_COLOR
            # Agrupar per valor proper (tolerància 10%)
            for ref_conc, color in CONC_COLORS.items():
                if abs(conc - ref_conc) <= max(0.01, ref_conc * 0.1):
                    return color
            return CONC_COLOR_DEFAULT

        def get_color_by_sensitivity(sens, valid):
            if not valid:
                return '#CCCCCC'
            elif sens == 700:
                return '#27AE60'  # Verd
            elif sens == 1000:
                return '#E74C3C'  # Vermell
            else:
                return '#3498DB'  # Blau

        def get_color_by_mode(mode, valid):
            if not valid:
                return '#CCCCCC'
            elif mode == 'BP':
                return '#E67E22'  # Taronja
            else:
                return '#3498DB'  # Blau

        # =====================================================================
        # HELPER: Scatter plot amb símbols i colors personalitzats
        # =====================================================================
        def scatter_with_markers(ax, x_vals, y_vals, modes_list, colors_list, valid_mask,
                                show_invalid=False):
            """Scatter plot amb símbols BP=quadrat, COLUMN=cercle.
            Si show_invalid=True, mostra invàlids amb marker 'x' semitransparent."""
            for i, (xi, yi, mode, color) in enumerate(zip(x_vals, y_vals, modes_list, colors_list)):
                if yi <= 0:
                    continue
                if valid_mask[i]:
                    marker = get_marker(mode)
                    ax.scatter(xi, yi, c=color, s=70, marker=marker,
                               edgecolors='white', linewidth=0.5, zorder=3,
                               picker=5)
                elif show_invalid:
                    ax.scatter(xi, yi, c=INVALID_COLOR, s=40, marker='x',
                               linewidth=1.5, zorder=2, alpha=0.5, picker=5)

        # =====================================================================
        # Gràfic 1: Àrea (colors per mode)
        # =====================================================================
        self.area_figure.clear()
        ax1 = self.area_figure.add_subplot(111)

        colors_mode = [get_color_by_mode(m, v) for m, v in zip(modes, is_valids)]
        modes_arr = np.array(modes)

        if len(x) > 0 and any(areas_d > 0):
            valid_area_mask = areas_d > 0
            scatter_with_markers(ax1, x, areas_d, modes, colors_mode, valid_area_mask)

            # Mitjana i ±σ de vàlids
            valid_areas = areas_d[np.array(is_valids) & (areas_d > 0)]
            if len(valid_areas) > 1:
                mean_a = np.mean(valid_areas)
                std_a = np.std(valid_areas)
                ax1.axhline(mean_a, color='#27AE60', linestyle='-', linewidth=2)
                ax1.axhspan(mean_a - std_a, mean_a + std_a, alpha=0.2, color='#27AE60')

            ax1.set_xlabel("nº SEQ", fontsize=9)
            ax1.set_ylabel("Àrea", fontsize=10)
            ax1.set_title("Evolució Àrea KHP", fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3, axis='y')

            y_min, y_max = ax1.get_ylim()
            interv_used = self._add_maintenance_markers(ax1, x, dates, y_min, y_max)
            items = [
                Line2D([0],[0], marker='o', color='w', markerfacecolor='#3498DB',
                       markersize=7, label='COLUMN'),
                Line2D([0],[0], marker='s', color='w', markerfacecolor='#E67E22',
                       markersize=7, label='BP'),
            ]
            if len(valid_areas) > 1:
                items.append(Line2D([0],[0], color='#27AE60', linewidth=2,
                                    label=f'Mitjana: {np.mean(valid_areas):.0f}'))
            self._build_legend(ax1, interv_used, items=items)
            self._setup_pick_tooltips(self.area_canvas, sorted_cals, x, lambda c: f"Àrea={c.get('area',0):.0f}")
        else:
            ax1.text(0.5, 0.5, "No hi ha dades", ha='center', va='center', fontsize=12, color='gray')
            ax1.set_xlim(0, 1)
            ax1.set_ylim(0, 1)

        self.area_figure.tight_layout()
        self.area_canvas.draw()

        # =====================================================================
        # Gràfic 2: t_max (Deriva) - colors per mode, símbols per mode
        # =====================================================================
        self.tmax_figure.clear()
        ax2 = self.tmax_figure.add_subplot(111)

        valid_tmax_mask = t_maxs > 0
        if any(valid_tmax_mask):
            scatter_with_markers(ax2, x, t_maxs, modes, colors_mode, valid_tmax_mask)

            items = []
            # Tendència separada per COLUMN i BP
            for mode_name, mode_color in [('COLUMN', '#3498DB'), ('BP', '#E67E22')]:
                mask = valid_tmax_mask & (modes_arr == mode_name)
                marker = 'o' if mode_name == 'COLUMN' else 's'
                if np.sum(mask) > 2:
                    mx, mt = x[mask], t_maxs[mask]
                    z = np.polyfit(mx, mt, 1)
                    p = np.poly1d(z)
                    ax2.plot(mx, p(mx), '--', color=mode_color, alpha=0.7)
                    items.append(Line2D([0],[0], marker=marker, linestyle='--', color=mode_color,
                                        markerfacecolor=mode_color, markersize=7,
                                        label=f'{mode_name}: {z[0]*10:+.3f} min/10 SEQ'))
                elif np.sum(mask) > 0:
                    items.append(Line2D([0],[0], marker=marker, color='w',
                                        markerfacecolor=mode_color, markersize=7,
                                        label=mode_name))

            ax2.set_xlabel("nº SEQ", fontsize=9)
            ax2.set_ylabel("t_max (min)", fontsize=10)
            ax2.set_title("Deriva Temps de Pic", fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)

            y_min, y_max = ax2.get_ylim()
            interv_used = self._add_maintenance_markers(ax2, x, dates, y_min, y_max)
            self._build_legend(ax2, interv_used, items=items)
            self._setup_pick_tooltips(self.tmax_canvas, sorted_cals, x, lambda c: f"t_max={c.get('t_retention',0):.2f} min")
        else:
            ax2.text(0.5, 0.5, "No hi ha dades de t_max", ha='center', va='center',
                    fontsize=12, color='gray')
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)

        self.tmax_figure.tight_layout()
        self.tmax_canvas.draw()

        # =====================================================================
        # Gràfic 3: RF_MASS - colors per VOLUM, símbols per mode
        # =====================================================================
        self.rfv_figure.clear()
        ax3 = self.rfv_figure.add_subplot(111)

        colors_vol = [get_color_by_volume(v, val) for v, val in zip(volumes, is_valids)]
        valid_rfmass_mask = rf_masses > 0

        if any(valid_rfmass_mask):
            scatter_with_markers(ax3, x, rf_masses, modes, colors_vol, valid_rfmass_mask,
                                show_invalid=True)

            valid_mask_arr = np.array(is_valids)
            valid_rfmass = rf_masses[valid_rfmass_mask & valid_mask_arr]
            if len(valid_rfmass) > 1:
                mean_rfmass = np.mean(valid_rfmass)
                std_rfmass = np.std(valid_rfmass)
                cv_rfmass = (std_rfmass / mean_rfmass * 100) if mean_rfmass > 0 else 0
                ax3.axhline(mean_rfmass, color='#27AE60', linestyle='-', linewidth=2, alpha=0.8)
                ax3.axhspan(mean_rfmass - 2*std_rfmass, mean_rfmass + 2*std_rfmass, alpha=0.15, color='#27AE60')
                ax3.axhline(mean_rfmass * 1.1, color='#E74C3C', linestyle='--', linewidth=1, alpha=0.5)
                ax3.axhline(mean_rfmass * 0.9, color='#E74C3C', linestyle='--', linewidth=1, alpha=0.5)
                ax3.set_title(f"RF_MASS · Mitjana: {mean_rfmass:.1f} · CV: {cv_rfmass:.1f}%",
                             fontsize=12, fontweight='bold')

                # Etiquetar outliers (fora ±2σ) amb nom SEQ
                for i in range(len(x)):
                    if rf_masses[i] > 0 and valid_mask_arr[i]:
                        if abs(rf_masses[i] - mean_rfmass) > 2 * std_rfmass:
                            ax3.annotate(seq_names[i], xy=(x[i], rf_masses[i]),
                                        fontsize=6, color='#E74C3C', alpha=0.8,
                                        xytext=(5, 5), textcoords='offset points',
                                        ha='left', va='bottom')
                    elif rf_masses[i] > 0 and not valid_mask_arr[i]:
                        # Invàlids: etiquetar sempre (l'usuari vol saber què són)
                        ax3.annotate(seq_names[i], xy=(x[i], rf_masses[i]),
                                    fontsize=6, color='#999', alpha=0.7,
                                    xytext=(5, -5), textcoords='offset points',
                                    ha='left', va='top')
            else:
                ax3.set_title("RF_MASS (Àrea/µg DOC)", fontsize=12, fontweight='bold')

            ax3.set_xlabel("nº SEQ", fontsize=9)
            ax3.set_ylabel("RF_MASS (Àrea/µg DOC)", fontsize=10)
            ax3.grid(True, alpha=0.3)

            # Llegenda: només volums realment presents + mode
            items = []
            vols_present = set(int(v) for v, val in zip(volumes, is_valids)
                              if val and v > 0)
            has_col = any(m == 'COLUMN' and v for m, v in zip(modes, is_valids))
            has_bp = any(m == 'BP' and v for m, v in zip(modes, is_valids))
            for vol_val in sorted(vols_present):
                color = VOL_COLORS.get(vol_val, VOL_COLOR_DEFAULT)
                label_vol = f'{vol_val} µL'
                if has_col:
                    items.append(Line2D([0],[0], marker='o', color='w',
                                        markerfacecolor=color, markersize=7,
                                        label=f'{label_vol} · COL'))
                if has_bp:
                    items.append(Line2D([0],[0], marker='s', color='w',
                                        markerfacecolor=color, markersize=7,
                                        label=f'{label_vol} · BP'))
            # Invàlids si n'hi ha
            n_invalid = sum(1 for v, rfm in zip(is_valids, rf_masses) if not v and rfm > 0)
            if n_invalid > 0:
                items.append(Line2D([0],[0], marker='x', color=INVALID_COLOR,
                                    markersize=7, linestyle='None',
                                    label=f'Exclosa ({n_invalid})'))
            y_min, y_max = ax3.get_ylim()
            interv_used = self._add_maintenance_markers(ax3, x, dates, y_min, y_max)
            self._build_legend(ax3, interv_used, items=items)
            self._setup_pick_tooltips(self.rfv_canvas, sorted_cals, x, lambda c: f"RF={c.get('rf_mass',0):.0f}")
        else:
            ax3.text(0.5, 0.5, "No hi ha dades de RF_MASS", ha='center', va='center',
                    fontsize=12, color='gray')
            ax3.set_xlim(0, 1)
            ax3.set_ylim(0, 1)

        self.rfv_figure.tight_layout()
        self.rfv_canvas.draw()

        # =====================================================================
        # Gràfic 4: D/254 - colors per concentració KHP
        # =====================================================================
        self.ratio_figure.clear()
        ax4 = self.ratio_figure.add_subplot(111)

        colors_conc = [get_color_by_conc(c, val) for c, val in zip(concs, is_valids)]
        valid_ratio_mask = doc_254s > 0

        if any(valid_ratio_mask):
            scatter_with_markers(ax4, x, doc_254s, modes, colors_conc, valid_ratio_mask)

            valid_r = doc_254s[valid_ratio_mask]
            mean_r = np.mean(valid_r)
            std_r = np.std(valid_r) if len(valid_r) > 1 else 0
            ax4.axhline(mean_r, color='#27AE60', linestyle='-', linewidth=1.5, alpha=0.7)
            ax4.axhspan(mean_r - 2*std_r, mean_r + 2*std_r, alpha=0.1, color='#27AE60')

            ax4.set_xlabel("nº SEQ", fontsize=9)
            ax4.set_ylabel("Ratio DOC/254", fontsize=10)
            ax4.set_title(f"Ratio DOC/254nm - Mitjana: {mean_r:.2f} ± {std_r:.2f}",
                         fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.3)

            # Llegenda: concentracions presents + mode
            items = []
            concs_present = set()
            for c_val, v in zip(concs, is_valids):
                if v and c_val > 0:
                    # Agrupar per valor de referència
                    matched = False
                    for ref_c in CONC_COLORS:
                        if abs(c_val - ref_c) <= max(0.01, ref_c * 0.1):
                            concs_present.add(ref_c)
                            matched = True
                            break
                    if not matched:
                        concs_present.add(round(c_val, 2))
            for c_val in sorted(concs_present):
                color = CONC_COLORS.get(c_val, CONC_COLOR_DEFAULT)
                items.append(Line2D([0],[0], marker='o', color='w',
                                    markerfacecolor=color, markersize=7,
                                    label=f'{c_val:g} ppm'))
            has_bp = any(m == 'BP' and v for m, v in zip(modes, is_valids))
            if has_bp:
                items.append(Line2D([0],[0], marker='s', color='w',
                                    markerfacecolor='gray', markersize=7,
                                    label='BP'))
            y_min, y_max = ax4.get_ylim()
            interv_used = self._add_maintenance_markers(ax4, x, dates, y_min, y_max)
            self._build_legend(ax4, interv_used, items=items)
            self._setup_pick_tooltips(self.ratio_canvas, sorted_cals, x, lambda c: f"D/254={c.get('d254_d',0):.2f}")
        else:
            ax4.text(0.5, 0.5, "No hi ha dades de D/254", ha='center', va='center',
                    fontsize=12, color='gray')
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)

        self.ratio_figure.tight_layout()
        self.ratio_canvas.draw()

        # =====================================================================
        # Gràfic 5: FWHM - colors per mode
        # =====================================================================
        self.fwhm_figure.clear()
        ax5 = self.fwhm_figure.add_subplot(111)

        valid_fwhm_mask = fwhms > 0
        if any(valid_fwhm_mask):
            scatter_with_markers(ax5, x, fwhms, modes, colors_mode, valid_fwhm_mask)

            items = []
            # Tendència separada per COLUMN i BP
            for mode_name, mode_color in [('COLUMN', '#3498DB'), ('BP', '#E67E22')]:
                mask = valid_fwhm_mask & (modes_arr == mode_name)
                marker = 'o' if mode_name == 'COLUMN' else 's'
                if np.sum(mask) > 2:
                    mx, mfw = x[mask], fwhms[mask]
                    z = np.polyfit(mx, mfw, 1)
                    p = np.poly1d(z)
                    trend_color = mode_color
                    ax5.plot(mx, p(mx), '--', color=trend_color, alpha=0.7)
                    items.append(Line2D([0],[0], marker=marker, linestyle='--', color=trend_color,
                                        markerfacecolor=trend_color, markersize=7,
                                        label=f'{mode_name}: {z[0]*10:+.3f} min/10 SEQ'))
                elif np.sum(mask) > 0:
                    items.append(Line2D([0],[0], marker=marker, color='w',
                                        markerfacecolor=mode_color, markersize=7,
                                        label=mode_name))

            ax5.axhline(1.5, color='#E74C3C', linestyle='--', linewidth=1.5, alpha=0.7)
            items.append(Line2D([0],[0], linestyle='--', color='#E74C3C',
                                label='Límit (1.5 min)'))

            ax5.set_xlabel("nº SEQ", fontsize=9)
            ax5.set_ylabel("FWHM (min)", fontsize=10)
            ax5.set_title("FWHM (Amplada de Pic)", fontsize=12, fontweight='bold')
            ax5.grid(True, alpha=0.3)

            y_min, y_max = ax5.get_ylim()
            interv_used = self._add_maintenance_markers(ax5, x, dates, y_min, y_max)
            self._build_legend(ax5, interv_used, items=items)
            self._setup_pick_tooltips(self.fwhm_canvas, sorted_cals, x, lambda c: f"FWHM={c.get('fwhm_doc',0):.2f} min")
        else:
            ax5.text(0.5, 0.5, "No hi ha dades de FWHM", ha='center', va='center',
                    fontsize=12, color='gray')
            ax5.set_xlim(0, 1)
            ax5.set_ylim(0, 1)

        self.fwhm_figure.tight_layout()
        self.fwhm_canvas.draw()

        # =====================================================================
        # Gràfic 6: UIB Ratio (Àrea Direct / Àrea UIB)
        #           2 subplots separats per sensibilitat (700 / 1000 ppb)
        # =====================================================================
        self.uib_figure.clear()

        # Calcular ratio Direct/UIB
        uib_ratios = np.zeros(len(areas_d))
        for i in range(len(areas_d)):
            if areas_u[i] > 0:
                uib_ratios[i] = areas_d[i] / areas_u[i]

        valid_uib_mask = uib_ratios > 0

        # Separar per sensibilitat
        uib_by_sens = {}
        for i in range(len(uib_ratios)):
            if not valid_uib_mask[i]:
                continue
            sens = int(uib_sens[i]) if uib_sens[i] > 0 else 0
            if sens not in uib_by_sens:
                uib_by_sens[sens] = {'x': [], 'y': [], 'modes': [], 'valids': [], 'cals': [], 'dates': []}
            uib_by_sens[sens]['x'].append(x[i])
            uib_by_sens[sens]['y'].append(uib_ratios[i])
            uib_by_sens[sens]['modes'].append(modes[i])
            uib_by_sens[sens]['valids'].append(is_valids[i])
            uib_by_sens[sens]['cals'].append(sorted_cals[i])
            uib_by_sens[sens]['dates'].append(dates[i])

        # Sensibilitats conegudes (700, 1000); agrupar altres a "Altres"
        known_sens = sorted([s for s in uib_by_sens if s in (700, 1000)])
        other_sens = [s for s in uib_by_sens if s not in (700, 1000)]
        if other_sens:
            known_sens.append(0)  # 0 = "Altres"
            combined = {'x': [], 'y': [], 'modes': [], 'valids': [], 'cals': [], 'dates': []}
            for s in other_sens:
                for k in combined:
                    combined[k].extend(uib_by_sens[s][k])
            uib_by_sens[0] = combined

        if known_sens:
            n_subplots = len(known_sens)
            axes_uib = self.uib_figure.subplots(1, n_subplots, sharey=True)
            if n_subplots == 1:
                axes_uib = [axes_uib]

            SENS_COLORS = {700: '#27AE60', 1000: '#E74C3C', 0: '#7F8C8D'}
            all_cals_uib, all_x_uib = [], []

            for ax_idx, sens_val in enumerate(known_sens):
                ax = axes_uib[ax_idx]
                sd = uib_by_sens[sens_val]
                x_s = np.array(sd['x'], dtype=float)
                y_s = np.array(sd['y'])
                modes_s = sd['modes']
                valids_s = sd['valids']
                color = SENS_COLORS.get(sens_val, '#7F8C8D')

                colors_s = [color if v else '#CCCCCC' for v in valids_s]
                scatter_with_markers(ax, x_s, y_s, modes_s, colors_s, np.ones(len(x_s), dtype=bool))

                valid_r = y_s[np.array(valids_s)]
                if len(valid_r) > 1:
                    mean_r = np.mean(valid_r)
                    std_r = np.std(valid_r)
                    ax.axhline(mean_r, color=color, linestyle='-', linewidth=2, alpha=0.7)
                    ax.axhspan(mean_r - std_r, mean_r + std_r, alpha=0.12, color=color)

                    # Tendència per mode
                    items = []
                    for mode_name, mode_color in [('COLUMN', '#2980B9'), ('BP', '#E67E22')]:
                        marker = 'o' if mode_name == 'COLUMN' else 's'
                        mask = np.array([m == mode_name and v for m, v in zip(modes_s, valids_s)])
                        if np.sum(mask) > 2:
                            z = np.polyfit(x_s[mask], y_s[mask], 1)
                            p = np.poly1d(z)
                            ax.plot(x_s[mask], p(x_s[mask]), '--', color=mode_color, alpha=0.6)
                            items.append(Line2D([0],[0], marker=marker, linestyle='--',
                                                color=mode_color, markerfacecolor=mode_color,
                                                markersize=6, label=mode_name))
                        elif np.sum(mask) > 0:
                            items.append(Line2D([0],[0], marker=marker, color='w',
                                                markerfacecolor=mode_color, markersize=6,
                                                label=mode_name))

                    sens_label = f"{sens_val} ppb" if sens_val > 0 else "Altres"
                    ax.set_title(f"UIB {sens_label} — D/U={mean_r:.2f}±{std_r:.2f} (n={len(valid_r)})",
                                fontsize=9, fontweight='bold')
                else:
                    items = []
                    sens_label = f"{sens_val} ppb" if sens_val > 0 else "Altres"
                    ax.set_title(f"UIB {sens_label}", fontsize=9, fontweight='bold')

                ax.set_xlabel("nº SEQ", fontsize=9)
                if ax_idx == 0:
                    ax.set_ylabel("Ratio D/U", fontsize=9)
                ax.grid(True, alpha=0.3)

                y_min, y_max = ax.get_ylim()
                interv_used = self._add_maintenance_markers(ax, x_s, sd['dates'], y_min, y_max)
                self._build_legend(ax, interv_used, items=items)

                all_cals_uib.extend(sd['cals'])
                all_x_uib.extend(sd['x'])

            self._setup_pick_tooltips(self.uib_canvas, all_cals_uib,
                                      np.array(all_x_uib, dtype=float),
                                      lambda c: f"D/U={c.get('area',0)/c.get('area_u',1):.2f}" if c.get('area_u',0)>0 else "D/U=N/A")
        else:
            ax6 = self.uib_figure.add_subplot(111)
            ax6.text(0.5, 0.5, "No hi ha dades UIB (cal DUAL mode)",
                    ha='center', va='center', fontsize=12, color='gray')
            ax6.set_xlim(0, 1)
            ax6.set_ylim(0, 1)

        self.uib_figure.tight_layout()
        self.uib_canvas.draw()

        # =====================================================================
        # Gràfic 7: Levey-Jennings (desviació % vs calibració vigent)
        #           2 subplots: COLUMN (esquerra) i BP (dreta)
        # =====================================================================
        self.lj_figure.clear()

        # Obtenir calibració vigent
        active_cal = get_active_global_calibration()

        # Recollir desviacions separades per mode
        lj_data = {'COLUMN': {'dev': [], 'x': [], 'cals': []},
                   'BP':     {'dev': [], 'x': [], 'cals': []}}

        for i, cal_entry in enumerate(sorted_cals):
            if not is_valids[i]:
                continue

            area = areas_d[i]
            conc = concs[i]
            vol = volumes[i]
            mode = modes[i]

            if area <= 0 or conc <= 0 or vol <= 0:
                continue
            if mode not in lj_data:
                continue

            # Obtenir RF i intercept per aquest mode/senyal
            if active_cal:
                rf_cur = get_rf_mass_cal(signal='direct', mode=mode.lower())
                int_cur = get_calibration_intercept(signal='direct', mode=mode.lower())
            else:
                rf_cur = None
                int_cur = 0

            if not rf_cur or rf_cur <= 0:
                continue

            # Àrea esperada = RF * µg + intercept
            ug_doc = conc * vol / 1000.0
            expected_area = rf_cur * ug_doc + int_cur
            if expected_area <= 0:
                continue

            deviation_pct = (area - expected_area) / expected_area * 100.0
            lj_data[mode]['dev'].append(deviation_pct)
            lj_data[mode]['x'].append(seq_numbers[i])
            lj_data[mode]['cals'].append(cal_entry)

        # Determinar quins modes tenen dades
        modes_with_data = [m for m in ('COLUMN', 'BP') if lj_data[m]['dev']]

        if modes_with_data and active_cal:
            n_subplots = len(modes_with_data)
            axes_lj = self.lj_figure.subplots(1, n_subplots, sharey=True)
            if n_subplots == 1:
                axes_lj = [axes_lj]

            LJ_YLIM = 50
            all_cals_lj = []
            all_x_lj = []

            def lj_color(d):
                if abs(d) <= 10: return '#27AE60'
                if abs(d) <= 20: return '#F39C12'
                return '#E74C3C'

            for ax_idx, mode_name in enumerate(modes_with_data):
                ax = axes_lj[ax_idx]
                md = lj_data[mode_name]
                dev_arr = np.array(md['dev'])
                x_lj = np.array(md['x'], dtype=float)
                marker = 'o' if mode_name == 'COLUMN' else 's'

                # RF info per títol
                rf_cur = get_rf_mass_cal(signal='direct', mode=mode_name.lower())
                int_cur = get_calibration_intercept(signal='direct', mode=mode_name.lower())

                in_range = np.abs(dev_arr) <= LJ_YLIM
                n_beyond = np.sum(~in_range)

                # Dibuixar punts
                for j in range(len(dev_arr)):
                    xj, dj = x_lj[j], dev_arr[j]
                    if in_range[j]:
                        ax.scatter(xj, dj, c=lj_color(dj), s=60, marker=marker,
                                   edgecolors='white', linewidth=0.5, zorder=3, picker=5)
                    else:
                        y_clamp = LJ_YLIM * 0.95 if dj > 0 else -LJ_YLIM * 0.95
                        ax.scatter(xj, y_clamp, c='#E74C3C', s=60, marker=marker,
                                   edgecolors='#8B0000', linewidth=2, zorder=4, picker=5)
                        sn = md['cals'][j].get('seq_name', '').replace('_SEQ', '')
                        ax.annotate(f'{sn} ({dj:+.0f}%)', xy=(xj, y_clamp),
                                    fontsize=5, color='#E74C3C', alpha=0.8,
                                    xytext=(3, -7 if dj > 0 else 7),
                                    textcoords='offset points', ha='left',
                                    va='top' if dj > 0 else 'bottom')

                # Bandes de control
                ax.axhline(0, color='black', linewidth=1, zorder=2)
                ax.axhspan(-10, 10, alpha=0.08, color='#27AE60', zorder=0)
                for lv, col_lv in [(10, '#F39C12'), (20, '#E74C3C')]:
                    ax.axhline(lv, color=col_lv, linewidth=1, linestyle='--', alpha=0.7)
                    ax.axhline(-lv, color=col_lv, linewidth=1, linestyle='--', alpha=0.7)

                # Etiquetes bandes (només al subplot dret o únic)
                if ax_idx == n_subplots - 1:
                    xlim_max = max(x_lj) + 1 if len(x_lj) > 0 else 1
                    for lv, col_lv in [(10, '#F39C12'), (20, '#E74C3C')]:
                        ax.text(xlim_max, lv, f'+{lv}%', fontsize=6, color=col_lv, va='center')
                        ax.text(xlim_max, -lv, f'-{lv}%', fontsize=6, color=col_lv, va='center')

                ax.set_ylim(-LJ_YLIM, LJ_YLIM)

                # Estadístiques
                n_ok = np.sum(np.abs(dev_arr) <= 10)
                n_warn = np.sum((np.abs(dev_arr) > 10) & (np.abs(dev_arr) <= 20))
                n_out = np.sum(np.abs(dev_arr) > 20)
                n_total = len(dev_arr)

                # Tendència
                dev_in = dev_arr[in_range]
                if len(dev_in) > 3:
                    x_in = x_lj[in_range]
                    z = np.polyfit(x_in, dev_in, 1)
                    p = np.poly1d(z)
                    ax.plot(x_in, p(x_in), '--', color='#8E44AD', alpha=0.5, linewidth=1.2)

                # Status
                if n_out > n_total * 0.3:
                    status_lj, status_color = "FORA DE CONTROL", '#E74C3C'
                elif n_warn + n_out > n_total * 0.3:
                    status_lj, status_color = "ATENCIÓ", '#F39C12'
                else:
                    status_lj, status_color = "EN CONTROL", '#27AE60'

                beyond_text = f"  ({n_beyond} fora)" if n_beyond > 0 else ""
                int_text = f"+{int_cur:.0f}" if int_cur else ""
                ax.set_title(
                    f"{mode_name} (RF={rf_cur:.0f}{int_text}) — {status_lj}\n"
                    f"OK:{n_ok}  Atenció:{n_warn}  Fora:{n_out}{beyond_text}  (n={n_total})",
                    fontsize=9, fontweight='bold', color=status_color
                )

                ax.set_xlabel("nº SEQ", fontsize=9)
                if ax_idx == 0:
                    ax.set_ylabel("Desviació vs recta vigent (%)", fontsize=9)
                ax.grid(True, alpha=0.2)

                # Intervencions
                y_min, y_max = ax.get_ylim()
                interv_used = self._add_maintenance_markers(ax, x_lj,
                    [md['cals'][j].get('date_processed', '')[:10] for j in range(len(x_lj))],
                    y_min, y_max)

                # Llegenda compacta
                legend_elements = [
                    Line2D([0],[0], marker=marker, color='w', markerfacecolor='#27AE60',
                           markersize=6, label='≤10%'),
                    Line2D([0],[0], marker=marker, color='w', markerfacecolor='#F39C12',
                           markersize=6, label='10-20%'),
                    Line2D([0],[0], marker=marker, color='w', markerfacecolor='#E74C3C',
                           markersize=6, label='>20%'),
                ]
                if n_beyond > 0:
                    legend_elements.append(
                        Line2D([0],[0], marker=marker, color='w', markerfacecolor='#E74C3C',
                               markeredgecolor='#8B0000', markeredgewidth=2,
                               markersize=6, label=f'Fora escala ({n_beyond})'))
                for cat, col in interv_used:
                    legend_elements.append(
                        Line2D([0],[0], linestyle='--', color=col, linewidth=1.2,
                               alpha=0.6, label=cat))
                ax.legend(handles=legend_elements, loc='upper right', fontsize=6, framealpha=0.8)

                # Acumular per tooltips
                for j, cal_lj in enumerate(md['cals']):
                    cal_lj['_lj_deviation'] = md['dev'][j]
                all_cals_lj.extend(md['cals'])
                all_x_lj.extend(md['x'])

            # Tooltips (sobre tots els subplots via el canvas comú)
            self._setup_pick_tooltips(self.lj_canvas, all_cals_lj,
                                      np.array(all_x_lj, dtype=float),
                                      lambda c: f"Desv={c.get('_lj_deviation',0):+.1f}%")

        else:
            ax7 = self.lj_figure.add_subplot(111)
            if not active_cal:
                msg = "No hi ha calibració vigent per calcular desviacions"
            else:
                msg = "No hi ha dades de producció KHP"
            ax7.text(0.5, 0.5, msg, ha='center', va='center', fontsize=12, color='gray')
            ax7.set_xlim(0, 1)
            ax7.set_ylim(0, 1)

        self.lj_figure.tight_layout()
        self.lj_canvas.draw()

    def _export_csv(self):
        """Exporta les calibracions visibles a CSV."""
        from PySide6.QtWidgets import QFileDialog
        import csv

        path, _ = QFileDialog.getSaveFileName(
            self, "Exportar Històric",
            "khp_history_export.csv",
            "CSV Files (*.csv)"
        )

        if not path:
            return

        try:
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)

                # Capçaleres
                headers = []
                for col in range(self.history_table.columnCount()):
                    headers.append(self.history_table.horizontalHeaderItem(col).text())
                writer.writerow(headers)

                # Dades
                for row in range(self.history_table.rowCount()):
                    row_data = []
                    for col in range(self.history_table.columnCount()):
                        item = self.history_table.item(row, col)
                        row_data.append(item.text() if item else "")
                    writer.writerow(row_data)

            QMessageBox.information(
                self, "Exportació Completa",
                f"Exportades {self.history_table.rowCount()} calibracions a:\n{path}"
            )

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error exportant: {e}")

    def _show_calibration_detail_dialog(self, cal):
        """Mostra un diàleg amb tots els detalls de la calibració."""
        from PySide6.QtWidgets import QDialog, QTextEdit

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Detall Calibració: {cal.get('seq_name', 'N/A')}")
        dialog.setMinimumSize(800, 600)

        layout = QVBoxLayout(dialog)

        # Crear tabs per organitzar la info
        tabs = QTabWidget()

        # === Tab 1: Resum ===
        summary_widget = QWidget()
        summary_layout = QVBoxLayout(summary_widget)

        # Info bàsica en grid
        info_group = QGroupBox("Informació Bàsica")
        info_grid = QGridLayout(info_group)

        row = 0
        basic_fields = [
            ("Seqüència", cal.get('seq_name', 'N/A')),
            ("Data", cal.get('date_processed', 'N/A')[:19].replace('T', ' ')),
            ("Mode", cal.get('mode', 'N/A')),
            ("KHP", f"{cal.get('conc_ppm', 0):g} ppm"),
            ("Volum", f"{cal.get('volume_uL', 0)} µL"),
            ("Font", cal.get('khp_source', 'LOCAL')),
        ]
        for label, value in basic_fields:
            info_grid.addWidget(QLabel(f"<b>{label}:</b>"), row, 0)
            info_grid.addWidget(QLabel(str(value)), row, 1)
            row += 1

        summary_layout.addWidget(info_group)

        # Mètriques principals
        metrics_group = QGroupBox("Mètriques Principals")
        metrics_grid = QGridLayout(metrics_group)

        row = 0
        metric_fields = [
            ("Àrea_D", f"{cal.get('area', 0):.1f}"),
            ("Àrea_U", f"{cal.get('area_u', 0):.1f}"),
            ("RF", f"{cal.get('rf', 0):.2f}"),
            ("RF_MASS", f"{cal.get('rf_mass', 0):.1f}"),
            ("t_max", f"{cal.get('t_retention', 0):.2f} min"),
            ("FWHM", f"{cal.get('fwhm_doc', 0):.2f} min"),
            ("SNR", f"{cal.get('snr', 0):.0f}"),
            ("Simetria", f"{cal.get('symmetry', 0):.2f}"),
            ("DOC/254", f"{cal.get('d254_d', 0):.2f}"),
            ("UIB/254", f"{cal.get('d254_u', 0):.2f}"),
            ("AR", f"{cal.get('area_ratio', 0):.2f}"),
            ("nP", f"{cal.get('n_peaks', 1)}"),
        ]

        col = 0
        for i, (label, value) in enumerate(metric_fields):
            row = i % 6
            col = (i // 6) * 2
            metrics_grid.addWidget(QLabel(f"<b>{label}:</b>"), row, col)
            metrics_grid.addWidget(QLabel(str(value)), row, col + 1)

        summary_layout.addWidget(metrics_group)

        # Qualitat
        quality_group = QGroupBox("Qualitat")
        quality_layout = QVBoxLayout(quality_group)

        cal_anoms = cal.get('calibration_anomalies', [])
        status = cal.get('status', 'OK')
        bg_doc = cal.get('bigaussian_doc') or {}
        r2_bg = bg_doc.get('r2', 0)

        if r2_bg > 0:
            quality_layout.addWidget(QLabel(f"<b>R² bigaussiana:</b> {r2_bg:.3f}"))
        quality_layout.addWidget(QLabel(f"<b>Estat:</b> {status}"))
        if cal_anoms:
            labels = [a.get('label', a.get('code', '')) for a in cal_anoms if isinstance(a, dict)]
            quality_layout.addWidget(QLabel(f"<b>Anomalies:</b> {', '.join(labels)}"))

        # Selecció de rèpliques
        selection = cal.get('selection', {})
        if selection:
            quality_layout.addWidget(QLabel(f"<b>Selecció:</b> {selection.get('method', 'N/A')}"))
            quality_layout.addWidget(QLabel(f"<b>Rèpliques:</b> {selection.get('selected_replicas', [])}"))
            if selection.get('is_manual'):
                quality_layout.addWidget(QLabel("<b style='color: blue;'>⚠️ Selecció Manual</b>"))

        summary_layout.addWidget(quality_group)
        summary_layout.addStretch()

        tabs.addTab(summary_widget, "📋 Resum")

        # === Tab 2: JSON Complet ===
        json_widget = QWidget()
        json_layout = QVBoxLayout(json_widget)

        json_text = QTextEdit()
        json_text.setReadOnly(True)
        json_text.setFont(QFont("Consolas", 9))
        json_text.setText(json.dumps(cal, indent=2, ensure_ascii=False, default=str))
        json_layout.addWidget(json_text)

        tabs.addTab(json_widget, "📄 JSON")

        # === Tab 3: Gràfic (si hi ha dades) ===
        # Intentar carregar perfil des de calibration_result.json
        seq_path = cal.get('seq_path', '')
        if seq_path:
            profile_fig = self._load_calibration_profile(seq_path, cal)
            if profile_fig:
                profile_widget = QWidget()
                profile_layout = QVBoxLayout(profile_widget)
                profile_canvas = FigureCanvas(profile_fig)
                profile_layout.addWidget(profile_canvas)
                tabs.addTab(profile_widget, "📈 Perfil")

        layout.addWidget(tabs)

        # Botons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        close_btn = QPushButton("Tancar")
        close_btn.clicked.connect(dialog.close)
        btn_layout.addWidget(close_btn)

        layout.addLayout(btn_layout)

        dialog.exec()

    def _load_calibration_profile(self, seq_path, cal):
        """Intenta carregar i mostrar el perfil de la calibració."""
        from pathlib import Path

        # Buscar calibration_result.json
        cal_result_path = Path(seq_path) / "CHECK" / "data" / "calibration_result.json"
        if not cal_result_path.exists():
            return None

        try:
            with open(cal_result_path, 'r', encoding='utf-8') as f:
                cal_result = json.load(f)

            # Crear figura amb subplots per Direct i UIB
            fig = Figure(figsize=(10, 6), dpi=100)

            # Buscar dades de rèpliques
            khp_direct = cal_result.get('khp_data_direct', {})
            khp_uib = cal_result.get('khp_data_uib', {})

            if khp_direct or khp_uib:
                ax = fig.add_subplot(111)

                # Mostrar info de rèpliques
                info_text = []

                if khp_direct:
                    reps = khp_direct.get('replicas', [])
                    for i, rep in enumerate(reps):
                        info_text.append(
                            f"Direct R{i+1}: Àrea={rep.get('area', 0):.1f}, "
                            f"t_max={rep.get('t_doc_max', 0):.2f}, "
                            f"SNR={rep.get('snr', 0):.0f}"
                        )

                if khp_uib:
                    reps = khp_uib.get('replicas', [])
                    for i, rep in enumerate(reps):
                        info_text.append(
                            f"UIB R{i+1}: Àrea={rep.get('area', 0):.1f}, "
                            f"t_max={rep.get('t_doc_max', 0):.2f}, "
                            f"SNR={rep.get('snr', 0):.0f}"
                        )

                # Mostrar com a text perquè no tenim els perfils raw
                ax.text(0.5, 0.5, "\n".join(info_text),
                       ha='center', va='center', fontsize=10,
                       transform=ax.transAxes,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                ax.set_title(f"Detall KHP - {cal.get('seq_name', 'N/A')}")
                ax.axis('off')

                fig.tight_layout()
                return fig

        except Exception as e:
            logger.warning(f"Error carregant perfil: {e}")

        return None
