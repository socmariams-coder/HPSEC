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
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from hpsec_calibrate import load_khp_history

import numpy as np

# Matplotlib
import matplotlib
matplotlib.use('QtAgg')
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
        title = QLabel("Històric de Calibracions")
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        layout.addWidget(title)

        # Info
        info = QLabel(
            "Visualitza totes les calibracions KHP registrades. "
            "Pots filtrar per mode, concentració o cercar per nom de seqüència."
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
            "Àrea_D", "Àrea_U", "RF_D", "RF_V",
            "t_max", "FWHM", "Sym", "SNR",
            "DOC/254", "UIB/254", "AR", "nP",
            "Sel", "Q", "Estat", "Motiu"
        ])

        # Tooltips capçaleres (veure docs/PARAMETRES_CALIBRACIO.md)
        # Columnes: Data(0), SEQ(1), Mode(2), KHP(3), Vol(4), Sens(5),
        #           Àrea_D(6), Àrea_U(7), RF_D(8), RF_V(9), t_max(10), FWHM(11),
        #           Sym(12), SNR(13), DOC/254(14), UIB/254(15), AR(16), nP(17),
        #           Sel(18), Q(19), Estat(20), Motiu(21)
        headers = self.history_table.horizontalHeader()
        self.history_table.horizontalHeaderItem(4).setToolTip("Vol: Volum d'injecció (µL)")
        self.history_table.horizontalHeaderItem(5).setToolTip("Sens: Sensibilitat UIB (ppb)")
        self.history_table.horizontalHeaderItem(6).setToolTip("Àrea_D: Àrea DOC Direct")
        self.history_table.horizontalHeaderItem(7).setToolTip("Àrea_U: Àrea DOC UIB")
        self.history_table.horizontalHeaderItem(8).setToolTip("RF_D: Response Factor Direct = Àrea/Conc")
        self.history_table.horizontalHeaderItem(9).setToolTip("RF_V: RF normalitzat per volum (RF/Vol×100) - CLAU!")
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

        # === TAB 4: Gràfic RF_V (clau!) ===
        rfv_widget = QWidget()
        rfv_layout = QVBoxLayout(rfv_widget)
        rfv_layout.setContentsMargins(0, 8, 0, 0)

        self.rfv_figure = Figure(figsize=(10, 5), dpi=100)
        self.rfv_canvas = FigureCanvas(self.rfv_figure)
        rfv_layout.addWidget(self.rfv_canvas)

        self.content_tabs.addTab(rfv_widget, "⚡ RF_V (Clau)")

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
        self._load_history()

    def _load_history(self):
        """Carrega tot l'històric de calibracions."""
        self._all_calibrations = []

        # Intentar carregar des de múltiples ubicacions
        possible_paths = []

        # 1. Path de la SEQ actual
        if self.main_window.seq_path:
            seq_path = Path(self.main_window.seq_path)
            possible_paths.extend([
                seq_path.parent / "khp_calibration_history.json",
                seq_path.parent.parent / "khp_calibration_history.json",
                seq_path / "CALDATA" / "khp_calibration_history.json",
            ])

        # 2. Paths comuns
        possible_paths.extend([
            Path.home() / "HPSEC_Data" / "khp_calibration_history.json",
            Path("D:/HPSEC/khp_calibration_history.json"),
            Path("C:/HPSEC/khp_calibration_history.json"),
        ])

        for hist_path in possible_paths:
            if hist_path.exists():
                try:
                    with open(hist_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            self._all_calibrations.extend(data)
                        print(f"[INFO] Carregades {len(data)} calibracions de {hist_path}")
                except Exception as e:
                    print(f"[WARNING] Error llegint {hist_path}: {e}")

        # També provar amb load_khp_history si tenim seq_path
        if self.main_window.seq_path and not self._all_calibrations:
            try:
                history = load_khp_history(self.main_window.seq_path)
                if history:
                    self._all_calibrations = history
            except Exception as e:
                print(f"[WARNING] Error amb load_khp_history: {e}")

        # Eliminar duplicats per seq_name + date
        seen = set()
        unique_cals = []
        for cal in self._all_calibrations:
            key = (cal.get('seq_name', ''), cal.get('date_processed', ''))
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
                concs.add(int(round(conc)))

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
            self.conc_filter.addItem(f"{c} ppm", c)
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

            # Filtre concentració
            if conc_filter and abs(cal.get('conc_ppm', 0) - conc_filter) >= 1:
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
            self.history_table.setItem(row, 3, QTableWidgetItem(f"{conc:.0f}"))

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

            # Col 9: RF_V (RF normalitzat per volum)
            rf_v = cal.get('rf_v', 0)
            self.history_table.setItem(row, 9, QTableWidgetItem(f"{rf_v:.3f}" if rf_v > 0 else "-"))

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

            # Col 19: Quality Score (v2 si disponible)
            q = cal.get('quality_score_v2', cal.get('quality_score', 0))
            q_item = QTableWidgetItem(f"{q:.0f}")
            if q >= 100:
                q_item.setBackground(QColor('#F5B7B1'))
            elif q >= 50:
                q_item.setBackground(QColor('#FCF3CF'))
            elif q > 20:
                q_item.setBackground(QColor('#FEF9E7'))
            else:
                q_item.setBackground(QColor('#D5F5E3'))
            self.history_table.setItem(row, 19, q_item)

            # Col 20: Estat (v2 si disponible)
            status_v2 = cal.get('status_v2', '')
            is_valid = cal.get('valid_for_calibration', True)
            is_outlier = cal.get('is_outlier', False) or cal.get('manual_outlier', False)

            if status_v2 == 'INVALID' or q >= 100:
                status = "INVALID"
                status_color = QColor('#F5B7B1')
            elif is_outlier:
                status = "EXCLÒS"
                status_color = QColor('#EBEDEF')
            elif not is_valid:
                status = "INVALID"
                status_color = QColor('#F5B7B1')
            elif status_v2:
                status = status_v2
                status_color = QColor('#D5F5E3') if status_v2 == 'OK' else QColor('#FCF3CF')
            else:
                status = "OK"
                status_color = QColor('#D5F5E3')

            status_item = QTableWidgetItem(status)
            status_item.setBackground(status_color)
            self.history_table.setItem(row, 20, status_item)

            # Col 21: Motiu
            quality_issues = cal.get('quality_issues_v2', cal.get('quality_issues', []))
            motiu = ""
            if quality_issues:
                motiu = ", ".join(quality_issues) if isinstance(quality_issues, list) else str(quality_issues)
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

    def _update_graphs(self, calibrations):
        """Actualitza tots els gràfics amb les calibracions filtrades."""
        # Ordenar per data per als gràfics
        sorted_cals = sorted(calibrations, key=lambda x: x.get('date_processed', ''))

        # Preparar dades
        dates = []
        seq_names = []
        areas = []
        t_maxs = []
        doc_254s = []
        rf_vs = []
        fwhms = []
        volumes = []
        modes = []
        is_valids = []

        for cal in sorted_cals:
            date_str = cal.get('date_processed', '')[:10]
            dates.append(date_str)
            seq_names.append(cal.get('seq_name', 'N/A').replace('_SEQ', ''))

            area = cal.get('area', 0)
            areas.append(area)

            # t_max
            t_max = cal.get('t_retention', 0)
            t_maxs.append(t_max)

            doc_254s.append(cal.get('d254_d', 0))

            # RF_V = RF / volum × 100
            vol = cal.get('volume_uL', 0)
            volumes.append(vol)
            rf_v = cal.get('rf_v', 0)
            rf_vs.append(rf_v)

            # FWHM
            fwhm = cal.get('fwhm_doc', 0)
            fwhms.append(fwhm)

            modes.append(cal.get('mode', 'COLUMN'))

            is_valid = cal.get('valid_for_calibration', True)
            is_outlier = cal.get('is_outlier', False) or cal.get('manual_outlier', False)
            is_valids.append(is_valid and not is_outlier)

        # Convertir a numpy
        areas = np.array(areas)
        t_maxs = np.array(t_maxs)
        doc_254s = np.array(doc_254s)
        rf_vs = np.array(rf_vs)
        fwhms = np.array(fwhms)
        volumes = np.array(volumes)

        # Colors per mode i validesa
        colors = []
        for mode, valid in zip(modes, is_valids):
            if not valid:
                colors.append('#CCCCCC')
            elif mode == 'BP':
                colors.append('#E67E22')
            else:
                colors.append('#3498DB')

        x = np.arange(len(dates))

        # === Gràfic 1: Àrea ===
        self.area_figure.clear()
        ax1 = self.area_figure.add_subplot(111)

        if len(x) > 0 and any(areas > 0):
            ax1.bar(x, areas, color=colors, alpha=0.8, edgecolor='white')

            # Mitjana i ±σ de vàlids
            valid_areas = areas[np.array(is_valids) & (areas > 0)]
            if len(valid_areas) > 1:
                mean_a = np.mean(valid_areas)
                std_a = np.std(valid_areas)
                ax1.axhline(mean_a, color='#27AE60', linestyle='-', linewidth=2, label=f'Mitjana: {mean_a:.0f}')
                ax1.axhspan(mean_a - std_a, mean_a + std_a, alpha=0.2, color='#27AE60')
                ax1.legend(loc='upper right')

            ax1.set_xticks(x)
            ax1.set_xticklabels(seq_names, rotation=45, ha='right', fontsize=8)
            ax1.set_ylabel("Àrea", fontsize=10)
            ax1.set_title("Evolució Àrea KHP", fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3, axis='y')
        else:
            ax1.text(0.5, 0.5, "No hi ha dades", ha='center', va='center', fontsize=12, color='gray')
            ax1.set_xlim(0, 1)
            ax1.set_ylim(0, 1)

        self.area_figure.tight_layout()
        self.area_canvas.draw()

        # === Gràfic 2: t_max (Deriva) ===
        self.tmax_figure.clear()
        ax2 = self.tmax_figure.add_subplot(111)

        valid_tmax_mask = (t_maxs > 0)
        if any(valid_tmax_mask):
            # Scatter plot amb colors
            for i, (t, c, name) in enumerate(zip(t_maxs, colors, seq_names)):
                if t > 0:
                    ax2.scatter(i, t, c=c, s=60, edgecolors='white', linewidth=0.5, zorder=3)

            # Línia de tendència si hi ha prou punts
            valid_x = x[valid_tmax_mask]
            valid_t = t_maxs[valid_tmax_mask]
            if len(valid_t) > 2:
                z = np.polyfit(valid_x, valid_t, 1)
                p = np.poly1d(z)
                ax2.plot(valid_x, p(valid_x), '--', color='#E74C3C', alpha=0.7,
                        label=f'Tendència: {z[0]*10:.3f} min/10 SEQ')
                ax2.legend(loc='upper right')

            # Mitjana i rang
            mean_t = np.mean(valid_t)
            std_t = np.std(valid_t)
            ax2.axhline(mean_t, color='#27AE60', linestyle='-', linewidth=1.5, alpha=0.7)
            ax2.axhspan(mean_t - 2*std_t, mean_t + 2*std_t, alpha=0.1, color='#27AE60')

            ax2.set_xticks(x)
            ax2.set_xticklabels(seq_names, rotation=45, ha='right', fontsize=8)
            ax2.set_ylabel("t_max (min)", fontsize=10)
            ax2.set_title(f"Deriva Temps de Pic · Mitjana: {mean_t:.2f} ± {std_t:.2f} min",
                         fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, "No hi ha dades de t_max", ha='center', va='center', fontsize=12, color='gray')
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)

        self.tmax_figure.tight_layout()
        self.tmax_canvas.draw()

        # === Gràfic 3: RF_V (Response Factor normalitzat per volum) - CLAU! ===
        self.rfv_figure.clear()
        ax3 = self.rfv_figure.add_subplot(111)

        valid_rfv_mask = (rf_vs > 0)
        if any(valid_rfv_mask):
            # Separar per volum per colors diferents
            colors_rfv = []
            for vol, valid in zip(volumes, is_valids):
                if not valid:
                    colors_rfv.append('#CCCCCC')
                elif vol == 100:
                    colors_rfv.append('#9B59B6')  # Lila per 100µL
                elif vol == 400:
                    colors_rfv.append('#1ABC9C')  # Verd per 400µL
                else:
                    colors_rfv.append('#3498DB')  # Blau per altres

            for i, (rfv, c, name) in enumerate(zip(rf_vs, colors_rfv, seq_names)):
                if rfv > 0:
                    ax3.scatter(i, rfv, c=c, s=60, edgecolors='white', linewidth=0.5, zorder=3)

            valid_rfv = rf_vs[valid_rfv_mask & np.array(is_valids)]
            if len(valid_rfv) > 1:
                mean_rfv = np.mean(valid_rfv)
                std_rfv = np.std(valid_rfv)
                cv_rfv = (std_rfv / mean_rfv * 100) if mean_rfv > 0 else 0
                ax3.axhline(mean_rfv, color='#27AE60', linestyle='-', linewidth=2, alpha=0.8)
                ax3.axhspan(mean_rfv - 2*std_rfv, mean_rfv + 2*std_rfv, alpha=0.15, color='#27AE60')
                # Límits d'alerta (±10%)
                ax3.axhline(mean_rfv * 1.1, color='#E74C3C', linestyle='--', linewidth=1, alpha=0.5)
                ax3.axhline(mean_rfv * 0.9, color='#E74C3C', linestyle='--', linewidth=1, alpha=0.5)

                ax3.set_title(f"RF_V (Normalitzat per Volum) · Mitjana: {mean_rfv:.3f} · CV: {cv_rfv:.1f}%",
                             fontsize=12, fontweight='bold')
            else:
                ax3.set_title("RF_V (Normalitzat per Volum)", fontsize=12, fontweight='bold')

            ax3.set_xticks(x)
            ax3.set_xticklabels(seq_names, rotation=45, ha='right', fontsize=8)
            ax3.set_ylabel("RF_V (RF/Vol×100)", fontsize=10)
            ax3.grid(True, alpha=0.3)

            # Llegenda per volums
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#9B59B6', markersize=8, label='100 µL'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#1ABC9C', markersize=8, label='400 µL'),
            ]
            ax3.legend(handles=legend_elements, loc='upper right')
        else:
            ax3.text(0.5, 0.5, "No hi ha dades de RF_V", ha='center', va='center', fontsize=12, color='gray')
            ax3.set_xlim(0, 1)
            ax3.set_ylim(0, 1)

        self.rfv_figure.tight_layout()
        self.rfv_canvas.draw()

        # === Gràfic 4: D/254 ===
        self.ratio_figure.clear()
        ax4 = self.ratio_figure.add_subplot(111)

        valid_ratio_mask = (doc_254s > 0)
        if any(valid_ratio_mask):
            for i, (r, c, name) in enumerate(zip(doc_254s, colors, seq_names)):
                if r > 0:
                    ax4.scatter(i, r, c=c, s=60, edgecolors='white', linewidth=0.5, zorder=3)

            valid_r = doc_254s[valid_ratio_mask]
            mean_r = np.mean(valid_r)
            std_r = np.std(valid_r)
            ax4.axhline(mean_r, color='#27AE60', linestyle='-', linewidth=1.5, alpha=0.7)
            ax4.axhspan(mean_r - 2*std_r, mean_r + 2*std_r, alpha=0.1, color='#27AE60')

            ax4.set_xticks(x)
            ax4.set_xticklabels(seq_names, rotation=45, ha='right', fontsize=8)
            ax4.set_ylabel("Ratio DOC/254", fontsize=10)
            ax4.set_title(f"Ratio DOC/254nm · Mitjana: {mean_r:.2f} ± {std_r:.2f}",
                         fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, "No hi ha dades de D/254", ha='center', va='center', fontsize=12, color='gray')
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)

        self.ratio_figure.tight_layout()
        self.ratio_canvas.draw()

        # === Gràfic 5: FWHM (indicador degradació columna) ===
        self.fwhm_figure.clear()
        ax5 = self.fwhm_figure.add_subplot(111)

        valid_fwhm_mask = (fwhms > 0)
        if any(valid_fwhm_mask):
            for i, (fw, c, name) in enumerate(zip(fwhms, colors, seq_names)):
                if fw > 0:
                    ax5.scatter(i, fw, c=c, s=60, edgecolors='white', linewidth=0.5, zorder=3)

            # Línia de tendència per detectar degradació
            valid_x = x[valid_fwhm_mask]
            valid_fw = fwhms[valid_fwhm_mask]
            if len(valid_fw) > 2:
                z = np.polyfit(valid_x, valid_fw, 1)
                p = np.poly1d(z)
                trend_color = '#E74C3C' if z[0] > 0.01 else '#27AE60'  # Vermell si augmenta
                ax5.plot(valid_x, p(valid_x), '--', color=trend_color, alpha=0.7,
                        label=f'Tendència: {z[0]*10:.3f} min/10 SEQ')
                ax5.legend(loc='upper left')

            mean_fw = np.mean(valid_fw)
            std_fw = np.std(valid_fw)
            ax5.axhline(mean_fw, color='#27AE60', linestyle='-', linewidth=1.5, alpha=0.7)
            # Límit d'alerta (1.5 min)
            ax5.axhline(1.5, color='#E74C3C', linestyle='--', linewidth=1.5, alpha=0.7, label='Límit (1.5 min)')

            ax5.set_xticks(x)
            ax5.set_xticklabels(seq_names, rotation=45, ha='right', fontsize=8)
            ax5.set_ylabel("FWHM (min)", fontsize=10)
            ax5.set_title(f"FWHM (Amplada de Pic) · Mitjana: {mean_fw:.2f} ± {std_fw:.2f} min",
                         fontsize=12, fontweight='bold')
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, "No hi ha dades de FWHM", ha='center', va='center', fontsize=12, color='gray')
            ax5.set_xlim(0, 1)
            ax5.set_ylim(0, 1)

        self.fwhm_figure.tight_layout()
        self.fwhm_canvas.draw()

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
            ("KHP", f"{cal.get('conc_ppm', 0)} ppm"),
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
            ("RF_V", f"{cal.get('rf_v', 0):.4f}"),
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

        q_score = cal.get('quality_score', 0)
        status = cal.get('status', 'OK')
        issues = cal.get('quality_issues', [])

        quality_layout.addWidget(QLabel(f"<b>Score:</b> {q_score}"))
        quality_layout.addWidget(QLabel(f"<b>Estat:</b> {status}"))
        if issues:
            issues_text = ", ".join(issues) if isinstance(issues, list) else str(issues)
            quality_layout.addWidget(QLabel(f"<b>Issues:</b> {issues_text}"))

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
        cal_result_path = Path(seq_path) / "CHECK" / "dades" / "calibration_result.json"
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
            print(f"[WARNING] Error carregant perfil: {e}")

        return None
