"""
HPSEC Suite - Global Calibration Panel (Consulta)
===================================================

Panell de CONSULTA amb dues vistes:
- Tab 0: Recta de Calibració — previsualització des de SEQ_CAL dedicades
- Tab 1: Control de Qualitat — Levey-Jennings per KHP de producció

Les accions (aplicar nova calibració) es fan des del wizard (CalibratePanel).
Aquí l'usuari pot veure i comparar regressions, però no aplicar-les.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGroupBox,
    QGridLayout, QTableWidget, QTableWidgetItem, QHeaderView,
    QComboBox, QMessageBox, QSplitter, QRadioButton, QButtonGroup,
    QSizePolicy, QCheckBox, QTabWidget, QListWidget,
    QListWidgetItem, QFrame
)
from PySide6.QtCore import Qt
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


class GlobalCalibrationPanel(QWidget):
    """Panell de calibració global: consulta de calibracions vigents i historial.

    Les accions (aplicar nova calibració) es fan des del wizard (CalibratePanel).
    Aquest panell és de consulta i previsualització.
    """

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self._all_calibrations = []
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

        # Tabs
        self.tabs = QTabWidget()
        self.cal_view = CalibrationLineView(self)
        self.qc_view = QCMonitorView(self)
        self.tabs.addTab(self.cal_view, "📐 Recta de Calibració")
        self.tabs.addTab(self.qc_view, "📊 Control de Qualitat")
        layout.addWidget(self.tabs, 1)

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
        self._last_result = None
        self._loading = False
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
        left_layout.addWidget(self._create_seq_selector(), 1)

        # Taula punts seleccionats
        left_layout.addWidget(self._create_points_table(), 1)

        # Resultats regressió
        left_layout.addWidget(self._create_results_group())

        # Stats per concentració
        left_layout.addWidget(self._create_stats_table())

        # Botons
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
        group = QGroupBox("Punts seleccionats")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(4, 4, 4, 4)

        self.points_table = QTableWidget()
        self.points_table.setColumnCount(8)
        self.points_table.setHorizontalHeaderLabels([
            "Usar", "SEQ", "Data", "Conc (ppm)", "Vol (µL)",
            "µg DOC", "Àrea", "RF_mass"
        ])

        header = self.points_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        for i in range(2, 8):
            header.setSectionResizeMode(i, QHeaderView.ResizeToContents)

        self.points_table.setAlternatingRowColors(True)
        self.points_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.points_table.verticalHeader().setVisible(False)

        layout.addWidget(self.points_table)
        return group

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

    def _create_stats_table(self):
        """Mini-taula d'estadístiques per concentració."""
        group = QGroupBox("Estadístiques per concentració")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(4, 4, 4, 4)

        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(5)
        self.stats_table.setHorizontalHeaderLabels(["Conc (ppm)", "n", "Mean Àrea", "CV%", "Mean RF"])
        self.stats_table.setMaximumHeight(100)
        self.stats_table.verticalHeader().setVisible(False)

        header = self.stats_table.horizontalHeader()
        for i in range(5):
            header.setSectionResizeMode(i, QHeaderView.Stretch)

        layout.addWidget(self.stats_table)
        return group

    def _create_buttons(self):
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        self.btn_recalculate = QPushButton("Recalcular")
        self.btn_recalculate.setToolTip("Recalcular regressió per previsualització (no aplica canvis)")
        self.btn_recalculate.clicked.connect(self._recalculate_regression)
        layout.addWidget(self.btn_recalculate)

        layout.addStretch()

        # Nota informativa: accions es fan des del wizard
        info_label = QLabel(
            "<i style='color: #7F8C8D;'>Per aplicar una nova calibració, "
            "processar una SEQ_CAL pel wizard</i>"
        )
        layout.addWidget(info_label)

        return widget

    # ---- Data & Refresh ----

    def set_data(self, cal_entries):
        """Rep les entrades filtrades _CAL."""
        self._cal_entries = cal_entries
        self._loading = True
        self._load_current_calibration()
        self._populate_seq_list()
        self._loading = False
        self._refresh_points_and_recalculate()

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
        """Pobla la taula amb punts de les SEQs seleccionades."""
        mode = self._get_mode()
        signal = self._get_signal()
        selected_seqs = self._get_selected_seq_names()

        self.points_table.setRowCount(0)
        self.points_table.blockSignals(True)

        # Recollir punts de les SEQs seleccionades
        filtered = []
        for seq_name in selected_seqs:
            for entry in self._grouped_by_seq.get(seq_name, []):
                filtered.append(entry)

        self.points_table.setRowCount(len(filtered))

        for row, cal in enumerate(filtered):
            conc = cal.get('conc_ppm', 0)
            vol = cal.get('volume_uL', 0)
            ug_doc = conc * vol / 1000.0 if conc > 0 and vol > 0 else 0

            if signal.lower() == 'uib':
                area = cal.get('area_u', 0)
            elif signal.lower() == '254':
                area = cal.get('area_254', 0) or cal.get('a254_area', 0) or 0
            else:
                area = cal.get('area', 0)

            rf_mass = area / ug_doc if ug_doc > 0 else 0
            is_outlier = cal.get('is_outlier', False)
            not_valid = not cal.get('valid_for_calibration', True)
            bad_point = is_outlier or not_valid or conc <= 0 or area <= 0

            # Checkbox
            chk = QCheckBox()
            chk.setChecked(not bad_point)
            chk.stateChanged.connect(self._on_point_toggled)
            chk_widget = QWidget()
            chk_layout = QHBoxLayout(chk_widget)
            chk_layout.addWidget(chk)
            chk_layout.setAlignment(Qt.AlignCenter)
            chk_layout.setContentsMargins(0, 0, 0, 0)
            self.points_table.setCellWidget(row, 0, chk_widget)

            # Dades
            items_text = [
                cal.get('seq_name', ''),
                str(cal.get('date', ''))[:10],
                f"{conc:g}",
                f"{vol:.0f}",
                f"{ug_doc:.3f}",
                f"{area:.1f}",
                f"{rf_mass:.1f}",
            ]

            for col, text in enumerate(items_text):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                if bad_point:
                    item.setForeground(QColor("#dc3545"))
                self.points_table.setItem(row, col + 1, item)

            # Tooltip
            issues = cal.get('calibration_issues', [])
            quality = cal.get('quality_score', 0)
            tip = f"Quality: {quality}"
            if is_outlier:
                tip += " | OUTLIER"
            if issues:
                tip += f" | Issues: {', '.join(str(i) for i in issues)}"
            for col in range(1, 8):
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
        mode = self._get_mode()
        selected_seqs = self._get_selected_seq_names()

        # Reconstruir llista filtrada (mateixa ordre que la taula)
        filtered = []
        for seq_name in selected_seqs:
            for entry in self._grouped_by_seq.get(seq_name, []):
                filtered.append(entry)

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

        self._update_stats_table(selected)
        self._update_preview_graph(result)
        self._update_comparison(result)

    def _update_stats_table(self, selected):
        """Estadístiques agrupades per concentració."""
        from collections import defaultdict

        signal = self._get_signal()
        groups = defaultdict(list)

        for cal in selected:
            conc = cal.get('conc_ppm', 0)
            vol = cal.get('volume_uL', 0)
            if conc <= 0 or vol <= 0:
                continue
            ug = conc * vol / 1000.0

            if signal.lower() == 'uib':
                area = cal.get('area_u', 0)
            elif signal.lower() == '254':
                area = cal.get('area_254', 0) or 0
            else:
                area = cal.get('area', 0)

            if area <= 0:
                continue
            rf_mass = area / ug if ug > 0 else 0
            groups[conc].append({"area": area, "rf_mass": rf_mass})

        self.stats_table.setRowCount(len(groups))
        for row, conc in enumerate(sorted(groups.keys())):
            vals = groups[conc]
            n = len(vals)
            areas = [v["area"] for v in vals]
            rfs = [v["rf_mass"] for v in vals]
            mean_area = np.mean(areas)
            cv = np.std(areas) / mean_area * 100 if mean_area > 0 and n > 1 else 0
            mean_rf = np.mean(rfs)

            for col, text in enumerate([
                f"{conc:g}", str(n), f"{mean_area:.1f}",
                f"{cv:.1f}", f"{mean_rf:.1f}"
            ]):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                if cv > 20:
                    item.setForeground(QColor("#dc3545"))
                self.stats_table.setItem(row, col, item)

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

    # Nota: _apply_calibration i _run_retroactive_requantification
    # s'han eliminat. Les accions es fan des del wizard (CalibratePanel).


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
