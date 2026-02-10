"""
HPSEC Suite - Global Calibration Panel
========================================

Panel per gestionar la calibració global (Calibration_Reference.json):
- Visualitzar calibració activa (rf_mass_cal, intercept, R²)
- Seleccionar punts KHP de l'historial (multi-conc, multi-vol)
- Fer regressió lineal → nou rf_mass_cal + intercept
- Aplicar com a nova calibració global
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGroupBox,
    QGridLayout, QTableWidget, QTableWidgetItem, QHeaderView,
    QComboBox, QMessageBox, QSplitter, QRadioButton, QButtonGroup,
    QInputDialog, QSizePolicy
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QColor

from pathlib import Path
from datetime import datetime
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from hpsec_calibrate import (
    get_active_global_calibration,
    load_khp_history,
    fit_calibration_from_history,
    add_calibration,
    load_calibration_reference,
)

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np


class GlobalCalibrationPanel(QWidget):
    """Panel per gestionar la calibració global."""

    calibration_updated = Signal()

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self._all_calibrations = []
        self._last_result = None
        self._loading = False
        self._setup_ui()

    def _setup_ui(self):
        """Configura la interfície."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)

        # Títol
        title = QLabel("Calibració Global")
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        layout.addWidget(title)

        subtitle = QLabel("Gestió de rf_mass_cal i intercept per mode (COLUMN/BP)")
        subtitle.setFont(QFont("Segoe UI", 9))
        subtitle.setStyleSheet("color: #666;")
        layout.addWidget(subtitle)

        # Splitter principal
        splitter = QSplitter(Qt.Horizontal)

        # === ESQUERRA: Controls ===
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        # 1. Calibració Actual
        left_layout.addWidget(self._create_current_cal_group())

        # 2. Selectors
        left_layout.addWidget(self._create_selectors_group())

        # 3. Taula de punts
        left_layout.addWidget(self._create_points_table(), 1)

        # 4. Resultats regressió
        left_layout.addWidget(self._create_results_group())

        # 5. Botons
        left_layout.addWidget(self._create_buttons())

        splitter.addWidget(left_widget)

        # === DRETA: Preview ===
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        # Gràfic matplotlib
        self.figure = Figure(figsize=(6, 4.5), dpi=100)
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

        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)

        layout.addWidget(splitter, 1)

    # =========================================================================
    # UI CREATION
    # =========================================================================

    def _create_current_cal_group(self):
        """Crea el grup 'Calibració Actual'."""
        group = QGroupBox("Calibració Actual")
        grid = QGridLayout(group)
        grid.setContentsMargins(8, 6, 8, 6)

        self.cur_rf_label = QLabel("—")
        self.cur_rf_label.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.cur_intercept_label = QLabel("—")
        self.cur_r2_label = QLabel("—")
        self.cur_npoints_label = QLabel("—")
        self.cur_valid_from_label = QLabel("—")
        self.cur_model_label = QLabel("—")

        grid.addWidget(QLabel("RF_mass_cal:"), 0, 0)
        grid.addWidget(self.cur_rf_label, 0, 1)
        grid.addWidget(QLabel("Intercept:"), 0, 2)
        grid.addWidget(self.cur_intercept_label, 0, 3)
        grid.addWidget(QLabel("R²:"), 1, 0)
        grid.addWidget(self.cur_r2_label, 1, 1)
        grid.addWidget(QLabel("n_punts:"), 1, 2)
        grid.addWidget(self.cur_npoints_label, 1, 3)
        grid.addWidget(QLabel("Vigent des de:"), 2, 0)
        grid.addWidget(self.cur_valid_from_label, 2, 1)
        grid.addWidget(QLabel("Model:"), 2, 2)
        grid.addWidget(self.cur_model_label, 2, 3)

        return group

    def _create_selectors_group(self):
        """Crea els selectors de mode, senyal i model."""
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

        layout.addSpacing(16)

        # Senyal: Direct / UIB
        layout.addWidget(QLabel("Senyal:"))
        self.signal_combo = QComboBox()
        self.signal_combo.addItems(["direct", "uib"])
        self.signal_combo.setFixedWidth(90)
        layout.addWidget(self.signal_combo)

        layout.addSpacing(16)

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

    def _create_points_table(self):
        """Crea la taula de punts KHP."""
        group = QGroupBox("Punts KHP disponibles")
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
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        for i in range(3, 8):
            header.setSectionResizeMode(i, QHeaderView.ResizeToContents)

        self.points_table.setAlternatingRowColors(True)
        self.points_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.points_table.verticalHeader().setVisible(False)

        layout.addWidget(self.points_table)
        return group

    def _create_results_group(self):
        """Crea el grup de resultats de regressió."""
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
        """Crea els botons d'acció."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        self.btn_recalculate = QPushButton("Recalcular")
        self.btn_recalculate.clicked.connect(self._recalculate_regression)
        layout.addWidget(self.btn_recalculate)

        layout.addStretch()

        self.btn_apply = QPushButton("Aplicar Nova Calibració")
        self.btn_apply.setStyleSheet(
            "QPushButton { background-color: #28a745; color: white; "
            "font-weight: bold; padding: 8px 16px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #218838; }"
            "QPushButton:disabled { background-color: #ccc; color: #666; }"
        )
        self.btn_apply.setEnabled(False)
        self.btn_apply.clicked.connect(self._apply_calibration)
        layout.addWidget(self.btn_apply)

        return widget

    # =========================================================================
    # DATA LOADING
    # =========================================================================

    def showEvent(self, event):
        """Carrega dades quan es mostra el tab."""
        super().showEvent(event)
        self._load_data()

    def _load_data(self):
        """Carrega calibració actual i històric KHP."""
        self._loading = True
        self._load_current_calibration()
        self._load_khp_history()
        self._refresh_points_table()
        self._loading = False
        self._recalculate_regression()

    def _load_current_calibration(self):
        """Carrega i mostra la calibració global activa."""
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
        if isinstance(r2_data, dict):
            r2_val = r2_data.get(mode)
        else:
            r2_val = r2_data
        self.cur_r2_label.setText(f"{r2_val:.4f}" if r2_val is not None else "—")

        # n_points
        np_data = cal.get('n_points')
        if isinstance(np_data, dict):
            np_val = np_data.get(mode)
        else:
            np_val = np_data
        self.cur_npoints_label.setText(str(np_val) if np_val is not None else "—")

        # Valid from
        self.cur_valid_from_label.setText(cal.get('valid_from', '—'))

        # Model
        self.cur_model_label.setText(cal.get('model', '—'))

    def _load_khp_history(self):
        """Carrega tot l'històric KHP."""
        self._all_calibrations = load_khp_history(None)

    def _get_mode(self):
        """Retorna mode seleccionat."""
        return "COLUMN" if self.radio_column.isChecked() else "BP"

    def _get_signal(self):
        """Retorna senyal seleccionat."""
        return self.signal_combo.currentText()

    def _get_model(self):
        """Retorna model seleccionat."""
        return "intercept" if self.radio_intercept.isChecked() else "origin"

    # =========================================================================
    # TAULA DE PUNTS
    # =========================================================================

    def _refresh_points_table(self):
        """Filtra i pobla la taula per mode seleccionat."""
        mode = self._get_mode()
        signal = self._get_signal()

        self.points_table.setRowCount(0)
        self.points_table.blockSignals(True)

        filtered = []
        for cal in self._all_calibrations:
            if cal.get('mode', '').upper() != mode.upper():
                continue
            filtered.append(cal)

        self.points_table.setRowCount(len(filtered))

        for row, cal in enumerate(filtered):
            conc = cal.get('conc_ppm', 0)
            vol = cal.get('volume_uL', 0)
            ug_doc = conc * vol / 1000.0 if conc > 0 and vol > 0 else 0

            if signal.lower() == 'uib':
                area = cal.get('area_u', 0)
            else:
                area = cal.get('area', 0)

            rf_mass = area / ug_doc if ug_doc > 0 else 0
            is_outlier = cal.get('is_outlier', False)
            not_valid = not cal.get('valid_for_calibration', True)
            bad_point = is_outlier or not_valid or conc <= 0 or area <= 0

            # Checkbox
            from PySide6.QtWidgets import QCheckBox
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
            items = [
                cal.get('seq_name', ''),
                str(cal.get('date', ''))[:10],
                f"{conc:.1f}",
                f"{vol:.0f}",
                f"{ug_doc:.2f}",
                f"{area:.1f}",
                f"{rf_mass:.1f}",
            ]

            for col, text in enumerate(items):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                if bad_point:
                    item.setForeground(QColor("#dc3545"))
                self.points_table.setItem(row, col + 1, item)

            # Tooltip
            issues = cal.get('calibration_issues', [])
            quality = cal.get('quality_score', 0)
            tip_parts = [f"Quality score: {quality}"]
            if is_outlier:
                tip_parts.append("OUTLIER")
            if not_valid:
                tip_parts.append("No vàlid per calibració")
            if issues:
                tip_parts.append(f"Issues: {', '.join(str(i) for i in issues)}")
            tooltip = " | ".join(tip_parts)
            for col in range(1, 8):
                item = self.points_table.item(row, col)
                if item:
                    item.setToolTip(tooltip)

        self.points_table.blockSignals(False)

    # =========================================================================
    # EVENTS
    # =========================================================================

    def _on_params_changed(self, *args):
        """Mode/senyal/model canviat → refresh."""
        self._load_current_calibration()
        self._refresh_points_table()
        if not self._loading:
            self._recalculate_regression()

    def _on_point_toggled(self, *args):
        """Checkbox canviat → recalcula."""
        if not self._loading:
            self._recalculate_regression()

    # =========================================================================
    # REGRESSIÓ
    # =========================================================================

    def _get_selected_calibrations(self):
        """Retorna llista de calibracions seleccionades (checkbox marcat)."""
        mode = self._get_mode()
        signal = self._get_signal()

        # Reconstruir llista filtrada (mateixa ordre que la taula)
        filtered = [c for c in self._all_calibrations
                    if c.get('mode', '').upper() == mode.upper()]

        selected = []
        for row in range(self.points_table.rowCount()):
            chk_widget = self.points_table.cellWidget(row, 0)
            if chk_widget:
                chk = chk_widget.findChild(type(None))
                # Find the QCheckBox within the widget
                from PySide6.QtWidgets import QCheckBox
                chk = chk_widget.findChild(QCheckBox)
                if chk and chk.isChecked() and row < len(filtered):
                    selected.append(filtered[row])

        return selected

    def _recalculate_regression(self):
        """Executa regressió amb els punts seleccionats."""
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
            self.res_rms_label.setText(f"{result['residuals_rms']:.2f}")
            self.btn_apply.setEnabled(True)
        else:
            self.res_rf_label.setText("—")
            self.res_intercept_label.setText("—")
            self.res_r2_label.setText("—")
            self.res_npoints_label.setText(str(result.get('n_points', 0)))
            self.res_rms_label.setText("—")
            self.btn_apply.setEnabled(False)

        self._update_preview_graph(result)
        self._update_comparison(result)

    # =========================================================================
    # GRÀFIC
    # =========================================================================

    def _update_preview_graph(self, result):
        """Actualitza scatter + línia regressió."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        mode = self._get_mode()
        signal = self._get_signal()

        # Punts seleccionats (blau) i exclosos (gris)
        selected = self._get_selected_calibrations()
        selected_names = {c.get('seq_name', '') + str(c.get('conc_ppm', ''))
                         + str(c.get('volume_uL', ''))
                         for c in selected}

        filtered = [c for c in self._all_calibrations
                    if c.get('mode', '').upper() == mode.upper()]

        x_sel, y_sel = [], []
        x_exc, y_exc = [], []

        for cal in filtered:
            conc = cal.get('conc_ppm', 0)
            vol = cal.get('volume_uL', 0)
            if conc <= 0 or vol <= 0:
                continue
            ug = conc * vol / 1000.0
            area = cal.get('area_u', 0) if signal == 'uib' else cal.get('area', 0)
            if area <= 0:
                continue

            key = cal.get('seq_name', '') + str(conc) + str(vol)
            if key in selected_names:
                x_sel.append(ug)
                y_sel.append(area)
            else:
                x_exc.append(ug)
                y_exc.append(area)

        if x_sel:
            ax.scatter(x_sel, y_sel, c='#2196F3', s=50, zorder=5,
                      label='Seleccionats', edgecolors='white', linewidth=0.5)
        if x_exc:
            ax.scatter(x_exc, y_exc, c='#aaa', s=40, zorder=4, marker='x',
                      label='Exclosos', linewidths=1.5)

        # Línia regressió nova
        if result.get('success'):
            rf = result['rf_mass_cal']
            intercept = result['intercept']
            r2 = result['r2']
            all_x = x_sel + x_exc
            if all_x:
                x_line = np.linspace(0, max(all_x) * 1.1, 100)
                y_line = rf * x_line + intercept
                eq = f"y = {rf:.1f}x + {intercept:.1f}" if intercept != 0 else f"y = {rf:.1f}x"
                ax.plot(x_line, y_line, 'r-', linewidth=2, label=f"Nova ({eq}, R²={r2:.4f})")

        # Línia calibració actual (discontinua)
        cal_actual = get_active_global_calibration()
        if cal_actual and x_sel:
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
                ax.plot(x_line, y_cur, '--', color='gray', linewidth=1.5,
                       alpha=0.7, label=f"Actual (RF={cur_rf:.0f})")

        ax.set_xlabel("µg DOC injectat")
        ax.set_ylabel("Àrea")
        ax.set_title(f"Regressió calibració — {mode} {signal}")
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.3)

        self.figure.tight_layout()
        self.canvas.draw()

    # =========================================================================
    # COMPARACIÓ
    # =========================================================================

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

        # Obtenir valors actuals
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

        if cur_rf is not None and cur_rf > 0:
            delta_rf = new_rf - cur_rf
            pct_rf = delta_rf / cur_rf * 100
            color_rf = "#dc3545" if abs(pct_rf) > 15 else "#28a745"
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

        if cur_rf is not None and cur_rf > 0 and abs(pct_rf) > 15:
            lines.append(
                "<br><span style='color:#dc3545; font-weight:bold;'>"
                "AVÍS: Variació RF > 15%</span>"
            )

        self.comparison_label.setText("".join(lines))

    # =========================================================================
    # APLICAR CALIBRACIÓ
    # =========================================================================

    def _apply_calibration(self):
        """Aplica la nova calibració al JSON global."""
        result = self._last_result
        if not result or not result.get('success'):
            return

        mode = self._get_mode()
        signal = self._get_signal()
        model = self._get_model()

        # Confirmació
        reply = QMessageBox.question(
            self,
            "Aplicar nova calibració",
            f"Vols aplicar la nova calibració?\n\n"
            f"Mode: {mode}, Senyal: {signal}\n"
            f"RF_mass_cal: {result['rf_mass_cal']:.1f}\n"
            f"Intercept: {result['intercept']:.1f}\n"
            f"R²: {result['r2']:.4f}, n={result['n_points']}\n\n"
            f"Això crearà una nova entrada i tancarà l'anterior.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        # Motiu del canvi
        reason, ok = QInputDialog.getText(
            self, "Motiu del canvi",
            "Descriu el motiu del canvi de calibració:"
        )
        if not ok:
            return

        # Construir rf_mass_cal preservant els valors existents
        cal = get_active_global_calibration()

        # Partir dels valors actuals
        if cal and isinstance(cal.get('rf_mass_cal'), dict):
            rf_mass_cal_values = {}
            for sig in ['direct', 'uib']:
                old_sig = cal['rf_mass_cal'].get(sig, {})
                rf_mass_cal_values[sig] = dict(old_sig) if isinstance(old_sig, dict) else {}
        else:
            rf_mass_cal_values = {
                'direct': {'column': 0, 'bp': 0},
                'uib': {'column': 0, 'bp': 0}
            }

        # Actualitzar mode/senyal concret
        rf_mass_cal_values[signal.lower()][mode.lower()] = round(result['rf_mass_cal'], 1)

        # Construir intercept preservant els valors existents
        if cal and isinstance(cal.get('intercept'), dict):
            intercept_values = {}
            for sig in ['direct', 'uib']:
                old_sig = cal['intercept'].get(sig, {})
                intercept_values[sig] = dict(old_sig) if isinstance(old_sig, dict) else {}
        else:
            intercept_values = {
                'direct': {'column': 0, 'bp': 0},
                'uib': {'column': 0, 'bp': 0}
            }

        if model == "origin":
            intercept_values[signal.lower()][mode.lower()] = 0
        else:
            intercept_values[signal.lower()][mode.lower()] = round(result['intercept'], 1)

        # R² i n_points com a dicts (preservar l'altre mode)
        r2_dict = {}
        npoints_dict = {}
        if cal:
            old_r2 = cal.get('r2', {})
            old_np = cal.get('n_points', {})
            if isinstance(old_r2, dict):
                r2_dict = dict(old_r2)
            if isinstance(old_np, dict):
                npoints_dict = dict(old_np)

        r2_dict[mode.lower()] = round(result['r2'], 4)
        npoints_dict[mode.lower()] = result['n_points']

        # Source info
        seq_refs = list({p['seq_name'] for p in result.get('points', [])})
        source = {
            'type': 'regression_from_history',
            'description': (
                f"Regressió {mode} {signal} ({model}): "
                f"RF={result['rf_mass_cal']:.1f}, "
                f"Int={result['intercept']:.1f}, "
                f"R²={result['r2']:.4f}, n={result['n_points']}"
            ),
            'seq_references': seq_refs,
        }

        valid_from = datetime.now().strftime('%Y-%m-%d')

        cal_id = add_calibration(
            rf_mass_cal_values=rf_mass_cal_values,
            source=source,
            valid_from=valid_from,
            r2=r2_dict,
            n_points=npoints_dict,
            reason=reason,
            intercept_values=intercept_values
        )

        if cal_id:
            QMessageBox.information(
                self,
                "Calibració aplicada",
                f"Nova calibració creada: {cal_id}\n\n"
                f"L'anterior s'ha tancat automàticament."
            )
            self._load_data()
            self.calibration_updated.emit()
        else:
            QMessageBox.warning(
                self,
                "Error",
                "No s'ha pogut guardar la nova calibració."
            )
