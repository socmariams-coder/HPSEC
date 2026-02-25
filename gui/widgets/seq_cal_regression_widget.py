"""
OBSOLET — Aquest widget NO s'utilitza. La funcionalitat equivalent està
implementada directament a global_calibration_panel.py (mètodes _on_seq_cal_*,
_plot_seq_cal_chromatogram, _run_seq_cal_regression, etc.).

Mantingut temporalment per referència. Es pot eliminar sense impacte.

---
Widget autònom per la regressió de SEQ_CAL (versió original, mai integrat).
Adaptat del codi eliminat al commit 973ea03 (analyze_panel ~950 línies).
"""

import logging
import os

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QGroupBox, QGridLayout, QCheckBox, QComboBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QScrollArea,
    QSizePolicy,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QColor, QBrush

logger = logging.getLogger(__name__)


class SeqCalRegressionWidget(QWidget):
    """Widget autònom de regressió per seqüències de calibració (SEQ_CAL).

    Signals:
        regression_updated(dict): Emès quan es recalcula la regressió.
            Dict conté: rf_mass_cal, intercept, r2, n_points, signal, ...
    """

    regression_updated = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._entries = []
        self._entries_direct = []
        self._entries_uib = []
        self._excluded = set()
        self._method = "COLUMN"
        self._signal = "direct"
        self._sensitivity = None
        self._regression = None
        self._seq_name = ""
        self._last_chrom_row = -1
        self._setup_ui()

    # =====================================================================
    # UI CONSTRUCTION
    # =====================================================================

    def _setup_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(10)
        layout.setContentsMargins(4, 4, 4, 4)

        # --- Empty state ---
        self._empty_label = QLabel(
            "<div style='text-align:center; padding:40px; color:#999;'>"
            "<span style='font-size:28px;'>📐</span><br><br>"
            "<b>Selecciona una SEQ_CAL al Dashboard</b><br>"
            "o fes clic a ↻ Reprocessar a la pestanya Recta de Calibració"
            "</div>"
        )
        self._empty_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._empty_label)

        # --- Main group (hidden until data loaded) ---
        self._main_group = QGroupBox("Regressió de Calibració (SEQ_CAL)")
        self._main_group.setVisible(False)
        self._main_group.setStyleSheet(
            "QGroupBox { font-weight: bold; color: #1A5276; border: 2px solid #27AE60; "
            "border-radius: 6px; margin-top: 8px; padding-top: 14px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; }"
        )
        main_layout = QVBoxLayout(self._main_group)
        main_layout.setSpacing(10)

        # Info detecció
        self._info_label = QLabel()
        self._info_label.setWordWrap(True)
        self._info_label.setStyleSheet(
            "background: #EBF5FB; border-radius: 4px; padding: 8px; "
            "color: #1A5276; font-weight: normal; font-size: 12px;"
        )
        main_layout.addWidget(self._info_label)

        # Selector senyal Direct/UIB + repair checkbox
        signal_frame = QFrame()
        signal_frame.setStyleSheet(
            "QFrame { background: #EBF5FB; border-radius: 4px; padding: 6px; }"
        )
        signal_layout = QHBoxLayout(signal_frame)
        signal_layout.setContentsMargins(8, 4, 8, 4)

        self._signal_label = QLabel("Senyal de calibració:")
        self._signal_label.setStyleSheet(
            "font-weight: bold; color: #1A5276; font-size: 11px;"
        )
        signal_layout.addWidget(self._signal_label)

        self._signal_combo = QComboBox()
        self._signal_combo.setMaximumWidth(220)
        self._signal_combo.setStyleSheet(
            "QComboBox { background: white; border: 1px solid #BDC3C7; "
            "border-radius: 3px; padding: 4px 8px; font-size: 11px; }"
        )
        self._signal_combo.currentIndexChanged.connect(self._on_signal_changed)
        signal_layout.addWidget(self._signal_combo)

        signal_layout.addSpacing(20)

        self._repair_check = QCheckBox("Usar àrea reparada")
        self._repair_check.setChecked(True)
        self._repair_check.setToolTip(
            "Quan activat, usa l'àrea corregida (paràbola) per pics amb cim irregular.\n"
            "Desactivar per usar l'àrea original sense reparació."
        )
        self._repair_check.setStyleSheet("font-size: 11px; color: #1A5276;")
        self._repair_check.stateChanged.connect(self._on_repair_toggled)
        signal_layout.addWidget(self._repair_check)

        signal_layout.addStretch()
        self._signal_frame = signal_frame
        main_layout.addWidget(signal_frame)

        # Warning sensibilitat UIB barrejada
        self._sensitivity_warning = QLabel()
        self._sensitivity_warning.setWordWrap(True)
        self._sensitivity_warning.setStyleSheet(
            "background: #FCF3CF; border: 1px solid #F39C12; border-radius: 4px; "
            "padding: 8px; color: #7D6608; font-size: 11px; font-weight: normal;"
        )
        self._sensitivity_warning.setVisible(False)
        main_layout.addWidget(self._sensitivity_warning)

        # Taula de punts de calibració
        self._points_table = QTableWidget()
        self._points_table.setColumnCount(12)
        self._points_table.setHorizontalHeaderLabels([
            "Sel", "Condició", "Conc (ppm)", "Vol (µL)", "µg DOC",
            "Àrea", "RF_mass", "A254", "DOC/254", "Anomalies", "Sens.", "Status"
        ])
        self._points_table.horizontalHeaderItem(0).setToolTip("Incloure punt a la regressió")
        self._points_table.horizontalHeaderItem(4).setToolTip("µg DOC injectat = ppm × µL / 1000")
        self._points_table.horizontalHeaderItem(6).setToolTip("RF_mass = Àrea × 1000 / (ppm × µL)")
        self._points_table.horizontalHeaderItem(7).setToolTip("Àrea integrada a 254nm (DAD)")
        self._points_table.horizontalHeaderItem(8).setToolTip("Ratio àrea DOC / àrea 254nm")
        self._points_table.horizontalHeaderItem(9).setToolTip("Indicadors d'anomalies detectades")
        self._points_table.horizontalHeaderItem(10).setToolTip(
            "Sensibilitat UIB (ppb) — només rellevant per senyal UIB"
        )
        self._points_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._points_table.setAlternatingRowColors(True)
        self._points_table.setMaximumHeight(220)
        self._points_table.verticalHeader().setVisible(False)
        self._points_table.setStyleSheet("""
            QTableWidget { font-size: 11px; gridline-color: #ddd; }
            QTableWidget::item { padding: 2px 4px; }
            QHeaderView::section {
                background-color: #f5f5f5; font-weight: bold;
                font-size: 10px; padding: 4px; border: none;
                border-bottom: 2px solid #ddd;
            }
        """)
        self._points_table.cellClicked.connect(self._on_row_clicked)
        self._points_table.cellDoubleClicked.connect(self._on_row_double_clicked)
        main_layout.addWidget(self._points_table)

        # Preview cromatograma (inicialment ocult)
        try:
            import matplotlib
            matplotlib.use('QtAgg')
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.figure import Figure
            self._chrom_figure = Figure(figsize=(8, 3), dpi=100)
            self._chrom_figure.set_facecolor("#FAFAFA")
            self._chrom_canvas = FigureCanvas(self._chrom_figure)
            self._chrom_canvas.setMinimumHeight(200)
            self._chrom_canvas.setMaximumHeight(250)
            self._chrom_canvas.setVisible(False)
            main_layout.addWidget(self._chrom_canvas)
            self._has_chrom = True
        except Exception:
            self._has_chrom = False

        # Resultats regressió
        reg_frame = QFrame()
        reg_frame.setStyleSheet(
            "QFrame { background: #F8F9FA; border: 1px solid #DEE2E6; "
            "border-radius: 4px; padding: 8px; }"
        )
        reg_grid = QGridLayout(reg_frame)
        reg_grid.setSpacing(6)

        self._reg_labels = {}
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
            self._reg_labels[key] = val
            reg_grid.addWidget(val, row, col + 1)

        main_layout.addWidget(reg_frame)

        # Comparació amb calibració vigent
        self._comparison_label = QLabel()
        self._comparison_label.setWordWrap(True)
        self._comparison_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._comparison_label.setStyleSheet(
            "padding: 8px; font-size: 12px; background: #FEFEFE; "
            "border: 1px solid #E0E0E0; border-radius: 4px;"
        )
        main_layout.addWidget(self._comparison_label)

        # Gràfic scatter amb matplotlib
        try:
            import matplotlib
            matplotlib.use('QtAgg')
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.figure import Figure
            self._scatter_figure = Figure(figsize=(8, 4), dpi=100)
            self._scatter_figure.set_facecolor("#FAFAFA")
            self._scatter_canvas = FigureCanvas(self._scatter_figure)
            self._scatter_canvas.setMinimumHeight(320)
            main_layout.addWidget(self._scatter_canvas)
            self._has_scatter = True
        except Exception:
            self._has_scatter = False
            fallback = QLabel("(Gràfic no disponible — instal·lar matplotlib)")
            main_layout.addWidget(fallback)

        # Botó recalcular
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self._recalc_btn = QPushButton("Recalcular")
        self._recalc_btn.setToolTip("Recalcular regressió amb els punts seleccionats")
        self._recalc_btn.clicked.connect(self._on_recalculate)
        self._recalc_btn.setStyleSheet(
            "QPushButton { background: #3498DB; color: white; border: none; "
            "border-radius: 4px; padding: 8px 16px; font-weight: bold; }"
            "QPushButton:hover { background: #2980B9; }"
        )
        btn_layout.addWidget(self._recalc_btn)
        main_layout.addLayout(btn_layout)

        layout.addWidget(self._main_group)
        scroll.setWidget(container)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    # =====================================================================
    # DATA LOADING — rep calib_result de CalSeqWorker
    # =====================================================================

    def set_data(self, calib_result, seq_name, seq_path="", imported_data=None):
        """Carrega dades de calibrate_from_import() i mostra la regressió.

        Args:
            calib_result: Dict de calibrate_from_import() amb
                calibrations_direct, calibrations_uib, etc.
            seq_name: Nom de la seqüència (ex: "292_SEQ_CAL_BP")
            seq_path: Path complet (per detectar mode)
            imported_data: Dict d'importació (per uib_sensitivity)
        """
        if not calib_result:
            self._main_group.setVisible(False)
            self._empty_label.setVisible(True)
            return

        self._seq_name = seq_name

        # Extreure calibracions per senyal
        cals_direct = calib_result.get('calibrations_direct', [])
        cals_uib = calib_result.get('calibrations_uib', [])
        cals = cals_direct or cals_uib or calib_result.get('calibrations', [])

        if not cals or len(cals) < 2:
            self._main_group.setVisible(False)
            self._empty_label.setText(
                "<div style='text-align:center; padding:40px; color:#E74C3C;'>"
                f"<b>{seq_name}</b><br><br>"
                f"Insuficients punts de calibració ({len(cals)}). Mínim 2."
                "</div>"
            )
            self._empty_label.setVisible(True)
            return

        # Determinar mode
        method = "COLUMN"
        if any(c.get('is_bp', False) for c in cals):
            method = "BP"
        elif imported_data and imported_data.get("method", "").upper() == "BP":
            method = "BP"
        if "_BP" in seq_name.upper():
            method = "BP"
        self._method = method

        # Sensibilitat UIB
        self._sensitivity = None
        if imported_data:
            self._sensitivity = imported_data.get("uib_sensitivity")

        # Construir entrades per senyal
        self._entries_direct = self._build_entries(cals_direct, 'direct', method, seq_name)
        self._entries_uib = self._build_entries(cals_uib, 'uib', method, seq_name)
        self._entries = self._entries_direct or self._entries_uib

        # Concentracions
        concs = set()
        for cal in cals:
            c = cal.get('conc_ppm', 0)
            if c > 0:
                concs.add(round(c, 4))

        # Auto-excloure punts UIB saturats
        self._excluded = set()
        for i, e in enumerate(self._entries):
            if e.get('uib_saturated'):
                self._excluded.add(i)

        # Configurar selector senyal
        has_direct = len(self._entries_direct) > 0
        has_uib = len(self._entries_uib) > 0
        has_both = has_direct and has_uib

        self._signal_combo.blockSignals(True)
        self._signal_combo.clear()
        if has_direct:
            self._signal_combo.addItem("DOC Direct", "direct")
        if has_uib:
            self._signal_combo.addItem("DOC UIB", "uib")
        self._signal = "direct" if has_direct else "uib"
        self._signal_combo.setCurrentIndex(0)
        self._signal_combo.blockSignals(False)

        self._signal_label.setVisible(has_both)
        self._signal_combo.setVisible(has_both)

        # Checkbox reparació: visible si hi ha entrades amb àrea reparada
        any_repaired = any(
            e.get('area_original') and e.get('area_original') != e.get('area', 0)
            for entries_list in (self._entries_direct, self._entries_uib)
            for e in entries_list
        )
        self._repair_check.setVisible(any_repaired)
        if any_repaired:
            self._repair_check.setChecked(True)
        self._signal_frame.setVisible(has_both or any_repaired)

        # Info
        conc_str = ", ".join(f"{c:g}" for c in sorted(concs))
        signals_str = []
        if has_direct:
            signals_str.append(f"Direct ({len(self._entries_direct)})")
        if has_uib:
            sens_info = f", sens={self._sensitivity} ppb" if self._sensitivity else ""
            signals_str.append(f"UIB ({len(self._entries_uib)}{sens_info})")
        self._info_label.setText(
            f"<b>Seqüència de calibració: {seq_name}</b> — "
            f"{len(self._entries)} punts, {len(concs)} concentracions "
            f"({conc_str} ppm), mode {method}<br>"
            f"Senyals disponibles: {', '.join(signals_str)}"
        )

        # Check barreja sensibilitats UIB
        self._check_uib_sensitivity_mixing()

        # Executar regressió
        self._run_regression()

        self._empty_label.setVisible(False)
        self._main_group.setVisible(True)

    def clear(self):
        """Neteja el widget i mostra l'estat buit."""
        self._entries = []
        self._entries_direct = []
        self._entries_uib = []
        self._excluded = set()
        self._regression = None
        self._main_group.setVisible(False)
        self._empty_label.setVisible(True)

    # =====================================================================
    # BUILD ENTRIES (from calibrate_from_import result)
    # =====================================================================

    def _build_entries(self, cal_list, signal_name, method, seq_name):
        """Construeix llista d'entrades de calibració per un senyal.

        Adaptat de l'antic _detect_seq_cal() de calibrate_panel.
        """
        entries = []
        for cal in cal_list:
            conc = cal.get('conc_ppm', 0)
            vol = cal.get('volume_uL', 0)
            area = cal.get('area', 0)
            if conc <= 0 or vol <= 0 or area <= 0:
                continue

            # Detectar saturació UIB: y_raw_max >= 95% sensibilitat
            # Prioritzar resultat del backend; fallback amb height+baseline (raw)
            uib_saturated = cal.get('uib_saturated', False)
            if not uib_saturated and signal_name == 'uib' and self._sensitivity:
                for rep in cal.get('replicas', []):
                    # height és NET (baseline-subtracted), cal sumar baseline per obtenir raw
                    h = rep.get('height', 0) or 0
                    bl = rep.get('baseline_stats', {}).get('mean', 0) or 0
                    y_raw_max = h + bl
                    if y_raw_max >= self._sensitivity * 0.95:
                        uib_saturated = True
                        break

            entry = {
                'seq_name': seq_name,
                'mode': method,
                'conc_ppm': conc,
                'volume_uL': vol,
                'area': area,
                'is_outlier': False,
                'valid_for_calibration': not uib_saturated if signal_name == 'uib' else True,
                'condition_key': cal.get('condition_key', f"KHP{conc:g}@{vol}µL"),
                'rf_mass': cal.get('rf_mass', 0),
                'quality_score': cal.get('quality_score', 0),
                'name_full': cal.get('name_full', ''),
                'a254_area': cal.get('a254_area', 0),
                'a254_doc_ratio': cal.get('a254_doc_ratio', 0),
                'has_irregular_top': cal.get('has_irregular_top', False),
                'irregular_top_repaired': cal.get('irregular_top_repaired', False),
                'area_uib': cal.get('area_uib', 0),
                'area_original': cal.get('area_original', 0),
                'area_repaired': cal.get('area_repaired', 0),
                'rf_mass_uib': cal.get('rf_mass_uib', 0),
                'has_timeout': cal.get('has_timeout', False),
                'timeout_severity': cal.get('timeout_severity', 'OK'),
                'uib_sensitivity': cal.get('uib_sensitivity'),
                'uib_saturated': uib_saturated,
                'quality_issues': cal.get('quality_issues', []),
                'replicas': cal.get('replicas', []),
            }
            if signal_name == 'uib':
                entry['area_u'] = area
            entries.append(entry)
        return entries

    # =====================================================================
    # UIB SENSITIVITY MIXING CHECK
    # =====================================================================

    def _check_uib_sensitivity_mixing(self):
        """Detecta barreja de sensibilitats UIB i auto-exclou la minoria."""
        if self._signal != "uib":
            self._sensitivity_warning.setVisible(False)
            return

        entries = self._entries
        if not entries:
            self._sensitivity_warning.setVisible(False)
            return

        sens_counts = {}
        for i, e in enumerate(entries):
            if i in self._excluded:
                continue
            s = e.get('uib_sensitivity')
            if s and s > 0:
                sens_counts.setdefault(s, []).append(i)

        unique_sens = sorted(sens_counts.keys())
        if len(unique_sens) <= 1:
            self._sensitivity_warning.setVisible(False)
            return

        majority_sens = max(sens_counts, key=lambda s: len(sens_counts[s]))
        minority_count = 0
        for s, indices in sens_counts.items():
            if s != majority_sens:
                for idx in indices:
                    self._excluded.add(idx)
                    minority_count += 1

        sens_str = ", ".join(f"{s:g} ppb ({len(sens_counts[s])} punts)" for s in unique_sens)
        self._sensitivity_warning.setText(
            f"⚠️ <b>Barreja de sensibilitats UIB detectada:</b> {sens_str}<br>"
            f"S'han auto-exclòs {minority_count} punt(s) amb sensibilitat ≠ {majority_sens:g} ppb. "
            f"Una regressió amb sensibilitats barrejades no seria vàlida."
        )
        self._sensitivity_warning.setVisible(True)

    # =====================================================================
    # REGRESSION
    # =====================================================================

    def _run_regression(self):
        """Executa la regressió i actualitza la UI."""
        from hpsec_calibrate import fit_calibration_from_history

        enabled = [e for i, e in enumerate(self._entries) if i not in self._excluded]

        if len(enabled) < 2:
            self._info_label.setText(
                f"<b>⚠ Insuficients punts ({len(enabled)})</b> — "
                "Mínim 2 punts per la regressió."
            )
            return

        reg_result = fit_calibration_from_history(
            enabled, mode=self._method, signal=self._signal, model="intercept"
        )
        reg_result['signal'] = self._signal
        reg_result['uib_sensitivity'] = self._sensitivity

        self._regression = reg_result
        self._update_ui()

        # Emetre senyal
        self.regression_updated.emit(reg_result)

    def _update_ui(self):
        """Actualitza tots els elements de la UI."""
        self._populate_table()

        reg = self._regression
        if reg and reg.get('success'):
            rf = reg['rf_mass_cal']
            intercept = reg['intercept']
            r2 = reg['r2']
            n_pts = reg['n_points']
            rms = reg.get('residuals_rms', 0)

            self._reg_labels['rf'].setText(f"<b>{rf:.1f}</b>")
            self._reg_labels['intercept'].setText(f"{intercept:.1f}")
            r2_color = '#27AE60' if r2 >= 0.99 else '#E67E22' if r2 >= 0.95 else '#E74C3C'
            self._reg_labels['r2'].setText(f"<b style='color: {r2_color}'>{r2:.6f}</b>")
            self._reg_labels['n_points'].setText(f"{n_pts}")
            self._reg_labels['rms'].setText(f"{rms:.2f}")
            self._reg_labels['model'].setText("intercept (lliure)")

            self._update_comparison(rf, intercept, r2)
            self._update_graph(reg)
        else:
            error = reg.get('error', 'Error desconegut') if reg else 'No result'
            for key in self._reg_labels:
                self._reg_labels[key].setText("-")
            self._comparison_label.setText(
                f"<i style='color: #E74C3C;'>Regressió fallida: {error}</i>"
            )

    # =====================================================================
    # TABLE
    # =====================================================================

    def _populate_table(self):
        """Omple la taula de punts de la regressió."""
        cal_entries = self._entries
        self._points_table.setRowCount(len(cal_entries))

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
            cb.setChecked(i not in self._excluded)
            cb.stateChanged.connect(lambda state, idx=i: self._on_point_toggled(idx, state))
            cb_widget = QWidget()
            cb_layout = QHBoxLayout(cb_widget)
            cb_layout.addWidget(cb)
            cb_layout.setAlignment(Qt.AlignCenter)
            cb_layout.setContentsMargins(0, 0, 0, 0)
            self._points_table.setCellWidget(i, 0, cb_widget)

            # Cols 1-6
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
                self._points_table.setItem(i, col, item)

            # Col 7: A254
            a254 = entry.get('a254_area', 0)
            a254_item = QTableWidgetItem(f"{a254:.0f}" if a254 else "-")
            a254_item.setFlags(a254_item.flags() & ~Qt.ItemIsEditable)
            self._points_table.setItem(i, 7, a254_item)

            # Col 8: DOC/254 ratio
            ratio = entry.get('a254_doc_ratio', 0)
            if not ratio and a254 and area:
                ratio = area / a254
            ratio_item = QTableWidgetItem(f"{ratio:.2f}" if ratio else "-")
            ratio_item.setFlags(ratio_item.flags() & ~Qt.ItemIsEditable)
            if ratio and (ratio < 0.5 or ratio > 10):
                ratio_item.setForeground(QBrush(QColor("#E67E22")))
            self._points_table.setItem(i, 8, ratio_item)

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
            self._points_table.setItem(i, 9, anomaly_item)

            # Col 10: Sensibilitat UIB
            sens = entry.get('uib_sensitivity')
            sens_item = QTableWidgetItem(f"{sens}" if sens else "-")
            sens_item.setFlags(sens_item.flags() & ~Qt.ItemIsEditable)
            self._points_table.setItem(i, 10, sens_item)

            # Col 11: Status badge
            badge_colors = {
                "OK": ("#D5F5E3", "#27AE60"),
                "CHECK": ("#FCF3CF", "#E67E22"),
                "INVALID": ("#FADBD8", "#E74C3C"),
            }
            bg, fg = badge_colors.get(status_text, ("#F0F0F0", "#666"))
            badge = QLabel(f" {status_text} ")
            badge.setAlignment(Qt.AlignCenter)
            badge.setStyleSheet(
                f"background: {bg}; color: {fg}; font-weight: bold; "
                f"font-size: 10px; border-radius: 3px; padding: 1px 6px;"
            )
            badge_w = QWidget()
            badge_l = QHBoxLayout(badge_w)
            badge_l.addWidget(badge)
            badge_l.setAlignment(Qt.AlignCenter)
            badge_l.setContentsMargins(2, 1, 2, 1)
            self._points_table.setCellWidget(i, 11, badge_w)

    # =====================================================================
    # CHROMATOGRAM PREVIEW
    # =====================================================================

    def _plot_chromatogram(self, ax, entry, zoom=False):
        """Dibuixa cromatograma amb R1+R2 superposades i àrees ombrejades.

        Args:
            ax: matplotlib Axes
            entry: dict amb 'replicas', 'conc_ppm', etc.
            zoom: Si True, fa zoom al voltant del pic principal.
        Returns:
            ax2 (eix secundari 254nm) o None
        """
        import numpy as np

        replicas = entry.get('replicas', [])
        use_repaired = self._repair_check.isChecked()

        doc_colors = ['#2196F3', '#1565C0']
        doc_styles = ['-', '--']
        fill_colors = ['#2196F3', '#1565C0']
        dad_colors = ['#9B59B6', '#8E44AD']
        dad_styles = ['-', ':']
        ax2 = None
        peak_times = []

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

                if y_repaired is not None:
                    y_repaired = np.asarray(y_repaired)
                    ax.plot(t_doc, y_repaired, color='#E74C3C', linewidth=1,
                            linestyle='--', label=f'{r_label} Reparat',
                            alpha=0.7 if r_idx == 0 else 0.4)

                peak_info = rep.get('peak_info', {})
                t_start = peak_info.get('t_start')
                t_end = peak_info.get('t_end')
                if t_start is not None and t_end is not None:
                    peak_times.extend([t_start, t_end])
                    mask = (t_doc >= t_start) & (t_doc <= t_end)
                    if np.any(mask):
                        if use_repaired and y_repaired is not None:
                            y_fill = y_repaired[mask]
                        else:
                            y_fill = y_doc[mask]
                        ax.fill_between(t_doc[mask], 0, y_fill,
                                        color=fill_colors[r_idx], alpha=0.12)
                    if r_idx == 0:
                        ax.axvline(t_start, color='gray', linewidth=0.5,
                                   linestyle=':', alpha=0.6)
                        ax.axvline(t_end, color='gray', linewidth=0.5,
                                   linestyle=':', alpha=0.6)

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

        if zoom and peak_times:
            t_center = (min(peak_times) + max(peak_times)) / 2
            peak_width = max(peak_times) - min(peak_times)
            margin = max(3.0 if self._method != "BP" else 2.0, peak_width)
            ax.set_xlim(t_center - margin, t_center + margin)

        conc = entry.get('conc_ppm', 0)
        name = entry.get('name_full', entry.get('condition_key', ''))
        n_rep = min(len(replicas), 2)
        repair_tag = " [reparat]" if use_repaired and entry.get('irregular_top_repaired') else ""
        zoom_tag = " — ZOOM" if zoom else ""
        ax.set_title(f"{name} ({conc:g} ppm) — {n_rep} rèpliques{repair_tag}{zoom_tag}",
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('Temps (min)', fontsize=9)
        ax.set_ylabel('Senyal DOC', fontsize=9, color='#2196F3')
        ax.tick_params(labelsize=8)

        lines, labels = ax.get_legend_handles_labels()
        if ax2:
            lines2, labels2 = ax2.get_legend_handles_labels()
            lines += lines2
            labels += labels2
        ax.legend(lines, labels, loc='upper right', fontsize=7, ncol=2)

        return ax2

    def _on_row_clicked(self, row, col):
        """Clic simple: preview cromatograma complet."""
        logger.info("_on_row_clicked: row=%d, col=%d, has_chrom=%s, n_entries=%d",
                     row, col, getattr(self, '_has_chrom', False), len(self._entries))
        if not getattr(self, '_has_chrom', False):
            logger.warning("_on_row_clicked: no chrom canvas")
            return
        if row < 0 or row >= len(self._entries):
            return

        self._last_chrom_row = row
        entry = self._entries[row]
        replicas = entry.get('replicas', [])
        logger.info("_on_row_clicked: entry conc=%s, n_replicas=%d",
                     entry.get('conc_ppm'), len(replicas))
        if not replicas:
            self._chrom_canvas.setVisible(False)
            return

        # Verificar que les rèpliques tenen dades de senyal
        for i, rep in enumerate(replicas[:2]):
            has_t = rep.get('t_doc') is not None
            has_y = rep.get('y_doc') is not None
            has_pi = bool(rep.get('peak_info'))
            logger.info("  R%d: t_doc=%s, y_doc=%s, peak_info=%s", i+1, has_t, has_y, has_pi)

        try:
            fig = self._chrom_figure
            fig.clear()
            ax = fig.add_subplot(111)
            self._plot_chromatogram(ax, entry, zoom=False)
            fig.tight_layout()
            self._chrom_canvas.setVisible(True)
            self._chrom_canvas.draw()
            logger.info("_on_row_clicked: chromatogram drawn OK")
        except Exception as e:
            logger.warning(f"Error preview cromatograma: {e}", exc_info=True)
            self._chrom_canvas.setVisible(False)

    def _on_row_double_clicked(self, row, col):
        """Doble clic: obre popup amb zoom al pic principal."""
        if row < 0 or row >= len(self._entries):
            return

        entry = self._entries[row]
        replicas = entry.get('replicas', [])
        if not replicas:
            return

        try:
            from PySide6.QtWidgets import QDialog, QVBoxLayout as QVBoxL
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.figure import Figure

            conc = entry.get('conc_ppm', 0)
            name = entry.get('name_full', entry.get('condition_key', ''))

            dlg = QDialog(self)
            dlg.setWindowTitle(f"Zoom pic — {name} ({conc:g} ppm)")
            dlg.resize(700, 450)
            layout = QVBoxL(dlg)
            layout.setContentsMargins(4, 4, 4, 4)

            fig = Figure(figsize=(9, 5), dpi=100)
            fig.set_facecolor("#FAFAFA")
            canvas = FigureCanvas(fig)
            layout.addWidget(canvas)

            ax = fig.add_subplot(111)
            self._plot_chromatogram(ax, entry, zoom=True)
            fig.tight_layout()
            canvas.draw()
            dlg.exec()
        except Exception as e:
            logger.warning(f"Error popup zoom: {e}")

    # =====================================================================
    # COMPARISON
    # =====================================================================

    def _update_comparison(self, new_rf, new_intercept, new_r2):
        """Mostra comparació amb calibració vigent."""
        from hpsec_calibrate import get_active_global_calibration
        from gui.widgets.analyze_panel._helpers import format_calibration_comparison_html

        current_cal = get_active_global_calibration()
        if not current_cal:
            self._comparison_label.setText("<i>No hi ha calibració vigent per comparar</i>")
            return

        signal = self._signal
        rf_cal = current_cal.get('rf_mass_cal', {})
        intercept_cal = current_cal.get('intercept', 0)

        if isinstance(rf_cal, dict):
            current_rf = rf_cal.get(signal, {}).get(self._method.lower(), 0)
        else:
            current_rf = float(rf_cal) if rf_cal else 0

        if isinstance(intercept_cal, dict):
            current_intercept = intercept_cal.get(signal, {}).get(self._method.lower(), 0)
        else:
            current_intercept = float(intercept_cal) if intercept_cal else 0

        current_r2 = current_cal.get('r2', {})
        if isinstance(current_r2, dict):
            current_r2_val = current_r2.get(self._method.lower(), 0) or 0
        else:
            current_r2_val = float(current_r2) if current_r2 else 0

        html = format_calibration_comparison_html(
            rf_vigent=current_rf, int_vigent=current_intercept,
            rf_new=new_rf, int_new=new_intercept,
            r2_new=new_r2, r2_vigent=current_r2_val,
            show_equation=True,
        )
        self._comparison_label.setText(html)

    # =====================================================================
    # SCATTER + RESIDUALS GRAPH
    # =====================================================================

    def _update_graph(self, reg_result):
        """Actualitza el gràfic scatter de regressió."""
        if not getattr(self, '_has_scatter', False):
            return
        try:
            import numpy as np

            entries = self._entries
            if not entries:
                self._scatter_figure.clear()
                self._scatter_canvas.draw()
                return

            self._scatter_figure.clear()
            gs = self._scatter_figure.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.35)
            ax_main = self._scatter_figure.add_subplot(gs[0])
            ax_res = self._scatter_figure.add_subplot(gs[1])

            excluded = self._excluded
            x_all, y_all, x_inc, y_inc, x_exc, y_exc = [], [], [], [], [], []
            labels_all, labels_inc = [], []
            for i, e in enumerate(entries):
                conc = e.get('conc_ppm', 0)
                vol = e.get('volume_uL', 0)
                x_val = conc * vol / 1000.0
                y_val = e.get('area', 0)
                x_all.append(x_val)
                y_all.append(y_val)
                lbl = f"{conc:g} ppm"
                labels_all.append(lbl)
                if i in excluded:
                    x_exc.append(x_val)
                    y_exc.append(y_val)
                else:
                    x_inc.append(x_val)
                    y_inc.append(y_val)
                    labels_inc.append(lbl)

            # Scatter per grups de concentració
            conc_groups = {}
            for xi, yi, lbl in zip(x_inc, y_inc, labels_inc):
                conc_groups.setdefault(lbl, ([], []))
                conc_groups[lbl][0].append(xi)
                conc_groups[lbl][1].append(yi)

            cmap = ['#2980B9', '#27AE60', '#8E44AD', '#E67E22', '#E74C3C', '#1ABC9C',
                     '#34495E', '#F39C12', '#D35400', '#7F8C8D']
            for idx_c, (lbl, (xs, ys)) in enumerate(sorted(conc_groups.items())):
                c = cmap[idx_c % len(cmap)]
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

            # Equació
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

            # Recta vigent (usa self._signal per respectar la selecció Direct/UIB)
            from hpsec_calibrate import get_rf_mass_cal, get_calibration_intercept
            current_rf = get_rf_mass_cal(signal=self._signal, mode=self._method.lower()) or 0
            current_intercept = get_calibration_intercept(signal=self._signal, mode=self._method.lower()) or 0
            if current_rf > 0:
                y_current = current_rf * x_line + current_intercept
                ax_main.plot(x_line, y_current, '--', color='#E67E22', linewidth=1.5, alpha=0.7,
                             label=f'Vigent: RF={current_rf:.0f}, int={current_intercept:.1f}')

            ax_main.set_xlabel('µg DOC injectat', fontsize=9)
            ax_main.set_ylabel('Àrea DOC', fontsize=9)
            ax_main.set_title(f'Recta de Calibració {self._method}', fontsize=10, fontweight='bold')
            ax_main.legend(fontsize=7, loc='lower right')
            ax_main.set_xlim(left=0)
            ax_main.set_ylim(bottom=min(0, min(y_all) - 10) if y_all else 0)
            ax_main.grid(True, alpha=0.3)
            ax_main.tick_params(labelsize=8)

            # Residuals (des de self._entries, exclòs els excluded)
            residuals = []
            inc_entries = []
            for i, e in enumerate(entries):
                if i in excluded:
                    continue
                x_val = e.get('conc_ppm', 0) * e.get('volume_uL', 0) / 1000.0
                y_val = e.get('area', 0)
                y_pred = new_rf * x_val + new_intercept
                residuals.append(y_val - y_pred)
                inc_entries.append(e)

            if residuals:
                rms = reg_result.get('residuals_rms', 0)
                colors = ['#27AE60' if abs(r) < rms * 2 else '#E67E22' if abs(r) < rms * 3 else '#E74C3C'
                          for r in residuals]
                ax_res.bar(range(len(residuals)), residuals, color=colors, alpha=0.8, edgecolor='white')
                ax_res.axhline(y=0, color='#333', linewidth=0.8)
                if rms > 0:
                    ax_res.axhline(y=rms, color='#E67E22', linewidth=0.5, linestyle='--', alpha=0.5)
                    ax_res.axhline(y=-rms, color='#E67E22', linewidth=0.5, linestyle='--', alpha=0.5)

                if inc_entries:
                    ax_res.set_xticks(range(len(inc_entries)))
                    ax_res.set_xticklabels([f"{e.get('conc_ppm', 0):g}" for e in inc_entries],
                                            fontsize=6, rotation=45)
                    ax_res.set_xlabel('ppm', fontsize=7)

                ax_res.set_title('Residuals', fontsize=9, fontweight='bold')
                ax_res.set_ylabel('Àrea obs - pred', fontsize=8)
                ax_res.tick_params(labelsize=7)
                ax_res.grid(True, alpha=0.2, axis='y')

            self._scatter_figure.tight_layout()
            self._scatter_canvas.draw()

        except Exception as e:
            logger.warning(f"Error actualitzant gràfic: {e}")
            try:
                self._scatter_figure.clear()
                self._scatter_canvas.draw()
            except Exception:
                pass

    # =====================================================================
    # INTERACTION HANDLERS
    # =====================================================================

    def _on_point_toggled(self, idx, state):
        """Quan l'usuari marca/desmarca un punt."""
        if state == 0:
            self._excluded.add(idx)
        else:
            self._excluded.discard(idx)

    def _on_signal_changed(self, index):
        """Quan l'usuari canvia el senyal (Direct/UIB)."""
        if index < 0:
            return
        signal = self._signal_combo.itemData(index)
        if not signal or signal == self._signal:
            return

        self._signal = signal

        if signal == "uib" and self._entries_uib:
            self._entries = self._entries_uib
        elif signal == "direct" and self._entries_direct:
            self._entries = self._entries_direct

        # Reset exclusions + auto-excloure saturats
        self._excluded = set()
        for i, e in enumerate(self._entries):
            if e.get('uib_saturated'):
                self._excluded.add(i)

        self._check_uib_sensitivity_mixing()
        self._run_regression()

    def _on_recalculate(self):
        """Recalcula la regressió amb els punts seleccionats."""
        if self._entries and self._method:
            self._run_regression()

    def _on_repair_toggled(self, state):
        """Toggle entre àrea reparada i àrea original."""
        use_repaired = (state != 0)

        changed = 0
        seen = set()
        for entries in (self._entries_direct, self._entries_uib):
            for entry in entries:
                if id(entry) in seen:
                    continue
                seen.add(id(entry))
                area_orig = entry.get('area_original')
                area_rep = entry.get('area_repaired')
                if not area_orig or not area_rep:
                    continue
                old_area = entry['area']
                if use_repaired:
                    entry['area'] = area_rep
                else:
                    entry['area'] = area_orig
                if entry['area'] != old_area:
                    changed += 1
                conc = entry.get('conc_ppm', 0)
                vol = entry.get('volume_uL', 0)
                if conc > 0 and vol > 0:
                    entry['rf_mass'] = entry['area'] * 1000.0 / (conc * vol)
                if 'area_u' in entry:
                    entry['area_u'] = entry['area']

        logger.info("Repair toggle: use_repaired=%s, changed=%d entries", use_repaired, changed)

        if self._entries and self._method:
            self._run_regression()

        # Redibuixar cromatograma si n'hi ha un visible
        last_row = getattr(self, '_last_chrom_row', -1)
        if last_row >= 0 and getattr(self, '_has_chrom', False):
            self._on_row_clicked(last_row, 0)

    # =====================================================================
    # PUBLIC GETTERS
    # =====================================================================

    @property
    def regression_result(self):
        """Retorna el resultat de regressió actual (o None)."""
        return self._regression

    @property
    def method(self):
        return self._method

    @property
    def signal(self):
        return self._signal
