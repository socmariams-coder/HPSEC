"""
HPSEC Suite - Review Summary Panel (Pas 4: Revisar)
=====================================================

Panel de revisió i validació abans de generar resultats.

Contingut:
- 3 cards informatives: Seqüència, Timeouts TOC, Control Qualitat
- Gràfic DOC ppm stacked per fraccions (BioP|HS|BB|SB|LMW)
- Gràfic A254 àrea per mostra
- Botó "Generar Resultats" → Excels a SEQ/RESULTATS/, SUMMARY a SEQ/CHECK/
- SEQ_CAL: Resum regressió + botó "Aplicar com a Nova Calibració" + retroactiu
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QScrollArea, QMessageBox, QSizePolicy, QProgressBar,
    QGroupBox, QGridLayout, QCheckBox, QDateEdit
)
from PySide6.QtCore import Qt, Signal, QThread, QDate
from PySide6.QtGui import QFont, QColor

import json as _json
import logging
import os

import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from gui.widgets.styles import (
    COLOR_SUCCESS, COLOR_WARNING, COLOR_ERROR,
    apply_panel_layout
)

# Fraction colors (consistent palette)
FRACTION_COLORS = {
    "BioP": "#3498DB",  # Blue
    "HS":   "#E74C3C",  # Red
    "BB":   "#F39C12",  # Orange
    "SB":   "#2ECC71",  # Green
    "LMW":  "#9B59B6",  # Purple
}
FRACTION_ORDER = ["BioP", "HS", "BB", "SB", "LMW"]


class GenerateWorker(QThread):
    """Worker per generar resultats en background."""
    progress = Signal(int, str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, samples_grouped, seq_path, calibration_data, mode, config):
        super().__init__()
        self.samples_grouped = samples_grouped
        self.seq_path = seq_path
        self.calibration_data = calibration_data
        self.mode = mode
        self.config = config

    def run(self):
        try:
            from hpsec_export import export_sequence, generate_summary_excel, DEFAULT_EXPORT_CONFIG
            config = self.config or DEFAULT_EXPORT_CONFIG
            results = {"excel_files": None, "summary": None, "errors": []}

            def progress_cb(pct, msg):
                self.progress.emit(pct, msg)

            # Excels individuals → SEQ/RESULTATS/
            resultats_path = str(Path(self.seq_path) / "RESULTATS")
            self.progress.emit(0, "Generant Excels individuals...")
            excel_result = export_sequence(
                self.samples_grouped,
                resultats_path,
                self.calibration_data,
                self.mode,
                config,
                progress_cb,
                seq_path=self.seq_path,
            )
            results["excel_files"] = excel_result
            results["errors"].extend(excel_result.get("errors", []))

            # SUMMARY.xlsx → SEQ/CHECK/
            self.progress.emit(90, "Generant SUMMARY.xlsx...")
            check_path = Path(self.seq_path) / "CHECK"
            check_path.mkdir(parents=True, exist_ok=True)
            summary_path = str(check_path / "SUMMARY.xlsx")
            summary_result = generate_summary_excel(
                self.samples_grouped,
                summary_path,
                self.calibration_data,
                self.mode,
                config,
            )
            results["summary"] = summary_result

            self.progress.emit(100, "Completat")
            self.finished.emit(results)

        except Exception as e:
            self.error.emit(str(e))


class ReviewSummaryPanel(QWidget):
    """Panel de revisió i generació de resultats (Pas 4)."""

    review_completed = Signal(dict)

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.worker = None
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(0)

        # Scrollable content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        content = QWidget()
        self.content_layout = QVBoxLayout(content)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(12)
        scroll.setWidget(content)

        # === 3 CARDS SUPERIORS ===
        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(12)

        self.card_seq = self._create_card("SEQÜÈNCIA")
        self.card_timeouts = self._create_card("TIMEOUTS TOC")
        self.card_quality = self._create_card("CONTROL QUALITAT")

        cards_layout.addWidget(self.card_seq, 1)
        cards_layout.addWidget(self.card_timeouts, 1)
        cards_layout.addWidget(self.card_quality, 1)
        self.content_layout.addLayout(cards_layout)

        # === GRÀFICS ===
        if HAS_MATPLOTLIB:
            # DOC stacked bar
            self.doc_figure = Figure(figsize=(10, 3.5), dpi=100)
            self.doc_figure.set_facecolor("#FAFAFA")
            self.doc_canvas = FigureCanvas(self.doc_figure)
            self.doc_canvas.setMinimumHeight(220)
            self.content_layout.addWidget(self.doc_canvas)

            # A254 bar
            self.dad_figure = Figure(figsize=(10, 2.5), dpi=100)
            self.dad_figure.set_facecolor("#FAFAFA")
            self.dad_canvas = FigureCanvas(self.dad_figure)
            self.dad_canvas.setMinimumHeight(180)
            self.content_layout.addWidget(self.dad_canvas)

        # === SECCIÓ SEQ_CAL: APLICAR CALIBRACIÓ ===
        self._build_seq_cal_apply_section()

        # === BOTÓ GENERAR + PROGRÉS ===
        gen_frame = QFrame()
        gen_frame.setStyleSheet("""
            QFrame {
                background-color: #f0f7ff;
                border: 1px solid #b8d4f0;
                border-radius: 8px;
            }
        """)
        gen_layout = QVBoxLayout(gen_frame)
        gen_layout.setContentsMargins(20, 16, 20, 16)
        gen_layout.setSpacing(8)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.generate_btn = QPushButton("Generar Resultats")
        self.generate_btn.setStyleSheet("""
            QPushButton {
                background-color: #2980B9; color: white;
                border: none; border-radius: 6px;
                padding: 12px 32px; font-size: 14px; font-weight: bold;
            }
            QPushButton:hover { background-color: #3498DB; }
            QPushButton:disabled { background-color: #BDC3C7; }
        """)
        self.generate_btn.clicked.connect(self._run_generate)
        btn_row.addWidget(self.generate_btn)
        btn_row.addStretch()
        gen_layout.addLayout(btn_row)

        self.paths_label = QLabel("")
        self.paths_label.setAlignment(Qt.AlignCenter)
        self.paths_label.setStyleSheet("color: #7f8c8d; font-size: 11px; border: none;")
        gen_layout.addWidget(self.paths_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximum(100)
        gen_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #2c3e50; font-size: 11px; border: none;")
        self.status_label.setVisible(False)
        gen_layout.addWidget(self.status_label)

        self.content_layout.addWidget(gen_frame)
        self.content_layout.addStretch()

        main_layout.addWidget(scroll)

    # ------------------------------------------------------------------
    # Card factory
    # ------------------------------------------------------------------

    def _create_card(self, title):
        """Crea un card frame amb títol i label de contingut."""
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
            }
        """)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(4)

        title_label = QLabel(title)
        title_label.setStyleSheet("font-weight: bold; font-size: 11px; color: #555; border: none;")
        layout.addWidget(title_label)

        content_label = QLabel("")
        content_label.setWordWrap(True)
        content_label.setTextFormat(Qt.RichText)
        content_label.setStyleSheet("font-size: 12px; color: #2c3e50; border: none;")
        content_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        layout.addWidget(content_label, 1)

        card._content_label = content_label
        return card

    # ------------------------------------------------------------------
    # Populate from data
    # ------------------------------------------------------------------

    def populate(self, processed_data):
        """Omple el panel amb les dades processades."""
        if not processed_data:
            return

        samples_grouped = processed_data.get("samples_grouped", {})
        summary = processed_data.get("summary", {})
        method = processed_data.get("method", "COLUMN")
        seq_name = processed_data.get("seq_name", "")
        seq_path = processed_data.get("seq_path", "")
        is_bp = method.upper() == "BP"

        # Separate regular and light samples
        regular = {}
        light = {}
        for name, data in samples_grouped.items():
            if data.get("analysis_type") == "light":
                light[name] = data
            else:
                regular[name] = data

        # Count injections (replicas)
        n_injections = sum(
            len(d.get("replicas", {})) for d in samples_grouped.values()
        )

        # --- CARD 1: SEQÜÈNCIA ---
        self._populate_seq_card(seq_name, method, summary, regular, light, n_injections)

        # --- CARD 2: TIMEOUTS ---
        self._populate_timeout_card(processed_data, is_bp)

        # --- CARD 3: QUALITAT ---
        self._populate_quality_card(processed_data, regular, light, is_bp)

        # --- GRÀFICS ---
        if HAS_MATPLOTLIB:
            self._plot_doc_chart(regular, light, is_bp)
            self._plot_dad_chart(regular, light)

        # --- SEQ_CAL: APLICAR ---
        self._populate_seq_cal_apply()

        # --- PATHS ---
        if seq_path:
            self.paths_label.setText(
                f"Excels individuals → {seq_path}/RESULTATS/ &nbsp;&nbsp;|&nbsp;&nbsp; "
                f"SUMMARY.xlsx → {seq_path}/CHECK/"
            )

    def _populate_seq_card(self, seq_name, method, summary, regular, light, n_injections):
        n_blancs = sum(1 for d in light.values() if d.get("sample_type") == "BLANK")
        n_controls = sum(1 for d in light.values() if d.get("sample_type") == "CONTROL")
        n_errors = summary.get("with_anomalies", 0)
        n_warnings = summary.get("with_replica_warnings", 0)

        html = f"""
        <b>{seq_name}</b><br>
        Mètode: <b>{method}</b><br>
        <br>
        Injeccions: <b>{n_injections}</b><br>
        Mostres: <b>{len(regular)}</b><br>
        Blancs: <b>{n_blancs}</b> &nbsp; Controls: <b>{n_controls}</b><br>
        KHP: <b>{summary.get('n_khp', 0)}</b><br>
        <br>
        Anomalies: <b>{n_errors}</b><br>
        Avisos rèpliques: <b>{n_warnings}</b>
        """
        self.card_seq._content_label.setText(html.strip())

    def _populate_timeout_card(self, processed_data, is_bp):
        """Agrega estadístiques de timeout de totes les rèpliques."""
        all_samples = processed_data.get("samples", [])
        n_with_to = 0
        total_to = 0
        total_major = 0
        zone_totals = {}
        dt_medians = []

        for sample in all_samples:
            ti = sample.get("timeout_info", {})
            n = ti.get("n_timeouts", 0)
            if n > 0:
                n_with_to += 1
                total_to += n
                total_major += ti.get("n_major_timeouts", 0)
                zs = ti.get("zone_summary", {})
                for zone, count in zs.items():
                    zone_totals[zone] = zone_totals.get(zone, 0) + count
            dm = ti.get("dt_median_sec")
            if dm:
                dt_medians.append(dm)

        n_total_inj = len(all_samples)
        avg_cadence = np.median(dt_medians) if dt_medians else 0

        # Build zone lines with severity icons
        if is_bp:
            zone_defs = [
                ("BP_PEAK", "CRITICAL"), ("BP_TAIL", "WARNING"), ("POST_RUN", "OK")
            ]
        else:
            zone_defs = [
                ("RUN_START", "OK"), ("BioP", "WARNING"), ("HS", "CRITICAL"),
                ("BB", "WARNING"), ("SB", "WARNING"), ("LMW", "INFO"), ("POST_RUN", "OK")
            ]

        zone_lines = []
        for zone, sev in zone_defs:
            count = zone_totals.get(zone, 0)
            if sev == "CRITICAL":
                icon = f"<span style='color:{COLOR_ERROR}'>{'&#10007;' if count > 0 else '&#10003;'}</span>"
            elif sev == "WARNING":
                icon = f"<span style='color:{COLOR_WARNING}'>{'&#9888;' if count > 0 else '&#10003;'}</span>"
            elif sev == "INFO":
                icon = f"<span style='color:#3498DB'>{'&#8505;' if count > 0 else '&#10003;'}</span>"
            else:
                icon = f"<span style='color:{COLOR_SUCCESS}'>&#10003;</span>"
            zone_lines.append(f"&nbsp;&nbsp;{zone}: <b>{count}</b> {icon}")

        # Key verdict: HS free (COLUMN) or BP_PEAK free (BP)
        critical_zone = "BP_PEAK" if is_bp else "HS"
        critical_count = zone_totals.get(critical_zone, 0)
        if critical_count == 0:
            verdict = f"<span style='color:{COLOR_SUCCESS}'><b>Zona {critical_zone} lliure &#10003;</b></span>"
        else:
            verdict = f"<span style='color:{COLOR_ERROR}'><b>Zona {critical_zone} afectada ({critical_count}) &#10007;</b></span>"

        html = f"""
        Inj. amb timeout: <b>{n_with_to}/{n_total_inj}</b>
        ({n_with_to*100//max(n_total_inj,1)}%)<br>
        Total TO: <b>{total_to}</b> &nbsp; Majors (&#8805;70s): <b>{total_major}</b><br>
        Cadència: <b>{avg_cadence:.2f}s</b><br>
        <br>
        <b>Per zona:</b><br>
        {'<br>'.join(zone_lines)}<br>
        <br>
        {verdict}
        """
        self.card_timeouts._content_label.setText(html.strip())

    def _populate_quality_card(self, processed_data, regular, light, is_bp):
        """Card control qualitat: fases, KHP vs calibració, blancs, problemes."""
        lines = []

        # --- Semàfor per fase ---
        # Import
        imported = self.main_window.imported_data
        if imported and imported.get("success"):
            imp_wl = imported.get("warning_level", "none")
            lines.append(self._phase_line("Import", imp_wl))
        else:
            lines.append(self._phase_line("Import", "error"))

        # Calibrate
        cal_data = self.main_window.calibration_data
        if cal_data and cal_data.get("success"):
            cal_wl = cal_data.get("warning_level", "none")
            lines.append(self._phase_line("Calibr", cal_wl))
        else:
            lines.append(self._phase_line("Calibr", "error" if cal_data else "pending"))

        # Analyze
        ana_wl = processed_data.get("warning_level", "none")
        lines.append(self._phase_line("Anàlisi", ana_wl))

        lines.append("<br>")

        # --- SEQ_CAL: REGRESSIÓ (resum a la card) ---
        is_seq_cal = cal_data.get('is_seq_cal', False) if cal_data else False
        if is_seq_cal:
            wizard = getattr(self.main_window, 'process_panel', None)
            ana_panel = getattr(wizard, 'analyze_panel', None) if wizard else None
            seq_cal_reg = getattr(ana_panel, '_seq_cal_regression', None) if ana_panel else None
            if not seq_cal_reg:
                seq_cal_reg = cal_data.get('seq_cal_regression') if cal_data else None
            if seq_cal_reg and seq_cal_reg.get('success'):
                rf_new = seq_cal_reg.get('rf_mass_cal', 0)
                intercept_new = seq_cal_reg.get('intercept', 0)
                r2 = seq_cal_reg.get('r2', 0)
                n_pts = seq_cal_reg.get('n_points', 0)
                r2_color = COLOR_SUCCESS if r2 >= 0.99 else (COLOR_WARNING if r2 >= 0.95 else COLOR_ERROR)
                method_mode = processed_data.get("method", "COLUMN")
                applied = cal_data.get('seq_cal_applied', False)
                applied_icon = (
                    f"<span style='color:{COLOR_SUCCESS}'>&#10003; Aplicada</span>"
                    if applied else
                    "<span style='color:#7F8C8D'>Pendent</span>"
                )
                lines.append("<b>REGRESSIÓ SEQ_CAL</b><br>")
                lines.append(f"&nbsp;&nbsp;Mode: {method_mode} | Punts: {n_pts}<br>")
                lines.append(f"&nbsp;&nbsp;RF: <b>{rf_new:.1f}</b> | Intercept: {intercept_new:.1f}<br>")
                lines.append(f"&nbsp;&nbsp;R²: <span style='color:{r2_color}'><b>{r2:.6f}</b></span><br>")
                lines.append(f"&nbsp;&nbsp;Estat: {applied_icon}<br>")
                lines.append("<br>")

        # --- KHP vs CALIBRACIÓ ---
        khp_samples = processed_data.get("khp_samples", [])
        if khp_samples and cal_data:
            lines.append("<b>KHP vs CALIBRACIÓ</b><br>")
            rf = cal_data.get("rf_direct") or cal_data.get("rf", 0)
            conc = cal_data.get("khp_conc", 5)
            expected_area = rf * conc if rf and conc else 0

            for khp in khp_samples:
                khp_name = khp.get("name", "KHP")
                areas = khp.get("areas", {})
                doc_areas = areas.get("DOC", {})
                actual_area = doc_areas.get("total", 0)
                if expected_area > 0 and actual_area > 0:
                    dev_pct = (actual_area - expected_area) / expected_area * 100
                    color = COLOR_SUCCESS if abs(dev_pct) < 5 else (COLOR_WARNING if abs(dev_pct) < 10 else COLOR_ERROR)
                    lines.append(
                        f"&nbsp;&nbsp;{khp_name}: {actual_area:.0f} vs {expected_area:.0f} "
                        f"<span style='color:{color}'>({dev_pct:+.1f}%)</span><br>"
                    )
                elif actual_area > 0:
                    lines.append(f"&nbsp;&nbsp;{khp_name}: {actual_area:.0f}<br>")

            lines.append("<br>")

        # --- BLANCS ---
        blancs = {n: d for n, d in light.items() if d.get("sample_type") == "BLANK"}
        if blancs:
            lines.append("<b>BLANCS (MQ)</b><br>")
            for name, data in blancs.items():
                rep_key = data.get("selected", {}).get("doc", "1")
                rep = data.get("replicas", {}).get(rep_key, {})
                area = rep.get("area_total", 0)
                area_254 = rep.get("area_254", 0)
                # Low area for blank is good
                icon = f"<span style='color:{COLOR_SUCCESS}'>&#10003;</span>"
                if area and abs(area) > 500:
                    icon = f"<span style='color:{COLOR_WARNING}'>&#9888;</span>"
                a254_str = f" &nbsp; A254: {area_254:.0f}" if area_254 else ""
                lines.append(f"&nbsp;&nbsp;{name}: DOC {area:.0f} {icon}{a254_str}<br>")
            lines.append("<br>")

        # --- MOSTRES AMB PROBLEMES ---
        problems = []
        for name, data in regular.items():
            if data.get("sample_valid") is False:
                reason = "NO VÀLIDA"
                rec = data.get("recommendation", {})
                if rec:
                    reason = rec.get("doc", {}).get("reason", reason)
                problems.append(f"&nbsp;&nbsp;{name}: <span style='color:{COLOR_ERROR}'>{reason}</span>")
            else:
                # Check SNR < 10
                sel = data.get("selected", {}).get("doc", "1")
                rep = data.get("replicas", {}).get(sel, {})
                snr = (rep.get("snr_info") or {}).get("snr_direct", 0)
                if snr and 0 < snr < 10:
                    problems.append(f"&nbsp;&nbsp;{name}: <span style='color:{COLOR_WARNING}'>SNR={snr:.0f}</span>")

        if problems:
            lines.append("<b>MOSTRES AMB PROBLEMES</b><br>")
            for p in problems[:8]:  # Max 8 to keep compact
                lines.append(p + "<br>")
            if len(problems) > 8:
                lines.append(f"&nbsp;&nbsp;<i>...i {len(problems)-8} més</i><br>")

        self.card_quality._content_label.setText("".join(lines))

    def _phase_line(self, name, level):
        """Genera línia de semàfor per fase."""
        if level in ("none", None):
            return f"<span style='color:{COLOR_SUCCESS}'>&#9679;</span> {name}: OK<br>"
        elif level == "warning":
            return f"<span style='color:{COLOR_WARNING}'>&#9679;</span> {name}: avisos<br>"
        elif level in ("blocker", "error"):
            return f"<span style='color:{COLOR_ERROR}'>&#9679;</span> {name}: errors<br>"
        elif level == "pending":
            return f"<span style='color:#BDC3C7'>&#9679;</span> {name}: -<br>"
        return f"<span style='color:{COLOR_SUCCESS}'>&#9679;</span> {name}: OK<br>"

    # ------------------------------------------------------------------
    # SEQ_CAL: Aplicar Calibració
    # ------------------------------------------------------------------

    def _build_seq_cal_apply_section(self):
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

        # --- Resum regressió ---
        summary_grid = QGridLayout()
        summary_grid.setSpacing(6)

        summary_grid.addWidget(QLabel("<b>Resum regressió:</b>"), 0, 0, 1, 4)

        self._cal_rf_label = QLabel("RF: —")
        self._cal_intercept_label = QLabel("Intercept: —")
        self._cal_r2_label = QLabel("R²: —")
        self._cal_npts_label = QLabel("Punts: —")
        self._cal_mode_label = QLabel("Mode: —")
        self._cal_rms_label = QLabel("RMS: —")

        summary_grid.addWidget(self._cal_rf_label, 1, 0)
        summary_grid.addWidget(self._cal_intercept_label, 1, 1)
        summary_grid.addWidget(self._cal_r2_label, 1, 2)
        summary_grid.addWidget(self._cal_npts_label, 1, 3)
        summary_grid.addWidget(self._cal_mode_label, 2, 0)
        summary_grid.addWidget(self._cal_rms_label, 2, 1)

        layout.addLayout(summary_grid)

        # --- Comparació vigent vs nova ---
        self._cal_comparison_label = QLabel("")
        self._cal_comparison_label.setTextFormat(Qt.RichText)
        self._cal_comparison_label.setWordWrap(True)
        self._cal_comparison_label.setStyleSheet("font-size: 11px; background: transparent;")
        layout.addWidget(self._cal_comparison_label)

        # --- Equació ---
        self._cal_equation_label = QLabel("")
        self._cal_equation_label.setAlignment(Qt.AlignCenter)
        self._cal_equation_label.setStyleSheet("""
            font-family: Consolas, monospace; font-size: 11px;
            background-color: #e8f4fd; border: 1px solid #b3d7f0;
            border-radius: 4px; padding: 4px 8px;
        """)
        layout.addWidget(self._cal_equation_label)

        # --- Gràfic scatter miniatura ---
        if HAS_MATPLOTLIB:
            self._cal_mini_figure = Figure(figsize=(7, 3), dpi=100)
            self._cal_mini_figure.set_facecolor("#f0f7ff")
            self._cal_mini_canvas = FigureCanvas(self._cal_mini_figure)
            self._cal_mini_canvas.setMinimumHeight(200)
            self._cal_mini_canvas.setMaximumHeight(280)
            layout.addWidget(self._cal_mini_canvas)

        # --- Separator ---
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep)

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

        self.content_layout.addWidget(self.seq_cal_apply_group)

        # State
        self._seq_cal_regression = None
        self._seq_cal_method = None
        self._retro_seq_checkboxes = []
        self._cal_applied = False

    def _populate_seq_cal_apply(self):
        """Omple la secció SEQ_CAL amb dades de regressió."""
        cal_data = self.main_window.calibration_data
        if not cal_data or not cal_data.get('is_seq_cal', False):
            self.seq_cal_apply_group.setVisible(False)
            return

        # Buscar regressió al analyze panel (via wizard)
        wizard = getattr(self.main_window, 'process_panel', None)
        ana_panel = getattr(wizard, 'analyze_panel', None) if wizard else None
        seq_cal_reg = getattr(ana_panel, '_seq_cal_regression', None) if ana_panel else None

        # Fallback: buscar al calibration_data
        if not seq_cal_reg:
            seq_cal_reg = cal_data.get('seq_cal_regression')

        if not seq_cal_reg or not seq_cal_reg.get('success'):
            self.seq_cal_apply_group.setVisible(False)
            return

        self._seq_cal_regression = seq_cal_reg
        self._seq_cal_method = getattr(ana_panel, '_seq_cal_method', None) if ana_panel else None
        if not self._seq_cal_method:
            processed = self.main_window.processed_data
            self._seq_cal_method = processed.get("method", "COLUMN") if processed else "COLUMN"

        rf_new = seq_cal_reg.get('rf_mass_cal', 0)
        intercept_new = seq_cal_reg.get('intercept', 0)
        r2 = seq_cal_reg.get('r2', 0)
        n_pts = seq_cal_reg.get('n_points', 0)
        rms = seq_cal_reg.get('residuals_rms', 0)

        # Resum labels
        self._cal_rf_label.setText(f"RF: <b>{rf_new:.1f}</b>")
        self._cal_intercept_label.setText(f"Intercept: <b>{intercept_new:.1f}</b>")

        r2_color = COLOR_SUCCESS if r2 >= 0.99 else (COLOR_WARNING if r2 >= 0.95 else COLOR_ERROR)
        self._cal_r2_label.setText(f"R²: <b style='color:{r2_color}'>{r2:.6f}</b>")
        self._cal_npts_label.setText(f"Punts: <b>{n_pts}</b>")
        self._cal_mode_label.setText(f"Mode: <b>{self._seq_cal_method}</b>")
        self._cal_rms_label.setText(f"RMS: <b>{rms:.2f}</b>" if rms else "RMS: —")

        # Comparació vigent vs nova
        self._update_cal_comparison(rf_new, intercept_new, r2)

        # Gràfic miniatura
        if HAS_MATPLOTLIB:
            self._plot_cal_mini_scatter(seq_cal_reg)

        # Date: usar data de la SEQ si disponible
        seq_path = getattr(self.main_window, 'seq_path', '')
        if seq_path:
            seq_name = Path(seq_path).name
            # Intentar extreure data del nom o del manifest
            import_data = self.main_window.imported_data
            if import_data:
                seq_date = import_data.get('date')
                if seq_date:
                    try:
                        from datetime import datetime as dt
                        d = dt.strptime(str(seq_date)[:10], '%Y-%m-%d')
                        self._cal_valid_from.setDate(QDate(d.year, d.month, d.day))
                    except (ValueError, TypeError):
                        pass

        # Check si ja aplicada
        self._cal_applied = cal_data.get('seq_cal_applied', False)
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

    def _update_cal_comparison(self, rf_new, intercept_new, r2_new):
        """Taula HTML comparant calibració vigent vs nova."""
        from hpsec_calibrate import get_active_global_calibration
        from gui.widgets.analyze_panel._helpers import format_calibration_comparison_html

        cal = get_active_global_calibration()
        if not cal:
            self._cal_comparison_label.setText("<i>No hi ha calibració vigent per comparar.</i>")
            return

        method = self._seq_cal_method or "COLUMN"
        mode_key = "bp" if method.upper() == "BP" else "column"

        # Extreure valors vigents del dict anidat
        rf_dict = cal.get('rf_mass_cal', {})
        rf_vigent = rf_dict.get('direct', {}).get(mode_key, 0) if isinstance(rf_dict, dict) else 0

        int_dict = cal.get('intercept', 0)
        if isinstance(int_dict, dict):
            intercept_vigent = int_dict.get('direct', {}).get(mode_key, 0)
        else:
            intercept_vigent = float(int_dict) if int_dict else 0

        r2_dict = cal.get('r2', 0)
        if isinstance(r2_dict, dict):
            r2_vigent = r2_dict.get(mode_key, 0) or 0
        else:
            r2_vigent = float(r2_dict) if r2_dict else 0

        html = format_calibration_comparison_html(
            rf_vigent=rf_vigent, int_vigent=intercept_vigent,
            rf_new=rf_new, int_new=intercept_new,
            r2_new=r2_new, r2_vigent=r2_vigent,
        )
        self._cal_comparison_label.setText(html)

        # Actualitzar equació
        eq = f"Àrea = {rf_new:.1f} × µg_DOC + {intercept_new:.1f}   (R² = {r2_new:.6f})"
        self._cal_equation_label.setText(eq)

    def _plot_cal_mini_scatter(self, reg_result):
        """Gràfic scatter miniatura de la regressió."""
        self._cal_mini_figure.clear()
        ax = self._cal_mini_figure.add_subplot(111)

        points = reg_result.get('points', [])
        if not points:
            ax.text(0.5, 0.5, "Sense dades", ha='center', va='center', transform=ax.transAxes)
            self._cal_mini_canvas.draw()
            return

        x_inc = [p['ug_doc'] for p in points if not p.get('excluded')]
        y_inc = [p['area'] for p in points if not p.get('excluded')]
        x_exc = [p['ug_doc'] for p in points if p.get('excluded')]
        y_exc = [p['area'] for p in points if p.get('excluded')]

        if x_inc:
            ax.scatter(x_inc, y_inc, c='#2980B9', s=35, zorder=5,
                      edgecolors='white', linewidth=0.5, label='Inclòs')
        if x_exc:
            ax.scatter(x_exc, y_exc, c='#E74C3C', s=35, marker='x',
                      zorder=5, linewidth=1.5, label='Exclòs')

        # Recta regressió nova
        rf = reg_result.get('rf_mass_cal', 0)
        intercept = reg_result.get('intercept', 0)
        all_x = x_inc + x_exc
        if all_x and rf > 0:
            x_line = np.linspace(0, max(all_x) * 1.1, 100)
            y_line = rf * x_line + intercept
            ax.plot(x_line, y_line, '-', color='#27AE60', linewidth=1.5,
                    label=f'Nova (RF={rf:.0f})')

            # Banda de predicció 95%
            if len(x_inc) >= 3:
                try:
                    from gui.widgets.analyze_panel._helpers import compute_prediction_band
                    band = compute_prediction_band(x_line, rf, intercept,
                                                   np.array(x_inc), np.array(y_inc))
                    if band:
                        ax.fill_between(x_line, band[0], band[1],
                                       alpha=0.10, color='#2980B9')
                except Exception:
                    pass

        # Recta vigent (referència)
        try:
            from hpsec_calibrate import get_rf_mass_cal, get_calibration_intercept
            method = self._seq_cal_method or "COLUMN"
            mode_key = "bp" if method.upper() == "BP" else "column"
            cal_sig = (self._seq_cal_regression or {}).get('signal', 'direct')
            rf_vig = get_rf_mass_cal(signal=cal_sig, mode=mode_key) or 0
            int_vig = get_calibration_intercept(signal=cal_sig, mode=mode_key) or 0
            if rf_vig > 0 and all_x:
                x_line_v = np.linspace(0, max(all_x) * 1.1, 100)
                y_line_v = rf_vig * x_line_v + int_vig
                ax.plot(x_line_v, y_line_v, '--', color='#E67E22', linewidth=1,
                       alpha=0.7, label=f'Vigent (RF={rf_vig:.0f})')
        except Exception:
            pass

        # Equació com a text overlay
        r2 = reg_result.get('r2', 0)
        n_pts = reg_result.get('n_points', len(x_inc))
        if rf > 0:
            if abs(intercept) > 0.5:
                eq_text = f"A = {rf:.1f} × µg + {intercept:.1f}"
            else:
                eq_text = f"A = {rf:.1f} × µg"
            eq_text += f"  (R²={r2:.4f}, n={n_pts})"
            ax.text(0.03, 0.97, eq_text, transform=ax.transAxes,
                    fontsize=7, fontfamily='monospace', verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='#ccc', alpha=0.9))

        ax.set_xlabel('µg DOC', fontsize=8)
        ax.set_ylabel('Àrea', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(fontsize=7, loc='lower right')
        ax.grid(True, alpha=0.2)
        self._cal_mini_figure.tight_layout()
        self._cal_mini_canvas.draw()

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
        # Netejar
        for cb in self._retro_seq_checkboxes:
            cb.deleteLater()
        self._retro_seq_checkboxes = []

        valid_from = self._cal_valid_from.date().toString("yyyy-MM-dd")

        # Buscar SEQs amb analysis_result.json al data_folder
        from hpsec_config import get_config
        cfg = get_config()
        data_folder = cfg.get("paths", "data_folder", default="")
        if not data_folder or not Path(data_folder).is_dir():
            self._retro_info_label.setText("No s'ha trobat el data_folder.")
            return

        seq_list = []
        current_seq = getattr(self.main_window, 'seq_path', '')
        current_name = Path(current_seq).name if current_seq else ''

        for item in sorted(Path(data_folder).iterdir()):
            if not item.is_dir() or '_SEQ' not in item.name.upper():
                continue
            # Saltar la SEQ actual (la que estem processant)
            if item.name == current_name:
                continue
            # Saltar _CAL (no requantificar SEQs de calibració)
            if '_CAL' in item.name.upper():
                continue
            # Buscar analysis_result.json
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

        # Resum confirmació
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
                add_calibration, get_active_global_calibration,
                get_rf_mass_cal, get_calibration_intercept,
                requantify_analysis_json, compute_calibration_fingerprint
            )

            # Construir rf_mass_cal_values preservant l'altra branca
            current_cal = get_active_global_calibration()
            if current_cal:
                rf_values = dict(current_cal.get('rf_mass_cal', {}))
                intercept_values = current_cal.get('intercept', {})
                if isinstance(intercept_values, dict):
                    intercept_values = dict(intercept_values)
                else:
                    intercept_values = {"direct": {"column": 0, "bp": 0}}
            else:
                rf_values = {"direct": {"column": 0, "bp": 0}, "uib": {"column": 0, "bp": 0}}
                intercept_values = {"direct": {"column": 0, "bp": 0}, "uib": {"column": 0, "bp": 0}}

            # Deep copy per no mutar
            import copy
            rf_values = copy.deepcopy(rf_values)
            intercept_values = copy.deepcopy(intercept_values)

            # Determinar senyal (direct o uib) del resultat de la regressió
            cal_signal = (self._seq_cal_regression or {}).get('signal', 'direct')

            # Actualitzar branca corresponent al senyal
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
            seq_path = getattr(self.main_window, 'seq_path', '')
            seq_name = Path(seq_path).name if seq_path else 'unknown'
            source = {
                "type": "SEQ_CAL",
                "description": f"Regressió from {seq_name}",
                "seq_references": [seq_name],
                "mode": method,
            }

            # Preparar regression_data complet per persistir al JSON
            reg_data = dict(self._seq_cal_regression) if self._seq_cal_regression else {}
            reg_data['mode'] = method
            reg_data['signal'] = cal_signal
            reg_data['model'] = reg_data.get('model', 'intercept')

            # add_calibration (amb regression_data per persistència)
            cal_id = add_calibration(
                rf_mass_cal_values=rf_values,
                source=source,
                valid_from=valid_from,
                r2=r2,
                n_points=n_pts,
                reason=f"SEQ_CAL wizard: {seq_name}",
                intercept_values=intercept_values,
                regression_data=reg_data,
            )

            if not cal_id:
                raise RuntimeError("add_calibration ha retornat None")

            logger.info(f"Nova calibració aplicada: {cal_id} (RF={rf_new:.1f}, mode={method})")

            # Marcar com aplicada
            cal_data = self.main_window.calibration_data
            if cal_data:
                cal_data['seq_cal_applied'] = True

            # --- Requantificació retroactiva ---
            retro_results = []
            if retroactive and retro_count > 0:
                self._cal_apply_status.setText(f"Requantificant {retro_count} SEQs...")

                # Obtenir RF/intercept per a les dues branques
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
            self._cal_applied = True
            self._cal_report_btn.setVisible(True)

            # Refrescar dashboard si existeix
            if hasattr(self.main_window, 'dashboard_panel') and self.main_window.dashboard_panel:
                try:
                    self.main_window.dashboard_panel.refresh_sequences()
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
            from hpsec_calibrate import get_active_global_calibration

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
                # Obrir el PDF
                import subprocess
                try:
                    os.startfile(pdf_path)
                except AttributeError:
                    subprocess.Popen(['xdg-open', pdf_path])
            else:
                QMessageBox.warning(self, "Error", "No s'ha pogut generar l'informe.")
        except Exception as e:
            logger.error(f"Error generant informe calibració: {e}")
            QMessageBox.critical(self, "Error", f"Error generant informe:\n{e}")

    # ------------------------------------------------------------------
    # Charts
    # ------------------------------------------------------------------

    def _plot_doc_chart(self, regular, light, is_bp):
        """Gràfic DOC: barres stacked per fraccions (COLUMN) o simples (BP)."""
        self.doc_figure.clear()
        ax = self.doc_figure.add_subplot(111)

        names = []
        fractions_data = {f: [] for f in FRACTION_ORDER}
        ppm_values = []

        # Regular samples
        for name in sorted(regular.keys()):
            data = regular[name]
            sel = data.get("selected", {}).get("doc", "1")
            rep = data.get("replicas", {}).get(sel, {})
            areas = (rep.get("areas") or {}).get("DOC", {})
            quant = data.get("quantification") or {}

            names.append(name)
            for frac in FRACTION_ORDER:
                fractions_data[frac].append(areas.get(frac, 0))
            ppm_values.append(quant.get("concentration_ppm") or 0)

        # Light samples (simple area_total, grey)
        light_start = len(names)
        for name in sorted(light.keys()):
            data = light[name]
            sel = data.get("selected", {}).get("doc", "1")
            rep = data.get("replicas", {}).get(sel, {})
            area = rep.get("area_total", 0)

            names.append(name)
            for frac in FRACTION_ORDER:
                fractions_data[frac].append(0)
            ppm_values.append(0)
            # Store area for light bar overlay
            fractions_data["BioP"][-1] = area  # Use BioP slot for total area

        if not names:
            ax.text(0.5, 0.5, "Sense dades", ha='center', va='center', transform=ax.transAxes)
            self.doc_canvas.draw()
            return

        x = np.arange(len(names))
        bar_width = 0.7

        if is_bp:
            # BP mode: simple bars (total area)
            totals = [sum(fractions_data[f][i] for f in FRACTION_ORDER) for i in range(len(names))]
            colors = ['#95a5a6' if i >= light_start else '#3498DB' for i in range(len(names))]
            ax.bar(x, totals, bar_width, color=colors, edgecolor='white', linewidth=0.5)
        else:
            # COLUMN mode: stacked fractions
            bottom = np.zeros(len(names))
            for frac in FRACTION_ORDER:
                values = np.array(fractions_data[frac], dtype=float)
                colors = []
                for i in range(len(names)):
                    if i >= light_start:
                        colors.append('#B0B0B0')  # Grey for light
                    else:
                        colors.append(FRACTION_COLORS[frac])
                ax.bar(x, values, bar_width, bottom=bottom, color=colors,
                       edgecolor='white', linewidth=0.3, label=frac if light_start > 0 or True else None)
                bottom += values

            # Legend (only fraction labels, no duplicates)
            handles = [
                ax.bar(0, 0, color=FRACTION_COLORS[f], label=f)[0]
                for f in FRACTION_ORDER
            ]
            ax.legend(handles=handles, labels=FRACTION_ORDER,
                      loc='upper right', fontsize=8, framealpha=0.8, ncol=len(FRACTION_ORDER))

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel("Àrea DOC", fontsize=9)
        ax.set_title("DOC per mostra (fraccions)", fontsize=10, fontweight='bold')
        ax.tick_params(axis='y', labelsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        self.doc_figure.tight_layout()
        self.doc_canvas.draw()

    def _plot_dad_chart(self, regular, light):
        """Gràfic A254 per mostra (barres simples)."""
        self.dad_figure.clear()
        ax = self.dad_figure.add_subplot(111)

        names = []
        areas_254 = []
        colors = []

        # Regular
        for name in sorted(regular.keys()):
            data = regular[name]
            sel = data.get("selected", {}).get("dad", data.get("selected", {}).get("doc", "1"))
            rep = data.get("replicas", {}).get(sel, {})
            a254 = (rep.get("areas") or {}).get("A254", {}).get("total", 0)
            names.append(name)
            areas_254.append(a254)
            colors.append('#E74C3C')

        # Light
        for name in sorted(light.keys()):
            data = light[name]
            sel = data.get("selected", {}).get("doc", "1")
            rep = data.get("replicas", {}).get(sel, {})
            a254 = rep.get("area_254", 0)
            names.append(name)
            areas_254.append(a254)
            colors.append('#B0B0B0')

        if not names:
            ax.text(0.5, 0.5, "Sense dades", ha='center', va='center', transform=ax.transAxes)
            self.dad_canvas.draw()
            return

        x = np.arange(len(names))
        ax.bar(x, areas_254, 0.7, color=colors, edgecolor='white', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel("Àrea A254", fontsize=9)
        ax.set_title("A254 per mostra", fontsize=10, fontweight='bold')
        ax.tick_params(axis='y', labelsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        self.dad_figure.tight_layout()
        self.dad_canvas.draw()

    # ------------------------------------------------------------------
    # Generate results
    # ------------------------------------------------------------------

    def _run_generate(self):
        """Genera Excels individuals + SUMMARY."""
        processed_data = self.main_window.processed_data
        if not processed_data:
            QMessageBox.warning(self, "Avís", "No hi ha dades processades.")
            return

        samples_grouped = processed_data.get("samples_grouped", {})
        if not samples_grouped:
            QMessageBox.warning(self, "Avís", "No hi ha mostres per exportar.")
            return

        seq_path = self.main_window.seq_path or processed_data.get("seq_path", "")
        if not seq_path:
            QMessageBox.warning(self, "Avís", "No s'ha trobat el path de la seqüència.")
            return

        method = processed_data.get("method", "COLUMN")
        calibration_data = self.main_window.calibration_data

        self.generate_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setVisible(True)
        self.status_label.setText("Generant...")

        self.worker = GenerateWorker(
            samples_grouped, seq_path, calibration_data, method, None
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_progress(self, pct, msg):
        self.progress_bar.setValue(pct)
        self.status_label.setText(msg)

    def _on_finished(self, results):
        self.generate_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

        errors = results.get("errors", [])
        excel_result = results.get("excel_files", {})
        summary_result = results.get("summary", {})
        n_exported = excel_result.get("n_exported", 0) if excel_result else 0

        if errors:
            self.status_label.setText(f"Completat amb {len(errors)} errors")
            QMessageBox.warning(self, "Avisos", f"Errors durant la generació:\n" + "\n".join(errors[:5]))
        else:
            self.status_label.setText(f"{n_exported} Excels + SUMMARY generats correctament")

        self.review_completed.emit(results)

    def _on_error(self, error_msg):
        self.generate_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Error")
        QMessageBox.critical(self, "Error", f"Error durant la generació:\n{error_msg}")

    # ------------------------------------------------------------------
    # Reset / showEvent
    # ------------------------------------------------------------------

    def reset(self):
        """Reseteja el panel."""
        self.card_seq._content_label.setText("")
        self.card_timeouts._content_label.setText("")
        self.card_quality._content_label.setText("")
        if HAS_MATPLOTLIB:
            self.doc_figure.clear()
            self.doc_canvas.draw()
            self.dad_figure.clear()
            self.dad_canvas.draw()
        self.progress_bar.setVisible(False)
        self.status_label.setVisible(False)
        self.status_label.setText("")
        self.paths_label.setText("")
        self.generate_btn.setEnabled(True)

        # Reset SEQ_CAL
        self.seq_cal_apply_group.setVisible(False)
        self._seq_cal_regression = None
        self._seq_cal_method = None
        self._cal_applied = False
        self._cal_apply_btn.setEnabled(True)
        self._cal_apply_status.setText("")
        self._retro_frame.setVisible(False)
        self._cal_retroactive_chk.setChecked(False)
        for cb in self._retro_seq_checkboxes:
            cb.deleteLater()
        self._retro_seq_checkboxes = []
        if HAS_MATPLOTLIB and hasattr(self, '_cal_mini_figure'):
            self._cal_mini_figure.clear()
            self._cal_mini_canvas.draw()

    def showEvent(self, event):
        """Quan es mostra el panel, omplir amb dades actuals."""
        super().showEvent(event)
        processed_data = self.main_window.processed_data
        if processed_data and processed_data.get("success"):
            self.populate(processed_data)
