"""
HPSEC Suite - Review Summary Panel (Pas 4: Revisar)
=====================================================

Panel de revisió i validació abans de generar resultats.

Contingut:
- 3 cards informatives: Seqüència, Timeouts TOC, Control Qualitat
- Gràfic DOC ppm stacked per fraccions (BioP|HS|BB|SB|LMW)
- Gràfic A254 àrea per mostra
- Botó "Generar Resultats" → Excels a SEQ/RESULTATS/, SUMMARY a SEQ/CHECK/
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QScrollArea, QMessageBox, QSizePolicy, QProgressBar,
)
from PySide6.QtCore import Qt, Signal, QThread
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

    def showEvent(self, event):
        """Quan es mostra el panel, omplir amb dades actuals."""
        super().showEvent(event)
        processed_data = self.main_window.processed_data
        if processed_data and processed_data.get("success"):
            self.populate(processed_data)
