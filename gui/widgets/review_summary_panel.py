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
    QTableWidget, QTableWidgetItem, QComboBox, QHeaderView,
    QGroupBox,
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


class BPDiscoveryWorker(QThread):
    """Worker per cercar dades BP en background."""
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, seq_path, sample_names, data_folder=None):
        super().__init__()
        self.seq_path = seq_path
        self.sample_names = sample_names
        self.data_folder = data_folder

    def run(self):
        try:
            from hpsec_consolidate import find_bp_for_samples
            result = find_bp_for_samples(
                self.seq_path, self.sample_names, self.data_folder
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class _BPReloadWorker(QThread):
    """Worker per recarregar dades BP des d'una BP diferent (canvi dropdown)."""
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, bp_path, sample_names, available_bps, column_seq_path):
        super().__init__()
        self.bp_path = bp_path
        self.sample_names = sample_names
        self.available_bps = available_bps
        self.column_seq_path = column_seq_path

    def run(self):
        try:
            from hpsec_consolidate import load_bp_data_for_sample
            result = {
                "primary_bp": {
                    "path": self.bp_path,
                    "name": Path(self.bp_path).name,
                },
                "available_bps": self.available_bps,
                "samples": {},
            }
            for name in self.sample_names:
                bp_data = load_bp_data_for_sample(self.bp_path, name)
                if bp_data:
                    result["samples"][name] = {
                        "bp_seq": Path(self.bp_path).name,
                        "bp_data": bp_data,
                        "source": "manual",
                    }
                else:
                    result["samples"][name] = {
                        "bp_seq": None,
                        "bp_data": None,
                        "source": None,
                    }
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class GenerateWorker(QThread):
    """Worker per generar resultats en background."""
    progress = Signal(int, str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, samples_grouped, seq_path, calibration_data, mode, config,
                 bp_resolved=None):
        super().__init__()
        self.samples_grouped = samples_grouped
        self.seq_path = seq_path
        self.calibration_data = calibration_data
        self.mode = mode
        self.config = config
        self.bp_resolved = bp_resolved

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
                bp_resolved=self.bp_resolved,
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
        self._bp_worker = None
        self._bp_resolved = None
        self._bp_available = []  # BPs disponibles per dropdown
        self._current_method = "COLUMN"
        self._current_seq_path = ""
        self._current_sample_names = []
        self._auto_generated = False  # Evitar doble generació
        self._populated_seq = ""  # Evitar doble populate
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

        # === SECCIÓ CONSOLIDACIÓ BP ===
        self.bp_group = QGroupBox("CONSOLIDACIÓ BP")
        self.bp_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold; font-size: 11px; color: #2c3e50;
                border: 1px solid #d5dbdb; border-radius: 6px;
                margin-top: 8px; padding-top: 18px;
                background-color: #fafafa;
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 12px; padding: 0 4px;
            }
        """)
        bp_layout = QVBoxLayout(self.bp_group)
        bp_layout.setContentsMargins(12, 8, 12, 8)
        bp_layout.setSpacing(6)

        # Dropdown per seleccionar BP
        bp_selector_row = QHBoxLayout()
        bp_selector_row.addWidget(QLabel("SEQ BP:"))
        self.bp_combo = QComboBox()
        self.bp_combo.setMinimumWidth(220)
        self.bp_combo.currentIndexChanged.connect(self._on_bp_combo_changed)
        bp_selector_row.addWidget(self.bp_combo)
        self.bp_status_label = QLabel("")
        self.bp_status_label.setStyleSheet("color: #7f8c8d; font-size: 11px;")
        bp_selector_row.addWidget(self.bp_status_label, 1)
        bp_layout.addLayout(bp_selector_row)

        # Taula de mostres BP
        self.bp_table = QTableWidget()
        self.bp_table.setColumnCount(5)
        self.bp_table.setHorizontalHeaderLabels(
            ["Mostra", "BP", "Rèplica", "ppm", "SNR"]
        )
        self.bp_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for col in range(1, 5):
            self.bp_table.horizontalHeader().setSectionResizeMode(
                col, QHeaderView.ResizeToContents
            )
        self.bp_table.verticalHeader().setVisible(False)
        self.bp_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.bp_table.setAlternatingRowColors(True)
        self.bp_table.setMaximumHeight(200)
        self.bp_table.setStyleSheet("""
            QTableWidget {
                font-size: 11px; border: 1px solid #e0e0e0;
                gridline-color: #f0f0f0;
            }
            QTableWidget::item { padding: 2px 6px; }
        """)
        bp_layout.addWidget(self.bp_table)

        # Info label (mostres sense BP, etc.)
        self.bp_info_label = QLabel("")
        self.bp_info_label.setStyleSheet("color: #7f8c8d; font-size: 10px; font-style: italic;")
        bp_layout.addWidget(self.bp_info_label)

        self.bp_group.setVisible(False)  # Oculta per defecte (només per COLUMN)
        self.content_layout.addWidget(self.bp_group)

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

        # Marcar seqüència populada (evitar re-populate a showEvent)
        self._populated_seq = seq_path or seq_name
        self._auto_generated = False

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

        # --- CONSOLIDACIÓ BP ---
        self._current_method = method
        self._current_seq_path = seq_path
        self._bp_resolved = None

        if method.upper() == "COLUMN" and seq_path:
            # Llançar cerca BP en background — auto-generate al callback
            self._current_sample_names = [
                name for name, d in regular.items()
                if d.get("analysis_type") != "khp"
            ]
            self.bp_group.setVisible(True)
            self.bp_status_label.setText("Cercant BP...")
            self.bp_table.setRowCount(0)
            self._launch_bp_discovery(seq_path, self._current_sample_names)
        else:
            self.bp_group.setVisible(False)
            # BP o mode sense consolidació: auto-generar directament
            self._auto_generate()

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
        zone_samples = {}  # zone → set(sample_name)
        dt_medians = []

        for sample in all_samples:
            sample_name = sample.get("name", "?")
            ti = sample.get("timeout_info", {})
            n = ti.get("n_timeouts", 0)
            if n > 0:
                n_with_to += 1
                total_to += n
                total_major += ti.get("n_major_timeouts", 0)
                zs = ti.get("zone_summary", {})
                for zone, count in zs.items():
                    zone_totals[zone] = zone_totals.get(zone, 0) + count
                    if count > 0:
                        zone_samples.setdefault(zone, set()).add(sample_name)
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
            line = f"&nbsp;&nbsp;{zone}: <b>{count}</b> {icon}"
            if count > 0 and zone in zone_samples:
                names = sorted(zone_samples[zone])
                if len(names) <= 5:
                    line += f" <span style='color:#888;font-size:10px'>({', '.join(names)})</span>"
                else:
                    line += f" <span style='color:#888;font-size:10px'>({', '.join(names[:4])}, +{len(names)-4})</span>"
            zone_lines.append(line)

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
        """Card control qualitat: resum mostres, problemes accionables, blancs."""
        lines = []

        # --- Resum mostres ---
        n_valid = sum(1 for d in regular.values() if d.get("sample_valid") is not False)
        n_invalid = sum(1 for d in regular.values() if d.get("sample_valid") is False)
        n_excluded = sum(
            1 for d in regular.values()
            if d.get("selected", {}).get("doc") == "Cap"
        )
        # Count SNR warnings among valid samples
        n_snr_warn = 0
        for data in regular.values():
            if data.get("sample_valid") is not False:
                sel = data.get("selected", {}).get("doc", "1")
                rep = data.get("replicas", {}).get(sel, {})
                snr = (rep.get("snr_info") or {}).get("snr_direct", 0)
                if snr and 0 < snr < 10:
                    n_snr_warn += 1
        n_problems = n_invalid + n_snr_warn

        parts = [f"<b>{n_valid}</b> vàlides"]
        if n_problems > 0:
            parts.append(f"<span style='color:{COLOR_WARNING}'><b>{n_problems}</b> problemes</span>")
        else:
            parts.append(f"<b>0</b> problemes")
        if n_excluded > 0:
            parts.append(f"<b>{n_excluded}</b> excloses")
        lines.append(f"Mostres: {' &middot; '.join(parts)}<br>")

        # --- PROBLEMES (accionables) ---
        problems_critical = []
        problems_warning = []
        for name, data in regular.items():
            if data.get("sample_valid") is False:
                reason = "NO VÀLIDA"
                rec = data.get("recommendation", {})
                if rec:
                    reason = rec.get("doc", {}).get("reason", reason)
                problems_critical.append(
                    f"&nbsp;&nbsp;&#10007; {name}: "
                    f"<span style='color:{COLOR_ERROR}'>{reason}</span>"
                )
            else:
                sel = data.get("selected", {}).get("doc", "1")
                rep = data.get("replicas", {}).get(sel, {})
                snr = (rep.get("snr_info") or {}).get("snr_direct", 0)
                if snr and 0 < snr < 10:
                    problems_warning.append(
                        f"&nbsp;&nbsp;&#9888; {name}: "
                        f"<span style='color:{COLOR_WARNING}'>SNR={snr:.0f}</span>"
                    )

        all_problems = problems_critical + problems_warning
        if all_problems:
            lines.append(f"<br><b>PROBLEMES ({len(all_problems)})</b><br>")
            for p in all_problems[:10]:
                lines.append(p + "<br>")
            if len(all_problems) > 10:
                lines.append(f"&nbsp;&nbsp;<i>...i {len(all_problems)-10} més</i><br>")

        # --- BLANCS ---
        blancs = {n: d for n, d in light.items() if d.get("sample_type") == "BLANK"}
        if blancs:
            lines.append(f"<br><b>BLANCS (MQ)</b><br>")
            for name, data in blancs.items():
                rep_key = data.get("selected", {}).get("doc", "1")
                rep = data.get("replicas", {}).get(rep_key, {})
                area = rep.get("area_total", 0)
                area_254 = rep.get("area_254", 0)
                icon = f"<span style='color:{COLOR_SUCCESS}'>&#10003;</span>"
                if area and abs(area) > 500:
                    icon = f"<span style='color:{COLOR_WARNING}'>&#9888;</span>"
                a254_str = f" &nbsp; A254: {area_254:.0f}" if area_254 else ""
                lines.append(f"&nbsp;&nbsp;{name}: DOC {area:.0f} {icon}{a254_str}<br>")

        self.card_quality._content_label.setText("".join(lines))


    # ------------------------------------------------------------------
    # BP Consolidation
    # ------------------------------------------------------------------

    def _launch_bp_discovery(self, seq_path, sample_names):
        """Llança BPDiscoveryWorker per cercar dades BP."""
        if self._bp_worker and self._bp_worker.isRunning():
            self._bp_worker.wait(2000)

        data_folder = str(Path(seq_path).parent)
        self._bp_worker = BPDiscoveryWorker(seq_path, sample_names, data_folder)
        self._bp_worker.finished.connect(self._on_bp_discovery_finished)
        self._bp_worker.error.connect(self._on_bp_discovery_error)
        self._bp_worker.start()

    def _on_bp_discovery_finished(self, result):
        """Gestiona el resultat de la cerca BP."""
        self._bp_resolved = result
        self._bp_available = result.get("available_bps", [])

        # Omplir dropdown
        self.bp_combo.blockSignals(True)
        self.bp_combo.clear()
        primary = result.get("primary_bp")
        selected_idx = 0

        if not self._bp_available:
            self.bp_combo.addItem("Cap BP trobada", None)
        else:
            for i, bp in enumerate(self._bp_available):
                bp_name = bp.get("name", "?")
                self.bp_combo.addItem(bp_name, bp.get("path"))
                if primary and bp.get("path") == primary.get("path"):
                    selected_idx = i

        self.bp_combo.setCurrentIndex(selected_idx)
        self.bp_combo.blockSignals(False)

        # Omplir taula
        self._populate_bp_table(result)

        # Auto-generar resultats ara que tenim BP resolt
        self._auto_generate()

    def _on_bp_discovery_error(self, error_msg):
        """Error durant la cerca BP."""
        logger.error(f"Error BP discovery: {error_msg}")
        self.bp_status_label.setText(f"Error: {error_msg}")
        self.bp_group.setVisible(False)

    def _populate_bp_table(self, bp_result):
        """Omple la taula de mostres BP amb el resultat de la cerca."""
        samples = bp_result.get("samples", {})
        n_linked = sum(1 for s in samples.values() if s.get("bp_data"))
        n_total = len(samples)

        self.bp_status_label.setText(f"({n_linked}/{n_total} mostres vinculades)")

        self.bp_table.setRowCount(n_total)

        # Ordenar: vinculades primer, després sense match
        sorted_names = sorted(
            samples.keys(),
            key=lambda n: (0 if samples[n].get("bp_data") else 1, n)
        )

        for row, name in enumerate(sorted_names):
            sdata = samples[name]
            bp_data = sdata.get("bp_data")

            # Col 0: Nom mostra
            item_name = QTableWidgetItem(name)
            self.bp_table.setItem(row, 0, item_name)

            if bp_data:
                # Col 1: Estat (✔ + font)
                source = sdata.get("source", "")
                source_tag = " *" if source == "name_search" else ""
                item_status = QTableWidgetItem(f"✔{source_tag}")
                item_status.setForeground(QColor(COLOR_SUCCESS))
                item_status.setTextAlignment(Qt.AlignCenter)
                if source == "name_search":
                    item_status.setToolTip(
                        f"Trobat per nom a {bp_data.get('seq_name', '?')}"
                    )
                self.bp_table.setItem(row, 1, item_status)

                # Col 2: Rèplica
                replica = bp_data.get("replica", "?")
                item_rep = QTableWidgetItem(f"R{replica}")
                item_rep.setTextAlignment(Qt.AlignCenter)
                self.bp_table.setItem(row, 2, item_rep)

                # Col 3: ppm
                ppm = bp_data.get("concentration_ppm")
                ppm_text = f"{ppm:.2f}" if ppm else "—"
                item_ppm = QTableWidgetItem(ppm_text)
                item_ppm.setTextAlignment(Qt.AlignCenter)
                self.bp_table.setItem(row, 3, item_ppm)

                # Col 4: SNR
                snr = bp_data.get("snr_direct")
                snr_text = f"{snr:.0f}" if snr else "—"
                item_snr = QTableWidgetItem(snr_text)
                item_snr.setTextAlignment(Qt.AlignCenter)
                self.bp_table.setItem(row, 4, item_snr)
            else:
                # Sense match — fila gris
                item_status = QTableWidgetItem("✘")
                item_status.setForeground(QColor("#bdc3c7"))
                item_status.setTextAlignment(Qt.AlignCenter)
                self.bp_table.setItem(row, 1, item_status)

                for col in range(2, 5):
                    item = QTableWidgetItem("—")
                    item.setForeground(QColor("#bdc3c7"))
                    item.setTextAlignment(Qt.AlignCenter)
                    self.bp_table.setItem(row, col, item)

                # Fons gris per la fila sencera
                for col in range(5):
                    it = self.bp_table.item(row, col)
                    if it:
                        it.setBackground(QColor("#f5f5f5"))

        # Info label
        n_missing = n_total - n_linked
        if n_missing > 0:
            self.bp_info_label.setText(
                f"{n_missing} mostr{'a' if n_missing == 1 else 'es'} sense dades BP"
            )
        else:
            self.bp_info_label.setText("Totes les mostres tenen dades BP vinculades")

    def _on_bp_combo_changed(self, index):
        """Quan l'usuari canvia la BP al dropdown, relança la cerca."""
        if index < 0 or not self._bp_available:
            return

        bp_path = self.bp_combo.currentData()
        if not bp_path:
            return

        # Relançar cerca amb la BP seleccionada com a primària
        self.bp_status_label.setText("Actualitzant...")

        if self._bp_worker and self._bp_worker.isRunning():
            self._bp_worker.wait(2000)

        # Carregar dades per cada mostra des de la BP seleccionada
        self._bp_worker = _BPReloadWorker(
            bp_path, self._current_sample_names,
            self._bp_available, self._current_seq_path
        )
        self._bp_worker.finished.connect(self._on_bp_reload_finished)
        self._bp_worker.error.connect(self._on_bp_discovery_error)
        self._bp_worker.start()

    def _on_bp_reload_finished(self, result):
        """Gestiona el resultat de la recàrrega BP (canvi de dropdown)."""
        self._bp_resolved = result
        self._populate_bp_table(result)

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

    def _auto_generate(self):
        """Auto-genera resultats si encara no s'ha fet per aquesta seqüència."""
        if self._auto_generated:
            return
        self._auto_generated = True
        self._run_generate(silent=True)

    def _run_generate(self, silent=False):
        """Genera Excels individuals + SUMMARY."""
        processed_data = self.main_window.processed_data
        if not processed_data:
            if not silent:
                QMessageBox.warning(self, "Avís", "No hi ha dades processades.")
            return

        samples_grouped = processed_data.get("samples_grouped", {})
        if not samples_grouped:
            if not silent:
                QMessageBox.warning(self, "Avís", "No hi ha mostres per exportar.")
            return

        seq_path = self.main_window.seq_path or processed_data.get("seq_path", "")
        if not seq_path:
            if not silent:
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
            samples_grouped, seq_path, calibration_data, method, None,
            bp_resolved=self._bp_resolved,
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
        self.generate_btn.setText("Regenerar Resultats")
        self.progress_bar.setVisible(False)

        errors = results.get("errors", [])
        excel_result = results.get("excel_files", {})
        summary_result = results.get("summary", {})
        n_exported = excel_result.get("n_exported", 0) if excel_result else 0

        if errors:
            self.status_label.setText(f"Completat amb {len(errors)} errors")
            self.status_label.setVisible(True)
            # Només mostrar diàleg si generació manual (no auto)
            if not self._auto_generated:
                QMessageBox.warning(self, "Avisos", f"Errors durant la generació:\n" + "\n".join(errors[:5]))
        else:
            self.status_label.setText(f"{n_exported} Excels + SUMMARY generats correctament")
            self.status_label.setVisible(True)

        # Escriure review_result.json
        self._write_review_result(results)

        self.review_completed.emit(results)

    def _on_error(self, error_msg):
        self.generate_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Error")
        QMessageBox.critical(self, "Error", f"Error durant la generació:\n{error_msg}")

    # ------------------------------------------------------------------
    # review_result.json
    # ------------------------------------------------------------------

    def _write_review_result(self, results):
        """Persisteix l'estat de la revisió a review_result.json."""
        try:
            from datetime import datetime

            seq_path = self._current_seq_path
            if not seq_path:
                return

            data_dir = Path(seq_path) / "CHECK" / "data"
            data_dir.mkdir(parents=True, exist_ok=True)

            excel_result = results.get("excel_files", {}) or {}
            n_exported = excel_result.get("n_exported", 0)
            n_skipped = excel_result.get("n_skipped", 0)

            # BP info
            bp_info = {}
            if self._bp_resolved:
                primary = self._bp_resolved.get("primary_bp")
                if primary:
                    bp_info["bp_seq_name"] = primary.get("name")
                    bp_info["bp_seq_path"] = primary.get("path")
                    bp_samples = self._bp_resolved.get("samples", {})
                    bp_info["n_linked"] = sum(
                        1 for s in bp_samples.values() if s.get("bp_data")
                    )
                    # mtime del analysis_result.json de la BP
                    bp_analysis = Path(primary["path"]) / "CHECK" / "data" / "analysis_result.json"
                    if bp_analysis.exists():
                        bp_info["bp_analysis_mtime"] = os.path.getmtime(str(bp_analysis))

            # Mostres descartades (sample_valid=False)
            processed_data = self.main_window.processed_data or {}
            samples_grouped = processed_data.get("samples_grouped", {})
            discarded = [
                name for name, d in samples_grouped.items()
                if d.get("sample_valid") is False
            ]

            review_data = {
                "success": not results.get("errors"),
                "timestamp": datetime.now().isoformat(),
                "version": "1.0",
                "seq_name": Path(seq_path).name,
                "method": self._current_method,
                "n_exported": n_exported,
                "n_skipped": n_skipped,
                "discarded_samples": discarded,
                "bp_info": bp_info,
                "summary_path": str(Path(seq_path) / "CHECK" / "SUMMARY.xlsx"),
            }

            review_path = data_dir / "review_result.json"
            with open(review_path, 'w', encoding='utf-8') as f:
                _json.dump(review_data, f, indent=2, ensure_ascii=False)

            logger.info(f"review_result.json escrit: {review_path}")
        except Exception as e:
            logger.error(f"Error escrivint review_result.json: {e}")

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
        self.generate_btn.setText("Generar Resultats")
        # BP state
        self._bp_resolved = None
        self._bp_available = []
        self.bp_group.setVisible(False)
        self.bp_table.setRowCount(0)
        self.bp_combo.clear()
        self.bp_status_label.setText("")
        self.bp_info_label.setText("")
        # Auto-generate state
        self._auto_generated = False
        self._populated_seq = ""

    def showEvent(self, event):
        """Quan es mostra el panel, omplir amb dades actuals."""
        super().showEvent(event)
        processed_data = self.main_window.processed_data
        if not processed_data or not processed_data.get("success"):
            return
        # Evitar re-populate si és la mateixa seqüència
        seq_id = processed_data.get("seq_path", "") or processed_data.get("seq_name", "")
        if seq_id and seq_id == self._populated_seq:
            return
        self.populate(processed_data)
