"""
HPSEC Suite — Quantify Panel (Pas 4)
=====================================

Aplica la recta de calibració global al resultat d'Analitzar:
    ppm = (Area − intercept) × 1000 / (rf_mass_cal × volume_uL)

Inputs:
- analysis_result d'Analitzar (sense quantification, només àrees)
- Calibration_Reference.json (recta activa per àmbit signal/sensitivity)

Outputs:
- analysis_result enriquit amb sample_group["quantification"] per cada mostra
- Bar charts DOC stacked per fraccions (COLUMN) o simples (BP)
- Bar chart DAD (à254 per defecte)
- Taula ppm DOC Direct/UIB + DAD + HCI + ratios

UI:
- Capçalera: info de la recta vigent (rf_mass_cal, intercept, R², n punts)
- Botó "Aplicar / Refrescar quantificació"
- Visualitzacions (bar charts + taula)
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame,
    QGroupBox, QTableWidget, QTableWidgetItem, QHeaderView, QScrollArea,
    QSizePolicy
)
from PySide6.QtCore import Qt, Signal, QThread, Slot
from PySide6.QtGui import QFont

import json
import logging
import os

logger = logging.getLogger(__name__)

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class QuantifyPanel(QWidget):
    """Panell del pas Quantificar."""

    quantification_completed = Signal(dict)  # Emès quan acaba la quantificació

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self._analysis_result = None
        self._quantification_result = None
        self._worker = None
        self._setup_ui()

    # ---------------------------------------------------------------- UI

    def _setup_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        outer.addWidget(scroll)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # === EMPTY STATE ===
        self.empty_state = QLabel(
            "Cal completar Analitzar abans de quantificar.\n"
            "Quan acabis l'anàlisi, aquest pas aplicarà la recta de calibració."
        )
        self.empty_state.setStyleSheet(
            "color: #6c757d; padding: 40px; background: #f8f9fa; "
            "border: 1px solid #dee2e6; border-radius: 6px; font-size: 12px;")
        self.empty_state.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.empty_state)

        # === HEADER: info recta calibració ===
        self._cal_info_group = QGroupBox("Recta de calibració vigent")
        self._cal_info_group.setVisible(False)
        cal_layout = QVBoxLayout(self._cal_info_group)
        self._cal_info_label = QLabel()
        self._cal_info_label.setStyleSheet("font-size: 11px; padding: 4px;")
        self._cal_info_label.setTextFormat(Qt.RichText)
        self._cal_info_label.setWordWrap(True)
        cal_layout.addWidget(self._cal_info_label)
        layout.addWidget(self._cal_info_group)

        # === BOTÓ APLICAR + GENERAR PDF ===
        btn_row = QHBoxLayout()
        self._apply_btn = QPushButton("▶ Aplicar quantificació")
        self._apply_btn.setStyleSheet(
            "QPushButton { background: #2E86AB; color: white; padding: 8px 16px; "
            "border: none; border-radius: 4px; font-size: 12px; font-weight: bold; }"
            "QPushButton:hover { background: #1f6080; }"
            "QPushButton:disabled { background: #adb5bd; }")
        self._apply_btn.clicked.connect(self._run_quantify)
        self._apply_btn.setVisible(False)
        btn_row.addWidget(self._apply_btn)

        # Botó PDF (només actiu després d'aplicar)
        self._pdf_btn = QPushButton("📄 Generar PDF")
        self._pdf_btn.setStyleSheet(
            "QPushButton { background: white; color: #2E86AB; padding: 8px 14px; "
            "border: 1px solid #2E86AB; border-radius: 4px; font-size: 11px; }"
            "QPushButton:hover { background: #EBF5FB; }"
            "QPushButton:disabled { color: #adb5bd; border-color: #ced4da; }")
        self._pdf_btn.setEnabled(False)
        self._pdf_btn.setVisible(False)
        self._pdf_btn.clicked.connect(self._generate_pdf)
        btn_row.addWidget(self._pdf_btn)

        btn_row.addStretch()
        self._status_label = QLabel()
        self._status_label.setStyleSheet("font-size: 11px; color: #555;")
        btn_row.addWidget(self._status_label)
        layout.addLayout(btn_row)

        # === CHARTS ===
        self._charts_frame = QFrame()
        self._charts_frame.setVisible(False)
        charts_layout = QVBoxLayout(self._charts_frame)
        charts_layout.setContentsMargins(0, 0, 0, 0)

        if HAS_MATPLOTLIB:
            self._doc_figure = Figure(figsize=(10, 3.2), dpi=100)
            self._doc_canvas = FigureCanvas(self._doc_figure)
            self._doc_canvas.setMinimumHeight(220)
            charts_layout.addWidget(self._doc_canvas)

            self._dad_figure = Figure(figsize=(10, 2.5), dpi=100)
            self._dad_canvas = FigureCanvas(self._dad_figure)
            self._dad_canvas.setMinimumHeight(180)
            charts_layout.addWidget(self._dad_canvas)
        layout.addWidget(self._charts_frame)

        # === TAULA RESULTATS ===
        self._table = QTableWidget()
        self._table.setVisible(False)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setStyleSheet(
            "QTableWidget { font-size: 11px; }"
            "QHeaderView::section { background: #f1f3f5; padding: 4px; "
            "border: 1px solid #dee2e6; font-weight: bold; }"
        )
        layout.addWidget(self._table)

        layout.addStretch()
        scroll.setWidget(content)

    # ---------------------------------------------------------------- API

    def load(self, analysis_result: dict):
        """Carrega el resultat d'Analitzar i prepara la UI."""
        self._analysis_result = analysis_result
        if not analysis_result or not analysis_result.get("samples_grouped"):
            self.empty_state.setVisible(True)
            self._cal_info_group.setVisible(False)
            self._apply_btn.setVisible(False)
            self._pdf_btn.setVisible(False)
            self._charts_frame.setVisible(False)
            self._table.setVisible(False)
            return

        self.empty_state.setVisible(False)
        self._apply_btn.setVisible(True)
        self._pdf_btn.setVisible(True)
        self._pdf_btn.setEnabled(False)  # només habilitat tras quantificar
        self._update_cal_info()

        # Si la quantificació ja està feta (do_quantify=True), mostrar-la directament
        if not analysis_result.get("quantification_pending", True):
            self._quantification_result = analysis_result
            self._pdf_btn.setEnabled(True)
            self._render_results()

    def _update_cal_info(self):
        """Actualitza la capçalera amb la info de la recta vigent."""
        try:
            from hpsec_calibrate import (
                get_active_global_calibration, get_rf_mass_cal,
                get_calibration_intercept
            )
            method = (self._analysis_result.get("method") or "COLUMN").lower()
            mode_key = "bp" if method == "bp" else "column"

            rf_dir = get_rf_mass_cal(signal='direct', mode=mode_key)
            int_dir = get_calibration_intercept(signal='direct', mode=mode_key)
            rf_uib = get_rf_mass_cal(signal='uib', mode=mode_key)
            int_uib = get_calibration_intercept(signal='uib', mode=mode_key)

            parts = []
            if rf_dir is not None:
                parts.append(
                    f"<b>Direct/{mode_key.upper()}:</b> "
                    f"RF={rf_dir:.1f}, intercept={int_dir:.2f}")
            if rf_uib is not None:
                parts.append(
                    f"<b>UIB/{mode_key.upper()}:</b> "
                    f"RF={rf_uib:.1f}, intercept={int_uib:.2f}")
            if not parts:
                self._cal_info_label.setText(
                    "<span style='color:#c0392b'>⚠ No hi ha recta de calibració activa.</span>")
                self._apply_btn.setEnabled(False)
            else:
                self._cal_info_label.setText(" · ".join(parts))
                self._apply_btn.setEnabled(True)
            self._cal_info_group.setVisible(True)
        except Exception as e:
            logger.warning("Error carregant info calibració: %s", e)
            self._cal_info_label.setText(
                f"<span style='color:#c0392b'>Error: {e}</span>")
            self._cal_info_group.setVisible(True)

    def _run_quantify(self):
        """Llança la quantificació en un thread separat."""
        if not self._analysis_result:
            self._status_label.setText(
                "<span style='color:#c0392b'>✗ Cal completar Analitzar abans.</span>")
            self._status_label.setTextFormat(Qt.RichText)
            return
        self._apply_btn.setEnabled(False)
        self._status_label.setText("Quantificant…")

        # v2.2.0: passar seq_path i mode explícitament (no confiar només en
        # els valors al JSON, que poden ser basename).
        seq_path = getattr(self.main_window, 'seq_path', None) \
                   or self._analysis_result.get('seq_path')
        method = self._analysis_result.get('method', 'COLUMN')
        mode = 'BP' if method.upper() == 'BP' else 'COLUMN'

        from .worker import QuantifyWorker
        self._worker = QuantifyWorker(
            self._analysis_result, seq_path=seq_path, mode=mode)
        self._worker.completed.connect(self._on_quantify_completed)
        self._worker.error.connect(self._on_quantify_error)
        self._worker.progress.connect(self._on_quantify_progress)
        self._worker.start()

    @Slot(str, int)
    def _on_quantify_progress(self, msg: str, pct: int):
        """Mostra progrés de la quantificació al status label."""
        self._status_label.setText(f"{msg} ({pct}%)")

    @Slot(dict)
    def _on_quantify_completed(self, result: dict):
        self._quantification_result = result
        self._apply_btn.setEnabled(True)
        self._pdf_btn.setEnabled(True)  # ara que tenim ppm, PDF disponible
        self._status_label.setText(
            f"<span style='color:#28a745'>✓ Quantificat ({len(result.get('samples_grouped', {}))} mostres)</span>"
        )
        self._status_label.setTextFormat(Qt.RichText)

        # Persistir al JSON
        self._persist_result()

        # Render UI
        self._render_results()

        # Notificar al wizard
        self.quantification_completed.emit(result)

    def _generate_pdf(self):
        """Genera PDF d'anàlisi amb ppm aplicats (v2.2.0)."""
        if not self._quantification_result:
            return
        seq_path = getattr(self.main_window, 'seq_path', None) \
                   or self._quantification_result.get('seq_path')
        if not seq_path:
            self._status_label.setText(
                "<span style='color:#c0392b'>✗ Sense seq_path per generar PDF</span>")
            self._status_label.setTextFormat(Qt.RichText)
            return

        self._pdf_btn.setEnabled(False)
        self._status_label.setText("Generant PDF…")
        try:
            from hpsec_reports import generate_analysis_report
            pdf_path = generate_analysis_report(
                seq_path, analysis_data=self._quantification_result)
            if pdf_path:
                self._status_label.setText(
                    f"<span style='color:#28a745'>✓ PDF generat: {pdf_path}</span>")
                self._status_label.setTextFormat(Qt.RichText)
                # Obrir automàticament
                import os
                try:
                    os.startfile(pdf_path)
                except Exception:
                    pass
            else:
                self._status_label.setText(
                    "<span style='color:#c0392b'>✗ Error generant PDF</span>")
                self._status_label.setTextFormat(Qt.RichText)
        except Exception as e:
            logger.exception("Error generating PDF")
            self._status_label.setText(
                f"<span style='color:#c0392b'>✗ Error: {e}</span>")
            self._status_label.setTextFormat(Qt.RichText)
        finally:
            self._pdf_btn.setEnabled(True)

    @Slot(str)
    def _on_quantify_error(self, msg: str):
        self._apply_btn.setEnabled(True)
        self._status_label.setText(
            f"<span style='color:#c0392b'>✗ Error: {msg}</span>")
        self._status_label.setTextFormat(Qt.RichText)

    def _persist_result(self):
        """Guarda el resultat enriquit al JSON d'Analitzar."""
        if not self._quantification_result:
            return
        seq_path = self._quantification_result.get("seq_path")
        if not seq_path:
            return
        json_path = os.path.join(seq_path, "CHECK", "data", "analysis_result.json")
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(self._quantification_result, f, indent=2,
                          default=str, ensure_ascii=False)
            logger.info("Quantificació persistida: %s", json_path)
        except Exception as e:
            logger.error("Error persistint quantification: %s", e)

    def _render_results(self):
        """Dibuixa bar charts + taula."""
        if not self._quantification_result:
            return
        self._charts_frame.setVisible(True)
        self._table.setVisible(True)
        if HAS_MATPLOTLIB:
            self._plot_doc_bars()
            self._plot_dad_bars()
        self._populate_table()

    def _plot_doc_bars(self):
        """Bar chart stacked DOC per fraccions (COLUMN) o simples (BP)."""
        self._doc_figure.clear()
        ax = self._doc_figure.add_subplot(111)

        samples = self._quantification_result.get("samples_grouped", {})
        method = (self._quantification_result.get("method") or "COLUMN").upper()
        is_bp = method == "BP"

        names, totals = [], []
        for name, sg in samples.items():
            q = sg.get("quantification") or {}
            if not q.get("valid", True):
                continue
            ppm = q.get("concentration_ppm_direct") or q.get("concentration_ppm")
            if ppm is None:
                continue
            names.append(name[:20])
            totals.append(ppm)

        if names:
            xs = list(range(len(names)))
            ax.bar(xs, totals, color="#2E86AB", alpha=0.85)
            ax.set_xticks(xs)
            ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
            ax.set_ylabel("ppm DOC Direct", fontsize=9)
            ax.set_title(f"DOC Direct per mostra ({method})", fontsize=10)
            ax.grid(True, axis='y', alpha=0.3)
        else:
            ax.text(0.5, 0.5, "Sense mostres quantificades",
                    ha='center', va='center', transform=ax.transAxes, color='#888')

        self._doc_figure.tight_layout()
        self._doc_canvas.draw()

    def _plot_dad_bars(self):
        """Bar chart DAD A254."""
        self._dad_figure.clear()
        ax = self._dad_figure.add_subplot(111)

        samples = self._quantification_result.get("samples_grouped", {})
        names, areas = [], []
        for name, sg in samples.items():
            sel_dad = (sg.get("selected") or {}).get("dad")
            rep = (sg.get("replicas") or {}).get(sel_dad, {}) if sel_dad else {}
            dad_areas = rep.get("areas", {}).get("DAD_254", {})
            total = dad_areas.get("total")
            if total is None:
                continue
            names.append(name[:20])
            areas.append(total)

        if names:
            xs = list(range(len(names)))
            ax.bar(xs, areas, color="#A23B72", alpha=0.85)
            ax.set_xticks(xs)
            ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
            ax.set_ylabel("A254 (mAU·min)", fontsize=9)
            ax.set_title("DAD à254 per mostra", fontsize=10)
            ax.grid(True, axis='y', alpha=0.3)
        else:
            ax.text(0.5, 0.5, "Sense dades DAD",
                    ha='center', va='center', transform=ax.transAxes, color='#888')

        self._dad_figure.tight_layout()
        self._dad_canvas.draw()

    def _populate_table(self):
        """Omple la taula de resultats."""
        samples = self._quantification_result.get("samples_grouped", {})
        headers = ["Mostra", "ppm DOC Direct", "ppm DOC UIB", "ppm DAD à254", "HCI", "Validesa"]
        self._table.setColumnCount(len(headers))
        self._table.setHorizontalHeaderLabels(headers)
        self._table.setRowCount(len(samples))

        for row, (name, sg) in enumerate(samples.items()):
            q = sg.get("quantification") or {}
            cells = [
                name,
                f"{q.get('concentration_ppm_direct') or 0:.3f}" if q.get('concentration_ppm_direct') is not None else "-",
                f"{q.get('concentration_ppm_uib') or 0:.3f}" if q.get('concentration_ppm_uib') is not None else "-",
                "-",  # TODO: ppm DAD if disponible
                f"{q.get('hci') or 0:.2f}" if q.get('hci') is not None else "-",
                "✓" if q.get('valid', True) else f"✗ {q.get('reason', '')}",
            ]
            for col, val in enumerate(cells):
                item = QTableWidgetItem(str(val))
                self._table.setItem(row, col, item)

        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
