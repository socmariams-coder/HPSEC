"""
HPSEC Suite — Quantify Panel (Pas 4)
=====================================

Aplica la recta de calibració global al resultat d'Analitzar i mostra:
- Header: info de la recta vigent + botó d'aplicar + status.
- Split horitzontal: taula (ppm + estadística per mostra) | detall de la
  mostra seleccionada (per rèplica + fraccions).
- Secció col·lapsable a sota: vista global agregada.

Quantifica TOTES les rèpliques vàlides de cada mostra; el ppm final és
el de la rèplica seleccionada (per backwards compat), però es mostren
les estadístiques (mean ± SD, RSD%) entre rèpliques.
"""

from __future__ import annotations

import json
import logging
import os

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QBrush, QColor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QFrame, QGroupBox, QHBoxLayout,
    QHeaderView, QLabel, QPushButton, QScrollArea, QSizePolicy, QSplitter,
    QTabWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

logger = logging.getLogger(__name__)

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# Colors per fraccions (consistents amb resta de Suite)
FRACTION_COLORS = {
    "BioP": "#E74C3C", "HS": "#F39C12", "BB": "#27AE60",
    "SB": "#3498DB", "LMW": "#95A5A6",
}
FRACTION_ORDER = ["BioP", "HS", "BB", "SB", "LMW"]
SUBZONE_ORDER = ["HS-1", "HS-2", "HS-3", "HS-4", "BB-1", "BB-2"]
SUBZONE_PARENT = {
    "HS-1": "HS", "HS-2": "HS", "HS-3": "HS", "HS-4": "HS",
    "BB-1": "BB", "BB-2": "BB",
}


# ============================================================================
# Panell
# ============================================================================


class QuantifyPanel(QWidget):
    """Panell del pas Quantificar."""

    quantification_completed = Signal(dict)

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self._analysis_result = None
        self._quantification_result = None
        self._worker = None
        self._selected_sample = None
        self._setup_ui()

    # ───────────────────────────────────────────────────────────── UI ─────

    def _setup_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.NoFrame)
        outer.addWidget(self._scroll)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # ─── Empty state ───
        self.empty_state = QLabel(
            "Cal completar Analitzar abans de quantificar.\n"
            "Quan acabis l'anàlisi, aquest pas aplicarà la recta de calibració."
        )
        self.empty_state.setStyleSheet(
            "color: #6c757d; padding: 40px; background: #f8f9fa; "
            "border: 1px solid #dee2e6; border-radius: 6px; font-size: 12px;")
        self.empty_state.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.empty_state)

        # ─── Header: recta + accions + status ───
        self._header_frame = QFrame()
        self._header_frame.setVisible(False)
        self._header_frame.setStyleSheet(
            "QFrame { background: #f8f9fa; border: 1px solid #dee2e6;"
            " border-radius: 4px; }")
        header_layout = QVBoxLayout(self._header_frame)
        header_layout.setContentsMargins(10, 8, 10, 8)
        header_layout.setSpacing(6)

        self._cal_info_label = QLabel()
        self._cal_info_label.setTextFormat(Qt.RichText)
        self._cal_info_label.setWordWrap(True)
        self._cal_info_label.setStyleSheet("font-size: 11px;")
        self._cal_info_label.linkActivated.connect(self._goto_global_calibration)
        header_layout.addWidget(self._cal_info_label)

        actions_row = QHBoxLayout()
        actions_row.setSpacing(6)
        self._apply_btn = QPushButton("▶ Aplicar quantificació")
        self._apply_btn.setStyleSheet(
            "QPushButton { background: #2E86AB; color: white; padding: 6px 14px;"
            " border: none; border-radius: 4px; font-size: 11px; font-weight: bold; }"
            "QPushButton:hover { background: #1f6080; }"
            "QPushButton:disabled { background: #adb5bd; }")
        self._apply_btn.clicked.connect(self._run_quantify)
        actions_row.addWidget(self._apply_btn)

        self._pdf_btn = QPushButton("📄 PDF")
        self._pdf_btn.setStyleSheet(
            "QPushButton { background: white; color: #2E86AB; padding: 6px 12px;"
            " border: 1px solid #2E86AB; border-radius: 4px; font-size: 11px; }"
            "QPushButton:hover { background: #EBF5FB; }"
            "QPushButton:disabled { color: #adb5bd; border-color: #ced4da; }")
        self._pdf_btn.setEnabled(False)
        self._pdf_btn.clicked.connect(self._generate_pdf)
        actions_row.addWidget(self._pdf_btn)

        actions_row.addStretch()
        self._status_label = QLabel()
        self._status_label.setTextFormat(Qt.RichText)
        self._status_label.setStyleSheet("font-size: 11px; color: #555;")
        actions_row.addWidget(self._status_label)
        header_layout.addLayout(actions_row)
        layout.addWidget(self._header_frame)

        # ─── Split principal: taula | detall ───
        self._main_splitter = QSplitter(Qt.Horizontal)
        self._main_splitter.setVisible(False)
        self._main_splitter.setChildrenCollapsible(False)
        self._main_splitter.setHandleWidth(4)

        # Taula esquerra
        self._table = QTableWidget()
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setAlternatingRowColors(True)
        self._table.setStyleSheet(
            "QTableWidget { font-size: 11px; }"
            "QHeaderView::section { background: #f1f3f5; padding: 4px;"
            " border: 1px solid #dee2e6; font-weight: bold; }"
        )
        self._table.setMinimumWidth(420)
        self._table.itemSelectionChanged.connect(self._on_table_selection)
        self._main_splitter.addWidget(self._table)

        # Detall dret amb tabs
        self._detail_frame = QFrame()
        self._detail_frame.setStyleSheet(
            "QFrame { border: 1px solid #dee2e6; border-radius: 4px;"
            " background: white; }")
        self._detail_frame.setMinimumWidth(380)
        detail_layout = QVBoxLayout(self._detail_frame)
        detail_layout.setContentsMargins(8, 8, 8, 8)
        detail_layout.setSpacing(6)

        self._detail_title = QLabel("Selecciona una mostra a l'esquerra")
        self._detail_title.setStyleSheet(
            "font-size: 12px; font-weight: bold; color: #2c3e50; padding-bottom: 4px;")
        detail_layout.addWidget(self._detail_title)

        self._detail_tabs = QTabWidget()
        self._detail_tabs.setDocumentMode(True)
        self._detail_tabs.setStyleSheet(
            "QTabBar::tab { padding: 4px 12px; font-size: 11px; }"
            "QTabBar::tab:selected { font-weight: bold; }")

        # Tab 1: Per rèplica (bar chart + estadística)
        tab_rep = QWidget()
        rep_layout = QVBoxLayout(tab_rep)
        rep_layout.setContentsMargins(0, 4, 0, 0)
        if HAS_MATPLOTLIB:
            self._rep_figure = Figure(figsize=(5, 3), dpi=100)
            self._rep_figure.set_facecolor("#FAFAFA")
            self._rep_canvas = FigureCanvas(self._rep_figure)
            self._rep_canvas.setMinimumHeight(240)
            rep_layout.addWidget(self._rep_canvas, 1)
        self._rep_stats = QLabel("—")
        self._rep_stats.setTextFormat(Qt.RichText)
        self._rep_stats.setStyleSheet(
            "font-size: 11px; padding: 6px; background: #f8f9fa;"
            " border: 1px solid #e9ecef; border-radius: 3px;")
        self._rep_stats.setWordWrap(True)
        rep_layout.addWidget(self._rep_stats)
        self._detail_tabs.addTab(tab_rep, "Per rèplica")

        # Tab 2: Fraccions (només COLUMN)
        tab_frac = QWidget()
        frac_layout = QVBoxLayout(tab_frac)
        frac_layout.setContentsMargins(0, 4, 0, 0)
        self._frac_table = QTableWidget()
        self._frac_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._frac_table.setStyleSheet(
            "QTableWidget { font-size: 11px; }"
            "QHeaderView::section { background: #f1f3f5; padding: 3px;"
            " border: 1px solid #dee2e6; font-weight: bold; }")
        frac_layout.addWidget(self._frac_table)
        self._detail_tabs.addTab(tab_frac, "Fraccions")
        self._frac_tab_idx = 1

        detail_layout.addWidget(self._detail_tabs, 1)
        self._main_splitter.addWidget(self._detail_frame)

        self._main_splitter.setSizes([520, 600])
        self._main_splitter.setStretchFactor(0, 45)
        self._main_splitter.setStretchFactor(1, 55)
        layout.addWidget(self._main_splitter, 1)

        # ─── Vista global col·lapsable ───
        self._global_collapsible = self._build_collapsible("Vista global de la seqüència")
        self._global_collapsible["frame"].setVisible(False)

        if HAS_MATPLOTLIB:
            # Bar chart agregat
            self._global_fig = Figure(figsize=(10, 3.5), dpi=100)
            self._global_fig.set_facecolor("#FAFAFA")
            self._global_canvas = FigureCanvas(self._global_fig)
            self._global_canvas.setMinimumHeight(260)
            self._global_collapsible["content_layout"].addWidget(self._global_canvas)
        layout.addWidget(self._global_collapsible["frame"])

        layout.addStretch()
        self._scroll.setWidget(content)

    def _build_collapsible(self, title):
        """Frame col·lapsable amb header clicable."""
        frame = QFrame()
        frame.setStyleSheet(
            "QFrame { border: 1px solid #E0E0E0; border-radius: 4px; }")
        fl = QVBoxLayout(frame)
        fl.setContentsMargins(0, 0, 0, 0)
        fl.setSpacing(0)

        btn = QPushButton(f"▶ {title}")
        btn.setStyleSheet(
            "QPushButton { border: none; text-align: left; padding: 6px 10px;"
            " font-size: 11px; font-weight: bold; color: #555;"
            " background: #F5F5F5; border-radius: 4px; }"
            "QPushButton:hover { background: #E8E8E8; }")
        fl.addWidget(btn)

        content = QWidget()
        content.setVisible(False)
        cl = QVBoxLayout(content)
        cl.setContentsMargins(6, 6, 6, 6)
        fl.addWidget(content)

        def _toggle():
            vis = not content.isVisible()
            content.setVisible(vis)
            btn.setText(f"{'▼' if vis else '▶'} {title}")

        btn.clicked.connect(_toggle)
        return {"frame": frame, "toggle_btn": btn,
                "content": content, "content_layout": cl}

    # ─────────────────────────────────────────────────────────── API ─────

    def load(self, analysis_result: dict):
        """Carrega el resultat d'Analitzar i prepara la UI."""
        self._analysis_result = analysis_result
        if not analysis_result or not analysis_result.get("samples_grouped"):
            self.empty_state.setVisible(True)
            self._header_frame.setVisible(False)
            self._main_splitter.setVisible(False)
            self._global_collapsible["frame"].setVisible(False)
            return

        self.empty_state.setVisible(False)
        self._header_frame.setVisible(True)
        self._main_splitter.setVisible(True)
        self._global_collapsible["frame"].setVisible(True)

        self._update_cal_info()

        # Si la quantificació ja s'havia fet, mostrar directament
        if not analysis_result.get("quantification_pending", True):
            self._quantification_result = analysis_result
            self._pdf_btn.setEnabled(True)
            self._render_results()
        else:
            # Encara no s'ha quantificat — mostrar taula amb àrees + ppm pendent
            self._render_pending()
            # Auto-executar si hi ha recta vigent: el pas no ha de dependre
            # d'un clic que, si s'oblida, deixa els ppm buits a l'export.
            # El botó queda com a re-execució manual.
            if self._apply_btn.isEnabled() and (
                    self._worker is None or not self._worker.isRunning()):
                self._run_quantify()

    # ────────────────────────────────────────────────── Header (recta) ────

    def _update_cal_info(self):
        if not self._analysis_result:
            return
        try:
            from hpsec_calibrate import get_rf_mass_cal, get_calibration_intercept
            method = (self._analysis_result.get("method") or "COLUMN").lower()
            mode_key = "bp" if method == "bp" else "column"
            rf_d = get_rf_mass_cal(signal="direct", mode=mode_key)
            in_d = get_calibration_intercept(signal="direct", mode=mode_key)
            rf_u = get_rf_mass_cal(signal="uib", mode=mode_key)
            in_u = get_calibration_intercept(signal="uib", mode=mode_key)
            parts = []
            if rf_d is not None:
                parts.append(
                    f"<b>Direct/{mode_key.upper()}:</b> RF={rf_d:.1f}"
                    f"&nbsp;·&nbsp;intercept={in_d:.2f}")
            if rf_u is not None:
                parts.append(
                    f"<b>UIB/{mode_key.upper()}:</b> RF={rf_u:.1f}"
                    f"&nbsp;·&nbsp;intercept={in_u:.2f}")
            if not parts:
                self._cal_info_label.setText(
                    "<span style='color:#c0392b'>"
                    "⚠ Cap recta de calibració activa per aquest mode.</span> "
                    "Per activar-ne una: processa una SEQ de calibració (nom amb "
                    "<b>_CAL</b>) o consulta l'estat a "
                    "<a href='goto_cal'>Calibració Global</a>.")
                self._apply_btn.setEnabled(False)
            else:
                self._cal_info_label.setText(
                    "Recta vigent:&nbsp;&nbsp;" + "&nbsp;&nbsp;|&nbsp;&nbsp;".join(parts))
                self._apply_btn.setEnabled(True)
        except Exception as e:
            logger.warning("Error info calibració: %s", e)
            self._cal_info_label.setText(
                f"<span style='color:#c0392b'>Error: {e}</span>")

    def _goto_global_calibration(self, _link=None):
        """Porta l'usuari al tab Calibració Global (escenari sense recta)."""
        try:
            self.main_window.tab_widget.setCurrentIndex(4)
        except Exception as e:
            logger.warning("No s'ha pogut navegar a Calibració Global: %s", e)

    # ─────────────────────────────────────────── Run quantify (worker) ────

    def _run_quantify(self):
        if not self._analysis_result:
            self._status_label.setText(
                "<span style='color:#c0392b'>✗ Cal completar Analitzar primer.</span>")
            return
        self._apply_btn.setEnabled(False)
        self._status_label.setText("Quantificant…")
        seq_path = getattr(self.main_window, "seq_path", None) \
            or self._analysis_result.get("seq_path")
        method = self._analysis_result.get("method", "COLUMN")
        mode = "BP" if method.upper() == "BP" else "COLUMN"
        from .worker import QuantifyWorker
        self._worker = QuantifyWorker(
            self._analysis_result, seq_path=seq_path, mode=mode)
        self._worker.completed.connect(self._on_quantify_completed)
        self._worker.error.connect(self._on_quantify_error)
        self._worker.progress.connect(self._on_quantify_progress)
        self._worker.start()

    @Slot(str, int)
    def _on_quantify_progress(self, msg: str, pct: int):
        self._status_label.setText(f"{msg} ({pct}%)")

    @Slot(dict)
    def _on_quantify_completed(self, result: dict):
        self._quantification_result = result
        self._apply_btn.setEnabled(True)
        self._pdf_btn.setEnabled(True)
        n = sum(1 for sg in result.get("samples_grouped", {}).values()
                if (sg.get("quantification") or {}).get("concentration_ppm_direct") is not None)
        self._status_label.setText(
            f"<span style='color:#28a745'>✓ Quantificat ({n} mostres)</span>")
        self._persist_result()
        self._render_results()
        self.quantification_completed.emit(result)

    @Slot(str)
    def _on_quantify_error(self, msg: str):
        self._apply_btn.setEnabled(True)
        self._status_label.setText(
            f"<span style='color:#c0392b'>✗ Error: {msg}</span>")

    def _persist_result(self):
        if not self._quantification_result:
            return
        seq_path = self._quantification_result.get("seq_path") \
            or getattr(self.main_window, "seq_path", "")
        if not seq_path:
            return
        json_path = os.path.join(seq_path, "CHECK", "data", "analysis_result.json")
        try:
            # Atòmic + NumpyEncoder (mateix encoder que save_analysis_result;
            # abans default=str convertia números en text — no impecable)
            from hpsec_utils import NumpyEncoder, _atomic_write_json
            _atomic_write_json(json_path, self._quantification_result,
                               indent=2, ensure_ascii=False, cls=NumpyEncoder)
        except Exception as e:
            logger.error("Error persistint quantificació: %s", e)

    def _generate_pdf(self):
        if not self._quantification_result:
            return
        seq_path = getattr(self.main_window, "seq_path", None) \
            or self._quantification_result.get("seq_path")
        if not seq_path:
            self._status_label.setText(
                "<span style='color:#c0392b'>✗ Sense seq_path</span>")
            return
        self._pdf_btn.setEnabled(False)
        self._status_label.setText("Generant PDF…")
        try:
            from hpsec_reports import generate_analysis_report
            pdf_path = generate_analysis_report(
                seq_path, analysis_data=self._quantification_result)
            if pdf_path:
                self._status_label.setText(
                    f"<span style='color:#28a745'>✓ PDF: {os.path.basename(pdf_path)}</span>")
                try:
                    os.startfile(pdf_path)
                except Exception:
                    pass
            else:
                self._status_label.setText(
                    "<span style='color:#c0392b'>✗ Error generant PDF</span>")
        except Exception as e:
            logger.exception("Error PDF")
            self._status_label.setText(
                f"<span style='color:#c0392b'>✗ {e}</span>")
        finally:
            self._pdf_btn.setEnabled(True)

    # ──────────────────────────────────────────── Render: taula + detall ──

    def _render_pending(self):
        """Estat pre-quantificació: mostra taula amb àrees, ppm en blanc."""
        self._populate_table(pending=True)

    def _render_results(self):
        self._populate_table(pending=False)
        # Si hi havia selecció prèvia, mantenir-la; sinó la primera
        if self._table.rowCount() > 0:
            if not self._table.currentItem():
                self._table.selectRow(0)
        self._render_global_chart()

    def _is_bp(self):
        method = (self._analysis_result or {}).get("method", "COLUMN") if self._analysis_result else "COLUMN"
        return method.upper() == "BP"

    def _populate_table(self, pending=False):
        samples = (self._analysis_result or {}).get("samples_grouped", {})
        # Filtrar mostres no-KHP
        rows = [(name, sg) for name, sg in samples.items()
                if sg.get("sample_type") != "KHP"]

        headers = ["Mostra", "n", "ppm̄ Direct ± SD", "RSD%", "ppm̄ UIB ± SD", "Δ% D↔U", "Estat"]
        self._table.setColumnCount(len(headers))
        self._table.setHorizontalHeaderLabels(headers)
        self._table.setRowCount(len(rows))

        for r, (name, sg) in enumerate(rows):
            q = sg.get("quantification") or {}
            stats = q.get("statistics") or {}
            sd = stats.get("direct") or {}
            su = stats.get("uib") or {}

            # Estat
            sample_type = sg.get("sample_type", "SAMPLE")
            valid = q.get("valid", True)
            reason = q.get("reason", "")

            if pending:
                ppm_d_txt = "—"
                ppm_u_txt = "—"
                rsd_txt = "—"
                delta_txt = "—"
                n_txt = "—"
                estat = "pendent"
                estat_color = "#888"
            elif sample_type == "BLANK":
                ppm_d_txt = "—"
                ppm_u_txt = "—"
                rsd_txt = "—"
                delta_txt = "—"
                n_txt = "—"
                estat = "BLANC"
                estat_color = "#888"
            elif not valid:
                ppm_d_txt = "—"
                ppm_u_txt = "—"
                rsd_txt = "—"
                delta_txt = "—"
                n_txt = "—"
                estat = "✗ " + (reason or "no vàlida")
                estat_color = "#c0392b"
            else:
                n = sd.get("n", 0)
                mean_d = sd.get("mean")
                sd_d = sd.get("sd")
                rsd_d = sd.get("rsd_pct")
                mean_u = su.get("mean")
                sd_u = su.get("sd")
                n_txt = str(n) if n else "—"
                ppm_d_txt = (f"{mean_d:.3f} ± {sd_d:.3f}" if mean_d is not None and sd_d is not None
                             else (f"{mean_d:.3f}" if mean_d is not None else "—"))
                ppm_u_txt = (f"{mean_u:.3f} ± {sd_u:.3f}" if mean_u is not None and sd_u is not None
                             else (f"{mean_u:.3f}" if mean_u is not None else "—"))
                rsd_txt = f"{rsd_d:.2f}" if rsd_d is not None else "—"
                if mean_d and mean_u:
                    delta = 100 * (mean_u - mean_d) / mean_d
                    delta_txt = f"{delta:+.1f}"
                else:
                    delta_txt = "—"
                estat = "✓"
                estat_color = "#28a745"

            cells = [name, n_txt, ppm_d_txt, rsd_txt, ppm_u_txt, delta_txt, estat]
            for c, val in enumerate(cells):
                item = QTableWidgetItem(str(val))
                if c == 6:
                    item.setForeground(QBrush(QColor(estat_color)))
                # Marca RSD alt
                if c == 3 and rsd_txt not in ("—",):
                    try:
                        rsd_v = float(rsd_txt)
                        if rsd_v > 10:
                            item.setForeground(QBrush(QColor("#c0392b")))
                        elif rsd_v > 5:
                            item.setForeground(QBrush(QColor("#d4a017")))
                    except ValueError:
                        pass
                if c >= 1:
                    item.setTextAlignment(Qt.AlignCenter)
                self._table.setItem(r, c, item)

        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        for i in range(1, len(headers)):
            header.setSectionResizeMode(i, QHeaderView.ResizeToContents)

        # Habilitar/deshabilitar tab Fraccions segons mètode
        if self._is_bp():
            self._detail_tabs.setTabEnabled(self._frac_tab_idx, False)
        else:
            self._detail_tabs.setTabEnabled(self._frac_tab_idx, True)

    def _on_table_selection(self):
        items = self._table.selectedItems()
        if not items:
            return
        row = items[0].row()
        name_item = self._table.item(row, 0)
        if not name_item:
            return
        name = name_item.text()
        self._selected_sample = name
        self._render_detail(name)

    def _render_detail(self, sample_name: str):
        sg = (self._analysis_result or {}).get("samples_grouped", {}).get(sample_name)
        if not sg:
            return
        self._detail_title.setText(sample_name)
        q = sg.get("quantification") or {}
        per_rep = q.get("per_replica") or {}
        stats = q.get("statistics") or {}

        # ─── Tab 1: per rèplica ───
        if HAS_MATPLOTLIB and per_rep:
            self._draw_replica_bars(per_rep, sample_data=sg,
                                     selected=q.get("selected_replica"))
        elif HAS_MATPLOTLIB:
            self._rep_figure.clear()
            ax = self._rep_figure.add_subplot(111)
            ax.text(0.5, 0.5, "Sense rèpliques quantificades",
                    ha="center", va="center", color="#888",
                    transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            self._rep_canvas.draw_idle()

        # Estadística
        if stats:
            sd = stats.get("direct") or {}
            su = stats.get("uib") or {}
            parts = []
            if sd.get("mean") is not None:
                rsd = sd.get("rsd_pct")
                rsd_txt = f"&nbsp;&nbsp;<i>RSD={rsd:.2f}%</i>" if rsd is not None else ""
                parts.append(
                    f"<b>Direct</b>: {sd['mean']:.3f} ± {sd.get('sd', 0):.3f} "
                    f"(n={sd.get('n', 0)}){rsd_txt}")
            if su.get("mean") is not None:
                rsd = su.get("rsd_pct")
                rsd_txt = f"&nbsp;&nbsp;<i>RSD={rsd:.2f}%</i>" if rsd is not None else ""
                parts.append(
                    f"<b>UIB</b>: {su['mean']:.3f} ± {su.get('sd', 0):.3f} "
                    f"(n={su.get('n', 0)}){rsd_txt}")
            if sd.get("mean") and su.get("mean"):
                delta = 100 * (su["mean"] - sd["mean"]) / sd["mean"]
                parts.append(f"<b>Δ Direct↔UIB</b>: {delta:+.2f}%")
            sel_rep = q.get("selected_replica")
            if sel_rep:
                parts.append(f"<i>Rèplica final: R{sel_rep}</i>")
            self._rep_stats.setText("<br>".join(parts) if parts else "—")
        else:
            self._rep_stats.setText("Sense estadística disponible")

        # ─── Tab 2: fraccions ───
        if not self._is_bp():
            self._populate_fractions_table(sg)

    def _draw_replica_bars(self, per_rep: dict, sample_data: dict = None,
                            selected: str = None):
        """Bar stacked: alçada = ppm Direct total, segments per fracció
        (proporcionals a les àrees integrades). Marker taronja = ppm UIB.

        Per BP (sense fraccions), una sola barra plena en color BB-blau.
        """
        self._rep_figure.clear()
        ax = self._rep_figure.add_subplot(111)

        keys = sorted(per_rep.keys(),
                      key=lambda x: (int(x) if str(x).isdigit() else 999))
        if not keys:
            ax.text(0.5, 0.5, "Sense rèpliques quantificades",
                    ha="center", va="center", color="#888", transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            self._rep_canvas.draw_idle()
            return

        # Llegir àrees de cada rèplica per derivar ppm per fracció
        replicas_full = (sample_data or {}).get("replicas") or {}
        is_bp = self._is_bp()

        labels = []
        ppm_direct_total = []
        ppm_uib_total = []
        # Per cada fracció, llista de ppm per cada rèplica (mateix ordre que keys)
        frac_ppm = {fn: [] for fn in FRACTION_ORDER}

        for k in keys:
            v = per_rep[k] or {}
            sib = v.get("source_label", "")
            labels.append(f"R{k}[{sib}]" if sib else f"R{k}")
            p_d = v.get("ppm_direct") or 0
            p_u = v.get("ppm_uib") or 0
            ppm_direct_total.append(p_d)
            ppm_uib_total.append(p_u)

            # Àrees de la rèplica per derivar ppm per fracció
            rd = replicas_full.get(k) or {}
            areas_doc = (rd.get("areas") or {}).get("DOC") or {}
            total_area = areas_doc.get("total") or 0
            for fn in FRACTION_ORDER:
                if is_bp:
                    # BP: cap fracció — tot el ppm va a una sola "fracció" pic
                    frac_ppm[fn].append(p_d if fn == "BioP" else 0)
                else:
                    a_frac = areas_doc.get(fn) or 0
                    if total_area > 0 and p_d > 0:
                        frac_ppm[fn].append(p_d * a_frac / total_area)
                    else:
                        frac_ppm[fn].append(0)

        x = list(range(len(labels)))
        w = 0.55

        # Stacked bars per fracció — colors del FRACTION_COLORS
        bottom = [0] * len(x)
        bars_per_frac = {}
        if is_bp:
            # BP: una sola barra plena, no stacked
            bars_per_frac["BP"] = ax.bar(x, ppm_direct_total, w,
                                          color="#2E86AB", alpha=0.85,
                                          label="DOC (BP)")
        else:
            for fn in FRACTION_ORDER:
                vals = frac_ppm[fn]
                if not any(vals):
                    continue
                bars = ax.bar(x, vals, w, bottom=bottom,
                              color=FRACTION_COLORS.get(fn, "#888"),
                              alpha=0.88, label=fn,
                              edgecolor="white", linewidth=0.5)
                bars_per_frac[fn] = bars
                bottom = [b + v for b, v in zip(bottom, vals)]

        # Etiqueta ppm Direct total a sobre de cada barra
        for i, ptot in enumerate(ppm_direct_total):
            if ptot:
                ax.text(i, ptot * 1.02, f"{ptot:.2f}",
                        ha="center", va="bottom", fontsize=9,
                        fontweight="bold", color="#2c3e50")

        # Marker per ppm UIB (línia horitzontal taronja al damunt de cada barra)
        for i, pu in enumerate(ppm_uib_total):
            if pu:
                ax.plot([i - w/2 * 0.85, i + w/2 * 0.85], [pu, pu],
                        color="#E67E22", lw=2.4, solid_capstyle="round",
                        label="UIB" if i == 0 else None)

        # Marca rèplica seleccionada amb un fons subtil
        if selected and str(selected) in keys:
            sel_idx = keys.index(str(selected))
            ax.axvspan(sel_idx - w/2 - 0.05, sel_idx + w/2 + 0.05,
                       alpha=0.06, color="#2E86AB", zorder=0)
            # I etiqueta sota la barra
            ax.text(sel_idx, -max(ppm_direct_total) * 0.05 if ppm_direct_total else 0,
                    "★ sel", ha="center", va="top",
                    fontsize=8, color="#2E86AB", style="italic")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("ppm DOC", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Llegenda compacta: fraccions principals + UIB marker
        handles, leg_labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, leg_labels, fontsize=7, loc="upper right",
                      framealpha=0.85, ncol=min(len(handles), 6))

        try:
            self._rep_figure.tight_layout()
        except Exception:
            pass
        self._rep_canvas.draw_idle()

    def _populate_fractions_table(self, sg: dict):
        """Taula de fraccions: àrea + % per rèplica + mitjana."""
        per_rep = (sg.get("quantification") or {}).get("per_replica") or {}
        replicas_full = sg.get("replicas") or {}
        # Llista de noms d'àrea a mostrar (principals + subzones)
        # Determinades pels keys disponibles a la primera rèplica vàlida
        first_areas = None
        for k in per_rep.keys():
            rd = replicas_full.get(k)
            if isinstance(rd, dict):
                first_areas = (rd.get("areas") or {}).get("DOC") or {}
                if first_areas:
                    break
        if not first_areas:
            self._frac_table.setRowCount(0)
            self._frac_table.setColumnCount(0)
            return

        # Ordenar: principals primer, després subzones
        all_keys = [k for k in first_areas.keys()
                    if k != "total" and isinstance(first_areas[k], (int, float))]
        principal = [k for k in FRACTION_ORDER if k in all_keys]
        subzones = [k for k in SUBZONE_ORDER if k in all_keys]
        others = [k for k in all_keys if k not in principal and k not in subzones]
        frac_keys = principal + subzones + sorted(others)

        rep_keys = sorted(per_rep.keys(), key=lambda x: (int(x) if x.isdigit() else 999))
        headers = ["Fracció"] + [f"R{k}" for k in rep_keys] + ["Mitjana", "% del total"]
        self._frac_table.setColumnCount(len(headers))
        self._frac_table.setHorizontalHeaderLabels(headers)
        self._frac_table.setRowCount(len(frac_keys))

        for r, fk in enumerate(frac_keys):
            # Indentar subzones per claredat visual
            display_name = f"  └─ {fk}" if fk in SUBZONE_ORDER else fk
            name_item = QTableWidgetItem(display_name)
            if fk in FRACTION_ORDER:
                color = FRACTION_COLORS.get(fk, "#555")
                f = QFont(); f.setBold(True)
                name_item.setFont(f)
                name_item.setForeground(QBrush(QColor(color)))
            elif fk in SUBZONE_ORDER:
                parent = SUBZONE_PARENT.get(fk)
                color = FRACTION_COLORS.get(parent, "#888")
                name_item.setForeground(QBrush(QColor(color)))
            self._frac_table.setItem(r, 0, name_item)

            vals = []
            totals = []
            for c, rk in enumerate(rep_keys, start=1):
                rd = replicas_full.get(rk) or {}
                areas = (rd.get("areas") or {}).get("DOC") or {}
                v = areas.get(fk, 0)
                tot = areas.get("total", 0) or 0
                if v:
                    vals.append(v)
                    totals.append(tot)
                txt = f"{v:.1f}" if v else "—"
                item = QTableWidgetItem(txt)
                item.setTextAlignment(Qt.AlignCenter)
                self._frac_table.setItem(r, c, item)

            mean_v = sum(vals) / len(vals) if vals else None
            mean_total = sum(totals) / len(totals) if totals else 0
            pct = (100 * mean_v / mean_total) if (mean_v and mean_total > 0) else None
            mean_txt = f"{mean_v:.1f}" if mean_v else "—"
            pct_txt = f"{pct:.1f}%" if pct is not None else "—"
            m_item = QTableWidgetItem(mean_txt)
            m_item.setTextAlignment(Qt.AlignCenter)
            p_item = QTableWidgetItem(pct_txt)
            p_item.setTextAlignment(Qt.AlignCenter)
            self._frac_table.setItem(r, len(headers) - 2, m_item)
            self._frac_table.setItem(r, len(headers) - 1, p_item)

        header = self._frac_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        for i in range(1, len(headers)):
            header.setSectionResizeMode(i, QHeaderView.ResizeToContents)

    # ─────────────────────────────────────────── Global aggregate chart ───

    def _render_global_chart(self):
        if not HAS_MATPLOTLIB or not self._quantification_result:
            return
        self._global_fig.clear()
        ax = self._global_fig.add_subplot(111)
        samples = self._quantification_result.get("samples_grouped", {})
        names, means, sds = [], [], []
        for name, sg in samples.items():
            if sg.get("sample_type") == "KHP":
                continue
            q = sg.get("quantification") or {}
            stats = (q.get("statistics") or {}).get("direct") or {}
            m = stats.get("mean")
            if m is None:
                continue
            names.append(name[:18])
            means.append(m)
            sds.append(stats.get("sd") or 0)
        if not names:
            ax.text(0.5, 0.5, "Sense mostres quantificades",
                    ha="center", va="center", color="#888", transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
        else:
            x = list(range(len(names)))
            ax.bar(x, means, yerr=sds, capsize=3,
                   color="#2E86AB", alpha=0.85, ecolor="#1f6080")
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("ppm DOC Direct (mitjana ± SD)", fontsize=9)
            ax.grid(True, axis="y", alpha=0.3)
            ax.tick_params(labelsize=8)
        try:
            self._global_fig.tight_layout()
        except Exception:
            pass
        self._global_canvas.draw_idle()
