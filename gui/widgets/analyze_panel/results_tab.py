"""
HPSEC Suite - Results Tab (extracted from AnalyzePanel)
========================================================

Contains all table + chart functionality for the Analyze panel:
- Unified results table (15 columns: DOC + DAD)
- Category selector bar (Mostres / Blancs / Control)
- DOC and DAD charts (stacked bars, overlays, timeout timeline)
- Sample detail dialog integration
- Repair dialog integration

This module is designed to be embedded inside AnalyzePanel as a sub-tab.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QComboBox,
    QFrame, QAbstractItemView, QMessageBox, QDialog,
    QCheckBox, QScrollArea
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QBrush, QFont

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
    from matplotlib.figure import Figure
    import matplotlib.cm as cm
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# --- Chart styling constants ---
_CHART_FONT = "Segoe UI"
_CHART_BG = "#FAFAFA"
_CHART_TITLE_SIZE = 9
_CHART_LABEL_SIZE = 8
_CHART_TICK_SIZE = 7

from gui.widgets.styles import (
    COLOR_SUCCESS, COLOR_WARNING, COLOR_ERROR,
)
from .dialogs import SampleDetailDialog
from .repair_dialog import JaggedPeakRepairDialog
from ._constants import (
    CRITICAL_ANOMALIES, WARNING_ANOMALIES,
    DAD_WL_MAIN, SIGNAL_KEYS_MAIN,
)
from hpsec_warnings import (
    has_anomaly, get_anomaly_codes, classify_anomalies,
    ANOMALY_CATALOG,
)
from ._helpers import (
    configure_table_style, populate_signal_summary, populate_fractions_table
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


if HAS_MATPLOTLIB:
    class _ClickableCanvas(FigureCanvas):
        """FigureCanvas amb suport de doble-clic (Qt level, no matplotlib)."""

        def __init__(self, figure, on_dblclick=None):
            super().__init__(figure)
            self._on_dblclick = on_dblclick
            self.setCursor(Qt.PointingHandCursor)

        def mouseDoubleClickEvent(self, event):
            if self._on_dblclick:
                self._on_dblclick()
            super().mouseDoubleClickEvent(event)

    class OverlayPopupDialog(QDialog):
        """Pop-up interactiu per cromatogrames superposats (DOC o DAD)."""

        def __init__(self, parent, title, plot_fn):
            super().__init__(parent)
            self.setWindowTitle(title)
            self.resize(1000, 600)
            self.setMinimumSize(700, 400)

            layout = QVBoxLayout(self)
            layout.setContentsMargins(4, 4, 4, 4)
            layout.setSpacing(2)

            self._figure = Figure(figsize=(10, 5.5), dpi=100)
            self._figure.set_facecolor(_CHART_BG)
            self._canvas = FigureCanvas(self._figure)
            self._toolbar = NavigationToolbar2QT(self._canvas, self)
            self._toolbar.setStyleSheet(
                "QToolBar { border: none; spacing: 4px; background: #f8f8f8; }"
            )

            layout.addWidget(self._toolbar)
            layout.addWidget(self._canvas, 1)

            # Barra d'instruccions
            hint = QLabel(
                "<span style='color:#777; font-size:10px'>"
                "\u2139\ufe0f Clic a la llegenda per mostrar/amagar traces"
                " &nbsp;|&nbsp; "
                "Usa la barra d'eines per zoom, pan i guardar PNG</span>"
            )
            hint.setAlignment(Qt.AlignCenter)
            hint.setStyleSheet(
                "QLabel { background: #f8f8f8; padding: 4px;"
                " border-top: 1px solid #eee; }")
            layout.addWidget(hint)

            # Dibuixar
            ax = self._figure.add_subplot(111)
            plot_fn(ax)
            self._style_ax(ax)
            self._setup_interactive_legend(ax)
            self._figure.tight_layout()
            self._canvas.draw()

        @staticmethod
        def _style_ax(ax):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(axis='both', labelsize=_CHART_TICK_SIZE)
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
                item.set_fontfamily(_CHART_FONT)

        def _setup_interactive_legend(self, ax):
            """Llegenda clicable: toggle visibilitat de cada trac\u0327a."""
            handles, labels = ax.get_legend_handles_labels()
            if not handles:
                return
            ncols = max(1, len(handles) // 15 + 1)
            legend = ax.legend(
                handles, labels, loc='upper left',
                bbox_to_anchor=(1.01, 1),
                fontsize=7, framealpha=0.95, ncol=ncols,
                handlelength=1.5, borderaxespad=0,
                prop={'family': _CHART_FONT},
            )

            # Map legend lines -> plot lines
            self._legend_map = {}
            for leg_line, orig_handle in zip(legend.get_lines(), handles):
                leg_line.set_picker(5)
                self._legend_map[leg_line] = orig_handle

            self._figure.canvas.mpl_connect(
                'pick_event', self._on_legend_pick)

        def _on_legend_pick(self, event):
            leg_line = event.artist
            orig = self._legend_map.get(leg_line)
            if orig is None:
                return
            vis = not orig.get_visible()
            orig.set_visible(vis)
            leg_line.set_alpha(1.0 if vis else 0.25)
            self._canvas.draw_idle()


class ResultsTab(QWidget):
    """Tab de resultats: taula unificada DOC+DAD + grafics resum.

    Encapsula tota la funcionalitat de taula i grafics que anteriorment
    residia directament a AnalyzePanel.
    """

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window

        # State
        self.samples_grouped = {}
        self._sample_row_map = {}       # sample_name -> row index
        self._selected_sample = None
        self._detail_dialog = None

        # Chart data
        self._chart_regular = {}
        self._chart_blank = {}
        self._chart_control = {}
        self._chart_khp = {}
        self._chart_is_bp = False
        self._charts_initialized = False
        self._charts_visible = True

        # Category buttons & checkboxes
        self._cat_buttons = {}
        self._cat_counts = {}
        self._sample_checkboxes = []

        self._setup_ui()

    # ------------------------------------------------------------------
    # UI Setup
    # ------------------------------------------------------------------

    def _setup_ui(self):
        """Construeix la UI: selector bar + taula + grafics."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # === F0: SELECTOR BAR (categories + DAD combo) ===
        sel_frame = QFrame()
        sel_frame.setStyleSheet(
            "QFrame { background: #fff; border: 1px solid #e0e0e0;"
            " border-radius: 6px; }"
        )
        sel_layout = QHBoxLayout(sel_frame)
        sel_layout.setContentsMargins(10, 6, 10, 6)
        sel_layout.setSpacing(6)

        for cat_key, label, color, checked in [
            ("sample", "Mostres", "#2E86AB", True),
            ("blank", "Blancs", "#95a5a6", False),
            ("control", "Control", "#888", False),
        ]:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setChecked(checked)
            btn.clicked.connect(self._on_cat_toggle)
            self._cat_buttons[cat_key] = btn
            sel_layout.addWidget(btn)

        self._update_cat_btn_styles()

        sel_layout.addWidget(QLabel(
            "<span style='color:#ccc'>|</span>"
        ))

        sel_layout.addWidget(QLabel(
            "<b style='font-size:11px;color:#555'>DAD:</b>"
        ))
        self._wl_combo = QComboBox()
        self._wl_combo.setStyleSheet(
            "QComboBox { font-size: 11px; padding: 2px 6px;"
            " border: 1px solid #ccc; border-radius: 3px; }"
        )
        for wl in ["254", "220", "252", "272", "290", "362"]:
            self._wl_combo.addItem(f"A{wl}", wl)
        self._wl_combo.currentIndexChanged.connect(self._on_wl_changed)
        sel_layout.addWidget(self._wl_combo)

        sel_layout.addStretch()
        layout.addWidget(sel_frame)

        # === UNIFIED TABLE ===
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(15)
        self.results_table.setHorizontalHeaderLabels([
            "Mostra", "Inj", "Sel DOC", "Sel DAD", "A_DOC", "ppm",
            "A_UIB", "ppm_U", "SNR", "A_254", "SNR_254",
            "R\u00b2_DOC", "R\u00b2_DAD", "HCI", "Estat"
        ])
        self.results_table.setMinimumHeight(180)
        configure_table_style(self.results_table)
        self._configure_unified_columns()
        layout.addWidget(self.results_table)

        # Connect table signals
        self.results_table.doubleClicked.connect(self._on_table_double_click)
        self.results_table.cellClicked.connect(self._on_table_cell_click)
        self.results_table.setToolTip("Doble-clic per detall complet")

        # === CHARTS SECTION ===
        self.charts_section = QFrame()
        self.charts_section.setVisible(False)
        charts_outer = QVBoxLayout(self.charts_section)
        charts_outer.setContentsMargins(0, 8, 0, 0)
        charts_outer.setSpacing(4)

        # Charts content
        self._charts_content = QWidget()
        self._charts_content.setVisible(True)
        self._charts_content_layout = QVBoxLayout(self._charts_content)
        self._charts_content_layout.setContentsMargins(0, 4, 0, 0)
        self._charts_content_layout.setSpacing(4)

        if HAS_MATPLOTLIB:
            # F1+F2: DOC barres + DOC overlay miniatura (costat)
            doc_row = QHBoxLayout()
            doc_row.setSpacing(4)

            self.doc_figure = Figure(figsize=(5, 2.8), dpi=100)
            self.doc_figure.set_facecolor(_CHART_BG)
            self.doc_canvas = FigureCanvas(self.doc_figure)
            self.doc_canvas.setMinimumHeight(170)
            doc_row.addWidget(self.doc_canvas, 3)

            # DOC overlay miniatura + boto ampliar
            doc_overlay_frame = QFrame()
            doc_overlay_lay = QVBoxLayout(doc_overlay_frame)
            doc_overlay_lay.setContentsMargins(0, 0, 0, 0)
            doc_overlay_lay.setSpacing(0)

            self.doc_overlay_figure = Figure(figsize=(3, 2.8), dpi=100)
            self.doc_overlay_figure.set_facecolor(_CHART_BG)
            self.doc_overlay_canvas = _ClickableCanvas(
                self.doc_overlay_figure,
                on_dblclick=lambda: self._open_overlay_popup("doc"))
            self.doc_overlay_canvas.setMinimumHeight(160)
            doc_overlay_lay.addWidget(self.doc_overlay_canvas, 1)

            doc_zoom_btn = QPushButton("\U0001f50d  Ampliar DOC overlay")
            doc_zoom_btn.setCursor(Qt.PointingHandCursor)
            doc_zoom_btn.setStyleSheet(
                "QPushButton { background: #e3ecf5; color: #446; border: none;"
                " font-size: 10px; font-weight: bold; padding: 5px; }"
                "QPushButton:hover { background: #c9daf0; color: #224; }"
            )
            doc_zoom_btn.clicked.connect(lambda _=None: self._open_overlay_popup("doc"))
            doc_overlay_lay.addWidget(doc_zoom_btn)

            doc_row.addWidget(doc_overlay_frame, 2)

            self._charts_content_layout.addLayout(doc_row)

            # F3+F4: DAD barres + DAD overlay miniatura (costat)
            dad_row = QHBoxLayout()
            dad_row.setSpacing(4)

            self.dad_figure = Figure(figsize=(5, 2.8), dpi=100)
            self.dad_figure.set_facecolor(_CHART_BG)
            self.dad_canvas = FigureCanvas(self.dad_figure)
            self.dad_canvas.setMinimumHeight(170)
            dad_row.addWidget(self.dad_canvas, 3)

            # DAD overlay miniatura + boto ampliar
            dad_overlay_frame = QFrame()
            dad_overlay_lay = QVBoxLayout(dad_overlay_frame)
            dad_overlay_lay.setContentsMargins(0, 0, 0, 0)
            dad_overlay_lay.setSpacing(0)

            self.dad_overlay_figure = Figure(figsize=(3, 2.8), dpi=100)
            self.dad_overlay_figure.set_facecolor(_CHART_BG)
            self.dad_overlay_canvas = _ClickableCanvas(
                self.dad_overlay_figure,
                on_dblclick=lambda: self._open_overlay_popup("dad"))
            self.dad_overlay_canvas.setMinimumHeight(160)
            dad_overlay_lay.addWidget(self.dad_overlay_canvas, 1)

            dad_zoom_btn = QPushButton("\U0001f50d  Ampliar DAD overlay")
            dad_zoom_btn.setCursor(Qt.PointingHandCursor)
            dad_zoom_btn.setStyleSheet(
                "QPushButton { background: #e3ecf5; color: #446; border: none;"
                " font-size: 10px; font-weight: bold; padding: 5px; }"
                "QPushButton:hover { background: #c9daf0; color: #224; }"
            )
            dad_zoom_btn.clicked.connect(lambda _=None: self._open_overlay_popup("dad"))
            dad_overlay_lay.addWidget(dad_zoom_btn)

            dad_row.addWidget(dad_overlay_frame, 2)

            self._charts_content_layout.addLayout(dad_row)

            # F5: Timeout timeline (full width, compact)
            self.timeout_figure = Figure(figsize=(10, 1.5), dpi=100)
            self.timeout_figure.set_facecolor(_CHART_BG)
            self.timeout_canvas = FigureCanvas(self.timeout_figure)
            self.timeout_canvas.setMinimumHeight(100)
            self.timeout_canvas.setMaximumHeight(130)
            self._charts_content_layout.addWidget(self.timeout_canvas)

        charts_outer.addWidget(self._charts_content)
        layout.addWidget(self.charts_section)

    def _configure_unified_columns(self):
        """Configura columnes de la taula unificada."""
        header = self.results_table.horizontalHeader()
        for i in range(self.results_table.columnCount()):
            if i == 14:  # Estat -- stretch
                header.setSectionResizeMode(i, QHeaderView.Stretch)
            else:
                header.setSectionResizeMode(i, QHeaderView.ResizeToContents)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def populate(self, result):
        """Omple taula i grafics amb el resultat de l'analisi.

        Args:
            result: dict amb 'samples_grouped', 'method', etc.
        """
        self.samples_grouped = result.get("samples_grouped", {})
        self._populate_table()
        self._populate_charts(result)

    def reset(self):
        """Reinicia tot l'estat del tab."""
        self.samples_grouped = {}
        self._sample_row_map = {}
        self._selected_sample = None
        self._detail_dialog = None

        self._chart_regular = {}
        self._chart_blank = {}
        self._chart_control = {}
        self._chart_khp = {}
        self._chart_is_bp = False
        self._charts_initialized = False

        self.results_table.setRowCount(0)
        self.charts_section.setVisible(False)
        self._charts_content.setVisible(True)

    # ------------------------------------------------------------------
    # Populate unified table
    # ------------------------------------------------------------------

    def _populate_table(self):
        """Omple la taula unificada amb els resultats (15 cols, selectors DOC/DAD independents)."""
        self.results_table.setRowCount(0)
        self._sample_row_map = {}
        n_ok, n_warning, n_error, n_blank, n_control = 0, 0, 0, 0, 0

        # F0 toggle state
        show_blank = (self._cat_buttons.get("blank")
                      and self._cat_buttons["blank"].isChecked())
        show_control = (self._cat_buttons.get("control")
                        and self._cat_buttons["control"].isChecked())

        # Separar mostres per tipologia (KHP exclos -- ja analitzat a Verificar)
        sample_names = []   # SAMPLE + PR (mostres reals + patrons)
        blank_names = []    # BLANK / MQ
        control_names = []  # CONTROL / Neteja (analisi lleugera)

        for name, sd in self.samples_grouped.items():
            st = sd.get("sample_type", "SAMPLE")
            if st == "KHP":
                continue
            elif st == "BLANK":
                blank_names.append(name)
            elif st == "CONTROL":
                control_names.append(name)
            else:
                sample_names.append(name)

        # Ordenar per index d'injeccio (ordre cronologic al MasterFile)
        def _min_inj_index(name):
            reps = self.samples_grouped[name].get("replicas", {})
            indices = [r.get("injection_index", 999) for r in reps.values()
                       if r.get("injection_index") is not None]
            return min(indices) if indices else 999

        for lst in (sample_names, blank_names, control_names):
            lst.sort(key=_min_inj_index)

        # --- Regular samples + Blancs (full rendering) ---
        full_render_list = [(name, False) for name in sample_names]
        if blank_names and show_blank:
            full_render_list.append((None, True))  # Separator marker
            full_render_list.extend([(name, True) for name in blank_names])

        for sample_name, is_blank in full_render_list:
            if sample_name is None:
                # Insert BLANCS separator
                n_cols = self.results_table.columnCount()
                sep_row = self.results_table.rowCount()
                self.results_table.insertRow(sep_row)
                sep_item = QTableWidgetItem("--- BLANCS / MQ ---")
                sep_item.setFlags(Qt.ItemIsEnabled)
                sep_font = QFont()
                sep_font.setBold(True)
                sep_item.setFont(sep_font)
                sep_item.setForeground(QBrush(QColor("#7f8c8d")))
                self.results_table.setItem(sep_row, 0, sep_item)
                self.results_table.setSpan(sep_row, 0, 1, n_cols)
                sep_bg = QBrush(QColor("#EAECEE"))
                for c in range(n_cols):
                    item = self.results_table.item(sep_row, c)
                    if item is None:
                        item = QTableWidgetItem("")
                        self.results_table.setItem(sep_row, c, item)
                    item.setBackground(sep_bg)
                continue

            sample_data = self.samples_grouped[sample_name]

            # BLANK: una fila per injeccio (no agrupat per replica)
            if is_blank:
                replicas = sample_data.get("replicas") or {}
                quantification = sample_data.get("quantification") or {}
                comparison = sample_data.get("comparison") or {}
                for rep_key in sorted(replicas.keys()):
                    rep_data = replicas[rep_key]
                    row = self.results_table.rowCount()
                    self.results_table.insertRow(row)

                    # Col 0: Nom amb R si >1 replica
                    display_name = f"{sample_name} R{rep_key}" if len(replicas) > 1 else sample_name
                    item_name = QTableWidgetItem(display_name)
                    item_name.setData(Qt.UserRole, sample_name)
                    self.results_table.setItem(row, 0, item_name)

                    # Col 1: Inj
                    idx = rep_data.get("injection_index")
                    inj_item = QTableWidgetItem(str(idx) if idx is not None else "-")
                    inj_item.setForeground(QBrush(QColor("#888")))
                    self.results_table.setItem(row, 1, inj_item)

                    # Col 2-3: Sense selectors (cada injeccio es mostra directament)
                    self.results_table.setItem(row, 2, QTableWidgetItem(f"R{rep_key}"))
                    self.results_table.setItem(row, 3, QTableWidgetItem(f"R{rep_key}"))

                    # Col 4: A_DOC amb tooltip fraccions
                    areas = rep_data.get("areas") or {}
                    doc_areas = areas.get("DOC") or {}
                    area_direct = doc_areas.get("total", 0)
                    a_doc_item = QTableWidgetItem(f"{area_direct:.0f}" if area_direct else "-")
                    frac_tip = []
                    for frac in FRACTION_ORDER:
                        fa = doc_areas.get(frac, 0)
                        if fa:
                            pct = (fa / area_direct * 100) if area_direct > 0 else 0
                            frac_tip.append(f"{frac}: {fa:.0f} ({pct:.0f}%)")
                    if frac_tip:
                        a_doc_item.setToolTip("Fraccions DOC:\n" + "\n".join(frac_tip))
                    self.results_table.setItem(row, 4, a_doc_item)

                    # Col 5: ppm
                    ppm_direct = quantification.get("concentration_ppm_direct") or quantification.get("concentration_ppm")
                    self.results_table.setItem(row, 5, QTableWidgetItem(
                        f"{ppm_direct:.2f}" if ppm_direct else "-"))

                    # Col 6: A_UIB
                    areas_uib = rep_data.get("areas_uib") or {}
                    area_uib = areas_uib.get("total", 0)
                    self.results_table.setItem(row, 6, QTableWidgetItem(
                        f"{area_uib:.0f}" if area_uib else "-"))

                    # Col 7: ppm_U
                    ppm_uib = quantification.get("concentration_ppm_uib")
                    self.results_table.setItem(row, 7, QTableWidgetItem(
                        f"{ppm_uib:.2f}" if ppm_uib else "-"))

                    # Col 8: SNR
                    snr_info = rep_data.get("snr_info") or {}
                    snr_direct = snr_info.get("snr_direct", 0)
                    snr_item = QTableWidgetItem(f"{snr_direct:.0f}" if snr_direct else "-")
                    if snr_direct and snr_direct < 10:
                        snr_item.setForeground(QBrush(QColor(COLOR_ERROR)))
                    elif snr_direct and snr_direct < 50:
                        snr_item.setForeground(QBrush(QColor(COLOR_WARNING)))
                    snr_uib = snr_info.get("snr_uib", 0)
                    if snr_uib:
                        snr_item.setToolTip(f"SNR UIB: {snr_uib:.0f}")
                    self.results_table.setItem(row, 8, snr_item)

                    # Col 9: A_254
                    area_254 = (areas.get("A254") or {}).get("total", 0)
                    self.results_table.setItem(row, 9, QTableWidgetItem(
                        f"{area_254:.1f}" if area_254 else "-"))

                    # Col 10: SNR_254
                    snr_info_dad = rep_data.get("snr_info_dad") or {}
                    snr_254 = (snr_info_dad.get("A254") or {}).get("snr", 0)
                    snr_254_item = QTableWidgetItem(f"{snr_254:.0f}" if snr_254 else "-")
                    if snr_254 and snr_254 < 10:
                        snr_254_item.setForeground(QBrush(QColor(COLOR_ERROR)))
                    elif snr_254 and snr_254 < 50:
                        snr_254_item.setForeground(QBrush(QColor(COLOR_WARNING)))
                    self.results_table.setItem(row, 10, snr_254_item)

                    # Col 11-12: No R2 (sense comparacio repliques)
                    self.results_table.setItem(row, 11, QTableWidgetItem("-"))
                    self.results_table.setItem(row, 12, QTableWidgetItem("-"))

                    # Col 13: HCI
                    hci_val = quantification.get("hci")
                    if hci_val is not None:
                        hci_char = quantification.get("hci_character", "")
                        abbrev = "HA" if "HA" in hci_char else "FA" if "FA" in hci_char else "Mix"
                        hci_item = QTableWidgetItem(f"{hci_val:.1f} {abbrev}")
                        if hci_val > 60:
                            hci_item.setBackground(QBrush(QColor("#FADBD8")))
                        elif hci_val < 40:
                            hci_item.setBackground(QBrush(QColor("#D6EAF8")))
                        else:
                            hci_item.setBackground(QBrush(QColor("#D5F5E3")))
                    else:
                        hci_item = QTableWidgetItem("-")
                    self.results_table.setItem(row, 13, hci_item)

                    # Col 14: Estat (anomalies, igual que mostres)
                    status_color, status_text, tooltip = self._classify_sample_status(
                        rep_data, rep_data, comparison, sample_data=sample_data)
                    status_item = QTableWidgetItem(status_text)
                    status_item.setForeground(QBrush(QColor(status_color)))
                    status_item.setToolTip(tooltip)
                    self.results_table.setItem(row, 14, status_item)

                    # Fons gris
                    blank_bg = QBrush(QColor("#F4F6F6"))
                    for c in range(self.results_table.columnCount()):
                        item = self.results_table.item(row, c)
                        if item:
                            item.setBackground(blank_bg)

                n_blank += 1
                continue

            # --- Regular sample rendering ---
            row = self.results_table.rowCount()
            self.results_table.insertRow(row)
            self._sample_row_map[sample_name] = row

            replicas = sample_data.get("replicas") or {}
            comparison = sample_data.get("comparison") or {}
            recommendation = sample_data.get("recommendation") or {}
            selected = sample_data.get("selected") or {"doc": "1", "dad": "1"}
            quantification = sample_data.get("quantification") or {}

            doc_rec = (recommendation.get("doc") or {}).get("replica", "1")
            dad_rec = (recommendation.get("dad") or {}).get("replica", "1")
            doc_sel = selected.get("doc", doc_rec)
            dad_sel = selected.get("dad", dad_rec)
            doc_rep = replicas.get(doc_sel, {})
            dad_rep = replicas.get(dad_sel, {})

            # Col 0: Sample name
            item_name = QTableWidgetItem(sample_name)
            item_name.setData(Qt.UserRole, sample_name)
            self.results_table.setItem(row, 0, item_name)

            # Col 1: Inj (injection indices)
            inj_indices = []
            for rk, rd in sorted(replicas.items()):
                idx = rd.get("injection_index")
                if idx is not None:
                    inj_indices.append(str(idx))
            inj_text = ", ".join(inj_indices) if inj_indices else "-"
            inj_item = QTableWidgetItem(inj_text)
            inj_item.setForeground(QBrush(QColor("#888")))
            if inj_indices:
                tip_parts = []
                for rk, rd in sorted(replicas.items()):
                    idx = rd.get("injection_index")
                    if idx is not None:
                        tip_parts.append(f"R{rk}: inj #{idx}")
                inj_item.setToolTip("\n".join(tip_parts))
            self.results_table.setItem(row, 1, inj_item)

            # Col 2: Sel DOC -- replica selector with (s) for suggested + "Cap" option
            doc_combo = QComboBox()
            doc_combo.setStyleSheet("QComboBox { border: none; background: transparent; padding: 2px; }")
            for rep_num in sorted(replicas.keys()):
                label = f"R{rep_num} (s)" if rep_num == doc_rec else f"R{rep_num}"
                doc_combo.addItem(label, rep_num)
                if rep_num == doc_sel:
                    doc_combo.setCurrentIndex(doc_combo.count() - 1)
            doc_combo.addItem("Cap", "none")
            if doc_sel == "none":
                doc_combo.setCurrentIndex(doc_combo.count() - 1)
            doc_combo.currentIndexChanged.connect(
                lambda idx, name=sample_name: self._on_doc_replica_changed(name)
            )
            self.results_table.setCellWidget(row, 2, doc_combo)

            # Col 3: Sel DAD -- replica selector with (s) for suggested + "Cap" option
            dad_combo = QComboBox()
            dad_combo.setStyleSheet("QComboBox { border: none; background: transparent; padding: 2px; }")
            for rep_num in sorted(replicas.keys()):
                label = f"R{rep_num} (s)" if rep_num == dad_rec else f"R{rep_num}"
                dad_combo.addItem(label, rep_num)
                if rep_num == dad_sel:
                    dad_combo.setCurrentIndex(dad_combo.count() - 1)
            dad_combo.addItem("Cap", "none")
            if dad_sel == "none":
                dad_combo.setCurrentIndex(dad_combo.count() - 1)
            dad_combo.currentIndexChanged.connect(
                lambda idx, name=sample_name: self._on_dad_replica_changed(name)
            )
            self.results_table.setCellWidget(row, 3, dad_combo)

            # --- DOC columns (from DOC replica) ---

            # Col 4: A_DOC
            areas = doc_rep.get("areas") or {}
            doc_areas = areas.get("DOC") or {}
            area_direct = doc_areas.get("total", 0)
            self.results_table.setItem(row, 4, QTableWidgetItem(
                f"{area_direct:.0f}" if area_direct else "-"))

            # Col 5: ppm
            ppm_direct = quantification.get("concentration_ppm_direct") or quantification.get("concentration_ppm")
            self.results_table.setItem(row, 5, QTableWidgetItem(
                f"{ppm_direct:.2f}" if ppm_direct else "-"))

            # Col 6: A_UIB
            areas_uib = doc_rep.get("areas_uib") or {}
            area_uib = areas_uib.get("total", 0)
            self.results_table.setItem(row, 6, QTableWidgetItem(
                f"{area_uib:.0f}" if area_uib else "-"))

            # Col 7: ppm_U
            ppm_uib = quantification.get("concentration_ppm_uib")
            self.results_table.setItem(row, 7, QTableWidgetItem(
                f"{ppm_uib:.2f}" if ppm_uib else "-"))

            # Col 8: SNR (DOC Direct)
            snr_info = doc_rep.get("snr_info") or {}
            snr_direct = snr_info.get("snr_direct", 0)
            snr_item = QTableWidgetItem(f"{snr_direct:.0f}" if snr_direct else "-")
            if snr_direct and snr_direct < 10:
                snr_item.setForeground(QBrush(QColor(COLOR_ERROR)))
            elif snr_direct and snr_direct < 50:
                snr_item.setForeground(QBrush(QColor(COLOR_WARNING)))
            snr_uib = snr_info.get("snr_uib", 0)
            if snr_uib:
                snr_item.setToolTip(f"SNR UIB: {snr_uib:.0f}")
            self.results_table.setItem(row, 8, snr_item)

            # --- DAD columns (from DAD replica) ---

            # Col 9: A_254
            dad_areas = (dad_rep.get("areas") or {})
            area_254 = (dad_areas.get("A254") or {}).get("total", 0)
            self.results_table.setItem(row, 9, QTableWidgetItem(
                f"{area_254:.1f}" if area_254 else "-"))

            # Col 10: SNR_254
            snr_info_dad = dad_rep.get("snr_info_dad") or {}
            snr_254 = (snr_info_dad.get("A254") or {}).get("snr", 0)
            snr_254_item = QTableWidgetItem(f"{snr_254:.0f}" if snr_254 else "-")
            if snr_254 and snr_254 < 10:
                snr_254_item.setForeground(QBrush(QColor(COLOR_ERROR)))
            elif snr_254 and snr_254 < 50:
                snr_254_item.setForeground(QBrush(QColor(COLOR_WARNING)))
            self.results_table.setItem(row, 10, snr_254_item)

            # --- Correlation columns (sample-level, not replica-specific) ---

            # Col 11: R2_DOC
            r2_doc = comparison.get("doc", {}).get("pearson", 0) if comparison else 0
            r2_doc_item = QTableWidgetItem(f"{r2_doc:.4f}" if r2_doc > 0 else "-")
            if 0 < r2_doc < 0.990:
                r2_doc_item.setForeground(QBrush(QColor(COLOR_WARNING)))
            self.results_table.setItem(row, 11, r2_doc_item)

            # Col 12: R2_DAD (min across wavelengths)
            dad_comp = comparison.get("dad", {}) if comparison else {}
            r2_dad_min = dad_comp.get("pearson_min", 0)
            wl_min = dad_comp.get("wavelength_min", "")
            if r2_dad_min > 0:
                cell_text = f"{r2_dad_min:.4f}"
                if 0 < r2_dad_min < 0.990 and wl_min:
                    cell_text += f" (A{wl_min})"
            else:
                cell_text = "-"
            r2_dad_item = QTableWidgetItem(cell_text)
            if 0 < r2_dad_min < 0.990:
                r2_dad_item.setForeground(QBrush(QColor(COLOR_WARNING)))
            pearson_per_wl = dad_comp.get("pearson_per_wavelength", {})
            if pearson_per_wl:
                tip_lines = []
                for wl, val in sorted(pearson_per_wl.items()):
                    marker = " <- min" if str(wl) == str(wl_min) else ""
                    warn = " !" if val < 0.990 else ""
                    tip_lines.append(f"A{wl}: {val:.4f}{warn}{marker}")
                r2_dad_item.setToolTip("\n".join(tip_lines))
            self.results_table.setItem(row, 12, r2_dad_item)

            # Col 13: HCI (Humic Character Index)
            hci_val = quantification.get("hci")
            if hci_val is not None:
                hci_char = quantification.get("hci_character", "")
                abbrev = "HA" if "HA" in hci_char else "FA" if "FA" in hci_char else "Mix"
                hci_item = QTableWidgetItem(f"{hci_val:.1f} {abbrev}")
                if hci_val > 60:
                    hci_item.setBackground(QBrush(QColor("#FADBD8")))
                elif hci_val < 40:
                    hci_item.setBackground(QBrush(QColor("#D6EAF8")))
                else:
                    hci_item.setBackground(QBrush(QColor("#D5F5E3")))
                hci_item.setToolTip(
                    f"Humic Character Index: {hci_val:.1f} ({hci_char})\n"
                    f"Model PCA+LDA v2.0")
            else:
                hci_item = QTableWidgetItem("-")
            self.results_table.setItem(row, 13, hci_item)

            # Col 14: Estat (considers both DOC and DAD replicas)
            status_color, status_text, tooltip = self._classify_sample_status(
                doc_rep, dad_rep, comparison, sample_data=sample_data)
            status_item = QTableWidgetItem(status_text)
            status_item.setForeground(QBrush(QColor(status_color)))
            status_item.setToolTip(tooltip)
            self.results_table.setItem(row, 14, status_item)

            # Count stats for regular samples
            if status_color == COLOR_ERROR:
                n_error += 1
            elif status_color == COLOR_WARNING:
                n_warning += 1
            else:
                n_ok += 1

        # --- CONTROL (light analysis) separator + simplified rows ---
        if control_names and show_control:
            n_cols = self.results_table.columnCount()
            sep_row = self.results_table.rowCount()
            self.results_table.insertRow(sep_row)
            sep_item = QTableWidgetItem("--- NETEJA ---")
            sep_item.setFlags(Qt.ItemIsEnabled)
            sep_font = QFont()
            sep_font.setBold(True)
            sep_item.setFont(sep_font)
            sep_item.setForeground(QBrush(QColor("#888888")))
            self.results_table.setItem(sep_row, 0, sep_item)
            self.results_table.setSpan(sep_row, 0, 1, n_cols)
            sep_bg = QBrush(QColor("#E8E8E8"))
            for c in range(n_cols):
                item = self.results_table.item(sep_row, c)
                if item is None:
                    item = QTableWidgetItem("")
                    self.results_table.setItem(sep_row, c, item)
                item.setBackground(sep_bg)

            for sample_name in control_names:
                sample_data = self.samples_grouped[sample_name]
                row = self.results_table.rowCount()
                self.results_table.insertRow(row)
                self._sample_row_map[sample_name] = row

                replicas = sample_data.get("replicas") or {}
                selected = sample_data.get("selected") or {}
                doc_sel = selected.get("doc", sorted(replicas.keys())[0] if replicas else "1")
                doc_rep = replicas.get(doc_sel, {})

                # Col 0: Sample name
                item_name = QTableWidgetItem(sample_name)
                item_name.setData(Qt.UserRole, sample_name)
                self.results_table.setItem(row, 0, item_name)

                # Col 1: Inj
                inj_indices = []
                for rk, rd in sorted(replicas.items()):
                    idx = rd.get("injection_index")
                    if idx is not None:
                        inj_indices.append(str(idx))
                inj_item = QTableWidgetItem(", ".join(inj_indices) if inj_indices else "-")
                inj_item.setForeground(QBrush(QColor("#888")))
                self.results_table.setItem(row, 1, inj_item)

                # Col 2-3: No selectors for control
                self.results_table.setItem(row, 2, QTableWidgetItem("-"))
                self.results_table.setItem(row, 3, QTableWidgetItem("-"))

                # Col 4: A_DOC (area_total from light analysis)
                area_total = doc_rep.get("area_total", 0)
                self.results_table.setItem(row, 4, QTableWidgetItem(
                    f"{area_total:.0f}" if area_total else "-"))

                # Col 5-7: No ppm, no UIB
                for c in (5, 6, 7):
                    self.results_table.setItem(row, c, QTableWidgetItem("-"))

                # Col 8: SNR
                snr = doc_rep.get("snr", 0)
                snr_item = QTableWidgetItem(f"{snr:.0f}" if snr else "-")
                if snr and snr < 10:
                    snr_item.setForeground(QBrush(QColor(COLOR_ERROR)))
                elif snr and snr < 50:
                    snr_item.setForeground(QBrush(QColor(COLOR_WARNING)))
                self.results_table.setItem(row, 8, snr_item)

                # Col 9-13: No DAD, no R2, no HCI
                for c in (9, 10, 11, 12, 13):
                    self.results_table.setItem(row, c, QTableWidgetItem("-"))

                # Col 14: Neteja
                type_item = QTableWidgetItem("Neteja")
                type_item.setForeground(QBrush(QColor("#888888")))
                self.results_table.setItem(row, 14, type_item)

                # Light grey background
                light_bg = QBrush(QColor("#F0F0F0"))
                for c in range(n_cols):
                    item = self.results_table.item(row, c)
                    if item:
                        item.setBackground(light_bg)

                n_control += 1

        # Update stats — return counts for the coordinator to use
        self._last_stats = {
            "n_ok": n_ok, "n_warning": n_warning, "n_error": n_error,
            "n_blank": n_blank, "n_control": n_control,
        }

    def get_last_stats(self):
        """Returns the stats from the last _populate_table call.

        Returns:
            dict with keys: n_ok, n_warning, n_error, n_blank, n_control
        """
        return getattr(self, '_last_stats', {
            "n_ok": 0, "n_warning": 0, "n_error": 0,
            "n_blank": 0, "n_control": 0,
        })

    # ------------------------------------------------------------------
    # Anomaly severity classification
    # ------------------------------------------------------------------

    def _classify_sample_status(self, doc_rep_data, dad_rep_data, comparison,
                                sample_data=None):
        """Classifica l'estat d'una mostra considerant ambdues repliques (DOC + DAD).

        Usa ANOMALY_CATALOG com a font de veritat per severitat, icones i labels.

        Args:
            doc_rep_data: Dades de la replica DOC seleccionada
            dad_rep_data: Dades de la replica DAD seleccionada
            comparison: Comparacio entre repliques
            sample_data: Dict complet del sample_group (per accedir a sample_valid, repaired)

        Returns (color, status_text, tooltip).
        """
        # Comprovar si l'usuari ha seleccionat "Cap"
        if sample_data:
            selected = sample_data.get("selected", {})
            if selected.get("doc") == "none":
                return COLOR_ERROR, "NO V\u00c0L", "Usuari ha seleccionat 'Cap' \u2014 No es quantificar\u00e0 ni exportar\u00e0"
            # Comprovar mostra no valida (ambdues repliques amb anomalies no reparables)
            if sample_data.get("sample_valid") is False and not sample_data.get("repaired"):
                reason = (sample_data.get("recommendation", {})
                          .get("doc", {}).get("reason", "Ambdues r\u00e8pliques amb anomalies cr\u00edtiques"))
                return COLOR_ERROR, "NO V\u00c0L", f"Mostra no v\u00e0lida \u2014 {reason}\nSeleccionar 'Cap' o generar noves dades"

        # Merge anomalies from both replicas (deduplicate by code)
        doc_anomalies = doc_rep_data.get("anomalies", [])
        dad_anomalies = dad_rep_data.get("anomalies", [])
        all_anomalies = list(doc_anomalies)
        existing_codes = get_anomaly_codes(all_anomalies)
        for a in dad_anomalies:
            code = a.get("code") if isinstance(a, dict) else str(a).replace("_REPAIRED", "")
            if code not in existing_codes:
                all_anomalies.append(a)
                existing_codes.add(code)

        classified = classify_anomalies(all_anomalies)
        timeout_info = doc_rep_data.get("timeout_info", {})
        timeout_severity = timeout_info.get("severity", "OK")
        n_timeouts = timeout_info.get("n_timeouts", 0)
        replica_warnings = []
        if comparison:
            for domain in ("doc", "dad"):
                replica_warnings.extend((comparison.get(domain) or {}).get("warnings", []))

        # Determine severity
        has_blocker = bool(classified["blocker"])
        has_warn = bool(classified["warning"] or classified["repaired"]
                        or (timeout_severity in ("WARNING", "CRITICAL"))
                        or replica_warnings)

        # Build concise status text
        n_blocker = len(classified["blocker"])
        n_warn = len(classified["warning"])
        n_repaired = len(classified["repaired"])
        # Repairable indicator
        can_repair = (sample_data and sample_data.get("repairable")
                      and not sample_data.get("repaired"))
        repair_icon = " \U0001f527" if can_repair else ""

        if has_blocker:
            status_color = COLOR_ERROR
            # Mostrar motiu concret del primer blocker
            first_blocker = classified["blocker"][0]
            b_code = first_blocker.get("code") if isinstance(first_blocker, dict) else str(first_blocker)
            b_entry = ANOMALY_CATALOG.get(b_code, {})
            b_label = b_entry.get("label", b_code)
            if n_blocker == 1:
                status_text = b_label + repair_icon
            else:
                status_text = f"{b_label} +{n_blocker - 1}" + repair_icon
        elif has_warn or n_repaired:
            status_color = COLOR_WARNING
            parts = []
            if n_warn:
                parts.append(f"{n_warn} av\u00eds" if n_warn == 1 else f"{n_warn} avisos")
            if n_repaired:
                parts.append(f"{n_repaired} reparat" if n_repaired == 1 else f"{n_repaired} reparats")
            if n_timeouts > 0:
                parts.append(f"{n_timeouts} timeout")
            status_text = " \u00b7 ".join(parts) + repair_icon
        else:
            status_color = COLOR_SUCCESS
            status_text = "OK"

        # Build tooltip with catalog labels + action hints
        tooltip_parts = []
        for key, label_prefix in [("blocker", "CR\u00cdTIC"), ("repaired", "REPARAT"),
                                    ("warning", "Av\u00eds"), ("info", "Info")]:
            items = classified[key]
            if items:
                for a in items:
                    code = a.get("code") if isinstance(a, dict) else str(a).replace("_REPAIRED", "")
                    entry = ANOMALY_CATALOG.get(code, {})
                    lbl = entry.get("label", code)
                    det = a.get("details", {}) if isinstance(a, dict) else {}
                    if det.get("snr"):
                        lbl += f" (SNR={det['snr']:.1f})"
                    line = f"{label_prefix}: {lbl}"
                    action = entry.get("action", "")
                    if action:
                        line += f"\n   \u2192 {action}"
                    tooltip_parts.append(line)

        if n_timeouts > 0:
            zone_summary = timeout_info.get("zone_summary", {})
            zones_str = ", ".join(zone_summary.keys()) if zone_summary else "?"
            tooltip_parts.append(
                f"Timeouts Direct: {n_timeouts} ({timeout_severity}) \u2014 zones: {zones_str}"
            )
            # UIB timeout propagat
            uib_ti = doc_rep_data.get("timeout_info_uib") or {}
            if uib_ti.get("n_timeouts", 0) > 0:
                uib_zone_summary = uib_ti.get("zone_summary", {})
                uib_in_peak = doc_rep_data.get("timeout_in_peak_uib", False)
                uib_zones_str = ", ".join(uib_zone_summary.keys()) if uib_zone_summary else "?"
                uib_tip = f"Timeouts UIB: {uib_ti['n_timeouts']} \u2014 zones: {uib_zones_str}"
                if uib_in_peak:
                    uib_tip += " \u2014 DINS DEL PIC UIB!"
                tooltip_parts.append(uib_tip)
        if replica_warnings:
            for rw in replica_warnings:
                if isinstance(rw, dict):
                    tooltip_parts.append(rw.get("label", rw.get("code", str(rw))))
                else:
                    tooltip_parts.append(str(rw))

        # Repairable hint
        if sample_data and sample_data.get("repairable") and not sample_data.get("repaired"):
            tooltip_parts.append("Pic amb cim irregular reparable \u2014 Clic a Estat per obrir di\u00e0leg de reparaci\u00f3")
        elif sample_data and sample_data.get("repaired"):
            tooltip_parts.append("Clic a Estat per desfer o veure detalls de la reparaci\u00f3")

        tooltip = "\n".join(tooltip_parts) if tooltip_parts else "OK"
        return status_color, status_text, tooltip

    # ------------------------------------------------------------------
    # Replica change (separate DOC / DAD handlers)
    # ------------------------------------------------------------------

    def _on_doc_replica_changed(self, sample_name):
        """Gestiona el canvi de replica DOC (inclou opcio 'Cap')."""
        if sample_name not in self.samples_grouped:
            return
        row = self._sample_row_map.get(sample_name)
        if row is None:
            return
        combo = self.results_table.cellWidget(row, 2)
        if combo:
            new_replica = combo.currentData()
            self.samples_grouped[sample_name]["selected"]["doc"] = new_replica
            if new_replica == "none":
                # Marcar mostra com no valida per DOC
                self.samples_grouped[sample_name]["sample_valid"] = False
                self.samples_grouped[sample_name]["quantification"] = {
                    "concentration_ppm": None,
                    "concentration_ppm_direct": None,
                    "concentration_ppm_uib": None,
                    "area_total": None,
                    "valid": False,
                    "reason": "Usuari ha seleccionat 'Cap' per DOC"
                }
            else:
                # Restaurar validesa si era "none" abans
                self.samples_grouped[sample_name]["sample_valid"] = True
                self._update_quantification(sample_name)
            self._update_doc_columns(row, sample_name)
            self._update_estat_column(row, sample_name)

    def _on_dad_replica_changed(self, sample_name):
        """Gestiona el canvi de replica DAD (inclou opcio 'Cap')."""
        if sample_name not in self.samples_grouped:
            return
        row = self._sample_row_map.get(sample_name)
        if row is None:
            return
        combo = self.results_table.cellWidget(row, 3)
        if combo:
            new_replica = combo.currentData()
            self.samples_grouped[sample_name]["selected"]["dad"] = new_replica
            self._update_dad_columns(row, sample_name)
            self._update_estat_column(row, sample_name)

    def _update_doc_columns(self, row, sample_name):
        """Actualitza columnes DOC (4-8) quan canvia la replica DOC."""
        sample_data = self.samples_grouped[sample_name]
        selected = sample_data.get("selected", {})
        doc_sel = selected.get("doc", "1")
        replicas = sample_data.get("replicas", {})

        # "Cap" seleccionat -> buidar columnes
        if doc_sel == "none":
            for col in (4, 5, 6, 7, 8, 13):
                item = self.results_table.item(row, col)
                if item:
                    item.setText("-")
                    if col == 13:
                        item.setBackground(QBrush(QColor("#FFFFFF")))
                        item.setToolTip("")
            return

        doc_rep = replicas.get(doc_sel, {})
        quantification = sample_data.get("quantification", {})

        areas = doc_rep.get("areas") or {}
        doc_areas = areas.get("DOC") or {}
        areas_uib = doc_rep.get("areas_uib") or {}

        # Col 4: A_DOC
        area_direct = doc_areas.get("total", 0)
        self.results_table.item(row, 4).setText(f"{area_direct:.0f}" if area_direct else "-")

        # Col 5: ppm
        ppm_direct = quantification.get("concentration_ppm_direct") or quantification.get("concentration_ppm")
        self.results_table.item(row, 5).setText(f"{ppm_direct:.2f}" if ppm_direct else "-")

        # Col 6: A_UIB
        area_uib = areas_uib.get("total", 0)
        self.results_table.item(row, 6).setText(f"{area_uib:.0f}" if area_uib else "-")

        # Col 7: ppm_U
        ppm_uib = quantification.get("concentration_ppm_uib")
        self.results_table.item(row, 7).setText(f"{ppm_uib:.2f}" if ppm_uib else "-")

        # Col 8: SNR (DOC Direct)
        snr_info = doc_rep.get("snr_info") or {}
        snr_direct = snr_info.get("snr_direct", 0)
        snr_item = self.results_table.item(row, 8)
        if snr_item:
            snr_item.setText(f"{snr_direct:.0f}" if snr_direct else "-")
            if snr_direct and snr_direct < 10:
                snr_item.setForeground(QBrush(QColor(COLOR_ERROR)))
            elif snr_direct and snr_direct < 50:
                snr_item.setForeground(QBrush(QColor(COLOR_WARNING)))
            else:
                snr_item.setForeground(QBrush(QColor("#000000")))
            snr_uib = snr_info.get("snr_uib", 0)
            snr_item.setToolTip(f"SNR UIB: {snr_uib:.0f}" if snr_uib else "")

        # Col 13: HCI (update from new quantification)
        hci_item = self.results_table.item(row, 13)
        if hci_item:
            hci_val = quantification.get("hci")
            if hci_val is not None:
                hci_char = quantification.get("hci_character", "")
                abbrev = "HA" if "HA" in hci_char else "FA" if "FA" in hci_char else "Mix"
                hci_item.setText(f"{hci_val:.1f} {abbrev}")
                if hci_val > 60:
                    hci_item.setBackground(QBrush(QColor("#FADBD8")))
                elif hci_val < 40:
                    hci_item.setBackground(QBrush(QColor("#D6EAF8")))
                else:
                    hci_item.setBackground(QBrush(QColor("#D5F5E3")))
                hci_item.setToolTip(
                    f"Humic Character Index: {hci_val:.1f} ({hci_char})\n"
                    f"Model PCA+LDA v2.0")
            else:
                hci_item.setText("-")
                hci_item.setBackground(QBrush(QColor("#FFFFFF")))
                hci_item.setToolTip("")

    def _update_dad_columns(self, row, sample_name):
        """Actualitza columnes DAD (9-10) quan canvia la replica DAD."""
        sample_data = self.samples_grouped[sample_name]
        selected = sample_data.get("selected", {})
        dad_sel = selected.get("dad", "1")
        replicas = sample_data.get("replicas", {})
        dad_rep = replicas.get(dad_sel, {})

        # Col 9: A_254
        dad_areas = (dad_rep.get("areas") or {})
        area_254 = (dad_areas.get("A254") or {}).get("total", 0)
        item_9 = self.results_table.item(row, 9)
        if item_9:
            item_9.setText(f"{area_254:.1f}" if area_254 else "-")

        # Col 10: SNR_254
        snr_info_dad = dad_rep.get("snr_info_dad") or {}
        snr_254 = (snr_info_dad.get("A254") or {}).get("snr", 0)
        snr_254_item = self.results_table.item(row, 10)
        if snr_254_item:
            snr_254_item.setText(f"{snr_254:.0f}" if snr_254 else "-")
            if snr_254 and snr_254 < 10:
                snr_254_item.setForeground(QBrush(QColor(COLOR_ERROR)))
            elif snr_254 and snr_254 < 50:
                snr_254_item.setForeground(QBrush(QColor(COLOR_WARNING)))
            else:
                snr_254_item.setForeground(QBrush(QColor("#000000")))

    def _update_estat_column(self, row, sample_name):
        """Actualitza la columna Estat (col 14) considerant ambdues repliques."""
        sample_data = self.samples_grouped[sample_name]
        selected = sample_data.get("selected", {})
        replicas = sample_data.get("replicas", {})
        comparison = sample_data.get("comparison") or {}
        doc_rep = replicas.get(selected.get("doc", "1"), {})
        dad_rep = replicas.get(selected.get("dad", "1"), {})

        status_color, status_text, tooltip = self._classify_sample_status(
            doc_rep, dad_rep, comparison, sample_data=sample_data)
        status_item = self.results_table.item(row, 14)
        if status_item:
            status_item.setText(status_text)
            status_item.setForeground(QBrush(QColor(status_color)))
            status_item.setToolTip(tooltip)

    # ------------------------------------------------------------------
    # Quantification recalculation
    # ------------------------------------------------------------------

    def _update_quantification(self, sample_name):
        """Recalcula la quantificacio per una mostra."""
        try:
            from hpsec_analyze import quantify_sample
            from hpsec_calibrate import get_all_active_calibrations

            sample_data = self.samples_grouped[sample_name]

            # Respectar exclusio de quantificacio
            if sample_data.get("skip_quantification"):
                sample_data["quantification"] = {
                    "concentration_ppm": None,
                    "concentration_ppm_direct": None,
                    "concentration_ppm_uib": None,
                    "valid": False,
                    "reason": sample_data["quantification"].get("reason",
                              "Exclosa de quantificaci\u00f3") if sample_data.get("quantification") else
                              "Exclosa de quantificaci\u00f3"
                }
                return

            selected_doc = sample_data["selected"]["doc"]
            selected_replica = sample_data["replicas"].get(selected_doc)

            if selected_replica:
                processed = self.main_window.processed_data or {}
                method = processed.get("method", "COLUMN")
                mode = "BP" if method == "BP" else "COLUMN"
                seq_date = processed.get("seq_date")

                calibration_data = None
                inj_volume = selected_replica.get("inj_volume")
                seq_path = processed.get("seq_path", "")

                if seq_path and inj_volume:
                    try:
                        active_cals = get_all_active_calibrations(seq_path, mode)
                        for cal in active_cals:
                            cal_vol = cal.get("volume_uL", 0)
                            if cal_vol > 0 and abs(cal_vol - inj_volume) < 0.1:
                                calibration_data = cal
                                break
                        if not calibration_data and active_cals:
                            calibration_data = active_cals[0]
                    except Exception as e:
                        logger.warning(f"Error carregant calibracions: {e}")

                if not calibration_data:
                    calibration_data = self.main_window.calibration_data
                    # Ensure volume context from replica is available
                    if calibration_data and inj_volume:
                        if not calibration_data.get("volume_uL"):
                            calibration_data = dict(calibration_data)
                            calibration_data["volume_uL"] = inj_volume

                quantification = quantify_sample(
                    selected_replica, calibration_data,
                    mode=mode, seq_date=seq_date
                )
                # Propagar HCI de la replica seleccionada
                hci = selected_replica.get("hci")
                if hci is not None:
                    quantification["hci"] = hci
                    quantification["hci_character"] = selected_replica.get("hci_character", "")
                sample_data["quantification"] = quantification
        except Exception as e:
            logger.error(f"Error recalculant quantificaci\u00f3: {e}")
            self.main_window.set_status(f"Error quantificaci\u00f3: {e}", 5000)

    # ------------------------------------------------------------------
    # Table interaction
    # ------------------------------------------------------------------

    def _on_table_cell_click(self, row, col):
        """Handler per clic a cel-la -- col 14 (Estat) obre dialeg reparacio multi."""
        if col != 14:
            return
        item = self.results_table.item(row, 0)
        if not item:
            return
        sample_name = item.data(Qt.UserRole)
        if not sample_name:
            return
        sample_data = self.samples_grouped.get(sample_name)
        if not sample_data:
            return

        targets = self._find_repair_targets(sample_name)
        if not targets:
            return

        self._open_repair_dialog(sample_name)

    def _find_repair_targets(self, sample_name):
        """Busca repliques/senyals amb anomalies de cim irregular (pendents, reparades o dismissed)."""
        sample_data = self.samples_grouped.get(sample_name, {})
        replicas = sample_data.get("replicas", {})
        targets = []

        for rep_key, rep_data in replicas.items():
            anomalies = rep_data.get("anomalies", [])
            for signal_type, anom_key in [
                ("direct", "IRREGULAR_TOP_DIRECT"),
                ("uib", "IRREGULAR_TOP_UIB"),
            ]:
                for a in anomalies:
                    if isinstance(a, dict) and a.get("code") == anom_key:
                        targets.append((rep_key, signal_type))
                        break
                    elif isinstance(a, str) and anom_key in a:
                        targets.append((rep_key, signal_type))
                        break

        return targets

    def _open_repair_dialog(self, sample_name, rep_key=None, signal_type=None):
        """Obre el dialeg multi-reparacio per totes les repliques x senyals."""
        sample_data = self.samples_grouped.get(sample_name)
        if not sample_data:
            return

        method = "COLUMN"
        if self.main_window.processed_data:
            method = self.main_window.processed_data.get("method", "COLUMN")

        dialog = JaggedPeakRepairDialog(
            sample_name, sample_data, method, parent=self
        )
        dialog.repair_completed.connect(self._on_repair_action)
        dialog.exec()

    def _on_repair_action(self, sample_name):
        """Actualitza la taula despres d'una accio de reparacio."""
        row = self._sample_row_map.get(sample_name)
        if row is not None:
            self._update_quantification(sample_name)
            self._update_doc_columns(row, sample_name)
            self._update_estat_column(row, sample_name)

    def _on_table_double_click(self, index):
        """Handler per doble clic -- obre SampleDetailDialog per totes les tipologies."""
        row = index.row()
        item = self.results_table.item(row, 0)
        if item:
            sample_name = item.data(Qt.UserRole)
            if not sample_name:
                return  # Separator row
            sample_data = self.samples_grouped.get(sample_name)
            if not sample_data:
                return
            self._show_detail(sample_name)

    def _show_detail(self, sample_name):
        """Mostra el dialeg de detall (no-modal per evitar perdua de finestra)."""
        if sample_name not in self.samples_grouped:
            return
        # Tancar dialeg anterior si existeix
        if self._detail_dialog is not None:
            try:
                self._detail_dialog.close()
            except RuntimeError:
                pass
        method = "COLUMN"
        if self.main_window.processed_data:
            method = self.main_window.processed_data.get("method", "COLUMN")
        dialog = SampleDetailDialog(
            sample_name,
            self.samples_grouped[sample_name],
            method,
            parent=self
        )
        self._detail_dialog = dialog  # Mantenir referencia
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.finished.connect(lambda: self._on_detail_closed(sample_name))
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _on_detail_closed(self, sample_name):
        """Actualitza taula despres de tancar el dialeg de detall."""
        row = self._sample_row_map.get(sample_name)
        if row is not None:
            sample_data = self.samples_grouped[sample_name]
            if sample_data.get("repaired"):
                self._update_quantification(sample_name)
                self._update_doc_columns(row, sample_name)
                self._update_estat_column(row, sample_name)
        self._detail_dialog = None

    # ------------------------------------------------------------------
    # Report PDF generation
    # ------------------------------------------------------------------

    def _generate_report(self):
        """Genera el report PDF d'analisi (cridat des del wizard header)."""
        processed_data = self.main_window.processed_data
        if not processed_data:
            QMessageBox.warning(self, "Av\u00eds", "No hi ha dades processades.")
            return

        seq_path = processed_data.get("seq_path", "")
        if not seq_path:
            QMessageBox.warning(self, "Av\u00eds", "No s'ha trobat el path de la seq\u00fc\u00e8ncia.")
            return

        try:
            from generate_analysis_report import generate_analysis_report

            # Passar dades en memoria (inclou seleccions actuals de l'usuari)
            report_data = dict(processed_data)
            report_data["samples_grouped"] = self.samples_grouped

            result = generate_analysis_report(
                seq_path, analysis_data=report_data
            )

            if result:
                QMessageBox.information(
                    self, "Report generat",
                    f"PDF generat correctament:\n{result}"
                )
                import os
                os.startfile(str(Path(result).parent))
            else:
                QMessageBox.warning(
                    self, "Error",
                    "No s'ha pogut generar el report PDF."
                )

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(
                self, "Error",
                f"Error generant el report:\n{str(e)}"
            )

    # ------------------------------------------------------------------
    # Charts section
    # ------------------------------------------------------------------

    def _populate_charts(self, processed_data):
        """Prepara dades pels grafics i mostra la seccio."""
        if not HAS_MATPLOTLIB or not processed_data:
            return

        samples_grouped = processed_data.get("samples_grouped", {})
        method = processed_data.get("method", "COLUMN")
        is_bp = method.upper() == "BP"

        regular = {}
        blank = {}
        control = {}
        khp = {}
        for name, data in samples_grouped.items():
            st = data.get("sample_type", "SAMPLE")
            if st == "KHP":
                khp[name] = data
            elif st == "BLANK":
                # Expandir BLANK per injeccio (una entrada per replica)
                replicas = data.get("replicas") or {}
                for rep_key in sorted(replicas.keys()):
                    rep_data = replicas[rep_key]
                    display = f"{name} R{rep_key}" if len(replicas) > 1 else name
                    blank[display] = {
                        **data,
                        "replicas": {rep_key: rep_data},
                        "selected": {"doc": rep_key, "dad": rep_key},
                        "_single_injection": True,
                    }
            elif st == "CONTROL":
                control[name] = data
            else:
                regular[name] = data

        self._chart_regular = regular
        self._chart_blank = blank
        self._chart_control = control
        self._chart_khp = khp
        self._chart_is_bp = is_bp

        try:
            self._plot_timeout_chart(processed_data, is_bp)
        except Exception as e:
            logger.error(f"Error plotting timeout chart: {e}")

        self._build_sample_checkboxes(regular, blank, control, khp)
        self.charts_section.setVisible(True)

        # Charts sempre visibles -- dibuixar directament
        self._charts_initialized = True
        self._redraw_charts()

    def _build_sample_checkboxes(self, regular, blank, control, khp):
        """Registra mostres per categoria (sense checkboxes individuals)."""
        self._sample_checkboxes = []
        for name in sorted(regular.keys()):
            self._sample_checkboxes.append((None, name, "sample"))
        for name in sorted(blank.keys()):
            self._sample_checkboxes.append((None, name, "blank"))
        for name in sorted(control.keys()):
            self._sample_checkboxes.append((None, name, "control"))
        for name in sorted(khp.keys()):
            self._sample_checkboxes.append((None, name, "khp"))

        self._cat_counts = {
            "sample": len(regular),
            "blank": len(blank),
            "control": len(control),
            "khp": len(khp),
        }
        self._update_cat_btn_styles()

    def _update_cat_btn_styles(self):
        """Actualitza estil dels botons toggle de categoria."""
        if not hasattr(self, '_cat_buttons'):
            return
        STYLES = {
            "sample":  ("#2E86AB", "#fff"),
            "blank":   ("#95a5a6", "#fff"),
            "control": ("#888",    "#fff"),
            "khp":     ("#1565C0", "#fff"),
        }
        for cat_key, btn in self._cat_buttons.items():
            color, fg = STYLES.get(cat_key, ("#666", "#fff"))
            count = self._cat_counts.get(cat_key, 0)
            base_label = btn.text().split('(')[0].strip()
            btn.setText(f"{base_label} ({count})" if count else base_label)
            if btn.isChecked():
                btn.setStyleSheet(
                    f"QPushButton {{ background: {color}; color: {fg};"
                    f" font-size: 10px; font-weight: bold; padding: 3px 10px;"
                    f" border: none; border-radius: 3px; }}"
                    f"QPushButton:hover {{ opacity: 0.9; }}"
                )
            else:
                btn.setStyleSheet(
                    f"QPushButton {{ background: #f0f0f0; color: {color};"
                    f" font-size: 10px; padding: 3px 10px;"
                    f" border: 1px solid #ddd; border-radius: 3px; }}"
                    f"QPushButton:hover {{ background: #e4e4e4; }}"
                )

    def _on_cat_toggle(self):
        """Un boto de categoria ha canviat -- actualitza taula i grafics."""
        self._update_cat_btn_styles()
        if self.samples_grouped:
            self._populate_table()
        self._redraw_charts()

    def _on_wl_changed(self):
        """Longitud d'ona DAD seleccionada ha canviat -- redibuixar grafics DAD."""
        if self._charts_initialized:
            checked = self._get_checked_samples()
            reg = {k: v for k, v in checked.items()
                   if v.get("sample_type") != "CONTROL"}
            light = {k: v for k, v in checked.items()
                     if v.get("sample_type") == "CONTROL"}
            try:
                self._plot_dad_chart(reg, light)
                self._plot_dad_overlay(reg, light)
            except Exception as e:
                logger.error(f"Error redrawing DAD charts: {e}")

    def _get_selected_wl(self):
        """Retorna la longitud d'ona DAD seleccionada (ex: '254')."""
        if hasattr(self, '_wl_combo'):
            return self._wl_combo.currentData() or "254"
        return "254"

    def _get_checked_samples(self):
        """Retorna dict {name: data} de mostres de categories actives."""
        all_data = {}
        all_data.update(self._chart_regular)
        all_data.update(self._chart_blank)
        all_data.update(self._chart_control)
        all_data.update(getattr(self, '_chart_khp', {}))

        active_cats = {cat for cat, btn in self._cat_buttons.items() if btn.isChecked()}
        checked = {}
        for _cb, name, cat in self._sample_checkboxes:
            if cat in active_cats and name in all_data:
                checked[name] = all_data[name]
        return checked

    def _redraw_charts(self):
        """Redibuixa els 4 grafics amb les mostres seleccionades."""
        if not HAS_MATPLOTLIB:
            return
        checked = self._get_checked_samples()
        # BLANK va amb regular (fraccions+ppm complets), nomes CONTROL va a light
        reg = {k: v for k, v in checked.items()
               if v.get("sample_type") != "CONTROL"}
        light = {k: v for k, v in checked.items()
                 if v.get("sample_type") == "CONTROL"}
        is_bp = getattr(self, '_chart_is_bp', False)
        try:
            self._plot_doc_chart(reg, light, is_bp)
            self._plot_dad_chart(reg, light)
            self._plot_doc_overlay(reg, light, is_bp)
            self._plot_dad_overlay(reg, light)
        except Exception as e:
            logger.error(f"Error redrawing charts: {e}")

    def _plot_timeout_chart(self, processed_data, is_bp):
        """Diagrama fraccions (eix temps) amb comptador timeouts per zona."""
        self.timeout_figure.clear()
        ax = self.timeout_figure.add_subplot(111)

        zone_totals = {}
        zone_injections = {}  # zona -> [inj_idx, ...]
        for sample in processed_data.get("samples", []):
            ti = sample.get("timeout_info") or {}
            inj_idx = sample.get("injection_index")
            for zone, count in (ti.get("zone_summary") or {}).items():
                if count > 0:
                    zone_totals[zone] = zone_totals.get(zone, 0) + count
                    if inj_idx is not None:
                        zone_injections.setdefault(zone, []).append(str(inj_idx))

        if is_bp:
            zones = [
                ("BP_PEAK", 0, 5, "#E74C3C"),
                ("BP_TAIL", 5, 10, "#F39C12"),
            ]
            x_max = 12
        else:
            from hpsec_config import ConfigManager
            cfg = ConfigManager()
            fractions = cfg.get_all_fractions(mode="COLUMN")
            max_dur = cfg.get("chromatogram", "max_duration_min", default=78.65)
            zones = []
            first_start = 10.8
            if fractions:
                first_start = fractions[0][1]["start"]
            if first_start > 0:
                zones.append(("RUN_START", 0, first_start, "#95a5a6"))
            for name, frac in fractions:
                color = FRACTION_COLORS.get(name, "#95a5a6")
                zones.append((name, frac["start"], frac["end"], color))
            last_end = zones[-1][2] if zones else 70
            if max_dur > last_end:
                zones.append(("POST_RUN", last_end, max_dur, "#d5dbdb"))
            x_max = max_dur

        for zone_name, t0, t1, color in zones:
            count = zone_totals.get(zone_name, 0)
            injs = zone_injections.get(zone_name, [])
            alpha = 0.8 if count > 0 else 0.3
            ax.barh(0, t1 - t0, left=t0, height=0.6, color=color,
                    alpha=alpha, edgecolor='white', linewidth=0.5)
            mid = (t0 + t1) / 2
            if count > 0:
                inj_str = ",".join(injs[:6])
                if len(injs) > 6:
                    inj_str += "..."
                label = f"{zone_name}\ninj {inj_str}"
                fw = 'bold'
                fc = '#c0392b'
            else:
                label = zone_name
                fw = 'normal'
                fc = '#555'
            ax.text(mid, 0, label, ha='center', va='center',
                    fontsize=5.5, fontweight=fw, color=fc, fontfamily=_CHART_FONT)

        ax.set_xlim(0, x_max)
        ax.set_yticks([])
        ax.set_xlabel("Temps (min)", fontsize=_CHART_TICK_SIZE,
                       fontfamily=_CHART_FONT)
        ax.set_title("Timeouts TOC", fontsize=_CHART_TITLE_SIZE,
                      fontweight='bold', fontfamily=_CHART_FONT)
        ax.tick_params(axis='x', labelsize=6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        self.timeout_figure.tight_layout(pad=0.3)
        self.timeout_canvas.draw()

    @staticmethod
    def _chart_short_label(name, data):
        """Retorna etiqueta curta per eix X: index injeccio o nom truncat."""
        sel = (data.get("selected") or {}).get("doc", "1")
        rep = (data.get("replicas") or {}).get(sel, {})
        idx = rep.get("injection_index")
        if idx is not None:
            return str(idx)
        # Truncar nom si massa llarg
        return name[:12] + "\u2026" if len(name) > 12 else name

    @staticmethod
    def _setup_bar_hover(figure, canvas, ax, x_positions, full_names, values_per_bar):
        """Configura tooltip hover sobre barres del grafic.

        values_per_bar: list of total values per bar position.
        """
        annot = ax.annotate("", xy=(0, 0), xytext=(0, 8),
                            textcoords="offset points", fontsize=7,
                            fontfamily=_CHART_FONT,
                            ha='center', va='bottom',
                            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                      ec="#ccc", alpha=0.95))
        annot.set_visible(False)

        def on_move(event):
            if event.inaxes != ax:
                if annot.get_visible():
                    annot.set_visible(False)
                    canvas.draw_idle()
                return
            # Trobar la barra mes propera
            for i, xp in enumerate(x_positions):
                if abs(event.xdata - xp) < 0.4:
                    val = values_per_bar[i] if i < len(values_per_bar) else 0
                    annot.xy = (xp, val)
                    annot.set_text(f"{full_names[i]}\n{val:.0f}")
                    annot.set_visible(True)
                    canvas.draw_idle()
                    return
            if annot.get_visible():
                annot.set_visible(False)
                canvas.draw_idle()

        figure._hover_cid = figure.canvas.mpl_connect('motion_notify_event', on_move)

    def _plot_doc_chart(self, regular, light, is_bp):
        """Grafic DOC: barres stacked per fraccions (COLUMN) o simples (BP)."""
        self.doc_figure.clear()
        ax = self.doc_figure.add_subplot(111)

        names = []
        labels = []
        fractions_data = {f: [] for f in FRACTION_ORDER}
        ppm_values = []

        for name in sorted(regular.keys()):
            data = regular[name]
            sel = (data.get("selected") or {}).get("doc", "1")
            rep = (data.get("replicas") or {}).get(sel, {})
            areas = (rep.get("areas") or {}).get("DOC", {})
            quant = data.get("quantification") or {}

            names.append(name)
            labels.append(self._chart_short_label(name, data))
            # BP no te fraccions -- usar total directament
            area_total = areas.get("total", 0) or 0
            has_fracs = any(areas.get(f, 0) for f in FRACTION_ORDER)
            if has_fracs:
                for frac in FRACTION_ORDER:
                    fractions_data[frac].append(areas.get(frac, 0) or 0)
            else:
                for frac in FRACTION_ORDER:
                    fractions_data[frac].append(0)
                fractions_data["BioP"][-1] = area_total
            ppm_values.append(quant.get("concentration_ppm") or 0)

        light_start = len(names)
        for name in sorted(light.keys()):
            data = light[name]
            sel = (data.get("selected") or {}).get("doc", "1")
            rep = (data.get("replicas") or {}).get(sel, {})
            area = rep.get("area_total", 0)

            names.append(name)
            labels.append(self._chart_short_label(name, data))
            for frac in FRACTION_ORDER:
                fractions_data[frac].append(0)
            ppm_values.append(0)
            fractions_data["BioP"][-1] = area

        if not names:
            ax.text(0.5, 0.5, "Sense dades", ha='center', va='center', transform=ax.transAxes)
            self.doc_canvas.draw()
            return

        x = np.arange(len(names))
        bar_width = 0.7
        totals = [sum(fractions_data[f][i] for f in FRACTION_ORDER) for i in range(len(names))]

        if is_bp:
            colors = ['#95a5a6' if i >= light_start else '#3498DB' for i in range(len(names))]
            ax.bar(x, totals, bar_width, color=colors, edgecolor='white', linewidth=0.5)
        else:
            bottom = np.zeros(len(names))
            for frac in FRACTION_ORDER:
                values = np.array(fractions_data[frac], dtype=float)
                colors = []
                for i in range(len(names)):
                    if i >= light_start:
                        colors.append('#B0B0B0')
                    else:
                        colors.append(FRACTION_COLORS[frac])
                ax.bar(x, values, bar_width, bottom=bottom, color=colors,
                       edgecolor='white', linewidth=0.3, label=frac)
                bottom += values

            handles = [
                ax.bar(0, 0, color=FRACTION_COLORS[f], label=f)[0]
                for f in FRACTION_ORDER
            ]
            ax.legend(handles=handles, labels=FRACTION_ORDER,
                      loc='lower center', bbox_to_anchor=(0.5, 1.01),
                      fontsize=_CHART_TICK_SIZE, framealpha=0.9,
                      ncol=len(FRACTION_ORDER), borderaxespad=0,
                      handlelength=1.2, columnspacing=0.8,
                      prop={'family': _CHART_FONT})

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, ha='center',
                           fontsize=_CHART_TICK_SIZE, fontfamily=_CHART_FONT)
        ax.set_ylabel("DOC", fontsize=_CHART_LABEL_SIZE,
                       fontfamily=_CHART_FONT)
        ax.set_title("DOC per mostra", fontsize=_CHART_TITLE_SIZE,
                      fontweight='bold', fontfamily=_CHART_FONT, pad=16)
        ax.tick_params(axis='y', labelsize=_CHART_TICK_SIZE)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        self._setup_bar_hover(self.doc_figure, self.doc_canvas, ax, x, names, totals)
        self.doc_figure.tight_layout()
        self.doc_canvas.draw()

    def _plot_dad_chart(self, regular, light):
        """Grafic DAD per mostra: barres stacked per fraccions (COLUMN) o simples (BP)."""
        self.dad_figure.clear()
        ax = self.dad_figure.add_subplot(111)
        wl = self._get_selected_wl()
        wl_key = f"A{wl}"
        is_bp = getattr(self, '_chart_is_bp', False)

        names = []
        labels = []
        fractions_data = {f: [] for f in FRACTION_ORDER}

        for name in sorted(regular.keys()):
            data = regular[name]
            selected = data.get("selected") or {}
            sel = selected.get("dad", selected.get("doc", "1"))
            rep = (data.get("replicas") or {}).get(sel, {})
            areas = (rep.get("areas") or {}).get(wl_key, {})
            names.append(name)
            labels.append(self._chart_short_label(name, data))
            # BP no te fraccions -- usar total directament
            area_total = areas.get("total", 0) or 0
            has_fracs = any(areas.get(f, 0) for f in FRACTION_ORDER)
            if has_fracs:
                for frac in FRACTION_ORDER:
                    fractions_data[frac].append(areas.get(frac, 0) or 0)
            else:
                for frac in FRACTION_ORDER:
                    fractions_data[frac].append(0)
                fractions_data["BioP"][-1] = area_total

        light_start = len(names)
        for name in sorted(light.keys()):
            data = light[name]
            sel = (data.get("selected") or {}).get("doc", "1")
            rep = (data.get("replicas") or {}).get(sel, {})
            area = ((rep.get("areas") or {}).get(wl_key) or {}).get("total", 0)
            names.append(name)
            labels.append(self._chart_short_label(name, data))
            for frac in FRACTION_ORDER:
                fractions_data[frac].append(0)
            fractions_data["BioP"][-1] = area

        if not names:
            ax.text(0.5, 0.5, "Sense dades", ha='center', va='center', transform=ax.transAxes)
            self.dad_canvas.draw()
            return

        x = np.arange(len(names))
        bar_width = 0.7
        totals = [sum(fractions_data[f][i] for f in FRACTION_ORDER) for i in range(len(names))]

        if is_bp:
            colors = ['#95a5a6' if i >= light_start else '#E74C3C' for i in range(len(names))]
            ax.bar(x, totals, bar_width, color=colors, edgecolor='white', linewidth=0.5)
        else:
            bottom = np.zeros(len(names))
            for frac in FRACTION_ORDER:
                values = np.array(fractions_data[frac], dtype=float)
                colors = []
                for i in range(len(names)):
                    if i >= light_start:
                        colors.append('#B0B0B0')
                    else:
                        colors.append(FRACTION_COLORS[frac])
                ax.bar(x, values, bar_width, bottom=bottom, color=colors,
                       edgecolor='white', linewidth=0.3, label=frac)
                bottom += values

            handles = [
                ax.bar(0, 0, color=FRACTION_COLORS[f], label=f)[0]
                for f in FRACTION_ORDER
            ]
            ax.legend(handles=handles, labels=FRACTION_ORDER,
                      loc='lower center', bbox_to_anchor=(0.5, 1.01),
                      fontsize=_CHART_TICK_SIZE, framealpha=0.9,
                      ncol=len(FRACTION_ORDER), borderaxespad=0,
                      handlelength=1.2, columnspacing=0.8,
                      prop={'family': _CHART_FONT})

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, ha='center',
                           fontsize=_CHART_TICK_SIZE, fontfamily=_CHART_FONT)
        ax.set_ylabel(wl_key, fontsize=_CHART_LABEL_SIZE,
                       fontfamily=_CHART_FONT)
        ax.set_title(f"{wl_key} per mostra", fontsize=_CHART_TITLE_SIZE,
                      fontweight='bold', fontfamily=_CHART_FONT, pad=16)
        ax.tick_params(axis='y', labelsize=_CHART_TICK_SIZE)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        self._setup_bar_hover(self.dad_figure, self.dad_canvas, ax, x, names, totals)
        self.dad_figure.tight_layout()
        self.dad_canvas.draw()

    @staticmethod
    def _get_line_style(data):
        """Retorna l'estil de linia segons el tipus de mostra."""
        st = data.get("sample_type", "SAMPLE")
        if st == "KHP":
            return '--'  # KHP: discontinua
        elif st in ("BLANK", "CONTROL"):
            return ':'   # Blanc/Control: punts
        elif st.startswith("PR"):
            return '-.'  # PR: punt-ratlla
        return '-'       # Mostres: solida

    def _draw_doc_overlay_on_ax(self, ax, all_samples, is_bp, is_popup=False):
        """Dibuixa cromatogrames DOC superposats sobre un ax donat."""
        n = len(all_samples)
        cmap = cm.get_cmap('tab20', max(n, 1))
        lw = 1.2 if is_popup else 0.9

        for i, (name, data) in enumerate(sorted(all_samples.items())):
            sel = (data.get("selected") or {}).get("doc", "1")
            rep = (data.get("replicas") or {}).get(sel, {})
            t = rep.get("t_doc")
            y = rep.get("y_doc_net")
            ls = self._get_line_style(data)
            short = self._chart_short_label(name, data)
            if t is not None and y is not None and len(t) > 0:
                ax.plot(t, y, label=short, linewidth=lw, alpha=0.75,
                        color=cmap(i), linestyle=ls)

        ax.set_xlim(0, 12 if is_bp else 70)
        ax.set_xlabel("Temps (min)", fontsize=_CHART_LABEL_SIZE,
                       fontfamily=_CHART_FONT)
        ax.set_ylabel("DOC (ppb)", fontsize=_CHART_LABEL_SIZE,
                       fontfamily=_CHART_FONT)
        ax.set_title("DOC superposats", fontsize=_CHART_TITLE_SIZE,
                      fontweight='bold', fontfamily=_CHART_FONT)

    def _draw_dad_overlay_on_ax(self, ax, all_samples, is_bp, wl,
                                 is_popup=False):
        """Dibuixa cromatogrames DAD superposats sobre un ax donat."""
        wl_key = f"A{wl}"
        n = len(all_samples)
        cmap = cm.get_cmap('tab20', max(n, 1))
        lw = 1.2 if is_popup else 0.9

        for i, (name, data) in enumerate(sorted(all_samples.items())):
            selected = data.get("selected") or {}
            sel = selected.get("dad", selected.get("doc", "1"))
            rep = (data.get("replicas") or {}).get(sel, {})
            df_dad = rep.get("df_dad")
            if df_dad is None:
                continue
            try:
                if df_dad.empty:
                    continue
            except AttributeError:
                continue

            t_col = None
            for c in df_dad.columns:
                if 'time' in str(c).lower():
                    t_col = c
                    break
            wl_col = None
            for c in df_dad.columns:
                if wl in str(c):
                    wl_col = c
                    break

            ls = self._get_line_style(data)
            short = self._chart_short_label(name, data)
            if t_col is not None and wl_col is not None:
                ax.plot(df_dad[t_col], df_dad[wl_col], label=short,
                        linewidth=lw, alpha=0.75, color=cmap(i), linestyle=ls)

        ax.set_xlim(0, 12 if is_bp else 70)
        ax.set_xlabel("Temps (min)", fontsize=_CHART_LABEL_SIZE,
                       fontfamily=_CHART_FONT)
        ax.set_ylabel(f"{wl_key} (mAU)", fontsize=_CHART_LABEL_SIZE,
                       fontfamily=_CHART_FONT)
        ax.set_title(f"{wl_key} superposats", fontsize=_CHART_TITLE_SIZE,
                      fontweight='bold', fontfamily=_CHART_FONT)

    def _plot_doc_overlay(self, regular, light, is_bp):
        """Miniatura DOC overlay (sense llegenda)."""
        self.doc_overlay_figure.clear()
        ax = self.doc_overlay_figure.add_subplot(111)
        all_samples = {**regular, **light}
        if not all_samples:
            ax.text(0.5, 0.5, "Sense dades", ha='center', va='center',
                    transform=ax.transAxes, color='#bbb',
                    fontsize=_CHART_LABEL_SIZE)
            self.doc_overlay_canvas.draw()
            return
        self._draw_doc_overlay_on_ax(ax, all_samples, is_bp)
        ax.tick_params(axis='both', labelsize=_CHART_TICK_SIZE)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Miniatura: sense llegenda, hint visual
        n = len(all_samples)
        ax.text(0.98, 0.02, f"{n} traces",
                transform=ax.transAxes, fontsize=7, ha='right', va='bottom',
                color='#aaa', fontfamily=_CHART_FONT)
        self.doc_overlay_figure.tight_layout(pad=0.5)
        self.doc_overlay_canvas.draw()

    def _plot_dad_overlay(self, regular, light):
        """Miniatura DAD overlay (sense llegenda)."""
        self.dad_overlay_figure.clear()
        ax = self.dad_overlay_figure.add_subplot(111)
        wl = self._get_selected_wl()
        all_samples = {**regular, **light}
        if not all_samples:
            ax.text(0.5, 0.5, "Sense dades", ha='center', va='center',
                    transform=ax.transAxes, color='#bbb',
                    fontsize=_CHART_LABEL_SIZE)
            self.dad_overlay_canvas.draw()
            return
        is_bp = getattr(self, '_chart_is_bp', False)
        self._draw_dad_overlay_on_ax(ax, all_samples, is_bp, wl)
        ax.tick_params(axis='both', labelsize=_CHART_TICK_SIZE)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        n = len(all_samples)
        ax.text(0.98, 0.02, f"{n} traces",
                transform=ax.transAxes, fontsize=7, ha='right', va='bottom',
                color='#aaa', fontfamily=_CHART_FONT)
        self.dad_overlay_figure.tight_layout(pad=0.5)
        self.dad_overlay_canvas.draw()

    def _open_overlay_popup(self, chart_type):
        """Obre pop-up interactiu per l'overlay DOC o DAD."""
        try:
            if not HAS_MATPLOTLIB:
                return
            checked = self._get_checked_samples()
            reg = {k: v for k, v in checked.items()
                   if v.get("sample_type") != "CONTROL"}
            light = {k: v for k, v in checked.items()
                     if v.get("sample_type") == "CONTROL"}
            all_samples = {**reg, **light}
            if not all_samples:
                return
            is_bp = getattr(self, '_chart_is_bp', False)

            if chart_type == "doc":
                title = "Cromatogrames DOC superposats"
                def plot_fn(ax):
                    self._draw_doc_overlay_on_ax(ax, all_samples, is_bp,
                                                  is_popup=True)
            else:
                wl = self._get_selected_wl()
                title = f"Cromatogrames A{wl} superposats"
                def plot_fn(ax):
                    self._draw_dad_overlay_on_ax(ax, all_samples, is_bp, wl,
                                                  is_popup=True)

            dlg = OverlayPopupDialog(self, title, plot_fn)
            dlg.exec()
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"Error opening overlay popup: {e}")
            QMessageBox.critical(self, "Error", f"Error obrint popup:\n{e}")

    def save_charts(self, seq_path):
        """Guarda els 5 grafics a SEQ/CHECK/plots/."""
        if not HAS_MATPLOTLIB or not seq_path:
            return
        try:
            plots_dir = Path(seq_path) / "CHECK" / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)

            for name, fig in [
                ("timeout_zones.png", self.timeout_figure),
                ("doc_areas.png", self.doc_figure),
                ("doc_overlay.png", self.doc_overlay_figure),
                ("dad_areas.png", self.dad_figure),
                ("dad_overlay.png", self.dad_overlay_figure),
            ]:
                fig.savefig(
                    str(plots_dir / name), dpi=150, bbox_inches='tight',
                    facecolor='#FAFAFA', edgecolor='none',
                )

            logger.info(f"Charts saved to {plots_dir}")
        except Exception as e:
            logger.error(f"Error saving charts: {e}")
