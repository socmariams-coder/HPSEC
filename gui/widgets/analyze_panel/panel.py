"""
HPSEC Suite - Analyze Panel (Fase 3) -- Table View
====================================================

Panel per la fase 3: Analisi de mostres.
- Single scrollable view (NO tabs)
- Selector row: DOC/DAD global replicas + category buttons
- QC miniatures row (collapsible)
- Charts section: DOC/DAD bars + overlays with category buttons
- Stats bar with summary counts
- 10-column table with DOC/DAD group headers
- Comparison COL<->BP collapsible section at bottom
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QHeaderView, QComboBox, QTableWidget, QTableWidgetItem,
    QFrame, QAbstractItemView, QProgressBar, QMessageBox, QDialog,
    QGroupBox, QGridLayout, QCheckBox, QScrollArea, QSizePolicy
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

# --- Estil global grafics ---
_CHART_FONT = "Segoe UI"
_CHART_BG = "#FAFAFA"
_CHART_TITLE_SIZE = 9
_CHART_LABEL_SIZE = 8
_CHART_TICK_SIZE = 7

from hpsec_analyze import analyze_sequence, save_analysis_result, load_analysis_result
from gui.widgets.styles import (
    COLOR_SUCCESS, COLOR_WARNING, COLOR_ERROR,
    apply_panel_layout, create_empty_state_widget
)
from .worker import AnalyzeWorker, SiblingAnalyzeWorker
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
    configure_table_style, populate_signal_summary, populate_fractions_table,
    classify_sample_status, resolve_doc_replica, find_repair_targets,
)
from .sequence_qc_tab import SequenceQCTab
# ComparisonTab moved to tab Mostres

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
            """Llegenda clicable: toggle visibilitat de cada traca."""
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


class AnalyzePanel(QWidget):
    """Panel d'analisi de mostres (Fase 3) -- Table View."""

    analyze_completed = Signal(dict)

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.samples_grouped = {}
        self.worker = None
        self._warnings_confirmed = False
        self._warnings_confirmed_by = ""
        self._selected_sample = None
        self._review_sample = None
        self._sample_row_map = {}       # sample_name -> row index
        self._row_sample_map = {}       # row index -> sample_name (reverse)
        self._status_initialized = False
        # Chart data
        self._chart_regular = {}
        self._chart_blank = {}
        self._chart_control = {}
        self._chart_khp = {}
        self._chart_is_bp = False
        # Siblings
        self._sibling_worker = None
        self._sibling_results = {}

        self._setup_ui()

    # ------------------------------------------------------------------
    # UI Setup
    # ------------------------------------------------------------------

    def _setup_ui(self):
        """Configura la interficie -- Single scrollable view amb taula."""
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        # Boto analitzar (amagat - l'accio es dispara des del wizard header)
        self.analyze_btn = QPushButton()
        self.analyze_btn.setVisible(False)
        self.analyze_btn.clicked.connect(self._run_analyze)

        # === SCROLL AREA per contenir tot el contingut ===
        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setFrameShape(QFrame.NoFrame)
        self._scroll_area.setStyleSheet("QScrollArea { border: none; }")

        scroll_content = QWidget()
        layout = QVBoxLayout(scroll_content)
        apply_panel_layout(layout)

        # === INFO PANEL (amagat) ===
        self.info_frame = QFrame()
        self.info_frame.setVisible(False)
        info_layout = QHBoxLayout(self.info_frame)
        self.import_info = QLabel()
        info_layout.addWidget(self.import_info)
        self.cal_info = QLabel()
        info_layout.addWidget(self.cal_info)
        self.status_indicator = QLabel()
        info_layout.addWidget(self.status_indicator)

        # Empty state
        self.empty_state = create_empty_state_widget(
            "\U0001f52c", "Preparant analisi...",
            "Carregant dades de la sequencia."
        )
        self.empty_state.setVisible(False)
        layout.addWidget(self.empty_state)

        # Status frame (mantingut per backward compat)
        self.status_frame = QFrame()
        self.status_frame.setVisible(False)
        status_layout = QVBoxLayout(self.status_frame)
        self.status_label = QLabel()
        status_layout.addWidget(self.status_label)

        # === PROGRESS ===
        self.progress_frame = QFrame()
        self.progress_frame.setVisible(False)
        progress_layout = QVBoxLayout(self.progress_frame)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        self.progress_bar = QProgressBar()
        self.progress_label = QLabel("Preparant...")
        progress_layout.addWidget(self.progress_label)
        progress_layout.addWidget(self.progress_bar)
        layout.addWidget(self.progress_frame)

        # === RESULTS FRAME ===
        self.results_frame = QFrame()
        self.results_frame.setVisible(False)
        results_layout = QVBoxLayout(self.results_frame)
        results_layout.setContentsMargins(0, 0, 0, 0)
        results_layout.setSpacing(8)

        # === SELECTOR BAR ===
        sel_frame = QFrame()
        sel_frame.setStyleSheet(
            "QFrame { background: #fff; border: 1px solid #e0e0e0;"
            " border-radius: 6px; }"
        )
        sel_layout = QHBoxLayout(sel_frame)
        sel_layout.setContentsMargins(10, 6, 10, 6)
        sel_layout.setSpacing(6)

        self._cat_buttons = {}
        self._cat_counts = {}
        self._sample_checkboxes = []

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
        results_layout.addWidget(sel_frame)

        # === QC MINIATURES (collapsible) ===
        self._qc_collapsible = self._build_collapsible_section(
            "QC Sequencia", collapsed=True)
        self._qc_tab = SequenceQCTab(main_window=self.main_window)
        self._qc_collapsible["content_layout"].addWidget(self._qc_tab)
        results_layout.addWidget(self._qc_collapsible["frame"])

        # === CHARTS SECTION ===
        self._charts_visible = True
        self._charts_initialized = False
        self.charts_section = QFrame()
        self.charts_section.setVisible(False)
        charts_outer = QVBoxLayout(self.charts_section)
        charts_outer.setContentsMargins(0, 8, 0, 0)
        charts_outer.setSpacing(4)

        self._charts_content = QWidget()
        self._charts_content.setVisible(True)
        self._charts_content_layout = QVBoxLayout(self._charts_content)
        self._charts_content_layout.setContentsMargins(0, 4, 0, 0)
        self._charts_content_layout.setSpacing(4)

        if HAS_MATPLOTLIB:
            # DOC barres + DOC overlay miniatura
            doc_row = QHBoxLayout()
            doc_row.setSpacing(4)

            self.doc_figure = Figure(figsize=(5, 2.8), dpi=100)
            self.doc_figure.set_facecolor(_CHART_BG)
            self.doc_canvas = FigureCanvas(self.doc_figure)
            self.doc_canvas.setMinimumHeight(170)
            doc_row.addWidget(self.doc_canvas, 3)

            self.doc_overlay_figure = Figure(figsize=(3, 2.8), dpi=100)
            self.doc_overlay_figure.set_facecolor(_CHART_BG)
            self.doc_overlay_canvas = _ClickableCanvas(
                self.doc_overlay_figure,
                on_dblclick=lambda: self._open_overlay_popup("doc"))
            self.doc_overlay_canvas.setMinimumHeight(160)
            doc_row.addWidget(self.doc_overlay_canvas, 2)

            self._charts_content_layout.addLayout(doc_row)

            # DAD barres + DAD overlay miniatura
            dad_row = QHBoxLayout()
            dad_row.setSpacing(4)

            self.dad_figure = Figure(figsize=(5, 2.8), dpi=100)
            self.dad_figure.set_facecolor(_CHART_BG)
            self.dad_canvas = FigureCanvas(self.dad_figure)
            self.dad_canvas.setMinimumHeight(170)
            dad_row.addWidget(self.dad_canvas, 3)

            self.dad_overlay_figure = Figure(figsize=(3, 2.8), dpi=100)
            self.dad_overlay_figure.set_facecolor(_CHART_BG)
            self.dad_overlay_canvas = _ClickableCanvas(
                self.dad_overlay_figure,
                on_dblclick=lambda: self._open_overlay_popup("dad"))
            self.dad_overlay_canvas.setMinimumHeight(160)
            dad_row.addWidget(self.dad_overlay_canvas, 2)

            self._charts_content_layout.addLayout(dad_row)

        charts_outer.addWidget(self._charts_content)
        results_layout.addWidget(self.charts_section)

        # === STATS BAR ===
        self._stats_label = QLabel()
        self._stats_label.setStyleSheet(
            "font-size: 11px; color: #555; background: #F8F9FA;"
            " border: 1px solid #E0E0E0; border-radius: 4px;"
            " padding: 6px 10px;")
        results_layout.addWidget(self._stats_label)

        # === TABLE WITH GROUP HEADERS ===
        self._table_container = QWidget()
        self._table_container_layout = QVBoxLayout(self._table_container)
        self._table_container_layout.setContentsMargins(0, 0, 0, 0)
        self._table_container_layout.setSpacing(0)
        self._build_table_with_group_headers()
        results_layout.addWidget(self._table_container)

        # === REVIEW PANEL (shown on row click) ===
        self._review_panel = QFrame()
        self._review_panel.setVisible(False)
        self._review_panel.setStyleSheet(
            "QFrame { border: 1px solid #DEE2E6; border-radius: 6px;"
            " background: white; }")
        review_layout = QVBoxLayout(self._review_panel)
        review_layout.setContentsMargins(8, 8, 8, 8)
        review_layout.setSpacing(6)

        # Navigation + action row (tot a dalt)
        nav_row = QHBoxLayout()
        nav_row.setSpacing(4)

        nav_btn_style = ("QPushButton { border: 1px solid #CED4DA; border-radius: 3px;"
                         " padding: 4px 10px; font-size: 11px; }"
                         "QPushButton:hover { background: #E9ECEF; }")
        action_btn_style = ("QPushButton { border: 1px solid {c}; border-radius: 3px;"
                            " padding: 4px 10px; font-size: 11px; color: {c}; }}"
                            "QPushButton:hover {{ background: {bg}; }}")

        self._review_prev_btn = QPushButton("\u25c0")
        self._review_prev_btn.setStyleSheet(nav_btn_style)
        self._review_prev_btn.setFixedWidth(32)
        self._review_prev_btn.setToolTip("Mostra anterior")
        self._review_prev_btn.clicked.connect(lambda: self._navigate_review(-1))
        nav_row.addWidget(self._review_prev_btn)

        self._review_repair_btn = QPushButton("Reparar pic")
        self._review_repair_btn.setStyleSheet(
            "QPushButton { border: 1px solid #E67E22; border-radius: 3px;"
            " padding: 4px 10px; font-size: 11px; color: #E67E22; }"
            "QPushButton:hover { background: #FEF9E7; }")
        self._review_repair_btn.clicked.connect(self._on_review_repair)
        nav_row.addWidget(self._review_repair_btn)

        self._review_compose_btn = QPushButton("Composar")
        self._review_compose_btn.setStyleSheet(
            "QPushButton { border: 1px solid #3498DB; border-radius: 3px;"
            " padding: 4px 10px; font-size: 11px; color: #3498DB; }"
            "QPushButton:hover { background: #EBF5FB; }")
        self._review_compose_btn.clicked.connect(self._on_review_compose)
        nav_row.addWidget(self._review_compose_btn)

        self._review_title = QLabel()
        self._review_title.setAlignment(Qt.AlignCenter)
        self._review_title.setStyleSheet("font-size: 12px; font-weight: bold;")
        nav_row.addWidget(self._review_title, 1)

        self._review_detail_btn = QPushButton("Detall")
        self._review_detail_btn.setStyleSheet(nav_btn_style)
        self._review_detail_btn.setToolTip("Obrir dialeg detall complet")
        self._review_detail_btn.clicked.connect(self._on_review_detail)
        nav_row.addWidget(self._review_detail_btn)

        self._review_close_btn = QPushButton("\u2715")
        self._review_close_btn.setStyleSheet(nav_btn_style)
        self._review_close_btn.setFixedWidth(28)
        self._review_close_btn.setToolTip("Tancar panel revisio")
        self._review_close_btn.clicked.connect(self._close_review)
        nav_row.addWidget(self._review_close_btn)

        self._review_next_btn = QPushButton("\u25b6")
        self._review_next_btn.setStyleSheet(nav_btn_style)
        self._review_next_btn.setFixedWidth(32)
        self._review_next_btn.setToolTip("Mostra seguent")
        self._review_next_btn.clicked.connect(lambda: self._navigate_review(1))
        nav_row.addWidget(self._review_next_btn)

        review_layout.addLayout(nav_row)

        # Chromatogram
        if HAS_MATPLOTLIB:
            self._review_figure = Figure(figsize=(8, 3), dpi=100)
            self._review_figure.set_facecolor("#FAFAFA")
            self._review_canvas = FigureCanvas(self._review_figure)
            self._review_canvas.setMinimumHeight(250)
            self._review_toolbar = NavigationToolbar2QT(
                self._review_canvas, self._review_panel)
            review_layout.addWidget(self._review_toolbar)
            review_layout.addWidget(self._review_canvas)

        # Controls row
        controls_row = QHBoxLayout()
        controls_row.addWidget(QLabel("<b>DOC:</b>"))
        self._review_doc_combo = QComboBox()
        self._review_doc_combo.setMinimumWidth(100)
        self._review_doc_combo.currentIndexChanged.connect(
            self._on_review_doc_changed)
        controls_row.addWidget(self._review_doc_combo)
        controls_row.addWidget(QLabel("<b>DAD:</b>"))
        self._review_dad_combo = QComboBox()
        self._review_dad_combo.setMinimumWidth(100)
        self._review_dad_combo.currentIndexChanged.connect(
            self._on_review_dad_changed)
        controls_row.addWidget(self._review_dad_combo)
        self._review_show_area = QCheckBox("Area")
        self._review_show_area.setStyleSheet("font-size: 10px;")
        self._review_show_area.setChecked(True)
        self._review_show_area.setToolTip("Mostrar/amagar ombrejat area integracio")
        self._review_show_area.toggled.connect(self._on_review_area_toggled)
        controls_row.addWidget(self._review_show_area)

        controls_row.addStretch()
        self._review_metrics = QLabel()
        self._review_metrics.setStyleSheet("font-size: 11px; color: #444;")
        controls_row.addWidget(self._review_metrics)
        review_layout.addLayout(controls_row)

        # Fractions + anomalies row
        info_row = QHBoxLayout()
        self._review_fractions = QLabel()
        self._review_fractions.setStyleSheet("font-size: 11px; color: #555;")
        info_row.addWidget(self._review_fractions, 1)
        self._review_anomalies = QLabel()
        self._review_anomalies.setStyleSheet("font-size: 11px;")
        self._review_anomalies.setWordWrap(True)
        info_row.addWidget(self._review_anomalies, 1)
        review_layout.addLayout(info_row)

        # (Action buttons moved to nav_row at top)

        results_layout.addWidget(self._review_panel)

        layout.addWidget(self.results_frame, 1)

        # Completar scroll area
        self._scroll_area.setWidget(scroll_content)
        outer_layout.addWidget(self._scroll_area, 1)

    def _build_collapsible_section(self, title, collapsed=True):
        """Build a collapsible section with header button and content area.

        Returns dict with 'frame', 'toggle_btn', 'content', 'content_layout'.
        """
        frame = QFrame()
        frame.setStyleSheet(
            "QFrame { border: 1px solid #E0E0E0; border-radius: 4px; }")
        frame_layout = QVBoxLayout(frame)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        frame_layout.setSpacing(0)

        toggle_btn = QPushButton(
            f"\u25b6 {title}" if collapsed else f"\u25bc {title}")
        toggle_btn.setStyleSheet(
            "QPushButton { border: none; text-align: left; padding: 6px 10px;"
            " font-size: 11px; font-weight: bold; color: #555;"
            " background: #F5F5F5; border-radius: 4px; }"
            "QPushButton:hover { background: #E8E8E8; }")
        frame_layout.addWidget(toggle_btn)

        content = QWidget()
        content.setVisible(not collapsed)
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(4, 4, 4, 4)
        content_layout.setSpacing(4)
        frame_layout.addWidget(content)

        def _toggle():
            vis = not content.isVisible()
            content.setVisible(vis)
            toggle_btn.setText(
                f"\u25bc {title}" if vis else f"\u25b6 {title}")

        toggle_btn.clicked.connect(_toggle)

        return {
            "frame": frame,
            "toggle_btn": toggle_btn,
            "content": content,
            "content_layout": content_layout,
        }

    # ------------------------------------------------------------------
    # Show / Reset / Check existing
    # ------------------------------------------------------------------

    def showEvent(self, event):
        super().showEvent(event)
        if not self._status_initialized or not self.samples_grouped:
            self._check_existing_analysis()
            self._update_status()
            self._status_initialized = True

    def reset(self):
        """Reinicia el panel al seu estat inicial."""
        self.samples_grouped = {}
        self.worker = None
        self._warnings_confirmed = False
        self._warnings_confirmed_by = ""
        self._selected_sample = None
        self._review_sample = None
        self._sample_row_map = {}
        self._row_sample_map = {}
        self._status_initialized = False
        self._chart_regular = {}
        self._chart_blank = {}
        self._chart_control = {}
        self._chart_khp = {}
        self._chart_is_bp = False
        self._charts_initialized = False
        self._sibling_worker = None
        self._sibling_results = {}

        # Clear table
        self._samples_table.setRowCount(0)

        self.empty_state.setVisible(True)
        self.info_frame.setVisible(False)
        self.status_frame.setVisible(False)
        self.progress_frame.setVisible(False)
        self.progress_bar.setValue(0)
        self.results_frame.setVisible(False)
        self._review_panel.setVisible(False)
        self.charts_section.setVisible(False)
        self._qc_tab.reset()
        # comparison moved to tab Mostres
        self._charts_content.setVisible(True)
        self.analyze_btn.setEnabled(True)
        self.status_indicator.setText("")

    def _build_table_with_group_headers(self):
        """Creates the sample table with DOC/DAD group header labels above."""
        # --- Group header row ---
        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(0)

        # Spacer for "Mostra" column (col 0) -- approximate width
        mostra_spacer = QLabel("")
        mostra_spacer.setMinimumWidth(140)
        mostra_spacer.setMaximumWidth(200)
        header_row.addWidget(mostra_spacer)

        doc_label = QLabel("DOC")
        doc_label.setAlignment(Qt.AlignCenter)
        doc_label.setStyleSheet(
            "font-weight: bold; font-size: 11px; color: #1A5276;"
            " background: #EBF5FB; border: 1px solid #D4E6F1;"
            " border-radius: 3px; padding: 3px 0; margin: 0 1px;")
        header_row.addWidget(doc_label, 6)  # spans 6 columns worth

        dad_label = QLabel("DAD")
        dad_label.setAlignment(Qt.AlignCenter)
        dad_label.setStyleSheet(
            "font-weight: bold; font-size: 11px; color: #7D6608;"
            " background: #FEF9E7; border: 1px solid #F9E79F;"
            " border-radius: 3px; padding: 3px 0; margin: 0 1px;")
        header_row.addWidget(dad_label, 3)  # spans 3 columns worth

        # Small spacer for scrollbar area
        sb_spacer = QLabel("")
        sb_spacer.setFixedWidth(16)
        header_row.addWidget(sb_spacer)

        self._table_container_layout.addLayout(header_row)

        # --- Table ---
        self._samples_table = QTableWidget()
        self._samples_table.setColumnCount(10)
        self._samples_table.setHorizontalHeaderLabels([
            "Mostra",
            "ppm", "ppm\u1d64\u1d62\u1d47", "SNR", "r\u00b2",
            "Timeout", "Pic",
            "A254", "SNR\u2082\u2085\u2084", "r\u00b2\u2082\u2085\u2084",
        ])

        configure_table_style(self._samples_table)
        self._samples_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._samples_table.setSelectionMode(QAbstractItemView.SingleSelection)

        # Column sizing
        header = self._samples_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        for col in range(1, 10):
            header.setSectionResizeMode(col, QHeaderView.ResizeToContents)

        # Minimum column widths for readability
        self._samples_table.setColumnWidth(0, 160)
        for col in range(1, 10):
            self._samples_table.setColumnWidth(col, 60)

        self._samples_table.setMinimumHeight(200)
        self._samples_table.clicked.connect(self._on_table_row_clicked)
        self._table_container_layout.addWidget(self._samples_table)

    def _check_existing_analysis(self):
        """Comprova si existeix analisi previa i la carrega automaticament."""
        seq_path = self.main_window.seq_path
        if not seq_path:
            return
        if self.samples_grouped:
            return

        sibling_paths = getattr(self.main_window, 'sibling_paths', [])
        all_paths = [seq_path] + sibling_paths

        if len(all_paths) > 1:
            results = {}
            for path in all_paths:
                try:
                    r = load_analysis_result(path)
                    if r and r.get("success"):
                        results[path] = r
                except Exception as e:
                    logger.warning("Error carregant analisi %s: %s", path, e)

            if results:
                merged = self._merge_sibling_samples(results)
                base = results.get(seq_path) or next(iter(results.values()))
                base_result = dict(base)
                base_result["samples_grouped"] = merged
                base_result["is_sibling_merge"] = True
                base_result["sibling_results"] = results
                self._load_existing_analysis(base_result)
        else:
            try:
                existing_analysis = load_analysis_result(seq_path)
                if existing_analysis and existing_analysis.get("success"):
                    self._load_existing_analysis(existing_analysis)
            except Exception as e:
                logger.warning(f"Error comprovant analisi existent: {e}")

    def _load_existing_analysis(self, result):
        """Carrega una analisi existent."""
        self.samples_grouped = (result.get("samples_grouped")
                                or result.get("samples_analyzed", {}))
        if self.samples_grouped:
            if self.main_window.seq_path:
                result["seq_path"] = self.main_window.seq_path
            self.main_window.processed_data = result
            try:
                self._populate_table()
                self._populate_charts(result)
                self._populate_sub_tabs(result)
            except Exception as e:
                import traceback
                traceback.print_exc()
                logger.error(f"Error loading existing analysis UI: {e}")
            self.empty_state.setVisible(False)
            self.info_frame.setVisible(False)
            self.status_frame.setVisible(False)
            self.results_frame.setVisible(True)
            self.main_window.set_status("Analisi carregada des de fitxer existent", 3000)
            self.analyze_completed.emit(result)

    def _populate_sub_tabs(self, result):
        """Propaga dades d'analisi als sub-tabs QC i Comparacio."""
        try:
            self._qc_tab.populate(result)
        except Exception as e:
            logger.warning(f"Error populating QC tab: {e}")
        # Comparison COL↔BP moved to tab Mostres

    def _update_status(self):
        """Actualitza l'indicador d'estat amb format professional."""
        imported_data = self.main_window.imported_data
        calibration_data = self.main_window.calibration_data

        if not imported_data:
            self.info_frame.setVisible(False)
            self.status_frame.setVisible(False)
            self.empty_state.setVisible(True)
            self.analyze_btn.setEnabled(False)
            return

        self.empty_state.setVisible(False)
        self.info_frame.setVisible(False)
        self.status_frame.setVisible(False)

        # Use analyzed sample count if available, else imported injections
        if self.samples_grouped:
            n_items = len(self.samples_grouped)
            item_label = "mostres"
        else:
            samples = imported_data.get("samples", {})
            n_items = len(samples)
            item_label = "injeccions"
        method = imported_data.get("method", "-")
        data_mode = imported_data.get("data_mode", "-")

        self.import_info.setText(
            f"<span style='color: #6c757d; font-size: 10px;'>DADES</span><br>"
            f"<b style='font-size: 13px;'>{n_items}</b> <span style='color: #495057;'>{item_label}</span><br>"
            f"<span style='color: #6c757d; font-size: 10px;'>{method} / {data_mode}</span>"
        )

        if calibration_data and calibration_data.get("success"):
            rf_mass_global = None
            intercept_global = 0
            try:
                from hpsec_calibrate import get_rf_mass_cal, get_calibration_intercept
                seq_method = imported_data.get("method", "COLUMN")
                mode_key = "bp" if seq_method.upper() == "BP" else "column"
                rf_mass_global = get_rf_mass_cal(signal='direct', mode=mode_key)
                intercept_global = get_calibration_intercept(signal='direct', mode=mode_key) or 0
            except Exception:
                pass

            if rf_mass_global and rf_mass_global > 0:
                rf_display = rf_mass_global
                intercept_display = intercept_global
                cal_note = "Global"
            else:
                rf_mass_local = calibration_data.get("rf_mass", 0)
                rf_direct = calibration_data.get("rf_direct", 0) or calibration_data.get("rf", 0)
                rf_display = rf_mass_local if rf_mass_local > 0 else rf_direct
                intercept_display = 0
                cal_note = "Local"

            if intercept_display and abs(intercept_display) > 0.01:
                recta_str = f"RF=<b>{rf_display:.1f}</b> \u00b7 b=<b>{intercept_display:.1f}</b>"
            else:
                recta_str = f"RF=<b>{rf_display:.1f}</b> \u00b7 origen"

            self.cal_info.setText(
                f"<span style='color: #6c757d; font-size: 10px;'>CALIBRACIO</span><br>"
                f"<span style='color: #27AE60;'>\u2713</span> <b style='font-size: 13px;'>{cal_note}</b><br>"
                f"<span style='color: #6c757d; font-size: 10px;'>{recta_str}</span>"
            )

            khp_conc = calibration_data.get("khp_conc", 0)
            shift = calibration_data.get("shift_direct", 0) or calibration_data.get("shift", 0)
            shift_sec = shift * 60 if shift else 0
            khp_source = calibration_data.get("khp_source", "LOCAL")
            rf_global_str = f"{rf_mass_global:.2f}" if rf_mass_global else "N/A"
            self.cal_info.setToolTip(
                f"Font: {khp_source}\n"
                f"Quantificacio: {cal_note}\n"
                f"Recta: ppm = (A - {intercept_display:.1f}) x 1000 / (RF x V)\n"
                f"RF_mass_cal (global): {rf_global_str}\n"
                f"Intercept (global): {intercept_global:.2f}\n"
                f"KHP SEQ: {khp_conc:g}ppm, shift={shift_sec:.1f}s"
            )
        else:
            self.cal_info.setText(
                f"<span style='color: #6c757d; font-size: 10px;'>CALIBRACIO</span><br>"
                f"<span style='color: #E67E22;'>\u26a0</span> <span style='color: #856404;'>No disponible</span><br>"
                f"<span style='color: #6c757d; font-size: 10px;'>S'usaran valors per defecte</span>"
            )
            self.cal_info.setToolTip("No hi ha calibracio disponible")

        if not self.samples_grouped:
            self.status_indicator.setText(
                f"<span style='background-color: #d4edda; color: #155724; "
                f"padding: 4px 12px; border-radius: 12px; font-size: 11px;'>"
                f"Llest per analitzar</span>"
            )
        self.analyze_btn.setEnabled(True)

    # ------------------------------------------------------------------
    # Run analysis
    # ------------------------------------------------------------------

    def _run_analyze(self):
        """Executa l'analisi."""
        sibling_imported = getattr(self.main_window, 'sibling_imported', {})
        sibling_calibrated = getattr(self.main_window, 'sibling_calibrated', {})
        logger.info("_run_analyze: sibling_imported keys=%s (n=%d)",
                     list(sibling_imported.keys()), len(sibling_imported))
        if len(sibling_imported) > 1:
            self._run_analyze_siblings(sibling_imported, sibling_calibrated)
            return

        imported_data = self.main_window.imported_data
        calibration_data = self.main_window.calibration_data
        seq_path = self.main_window.seq_path

        if not imported_data and seq_path:
            from hpsec_import import import_from_manifest
            self.main_window.set_status("Carregant dades d'importacio...")
            try:
                imported_data = import_from_manifest(seq_path)
            except Exception as e:
                logger.warning(f"Error carregant import: {e}")
                imported_data = None
            if imported_data and imported_data.get('success'):
                self.main_window.imported_data = imported_data

        if not calibration_data and seq_path:
            import json
            cal_path = Path(seq_path) / "CHECK" / "data" / "calibration_result.json"
            if cal_path.exists():
                self.main_window.set_status("Carregant dades de calibracio...")
                with open(cal_path, 'r', encoding='utf-8') as f:
                    cal_file = json.load(f)
                calibrations = cal_file.get("calibrations", [])
                if calibrations:
                    active_cal = None
                    for cal in calibrations:
                        if cal.get("is_active", False):
                            active_cal = cal
                            break
                    if not active_cal:
                        active_cal = calibrations[0]

                    area = active_cal.get("area", 0)
                    conc = active_cal.get("conc_ppm", 5)
                    volume = active_cal.get("volume_uL", 0)
                    rf = active_cal.get("rf", 0)
                    if rf == 0 and conc > 0 and area > 0:
                        rf = area / conc
                    rf_direct = active_cal.get("rf_direct", rf)
                    rf_uib = active_cal.get("rf_uib", 0)
                    rf_mass = active_cal.get("rf_mass", 0)

                    calibration_data = {
                        "success": True,
                        "mode": active_cal.get("mode", "DUAL"),
                        "rf_direct": rf_direct,
                        "rf_uib": rf_uib,
                        "rf": rf,
                        "rf_mass": rf_mass,
                        "shift_direct": active_cal.get("shift_direct") or active_cal.get("shift_min", 0),
                        "shift_uib": active_cal.get("shift_uib") or active_cal.get("shift_min_u", 0),
                        "khp_area_direct": area,
                        "khp_area_uib": active_cal.get("area_u", 0),
                        "khp_area": area,
                        "khp_conc": conc,
                        "volume_uL": volume,
                        "khp_source": active_cal.get("khp_source", "LOCAL"),
                        "calibration": active_cal,
                        "errors": [],
                        "loaded_from_json": True,
                    }
                    self.main_window.calibration_data = calibration_data

        if not imported_data:
            QMessageBox.warning(self, "Avis", "No s'han trobat dades d'importacio.")
            self.analyze_completed.emit({'success': False, 'errors': ["No hi ha dades d'importacio"]})
            return

        samples = imported_data.get("samples", {})
        if not samples:
            QMessageBox.warning(self, "Avis", "No s'han trobat mostres a les dades importades.")
            self.analyze_completed.emit({'success': False, 'errors': ["No hi ha mostres a les dades"]})
            return

        self.analyze_btn.setEnabled(False)
        self.empty_state.setVisible(False)
        self.status_frame.setVisible(False)
        self.progress_frame.setVisible(True)
        self.progress_bar.setValue(0)
        self.results_frame.setVisible(False)

        if self.worker is not None:
            self.worker.wait()
        self.worker = AnalyzeWorker(imported_data, calibration_data)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_progress(self, msg, pct):
        self.progress_label.setText(msg)
        self.progress_bar.setValue(pct)

    def _on_finished(self, result):
        """Gestiona la finalitzacio de l'analisi."""
        if self.worker is not None:
            self.worker.wait()
        self.progress_frame.setVisible(False)
        self.analyze_btn.setEnabled(True)

        if not result or not result.get("success"):
            error_msg = result.get("error", "Error desconegut") if result else "Resultat buit"
            self._show_inline_message(error_msg, level="error")
            self._update_status()
            self.analyze_completed.emit(result or {"success": False, "error": error_msg})
            return

        self.main_window.processed_data = result
        self.samples_grouped = result.get("samples_grouped", {})

        save_analysis_result(result)

        n_samples = len(self.samples_grouped)
        self.progress_label.setText(f"Preparant taula ({n_samples} mostres)...")
        self.progress_bar.setValue(95)
        self.progress_frame.setVisible(True)
        from PySide6.QtWidgets import QApplication
        QApplication.processEvents()

        self._populate_table()
        self.results_frame.setVisible(True)

        self.progress_label.setText("Generant grafics...")
        self.progress_bar.setValue(97)
        QApplication.processEvents()

        self._populate_charts(result)
        self._populate_sub_tabs(result)

        self.progress_frame.setVisible(False)
        self.status_frame.setVisible(False)
        self.analyze_completed.emit(result)

    def _on_error(self, error_msg):
        logger.error(f"Error durant analisi: {error_msg}")
        self.progress_frame.setVisible(False)
        self.analyze_btn.setEnabled(True)
        self._show_inline_message(str(error_msg), level="error")
        self.analyze_completed.emit({"success": False, "error": error_msg})

    def _show_inline_message(self, message, level="info"):
        """Mostra un missatge inline al panell (error/warning/info)."""
        colors = {
            "error": ("background: #FADBD8; border: 1px solid #E74C3C; "
                      "border-radius: 6px; padding: 10px;",
                      "#922B21"),
            "warning": ("background: #FCF3CF; border: 1px solid #F39C12; "
                        "border-radius: 6px; padding: 10px;",
                        "#7D6608"),
            "info": ("background: #D6EAF8; border: 1px solid #2980B9; "
                     "border-radius: 6px; padding: 10px;",
                     "#1A5276"),
        }
        frame_style, text_color = colors.get(level, colors["info"])
        icon = {"error": "\u274c", "warning": "\u26a0\ufe0f", "info": "\u2139\ufe0f"}.get(level, "")
        self.status_frame.setStyleSheet(f"QFrame {{ {frame_style} }}")
        self.status_label.setStyleSheet(f"color: {text_color}; font-size: 12px;")
        self.status_label.setText(f"{icon} {message}")
        self.status_frame.setVisible(True)

    # ------------------------------------------------------------------
    # Siblings analyze
    # ------------------------------------------------------------------

    def _run_analyze_siblings(self, sibling_imported, sibling_calibrated):
        """Analitza N siblings independentment."""
        import os
        n = len(sibling_imported)
        names = [os.path.basename(p) for p in sibling_imported]
        logger.info("Analisi siblings: %d carpetes (%s)", n, ", ".join(names))

        self.analyze_btn.setEnabled(False)
        self.empty_state.setVisible(False)
        self.status_frame.setVisible(False)
        self.progress_frame.setVisible(True)
        self.progress_bar.setValue(0)
        self.results_frame.setVisible(False)
        self._sibling_results = {}

        if self._sibling_worker is not None:
            self._sibling_worker.wait()

        self._sibling_worker = SiblingAnalyzeWorker(
            sibling_imported, sibling_calibrated
        )
        self._sibling_worker.progress.connect(self._on_progress)
        self._sibling_worker.sibling_finished.connect(self._on_sibling_analyze_finished)
        self._sibling_worker.all_finished.connect(self._on_all_siblings_analyze_finished)
        self._sibling_worker.error.connect(self._on_error)
        self._sibling_worker.start()

    def _on_sibling_analyze_finished(self, path, result):
        """Callback per cada sibling analitzat."""
        import os
        name = os.path.basename(path)
        self._sibling_results[path] = result
        self.main_window.sibling_analyzed[path] = result
        ok = result.get("success", False)
        logger.info("Sibling analitzat %s: %s", name, "OK" if ok else "ERROR")

    def _on_all_siblings_analyze_finished(self, results):
        """Callback quan tots els siblings han estat analitzats."""
        import os

        if self._sibling_worker is not None:
            self._sibling_worker.wait()

        self.progress_frame.setVisible(False)
        self.analyze_btn.setEnabled(True)
        self._sibling_results = results

        merged = self._merge_sibling_samples(results)

        if not merged:
            self._show_inline_message("Cap sibling analitzat correctament.", level="error")
            self.analyze_completed.emit({"success": False, "error": "Cap sibling OK"})
            return

        primary_path = self.main_window.seq_path
        if primary_path in results and results[primary_path].get("success"):
            base_result = dict(results[primary_path])
        else:
            base_result = dict(next(
                r for r in results.values() if r.get("success")
            ))

        base_result["samples_grouped"] = merged
        base_result["is_sibling_merge"] = True
        base_result["sibling_results"] = results

        self.main_window.processed_data = base_result
        self.samples_grouped = merged

        from PySide6.QtWidgets import QApplication
        n_samples = len(merged)
        self.progress_label.setText(f"Preparant taula ({n_samples} mostres)...")
        self.progress_bar.setValue(95)
        self.progress_frame.setVisible(True)
        QApplication.processEvents()

        self._populate_table()
        self.results_frame.setVisible(True)

        self.progress_label.setText("Generant grafics...")
        self.progress_bar.setValue(97)
        QApplication.processEvents()

        self._populate_charts(base_result)
        self._populate_sub_tabs(base_result)

        self.progress_frame.setVisible(False)
        self.status_frame.setVisible(False)

        n = len(results)
        n_ok = sum(1 for r in results.values() if r.get("success"))
        self.main_window.set_status(
            f"Analisi completada: {n_ok}/{n} carpetes, {n_samples} mostres", 5000
        )

        self.analyze_completed.emit(base_result)

    def _merge_sibling_samples(self, results):
        """Fusiona samples_grouped de N siblings en un dict unificat."""
        import os
        merged = {}
        no_merge_types = {"KHP", "BLANK", "CONTROL"}

        for path, result in results.items():
            if not result.get("success"):
                continue

            suffix = self._get_sibling_suffix(path)
            label = suffix if suffix else "A"
            samples = result.get("samples_grouped", {})

            for name, data in samples.items():
                sample_type = data.get("sample_type", "SAMPLE")

                replicas = data.get("replicas", {})
                for rk, rd in replicas.items():
                    if isinstance(rd, dict):
                        rd["_source_path"] = path
                        rd["_source_label"] = label

                if sample_type in no_merge_types:
                    display = f"{name} [{label}]" if len(results) > 1 else name
                    if display in merged:
                        display = f"{name} [{os.path.basename(path)}]"
                    data["_source_path"] = path
                    data["_source_label"] = label
                    merged[display] = data
                elif name in merged:
                    existing = merged[name]
                    existing_reps = existing.get("replicas", {})
                    max_rep = max(
                        (int(k) for k in existing_reps if str(k).isdigit()),
                        default=0
                    )
                    for rk in sorted(replicas.keys()):
                        max_rep += 1
                        new_key = str(max_rep)
                        existing_reps[new_key] = replicas[rk]
                else:
                    data["_source_path"] = path
                    data["_source_label"] = label
                    merged[name] = data

        return merged

    @staticmethod
    def _get_sibling_suffix(path):
        """Extreu el suffix del sibling."""
        import os, re
        name = os.path.basename(path)
        clean = name.replace("_SEQ", "").replace("_BP", "").replace("_CAL", "")
        clean = clean.rstrip("_")
        m = re.match(r'^(\d+)([A-Z]?)$', clean)
        if m:
            return m.group(2)
        return ""

    # ------------------------------------------------------------------
    # Populate table
    # ------------------------------------------------------------------

    def _populate_table(self):
        """Fill the samples table with analysis results."""
        try:
            self._populate_table_inner()
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"Error populating table: {e}")

    def _populate_table_inner(self):
        """Internal table population (wrapped for safety)."""
        table = self._samples_table
        table.setRowCount(0)
        self._sample_row_map = {}
        self._row_sample_map = {}

        show_blank = (self._cat_buttons.get("blank")
                      and self._cat_buttons["blank"].isChecked())
        show_control = (self._cat_buttons.get("control")
                        and self._cat_buttons["control"].isChecked())

        # Separate samples by type
        sample_names = []
        blank_names = []
        control_names = []

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

        # Sort by injection index
        def _sort_key(name):
            sg = self.samples_grouped[name]
            reps = sg.get("replicas", {})
            keys = []
            for r in reps.values():
                if not isinstance(r, dict):
                    continue
                label = r.get("_source_label", "A")
                idx = r.get("injection_index", 999)
                sib_order = 0 if label in ("", "A") else ord(label) - ord("A")
                keys.append((sib_order, idx))
            return min(keys) if keys else (0, 999)

        for lst in (sample_names, blank_names, control_names):
            lst.sort(key=_sort_key)

        n_ok, n_warning, n_error = 0, 0, 0
        grey_bg = QColor("#F0F0F0")
        sep_bg = QColor("#EAECEE")

        # --- Regular samples ---
        for name in sample_names:
            sample_data = self.samples_grouped[name]
            row = table.rowCount()
            table.insertRow(row)
            self._sample_row_map[name] = row
            self._row_sample_map[row] = name

            # Classify for stats
            _, doc_rep = resolve_doc_replica(sample_data)
            selected = sample_data.get("selected", {}) or {}
            dad_sel = selected.get("dad", selected.get("doc", "1"))
            dad_rep = (sample_data.get("replicas", {}) or {}).get(dad_sel, {})
            comparison = sample_data.get("comparison", {})
            (sc, _, _, _, _, _) = classify_sample_status(
                doc_rep, dad_rep, comparison, sample_data=sample_data)
            if sc == COLOR_ERROR:
                n_error += 1
            elif sc == COLOR_WARNING:
                n_warning += 1
            else:
                n_ok += 1

            self._fill_sample_row(table, row, name, sample_data,
                                  doc_rep, dad_rep, comparison)

        # --- Blancs separator + rows ---
        if blank_names and show_blank:
            row = table.rowCount()
            table.insertRow(row)
            sep_item = QTableWidgetItem("--- BLANCS / MQ ---")
            sep_item.setTextAlignment(Qt.AlignCenter)
            sep_font = QFont()
            sep_font.setBold(True)
            sep_item.setFont(sep_font)
            sep_item.setForeground(QBrush(QColor("#7f8c8d")))
            table.setItem(row, 0, sep_item)
            table.setSpan(row, 0, 1, 10)
            for c in range(10):
                it = table.item(row, c) or QTableWidgetItem("")
                it.setBackground(QBrush(sep_bg))
                table.setItem(row, c, it)

            for name in blank_names:
                sample_data = self.samples_grouped[name]
                row = table.rowCount()
                table.insertRow(row)
                self._sample_row_map[name] = row
                self._row_sample_map[row] = name
                self._fill_blank_row(table, row, name, sample_data, grey_bg)

        # --- Control separator + rows ---
        if control_names and show_control:
            row = table.rowCount()
            table.insertRow(row)
            sep_item = QTableWidgetItem("--- NETEJA ---")
            sep_item.setTextAlignment(Qt.AlignCenter)
            sep_font = QFont()
            sep_font.setBold(True)
            sep_item.setFont(sep_font)
            sep_item.setForeground(QBrush(QColor("#888")))
            table.setItem(row, 0, sep_item)
            table.setSpan(row, 0, 1, 10)
            for c in range(10):
                it = table.item(row, c) or QTableWidgetItem("")
                it.setBackground(QBrush(QColor("#E8E8E8")))
                table.setItem(row, c, it)

            for name in control_names:
                sample_data = self.samples_grouped[name]
                row = table.rowCount()
                table.insertRow(row)
                self._sample_row_map[name] = row
                self._row_sample_map[row] = name
                self._fill_control_row(table, row, name, grey_bg)

        # Update stats
        total = n_ok + n_warning + n_error
        n_blank = len(blank_names)
        n_control = len(control_names)

        parts = [f"<b>{total}</b> mostres"]
        if n_blank > 0:
            parts.append(f"{n_blank} blancs")
        if n_control > 0:
            parts.append(f"{n_control} neteja")
        counts = " \u00b7 ".join(parts)

        status_parts = []
        status_parts.append(f"<span style='color:#27AE60'>\u2713</span>&nbsp;{n_ok}")
        if n_warning > 0:
            status_parts.append(f"<span style='color:#F39C12'>\u26a0</span>&nbsp;{n_warning}")
        if n_error > 0:
            status_parts.append(f"<span style='color:#E74C3C'>\u2718</span>&nbsp;{n_error}")
        status_str = " &nbsp;".join(status_parts)

        self._stats_label.setText(f"{counts} &nbsp;|&nbsp; {status_str}")
        self.status_indicator.setText(
            f"<span style='background-color: #f8f9fa; color: #2c3e50; "
            f"padding: 4px 12px; border-radius: 12px; font-size: 11px;'>"
            f"{counts} &nbsp;|&nbsp; {status_str}</span>"
        )

    # ------------------------------------------------------------------
    # Table row fill helpers
    # ------------------------------------------------------------------

    def _fill_sample_row(self, table, row, name, sample_data,
                         doc_rep, dad_rep, comparison):
        """Fill one regular sample row in the table."""
        selected = sample_data.get("selected", {}) or {}
        quantification = sample_data.get("quantification", {}) or {}

        # Col 0: Mostra
        name_item = QTableWidgetItem(name)
        name_item.setFont(QFont("Segoe UI", 10))
        table.setItem(row, 0, name_item)

        # Col 1: ppm (Direct)
        ppm_direct = quantification.get("concentration_ppm_direct")
        ppm_text = f"{ppm_direct:.2f}" if ppm_direct is not None else "\u2014"
        ppm_item = QTableWidgetItem(ppm_text)
        ppm_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
        area_doc = ((doc_rep.get("areas") or {}).get("DOC") or {}).get("total", 0)
        ppm_item.setToolTip(f"A_DOC = {area_doc:.1f}")
        table.setItem(row, 1, ppm_item)

        # Col 2: ppm_uib
        ppm_uib = quantification.get("concentration_ppm_uib")
        ppm_uib_text = f"{ppm_uib:.2f}" if ppm_uib is not None else "\u2014"
        ppm_uib_item = QTableWidgetItem(ppm_uib_text)
        ppm_uib_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
        area_uib = (doc_rep.get("areas_uib") or {}).get("total", 0)
        ppm_uib_item.setToolTip(f"A_UIB = {area_uib:.1f}")
        table.setItem(row, 2, ppm_uib_item)

        # Col 3: SNR
        snr_info = doc_rep.get("snr_info") or {}
        snr_direct = snr_info.get("snr_direct", 0)
        below_lod = quantification.get("below_lod", False)
        below_loq = quantification.get("below_loq", False)
        if below_lod:
            snr_text = f"<LOD ({snr_direct:.0f})"
            snr_item = QTableWidgetItem(snr_text)
            snr_item.setForeground(QBrush(QColor(COLOR_ERROR)))
        elif below_loq:
            snr_text = f"<LOQ ({snr_direct:.0f})"
            snr_item = QTableWidgetItem(snr_text)
            snr_item.setForeground(QBrush(QColor(COLOR_WARNING)))
        else:
            snr_text = f"{snr_direct:.0f}" if snr_direct else "\u2014"
            snr_item = QTableWidgetItem(snr_text)
        snr_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
        table.setItem(row, 3, snr_item)

        # Col 4: r2 DOC
        r2_doc = (comparison.get("doc") or {}).get("pearson", 0)
        r2_uib = (comparison.get("doc") or {}).get("pearson_uib", 0)
        if not comparison or not comparison.get("doc"):
            r2_text = "\u2014"
            r2_item = QTableWidgetItem(r2_text)
            r2_item.setForeground(QBrush(QColor("#aaa")))
        elif r2_doc >= 0.99:
            r2_text = "\u2713"
            r2_item = QTableWidgetItem(r2_text)
            r2_item.setForeground(QBrush(QColor(COLOR_SUCCESS)))
        else:
            r2_text = "\u26a0"
            r2_item = QTableWidgetItem(r2_text)
            r2_item.setForeground(QBrush(QColor(COLOR_WARNING)))
        r2_item.setTextAlignment(Qt.AlignCenter)
        tip_parts = [f"r\u00b2 DOC = {r2_doc:.4f}"]
        if r2_uib:
            tip_parts.append(f"r\u00b2 UIB = {r2_uib:.4f}")
        r2_item.setToolTip("\n".join(tip_parts))
        table.setItem(row, 4, r2_item)

        # Col 5: Timeout
        timeout_info = doc_rep.get("timeout_info") or {}
        n_timeouts = timeout_info.get("n_timeouts", 0)
        timeout_severity = timeout_info.get("severity", "OK")
        zone_summary = timeout_info.get("zone_summary", {})
        sel_key = selected.get("doc", "1")
        sel_rep = (sample_data.get("replicas", {}) or {}).get(sel_key, {})
        composed = bool(sel_rep.get("timeout_composition"))

        if n_timeouts > 0:
            if composed:
                to_text = "\u23f1\u2713"
                to_color = COLOR_SUCCESS
            else:
                # Show highest severity zone
                zone_names = list(zone_summary.keys()) if zone_summary else []
                zone_str = zone_names[0] if zone_names else ""
                to_text = f"\u23f1 {zone_str}"
                if "PIC" in str(zone_summary) or timeout_severity == "CRITICAL":
                    to_color = COLOR_ERROR
                elif any(z in ("HS",) for z in zone_names):
                    to_color = COLOR_ERROR
                elif any(z in ("BB",) for z in zone_names):
                    to_color = COLOR_WARNING
                else:
                    to_color = "#888"
            to_item = QTableWidgetItem(to_text)
            to_item.setForeground(QBrush(QColor(to_color)))
            zones_str = ", ".join(zone_summary.keys()) if zone_summary else "?"
            to_item.setToolTip(
                f"Timeouts: {n_timeouts} ({timeout_severity})\n"
                f"Zones: {zones_str}"
                + ("\nComposat" if composed else ""))
        else:
            to_item = QTableWidgetItem("")
        to_item.setTextAlignment(Qt.AlignCenter)
        table.setItem(row, 5, to_item)

        # Col 6: Pic (irregular top / saturated)
        anomalies = doc_rep.get("anomalies", [])
        has_irregular = (
            has_anomaly(anomalies, "IRREGULAR_TOP_DIRECT")
            or has_anomaly(anomalies, "IRREGULAR_TOP_UIB"))
        is_repaired = sample_data.get("repaired", False)
        is_saturated = doc_rep.get("uib_saturated", False)

        if is_saturated:
            pic_text = "SAT"
            pic_item = QTableWidgetItem(pic_text)
            pic_item.setForeground(QBrush(QColor(COLOR_ERROR)))
            pic_item.setToolTip("Senyal UIB saturat")
        elif is_repaired:
            pic_text = "\u2713 rep"
            pic_item = QTableWidgetItem(pic_text)
            pic_item.setForeground(QBrush(QColor(COLOR_SUCCESS)))
            pic_item.setToolTip("Cim irregular reparat")
        elif has_irregular:
            pic_text = "\u26a0 irreg"
            pic_item = QTableWidgetItem(pic_text)
            pic_item.setForeground(QBrush(QColor(COLOR_WARNING)))
            pic_item.setToolTip("Cim irregular detectat")
        else:
            pic_item = QTableWidgetItem("")
        pic_item.setTextAlignment(Qt.AlignCenter)
        table.setItem(row, 6, pic_item)

        # DAD columns
        dad_sel = selected.get("dad", selected.get("doc", "1"))
        dad_rep_data = (sample_data.get("replicas", {}) or {}).get(dad_sel, {})
        areas_dad = dad_rep_data.get("areas") or {}
        snr_dad = dad_rep_data.get("snr_info_dad") or {}

        # Col 7: A254
        a254_dict = areas_dad.get("A254") or {}
        a254_total = a254_dict.get("total", 0) if isinstance(a254_dict, dict) else 0
        a254_text = f"{a254_total:.0f}" if a254_total else "\u2014"
        a254_item = QTableWidgetItem(a254_text)
        a254_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
        # Tooltip with other wavelengths
        a220 = ((areas_dad.get("A220") or {}).get("total", 0)
                if isinstance(areas_dad.get("A220"), dict) else 0)
        a272 = ((areas_dad.get("A272") or {}).get("total", 0)
                if isinstance(areas_dad.get("A272"), dict) else 0)
        a290 = ((areas_dad.get("A290") or {}).get("total", 0)
                if isinstance(areas_dad.get("A290"), dict) else 0)
        a254_item.setToolTip(
            f"A220 = {a220:.0f}\nA272 = {a272:.0f}\nA290 = {a290:.0f}")
        table.setItem(row, 7, a254_item)

        # Col 8: SNR_254
        snr_254_entry = snr_dad.get("A254") or {}
        snr_254 = (snr_254_entry.get("snr", 0)
                   if isinstance(snr_254_entry, dict) else 0)
        snr_254_text = f"{snr_254:.0f}" if snr_254 else "\u2014"
        snr_254_item = QTableWidgetItem(snr_254_text)
        snr_254_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
        # Tooltip with SNR per wavelength
        snr_tip_parts = []
        for wl_key in ("A220", "A254", "A272", "A290", "A362"):
            entry = snr_dad.get(wl_key) or {}
            val = entry.get("snr", 0) if isinstance(entry, dict) else 0
            snr_tip_parts.append(f"SNR_{wl_key} = {val:.0f}")
        snr_254_item.setToolTip("\n".join(snr_tip_parts))
        table.setItem(row, 8, snr_254_item)

        # Col 9: r2_254
        dad_comp = comparison.get("dad") or {}
        pearson_per_wl = dad_comp.get("pearson_per_wavelength") or {}
        r2_254 = pearson_per_wl.get("254", 0) or pearson_per_wl.get("A254", 0)
        if not dad_comp or not pearson_per_wl:
            r2_254_text = "\u2014"
            r2_254_item = QTableWidgetItem(r2_254_text)
            r2_254_item.setForeground(QBrush(QColor("#aaa")))
        elif r2_254 >= 0.99:
            r2_254_text = "\u2713"
            r2_254_item = QTableWidgetItem(r2_254_text)
            r2_254_item.setForeground(QBrush(QColor(COLOR_SUCCESS)))
        else:
            r2_254_text = "\u26a0"
            r2_254_item = QTableWidgetItem(r2_254_text)
            r2_254_item.setForeground(QBrush(QColor(COLOR_WARNING)))
        r2_254_item.setTextAlignment(Qt.AlignCenter)
        # Tooltip with r2 per wavelength
        r2_tip_parts = []
        for wl_k, wl_v in pearson_per_wl.items():
            r2_tip_parts.append(f"r\u00b2_{wl_k} = {wl_v:.4f}")
        r2_254_item.setToolTip("\n".join(r2_tip_parts) if r2_tip_parts else "Sense comparacio")
        table.setItem(row, 9, r2_254_item)

    def _fill_blank_row(self, table, row, name, sample_data, bg_color):
        """Fill a BLANK row with simplified data."""
        quantification = sample_data.get("quantification", {}) or {}
        _, doc_rep = resolve_doc_replica(sample_data)

        # Col 0: Mostra
        name_item = QTableWidgetItem(name)
        name_item.setBackground(QBrush(bg_color))
        table.setItem(row, 0, name_item)

        # Col 1: ppm (if available)
        ppm = quantification.get("concentration_ppm_direct")
        ppm_text = f"{ppm:.2f}" if ppm is not None else "\u2014"
        ppm_item = QTableWidgetItem(ppm_text)
        ppm_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
        ppm_item.setBackground(QBrush(bg_color))
        table.setItem(row, 1, ppm_item)

        # Cols 2-9: dashes with grey background
        for c in range(2, 10):
            it = QTableWidgetItem("\u2014")
            it.setTextAlignment(Qt.AlignCenter)
            it.setForeground(QBrush(QColor("#aaa")))
            it.setBackground(QBrush(bg_color))
            table.setItem(row, c, it)

    def _fill_control_row(self, table, row, name, bg_color):
        """Fill a CONTROL (Neteja) row."""
        # Col 0: Mostra
        name_item = QTableWidgetItem(name)
        name_item.setBackground(QBrush(bg_color))
        table.setItem(row, 0, name_item)

        # Col 1: "Neteja" label
        label_item = QTableWidgetItem("Neteja")
        label_item.setTextAlignment(Qt.AlignCenter)
        label_item.setForeground(QBrush(QColor("#888")))
        label_item.setBackground(QBrush(bg_color))
        table.setItem(row, 1, label_item)

        # Cols 2-9: dashes with grey background
        for c in range(2, 10):
            it = QTableWidgetItem("\u2014")
            it.setTextAlignment(Qt.AlignCenter)
            it.setForeground(QBrush(QColor("#aaa")))
            it.setBackground(QBrush(bg_color))
            table.setItem(row, c, it)

    # ------------------------------------------------------------------
    # (Card management removed -- table is read-only)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Quantification recalculation
    # ------------------------------------------------------------------

    def _save_current_analysis(self):
        """Guarda l'estat actual de l'analisi a JSON."""
        try:
            processed = self.main_window.processed_data
            if not processed:
                return
            processed["samples_grouped"] = self.samples_grouped
            from hpsec_analyze import save_analysis_result
            save_analysis_result(processed)
        except Exception as e:
            logger.warning("Error guardant analisi: %s", e)

    # ------------------------------------------------------------------
    # Review panel (inline sample review below table)
    # ------------------------------------------------------------------

    def _on_table_row_clicked(self, index):
        """Show review panel for clicked sample."""
        row = index.row()
        sample_name = self._row_sample_map.get(row)
        if not sample_name:
            return
        self._show_review(sample_name)

    def _show_review(self, sample_name):
        """Display the review panel for a sample."""
        self._review_sample = sample_name
        sample_data = self.samples_grouped.get(sample_name)
        if not sample_data:
            return

        self._review_panel.setVisible(True)

        # Update title with navigation info
        names = list(self._sample_row_map.keys())
        idx = names.index(sample_name) if sample_name in names else 0
        self._review_title.setText(
            f"{sample_name}  ({idx + 1}/{len(names)})")
        self._review_prev_btn.setEnabled(idx > 0)
        self._review_next_btn.setEnabled(idx < len(names) - 1)

        # Build replica combos
        self._build_review_combos(sample_data)

        # Draw chromatogram
        self._draw_review_chromatogram(sample_data)

        # Update metrics
        self._update_review_metrics(sample_data)

        # Update fractions
        self._update_review_fractions(sample_data)

        # Update anomalies
        self._update_review_anomalies(sample_data)

        # Show/hide action buttons
        has_repair = bool(find_repair_targets(sample_name, self.samples_grouped))
        self._review_repair_btn.setVisible(True)
        tc = sample_data.get("timeout_composability", {})
        has_compose = tc.get("composable", False)
        has_timeouts = any(
            r.get("timeout_info", {}).get("n_timeouts", 0) > 0
            for r in sample_data.get("replicas", {}).values()
            if isinstance(r, dict))
        self._review_compose_btn.setVisible(has_compose or has_timeouts)

        # Scroll to make review panel visible
        self._scroll_area.ensureWidgetVisible(self._review_panel, 50, 50)

    def _close_review(self):
        """Hide the review panel."""
        self._review_panel.setVisible(False)
        self._review_sample = None

    def _navigate_review(self, direction):
        """Navigate to prev/next sample in review."""
        if not self._review_sample:
            return
        names = list(self._sample_row_map.keys())
        try:
            idx = names.index(self._review_sample)
        except ValueError:
            return
        new_idx = idx + direction
        if 0 <= new_idx < len(names):
            self._show_review(names[new_idx])
            row = self._sample_row_map.get(names[new_idx])
            if row is not None:
                self._samples_table.selectRow(row)

    def _build_review_combos(self, sample_data):
        """Populate DOC and DAD replica combos."""
        replicas = sample_data.get("replicas", {})
        recommendation = sample_data.get("recommendation", {})
        selected = sample_data.get("selected", {})

        doc_rec = (recommendation.get("doc") or {}).get("replica", "1")
        doc_sel = selected.get("doc", doc_rec)
        dad_rec = (recommendation.get("dad") or {}).get("replica", "1")
        dad_sel = selected.get("dad", dad_rec)

        # DOC combo
        self._review_doc_combo.blockSignals(True)
        self._review_doc_combo.clear()
        for rep_num in sorted(replicas.keys(),
                              key=lambda x: int(x) if x.isdigit() else 999):
            label = f"R{rep_num}"
            rep_data = replicas.get(rep_num, {})
            if isinstance(rep_data, dict):
                idx = rep_data.get("injection_index")
                if idx is not None:
                    label += f" ({idx})"
            if rep_num == doc_rec:
                label += " ★"
            self._review_doc_combo.addItem(label, rep_num)
            if rep_num == doc_sel:
                self._review_doc_combo.setCurrentIndex(
                    self._review_doc_combo.count() - 1)
        # Add "Comp" if any replica has timeout_composition
        has_composition = any(
            r.get("timeout_composition")
            for r in replicas.values() if isinstance(r, dict))
        if has_composition:
            self._review_doc_combo.addItem("Comp", "comp")
            if doc_sel == "comp":
                self._review_doc_combo.setCurrentIndex(
                    self._review_doc_combo.count() - 1)
        self._review_doc_combo.addItem("Cap", "none")
        if doc_sel == "none":
            self._review_doc_combo.setCurrentIndex(
                self._review_doc_combo.count() - 1)
        self._review_doc_combo.blockSignals(False)

        # DAD combo
        self._review_dad_combo.blockSignals(True)
        self._review_dad_combo.clear()
        for rep_num in sorted(replicas.keys(),
                              key=lambda x: int(x) if x.isdigit() else 999):
            label = f"R{rep_num}"
            if rep_num == dad_rec:
                label += " ★"
            self._review_dad_combo.addItem(label, rep_num)
            if rep_num == dad_sel:
                self._review_dad_combo.setCurrentIndex(
                    self._review_dad_combo.count() - 1)
        self._review_dad_combo.addItem("Cap", "none")
        if dad_sel == "none":
            self._review_dad_combo.setCurrentIndex(
                self._review_dad_combo.count() - 1)
        self._review_dad_combo.blockSignals(False)

    def _draw_review_chromatogram(self, sample_data, show_area=None):
        """Draw R1+R2 DOC chromatogram with timeout zones and optional area shading."""
        if not HAS_MATPLOTLIB:
            return
        if show_area is None:
            show_area = (hasattr(self, '_review_show_area')
                         and self._review_show_area.isChecked())
        self._review_figure.clear()
        ax = self._review_figure.add_subplot(111)

        replicas = sample_data.get("replicas", {})
        colors = ['#2E86AB', '#E74C3C', '#F39C12', '#27AE60']

        for i, (rk, rd) in enumerate(sorted(
                replicas.items(),
                key=lambda x: int(x[0]) if x[0].isdigit() else 999)):
            if not isinstance(rd, dict):
                continue
            t = rd.get("t_doc")
            y = rd.get("y_doc_net")
            if t is not None and y is not None and len(t) > 0:
                color = colors[i % len(colors)]

                # If repaired, show original as thin dotted + repaired as thick solid
                y_orig = rd.get("y_doc_net_original")
                is_repaired = (y_orig is not None and len(y_orig) == len(t))
                if is_repaired:
                    import numpy as _np
                    y_orig_arr = _np.asarray(y_orig)
                    y_arr = _np.asarray(y)
                    if not _np.allclose(y_orig_arr, y_arr, atol=1e-6):
                        ax.plot(t, y_orig, color=color, lw=0.8, ls=':',
                                alpha=0.35, label=f"R{rk} orig")
                        ax.plot(t, y, label=f"R{rk} rep",
                                color=color, lw=1.5, alpha=0.9)
                    else:
                        is_repaired = False

                if not is_repaired:
                    ax.plot(t, y, label=f"R{rk}",
                            color=color, lw=1.0, alpha=0.8)

            # Integration area shading
            if show_area:
                peak_info = rd.get("peak_info") or {}
                li = peak_info.get("peak_left_idx", 0)
                ri = peak_info.get("peak_right_idx", 0)
                if t is not None and y is not None and 0 < li < ri < len(t):
                    import numpy as _np
                    t_arr = _np.asarray(t)
                    y_arr = _np.asarray(y)
                    ax.fill_between(t_arr[li:ri+1], 0, y_arr[li:ri+1],
                                    alpha=0.12, color=color, zorder=1)
                    ax.axvline(t_arr[li], color=color, ls=':', lw=0.6, alpha=0.4)
                    ax.axvline(t_arr[ri], color=color, ls=':', lw=0.6, alpha=0.4)

            # Timeout zones
            timeout_info = rd.get("timeout_info", {})
            if timeout_info.get("n_timeouts", 0) > 0:
                for to in timeout_info.get("timeouts", []):
                    aff_start = to.get("affected_start_min", 0)
                    aff_end = to.get("affected_end_min", 0)
                    ax.axvspan(aff_start, aff_end, alpha=0.12,
                               color=colors[i % len(colors)])

        # Draw composed signal if exists
        selected = (sample_data.get("selected") or {}).get("doc", "1")
        sel_rep = replicas.get(selected, {})
        if isinstance(sel_rep, dict) and sel_rep.get("timeout_composition"):
            t_comp = sel_rep.get("t_doc")
            y_comp = sel_rep.get("y_doc_net")
            if t_comp is not None and y_comp is not None and len(t_comp) > 0:
                ax.plot(t_comp, y_comp, color="#333333", lw=1.5, ls="--",
                        alpha=0.9, label="Compost", zorder=5)

        processed = self.main_window.processed_data or {}
        method = processed.get("method", "COLUMN")
        is_bp = "BP" in method.upper() if method else False
        ax.set_xlim(0, 12 if is_bp else 70)
        ax.set_xlabel("min", fontsize=8)
        ax.set_ylabel("ppb", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(fontsize=7, loc="upper right", framealpha=0.7)

        try:
            self._review_figure.tight_layout()
        except Exception:
            pass
        self._review_canvas.draw_idle()

    def _update_review_metrics(self, sample_data):
        """Update metrics label."""
        _, doc_rep = resolve_doc_replica(sample_data)
        quant = sample_data.get("quantification", {})
        parts = []
        ppm = (quant.get("concentration_ppm_direct")
               or quant.get("concentration_ppm"))
        if ppm:
            parts.append(f"ppm: <b>{ppm:.2f}</b>")
        ppm_u = quant.get("concentration_ppm_uib")
        if ppm_u:
            parts.append(f"UIB: <b>{ppm_u:.2f}</b>")
        snr = (doc_rep.get("snr_info") or {}).get("snr_direct", 0)
        if snr:
            parts.append(f"SNR: {snr:.0f}")
        area = ((doc_rep.get("areas") or {}).get("DOC") or {}).get("total", 0)
        if area:
            parts.append(f"A: {area:.0f}")
        self._review_metrics.setText(" | ".join(parts))

    def _update_review_fractions(self, sample_data):
        """Update fractions text."""
        _, doc_rep = resolve_doc_replica(sample_data)
        areas = (doc_rep.get("areas") or {}).get("DOC") or {}
        total = areas.get("total", 0) or 0
        if total <= 0:
            self._review_fractions.setText("")
            return
        parts = []
        for frac in FRACTION_ORDER:
            val = areas.get(frac, 0) or 0
            pct = val / total * 100 if total > 0 else 0
            color = FRACTION_COLORS.get(frac, "#666")
            parts.append(
                f"<span style='color:{color}'>{frac}</span> {pct:.0f}%")
        self._review_fractions.setText("Fraccions: " + " · ".join(parts))

    def _update_review_anomalies(self, sample_data):
        """Update anomalies text."""
        replicas = sample_data.get("replicas", {})
        all_anomalies = []
        seen = set()
        for rk, rd in replicas.items():
            if not isinstance(rd, dict):
                continue
            for a in rd.get("anomalies", []):
                code = a.get("code") if isinstance(a, dict) else str(a)
                if code not in seen:
                    all_anomalies.append(a)
                    seen.add(code)
        if not all_anomalies:
            self._review_anomalies.setText("")
            return
        classified = classify_anomalies(all_anomalies)
        parts = []
        for key, icon, color in [
            ("blocker", "⛔", "#E74C3C"),
            ("warning", "⚠", "#F39C12"),
            ("info", "ℹ", "#3498DB"),
        ]:
            for a in classified.get(key, []):
                code = a.get("code") if isinstance(a, dict) else str(a)
                entry = ANOMALY_CATALOG.get(code, {})
                lbl = ((a.get("label") if isinstance(a, dict) else None)
                       or entry.get("label", code))
                action = entry.get("action", "")
                line = f"<span style='color:{color}'>{icon} {lbl}</span>"
                if action:
                    line += (f" <span style='color:#999'>"
                             f"-> {action}</span>")
                parts.append(line)
        self._review_anomalies.setText("<br>".join(parts))

    def _on_review_doc_changed(self):
        """DOC replica changed in review panel."""
        if not self._review_sample:
            return
        new_rep = self._review_doc_combo.currentData()
        if new_rep is None:
            return
        sample_data = self.samples_grouped.get(self._review_sample, {})
        sample_data.setdefault("selected", {})["doc"] = new_rep

        if new_rep == "none":
            sample_data["sample_valid"] = False
        else:
            sample_data["sample_valid"] = True

        # Recalculate quantification
        self._update_quantification(self._review_sample)

        # Redraw
        self._draw_review_chromatogram(sample_data)
        self._update_review_metrics(sample_data)
        self._update_review_fractions(sample_data)

        # Update table row
        row = self._sample_row_map.get(self._review_sample)
        if row is not None:
            _, doc_rep = resolve_doc_replica(sample_data)
            dad_sel = (sample_data.get("selected") or {}).get("dad", "1")
            dad_rep = (sample_data.get("replicas") or {}).get(dad_sel, {})
            comparison = sample_data.get("comparison", {})
            self._fill_sample_row(
                self._samples_table, row, self._review_sample,
                sample_data, doc_rep, dad_rep, comparison)

        self._save_current_analysis()

    def _on_review_dad_changed(self):
        """DAD replica changed in review panel."""
        if not self._review_sample:
            return
        new_rep = self._review_dad_combo.currentData()
        if new_rep is None:
            return
        sample_data = self.samples_grouped.get(self._review_sample, {})
        sample_data.setdefault("selected", {})["dad"] = new_rep

        # Update table row
        row = self._sample_row_map.get(self._review_sample)
        if row is not None:
            _, doc_rep = resolve_doc_replica(sample_data)
            dad_sel = (sample_data.get("selected") or {}).get("dad", "1")
            dad_rep = (sample_data.get("replicas") or {}).get(dad_sel, {})
            comparison = sample_data.get("comparison", {})
            self._fill_sample_row(
                self._samples_table, row, self._review_sample,
                sample_data, doc_rep, dad_rep, comparison)

        self._save_current_analysis()

    def _on_review_repair(self):
        """Open repair dialog for current review sample."""
        self._open_dialog_with_nav("repair")

    def _on_review_compose(self):
        """Open composition dialog for current review sample."""
        self._open_dialog_with_nav("compose")

    def _open_dialog_with_nav(self, dialog_type, sample_name=None):
        """Open repair or compose dialog with navigation between samples."""
        if sample_name is None:
            sample_name = self._review_sample
        if not sample_name:
            return
        sample_data = self.samples_grouped.get(sample_name)
        if not sample_data:
            return

        method = "COLUMN"
        if self.main_window.processed_data:
            method = self.main_window.processed_data.get("method", "COLUMN")
        is_bp = method.upper() == "BP"

        if dialog_type == "repair":
            dialog = JaggedPeakRepairDialog(
                sample_name, sample_data, method, force=True, parent=self)
            dialog.repair_completed.connect(self._on_review_repair_done)
        else:
            from .composition_dialog import TimeoutCompositionDialog
            dialog = TimeoutCompositionDialog(
                sample_name, sample_data, is_bp=is_bp, parent=self)
            dialog.composition_completed.connect(self._on_review_repair_done)

        # Connect navigation
        def _on_navigate(direction):
            # Save any pending changes
            if hasattr(dialog, '_any_changed') and dialog._any_changed:
                self._on_review_repair_done(sample_name)
            dialog.close()
            # Find next sample
            names = list(self._sample_row_map.keys())
            try:
                idx = names.index(sample_name)
            except ValueError:
                return
            new_idx = idx + direction
            if 0 <= new_idx < len(names):
                new_name = names[new_idx]
                self._show_review(new_name)
                # Open same dialog type for new sample
                self._open_dialog_with_nav(dialog_type, new_name)

        dialog.navigate_requested.connect(_on_navigate)
        dialog.exec()

    def _on_review_area_toggled(self):
        """Toggle area shading on review chromatogram."""
        if self._review_sample:
            sample_data = self.samples_grouped.get(self._review_sample)
            if sample_data:
                self._draw_review_chromatogram(sample_data)

    def _on_review_detail(self):
        """Open full SampleDetailDialog for current review sample."""
        if not self._review_sample:
            return
        sample_data = self.samples_grouped.get(self._review_sample)
        if not sample_data:
            return
        method = "COLUMN"
        if self.main_window.processed_data:
            method = self.main_window.processed_data.get("method", "COLUMN")
        from .dialogs import SampleDetailDialog
        dialog = SampleDetailDialog(
            self._review_sample, sample_data, method, parent=self.window())
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.show()

    def _on_review_repair_done(self, sample_name):
        """After repair or compose, refresh review and table."""
        self._update_quantification(sample_name)
        sample_data = self.samples_grouped.get(sample_name)
        if sample_data:
            self._show_review(sample_name)
        self._populate_table()
        self._save_current_analysis()

    def _update_quantification(self, sample_name):
        """Recalcula la quantificacio per una mostra."""
        try:
            from hpsec_analyze import quantify_sample
            from hpsec_calibrate import get_all_active_calibrations

            sample_data = self.samples_grouped[sample_name]

            if sample_data.get("skip_quantification"):
                sample_data["quantification"] = {
                    "concentration_ppm": None,
                    "concentration_ppm_direct": None,
                    "concentration_ppm_uib": None,
                    "valid": False,
                    "reason": sample_data["quantification"].get("reason",
                              "Exclosa de quantificacio") if sample_data.get("quantification") else
                              "Exclosa de quantificacio"
                }
                return

            selected_doc = sample_data["selected"]["doc"]
            _, selected_replica = resolve_doc_replica(sample_data)

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
                    if calibration_data and inj_volume:
                        if not calibration_data.get("volume_uL"):
                            calibration_data = dict(calibration_data)
                            calibration_data["volume_uL"] = inj_volume

                quantification = quantify_sample(
                    selected_replica, calibration_data,
                    mode=mode, seq_date=seq_date
                )
                hci = selected_replica.get("hci")
                if hci is not None:
                    quantification["hci"] = hci
                    quantification["hci_character"] = selected_replica.get("hci_character", "")
                sample_data["quantification"] = quantification
        except Exception as e:
            logger.error(f"Error recalculant quantificacio: {e}")
            self.main_window.set_status(f"Error quantificacio: {e}", 5000)

    # ------------------------------------------------------------------
    # Detail dialog (backward compat for double-click from card)
    # ------------------------------------------------------------------

    def _show_detail(self, sample_name):
        """Mostra el dialeg de detall."""
        if sample_name not in self.samples_grouped:
            return
        if hasattr(self, '_detail_dialog') and self._detail_dialog is not None:
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
        self._detail_dialog = dialog
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.finished.connect(lambda: self._on_detail_closed(sample_name))
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _on_detail_closed(self, sample_name):
        """Actualitza taula despres de tancar el dialeg de detall."""
        sample_data = self.samples_grouped.get(sample_name, {})
        if sample_data.get("repaired"):
            self._update_quantification(sample_name)
        # Refresh the table to reflect any changes
        if self.samples_grouped:
            self._populate_table()
        self._detail_dialog = None

    # ------------------------------------------------------------------
    # Repair/Compose (backward compat methods)
    # ------------------------------------------------------------------

    def _find_repair_targets(self, sample_name):
        """Backward compat wrapper."""
        return find_repair_targets(sample_name, self.samples_grouped)

    def _open_repair_dialog_multi(self, sample_name):
        """Obre el dialeg multi-reparacio."""
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

    def _open_repair_dialog(self, sample_name, rep_key=None, signal_type=None):
        """Backward compat."""
        self._open_repair_dialog_multi(sample_name)

    def _open_composition_dialog(self, sample_name):
        """Obre el dialeg de composicio de repliques per timeout."""
        sample_data = self.samples_grouped.get(sample_name)
        if not sample_data:
            return
        is_bp = False
        if self.main_window.processed_data:
            is_bp = self.main_window.processed_data.get("method", "COLUMN").upper() == "BP"
        from .composition_dialog import TimeoutCompositionDialog
        dialog = TimeoutCompositionDialog(
            sample_name, sample_data, is_bp=is_bp, parent=self
        )
        dialog.composition_completed.connect(self._on_repair_action)
        dialog.exec()

    def _on_repair_action(self, sample_name):
        """Actualitza la taula despres d'una accio de reparacio o composicio."""
        self._update_quantification(sample_name)
        if self.samples_grouped:
            self._populate_table()
        self._save_current_analysis()

    # ------------------------------------------------------------------
    # Backward compat: _classify_sample_status and _resolve_doc_replica
    # ------------------------------------------------------------------

    def _classify_sample_status(self, doc_rep_data, dad_rep_data, comparison,
                                sample_data=None):
        """Backward compat wrapper."""
        return classify_sample_status(doc_rep_data, dad_rep_data, comparison,
                                      sample_data=sample_data)

    def _resolve_doc_replica(self, sample_data):
        """Backward compat wrapper."""
        return resolve_doc_replica(sample_data)

    # ------------------------------------------------------------------
    # Report PDF generation
    # ------------------------------------------------------------------

    def _generate_report(self):
        """Genera el report PDF d'analisi (cridat des del wizard header)."""
        processed_data = self.main_window.processed_data
        if not processed_data:
            QMessageBox.warning(self, "Avis", "No hi ha dades processades.")
            return

        seq_path = processed_data.get("seq_path", "")
        if not seq_path:
            QMessageBox.warning(self, "Avis", "No s'ha trobat el path de la sequencia.")
            return

        try:
            from generate_analysis_report import generate_analysis_report

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
    # Charts section (migrated from ReviewSummaryPanel)
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

        self._build_sample_checkboxes(regular, blank, control, khp)
        self.charts_section.setVisible(True)

        self._charts_initialized = True
        self._redraw_charts()

    def _build_sample_checkboxes(self, regular, blank, control, khp):
        """Registra mostres per categoria."""
        self._sample_checkboxes = []
        for name in self._chart_sorted_names(regular):
            self._sample_checkboxes.append((None, name, "sample"))
        for name in self._chart_sorted_names(blank):
            self._sample_checkboxes.append((None, name, "blank"))
        for name in self._chart_sorted_names(control):
            self._sample_checkboxes.append((None, name, "control"))
        for name in self._chart_sorted_names(khp):
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
        """Longitud d'ona DAD seleccionada ha canviat."""
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
        """Retorna la longitud d'ona DAD seleccionada."""
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

    @staticmethod
    def _chart_sorted_names(samples_dict):
        """Ordena noms de mostra per sibling + injection_index."""
        def _key(name):
            data = samples_dict[name]
            reps = data.get("replicas", {})
            keys = []
            for r in reps.values():
                if not isinstance(r, dict):
                    continue
                label = r.get("_source_label", "A")
                idx = r.get("injection_index", 999)
                sib_order = 0 if label in ("", "A") else ord(label) - ord("A")
                keys.append((sib_order, idx))
            return min(keys) if keys else (0, 999)
        return sorted(samples_dict.keys(), key=_key)

    @staticmethod
    def _chart_short_label(name, data):
        """Retorna etiqueta curta per eix X."""
        sel = (data.get("selected") or {}).get("doc", "1")
        rep = (data.get("replicas") or {}).get(sel, {})
        idx = rep.get("injection_index")
        if idx is not None:
            label = rep.get("_source_label", "") if isinstance(rep, dict) else ""
            suffix = label if label and label != "A" else ""
            return f"{idx}{suffix}"
        return name[:12] + "\u2026" if len(name) > 12 else name

    @staticmethod
    def _setup_bar_hover(figure, canvas, ax, x_positions, full_names, values_per_bar):
        """Configura tooltip hover sobre barres del grafic."""
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

        for name in self._chart_sorted_names(regular):
            data = regular[name]
            sel = (data.get("selected") or {}).get("doc", "1")
            rep = (data.get("replicas") or {}).get(sel, {})
            areas = (rep.get("areas") or {}).get("DOC", {})
            quant = data.get("quantification") or {}

            names.append(name)
            labels.append(self._chart_short_label(name, data))
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
        for name in self._chart_sorted_names(light):
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
                      loc='upper center', bbox_to_anchor=(0.5, -0.12),
                      fontsize=_CHART_TICK_SIZE, framealpha=0.9,
                      ncol=len(FRACTION_ORDER), borderaxespad=0,
                      handlelength=1.2, columnspacing=0.8,
                      prop={'family': _CHART_FONT})

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, ha='center',
                           fontsize=_CHART_TICK_SIZE, fontfamily=_CHART_FONT)
        ax.set_ylabel("DOC", fontsize=_CHART_LABEL_SIZE,
                       fontfamily=_CHART_FONT)
        ax.set_title("Distribucio per fraccions", fontsize=_CHART_TITLE_SIZE,
                      fontweight='bold', fontfamily=_CHART_FONT, pad=4)
        ax.tick_params(axis='y', labelsize=_CHART_TICK_SIZE)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        self._setup_bar_hover(self.doc_figure, self.doc_canvas, ax, x, names, totals)
        self.doc_figure.tight_layout()
        self.doc_figure.subplots_adjust(bottom=0.22)
        self.doc_canvas.draw()

    def _plot_dad_chart(self, regular, light):
        """Grafic DAD per mostra."""
        self.dad_figure.clear()
        ax = self.dad_figure.add_subplot(111)
        wl = self._get_selected_wl()
        wl_key = f"A{wl}"
        is_bp = getattr(self, '_chart_is_bp', False)

        names = []
        labels = []
        fractions_data = {f: [] for f in FRACTION_ORDER}

        for name in self._chart_sorted_names(regular):
            data = regular[name]
            selected = data.get("selected") or {}
            sel = selected.get("dad", selected.get("doc", "1"))
            rep = (data.get("replicas") or {}).get(sel, {})
            areas = (rep.get("areas") or {}).get(wl_key, {})
            names.append(name)
            labels.append(self._chart_short_label(name, data))
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
        for name in self._chart_sorted_names(light):
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
                      loc='upper center', bbox_to_anchor=(0.5, -0.12),
                      fontsize=_CHART_TICK_SIZE, framealpha=0.9,
                      ncol=len(FRACTION_ORDER), borderaxespad=0,
                      handlelength=1.2, columnspacing=0.8,
                      prop={'family': _CHART_FONT})

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, ha='center',
                           fontsize=_CHART_TICK_SIZE, fontfamily=_CHART_FONT)
        ax.set_ylabel(wl_key, fontsize=_CHART_LABEL_SIZE,
                       fontfamily=_CHART_FONT)
        ax.set_title("Distribucio per fraccions", fontsize=_CHART_TITLE_SIZE,
                      fontweight='bold', fontfamily=_CHART_FONT, pad=4)
        ax.tick_params(axis='y', labelsize=_CHART_TICK_SIZE)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        self._setup_bar_hover(self.dad_figure, self.dad_canvas, ax, x, names, totals)
        self.dad_figure.tight_layout()
        self.dad_figure.subplots_adjust(bottom=0.22)
        self.dad_canvas.draw()

    @staticmethod
    def _get_line_style(data):
        """Retorna l'estil de linia segons el tipus de mostra."""
        st = data.get("sample_type", "SAMPLE")
        if st == "KHP":
            return '--'
        elif st in ("BLANK", "CONTROL"):
            return ':'
        elif st.startswith("PR"):
            return '-.'
        return '-'

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
        ax.set_title("Cromatogrames superposats", fontsize=_CHART_TITLE_SIZE,
                      fontweight='bold', fontfamily=_CHART_FONT, pad=4)

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
        ax.set_title("Cromatogrames superposats", fontsize=_CHART_TITLE_SIZE,
                      fontweight='bold', fontfamily=_CHART_FONT, pad=4)

    def _plot_doc_overlay(self, regular, light, is_bp):
        """Miniatura DOC overlay."""
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
        n = len(all_samples)
        ax.text(0.98, 0.02, f"{n} traces",
                transform=ax.transAxes, fontsize=7, ha='right', va='bottom',
                color='#aaa', fontfamily=_CHART_FONT)
        ax.text(0.02, 0.02, "+ Ampliar",
                transform=ax.transAxes, fontsize=7, ha='left', va='bottom',
                color='#446', fontfamily=_CHART_FONT,
                bbox=dict(boxstyle='round,pad=0.3', fc='#e3ecf5', ec='#c0c0c0',
                          alpha=0.85))
        self.doc_overlay_figure.tight_layout(pad=0.5)
        self.doc_overlay_canvas.draw()

    def _plot_dad_overlay(self, regular, light):
        """Miniatura DAD overlay."""
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
        ax.text(0.02, 0.02, "+ Ampliar",
                transform=ax.transAxes, fontsize=7, ha='left', va='bottom',
                color='#446', fontfamily=_CHART_FONT,
                bbox=dict(boxstyle='round,pad=0.3', fc='#e3ecf5', ec='#c0c0c0',
                          alpha=0.85))
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
