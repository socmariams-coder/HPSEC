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
    QGroupBox, QGridLayout, QCheckBox, QScrollArea, QSizePolicy,
    QSplitter, QTabWidget
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

        # v2.2.0+: Selector bar (categoria Mostres/Blancs/Control + combo λ DAD)
        # eliminat. Blancs/Controls sempre visibles a la taula amb separador.
        # Longitud d'ona DAD fixa a 254 nm a la vista principal; resta de λ
        # disponibles al diàleg de detall.
        self._cat_buttons = {}
        self._cat_counts = {}
        self._sample_checkboxes = []
        self._wl_combo = None

        # === CHARTS SECTION ===
        # v2.2.0: bar charts DOC/DAD reubicats al pas Quantificar.
        # Mantenim els canvas instanciats (referencies en altres mètodes)
        # però la secció es manté oculta permanentment a Analitzar.
        self._charts_visible = False
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
        # v2.2.0: la taula s'incorpora a un QSplitter horitzontal junt amb
        # el review panel (s'afegeix al final del setup_ui).
        self._table_container = QWidget()
        self._table_container_layout = QVBoxLayout(self._table_container)
        self._table_container_layout.setContentsMargins(0, 0, 0, 0)
        self._table_container_layout.setSpacing(0)
        self._build_table_with_group_headers()

        # === REVIEW PANEL (v2.2.0: sempre visible com a panell central de la dreta) ===
        self._review_panel = QFrame()
        self._review_panel.setVisible(True)  # v2.2.0: sempre visible al split
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
        # v2.2.0: panell sempre visible al split \u2014 amagar bot\u00f3 close
        self._review_close_btn.setVisible(False)
        nav_row.addWidget(self._review_close_btn)

        self._review_next_btn = QPushButton("\u25b6")
        self._review_next_btn.setStyleSheet(nav_btn_style)
        self._review_next_btn.setFixedWidth(32)
        self._review_next_btn.setToolTip("Mostra seguent")
        self._review_next_btn.clicked.connect(lambda: self._navigate_review(1))
        nav_row.addWidget(self._review_next_btn)

        review_layout.addLayout(nav_row)

        # === REVIEW CONTENT (single chart, no tabs) ===
        # v2.2.0+: pestanyes Cromatograma|Comparar eliminades; el panell mostra
        # directament el cromatograma de la mostra seleccionada amb DOC (eix Y
        # esquerra) + DAD 254 nm (eix Y dret).
        if HAS_MATPLOTLIB:
            self._review_figure = Figure(figsize=(8, 3), dpi=100)
            self._review_figure.set_facecolor("#FAFAFA")
            self._review_canvas = FigureCanvas(self._review_figure)
            self._review_canvas.setMinimumHeight(280)
            self._review_toolbar = NavigationToolbar2QT(
                self._review_canvas, self._review_panel)
            review_layout.addWidget(self._review_toolbar)
            review_layout.addWidget(self._review_canvas, 1)

        # Controls row: només Area + metrics (sense combos DOC/DAD per defecte;
        # la selecció ve de la taula via radios; λ fixa a 254 nm — la resta a
        # diàleg de detall).
        controls_row = QHBoxLayout()
        self._review_show_area = QCheckBox("Àrea")
        self._review_show_area.setStyleSheet("font-size: 10px;")
        self._review_show_area.setChecked(True)
        self._review_show_area.setToolTip("Mostrar/amagar ombrejat àrea integració")
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

        # Combos eliminats però referenciats encara per codi antic — stubs
        # silenciosos perquè els handlers existents no peten.
        self._review_doc_combo = None
        self._review_dad_combo = None
        self._review_tabs = None


        # === SPLIT 35/65: TAULA (esquerra) | REVIEW PANEL (dreta) ===
        # v2.2.0: vista doble simultània per maximitzar comoditat.
        self._main_splitter = QSplitter(Qt.Horizontal)
        self._main_splitter.setChildrenCollapsible(False)
        self._main_splitter.setHandleWidth(4)
        self._main_splitter.setStyleSheet(
            "QSplitter::handle { background: #E0E0E0; }"
            "QSplitter::handle:hover { background: #ADB5BD; }"
        )
        self._main_splitter.addWidget(self._table_container)
        self._main_splitter.addWidget(self._review_panel)
        # Ratio inicial 35/65
        self._main_splitter.setSizes([350, 650])
        self._main_splitter.setStretchFactor(0, 35)
        self._main_splitter.setStretchFactor(1, 65)
        results_layout.addWidget(self._main_splitter, 1)

        # === QC SEQÜÈNCIA (sota el split — collapsible, col·lapsat per defecte) ===
        self._qc_collapsible = self._build_collapsible_section(
            "QC Sequencia", collapsed=True)
        self._qc_tab = SequenceQCTab(main_window=self.main_window)
        self._qc_collapsible["content_layout"].addWidget(self._qc_tab)
        results_layout.addWidget(self._qc_collapsible["frame"])

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
        # v2.2.0: review_panel és sempre visible al split — no ocultar
        # al reset (el splitter es manté ocult perquè results_frame ho fa)
        self.charts_section.setVisible(False)
        self._qc_tab.reset()
        # comparison moved to tab Mostres
        self._charts_content.setVisible(True)
        self.analyze_btn.setEnabled(True)
        self.status_indicator.setText("")

    def _build_table_with_group_headers(self):
        """Creates the minimal sample list (Mostra | R\u00e8plica DOC | R\u00e8plica DAD).

        v2.2.0+: la taula es redueix a una llista de selecci\u00f3 amb radios
        in-line per r\u00e8plica DOC/DAD. Tota la resta d'informaci\u00f3 (SNR, r\u00b2,
        anomalies, fraccions, A254...) viu al panell de revisi\u00f3 a la dreta.
        """
        # --- Table ---
        self._samples_table = QTableWidget()
        self._samples_table.setColumnCount(3)
        self._samples_table.setHorizontalHeaderLabels([
            "Mostra", "R\u00e8plica DOC", "R\u00e8plica DAD",
        ])

        configure_table_style(self._samples_table)
        self._samples_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._samples_table.setSelectionMode(QAbstractItemView.SingleSelection)

        header = self._samples_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)

        self._samples_table.setColumnWidth(0, 160)
        self._samples_table.setColumnWidth(1, 130)
        self._samples_table.setColumnWidth(2, 110)

        # Al\u00e7ada ajustada als botons compactes (20px + petit marge)
        self._samples_table.verticalHeader().setDefaultSectionSize(24)

        self._samples_table.setMinimumHeight(300)
        self._samples_table.setMinimumWidth(320)
        self._samples_table.clicked.connect(self._on_table_row_clicked)
        self._table_container_layout.addWidget(self._samples_table)
        self._table_container.setMinimumWidth(320)

    def _check_existing_analysis(self):
        """Comprova si existeix analisi previa i la carrega automaticament.

        v2.2.0: si NO hi ha anàlisi prèvia i hi ha imported_data + calibration_data
        disponibles, dispara auto-anàlisi en background.
        """
        seq_path = self.main_window.seq_path
        if not seq_path:
            return
        if self.samples_grouped:
            return
        # Worker en marxa? No re-disparar
        if self.worker and self.worker.isRunning():
            return

        sibling_paths = getattr(self.main_window, 'sibling_paths', [])
        all_paths = [seq_path] + sibling_paths

        loaded_any = False
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
                loaded_any = True
        else:
            try:
                existing_analysis = load_analysis_result(seq_path)
                if existing_analysis and existing_analysis.get("success"):
                    self._load_existing_analysis(existing_analysis)
                    loaded_any = True
            except Exception as e:
                logger.warning(f"Error comprovant analisi existent: {e}")

        # v2.2.0: Background trigger lazy
        # Si no s'ha trobat anàlisi prèvia i hi ha dades preparades, llançar auto-anàlisi
        if not loaded_any and not self.samples_grouped:
            self._maybe_auto_analyze()

    def _maybe_auto_analyze(self):
        """Llança auto-anàlisi en background si les dades necessàries estan disponibles.

        v2.2.0: pipeline lazy — quan l'usuari arriba al tab Analitzar, si el
        delay està fixat (Verificar completat) i hi ha import_data,
        l'anàlisi s'executa automàticament en un thread separat. La UI no
        es bloqueja i mostra la progress_bar normal.
        """
        if self.worker and self.worker.isRunning():
            logger.debug("auto-analyze: worker already running")
            return

        seq_path = self.main_window.seq_path
        if not seq_path:
            logger.debug("auto-analyze: no seq_path")
            return

        # Verificar que tenim imported_data
        imported_data = self.main_window.imported_data
        if not imported_data:
            # Provar de carregar des de manifest
            try:
                from hpsec_import import import_from_manifest
                imported_data = import_from_manifest(seq_path)
                if imported_data and imported_data.get('success'):
                    self.main_window.imported_data = imported_data
            except Exception as e:
                logger.warning("auto-analyze: error carregant import: %s", e)
                return

        if not imported_data or not imported_data.get('success'):
            logger.info("auto-analyze: sense imported_data — esperar Importar")
            return

        # Verificar calibration_data (delay fixat + KHP)
        calibration_data = self.main_window.calibration_data
        if not calibration_data and seq_path:
            cal_json = Path(seq_path) / "CHECK" / "data" / "calibration_result.json"
            if not cal_json.exists():
                logger.info("auto-analyze: sense calibration_result.json — esperar Verificar")
                return

        logger.info("auto-analyze: disparant anàlisi en background per %s",
                     self.main_window.seq_name)
        # _run_analyze és asincron (QThread) — no bloqueja
        try:
            self._run_analyze()
        except Exception as e:
            logger.warning("auto-analyze: error disparant _run_analyze: %s", e)

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
        """Executa l'analisi.

        v2.2.0: early-return si ja s'està executant (no bloqueig UI thread).
        """
        # Early-return: worker actiu (v2.2.0 — abans bloquejava UI amb wait())
        if self.worker is not None and self.worker.isRunning():
            logger.info("_run_analyze: worker already running — skip")
            return

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

        # v2.2.0: no waitejar el worker anterior aquí — l'early-return a dalt
        # ja ho garanteix. wait() bloquejava la UI thread fins acabar.
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

        # v2.2.0: auto-seleccionar primera mostra perquè el panel central
        # (review) mostri contingut des del primer moment al split layout.
        try:
            if self.samples_grouped:
                first_name = next(iter(self.samples_grouped.keys()))
                self._show_review(first_name)
        except Exception as e:
            logger.debug("Auto-select first sample failed: %s", e)

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
            return
        # Auto-selecció: primera mostra → poblar gràfica per defecte
        try:
            if not self._review_sample and self._sample_row_map:
                first_name = next(iter(self._sample_row_map.keys()))
                self._samples_table.selectRow(self._sample_row_map[first_name])
                self._show_review(first_name)
        except Exception as e:
            logger.debug("Auto-select first sample skipped: %s", e)

    def _populate_table_inner(self):
        """Internal table population (wrapped for safety)."""
        table = self._samples_table
        table.setRowCount(0)
        self._sample_row_map = {}
        self._row_sample_map = {}

        # v2.2.0+: blancs i controls sempre visibles (selector eliminat)
        show_blank = True
        show_control = True

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
            table.setSpan(row, 0, 1, 3)
            for c in range(3):
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
            table.setSpan(row, 0, 1, 3)
            for c in range(3):
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

    # ------------------------------------------------------------------
    # v2.2.0: Replica selector inline (radio toggle strip per fila)
    # ------------------------------------------------------------------

    def _create_replica_strip(self, sample_name, signal, replicas_dict,
                               current, allow_comp=True):
        """Crea widget compacte amb botons toggle R1/R2/.../Comp/Cap.

        Args:
            sample_name: nom mostra (string)
            signal: 'doc' o 'dad'
            replicas_dict: dict {key: replica_data}
            current: clau actualment seleccionada
            allow_comp: si True, afegir bot\u00f3 'Comp' (nom\u00e9s DOC + timeout)
        """
        from PySide6.QtWidgets import QButtonGroup
        container = QWidget()
        h = QHBoxLayout(container)
        h.setContentsMargins(2, 1, 2, 1)
        h.setSpacing(1)

        group = QButtonGroup(container)
        group.setExclusive(True)
        # Mantenir refer\u00e8ncia perqu\u00e8 no es destrueixi
        if not hasattr(self, '_replica_groups'):
            self._replica_groups = {}
        self._replica_groups[(sample_name, signal)] = group

        rep_keys = sorted([k for k in replicas_dict.keys() if k not in (None, "")])
        options = [(k, f"R{k}") for k in rep_keys]
        if allow_comp and signal == "doc":
            options.append(("Comp", "Cp"))
        # v2.2.0: clau "none" canonical (consistent amb la resta del codebase).
        # El label es mant\u00e9 "\u2014" per claredat visual.
        options.append(("none", "\u2014"))

        # Botons compactes: alçada menor + padding intern reduït perquè el
        # text quedi centrat sense talls. min-width 26px per assegurar que
        # 'Cp' i '—' es vegin sencers.
        btn_style = (
            "QPushButton { font-size: 10px; padding: 0px 6px; min-width: 26px;"
            " border: 1px solid #ced4da; border-radius: 3px;"
            " background: white; color: #495057; }"
            "QPushButton:hover { background: #e9ecef; }"
            "QPushButton:checked { background: #2E86AB; color: white;"
            " border-color: #1f6080; font-weight: bold; }")

        for key, label in options:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setFixedHeight(20)
            btn.setStyleSheet(btn_style)
            if str(key) == str(current):
                btn.setChecked(True)
            group.addButton(btn)
            btn.toggled.connect(
                lambda checked, s=sample_name, sig=signal, k=key:
                self._on_replica_changed(s, sig, k) if checked else None)
            h.addWidget(btn)
        h.addStretch()
        return container

    def _on_replica_changed(self, sample_name, signal, replica_key):
        """Callback quan l'usuari canvia la r\u00e8plica seleccionada per una mostra.

        Actualitza el state i refresca el review panel si la mostra activa
        \u00e9s aquesta.
        """
        sample_data = self.samples_grouped.get(sample_name)
        if not sample_data:
            return
        selected = sample_data.setdefault("selected", {})
        previous = selected.get(signal)
        if str(previous) == str(replica_key):
            return  # Sense canvi
        selected[signal] = replica_key
        logger.info("Selecci\u00f3 r\u00e8plica %s/%s: %s \u2192 %s",
                    sample_name, signal, previous, replica_key)
        # Si la mostra activa al review \u00e9s aquesta, refrescar cromatograma
        if self._review_sample == sample_name:
            try:
                self._show_review(sample_name)
            except Exception as e:
                logger.warning("Error refresh review panel: %s", e)

    def _fill_sample_row(self, table, row, name, sample_data,
                         doc_rep, dad_rep, comparison):
        """Fill one regular sample row — Mostra | Rèplica DOC | Rèplica DAD.

        v2.2.0+: la taula és una llista de selecció amb radios in-line.
        Tota la resta d'info (SNR, r², anomalies, fraccions, A254…) viu al
        panell de revisió a la dreta.
        """
        selected = sample_data.get("selected", {}) or {}
        replicas_dict = sample_data.get("replicas", {}) or {}

        # Col 0: Mostra (amb tooltip resum d'info que abans tenia col·lumna pròpia)
        name_item = QTableWidgetItem(name)
        name_item.setFont(QFont("Segoe UI", 10))
        tip_parts = []
        snr = (doc_rep.get("snr_info") or {}).get("snr_direct", 0)
        if snr:
            tip_parts.append(f"SNR DOC = {snr:.0f}")
        r2_doc = (comparison.get("doc") or {}).get("pearson")
        if r2_doc is not None:
            tip_parts.append(f"r² DOC = {r2_doc:.3f}")
        if doc_rep.get("uib_saturated"):
            tip_parts.append("⚠ UIB saturat")
        n_to = (doc_rep.get("timeout_info") or {}).get("n_timeouts", 0)
        if n_to:
            tip_parts.append(f"timeouts: {n_to}")
        anomalies = doc_rep.get("anomalies", [])
        if has_anomaly(anomalies, "IRREGULAR_TOP_DIRECT") or has_anomaly(anomalies, "IRREGULAR_TOP_UIB"):
            tip_parts.append("⚠ cim irregular")
        if sample_data.get("repaired"):
            tip_parts.append("✓ reparat")
        if tip_parts:
            name_item.setToolTip("\n".join(tip_parts))
        table.setItem(row, 0, name_item)

        # Col 1: Rèplica DOC (widget amb botons toggle R1/R2/Comp/Cap)
        doc_sel = selected.get("doc", "1")
        tc = sample_data.get("timeout_composability", {}) or {}
        doc_strip = self._create_replica_strip(
            name, "doc", replicas_dict, doc_sel,
            allow_comp=bool(tc.get("composable")))
        table.setCellWidget(row, 1, doc_strip)

        # Col 2: Rèplica DAD (widget radio R1/R2/Cap)
        dad_sel = selected.get("dad", selected.get("doc", "1"))
        dad_strip = self._create_replica_strip(
            name, "dad", replicas_dict, dad_sel, allow_comp=False)
        table.setCellWidget(row, 2, dad_strip)

    def _fill_blank_row(self, table, row, name, sample_data, bg_color):
        """Fill a BLANK row — only name + 'Blanc' label (v2.2.0+ 3-col layout)."""
        name_item = QTableWidgetItem(name)
        name_item.setBackground(QBrush(bg_color))
        table.setItem(row, 0, name_item)

        label_item = QTableWidgetItem("Blanc")
        label_item.setTextAlignment(Qt.AlignCenter)
        label_item.setForeground(QBrush(QColor("#888")))
        label_item.setBackground(QBrush(bg_color))
        table.setItem(row, 1, label_item)

        empty = QTableWidgetItem("")
        empty.setBackground(QBrush(bg_color))
        table.setItem(row, 2, empty)

    def _fill_control_row(self, table, row, name, bg_color):
        """Fill a CONTROL (Neteja) row (v2.2.0+ 3-col layout)."""
        name_item = QTableWidgetItem(name)
        name_item.setBackground(QBrush(bg_color))
        table.setItem(row, 0, name_item)

        label_item = QTableWidgetItem("Neteja")
        label_item.setTextAlignment(Qt.AlignCenter)
        label_item.setForeground(QBrush(QColor("#888")))
        label_item.setBackground(QBrush(bg_color))
        table.setItem(row, 1, label_item)

        empty = QTableWidgetItem("")
        empty.setBackground(QBrush(bg_color))
        table.setItem(row, 2, empty)

    # ------------------------------------------------------------------
    # (Card management removed -- table is read-only)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Quantification recalculation
    # ------------------------------------------------------------------

    def _save_current_analysis(self):
        """Guarda l'estat actual de l'anàlisi a JSON.

        v2.2.0: NO toca la quantificació (que es fa al pas 4). Si el
        quantification_pending estava True, ha de quedar True després
        d'aquest save — les modificacions a Analitzar (selecció rèplica,
        repair...) invaliden la quantificació prèvia.
        """
        try:
            processed = self.main_window.processed_data
            if not processed:
                logger.warning("_save_current_analysis: processed_data is None — NO ES GUARDA")
                return
            processed["samples_grouped"] = self.samples_grouped
            # v2.2.0: si l'usuari modifica seleccions a Analitzar, la
            # quantificació actual queda obsoleta. Marcar pending.
            processed["quantification_pending"] = True
            # Invalidar quantification dels samples_grouped (sense esborrar
            # del tot — només marcar perquè el pas Quantificar la regeneri)
            for sg in self.samples_grouped.values():
                if isinstance(sg, dict) and sg.get("quantification"):
                    sg["quantification"]["valid"] = False
                    sg["quantification"]["reason"] = "Estale (cal reaplicar recta)"
            from hpsec_analyze import save_analysis_result
            save_analysis_result(processed)
            logger.info("_save_current_analysis: guardat OK (quantification_pending=True)")
        except Exception as e:
            logger.warning("Error guardant analisi: %s", e)
            import traceback; traceback.print_exc()

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

        # v2.2.0: actualitzar tab "Comparar R1↔R2"
        self._update_compare_tab(sample_data)

        # Scroll to make review panel visible
        self._scroll_area.ensureWidgetVisible(self._review_panel, 50, 50)

    # ------------------------------------------------------------------
    # v2.2.0: Tab "Comparar R1↔R2" — overlay rèpliques + estadístiques
    # ------------------------------------------------------------------

    def _update_compare_tab(self, sample_data):
        """v2.2.0+: tab Comparar eliminat — funció no-op."""
        return
        # === codi original mantingut sota return per referència ===
        if not HAS_MATPLOTLIB or not hasattr(self, '_compare_figure'):
            return

        replicas = sample_data.get("replicas", {}) or {}
        valid_reps = [k for k, r in replicas.items()
                       if isinstance(r, dict) and r.get("t_doc") is not None
                       and r.get("y_doc") is not None]

        self._compare_figure.clear()

        # Habilitar/deshabilitar tab segons disponibilitat
        if len(valid_reps) < 2:
            ax = self._compare_figure.add_subplot(111)
            ax.text(0.5, 0.5,
                    "Mostra amb només 1 rèplica — sense comparació",
                    ha='center', va='center', fontsize=11, color='#888',
                    transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            self._compare_canvas.draw()
            self._compare_stats.setText(
                "<i>No es pot comparar amb una sola rèplica.</i>")
            self._review_tabs.setTabEnabled(self._review_compare_tab_idx, False)
            return

        self._review_tabs.setTabEnabled(self._review_compare_tab_idx, True)

        rep_keys = sorted(valid_reps)[:2]  # primeres 2 rèpliques
        r1_data = replicas[rep_keys[0]]
        r2_data = replicas[rep_keys[1]]

        # === Plot 1: overlay R1+R2 ===
        ax_top = self._compare_figure.add_subplot(2, 1, 1)
        try:
            import numpy as np
            t1 = np.asarray(r1_data["t_doc"], dtype=float)
            y1 = np.asarray(r1_data["y_doc"], dtype=float)
            t2 = np.asarray(r2_data["t_doc"], dtype=float)
            y2 = np.asarray(r2_data["y_doc"], dtype=float)
            ax_top.plot(t1, y1, color="#2E86AB", linewidth=1.4,
                        label=f"R{rep_keys[0]}", alpha=0.85)
            ax_top.plot(t2, y2, color="#A23B72", linewidth=1.4,
                        label=f"R{rep_keys[1]}", alpha=0.85)
            ax_top.set_ylabel("DOC (ppb)", fontsize=9)
            ax_top.legend(fontsize=9, loc='best')
            ax_top.grid(True, alpha=0.3)
            ax_top.tick_params(labelsize=8)
        except Exception as e:
            logger.warning("Error compare overlay: %s", e)

        # === Plot 2: diferència (R1 - R2) interpolat ===
        ax_bot = self._compare_figure.add_subplot(2, 1, 2)
        try:
            common_t = np.linspace(max(t1.min(), t2.min()),
                                    min(t1.max(), t2.max()), 800)
            y1_i = np.interp(common_t, t1, y1)
            y2_i = np.interp(common_t, t2, y2)
            diff = y1_i - y2_i
            ax_bot.plot(common_t, diff, color="#666", linewidth=1.0)
            ax_bot.fill_between(common_t, 0, diff,
                                 where=(diff >= 0), color="#2E86AB", alpha=0.3)
            ax_bot.fill_between(common_t, 0, diff,
                                 where=(diff < 0), color="#A23B72", alpha=0.3)
            ax_bot.axhline(0, color="#aaa", linewidth=0.5)
            ax_bot.set_xlabel("t (min)", fontsize=9)
            ax_bot.set_ylabel(f"R{rep_keys[0]} − R{rep_keys[1]}", fontsize=9)
            ax_bot.grid(True, alpha=0.3)
            ax_bot.tick_params(labelsize=8)
        except Exception as e:
            logger.warning("Error compare diff: %s", e)

        self._compare_figure.tight_layout()
        self._compare_canvas.draw()

        # === Estadístiques ===
        try:
            comparison = sample_data.get("comparison") or {}
            doc_comp = comparison.get("doc") or {}
            pearson = doc_comp.get("pearson", 0)
            rsd_area = doc_comp.get("rsd_area", None)
            area_r1 = ((r1_data.get("areas") or {}).get("DOC") or {}).get("total", 0)
            area_r2 = ((r2_data.get("areas") or {}).get("DOC") or {}).get("total", 0)
            snr_r1 = (r1_data.get("snr_info") or {}).get("snr_direct", 0)
            snr_r2 = (r2_data.get("snr_info") or {}).get("snr_direct", 0)

            d_area = area_r1 - area_r2
            d_area_pct = (d_area / area_r1 * 100) if area_r1 else 0

            r2_text_color = "#28a745" if pearson >= 0.99 else "#d4a017"
            stats_html = (
                f"<b>Pearson r²:</b> "
                f"<span style='color:{r2_text_color};'>{pearson:.4f}</span> · "
                f"<b>Àrea R{rep_keys[0]}:</b> {area_r1:.1f} · "
                f"<b>Àrea R{rep_keys[1]}:</b> {area_r2:.1f} · "
                f"<b>Δàrea:</b> {d_area:+.1f} ({d_area_pct:+.1f}%) · "
                f"<b>SNR:</b> R{rep_keys[0]}={snr_r1:.0f} / R{rep_keys[1]}={snr_r2:.0f}"
            )
            if rsd_area is not None:
                stats_html += f" · <b>RSD:</b> {rsd_area:.2f}%"
            self._compare_stats.setText(stats_html)
        except Exception as e:
            logger.debug("Error compare stats: %s", e)
            self._compare_stats.setText(
                "<i>Estadístiques no disponibles.</i>")

    def _apply_compare_action(self, replica_key):
        """v2.2.0+: tab Comparar eliminat — funció no-op."""
        return

    def _close_review(self):
        """Clear review selection (v2.2.0: panell sempre visible al split, només deselecciona)."""
        self._review_sample = None
        # Mantenim el panell visible amb el contingut actual; només neteja l'estat
        # de selecció. La crida explícita només es fa via accions internes/legacy.

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
        """v2.2.0+: combos DOC/DAD eliminats — funció no-op (selecció via radios de la taula)."""
        return
        # === codi original mantingut sota return per referència ===
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
            if t is not None and y is not None and len(t) > 10:
                import numpy as _np
                from hpsec_core import find_peak_boundaries as _fpb
                t_arr = _np.asarray(t, dtype=float)
                y_arr = _np.asarray(y, dtype=float)
                y_pos = _np.maximum(y_arr, 0)

                processed = self.main_window.processed_data or {}
                _method = processed.get("method", "COLUMN")
                _is_bp = "BP" in _method.upper() if _method else False

                if _is_bp:
                    # BP: trapezoid complet (el que quantifica la Suite)
                    ax.fill_between(t_arr, 0, y_pos,
                                    alpha=0.12, color=color, zorder=1)
                else:
                    # COLUMN: fraccions colorades
                    try:
                        from hpsec_config import get_config as _gc
                        _cfg = _gc()
                        _fracs = {fn: (fi["start"], fi["end"])
                                  for fn, fi in _cfg.get_all_fractions()}
                    except Exception:
                        _fracs = {"BioP": (10.8, 18), "HS": (18, 23),
                                  "BB": (23, 26), "SB": (26, 32), "LMW": (32, 70)}
                    _fcolors = {"BioP": "#E74C3C", "HS": "#F39C12",
                                "BB": "#27AE60", "SB": "#3498DB", "LMW": "#95A5A6"}
                    for fn, (t0, t1) in _fracs.items():
                        mask = (t_arr >= t0) & (t_arr <= t1)
                        if mask.any():
                            fc = _fcolors.get(fn, "#CCC")
                            ax.fill_between(t_arr[mask], 0, y_pos[mask],
                                            alpha=0.10, color=fc, zorder=1)
                            ax.axvline(t0, color=fc, ls='--', lw=0.3, alpha=0.3)

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
        ax.set_ylabel("DOC (ppb)", fontsize=8, color="#2E86AB")
        ax.tick_params(labelsize=7, axis='y', colors="#2E86AB")
        ax.tick_params(labelsize=7, axis='x')
        ax.spines['top'].set_visible(False)

        # === DAD 254 nm a eix Y secundari (dreta) ===
        # La selecció DAD ve dels radios DOC/DAD de la taula. Si la rèplica
        # seleccionada té DAD disponible, es plota com a línia ataronjada al
        # twin axis. La resta de λ s'accedeixen al diàleg de detall.
        try:
            dad_sel = (sample_data.get("selected") or {}).get(
                "dad", (sample_data.get("selected") or {}).get("doc", "1"))
            dad_data = (sample_data.get("replicas") or {}).get(dad_sel, {})
            df_dad = dad_data.get("df_dad")
            t_dad = dad_data.get("t_dad")
            y_254 = None
            if df_dad is not None and hasattr(df_dad, "get"):
                # df_dad pot ser un DataFrame amb columnes 'Time' i wavelengths
                if hasattr(df_dad, "columns"):
                    for col_name in ("254", "A254", 254, 254.0):
                        if col_name in df_dad.columns:
                            y_254 = df_dad[col_name].values
                            if t_dad is None and "Time" in df_dad.columns:
                                t_dad = df_dad["Time"].values
                            break
            if y_254 is None:
                # Fallback: alguns formats guarden els arrays directament
                signals_dad = dad_data.get("signals_dad") or {}
                for k in ("254", "A254", 254):
                    if k in signals_dad:
                        y_254 = signals_dad[k]
                        break
            if y_254 is not None and t_dad is not None and len(y_254) > 0:
                import numpy as _np
                t_dad_arr = _np.asarray(t_dad, dtype=float)
                y_254_arr = _np.asarray(y_254, dtype=float)
                ax_dad = ax.twinx()
                ax_dad.plot(t_dad_arr, y_254_arr, color="#E67E22",
                            lw=1.0, alpha=0.8, label="DAD A254")
                ax_dad.set_ylabel("DAD A254 (mAU)", fontsize=8, color="#E67E22")
                ax_dad.tick_params(labelsize=7, axis='y', colors="#E67E22")
                ax_dad.spines['top'].set_visible(False)
                # Legenda combinada
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax_dad.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2,
                          fontsize=7, loc="upper right", framealpha=0.7)
            else:
                ax.spines['right'].set_visible(False)
                ax.legend(fontsize=7, loc="upper right", framealpha=0.7)
        except Exception as e:
            logger.debug("DAD twin axis skipped: %s", e)
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
        quant = sample_data.get("quantification", {}) or {}
        parts = []
        # v2.2.0+: si la quantificació està pendent, no mostrar ppm
        # (encara no s'ha aplicat la recta de calibració al pas Quantificar).
        processed = self.main_window.processed_data or {}
        quant_pending = bool(processed.get("quantification_pending"))
        if not quant_pending:
            ppm = (quant.get("concentration_ppm_direct")
                   or quant.get("concentration_ppm"))
            if ppm:
                parts.append(f"ppm: <b>{ppm:.2f}</b>")
            ppm_u = quant.get("concentration_ppm_uib")
            if ppm_u:
                parts.append(f"UIB: <b>{ppm_u:.2f}</b>")
        else:
            parts.append("<i style='color:#888'>ppm pendent</i>")
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
        """v2.2.0+: combo eliminat — no-op."""
        return
        # === codi original mantingut sota return ===
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
        """v2.2.0+: combo eliminat — no-op."""
        return
        # === codi original mantingut sota return ===
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
        logger.info("_on_review_repair: _review_sample=%s", self._review_sample)
        self._open_dialog_with_nav("repair")

    def _on_review_compose(self):
        """Open composition dialog for current review sample."""
        self._open_dialog_with_nav("compose")

    def _open_dialog_with_nav(self, dialog_type, sample_name=None):
        """Open repair or compose dialog with navigation between samples."""
        if sample_name is None:
            sample_name = self._review_sample
        if not sample_name:
            logger.warning("_open_dialog_with_nav: no sample_name (review_sample=%s)", self._review_sample)
            return
        sample_data = self.samples_grouped.get(sample_name)
        if not sample_data:
            logger.warning("_open_dialog_with_nav: sample '%s' not in samples_grouped (keys=%s)",
                          sample_name, list(self.samples_grouped.keys())[:5])
            return
        logger.info("_open_dialog_with_nav: %s type=%s replicas=%s",
                    sample_name, dialog_type, list(sample_data.get("replicas", {}).keys()))

        method = "COLUMN"
        if self.main_window.processed_data:
            method = self.main_window.processed_data.get("method", "COLUMN")
        is_bp = method.upper() == "BP"

        if dialog_type == "repair":
            # Verificar que hi ha dades DOC a les rèpliques
            has_data = False
            for rk, rd in sample_data.get("replicas", {}).items():
                if isinstance(rd, dict):
                    t_doc = rd.get("t_doc")
                    if t_doc is not None and len(t_doc) > 0:
                        has_data = True
                        break
            if not has_data:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.information(self, "Reparar pic",
                    f"No hi ha dades de cromatograma per {sample_name}.\n"
                    "Pot ser que calgui reprocessar (Verificar + Analitzar).")
                return
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
        """Prepara dades pels grafics i mostra la seccio.

        v2.2.0: charts (DOC bars, DAD bars) reubicats al pas Quantificar.
        Aquesta funció no fa res — es manté per backward compat amb
        invocacions externes que encara la criden.
        """
        return  # v2.2.0: dead path
        # ----------- Codi original deprecated -----------
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
        # v2.2.0: charts (ppm + bars) reubicats al pas Quantificar.
        # Mantenim charts_section invisible permanentment a Analitzar.
        self.charts_section.setVisible(False)

        self._charts_initialized = True
        # No re-dibuixar charts (no es veuen)

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
