"""
HPSEC Suite - Analyze Panel (Fase 3) — Taula Unificada DOC+DAD
================================================================

Panel per la fase 3: Anàlisi de mostres.
- Taula unificada (1 fila per mostra) amb DOC + DAD principals
- Panel de fraccions visible al seleccionar mostra
- Classificació d'anomalies per severitat (no falsos avisos)
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QComboBox,
    QFrame, QAbstractItemView, QProgressBar, QMessageBox, QDialog,
    QGroupBox, QGridLayout, QCheckBox, QScrollArea, QTabWidget
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

# --- Estil global gràfics ---
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
    configure_table_style, populate_signal_summary, populate_fractions_table
)
from .sequence_qc_tab import SequenceQCTab
from .comparison_tab import ComparisonTab

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
            """Llegenda clicable: toggle visibilitat de cada traça."""
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
    """Panel d'anàlisi de mostres (Fase 3) — Taula unificada."""

    analyze_completed = Signal(dict)

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.samples_grouped = {}
        self.worker = None
        self._warnings_confirmed = False
        self._warnings_confirmed_by = ""
        self._selected_sample = None
        self._sample_row_map = {}       # P3: sample_name → row index (O(1) lookup)
        self._status_initialized = False  # B3: avoid redundant showEvent work
        # Chart data
        self._chart_regular = {}
        self._chart_blank = {}
        self._chart_control = {}
        self._chart_khp = {}
        self._chart_is_bp = False
        # Siblings
        self._sibling_worker = None
        self._sibling_results = {}  # {path: analysis_result}

        self._setup_ui()

    # ------------------------------------------------------------------
    # UI Setup
    # ------------------------------------------------------------------

    def _setup_ui(self):
        """Configura la interfície — Taula unificada + panel fraccions."""
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        # Botó analitzar (amagat - l'acció es dispara des del wizard header)
        self.analyze_btn = QPushButton()
        self.analyze_btn.setVisible(False)
        self.analyze_btn.clicked.connect(self._run_analyze)

        # === SCROLL AREA per contenir tot el contingut ===
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setStyleSheet("QScrollArea { border: none; }")

        scroll_content = QWidget()
        layout = QVBoxLayout(scroll_content)
        apply_panel_layout(layout)

        # === INFO PANEL (amagat — info ja visible a la taula) ===
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
            "🔬", "Preparant anàlisi...",
            "Carregant dades de la seqüència."
        )
        self.empty_state.setVisible(False)
        layout.addWidget(self.empty_state)

        # Status frame (mantingut per backward compat, sempre amagat)
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

        # === F0: SELECTOR BAR (primer element — filtra taula + gràfics) ===
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

        # === UNIFIED TABLE ===
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(16)
        self.results_table.setHorizontalHeaderLabels([
            "Mostra", "Inj", "Sel DOC", "Sel DAD", "A_DOC", "ppm",
            "A_UIB", "ppm_U", "SNR", "A_254", "SNR_254",
            "R²_DOC", "R²_DAD", "HCI", "Estat", "Rpar."
        ])
        self.results_table.setMinimumHeight(180)
        configure_table_style(self.results_table)
        self._configure_unified_columns()
        results_layout.addWidget(self.results_table)

        # Connect table signals
        self.results_table.doubleClicked.connect(self._on_table_double_click)
        self.results_table.cellClicked.connect(self._on_table_cell_click)
        self.results_table.setToolTip("Doble-clic per detall complet")

        # (results_frame s'afegirà al tab "Resultats" més avall)

        # === CHARTS SECTION (sempre visible, sense collapsible) ===
        self._charts_visible = True
        self._charts_initialized = False
        self.charts_section = QFrame()
        self.charts_section.setVisible(False)
        charts_outer = QVBoxLayout(self.charts_section)
        charts_outer.setContentsMargins(0, 8, 0, 0)
        charts_outer.setSpacing(4)

        # Charts content (sempre visible)
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

            # DOC overlay miniatura + botó ampliar
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

            # DAD overlay miniatura + botó ampliar
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

        # === TAB WIDGET (Resultats + QC + Comparació) ===
        self._tab_widget = QTabWidget()
        self._tab_widget.setVisible(False)
        self._tab_widget.setStyleSheet(
            "QTabWidget::pane { border: none; }"
            "QTabBar::tab { font-size: 11px; padding: 6px 16px; }"
            "QTabBar::tab:selected { font-weight: bold; }"
        )

        # Tab 0: Resultats (taula + gràfics existents)
        results_tab_container = QWidget()
        results_tab_layout = QVBoxLayout(results_tab_container)
        results_tab_layout.setContentsMargins(0, 4, 0, 0)
        results_tab_layout.setSpacing(4)
        results_tab_layout.addWidget(self.results_frame)
        results_tab_layout.addWidget(self.charts_section)
        self._tab_widget.addTab(results_tab_container, "Resultats")

        # Tab 1: QC Seqüència
        self._qc_tab = SequenceQCTab(main_window=self.main_window)
        self._tab_widget.addTab(self._qc_tab, "QC Seqüència")

        # Tab 2: Comparació COL↔BP
        self._comparison_tab = ComparisonTab(main_window=self.main_window)
        self._tab_widget.addTab(self._comparison_tab, "Comparació COL↔BP")

        layout.addWidget(self._tab_widget, 1)

        # Completar scroll area
        scroll_area.setWidget(scroll_content)
        outer_layout.addWidget(scroll_area, 1)

    def _configure_unified_columns(self):
        """Configura columnes de la taula unificada."""
        header = self.results_table.horizontalHeader()
        for i in range(self.results_table.columnCount()):
            if i == 14:  # Estat — stretch
                header.setSectionResizeMode(i, QHeaderView.Stretch)
            elif i == 15:  # Rpar. — compact
                header.setSectionResizeMode(i, QHeaderView.ResizeToContents)
            else:
                header.setSectionResizeMode(i, QHeaderView.ResizeToContents)

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
        self._sample_row_map = {}
        self._status_initialized = False
        self._chart_regular = {}
        self._chart_blank = {}
        self._chart_control = {}
        self._chart_khp = {}
        self._chart_is_bp = False
        self._charts_initialized = False
        self._sibling_worker = None
        self._sibling_results = {}

        self.results_table.setRowCount(0)

        self.empty_state.setVisible(True)
        self.info_frame.setVisible(False)
        self.status_frame.setVisible(False)
        self.progress_frame.setVisible(False)
        self.progress_bar.setValue(0)
        self.results_frame.setVisible(False)
        self.charts_section.setVisible(False)
        self._tab_widget.setVisible(False)
        self._tab_widget.setCurrentIndex(0)
        self._qc_tab.reset()
        self._comparison_tab.reset()
        self._charts_content.setVisible(True)
        self.analyze_btn.setEnabled(True)
        self.status_indicator.setText("")

    def _check_existing_analysis(self):
        """Comprova si existeix anàlisi prèvia i la carrega automàticament.

        Si hi ha siblings, carrega i fusiona tots els resultats.
        """
        seq_path = self.main_window.seq_path
        if not seq_path:
            return
        if self.samples_grouped:
            return

        sibling_paths = getattr(self.main_window, 'sibling_paths', [])
        all_paths = [seq_path] + sibling_paths

        if len(all_paths) > 1:
            # Carregar anàlisi de cada sibling i fusionar
            results = {}
            for path in all_paths:
                try:
                    r = load_analysis_result(path)
                    if r and r.get("success"):
                        results[path] = r
                except Exception as e:
                    logger.warning("Error carregant anàlisi %s: %s", path, e)

            if results:
                merged = self._merge_sibling_samples(results)
                # Usar primari com a base
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
                logger.warning(f"Error comprovant anàlisi existent: {e}")

    def _load_existing_analysis(self, result):
        """Carrega una anàlisi existent."""
        self.samples_grouped = (result.get("samples_grouped")
                                or result.get("samples_analyzed", {}))
        if self.samples_grouped:
            self.main_window.processed_data = result  # B1: needed for method/seq_path
            self._populate_table()
            self._populate_charts(result)
            self._populate_sub_tabs(result)
            self.empty_state.setVisible(False)
            self.info_frame.setVisible(False)
            self.status_frame.setVisible(False)
            self.results_frame.setVisible(True)
            self._tab_widget.setVisible(True)
            self.main_window.set_status("Anàlisi carregada des de fitxer existent", 3000)
            self.analyze_completed.emit(result)

    def _populate_sub_tabs(self, result):
        """Propaga dades d'anàlisi als sub-tabs QC i Comparació."""
        try:
            self._qc_tab.populate(result)
        except Exception as e:
            logger.warning(f"Error populating QC tab: {e}")
        try:
            self._comparison_tab.populate(result)
        except Exception as e:
            logger.warning(f"Error populating Comparison tab: {e}")

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
            # Get rf_mass_cal + intercept GLOBAL (what quantify_sample actually uses)
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

            # Display: prioritzar calibració global (el que realment usa quantify_sample)
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

            # Build regression line text
            if intercept_display and abs(intercept_display) > 0.01:
                recta_str = f"RF=<b>{rf_display:.1f}</b> · b=<b>{intercept_display:.1f}</b>"
            else:
                recta_str = f"RF=<b>{rf_display:.1f}</b> · origen"

            self.cal_info.setText(
                f"<span style='color: #6c757d; font-size: 10px;'>CALIBRACIÓ</span><br>"
                f"<span style='color: #27AE60;'>✓</span> <b style='font-size: 13px;'>{cal_note}</b><br>"
                f"<span style='color: #6c757d; font-size: 10px;'>{recta_str}</span>"
            )

            # Tooltip: detalls complets per si vol aprofundir
            khp_conc = calibration_data.get("khp_conc", 0)
            shift = calibration_data.get("shift_direct", 0) or calibration_data.get("shift", 0)
            shift_sec = shift * 60 if shift else 0
            khp_source = calibration_data.get("khp_source", "LOCAL")
            rf_global_str = f"{rf_mass_global:.2f}" if rf_mass_global else "N/A"
            self.cal_info.setToolTip(
                f"Font: {khp_source}\n"
                f"Quantificació: {cal_note}\n"
                f"Recta: ppm = (A - {intercept_display:.1f}) × 1000 / (RF × V)\n"
                f"RF_mass_cal (global): {rf_global_str}\n"
                f"Intercept (global): {intercept_global:.2f}\n"
                f"KHP SEQ: {khp_conc:g}ppm, shift={shift_sec:.1f}s"
            )
        else:
            self.cal_info.setText(
                f"<span style='color: #6c757d; font-size: 10px;'>CALIBRACIÓ</span><br>"
                f"<span style='color: #E67E22;'>⚠</span> <span style='color: #856404;'>No disponible</span><br>"
                f"<span style='color: #6c757d; font-size: 10px;'>S'usaran valors per defecte</span>"
            )
            self.cal_info.setToolTip("No hi ha calibració disponible")

        if not self.samples_grouped:
            self.status_indicator.setText(
                f"<span style='background-color: #d4edda; color: #155724; "
                f"padding: 4px 12px; border-radius: 12px; font-size: 11px;'>"
                f"Llest per analitzar</span>"
            )
        # Si ja hi ha resultats, status_indicator s'actualitza des de _populate_table
        self.analyze_btn.setEnabled(True)

    # ------------------------------------------------------------------
    # Run analysis
    # ------------------------------------------------------------------

    def _run_analyze(self):
        """Executa l'anàlisi."""
        # Detectar mode siblings
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

        # Auto-load if not in memory
        if not imported_data and seq_path:
            from hpsec_import import import_from_manifest
            self.main_window.set_status("Carregant dades d'importació...")
            try:
                imported_data = import_from_manifest(seq_path)
            except Exception as e:
                logger.warning(f"Error carregant import: {e}")
                imported_data = None
            if imported_data and imported_data.get('success'):
                self.main_window.imported_data = imported_data

        # ensure_data_loaded() es fa dins del AnalyzeWorker (thread)
        # per no bloquejar la UI si cal llegir MasterFile + CSV + Export3D

        if not calibration_data and seq_path:
            import json
            cal_path = Path(seq_path) / "CHECK" / "data" / "calibration_result.json"
            if cal_path.exists():
                self.main_window.set_status("Carregant dades de calibració...")
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

                    # Reconstruct full calibration_data matching calibrate_panel output
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
            QMessageBox.warning(self, "Avís", "No s'han trobat dades d'importació.")
            self.analyze_completed.emit({'success': False, 'errors': ["No hi ha dades d'importació"]})
            return

        samples = imported_data.get("samples", {})
        if not samples:
            QMessageBox.warning(self, "Avís", "No s'han trobat mostres a les dades importades.")
            self.analyze_completed.emit({'success': False, 'errors': ["No hi ha mostres a les dades"]})
            return

        self.analyze_btn.setEnabled(False)
        self.empty_state.setVisible(False)
        self.status_frame.setVisible(False)
        self.progress_frame.setVisible(True)
        self.progress_bar.setValue(0)
        self.results_frame.setVisible(False)
        self._tab_widget.setVisible(False)

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
        """Gestiona la finalització de l'anàlisi."""
        if self.worker is not None:
            self.worker.wait()
        self.progress_frame.setVisible(False)
        self.analyze_btn.setEnabled(True)

        if not result or not result.get("success"):
            error_msg = result.get("error", "Error desconegut") if result else "Resultat buit"
            # Mostrar error inline (visible i persistent)
            self._show_inline_message(error_msg, level="error")
            self._update_status()
            self.analyze_completed.emit(result or {"success": False, "error": error_msg})
            return

        self.main_window.processed_data = result
        self.samples_grouped = result.get("samples_grouped", {})

        save_analysis_result(result)

        # Feedback visual mentre es preparen taula + gràfics (pot trigar 3-10s)
        n_samples = len(self.samples_grouped)
        self.progress_label.setText(f"Preparant taula ({n_samples} mostres)...")
        self.progress_bar.setValue(95)
        self.progress_frame.setVisible(True)
        from PySide6.QtWidgets import QApplication
        QApplication.processEvents()

        self._populate_table()
        self.results_frame.setVisible(True)

        self.progress_label.setText("Generant gràfics...")
        self.progress_bar.setValue(97)
        QApplication.processEvents()

        self._populate_charts(result)
        self._populate_sub_tabs(result)
        self._tab_widget.setVisible(True)

        self.progress_frame.setVisible(False)
        self.status_frame.setVisible(False)
        self.analyze_completed.emit(result)

    def _on_error(self, error_msg):
        logger.error(f"Error durant anàlisi: {error_msg}")
        self.progress_frame.setVisible(False)
        self.analyze_btn.setEnabled(True)
        # Mostrar error inline en lloc de QMessageBox
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
        logger.info("Anàlisi siblings: %d carpetes (%s)", n, ", ".join(names))

        self.analyze_btn.setEnabled(False)
        self.empty_state.setVisible(False)
        self.status_frame.setVisible(False)
        self.progress_frame.setVisible(True)
        self.progress_bar.setValue(0)
        self.results_frame.setVisible(False)
        self._tab_widget.setVisible(False)
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

        # Fusionar samples_grouped de tots els siblings
        merged = self._merge_sibling_samples(results)

        if not merged:
            self._show_inline_message("Cap sibling analitzat correctament.", level="error")
            self.analyze_completed.emit({"success": False, "error": "Cap sibling OK"})
            return

        # Construir resultat unificat (usar el primari com a base)
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

        # Preparar taula + gràfics (mateixa lògica que _on_finished)
        from PySide6.QtWidgets import QApplication
        n_samples = len(merged)
        self.progress_label.setText(f"Preparant taula ({n_samples} mostres)...")
        self.progress_bar.setValue(95)
        self.progress_frame.setVisible(True)
        QApplication.processEvents()

        self._populate_table()
        self.results_frame.setVisible(True)

        self.progress_label.setText("Generant gràfics...")
        self.progress_bar.setValue(97)
        QApplication.processEvents()

        self._populate_charts(base_result)
        self._populate_sub_tabs(base_result)
        self._tab_widget.setVisible(True)

        self.progress_frame.setVisible(False)
        self.status_frame.setVisible(False)

        n = len(results)
        n_ok = sum(1 for r in results.values() if r.get("success"))
        self.main_window.set_status(
            f"Anàlisi completada: {n_ok}/{n} carpetes, {n_samples} mostres", 5000
        )

        self.analyze_completed.emit(base_result)

    def _merge_sibling_samples(self, results):
        """Fusiona samples_grouped de N siblings en un dict unificat.

        Mostres amb el MATEIX nom es fusionen en una sola entrada amb
        rèpliques renumerades seqüencialment. Cada rèplica porta
        metadades d'origen (_source_path, _source_label).

        KHP/BLANK/CONTROL NO es fusionen entre siblings (condicions
        de calibració pròpies). Es diferencien amb suffix [A]/[B].

        Exemples:
            LQ0468 SEQ_A R1,R2 + SEQ_B R1,R2
            → LQ0468 amb R1,R2 (d'A) + R3,R4 (de B)
            Columna Inj: "5, 6, 3B, 4B"
        """
        import os
        merged = {}
        # Tipus que NO es fusionen (cada sibling té condicions pròpies)
        no_merge_types = {"KHP", "BLANK", "CONTROL"}

        for path, result in results.items():
            if not result.get("success"):
                continue

            suffix = self._get_sibling_suffix(path)
            label = suffix if suffix else "A"
            samples = result.get("samples_grouped", {})

            for name, data in samples.items():
                sample_type = data.get("sample_type", "SAMPLE")

                # Marcar traçabilitat a cada rèplica
                replicas = data.get("replicas", {})
                for rk, rd in replicas.items():
                    if isinstance(rd, dict):
                        rd["_source_path"] = path
                        rd["_source_label"] = label

                if sample_type in no_merge_types:
                    # KHP/BLANK/CONTROL: no fusionar, afegir suffix
                    display = f"{name} [{label}]" if len(results) > 1 else name
                    if display in merged:
                        display = f"{name} [{os.path.basename(path)}]"
                    data["_source_path"] = path
                    data["_source_label"] = label
                    merged[display] = data
                elif name in merged:
                    # SAMPLE ja existent → fusionar rèpliques
                    existing = merged[name]
                    existing_reps = existing.get("replicas", {})
                    # Trobar el número de rèplica màxim existent
                    max_rep = max(
                        (int(k) for k in existing_reps if str(k).isdigit()),
                        default=0
                    )
                    # Afegir noves rèpliques amb números seqüencials
                    for rk in sorted(replicas.keys()):
                        max_rep += 1
                        new_key = str(max_rep)
                        existing_reps[new_key] = replicas[rk]
                else:
                    # Primera vegada que apareix → afegir directament
                    data["_source_path"] = path
                    data["_source_label"] = label
                    merged[name] = data

        return merged

    @staticmethod
    def _get_sibling_suffix(path):
        """Extreu el suffix del sibling ('' per primari, 'B' per 282B_SEQ, etc)."""
        import os, re
        name = os.path.basename(path)
        # Extreure número + suffix: 282_SEQ → ('282', ''), 282B_SEQ → ('282', 'B')
        clean = name.replace("_SEQ", "").replace("_BP", "").replace("_CAL", "")
        clean = clean.rstrip("_")
        m = re.match(r'^(\d+)([A-Z]?)$', clean)
        if m:
            return m.group(2)  # '' o 'B', 'C', etc
        return ""

    # ------------------------------------------------------------------
    # (SEQ_CAL regression removed — ara a GlobalCalibrationPanel, tab 5)
    # ------------------------------------------------------------------

    # === REMOVED: 14 mètodes SEQ_CAL (~950 línies) ===
    # _build_seq_cal_regression_section, _on_calibration_data_updated,
    # _check_and_show_seq_cal, _check_uib_sensitivity_mixing,
    # _run_seq_cal_regression, _update_seq_cal_ui, _populate_seq_cal_table,
    # _on_seq_cal_row_clicked, _on_seq_cal_point_toggled,
    # _on_seq_cal_signal_changed, _on_seq_cal_recalculate,
    # _on_seq_cal_repair_toggled, _update_seq_cal_comparison,
    # _update_seq_cal_graph
    # ==================================================


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

        # Separar mostres per tipologia (KHP exclòs — ja analitzat a Verificar)
        sample_names = []   # SAMPLE + PR (mostres reals + patrons)
        blank_names = []    # BLANK / MQ
        control_names = []  # CONTROL / Neteja (anàlisi lleugera)

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

        # Ordenar per índex d'injecció mínim (ordre cronològic).
        # Per mostres fusionades (siblings), la primera injecció determina la posició.
        def _sort_key(name):
            sg = self.samples_grouped[name]
            reps = sg.get("replicas", {})
            # Construir tuples (sibling_order, injection_index) per cada rèplica
            keys = []
            for r in reps.values():
                if not isinstance(r, dict):
                    continue
                label = r.get("_source_label", "A")
                idx = r.get("injection_index", 999)
                # A=0, B=1, C=2... per ordenar siblings
                sib_order = 0 if label in ("", "A") else ord(label) - ord("A")
                keys.append((sib_order, idx))
            return min(keys) if keys else (0, 999)

        for lst in (sample_names, blank_names, control_names):
            lst.sort(key=_sort_key)

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

            # BLANK: una fila per injecció (no agrupat per rèplica)
            # Rendering idèntic a mostres regulars, però cada injecció per separat
            if is_blank:
                replicas = sample_data.get("replicas") or {}
                quantification = sample_data.get("quantification") or {}
                comparison = sample_data.get("comparison") or {}
                for rep_key in sorted(replicas.keys()):
                    rep_data = replicas[rep_key]
                    row = self.results_table.rowCount()
                    self.results_table.insertRow(row)

                    # Col 0: Nom amb R si >1 rèplica
                    display_name = f"{sample_name} R{rep_key}" if len(replicas) > 1 else sample_name
                    item_name = QTableWidgetItem(display_name)
                    item_name.setData(Qt.UserRole, sample_name)
                    self.results_table.setItem(row, 0, item_name)

                    # Col 1: Inj
                    idx = rep_data.get("injection_index")
                    inj_item = QTableWidgetItem(str(idx) if idx is not None else "-")
                    inj_item.setForeground(QBrush(QColor("#888")))
                    self.results_table.setItem(row, 1, inj_item)

                    # Col 2-3: Sense selectors (cada injecció es mostra directament)
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

                    # Col 11-12: No R2 (sense comparació rèpliques)
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

                    # Col 14-15: Estat + Reparació
                    (status_color, status_text, s_tooltip,
                     repair_color, repair_text, r_tooltip) = self._classify_sample_status(
                        rep_data, rep_data, comparison, sample_data=sample_data)
                    status_item = QTableWidgetItem(status_text)
                    status_item.setForeground(QBrush(QColor(status_color)))
                    status_item.setToolTip(s_tooltip)
                    self.results_table.setItem(row, 14, status_item)
                    repair_item = QTableWidgetItem(repair_text)
                    repair_item.setForeground(QBrush(QColor(repair_color)))
                    repair_item.setToolTip(r_tooltip)
                    self.results_table.setItem(row, 15, repair_item)

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

            # Col 1: Inj (injection indices amb suffix sibling per rèplica)
            # Cada rèplica pot venir d'un sibling diferent: "5, 6, 3B, 4B"
            inj_indices = []
            tip_parts = []
            for rk, rd in sorted(replicas.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 999):
                idx = rd.get("injection_index")
                rep_label = rd.get("_source_label", "") if isinstance(rd, dict) else ""
                rep_suffix = rep_label if rep_label and rep_label != "A" else ""
                if idx is not None:
                    inj_indices.append(f"{idx}{rep_suffix}")
                import os as _os
                src = rd.get("_source_path", "") if isinstance(rd, dict) else ""
                src_name = _os.path.basename(src) if src else ""
                if idx is not None:
                    tip_parts.append(
                        f"R{rk}: inj #{idx}{rep_suffix}"
                        + (f" ({src_name})" if src_name else "")
                    )
            inj_text = ", ".join(inj_indices) if inj_indices else "-"
            inj_item = QTableWidgetItem(inj_text)
            inj_item.setForeground(QBrush(QColor("#888")))
            if tip_parts:
                inj_item.setToolTip("\n".join(tip_parts))
            self.results_table.setItem(row, 1, inj_item)

            # Col 2: Sel DOC — replica selector with (s) for suggested + "Cap" option
            doc_combo = QComboBox()
            doc_combo.setStyleSheet("QComboBox { border: none; background: transparent; padding: 2px; }")
            for rep_num in sorted(replicas.keys(), key=lambda x: int(x) if x.isdigit() else 999):
                label = f"R{rep_num}"
                # Mostrar injecció + sibling si ve d'un sibling
                rep_data = replicas.get(rep_num, {})
                if isinstance(rep_data, dict):
                    _idx = rep_data.get("injection_index")
                    _slabel = rep_data.get("_source_label", "")
                    _ssuffix = _slabel if _slabel and _slabel != "A" else ""
                    if _idx is not None:
                        label += f" (inj {_idx}{_ssuffix})"
                if rep_num == doc_rec:
                    label += " ★"
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

            # Col 3: Sel DAD — replica selector with (s) for suggested + "Cap" option
            dad_combo = QComboBox()
            dad_combo.setStyleSheet("QComboBox { border: none; background: transparent; padding: 2px; }")
            for rep_num in sorted(replicas.keys(), key=lambda x: int(x) if x.isdigit() else 999):
                label = f"R{rep_num}"
                rep_data = replicas.get(rep_num, {})
                if isinstance(rep_data, dict):
                    _idx = rep_data.get("injection_index")
                    _slabel = rep_data.get("_source_label", "")
                    _ssuffix = _slabel if _slabel and _slabel != "A" else ""
                    if _idx is not None:
                        label += f" (inj {_idx}{_ssuffix})"
                if rep_num == dad_rec:
                    label += " ★"
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

            # Col 11: R²_DOC (min pairwise si >1 parella)
            pairwise = sample_data.get("pairwise_comparisons", {})
            if pairwise and len(pairwise) > 1:
                r2_doc = min(
                    (c.get("doc", {}).get("pearson", 0) for c in pairwise.values()),
                    default=0)
            else:
                r2_doc = comparison.get("doc", {}).get("pearson", 0) if comparison else 0
            r2_doc_item = QTableWidgetItem(f"{r2_doc:.4f}" if r2_doc > 0 else "-")
            if 0 < r2_doc < 0.990:
                r2_doc_item.setForeground(QBrush(QColor(COLOR_WARNING)))
            self.results_table.setItem(row, 11, r2_doc_item)

            # Col 12: R²_DAD (min across wavelengths, min pairwise si >1 parella)
            if pairwise and len(pairwise) > 1:
                dad_comp = min(
                    (c.get("dad", {}) for c in pairwise.values()),
                    key=lambda d: d.get("pearson_min", 0),
                    default={})
            else:
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

            # Col 14-15: Estat + Reparació
            (status_color, status_text, s_tooltip,
             repair_color, repair_text, r_tooltip) = self._classify_sample_status(
                doc_rep, dad_rep, comparison, sample_data=sample_data)
            status_item = QTableWidgetItem(status_text)
            status_item.setForeground(QBrush(QColor(status_color)))
            status_item.setToolTip(s_tooltip)
            self.results_table.setItem(row, 14, status_item)
            repair_item = QTableWidgetItem(repair_text)
            repair_item.setForeground(QBrush(QColor(repair_color)))
            repair_item.setToolTip(r_tooltip)
            self.results_table.setItem(row, 15, repair_item)

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

                # Col 14-15: Neteja / sense reparació
                type_item = QTableWidgetItem("Neteja")
                type_item.setForeground(QBrush(QColor("#888888")))
                self.results_table.setItem(row, 14, type_item)
                self.results_table.setItem(row, 15, QTableWidgetItem(""))

                # Light grey background
                light_bg = QBrush(QColor("#F0F0F0"))
                for c in range(n_cols):
                    item = self.results_table.item(row, c)
                    if item:
                        item.setBackground(light_bg)

                n_control += 1

        # Update stats
        total = n_ok + n_warning + n_error
        parts = [f"<b>{total}</b> mostres"]
        if n_blank > 0:
            parts.append(f"{n_blank} blancs")
        if n_control > 0:
            parts.append(f"{n_control} neteja")
        counts = " &middot; ".join(parts)

        status_parts = []
        status_parts.append(f"<span style='color:#27AE60'>*</span>&nbsp;{n_ok}")
        if n_warning > 0:
            status_parts.append(f"<span style='color:#F39C12'>*</span>&nbsp;{n_warning}")
        if n_error > 0:
            status_parts.append(f"<span style='color:#E74C3C'>*</span>&nbsp;{n_error}")
        status_str = " &nbsp;".join(status_parts)

        self.status_indicator.setText(
            f"<span style='background-color: #f8f9fa; color: #2c3e50; "
            f"padding: 4px 12px; border-radius: 12px; font-size: 11px;'>"
            f"{counts} &nbsp;|&nbsp; {status_str}</span>"
        )

    # ------------------------------------------------------------------
    # Anomaly severity classification
    # ------------------------------------------------------------------

    def _classify_sample_status(self, doc_rep_data, dad_rep_data, comparison,
                                sample_data=None):
        """Classifica l'estat d'una mostra: anomalies (col 14) + reparació (col 15).

        Usa ANOMALY_CATALOG com a font de veritat per severitat, icones i labels.
        Separa anomalies generals de l'estat de reparació irregular_top.

        Returns (status_color, status_text, status_tooltip,
                 repair_color, repair_text, repair_tooltip).
        """
        # --- Defaults reparació ---
        repair_color = "#888"
        repair_text = ""
        repair_tooltip = ""

        # Comprovar si l'usuari ha seleccionat "Cap"
        if sample_data:
            selected = sample_data.get("selected", {})
            if selected.get("doc") == "none":
                return (COLOR_ERROR, "NO VÀL",
                        "Usuari ha seleccionat 'Cap' — No es quantificarà ni exportarà",
                        repair_color, repair_text, repair_tooltip)
            if sample_data.get("sample_valid") is False and not sample_data.get("repaired"):
                reason = (sample_data.get("recommendation", {})
                          .get("doc", {}).get("reason", "Ambdues rèpliques amb anomalies crítiques"))
                return (COLOR_ERROR, "NO VÀL",
                        f"Mostra no vàlida — {reason}\nSeleccionar 'Cap' o generar noves dades",
                        repair_color, repair_text, repair_tooltip)

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

        # Separar anomalies de reparació (IRREGULAR_TOP) de la resta
        repair_codes = {"IRREGULAR_TOP", "IRREGULAR_TOP_DIRECT", "IRREGULAR_TOP_UIB"}
        anomalies_general = [a for a in all_anomalies
                             if (a.get("code") if isinstance(a, dict) else str(a).split("_REPAIRED")[0])
                             not in repair_codes]
        anomalies_repair = [a for a in all_anomalies
                            if (a.get("code") if isinstance(a, dict) else str(a).split("_REPAIRED")[0])
                            in repair_codes]

        # --- COLUMNA ESTAT (col 14): anomalies generals ---
        classified = classify_anomalies(anomalies_general)
        timeout_info = doc_rep_data.get("timeout_info", {})
        timeout_severity = timeout_info.get("severity", "OK")
        n_timeouts = timeout_info.get("n_timeouts", 0)
        replica_warnings = []
        if comparison:
            for domain in ("doc", "dad"):
                replica_warnings.extend((comparison.get(domain) or {}).get("warnings", []))

        has_blocker = bool(classified["blocker"])
        has_warn = bool(classified["warning"]
                        or (timeout_severity in ("WARNING", "CRITICAL"))
                        or replica_warnings)

        n_blocker = len(classified["blocker"])
        n_warn = len(classified["warning"])

        if has_blocker:
            status_color = COLOR_ERROR
            first_blocker = classified["blocker"][0]
            b_code = first_blocker.get("code") if isinstance(first_blocker, dict) else str(first_blocker)
            b_entry = ANOMALY_CATALOG.get(b_code, {})
            b_label = (first_blocker.get("label") if isinstance(first_blocker, dict) else None
                       ) or b_entry.get("label", b_code)
            status_text = f"{b_label} +{n_blocker - 1}" if n_blocker > 1 else b_label
        elif has_warn:
            status_color = COLOR_WARNING
            parts = []
            if n_warn:
                parts.append(f"{n_warn} avís" if n_warn == 1 else f"{n_warn} avisos")
            if n_timeouts > 0:
                parts.append(f"{n_timeouts} timeout")
            status_text = " \u00b7 ".join(parts)
        else:
            status_color = COLOR_SUCCESS
            status_text = "OK"

        # Tooltip anomalies
        tooltip_parts = []
        for key, label_prefix in [("blocker", "CRÍTIC"), ("warning", "Avís"), ("info", "Info")]:
            for a in classified[key]:
                code = a.get("code") if isinstance(a, dict) else str(a)
                entry = ANOMALY_CATALOG.get(code, {})
                lbl = (a.get("label") if isinstance(a, dict) else None) or entry.get("label", code)
                det = a.get("details", {}) if isinstance(a, dict) else {}
                if det.get("snr"):
                    lbl += f" (SNR={det['snr']:.1f})"
                line = f"{label_prefix}: {lbl}"
                action = entry.get("action", "")
                if action:
                    line += f"\n   \u2192 {action}"
                tooltip_parts.append(line)
        if n_timeouts > 0:
            zones = timeout_info.get("zones", [])
            tooltip_parts.append(
                f"Timeouts Direct: {n_timeouts} ({timeout_severity}) — zones: {', '.join(zones) if zones else '?'}")
            uib_ti = doc_rep_data.get("timeout_info_uib") or {}
            if uib_ti.get("n_timeouts", 0) > 0:
                uib_zones = uib_ti.get("zones", [])
                uib_in_peak = doc_rep_data.get("timeout_in_peak_uib", False)
                uib_tip = f"Timeouts UIB: {uib_ti['n_timeouts']} — zones: {', '.join(uib_zones) if uib_zones else '?'}"
                if uib_in_peak:
                    uib_tip += " — DINS DEL PIC UIB!"
                tooltip_parts.append(uib_tip)
        if replica_warnings:
            for rw in replica_warnings:
                tooltip_parts.append(rw.get("label", rw.get("code", str(rw))) if isinstance(rw, dict) else str(rw))
        status_tooltip = "\n".join(tooltip_parts) if tooltip_parts else "OK"

        # --- COLUMNA REPARACIÓ (col 15): irregular_top ---
        if anomalies_repair:
            classified_r = classify_anomalies(anomalies_repair)
            n_repaired = len(classified_r["repaired"])
            n_pending = len(classified_r["blocker"]) + len(classified_r["warning"])
            can_repair = (sample_data and sample_data.get("repairable")
                          and not sample_data.get("repaired"))

            if n_repaired > 0 and n_pending == 0:
                repair_color = COLOR_SUCCESS
                repair_text = f"✓ ({n_repaired})"
                repair_tooltip = "Reparació aplicada — Clic per desfer o veure detalls"
            elif n_pending > 0:
                repair_color = COLOR_ERROR
                repair_text = "🔧" if can_repair else "⚠"
                repair_tooltip = "Cim irregular detectat — Clic per revisar i reparar"
            # Afegir detalls per cada rèplica
            rp_details = []
            for a in anomalies_repair:
                code = a.get("code", "") if isinstance(a, dict) else str(a)
                repaired = a.get("repaired", False) if isinstance(a, dict) else "_REPAIRED" in str(a)
                det = a.get("details", {}) if isinstance(a, dict) else {}
                depth = det.get("max_depth", 0)
                n_v = det.get("n_valleys", 0)
                signal = "Direct" if "DIRECT" in code else ("UIB" if "UIB" in code else "DOC")
                state = "reparat" if repaired else "pendent"
                rp_details.append(f"{signal}: {n_v} valls (prof. {depth:.1%}) — {state}")
            if rp_details:
                repair_tooltip = "\n".join(rp_details) + "\n\nClic per obrir diàleg de reparació"

        return (status_color, status_text, status_tooltip,
                repair_color, repair_text, repair_tooltip)

    # ------------------------------------------------------------------
    # Replica change (separate DOC / DAD handlers)
    # ------------------------------------------------------------------

    def _on_doc_replica_changed(self, sample_name):
        """Gestiona el canvi de rèplica DOC (inclou opció 'Cap')."""
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
                # Marcar mostra com no vàlida per DOC
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
        """Gestiona el canvi de rèplica DAD (inclou opció 'Cap')."""
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
        """Actualitza columnes DOC (4-8) quan canvia la r\u00e8plica DOC."""
        sample_data = self.samples_grouped[sample_name]
        selected = sample_data.get("selected", {})
        doc_sel = selected.get("doc", "1")
        replicas = sample_data.get("replicas", {})

        # "Cap" seleccionat → buidar columnes
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
        """Actualitza columnes DAD (9-10) quan canvia la rèplica DAD."""
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
        """Actualitza les columnes Estat (col 14) i Reparació (col 15)."""
        sample_data = self.samples_grouped[sample_name]
        selected = sample_data.get("selected", {})
        replicas = sample_data.get("replicas", {})
        comparison = sample_data.get("comparison") or {}
        doc_rep = replicas.get(selected.get("doc", "1"), {})
        dad_rep = replicas.get(selected.get("dad", "1"), {})

        (status_color, status_text, s_tooltip,
         repair_color, repair_text, r_tooltip) = self._classify_sample_status(
            doc_rep, dad_rep, comparison, sample_data=sample_data)
        status_item = self.results_table.item(row, 14)
        if status_item:
            status_item.setText(status_text)
            status_item.setForeground(QBrush(QColor(status_color)))
            status_item.setToolTip(s_tooltip)
        repair_item = self.results_table.item(row, 15)
        if repair_item:
            repair_item.setText(repair_text)
            repair_item.setForeground(QBrush(QColor(repair_color)))
            repair_item.setToolTip(r_tooltip)

    # ------------------------------------------------------------------
    # Quantification recalculation
    # ------------------------------------------------------------------

    def _update_quantification(self, sample_name):
        """Recalcula la quantificació per una mostra."""
        try:
            from hpsec_analyze import quantify_sample
            from hpsec_calibrate import get_all_active_calibrations

            sample_data = self.samples_grouped[sample_name]

            # Respectar exclusió de quantificació
            if sample_data.get("skip_quantification"):
                sample_data["quantification"] = {
                    "concentration_ppm": None,
                    "concentration_ppm_direct": None,
                    "concentration_ppm_uib": None,
                    "valid": False,
                    "reason": sample_data["quantification"].get("reason",
                              "Exclosa de quantificació") if sample_data.get("quantification") else
                              "Exclosa de quantificació"
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
                # Propagar HCI de la rèplica seleccionada
                hci = selected_replica.get("hci")
                if hci is not None:
                    quantification["hci"] = hci
                    quantification["hci_character"] = selected_replica.get("hci_character", "")
                sample_data["quantification"] = quantification
        except Exception as e:
            logger.error(f"Error recalculant quantificació: {e}")
            self.main_window.set_status(f"Error quantificació: {e}", 5000)

    # ------------------------------------------------------------------
    # Table interaction
    # ------------------------------------------------------------------

    def _on_table_cell_click(self, row, col):
        """Handler per clic a cel·la — col 15 (Rpar.) obre diàleg reparació si aplicable."""
        if col != 15:
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

        if len(targets) == 1:
            rep_key, signal_type = targets[0]
            self._open_repair_dialog(sample_name, rep_key, signal_type)
        else:
            # Multiple targets — show context menu
            from PySide6.QtWidgets import QMenu
            from PySide6.QtGui import QCursor
            menu = QMenu(self)
            for rep_key, signal_type in targets:
                label = f"R{rep_key} {signal_type.upper()}"
                action = menu.addAction(label)
                action.triggered.connect(
                    lambda _checked, rk=rep_key, st=signal_type:
                    self._open_repair_dialog(sample_name, rk, st)
                )
            menu.exec(QCursor.pos())

    def _find_repair_targets(self, sample_name):
        """Busca rèpliques/senyals amb anomalies de cim irregular (pendents, reparades o dismissed)."""
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

    def _open_repair_dialog(self, sample_name, rep_key, signal_type):
        """Obre el diàleg de reparació de pic irregular."""
        sample_data = self.samples_grouped.get(sample_name)
        if not sample_data:
            return

        method = "COLUMN"
        if self.main_window.processed_data:
            method = self.main_window.processed_data.get("method", "COLUMN")

        dialog = JaggedPeakRepairDialog(
            sample_name, sample_data, rep_key, signal_type, method, parent=self
        )
        dialog.repair_applied.connect(self._on_repair_action)
        dialog.repair_undone.connect(self._on_repair_action)
        dialog.dismissed.connect(self._on_repair_action)
        dialog.reactivated.connect(self._on_repair_action)
        dialog.exec()

    def _on_repair_action(self, sample_name, rep_key, signal_type):
        """Actualitza la taula després d'una acció de reparació."""
        row = self._sample_row_map.get(sample_name)
        if row is not None:
            self._update_quantification(sample_name)
            self._update_doc_columns(row, sample_name)
            self._update_estat_column(row, sample_name)

    def _on_table_double_click(self, index):
        """Handler per doble clic — obre SampleDetailDialog per totes les tipologies."""
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
        """Mostra el diàleg de detall (no-modal per evitar pèrdua de finestra)."""
        if sample_name not in self.samples_grouped:
            return
        # Tancar diàleg anterior si existeix
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
        self._detail_dialog = dialog  # Mantenir referència
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.finished.connect(lambda: self._on_detail_closed(sample_name))
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _on_detail_closed(self, sample_name):
        """Actualitza taula després de tancar el diàleg de detall."""
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
        """Genera el report PDF d'anàlisi (cridat des del wizard header)."""
        processed_data = self.main_window.processed_data
        if not processed_data:
            QMessageBox.warning(self, "Avís", "No hi ha dades processades.")
            return

        seq_path = processed_data.get("seq_path", "")
        if not seq_path:
            QMessageBox.warning(self, "Avís", "No s'ha trobat el path de la seqüència.")
            return

        try:
            from generate_analysis_report import generate_analysis_report

            # Passar dades en memòria (inclou seleccions actuals de l'usuari)
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
        """Prepara dades pels gràfics i mostra la secció."""
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
                # Expandir BLANK per injecció (una entrada per rèplica)
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

        # Charts sempre visibles — dibuixar directament
        self._charts_initialized = True
        self._redraw_charts()

    def _build_sample_checkboxes(self, regular, blank, control, khp):
        """Registra mostres per categoria (sense checkboxes individuals)."""
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
        """Un botó de categoria ha canviat — actualitza taula i gràfics."""
        self._update_cat_btn_styles()
        if self.samples_grouped:
            self._populate_table()
        self._redraw_charts()

    def _on_wl_changed(self):
        """Longitud d'ona DAD seleccionada ha canviat — redibuixar gràfics DAD."""
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
        """Redibuixa els 4 gràfics amb les mostres seleccionades."""
        if not HAS_MATPLOTLIB:
            return
        checked = self._get_checked_samples()
        # BLANK va amb regular (fraccions+ppm complets), només CONTROL va a light
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
        zone_injections = {}  # zona → [inj_idx, ...]
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
    def _chart_sorted_names(samples_dict):
        """Ordena noms de mostra per sibling + injection_index (cronològic)."""
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
        """Retorna etiqueta curta per eix X: índex injecció + suffix sibling.

        Exemples: "5" (primari/A), "5B" (sibling B), "5C" (sibling C).
        La label ve de la rèplica seleccionada (pot ser d'un sibling diferent).
        """
        sel = (data.get("selected") or {}).get("doc", "1")
        rep = (data.get("replicas") or {}).get(sel, {})
        idx = rep.get("injection_index")
        if idx is not None:
            label = rep.get("_source_label", "") if isinstance(rep, dict) else ""
            suffix = label if label and label != "A" else ""
            return f"{idx}{suffix}"
        # Truncar nom si massa llarg
        return name[:12] + "…" if len(name) > 12 else name

    @staticmethod
    def _setup_bar_hover(figure, canvas, ax, x_positions, full_names, values_per_bar):
        """Configura tooltip hover sobre barres del gràfic.

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
            # Trobar la barra més propera
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
        """Gràfic DOC: barres stacked per fraccions (COLUMN) o simples (BP)."""
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
            # BP no té fraccions — usar total directament
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
        """Gràfic DAD per mostra: barres stacked per fraccions (COLUMN) o simples (BP)."""
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
            # BP no té fraccions — usar total directament
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
        """Retorna l'estil de línia segons el tipus de mostra."""
        st = data.get("sample_type", "SAMPLE")
        if st == "KHP":
            return '--'  # KHP: discontínua
        elif st in ("BLANK", "CONTROL"):
            return ':'   # Blanc/Control: punts
        elif st.startswith("PR"):
            return '-.'  # PR: punt-ratlla
        return '-'       # Mostres: sòlida

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
        """Guarda els 5 gràfics a SEQ/CHECK/plots/."""
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

