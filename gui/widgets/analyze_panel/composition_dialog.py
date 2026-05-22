# -*- coding: utf-8 -*-
"""
HPSEC Suite - Timeout Composition Dialog (v4)
==============================================

Composar cromatograma net combinant segments de dues rèpliques.

Layout:
  1. Header + status
  2. Barra de segments (Qt widgets): per cada tram [R1 | R2 | Interp] + límits
     + botons afegir/eliminar
  3. Gràfic cromatograma (només visualització, zoom/pan via toolbar)
  4. Gràfic barres fraccions (R1 vs R2 vs Compost + àrees + ppm)
  5. Botons aplicar/tancar
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame,
    QSizePolicy, QMessageBox, QWidget, QComboBox, QDoubleSpinBox,
    QScrollArea, QGridLayout
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QColor

import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# Colors
_SRC_COLORS = {
    "1": "#2E86AB",   # blue — R1
    "2": "#E67E22",   # orange — R2
    "interp": "#888888",  # grey — interpolation
}
_COMPOSED_COLOR = "#555555"
_TIMEOUT_COLORS = {"r1": "#E74C3C", "r2": "#F39C12"}

FRACTION_COLORS = {
    "BioP": "#3498DB",
    "HS":   "#E74C3C",
    "BB":   "#F39C12",
    "SB":   "#2ECC71",
    "LMW":  "#9B59B6",
}
FRACTION_ORDER = ["BioP", "HS", "BB", "SB", "LMW"]


class TimeoutCompositionDialog(QDialog):
    """Dialeg per composar cromatograma a partir de segments de dues repliques."""

    composition_completed = Signal(str)  # sample_name
    navigate_requested = Signal(int)  # direction: -1 prev, +1 next

    def __init__(self, sample_name, sample_data, is_bp=False, parent=None):
        super().__init__(parent)
        self.sample_name = sample_name
        self.sample_data = sample_data
        self.is_bp = is_bp

        self.setWindowTitle(f"Composicio timeout — {sample_name}")
        self.setMinimumSize(700, 500)
        self.resize(950, 650)

        replicas = sample_data.get("replicas", {})
        self.rep_keys = sorted(replicas.keys())
        if len(self.rep_keys) < 2:
            return

        self.r1 = replicas[self.rep_keys[0]]
        self.r2 = replicas[self.rep_keys[1]]
        self.t1 = np.asarray(self.r1.get("t_doc", []), dtype=float)
        self.y1 = np.asarray(self.r1.get("y_doc_net", []), dtype=float)
        self.t2 = np.asarray(self.r2.get("t_doc", []), dtype=float)
        self.y2 = np.asarray(self.r2.get("y_doc_net", []), dtype=float)
        self.ti1 = self.r1.get("timeout_info", {})
        self.ti2 = self.r2.get("timeout_info", {})

        self._t_max = float(max(
            self.t1[-1] if len(self.t1) else 70,
            self.t2[-1] if len(self.t2) else 70))

        from hpsec_core import check_timeout_composability
        run_dur = 12.0 if is_bp else 70.0
        self.comp_result = check_timeout_composability(
            self.ti1, self.ti2, run_duration_min=run_dur)
        self.segments = [s.copy() for s in self.comp_result.get("segments", [])]

        self._composed_y = None
        self._composed_t = None
        self._updating_widgets = False  # guard against feedback loops

        self._setup_ui()
        self._update_preview()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(3)

        nav_s = ("QPushButton { border: 1px solid #CED4DA; border-radius: 3px;"
                 " padding: 3px 8px; font-size: 11px; }"
                 "QPushButton:hover { background: #E9ECEF; }")

        # === TOP BAR: nav + apply + method + name + close ===
        top_row = QHBoxLayout()
        top_row.setSpacing(4)

        prev_btn = QPushButton("\u25c0")
        prev_btn.setStyleSheet(nav_s)
        prev_btn.setFixedWidth(28)
        prev_btn.clicked.connect(lambda: self.navigate_requested.emit(-1))
        top_row.addWidget(prev_btn)

        self.apply_btn = QPushButton("Aplicar")
        self.apply_btn.setStyleSheet(
            "QPushButton { border: 1px solid #27AE60; border-radius: 3px;"
            " padding: 3px 10px; font-size: 11px; color: white;"
            " background: #27AE60; font-weight: bold; }"
            "QPushButton:hover { background: #219A52; }")
        self.apply_btn.clicked.connect(self._on_apply)
        top_row.addWidget(self.apply_btn)

        top_row.addWidget(QLabel("<span style='color:#ccc'>|</span>"))

        # Method
        top_row.addWidget(QLabel("<b style='font-size:10px'>Metode:</b>"))
        from hpsec_core import COMPOSE_METHODS, COMPOSE_METHOD_DEFAULT
        self._method_combo = QComboBox()
        self._method_combo.setStyleSheet("font-size: 10px; padding: 1px 4px;")
        self._method_combo.setMaximumWidth(200)
        for key, label in COMPOSE_METHODS.items():
            self._method_combo.addItem(label, key)
            if key == COMPOSE_METHOD_DEFAULT:
                self._method_combo.setCurrentIndex(self._method_combo.count() - 1)
        self._method_combo.currentIndexChanged.connect(self._update_preview)
        top_row.addWidget(self._method_combo)

        # Status + name
        comp = self.comp_result
        if comp.get("composable"):
            status_icon = "\u2714"
            status_color = "#27AE60"
        else:
            status_icon = "\u2718"
            status_color = "#E74C3C"

        r1k, r2k = self.rep_keys
        c1, c2 = _SRC_COLORS["1"], _SRC_COLORS["2"]
        header_lbl = QLabel(
            f"<span style='color:{status_color}'>{status_icon}</span>"
            f" <b>{self.sample_name}</b>"
            f" <span style='font-size:10px;color:#888'>"
            f"<span style='color:{c1}'>R{r1k}</span>"
            f" <span style='color:{c2}'>R{r2k}</span></span>")
        header_lbl.setAlignment(Qt.AlignCenter)
        top_row.addWidget(header_lbl, 1)

        close_btn = QPushButton("Tancar")
        close_btn.setStyleSheet(nav_s)
        close_btn.clicked.connect(self.close)
        top_row.addWidget(close_btn)

        next_btn = QPushButton("\u25b6")
        next_btn.setStyleSheet(nav_s)
        next_btn.setFixedWidth(28)
        next_btn.clicked.connect(lambda: self.navigate_requested.emit(1))
        top_row.addWidget(next_btn)

        layout.addLayout(top_row)

        # =============================================================
        # SEGMENT CONTROL BAR
        # =============================================================
        seg_frame = QFrame()
        seg_frame.setStyleSheet(
            "QFrame#segFrame { background: #f5f6f8; border: 1px solid #ddd; "
            "border-radius: 4px; }")
        seg_frame.setObjectName("segFrame")
        seg_outer = QVBoxLayout(seg_frame)
        seg_outer.setContentsMargins(8, 6, 8, 6)
        seg_outer.setSpacing(4)

        # Header row: title + buttons
        seg_header = QHBoxLayout()
        seg_header.addWidget(QLabel(
            "<b style='font-size:11px'>Segments</b>"))
        seg_header.addStretch()

        btn_ss = ("QPushButton { padding: 3px 8px; font-size: 10px; "
                  "border: 1px solid #bbb; border-radius: 3px; background: white; }"
                  "QPushButton:hover { background: #e0e0e0; }")

        self._add_btn = QPushButton("+ Afegir")
        self._add_btn.setToolTip("Divideix el segment més llarg en dos")
        self._add_btn.setStyleSheet(btn_ss)
        self._add_btn.clicked.connect(self._split_longest_segment)
        seg_header.addWidget(self._add_btn)

        self._del_btn = QPushButton("\u2212 Eliminar últim")
        self._del_btn.setToolTip("Elimina l'últim límit (uneix els dos últims segments)")
        self._del_btn.setStyleSheet(btn_ss)
        self._del_btn.clicked.connect(self._remove_last_boundary)
        seg_header.addWidget(self._del_btn)

        self._reset_btn = QPushButton("Reiniciar")
        self._reset_btn.setToolTip("Torna als segments automàtics originals")
        self._reset_btn.setStyleSheet(btn_ss)
        self._reset_btn.clicked.connect(self._reset_segments)
        seg_header.addWidget(self._reset_btn)

        seg_outer.addLayout(seg_header)

        # Segment widgets container (rebuilt dynamically)
        self._seg_container = QWidget()
        self._seg_container_layout = QHBoxLayout(self._seg_container)
        self._seg_container_layout.setContentsMargins(0, 0, 0, 0)
        self._seg_container_layout.setSpacing(2)
        seg_outer.addWidget(self._seg_container)

        layout.addWidget(seg_frame)

        # =============================================================
        # CHROMATOGRAM PLOT
        # =============================================================
        if HAS_MATPLOTLIB:
            self._fig = Figure(figsize=(10, 3.5), dpi=100)
            self._fig.set_facecolor('white')
            self._ax = self._fig.add_subplot(111)
            self._canvas = FigureCanvas(self._fig)
            self._canvas.setMinimumHeight(200)
            self._canvas.setSizePolicy(
                QSizePolicy.Expanding, QSizePolicy.Expanding)

            self._toolbar = NavigationToolbar2QT(self._canvas, self)
            self._toolbar.setStyleSheet(
                "QToolBar { border: none; spacing: 2px; }"
                "QToolButton { padding: 2px; }")
            layout.addWidget(self._toolbar)
            layout.addWidget(self._canvas, stretch=60)

            # Cursor on hover
            self._canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
            self._cursor_label = QLabel("")
            self._cursor_label.setStyleSheet(
                "font-size: 9px; color: #888; font-family: monospace;")
            layout.addWidget(self._cursor_label)

            # FRACTION BARS PLOT eliminat — al pas Quantificar es veuen les
            # àrees finals. Aquí el focus és el cromatograma compost.
            self._bar_fig = None
            self._bar_ax = None
            self._bar_canvas = None

        # (Action buttons are in the top bar)

        # Build initial segment widgets
        self._rebuild_segment_widgets()

    # ------------------------------------------------------------------
    # Segment widget bar
    # ------------------------------------------------------------------

    def _rebuild_segment_widgets(self):
        """Reconstrueix la barra de widgets de segments."""
        # Clear existing
        while self._seg_container_layout.count():
            item = self._seg_container_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        self._seg_combos = []
        self._seg_spins_start = []
        self._seg_spins_end = []

        r1k, r2k = self.rep_keys

        for i, seg in enumerate(self.segments):
            # Segment card
            card = QFrame()
            src = seg.get("source", seg.get("chosen", "1"))
            card_color = _SRC_COLORS.get(src, '#ccc')
            card.setStyleSheet(
                f"QFrame {{ background: white; border: 2px solid {card_color}; "
                f"border-radius: 4px; }}")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(6, 4, 6, 4)
            card_layout.setSpacing(2)

            # Row 1: segment number + source combo
            top_row = QHBoxLayout()
            top_row.setSpacing(4)
            num_lbl = QLabel(f"<b>#{i+1}</b>")
            num_lbl.setStyleSheet("border: none; font-size: 10px; color: #666;")
            top_row.addWidget(num_lbl)

            combo = QComboBox()
            combo.setStyleSheet(
                "QComboBox { border: 1px solid #ccc; border-radius: 2px; "
                "padding: 2px 4px; font-size: 11px; font-weight: bold; }")
            combo.addItem(f"R{r1k}", "1")
            combo.addItem(f"R{r2k}", "2")
            combo.addItem("Interp", "interp")
            # Set current
            if src == "1":
                combo.setCurrentIndex(0)
            elif src == "2":
                combo.setCurrentIndex(1)
            else:
                combo.setCurrentIndex(2)
            combo.currentIndexChanged.connect(
                lambda _, idx=i: self._on_seg_source_changed(idx))
            top_row.addWidget(combo)
            self._seg_combos.append(combo)
            card_layout.addLayout(top_row)

            # Row 2: time range
            time_row = QHBoxLayout()
            time_row.setSpacing(2)

            spin_s = QDoubleSpinBox()
            spin_s.setRange(0, self._t_max)
            spin_s.setDecimals(1)
            spin_s.setSingleStep(0.1)
            spin_s.setValue(seg["t_start"])
            spin_s.setSuffix(" min")
            spin_s.setStyleSheet(
                "QDoubleSpinBox { border: 1px solid #ddd; border-radius: 2px; "
                "padding: 1px; font-size: 10px; }")
            spin_s.setFixedWidth(75)
            spin_s.valueChanged.connect(
                lambda v, idx=i: self._on_seg_time_changed(idx, "start", v))
            time_row.addWidget(spin_s)
            self._seg_spins_start.append(spin_s)

            dash = QLabel("\u2013")
            dash.setStyleSheet("border: none; font-size: 10px; color: #999;")
            dash.setFixedWidth(10)
            dash.setAlignment(Qt.AlignCenter)
            time_row.addWidget(dash)

            spin_e = QDoubleSpinBox()
            spin_e.setRange(0, self._t_max)
            spin_e.setDecimals(1)
            spin_e.setSingleStep(0.1)
            spin_e.setValue(seg["t_end"])
            spin_e.setSuffix(" min")
            spin_e.setStyleSheet(
                "QDoubleSpinBox { border: 1px solid #ddd; border-radius: 2px; "
                "padding: 1px; font-size: 10px; }")
            spin_e.setFixedWidth(75)
            spin_e.valueChanged.connect(
                lambda v, idx=i: self._on_seg_time_changed(idx, "end", v))
            time_row.addWidget(spin_e)
            self._seg_spins_end.append(spin_e)

            card_layout.addLayout(time_row)
            self._seg_container_layout.addWidget(card)

            # Arrow between segments (except after last)
            if i < len(self.segments) - 1:
                arrow = QLabel("\u25b6")
                arrow.setStyleSheet("color: #bbb; font-size: 14px;")
                arrow.setFixedWidth(14)
                arrow.setAlignment(Qt.AlignCenter)
                self._seg_container_layout.addWidget(arrow)

        self._seg_container_layout.addStretch()
        self._del_btn.setEnabled(len(self.segments) > 1)

    def _on_seg_source_changed(self, seg_idx):
        """L'usuari ha canviat la font d'un segment."""
        if self._updating_widgets or seg_idx >= len(self.segments):
            return
        combo = self._seg_combos[seg_idx]
        new_src = combo.currentData()
        self.segments[seg_idx]["source"] = new_src
        # Update card border color
        card = combo.parentWidget()
        if card:
            color = _SRC_COLORS.get(new_src, '#ccc')
            card.setStyleSheet(
                f"QFrame {{ background: white; border: 2px solid {color}; "
                f"border-radius: 4px; }}")
        self._update_preview()

    def _on_seg_time_changed(self, seg_idx, which, value):
        """L'usuari ha canviat un límit temporal."""
        if self._updating_widgets or seg_idx >= len(self.segments):
            return
        if which == "start":
            self.segments[seg_idx]["t_start"] = round(value, 1)
            # Sync with previous segment's end
            if seg_idx > 0:
                self.segments[seg_idx - 1]["t_end"] = round(value, 1)
                self._updating_widgets = True
                self._seg_spins_end[seg_idx - 1].setValue(value)
                self._updating_widgets = False
        else:
            self.segments[seg_idx]["t_end"] = round(value, 1)
            # Sync with next segment's start
            if seg_idx < len(self.segments) - 1:
                self.segments[seg_idx + 1]["t_start"] = round(value, 1)
                self._updating_widgets = True
                self._seg_spins_start[seg_idx + 1].setValue(value)
                self._updating_widgets = False
        self._update_preview()

    # ------------------------------------------------------------------
    # Segment operations
    # ------------------------------------------------------------------

    def _reset_segments(self):
        from hpsec_core import check_timeout_composability
        run_dur = 12.0 if self.is_bp else 70.0
        comp = check_timeout_composability(
            self.ti1, self.ti2, run_duration_min=run_dur)
        self.segments = [s.copy() for s in comp.get("segments", [])]
        self._rebuild_segment_widgets()
        self._update_preview()

    def _split_longest_segment(self):
        """Divideix el segment més llarg en dos al punt mig."""
        if not self.segments:
            return
        longest_i = max(range(len(self.segments)),
                        key=lambda i: self.segments[i]["t_end"] - self.segments[i]["t_start"])
        seg = self.segments[longest_i]
        t_mid = round((seg["t_start"] + seg["t_end"]) / 2, 1)
        if t_mid <= seg["t_start"] + 0.2 or t_mid >= seg["t_end"] - 0.2:
            return
        other_src = "2" if seg.get("source", "1") == "1" else "1"
        new_seg = {
            "source": other_src,
            "t_start": t_mid,
            "t_end": seg["t_end"],
        }
        seg["t_end"] = t_mid
        self.segments.insert(longest_i + 1, new_seg)
        self._rebuild_segment_widgets()
        self._update_preview()

    def _remove_last_boundary(self):
        """Elimina l'últim límit entre segments (uneix els 2 últims)."""
        if len(self.segments) <= 1:
            return
        last = self.segments.pop()
        self.segments[-1]["t_end"] = last["t_end"]
        self._rebuild_segment_widgets()
        self._update_preview()

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def _update_preview(self):
        from hpsec_core import compose_replicas

        if len(self.t1) == 0 or len(self.y1) == 0:
            return
        if len(self.t2) == 0 or len(self.y2) == 0:
            return

        # Handle "interp" sources: compose_replicas expects "1" or "2",
        # so we build effective segments with interpolation applied after.
        eff_segments = []
        for seg in self.segments:
            src = seg.get("source", "1")
            if src == "interp":
                # Use source "1" as placeholder — we'll replace with interpolation
                eff_segments.append({**seg, "source": "1"})
            else:
                eff_segments.append(seg)

        # Get selected method
        method = "nearest"
        if hasattr(self, '_method_combo'):
            method = self._method_combo.currentData() or "nearest"

        t_out, y_out, meta = compose_replicas(
            self.t1, self.y1, self.t2, self.y2, eff_segments,
            method=method,
            timeout_info_1=self.ti1, timeout_info_2=self.ti2)

        # Apply interpolation for "interp" segments
        # Use the ORIGINAL segment sources (not compose output) for boundary values
        for si, seg in enumerate(self.segments):
            if seg.get("source") != "interp":
                continue
            t_s, t_e = seg["t_start"], seg["t_end"]
            mask = (t_out >= t_s) & (t_out <= t_e)
            if not np.any(mask):
                continue
            idx = np.where(mask)[0]
            if len(idx) < 2:
                continue

            # Get boundary values from adjacent segments' ORIGINAL sources
            # Left boundary: last point of previous segment
            if si > 0:
                prev_src = self.segments[si - 1].get("source", "1")
                src_left = self.y1 if prev_src == "1" else (
                    np.interp(self.t1, self.t2, self.y2) if len(self.t1) != len(self.t2)
                    or not np.allclose(self.t1, self.t2, atol=0.001) else self.y2)
                y_start = float(src_left[max(0, idx[0] - 1)])
            else:
                y_start = float(y_out[max(0, idx[0] - 1)])

            # Right boundary: first point of next segment
            if si < len(self.segments) - 1:
                next_src = self.segments[si + 1].get("source", "1")
                src_right = self.y1 if next_src == "1" else (
                    np.interp(self.t1, self.t2, self.y2) if len(self.t1) != len(self.t2)
                    or not np.allclose(self.t1, self.t2, atol=0.001) else self.y2)
                y_end = float(src_right[min(len(src_right) - 1, idx[-1] + 1)])
            else:
                y_end = float(y_out[min(len(y_out) - 1, idx[-1] + 1)])

            # Linear interpolation strictly within the segment
            y_out[idx] = np.linspace(y_start, y_end, len(idx))

        self._composed_y = y_out
        self._composed_t = t_out
        self._composed_meta = meta

        if hasattr(self, '_ax'):
            self._draw_chromatogram(t_out, y_out)
        # _draw_fraction_bars eliminat (panell fraccions tret del diàleg)

    def _draw_chromatogram(self, t_out, y_out):
        ax = self._ax

        # Preserve zoom if user has zoomed in
        prev_xlim = ax.get_xlim()
        prev_ylim = ax.get_ylim()
        has_zoom = hasattr(self, '_has_drawn_once')

        ax.clear()

        x_max = 12 if self.is_bp else 70
        r1k, r2k = self.rep_keys

        # --- Segment background fills ---
        for seg in self.segments:
            src = seg.get("source", seg.get("chosen", "1"))
            color = _SRC_COLORS.get(src, '#ccc')
            ax.axvspan(seg["t_start"], seg["t_end"],
                       alpha=0.08, color=color, zorder=0)

        # --- R1 and R2 prominent ---
        ax.plot(self.t1, self.y1, color=_SRC_COLORS["1"], lw=1.2, alpha=0.8,
                label=f'R{r1k}', zorder=3)
        ax.plot(self.t2, self.y2, color=_SRC_COLORS["2"], lw=1.2, alpha=0.8,
                label=f'R{r2k}', zorder=3)

        # --- Composed line (thin dashed) ---
        ax.plot(t_out, y_out, color=_COMPOSED_COLOR, lw=1.2, ls='--',
                alpha=0.8, label='Compost', zorder=4)

        # --- Timeout markers on x-axis ---
        self._draw_timeout_markers(ax)

        # --- Segment boundary lines ---
        for seg in self.segments:
            if seg["t_start"] > 0.1:
                ax.axvline(seg["t_start"], color='#333', ls=':', lw=1.0,
                           alpha=0.5, zorder=6)

        # --- Segment source labels ---
        for seg in self.segments:
            t_mid = (seg["t_start"] + seg["t_end"]) / 2
            src = seg.get("source", seg.get("chosen", "1"))
            if src == "interp":
                src_label = "Interp"
            else:
                src_idx = int(src) - 1 if src in ("1", "2") else 0
                src_label = f"R{self.rep_keys[src_idx]}"
            color = _SRC_COLORS.get(src, '#999')
            ax.annotate(src_label,
                        xy=(t_mid, 0.97), xycoords=('data', 'axes fraction'),
                        fontsize=8, fontweight='bold', color=color,
                        ha='center', va='top',
                        bbox=dict(boxstyle='round,pad=0.15',
                                  facecolor='white', edgecolor=color,
                                  alpha=0.85, lw=0.7))

        # Set default limits
        default_xlim = (0, x_max)
        y_max = max(np.max(self.y1) if len(self.y1) else 1,
                    np.max(self.y2) if len(self.y2) else 1) * 1.15
        default_ylim = (0, max(y_max, 1))

        # Restore zoom if user had zoomed, otherwise use defaults
        if has_zoom and prev_xlim != (0.0, 1.0):
            ax.set_xlim(prev_xlim)
            ax.set_ylim(prev_ylim)
        else:
            ax.set_xlim(default_xlim)
            ax.set_ylim(default_ylim)

        self._has_drawn_once = True

        ax.set_xlabel("Temps (min)", fontsize=9)
        ax.set_ylabel("DOC (ppb)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.12, lw=0.3)

        try:
            self._fig.tight_layout(pad=0.8)
        except Exception:
            pass
        self._canvas.draw_idle()

    def _draw_timeout_markers(self, ax):
        """Marca timeouts a l'eix X: bracket + triangle, sense ombrejat."""
        def _draw_to(ti, color, label):
            if not ti:
                return
            for to in ti.get('timeouts', []):
                t_s = to.get('affected_start_min',
                             to.get('t_start_min', 0) - 0.5)
                t_e = to.get('affected_end_min',
                             to.get('t_end_min', 0) + 1.0)
                t_mid = (t_s + t_e) / 2
                ax.annotate('', xy=(t_s, 0), xycoords=('data', 'axes fraction'),
                            xytext=(t_e, 0), textcoords=('data', 'axes fraction'),
                            arrowprops=dict(arrowstyle='|-|,widthA=0.4,widthB=0.4',
                                            color=color, lw=1.5))
                ax.plot(t_mid, 0, marker='^', color=color, markersize=7,
                        transform=ax.get_xaxis_transform(), clip_on=False,
                        zorder=15)
                ax.annotate(f'TO {label}', xy=(t_mid, 0),
                            xycoords=('data', 'axes fraction'),
                            xytext=(0, -14), textcoords='offset points',
                            fontsize=7, color=color, ha='center', va='top',
                            fontweight='bold', clip_on=False)

        _draw_to(self.ti1, _TIMEOUT_COLORS["r1"], f'R{self.rep_keys[0]}')
        _draw_to(self.ti2, _TIMEOUT_COLORS["r2"], f'R{self.rep_keys[1]}')

    # ------------------------------------------------------------------
    # Mouse: cursor info only
    # ------------------------------------------------------------------

    def _on_mouse_move(self, event):
        if not hasattr(self, '_ax') or event.inaxes != self._ax:
            if hasattr(self, '_cursor_label'):
                self._cursor_label.setText("")
            return
        t = event.xdata
        y = event.ydata
        if t is not None and y is not None:
            r1_val = self._interp_at(self.t1, self.y1, t)
            r2_val = self._interp_at(self.t2, self.y2, t)
            parts = [f"t = {t:.2f} min"]
            if r1_val is not None:
                parts.append(f"R{self.rep_keys[0]} = {r1_val:.1f}")
            if r2_val is not None:
                parts.append(f"R{self.rep_keys[1]} = {r2_val:.1f}")
            self._cursor_label.setText("  " + "    ".join(parts))

    @staticmethod
    def _interp_at(t_arr, y_arr, t_val):
        if len(t_arr) == 0 or t_val < t_arr[0] or t_val > t_arr[-1]:
            return None
        return float(np.interp(t_val, t_arr, y_arr))

    # ------------------------------------------------------------------
    # Fraction bars
    # ------------------------------------------------------------------

    def _area_to_ppm(self, area):
        quant = self.sample_data.get("quantification", {})
        rf = quant.get("rf_mass_cal_used")
        intercept = quant.get("intercept", 0) or 0
        vol = None
        for rk in self.rep_keys:
            rep = self.sample_data.get("replicas", {}).get(rk, {})
            vol = rep.get("inj_volume")
            if vol:
                break
        if not vol:
            vol = 400 if not self.is_bp else 100
        if rf and rf > 0 and area > 0:
            return max(0, (area - intercept)) * 1000 / (rf * vol)
        return None

    def _draw_fraction_bars(self, t_out, y_out):
        ax = self._bar_ax
        ax.clear()

        try:
            from hpsec_analyze import calcular_fraccions_temps
            areas_r1 = calcular_fraccions_temps(self.t1, self.y1) or {}
            areas_r2 = calcular_fraccions_temps(self.t2, self.y2) or {}
            areas_comp = calcular_fraccions_temps(t_out, y_out) or {}
        except Exception:
            self._bar_canvas.draw_idle()
            return

        frac_names = FRACTION_ORDER
        n = len(frac_names)
        y_pos = np.arange(n) * 1.2  # more spacing between fractions
        bar_h = 0.30
        r1k, r2k = self.rep_keys

        vals_r1 = [areas_r1.get(fn, 0) or 0 for fn in frac_names]
        vals_r2 = [areas_r2.get(fn, 0) or 0 for fn in frac_names]
        vals_comp = [areas_comp.get(fn, 0) or 0 for fn in frac_names]
        frac_colors = [FRACTION_COLORS.get(fn, '#999') for fn in frac_names]

        ax.barh(y_pos + bar_h, vals_r1, bar_h, color=frac_colors,
                alpha=0.85, label=f'R{r1k}', edgecolor='white', lw=0.5)
        ax.barh(y_pos, vals_r2, bar_h, color=frac_colors,
                alpha=0.45, label=f'R{r2k}', edgecolor='white',
                lw=0.5, hatch='///')
        ax.barh(y_pos - bar_h, vals_comp, bar_h, color=frac_colors,
                alpha=0.7, label='Compost', edgecolor='#333', lw=1.0)

        x_max_all = max(max(vals_r1, default=0), max(vals_r2, default=0),
                        max(vals_comp, default=0))
        margin = x_max_all * 0.01

        for i in range(n):
            if vals_r1[i] > 0:
                ax.text(vals_r1[i] + margin, y_pos[i] + bar_h,
                        f"{vals_r1[i]:.0f}", fontsize=6, color='#444',
                        va='center', ha='left')
            if vals_r2[i] > 0:
                ax.text(vals_r2[i] + margin, y_pos[i],
                        f"{vals_r2[i]:.0f}", fontsize=6, color='#888',
                        va='center', ha='left')
            if vals_comp[i] > 0:
                v_mean = (vals_r1[i] + vals_r2[i]) / 2
                diff_str = ""
                diff_color = '#444'
                if v_mean > 0:
                    pct = (vals_comp[i] - v_mean) / v_mean * 100
                    diff_color = '#C62828' if abs(pct) > 15 else '#666'
                    diff_str = f"  ({pct:+.0f}%)"
                ax.text(vals_comp[i] + margin, y_pos[i] - bar_h,
                        f"{vals_comp[i]:.0f}{diff_str}",
                        fontsize=6, fontweight='bold', color=diff_color,
                        va='center', ha='left')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(frac_names, fontsize=9)
        ax.tick_params(axis='x', labelsize=7, length=2)
        ax.set_xlabel("Àrea", fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.invert_yaxis()
        ax.set_xlim(right=x_max_all * 1.25 if x_max_all > 0 else 1)
        ax.legend(fontsize=7, loc='lower right', framealpha=0.7, ncol=3)

        # ppm summary title
        total_r1 = areas_r1.get("total", 0) or 0
        total_r2 = areas_r2.get("total", 0) or 0
        total_comp = areas_comp.get("total", 0) or 0
        ppm_r1 = self._area_to_ppm(total_r1)
        ppm_r2 = self._area_to_ppm(total_r2)
        ppm_comp = self._area_to_ppm(total_comp)
        parts = []
        if ppm_r1 is not None:
            parts.append(f"R{r1k}: {total_r1:.0f} ({ppm_r1:.2f} ppm)")
        if ppm_r2 is not None:
            parts.append(f"R{r2k}: {total_r2:.0f} ({ppm_r2:.2f} ppm)")
        if ppm_comp is not None:
            parts.append(f"Compost: {total_comp:.0f} ({ppm_comp:.2f} ppm)")
        if parts:
            ax.set_title("  |  ".join(parts),
                         fontsize=8, loc='left', pad=4, color='#333')

        try:
            self._bar_fig.tight_layout(pad=0.5)
        except Exception:
            pass
        self._bar_canvas.draw_idle()

    # ------------------------------------------------------------------
    # Apply
    # ------------------------------------------------------------------

    def _on_apply(self):
        if self._composed_y is None:
            return

        reply = QMessageBox.question(
            self,
            "Aplicar composició",
            "Es substituirà el cromatograma DOC de la rèplica seleccionada "
            "amb el senyal composat.\n\n"
            "L'original es conserva (reversible).\n\nContinuar?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        if reply != QMessageBox.Yes:
            return

        selected = self.sample_data.get("selected", {})
        sel_key = selected.get("doc", self.rep_keys[0])
        sel_rep = self.sample_data["replicas"].get(sel_key)
        if not sel_rep:
            sel_rep = self.sample_data["replicas"][self.rep_keys[0]]
            sel_key = self.rep_keys[0]

        y_current = sel_rep.get("y_doc_net")
        if y_current is not None:
            sel_rep["y_doc_net_pre_composition"] = (
                np.asarray(y_current).copy()
            )

        sel_rep["y_doc_net"] = self._composed_y.copy()
        sel_rep["timeout_composition"] = {
            "segments": self.segments,
            "source_replicas": list(self.rep_keys),
            "blend_width_min": 0.2,
            "composed_from": "timeout_repair",
        }

        try:
            from hpsec_analyze import calcular_fraccions_temps
            new_areas = calcular_fraccions_temps(
                np.asarray(sel_rep.get("t_doc", []), dtype=float),
                self._composed_y
            )
            if new_areas:
                sel_rep["areas"] = new_areas
        except Exception as e:
            logger.warning(f"Error recalculant àrees: {e}")

        logger.info(f"Composició aplicada a {self.sample_name} R{sel_key}")

        # Update UI to show applied state
        self.apply_btn.setText("Composicio aplicada")
        self.apply_btn.setEnabled(False)
        self.apply_btn.setStyleSheet(
            "QPushButton { background-color: #95a5a6; color: white; "
            "font-weight: bold; padding: 8px 16px; border-radius: 4px; }")

        # Redraw chromatogram to show final result
        self._update_preview()

        self.composition_completed.emit(self.sample_name)
