"""
HPSEC Suite - Analyze Dialogs
==============================

SampleDetailDialog — Comparació visual R1 vs R2.

Layout:
  Header:  nom mostra | mode | ppm | botons selecció
  Cos:     per cada senyal (DOC, UIB, A254):
           - Esquerra (65%): cromatograma R1+R2 overlay + diferència ×5
           - Dreta (35%): barres fraccions R1 vs R2
  Peu:     anomalies cara a cara (R1 | R2) + timeouts
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QDialog, QGroupBox, QGridLayout, QSplitter,
    QScrollArea, QFrame, QComboBox, QSizePolicy
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QColor

import numpy as np
import logging

from ._constants import DAD_WL_ALL, SIGNAL_KEYS_ALL, FRACTION_NAMES, FRACTION_RANGES
from ._helpers import configure_table_style, populate_fractions_table
from hpsec_warnings import classify_anomalies, ANOMALY_CATALOG

logger = logging.getLogger(__name__)

# Matplotlib
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Colors
_C_R1 = '#1565C0'      # Blau R1
_C_R2 = '#E65100'      # Taronja R2
_C_DIFF = '#888888'     # Gris diferència
_C_UIB_R1 = '#2E7D32'  # Verd fosc UIB R1
_C_UIB_R2 = '#66BB6A'  # Verd clar UIB R2
_LW = 0.9

# Fraction colors (consistent with panel.py)
FRACTION_COLORS = {
    "BioP": "#3498DB",
    "HS":   "#E74C3C",
    "BB":   "#F39C12",
    "SB":   "#2ECC71",
    "LMW":  "#9B59B6",
}
FRACTION_ORDER = ["BioP", "HS", "BB", "SB", "LMW"]


class SampleDetailDialog(QDialog):
    """Diàleg de detall — comparació visual R1 vs R2."""

    replica_changed = Signal(str, str, str)  # sample_name, domain (doc/dad), new_replica

    def __init__(self, sample_name, sample_data, method, parent=None):
        super().__init__(parent)
        self.sample_name = sample_name
        self.sample_data = sample_data
        self.method = method
        self.is_bp = method.upper() == "BP"

        self.setWindowTitle(f"Detall: {sample_name}")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 950)
        self.setModal(True)

        self._setup_ui()

    # ------------------------------------------------------------------
    # UI Setup
    # ------------------------------------------------------------------

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # === HEADER ===
        layout.addWidget(self._build_header())

        # === MAIN CONTENT (scrollable) ===
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea { border: none; }")

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(12)

        # Signal blocks
        if HAS_MATPLOTLIB:
            self._add_signal_blocks(content_layout)
        else:
            lbl = QLabel("Matplotlib no disponible — instal·la matplotlib.")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("color: #999; font-style: italic; padding: 40px;")
            content_layout.addWidget(lbl)

        # Anomalies face-to-face
        anom_widget = self._build_anomalies_section()
        if anom_widget:
            content_layout.addWidget(anom_widget)

        # Quality metrics section (compact)
        content_layout.addWidget(self._build_quality_section())

        content_layout.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll, 1)

        # === FOOTER ===
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        close_btn = QPushButton("Tancar")
        close_btn.setStyleSheet(
            "QPushButton { padding: 6px 24px; font-size: 12px; }"
        )
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------

    def _build_header(self):
        frame = QFrame()
        frame.setStyleSheet(
            "QFrame { background: #f8f9fa; border: 1px solid #dee2e6; "
            "border-radius: 6px; padding: 8px; }"
        )
        h = QHBoxLayout(frame)
        h.setContentsMargins(12, 6, 12, 6)
        h.setSpacing(16)

        # Nom
        name_lbl = QLabel(f"<b style='font-size:14px'>{self.sample_name}</b>")
        h.addWidget(name_lbl)

        # Mode
        mode_lbl = QLabel(f"<span style='color:#666'>{self.method}</span>")
        h.addWidget(mode_lbl)

        # ppm
        quant = self.sample_data.get("quantification", {})
        ppm_d = quant.get("concentration_ppm_direct") or quant.get("concentration_ppm")
        ppm_u = quant.get("concentration_ppm_uib")
        ppm_parts = []
        if ppm_d:
            ppm_parts.append(f"<b style='color:#1565C0'>{ppm_d:.3f} ppm</b>")
        if ppm_u:
            ppm_parts.append(f"<b style='color:#2E7D32'>{ppm_u:.3f} ppm<sub>UIB</sub></b>")
        if ppm_parts:
            h.addWidget(QLabel(" | ".join(ppm_parts)))

        # HCI
        hci = quant.get("hci")
        if hci is not None:
            char = quant.get("hci_character", "")
            colors = {"HA": "#E74C3C", "FA": "#3498DB"}
            c = colors.get(char[:2], "#27AE60") if char else "#27AE60"
            h.addWidget(QLabel(f"<b style='color:{c}'>HCI {hci:.1f} ({char})</b>"))

        h.addStretch()

        # Selector repliques (info only — real selection in panel.py combos)
        selected = self.sample_data.get("selected", {})
        replicas = self.sample_data.get("replicas", {})
        rep_keys = sorted(replicas.keys())
        if len(rep_keys) >= 2:
            sel_lbl = QLabel(
                f"<span style='color:#888; font-size:11px'>"
                f"DOC: R{selected.get('doc', '?')} | "
                f"DAD: R{selected.get('dad', '?')}</span>"
            )
            h.addWidget(sel_lbl)

        return frame

    # ------------------------------------------------------------------
    # Signal blocks (DOC, UIB, A254)
    # ------------------------------------------------------------------

    def _add_signal_blocks(self, layout):
        """Afegeix blocs de senyal: cromatograma R1+R2 + barres fraccions."""
        replicas = self.sample_data.get("replicas", {})
        rep_keys = sorted(replicas.keys())
        if not rep_keys:
            return

        r1 = replicas.get(rep_keys[0], {})
        r2 = replicas.get(rep_keys[1], {}) if len(rep_keys) > 1 else None
        comparison = self.sample_data.get("comparison", {})
        selected = self.sample_data.get("selected", {})
        doc_sel = selected.get("doc", rep_keys[0])
        dad_sel = selected.get("dad", rep_keys[0])

        # Fraccions from config
        from hpsec_config import get_config
        cfg = get_config()
        mode = "BP" if self.is_bp else "COLUMN"
        fracs = cfg.get_all_fractions(mode)
        x_min, x_max = (0, 15) if self.is_bp else (0, 70)

        # --- DOC Direct ---
        t1 = _as_array(r1.get("t_doc"))
        y1 = _as_array(r1.get("y_doc_net"))
        t2 = _as_array(r2.get("t_doc")) if r2 else None
        y2 = _as_array(r2.get("y_doc_net")) if r2 else None

        doc_comp = comparison.get("doc", {})
        pearson_doc = doc_comp.get("pearson", 0)

        doc_block = self._build_signal_block(
            title="DOC Direct",
            t1=t1, y1=y1, t2=t2, y2=y2,
            c1=_C_R1, c2=_C_R2,
            rep_keys=rep_keys,
            areas_r1=_get_frac_areas(r1, "DOC"),
            areas_r2=_get_frac_areas(r2, "DOC") if r2 else {},
            r2_val=pearson_doc,
            fracs=fracs, x_range=(x_min, x_max),
            selected_rep=doc_sel,
            timeout_r1=r1.get("timeout_info"),
            timeout_r2=r2.get("timeout_info") if r2 else None,
        )
        if doc_block:
            layout.addWidget(doc_block)

        # --- DOC UIB ---
        y1_uib = _as_array(r1.get("y_doc_uib_net"))
        y2_uib = _as_array(r2.get("y_doc_uib_net")) if r2 else None

        has_uib = (y1_uib is not None and t1 is not None
                   and len(y1_uib) == len(t1))
        if has_uib:
            uib_block = self._build_signal_block(
                title="DOC UIB",
                t1=t1, y1=y1_uib, t2=t2, y2=y2_uib,
                c1=_C_UIB_R1, c2=_C_UIB_R2,
                rep_keys=rep_keys,
                areas_r1=_get_uib_areas(r1),
                areas_r2=_get_uib_areas(r2) if r2 else {},
                fracs=fracs, x_range=(x_min, x_max),
                selected_rep=doc_sel,
                timeout_r1=r1.get("timeout_info_uib", r1.get("timeout_info")),
                timeout_r2=(r2.get("timeout_info_uib", r2.get("timeout_info"))
                            if r2 else None),
            )
            if uib_block:
                layout.addWidget(uib_block)

        # --- DAD A254 ---
        df_dad1 = r1.get("df_dad")
        df_dad2 = r2.get("df_dad") if r2 else None
        dad_comp = comparison.get("dad", {})
        pearson_per_wl = dad_comp.get("pearson_per_wavelength", {})

        if df_dad1 is not None and hasattr(df_dad1, 'columns'):
            t_dad1 = df_dad1['time (min)'].values if 'time (min)' in df_dad1.columns else None
            t_dad2 = (df_dad2['time (min)'].values
                      if df_dad2 is not None and 'time (min)' in df_dad2.columns
                      else None)

            for wl in ['254', '220', '272', '290']:
                if wl not in df_dad1.columns:
                    continue
                y_d1 = df_dad1[wl].values
                y_d2 = df_dad2[wl].values if (df_dad2 is not None
                                               and wl in df_dad2.columns) else None
                r2_wl = (pearson_per_wl.get(f"A{wl}", 0)
                         or pearson_per_wl.get(wl, 0))
                areas_key = f"A{wl}"

                dad_block = self._build_signal_block(
                    title=f"DAD A{wl}",
                    t1=t_dad1, y1=y_d1, t2=t_dad2, y2=y_d2,
                    c1=_C_R1, c2=_C_R2,
                    rep_keys=rep_keys,
                    areas_r1=_get_frac_areas(r1, areas_key),
                    areas_r2=_get_frac_areas(r2, areas_key) if r2 else {},
                    r2_val=r2_wl,
                    fracs=fracs, x_range=(x_min, x_max),
                    selected_rep=dad_sel,
                )
                if dad_block:
                    layout.addWidget(dad_block)

    def _build_signal_block(self, title, t1, y1, t2, y2,
                            c1, c2, rep_keys,
                            areas_r1, areas_r2,
                            r2_val=None, fracs=None,
                            x_range=(0, 70),
                            selected_rep=None,
                            timeout_r1=None, timeout_r2=None):
        """Construeix un bloc per un senyal: cromatograma + barres.

        Args:
            selected_rep: rèplica seleccionada (e.g. "1") — marca amb ★
        """
        if t1 is None or y1 is None:
            return None

        frame = QFrame()
        frame.setStyleSheet(
            "QFrame { border: 1px solid #e0e0e0; border-radius: 4px; "
            "background: white; }"
        )
        v_outer = QVBoxLayout(frame)
        v_outer.setContentsMargins(4, 2, 4, 4)
        v_outer.setSpacing(2)

        # --- TITLE BAR: senyal + mètriques compactes ---
        title_parts = [f"<b>{title}</b>"]

        # Àrees totals R1/R2
        total_r1 = areas_r1.get("total", 0) or 0
        total_r2 = areas_r2.get("total", 0) or 0
        if total_r1 > 0:
            title_parts.append(
                f"<span style='color:{c1}'>R{rep_keys[0]}: {total_r1:.0f}</span>")
        if total_r2 > 0 and len(rep_keys) > 1:
            title_parts.append(
                f"<span style='color:{c2}'>R{rep_keys[1]}: {total_r2:.0f}</span>")

        # Diff % entre rèpliques
        if total_r1 > 0 and total_r2 > 0:
            avg = (total_r1 + total_r2) / 2
            diff_pct = (total_r1 - total_r2) / avg * 100
            dc = '#C62828' if abs(diff_pct) > 15 else ('#E67E22' if abs(diff_pct) > 5 else '#27AE60')
            title_parts.append(f"<span style='color:{dc}'>\u0394{diff_pct:+.1f}%</span>")

        # Pearson R²
        if r2_val and r2_val > 0:
            rc = '#C62828' if r2_val < 0.990 else ('#E67E22' if r2_val < 0.998 else '#27AE60')
            title_parts.append(f"<span style='color:{rc}'>R\u00b2={r2_val:.4f}</span>")

        # Rèplica seleccionada
        if selected_rep and len(rep_keys) > 1:
            title_parts.append(
                f"<span style='color:#1565C0'>\u2605 R{selected_rep}</span>")

        title_lbl = QLabel(" &nbsp;|&nbsp; ".join(title_parts))
        title_lbl.setStyleSheet(
            "font-size: 10px; padding: 2px 6px; border: none; "
            "background: #f8f9fa; border-radius: 3px;"
        )
        v_outer.addWidget(title_lbl)

        # --- CONTENT ROW: chromatogram + bars ---
        h_layout = QHBoxLayout()
        h_layout.setContentsMargins(0, 0, 0, 0)
        h_layout.setSpacing(4)

        has_r2 = (t2 is not None and y2 is not None
                  and len(y2) > 10)

        fig_h = 2.0
        fig = Figure(figsize=(7, fig_h), dpi=100)
        fig.set_facecolor('white')
        canvas = FigureCanvas(fig)
        canvas.setMinimumHeight(int(fig_h * 95))
        canvas.setMaximumHeight(int(fig_h * 110))

        ax = fig.add_subplot(111)

        # Rèplica seleccionada amb ★
        sel_mark_1 = " \u2605" if selected_rep == rep_keys[0] else ""
        sel_mark_2 = (" \u2605" if len(rep_keys) > 1
                      and selected_rep == rep_keys[1] else "")

        ax.plot(t1, y1, color=c1, lw=_LW,
                label=f'R{rep_keys[0]}{sel_mark_1}', zorder=3)

        if has_r2:
            ax.plot(t2, y2, color=c2, lw=_LW, alpha=0.7,
                    label=f'R{rep_keys[1]}{sel_mark_2}', zorder=2)

            # Difference trace (×5, interpolated to same grid)
            try:
                y2_interp = np.interp(t1, t2, y2)
                diff = (y1 - y2_interp) * 5
                ax.plot(t1, diff, color=_C_DIFF, lw=0.5, alpha=0.5,
                        label='Diff \u00d75', zorder=1)
                ax.axhline(0, color='#ddd', lw=0.3, zorder=0)
            except Exception:
                pass

        ax.set_xlim(*x_range)
        ax.tick_params(labelsize=6, length=2, pad=1)
        ax.grid(True, alpha=0.15, lw=0.3)
        ax.legend(loc='upper right', fontsize=5.5, ncol=3,
                  framealpha=0.7, handlelength=1.2)

        # Fraction vertical lines
        if fracs and not self.is_bp:
            for fname, finfo in fracs:
                s = finfo['start']
                if 0 < s <= x_range[1]:
                    ax.axvline(s, color='#bbb', ls=':', lw=0.4, zorder=0)

        # Timeout zones
        if timeout_r1 or timeout_r2:
            from ._helpers import draw_timeout_zones_on_ax
            draw_timeout_zones_on_ax(ax, timeout_r1, timeout_r2)

        fig.tight_layout(pad=0.3)
        canvas.draw()
        h_layout.addWidget(canvas, 65)
        v_outer.addLayout(h_layout)

        # --- RIGHT: Fraction bars R1 vs R2 (35%) ---
        if self.is_bp or not fracs:
            # BP: single total bar
            bar_widget = self._build_total_bars(
                areas_r1, areas_r2, rep_keys, has_r2)
        else:
            bar_widget = self._build_fraction_bars(
                areas_r1, areas_r2, rep_keys, has_r2, fracs)
        h_layout.addWidget(bar_widget, 35)

        return frame

    # ------------------------------------------------------------------
    # Fraction bars (R1 vs R2)
    # ------------------------------------------------------------------

    def _build_fraction_bars(self, areas_r1, areas_r2, rep_keys, has_r2, fracs):
        """Barres fraccions R1 vs R2 (horizontal grouped)."""
        fig = Figure(figsize=(3.5, 2.2), dpi=100)
        fig.set_facecolor('white')
        canvas = FigureCanvas(fig)
        canvas.setMinimumHeight(200)
        canvas.setMaximumHeight(250)

        ax = fig.add_subplot(111)

        frac_names = [fn for fn, _ in fracs]
        n = len(frac_names)
        y_pos = np.arange(n)
        bar_h = 0.35

        vals_r1 = [areas_r1.get(fn, 0) or 0 for fn in frac_names]
        colors = [FRACTION_COLORS.get(fn, '#999') for fn in frac_names]

        ax.barh(y_pos + bar_h/2, vals_r1, bar_h, color=colors,
                alpha=0.85, label=f'R{rep_keys[0]}', edgecolor='white', lw=0.5)

        if has_r2:
            vals_r2 = [areas_r2.get(fn, 0) or 0 for fn in frac_names]
            ax.barh(y_pos - bar_h/2, vals_r2, bar_h, color=colors,
                    alpha=0.45, label=f'R{rep_keys[1]}', edgecolor='white',
                    lw=0.5, hatch='///')

            # Diff % labels
            for i, (v1, v2) in enumerate(zip(vals_r1, vals_r2)):
                if v1 > 0 and v2 > 0:
                    pct = (v1 - v2) / ((v1 + v2) / 2) * 100
                    c = '#C62828' if abs(pct) > 15 else '#888'
                    ax.text(max(v1, v2) * 1.02, y_pos[i],
                            f"{pct:+.0f}%", fontsize=5.5, color=c,
                            va='center', ha='left')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(frac_names, fontsize=6.5)
        ax.tick_params(axis='x', labelsize=5.5, length=2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.invert_yaxis()

        if has_r2:
            ax.legend(fontsize=5.5, loc='lower right', framealpha=0.7)

        fig.tight_layout(pad=0.4)
        canvas.draw()
        return canvas

    def _build_total_bars(self, areas_r1, areas_r2, rep_keys, has_r2):
        """Barra total per BP (sense fraccions)."""
        fig = Figure(figsize=(3.5, 2.2), dpi=100)
        fig.set_facecolor('white')
        canvas = FigureCanvas(fig)
        canvas.setMinimumHeight(200)
        canvas.setMaximumHeight(250)

        ax = fig.add_subplot(111)

        total_r1 = areas_r1.get("total", 0) or 0
        labels = [f'R{rep_keys[0]}']
        vals = [total_r1]
        colors_list = [_C_R1]

        if has_r2:
            total_r2 = areas_r2.get("total", 0) or 0
            labels.append(f'R{rep_keys[1]}')
            vals.append(total_r2)
            colors_list.append(_C_R2)

            if total_r1 > 0 and total_r2 > 0:
                pct = (total_r1 - total_r2) / ((total_r1 + total_r2) / 2) * 100
                c = '#C62828' if abs(pct) > 15 else '#555'
                ax.text(0.5, 0.95, f"Diff: {pct:+.1f}%",
                        transform=ax.transAxes, fontsize=8, color=c,
                        ha='center', va='top', fontweight='bold')

        ax.bar(labels, vals, color=colors_list, alpha=0.8, width=0.5)
        ax.set_ylabel("Àrea total", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        fig.tight_layout(pad=0.4)
        canvas.draw()
        return canvas

    # ------------------------------------------------------------------
    # Anomalies face-to-face
    # ------------------------------------------------------------------

    def _build_anomalies_section(self):
        """Anomalies R1 | R2 cara a cara."""
        replicas = self.sample_data.get("replicas", {})
        rep_keys = sorted(replicas.keys())
        if len(rep_keys) < 2:
            # Single replica — simple list
            r1 = replicas.get(rep_keys[0], {})
            anoms = r1.get("anomalies", [])
            comparison = self.sample_data.get("comparison", {})
            for domain in ("doc", "dad"):
                for w in (comparison.get(domain) or {}).get("warnings", []):
                    anoms.append(w)
            if not anoms:
                return None
            return self._build_simple_anomalies(anoms)

        r1 = replicas.get(rep_keys[0], {})
        r2 = replicas.get(rep_keys[1], {})
        anoms_r1 = list(r1.get("anomalies", []))
        anoms_r2 = list(r2.get("anomalies", []))

        # Add comparison warnings
        comparison = self.sample_data.get("comparison", {})
        comp_anoms = []
        for domain in ("doc", "dad"):
            for w in (comparison.get(domain) or {}).get("warnings", []):
                comp_anoms.append(w)

        if not anoms_r1 and not anoms_r2 and not comp_anoms:
            return None

        frame = QFrame()
        frame.setStyleSheet(
            "QFrame { border: 1px solid #e0e0e0; border-radius: 4px; "
            "background: #fafafa; }"
        )
        outer = QVBoxLayout(frame)
        outer.setContentsMargins(8, 6, 8, 6)
        outer.setSpacing(4)

        title = QLabel("<b>Anomalies</b>")
        title.setStyleSheet("font-size: 11px; color: #333; border: none;")
        outer.addWidget(title)

        # Grid: R1 | R2
        grid = QGridLayout()
        grid.setSpacing(4)

        # Headers
        for col, rep_key in enumerate(rep_keys[:2]):
            hdr = QLabel(f"<b>R{rep_key}</b>")
            hdr.setAlignment(Qt.AlignmentFlag.AlignCenter)
            hdr.setStyleSheet(
                "font-size: 10px; color: #555; border: none; "
                "padding: 2px; background: #eee; border-radius: 3px;"
            )
            grid.addWidget(hdr, 0, col)

        max_rows = max(len(anoms_r1), len(anoms_r2))
        for i in range(max_rows):
            for col, anoms in enumerate([anoms_r1, anoms_r2]):
                if i < len(anoms):
                    a = anoms[i]
                    grid.addWidget(self._anomaly_label(a), i + 1, col)
                else:
                    spacer = QLabel("")
                    spacer.setStyleSheet("border: none;")
                    grid.addWidget(spacer, i + 1, col)

        outer.addLayout(grid)

        # Comparison anomalies (shared)
        if comp_anoms:
            comp_lbl = QLabel("<b>Comparació:</b>")
            comp_lbl.setStyleSheet("font-size: 10px; color: #666; border: none; margin-top: 4px;")
            outer.addWidget(comp_lbl)
            for a in comp_anoms:
                outer.addWidget(self._anomaly_label(a))

        return frame

    def _build_simple_anomalies(self, anoms):
        """Llista simple d'anomalies (una sola rèplica)."""
        frame = QFrame()
        frame.setStyleSheet(
            "QFrame { border: 1px solid #e0e0e0; border-radius: 4px; "
            "background: #fafafa; }"
        )
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(2)

        title = QLabel("<b>Anomalies</b>")
        title.setStyleSheet("font-size: 11px; color: #333; border: none;")
        lay.addWidget(title)

        for a in anoms:
            lay.addWidget(self._anomaly_label(a))
        return frame

    def _anomaly_label(self, a):
        """Crea un QLabel per una anomalia amb icona, severitat i acció."""
        SEVERITY_STYLE = {
            "blocker": ("&#10008;", "#E74C3C"),
            "critical": ("&#10008;", "#E74C3C"),
            "repaired": ("&#10004;", "#27AE60"),
            "warning": ("&#9888;", "#F39C12"),
            "info": ("&#8505;", "#3498DB"),
        }

        if isinstance(a, dict):
            code = a.get("code", "?")
            severity = a.get("severity", "info")
            repaired = a.get("repaired", False)
        else:
            code = str(a)
            severity = "info"
            repaired = False

        if repaired:
            severity = "repaired"

        entry = ANOMALY_CATALOG.get(code, {})
        # Prioritzar label de l'anomalia (override_label) sobre el catàleg
        if isinstance(a, dict) and a.get("label"):
            label_text = a["label"]
        else:
            label_text = entry.get("label", code)
        action = entry.get("action", "")
        icon, color = SEVERITY_STYLE.get(severity, ("?", "#666"))

        html = f"<span style='color:{color}'>{icon}</span> {label_text}"
        if action:
            html += f" <span style='color:#aaa; font-size:9px'>({action})</span>"

        lbl = QLabel(html)
        lbl.setWordWrap(True)
        lbl.setStyleSheet("font-size: 10px; padding: 1px 4px; border: none;")
        return lbl

    # ------------------------------------------------------------------
    # Quality metrics (compact summary)
    # ------------------------------------------------------------------

    def _build_quality_section(self):
        """Secció compacta de mètriques de qualitat."""
        frame = QFrame()
        frame.setStyleSheet(
            "QFrame { border: 1px solid #e0e0e0; border-radius: 4px; "
            "background: #f8f9fa; padding: 6px 10px; }"
        )
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(2)

        replicas = self.sample_data.get("replicas", {})
        selected = self.sample_data.get("selected", {})
        doc_sel = selected.get("doc", "1")
        doc_rep = replicas.get(doc_sel, {})
        comparison = self.sample_data.get("comparison", {})
        quant = self.sample_data.get("quantification", {})

        parts = []

        # SNR
        snr_info = doc_rep.get("snr_info", {})
        snr = snr_info.get("snr_direct", 0)
        if snr:
            c = "#27AE60" if snr >= 50 else ("#F39C12" if snr >= 10 else "#E74C3C")
            parts.append(f"<span style='color:{c}'>SNR: {snr:.0f}</span>")

        # LOD/LOQ ppm
        lod_ppm = quant.get("lod_ppm")
        loq_ppm = quant.get("loq_ppm")
        if lod_ppm is not None:
            parts.append(f"LOD: {lod_ppm:.3f} ppm")
        if loq_ppm is not None:
            parts.append(f"LOQ: {loq_ppm:.3f} ppm")

        # R² DOC
        r2_doc = comparison.get("doc", {}).get("pearson", 0) if comparison else 0
        if r2_doc > 0:
            c = "#27AE60" if r2_doc >= 0.99 else ("#F39C12" if r2_doc >= 0.95 else "#E74C3C")
            parts.append(f"<span style='color:{c}'>R²_DOC: {r2_doc:.4f}</span>")

        # R² DAD
        dad_comp = comparison.get("dad", {}) if comparison else {}
        r2_dad = dad_comp.get("pearson_min", 0)
        if r2_dad > 0:
            c = "#27AE60" if r2_dad >= 0.99 else ("#F39C12" if r2_dad >= 0.95 else "#E74C3C")
            parts.append(f"<span style='color:{c}'>R²_DAD: {r2_dad:.4f}</span>")

        # Injection indices
        inj_parts = []
        for rk, rd in sorted(replicas.items()):
            idx = rd.get("injection_index")
            if idx is not None:
                inj_parts.append(str(idx))
        if inj_parts:
            parts.append(f"Inj: {', '.join(inj_parts)}")

        html = " &middot; ".join(parts) if parts else "—"
        lbl = QLabel(f"<span style='font-size: 11px'>{html}</span>")
        lbl.setWordWrap(True)
        lbl.setStyleSheet("border: none;")
        layout.addWidget(lbl)

        return frame

    # ------------------------------------------------------------------
    # Fractions table (compacta, a baix)
    # ------------------------------------------------------------------

    def _build_fractions_section(self):
        """Taula fraccions completa (tots els senyals)."""
        group = QGroupBox("Fraccions" if not self.is_bp else "Totals")
        group.setStyleSheet(
            "QGroupBox { font-size: 11px; font-weight: bold; border: 1px solid #ddd; "
            "border-radius: 4px; margin-top: 6px; padding-top: 14px; }"
            "QGroupBox::title { subcontrol-position: top left; padding: 0 6px; }"
        )
        layout = QVBoxLayout(group)
        layout.setContentsMargins(4, 4, 4, 4)

        selected = self.sample_data.get("selected", {})
        rep_sel = selected.get("doc", "1")
        rep_data = (self.sample_data.get("replicas") or {}).get(rep_sel, {})

        tbl = QTableWidget()
        configure_table_style(tbl, compact=True)
        populate_fractions_table(
            tbl, rep_data, self.is_bp, DAD_WL_ALL,
            show_ratio=not self.is_bp
        )

        max_rows = tbl.rowCount()
        tbl.setMaximumHeight(min(350, 28 + 22 * max_rows))
        layout.addWidget(tbl)
        return group


# ======================================================================
# Helpers
# ======================================================================

def _as_array(val):
    """Converteix a numpy array si no és None."""
    if val is None:
        return None
    arr = np.asarray(val)
    if arr.size < 5:
        return None
    return arr


def _get_frac_areas(rep_data, signal_key):
    """Extreu àrees per fracció d'un rep_data per un senyal."""
    if rep_data is None:
        return {}
    areas = rep_data.get("areas") or {}
    sig_areas = areas.get(signal_key) or {}
    return sig_areas


def _get_uib_areas(rep_data):
    """Extreu àrees UIB."""
    if rep_data is None:
        return {}
    return rep_data.get("areas_uib") or {}
