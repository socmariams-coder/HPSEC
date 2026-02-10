"""
HPSEC Suite - Analyze Dialogs
==============================

SampleDetailDialog amb gràfics, taula fraccions completa i resum senyals.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QDialog, QGroupBox, QGridLayout, QSplitter,
    QScrollArea
)
from PySide6.QtCore import Qt

import numpy as np

from ._constants import DAD_WL_ALL, SIGNAL_KEYS_ALL
from ._helpers import (
    configure_table_style, populate_signal_summary, populate_fractions_table
)

# Matplotlib
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class SampleDetailDialog(QDialog):
    """Diàleg de detall d'una mostra amb gràfics i estadístiques completes."""

    def __init__(self, sample_name, sample_data, method, parent=None):
        super().__init__(parent)
        self.sample_name = sample_name
        self.sample_data = sample_data
        self.method = method
        self.is_bp = method.upper() == "BP"

        self.setWindowTitle(f"Detall: {sample_name}")
        self.setMinimumSize(1200, 850)
        self.resize(1400, 1000)
        self.setModal(True)

        self._setup_ui()

    def _count_graph_rows(self):
        """Compte files de gràfics: DOC/UIB + parells DAD."""
        replicas = self.sample_data.get("replicas", {})
        n_wl = 0
        if replicas:
            r1 = replicas.get(sorted(replicas.keys())[0], {})
            df_dad = r1.get("df_dad")
            if df_dad is not None and hasattr(df_dad, 'columns'):
                n_wl = sum(1 for c in df_dad.columns if c != 'time (min)')
        n_dad_rows = max((n_wl + 1) // 2, 1)  # parells de λ
        return 1 + n_dad_rows  # DOC/UIB row + DAD rows

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Splitter principal
        splitter = QSplitter(Qt.Horizontal)

        # === LEFT: GRAPHS (scrollable) ===
        if HAS_MATPLOTLIB:
            n_graph_rows = self._count_graph_rows()
            n_total_rows = n_graph_rows + 1  # + table row
            fig_h = max(6, n_graph_rows * 1.1 + 2.8)
            self.figure = Figure(figsize=(7.5, fig_h), dpi=100)
            self.canvas = FigureCanvas(self.figure)
            self.canvas.setMinimumHeight(int(fig_h * 100))

            graph_scroll = QScrollArea()
            graph_scroll.setWidgetResizable(True)
            graph_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            graph_scroll.setWidget(self.canvas)
            splitter.addWidget(graph_scroll)
            self._plot_signals()
        else:
            graph_widget = QWidget()
            graph_layout = QVBoxLayout(graph_widget)
            graph_layout.setContentsMargins(0, 0, 0, 0)
            no_plot = QLabel("Matplotlib no disponible.\nInstal·la matplotlib per veure gràfics.")
            no_plot.setAlignment(Qt.AlignCenter)
            no_plot.setStyleSheet("color: #666; font-style: italic;")
            graph_layout.addWidget(no_plot)
            splitter.addWidget(graph_widget)

        # === RIGHT: STATS ===
        stats_scroll = QScrollArea()
        stats_scroll.setWidgetResizable(True)
        stats_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        stats_scroll.setStyleSheet("QScrollArea { border: none; }")

        stats_widget = QWidget()
        stats_layout = QVBoxLayout(stats_widget)
        stats_layout.setContentsMargins(8, 0, 8, 0)
        stats_layout.setSpacing(12)

        # Info general
        stats_layout.addWidget(self._create_info_group())

        # Comparació rèpliques
        if len(self.sample_data.get("replicas", {})) > 1:
            stats_layout.addWidget(self._create_comparison_group())

        # Signal summary (tmax + àrea + SNR)
        stats_layout.addWidget(self._create_signal_summary_group())

        # Timeouts info
        timeout_group = self._create_timeout_group()
        if timeout_group:
            stats_layout.addWidget(timeout_group)

        # Fractions table (complete, all wavelengths)
        stats_layout.addWidget(self._create_fractions_group())

        # Pics HS info (COLUMN)
        if not self.is_bp:
            peaks_group = self._create_peaks_hs_group()
            if peaks_group:
                stats_layout.addWidget(peaks_group)

        stats_layout.addStretch()
        stats_scroll.setWidget(stats_widget)
        splitter.addWidget(stats_scroll)

        splitter.setSizes([600, 500])
        layout.addWidget(splitter)

        # Close button
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        close_btn = QPushButton("Tancar")
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

    # ------------------------------------------------------------------
    # Info group
    # ------------------------------------------------------------------

    def _create_info_group(self):
        group = QGroupBox("Informació General")
        layout = QGridLayout(group)
        layout.setSpacing(8)

        selected = self.sample_data.get("selected", {})
        quantification = self.sample_data.get("quantification", {})

        row = 0
        layout.addWidget(QLabel("<b>Mostra:</b>"), row, 0)
        layout.addWidget(QLabel(self.sample_name), row, 1)
        row += 1

        layout.addWidget(QLabel("<b>Rèplica:</b>"), row, 0)
        layout.addWidget(QLabel(f"R{selected.get('doc', '?')}"), row, 1)
        row += 1

        layout.addWidget(QLabel("<b>Mode:</b>"), row, 0)
        layout.addWidget(QLabel(self.method), row, 1)
        row += 1

        # Concentrations (use pre-calculated values from quantify_sample,
        # which uses rf_mass_cal global — NOT the old area/rf_direct formula)
        conc_direct = (quantification.get("concentration_ppm_direct")
                       or quantification.get("concentration_ppm"))
        conc_uib = quantification.get("concentration_ppm_uib")
        area = quantification.get("area_total", 0)
        cal_source = quantification.get("calibration_source", "")

        layout.addWidget(QLabel("<b>ppm Direct:</b>"), row, 0)
        conc_d_label = QLabel(f"{conc_direct:.3f} ppm" if conc_direct else "-")
        conc_d_label.setStyleSheet("font-weight: bold; color: #2E86AB;")
        layout.addWidget(conc_d_label, row, 1)
        row += 1

        if conc_uib:
            layout.addWidget(QLabel("<b>ppm UIB:</b>"), row, 0)
            conc_u_label = QLabel(f"{conc_uib:.3f} ppm")
            conc_u_label.setStyleSheet("font-weight: bold; color: #2A9D8F;")
            layout.addWidget(conc_u_label, row, 1)
            row += 1

        layout.addWidget(QLabel("<b>Àrea total:</b>"), row, 0)
        layout.addWidget(QLabel(f"{area:.1f}" if area else "-"), row, 1)
        row += 1

        if cal_source:
            layout.addWidget(QLabel("<b>Calibració:</b>"), row, 0)
            layout.addWidget(QLabel(f"{cal_source}"), row, 1)

        return group

    # ------------------------------------------------------------------
    # Comparison group
    # ------------------------------------------------------------------

    def _create_comparison_group(self):
        group = QGroupBox("Comparació Rèpliques")
        layout = QGridLayout(group)
        layout.setSpacing(8)

        comparison = self.sample_data.get("comparison", {})
        doc_comp = comparison.get("doc", {})
        dad_comp = comparison.get("dad", {})

        row = 0

        # Pearson DOC
        pearson = doc_comp.get("pearson", 0)
        layout.addWidget(QLabel("Pearson DOC:"), row, 0)
        p_label = QLabel(f"{pearson:.4f}")
        if 0 < pearson < 0.995:
            p_label.setStyleSheet("color: #F39C12; font-weight: bold;")
        layout.addWidget(p_label, row, 1)
        row += 1

        # Diff area
        area_diff = doc_comp.get("area_diff_pct", 0)
        layout.addWidget(QLabel("Diff àrea DOC:"), row, 0)
        diff_label = QLabel(f"{area_diff:.1f}%")
        if area_diff > 10:
            diff_label.setStyleSheet("color: #F39C12; font-weight: bold;")
        layout.addWidget(diff_label, row, 1)
        row += 1

        # Pearson DAD 254
        pearson_254 = dad_comp.get("pearson_254", 0)
        if pearson_254:
            layout.addWidget(QLabel("Pearson DAD 254:"), row, 0)
            p254_label = QLabel(f"{pearson_254:.4f}")
            if 0 < pearson_254 < 0.995:
                p254_label.setStyleSheet("color: #F39C12; font-weight: bold;")
            layout.addWidget(p254_label, row, 1)
            row += 1

        # Warnings
        warnings = doc_comp.get("warnings", [])
        if warnings:
            layout.addWidget(QLabel("<b>Warnings:</b>"), row, 0, 1, 2)
            row += 1
            for w in warnings[:5]:
                w_label = QLabel(f"⚠ {w}")
                w_label.setStyleSheet("color: #F39C12; font-size: 11px;")
                w_label.setWordWrap(True)
                layout.addWidget(w_label, row, 0, 1, 2)
                row += 1

        return group

    # ------------------------------------------------------------------
    # Signal summary group (tmax + area + SNR per signal)
    # ------------------------------------------------------------------

    def _create_signal_summary_group(self):
        group = QGroupBox("Resum Senyals")
        layout = QVBoxLayout(group)

        tbl = QTableWidget()
        tbl.setColumnCount(4)
        tbl.setHorizontalHeaderLabels(["Senyal", "tmax (min)", "Àrea total", "SNR"])
        configure_table_style(tbl, compact=True)

        selected = self.sample_data.get("selected", {})
        rep_sel = selected.get("doc", "1")
        rep_data = (self.sample_data.get("replicas") or {}).get(rep_sel, {})

        populate_signal_summary(tbl, rep_data, SIGNAL_KEYS_ALL, show_timeouts=False)

        tbl.setMaximumHeight(30 + 26 * tbl.rowCount())
        layout.addWidget(tbl)
        return group

    # ------------------------------------------------------------------
    # Timeout group
    # ------------------------------------------------------------------

    def _create_timeout_group(self):
        selected = self.sample_data.get("selected", {})
        rep_sel = selected.get("doc", "1")
        rep_data = (self.sample_data.get("replicas") or {}).get(rep_sel, {})
        timeout_info = rep_data.get("timeout_info") or {}
        n_timeouts = timeout_info.get("n_timeouts", 0)

        if n_timeouts == 0:
            return None

        group = QGroupBox(f"Timeouts ({n_timeouts})")
        layout = QVBoxLayout(group)

        severity = timeout_info.get("severity", "INFO")
        zones = timeout_info.get("zones", [])
        durations = timeout_info.get("durations", [])

        color = "#E65100" if severity in ("WARNING", "CRITICAL") else "#1565C0"
        info_label = QLabel(
            f"<span style='color:{color}'><b>Severitat: {severity}</b></span><br>"
            f"Zones afectades: {', '.join(zones) if zones else 'N/A'}"
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        if durations:
            for i, dur in enumerate(durations[:5]):
                dur_label = QLabel(f"  Timeout {i+1}: {dur}")
                dur_label.setStyleSheet("color: #666; font-size: 11px;")
                layout.addWidget(dur_label)

        # UIB propagation note
        note = QLabel(
            "<i style='color:#888;'>Nota: Timeouts DOC Direct també afecten UIB "
            "(mateix detector, senyal simultani).</i>"
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        return group

    # ------------------------------------------------------------------
    # Fractions group (complete, all wavelengths)
    # ------------------------------------------------------------------

    def _create_fractions_group(self):
        group = QGroupBox("Àrees per Fracció" if not self.is_bp else "Àrees Totals")
        layout = QVBoxLayout(group)

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
        tbl.setMaximumHeight(min(400, 30 + 24 * max_rows))
        layout.addWidget(tbl)
        return group

    # ------------------------------------------------------------------
    # Peaks HS group
    # ------------------------------------------------------------------

    def _create_peaks_hs_group(self):
        selected = self.sample_data.get("selected", {})
        rep_sel = selected.get("doc", "1")
        rep_data = (self.sample_data.get("replicas") or {}).get(rep_sel, {})
        n_peaks = rep_data.get("n_peaks_254_HS", 0)

        if not n_peaks:
            return None

        group = QGroupBox("Pics zona HS (254nm)")
        layout = QVBoxLayout(group)
        peaks_label = QLabel(
            f"<b>{n_peaks}</b> pics detectats a 254nm dins zona HS (18-23 min)"
        )
        peaks_label.setWordWrap(True)
        layout.addWidget(peaks_label)
        return group

    # ------------------------------------------------------------------
    # Graphs
    # ------------------------------------------------------------------

    def _plot_signals(self):
        """Gràfics grid 2 columnes (Proposta D): DOC|UIB + parells DAD + taula."""
        if not HAS_MATPLOTLIB:
            return

        from matplotlib.lines import Line2D

        self.figure.clear()
        replicas = self.sample_data.get("replicas", {})
        if not replicas:
            return

        rep_keys = sorted(replicas.keys())
        r1 = replicas.get(rep_keys[0], {})
        r2 = replicas.get(rep_keys[1], {}) if len(rep_keys) > 1 else None
        comparison = self.sample_data.get("comparison", {})
        doc_comp = comparison.get("doc", {})
        dad_comp = comparison.get("dad", {})

        # DAD wavelength columns
        df_dad1 = r1.get("df_dad")
        wl_cols = []
        if df_dad1 is not None and hasattr(df_dad1, 'columns'):
            wl_cols = [c for c in df_dad1.columns if c != 'time (min)']
            wl_cols.sort(key=lambda x: int(x) if str(x).isdigit() else 0)

        # Colors
        C1 = '#1565C0'
        C2 = '#E65100'
        C_UIB = '#2E7D32'
        C_UIB2 = '#66BB6A'
        LW = 0.7

        # X-axis limits
        x_min, x_max = (0, 15) if self.is_bp else (0, 70)

        # Selected replica data
        selected = self.sample_data.get("selected", {})
        rep_sel = selected.get("doc", rep_keys[0])
        sel_data = (replicas or {}).get(rep_sel, r1)
        sel_areas = sel_data.get("areas") or {}
        areas_uib = sel_data.get("areas_uib", {})
        n_peaks_per_wl = sel_data.get("n_peaks_per_wl", {})

        # Quantification
        quant = self.sample_data.get("quantification", {})
        ppm_direct = quant.get("concentration_ppm_direct") or quant.get("concentration_ppm", 0)
        ppm_uib = quant.get("concentration_ppm_uib", 0)

        # Fraccions from config
        from hpsec_config import get_config
        cfg = get_config()
        mode = "BP" if self.is_bp else "COLUMN"
        fracs = cfg.get_all_fractions(mode)

        # R² values
        pearson_doc = doc_comp.get("pearson", 0)
        pearson_per_wl = dad_comp.get("pearson_per_wavelength", {})

        # ── Annotation helper ──
        def _annotate(ax, r2v=None, ppm=None, sig_key=None):
            line1 = []
            if r2v and r2v > 0:
                line1.append(f"R\u00b2={r2v:.4f}")
            if ppm:
                line1.append(f"{ppm:.2f} ppm")
            # Pics nomes HS
            n_hs = n_peaks_per_wl.get(sig_key, {}).get("HS", 0)
            line2 = f"{n_hs}p HS" if n_hs else ""
            lines = []
            if line1:
                lines.append("  ".join(line1))
            if line2:
                lines.append(line2)
            if lines:
                clr = '#C62828' if (r2v and r2v < 0.990) else '#555'
                ax.text(0.99, 0.92, "\n".join(lines),
                        transform=ax.transAxes, fontsize=4.5,
                        color=clr, ha='right', va='top', linespacing=1.3)

        # ── Fraction vlines helper ──
        def _add_vlines(ax):
            for fname, finfo in fracs:
                s = finfo['start']
                if s > 0 and s <= x_max:
                    ax.axvline(s, color='#999', ls=':', lw=0.5, zorder=0)

        # ── GridSpec: 2 columns ──
        pairs = []
        for i in range(0, len(wl_cols), 2):
            if i + 1 < len(wl_cols):
                pairs.append((wl_cols[i], wl_cols[i + 1]))
            else:
                pairs.append((wl_cols[i], None))

        n_graph_rows = 1 + len(pairs)  # DOC/UIB + DAD pairs
        n_total_rows = n_graph_rows + 1  # + table
        h_graphs = [1.0] * n_graph_rows
        h_table = [2.5]
        heights = h_graphs + h_table

        gs = self.figure.add_gridspec(
            n_total_rows, 2,
            height_ratios=heights,
            hspace=0.30, wspace=0.22,
            top=0.94, bottom=0.03, left=0.08, right=0.97
        )

        all_graph_axes = []

        # ── Row 0: DOC Direct | DOC UIB ──
        ax_doc = self.figure.add_subplot(gs[0, 0])
        ax_uib = self.figure.add_subplot(gs[0, 1])
        all_graph_axes.extend([ax_doc, ax_uib])

        # DOC Direct
        t1 = r1.get("t_doc")
        y1_d = r1.get("y_doc_net")
        y1_u = r1.get("y_doc_uib_net")
        t2_arr, y2_d_arr, y2_u_arr = None, None, None

        if t1 is not None and y1_d is not None:
            t1 = np.asarray(t1)
            y1_d = np.asarray(y1_d)
            ax_doc.plot(t1, y1_d, color=C1, lw=LW, label=f'R{rep_keys[0]}')
            if r2:
                t2_arr = r2.get("t_doc")
                y2_d_arr = r2.get("y_doc_net")
                if t2_arr is not None and y2_d_arr is not None:
                    t2_arr = np.asarray(t2_arr)
                    y2_d_arr = np.asarray(y2_d_arr)
                    ax_doc.plot(t2_arr, y2_d_arr, color=C2, lw=LW, alpha=0.7,
                                label=f'R{rep_keys[1]}')

        ax_doc.set_ylabel("DOC", fontsize=6.5, labelpad=2)
        ax_doc.tick_params(labelsize=5.5, length=2, pad=1)
        ax_doc.grid(True, alpha=0.2, lw=0.3)
        ax_doc.set_xlim(x_min, x_max)
        _add_vlines(ax_doc)
        ax_doc.legend(loc='upper left', fontsize=5, ncol=2,
                      framealpha=0.7, handlelength=1.2)
        _annotate(ax_doc, r2v=pearson_doc, ppm=ppm_direct, sig_key="DOC")

        # DOC UIB
        has_uib = y1_u is not None
        if has_uib:
            y1_u = np.asarray(y1_u)
            if len(y1_u) == len(t1):
                ax_uib.plot(t1, y1_u, color=C_UIB, lw=LW, label=f'R{rep_keys[0]}')
                if r2:
                    y2_u_arr = r2.get("y_doc_uib_net")
                    if y2_u_arr is not None and t2_arr is not None:
                        y2_u_arr = np.asarray(y2_u_arr)
                        if len(y2_u_arr) == len(t2_arr):
                            ax_uib.plot(t2_arr, y2_u_arr, color=C_UIB2, lw=LW,
                                        alpha=0.7, label=f'R{rep_keys[1]}')
                ax_uib.legend(loc='upper left', fontsize=5, ncol=2,
                              framealpha=0.7, handlelength=1.2)
                _annotate(ax_uib, ppm=ppm_uib, sig_key="UIB")
            else:
                has_uib = False

        if not has_uib:
            ax_uib.text(0.5, 0.5, "UIB no disponible",
                        ha='center', va='center',
                        transform=ax_uib.transAxes, fontsize=8, color='#aaa')

        ax_uib.set_ylabel("UIB", fontsize=6.5, labelpad=2)
        ax_uib.tick_params(labelsize=5.5, length=2, pad=1)
        ax_uib.grid(True, alpha=0.2, lw=0.3)
        ax_uib.set_xlim(x_min, x_max)
        _add_vlines(ax_uib)

        # ── DAD rows (parells) ──
        for row_i, (wl_left, wl_right) in enumerate(pairs):
            for col_j, wl in enumerate([wl_left, wl_right]):
                if wl is None:
                    ax = self.figure.add_subplot(gs[row_i + 1, col_j])
                    ax.axis('off')
                    all_graph_axes.append(ax)
                    continue

                ax = self.figure.add_subplot(gs[row_i + 1, col_j])
                all_graph_axes.append(ax)

                if (df_dad1 is not None and 'time (min)' in df_dad1.columns
                        and wl in df_dad1.columns):
                    ax.plot(df_dad1['time (min)'].values,
                            df_dad1[wl].values, color=C1, lw=LW)
                    if r2:
                        df_dad2 = r2.get("df_dad")
                        if (df_dad2 is not None
                                and hasattr(df_dad2, 'columns')
                                and wl in df_dad2.columns):
                            ax.plot(df_dad2['time (min)'].values,
                                    df_dad2[wl].values,
                                    color=C2, lw=LW, alpha=0.7)

                wl_label = f"A{wl}" if not str(wl).startswith('A') else wl
                ax.set_ylabel(wl_label, fontsize=6.5, labelpad=2)
                ax.grid(True, alpha=0.2, lw=0.3)
                ax.tick_params(labelsize=5.5, length=2, pad=1)
                ax.set_xlim(x_min, x_max)
                _add_vlines(ax)

                # R² (clau pot ser "A254" o "254")
                wl_key = f"A{wl}" if not str(wl).startswith('A') else wl
                r2v = pearson_per_wl.get(wl_key, 0) or pearson_per_wl.get(str(wl), 0)
                _annotate(ax, r2v=r2v, sig_key=wl_key)

        # X label on bottom row
        bottom_row = n_graph_rows - 1
        for col_j in range(2):
            idx = 2 + bottom_row * 2 + col_j
            if idx < len(all_graph_axes):
                ax = all_graph_axes[idx]
                if ax.axison:
                    ax.set_xlabel("Temps (min)", fontsize=6.5)

        # ── Fraction table (bottom, spans 2 columns) ──
        if not self.is_bp and fracs:
            ax_tbl = self.figure.add_subplot(gs[n_graph_rows, :])
            ax_tbl.axis('off')

            doc_areas = sel_areas.get("DOC", {})
            doc_total = doc_areas.get("total", 0)
            uib_total = areas_uib.get("total", 0)

            # Header: Senyal | BioP (10.8-18) | HS (18-23) | ... | TOTAL (0-70)
            col_labels = ["Senyal"]
            for fname, finfo in fracs:
                col_labels.append(f"{fname} ({finfo['start']:g}\u2013{finfo['end']:g})")
            col_labels.append(f"TOTAL (0\u2013{x_max:g})")

            # Signal rows
            signal_names = ["DOC"]
            if has_uib:
                signal_names.append("UIB")
            for wl in wl_cols:
                wl_lbl = f"A{wl}" if not str(wl).startswith('A') else wl
                signal_names.append(wl_lbl)

            rows = []
            for sig in signal_names:
                row = [sig]
                if sig == "DOC":
                    sig_areas, sig_total = doc_areas, doc_total
                elif sig == "UIB":
                    sig_areas, sig_total = areas_uib, uib_total
                else:
                    sig_areas = sel_areas.get(sig, {})
                    sig_total = sig_areas.get("total", 0)
                for fname, _finfo in fracs:
                    fval = sig_areas.get(fname, 0)
                    pct = (fval / sig_total * 100) if sig_total > 0 else 0
                    row.append(f"{pct:.1f}")
                row.append("100" if sig_total > 0 else "\u2013")
                rows.append(row)

            tbl = ax_tbl.table(cellText=rows, colLabels=col_labels,
                               loc='upper center', cellLoc='center')
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(6)
            tbl.scale(1, 1.2)
            for key, cell in tbl.get_celld().items():
                cell.set_linewidth(0.3)
                cell.set_height(0.08)
                if key[0] == 0:
                    cell.set_facecolor('#E0E0E0')
                    cell.set_text_props(fontweight='bold', fontsize=5.5)
                elif key[1] == 0:
                    cell.set_facecolor('#F5F5F5')
                    cell.set_text_props(fontweight='bold', fontsize=6)
                else:
                    cell.set_facecolor('white')

        # ── Title ──
        rep_label = f"R{rep_keys[0]}"
        if r2:
            rep_label += f"+R{rep_keys[1]}"
        self.figure.suptitle(
            f"{self.sample_name}  |  {self.method}  |  {rep_label}",
            fontsize=9, fontweight='bold', y=0.98)

        self.canvas.draw()

