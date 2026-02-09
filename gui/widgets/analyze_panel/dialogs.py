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
        self.setMinimumSize(1100, 750)
        self.setModal(True)

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Splitter principal
        splitter = QSplitter(Qt.Horizontal)

        # === LEFT: GRAPHS ===
        graph_widget = QWidget()
        graph_layout = QVBoxLayout(graph_widget)
        graph_layout.setContentsMargins(0, 0, 0, 0)

        if HAS_MATPLOTLIB:
            self.figure = Figure(figsize=(8, 9), dpi=100)
            self.canvas = FigureCanvas(self.figure)
            graph_layout.addWidget(self.canvas)
            self._plot_signals()
        else:
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
        if not HAS_MATPLOTLIB:
            return

        self.figure.clear()
        replicas = self.sample_data.get("replicas", {})
        if not replicas:
            return

        rep_keys = sorted(replicas.keys())
        colors = {'r1': '#2196F3', 'r2': '#FF5722'}

        r1_data = replicas.get(rep_keys[0], {})
        r2_data = replicas.get(rep_keys[1], {}) if len(rep_keys) > 1 else None

        n_plots = 3
        axes = self.figure.subplots(n_plots, 1, sharex=True)

        # === Plot 1: DOC Direct ===
        ax1 = axes[0]
        t1 = r1_data.get("t_doc")
        y1 = r1_data.get("y_doc_net")

        if t1 is not None and y1 is not None:
            t1 = np.asarray(t1)
            y1 = np.asarray(y1)
            ax1.plot(t1, y1, color=colors['r1'], label=f'R{rep_keys[0]}', linewidth=1)

            if r2_data:
                t2 = r2_data.get("t_doc")
                y2 = r2_data.get("y_doc_net")
                if t2 is not None and y2 is not None:
                    t2 = np.asarray(t2)
                    y2 = np.asarray(y2)
                    ax1.plot(t2, y2, color=colors['r2'], label=f'R{rep_keys[1]}',
                            linewidth=1, linestyle='--', alpha=0.8)

        ax1.set_ylabel("DOC Direct (mAU)", fontsize=9)
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_title("DOC Direct", fontsize=10, fontweight='bold', loc='left')

        # === Plot 2: DOC UIB ===
        ax2 = axes[1]
        y1_uib = r1_data.get("y_doc_uib_net")

        if y1_uib is not None and t1 is not None:
            y1_uib = np.asarray(y1_uib)
            ax2.plot(t1, y1_uib, color=colors['r1'], label=f'R{rep_keys[0]}', linewidth=1)

            if r2_data:
                y2_uib = r2_data.get("y_doc_uib_net")
                if y2_uib is not None:
                    y2_uib = np.asarray(y2_uib)
                    t2 = r2_data.get("t_doc")
                    if t2 is not None:
                        t2 = np.asarray(t2)
                        ax2.plot(t2, y2_uib, color=colors['r2'], label=f'R{rep_keys[1]}',
                                linewidth=1, linestyle='--', alpha=0.8)

            ax2.set_title("DOC UIB", fontsize=10, fontweight='bold', loc='left')
        else:
            ax2.text(0.5, 0.5, "UIB no disponible", ha='center', va='center',
                    transform=ax2.transAxes, fontsize=10, color='#666')
            ax2.set_title("DOC UIB", fontsize=10, fontweight='bold', loc='left')

        ax2.set_ylabel("DOC UIB (mAU)", fontsize=9)
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True, alpha=0.3)

        # === Plot 3: DAD 254 ===
        ax3 = axes[2]
        df_dad1 = r1_data.get("df_dad")

        if df_dad1 is not None and not df_dad1.empty:
            wl_col = None
            for col in ['254', 'A254']:
                if col in df_dad1.columns:
                    wl_col = col
                    break

            if wl_col and 'time (min)' in df_dad1.columns:
                t_dad1 = df_dad1['time (min)'].values
                y_254_1 = df_dad1[wl_col].values
                ax3.plot(t_dad1, y_254_1, color=colors['r1'], label=f'R{rep_keys[0]}', linewidth=1)

                if r2_data:
                    df_dad2 = r2_data.get("df_dad")
                    if df_dad2 is not None and not df_dad2.empty and wl_col in df_dad2.columns:
                        t_dad2 = df_dad2['time (min)'].values
                        y_254_2 = df_dad2[wl_col].values
                        ax3.plot(t_dad2, y_254_2, color=colors['r2'], label=f'R{rep_keys[1]}',
                                linewidth=1, linestyle='--', alpha=0.8)

        ax3.set_ylabel("DAD 254nm (mAU)", fontsize=9)
        ax3.set_xlabel("Temps (min)", fontsize=9)
        ax3.legend(loc='upper right', fontsize=8)
        ax3.grid(True, alpha=0.3)
        ax3.set_title("DAD 254nm", fontsize=10, fontweight='bold', loc='left')

        # Fraction zones (COLUMN)
        if not self.is_bp:
            zones = [
                (0, 18, "BioP", "#E3F2FD"),
                (18, 23, "HS", "#FFF3E0"),
                (23, 30, "BB", "#F3E5F5"),
                (30, 40, "SB", "#E8F5E9"),
                (40, 70, "LMW", "#FCE4EC"),
            ]
            for ax in axes:
                for start, end, name, color in zones:
                    ax.axvspan(start, end, alpha=0.15, color=color, zorder=0)

        self.figure.tight_layout()
        self.canvas.draw()

