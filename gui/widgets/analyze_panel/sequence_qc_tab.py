"""
HPSEC Suite - Sequence QC Tab
=============================

Control de qualitat a nivell de seqüència:
- Evolució baseline al llarg de les injeccions
- Seguiment àrea blancs (MQ) amb context carry-over
- Efectivitat rentats NaOH
- Timeline timeouts TOC
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QScrollArea,
    QSplitter, QGroupBox
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

import numpy as np
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sorted_by_injection(samples_grouped: dict) -> list:
    """Retorna llista de (name, data) ordenada per injection_index."""
    entries = []
    for name, sdata in samples_grouped.items():
        reps = sdata.get("replicas", {})
        # Agafar injection_index del primer replica
        first_rep = next(iter(reps.values()), {}) if reps else sdata
        idx = first_rep.get("injection_index", 999)
        entries.append((name, sdata, idx))
    entries.sort(key=lambda x: x[2])
    return entries


def _get_baseline_and_snr(rep_data: dict):
    """Extreu baseline level i SNR d'una rèplica."""
    pi = rep_data.get("peak_info", {})
    si = rep_data.get("snr_info", {})
    baseline = pi.get("baseline_level") or si.get("baseline_noise_direct")
    snr = si.get("snr_direct") or si.get("snr_value")
    return baseline, snr


def _get_area_total(rep_data: dict):
    """Extreu àrea total DOC d'una rèplica."""
    areas = rep_data.get("areas", {})
    doc = areas.get("DOC", areas)
    return doc.get("total", doc.get("total_all", 0))


# ---------------------------------------------------------------------------
# Main widget
# ---------------------------------------------------------------------------

class SequenceQCTab(QWidget):
    """Tab de control de qualitat a nivell de seqüència."""

    def __init__(self, main_window=None):
        super().__init__()
        self._main_window = main_window
        self._populated = False
        self._canvases = []  # keep references
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        layout.addWidget(scroll)

        self._container = QWidget()
        self._content = QVBoxLayout(self._container)
        self._content.setSpacing(12)

        # Placeholder per quan no hi ha dades
        self._placeholder = QLabel("Analitza una seqüència per veure el QC")
        self._placeholder.setFont(QFont("Segoe UI", 11))
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet("color: #888; padding: 40px;")
        self._content.addWidget(self._placeholder)

        self._content.addStretch()
        scroll.setWidget(self._container)

    def populate(self, result: dict):
        """Pobla el tab amb dades d'anàlisi."""
        self._populated = True
        self._clear_content()

        if not result:
            self._placeholder.setText("Sense dades d'anàlisi")
            self._placeholder.setVisible(True)
            return

        samples_grouped = result.get("samples_grouped", {})
        if not samples_grouped:
            # Fallback: construir des de samples[]
            samples_list = result.get("samples", [])
            if isinstance(samples_list, list):
                for s in samples_list:
                    n = s.get("name", "?")
                    if n not in samples_grouped:
                        samples_grouped[n] = {"replicas": {}, "sample_type": s.get("sample_type", "SAMPLE")}
                    samples_grouped[n]["replicas"][s.get("replica", "1")] = s

        if not samples_grouped:
            self._placeholder.setText("Sense mostres analitzades")
            self._placeholder.setVisible(True)
            return

        self._placeholder.setVisible(False)
        seq_name = result.get("seq_name", "")
        method = result.get("method", "?")

        # Header
        header = QLabel(f"QC Seqüència: {seq_name} ({method})")
        header.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self._content.addWidget(header)

        sorted_samples = _sorted_by_injection(samples_grouped)

        # 1. Evolució baseline + SNR
        self._add_baseline_chart(sorted_samples, method)

        # 2. Seguiment blancs (MQ)
        self._add_blank_tracking(sorted_samples)

        # 3. Timeline timeouts
        self._add_timeout_timeline(sorted_samples, method)

        # 4. Efectivitat NaOH
        self._add_naoh_effectiveness(sorted_samples)

        self._content.addStretch()

    def _clear_content(self):
        """Neteja tot el contingut excepte placeholder."""
        for canvas in self._canvases:
            try:
                canvas.setParent(None)
                canvas.deleteLater()
            except Exception:
                pass
        self._canvases = []

        while self._content.count() > 0:
            item = self._content.takeAt(0)
            w = item.widget()
            if w and w is not self._placeholder:
                w.setParent(None)
                w.deleteLater()
            elif item.layout():
                # Clear sub-layouts
                pass

        self._placeholder.setVisible(True)
        self._content.addWidget(self._placeholder)
        self._content.addStretch()

    def _add_baseline_chart(self, sorted_samples, method):
        """Gràfic: evolució baseline i SNR per injecció."""
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
        except ImportError:
            return

        inj_indices = []
        baselines = []
        snrs = []
        labels = []
        colors = []

        for name, sdata, idx in sorted_samples:
            stype = sdata.get("sample_type", "SAMPLE")
            reps = sdata.get("replicas", {})
            for rep_key, rep_data in reps.items():
                bl, snr = _get_baseline_and_snr(rep_data)
                if bl is not None:
                    inj_idx = rep_data.get("injection_index", idx)
                    inj_indices.append(inj_idx)
                    baselines.append(bl)
                    snrs.append(snr or 0)
                    labels.append(f"{name} R{rep_key}")
                    if stype == "BLANK":
                        colors.append("#4dabf7")
                    elif stype == "CONTROL":
                        colors.append("#adb5bd")
                    elif stype == "KHP":
                        colors.append("#ffa94d")
                    else:
                        colors.append("#51cf66")

        if not inj_indices:
            return

        group = QGroupBox("Evolució Baseline i SNR")
        group.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        group_layout = QVBoxLayout(group)

        fig = Figure(figsize=(10, 3.5), dpi=96)
        fig.set_facecolor("white")
        canvas = FigureCanvasQTAgg(fig)
        self._canvases.append(canvas)

        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        # Baseline
        ax1.scatter(inj_indices, baselines, c=colors, s=20, alpha=0.8, edgecolors="none")
        if len(baselines) > 2:
            z = np.polyfit(inj_indices, baselines, 1)
            x_line = np.linspace(min(inj_indices), max(inj_indices), 50)
            ax1.plot(x_line, np.polyval(z, x_line), "r--", alpha=0.5, linewidth=1)
            slope_per_inj = z[0]
            ax1.set_title(f"Baseline (tendència: {slope_per_inj:+.2f}/inj)", fontsize=9)
        else:
            ax1.set_title("Baseline", fontsize=9)
        ax1.set_xlabel("Injecció #", fontsize=8)
        ax1.set_ylabel("Baseline (ppb)", fontsize=8)
        ax1.tick_params(labelsize=7)

        # SNR
        valid_snr = [(i, s) for i, s in zip(inj_indices, snrs) if s > 0]
        if valid_snr:
            xs, ys = zip(*valid_snr)
            cs = [colors[inj_indices.index(x)] for x in xs]
            ax2.scatter(xs, ys, c=cs, s=20, alpha=0.8, edgecolors="none")
            ax2.axhline(10, color="orange", linestyle="--", alpha=0.4, linewidth=0.8)
            ax2.axhline(3, color="red", linestyle="--", alpha=0.4, linewidth=0.8)
            ax2.text(max(xs), 10, "LOQ", fontsize=7, color="orange", va="bottom")
            ax2.text(max(xs), 3, "LOD", fontsize=7, color="red", va="bottom")
        ax2.set_title("SNR", fontsize=9)
        ax2.set_xlabel("Injecció #", fontsize=8)
        ax2.set_ylabel("SNR", fontsize=8)
        ax2.tick_params(labelsize=7)

        fig.tight_layout()
        canvas.setFixedHeight(280)
        group_layout.addWidget(canvas)

        # Llegenda
        legend_layout = QHBoxLayout()
        for label, color in [("Mostra", "#51cf66"), ("Blanc", "#4dabf7"),
                             ("Control", "#adb5bd"), ("KHP", "#ffa94d")]:
            lbl = QLabel(f"● {label}")
            lbl.setStyleSheet(f"color: {color}; font-size: 9px; font-weight: bold;")
            legend_layout.addWidget(lbl)
        legend_layout.addStretch()
        group_layout.addLayout(legend_layout)

        self._content.addWidget(group)

    def _add_blank_tracking(self, sorted_samples):
        """Seguiment àrees blancs amb context (carry-over)."""
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
        except ImportError:
            return

        # Trobar blancs i el que hi ha abans/després
        all_entries = []
        for name, sdata, idx in sorted_samples:
            stype = sdata.get("sample_type", "SAMPLE")
            reps = sdata.get("replicas", {})
            first_rep = next(iter(reps.values()), {}) if reps else sdata
            area = _get_area_total(first_rep)
            inj_idx = first_rep.get("injection_index", idx)
            all_entries.append((name, stype, area, inj_idx))

        blank_entries = [(n, a, i) for n, st, a, i in all_entries if st == "BLANK"]
        if not blank_entries:
            return

        group = QGroupBox(f"Seguiment Blancs ({len(blank_entries)} blancs)")
        group.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        group_layout = QVBoxLayout(group)

        fig = Figure(figsize=(10, 2.8), dpi=96)
        fig.set_facecolor("white")
        canvas = FigureCanvasQTAgg(fig)
        self._canvases.append(canvas)
        ax = fig.add_subplot(111)

        b_names = [b[0] for b in blank_entries]
        b_areas = [b[1] for b in blank_entries]
        b_positions = list(range(len(blank_entries)))

        bars = ax.bar(b_positions, b_areas, color="#4dabf7", alpha=0.8, width=0.6)

        # Context: què hi havia abans de cada blanc
        for i, (bname, barea, bidx) in enumerate(blank_entries):
            # Trobar injecció anterior
            prev = [e for e in all_entries if e[3] < bidx]
            if prev:
                prev_entry = max(prev, key=lambda x: x[3])
                prev_name = prev_entry[0][:12]
                prev_type = prev_entry[1]
                ax.annotate(f"← {prev_name}\n({prev_type})",
                           xy=(i, barea), xytext=(i, barea * 1.1 + max(b_areas) * 0.05),
                           fontsize=6, ha="center", color="#666",
                           arrowprops=dict(arrowstyle="-", color="#ccc", lw=0.5))

        ax.set_xticks(b_positions)
        ax.set_xticklabels(b_names, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Àrea DOC total", fontsize=8)
        ax.set_title("Àrea blancs (carry-over)", fontsize=9)
        ax.tick_params(labelsize=7)

        # Línia mitjana
        mean_area = np.mean(b_areas)
        ax.axhline(mean_area, color="red", linestyle="--", alpha=0.4, linewidth=0.8)
        ax.text(len(b_positions) - 0.5, mean_area, f"x̄={mean_area:.0f}",
                fontsize=7, color="red", va="bottom")

        fig.tight_layout()
        canvas.setFixedHeight(240)
        group_layout.addWidget(canvas)
        self._content.addWidget(group)

    def _add_timeout_timeline(self, sorted_samples, method):
        """Timeline de timeouts TOC al llarg de la seqüència."""
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
        except ImportError:
            return

        timeout_data = []
        for name, sdata, idx in sorted_samples:
            reps = sdata.get("replicas", {})
            for rep_key, rep_data in reps.items():
                ti = rep_data.get("timeout_info", {})
                n_to = ti.get("n_timeouts", 0)
                positions = ti.get("t_positions", [])
                inj_idx = rep_data.get("injection_index", idx)
                stype = sdata.get("sample_type", "SAMPLE")
                in_peak = rep_data.get("timeout_in_peak", False)
                timeout_data.append({
                    "name": name, "rep": rep_key, "idx": inj_idx,
                    "n": n_to, "positions": positions, "type": stype,
                    "in_peak": in_peak
                })

        any_timeouts = any(d["n"] > 0 for d in timeout_data)
        if not any_timeouts:
            return

        group = QGroupBox("Timeline Timeouts TOC")
        group.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        group_layout = QVBoxLayout(group)

        fig = Figure(figsize=(10, 3), dpi=96)
        fig.set_facecolor("white")
        canvas = FigureCanvasQTAgg(fig)
        self._canvases.append(canvas)
        ax = fig.add_subplot(111)

        # Per cada injecció, marcar timeouts com a punts
        for d in timeout_data:
            for t_pos in d["positions"]:
                color = "red" if d["in_peak"] else "orange"
                marker = "x" if d["in_peak"] else "o"
                ax.scatter(d["idx"], t_pos, c=color, marker=marker, s=30, alpha=0.7,
                          zorder=5)

        # Línia predicció timeout (si hi ha prou punts)
        all_idx_pos = [(d["idx"], p) for d in timeout_data for p in d["positions"]]
        if len(all_idx_pos) >= 3:
            xs, ys = zip(*all_idx_pos)
            try:
                z = np.polyfit(xs, ys, 1)
                x_line = np.linspace(min(xs), max(xs), 50)
                ax.plot(x_line, np.polyval(z, x_line), "b--", alpha=0.3, linewidth=1)
                drift = z[0]
                ax.set_title(f"Timeouts TOC (drift: {drift:+.2f} min/inj)", fontsize=9)
            except Exception:
                ax.set_title("Timeouts TOC", fontsize=9)
        else:
            ax.set_title("Timeouts TOC", fontsize=9)

        ax.set_xlabel("Injecció #", fontsize=8)
        ax.set_ylabel("Posició timeout (min)", fontsize=8)
        ax.tick_params(labelsize=7)

        # Llegenda
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="orange", markersize=6,
                   label="Timeout fora pic"),
            Line2D([0], [0], marker="x", color="w", markerfacecolor="red",
                   markeredgecolor="red", markersize=6, label="Timeout dins pic"),
        ]
        ax.legend(handles=handles, fontsize=7, loc="upper right")

        fig.tight_layout()
        canvas.setFixedHeight(240)
        group_layout.addWidget(canvas)
        self._content.addWidget(group)

    def _add_naoh_effectiveness(self, sorted_samples):
        """Analitza efectivitat dels rentats NaOH."""
        # Trobar parells pre-NaOH → NaOH → post-NaOH (blanc)
        all_entries = []
        for name, sdata, idx in sorted_samples:
            stype = sdata.get("sample_type", "SAMPLE")
            reps = sdata.get("replicas", {})
            first_rep = next(iter(reps.values()), {}) if reps else sdata
            area = _get_area_total(first_rep)
            inj_idx = first_rep.get("injection_index", idx)
            all_entries.append((name, stype, area, inj_idx))

        # Trobar seqüències NaOH → Blanc
        wash_pairs = []
        for i, (name, stype, area, idx) in enumerate(all_entries):
            if stype == "CONTROL" and "NAOH" in name.upper():
                # Buscar blanc posterior
                for j in range(i + 1, min(i + 3, len(all_entries))):
                    if all_entries[j][1] == "BLANK":
                        # Buscar mostra pre-NaOH
                        pre_area = None
                        if i > 0:
                            pre_area = all_entries[i - 1][2]
                        post_area = all_entries[j][2]
                        wash_pairs.append({
                            "naoh": name,
                            "blank": all_entries[j][0],
                            "pre_area": pre_area,
                            "post_area": post_area,
                            "idx": idx,
                        })
                        break

        if not wash_pairs:
            return

        group = QGroupBox(f"Efectivitat Rentats NaOH ({len(wash_pairs)} cicles)")
        group.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        group_layout = QVBoxLayout(group)

        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
        except ImportError:
            return

        fig = Figure(figsize=(10, 2.5), dpi=96)
        fig.set_facecolor("white")
        canvas = FigureCanvasQTAgg(fig)
        self._canvases.append(canvas)
        ax = fig.add_subplot(111)

        x = list(range(len(wash_pairs)))
        pre_areas = [wp["pre_area"] or 0 for wp in wash_pairs]
        post_areas = [wp["post_area"] for wp in wash_pairs]
        reductions = []
        for pre, post in zip(pre_areas, post_areas):
            if pre > 0:
                reductions.append((1 - post / pre) * 100)
            else:
                reductions.append(0)

        width = 0.35
        ax.bar([i - width / 2 for i in x], pre_areas, width, label="Pre-NaOH",
               color="#ff8787", alpha=0.8)
        ax.bar([i + width / 2 for i in x], post_areas, width, label="Post-NaOH (blanc)",
               color="#4dabf7", alpha=0.8)

        # Etiquetes reducció %
        for i, red in enumerate(reductions):
            if red > 0:
                ax.text(i, max(pre_areas[i], post_areas[i]) * 1.05,
                        f"-{red:.0f}%", ha="center", fontsize=7, color="green")

        labels_x = [f"#{wp['idx']}" for wp in wash_pairs]
        ax.set_xticks(x)
        ax.set_xticklabels(labels_x, fontsize=7)
        ax.set_ylabel("Àrea DOC", fontsize=8)
        ax.set_title("Àrea pre vs post NaOH", fontsize=9)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)

        fig.tight_layout()
        canvas.setFixedHeight(220)
        group_layout.addWidget(canvas)
        self._content.addWidget(group)

    def reset(self):
        """Reinicia el tab al seu estat inicial."""
        self._populated = False
        self._clear_content()
