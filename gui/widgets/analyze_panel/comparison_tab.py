"""
HPSEC Suite - COL/BP Comparison Tab
=====================================

Comparació automàtica entre seqüències COLUMN i BP bessones:
- Detecció automàtica bessona (per número SEQ ±2)
- Taula ppm side-by-side amb ratio BP/COL
- Scatter + Bland-Altman
- Outliers amb causa probable
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QScrollArea,
    QGroupBox, QTableWidget, QTableWidgetItem, QHeaderView, QComboBox
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont, QColor

import os
import json
import numpy as np
import logging

logger = logging.getLogger(__name__)


# Ratio BP/COL esperat (mediana empírica sense outliers)
EXPECTED_RATIO = 0.956
RATIO_WARN_LOW = 0.75
RATIO_WARN_HIGH = 1.25


class _TwinSearchWorker(QThread):
    """Worker per cercar bessona per coincidencia de noms de mostra."""
    finished = Signal(dict)

    def __init__(self, seq_path, method, current_sample_names=None):
        super().__init__()
        self._seq_path = seq_path
        self._method = method
        self._current_sample_names = current_sample_names or set()

    def run(self):
        result = {"found": False, "twins": [], "siblings": [],
                  "twin_type": None, "error": None}
        try:
            from hpsec_consolidate import find_related_sequences
            related = find_related_sequences(self._seq_path, search_range=10)

            is_bp = self._method and "BP" in self._method.upper()
            if is_bp:
                candidates = related.get("column_seqs", [])
                twin_type = "COLUMN"
            else:
                candidates = related.get("bp_seqs", [])
                twin_type = "BP"

            if not candidates:
                result["error"] = f"No s'ha trobat cap sequencia {twin_type} bessona"
                self.finished.emit(result)
                return

            result["twin_type"] = twin_type

            # Carregar cada candidata i mesurar coincidencia de mostres
            scored = []
            for candidate in candidates:
                data_folder = os.path.join(candidate, "CHECK", "data")
                json_path = os.path.join(data_folder, "analysis_result.json")
                if not os.path.exists(json_path):
                    continue
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        twin_data = json.load(f)
                    twin_grouped = twin_data.get("samples_grouped", {})
                    twin_names = {n for n, s in twin_grouped.items()
                                  if s.get("sample_type", "SAMPLE") == "SAMPLE"}
                    overlap = self._current_sample_names & twin_names
                    pct = len(overlap) / len(self._current_sample_names) * 100 \
                        if self._current_sample_names else 0
                    scored.append({
                        "path": candidate,
                        "name": os.path.basename(candidate),
                        "data": twin_data,
                        "overlap": len(overlap),
                        "overlap_pct": pct,
                        "twin_names": twin_names,
                    })
                except Exception as e:
                    logger.warning(f"Error llegint {json_path}: {e}")

            if not scored:
                names = [os.path.basename(c) for c in candidates]
                result["error"] = (f"Bessones {twin_type} trobades ({', '.join(names)}) "
                                   "pero cap amb analisi completada")
                self.finished.emit(result)
                return

            # Ordenar per coincidencia descendent
            scored.sort(key=lambda s: s["overlap"], reverse=True)

            # Agafar la millor (o les que comparteixin mostres)
            best = scored[0]
            if best["overlap"] == 0:
                result["error"] = (
                    f"Cap bessona {twin_type} comparteix mostres amb la SEQ actual.\n"
                    f"Candidates: {', '.join(s['name'] for s in scored)}")
                self.finished.emit(result)
                return

            # Incloure totes les bessones que comparteixin >=50% de les seves mostres
            # amb les de la SEQ actual (pot ser >1 si les mostres estan repartides)
            for s in scored:
                if s["overlap"] > 0:
                    result["twins"].append({
                        "path": s["path"],
                        "name": s["name"],
                        "data": s["data"],
                        "overlap": s["overlap"],
                        "overlap_pct": s["overlap_pct"],
                    })
                    logger.info(
                        f"Bessona {s['name']}: {s['overlap']} mostres comunes "
                        f"({s['overlap_pct']:.0f}%)")

            result["found"] = True
            # No siblings — ja no barregem SEQs del mateix tipus

        except Exception as e:
            result["error"] = str(e)

        self.finished.emit(result)


class ComparisonTab(QWidget):
    """Tab de comparació COLUMN vs BP."""

    def __init__(self, main_window=None):
        super().__init__()
        self._main_window = main_window
        self._populated = False
        self._worker = None
        self._canvases = []
        self._current_data = None
        self._twin_data = None
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

        self._status_label = QLabel("Analitza una seqüència per comparar COL/BP")
        self._status_label.setFont(QFont("Segoe UI", 11))
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_label.setStyleSheet("color: #888; padding: 40px;")
        self._content.addWidget(self._status_label)

        self._content.addStretch()
        scroll.setWidget(self._container)

    def populate(self, result: dict):
        """Pobla el tab: cerca bessona en background."""
        self._populated = True
        self._current_data = result
        self._clear_content()

        if not result:
            self._status_label.setText("Sense dades d'anàlisi")
            self._status_label.setVisible(True)
            return

        seq_path = result.get("seq_path", "")
        method = result.get("method", "")

        # Si seq_path no és absolut, recuperar-lo del main_window
        if not seq_path or not os.path.isdir(seq_path):
            if self._main_window and hasattr(self._main_window, "seq_path"):
                seq_path = self._main_window.seq_path or ""

        if not seq_path or not os.path.isdir(seq_path):
            self._status_label.setText("Path de seqüència no vàlid")
            self._status_label.setVisible(True)
            return

        self._status_label.setText("Cercant bessona per nom de mostra...")
        self._status_label.setVisible(True)

        # Extreure noms de mostra actuals (nomes SAMPLE, no KHP/BLANK/CONTROL)
        current_grouped = result.get("samples_grouped", {})
        current_sample_names = {
            n for n, s in current_grouped.items()
            if s.get("sample_type", "SAMPLE") == "SAMPLE"
        }

        self._worker = _TwinSearchWorker(seq_path, method, current_sample_names)
        self._worker.finished.connect(self._on_twin_found)
        self._worker.start()

    def _on_twin_found(self, result: dict):
        """Callback quan la cerca de bessona acaba."""
        self._clear_content()

        if not result["found"]:
            self._status_label.setText(result.get("error", "Bessona no trobada"))
            self._status_label.setVisible(True)
            return

        self._status_label.setVisible(False)
        twins = result.get("twins", [])
        twin_type = result.get("twin_type", "?")

        # Combinar samples_grouped de les bessones (nomes les que comparteixen mostres)
        combined_twin_grouped = {}
        twin_names = []
        for twin in twins:
            twin_names.append(
                f"{twin['name']} ({twin.get('overlap', 0)} mostres, "
                f"{twin.get('overlap_pct', 0):.0f}%)")
            sg = twin["data"].get("samples_grouped", {})
            for name, sdata in sg.items():
                if name not in combined_twin_grouped:
                    combined_twin_grouped[name] = sdata

        self._twin_data = {"samples_grouped": combined_twin_grouped}
        self._current_data_combined = self._current_data

        current_method = self._current_data.get("method", "?")
        current_name = self._current_data.get("seq_name", "?")
        twin_label = " + ".join(twin_names)

        header_text = f"Comparacio: {current_name} ({current_method}) vs {twin_label} ({twin_type})"
        header = QLabel(header_text)
        header.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        header.setWordWrap(True)
        self._content.addWidget(header)

        # Emparejar mostres per nom
        pairs = self._match_samples()

        if not pairs:
            lbl = QLabel("No s'han trobat mostres comunes entre les dues seqüències")
            lbl.setStyleSheet("color: #e03131; padding: 20px;")
            self._content.addWidget(lbl)
            return

        # Taula comparativa
        self._add_comparison_table(pairs, current_method)

        # Gràfics
        self._add_scatter_and_bland_altman(pairs, current_name, twin_label)

        # Resum estadístic
        self._add_summary(pairs)

        self._content.addStretch()

    def _match_samples(self):
        """Emparella mostres entre seqüència actual (+germanes) i bessona per nom."""
        current_src = getattr(self, "_current_data_combined", self._current_data)
        if not current_src or not self._twin_data:
            return []

        current_grouped = current_src.get("samples_grouped", {})
        twin_grouped = self._twin_data.get("samples_grouped", {})

        current_method = self._current_data.get("method", "COLUMN")
        is_current_bp = "BP" in current_method.upper()

        pairs = []
        for name in current_grouped:
            if name in twin_grouped:
                c_data = current_grouped[name]
                t_data = twin_grouped[name]

                # Ignorar blancs/controls/KHP
                c_type = c_data.get("sample_type", "SAMPLE")
                t_type = t_data.get("sample_type", "SAMPLE")
                if c_type != "SAMPLE" or t_type != "SAMPLE":
                    continue

                c_quant = c_data.get("quantification", {})
                t_quant = t_data.get("quantification", {})
                c_ppm = c_quant.get("concentration_ppm")
                t_ppm = t_quant.get("concentration_ppm")

                if c_ppm is None or t_ppm is None:
                    continue
                if c_ppm <= 0 and t_ppm <= 0:
                    continue

                if is_current_bp:
                    bp_ppm, col_ppm = c_ppm, t_ppm
                    bp_data, col_data = c_data, t_data
                else:
                    col_ppm, bp_ppm = c_ppm, t_ppm
                    col_data, bp_data = c_data, t_data

                ratio = bp_ppm / col_ppm if col_ppm > 0 else None
                diff = bp_ppm - col_ppm

                # Extreure A254 de cada mode
                col_a254 = self._get_area_a254(col_data)
                bp_a254 = self._get_area_a254(bp_data)
                ratio_a254 = (bp_a254 / col_a254) if col_a254 and col_a254 > 0 and bp_a254 else None

                # Detectar anomalies per causa probable
                cause = self._diagnose_outlier(name, c_data, t_data, ratio, is_current_bp)

                pairs.append({
                    "name": name,
                    "col_ppm": col_ppm,
                    "bp_ppm": bp_ppm,
                    "ratio": ratio,
                    "diff": diff,
                    "col_a254": col_a254,
                    "bp_a254": bp_a254,
                    "ratio_a254": ratio_a254,
                    "cause": cause,
                })

        pairs.sort(key=lambda p: p["name"])
        return pairs

    @staticmethod
    def _get_area_a254(sample_data):
        """Extreu area total A254 de la replica seleccionada."""
        selected = sample_data.get("selected", {})
        sel = selected.get("dad", selected.get("doc", "1"))
        replicas = sample_data.get("replicas", {})
        rep = replicas.get(sel, {})
        if not isinstance(rep, dict):
            return None
        areas = rep.get("areas", {})
        a254 = areas.get("A254", {})
        total = a254.get("total", 0)
        return total if total and total > 0 else None

    def _diagnose_outlier(self, name, c_data, t_data, ratio, is_current_bp):
        """Diagnostica causa probable d'un outlier."""
        if ratio is None:
            return ""
        if RATIO_WARN_LOW <= ratio <= RATIO_WARN_HIGH:
            return ""

        causes = []
        bp_data = c_data if is_current_bp else t_data
        col_data = t_data if is_current_bp else c_data

        # Check UIB saturation
        bp_reps = bp_data.get("replicas", {})
        for rk, rd in bp_reps.items():
            if rd.get("uib_saturated"):
                causes.append("Saturació UIB (BP)")
                break

        col_reps = col_data.get("replicas", {})
        for rk, rd in col_reps.items():
            if rd.get("uib_saturated"):
                causes.append("Saturació UIB (COL)")
                break

        # Check timeout in peak
        for rk, rd in bp_reps.items():
            if rd.get("timeout_in_peak"):
                causes.append("Timeout dins pic (BP)")
                break
        for rk, rd in col_reps.items():
            if rd.get("timeout_in_peak"):
                causes.append("Timeout dins pic (COL)")
                break

        # Check irregular top
        for rk, rd in bp_reps.items():
            if rd.get("irregular_top_direct") or rd.get("irregular_top_uib"):
                causes.append("Pic irregular (BP)")
                break
        for rk, rd in col_reps.items():
            if rd.get("irregular_top_direct") or rd.get("irregular_top_uib"):
                causes.append("Pic irregular (COL)")
                break

        # If ratio very far off, suspect volume
        if ratio and (ratio < 0.5 or ratio > 2.0):
            causes.append("Possible volum incorrecte")

        return "; ".join(causes) if causes else "Causa desconeguda"

    def _add_comparison_table(self, pairs, current_method):
        """Taula ppm + A254 side-by-side."""
        group = QGroupBox(f"Comparacio DOC + A254 ({len(pairs)} mostres comunes)")
        group.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        group_layout = QVBoxLayout(group)

        # Check if any pair has A254 data
        has_a254 = any(p.get("col_a254") or p.get("bp_a254") for p in pairs)

        if has_a254:
            n_cols = 8
            headers = ["Mostra", "ppm COL", "ppm BP", "Ratio DOC",
                        "A254 COL", "A254 BP", "Ratio A254", "Causa"]
        else:
            n_cols = 5
            headers = ["Mostra", "ppm COL", "ppm BP", "Ratio BP/COL", "Causa"]

        table = QTableWidget(len(pairs), n_cols)
        table.setHorizontalHeaderLabels(headers)
        table.horizontalHeader().setStretchLastSection(True)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        for col in range(1, n_cols - 1):
            table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        table.verticalHeader().setVisible(False)
        table.setAlternatingRowColors(True)

        for row, pair in enumerate(pairs):
            ci = 0  # column index

            # Nom
            table.setItem(row, ci, QTableWidgetItem(pair["name"]))
            ci += 1

            # ppm COL
            item_col = QTableWidgetItem(f"{pair['col_ppm']:.3f}")
            item_col.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            table.setItem(row, ci, item_col)
            ci += 1

            # ppm BP
            item_bp = QTableWidgetItem(f"{pair['bp_ppm']:.3f}")
            item_bp.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            table.setItem(row, ci, item_bp)
            ci += 1

            # Ratio DOC
            ratio = pair["ratio"]
            if ratio is not None:
                item_ratio = QTableWidgetItem(f"{ratio:.3f}")
                item_ratio.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if ratio < RATIO_WARN_LOW or ratio > RATIO_WARN_HIGH:
                    item_ratio.setBackground(QColor("#ffe0e0"))
                    item_ratio.setForeground(QColor("#c92a2a"))
                elif abs(ratio - EXPECTED_RATIO) > 0.1:
                    item_ratio.setBackground(QColor("#fff3bf"))
                table.setItem(row, ci, item_ratio)
            else:
                table.setItem(row, ci, QTableWidgetItem("—"))
            ci += 1

            # A254 columns (if available)
            if has_a254:
                col_a254 = pair.get("col_a254")
                bp_a254 = pair.get("bp_a254")
                ratio_a254 = pair.get("ratio_a254")

                # A254 COL
                item_a254_col = QTableWidgetItem(f"{col_a254:.1f}" if col_a254 else "—")
                item_a254_col.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                table.setItem(row, ci, item_a254_col)
                ci += 1

                # A254 BP
                item_a254_bp = QTableWidgetItem(f"{bp_a254:.1f}" if bp_a254 else "—")
                item_a254_bp.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                table.setItem(row, ci, item_a254_bp)
                ci += 1

                # Ratio A254
                if ratio_a254 is not None:
                    item_ra = QTableWidgetItem(f"{ratio_a254:.3f}")
                    item_ra.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                    if ratio_a254 < RATIO_WARN_LOW or ratio_a254 > RATIO_WARN_HIGH:
                        item_ra.setBackground(QColor("#ffe0e0"))
                        item_ra.setForeground(QColor("#c92a2a"))
                    elif abs(ratio_a254 - EXPECTED_RATIO) > 0.1:
                        item_ra.setBackground(QColor("#fff3bf"))
                    table.setItem(row, ci, item_ra)
                else:
                    table.setItem(row, ci, QTableWidgetItem("—"))
                ci += 1

            # Causa
            cause_item = QTableWidgetItem(pair.get("cause", ""))
            if pair.get("cause"):
                cause_item.setForeground(QColor("#e03131"))
            table.setItem(row, ci, cause_item)

        table.setMaximumHeight(min(400, 30 + len(pairs) * 26))
        group_layout.addWidget(table)
        self._content.addWidget(group)

    def _add_scatter_and_bland_altman(self, pairs, current_name, twin_name):
        """Scatter DOC + A254 COL vs BP + Bland-Altman."""
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
        except ImportError:
            return

        col_ppms = np.array([p["col_ppm"] for p in pairs])
        bp_ppms = np.array([p["bp_ppm"] for p in pairs])
        ratios = np.array([p["ratio"] for p in pairs if p["ratio"] is not None])

        has_a254 = any(p.get("col_a254") and p.get("bp_a254") for p in pairs)

        group = QGroupBox("Grafics comparatius")
        group.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        group_layout = QVBoxLayout(group)

        n_plots = 4 if has_a254 else 3
        fig = Figure(figsize=(3.3 * n_plots, 4), dpi=96)
        fig.set_facecolor("white")
        canvas = FigureCanvasQTAgg(fig)
        self._canvases.append(canvas)

        outlier_mask = np.array([
            p["ratio"] is not None and (p["ratio"] < RATIO_WARN_LOW or p["ratio"] > RATIO_WARN_HIGH)
            for p in pairs
        ])

        # --- Scatter DOC ---
        ax1 = fig.add_subplot(1, n_plots, 1)

        ax1.scatter(col_ppms[~outlier_mask], bp_ppms[~outlier_mask],
                    c="#228be6", s=25, alpha=0.7, label="Normal", edgecolors="none")
        if outlier_mask.any():
            ax1.scatter(col_ppms[outlier_mask], bp_ppms[outlier_mask],
                        c="#e03131", s=35, alpha=0.8, marker="x", label="Outlier")

        max_val = max(col_ppms.max(), bp_ppms.max()) * 1.1
        ax1.plot([0, max_val], [0, max_val], "k--", alpha=0.3, linewidth=0.8)

        if len(col_ppms) >= 3:
            try:
                z = np.polyfit(col_ppms, bp_ppms, 1)
                x_fit = np.linspace(0, max_val, 50)
                ax1.plot(x_fit, np.polyval(z, x_fit), "r-", alpha=0.5, linewidth=1)
                ax1.set_title(f"DOC ppm (slope={z[0]:.3f})", fontsize=9)
            except Exception:
                ax1.set_title("DOC ppm COL vs BP", fontsize=9)
        else:
            ax1.set_title("DOC ppm COL vs BP", fontsize=9)

        ax1.set_xlabel("ppm COLUMN", fontsize=8)
        ax1.set_ylabel("ppm BP", fontsize=8)
        ax1.tick_params(labelsize=7)
        ax1.legend(fontsize=7)

        # --- Scatter A254 ---
        if has_a254:
            ax_a254 = fig.add_subplot(1, n_plots, 2)

            col_a254_vals = []
            bp_a254_vals = []
            a254_outlier = []
            a254_names = []
            for p in pairs:
                ca = p.get("col_a254")
                ba = p.get("bp_a254")
                if ca and ba:
                    col_a254_vals.append(ca)
                    bp_a254_vals.append(ba)
                    ra = p.get("ratio_a254")
                    a254_outlier.append(
                        ra is not None and (ra < RATIO_WARN_LOW or ra > RATIO_WARN_HIGH))
                    a254_names.append(p["name"])

            col_a254_arr = np.array(col_a254_vals)
            bp_a254_arr = np.array(bp_a254_vals)
            a254_out = np.array(a254_outlier)

            ax_a254.scatter(col_a254_arr[~a254_out], bp_a254_arr[~a254_out],
                            c="#40c057", s=25, alpha=0.7, label="Normal", edgecolors="none")
            if a254_out.any():
                ax_a254.scatter(col_a254_arr[a254_out], bp_a254_arr[a254_out],
                                c="#e03131", s=35, alpha=0.8, marker="x", label="Outlier")

            max_a254 = max(col_a254_arr.max(), bp_a254_arr.max()) * 1.1
            ax_a254.plot([0, max_a254], [0, max_a254], "k--", alpha=0.3, linewidth=0.8)

            if len(col_a254_arr) >= 3:
                try:
                    z254 = np.polyfit(col_a254_arr, bp_a254_arr, 1)
                    x_fit254 = np.linspace(0, max_a254, 50)
                    ax_a254.plot(x_fit254, np.polyval(z254, x_fit254),
                                 "r-", alpha=0.5, linewidth=1)
                    ax_a254.set_title(f"A254 (slope={z254[0]:.3f})", fontsize=9)
                except Exception:
                    ax_a254.set_title("A254 COL vs BP", fontsize=9)
            else:
                ax_a254.set_title("A254 COL vs BP", fontsize=9)

            ax_a254.set_xlabel("A254 COLUMN", fontsize=8)
            ax_a254.set_ylabel("A254 BP", fontsize=8)
            ax_a254.tick_params(labelsize=7)
            ax_a254.legend(fontsize=7)

        # --- Bland-Altman ---
        ax2 = fig.add_subplot(1, n_plots, n_plots - 1)
        means = (col_ppms + bp_ppms) / 2
        diffs = bp_ppms - col_ppms

        ax2.scatter(means[~outlier_mask], diffs[~outlier_mask],
                    c="#228be6", s=25, alpha=0.7, edgecolors="none")
        if outlier_mask.any():
            ax2.scatter(means[outlier_mask], diffs[outlier_mask],
                        c="#e03131", s=35, alpha=0.8, marker="x")

        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs)
        ax2.axhline(mean_diff, color="blue", linestyle="-", alpha=0.4)
        ax2.axhline(mean_diff + 1.96 * std_diff, color="red", linestyle="--", alpha=0.3)
        ax2.axhline(mean_diff - 1.96 * std_diff, color="red", linestyle="--", alpha=0.3)
        ax2.axhline(0, color="black", linestyle="-", alpha=0.2)

        ax2.set_title(f"Bland-Altman (bias={mean_diff:.3f})", fontsize=9)
        ax2.set_xlabel("Mitjana (COL+BP)/2", fontsize=8)
        ax2.set_ylabel("Diferencia (BP-COL)", fontsize=8)
        ax2.tick_params(labelsize=7)

        # --- Histograma ratios ---
        ax3 = fig.add_subplot(1, n_plots, n_plots)
        if len(ratios) >= 3:
            ax3.hist(ratios, bins=min(15, len(ratios)), color="#228be6", alpha=0.7,
                     edgecolor="white", label="DOC")
            ax3.axvline(EXPECTED_RATIO, color="green", linestyle="--", alpha=0.5,
                        label=f"Esperada ({EXPECTED_RATIO})")
            ax3.axvline(np.median(ratios), color="red", linestyle="-", alpha=0.5,
                        label=f"Med DOC ({np.median(ratios):.3f})")

            # Overlay A254 ratios if available
            if has_a254:
                ratios_a254 = [p["ratio_a254"] for p in pairs
                               if p.get("ratio_a254") is not None]
                if len(ratios_a254) >= 3:
                    ra_arr = np.array(ratios_a254)
                    ax3.hist(ra_arr, bins=min(15, len(ra_arr)), color="#40c057",
                             alpha=0.4, edgecolor="white", label="A254")
                    ax3.axvline(np.median(ra_arr), color="#2b8a3e", linestyle="-",
                                alpha=0.5,
                                label=f"Med A254 ({np.median(ra_arr):.3f})")
            ax3.legend(fontsize=6)
        ax3.set_title("Distribucio ratio BP/COL", fontsize=9)
        ax3.set_xlabel("Ratio", fontsize=8)
        ax3.set_ylabel("N", fontsize=8)
        ax3.tick_params(labelsize=7)

        fig.tight_layout()
        canvas.setFixedHeight(320)
        group_layout.addWidget(canvas)
        self._content.addWidget(group)

    def _add_summary(self, pairs):
        """Resum estadistic de la comparacio (DOC + A254)."""
        ratios = [p["ratio"] for p in pairs if p["ratio"] is not None]
        outliers = [p for p in pairs if p["ratio"] is not None
                    and (p["ratio"] < RATIO_WARN_LOW or p["ratio"] > RATIO_WARN_HIGH)]

        if not ratios:
            return

        r_arr = np.array(ratios)
        median_r = np.median(r_arr)
        mean_r = np.mean(r_arr)
        std_r = np.std(r_arr)
        cv = std_r / mean_r * 100 if mean_r > 0 else 0

        group = QGroupBox("Resum estadistic")
        group.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        group_layout = QVBoxLayout(group)

        stats_text = (
            f"<b>DOC:</b> {len(pairs)} mostres  |  "
            f"Ratio BP/COL: mediana={median_r:.3f}, mitjana={mean_r:.3f}, "
            f"SD={std_r:.3f}, CV={cv:.1f}%  |  "
            f"Outliers: {len(outliers)}/{len(pairs)}"
        )

        # A254 stats
        ratios_a254 = [p["ratio_a254"] for p in pairs if p.get("ratio_a254") is not None]
        if ratios_a254:
            ra_arr = np.array(ratios_a254)
            median_a = np.median(ra_arr)
            mean_a = np.mean(ra_arr)
            std_a = np.std(ra_arr)
            cv_a = std_a / mean_a * 100 if mean_a > 0 else 0
            outliers_a254 = [r for r in ratios_a254
                             if r < RATIO_WARN_LOW or r > RATIO_WARN_HIGH]
            stats_text += (
                f"<br><b>A254:</b> {len(ratios_a254)} mostres  |  "
                f"Ratio BP/COL: mediana={median_a:.3f}, mitjana={mean_a:.3f}, "
                f"SD={std_a:.3f}, CV={cv_a:.1f}%  |  "
                f"Outliers: {len(outliers_a254)}/{len(ratios_a254)}"
            )

        stats_lbl = QLabel(stats_text)
        stats_lbl.setFont(QFont("Segoe UI", 9))
        stats_lbl.setWordWrap(True)
        group_layout.addWidget(stats_lbl)

        # Valoracio
        if len(outliers) == 0 and cv < 15:
            val_text = "Concordanca COL/BP excel-lent"
            val_color = "#2b8a3e"
        elif len(outliers) <= 2 and cv < 25:
            val_text = "Concordanca COL/BP acceptable — revisar outliers"
            val_color = "#e67700"
        else:
            val_text = ("Concordanca COL/BP deficient — probable problema sistematic "
                       "(volums, calibracio, delay)")
            val_color = "#c92a2a"

        val_lbl = QLabel(val_text)
        val_lbl.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        val_lbl.setStyleSheet(f"color: {val_color}; padding: 6px;")
        group_layout.addWidget(val_lbl)

        self._content.addWidget(group)

    def _clear_content(self):
        """Neteja tot excepte status_label."""
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
            if w and w is not self._status_label:
                w.setParent(None)
                w.deleteLater()

        self._status_label.setVisible(True)
        self._content.addWidget(self._status_label)
        self._content.addStretch()

    def reset(self):
        """Reinicia el tab."""
        self._populated = False
        self._current_data = None
        self._twin_data = None
        if self._worker and self._worker.isRunning():
            self._worker.quit()
            self._worker.wait(1000)
        self._worker = None
        self._clear_content()
        self._status_label.setText("Analitza una seqüència per comparar COL/BP")
