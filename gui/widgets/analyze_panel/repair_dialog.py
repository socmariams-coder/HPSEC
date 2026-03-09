"""
HPSEC Suite - Jagged Peak Repair Dialog
========================================

Diàleg dedicat per reparació de pics irregulars (jagged/batman).
Mostra gràfic before/after, taula de paràmetres, i botons d'acció.
Accessible des de la columna Estat de la taula d'anàlisi.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QFrame,
    QMessageBox, QGroupBox
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QBrush, QFont

import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class JaggedPeakRepairDialog(QDialog):
    """Diàleg de reparació de pic irregular amb preview gràfic i comparació paràmetres."""

    repair_applied = Signal(str, str, str)   # sample_name, rep_key, signal
    repair_undone = Signal(str, str, str)
    dismissed = Signal(str, str, str)
    reactivated = Signal(str, str, str)

    def __init__(self, sample_name, sample_data, replica_key, signal_type, method, parent=None):
        super().__init__(parent)
        self.sample_name = sample_name
        self.sample_data = sample_data
        self.replica_key = replica_key
        self.signal_type = signal_type
        self.method = method
        self.is_bp = method.upper() == "BP"

        self._state = self._determine_state()

        signal_label = signal_type.upper()
        self.setWindowTitle(f"Reparació Cim Irregular — {sample_name} (R{replica_key} {signal_label})")
        self.setMinimumSize(750, 600)
        self.resize(850, 700)
        self.setModal(True)

        self._preview = None
        self._setup_ui()

    def _determine_state(self) -> str:
        """Determina l'estat: 'needs_repair' | 'repaired' | 'dismissed'."""
        replicas = self.sample_data.get("replicas", {})
        rep_data = replicas.get(self.replica_key, {})
        anomalies = rep_data.get("anomalies", [])
        anom_key = ("IRREGULAR_TOP_DIRECT" if self.signal_type == "direct"
                    else "IRREGULAR_TOP_UIB")

        for a in anomalies:
            if isinstance(a, dict) and a.get("code") == anom_key:
                if a.get("dismissed"):
                    return "dismissed"
                if a.get("repaired"):
                    return "repaired"
                return "needs_repair"
        return "needs_repair"

    def _get_replica_data(self):
        """Retorna les dades de la rèplica."""
        return self.sample_data.get("replicas", {}).get(self.replica_key, {})

    def _get_signal_arrays(self):
        """Retorna (t, y) del senyal corresponent.

        Per l'estat 'repaired', retorna el senyal ORIGINAL (backup) com a y,
        ja que y_doc_net ja conté el senyal reparat.
        """
        rep_data = self._get_replica_data()
        t = np.asarray(rep_data.get("t_doc", []))
        if self.signal_type == "direct":
            y_key = "y_doc_net"
            y_orig_key = "y_doc_net_original"
        else:
            y_key = "y_doc_uib_net"
            y_orig_key = "y_doc_uib_net_original"

        if self._state == "repaired":
            # En estat 'repaired', y_doc_net ja és el senyal reparat.
            # Llegim el backup original per mostrar before/after correctament.
            y_original = np.asarray(rep_data.get(y_orig_key, []))
            y_current = np.asarray(rep_data.get(y_key, []))
            if len(y_original) > 0:
                return t, y_original, y_current
            # Fallback: si no hi ha backup, usar el senyal actual
            return t, y_current, y_current
        else:
            y = np.asarray(rep_data.get(y_key, []))
            return t, y, None

    def _compute_repair_preview(self) -> dict:
        """Calcula preview de la reparació sense modificar les dades.

        Extreu el segment al voltant del pic (com fa detect_main_peak)
        per assegurar que find_tangents_and_anchors treballa sobre el
        segment correcte, no el cromatograma complet.
        """
        from hpsec_core import (
            repair_with_parabola, detect_irregular_top,
            calc_top_smoothness, find_tangents_and_anchors
        )
        from hpsec_analyze import calcular_fraccions_temps

        t, y, y_already_repaired = self._get_signal_arrays()
        if len(t) == 0 or len(y) == 0:
            return {}

        # Per l'estat 'repaired', ja tenim original i reparat
        if self._state == "repaired" and y_already_repaired is not None:
            # Segment al voltant del pic per detect/tangent (com a l'anàlisi)
            peak_idx_r = int(np.argmax(y))
            t_peak_r = float(t[peak_idx_r])
            hw_r = 3.0 if self.is_bp else 5.0
            seg_mask_r = (t >= t_peak_r - hw_r) & (t <= t_peak_r + hw_r)
            t_seg_r = t[seg_mask_r]
            y_seg_r = y[seg_mask_r]

            irr_orig = detect_irregular_top(t_seg_r, y_seg_r)
            smooth_orig = calc_top_smoothness(t_seg_r, y_seg_r)
            tangent_info = find_tangents_and_anchors(t_seg_r, y_seg_r)
            try:
                areas_orig = calcular_fraccions_temps(t, y)
            except Exception:
                areas_orig = {}

            y_rep_seg_r = y_already_repaired[seg_mask_r]
            irr_rep = detect_irregular_top(t_seg_r, y_rep_seg_r)
            smooth_rep = calc_top_smoothness(t_seg_r, y_rep_seg_r)
            try:
                areas_rep = calcular_fraccions_temps(t, y_already_repaired)
            except Exception:
                areas_rep = {}

            # Obtenir repair_info del rep_data (guardat quan es va reparar)
            rep_data = self._get_replica_data()
            irr_key = ("irregular_top_direct" if self.signal_type == "direct"
                       else "irregular_top_uib")
            stored_repair_info = rep_data.get(f"{irr_key}_repair_info", {})

            return {
                "t": t, "y_original": y, "y_repaired": y_already_repaired,
                "was_repaired": True,
                "repair_info": stored_repair_info,
                "irr_orig": irr_orig, "irr_rep": irr_rep,
                "smooth_orig": smooth_orig, "smooth_rep": smooth_rep,
                "areas_orig": areas_orig, "areas_rep": areas_rep,
                "tangent_info": tangent_info,
            }

        # Estats 'needs_repair' i 'dismissed': calcular preview de reparació

        # Extreure segment al voltant del pic (com detect_main_peak L1648-1653)
        peak_idx = int(np.argmax(y))
        t_peak = float(t[peak_idx])
        half_window = 3.0 if self.is_bp else 5.0
        seg_mask = (t >= t_peak - half_window) & (t <= t_peak + half_window)
        t_seg = t[seg_mask]
        y_seg = y[seg_mask]

        # Anàlisi original (sobre segment)
        irr_orig = detect_irregular_top(t_seg, y_seg)
        smooth_orig = calc_top_smoothness(t_seg, y_seg)
        tangent_info = find_tangents_and_anchors(t_seg, y_seg)

        try:
            areas_orig = calcular_fraccions_temps(t, y)
        except Exception:
            areas_orig = {}

        # Reparació preview (sobre segment, com fa detect_main_peak)
        y_seg_repaired, repair_info, was_repaired = repair_with_parabola(
            t_seg, y_seg, force=True
        )

        # Mapejar reparació del segment al cromatograma complet
        y_repaired = y.copy()
        if was_repaired:
            y_repaired[seg_mask] = y_seg_repaired

            irr_rep = detect_irregular_top(t_seg, y_seg_repaired)
            smooth_rep = calc_top_smoothness(t_seg, y_seg_repaired)
            try:
                areas_rep = calcular_fraccions_temps(t, y_repaired)
            except Exception:
                areas_rep = {}
        else:
            irr_rep = irr_orig
            smooth_rep = smooth_orig
            areas_rep = areas_orig

        return {
            "t": t, "y_original": y, "y_repaired": y_repaired,
            "was_repaired": was_repaired,
            "repair_info": repair_info,
            "irr_orig": irr_orig, "irr_rep": irr_rep,
            "smooth_orig": smooth_orig, "smooth_rep": smooth_rep,
            "areas_orig": areas_orig, "areas_rep": areas_rep,
            "tangent_info": tangent_info,
        }

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Banner for dismissed state
        if self._state == "dismissed":
            banner = QLabel("Descartat com a fals positiu")
            banner.setAlignment(Qt.AlignCenter)
            banner.setStyleSheet(
                "background: #D5F5E3; color: #1B7A3D; font-weight: bold;"
                " padding: 8px; border-radius: 4px; font-size: 12px;"
            )
            layout.addWidget(banner)

        # Compute preview (for needs_repair and repaired states)
        if self._state != "dismissed":
            self._preview = self._compute_repair_preview()

        # Chromatogram plot
        if HAS_MATPLOTLIB and self._preview:
            self._setup_plot(layout)
        elif not HAS_MATPLOTLIB:
            no_plot = QLabel("Matplotlib no disponible")
            no_plot.setAlignment(Qt.AlignCenter)
            no_plot.setStyleSheet("color: #888; font-style: italic;")
            layout.addWidget(no_plot)

        # Parameters table
        if self._preview:
            self._setup_params_table(layout)

        # Method info
        if self._preview and self._preview.get("was_repaired"):
            self._setup_method_info(layout)

        # Action buttons
        self._setup_buttons(layout)

    def _setup_plot(self, parent_layout):
        """Crea el gràfic matplotlib amb overlay original/reparat."""
        preview = self._preview
        t = preview["t"]
        y_orig = preview["y_original"]
        y_rep = preview["y_repaired"]
        repair_info = preview.get("repair_info", {})

        # Determine zoom range: ±3 min around peak
        peak_idx = int(np.argmax(y_orig))
        t_peak = float(t[peak_idx])
        t_margin = 3.0
        t_lo = t_peak - t_margin
        t_hi = t_peak + t_margin

        fig = Figure(figsize=(8, 3.5), dpi=100)
        fig.set_facecolor("#FAFAFA")
        ax = fig.add_subplot(111)

        # Mask for zoom
        mask = (t >= t_lo) & (t <= t_hi)
        t_z = t[mask]
        y_orig_z = y_orig[mask]
        y_rep_z = y_rep[mask]

        # Original signal
        ax.plot(t_z, y_orig_z, color="#2E86AB", lw=1.2, label="Original", zorder=2)

        # Repaired signal (if different)
        if self._state == "needs_repair" and preview.get("was_repaired"):
            ax.plot(t_z, y_rep_z, color="#E67E22", lw=1.8, ls="--",
                    label="Reparat (preview)", zorder=3)
        elif self._state == "repaired":
            # Show current (repaired) vs original backup
            ax.plot(t_z, y_rep_z, color="#E67E22", lw=1.8, ls="--",
                    label="Reparat (actual)", zorder=3)

        # Repair zone (grey band)
        t_anchor_left = repair_info.get("t_anchor_left")
        t_anchor_right = repair_info.get("t_anchor_right")
        if t_anchor_left is not None and t_anchor_right is not None:
            ax.axvspan(t_anchor_left, t_anchor_right, alpha=0.12,
                       color="#888888", label="Zona reparada", zorder=1)

        # Valleys (red dots)
        irr = preview.get("irr_orig", {})
        valleys = irr.get("valleys", [])
        t_top = irr.get("t_top")
        y_top = irr.get("y_top")
        if valleys and t_top is not None and y_top is not None:
            t_top = np.asarray(t_top)
            y_top = np.asarray(y_top)
            for v_idx in valleys:
                if v_idx < len(t_top):
                    ax.plot(t_top[v_idx], y_top[v_idx], 'rv', ms=7, zorder=5)
            # Label only once
            ax.plot([], [], 'rv', ms=7, label="Valls detectades")

        # Tangent lines (green dotted)
        tangent = preview.get("tangent_info")
        if tangent:
            t_range_left = np.linspace(t_lo, t_peak, 50)
            y_tang_left = tangent["slope_L"] * t_range_left + tangent["intercept_L"]
            t_range_right = np.linspace(t_peak, t_hi, 50)
            y_tang_right = tangent["slope_R"] * t_range_right + tangent["intercept_R"]
            # Clip tangents to reasonable range
            y_min_plot = float(np.min(y_orig_z)) * 0.8
            y_max_plot = float(np.max(y_orig_z)) * 1.3
            mask_l = (y_tang_left >= y_min_plot) & (y_tang_left <= y_max_plot)
            mask_r = (y_tang_right >= y_min_plot) & (y_tang_right <= y_max_plot)
            if np.any(mask_l):
                ax.plot(t_range_left[mask_l], y_tang_left[mask_l],
                        color="#27AE60", ls=":", lw=1, zorder=1)
            if np.any(mask_r):
                ax.plot(t_range_right[mask_r], y_tang_right[mask_r],
                        color="#27AE60", ls=":", lw=1, zorder=1)
            ax.plot([], [], color="#27AE60", ls=":", lw=1, label="Tangents")

        # Fraction boundaries (vertical dashed grey)
        try:
            from hpsec_config import ConfigManager
            config = ConfigManager()
            fracs = config.get("time_fractions", {})
            if self.is_bp:
                frac_def = fracs.get("BP", {})
            else:
                frac_def = fracs.get("COLUMN", {})
            for fname, (f_start, f_end) in frac_def.items():
                if t_lo <= f_start <= t_hi:
                    ax.axvline(f_start, color="#CCCCCC", ls="--", lw=0.7, zorder=0)
                if t_lo <= f_end <= t_hi:
                    ax.axvline(f_end, color="#CCCCCC", ls="--", lw=0.7, zorder=0)
        except Exception:
            pass

        ax.set_xlabel("Temps (min)", fontsize=9)
        ax.set_ylabel("Senyal DOC (ppb)", fontsize=9)
        ax.legend(fontsize=8, loc="upper right", framealpha=0.8)
        ax.tick_params(labelsize=8)
        fig.tight_layout()

        canvas = FigureCanvas(fig)
        canvas.setMinimumHeight(250)
        parent_layout.addWidget(canvas)

    def _setup_params_table(self, parent_layout):
        """Crea la taula de comparació paràmetres original vs reparat."""
        preview = self._preview
        if not preview:
            return

        areas_orig = preview.get("areas_orig", {})
        areas_rep = preview.get("areas_rep", {})
        irr_orig = preview.get("irr_orig", {})
        irr_rep = preview.get("irr_rep", {})
        smooth_orig = preview.get("smooth_orig", {})
        smooth_rep = preview.get("smooth_rep", {})
        y_orig = preview.get("y_original", np.array([]))
        y_rep = preview.get("y_repaired", np.array([]))

        # Build rows: (label, val_orig, val_rep)
        rows = []

        # Main params
        area_o = areas_orig.get("total", 0) or 0
        area_r = areas_rep.get("total", 0) or 0
        rows.append(("Àrea total", area_o, area_r))

        h_o = float(np.max(y_orig)) if len(y_orig) > 0 else 0
        h_r = float(np.max(y_rep)) if len(y_rep) > 0 else 0
        rows.append(("Altura màxima", h_o, h_r))

        sm_o = smooth_orig.get("smoothness", 0) if isinstance(smooth_orig, dict) else 0
        sm_r = smooth_rep.get("smoothness", 0) if isinstance(smooth_rep, dict) else 0
        rows.append(("Smoothness", sm_o, sm_r))

        nv_o = irr_orig.get("n_valleys", 0)
        nv_r = irr_rep.get("n_valleys", 0)
        rows.append(("Valls detectades", nv_o, nv_r))

        md_o = irr_orig.get("max_depth", 0)
        md_r = irr_rep.get("max_depth", 0)
        rows.append(("Prof. max vall", md_o, md_r))

        # Separator + fractions
        fracs = ["BioP", "HS", "BB", "SB", "LMW"]
        for f in fracs:
            fo = areas_orig.get(f, 0) or 0
            fr = areas_rep.get(f, 0) or 0
            rows.append((f, fo, fr))

        # Create table
        table = QTableWidget(len(rows), 4)
        table.setHorizontalHeaderLabels(["Paràmetre", "Original", "Reparat", "Δ"])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionMode(QTableWidget.NoSelection)
        table.setAlternatingRowColors(True)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for col in (1, 2, 3):
            table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeToContents)
        table.setMaximumHeight(min(30 * len(rows) + 32, 360))

        for i, (label, val_o, val_r) in enumerate(rows):
            # Label
            item_l = QTableWidgetItem(label)
            if label in fracs:
                item_l.setFont(QFont("", -1, -1, True))
            table.setItem(i, 0, item_l)

            # Original
            fmt = f"{val_o:.1f}" if isinstance(val_o, float) else str(val_o)
            table.setItem(i, 1, QTableWidgetItem(fmt))

            # Repaired
            fmt_r = f"{val_r:.1f}" if isinstance(val_r, float) else str(val_r)
            table.setItem(i, 2, QTableWidgetItem(fmt_r))

            # Delta
            delta_item = QTableWidgetItem()
            if isinstance(val_o, (int, float)) and isinstance(val_r, (int, float)):
                if val_o != 0:
                    pct = (val_r - val_o) / abs(val_o) * 100
                    delta_item.setText(f"{pct:+.1f}%")
                    if abs(pct) > 15:
                        delta_item.setForeground(QBrush(QColor("#E74C3C")))
                    elif abs(pct) > 5:
                        delta_item.setForeground(QBrush(QColor("#F39C12")))
                    else:
                        delta_item.setForeground(QBrush(QColor("#27AE60")))
                elif val_r != val_o:
                    diff = val_r - val_o
                    delta_item.setText(f"{diff:+.1f}")
                else:
                    delta_item.setText("—")
            table.setItem(i, 3, delta_item)

        parent_layout.addWidget(table)

    def _setup_method_info(self, parent_layout):
        """Mostra informació del mètode de reparació."""
        preview = self._preview
        repair_info = preview.get("repair_info", {})

        from hpsec_core import REPAIR_FACTOR

        t_anchor_l = repair_info.get("t_anchor_left", "?")
        t_anchor_r = repair_info.get("t_anchor_right", "?")
        t_l_str = f"{t_anchor_l:.1f}" if isinstance(t_anchor_l, (int, float)) else str(t_anchor_l)
        t_r_str = f"{t_anchor_r:.1f}" if isinstance(t_anchor_r, (int, float)) else str(t_anchor_r)

        info_frame = QFrame()
        info_frame.setStyleSheet(
            "QFrame { background: #F8F9FA; border: 1px solid #DEE2E6;"
            " border-radius: 4px; padding: 8px; }"
        )
        info_layout = QVBoxLayout(info_frame)
        info_layout.setContentsMargins(8, 6, 8, 6)
        info_layout.setSpacing(2)

        info_layout.addWidget(QLabel(
            f"<b>Mètode:</b> Interpolació parabòlica (3 punts), factor {REPAIR_FACTOR}"
        ))
        info_layout.addWidget(QLabel(
            f"<b>Ancoratges:</b> t={t_l_str} min (esq), t={t_r_str} min (drt)"
        ))
        parent_layout.addWidget(info_frame)

    def _setup_buttons(self, parent_layout):
        """Configura els botons d'acció segons l'estat."""
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        if self._state == "needs_repair":
            # Apply repair
            apply_btn = QPushButton("Aplicar Reparació")
            apply_btn.setStyleSheet(
                "QPushButton { background-color: #2E86AB; color: white;"
                " font-weight: bold; padding: 8px 20px; border-radius: 4px; }"
                "QPushButton:hover { background-color: #236B8E; }"
            )
            apply_btn.clicked.connect(self._on_apply_repair)
            btn_layout.addWidget(apply_btn)

            # Dismiss
            dismiss_btn = QPushButton("Descartar (Fals Positiu)")
            dismiss_btn.setStyleSheet(
                "QPushButton { background-color: #95A5A6; color: white;"
                " padding: 8px 20px; border-radius: 4px; }"
                "QPushButton:hover { background-color: #7F8C8D; }"
            )
            dismiss_btn.clicked.connect(self._on_dismiss)
            btn_layout.addWidget(dismiss_btn)

        elif self._state == "repaired":
            # Undo repair
            undo_btn = QPushButton("Desfer Reparació")
            undo_btn.setStyleSheet(
                "QPushButton { background-color: #E74C3C; color: white;"
                " font-weight: bold; padding: 8px 20px; border-radius: 4px; }"
                "QPushButton:hover { background-color: #C0392B; }"
            )
            undo_btn.clicked.connect(self._on_undo_repair)
            btn_layout.addWidget(undo_btn)

        elif self._state == "dismissed":
            # Reactivate
            react_btn = QPushButton("Reactivar")
            react_btn.setStyleSheet(
                "QPushButton { background-color: #F39C12; color: white;"
                " font-weight: bold; padding: 8px 20px; border-radius: 4px; }"
                "QPushButton:hover { background-color: #E67E22; }"
            )
            react_btn.clicked.connect(self._on_reactivate)
            btn_layout.addWidget(react_btn)

        # Cancel
        cancel_btn = QPushButton("Cancel·lar")
        cancel_btn.setStyleSheet(
            "QPushButton { padding: 8px 20px; border-radius: 4px;"
            " border: 1px solid #CED4DA; }"
            "QPushButton:hover { background-color: #E9ECEF; }"
        )
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        parent_layout.addLayout(btn_layout)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _on_apply_repair(self):
        """Aplica la reparació i emet senyal."""
        try:
            from hpsec_analyze import repair_irregular_top_in_replica
            from hpsec_warnings import get_anomaly_codes

            rep_data = self._get_replica_data()
            if not rep_data:
                QMessageBox.warning(self, "Error", f"No s'ha trobat la rèplica R{self.replica_key}")
                return

            result = repair_irregular_top_in_replica(rep_data, signal=self.signal_type)

            if result.get("repaired"):
                # Marcar sample_data
                self.sample_data["repaired"] = True
                if "repair_history" not in self.sample_data:
                    self.sample_data["repair_history"] = []
                self.sample_data["repair_history"].append({
                    "replica": self.replica_key,
                    "signal": self.signal_type,
                    "repair_info": result.get("repair_info", {}),
                    "original_areas": result.get("original_areas", {}),
                })

                # Check remaining anomalies
                remaining = rep_data.get("anomalies", [])
                remaining_codes = get_anomaly_codes(remaining)
                still_irreparable = bool(remaining_codes & {"NO_PEAK", "TIMEOUT_IN_PEAK"})
                still_irregular = any(
                    isinstance(a, dict)
                    and a.get("code") in ("IRREGULAR_TOP_DIRECT", "IRREGULAR_TOP_UIB")
                    and not a.get("repaired") and not a.get("dismissed")
                    for a in remaining
                )
                if not still_irreparable and not still_irregular:
                    self.sample_data["sample_valid"] = True
                    rec = self.sample_data.get("recommendation", {})
                    if rec.get("doc"):
                        rec["doc"]["valid"] = True
                        rec["doc"]["reason"] = "Cim irregular reparat amb paràbola"

                self.repair_applied.emit(self.sample_name, self.replica_key, self.signal_type)
                self.accept()
            else:
                reason = result.get("reason", "Error desconegut")
                QMessageBox.warning(self, "Reparació no possible", f"No s'ha pogut reparar: {reason}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error durant la reparació:\n{str(e)}")

    def _on_dismiss(self):
        """Marca l'anomalia com a fals positiu."""
        from hpsec_warnings import mark_dismissed

        rep_data = self._get_replica_data()
        anomalies = rep_data.get("anomalies", [])
        anom_key = ("IRREGULAR_TOP_DIRECT" if self.signal_type == "direct"
                    else "IRREGULAR_TOP_UIB")

        if mark_dismissed(anomalies, anom_key):
            # Check if sample becomes valid after dismiss
            from hpsec_warnings import classify_anomalies
            classified = classify_anomalies(anomalies)
            if not classified["blocker"]:
                self.sample_data["sample_valid"] = True

            self.dismissed.emit(self.sample_name, self.replica_key, self.signal_type)
            self.accept()
        else:
            QMessageBox.warning(self, "Error", "No s'ha trobat l'anomalia per descartar")

    def _on_undo_repair(self):
        """Desfà la reparació."""
        try:
            from hpsec_analyze import undo_repair_in_replica

            rep_data = self._get_replica_data()
            result = undo_repair_in_replica(rep_data, signal=self.signal_type)

            if result.get("undone"):
                self.sample_data["repaired"] = False
                self.sample_data["sample_valid"] = False
                self.repair_undone.emit(self.sample_name, self.replica_key, self.signal_type)
                self.accept()
            else:
                reason = result.get("reason", "Error desconegut")
                QMessageBox.warning(self, "Error", f"No s'ha pogut desfer: {reason}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error desfent la reparació:\n{str(e)}")

    def _on_reactivate(self):
        """Reactiva una anomalia dismissed."""
        from hpsec_warnings import unmark_dismissed

        rep_data = self._get_replica_data()
        anomalies = rep_data.get("anomalies", [])
        anom_key = ("IRREGULAR_TOP_DIRECT" if self.signal_type == "direct"
                    else "IRREGULAR_TOP_UIB")

        if unmark_dismissed(anomalies, anom_key):
            self.sample_data["sample_valid"] = False
            self.reactivated.emit(self.sample_name, self.replica_key, self.signal_type)
            self.accept()
        else:
            QMessageBox.warning(self, "Error", "No s'ha trobat l'anomalia per reactivar")
