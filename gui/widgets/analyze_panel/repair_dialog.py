"""
HPSEC Suite - Jagged Peak Repair Dialog (Multi)
================================================

Finestra única per reparar pics irregulars de TOTES les rèpliques × senyals
d'una mostra. Slider interactiu per ajustar el factor de correcció (REPAIR_FACTOR)
amb preview en temps real.

Layout:
  - Header: nom mostra + recompte afectats
  - Factor slider (0.50 – 1.20) amb preview instantani
  - Grid de cards (1 per rèplica×senyal afectat): gràfic + params
  - Botons: Aplicar seleccionats / Aplicar tots / Descartar seleccionats / Tancar
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame,
    QScrollArea, QWidget, QCheckBox, QDoubleSpinBox, QSlider,
    QMessageBox, QGridLayout, QSizePolicy
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont

import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def _get_signal_arrays(rep_data, signal_type, state):
    """Extreu (t, y_original) d'una rèplica. Si repaired, retorna el backup."""
    t = np.asarray(rep_data.get("t_doc", []))
    if signal_type == "direct":
        y_key, y_orig_key = "y_doc_net", "y_doc_net_original"
    else:
        y_key, y_orig_key = "y_doc_uib_net", "y_doc_uib_net_original"

    if state == "repaired":
        y_original = np.asarray(rep_data.get(y_orig_key, []))
        if len(y_original) == 0:
            y_original = np.asarray(rep_data.get(y_key, []))
    else:
        y_original = np.asarray(rep_data.get(y_key, []))

    return t, y_original


def _compute_preview(t, y, factor, is_bp):
    """Calcula preview de reparació amb un factor donat. Retorna dict o {}."""
    from hpsec_core import (
        repair_with_parabola, detect_irregular_top,
        calc_top_smoothness, find_tangents_and_anchors
    )
    from hpsec_analyze import calcular_fraccions_temps

    if len(t) == 0 or len(y) == 0:
        return {}

    peak_idx = int(np.argmax(y))
    t_peak = float(t[peak_idx])
    hw = 3.0 if is_bp else 5.0
    seg_mask = (t >= t_peak - hw) & (t <= t_peak + hw)
    t_seg = t[seg_mask]
    y_seg = y[seg_mask]

    irr_orig = detect_irregular_top(t_seg, y_seg)
    smooth_orig = calc_top_smoothness(t_seg, y_seg)
    tangent_info = find_tangents_and_anchors(t_seg, y_seg)

    try:
        areas_orig = calcular_fraccions_temps(t, y)
    except Exception:
        areas_orig = {}

    y_seg_rep, repair_info, was_repaired = repair_with_parabola(
        t_seg, y_seg, factor=factor, force=True
    )

    y_repaired = y.copy()
    if was_repaired:
        y_repaired[seg_mask] = y_seg_rep
        irr_rep = detect_irregular_top(t_seg, y_seg_rep)
        smooth_rep = calc_top_smoothness(t_seg, y_seg_rep)
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
        "was_repaired": was_repaired, "repair_info": repair_info,
        "irr_orig": irr_orig, "irr_rep": irr_rep,
        "smooth_orig": smooth_orig, "smooth_rep": smooth_rep,
        "areas_orig": areas_orig, "areas_rep": areas_rep,
        "tangent_info": tangent_info, "seg_mask": seg_mask,
    }


# =====================================================================
# RepairCard — un panell per cada rèplica × senyal
# =====================================================================

class _RepairCard(QFrame):
    """Card amb gràfic + params per una rèplica × senyal."""

    def __init__(self, rep_key, signal_type, state, t, y_original, is_bp, factor, parent=None):
        super().__init__(parent)
        self.rep_key = rep_key
        self.signal_type = signal_type
        self.state = state  # needs_repair | repaired | dismissed
        self.t = t
        self.y_original = y_original
        self.is_bp = is_bp
        self._factor = factor

        self.setFrameStyle(QFrame.StyledPanel | QFrame.Plain)
        self.setStyleSheet(
            "QFrame { border: 1px solid #DEE2E6; border-radius: 6px;"
            " background: white; }"
        )

        self._preview = None
        self._line_orig = None
        self._line_rep = None
        self._repair_zone = None
        self._ax = None
        self._canvas = None

        self._setup_ui()
        self._update_preview(factor)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)

        # Header: checkbox + label
        header = QHBoxLayout()
        self.checkbox = QCheckBox()
        self.checkbox.setChecked(self.state == "needs_repair")
        header.addWidget(self.checkbox)

        signal_label = self.signal_type.upper()
        state_icons = {"needs_repair": "", "repaired": " [reparat]", "dismissed": " [descartat]"}
        label = QLabel(f"<b>R{self.rep_key} {signal_label}</b>{state_icons.get(self.state, '')}")
        header.addWidget(label)
        header.addStretch()

        # Params summary (updated on factor change)
        self._params_label = QLabel("")
        self._params_label.setStyleSheet("color: #555; font-size: 11px;")
        header.addWidget(self._params_label)
        layout.addLayout(header)

        # Chart
        if HAS_MATPLOTLIB and len(self.t) > 0:
            fig = Figure(figsize=(5, 2.2), dpi=100)
            fig.set_facecolor("#FAFAFA")
            self._ax = fig.add_subplot(111)
            self._canvas = FigureCanvas(fig)
            self._canvas.setMinimumHeight(160)
            self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            layout.addWidget(self._canvas)

    def _update_preview(self, factor):
        """Recalcula preview amb nou factor i actualitza gràfic."""
        self._factor = factor
        self._preview = _compute_preview(self.t, self.y_original, factor, self.is_bp)

        if not self._preview or not self._ax:
            return

        ax = self._ax
        preview = self._preview

        # Zoom range
        peak_idx = int(np.argmax(self.y_original))
        t_peak = float(self.t[peak_idx])
        margin = 3.0
        t_lo, t_hi = t_peak - margin, t_peak + margin
        mask = (self.t >= t_lo) & (self.t <= t_hi)
        t_z = self.t[mask]
        y_orig_z = self.y_original[mask]
        y_rep_z = preview["y_repaired"][mask]

        ax.clear()

        # Original
        ax.plot(t_z, y_orig_z, color="#2E86AB", lw=1.0, label="Original")

        # Repaired preview
        if preview.get("was_repaired"):
            ax.plot(t_z, y_rep_z, color="#E67E22", lw=1.5, ls="--", label="Reparat")

        # Repair zone
        ri = preview.get("repair_info", {})
        t_al = ri.get("t_anchor_left")
        t_ar = ri.get("t_anchor_right")
        if t_al is not None and t_ar is not None:
            ax.axvspan(t_al, t_ar, alpha=0.10, color="#888")

        # Valleys
        irr = preview.get("irr_orig", {})
        valleys = irr.get("valleys", [])
        t_top = irr.get("t_top")
        y_top = irr.get("y_top")
        if valleys and t_top is not None:
            t_top_a = np.asarray(t_top)
            y_top_a = np.asarray(y_top)
            for vi in valleys:
                if vi < len(t_top_a):
                    ax.plot(t_top_a[vi], y_top_a[vi], 'rv', ms=5, zorder=5)

        # Tangents
        tg = preview.get("tangent_info")
        if tg:
            for side, t_range in [("L", np.linspace(t_lo, t_peak, 30)),
                                   ("R", np.linspace(t_peak, t_hi, 30))]:
                y_tang = tg[f"slope_{side}"] * t_range + tg[f"intercept_{side}"]
                y_min_p = float(np.min(y_orig_z)) * 0.8
                y_max_p = float(np.max(y_orig_z)) * 1.3
                m = (y_tang >= y_min_p) & (y_tang <= y_max_p)
                if np.any(m):
                    ax.plot(t_range[m], y_tang[m], color="#27AE60", ls=":", lw=0.8)

        ax.set_xlabel("min", fontsize=8)
        ax.set_ylabel("ppb", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc="upper right", framealpha=0.7)
        try:
            self._canvas.figure.tight_layout()
        except Exception:
            pass
        self._canvas.draw_idle()

        # Update params summary
        areas_o = preview.get("areas_orig", {})
        areas_r = preview.get("areas_rep", {})
        a_o = areas_o.get("total", 0) or 0
        a_r = areas_r.get("total", 0) or 0
        delta_pct = ((a_r - a_o) / a_o * 100) if a_o else 0
        sm_o = preview.get("smooth_orig", {})
        sm_r = preview.get("smooth_rep", {})
        s_o = sm_o.get("smoothness", 0) if isinstance(sm_o, dict) else 0
        s_r = sm_r.get("smoothness", 0) if isinstance(sm_r, dict) else 0
        nv = preview.get("irr_orig", {}).get("n_valleys", 0)

        delta_color = "#27AE60" if abs(delta_pct) <= 5 else "#F39C12" if abs(delta_pct) <= 15 else "#E74C3C"
        self._params_label.setText(
            f"<span style='color:{delta_color}'>{delta_pct:+.1f}% area</span>"
            f" | smooth {s_o:.0%}→{s_r:.0%} | {nv} valls"
        )

    def update_factor(self, factor):
        """Actualitza preview amb nou factor."""
        if self.state == "dismissed":
            return  # No preview per dismissed
        self._update_preview(factor)


# =====================================================================
# JaggedPeakRepairDialog — finestra principal multi-reparació
# =====================================================================

class JaggedPeakRepairDialog(QDialog):
    """Finestra de reparació multi: totes les rèpliques × senyals d'una mostra."""

    repair_completed = Signal(str)  # sample_name — emès quan qualsevol reparació s'aplica

    def __init__(self, sample_name, sample_data, method, force=False, parent=None):
        super().__init__(parent)
        self.sample_name = sample_name
        self.sample_data = sample_data
        self.method = method
        self.is_bp = method.upper() == "BP"
        self._any_changed = False
        self._force = force

        from hpsec_core import REPAIR_FACTOR
        self._factor = REPAIR_FACTOR
        self._default_factor = REPAIR_FACTOR

        # Trobar tots els targets (auto-detectats o tots si force)
        self._targets = self._find_all_targets()

        n = len(self._targets)
        self.setWindowTitle(f"Reparació Cim Irregular — {sample_name} ({n} senyals)")
        self.setMinimumSize(600, 480)
        # Mida adaptativa: més cards → més alt
        h = min(900, 400 + n * 220)
        self.resize(700, h)
        self.setModal(True)

        self._cards = []
        self._setup_ui()

    def _find_all_targets(self):
        """Busca totes les rèpliques × senyals amb IRREGULAR_TOP.
        Si force=True, inclou TOTES les rèpliques/senyals amb dades."""
        targets = []
        replicas = self.sample_data.get("replicas", {})
        found_auto = set()

        # First pass: auto-detected anomalies
        for rep_key in sorted(replicas.keys()):
            rep_data = replicas[rep_key]
            if not isinstance(rep_data, dict):
                continue
            anomalies = rep_data.get("anomalies", [])
            for signal_type in ("direct", "uib"):
                anom_key = f"IRREGULAR_TOP_{signal_type.upper()}"
                for a in anomalies:
                    if isinstance(a, dict) and a.get("code") == anom_key:
                        state = ("dismissed" if a.get("dismissed")
                                 else "repaired" if a.get("repaired")
                                 else "needs_repair")
                        targets.append((rep_key, signal_type, state))
                        found_auto.add((rep_key, signal_type))
                        break
                    elif isinstance(a, str) and anom_key in a:
                        state = "repaired" if "REPAIRED" in a else "needs_repair"
                        targets.append((rep_key, signal_type, state))
                        found_auto.add((rep_key, signal_type))
                        break

        # Second pass: if force or no auto targets, add all replicas with data
        if self._force or not targets:
            for rep_key in sorted(replicas.keys()):
                rep_data = replicas[rep_key]
                if not isinstance(rep_data, dict):
                    continue
                for signal_type in ("direct", "uib"):
                    if (rep_key, signal_type) in found_auto:
                        continue
                    t, y = _get_signal_arrays(rep_data, signal_type, "needs_repair")
                    if len(t) > 0 and len(y) > 0:
                        targets.append((rep_key, signal_type, "needs_repair"))

        return targets

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # === HEADER ===
        n_pending = sum(1 for _, _, s in self._targets if s == "needs_repair")
        n_repaired = sum(1 for _, _, s in self._targets if s == "repaired")
        n_dismissed = sum(1 for _, _, s in self._targets if s == "dismissed")

        parts = []
        if n_pending:
            parts.append(f"<span style='color:#E74C3C'>{n_pending} pendents</span>")
        if n_repaired:
            parts.append(f"<span style='color:#27AE60'>{n_repaired} reparats</span>")
        if n_dismissed:
            parts.append(f"<span style='color:#95A5A6'>{n_dismissed} descartats</span>")

        header = QLabel(f"<b>{self.sample_name}</b> — {' | '.join(parts)}")
        header.setStyleSheet("font-size: 13px; padding: 4px;")
        layout.addWidget(header)

        # === FACTOR CONTROL ===
        factor_frame = QFrame()
        factor_frame.setStyleSheet(
            "QFrame { background: #F0F4F8; border: 1px solid #D0D7DE;"
            " border-radius: 4px; padding: 6px; }"
        )
        factor_layout = QHBoxLayout(factor_frame)
        factor_layout.setContentsMargins(8, 4, 8, 4)

        factor_layout.addWidget(QLabel("<b>Factor correcció:</b>"))

        self._factor_spin = QDoubleSpinBox()
        self._factor_spin.setRange(0.50, 1.20)
        self._factor_spin.setSingleStep(0.05)
        self._factor_spin.setDecimals(2)
        self._factor_spin.setValue(self._factor)
        self._factor_spin.setFixedWidth(70)
        factor_layout.addWidget(self._factor_spin)

        self._factor_slider = QSlider(Qt.Horizontal)
        self._factor_slider.setRange(50, 120)  # *100
        self._factor_slider.setValue(int(self._factor * 100))
        self._factor_slider.setTickPosition(QSlider.TicksBelow)
        self._factor_slider.setTickInterval(10)
        factor_layout.addWidget(self._factor_slider)

        # Labels min/max
        factor_layout.addWidget(QLabel(
            f"<span style='color:#888; font-size:10px'>"
            f"(default: {self._default_factor})</span>"
        ))

        layout.addWidget(factor_frame)

        # Connect slider ↔ spinbox
        self._factor_slider.valueChanged.connect(self._on_slider_changed)
        self._factor_spin.valueChanged.connect(self._on_spin_changed)

        # === CARDS SCROLL AREA ===
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumHeight(120)
        scroll_widget = QWidget()
        self._cards_layout = QGridLayout(scroll_widget)
        self._cards_layout.setSpacing(8)

        replicas = self.sample_data.get("replicas", {})
        for i, (rep_key, signal_type, state) in enumerate(self._targets):
            rep_data = replicas.get(rep_key, {})
            t, y_original = _get_signal_arrays(rep_data, signal_type, state)

            card = _RepairCard(
                rep_key, signal_type, state, t, y_original,
                self.is_bp, self._factor, parent=self
            )
            # 2 columnes si ≥ 4 targets, sinó 1 columna
            cols = 2 if len(self._targets) >= 4 else 1
            row_idx = i // cols
            col_idx = i % cols
            self._cards_layout.addWidget(card, row_idx, col_idx)
            self._cards.append(card)

        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll, stretch=1)

        # === BUTTONS ===
        btn_layout = QHBoxLayout()

        # Select/deselect all
        select_all_btn = QPushButton("Seleccionar tots")
        select_all_btn.setStyleSheet("padding: 6px 12px;")
        select_all_btn.clicked.connect(self._select_all)
        btn_layout.addWidget(select_all_btn)

        deselect_btn = QPushButton("Cap")
        deselect_btn.setStyleSheet("padding: 6px 12px;")
        deselect_btn.clicked.connect(self._deselect_all)
        btn_layout.addWidget(deselect_btn)

        btn_layout.addStretch()

        # Dismiss selected
        dismiss_btn = QPushButton("Descartar seleccionats")
        dismiss_btn.setStyleSheet(
            "QPushButton { background: #95A5A6; color: white;"
            " padding: 6px 14px; border-radius: 4px; }"
            "QPushButton:hover { background: #7F8C8D; }"
        )
        dismiss_btn.clicked.connect(self._on_dismiss_selected)
        btn_layout.addWidget(dismiss_btn)

        # Undo selected (for repaired)
        if n_repaired > 0:
            undo_btn = QPushButton("Desfer seleccionats")
            undo_btn.setStyleSheet(
                "QPushButton { background: #E74C3C; color: white;"
                " padding: 6px 14px; border-radius: 4px; }"
                "QPushButton:hover { background: #C0392B; }"
            )
            undo_btn.clicked.connect(self._on_undo_selected)
            btn_layout.addWidget(undo_btn)

        # Apply selected
        apply_btn = QPushButton("Aplicar seleccionats")
        apply_btn.setStyleSheet(
            "QPushButton { background: #2E86AB; color: white; font-weight: bold;"
            " padding: 6px 14px; border-radius: 4px; }"
            "QPushButton:hover { background: #236B8E; }"
        )
        apply_btn.clicked.connect(self._on_apply_selected)
        btn_layout.addWidget(apply_btn)

        # Close
        close_btn = QPushButton("Tancar")
        close_btn.setStyleSheet(
            "QPushButton { padding: 6px 14px; border-radius: 4px;"
            " border: 1px solid #CED4DA; }"
            "QPushButton:hover { background: #E9ECEF; }"
        )
        close_btn.clicked.connect(self._on_close)
        btn_layout.addWidget(close_btn)

        layout.addLayout(btn_layout)

    # ------------------------------------------------------------------
    # Factor sync
    # ------------------------------------------------------------------

    def _on_slider_changed(self, value):
        new_factor = value / 100.0
        self._factor_spin.blockSignals(True)
        self._factor_spin.setValue(new_factor)
        self._factor_spin.blockSignals(False)
        self._update_all_cards(new_factor)

    def _on_spin_changed(self, value):
        self._factor_slider.blockSignals(True)
        self._factor_slider.setValue(int(value * 100))
        self._factor_slider.blockSignals(False)
        self._update_all_cards(value)

    def _update_all_cards(self, factor):
        self._factor = factor
        for card in self._cards:
            card.update_factor(factor)

    # ------------------------------------------------------------------
    # Selection helpers
    # ------------------------------------------------------------------

    def _select_all(self):
        for card in self._cards:
            card.checkbox.setChecked(True)

    def _deselect_all(self):
        for card in self._cards:
            card.checkbox.setChecked(False)

    def _get_selected_cards(self):
        return [c for c in self._cards if c.checkbox.isChecked()]

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _on_apply_selected(self):
        """Aplica reparació als cards seleccionats."""
        from hpsec_analyze import repair_irregular_top_in_replica
        from hpsec_warnings import get_anomaly_codes, ANOMALY_CATALOG, classify_anomalies

        selected = self._get_selected_cards()
        applicable = [c for c in selected if c.state in ("needs_repair", "dismissed")]
        if not applicable:
            QMessageBox.information(self, "Info", "Cap senyal seleccionat pendent de reparació.")
            return

        replicas = self.sample_data.get("replicas", {})
        n_ok, n_fail = 0, 0

        for card in applicable:
            rep_data = replicas.get(card.rep_key, {})
            if not rep_data:
                n_fail += 1
                continue

            # Si era dismissed, reactivar primer
            if card.state == "dismissed":
                from hpsec_warnings import unmark_dismissed
                anomalies = rep_data.get("anomalies", [])
                anom_key = f"IRREGULAR_TOP_{card.signal_type.upper()}"
                unmark_dismissed(anomalies, anom_key)

            result = repair_irregular_top_in_replica(
                rep_data, signal=card.signal_type, factor=self._factor
            )

            if result.get("repaired"):
                n_ok += 1
                card.state = "repaired"
                self._any_changed = True

                # Traçabilitat
                self.sample_data["repaired"] = True
                if "repair_history" not in self.sample_data:
                    self.sample_data["repair_history"] = []
                self.sample_data["repair_history"].append({
                    "replica": card.rep_key,
                    "signal": card.signal_type,
                    "factor": self._factor,
                    "repair_info": result.get("repair_info", {}),
                    "original_areas": result.get("original_areas", {}),
                })
            else:
                n_fail += 1
                logger.warning("Reparació fallida R%s %s: %s",
                               card.rep_key, card.signal_type,
                               result.get("reason", "unknown"))

        # Actualitzar sample_valid
        self._update_sample_validity()

        if n_ok > 0:
            self.repair_completed.emit(self.sample_name)

        msg = f"Reparats: {n_ok}"
        if n_fail:
            msg += f" | Fallits: {n_fail}"
        QMessageBox.information(self, "Reparació", msg)

        # Refresh cards
        self._refresh_after_action()

    def _on_dismiss_selected(self):
        """Descarta els cards seleccionats com a fals positiu."""
        from hpsec_warnings import mark_dismissed

        selected = self._get_selected_cards()
        applicable = [c for c in selected if c.state == "needs_repair"]
        if not applicable:
            QMessageBox.information(self, "Info", "Cap senyal pendent per descartar.")
            return

        replicas = self.sample_data.get("replicas", {})
        n_ok = 0

        for card in applicable:
            rep_data = replicas.get(card.rep_key, {})
            anomalies = rep_data.get("anomalies", [])
            anom_key = f"IRREGULAR_TOP_{card.signal_type.upper()}"
            if mark_dismissed(anomalies, anom_key):
                n_ok += 1
                card.state = "dismissed"
                self._any_changed = True

        self._update_sample_validity()

        if n_ok > 0:
            self.repair_completed.emit(self.sample_name)
            QMessageBox.information(self, "Descartat", f"{n_ok} senyals descartats com a fals positiu.")

        self._refresh_after_action()

    def _on_undo_selected(self):
        """Desfà la reparació dels cards seleccionats."""
        from hpsec_analyze import undo_repair_in_replica

        selected = self._get_selected_cards()
        applicable = [c for c in selected if c.state == "repaired"]
        if not applicable:
            QMessageBox.information(self, "Info", "Cap senyal seleccionat amb reparació per desfer.")
            return

        replicas = self.sample_data.get("replicas", {})
        n_ok = 0

        for card in applicable:
            rep_data = replicas.get(card.rep_key, {})
            result = undo_repair_in_replica(rep_data, signal=card.signal_type)
            if result.get("undone"):
                n_ok += 1
                card.state = "needs_repair"
                self._any_changed = True

        if n_ok > 0:
            self.sample_data["repaired"] = False
            self.sample_data["sample_valid"] = False
            self.repair_completed.emit(self.sample_name)
            QMessageBox.information(self, "Desfet", f"{n_ok} reparacions desfetes.")

        self._refresh_after_action()

    def _update_sample_validity(self):
        """Actualitza sample_valid basant-se en les anomalies residuals."""
        from hpsec_warnings import get_anomaly_codes, ANOMALY_CATALOG

        replicas = self.sample_data.get("replicas", {})
        all_ok = True

        for rep_key, rep_data in replicas.items():
            anomalies = rep_data.get("anomalies", [])
            for a in anomalies:
                if not isinstance(a, dict):
                    continue
                code = a.get("code", "")
                if a.get("dismissed") or a.get("repaired"):
                    continue
                entry = ANOMALY_CATALOG.get(code, {})
                if entry.get("invalidates"):
                    all_ok = False
                    break

        self.sample_data["sample_valid"] = all_ok
        if all_ok:
            rec = self.sample_data.get("recommendation", {})
            if rec.get("doc"):
                rec["doc"]["valid"] = True

    def _refresh_after_action(self):
        """Reconstrueix els cards després d'una acció (per reflectir nous estats)."""
        # Actualitzar targets
        self._targets = self._find_all_targets()

        # Actualitzar cards existents (no reconstruir, actualitzar estat + preview)
        replicas = self.sample_data.get("replicas", {})
        for i, card in enumerate(self._cards):
            if i < len(self._targets):
                rep_key, signal_type, state = self._targets[i]
                rep_data = replicas.get(rep_key, {})
                t, y_original = _get_signal_arrays(rep_data, signal_type, state)
                card.rep_key = rep_key
                card.signal_type = signal_type
                card.state = state
                card.t = t
                card.y_original = y_original
                card.checkbox.setChecked(state == "needs_repair")
                card.update_factor(self._factor)

    def _on_close(self):
        """Tanca el diàleg."""
        if self._any_changed:
            self.accept()
        else:
            self.reject()
