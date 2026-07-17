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
    QScrollArea, QWidget, QCheckBox, QDoubleSpinBox,
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


def _compute_preview(t, y, factor, is_bp, anchor_left_t=None, anchor_right_t=None):
    """Calcula preview de reparacio amb un factor donat. Retorna dict o {}."""
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
        t_seg, y_seg, factor=factor, force=True,
        anchor_left_t=anchor_left_t, anchor_right_t=anchor_right_t
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


def make_calibration_replica_entry(d, y_key, signal, is_bp, override=None):
    """Construeix l'entrada de rèplica de l'adaptador de calibració per al diàleg.

    d: dict de rèplica d'analizar_khp_data (t_doc, y_doc, peak_info, area...).
    Injecta:
      - _peak_ctx: context del pic (peak_idx/left_idx/right_idx/baseline/area)
        perquè el preview usi recompute_area_with_repair (idèntic al persistit).
      - Si hi ha override desat (manual_repairs.json): estat 'repaired' amb els
        seus ancoratges (_manual_repair) i backup del senyal perquè "Desfer"
        funcioni dins el diàleg.
    """
    t = np.asarray(d.get("t_doc"), dtype=float)
    y = np.asarray(d.get("y_doc"), dtype=float)
    entry = {
        "t_doc": t,
        y_key: y,
        "is_bp": bool(is_bp),
        "anomalies": [],
    }

    peak_info = d.get("peak_info") or {}
    peak_idx = peak_info.get("peak_idx")
    left_idx = peak_info.get("left_idx",
                             peak_info.get("peak_left_idx", d.get("peak_left_idx")))
    right_idx = peak_info.get("right_idx",
                              peak_info.get("peak_right_idx", d.get("peak_right_idx")))
    if peak_idx is not None and left_idx is not None and right_idx is not None:
        entry["_peak_ctx"] = {
            "peak_idx": int(peak_idx),
            "left_idx": int(left_idx),
            "right_idx": int(right_idx),
            "baseline": peak_info.get("baseline_level", 0) or 0,
            # Àrea de referència PRE-override: així el Δ% del preview coincideix
            # amb el que persistirà apply_manual_repair_to_khp (que parteix de
            # l'àrea sense la reparació manual).
            "area": d.get("area_pre_manual", d.get("area")),
        }

    if override:
        entry["anomalies"] = [{
            "code": f"IRREGULAR_TOP_{signal.upper()}",
            "repaired": True,
            "repair_info": {},
        }]
        entry["_manual_repair"] = override
        entry[f"{y_key}_original"] = y.copy()

    return entry


def sync_repair_cards_to_overrides(dialog, seq_path, name, default_signal):
    """Persisteix als overrides (manual_repairs.json) NOMÉS els cards modificats
    en aquesta sessió del diàleg. Cards no tocats no es toquen (els overrides
    existents es conserven tal qual).

    Returns:
        (changed, repaired_reps): si s'ha escrit res, i les rèpliques desades.
    """
    from hpsec_calibrate import (
        load_manual_repairs, set_manual_repair, remove_manual_repair,
        manual_repair_key,
    )
    existing = load_manual_repairs(seq_path)
    changed = False
    repaired_reps = []
    for card in getattr(dialog, "_cards", []):
        rk = getattr(card, "rep_key", None)
        sig = getattr(card, "signal_type", default_signal)
        if rk is None or not getattr(card, "_session_modified", False):
            continue
        key = manual_repair_key(name, rk, sig)
        if getattr(card, "state", "") == "repaired":
            set_manual_repair(seq_path, name, rk, sig,
                              card._anchor_left_spin.value(),
                              card._anchor_right_spin.value(),
                              getattr(dialog, "_factor", None))
            changed = True
            repaired_reps.append(rk)
        elif key in existing:
            remove_manual_repair(seq_path, name, rk, sig)
            changed = True
    return changed, repaired_reps


# =====================================================================
# RepairCard — un panell per cada rèplica × senyal
# =====================================================================

class _RepairCard(QFrame):
    """Card amb gràfic + params per una rèplica × senyal."""

    def __init__(self, rep_key, signal_type, state, t, y_original, is_bp, factor,
                 saved_anchors=None, peak_ctx=None, parent=None):
        """
        Args:
            saved_anchors: tupla (left_t, right_t) d'un override desat — inicialitza
                els ancoratges manuals abans del primer preview.
            peak_ctx: dict {peak_idx, left_idx, right_idx, baseline, area} del pic
                d'analizar_khp_data — si present, el preview usa
                recompute_area_with_repair (idèntic al que es persisteix).
        """
        super().__init__(parent)
        self.rep_key = rep_key
        self.signal_type = signal_type
        self.state = state  # needs_repair | repaired | dismissed
        self.t = t
        self.y_original = y_original
        self.is_bp = is_bp
        self._factor = factor
        self._peak_ctx = peak_ctx
        self._session_modified = False  # True si l'usuari ha aplicat/desfet/descartat

        self.setFrameStyle(QFrame.StyledPanel | QFrame.Plain)
        self.setStyleSheet(
            "QFrame { border: 1px solid #E8E8E8; border-radius: 4px;"
            " background: white; }"
        )

        self._preview = None
        self._line_orig = None
        self._line_rep = None
        self._repair_zone = None
        self._ax = None
        self._canvas = None

        self._setup_ui()
        # Override desat: aplicar els seus ancoratges abans del primer preview
        if saved_anchors is not None and len(self.t) > 0:
            a_left, a_right = saved_anchors
            if a_left is not None and a_right is not None:
                self._updating_anchors = True
                self._anchor_left_spin.setRange(float(self.t[0]), float(self.t[-1]))
                self._anchor_right_spin.setRange(float(self.t[0]), float(self.t[-1]))
                self._anchor_left_spin.setValue(float(a_left))
                self._anchor_right_spin.setValue(float(a_right))
                self._updating_anchors = False
                self._anchor_left_manual = float(a_left)
                self._anchor_right_manual = float(a_right)
        self._update_preview(factor)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)

        # Header: checkbox + label + state badge
        header = QHBoxLayout()
        self.checkbox = QCheckBox()
        self.checkbox.setChecked(self.state in ("needs_repair", "repaired"))
        header.addWidget(self.checkbox)

        signal_label = self.signal_type.upper()
        label = QLabel(f"<b>R{self.rep_key} {signal_label}</b>")
        header.addWidget(label)

        # State badge
        self._state_badge = QLabel()
        self._update_state_badge()
        header.addWidget(self._state_badge)
        header.addStretch()

        # Params summary (updated on factor change)
        self._params_label = QLabel("")
        self._params_label.setStyleSheet("color: #555; font-size: 11px;")
        header.addWidget(self._params_label)
        layout.addLayout(header)

        # Chart
        if HAS_MATPLOTLIB and len(self.t) > 0:
            fig = Figure(figsize=(5, 1.8), dpi=100)
            fig.set_facecolor("#FAFAFA")
            self._ax = fig.add_subplot(111)
            self._canvas = FigureCanvas(fig)
            self._canvas.setMinimumHeight(130)
            self._canvas.setMaximumHeight(200)
            self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            layout.addWidget(self._canvas)

        # Anchor controls
        anchor_row = QHBoxLayout()
        anchor_row.setSpacing(4)
        anchor_row.addWidget(QLabel(
            "<span style='font-size:10px;color:#666'>Ancoratge:</span>"))

        self._anchor_left_spin = QDoubleSpinBox()
        self._anchor_left_spin.setPrefix("E ")
        self._anchor_left_spin.setSuffix(" min")
        self._anchor_left_spin.setDecimals(2)
        self._anchor_left_spin.setSingleStep(0.05)
        self._anchor_left_spin.setStyleSheet("font-size: 10px;")
        self._anchor_left_spin.setFixedWidth(95)
        self._anchor_left_spin.setToolTip("Ancoratge esquerre (inici reparacio)")
        anchor_row.addWidget(self._anchor_left_spin)

        self._anchor_right_spin = QDoubleSpinBox()
        self._anchor_right_spin.setPrefix("D ")
        self._anchor_right_spin.setSuffix(" min")
        self._anchor_right_spin.setDecimals(2)
        self._anchor_right_spin.setSingleStep(0.05)
        self._anchor_right_spin.setStyleSheet("font-size: 10px;")
        self._anchor_right_spin.setFixedWidth(95)
        self._anchor_right_spin.setToolTip("Ancoratge dret (fi reparacio)")
        anchor_row.addWidget(self._anchor_right_spin)

        self._anchor_auto_btn = QPushButton("Auto")
        self._anchor_auto_btn.setStyleSheet(
            "QPushButton { font-size: 9px; padding: 1px 6px; }")
        self._anchor_auto_btn.setToolTip("Tornar als ancoratges automatics")
        self._anchor_auto_btn.clicked.connect(self._reset_anchors)
        anchor_row.addWidget(self._anchor_auto_btn)

        self._anchor_copy_btn = QPushButton("Copiar a les altres")
        self._anchor_copy_btn.setStyleSheet(
            "QPushButton { font-size: 9px; padding: 1px 6px; color: #2563EB; }")
        self._anchor_copy_btn.setToolTip(
            "Copia aquests ancoratges a totes les altres cards")
        anchor_row.addWidget(self._anchor_copy_btn)
        # Connected from parent dialog (needs access to other cards)

        anchor_row.addStretch()
        layout.addLayout(anchor_row)

        # Initialize anchor values (will be set on first preview)
        self._anchor_left_manual = None
        self._anchor_right_manual = None
        self._anchor_left_spin.valueChanged.connect(self._on_anchor_changed)
        self._anchor_right_spin.valueChanged.connect(self._on_anchor_changed)
        self._updating_anchors = False

    def _on_anchor_changed(self):
        if self._updating_anchors:
            return
        self._anchor_left_manual = self._anchor_left_spin.value()
        self._anchor_right_manual = self._anchor_right_spin.value()
        self._update_preview(self._factor)

    def _reset_anchors(self):
        self._anchor_left_manual = None
        self._anchor_right_manual = None
        self._update_preview(self._factor)

    def _apply_peak_ctx_preview(self, factor):
        """Recalcula el preview amb recompute_area_with_repair (context de pic).

        Sobreescriu y_repaired/repair_info del preview i afegeix ctx_area_orig/
        ctx_area_new perquè el Δ% mostri l'efecte real sobre l'àrea de calibració.
        Si el recàlcul no és possible, deixa el preview estàndard intacte.
        """
        from hpsec_core import recompute_area_with_repair
        pc = self._peak_ctx
        try:
            res = recompute_area_with_repair(
                self.t, self.y_original,
                pc.get("peak_idx"), pc.get("left_idx"), pc.get("right_idx"),
                pc.get("baseline", 0), self.is_bp,
                anchor_left_t=self._anchor_left_manual,
                anchor_right_t=self._anchor_right_manual,
                factor=factor, original_area=pc.get("area"))
        except Exception:
            res = None
        if not res:
            return
        self._preview["y_repaired"] = np.asarray(res["y_repaired"])
        self._preview["was_repaired"] = True
        if res.get("repair_info"):
            self._preview["repair_info"] = res["repair_info"]
        self._preview["ctx_area_orig"] = pc.get("area")
        self._preview["ctx_area_new"] = res["new_area"]

    def _update_state_badge(self):
        """Update the visual state badge (minimal style)."""
        def _ss(border, bg):
            return (f"QFrame {{ border: 1px solid {border}; border-radius: 4px;"
                    f" background: {bg}; }}")
        def _badge(c, text):
            return (f"<span style='color:{c}; font-size:10px; font-weight:bold'>"
                    f"{text}</span>")
        if self.state == "repaired":
            self._state_badge.setText(_badge('#5B9F5B', '\u2713 reparat'))
            self.setStyleSheet(_ss('#B5D8B5', '#FCFEFC'))
        elif self.state == "dismissed":
            self._state_badge.setText(_badge('#999', 'descartat'))
            self.setStyleSheet(_ss('#DDD', '#FAFAFA'))
        elif self.state == "needs_repair":
            self._state_badge.setText(_badge('#EF4444', 'pendent'))
            self.setStyleSheet(_ss('#E0C4C0', 'white'))
        else:
            self._state_badge.setText("")
            self.setStyleSheet(_ss('#DEE2E6', 'white'))

    def _update_preview(self, factor):
        """Recalcula preview amb nou factor i actualitza grafic."""
        self._factor = factor
        self._preview = _compute_preview(
            self.t, self.y_original, factor, self.is_bp,
            anchor_left_t=self._anchor_left_manual,
            anchor_right_t=self._anchor_right_manual)

        # Context de pic (calibració): el preview usa recompute_area_with_repair
        # (finestra de pic + baseline) perquè el Δ% mostrat sigui exactament el
        # que persistirà apply_manual_repair_to_khp. Fallback silenciós al
        # comportament estàndard si el context no és aplicable.
        if self._peak_ctx and self._preview:
            self._apply_peak_ctx_preview(factor)

        if not self._preview or not self._ax:
            return

        ax = self._ax
        preview = self._preview

        # Update anchor spinboxes with current values
        ri = preview.get("repair_info", {})
        t_al = ri.get("t_anchor_left")
        t_ar = ri.get("t_anchor_right")
        if t_al is not None and t_ar is not None and hasattr(self, '_anchor_left_spin'):
            self._updating_anchors = True
            self._anchor_left_spin.setRange(float(self.t[0]), float(self.t[-1]))
            self._anchor_right_spin.setRange(float(self.t[0]), float(self.t[-1]))
            if self._anchor_left_manual is None:
                self._anchor_left_spin.setValue(t_al)
            if self._anchor_right_manual is None:
                self._anchor_right_spin.setValue(t_ar)
            self._updating_anchors = False

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
        ax.plot(t_z, y_orig_z, color="#2563EB", lw=1.0, label="Original")

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
        if preview.get("ctx_area_new") is not None:
            # Àrees del context de pic (coincideixen amb el que es persistirà)
            a_o = preview.get("ctx_area_orig") or 0
            a_r = preview.get("ctx_area_new") or 0
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

    repair_completed = Signal(str)  # sample_name
    navigate_requested = Signal(int)  # direction: -1 prev, +1 next

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

        # Claus (rep_key, signal) modificades en aquesta sessió — font de veritat
        # per _session_modified dels cards (sobreviu la reassignació de cards a
        # _refresh_after_action).
        self._modified_keys = set()

        # Si algun override carregat (calibració) porta factor, usar-lo d'inici
        for _rd in (sample_data.get("replicas") or {}).values():
            if not isinstance(_rd, dict):
                continue
            _mr = _rd.get("_manual_repair")
            if isinstance(_mr, dict) and _mr.get("factor") is not None:
                self._factor = float(_mr["factor"])
                break

        # Trobar tots els targets (auto-detectats o tots si force)
        self._targets = self._find_all_targets()

        n = len(self._targets)
        self.setWindowTitle(f"Reparació — {sample_name} ({n} senyals)")
        # Mida adaptativa: 2 cards = compacte, 4 = mes gran
        if n <= 2:
            self.setMinimumSize(650, 350)
            self.resize(750, 450)
        else:
            self.setMinimumSize(700, 450)
            self.resize(800, 550)
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
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        btn_s = ("QPushButton { border: 1px solid {c}; border-radius: 3px;"
                 " padding: 4px 10px; font-size: 11px; color: {c}; }}"
                 "QPushButton:hover {{ background: {bg}; }}")
        nav_s = ("QPushButton { border: 1px solid #CED4DA; border-radius: 3px;"
                 " padding: 4px 8px; font-size: 11px; }"
                 "QPushButton:hover { background: #E9ECEF; }")

        # === TOP BAR: nav + actions + factor + title ===
        top_row = QHBoxLayout()
        top_row.setSpacing(4)

        prev_btn = QPushButton("\u25c0")
        prev_btn.setStyleSheet(nav_s)
        prev_btn.setFixedWidth(28)
        prev_btn.setToolTip("Mostra anterior")
        prev_btn.clicked.connect(lambda: self.navigate_requested.emit(-1))
        top_row.addWidget(prev_btn)

        apply_btn = QPushButton("Aplicar")
        apply_btn.setStyleSheet(
            "QPushButton { border: 1px solid #4A90A4; border-radius: 3px;"
            " padding: 4px 10px; font-size: 11px; color: white;"
            " background: #4A90A4; font-weight: bold; }"
            "QPushButton:hover { background: #3A7A8E; }")
        apply_btn.setToolTip("Aplicar reparacio als seleccionats")
        apply_btn.clicked.connect(self._on_apply_selected)
        top_row.addWidget(apply_btn)

        undo_btn = QPushButton("Desfer")
        undo_btn.setStyleSheet(nav_s)
        undo_btn.setToolTip("Desfer reparacio dels seleccionats")
        undo_btn.clicked.connect(self._on_undo_selected)
        top_row.addWidget(undo_btn)

        dismiss_btn = QPushButton("Descartar")
        dismiss_btn.setStyleSheet(nav_s)
        dismiss_btn.setToolTip("Descartar seleccionats (no reparar)")
        dismiss_btn.clicked.connect(self._on_dismiss_selected)
        top_row.addWidget(dismiss_btn)

        top_row.addWidget(QLabel(
            "<span style='color:#999'>|</span>"))

        top_row.addWidget(QLabel(
            "<b style='font-size:11px'>Factor:</b>"))
        self._factor_spin = QDoubleSpinBox()
        self._factor_spin.setRange(0.50, 1.20)
        self._factor_spin.setSingleStep(0.1)
        self._factor_spin.setDecimals(2)
        self._factor_spin.setValue(self._factor)
        self._factor_spin.setFixedWidth(65)
        self._factor_spin.setStyleSheet("font-size: 11px;")
        self._factor_spin.valueChanged.connect(self._on_spin_changed)
        top_row.addWidget(self._factor_spin)

        self._header_label = QLabel(f"<b>{self.sample_name}</b>")
        self._header_label.setAlignment(Qt.AlignCenter)
        self._header_label.setStyleSheet("font-size: 12px;")
        top_row.addWidget(self._header_label, 1)

        close_btn = QPushButton("Tancar")
        close_btn.setStyleSheet(nav_s)
        close_btn.clicked.connect(self._on_close)
        top_row.addWidget(close_btn)

        next_btn = QPushButton("\u25b6")
        next_btn.setStyleSheet(nav_s)
        next_btn.setFixedWidth(28)
        next_btn.setToolTip("Mostra seguent")
        next_btn.clicked.connect(lambda: self.navigate_requested.emit(1))
        top_row.addWidget(next_btn)

        layout.addLayout(top_row)

        # === CARDS GRID (always 4 if DUAL, 2 if DIRECT) ===
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumHeight(200)
        scroll_widget = QWidget()
        self._cards_layout = QGridLayout(scroll_widget)
        self._cards_layout.setSpacing(6)

        replicas = self.sample_data.get("replicas", {})
        detected_set = {(rk, st) for rk, st, _ in self._targets}
        n_targets = len(self._targets)

        for i, (rep_key, signal_type, state) in enumerate(self._targets):
            rep_data = replicas.get(rep_key, {})
            t, y_original = _get_signal_arrays(rep_data, signal_type, state)

            # Context de calibració (opcional): override desat + context de pic
            saved_anchors = None
            peak_ctx = None
            if isinstance(rep_data, dict):
                _mr = rep_data.get("_manual_repair")
                if (isinstance(_mr, dict)
                        and _mr.get("anchor_left_t") is not None
                        and _mr.get("anchor_right_t") is not None):
                    saved_anchors = (_mr["anchor_left_t"], _mr["anchor_right_t"])
                peak_ctx = rep_data.get("_peak_ctx")

            # Pre-check if anomaly detected
            is_detected = state in ("needs_repair", "repaired")
            card = _RepairCard(
                rep_key, signal_type, state, t, y_original,
                self.is_bp, self._factor,
                saved_anchors=saved_anchors, peak_ctx=peak_ctx, parent=self
            )
            card.checkbox.setChecked(is_detected)

            # Grid: 2 columns always
            cols = 2
            row_idx = i // cols
            col_idx = i % cols
            self._cards_layout.addWidget(card, row_idx, col_idx)
            self._cards.append(card)

        # Connectar "Copiar a les altres" de cada card
        for card in self._cards:
            card._anchor_copy_btn.clicked.connect(
                lambda checked=False, c=card: self._copy_anchors_from(c))

        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll, stretch=1)

    # ------------------------------------------------------------------
    # Factor sync
    # ------------------------------------------------------------------

    def _copy_anchors_from(self, source_card):
        """Copia anchors d'una card a totes les altres."""
        left = source_card._anchor_left_spin.value()
        right = source_card._anchor_right_spin.value()
        n_copied = 0
        for card in self._cards:
            if card is source_card:
                continue
            card._updating_anchors = True
            card._anchor_left_spin.setValue(left)
            card._anchor_right_spin.setValue(right)
            card._updating_anchors = False
            card._anchor_left_manual = left
            card._anchor_right_manual = right
            card._update_preview(card._factor)
            n_copied += 1

    def _on_spin_changed(self, value):
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

            # Passar anchors manuals si l'usuari els ha modificat
            repair_kwargs = {
                "signal": card.signal_type,
                "factor": self._factor,
            }
            if hasattr(card, '_anchor_left_manual') and card._anchor_left_manual is not None:
                repair_kwargs["anchor_left_t"] = card._anchor_left_manual
            if hasattr(card, '_anchor_right_manual') and card._anchor_right_manual is not None:
                repair_kwargs["anchor_right_t"] = card._anchor_right_manual

            result = repair_irregular_top_in_replica(rep_data, **repair_kwargs)

            if result.get("repaired"):
                n_ok += 1
                card.state = "repaired"
                card._session_modified = True
                self._modified_keys.add((card.rep_key, card.signal_type))
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
                card._session_modified = True
                self._modified_keys.add((card.rep_key, card.signal_type))
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
                card._session_modified = True
                self._modified_keys.add((card.rep_key, card.signal_type))
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
                # Reassignar el context per-rèplica: l'ordre dels targets pot
                # canviar després d'una acció i els cards es remapegen sencers
                card._peak_ctx = (rep_data.get("_peak_ctx")
                                  if isinstance(rep_data, dict) else None)
                card._session_modified = (rep_key, signal_type) in self._modified_keys
                card.checkbox.setChecked(state in ("needs_repair", "repaired"))
                card._update_state_badge()
                card.update_factor(self._factor)

    def _on_close(self):
        """Tanca el diàleg."""
        if self._any_changed:
            self.accept()
        else:
            self.reject()
