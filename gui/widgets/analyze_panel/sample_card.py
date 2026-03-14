"""
HPSEC Suite - SampleCard widget for AnalyzePanel
=================================================

Compact/expandable card per mostra. Substitueix la taula QTableWidget.
- Mode compacte: una fila amb icona estat + nom + ppm + R2 + HCI + hints
- Mode expandit: cromatograma + selectors replica + reparacio + fraccions + anomalies
- Nomes UN card expandit a la vegada
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QComboBox, QSizePolicy, QGroupBox, QCheckBox,
    QDoubleSpinBox, QSlider, QGridLayout
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QColor, QCursor

import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from gui.widgets.styles import COLOR_SUCCESS, COLOR_WARNING, COLOR_ERROR
from ._helpers import (
    classify_sample_status, resolve_doc_replica, find_repair_targets,
    draw_timeout_zones_on_ax,
)
from ._constants import FRACTION_NAMES, FRACTION_RANGES
from hpsec_warnings import (
    get_anomaly_codes, classify_anomalies, ANOMALY_CATALOG,
)


# Fraction colors (consistent palette)
FRACTION_COLORS = {
    "BioP": "#3498DB",
    "HS":   "#E74C3C",
    "BB":   "#F39C12",
    "SB":   "#2ECC71",
    "LMW":  "#9B59B6",
}
FRACTION_ORDER = ["BioP", "HS", "BB", "SB", "LMW"]

_CARD_STYLE = """
QFrame#sampleCard {
    background: white;
    border: 1px solid #DEE2E6;
    border-radius: 6px;
    margin: 2px 0;
}
QFrame#sampleCard:hover {
    border-color: #ADB5BD;
}
"""
_CARD_EXPANDED_STYLE = """
QFrame#sampleCardExpanded {
    background: white;
    border: 2px solid #2E86AB;
    border-radius: 6px;
    margin: 2px 0;
}
"""
_CARD_BLANK_STYLE = """
QFrame#sampleCard {
    background: #F4F6F6;
    border: 1px solid #DEE2E6;
    border-radius: 6px;
    margin: 2px 0;
}
"""

_ACTION_BTN_TEMPLATE = (
    "QPushButton {{ border: 1px solid {border}; border-radius: 3px; "
    "font-size: 10px; font-weight: bold; color: {fg}; "
    "background: {bg}; min-width: 22px; max-height: 20px; padding: 0 2px; }}"
    "QPushButton:hover {{ background: {hover}; }}"
)


class SampleCard(QFrame):
    """Card for a single sample: compact (one-line) or expanded (inline detail)."""

    expand_requested = Signal(str)   # sample_name
    data_changed = Signal(str)       # sample_name (after repair/compose/replica change)

    def __init__(self, sample_name, sample_data, samples_grouped,
                 main_window, is_blank=False, is_control=False, parent=None):
        super().__init__(parent)
        self.sample_name = sample_name
        self.sample_data = sample_data
        self.samples_grouped = samples_grouped
        self.main_window = main_window
        self.is_blank = is_blank
        self.is_control = is_control
        self._expanded = False
        self._chromatogram_rendered = False
        self._canvas = None
        self._figure = None
        self._expanded_widget = None

        self.setObjectName("sampleCard")
        if is_blank or is_control:
            self.setStyleSheet(_CARD_BLANK_STYLE)
        else:
            self.setStyleSheet(_CARD_STYLE)

        self.setCursor(QCursor(Qt.PointingHandCursor))

        self._main_layout = QVBoxLayout(self)
        self._main_layout.setContentsMargins(8, 6, 8, 6)
        self._main_layout.setSpacing(0)

        self._build_compact_row()

    # ------------------------------------------------------------------
    # Compact row
    # ------------------------------------------------------------------

    def _build_compact_row(self):
        """Build the single-line compact view."""
        self._compact_widget = QWidget()
        compact_layout = QHBoxLayout(self._compact_widget)
        compact_layout.setContentsMargins(0, 0, 0, 0)
        compact_layout.setSpacing(8)

        # Status icon
        self._status_icon = QLabel()
        self._status_icon.setFixedWidth(20)
        self._status_icon.setAlignment(Qt.AlignCenter)
        compact_layout.addWidget(self._status_icon)

        # Sample name
        self._name_label = QLabel(f"<b>{self.sample_name}</b>")
        self._name_label.setStyleSheet("font-size: 12px;")
        compact_layout.addWidget(self._name_label)

        compact_layout.addStretch()

        if not self.is_control:
            # ppm
            self._ppm_label = QLabel()
            self._ppm_label.setStyleSheet("font-size: 11px; color: #333;")
            compact_layout.addWidget(self._ppm_label)

            # ppm_U
            self._ppm_u_label = QLabel()
            self._ppm_u_label.setStyleSheet("font-size: 11px; color: #666;")
            compact_layout.addWidget(self._ppm_u_label)

            # R2
            self._r2_label = QLabel()
            self._r2_label.setStyleSheet("font-size: 11px;")
            compact_layout.addWidget(self._r2_label)

            # HCI
            self._hci_label = QLabel()
            self._hci_label.setStyleSheet("font-size: 11px;")
            compact_layout.addWidget(self._hci_label)

            # Action hints (R / C badges)
            self._action_hints = QLabel()
            self._action_hints.setStyleSheet("font-size: 10px;")
            compact_layout.addWidget(self._action_hints)
        else:
            self._ppm_label = None
            self._ppm_u_label = None
            self._r2_label = None
            self._hci_label = None
            self._action_hints = None

        # Expand button
        self._expand_btn = QPushButton("\u25bc")
        self._expand_btn.setFixedSize(24, 24)
        self._expand_btn.setStyleSheet(
            "QPushButton { border: none; font-size: 12px; color: #888; }"
            "QPushButton:hover { color: #2E86AB; }"
        )
        self._expand_btn.clicked.connect(self._on_expand_clicked)
        compact_layout.addWidget(self._expand_btn)

        self._main_layout.addWidget(self._compact_widget)
        try:
            self._update_compact_data()
        except Exception as e:
            logger.error(f"Error building card {self.sample_name}: {e}")
            import traceback; traceback.print_exc()

    def _update_compact_data(self):
        """Refresh all compact-row labels from sample_data."""
        replicas = self.sample_data.get("replicas", {})
        comparison = self.sample_data.get("comparison", {})
        quantification = self.sample_data.get("quantification", {})
        _, doc_rep = resolve_doc_replica(self.sample_data)

        selected = self.sample_data.get("selected", {})
        dad_sel = selected.get("dad", "1")
        dad_rep = replicas.get(dad_sel, {})

        # Status classification
        (status_color, status_text, s_tooltip,
         repair_color, repair_text, r_tooltip) = classify_sample_status(
            doc_rep, dad_rep, comparison, sample_data=self.sample_data)

        self._status_icon.setText(status_text)
        self._status_icon.setStyleSheet(
            f"font-size: 14px; color: {status_color};")
        self._status_icon.setToolTip(s_tooltip)

        if self.is_control:
            return

        # ppm
        ppm_direct = (quantification.get("concentration_ppm_direct")
                      or quantification.get("concentration_ppm"))
        is_estimated = doc_rep.get("direct_estimated_from_uib", False)
        if ppm_direct:
            ppm_txt = f"ppm: {ppm_direct:.2f}" + ("*" if is_estimated else "")
        else:
            ppm_txt = "ppm: -"
        self._ppm_label.setText(ppm_txt)

        # Tooltip for ppm
        areas = doc_rep.get("areas") or {}
        doc_areas = areas.get("DOC") or {}
        area_direct = doc_areas.get("total", 0)
        snr_info = doc_rep.get("snr_info") or {}
        snr_direct = snr_info.get("snr_direct", 0)
        ppm_tip = []
        if is_estimated:
            factor = doc_rep.get("direct_estimation_factor", 0.70)
            ppm_tip.append(f"* Estimat des d'UIB (factor {factor:.2f})")
        if area_direct:
            ppm_tip.append(f"A_DOC: {area_direct:.0f}")
        if snr_direct:
            ppm_tip.append(f"SNR: {snr_direct:.0f}")
        self._ppm_label.setToolTip("\n".join(ppm_tip) if ppm_tip else "")

        # ppm_U
        ppm_uib = quantification.get("concentration_ppm_uib")
        self._ppm_u_label.setText(f"UIB: {ppm_uib:.2f}" if ppm_uib else "")
        areas_uib = doc_rep.get("areas_uib") or {}
        area_uib = areas_uib.get("total", 0)
        snr_uib = snr_info.get("snr_uib", 0)
        ppm_u_tip = []
        if area_uib:
            ppm_u_tip.append(f"A_UIB: {area_uib:.0f}")
        if snr_uib:
            ppm_u_tip.append(f"SNR_UIB: {snr_uib:.0f}")
        self._ppm_u_label.setToolTip(" \u00b7 ".join(ppm_u_tip) if ppm_u_tip else "")

        # R2
        pairwise = self.sample_data.get("pairwise_comparisons", {})
        if pairwise and len(pairwise) > 1:
            r2_doc = min(
                (c.get("doc", {}).get("pearson", 0) for c in pairwise.values()),
                default=0)
            dad_comp = min(
                (c.get("dad", {}) for c in pairwise.values()),
                key=lambda d: d.get("pearson_min", 0), default={})
        else:
            r2_doc = comparison.get("doc", {}).get("pearson", 0) if comparison else 0
            dad_comp = comparison.get("dad", {}) if comparison else {}
        r2_dad_min = dad_comp.get("pearson_min", 0)
        r2_vals = [v for v in (r2_doc, r2_dad_min) if v > 0]
        r2_min = min(r2_vals) if r2_vals else 0

        if r2_min > 0:
            r2_color = (COLOR_SUCCESS if r2_min >= 0.99
                        else COLOR_WARNING if r2_min >= 0.95
                        else COLOR_ERROR)
            self._r2_label.setText(f"R\u00b2: {r2_min:.3f}")
            self._r2_label.setStyleSheet(f"font-size: 11px; color: {r2_color};")
        else:
            self._r2_label.setText("")

        r2_tip = []
        if r2_doc > 0:
            r2_tip.append(f"DOC: {r2_doc:.4f}")
        if r2_dad_min > 0:
            r2_tip.append(f"DAD: {r2_dad_min:.4f}")
        pearson_per_wl = dad_comp.get("pearson_per_wavelength", {})
        if pearson_per_wl:
            for wl, val in sorted(pearson_per_wl.items()):
                warn = " !" if val < 0.990 else ""
                r2_tip.append(f"  A{wl}: {val:.4f}{warn}")
        self._r2_label.setToolTip("\n".join(r2_tip) if r2_tip else "")

        # HCI
        hci_val = quantification.get("hci")
        if hci_val is not None:
            hci_char = quantification.get("hci_character", "")
            abbrev = "HA" if "HA" in hci_char else "FA" if "FA" in hci_char else "Mix"
            hci_color = ("#E74C3C" if hci_val > 60
                         else "#3498DB" if hci_val < 40
                         else "#27AE60")
            self._hci_label.setText(f"HCI: {hci_val:.0f} {abbrev}")
            self._hci_label.setStyleSheet(f"font-size: 11px; color: {hci_color};")
            self._hci_label.setToolTip(
                f"Humic Character Index: {hci_val:.1f} ({hci_char})\n"
                f"Model PCA+LDA v2.0")
        else:
            self._hci_label.setText("")
            self._hci_label.setToolTip("")

        # Action hints (R/C)
        hints = []
        has_repair = bool(find_repair_targets(self.sample_name, self.samples_grouped))
        is_repaired = self.sample_data.get("repaired", False)
        tc = self.sample_data.get("timeout_composability", {})
        has_composition = tc.get("composable", False)

        if has_repair:
            if is_repaired:
                hints.append("<span style='color:#27AE60; font-weight:bold;'>R\u2713</span>")
            else:
                hints.append("<span style='color:#E67E22; font-weight:bold;'>R</span>")
        if has_composition:
            sel_key = (self.sample_data.get("selected", {}) or {}).get("doc", "1")
            sel_rep = (self.sample_data.get("replicas", {}) or {}).get(sel_key, {})
            if sel_rep.get("timeout_composition"):
                hints.append("<span style='color:#27AE60; font-weight:bold;'>C\u2713</span>")
            else:
                hints.append("<span style='color:#3498DB; font-weight:bold;'>C</span>")

        self._action_hints.setText(" ".join(hints) if hints else "")
        self._action_hints.setToolTip(r_tooltip)

    # ------------------------------------------------------------------
    # Expand / Collapse
    # ------------------------------------------------------------------

    def mouseDoubleClickEvent(self, event):
        """Double-click on compact area opens full SampleDetailDialog."""
        if not self._expanded:
            self._open_detail_dialog()
        super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event):
        """Single click toggles expand (unless on a widget)."""
        # Let child widgets handle their own clicks
        child = self.childAt(event.pos())
        if child and isinstance(child, (QPushButton, QComboBox, QCheckBox,
                                        QDoubleSpinBox, QSlider)):
            super().mousePressEvent(event)
            return
        if event.button() == Qt.LeftButton:
            self._on_expand_clicked()
        super().mousePressEvent(event)

    def _on_expand_clicked(self):
        """Toggle expand state."""
        if self._expanded:
            self.collapse()
        else:
            self.expand_requested.emit(self.sample_name)

    def expand(self):
        """Show expanded content."""
        if self._expanded:
            return
        self._expanded = True
        self._expand_btn.setText("\u25b2")  # up arrow
        if self.is_blank or self.is_control:
            self.setObjectName("sampleCardExpanded")
            self.setStyleSheet(_CARD_EXPANDED_STYLE)
        else:
            self.setObjectName("sampleCardExpanded")
            self.setStyleSheet(_CARD_EXPANDED_STYLE)

        if self._expanded_widget is None:
            self._build_expanded_content()
        self._expanded_widget.setVisible(True)

    def collapse(self):
        """Hide expanded content."""
        if not self._expanded:
            return
        self._expanded = False
        self._expand_btn.setText("\u25bc")  # down arrow
        self.setObjectName("sampleCard")
        if self.is_blank or self.is_control:
            self.setStyleSheet(_CARD_BLANK_STYLE)
        else:
            self.setStyleSheet(_CARD_STYLE)
        if self._expanded_widget:
            self._expanded_widget.setVisible(False)

    @property
    def is_expanded(self):
        return self._expanded

    # ------------------------------------------------------------------
    # Expanded content
    # ------------------------------------------------------------------

    def _build_expanded_content(self):
        """Build the full inline detail widget (lazy, only on first expand)."""
        self._expanded_widget = QWidget()
        exp_layout = QVBoxLayout(self._expanded_widget)
        exp_layout.setContentsMargins(0, 8, 0, 4)
        exp_layout.setSpacing(6)

        # --- 1. Navigation row ---
        nav_row = QHBoxLayout()
        nav_row.setSpacing(8)
        self._prev_btn = QPushButton("\u25c0 Anterior")
        self._prev_btn.setStyleSheet(
            "QPushButton { border: 1px solid #CED4DA; border-radius: 3px;"
            " padding: 3px 10px; font-size: 10px; }"
            "QPushButton:hover { background: #E9ECEF; }")
        self._prev_btn.clicked.connect(lambda: self._navigate(-1))
        nav_row.addWidget(self._prev_btn)

        self._nav_label = QLabel()
        self._nav_label.setAlignment(Qt.AlignCenter)
        self._nav_label.setStyleSheet("font-size: 11px; color: #555;")
        nav_row.addWidget(self._nav_label, 1)

        self._next_btn = QPushButton("Seguent \u25b6")
        self._next_btn.setStyleSheet(
            "QPushButton { border: 1px solid #CED4DA; border-radius: 3px;"
            " padding: 3px 10px; font-size: 10px; }"
            "QPushButton:hover { background: #E9ECEF; }")
        self._next_btn.clicked.connect(lambda: self._navigate(1))
        nav_row.addWidget(self._next_btn)

        self._detail_btn = QPushButton("Detall complet...")
        self._detail_btn.setStyleSheet(
            "QPushButton { border: 1px solid #2E86AB; border-radius: 3px;"
            " padding: 3px 10px; font-size: 10px; color: #2E86AB; }"
            "QPushButton:hover { background: #E3F2FD; }")
        self._detail_btn.clicked.connect(self._open_detail_dialog)
        nav_row.addWidget(self._detail_btn)

        exp_layout.addLayout(nav_row)

        # --- 2. Chromatogram ---
        if HAS_MATPLOTLIB and not self.is_control:
            self._figure = Figure(figsize=(7, 2.5), dpi=100)
            self._figure.set_facecolor("#FAFAFA")
            self._canvas = FigureCanvas(self._figure)
            self._canvas.setMinimumHeight(200)
            self._canvas.setMaximumHeight(280)
            self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            exp_layout.addWidget(self._canvas)

        # --- 3. Controls row: replicas + key metrics ---
        if not self.is_control:
            controls_row = QHBoxLayout()
            controls_row.setSpacing(10)

            controls_row.addWidget(QLabel("<b>DOC:</b>"))
            self._doc_combo = QComboBox()
            self._doc_combo.setMinimumWidth(90)
            self._build_doc_combo()
            self._doc_combo.currentIndexChanged.connect(self._on_doc_changed)
            controls_row.addWidget(self._doc_combo)

            controls_row.addWidget(QLabel("<b>DAD:</b>"))
            self._dad_combo = QComboBox()
            self._dad_combo.setMinimumWidth(90)
            self._build_dad_combo()
            self._dad_combo.currentIndexChanged.connect(self._on_dad_changed)
            controls_row.addWidget(self._dad_combo)

            controls_row.addStretch()

            # Key metrics
            self._metrics_label = QLabel()
            self._metrics_label.setStyleSheet("font-size: 11px; color: #444;")
            controls_row.addWidget(self._metrics_label)

            exp_layout.addLayout(controls_row)

        # --- 4. Repair section ---
        if not self.is_control:
            self._repair_group = QGroupBox("Reparar pic irregular")
            self._repair_group.setStyleSheet(
                "QGroupBox { font-size: 11px; font-weight: bold; color: #555;"
                " border: 1px solid #E0E0E0; border-radius: 4px;"
                " margin-top: 6px; padding-top: 14px; }"
                "QGroupBox::title { subcontrol-position: top left; padding: 0 6px; }")
            repair_layout = QVBoxLayout(self._repair_group)
            repair_layout.setSpacing(4)

            has_repair = bool(find_repair_targets(self.sample_name, self.samples_grouped))
            is_repaired = self.sample_data.get("repaired", False)

            if has_repair:
                if is_repaired:
                    repair_status = QLabel(
                        "<span style='color:#27AE60'>\u2713 Reparacio aplicada</span>")
                else:
                    repair_status = QLabel(
                        "<span style='color:#E67E22'>Cim irregular detectat</span>")
                repair_layout.addWidget(repair_status)

            repair_btn = QPushButton("Obrir dialeg de reparacio...")
            repair_btn.setStyleSheet(
                "QPushButton { border: 1px solid #E67E22; border-radius: 3px;"
                " padding: 4px 12px; font-size: 11px; color: #E67E22; }"
                "QPushButton:hover { background: #FEF9E7; }")
            repair_btn.clicked.connect(self._on_repair_click)
            repair_layout.addWidget(repair_btn, alignment=Qt.AlignLeft)

            force_repair_btn = QPushButton("Forcar reparacio")
            force_repair_btn.setStyleSheet(
                "QPushButton { border: 1px solid #CED4DA; border-radius: 3px;"
                " padding: 4px 12px; font-size: 10px; color: #888; }"
                "QPushButton:hover { background: #F0F0F0; }")
            force_repair_btn.setToolTip("Forcar reparacio encara que no s'hagi auto-detectat")
            force_repair_btn.clicked.connect(self._on_repair_click)
            repair_layout.addWidget(force_repair_btn, alignment=Qt.AlignLeft)

            self._repair_group.setVisible(has_repair or not self.is_blank)
            exp_layout.addWidget(self._repair_group)

        # --- 5. Compose section ---
        if not self.is_control:
            tc = self.sample_data.get("timeout_composability", {})
            has_composition = tc.get("composable", False)
            timeout_info = {}
            replicas = self.sample_data.get("replicas", {})
            for rv in replicas.values():
                if isinstance(rv, dict) and rv.get("timeout_info", {}).get("n_timeouts", 0) > 0:
                    timeout_info = rv.get("timeout_info", {})
                    break

            if has_composition or timeout_info.get("n_timeouts", 0) > 0:
                self._compose_group = QGroupBox("Composicio repliques (timeout)")
                self._compose_group.setStyleSheet(
                    "QGroupBox { font-size: 11px; font-weight: bold; color: #555;"
                    " border: 1px solid #E0E0E0; border-radius: 4px;"
                    " margin-top: 6px; padding-top: 14px; }"
                    "QGroupBox::title { subcontrol-position: top left; padding: 0 6px; }")
                compose_layout = QVBoxLayout(self._compose_group)
                compose_layout.setSpacing(4)

                sel_key = (self.sample_data.get("selected", {}) or {}).get("doc", "1")
                sel_rep = replicas.get(sel_key, {})
                composed = bool(sel_rep.get("timeout_composition"))

                if composed:
                    compose_status = QLabel(
                        "<span style='color:#27AE60'>\u2713 Composicio aplicada</span>")
                elif has_composition:
                    coverage = tc.get("coverage_pct", 100)
                    compose_status = QLabel(
                        f"<span style='color:#3498DB'>Composable ({coverage:.0f}% cobertura)</span>")
                else:
                    compose_status = QLabel(
                        "<span style='color:#888'>Timeouts detectats, no composable</span>")
                compose_layout.addWidget(compose_status)

                if has_composition:
                    compose_btn = QPushButton("Composar repliques...")
                    compose_btn.setStyleSheet(
                        "QPushButton { border: 1px solid #3498DB; border-radius: 3px;"
                        " padding: 4px 12px; font-size: 11px; color: #3498DB; }"
                        "QPushButton:hover { background: #EBF5FB; }")
                    compose_btn.clicked.connect(self._on_compose_click)
                    compose_layout.addWidget(compose_btn, alignment=Qt.AlignLeft)

                exp_layout.addWidget(self._compose_group)

        # --- 6. Fractions ---
        if not self.is_control:
            self._fractions_label = QLabel()
            self._fractions_label.setStyleSheet(
                "font-size: 11px; color: #555; padding: 2px 0;")
            self._fractions_label.setWordWrap(True)
            exp_layout.addWidget(self._fractions_label)

        # --- 7. Anomalies list ---
        self._anomalies_label = QLabel()
        self._anomalies_label.setStyleSheet(
            "font-size: 11px; color: #666; padding: 2px 0;")
        self._anomalies_label.setWordWrap(True)
        exp_layout.addWidget(self._anomalies_label)

        self._main_layout.addWidget(self._expanded_widget)
        self._expanded_widget.setVisible(True)

        # Populate expanded data
        self._update_expanded_data()

    def _build_doc_combo(self):
        """Populate the DOC replica combo."""
        replicas = self.sample_data.get("replicas", {})
        recommendation = self.sample_data.get("recommendation", {})
        selected = self.sample_data.get("selected", {})
        doc_rec = (recommendation.get("doc") or {}).get("replica", "1")
        doc_sel = selected.get("doc", doc_rec)

        self._doc_combo.blockSignals(True)
        self._doc_combo.clear()
        for rep_num in sorted(replicas.keys(), key=lambda x: int(x) if x.isdigit() else 999):
            label = f"R{rep_num}"
            rep_data = replicas.get(rep_num, {})
            if isinstance(rep_data, dict):
                _idx = rep_data.get("injection_index")
                _slabel = rep_data.get("_source_label", "")
                _ssuffix = _slabel if _slabel and _slabel != "A" else ""
                if _idx is not None:
                    label += f" ({_idx}{_ssuffix})"
            if rep_num == doc_rec:
                label += " \u2605"  # star
            self._doc_combo.addItem(label, rep_num)
            if rep_num == doc_sel:
                self._doc_combo.setCurrentIndex(self._doc_combo.count() - 1)

        # "Comp" if any replica has timeout_composition
        has_composition = any(
            r.get("timeout_composition") for r in replicas.values()
            if isinstance(r, dict))
        if has_composition:
            self._doc_combo.addItem("Comp", "comp")
            if doc_sel == "comp":
                self._doc_combo.setCurrentIndex(self._doc_combo.count() - 1)
        self._doc_combo.addItem("Cap", "none")
        if doc_sel == "none":
            self._doc_combo.setCurrentIndex(self._doc_combo.count() - 1)
        self._doc_combo.blockSignals(False)

    def _build_dad_combo(self):
        """Populate the DAD replica combo."""
        replicas = self.sample_data.get("replicas", {})
        recommendation = self.sample_data.get("recommendation", {})
        selected = self.sample_data.get("selected", {})
        dad_rec = (recommendation.get("dad") or {}).get("replica", "1")
        dad_sel = selected.get("dad", dad_rec)

        self._dad_combo.blockSignals(True)
        self._dad_combo.clear()
        for rep_num in sorted(replicas.keys(), key=lambda x: int(x) if x.isdigit() else 999):
            label = f"R{rep_num}"
            rep_data = replicas.get(rep_num, {})
            if isinstance(rep_data, dict):
                _idx = rep_data.get("injection_index")
                _slabel = rep_data.get("_source_label", "")
                _ssuffix = _slabel if _slabel and _slabel != "A" else ""
                if _idx is not None:
                    label += f" ({_idx}{_ssuffix})"
            if rep_num == dad_rec:
                label += " \u2605"
            self._dad_combo.addItem(label, rep_num)
            if rep_num == dad_sel:
                self._dad_combo.setCurrentIndex(self._dad_combo.count() - 1)
        self._dad_combo.addItem("Cap", "none")
        if dad_sel == "none":
            self._dad_combo.setCurrentIndex(self._dad_combo.count() - 1)
        self._dad_combo.blockSignals(False)

    def _update_expanded_data(self):
        """Refresh expanded content data."""
        self._update_nav_label()
        self._render_chromatogram()
        if not self.is_control:
            self._update_metrics()
            self._update_fractions()
        self._update_anomalies()

    def _update_nav_label(self):
        """Update navigation label (sample N/M)."""
        if not hasattr(self, '_nav_label'):
            return
        # Find index among all cards in parent
        parent = self.parent()
        if parent is None:
            self._nav_label.setText(f"<b>{self.sample_name}</b>")
            return
        # Try to find our index
        cards = parent.findChildren(SampleCard)
        # Filter to cards in same category
        same_cards = [c for c in cards
                      if c.is_blank == self.is_blank
                      and c.is_control == self.is_control
                      and not c.is_blank and not c.is_control]  # only regular
        if not same_cards:
            same_cards = cards
        try:
            idx = same_cards.index(self)
            self._nav_label.setText(
                f"<b>{self.sample_name}</b>  ({idx + 1}/{len(same_cards)})")
        except ValueError:
            self._nav_label.setText(f"<b>{self.sample_name}</b>")

    def _render_chromatogram(self):
        """Render DOC chromatogram (lazy, only on first expand)."""
        if self._chromatogram_rendered or not HAS_MATPLOTLIB or self._figure is None:
            return
        self._chromatogram_rendered = True

        replicas = self.sample_data.get("replicas", {})
        ax = self._figure.add_subplot(111)

        # Plot all DOC replicas overlaid
        colors = ['#2E86AB', '#E74C3C', '#F39C12', '#27AE60', '#9B59B6']
        for i, (rk, rd) in enumerate(sorted(replicas.items(),
                                            key=lambda x: int(x[0]) if x[0].isdigit() else 999)):
            if not isinstance(rd, dict):
                continue
            t = rd.get("t_doc")
            y = rd.get("y_doc_net")
            if t is not None and y is not None and len(t) > 0:
                color = colors[i % len(colors)]
                label_suffix = rd.get("_source_label", "")
                suffix_txt = f" ({label_suffix})" if label_suffix and label_suffix != "A" else ""
                ax.plot(t, y, label=f"R{rk}{suffix_txt}",
                        linewidth=1.0, alpha=0.8, color=color)

            # Draw timeout zones for this replica
            timeout_info_r = rd.get("timeout_info")
            if timeout_info_r and timeout_info_r.get("n_timeouts", 0) > 0:
                draw_timeout_zones_on_ax(ax, timeout_info_r,
                                          color_r1=colors[i % len(colors)])

        # Determine x limits
        processed = self.main_window.processed_data or {}
        method = processed.get("method", "COLUMN")
        is_bp = method.upper() == "BP"
        ax.set_xlim(0, 12 if is_bp else 70)
        ax.set_xlabel("min", fontsize=8)
        ax.set_ylabel("ppb", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(fontsize=7, loc="upper right", framealpha=0.7)

        try:
            self._figure.tight_layout()
        except Exception:
            pass
        self._canvas.draw_idle()

    def _update_metrics(self):
        """Update the metrics label in expanded view."""
        if not hasattr(self, '_metrics_label'):
            return
        _, doc_rep = resolve_doc_replica(self.sample_data)
        quant = self.sample_data.get("quantification", {})

        parts = []
        ppm = (quant.get("concentration_ppm_direct")
               or quant.get("concentration_ppm"))
        if ppm:
            parts.append(f"ppm: <b>{ppm:.2f}</b>")
        ppm_u = quant.get("concentration_ppm_uib")
        if ppm_u:
            parts.append(f"UIB: <b>{ppm_u:.2f}</b>")

        snr_info = doc_rep.get("snr_info") or {}
        snr = snr_info.get("snr_direct", 0)
        if snr:
            parts.append(f"SNR: {snr:.0f}")

        areas = doc_rep.get("areas") or {}
        area_doc = (areas.get("DOC") or {}).get("total", 0)
        if area_doc:
            parts.append(f"A: {area_doc:.0f}")

        self._metrics_label.setText(" | ".join(parts) if parts else "")

    def _update_fractions(self):
        """Update fractions text."""
        if not hasattr(self, '_fractions_label'):
            return
        _, doc_rep = resolve_doc_replica(self.sample_data)
        areas = doc_rep.get("areas") or {}
        doc_areas = areas.get("DOC") or {}
        total = doc_areas.get("total", 0) or 0

        if total <= 0:
            self._fractions_label.setText("")
            return

        parts = []
        for frac in FRACTION_ORDER:
            val = doc_areas.get(frac, 0) or 0
            pct = val / total * 100 if total > 0 else 0
            color = FRACTION_COLORS.get(frac, "#666")
            parts.append(
                f"<span style='color:{color}'>{frac}</span> {pct:.0f}%")

        self._fractions_label.setText(
            "Fraccions: " + " \u00b7 ".join(parts))

    def _update_anomalies(self):
        """Update anomalies list text."""
        if not hasattr(self, '_anomalies_label'):
            return

        # Merge anomalies from all replicas
        replicas = self.sample_data.get("replicas", {})
        all_anomalies = []
        seen_codes = set()
        for rk, rd in replicas.items():
            if not isinstance(rd, dict):
                continue
            for a in rd.get("anomalies", []):
                code = a.get("code") if isinstance(a, dict) else str(a)
                if code not in seen_codes:
                    all_anomalies.append(a)
                    seen_codes.add(code)

        if not all_anomalies:
            self._anomalies_label.setText("")
            return

        classified = classify_anomalies(all_anomalies)
        parts = []
        for key, icon, color in [
            ("blocker", "\u26d4", "#E74C3C"),
            ("warning", "\u26a0", "#F39C12"),
            ("info", "\u2139", "#3498DB"),
            ("repaired", "\u2713", "#27AE60"),
        ]:
            for a in classified.get(key, []):
                code = a.get("code") if isinstance(a, dict) else str(a)
                entry = ANOMALY_CATALOG.get(code, {})
                lbl = (a.get("label") if isinstance(a, dict) else None) or entry.get("label", code)
                action = entry.get("action", "")
                line = f"<span style='color:{color}'>{icon} {lbl}</span>"
                if action:
                    line += f" <span style='color:#999; font-size:10px'>\u2192 {action}</span>"
                parts.append(line)

        self._anomalies_label.setText("<br>".join(parts))

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _navigate(self, direction):
        """Navigate to prev/next card. direction: -1 or +1."""
        parent = self.parent()
        if parent is None:
            return
        cards = [c for c in parent.findChildren(SampleCard)
                 if not c.is_blank and not c.is_control]
        if not cards:
            cards = parent.findChildren(SampleCard)
        try:
            idx = cards.index(self)
        except ValueError:
            return
        new_idx = idx + direction
        if 0 <= new_idx < len(cards):
            target = cards[new_idx]
            # Request expand on the target (which will collapse us)
            target.expand_requested.emit(target.sample_name)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _on_doc_changed(self):
        """DOC replica changed."""
        new_replica = self._doc_combo.currentData()
        if new_replica is None:
            return
        self.sample_data.setdefault("selected", {})["doc"] = new_replica

        if new_replica == "none":
            self.sample_data["sample_valid"] = False
            self.sample_data["quantification"] = {
                "concentration_ppm": None,
                "concentration_ppm_direct": None,
                "concentration_ppm_uib": None,
                "area_total": None,
                "valid": False,
                "reason": "Usuari ha seleccionat 'Cap' per DOC"
            }
        else:
            self.sample_data["sample_valid"] = True

        # Re-render chromatogram
        self._chromatogram_rendered = False
        if self._figure:
            self._figure.clear()
            self._render_chromatogram()

        self._update_compact_data()
        if hasattr(self, '_metrics_label'):
            self._update_metrics()
        if hasattr(self, '_fractions_label'):
            self._update_fractions()
        self.data_changed.emit(self.sample_name)

    def _on_dad_changed(self):
        """DAD replica changed."""
        new_replica = self._dad_combo.currentData()
        if new_replica is None:
            return
        self.sample_data.setdefault("selected", {})["dad"] = new_replica
        self._update_compact_data()
        self.data_changed.emit(self.sample_name)

    def _on_repair_click(self):
        """Open repair dialog."""
        from .repair_dialog import JaggedPeakRepairDialog
        method = "COLUMN"
        if self.main_window.processed_data:
            method = self.main_window.processed_data.get("method", "COLUMN")
        dialog = JaggedPeakRepairDialog(
            self.sample_name, self.sample_data, method, parent=self)
        dialog.repair_completed.connect(self._on_repair_done)
        dialog.exec()

    def _on_compose_click(self):
        """Open composition dialog."""
        from .composition_dialog import TimeoutCompositionDialog
        is_bp = False
        if self.main_window.processed_data:
            is_bp = self.main_window.processed_data.get("method", "COLUMN").upper() == "BP"
        dialog = TimeoutCompositionDialog(
            self.sample_name, self.sample_data, is_bp=is_bp, parent=self)
        dialog.composition_completed.connect(self._on_repair_done)
        dialog.exec()

    def _on_repair_done(self, sample_name):
        """After repair or compose action."""
        # Refresh combos (may need to add "Comp")
        if hasattr(self, '_doc_combo'):
            self._build_doc_combo()
        self._update_compact_data()
        if hasattr(self, '_metrics_label'):
            self._update_metrics()
        if hasattr(self, '_fractions_label'):
            self._update_fractions()
        self._update_anomalies()
        # Re-render chromatogram
        self._chromatogram_rendered = False
        if self._figure:
            self._figure.clear()
            self._render_chromatogram()
        self.data_changed.emit(sample_name)

    def _open_detail_dialog(self):
        """Open the full SampleDetailDialog (same as old double-click)."""
        from .dialogs import SampleDetailDialog
        method = "COLUMN"
        if self.main_window.processed_data:
            method = self.main_window.processed_data.get("method", "COLUMN")
        dialog = SampleDetailDialog(
            self.sample_name,
            self.sample_data,
            method,
            parent=self.window()
        )
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    # ------------------------------------------------------------------
    # Public refresh
    # ------------------------------------------------------------------

    def refresh(self):
        """Full refresh of card data (compact + expanded if visible)."""
        self._update_compact_data()
        if self._expanded and self._expanded_widget and self._expanded_widget.isVisible():
            self._update_expanded_data()
