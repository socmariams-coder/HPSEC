"""Detall i revisió manual d'una rèplica KHP.

Permet:
  - Veure el cromatograma amb pic, baseline i límits d'integració
  - Veure el fit bigaussià
  - Forçar reparació (paràbola) per pics dubtosos
  - Marcar/desmarcar com a outlier manualment
"""
from __future__ import annotations

import numpy as np

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame,
    QSizePolicy, QMessageBox,
)

try:
    import matplotlib
    matplotlib.use("Qt5Agg", force=False)
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False


class KHPDetailDialog(QDialog):
    """Diàleg de revisió i reparació d'una rèplica KHP."""

    repair_applied = Signal(int, str, dict)   # rep_num, signal, new_data
    outlier_toggled = Signal(int, str, bool)  # rep_num, signal, is_outlier

    def __init__(self, khp_data: dict, signal: str = "direct", parent=None):
        """
        Args:
            khp_data: dict amb t_doc, y_doc, peak_info, area, bigaussian_doc,
                      calibration_anomalies, is_outlier, etc.
            signal: "direct" o "uib"
        """
        super().__init__(parent)
        self.khp_data = khp_data
        self.signal = signal
        self._repaired_data = None  # Resultat de reparació pendent d'aplicar

        rep_num = khp_data.get('replica_num', 1)
        conc = khp_data.get('conc_ppm', 0)
        is_bp = khp_data.get('is_bp', False)
        method = "BP" if is_bp else "COL"
        self.setWindowTitle(f"Detall KHP — R{rep_num} {method} {conc:g} ppm — {signal.upper()}")
        self.resize(900, 650)

        self._build_ui()
        self._render_chromatogram()
        self._render_metrics()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Header amb info
        header = QHBoxLayout()
        rep_num = self.khp_data.get('replica_num', 1)
        conc = self.khp_data.get('conc_ppm', 0)
        vol = self.khp_data.get('volume_uL', 0)
        ug = conc * vol / 1000 if conc and vol else 0
        title = QLabel(
            f"<b>Rèplica R{rep_num}</b> &nbsp;|&nbsp; "
            f"{conc:g} ppm × {vol:g} µL = <b>{ug:.4f} µg DOC</b>"
        )
        title.setStyleSheet("font-size: 12px;")
        header.addWidget(title)
        header.addStretch()

        is_outlier = bool(self.khp_data.get('is_outlier', False))
        self._outlier_badge = QLabel()
        self._update_outlier_badge(is_outlier)
        header.addWidget(self._outlier_badge)
        layout.addLayout(header)

        # Cromatograma
        if HAS_MATPLOTLIB:
            self._fig = Figure(figsize=(8, 4), dpi=100)
            self._fig.set_facecolor("white")
            self._ax = self._fig.add_subplot(111)
            self._canvas = FigureCanvas(self._fig)
            self._canvas.setMinimumHeight(320)
            self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            layout.addWidget(self._canvas)
        else:
            layout.addWidget(QLabel("matplotlib no disponible"))

        # Mètriques
        metrics_frame = QFrame()
        metrics_frame.setFrameStyle(QFrame.StyledPanel)
        metrics_frame.setStyleSheet(
            "QFrame { background: #FAFAFA; border: 1px solid #E0E0E0; border-radius: 4px; }"
        )
        self._metrics_layout = QVBoxLayout(metrics_frame)
        self._metrics_layout.setSpacing(4)
        self._metrics_label = QLabel()
        self._metrics_label.setStyleSheet("font-family: monospace; font-size: 11px;")
        self._metrics_label.setWordWrap(True)
        self._metrics_layout.addWidget(self._metrics_label)
        layout.addWidget(metrics_frame)

        # Botons
        btn_row = QHBoxLayout()
        self._btn_repair = QPushButton("⚙ Reparar pic (paràbola)")
        self._btn_repair.setToolTip(
            "Aplicar reparació amb paràbola al pic seleccionat (force=True).\n"
            "Útil quan el pic té dents subtils que detect_irregular_top no detecta."
        )
        self._btn_repair.clicked.connect(self._on_repair_clicked)
        btn_row.addWidget(self._btn_repair)

        self._btn_outlier = QPushButton()
        self._update_outlier_button()
        self._btn_outlier.clicked.connect(self._on_toggle_outlier)
        btn_row.addWidget(self._btn_outlier)

        btn_row.addStretch()
        btn_close = QPushButton("Tancar")
        btn_close.clicked.connect(self.accept)
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

    def _update_outlier_badge(self, is_outlier: bool):
        if is_outlier:
            self._outlier_badge.setText("<b style='color:#C62828'>OUTLIER (exclosa de la calibració)</b>")
        else:
            self._outlier_badge.setText("<span style='color:#2E7D32'>● Vàlida per calibrar</span>")

    def _update_outlier_button(self):
        is_outlier = bool(self.khp_data.get('is_outlier', False))
        if is_outlier:
            self._btn_outlier.setText("✓ Desmarcar outlier")
            self._btn_outlier.setStyleSheet(
                "QPushButton { background: #FFE0B2; color: #BF360C; padding: 6px 12px; }")
        else:
            self._btn_outlier.setText("⚠ Marcar com a outlier")
            self._btn_outlier.setStyleSheet(
                "QPushButton { background: #FFCDD2; color: #B71C1C; padding: 6px 12px; }")

    def _render_chromatogram(self):
        if not HAS_MATPLOTLIB:
            return
        ax = self._ax
        ax.clear()

        t = np.asarray(self.khp_data.get('t_doc', []))
        y = np.asarray(self.khp_data.get('y_doc', []))
        if len(t) < 5 or len(y) < 5:
            ax.text(0.5, 0.5, "Sense dades de cromatograma",
                   ha='center', va='center', transform=ax.transAxes, color='gray')
            self._canvas.draw()
            return

        is_bp = bool(self.khp_data.get('is_bp', False))
        peak_info = self.khp_data.get('peak_info') or {}
        baseline = peak_info.get('baseline_level', 0) or 0

        # Senyal original
        ax.plot(t, y, '-', color='black', lw=1.0, alpha=0.8, label='senyal DOC')

        # Senyal reparat (si disponible)
        y_repaired = self.khp_data.get('y_doc_repaired')
        if y_repaired is not None and len(y_repaired) == len(y):
            y_rep = np.asarray(y_repaired)
            if not np.allclose(y_rep, y, atol=1e-6):
                ax.plot(t, y_rep, '--', color='red', lw=1.4, alpha=0.85,
                       label='reparat (auto)')

        # Senyal reparat NOU (pendent d'aplicar)
        if self._repaired_data is not None:
            y_new = self._repaired_data.get('y_repaired')
            if y_new is not None:
                ax.plot(t, y_new, ':', color='#1976D2', lw=2.0, alpha=0.9,
                       label='reparat (force, pendent)')

        # Baseline
        ax.axhline(baseline, color='brown', lw=1.4, ls='-', alpha=0.7,
                  label=f'baseline = {baseline:.2f}')

        # Pic màxim
        p_idx = peak_info.get('peak_idx')
        if p_idx is not None and 0 <= p_idx < len(t):
            ax.plot(t[p_idx], y[p_idx], 'v', color='red', markersize=10, zorder=5)

        # Límits d'integració
        l_idx = peak_info.get('left_idx', peak_info.get('peak_left_idx'))
        r_idx = peak_info.get('right_idx', peak_info.get('peak_right_idx'))
        if l_idx is not None and r_idx is not None and l_idx < r_idx:
            mask = np.zeros(len(t), dtype=bool)
            mask[l_idx:r_idx+1] = True
            ax.fill_between(t[mask], baseline, y[mask], color='green', alpha=0.20,
                          label=f"àrea = {peak_info.get('area', self.khp_data.get('area', 0)):.1f}")
            ax.axvline(t[l_idx], color='green', lw=1.2, ls='--', alpha=0.7)
            ax.axvline(t[r_idx], color='green', lw=1.2, ls='--', alpha=0.7)

        # Bigauss fit overlay
        bg = self.khp_data.get('bigaussian_doc') or {}
        if bg.get('mu') and bg.get('amplitude'):
            mu = bg['mu']; A = bg['amplitude']
            sl = bg.get('sigma_left', 0); sr = bg.get('sigma_right', 0)
            if sl > 0 and sr > 0:
                t_fit = np.linspace(mu - 4*sl, mu + 4*sr, 200)
                y_fit = np.where(t_fit < mu,
                                A * np.exp(-((t_fit - mu)**2) / (2*sl**2)),
                                A * np.exp(-((t_fit - mu)**2) / (2*sr**2)))
                ax.plot(t_fit, y_fit + baseline, '-', color='#1976D2', lw=1.0,
                       alpha=0.6, label=f"bigauss R²={bg.get('r2', 0):.3f}")

        ax.set_xlabel('t (min)')
        ax.set_ylabel('DOC (ppb)')
        ax.grid(alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
        ax.set_xlim(0, 10) if is_bp else ax.set_xlim(15, 32)

        self._fig.tight_layout()
        self._canvas.draw()

    def _render_metrics(self):
        khp = self.khp_data
        peak_info = khp.get('peak_info') or {}
        area = peak_info.get('area', khp.get('area', 0))
        bg = khp.get('bigaussian_doc') or {}
        anomalies = khp.get('calibration_anomalies', [])

        lines = []
        lines.append(f"Àrea: {area:>9.2f}        Pic alçada: {peak_info.get('height', 0):>7.1f} ppb")
        lines.append(f"t_max: {peak_info.get('t_max', 0):>8.3f} min   FWHM: {khp.get('fwhm_doc', 0):>6.3f} min   Asym: {khp.get('symmetry', 1.0):.3f}")
        lines.append(f"Bigauss: status={bg.get('status', '?'):<8} R²={bg.get('r2', 0):.4f}  asym={bg.get('asymmetry', 1.0):.2f}")
        lines.append(f"SNR: {khp.get('snr', 0):.1f}    Shift vs 254: {khp.get('shift_sec', 0):+.1f} s")

        if self._repaired_data:
            new_area = self._repaired_data.get('new_area', 0)
            delta_pct = ((new_area - area) / area * 100) if area > 0 else 0
            lines.append("")
            lines.append(f"📊 Reparat (pendent): àrea = {new_area:.2f} (Δ {delta_pct:+.2f}%)")

        if anomalies:
            lines.append("")
            lines.append("Anomalies:")
            for a in anomalies:
                if isinstance(a, dict):
                    sev = a.get('severity', 'info')
                    icon = '✘' if sev == 'blocker' else '⚠' if sev == 'warning' else 'ℹ'
                    label = a.get('label', a.get('code', ''))
                    lines.append(f"  {icon} [{sev}] {label}")

        self._metrics_label.setText('<br>'.join(line.replace(' ', '&nbsp;') for line in lines))

    def _on_repair_clicked(self):
        """Aplica reparació amb paràbola (force=True) i mostra el resultat."""
        try:
            from hpsec_core import repair_with_parabola, find_peak_boundaries
            from scipy.integrate import trapezoid
        except ImportError as e:
            QMessageBox.warning(self, "Error", f"Imports fallits: {e}")
            return

        t = np.asarray(self.khp_data.get('t_doc', []))
        y = np.asarray(self.khp_data.get('y_doc', []))
        peak_info = self.khp_data.get('peak_info') or {}
        p_idx = peak_info.get('peak_idx')
        l_idx = peak_info.get('left_idx', peak_info.get('peak_left_idx'))
        r_idx = peak_info.get('right_idx', peak_info.get('peak_right_idx'))
        baseline = peak_info.get('baseline_level', 0) or 0
        is_bp = bool(self.khp_data.get('is_bp', False))

        if not (len(t) > 5 and p_idx is not None and l_idx is not None and r_idx is not None):
            QMessageBox.warning(self, "Reparació", "Falten dades del pic per reparar.")
            return

        # Segment al voltant del pic per la reparació
        half_w = 3.0 if is_bp else 5.0
        seg_mask = (t >= t[p_idx] - half_w) & (t <= t[p_idx] + half_w)

        try:
            y_seg_rep, repair_info, was_repaired = repair_with_parabola(
                t[seg_mask], y[seg_mask], force=True)
        except Exception as e:
            QMessageBox.warning(self, "Reparació", f"Error a repair_with_parabola: {e}")
            return

        if not was_repaired:
            QMessageBox.information(
                self, "Reparació",
                "La paràbola no ha pogut reparar el pic\n"
                "(possiblement els ancoratges no han trobat punts vàlids)."
            )
            return

        # Reconstruir senyal complet i recalcular àrea
        y_full_repaired = y.copy()
        y_full_repaired[seg_mask] = y_seg_rep

        # Recalcular límits sobre senyal reparat
        try:
            l_new, r_new = find_peak_boundaries(
                t, y_full_repaired, p_idx, baseline, is_bp=is_bp)
        except Exception:
            l_new, r_new = l_idx, r_idx

        new_area = float(trapezoid(
            np.maximum(y_full_repaired[l_new:r_new+1] - baseline, 0),
            t[l_new:r_new+1]
        ))

        self._repaired_data = {
            'y_repaired': y_full_repaired,
            'new_area': new_area,
            'new_left_idx': int(l_new),
            'new_right_idx': int(r_new),
            'repair_info': repair_info,
        }

        # Re-render amb el repair pendent
        self._render_chromatogram()
        self._render_metrics()

        # Modificar botó per "Aplicar"
        self._btn_repair.setText("✓ Aplicar reparació")
        try:
            self._btn_repair.clicked.disconnect()
        except Exception:
            pass
        self._btn_repair.clicked.connect(self._apply_repair)
        self._btn_repair.setStyleSheet(
            "QPushButton { background: #C8E6C9; color: #1B5E20; padding: 6px 12px; font-weight: bold; }")

    def _apply_repair(self):
        """Confirma la reparació i emet el senyal."""
        if not self._repaired_data:
            return
        rep_num = self.khp_data.get('replica_num', 1)
        self.repair_applied.emit(rep_num, self.signal, self._repaired_data)
        QMessageBox.information(
            self, "Reparació aplicada",
            f"Àrea reparada: {self._repaired_data['new_area']:.2f}\n"
            "Cal recalcular la calibració per que es propagui."
        )
        # Reset
        self._repaired_data = None

    def _on_toggle_outlier(self):
        is_outlier_now = bool(self.khp_data.get('is_outlier', False))
        new_state = not is_outlier_now
        rep_num = self.khp_data.get('replica_num', 1)
        self.khp_data['is_outlier'] = new_state
        self._update_outlier_badge(new_state)
        self._update_outlier_button()
        self.outlier_toggled.emit(rep_num, self.signal, new_state)
