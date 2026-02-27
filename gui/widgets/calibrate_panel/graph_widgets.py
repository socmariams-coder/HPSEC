"""
HPSEC Suite - Calibrate Graph Widgets
=====================================

Widgets de visualització per calibració KHP.
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

# Importar funció bigaussiana per plotar el fit
try:
    from hpsec_core import bigaussian
    HAS_BIGAUSSIAN = True
except ImportError:
    HAS_BIGAUSSIAN = False


class KHPReplicaGraphWidget(QWidget):
    """Widget que mostra gràfics de KHP per rèplica: DOC Direct+UIB+254nm unificats."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(10, 3.5), dpi=100)
        self.canvas = FigureCanvas(self.figure)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

        self.setMinimumHeight(250)

    def plot_replicas(self, replicas_direct, replicas_uib=None):
        """
        Grafica rèpliques: 1 fila × N cols, cada subplot amb DOC+UIB+254nm superposats.

        Args:
            replicas_direct: Lista de dicts amb dades Direct per cada rèplica
            replicas_uib: Lista de dicts amb dades UIB per cada rèplica (opcional)
        """
        self.figure.clear()

        if not replicas_direct:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "No hi ha dades KHP disponibles",
                   ha='center', va='center', fontsize=12, color='gray')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            self.canvas.draw()
            return

        n_replicas = len(replicas_direct)
        has_uib = replicas_uib and len(replicas_uib) > 0

        # 1 row × N cols: all signals unified per subplot
        for i, rep_direct in enumerate(replicas_direct):
            ax = self.figure.add_subplot(1, n_replicas, i + 1)
            rep_uib = replicas_uib[i] if has_uib and i < len(replicas_uib) else None
            self._plot_unified(ax, rep_direct, rep_uib, i + 1)

        self.figure.tight_layout()
        self.canvas.draw()

    def _plot_unified(self, ax, rep_direct, rep_uib, replica_num):
        """Plot unified subplot: Direct (blau) + UIB (verd) + 254nm (taronja, eix dret).
        Inclou zoom inset de la zona del pic."""
        t_doc = rep_direct.get('t_doc')
        y_doc = rep_direct.get('y_doc')
        area = rep_direct.get('area', 0)

        if t_doc is None or y_doc is None:
            self._plot_doc_fallback(ax, rep_direct, replica_num)
            return

        t_doc = np.asarray(t_doc)
        y_doc = np.asarray(y_doc)

        # Colors consistents amb analyze panel
        color_direct = '#1565C0'   # Deep blue (analyze R1)
        color_uib = '#2E7D32'     # Dark green (analyze UIB)
        color_254 = '#E67E22'     # Orange (254nm)

        # --- Direct DOC curve ---
        ax.plot(t_doc, y_doc, color=color_direct, linewidth=1.2, label='Direct')

        # Shaded integrated area (Direct)
        li = rep_direct.get('peak_left_idx', 0)
        ri = rep_direct.get('peak_right_idx', len(t_doc) - 1)
        if 0 <= li < ri < len(t_doc):
            ax.fill_between(t_doc[li:ri+1], 0, y_doc[li:ri+1],
                           alpha=0.15, color=color_direct)
            ax.axvline(t_doc[li], color=color_direct, linestyle=':', linewidth=0.8, alpha=0.5)
            ax.axvline(t_doc[ri], color=color_direct, linestyle=':', linewidth=0.8, alpha=0.5)

        # Cim irregular: show original if repaired
        if rep_direct.get('irregular_top_repaired', rep_direct.get('batman_repaired')) and rep_direct.get('y_doc_repaired') is not None:
            y_repaired = np.asarray(rep_direct['y_doc_repaired'])
            if len(y_repaired) == len(t_doc):
                ax.plot(t_doc, y_repaired, color='#E74C3C', linewidth=0.8,
                       alpha=0.4, linestyle=':')

        # Bigaussian fit (dashed over DOC)
        self._plot_bigaussian_fit(ax, rep_direct)

        # --- UIB DOC curve (overlaid) ---
        area_uib = 0
        has_uib_curve = False
        t_uib_arr = None
        y_uib_arr = None
        li_u = 0
        ri_u = 0
        if rep_uib is not None:
            t_uib = rep_uib.get('t_doc')
            y_uib = rep_uib.get('y_doc')
            area_uib = rep_uib.get('area', 0)
            if t_uib is not None and y_uib is not None:
                has_uib_curve = True
                t_uib_arr = np.asarray(t_uib)
                y_uib_arr = np.asarray(y_uib)
                ax.plot(t_uib_arr, y_uib_arr, color=color_uib, linewidth=1.2, label='UIB')
                li_u = rep_uib.get('peak_left_idx', 0)
                ri_u = rep_uib.get('peak_right_idx', len(t_uib_arr) - 1)
                if 0 <= li_u < ri_u < len(t_uib_arr):
                    ax.fill_between(t_uib_arr[li_u:ri_u+1], 0, y_uib_arr[li_u:ri_u+1],
                                   alpha=0.10, color=color_uib)

        # --- 254nm (secondary Y axis) ---
        a254_area = rep_direct.get('a254_area', 0)
        t_dad = rep_direct.get('t_dad')
        y_dad = rep_direct.get('y_dad_254')
        has_254 = False

        if t_dad is not None and y_dad is not None:
            t_dad = np.asarray(t_dad)
            y_dad = np.asarray(y_dad)
            if len(t_dad) > 0 and len(y_dad) > 0:
                has_254 = True
                ax2 = ax.twinx()
                ax2.plot(t_dad, y_dad, color=color_254, linewidth=0.8, alpha=0.7, label='254nm')
                ax2.set_ylabel('254nm', fontsize=6, color=color_254)
                ax2.tick_params(axis='y', labelsize=5, colors=color_254)
                ax2.spines['right'].set_color(color_254)

                dad_peak = rep_direct.get('dad_peak_info', {})
                if dad_peak and dad_peak.get('valid'):
                    d_li = dad_peak.get('left_idx', 0)
                    d_ri = dad_peak.get('right_idx', len(t_dad) - 1)
                    if 0 <= d_li < d_ri < len(t_dad):
                        ax.axvline(t_dad[d_li], color=color_254, linestyle='--', linewidth=0.6, alpha=0.4)
                        ax.axvline(t_dad[d_ri], color=color_254, linestyle='--', linewidth=0.6, alpha=0.4)

        # Axes formatting
        ax.set_xlabel('Temps (min)', fontsize=7)
        ax.set_ylabel('DOC (ppb)', fontsize=7)
        ax.tick_params(axis='both', labelsize=6)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"R{replica_num}", fontsize=8, fontweight='bold')

        # --- Compact legend ---
        legend_parts = [f"Direct A={area:.0f}"]
        if has_uib_curve:
            legend_parts.append(f"UIB A={area_uib:.0f}")
        if has_254:
            legend_parts.append(f"254nm A={a254_area:.1f}")
        legend_text = " | ".join(legend_parts)
        ax.text(0.97, 0.97, legend_text,
                transform=ax.transAxes, fontsize=5.5,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='#BDC3C7'),
                family='monospace')

        # --- Timeout zones ---
        self._draw_timeout_zones(ax, rep_direct)

        # --- Zoom inset (peak area) ---
        self._add_peak_zoom_inset(ax, t_doc, y_doc, li, ri, color_direct,
                                  t_uib_arr, y_uib_arr, li_u, ri_u, color_uib,
                                  has_uib_curve)

    def _add_peak_zoom_inset(self, ax, t_doc, y_doc, li, ri, color_direct,
                             t_uib, y_uib, li_u, ri_u, color_uib, has_uib):
        """Afegeix un inset amb zoom de la zona del pic."""
        if not (0 <= li < ri < len(t_doc)):
            return

        # Determine zoom window: peak ± 1 min margin
        t_left = t_doc[li]
        t_right = t_doc[ri]
        peak_width = t_right - t_left
        margin = max(peak_width * 0.5, 0.3)
        zoom_t_min = t_left - margin
        zoom_t_max = t_right + margin

        # Y range within zoom window
        mask = (t_doc >= zoom_t_min) & (t_doc <= zoom_t_max)
        if mask.sum() < 3:
            return
        y_in_window = y_doc[mask]
        zoom_y_max = float(np.max(y_in_window)) * 1.15
        zoom_y_min = max(0, float(np.min(y_in_window)) - zoom_y_max * 0.05)

        # Create inset axes (lower-right corner)
        try:
            axins = ax.inset_axes([0.55, 0.02, 0.43, 0.45])  # [x, y, width, height]
        except AttributeError:
            return  # matplotlib < 3.0

        # Plot Direct in inset
        axins.plot(t_doc, y_doc, color=color_direct, linewidth=1.0)
        axins.fill_between(t_doc[li:ri+1], 0, y_doc[li:ri+1],
                          alpha=0.20, color=color_direct)
        axins.axvline(t_doc[li], color=color_direct, linestyle=':', linewidth=0.6, alpha=0.6)
        axins.axvline(t_doc[ri], color=color_direct, linestyle=':', linewidth=0.6, alpha=0.6)

        # Plot UIB in inset
        if has_uib and t_uib is not None and y_uib is not None:
            axins.plot(t_uib, y_uib, color=color_uib, linewidth=1.0)
            if 0 <= li_u < ri_u < len(t_uib):
                axins.fill_between(t_uib[li_u:ri_u+1], 0, y_uib[li_u:ri_u+1],
                                  alpha=0.12, color=color_uib)

        axins.set_xlim(zoom_t_min, zoom_t_max)
        axins.set_ylim(zoom_y_min, zoom_y_max)
        axins.tick_params(axis='both', labelsize=4)
        axins.grid(True, alpha=0.2)

        # Draw rectangle indicator on main axes
        try:
            ax.indicate_inset_zoom(axins, edgecolor='#666', linewidth=0.5, alpha=0.5)
        except AttributeError:
            pass

    def _plot_doc_fallback(self, ax, rep, replica_num):
        """Fallback when no raw DOC signal available."""
        area = rep.get('area', 0)
        snr = rep.get('snr', 0)
        bigauss = rep.get('bigaussian_doc')

        if HAS_BIGAUSSIAN and bigauss and bigauss.get('status') in ['VALID', 'CHECK']:
            try:
                amp = bigauss.get('amplitude', 0)
                mu = bigauss.get('mu', 0)
                sigma_l = bigauss.get('sigma_left', 0)
                sigma_r = bigauss.get('sigma_right', 0)
                r2 = bigauss.get('r2', 0)
                if amp > 0 and mu > 0 and sigma_l > 0 and sigma_r > 0:
                    t_fit = np.linspace(mu - 4*sigma_l, mu + 4*sigma_r, 200)
                    y_fit = bigaussian(t_fit, amp, mu, sigma_l, sigma_r, 0)
                    fit_color = '#27AE60' if bigauss.get('status') == 'VALID' else '#F39C12'
                    ax.plot(t_fit, y_fit, color=fit_color, linewidth=1.5,
                           label=f'Fit (R\u00B2={r2:.3f})')
                    ax.fill_between(t_fit, 0, y_fit, alpha=0.15, color=fit_color)
                    ax.set_xlabel('Temps (min)', fontsize=7)
                    ax.set_ylabel('DOC (mAU)', fontsize=7)
                    ax.grid(True, alpha=0.3)
                    ax.set_title(f"R{replica_num}: A={area:.0f} [fit]",
                                fontsize=8, fontweight='bold')
                    ax.legend(fontsize=6, loc='upper right')
                    return
            except Exception:
                pass

        info_lines = []
        if area > 0:
            info_lines.append(f"\u00c0rea: {area:.0f}")
        if snr > 0:
            info_lines.append(f"SNR: {snr:.0f}")
        t_max = rep.get('t_retention', 0) or rep.get('t_max', 0) or rep.get('t_doc_max', 0)
        if t_max > 0:
            info_lines.append(f"t_max: {t_max:.2f} min")
        if info_lines:
            ax.text(0.5, 0.55, "\n".join(info_lines), ha='center', va='center',
                   fontsize=9, color='#2C3E50', family='monospace')
            ax.text(0.5, 0.25, "(sense senyal original)", ha='center', va='center',
                   fontsize=8, color='gray', style='italic')
        else:
            ax.text(0.5, 0.5, "Sense dades", ha='center', va='center',
                   fontsize=10, color='gray')
        ax.set_title(f"R{replica_num}", fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

    def _plot_bigaussian_fit(self, ax, rep):
        """Plot bigaussian fit as dashed line over DOC."""
        bigauss = rep.get('bigaussian_doc')
        if not HAS_BIGAUSSIAN or not bigauss:
            return
        if bigauss.get('status') not in ('VALID', 'CHECK'):
            return
        try:
            amp = bigauss.get('amplitude', 0)
            mu = bigauss.get('mu', 0)
            sigma_l = bigauss.get('sigma_left', 0)
            sigma_r = bigauss.get('sigma_right', 0)
            if amp > 0 and mu > 0 and sigma_l > 0 and sigma_r > 0:
                t_fit = np.linspace(mu - 4*sigma_l, mu + 4*sigma_r, 200)
                y_fit = bigaussian(t_fit, amp, mu, sigma_l, sigma_r, 0)
                fit_color = '#27AE60' if bigauss.get('status') == 'VALID' else '#F39C12'
                ax.plot(t_fit, y_fit, color=fit_color, linewidth=1.3, linestyle='--',
                       alpha=0.8)
        except Exception:
            pass

    def _draw_timeout_zones(self, ax, rep):
        """Draw timeout zones and annotations on a subplot."""
        if not rep.get('has_timeout'):
            return
        timeout_info = rep.get('timeout_info', {})
        timeouts_list = timeout_info.get('timeouts', [])
        peak_info = rep.get('peak_info', {})
        t_peak = peak_info.get('t_max', 0)

        affects_main_peak = False
        for to in timeouts_list:
            t_start = to.get('t_start_min', 0)
            t_end = to.get('t_end_min', 0)
            affected_start = to.get('affected_start_min', t_start - 0.5)
            affected_end = to.get('affected_end_min', t_end + 1.0)
            hits_peak = t_peak > 0 and affected_start <= t_peak <= affected_end
            if hits_peak:
                affects_main_peak = True
                color = '#E74C3C'
                alpha = 0.35
            else:
                color = '#F39C12'
                alpha = 0.2
            ax.axvspan(affected_start, affected_end, alpha=alpha, color=color, zorder=0)
            ax.axvline(t_start, color=color, linestyle='--', linewidth=1.5, alpha=0.8)

        if timeouts_list:
            first_to = timeouts_list[0]
            t_label = first_to.get('t_start_min', 0)
            if affects_main_peak:
                ax.annotate(f'!! TO@{t_label:.1f} PIC!', xy=(0.02, 0.88),
                           xycoords='axes fraction', fontsize=6, color='#C0392B',
                           va='top', fontweight='bold')
            else:
                ax.annotate(f'TO@{t_label:.1f}', xy=(0.02, 0.88),
                           xycoords='axes fraction', fontsize=6, color='#E67E22', va='top')

    def clear(self):
        self.figure.clear()
        self.canvas.draw()


class CalibrationLineWidget(QWidget):
    """Widget que mostra la recta de calibració guardada (regression_data)
    amb punts Direct + UIB i el punt actual de la SEQ."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(6, 2.8), dpi=100)
        self.canvas = FigureCanvas(self.figure)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

        self.setMinimumHeight(200)
        self.setMaximumHeight(300)

    def plot_stored_regression(self, cal_direct, cal_uib=None,
                               current_direct=None, current_uib=None,
                               current_mode='column', current_seq_name='',
                               warning_pct=5.0, fail_pct=10.0):
        """
        Mostra la recta de calibració guardada a Calibration_Reference.json.

        Args:
            cal_direct: dict calibració activa Direct (amb regression_data)
            cal_uib: dict calibració activa UIB (opcional)
            current_direct: dict {'ug_doc': float, 'area': float} punt actual Direct
            current_uib: dict {'ug_doc': float, 'area': float} punt actual UIB
            current_mode: 'column' o 'bp'
            current_seq_name: Nom curt de la SEQ actual (ex: "291")
            warning_pct: % tolerància warning
            fail_pct: % tolerància fail
        """
        self.figure.clear()

        has_uib = cal_uib is not None and cal_uib.get('regression_data')
        n_cols = 2 if has_uib else 1

        mode_key = current_mode.lower()
        if mode_key not in ('column', 'bp'):
            mode_key = 'column'

        # --- Direct subplot ---
        ax_d = self.figure.add_subplot(1, n_cols, 1)
        self._plot_regression_subplot(
            ax_d, cal_direct, current_direct, mode_key,
            current_seq_name, warning_pct, fail_pct,
            signal_label='Direct', color_line='#1565C0', color_point='#1565C0'
        )

        # --- UIB subplot ---
        if has_uib:
            ax_u = self.figure.add_subplot(1, n_cols, 2)
            self._plot_regression_subplot(
                ax_u, cal_uib, current_uib, mode_key,
                current_seq_name, warning_pct, fail_pct,
                signal_label='UIB', color_line='#2E7D32', color_point='#2E7D32'
            )

        self.figure.tight_layout()
        self.canvas.draw()

    def _plot_regression_subplot(self, ax, cal, current_point, mode_key,
                                  current_seq_name, warning_pct, fail_pct,
                                  signal_label='Direct', color_line='#1565C0',
                                  color_point='#1565C0'):
        """Ploteja un subplot amb regressió guardada + punt actual."""

        if not cal:
            ax.text(0.5, 0.5, f"No hi ha calibració {signal_label}",
                   ha='center', va='center', fontsize=9, color='gray')
            ax.axis('off')
            return

        # Extreure RF i intercept per al mode actiu
        rf_dict = cal.get('rf_mass_cal', {})
        intercept_dict = cal.get('intercept', {})

        # Suport format planer i nested
        if isinstance(rf_dict, dict):
            rf = rf_dict.get(mode_key, 0)
        else:
            rf = float(rf_dict) if rf_dict else 0

        if isinstance(intercept_dict, dict):
            intercept = intercept_dict.get(mode_key, 0)
        else:
            intercept = float(intercept_dict) if intercept_dict else 0

        if rf <= 0:
            ax.text(0.5, 0.5, f"RF=0 per {signal_label}/{mode_key}",
                   ha='center', va='center', fontsize=9, color='gray')
            ax.axis('off')
            return

        # Punts de la regressió guardada
        reg_data = cal.get('regression_data', {})
        points = reg_data.get('points', [])
        r2 = cal.get('r2', reg_data.get('r2', 0))
        if isinstance(r2, dict):
            r2 = r2.get(mode_key, 0)
        rms = reg_data.get('residuals_rms', 0)

        # Separar punts inclosos / exclosos
        x_inc, y_inc = [], []
        x_exc, y_exc = [], []
        for p in points:
            ug = p.get('ug_doc', 0)
            area = p.get('area', 0)
            if ug > 0 and area > 0:
                if p.get('excluded', False):
                    x_exc.append(ug)
                    y_exc.append(area)
                else:
                    x_inc.append(ug)
                    y_inc.append(area)

        # Eixos: cobrir tots els punts + punt actual
        all_x = x_inc + x_exc
        all_y = y_inc + y_exc
        if current_point and current_point.get('ug_doc', 0) > 0:
            all_x.append(current_point['ug_doc'])
            all_y.append(current_point.get('area', 0))

        if all_x:
            margin = 0.2
            x_min = max(0, min(all_x) * (1 - margin))
            x_max = max(all_x) * (1 + margin)
        else:
            x_min, x_max = 0, 3

        x_line = np.linspace(x_min, x_max, 100)
        y_line = rf * x_line + intercept

        # Recta de regressió
        eq_label = f'{signal_label}: RF={rf:.0f}'
        if intercept:
            eq_label += f'{intercept:+.0f}'
        if r2:
            eq_label += f' (R²={r2:.4f})'
        ax.plot(x_line, y_line, color=color_line, linewidth=2.0, label=eq_label)

        # Bandes tolerància
        y_w_up = y_line * (1 + warning_pct / 100)
        y_w_lo = y_line * (1 - warning_pct / 100)
        y_f_up = y_line * (1 + fail_pct / 100)
        y_f_lo = y_line * (1 - fail_pct / 100)
        ax.fill_between(x_line, y_f_lo, y_f_up, alpha=0.06, color='#E74C3C')
        ax.fill_between(x_line, y_w_lo, y_w_up, alpha=0.10, color='#F39C12')

        # Punts inclosos (cercles)
        if x_inc:
            ax.scatter(x_inc, y_inc, c=color_point, marker='o', s=40,
                      zorder=5, edgecolors='white', linewidths=0.5, alpha=0.8,
                      label=f'Punts cal. ({len(x_inc)})')

        # Punts exclosos (× vermell)
        if x_exc:
            ax.scatter(x_exc, y_exc, c='#E74C3C', marker='x', s=50,
                      zorder=5, linewidths=1.5, alpha=0.6, label=f'Exclosos ({len(x_exc)})')

        # Punt actual (quadrat verd gran)
        if current_point and current_point.get('ug_doc', 0) > 0:
            cx = current_point['ug_doc']
            cy = current_point.get('area', 0)
            ax.scatter(cx, cy, c='#27AE60', marker='s', s=80,
                      zorder=10, edgecolors='white', linewidths=1.0,
                      label=f'▸{current_seq_name}◂')

            # Desviació vs recta
            y_expected = rf * cx + intercept
            if y_expected > 0:
                dev_pct = (cy - y_expected) / y_expected * 100
                dev_color = '#27AE60' if abs(dev_pct) < warning_pct else (
                    '#F39C12' if abs(dev_pct) < fail_pct else '#E74C3C')
                ax.annotate(f'{dev_pct:+.1f}%', (cx, cy), fontsize=7,
                           fontweight='bold', color=dev_color,
                           xytext=(5, -10), textcoords='offset points')

        # Format
        y_max = max(all_y + [rf * x_max + intercept]) * 1.15 if all_y else rf * x_max * 1.15
        ax.set_xlabel('µg DOC', fontsize=8)
        ax.set_ylabel('Àrea', fontsize=8)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, max(y_max, 1))
        ax.tick_params(axis='both', labelsize=6)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=6, framealpha=0.9)

        # Font de la calibració
        source = cal.get('source', {})
        source_desc = source.get('description', cal.get('id', ''))
        if source_desc and len(source_desc) > 40:
            source_desc = source_desc[:37] + '...'
        ax.set_title(f'{signal_label} — {mode_key.upper()}', fontsize=8, fontweight='bold')

    def clear(self):
        self.figure.clear()
        self.canvas.draw()


class HistoryBarWidget(QWidget):
    """Widget compacte per gràfic de barres històric de KHP amb selecció clicable."""

    from PySide6.QtCore import Signal as _Signal
    bar_selected = _Signal(int)  # Índex real de la barra seleccionada

    def __init__(self, parent=None, ylabel="Àrea", value_key="area"):
        super().__init__(parent)
        self.figure = Figure(figsize=(5, 2.2), dpi=100)
        self.canvas = FigureCanvas(self.figure)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

        self.setMinimumHeight(160)
        self.setMaximumHeight(200)
        self.history_data = []
        self._ylabel = ylabel
        self._value_key = value_key
        self._bars = []
        self._bar_real_indices = []
        self._selected_idx = -1
        self._offset = 0

        # Connexió click
        self.canvas.mpl_connect('button_press_event', self._on_click)

    def _on_click(self, event):
        """Gestiona clics sobre les barres."""
        if event.inaxes is None or not self._bars:
            return
        for i, bar in enumerate(self._bars):
            if bar.contains(event)[0]:
                real_idx = self._bar_real_indices[i]
                self._selected_idx = real_idx
                self._highlight_bar(i)
                self.bar_selected.emit(real_idx)
                return

    def _highlight_bar(self, bar_idx):
        """Marca la barra seleccionada amb un contorn gruixut."""
        for i, bar in enumerate(self._bars):
            bar.set_linewidth(3 if i == bar_idx else 1)
            if i == bar_idx:
                bar.set_edgecolor('#2C3E50')
        self.canvas.draw_idle()

    def plot_history(self, history_list, current_seq_name, valid_indices=None):
        """
        Gràfic de barres compacte amb últimes calibracions.

        Args:
            history_list: Llista de calibracions (ja filtrades i ordenades)
            current_seq_name: Nom de la SEQ actual per marcar-la
            valid_indices: Set d'índexs de calibracions vàlides (no outliers)
        """
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self.history_data = history_list
        self._bars = []
        self._bar_real_indices = []
        self._selected_idx = -1

        if not history_list:
            ax.text(0.5, 0.5, "No hi ha hist\u00f2ric",
                   ha='center', va='center', fontsize=10, color='gray')
            ax.axis('off')
            self.canvas.draw()
            return

        if valid_indices is None:
            valid_indices = set(range(len(history_list)))

        # Últimes 10
        display_cals = history_list[-10:] if len(history_list) > 10 else history_list
        self._offset = len(history_list) - len(display_cals)

        current_short = current_seq_name.replace('_SEQ', '').replace('_BP', '') if current_seq_name else ""

        seq_names = []
        values = []
        colors = []
        edge_colors = []

        for i, cal in enumerate(display_cals):
            real_idx = self._offset + i
            self._bar_real_indices.append(real_idx)
            name = cal.get('seq_name', 'N/A').replace('_SEQ', '').replace('_BP', '')
            seq_names.append(name)
            val = cal.get(self._value_key, 0)
            values.append(val)

            is_valid = real_idx in valid_indices
            is_current = current_short and current_short == name
            is_outlier = cal.get('is_outlier', False)

            if is_current:
                colors.append('#27AE60')
                edge_colors.append('#1E8449')
            elif is_outlier or not is_valid:
                colors.append('#E74C3C')
                edge_colors.append('#C0392B')
            else:
                colors.append('#5DADE2')
                edge_colors.append('#2E86AB')

        x = range(len(seq_names))
        self._bars = list(ax.bar(x, values, color=colors, edgecolor=edge_colors,
                                 linewidth=1, picker=True))

        # Mitjana de vàlids
        valid_vals = [v for i, v in enumerate(values)
                     if (self._offset + i) in valid_indices and v > 0]
        if valid_vals:
            mean_val = np.mean(valid_vals)
            std_val = np.std(valid_vals) if len(valid_vals) > 1 else 0
            ax.axhline(mean_val, color='#27AE60', linestyle='-', linewidth=2, zorder=5)
            if std_val > 0:
                ax.axhspan(mean_val - std_val, mean_val + std_val,
                          alpha=0.2, color='#27AE60', zorder=1)
            ax.text(len(x) - 0.3, mean_val, f'{mean_val:.2f}' if mean_val < 10 else f'{mean_val:.0f}',
                   fontsize=8, color='#1E8449', va='center', fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(seq_names, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel(self._ylabel, fontsize=8)
        ax.tick_params(axis='y', labelsize=7)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xlim(-0.5, len(x) - 0.5)
        ax.set_ylim(bottom=0)

        self.figure.tight_layout()
        self.canvas.draw()

    def clear(self):
        self.figure.clear()
        self._bars = []
        self._bar_real_indices = []
        self._selected_idx = -1
        self.canvas.draw()
