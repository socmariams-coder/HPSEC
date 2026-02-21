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
    """Widget que mostra gràfics de KHP per rèplica amb DOC (fila 1) i DAD 254nm (fila 2)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

        self.setMinimumHeight(350)

    def plot_replicas(self, replicas_direct, replicas_uib=None):
        """
        Grafica rèpliques: fila 1 = DOC (Direct+UIB superposats), fila 2 = 254nm.

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

        # Check if any replica has 254nm data
        has_any_254 = any(
            rep.get('t_dad') is not None and rep.get('y_dad_254') is not None
            for rep in replicas_direct
        )

        # Grid: 2 rows (DOC + 254nm) x N cols, or 1 row if no 254nm
        n_rows = 2 if has_any_254 else 1
        n_cols = n_replicas

        # Row 1: DOC plots (Direct + UIB overlaid)
        for i, rep_direct in enumerate(replicas_direct):
            ax = self.figure.add_subplot(n_rows, n_cols, i + 1)
            rep_uib = replicas_uib[i] if has_uib and i < len(replicas_uib) else None
            self._plot_doc(ax, rep_direct, rep_uib, i + 1)

        # Row 2: 254nm plots
        if has_any_254:
            for i, rep in enumerate(replicas_direct):
                ax = self.figure.add_subplot(n_rows, n_cols, n_cols + i + 1)
                self._plot_254(ax, rep, i + 1)

        self.figure.tight_layout()
        self.canvas.draw()

    def _plot_doc(self, ax, rep_direct, rep_uib, replica_num):
        """Plot DOC subplot: Direct (blau) + UIB (verd) superposats, amb àrea ombrejada i fit."""
        t_doc = rep_direct.get('t_doc')
        y_doc = rep_direct.get('y_doc')
        area = rep_direct.get('area', 0)
        snr = rep_direct.get('snr', 0)

        if t_doc is None or y_doc is None:
            self._plot_doc_fallback(ax, rep_direct, replica_num)
            return

        t_doc = np.asarray(t_doc)
        y_doc = np.asarray(y_doc)

        # --- Direct DOC curve ---
        color_direct = '#2E86AB'
        ax.plot(t_doc, y_doc, color=color_direct, linewidth=1.2)

        # Shaded integrated area (Direct)
        li = rep_direct.get('peak_left_idx', 0)
        ri = rep_direct.get('peak_right_idx', len(t_doc) - 1)
        if 0 <= li < ri < len(t_doc):
            ax.fill_between(t_doc[li:ri+1], 0, y_doc[li:ri+1],
                           alpha=0.15, color=color_direct)

        # Cim irregular: show original if repaired (jagged/batman)
        if rep_direct.get('irregular_top_repaired', rep_direct.get('batman_repaired')) and rep_direct.get('y_doc_repaired') is not None:
            y_repaired = np.asarray(rep_direct['y_doc_repaired'])
            if len(y_repaired) == len(t_doc):
                ax.plot(t_doc, y_repaired, color='#E74C3C', linewidth=0.8,
                       alpha=0.4, linestyle=':')

        # Bigaussian fit (dashed over DOC)
        self._plot_bigaussian_fit(ax, rep_direct)

        # --- UIB DOC curve (overlaid in green) ---
        area_uib = 0
        has_uib_curve = False
        if rep_uib is not None:
            t_uib = rep_uib.get('t_doc')
            y_uib = rep_uib.get('y_doc')
            area_uib = rep_uib.get('area', 0)
            if t_uib is not None and y_uib is not None:
                has_uib_curve = True
                t_uib = np.asarray(t_uib)
                y_uib = np.asarray(y_uib)
                color_uib = '#27AE60'
                ax.plot(t_uib, y_uib, color=color_uib, linewidth=1.2)
                li_u = rep_uib.get('peak_left_idx', 0)
                ri_u = rep_uib.get('peak_right_idx', len(t_uib) - 1)
                if 0 <= li_u < ri_u < len(t_uib):
                    ax.fill_between(t_uib[li_u:ri_u+1], 0, y_uib[li_u:ri_u+1],
                                   alpha=0.10, color=color_uib)

        # Axes formatting
        ax.set_xlabel('Temps (min)', fontsize=7)
        ax.set_ylabel('DOC (mAU)', fontsize=7)
        ax.tick_params(axis='both', labelsize=6)
        ax.grid(True, alpha=0.3)

        # --- Title ---
        ax.set_title(f"R{replica_num} DOC", fontsize=8, fontweight='bold')

        # --- Info box: llegenda + mètriques unificades ---
        self._draw_info_box(ax, rep_direct, area, snr, has_uib_curve, area_uib)

        # --- Timeout zones ---
        self._draw_timeout_zones(ax, rep_direct)

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
                    ax.set_title(f"R{replica_num} DOC: A={area:.0f} [fit]",
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
        ax.set_title(f"R{replica_num} DOC", fontsize=8)
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
            r2 = bigauss.get('r2', 0)
            if amp > 0 and mu > 0 and sigma_l > 0 and sigma_r > 0:
                t_fit = np.linspace(mu - 4*sigma_l, mu + 4*sigma_r, 200)
                y_fit = bigaussian(t_fit, amp, mu, sigma_l, sigma_r, 0)
                fit_color = '#27AE60' if bigauss.get('status') == 'VALID' else '#F39C12'
                ax.plot(t_fit, y_fit, color=fit_color, linewidth=1.3, linestyle='--',
                       alpha=0.8, label='Fit BiGauss')
        except Exception:
            pass

    def _draw_info_box(self, ax, rep, area, snr, has_uib, area_uib):
        """Bloc únic amb llegenda + mètriques a upper right."""
        lines = []

        # Senyals
        lines.append(f"\u2501 Direct  A={area:.0f}")
        if has_uib:
            lines.append(f"\u2501 UIB     A={area_uib:.0f}")

        # SNR
        if snr > 0:
            lines.append(f"SNR={snr:.0f}")

        # Bigaussian R²
        bg = rep.get('bigaussian_doc', {})
        if bg:
            r2 = bg.get('r2', 0)
            status = bg.get('status', '')
            if r2 > 0:
                tag = 'OK' if status == 'VALID' else ('!!' if status in ('CHECK', 'INVALID') else '')
                lines.append(f"R\u00B2={r2:.4f} {tag}")

        # Anomalies — cim irregular (jagged/batman)
        if rep.get('has_irregular_top', rep.get('has_batman')):
            lines.append("Pic_J" + (" (rep)" if rep.get('irregular_top_repaired', rep.get('batman_repaired')) else " !!"))
        qs = rep.get('quality_score', 0)
        if qs > 0:
            lines.append(f"QS={qs}")

        text = "\n".join(lines)
        props = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='#BDC3C7')
        ax.text(0.98, 0.97, text, transform=ax.transAxes, fontsize=6,
               verticalalignment='top', horizontalalignment='right',
               bbox=props, family='monospace')

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

    def _plot_254(self, ax, rep, replica_num):
        """Plot 254nm subplot with shaded area."""
        t_dad = rep.get('t_dad')
        y_dad = rep.get('y_dad_254')
        a254_area = rep.get('a254_area', 0)

        if t_dad is None or y_dad is None:
            ax.text(0.5, 0.5, "Sense 254nm", ha='center', va='center',
                   fontsize=9, color='gray')
            ax.set_title(f"R{replica_num} 254nm", fontsize=8)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            return

        t_dad = np.asarray(t_dad)
        y_dad = np.asarray(y_dad)
        color_254 = '#E67E22'

        ax.plot(t_dad, y_dad, color=color_254, linewidth=1.0, label=f'254nm (A={a254_area:.1f})')

        # Shaded integrated area for 254nm
        dad_peak = rep.get('dad_peak_info', {})
        if dad_peak and dad_peak.get('valid'):
            d_li = dad_peak.get('left_idx', 0)
            d_ri = dad_peak.get('right_idx', len(t_dad) - 1)
            if 0 <= d_li < d_ri < len(t_dad):
                ax.fill_between(t_dad[d_li:d_ri+1], 0, y_dad[d_li:d_ri+1],
                               alpha=0.2, color=color_254)

        ax.set_xlabel('Temps (min)', fontsize=7)
        ax.set_ylabel('254nm (mAU)', fontsize=7)
        ax.tick_params(axis='both', labelsize=6)
        ax.grid(True, alpha=0.3)

        # Title with key info (G5)
        bg_254 = rep.get('bigaussian_254', {})
        r2_254 = bg_254.get('r2', 0) if bg_254 else 0
        status_254 = bg_254.get('status', '') if bg_254 else ''
        status_icon = 'OK' if status_254 == 'VALID' else ('!!' if status_254 in ('CHECK', 'INVALID') else '')
        title_parts = [f"R{replica_num} 254nm: A={a254_area:.1f}"]
        if r2_254 > 0:
            title_parts.append(f"R\u00B2={r2_254:.3f}")
        if status_icon:
            title_parts.append(status_icon)
        ax.set_title("  ".join(title_parts), fontsize=8, fontweight='bold')

    def clear(self):
        self.figure.clear()
        self.canvas.draw()


class CalibrationLineWidget(QWidget):
    """Widget que mostra la recta de calibració amb punts de SEQs recents."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(5, 2.5), dpi=100)
        self.canvas = FigureCanvas(self.figure)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

        self.setMinimumHeight(200)
        self.setMaximumHeight(280)

    def plot_calibration(self, qc_history, current_seq_name, rf_mass_cal=682,
                         warning_pct=5.0, fail_pct=10.0, n_context=2,
                         rf_mass_cal_bp=None, current_mode='column',
                         intercept_col=0, intercept_bp=0):
        """
        Gràfic de recta de calibració amb punts de SEQs recents.

        Args:
            qc_history: Llista d'entrades del QC_History.json
            current_seq_name: Nom de la SEQ actual
            rf_mass_cal: Pendent de la recta COLUMN (rf_mass de calibració)
            warning_pct: % tolerància warning
            fail_pct: % tolerància fail
            n_context: Nombre de SEQs a mostrar abans/després de l'actual
            rf_mass_cal_bp: Pendent de la recta BP (opcional)
            current_mode: Mode actual ('column' o 'bp') per destacar la recta activa
        """
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # Recollir punts de dades per autoajustar eixos
        data_x = []
        data_y = []

        # Punts de QC history (1 punt per SEQ = mitjana de rèpliques)
        entries_to_plot = []
        current_short = ""
        if qc_history:
            current_short = current_seq_name.replace('_SEQ', '').replace('_BP', '') if current_seq_name else ""

            # Agrupar entrades per SEQ i fer mitjana
            from collections import defaultdict
            seq_groups = defaultdict(list)
            for entry in qc_history:
                seq_key = entry.get('seq_name', '')
                seq_groups[seq_key].append(entry)

            averaged_entries = []
            for seq_key, group in seq_groups.items():
                areas = [e.get('measured', {}).get('area', 0) for e in group if e.get('measured', {}).get('area', 0) > 0]
                if not areas:
                    continue
                ref = group[-1]
                avg_entry = dict(ref)
                avg_measured = dict(ref.get('measured', {}))
                avg_measured['area'] = float(np.mean(areas))
                avg_entry['measured'] = avg_measured
                averaged_entries.append(avg_entry)

            entries = sorted(averaged_entries, key=lambda e: e.get('seq_name', ''))

            current_idx = None
            for i, entry in enumerate(entries):
                name = entry.get('seq_name', '').replace('_SEQ', '').replace('_BP', '')
                if name == current_short:
                    current_idx = i
                    break

            if current_idx is not None:
                start = max(0, current_idx - n_context)
                end = min(len(entries), current_idx + n_context + 1)
                entries_to_plot = entries[start:end]
            else:
                entries_to_plot = entries[-5:] if len(entries) > 5 else entries

            # Recollir coordenades per autoajustar
            for entry in entries_to_plot:
                measured = entry.get('measured', {})
                area = measured.get('area', 0)
                conc = entry.get('khp_conc_ppm', 0)
                volume = entry.get('volume_uL', 0)
                if area > 0 and conc > 0 and volume > 0:
                    ug_doc = conc * volume / 1000
                    data_x.append(ug_doc)
                    data_y.append(area)

        # Autoajustar eixos basant-se en les dades
        if data_x:
            margin = 0.3
            x_min = min(data_x) * (1 - margin)
            x_max = max(data_x) * (1 + margin)
        else:
            x_min, x_max = 0, 5

        x_min = max(0, x_min)
        x_line = np.linspace(x_min, x_max, 100)

        # Y max: consider line values at x_max AND data points, start from 0
        all_y_vals = list(data_y)
        all_y_vals.append(rf_mass_cal * x_max + intercept_col)
        if rf_mass_cal_bp and rf_mass_cal_bp > 0:
            all_y_vals.append(rf_mass_cal_bp * x_max + intercept_bp)
        y_max = max(all_y_vals) * 1.1 if all_y_vals else rf_mass_cal * 5 * 1.1

        # Recta COLUMN
        is_column_active = 'column' in current_mode.lower() or 'dual' in current_mode.lower()
        y_line = rf_mass_cal * x_line + intercept_col
        lw_col = 2.0 if is_column_active else 1.0
        alpha_col = 1.0 if is_column_active else 0.5
        col_label = f'Column rf={rf_mass_cal:.0f}'
        if intercept_col:
            col_label += f' +{intercept_col:.0f}' if intercept_col > 0 else f' {intercept_col:.0f}'
        ax.plot(x_line, y_line, color='#2C3E50', linewidth=lw_col, alpha=alpha_col,
                label=col_label)

        # Bandes de tolerància (només per la recta activa)
        active_rf = rf_mass_cal
        active_intercept = intercept_col
        if rf_mass_cal_bp and 'bp' in current_mode.lower():
            active_rf = rf_mass_cal_bp
            active_intercept = intercept_bp

        y_active = active_rf * x_line + active_intercept
        y_warning_upper = y_active * (1 + warning_pct / 100)
        y_warning_lower = y_active * (1 - warning_pct / 100)
        y_fail_upper = y_active * (1 + fail_pct / 100)
        y_fail_lower = y_active * (1 - fail_pct / 100)

        ax.fill_between(x_line, y_fail_lower, y_fail_upper,
                       alpha=0.08, color='#E74C3C', label=f'±{fail_pct:.0f}%')
        ax.fill_between(x_line, y_warning_lower, y_warning_upper,
                       alpha=0.12, color='#F39C12', label=f'±{warning_pct:.0f}%')

        # Recta BP (si disponible)
        if rf_mass_cal_bp and rf_mass_cal_bp > 0:
            is_bp_active = 'bp' in current_mode.lower()
            y_line_bp = rf_mass_cal_bp * x_line + intercept_bp
            lw_bp = 2.0 if is_bp_active else 1.0
            alpha_bp = 1.0 if is_bp_active else 0.5
            bp_label = f'BP rf={rf_mass_cal_bp:.0f}'
            if intercept_bp:
                bp_label += f' +{intercept_bp:.0f}' if intercept_bp > 0 else f' {intercept_bp:.0f}'
            ax.plot(x_line, y_line_bp, color='#8E44AD', linewidth=lw_bp, alpha=alpha_bp,
                    linestyle='--' if not is_bp_active else '-',
                    label=bp_label)

        # Plotar punts
        for entry in entries_to_plot:
            seq_name = entry.get('seq_name', '').replace('_SEQ', '').replace('_BP', '')
            measured = entry.get('measured', {})
            area = measured.get('area', 0)
            conc = entry.get('khp_conc_ppm', 0)
            volume = entry.get('volume_uL', 0)

            if area > 0 and conc > 0 and volume > 0:
                ug_doc = conc * volume / 1000
                is_current = seq_name == current_short
                status = entry.get('qc_result', {}).get('status', 'UNKNOWN')

                if is_current:
                    color = '#27AE60'
                    marker = 's'
                    size = 80
                    zorder = 10
                elif status == 'PASS':
                    color = '#3498DB'
                    marker = 'o'
                    size = 50
                    zorder = 5
                elif status == 'WARNING':
                    color = '#F39C12'
                    marker = '^'
                    size = 60
                    zorder = 6
                else:
                    color = '#E74C3C'
                    marker = 'x'
                    size = 70
                    zorder = 7

                ax.scatter(ug_doc, area, c=color, marker=marker, s=size,
                          zorder=zorder, edgecolors='white', linewidths=0.5)
                ax.annotate(seq_name, (ug_doc, area), fontsize=7,
                           xytext=(3, 3), textcoords='offset points',
                           color=color if is_current else 'gray')

        ax.set_xlabel('µg DOC', fontsize=9)
        ax.set_ylabel('Àrea (mAU·min)', fontsize=9)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, y_max)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=7, framealpha=0.9)

        self.figure.tight_layout()
        self.canvas.draw()

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
            # Restaurar edge color original per les no seleccionades
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
            ax.text(0.5, 0.5, "No hi ha històric",
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

        self.figure.tight_layout()
        self.canvas.draw()

    def clear(self):
        self.figure.clear()
        self._bars = []
        self._bar_real_indices = []
        self._selected_idx = -1
        self.canvas.draw()
