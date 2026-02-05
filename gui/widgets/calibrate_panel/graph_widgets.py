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


class KHPReplicaGraphWidget(QWidget):
    """Widget que mostra gràfics de KHP per rèplica amb DOC i DAD 254nm."""

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
        Grafica rèpliques amb DOC i DAD 254nm.

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

        # Configurar subplots: una fila per Direct, una per UIB si existeix
        n_rows = 2 if has_uib else 1
        n_cols = n_replicas

        colors_doc = ['#2E86AB', '#1A5276']  # Blaus per DOC
        colors_dad = ['#E67E22', '#D35400']  # Taronges per DAD

        # Plotar Direct (fila superior)
        for i, rep in enumerate(replicas_direct):
            ax = self.figure.add_subplot(n_rows, n_cols, i + 1)
            self._plot_single_replica(ax, rep, f"R{i+1} Direct", colors_doc[i % 2], colors_dad[i % 2])

        # Plotar UIB (fila inferior) si existeix
        if has_uib:
            for i, rep in enumerate(replicas_uib):
                ax = self.figure.add_subplot(n_rows, n_cols, n_cols + i + 1)
                self._plot_single_replica(ax, rep, f"R{i+1} UIB", colors_doc[i % 2], colors_dad[i % 2])

        self.figure.tight_layout()
        self.canvas.draw()

    def _plot_single_replica(self, ax, rep, title, color_doc, color_dad):
        """Plotar una rèplica amb DOC i DAD 254nm."""
        t_doc = rep.get('t_doc')
        y_doc = rep.get('y_doc')
        area = rep.get('area', 0)
        snr = rep.get('snr', 0)

        if t_doc is None or y_doc is None:
            ax.text(0.5, 0.5, "Sense dades", ha='center', va='center', fontsize=10, color='gray')
            ax.set_title(title, fontsize=9)
            return

        t_doc = np.asarray(t_doc)
        y_doc = np.asarray(y_doc)

        # Plotar DOC
        ax.plot(t_doc, y_doc, color=color_doc, linewidth=1.2, label=f'DOC (A={area:.1f})')

        # Marcar pic principal
        peak_info = rep.get('peak_info', {})
        if peak_info:
            t_max = peak_info.get('t_max', 0)
            y_max = peak_info.get('y_max', 0)
            if t_max > 0 and y_max > 0:
                ax.plot(t_max, y_max, 'o', color=color_doc, markersize=5)

        ax.set_xlabel('Temps (min)', fontsize=8)
        ax.set_ylabel('DOC (mAU)', fontsize=8, color=color_doc)
        ax.tick_params(axis='y', labelcolor=color_doc, labelsize=7)
        ax.tick_params(axis='x', labelsize=7)

        # Plotar DAD 254nm si disponible (eix secundari)
        t_dad = rep.get('t_dad')
        y_dad = rep.get('y_dad_254')
        a254_area = rep.get('a254_area', 0)

        if t_dad is not None and y_dad is not None:
            t_dad = np.asarray(t_dad)
            y_dad = np.asarray(y_dad)
            if len(t_dad) > 0 and len(y_dad) > 0:
                ax2 = ax.twinx()
                ax2.plot(t_dad, y_dad, color=color_dad, linewidth=1.0, linestyle='--',
                        label=f'254nm (A={a254_area:.1f})', alpha=0.8)
                ax2.set_ylabel('254nm (mAU)', fontsize=8, color=color_dad)
                ax2.tick_params(axis='y', labelcolor=color_dad, labelsize=7)

        # Títol amb info
        title_text = f"{title}: A={area:.0f}"
        if snr > 0:
            title_text += f", SNR={snr:.0f}"
        ax.set_title(title_text, fontsize=9, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Indicar anomalies
        if rep.get('has_batman'):
            ax.annotate('⚠ Batman', xy=(0.02, 0.98), xycoords='axes fraction',
                       fontsize=7, color='red', va='top')

        # Marcar timeout amb zona afectada
        if rep.get('has_timeout'):
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

                # Comprovar si afecta el pic principal
                hits_peak = t_peak > 0 and affected_start <= t_peak <= affected_end

                if hits_peak:
                    affects_main_peak = True
                    color = '#E74C3C'  # Vermell - AFECTA PIC
                    alpha = 0.35
                else:
                    color = '#F39C12'  # Taronja - no afecta pic
                    alpha = 0.2

                # Zona afectada (fons)
                ax.axvspan(affected_start, affected_end, alpha=alpha, color=color, zorder=0)
                # Línia vertical al punt exacte del timeout
                ax.axvline(t_start, color=color, linestyle='--', linewidth=1.5, alpha=0.8)

            # Anotació amb temps i warning si afecta pic
            if timeouts_list:
                first_to = timeouts_list[0]
                t_label = first_to.get('t_start_min', 0)
                if affects_main_peak:
                    ax.annotate(f'⚠ TO@{t_label:.1f} PIC!', xy=(0.02, 0.88), xycoords='axes fraction',
                               fontsize=7, color='#C0392B', va='top', fontweight='bold')
                else:
                    ax.annotate(f'TO@{t_label:.1f}', xy=(0.02, 0.88), xycoords='axes fraction',
                               fontsize=7, color='#E67E22', va='top')

    def clear(self):
        self.figure.clear()
        self.canvas.draw()


class HistoryBarWidget(QWidget):
    """Widget compacte per gràfic de barres històric de KHP."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(5, 2.2), dpi=100)
        self.canvas = FigureCanvas(self.figure)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

        self.setMinimumHeight(160)
        self.setMaximumHeight(200)
        self.history_data = []

    def plot_history(self, history_list, current_seq_name, valid_indices=None):
        """
        Gràfic de barres compacte amb últimes calibracions.

        Args:
            history_list: Llista de calibracions (ja filtrades i ordenades)
            current_seq_name: Nom de la SEQ actual per marcar-la
            valid_indices: Set d'índexs de calibracions vàlides (no outliers)
        """
        import re

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self.history_data = history_list

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
        offset = len(history_list) - len(display_cals)

        current_short = current_seq_name.replace('_SEQ', '').replace('_BP', '') if current_seq_name else ""

        seq_names = []
        areas = []
        colors = []
        edge_colors = []

        for i, cal in enumerate(display_cals):
            real_idx = offset + i
            name = cal.get('seq_name', 'N/A').replace('_SEQ', '').replace('_BP', '')
            seq_names.append(name)
            area = cal.get('area', 0)
            areas.append(area)

            is_valid = real_idx in valid_indices
            is_current = current_short and current_short == name
            is_outlier = cal.get('is_outlier', False)

            # Colors nets sense patrons
            if is_current:
                colors.append('#27AE60')  # Verd per actual
                edge_colors.append('#1E8449')
            elif is_outlier or not is_valid:
                colors.append('#E74C3C')  # Vermell NOMÉS per outliers
                edge_colors.append('#C0392B')
            else:
                colors.append('#5DADE2')  # Blau per vàlids
                edge_colors.append('#2E86AB')

        # Barres netes sense patrons
        x = range(len(seq_names))
        bars = ax.bar(x, areas, color=colors, edgecolor=edge_colors, linewidth=1)

        # Mitjana de vàlids (només dels que es mostren)
        valid_areas = [a for i, a in enumerate(areas) if (offset + i) in valid_indices and a > 0]
        if valid_areas:
            mean_area = np.mean(valid_areas)
            std_area = np.std(valid_areas) if len(valid_areas) > 1 else 0
            ax.axhline(mean_area, color='#27AE60', linestyle='-', linewidth=2, zorder=5)
            if std_area > 0:
                ax.axhspan(mean_area - std_area, mean_area + std_area,
                          alpha=0.2, color='#27AE60', zorder=1)
            ax.text(len(x) - 0.3, mean_area, f'{mean_area:.0f}',
                   fontsize=8, color='#1E8449', va='center', fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(seq_names, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel("Àrea", fontsize=8)
        ax.tick_params(axis='y', labelsize=7)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xlim(-0.5, len(x) - 0.5)

        self.figure.tight_layout()
        self.canvas.draw()

    def clear(self):
        self.figure.clear()
        self.canvas.draw()
