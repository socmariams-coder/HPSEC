"""
HPSEC Suite - Calibrate Panel
==============================

Panel para la fase 2: Calibración KHP.
Muestra gráficos de réplicas, métricas detalladas y comparación histórica.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QGroupBox,
    QGridLayout, QFrame, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QSplitter, QScrollArea, QSizePolicy
)
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QFont, QColor

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from hpsec_calibrate import calibrate_from_import

# Matplotlib imports
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np


class CalibrateWorker(QThread):
    """Worker thread para calibración asíncrona."""
    progress = Signal(int, str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, imported_data, config=None):
        super().__init__()
        self.imported_data = imported_data
        self.config = config

    def run(self):
        try:
            def progress_cb(pct, msg):
                self.progress.emit(int(pct), msg)

            result = calibrate_from_import(
                self.imported_data,
                config=self.config,
                progress_callback=progress_cb
            )
            self.finished.emit(result)

        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n{traceback.format_exc()}")


class KHPGraphWidget(QWidget):
    """Widget que muestra gráficos de KHP."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(8, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

        self.setMinimumHeight(300)

    def plot_khp_data(self, khp_data_list, signal_type="Direct"):
        """
        Grafica múltiples réplicas de KHP.

        Args:
            khp_data_list: Lista de dicts con t_doc, y_doc, peak_info, etc.
            signal_type: "Direct" o "UIB"
        """
        self.figure.clear()

        if not khp_data_list:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "No hi ha dades KHP disponibles",
                   ha='center', va='center', fontsize=12)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            self.canvas.draw()
            return

        ax = self.figure.add_subplot(111)

        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']

        for i, khp in enumerate(khp_data_list):
            t_doc = khp.get('t_doc')
            y_doc = khp.get('y_doc')

            if t_doc is None or y_doc is None:
                continue

            t_doc = np.asarray(t_doc)
            y_doc = np.asarray(y_doc)

            # Color y label
            color = colors[i % len(colors)]
            filename = khp.get('filename', f'R{i+1}')
            area = khp.get('area', 0)
            quality = khp.get('quality_score', 0)

            label = f"{filename} (A={area:.1f}"
            if quality > 0:
                label += f", Q={quality})"
            else:
                label += ")"

            # Línea del cromatograma
            ax.plot(t_doc, y_doc, color=color, linewidth=1.2, label=label, alpha=0.8)

            # Marcar pico principal
            peak_info = khp.get('peak_info', {})
            if peak_info:
                t_max = peak_info.get('t_max', 0)
                y_max = peak_info.get('y_max', 0)
                if t_max > 0 and y_max > 0:
                    ax.plot(t_max, y_max, 'o', color=color, markersize=6)

                # Marcar límites de integración
                t_start = peak_info.get('t_start', 0)
                t_end = peak_info.get('t_end', 0)
                if t_start > 0 and t_end > 0:
                    ax.axvline(t_start, color=color, linestyle='--', alpha=0.3, linewidth=0.8)
                    ax.axvline(t_end, color=color, linestyle='--', alpha=0.3, linewidth=0.8)

            # Indicar anomalías
            if khp.get('has_batman'):
                t_max = peak_info.get('t_max', t_doc[len(t_doc)//2])
                ax.annotate('Batman', xy=(t_max, y_doc.max()),
                           fontsize=8, color='red', ha='center')

            if khp.get('has_timeout'):
                timeout_info = khp.get('timeout_info', {})
                for t_timeout in timeout_info.get('timeout_positions', [])[:2]:
                    ax.axvspan(t_timeout - 0.5, t_timeout + 1.5,
                              alpha=0.15, color='orange', label='_nolegend_')

        ax.set_xlabel('Temps (min)')
        ax.set_ylabel('Senyal DOC (mAU)')
        ax.set_title(f'KHP - Senyal {signal_type}')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

        self.figure.tight_layout()
        self.canvas.draw()

    def clear(self):
        self.figure.clear()
        self.canvas.draw()


class CalibratePanel(QWidget):
    """Panel de calibración con gráficos y métricas detalladas."""

    calibration_completed = Signal(dict)

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.calibration_data = None
        self.worker = None

        self._setup_ui()

    def _setup_ui(self):
        """Configura la interfaz."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)

        # Título
        title = QLabel("Calibració KHP")
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        layout.addWidget(title)

        # Info
        info = QLabel(
            "La calibració analitza les mostres KHP per calcular els factors de correcció."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        # Contenedor principal con scroll
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setSpacing(16)

        # === SECCIÓN: Resumen de Calibración ===
        self.summary_group = QGroupBox("Resum de Calibració")
        self.summary_group.setVisible(False)
        summary_layout = QGridLayout(self.summary_group)

        self.result_labels = {}
        items = [
            ("mode", "Mode:", 0, 0),
            ("khp_source", "Font KHP:", 0, 2),
            ("factor_direct", "Factor Direct:", 1, 0),
            ("factor_uib", "Factor UIB:", 1, 2),
            ("shift_uib", "Shift UIB:", 2, 0),
            ("khp_conc", "Concentració KHP:", 2, 2),
            ("n_replicas", "Rèpliques:", 3, 0),
            ("selected", "Seleccionat:", 3, 2),
        ]

        for key, label_text, row, col in items:
            lbl = QLabel(label_text)
            lbl.setStyleSheet("font-weight: bold;")
            summary_layout.addWidget(lbl, row, col)

            val = QLabel("-")
            self.result_labels[key] = val
            summary_layout.addWidget(val, row, col + 1)

        content_layout.addWidget(self.summary_group)

        # === SECCIÓN: Gráficos de KHP ===
        self.graphs_group = QGroupBox("Gràfics KHP")
        self.graphs_group.setVisible(False)
        graphs_layout = QHBoxLayout(self.graphs_group)

        # Gráfico Direct
        direct_frame = QFrame()
        direct_layout = QVBoxLayout(direct_frame)
        direct_layout.setContentsMargins(0, 0, 0, 0)
        direct_label = QLabel("DOC Direct")
        direct_label.setFont(QFont("Segoe UI", 10, QFont.Bold))
        direct_label.setAlignment(Qt.AlignCenter)
        direct_layout.addWidget(direct_label)
        self.graph_direct = KHPGraphWidget()
        direct_layout.addWidget(self.graph_direct)
        graphs_layout.addWidget(direct_frame)

        # Gráfico UIB
        uib_frame = QFrame()
        uib_layout = QVBoxLayout(uib_frame)
        uib_layout.setContentsMargins(0, 0, 0, 0)
        uib_label = QLabel("DOC UIB")
        uib_label.setFont(QFont("Segoe UI", 10, QFont.Bold))
        uib_label.setAlignment(Qt.AlignCenter)
        uib_layout.addWidget(uib_label)
        self.graph_uib = KHPGraphWidget()
        uib_layout.addWidget(self.graph_uib)
        graphs_layout.addWidget(uib_frame)

        content_layout.addWidget(self.graphs_group)

        # === SECCIÓN: Tabla de Métricas por Réplica ===
        self.metrics_group = QGroupBox("Mètriques per Rèplica")
        self.metrics_group.setVisible(False)
        metrics_layout = QVBoxLayout(self.metrics_group)

        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(10)
        self.metrics_table.setHorizontalHeaderLabels([
            "Rèplica", "Senyal", "Àrea", "Shift (s)", "Simetria",
            "SNR", "Pics", "Qualitat", "Batman", "Timeout"
        ])
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.metrics_table.setAlternatingRowColors(True)
        self.metrics_table.setMaximumHeight(200)
        # Permetre selecció i còpia
        self.metrics_table.setSelectionMode(QTableWidget.ExtendedSelection)
        self.metrics_table.setSelectionBehavior(QTableWidget.SelectItems)
        metrics_layout.addWidget(self.metrics_table)

        content_layout.addWidget(self.metrics_group)

        # === SECCIÓN: Validación y Problemas ===
        self.validation_group = QGroupBox("Validació i Problemes")
        self.validation_group.setVisible(False)
        validation_layout = QVBoxLayout(self.validation_group)

        self.validation_label = QLabel()
        self.validation_label.setWordWrap(True)
        self.validation_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        validation_layout.addWidget(self.validation_label)

        content_layout.addWidget(self.validation_group)

        # === SECCIÓN: Comparación Histórica ===
        self.history_group = QGroupBox("Comparació Històrica")
        self.history_group.setVisible(False)
        history_layout = QVBoxLayout(self.history_group)

        self.history_label = QLabel()
        self.history_label.setWordWrap(True)
        self.history_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        history_layout.addWidget(self.history_label)

        content_layout.addWidget(self.history_group)

        # Spacer
        content_layout.addStretch()

        scroll.setWidget(content_widget)
        layout.addWidget(scroll, 1)

        # Botones
        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch()

        self.calibrate_btn = QPushButton("Calibrar")
        self.calibrate_btn.clicked.connect(self._run_calibrate)
        buttons_layout.addWidget(self.calibrate_btn)

        self.next_btn = QPushButton("Següent →")
        self.next_btn.setEnabled(False)
        self.next_btn.clicked.connect(self._go_next)
        buttons_layout.addWidget(self.next_btn)

        layout.addLayout(buttons_layout)

    def _run_calibrate(self):
        """Ejecuta la calibración."""
        imported_data = self.main_window.imported_data
        if not imported_data:
            return

        self.calibrate_btn.setEnabled(False)
        self.main_window.show_progress(0)

        # Limpiar resultados anteriores
        self.summary_group.setVisible(False)
        self.graphs_group.setVisible(False)
        self.metrics_group.setVisible(False)
        self.validation_group.setVisible(False)
        self.history_group.setVisible(False)

        self.worker = CalibrateWorker(imported_data)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_progress(self, pct, msg):
        self.main_window.show_progress(pct)
        self.main_window.set_status(msg)

    def _on_finished(self, result):
        self.main_window.show_progress(-1)
        self.calibrate_btn.setEnabled(True)

        self.calibration_data = result
        self.main_window.calibration_data = result

        # Mostrar resultados
        self._update_summary(result)
        self._update_graphs(result)
        self._update_metrics_table(result)
        self._update_validation(result)
        self._update_history(result)

        self.next_btn.setEnabled(True)
        self.main_window.enable_tab(2)
        self.main_window.set_status("Calibració completada", 5000)

    def _on_error(self, error_msg):
        self.main_window.show_progress(-1)
        self.calibrate_btn.setEnabled(True)

        # Continuar con defaults
        self.calibration_data = {
            "success": False,
            "factor_direct": 0,
            "factor_uib": 0,
            "shift_uib": 0,
            "errors": [error_msg]
        }
        self.main_window.calibration_data = self.calibration_data

        self.summary_group.setVisible(True)
        self.result_labels["mode"].setText("Defaults (sense KHP)")
        self.result_labels["khp_source"].setText("-")
        self.result_labels["factor_direct"].setText("-")
        self.result_labels["factor_uib"].setText("-")
        self.result_labels["shift_uib"].setText("-")
        self.result_labels["khp_conc"].setText("-")
        self.result_labels["n_replicas"].setText("-")
        self.result_labels["selected"].setText("-")

        # Mostrar error en validación
        self.validation_group.setVisible(True)
        self.validation_label.setText(
            f"<span style='color: red;'><b>Error durant la calibració:</b></span><br>"
            f"<pre>{error_msg}</pre>"
        )

        self.next_btn.setEnabled(True)
        self.main_window.enable_tab(2)

    def _update_summary(self, result):
        """Actualiza el resumen de calibración."""
        self.summary_group.setVisible(True)

        mode = result.get("mode", "-")
        self.result_labels["mode"].setText(mode)
        self.result_labels["khp_source"].setText(result.get("khp_source", "local"))

        # Factor Direct
        factor_direct = result.get("factor_direct", 0)
        if factor_direct > 0:
            self.result_labels["factor_direct"].setText(f"{factor_direct:.6f}")
        else:
            self.result_labels["factor_direct"].setText("-")

        # Factor UIB
        factor_uib = result.get("factor_uib", 0)
        if factor_uib > 0:
            self.result_labels["factor_uib"].setText(f"{factor_uib:.6f}")
        else:
            self.result_labels["factor_uib"].setText("-")

        # Shift UIB
        shift_uib = result.get("shift_uib", 0)
        if shift_uib != 0:
            self.result_labels["shift_uib"].setText(f"{shift_uib:.3f} min ({shift_uib*60:.1f} s)")
        else:
            self.result_labels["shift_uib"].setText("-")

        # Concentración
        khp_conc = result.get("khp_conc", 0)
        if khp_conc > 0:
            self.result_labels["khp_conc"].setText(f"{khp_conc:.1f} ppm")
        else:
            self.result_labels["khp_conc"].setText("-")

        # Número de réplicas y selección
        n_replicas = 0
        selected_info = "-"

        khp_data_main = result.get("khp_data_direct") or result.get("khp_data_uib")
        if khp_data_main and isinstance(khp_data_main, dict):
            n_replicas = khp_data_main.get("n_replicas", 1)
            status = khp_data_main.get("status", "")
            filename = khp_data_main.get("filename", "")
            rsd = khp_data_main.get("rsd", 0)

            if status:
                selected_info = status
            elif filename:
                selected_info = filename

            if rsd > 0:
                selected_info += f" (RSD: {rsd:.1f}%)"

        self.result_labels["n_replicas"].setText(str(n_replicas))
        self.result_labels["selected"].setText(selected_info)

    def _extract_all_replicas(self, khp_data):
        """
        Extrae todas las réplicas de los datos KHP.

        khp_data puede ser:
        - Un dict con 'all_khp_data' o 'replicas' (resultado de select_best_khp)
        - Una lista de réplicas directamente
        - Un dict individual (única réplica)
        """
        if not khp_data:
            return []

        if isinstance(khp_data, list):
            return khp_data

        if isinstance(khp_data, dict):
            # Buscar lista de réplicas en diferentes claves
            replicas = khp_data.get('all_khp_data') or khp_data.get('replicas')
            if replicas and isinstance(replicas, list):
                return replicas
            # Es un dict individual
            return [khp_data]

        return []

    def _update_graphs(self, result):
        """Actualiza los gráficos de KHP."""
        khp_data_direct = result.get("khp_data_direct")
        khp_data_uib = result.get("khp_data_uib")

        has_graphs = False

        # Preparar datos para gráficos - extraer todas las réplicas
        direct_list = self._extract_all_replicas(khp_data_direct)
        uib_list = self._extract_all_replicas(khp_data_uib)

        has_graphs = len(direct_list) > 0 or len(uib_list) > 0

        if has_graphs:
            self.graphs_group.setVisible(True)
            self.graph_direct.plot_khp_data(direct_list, "Direct")
            self.graph_uib.plot_khp_data(uib_list, "UIB")
        else:
            self.graphs_group.setVisible(False)

    def _update_metrics_table(self, result):
        """Actualiza la tabla de métricas por réplica."""
        self.metrics_table.setRowCount(0)

        khp_data_direct = result.get("khp_data_direct")
        khp_data_uib = result.get("khp_data_uib")

        all_data = []

        # Recopilar datos Direct - todas las réplicas
        direct_list = self._extract_all_replicas(khp_data_direct)
        print(f"[DEBUG] direct_list: {len(direct_list)} replicas")
        for d in direct_list:
            d_copy = d.copy()  # No modificar original
            d_copy['_signal'] = 'Direct'
            all_data.append(d_copy)
            print(f"  - {d.get('filename')}: SNR={d.get('snr', 0):.1f}")

        # Recopilar datos UIB - todas las réplicas
        uib_list = self._extract_all_replicas(khp_data_uib)
        print(f"[DEBUG] uib_list: {len(uib_list)} replicas")
        for d in uib_list:
            d_copy = d.copy()
            d_copy['_signal'] = 'UIB'
            all_data.append(d_copy)

        print(f"[DEBUG] all_data total: {len(all_data)}")

        if not all_data:
            self.metrics_group.setVisible(False)
            return

        self.metrics_group.setVisible(True)

        for khp in all_data:
            row = self.metrics_table.rowCount()
            self.metrics_table.insertRow(row)

            # Réplica
            filename = khp.get('filename', '?')
            self.metrics_table.setItem(row, 0, QTableWidgetItem(filename))

            # Señal
            signal = khp.get('_signal', '?')
            self.metrics_table.setItem(row, 1, QTableWidgetItem(signal))

            # Área
            area = khp.get('area', 0)
            self.metrics_table.setItem(row, 2, QTableWidgetItem(f"{area:.1f}"))

            # Shift
            shift_sec = khp.get('shift_sec', khp.get('shift_min', 0) * 60)
            self.metrics_table.setItem(row, 3, QTableWidgetItem(f"{shift_sec:.1f}"))

            # Simetría
            symmetry = khp.get('symmetry', 1.0)
            item_sym = QTableWidgetItem(f"{symmetry:.2f}")
            if symmetry < 0.8 or symmetry > 1.5:
                item_sym.setBackground(QColor(255, 200, 100))
            self.metrics_table.setItem(row, 4, item_sym)

            # SNR
            snr = khp.get('snr', 0)
            item_snr = QTableWidgetItem(f"{snr:.1f}")
            if snr < 10:
                item_snr.setBackground(QColor(255, 200, 100))
            self.metrics_table.setItem(row, 5, item_snr)

            # Número de picos
            n_peaks = khp.get('all_peaks_count', 1)
            item_peaks = QTableWidgetItem(str(n_peaks))
            if n_peaks > 2:
                item_peaks.setBackground(QColor(255, 200, 100))
            self.metrics_table.setItem(row, 6, item_peaks)

            # Quality score
            quality = khp.get('quality_score', 0)
            item_q = QTableWidgetItem(str(quality))
            if quality > 100:
                item_q.setBackground(QColor(255, 150, 150))
            elif quality > 50:
                item_q.setBackground(QColor(255, 200, 100))
            elif quality > 0:
                item_q.setBackground(QColor(255, 255, 150))
            else:
                item_q.setBackground(QColor(150, 255, 150))
            self.metrics_table.setItem(row, 7, item_q)

            # Batman
            has_batman = khp.get('has_batman', False)
            item_bat = QTableWidgetItem("Sí" if has_batman else "No")
            if has_batman:
                item_bat.setBackground(QColor(255, 150, 150))
            self.metrics_table.setItem(row, 8, item_bat)

            # Timeout
            has_timeout = khp.get('has_timeout', False)
            item_to = QTableWidgetItem("Sí" if has_timeout else "No")
            if has_timeout:
                severity = khp.get('timeout_severity', 'OK')
                if severity == 'CRITICAL':
                    item_to.setBackground(QColor(255, 100, 100))
                elif severity == 'WARNING':
                    item_to.setBackground(QColor(255, 200, 100))
                else:
                    item_to.setBackground(QColor(255, 255, 150))
            self.metrics_table.setItem(row, 9, item_to)

    def _update_validation(self, result):
        """Actualiza la sección de validación."""
        errors = result.get("errors", [])

        # Recopilar quality_issues de todos los KHP (todas las réplicas)
        all_issues = []

        for key in ["khp_data_direct", "khp_data_uib"]:
            khp_data = result.get(key)
            replicas = self._extract_all_replicas(khp_data)
            for d in replicas:
                issues = d.get('quality_issues', [])
                all_issues.extend(issues)

        # Deduplicar
        all_issues = list(set(all_issues))

        if not errors and not all_issues:
            self.validation_group.setVisible(False)
            return

        self.validation_group.setVisible(True)

        html = ""

        if errors:
            html += "<span style='color: red;'><b>Errors:</b></span><ul>"
            for e in errors:
                html += f"<li>{e}</li>"
            html += "</ul>"

        if all_issues:
            html += "<span style='color: orange;'><b>Problemes de qualitat:</b></span><ul>"
            for issue in all_issues:
                html += f"<li>{issue}</li>"
            html += "</ul>"

        self.validation_label.setText(html)

    def _update_history(self, result):
        """Actualiza la comparación histórica."""
        # Buscar comparación histórica en los datos
        historical_info = None

        # Primero buscar en el khp_data principal (best selected)
        for key in ["khp_data_direct", "khp_data_uib"]:
            khp_data = result.get(key)
            if khp_data and isinstance(khp_data, dict):
                # Buscar en el nivel superior
                hist = khp_data.get('historical_comparison')
                if hist:
                    historical_info = hist
                    break
                # Buscar en stats
                stats = khp_data.get('stats', {})
                if stats:
                    hist = stats.get('historical_comparison')
                    if hist:
                        historical_info = hist
                        break

        # También buscar en calibration entry si está disponible
        if not historical_info:
            calibration = result.get("calibration")
            if calibration and isinstance(calibration, dict):
                validation = calibration.get("validation", {})
                if validation:
                    hist = validation.get("historical_comparison")
                    if hist:
                        historical_info = hist

        if not historical_info:
            self.history_group.setVisible(False)
            return

        self.history_group.setVisible(True)

        status = historical_info.get("status", "UNKNOWN")
        area_dev = historical_info.get("area_deviation_pct", 0)
        cr_dev = historical_info.get("concentration_ratio_deviation_pct", 0)

        stats = historical_info.get("historical_stats", {})
        n_cals = stats.get("n_calibrations", 0) if stats else 0
        mean_area = stats.get("mean_area", 0) if stats else 0
        std_area = stats.get("std_area", 0) if stats else 0

        # Color según estado
        if status == "OK":
            color = "green"
            icon = "✓"
        elif status == "WARNING":
            color = "orange"
            icon = "⚠"
        elif status == "INVALID":
            color = "red"
            icon = "✗"
        else:
            color = "gray"
            icon = "?"

        html = f"<span style='color: {color}; font-size: 14pt;'><b>{icon} {status}</b></span><br><br>"

        if n_cals > 0:
            html += f"<b>Calibracions comparades:</b> {n_cals}<br>"
            html += f"<b>Àrea mitjana històrica:</b> {mean_area:.1f} ± {std_area:.1f}<br>"
            html += f"<b>Desviació àrea actual:</b> {area_dev:.1f}%<br>"

            if cr_dev > 0:
                html += f"<b>Desviació ratio concentració:</b> {cr_dev:.1f}%<br>"
        else:
            html += "<i>Dades històriques insuficients per comparar.</i>"

        # Añadir issues/warnings del histórico
        issues = historical_info.get("issues", [])
        warnings = historical_info.get("warnings", [])

        if issues:
            html += "<br><span style='color: red;'><b>Problemes:</b></span><ul>"
            for i in issues:
                html += f"<li>{i}</li>"
            html += "</ul>"

        if warnings:
            html += "<br><span style='color: orange;'><b>Avisos:</b></span><ul>"
            for w in warnings:
                html += f"<li>{w}</li>"
            html += "</ul>"

        self.history_label.setText(html)

    def _go_next(self):
        self.main_window.go_to_tab(2)
