"""
HPSEC Suite - Analyze Panel
============================

Panel per la fase 3: Anàlisi de mostres (detecció anomalies, selecció rèpliques).
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QGroupBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QHBoxLayout, QFrame
)
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QFont

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from hpsec_analyze import analyze_sequence, save_analysis_result


class AnalyzeWorker(QThread):
    """Worker thread per anàlisi asíncrona."""
    progress = Signal(str, int)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, imported_data, calibration_data, config=None):
        super().__init__()
        self.imported_data = imported_data
        self.calibration_data = calibration_data
        self.config = config

    def run(self):
        try:
            def progress_cb(msg, pct):
                self.progress.emit(msg, int(pct))

            result = analyze_sequence(
                self.imported_data,
                self.calibration_data,
                config=self.config,
                progress_callback=progress_cb
            )
            self.finished.emit(result)

        except Exception as e:
            self.error.emit(str(e))


class AnalyzePanel(QWidget):
    """Panel d'anàlisi de mostres."""

    analyze_completed = Signal(dict)

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.processed_data = None
        self.worker = None

        self._setup_ui()

    def _setup_ui(self):
        """Configura la interfaz."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)  # Marges reduïts per més espai
        layout.setSpacing(16)

        # Título
        title = QLabel("Anàlisi de Mostres")
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        layout.addWidget(title)

        info = QLabel(
            "L'anàlisi detecta anomalies i calcula àrees per cada mostra."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        # Tabla de resultados
        self.results_frame = QFrame()
        self.results_frame.setProperty("card", True)
        self.results_frame.setVisible(False)

        results_layout = QVBoxLayout(self.results_frame)

        self.samples_table = QTableWidget()
        self.samples_table.setColumnCount(6)
        self.samples_table.setHorizontalHeaderLabels([
            "Mostra", "Rep", "Status", "Anomalies", "R²", "Àrea"
        ])
        self.samples_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.samples_table.setAlternatingRowColors(True)
        results_layout.addWidget(self.samples_table)

        layout.addWidget(self.results_frame)
        layout.addStretch()

        # Botones
        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch()

        self.analyze_btn = QPushButton("Analitzar")
        self.analyze_btn.clicked.connect(self._run_analyze)
        buttons_layout.addWidget(self.analyze_btn)

        self.next_btn = QPushButton("Següent →")
        self.next_btn.setEnabled(False)
        self.next_btn.clicked.connect(self._go_next)
        buttons_layout.addWidget(self.next_btn)

        layout.addLayout(buttons_layout)

    def _run_analyze(self):
        """Ejecuta el procesamiento."""
        imported_data = self.main_window.imported_data
        calibration_data = self.main_window.calibration_data

        if not imported_data:
            return

        self.analyze_btn.setEnabled(False)
        self.main_window.show_progress(0)

        self.worker = AnalyzeWorker(imported_data, calibration_data)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_progress(self, msg, pct):
        self.main_window.show_progress(pct)
        self.main_window.set_status(msg)

    def _on_finished(self, result):
        self.main_window.show_progress(-1)
        self.analyze_btn.setEnabled(True)

        self.processed_data = result
        self.main_window.processed_data = result

        # Guardar resultat a JSON
        if result and result.get('success'):
            save_analysis_result(result)

        # Mostrar resultados
        self.results_frame.setVisible(True)
        self.samples_table.setRowCount(0)

        all_samples = (
            result.get("samples", []) +
            result.get("khp_samples", []) +
            result.get("control_samples", [])
        )

        for sample in all_samples:
            row = self.samples_table.rowCount()
            self.samples_table.insertRow(row)

            self.samples_table.setItem(row, 0, QTableWidgetItem(sample.get("name", "-")))
            self.samples_table.setItem(row, 1, QTableWidgetItem(str(sample.get("replica", "-"))))

            status = "OK" if sample.get("processed") and not sample.get("anomalies") else "CHECK"
            self.samples_table.setItem(row, 2, QTableWidgetItem(status))

            anomalies = ", ".join(sample.get("anomalies", [])) or "-"
            self.samples_table.setItem(row, 3, QTableWidgetItem(anomalies[:30]))

            r2 = sample.get("peak_info", {}).get("r2", 0)
            self.samples_table.setItem(row, 4, QTableWidgetItem(f"{r2:.4f}"))

            area = sample.get("peak_info", {}).get("area", 0)
            self.samples_table.setItem(row, 5, QTableWidgetItem(f"{area:.1f}"))

        self.next_btn.setEnabled(True)
        self.main_window.set_status("Anàlisi completada", 5000)

        # Emetre senyal per notificar al wizard
        self.analyze_completed.emit(result)

    def _on_error(self, error_msg):
        self.main_window.show_progress(-1)
        self.analyze_btn.setEnabled(True)
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.critical(self, "Error", f"Error durant el processament:\n{error_msg}")

    def _go_next(self):
        self.main_window.go_to_tab(4)  # Revisar
