"""
HPSEC Suite - Calibrate Panel
==============================

Panel para la fase 2: Calibración KHP.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QGroupBox,
    QGridLayout, QFrame, QHBoxLayout
)
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QFont

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from hpsec_calibrate import calibrate_from_import


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
            self.error.emit(str(e))


class CalibratePanel(QWidget):
    """Panel de calibración."""

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
        layout.setSpacing(16)

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

        # Resultados
        self.results_frame = QFrame()
        self.results_frame.setProperty("card", True)
        self.results_frame.setVisible(False)

        results_layout = QVBoxLayout(self.results_frame)

        results_title = QLabel("Resultat de la Calibració")
        results_title.setFont(QFont("Segoe UI", 12, QFont.Bold))
        results_layout.addWidget(results_title)

        # Grid de resultados
        grid = QGridLayout()
        self.result_labels = {}

        items = [
            ("mode", "Mode:"),
            ("khp_source", "Font KHP:"),
            ("factor_direct", "Factor Direct:"),
            ("factor_uib", "Factor UIB:"),
            ("shift_uib", "Shift UIB:"),
            ("khp_conc", "Concentració KHP:"),
        ]

        for i, (key, label) in enumerate(items):
            lbl = QLabel(label)
            lbl.setStyleSheet("font-weight: bold;")
            grid.addWidget(lbl, i, 0)

            val = QLabel("-")
            self.result_labels[key] = val
            grid.addWidget(val, i, 1)

        results_layout.addLayout(grid)
        layout.addWidget(self.results_frame)

        layout.addStretch()

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
        self.results_frame.setVisible(True)
        self.result_labels["mode"].setText(result.get("mode", "-"))
        self.result_labels["khp_source"].setText(result.get("khp_source", "local"))
        self.result_labels["factor_direct"].setText(f"{result.get('factor_direct', 0):.6f}")
        self.result_labels["factor_uib"].setText(f"{result.get('factor_uib', 0):.6f}")
        self.result_labels["shift_uib"].setText(f"{result.get('shift_uib', 0):.3f} min")
        self.result_labels["khp_conc"].setText(f"{result.get('khp_conc', 0):.1f} ppm")

        self.next_btn.setEnabled(True)
        self.main_window.enable_tab(2)
        self.main_window.set_status("Calibració completada", 5000)

    def _on_error(self, error_msg):
        self.main_window.show_progress(-1)
        self.calibrate_btn.setEnabled(True)
        # Continuar con defaults
        self.calibration_data = {"success": False, "factor_direct": 0, "factor_uib": 0, "shift_uib": 0}
        self.main_window.calibration_data = self.calibration_data
        self.results_frame.setVisible(True)
        self.result_labels["mode"].setText("Defaults (sense KHP)")
        self.next_btn.setEnabled(True)
        self.main_window.enable_tab(2)

    def _go_next(self):
        self.main_window.go_to_tab(2)
