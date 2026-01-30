"""
HPSEC Suite - Export Panel
===========================

Panel para la fase 5: Exportación de resultados.
"""

from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout,
    QCheckBox, QLineEdit, QFileDialog, QGroupBox, QFrame,
    QProgressBar, QMessageBox
)
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QFont

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from hpsec_process import write_consolidated_excel


class ExportWorker(QThread):
    """Worker thread para exportación asíncrona."""
    progress = Signal(int, str)
    finished = Signal(int, list)  # n_exported, errors
    error = Signal(str)

    def __init__(self, main_window, output_path, only_selected=True):
        super().__init__()
        self.main_window = main_window
        self.output_path = output_path
        self.only_selected = only_selected

    def run(self):
        try:
            # TODO: Implementar exportación real
            # Por ahora, simulamos
            import time
            for i in range(10):
                self.progress.emit((i + 1) * 10, f"Exportant {i + 1}/10...")
                time.sleep(0.2)

            self.finished.emit(10, [])

        except Exception as e:
            self.error.emit(str(e))


class ExportPanel(QWidget):
    """Panel de exportación."""

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.worker = None

        self._setup_ui()

    def _setup_ui(self):
        """Configura la interfaz."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        # Título
        title = QLabel("Exportar Resultats")
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        layout.addWidget(title)

        # Opciones de exportación
        options_group = QGroupBox("Opcions d'Exportació")
        options_layout = QVBoxLayout(options_group)

        self.excel_check = QCheckBox("Generar Excels consolidats")
        self.excel_check.setChecked(True)
        options_layout.addWidget(self.excel_check)

        self.pdf_check = QCheckBox("Generar informe PDF")
        self.pdf_check.setChecked(False)
        self.pdf_check.setEnabled(False)  # TODO
        options_layout.addWidget(self.pdf_check)

        self.only_selected_check = QCheckBox("Només rèpliques seleccionades")
        self.only_selected_check.setChecked(True)
        options_layout.addWidget(self.only_selected_check)

        layout.addWidget(options_group)

        # Carpeta de salida
        output_group = QGroupBox("Carpeta de Sortida")
        output_layout = QHBoxLayout(output_group)

        self.output_path_input = QLineEdit()
        self.output_path_input.setPlaceholderText("Selecciona carpeta de sortida...")
        output_layout.addWidget(self.output_path_input, 1)

        browse_btn = QPushButton("Examinar...")
        browse_btn.clicked.connect(self._browse_output)
        output_layout.addWidget(browse_btn)

        layout.addWidget(output_group)

        # Resumen
        self.summary_frame = QFrame()
        self.summary_frame.setProperty("card", True)
        summary_layout = QVBoxLayout(self.summary_frame)

        self.summary_label = QLabel("Selecciona una seqüència per veure el resum.")
        summary_layout.addWidget(self.summary_label)

        layout.addWidget(self.summary_frame)

        layout.addStretch()

        # Botones
        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch()

        self.export_btn = QPushButton("Exportar")
        self.export_btn.clicked.connect(self._run_export)
        buttons_layout.addWidget(self.export_btn)

        layout.addLayout(buttons_layout)

    def showEvent(self, event):
        """Se llama cuando el panel se hace visible."""
        super().showEvent(event)
        self._update_summary()

    def _update_summary(self):
        """Actualiza el resumen."""
        processed_data = self.main_window.processed_data
        review_data = self.main_window.review_data

        if not processed_data:
            return

        # Calcular estadísticas
        all_samples = (
            processed_data.get("samples", []) +
            processed_data.get("khp_samples", []) +
            processed_data.get("control_samples", [])
        )

        n_samples = len(set(s.get("name") for s in all_samples))
        n_replicas = len(all_samples)

        # Default output path
        seq_path = self.main_window.seq_path
        if seq_path and not self.output_path_input.text():
            default_output = str(Path(seq_path) / "Dades_Consolidades")
            self.output_path_input.setText(default_output)

        self.summary_label.setText(
            f"<b>Mostres a exportar:</b> {n_samples}<br>"
            f"<b>Rèpliques totals:</b> {n_replicas}<br><br>"
            f"Els fitxers es guardaran a la carpeta de sortida seleccionada."
        )

    def _browse_output(self):
        """Abre diálogo para seleccionar carpeta de salida."""
        path = QFileDialog.getExistingDirectory(
            self,
            "Selecciona carpeta de sortida",
            self.output_path_input.text()
        )
        if path:
            self.output_path_input.setText(path)

    def _run_export(self):
        """Ejecuta la exportación."""
        output_path = self.output_path_input.text()
        if not output_path:
            QMessageBox.warning(self, "Avís", "Selecciona una carpeta de sortida.")
            return

        # Crear carpeta si no existe
        Path(output_path).mkdir(parents=True, exist_ok=True)

        self.export_btn.setEnabled(False)
        self.main_window.show_progress(0)

        self.worker = ExportWorker(
            self.main_window,
            output_path,
            only_selected=self.only_selected_check.isChecked()
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_progress(self, pct, msg):
        self.main_window.show_progress(pct)
        self.main_window.set_status(msg)

    def _on_finished(self, n_exported, errors):
        self.main_window.show_progress(-1)
        self.export_btn.setEnabled(True)

        msg = f"Exportació completada!\n\nFitxers generats: {n_exported}"
        if errors:
            msg += f"\n\nErrors ({len(errors)}):\n" + "\n".join(errors[:5])

        QMessageBox.information(self, "Exportació Completada", msg)
        self.main_window.set_status("Exportació completada", 5000)

    def _on_error(self, error_msg):
        self.main_window.show_progress(-1)
        self.export_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", f"Error durant l'exportació:\n{error_msg}")
