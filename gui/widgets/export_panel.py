"""
HPSEC Suite - Export Panel
===========================

Panel per la fase 5: Exportació de resultats.

Funcionalitats:
- Generació d'Excels finals (un per mostra)
- Excel resum amb totes les mostres
- Opcions de format i contingut
"""

from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout,
    QCheckBox, QLineEdit, QFileDialog, QGroupBox, QFrame,
    QProgressBar, QMessageBox, QGridLayout
)
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QFont

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from hpsec_export import export_sequence, generate_summary_excel, DEFAULT_EXPORT_CONFIG
from gui.widgets.styles import (
    PANEL_MARGINS, PANEL_SPACING, STYLE_PLACEHOLDER, STYLE_LABEL_SECONDARY,
    create_title_font, apply_panel_layout, create_empty_state_widget
)


class ExportWorker(QThread):
    """Worker thread per exportació asíncrona."""
    progress = Signal(int, str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, samples_grouped, output_path, calibration_data, mode, options):
        super().__init__()
        self.samples_grouped = samples_grouped
        self.output_path = output_path
        self.calibration_data = calibration_data
        self.mode = mode
        self.options = options

    def run(self):
        try:
            results = {"excel_files": None, "summary": None, "errors": []}

            # Callback per progrés
            def progress_cb(pct, msg):
                self.progress.emit(pct, msg)

            # Exportar Excels individuals
            if self.options.get("individual_excels", True):
                self.progress.emit(0, "Exportant fitxers individuals...")
                excel_result = export_sequence(
                    self.samples_grouped,
                    self.output_path,
                    self.calibration_data,
                    self.mode,
                    DEFAULT_EXPORT_CONFIG,
                    progress_cb,
                )
                results["excel_files"] = excel_result
                results["errors"].extend(excel_result.get("errors", []))

            # Generar Excel resum
            if self.options.get("summary_excel", True):
                self.progress.emit(90, "Generant resum...")
                summary_path = str(Path(self.output_path) / "SUMMARY.xlsx")
                summary_result = generate_summary_excel(
                    self.samples_grouped,
                    summary_path,
                    self.calibration_data,
                    self.mode,
                    DEFAULT_EXPORT_CONFIG,
                )
                results["summary"] = summary_result

            self.progress.emit(100, "Completat")
            self.finished.emit(results)

        except Exception as e:
            self.error.emit(str(e))


class ExportPanel(QWidget):
    """Panel d'exportació."""

    export_completed = Signal(dict)

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.worker = None

        self._setup_ui()

    def _setup_ui(self):
        """Configura la interfície."""
        layout = QVBoxLayout(self)
        apply_panel_layout(layout)

        # Títol
        title = QLabel("Exportar Resultats")
        title.setFont(create_title_font())
        layout.addWidget(title)

        # Info
        info = QLabel(
            "Genera els fitxers Excel finals amb les rèpliques seleccionades "
            "i la quantificació aplicada."
        )
        info.setWordWrap(True)
        info.setStyleSheet(STYLE_LABEL_SECONDARY)
        layout.addWidget(info)

        # Opcions d'exportació
        options_group = QGroupBox("Opcions d'Exportació")
        options_layout = QVBoxLayout(options_group)

        self.individual_check = QCheckBox("Generar Excel per cada mostra")
        self.individual_check.setChecked(True)
        self.individual_check.setToolTip(
            "Crea un fitxer Excel per cada mostra amb fulls:\n"
            "• ID: Traçabilitat (fitxers, shifts, quantificació)\n"
            "• DOC: Cromatogrames DOC (final + raw)\n"
            "• DAD: 6 longituds d'ona\n"
            "• RESULTS: Integracions per fraccions"
        )
        options_layout.addWidget(self.individual_check)

        self.summary_check = QCheckBox("Generar Excel resum")
        self.summary_check.setChecked(True)
        self.summary_check.setToolTip(
            "Crea SUMMARY.xlsx amb una fila per mostra:\n"
            "concentració, SNR, warnings"
        )
        options_layout.addWidget(self.summary_check)

        self.pdf_check = QCheckBox("Generar informe PDF")
        self.pdf_check.setChecked(False)
        self.pdf_check.setEnabled(False)
        self.pdf_check.setToolTip("Pendent d'implementar")
        options_layout.addWidget(self.pdf_check)

        layout.addWidget(options_group)

        # Carpeta de sortida
        output_group = QGroupBox("Carpeta de Sortida")
        output_layout = QHBoxLayout(output_group)

        self.output_path_input = QLineEdit()
        self.output_path_input.setPlaceholderText("Selecciona carpeta de sortida...")
        output_layout.addWidget(self.output_path_input, 1)

        browse_btn = QPushButton("Examinar...")
        browse_btn.clicked.connect(self._browse_output)
        output_layout.addWidget(browse_btn)

        layout.addWidget(output_group)

        # Resum
        self.summary_frame = QFrame()
        self.summary_frame.setFrameStyle(QFrame.StyledPanel)
        self.summary_frame.setStyleSheet("background-color: #f5f5f5; padding: 12px;")
        summary_layout = QGridLayout(self.summary_frame)

        self.n_samples_label = QLabel("Mostres: -")
        self.n_samples_label.setFont(QFont("Segoe UI", 10))
        summary_layout.addWidget(self.n_samples_label, 0, 0)

        self.method_label = QLabel("Mètode: -")
        self.method_label.setFont(QFont("Segoe UI", 10))
        summary_layout.addWidget(self.method_label, 0, 1)

        self.calibration_label = QLabel("Calibració: -")
        self.calibration_label.setFont(QFont("Segoe UI", 10))
        summary_layout.addWidget(self.calibration_label, 1, 0, 1, 2)

        layout.addWidget(self.summary_frame)

        # Empty state (quan no hi ha dades)
        self.empty_state = create_empty_state_widget(
            "📤",
            "No hi ha dades per exportar",
            "Completa primer l'anàlisi per poder generar els fitxers d'exportació."
        )
        self.empty_state.setVisible(True)
        layout.addWidget(self.empty_state)

        # Barra de progrés
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #666;")
        layout.addWidget(self.status_label)

        layout.addStretch()

        # Botons
        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch()

        self.export_btn = QPushButton("Exportar")
        self.export_btn.setStyleSheet("font-weight: bold; padding: 8px 24px;")
        self.export_btn.clicked.connect(self._run_export)
        buttons_layout.addWidget(self.export_btn)

        layout.addLayout(buttons_layout)

    def showEvent(self, event):
        """Es crida quan el panel es fa visible."""
        super().showEvent(event)
        self._update_summary()

    def reset(self):
        """Reinicia el panel al seu estat inicial."""
        self.worker = None
        self.output_path_input.clear()
        self.individual_check.setChecked(True)
        self.summary_check.setChecked(True)
        self.pdf_check.setChecked(False)
        self.n_samples_label.setText("Mostres: -")
        self.method_label.setText("Mètode: -")
        self.calibration_label.setText("Calibració: -")
        self.empty_state.setVisible(True)
        self.summary_frame.setVisible(False)
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("")
        self.export_btn.setEnabled(True)

    def _update_summary(self):
        """Actualitza el resum."""
        processed_data = self.main_window.processed_data

        if not processed_data or not processed_data.get("samples_grouped"):
            self.n_samples_label.setText("Mostres: -")
            self.method_label.setText("Mètode: -")
            self.calibration_label.setText("Calibració: -")
            self.empty_state.setVisible(True)
            self.summary_frame.setVisible(False)
            return

        # Hi ha dades - amagar empty state
        self.empty_state.setVisible(False)
        self.summary_frame.setVisible(True)

        # Comptar mostres
        samples_grouped = processed_data.get("samples_grouped", {})
        n_samples = len(samples_grouped)

        # Mètode
        method = processed_data.get("method", "COLUMN")

        # Calibració
        cal_data = self.main_window.calibration_data
        if cal_data:
            cal_info = f"KHP{cal_data.get('khp_conc_ppm', '?')} @ {cal_data.get('date', '?')}"
        else:
            cal_info = "No disponible"

        self.n_samples_label.setText(f"Mostres: {n_samples}")
        self.method_label.setText(f"Mètode: {method}")
        self.calibration_label.setText(f"Calibració: {cal_info}")

        # Default output path
        seq_path = self.main_window.seq_path
        if seq_path and not self.output_path_input.text():
            default_output = str(Path(seq_path) / "Export")
            self.output_path_input.setText(default_output)

    def _browse_output(self):
        """Obre diàleg per seleccionar carpeta de sortida."""
        path = QFileDialog.getExistingDirectory(
            self,
            "Selecciona carpeta de sortida",
            self.output_path_input.text()
        )
        if path:
            self.output_path_input.setText(path)

    def _run_export(self):
        """Executa l'exportació."""
        output_path = self.output_path_input.text()
        if not output_path:
            QMessageBox.warning(self, "Avís", "Selecciona una carpeta de sortida.")
            return

        processed_data = self.main_window.processed_data
        if not processed_data:
            QMessageBox.warning(self, "Avís", "No hi ha dades processades.")
            return

        samples_grouped = processed_data.get("samples_grouped", {})
        if not samples_grouped:
            QMessageBox.warning(self, "Avís", "No hi ha mostres per exportar.")
            return

        # Crear carpeta si no existeix
        Path(output_path).mkdir(parents=True, exist_ok=True)

        # Opcions
        options = {
            "individual_excels": self.individual_check.isChecked(),
            "summary_excel": self.summary_check.isChecked(),
        }

        # Deshabilitar botó i mostrar progrés
        self.export_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        # Mode i calibració
        method = processed_data.get("method", "COLUMN")
        calibration_data = self.main_window.calibration_data

        # Crear worker
        self.worker = ExportWorker(
            samples_grouped,
            output_path,
            calibration_data,
            method,
            options,
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_progress(self, pct, msg):
        """Gestiona el progrés."""
        self.progress_bar.setValue(pct)
        self.status_label.setText(msg)

    def _on_finished(self, results):
        """Gestiona la finalització."""
        self.progress_bar.setVisible(False)
        self.export_btn.setEnabled(True)

        # Resum
        excel_result = results.get("excel_files", {})
        summary_result = results.get("summary", {})
        errors = results.get("errors", [])

        n_exported = excel_result.get("n_exported", 0) if excel_result else 0

        msg = f"Exportació completada!\n\n"
        msg += f"Fitxers Excel generats: {n_exported}\n"

        if summary_result:
            msg += f"Excel resum: SUMMARY.xlsx ({summary_result.get('n_samples', 0)} mostres)\n"

        if errors:
            msg += f"\nErrors ({len(errors)}):\n"
            msg += "\n".join(errors[:5])
            if len(errors) > 5:
                msg += f"\n... i {len(errors) - 5} més"

        QMessageBox.information(self, "Exportació Completada", msg)

        self.status_label.setText(f"Exportat: {n_exported} fitxers")
        self.export_completed.emit(results)

        # Obrir carpeta
        output_path = self.output_path_input.text()
        if output_path and n_exported > 0:
            import os
            os.startfile(output_path)

    def _on_error(self, error_msg):
        """Gestiona errors."""
        self.progress_bar.setVisible(False)
        self.export_btn.setEnabled(True)
        self.status_label.setText("Error")
        QMessageBox.critical(self, "Error", f"Error durant l'exportació:\n{error_msg}")
