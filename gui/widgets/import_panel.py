"""
HPSEC Suite - Import Panel
===========================

Panel para la fase 1: Importación de secuencias.
"""

import os
from pathlib import Path
from datetime import datetime

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QFileDialog, QGroupBox, QGridLayout, QFrame,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
    QRadioButton, QButtonGroup, QSplitter, QTextEdit
)
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QFont

# Importar módulos de backend
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from hpsec_import import (
    import_sequence, load_manifest, import_from_manifest,
    generate_import_manifest, save_import_manifest
)


class ImportWorker(QThread):
    """Worker thread para importación asíncrona."""
    progress = Signal(int, str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, seq_path, use_manifest=False, manifest=None):
        super().__init__()
        self.seq_path = seq_path
        self.use_manifest = use_manifest
        self.manifest = manifest

    def run(self):
        try:
            def progress_cb(pct, msg):
                self.progress.emit(int(pct), msg)

            if self.use_manifest and self.manifest:
                result = import_from_manifest(
                    self.seq_path,
                    manifest=self.manifest,
                    progress_callback=progress_cb
                )
            else:
                result = import_sequence(
                    self.seq_path,
                    progress_callback=progress_cb
                )

            self.finished.emit(result)

        except Exception as e:
            self.error.emit(str(e))


class ImportPanel(QWidget):
    """Panel de importación de secuencias."""

    # Señales
    import_completed = Signal(dict)

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.seq_path = None
        self.existing_manifest = None
        self.imported_data = None
        self.worker = None

        self._setup_ui()

    def _setup_ui(self):
        """Configura la interfaz del panel."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        # === Sección: Selección de carpeta ===
        folder_group = QGroupBox("Seleccionar Seqüència")
        folder_layout = QHBoxLayout(folder_group)

        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("Selecciona una carpeta SEQ...")
        self.path_input.setReadOnly(True)
        folder_layout.addWidget(self.path_input, 1)

        self.browse_btn = QPushButton("Examinar...")
        self.browse_btn.clicked.connect(self._browse_folder)
        folder_layout.addWidget(self.browse_btn)

        layout.addWidget(folder_group)

        # === Sección: Info de manifest existente ===
        self.manifest_frame = QFrame()
        self.manifest_frame.setProperty("card", True)
        self.manifest_frame.setVisible(False)

        manifest_layout = QVBoxLayout(self.manifest_frame)

        manifest_title = QLabel("Importació Anterior Detectada")
        manifest_title.setFont(QFont("Segoe UI", 12, QFont.Bold))
        manifest_layout.addWidget(manifest_title)

        self.manifest_info = QLabel()
        self.manifest_info.setWordWrap(True)
        manifest_layout.addWidget(self.manifest_info)

        # Opciones de importación
        options_layout = QVBoxLayout()
        self.import_mode_group = QButtonGroup(self)

        self.use_manifest_radio = QRadioButton(
            "Usar importació anterior (ràpid - llegeix dades segons manifest)"
        )
        self.use_manifest_radio.setChecked(True)
        self.import_mode_group.addButton(self.use_manifest_radio)
        options_layout.addWidget(self.use_manifest_radio)

        self.full_import_radio = QRadioButton(
            "Reimportar completament (detectar i aparellar de nou)"
        )
        self.import_mode_group.addButton(self.full_import_radio)
        options_layout.addWidget(self.full_import_radio)

        manifest_layout.addLayout(options_layout)
        layout.addWidget(self.manifest_frame)

        # === Sección: Resultados de importación ===
        self.results_frame = QFrame()
        self.results_frame.setProperty("card", True)
        self.results_frame.setVisible(False)

        results_layout = QVBoxLayout(self.results_frame)

        results_title = QLabel("Resultat de la Importació")
        results_title.setFont(QFont("Segoe UI", 12, QFont.Bold))
        results_layout.addWidget(results_title)

        # Grid de estadísticas
        stats_grid = QGridLayout()
        self.stats_labels = {}

        stats = [
            ("seq_name", "Seqüència:"),
            ("method", "Mètode:"),
            ("data_mode", "Mode dades:"),
            ("samples", "Mostres:"),
            ("khp", "KHP:"),
            ("controls", "Controls:"),
            ("direct", "Amb Direct:"),
            ("uib", "Amb UIB:"),
            ("dad", "Amb DAD:"),
        ]

        for i, (key, label) in enumerate(stats):
            row, col = i // 3, (i % 3) * 2
            lbl = QLabel(label)
            lbl.setStyleSheet("font-weight: bold;")
            stats_grid.addWidget(lbl, row, col)

            val = QLabel("-")
            self.stats_labels[key] = val
            stats_grid.addWidget(val, row, col + 1)

        results_layout.addLayout(stats_grid)

        # Tabla de muestras
        self.samples_table = QTableWidget()
        self.samples_table.setColumnCount(6)
        self.samples_table.setHorizontalHeaderLabels([
            "Mostra", "Tipus", "Rep", "DOC Direct", "DOC UIB", "DAD"
        ])
        self.samples_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.samples_table.setAlternatingRowColors(True)
        results_layout.addWidget(self.samples_table)

        # Warnings
        self.warnings_label = QLabel()
        self.warnings_label.setProperty("status", "warning")
        self.warnings_label.setVisible(False)
        self.warnings_label.setWordWrap(True)
        results_layout.addWidget(self.warnings_label)

        layout.addWidget(self.results_frame)

        # Spacer
        layout.addStretch()

        # === Botones de acción ===
        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch()

        self.save_manifest_btn = QPushButton("Guardar Manifest")
        self.save_manifest_btn.setProperty("secondary", True)
        self.save_manifest_btn.setVisible(False)
        self.save_manifest_btn.clicked.connect(self._save_manifest)
        buttons_layout.addWidget(self.save_manifest_btn)

        self.import_btn = QPushButton("Importar")
        self.import_btn.setEnabled(False)
        self.import_btn.clicked.connect(self._run_import)
        buttons_layout.addWidget(self.import_btn)

        self.next_btn = QPushButton("Següent →")
        self.next_btn.setEnabled(False)
        self.next_btn.clicked.connect(self._go_next)
        buttons_layout.addWidget(self.next_btn)

        layout.addLayout(buttons_layout)

    def set_sequence_path(self, path):
        """Establece la ruta de la secuencia."""
        self.seq_path = path
        self.path_input.setText(path)
        self._check_manifest()

    def _browse_folder(self):
        """Abre diálogo para seleccionar carpeta."""
        path = QFileDialog.getExistingDirectory(
            self,
            "Selecciona carpeta SEQ",
            "",
            QFileDialog.ShowDirsOnly
        )
        if path:
            self.set_sequence_path(path)

    def _check_manifest(self):
        """Comprueba si existe un manifest."""
        self.existing_manifest = load_manifest(self.seq_path)

        if self.existing_manifest:
            self._show_manifest_info()
        else:
            self.manifest_frame.setVisible(False)

        self.import_btn.setEnabled(True)
        self.results_frame.setVisible(False)
        self.next_btn.setEnabled(False)

    def _show_manifest_info(self):
        """Muestra información del manifest existente."""
        manifest = self.existing_manifest
        seq = manifest.get("sequence", {})
        summary = manifest.get("summary", {})
        generated = manifest.get("generated_at", "")[:19].replace("T", " ")

        info_text = f"""
<b>Seqüència:</b> {seq.get('name', 'N/A')}<br>
<b>Importat el:</b> {generated}<br>
<b>Mode:</b> {seq.get('method', 'N/A')} / {seq.get('data_mode', 'N/A')}<br><br>
<b>Mostres:</b> {summary.get('total_samples', 0)} |
<b>KHP:</b> {summary.get('total_khp', 0)} |
<b>Controls:</b> {summary.get('total_controls', 0)}<br>
<b>Rèpliques:</b> {summary.get('total_replicas', 0)}
(Direct: {summary.get('replicas_with_direct', 0)},
UIB: {summary.get('replicas_with_uib', 0)},
DAD: {summary.get('replicas_with_dad', 0)})
        """
        self.manifest_info.setText(info_text.strip())
        self.manifest_frame.setVisible(True)
        self.use_manifest_radio.setChecked(True)

    def _run_import(self):
        """Ejecuta la importación."""
        if not self.seq_path:
            return

        self.import_btn.setEnabled(False)
        self.browse_btn.setEnabled(False)
        self.main_window.show_progress(0)

        use_manifest = (
            self.existing_manifest and
            self.use_manifest_radio.isChecked()
        )

        self.worker = ImportWorker(
            self.seq_path,
            use_manifest=use_manifest,
            manifest=self.existing_manifest if use_manifest else None
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_import_finished)
        self.worker.error.connect(self._on_import_error)
        self.worker.start()

    def _on_progress(self, pct, msg):
        """Handler de progreso."""
        self.main_window.show_progress(pct)
        self.main_window.set_status(msg)

    def _on_import_finished(self, result):
        """Handler cuando termina la importación."""
        self.main_window.show_progress(-1)
        self.import_btn.setEnabled(True)
        self.browse_btn.setEnabled(True)

        if not result.get("success"):
            errors = result.get("errors", ["Error desconegut"])
            QMessageBox.critical(
                self,
                "Error d'Importació",
                f"Error: {errors[0] if errors else 'Error desconegut'}"
            )
            return

        self.imported_data = result
        self.main_window.imported_data = result
        self._show_results(result)

        self.next_btn.setEnabled(True)
        self.save_manifest_btn.setVisible(True)
        self.main_window.enable_tab(1)
        self.main_window.set_status("Importació completada", 5000)

    def _on_import_error(self, error_msg):
        """Handler de error."""
        self.main_window.show_progress(-1)
        self.import_btn.setEnabled(True)
        self.browse_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", f"Error durant la importació:\n{error_msg}")

    def _show_results(self, result):
        """Muestra los resultados de la importación."""
        self.results_frame.setVisible(True)

        # Generar manifest para obtener estadísticas
        manifest = generate_import_manifest(result)
        summary = manifest.get("summary", {})
        seq = manifest.get("sequence", {})

        # Actualizar estadísticas
        self.stats_labels["seq_name"].setText(seq.get("name", "-"))
        self.stats_labels["method"].setText(seq.get("method", "-"))
        self.stats_labels["data_mode"].setText(seq.get("data_mode", "-"))
        self.stats_labels["samples"].setText(str(summary.get("total_samples", 0)))
        self.stats_labels["khp"].setText(str(summary.get("total_khp", 0)))
        self.stats_labels["controls"].setText(str(summary.get("total_controls", 0)))
        self.stats_labels["direct"].setText(str(summary.get("replicas_with_direct", 0)))
        self.stats_labels["uib"].setText(str(summary.get("replicas_with_uib", 0)))
        self.stats_labels["dad"].setText(str(summary.get("replicas_with_dad", 0)))

        # Llenar tabla
        samples = manifest.get("samples", [])
        self.samples_table.setRowCount(0)

        for sample in samples:
            for rep in sample.get("replicas", []):
                row = self.samples_table.rowCount()
                self.samples_table.insertRow(row)

                self.samples_table.setItem(row, 0, QTableWidgetItem(sample["name"]))
                self.samples_table.setItem(row, 1, QTableWidgetItem(sample["type"]))
                self.samples_table.setItem(row, 2, QTableWidgetItem(str(rep["replica"])))

                # Direct
                d = rep.get("direct", {})
                direct_str = f"{d['n_points']} pts" if d else "-"
                self.samples_table.setItem(row, 3, QTableWidgetItem(direct_str))

                # UIB
                u = rep.get("uib", {})
                uib_str = f"{u['n_points']} pts" if u else "-"
                self.samples_table.setItem(row, 4, QTableWidgetItem(uib_str))

                # DAD
                dad = rep.get("dad", {})
                dad_str = f"{dad['n_points']} pts" if dad else "-"
                self.samples_table.setItem(row, 5, QTableWidgetItem(dad_str))

        # Warnings
        warnings = manifest.get("warnings", [])
        if warnings:
            self.warnings_label.setText("⚠️ " + " | ".join(warnings))
            self.warnings_label.setVisible(True)
        else:
            self.warnings_label.setVisible(False)

    def _save_manifest(self):
        """Guarda el manifest JSON."""
        if not self.imported_data:
            return

        output_path = save_import_manifest(self.imported_data)
        QMessageBox.information(
            self,
            "Manifest Guardat",
            f"Manifest guardat a:\n{output_path}"
        )

    def _go_next(self):
        """Navega al siguiente tab."""
        self.main_window.go_to_tab(1)
