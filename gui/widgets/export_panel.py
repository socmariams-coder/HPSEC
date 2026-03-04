"""
HPSEC Suite - Export Panel (Pas 4: Exportar)
=============================================

Panel d'exportació de resultats amb principis FAIR.

Contingut:
- Resum pre-exportació: comptadors (N mostres, M excloses, K avisos)
- Opcions exportació: checkboxes (Excels individuals, SUMMARY, PDF report)
- Secció FAIR (placeholder): metadades traçabilitat
- Consolidació BP (COLUMN mode)
- Botó "Generar Resultats" + barra progrés
- Escriu review_result.json per al dashboard
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QScrollArea, QMessageBox, QProgressBar,
    QTableWidget, QTableWidgetItem, QComboBox, QHeaderView,
    QGroupBox, QCheckBox, QGridLayout, QFileDialog, QLineEdit,
)
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QFont, QColor

import json as _json
import logging
import os

from pathlib import Path

logger = logging.getLogger(__name__)

from hpsec_export import (
    export_sequence, generate_summary_excel, generate_summary_csv,
    write_metadata_json, create_export_zip,
    DEFAULT_EXPORT_CONFIG,
)
from gui.widgets.styles import (
    COLOR_SUCCESS, COLOR_WARNING, COLOR_ERROR,
    apply_panel_layout
)


class BPDiscoveryWorker(QThread):
    """Worker per cercar dades BP en background."""
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, seq_path, sample_names, data_folder):
        super().__init__()
        self.seq_path = seq_path
        self.sample_names = sample_names
        self.data_folder = data_folder

    def run(self):
        try:
            from hpsec_consolidate import find_bp_for_samples
            result = find_bp_for_samples(
                self.seq_path, self.sample_names, self.data_folder
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class _BPReloadWorker(QThread):
    """Worker per rellegir dades BP des d'una BP diferent."""
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, bp_path, sample_names, available_bps, seq_path):
        super().__init__()
        self.bp_path = bp_path
        self.sample_names = sample_names
        self.available_bps = available_bps
        self.seq_path = seq_path

    def run(self):
        try:
            from hpsec_consolidate import find_bp_for_samples
            result = find_bp_for_samples(
                self.seq_path, self.sample_names,
                str(Path(self.seq_path).parent),
                preferred_bp_path=self.bp_path
            )
            result["available_bps"] = self.available_bps
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class GenerateWorker(QThread):
    """Worker per generar resultats en background."""
    progress = Signal(int, str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, samples_grouped, seq_path, calibration_data, mode, config,
                 bp_resolved=None, generate_pdf=False,
                 export_raw=False, export_processed=False,
                 csv_summary=False, csv_separator=";",
                 export_zip=False, export_metadata=True,
                 custom_output_dir=None):
        super().__init__()
        self.samples_grouped = samples_grouped
        self.seq_path = seq_path
        self.calibration_data = calibration_data
        self.mode = mode
        self.config = config
        self.bp_resolved = bp_resolved
        self.generate_pdf = generate_pdf
        self.export_raw = export_raw
        self.export_processed = export_processed
        self.csv_summary = csv_summary
        self.csv_separator = csv_separator
        self.export_zip = export_zip
        self.export_metadata = export_metadata
        self.custom_output_dir = custom_output_dir

    def run(self):
        try:
            config = self.config or DEFAULT_EXPORT_CONFIG
            results = {"excel_files": None, "summary": None, "errors": []}

            def progress_cb(pct, msg):
                self.progress.emit(pct, msg)

            # Excels individuals (+ RAW/PROCESSED CSVs)
            if self.custom_output_dir:
                resultats_path = self.custom_output_dir
            else:
                resultats_path = str(Path(self.seq_path) / "RESULTATS")
            self.progress.emit(0, "Generant Excels individuals...")
            excel_result = export_sequence(
                self.samples_grouped,
                resultats_path,
                self.calibration_data,
                self.mode,
                config,
                progress_cb,
                seq_path=self.seq_path,
                bp_resolved=self.bp_resolved,
                export_raw=self.export_raw,
                export_processed=self.export_processed,
                csv_separator=self.csv_separator,
            )
            results["excel_files"] = excel_result
            results["errors"].extend(excel_result.get("errors", []))
            results["n_raw"] = len(excel_result.get("raw_files", []))
            results["n_processed"] = len(excel_result.get("processed_files", []))

            # SUMMARY.xlsx → custom dir or SEQ/CHECK/
            self.progress.emit(80, "Generant SUMMARY.xlsx...")
            if self.custom_output_dir:
                summary_dir = Path(self.custom_output_dir)
            else:
                summary_dir = Path(self.seq_path) / "CHECK"
            summary_dir.mkdir(parents=True, exist_ok=True)
            summary_path = str(summary_dir / "SUMMARY.xlsx")
            summary_result = generate_summary_excel(
                self.samples_grouped,
                summary_path,
                self.calibration_data,
                self.mode,
                config,
            )
            results["summary"] = summary_result

            # SUMMARY.csv → same dir as SUMMARY.xlsx
            if self.csv_summary:
                self.progress.emit(84, "Generant SUMMARY.csv...")
                try:
                    csv_summary_path = str(summary_dir / "SUMMARY.csv")
                    generate_summary_csv(
                        self.samples_grouped,
                        csv_summary_path,
                        self.calibration_data,
                        self.mode,
                        config,
                        separator=self.csv_separator,
                    )
                    results["csv_summary"] = csv_summary_path
                except Exception as e:
                    results["errors"].append(f"CSV summary: {e}")

            # metadata.json → SEQ/RESULTATS/
            if self.export_metadata and (self.export_raw or self.export_processed):
                self.progress.emit(87, "Generant metadata.json...")
                try:
                    meta_path = str(Path(resultats_path) / "metadata.json")
                    export_opts = {
                        "raw": self.export_raw,
                        "processed": self.export_processed,
                        "csv_separator": repr(self.csv_separator),
                        "zip": self.export_zip,
                    }
                    write_metadata_json(
                        meta_path, self.samples_grouped, self.mode,
                        self.calibration_data, config,
                        self.seq_path, export_opts,
                    )
                    results["metadata"] = meta_path
                except Exception as e:
                    results["errors"].append(f"metadata.json: {e}")

            # PDF analysis report
            if self.generate_pdf:
                self.progress.emit(90, "Generant PDF anàlisi...")
                try:
                    from generate_analysis_report import generate_analysis_report
                    report_data = {
                        "samples_grouped": self.samples_grouped,
                        "method": self.mode,
                        "seq_path": self.seq_path,
                        "seq_name": Path(self.seq_path).name,
                        "success": True,
                    }
                    pdf_path = generate_analysis_report(
                        self.seq_path, analysis_data=report_data
                    )
                    results["pdf_report"] = pdf_path
                except Exception as e:
                    results["errors"].append(f"PDF report: {e}")

            # ZIP packaging
            if self.export_zip:
                self.progress.emit(95, "Creant ZIP...")
                try:
                    seq_name = Path(self.seq_path).name
                    zip_path = str(Path(resultats_path) / f"{seq_name}_HPSEC_EXPORT.zip")
                    create_export_zip(resultats_path, zip_path)
                    results["zip_path"] = zip_path
                except Exception as e:
                    results["errors"].append(f"ZIP: {e}")

            self.progress.emit(100, "Completat")
            self.finished.emit(results)

        except Exception as e:
            self.error.emit(str(e))


class ExportPanel(QWidget):
    """Panel d'exportació de resultats (Pas 4: Exportar)."""

    export_completed = Signal(dict)

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.worker = None
        self._bp_worker = None
        self._bp_resolved = None
        self._bp_available = []
        self._current_method = "COLUMN"
        self._current_seq_path = ""
        self._current_sample_names = []
        self._auto_generated = False
        self._populated_seq = ""
        self._export_worker = None
        self._export_temp_dir = None
        self._export_zip_path = None

        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(0)

        # Scrollable content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        content = QWidget()
        self.content_layout = QVBoxLayout(content)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(12)
        scroll.setWidget(content)

        _GRP_STYLE = """
            QGroupBox {
                font-weight: bold; font-size: 11px; color: #2c3e50;
                border: 1px solid #d5dbdb; border-radius: 6px;
                margin-top: 8px; padding-top: 18px;
                background-color: #fafafa;
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 12px; padding: 0 4px;
            }
        """

        # === HEADER / SUMMARY ===
        self.header_label = QLabel()
        self.header_label.setTextFormat(Qt.RichText)
        self.header_label.setWordWrap(True)
        self.header_label.setStyleSheet(
            "font-size: 12px; color: #2c3e50; padding: 6px 10px;"
            "background: #fff; border: 1px solid #e0e0e0; border-radius: 6px;"
        )
        self.content_layout.addWidget(self.header_label)

        # =================================================================
        # PART 1: RESULTATS GENERATS (auto-generació)
        # =================================================================
        self.results_frame = QFrame()
        self.results_frame.setStyleSheet("""
            QFrame {
                background-color: #f0f9f0;
                border: 1px solid #b8e0b8;
                border-radius: 8px;
            }
        """)
        results_layout = QVBoxLayout(self.results_frame)
        results_layout.setContentsMargins(16, 12, 16, 12)
        results_layout.setSpacing(4)

        self.results_status_label = QLabel()
        self.results_status_label.setTextFormat(Qt.RichText)
        self.results_status_label.setWordWrap(True)
        self.results_status_label.setStyleSheet(
            "font-size: 12px; color: #2c3e50; border: none; background: transparent;"
        )
        results_layout.addWidget(self.results_status_label)

        self.results_path_label = QLabel()
        self.results_path_label.setTextFormat(Qt.RichText)
        self.results_path_label.setWordWrap(True)
        self.results_path_label.setStyleSheet(
            "font-size: 10px; color: #7f8c8d; border: none; background: transparent;"
        )
        results_layout.addWidget(self.results_path_label)

        self.results_frame.setVisible(False)
        self.content_layout.addWidget(self.results_frame)

        # =================================================================
        # PART 2: CONSOLIDACIÓ BP (si COLUMN)
        # =================================================================
        # === SECCIÓ CONSOLIDACIÓ BP ===
        self.bp_group = QGroupBox("CONSOLIDACIÓ BP")
        self.bp_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold; font-size: 11px; color: #2c3e50;
                border: 1px solid #d5dbdb; border-radius: 6px;
                margin-top: 8px; padding-top: 18px;
                background-color: #fafafa;
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 12px; padding: 0 4px;
            }
        """)
        bp_layout = QVBoxLayout(self.bp_group)
        bp_layout.setContentsMargins(12, 8, 12, 8)
        bp_layout.setSpacing(6)

        bp_selector_row = QHBoxLayout()
        bp_selector_row.addWidget(QLabel("SEQ BP:"))
        self.bp_combo = QComboBox()
        self.bp_combo.setMinimumWidth(220)
        self.bp_combo.currentIndexChanged.connect(self._on_bp_combo_changed)
        bp_selector_row.addWidget(self.bp_combo)
        self.bp_status_label = QLabel("")
        self.bp_status_label.setStyleSheet("color: #7f8c8d; font-size: 11px;")
        bp_selector_row.addWidget(self.bp_status_label, 1)
        bp_layout.addLayout(bp_selector_row)

        self.bp_table = QTableWidget()
        self.bp_table.setColumnCount(5)
        self.bp_table.setHorizontalHeaderLabels(
            ["Mostra", "BP", "Rèplica", "ppm", "SNR"]
        )
        self.bp_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for col in range(1, 5):
            self.bp_table.horizontalHeader().setSectionResizeMode(
                col, QHeaderView.ResizeToContents
            )
        self.bp_table.verticalHeader().setVisible(False)
        self.bp_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.bp_table.setAlternatingRowColors(True)
        self.bp_table.setMaximumHeight(200)
        self.bp_table.setStyleSheet("""
            QTableWidget {
                font-size: 11px; border: 1px solid #e0e0e0;
                gridline-color: #f0f0f0;
            }
            QTableWidget::item { padding: 2px 6px; }
        """)
        bp_layout.addWidget(self.bp_table)

        self.bp_info_label = QLabel("")
        self.bp_info_label.setStyleSheet("color: #7f8c8d; font-size: 10px; font-style: italic;")
        bp_layout.addWidget(self.bp_info_label)

        self.bp_group.setVisible(False)
        self.content_layout.addWidget(self.bp_group)

        # =================================================================
        # PART 3: EXPORT ADDICIONAL
        # =================================================================
        export_group = QGroupBox("Export addicional")
        export_group.setStyleSheet(_GRP_STYLE)
        export_layout = QVBoxLayout(export_group)
        export_layout.setContentsMargins(12, 8, 12, 8)
        export_layout.setSpacing(6)

        export_hint = QLabel("Selecciona contingut i destí per exportar a una altra carpeta o ZIP.")
        export_hint.setStyleSheet("color: #7f8c8d; font-size: 10px; font-style: italic;")
        export_hint.setWordWrap(True)
        export_layout.addWidget(export_hint)

        # --- Contingut ---
        content_label = QLabel("Contingut")
        content_label.setStyleSheet("font-weight: bold; font-size: 10px; color: #5D6D7E; margin-top: 2px;")
        export_layout.addWidget(content_label)

        # Row 1: Excels + SUMMARY
        row1 = QHBoxLayout()
        self.exp_excel_check = QCheckBox("Excels individuals")
        self.exp_excel_check.setChecked(True)
        self.exp_excel_check.setToolTip("Excel per mostra: ID + DOC + DAD + RESULTS")
        row1.addWidget(self.exp_excel_check)
        self.exp_summary_check = QCheckBox("SUMMARY.xlsx")
        self.exp_summary_check.setChecked(True)
        self.exp_summary_check.setToolTip("Taula resum amb una fila per mostra")
        row1.addWidget(self.exp_summary_check)
        row1.addStretch()
        export_layout.addLayout(row1)

        # Row 2: RAW + PROCESSED
        row2 = QHBoxLayout()
        self.exp_raw_check = QCheckBox("RAW (DOC + DAD 101\u03bb)")
        self.exp_raw_check.setChecked(False)
        self.exp_raw_check.setToolTip(
            "CSV amb senyals crus DOC (Direct+UIB)\n"
            "i DAD complet (101 \u03bb, downsampled dt=0.04 min).\n"
            "BP: espectre DAD a t_max amb totes les \u03bb."
        )
        row2.addWidget(self.exp_raw_check)
        self.exp_processed_check = QCheckBox("PROCESSED (DOC + DAD 6\u03bb)")
        self.exp_processed_check.setChecked(False)
        self.exp_processed_check.setToolTip(
            "CSV amb senyals processats: DOC net + DAD 6\u03bb,\n"
            "fraccions integrades i concentracions ppm.\n"
            "BP: cromatograma DOC+DAD (sense fraccions)."
        )
        row2.addWidget(self.exp_processed_check)
        row2.addStretch()
        export_layout.addLayout(row2)

        # Row 3: CSV summary + PDF + metadata
        row3 = QHBoxLayout()
        self.exp_csv_summary_check = QCheckBox("CSV SUMMARY")
        self.exp_csv_summary_check.setChecked(False)
        self.exp_csv_summary_check.setToolTip("SUMMARY.csv amb metadades i fingerprints")
        row3.addWidget(self.exp_csv_summary_check)
        self.exp_pdf_check = QCheckBox("Informe PDF")
        self.exp_pdf_check.setChecked(False)
        self.exp_pdf_check.setToolTip("REPORT_Analysis amb cromatogrames i estadístiques")
        row3.addWidget(self.exp_pdf_check)
        self.exp_metadata_check = QCheckBox("metadata.json")
        self.exp_metadata_check.setChecked(True)
        self.exp_metadata_check.setToolTip("Fitxer FAIR amb fingerprints, llista mostres, calibració")
        row3.addWidget(self.exp_metadata_check)
        row3.addStretch()
        export_layout.addLayout(row3)

        # Separador CSV
        sep_row = QHBoxLayout()
        sep_row.addWidget(QLabel("Separador CSV:"))
        self.csv_separator_combo = QComboBox()
        self.csv_separator_combo.addItem("; Punt i coma", ";")
        self.csv_separator_combo.addItem(", Coma", ",")
        self.csv_separator_combo.addItem("TAB Tabulador", "\t")
        self.csv_separator_combo.setToolTip(
            "Separador pels CSV. Punt i coma recomanat per Excel EU.\n"
            "Decimals sempre amb punt (.)"
        )
        self.csv_separator_combo.setMaximumWidth(180)
        sep_row.addWidget(self.csv_separator_combo)
        sep_row.addStretch()
        export_layout.addLayout(sep_row)

        # --- Separador visual ---
        exp_sep = QFrame()
        exp_sep.setFixedHeight(1)
        exp_sep.setStyleSheet("background-color: #d5dbdb;")
        export_layout.addWidget(exp_sep)

        # --- Destí ---
        dest_label = QLabel("Destí")
        dest_label.setStyleSheet("font-weight: bold; font-size: 10px; color: #5D6D7E; margin-top: 2px;")
        export_layout.addWidget(dest_label)

        # Carpeta
        dest_folder_row = QHBoxLayout()
        self.dest_folder_radio = QCheckBox("Carpeta:")
        self.dest_folder_radio.setChecked(False)
        dest_folder_row.addWidget(self.dest_folder_radio)
        self.dest_path_edit = QLineEdit()
        self.dest_path_edit.setPlaceholderText("Selecciona carpeta (SharePoint, OneDrive, local...)")
        self.dest_path_edit.setReadOnly(True)
        self.dest_path_edit.setEnabled(False)
        self.dest_path_edit.setStyleSheet(
            "font-size: 11px; padding: 4px 8px; border: 1px solid #d5dbdb; border-radius: 4px;"
        )
        dest_folder_row.addWidget(self.dest_path_edit, 1)
        self.dest_browse_btn = QPushButton("Seleccionar...")
        self.dest_browse_btn.setStyleSheet(
            "font-size: 11px; padding: 4px 12px; border: 1px solid #2980B9;"
            "border-radius: 4px; color: #2980B9;"
        )
        self.dest_browse_btn.setEnabled(False)
        self.dest_browse_btn.clicked.connect(self._browse_dest_folder)
        dest_folder_row.addWidget(self.dest_browse_btn)
        export_layout.addLayout(dest_folder_row)

        # ZIP
        zip_row = QHBoxLayout()
        self.dest_zip_radio = QCheckBox("ZIP:")
        self.dest_zip_radio.setChecked(False)
        zip_row.addWidget(self.dest_zip_radio)
        self.zip_path_edit = QLineEdit()
        self.zip_path_edit.setPlaceholderText("Nom del fitxer ZIP...")
        self.zip_path_edit.setReadOnly(True)
        self.zip_path_edit.setEnabled(False)
        self.zip_path_edit.setStyleSheet(
            "font-size: 11px; padding: 4px 8px; border: 1px solid #d5dbdb; border-radius: 4px;"
        )
        zip_row.addWidget(self.zip_path_edit, 1)
        self.zip_browse_btn = QPushButton("Seleccionar...")
        self.zip_browse_btn.setStyleSheet(
            "font-size: 11px; padding: 4px 12px; border: 1px solid #2980B9;"
            "border-radius: 4px; color: #2980B9;"
        )
        self.zip_browse_btn.setEnabled(False)
        self.zip_browse_btn.clicked.connect(self._browse_zip_dest)
        zip_row.addWidget(self.zip_browse_btn)
        export_layout.addLayout(zip_row)

        # Toggle destí
        def _toggle_folder(checked):
            self.dest_path_edit.setEnabled(checked)
            self.dest_browse_btn.setEnabled(checked)
        def _toggle_zip(checked):
            self.zip_path_edit.setEnabled(checked)
            self.zip_browse_btn.setEnabled(checked)
        self.dest_folder_radio.toggled.connect(_toggle_folder)
        self.dest_zip_radio.toggled.connect(_toggle_zip)

        # Botó exportar
        exp_btn_row = QHBoxLayout()
        exp_btn_row.addStretch()
        self.export_btn = QPushButton("Exportar")
        self.export_btn.setStyleSheet("""
            QPushButton {
                background-color: #27AE60; color: white;
                border: none; border-radius: 6px;
                padding: 10px 28px; font-size: 13px; font-weight: bold;
            }
            QPushButton:hover { background-color: #2ECC71; }
            QPushButton:disabled { background-color: #BDC3C7; }
        """)
        self.export_btn.clicked.connect(self._run_export)
        exp_btn_row.addWidget(self.export_btn)
        exp_btn_row.addStretch()
        export_layout.addLayout(exp_btn_row)

        self.export_progress = QProgressBar()
        self.export_progress.setVisible(False)
        self.export_progress.setMaximum(100)
        export_layout.addWidget(self.export_progress)

        self.export_status_label = QLabel("")
        self.export_status_label.setAlignment(Qt.AlignCenter)
        self.export_status_label.setStyleSheet("color: #2c3e50; font-size: 11px;")
        self.export_status_label.setVisible(False)
        export_layout.addWidget(self.export_status_label)

        self.content_layout.addWidget(export_group)

        # =================================================================
        # TRAÇABILITAT FAIR (read-only info)
        # =================================================================
        fair_group = QGroupBox("Traçabilitat (FAIR)")
        fair_group.setStyleSheet(_GRP_STYLE.replace("#fafafa", "#f8f9fa"))
        fair_layout = QVBoxLayout(fair_group)
        fair_layout.setContentsMargins(12, 8, 12, 8)

        self.fair_info = QLabel()
        self.fair_info.setWordWrap(True)
        self.fair_info.setStyleSheet("color: #5D6D7E; font-size: 11px;")
        fair_layout.addWidget(self.fair_info)

        self.content_layout.addWidget(fair_group)

        # --- Progrés auto-generació (ocult, reutilitzat internament) ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximum(100)
        self.content_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #2c3e50; font-size: 11px;")
        self.status_label.setVisible(False)
        self.content_layout.addWidget(self.status_label)

        # Placeholder per compatibilitat (generate_btn i paths_label usats internament)
        self.generate_btn = QPushButton()
        self.generate_btn.setVisible(False)
        self.paths_label = QLabel()
        self.paths_label.setVisible(False)

        self.content_layout.addStretch()

        main_layout.addWidget(scroll)

    # ------------------------------------------------------------------
    # Populate from data
    # ------------------------------------------------------------------

    def populate(self, processed_data):
        """Omple el panel amb les dades processades."""
        if not processed_data:
            return

        samples_grouped = processed_data.get("samples_grouped", {})
        summary = processed_data.get("summary", {})
        method = processed_data.get("method", "COLUMN")
        seq_name = processed_data.get("seq_name", "")
        seq_path = processed_data.get("seq_path", "")
        is_bp = method.upper() == "BP"

        self._populated_seq = seq_path or seq_name
        self._auto_generated = False

        # Separate categories
        regular = {}
        light = {}
        khp = {}
        for name, data in samples_grouped.items():
            if data.get("analysis_type") == "khp":
                khp[name] = data
            elif (data.get("analysis_type") == "light"
                  or data.get("sample_type") in ("BLANK", "CONTROL")):
                light[name] = data
            else:
                regular[name] = data

        n_injections = sum(
            len(d.get("replicas", {})) for d in samples_grouped.values()
        )

        # --- HEADER ---
        self._populate_header(
            seq_name, method, summary, regular, light, khp, n_injections, is_bp,
            processed_data,
        )

        # --- FAIR INFO ---
        self._update_fair_info(processed_data)

        # --- CONSOLIDACIÓ BP ---
        self._current_method = method
        self._current_seq_path = seq_path
        self._bp_resolved = None

        if method.upper() == "COLUMN" and seq_path:
            self._current_sample_names = [
                name for name, d in regular.items()
                if d.get("analysis_type") != "khp"
            ]
            self.bp_group.setVisible(True)
            self.bp_status_label.setText("Cercant BP...")
            self.bp_table.setRowCount(0)
            self._launch_bp_discovery(seq_path, self._current_sample_names)
        else:
            self.bp_group.setVisible(False)
            self._auto_generate()

        # --- PATHS ---
        self._update_paths_label(seq_path)

    def _populate_header(self, seq_name, method, summary, regular,
                         light, khp, n_injections, is_bp, processed_data):
        """Header compacte de 2 línies."""
        n_blancs = sum(1 for d in light.values() if d.get("sample_type") == "BLANK")
        n_controls = sum(1 for d in light.values() if d.get("sample_type") == "CONTROL")
        n_invalid = sum(1 for d in regular.values() if d.get("sample_valid") is False)
        n_errors = summary.get("with_anomalies", 0)

        zone_totals = {}
        for sample in processed_data.get("samples", []):
            ti = sample.get("timeout_info") or {}
            for zone, count in (ti.get("zone_summary") or {}).items():
                zone_totals[zone] = zone_totals.get(zone, 0) + count
        critical_zone = "BP_PEAK" if is_bp else "HS"
        critical_count = zone_totals.get(critical_zone, 0)
        if critical_count == 0:
            to_verdict = f"<span style='color:{COLOR_SUCCESS}'>TO {critical_zone} lliure</span>"
        else:
            to_verdict = (f"<span style='color:{COLOR_ERROR}'>"
                          f"TO {critical_zone} afectat ({critical_count})</span>")

        line1 = (
            f"<b>{seq_name}</b> &nbsp;&middot;&nbsp; {method}"
            f" &nbsp;&middot;&nbsp; {n_injections} inj"
        )
        parts2 = [f"<b>{len(regular)}</b> mostres"]
        if n_blancs:
            parts2.append(f"{n_blancs} blancs")
        if n_controls:
            parts2.append(f"{n_controls} controls")
        if khp:
            parts2.append(f"{len(khp)} KHP")
        if n_invalid:
            parts2.append(f"<span style='color:{COLOR_ERROR}'>{n_invalid} no vàlides</span>")
        if n_errors:
            parts2.append(f"<span style='color:{COLOR_WARNING}'>{n_errors} anomalies</span>")
        parts2.append(to_verdict)
        line2 = " &middot; ".join(parts2)

        self.header_label.setText(f"{line1}<br>{line2}")

    def _update_fair_info(self, processed_data):
        """Actualitza la secció FAIR amb metadades de traçabilitat."""
        parts = []
        try:
            from hpsec_export import __version__ as export_version
            parts.append(f"Format: <b>HPSEC Export v{export_version}</b> "
                         "&nbsp;&middot;&nbsp; Encoding: UTF-8")
        except Exception:
            parts.append("Format: <b>HPSEC Export</b> &nbsp;&middot;&nbsp; Encoding: UTF-8")

        method = processed_data.get("method", "?")
        seq_name = processed_data.get("seq_name", "?")
        parts.append(f"Seqüència: <b>{seq_name}</b> ({method})")

        # Calibration info
        cal_data = self.main_window.calibration_data
        if cal_data and cal_data.get("success"):
            khp_source = cal_data.get("khp_source", "?")
            parts.append(f"Calibració: {khp_source}")

        # Fingerprints
        try:
            from hpsec_config import Config
            cfg = Config()
            fp = cfg.compute_fingerprint()
            parts.append(f"Config fingerprint: <code>{fp}</code>")
        except Exception:
            pass
        try:
            from hpsec_calibrate import compute_calibration_fingerprint
            cal_fp = compute_calibration_fingerprint()
            parts.append(f"Calibració fingerprint: <code>{cal_fp}</code>")
        except Exception:
            pass

        # CSV separator info
        csv_sep = self.csv_separator_combo.currentData()
        any_csv = (self.exp_raw_check.isChecked() or self.exp_processed_check.isChecked()
                   or self.exp_csv_summary_check.isChecked())
        if any_csv:
            sep_name = {";": "punt i coma", ",": "coma", "\t": "tabulador"}.get(csv_sep, csv_sep)
            parts.append(f"Separador CSV: <code>{sep_name}</code>")
            parts.append("Decimals: <code>.</code> (punt) &nbsp;&middot;&nbsp; "
                         "Dates: <code>ISO 8601</code>")

        # FAIR output summary
        outputs = []
        if self.exp_raw_check.isChecked():
            outputs.append("RAW (DOC cru + DAD 101λ)")
        if self.exp_processed_check.isChecked():
            outputs.append("PROCESSED (DOC net + fraccions)")
        if self.dest_zip_radio.isChecked():
            outputs.append("ZIP")
        if outputs:
            parts.append("FAIR: " + " + ".join(outputs))

        self.fair_info.setText("<br>".join(parts))

    # ------------------------------------------------------------------
    # BP Consolidation
    # ------------------------------------------------------------------

    def _launch_bp_discovery(self, seq_path, sample_names):
        """Llança BPDiscoveryWorker per cercar dades BP."""
        if self._bp_worker and self._bp_worker.isRunning():
            self._bp_worker.wait(2000)

        data_folder = str(Path(seq_path).parent)
        self._bp_worker = BPDiscoveryWorker(seq_path, sample_names, data_folder)
        self._bp_worker.finished.connect(self._on_bp_discovery_finished)
        self._bp_worker.error.connect(self._on_bp_discovery_error)
        self._bp_worker.start()

    def _on_bp_discovery_finished(self, result):
        """Gestiona el resultat de la cerca BP."""
        self._bp_resolved = result
        self._bp_available = result.get("available_bps", [])

        self.bp_combo.blockSignals(True)
        self.bp_combo.clear()
        primary = result.get("primary_bp")
        selected_idx = 0

        if not self._bp_available:
            self.bp_combo.addItem("Cap BP trobada", None)
        else:
            for i, bp in enumerate(self._bp_available):
                bp_name = bp.get("name", "?")
                self.bp_combo.addItem(bp_name, bp.get("path"))
                if primary and bp.get("path") == primary.get("path"):
                    selected_idx = i

        self.bp_combo.setCurrentIndex(selected_idx)
        self.bp_combo.blockSignals(False)

        self._populate_bp_table(result)
        self._auto_generate()

    def _on_bp_discovery_error(self, error_msg):
        """Error durant la cerca BP."""
        logger.error(f"Error BP discovery: {error_msg}")
        self.bp_status_label.setText(f"Error: {error_msg}")
        self.bp_group.setVisible(False)
        self._auto_generate()

    def _populate_bp_table(self, bp_result):
        """Omple la taula de mostres BP."""
        samples = bp_result.get("samples", {})
        n_linked = sum(1 for s in samples.values() if s.get("bp_data"))
        n_total = len(samples)

        self.bp_status_label.setText(f"({n_linked}/{n_total} mostres vinculades)")

        self.bp_table.setRowCount(n_total)

        sorted_names = sorted(
            samples.keys(),
            key=lambda n: (0 if samples[n].get("bp_data") else 1, n)
        )

        for row, name in enumerate(sorted_names):
            sdata = samples[name]
            bp_data = sdata.get("bp_data")

            item_name = QTableWidgetItem(name)
            self.bp_table.setItem(row, 0, item_name)

            if bp_data:
                source = sdata.get("source", "")
                source_tag = " *" if source == "name_search" else ""
                item_status = QTableWidgetItem(f"\u2714{source_tag}")
                item_status.setForeground(QColor(COLOR_SUCCESS))
                item_status.setTextAlignment(Qt.AlignCenter)
                if source == "name_search":
                    item_status.setToolTip(
                        f"Trobat per nom a {bp_data.get('seq_name', '?')}"
                    )
                self.bp_table.setItem(row, 1, item_status)

                replica = bp_data.get("replica", "?")
                item_rep = QTableWidgetItem(f"R{replica}")
                item_rep.setTextAlignment(Qt.AlignCenter)
                self.bp_table.setItem(row, 2, item_rep)

                ppm = bp_data.get("concentration_ppm")
                ppm_text = f"{ppm:.2f}" if ppm else "\u2014"
                item_ppm = QTableWidgetItem(ppm_text)
                item_ppm.setTextAlignment(Qt.AlignCenter)
                self.bp_table.setItem(row, 3, item_ppm)

                snr = bp_data.get("snr_direct")
                snr_text = f"{snr:.0f}" if snr else "\u2014"
                item_snr = QTableWidgetItem(snr_text)
                item_snr.setTextAlignment(Qt.AlignCenter)
                self.bp_table.setItem(row, 4, item_snr)
            else:
                item_status = QTableWidgetItem("\u2718")
                item_status.setForeground(QColor("#bdc3c7"))
                item_status.setTextAlignment(Qt.AlignCenter)
                self.bp_table.setItem(row, 1, item_status)

                for col in range(2, 5):
                    item = QTableWidgetItem("\u2014")
                    item.setForeground(QColor("#bdc3c7"))
                    item.setTextAlignment(Qt.AlignCenter)
                    self.bp_table.setItem(row, col, item)

                for col in range(5):
                    it = self.bp_table.item(row, col)
                    if it:
                        it.setBackground(QColor("#f5f5f5"))

        n_missing = n_total - n_linked
        if n_missing > 0:
            self.bp_info_label.setText(
                f"{n_missing} mostr{'a' if n_missing == 1 else 'es'} sense dades BP"
            )
        else:
            self.bp_info_label.setText("Totes les mostres tenen dades BP vinculades")

    def _on_bp_combo_changed(self, index):
        """Quan l'usuari canvia la BP al dropdown, relança la cerca."""
        if index < 0 or not self._bp_available:
            return

        bp_path = self.bp_combo.currentData()
        if not bp_path:
            return

        self.bp_status_label.setText("Actualitzant...")

        if self._bp_worker and self._bp_worker.isRunning():
            self._bp_worker.wait(2000)

        self._bp_worker = _BPReloadWorker(
            bp_path, self._current_sample_names,
            self._bp_available, self._current_seq_path
        )
        self._bp_worker.finished.connect(self._on_bp_reload_finished)
        self._bp_worker.error.connect(self._on_bp_discovery_error)
        self._bp_worker.start()

    def _on_bp_reload_finished(self, result):
        """Gestiona el resultat de la recàrrega BP."""
        self._bp_resolved = result
        self._populate_bp_table(result)

    # ------------------------------------------------------------------
    # Generate results
    # ------------------------------------------------------------------

    def _update_paths_label(self, seq_path=None):
        """Actualitza el label de paths amb la carpeta destí."""
        if not seq_path:
            seq_path = self._current_seq_path
        if not seq_path:
            return
        dest = f"{seq_path}/RESULTATS"
        self.paths_label.setText(
            f"Excels + CSV \u2192 {dest}/ &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"SUMMARY \u2192 {seq_path}/CHECK/"
        )

    def _browse_dest_folder(self):
        """Obre diàleg per seleccionar carpeta de destí."""
        # Intentar obrir al OneDrive/SharePoint si existeix
        start_dir = ""
        for candidate in [
            os.path.expanduser("~/OneDrive - Universitat de Girona"),
            os.path.expanduser("~/OneDrive"),
            self._current_seq_path,
        ]:
            if candidate and os.path.isdir(candidate):
                start_dir = candidate
                break

        folder = QFileDialog.getExistingDirectory(
            self, "Selecciona carpeta de destí", start_dir
        )
        if folder:
            self.dest_path_edit.setText(folder)
            self._update_paths_label()

    def _auto_generate(self):
        """Auto-genera resultats si encara no s'ha fet per aquesta seqüència."""
        if self._auto_generated:
            return
        self._auto_generated = True
        self._run_generate(silent=True)

    def _run_generate(self, silent=False):
        """Auto-genera Excels individuals + SUMMARY a SEQ/RESULTATS/ + SEQ/CHECK/."""
        processed_data = self.main_window.processed_data
        if not processed_data:
            if not silent:
                QMessageBox.warning(self, "Avís", "No hi ha dades processades.")
            return

        samples_grouped = processed_data.get("samples_grouped", {})
        if not samples_grouped:
            if not silent:
                QMessageBox.warning(self, "Avís", "No hi ha mostres per exportar.")
            return

        seq_path = self.main_window.seq_path or processed_data.get("seq_path", "")
        if not seq_path:
            if not silent:
                QMessageBox.warning(self, "Avís", "No s'ha trobat el path de la seqüència.")
            return

        method = processed_data.get("method", "COLUMN")
        calibration_data = self.main_window.calibration_data

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setVisible(True)
        self.status_label.setText("Generant resultats...")

        if self.worker is not None:
            self.worker.wait()

        # Auto-generation: Excels + SUMMARY only (no CSV/RAW/ZIP)
        self.worker = GenerateWorker(
            samples_grouped, seq_path, calibration_data, method, None,
            bp_resolved=self._bp_resolved,
            generate_pdf=False,
            export_raw=False,
            export_processed=False,
            csv_summary=False,
            csv_separator=";",
            export_zip=False,
            export_metadata=False,
            custom_output_dir=None,
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_progress(self, pct, msg):
        self.progress_bar.setValue(pct)
        self.status_label.setText(msg)

    def _on_finished(self, results):
        if self.worker is not None:
            self.worker.wait()
        self.progress_bar.setVisible(False)
        self.status_label.setVisible(False)

        errors = results.get("errors", [])
        excel_result = results.get("excel_files", {})
        n_exported = excel_result.get("n_exported", 0) if excel_result else 0
        seq_path = self._current_seq_path

        # Update results_frame
        if errors:
            self.results_status_label.setText(
                f"<b>\u26a0 {n_exported} Excels + SUMMARY generats</b> "
                f"amb {len(errors)} error{'s' if len(errors) > 1 else ''}"
            )
            self.results_frame.setStyleSheet("""
                QFrame {
                    background-color: #fef9e7;
                    border: 1px solid #f0d58c;
                    border-radius: 8px;
                }
            """)
        else:
            self.results_status_label.setText(
                f"<b>\u2714 {n_exported} Excels + SUMMARY generats correctament</b>"
            )
            self.results_frame.setStyleSheet("""
                QFrame {
                    background-color: #f0f9f0;
                    border: 1px solid #b8e0b8;
                    border-radius: 8px;
                }
            """)

        self.results_path_label.setText(
            f"Excels \u2192 <code>{seq_path}/RESULTATS/</code> "
            f"&nbsp;&middot;&nbsp; SUMMARY \u2192 <code>{seq_path}/CHECK/</code>"
        )
        self.results_frame.setVisible(True)

        # Save charts from analyze panel
        try:
            analyze_panel = getattr(self.main_window, '_wizard_analyze_panel', None)
            if analyze_panel and hasattr(analyze_panel, 'save_charts'):
                analyze_panel.save_charts(seq_path)
        except Exception as e:
            logger.error(f"Error saving charts: {e}")

        # Write review_result.json
        self._write_review_result(results)

        self.export_completed.emit(results)

    def _on_error(self, error_msg):
        self.progress_bar.setVisible(False)
        self.status_label.setVisible(False)
        self.results_status_label.setText(
            f"<b>\u2718 Error generant resultats</b>"
        )
        self.results_frame.setStyleSheet("""
            QFrame {
                background-color: #fde8e8;
                border: 1px solid #e0b8b8;
                border-radius: 8px;
            }
        """)
        self.results_frame.setVisible(True)
        QMessageBox.critical(self, "Error", f"Error durant la generació:\n{error_msg}")

    # ------------------------------------------------------------------
    # Additional export
    # ------------------------------------------------------------------

    def _browse_zip_dest(self):
        """Obre diàleg per seleccionar destí del fitxer ZIP."""
        seq_name = Path(self._current_seq_path).name if self._current_seq_path else "export"
        default_name = f"{seq_name}_HPSEC_EXPORT.zip"

        start_dir = ""
        for candidate in [
            os.path.expanduser("~/OneDrive - Universitat de Girona"),
            os.path.expanduser("~/OneDrive"),
            self._current_seq_path,
        ]:
            if candidate and os.path.isdir(candidate):
                start_dir = candidate
                break

        path, _ = QFileDialog.getSaveFileName(
            self, "Desa fitxer ZIP", os.path.join(start_dir, default_name),
            "ZIP (*.zip)"
        )
        if path:
            self.zip_path_edit.setText(path)

    def _run_export(self):
        """Exporta contingut seleccionat al destí triat (carpeta o ZIP)."""
        processed_data = self.main_window.processed_data
        if not processed_data:
            QMessageBox.warning(self, "Avís", "No hi ha dades processades.")
            return

        samples_grouped = processed_data.get("samples_grouped", {})
        if not samples_grouped:
            QMessageBox.warning(self, "Avís", "No hi ha mostres per exportar.")
            return

        # Validar destí
        use_folder = self.dest_folder_radio.isChecked()
        use_zip = self.dest_zip_radio.isChecked()
        if not use_folder and not use_zip:
            QMessageBox.warning(self, "Avís",
                                "Selecciona un destí: carpeta o ZIP.")
            return

        custom_dir = None
        zip_dest = None

        if use_folder:
            custom_dir = self.dest_path_edit.text().strip()
            if not custom_dir:
                QMessageBox.warning(self, "Avís", "Selecciona una carpeta de destí.")
                return

        if use_zip:
            zip_dest = self.zip_path_edit.text().strip()
            if not zip_dest:
                QMessageBox.warning(self, "Avís", "Selecciona un destí pel fitxer ZIP.")
                return
            # ZIP: export to temp dir first, then create ZIP at zip_dest
            if not custom_dir:
                import tempfile
                custom_dir = tempfile.mkdtemp(prefix="hpsec_export_")
                self._export_temp_dir = custom_dir
            self._export_zip_path = zip_dest

        if not use_zip:
            self._export_temp_dir = None
            self._export_zip_path = None

        seq_path = self.main_window.seq_path or processed_data.get("seq_path", "")
        method = processed_data.get("method", "COLUMN")
        calibration_data = self.main_window.calibration_data
        csv_sep = self.csv_separator_combo.currentData() or ";"

        self.export_btn.setEnabled(False)
        self.export_progress.setVisible(True)
        self.export_progress.setValue(0)
        self.export_status_label.setVisible(True)
        self.export_status_label.setText("Exportant...")

        if self._export_worker is not None:
            self._export_worker.wait()

        self._export_worker = GenerateWorker(
            samples_grouped, seq_path, calibration_data, method, None,
            bp_resolved=self._bp_resolved,
            generate_pdf=self.exp_pdf_check.isChecked(),
            export_raw=self.exp_raw_check.isChecked(),
            export_processed=self.exp_processed_check.isChecked(),
            csv_summary=self.exp_csv_summary_check.isChecked(),
            csv_separator=csv_sep,
            export_zip=False,  # handled manually below
            export_metadata=self.exp_metadata_check.isChecked(),
            custom_output_dir=custom_dir,
        )
        self._export_worker.progress.connect(self._on_export_progress)
        self._export_worker.finished.connect(self._on_export_finished)
        self._export_worker.error.connect(self._on_export_error)
        self._export_worker.start()

    def _on_export_progress(self, pct, msg):
        self.export_progress.setValue(pct)
        self.export_status_label.setText(msg)

    def _on_export_finished(self, results):
        if self._export_worker is not None:
            self._export_worker.wait()

        errors = results.get("errors", [])
        excel_result = results.get("excel_files", {})
        n_exported = excel_result.get("n_exported", 0) if excel_result else 0

        # ZIP packaging (if requested)
        if self._export_zip_path and self._export_temp_dir:
            try:
                create_export_zip(self._export_temp_dir, self._export_zip_path)
                # Clean up temp dir
                import shutil
                shutil.rmtree(self._export_temp_dir, ignore_errors=True)
            except Exception as e:
                errors.append(f"ZIP: {e}")

        self.export_btn.setEnabled(True)
        self.export_progress.setVisible(False)

        # Status summary
        extras = []
        n_raw = results.get("n_raw", 0)
        n_proc = results.get("n_processed", 0)
        if n_raw:
            extras.append(f"{n_raw} RAW")
        if n_proc:
            extras.append(f"{n_proc} PROC")
        if results.get("csv_summary"):
            extras.append("CSV")
        if results.get("pdf_report"):
            extras.append("PDF")
        if results.get("metadata"):
            extras.append("meta")
        if self._export_zip_path:
            extras.append("ZIP")
        extras_str = (" + " + " + ".join(extras)) if extras else ""

        if errors:
            self.export_status_label.setText(
                f"\u26a0 {n_exported} fitxers{extras_str} — {len(errors)} error(s)")
            self.export_status_label.setStyleSheet(
                f"color: {COLOR_WARNING}; font-size: 11px;")
            QMessageBox.warning(self, "Avisos",
                                "Errors durant l'export:\n" + "\n".join(errors[:5]))
        else:
            dest_name = self._export_zip_path or self.dest_path_edit.text()
            self.export_status_label.setText(
                f"\u2714 {n_exported} Excels{extras_str} exportats a {Path(dest_name).name}")
            self.export_status_label.setStyleSheet(
                f"color: {COLOR_SUCCESS}; font-size: 11px;")

        self.export_status_label.setVisible(True)
        self._export_temp_dir = None
        self._export_zip_path = None

    def _on_export_error(self, error_msg):
        self.export_btn.setEnabled(True)
        self.export_progress.setVisible(False)
        self.export_status_label.setText("\u2718 Error")
        self.export_status_label.setStyleSheet(f"color: {COLOR_ERROR}; font-size: 11px;")
        self.export_status_label.setVisible(True)
        QMessageBox.critical(self, "Error", f"Error durant l'export:\n{error_msg}")
        # Clean up temp dir if exists
        if self._export_temp_dir:
            import shutil
            shutil.rmtree(self._export_temp_dir, ignore_errors=True)
            self._export_temp_dir = None

    # ------------------------------------------------------------------
    # review_result.json
    # ------------------------------------------------------------------

    def _write_review_result(self, results):
        """Persisteix l'estat de la revisió a review_result.json."""
        try:
            from datetime import datetime

            seq_path = self._current_seq_path
            if not seq_path:
                return

            data_dir = Path(seq_path) / "CHECK" / "data"
            data_dir.mkdir(parents=True, exist_ok=True)

            excel_result = results.get("excel_files", {}) or {}
            n_exported = excel_result.get("n_exported", 0)
            n_skipped = excel_result.get("n_skipped", 0)

            # BP info
            bp_info = {}
            if self._bp_resolved:
                primary = self._bp_resolved.get("primary_bp")
                if primary:
                    bp_info["bp_seq_name"] = primary.get("name")
                    bp_info["bp_seq_path"] = primary.get("path")
                    bp_samples = self._bp_resolved.get("samples", {})
                    bp_info["n_linked"] = sum(
                        1 for s in bp_samples.values() if s.get("bp_data")
                    )
                    bp_analysis = Path(primary["path"]) / "CHECK" / "data" / "analysis_result.json"
                    if bp_analysis.exists():
                        bp_info["bp_analysis_mtime"] = os.path.getmtime(str(bp_analysis))

            # Discarded samples
            processed_data = self.main_window.processed_data or {}
            samples_grouped = processed_data.get("samples_grouped", {})
            discarded = [
                name for name, d in samples_grouped.items()
                if d.get("sample_valid") is False
            ]

            from hpsec_version import SUITE_VERSION
            review_data = {
                "success": not results.get("errors"),
                "timestamp": datetime.now().isoformat(),
                "suite_version": SUITE_VERSION,
                "seq_name": Path(seq_path).name,
                "method": self._current_method,
                "n_exported": n_exported,
                "n_skipped": n_skipped,
                "discarded_samples": discarded,
                "bp_info": bp_info,
                "summary_path": str(Path(seq_path) / "CHECK" / "SUMMARY.xlsx"),
            }

            review_path = data_dir / "review_result.json"
            with open(review_path, 'w', encoding='utf-8') as f:
                _json.dump(review_data, f, indent=2, ensure_ascii=False)

            logger.info(f"review_result.json escrit: {review_path}")
        except Exception as e:
            logger.error(f"Error escrivint review_result.json: {e}")

    # ------------------------------------------------------------------
    # Reset / showEvent
    # ------------------------------------------------------------------

    def reset(self):
        """Reseteja el panel."""
        self.header_label.setText("")
        self.fair_info.setText("")
        self.progress_bar.setVisible(False)
        self.status_label.setVisible(False)
        self.status_label.setText("")
        self.paths_label.setText("")
        self.results_frame.setVisible(False)
        self.results_status_label.setText("")
        self.results_path_label.setText("")
        self._bp_resolved = None
        self._bp_available = []
        self.bp_group.setVisible(False)
        self.bp_table.setRowCount(0)
        self.bp_combo.clear()
        self.bp_status_label.setText("")
        self.bp_info_label.setText("")
        self._auto_generated = False
        self._populated_seq = ""
        # Export addicional
        self.exp_excel_check.setChecked(True)
        self.exp_summary_check.setChecked(True)
        self.exp_pdf_check.setChecked(False)
        self.exp_raw_check.setChecked(False)
        self.exp_processed_check.setChecked(False)
        self.exp_csv_summary_check.setChecked(False)
        self.exp_metadata_check.setChecked(True)
        self.csv_separator_combo.setCurrentIndex(0)
        self.dest_folder_radio.setChecked(False)
        self.dest_zip_radio.setChecked(False)
        self.dest_path_edit.clear()
        self.zip_path_edit.clear()
        self.export_progress.setVisible(False)
        self.export_status_label.setVisible(False)
        self.export_status_label.setText("")

    def showEvent(self, event):
        """Quan es mostra el panel, omplir amb dades actuals."""
        super().showEvent(event)
        processed_data = self.main_window.processed_data
        if not processed_data or not processed_data.get("success"):
            return
        seq_id = processed_data.get("seq_path", "") or processed_data.get("seq_name", "")
        if seq_id and seq_id == self._populated_seq:
            return
        self.populate(processed_data)
