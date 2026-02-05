"""
HPSEC Suite - Analyze Panel (Fase 3)
=====================================

Panel per la fase 3: Anàlisi de mostres.

Funcionalitats:
- Executar anàlisi (detecció anomalies, càlcul àrees)
- Taula de resultats amb mètriques clau
- Selecció de rèplica per DOC i DAD (dropdown)
- Visualització de detalls amb gràfics comparatius

Mètriques mostrades:
- R² Pearson entre rèpliques (DOC Direct, UIB)
- SNR DOC (Direct/UIB)
- SNR DAD (millor i pitjor longitud d'ona)
- Àrees per fraccions (COLUMN) o total (BP)
- Concentració ppm (segons rèplica DOC seleccionada)
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QComboBox,
    QDialog, QGroupBox, QGridLayout, QSplitter, QFrame,
    QAbstractItemView, QProgressBar, QMessageBox, QScrollArea
)
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QFont, QColor, QBrush

import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from hpsec_analyze import analyze_sequence, save_analysis_result, load_analysis_result
from gui.widgets.styles import (
    PANEL_MARGINS, PANEL_SPACING, STYLE_WARNING_BAR, STYLE_WARNING_TEXT,
    STYLE_LABEL_SECONDARY, COLOR_SUCCESS, COLOR_WARNING, COLOR_ERROR,
    create_title_font, apply_panel_layout, create_empty_state_widget
)

# Matplotlib per gràfics
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


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
            import traceback
            self.error.emit(f"{str(e)}\n\n{traceback.format_exc()}")


class AnalyzePanel(QWidget):
    """Panel d'anàlisi de mostres (Fase 3)."""

    analyze_completed = Signal(dict)

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.samples_grouped = {}
        self.worker = None

        self._setup_ui()

    def _setup_ui(self):
        """Configura la interfície."""
        layout = QVBoxLayout(self)
        apply_panel_layout(layout)

        # === HEADER ===
        header_layout = QHBoxLayout()

        title = QLabel("Anàlisi de Mostres")
        title.setFont(create_title_font())
        header_layout.addWidget(title)

        header_layout.addStretch()

        # Botó analitzar (amagat - l'acció és al header del wizard)
        self.analyze_btn = QPushButton("▶ Analitzar")
        self.analyze_btn.setStyleSheet("""
            QPushButton {
                background-color: #27AE60; color: white;
                border: none; border-radius: 4px;
                padding: 8px 20px; font-weight: bold;
            }
            QPushButton:hover { background-color: #219A52; }
            QPushButton:disabled { background-color: #BDC3C7; }
        """)
        self.analyze_btn.clicked.connect(self._run_analyze)
        self.analyze_btn.setVisible(False)  # Amagat - acció al header del wizard
        header_layout.addWidget(self.analyze_btn)

        layout.addLayout(header_layout)

        # === BARRA D'AVISOS (consistent per tots els panels) ===
        self.warnings_bar = QFrame()
        self.warnings_bar.setVisible(False)
        self.warnings_bar.setStyleSheet(STYLE_WARNING_BAR)
        warnings_bar_layout = QHBoxLayout(self.warnings_bar)
        warnings_bar_layout.setContentsMargins(12, 8, 12, 8)

        self.warnings_icon = QLabel("⚠")
        self.warnings_icon.setStyleSheet("font-size: 16px; border: none;")
        warnings_bar_layout.addWidget(self.warnings_icon)

        self.warnings_text = QLabel()
        self.warnings_text.setStyleSheet(STYLE_WARNING_TEXT + " border: none;")
        self.warnings_text.setWordWrap(True)
        warnings_bar_layout.addWidget(self.warnings_text, 1)

        layout.addWidget(self.warnings_bar)

        # === INFO PANEL (professional) ===
        self.info_frame = QFrame()
        self.info_frame.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border: 1px solid #e9ecef;
                border-radius: 6px;
            }
        """)
        info_layout = QHBoxLayout(self.info_frame)
        info_layout.setContentsMargins(16, 12, 16, 12)
        info_layout.setSpacing(24)

        # Columna 1: Dades importades
        self.import_info = QLabel()
        self.import_info.setStyleSheet("border: none;")
        info_layout.addWidget(self.import_info)

        # Separador vertical
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.VLine)
        sep1.setStyleSheet("background-color: #dee2e6; border: none; max-width: 1px;")
        info_layout.addWidget(sep1)

        # Columna 2: Calibració
        self.cal_info = QLabel()
        self.cal_info.setStyleSheet("border: none;")
        info_layout.addWidget(self.cal_info)

        info_layout.addStretch()

        # Indicador d'estat
        self.status_indicator = QLabel()
        self.status_indicator.setStyleSheet("border: none;")
        info_layout.addWidget(self.status_indicator)

        layout.addWidget(self.info_frame)

        # Empty state (quan no hi ha dades)
        self.empty_state = create_empty_state_widget(
            "🔬",
            "No hi ha dades importades",
            "Vés a la pestanya «1. Importar» per carregar les dades de la seqüència."
        )
        self.empty_state.setVisible(False)
        layout.addWidget(self.empty_state)

        # Frame antic per missatges d'error (ocult normalment)
        self.status_frame = QFrame()
        self.status_frame.setVisible(False)
        status_layout = QVBoxLayout(self.status_frame)
        status_layout.setContentsMargins(12, 8, 12, 8)
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        status_layout.addWidget(self.status_label)
        layout.addWidget(self.status_frame)

        # === PROGRESS ===
        self.progress_frame = QFrame()
        self.progress_frame.setVisible(False)
        progress_layout = QVBoxLayout(self.progress_frame)
        progress_layout.setContentsMargins(0, 0, 0, 0)

        self.progress_bar = QProgressBar()
        self.progress_label = QLabel("Preparant...")
        progress_layout.addWidget(self.progress_label)
        progress_layout.addWidget(self.progress_bar)

        layout.addWidget(self.progress_frame)

        # === TAULA DE RESULTATS ===
        self.results_frame = QFrame()
        self.results_frame.setVisible(False)
        results_layout = QVBoxLayout(self.results_frame)
        results_layout.setContentsMargins(0, 0, 0, 0)
        results_layout.setSpacing(8)

        # === SELECTOR DOC / DAD ===
        selector_frame = QFrame()
        selector_frame.setStyleSheet("""
            QFrame { background-color: #f8f9fa; border-radius: 6px; }
        """)
        selector_layout = QHBoxLayout(selector_frame)
        selector_layout.setContentsMargins(12, 8, 12, 8)

        self.doc_btn = QPushButton("DOC (Direct/UIB)")
        self.doc_btn.setCheckable(True)
        self.doc_btn.setChecked(True)
        self.doc_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498DB; color: white;
                border: none; border-radius: 4px;
                padding: 8px 16px; font-weight: bold;
            }
            QPushButton:!checked {
                background-color: #e9ecef; color: #495057;
            }
            QPushButton:hover:!checked { background-color: #dee2e6; }
        """)
        self.doc_btn.clicked.connect(lambda: self._switch_view("DOC"))
        selector_layout.addWidget(self.doc_btn)

        self.dad_btn = QPushButton("DAD (6λ)")
        self.dad_btn.setCheckable(True)
        self.dad_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498DB; color: white;
                border: none; border-radius: 4px;
                padding: 8px 16px; font-weight: bold;
            }
            QPushButton:!checked {
                background-color: #e9ecef; color: #495057;
            }
            QPushButton:hover:!checked { background-color: #dee2e6; }
        """)
        self.dad_btn.clicked.connect(lambda: self._switch_view("DAD"))
        selector_layout.addWidget(self.dad_btn)

        selector_layout.addStretch()

        # Llegenda
        legend = QLabel(
            "<span style='color:#27AE60'>●</span> OK &nbsp;"
            "<span style='color:#F39C12'>●</span> Warning &nbsp;"
            "<span style='color:#E74C3C'>●</span> Error"
        )
        legend.setStyleSheet("color: #666;")
        selector_layout.addWidget(legend)

        results_layout.addWidget(selector_frame)

        # === TAULA DOC ===
        self.doc_table = QTableWidget()
        self.doc_table.setColumnCount(10)
        self.doc_table.setHorizontalHeaderLabels([
            "Mostra", "Rep", "A_Direct", "ppm_D", "A_UIB", "ppm_U",
            "R²", "SNR_D", "SNR_U", "Estat"
        ])
        self._configure_table(self.doc_table)
        self._configure_doc_columns()
        results_layout.addWidget(self.doc_table)

        # === TAULA DAD ===
        self.dad_table = QTableWidget()
        self.dad_table.setColumnCount(11)
        self.dad_table.setHorizontalHeaderLabels([
            "Mostra", "Rep", "A_254", "SNR_220", "SNR_252", "SNR_254",
            "SNR_272", "SNR_290", "SNR_362", "R²_min", "Estat"
        ])
        self._configure_table(self.dad_table)
        self._configure_dad_columns()
        self.dad_table.setVisible(False)
        results_layout.addWidget(self.dad_table)

        # Mantenir referència per compatibilitat
        self.results_table = self.doc_table


        # Connectar doble clic per veure detall
        self.doc_table.doubleClicked.connect(self._on_table_double_click)
        self.dad_table.doubleClicked.connect(self._on_table_double_click)

        # Resum estadístic amb botó detall
        self.stats_frame = QFrame()
        self.stats_frame.setStyleSheet("background-color: #f8f9fa; border-radius: 4px; padding: 8px;")
        stats_layout = QHBoxLayout(self.stats_frame)
        stats_layout.setContentsMargins(12, 8, 12, 8)

        self.stats_label = QLabel()
        self.stats_label.setFont(QFont("Segoe UI", 10))
        stats_layout.addWidget(self.stats_label)

        stats_layout.addStretch()

        # Botó detall
        self.detail_btn = QPushButton("📊 Detall")
        self.detail_btn.setEnabled(False)
        self.detail_btn.setToolTip("Mostra gràfics i estadístiques detallades de la mostra seleccionada")
        self.detail_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498DB; color: white;
                border: none; border-radius: 4px;
                padding: 6px 14px; font-weight: bold;
            }
            QPushButton:hover { background-color: #2980B9; }
            QPushButton:disabled { background-color: #BDC3C7; }
        """)
        self.detail_btn.clicked.connect(self._on_detail_clicked)
        stats_layout.addWidget(self.detail_btn)

        results_layout.addWidget(self.stats_frame)

        layout.addWidget(self.results_frame, 1)  # Stretch

        # === BOTONS NAVEGACIÓ ===
        nav_layout = QHBoxLayout()
        nav_layout.addStretch()

        self.next_btn = QPushButton("Següent: Consolidar →")
        self.next_btn.setEnabled(False)
        self.next_btn.setStyleSheet("font-weight: bold; padding: 8px 16px;")
        self.next_btn.clicked.connect(self._go_next)
        nav_layout.addWidget(self.next_btn)

        layout.addLayout(nav_layout)

    def _configure_table(self, table):
        """Configura estil comú per les taules."""
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.SingleSelection)
        table.setAlternatingRowColors(True)
        table.verticalHeader().setVisible(False)
        # Connectar selecció per habilitar botó detall
        table.itemSelectionChanged.connect(self._on_table_selection_changed)
        table.setStyleSheet("""
            QTableWidget {
                gridline-color: #ddd;
                background-color: white;
                alternate-background-color: #f9f9f9;
            }
            QTableWidget::item { padding: 4px 6px; }
            QTableWidget::item:selected {
                background-color: #E3F2FD;
                color: black;
            }
            QHeaderView::section {
                background-color: #f5f5f5;
                padding: 6px;
                border: none;
                border-bottom: 2px solid #ddd;
                font-weight: bold;
                font-size: 11px;
            }
        """)

    def _configure_doc_columns(self):
        """Configura columnes de la taula DOC."""
        header = self.doc_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)  # Mostra
        for i in range(1, 10):
            header.setSectionResizeMode(i, QHeaderView.Fixed)
        # Mides compactes: Rep, A_Dir, ppm_D, A_UIB, ppm_U, R², SNR_D, SNR_U, Estat
        widths = [45, 65, 55, 65, 55, 55, 55, 55, 40]
        for i, w in enumerate(widths):
            self.doc_table.setColumnWidth(i + 1, w)

    def _configure_dad_columns(self):
        """Configura columnes de la taula DAD."""
        header = self.dad_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)  # Mostra
        for i in range(1, 11):
            header.setSectionResizeMode(i, QHeaderView.Fixed)
        # Mides compactes: Rep, A_254, SNR_220..362 (6), R²_min, Estat
        widths = [45, 55, 45, 45, 45, 45, 45, 45, 55, 40]
        for i, w in enumerate(widths):
            self.dad_table.setColumnWidth(i + 1, w)

    def _switch_view(self, view):
        """Canvia entre vista DOC i DAD."""
        if view == "DOC":
            self.doc_btn.setChecked(True)
            self.dad_btn.setChecked(False)
            self.doc_table.setVisible(True)
            self.dad_table.setVisible(False)
        else:
            self.doc_btn.setChecked(False)
            self.dad_btn.setChecked(True)
            self.doc_table.setVisible(False)
            self.dad_table.setVisible(True)

    def showEvent(self, event):
        """Es crida quan el panel es fa visible."""
        super().showEvent(event)
        self._check_existing_analysis()
        self._update_status()

    def reset(self):
        """Reinicia el panel al seu estat inicial."""
        self.samples_grouped = {}
        self.worker = None
        self.doc_table.setRowCount(0)
        self.dad_table.setRowCount(0)
        self.warnings_bar.setVisible(False)
        self.warnings_text.setText("")
        self.empty_state.setVisible(True)
        self.info_frame.setVisible(False)
        self.status_frame.setVisible(False)
        self.progress_frame.setVisible(False)
        self.progress_bar.setValue(0)
        self.results_frame.setVisible(False)
        self.analyze_btn.setEnabled(True)
        self.next_btn.setEnabled(False)
        self.detail_btn.setEnabled(False)
        self.stats_label.setText("")
        self._switch_view("DOC")

    def _check_existing_analysis(self):
        """Comprova si existeix anàlisi prèvia i la carrega automàticament."""
        seq_path = self.main_window.seq_path
        if not seq_path:
            return

        # Si ja tenim resultats carregats, no tornar a carregar
        if self.samples_grouped:
            return

        try:
            existing_analysis = load_analysis_result(seq_path)
            if existing_analysis and existing_analysis.get("success"):
                self._load_existing_analysis(existing_analysis)
        except Exception as e:
            print(f"[WARNING] Error comprovant anàlisi existent: {e}")

    def _load_existing_analysis(self, result):
        """Carrega una anàlisi existent."""
        # Processar resultats (similar a _on_finished)
        self.samples_grouped = result.get("samples_analyzed", {})

        if self.samples_grouped:
            self._populate_table()
            self.status_frame.setVisible(False)
            self.results_frame.setVisible(True)
            self.next_btn.setEnabled(True)
            self.main_window.set_status("Anàlisi carregada des de fitxer existent", 3000)

            # Emetre senyal
            self.analyze_completed.emit(result)

    def _update_status(self):
        """Actualitza l'indicador d'estat amb format professional."""
        imported_data = self.main_window.imported_data
        calibration_data = self.main_window.calibration_data

        if not imported_data:
            # Mostrar empty state, amagar info frame
            self.info_frame.setVisible(False)
            self.status_frame.setVisible(False)
            self.empty_state.setVisible(True)
            self.analyze_btn.setEnabled(False)
            return

        # Amagar empty state, mostrar info frame
        self.empty_state.setVisible(False)
        self.info_frame.setVisible(True)
        self.status_frame.setVisible(False)

        # Comptar mostres
        samples = imported_data.get("samples", {})
        n_samples = len(samples)
        n_replicas = sum(len(reps) for reps in samples.values())
        method = imported_data.get("method", "-")
        data_mode = imported_data.get("data_mode", "-")

        # Info importació
        self.import_info.setText(
            f"<span style='color: #6c757d; font-size: 10px;'>DADES</span><br>"
            f"<b style='font-size: 13px;'>{n_samples}</b> <span style='color: #495057;'>mostres</span> · "
            f"<b>{n_replicas}</b> <span style='color: #495057;'>rèpliques</span><br>"
            f"<span style='color: #6c757d; font-size: 10px;'>{method} / {data_mode}</span>"
        )

        # Info calibració
        if calibration_data and calibration_data.get("success"):
            khp_conc = calibration_data.get("khp_conc", 0)
            rf = calibration_data.get("rf_direct", 0) or calibration_data.get("rf", 0)
            self.cal_info.setText(
                f"<span style='color: #6c757d; font-size: 10px;'>CALIBRACIÓ</span><br>"
                f"<span style='color: #27AE60;'>✓</span> <b style='font-size: 13px;'>KHP {khp_conc:.0f}ppm</b><br>"
                f"<span style='color: #6c757d; font-size: 10px;'>RF: {rf:.0f}</span>"
            )
        else:
            self.cal_info.setText(
                f"<span style='color: #6c757d; font-size: 10px;'>CALIBRACIÓ</span><br>"
                f"<span style='color: #E67E22;'>⚠</span> <span style='color: #856404;'>No disponible</span><br>"
                f"<span style='color: #6c757d; font-size: 10px;'>S'usaran valors per defecte</span>"
            )

        # Indicador d'estat
        self.status_indicator.setText(
            f"<span style='background-color: #d4edda; color: #155724; "
            f"padding: 4px 12px; border-radius: 12px; font-size: 11px;'>"
            f"Llest per analitzar</span>"
        )

        self.analyze_btn.setEnabled(True)

    def _run_analyze(self):
        """Executa l'anàlisi."""
        imported_data = self.main_window.imported_data
        calibration_data = self.main_window.calibration_data

        print(f"[DEBUG] _run_analyze: imported_data={imported_data is not None}")
        print(f"[DEBUG] _run_analyze: calibration_data={calibration_data is not None}")

        if not imported_data:
            QMessageBox.warning(self, "Avís", "No hi ha dades importades.\n\nVés a la pestanya '1. Importar' primer.")
            return

        # Verificar que hi ha mostres
        samples = imported_data.get("samples", {})
        if not samples:
            QMessageBox.warning(self, "Avís", "No s'han trobat mostres a les dades importades.")
            return

        print(f"[DEBUG] Iniciant anàlisi amb {len(samples)} mostres")

        # Mostrar progrés
        self.analyze_btn.setEnabled(False)
        self.progress_frame.setVisible(True)
        self.progress_bar.setValue(0)
        self.results_frame.setVisible(False)

        # Iniciar worker
        self.worker = AnalyzeWorker(imported_data, calibration_data)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_progress(self, msg, pct):
        """Actualitza el progrés."""
        self.progress_label.setText(msg)
        self.progress_bar.setValue(pct)

    def _on_finished(self, result):
        """Gestiona la finalització de l'anàlisi."""
        print(f"[DEBUG] _on_finished: result success={result.get('success') if result else None}")

        self.progress_frame.setVisible(False)
        self.analyze_btn.setEnabled(True)

        if not result or not result.get("success"):
            error_msg = result.get("error", "Error desconegut") if result else "Resultat buit"
            print(f"[DEBUG] Error: {error_msg}")
            QMessageBox.critical(self, "Error", f"Error durant l'anàlisi:\n{error_msg}")
            self._update_status()
            return

        # Amagar frame d'estat, mostrar resultats
        self.status_frame.setVisible(False)

        print(f"[DEBUG] samples_grouped keys: {list(self.samples_grouped.keys())}")
        print(f"[DEBUG] n_samples_grouped: {len(self.samples_grouped)}")

        # Debug primer sample
        if self.samples_grouped:
            first_name = list(self.samples_grouped.keys())[0]
            first = self.samples_grouped[first_name]
            print(f"[DEBUG] First sample keys: {list(first.keys())}")
            if first.get("replicas"):
                rep_keys = list(first["replicas"].keys())
                print(f"[DEBUG] Replica keys: {rep_keys}")
                if rep_keys:
                    first_rep = first["replicas"][rep_keys[0]]
                    print(f"[DEBUG] First replica keys: {list(first_rep.keys())}")
                    print(f"[DEBUG] areas: {first_rep.get('areas')}")
                    print(f"[DEBUG] snr_info: {first_rep.get('snr_info')}")
                    print(f"[DEBUG] snr_info_dad: {first_rep.get('snr_info_dad')}")
            print(f"[DEBUG] comparison: {first.get('comparison')}")
            print(f"[DEBUG] quantification: {first.get('quantification')}")

        # Guardar resultat
        self.main_window.processed_data = result
        self.samples_grouped = result.get("samples_grouped", {})

        print(f"[DEBUG] result keys: {list(result.keys())}")
        print(f"[DEBUG] samples_grouped from result: {result.get('samples_grouped') is not None}")

        # Guardar a JSON
        save_analysis_result(result)

        # Mostrar resultats
        self._populate_table()
        self.results_frame.setVisible(True)

        # Mostrar avisos si n'hi ha
        self._show_warnings(result)

        # Habilitar navegació
        self.next_btn.setEnabled(True)

        # Emetre senyal
        self.analyze_completed.emit(result)

    def _show_warnings(self, result):
        """Mostra avisos a la barra superior si n'hi ha."""
        warnings = result.get("warnings", [])
        anomalies = result.get("anomalies_summary", {})

        # Comptar anomalies
        n_timeouts = anomalies.get("timeouts", 0)
        n_batman = anomalies.get("batman", 0)
        n_low_snr = anomalies.get("low_snr", 0)

        warning_parts = []
        if n_timeouts > 0:
            warning_parts.append(f"{n_timeouts} timeouts")
        if n_batman > 0:
            warning_parts.append(f"{n_batman} batman")
        if n_low_snr > 0:
            warning_parts.append(f"{n_low_snr} SNR baix")
        if warnings:
            warning_parts.extend(warnings[:2])  # Màxim 2 warnings addicionals

        if warning_parts:
            self.warnings_bar.setVisible(True)
            self.warnings_text.setText(
                f"<b>{len(warning_parts)} avisos:</b> " + " · ".join(warning_parts)
            )
        else:
            self.warnings_bar.setVisible(False)

    def _on_error(self, error_msg):
        """Gestiona errors."""
        self.progress_frame.setVisible(False)
        self.analyze_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", f"Error durant l'anàlisi:\n{error_msg}")

    def _populate_table(self):
        """Omple les dues taules (DOC i DAD) amb els resultats."""
        stats = self._populate_doc_table()
        self._populate_dad_table()

        # Actualitzar estadístiques
        n_ok, n_warning, n_error = stats
        total = n_ok + n_warning + n_error
        self.stats_label.setText(
            f"<b>Total:</b> {total} mostres &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"<span style='color:#27AE60'>●</span> OK: {n_ok} &nbsp;&nbsp;"
            f"<span style='color:#F39C12'>●</span> Warning: {n_warning} &nbsp;&nbsp;"
            f"<span style='color:#E74C3C'>●</span> Error: {n_error}"
        )

    def _populate_doc_table(self):
        """Omple la taula DOC (Direct/UIB)."""
        self.doc_table.setRowCount(0)
        n_ok, n_warning, n_error = 0, 0, 0

        for sample_name in sorted(self.samples_grouped.keys()):
            sample_data = self.samples_grouped[sample_name]
            row = self.doc_table.rowCount()
            self.doc_table.insertRow(row)

            replicas = sample_data.get("replicas") or {}
            comparison = sample_data.get("comparison") or {}
            recommendation = sample_data.get("recommendation") or {}
            selected = sample_data.get("selected") or {"doc": "1", "dad": "1"}
            quantification = sample_data.get("quantification") or {}

            doc_rec = (recommendation.get("doc") or {}).get("replica", "1")
            doc_sel = selected.get("doc", doc_rec)
            rep_data = replicas.get(doc_sel, {})

            # Col 0: Mostra
            item_name = QTableWidgetItem(sample_name)
            item_name.setData(Qt.UserRole, sample_name)
            self.doc_table.setItem(row, 0, item_name)

            # Col 1: Selector de rèplica DOC (sense estrella - s'entén que és la seleccionada)
            replica_combo = QComboBox()
            replica_combo.setStyleSheet("QComboBox { border: none; background: transparent; padding: 2px; }")
            for rep_num in sorted(replicas.keys()):
                replica_combo.addItem(f"R{rep_num}", rep_num)
                if rep_num == doc_sel:
                    replica_combo.setCurrentIndex(replica_combo.count() - 1)
            replica_combo.currentIndexChanged.connect(
                lambda idx, name=sample_name: self._on_doc_replica_changed(name)
            )
            self.doc_table.setCellWidget(row, 1, replica_combo)

            # Col 2: Àrea DOC Direct
            areas = rep_data.get("areas") or {}
            doc_areas = areas.get("DOC") or {}
            area_direct = doc_areas.get("total", 0)
            self.doc_table.setItem(row, 2, QTableWidgetItem(f"{area_direct:.0f}" if area_direct else "-"))

            # Col 3: ppm Direct
            ppm_direct = quantification.get("concentration_ppm_direct") or quantification.get("concentration_ppm")
            self.doc_table.setItem(row, 3, QTableWidgetItem(f"{ppm_direct:.2f}" if ppm_direct else "-"))

            # Col 4: Àrea DOC UIB
            areas_uib = rep_data.get("areas_uib") or {}
            area_uib = areas_uib.get("total", 0)
            self.doc_table.setItem(row, 4, QTableWidgetItem(f"{area_uib:.0f}" if area_uib else "-"))

            # Col 5: ppm UIB
            ppm_uib = quantification.get("concentration_ppm_uib")
            self.doc_table.setItem(row, 5, QTableWidgetItem(f"{ppm_uib:.2f}" if ppm_uib else "-"))

            # Col 6: R² DOC (Pearson entre rèpliques)
            r2_doc = comparison.get("doc", {}).get("pearson", 0) if comparison else 0
            r2_item = QTableWidgetItem(f"{r2_doc:.3f}" if r2_doc > 0 else "-")
            if 0 < r2_doc < 0.995:
                r2_item.setForeground(QBrush(QColor("#F39C12")))
            self.doc_table.setItem(row, 6, r2_item)

            # Col 7: SNR Direct
            snr_info = rep_data.get("snr_info", {})
            snr_direct = snr_info.get("snr_direct", 0)
            self.doc_table.setItem(row, 7, QTableWidgetItem(f"{snr_direct:.0f}" if snr_direct else "-"))

            # Col 8: SNR UIB
            snr_uib = snr_info.get("snr_uib", 0)
            self.doc_table.setItem(row, 8, QTableWidgetItem(f"{snr_uib:.0f}" if snr_uib else "-"))

            # Col 9: Estat
            anomalies = rep_data.get("anomalies", [])
            warnings = comparison.get("doc", {}).get("warnings", []) if comparison else []
            timeout_info = rep_data.get("timeout_info", {})

            status_parts = []
            if timeout_info.get("n_timeouts", 0) > 0:
                status_parts.append("⚠T")
            if any("BATMAN" in a for a in anomalies):
                status_parts.append("🦇")
            if "LOW_SNR" in anomalies:
                status_parts.append("↓")

            if anomalies:
                status_color = "#E74C3C"
                n_error += 1
            elif warnings:
                status_color = "#F39C12"
                n_warning += 1
            else:
                status_color = "#27AE60"
                n_ok += 1

            status_text = " ".join(status_parts) if status_parts else "✓"
            status_item = QTableWidgetItem(status_text)
            status_item.setForeground(QBrush(QColor(status_color)))
            status_item.setTextAlignment(Qt.AlignCenter)
            tooltip = []
            if anomalies:
                tooltip.extend(anomalies)
            if warnings:
                tooltip.extend(warnings)
            if timeout_info.get("n_timeouts", 0) > 0:
                tooltip.append(f"Timeouts: {timeout_info.get('n_timeouts')} ({timeout_info.get('severity', 'INFO')})")
            status_item.setToolTip("\n".join(tooltip) if tooltip else "OK")
            self.doc_table.setItem(row, 9, status_item)

        return n_ok, n_warning, n_error

    def _populate_dad_table(self):
        """Omple la taula DAD (6 longituds d'ona)."""
        self.dad_table.setRowCount(0)

        for sample_name in sorted(self.samples_grouped.keys()):
            sample_data = self.samples_grouped[sample_name]
            row = self.dad_table.rowCount()
            self.dad_table.insertRow(row)

            replicas = sample_data.get("replicas") or {}
            comparison = sample_data.get("comparison") or {}
            recommendation = sample_data.get("recommendation") or {}
            selected = sample_data.get("selected") or {"doc": "1", "dad": "1"}

            dad_rec = (recommendation.get("dad") or {}).get("replica", "1")
            dad_sel = selected.get("dad", dad_rec)
            rep_data = replicas.get(dad_sel, {})

            # Col 0: Mostra
            item_name = QTableWidgetItem(sample_name)
            item_name.setData(Qt.UserRole, sample_name)
            self.dad_table.setItem(row, 0, item_name)

            # Col 1: Selector de rèplica DAD (sense estrella - s'entén que és la seleccionada)
            replica_combo = QComboBox()
            replica_combo.setStyleSheet("QComboBox { border: none; background: transparent; padding: 2px; }")
            for rep_num in sorted(replicas.keys()):
                replica_combo.addItem(f"R{rep_num}", rep_num)
                if rep_num == dad_sel:
                    replica_combo.setCurrentIndex(replica_combo.count() - 1)
            replica_combo.currentIndexChanged.connect(
                lambda idx, name=sample_name: self._on_dad_replica_changed(name)
            )
            self.dad_table.setCellWidget(row, 1, replica_combo)

            # Col 2: Àrea 254nm
            areas = rep_data.get("areas") or {}
            area_254 = areas.get("A254", {}).get("total", 0)
            self.dad_table.setItem(row, 2, QTableWidgetItem(f"{area_254:.0f}" if area_254 else "-"))

            # Col 3-8: SNR per λ (220, 252, 254, 272, 290, 362)
            snr_dad = rep_data.get("snr_info_dad", {})
            wavelengths = ['220', '252', '254', '272', '290', '362']
            for i, wl in enumerate(wavelengths):
                snr_val = snr_dad.get(f"A{wl}", {}).get("snr", 0) if isinstance(snr_dad.get(f"A{wl}"), dict) else 0
                self.dad_table.setItem(row, 3 + i, QTableWidgetItem(f"{snr_val:.0f}" if snr_val else "-"))

            # Col 9: R²_min (mínim entre rèpliques per λ)
            dad_comparison = comparison.get("dad", {})
            r2_min = dad_comparison.get("pearson_min", 0)
            wl_min = dad_comparison.get("wavelength_min", "")
            if r2_min and r2_min > 0:
                r2_text = f"{r2_min:.3f}"
                if wl_min:
                    r2_text += f"({wl_min})"
            else:
                # Fallback a pearson_254
                r2_254 = dad_comparison.get("pearson_254", 0)
                r2_text = f"{r2_254:.3f}" if r2_254 > 0 else "-"

            r2_item = QTableWidgetItem(r2_text)
            if 0 < r2_min < 0.995:
                r2_item.setForeground(QBrush(QColor("#F39C12")))
            self.dad_table.setItem(row, 9, r2_item)

            # Col 10: Estat
            dad_warnings = dad_comparison.get("warnings", [])
            if dad_warnings:
                status_item = QTableWidgetItem("●")
                status_item.setForeground(QBrush(QColor("#F39C12")))
                status_item.setToolTip("\n".join(dad_warnings))
            else:
                status_item = QTableWidgetItem("●")
                status_item.setForeground(QBrush(QColor("#27AE60")))
                status_item.setToolTip("OK")
            status_item.setTextAlignment(Qt.AlignCenter)
            self.dad_table.setItem(row, 10, status_item)

    def _on_doc_replica_changed(self, sample_name):
        """Gestiona el canvi de rèplica DOC."""
        if sample_name not in self.samples_grouped:
            return

        for row in range(self.doc_table.rowCount()):
            item = self.doc_table.item(row, 0)
            if item and item.data(Qt.UserRole) == sample_name:
                combo = self.doc_table.cellWidget(row, 1)
                if combo:
                    new_replica = combo.currentData()
                    self.samples_grouped[sample_name]["selected"]["doc"] = new_replica
                    self._update_quantification(sample_name)
                    self._update_doc_row(row, sample_name)
                break

    def _on_dad_replica_changed(self, sample_name):
        """Gestiona el canvi de rèplica DAD."""
        if sample_name not in self.samples_grouped:
            return

        for row in range(self.dad_table.rowCount()):
            item = self.dad_table.item(row, 0)
            if item and item.data(Qt.UserRole) == sample_name:
                combo = self.dad_table.cellWidget(row, 1)
                if combo:
                    new_replica = combo.currentData()
                    self.samples_grouped[sample_name]["selected"]["dad"] = new_replica
                    self._update_dad_row(row, sample_name)
                break

    def _update_doc_row(self, row, sample_name):
        """Actualitza una fila de la taula DOC."""
        sample_data = self.samples_grouped[sample_name]
        selected = sample_data.get("selected", {})
        doc_sel = selected.get("doc", "1")
        replicas = sample_data.get("replicas", {})
        rep_data = replicas.get(doc_sel, {})
        quantification = sample_data.get("quantification", {})

        # Àrea Direct
        areas = rep_data.get("areas") or {}
        doc_areas = areas.get("DOC") or {}
        area_direct = doc_areas.get("total", 0)
        self.doc_table.item(row, 2).setText(f"{area_direct:.0f}" if area_direct else "-")

        # ppm Direct
        ppm_direct = quantification.get("concentration_ppm_direct") or quantification.get("concentration_ppm")
        self.doc_table.item(row, 3).setText(f"{ppm_direct:.2f}" if ppm_direct else "-")

        # Àrea UIB
        areas_uib = rep_data.get("areas_uib") or {}
        area_uib = areas_uib.get("total", 0)
        self.doc_table.item(row, 4).setText(f"{area_uib:.0f}" if area_uib else "-")

        # ppm UIB
        ppm_uib = quantification.get("concentration_ppm_uib")
        self.doc_table.item(row, 5).setText(f"{ppm_uib:.2f}" if ppm_uib else "-")

        # SNR Direct/UIB
        snr_info = rep_data.get("snr_info", {})
        self.doc_table.item(row, 7).setText(f"{snr_info.get('snr_direct', 0):.0f}" if snr_info.get('snr_direct') else "-")
        self.doc_table.item(row, 8).setText(f"{snr_info.get('snr_uib', 0):.0f}" if snr_info.get('snr_uib') else "-")

    def _update_dad_row(self, row, sample_name):
        """Actualitza una fila de la taula DAD."""
        sample_data = self.samples_grouped[sample_name]
        selected = sample_data.get("selected", {})
        dad_sel = selected.get("dad", "1")
        replicas = sample_data.get("replicas", {})
        rep_data = replicas.get(dad_sel, {})

        # Àrea 254
        areas = rep_data.get("areas") or {}
        area_254 = areas.get("A254", {}).get("total", 0)
        self.dad_table.item(row, 2).setText(f"{area_254:.0f}" if area_254 else "-")

        # SNR per λ
        snr_dad = rep_data.get("snr_info_dad", {})
        wavelengths = ['220', '252', '254', '272', '290', '362']
        for i, wl in enumerate(wavelengths):
            snr_val = snr_dad.get(f"A{wl}", {}).get("snr", 0) if isinstance(snr_dad.get(f"A{wl}"), dict) else 0
            self.dad_table.item(row, 3 + i).setText(f"{snr_val:.0f}" if snr_val else "-")

    def _format_snr_dad(self, rep_data):
        """Formata SNR DAD mostrant millor i pitjor WL."""
        snr_dad = rep_data.get("snr_info_dad", {})
        if not snr_dad:
            return "-"

        # Trobar millor i pitjor
        best_wl = None
        best_snr = 0
        worst_wl = None
        worst_snr = float('inf')

        for wl_key, info in snr_dad.items():
            snr = info.get("snr", 0) if isinstance(info, dict) else 0
            if snr > 0:
                if snr > best_snr:
                    best_snr = snr
                    best_wl = wl_key.replace("A", "")
                if snr < worst_snr:
                    worst_snr = snr
                    worst_wl = wl_key.replace("A", "")

        if best_wl and worst_wl and best_wl != worst_wl:
            return f"{best_snr:.0f}({best_wl}) / {worst_snr:.0f}({worst_wl})"
        elif best_wl:
            return f"{best_snr:.0f}({best_wl})"
        else:
            return "-"

    def _update_quantification(self, sample_name):
        """Recalcula la quantificació per una mostra."""
        try:
            from hpsec_analyze import quantify_sample

            sample_data = self.samples_grouped[sample_name]
            selected_doc = sample_data["selected"]["doc"]
            selected_replica = sample_data["replicas"].get(selected_doc)

            if selected_replica:
                calibration_data = self.main_window.calibration_data
                method = self.main_window.processed_data.get("method", "COLUMN")
                mode = "BP" if method == "BP" else "COLUMN"

                quantification = quantify_sample(selected_replica, calibration_data, mode=mode)
                sample_data["quantification"] = quantification
        except Exception as e:
            print(f"Error recalculant quantificació: {e}")

    def _on_table_selection_changed(self):
        """Gestiona canvi de selecció a les taules."""
        # Obtenir taula activa
        table = self.doc_table if self.doc_table.isVisible() else self.dad_table
        selected_rows = table.selectionModel().selectedRows()
        self.detail_btn.setEnabled(len(selected_rows) == 1)

    def _on_detail_clicked(self):
        """Handler per botó detall."""
        # Obtenir taula activa
        table = self.doc_table if self.doc_table.isVisible() else self.dad_table
        selected_rows = table.selectionModel().selectedRows()
        if not selected_rows:
            return

        row = selected_rows[0].row()
        item = table.item(row, 0)
        if item:
            sample_name = item.data(Qt.UserRole) or item.text()
            self._show_detail(sample_name)

    def _on_table_double_click(self, index):
        """Handler per doble clic a la taula."""
        # Obtenir taula activa
        table = self.doc_table if self.doc_table.isVisible() else self.dad_table
        row = index.row()
        item = table.item(row, 0)
        if item:
            sample_name = item.data(Qt.UserRole) or item.text()
            self._show_detail(sample_name)

    def _show_detail(self, sample_name):
        """Mostra el diàleg de detall."""
        if sample_name not in self.samples_grouped:
            return

        method = self.main_window.processed_data.get("method", "COLUMN")
        dialog = SampleDetailDialog(
            sample_name,
            self.samples_grouped[sample_name],
            method,
            parent=self
        )
        dialog.exec()

    def _go_next(self):
        """Navega a la següent fase (Consolidar)."""
        # Actualitzar processed_data
        if self.main_window.processed_data:
            self.main_window.processed_data["samples_grouped"] = self.samples_grouped

        self.analyze_completed.emit(self.main_window.processed_data)


class SampleDetailDialog(QDialog):
    """Diàleg de detall d'una mostra amb gràfics i estadístiques."""

    def __init__(self, sample_name, sample_data, method, parent=None):
        super().__init__(parent)
        self.sample_name = sample_name
        self.sample_data = sample_data
        self.method = method
        self.is_bp = method.upper() == "BP"

        self.setWindowTitle(f"Detall: {sample_name}")
        self.setMinimumSize(1000, 700)
        self.setModal(True)

        self._setup_ui()

    def _setup_ui(self):
        """Configura la interfície."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Splitter principal
        splitter = QSplitter(Qt.Horizontal)

        # === PANEL ESQUERRE: GRÀFICS ===
        graph_widget = QWidget()
        graph_layout = QVBoxLayout(graph_widget)
        graph_layout.setContentsMargins(0, 0, 0, 0)

        if HAS_MATPLOTLIB:
            self.figure = Figure(figsize=(8, 9), dpi=100)
            self.canvas = FigureCanvas(self.figure)
            graph_layout.addWidget(self.canvas)
            self._plot_signals()
        else:
            no_plot = QLabel("Matplotlib no disponible.\nInstal·la matplotlib per veure gràfics.")
            no_plot.setAlignment(Qt.AlignCenter)
            no_plot.setStyleSheet("color: #666; font-style: italic;")
            graph_layout.addWidget(no_plot)

        splitter.addWidget(graph_widget)

        # === PANEL DRET: ESTADÍSTIQUES ===
        stats_scroll = QScrollArea()
        stats_scroll.setWidgetResizable(True)
        stats_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        stats_scroll.setStyleSheet("QScrollArea { border: none; }")

        stats_widget = QWidget()
        stats_layout = QVBoxLayout(stats_widget)
        stats_layout.setContentsMargins(8, 0, 8, 0)
        stats_layout.setSpacing(12)

        # Info general
        info_group = self._create_info_group()
        stats_layout.addWidget(info_group)

        # Comparació rèpliques
        if len(self.sample_data.get("replicas", {})) > 1:
            comparison_group = self._create_comparison_group()
            stats_layout.addWidget(comparison_group)

        # SNR per senyal
        snr_group = self._create_snr_group()
        stats_layout.addWidget(snr_group)

        # Àrees per fraccions (COLUMN)
        if not self.is_bp:
            fractions_group = self._create_fractions_group()
            stats_layout.addWidget(fractions_group)

        stats_layout.addStretch()
        stats_scroll.setWidget(stats_widget)
        splitter.addWidget(stats_scroll)

        splitter.setSizes([650, 350])
        layout.addWidget(splitter)

        # Botó tancar
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        close_btn = QPushButton("Tancar")
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)

        layout.addLayout(btn_layout)

    def _create_info_group(self):
        """Crea el grup d'informació general."""
        group = QGroupBox("Informació General")
        layout = QGridLayout(group)
        layout.setSpacing(8)

        selected = self.sample_data.get("selected", {})
        quantification = self.sample_data.get("quantification", {})

        row = 0

        # Mostra
        layout.addWidget(QLabel("<b>Mostra:</b>"), row, 0)
        layout.addWidget(QLabel(self.sample_name), row, 1)
        row += 1

        # Rèplica seleccionada
        layout.addWidget(QLabel("<b>Rèplica:</b>"), row, 0)
        layout.addWidget(QLabel(f"R{selected.get('doc', '?')}"), row, 1)
        row += 1

        # Mode
        layout.addWidget(QLabel("<b>Mode:</b>"), row, 0)
        layout.addWidget(QLabel(self.method), row, 1)
        row += 1

        # Concentració
        conc = quantification.get("concentration_ppm")
        layout.addWidget(QLabel("<b>Concentració:</b>"), row, 0)
        conc_label = QLabel(f"{conc:.3f} ppm" if conc else "-")
        conc_label.setStyleSheet("font-weight: bold; color: #2E86AB;")
        layout.addWidget(conc_label, row, 1)
        row += 1

        # Àrea total
        area = quantification.get("area_total")
        layout.addWidget(QLabel("<b>Àrea total:</b>"), row, 0)
        layout.addWidget(QLabel(f"{area:.1f}" if area else "-"), row, 1)

        return group

    def _create_comparison_group(self):
        """Crea el grup de comparació entre rèpliques."""
        group = QGroupBox("Comparació R1 vs R2")
        layout = QGridLayout(group)
        layout.setSpacing(8)

        comparison = self.sample_data.get("comparison", {})
        doc_comp = comparison.get("doc", {})

        row = 0

        # Pearson DOC
        pearson = doc_comp.get("pearson", 0)
        layout.addWidget(QLabel("Pearson DOC:"), row, 0)
        p_label = QLabel(f"{pearson:.4f}")
        if pearson > 0 and pearson < 0.995:
            p_label.setStyleSheet("color: #F39C12; font-weight: bold;")
        layout.addWidget(p_label, row, 1)
        row += 1

        # Diferència àrea
        area_diff = doc_comp.get("area_diff_pct", 0)
        layout.addWidget(QLabel("Diff àrea:"), row, 0)
        diff_label = QLabel(f"{area_diff:.1f}%")
        if area_diff > 10:
            diff_label.setStyleSheet("color: #F39C12; font-weight: bold;")
        layout.addWidget(diff_label, row, 1)
        row += 1

        # Warnings
        warnings = doc_comp.get("warnings", [])
        if warnings:
            layout.addWidget(QLabel("<b>Warnings:</b>"), row, 0, 1, 2)
            row += 1
            for w in warnings[:5]:
                w_label = QLabel(f"⚠ {w}")
                w_label.setStyleSheet("color: #F39C12; font-size: 11px;")
                w_label.setWordWrap(True)
                layout.addWidget(w_label, row, 0, 1, 2)
                row += 1

        return group

    def _create_snr_group(self):
        """Crea el grup de SNR per senyal."""
        group = QGroupBox("SNR per Senyal")
        layout = QGridLayout(group)
        layout.setSpacing(8)

        selected = self.sample_data.get("selected", {})
        doc_sel = selected.get("doc", "1")
        rep_data = (self.sample_data.get("replicas") or {}).get(doc_sel, {})

        snr_info = rep_data.get("snr_info", {})
        snr_dad = rep_data.get("snr_info_dad", {})

        row = 0

        # DOC Direct
        snr_direct = snr_info.get("snr_direct", 0)
        layout.addWidget(QLabel("DOC Direct:"), row, 0)
        layout.addWidget(QLabel(f"{snr_direct:.1f}" if snr_direct else "-"), row, 1)
        row += 1

        # DOC UIB
        snr_uib = snr_info.get("snr_uib", 0)
        if snr_uib:
            layout.addWidget(QLabel("DOC UIB:"), row, 0)
            layout.addWidget(QLabel(f"{snr_uib:.1f}"), row, 1)
            row += 1

        # DAD per longitud d'ona
        if snr_dad:
            layout.addWidget(QLabel("<b>DAD:</b>"), row, 0, 1, 2)
            row += 1

            for wl_key in sorted(snr_dad.keys()):
                info = snr_dad[wl_key]
                snr = info.get("snr", 0) if isinstance(info, dict) else 0
                wl = wl_key.replace("A", "")
                layout.addWidget(QLabel(f"  {wl} nm:"), row, 0)
                layout.addWidget(QLabel(f"{snr:.1f}" if snr else "-"), row, 1)
                row += 1

        return group

    def _create_fractions_group(self):
        """Crea el grup d'àrees per fraccions (només COLUMN)."""
        group = QGroupBox("Àrees per Fracció")
        layout = QGridLayout(group)
        layout.setSpacing(8)

        selected = self.sample_data.get("selected", {})
        doc_sel = selected.get("doc", "1")
        rep_data = (self.sample_data.get("replicas") or {}).get(doc_sel, {})

        # Fraccions estan a areas.DOC
        areas = rep_data.get("areas") or {}
        fractions = areas.get("DOC") or {}

        # Header
        layout.addWidget(QLabel("<b>Fracció</b>"), 0, 0)
        layout.addWidget(QLabel("<b>Rang</b>"), 0, 1)
        layout.addWidget(QLabel("<b>DOC</b>"), 0, 2)
        layout.addWidget(QLabel("<b>%</b>"), 0, 3)

        fraction_ranges = {
            "BioP": "0-18",
            "HS": "18-23",
            "BB": "23-30",
            "SB": "30-40",
            "LMW": "40-70",
        }

        total_area = sum(fractions.get(f, 0) for f in fraction_ranges.keys())

        row = 1
        for frac_name, rang in fraction_ranges.items():
            area = fractions.get(frac_name, 0)
            pct = (area / total_area * 100) if total_area > 0 else 0

            layout.addWidget(QLabel(frac_name), row, 0)
            layout.addWidget(QLabel(rang), row, 1)
            layout.addWidget(QLabel(f"{area:.1f}" if area else "-"), row, 2)
            layout.addWidget(QLabel(f"{pct:.1f}%"), row, 3)
            row += 1

        # Total
        layout.addWidget(QLabel("<b>Total</b>"), row, 0)
        layout.addWidget(QLabel("0-70"), row, 1)
        layout.addWidget(QLabel(f"<b>{total_area:.1f}</b>"), row, 2)
        layout.addWidget(QLabel("<b>100%</b>"), row, 3)

        return group

    def _plot_signals(self):
        """Genera els gràfics dels senyals."""
        if not HAS_MATPLOTLIB:
            return

        self.figure.clear()

        replicas = self.sample_data.get("replicas", {})
        if not replicas:
            return

        rep_keys = sorted(replicas.keys())
        colors = {'r1': '#2196F3', 'r2': '#FF5722'}

        # Obtenir dades de les rèpliques
        r1_data = replicas.get(rep_keys[0], {})
        r2_data = replicas.get(rep_keys[1], {}) if len(rep_keys) > 1 else None

        # Crear subplots
        n_plots = 3
        axes = self.figure.subplots(n_plots, 1, sharex=True)

        # === Plot 1: DOC Direct ===
        ax1 = axes[0]
        t1 = r1_data.get("t_doc")
        y1 = r1_data.get("y_doc_net")

        if t1 is not None and y1 is not None:
            t1 = np.asarray(t1)
            y1 = np.asarray(y1)
            ax1.plot(t1, y1, color=colors['r1'], label=f'R{rep_keys[0]}', linewidth=1)

            if r2_data:
                t2 = r2_data.get("t_doc")
                y2 = r2_data.get("y_doc_net")
                if t2 is not None and y2 is not None:
                    t2 = np.asarray(t2)
                    y2 = np.asarray(y2)
                    ax1.plot(t2, y2, color=colors['r2'], label=f'R{rep_keys[1]}',
                            linewidth=1, linestyle='--', alpha=0.8)

        ax1.set_ylabel("DOC Direct (mAU)", fontsize=9)
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_title("DOC Direct", fontsize=10, fontweight='bold', loc='left')

        # === Plot 2: DOC UIB ===
        ax2 = axes[1]
        y1_uib = r1_data.get("y_doc_uib_net")

        if y1_uib is not None and t1 is not None:
            y1_uib = np.asarray(y1_uib)
            ax2.plot(t1, y1_uib, color=colors['r1'], label=f'R{rep_keys[0]}', linewidth=1)

            if r2_data:
                y2_uib = r2_data.get("y_doc_uib_net")
                if y2_uib is not None:
                    y2_uib = np.asarray(y2_uib)
                    t2 = r2_data.get("t_doc")
                    if t2 is not None:
                        t2 = np.asarray(t2)
                        ax2.plot(t2, y2_uib, color=colors['r2'], label=f'R{rep_keys[1]}',
                                linewidth=1, linestyle='--', alpha=0.8)

            ax2.set_title("DOC UIB", fontsize=10, fontweight='bold', loc='left')
        else:
            ax2.text(0.5, 0.5, "UIB no disponible", ha='center', va='center',
                    transform=ax2.transAxes, fontsize=10, color='#666')
            ax2.set_title("DOC UIB", fontsize=10, fontweight='bold', loc='left')

        ax2.set_ylabel("DOC UIB (mAU)", fontsize=9)
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True, alpha=0.3)

        # === Plot 3: DAD 254 ===
        ax3 = axes[2]
        df_dad1 = r1_data.get("df_dad")

        if df_dad1 is not None and not df_dad1.empty:
            # Buscar columna 254
            wl_col = None
            for col in ['254', 'A254']:
                if col in df_dad1.columns:
                    wl_col = col
                    break

            if wl_col and 'time (min)' in df_dad1.columns:
                t_dad1 = df_dad1['time (min)'].values
                y_254_1 = df_dad1[wl_col].values
                ax3.plot(t_dad1, y_254_1, color=colors['r1'], label=f'R{rep_keys[0]}', linewidth=1)

                if r2_data:
                    df_dad2 = r2_data.get("df_dad")
                    if df_dad2 is not None and not df_dad2.empty and wl_col in df_dad2.columns:
                        t_dad2 = df_dad2['time (min)'].values
                        y_254_2 = df_dad2[wl_col].values
                        ax3.plot(t_dad2, y_254_2, color=colors['r2'], label=f'R{rep_keys[1]}',
                                linewidth=1, linestyle='--', alpha=0.8)

        ax3.set_ylabel("DAD 254nm (mAU)", fontsize=9)
        ax3.set_xlabel("Temps (min)", fontsize=9)
        ax3.legend(loc='upper right', fontsize=8)
        ax3.grid(True, alpha=0.3)
        ax3.set_title("DAD 254nm", fontsize=10, fontweight='bold', loc='left')

        # Zones (només COLUMN)
        if not self.is_bp:
            zones = [
                (0, 18, "BioP", "#E3F2FD"),
                (18, 23, "HS", "#FFF3E0"),
                (23, 30, "BB", "#F3E5F5"),
                (30, 40, "SB", "#E8F5E9"),
                (40, 70, "LMW", "#FCE4EC"),
            ]
            for ax in axes:
                for start, end, name, color in zones:
                    ax.axvspan(start, end, alpha=0.15, color=color, zorder=0)

        self.figure.tight_layout()
        self.canvas.draw()
