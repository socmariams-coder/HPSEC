"""
HPSEC Suite - Analyze Panel (Fase 3) — Taula Unificada DOC+DAD
================================================================

Panel per la fase 3: Anàlisi de mostres.
- Taula unificada (1 fila per mostra) amb DOC + DAD principals
- Panel de fraccions visible al seleccionar mostra
- Classificació d'anomalies per severitat (no falsos avisos)
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QComboBox,
    QFrame, QAbstractItemView, QProgressBar, QMessageBox
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QBrush, QFont

from pathlib import Path

from hpsec_analyze import analyze_sequence, save_analysis_result, load_analysis_result
from gui.widgets.styles import (
    COLOR_SUCCESS, COLOR_WARNING, COLOR_ERROR,
    apply_panel_layout, create_empty_state_widget
)
from .worker import AnalyzeWorker
from .dialogs import SampleDetailDialog
from ._constants import (
    CRITICAL_ANOMALIES, WARNING_ANOMALIES,
    DAD_WL_MAIN, SIGNAL_KEYS_MAIN,
)
from ._helpers import (
    configure_table_style, populate_signal_summary, populate_fractions_table
)


class AnalyzePanel(QWidget):
    """Panel d'anàlisi de mostres (Fase 3) — Taula unificada."""

    analyze_completed = Signal(dict)

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.samples_grouped = {}
        self.worker = None
        self._warnings_confirmed = False
        self._warnings_confirmed_by = ""
        self._selected_sample = None
        self._sample_row_map = {}       # P3: sample_name → row index (O(1) lookup)
        self._status_initialized = False  # B3: avoid redundant showEvent work

        self._setup_ui()

    # ------------------------------------------------------------------
    # UI Setup
    # ------------------------------------------------------------------

    def _setup_ui(self):
        """Configura la interfície — Taula unificada + panel fraccions."""
        layout = QVBoxLayout(self)
        apply_panel_layout(layout)

        # Botó analitzar (amagat - l'acció es dispara des del wizard header)
        self.analyze_btn = QPushButton()
        self.analyze_btn.setVisible(False)
        self.analyze_btn.clicked.connect(self._run_analyze)

        # === INFO PANEL ===
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

        self.import_info = QLabel()
        self.import_info.setStyleSheet("border: none;")
        info_layout.addWidget(self.import_info)

        sep1 = QFrame()
        sep1.setFrameShape(QFrame.VLine)
        sep1.setStyleSheet("background-color: #dee2e6; border: none; max-width: 1px;")
        info_layout.addWidget(sep1)

        self.cal_info = QLabel()
        self.cal_info.setStyleSheet("border: none;")
        info_layout.addWidget(self.cal_info)

        info_layout.addStretch()

        self.status_indicator = QLabel()
        self.status_indicator.setStyleSheet("border: none;")
        info_layout.addWidget(self.status_indicator)

        layout.addWidget(self.info_frame)

        # Empty state
        self.empty_state = create_empty_state_widget(
            "🔬", "Preparant anàlisi...",
            "Carregant dades de la seqüència."
        )
        self.empty_state.setVisible(False)
        layout.addWidget(self.empty_state)

        # Status frame (error messages)
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

        # === RESULTS FRAME ===
        self.results_frame = QFrame()
        self.results_frame.setVisible(False)
        results_layout = QVBoxLayout(self.results_frame)
        results_layout.setContentsMargins(0, 0, 0, 0)
        results_layout.setSpacing(8)

        # === WARNINGS BAR ===
        self.warnings_bar = QFrame()
        self.warnings_bar.setVisible(False)
        self.warnings_bar.setStyleSheet("""
            QFrame {
                background-color: #FFF3E0;
                border: 1px solid #FFB74D;
                border-radius: 6px;
            }
        """)
        warnings_bar_layout = QHBoxLayout(self.warnings_bar)
        warnings_bar_layout.setContentsMargins(12, 8, 12, 8)

        self.warnings_icon = QLabel("⚠️")
        self.warnings_icon.setStyleSheet("border: none;")
        warnings_bar_layout.addWidget(self.warnings_icon)

        self.warnings_text = QLabel()
        self.warnings_text.setStyleSheet("border: none; color: #E65100;")
        self.warnings_text.setWordWrap(True)
        warnings_bar_layout.addWidget(self.warnings_text, 1)

        self.warnings_details_btn = QPushButton("Veure detalls")
        self.warnings_details_btn.setStyleSheet("""
            QPushButton {
                background-color: #FFB74D; color: white;
                border: none; border-radius: 4px;
                padding: 4px 12px;
            }
            QPushButton:hover { background-color: #FFA726; }
        """)
        self.warnings_details_btn.clicked.connect(self._show_warnings_details)
        warnings_bar_layout.addWidget(self.warnings_details_btn)

        results_layout.addWidget(self.warnings_bar)

        # === LEGEND BAR ===
        legend_frame = QFrame()
        legend_frame.setStyleSheet("QFrame { background-color: #f8f9fa; border-radius: 4px; }")
        legend_layout = QHBoxLayout(legend_frame)
        legend_layout.setContentsMargins(12, 6, 12, 6)
        legend = QLabel(
            "<span style='color:#27AE60'>●</span> OK &nbsp;"
            "<span style='color:#F39C12'>●</span> Warning &nbsp;"
            "<span style='color:#E74C3C'>●</span> Error &nbsp;&nbsp;|&nbsp;&nbsp;"
            "<span style='color:#666; font-size:10px;'>Clic = fraccions · Doble-clic = detall complet</span>"
        )
        legend.setStyleSheet("color: #666;")
        legend_layout.addWidget(legend)
        legend_layout.addStretch()
        results_layout.addWidget(legend_frame)

        # === UNIFIED TABLE ===
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(11)
        self.results_table.setHorizontalHeaderLabels([
            "Mostra", "Rep", "A_DOC", "ppm", "A_UIB", "ppm_U",
            "A_254", "tmax_254", "Pics_HS", "R²_rep", "Estat"
        ])
        configure_table_style(self.results_table)
        self._configure_unified_columns()
        results_layout.addWidget(self.results_table)

        # Connect table signals
        self.results_table.itemSelectionChanged.connect(self._on_table_selection_changed)
        self.results_table.doubleClicked.connect(self._on_table_double_click)
        self.results_table.setToolTip("Clic per veure fraccions · Doble-clic per detall complet")

        # === FRACTIONS PANEL (below table, visible on selection) ===
        self.fractions_frame = QFrame()
        self.fractions_frame.setVisible(False)
        self.fractions_frame.setStyleSheet("""
            QFrame#fractionsFrame {
                background-color: #fafbfc;
                border: 1px solid #e1e4e8;
                border-radius: 6px;
            }
        """)
        self.fractions_frame.setObjectName("fractionsFrame")
        fractions_outer = QVBoxLayout(self.fractions_frame)
        fractions_outer.setContentsMargins(12, 8, 12, 8)
        fractions_outer.setSpacing(8)

        # Fractions header
        self.fractions_header = QLabel()
        self.fractions_header.setStyleSheet("border: none; font-weight: bold; font-size: 12px;")
        fractions_outer.addWidget(self.fractions_header)

        # Signal summary table
        self.signal_summary_table = QTableWidget()
        self.signal_summary_table.setColumnCount(5)
        self.signal_summary_table.setHorizontalHeaderLabels([
            "Senyal", "tmax (min)", "Àrea total", "SNR", "Timeouts"
        ])
        configure_table_style(self.signal_summary_table)
        fractions_outer.addWidget(self.signal_summary_table)

        # Fractions data table
        self.fractions_table = QTableWidget()
        configure_table_style(self.fractions_table)
        fractions_outer.addWidget(self.fractions_table)

        results_layout.addWidget(self.fractions_frame)

        # === STATS BAR ===
        self.stats_frame = QFrame()
        self.stats_frame.setStyleSheet("background-color: #f8f9fa; border-radius: 4px; padding: 8px;")
        stats_layout = QHBoxLayout(self.stats_frame)
        stats_layout.setContentsMargins(12, 8, 12, 8)

        self.stats_label = QLabel()
        self.stats_label.setFont(QFont("Segoe UI", 10))
        stats_layout.addWidget(self.stats_label)
        stats_layout.addStretch()

        results_layout.addWidget(self.stats_frame)

        layout.addWidget(self.results_frame, 1)

    def _configure_unified_columns(self):
        """Configura columnes de la taula unificada."""
        header = self.results_table.horizontalHeader()
        for i in range(self.results_table.columnCount()):
            header.setSectionResizeMode(i, QHeaderView.ResizeToContents)

    # ------------------------------------------------------------------
    # Show / Reset / Check existing
    # ------------------------------------------------------------------

    def showEvent(self, event):
        super().showEvent(event)
        if not self._status_initialized or not self.samples_grouped:
            self._check_existing_analysis()
            self._update_status()
            self._status_initialized = True

    def reset(self):
        """Reinicia el panel al seu estat inicial."""
        self.samples_grouped = {}
        self.worker = None
        self._warnings_confirmed = False
        self._warnings_confirmed_by = ""
        self._selected_sample = None
        self._sample_row_map = {}
        self._status_initialized = False

        self.results_table.setRowCount(0)

        self.empty_state.setVisible(True)
        self.info_frame.setVisible(False)
        self.status_frame.setVisible(False)
        self.progress_frame.setVisible(False)
        self.progress_bar.setValue(0)
        self.results_frame.setVisible(False)
        self.fractions_frame.setVisible(False)
        self.analyze_btn.setEnabled(True)
        self.stats_label.setText("")

    def _check_existing_analysis(self):
        """Comprova si existeix anàlisi prèvia i la carrega automàticament."""
        seq_path = self.main_window.seq_path
        if not seq_path:
            return
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
        self.samples_grouped = (result.get("samples_grouped")
                                or result.get("samples_analyzed", {}))
        if self.samples_grouped:
            self.main_window.processed_data = result  # B1: needed for method/seq_path
            self._populate_table()
            self.status_frame.setVisible(False)
            self.results_frame.setVisible(True)
            self.main_window.set_status("Anàlisi carregada des de fitxer existent", 3000)
            self.analyze_completed.emit(result)

    def _update_status(self):
        """Actualitza l'indicador d'estat amb format professional."""
        imported_data = self.main_window.imported_data
        calibration_data = self.main_window.calibration_data

        if not imported_data:
            self.info_frame.setVisible(False)
            self.status_frame.setVisible(False)
            self.empty_state.setVisible(True)
            self.analyze_btn.setEnabled(False)
            return

        self.empty_state.setVisible(False)
        self.info_frame.setVisible(True)
        self.status_frame.setVisible(False)

        samples = imported_data.get("samples", {})
        n_samples = len(samples)
        n_replicas = sum(len(reps) for reps in samples.values())
        method = imported_data.get("method", "-")
        data_mode = imported_data.get("data_mode", "-")

        self.import_info.setText(
            f"<span style='color: #6c757d; font-size: 10px;'>DADES</span><br>"
            f"<b style='font-size: 13px;'>{n_samples}</b> <span style='color: #495057;'>mostres</span> · "
            f"<b>{n_replicas}</b> <span style='color: #495057;'>rèpliques</span><br>"
            f"<span style='color: #6c757d; font-size: 10px;'>{method} / {data_mode}</span>"
        )

        if calibration_data and calibration_data.get("success"):
            khp_conc = calibration_data.get("khp_conc", 0)
            rf_mass_local = calibration_data.get("rf_mass", 0)
            rf_direct = calibration_data.get("rf_direct", 0) or calibration_data.get("rf", 0)
            shift = calibration_data.get("shift_direct", 0) or calibration_data.get("shift", 0)
            shift_sec = shift * 60 if shift else 0
            khp_source = calibration_data.get("khp_source", "LOCAL")
            volume_uL = calibration_data.get("volume_uL", 0)
            # Extract volume from nested calibration if not at top level
            if not volume_uL:
                cal_inner = calibration_data.get("calibration", {})
                volume_uL = cal_inner.get("volume_uL", 0) if cal_inner else 0

            # Get rf_mass_cal GLOBAL (what quantify_sample actually uses)
            rf_mass_global = None
            try:
                from hpsec_calibrate import get_rf_mass_cal
                seq_method = imported_data.get("method", "COLUMN")
                mode_key = "bp" if seq_method.upper() == "BP" else "column"
                rf_mass_global = get_rf_mass_cal(signal='direct', mode=mode_key)
            except Exception:
                pass

            is_alt = "ALTERNATIU" in str(khp_source) or "SIBLING" in str(khp_source)
            color = "#E67E22" if is_alt else "#27AE60"
            icon = "⚠" if is_alt else "✓"

            # Build label: "KHP 2ppm @ 100µL" or "KHP 2ppm" if no volume
            khp_label = f"KHP {khp_conc:.0f}ppm"
            if volume_uL > 0:
                khp_label += f" @ {volume_uL:.0f}µL"

            # Display: rf_mass_cal global if available, else rf_mass local, else rf_direct
            if rf_mass_global and rf_mass_global > 0:
                rf_display = rf_mass_global
                rf_type = "RF_cal"
                cal_note = "GLOBAL (Calibration_Reference)"
            elif rf_mass_local > 0:
                rf_display = rf_mass_local
                rf_type = "RF_mass"
                cal_note = "LOCAL (SEQ)"
            else:
                rf_display = rf_direct
                rf_type = "RF"
                cal_note = "LOCAL (àrea/ppm)"

            self.cal_info.setText(
                f"<span style='color: #6c757d; font-size: 10px;'>CALIBRACIÓ</span><br>"
                f"<span style='color: {color};'>{icon}</span> <b style='font-size: 13px;'>{khp_label}</b><br>"
                f"<span style='color: #6c757d; font-size: 10px;'>"
                f"{rf_type}: <b>{rf_display:.1f}</b> · Shift: <b>{shift_sec:.1f}s</b></span>"
            )
            rf_global_str = f"{rf_mass_global:.2f}" if rf_mass_global else "N/A"
            self.cal_info.setToolTip(
                f"Font: {khp_source}\n"
                f"Quantificació: {cal_note}\n"
                f"RF_mass_cal (global): {rf_global_str}\n"
                f"RF_mass (local): {rf_mass_local:.2f}\n"
                f"RF_direct (local): {rf_direct:.2f}\n"
                f"Shift: {shift_sec:.2f}s ({shift:.4f} min)\n"
                f"Volum: {volume_uL:.0f} µL"
            )
        else:
            self.cal_info.setText(
                f"<span style='color: #6c757d; font-size: 10px;'>CALIBRACIÓ</span><br>"
                f"<span style='color: #E67E22;'>⚠</span> <span style='color: #856404;'>No disponible</span><br>"
                f"<span style='color: #6c757d; font-size: 10px;'>S'usaran valors per defecte</span>"
            )
            self.cal_info.setToolTip("No hi ha calibració disponible")

        if self.samples_grouped:
            n = len(self.samples_grouped)
            self.status_indicator.setText(
                f"<span style='background-color: #d4edda; color: #155724; "
                f"padding: 4px 12px; border-radius: 12px; font-size: 11px;'>"
                f"Anàlisi completada ({n} mostres)</span>"
            )
        else:
            self.status_indicator.setText(
                f"<span style='background-color: #d4edda; color: #155724; "
                f"padding: 4px 12px; border-radius: 12px; font-size: 11px;'>"
                f"Llest per analitzar</span>"
            )
        self.analyze_btn.setEnabled(True)

    # ------------------------------------------------------------------
    # Run analysis
    # ------------------------------------------------------------------

    def _run_analyze(self):
        """Executa l'anàlisi."""
        imported_data = self.main_window.imported_data
        calibration_data = self.main_window.calibration_data
        seq_path = self.main_window.seq_path

        # Auto-load if not in memory
        if not imported_data and seq_path:
            from hpsec_import import import_from_manifest
            self.main_window.set_status("Carregant dades d'importació...")
            imported_data = import_from_manifest(seq_path)
            if imported_data and imported_data.get('success'):
                self.main_window.imported_data = imported_data

        if not calibration_data and seq_path:
            import json
            cal_path = Path(seq_path) / "CHECK" / "data" / "calibration_result.json"
            if cal_path.exists():
                self.main_window.set_status("Carregant dades de calibració...")
                with open(cal_path, 'r', encoding='utf-8') as f:
                    cal_file = json.load(f)
                calibrations = cal_file.get("calibrations", [])
                if calibrations:
                    active_cal = None
                    for cal in calibrations:
                        if cal.get("is_active", False):
                            active_cal = cal
                            break
                    if not active_cal:
                        active_cal = calibrations[0]

                    # Reconstruct full calibration_data matching calibrate_panel output
                    area = active_cal.get("area", 0)
                    conc = active_cal.get("conc_ppm", 5)
                    volume = active_cal.get("volume_uL", 0)
                    rf = active_cal.get("rf", 0)
                    if rf == 0 and conc > 0 and area > 0:
                        rf = area / conc
                    rf_direct = active_cal.get("rf_direct", rf)
                    rf_uib = active_cal.get("rf_uib", 0)
                    rf_mass = active_cal.get("rf_mass", 0)

                    calibration_data = {
                        "success": True,
                        "mode": active_cal.get("mode", "DUAL"),
                        "rf_direct": rf_direct,
                        "rf_uib": rf_uib,
                        "rf": rf,
                        "rf_mass": rf_mass,
                        "shift_direct": active_cal.get("shift_direct") or active_cal.get("shift_min", 0),
                        "shift_uib": active_cal.get("shift_uib") or active_cal.get("shift_min_u", 0),
                        "khp_area_direct": area,
                        "khp_area_uib": active_cal.get("area_u", 0),
                        "khp_area": area,
                        "khp_conc": conc,
                        "volume_uL": volume,
                        "khp_source": active_cal.get("khp_source", "LOCAL"),
                        "calibration": active_cal,
                        "errors": [],
                        "loaded_from_json": True,
                    }
                    self.main_window.calibration_data = calibration_data

        if not imported_data:
            QMessageBox.warning(self, "Avís", "No s'han trobat dades d'importació.")
            return

        samples = imported_data.get("samples", {})
        if not samples:
            QMessageBox.warning(self, "Avís", "No s'han trobat mostres a les dades importades.")
            return

        self.analyze_btn.setEnabled(False)
        self.progress_frame.setVisible(True)
        self.progress_bar.setValue(0)
        self.results_frame.setVisible(False)

        self.worker = AnalyzeWorker(imported_data, calibration_data)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_progress(self, msg, pct):
        self.progress_label.setText(msg)
        self.progress_bar.setValue(pct)

    def _on_finished(self, result):
        """Gestiona la finalització de l'anàlisi."""
        self.progress_frame.setVisible(False)
        self.analyze_btn.setEnabled(True)

        if not result or not result.get("success"):
            error_msg = result.get("error", "Error desconegut") if result else "Resultat buit"
            QMessageBox.critical(self, "Error", f"Error durant l'anàlisi:\n{error_msg}")
            self._update_status()
            return

        self.status_frame.setVisible(False)
        self.main_window.processed_data = result
        self.samples_grouped = result.get("samples_grouped", {})

        save_analysis_result(result)
        self._populate_table()
        self.results_frame.setVisible(True)
        self._show_warnings_bar(result)
        self.analyze_completed.emit(result)

    def _on_error(self, error_msg):
        self.progress_frame.setVisible(False)
        self.analyze_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", f"Error durant l'anàlisi:\n{error_msg}")

    # ------------------------------------------------------------------
    # Warnings bar
    # ------------------------------------------------------------------

    def _show_warnings_bar(self, result):
        """Mostra la barra d'avisos si hi ha warnings estructurats."""
        warnings_structured = result.get("warnings_structured", [])
        warning_level = result.get("warning_level", "none")

        if warning_level in ("warning", "blocker") and warnings_structured:
            pending_warnings = [w for w in warnings_structured if not w.get("dismissed", False)]
            if pending_warnings:
                n_warnings = len(pending_warnings)
                by_code = {}
                for w in pending_warnings:
                    code = w.get("code", "UNKNOWN")
                    by_code[code] = by_code.get(code, 0) + 1

                # Build code→message lookup (O(n) instead of O(n²))
                code_to_msg = {}
                for w in pending_warnings:
                    c = w.get("code", "UNKNOWN")
                    if c not in code_to_msg:
                        code_to_msg[c] = w.get("message", c)

                summary_parts = []
                for code, count in list(by_code.items())[:3]:
                    msg = code_to_msg.get(code, code)
                    if count > 1:
                        summary_parts.append(f"{msg} ({count}x)")
                    else:
                        summary_parts.append(msg)
                summary = " | ".join(summary_parts)
                if len(by_code) > 3:
                    summary += f" | +{len(by_code) - 3} més"

                if warning_level == "blocker":
                    self.warnings_bar.setStyleSheet("""
                        QFrame {
                            background-color: #FFEBEE;
                            border: 1px solid #E57373;
                            border-radius: 6px;
                        }
                    """)
                    self.warnings_icon.setText("🚫")
                    self.warnings_text.setStyleSheet("border: none; color: #C62828;")
                else:
                    self.warnings_bar.setStyleSheet("""
                        QFrame {
                            background-color: #FFF3E0;
                            border: 1px solid #FFB74D;
                            border-radius: 6px;
                        }
                    """)
                    self.warnings_icon.setText("⚠️")
                    self.warnings_text.setStyleSheet("border: none; color: #E65100;")

                self.warnings_text.setText(f"<b>{n_warnings} avisos:</b> {summary}")
                self.warnings_bar.setVisible(True)
                self._current_warnings = pending_warnings
            else:
                self.warnings_bar.setVisible(False)
                self._current_warnings = []
        else:
            self.warnings_bar.setVisible(False)
            self._current_warnings = []

    def _show_warnings_details(self):
        """Mostra un diàleg amb tots els avisos."""
        warnings = getattr(self, "_current_warnings", [])
        if not warnings:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Avisos d'Anàlisi ({len(warnings)})")
        dialog.setMinimumWidth(500)
        layout = QVBoxLayout(dialog)
        for w in warnings:
            level = w.get("level", "info")
            icon = {"blocker": "🚫", "warning": "⚠️", "info": "ℹ️"}.get(level, "•")
            color = {"blocker": "#C62828", "warning": "#E65100", "info": "#1565C0"}.get(level, "#666")
            message = w.get("message", w.get("code", "Avís"))
            sample = w.get("sample", "")
            label = QLabel(f'{icon} <span style="color:{color}"><b>{message}</b></span>')
            if sample:
                label.setText(label.text() + f' <span style="color:#666">[{sample}]</span>')
            label.setWordWrap(True)
            layout.addWidget(label)
        close_btn = QPushButton("Tancar")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        dialog.exec()

    # ------------------------------------------------------------------
    # Populate unified table
    # ------------------------------------------------------------------

    def _populate_table(self):
        """Omple la taula unificada amb els resultats."""
        self.results_table.setRowCount(0)
        self._sample_row_map = {}
        n_ok, n_warning, n_error = 0, 0, 0

        for sample_name in sorted(self.samples_grouped.keys()):
            sample_data = self.samples_grouped[sample_name]
            row = self.results_table.rowCount()
            self.results_table.insertRow(row)
            self._sample_row_map[sample_name] = row

            replicas = sample_data.get("replicas") or {}
            comparison = sample_data.get("comparison") or {}
            recommendation = sample_data.get("recommendation") or {}
            selected = sample_data.get("selected") or {"doc": "1", "dad": "1"}
            quantification = sample_data.get("quantification") or {}

            # Select replica (unified - same for DOC and DAD)
            doc_rec = (recommendation.get("doc") or {}).get("replica", "1")
            rep_sel = selected.get("doc", doc_rec)
            rep_data = replicas.get(rep_sel, {})

            # Col 0: Sample name
            item_name = QTableWidgetItem(sample_name)
            item_name.setData(Qt.UserRole, sample_name)
            self.results_table.setItem(row, 0, item_name)

            # Col 1: Replica selector (unified)
            replica_combo = QComboBox()
            replica_combo.setStyleSheet("QComboBox { border: none; background: transparent; padding: 2px; }")
            for rep_num in sorted(replicas.keys()):
                replica_combo.addItem(f"R{rep_num}", rep_num)
                if rep_num == rep_sel:
                    replica_combo.setCurrentIndex(replica_combo.count() - 1)
            replica_combo.currentIndexChanged.connect(
                lambda idx, name=sample_name: self._on_replica_changed(name)
            )
            self.results_table.setCellWidget(row, 1, replica_combo)

            # Col 2: A_DOC (area total DOC Direct)
            areas = rep_data.get("areas") or {}
            doc_areas = areas.get("DOC") or {}
            area_direct = doc_areas.get("total", 0)
            self.results_table.setItem(row, 2, QTableWidgetItem(
                f"{area_direct:.0f}" if area_direct else "-"))

            # Col 3: ppm (DOC Direct concentration)
            ppm_direct = quantification.get("concentration_ppm_direct") or quantification.get("concentration_ppm")
            self.results_table.setItem(row, 3, QTableWidgetItem(
                f"{ppm_direct:.2f}" if ppm_direct else "-"))

            # Col 4: A_UIB (area total DOC UIB)
            areas_uib = rep_data.get("areas_uib") or {}
            area_uib = areas_uib.get("total", 0)
            self.results_table.setItem(row, 4, QTableWidgetItem(
                f"{area_uib:.0f}" if area_uib else "-"))

            # Col 5: ppm_U (UIB concentration)
            ppm_uib = quantification.get("concentration_ppm_uib")
            self.results_table.setItem(row, 5, QTableWidgetItem(
                f"{ppm_uib:.2f}" if ppm_uib else "-"))

            # Col 6: A_254 (area total DAD 254nm)
            area_254 = areas.get("A254", {}).get("total", 0)
            self.results_table.setItem(row, 6, QTableWidgetItem(
                f"{area_254:.0f}" if area_254 else "-"))

            # Col 7: tmax_254 (retention time of main peak at 254nm)
            tmax_signals = rep_data.get("tmax_signals") or {}
            tmax_254 = tmax_signals.get("A254", 0)
            self.results_table.setItem(row, 7, QTableWidgetItem(
                f"{tmax_254:.2f}" if tmax_254 else "-"))

            # Col 8: Pics_HS (number of peaks at 254nm in HS zone, COLUMN only)
            n_peaks_hs = rep_data.get("n_peaks_254_HS", 0)
            self.results_table.setItem(row, 8, QTableWidgetItem(
                str(n_peaks_hs) if n_peaks_hs else "-"))

            # Col 9: R²_rep (Pearson DOC between replicas)
            r2_doc = comparison.get("doc", {}).get("pearson", 0) if comparison else 0
            r2_item = QTableWidgetItem(f"{r2_doc:.3f}" if r2_doc > 0 else "-")
            if 0 < r2_doc < 0.995:
                r2_item.setForeground(QBrush(QColor(COLOR_WARNING)))
            self.results_table.setItem(row, 9, r2_item)

            # Col 10: Estat (with corrected severity logic)
            status_color, status_text, tooltip = self._classify_sample_status(
                rep_data, comparison)
            status_item = QTableWidgetItem(status_text)
            status_item.setForeground(QBrush(QColor(status_color)))
            status_item.setTextAlignment(Qt.AlignCenter)
            status_item.setToolTip(tooltip)
            self.results_table.setItem(row, 10, status_item)

            # Count stats
            if status_color == COLOR_ERROR:
                n_error += 1
            elif status_color == COLOR_WARNING:
                n_warning += 1
            else:
                n_ok += 1

        # Update stats bar
        total = n_ok + n_warning + n_error
        self.stats_label.setText(
            f"<b>Total:</b> {total} mostres &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"<span style='color:#27AE60'>●</span> OK: {n_ok} &nbsp;&nbsp;"
            f"<span style='color:#F39C12'>●</span> Warning: {n_warning} &nbsp;&nbsp;"
            f"<span style='color:#E74C3C'>●</span> Error: {n_error}"
        )

    # ------------------------------------------------------------------
    # Anomaly severity classification
    # ------------------------------------------------------------------

    def _classify_sample_status(self, rep_data, comparison):
        """Classifica l'estat d'una mostra amb lògica corregida.

        Returns (color, status_text, tooltip).
        """
        anomalies = rep_data.get("anomalies", [])
        timeout_info = rep_data.get("timeout_info", {})
        replica_warnings = comparison.get("doc", {}).get("warnings", []) if comparison else []

        # Classify anomalies by severity
        has_critical = any(a in CRITICAL_ANOMALIES for a in anomalies)
        has_warning = any(a in WARNING_ANOMALIES for a in anomalies)

        # Timeout severity (already calculated per zone)
        timeout_severity = timeout_info.get("severity", "OK")
        has_timeout_warning = timeout_severity in ("WARNING", "CRITICAL")
        n_timeouts = timeout_info.get("n_timeouts", 0)

        # Build status icons
        status_parts = []
        if n_timeouts > 0:
            status_parts.append(f"T({n_timeouts})")
        if any("BATMAN" in a for a in anomalies):
            status_parts.append("B")
        if "NO_PEAK" in anomalies:
            status_parts.append("!")

        # Determine color
        if has_critical:
            status_color = COLOR_ERROR
        elif has_warning or has_timeout_warning or replica_warnings:
            status_color = COLOR_WARNING
        else:
            status_color = COLOR_SUCCESS

        status_text = " ".join(status_parts) if status_parts else "✓"

        # Build tooltip
        tooltip_parts = []
        # Critical/warning anomalies
        critical_found = [a for a in anomalies if a in CRITICAL_ANOMALIES]
        warning_found = [a for a in anomalies if a in WARNING_ANOMALIES]
        info_found = [a for a in anomalies if a not in CRITICAL_ANOMALIES and a not in WARNING_ANOMALIES]

        if critical_found:
            tooltip_parts.append(f"CRÍTIC: {', '.join(critical_found)}")
        if warning_found:
            tooltip_parts.append(f"Avís: {', '.join(warning_found)}")
        if info_found:
            tooltip_parts.append(f"Info: {', '.join(info_found)}")
        if n_timeouts > 0:
            zones = timeout_info.get("zones", [])
            zone_str = ", ".join(zones) if zones else "?"
            tooltip_parts.append(
                f"Timeouts: {n_timeouts} ({timeout_severity}) — zones: {zone_str}"
            )
            tooltip_parts.append("Nota: timeouts DOC Direct també afecten UIB (mateix detector)")
        if replica_warnings:
            tooltip_parts.extend(replica_warnings)

        tooltip = "\n".join(tooltip_parts) if tooltip_parts else "OK"
        return status_color, status_text, tooltip

    # ------------------------------------------------------------------
    # Replica change (unified handler)
    # ------------------------------------------------------------------

    def _on_replica_changed(self, sample_name):
        """Gestiona el canvi de rèplica (selector únic)."""
        if sample_name not in self.samples_grouped:
            return

        row = self._sample_row_map.get(sample_name)
        if row is None:
            return

        combo = self.results_table.cellWidget(row, 1)
        if combo:
            new_replica = combo.currentData()
            self.samples_grouped[sample_name]["selected"]["doc"] = new_replica
            self.samples_grouped[sample_name]["selected"]["dad"] = new_replica
            self._update_quantification(sample_name)
            self._update_row(row, sample_name)
            if self._selected_sample == sample_name:
                self._update_fractions_panel(sample_name)

    def _update_row(self, row, sample_name):
        """Actualitza una fila de la taula unificada."""
        sample_data = self.samples_grouped[sample_name]
        selected = sample_data.get("selected", {})
        rep_sel = selected.get("doc", "1")
        replicas = sample_data.get("replicas", {})
        rep_data = replicas.get(rep_sel, {})
        quantification = sample_data.get("quantification", {})

        areas = rep_data.get("areas") or {}
        doc_areas = areas.get("DOC") or {}
        areas_uib = rep_data.get("areas_uib") or {}

        # Col 2: A_DOC
        area_direct = doc_areas.get("total", 0)
        self.results_table.item(row, 2).setText(f"{area_direct:.0f}" if area_direct else "-")

        # Col 3: ppm
        ppm_direct = quantification.get("concentration_ppm_direct") or quantification.get("concentration_ppm")
        self.results_table.item(row, 3).setText(f"{ppm_direct:.2f}" if ppm_direct else "-")

        # Col 4: A_UIB
        area_uib = areas_uib.get("total", 0)
        self.results_table.item(row, 4).setText(f"{area_uib:.0f}" if area_uib else "-")

        # Col 5: ppm_U
        ppm_uib = quantification.get("concentration_ppm_uib")
        self.results_table.item(row, 5).setText(f"{ppm_uib:.2f}" if ppm_uib else "-")

        # Col 6: A_254
        area_254 = areas.get("A254", {}).get("total", 0)
        self.results_table.item(row, 6).setText(f"{area_254:.0f}" if area_254 else "-")

        # Col 7: tmax_254
        tmax_signals = rep_data.get("tmax_signals") or {}
        tmax_254 = tmax_signals.get("A254", 0)
        self.results_table.item(row, 7).setText(f"{tmax_254:.2f}" if tmax_254 else "-")

        # Col 8: Pics_HS
        n_peaks_hs = rep_data.get("n_peaks_254_HS", 0)
        self.results_table.item(row, 8).setText(str(n_peaks_hs) if n_peaks_hs else "-")

        # Col 9: R²_rep (doesn't change with replica, but refresh for consistency)
        comparison = sample_data.get("comparison") or {}
        r2_doc = comparison.get("doc", {}).get("pearson", 0) if comparison else 0
        r2_item = self.results_table.item(row, 9)
        if r2_item:
            r2_item.setText(f"{r2_doc:.3f}" if r2_doc > 0 else "-")
            if 0 < r2_doc < 0.995:
                r2_item.setForeground(QBrush(QColor(COLOR_WARNING)))
            else:
                r2_item.setForeground(QBrush(QColor("#000000")))

        # Col 10: Estat (B2: must update — different replicas have different anomalies)
        status_color, status_text, tooltip = self._classify_sample_status(
            rep_data, comparison)
        status_item = self.results_table.item(row, 10)
        if status_item:
            status_item.setText(status_text)
            status_item.setForeground(QBrush(QColor(status_color)))
            status_item.setToolTip(tooltip)

    # ------------------------------------------------------------------
    # Quantification recalculation
    # ------------------------------------------------------------------

    def _update_quantification(self, sample_name):
        """Recalcula la quantificació per una mostra."""
        try:
            from hpsec_analyze import quantify_sample
            from hpsec_calibrate import get_all_active_calibrations

            sample_data = self.samples_grouped[sample_name]
            selected_doc = sample_data["selected"]["doc"]
            selected_replica = sample_data["replicas"].get(selected_doc)

            if selected_replica:
                processed = self.main_window.processed_data or {}
                method = processed.get("method", "COLUMN")
                mode = "BP" if method == "BP" else "COLUMN"
                seq_date = processed.get("seq_date")

                calibration_data = None
                inj_volume = selected_replica.get("inj_volume")
                seq_path = processed.get("seq_path", "")

                if seq_path and inj_volume:
                    try:
                        active_cals = get_all_active_calibrations(seq_path, mode)
                        for cal in active_cals:
                            cal_vol = cal.get("volume_uL", 0)
                            if cal_vol > 0 and abs(cal_vol - inj_volume) < 0.1:
                                calibration_data = cal
                                break
                        if not calibration_data and active_cals:
                            calibration_data = active_cals[0]
                    except Exception as e:
                        print(f"[WARN] Error carregant calibracions: {e}")

                if not calibration_data:
                    calibration_data = self.main_window.calibration_data
                    # Ensure volume context from replica is available
                    if calibration_data and inj_volume:
                        if not calibration_data.get("volume_uL"):
                            calibration_data = dict(calibration_data)
                            calibration_data["volume_uL"] = inj_volume

                quantification = quantify_sample(
                    selected_replica, calibration_data,
                    mode=mode, seq_date=seq_date
                )
                sample_data["quantification"] = quantification
        except Exception as e:
            print(f"Error recalculant quantificació: {e}")
            self.main_window.set_status(f"Error quantificació: {e}", 5000)

    # ------------------------------------------------------------------
    # Fractions panel (below table)
    # ------------------------------------------------------------------

    def _update_fractions_panel(self, sample_name):
        """Actualitza el panel de fraccions per la mostra seleccionada."""
        if sample_name not in self.samples_grouped:
            self.fractions_frame.setVisible(False)
            return

        sample_data = self.samples_grouped[sample_name]
        selected = sample_data.get("selected", {})
        rep_sel = selected.get("doc", "1")
        replicas = sample_data.get("replicas", {})
        rep_data = replicas.get(rep_sel, {})

        if not rep_data:
            self.fractions_frame.setVisible(False)
            return

        method = ""
        if self.main_window.processed_data:
            method = self.main_window.processed_data.get("method", "COLUMN")
        is_bp = method.upper() == "BP"

        self.fractions_header.setText(
            f"<b>{sample_name}</b> — R{rep_sel} — "
            f"{'BP (total)' if is_bp else 'COLUMN (fraccions)'}"
        )

        # --- Signal summary table ---
        self._populate_signal_summary(rep_data)

        # --- Fractions table ---
        self._populate_fractions_table(rep_data, is_bp)

        self.fractions_frame.setVisible(True)

    def _populate_signal_summary(self, rep_data):
        """Omple la taula resum de senyals (tmax, àrea, SNR, timeouts)."""
        populate_signal_summary(
            self.signal_summary_table, rep_data,
            SIGNAL_KEYS_MAIN, show_timeouts=True
        )

    def _populate_fractions_table(self, rep_data, is_bp):
        """Omple la taula de fraccions per COLUMN (o total per BP)."""
        populate_fractions_table(
            self.fractions_table, rep_data, is_bp,
            DAD_WL_MAIN, show_ratio=False
        )

    # ------------------------------------------------------------------
    # Table interaction
    # ------------------------------------------------------------------

    def _on_table_selection_changed(self):
        """Gestiona canvi de selecció — mostra fraccions."""
        selected_rows = self.results_table.selectionModel().selectedRows()

        if len(selected_rows) == 1:
            row = selected_rows[0].row()
            item = self.results_table.item(row, 0)
            if item:
                sample_name = item.data(Qt.UserRole) or item.text()
                self._selected_sample = sample_name
                self._update_fractions_panel(sample_name)
                return

        self._selected_sample = None
        self.fractions_frame.setVisible(False)

    def _on_table_double_click(self, index):
        """Handler per doble clic — obre SampleDetailDialog."""
        row = index.row()
        item = self.results_table.item(row, 0)
        if item:
            sample_name = item.data(Qt.UserRole) or item.text()
            self._show_detail(sample_name)

    def _show_detail(self, sample_name):
        """Mostra el diàleg de detall."""
        if sample_name not in self.samples_grouped:
            return
        method = "COLUMN"
        if self.main_window.processed_data:
            method = self.main_window.processed_data.get("method", "COLUMN")
        dialog = SampleDetailDialog(
            sample_name,
            self.samples_grouped[sample_name],
            method,
            parent=self
        )
        dialog.exec()

