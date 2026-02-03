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

from hpsec_analyze import analyze_sequence, save_analysis_result

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
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # === HEADER ===
        header_layout = QHBoxLayout()

        title = QLabel("Anàlisi de Mostres")
        title.setFont(QFont("Segoe UI", 14, QFont.Bold))
        header_layout.addWidget(title)

        header_layout.addStretch()

        # Botó analitzar
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
        header_layout.addWidget(self.analyze_btn)

        layout.addLayout(header_layout)

        # === STATUS INFO ===
        self.status_frame = QFrame()
        self.status_frame.setStyleSheet(
            "background-color: #FFF3E0; border-radius: 4px; padding: 12px;"
        )
        status_layout = QVBoxLayout(self.status_frame)
        status_layout.setContentsMargins(12, 8, 12, 8)
        status_layout.setSpacing(4)

        self.status_label = QLabel()
        self.status_label.setFont(QFont("Segoe UI", 10))
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

        # Llegenda
        legend = QLabel(
            "<span style='color:#27AE60'>●</span> OK &nbsp;&nbsp;"
            "<span style='color:#F39C12'>●</span> Warning &nbsp;&nbsp;"
            "<span style='color:#E74C3C'>●</span> Error"
        )
        legend.setStyleSheet("color: #666; margin-bottom: 4px;")
        results_layout.addWidget(legend)

        # Taula
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(9)
        self.results_table.setHorizontalHeaderLabels([
            "Mostra", "Rèplica", "R² DOC", "SNR DOC",
            "SNR DAD", "Àrea", "[ppm]", "Status", "Detall"
        ])

        # Configurar taula
        self.results_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.results_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.results_table.setAlternatingRowColors(True)
        self.results_table.verticalHeader().setVisible(False)

        # Estil per cel·les no editables
        self.results_table.setStyleSheet("""
            QTableWidget {
                gridline-color: #ddd;
                background-color: white;
                alternate-background-color: #f9f9f9;
            }
            QTableWidget::item {
                padding: 4px 8px;
            }
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
            }
        """)

        # Mida columnes
        header = self.results_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)  # Mostra
        header.setSectionResizeMode(1, QHeaderView.Fixed)    # Rèplica
        header.setSectionResizeMode(2, QHeaderView.Fixed)    # R² DOC
        header.setSectionResizeMode(3, QHeaderView.Fixed)    # SNR DOC
        header.setSectionResizeMode(4, QHeaderView.Fixed)    # SNR DAD
        header.setSectionResizeMode(5, QHeaderView.Fixed)    # Àrea
        header.setSectionResizeMode(6, QHeaderView.Fixed)    # ppm
        header.setSectionResizeMode(7, QHeaderView.Fixed)    # Status
        header.setSectionResizeMode(8, QHeaderView.Fixed)    # Detall

        self.results_table.setColumnWidth(1, 90)
        self.results_table.setColumnWidth(2, 80)
        self.results_table.setColumnWidth(3, 100)
        self.results_table.setColumnWidth(4, 120)
        self.results_table.setColumnWidth(5, 80)
        self.results_table.setColumnWidth(6, 70)
        self.results_table.setColumnWidth(7, 60)
        self.results_table.setColumnWidth(8, 70)

        results_layout.addWidget(self.results_table)

        # Resum estadístic
        self.stats_frame = QFrame()
        self.stats_frame.setStyleSheet("background-color: #f8f9fa; border-radius: 4px; padding: 8px;")
        stats_layout = QHBoxLayout(self.stats_frame)
        stats_layout.setContentsMargins(12, 8, 12, 8)

        self.stats_label = QLabel()
        self.stats_label.setFont(QFont("Segoe UI", 10))
        stats_layout.addWidget(self.stats_label)

        stats_layout.addStretch()

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

    def showEvent(self, event):
        """Es crida quan el panel es fa visible."""
        super().showEvent(event)
        self._update_status()

    def _update_status(self):
        """Actualitza l'indicador d'estat."""
        imported_data = self.main_window.imported_data
        calibration_data = self.main_window.calibration_data

        if not imported_data:
            self.status_frame.setStyleSheet(
                "background-color: #FFEBEE; border-radius: 4px; padding: 12px;"
            )
            self.status_label.setText(
                "⚠ <b>No hi ha dades importades.</b><br>"
                "Vés a la pestanya <b>1. Importar</b> per carregar les dades de la seqüència."
            )
            self.analyze_btn.setEnabled(False)
            return

        # Comptar mostres
        samples = imported_data.get("samples", {})
        n_samples = len(samples)
        n_replicas = sum(len(reps) for reps in samples.values())

        cal_status = "✓ Calibració disponible" if calibration_data else "⚠ Sense calibració"

        self.status_frame.setStyleSheet(
            "background-color: #E8F5E9; border-radius: 4px; padding: 12px;"
        )
        self.status_label.setText(
            f"✓ <b>Dades carregades:</b> {n_samples} mostres, {n_replicas} rèpliques<br>"
            f"{cal_status}<br><br>"
            f"Prem <b>Analitzar</b> per processar les mostres."
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

        # Guardar resultat
        self.main_window.processed_data = result
        self.samples_grouped = result.get("samples_grouped", {})

        # Guardar a JSON
        save_analysis_result(result)

        # Mostrar resultats
        self._populate_table()
        self.results_frame.setVisible(True)

        # Habilitar navegació
        self.next_btn.setEnabled(True)

        # Emetre senyal
        self.analyze_completed.emit(result)

    def _on_error(self, error_msg):
        """Gestiona errors."""
        self.progress_frame.setVisible(False)
        self.analyze_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", f"Error durant l'anàlisi:\n{error_msg}")

    def _populate_table(self):
        """Omple la taula amb els resultats."""
        self.results_table.setRowCount(0)

        n_ok = 0
        n_warning = 0
        n_error = 0

        for sample_name in sorted(self.samples_grouped.keys()):
            sample_data = self.samples_grouped[sample_name]
            row = self.results_table.rowCount()
            self.results_table.insertRow(row)

            replicas = sample_data.get("replicas") or {}
            comparison = sample_data.get("comparison") or {}
            recommendation = sample_data.get("recommendation") or {}
            selected = sample_data.get("selected") or {"doc": "1", "dad": "1"}
            quantification = sample_data.get("quantification") or {}

            # Col 0: Mostra
            item_name = QTableWidgetItem(sample_name)
            item_name.setData(Qt.UserRole, sample_name)
            self.results_table.setItem(row, 0, item_name)

            # Col 1: Selector de rèplica (dropdown)
            replica_combo = QComboBox()
            replica_combo.setStyleSheet("QComboBox { border: none; background: transparent; }")

            doc_rec = (recommendation.get("doc") or {}).get("replica", "1")
            doc_sel = selected.get("doc", doc_rec)

            for rep_num in sorted(replicas.keys()):
                is_rec = rep_num == doc_rec
                label = f"R{rep_num}" + (" ★" if is_rec else "")
                replica_combo.addItem(label, rep_num)
                if rep_num == doc_sel:
                    replica_combo.setCurrentIndex(replica_combo.count() - 1)

            replica_combo.currentIndexChanged.connect(
                lambda idx, name=sample_name: self._on_replica_changed(name)
            )
            self.results_table.setCellWidget(row, 1, replica_combo)

            # Obtenir dades de la rèplica seleccionada
            rep_data = replicas.get(doc_sel, {})

            # Col 2: R² DOC (Pearson entre rèpliques)
            r2_doc = comparison.get("doc", {}).get("pearson", 0) if comparison else 0
            r2_text = f"{r2_doc:.4f}" if r2_doc > 0 else "-"
            r2_item = QTableWidgetItem(r2_text)
            if r2_doc > 0 and r2_doc < 0.995:
                r2_item.setForeground(QBrush(QColor("#F39C12")))
            self.results_table.setItem(row, 2, r2_item)

            # Col 3: SNR DOC (Direct / UIB)
            snr_info = rep_data.get("snr_info", {})
            snr_direct = snr_info.get("snr_direct", 0)
            snr_uib = snr_info.get("snr_uib", 0)

            if snr_uib and snr_uib > 0:
                snr_text = f"{snr_direct:.0f} / {snr_uib:.0f}"
            else:
                snr_text = f"{snr_direct:.0f}" if snr_direct else "-"
            self.results_table.setItem(row, 3, QTableWidgetItem(snr_text))

            # Col 4: SNR DAD (millor/pitjor WL)
            snr_dad_text = self._format_snr_dad(rep_data)
            self.results_table.setItem(row, 4, QTableWidgetItem(snr_dad_text))

            # Col 5: Àrea total
            area = rep_data.get("area_total", 0)
            area_text = f"{area:.0f}" if area else "-"
            self.results_table.setItem(row, 5, QTableWidgetItem(area_text))

            # Col 6: Concentració ppm
            conc = quantification.get("concentration_ppm") if quantification else None
            conc_text = f"{conc:.2f}" if conc is not None else "-"
            self.results_table.setItem(row, 6, QTableWidgetItem(conc_text))

            # Col 7: Status
            anomalies = rep_data.get("anomalies", [])
            warnings = comparison.get("doc", {}).get("warnings", []) if comparison else []

            if anomalies:
                status_text = "●"
                status_color = "#E74C3C"  # Error
                n_error += 1
            elif warnings:
                status_text = "●"
                status_color = "#F39C12"  # Warning
                n_warning += 1
            else:
                status_text = "●"
                status_color = "#27AE60"  # OK
                n_ok += 1

            status_item = QTableWidgetItem(status_text)
            status_item.setForeground(QBrush(QColor(status_color)))
            status_item.setTextAlignment(Qt.AlignCenter)
            status_item.setToolTip("\n".join(anomalies + warnings) if (anomalies or warnings) else "OK")
            self.results_table.setItem(row, 7, status_item)

            # Col 8: Botó detall
            detail_btn = QPushButton("👁")
            detail_btn.setFixedSize(40, 28)
            detail_btn.setStyleSheet("""
                QPushButton {
                    border: 1px solid #ddd; border-radius: 4px;
                    background: white; font-size: 14px;
                }
                QPushButton:hover { background: #E3F2FD; }
            """)
            detail_btn.clicked.connect(lambda checked, name=sample_name: self._show_detail(name))
            self.results_table.setCellWidget(row, 8, detail_btn)

        # Actualitzar estadístiques
        total = n_ok + n_warning + n_error
        self.stats_label.setText(
            f"<b>Total:</b> {total} mostres &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"<span style='color:#27AE60'>●</span> OK: {n_ok} &nbsp;&nbsp;"
            f"<span style='color:#F39C12'>●</span> Warning: {n_warning} &nbsp;&nbsp;"
            f"<span style='color:#E74C3C'>●</span> Error: {n_error}"
        )

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

    def _on_replica_changed(self, sample_name):
        """Gestiona el canvi de rèplica seleccionada."""
        if sample_name not in self.samples_grouped:
            return

        # Trobar la fila
        for row in range(self.results_table.rowCount()):
            item = self.results_table.item(row, 0)
            if item and item.data(Qt.UserRole) == sample_name:
                combo = self.results_table.cellWidget(row, 1)
                if combo:
                    new_replica = combo.currentData()
                    self.samples_grouped[sample_name]["selected"]["doc"] = new_replica
                    self.samples_grouped[sample_name]["selected"]["dad"] = new_replica

                    # Recalcular quantificació
                    self._update_quantification(sample_name)

                    # Actualitzar fila
                    self._update_row(row, sample_name)
                break

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

    def _update_row(self, row, sample_name):
        """Actualitza una fila de la taula."""
        sample_data = self.samples_grouped[sample_name]
        selected = sample_data.get("selected", {})
        doc_sel = selected.get("doc", "1")

        replicas = sample_data.get("replicas", {})
        rep_data = replicas.get(doc_sel, {})
        quantification = sample_data.get("quantification", {})

        # SNR DOC
        snr_info = rep_data.get("snr_info", {})
        snr_direct = snr_info.get("snr_direct", 0)
        snr_uib = snr_info.get("snr_uib", 0)

        if snr_uib and snr_uib > 0:
            snr_text = f"{snr_direct:.0f} / {snr_uib:.0f}"
        else:
            snr_text = f"{snr_direct:.0f}" if snr_direct else "-"
        self.results_table.item(row, 3).setText(snr_text)

        # SNR DAD
        snr_dad_text = self._format_snr_dad(rep_data)
        self.results_table.item(row, 4).setText(snr_dad_text)

        # Àrea
        area = rep_data.get("area_total", 0)
        self.results_table.item(row, 5).setText(f"{area:.0f}" if area else "-")

        # Concentració
        conc = quantification.get("concentration_ppm") if quantification else None
        self.results_table.item(row, 6).setText(f"{conc:.2f}" if conc is not None else "-")

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

        fractions = rep_data.get("fractions", {})

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
