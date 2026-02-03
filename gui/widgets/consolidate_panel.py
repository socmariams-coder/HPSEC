"""
HPSEC Suite - Consolidate Panel (Fase 4)
=========================================

Panel per la fase 4: Consolidació multi-seqüència.

Funcionalitats:
- Cercar SEQs relacionades (COLUMN + BP)
- Vincular mostres per nom entre seqüències
- Mostrar dades BP vinculades
- Seleccionar versió BP si n'hi ha múltiples
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QComboBox,
    QGroupBox, QFrame, QAbstractItemView, QMessageBox,
    QProgressBar
)
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QFont, QColor, QBrush

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from hpsec_consolidate import (
    find_related_sequences,
    find_matching_bp_sequence,
    load_bp_data_for_sample,
    detect_seq_type,
)


class ConsolidateWorker(QThread):
    """Worker per cercar i vincular dades BP."""
    progress = Signal(str, int)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, seq_path, samples_grouped):
        super().__init__()
        self.seq_path = seq_path
        self.samples_grouped = samples_grouped

    def run(self):
        try:
            result = {
                "bp_seq_found": None,
                "bp_linked": {},
                "bp_not_found": [],
            }

            # Cercar SEQ BP relacionada
            self.progress.emit("Cercant SEQ BP relacionada...", 10)
            bp_path = find_matching_bp_sequence(self.seq_path)

            if not bp_path:
                self.progress.emit("No s'ha trobat SEQ BP", 100)
                self.finished.emit(result)
                return

            result["bp_seq_found"] = bp_path

            # Vincular mostres
            total = len(self.samples_grouped)
            for i, sample_name in enumerate(self.samples_grouped.keys()):
                pct = 20 + int((i / total) * 70)
                self.progress.emit(f"Vinculant {sample_name}...", pct)

                bp_data = load_bp_data_for_sample(bp_path, sample_name)
                if bp_data:
                    result["bp_linked"][sample_name] = bp_data
                else:
                    result["bp_not_found"].append(sample_name)

            self.progress.emit("Consolidació completada", 100)
            self.finished.emit(result)

        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n\n{traceback.format_exc()}")


class ConsolidatePanel(QWidget):
    """Panel de consolidació multi-seqüència (Fase 4)."""

    review_completed = Signal(dict)  # Mantenim nom per compatibilitat

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.bp_data = {}
        self.worker = None

        self._setup_ui()

    def _setup_ui(self):
        """Configura la interfície."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # === HEADER ===
        header_layout = QHBoxLayout()

        title = QLabel("Consolidació COLUMN + BP")
        title.setFont(QFont("Segoe UI", 14, QFont.Bold))
        header_layout.addWidget(title)

        header_layout.addStretch()

        # Botó cercar
        self.search_btn = QPushButton("🔍 Cercar BP")
        self.search_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498DB; color: white;
                border: none; border-radius: 4px;
                padding: 8px 20px; font-weight: bold;
            }
            QPushButton:hover { background-color: #2980B9; }
            QPushButton:disabled { background-color: #BDC3C7; }
        """)
        self.search_btn.clicked.connect(self._search_bp)
        header_layout.addWidget(self.search_btn)

        layout.addLayout(header_layout)

        # === INFO SEQ ACTUAL ===
        self.current_seq_frame = QFrame()
        self.current_seq_frame.setStyleSheet(
            "background-color: #E8F5E9; border-radius: 4px; padding: 8px;"
        )
        current_layout = QHBoxLayout(self.current_seq_frame)
        current_layout.setContentsMargins(12, 8, 12, 8)

        self.current_seq_label = QLabel("Seqüència actual: -")
        self.current_seq_label.setFont(QFont("Segoe UI", 10))
        current_layout.addWidget(self.current_seq_label)

        current_layout.addStretch()

        self.current_type_label = QLabel()
        self.current_type_label.setFont(QFont("Segoe UI", 10, QFont.Bold))
        current_layout.addWidget(self.current_type_label)

        layout.addWidget(self.current_seq_frame)

        # === PROGRESS ===
        self.progress_frame = QFrame()
        self.progress_frame.setVisible(False)
        progress_layout = QVBoxLayout(self.progress_frame)
        progress_layout.setContentsMargins(0, 0, 0, 0)

        self.progress_label = QLabel()
        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_label)
        progress_layout.addWidget(self.progress_bar)

        layout.addWidget(self.progress_frame)

        # === RESULTAT BP ===
        self.bp_result_frame = QFrame()
        self.bp_result_frame.setVisible(False)
        bp_layout = QVBoxLayout(self.bp_result_frame)
        bp_layout.setContentsMargins(0, 0, 0, 0)

        # Info BP trobat
        self.bp_found_frame = QFrame()
        self.bp_found_frame.setStyleSheet(
            "background-color: #E3F2FD; border-radius: 4px; padding: 8px;"
        )
        bp_found_layout = QHBoxLayout(self.bp_found_frame)

        self.bp_found_label = QLabel()
        self.bp_found_label.setFont(QFont("Segoe UI", 10))
        bp_found_layout.addWidget(self.bp_found_label)

        bp_layout.addWidget(self.bp_found_frame)

        # Taula de vinculació
        self.link_table = QTableWidget()
        self.link_table.setColumnCount(5)
        self.link_table.setHorizontalHeaderLabels([
            "Mostra", "BP Trobat", "Àrea BP", "[ppm] BP", "Status"
        ])
        self.link_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.link_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.link_table.setAlternatingRowColors(True)
        self.link_table.verticalHeader().setVisible(False)

        self.link_table.setStyleSheet("""
            QTableWidget {
                gridline-color: #ddd;
                background-color: white;
                alternate-background-color: #f9f9f9;
            }
            QHeaderView::section {
                background-color: #f5f5f5;
                padding: 6px;
                border: none;
                border-bottom: 2px solid #ddd;
                font-weight: bold;
            }
        """)

        header = self.link_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Fixed)
        header.setSectionResizeMode(2, QHeaderView.Fixed)
        header.setSectionResizeMode(3, QHeaderView.Fixed)
        header.setSectionResizeMode(4, QHeaderView.Fixed)

        self.link_table.setColumnWidth(1, 80)
        self.link_table.setColumnWidth(2, 100)
        self.link_table.setColumnWidth(3, 100)
        self.link_table.setColumnWidth(4, 60)

        bp_layout.addWidget(self.link_table)

        # Resum
        self.bp_stats_label = QLabel()
        self.bp_stats_label.setStyleSheet("color: #666; margin-top: 8px;")
        bp_layout.addWidget(self.bp_stats_label)

        layout.addWidget(self.bp_result_frame, 1)

        # === BOTONS NAVEGACIÓ ===
        nav_layout = QHBoxLayout()
        nav_layout.addStretch()

        self.next_btn = QPushButton("Següent: Exportar →")
        self.next_btn.setEnabled(False)
        self.next_btn.setStyleSheet("font-weight: bold; padding: 8px 16px;")
        self.next_btn.clicked.connect(self._go_next)
        nav_layout.addWidget(self.next_btn)

        layout.addLayout(nav_layout)

    def showEvent(self, event):
        """Es crida quan el panel es fa visible."""
        super().showEvent(event)
        self._update_current_seq_info()

    def _update_current_seq_info(self):
        """Actualitza la info de la seqüència actual."""
        seq_path = self.main_window.seq_path
        if not seq_path:
            self.current_seq_label.setText("Seqüència actual: -")
            self.current_type_label.setText("")
            return

        seq_name = Path(seq_path).name
        seq_type = detect_seq_type(seq_name)

        self.current_seq_label.setText(f"Seqüència actual: <b>{seq_name}</b>")

        if seq_type == "COLUMN":
            self.current_type_label.setText("COLUMN")
            self.current_type_label.setStyleSheet("color: #27AE60;")
            self.search_btn.setEnabled(True)
        elif seq_type == "BP":
            self.current_type_label.setText("BP")
            self.current_type_label.setStyleSheet("color: #3498DB;")
            self.search_btn.setEnabled(False)
            self.search_btn.setToolTip("La seqüència actual ja és BP")
        else:
            self.current_type_label.setText("?")
            self.current_type_label.setStyleSheet("color: #95A5A6;")

    def _search_bp(self):
        """Cerca SEQ BP relacionada."""
        seq_path = self.main_window.seq_path
        processed_data = self.main_window.processed_data

        if not seq_path or not processed_data:
            QMessageBox.warning(self, "Avís", "No hi ha dades processades.")
            return

        samples_grouped = processed_data.get("samples_grouped", {})
        if not samples_grouped:
            QMessageBox.warning(self, "Avís", "No hi ha mostres per consolidar.")
            return

        # Iniciar cerca
        self.search_btn.setEnabled(False)
        self.progress_frame.setVisible(True)
        self.progress_bar.setValue(0)

        self.worker = ConsolidateWorker(seq_path, samples_grouped)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_progress(self, msg, pct):
        """Actualitza el progrés."""
        self.progress_label.setText(msg)
        self.progress_bar.setValue(pct)

    def _on_finished(self, result):
        """Gestiona la finalització de la cerca."""
        self.progress_frame.setVisible(False)
        self.search_btn.setEnabled(True)
        self.bp_result_frame.setVisible(True)

        bp_path = result.get("bp_seq_found")
        self.bp_data = result.get("bp_linked", {})

        if bp_path:
            bp_name = Path(bp_path).name
            self.bp_found_label.setText(f"✓ SEQ BP trobada: <b>{bp_name}</b>")
            self.bp_found_frame.setStyleSheet(
                "background-color: #E8F5E9; border-radius: 4px; padding: 8px;"
            )
        else:
            self.bp_found_label.setText("✗ No s'ha trobat cap SEQ BP relacionada")
            self.bp_found_frame.setStyleSheet(
                "background-color: #FFEBEE; border-radius: 4px; padding: 8px;"
            )

        # Omplir taula
        self._populate_link_table(result)

        # Guardar a processed_data
        if self.main_window.processed_data:
            self.main_window.processed_data["bp_consolidation"] = result

        self.next_btn.setEnabled(True)

    def _on_error(self, error_msg):
        """Gestiona errors."""
        self.progress_frame.setVisible(False)
        self.search_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", f"Error durant la cerca:\n{error_msg}")

    def _populate_link_table(self, result):
        """Omple la taula de vinculació."""
        self.link_table.setRowCount(0)

        samples_grouped = self.main_window.processed_data.get("samples_grouped", {})
        bp_linked = result.get("bp_linked", {})
        bp_not_found = result.get("bp_not_found", [])

        n_linked = 0
        n_not_found = 0

        for sample_name in sorted(samples_grouped.keys()):
            row = self.link_table.rowCount()
            self.link_table.insertRow(row)

            # Col 0: Mostra
            self.link_table.setItem(row, 0, QTableWidgetItem(sample_name))

            # BP data
            bp_data = bp_linked.get(sample_name)

            if bp_data:
                n_linked += 1

                # Col 1: BP trobat
                self.link_table.setItem(row, 1, QTableWidgetItem("✓"))

                # Col 2: Àrea BP
                area = bp_data.get("area_total")
                area_text = f"{area:.0f}" if area else "-"
                self.link_table.setItem(row, 2, QTableWidgetItem(area_text))

                # Col 3: ppm BP
                conc = bp_data.get("concentration_ppm")
                conc_text = f"{conc:.2f}" if conc else "-"
                self.link_table.setItem(row, 3, QTableWidgetItem(conc_text))

                # Col 4: Status
                status_item = QTableWidgetItem("●")
                status_item.setForeground(QBrush(QColor("#27AE60")))
                status_item.setTextAlignment(Qt.AlignCenter)
                self.link_table.setItem(row, 4, status_item)
            else:
                n_not_found += 1

                # Col 1: BP no trobat
                self.link_table.setItem(row, 1, QTableWidgetItem("✗"))

                # Col 2-3: buit
                self.link_table.setItem(row, 2, QTableWidgetItem("-"))
                self.link_table.setItem(row, 3, QTableWidgetItem("-"))

                # Col 4: Status
                status_item = QTableWidgetItem("●")
                status_item.setForeground(QBrush(QColor("#F39C12")))
                status_item.setTextAlignment(Qt.AlignCenter)
                self.link_table.setItem(row, 4, status_item)

        # Resum
        total = n_linked + n_not_found
        self.bp_stats_label.setText(
            f"Vinculades: {n_linked}/{total} mostres | "
            f"No trobades: {n_not_found}"
        )

    def _go_next(self):
        """Navega a la següent fase (Exportar)."""
        self.review_completed.emit(self.main_window.processed_data or {})
