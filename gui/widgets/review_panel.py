"""
HPSEC Suite - Review Panel
===========================

Panel para la fase 4: Revisión y selección de réplicas.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout,
    QTableWidget, QTableWidgetItem, QHeaderView, QComboBox,
    QFrame, QGroupBox
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont


class ReviewPanel(QWidget):
    """Panel de revisión de réplicas."""

    review_completed = Signal(dict)

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.review_data = {}
        self.replica_combos = {}

        self._setup_ui()

    def _setup_ui(self):
        """Configura la interfaz."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        # Título
        title = QLabel("Revisió de Rèpliques")
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        layout.addWidget(title)

        info = QLabel(
            "Selecciona la rèplica preferida per cada mostra. "
            "AUTO selecciona automàticament la millor segons R² i absència d'anomalies."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        # Tabla de selección
        self.review_table = QTableWidget()
        self.review_table.setColumnCount(5)
        self.review_table.setHorizontalHeaderLabels([
            "Mostra", "Rèpliques", "Selecció", "R² Millor", "Anomalies"
        ])
        self.review_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.review_table.setAlternatingRowColors(True)
        layout.addWidget(self.review_table)

        # Resumen
        self.summary_label = QLabel()
        layout.addWidget(self.summary_label)

        layout.addStretch()

        # Botones
        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch()

        self.refresh_btn = QPushButton("Actualitzar")
        self.refresh_btn.clicked.connect(self._refresh_data)
        buttons_layout.addWidget(self.refresh_btn)

        self.next_btn = QPushButton("Següent →")
        self.next_btn.clicked.connect(self._go_next)
        buttons_layout.addWidget(self.next_btn)

        layout.addLayout(buttons_layout)

    def showEvent(self, event):
        """Se llama cuando el panel se hace visible."""
        super().showEvent(event)
        self._refresh_data()

    def _refresh_data(self):
        """Actualiza los datos del panel."""
        processed_data = self.main_window.processed_data
        if not processed_data:
            return

        # Agrupar por nombre de muestra
        samples_by_name = {}
        all_samples = (
            processed_data.get("samples", []) +
            processed_data.get("khp_samples", []) +
            processed_data.get("control_samples", [])
        )

        for s in all_samples:
            name = s.get("name", "?")
            if name not in samples_by_name:
                samples_by_name[name] = []
            samples_by_name[name].append(s)

        # Llenar tabla
        self.review_table.setRowCount(0)
        self.replica_combos = {}

        for name, replicas in sorted(samples_by_name.items()):
            row = self.review_table.rowCount()
            self.review_table.insertRow(row)

            # Nombre
            self.review_table.setItem(row, 0, QTableWidgetItem(name))

            # Réplicas disponibles
            rep_nums = [str(r.get("replica", "?")) for r in replicas]
            self.review_table.setItem(row, 1, QTableWidgetItem(", ".join(rep_nums)))

            # Combobox de selección
            combo = QComboBox()
            combo.addItem("AUTO")
            for rep_num in rep_nums:
                combo.addItem(rep_num)
            self.replica_combos[name] = combo
            self.review_table.setCellWidget(row, 2, combo)

            # Mejor R²
            best_r2 = max(r.get("peak_info", {}).get("r2", 0) for r in replicas)
            self.review_table.setItem(row, 3, QTableWidgetItem(f"{best_r2:.4f}"))

            # Anomalías (de la mejor réplica)
            best_rep = max(replicas, key=lambda r: r.get("peak_info", {}).get("r2", 0))
            anomalies = ", ".join(best_rep.get("anomalies", [])) or "-"
            self.review_table.setItem(row, 4, QTableWidgetItem(anomalies[:30]))

            # Inicializar review_data
            self.review_data[name] = {
                "selected": "AUTO",
                "replicas": replicas
            }

        # Actualizar resumen
        n_auto = sum(1 for c in self.replica_combos.values() if c.currentText() == "AUTO")
        n_manual = len(self.replica_combos) - n_auto
        self.summary_label.setText(
            f"Total mostres: {len(samples_by_name)} | AUTO: {n_auto} | Manual: {n_manual}"
        )

        self.main_window.enable_tab(4)

    def _go_next(self):
        """Guarda selecciones y navega al siguiente tab."""
        # Guardar selecciones
        for name, combo in self.replica_combos.items():
            self.review_data[name]["selected"] = combo.currentText()

        self.main_window.review_data = self.review_data
        self.main_window.go_to_tab(4)
