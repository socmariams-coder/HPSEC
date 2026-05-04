"""
HPSEC Suite - Compare Panel (Pas 4: Comparar COLUMN <-> BP)
============================================================

Panell per comparar resultats COLUMN amb la SEQ BP twin.
Nomes visible quan la SEQ activa es COLUMN i te una twin BP detectada.

Estats:
  - Twin analitzada: mostra comparacio (scatter, fraccions, Bland-Altman)
  - Twin importada pero no analitzada: missatge + boto per anar al dashboard
  - Twin no importada: missatge informatiu, es pot saltar
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QSizePolicy, QGroupBox
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QColor

import logging
import os
import json

logger = logging.getLogger(__name__)


class ComparePanel(QWidget):
    """Panell de comparacio COLUMN <-> BP."""

    go_to_dashboard = Signal()  # Senyal per tornar al dashboard

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self._twin_state = "unknown"  # "ready", "not_analyzed", "not_imported", "not_applicable"
        self._twin_seq_name = ""
        self._twin_seq_path = ""
        self._twin_match_pct = 0
        self._comparison_data = None

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Header
        self._header_label = QLabel()
        self._header_label.setWordWrap(True)
        font = self._header_label.font()
        font.setPointSize(10)
        self._header_label.setFont(font)
        layout.addWidget(self._header_label)

        # Status frame
        self._status_frame = QFrame()
        self._status_frame.setFrameShape(QFrame.StyledPanel)
        status_layout = QVBoxLayout(self._status_frame)
        status_layout.setContentsMargins(12, 12, 12, 12)
        status_layout.setSpacing(8)

        self._status_icon = QLabel()
        self._status_icon.setAlignment(Qt.AlignCenter)
        font_icon = self._status_icon.font()
        font_icon.setPointSize(24)
        self._status_icon.setFont(font_icon)
        status_layout.addWidget(self._status_icon)

        self._status_text = QLabel()
        self._status_text.setAlignment(Qt.AlignCenter)
        self._status_text.setWordWrap(True)
        status_layout.addWidget(self._status_text)

        self._action_btn = QPushButton()
        self._action_btn.setVisible(False)
        self._action_btn.setFixedHeight(32)
        self._action_btn.clicked.connect(self._on_action)
        status_layout.addWidget(self._action_btn, alignment=Qt.AlignCenter)

        layout.addWidget(self._status_frame)

        # Comparison content (placeholder — es pobla quan twin ready)
        self._comparison_frame = QFrame()
        self._comparison_frame.setVisible(False)
        self._comparison_layout = QVBoxLayout(self._comparison_frame)
        self._comparison_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._comparison_frame)

        layout.addStretch()

    def set_twin_info(self, twin_seq_name, twin_seq_path, twin_match_pct):
        """Configura la informacio de la twin (cridat al carregar SEQ al wizard)."""
        self._twin_seq_name = twin_seq_name
        self._twin_seq_path = twin_seq_path
        self._twin_match_pct = twin_match_pct
        self._detect_twin_state()
        self._update_ui()

    def clear(self):
        """Neteja el panell (sense twin)."""
        self._twin_state = "not_applicable"
        self._twin_seq_name = ""
        self._twin_seq_path = ""
        self._twin_match_pct = 0
        self._comparison_data = None
        self._update_ui()

    def _detect_twin_state(self):
        """Detecta l'estat de la twin BP."""
        if not self._twin_seq_path:
            self._twin_state = "not_applicable"
            return

        # Comprovar si la twin te analysis_result.json
        analysis_path = os.path.join(
            self._twin_seq_path, "CHECK", "data", "analysis_result.json")
        manifest_path = os.path.join(
            self._twin_seq_path, "CHECK", "data", "import_manifest.json")

        if os.path.isfile(analysis_path):
            self._twin_state = "ready"
        elif os.path.isfile(manifest_path):
            self._twin_state = "not_analyzed"
        else:
            self._twin_state = "not_imported"

    def _update_ui(self):
        """Actualitza la UI segons l'estat de la twin."""
        self._comparison_frame.setVisible(False)
        self._action_btn.setVisible(False)

        if self._twin_state == "not_applicable":
            self._header_label.setText(
                "Comparar COLUMN / BP")
            self._status_icon.setText("--")
            self._status_text.setText(
                "Aquesta seqüència no té una twin BP detectada.\n"
                "Es pot saltar aquest pas.")
            self._status_frame.setStyleSheet(
                "QFrame { background: #F8F9FA; border: 1px solid #DEE2E6; border-radius: 4px; }")

        elif self._twin_state == "not_imported":
            self._header_label.setText(
                f"Comparar amb {self._twin_seq_name}")
            self._status_icon.setText("?")
            self._status_text.setText(
                f"La twin {self._twin_seq_name} no ha estat importada.\n"
                f"Importa-la i analitza-la per poder comparar.\n"
                f"({self._twin_match_pct:.0f}% mostres coincidents)")
            self._action_btn.setText("Anar al Dashboard")
            self._action_btn.setVisible(True)
            self._status_frame.setStyleSheet(
                "QFrame { background: #FFF3CD; border: 1px solid #FFEEBA; border-radius: 4px; }")

        elif self._twin_state == "not_analyzed":
            self._header_label.setText(
                f"Comparar amb {self._twin_seq_name}")
            self._status_icon.setText("!")
            self._status_text.setText(
                f"La twin {self._twin_seq_name} està importada però no analitzada.\n"
                f"Analitza-la per poder comparar.\n"
                f"({self._twin_match_pct:.0f}% mostres coincidents)")
            self._action_btn.setText("Anar al Dashboard")
            self._action_btn.setVisible(True)
            self._status_frame.setStyleSheet(
                "QFrame { background: #FFF3CD; border: 1px solid #FFEEBA; border-radius: 4px; }")

        elif self._twin_state == "ready":
            self._header_label.setText(
                f"Comparar amb {self._twin_seq_name} "
                f"({self._twin_match_pct:.0f}% mostres coincidents)")
            self._status_frame.setVisible(False)
            self._comparison_frame.setVisible(True)
            self._load_comparison()

    def _on_action(self):
        """Accio del boto (anar al dashboard)."""
        self.go_to_dashboard.emit()

    def _load_comparison(self):
        """Carrega i mostra la comparacio COLUMN <-> BP."""
        # Netejar contingut anterior
        while self._comparison_layout.count():
            item = self._comparison_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        try:
            # Carregar analisi de la twin
            analysis_path = os.path.join(
                self._twin_seq_path, "CHECK", "data", "analysis_result.json")
            with open(analysis_path, "r", encoding="utf-8") as f:
                twin_data = json.load(f)

            # Carregar analisi propia
            processed = self.main_window.processed_data or {}
            own_samples = processed.get("samples_grouped", {})

            twin_samples = twin_data.get("samples_grouped", {})
            twin_method = twin_data.get("method", "BP")

            # Trobar mostres comunes
            common_names = set(own_samples.keys()) & set(twin_samples.keys())
            # Filtrar blancs/controls
            common_names = {n for n in common_names
                          if not any(tag in n.upper()
                                     for tag in ["MQ", "NAOH", "BUFFER", "KHP", "BLANC"])}

            if not common_names:
                lbl = QLabel("No s'han trobat mostres comunes entre les dues seqüències.")
                lbl.setAlignment(Qt.AlignCenter)
                self._comparison_layout.addWidget(lbl)
                return

            # Resum
            summary = QLabel(
                f"<b>{len(common_names)} mostres comunes</b> entre "
                f"COLUMN i {twin_method}")
            summary.setAlignment(Qt.AlignCenter)
            self._comparison_layout.addWidget(summary)

            # Taula comparativa basica
            self._build_comparison_table(own_samples, twin_samples, common_names)

        except Exception as e:
            logger.error("Error carregant comparacio: %s", e)
            lbl = QLabel(f"Error carregant dades: {e}")
            lbl.setWordWrap(True)
            self._comparison_layout.addWidget(lbl)

    def _build_comparison_table(self, own_samples, twin_samples, common_names):
        """Construeix la taula i grafics de comparacio."""
        from PySide6.QtWidgets import QTableWidget, QTableWidgetItem, QHeaderView

        # Extreure dades per comparar
        rows = []
        for name in sorted(common_names):
            own = own_samples.get(name, {})
            twin = twin_samples.get(name, {})

            own_quant = own.get("quantification", {})
            twin_quant = twin.get("quantification", {})

            ppm_col = own_quant.get("concentration_ppm_direct") or own_quant.get("concentration_ppm")
            ppm_bp = twin_quant.get("concentration_ppm_direct") or twin_quant.get("concentration_ppm")

            rows.append({
                "name": name,
                "ppm_col": ppm_col,
                "ppm_bp": ppm_bp,
            })

        # Taula
        table = QTableWidget()
        table.setColumnCount(5)
        table.setHorizontalHeaderLabels(["Mostra", "ppm COL", "ppm BP", "Diff %", "Ratio"])
        table.setRowCount(len(rows))
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        for i in range(1, 5):
            header.setSectionResizeMode(i, QHeaderView.ResizeToContents)

        for i, r in enumerate(rows):
            table.setItem(i, 0, QTableWidgetItem(r["name"]))
            if r["ppm_col"] is not None:
                table.setItem(i, 1, QTableWidgetItem(f"{r['ppm_col']:.2f}"))
            else:
                table.setItem(i, 1, QTableWidgetItem("-"))
            if r["ppm_bp"] is not None:
                table.setItem(i, 2, QTableWidgetItem(f"{r['ppm_bp']:.2f}"))
            else:
                table.setItem(i, 2, QTableWidgetItem("-"))

            if r["ppm_col"] and r["ppm_bp"] and r["ppm_col"] > 0:
                diff_pct = (r["ppm_bp"] - r["ppm_col"]) / r["ppm_col"] * 100
                ratio = r["ppm_bp"] / r["ppm_col"]
                diff_item = QTableWidgetItem(f"{diff_pct:+.1f}%")
                if abs(diff_pct) > 20:
                    diff_item.setForeground(QColor("#E74C3C"))
                elif abs(diff_pct) > 10:
                    diff_item.setForeground(QColor("#F39C12"))
                table.setItem(i, 3, diff_item)
                table.setItem(i, 4, QTableWidgetItem(f"{ratio:.3f}"))
            else:
                table.setItem(i, 3, QTableWidgetItem("-"))
                table.setItem(i, 4, QTableWidgetItem("-"))

        table.setMinimumHeight(200)
        self._comparison_layout.addWidget(table)

        # TODO: grafics (scatter ppm_col vs ppm_bp, Bland-Altman)

    def reset(self):
        """Reset del panell."""
        self.clear()
