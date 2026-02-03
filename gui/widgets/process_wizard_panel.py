# -*- coding: utf-8 -*-
"""
HPSEC Suite - Process Wizard Panel v2.0
========================================

Panel per processar seqüències amb pestanyes per cada fase:
1. Importar - Llegir dades RAW
2. Calibrar - Validar KHP i calcular factors
3. Analitzar - Detectar anomalies i calcular àrees
4. Revisar - Seleccionar rèpliques

Estructura visual optimitzada:
- Header mínim amb nom SEQ i botó tornar
- Pestanyes per cada fase (màxim espai per contingut)
- Icones d'estat a les pestanyes (✓/⚠/○)
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTabWidget, QFrame, QFileDialog, QMessageBox, QSizePolicy
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont

from pathlib import Path
import sys
import os

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gui.widgets.import_panel import ImportPanel
from gui.widgets.calibrate_panel import CalibratePanel
from gui.widgets.analyze_panel import AnalyzePanel
from gui.widgets.consolidate_panel import ConsolidatePanel


# Colors per estat
COLOR_OK = "#27AE60"
COLOR_WARNING = "#F39C12"
COLOR_ERROR = "#E74C3C"
COLOR_PENDING = "#95A5A6"
COLOR_CURRENT = "#2E86AB"


class ProcessWizardPanel(QWidget):
    """
    Panel per processar seqüències amb pestanyes.

    Cada fase té la seva pestanya amb tot l'espai disponible.
    """

    process_completed = Signal(dict)
    sequence_loaded = Signal(str)

    TAB_NAMES = ["1. Importar", "2. Calibrar", "3. Analitzar", "4. Consolidar"]
    TAB_ICONS = {
        "pending": "○",
        "current": "►",
        "ok": "✓",
        "warning": "⚠",
        "error": "✗",
    }

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.tab_states = ["pending", "pending", "pending", "pending"]

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # === HEADER MÍNIM ===
        header = self._create_minimal_header()
        layout.addWidget(header)

        # === PESTANYES ===
        self.tab_widget = QTabWidget()
        self.tab_widget.setDocumentMode(True)
        self.tab_widget.setTabPosition(QTabWidget.North)

        # Crear panels
        self.import_panel = ImportPanel(self.main_window)
        self.calibrate_panel = CalibratePanel(self.main_window)
        self.analyze_panel = AnalyzePanel(self.main_window)
        self.consolidate_panel = ConsolidatePanel(self.main_window)

        # Afegir pestanyes
        self.tab_widget.addTab(self.import_panel, self._tab_title(0))
        self.tab_widget.addTab(self.calibrate_panel, self._tab_title(1))
        self.tab_widget.addTab(self.analyze_panel, self._tab_title(2))
        self.tab_widget.addTab(self.consolidate_panel, self._tab_title(3))

        # Amagar botons de navegació dels panels (innecessaris amb pestanyes)
        self._hide_panel_navigation()

        # Connectar senyals
        self._connect_panel_signals()
        self.tab_widget.currentChanged.connect(self._on_tab_changed)

        layout.addWidget(self.tab_widget)

    def _create_minimal_header(self) -> QFrame:
        """Crea header mínim amb botó tornar i nom SEQ."""
        frame = QFrame()
        frame.setFixedHeight(40)
        frame.setStyleSheet("background-color: #f8f9fa; border-bottom: 1px solid #dee2e6;")

        layout = QHBoxLayout(frame)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(12)

        # Botó tornar
        self.back_btn = QPushButton("← Dashboard")
        self.back_btn.setFixedWidth(100)
        self.back_btn.setStyleSheet("background-color: transparent; border: none; color: #2E86AB;")
        self.back_btn.setCursor(Qt.PointingHandCursor)
        self.back_btn.clicked.connect(self._go_to_dashboard)
        layout.addWidget(self.back_btn)

        # Separador
        sep = QLabel("|")
        sep.setStyleSheet("color: #ccc;")
        layout.addWidget(sep)

        # Nom SEQ
        self.seq_label = QLabel("Cap seqüència")
        self.seq_label.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.seq_label.setStyleSheet("color: #2E86AB;")
        layout.addWidget(self.seq_label)

        # Info addicional (method/mode)
        self.seq_info = QLabel()
        self.seq_info.setStyleSheet("color: #666;")
        layout.addWidget(self.seq_info)

        layout.addStretch()

        # Navegació entre seqüències
        nav_style = """
            QPushButton {
                background-color: #3498DB; color: white; border: none;
                border-radius: 3px; padding: 4px 10px; font-weight: bold;
            }
            QPushButton:hover { background-color: #2980B9; }
            QPushButton:disabled { background-color: #BDC3C7; color: #7F8C8D; }
        """

        self.prev_seq_btn = QPushButton("◀ Anterior")
        self.prev_seq_btn.setFixedWidth(85)
        self.prev_seq_btn.setStyleSheet(nav_style)
        self.prev_seq_btn.setCursor(Qt.PointingHandCursor)
        self.prev_seq_btn.clicked.connect(self._go_prev_sequence)
        self.prev_seq_btn.setEnabled(False)
        layout.addWidget(self.prev_seq_btn)

        self.next_seq_btn = QPushButton("Següent ▶")
        self.next_seq_btn.setFixedWidth(85)
        self.next_seq_btn.setStyleSheet(nav_style)
        self.next_seq_btn.setCursor(Qt.PointingHandCursor)
        self.next_seq_btn.clicked.connect(self._go_next_sequence)
        self.next_seq_btn.setEnabled(False)
        layout.addWidget(self.next_seq_btn)

        # Separador
        sep2 = QLabel("|")
        sep2.setStyleSheet("color: #ccc;")
        layout.addWidget(sep2)

        # Botó canviar SEQ
        self.select_btn = QPushButton("Canviar...")
        self.select_btn.setFixedWidth(80)
        self.select_btn.clicked.connect(self._select_sequence)
        layout.addWidget(self.select_btn)

        return frame

    def _tab_title(self, index: int) -> str:
        """Genera títol de pestanya amb icona d'estat."""
        state = self.tab_states[index]
        icon = self.TAB_ICONS.get(state, "○")
        return f"{icon} {self.TAB_NAMES[index]}"

    def _update_tab_titles(self):
        """Actualitza els títols de totes les pestanyes."""
        for i in range(4):
            self.tab_widget.setTabText(i, self._tab_title(i))

            # Color segons estat
            state = self.tab_states[i]
            if state == "ok":
                self.tab_widget.tabBar().setTabTextColor(i, Qt.darkGreen)
            elif state == "warning":
                self.tab_widget.tabBar().setTabTextColor(i, Qt.darkYellow)
            elif state == "error":
                self.tab_widget.tabBar().setTabTextColor(i, Qt.red)
            elif state == "current":
                self.tab_widget.tabBar().setTabTextColor(i, Qt.blue)
            else:
                self.tab_widget.tabBar().setTabTextColor(i, Qt.gray)

    def _set_tab_state(self, index: int, state: str):
        """Estableix l'estat d'una pestanya."""
        if 0 <= index < 4:
            self.tab_states[index] = state
            self._update_tab_titles()

    def _hide_panel_navigation(self):
        """Amaga botons de navegació dels panels."""
        for panel in [self.import_panel, self.calibrate_panel,
                      self.analyze_panel, self.consolidate_panel]:
            if hasattr(panel, 'next_btn'):
                panel.next_btn.setVisible(False)
            if hasattr(panel, 'prev_btn'):
                panel.prev_btn.setVisible(False)

    def _connect_panel_signals(self):
        """Connecta senyals dels panels."""
        self.import_panel.import_completed.connect(self._on_import_completed)
        self.calibrate_panel.calibration_completed.connect(self._on_calibrate_completed)
        self.analyze_panel.analyze_completed.connect(self._on_analyze_completed)
        self.consolidate_panel.review_completed.connect(self._on_review_completed)

    def _go_to_dashboard(self):
        """Torna al Dashboard."""
        self.main_window.tab_widget.setCurrentIndex(0)

    def _select_sequence(self):
        """Obre diàleg per seleccionar SEQ."""
        from hpsec_config import get_config
        cfg = get_config()
        data_folder = cfg.get("paths", "data_folder")

        path = QFileDialog.getExistingDirectory(
            self, "Selecciona carpeta SEQ", data_folder, QFileDialog.ShowDirsOnly
        )

        if path:
            self._load_sequence(path)

    def _load_sequence(self, seq_path: str):
        """Carrega una seqüència."""
        if not os.path.isdir(seq_path):
            QMessageBox.warning(self, "Error", f"Carpeta no vàlida:\n{seq_path}")
            return

        seq_name = os.path.basename(seq_path)

        # Actualitzar header
        self.seq_label.setText(seq_name)

        # Detectar method/mode si hi ha manifest
        self._update_seq_info(seq_path)

        # Notificar main_window
        self.main_window.seq_path = seq_path

        # Reset estats
        self.tab_states = ["current", "pending", "pending", "pending"]
        self._update_tab_titles()

        # Anar a primera pestanya
        self.tab_widget.setCurrentIndex(0)

        # Carregar al panel d'import
        self.import_panel.load_from_dashboard(seq_path)

        self.sequence_loaded.emit(seq_path)

    def _update_seq_info(self, seq_path: str):
        """Actualitza info de la seqüència des del manifest."""
        try:
            from hpsec_import import load_manifest
            manifest = load_manifest(seq_path)
            if manifest:
                seq_info = manifest.get("sequence", {})
                method = seq_info.get("method", "")
                mode = seq_info.get("data_mode", "")
                if method or mode:
                    self.seq_info.setText(f"({method} / {mode})")
                    return
        except:
            pass
        self.seq_info.setText("")

    def _on_tab_changed(self, index: int):
        """Quan canvia la pestanya activa."""
        # Marcar com a current si estava pending
        if self.tab_states[index] == "pending":
            self.tab_states[index] = "current"
            self._update_tab_titles()

    def _on_import_completed(self, data):
        """Callback quan import completa."""
        if data and data.get('success'):
            warnings = data.get('warnings', [])
            self._set_tab_state(0, "warning" if warnings else "ok")
            self._set_tab_state(1, "current")
            # Auto-avançar a calibrar
            self.tab_widget.setCurrentIndex(1)
        else:
            self._set_tab_state(0, "error")

    def _on_calibrate_completed(self, data):
        """Callback quan calibració completa."""
        if data:
            if data.get('success'):
                khp_source = data.get('khp_source', '').upper()
                if khp_source in ('LOCAL', 'SEQ', 'DIRECT', 'UIB', 'DUAL'):
                    self._set_tab_state(1, "ok")
                elif 'SIBLING' in khp_source:
                    self._set_tab_state(1, "warning")
                else:
                    self._set_tab_state(1, "warning")
            else:
                self._set_tab_state(1, "error")

            self._set_tab_state(2, "current")
            self.tab_widget.setCurrentIndex(2)
        else:
            self._set_tab_state(1, "error")

    def _on_analyze_completed(self, data):
        """Callback quan anàlisi completa."""
        if data and data.get('success'):
            self._set_tab_state(2, "ok")
            self._set_tab_state(3, "current")
            self.tab_widget.setCurrentIndex(3)
        else:
            self._set_tab_state(2, "error")

    def _on_review_completed(self, data):
        """Callback quan revisió completa."""
        self._set_tab_state(3, "ok")
        self.process_completed.emit(data)

        QMessageBox.information(
            self, "Completat",
            "Seqüència processada correctament.\n\n"
            "Exporta els resultats des de la pestanya 'Exportar'."
        )

    def load_sequence_from_dashboard(self, seq_path: str):
        """Carrega seqüència des del Dashboard."""
        self._load_sequence(seq_path)

    def load_sequence_with_state(self, seq_path: str, states: list = None):
        """
        Carrega seqüència amb estats predefinits.

        Args:
            seq_path: Path de la seqüència
            states: Llista de 4 estats ['ok', 'warning', 'pending', 'pending']
        """
        self._load_sequence(seq_path)

        if states and len(states) == 4:
            self.tab_states = states
            self._update_tab_titles()

            # Anar a primera pestanya no completada
            for i, state in enumerate(states):
                if state in ("pending", "current"):
                    self.tab_widget.setCurrentIndex(i)
                    break

    def set_sequence_list(self, sequences: list, current_path: str = None):
        """
        Estableix la llista de seqüències per navegació.

        Args:
            sequences: Llista de SequenceState o paths
            current_path: Path de la seqüència actual
        """
        self._sequence_list = []
        for seq in sequences:
            if hasattr(seq, 'seq_path'):
                self._sequence_list.append(seq.seq_path)
            else:
                self._sequence_list.append(str(seq))

        self._current_seq_index = -1
        if current_path and current_path in self._sequence_list:
            self._current_seq_index = self._sequence_list.index(current_path)

        self._update_nav_buttons()

    def _update_nav_buttons(self):
        """Actualitza estat dels botons de navegació."""
        if not hasattr(self, '_sequence_list'):
            self._sequence_list = []

        n = len(self._sequence_list)
        idx = getattr(self, '_current_seq_index', -1)

        self.prev_seq_btn.setEnabled(idx > 0)
        self.next_seq_btn.setEnabled(idx >= 0 and idx < n - 1)

        # Tooltip amb info
        if n > 0 and idx >= 0:
            self.prev_seq_btn.setToolTip(
                f"Anterior: {os.path.basename(self._sequence_list[idx-1])}" if idx > 0 else ""
            )
            self.next_seq_btn.setToolTip(
                f"Següent: {os.path.basename(self._sequence_list[idx+1])}" if idx < n - 1 else ""
            )

    def _go_prev_sequence(self):
        """Navega a la seqüència anterior."""
        if not hasattr(self, '_sequence_list') or not self._sequence_list:
            return

        idx = getattr(self, '_current_seq_index', -1)
        if idx > 0:
            current_tab = self.tab_widget.currentIndex()
            self._current_seq_index = idx - 1
            new_path = self._sequence_list[self._current_seq_index]
            self._load_sequence_at_tab(new_path, current_tab)
            self._update_nav_buttons()

    def _go_next_sequence(self):
        """Navega a la seqüència següent."""
        if not hasattr(self, '_sequence_list') or not self._sequence_list:
            return

        idx = getattr(self, '_current_seq_index', -1)
        if idx >= 0 and idx < len(self._sequence_list) - 1:
            current_tab = self.tab_widget.currentIndex()
            self._current_seq_index = idx + 1
            new_path = self._sequence_list[self._current_seq_index]
            self._load_sequence_at_tab(new_path, current_tab)
            self._update_nav_buttons()

    def _load_sequence_at_tab(self, seq_path: str, target_tab: int):
        """Carrega una seqüència i va directament al tab indicat."""
        if not os.path.isdir(seq_path):
            QMessageBox.warning(self, "Error", f"Carpeta no vàlida:\n{seq_path}")
            return

        seq_name = os.path.basename(seq_path)
        self.seq_label.setText(seq_name)
        self._update_seq_info(seq_path)
        self.main_window.seq_path = seq_path

        # Carregar dades segons el tab destí
        from hpsec_import import load_manifest, import_from_manifest

        manifest = load_manifest(seq_path)
        if manifest:
            # Carregar imported_data des del manifest
            imported_data = import_from_manifest(seq_path, manifest)
            if imported_data and imported_data.get('success'):
                self.main_window.imported_data = imported_data

        # Reset dades dels panels per forçar recàrrega
        self.main_window.calibration_data = None
        self.calibrate_panel.calibration_data = None

        # Anar al tab sol·licitat
        self.tab_widget.setCurrentIndex(target_tab)

        # Actualitzar panel corresponent
        if target_tab == 0:
            self.import_panel.load_from_dashboard(seq_path)
        elif target_tab == 1:
            # Calibrar: showEvent farà _check_existing_calibration
            # Forçar refresc
            self.calibrate_panel._check_existing_calibration()
        elif target_tab == 2:
            # Analitzar
            pass
        elif target_tab == 3:
            # Revisar
            pass

        self.sequence_loaded.emit(seq_path)
