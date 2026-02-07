"""
HPSEC Suite - Main Window (PySide6)
====================================

Finestra principal amb estructura simplificada:
- Dashboard: Vista general de totes les SEQs
- Processar: Wizard de 4 etapes per noves seqüències
- Exportar: Generació de reports (opcional)
- Auxiliars: Històric, Manteniment, Configuració
"""

import sys
import os
from pathlib import Path
from datetime import datetime

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QPushButton, QFileDialog, QProgressBar,
    QMessageBox, QFrame, QStatusBar
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QAction

# Importar estilos
from gui.styles.theme import STYLESHEET, COLORS

# Importar widgets
from gui.widgets.dashboard_panel import DashboardPanel
from gui.widgets.process_wizard_panel import ProcessWizardPanel
from gui.widgets.export_panel import ExportPanel
from gui.widgets.samples_db_panel import SamplesDBPanel
from gui.widgets.maintenance_panel import MaintenancePanel
from gui.widgets.history_panel import HistoryPanel
from gui.widgets.config_panel import ConfigPanel


class HPSECSuiteWindow(QMainWindow):
    """Finestra principal de HPSEC Suite."""

    def __init__(self):
        super().__init__()

        self.setWindowTitle("HPSEC Suite v2.0")
        self.setMinimumSize(1200, 800)

        # Estat de l'aplicació
        self.seq_path = None
        self.imported_data = None
        self.calibration_data = None
        self.processed_data = None
        self.review_data = None
        self.review_completed = False
        self.manifest_saved = False
        self.has_unsaved_changes = False

        # Configurar UI
        self._setup_ui()
        self._setup_menubar()
        self._setup_statusbar()

        # Aplicar estil
        self.setStyleSheet(STYLESHEET)

    def _setup_ui(self):
        """Configura la interfície principal."""
        # Widget central
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Header
        header = self._create_header()
        main_layout.addWidget(header)

        # Tabs principals
        self.tab_widget = QTabWidget()
        self.tab_widget.setDocumentMode(True)
        self.tab_widget.setTabPosition(QTabWidget.North)

        # === TABS PRINCIPALS ===

        # Tab 0: Dashboard - Vista general i selector de seqüències
        self.dashboard_panel = DashboardPanel(self)
        self.dashboard_panel.sequence_selected.connect(self._on_sequence_selected)
        self.tab_widget.addTab(self.dashboard_panel, "📋 Dashboard")

        # Tab 1: Processar - Wizard
        self.process_panel = ProcessWizardPanel(self)
        self.process_panel.process_completed.connect(self._on_process_completed)
        self.process_panel.sequence_loaded.connect(self._on_wizard_sequence_loaded)
        self.tab_widget.addTab(self.process_panel, "▶ Processar")

        # Tab 2: Exportar
        self.export_panel = ExportPanel(self)
        self.tab_widget.addTab(self.export_panel, "📄 Exportar")

        # Tab 3: Mostres (Base de Dades)
        self.samples_db_panel = SamplesDBPanel(self)
        self.tab_widget.addTab(self.samples_db_panel, "🔬 Mostres")

        # Tab 4: Històric (Calibracions)
        self.history_panel = HistoryPanel(self)
        self.tab_widget.addTab(self.history_panel, "📊 Històric")

        # Tab 5: Manteniment
        self.maintenance_panel = MaintenancePanel(self)
        self.tab_widget.addTab(self.maintenance_panel, "🔧 Manteniment")

        # Tab 6: Configuració
        self.config_panel = ConfigPanel(self)
        self.tab_widget.addTab(self.config_panel, "⚙ Configuració")

        # Connectar senyals
        self.tab_widget.currentChanged.connect(self._on_tab_changed)

        main_layout.addWidget(self.tab_widget)

    def _create_header(self):
        """Crea el header amb títol i info."""
        header = QFrame()
        header.setObjectName("header")
        header.setFixedHeight(60)

        layout = QHBoxLayout(header)
        layout.setContentsMargins(20, 10, 20, 10)

        # Títol
        title = QLabel("HPSEC Suite")
        title.setObjectName("headerTitle")
        title.setFont(QFont("Segoe UI", 18, QFont.Bold))
        layout.addWidget(title)

        # Subtítol
        subtitle = QLabel("Anàlisi de NOM per HPSEC-DAD-DOC")
        subtitle.setObjectName("headerSubtitle")
        subtitle.setFont(QFont("Segoe UI", 10))
        layout.addWidget(subtitle)

        layout.addStretch()

        # Info UdG/LEQUIA
        info = QLabel("Serveis Tècnics de Recerca · UdG")
        info.setObjectName("headerInfo")
        info.setFont(QFont("Segoe UI", 9))
        layout.addWidget(info)

        return header

    def _setup_menubar(self):
        """Configura la barra de menú."""
        menubar = self.menuBar()

        # Menú Arxiu
        file_menu = menubar.addMenu("&Arxiu")

        open_action = QAction("&Obrir Seqüència...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._open_sequence)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        exit_action = QAction("&Sortir", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Menú Ajuda
        help_menu = menubar.addMenu("&Ajuda")

        about_action = QAction("&Sobre...", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _setup_statusbar(self):
        """Configura la barra d'estat."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Barra de progrés
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(200)
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)

        self.status_bar.showMessage("Llest")

    def _on_tab_changed(self, index):
        """Handler quan canvia el tab."""
        pass

    def _open_sequence(self):
        """Obre diàleg per seleccionar carpeta SEQ."""
        from hpsec_config import get_config
        cfg = get_config()
        data_folder = cfg.get("paths", "data_folder")

        path = QFileDialog.getExistingDirectory(
            self,
            "Selecciona carpeta SEQ",
            data_folder,
            QFileDialog.ShowDirsOnly
        )
        if path:
            self.load_sequence(path)
            # Anar al wizard
            self.tab_widget.setCurrentIndex(1)

    def _show_about(self):
        """Mostra diàleg Sobre."""
        QMessageBox.about(
            self,
            "Sobre HPSEC Suite",
            """<h3>HPSEC Suite v2.0</h3>
            <p>Anàlisi de NOM per HPSEC-DAD-DOC</p>
            <p>Serveis Tècnics de Recerca<br>
            Universitat de Girona</p>
            <p>LEQUIA Research Group</p>"""
        )

    # === Mètodes per comunicació entre panels ===

    def enable_tab(self, index):
        """Habilita un tab específic."""
        self.tab_widget.setTabEnabled(index, True)

    def go_to_tab(self, index):
        """Navega a un tab específic del main window."""
        self.tab_widget.setCurrentIndex(index)

    def go_to_process_step(self, step_index):
        """
        Navega a una etapa específica del process wizard.
        0=Importar, 1=Calibrar, 2=Analitzar, 3=Consolidar
        """
        # Assegurar que estem al tab de Processar
        self.tab_widget.setCurrentIndex(1)  # Tab "Processar"
        # Navegar dins del wizard
        if hasattr(self.process_panel, 'tab_widget'):
            self.process_panel.tab_widget.setCurrentIndex(step_index)

    def set_status(self, message, timeout=0):
        """Mostra missatge a la barra d'estat."""
        self.status_bar.showMessage(message, timeout)

    def show_progress(self, value, maximum=100):
        """Mostra/actualitza la barra de progrés."""
        if value < 0:
            self.progress_bar.setVisible(False)
        else:
            self.progress_bar.setVisible(True)
            self.progress_bar.setMaximum(maximum)
            self.progress_bar.setValue(value)

    def mark_review_completed(self):
        """Marca que la revisió s'ha completat."""
        self.review_completed = True

    def mark_manifest_saved(self):
        """Marca que el manifest s'ha guardat."""
        self.manifest_saved = True
        self.has_unsaved_changes = False

    def mark_unsaved_changes(self):
        """Marca que hi ha canvis sense guardar."""
        self.has_unsaved_changes = True

    def load_sequence(self, seq_path):
        """Carrega una seqüència."""
        if not os.path.isdir(seq_path):
            return False

        self.seq_path = seq_path
        seq_name = os.path.basename(seq_path)

        # Reset estat
        self.imported_data = None
        self.calibration_data = None
        self.processed_data = None
        self.review_data = None
        self.review_completed = False
        self.manifest_saved = False
        self.has_unsaved_changes = False

        # Actualitzar títol
        self.setWindowTitle(f"HPSEC Suite - {seq_name}")

        # Carregar al wizard
        self.process_panel.load_sequence_from_dashboard(seq_path)

        return True

    def _on_sequence_selected(self, seq_path, phase):
        """Callback quan es selecciona una seqüència al Dashboard."""
        import os
        seq_name = os.path.basename(seq_path)

        self.set_status(f"Carregant {seq_name}...")
        self.load_sequence(seq_path)

        # Anar al wizard - la navegació interna la gestiona el ProcessWizardPanel
        # que detecta automàticament la primera etapa que necessita atenció (warning o pending)
        self.tab_widget.setCurrentIndex(1)

        self.set_status(f"{seq_name} carregat", 3000)

    def _on_wizard_sequence_loaded(self, seq_path):
        """Callback quan el wizard carrega una seqüència."""
        seq_name = os.path.basename(seq_path)
        self.setWindowTitle(f"HPSEC Suite - {seq_name}")

    def _on_process_completed(self, data):
        """Callback quan el wizard completa el procés."""
        self.review_completed = True
        # Habilitar exportació
        self.tab_widget.setTabEnabled(2, True)

    def closeEvent(self, event):
        """Gestiona el tancament de la finestra."""
        # Si no hi ha dades importades, tancar directament
        if self.imported_data is None:
            event.accept()
            return

        # Si la revisió s'ha completat, tancar directament
        if self.review_completed:
            event.accept()
            return

        # Si el manifest està guardat i no hi ha canvis pendents
        if self.manifest_saved and not self.has_unsaved_changes:
            event.accept()
            return

        # Mostrar avís
        if self.has_unsaved_changes:
            msg = (
                "Hi ha canvis sense guardar.\n\n"
                "Vols tancar sense guardar?"
            )
        else:
            msg = (
                "El procés no s'ha completat.\n\n"
                "Si tanques ara, hauràs de repetir el procés.\n"
                "El manifest d'importació es manté guardat.\n\n"
                "Segur que vols tancar?"
            )

        reply = QMessageBox.warning(
            self,
            "Tancar sense completar",
            msg,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


def main():
    """Punt d'entrada principal."""
    app = QApplication(sys.argv)

    # Configurar aplicació
    app.setApplicationName("HPSEC Suite")
    app.setOrganizationName("UdG-LEQUIA")
    app.setStyle("Fusion")

    # Crear i mostrar finestra principal
    window = HPSECSuiteWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
