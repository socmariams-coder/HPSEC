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
import logging
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

# Importar widgets essencials (només Tab 0 — visible a l'inici)
from gui.widgets.dashboard_panel import DashboardPanel
# Tabs 1-7: lazy import quan l'usuari hi clica (estalvia ~3s d'arrencada)


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

        # Tabs 1-7: placeholders (lazy loading — s'instancien al primer clic)
        self._lazy_tabs = {}
        lazy_tab_defs = [
            (1, "▶ Processar", "process_panel", "gui.widgets.process_wizard_panel", "ProcessWizardPanel"),
            (2, "📄 Exportar", "export_panel", "gui.widgets.export_panel", "ExportPanel"),
            (3, "🔬 Mostres", "samples_db_panel", "gui.widgets.samples_db_panel", "SamplesDBPanel"),
            (4, "📊 Històric", "history_panel", "gui.widgets.history_panel", "HistoryPanel"),
            (5, "📐 Calibració Global", "global_cal_panel", "gui.widgets.global_calibration_panel", "GlobalCalibrationPanel"),
            (6, "🔧 Manteniment", "maintenance_panel", "gui.widgets.maintenance_panel", "MaintenancePanel"),
            (7, "⚙ Configuració", "config_panel", "gui.widgets.config_panel", "ConfigPanel"),
        ]
        for tab_idx, label, attr_name, module_path, class_name in lazy_tab_defs:
            placeholder = QWidget()
            self.tab_widget.addTab(placeholder, label)
            self._lazy_tabs[tab_idx] = (attr_name, module_path, class_name)
            setattr(self, attr_name, None)  # Inicialitzar a None

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
        """Handler quan canvia el tab. Lazy loading de tabs 1-7."""
        if index in self._lazy_tabs:
            attr_name, module_path, class_name = self._lazy_tabs.pop(index)
            import importlib
            module = importlib.import_module(module_path)
            panel_class = getattr(module, class_name)
            panel = panel_class(self)
            setattr(self, attr_name, panel)
            # Inserir panell real dins el placeholder (manté índexs estables)
            placeholder = self.tab_widget.widget(index)
            layout = QVBoxLayout(placeholder)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(panel)
            # Connectar senyals post-creació
            if attr_name == "process_panel":
                panel.process_completed.connect(self._on_process_completed)
                panel.sequence_loaded.connect(self._on_wizard_sequence_loaded)

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

    def _ensure_panel(self, tab_index):
        """Assegura que un panell lazy s'ha creat (forçant _on_tab_changed)."""
        if tab_index in self._lazy_tabs:
            self._on_tab_changed(tab_index)

    def go_to_process_step(self, step_index):
        """
        Navega a una etapa específica del process wizard.
        0=Importar, 1=Calibrar, 2=Analitzar, 3=Consolidar
        """
        # Assegurar que estem al tab de Processar
        self._ensure_panel(1)
        self.tab_widget.setCurrentIndex(1)  # Tab "Processar"
        # Navegar dins del wizard
        if self.process_panel and hasattr(self.process_panel, 'tab_widget'):
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

        # Carregar al wizard (assegurar que existeix)
        self._ensure_panel(1)
        if self.process_panel:
            self.process_panel.load_sequence_from_dashboard(seq_path)

        return True

    def _on_sequence_selected(self, seq_path, phase):
        """Callback quan es selecciona una seqüència al Dashboard."""
        import os
        seq_name = os.path.basename(seq_path)

        try:
            self.set_status(f"Carregant {seq_name}...")

            # SEQ_CAL → directament al tab Calibració Global (tab 5)
            if "_CAL" in seq_name.upper():
                self._load_seq_cal(seq_path)
            else:
                # Flux normal: wizard de 4 passos
                self.load_sequence(seq_path)
                self.tab_widget.setCurrentIndex(1)

            self.set_status(f"{seq_name} carregat", 3000)
        finally:
            # Sempre restaurar cursor i overlay, fins i tot si hi ha error
            self.dashboard_panel.hide_loading_overlay()

    def _load_seq_cal(self, seq_path):
        """Carrega una SEQ_CAL directament al tab Calibració Global (tab 5)."""
        import os
        seq_name = os.path.basename(seq_path)

        # Actualitzar estat i títol
        self.seq_path = seq_path
        self.setWindowTitle(f"HPSEC Suite - {seq_name}")

        # Assegurar que el panell Calibració Global existeix (lazy loading)
        self._ensure_panel(5)

        # Navegar al tab 5
        self.tab_widget.setCurrentIndex(5)

        # Carregar la SEQ_CAL al panell
        if self.global_cal_panel:
            self.global_cal_panel.load_seq_cal(seq_path)

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
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

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
