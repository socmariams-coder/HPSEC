"""
HPSEC Suite - Main Window (PySide6)
====================================

Finestra principal amb estructura simplificada:
- Processar: QStackedWidget (Dashboard + Wizard)
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
    QMessageBox, QFrame, QStatusBar, QStackedWidget
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QAction

# Importar estilos
from gui.styles.theme import STYLESHEET, COLORS

# Importar widgets essencials (Dashboard dins Tab 0 — visible a l'inici)
from gui.widgets.dashboard_panel import DashboardPanel
# Tabs 1-6 + Wizard: lazy import quan l'usuari hi clica (estalvia ~3s d'arrencada)


class HPSECSuiteWindow(QMainWindow):
    """Finestra principal de HPSEC Suite."""

    def __init__(self):
        super().__init__()

        from hpsec_version import SUITE_FULL
        self.setWindowTitle(SUITE_FULL)
        self.setMinimumSize(1200, 800)

        # Estat de l'aplicació
        self.seq_path = None
        self.sibling_paths = []  # [sibB, sibC, ...] — buit si no té siblings
        self.imported_data = None
        self.calibration_data = None
        self.processed_data = None
        self.review_data = None
        self.review_completed = False
        self.manifest_saved = False
        self.has_unsaved_changes = False

        # Siblings: dades independents per cada sibling
        self.sibling_imported = {}    # {path: imported_data}
        self.sibling_calibrated = {}  # {path: calibration_data}
        self.sibling_analyzed = {}    # {path: analysis_data}

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

        # Tab 0: Processar — QStackedWidget (Dashboard + Wizard)
        self._process_stack = QStackedWidget()

        # Page 0: Dashboard (eager)
        self.dashboard_panel = DashboardPanel(self)
        self.dashboard_panel.sequence_selected.connect(self._on_sequence_selected)
        self._process_stack.addWidget(self.dashboard_panel)

        # Page 1: Wizard (lazy — placeholder)
        self._wizard_placeholder = QWidget()
        self._process_stack.addWidget(self._wizard_placeholder)
        self._wizard_loaded = False
        self.process_panel = None

        self.tab_widget.addTab(self._process_stack, "▶ Processar")

        # Tabs 1-6: placeholders (lazy loading — s'instancien al primer clic)
        self._lazy_tabs = {}
        lazy_tab_defs = [
            (1, "📄 Exportar", "export_panel", "gui.widgets.export_panel", "ExportPanel"),
            (2, "🔬 Mostres", "samples_db_panel", "gui.widgets.samples_db_panel", "SamplesDBPanel"),
            (3, "📊 QC / KHP", "history_panel", "gui.widgets.history_panel", "HistoryPanel"),
            (4, "📐 Calibració Global", "global_cal_panel", "gui.widgets.global_calibration_panel", "GlobalCalibrationPanel"),
            (5, "🔧 Manteniment", "maintenance_panel", "gui.widgets.maintenance_panel", "MaintenancePanel"),
            (6, "⚙ Configuració", "config_panel", "gui.widgets.config_panel", "ConfigPanel"),
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

        # Info institucional
        inst_layout = QVBoxLayout()
        inst_layout.setSpacing(0)
        info = QLabel("Serveis Tècnics de Recerca — Universitat de Girona")
        info.setObjectName("headerInfo")
        info.setFont(QFont("Segoe UI", 9))
        info.setAlignment(Qt.AlignRight)
        inst_layout.addWidget(info)
        dev_label = QLabel("desenvolupat per LEQUIA")
        dev_label.setObjectName("headerDev")
        dev_label.setFont(QFont("Segoe UI", 8))
        dev_label.setStyleSheet("color: #999;")
        dev_label.setAlignment(Qt.AlignRight)
        inst_layout.addWidget(dev_label)
        layout.addLayout(inst_layout)

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
        """Handler quan canvia el tab. Lazy loading de tabs 1-6."""
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

    def _open_sequence(self):
        """Obre diàleg per seleccionar carpeta SEQ."""
        from hpsec_config import get_data_folder
        from gui.settings import recall_dir, remember_dir
        start_dir = recall_dir("last_seq_dir", get_data_folder())

        path = QFileDialog.getExistingDirectory(
            self,
            "Selecciona carpeta SEQ",
            start_dir,
            QFileDialog.ShowDirsOnly
        )
        if path:
            remember_dir("last_seq_dir", os.path.dirname(path))
            self.load_sequence(path)
            # Anar al wizard
            self._show_wizard()

    def _show_about(self):
        """Mostra diàleg Sobre."""
        from hpsec_version import SUITE_FULL
        QMessageBox.about(
            self,
            "Sobre HPSEC Suite",
            f"""<h3>{SUITE_FULL}</h3>
            <p>Anàlisi de NOM per HPSEC-DAD-DOC</p>
            <p><b>Serveis Tècnics de Recerca</b><br>
            Universitat de Girona</p>
            <p style='font-size:10px; color:#888;'>
            Desenvolupat per LEQUIA — Laboratori d'Enginyeria Química i Ambiental<br>
            Projecte finançat per l'ACA (Agència Catalana de l'Aigua)</p>"""
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

    def show_dashboard(self):
        """Mostra la llista de seqüències (page 0 del stacked)."""
        self._process_stack.setCurrentIndex(0)
        self.tab_widget.setCurrentIndex(0)

    def _show_wizard(self):
        """Mostra el wizard dins el tab Processar (page 1 del stacked)."""
        self._ensure_wizard()
        self._process_stack.setCurrentIndex(1)
        self.tab_widget.setCurrentIndex(0)

    def _ensure_wizard(self):
        """Crea el wizard lazy si no existeix."""
        if self._wizard_loaded:
            return
        from gui.widgets.process_wizard_panel import ProcessWizardPanel
        self.process_panel = ProcessWizardPanel(self)
        self.process_panel.process_completed.connect(self._on_process_completed)
        self.process_panel.sequence_loaded.connect(self._on_wizard_sequence_loaded)
        layout = QVBoxLayout(self._wizard_placeholder)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.process_panel)
        self._wizard_loaded = True

    def go_to_process_step(self, step_index):
        """
        Navega a una etapa específica del process wizard.
        0=Importar, 1=Verificar, 2=Analitzar, 3=Quantificar, 4=Exportar
        """
        self._show_wizard()
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

    def load_sequence(self, seq_path, siblings=None):
        """Carrega una seqüència (opcionalment amb siblings)."""
        if not os.path.isdir(seq_path):
            return False

        self.seq_path = seq_path
        self.sibling_paths = siblings or []
        seq_name = os.path.basename(seq_path)

        # Reset estat
        self.imported_data = None
        self.calibration_data = None
        self.processed_data = None
        self.review_data = None
        self.review_completed = False
        self.manifest_saved = False
        self.has_unsaved_changes = False

        # Reset siblings
        self.sibling_imported = {}
        self.sibling_calibrated = {}
        self.sibling_analyzed = {}

        # Actualitzar títol
        suffix = f" [+{len(self.sibling_paths)}]" if self.sibling_paths else ""
        self.setWindowTitle(f"HPSEC Suite - {seq_name}{suffix}")

        # Carregar al wizard (assegurar que existeix)
        self._ensure_wizard()
        if self.process_panel:
            self.process_panel.load_sequence_from_dashboard(seq_path, siblings=self.sibling_paths)

        return True

    def _on_sequence_selected(self, seq_path, phase):
        """Callback quan es selecciona una seqüència al Dashboard."""
        import os
        seq_name = os.path.basename(seq_path)

        # Buscar siblings des del SequenceState del dashboard
        siblings = []
        for s in self.dashboard_panel.sequences:
            if s.seq_path == seq_path:
                siblings = s.siblings if hasattr(s, 'siblings') else []
                break

        try:
            self.set_status(f"Carregant {seq_name}...")

            # SEQ_CAL → directament al tab Calibració Global (tab 4)
            if "_CAL" in seq_name.upper():
                self._load_seq_cal(seq_path)
            else:
                # Flux normal: wizard de 4 passos
                self.load_sequence(seq_path, siblings=siblings)
                self._show_wizard()

            self.set_status(f"{seq_name} carregat", 3000)
        finally:
            # Sempre restaurar cursor i overlay, fins i tot si hi ha error
            self.dashboard_panel.hide_loading_overlay()

    def _load_seq_cal(self, seq_path):
        """Carrega una SEQ_CAL directament al tab Calibració Global (tab 4)."""
        import os
        seq_name = os.path.basename(seq_path)

        # Actualitzar estat i títol
        self.seq_path = seq_path
        self.setWindowTitle(f"HPSEC Suite - {seq_name}")

        # Assegurar que el panell Calibració Global existeix (lazy loading)
        self._ensure_panel(4)

        # Navegar al tab 4
        self.tab_widget.setCurrentIndex(4)

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
        self.tab_widget.setTabEnabled(1, True)

    def closeEvent(self, event):
        """Gestiona el tancament de la finestra."""
        if self.imported_data is None:
            event.accept()
            return

        reply = QMessageBox.question(
            self,
            "Tancar",
            "Les etapes completades es desen automàticament.\n\n"
            "Segur que vols tancar?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


def get_log_path():
    """Ruta del fitxer de log de la sessió (carpeta de l'usuari)."""
    log_dir = Path.home() / ".hpsec"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / "hpsec_suite.log"


def main():
    """Punt d'entrada principal."""
    # Configurar logging: consola + fitxer (l'usuari normalment no veu la
    # consola; els missatges d'error de la GUI remeten a aquest fitxer)
    log_path = get_log_path()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
        ],
    )
    logging.getLogger(__name__).info("Log de sessió: %s", log_path)

    # Excepcions no capturades: al log + diàleg visible (abans només stderr,
    # que no existeix quan s'obre amb doble clic)
    import traceback as _tb
    _original_excepthook = sys.excepthook
    def _excepthook(exc_type, exc_val, exc_tb):
        _tb.print_exception(exc_type, exc_val, exc_tb)
        logging.getLogger(__name__).critical(
            "Excepció no capturada", exc_info=(exc_type, exc_val, exc_tb))
        try:
            QMessageBox.critical(
                None, "Error inesperat",
                f"S'ha produït un error inesperat:\n\n{exc_val}\n\n"
                f"El detall tècnic és a:\n{log_path}")
        except Exception:
            pass
        _original_excepthook(exc_type, exc_val, exc_tb)
    sys.excepthook = _excepthook

    app = QApplication(sys.argv)

    # Configurar aplicació
    app.setApplicationName("HPSEC Suite")
    app.setOrganizationName("UdG-STRs")
    app.setStyle("Fusion")

    # Crear i mostrar finestra principal
    window = HPSECSuiteWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
