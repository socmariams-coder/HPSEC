"""
HPSEC Suite - Main Window (PySide6)
====================================

Ventana principal con navegación por tabs y estilo moderno.
"""

import sys
from pathlib import Path
from datetime import datetime

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QPushButton, QFileDialog, QProgressBar,
    QMessageBox, QFrame, QSplitter, QStatusBar, QToolBar,
    QSizePolicy, QSpacerItem, QGroupBox, QGridLayout
)
from PySide6.QtCore import Qt, QSize, Signal, QThread
from PySide6.QtGui import QFont, QIcon, QAction

# Importar estilos
from gui.styles.theme import STYLESHEET, COLORS

# Importar widgets personalizados
from gui.widgets.import_panel import ImportPanel
from gui.widgets.calibrate_panel import CalibratePanel
from gui.widgets.process_panel import ProcessPanel
from gui.widgets.review_panel import ReviewPanel
from gui.widgets.export_panel import ExportPanel
from gui.widgets.maintenance_panel import MaintenancePanel
from gui.widgets.history_panel import HistoryPanel
from gui.widgets.config_panel import ConfigPanel


class HPSECSuiteWindow(QMainWindow):
    """Ventana principal de HPSEC Suite."""

    def __init__(self):
        super().__init__()

        self.setWindowTitle("HPSEC Suite v1.0")
        self.setMinimumSize(1200, 800)

        # Estado de la aplicación
        self.seq_path = None
        self.imported_data = None
        self.calibration_data = None
        self.processed_data = None
        self.review_data = None
        self.review_completed = False  # Indica si s'ha completat la revisió
        self.manifest_saved = False  # Indica si el manifest s'ha guardat
        self.has_unsaved_changes = False  # Indica si hi ha canvis sense guardar

        # Configurar UI
        self._setup_ui()
        self._setup_menubar()
        self._setup_statusbar()

        # Aplicar estilo
        self.setStyleSheet(STYLESHEET)

    def _setup_ui(self):
        """Configura la interfaz principal."""
        # Widget central
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Header con título y logo
        header = self._create_header()
        main_layout.addWidget(header)

        # Tabs principales
        self.tab_widget = QTabWidget()
        self.tab_widget.setDocumentMode(True)
        self.tab_widget.setTabPosition(QTabWidget.North)

        # Crear paneles de pipeline
        self.import_panel = ImportPanel(self)
        self.calibrate_panel = CalibratePanel(self)
        self.process_panel = ProcessPanel(self)
        self.review_panel = ReviewPanel(self)
        self.export_panel = ExportPanel(self)

        # Crear paneles auxiliars (fora de pipeline)
        self.maintenance_panel = MaintenancePanel(self)
        self.history_panel = HistoryPanel(self)
        self.config_panel = ConfigPanel(self)

        # Añadir tabs de pipeline
        self.tab_widget.addTab(self.import_panel, "1. Importar")
        self.tab_widget.addTab(self.calibrate_panel, "2. Calibrar")
        self.tab_widget.addTab(self.process_panel, "3. Procesar")
        self.tab_widget.addTab(self.review_panel, "4. Revisar")
        self.tab_widget.addTab(self.export_panel, "5. Exportar")

        # Separador visual i tabs auxiliars (sempre habilitats)
        self.tab_widget.addTab(QWidget(), "")  # Separador buit (índex 5)
        self.tab_widget.setTabEnabled(5, False)  # Separador no clicable
        self.tab_widget.addTab(self.history_panel, "📊 Històric")      # índex 6
        self.tab_widget.addTab(self.maintenance_panel, "🔧 Manteniment")  # índex 7
        self.tab_widget.addTab(self.config_panel, "⚙️ Configuració")    # índex 8

        # Deshabilitar tabs de pipeline fins completar l'anterior
        for i in range(1, 5):
            self.tab_widget.setTabEnabled(i, False)
        # Tabs auxiliars (6, 7, 8) sempre habilitats

        # Conectar señales
        self.tab_widget.currentChanged.connect(self._on_tab_changed)

        main_layout.addWidget(self.tab_widget)

    def _create_header(self):
        """Crea el header con título y info."""
        header = QFrame()
        header.setObjectName("header")
        header.setFixedHeight(60)

        layout = QHBoxLayout(header)
        layout.setContentsMargins(20, 10, 20, 10)

        # Título
        title = QLabel("HPSEC Suite")
        title.setObjectName("headerTitle")
        title.setFont(QFont("Segoe UI", 18, QFont.Bold))
        layout.addWidget(title)

        # Subtítulo
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

        # Menú Archivo
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

        # Menú Ayuda
        help_menu = menubar.addMenu("&Ajuda")

        about_action = QAction("&Sobre...", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _setup_statusbar(self):
        """Configura la barra de estado."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Barra de progreso
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(200)
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)

        self.status_bar.showMessage("Llest")

    def _on_tab_changed(self, index):
        """Handler cuando cambia el tab."""
        pass

    def _open_sequence(self):
        """Abre diálogo para seleccionar carpeta SEQ."""
        path = QFileDialog.getExistingDirectory(
            self,
            "Selecciona carpeta SEQ",
            "",
            QFileDialog.ShowDirsOnly
        )
        if path:
            self.seq_path = path
            self.import_panel.set_sequence_path(path)
            self.tab_widget.setCurrentIndex(0)

    def _show_about(self):
        """Muestra diálogo Acerca de."""
        QMessageBox.about(
            self,
            "Sobre HPSEC Suite",
            """<h3>HPSEC Suite v1.0</h3>
            <p>Anàlisi de NOM per HPSEC-DAD-DOC</p>
            <p>Serveis Tècnics de Recerca<br>
            Universitat de Girona</p>
            <p>LEQUIA Research Group</p>"""
        )

    # === Métodos para comunicación entre paneles ===

    def enable_tab(self, index):
        """Habilita un tab específico."""
        self.tab_widget.setTabEnabled(index, True)

    def go_to_tab(self, index):
        """Navega a un tab específico."""
        self.tab_widget.setCurrentIndex(index)

    def set_status(self, message, timeout=0):
        """Muestra mensaje en la barra de estado."""
        self.status_bar.showMessage(message, timeout)

    def show_progress(self, value, maximum=100):
        """Muestra/actualiza la barra de progreso."""
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

        # Si el manifest està guardat i no hi ha canvis pendents, tancar directament
        if self.manifest_saved and not self.has_unsaved_changes:
            event.accept()
            return

        # Determinar en quina fase estem (només tabs de pipeline)
        current_tab = self.tab_widget.currentIndex()
        tab_names = ["Importar", "Calibrar", "Processar", "Revisar", "Exportar"]
        if current_tab < len(tab_names):
            current_phase = tab_names[current_tab]
        elif current_tab >= 6:
            # Tabs auxiliars (Històric, Manteniment, Configuració) - no cal avís especial
            current_phase = "Auxiliar"
        else:
            current_phase = "?"

        # Mostrar avís
        if self.has_unsaved_changes:
            msg = (
                "Hi ha canvis sense guardar.\n\n"
                "Vols tancar sense guardar?"
            )
        else:
            msg = (
                f"El procés no s'ha completat (fase actual: {current_phase}).\n\n"
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
    """Punto de entrada principal."""
    app = QApplication(sys.argv)

    # Configurar aplicación
    app.setApplicationName("HPSEC Suite")
    app.setOrganizationName("UdG-LEQUIA")
    app.setStyle("Fusion")  # Estilo moderno multiplataforma

    # Crear y mostrar ventana principal
    window = HPSECSuiteWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
