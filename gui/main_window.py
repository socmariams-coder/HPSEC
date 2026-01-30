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

        # Crear paneles
        self.import_panel = ImportPanel(self)
        self.calibrate_panel = CalibratePanel(self)
        self.process_panel = ProcessPanel(self)
        self.review_panel = ReviewPanel(self)
        self.export_panel = ExportPanel(self)

        # Añadir tabs
        self.tab_widget.addTab(self.import_panel, "1. Importar")
        self.tab_widget.addTab(self.calibrate_panel, "2. Calibrar")
        self.tab_widget.addTab(self.process_panel, "3. Procesar")
        self.tab_widget.addTab(self.review_panel, "4. Revisar")
        self.tab_widget.addTab(self.export_panel, "5. Exportar")

        # Deshabilitar tabs hasta que se complete el anterior
        for i in range(1, 5):
            self.tab_widget.setTabEnabled(i, False)

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
