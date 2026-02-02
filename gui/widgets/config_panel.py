"""
HPSEC Suite - Configuration Panel
==================================

Panel per gestionar la configuració del sistema.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGroupBox,
    QGridLayout, QFrame, QSpinBox, QDoubleSpinBox, QComboBox,
    QCheckBox, QLineEdit, QMessageBox, QScrollArea, QTabWidget,
    QFileDialog
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from pathlib import Path
import json


class ConfigPanel(QWidget):
    """Panel de configuració del sistema."""

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.config = {}
        self._setup_ui()
        self._load_config()

    def _setup_ui(self):
        """Configura la interfície."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        # Títol
        title = QLabel("Configuració")
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        layout.addWidget(title)

        # Tabs de configuració
        self.tabs = QTabWidget()

        # Tab General
        self.tabs.addTab(self._create_general_tab(), "General")

        # Tab Detecció
        self.tabs.addTab(self._create_detection_tab(), "Detecció")

        # Tab Calibració
        self.tabs.addTab(self._create_calibration_tab(), "Calibració")

        # Tab Paths
        self.tabs.addTab(self._create_paths_tab(), "Paths")

        layout.addWidget(self.tabs)

        # Botons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.reset_btn = QPushButton("Restaurar Defectes")
        self.reset_btn.clicked.connect(self._reset_defaults)
        btn_layout.addWidget(self.reset_btn)

        self.save_btn = QPushButton("Guardar Configuració")
        self.save_btn.clicked.connect(self._save_config)
        self.save_btn.setStyleSheet("QPushButton { padding: 8px 16px; font-weight: bold; }")
        btn_layout.addWidget(self.save_btn)

        layout.addLayout(btn_layout)

    def _create_general_tab(self):
        """Crea el tab de configuració general."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(16)

        # Grup: Opcions generals
        general_group = QGroupBox("Opcions Generals")
        general_layout = QGridLayout(general_group)

        general_layout.addWidget(QLabel("Idioma:"), 0, 0)
        self.language_combo = QComboBox()
        self.language_combo.addItem("Català", "ca")
        self.language_combo.addItem("Castellano", "es")
        self.language_combo.addItem("English", "en")
        general_layout.addWidget(self.language_combo, 0, 1)

        general_layout.addWidget(QLabel("Format data:"), 1, 0)
        self.date_format_combo = QComboBox()
        self.date_format_combo.addItem("DD/MM/YYYY", "dd/MM/yyyy")
        self.date_format_combo.addItem("YYYY-MM-DD", "yyyy-MM-dd")
        self.date_format_combo.addItem("MM/DD/YYYY", "MM/dd/yyyy")
        general_layout.addWidget(self.date_format_combo, 1, 1)

        self.auto_backup_cb = QCheckBox("Crear còpies de seguretat automàtiques")
        self.auto_backup_cb.setChecked(True)
        general_layout.addWidget(self.auto_backup_cb, 2, 0, 1, 2)

        self.show_tooltips_cb = QCheckBox("Mostrar tooltips d'ajuda")
        self.show_tooltips_cb.setChecked(True)
        general_layout.addWidget(self.show_tooltips_cb, 3, 0, 1, 2)

        layout.addWidget(general_group)

        # Grup: Exportació
        export_group = QGroupBox("Exportació per Defecte")
        export_layout = QGridLayout(export_group)

        self.export_pdf_cb = QCheckBox("Generar PDF")
        self.export_pdf_cb.setChecked(True)
        export_layout.addWidget(self.export_pdf_cb, 0, 0)

        self.export_excel_cb = QCheckBox("Generar Excel")
        self.export_excel_cb.setChecked(True)
        export_layout.addWidget(self.export_excel_cb, 0, 1)

        self.export_csv_cb = QCheckBox("Generar CSV")
        self.export_csv_cb.setChecked(False)
        export_layout.addWidget(self.export_csv_cb, 1, 0)

        layout.addWidget(export_group)

        layout.addStretch()
        return widget

    def _create_detection_tab(self):
        """Crea el tab de configuració de detecció."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(16)

        # Grup: Thresholds
        thresh_group = QGroupBox("Llindars de Detecció")
        thresh_layout = QGridLayout(thresh_group)

        thresh_layout.addWidget(QLabel("Llindar soroll (mAU):"), 0, 0)
        self.noise_thresh = QDoubleSpinBox()
        self.noise_thresh.setRange(1.0, 100.0)
        self.noise_thresh.setValue(20.0)
        self.noise_thresh.setDecimals(1)
        thresh_layout.addWidget(self.noise_thresh, 0, 1)

        thresh_layout.addWidget(QLabel("SNR mínim:"), 1, 0)
        self.snr_thresh = QDoubleSpinBox()
        self.snr_thresh.setRange(1.0, 50.0)
        self.snr_thresh.setValue(10.0)
        self.snr_thresh.setDecimals(1)
        thresh_layout.addWidget(self.snr_thresh, 1, 1)

        thresh_layout.addWidget(QLabel("Pearson mínim:"), 2, 0)
        self.pearson_thresh = QDoubleSpinBox()
        self.pearson_thresh.setRange(0.900, 1.000)
        self.pearson_thresh.setValue(0.995)
        self.pearson_thresh.setDecimals(3)
        self.pearson_thresh.setSingleStep(0.001)
        thresh_layout.addWidget(self.pearson_thresh, 2, 1)

        layout.addWidget(thresh_group)

        # Grup: Timeout
        timeout_group = QGroupBox("Detecció Timeout (TOC)")
        timeout_layout = QGridLayout(timeout_group)

        timeout_layout.addWidget(QLabel("Llindar timeout (s):"), 0, 0)
        self.timeout_thresh = QSpinBox()
        self.timeout_thresh.setRange(30, 120)
        self.timeout_thresh.setValue(60)
        timeout_layout.addWidget(self.timeout_thresh, 0, 1)

        timeout_layout.addWidget(QLabel("Zona pre-timeout (min):"), 1, 0)
        self.pre_zone = QDoubleSpinBox()
        self.pre_zone.setRange(0.0, 2.0)
        self.pre_zone.setValue(0.5)
        self.pre_zone.setDecimals(1)
        timeout_layout.addWidget(self.pre_zone, 1, 1)

        timeout_layout.addWidget(QLabel("Zona post-timeout (min):"), 2, 0)
        self.post_zone = QDoubleSpinBox()
        self.post_zone.setRange(0.0, 3.0)
        self.post_zone.setValue(1.0)
        self.post_zone.setDecimals(1)
        timeout_layout.addWidget(self.post_zone, 2, 1)

        layout.addWidget(timeout_group)

        # Grup: Batman
        batman_group = QGroupBox("Detecció Batman")
        batman_layout = QGridLayout(batman_group)

        batman_layout.addWidget(QLabel("Separació màxima pics (min):"), 0, 0)
        self.batman_sep = QDoubleSpinBox()
        self.batman_sep.setRange(0.1, 2.0)
        self.batman_sep.setValue(0.5)
        self.batman_sep.setDecimals(2)
        batman_layout.addWidget(self.batman_sep, 0, 1)

        batman_layout.addWidget(QLabel("Factor reparació:"), 1, 0)
        self.repair_factor = QDoubleSpinBox()
        self.repair_factor.setRange(0.5, 1.0)
        self.repair_factor.setValue(0.85)
        self.repair_factor.setDecimals(2)
        batman_layout.addWidget(self.repair_factor, 1, 1)

        layout.addWidget(batman_group)

        layout.addStretch()
        return widget

    def _create_calibration_tab(self):
        """Crea el tab de configuració de calibració."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(16)

        # Grup: Paràmetres KHP
        khp_group = QGroupBox("Paràmetres KHP")
        khp_layout = QGridLayout(khp_group)

        khp_layout.addWidget(QLabel("Concentració per defecte (ppm):"), 0, 0)
        self.khp_conc_default = QDoubleSpinBox()
        self.khp_conc_default.setRange(1.0, 20.0)
        self.khp_conc_default.setValue(5.0)
        self.khp_conc_default.setDecimals(1)
        khp_layout.addWidget(self.khp_conc_default, 0, 1)

        khp_layout.addWidget(QLabel("RSD màxim per mitjana (%):"), 1, 0)
        self.rsd_max = QDoubleSpinBox()
        self.rsd_max.setRange(1.0, 30.0)
        self.rsd_max.setValue(10.0)
        self.rsd_max.setDecimals(1)
        khp_layout.addWidget(self.rsd_max, 1, 1)

        khp_layout.addWidget(QLabel("Quality score màxim (vàlid):"), 2, 0)
        self.quality_max = QSpinBox()
        self.quality_max.setRange(50, 200)
        self.quality_max.setValue(100)
        khp_layout.addWidget(self.quality_max, 2, 1)

        layout.addWidget(khp_group)

        # Grup: Volums injecció
        vol_group = QGroupBox("Volums d'Injecció per Defecte")
        vol_layout = QGridLayout(vol_group)

        vol_layout.addWidget(QLabel("COLUMN (µL):"), 0, 0)
        self.vol_column = QSpinBox()
        self.vol_column.setRange(50, 1000)
        self.vol_column.setValue(400)
        vol_layout.addWidget(self.vol_column, 0, 1)

        vol_layout.addWidget(QLabel("BP (µL):"), 1, 0)
        self.vol_bp = QSpinBox()
        self.vol_bp.setRange(50, 500)
        self.vol_bp.setValue(100)
        vol_layout.addWidget(self.vol_bp, 1, 1)

        layout.addWidget(vol_group)

        # Grup: Històric
        hist_group = QGroupBox("Comparació Històrica")
        hist_layout = QGridLayout(hist_group)

        hist_layout.addWidget(QLabel("Mínim calibracions per mitjana:"), 0, 0)
        self.min_cals_avg = QSpinBox()
        self.min_cals_avg.setRange(2, 10)
        self.min_cals_avg.setValue(2)
        hist_layout.addWidget(self.min_cals_avg, 0, 1)

        self.use_historical_fallback = QCheckBox("Usar mitjana històrica si KHP falla")
        self.use_historical_fallback.setChecked(True)
        hist_layout.addWidget(self.use_historical_fallback, 1, 0, 1, 2)

        layout.addWidget(hist_group)

        layout.addStretch()
        return widget

    def _create_paths_tab(self):
        """Crea el tab de configuració de paths."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(16)

        # Grup: Directoris
        dirs_group = QGroupBox("Directoris")
        dirs_layout = QGridLayout(dirs_group)

        # Directori dades
        dirs_layout.addWidget(QLabel("Directori dades:"), 0, 0)
        self.data_dir_edit = QLineEdit()
        self.data_dir_edit.setPlaceholderText("Per defecte: ~/HPSEC_Data")
        dirs_layout.addWidget(self.data_dir_edit, 0, 1)
        data_browse = QPushButton("...")
        data_browse.setMaximumWidth(30)
        data_browse.clicked.connect(lambda: self._browse_dir(self.data_dir_edit))
        dirs_layout.addWidget(data_browse, 0, 2)

        # Directori exportacions
        dirs_layout.addWidget(QLabel("Directori exportacions:"), 1, 0)
        self.export_dir_edit = QLineEdit()
        self.export_dir_edit.setPlaceholderText("Per defecte: dins de cada SEQ")
        dirs_layout.addWidget(self.export_dir_edit, 1, 1)
        export_browse = QPushButton("...")
        export_browse.setMaximumWidth(30)
        export_browse.clicked.connect(lambda: self._browse_dir(self.export_dir_edit))
        dirs_layout.addWidget(export_browse, 1, 2)

        # Directori històric
        dirs_layout.addWidget(QLabel("Fitxer històric KHP:"), 2, 0)
        self.history_file_edit = QLineEdit()
        self.history_file_edit.setPlaceholderText("Per defecte: khp_calibration_history.json")
        dirs_layout.addWidget(self.history_file_edit, 2, 1)

        layout.addWidget(dirs_group)

        # Info
        info = QLabel(
            "Nota: Els paths relatius són relatius al directori de la SEQ actual."
        )
        info.setStyleSheet("color: #666; font-style: italic;")
        info.setWordWrap(True)
        layout.addWidget(info)

        layout.addStretch()
        return widget

    def _browse_dir(self, line_edit):
        """Obre diàleg per seleccionar directori."""
        path = QFileDialog.getExistingDirectory(self, "Selecciona Directori")
        if path:
            line_edit.setText(path)

    def _get_config_file(self):
        """Retorna el path del fitxer de configuració."""
        config_dir = Path.home() / "HPSEC_Data"
        config_dir.mkdir(exist_ok=True)
        return config_dir / "hpsec_config.json"

    def _load_config(self):
        """Carrega la configuració des del fitxer."""
        config_file = self._get_config_file()
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
                self._apply_config_to_ui()
            except Exception as e:
                print(f"[WARNING] Error carregant configuració: {e}")

    def _apply_config_to_ui(self):
        """Aplica la configuració als widgets."""
        # General
        lang = self.config.get('language', 'ca')
        idx = self.language_combo.findData(lang)
        if idx >= 0:
            self.language_combo.setCurrentIndex(idx)

        # Detection
        self.noise_thresh.setValue(self.config.get('noise_threshold', 20.0))
        self.snr_thresh.setValue(self.config.get('snr_threshold', 10.0))
        self.pearson_thresh.setValue(self.config.get('pearson_threshold', 0.995))
        self.timeout_thresh.setValue(self.config.get('timeout_threshold_sec', 60))

        # Calibration
        self.khp_conc_default.setValue(self.config.get('khp_conc_default', 5.0))
        self.rsd_max.setValue(self.config.get('rsd_max', 10.0))
        self.quality_max.setValue(self.config.get('quality_max', 100))
        self.vol_column.setValue(self.config.get('volume_column', 400))
        self.vol_bp.setValue(self.config.get('volume_bp', 100))

        # Paths
        self.data_dir_edit.setText(self.config.get('data_dir', ''))
        self.export_dir_edit.setText(self.config.get('export_dir', ''))
        self.history_file_edit.setText(self.config.get('history_file', ''))

    def _save_config(self):
        """Guarda la configuració."""
        self.config = {
            # General
            'language': self.language_combo.currentData(),
            'date_format': self.date_format_combo.currentData(),
            'auto_backup': self.auto_backup_cb.isChecked(),
            'show_tooltips': self.show_tooltips_cb.isChecked(),
            'export_pdf': self.export_pdf_cb.isChecked(),
            'export_excel': self.export_excel_cb.isChecked(),
            'export_csv': self.export_csv_cb.isChecked(),

            # Detection
            'noise_threshold': self.noise_thresh.value(),
            'snr_threshold': self.snr_thresh.value(),
            'pearson_threshold': self.pearson_thresh.value(),
            'timeout_threshold_sec': self.timeout_thresh.value(),
            'pre_zone_min': self.pre_zone.value(),
            'post_zone_min': self.post_zone.value(),
            'batman_max_sep': self.batman_sep.value(),
            'repair_factor': self.repair_factor.value(),

            # Calibration
            'khp_conc_default': self.khp_conc_default.value(),
            'rsd_max': self.rsd_max.value(),
            'quality_max': self.quality_max.value(),
            'volume_column': self.vol_column.value(),
            'volume_bp': self.vol_bp.value(),
            'min_cals_average': self.min_cals_avg.value(),
            'use_historical_fallback': self.use_historical_fallback.isChecked(),

            # Paths
            'data_dir': self.data_dir_edit.text(),
            'export_dir': self.export_dir_edit.text(),
            'history_file': self.history_file_edit.text(),
        }

        try:
            config_file = self._get_config_file()
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)

            self.main_window.set_status("Configuració guardada", 3000)
            QMessageBox.information(
                self, "Configuració Guardada",
                f"Configuració guardada a:\n{config_file}"
            )
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error guardant configuració: {e}")

    def _reset_defaults(self):
        """Restaura els valors per defecte."""
        reply = QMessageBox.question(
            self, "Confirmar",
            "Vols restaurar tots els valors per defecte?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            # Detection
            self.noise_thresh.setValue(20.0)
            self.snr_thresh.setValue(10.0)
            self.pearson_thresh.setValue(0.995)
            self.timeout_thresh.setValue(60)
            self.pre_zone.setValue(0.5)
            self.post_zone.setValue(1.0)
            self.batman_sep.setValue(0.5)
            self.repair_factor.setValue(0.85)

            # Calibration
            self.khp_conc_default.setValue(5.0)
            self.rsd_max.setValue(10.0)
            self.quality_max.setValue(100)
            self.vol_column.setValue(400)
            self.vol_bp.setValue(100)
            self.min_cals_avg.setValue(2)

            # Paths
            self.data_dir_edit.clear()
            self.export_dir_edit.clear()
            self.history_file_edit.clear()

            self.main_window.set_status("Valors per defecte restaurats", 3000)

    def get_config(self):
        """Retorna la configuració actual (per usar des d'altres mòduls)."""
        return self.config.copy()
