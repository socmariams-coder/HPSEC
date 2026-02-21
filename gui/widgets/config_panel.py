"""
HPSEC Suite - Configuration Panel (v2)
=======================================

Panel per gestionar la configuració del sistema.
3 tabs per impacte: Anàlisi (retroactiu), Seqüència (futur), Sistema (immediat).
Tots els paràmetres editables, badges d'impacte, detecció de canvis.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGroupBox,
    QGridLayout, QFrame, QSpinBox, QDoubleSpinBox, QComboBox,
    QCheckBox, QLineEdit, QMessageBox, QScrollArea, QTabWidget,
    QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView,
    QStyledItemDelegate, QSizePolicy
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont

from hpsec_config import get_config, REPROCESS_SECTIONS, FUTURE_SECTIONS, IMMEDIATE_SECTIONS
from gui.widgets.styles import (
    STYLE_BADGE_RETROACTIVE, STYLE_BADGE_FUTURE, STYLE_SECTION_CHANGED
)

import copy


# =============================================================================
# CUSTOM WIDGETS
# =============================================================================

class ConfigSection(QFrame):
    """Widget reutilitzable per cada secció de configuració."""

    def __init__(self, title, badge_text=None, badge_style=None, parent=None):
        super().__init__(parent)
        self._dirty = False
        self._default_style = ""

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # Header: títol + badge
        header = QHBoxLayout()
        title_label = QLabel(title)
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title_label.setFont(title_font)
        header.addWidget(title_label)

        if badge_text and badge_style:
            badge = QLabel(badge_text)
            badge.setStyleSheet(badge_style)
            badge.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            header.addWidget(badge)

        header.addStretch()
        main_layout.addLayout(header)

        # Content frame
        self.content = QFrame()
        self.content_layout = QGridLayout(self.content)
        self.content_layout.setContentsMargins(4, 4, 4, 4)
        self.content_layout.setSpacing(6)
        main_layout.addWidget(self.content)

    def mark_dirty(self, dirty=True):
        """Ressalta la secció si té canvis."""
        if dirty and not self._dirty:
            self._default_style = self.styleSheet()
            self.setStyleSheet(STYLE_SECTION_CHANGED)
        elif not dirty and self._dirty:
            self.setStyleSheet(self._default_style)
        self._dirty = dirty


class TimeFractionsEditor(QTableWidget):
    """Editor de fraccions temporals amb 5 files fixes."""

    FRACTION_KEYS = ["BioP", "HS", "BB", "SB", "LMW"]
    FRACTION_NAMES = ["Biopolímers", "Àcids Húmics", "Building Blocks", "Salt Boundary", "Low Molecular Weight"]

    def __init__(self, parent=None):
        super().__init__(5, 3, parent)
        self.setHorizontalHeaderLabels(["Fracció", "Inici (min)", "Fi (min)"])
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.setMaximumHeight(180)
        self.verticalHeader().setVisible(False)

        for row, (key, name) in enumerate(zip(self.FRACTION_KEYS, self.FRACTION_NAMES)):
            # Col 0: nom (read-only)
            item = QTableWidgetItem(f"{key} ({name})")
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.setItem(row, 0, item)

    def load(self, fractions_dict):
        """Carrega valors des del dict de config."""
        for row, key in enumerate(self.FRACTION_KEYS):
            frac = fractions_dict.get(key, {})
            start_item = QTableWidgetItem(str(frac.get("start", 0.0)))
            end_item = QTableWidgetItem(str(frac.get("end", 0.0)))
            self.setItem(row, 1, start_item)
            self.setItem(row, 2, end_item)

    def save(self):
        """Retorna dict amb les fraccions."""
        result = {}
        for row, key in enumerate(self.FRACTION_KEYS):
            try:
                start = float(self.item(row, 1).text())
                end = float(self.item(row, 2).text())
            except (ValueError, AttributeError):
                start, end = 0.0, 0.0
            result[key] = {
                "start": start,
                "end": end,
                "name": self.FRACTION_NAMES[row]
            }
        return result

    def validate(self):
        """Valida contigüitat: end[i] == start[i+1]."""
        errors = []
        for row in range(len(self.FRACTION_KEYS) - 1):
            try:
                end_val = float(self.item(row, 2).text())
                next_start = float(self.item(row + 1, 1).text())
                if abs(end_val - next_start) > 0.01:
                    errors.append(
                        f"{self.FRACTION_KEYS[row]} fi ({end_val}) != "
                        f"{self.FRACTION_KEYS[row + 1]} inici ({next_start})"
                    )
            except (ValueError, AttributeError):
                errors.append(f"Valor invàlid a fila {row + 1}")
        return errors


class TimeoutZonesEditor(QTableWidget):
    """Editor de zones timeout amb severitat."""

    ZONE_KEYS = ["RUN_START", "BioP", "HS", "BB", "SB", "LMW", "POST_RUN"]
    SEVERITIES = ["OK", "INFO", "WARNING", "CRITICAL"]

    def __init__(self, parent=None):
        super().__init__(7, 4, parent)
        self.setHorizontalHeaderLabels(["Zona", "Inici (min)", "Fi (min)", "Severitat"])
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.setMaximumHeight(230)
        self.verticalHeader().setVisible(False)

        for row, key in enumerate(self.ZONE_KEYS):
            item = QTableWidgetItem(key)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.setItem(row, 0, item)

    def load(self, zones_dict):
        """Carrega valors des del dict de config."""
        for row, key in enumerate(self.ZONE_KEYS):
            zone = zones_dict.get(key, {})
            self.setItem(row, 1, QTableWidgetItem(str(zone.get("start", 0))))
            self.setItem(row, 2, QTableWidgetItem(str(zone.get("end", 0))))
            # Severity: combo
            combo = QComboBox()
            combo.addItems(self.SEVERITIES)
            sev = zone.get("severity", "OK")
            idx = combo.findText(sev)
            if idx >= 0:
                combo.setCurrentIndex(idx)
            self.setCellWidget(row, 3, combo)

    def save(self):
        """Retorna dict amb les zones."""
        result = {}
        for row, key in enumerate(self.ZONE_KEYS):
            try:
                start = float(self.item(row, 1).text())
                end = float(self.item(row, 2).text())
            except (ValueError, AttributeError):
                start, end = 0, 0
            combo = self.cellWidget(row, 3)
            severity = combo.currentText() if combo else "OK"
            result[key] = {"start": start, "end": end, "severity": severity}
        return result


class PatternListEditor(QLineEdit):
    """Editor de patrons separats per coma."""

    def __init__(self, placeholder="MQ, BLANK, BLK, H2O...", parent=None):
        super().__init__(parent)
        self.setPlaceholderText(placeholder)

    def load(self, patterns_list):
        """Carrega llista de patrons."""
        self.setText(", ".join(patterns_list) if patterns_list else "")

    def save(self):
        """Retorna llista de patrons."""
        text = self.text().strip()
        if not text:
            return []
        return [p.strip() for p in text.split(",") if p.strip()]


class WavelengthSelector(QWidget):
    """Selector de wavelengths amb checkboxes + primary combo."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Grid de checkboxes
        self._checkboxes = {}
        grid_layout = QGridLayout()
        layout.addLayout(grid_layout)

        # Primary combo
        primary_layout = QHBoxLayout()
        primary_layout.addWidget(QLabel("Primary:"))
        self.primary_combo = QComboBox()
        primary_layout.addWidget(self.primary_combo)
        primary_layout.addStretch()
        layout.addLayout(primary_layout)

    def load(self, wavelengths_dict):
        """Carrega des del dict de config."""
        available = wavelengths_dict.get("available", [])
        selected = wavelengths_dict.get("selected", [])
        primary = wavelengths_dict.get("primary", 254)

        # Clear existing checkboxes
        for cb in self._checkboxes.values():
            cb.setParent(None)
        self._checkboxes.clear()

        # Find the grid layout (first child layout)
        grid_layout = self.layout().itemAt(0).layout()
        # Clear grid layout
        while grid_layout.count():
            item = grid_layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)

        # Create checkboxes
        for i, wl in enumerate(sorted(available)):
            cb = QCheckBox(f"{wl} nm")
            cb.setChecked(wl in selected)
            self._checkboxes[wl] = cb
            grid_layout.addWidget(cb, i // 4, i % 4)

        # Primary combo
        self.primary_combo.clear()
        self.primary_combo.addItems([str(wl) for wl in sorted(available)])
        idx = self.primary_combo.findText(str(primary))
        if idx >= 0:
            self.primary_combo.setCurrentIndex(idx)

    def save(self):
        """Retorna dict amb wavelengths."""
        available = sorted(self._checkboxes.keys())
        selected = [wl for wl, cb in sorted(self._checkboxes.items()) if cb.isChecked()]
        try:
            primary = int(self.primary_combo.currentText())
        except (ValueError, AttributeError):
            primary = 254
        return {
            "available": available,
            "selected": selected,
            "primary": primary,
        }


# =============================================================================
# CONFIG PANEL PRINCIPAL
# =============================================================================

class ConfigPanel(QWidget):
    """Panel de configuració del sistema amb 3 tabs per impacte."""

    config_changed = Signal()

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self._original_values = {}
        self._widgets = {}  # key -> widget mapping
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

        # Tabs per impacte
        self.tabs = QTabWidget()
        self.tabs.addTab(self._create_analysis_tab(), "Anàlisi")
        self.tabs.addTab(self._create_sequence_tab(), "Seqüència")
        self.tabs.addTab(self._create_system_tab(), "Sistema")
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

    # =========================================================================
    # TAB ANÀLISI (RETROACTIU)
    # =========================================================================

    def _create_analysis_tab(self):
        """Crea el tab Anàlisi — seccions amb impacte retroactiu."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(16)

        # --- Secció: Detecció ---
        sec_det = ConfigSection("Detecció d'Anomalies", "Retroactiu", STYLE_BADGE_RETROACTIVE)
        g = sec_det.content_layout
        row = 0

        row = self._add_double_spin(g, row, "detection.irregular_top_max_sep_min",
            "Separació màx. cim irregular (min):", 0.1, 2.0, 0.5, 2)
        row = self._add_double_spin(g, row, "detection.irregular_top_drop_min",
            "Caiguda mín. cim irregular:", 0.01, 0.50, 0.05, 2)
        row = self._add_double_spin(g, row, "detection.irregular_top_drop_max",
            "Caiguda màx. cim irregular:", 0.10, 1.00, 0.50, 2)
        row = self._add_double_spin(g, row, "detection.timeout_min_duration",
            "Durada mín. timeout (s):", 1.0, 30.0, 5.0, 1)
        row = self._add_double_spin(g, row, "detection.timeout_major",
            "Timeout major (s):", 30.0, 120.0, 74.0, 1)
        row = self._add_double_spin(g, row, "detection.ears_threshold",
            "Llindar ears (% alçada):", 0.01, 0.50, 0.10, 2)
        row = self._add_double_spin(g, row, "detection.ears_max_sep_min",
            "Separació màx. ears (min):", 0.1, 2.0, 0.5, 2)
        row = self._add_double_spin(g, row, "detection.irr_smoothness_threshold",
            "Llindar irregularitat (%):", 0.05, 0.50, 0.18, 2)

        layout.addWidget(sec_det)

        # --- Secció: Qualitat ---
        sec_qual = ConfigSection("Llindars de Qualitat", "Retroactiu", STYLE_BADGE_RETROACTIVE)
        g = sec_qual.content_layout
        row = 0

        row = self._add_double_spin(g, row, "quality.r2_valid",
            "R² vàlid:", 0.900, 1.000, 0.987, 3, step=0.001)
        row = self._add_double_spin(g, row, "quality.r2_check",
            "R² check:", 0.900, 1.000, 0.980, 3, step=0.001)
        row = self._add_double_spin(g, row, "quality.pearson_min",
            "Pearson mínim:", 0.900, 1.000, 0.995, 3, step=0.001)
        row = self._add_double_spin(g, row, "quality.pearson_warning",
            "Pearson warning:", 0.900, 1.000, 0.990, 3, step=0.001)
        row = self._add_double_spin(g, row, "quality.snr_min",
            "SNR mínim:", 1.0, 50.0, 10.0, 1)
        row = self._add_double_spin(g, row, "quality.snr_ratio_threshold",
            "SNR ratio threshold:", 0.5, 5.0, 1.5, 1)
        row = self._add_double_spin(g, row, "quality.area_diff_warning",
            "Diferència àrea warning (%):", 1.0, 50.0, 15.0, 1)
        row = self._add_double_spin(g, row, "quality.area_diff_critical",
            "Diferència àrea crítica (%):", 5.0, 100.0, 30.0, 1)

        layout.addWidget(sec_qual)

        # --- Secció: Fraccions Temporals ---
        sec_frac = ConfigSection("Fraccions Temporals", "Retroactiu", STYLE_BADGE_RETROACTIVE)
        self.fractions_editor = TimeFractionsEditor()
        self._widgets["time_fractions"] = self.fractions_editor
        sec_frac.content_layout.addWidget(self.fractions_editor, 0, 0, 1, 2)
        layout.addWidget(sec_frac)

        # --- Secció: Zones Timeout ---
        sec_tz = ConfigSection("Zones Timeout (TOC)", "Retroactiu", STYLE_BADGE_RETROACTIVE)
        self.timeout_zones_editor = TimeoutZonesEditor()
        self._widgets["timeout_zones"] = self.timeout_zones_editor
        sec_tz.content_layout.addWidget(self.timeout_zones_editor, 0, 0, 1, 2)
        layout.addWidget(sec_tz)

        # --- Secció: Baseline ---
        sec_bl = ConfigSection("Càlcul Baseline", "Retroactiu", STYLE_BADGE_RETROACTIVE)
        g = sec_bl.content_layout
        row = 0

        row = self._add_int_spin(g, row, "baseline.bp_end_pct",
            "BP final (% cromatograma):", 5, 50, 20)
        row = self._add_int_spin(g, row, "baseline.column_start_pct",
            "COLUMN inici (% cromatograma):", 5, 50, 15)
        row = self._add_combo(g, row, "baseline.method",
            "Mètode:", ["mode", "median"], "mode")
        row = self._add_int_spin(g, row, "baseline.stats_percentile_low",
            "Percentil baix:", 1, 20, 5)
        row = self._add_int_spin(g, row, "baseline.stats_percentile_high",
            "Percentil alt:", 20, 80, 40)
        row = self._add_double_spin(g, row, "baseline.min_noise_mau",
            "Soroll mínim (mAU):", 0.001, 1.0, 0.01, 3, step=0.001)

        layout.addWidget(sec_bl)

        # --- Secció: Cromatograma ---
        sec_chrom = ConfigSection("Cromatograma", "Retroactiu", STYLE_BADGE_RETROACTIVE)
        g = sec_chrom.content_layout
        row = 0

        row = self._add_double_spin(g, row, "chromatogram.max_duration_min",
            "Durada màxima (min):", 10.0, 120.0, 78.65, 2)
        row = self._add_double_spin(g, row, "chromatogram.baseline_window_bp",
            "Finestra baseline BP (min):", 0.1, 5.0, 1.0, 1)
        row = self._add_double_spin(g, row, "chromatogram.baseline_window_column",
            "Finestra baseline COLUMN (min):", 1.0, 30.0, 10.0, 1)
        row = self._add_int_spin(g, row, "chromatogram.smoothing_window",
            "Finestra suavitzat:", 3, 51, 11)
        row = self._add_int_spin(g, row, "chromatogram.smoothing_order",
            "Ordre suavitzat:", 1, 5, 3)

        layout.addWidget(sec_chrom)

        # --- Secció: Wavelengths ---
        sec_wl = ConfigSection("Longituds d'Ona (DAD)", "Retroactiu", STYLE_BADGE_RETROACTIVE)
        self.wl_selector = WavelengthSelector()
        self._widgets["wavelengths"] = self.wl_selector
        sec_wl.content_layout.addWidget(self.wl_selector, 0, 0, 1, 2)
        layout.addWidget(sec_wl)

        # --- Secció: DAD ---
        sec_dad = ConfigSection("Paràmetres DAD", "Retroactiu", STYLE_BADGE_RETROACTIVE)
        g = sec_dad.content_layout
        row = 0

        row = self._add_double_spin(g, row, "dad.drift_warning",
            "Deriva warning (mAU):", 0.1, 10.0, 1.0, 1)
        row = self._add_double_spin(g, row, "dad.drift_critical",
            "Deriva crítica (mAU):", 0.5, 20.0, 3.0, 1)
        row = self._add_double_spin(g, row, "dad.noise_warning",
            "Soroll warning (mAU):", 0.01, 5.0, 0.5, 2)
        row = self._add_double_spin(g, row, "dad.doc_correlation_min",
            "Correlació DOC mínima:", 0.5, 1.0, 0.90, 2, step=0.01)
        row = self._add_double_spin(g, row, "dad.doc_correlation_warning",
            "Correlació DOC warning:", 0.5, 1.0, 0.95, 2, step=0.01)

        layout.addWidget(sec_dad)

        layout.addStretch()
        scroll.setWidget(container)
        return scroll

    # =========================================================================
    # TAB SEQÜÈNCIA (FUTUR)
    # =========================================================================

    def _create_sequence_tab(self):
        """Crea el tab Seqüència — seccions amb impacte futur."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(16)

        # --- Secció: Calibració ---
        sec_cal = ConfigSection("Calibració", "Futur", STYLE_BADGE_FUTURE)
        g = sec_cal.content_layout
        row = 0

        row = self._add_double_spin(g, row, "calibration.khp_conc_default",
            "Concentració KHP (ppm):", 1.0, 20.0, 5.0, 1)
        row = self._add_double_spin(g, row, "calibration.rsd_max",
            "RSD màxim (%):", 1.0, 30.0, 10.0, 1)
        row = self._add_int_spin(g, row, "calibration.quality_max",
            "Quality score màxim:", 50, 200, 100)
        row = self._add_int_spin(g, row, "calibration.volume_column",
            "Volum COLUMN (µL):", 50, 1000, 400)
        row = self._add_int_spin(g, row, "calibration.volume_bp",
            "Volum BP (µL):", 50, 500, 100)
        row = self._add_int_spin(g, row, "calibration.min_cals_average",
            "Mín. calibracions per mitjana:", 2, 10, 2)
        row = self._add_checkbox(g, row, "calibration.use_historical_fallback",
            "Usar mitjana històrica si KHP falla", True)

        layout.addWidget(sec_cal)

        # --- Secció: Injeccions Blanc ---
        sec_blank = ConfigSection("Injeccions Blanc", "Futur", STYLE_BADGE_FUTURE)
        g = sec_blank.content_layout
        g.addWidget(QLabel("Patrons de nom:"), 0, 0)
        self.blank_patterns = PatternListEditor("MQ, BLANK, BLK, H2O...")
        self._widgets["blank_injections.patterns"] = self.blank_patterns
        g.addWidget(self.blank_patterns, 0, 1)
        layout.addWidget(sec_blank)

        # --- Secció: Injeccions Control ---
        sec_ctrl = ConfigSection("Injeccions Control", "Futur", STYLE_BADGE_FUTURE)
        g = sec_ctrl.content_layout
        g.addWidget(QLabel("Patrons de nom:"), 0, 0)
        self.control_patterns = PatternListEditor("NAOH, WASH, CONTROL...")
        self._widgets["control_injections.patterns"] = self.control_patterns
        g.addWidget(self.control_patterns, 0, 1)

        self.ignore_orphan_cb = QCheckBox("Ignorar controls orfes (no trobats a 1-HPLC-SEQ)")
        self._widgets["control_injections.ignore_orphan"] = self.ignore_orphan_cb
        g.addWidget(self.ignore_orphan_cb, 1, 0, 1, 2)
        layout.addWidget(sec_ctrl)

        # --- Secció: Planificació Seqüència ---
        sec_seq = ConfigSection("Planificació Seqüència", "Futur", STYLE_BADGE_FUTURE)
        g = sec_seq.content_layout
        row = 0

        row = self._add_double_spin(g, row, "sequence.sample_duration_column",
            "Durada mostra COLUMN (min):", 10.0, 120.0, 78.65, 2)
        row = self._add_double_spin(g, row, "sequence.sample_duration_bp",
            "Durada mostra BP (min):", 5.0, 30.0, 12.0, 1)
        row = self._add_double_spin(g, row, "sequence.flow_rate_column",
            "Flux COLUMN (mL/min):", 0.1, 2.0, 0.75, 2, step=0.05)
        row = self._add_double_spin(g, row, "sequence.flow_rate_bp",
            "Flux BP (mL/min):", 0.1, 2.0, 0.75, 2, step=0.05)
        row = self._add_double_spin(g, row, "sequence.toc_cycle_min",
            "Cicle TOC (min):", 30.0, 120.0, 77.2, 1)
        row = self._add_int_spin(g, row, "sequence.toc_timeout_sec",
            "Timeout TOC (s):", 30, 300, 74)

        layout.addWidget(sec_seq)

        layout.addStretch()
        scroll.setWidget(container)
        return scroll

    # =========================================================================
    # TAB SISTEMA (IMMEDIAT)
    # =========================================================================

    def _create_system_tab(self):
        """Crea el tab Sistema — paths i opcions immediates."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(16)

        # --- Secció: Directoris ---
        sec_paths = ConfigSection("Directoris")
        g = sec_paths.content_layout

        # Directori dades
        g.addWidget(QLabel("Directori dades:"), 0, 0)
        self.data_dir_edit = QLineEdit()
        self.data_dir_edit.setPlaceholderText("Carpeta amb les SEQs")
        self._widgets["paths.data_folder"] = self.data_dir_edit
        g.addWidget(self.data_dir_edit, 0, 1)
        data_browse = QPushButton("...")
        data_browse.setMaximumWidth(30)
        data_browse.clicked.connect(lambda: self._browse_dir(self.data_dir_edit))
        g.addWidget(data_browse, 0, 2)

        # Directori registry
        g.addWidget(QLabel("Directori REGISTRY:"), 1, 0)
        self.registry_dir_edit = QLineEdit()
        self.registry_dir_edit.setPlaceholderText("Carpeta REGISTRY (KHP_History, etc.)")
        self._widgets["paths.registry_folder"] = self.registry_dir_edit
        g.addWidget(self.registry_dir_edit, 1, 1)
        registry_browse = QPushButton("...")
        registry_browse.setMaximumWidth(30)
        registry_browse.clicked.connect(lambda: self._browse_dir(self.registry_dir_edit))
        g.addWidget(registry_browse, 1, 2)

        # Fitxer manteniment
        g.addWidget(QLabel("Excel manteniment:"), 2, 0)
        self.maint_path_edit = QLineEdit()
        self.maint_path_edit.setPlaceholderText("Registre manteniment HPLC-DAD-TOC.xlsx")
        self._widgets["paths.maintenance_excel"] = self.maint_path_edit
        g.addWidget(self.maint_path_edit, 2, 1)
        maint_browse = QPushButton("...")
        maint_browse.setMaximumWidth(30)
        maint_browse.clicked.connect(lambda: self._browse_file(self.maint_path_edit, "Excel (*.xlsx *.xls)"))
        g.addWidget(maint_browse, 2, 2)

        layout.addWidget(sec_paths)

        # Info
        info = QLabel(
            "Nota: Canviar el directori de dades i clicar 'Guardar'.\n"
            "Després clicar 'Actualitzar' al Dashboard per veure les noves seqüències."
        )
        info.setStyleSheet("color: #666; font-style: italic;")
        info.setWordWrap(True)
        layout.addWidget(info)

        layout.addStretch()
        return widget

    # =========================================================================
    # WIDGET FACTORY HELPERS
    # =========================================================================

    def _add_double_spin(self, grid, row, key, label, min_val, max_val, default, decimals, step=None):
        """Afegeix un QDoubleSpinBox al grid i el registra."""
        grid.addWidget(QLabel(label), row, 0)
        spin = QDoubleSpinBox()
        spin.setRange(min_val, max_val)
        spin.setValue(default)
        spin.setDecimals(decimals)
        if step:
            spin.setSingleStep(step)
        grid.addWidget(spin, row, 1)
        self._widgets[key] = spin
        return row + 1

    def _add_int_spin(self, grid, row, key, label, min_val, max_val, default):
        """Afegeix un QSpinBox al grid i el registra."""
        grid.addWidget(QLabel(label), row, 0)
        spin = QSpinBox()
        spin.setRange(min_val, max_val)
        spin.setValue(default)
        grid.addWidget(spin, row, 1)
        self._widgets[key] = spin
        return row + 1

    def _add_combo(self, grid, row, key, label, options, default):
        """Afegeix un QComboBox al grid i el registra."""
        grid.addWidget(QLabel(label), row, 0)
        combo = QComboBox()
        combo.addItems(options)
        idx = combo.findText(default)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        grid.addWidget(combo, row, 1)
        self._widgets[key] = combo
        return row + 1

    def _add_checkbox(self, grid, row, key, label, default):
        """Afegeix un QCheckBox al grid i el registra."""
        cb = QCheckBox(label)
        cb.setChecked(default)
        grid.addWidget(cb, row, 0, 1, 2)
        self._widgets[key] = cb
        return row + 1

    # =========================================================================
    # BROWSE HELPERS
    # =========================================================================

    def _browse_dir(self, line_edit):
        """Obre diàleg per seleccionar directori."""
        current = line_edit.text().strip()
        start_dir = current if current else ""
        path = QFileDialog.getExistingDirectory(self, "Selecciona Directori", start_dir)
        if path:
            line_edit.setText(path)

    def _browse_file(self, line_edit, filter_str):
        """Obre diàleg per seleccionar fitxer."""
        current = line_edit.text().strip()
        import os
        start_dir = os.path.dirname(current) if current else ""
        path, _ = QFileDialog.getOpenFileName(self, "Selecciona Fitxer", start_dir, filter_str)
        if path:
            line_edit.setText(path)

    # =========================================================================
    # LOAD / APPLY CONFIG
    # =========================================================================

    def _load_config(self):
        """Carrega la configuració des del ConfigManager principal."""
        cfg = get_config()
        cfg.reload()
        self._apply_config_to_ui(cfg)
        self._original_values = self._capture_current_values()

    def _apply_config_to_ui(self, cfg):
        """Aplica la configuració als widgets."""
        # Spin boxes i combos (clau puntejada → cfg.get(*keys))
        for key, widget in self._widgets.items():
            keys = key.split(".")

            if isinstance(widget, QDoubleSpinBox):
                val = cfg.get(*keys)
                if val is not None:
                    widget.setValue(float(val))
            elif isinstance(widget, QSpinBox):
                val = cfg.get(*keys)
                if val is not None:
                    widget.setValue(int(val))
            elif isinstance(widget, QComboBox):
                val = cfg.get(*keys)
                if val is not None:
                    idx = widget.findText(str(val))
                    if idx >= 0:
                        widget.setCurrentIndex(idx)
            elif isinstance(widget, QCheckBox):
                val = cfg.get(*keys)
                if val is not None:
                    widget.setChecked(bool(val))
            elif isinstance(widget, QLineEdit):
                val = cfg.get(*keys)
                widget.setText(str(val) if val else "")

        # Widgets especials
        fractions = cfg.get("time_fractions", default={})
        self.fractions_editor.load(fractions)

        timeout_zones = cfg.get("timeout_zones", default={})
        self.timeout_zones_editor.load(timeout_zones)

        blank_patterns = cfg.get("blank_injections", "patterns", default=[])
        self.blank_patterns.load(blank_patterns)

        control_patterns = cfg.get("control_injections", "patterns", default=[])
        self.control_patterns.load(control_patterns)

        wavelengths = cfg.get("wavelengths", default={})
        self.wl_selector.load(wavelengths)

    # =========================================================================
    # DIRTY TRACKING
    # =========================================================================

    def _capture_current_values(self):
        """Captura tots els valors actuals per comparar després."""
        values = {}
        for key, widget in self._widgets.items():
            if isinstance(widget, QDoubleSpinBox):
                values[key] = widget.value()
            elif isinstance(widget, QSpinBox):
                values[key] = widget.value()
            elif isinstance(widget, QComboBox):
                values[key] = widget.currentText()
            elif isinstance(widget, QCheckBox):
                values[key] = widget.isChecked()
            elif isinstance(widget, QLineEdit):
                values[key] = widget.text()
            elif isinstance(widget, TimeFractionsEditor):
                values[key] = str(widget.save())
            elif isinstance(widget, TimeoutZonesEditor):
                values[key] = str(widget.save())
            elif isinstance(widget, WavelengthSelector):
                values[key] = str(widget.save())
            elif isinstance(widget, PatternListEditor):
                values[key] = str(widget.save())
        return values

    def _get_changed_sections(self):
        """Retorna noms de seccions amb valors canviats."""
        current = self._capture_current_values()
        changed = set()
        for key, old_val in self._original_values.items():
            if current.get(key) != old_val:
                changed.add(key.split(".")[0])
        return changed

    def _classify_changes(self):
        """Classifica les seccions canviades per impacte."""
        changed = self._get_changed_sections()
        retroactive = changed & REPROCESS_SECTIONS
        future = changed & FUTURE_SECTIONS
        immediate = changed & IMMEDIATE_SECTIONS
        return retroactive, future, immediate

    # =========================================================================
    # SAVE CONFIG
    # =========================================================================

    def _save_config(self):
        """Guarda la configuració amb diàleg d'impacte si cal."""
        # Validar fraccions
        frac_errors = self.fractions_editor.validate()
        if frac_errors:
            QMessageBox.warning(
                self, "Error de Validació",
                "Errors a les fraccions temporals:\n" + "\n".join(frac_errors)
            )
            return

        retroactive, future, immediate = self._classify_changes()

        if not retroactive and not future and not immediate:
            QMessageBox.information(self, "Sense canvis", "No hi ha canvis per guardar.")
            return

        # Diàleg segons impacte
        if retroactive:
            sections_text = ", ".join(sorted(retroactive))
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Canvis retroactius")
            msg.setText(
                f"Has canviat seccions amb impacte retroactiu:\n{sections_text}\n\n"
                "Les seqüències ja analitzades poden donar resultats diferents."
            )
            save_btn = msg.addButton("Guardar", QMessageBox.AcceptRole)
            save_mark_btn = msg.addButton("Guardar i marcar obsoletes", QMessageBox.AcceptRole)
            msg.addButton("Cancel·lar", QMessageBox.RejectRole)
            msg.exec()

            clicked = msg.clickedButton()
            if clicked == save_btn:
                self._do_save()
            elif clicked == save_mark_btn:
                self._do_save(mark_stale=True)
            # Cancel·lar: no fem res

        elif future:
            reply = QMessageBox.information(
                self, "Canvis futurs",
                "Aquests canvis només afectaran el proper processament.\n"
                "Les seqüències ja analitzades no es veuen afectades.",
                QMessageBox.Ok | QMessageBox.Cancel
            )
            if reply == QMessageBox.Ok:
                self._do_save()

        else:
            # Només immediats
            self._do_save()

    def _do_save(self, mark_stale=False):
        """Escriu els valors al ConfigManager i guarda."""
        cfg = get_config()

        # Escriure tots els widgets simples
        for key, widget in self._widgets.items():
            keys = key.split(".")

            if isinstance(widget, QDoubleSpinBox):
                cfg.set(*keys, widget.value())
            elif isinstance(widget, QSpinBox):
                cfg.set(*keys, widget.value())
            elif isinstance(widget, QComboBox):
                cfg.set(*keys, widget.currentText())
            elif isinstance(widget, QCheckBox):
                cfg.set(*keys, widget.isChecked())
            elif isinstance(widget, QLineEdit):
                text = widget.text().strip()
                if text:
                    cfg.set(*keys, text)

        # Widgets especials
        cfg.set_section("time_fractions", self.fractions_editor.save())
        cfg.set_section("timeout_zones", self.timeout_zones_editor.save())
        cfg.set_section("wavelengths", self.wl_selector.save())
        cfg.set("blank_injections", "patterns", self.blank_patterns.save())
        cfg.set("control_injections", "patterns", self.control_patterns.save())

        if cfg.save():
            self._original_values = self._capture_current_values()
            self.main_window.set_status("Configuració guardada a hpsec_config.json", 3000)

            if mark_stale:
                QMessageBox.information(
                    self, "Configuració Guardada",
                    "Configuració guardada. Les seqüències analitzades\n"
                    "es mostraran com a obsoletes al Dashboard."
                )
            else:
                QMessageBox.information(
                    self, "Configuració Guardada",
                    f"Configuració guardada a:\n{cfg.config_path}"
                )
            self.config_changed.emit()
        else:
            QMessageBox.warning(self, "Error", "Error guardant configuració")

    # =========================================================================
    # RESET DEFAULTS
    # =========================================================================

    def _reset_defaults(self):
        """Restaura els valors per defecte."""
        reply = QMessageBox.question(
            self, "Confirmar",
            "Vols restaurar tots els valors per defecte?\n"
            "Nota: Les fraccions temporals del fitxer JSON es mantindran.",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            cfg = get_config()
            cfg.reset_to_defaults()
            cfg.reload()
            self._apply_config_to_ui(cfg)
            self._original_values = self._capture_current_values()
            self.main_window.set_status("Valors per defecte restaurats", 3000)

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def get_config(self):
        """Retorna la configuració actual (per usar des d'altres mòduls)."""
        return get_config()
