"""
HPSEC Suite - Import Panel v3.0
================================

Panel per a la fase 1: Importació de seqüències.
- Columnes separades per punts i fitxers
- Colors segons tipus de match (EXACT/FUZZY/MANUAL/NONE)
- Dropdown per assignació manual d'orfes
- Verificació obligatòria per FUZZY
"""

import os
import json
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QFrame, QTableWidget, QTableWidgetItem,
    QHeaderView, QMessageBox, QRadioButton, QButtonGroup, QComboBox
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QColor, QBrush

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from hpsec_import import (
    import_sequence, load_manifest, import_from_manifest,
    generate_import_manifest, save_import_manifest,
    llegir_doc_uib, llegir_dad_export3d, llegir_dad_1a,
    get_baseline_value
)
import numpy as np

# Importar components del paquet
from .delegates import ComboBoxDelegate, FileAssignmentDelegate
from .worker import ImportWorker
from .dialogs import OrphanFilesDialog, ChromatogramPreviewDialog

# Importar estils compartits
from gui.widgets.styles import (
    PANEL_MARGINS, PANEL_SPACING,
    COLOR_SUCCESS, COLOR_WARNING, COLOR_ERROR,
    apply_panel_layout
)

CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "hpsec_config.json"

# Colors per tipus de match
MATCH_COLORS = {
    "EXACT": QColor("#D5F5E3"),    # Verd clar
    "FUZZY": QColor("#FCF3CF"),    # Groc/taronja clar
    "MANUAL": QColor("#D6EAF8"),   # Blau clar
    "NONE": QColor("#FADBD8"),     # Vermell/rosa clar
    "NORMAL": QColor("#FFFFFF"),   # Blanc
}


def load_sample_types_config():
    """Carrega configuració de tipus de mostra."""
    default_types = {
        "MOSTRA": {"label": "MOSTRA", "color": "#2E86AB", "patterns": []},
        "PATRÓ_CAL": {"label": "PATRÓ_CAL", "color": "#2A9D8F", "patterns": ["KHP"]},
        "PATRÓ_REF": {"label": "PATRÓ_REF", "color": "#9B59B6", "patterns": ["REF", "QC"]},
        "CONTROL": {"label": "CONTROL", "color": "#F6AE2D", "patterns": ["NaOH", "CONTROL"]},
        "BLANC": {"label": "BLANC", "color": "#888888", "patterns": ["MQ", "BLANK", "BLK"]},
    }
    try:
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                config = json.load(f)
                return config.get("sample_types", default_types)
    except:
        pass
    return default_types


def detect_sample_type(sample_name, original_type, config):
    """Detecta el tipus de mostra basant-se en patrons del config."""
    name_upper = sample_name.upper()

    for type_key, type_info in config.items():
        patterns = type_info.get("patterns", [])
        for pattern in patterns:
            if pattern.upper() in name_upper:
                return type_key

    translations = {
        "SAMPLE": "MOSTRA",
        "KHP": "PATRÓ_CAL",
        "CONTROL": "CONTROL",
        "BLANK": "BLANC",
    }
    return translations.get(original_type.upper(), "MOSTRA")


class ImportPanel(QWidget):
    """Panel d'importació de seqüències."""

    import_completed = Signal(dict)
    warnings_dismissed = Signal()  # Senyal quan s'han descartat els avisos

    # Columnes base (s'ajusten segons mode a _setup_table_columns)
    COL_INJ = 0
    COL_MOSTRA = 1
    COL_TIPUS = 2
    COL_REP = 3
    COL_INJ_VOL = 4  # Volum d'injecció (µL)
    COL_DIRECT_PTS = 5
    COL_DIRECT_FILE = 6
    # Columnes dinàmiques (s'ajusten segons mode de dades)
    # Per DUAL: UIB=7,8, Estat_UIB=9, DAD=10,11, Estat_DAD=12
    # Per DIRECT: DAD=7,8, Estat_DAD=9
    COL_UIB_PTS_ACTUAL = 7
    COL_UIB_FILE_ACTUAL = 8
    COL_ESTAT_UIB = 9  # I08: Estat separat per UIB
    COL_DAD_PTS_ACTUAL = 10
    COL_DAD_FILE_ACTUAL = 11
    COL_ESTAT_DAD = 12  # I08: Estat separat per DAD
    COL_ESTAT = 12  # Compatibilitat (apunta a DAD per defecte)

    # Tipus de mostra que requereixen assignació obligatòria de fitxers
    TYPES_REQUIRE_ASSIGNMENT = {"MOSTRA", "PATRÓ_CAL", "PATRÓ_REF"}
    # Tipus de mostra que permeten assignació opcional
    TYPES_OPTIONAL_ASSIGNMENT = {"CONTROL", "BLANC"}

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.seq_path = None
        self.existing_manifest = None
        self.imported_data = None
        self.worker = None
        self.sample_types_config = load_sample_types_config()
        self._sample_data = []
        self._orphan_files = {"uib": [], "dad": []}
        self._match_types = {}  # (row, col) -> match_type
        self._unverified_fuzzy = set()  # Set of rows needing verification
        self._manual_assignments = {}  # (sample_name, replica) -> {col: filename}
        self._data_mode = "DUAL"  # DUAL, DIRECT, UIB
        self._import_warnings = []  # Warnings d'importació
        self._loaded_from_manifest = False  # Si s'ha carregat des de manifest existent
        self._orphan_warning_dismissed = False  # Si l'usuari ha marcat l'avís d'orfes com revisat
        self._warnings_confirmed = False  # Si l'usuari ha confirmat els warnings (FUZZY, etc.)
        self._warnings_confirmed_by = None  # G05: Qui ha confirmat (traçabilitat)

        self._setup_ui()

    def reset(self):
        """Reinicia el panel al seu estat inicial."""
        self.seq_path = None
        self.existing_manifest = None
        self.imported_data = None
        self.worker = None
        self._sample_data = []
        self._orphan_files = {"uib": [], "dad": []}
        self._match_types = {}
        self._unverified_fuzzy = set()
        self._manual_assignments = {}
        self._data_mode = "DUAL"
        self._import_warnings = []
        self._loaded_from_manifest = False
        self._orphan_warning_dismissed = False
        self._warnings_confirmed = False
        self._warnings_confirmed_by = None

        # Reset UI elements
        self.path_input.clear()
        self.info_frame.setVisible(False)
        self.table_help.setVisible(False)
        self.samples_table.setRowCount(0)
        self.samples_table.setVisible(False)

        # Reset warnings frame (orfes)
        if hasattr(self, 'warnings_frame'):
            self.warnings_frame.setVisible(False)
        if hasattr(self, 'warnings_label'):
            self.warnings_label.setText("")
        if hasattr(self, 'orphans_btn'):
            self.orphans_btn.setVisible(False)
        if hasattr(self, 'confirm_btn'):
            self.confirm_btn.setVisible(False)
        if hasattr(self, 'refresh_btn'):
            self.refresh_btn.setVisible(False)
        if hasattr(self, 'dismiss_btn'):
            self.dismiss_btn.setVisible(False)

        # Mostrar placeholder
        if hasattr(self, 'placeholder'):
            self.placeholder.setVisible(True)

    def _setup_ui(self):
        """Configura la interfície del panel."""
        layout = QVBoxLayout(self)
        apply_panel_layout(layout)

        # Camp ocult per compatibilitat (usat internament)
        self.path_input = QLineEdit()
        self.path_input.setVisible(False)
        layout.addWidget(self.path_input)

        # Botons ocults per compatibilitat (accions ara al header del wizard)
        self.import_btn = QPushButton()
        self.import_btn.setVisible(False)
        self.save_btn = QPushButton()
        self.save_btn.setVisible(False)
        self.next_btn = QPushButton()
        self.next_btn.setVisible(False)

        # === MANIFEST INFO (eliminat - redundant amb header del wizard) ===
        # El nom SEQ ja apareix al header i el botó "Importar" gestiona reimportació
        self.manifest_frame = QFrame()
        self.manifest_frame.setVisible(False)  # Mai visible - mantingut per compatibilitat
        self.manifest_info = QLabel()
        self.use_manifest_radio = QRadioButton()
        self.full_import_radio = QRadioButton()
        self.import_mode_group = QButtonGroup(self)

        # Nota: Avisos es gestionen des del wizard header

        # === INFO BARRA (resum injeccions) ===
        self.info_frame = QFrame()
        self.info_frame.setVisible(False)
        self.info_frame.setFixedHeight(28)
        info_layout = QHBoxLayout(self.info_frame)
        info_layout.setContentsMargins(0, 2, 0, 2)
        info_layout.setSpacing(16)

        self.total_label = QLabel()
        self.total_label.setStyleSheet("font-weight: bold; color: #2E86AB;")
        info_layout.addWidget(self.total_label)

        info_layout.addStretch()

        # Comptador de fitxers UIB i DAD
        self.files_label = QLabel()
        self.files_label.setStyleSheet("color: #666;")
        info_layout.addWidget(self.files_label)

        layout.addWidget(self.info_frame)

        # === TAULA DE MOSTRES ===
        # Nota d'ajuda
        self.table_help = QLabel("💡 Doble-clic a una fila per veure la gràfica")
        self.table_help.setStyleSheet("color: #666; font-size: 11px; padding: 2px;")
        self.table_help.setVisible(False)
        layout.addWidget(self.table_help)

        self.samples_table = QTableWidget()
        self.samples_table.setToolTip("Doble-clic per veure gràfica DOC + DAD 254nm")

        # Amagar numeració automàtica de files
        self.samples_table.verticalHeader().setVisible(False)

        # NO usar colors alternats (interfereixen amb colors de match)
        self.samples_table.setAlternatingRowColors(False)

        # Permetre ordenar
        self.samples_table.setSortingEnabled(True)

        self.samples_table.cellDoubleClicked.connect(self._on_cell_double_clicked)
        self.samples_table.cellChanged.connect(self._on_cell_changed)
        self.samples_table.setEditTriggers(
            QTableWidget.DoubleClicked | QTableWidget.EditKeyPressed
        )
        self.samples_table.setVisible(False)

        layout.addWidget(self.samples_table, 1)

        # Referència dummy per compatibilitat amb wizard (el wizard l'amaga)
        self.next_btn = QPushButton()
        self.next_btn.setVisible(False)

        # Botons legacy (mantinguts com a dummies per compatibilitat)
        self.warnings_frame = QFrame()
        self.warnings_frame.setVisible(False)
        self.confirm_btn = QPushButton()
        self.confirm_btn.setVisible(False)
        self.orphans_btn = QPushButton()
        self.orphans_btn.setVisible(False)
        self.refresh_btn = QPushButton()
        self.refresh_btn.setVisible(False)
        self.dismiss_btn = QPushButton()
        self.dismiss_btn.setVisible(False)

        # === PLACEHOLDER ===
        self.placeholder = QLabel("Preparant importació...")
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.placeholder.setStyleSheet("color: #888; font-size: 14px;")
        layout.addWidget(self.placeholder, 1)

    def _get_file_options(self, row, col):
        """Retorna opcions pel dropdown de fitxers."""
        options = ["(cap)"]

        # Determinar si és UIB o DAD segons la columna
        if self._data_mode == "DIRECT":
            if col == self.COL_DAD_FILE_ACTUAL:
                orphans = self._orphan_files.get("dad", [])
            else:
                return options
        else:
            if col == self.COL_UIB_FILE_ACTUAL:
                orphans = self._orphan_files.get("uib", [])
            elif col == self.COL_DAD_FILE_ACTUAL:
                orphans = self._orphan_files.get("dad", [])
            else:
                return options

        # Afegir valor actual si existeix
        current = self.samples_table.item(row, col)
        if current and current.text() and current.text() != "-" and current.text() != "(cap)":
            options.append(current.text())

        # Afegir orfes
        for f in orphans:
            fname = Path(f).name if "/" in f or "\\" in f else f
            if fname not in options:
                options.append(fname)

        return options

    def set_sequence_path(self, path):
        self.seq_path = path
        self.main_window.seq_path = path  # Actualitzar també el main_window
        self.path_input.setText(path)
        self._check_manifest()

    def load_from_dashboard(self, seq_path):
        """Carrega una seqüència des del Dashboard - auto-carrega si hi ha manifest."""
        self.set_sequence_path(seq_path)

        # Si hi ha manifest existent, carregar automàticament
        if self.existing_manifest:
            # Amagar placeholder immediatament - es mostrarà la taula quan acabi
            self.placeholder.setVisible(False)
            self._auto_load_from_manifest()

    def _go_to_dashboard(self):
        """Torna al Dashboard per seleccionar una seqüència."""
        self.main_window.tab_widget.setCurrentIndex(0)

    def _check_manifest(self):
        self.existing_manifest = load_manifest(self.seq_path)

        if self.existing_manifest:
            self._show_manifest_info()
        else:
            self.manifest_frame.setVisible(False)

        self.import_btn.setEnabled(True)
        self.samples_table.setVisible(False)
        self.table_help.setVisible(False)
        self.info_frame.setVisible(False)
        self.warnings_frame.setVisible(False)
        self.placeholder.setVisible(True)
        self.next_btn.setEnabled(False)
        self.save_btn.setEnabled(False)

    def _show_manifest_info(self):
        """Registra info del manifest (frame eliminat - info ja al header wizard)."""
        # Mantingut per compatibilitat però ja no mostra res
        # La info del SEQ es mostra al header del ProcessWizardPanel
        pass

    def _auto_load_from_manifest(self):
        """Carrega automàticament des del manifest existent."""
        self._loaded_from_manifest = True
        self.main_window.show_progress(0)

        self.worker = ImportWorker(
            self.seq_path,
            use_manifest=True,
            manifest=self.existing_manifest
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_import_finished)
        self.worker.error.connect(self._on_import_error)
        self.worker.start()

    def _run_import(self, force_reimport=False):
        """Executa importació. Si force_reimport=True, reimporta tot."""
        if not self.seq_path:
            return

        self.main_window.show_progress(0)

        # Si ja hi ha manifest i no forcem reimportació, usar-lo
        use_manifest = self.existing_manifest and not force_reimport
        self._loaded_from_manifest = use_manifest

        self.worker = ImportWorker(
            self.seq_path,
            use_manifest=use_manifest,
            manifest=self.existing_manifest if use_manifest else None
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_import_finished)
        self.worker.error.connect(self._on_import_error)
        self.worker.start()

    def _on_progress(self, pct, msg):
        self.main_window.show_progress(pct)
        self.main_window.set_status(msg)

    def _on_import_finished(self, result):
        self.main_window.show_progress(-1)
        self.import_btn.setEnabled(True)

        if not result.get("success"):
            errors = result.get("errors", ["Error desconegut"])
            QMessageBox.critical(self, "Error d'Importació", f"Error: {errors[0]}")
            self.import_completed.emit({'success': False, 'errors': errors})
            return

        # Verificar si cal preguntar la sensibilitat UIB
        self._check_uib_sensitivity(result)

        self.imported_data = result
        self.main_window.imported_data = result
        # Restaurar estat "revisat" d'orfes i warnings si es van marcar anteriorment
        self._orphan_warning_dismissed = result.get("orphan_warning_dismissed", False)
        self._warnings_confirmed = result.get("warnings_confirmed", False)
        self._warnings_confirmed_by = result.get("warnings_confirmed_by", None)  # G05: traçabilitat

        self._show_results(result)

        # Nota: Els avisos es gestionen des del wizard header

        try:
            save_import_manifest(result)
            self.main_window.mark_manifest_saved()
        except Exception as e:
            print(f"Warning: No s'ha pogut guardar manifest: {e}")

        self._update_next_button_state()
        self.save_btn.setEnabled(True)
        self.main_window.enable_tab(1)
        self.main_window.set_status("Importació completada", 5000)

        # Emetre senyal per al wizard
        self.import_completed.emit({
            'success': True,
            'warnings': result.get('warnings', []),
            'orphan_files': result.get('orphan_files', {}),
            'warnings_confirmed': self._warnings_confirmed,
            'orphan_warning_dismissed': self._orphan_warning_dismissed,
        })

    # Nota: _show_warnings_bar eliminada - avisos es gestionen des del wizard header

    def _check_uib_sensitivity(self, result):
        """Verifica si cal preguntar la sensibilitat UIB i actualitza el MasterFile."""
        # Només per mode DUAL o UIB
        data_mode = result.get("data_mode", "")
        if data_mode not in ["DUAL", "UIB"]:
            return

        # Verificar si la sensibilitat UIB està definida
        uib_sens = result.get("uib_sensitivity")
        if uib_sens is not None and uib_sens not in ["None", "", None]:
            return

        # Preguntar a l'usuari (camp lliure)
        from PySide6.QtWidgets import QInputDialog
        text, ok = QInputDialog.getText(
            self,
            "Sensibilitat UIB",
            "Indica la sensibilitat UIB (ex: 700, 1000, o deixa buit si no aplica):",
            text=""
        )

        if ok and text.strip():
            sens_value = text.strip()

            # Actualitzar el resultat
            result["uib_sensitivity"] = sens_value

            # Actualitzar el MasterFile si existeix
            master_file = result.get("master_file")
            if master_file and os.path.exists(master_file):
                try:
                    import openpyxl
                    wb = openpyxl.load_workbook(master_file)
                    if "0-INFO" in wb.sheetnames:
                        ws = wb["0-INFO"]
                        ws["B5"] = sens_value
                        wb.save(master_file)
                        print(f"Actualitzat UIB sensitivity a MasterFile: {sens_value}")
                except Exception as e:
                    print(f"Warning: No s'ha pogut actualitzar MasterFile: {e}")

    def _on_import_error(self, error_msg):
        self.main_window.show_progress(-1)
        self.import_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", f"Error durant la importació:\n{error_msg}")

    # =========================================================================
    # MÈTODES AUXILIARS PER _show_results (descomposició)
    # =========================================================================

    def _init_results_state(self):
        """Inicialitza l'estat per mostrar resultats."""
        self.placeholder.setVisible(False)
        self._match_types = {}
        self._unverified_fuzzy = set()
        self._manual_assignments = {}
        if not self._loaded_from_manifest:
            self._orphan_warning_dismissed = False
            self._warnings_confirmed = False

    def _process_orphan_files(self, manifest, samples, result):
        """Processa i filtra fitxers orfes."""
        # Orfes del manifest (noms) per mostrar - només els no suggerits
        self._orphan_files = {
            "uib": manifest.get("orphan_files", {}).get("uib", []),
            "dad": manifest.get("orphan_files", {}).get("dad", []),
        }
        # TOTS els orfes amb paths complets (per comptar punts, inclou suggerits)
        self._orphan_files_full = {
            "uib": result.get("all_orphan_files", result.get("orphan_files", {})).get("uib", []),
            "dad": result.get("all_orphan_files", result.get("orphan_files", {})).get("dad", []),
        }

        # Si carregat des de manifest, filtrar orfes que ja estan assignats a mostres
        if self._loaded_from_manifest:
            assigned_uib = set()
            assigned_dad = set()
            for sample in samples:
                for rep in sample.get("replicas", []):
                    uib_info = rep.get("uib", {})
                    dad_info = rep.get("dad", {})
                    if uib_info and uib_info.get("file"):
                        assigned_uib.add(Path(uib_info["file"]).name)
                    if dad_info and dad_info.get("file") and not dad_info.get("file", "").startswith("["):
                        assigned_dad.add(Path(dad_info["file"]).name)
            # Treure fitxers assignats de la llista d'orfes
            self._orphan_files["uib"] = [f for f in self._orphan_files["uib"] if Path(f).name not in assigned_uib]
            self._orphan_files["dad"] = [f for f in self._orphan_files["dad"] if Path(f).name not in assigned_dad]

    def _build_injection_list(self, samples):
        """Construeix llista plana d'injeccions ordenada per line_num."""
        all_injections = []
        for sample in samples:
            original_type = sample.get("type", "SAMPLE")
            sample_type = detect_sample_type(
                sample["name"], original_type, self.sample_types_config
            )
            original_name = sample.get("original_name", sample["name"])

            for rep in sample.get("replicas", []):
                inj_info = rep.get("injection_info") or rep.get("injection") or {}
                line_num = inj_info.get("line_num")
                if line_num is None:
                    d = rep.get("direct", {})
                    line_num = d.get("row_start") if d else 999999
                if line_num is None:
                    line_num = 999999

                all_injections.append({
                    "sample_name": sample["name"],
                    "original_name": original_name,
                    "sample_type": sample_type,
                    "rep": rep,
                    "line_num": line_num,
                })

        all_injections.sort(key=lambda x: (x["line_num"], x["sample_name"]))
        return all_injections

    def _update_info_bar(self, result, all_injections):
        """Actualitza la barra d'informació amb resum de la seqüència."""
        total_injections = len(all_injections)
        stats = result.get("stats", {})
        master_line_count = stats.get("master_line_count", result.get("master_line_count", total_injections))

        # Recollir volums d'injecció
        volumes = []
        for inj in all_injections:
            rep = inj.get("rep", {})
            inj_info = rep.get("injection_info") or rep.get("injection") or {}
            vol = inj_info.get("inj_volume")
            if vol is not None:
                volumes.append(vol)

        # Construir resum
        method = result.get("method", "COLUMN")
        info_parts = []

        # Injeccions (amb warning si no coincideixen)
        if master_line_count > total_injections:
            info_parts.append(f"⚠️ {total_injections}/{master_line_count} inj")
            has_warning = True
        else:
            info_parts.append(f"{total_injections} inj")
            has_warning = False

        info_parts.append(method)
        info_parts.append(self._data_mode)

        # Volum d'injecció
        if volumes:
            vol_min, vol_max = min(volumes), max(volumes)
            if vol_min == vol_max:
                info_parts.append(f"{int(vol_min)}µL")
            else:
                info_parts.append(f"{int(vol_min)}-{int(vol_max)}µL")

        # Sensibilitat UIB
        if self._data_mode in ["DUAL", "UIB"]:
            uib_sens = result.get("uib_sensitivity")
            if uib_sens is not None:
                try:
                    sens_val = int(float(uib_sens))
                    info_parts.append(f"UIB:{sens_val}ppb")
                except (ValueError, TypeError):
                    info_parts.append(f"UIB:{uib_sens}")

        self.total_label.setText(" · ".join(info_parts))
        self.total_label.setStyleSheet(
            "font-weight: bold; color: #E74C3C;" if has_warning else "font-weight: bold; color: #2E86AB;"
        )

        # Comptar fitxers
        uib_used = stats.get("uib_files_used", 0)
        uib_orphan = stats.get("orphan_uib", 0)
        dad_used = stats.get("dad_files_used", 0)
        dad_orphan = stats.get("orphan_dad", 0)

        files_parts = []
        if self._data_mode in ["DUAL", "UIB"]:
            files_parts.append(f"UIB: {uib_used + uib_orphan}")
        files_parts.append(f"DAD: {dad_used + dad_orphan}")
        self.files_label.setText(" · ".join(files_parts))
        self.info_frame.setVisible(True)

    def _populate_row_basic(self, row, injection_num, inj):
        """Omple les columnes bàsiques d'una fila (Inj, Mostra, Tipus, Rep, Vol, Direct)."""
        sample_name = inj["sample_name"]
        original_name = inj.get("original_name", sample_name)
        sample_type = inj["sample_type"]
        rep = inj["rep"]

        # Inj
        inj_item = QTableWidgetItem()
        inj_item.setData(Qt.DisplayRole, injection_num)
        inj_item.setTextAlignment(Qt.AlignCenter)
        inj_item.setFlags(inj_item.flags() & ~Qt.ItemIsEditable)
        self.samples_table.setItem(row, self.COL_INJ, inj_item)

        # Mostra
        if original_name != sample_name:
            name_item = QTableWidgetItem(original_name)
            name_item.setToolTip(f"Nom únic: {sample_name}\nNom MasterFile: {original_name}")
            name_item.setForeground(QBrush(QColor("#2E86AB")))
        else:
            name_item = QTableWidgetItem(sample_name)
        name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
        name_item.setData(Qt.UserRole, sample_name)
        self.samples_table.setItem(row, self.COL_MOSTRA, name_item)

        # Tipus
        type_item = QTableWidgetItem(sample_type)
        type_item.setTextAlignment(Qt.AlignCenter)
        self.samples_table.setItem(row, self.COL_TIPUS, type_item)

        # Rep
        rep_item = QTableWidgetItem(str(rep.get("replica", "?")))
        rep_item.setTextAlignment(Qt.AlignCenter)
        rep_item.setFlags(rep_item.flags() & ~Qt.ItemIsEditable)
        self.samples_table.setItem(row, self.COL_REP, rep_item)

        # Volum
        inj_info = rep.get("injection_info") or rep.get("injection") or {}
        inj_vol = inj_info.get("inj_volume")
        vol_text = f"{int(inj_vol)}" if inj_vol else "-"
        vol_item = QTableWidgetItem(vol_text)
        vol_item.setTextAlignment(Qt.AlignCenter)
        vol_item.setFlags(vol_item.flags() & ~Qt.ItemIsEditable)
        self.samples_table.setItem(row, self.COL_INJ_VOL, vol_item)

        # Direct
        d = rep.get("direct", {})
        direct_pts = d.get("n_points", 0) if d else 0
        row_start = d.get("row_start", "") if d else ""
        row_end = d.get("row_end", "") if d else ""
        direct_file = f"{row_start}-{row_end}" if row_start and row_end else "-"

        self._add_simple_cell(row, self.COL_DIRECT_PTS, str(direct_pts) if direct_pts else "-")
        self._add_simple_cell(row, self.COL_DIRECT_FILE, direct_file)

        return direct_pts

    def _populate_row_uib(self, row, rep, sample_name, sample_type, requires_assignment, optional_can_assign):
        """Omple les columnes UIB d'una fila. Retorna (review_signals, missing_signals, needs_review)."""
        review_signals = []
        missing_signals = []
        needs_review = False

        u = rep.get("uib", {})
        uib_pts = u.get("n_points", 0) if u else 0
        uib_file = u.get("file", "") if u else ""
        uib_suggestion = rep.get("uib_suggestion")

        if uib_file:
            uib_file = Path(uib_file).name

        self._add_simple_cell(row, self.COL_UIB_PTS_ACTUAL, str(uib_pts) if uib_pts else "-")

        if uib_suggestion and (requires_assignment or optional_can_assign):
            suggested_file = uib_suggestion.get("file", "")
            confidence = uib_suggestion.get("confidence", 0)
            suggestion_status = uib_suggestion.get("status", "SUGGESTED")
            replica_num = rep.get("replica", 1)
            display_name = f"{sample_name}_R{replica_num}"

            # Mostrar cel·la de suggeriment si no està confirmat, independentment de la font
            if suggestion_status == "CONFIRMED" or self._warnings_confirmed:
                self._add_simple_cell(row, self.COL_UIB_FILE_ACTUAL, display_name)
            else:
                self._add_suggestion_cell(row, self.COL_UIB_FILE_ACTUAL, suggested_file, confidence, display_name)
                if requires_assignment:
                    review_signals.append(f"UIB {int(confidence)}%")
                needs_review = True

            n_points = self._count_file_points(suggested_file, "uib")
            if n_points > 0:
                self.samples_table.item(row, self.COL_UIB_PTS_ACTUAL).setText(str(n_points))

        elif uib_file:
            # Fitxer ja assignat (des de manifest o durant importació)
            replica_num = rep.get("replica", 1)
            display_name = f"{sample_name}_R{replica_num}"
            self._add_simple_cell(row, self.COL_UIB_FILE_ACTUAL, display_name)
            if not uib_pts:
                n_points = self._count_file_points(uib_file, "uib")
                if n_points > 0:
                    self.samples_table.item(row, self.COL_UIB_PTS_ACTUAL).setText(str(n_points))

        elif not uib_pts and self._orphan_files.get("uib") and requires_assignment:
            self._add_file_cell(row, self.COL_UIB_FILE_ACTUAL, "-", editable=True)
            if not self._warnings_confirmed:
                missing_signals.append("UIB")
                needs_review = True

        elif optional_can_assign and self._orphan_files.get("uib"):
            display_val = uib_file if uib_file else "-"
            self._add_file_cell(row, self.COL_UIB_FILE_ACTUAL, display_val, editable=True)

        else:
            self._add_simple_cell(row, self.COL_UIB_FILE_ACTUAL, uib_file if uib_file else "-")

        return review_signals, missing_signals, needs_review, uib_pts

    def _populate_row_dad(self, row, rep, sample_name, sample_type, requires_assignment, optional_can_assign):
        """Omple les columnes DAD d'una fila. Retorna (review_signals, missing_signals, needs_review)."""
        review_signals = []
        missing_signals = []
        needs_review = False

        dad = rep.get("dad", {})
        dad_pts = dad.get("n_points", 0) if dad else 0
        dad_suggestion = rep.get("dad_suggestion")

        # Obtenir el fitxer DAD
        dad_file = ""
        if dad:
            dad_file = dad.get("file", "")
            if not dad_file and dad_pts > 0:
                source = dad.get("source", "")
                if source == "masterfile":
                    dad_file = "[MasterFile]"
                elif source in ["export3d", "csv"]:
                    dad_file = f"[{source}]"
            elif dad_file:
                dad_file = Path(dad_file).name if "/" in dad_file or "\\" in dad_file else dad_file

        self._add_simple_cell(row, self.COL_DAD_PTS_ACTUAL, str(dad_pts) if dad_pts else "-")

        if dad_suggestion and (requires_assignment or optional_can_assign):
            suggested_file = dad_suggestion.get("file", "")
            confidence = dad_suggestion.get("confidence", 0)
            suggestion_status = dad_suggestion.get("status", "SUGGESTED")
            replica_num = rep.get("replica", 1)
            display_name = f"{sample_name}_R{replica_num}"

            # Mostrar cel·la de suggeriment si no està confirmat, independentment de la font
            if suggestion_status == "CONFIRMED" or self._warnings_confirmed:
                self._add_simple_cell(row, self.COL_DAD_FILE_ACTUAL, display_name)
            else:
                self._add_suggestion_cell(row, self.COL_DAD_FILE_ACTUAL, suggested_file, confidence, display_name)
                if requires_assignment:
                    review_signals.append(f"DAD {int(confidence)}%")
                needs_review = True

            n_points = self._count_file_points(suggested_file, "dad")
            if n_points > 0:
                self.samples_table.item(row, self.COL_DAD_PTS_ACTUAL).setText(str(n_points))
                dad_pts = n_points

        elif dad_file and not dad_file.startswith("["):
            replica_num = rep.get("replica", 1)
            display_name = f"{sample_name}_R{replica_num}"
            self._add_simple_cell(row, self.COL_DAD_FILE_ACTUAL, display_name)
            if not dad_pts:
                n_points = self._count_file_points(dad_file, "dad")
                if n_points > 0:
                    self.samples_table.item(row, self.COL_DAD_PTS_ACTUAL).setText(str(n_points))

        elif not dad_pts and not dad_file and self._orphan_files.get("dad") and requires_assignment:
            self._add_file_cell(row, self.COL_DAD_FILE_ACTUAL, "-", editable=True)
            if not self._warnings_confirmed:
                missing_signals.append("DAD")
                needs_review = True

        elif optional_can_assign and self._orphan_files.get("dad"):
            display_val = dad_file if dad_file else "-"
            self._add_file_cell(row, self.COL_DAD_FILE_ACTUAL, display_val, editable=True)

        else:
            self._add_simple_cell(row, self.COL_DAD_FILE_ACTUAL, dad_file if dad_file else "-")

        return review_signals, missing_signals, needs_review, dad_pts

    def _populate_row_estat(self, row, review_signals, missing_signals, needs_review):
        """Omple les columnes d'estat (UIB i DAD) d'una fila."""
        # I08: Separar estat per UIB i DAD
        uib_review = [s for s in review_signals if "UIB" in s]
        dad_review = [s for s in review_signals if "DAD" in s]
        uib_missing = "UIB" in missing_signals
        dad_missing = "DAD" in missing_signals

        def create_estat_item(review_list, is_missing):
            if review_list:
                # Extreure només el percentatge si existeix
                text = "Revisar"
                color = QColor("#FCF3CF")  # Groc
            elif is_missing:
                text = "Assignar"
                color = QColor("#FADBD8")  # Rosa
            else:
                text = "OK"
                color = QColor("#D5F5E3")  # Verd
            item = QTableWidgetItem(text)
            item.setTextAlignment(Qt.AlignCenter)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            item.setBackground(QBrush(color))
            return item

        # Estat UIB (només si existeix la columna)
        if self.COL_ESTAT_UIB is not None:
            uib_item = create_estat_item(uib_review, uib_missing)
            self.samples_table.setItem(row, self.COL_ESTAT_UIB, uib_item)

        # Estat DAD
        dad_item = create_estat_item(dad_review, dad_missing)
        self.samples_table.setItem(row, self.COL_ESTAT_DAD, dad_item)

        if needs_review:
            self._unverified_fuzzy.add(row)

    # =========================================================================
    # _show_results PRINCIPAL (refactoritzat)
    # =========================================================================

    def _show_results(self, result):
        """Mostra els resultats d'importació a la taula."""
        # Inicialitzar estat
        self._init_results_state()

        # Guardar warnings d'importació per mostrar-los
        self._import_warnings = [w for w in result.get("warnings", []) if "⚠️" in w]

        # Processar manifest
        manifest = generate_import_manifest(result)
        samples = manifest.get("samples", [])
        seq_info = manifest.get("sequence", {})
        self._data_mode = seq_info.get("data_mode", "DUAL")

        # Processar fitxers orfes
        self._process_orphan_files(manifest, samples, result)

        # Configurar columnes segons mode
        self._setup_table_columns()

        # Construir llista d'injeccions ordenada
        all_injections = self._build_injection_list(samples)

        # Actualitzar barra d'informació
        self._update_info_bar(result, all_injections)

        # Omplir taula
        self._populate_table(all_injections)

        # Warnings
        self._update_warnings()

    def _populate_table(self, all_injections):
        """Omple la taula amb les injeccions."""
        self.samples_table.setSortingEnabled(False)
        self.samples_table.blockSignals(True)
        self.samples_table.setRowCount(0)
        self._sample_data = []

        for injection_num, inj in enumerate(all_injections, 1):
            row = self.samples_table.rowCount()
            self.samples_table.insertRow(row)

            sample_name = inj["sample_name"]
            sample_type = inj["sample_type"]
            rep = inj["rep"]

            # Columnes bàsiques (Inj, Mostra, Tipus, Rep, Vol, Direct)
            direct_pts = self._populate_row_basic(row, injection_num, inj)

            # Determinar si requereix assignació
            requires_assignment = sample_type in self.TYPES_REQUIRE_ASSIGNMENT
            optional_can_assign = sample_type in self.TYPES_OPTIONAL_ASSIGNMENT

            # Acumuladors per estat
            all_review_signals = []
            all_missing_signals = []
            needs_review = False
            uib_pts = 0
            dad_pts = 0

            # UIB (només si mode DUAL o UIB)
            if self._data_mode in ["DUAL", "UIB"]:
                review_uib, missing_uib, review_uib_flag, uib_pts = self._populate_row_uib(
                    row, rep, sample_name, sample_type, requires_assignment, optional_can_assign
                )
                all_review_signals.extend(review_uib)
                all_missing_signals.extend(missing_uib)
                needs_review = needs_review or review_uib_flag

            # DAD
            review_dad, missing_dad, review_dad_flag, dad_pts = self._populate_row_dad(
                row, rep, sample_name, sample_type, requires_assignment, optional_can_assign
            )
            all_review_signals.extend(review_dad)
            all_missing_signals.extend(missing_dad)
            needs_review = needs_review or review_dad_flag

            # Estat
            self._populate_row_estat(row, all_review_signals, all_missing_signals, needs_review)

            # Guardar per preview i lògica
            u = rep.get("uib", {})
            self._sample_data.append({
                "name": sample_name,
                "type": sample_type,
                "replica": rep.get("replica"),
                "direct_pts": direct_pts,
                "uib_pts": u.get("n_points", 0) if u else uib_pts,
                "dad_pts": dad_pts,
            })

        self.samples_table.blockSignals(False)
        self.samples_table.setSortingEnabled(True)
        self.samples_table.setVisible(True)
        self.table_help.setVisible(True)

    def _add_simple_cell(self, row, col, text):
        """Afegeix una cel·la simple no editable."""
        item = QTableWidgetItem(text)
        item.setTextAlignment(Qt.AlignCenter)
        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        self.samples_table.setItem(row, col, item)

    def _add_file_cell(self, row, col, text, editable=False):
        """Afegeix una cel·la de fitxer, potencialment editable."""
        item = QTableWidgetItem(text)
        item.setTextAlignment(Qt.AlignCenter)
        if not editable:
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        else:
            item.setBackground(QBrush(QColor("#FADBD8")))  # Rosa per indicar que cal assignar
        self.samples_table.setItem(row, col, item)

    def _add_suggestion_cell(self, row, col, filename, confidence, display_name=None):
        """Afegeix una cel·la amb suggeriment de matching (editable per confirmar/canviar).

        Args:
            row: Fila de la taula
            col: Columna de la taula
            filename: Nom real del fitxer orfe
            confidence: Percentatge de confiança del match
            display_name: Nom a mostrar (format llistat injeccions). Si None, usa filename.
        """
        show_name = display_name if display_name else filename
        item = QTableWidgetItem(show_name)
        item.setTextAlignment(Qt.AlignCenter)
        # Editable per permetre canviar el suggeriment
        item.setBackground(QBrush(QColor("#FCF3CF")))  # Groc per indicar revisar
        # Tooltip mostra el nom real del fitxer orfe
        item.setToolTip(f"Fitxer: {filename}\nConfiança: {confidence:.0f}%\nDoble-clic per canviar.")
        # Guardar el path real del fitxer com a data
        item.setData(Qt.UserRole, filename)
        self.samples_table.setItem(row, col, item)
        # Marcar com a suggeriment
        self._match_types[(row, col)] = "SUGGESTED"

    def _setup_table_columns(self):
        """Configura les columnes segons el mode de dades."""
        if self._data_mode == "DIRECT":
            # Sense UIB: Inj, Mostra, Tipus, Rep, Vol, Direct, Fila, DAD, Fitxer DAD, Estat DAD
            self.samples_table.setColumnCount(10)
            headers = ["Inj", "Mostra", "Tipus", "Rep", "Inj Vol", "Direct", "Fila", "DAD", "Fitxer DAD", "Estat DAD"]
            self.COL_DAD_PTS_ACTUAL = 7
            self.COL_DAD_FILE_ACTUAL = 8
            self.COL_ESTAT_UIB = None  # No UIB en mode DIRECT
            self.COL_ESTAT_DAD = 9
            self.COL_ESTAT = 9  # Compatibilitat
        else:
            # DUAL o UIB: Inj, Mostra, Tipus, Rep, Vol, Direct, Fila, UIB, Fitxer UIB, Estat UIB, DAD, Fitxer DAD, Estat DAD
            self.samples_table.setColumnCount(13)
            headers = ["Inj", "Mostra", "Tipus", "Rep", "Inj Vol", "Direct", "Fila", "UIB", "Fitxer UIB", "Estat UIB", "DAD", "Fitxer DAD", "Estat DAD"]
            self.COL_UIB_PTS_ACTUAL = 7
            self.COL_UIB_FILE_ACTUAL = 8
            self.COL_ESTAT_UIB = 9
            self.COL_DAD_PTS_ACTUAL = 10
            self.COL_DAD_FILE_ACTUAL = 11
            self.COL_ESTAT_DAD = 12
            self.COL_ESTAT = 12  # Compatibilitat (apunta a DAD)

        self.samples_table.setHorizontalHeaderLabels(headers)

        # Configurar delegates
        type_delegate = ComboBoxDelegate(list(self.sample_types_config.keys()), self)
        self.samples_table.setItemDelegateForColumn(self.COL_TIPUS, type_delegate)

        if self._data_mode == "DIRECT":
            dad_delegate = FileAssignmentDelegate(self._get_file_options, self)
            self.samples_table.setItemDelegateForColumn(self.COL_DAD_FILE_ACTUAL, dad_delegate)
        else:
            uib_delegate = FileAssignmentDelegate(self._get_file_options, self)
            dad_delegate = FileAssignmentDelegate(self._get_file_options, self)
            self.samples_table.setItemDelegateForColumn(self.COL_UIB_FILE_ACTUAL, uib_delegate)
            self.samples_table.setItemDelegateForColumn(self.COL_DAD_FILE_ACTUAL, dad_delegate)

        # Configurar mides
        header = self.samples_table.horizontalHeader()
        for col in range(self.samples_table.columnCount()):
            header.setSectionResizeMode(col, QHeaderView.ResizeToContents)

    def _add_data_cell(self, row, col, text, match_type, editable=False):
        """Afegeix una cel·la amb color segons match_type."""
        item = QTableWidgetItem(text)
        item.setTextAlignment(Qt.AlignCenter)

        if not editable:
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)

        # Color de fons
        color = MATCH_COLORS.get(match_type, MATCH_COLORS["NORMAL"])
        item.setBackground(QBrush(color))

        # Guardar match type
        self._match_types[(row, col)] = match_type
        if match_type == "FUZZY":
            self._unverified_fuzzy.add((row, col))

        self.samples_table.setItem(row, col, item)

    def _on_cell_changed(self, row, col):
        """Handler quan canvia una cel·la."""
        # Determinar si és una columna de fitxer
        is_file_col = False
        pts_col = None
        file_type = None

        if self._data_mode == "DIRECT":
            if col == self.COL_DAD_FILE_ACTUAL:
                is_file_col = True
                pts_col = self.COL_DAD_PTS_ACTUAL
                file_type = "dad"
        else:
            if col == self.COL_UIB_FILE_ACTUAL:
                is_file_col = True
                pts_col = self.COL_UIB_PTS_ACTUAL
                file_type = "uib"
            elif col == self.COL_DAD_FILE_ACTUAL:
                is_file_col = True
                pts_col = self.COL_DAD_PTS_ACTUAL
                file_type = "dad"

        if is_file_col:
            # Marcar com a MANUAL i actualitzar color
            item = self.samples_table.item(row, col)
            if item:
                new_value = item.text()
                # Obtenir nom real del fitxer (pot ser diferent del display name)
                actual_filename = item.data(Qt.UserRole)
                if not actual_filename:
                    actual_filename = new_value  # Fallback al text visible

                # Obtenir nom i rèplica de la taula (funciona amb taula ordenada)
                name_item = self.samples_table.item(row, self.COL_MOSTRA)
                rep_item = self.samples_table.item(row, self.COL_REP)
                sample_name = name_item.data(Qt.UserRole) if name_item else ""
                try:
                    replica = int(rep_item.text()) if rep_item else 1
                except:
                    replica = 1

                # Si l'usuari ha seleccionat un fitxer del dropdown (nom real),
                # convertir a format llistat d'injeccions
                if new_value and new_value not in ["-", "(cap)"]:
                    # Si el text és un nom de fitxer (conté extensió o és diferent del format esperat)
                    # convertir-lo al format del llistat d'injeccions
                    expected_display = f"{sample_name}_R{replica}"
                    if new_value != expected_display and (
                        "." in new_value or "_R" not in new_value or not new_value.startswith(sample_name)
                    ):
                        # Guardar el fitxer real seleccionat
                        actual_filename = new_value
                        item.setData(Qt.UserRole, actual_filename)
                        # Mostrar el nom segons llistat d'injeccions
                        self.samples_table.blockSignals(True)
                        item.setText(expected_display)
                        item.setToolTip(f"Fitxer: {actual_filename}")
                        self.samples_table.blockSignals(False)

                    # Guardar assignació manual per al manifest (amb el path real)
                    # Usar clau (sample_name, replica) en lloc de row per suportar taula ordenada
                    if not hasattr(self, '_manual_assignments'):
                        self._manual_assignments = {}
                    key = (sample_name, replica)
                    self._manual_assignments.setdefault(key, {})[col] = actual_filename
                    # Marcar que hi ha canvis sense guardar
                    self.main_window.mark_unsaved_changes()
                    # Si era un suggeriment, marcar com CONFIRMED (verd)
                    # Sinó, marcar com MANUAL (blau)
                    prev_type = self._match_types.get((row, col), "")
                    if prev_type == "SUGGESTED":
                        item.setBackground(QBrush(MATCH_COLORS["EXACT"]))  # Verd clar = confirmat
                        self._match_types[(row, col)] = "CONFIRMED"
                    elif prev_type not in ["CONFIRMED", "EXACT"]:
                        item.setBackground(QBrush(MATCH_COLORS["MANUAL"]))  # Blau = manual
                        self._match_types[(row, col)] = "MANUAL"

                    # Actualitzar nombre de punts (usant el nom real del fitxer)
                    if pts_col is not None:
                        n_points = self._count_file_points(actual_filename, file_type)
                        pts_item = self.samples_table.item(row, pts_col)
                        if pts_item and n_points > 0:
                            pts_item.setText(str(n_points))
                            # Actualitzar _sample_data (buscar per nom/replica)
                            name_item = self.samples_table.item(row, self.COL_MOSTRA)
                            rep_item = self.samples_table.item(row, self.COL_REP)
                            if name_item and rep_item:
                                s_name = name_item.data(Qt.UserRole)
                                s_rep = rep_item.text()
                                for data in self._sample_data:
                                    if (data.get("name") == s_name and
                                        str(data.get("replica", "")) == s_rep):
                                        if file_type == "uib":
                                            data["uib_pts"] = n_points
                                        elif file_type == "dad":
                                            data["dad_pts"] = n_points
                                        break

                    # Carregar dades del fitxer per a la gràfica
                    self._load_and_store_file_data(actual_filename, file_type, sample_name, replica)
                    # Auto-guardar manifest per persistir assignació (I10)
                    try:
                        save_import_manifest(self.imported_data)
                        self.main_window.mark_manifest_saved()
                    except Exception as e:
                        print(f"Warning: No s'ha pogut auto-guardar manifest: {e}")
                else:
                    # Si es tria "(cap)", la cel·la queda sense assignació
                    item.setBackground(QBrush(MATCH_COLORS["NORMAL"]))
                    self._match_types[(row, col)] = "NONE"
                    # Restaurar punts a "-"
                    if pts_col is not None:
                        pts_item = self.samples_table.item(row, pts_col)
                        if pts_item:
                            pts_item.setText("-")

                # Recalcular l'estat de la fila
                self._update_row_state(row)

                self._update_next_button_state()

    def _count_file_points(self, filename, file_type):
        """Compta el nombre de punts d'un fitxer orfe."""
        if not filename or filename in ["-", "(cap)"]:
            return 0

        # Buscar el path complet del fitxer (usant llista amb paths complets)
        orphan_list = self._orphan_files_full.get(file_type, [])
        full_path = None
        for f in orphan_list:
            if Path(f).name == filename:
                full_path = f
                break

        if not full_path or not os.path.exists(full_path):
            return 0

        try:
            import pandas as pd
            # DAD files sovint són UTF-16, provar primer
            encodings = ['utf-16', 'utf-8', 'latin-1', 'cp1252']
            best_count = 0

            for encoding in encodings:
                try:
                    count = 0
                    if file_type == "uib":
                        df = pd.read_csv(full_path, sep=None, engine='python',
                                        encoding=encoding)
                        count = len(df)
                    elif file_type == "dad":
                        if full_path.lower().endswith('.csv'):
                            df = pd.read_csv(full_path, sep=None, engine='python',
                                            encoding=encoding)
                            count = len(df)
                        else:
                            with open(full_path, 'r', encoding=encoding) as f:
                                lines = f.readlines()
                            data_lines = [l for l in lines if l.strip() and not l.startswith('#')]
                            count = max(0, len(data_lines) - 1)

                    # Guardar el millor resultat (més files = encoding correcte)
                    if count > best_count:
                        best_count = count
                        # Si tenim un bon nombre de punts, retornar
                        if count > 100:
                            return count
                except (UnicodeDecodeError, UnicodeError):
                    continue
                except Exception:
                    continue

            return best_count

        except Exception as e:
            print(f"Warning: No s'ha pogut comptar punts de {filename}: {e}")

        return 0

    def _load_and_store_file_data(self, filename, file_type, sample_name, replica):
        """
        Carrega les dades d'un fitxer orfe assignat manualment i les guarda a imported_data.

        Args:
            filename: Nom del fitxer (sense path)
            file_type: "uib" o "dad"
            sample_name: Nom de la mostra
            replica: Número de rèplica
        """
        if not filename or filename in ["-", "(cap)"]:
            return

        print(f"[DEBUG _load_and_store] Intentant carregar {filename} per {sample_name} R{replica}")

        # Obtenir referència a les dades de la rèplica primer
        samples = self.imported_data.get("samples", {})
        if sample_name not in samples:
            print(f"[DEBUG _load_and_store] ERROR: Mostra '{sample_name}' no trobada a imported_data")
            print(f"[DEBUG _load_and_store] Samples disponibles: {list(samples.keys())[:5]}...")
            return

        rep_data = samples[sample_name].get("replicas", {}).get(str(replica))
        if rep_data is None:
            print(f"[DEBUG _load_and_store] ERROR: Rèplica {replica} no trobada per '{sample_name}'")
            print(f"[DEBUG _load_and_store] Rèpliques disponibles: {list(samples[sample_name].get('replicas', {}).keys())}")
            return

        # Buscar el path complet del fitxer
        full_path = None

        # 1. Primer buscar a la llista d'orfes completa
        orphan_list = self._orphan_files_full.get(file_type, [])
        for f in orphan_list:
            if Path(f).name == filename:
                full_path = f
                break

        # 2. Si no es troba, mirar si hi ha un suggeriment amb el path
        if not full_path:
            suggestion_key = f"{file_type}_suggestion"
            suggestion = rep_data.get(suggestion_key, {})
            if suggestion:
                # El suggeriment pot tenir el path complet a "file"
                sugg_file = suggestion.get("file", "")
                if sugg_file and Path(sugg_file).name == filename:
                    full_path = sugg_file
                elif sugg_file:
                    # Construir path a partir del seq_path
                    if hasattr(self, 'seq_path') and self.seq_path:
                        if file_type == "uib":
                            test_path = os.path.join(self.seq_path, "CSV", sugg_file)
                        else:
                            test_path = os.path.join(self.seq_path, "Export3d", sugg_file)
                        if os.path.exists(test_path):
                            full_path = test_path

        # 3. Construir path manualment si encara no es troba
        if not full_path and hasattr(self, 'seq_path') and self.seq_path:
            if file_type == "uib":
                possible_paths = [
                    os.path.join(self.seq_path, "CSV", filename),
                    os.path.join(self.seq_path, "csv", filename),
                ]
            else:
                possible_paths = [
                    os.path.join(self.seq_path, "Export3d", filename),
                    os.path.join(self.seq_path, "Export3D", filename),
                    os.path.join(self.seq_path, "CSV", filename),
                    os.path.join(self.seq_path, "csv", filename),
                ]
            for p in possible_paths:
                if os.path.exists(p):
                    full_path = p
                    break

        if not full_path or not os.path.exists(full_path):
            print(f"Warning: No s'ha trobat el fitxer {filename} (file_type={file_type})")
            return

        try:
            if file_type == "uib":
                # Carregar UIB
                df, status = llegir_doc_uib(full_path)
                if not df.empty and "OK" in status:
                    t = df["time (min)"].values
                    y = df["DOC"].values

                    # Calcular baseline i y_net (CRÍTIC per areas_uib)
                    baseline = None
                    y_net = None
                    if len(t) > 10:
                        # Determinar mode (BP o COLUMN)
                        method = self.imported_data.get("method", "COLUMN")
                        mode = "BP" if method == "BP" else "COLUMN"
                        baseline = get_baseline_value(t, y, mode=mode)
                        y_net = np.array(y) - baseline

                    rep_data["uib"] = {
                        "t": t,
                        "y": y,
                        "y_raw": y,
                        "y_net": y_net,
                        "baseline": baseline,
                        "file": filename,
                        "n_points": len(df),
                        "manual_assignment": True,
                    }
                    bl_val = baseline if baseline is not None else 0
                    print(f"[DEBUG _load_and_store] Carregat UIB: {filename} per {sample_name} R{replica}")
                    print(f"[DEBUG _load_and_store] rep_data['uib'] keys: {list(rep_data['uib'].keys())}")
                    print(f"[DEBUG _load_and_store] t is not None: {t is not None}, len(t)={len(t) if t is not None else 0}")
                    # Treure de la llista d'orfes
                    if "orphan_files" in self.imported_data:
                        uib_orphans = self.imported_data["orphan_files"].get("uib", [])
                        self.imported_data["orphan_files"]["uib"] = [
                            f for f in uib_orphans if Path(f).name != filename
                        ]

            elif file_type == "dad":
                # Provar primer Export3D, després DAD1A
                df, status = llegir_dad_export3d(full_path)
                if df.empty or "Error" in status:
                    df, status = llegir_dad_1a(full_path)

                if not df.empty and "OK" in status:
                    # Assegurar que tenim una còpia del DataFrame
                    df = df.copy()
                    rep_data["dad"] = {
                        "df": df,
                        "t": df["time (min)"].values if "time (min)" in df.columns else None,
                        "file": filename,
                        "n_points": len(df),
                        "manual_assignment": True,
                        "source": "manual",
                    }
                    print(f"Carregat DAD: {filename} ({len(df)} punts, columnes: {list(df.columns)[:5]})")
                    # Treure de la llista d'orfes
                    if "orphan_files" in self.imported_data:
                        dad_orphans = self.imported_data["orphan_files"].get("dad", [])
                        self.imported_data["orphan_files"]["dad"] = [
                            f for f in dad_orphans if Path(f).name != filename
                        ]
                else:
                    print(f"Warning: No s'han pogut llegir dades DAD de {filename}: {status}")

        except Exception as e:
            print(f"Error carregant {filename}: {e}")

    def _update_row_state(self, row):
        """Actualitza l'estat d'una fila específica."""
        # Obtenir tipus de mostra (buscar per nom/replica per suportar taula ordenada)
        sample_type = "MOSTRA"
        name_item = self.samples_table.item(row, self.COL_MOSTRA)
        rep_item = self.samples_table.item(row, self.COL_REP)
        if name_item and rep_item:
            s_name = name_item.data(Qt.UserRole)
            s_rep = rep_item.text()
            for data in self._sample_data:
                if (data.get("name") == s_name and
                    str(data.get("replica", "")) == s_rep):
                    sample_type = data.get("type", "MOSTRA")
                    break

        # BLANC i CONTROL no requereixen assignació de DAD/UIB
        # Només MOSTRA, PATRÓ_CAL, PATRÓ_REF necessiten verificació
        requires_assignment = sample_type in self.TYPES_REQUIRE_ASSIGNMENT

        missing = []
        pending_review = []

        if requires_assignment:
            # Comprovar UIB
            if self._data_mode in ["DUAL", "UIB"]:
                uib_item = self.samples_table.item(row, self.COL_UIB_FILE_ACTUAL)
                if uib_item:
                    val = uib_item.text()
                    match_type = self._match_types.get((row, self.COL_UIB_FILE_ACTUAL), "")
                    if val == "-" and self._orphan_files.get("uib"):
                        missing.append("UIB")
                    elif match_type == "SUGGESTED":
                        pending_review.append("UIB")

            # Comprovar DAD
            dad_item = self.samples_table.item(row, self.COL_DAD_FILE_ACTUAL)
            if dad_item:
                val = dad_item.text()
                match_type = self._match_types.get((row, self.COL_DAD_FILE_ACTUAL), "")
                if val == "-" and self._orphan_files.get("dad"):
                    missing.append("DAD")
                elif match_type == "SUGGESTED":
                    pending_review.append("DAD")

        # I08: Actualitzar Estat UIB i DAD per separat
        uib_missing = "UIB" in missing
        dad_missing = "DAD" in missing
        uib_review = "UIB" in pending_review
        dad_review = "DAD" in pending_review

        def update_estat_cell(col, is_missing, is_review):
            if col is None:
                return
            item = self.samples_table.item(row, col)
            if item:
                if is_missing:
                    item.setText("Assignar")
                    item.setBackground(QBrush(QColor("#FADBD8")))  # Rosa
                elif is_review:
                    item.setText("Revisar")
                    item.setBackground(QBrush(QColor("#FCF3CF")))  # Groc
                else:
                    item.setText("OK")
                    item.setBackground(QBrush(QColor("#D5F5E3")))  # Verd

        # Actualitzar UIB (si existeix)
        if self.COL_ESTAT_UIB is not None:
            update_estat_cell(self.COL_ESTAT_UIB, uib_missing, uib_review)

        # Actualitzar DAD
        update_estat_cell(self.COL_ESTAT_DAD, dad_missing, dad_review)

        # Marcar fila com no verificada si hi ha pendents
        if missing or pending_review:
            self._unverified_fuzzy.add(row)
        else:
            self._unverified_fuzzy.discard(row)

    def _on_cell_double_clicked(self, row, col):
        """Handler de doble clic."""
        # Determinar columnes editables
        editable_cols = [self.COL_TIPUS]
        if self._data_mode == "DIRECT":
            editable_cols.append(self.COL_DAD_FILE_ACTUAL)
        else:
            editable_cols.extend([self.COL_UIB_FILE_ACTUAL, self.COL_DAD_FILE_ACTUAL])

        # Si és columna editable, deixar que el delegate s'encarregui
        if col in editable_cols:
            return

        # Altrament, mostrar preview
        # Obtenir nom i rèplica de les cel·les (per funcionar amb taula ordenada)
        name_item = self.samples_table.item(row, self.COL_MOSTRA)
        rep_item = self.samples_table.item(row, self.COL_REP)

        if name_item and rep_item:
            sample_name = name_item.data(Qt.UserRole)
            rep_text = rep_item.text()

            # Buscar les dades corresponents a _sample_data
            sample_data = None
            for data in self._sample_data:
                # Comparar amb conversió a string per evitar problemes de tipus
                if (data.get("name") == sample_name and
                    str(data.get("replica", "")) == rep_text):
                    sample_data = data
                    break

            if sample_data:
                try:
                    replica = int(rep_text) if rep_text.isdigit() else 1
                except:
                    replica = 1
                dialog = ChromatogramPreviewDialog(
                    self,
                    sample_name,
                    replica,
                    sample_data,
                    self.imported_data
                )
                dialog.exec()

    def _update_warnings(self):
        """Actualitza avisos al CommonToolbar (G01-G06: estructura unificada)."""
        # Recollir TOTS els avisos en una llista
        warnings_list = []

        # 1. Warnings d'importació (injeccions faltants, etc.)
        import_warnings = getattr(self, '_import_warnings', [])
        for w in import_warnings:
            clean_w = w.replace("⚠️ ", "").replace("⚠", "").strip()
            if clean_w:
                warnings_list.append(clean_w)

        # 2. Suggeriments pendents
        pending_suggestions = 0
        suggestion_samples = []
        for (r, c), mt in self._match_types.items():
            if mt == "SUGGESTED":
                pending_suggestions += 1
                name_item = self.samples_table.item(r, self.COL_MOSTRA)
                if name_item and name_item.text() not in suggestion_samples:
                    suggestion_samples.append(name_item.text())

        if pending_suggestions:
            samples_preview = ", ".join(suggestion_samples[:3])
            if len(suggestion_samples) > 3:
                samples_preview += f"... (+{len(suggestion_samples)-3})"
            warnings_list.append(f"{pending_suggestions} suggeriments FUZZY: {samples_preview}")

        # 3. Fitxers orfes
        unassigned_uib, unassigned_dad = self._count_unassigned_orphans()
        if unassigned_uib > 0:
            warnings_list.append(f"{unassigned_uib} fitxers UIB sense assignar")
        if unassigned_dad > 0:
            warnings_list.append(f"{unassigned_dad} fitxers DAD sense assignar")

        # Nota: Avisos es gestionen des del wizard header

        # Amagar elements legacy
        self.warnings_frame.setVisible(False)

    def _get_assigned_files_from_table(self, include_path_variants=False):
        """Obté els fitxers assignats des de la taula.

        Args:
            include_path_variants: Si True, afegeix també Path(val).name per matching més flexible.

        Returns:
            tuple: (assigned_uib: set, assigned_dad: set)
        """
        assigned_uib = set()
        assigned_dad = set()

        for row in range(self.samples_table.rowCount()):
            if self._data_mode in ["DUAL", "UIB"]:
                uib_item = self.samples_table.item(row, self.COL_UIB_FILE_ACTUAL)
                if uib_item:
                    val = uib_item.data(Qt.UserRole) or uib_item.text()
                    if val and val not in ["-", "(cap)"]:
                        assigned_uib.add(val)
                        if include_path_variants:
                            assigned_uib.add(Path(val).name)

            dad_item = self.samples_table.item(row, self.COL_DAD_FILE_ACTUAL)
            if dad_item:
                val = dad_item.data(Qt.UserRole) or dad_item.text()
                if val and val not in ["-", "(cap)", "[MasterFile]", "[export3d]", "[csv]"]:
                    assigned_dad.add(val)
                    if include_path_variants:
                        assigned_dad.add(Path(val).name)

        return assigned_uib, assigned_dad

    def _count_unassigned_orphans(self):
        """Compta quants orfes encara no estan assignats."""
        assigned_uib, assigned_dad = self._get_assigned_files_from_table()

        orphan_uib = self._orphan_files.get("uib", [])
        orphan_dad = self._orphan_files.get("dad", [])

        unassigned_uib = sum(1 for f in orphan_uib if Path(f).name not in assigned_uib)
        unassigned_dad = sum(1 for f in orphan_dad if Path(f).name not in assigned_dad)

        return unassigned_uib, unassigned_dad

    def _confirm_all_suggestions(self):
        """Confirma tots els suggeriments automàtics i carrega les dades."""
        confirmed = 0
        self.samples_table.blockSignals(True)

        for row in range(self.samples_table.rowCount()):
            # Obtenir nom i rèplica de la fila
            name_item = self.samples_table.item(row, self.COL_MOSTRA)
            rep_item = self.samples_table.item(row, self.COL_REP)
            if not name_item or not rep_item:
                continue
            sample_name = name_item.data(Qt.UserRole)
            try:
                replica = int(rep_item.text())
            except:
                replica = 1

            # Comprovar UIB
            if self._data_mode in ["DUAL", "UIB"]:
                if self._match_types.get((row, self.COL_UIB_FILE_ACTUAL)) == "SUGGESTED":
                    item = self.samples_table.item(row, self.COL_UIB_FILE_ACTUAL)
                    if item:
                        item.setBackground(QBrush(MATCH_COLORS["EXACT"]))
                        self._match_types[(row, self.COL_UIB_FILE_ACTUAL)] = "CONFIRMED"
                        confirmed += 1
                        # Carregar dades del fitxer
                        filename = item.data(Qt.UserRole)
                        if filename:
                            self._load_and_store_file_data(filename, "uib", sample_name, replica)
                            # Guardar a _manual_assignments per persistència
                            key = (sample_name, replica)
                            self._manual_assignments.setdefault(key, {})[self.COL_UIB_FILE_ACTUAL] = filename

            # Comprovar DAD
            if self._match_types.get((row, self.COL_DAD_FILE_ACTUAL)) == "SUGGESTED":
                item = self.samples_table.item(row, self.COL_DAD_FILE_ACTUAL)
                if item:
                    item.setBackground(QBrush(MATCH_COLORS["EXACT"]))
                    self._match_types[(row, self.COL_DAD_FILE_ACTUAL)] = "CONFIRMED"
                    confirmed += 1
                    # Carregar dades del fitxer
                    filename = item.data(Qt.UserRole)
                    if filename:
                        self._load_and_store_file_data(filename, "dad", sample_name, replica)
                        # Guardar a _manual_assignments per persistència
                        key = (sample_name, replica)
                        self._manual_assignments.setdefault(key, {})[self.COL_DAD_FILE_ACTUAL] = filename

        self.samples_table.blockSignals(False)

        # Marcar canvis sense guardar perquè es persisteixin
        if confirmed > 0:
            self.main_window.mark_unsaved_changes()

        # Recalcular estats
        self._recalculate_row_states()
        self._update_warnings()
        self._update_next_button_state()

        if confirmed > 0:
            # Marcar warnings com a confirmats per evitar que reapareguin
            self._warnings_confirmed = True
            if self.imported_data:
                self.imported_data["warnings_confirmed"] = True

            # Guardar manifest immediatament
            try:
                print(f"[DEBUG confirm] Guardant manifest amb {confirmed} confirmacions...")
                self._apply_manual_assignments()
                manifest_path = save_import_manifest(self.imported_data)
                print(f"[DEBUG confirm] Manifest guardat a: {manifest_path}")
                self.main_window.mark_manifest_saved()
                self.warnings_frame.setVisible(False)
                QMessageBox.information(self, "Confirmat", f"S'han confirmat {confirmed} suggeriments i s'han guardat.")
            except Exception as e:
                import traceback
                print(f"[DEBUG confirm] ERROR: {e}")
                traceback.print_exc()
                QMessageBox.warning(self, "Avís", f"S'han confirmat {confirmed} suggeriments però no s'han pogut guardar: {e}")


    def _recalculate_row_states(self):
        """Recalcula l'estat de cada fila basant-se en assignacions actuals."""
        self.samples_table.blockSignals(True)
        self._unverified_fuzzy.clear()

        for row in range(self.samples_table.rowCount()):
            # Obtenir tipus de mostra (buscar per nom/replica)
            sample_type = "MOSTRA"
            name_item = self.samples_table.item(row, self.COL_MOSTRA)
            rep_item = self.samples_table.item(row, self.COL_REP)
            if name_item and rep_item:
                s_name = name_item.data(Qt.UserRole)
                s_rep = rep_item.text()
                for data in self._sample_data:
                    if (data.get("name") == s_name and
                        str(data.get("replica", "")) == s_rep):
                        sample_type = data.get("type", "MOSTRA")
                        break

            # BLANC i CONTROL no requereixen assignació
            requires_assignment = sample_type in self.TYPES_REQUIRE_ASSIGNMENT

            missing = []

            if requires_assignment:
                # Comprovar UIB
                if self._data_mode in ["DUAL", "UIB"]:
                    uib_item = self.samples_table.item(row, self.COL_UIB_FILE_ACTUAL)
                    if uib_item:
                        val = uib_item.text()
                        if val == "-" and self._orphan_files.get("uib"):
                            missing.append("UIB")

                # Comprovar DAD
                dad_item = self.samples_table.item(row, self.COL_DAD_FILE_ACTUAL)
                if dad_item:
                    val = dad_item.text()
                    if val == "-" and self._orphan_files.get("dad"):
                        missing.append("DAD")

            # I08: Actualitzar Estat UIB i DAD per separat
            uib_missing = "UIB" in missing
            dad_missing = "DAD" in missing

            def update_estat(col, is_missing):
                if col is None:
                    return
                item = self.samples_table.item(row, col)
                if item:
                    if is_missing:
                        item.setText("Assignar")
                        item.setBackground(QBrush(QColor("#FADBD8")))
                    else:
                        item.setText("OK")
                        item.setBackground(QBrush(QColor("#D5F5E3")))

            if self.COL_ESTAT_UIB is not None:
                update_estat(self.COL_ESTAT_UIB, uib_missing)
            update_estat(self.COL_ESTAT_DAD, dad_missing)

            if missing:
                self._unverified_fuzzy.add(row)

        self.samples_table.blockSignals(False)

    def _update_next_button_state(self):
        """Actualitza l'estat del botó Següent."""
        # Verificar si hi ha FUZZY sense verificar
        if self._unverified_fuzzy:
            self.next_btn.setEnabled(False)
            self.next_btn.setToolTip(
                f"Cal verificar {len(self._unverified_fuzzy)} assignacions (taronja)"
            )
        else:
            self.next_btn.setEnabled(True)
            self.next_btn.setToolTip("")

    def _refresh_orphan_count(self):
        """Actualitza el comptador d'orfes i la llista després d'assignacions manuals."""
        assigned_uib, assigned_dad = self._get_assigned_files_from_table(include_path_variants=True)

        # Actualitzar llista d'orfes (treure els assignats)
        orig_uib = self._orphan_files.get("uib", [])
        orig_dad = self._orphan_files.get("dad", [])

        self._orphan_files["uib"] = [f for f in orig_uib if Path(f).name not in assigned_uib]
        self._orphan_files["dad"] = [f for f in orig_dad if Path(f).name not in assigned_dad]

        # Actualitzar warnings
        self._update_warnings()

    def _dismiss_orphan_warning(self):
        """Marca l'avís d'orfes com a revisat i amaga la barra d'avisos."""
        self._orphan_warning_dismissed = True
        self.warnings_frame.setVisible(False)
        # Guardar al manifest que l'avís ha estat revisat
        if self.imported_data:
            self.imported_data["orphan_warning_dismissed"] = True
            self.imported_data["warnings_confirmed"] = True  # Marcar warnings com confirmats
            try:
                save_import_manifest(self.imported_data)
                self.main_window.set_status("Avís marcat com a revisat", 3000)
            except Exception as e:
                print(f"Warning: No s'ha pogut guardar estat revisat: {e}")

        # Notificar al wizard que els warnings s'han descartat
        self.warnings_dismissed.emit()

    def _show_orphans(self):
        # Preparar dades amb punts per cada fitxer orfe
        orphans_with_info = {"uib": [], "dad": []}

        for file_type in ["uib", "dad"]:
            files = self._orphan_files.get(file_type, [])
            for f in sorted(files):  # Ordenar alfabèticament
                n_points = self._count_file_points(f, file_type)
                orphans_with_info[file_type].append({
                    "file": f,
                    "n_points": n_points
                })

        dialog = OrphanFilesDialog(self, orphans_with_info)
        dialog.exec()

    def _on_warnings_confirmed(self, initials: str):
        """Handler quan es confirmen avisos via CommonToolbar."""
        self._warnings_confirmed = True
        self._warnings_confirmed_by = initials
        self._orphan_warning_dismissed = True

        # Guardar al manifest
        if self.imported_data:
            self.imported_data["warnings_confirmed"] = True
            self.imported_data["warnings_confirmed_by"] = initials
            self.imported_data["orphan_warning_dismissed"] = True
            try:
                save_import_manifest(self.imported_data)
                self.main_window.set_status(f"Avisos confirmats per {initials}", 3000)
            except Exception as e:
                print(f"Warning: No s'ha pogut guardar: {e}")

        # Notificar wizard
        self.warnings_dismissed.emit()
        self._update_next_button_state()

    def _on_notes_changed(self, notes: str):
        """Handler quan canvien les notes via CommonToolbar."""
        if self.imported_data:
            self.imported_data["notes"] = notes
            try:
                save_import_manifest(self.imported_data)
            except Exception as e:
                print(f"Warning: No s'ha pogut guardar notes: {e}")

    def _save_manifest(self):
        """Guarda el manifest amb les assignacions actuals."""
        if not self.imported_data:
            QMessageBox.warning(self, "Sense dades", "No hi ha dades per guardar.")
            return

        try:
            # Aplicar assignacions manuals
            self._apply_manual_assignments()
            # Guardar manifest
            save_import_manifest(self.imported_data)
            self.main_window.mark_manifest_saved()
            self.main_window.set_status("Manifest guardat correctament", 3000)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error guardant manifest:\n{e}")

    def _go_next(self):
        # Verificar si hi ha orfes sense assignar
        unassigned_uib, unassigned_dad = self._count_unassigned_orphans()
        has_orphans = unassigned_uib > 0 or unassigned_dad > 0
        has_unsaved = self.main_window.has_unsaved_changes

        # Si s'ha carregat des de manifest sense fer canvis, passar directament
        # (els orfes ja eren coneguts quan es va guardar el manifest)
        if self._loaded_from_manifest and not has_unsaved:
            self.main_window.go_to_tab(1)
            return

        # Si no hi ha orfes ni canvis sense guardar, passar directament
        if not has_orphans and not has_unsaved:
            self.main_window.go_to_tab(1)
            return

        # Construir missatge de confirmació
        msg_parts = []
        if has_orphans:
            msg_parts.append("Hi ha fitxers orfes sense assignar:")
            if unassigned_uib:
                msg_parts.append(f"  • {unassigned_uib} fitxers UIB")
            if unassigned_dad:
                msg_parts.append(f"  • {unassigned_dad} fitxers DAD")
            msg_parts.append("")

        if has_unsaved:
            msg_parts.append("Es guardaran els canvis i es passarà a la fase de calibració.")
        else:
            msg_parts.append("Es passarà a la fase de calibració.")

        msg_parts.append("\nVols continuar?")

        reply = QMessageBox.question(
            self, "Continuar",
            "\n".join(msg_parts),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        if reply != QMessageBox.Yes:
            return

        # Aplicar assignacions manuals i guardar si hi ha canvis
        if has_unsaved:
            self._apply_manual_assignments()
            try:
                save_import_manifest(self.imported_data)
                self.main_window.mark_manifest_saved()
            except Exception as e:
                print(f"Warning: No s'ha pogut guardar manifest: {e}")

        self.main_window.go_to_tab(1)

    def _apply_manual_assignments(self):
        """Aplica les assignacions manuals a imported_data."""
        if not hasattr(self, '_manual_assignments') or not self._manual_assignments:
            return

        # Iterar per les assignacions guardades amb clau (sample_name, replica)
        for key, assignments in self._manual_assignments.items():
            sample_name, replica = key

            # Obtenir dades de la mostra
            samples = self.imported_data.get("samples", {})
            if sample_name not in samples:
                continue

            rep_data = samples[sample_name].get("replicas", {}).get(str(replica))
            if not rep_data:
                continue

            # Aplicar assignacions
            for col, filename in assignments.items():
                if filename in ["-", "(cap)", ""]:
                    continue

                # Determinar tipus de senyal
                if col == self.COL_UIB_FILE_ACTUAL:
                    signal_type = "uib"
                elif col == self.COL_DAD_FILE_ACTUAL:
                    signal_type = "dad"
                else:
                    continue

                # Marcar com a assignació manual
                # Assegurar que el dict existeix (pot ser None o absent)
                if signal_type not in rep_data or rep_data[signal_type] is None:
                    rep_data[signal_type] = {}
                rep_data[signal_type]["manual_assignment"] = True
                rep_data[signal_type]["manual_file"] = filename
                rep_data[signal_type]["file"] = filename  # També guardar com a file per compatibilitat

        # Actualitzar manifest
        try:
            save_import_manifest(self.imported_data)
        except Exception as e:
            print(f"Warning: No s'ha pogut actualitzar manifest: {e}")
