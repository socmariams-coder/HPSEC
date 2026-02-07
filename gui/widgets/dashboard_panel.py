# -*- coding: utf-8 -*-
"""
HPSEC Suite - Dashboard Panel
==============================

Vista general de totes les seqüències.
Disseny minimalista amb informació clara per columnes.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QFrame,
    QProgressBar, QComboBox, QLineEdit, QMessageBox, QInputDialog,
    QStyledItemDelegate
)
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QFont, QColor, QBrush

import os
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gui.models.sequence_state import SequenceState, Phase, get_all_sequences
from hpsec_config import get_config
from hpsec_import import import_sequence, import_sequence_pack, save_import_manifest, import_from_manifest, load_manifest
from hpsec_calibrate import calibrate_from_import
from hpsec_analyze import analyze_sequence, save_analysis_result
from hpsec_reports import generate_import_plots, generate_calibration_plots, generate_analysis_plots

# Contrasenya per operacions batch i reset
BATCH_PASSWORD = "LEQUIA"


class SortableTableItem(QTableWidgetItem):
    """Item que ordena per UserRole si existeix, sinó per text."""
    def __lt__(self, other):
        my_data = self.data(Qt.UserRole)
        other_data = other.data(Qt.UserRole) if other else None

        # Si tots dos tenen UserRole numèric, ordenar per això
        if my_data is not None and other_data is not None:
            try:
                return float(my_data) < float(other_data)
            except (TypeError, ValueError):
                pass

        # Altrament, ordenar per text
        return self.text() < (other.text() if other else "")

# Colors per fases
COLOR_OK = "#27AE60"       # Verd
COLOR_WARNING = "#F39C12"  # Taronja
COLOR_ERROR = "#E74C3C"    # Vermell
COLOR_PENDING = "#BDC3C7"  # Gris
COLOR_CURRENT = "#2E86AB"  # Blau (fase actual)


# =============================================================================
# FUNCIONS CORE PER FASES INDIVIDUALS
# Cada funció executa UNA sola fase - cridem les funcions de hpsec_*.py
# =============================================================================

def run_import(seq_path, default_uib_sensitivity=None, siblings=None):
    """
    Executa IMPORT per una seqüència. Retorna (success, message, data).

    Args:
        seq_path: Path de la seqüència principal
        default_uib_sensitivity: Sensibilitat UIB per defecte (opcional)
        siblings: Llista de paths de siblings (282B_SEQ, 282C_SEQ...) o None
    """
    try:
        # Si hi ha siblings, usar import_sequence_pack
        if siblings:
            all_paths = [seq_path] + siblings
            result = import_sequence_pack(all_paths)
        else:
            result = import_sequence(seq_path)

        if result and result.get('success'):
            # Aplicar sensibilitat UIB per defecte si cal
            data_mode = result.get("data_mode", "")
            current_uib_sens = result.get("uib_sensitivity")
            if data_mode in ["DUAL", "UIB"] and not current_uib_sens and default_uib_sensitivity:
                result["uib_sensitivity"] = default_uib_sensitivity
                # Actualitzar MasterFile si existeix
                try:
                    master_file = result.get("master_file")
                    if master_file and os.path.exists(master_file):
                        import openpyxl
                        wb = openpyxl.load_workbook(master_file)
                        if "0-INFO" in wb.sheetnames:
                            ws = wb["0-INFO"]
                            ws["B5"] = default_uib_sensitivity
                            wb.save(master_file)
                except Exception:
                    pass  # Continuar sense actualitzar MasterFile

            save_import_manifest(result)
            # Generar gràfics
            try:
                generate_import_plots(seq_path, result)
            except Exception:
                pass  # Continuar sense gràfics
            return True, "OK", result
        errors = result.get('errors', ['?']) if result else ['?']
        return False, f"Error: {errors[0]}", None
    except Exception as e:
        return False, str(e), None


def run_calibrate(seq_path):
    """Executa CALIBRATE per una seqüència. Retorna (success, message, data)."""
    try:
        # IMPORTANT: El manifest JSON només conté metadades, no les dades reals.
        # Cal usar import_from_manifest per carregar les dades des dels fitxers.
        manifest_path = Path(seq_path) / "CHECK" / "data" / "import_manifest.json"
        if not manifest_path.exists():
            return False, "No importat", None

        # Reimportar dades usant el manifest
        imported = import_from_manifest(seq_path)
        if not imported or not imported.get("success"):
            errors = imported.get("errors", ["Error desconegut"]) if imported else ["Error importació"]
            return False, f"Error importació: {errors[0]}", None

        result = calibrate_from_import(imported)
        if result and result.get('success'):
            # Generar gràfics
            try:
                generate_calibration_plots(seq_path, result, imported)
            except Exception:
                pass  # Continuar sense gràfics
            return True, "OK", result
        return False, "Sense KHP", None
    except Exception as e:
        return False, str(e), None


def run_analyze(seq_path):
    """Executa ANALYZE per una seqüència. Retorna (success, message, data)."""
    try:
        import json
        data_path = Path(seq_path) / "CHECK" / "data"

        # Llegir manifest i reimportar les dades completes
        # IMPORTANT: El manifest JSON només conté metadades, no les dades reals.
        # Cal usar import_from_manifest per carregar les dades des dels fitxers.
        manifest_path = data_path / "import_manifest.json"
        if not manifest_path.exists():
            return False, "No importat", None

        # Reimportar dades usant el manifest
        imported = import_from_manifest(seq_path)
        if not imported or not imported.get("success"):
            errors = imported.get("errors", ["Error desconegut"]) if imported else ["Error importació"]
            return False, f"Error importació: {errors[0]}", None

        # Llegir calibració (opcional)
        cal_path = data_path / "calibration_result.json"
        if cal_path.exists():
            with open(cal_path, "r", encoding="utf-8") as f:
                calibrated = json.load(f)
        else:
            calibrated = {"factor": 1.0, "shift_uib": 0, "shift_direct": 0}

        result = analyze_sequence(imported, calibrated)
        if result and result.get('success'):
            save_analysis_result(result)
            # Generar gràfics
            try:
                generate_analysis_plots(seq_path, result)
            except Exception:
                pass  # Continuar sense gràfics
            return True, "OK", result
        errors = result.get('errors', ['?']) if result else ['?']
        return False, f"Error: {errors[0]}", None
    except Exception as e:
        return False, str(e), None




class SingleSeqWorker(QThread):
    """
    Worker per processar UNA seqüència (cas habitual).
    Executa totes les fases pendents: Import → Calibrate → Analyze.
    """
    progress = Signal(str)
    finished = Signal(bool, str)

    def __init__(self, seq_state):
        super().__init__()
        self.seq = seq_state

    def run(self):
        """Processa una seqüència executant les fases pendents."""
        seq_path = self.seq.seq_path
        siblings = self.seq.siblings if hasattr(self.seq, 'siblings') else []
        errors = []

        # IMPORT (si pendent)
        if not self.seq.import_status.completed:
            if siblings:
                self.progress.emit(f"Importar pack [{len(siblings)+1} carpetes]...")
            else:
                self.progress.emit("Importar...")
            ok, msg, _ = run_import(seq_path, siblings=siblings)
            if not ok:
                self.finished.emit(False, f"Import: {msg}")
                return

        # CALIBRATE (si pendent)
        if not self.seq.calibrate_status.completed:
            self.progress.emit("Calibrar...")
            ok, msg, _ = run_calibrate(seq_path)
            # Calibrar pot fallar sense KHP, continuem

        # ANALYZE (si pendent)
        if not self.seq.analyze_status.completed:
            self.progress.emit("Analitzar...")
            ok, msg, _ = run_analyze(seq_path)
            if not ok:
                self.finished.emit(False, f"Analyze: {msg}")
                return

        self.finished.emit(True, "Completat")


class BatchWorker(QThread):
    """
    Worker per processar múltiples seqüències.

    EXECUCIÓ VERTICAL: per cada fase, processa TOTES les seqüències.
    Això permet veure el progrés per etapa i és més eficient.
    """
    progress = Signal(int, int, str)  # current, total, message
    seq_completed = Signal(str, bool, str)  # seq_name, success, message
    finished = Signal(int, int)  # success_count, fail_count

    def __init__(self, sequences, phases, default_uib_sensitivity=None):
        super().__init__()
        self.sequences = sequences
        self.phases = phases
        self.default_uib_sensitivity = default_uib_sensitivity
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True

    def run(self):
        """Execució VERTICAL: cada fase per totes les seqüències."""
        total_ok, total_fail = 0, 0
        n_seqs = len(self.sequences)

        # VERTICAL: per cada fase
        for phase in self.phases:
            if self._stop_requested:
                break

            if phase == Phase.IMPORT:
                phase_name = "Importar"
                # Usar funció que passa siblings
                def import_runner(seq):
                    siblings = seq.siblings if hasattr(seq, 'siblings') else []
                    return run_import(seq.seq_path, self.default_uib_sensitivity, siblings)
                runner = import_runner
            elif phase == Phase.CALIBRATE:
                phase_name = "Calibrar"
                runner = lambda seq: run_calibrate(seq.seq_path)
            elif phase == Phase.ANALYZE:
                phase_name = "Analitzar"
                runner = lambda seq: run_analyze(seq.seq_path)
            else:
                continue

            # Processar TOTES les seqüències per aquesta fase
            for i, seq in enumerate(self.sequences):
                if self._stop_requested:
                    break

                # Mostrar si és pack
                if phase == Phase.IMPORT and hasattr(seq, 'siblings') and seq.siblings:
                    display_name = f"{seq.seq_name} [pack {len(seq.siblings)+1}]"
                else:
                    display_name = seq.seq_name
                self.progress.emit(i + 1, n_seqs, f"{phase_name}: {display_name}")

                ok, msg, _ = runner(seq)
                self.seq_completed.emit(seq.seq_name, ok, msg)

                if ok:
                    total_ok += 1
                else:
                    total_fail += 1

        self.finished.emit(total_ok, total_fail)


class DashboardPanel(QWidget):
    """Dashboard - Vista general de seqüències."""

    sequence_selected = Signal(str, str)

    # Noms de les etapes
    STAGE_NAMES = ["Importar", "Calibrar", "Analitzar", "Consolidar"]

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.sequences = []
        self.filtered_sequences = []
        self.batch_worker = None
        self.single_worker = None

        self._setup_ui()
        self.refresh_sequences()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # === HEADER: Títol amb carpeta ===
        header = QHBoxLayout()
        header.setSpacing(16)

        cfg = get_config()
        data_folder = cfg.get("paths", "data_folder")

        self.lbl_title = QLabel(f"Seqüències - {data_folder}")
        self.lbl_title.setFont(QFont("Segoe UI", 14, QFont.Bold))
        header.addWidget(self.lbl_title)

        header.addStretch()
        layout.addLayout(header)

        # === FILA 1: Estadístiques + Botons ===
        stats_row = QHBoxLayout()
        stats_row.setSpacing(16)

        # Estadístiques per etapa
        self.lbl_stats = QLabel()
        self.lbl_stats.setFont(QFont("Segoe UI", 10))
        stats_row.addWidget(self.lbl_stats)

        stats_row.addStretch()

        # Botons
        self.refresh_btn = QPushButton("Actualitzar")
        self.refresh_btn.clicked.connect(self.refresh_sequences)
        stats_row.addWidget(self.refresh_btn)

        self.btn_batch = QPushButton("Processar Batch")
        self.btn_batch.clicked.connect(self._process_pending)
        stats_row.addWidget(self.btn_batch)

        self.btn_reset = QPushButton("Reset")
        self.btn_reset.clicked.connect(self._reset_sequences)
        self.btn_reset.setToolTip("Esborra dades processades (requereix contrasenya)")
        stats_row.addWidget(self.btn_reset)

        layout.addLayout(stats_row)

        # === FILA 2: Filtres ===
        filter_layout = QHBoxLayout()
        filter_layout.setSpacing(12)

        # Filtre Tipus (Column/BP)
        filter_layout.addWidget(QLabel("Tipus:"))
        self.filter_tipus = QComboBox()
        self.filter_tipus.addItems(["Tots", "Column", "BP"])
        self.filter_tipus.setMinimumWidth(80)
        self.filter_tipus.currentTextChanged.connect(self._apply_filter)
        filter_layout.addWidget(self.filter_tipus)

        # Filtre Mode (Dual/Direct/UIB)
        filter_layout.addWidget(QLabel("Mode:"))
        self.filter_mode = QComboBox()
        self.filter_mode.addItems(["Tots", "DUAL", "DIRECT", "UIB"])
        self.filter_mode.setMinimumWidth(80)
        self.filter_mode.currentTextChanged.connect(self._apply_filter)
        filter_layout.addWidget(self.filter_mode)

        # Filtre Estat
        filter_layout.addWidget(QLabel("Estat:"))
        self.filter_estat = QComboBox()
        self.filter_estat.addItems(["Tots", "Pendent", "En curs", "Complet", "Error"])
        self.filter_estat.setMinimumWidth(90)
        self.filter_estat.currentTextChanged.connect(self._apply_filter)
        filter_layout.addWidget(self.filter_estat)

        filter_layout.addSpacing(20)

        # Cerca
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Cercar...")
        self.search_edit.setMaximumWidth(150)
        self.search_edit.textChanged.connect(self._apply_filter)
        filter_layout.addWidget(self.search_edit)

        filter_layout.addStretch()

        layout.addLayout(filter_layout)

        # === TAULA DE SEQÜÈNCIES ===
        self.table = QTableWidget()
        self.table.setColumnCount(13)
        self.table.setHorizontalHeaderLabels([
            "#", "Seqüència", "Data", "Tipus", "Mode", "M", "PC", "PR",
            "Importar", "Calibrar", "Analitzar", "Consolidar", "Notes"
        ])

        # Tooltips per capçaleres
        self.table.horizontalHeaderItem(5).setToolTip("Mostres")
        self.table.horizontalHeaderItem(6).setToolTip("Patrons de Calibració (KHP)")
        self.table.horizontalHeaderItem(7).setToolTip("Patrons de Referència")
        self.table.horizontalHeaderItem(12).setToolTip("Doble-clic per afegir notes")

        # Configurar columnes - autoajust amb mínims per capçaleres
        h = self.table.horizontalHeader()

        # Primer: ResizeToContents per totes (ajusta a contingut)
        for i in range(self.table.columnCount() - 1):  # Totes menys Notes
            h.setSectionResizeMode(i, QHeaderView.ResizeToContents)

        # Notes expandeix per omplir espai restant
        h.setSectionResizeMode(12, QHeaderView.Stretch)

        # Mínims per assegurar que capçaleres es veuen
        self._header_min_widths = {
            0: 30,    # #
            1: 100,   # Seqüència
            2: 70,    # Data
            3: 55,    # Tipus
            4: 50,    # Mode
            5: 30,    # M
            6: 32,    # PC
            7: 32,    # PR
            8: 65,    # Importar
            9: 62,    # Calibrar
            10: 68,   # Analitzar
            11: 78,   # Consolidar
        }

        # Estil per mantenir colors dels punts en selecció
        self.table.setStyleSheet("""
            QTableWidget {
                gridline-color: #ddd;
            }
            QTableWidget::item:selected {
                background-color: #d5e8f7;
            }
            QTableWidget::item:hover {
                background-color: #ecf0f1;
            }
        """)

        # Permetre ordenació
        self.table.setSortingEnabled(True)
        self.table.sortByColumn(0, Qt.DescendingOrder)

        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(False)  # Treure ombrejat alternatiu
        self.table.cellDoubleClicked.connect(self._on_double_click)

        # Menú contextual (clic dret)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._show_context_menu)

        layout.addWidget(self.table)

    def refresh_sequences(self):
        cfg = get_config()
        data_folder = cfg.get("paths", "data_folder")
        self.sequences = get_all_sequences(data_folder)

        # Actualitzar títol amb carpeta
        self.lbl_title.setText(f"{len(self.sequences)} Seqüències - {data_folder}")

        self._apply_filter()
        self._update_stats()

    def _apply_filter(self):
        filter_tipus = self.filter_tipus.currentText()
        filter_mode = self.filter_mode.currentText()
        filter_estat = self.filter_estat.currentText()
        search_text = self.search_edit.text().lower()

        self.filtered_sequences = []

        for seq in self.sequences:
            # Filtre cerca
            if search_text and search_text not in seq.seq_name.lower():
                continue

            # Filtre tipus (Column/BP)
            if filter_tipus == "Column" and "_BP" in seq.seq_name.upper():
                continue
            elif filter_tipus == "BP" and "_BP" not in seq.seq_name.upper():
                continue

            # Filtre mode (Dual/Direct/UIB)
            if filter_mode != "Tots":
                if seq.data_mode.upper() != filter_mode.upper():
                    continue

            # Filtre estat
            if filter_estat == "Pendent" and seq.progress_pct > 0:
                continue
            elif filter_estat == "En curs" and (seq.progress_pct == 0 or seq.progress_pct == 100):
                continue
            elif filter_estat == "Complet" and seq.progress_pct < 100:
                continue
            elif filter_estat == "Error":
                # Filtrar per seqüències amb errors
                has_error = (
                    seq.import_state == 'error' or
                    seq.calibrate_state == 'error' or
                    seq.analyze_state == 'error' or
                    seq.review_state == 'error'
                )
                if not has_error:
                    continue

            self.filtered_sequences.append(seq)

        self._update_table()

    def _update_table(self):
        # Bloquejar signals i sorting mentre actualitzem
        self.table.blockSignals(True)
        self.table.setSortingEnabled(False)
        self.table.setRowCount(0)

        for idx, seq in enumerate(self.filtered_sequences, 1):
            row = self.table.rowCount()
            self.table.insertRow(row)

            # Col 0: # (per ordenar numèricament)
            item_num = SortableTableItem(str(idx))
            item_num.setData(Qt.UserRole, idx)
            item_num.setTextAlignment(Qt.AlignCenter)
            item_num.setFlags(item_num.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 0, item_num)

            # Col 1: Nom (amb indicador de siblings si n'hi ha)
            display_name = seq.seq_name
            if seq.siblings:
                display_name = f"{seq.seq_name} [+{len(seq.siblings)}]"
            item_name = QTableWidgetItem(display_name)
            item_name.setData(Qt.UserRole, seq.seq_path)
            item_name.setFlags(item_name.flags() & ~Qt.ItemIsEditable)
            # Tooltip amb detall de siblings
            if seq.siblings:
                sibling_names = [os.path.basename(s) for s in seq.siblings]
                item_name.setToolTip(f"Pack amb {len(seq.siblings)} siblings:\n" + "\n".join(sibling_names))
            self.table.setItem(row, 1, item_name)

            # Col 2: Data (amb valor ordenable)
            date_display = seq.seq_date if seq.seq_date else "-"
            item_date = SortableTableItem(date_display)
            item_date.setTextAlignment(Qt.AlignCenter)
            item_date.setForeground(QColor("#666"))
            item_date.setFlags(item_date.flags() & ~Qt.ItemIsEditable)
            # Guardar data en format ordenable (YYYYMMDD)
            if seq.seq_date and seq.seq_date != "-":
                try:
                    # Format DD/MM/YY -> YYYYMMDD
                    parts = seq.seq_date.split('/')
                    if len(parts) == 3:
                        year = int(parts[2])
                        year = 2000 + year if year < 100 else year
                        sort_val = year * 10000 + int(parts[1]) * 100 + int(parts[0])
                        item_date.setData(Qt.UserRole, sort_val)
                except:
                    item_date.setData(Qt.UserRole, 0)
            else:
                item_date.setData(Qt.UserRole, 0)
            self.table.setItem(row, 2, item_date)

            # Col 3: Tipus
            item_tipus = QTableWidgetItem(seq.method if seq.method else "-")
            item_tipus.setTextAlignment(Qt.AlignCenter)
            item_tipus.setForeground(QColor("#666"))
            item_tipus.setFlags(item_tipus.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 3, item_tipus)

            # Col 4: Mode
            item_mode = QTableWidgetItem(seq.data_mode if seq.data_mode else "-")
            item_mode.setTextAlignment(Qt.AlignCenter)
            item_mode.setForeground(QColor("#666"))
            item_mode.setFlags(item_mode.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 4, item_mode)

            # Col 5: M (Mostres)
            n_samples = seq.n_samples if seq.n_samples else 0
            item_m = SortableTableItem(str(n_samples))
            item_m.setData(Qt.UserRole, n_samples)
            item_m.setTextAlignment(Qt.AlignCenter)
            item_m.setFlags(item_m.flags() & ~Qt.ItemIsEditable)
            item_m.setToolTip(f"Mostres: {n_samples}")
            self.table.setItem(row, 5, item_m)

            # Col 6: PC (Patrons Calibració)
            n_khp = seq.n_khp if seq.n_khp else 0
            item_pc = SortableTableItem(str(n_khp))
            item_pc.setData(Qt.UserRole, n_khp)
            item_pc.setTextAlignment(Qt.AlignCenter)
            item_pc.setFlags(item_pc.flags() & ~Qt.ItemIsEditable)
            item_pc.setToolTip(f"Patrons de Calibració (KHP): {n_khp}")
            self.table.setItem(row, 6, item_pc)

            # Col 7: PR (Patrons Referència)
            n_pr = seq.n_pr if seq.n_pr else 0
            item_pr = SortableTableItem(str(n_pr))
            item_pr.setData(Qt.UserRole, n_pr)
            item_pr.setTextAlignment(Qt.AlignCenter)
            item_pr.setFlags(item_pr.flags() & ~Qt.ItemIsEditable)
            item_pr.setToolTip(f"Patrons de Referència: {n_pr}")
            self.table.setItem(row, 7, item_pr)

            # Col 8-11: Fases (Importar, Calibrar, Analitzar, Consolidar)
            phases_data = [
                (seq.import_status, seq.import_state, "Importar", seq.import_warnings),
                (seq.calibrate_status, seq.calibrate_state, "Calibrar", []),
                (seq.analyze_status, seq.analyze_state, "Analitzar", seq.analyze_warnings),
                (seq.review_status, seq.review_state, "Consolidar", []),
            ]

            current_phase_idx = None
            for i, (status, _, _, _) in enumerate(phases_data):
                if not status.completed:
                    current_phase_idx = i
                    break

            for col_offset, (status, state, phase_name, phase_warnings) in enumerate(phases_data):
                col = col_offset + 8
                item = QTableWidgetItem()
                item.setTextAlignment(Qt.AlignCenter)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)

                # Font pels indicadors
                font = item.font()
                font.setPointSize(11)
                item.setFont(font)

                # Determinar icona, color i tooltip segons estat
                if state == 'ok':
                    item.setText("✔")
                    item.setForeground(QColor(COLOR_OK))
                    timestamp = status.timestamp[:16] if status.timestamp else ""
                    tooltip = f"{phase_name}: Completat"
                    if timestamp:
                        tooltip += f"\n{timestamp}"
                elif state == 'warning':
                    item.setText("⚠")
                    item.setForeground(QColor(COLOR_WARNING))
                    if phase_name == "Calibrar":
                        tooltip = f"{phase_name}: KHP sibling ({seq.khp_source})"
                    elif phase_warnings:
                        tooltip = f"{phase_name}: Avisos\n" + "\n".join(phase_warnings[:3])
                        if len(phase_warnings) > 3:
                            tooltip += f"\n... i {len(phase_warnings)-3} més"
                    else:
                        tooltip = f"{phase_name}: Avisos"
                elif state == 'error':
                    item.setText("×")
                    item.setForeground(QColor(COLOR_ERROR))
                    if phase_name == "Importar":
                        tooltip = f"{phase_name}: Error MasterFile"
                    elif phase_name == "Calibrar" and not seq.has_khp:
                        tooltip = f"{phase_name}: Només històric!"
                    elif status.errors:
                        tooltip = f"{phase_name}: Error\n" + "\n".join(status.errors)
                    else:
                        tooltip = f"{phase_name}: Error"
                else:  # pending
                    item.setText("○")
                    if current_phase_idx == col_offset:
                        item.setForeground(QColor(COLOR_CURRENT))
                        tooltip = f"{phase_name}: Pendent (següent)"
                    else:
                        item.setForeground(QColor(COLOR_PENDING))
                        tooltip = f"{phase_name}: Pendent"

                item.setToolTip(tooltip)
                self.table.setItem(row, col, item)

            # Col 12: Notes (JSON + manuals, doble-clic per veure/editar)
            json_notes = self._load_json_notes(seq.seq_path)
            manual_notes = seq.notes if seq.notes else ""

            # Construir preview combinat
            preview_parts = []
            tooltip_parts = []
            has_anomaly = False
            has_warning = False

            # Notes dels JSON (tot: warnings, anomalies, notes)
            for jn in json_notes[:4]:  # Màxim 4 per preview
                stage = jn.get("stage", "?")
                ntype = jn.get("type", "")
                content = jn.get("content", "")[:35]

                # Prefix segons tipus
                if ntype == "ANOM":
                    prefix = "!"
                    has_anomaly = True
                elif ntype == "WARN":
                    prefix = "W"
                    has_warning = True
                elif ntype == "QUAL":
                    prefix = "Q"
                elif ntype == "NOTE":
                    prefix = "N"
                else:
                    prefix = ""

                preview_parts.append(f"[{stage}:{prefix}] {content}")
                tooltip_parts.append(f"[{stage}] ({ntype}) {jn.get('content', '')}")

            # Notes manuals
            if manual_notes:
                preview_parts.append(f"[MAN] {manual_notes[:25]}")
                tooltip_parts.append(f"[Manual] {manual_notes}")

            if preview_parts:
                preview = " | ".join(preview_parts)
                if len(preview) > 80:
                    preview = preview[:77] + "..."
                tooltip = "\n".join(tooltip_parts)
                # Colors segons gravetat
                if has_anomaly:
                    color = QColor("#C62828")  # Vermell per anomalies
                elif has_warning:
                    color = QColor("#E65100")  # Taronja per warnings
                else:
                    color = QColor("#1565C0")  # Blau per notes
            else:
                preview = ""
                tooltip = "Doble-clic per afegir notes"
                color = QColor("#999")

            item_notes = QTableWidgetItem(preview)
            item_notes.setToolTip(tooltip)
            item_notes.setForeground(color)
            item_notes.setFlags(item_notes.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 12, item_notes)

        # Reactivar sorting i signals
        self.table.setSortingEnabled(True)
        self.table.sortByColumn(0, Qt.DescendingOrder)
        self.table.blockSignals(False)

        # Aplicar mínims de capçalera
        self._apply_min_widths()

    def _apply_min_widths(self):
        """Aplica amplades mínimes per assegurar capçaleres visibles."""
        h = self.table.horizontalHeader()
        for col, min_width in self._header_min_widths.items():
            if h.sectionSize(col) < min_width:
                h.resizeSection(col, min_width)

    def _update_stats(self):
        """Actualitza estadístiques per etapa."""
        total = len(self.sequences)
        if total == 0:
            self.lbl_stats.setText("Cap seqüència")
            return

        imported = sum(1 for s in self.sequences if s.import_status.completed)
        calibrated = sum(1 for s in self.sequences if s.calibrate_status.completed)
        analyzed = sum(1 for s in self.sequences if s.analyze_status.completed)
        consolidated = sum(1 for s in self.sequences if s.review_status.completed)

        # Comptar errors
        errors = sum(1 for s in self.sequences if (
            s.import_state == 'error' or
            s.calibrate_state == 'error' or
            s.analyze_state == 'error'
        ))

        stats_text = (
            f"Importades: {imported}/{total} | "
            f"Calibrades: {calibrated}/{total} | "
            f"Analitzades: {analyzed}/{total} | "
            f"Consolidades: {consolidated}/{total}"
        )

        if errors > 0:
            stats_text += f" | Errors: {errors}"

        self.lbl_stats.setText(stats_text)

    def _show_context_menu(self, pos):
        """Mostra menú contextual amb opcions per la seqüència."""
        from PySide6.QtWidgets import QMenu

        row = self.table.rowAt(pos.y())
        if row < 0:
            return

        item_name = self.table.item(row, 1)
        if not item_name:
            return

        seq_path = item_name.data(Qt.UserRole)
        seq = None
        for s in self.filtered_sequences:
            if s.seq_path == seq_path:
                seq = s
                break

        if not seq:
            return

        menu = QMenu(self)

        # Opció: Processar (totes les etapes pendents)
        siblings = seq.siblings if hasattr(seq, 'siblings') else []
        if siblings:
            action_text = f"▶ Processar {seq.seq_name} [pack {len(siblings)+1}]"
        else:
            action_text = f"▶ Processar {seq.seq_name}"
        action_process = menu.addAction(action_text)
        action_process.triggered.connect(lambda: self._process_single(seq))

        menu.addSeparator()

        # Opcions individuals per etapa
        if not seq.import_status.completed:
            action_import = menu.addAction("  → Importar")
            action_import.triggered.connect(lambda: self._run_single_phase(seq, "import"))

        if seq.import_status.completed and not seq.calibrate_status.completed:
            action_cal = menu.addAction("  → Calibrar")
            action_cal.triggered.connect(lambda: self._run_single_phase(seq, "calibrate"))

        if seq.import_status.completed and not seq.analyze_status.completed:
            action_analyze = menu.addAction("  → Analitzar")
            action_analyze.triggered.connect(lambda: self._run_single_phase(seq, "analyze"))

        menu.addSeparator()

        # Obrir al wizard
        action_wizard = menu.addAction("Obrir al Wizard...")
        action_wizard.triggered.connect(lambda: self._open_in_wizard(seq))

        menu.exec(self.table.mapToGlobal(pos))

    def _run_single_phase(self, seq, phase_name):
        """Executa una sola fase per una seqüència."""
        self._set_controls_enabled(False)

        # Mostrar si és pack
        siblings = seq.siblings if hasattr(seq, 'siblings') else []
        if phase_name == "import" and siblings:
            status_msg = f"{seq.seq_name} [pack {len(siblings)+1}]: {phase_name}..."
        else:
            status_msg = f"{seq.seq_name}: {phase_name}..."
        self.main_window.set_status(status_msg)

        if phase_name == "import":
            ok, msg, _ = run_import(seq.seq_path, siblings=siblings)
        elif phase_name == "calibrate":
            ok, msg, _ = run_calibrate(seq.seq_path)
        elif phase_name == "analyze":
            ok, msg, _ = run_analyze(seq.seq_path)
        else:
            ok, msg = False, "Fase desconeguda"

        self._set_controls_enabled(True)
        self.main_window.set_status(
            f"{seq.seq_name}: {msg}" if ok else f"{seq.seq_name}: ERROR - {msg}",
            5000
        )
        self.refresh_sequences()

    def _on_double_click(self, row, col):
        """Doble-clic obre directament al wizard o edita notes."""
        item_name = self.table.item(row, 1)
        if not item_name:
            return

        seq_path = item_name.data(Qt.UserRole)
        seq = None
        for s in self.filtered_sequences:
            if s.seq_path == seq_path:
                seq = s
                break

        if not seq:
            return

        # Si és la columna Notes, obrir popup per editar
        if col == 12:
            self._edit_notes_popup(row, seq)
            return

        # Altrament, obrir al wizard
        self._open_in_wizard(seq)

    def _edit_notes_popup(self, row, seq: SequenceState):
        """Obre un diàleg per editar les notes i veure observacions dels JSON."""
        from PySide6.QtWidgets import (
            QDialog, QVBoxLayout, QHBoxLayout, QTextEdit, QDialogButtonBox,
            QLabel, QGroupBox, QScrollArea
        )

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Notes i Observacions - {seq.seq_name}")
        dialog.setMinimumSize(550, 400)

        layout = QVBoxLayout(dialog)

        # === SECCIÓ 1: Notes dels JSON (warnings confirmats) ===
        json_notes = self._load_json_notes(seq.seq_path)
        if json_notes:
            obs_group = QGroupBox("Observacions de processament")
            obs_layout = QVBoxLayout(obs_group)
            obs_layout.setSpacing(4)

            for note in json_notes:
                note_frame = QLabel()
                stage = note.get("stage", "?")
                reviewer = note.get("reviewer", "")
                content = note.get("content", "")
                date = note.get("date", "")[:10] if note.get("date") else ""

                html = f"<b>[{stage}]</b> "
                if reviewer:
                    html += f"<span style='color:#666;'>({reviewer} {date})</span><br>"
                html += f"<span style='color:#333;'>{content}</span>"
                note_frame.setText(html)
                note_frame.setWordWrap(True)
                note_frame.setStyleSheet("""
                    background-color: #FFF8E1;
                    border: 1px solid #FFE082;
                    border-radius: 4px;
                    padding: 8px;
                    margin: 2px;
                """)
                obs_layout.addWidget(note_frame)

            layout.addWidget(obs_group)
        else:
            no_obs = QLabel("<i style='color:#888;'>Sense observacions de processament</i>")
            layout.addWidget(no_obs)

        # === SECCIÓ 2: Notes manuals (editables) ===
        notes_group = QGroupBox("Notes manuals")
        notes_layout = QVBoxLayout(notes_group)

        text_edit = QTextEdit()
        text_edit.setPlaceholderText("Escriu notes sobre aquesta seqüència...")
        text_edit.setText(seq.notes if seq.notes else "")
        text_edit.setMinimumHeight(100)
        notes_layout.addWidget(text_edit)

        layout.addWidget(notes_group)

        # Botons
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec() == QDialog.Accepted:
            new_notes = text_edit.toPlainText().strip()
            if seq.save_notes(new_notes):
                # Actualitzar la cel·la de la taula
                self.table.blockSignals(True)
                item_notes = self.table.item(row, 12)
                if item_notes:
                    # Mostrar només primera línia o resum
                    preview = new_notes.split('\n')[0][:50]
                    if len(new_notes) > 50 or '\n' in new_notes:
                        preview += "..."
                    item_notes.setText(preview)
                self.table.blockSignals(False)
                self.main_window.set_status(f"Notes guardades: {seq.seq_name}", 3000)
            else:
                QMessageBox.warning(
                    self, "Error",
                    "No s'han pogut guardar les notes.\n"
                    "Cal importar la seqüència primer."
                )

    def _load_json_notes(self, seq_path: str) -> list:
        """Carrega TOT dels JSON: warnings, anomalies, notes."""
        import json
        from pathlib import Path

        notes = []
        data_path = Path(seq_path) / "CHECK" / "data"

        if not data_path.exists():
            return notes

        json_files = {
            "import_manifest.json": "IMP",
            "calibration_result.json": "CAL",
            "analysis_result.json": "ANA",
            "consolidation.json": "CON",
        }

        for filename, stage_name in json_files.items():
            json_file = data_path / filename
            if not json_file.exists():
                continue

            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 1. WARNINGS pendents
                warnings = data.get("warnings", [])
                if isinstance(warnings, list):
                    for w in warnings[:3]:  # Màxim 3 per etapa
                        if isinstance(w, str) and w.strip():
                            notes.append({
                                "stage": stage_name,
                                "type": "WARN",
                                "content": w[:80],
                            })
                        elif isinstance(w, dict):
                            msg = w.get("message", w.get("code", ""))
                            if msg:
                                notes.append({
                                    "stage": stage_name,
                                    "type": "WARN",
                                    "content": msg[:80],
                                })

                # 2. ANOMALIES (batman, timeout, etc.) - analysis_result
                if filename == "analysis_result.json":
                    samples = data.get("samples_analyzed", {})
                    batman_count = 0
                    timeout_count = 0
                    for sample_name, sample_data in samples.items():
                        if sample_data.get("batman_direct") or sample_data.get("batman_uib"):
                            batman_count += 1
                        if sample_data.get("has_timeout"):
                            timeout_count += 1

                    if batman_count > 0:
                        notes.append({
                            "stage": stage_name,
                            "type": "ANOM",
                            "content": f"BATMAN detectat en {batman_count} mostres",
                        })
                    if timeout_count > 0:
                        notes.append({
                            "stage": stage_name,
                            "type": "ANOM",
                            "content": f"TIMEOUT en {timeout_count} mostres",
                        })

                # 3. CALIBRACIÓ - problemes KHP
                if filename == "calibration_result.json":
                    cals = data.get("calibrations", [])
                    for cal in cals:
                        if cal.get("has_batman"):
                            notes.append({
                                "stage": stage_name,
                                "type": "ANOM",
                                "content": "KHP amb batman",
                            })
                        if cal.get("has_timeout"):
                            notes.append({
                                "stage": stage_name,
                                "type": "ANOM",
                                "content": "KHP amb timeout",
                            })
                        # Quality issues
                        for issue in cal.get("quality_issues", [])[:2]:
                            notes.append({
                                "stage": stage_name,
                                "type": "QUAL",
                                "content": issue[:60],
                            })

                # 4. NOTES D'USUARI (confirmació warnings)
                wc = data.get("warnings_confirmed")
                if isinstance(wc, dict):
                    user_note = wc.get("user_note", "")
                    if user_note:
                        notes.append({
                            "stage": stage_name,
                            "type": "NOTE",
                            "content": user_note[:80],
                            "reviewer": wc.get("reviewer", ""),
                        })

            except Exception as e:
                pass

        return notes

    def _open_in_wizard(self, seq: SequenceState):
        """Obre la seqüència al wizard per processar/revisar."""
        # El senyal sequence_selected és captat per main_window._on_sequence_selected
        # que ja fa load_sequence i navega al tab. No cal fer-ho aquí directament.
        self.sequence_selected.emit(seq.seq_path, seq.current_phase.value)

    def _process_single(self, seq: SequenceState):
        # Construir missatge amb info de siblings
        siblings = seq.siblings if hasattr(seq, 'siblings') else []
        if siblings:
            sibling_names = [os.path.basename(s) for s in siblings]
            sibling_info = f"\n\nPack amb {len(siblings)} siblings:\n• " + "\n• ".join(sibling_names)
        else:
            sibling_info = ""

        reply = QMessageBox.question(
            self, "Processar",
            f"Processar {seq.seq_name}?{sibling_info}\n\n"
            f"Executarà: {seq.next_action} i següents",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        self._set_controls_enabled(False)
        self.main_window.set_status(f"Processant {seq.seq_name}...")

        self.single_worker = SingleSeqWorker(seq)
        self.single_worker.progress.connect(
            lambda msg: self.main_window.set_status(f"{seq.seq_name}: {msg}")
        )
        self.single_worker.finished.connect(
            lambda ok, msg: self._on_single_finished(seq.seq_name, ok, msg)
        )
        self.single_worker.start()

    def _on_single_finished(self, seq_name, success, message):
        self._set_controls_enabled(True)
        self.main_window.set_status(
            f"{seq_name}: {message}" if success else f"{seq_name}: ERROR - {message}",
            5000
        )
        self.refresh_sequences()

    def _process_pending(self):
        """Processa seqüències en batch (requereix contrasenya)."""
        password, ok = QInputDialog.getText(
            self,
            "Autenticació requerida",
            "Processament batch.\n\nIntrodueix la contrasenya d'administrador:",
            QLineEdit.Password
        )

        if not ok:
            return

        if password != BATCH_PASSWORD:
            QMessageBox.warning(
                self, "Error",
                "Contrasenya incorrecta.\n\n"
                "El processament batch requereix permisos d'administrador."
            )
            return

        # Comptar seqüències per cada operació possible
        need_import = [s for s in self.sequences if not s.import_status.completed]
        need_calibrate = [s for s in self.sequences if s.import_status.completed and not s.calibrate_status.completed]
        need_analyze = [s for s in self.sequences if s.calibrate_status.completed and not s.analyze_status.completed]
        need_full = [s for s in self.sequences if s.progress_pct < 100]

        # Diàleg per triar operació
        options = []
        options.append(f"Només IMPORTAR ({len(need_import)} seqs)")
        options.append(f"Només CALIBRAR ({len(need_calibrate)} seqs)")
        options.append(f"Només ANALITZAR ({len(need_analyze)} seqs)")
        options.append(f"PIPELINE COMPLET ({len(need_full)} seqs)")

        choice, ok = QInputDialog.getItem(
            self,
            "Operació Batch",
            "Quina operació vols executar?",
            options,
            current=3,
            editable=False
        )

        if not ok:
            return

        # Determinar fases i seqüències segons selecció
        if "IMPORTAR" in choice:
            phases = [Phase.IMPORT]
            target_seqs = need_import
            op_name = "Importar"
        elif "CALIBRAR" in choice:
            phases = [Phase.CALIBRATE]
            target_seqs = need_calibrate
            op_name = "Calibrar"
        elif "ANALITZAR" in choice:
            phases = [Phase.ANALYZE]
            target_seqs = need_analyze
            op_name = "Analitzar"
        else:
            phases = [Phase.IMPORT, Phase.CALIBRATE, Phase.ANALYZE]
            target_seqs = need_full
            op_name = "Pipeline complet"

        if not target_seqs:
            QMessageBox.information(
                self, "Info",
                f"No hi ha seqüències pendents per: {op_name}"
            )
            return

        reply = QMessageBox.question(
            self, "Confirmar Batch",
            f"{op_name}: {len(target_seqs)} seqüències\n\n"
            "Vols continuar?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        # Verificar si alguna seqüència necessita sensibilitat UIB
        default_uib_sensitivity = None
        if Phase.IMPORT in phases:
            seqs_need_uib = self._get_seqs_needing_uib_sensitivity(target_seqs)
            if seqs_need_uib:
                from PySide6.QtWidgets import QInputDialog
                sens, ok = QInputDialog.getText(
                    self,
                    "Sensibilitat UIB",
                    f"{len(seqs_need_uib)} seqüències DUAL/UIB sense sensibilitat UIB definida.\n"
                    "Indica la sensibilitat UIB per defecte (ex: 700, 1000):\n\n"
                    "Seqüències: " + ", ".join([s.seq_name for s in seqs_need_uib[:5]]) +
                    ("..." if len(seqs_need_uib) > 5 else ""),
                    text="1000"
                )
                if ok and sens.strip():
                    default_uib_sensitivity = sens.strip()

        self._set_controls_enabled(False)

        self.batch_worker = BatchWorker(target_seqs, phases, default_uib_sensitivity)
        self.batch_worker.progress.connect(self._on_batch_progress)
        self.batch_worker.seq_completed.connect(self._on_seq_completed)
        self.batch_worker.finished.connect(self._on_batch_finished)
        self.batch_worker.start()

    def _reset_sequences(self):
        """Reset de seqüències (esborra CHECK/data/)."""
        password, ok = QInputDialog.getText(
            self,
            "Autenticació requerida",
            "ATENCIÓ: Això esborrarà les dades processades.\n\n"
            "Introdueix la contrasenya d'administrador:",
            QLineEdit.Password
        )

        if not ok:
            return

        if password != BATCH_PASSWORD:
            QMessageBox.warning(
                self, "Error",
                "Contrasenya incorrecta."
            )
            return

        # Triar tipus de reset
        options = [
            "Reset IMPORTACIÓ (només manifest.json)",
            "Reset COMPLET (tot CHECK/data/)"
        ]

        choice, ok = QInputDialog.getItem(
            self,
            "Tipus de Reset",
            "Quin tipus de reset vols fer?",
            options,
            current=0,
            editable=False
        )

        if not ok:
            return

        # Triar seqüències afectades
        seq_options = ["Totes les seqüències", "Només les filtrades"]
        seq_choice, ok = QInputDialog.getItem(
            self,
            "Seqüències afectades",
            "A quines seqüències?",
            seq_options,
            current=0,
            editable=False
        )

        if not ok:
            return

        target_seqs = self.sequences if "Totes" in seq_choice else self.filtered_sequences

        reply = QMessageBox.warning(
            self, "Confirmar Reset",
            f"Estàs segur?\n\n"
            f"Això afectarà {len(target_seqs)} seqüències.\n"
            f"Operació: {choice}",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        # Executar reset
        import shutil
        reset_full = "COMPLET" in choice
        count = 0

        for seq in target_seqs:
            check_data = os.path.join(seq.seq_path, "CHECK", "data")
            if os.path.exists(check_data):
                if reset_full:
                    # Esborrar tot CHECK/data/
                    try:
                        shutil.rmtree(check_data)
                        count += 1
                    except Exception as e:
                        print(f"Error esborrant {check_data}: {e}")
                else:
                    # Només esborrar manifest.json
                    manifest = os.path.join(check_data, "import_manifest.json")
                    if os.path.exists(manifest):
                        try:
                            os.remove(manifest)
                            count += 1
                        except Exception as e:
                            print(f"Error esborrant {manifest}: {e}")

        QMessageBox.information(
            self, "Reset completat",
            f"S'han resetejat {count} seqüències."
        )

        self.refresh_sequences()

    def _on_batch_progress(self, current, total, message):
        pct = int(100 * current / total) if total > 0 else 0
        self.main_window.show_progress(pct)
        self.main_window.set_status(f"[{current}/{total}] {message}")

    def _on_seq_completed(self, seq_name, success, message):
        """Actualitza la fila de la seqüència completada a la taula."""
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 1)
            if item and item.text() == seq_name:
                seq_path = item.data(Qt.UserRole)
                for seq in self.sequences:
                    if seq.seq_path == seq_path:
                        seq.refresh()
                        self._update_table_row(row, seq)
                        break
                break
        self._update_stats()

    def _update_table_row(self, row, seq: SequenceState):
        """Actualitza una sola fila de la taula amb l'estat actual de la seqüència."""
        # Actualitzar comptadors M, PC, PR (cols 5, 6, 7)
        n_samples = seq.n_samples if seq.n_samples else 0
        n_khp = seq.n_khp if seq.n_khp else 0
        n_pr = seq.n_pr if seq.n_pr else 0

        self.table.item(row, 5).setText(str(n_samples))
        self.table.item(row, 5).setData(Qt.UserRole, n_samples)
        self.table.item(row, 6).setText(str(n_khp))
        self.table.item(row, 6).setData(Qt.UserRole, n_khp)
        self.table.item(row, 7).setText(str(n_pr))
        self.table.item(row, 7).setData(Qt.UserRole, n_pr)

        phases_data = [
            (seq.import_status, seq.import_state, "Importar", seq.import_warnings),
            (seq.calibrate_status, seq.calibrate_state, "Calibrar", []),
            (seq.analyze_status, seq.analyze_state, "Analitzar", seq.analyze_warnings),
            (seq.review_status, seq.review_state, "Consolidar", []),
        ]

        current_phase_idx = None
        for i, (status, _, _, _) in enumerate(phases_data):
            if not status.completed:
                current_phase_idx = i
                break

        for col_offset, (status, state, phase_name, phase_warnings) in enumerate(phases_data):
            col = col_offset + 8
            item = QTableWidgetItem()
            item.setTextAlignment(Qt.AlignCenter)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)

            font = item.font()
            font.setPointSize(11)
            item.setFont(font)

            if state == 'ok':
                item.setText("✔")
                item.setForeground(QColor(COLOR_OK))
                item.setToolTip(f"{phase_name}: Completat")
            elif state == 'warning':
                item.setText("⚠")
                item.setForeground(QColor(COLOR_WARNING))
                warns = phase_warnings[:3] if phase_warnings else []
                tooltip = f"{phase_name}: Avisos"
                if warns:
                    tooltip += "\n" + "\n".join(warns)
                item.setToolTip(tooltip)
            elif state == 'error':
                item.setText("×")
                item.setForeground(QColor(COLOR_ERROR))
                if phase_name == "Importar":
                    item.setToolTip(f"{phase_name}: Error MasterFile")
                else:
                    item.setToolTip(f"{phase_name}: Error")
            elif col_offset == current_phase_idx:
                item.setText("○")
                item.setForeground(QColor(COLOR_CURRENT))
                item.setToolTip(f"{phase_name}: En curs...")
            else:
                item.setText("○")
                item.setForeground(QColor(COLOR_PENDING))
                item.setToolTip(f"{phase_name}: Pendent")

            self.table.setItem(row, col, item)

        # Actualitzar notes (col 12) - preview
        notes_text = seq.notes if seq.notes else ""
        if notes_text:
            preview = notes_text.split('\n')[0][:50]
            if len(notes_text) > 50 or '\n' in notes_text:
                preview += "..."
        else:
            preview = ""
        current_notes = self.table.item(row, 12)
        if current_notes:
            current_notes.setText(preview)
            current_notes.setToolTip(notes_text if notes_text else "Doble-clic per afegir notes")

    def _on_batch_finished(self, success, fail):
        self.main_window.show_progress(-1)
        self._set_controls_enabled(True)
        self.refresh_sequences()

        QMessageBox.information(
            self, "Completat",
            f"Correctes: {success}\nErrors: {fail}"
        )

    def _get_seqs_needing_uib_sensitivity(self, sequences):
        """
        Retorna les seqüències DUAL/UIB que no tenen sensibilitat UIB definida.
        Només verifica seqüències que encara no han estat importades.
        """
        need_uib = []
        for seq in sequences:
            # Només seqüències pendents d'importar
            if seq.import_status.completed:
                continue

            # Detectar mode pel nom del directori o estimació
            # Les seqüències entre 269-274 són DUAL (100µL)
            # Les seqüències >= 275 poden ser DUAL (400µL)
            try:
                seq_num = int(seq.seq_name.rstrip("ABCDEF_SEQ").rstrip("_BP"))
            except ValueError:
                continue

            # Heurística: seqüències modernes (>=269) poden ser DUAL
            if seq_num >= 269:
                # Verificar si té MasterFile amb sensibilitat UIB
                seq_path = Path(seq.seq_path)
                master_files = list(seq_path.glob("*MasterFile*.xlsx"))
                if master_files:
                    try:
                        import openpyxl
                        wb = openpyxl.load_workbook(master_files[0], read_only=True, data_only=True)
                        if "0-INFO" in wb.sheetnames:
                            ws = wb["0-INFO"]
                            uib_sens = ws["B5"].value
                            if uib_sens:
                                continue  # Ja té sensibilitat definida
                    except Exception:
                        pass

                # Si arribem aquí, potencialment necessita sensibilitat UIB
                need_uib.append(seq)

        return need_uib

    def _set_controls_enabled(self, enabled):
        self.refresh_btn.setEnabled(enabled)
        self.btn_batch.setEnabled(enabled)
        self.btn_reset.setEnabled(enabled)
        self.filter_tipus.setEnabled(enabled)
        self.filter_mode.setEnabled(enabled)
        self.filter_estat.setEnabled(enabled)
        self.search_edit.setEnabled(enabled)
        self.table.setEnabled(enabled)
