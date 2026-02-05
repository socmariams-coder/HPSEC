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
    QTabWidget, QFrame, QMessageBox, QSizePolicy,
    QDialog, QLineEdit, QTextEdit, QCheckBox, QDialogButtonBox
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont

from gui.widgets.styles import (
    COLOR_SUCCESS, COLOR_WARNING, COLOR_ERROR, COLOR_PRIMARY
)


class WarningReviewDialog(QDialog):
    """Diàleg per revisar avisos: afegir nota i/o marcar com a OK."""

    def __init__(self, parent, last_reviewer=""):
        super().__init__(parent)
        self.setWindowTitle("Revisar Avisos")
        self.setMinimumWidth(400)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Qui revisa
        layout.addWidget(QLabel("Nom o inicials de qui revisa:"))
        self.reviewer_input = QLineEdit(last_reviewer)
        self.reviewer_input.setPlaceholderText("Ex: MGA, Joan, etc.")
        layout.addWidget(self.reviewer_input)

        # Nota opcional
        layout.addWidget(QLabel("Nota (opcional):"))
        self.note_input = QTextEdit()
        self.note_input.setPlaceholderText("Afegeix comentaris sobre la revisió...")
        self.note_input.setMaximumHeight(80)
        layout.addWidget(self.note_input)

        # Checkbox: marcar com a OK
        self.mark_ok_checkbox = QCheckBox("Marcar com a revisat (passar a OK)")
        self.mark_ok_checkbox.setChecked(True)
        self.mark_ok_checkbox.setToolTip(
            "Si desmarca, s'afegeix la nota però l'avís queda pendent"
        )
        layout.addWidget(self.mark_ok_checkbox)

        # Botons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _validate_and_accept(self):
        """Valida que hi hagi nom de revisor."""
        if not self.reviewer_input.text().strip():
            QMessageBox.warning(self, "Falta informació", "Cal indicar qui revisa.")
            self.reviewer_input.setFocus()
            return
        self.accept()

    def get_result(self) -> dict:
        """Retorna el resultat del diàleg."""
        return {
            "reviewer": self.reviewer_input.text().strip(),
            "note": self.note_input.toPlainText().strip(),
            "mark_as_ok": self.mark_ok_checkbox.isChecked(),
        }

from pathlib import Path
import sys
import os

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gui.widgets.import_panel import ImportPanel
from gui.widgets.calibrate_panel import CalibratePanel
from gui.widgets.analyze_panel import AnalyzePanel
from gui.widgets.consolidate_panel import ConsolidatePanel


# Colors per estat (importats de styles.py)
COLOR_OK = COLOR_SUCCESS
COLOR_PENDING = "#95A5A6"
COLOR_CURRENT = COLOR_PRIMARY


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
        """Crea header amb botó tornar, nom SEQ, indicador de tasques i accions."""
        frame = QFrame()
        frame.setFixedHeight(44)
        frame.setStyleSheet("background-color: #f8f9fa; border-bottom: 1px solid #dee2e6;")

        layout = QHBoxLayout(frame)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        # Botó tornar al Dashboard
        self.back_btn = QPushButton("←")
        self.back_btn.setFixedSize(32, 32)
        self.back_btn.setToolTip("Tornar al Dashboard")
        self.back_btn.setStyleSheet("""
            QPushButton {
                background-color: #e9ecef; border: none; border-radius: 4px;
                font-weight: bold; font-size: 14px;
            }
            QPushButton:hover { background-color: #dee2e6; }
        """)
        self.back_btn.setCursor(Qt.PointingHandCursor)
        self.back_btn.clicked.connect(self._go_to_dashboard)
        layout.addWidget(self.back_btn)

        # Nom SEQ
        self.seq_label = QLabel("")
        self.seq_label.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.seq_label.setStyleSheet("color: #2E86AB;")
        layout.addWidget(self.seq_label)

        # Info addicional (method/mode)
        self.seq_info = QLabel()
        self.seq_info.setStyleSheet("color: #666; font-size: 10px;")
        layout.addWidget(self.seq_info)

        layout.addStretch()

        # === INDICADOR DE TASQUES ===
        self.task_indicator = QLabel("○ Pendent")
        self.task_indicator.setStyleSheet("""
            QLabel {
                background-color: #fff3cd; color: #856404;
                padding: 4px 10px; border-radius: 10px;
                font-size: 11px; font-weight: bold;
            }
        """)
        layout.addWidget(self.task_indicator)

        # Botó confirmar avisos (només visible quan hi ha warnings)
        self.confirm_warnings_btn = QPushButton("✓ Confirmar avisos")
        self.confirm_warnings_btn.setStyleSheet("""
            QPushButton {
                background-color: #F39C12; color: white; border: none;
                border-radius: 4px; padding: 6px 12px; font-weight: bold;
            }
            QPushButton:hover { background-color: #E67E22; }
        """)

        self.confirm_warnings_btn.setToolTip("Marca els avisos com a revisats i continua")
        self.confirm_warnings_btn.clicked.connect(self._on_confirm_warnings)
        self.confirm_warnings_btn.setVisible(False)
        layout.addWidget(self.confirm_warnings_btn)

        # Botó revertir confirmació (només visible quan avisos confirmats)
        self.revert_warnings_btn = QPushButton("↩ Revisar de nou")
        self.revert_warnings_btn.setStyleSheet("""
            QPushButton {
                background-color: #95A5A6; color: white; border: none;
                border-radius: 4px; padding: 6px 12px; font-weight: bold;
            }
            QPushButton:hover { background-color: #7F8C8D; }
        """)
        self.revert_warnings_btn.setToolTip("Torna a mostrar els avisos per revisar-los")
        self.revert_warnings_btn.clicked.connect(self._on_revert_warnings)
        self.revert_warnings_btn.setVisible(False)
        layout.addWidget(self.revert_warnings_btn)

        # Botó afegir nota (visible quan etapa completada sense warnings)
        self.add_note_btn = QPushButton("📝 Afegir nota")
        self.add_note_btn.setStyleSheet("""
            QPushButton {
                background-color: #6C757D; color: white; border: none;
                border-radius: 4px; padding: 6px 12px; font-weight: bold;
            }
            QPushButton:hover { background-color: #5A6268; }
        """)
        self.add_note_btn.setToolTip("Afegir una nota o comentari a aquesta etapa")
        self.add_note_btn.clicked.connect(self._on_add_note)
        self.add_note_btn.setVisible(False)
        layout.addWidget(self.add_note_btn)

        layout.addSpacing(10)

        # === BOTONS D'ACCIÓ ===
        btn_style = """
            QPushButton {
                background-color: #3498DB; color: white; border: none;
                border-radius: 4px; padding: 6px 14px; font-weight: bold;
            }
            QPushButton:hover { background-color: #2980B9; }
            QPushButton:disabled { background-color: #BDC3C7; color: #7F8C8D; }
        """

        self.action_btn = QPushButton("Executar")
        self.action_btn.setStyleSheet(btn_style)
        self.action_btn.clicked.connect(self._on_action_clicked)
        layout.addWidget(self.action_btn)

        self.next_step_btn = QPushButton("Següent →")
        self.next_step_btn.setStyleSheet(btn_style.replace("#3498DB", "#27AE60").replace("#2980B9", "#1E8449"))
        self.next_step_btn.setEnabled(False)
        self.next_step_btn.clicked.connect(self._go_next_step)
        layout.addWidget(self.next_step_btn)

        return frame

    def _on_action_clicked(self):
        """Executa l'acció del panell actual."""
        current_idx = self.tab_widget.currentIndex()
        state = self.tab_states[current_idx]
        force_redo = state in ("ok", "warning")  # Reprocessar si ja estava fet

        # Si és reprocessament, comprovar si invalida etapes posteriors
        if force_redo:
            if not self._confirm_reprocess(current_idx):
                return  # Usuari ha cancel·lat

        if current_idx == 0:  # Importar
            self.import_panel._run_import(force_reimport=force_redo)
        elif current_idx == 1:  # Calibrar
            if hasattr(self.calibrate_panel, '_run_calibrate'):
                self.calibrate_panel._run_calibrate()
        elif current_idx == 2:  # Analitzar
            if hasattr(self.analyze_panel, '_run_analyze'):
                self.analyze_panel._run_analyze()
        elif current_idx == 3:  # Consolidar
            if hasattr(self.consolidate_panel, '_search_bp'):
                self.consolidate_panel._search_bp()

    def _confirm_reprocess(self, current_idx: int) -> bool:
        """
        Confirma el reprocessament si hi ha etapes posteriors completades.
        Retorna True si l'usuari confirma, False si cancel·la.
        """
        # Comprovar si hi ha etapes posteriors completades
        later_completed = []
        stage_names = {0: "Importar", 1: "Calibrar", 2: "Analitzar", 3: "Consolidar"}

        for i in range(current_idx + 1, 4):
            if self.tab_states[i] in ("ok", "warning"):
                later_completed.append(stage_names[i])

        # Si no hi ha etapes posteriors completades, continuar sense preguntar
        if not later_completed:
            return True

        # Construir missatge d'avís
        current_name = stage_names[current_idx]
        stages_list = ", ".join(later_completed)

        reply = QMessageBox.warning(
            self,
            f"Reprocessar {current_name}",
            f"Si reprocesses '{current_name}', les etapes posteriors "
            f"s'invalidaran i caldrà tornar-les a executar:\n\n"
            f"  • {stages_list}\n\n"
            f"Vols continuar?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            # Invalidar etapes posteriors
            self._invalidate_later_stages(current_idx)
            return True

        return False

    def _invalidate_later_stages(self, from_idx: int):
        """Marca les etapes posteriors com a pendents i neteja les dades."""
        for i in range(from_idx + 1, 4):
            if self.tab_states[i] in ("ok", "warning"):
                self.tab_states[i] = "pending"

        # Actualitzar títols de pestanyes
        self._update_tab_titles()

        # Netejar dades cached dels panels posteriors i main_window
        if from_idx < 1:  # Si reimportem, netejar calibració
            self.calibrate_panel.calibration_data = None
            self.main_window.calibration_data = None
            # Reset UI del panel calibrar
            if hasattr(self.calibrate_panel, 'summary_group'):
                self.calibrate_panel.summary_group.setVisible(False)
            if hasattr(self.calibrate_panel, 'next_btn'):
                self.calibrate_panel.next_btn.setEnabled(False)

        if from_idx < 2:  # Si reimportem o recalibrem, netejar anàlisi
            if hasattr(self.analyze_panel, 'samples_grouped'):
                self.analyze_panel.samples_grouped = {}
            self.main_window.processed_data = None
            # Reset UI del panel analitzar
            if hasattr(self.analyze_panel, 'results_frame'):
                self.analyze_panel.results_frame.setVisible(False)
            if hasattr(self.analyze_panel, 'status_frame'):
                self.analyze_panel.status_frame.setVisible(True)
            if hasattr(self.analyze_panel, 'next_btn'):
                self.analyze_panel.next_btn.setEnabled(False)

        if from_idx < 3:  # Si reimportem, recalibrem o reanalitzem, netejar consolidació
            if hasattr(self.consolidate_panel, 'bp_data'):
                self.consolidate_panel.bp_data = {}
            self.main_window.review_data = None
            self.main_window.review_completed = False
            # Reset UI del panel consolidar
            if hasattr(self.consolidate_panel, 'bp_result_frame'):
                self.consolidate_panel.bp_result_frame.setVisible(False)
            if hasattr(self.consolidate_panel, 'next_btn'):
                self.consolidate_panel.next_btn.setEnabled(False)

    def _go_next_step(self):
        """Avança al següent pas del wizard."""
        current_idx = self.tab_widget.currentIndex()
        if current_idx < 3:
            self.tab_widget.setCurrentIndex(current_idx + 1)

    def _on_confirm_warnings(self):
        """Obre diàleg per revisar avisos: afegir nota i/o marcar com a OK."""
        current_idx = self.tab_widget.currentIndex()

        if self.tab_states[current_idx] != "warning":
            return

        # Mostrar diàleg de revisió
        dialog = WarningReviewDialog(
            self,
            last_reviewer=getattr(self, '_last_reviewer', "")
        )

        if dialog.exec():
            result = dialog.get_result()
            reviewer = result["reviewer"]
            note = result["note"]
            mark_as_ok = result["mark_as_ok"]

            self._last_reviewer = reviewer

            # Guardar al JSON
            self._save_warnings_confirmation(current_idx, reviewer, note, mark_as_ok)

            if mark_as_ok:
                self._set_tab_state(current_idx, "ok")
                self.main_window.set_status(f"Avisos confirmats per {reviewer}", 2000)

                # Avançar a següent etapa pendent
                next_pending = next(
                    (i for i in range(current_idx + 1, 4)
                     if self.tab_states[i] in ("pending", "warning", "current")),
                    None
                )
                if next_pending is not None:
                    self.tab_widget.setCurrentIndex(next_pending)
            else:
                self.main_window.set_status(f"Nota afegida per {reviewer} (warning pendent)", 2000)
                # Actualitzar header per mostrar que hi ha notes
                self._update_header_for_tab(current_idx)

    def _on_revert_warnings(self):
        """Reverteix la confirmació d'avisos (torna a warning)."""
        current_idx = self.tab_widget.currentIndex()

        if self._revert_warnings_confirmation(current_idx):
            self._set_tab_state(current_idx, "warning")
            self.main_window.set_status("Confirmació revertida - revisar avisos", 2000)

    def _on_add_note(self):
        """Obre diàleg per afegir una nota a l'etapa actual (sense warnings)."""
        current_idx = self.tab_widget.currentIndex()

        # Només permetre si l'etapa està completada (ok)
        if self.tab_states[current_idx] != "ok":
            return

        # Diàleg simplificat per afegir nota
        dialog = QDialog(self)
        dialog.setWindowTitle("Afegir Nota")
        dialog.setMinimumWidth(400)

        layout = QVBoxLayout(dialog)
        layout.setSpacing(12)

        # Qui afegeix la nota
        layout.addWidget(QLabel("Nom o inicials:"))
        reviewer_input = QLineEdit(getattr(self, '_last_reviewer', ""))
        reviewer_input.setPlaceholderText("Ex: MGA, Joan, etc.")
        layout.addWidget(reviewer_input)

        # Nota
        layout.addWidget(QLabel("Nota:"))
        note_input = QTextEdit()
        note_input.setPlaceholderText("Escriu el teu comentari...")
        note_input.setMaximumHeight(100)
        layout.addWidget(note_input)

        # Botons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec():
            reviewer = reviewer_input.text().strip()
            note = note_input.toPlainText().strip()

            if not reviewer:
                QMessageBox.warning(self, "Falta informació", "Cal indicar qui afegeix la nota.")
                return

            if not note:
                QMessageBox.warning(self, "Falta informació", "Cal escriure una nota.")
                return

            self._last_reviewer = reviewer

            # Guardar la nota (sense marcar com a warning)
            self._save_note(current_idx, reviewer, note)
            self.main_window.set_status(f"Nota afegida per {reviewer}", 2000)

    def _save_note(self, stage_idx: int, reviewer: str, note: str):
        """Guarda una nota al JSON corresponent."""
        from datetime import datetime
        import json

        seq_path = self.main_window.seq_path
        if not seq_path:
            return

        data_path = Path(seq_path) / "CHECK" / "data"

        json_files = {
            0: "import_manifest.json",
            1: "calibration_result.json",
            2: "analysis_result.json",
            3: "consolidation.json",
        }

        filename = json_files.get(stage_idx)
        if not filename:
            return

        json_file = data_path / filename
        if not json_file.exists():
            return

        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Afegir nota a l'historial
            note_entry = {
                "timestamp": datetime.now().isoformat(),
                "reviewer": reviewer,
                "note": note,
            }

            if "user_notes" not in data:
                data["user_notes"] = []
            data["user_notes"].append(note_entry)

            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)

        except Exception as e:
            QMessageBox.warning(self, "Error", f"No s'ha pogut guardar la nota: {e}")

    def _save_warnings_confirmation(self, stage_idx: int, reviewer: str, user_note: str = "", mark_as_ok: bool = True):
        """Guarda la revisió d'avisos al JSON corresponent."""
        from datetime import datetime
        import json

        seq_path = self.main_window.seq_path
        if not seq_path:
            return

        data_path = Path(seq_path) / "CHECK" / "data"

        # Determinar fitxer i camps de warnings segons etapa
        json_files = {
            0: ("import_manifest.json", ["warnings", "orphan_files"]),
            1: ("calibration_result.json", ["warnings", "khp_warnings"]),
            2: ("analysis_result.json", ["warnings", "anomalies"]),
            3: ("consolidation.json", ["warnings"]),
        }

        filename, warning_fields = json_files.get(stage_idx, ("", []))
        json_file = data_path / filename
        if not json_file.exists():
            return

        try:
            # Llegir JSON existent
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Recollir els avisos actuals
            warnings_notes = self._collect_warnings(data, warning_fields, stage_idx)

            # Crear entrada de revisió
            review_entry = {
                "timestamp": datetime.now().isoformat(),
                "reviewer": reviewer,
                "user_note": user_note,
                "auto_notes": warnings_notes,
                "marked_as_ok": mark_as_ok,
            }

            # Si marca com OK, guardar a warnings_confirmed
            if mark_as_ok:
                data["warnings_confirmed"] = review_entry
            else:
                # Si NO marca com OK, afegir a historial de revisions (manté warning)
                if "warnings_reviews" not in data:
                    data["warnings_reviews"] = []
                data["warnings_reviews"].append(review_entry)

            # Guardar
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)

            action = "confirmat" if mark_as_ok else "revisat (nota afegida)"
            print(f"[INFO] Avís {action} a {json_file.name} per {reviewer}")

        except Exception as e:
            print(f"[WARNING] No s'ha pogut guardar revisió: {e}")

    def _collect_warnings(self, data: dict, warning_fields: list, stage_idx: int) -> list:
        """Recull els avisos del JSON en format llegible per guardar com a notes."""
        notes = []
        stage_names = {0: "Importar", 1: "Calibrar", 2: "Analitzar", 3: "Consolidar"}

        for field in warning_fields:
            value = data.get(field)
            if not value:
                continue

            if isinstance(value, list) and len(value) > 0:
                # Llista de warnings (strings)
                for w in value:
                    if isinstance(w, str):
                        notes.append(f"[{field}] {w}")
                    elif isinstance(w, dict):
                        # Warning estructurat
                        msg = w.get("message") or w.get("msg") or str(w)
                        notes.append(f"[{field}] {msg}")

            elif isinstance(value, dict):
                # Dict de warnings (com orphan_files)
                for key, items in value.items():
                    if items and isinstance(items, list) and len(items) > 0:
                        notes.append(f"[{field}.{key}] {len(items)} elements: {', '.join(str(i) for i in items[:5])}")
                        if len(items) > 5:
                            notes[-1] += f"... (+{len(items)-5} més)"

        if not notes:
            notes.append(f"Etapa {stage_names.get(stage_idx, stage_idx)} revisada sense avisos específics")

        return notes

    def _revert_warnings_confirmation(self, stage_idx: int):
        """Reverteix la confirmació d'avisos (torna a warning)."""
        from datetime import datetime
        import json

        seq_path = self.main_window.seq_path
        if not seq_path:
            return False

        data_path = Path(seq_path) / "CHECK" / "data"

        json_files = {
            0: "import_manifest.json",
            1: "calibration_result.json",
            2: "analysis_result.json",
            3: "consolidation.json",
        }

        json_file = data_path / json_files.get(stage_idx, "")
        if not json_file.exists():
            return False

        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Eliminar confirmació
            if "warnings_confirmed" in data:
                del data["warnings_confirmed"]

                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False, default=str)

                return True

        except Exception as e:
            print(f"[WARNING] No s'ha pogut revertir confirmació: {e}")

        return False

    def _update_header_for_tab(self, index):
        """Actualitza el header segons la pestanya activa."""
        tab_names = {0: "Importar", 1: "Calibrar", 2: "Analitzar", 3: "Consolidar"}
        base_name = tab_names.get(index, "Executar")

        state = self.tab_states[index]

        # Botó d'acció: "Reprocessar" si ja fet, sinó el nom de l'acció
        if state in ("ok", "warning"):
            self.action_btn.setText(f"↻ Re{base_name.lower()}")
            self.action_btn.setToolTip(f"Tornar a executar {base_name.lower()}")
        else:
            self.action_btn.setText(base_name)
            self.action_btn.setToolTip(f"Executar {base_name.lower()}")

        # Comprovar si hi ha avisos confirmats (per mostrar botó revertir)
        has_confirmed_warnings = self._has_confirmed_warnings(index)

        # Indicador de tasques i botons confirmar/revertir avisos
        if state == "ok":
            if has_confirmed_warnings:
                self.task_indicator.setText("✓ Avisos revisats")
            else:
                self.task_indicator.setText("✓ Completat")
            self.task_indicator.setStyleSheet("""
                QLabel {
                    background-color: #d4edda; color: #155724;
                    padding: 4px 10px; border-radius: 10px;
                    font-size: 11px; font-weight: bold;
                }
            """)
            self.confirm_warnings_btn.setVisible(False)
            self.revert_warnings_btn.setVisible(has_confirmed_warnings)
            self.add_note_btn.setVisible(True)  # Permetre afegir notes
            self.next_step_btn.setEnabled(index < 3)
        elif state == "warning":
            self.task_indicator.setText("⚠ Revisar avisos")
            self.task_indicator.setStyleSheet("""
                QLabel {
                    background-color: #fff3cd; color: #856404;
                    padding: 4px 10px; border-radius: 10px;
                    font-size: 11px; font-weight: bold;
                }
            """)
            self.confirm_warnings_btn.setVisible(True)
            self.revert_warnings_btn.setVisible(False)
            self.add_note_btn.setVisible(False)
            self.next_step_btn.setEnabled(False)  # No permetre avançar fins confirmar
        elif state == "error":
            self.task_indicator.setText("✗ Error")
            self.task_indicator.setStyleSheet("""
                QLabel {
                    background-color: #f8d7da; color: #721c24;
                    padding: 4px 10px; border-radius: 10px;
                    font-size: 11px; font-weight: bold;
                }
            """)
            self.confirm_warnings_btn.setVisible(False)
            self.revert_warnings_btn.setVisible(False)
            self.add_note_btn.setVisible(False)
            self.next_step_btn.setEnabled(False)
        else:  # pending/current
            self.task_indicator.setText(f"○ {base_name}")
            self.task_indicator.setStyleSheet("""
                QLabel {
                    background-color: #e2e3e5; color: #383d41;
                    padding: 4px 10px; border-radius: 10px;
                    font-size: 11px; font-weight: bold;
                }
            """)
            self.confirm_warnings_btn.setVisible(False)
            self.revert_warnings_btn.setVisible(False)
            self.add_note_btn.setVisible(False)
            self.next_step_btn.setEnabled(False)

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
            # Actualitzar header si és la pestanya actual
            if self.tab_widget.currentIndex() == index:
                self._update_header_for_tab(index)

    def _hide_panel_navigation(self):
        """Amaga botons de navegació i acció dels panels (els botons són al header del wizard)."""
        for panel in [self.import_panel, self.calibrate_panel,
                      self.analyze_panel, self.consolidate_panel]:
            if hasattr(panel, 'next_btn'):
                panel.next_btn.setVisible(False)
            if hasattr(panel, 'prev_btn'):
                panel.prev_btn.setVisible(False)
            # Amagar botons d'acció específics dels panels
            if hasattr(panel, 'calibrate_btn'):
                panel.calibrate_btn.setVisible(False)
            if hasattr(panel, 'import_btn'):
                panel.import_btn.setVisible(False)
            if hasattr(panel, 'analyze_btn'):
                panel.analyze_btn.setVisible(False)
            if hasattr(panel, 'search_btn'):
                panel.search_btn.setVisible(False)

    def _connect_panel_signals(self):
        """Connecta senyals dels panels."""
        self.import_panel.import_completed.connect(self._on_import_completed)
        self.calibrate_panel.calibration_completed.connect(self._on_calibrate_completed)
        self.analyze_panel.analyze_completed.connect(self._on_analyze_completed)
        self.consolidate_panel.review_completed.connect(self._on_review_completed)

    def _go_to_dashboard(self):
        """Torna al Dashboard."""
        self.main_window.tab_widget.setCurrentIndex(0)

    def _load_sequence(self, seq_path: str):
        """Carrega una seqüència i detecta etapes completades."""
        if not os.path.isdir(seq_path):
            QMessageBox.warning(self, "Error", f"Carpeta no vàlida:\n{seq_path}")
            return

        seq_name = os.path.basename(seq_path)

        # IMPORTANT: Reset tots els panels abans de carregar nova SEQ
        self._reset_all_panels()

        # Actualitzar header
        self.seq_label.setText(seq_name)

        # Detectar method/mode si hi ha manifest
        self._update_seq_info(seq_path)

        # Notificar main_window
        self.main_window.seq_path = seq_path

        # Detectar etapes completades
        self.tab_states = self._detect_completed_stages(seq_path)
        self._update_tab_titles()

        # Anar a primera etapa que necessita atenció (warning o pending)
        first_needs_attention = next(
            (i for i, s in enumerate(self.tab_states) if s in ("warning", "pending", "current")),
            0
        )
        self.tab_widget.setCurrentIndex(first_needs_attention)
        self._update_header_for_tab(first_needs_attention)

        # Carregar al panel d'import
        self.import_panel.load_from_dashboard(seq_path)

        self.sequence_loaded.emit(seq_path)

    def _reset_all_panels(self):
        """Reseteja tots els panels quan es carrega una nova SEQ."""
        # Reset main_window data
        self.main_window.imported_data = None
        self.main_window.calibration_data = None
        self.main_window.processed_data = None
        self.main_window.review_data = None
        self.main_window.review_completed = False

        # Reset Calibrate panel
        self.calibrate_panel.calibration_data = None
        self.calibrate_panel._all_calibrations = []
        self.calibrate_panel._current_condition_key = None
        if hasattr(self.calibrate_panel, 'summary_group'):
            self.calibrate_panel.summary_group.setVisible(False)
        if hasattr(self.calibrate_panel, 'graphs_group'):
            self.calibrate_panel.graphs_group.setVisible(False)
        if hasattr(self.calibrate_panel, 'metrics_group'):
            self.calibrate_panel.metrics_group.setVisible(False)
        if hasattr(self.calibrate_panel, 'history_group'):
            self.calibrate_panel.history_group.setVisible(False)
        if hasattr(self.calibrate_panel, 'validation_group'):
            self.calibrate_panel.validation_group.setVisible(False)
        if hasattr(self.calibrate_panel, 'condition_selector_frame'):
            self.calibrate_panel.condition_selector_frame.setVisible(False)
        if hasattr(self.calibrate_panel, 'next_btn'):
            self.calibrate_panel.next_btn.setEnabled(False)

        # Reset Analyze panel
        if hasattr(self.analyze_panel, 'analysis_data'):
            self.analyze_panel.analysis_data = None
        if hasattr(self.analyze_panel, 'results_group'):
            self.analyze_panel.results_group.setVisible(False)
        if hasattr(self.analyze_panel, 'next_btn'):
            self.analyze_panel.next_btn.setEnabled(False)

        # Reset Consolidate panel
        if hasattr(self.consolidate_panel, 'consolidate_data'):
            self.consolidate_panel.consolidate_data = None
        if hasattr(self.consolidate_panel, 'results_frame'):
            self.consolidate_panel.results_frame.setVisible(False)

    def _detect_completed_stages(self, seq_path: str) -> list:
        """Detecta quines etapes estan completades basant-se en fitxers existents."""
        import json
        from pathlib import Path

        states = ["pending", "pending", "pending", "pending"]

        try:
            data_path = Path(seq_path) / "CHECK" / "data"
            if not data_path.exists():
                return states

            json_files = {
                0: ("import_manifest.json", ["warnings", "orphan_files"]),
                1: ("calibration_result.json", ["warnings", "khp_warnings"]),
                2: ("analysis_result.json", ["warnings", "anomalies"]),
                3: ("consolidation.json", ["warnings"]),
            }

            for idx, (filename, warning_fields) in json_files.items():
                json_path = data_path / filename
                if json_path.exists() and json_path.is_file():
                    try:
                        with open(json_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)

                        # Comprovar si hi ha warnings
                        has_warnings = self._check_has_warnings(data, warning_fields)

                        # Comprovar si els warnings estan confirmats
                        warnings_confirmed = data.get("warnings_confirmed") is not None

                        if has_warnings and not warnings_confirmed:
                            states[idx] = "warning"
                        else:
                            states[idx] = "ok"

                    except Exception as e:
                        print(f"[WARNING] Error llegint {filename}: {e}")
                        states[idx] = "ok"  # Assumir ok si no podem llegir

        except Exception as e:
            print(f"[WARNING] Error detectant etapes: {e}")

        # Marcar primera etapa pendent com a "current"
        for i, state in enumerate(states):
            if state == "pending":
                states[i] = "current"
                break

        return states

    def _check_has_warnings(self, data: dict, warning_fields: list) -> bool:
        """Comprova si les dades tenen warnings significatius."""
        for field in warning_fields:
            value = data.get(field)
            if value:
                # Si és una llista, comprovar si té elements
                if isinstance(value, list) and len(value) > 0:
                    return True
                # Si és un dict (com orphan_files), comprovar si té contingut
                elif isinstance(value, dict):
                    for v in value.values():
                        if v and (isinstance(v, list) and len(v) > 0):
                            return True
        return False

    def _has_confirmed_warnings(self, stage_idx: int) -> bool:
        """Comprova si l'etapa té avisos confirmats al JSON."""
        import json

        try:
            seq_path = self.main_window.seq_path
            if not seq_path:
                return False

            data_path = Path(seq_path) / "CHECK" / "data"
            if not data_path.exists():
                return False

            json_files = {
                0: "import_manifest.json",
                1: "calibration_result.json",
                2: "analysis_result.json",
                3: "consolidation.json",
            }

            filename = json_files.get(stage_idx)
            if not filename:
                return False

            json_file = data_path / filename
            if not json_file.exists() or not json_file.is_file():
                return False

            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get("warnings_confirmed") is not None
        except Exception as e:
            print(f"[WARNING] Error checking confirmed warnings: {e}")
            return False

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
        # Actualitzar header amb botons i indicador
        self._update_header_for_tab(index)

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
            self._set_tab_state(3, "pending")
            # NO saltar automàticament a Consolidar - deixar que l'usuari vegi els resultats
            self._update_header_for_tab(2)
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
