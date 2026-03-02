# -*- coding: utf-8 -*-
"""
HPSEC Suite - Process Wizard Panel v2.0
========================================

Panel per processar seqüències amb pestanyes per cada fase:
1. Importar - Llegir dades RAW
2. Verificar - QA/QC KHP i diagnòstic delay
3. Analitzar - Detectar anomalies i calcular àrees
4. Revisar - Revisió de qualitat i generació de resultats

Estructura visual optimitzada:
- Header mínim amb nom SEQ i botó tornar
- Pestanyes per cada fase (màxim espai per contingut)
- Icones d'estat a les pestanyes (✓/⚠/○)
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTabWidget, QFrame, QMessageBox, QSizePolicy, QScrollArea,
    QDialog, QLineEdit, QTextEdit, QCheckBox, QDialogButtonBox, QMenu
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont

import logging

from gui.widgets.styles import (
    COLOR_SUCCESS, COLOR_WARNING, COLOR_ERROR, COLOR_PRIMARY
)

logger = logging.getLogger(__name__)


class WarningBarWidget(QFrame):
    """Barra d'avisos persistent al wizard.

    Mostra avisos NOMÉS de la fase activa (no acumulats entre fases).
    Desplegada per defecte si ≤3 avisos, plegada si >3.
    """

    SEVERITY_COLORS = {
        "blocker": ("#D32F2F", "#FFEBEE", "#FFCDD2"),
        "warning": ("#F57C00", "#FFF8E1", "#FFE082"),
        "info": ("#1976D2", "#E3F2FD", "#BBDEFB"),
    }
    SEVERITY_ICONS = {
        "blocker": "\u26d4",
        "warning": "\u26a0",
        "info": "\u2139",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._warnings = []
        self._expanded = True
        self._setup_ui()
        self.setVisible(False)

    def _setup_ui(self):
        self._main_layout = QVBoxLayout(self)
        self._main_layout.setContentsMargins(12, 6, 12, 6)
        self._main_layout.setSpacing(4)

        header_row = QHBoxLayout()
        header_row.setSpacing(8)

        self._summary_label = QLabel()
        self._summary_label.setStyleSheet("font-weight: bold; font-size: 11px;")
        header_row.addWidget(self._summary_label, 1)

        self._toggle_btn = QPushButton("\u25b2")
        self._toggle_btn.setFixedSize(24, 24)
        self._toggle_btn.setStyleSheet(
            "QPushButton { background: transparent; border: none; font-size: 11px; "
            "color: #666; font-weight: bold; } QPushButton:hover { color: #333; }"
        )
        self._toggle_btn.setCursor(Qt.PointingHandCursor)
        self._toggle_btn.clicked.connect(self._toggle_expand)
        header_row.addWidget(self._toggle_btn)

        self._main_layout.addLayout(header_row)

        self._detail_widget = QWidget()
        self._detail_layout = QVBoxLayout(self._detail_widget)
        self._detail_layout.setContentsMargins(0, 2, 0, 0)
        self._detail_layout.setSpacing(2)
        self._main_layout.addWidget(self._detail_widget)

    def _toggle_expand(self):
        self._expanded = not self._expanded
        self._detail_widget.setVisible(self._expanded)
        self._toggle_btn.setText("\u25b2" if self._expanded else "\u25bc")

    def update_warnings(self, warnings: list):
        """Actualitza la barra amb avisos de la fase activa."""
        self._warnings = warnings

        while self._detail_layout.count():
            item = self._detail_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not warnings:
            self.setVisible(False)
            return

        max_sev = "info"
        for w in warnings:
            sev = w.get("severity", w.get("level", "info"))
            if sev == "blocker":
                max_sev = "blocker"
                break
            elif sev == "warning":
                max_sev = "warning"

        text_color, bg_color, border_color = self.SEVERITY_COLORS.get(
            max_sev, self.SEVERITY_COLORS["info"]
        )
        self.setStyleSheet(
            f"QFrame {{ background-color: {bg_color}; "
            f"border: 1px solid {border_color}; border-radius: 4px; }}"
        )

        icon = self.SEVERITY_ICONS.get(max_sev, "\u2022")
        n = len(warnings)
        self._summary_label.setText(
            f'{icon} {n} av{"is" if n == 1 else "isos"}'
        )
        self._summary_label.setStyleSheet(
            f"font-weight: bold; font-size: 11px; color: {text_color}; border: none;"
        )
        self._toggle_btn.setStyleSheet(
            f"QPushButton {{ background: transparent; border: none; "
            f"font-size: 11px; color: {text_color}; font-weight: bold; }}"
        )

        for w in warnings[:10]:
            sev = w.get("severity", w.get("level", "info"))
            w_icon = self.SEVERITY_ICONS.get(sev, "\u2022")
            w_color = self.SEVERITY_COLORS.get(sev, self.SEVERITY_COLORS["info"])[0]
            message = w.get("message", w.get("label", w.get("code", "Avis")))
            action = w.get("action", "")

            text = f'{w_icon} <span style="color:{w_color}">{message}</span>'
            if action:
                text += f'<br><span style="color:#666; font-size:10px; margin-left:18px;">   \u2192 {action}</span>'

            label = QLabel(text)
            label.setWordWrap(True)
            label.setStyleSheet("border: none; padding: 1px 0;")
            self._detail_layout.addWidget(label)

        if n > 10:
            more = QLabel(f"... i {n - 10} mes")
            more.setStyleSheet("color: #666; font-style: italic; border: none;")
            self._detail_layout.addWidget(more)

        self._expanded = n <= 3
        self._detail_widget.setVisible(self._expanded)
        self._toggle_btn.setText("\u25b2" if self._expanded else "\u25bc")

        self.setVisible(True)


class WarningSkipDialog(QDialog):
    """Diàleg per saltar avisos (no bloquejants) amb nota obligatòria."""

    def __init__(self, parent, warning_level="warning", last_reviewer=""):
        super().__init__(parent)
        self.setWindowTitle("Continuar amb Avisos Pendents")
        self.setMinimumWidth(450)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Avís
        if warning_level == "warning":
            icon = "⚠"
            color = "#856404"
            bg = "#fff3cd"
            msg = "Hi ha avisos pendents de revisar."
        else:  # info
            icon = "ℹ"
            color = "#004085"
            bg = "#cce5ff"
            msg = "Hi ha informació disponible."

        warning_frame = QFrame()
        warning_frame.setStyleSheet(f"background-color: {bg}; border-radius: 6px; padding: 8px;")
        warning_layout = QHBoxLayout(warning_frame)

        icon_label = QLabel(icon)
        icon_label.setStyleSheet(f"font-size: 24px; color: {color};")
        warning_layout.addWidget(icon_label)

        msg_label = QLabel(f"<b>{msg}</b><br>Pots continuar afegint una nota explicativa.")
        msg_label.setStyleSheet(f"color: {color};")
        msg_label.setWordWrap(True)
        warning_layout.addWidget(msg_label, 1)

        layout.addWidget(warning_frame)

        # Qui revisa
        layout.addWidget(QLabel("Nom o inicials:"))
        self.reviewer_input = QLineEdit(last_reviewer)
        self.reviewer_input.setPlaceholderText("Ex: MGA, Joan, etc.")
        layout.addWidget(self.reviewer_input)

        # Nota obligatòria
        layout.addWidget(QLabel("Nota explicativa (obligatòria):"))
        self.note_input = QTextEdit()
        self.note_input.setPlaceholderText(
            "Explica per què continues sense revisar els avisos...\n"
            "Ex: 'Avisos de smoothness no rellevants per BP'"
        )
        self.note_input.setMaximumHeight(100)
        layout.addWidget(self.note_input)

        # Botons
        buttons = QDialogButtonBox()
        self.continue_btn = buttons.addButton("Continuar \u2192", QDialogButtonBox.AcceptRole)
        self.continue_btn.setStyleSheet("""
            QPushButton {
                background-color: #F39C12; color: white; border: none;
                border-radius: 4px; padding: 8px 16px; font-weight: bold;
            }
            QPushButton:hover { background-color: #E67E22; }
        """)
        cancel_btn = buttons.addButton("Cancel\u00b7lar", QDialogButtonBox.RejectRole)
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _validate_and_accept(self):
        """Valida que hi hagi nom i nota."""
        if not self.reviewer_input.text().strip():
            QMessageBox.warning(self, "Falta informació", "Cal indicar qui ets.")
            self.reviewer_input.setFocus()
            return

        if not self.note_input.toPlainText().strip():
            QMessageBox.warning(
                self, "Falta informació",
                "Cal afegir una nota explicativa per continuar sense revisar els avisos."
            )
            self.note_input.setFocus()
            return

        self.accept()

    def get_result(self) -> dict:
        """Retorna el resultat del diàleg."""
        return {
            "reviewer": self.reviewer_input.text().strip(),
            "note": self.note_input.toPlainText().strip(),
        }


from pathlib import Path
import sys
import os

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gui.widgets.import_panel import ImportPanel
from gui.widgets.calibrate_panel import CalibratePanel
from gui.widgets.analyze_panel import AnalyzePanel
from gui.widgets.export_panel import ExportPanel
from gui.widgets.review_summary_panel import ReviewSummaryPanel


class ProcessWizardPanel(QWidget):
    """
    Panel per processar seqüències amb pestanyes.

    Cada fase té la seva pestanya amb tot l'espai disponible.
    """

    process_completed = Signal(dict)
    sequence_loaded = Signal(str)

    TAB_NAMES = ["1. Importar", "2. Verificar", "3. Analitzar", "4. Revisar"]
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

        # === BARRA D'AVISOS (entre header i pestanyes) ===
        self._warning_bar = WarningBarWidget()
        layout.addWidget(self._warning_bar)

        # === PESTANYES ===
        self.tab_widget = QTabWidget()
        self.tab_widget.setDocumentMode(True)
        self.tab_widget.setTabPosition(QTabWidget.North)

        # Crear panels
        self.import_panel = ImportPanel(self.main_window)
        self.calibrate_panel = CalibratePanel(self.main_window)
        self.analyze_panel = AnalyzePanel(self.main_window)
        self.export_panel = ExportPanel(self.main_window)  # Kept for independent use
        self.review_panel = ReviewSummaryPanel(self.main_window)

        # Afegir pestanyes
        self.tab_widget.addTab(self.import_panel, self._tab_title(0))
        self.tab_widget.addTab(self.calibrate_panel, self._tab_title(1))
        self.tab_widget.addTab(self.analyze_panel, self._tab_title(2))
        self.tab_widget.addTab(self.review_panel, self._tab_title(3))

        # Amagar botons de navegació dels panels (innecessaris amb pestanyes)
        self._hide_panel_navigation()

        # Context menu al tab bar (clic dret per reset)
        self.tab_widget.tabBar().setContextMenuPolicy(Qt.CustomContextMenu)
        self.tab_widget.tabBar().customContextMenuRequested.connect(self._on_tab_context_menu)

        # Connectar senyals
        self._connect_panel_signals()
        self.tab_widget.currentChanged.connect(self._on_tab_changed)

        layout.addWidget(self.tab_widget)

    def _create_minimal_header(self) -> QFrame:
        """Crea header simplificat amb layout fix.

        Estructura:
        [←] SEQ_286_BP (info)  [status_indicator]  [📝 N]  [Acció]  [Següent →]

        Elements:
        - status_indicator: fusió de task_indicator + warnings_btn en un sol botó clicable
        - note_btn: sempre actiu, mostra comptador de notes si n'hi ha
        - action_btn: sempre visible (disabled quan no aplicable)
        - next_step_btn: amb tooltips contextuals
        """
        frame = QFrame()
        frame.setFixedHeight(48)
        frame.setStyleSheet("background-color: #f8f9fa; border-bottom: 1px solid #dee2e6;")

        layout = QHBoxLayout(frame)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(12)

        # === SECCIÓ CONTEXT ===
        self.back_btn = QPushButton("\u2190")
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

        self.seq_label = QLabel("")
        self.seq_label.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.seq_label.setStyleSheet("color: #2E86AB;")
        layout.addWidget(self.seq_label)

        self.seq_info = QLabel()
        self.seq_info.setStyleSheet("color: #666; font-size: 10px;")
        layout.addWidget(self.seq_info)

        layout.addStretch()

        # === STATUS INDICATOR (fusió task_indicator + warnings_btn) ===
        self.status_indicator = QPushButton("\u25CB Pendent")
        self.status_indicator.setMinimumWidth(120)
        self.status_indicator.setCursor(Qt.PointingHandCursor)
        self.status_indicator.clicked.connect(self._on_status_indicator_clicked)
        self._set_status_indicator_style("pending")
        layout.addWidget(self.status_indicator)

        layout.addSpacing(8)

        # === BOTÓ NOTES (sempre actiu, amb comptador) ===
        self.note_btn = QPushButton("\U0001f4dd")
        self.note_btn.setFixedWidth(40)
        self.note_btn.setStyleSheet("""
            QPushButton {
                background-color: #6C757D; color: white; border: none;
                border-radius: 4px; padding: 6px; font-size: 14px;
            }
            QPushButton:hover { background-color: #5A6268; }
        """)
        self.note_btn.setToolTip("Notes i comentaris")
        self.note_btn.clicked.connect(self._on_add_note)
        layout.addWidget(self.note_btn)

        layout.addSpacing(8)

        # === SECCIÓ NAVEGACIÓ ===
        self.action_btn = QPushButton("Executar")
        self.action_btn.setFixedWidth(110)
        self.action_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498DB; color: white; border: none;
                border-radius: 4px; padding: 6px 14px; font-weight: bold;
            }
            QPushButton:hover { background-color: #2980B9; }
            QPushButton:disabled { background-color: #BDC3C7; color: #7F8C8D; }
        """)
        self.action_btn.clicked.connect(self._on_action_clicked)
        layout.addWidget(self.action_btn)

        self.next_step_btn = QPushButton("Seg\u00fcent \u2192")
        self.next_step_btn.setFixedWidth(100)
        self.next_step_btn.setStyleSheet("""
            QPushButton {
                background-color: #27AE60; color: white; border: none;
                border-radius: 4px; padding: 6px 14px; font-weight: bold;
            }
            QPushButton:hover { background-color: #1E8449; }
            QPushButton:disabled { background-color: #BDC3C7; color: #7F8C8D; }
        """)
        self.next_step_btn.setEnabled(False)
        self.next_step_btn.clicked.connect(self._go_next_step)
        layout.addWidget(self.next_step_btn)

        # Estat intern
        self._current_warning_level = "none"
        self._warnings_confirmed_by = None

        return frame

    def _set_status_indicator_style(self, level: str):
        """Aplica estil al status_indicator segons el nivell."""
        styles = {
            "pending": """
                QPushButton {
                    background-color: #e2e3e5; color: #383d41; border: none;
                    border-radius: 12px; padding: 4px 14px;
                    font-size: 11px; font-weight: bold;
                }
                QPushButton:hover { background-color: #d6d8db; }
                QPushButton:disabled { background-color: #e2e3e5; color: #383d41; }
            """,
            "executing": """
                QPushButton {
                    background-color: #cce5ff; color: #004085; border: none;
                    border-radius: 12px; padding: 4px 14px;
                    font-size: 11px; font-weight: bold;
                }
                QPushButton:disabled { background-color: #cce5ff; color: #004085; }
            """,
            "ok": """
                QPushButton {
                    background-color: #d4edda; color: #155724; border: none;
                    border-radius: 12px; padding: 4px 14px;
                    font-size: 11px; font-weight: bold;
                }
                QPushButton:hover { background-color: #c3e6cb; }
            """,
            "confirmed": """
                QPushButton {
                    background-color: #27AE60; color: white; border: none;
                    border-radius: 12px; padding: 4px 14px;
                    font-size: 11px; font-weight: bold;
                }
                QPushButton:hover { background-color: #1E8449; }
            """,
            "info": """
                QPushButton {
                    background-color: #cce5ff; color: #004085; border: none;
                    border-radius: 12px; padding: 4px 14px;
                    font-size: 11px; font-weight: bold;
                }
                QPushButton:hover { background-color: #b8daff; }
            """,
            "warning": """
                QPushButton {
                    background-color: #F39C12; color: white; border: none;
                    border-radius: 12px; padding: 4px 14px;
                    font-size: 11px; font-weight: bold;
                }
                QPushButton:hover { background-color: #E67E22; }
            """,
            "blocker": """
                QPushButton {
                    background-color: #E74C3C; color: white; border: none;
                    border-radius: 12px; padding: 4px 14px;
                    font-size: 11px; font-weight: bold;
                }
                QPushButton:hover { background-color: #C0392B; }
            """,
            "error": """
                QPushButton {
                    background-color: #f8d7da; color: #721c24; border: none;
                    border-radius: 12px; padding: 4px 14px;
                    font-size: 11px; font-weight: bold;
                }
                QPushButton:hover { background-color: #f5c6cb; }
            """,
        }
        self.status_indicator.setStyleSheet(styles.get(level, styles["pending"]))

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
            # Netejar dades pre-carregades perquè reimporti realment del MasterFile
            self.main_window.imported_data = None
            self.import_panel._run_import(force_reimport=force_redo)
        elif current_idx == 1:  # Calibrar
            if hasattr(self.calibrate_panel, '_run_calibrate'):
                self.calibrate_panel._run_calibrate()
        elif current_idx == 2:  # Analitzar
            if hasattr(self.analyze_panel, '_run_analyze'):
                self.analyze_panel._run_analyze()
        elif current_idx == 3:  # Revisar
            if hasattr(self.review_panel, '_run_generate'):
                self.review_panel._run_generate()

    def _confirm_reprocess(self, current_idx: int) -> bool:
        """
        Confirma el reprocessament si hi ha etapes posteriors completades.
        Retorna True si l'usuari confirma, False si cancel·la.
        """
        # Comprovar si hi ha etapes posteriors completades
        later_completed = []
        stage_names = {0: "Importar", 1: "Verificar", 2: "Analitzar", 3: "Revisar"}

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
            if hasattr(self.calibrate_panel, 'compact_header'):
                self.calibrate_panel.compact_header.setVisible(False)
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

        if from_idx < 3:  # Si reimportem, recalibrem o reanalitzem, netejar revisió
            if hasattr(self.review_panel, 'reset'):
                self.review_panel.reset()

    def _on_tab_context_menu(self, pos):
        """Context menu al clicar dret sobre un tab: permet reset des d'aquí."""
        tab_idx = self.tab_widget.tabBar().tabAt(pos)
        if tab_idx < 0:
            return

        stage_names = {0: "Importar", 1: "Verificar", 2: "Analitzar", 3: "Revisar"}
        menu = QMenu(self)
        action = menu.addAction(f"↺ Reset des de '{stage_names[tab_idx]}'")
        action.triggered.connect(lambda: self._reset_from_stage(tab_idx))
        menu.exec(self.tab_widget.tabBar().mapToGlobal(pos))

    def _reset_from_stage(self, stage_idx):
        """Reset des d'una etapa concreta (esborra JSONs + outputs cascade)."""
        from hpsec_reset import reset_stage, STAGE_NAMES

        seq_path = getattr(self.main_window, 'seq_path', None)
        if not seq_path:
            QMessageBox.warning(self, "Error", "Cap seqüència carregada.")
            return

        stages_affected = [STAGE_NAMES[s] for s in range(stage_idx, 4)]

        reply = QMessageBox.warning(
            self,
            f"Reset des de '{STAGE_NAMES[stage_idx]}'",
            f"Esborrarà dades de: {', '.join(stages_affected)}\n\n"
            "Vols continuar?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        result = reset_stage(seq_path, stage_idx)

        # Netejar dades al main_window
        if stage_idx <= 0:
            self.main_window.imported_data = None
        if stage_idx <= 1:
            self.main_window.calibration_data = None
        if stage_idx <= 2:
            self.main_window.processed_data = None
        if stage_idx <= 3:
            self.main_window.review_data = None
            self.main_window.review_completed = False

        # Reset panels UI (invalidar des d'una etapa abans per netejar tot)
        self._invalidate_later_stages(max(0, stage_idx - 1))

        # Marcar l'etapa reseteada com a pending
        for i in range(stage_idx, 4):
            self.tab_states[i] = "pending"

        # Marcar primera pendent com a current
        for i, state in enumerate(self.tab_states):
            if state == "pending":
                self.tab_states[i] = "current"
                break

        self._update_tab_titles()
        self._update_header_for_tab(self.tab_widget.currentIndex())

        n_deleted = len(result.get("deleted", []))
        self.main_window.set_status(
            f"Reset: {n_deleted} fitxers esborrats des de '{STAGE_NAMES[stage_idx]}'",
            5000
        )

    def _go_next_step(self):
        """Avança al següent pas del wizard i executa l'operació automàticament.

        Si hi ha avisos WARNING (no BLOCKER), demana confirmació amb nota.
        """
        current_idx = self.tab_widget.currentIndex()
        if current_idx >= 3:
            return

        state = self.tab_states[current_idx]
        warning_level = self._get_warning_level(current_idx)

        # Si hi ha avisos WARNING pendents, demanar nota abans d'avançar
        if state == "warning" and warning_level in ("warning", "info"):
            dialog = WarningSkipDialog(
                self,
                warning_level=warning_level,
                last_reviewer=getattr(self, '_last_reviewer', "")
            )

            if dialog.exec():
                result = dialog.get_result()
                reviewer = result["reviewer"]
                note = result["note"]

                self._last_reviewer = reviewer

                # Guardar la nota i marcar com "skip with note"
                self._save_warning_skip(current_idx, reviewer, note)

                # Marcar com OK (amb avisos reconeguts)
                self._set_tab_state(current_idx, "ok")

                # Avançar i executar
                self._advance_and_execute(current_idx + 1)
            # Si cancel·la, no avança
        else:
            # Sense avisos o ja confirmat, avança i executa
            self._advance_and_execute(current_idx + 1)

    def _advance_and_execute(self, next_idx: int):
        """Navega a la pestanya indicada i executa l'operació corresponent."""
        if next_idx > 3:
            return

        # Navegar a la pestanya
        self.tab_widget.setCurrentIndex(next_idx)

        # Executar l'operació de la nova etapa (amb petit delay per actualitzar UI)
        from PySide6.QtCore import QTimer
        QTimer.singleShot(100, lambda: self._execute_stage(next_idx))

    def _execute_stage(self, stage_idx: int):
        """Executa l'operaci\u00f3 de l'etapa indicada."""
        stage_names = {1: "Calibrant", 2: "Analitzant", 3: "Revisant"}
        stage_name = stage_names.get(stage_idx, "Executant")

        self._show_executing_state(stage_name)

        try:
            if stage_idx == 1:  # Calibrar
                if hasattr(self.calibrate_panel, '_run_calibrate'):
                    self.calibrate_panel._run_calibrate()
            elif stage_idx == 2:  # Analitzar
                if hasattr(self.analyze_panel, '_run_analyze'):
                    self.analyze_panel._run_analyze()
            elif stage_idx == 3:  # Revisar
                self._update_header_for_tab(stage_idx)
        except Exception as e:
            logger.error(f"_execute_stage({stage_idx}): {e}")
            self._set_tab_state(stage_idx, "error")
            self._update_header_for_tab(stage_idx)

    def _show_executing_state(self, stage_name: str):
        """Mostra l'estat d'execuci\u00f3 en curs."""
        self.status_indicator.setText(f"\u25CF {stage_name}...")
        self.status_indicator.setEnabled(False)
        self._set_status_indicator_style("executing")
        self.action_btn.setEnabled(False)
        self.next_step_btn.setEnabled(False)

    def _save_warning_skip(self, stage_idx: int, reviewer: str, note: str):
        """Guarda la nota de salt d'avisos."""
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

            # Guardar com a "skip amb nota"
            data["warnings_confirmed"] = {
                "timestamp": datetime.now().isoformat(),
                "reviewer": reviewer,
                "user_note": note,
                "skipped_with_note": True,
                "marked_as_ok": True,
            }

            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)

            self.main_window.set_status(f"Avisos reconeguts per {reviewer}", 2000)

        except Exception as e:
            logger.warning(f"No s'ha pogut guardar: {e}")

    def _on_confirm_warnings(self):
        """Marca avisos com a revisats (ja visibles a la barra)."""
        current_idx = self.tab_widget.currentIndex()
        if self.tab_states[current_idx] != "warning":
            return
        self._set_tab_state(current_idx, "ok")
        self.main_window.set_status("Avisos marcats com a revisats", 2000)
        self._update_warning_bar()

    def _on_revert_warnings(self):
        """Reverteix la confirmació d'avisos (torna a warning)."""
        current_idx = self.tab_widget.currentIndex()

        if self._revert_warnings_confirmation(current_idx):
            self._set_tab_state(current_idx, "warning")
            self.main_window.set_status("Confirmació revertida - revisar avisos", 2000)

    def _load_existing_notes(self, stage_idx=None):
        """Carrega totes les notes existents de totes les etapes."""
        import json

        seq_path = self.main_window.seq_path
        if not seq_path:
            return []

        data_path = Path(seq_path) / "CHECK" / "data"

        notes = []
        seen = set()  # Evitar duplicats (per timestamp+reviewer)

        # Buscar notes als JSONs de cada etapa
        json_files = {
            "import": "import_manifest.json",
            "calibrate": "calibration_result.json",
            "analyze": "analysis_result.json",
        }
        for stage, filename in json_files.items():
            json_file = data_path / filename
            if json_file.exists():
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    for n in data.get("user_notes", []):
                        key = (n.get("timestamp", ""), n.get("reviewer", ""))
                        if key not in seen:
                            if "stage" not in n:
                                n["stage"] = stage
                            notes.append(n)
                            seen.add(key)
                except Exception:
                    pass

        # Buscar notes al fitxer general
        notes_file = data_path / "user_notes.json"
        if notes_file.exists():
            try:
                with open(notes_file, 'r', encoding='utf-8') as f:
                    notes_data = json.load(f)
                for n in notes_data.get("notes", []):
                    key = (n.get("timestamp", ""), n.get("reviewer", ""))
                    if key not in seen:
                        notes.append(n)
                        seen.add(key)
            except Exception:
                pass

        # Ordenar per timestamp
        notes.sort(key=lambda n: n.get("timestamp", ""))
        return notes

    def _on_add_note(self):
        """Obre diàleg NO MODAL per veure notes existents i afegir-ne de noves.

        No-modal: permet fer scroll al panell de darrere mentre el diàleg és obert.
        """
        # Tancar diàleg anterior si existeix
        if hasattr(self, '_notes_dialog') and self._notes_dialog is not None:
            try:
                self._notes_dialog.close()
            except RuntimeError:
                pass
            self._notes_dialog = None

        current_idx = self.tab_widget.currentIndex()
        stage_names = {0: "Importar", 1: "Verificar", 2: "Analitzar", 3: "Revisar"}
        stage_name = stage_names.get(current_idx, "Etapa")

        # Carregar totes les notes
        existing_notes = self._load_existing_notes()
        stage_labels = {"import": "Importar", "calibrate": "Verificar",
                       "analyze": "Analitzar", "export": "Revisar"}

        dialog = QDialog(self)
        dialog.setWindowTitle("Notes")
        dialog.setMinimumWidth(550)
        dialog.setMinimumHeight(450)
        # No-modal: permet interactuar amb la finestra principal
        dialog.setModal(False)
        dialog.setAttribute(Qt.WA_DeleteOnClose)
        self._notes_dialog = dialog

        layout = QVBoxLayout(dialog)
        layout.setSpacing(8)

        # === Notes existents (totes les etapes) ===
        if existing_notes:
            layout.addWidget(QLabel(f"<b>Notes ({len(existing_notes)}):</b>"))

            notes_scroll = QScrollArea()
            notes_scroll.setWidgetResizable(True)
            notes_scroll.setMaximumHeight(250)
            notes_widget = QWidget()
            notes_layout = QVBoxLayout(notes_widget)
            notes_layout.setSpacing(6)

            for note in existing_notes:
                ts = note.get("timestamp", "")[:16].replace("T", " ")
                reviewer = note.get("reviewer", "?")
                text = note.get("note", "")
                stage = note.get("stage", "")
                stage_display = stage_labels.get(stage, stage)

                note_frame = QFrame()
                note_frame.setStyleSheet(
                    "QFrame { background: #F8F9FA; border: 1px solid #DEE2E6; "
                    "border-radius: 4px; padding: 6px; }"
                )
                note_fl = QVBoxLayout(note_frame)
                note_fl.setContentsMargins(6, 4, 6, 4)
                note_fl.setSpacing(2)

                header = QLabel(
                    f"<b>{reviewer}</b> · "
                    f"<span style='color:#2E86AB'>[{stage_display}]</span> · "
                    f"<span style='color:#888'>{ts}</span>"
                )
                header.setTextFormat(Qt.RichText)
                note_fl.addWidget(header)

                body = QLabel(text)
                body.setWordWrap(True)
                body.setTextInteractionFlags(Qt.TextSelectableByMouse)
                note_fl.addWidget(body)

                notes_layout.addWidget(note_frame)

            notes_layout.addStretch()
            notes_scroll.setWidget(notes_widget)
            layout.addWidget(notes_scroll)

            # Separador
            sep = QFrame()
            sep.setFrameShape(QFrame.HLine)
            sep.setStyleSheet("color: #DEE2E6;")
            layout.addWidget(sep)
        else:
            layout.addWidget(QLabel("<i>No hi ha notes anteriors.</i>"))

        # === Afegir nova nota ===
        layout.addWidget(QLabel(f"<b>Afegir nota a [{stage_name}]:</b>"))

        # Qui afegeix la nota
        reviewer_layout = QHBoxLayout()
        reviewer_layout.addWidget(QLabel("Revisor:"))
        reviewer_input = QLineEdit(getattr(self, '_last_reviewer', ""))
        reviewer_input.setPlaceholderText("Ex: MGA, Joan, etc.")
        reviewer_layout.addWidget(reviewer_input)
        layout.addLayout(reviewer_layout)

        # Nota
        note_input = QTextEdit()
        note_input.setPlaceholderText("Escriu el teu comentari...")
        note_input.setMinimumHeight(80)
        layout.addWidget(note_input, 1)

        # Botons
        btn_layout = QHBoxLayout()
        btn_close = QPushButton("Tancar")
        btn_close.clicked.connect(dialog.close)
        btn_layout.addWidget(btn_close)
        btn_layout.addStretch()
        btn_save = QPushButton("Guardar nota")
        btn_save.setStyleSheet(
            "QPushButton { background: #3498DB; color: white; border: none; "
            "border-radius: 4px; padding: 6px 16px; font-weight: bold; }"
            "QPushButton:hover { background: #2980B9; }"
        )
        btn_layout.addWidget(btn_save)
        layout.addLayout(btn_layout)

        def _save_and_close():
            reviewer = reviewer_input.text().strip()
            note = note_input.toPlainText().strip()
            if not reviewer:
                QMessageBox.warning(dialog, "Falta informació", "Cal indicar qui afegeix la nota.")
                return
            if not note:
                dialog.close()
                return
            self._last_reviewer = reviewer
            self._save_note(current_idx, reviewer, note)
            self.main_window.set_status(f"Nota afegida per {reviewer}", 2000)
            dialog.close()

        btn_save.clicked.connect(_save_and_close)

        dialog.show()

    def _save_note(self, stage_idx: int, reviewer: str, note: str):
        """Guarda una nota al JSON corresponent.

        SEMPRE funciona: si el JSON no existeix, crea un fitxer de notes separat.
        """
        from datetime import datetime
        import json

        seq_path = self.main_window.seq_path
        if not seq_path:
            QMessageBox.warning(self, "Avís", "No hi ha cap seqüència seleccionada.")
            return

        data_path = Path(seq_path) / "CHECK" / "data"
        data_path.mkdir(parents=True, exist_ok=True)

        json_files = {
            0: "import_manifest.json",
            1: "calibration_result.json",
            2: "analysis_result.json",
        }
        stage_names = {0: "import", 1: "calibrate", 2: "analyze", 3: "export"}

        filename = json_files.get(stage_idx)
        stage_name = stage_names.get(stage_idx, "unknown")

        note_entry = {
            "timestamp": datetime.now().isoformat(),
            "reviewer": reviewer,
            "note": note,
            "stage": stage_name,
        }

        try:
            json_file = data_path / filename if filename else None

            # Si el JSON de l'etapa existeix, afegir la nota allà
            if json_file and json_file.exists():
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if "user_notes" not in data:
                    data["user_notes"] = []
                data["user_notes"].append(note_entry)

                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False, default=str)

            else:
                # Si no existeix, guardar a un fitxer de notes general
                notes_file = data_path / "user_notes.json"
                if notes_file.exists():
                    with open(notes_file, 'r', encoding='utf-8') as f:
                        notes_data = json.load(f)
                else:
                    notes_data = {"notes": []}

                notes_data["notes"].append(note_entry)

                with open(notes_file, 'w', encoding='utf-8') as f:
                    json.dump(notes_data, f, indent=2, ensure_ascii=False, default=str)

            self.main_window.set_status(f"Nota guardada: {stage_name}", 2000)
            self._update_note_btn()

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
            logger.info(f"Avís {action} a {json_file.name} per {reviewer}")

        except Exception as e:
            logger.warning(f"No s'ha pogut guardar revisió: {e}")

    def _collect_warnings(self, data: dict, warning_fields: list, stage_idx: int) -> list:
        """Recull els avisos del JSON en format llegible per guardar com a notes."""
        notes = []
        stage_names = {0: "Importar", 1: "Verificar", 2: "Analitzar", 3: "Revisar"}

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
            logger.warning(f"No s'ha pogut revertir confirmació: {e}")

        return False

    def _update_header_for_tab(self, index):
        """Actualitza el header segons la pestanya activa.

        Layout ESTABLE: status_indicator + note_btn + action_btn + next_step_btn.
        Tots sempre visibles. Només canvien: enabled/disabled, text, color/estil, tooltip.
        """
        tab_names = {0: "Importar", 1: "Verificar", 2: "Analitzar", 3: "Revisar"}
        base_name = tab_names.get(index, "Executar")
        state = self.tab_states[index]
        has_confirmed = self._has_confirmed_warnings(index)
        warning_level = self._get_warning_level(index)

        # === STATUS INDICATOR (un sol botó per estat + avisos) ===
        if state in ("pending", "current"):
            self.status_indicator.setText("\u25CB Pendent")
            self.status_indicator.setEnabled(False)
            self.status_indicator.setToolTip("Executa primer l'etapa")
            self._set_status_indicator_style("pending")

        elif state == "ok":
            if has_confirmed:
                self.status_indicator.setText("\u2713 Revisat")
                self.status_indicator.setEnabled(True)
                self.status_indicator.setToolTip("Avisos revisats. Clic per revertir.")
                self._set_status_indicator_style("confirmed")
            else:
                self.status_indicator.setText("\u2713 OK")
                self.status_indicator.setEnabled(False)
                self.status_indicator.setToolTip("Tot correcte")
                self._set_status_indicator_style("ok")

        elif state == "warning":
            if warning_level == "blocker":
                self.status_indicator.setText("\u26D4 Errors")
                self.status_indicator.setEnabled(True)
                self.status_indicator.setToolTip("Hi ha errors que bloquegen. Clic per veure.")
                self._set_status_indicator_style("blocker")
            elif warning_level == "warning":
                self.status_indicator.setText("\u26A0 Revisar")
                self.status_indicator.setEnabled(True)
                self.status_indicator.setToolTip("Hi ha avisos pendents. Clic per revisar.")
                self._set_status_indicator_style("warning")
            else:  # info
                self.status_indicator.setText("\u2139 Info")
                self.status_indicator.setEnabled(True)
                self.status_indicator.setToolTip("Hi ha informaci\u00f3 disponible. Clic per veure.")
                self._set_status_indicator_style("info")

        elif state == "error":
            self.status_indicator.setText("\u2717 Error")
            self.status_indicator.setEnabled(True)
            self.status_indicator.setToolTip("Clic per veure detalls de l'error")
            self._set_status_indicator_style("error")

        # === BOTÓ ACCIÓ (sempre visible, disabled quan no aplicable) ===
        if state in ("ok", "warning", "error"):
            self.action_btn.setText("\u21BB Refer")
            self.action_btn.setToolTip(f"Tornar a executar {base_name.lower()}")
            self.action_btn.setEnabled(True)
        elif state in ("pending", "current") and index > 0:
            deps_ok = all(
                self.tab_states[i] in ("ok", "warning")
                for i in range(index)
            )
            self.action_btn.setText("\u25B6 Executar")
            self.action_btn.setToolTip(f"Executar {base_name.lower()}")
            self.action_btn.setEnabled(deps_ok)
        elif state in ("pending", "current") and index == 0:
            self.action_btn.setText("\u25B6 Executar")
            self.action_btn.setToolTip(f"Executar {base_name.lower()}")
            self.action_btn.setEnabled(False)
        else:
            self.action_btn.setText("Executar")
            self.action_btn.setEnabled(False)

        # === BOTÓ NOTES (comptador) ===
        self._update_note_btn()

        # === BOTÓ SEG\u00dcENT (amb tooltips contextuals) ===
        can_proceed = False
        tooltip = ""
        if index >= 3:
            tooltip = "\u00daltima etapa"
        elif state == "ok":
            can_proceed = True
            next_name = tab_names.get(index + 1, "")
            tooltip = f"Avan\u00e7ar a {next_name}"
        elif state == "warning" and warning_level != "blocker":
            can_proceed = True
            tooltip = "Avan\u00e7ar (es demanar\u00e0 nota)"
        elif state == "warning" and warning_level == "blocker":
            tooltip = "Cal resoldre els errors primer"
        elif state == "error":
            tooltip = "Cal corregir l'error primer"
        else:
            tooltip = "Cal executar l'etapa primer"

        self.next_step_btn.setEnabled(can_proceed)
        self.next_step_btn.setToolTip(tooltip)

    def _on_status_indicator_clicked(self):
        """Handler pel status_indicator unificat.

        Ja no obre diàleg d'avisos (la barra els mostra directament).
        Només gestiona revert confirmacions i errors.
        """
        current_idx = self.tab_widget.currentIndex()
        state = self.tab_states[current_idx]
        has_confirmed = self._has_confirmed_warnings(current_idx)

        if state == "ok" and has_confirmed:
            self._on_revert_warnings()
        elif state == "warning":
            # La barra ja mostra els avisos; si l'usuari clica, mark as OK
            self._set_tab_state(current_idx, "ok")
            self.main_window.set_status("Avisos marcats com a revisats", 2000)
            self._update_warning_bar()

        elif state == "error":
            self._show_error_details(current_idx)

    def _update_note_btn(self):
        """Actualitza el bot\u00f3 de notes amb comptador."""
        notes = self._load_existing_notes()
        n = len(notes)
        if n > 0:
            self.note_btn.setText(f"\U0001f4dd {n}")
            self.note_btn.setFixedWidth(55)
            self.note_btn.setToolTip(f"{n} nota{'s' if n > 1 else ''} - Clic per veure")
            self.note_btn.setStyleSheet("""
                QPushButton {
                    background-color: #2E86AB; color: white; border: none;
                    border-radius: 4px; padding: 6px; font-size: 13px; font-weight: bold;
                }
                QPushButton:hover { background-color: #236B8E; }
            """)
        else:
            self.note_btn.setText("\U0001f4dd")
            self.note_btn.setFixedWidth(40)
            self.note_btn.setToolTip("Afegir nota o comentari")
            self.note_btn.setStyleSheet("""
                QPushButton {
                    background-color: #6C757D; color: white; border: none;
                    border-radius: 4px; padding: 6px; font-size: 14px;
                }
                QPushButton:hover { background-color: #5A6268; }
            """)

    def _get_warning_level(self, stage_idx: int) -> str:
        """Determina el nivell màxim d'avisos per l'etapa.

        Sempre calcula des de warnings_structured (font única).

        Retorna: 'blocker', 'warning', 'info', o 'none'
        """
        try:
            warnings = self._get_warnings_list(stage_idx)
            if warnings:
                from hpsec_warnings import get_max_warning_level
                return get_max_warning_level(warnings)
            return "none"
        except Exception:
            if self.tab_states[stage_idx] == "warning":
                return "warning"
            elif self.tab_states[stage_idx] == "error":
                return "blocker"
            return "none"

    def _get_warnings_list(self, stage_idx: int) -> list:
        """Obté la llista d'avisos estructurats per l'etapa.

        Returns:
            Llista d'avisos amb format {"code", "severity", "message", ...}
        """
        try:
            if stage_idx == 0:
                data = self.main_window.imported_data
            elif stage_idx == 1:
                data = self.main_window.calibration_data
            elif stage_idx == 2:
                data = self.main_window.processed_data
            else:
                data = None

            if not data:
                return []

            return data.get("warnings_structured", [])

        except Exception:
            return []

    def _update_warning_bar(self):
        """Actualitza la barra d'avisos amb la fase activa."""
        current_idx = self.tab_widget.currentIndex()
        warnings = self._get_warnings_list(current_idx)
        self._warning_bar.update_warnings(warnings)

    def _show_error_details(self, stage_idx: int):
        """Mostra els detalls d'un error en un diàleg."""
        stage_names = {0: "Importar", 1: "Verificar", 2: "Analitzar", 3: "Revisar"}
        stage_name = stage_names.get(stage_idx, "Desconegut")

        # Intentar llegir errors del JSON
        import json
        error_msg = "Error desconegut"

        try:
            seq_path = self.main_window.seq_path
            if seq_path:
                data_path = Path(seq_path) / "CHECK" / "data"
                json_files = {
                    0: "import_manifest.json",
                    1: "calibration_result.json",
                    2: "analysis_result.json",
                }

                filename = json_files.get(stage_idx)
                if filename:
                    json_file = data_path / filename
                    if json_file.exists():
                        with open(json_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        error_msg = data.get("error", data.get("error_message", "Error no especificat"))
        except:
            pass

        QMessageBox.critical(
            self,
            f"Error a {stage_name}",
            f"S'ha produït un error durant l'etapa '{stage_name}':\n\n{error_msg}\n\n"
            f"Prova a reexecutar l'etapa o revisa les dades d'entrada."
        )

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
            if self.tab_widget.currentIndex() == index:
                self._update_header_for_tab(index)

    def _hide_panel_navigation(self):
        """Amaga botons de navegació i acció dels panels (els botons són al header del wizard)."""
        for panel in [self.import_panel, self.calibrate_panel,
                      self.analyze_panel, self.review_panel]:
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
        self.import_panel.warnings_dismissed.connect(self._on_import_warnings_dismissed)
        self.calibrate_panel.calibration_completed.connect(self._on_calibrate_completed)
        self.analyze_panel.analyze_completed.connect(self._on_analyze_completed)
        self.review_panel.review_completed.connect(self._on_export_completed)

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

        # Pre-carregar dades des de JSON si etapes anteriors ja estan completades
        self._preload_completed_stages(seq_path)

        # Anar a primera etapa que necessita atenció (warning o pending)
        # Si tot és "ok", anar a l'última etapa (Revisar)
        first_needs_attention = next(
            (i for i, s in enumerate(self.tab_states) if s in ("warning", "pending", "current")),
            3  # Tot completat → anar a Revisar
        )
        self.tab_widget.setCurrentIndex(first_needs_attention)
        self._update_header_for_tab(first_needs_attention)

        # Carregar sempre al panel d'import:
        # - "pending"/"current": auto-import (no hi ha JSON)
        # - "warning"/"ok": hi ha JSON → carregar des de manifest per mostrar taula
        self.import_panel.load_from_dashboard(seq_path)

        self.sequence_loaded.emit(seq_path)

        # AUTO-EXECUTAR primera etapa pendent
        state = self.tab_states[first_needs_attention]
        if state in ("pending", "current"):
            # Executar amb delay per deixar que la UI s'actualitzi
            from PySide6.QtCore import QTimer
            QTimer.singleShot(200, lambda: self._auto_execute_pending(first_needs_attention))

    def _auto_execute_pending(self, stage_idx: int):
        """Auto-executa l'etapa pendent indicada."""
        if stage_idx == 0:  # Importar
            # Si l'import panel ja té un worker en marxa (carregant manifest), no executar
            if self.import_panel.worker and self.import_panel.worker.isRunning():
                return
            if hasattr(self.import_panel, '_run_import'):
                self._show_executing_state("Importar")
                self.import_panel._run_import(force_reimport=True)
        elif stage_idx == 1:  # Calibrar
            self._execute_stage(1)
        elif stage_idx == 2:  # Analitzar
            self._execute_stage(2)
        elif stage_idx == 3:  # Revisar
            # No auto-executa; l'usuari revisa i clica "Generar Resultats"
            pass

    def _reset_all_panels(self):
        """Reseteja tots els panels quan es carrega una nova SEQ."""
        self.main_window.imported_data = None
        self.main_window.calibration_data = None
        self.main_window.processed_data = None
        self.main_window.review_data = None
        self.main_window.review_completed = False



        # Usar els mètodes reset() de cada panel
        if hasattr(self.import_panel, 'reset'):
            self.import_panel.reset()

        if hasattr(self.calibrate_panel, 'reset'):
            self.calibrate_panel.reset()

        if hasattr(self.analyze_panel, 'reset'):
            self.analyze_panel.reset()

        if hasattr(self.review_panel, 'reset'):
            self.review_panel.reset()

    def _preload_completed_stages(self, seq_path: str):
        """Pre-carrega dades des de JSON per etapes ja completades (evita reimportar)."""
        from pathlib import Path
        import json

        data_path = Path(seq_path) / "CHECK" / "data"
        if not data_path.exists():
            return

        # Import: carregar metadades del manifest per a la info_frame
        # dels panels posteriors (cal/ana/review necessiten method, data_mode, samples, etc.)
        if self.tab_states[0] in ("ok", "warning") and not self.main_window.imported_data:
            manifest_path = data_path / "import_manifest.json"
            if manifest_path.exists():
                try:
                    with open(manifest_path, 'r', encoding='utf-8') as f:
                        manifest = json.load(f)
                    if manifest.get("success") or manifest.get("samples"):
                        # Extreure metadades correctament del manifest
                        seq_info = manifest.get("sequence", {})
                        mf_info = manifest.get("master_file", {})
                        method = (manifest.get("method")
                                  or seq_info.get("method", "COLUMN"))
                        data_mode = (manifest.get("data_mode")
                                     or seq_info.get("data_mode", "DUAL"))
                        master_path = (manifest.get("masterfile_path")
                                       or mf_info.get("path", ""))

                        # Convertir samples de llista (format manifest) a dict
                        # (format import_sequence) per compatibilitat amb
                        # ensure_data_loaded() i la resta del pipeline
                        manifest_samples = manifest.get("samples", [])
                        samples_dict = {}
                        if isinstance(manifest_samples, list):
                            for s_info in manifest_samples:
                                s_name = s_info.get("name", "")
                                if not s_name:
                                    continue
                                reps_dict = {}
                                for r_info in s_info.get("replicas", []):
                                    r_num = str(r_info.get("replica",
                                                r_info.get("rep_num", "1")))
                                    reps_dict[r_num] = {
                                        "direct": r_info.get("direct"),
                                        "uib": r_info.get("uib"),
                                        "dad": r_info.get("dad"),
                                        "dad_source": (r_info.get("dad", {}) or {}).get("source"),
                                        "has_data": False,
                                        "injection_info": r_info.get("injection"),
                                    }
                                samples_dict[s_name] = {
                                    "type": s_info.get("type", "SAMPLE"),
                                    "original_name": s_info.get("original_name", s_name),
                                    "replicas": reps_dict,
                                }
                        elif isinstance(manifest_samples, dict):
                            samples_dict = manifest_samples

                        # Construir imported_data lleuger (metadades only)
                        self.main_window.imported_data = {
                            "success": True,
                            "seq_path": seq_path,
                            "seq_name": seq_info.get("name",
                                        os.path.basename(seq_path)),
                            "method": method,
                            "data_mode": data_mode,
                            "uib_sensitivity": seq_info.get("uib_sensitivity"),
                            "samples": samples_dict,
                            "master_file": master_path,
                            "khp_samples": [s.get("name", "") for s in manifest_samples
                                            if isinstance(s, dict) and s.get("type") == "KHP"]
                                           if isinstance(manifest_samples, list) else [],
                            "control_samples": [s.get("name", "") for s in manifest_samples
                                                if isinstance(s, dict) and s.get("type") == "CONTROL"]
                                               if isinstance(manifest_samples, list) else [],
                            "stats": manifest.get("stats", {}),
                            "data_deferred": True,
                        }
                except Exception as e:
                    logger.warning(f"Error pre-carregant import: {e}")

        # Calibració: carregar des de calibration_result.json
        if self.tab_states[1] in ("ok", "warning") and not self.main_window.calibration_data:
            cal_path = data_path / "calibration_result.json"
            if cal_path.exists():
                try:
                    with open(cal_path, 'r', encoding='utf-8') as f:
                        cal_file = json.load(f)
                    calibrations = cal_file.get("calibrations", [])
                    if calibrations:
                        active_cal = next(
                            (c for c in calibrations if c.get("is_active")),
                            calibrations[0]
                        )
                        area = active_cal.get("area", 0)
                        conc = active_cal.get("conc_ppm", 5)
                        rf = active_cal.get("rf", 0)
                        if rf == 0 and conc > 0 and area > 0:
                            rf = area / conc
                        self.main_window.calibration_data = {
                            "success": True,
                            "mode": active_cal.get("mode", "DUAL"),
                            "rf_direct": active_cal.get("rf_direct", rf),
                            "rf_uib": active_cal.get("rf_uib", 0),
                            "rf": rf,
                            "rf_mass": active_cal.get("rf_mass", 0),
                            "shift_direct": active_cal.get("shift_direct") or active_cal.get("shift_min", 0),
                            "shift_uib": active_cal.get("shift_uib") or active_cal.get("shift_min_u", 0),
                            "khp_area_direct": area,
                            "khp_area_uib": active_cal.get("area_u", 0),
                            "khp_area": area,
                            "khp_conc": conc,
                            "volume_uL": active_cal.get("volume_uL", 0),
                            "khp_source": active_cal.get("khp_source", "LOCAL"),
                            "calibration": active_cal,
                            "errors": [],
                            "loaded_from_json": True,
                        }
                except Exception as e:
                    logger.warning(f"Error pre-carregant calibració: {e}")

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
                3: ("review_result.json", []),
            }

            for idx, (filename, warning_fields) in json_files.items():
                json_path = data_path / filename
                if json_path.exists() and json_path.is_file():
                    try:
                        with open(json_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)

                        # Comprovar si hi ha ERRORS (prefixats amb "ERROR")
                        has_errors = False
                        for field in ["warnings", "errors"]:
                            for msg in data.get(field, []):
                                if str(msg).upper().startswith("ERROR"):
                                    has_errors = True
                                    break
                            if has_errors:
                                break

                        if has_errors:
                            # JSON amb errors = etapa NO completada
                            states[idx] = "pending"
                            continue

                        # Comprovar si hi ha warnings
                        has_warnings = self._check_has_warnings(data, warning_fields)

                        # Comprovar si els warnings estan confirmats
                        # Pot ser True (booleà) o dict amb marked_as_ok=True (des del wizard)
                        wc = data.get("warnings_confirmed")
                        warnings_confirmed = (wc is True) or (isinstance(wc, dict) and wc.get("marked_as_ok") is True)

                        if has_warnings and not warnings_confirmed:
                            states[idx] = "warning"
                        else:
                            states[idx] = "ok"

                    except Exception as e:
                        logger.warning(f"Error llegint {filename}: {e}")
                        states[idx] = "ok"  # Assumir ok si no podem llegir

        except Exception as e:
            logger.warning(f"Error detectant etapes: {e}")

        # Marcar primera etapa pendent com a "current"
        for i, state in enumerate(states):
            if state == "pending":
                states[i] = "current"
                break

        return states

    # Warnings trivials que no mostren triangle (resolts automàticament)
    _TRIVIAL_WARNINGS = {
        'importat des de manifest existent',
        '4-toc_calc no trobat al masterfile, calculant automàticament...',
    }

    def _check_has_warnings(self, data: dict, warning_fields: list) -> bool:
        """Comprova si les dades tenen warnings significatius (no trivials)."""
        for field in warning_fields:
            value = data.get(field)
            if value:
                # Si és una llista, comprovar si té elements no trivials
                if isinstance(value, list) and len(value) > 0:
                    real_warnings = [
                        w for w in value
                        if isinstance(w, dict)
                        or (isinstance(w, str) and w.strip().lower() not in self._TRIVIAL_WARNINGS)
                    ]
                    if real_warnings:
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
                # Tab 3 (Revisar) no té JSON d'estat
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
            logger.warning(f"Error checking confirmed warnings: {e}")
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
        if self.tab_states[index] == "pending":
            self.tab_states[index] = "current"
            self._update_tab_titles()
        self._update_header_for_tab(index)
        self._update_warning_bar()

        state = self.tab_states[index]
        if state in ("ok", "warning"):
            self._load_existing_data_for_tab(index)

    def _load_existing_data_for_tab(self, index: int):
        """Carrega dades existents al panel quan es navega a una etapa completada."""
        if index == 1:  # Calibrar
            if hasattr(self.calibrate_panel, '_check_existing_calibration'):
                self.calibrate_panel._check_existing_calibration()
        elif index == 2:  # Analitzar
            if hasattr(self.analyze_panel, '_check_existing_analysis'):
                self.analyze_panel._check_existing_analysis()
        elif index == 3:  # Revisar
            # Review panel s'actualitza automàticament via showEvent
            pass

    def _on_import_completed(self, data):
        """Callback quan import completa."""
        self.action_btn.setEnabled(True)

        if data and data.get('success'):
            warnings = data.get('warnings_structured', [])
            from hpsec_warnings import get_max_warning_level
            warning_level = get_max_warning_level(warnings)
            if warning_level == 'blocker':
                self._set_tab_state(0, "error")
            elif warnings:
                self._set_tab_state(0, "warning")
            else:
                self._set_tab_state(0, "ok")

            if self.tab_states[1] not in ("ok", "warning"):
                self._set_tab_state(1, "pending")
            self._update_header_for_tab(self.tab_widget.currentIndex())
        else:
            self._set_tab_state(0, "error")
            self._update_header_for_tab(self.tab_widget.currentIndex())

        self._update_warning_bar()

    def _on_import_warnings_dismissed(self):
        """Callback quan els warnings d'importació es descarten des del panel."""
        if self.tab_states[0] == "warning":
            self._set_tab_state(0, "ok")
            self.main_window.set_status("Avisos d'importació marcats com a revisats", 2000)

    def _on_calibrate_completed(self, data):
        """Callback quan calibració completa."""
        self.action_btn.setEnabled(True)

        if data:
            if data.get('success'):
                warnings = data.get('warnings_structured', [])
                from hpsec_warnings import get_max_warning_level
                warning_level = get_max_warning_level(warnings)
                if warning_level in ('blocker', 'warning'):
                    self._set_tab_state(1, "warning")
                else:
                    self._set_tab_state(1, "ok")
            else:
                self._set_tab_state(1, "error")

            if self.tab_states[2] not in ("ok", "warning"):
                self._set_tab_state(2, "pending")
            self._update_header_for_tab(self.tab_widget.currentIndex())
        else:
            self._set_tab_state(1, "error")
            self._update_header_for_tab(self.tab_widget.currentIndex())

        self._update_warning_bar()

    def _on_analyze_completed(self, data):
        """Callback quan anàlisi completa."""
        self.action_btn.setEnabled(True)

        if data and data.get('success'):
            warnings = data.get('warnings_structured', [])
            from hpsec_warnings import get_max_warning_level
            warning_level = get_max_warning_level(warnings)
            if warning_level in ('blocker', 'warning'):
                self._set_tab_state(2, "warning")
            else:
                self._set_tab_state(2, "ok")

            if self.tab_states[3] not in ("ok", "warning"):
                self._set_tab_state(3, "pending")
            self._update_header_for_tab(self.tab_widget.currentIndex())
        else:
            self._set_tab_state(2, "error")
            self._update_header_for_tab(self.tab_widget.currentIndex())

        self._update_warning_bar()

    def _on_export_completed(self, data):
        """Callback quan l'exportació completa."""
        self._set_tab_state(3, "ok")
        self.process_completed.emit(data)
        self._update_header_for_tab(3)

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
