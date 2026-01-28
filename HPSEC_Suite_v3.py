"""
HPSEC Suite v3.0
================
Aplicació unificada per al processament de dades HPSEC.

Basat en v1 (wizard 4 passos) + backend modular de v2.

Pipeline:
1. Consolidar - Llegir fitxers .D i crear Excel consolidats
2. Calibrar - Calcular factor de calibració amb KHP
3. QC - Detectar anomalies i seleccionar millors rèpliques
4. Exportar - Generar fitxers finals i informes PDF

Autor: LEQUIA/STRs
"""

import os
import sys
import glob
import re
import json
import threading
from datetime import datetime
from pathlib import Path

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import pandas as pd

# Matplotlib
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from scipy.stats import pearsonr
from scipy.integrate import trapezoid

import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# MÒDULS HPSEC (Backend modular)
# =============================================================================
from hpsec_config import get_config, ConfigManager
from hpsec_consolidate import (
    consolidate_sequence,
    check_sequence_files,
    match_sample_confidence,
    get_valid_samples_from_hplc_seq,
    CONFIDENCE_THRESHOLD,
)
from hpsec_calibrate import (
    calibrate_sequence,
    find_khp_in_folder,
    load_khp_history,
    save_khp_history,
    register_calibration,
    mark_calibration_as_outlier,
    get_active_calibration,
)
from hpsec_replica import (
    evaluate_replica,
    select_best_replica,
    compare_replicas,
    evaluate_dad,
)
from hpsec_utils import baseline_stats

# =============================================================================
# CONSTANTS
# =============================================================================
APP_NAME = "HPSEC Suite"
VERSION = "3.0.0"

# Colors (paleta professional)
COLORS = {
    "primary": "#2E86AB",      # Blau principal
    "secondary": "#A23B72",    # Magenta
    "success": "#28A745",      # Verd
    "warning": "#F18F01",      # Taronja
    "error": "#C73E1D",        # Vermell
    "dark": "#1A1A2E",         # Fosc
    "light": "#F5F5F5",        # Gris clar
    "white": "#FFFFFF",
    "text": "#333333",
    "text_light": "#666666",
}

# Thresholds per defecte
DEFAULT_MIN_CORR = 0.995
DEFAULT_MAX_DIFF = 10.0

# =============================================================================
# UTILITATS
# =============================================================================

def ensure_check_folder(base_path):
    """Crea carpeta CHECK si no existeix."""
    check_path = os.path.join(base_path, "CHECK")
    os.makedirs(check_path, exist_ok=True)
    return check_path


def detect_mode(seq_path):
    """Detecta si és COLUMN o BP."""
    name = os.path.basename(seq_path).upper()
    if "_BP" in name or name.endswith("BP"):
        return "BP"
    return "COLUMN"


def find_latest_seq(base_path):
    """Troba la carpeta SEQ més recent."""
    if not os.path.isdir(base_path):
        return None

    seq_folders = []
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path) and "_SEQ" in item.upper():
            mtime = os.path.getmtime(item_path)
            seq_folders.append((mtime, item_path))

    if seq_folders:
        seq_folders.sort(reverse=True)
        return seq_folders[0][1]
    return None


# =============================================================================
# DIÀLEG DE REVISIÓ D'ASSIGNACIONS
# =============================================================================

class FileAssignmentReviewDialog:
    """
    Diàleg per revisar i corregir assignacions de fitxers UIB a mostres.

    Apareix després de la consolidació si:
    - Hi ha fitxers orfes (no pertanyen a la SEQ)
    - Hi ha matches amb confiança baixa (< 85%)
    """

    def __init__(self, parent, file_check, valid_samples, seq_path):
        self.parent = parent
        self.file_check = file_check
        self.valid_samples = list(valid_samples) if valid_samples else []
        self.seq_path = seq_path
        self.result = None  # Dict amb correccions o None si cancel

        # Crear finestra
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Revisió d'Assignacions - HPSEC Suite")
        self.dialog.geometry("800x500")
        self.dialog.transient(parent)
        self.dialog.grab_set()

        self._build_ui()
        self.dialog.wait_window()

    def _build_ui(self):
        """Construeix la interfície del diàleg."""
        main = ttk.Frame(self.dialog, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        # Títol
        ttk.Label(main, text="⚠️ Revisió d'Assignacions de Fitxers",
                  font=("Segoe UI", 12, "bold")).pack(pady=(0, 10))

        # Info
        orphans = self.file_check.get("seq_orphan_files", [])
        low_conf = self.file_check.get("low_confidence_matches", [])

        info_text = f"S'han detectat {len(orphans)} fitxers orfes i {len(low_conf)} matches amb confiança baixa.\n"
        info_text += "Podeu assignar manualment els fitxers a les mostres correctes o ignorar-los."
        ttk.Label(main, text=info_text, wraplength=750).pack(pady=(0, 10))

        # Frame amb scroll per la taula
        table_frame = ttk.Frame(main)
        table_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Treeview
        columns = ("file", "status", "suggestion", "action")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=12)

        self.tree.heading("file", text="Fitxer UIB")
        self.tree.heading("status", text="Estat")
        self.tree.heading("suggestion", text="Suggeriment")
        self.tree.heading("action", text="Acció")

        self.tree.column("file", width=200)
        self.tree.column("status", width=120)
        self.tree.column("suggestion", width=180)
        self.tree.column("action", width=200)

        # Scrollbar
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Omplir dades
        self.assignments = {}  # file -> sample or "IGNORE"

        # Fitxers orfes
        for f in orphans:
            self.tree.insert("", tk.END, values=(f, "ORFE", "-", "Ignorar"))
            self.assignments[f] = "IGNORE"

        # Matches baixa confiança
        for m in low_conf:
            f = m.get("file", "")
            conf = m.get("confidence", 0)
            match_to = m.get("matched_to", "-")
            self.tree.insert("", tk.END,
                           values=(f, f"BAIXA ({conf:.0f}%)", match_to, f"Usar: {match_to}"))
            self.assignments[f] = match_to

        # Bind doble clic per editar
        self.tree.bind("<Double-1>", self._on_double_click)

        # Frame per selector (apareix al seleccionar)
        self.edit_frame = ttk.Frame(main)
        self.edit_frame.pack(fill=tk.X, pady=5)

        ttk.Label(self.edit_frame, text="Assignar fitxer seleccionat a:").pack(side=tk.LEFT, padx=5)

        self.sample_var = tk.StringVar()
        self.sample_combo = ttk.Combobox(self.edit_frame, textvariable=self.sample_var,
                                         values=["IGNORAR"] + sorted(self.valid_samples),
                                         state="readonly", width=30)
        self.sample_combo.pack(side=tk.LEFT, padx=5)

        ttk.Button(self.edit_frame, text="Aplicar", command=self._apply_assignment).pack(side=tk.LEFT, padx=5)

        # Botons
        btn_frame = ttk.Frame(main)
        btn_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Button(btn_frame, text="Cancel·lar", command=self._cancel).pack(side=tk.RIGHT, padx=5)
        ttk.Button(btn_frame, text="Guardar i Reprocessar", command=self._save).pack(side=tk.RIGHT, padx=5)
        ttk.Button(btn_frame, text="Ignorar Tot i Continuar", command=self._ignore_all).pack(side=tk.RIGHT, padx=5)

    def _on_double_click(self, event):
        """Selecciona element per editar."""
        selection = self.tree.selection()
        if selection:
            item = self.tree.item(selection[0])
            filename = item["values"][0]
            current_action = self.assignments.get(filename, "IGNORE")

            if current_action == "IGNORE":
                self.sample_var.set("IGNORAR")
            else:
                self.sample_var.set(current_action)

    def _apply_assignment(self):
        """Aplica l'assignació seleccionada."""
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("Avís", "Selecciona un fitxer primer")
            return

        item_id = selection[0]
        item = self.tree.item(item_id)
        filename = item["values"][0]

        new_sample = self.sample_var.get()
        if new_sample == "IGNORAR":
            self.assignments[filename] = "IGNORE"
            action_text = "Ignorar"
        else:
            self.assignments[filename] = new_sample
            action_text = f"Usar: {new_sample}"

        # Actualitzar taula
        self.tree.set(item_id, "action", action_text)

    def _cancel(self):
        """Cancel·la el diàleg."""
        self.result = None
        self.dialog.destroy()

    def _ignore_all(self):
        """Ignora tots els fitxers i continua."""
        self.result = {"action": "IGNORE_ALL", "assignments": {}}
        self.dialog.destroy()

    def _save(self):
        """Guarda les assignacions i indica reprocessar."""
        # Filtrar assignacions reals (no IGNORE)
        real_assignments = {f: s for f, s in self.assignments.items() if s != "IGNORE"}

        self.result = {
            "action": "REPROCESS" if real_assignments else "CONTINUE",
            "assignments": real_assignments,
            "ignored": [f for f, s in self.assignments.items() if s == "IGNORE"]
        }
        self.dialog.destroy()


# =============================================================================
# CLASSE PRINCIPAL
# =============================================================================

class HPSECSuiteV3:
    """Aplicació principal HPSEC Suite v3."""

    def __init__(self, root):
        self.root = root
        self.root.title(f"{APP_NAME} v{VERSION}")
        self.root.geometry("1100x900")
        self.root.minsize(950, 800)

        # Configuració
        self.config = get_config()

        # Estat
        self.seq_path = None
        self.current_step = 0
        self.steps = ["Consolidar", "Calibrar", "QC", "Exportar"]

        # Dades
        self.consolidated_data = {}
        self.calibration_data = {}
        self.qc_results = {}
        self.selected_replicas = {}

        # Flags
        self.is_processing = False

        # Construir GUI
        self._setup_styles()
        self._build_gui()

        # Detectar última SEQ al iniciar
        self._check_auto_detect()

    def _setup_styles(self):
        """Configura estils TTK."""
        style = ttk.Style()
        style.theme_use('clam')

        # Botons
        style.configure("Primary.TButton",
                       font=("Segoe UI", 10, "bold"),
                       padding=(20, 10))

        style.configure("Success.TButton",
                       font=("Segoe UI", 10, "bold"),
                       padding=(20, 10))

        # Treeview
        style.configure("Treeview",
                       font=("Segoe UI", 9),
                       rowheight=25)
        style.configure("Treeview.Heading",
                       font=("Segoe UI", 9, "bold"))

    def _build_gui(self):
        """Construeix la interfície gràfica."""
        # IMPORTANT: Ordre de pack determina layout
        # 1. Header (TOP)
        # 2. Steps indicator (TOP)
        # 3. Footer (BOTTOM) - ha d'anar ABANS del main_area
        # 4. Action bar (BOTTOM) - botons d'acció fixos
        # 5. Main area (FILL) - contingut scrollable

        self._build_header()
        self._build_steps_indicator()
        self._build_footer()      # Footer ABANS per fixar-lo al fons
        self._build_action_bar()  # Barra d'accions fixa
        self._build_main_area()   # Contingut scrollable al mig

        # Mostrar primer pas
        self._show_step(0)

    # =========================================================================
    # HEADER
    # =========================================================================

    def _build_header(self):
        """Construeix la capçalera."""
        header = tk.Frame(self.root, bg=COLORS["primary"], height=60)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        # Títol
        tk.Label(header, text=APP_NAME,
                font=("Segoe UI", 16, "bold"),
                bg=COLORS["primary"], fg=COLORS["white"]).pack(side=tk.LEFT, padx=20, pady=15)

        tk.Label(header, text=f"v{VERSION}",
                font=("Segoe UI", 10),
                bg=COLORS["primary"], fg=COLORS["light"]).pack(side=tk.LEFT, pady=15)

        # Botons dreta
        btn_config = tk.Button(header, text="⚙",
                              command=self._open_config,
                              bg=COLORS["white"], fg=COLORS["primary"],
                              font=("Segoe UI", 12),
                              relief="flat", width=3)
        btn_config.pack(side=tk.RIGHT, padx=10, pady=15)

        btn_repo = tk.Button(header, text="Repositori",
                            command=self._open_repositori,
                            bg=COLORS["success"], fg=COLORS["white"],
                            font=("Segoe UI", 9, "bold"),
                            relief="flat", padx=15, pady=5)
        btn_repo.pack(side=tk.RIGHT, padx=5, pady=15)

        btn_folder = tk.Button(header, text="Seleccionar SEQ",
                              command=self._select_folder,
                              bg=COLORS["white"], fg=COLORS["primary"],
                              font=("Segoe UI", 9, "bold"),
                              relief="flat", padx=15, pady=5)
        btn_folder.pack(side=tk.RIGHT, padx=5, pady=15)

        # Path actual
        self.path_var = tk.StringVar(value="Cap carpeta seleccionada")
        tk.Label(header, textvariable=self.path_var,
                font=("Segoe UI", 9),
                bg=COLORS["primary"], fg=COLORS["light"]).pack(side=tk.RIGHT, padx=10, pady=15)

    # =========================================================================
    # STEPS INDICATOR
    # =========================================================================

    def _build_steps_indicator(self):
        """Construeix l'indicador de passos."""
        steps_frame = tk.Frame(self.root, bg=COLORS["light"], height=50)
        steps_frame.pack(fill=tk.X)
        steps_frame.pack_propagate(False)

        self.step_labels = []

        for i, step in enumerate(self.steps):
            frame = tk.Frame(steps_frame, bg=COLORS["light"])
            frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

            text = f"{i+1}. {step}"
            lbl = tk.Label(frame, text=text,
                          font=("Segoe UI", 11),
                          bg=COLORS["light"], fg=COLORS["dark"],
                          cursor="hand2")
            lbl.pack(expand=True)
            lbl.bind("<Button-1>", lambda e, idx=i: self._on_step_click(idx))

            self.step_labels.append(lbl)

            # Fletxa separadora
            if i < len(self.steps) - 1:
                sep = tk.Label(steps_frame, text="→",
                              font=("Segoe UI", 14),
                              bg=COLORS["light"], fg=COLORS["dark"])
                sep.pack(side=tk.LEFT, padx=5)

    # =========================================================================
    # ACTION BAR (Botons fixos)
    # =========================================================================

    def _build_action_bar(self):
        """Construeix la barra d'accions fixa amb botons per cada pas."""
        self.action_bar = tk.Frame(self.root, bg=COLORS["white"], height=80)
        self.action_bar.pack(fill=tk.X, side=tk.BOTTOM)
        self.action_bar.pack_propagate(False)

        # Separador superior
        ttk.Separator(self.action_bar, orient="horizontal").pack(fill=tk.X)

        # Container pels botons
        btn_container = tk.Frame(self.action_bar, bg=COLORS["white"])
        btn_container.pack(expand=True, fill=tk.BOTH, pady=10)

        # === Frame per cada pas (només un visible a la vegada) ===
        self.action_frames = []

        # Pas 1: Consolidar
        frame1 = tk.Frame(btn_container, bg=COLORS["white"])
        self.action_frames.append(frame1)

        self.progress_con = ttk.Progressbar(frame1, mode='determinate', length=300)
        self.progress_con.pack(side=tk.LEFT, padx=(20, 10))

        self.lbl_con_progress = tk.Label(frame1, text="",
                                         font=("Segoe UI", 9),
                                         bg=COLORS["white"], fg=COLORS["text_light"],
                                         width=30)
        self.lbl_con_progress.pack(side=tk.LEFT, padx=10)

        self.btn_consolidar = tk.Button(frame1, text="▶ Consolidar Dades",
                                        command=self._run_consolidation,
                                        bg=COLORS["primary"], fg=COLORS["white"],
                                        font=("Segoe UI", 11, "bold"),
                                        relief="flat", padx=25, pady=8,
                                        state="disabled")
        self.btn_consolidar.pack(side=tk.RIGHT, padx=20)

        # Pas 2: Calibrar
        frame2 = tk.Frame(btn_container, bg=COLORS["white"])
        self.action_frames.append(frame2)

        self.btn_calibrar = tk.Button(frame2, text="▶ Calibrar",
                                      command=self._run_calibration,
                                      bg=COLORS["primary"], fg=COLORS["white"],
                                      font=("Segoe UI", 11, "bold"),
                                      relief="flat", padx=25, pady=8,
                                      state="disabled")
        self.btn_calibrar.pack(side=tk.RIGHT, padx=20)

        self.lbl_cal_status = tk.Label(frame2, text="",
                                       font=("Segoe UI", 10),
                                       bg=COLORS["white"], fg=COLORS["text_light"])
        self.lbl_cal_status.pack(side=tk.LEFT, padx=20)

        # Pas 3: QC
        frame3 = tk.Frame(btn_container, bg=COLORS["white"])
        self.action_frames.append(frame3)

        self.btn_qc = tk.Button(frame3, text="▶ Analitzar Qualitat",
                                command=self._run_qc,
                                bg=COLORS["primary"], fg=COLORS["white"],
                                font=("Segoe UI", 11, "bold"),
                                relief="flat", padx=25, pady=8,
                                state="disabled")
        self.btn_qc.pack(side=tk.RIGHT, padx=20)

        # Pas 4: Exportar
        frame4 = tk.Frame(btn_container, bg=COLORS["white"])
        self.action_frames.append(frame4)

        self.btn_export = tk.Button(frame4, text="▶ Exportar Resultats",
                                    command=self._run_export,
                                    bg=COLORS["success"], fg=COLORS["white"],
                                    font=("Segoe UI", 11, "bold"),
                                    relief="flat", padx=25, pady=8,
                                    state="disabled")
        self.btn_export.pack(side=tk.RIGHT, padx=20)

    def _show_action_bar(self, step_idx):
        """Mostra els botons corresponents al pas actual."""
        for i, frame in enumerate(self.action_frames):
            if i == step_idx:
                frame.pack(fill=tk.BOTH, expand=True)
            else:
                frame.pack_forget()

    # =========================================================================
    # MAIN AREA (Scrollable)
    # =========================================================================

    def _build_main_area(self):
        """Construeix l'àrea principal amb scroll."""
        # Container principal
        self.main_container = tk.Frame(self.root, bg=COLORS["white"])
        self.main_container.pack(fill=tk.BOTH, expand=True)

        # Canvas amb scrollbar per contingut
        self.canvas = tk.Canvas(self.main_container, bg=COLORS["white"], highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.main_container, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg=COLORS["white"])

        # Configurar scroll
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        # Fer que el frame interior s'expandeixi horitzontalment
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Mousewheel scroll
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        # Pack canvas i scrollbar
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Main frame dins el scrollable
        self.main_frame = tk.Frame(self.scrollable_frame, bg=COLORS["white"])
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Un frame per cada pas
        self.step_frames = []
        for i in range(4):
            frame = tk.Frame(self.main_frame, bg=COLORS["white"])
            self.step_frames.append(frame)

        # Construir contingut de cada pas
        self._build_step_consolidar(self.step_frames[0])
        self._build_step_calibrar(self.step_frames[1])
        self._build_step_qc(self.step_frames[2])
        self._build_step_exportar(self.step_frames[3])

    def _on_canvas_configure(self, event):
        """Ajusta l'amplada del frame interior al canvas."""
        self.canvas.itemconfig(self.canvas_window, width=event.width)

    def _on_mousewheel(self, event):
        """Scroll amb roda del ratolí."""
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    # =========================================================================
    # PAS 1: CONSOLIDAR
    # =========================================================================

    def _build_step_consolidar(self, parent):
        """Construeix el pas de consolidació."""
        # Títol + botó obrir carpeta
        header_frame = tk.Frame(parent, bg=COLORS["white"])
        header_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(header_frame, text="Pas 1: Consolidació de Dades",
                font=("Segoe UI", 14, "bold"),
                bg=COLORS["white"], fg=COLORS["primary"]).pack(side=tk.LEFT)

        self.btn_open_seq_folder = tk.Button(header_frame, text="Obrir Carpeta",
                                             command=self._open_seq_folder,
                                             bg=COLORS["light"], fg=COLORS["dark"],
                                             font=("Segoe UI", 9),
                                             relief="flat", padx=10, pady=2)
        self.btn_open_seq_folder.pack(side=tk.RIGHT)

        tk.Label(parent, text="Llegeix fitxers UIB (CSV) i DAD (Export3D) i crea Excel consolidats per cada mostra.",
                font=("Segoe UI", 10),
                bg=COLORS["white"], fg=COLORS["dark"]).pack(anchor="w", pady=(0, 10))

        # Info frame
        info_frame = tk.LabelFrame(parent, text="Informació de la Seqüència",
                                   font=("Segoe UI", 10, "bold"),
                                   bg=COLORS["white"])
        info_frame.pack(fill=tk.X, pady=10)

        # MasterFile
        self.lbl_con_master = tk.Label(info_frame, text="MasterFile: -",
                                       font=("Segoe UI", 10),
                                       bg=COLORS["white"])
        self.lbl_con_master.pack(anchor="w", padx=10, pady=3)

        # Fitxers dades
        self.lbl_con_uib = tk.Label(info_frame, text="Fitxers UIB (CSV): -",
                                    font=("Segoe UI", 10),
                                    bg=COLORS["white"])
        self.lbl_con_uib.pack(anchor="w", padx=10, pady=3)

        self.lbl_con_dad = tk.Label(info_frame, text="Fitxers DAD (Export3D): -",
                                    font=("Segoe UI", 10),
                                    bg=COLORS["white"])
        self.lbl_con_dad.pack(anchor="w", padx=10, pady=3)

        # Mode
        self.lbl_con_mode = tk.Label(info_frame, text="Mode: -",
                                     font=("Segoe UI", 10),
                                     bg=COLORS["white"])
        self.lbl_con_mode.pack(anchor="w", padx=10, pady=3)

        # Estat consolidació
        self.lbl_con_status = tk.Label(info_frame, text="Estat: -",
                                       font=("Segoe UI", 10),
                                       bg=COLORS["white"])
        self.lbl_con_status.pack(anchor="w", padx=10, pady=3)

        # Frame per botons PDF (inicialment amagat)
        self.pdf_frame = tk.Frame(info_frame, bg=COLORS["white"])
        self.pdf_frame.pack(anchor="w", padx=10, pady=5)

        self.btn_open_report = tk.Button(self.pdf_frame, text="Obrir Informe PDF",
                                         command=self._open_consolidation_report,
                                         bg=COLORS["secondary"], fg=COLORS["white"],
                                         font=("Segoe UI", 9),
                                         relief="flat", padx=10, pady=3)

        self.btn_open_chromato = tk.Button(self.pdf_frame, text="Obrir Cromatogrames PDF",
                                           command=self._open_chromatograms_report,
                                           bg=COLORS["secondary"], fg=COLORS["white"],
                                           font=("Segoe UI", 9),
                                           relief="flat", padx=10, pady=3)

        # === FRAME RESUM CONSOLIDACIÓ (inicialment amagat) ===
        self.summary_frame = tk.LabelFrame(parent, text="Resum Consolidació",
                                           font=("Segoe UI", 10, "bold"),
                                           bg=COLORS["white"])
        # No pack() encara - es mostrarà després de consolidar

        # Subframe per estadístiques en 2 columnes
        summary_content = tk.Frame(self.summary_frame, bg=COLORS["white"])
        summary_content.pack(fill=tk.X, padx=10, pady=5)

        # Columna esquerra: Comptadors
        left_col = tk.Frame(summary_content, bg=COLORS["white"])
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        self.lbl_sum_samples = tk.Label(left_col, text="Mostres: -",
                                        font=("Segoe UI", 10),
                                        bg=COLORS["white"])
        self.lbl_sum_samples.pack(anchor="w")

        self.lbl_sum_alignment = tk.Label(left_col, text="Alineació: -",
                                          font=("Segoe UI", 10),
                                          bg=COLORS["white"])
        self.lbl_sum_alignment.pack(anchor="w")

        self.lbl_sum_snr = tk.Label(left_col, text="SNR mediana: -",
                                    font=("Segoe UI", 10),
                                    bg=COLORS["white"])
        self.lbl_sum_snr.pack(anchor="w")

        self.lbl_sum_lod = tk.Label(left_col, text="LOD: -",
                                    font=("Segoe UI", 10),
                                    bg=COLORS["white"])
        self.lbl_sum_lod.pack(anchor="w")

        self.lbl_sum_generated = tk.Label(left_col, text="",
                                          font=("Segoe UI", 9),
                                          bg=COLORS["white"], fg=COLORS["text_light"])
        self.lbl_sum_generated.pack(anchor="w", pady=(5, 0))

        # Columna dreta: Timeouts
        right_col = tk.Frame(summary_content, bg=COLORS["white"])
        right_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(right_col, text="Timeouts TOC:",
                font=("Segoe UI", 10, "bold"),
                bg=COLORS["white"]).pack(anchor="w")

        # Frame per les barres de severitat
        self.timeout_bars_frame = tk.Frame(right_col, bg=COLORS["white"])
        self.timeout_bars_frame.pack(anchor="w", pady=5)

        self.lbl_timeout_ok = tk.Label(self.timeout_bars_frame, text="OK: -",
                                       font=("Segoe UI", 9),
                                       bg="#c6efce", fg="#006100", padx=8, pady=2)
        self.lbl_timeout_ok.pack(side=tk.LEFT, padx=2)

        self.lbl_timeout_info = tk.Label(self.timeout_bars_frame, text="INFO: -",
                                         font=("Segoe UI", 9),
                                         bg="#cce5ff", fg="#004085", padx=8, pady=2)
        self.lbl_timeout_info.pack(side=tk.LEFT, padx=2)

        self.lbl_timeout_warn = tk.Label(self.timeout_bars_frame, text="WARN: -",
                                         font=("Segoe UI", 9),
                                         bg="#fff3cd", fg="#856404", padx=8, pady=2)
        self.lbl_timeout_warn.pack(side=tk.LEFT, padx=2)

        self.lbl_timeout_crit = tk.Label(self.timeout_bars_frame, text="CRIT: -",
                                         font=("Segoe UI", 9),
                                         bg="#f8d7da", fg="#721c24", padx=8, pady=2)
        self.lbl_timeout_crit.pack(side=tk.LEFT, padx=2)

        self.lbl_sum_critical = tk.Label(right_col, text="",
                                         font=("Segoe UI", 9),
                                         bg=COLORS["white"], fg=COLORS["error"],
                                         wraplength=300)
        self.lbl_sum_critical.pack(anchor="w")

        # Warnings
        self.lbl_sum_warnings = tk.Label(self.summary_frame, text="",
                                         font=("Segoe UI", 9),
                                         bg=COLORS["white"], fg=COLORS["warning"],
                                         wraplength=600, justify=tk.LEFT)
        self.lbl_sum_warnings.pack(anchor="w", padx=10, pady=5)

        # NOTA: Progress bar i botó "Consolidar" ara estan a la Action Bar (sempre visibles)

    # =========================================================================
    # PAS 2: CALIBRAR
    # =========================================================================

    def _build_step_calibrar(self, parent):
        """Construeix el pas de calibració."""
        tk.Label(parent, text="Pas 2: Calibració amb KHP",
                font=("Segoe UI", 14, "bold"),
                bg=COLORS["white"], fg=COLORS["primary"]).pack(anchor="w", pady=(0, 10))

        tk.Label(parent, text="Calcula el factor de calibració utilitzant les mostres KHP.",
                font=("Segoe UI", 10),
                bg=COLORS["white"], fg=COLORS["dark"]).pack(anchor="w", pady=(0, 10))

        # Frame principal amb dues columnes
        main_frame = tk.Frame(parent, bg=COLORS["white"])
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Columna esquerra: Info
        left_frame = tk.Frame(main_frame, bg=COLORS["white"], width=350)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_frame.pack_propagate(False)

        # Info KHP
        info_frame = tk.LabelFrame(left_frame, text="Informació KHP",
                                   font=("Segoe UI", 10, "bold"),
                                   bg=COLORS["white"])
        info_frame.pack(fill=tk.X, pady=5)

        self.lbl_cal_source = tk.Label(info_frame, text="Origen: -",
                                       font=("Segoe UI", 9),
                                       bg=COLORS["white"], fg=COLORS["text_light"])
        self.lbl_cal_source.pack(anchor="w", padx=10, pady=2)

        self.lbl_cal_khp = tk.Label(info_frame, text="KHP: -",
                                    font=("Segoe UI", 10),
                                    bg=COLORS["white"])
        self.lbl_cal_khp.pack(anchor="w", padx=10, pady=2)

        self.lbl_cal_conc = tk.Label(info_frame, text="Concentració: -",
                                     font=("Segoe UI", 10),
                                     bg=COLORS["white"])
        self.lbl_cal_conc.pack(anchor="w", padx=10, pady=2)

        self.lbl_cal_replicas = tk.Label(info_frame, text="Rèpliques: -",
                                         font=("Segoe UI", 10),
                                         bg=COLORS["white"])
        self.lbl_cal_replicas.pack(anchor="w", padx=10, pady=2)

        # Resultats calibració
        results_frame = tk.LabelFrame(left_frame, text="Resultats Calibració",
                                      font=("Segoe UI", 10, "bold"),
                                      bg=COLORS["white"])
        results_frame.pack(fill=tk.X, pady=5)

        self.lbl_cal_factor = tk.Label(results_frame, text="Factor: -",
                                       font=("Segoe UI", 12, "bold"),
                                       bg=COLORS["white"], fg=COLORS["primary"])
        self.lbl_cal_factor.pack(anchor="w", padx=10, pady=3)

        self.lbl_cal_area = tk.Label(results_frame, text="Àrea: -",
                                     font=("Segoe UI", 10),
                                     bg=COLORS["white"])
        self.lbl_cal_area.pack(anchor="w", padx=10, pady=2)

        self.lbl_cal_rsd = tk.Label(results_frame, text="RSD: -",
                                    font=("Segoe UI", 10),
                                    bg=COLORS["white"])
        self.lbl_cal_rsd.pack(anchor="w", padx=10, pady=2)

        self.lbl_cal_shift = tk.Label(results_frame, text="Shift DOC-DAD: -",
                                      font=("Segoe UI", 10),
                                      bg=COLORS["white"])
        self.lbl_cal_shift.pack(anchor="w", padx=10, pady=2)

        # Qualitat
        quality_frame = tk.LabelFrame(left_frame, text="Qualitat",
                                      font=("Segoe UI", 10, "bold"),
                                      bg=COLORS["white"])
        quality_frame.pack(fill=tk.X, pady=5)

        self.lbl_cal_symmetry = tk.Label(quality_frame, text="Simetria: -",
                                         font=("Segoe UI", 10),
                                         bg=COLORS["white"])
        self.lbl_cal_symmetry.pack(anchor="w", padx=10, pady=2)

        self.lbl_cal_snr = tk.Label(quality_frame, text="SNR: -",
                                    font=("Segoe UI", 10),
                                    bg=COLORS["white"])
        self.lbl_cal_snr.pack(anchor="w", padx=10, pady=2)

        self.lbl_cal_quality = tk.Label(quality_frame, text="Estat: -",
                                        font=("Segoe UI", 10, "bold"),
                                        bg=COLORS["white"])
        self.lbl_cal_quality.pack(anchor="w", padx=10, pady=2)

        self.lbl_cal_issues = tk.Label(quality_frame, text="",
                                       font=("Segoe UI", 9),
                                       bg=COLORS["white"], fg=COLORS["warning"],
                                       wraplength=320, justify=tk.LEFT)
        self.lbl_cal_issues.pack(anchor="w", padx=10, pady=2)

        # Columna dreta: Gràfic
        right_frame = tk.Frame(main_frame, bg=COLORS["white"])
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.cal_canvas_frame = tk.Frame(right_frame, bg=COLORS["light"])
        self.cal_canvas_frame.pack(fill=tk.BOTH, expand=True)

        # Botons secundaris (a sota del gràfic)
        # NOTA: Botó principal "Calibrar" ara està a la Action Bar
        btn_frame = tk.Frame(parent, bg=COLORS["white"])
        btn_frame.pack(pady=10)

        self.btn_khp_history = tk.Button(btn_frame, text="Històric KHP",
                                         command=self._show_khp_history,
                                         bg=COLORS["secondary"], fg=COLORS["white"],
                                         font=("Segoe UI", 11, "bold"),
                                         relief="flat", padx=30, pady=10)
        self.btn_khp_history.pack(side=tk.LEFT, padx=10)

        self.btn_outlier = tk.Button(btn_frame, text="❌ Descartar (Outlier)",
                                     command=self._mark_as_outlier,
                                     bg=COLORS["warning"], fg=COLORS["dark"],
                                     font=("Segoe UI", 10, "bold"),
                                     relief="flat", padx=20, pady=10)
        # Amagat inicialment

    # =========================================================================
    # PAS 3: QC (QUALITY CONTROL)
    # =========================================================================

    def _build_step_qc(self, parent):
        """Construeix el pas de QC."""
        tk.Label(parent, text="Pas 3: Quality Control (QC)",
                font=("Segoe UI", 14, "bold"),
                bg=COLORS["white"], fg=COLORS["primary"]).pack(anchor="w", pady=(0, 15))

        tk.Label(parent, text="Detecta anomalies (TimeOUT, Batman) i selecciona les millors rèpliques.",
                font=("Segoe UI", 10),
                bg=COLORS["white"], fg=COLORS["dark"]).pack(anchor="w", pady=(0, 10))

        # Taula de resultats
        table_frame = tk.Frame(parent, bg=COLORS["white"])
        table_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Definir columnes
        cols = ["Mostra", "Estat", "DOC R", "DOC Δ%", "Sel DOC", "DAD Qual", "Sel DAD"]
        self.tree_qc = ttk.Treeview(table_frame, columns=cols, show='headings', height=12)

        widths = {"Mostra": 160, "Estat": 70, "DOC R": 70, "DOC Δ%": 70,
                  "Sel DOC": 70, "DAD Qual": 90, "Sel DAD": 70}
        for col in cols:
            self.tree_qc.heading(col, text=col)
            self.tree_qc.column(col, width=widths.get(col, 80), anchor="center")

        # Scrollbar
        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree_qc.yview)
        self.tree_qc.configure(yscrollcommand=vsb.set)

        self.tree_qc.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        # Tags de colors
        self.tree_qc.tag_configure('OK', background='#c6efce', foreground='#006100')
        self.tree_qc.tag_configure('WARN', background='#fff3cd', foreground='#856404')
        self.tree_qc.tag_configure('FAIL', background='#f8d7da', foreground='#721c24')
        self.tree_qc.tag_configure('MODIFIED', background='#cce5ff', foreground='#004085')

        # Bind events
        self.tree_qc.bind("<Double-1>", self._on_qc_double_click)
        self.tree_qc.bind("<Button-3>", self._on_qc_right_click)

        # Menú contextual
        self.qc_menu = tk.Menu(self.root, tearoff=0)
        self.qc_menu.add_command(label="DOC → Rèplica 1", command=lambda: self._force_selection("DOC", "1"))
        self.qc_menu.add_command(label="DOC → Rèplica 2", command=lambda: self._force_selection("DOC", "2"))
        self.qc_menu.add_separator()
        self.qc_menu.add_command(label="DAD → Rèplica 1", command=lambda: self._force_selection("DAD", "1"))
        self.qc_menu.add_command(label="DAD → Rèplica 2", command=lambda: self._force_selection("DAD", "2"))
        self.qc_menu.add_separator()
        self.qc_menu.add_command(label="Obrir gràfic QAQC", command=self._open_qaqc_plot)

        # Progrés
        # NOTA: Botó principal "QC" ara està a la Action Bar
        self.progress_qc = ttk.Progressbar(parent, mode='determinate', length=400)
        self.progress_qc.pack(pady=10)

    # =========================================================================
    # PAS 4: EXPORTAR
    # =========================================================================

    def _build_step_exportar(self, parent):
        """Construeix el pas d'exportació."""
        tk.Label(parent, text="Pas 4: Exportació Final",
                font=("Segoe UI", 14, "bold"),
                bg=COLORS["white"], fg=COLORS["primary"]).pack(anchor="w", pady=(0, 15))

        tk.Label(parent, text="Genera els fitxers finals i els informes PDF.",
                font=("Segoe UI", 10),
                bg=COLORS["white"], fg=COLORS["dark"]).pack(anchor="w", pady=(0, 20))

        # Opcions d'exportació
        options_frame = tk.LabelFrame(parent, text="Opcions",
                                      font=("Segoe UI", 10, "bold"),
                                      bg=COLORS["white"])
        options_frame.pack(fill=tk.X, pady=10)

        self.var_export_pdf = tk.BooleanVar(value=True)
        self.var_export_excel = tk.BooleanVar(value=True)
        self.var_export_selected = tk.BooleanVar(value=True)

        tk.Checkbutton(options_frame, text="Generar informe PDF",
                      variable=self.var_export_pdf,
                      font=("Segoe UI", 10),
                      bg=COLORS["white"]).pack(anchor="w", padx=10, pady=5)

        tk.Checkbutton(options_frame, text="Generar Excel resum",
                      variable=self.var_export_excel,
                      font=("Segoe UI", 10),
                      bg=COLORS["white"]).pack(anchor="w", padx=10, pady=5)

        tk.Checkbutton(options_frame, text="Copiar rèpliques seleccionades",
                      variable=self.var_export_selected,
                      font=("Segoe UI", 10),
                      bg=COLORS["white"]).pack(anchor="w", padx=10, pady=5)

        # Resum
        self.export_summary = tk.Label(parent, text="",
                                       font=("Segoe UI", 10),
                                       bg=COLORS["white"], fg=COLORS["dark"],
                                       justify=tk.LEFT)
        self.export_summary.pack(anchor="w", pady=20)

        # Progrés
        # NOTA: Botó principal "Exportar" ara està a la Action Bar
        self.progress_export = ttk.Progressbar(parent, mode='determinate', length=400)
        self.progress_export.pack(pady=10)

        # Botó secundari
        btn_frame = tk.Frame(parent, bg=COLORS["white"])
        btn_frame.pack(pady=20)

        self.btn_open_folder = tk.Button(btn_frame, text="Obrir Carpeta CHECK",
                                         command=self._open_check_folder,
                                         bg=COLORS["dark"], fg=COLORS["white"],
                                         font=("Segoe UI", 11, "bold"),
                                         relief="flat", padx=30, pady=10,
                                         state="disabled")
        self.btn_open_folder.pack(side=tk.LEFT, padx=10)

    # =========================================================================
    # FOOTER
    # =========================================================================

    def _build_footer(self):
        """Construeix el footer amb navegació."""
        footer = tk.Frame(self.root, bg=COLORS["light"], height=60)
        footer.pack(fill=tk.X, side=tk.BOTTOM)
        footer.pack_propagate(False)

        self.btn_prev = tk.Button(footer, text="← Anterior",
                                  command=self._prev_step,
                                  bg=COLORS["dark"], fg=COLORS["white"],
                                  font=("Segoe UI", 10),
                                  relief="flat", padx=20, pady=8,
                                  state="disabled")
        self.btn_prev.pack(side=tk.LEFT, padx=20, pady=12)

        self.btn_next = tk.Button(footer, text="Següent →",
                                  command=self._next_step,
                                  bg=COLORS["primary"], fg=COLORS["white"],
                                  font=("Segoe UI", 10),
                                  relief="flat", padx=20, pady=8)
        self.btn_next.pack(side=tk.RIGHT, padx=20, pady=12)

        # Status
        self.lbl_status = tk.Label(footer, text="Selecciona una carpeta SEQ per començar",
                                   font=("Segoe UI", 9),
                                   bg=COLORS["light"], fg=COLORS["dark"])
        self.lbl_status.pack(expand=True)

    # =========================================================================
    # NAVEGACIÓ
    # =========================================================================

    def _show_step(self, step_idx):
        """Mostra un pas específic."""
        # Amagar tots els frames de contingut
        for frame in self.step_frames:
            frame.pack_forget()

        # Mostrar el frame seleccionat
        self.step_frames[step_idx].pack(fill=tk.BOTH, expand=True)
        self.current_step = step_idx

        # Mostrar barra d'accions corresponent
        self._show_action_bar(step_idx)

        # Scroll al principi
        self.canvas.yview_moveto(0)

        # Actualitzar indicadors
        for i, lbl in enumerate(self.step_labels):
            if i == step_idx:
                lbl.configure(bg=COLORS["primary"], fg=COLORS["white"],
                             font=("Segoe UI", 11, "bold"))
            elif i < step_idx:
                lbl.configure(bg=COLORS["success"], fg=COLORS["white"],
                             font=("Segoe UI", 11))
            else:
                lbl.configure(bg=COLORS["light"], fg=COLORS["dark"],
                             font=("Segoe UI", 11))

        # Actualitzar botons navegació
        self.btn_prev.configure(state="normal" if step_idx > 0 else "disabled")
        self.btn_next.configure(state="normal" if step_idx < 3 else "disabled")

        if step_idx == 3:
            self.btn_next.configure(text="Finalitzar")
        else:
            self.btn_next.configure(text="Següent →")

    def _next_step(self):
        """Avança al següent pas."""
        if self.current_step < 3:
            self._show_step(self.current_step + 1)

    def _prev_step(self):
        """Retrocedeix al pas anterior."""
        if self.current_step > 0:
            self._show_step(self.current_step - 1)

    def _on_step_click(self, idx):
        """Navega a un pas clicant l'indicador."""
        self._show_step(idx)

    # =========================================================================
    # SELECCIÓ DE CARPETA
    # =========================================================================

    def _check_auto_detect(self):
        """Comprova si oferir auto-detecció de SEQ."""
        base_path = self.config.get("paths", "data_folder",
                                    default="C:/Users/Lequia/Desktop/Dades2")
        latest = find_latest_seq(base_path)

        if latest:
            seq_name = os.path.basename(latest)
            if messagebox.askyesno("Última SEQ detectada",
                                   f"S'ha detectat la seqüència més recent:\n\n{seq_name}\n\nVols processar-la?"):
                self._load_folder(latest)

    def _select_folder(self):
        """Selecciona una carpeta SEQ."""
        base_path = self.config.get("paths", "data_folder",
                                    default="C:/Users/Lequia/Desktop/Dades2")
        folder = filedialog.askdirectory(title="Selecciona carpeta SEQ",
                                        initialdir=base_path)
        if folder:
            self._load_folder(folder)

    def _load_folder(self, folder):
        """Carrega una carpeta SEQ i analitza el seu contingut."""
        self.seq_path = folder
        self.path_var.set(os.path.basename(folder))

        # Reset dades
        self.consolidated_data = {}
        self.calibration_data = {}
        self.qc_results = {}
        self.selected_replicas = {}
        self.is_processing = False

        # Analitzar contingut
        self._analyze_folder()

        # Activar botó consolidar
        self.btn_consolidar.configure(state="normal")
        self.lbl_status.configure(text=f"Carpeta carregada: {os.path.basename(folder)}")

        # Anar al primer pas
        self._show_step(0)

    def _analyze_folder(self):
        """Analitza el contingut de la carpeta SEQ."""
        if not self.seq_path:
            return

        mode = detect_mode(self.seq_path)

        # Buscar MasterFile
        master_files = glob.glob(os.path.join(self.seq_path, "*MasterFile*.xlsx"))
        master_files = [f for f in master_files if "~$" not in f and "backup" not in f.lower()]
        if master_files:
            master_name = os.path.basename(master_files[0])
            self.lbl_con_master.configure(text=f"MasterFile: {master_name}", fg=COLORS["success"])
        else:
            self.lbl_con_master.configure(text="MasterFile: No trobat", fg=COLORS["warning"])

        # Obtenir info de fitxers UIB/DAD
        info = check_sequence_files(self.seq_path)
        uib_count = info.get("uib", {}).get("count_found", 0)
        dad_count = info.get("dad", {}).get("count_found", 0)

        # Determinar mode de dades (UIB, Direct, Dual)
        if uib_count > 0 and dad_count > 0:
            data_mode = "DUAL (UIB + Direct)"
        elif uib_count > 0:
            data_mode = "UIB"
        elif dad_count > 0:
            data_mode = "Direct"
        else:
            data_mode = "Sense dades"

        # Actualitzar labels
        self.lbl_con_uib.configure(text=f"Fitxers UIB (CSV): {uib_count}")
        self.lbl_con_dad.configure(text=f"Fitxers DAD (Export3D): {dad_count}")
        self.lbl_con_mode.configure(text=f"Mode: {mode} | Dades: {data_mode}")

        # Amagar botons PDF i resum per defecte
        self.btn_open_report.pack_forget()
        self.btn_open_chromato.pack_forget()
        self.summary_frame.pack_forget()

        # Comprovar si ja està consolidat
        con_folder = os.path.join(self.seq_path, "Resultats_Consolidats")
        if os.path.isdir(con_folder):
            con_files = glob.glob(os.path.join(con_folder, "*.xlsx"))
            con_files = [f for f in con_files if "~$" not in f]  # Excloure temporals
            if con_files:
                self.lbl_con_status.configure(text=f"Estat: Ja consolidat ({len(con_files)} fitxers)",
                                             fg=COLORS["success"])
                self.consolidated_data = {"path": con_folder, "files": con_files}
                self.btn_calibrar.configure(state="normal")

                # Buscar PDFs generats a CHECK/
                check_folder = os.path.join(self.seq_path, "CHECK")
                pdf_report = glob.glob(os.path.join(check_folder, "REPORT_Consolidacio*.pdf"))
                pdf_chromato = glob.glob(os.path.join(check_folder, "REPORT_Cromatogrames*.pdf"))

                # Carregar resum de consolidació existent
                json_path = os.path.join(check_folder, "consolidation.json")
                if os.path.exists(json_path):
                    try:
                        with open(json_path, 'r', encoding='utf-8') as f:
                            summary = json.load(f)
                        self._show_consolidation_summary(summary)
                    except Exception as e:
                        pass  # Ignorar errors de lectura

                if pdf_report:
                    self.consolidated_data["pdf_report"] = pdf_report[0]
                    self.btn_open_report.configure(text="Obrir Informe PDF", state="normal",
                                                   bg=COLORS["secondary"], fg=COLORS["white"])
                    self.btn_open_report.pack(side=tk.LEFT, padx=(0, 5))
                else:
                    self.btn_open_report.configure(text="Generar Informe PDF", state="disabled",
                                                   bg=COLORS["light"], fg=COLORS["text_light"])
                    self.btn_open_report.pack(side=tk.LEFT, padx=(0, 5))

                if pdf_chromato:
                    self.consolidated_data["pdf_chromato"] = pdf_chromato[0]
                    self.btn_open_chromato.configure(text="Obrir Cromatogrames PDF", state="normal",
                                                     bg=COLORS["secondary"], fg=COLORS["white"])
                    self.btn_open_chromato.pack(side=tk.LEFT)
                else:
                    self.btn_open_chromato.configure(text="Generar Cromatogrames PDF", state="disabled",
                                                     bg=COLORS["light"], fg=COLORS["text_light"])
                    self.btn_open_chromato.pack(side=tk.LEFT)
            else:
                self.lbl_con_status.configure(text="Estat: Pendent de consolidar",
                                             fg=COLORS["dark"])
        else:
            self.lbl_con_status.configure(text="Estat: Pendent de consolidar",
                                         fg=COLORS["dark"])

    # =========================================================================
    # EXECUCIÓ: CONSOLIDAR
    # =========================================================================

    def _run_consolidation(self):
        """Executa la consolidació."""
        if self.is_processing or not self.seq_path:
            return

        self.is_processing = True
        self.btn_consolidar.configure(state="disabled")
        self.progress_con['value'] = 0
        self.lbl_con_status.configure(text="Estat: Consolidant...", fg=COLORS["dark"])
        self.lbl_con_progress.configure(text="Iniciant...")

        def consolidate_thread():
            try:
                def on_progress(pct, item):
                    """Callback: pct=percentatge (0-100), item=nom mostra"""
                    self.root.after(0, lambda: self._update_consolidation_progress(pct, item))

                result = consolidate_sequence(self.seq_path, progress_callback=on_progress)

                self.root.after(0, lambda: self._consolidation_done(result))
            except Exception as e:
                self.root.after(0, lambda: self._consolidation_error(str(e)))

        thread = threading.Thread(target=consolidate_thread, daemon=True)
        thread.start()

    def _update_consolidation_progress(self, pct, item):
        """Actualitza progrés de consolidació."""
        self.progress_con['value'] = pct
        self.lbl_con_progress.configure(text=f"Consolidant: {item}")

    def _consolidation_done(self, result):
        """Callback quan la consolidació acaba."""
        self.is_processing = False
        self.progress_con['value'] = 100
        self.lbl_con_progress.configure(text="")

        if result.get('success', False):
            n_processed = result.get('processed_count', 0)
            files = result.get('files', [])
            n_files = len(files) if files else n_processed
            mode = result.get('mode', 'N/A')
            is_bp = result.get('bp', False)

            # Si no s'ha processat res, intentar carregar dades existents
            if n_processed == 0:
                check_folder = os.path.join(self.seq_path, "CHECK")
                json_path = os.path.join(check_folder, "consolidation.json")
                if os.path.exists(json_path):
                    try:
                        with open(json_path, 'r', encoding='utf-8') as f:
                            existing = json.load(f)
                        # Actualitzar amb dades existents
                        mode = existing.get('meta', {}).get('mode', mode)
                        is_bp = existing.get('meta', {}).get('method', '') == 'BP'
                        n_files = existing.get('counts', {}).get('total_samples', 0)
                        # Guardar summary per mostrar
                        result['consolidation_summary'] = existing
                        self.lbl_con_status.configure(
                            text=f"Estat: Ja consolidat ({n_files} mostres)",
                            fg=COLORS["success"])
                    except Exception:
                        pass
                else:
                    self.lbl_con_status.configure(
                        text=f"Estat: Consolidat ({n_files} fitxers)",
                        fg=COLORS["success"])
            else:
                self.lbl_con_status.configure(
                    text=f"Estat: Consolidat ({n_files} fitxers)",
                    fg=COLORS["success"])

            mode_text = f"{mode}" + (" BP" if is_bp else " COLUMN")

            con_folder = result.get('output_path', os.path.join(self.seq_path, "Resultats_Consolidats"))
            con_files = glob.glob(os.path.join(con_folder, "*.xlsx"))
            con_files = [f for f in con_files if "~$" not in f]

            self.consolidated_data = {
                "path": con_folder,
                "files": con_files,
                "result": result
            }

            # Actualitzar mode a la GUI
            self.lbl_con_mode.configure(text=f"Mode: {mode_text}")

            # Buscar PDFs generats a CHECK/
            check_folder = os.path.join(self.seq_path, "CHECK")
            pdf_report = glob.glob(os.path.join(check_folder, "REPORT_Consolidacio*.pdf"))
            pdf_chromato = glob.glob(os.path.join(check_folder, "REPORT_Cromatogrames*.pdf"))

            if pdf_report:
                self.consolidated_data["pdf_report"] = pdf_report[0]
                self.btn_open_report.configure(text="Obrir Informe PDF", state="normal",
                                               bg=COLORS["secondary"], fg=COLORS["white"])
                self.btn_open_report.pack(side=tk.LEFT, padx=(0, 5))
            else:
                self.btn_open_report.configure(text="Generar Informe PDF", state="disabled",
                                               bg=COLORS["light"], fg=COLORS["text_light"])
                self.btn_open_report.pack(side=tk.LEFT, padx=(0, 5))

            if pdf_chromato:
                self.consolidated_data["pdf_chromato"] = pdf_chromato[0]
                self.btn_open_chromato.configure(text="Obrir Cromatogrames PDF", state="normal",
                                                 bg=COLORS["secondary"], fg=COLORS["white"])
                self.btn_open_chromato.pack(side=tk.LEFT)
            else:
                self.btn_open_chromato.configure(text="Generar Cromatogrames PDF", state="disabled",
                                                 bg=COLORS["light"], fg=COLORS["text_light"])
                self.btn_open_chromato.pack(side=tk.LEFT)

            # === MOSTRAR RESUM DE CONSOLIDACIÓ ===
            summary = result.get('consolidation_summary', {})
            if summary:
                self._show_consolidation_summary(summary)
            else:
                # Intentar llegir del JSON si existeix
                json_path = os.path.join(check_folder, "consolidation.json")
                if os.path.exists(json_path):
                    try:
                        with open(json_path, 'r', encoding='utf-8') as f:
                            summary = json.load(f)
                        self._show_consolidation_summary(summary)
                    except:
                        pass

            # === VERIFICAR SI CAL REVISIÓ MANUAL ===
            file_check = result.get('file_check', {})
            needs_review = file_check.get('needs_review', False)
            orphans = file_check.get('seq_orphan_files', [])
            low_conf = file_check.get('low_confidence_matches', [])

            if needs_review and (orphans or low_conf):
                # Mostrar diàleg de revisió
                valid_samples = set(result.get('valid_samples', []))
                review_dialog = FileAssignmentReviewDialog(
                    self.root, file_check, valid_samples, self.seq_path
                )

                if review_dialog.result:
                    action = review_dialog.result.get('action', 'CONTINUE')
                    if action == 'REPROCESS':
                        # Guardar assignacions i reprocessar
                        assignments = review_dialog.result.get('assignments', {})
                        messagebox.showinfo(
                            "Assignacions guardades",
                            f"S'han guardat {len(assignments)} assignacions manuals.\n"
                            "Cal reprocessar la consolidació per aplicar els canvis."
                        )
                        # TODO: Implementar reprocessament amb assignacions manuals
                    elif action == 'IGNORE_ALL':
                        pass  # Continuar normalment

            # Activar següent pas
            self.btn_calibrar.configure(state="normal")
            messagebox.showinfo("Consolidació", f"Consolidació completada!\n\n{n_files} fitxers generats.\nMode: {mode_text}")
        else:
            errors = result.get('errors', ['Error desconegut'])
            error_text = errors[0] if errors else 'Error desconegut'
            self.lbl_con_status.configure(text=f"Estat: Error - {error_text}",
                                         fg=COLORS["error"])
            self.btn_consolidar.configure(state="normal")

    def _show_consolidation_summary(self, summary):
        """Mostra el resum de consolidació a la GUI."""
        # Mostrar el frame (ja no usem before= perquè progress està a action bar)
        self.summary_frame.pack(fill=tk.X, pady=10)

        # Scroll per mostrar el resum
        self.root.after(100, lambda: self.canvas.yview_moveto(1.0))

        # Comptadors
        counts = summary.get('counts', {})
        total = counts.get('total_samples', 0)
        khp = counts.get('khp_samples', 0)
        ctrl = counts.get('control_samples', 0)
        reg = counts.get('regular_samples', 0)
        self.lbl_sum_samples.configure(text=f"Injeccions: {total} ({reg} mostres, {khp} KHP, {ctrl} controls)")

        # Alineació
        alignment = summary.get('alignment', {})
        if alignment:
            shift_uib = alignment.get('shift_uib', 0) * 60 if alignment.get('shift_uib') else 0
            shift_direct = alignment.get('shift_direct', 0) * 60 if alignment.get('shift_direct') else 0
            source = alignment.get('source', 'N/A')
            self.lbl_sum_alignment.configure(
                text=f"Alineació: UIB={shift_uib:.1f}s, Direct={shift_direct:.1f}s ({source})")
        else:
            self.lbl_sum_alignment.configure(text="Alineació: No aplicada")

        # SNR
        quality = summary.get('quality', {})
        snr_direct = quality.get('snr_direct', {})
        if snr_direct:
            self.lbl_sum_snr.configure(
                text=f"SNR Direct: mediana={snr_direct.get('median', '-')}, min={snr_direct.get('min', '-')}")
        else:
            self.lbl_sum_snr.configure(text="SNR: -")

        # LOD
        lod_direct = quality.get('lod_direct_mau')
        lod_uib = quality.get('lod_uib_mau')
        if lod_direct:
            lod_text = f"LOD: Direct={lod_direct:.2f} mAU"
            if lod_uib:
                lod_text += f", UIB={lod_uib:.2f} mAU"
            self.lbl_sum_lod.configure(text=lod_text)
        else:
            self.lbl_sum_lod.configure(text="LOD: -")

        # Data generació i versió
        meta = summary.get('meta', {})
        generated_at = meta.get('generated_at', '')
        script_version = meta.get('script_version', '')
        if generated_at:
            # Format: 2026-01-28T12:01:34.109120 -> 28/01/2026 12:01
            try:
                dt = datetime.fromisoformat(generated_at)
                date_str = dt.strftime("%d/%m/%Y %H:%M")
            except:
                date_str = generated_at[:16] if len(generated_at) > 16 else generated_at
            gen_text = f"Generat: {date_str}"
            if script_version:
                gen_text += f" (v{script_version})"
            self.lbl_sum_generated.configure(text=gen_text)
        else:
            self.lbl_sum_generated.configure(text="")

        # Timeouts
        timeouts = summary.get('timeouts', {})
        sev_counts = timeouts.get('severity_counts', {})

        self.lbl_timeout_ok.configure(text=f"OK: {sev_counts.get('OK', 0)}")
        self.lbl_timeout_info.configure(text=f"INFO: {sev_counts.get('INFO', 0)}")
        self.lbl_timeout_warn.configure(text=f"WARN: {sev_counts.get('WARNING', 0)}")
        self.lbl_timeout_crit.configure(text=f"CRIT: {sev_counts.get('CRITICAL', 0)}")

        # Mostres crítiques
        critical = timeouts.get('critical_samples', [])
        if critical:
            self.lbl_sum_critical.configure(text=f"Mostres crítiques: {', '.join(critical[:5])}")
        else:
            self.lbl_sum_critical.configure(text="")

        # Warnings
        warnings = summary.get('warnings', [])
        if warnings:
            # Filtrar només els més importants
            important = [w for w in warnings if not w.startswith("PDF generat")][:3]
            if important:
                self.lbl_sum_warnings.configure(text="Avisos: " + " | ".join(important))
            else:
                self.lbl_sum_warnings.configure(text="")
        else:
            self.lbl_sum_warnings.configure(text="")

    def _consolidation_error(self, error):
        """Callback quan hi ha error de consolidació."""
        self.is_processing = False
        self.lbl_con_status.configure(text=f"✗ Error: {error}", fg=COLORS["error"])
        self.btn_consolidar.configure(state="normal")
        messagebox.showerror("Error", f"Error durant la consolidació:\n\n{error}")

    def _open_seq_folder(self):
        """Obre la carpeta SEQ actual."""
        if self.seq_path and os.path.isdir(self.seq_path):
            self._open_path(self.seq_path)

    def _open_consolidation_report(self):
        """Obre el PDF d'informe de consolidació."""
        pdf_path = self.consolidated_data.get("pdf_report")
        if pdf_path and os.path.exists(pdf_path):
            self._open_path(pdf_path)
        else:
            messagebox.showinfo("PDF no trobat", "No s'ha trobat l'informe de consolidació.")

    def _open_chromatograms_report(self):
        """Obre el PDF de cromatogrames."""
        pdf_path = self.consolidated_data.get("pdf_chromato")
        if pdf_path and os.path.exists(pdf_path):
            self._open_path(pdf_path)
        else:
            messagebox.showinfo("PDF no trobat", "No s'ha trobat el PDF de cromatogrames.")

    def _open_path(self, path):
        """Obre un path amb l'aplicació per defecte del sistema."""
        import platform
        import subprocess
        try:
            if platform.system() == 'Windows':
                os.startfile(path)
            elif platform.system() == 'Darwin':
                subprocess.run(['open', path])
            else:
                subprocess.run(['xdg-open', path])
        except Exception as e:
            messagebox.showerror("Error", f"No s'ha pogut obrir:\n{path}\n\n{e}")

    # =========================================================================
    # EXECUCIÓ: CALIBRAR
    # =========================================================================

    def _run_calibration(self):
        """Executa la calibració."""
        if self.is_processing or not self.seq_path:
            return

        self.is_processing = True
        self.btn_calibrar.configure(state="disabled")

        def calibrate_thread():
            try:
                result = calibrate_sequence(self.seq_path)
                self.root.after(0, lambda: self._calibration_done(result))
            except Exception as e:
                self.root.after(0, lambda: self._calibration_error(str(e)))

        thread = threading.Thread(target=calibrate_thread, daemon=True)
        thread.start()

    def _calibration_done(self, result):
        """Callback quan la calibració acaba."""
        self.is_processing = False

        if result.get('success', False):
            khp_data = result.get('khp_data', {})
            calibration = result.get('calibration', {})
            khp_source = result.get('khp_source', 'LOCAL')

            # Extreure dades
            factor = calibration.get('factor', 0)
            conc = khp_data.get('conc_ppm', 0)
            area = khp_data.get('area', 0)
            n_replicas = khp_data.get('n_replicas', 1)
            rsd = khp_data.get('rsd', 0)
            shift_sec = khp_data.get('shift_sec', 0)
            symmetry = khp_data.get('symmetry', 1.0)
            snr = khp_data.get('snr', 0)
            quality_issues = khp_data.get('quality_issues', [])
            quality_score = khp_data.get('quality_score', 0)
            status = khp_data.get('status', 'OK')
            khp_filename = khp_data.get('filename', 'N/A')

            # Actualitzar etiquetes - Info KHP
            self.lbl_cal_source.configure(text=f"Origen: {khp_source}")
            self.lbl_cal_khp.configure(text=f"KHP: {khp_filename}")
            self.lbl_cal_conc.configure(text=f"Concentració: {conc} ppm")
            self.lbl_cal_replicas.configure(text=f"Rèpliques: {n_replicas}")

            # Actualitzar etiquetes - Resultats
            self.lbl_cal_factor.configure(text=f"Factor: {factor:.6f}")
            self.lbl_cal_area.configure(text=f"Àrea: {area:.2f}")

            if n_replicas > 1:
                self.lbl_cal_rsd.configure(text=f"RSD: {rsd:.1f}%")
            else:
                self.lbl_cal_rsd.configure(text="RSD: - (única rèplica)")

            self.lbl_cal_shift.configure(text=f"Shift DOC-DAD: {shift_sec:.1f} s")

            # Actualitzar etiquetes - Qualitat
            self.lbl_cal_symmetry.configure(text=f"Simetria: {symmetry:.2f}")
            self.lbl_cal_snr.configure(text=f"SNR: {snr:.1f}")

            # Estat de qualitat amb colors
            if quality_score < 20:
                self.lbl_cal_quality.configure(text="Estat: ✓ EXCEL·LENT", fg=COLORS["success"])
            elif quality_score < 50:
                self.lbl_cal_quality.configure(text="Estat: ✓ BO", fg=COLORS["success"])
            elif quality_score < 100:
                self.lbl_cal_quality.configure(text="Estat: ⚠ REVISAR", fg=COLORS["warning"])
            else:
                self.lbl_cal_quality.configure(text="Estat: ✗ PROBLEMES", fg=COLORS["error"])

            # Mostrar issues si n'hi ha
            if quality_issues:
                issues_text = " | ".join(quality_issues[:3])  # Màxim 3
                self.lbl_cal_issues.configure(text=issues_text)
            else:
                self.lbl_cal_issues.configure(text="")

            self.calibration_data = result

            # Activar següent pas
            self.btn_qc.configure(state="normal")

            # Mostrar gràfic
            self._plot_calibration(result)

            # Mostrar botó outlier
            self.btn_outlier.pack(side=tk.LEFT, padx=10)
        else:
            errors = result.get('errors', ['Error desconegut'])
            self.lbl_cal_factor.configure(text=f"Error: {errors[0]}", fg=COLORS["error"])
            self.btn_calibrar.configure(state="normal")

    def _calibration_error(self, error):
        """Callback quan hi ha error de calibració."""
        self.is_processing = False
        self.btn_calibrar.configure(state="normal")
        messagebox.showerror("Error", f"Error durant la calibració:\n\n{error}")

    def _plot_calibration(self, result):
        """Mostra gràfic de calibració amb cromatograma DOC."""
        # Netejar canvas anterior
        for widget in self.cal_canvas_frame.winfo_children():
            widget.destroy()

        khp_data = result.get('khp_data', {})
        calibration = result.get('calibration', {})

        # Obtenir dades del cromatograma
        t_doc = khp_data.get('t_doc')
        y_doc = khp_data.get('y_doc')
        t_dad = khp_data.get('t_dad')
        y_dad = khp_data.get('y_dad_254')
        peak_info = khp_data.get('peak_info', {})

        has_dad = t_dad is not None and y_dad is not None and len(t_dad) > 0

        # Crear figura amb subplots
        if has_dad:
            fig = Figure(figsize=(8, 4), dpi=100)
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212)
        else:
            fig = Figure(figsize=(8, 3), dpi=100)
            ax1 = fig.add_subplot(111)
            ax2 = None

        # Plot DOC
        if t_doc is not None and y_doc is not None and len(t_doc) > 0:
            ax1.plot(t_doc, y_doc, color=COLORS["primary"], linewidth=1.2, label='DOC')

            # Marcar àrea del pic
            left_idx = khp_data.get('peak_left_idx', peak_info.get('left_idx', 0))
            right_idx = khp_data.get('peak_right_idx', peak_info.get('right_idx', len(y_doc)-1))

            if 0 <= left_idx < len(t_doc) and 0 <= right_idx < len(t_doc):
                ax1.fill_between(t_doc[left_idx:right_idx+1], y_doc[left_idx:right_idx+1],
                                alpha=0.3, color=COLORS["success"], label='Àrea integrada')

                # Línies verticals als límits
                ax1.axvline(t_doc[left_idx], color=COLORS["success"], linestyle='--', alpha=0.5)
                ax1.axvline(t_doc[right_idx], color=COLORS["success"], linestyle='--', alpha=0.5)

            # Marcar pic principal
            t_max = peak_info.get('t_max', khp_data.get('t_doc_max', 0))
            if t_max:
                peak_idx = np.argmin(np.abs(t_doc - t_max))
                ax1.plot(t_max, y_doc[peak_idx], 'o', color=COLORS["error"],
                        markersize=8, label=f'Pic: {t_max:.2f} min')

            # Info al gràfic
            factor = calibration.get('factor', 0)
            area = khp_data.get('area', 0)
            conc = khp_data.get('conc_ppm', 0)

            info_text = f"Factor: {factor:.6f}  |  Àrea: {area:.1f}  |  {conc} ppm"
            ax1.set_title(f"Cromatograma KHP - {info_text}", fontsize=10)
            ax1.set_xlabel("Temps (min)")
            ax1.set_ylabel("Senyal DOC (mV)")
            ax1.legend(loc='upper right', fontsize=8)
            ax1.grid(True, alpha=0.3)

        # Plot DAD 254nm
        if has_dad and ax2 is not None:
            ax2.plot(t_dad, y_dad, color=COLORS["secondary"], linewidth=1.0, label='DAD 254nm')

            # Marcar shift
            shift_sec = khp_data.get('shift_sec', 0)
            t_doc_max = khp_data.get('t_doc_max', 0)
            t_dad_max = khp_data.get('t_dad_max', 0)

            if t_dad_max:
                dad_peak_idx = np.argmin(np.abs(t_dad - t_dad_max))
                ax2.plot(t_dad_max, y_dad[dad_peak_idx], 'o', color=COLORS["error"],
                        markersize=6, label=f'Pic DAD: {t_dad_max:.2f} min')

            ax2.set_title(f"DAD 254nm - Shift: {shift_sec:.1f} s", fontsize=10)
            ax2.set_xlabel("Temps (min)")
            ax2.set_ylabel("Absorbància (mAU)")
            ax2.legend(loc='upper right', fontsize=8)
            ax2.grid(True, alpha=0.3)

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.cal_canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _show_khp_history(self):
        """Mostra l'històric de KHP."""
        # TODO: Implementar diàleg d'històric KHP
        messagebox.showinfo("Històric KHP", "Funcionalitat pendent d'implementar.")

    def _mark_as_outlier(self):
        """Marca la calibració actual com a outlier."""
        if messagebox.askyesno("Confirmar", "Vols marcar aquesta calibració com a outlier?"):
            # TODO: Implementar marcar outlier
            messagebox.showinfo("Outlier", "Calibració marcada com a outlier.")

    # =========================================================================
    # EXECUCIÓ: QC
    # =========================================================================

    def _run_qc(self):
        """Executa el Quality Control."""
        if self.is_processing or not self.seq_path:
            return

        self.is_processing = True
        self.btn_qc.configure(state="disabled")
        self.progress_qc['value'] = 0

        # Netejar taula
        for item in self.tree_qc.get_children():
            self.tree_qc.delete(item)

        def qc_thread():
            try:
                self._run_qc_process()
            except Exception as e:
                import traceback
                self.root.after(0, lambda: self._qc_error(str(e) + "\n" + traceback.format_exc()))

        thread = threading.Thread(target=qc_thread, daemon=True)
        thread.start()

    def _run_qc_process(self):
        """Procés de QC (en thread separat)."""
        con_path = self.consolidated_data.get("path", "")
        if not con_path:
            con_path = os.path.join(self.seq_path, "Resultats_Consolidats")

        xlsx_files = glob.glob(os.path.join(con_path, "*.xlsx"))
        xlsx_files = [f for f in xlsx_files if "~$" not in f]

        # Agrupar per mostra
        samples = {}
        for f in xlsx_files:
            name = os.path.basename(f).replace(".xlsx", "")
            match = re.search(r"(.+)_R(\d+)$", name)
            if match:
                base = match.group(1)
                rep = match.group(2)
                if base not in samples:
                    samples[base] = {}
                samples[base][rep] = f

        total = len(samples)
        mode = detect_mode(self.seq_path)

        for i, (sample_name, reps) in enumerate(samples.items()):
            pct = int(100 * (i + 1) / total)
            self.root.after(0, lambda p=pct, s=sample_name: self._update_qc_progress(p, s))

            # Avaluar cada rèplica
            evals = {}
            for rep_id, filepath in reps.items():
                try:
                    t, y = self._read_doc_from_file(filepath)
                    evaluation = evaluate_replica(t, y, method=mode)
                    evaluation['filepath'] = filepath
                    evals[rep_id] = evaluation
                except Exception as e:
                    evals[rep_id] = {'valid': False, 'error': str(e), 'filepath': filepath}

            # Comparar rèpliques
            doc_r, doc_diff = None, None
            if len(evals) >= 2:
                rep_ids = sorted(evals.keys())
                if evals[rep_ids[0]].get('valid') and evals[rep_ids[1]].get('valid'):
                    try:
                        t1, y1 = self._read_doc_from_file(evals[rep_ids[0]]['filepath'])
                        t2, y2 = self._read_doc_from_file(evals[rep_ids[1]]['filepath'])
                        comparison = compare_replicas(t1, y1, t2, y2)
                        doc_r = comparison.get('pearson')
                        doc_diff = comparison.get('area_diff_pct')
                    except:
                        pass

            # Seleccionar millor
            sel_doc = "1"
            sel_dad = "1"

            if len(evals) >= 2:
                rep_ids = sorted(evals.keys())
                e1, e2 = evals.get(rep_ids[0], {}), evals.get(rep_ids[1], {})

                if e1.get('valid') and e2.get('valid'):
                    selection = select_best_replica(e1, e2, method=mode)
                    sel_doc = selection.get('best', '1')
                elif e1.get('valid'):
                    sel_doc = rep_ids[0]
                elif e2.get('valid'):
                    sel_doc = rep_ids[1]

            sel_dad = sel_doc  # Per ara, mateixa selecció

            # Determinar estat
            if doc_r is None:
                estat = "?"
                tag = "WARN"
            elif doc_r >= DEFAULT_MIN_CORR and (doc_diff is None or doc_diff <= DEFAULT_MAX_DIFF):
                estat = "OK"
                tag = "OK"
            elif doc_r >= 0.99:
                estat = "CHECK"
                tag = "WARN"
            else:
                estat = "FAIL"
                tag = "FAIL"

            # Formatar valors
            doc_r_str = f"{doc_r:.3f}" if doc_r is not None else "-"
            doc_diff_str = f"{doc_diff:.1f}%" if doc_diff is not None else "-"

            values = (sample_name, estat, doc_r_str, doc_diff_str,
                     f"R{sel_doc}", "-", f"R{sel_dad}")

            # Guardar resultats
            self.selected_replicas[sample_name] = {
                "sel_doc": sel_doc,
                "sel_dad": sel_dad,
                "reps": reps,
                "evals": evals,
                "doc_r": doc_r,
                "doc_diff": doc_diff,
            }

            # Actualitzar taula
            self.root.after(0, lambda v=values, t=tag: self._add_qc_row(v, t))

        self.root.after(0, self._qc_done)

    def _read_doc_from_file(self, filepath):
        """Llegeix dades DOC d'un fitxer consolidat."""
        try:
            xls = pd.ExcelFile(filepath)

            # Buscar full DOC
            for sheet in ['DOC_Direct', 'DOC_UIB', 'DOC', 'ID']:
                if sheet in xls.sheet_names:
                    df = pd.read_excel(xls, sheet)

                    # Buscar columnes temps i senyal
                    t_col = None
                    y_col = None

                    for col in df.columns:
                        col_lower = str(col).lower()
                        if 'time' in col_lower or 'min' in col_lower or col_lower == 't':
                            t_col = col
                        elif 'doc' in col_lower or 'signal' in col_lower or 'net' in col_lower:
                            y_col = col

                    if t_col and y_col:
                        t = df[t_col].values
                        y = df[y_col].values
                        return np.asarray(t, dtype=float), np.asarray(y, dtype=float)

            return np.array([]), np.array([])
        except Exception as e:
            return np.array([]), np.array([])

    def _update_qc_progress(self, pct, sample):
        """Actualitza progrés de QC."""
        self.progress_qc['value'] = pct

    def _add_qc_row(self, values, tag):
        """Afegeix una fila a la taula QC."""
        self.tree_qc.insert("", "end", values=values, tags=(tag,))

    def _qc_done(self):
        """Callback quan el QC acaba."""
        self.is_processing = False
        self.progress_qc['value'] = 100
        self.btn_export.configure(state="normal")

        n_ok = sum(1 for k, v in self.selected_replicas.items()
                   if v.get('doc_r') and v['doc_r'] >= DEFAULT_MIN_CORR)
        total = len(self.selected_replicas)

        messagebox.showinfo("QC Completat",
                           f"Quality Control completat!\n\n"
                           f"Mostres: {total}\n"
                           f"OK: {n_ok}\n"
                           f"Revisar: {total - n_ok}")

    def _qc_error(self, error):
        """Callback quan hi ha error de QC."""
        self.is_processing = False
        self.btn_qc.configure(state="normal")
        messagebox.showerror("Error", f"Error durant el QC:\n\n{error}")

    def _on_qc_double_click(self, event):
        """Doble clic a la taula QC."""
        self._open_qaqc_plot()

    def _on_qc_right_click(self, event):
        """Clic dret a la taula QC."""
        item = self.tree_qc.identify_row(event.y)
        if item:
            self.tree_qc.selection_set(item)
            self.qc_menu.post(event.x_root, event.y_root)

    def _force_selection(self, signal_type, replica):
        """Força la selecció d'una rèplica."""
        selection = self.tree_qc.selection()
        if not selection:
            return

        item = selection[0]
        values = list(self.tree_qc.item(item, 'values'))
        sample_name = values[0]

        if signal_type == "DOC":
            values[4] = f"R{replica}"
            if sample_name in self.selected_replicas:
                self.selected_replicas[sample_name]['sel_doc'] = replica
        else:
            values[6] = f"R{replica}"
            if sample_name in self.selected_replicas:
                self.selected_replicas[sample_name]['sel_dad'] = replica

        self.tree_qc.item(item, values=values, tags=('MODIFIED',))

    def _open_qaqc_plot(self):
        """Obre el gràfic QAQC de la mostra seleccionada."""
        selection = self.tree_qc.selection()
        if not selection:
            return

        values = self.tree_qc.item(selection[0], 'values')
        sample_name = values[0]

        check_folder = os.path.join(self.seq_path, "CHECK")

        # Buscar gràfic
        for pattern in [f"{sample_name}_QAQC.png", f"{sample_name}.png"]:
            path = os.path.join(check_folder, pattern)
            if os.path.exists(path):
                import platform
                import subprocess
                if platform.system() == 'Windows':
                    os.startfile(path)
                elif platform.system() == 'Darwin':
                    subprocess.run(['open', path])
                else:
                    subprocess.run(['xdg-open', path])
                return

        messagebox.showinfo("Gràfic no trobat",
                           f"No s'ha trobat el gràfic QAQC per {sample_name}.\n\n"
                           f"Exporta primer els resultats per generar els gràfics.")

    # =========================================================================
    # EXECUCIÓ: EXPORTAR
    # =========================================================================

    def _run_export(self):
        """Executa l'exportació."""
        if self.is_processing or not self.seq_path:
            return

        self.is_processing = True
        self.btn_export.configure(state="disabled")
        self.progress_export['value'] = 0

        def export_thread():
            try:
                self._run_export_process()
            except Exception as e:
                self.root.after(0, lambda: self._export_error(str(e)))

        thread = threading.Thread(target=export_thread, daemon=True)
        thread.start()

    def _run_export_process(self):
        """Procés d'exportació (en thread separat)."""
        check_folder = ensure_check_folder(self.seq_path)

        total_steps = 3
        current = 0

        # 1. Guardar resum JSON
        current += 1
        self.root.after(0, lambda: self._update_export_progress(int(100*current/total_steps), "Guardant resum..."))

        summary = {
            "seq_path": self.seq_path,
            "seq_name": os.path.basename(self.seq_path),
            "generated_at": datetime.now().isoformat(),
            "calibration": self.calibration_data,
            "samples": []
        }

        for name, data in self.selected_replicas.items():
            summary["samples"].append({
                "name": name,
                "sel_doc": data.get('sel_doc'),
                "sel_dad": data.get('sel_dad'),
                "doc_r": data.get('doc_r'),
                "doc_diff": data.get('doc_diff'),
            })

        summary_path = os.path.join(check_folder, "processing_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

        # 2. Exportar Excel
        if self.var_export_excel.get():
            current += 1
            self.root.after(0, lambda: self._update_export_progress(int(100*current/total_steps), "Generant Excel..."))

            excel_path = os.path.join(check_folder, "replica_selections.xlsx")
            self._export_excel(excel_path)

        # 3. Copiar seleccionats
        if self.var_export_selected.get():
            current += 1
            self.root.after(0, lambda: self._update_export_progress(int(100*current/total_steps), "Copiant fitxers..."))

            selected_folder = os.path.join(check_folder, "Selected")
            os.makedirs(selected_folder, exist_ok=True)

            import shutil
            for name, data in self.selected_replicas.items():
                sel = data.get('sel_doc', '1')
                reps = data.get('reps', {})
                if sel in reps:
                    src = reps[sel]
                    if os.path.exists(src):
                        dst = os.path.join(selected_folder, os.path.basename(src))
                        shutil.copy2(src, dst)

        self.root.after(0, lambda: self._export_done(check_folder))

    def _update_export_progress(self, pct, msg):
        """Actualitza progrés d'exportació."""
        self.progress_export['value'] = pct
        self.export_summary.configure(text=msg)

    def _export_excel(self, path):
        """Exporta resultats a Excel."""
        rows = []
        for name, data in self.selected_replicas.items():
            rows.append({
                "Mostra": name,
                "Sel_DOC": f"R{data.get('sel_doc', '?')}",
                "Sel_DAD": f"R{data.get('sel_dad', '?')}",
                "Pearson_R": data.get('doc_r'),
                "Diff_Pct": data.get('doc_diff'),
            })

        df = pd.DataFrame(rows)
        df.to_excel(path, index=False)

    def _export_done(self, check_folder):
        """Callback quan l'exportació acaba."""
        self.is_processing = False
        self.progress_export['value'] = 100
        self.btn_open_folder.configure(state="normal")

        self.export_summary.configure(
            text=f"✓ Exportació completada!\n\nFitxers guardats a:\n{check_folder}"
        )

        messagebox.showinfo("Exportació", "Exportació completada correctament!")

    def _export_error(self, error):
        """Callback quan hi ha error d'exportació."""
        self.is_processing = False
        self.btn_export.configure(state="normal")
        messagebox.showerror("Error", f"Error durant l'exportació:\n\n{error}")

    def _open_check_folder(self):
        """Obre la carpeta CHECK."""
        check_folder = os.path.join(self.seq_path, "CHECK")
        if os.path.isdir(check_folder):
            import platform
            import subprocess
            if platform.system() == 'Windows':
                os.startfile(check_folder)
            elif platform.system() == 'Darwin':
                subprocess.run(['open', check_folder])
            else:
                subprocess.run(['xdg-open', check_folder])

    # =========================================================================
    # FUNCIONS AUXILIARS
    # =========================================================================

    def _open_config(self):
        """Obre el modal de configuració."""
        # TODO: Implementar modal config
        messagebox.showinfo("Configuració", "Funcionalitat pendent d'implementar.")

    def _open_repositori(self):
        """Obre la finestra de repositori."""
        # TODO: Implementar finestra repositori
        messagebox.showinfo("Repositori", "Funcionalitat pendent d'implementar.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    root = tk.Tk()
    app = HPSECSuiteV3(root)
    root.mainloop()


if __name__ == "__main__":
    main()
