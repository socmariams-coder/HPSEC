"""
HPSEC Suite v3.1
================
Aplicació unificada per al processament de dades HPSEC.

Basat en v1 (wizard) + backend modular de 5 fases.

Pipeline (5 fases):
1. Importar  - Llegir fitxers font, aparellar mostres (hpsec_import.py)
2. Calibrar  - Validació KHP, càlcul factor calibració (hpsec_calibrate.py)
3. Processar - Alineació, baseline, àrees, SNR (hpsec_process.py)
4. Revisar   - Comparar rèpliques, seleccionar millor (hpsec_review.py)
5. Exportar  - Generar fitxers finals i informes PDF

Eines:
- Planificador de seqüències (hpsec_planner_gui.py)
- Configuració (modal)

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
    set_calibration_override,
    load_local_calibrations,
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
VERSION = "3.1.0"

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
        self.steps = ["Importar", "Calibrar", "Processar", "Revisar", "Exportar"]

        # Dades
        self.consolidated_data = {}
        self.calibration_data = {}
        self.qc_results = {}
        self.selected_replicas = {}
        self.khp_area = None  # Àrea KHP per càlcul concentració
        self.khp_conc = None  # Concentració KHP (ppm)
        self.khp_source_seq = None  # SEQ origen del KHP usat per calibració

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
        btn_config = tk.Button(header, text="[=]",
                              command=self._open_config,
                              bg=COLORS["white"], fg=COLORS["primary"],
                              font=("Segoe UI", 10),
                              relief="flat", width=3)
        btn_config.pack(side=tk.RIGHT, padx=10, pady=15)

        # Menú Eines (desplegable)
        self.btn_eines = tk.Menubutton(header, text="Eines",
                                       bg=COLORS["secondary"], fg=COLORS["white"],
                                       font=("Segoe UI", 9, "bold"),
                                       relief="flat", padx=12, pady=5)
        self.btn_eines.pack(side=tk.RIGHT, padx=5, pady=15)

        self.menu_eines = tk.Menu(self.btn_eines, tearoff=0)
        self.btn_eines.configure(menu=self.menu_eines)
        self.menu_eines.add_command(label="Planificador Sequencies", command=self._open_planner)
        self.menu_eines.add_command(label="Comparador UIB/Direct", command=self._open_comparador)
        self.menu_eines.add_separator()
        self.menu_eines.add_command(label="Repositori KHP", command=self._open_repositori)

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

        # Pas 1: Importar
        frame1 = tk.Frame(btn_container, bg=COLORS["white"])
        self.action_frames.append(frame1)

        self.progress_import = ttk.Progressbar(frame1, mode='determinate', length=300)
        self.progress_import.pack(side=tk.LEFT, padx=(20, 10))

        self.lbl_import_progress = tk.Label(frame1, text="",
                                            font=("Segoe UI", 9),
                                            bg=COLORS["white"], fg=COLORS["text_light"],
                                            width=30)
        self.lbl_import_progress.pack(side=tk.LEFT, padx=10)

        self.btn_importar = tk.Button(frame1, text="▶ Importar Dades",
                                      command=self._run_import,
                                      bg=COLORS["primary"], fg=COLORS["white"],
                                      font=("Segoe UI", 11, "bold"),
                                      relief="flat", padx=25, pady=8,
                                      state="disabled")
        self.btn_importar.pack(side=tk.RIGHT, padx=20)

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

        # Pas 3: Processar
        frame3 = tk.Frame(btn_container, bg=COLORS["white"])
        self.action_frames.append(frame3)

        self.progress_process = ttk.Progressbar(frame3, mode='determinate', length=300)
        self.progress_process.pack(side=tk.LEFT, padx=(20, 10))

        self.btn_processar = tk.Button(frame3, text="▶ Processar",
                                       command=self._run_process,
                                       bg=COLORS["primary"], fg=COLORS["white"],
                                       font=("Segoe UI", 11, "bold"),
                                       relief="flat", padx=25, pady=8,
                                       state="disabled")
        self.btn_processar.pack(side=tk.RIGHT, padx=20)

        # Pas 4: Revisar
        frame4 = tk.Frame(btn_container, bg=COLORS["white"])
        self.action_frames.append(frame4)

        self.btn_revisar = tk.Button(frame4, text="▶ Revisar Repliques",
                                     command=self._run_review,
                                     bg=COLORS["primary"], fg=COLORS["white"],
                                     font=("Segoe UI", 11, "bold"),
                                     relief="flat", padx=25, pady=8,
                                     state="disabled")
        self.btn_revisar.pack(side=tk.RIGHT, padx=20)

        # Pas 5: Exportar
        frame5 = tk.Frame(btn_container, bg=COLORS["white"])
        self.action_frames.append(frame5)

        self.btn_export = tk.Button(frame5, text="▶ Exportar Resultats",
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

        # Un frame per cada pas (5 fases)
        self.step_frames = []
        for i in range(5):
            frame = tk.Frame(self.main_frame, bg=COLORS["white"])
            self.step_frames.append(frame)

        # Construir contingut de cada pas
        self._build_step_importar(self.step_frames[0])
        self._build_step_calibrar(self.step_frames[1])
        self._build_step_processar(self.step_frames[2])
        self._build_step_revisar(self.step_frames[3])
        self._build_step_exportar(self.step_frames[4])

    def _on_canvas_configure(self, event):
        """Ajusta l'amplada del frame interior al canvas."""
        self.canvas.itemconfig(self.canvas_window, width=event.width)

    def _on_mousewheel(self, event):
        """Scroll amb roda del ratolí."""
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    # =========================================================================
    # PAS 1: CONSOLIDAR
    # =========================================================================

    def _build_step_importar(self, parent):
        """Construeix el pas d'importació (Fase 1) - Disseny unificat amb taula central."""

        # === HEADER ===
        header = tk.Frame(parent, bg=COLORS["white"])
        header.pack(fill=tk.X, pady=(0, 5))

        tk.Label(header, text="Pas 1: Importar",
                font=("Segoe UI", 13, "bold"),
                bg=COLORS["white"], fg=COLORS["primary"]).pack(side=tk.LEFT)

        self.btn_open_seq_folder = tk.Button(header, text="Obrir Carpeta",
                                             command=self._open_seq_folder,
                                             bg=COLORS["light"], fg=COLORS["dark"],
                                             font=("Segoe UI", 8),
                                             relief="flat", padx=8, pady=2)
        self.btn_open_seq_folder.pack(side=tk.RIGHT)

        # === FILA SUPERIOR: SEQÜÈNCIA | KHP ===
        top_row = tk.Frame(parent, bg=COLORS["white"])
        top_row.pack(fill=tk.X, pady=(0, 5))
        top_row.columnconfigure(0, weight=1)
        top_row.columnconfigure(1, weight=1)

        # -- Columna esquerra: SEQÜÈNCIA --
        seq_frame = tk.LabelFrame(top_row, text="SEQÜÈNCIA", font=("Segoe UI", 9, "bold"),
                                  bg=COLORS["white"], fg=COLORS["dark"])
        seq_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 3), pady=0)

        # Línia 1: SEQ · Mode
        self.lbl_seq_header = tk.Label(seq_frame, text="- · -",
                                       font=("Segoe UI", 10, "bold"),
                                       bg=COLORS["white"], fg=COLORS["primary"])
        self.lbl_seq_header.pack(anchor="w", padx=8, pady=(4, 2))

        # Info en línia: Mode dades | UIB | Volum
        info_line = tk.Frame(seq_frame, bg=COLORS["white"])
        info_line.pack(fill=tk.X, padx=8, pady=2)

        self.lbl_data_mode = tk.Label(info_line, text="Dades: -",
                                      font=("Segoe UI", 8), bg=COLORS["white"])
        self.lbl_data_mode.pack(side=tk.LEFT, padx=(0, 10))

        self.lbl_uib_sens = tk.Label(info_line, text="UIB: -",
                                     font=("Segoe UI", 8), bg=COLORS["white"])
        self.lbl_uib_sens.pack(side=tk.LEFT, padx=(0, 10))

        self.lbl_inj_vol = tk.Label(info_line, text="Vol: -",
                                    font=("Segoe UI", 8), bg=COLORS["white"])
        self.lbl_inj_vol.pack(side=tk.LEFT)

        # Comptadors exactes: x mostres, y KHP, z control
        self.lbl_samples_count = tk.Label(seq_frame, text="Mostres: -",
                                          font=("Segoe UI", 9), bg=COLORS["white"])
        self.lbl_samples_count.pack(anchor="w", padx=8, pady=2)

        # Estat + MasterFile
        status_line = tk.Frame(seq_frame, bg=COLORS["white"])
        status_line.pack(fill=tk.X, padx=8, pady=(0, 4))

        self.lbl_con_status = tk.Label(status_line, text="Pendent",
                                       font=("Segoe UI", 8), bg=COLORS["white"],
                                       fg=COLORS["text_light"])
        self.lbl_con_status.pack(side=tk.LEFT)

        self.lbl_con_master = tk.Label(status_line, text="",
                                       font=("Segoe UI", 8),
                                       bg=COLORS["white"], fg=COLORS["text_light"])
        self.lbl_con_master.pack(side=tk.RIGHT)

        # -- Columna dreta: KHP DETECTATS --
        khp_frame = tk.LabelFrame(top_row, text="KHP DETECTATS", font=("Segoe UI", 9, "bold"),
                                  bg=COLORS["white"], fg=COLORS["dark"])
        khp_frame.grid(row=0, column=1, sticky="nsew", padx=(3, 0), pady=0)

        # Header KHP
        self.lbl_khp_header = tk.Label(khp_frame, text="Cap KHP trobat",
                                       font=("Segoe UI", 9),
                                       bg=COLORS["white"], fg=COLORS["text_light"])
        self.lbl_khp_header.pack(anchor="w", padx=8, pady=(4, 2))

        # Taula KHP (Treeview compacte)
        khp_table_frame = tk.Frame(khp_frame, bg=COLORS["white"])
        khp_table_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=2)

        khp_cols = ("rep", "area_doc", "a254", "ratio", "snr", "status")
        self.tree_khp = ttk.Treeview(khp_table_frame, columns=khp_cols, show='headings', height=2)

        col_widths = {"rep": 30, "area_doc": 65, "a254": 50, "ratio": 50, "snr": 45, "status": 30}
        col_texts = {"rep": "", "area_doc": "DOC", "a254": "A254", "ratio": "Ratio", "snr": "SNR", "status": ""}
        for col in khp_cols:
            self.tree_khp.heading(col, text=col_texts[col])
            self.tree_khp.column(col, width=col_widths[col], anchor="center", stretch=False)

        self.tree_khp.tag_configure('ok', foreground='#006100')
        self.tree_khp.tag_configure('warn', foreground='#856404')
        self.tree_khp.tag_configure('fail', foreground='#721c24')
        self.tree_khp.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # RSD i selecció
        self.lbl_khp_rsd = tk.Label(khp_frame, text="",
                                    font=("Segoe UI", 8), bg=COLORS["white"],
                                    fg=COLORS["text_light"])
        self.lbl_khp_rsd.pack(anchor="w", padx=8, pady=(0, 4))

        # === TAULA CENTRAL UNIFICADA: FITXERS/MOSTRES ===
        files_frame = tk.LabelFrame(parent, text="FITXERS IMPORTATS", font=("Segoe UI", 9, "bold"),
                                    bg=COLORS["white"], fg=COLORS["dark"])
        files_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Treeview amb scroll
        tree_container = tk.Frame(files_frame, bg=COLORS["white"])
        tree_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Columnes: Mostra | Direct | UIB | Timeout | SNR | Status
        file_cols = ("mostra", "direct", "uib", "timeout", "snr", "status")
        self.tree_files = ttk.Treeview(tree_container, columns=file_cols, show='headings', height=8)

        # Amplades proporcionals - columnes amb contingut variable s'expandeixen
        col_config = {
            #           header,    width, minw, anchor, stretch
            "mostra":  ("Mostra",    120,   80,  "w",    True),   # expandeix
            "direct":  ("Direct",     85,   70,  "center", False), # fix
            "uib":     ("UIB",       115,   80,  "w",    True),   # expandeix
            "timeout": ("Timeout",    60,   50,  "center", False), # fix
            "snr":     ("SNR",        50,   40,  "e",    False),  # fix
            "status":  ("Status",    200,  120,  "w",    True),   # expandeix
        }
        for col, (text, width, minw, anchor, stretch) in col_config.items():
            self.tree_files.heading(col, text=text)
            self.tree_files.column(col, width=width, minwidth=minw, anchor=anchor, stretch=stretch)

        # Tags per colors
        self.tree_files.tag_configure('ok', foreground='#006100')
        self.tree_files.tag_configure('warn', foreground='#856404')
        self.tree_files.tag_configure('crit', foreground='#721c24')
        self.tree_files.tag_configure('khp', foreground='#004085', background='#e7f1ff')
        self.tree_files.tag_configure('ctrl', foreground='#155724', background='#d4edda')
        self.tree_files.tag_configure('low_conf', foreground='#856404', background='#fff3cd')  # Confiança baixa
        self.tree_files.tag_configure('dup_error', foreground='#721c24', background='#f8d7da')  # Files duplicades
        self.tree_files.tag_configure('ignored', foreground='#999999')  # Ignorat (gris)

        # Scrollbar
        scrollbar = ttk.Scrollbar(tree_container, orient="vertical", command=self.tree_files.yview)
        self.tree_files.configure(yscrollcommand=scrollbar.set)

        self.tree_files.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Bind doble-clic per editar assignació
        self.tree_files.bind("<Double-1>", self._on_file_double_click)

        # === BARRA INFERIOR: STATS + BOTONS ===
        stats_bar = tk.Frame(parent, bg=COLORS["light"])
        stats_bar.pack(fill=tk.X, pady=(5, 0))

        # Stats
        self.lbl_res_snr = tk.Label(stats_bar, text="SNR: -",
                                    font=("Segoe UI", 8), bg=COLORS["light"])
        self.lbl_res_snr.pack(side=tk.LEFT, padx=8, pady=4)

        self.lbl_res_lod = tk.Label(stats_bar, text="LOD: -",
                                    font=("Segoe UI", 8), bg=COLORS["light"])
        self.lbl_res_lod.pack(side=tk.LEFT, padx=8, pady=4)

        self.lbl_res_baseline = tk.Label(stats_bar, text="",
                                         font=("Segoe UI", 8), bg=COLORS["light"],
                                         fg=COLORS["text_light"])
        self.lbl_res_baseline.pack(side=tk.LEFT, padx=8, pady=4)

        # Botons PDF
        self.btn_open_report = tk.Button(stats_bar, text="Informe",
                                         command=self._open_consolidation_report,
                                         bg=COLORS["secondary"], fg=COLORS["white"],
                                         font=("Segoe UI", 8), relief="flat", padx=6, pady=2)
        self.btn_open_report.pack(side=tk.RIGHT, padx=2, pady=3)

        self.btn_open_chromato = tk.Button(stats_bar, text="Cromatogrames",
                                           command=self._open_chromatograms_report,
                                           bg=COLORS["secondary"], fg=COLORS["white"],
                                           font=("Segoe UI", 8), relief="flat", padx=6, pady=2)
        self.btn_open_chromato.pack(side=tk.RIGHT, padx=2, pady=3)

        # Comptador timeouts inline
        self.lbl_timeout_summary = tk.Label(stats_bar, text="",
                                            font=("Segoe UI", 8), bg=COLORS["light"],
                                            fg=COLORS["text_light"])
        self.lbl_timeout_summary.pack(side=tk.RIGHT, padx=10, pady=4)

        # === Compatibilitat amb codi existent (aliases) ===
        self.lbl_con_uib = self.lbl_data_mode  # Reutilitzem
        self.lbl_con_dad = self.lbl_data_mode
        self.lbl_con_mode = self.lbl_data_mode
        self.lbl_res_processed = self.lbl_samples_count
        self.lbl_sum_samples = self.lbl_samples_count
        self.lbl_sum_snr = self.lbl_res_snr
        self.lbl_sum_lod = self.lbl_res_lod
        self.lbl_sum_warnings = self.lbl_res_baseline
        self.lbl_res_warnings = self.lbl_res_baseline
        # Timeouts
        self.lbl_timeout_ok = self.lbl_timeout_summary
        self.lbl_timeout_info = self.lbl_timeout_summary
        self.lbl_timeout_warn = self.lbl_timeout_summary
        self.lbl_timeout_crit = self.lbl_timeout_summary
        self.lbl_timeout_affected = self.lbl_timeout_summary
        self.lbl_sum_critical = self.lbl_timeout_summary
        self.lbl_sum_generated = self.lbl_res_baseline
        # Frames
        self.summary_frame = stats_bar
        self.results_frame = stats_bar

    def _on_file_double_click(self, event):
        """Mostra detalls del fitxer seleccionat i permet acceptar/modificar assignació."""
        selection = self.tree_files.selection()
        if not selection:
            return
        item = self.tree_files.item(selection[0])
        values = item['values']
        if not values:
            return

        # Columnes: mostra, direct, uib, timeout, snr, status
        mostra = values[0] if len(values) > 0 else "-"
        direct_files = values[1] if len(values) > 1 else "-"
        uib = values[2] if len(values) > 2 else "-"
        timeout = values[3] if len(values) > 3 else "-"
        snr = values[4] if len(values) > 4 else "-"
        status = values[5] if len(values) > 5 else "OK"

        has_error = "⚠" in str(status)

        # Crear diàleg personalitzat
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Mostra: {mostra}")
        dialog.geometry("400x280")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()

        # Centrar
        dialog.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - 400) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - 280) // 2
        dialog.geometry(f"+{x}+{y}")

        # Info
        info_frame = tk.Frame(dialog, padx=15, pady=10)
        info_frame.pack(fill=tk.X)

        tk.Label(info_frame, text=f"Mostra: {mostra}", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        tk.Label(info_frame, text=f"DOC Direct (files): {direct_files}", font=("Segoe UI", 10)).pack(anchor="w", pady=2)
        tk.Label(info_frame, text=f"DOC UIB: {uib}", font=("Segoe UI", 10)).pack(anchor="w", pady=2)
        tk.Label(info_frame, text=f"Timeout: {timeout}  |  SNR: {snr}", font=("Segoe UI", 10)).pack(anchor="w", pady=2)

        # Status amb color
        status_color = "#721c24" if has_error else "#006100"
        tk.Label(info_frame, text=f"Status: {status}", font=("Segoe UI", 10, "bold"), fg=status_color).pack(anchor="w", pady=(5, 0))

        # Separador
        ttk.Separator(dialog, orient="horizontal").pack(fill=tk.X, pady=10)

        # Opcions
        options_frame = tk.Frame(dialog, padx=15)
        options_frame.pack(fill=tk.X)

        is_ignored = "IGNORAT" in str(status)

        if has_error or is_ignored:
            tk.Label(options_frame, text="Accions disponibles:", font=("Segoe UI", 10, "bold")).pack(anchor="w")

            def ignorar_mostra():
                # Marcar com a ignorada al JSON i actualitzar UI
                self._set_sample_ignored(mostra, True)
                # Actualitzar la fila a la taula (ordre: mostra, direct, uib, timeout, snr, status)
                self.tree_files.item(selection[0],
                    values=(mostra, direct_files, uib, timeout, snr, "IGNORAT"),
                    tags=('ignored',))
                dialog.destroy()

            def restaurar_mostra():
                # Treure marca d'ignorada
                self._set_sample_ignored(mostra, False)
                # Refrescar la taula (recarregar)
                dialog.destroy()
                self._show_consolidation_summary(self._last_consolidation_summary)

            btn_frame = tk.Frame(options_frame)
            btn_frame.pack(fill=tk.X, pady=10)

            if is_ignored:
                tk.Button(btn_frame, text="↩️ Restaurar mostra", command=restaurar_mostra,
                         width=18, bg="#d4edda", fg="#155724").pack(side=tk.LEFT, padx=5)
            else:
                tk.Button(btn_frame, text="🚫 Ignorar mostra", command=ignorar_mostra,
                         width=18, bg="#f8d7da", fg="#721c24").pack(side=tk.LEFT, padx=5)
        else:
            tk.Label(options_frame, text="✓ Assignació correcta", font=("Segoe UI", 10), fg="#006100").pack(anchor="w")

        # Botó tancar
        tk.Button(dialog, text="Tancar", command=dialog.destroy, width=10).pack(pady=15)

    def _set_sample_ignored(self, sample_name, ignored):
        """Marca o desmarca una mostra com a ignorada al JSON."""
        if not self.seq_path:
            return

        json_path = os.path.join(self.seq_path, "CHECK", "consolidation.json")
        if not os.path.exists(json_path):
            return

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                summary = json.load(f)

            # Inicialitzar llista d'ignorats si no existeix
            if 'ignored_samples' not in summary:
                summary['ignored_samples'] = []

            if ignored:
                if sample_name not in summary['ignored_samples']:
                    summary['ignored_samples'].append(sample_name)
            else:
                if sample_name in summary['ignored_samples']:
                    summary['ignored_samples'].remove(sample_name)

            # Guardar JSON
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

        except Exception as e:
            messagebox.showerror("Error", f"No s'ha pogut actualitzar el JSON:\n{e}")

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
    # PAS 3: PROCESSAR
    # =========================================================================

    def _build_step_processar(self, parent):
        """Construeix el pas de processament (Fase 3)."""
        tk.Label(parent, text="Pas 3: Processar Dades",
                font=("Segoe UI", 14, "bold"),
                bg=COLORS["white"], fg=COLORS["primary"]).pack(anchor="w", pady=(0, 15))

        tk.Label(parent, text="Aplica alineacio temporal, correccio baseline, deteccio pics i calcul d'arees.",
                font=("Segoe UI", 10),
                bg=COLORS["white"], fg=COLORS["dark"]).pack(anchor="w", pady=(0, 10))

        # Info processament
        info_frame = tk.LabelFrame(parent, text="Processament",
                                   font=("Segoe UI", 10, "bold"),
                                   bg=COLORS["white"])
        info_frame.pack(fill=tk.X, pady=10)

        self.lbl_proc_alignment = tk.Label(info_frame, text="Alineacio: -",
                                           font=("Segoe UI", 10),
                                           bg=COLORS["white"])
        self.lbl_proc_alignment.pack(anchor="w", padx=10, pady=3)

        self.lbl_proc_baseline = tk.Label(info_frame, text="Baseline: -",
                                          font=("Segoe UI", 10),
                                          bg=COLORS["white"])
        self.lbl_proc_baseline.pack(anchor="w", padx=10, pady=3)

        self.lbl_proc_peaks = tk.Label(info_frame, text="Pics detectats: -",
                                       font=("Segoe UI", 10),
                                       bg=COLORS["white"])
        self.lbl_proc_peaks.pack(anchor="w", padx=10, pady=3)

        self.lbl_proc_areas = tk.Label(info_frame, text="Arees calculades: -",
                                       font=("Segoe UI", 10),
                                       bg=COLORS["white"])
        self.lbl_proc_areas.pack(anchor="w", padx=10, pady=3)

        # Estadístiques
        stats_frame = tk.LabelFrame(parent, text="Estadistiques",
                                    font=("Segoe UI", 10, "bold"),
                                    bg=COLORS["white"])
        stats_frame.pack(fill=tk.X, pady=10)

        self.lbl_proc_snr = tk.Label(stats_frame, text="SNR mediana: -",
                                     font=("Segoe UI", 10),
                                     bg=COLORS["white"])
        self.lbl_proc_snr.pack(anchor="w", padx=10, pady=3)

        self.lbl_proc_lod = tk.Label(stats_frame, text="LOD: -",
                                     font=("Segoe UI", 10),
                                     bg=COLORS["white"])
        self.lbl_proc_lod.pack(anchor="w", padx=10, pady=3)

        self.lbl_proc_timeouts = tk.Label(stats_frame, text="Timeouts: -",
                                          font=("Segoe UI", 10),
                                          bg=COLORS["white"])
        self.lbl_proc_timeouts.pack(anchor="w", padx=10, pady=3)

        # Progrés
        self.progress_proc = ttk.Progressbar(parent, mode='determinate', length=400)
        self.progress_proc.pack(pady=10)

    # =========================================================================
    # PAS 4: REVISAR REPLIQUES
    # =========================================================================

    def _build_step_revisar(self, parent):
        """Construeix el pas de revisió de rèpliques (Fase 4)."""
        tk.Label(parent, text="Pas 4: Revisar Repliques",
                font=("Segoe UI", 14, "bold"),
                bg=COLORS["white"], fg=COLORS["primary"]).pack(anchor="w", pady=(0, 15))

        tk.Label(parent, text="Compara repliques, detecta anomalies i selecciona les millors (DOC i DAD independent).",
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
        self.progress_review = ttk.Progressbar(parent, mode='determinate', length=400)
        self.progress_review.pack(pady=10)

    # =========================================================================
    # PAS 4: EXPORTAR
    # =========================================================================

    def _build_step_exportar(self, parent):
        """Construeix el pas d'exportació."""
        tk.Label(parent, text="Pas 5: Exportar Resultats",
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
        self.btn_next.configure(state="normal" if step_idx < 4 else "disabled")

        if step_idx == 4:
            self.btn_next.configure(text="Finalitzar")
        else:
            self.btn_next.configure(text="Següent →")

    def _next_step(self):
        """Avança al següent pas."""
        if self.current_step < 4:
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
        self.khp_area = None
        self.khp_conc = None
        self.khp_source_seq = None
        self.is_processing = False

        # Analitzar contingut
        self._analyze_folder()

        # Carregar calibració existent si n'hi ha
        self._load_existing_calibration()

        # Activar botó consolidar
        self.btn_importar.configure(state="normal")
        self.lbl_status.configure(text=f"Carpeta carregada: {os.path.basename(folder)}")

        # Anar al primer pas
        self._show_step(0)

    def _analyze_folder(self):
        """Analitza el contingut de la carpeta SEQ i omple la nova UI."""
        if not self.seq_path:
            return

        seq_name = os.path.basename(self.seq_path)
        mode = detect_mode(self.seq_path)

        # Extreure número SEQ i data (si disponible)
        seq_num = None
        import re
        match = re.search(r'^(\d+)', seq_name)
        if match:
            seq_num = int(match.group(1))

        # Determinar volum d'injecció
        is_bp = mode == "BP"
        if is_bp:
            inj_vol = 100
        elif seq_num and 256 <= seq_num <= 274:
            inj_vol = 100
        else:
            inj_vol = 400

        # Determinar sensibilitat UIB
        if seq_num and 269 <= seq_num <= 274:
            uib_sens = "700 ppb"
        else:
            uib_sens = "1000 ppb"

        # Buscar MasterFile
        # Buscar MasterFile i comptar mostres precises
        master_files = glob.glob(os.path.join(self.seq_path, "*MasterFile*.xlsx"))
        master_files = [f for f in master_files if "~$" not in f and "backup" not in f.lower()]

        n_mostres, n_khp, n_ctrl = 0, 0, 0
        if master_files:
            master_name = os.path.basename(master_files[0])
            self.lbl_con_master.configure(text=master_name, fg=COLORS["text_light"])
            # Comptar mostres des del MasterFile
            try:
                import pandas as pd
                df = pd.read_excel(master_files[0], sheet_name='Mostres', usecols=['Mostra'])
                for m in df['Mostra'].dropna():
                    m_str = str(m).upper()
                    if 'KHP' in m_str:
                        n_khp += 1
                    elif 'CTRL' in m_str or 'CONTROL' in m_str or 'BLANC' in m_str:
                        n_ctrl += 1
                    else:
                        n_mostres += 1
            except Exception:
                pass
        else:
            self.lbl_con_master.configure(text="No MasterFile!", fg=COLORS["warning"])

        # Obtenir info de fitxers UIB/DAD
        info = check_sequence_files(self.seq_path)
        uib_count = info.get("uib", {}).get("count_found", 0)
        dad_count = info.get("dad", {}).get("count_found", 0)

        # Determinar mode de dades
        has_direct = dad_count > 0
        has_uib = uib_count > 0
        if has_direct and has_uib:
            data_mode = "DUAL"
        elif has_uib:
            data_mode = "UIB"
        elif has_direct:
            data_mode = "Direct"
        else:
            data_mode = "-"

        # === Actualitzar UI SEQÜÈNCIA ===
        self.lbl_seq_header.configure(text=f"{seq_name}  ·  {mode}")
        self.lbl_data_mode.configure(text=f"{data_mode}")
        self.lbl_uib_sens.configure(text=f"{uib_sens}" if has_uib else "-")
        self.lbl_inj_vol.configure(text=f"{inj_vol} µL")

        # Comptadors precisos
        if n_mostres > 0 or n_khp > 0 or n_ctrl > 0:
            self.lbl_samples_count.configure(text=f"{n_mostres} mostres, {n_khp} KHP, {n_ctrl} control")
        else:
            # Fallback a estimació
            n_est = max(dad_count, uib_count) // 2 if max(dad_count, uib_count) > 0 else 0
            self.lbl_samples_count.configure(text=f"~{n_est} injeccions (estimat)")

        # Estat inicial
        self.lbl_con_status.configure(text="Pendent", fg=COLORS["text_light"])

        # === Netejar KHP ===
        for item in self.tree_khp.get_children():
            self.tree_khp.delete(item)
        self.lbl_khp_header.configure(text="Analitzant...", fg=COLORS["text_light"])
        self.lbl_khp_rsd.configure(text="")

        # === Netejar taula fitxers ===
        for item in self.tree_files.get_children():
            self.tree_files.delete(item)
        self.lbl_timeout_summary.configure(text="")
        self.lbl_res_snr.configure(text="SNR: -")
        self.lbl_res_lod.configure(text="LOD: -")
        self.lbl_res_baseline.configure(text="")

        # === Buscar KHP existents ===
        self._analyze_khp_files()

        # === Comprovar si ja consolidat ===
        con_folder = os.path.join(self.seq_path, "Resultats_Consolidats")
        if os.path.isdir(con_folder):
            con_files = glob.glob(os.path.join(con_folder, "*.xlsx"))
            con_files = [f for f in con_files if "~$" not in f]
            if con_files:
                self.lbl_con_status.configure(text=f"Importat ({len(con_files)} fitxers)",
                                             fg=COLORS["success"])
                self.consolidated_data = {"path": con_folder, "files": con_files}
                self.btn_calibrar.configure(state="normal")
                self._load_consolidation_summary()

    def _analyze_khp_files(self):
        """Analitza els fitxers KHP i omple la taula."""
        con_folder = os.path.join(self.seq_path, "Resultats_Consolidats")
        khp_files = glob.glob(os.path.join(con_folder, "KHP*.xlsx"))
        khp_files = [f for f in khp_files if "~$" not in f]

        if not khp_files:
            self.lbl_khp_header.configure(text="Cap KHP trobat", fg=COLORS["text_light"])
            return

        # Agrupar per concentració
        from hpsec_calibrate import analizar_khp_consolidado, extract_khp_conc

        khp_by_conc = {}
        for f in khp_files:
            conc = extract_khp_conc(os.path.basename(f))
            if conc not in khp_by_conc:
                khp_by_conc[conc] = []
            khp_by_conc[conc].append(f)

        # Mostrar info per cada concentració
        for conc, files in khp_by_conc.items():
            self.lbl_khp_header.configure(
                text=f"KHP{conc} ({conc} ppm) · {len(files)} repliques",
                fg=COLORS["primary"]
            )

            areas = []
            for i, f in enumerate(files):
                try:
                    result = analizar_khp_consolidado(f)
                    if result:
                        area = result.get('area', 0)
                        a254 = result.get('a254_area', 0)
                        ratio = result.get('a254_doc_ratio', 0)
                        snr = result.get('snr', 0)
                        issues = result.get('quality_issues', [])

                        areas.append(area)

                        # Determinar estat
                        if issues or snr < 10:
                            status = "!"
                            tag = 'warn'
                        else:
                            status = "ok"
                            tag = 'ok'

                        self.tree_khp.insert("", "end",
                            values=(f"R{i+1}", f"{area:.1f}", f"{a254:.2f}",
                                   f"{ratio:.4f}", f"{snr:.0f}", status),
                            tags=(tag,))
                except Exception as e:
                    self.tree_khp.insert("", "end",
                        values=(f"R{i+1}", "Error", "-", "-", "-", "!"),
                        tags=('fail',))

            # RSD
            if len(areas) >= 2:
                import numpy as np
                mean_area = np.mean(areas)
                std_area = np.std(areas)
                rsd = (std_area / mean_area * 100) if mean_area > 0 else 0
                selection = "Promig" if rsd < 10 else "Millor qualitat"
                self.lbl_khp_rsd.configure(text=f"RSD: {rsd:.1f}% · Seleccio: {selection}")
            break  # Només mostrem el primer grup KHP

    def _load_consolidation_summary(self):
        """Carrega resum de consolidació existent."""
        check_folder = os.path.join(self.seq_path, "CHECK")
        json_path = os.path.join(check_folder, "consolidation.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    summary = json.load(f)
                self._show_consolidation_summary(summary)
            except Exception:
                pass

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
    # EXECUCIÓ: IMPORTAR (Fase 1)
    # =========================================================================

    def _run_import(self):
        """Executa la importació de dades (Fase 1)."""
        if self.is_processing or not self.seq_path:
            return

        self.is_processing = True
        self.btn_importar.configure(state="disabled")
        self.progress_import['value'] = 0
        self.lbl_con_status.configure(text="Estat: Important...", fg=COLORS["dark"])
        self.lbl_import_progress.configure(text="Iniciant...")

        def import_thread():
            try:
                def on_progress(pct, item):
                    """Callback: pct=percentatge (0-100), item=nom mostra"""
                    self.root.after(0, lambda: self._update_import_progress(pct, item))

                # Usa consolidate_sequence que ara fa la fase d'importació
                result = consolidate_sequence(self.seq_path, progress_callback=on_progress)

                self.root.after(0, lambda: self._import_done(result))
            except Exception as e:
                self.root.after(0, lambda: self._import_error(str(e)))

        thread = threading.Thread(target=import_thread, daemon=True)
        thread.start()

    def _update_import_progress(self, pct, item):
        """Actualitza progrés d'importació."""
        self.progress_import['value'] = pct
        self.lbl_import_progress.configure(text=f"Important: {item}")

    def _import_done(self, result):
        """Callback quan la importació acaba."""
        self.is_processing = False
        self.progress_import['value'] = 100
        self.lbl_import_progress.configure(text="")

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
            self.btn_importar.configure(state="normal")

    def _show_consolidation_summary(self, summary):
        """Mostra el resum de consolidació a la GUI (taula unificada)."""

        # === COMPTADORS PRECISOS ===
        counts = summary.get('counts', {})
        total = counts.get('total_samples', 0)
        khp = counts.get('khp_samples', 0)
        ctrl = counts.get('control_samples', 0)
        reg = counts.get('regular_samples', 0)
        self.lbl_samples_count.configure(text=f"{reg} mostres, {khp} KHP, {ctrl} control")

        # === TAULA FITXERS - Tot directament del JSON (eficient!) ===
        for item in self.tree_files.get_children():
            self.tree_files.delete(item)

        samples = summary.get('samples', [])
        n_low_conf = 0  # Comptador de baixa confiança

        # Obtenir samples ignorats
        ignored_samples = set(summary.get('ignored_samples', []))

        # Obtenir samples afectats per duplicate rows
        quality = summary.get('quality', {})
        dup_rows = quality.get('duplicate_rows', [])
        dup_samples_set = set()
        for dup in dup_rows:
            for sample_name in dup.get('samples', []):
                dup_samples_set.add(sample_name)

        for s in samples:
            name = s.get('name', '-')
            mostra = s.get('mostra', '-')
            replica = s.get('replica', '')
            mostra_display = f"{mostra}_R{replica}" if replica else mostra

            # Info directa del JSON (afegida al backend)
            file_dad = s.get('file_dad', '')
            file_uib = s.get('file_uib', '')
            row_start = s.get('row_start', 0)
            row_end = s.get('row_end', 0)
            npts = s.get('npts', 0)
            match_conf = s.get('match_confidence', 100.0)
            doc_mode = s.get('doc_mode', '')

            # DOC Direct status (per mostrar errors)
            doc_direct_status = s.get('doc_direct_status', 'OK')
            doc_direct_message = s.get('doc_direct_message', '')
            has_doc_error = doc_direct_status not in ['OK', '']

            # Columna Direct: mostrar només rang de files MasterFile
            if row_start and row_end:
                direct_text = f"{row_start}-{row_end}"
            else:
                direct_text = "-"

            # Columna UIB: fitxer (o "-" si no n'hi ha)
            if file_uib:
                uib_name = os.path.splitext(file_uib)[0]
                uib_text = uib_name
            else:
                uib_text = "-"

            # Comprovar si la mostra està ignorada
            is_ignored = mostra_display in ignored_samples or name in ignored_samples

            # Columna Status: explicació clara
            if is_ignored:
                status_text = "IGNORAT"
            elif doc_direct_status == 'ORFE_HPLC_SEQ':
                status_text = "⚠ UIB existeix, falta a 1-HPLC-SEQ"
            elif doc_direct_status == 'FALTA_HPLC_SEQ':
                status_text = "⚠ Línia absent a 1-HPLC-SEQ"
            elif doc_direct_status == 'SENSE_DADES':
                status_text = "⚠ Sense dades DOC"
            elif doc_direct_status == 'SOBREESCRIT':
                status_text = "⚠ DOC sobreescrit (memòria)"
            elif has_doc_error:
                status_text = f"⚠ {doc_direct_status}"
            elif match_conf < 85:
                status_text = f"⚠ Baixa confiança ({match_conf:.0f}%)"
            elif doc_mode == 'UIB' and not file_dad:
                status_text = "Només UIB"
            elif doc_mode == 'DIRECT' and not file_uib:
                status_text = "Només Direct"
            else:
                status_text = "OK"

            # Determinar tag per color
            name_upper = name.upper()
            if is_ignored:
                tag = 'ignored'  # Gris: ignorat
            elif has_doc_error:
                tag = 'dup_error'  # Vermell: error MasterFile
            elif name in dup_samples_set:
                tag = 'dup_error'  # Vermell: files duplicades
            elif match_conf < 85:
                tag = 'low_conf'  # Groc: baixa confiança
                n_low_conf += 1
            elif 'KHP' in name_upper:
                tag = 'khp'
            elif 'MQ' in name_upper or 'NAOH' in name_upper or 'CTRL' in name_upper or 'CONTROL' in name_upper or 'BLANC' in name_upper:
                tag = 'ctrl'
            else:
                tag = 'ok'

            # Timeout info
            timeout_sev = s.get('timeout_severity', 'OK')
            timeout_zones = s.get('timeout_zones', [])
            if timeout_sev == 'CRITICAL':
                timeout_text = ','.join(timeout_zones) if timeout_zones else "CRIT"
                if tag == 'ok':
                    tag = 'crit'
            elif timeout_sev == 'WARNING':
                timeout_text = ','.join(timeout_zones) if timeout_zones else "WARN"
                if tag == 'ok':
                    tag = 'warn'
            elif timeout_sev == 'INFO':
                timeout_text = "info"
            else:
                timeout_text = "-"

            # SNR
            snr = s.get('snr_direct', s.get('snr_uib', 0))
            snr_text = f"{snr:.0f}" if snr else "-"

            # Ajustar tag si peak no és vàlid
            if not s.get('peak_valid', True):
                if tag == 'ok':
                    tag = 'warn'
                status_text = status_text + " (!)" if status_text == "OK" else status_text

            # Columnes: mostra, direct, uib, timeout, snr, status (status al final)
            self.tree_files.insert("", "end",
                values=(mostra_display, direct_text, uib_text, timeout_text, snr_text, status_text),
                tags=(tag,))

        # Mostrar avís si hi ha assignacions de baixa confiança
        if n_low_conf > 0:
            self.lbl_res_baseline.configure(
                text=f"⚠ {n_low_conf} assignacions amb baixa confiança - revisar!",
                fg=COLORS["warning"])

        # === RESUM TIMEOUTS ===
        timeouts = summary.get('timeouts', {})
        sev_counts = timeouts.get('severity_counts', {})
        n_ok = sev_counts.get('OK', 0)
        n_info = sev_counts.get('INFO', 0)
        n_warn = sev_counts.get('WARNING', 0)
        n_crit = sev_counts.get('CRITICAL', 0)

        timeout_parts = []
        if n_crit > 0:
            timeout_parts.append(f"CRIT:{n_crit}")
        if n_warn > 0:
            timeout_parts.append(f"WARN:{n_warn}")
        if n_info > 0:
            timeout_parts.append(f"INFO:{n_info}")
        if timeout_parts:
            self.lbl_timeout_summary.configure(text=f"Timeouts: {' '.join(timeout_parts)}")
        else:
            self.lbl_timeout_summary.configure(text="Timeouts: OK")

        # === KHP ALIGNMENT INFO ===
        alignment = summary.get('alignment', {})
        if alignment:
            khp_validation = alignment.get('khp_validation', 'N/A')
            khp_file = alignment.get('khp_file', '-')
            khp_metrics = alignment.get('khp_metrics', {})
            area_doc = khp_metrics.get('area_doc', 0)

            from hpsec_calibrate import extract_khp_conc
            khp_conc = extract_khp_conc(khp_file) if khp_file else None

            if area_doc > 0:
                self.khp_area = area_doc
                self.khp_conc = khp_conc
                self.khp_source_seq = os.path.basename(self.seq_path)

                if khp_validation == 'VALID':
                    self.lbl_khp_header.configure(
                        text=f"✓ {khp_file} ({area_doc:.1f})", fg=COLORS["success"])
                elif khp_validation == 'INVALID':
                    self.lbl_khp_header.configure(
                        text=f"✗ {khp_file} ({area_doc:.1f})", fg=COLORS["error"])
                else:
                    self.lbl_khp_header.configure(
                        text=f"⚠ {khp_file} ({area_doc:.1f})", fg=COLORS["warning"])

                # Històric
                hist = khp_metrics.get('historical_comparison', {})
                if hist:
                    hist_status = hist.get('status', 'N/A')
                    hist_dev = hist.get('area_deviation_pct', 0)
                    self.lbl_khp_rsd.configure(text=f"Hist: {hist_status} ({hist_dev:.1f}%)")
            else:
                self.khp_area = None
                self.khp_conc = None
                self.khp_source_seq = None
        else:
            self.khp_area = None
            self.khp_conc = None
            self.khp_source_seq = None

        # === QUALITAT ===
        quality = summary.get('quality', {})
        snr_direct = quality.get('snr_direct', {})
        if snr_direct:
            self.lbl_res_snr.configure(
                text=f"SNR: {snr_direct.get('median', '-')}")
        else:
            self.lbl_res_snr.configure(text="SNR: -")

        lod_direct = quality.get('lod_direct_mau')
        if lod_direct:
            self.lbl_res_lod.configure(text=f"LOD: {lod_direct:.2f} mAU")
        else:
            self.lbl_res_lod.configure(text="LOD: -")

        # === QUALITY ISSUES ===
        quality_issues = summary.get('quality_issues', [])
        errors = [q for q in quality_issues if q.get('severity') == 'ERROR']
        warnings = [q for q in quality_issues if q.get('severity') == 'WARNING']

        if errors:
            # Mostrar errors de qualitat (prioritat màxima)
            self.lbl_res_baseline.configure(
                text=f"⛔ {len(errors)} ERROR{'S' if len(errors) > 1 else ''}: {errors[0].get('message', '')}",
                fg=COLORS["error"])
        elif warnings:
            # Mostrar warnings de qualitat
            self.lbl_res_baseline.configure(
                text=f"⚠ {len(warnings)} AVÍS: {warnings[0].get('message', '')}",
                fg=COLORS["warning"])
        elif n_low_conf > 0:
            # Ja mostrat abans (baixa confiança)
            pass
        else:
            # Data generació (només si no hi ha issues)
            meta = summary.get('meta', {})
            generated_at = meta.get('generated_at', '')
            if generated_at:
                try:
                    dt = datetime.fromisoformat(generated_at)
                    self.lbl_res_baseline.configure(
                        text=dt.strftime("%d/%m %H:%M"), fg=COLORS["text"])
                except:
                    self.lbl_res_baseline.configure(text="")

    def _import_error(self, error):
        """Callback quan hi ha error d'importació."""
        self.is_processing = False
        self.lbl_con_status.configure(text=f"Error: {error}", fg=COLORS["error"])
        self.btn_importar.configure(state="normal")
        messagebox.showerror("Error", f"Error durant la importacio:\n\n{error}")

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

    def _load_existing_calibration(self):
        """
        Carrega dades de calibració existent si la SEQ ja té calibració a CALDATA.
        Això permet veure les dades sense haver de prémer "Calibrar" de nou.
        """
        if not self.seq_path:
            return

        # Obtenir calibració activa del CALDATA local
        active_cal = get_active_calibration(self.seq_path)
        if not active_cal:
            return

        # Extreure dades de la calibració existent
        factor = active_cal.get('factor', 0)
        conc = active_cal.get('conc_ppm', 0)
        area = active_cal.get('area', 0)
        n_replicas = active_cal.get('n_replicas', 1)
        rsd = active_cal.get('rsd', 0)
        shift_sec = active_cal.get('shift_sec', 0)
        symmetry = active_cal.get('symmetry', 1.0)
        snr = active_cal.get('snr', 0)
        quality_issues = active_cal.get('quality_issues', [])
        quality_score = active_cal.get('quality_score', 0)
        status = active_cal.get('status', 'OK')
        khp_filename = active_cal.get('khp_file', 'N/A')
        khp_source = active_cal.get('khp_source', 'LOCAL')

        # Actualitzar variables globals
        if area > 0 and conc > 0:
            self.khp_area = area
            self.khp_conc = conc
            self.khp_source_seq = active_cal.get('seq_name', os.path.basename(self.seq_path))

        # Actualitzar etiquetes - Info KHP
        self.lbl_cal_source.configure(text=f"Origen: {khp_source} (existent)")
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

        # Obtenir camps de validació
        valid_for_cal = active_cal.get('valid_for_calibration', True)
        valid_for_shift = active_cal.get('valid_for_shift', True)
        cal_issues = active_cal.get('calibration_issues', [])
        shift_issues = active_cal.get('shift_issues', [])

        # Estat de qualitat amb colors
        if not valid_for_cal:
            self.lbl_cal_quality.configure(
                text="Estat: ✗ INVÀLID CALIBRACIÓ",
                fg=COLORS["error"]
            )
        elif not valid_for_shift:
            self.lbl_cal_quality.configure(
                text="Estat: ⚠ INVÀLID SHIFT (cal OK)",
                fg=COLORS["warning"]
            )
        elif quality_score < 20:
            self.lbl_cal_quality.configure(text="Estat: ✓ EXCEL·LENT", fg=COLORS["success"])
        elif quality_score < 50:
            self.lbl_cal_quality.configure(text="Estat: ✓ BO", fg=COLORS["success"])
        elif quality_score < 100:
            self.lbl_cal_quality.configure(text="Estat: ⚠ REVISAR", fg=COLORS["warning"])
        else:
            self.lbl_cal_quality.configure(text="Estat: ✗ PROBLEMES", fg=COLORS["error"])

        # Mostrar issues
        all_issues = cal_issues + shift_issues + quality_issues
        seen = set()
        unique_issues = []
        for issue in all_issues:
            if issue not in seen:
                seen.add(issue)
                unique_issues.append(issue)

        if unique_issues:
            issues_text = "\n".join(unique_issues[:5])
            if len(unique_issues) > 5:
                issues_text += f"\n... i {len(unique_issues) - 5} més"
            self.lbl_cal_issues.configure(text=issues_text)
        else:
            self.lbl_cal_issues.configure(text="")

        # Guardar dades de calibració
        self.calibration_data = {
            'success': True,
            'khp_data': active_cal,
            'calibration': active_cal,
            'khp_source': khp_source,
            'from_existing': True
        }

        # Activar següent pas (Processar)
        self.btn_processar.configure(state="normal")

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

            # Obtenir nous camps de validació
            valid_for_cal = calibration.get('valid_for_calibration', True)
            valid_for_shift = calibration.get('valid_for_shift', True)
            cal_issues = calibration.get('calibration_issues', [])
            shift_issues = calibration.get('shift_issues', [])

            # Estat de qualitat amb colors - basat en validació, no només quality_score
            if not valid_for_cal:
                self.lbl_cal_quality.configure(
                    text="Estat: ✗ INVÀLID CALIBRACIÓ",
                    fg=COLORS["error"]
                )
            elif not valid_for_shift:
                self.lbl_cal_quality.configure(
                    text="Estat: ⚠ INVÀLID SHIFT (cal OK)",
                    fg=COLORS["warning"]
                )
            elif quality_score < 20:
                self.lbl_cal_quality.configure(text="Estat: ✓ EXCEL·LENT", fg=COLORS["success"])
            elif quality_score < 50:
                self.lbl_cal_quality.configure(text="Estat: ✓ BO", fg=COLORS["success"])
            elif quality_score < 100:
                self.lbl_cal_quality.configure(text="Estat: ⚠ REVISAR", fg=COLORS["warning"])
            else:
                self.lbl_cal_quality.configure(text="Estat: ✗ PROBLEMES", fg=COLORS["error"])

            # Mostrar issues - prioritzar cal_issues i shift_issues sobre quality_issues
            all_issues = cal_issues + shift_issues + quality_issues
            # Eliminar duplicats mantenint ordre
            seen = set()
            unique_issues = []
            for issue in all_issues:
                if issue not in seen:
                    seen.add(issue)
                    unique_issues.append(issue)

            if unique_issues:
                issues_text = " | ".join(unique_issues[:3])  # Màxim 3
                self.lbl_cal_issues.configure(text=issues_text)
            else:
                self.lbl_cal_issues.configure(text="")

            self.calibration_data = result

            # Activar següent pas (Processar és el pas 3)
            self.btn_processar.configure(state="normal")

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
        """Mostra gràfic de calibració amb les dues rèpliques KHP i històric."""
        # Netejar canvas anterior
        for widget in self.cal_canvas_frame.winfo_children():
            widget.destroy()

        khp_data = result.get('khp_data', {})
        calibration = result.get('calibration', {})

        # Obtenir rèpliques individuals
        replicas = khp_data.get('replicas', [])
        if not replicas:
            # Si no hi ha replicas separades, usar all_khp_data
            replicas = khp_data.get('all_khp_data', [])
        if not replicas:
            # Fallback: crear una sola "rèplica" amb les dades principals
            replicas = [khp_data]

        has_dad = any(r.get('t_dad') is not None and r.get('y_dad_254') is not None
                     for r in replicas)

        # Crear figura amb 3 subplots: DOC (amb rèpliques), DAD, Històric
        fig = Figure(figsize=(10, 6), dpi=100)

        if has_dad:
            ax1 = fig.add_subplot(221)  # DOC
            ax2 = fig.add_subplot(223)  # DAD
            ax3 = fig.add_subplot(122)  # Històric (gran, a la dreta)
        else:
            ax1 = fig.add_subplot(121)  # DOC
            ax2 = None
            ax3 = fig.add_subplot(122)  # Històric

        # Colors per rèpliques
        rep_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        # Plot DOC - TOTES LES RÈPLIQUES
        conc = khp_data.get('conc_ppm', 0)
        areas_list = []

        for i, rep in enumerate(replicas):
            t_doc = rep.get('t_doc')
            y_doc = rep.get('y_doc')

            if t_doc is None or y_doc is None or len(t_doc) == 0:
                continue

            color = rep_colors[i % len(rep_colors)]
            rep_name = rep.get('filename', f'R{i+1}')
            # Simplificar nom
            if '_R' in rep_name:
                rep_name = 'R' + rep_name.split('_R')[-1].split('.')[0].split('_')[0]

            area = rep.get('area', 0)
            areas_list.append(area)

            ax1.plot(t_doc, y_doc, color=color, linewidth=1.2,
                    label=f'{rep_name}: {area:.1f}', alpha=0.8)

            # Marcar pic principal de cada rèplica
            t_max = rep.get('t_doc_max', 0)
            if t_max:
                peak_idx = np.argmin(np.abs(np.array(t_doc) - t_max))
                ax1.plot(t_max, y_doc[peak_idx], 'o', color=color, markersize=6)

        # Títol amb estadístiques
        if len(areas_list) >= 2:
            mean_area = np.mean(areas_list)
            rsd = (np.std(areas_list) / mean_area * 100) if mean_area > 0 else 0
            ax1.set_title(f"KHP{conc} - Mitjana: {mean_area:.1f} (RSD {rsd:.1f}%)", fontsize=9)
        else:
            ax1.set_title(f"KHP{conc} - Àrea: {areas_list[0]:.1f}" if areas_list else "KHP", fontsize=9)

        ax1.set_xlabel("Temps (min)", fontsize=8)
        ax1.set_ylabel("Senyal DOC (mV)", fontsize=8)
        ax1.legend(loc='upper right', fontsize=7)
        ax1.grid(True, alpha=0.3)

        # Plot DAD 254nm - TOTES LES RÈPLIQUES
        if has_dad and ax2 is not None:
            for i, rep in enumerate(replicas):
                t_dad = rep.get('t_dad')
                y_dad = rep.get('y_dad_254')

                if t_dad is None or y_dad is None or len(t_dad) == 0:
                    continue

                color = rep_colors[i % len(rep_colors)]
                rep_name = rep.get('filename', f'R{i+1}')
                if '_R' in rep_name:
                    rep_name = 'R' + rep_name.split('_R')[-1].split('.')[0].split('_')[0]

                ax2.plot(t_dad, y_dad, color=color, linewidth=1.0,
                        label=rep_name, alpha=0.8)

                t_dad_max = rep.get('t_dad_max', 0)
                if t_dad_max and len(t_dad) > 0:
                    dad_peak_idx = np.argmin(np.abs(np.array(t_dad) - t_dad_max))
                    ax2.plot(t_dad_max, y_dad[dad_peak_idx], 'o', color=color, markersize=5)

            shift_sec = khp_data.get('shift_sec', 0)
            ax2.set_title(f"DAD 254nm - Shift: {shift_sec:.1f} s", fontsize=9)
            ax2.set_xlabel("Temps (min)", fontsize=8)
            ax2.set_ylabel("Absorbància (mAU)", fontsize=8)
            ax2.legend(loc='upper right', fontsize=7)
            ax2.grid(True, alpha=0.3)

        # Plot Històric KHP
        self._plot_khp_history(ax3, khp_data)

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.cal_canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _plot_khp_history(self, ax, current_khp_data):
        """Mostra gràfic d'històric KHP amb àrees per SEQ (inclou outliers marcats)."""
        try:
            # Carregar històric
            history = load_khp_history(self.seq_path)
            if not history:
                ax.text(0.5, 0.5, "No hi ha històric disponible",
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=12, color='gray')
                ax.set_title("Històric KHP")
                return

            # Filtrar per mode i concentració similar (INCLOU outliers per veure panorama complet)
            current_conc = current_khp_data.get('conc_ppm', 2)
            mode = "BP" if "BP" in os.path.basename(self.seq_path).upper() else "COLUMN"

            # Recopilar TOTES les calibracions del mode/conc (inclou outliers)
            all_cals = []
            for cal in history:
                if cal.get('mode') != mode:
                    continue
                cal_conc = cal.get('conc_ppm', 0)
                if abs(cal_conc - current_conc) > 0.5:
                    continue
                if cal.get('area', 0) <= 0:
                    continue
                all_cals.append(cal)

            if not all_cals:
                ax.text(0.5, 0.5, f"No hi ha històric per {mode} KHP{current_conc:.0f}",
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=10, color='gray')
                ax.set_title("Històric KHP")
                return

            # Extreure número de SEQ per ordenar
            def get_seq_num(cal):
                seq_name = cal.get('seq_name', '')
                match = re.search(r'(\d+)', seq_name)
                return int(match.group(1)) if match else 0

            # Ordenar per número de SEQ (ascendent = cronològic)
            all_cals.sort(key=get_seq_num)

            # Preparar dades
            seq_names = [cal.get('seq_name', 'N/A').replace('_SEQ', '').replace('_BP', '') for cal in all_cals]
            areas = [cal.get('area', 0) for cal in all_cals]
            is_outliers = [cal.get('is_outlier', False) for cal in all_cals]

            # Identificar SEQ actual
            current_seq = os.path.basename(self.seq_path)
            current_seq_short = current_seq.replace('_SEQ', '').replace('_BP', '')

            # Colors: vermell per actual, gris per outliers, blau per vàlids
            colors = []
            for i, (name, is_out) in enumerate(zip(seq_names, is_outliers)):
                if current_seq_short in name:
                    colors.append(COLORS["error"])  # Actual
                elif is_out:
                    colors.append('#999999')  # Outlier = gris
                else:
                    colors.append(COLORS["primary"])  # Vàlid = blau

            # Gràfic de barres
            bars = ax.bar(range(len(seq_names)), areas, color=colors, alpha=0.7, edgecolor='black')

            # Marcar outliers amb patró
            for i, (bar, is_out) in enumerate(zip(bars, is_outliers)):
                if is_out:
                    bar.set_hatch('///')
                    bar.set_alpha(0.5)

            # Mitjana i std només de calibracions NO outlier
            valid_areas = [a for a, is_out in zip(areas, is_outliers) if not is_out]
            n_valid = len(valid_areas)

            if valid_areas:
                mean_area = np.mean(valid_areas)
                std_area = np.std(valid_areas) if len(valid_areas) > 1 else 0
                ax.axhline(mean_area, color='green', linestyle='--', linewidth=2,
                          label=f'Mitjana: {mean_area:.1f} (n={n_valid})')
                if std_area > 0:
                    ax.axhspan(mean_area - std_area, mean_area + std_area, alpha=0.2,
                              color='green', label=f'±1σ ({std_area:.1f})')

            # Etiquetes
            ax.set_xticks(range(len(seq_names)))
            ax.set_xticklabels(seq_names, rotation=45, ha='right', fontsize=7)
            ax.set_ylabel("Àrea KHP", fontsize=9)

            n_outliers = sum(is_outliers)
            title = f"Històric {mode} KHP{current_conc:.0f}"
            if n_outliers > 0:
                title += f" (vàlids={n_valid}, outliers={n_outliers})"
            else:
                title += f" (n={n_valid})"
            ax.set_title(title, fontsize=10)

            ax.legend(loc='upper right', fontsize=7)
            ax.grid(True, alpha=0.3, axis='y')

            # Afegir valors a les barres
            for bar, area in zip(bars, areas):
                height = bar.get_height()
                ax.annotate(f'{area:.0f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=6, rotation=90)

        except Exception as e:
            ax.text(0.5, 0.5, f"Error carregant històric:\n{str(e)[:50]}",
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=10, color='red')
            ax.set_title("Històric KHP")

    def _show_khp_history(self):
        """Mostra diàleg amb l'històric complet de KHP amb opció de selecció."""
        if not self.seq_path:
            messagebox.showwarning("Avís", "Cal seleccionar una seqüència primer.")
            return

        try:
            history = load_khp_history(self.seq_path)
            if not history:
                messagebox.showinfo("Històric KHP", "No hi ha històric disponible.")
                return

            # Crear finestra
            dialog = tk.Toplevel(self.root)
            dialog.title("Històric KHP - Seleccionar Calibració")
            dialog.geometry("900x550")
            dialog.transient(self.root)
            dialog.grab_set()

            # Guardar referència a l'històric per la selecció
            self._khp_history_data = history
            self._khp_history_dialog = dialog

            # Header amb info
            header = tk.Frame(dialog, bg=COLORS["secondary"], height=50)
            header.pack(fill=tk.X)
            header.pack_propagate(False)

            tk.Label(header, text="Històric de Calibracions KHP",
                    font=("Segoe UI", 12, "bold"),
                    bg=COLORS["secondary"], fg=COLORS["white"]).pack(side=tk.LEFT, padx=20, pady=12)

            # Info actual
            current_info = f"Actual: {self.khp_source_seq or 'Cap'}"
            if self.khp_area and self.khp_conc:
                current_info += f" | Àrea: {self.khp_area:.1f} | Conc: {self.khp_conc} ppm"
            tk.Label(header, text=current_info,
                    font=("Segoe UI", 10),
                    bg=COLORS["secondary"], fg=COLORS["light"]).pack(side=tk.RIGHT, padx=20, pady=12)

            # Frame principal
            main_frame = tk.Frame(dialog, bg=COLORS["white"])
            main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            # Filtre de mode
            filter_frame = tk.Frame(main_frame, bg=COLORS["white"])
            filter_frame.pack(fill=tk.X, pady=(0, 10))

            tk.Label(filter_frame, text="Mode:",
                    font=("Segoe UI", 10),
                    bg=COLORS["white"]).pack(side=tk.LEFT, padx=(0, 5))

            self._khp_mode_var = tk.StringVar(value="TOTS")
            current_mode = detect_mode(self.seq_path) if self.seq_path else "TOTS"

            for mode in ["TOTS", "COLUMN", "BP"]:
                rb = tk.Radiobutton(filter_frame, text=mode, variable=self._khp_mode_var, value=mode,
                                   font=("Segoe UI", 10), bg=COLORS["white"],
                                   command=lambda: self._filter_khp_history())
                rb.pack(side=tk.LEFT, padx=5)

            # Taula
            columns = ("SEQ", "Mode", "KHP", "Àrea", "Factor", "Shift (s)", "SNR", "Status")
            self._khp_tree = ttk.Treeview(main_frame, columns=columns, show='headings', height=15)

            widths = {"SEQ": 110, "Mode": 70, "KHP": 70, "Àrea": 80, "Factor": 90, "Shift (s)": 70, "SNR": 60, "Status": 80}
            for col in columns:
                self._khp_tree.heading(col, text=col, command=lambda c=col: self._sort_khp_history(c))
                self._khp_tree.column(col, width=widths.get(col, 80), anchor="center")

            # Tags de colors
            self._khp_tree.tag_configure('OK', background='#c6efce', foreground='#006100')
            self._khp_tree.tag_configure('WARN', background='#fff3cd', foreground='#856404')
            self._khp_tree.tag_configure('FAIL', background='#f8d7da', foreground='#721c24')
            self._khp_tree.tag_configure('outlier', foreground='gray')
            self._khp_tree.tag_configure('current', background='#cce5ff', foreground='#004085')

            # Scrollbar
            scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self._khp_tree.yview)
            self._khp_tree.configure(yscrollcommand=scrollbar.set)

            self._khp_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

            # Omplir taula
            self._populate_khp_history()

            # Frame inferior amb botons
            btn_frame = tk.Frame(dialog, bg=COLORS["light"], height=60)
            btn_frame.pack(fill=tk.X, side=tk.BOTTOM)
            btn_frame.pack_propagate(False)

            # Info selecció
            self._khp_info_lbl = tk.Label(btn_frame, text="Selecciona una calibració per aplicar-la",
                                          font=("Segoe UI", 9, "italic"),
                                          bg=COLORS["light"], fg=COLORS["dark"])
            self._khp_info_lbl.pack(side=tk.LEFT, padx=20, pady=15)

            # Botons dreta
            tk.Button(btn_frame, text="Tancar",
                     command=dialog.destroy,
                     bg=COLORS["dark"], fg=COLORS["white"],
                     font=("Segoe UI", 10),
                     relief="flat", padx=20, pady=5).pack(side=tk.RIGHT, padx=10, pady=15)

            self._khp_apply_btn = tk.Button(btn_frame, text="Aplicar Seleccionat",
                                            command=self._apply_khp_from_history,
                                            bg=COLORS["primary"], fg=COLORS["white"],
                                            font=("Segoe UI", 10, "bold"),
                                            relief="flat", padx=20, pady=5,
                                            state="disabled")
            self._khp_apply_btn.pack(side=tk.RIGHT, padx=10, pady=15)

            # Binds
            self._khp_tree.bind("<<TreeviewSelect>>", self._on_khp_select)
            self._khp_tree.bind("<Double-1>", lambda e: self._apply_khp_from_history())

        except Exception as e:
            import traceback
            messagebox.showerror("Error", f"Error carregant històric: {e}\n{traceback.format_exc()}")

    def _populate_khp_history(self):
        """Omple la taula d'històric KHP amb filtre aplicat."""
        # Netejar
        for item in self._khp_tree.get_children():
            self._khp_tree.delete(item)

        mode_filter = self._khp_mode_var.get()
        current_seq = os.path.basename(self.seq_path) if self.seq_path else ""

        # Ordenar per número de SEQ
        def get_seq_num(cal):
            match = re.search(r'(\d+)', cal.get('seq_name', ''))
            return int(match.group(1)) if match else 0

        sorted_history = sorted(self._khp_history_data, key=get_seq_num)

        for cal in sorted_history:
            # Filtrar per mode
            cal_mode = cal.get('mode', 'N/A')
            if mode_filter != "TOTS" and cal_mode != mode_filter:
                continue

            # Calcular factor si no existeix
            area = cal.get('area', 0)
            conc = cal.get('conc_ppm', 0)
            factor = cal.get('factor', conc / area if area > 0 else 0)

            # Determinar status mostrat
            valid_for_cal = cal.get('valid_for_calibration', True)
            valid_for_shift = cal.get('valid_for_shift', True)
            manual_override = cal.get('manual_override')
            stored_status = cal.get('status', 'OK')

            if manual_override is True:
                display_status = "MANUAL_OK"
            elif manual_override is False:
                display_status = "MANUAL_INV"
            elif not valid_for_cal:
                display_status = "INV_CAL"
            elif not valid_for_shift:
                display_status = "INV_SHIFT"
            else:
                display_status = stored_status if stored_status else "OK"

            values = (
                cal.get('seq_name', 'N/A'),
                cal_mode,
                f"KHP{conc:.0f}",
                f"{area:.1f}",
                f"{factor:.6f}" if factor else '-',
                f"{cal.get('shift_sec', 0):.1f}",
                f"{cal.get('snr', 0):.1f}" if cal.get('snr') else '-',
                display_status
            )

            # Determinar tag
            is_outlier = cal.get('is_outlier', False) or not valid_for_cal
            is_current = cal.get('seq_name', '') == self.khp_source_seq
            status = display_status

            if is_outlier:
                tag = 'outlier'
            elif is_current:
                tag = 'current'
            elif status in ['FAIL', 'INVALID']:
                tag = 'FAIL'
            elif status in ['WARN', 'WARNING', 'VALID_WITH_WARNINGS']:
                tag = 'WARN'
            else:
                tag = 'OK'

            self._khp_tree.insert('', 'end', values=values, tags=(tag,), iid=cal.get('seq_name', ''))

    def _filter_khp_history(self):
        """Re-filtra la taula d'històric."""
        self._populate_khp_history()

    def _sort_khp_history(self, col):
        """Ordena la taula per columna."""
        items = [(self._khp_tree.set(k, col), k) for k in self._khp_tree.get_children("")]
        try:
            items.sort(key=lambda t: float(t[0].replace(",", ".")), reverse=True)
        except ValueError:
            items.sort(reverse=True)
        for index, (_, k) in enumerate(items):
            self._khp_tree.move(k, "", index)

    def _on_khp_select(self, event):
        """Gestiona la selecció d'una fila de l'històric."""
        selection = self._khp_tree.selection()
        if selection:
            self._khp_apply_btn.configure(state="normal")
            seq_name = selection[0]
            # Trobar l'entrada
            for cal in self._khp_history_data:
                if cal.get("seq_name") == seq_name:
                    area = cal.get('area', 0)
                    conc = cal.get('conc_ppm', 0)
                    self._khp_info_lbl.configure(
                        text=f"Seleccionat: {seq_name} | Àrea: {area:.1f} | Conc: {conc} ppm"
                    )
                    break
        else:
            self._khp_apply_btn.configure(state="disabled")
            self._khp_info_lbl.configure(text="Selecciona una calibració per aplicar-la")

    def _apply_khp_from_history(self):
        """Aplica la calibració KHP seleccionada de l'històric."""
        selection = self._khp_tree.selection()
        if not selection:
            return

        seq_name = selection[0]
        selected_cal = None
        for cal in self._khp_history_data:
            if cal.get("seq_name") == seq_name:
                selected_cal = cal
                break

        if not selected_cal:
            messagebox.showerror("Error", "No s'ha trobat la calibració seleccionada.")
            return

        # Verificar que no és outlier
        if selected_cal.get('is_outlier', False):
            if not messagebox.askyesno("Avís",
                                       "Aquesta calibració està marcada com a OUTLIER.\n"
                                       "Segur que vols usar-la?"):
                return

        # Aplicar la calibració
        area = selected_cal.get('area', 0)
        conc = selected_cal.get('conc_ppm', 0)

        if area <= 0 or conc <= 0:
            messagebox.showerror("Error", "La calibració seleccionada no té dades vàlides.")
            return

        # Actualitzar les variables de calibració
        self.khp_area = area
        self.khp_conc = conc
        self.khp_source_seq = seq_name

        # Actualitzar la UI amb KHP aplicat
        self.lbl_khp_header.configure(
            text=f"KHP: {area:.1f} mAU·min ({conc} ppm) - de {seq_name}",
            fg=COLORS["primary"])

        # Recalcular concentracions si ja tenim resultats QC
        if self.selected_replicas:
            for name, data in self.selected_replicas.items():
                sel_area = data.get('area')
                if sel_area and self.khp_area and self.khp_conc:
                    data['concentration'] = (sel_area / self.khp_area) * self.khp_conc
                # Actualitzar info de calibració per cada mostra
                data['cal_khp_area'] = self.khp_area
                data['cal_khp_conc'] = self.khp_conc
                data['cal_khp_seq'] = self.khp_source_seq

        # Tancar diàleg
        self._khp_history_dialog.destroy()

        messagebox.showinfo("Calibració Aplicada",
                          f"S'ha aplicat la calibració de {seq_name}:\n\n"
                          f"Àrea KHP: {area:.1f}\n"
                          f"Concentració: {conc} ppm\n\n"
                          f"Les concentracions de les mostres s'han recalculat.")

    def _mark_as_outlier(self):
        """Marca la calibració actual com a outlier amb opció shift/calibració."""
        if not hasattr(self, 'calibration_data') or not self.calibration_data:
            messagebox.showerror("Error", "No hi ha calibració carregada.")
            return

        calibration = self.calibration_data.get('calibration', {})
        cal_id = calibration.get('cal_id')
        if not cal_id:
            messagebox.showerror("Error", "No s'ha trobat l'ID de la calibració.")
            return

        # Diàleg per triar tipus d'invalidació
        dialog = tk.Toplevel(self.root)
        dialog.title("Marcar com a invàlid")
        dialog.geometry("400x250")
        dialog.transient(self.root)
        dialog.grab_set()

        # Centrar
        dialog.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - 400) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - 250) // 2
        dialog.geometry(f"+{x}+{y}")

        ttk.Label(dialog, text="Què vols invalidar?", font=("Segoe UI", 11, "bold")).pack(pady=15)

        var_type = tk.StringVar(value="calibration")

        ttk.Radiobutton(dialog, text="Invàlid per CALIBRACIÓ (factor quantitatiu)\n→ No usar per calcular concentracions",
                       variable=var_type, value="calibration").pack(anchor="w", padx=30, pady=5)
        ttk.Radiobutton(dialog, text="Invàlid per SHIFT (alineació temporal)\n→ No usar per alinear senyals",
                       variable=var_type, value="shift").pack(anchor="w", padx=30, pady=5)
        ttk.Radiobutton(dialog, text="Invàlid per AMBDÓS",
                       variable=var_type, value="both").pack(anchor="w", padx=30, pady=5)

        ttk.Label(dialog, text="Motiu (opcional):").pack(anchor="w", padx=30, pady=(15, 5))
        reason_entry = ttk.Entry(dialog, width=40)
        reason_entry.pack(padx=30)

        def apply_override():
            override_type = var_type.get()
            reason = reason_entry.get().strip() or f"Marcat manualment ({override_type})"

            try:
                if override_type == "calibration":
                    # Només invalidar calibració, shift pot ser vàlid
                    result = set_calibration_override(self.seq_path, cal_id, False, reason)
                elif override_type == "shift":
                    # Només invalidar shift (cal actualitzar el camp manualment)
                    result = set_calibration_override(self.seq_path, cal_id, False, f"SHIFT: {reason}")
                else:  # both
                    result = set_calibration_override(self.seq_path, cal_id, False, f"AMBDÓS: {reason}")

                if result.get('success'):
                    messagebox.showinfo("Fet", result.get('message', 'Calibració marcada com a invàlida.'))
                    dialog.destroy()
                    # Refrescar històric
                    self._populate_khp_history()
                else:
                    messagebox.showerror("Error", result.get('message', 'Error desconegut'))
            except Exception as e:
                messagebox.showerror("Error", f"Error aplicant override: {e}")

        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=20)
        ttk.Button(btn_frame, text="Aplicar", command=apply_override).pack(side=tk.LEFT, padx=10)
        ttk.Button(btn_frame, text="Cancel·lar", command=dialog.destroy).pack(side=tk.LEFT, padx=10)

    # =========================================================================
    # EXECUCIÓ: PROCESSAR (Fase 3)
    # =========================================================================

    def _run_process(self):
        """Executa el processament de dades (Fase 3)."""
        if self.is_processing or not self.seq_path:
            return

        self.is_processing = True
        self.btn_processar.configure(state="disabled")
        self.progress_process['value'] = 0

        def process_thread():
            try:
                # TODO: Implementar processament real amb hpsec_process.py
                # Per ara, simplement activa el següent pas
                import time
                for i in range(10):
                    time.sleep(0.1)
                    pct = (i + 1) * 10
                    self.root.after(0, lambda p=pct: self._update_process_progress(p, "Processant..."))

                self.root.after(0, self._process_done)
            except Exception as e:
                import traceback
                self.root.after(0, lambda: self._process_error(str(e)))

        thread = threading.Thread(target=process_thread, daemon=True)
        thread.start()

    def _update_process_progress(self, pct, msg):
        """Actualitza progrés de processament."""
        self.progress_process['value'] = pct

    def _process_done(self):
        """Callback quan el processament acaba."""
        self.is_processing = False
        self.progress_process['value'] = 100
        self.btn_processar.configure(state="normal")
        self.btn_revisar.configure(state="normal")

        self.lbl_proc_alignment.configure(text="Alineacio: Aplicada")
        self.lbl_proc_baseline.configure(text="Baseline: Corregit")
        self.lbl_proc_peaks.configure(text="Pics detectats: OK")
        self.lbl_proc_areas.configure(text="Arees calculades: OK")

    def _process_error(self, error):
        """Callback quan hi ha error de processament."""
        self.is_processing = False
        self.btn_processar.configure(state="normal")
        messagebox.showerror("Error", f"Error durant el processament:\n\n{error}")

    # =========================================================================
    # EXECUCIÓ: REVISAR (Fase 4)
    # =========================================================================

    def _run_review(self):
        """Executa la revisió de rèpliques (Fase 4)."""
        if self.is_processing or not self.seq_path:
            return

        self.is_processing = True
        self.btn_revisar.configure(state="disabled")
        self.progress_review['value'] = 0

        # Netejar taula
        for item in self.tree_qc.get_children():
            self.tree_qc.delete(item)

        def qc_thread():
            try:
                self._run_review_process()
            except Exception as e:
                import traceback
                self.root.after(0, lambda: self._review_error(str(e) + "\n" + traceback.format_exc()))

        thread = threading.Thread(target=qc_thread, daemon=True)
        thread.start()

    def _run_review_process(self):
        """
        Carrega i mostra resultats del processament des del JSON.

        NO fa processament - només llegeix processing_summary.json
        generat per hpsec_pipeline.py
        """
        # Buscar processing_summary.json
        check_folder = os.path.join(self.seq_path, "CHECK")
        summary_path = os.path.join(check_folder, "processing_summary.json")

        if not os.path.exists(summary_path):
            self.root.after(0, lambda: self._review_error(
                "No s'ha trobat processing_summary.json\n\n"
                "Cal executar primer el processament amb hpsec_pipeline.py"
            ))
            return

        # Llegir JSON
        try:
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary = json.load(f)
        except Exception as e:
            self.root.after(0, lambda: self._review_error(f"Error llegint JSON: {e}"))
            return

        samples = summary.get('samples', [])
        total = len(samples)

        if total == 0:
            self.root.after(0, lambda: self._review_error("No hi ha mostres al JSON"))
            return

        # Processar cada mostra del JSON
        for i, sample in enumerate(samples):
            pct = int(100 * (i + 1) / total)
            sample_name = sample.get('name', 'N/A')
            self.root.after(0, lambda p=pct, s=sample_name: self._update_review_progress(p, s))

            # Extreure dades del JSON
            selected_doc = sample.get('selected_doc') or sample.get('best_replica')
            selected_dad = sample.get('selected_dad') or selected_doc
            selected = sample.get('selected', selected_doc)

            # Comparativa
            comparison = sample.get('comparison', {})
            doc_r = comparison.get('pearson')
            doc_diff = comparison.get('area_diff_pct')

            # Determinar estat
            confidence = sample.get('confidence', 0)
            if selected and confidence >= 0.8:
                estat = "OK"
                tag = "OK"
            elif selected and confidence >= 0.5:
                estat = "CHECK"
                tag = "WARN"
            elif selected:
                estat = "CHECK"
                tag = "WARN"
            else:
                estat = "FAIL"
                tag = "FAIL"

            # Si és MIXED, marcar com CHECK
            if selected == "MIXED":
                estat = "MIXED"
                tag = "WARN"

            # Qualitat DAD
            dad_quality = "-"
            if selected_dad:
                rep_data = sample.get('replicas', {}).get(selected_dad, {})
                dad_eval = rep_data.get('dad_eval', {})
                if dad_eval:
                    dad_quality = dad_eval.get('quality', '-')

            # Formatar valors
            doc_r_str = f"{doc_r:.3f}" if doc_r is not None else "-"
            doc_diff_str = f"{doc_diff:.1f}%" if doc_diff is not None else "-"

            # Formatar selecció
            sel_doc_str = selected_doc if selected_doc else "-"
            sel_dad_str = selected_dad if selected_dad else "-"

            values = (sample_name, estat, doc_r_str, doc_diff_str,
                     sel_doc_str, dad_quality, sel_dad_str)

            # Guardar resultats per ús posterior
            self.selected_replicas[sample_name] = {
                "sel_doc": selected_doc,
                "sel_dad": selected_dad,
                "selected": selected,
                "doc_r": doc_r,
                "doc_diff": doc_diff,
                "confidence": confidence,
                "areas_fraction": sample.get('areas_fraction', {}),
                "concentration": sample.get('concentration_ppm'),
                "conc_fraction": sample.get('conc_fraction', {}),
                "transformation": sample.get('transformation', {}),
                "replicas": sample.get('replicas', {}),
                "warnings": sample.get('warnings', []),
            }

            # Actualitzar taula
            self.root.after(0, lambda v=values, t=tag: self._add_qc_row(v, t))

        self.root.after(0, self._review_done)

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

    def _update_review_progress(self, pct, sample):
        """Actualitza progrés de QC."""
        self.progress_review['value'] = pct

    def _add_qc_row(self, values, tag):
        """Afegeix una fila a la taula QC."""
        self.tree_qc.insert("", "end", values=values, tags=(tag,))

    def _review_done(self):
        """Callback quan la revisió acaba."""
        self.is_processing = False
        self.progress_review['value'] = 100
        self.btn_export.configure(state="normal")

        n_ok = sum(1 for k, v in self.selected_replicas.items()
                   if v.get('doc_r') and v['doc_r'] >= DEFAULT_MIN_CORR)
        total = len(self.selected_replicas)

        messagebox.showinfo("Revisio Completada",
                           f"Revisio de repliques completada!\n\n"
                           f"Mostres: {total}\n"
                           f"OK: {n_ok}\n"
                           f"Revisar: {total - n_ok}")

    def _review_error(self, error):
        """Callback quan hi ha error de revisió."""
        self.is_processing = False
        self.btn_revisar.configure(state="normal")
        messagebox.showerror("Error", f"Error durant la revisio:\n\n{error}")

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

        # Info de calibració activa (pot ser diferent de la local)
        active_calibration = {
            "khp_area": self.khp_area,
            "khp_conc_ppm": self.khp_conc,
            "khp_source_seq": self.khp_source_seq,
            "factor": self.khp_conc / self.khp_area if self.khp_area and self.khp_conc else None,
        }

        summary = {
            "seq_path": self.seq_path,
            "seq_name": os.path.basename(self.seq_path),
            "generated_at": datetime.now().isoformat(),
            "calibration_local": self.calibration_data,
            "calibration_active": active_calibration,
            "samples": []
        }

        for name, data in self.selected_replicas.items():
            sample_entry = {
                "name": name,
                "sel_doc": data.get('sel_doc'),
                "sel_dad": data.get('sel_dad'),
                "doc_r": data.get('doc_r'),
                "doc_diff": data.get('doc_diff'),
                "area_total": data.get('area'),
                "concentration_ppm": data.get('concentration'),
                "areas_fraction": data.get('areas_fraction', {}),
                "calibration": {
                    "khp_area": data.get('cal_khp_area'),
                    "khp_conc_ppm": data.get('cal_khp_conc'),
                    "khp_source_seq": data.get('cal_khp_seq'),
                }
            }
            summary["samples"].append(sample_entry)

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
        """Exporta resultats a Excel amb concentracions i àrees per fracció."""
        rows = []
        for name, data in self.selected_replicas.items():
            row = {
                "Mostra": name,
                "Sel_DOC": f"R{data.get('sel_doc', '?')}",
                "Sel_DAD": f"R{data.get('sel_dad', '?')}",
                "Pearson_R": data.get('doc_r'),
                "Diff_Pct": data.get('doc_diff'),
                "Area_Total": data.get('area'),
            }

            # Afegir àrees per fracció
            fractions = data.get('areas_fraction', {})
            for frac in ['BioP', 'HS', 'BB', 'SB', 'LMW']:
                row[f'Area_{frac}'] = fractions.get(frac)

            # Afegir concentració si disponible
            if data.get('concentration') is not None:
                row["Conc_ppm"] = data.get('concentration')

            rows.append(row)

        df = pd.DataFrame(rows)

        # Afegir info KHP a la capçalera
        with pd.ExcelWriter(path, engine='openpyxl') as writer:
            # Escriure dades
            df.to_excel(writer, index=False, sheet_name='Resultats', startrow=3)

            # Afegir capçalera amb info KHP
            ws = writer.sheets['Resultats']
            ws['A1'] = f"SEQ: {os.path.basename(self.seq_path)}"
            if self.khp_area and self.khp_conc:
                source_info = f" (de {self.khp_source_seq})" if self.khp_source_seq else ""
                ws['A2'] = f"KHP: Àrea={self.khp_area:.1f}, Conc={self.khp_conc} ppm{source_info}"
                factor = self.khp_conc / self.khp_area
                ws['A3'] = f"Factor: {factor:.6f} | Fórmula: Conc = Àrea_mostra × {factor:.6f}"

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

    def _open_planner(self):
        """Obre el planificador de seqüències."""
        try:
            from hpsec_planner_gui import SequencePlannerGUI
            planner = SequencePlannerGUI(parent=self.root)
        except ImportError as e:
            messagebox.showerror("Error", f"No s'ha pogut carregar el planificador:\n{e}")
        except Exception as e:
            messagebox.showerror("Error", f"Error obrint planificador:\n{e}")

    def _open_comparador(self):
        """Obre el comparador UIB/Direct."""
        # TODO: Implementar comparador
        messagebox.showinfo("Comparador", "Funcionalitat pendent d'implementar.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    root = tk.Tk()
    app = HPSECSuiteV3(root)
    root.mainloop()


if __name__ == "__main__":
    main()
