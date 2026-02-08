# -*- coding: utf-8 -*-
"""
hpsec_planner_gui.py - Planificador de Seqüències HPSEC (GUI v3)
================================================================

Planificador complet de seqüències amb:
- Mostres (sempre duplicats R1/R2)
- KHP (patró, duplicat)
- MQ (control)
- NaOH (neteja)

Funcionalitats:
- Timeline visual amb tipus d'injecció
- Predicció timeouts
- Warnings per mostres en zona crítica
- Suggeriments de posició MQ/NaOH
- Comparació escenaris

v3.0 - 2026-01-29 - Seqüències completes amb controls
"""

import os
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional, Dict
from enum import Enum

import numpy as np

# Importar funcions del planner base
from hpsec_planner import (
    get_zones, get_critical_zone,
    TOC_CYCLE_MIN
)

# Matplotlib
try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle, FancyBboxPatch
    from matplotlib.lines import Line2D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# =============================================================================
# CONSTANTS I CONFIGURACIÓ
# =============================================================================

# Duracions per defecte (minuts)
DURATIONS = {
    "COLUMN": {
        "sample": 78.65,      # Mostra COLUMN
        "khp": 78.65,         # KHP COLUMN
        "mq": 78.65,          # MQ COLUMN
        "naoh": 78.65,        # NaOH COLUMN
        "optimal": 77.2,      # Duració òptima
    },
    "BP": {
        "sample": 15.0,       # Mostra BP
        "khp": 15.0,          # KHP BP
        "mq": 15.0,           # MQ BP
        "naoh": 15.0,         # NaOH BP
        "optimal": 15.0,
    }
}

# Colors per tipus d'injecció
INJECTION_COLORS = {
    "SAMPLE": "#3498DB",      # Blau
    "KHP": "#9B59B6",         # Violeta
    "MQ": "#1ABC9C",          # Verd menta
    "NAOH": "#E67E22",        # Taronja
}

# Colors severitat
SEVERITY_COLORS = {
    "CRITICAL": "#E74C3C",
    "WARNING": "#F39C12",
    "INFO": "#3498DB",
    "OK": "#27AE60",
}

# Colors zones cromatograma
ZONE_COLORS = {
    "EARLY": "#FFC0CB",   # Rosa clar (pre-pic)
    "BioP": "#FF6B6B",
    "HS": "#EE5A24",
    "BB": "#F9CA24",
    "SB": "#A3CB38",
    "LMW": "#1289A7",
    "POST_RUN": "#C4E538",
    "MAIN": "#EE5A24",
    "TAIL": "#A3CB38",
}

# Colors UI
COLORS = {
    "bg": "#F5F6FA",
    "card": "#FFFFFF",
    "text": "#2C3E50",
    "text_muted": "#7F8C8D",
    "border": "#BDC3C7",
    "accent": "#3498DB",
}


# =============================================================================
# MODELS DE DADES
# =============================================================================

class InjectionType(Enum):
    SAMPLE = "SAMPLE"
    KHP = "KHP"
    MQ = "MQ"
    NAOH = "NAOH"


@dataclass
class Injection:
    """Representa una injecció a la seqüència."""
    position: int           # Posició a la seqüència (1, 2, 3...)
    inj_type: InjectionType # Tipus
    name: str               # Nom (Sample1_R1, KHP_R1, MQ, etc.)
    replica: Optional[int]  # 1 o 2 per mostres/KHP, None per MQ/NaOH
    sample_num: Optional[int]  # Número de mostra (1, 2, 3...)
    duration: float         # Duració (min)
    start_time: float       # Temps inici absolut (min)
    end_time: float         # Temps final absolut (min)
    timeout_pos: Optional[float]  # Posició timeout dins injecció (min) o None
    timeout_zone: Optional[str]   # Zona del timeout
    timeout_severity: str   # OK, INFO, WARNING, CRITICAL
    in_critical: bool       # True si timeout en zona crítica


@dataclass
class SequencePlan:
    """Pla complet de seqüència."""
    mode: str                    # COLUMN o BP
    n_samples: int               # Nombre de mostres
    injections: List[Injection]  # Llista d'injeccions
    total_duration: float        # Duració total (min)
    total_injections: int        # Nombre total d'injeccions

    # Estadístiques
    samples_in_critical: int     # Mostres en zona crítica
    samples_in_warning: int      # Mostres en zona warning
    khp_in_critical: int         # KHP en zona crítica
    controls_in_critical: int    # Controls en zona crítica


# =============================================================================
# GENERADOR DE SEQÜÈNCIES
# =============================================================================

class SequenceGenerator:
    """Genera i analitza seqüències HPSEC."""

    def __init__(self, mode: str = "COLUMN"):
        self.mode = mode
        self.durations = DURATIONS[mode]
        self.zones = get_zones(mode)
        self.critical_start, self.critical_end = get_critical_zone(mode)

    def generate_sequence(self,
                          n_samples: int,
                          khp_start: bool = True,
                          khp_end: bool = True,
                          mq_positions: List[int] = None,    # Llista posicions MQ (per injeccio)
                          naoh_positions: List[int] = None,  # Llista posicions NaOH (per injeccio)
                          mq_at_start: bool = False,
                          naoh_at_end: bool = False,
                          t0: float = 40.0,
                          sample_duration: Optional[float] = None,
                          ) -> SequencePlan:
        """
        Genera una sequencia completa.

        Args:
            n_samples: Nombre de mostres (cada una amb R1 i R2 = 2 injeccions)
            khp_start: KHP al principi (duplicat = 2 injeccions)
            khp_end: KHP al final (duplicat = 2 injeccions)
            mq_positions: Llista d'INJECCIONS despres de les quals inserir MQ
            naoh_positions: Llista d'INJECCIONS despres de les quals inserir NaOH
            mq_at_start: MQ al principi
            naoh_at_end: NaOH al final
            t0: Temps des de l'ultim reload TOC (min)
            sample_duration: Duracio per injeccio (si None, usa default)

        Returns:
            SequencePlan amb tota la informacio
        """
        duration = sample_duration or self.durations["sample"]
        mq_positions = set(mq_positions or [])
        naoh_positions = set(naoh_positions or [])

        injections = []
        current_time = 0.0
        inj_num = 0  # Comptador d'injeccions (per inserir MQ/NaOH)

        # === MQ INICIAL ===
        if mq_at_start:
            inj_num += 1
            inj = self._create_injection(
                inj_num, InjectionType.MQ, f"MQ_0", None, None,
                self.durations["mq"], current_time, t0
            )
            injections.append(inj)
            current_time = inj.end_time

        # === KHP INICIAL ===
        if khp_start:
            for replica in [1, 2]:
                inj_num += 1
                inj = self._create_injection(
                    inj_num, InjectionType.KHP, f"KHP_R{replica}", replica, None,
                    self.durations["khp"], current_time, t0
                )
                injections.append(inj)
                current_time = inj.end_time

                # MQ/NaOH despres d'aquesta injeccio?
                if inj_num in mq_positions:
                    inj_num += 1
                    ctrl = self._create_injection(
                        inj_num, InjectionType.MQ, f"MQ_{inj_num-1}", None, None,
                        self.durations["mq"], current_time, t0
                    )
                    injections.append(ctrl)
                    current_time = ctrl.end_time

                if inj_num in naoh_positions:
                    inj_num += 1
                    ctrl = self._create_injection(
                        inj_num, InjectionType.NAOH, f"NaOH_{inj_num-1}", None, None,
                        self.durations["naoh"], current_time, t0
                    )
                    injections.append(ctrl)
                    current_time = ctrl.end_time

        # === MOSTRES (cada mostra = 2 injeccions: R1 + R2) ===
        for sample_num in range(1, n_samples + 1):
            for replica in [1, 2]:
                inj_num += 1
                inj = self._create_injection(
                    inj_num, InjectionType.SAMPLE, f"M{sample_num}_R{replica}",
                    replica, sample_num, duration, current_time, t0
                )
                injections.append(inj)
                current_time = inj.end_time

                # MQ despres d'aquesta injeccio?
                if inj_num in mq_positions:
                    inj_num += 1
                    ctrl = self._create_injection(
                        inj_num, InjectionType.MQ, f"MQ_{inj_num-1}", None, None,
                        self.durations["mq"], current_time, t0
                    )
                    injections.append(ctrl)
                    current_time = ctrl.end_time

                # NaOH despres d'aquesta injeccio?
                if inj_num in naoh_positions:
                    inj_num += 1
                    ctrl = self._create_injection(
                        inj_num, InjectionType.NAOH, f"NaOH_{inj_num-1}", None, None,
                        self.durations["naoh"], current_time, t0
                    )
                    injections.append(ctrl)
                    current_time = ctrl.end_time

        # === KHP FINAL ===
        if khp_end:
            for replica in [1, 2]:
                inj_num += 1
                inj = self._create_injection(
                    inj_num, InjectionType.KHP, f"KHP_end_R{replica}", replica, None,
                    self.durations["khp"], current_time, t0
                )
                injections.append(inj)
                current_time = inj.end_time

                # MQ/NaOH despres d'aquesta injeccio?
                if inj_num in mq_positions:
                    inj_num += 1
                    ctrl = self._create_injection(
                        inj_num, InjectionType.MQ, f"MQ_{inj_num-1}", None, None,
                        self.durations["mq"], current_time, t0
                    )
                    injections.append(ctrl)
                    current_time = ctrl.end_time

                if inj_num in naoh_positions:
                    inj_num += 1
                    ctrl = self._create_injection(
                        inj_num, InjectionType.NAOH, f"NaOH_{inj_num-1}", None, None,
                        self.durations["naoh"], current_time, t0
                    )
                    injections.append(ctrl)
                    current_time = ctrl.end_time

        # === NAOH FINAL ===
        if naoh_at_end:
            inj_num += 1
            inj = self._create_injection(
                inj_num, InjectionType.NAOH, "NaOH_end", None, None,
                self.durations["naoh"], current_time, t0
            )
            injections.append(inj)
            current_time = inj.end_time

        # === CALCULAR ESTADÍSTIQUES ===
        # Critical: timeout dins zona crítica O severitat CRITICAL
        samples_crit = sum(1 for i in injections
                          if i.inj_type == InjectionType.SAMPLE and
                          (i.in_critical or i.timeout_severity == "CRITICAL"))
        # Warning: severitat WARNING i NO crític
        samples_warn = sum(1 for i in injections
                          if i.inj_type == InjectionType.SAMPLE and
                          i.timeout_severity == "WARNING" and not i.in_critical)
        khp_crit = sum(1 for i in injections
                       if i.inj_type == InjectionType.KHP and
                       (i.in_critical or i.timeout_severity == "CRITICAL"))
        controls_crit = sum(1 for i in injections
                           if i.inj_type in [InjectionType.MQ, InjectionType.NAOH] and
                           (i.in_critical or i.timeout_severity == "CRITICAL"))

        return SequencePlan(
            mode=self.mode,
            n_samples=n_samples,
            injections=injections,
            total_duration=current_time,
            total_injections=len(injections),
            samples_in_critical=samples_crit,
            samples_in_warning=samples_warn,
            khp_in_critical=khp_crit,
            controls_in_critical=controls_crit,
        )

    def _create_injection(self, position: int, inj_type: InjectionType,
                          name: str, replica: Optional[int], sample_num: Optional[int],
                          duration: float, start_time: float, t0: float) -> Injection:
        """Crea una injecció amb càlcul de timeout."""
        end_time = start_time + duration

        # Calcular timeout dins aquesta injecció
        timeout_pos = self._calculate_timeout_in_injection(start_time, end_time, t0)

        # Determinar zona i severitat
        timeout_zone = None
        timeout_severity = "OK"
        in_critical = False

        if timeout_pos is not None:
            for zname, zdata in self.zones.items():
                if zdata["start"] <= timeout_pos < zdata["end"]:
                    timeout_zone = zname
                    timeout_severity = zdata["severity"]
                    break

            in_critical = self.critical_start <= timeout_pos < self.critical_end

        return Injection(
            position=position,
            inj_type=inj_type,
            name=name,
            replica=replica,
            sample_num=sample_num,
            duration=duration,
            start_time=start_time,
            end_time=end_time,
            timeout_pos=timeout_pos,
            timeout_zone=timeout_zone,
            timeout_severity=timeout_severity,
            in_critical=in_critical,
        )

    def _calculate_timeout_in_injection(self, start_time: float, end_time: float,
                                         t0: float) -> Optional[float]:
        """
        Calcula si hi ha timeout dins una injeccio i retorna la posicio relativa.

        T0 = temps transcorregut des de l'ultim reload TOC quan comenca la sequencia.
        Primer timeout = (77.2 - T0) minuts des de l'inici de la sequencia.
        Timeouts posteriors cada 77.2 min.
        """
        # Temps fins al primer timeout des de l'inici de la sequencia
        time_to_first = TOC_CYCLE_MIN - t0
        if time_to_first <= 0:
            time_to_first = TOC_CYCLE_MIN  # T0=77.2 significa que acaba de fer reload

        # Timeouts ocorren a: time_to_first, time_to_first + 77.2, time_to_first + 2*77.2...
        # Buscar el primer timeout >= start_time
        if time_to_first >= start_time:
            first_timeout = time_to_first
        else:
            cycles = int((start_time - time_to_first) / TOC_CYCLE_MIN)
            first_timeout = time_to_first + (cycles + 1) * TOC_CYCLE_MIN

        # Si el timeout cau dins la injeccio, retornar posicio relativa
        if first_timeout < end_time:
            return first_timeout - start_time

        return None

    def suggest_control_positions(self, plan: SequencePlan) -> List[Dict]:
        """
        Suggereix posicions per afegir MQ/NaOH per evitar timeouts crítics.

        Returns:
            Llista de suggeriments amb posició i raó
        """
        suggestions = []

        # Trobar mostres en zona crítica
        critical_samples = [i for i in plan.injections
                           if i.inj_type == InjectionType.SAMPLE and i.in_critical]

        if not critical_samples:
            return suggestions

        # Agrupar consecutives
        groups = []
        current_group = []

        for inj in critical_samples:
            if not current_group or inj.position == current_group[-1].position + 1:
                current_group.append(inj)
            else:
                if current_group:
                    groups.append(current_group)
                current_group = [inj]
        if current_group:
            groups.append(current_group)

        # Generar suggeriments
        for group in groups:
            first = group[0]
            last = group[-1]

            # Suggerir inserir control abans del primer afectat
            suggestions.append({
                "insert_before": first.position,
                "injection_name": first.name,
                "type": "MQ",  # O NaOH
                "reason": f"Timeout en zona {first.timeout_zone} ({first.timeout_pos:.1f} min)",
                "affected": [i.name for i in group],
                "effect": "Desplaça timeout ~77 min endavant",
            })

        return suggestions


# =============================================================================
# GUI PRINCIPAL
# =============================================================================

class SequencePlannerGUI:
    """GUI per planificar seqüències HPSEC."""

    def __init__(self, parent=None):
        self.standalone = parent is None

        if self.standalone:
            self.root = tk.Tk()
            self.root.title("HPSEC Sequence Planner")
        else:
            self.root = tk.Toplevel(parent)
            self.root.title("Planificador de Seqüències")
            self.root.transient(parent)

        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        self.root.configure(bg=COLORS["bg"])

        # Variables
        self.var_mode = tk.StringVar(value="COLUMN")
        self.var_samples = tk.IntVar(value=10)
        self.var_duration = tk.DoubleVar(value=78.65)
        self.var_t0 = tk.DoubleVar(value=40.0)

        # KHP
        self.var_khp_start = tk.BooleanVar(value=True)
        self.var_khp_end = tk.BooleanVar(value=True)

        # Controls (opcions fixes)
        self.var_mq_start = tk.BooleanVar(value=False)
        self.var_naoh_end = tk.BooleanVar(value=True)

        # Auto-calc
        self.var_auto_calc = tk.BooleanVar(value=True)

        # Dades
        self.generator = SequenceGenerator("COLUMN")
        self.plan = None
        self.plan_column = None  # Per comparació dual
        self.plan_bp = None      # Per comparació dual

        # Construir UI
        self._setup_styles()
        self._build_ui()

        # Calcular inicial
        self.root.after(100, self._calculate)

    def _setup_styles(self):
        """Configura estils."""
        style = ttk.Style()
        style.configure("Card.TFrame", background=COLORS["card"])
        style.configure("Card.TLabel", background=COLORS["card"],
                       font=("Segoe UI", 10))
        style.configure("CardTitle.TLabel", background=COLORS["card"],
                       font=("Segoe UI", 11, "bold"))
        style.configure("CardMuted.TLabel", background=COLORS["card"],
                       font=("Segoe UI", 9), foreground=COLORS["text_muted"])
        style.configure("Big.TLabel", font=("Segoe UI", 28, "bold"))
        style.configure("Stat.TLabel", font=("Segoe UI", 14))

    def _build_ui(self):
        """Construeix la interfície."""
        # Container principal
        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Layout: esquerra (config) | dreta (resultats)
        main.columnconfigure(0, weight=0, minsize=380)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        # === ESQUERRA: Configuració (amb scroll) ===
        left_container = ttk.Frame(main)
        left_container.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        # Canvas amb scrollbar
        canvas = tk.Canvas(left_container, width=360, highlightthickness=0)
        scrollbar = ttk.Scrollbar(left_container, orient="vertical", command=canvas.yview)
        left = ttk.Frame(canvas)

        left.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=left, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Scroll amb roda del ratolí
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        self._build_mode_card(left)
        self._build_samples_card(left)
        self._build_khp_card(left)
        self._build_controls_card(left)
        self._build_timing_card(left)

        # === DRETA: Resultats ===
        right = ttk.Frame(main)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        self._build_summary_bar(right)
        self._build_results_notebook(right)
        self._build_bottom_bar(right)

    def _create_card(self, parent, title):
        """Crea una card."""
        card = ttk.Frame(parent, style="Card.TFrame", padding=12)
        card.pack(fill=tk.X, pady=(0, 8))

        if title:
            ttk.Label(card, text=title, style="CardTitle.TLabel").pack(anchor="w")
            ttk.Separator(card, orient="horizontal").pack(fill=tk.X, pady=(5, 8))

        return card

    def _build_mode_card(self, parent):
        """Card de selecció de mode."""
        card = self._create_card(parent, "Mode")

        frame = ttk.Frame(card, style="Card.TFrame")
        frame.pack(fill=tk.X)

        for mode in ["COLUMN", "BP"]:
            rb = ttk.Radiobutton(frame, text=mode, variable=self.var_mode,
                                value=mode, command=self._on_mode_change)
            rb.pack(side=tk.LEFT, padx=(0, 20))

        # Info duració
        self.lbl_mode_info = ttk.Label(card, text="", style="CardMuted.TLabel")
        self.lbl_mode_info.pack(anchor="w", pady=(5, 0))
        self._update_mode_info()

    def _build_samples_card(self, parent):
        """Card de mostres."""
        card = self._create_card(parent, "Mostres")

        # Nombre de mostres
        row = ttk.Frame(card, style="Card.TFrame")
        row.pack(fill=tk.X, pady=3)

        ttk.Label(row, text="Nombre de mostres:", style="Card.TLabel").pack(side=tk.LEFT)

        spin = ttk.Spinbox(row, from_=1, to=100, textvariable=self.var_samples,
                          width=6, command=self._on_param_change)
        spin.pack(side=tk.RIGHT)
        spin.bind("<Return>", lambda e: self._on_param_change())

        ttk.Label(card, text="(cada mostra = 2 injeccions: R1 + R2)",
                 style="CardMuted.TLabel").pack(anchor="w")

    def _build_khp_card(self, parent):
        """Card de KHP."""
        card = self._create_card(parent, "Patró KHP")

        ttk.Checkbutton(card, text="KHP al principi (duplicat)",
                       variable=self.var_khp_start,
                       command=self._on_param_change).pack(anchor="w")

        ttk.Checkbutton(card, text="KHP al final (duplicat)",
                       variable=self.var_khp_end,
                       command=self._on_param_change).pack(anchor="w")

    def _build_controls_card(self, parent):
        """Card de controls (MQ/NaOH)."""
        card = self._create_card(parent, "Controls MQ/NaOH")

        # Explicacio
        ttk.Label(card, text="Inserir MQ despres d'INJECCIO #:",
                 style="Card.TLabel").pack(anchor="w")
        self.var_mq_positions = tk.StringVar(value="")
        mq_entry = ttk.Entry(card, textvariable=self.var_mq_positions, width=20)
        mq_entry.pack(anchor="w", pady=(2, 0))
        ttk.Label(card, text="Ex: 5,10 = MQ despres inj 5 i 10",
                 style="CardMuted.TLabel").pack(anchor="w")

        ttk.Separator(card, orient="horizontal").pack(fill=tk.X, pady=6)

        ttk.Label(card, text="Inserir NaOH despres d'INJECCIO #:",
                 style="Card.TLabel").pack(anchor="w")
        self.var_naoh_positions = tk.StringVar(value="")
        naoh_entry = ttk.Entry(card, textvariable=self.var_naoh_positions, width=20)
        naoh_entry.pack(anchor="w", pady=(2, 0))
        ttk.Label(card, text="Ex: 10 = NaOH despres inj 10",
                 style="CardMuted.TLabel").pack(anchor="w")

        ttk.Separator(card, orient="horizontal").pack(fill=tk.X, pady=6)

        ttk.Checkbutton(card, text="MQ al principi (abans KHP)",
                       variable=self.var_mq_start,
                       command=self._on_param_change).pack(anchor="w")
        ttk.Checkbutton(card, text="NaOH al final (despres KHP)",
                       variable=self.var_naoh_end,
                       command=self._on_param_change).pack(anchor="w")

    def _build_timing_card(self, parent):
        """Card de timing."""
        self.timing_card = self._create_card(parent, "Timing TOC")

        # Duració
        row1 = ttk.Frame(self.timing_card, style="Card.TFrame")
        row1.pack(fill=tk.X, pady=3)

        ttk.Label(row1, text="Duracio mostra (min):", style="Card.TLabel").pack(side=tk.LEFT)

        self.dur_frame = ttk.Frame(row1, style="Card.TFrame")
        self.dur_frame.pack(side=tk.RIGHT)

        ttk.Entry(self.dur_frame, textvariable=self.var_duration, width=8).pack(side=tk.LEFT)

        # Botons duració (es recreen quan canvia mode)
        self.dur_btn_frame = ttk.Frame(self.dur_frame, style="Card.TFrame")
        self.dur_btn_frame.pack(side=tk.LEFT)
        self._update_duration_buttons()

        # T0
        row2 = ttk.Frame(self.timing_card, style="Card.TFrame")
        row2.pack(fill=tk.X, pady=3)

        ttk.Label(row2, text="T0 (min des d'ultim reload):", style="Card.TLabel").pack(side=tk.LEFT)

        t0_spin = ttk.Spinbox(row2, from_=0, to=77, textvariable=self.var_t0,
                             width=6, command=self._on_param_change)
        t0_spin.pack(side=tk.RIGHT)

        ttk.Label(self.timing_card, text=f"T0 petit = mes temps fins timeout",
                 style="CardMuted.TLabel").pack(anchor="w")
        ttk.Label(self.timing_card, text=f"Cicle TOC: {TOC_CYCLE_MIN} min",
                 style="CardMuted.TLabel").pack(anchor="w")

    def _update_duration_buttons(self):
        """Actualitza botons duració segons mode."""
        # Esborrar botons anteriors
        for w in self.dur_btn_frame.winfo_children():
            w.destroy()

        mode = self.var_mode.get()
        if mode == "COLUMN":
            ttk.Button(self.dur_btn_frame, text="77.2", width=5,
                      command=lambda: self._set_duration(77.2)).pack(side=tk.LEFT, padx=2)
            ttk.Button(self.dur_btn_frame, text="78.65", width=5,
                      command=lambda: self._set_duration(78.65)).pack(side=tk.LEFT, padx=2)
        else:  # BP
            ttk.Button(self.dur_btn_frame, text="12", width=4,
                      command=lambda: self._set_duration(12.0)).pack(side=tk.LEFT, padx=2)
            ttk.Button(self.dur_btn_frame, text="15", width=4,
                      command=lambda: self._set_duration(15.0)).pack(side=tk.LEFT, padx=2)

    def _build_summary_bar(self, parent):
        """Barra de resum superior."""
        frame = ttk.Frame(parent, style="Card.TFrame", padding=15)
        frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        # Stats principals
        stats_frame = ttk.Frame(frame, style="Card.TFrame")
        stats_frame.pack(fill=tk.X)

        # Total injeccions
        col1 = ttk.Frame(stats_frame, style="Card.TFrame")
        col1.pack(side=tk.LEFT, expand=True)
        self.lbl_total_inj = ttk.Label(col1, text="--", style="Big.TLabel")
        self.lbl_total_inj.pack()
        ttk.Label(col1, text="Injeccions", style="CardMuted.TLabel").pack()

        # Durada total
        col2 = ttk.Frame(stats_frame, style="Card.TFrame")
        col2.pack(side=tk.LEFT, expand=True)
        self.lbl_total_time = ttk.Label(col2, text="--", style="Big.TLabel")
        self.lbl_total_time.pack()
        ttk.Label(col2, text="Hores", style="CardMuted.TLabel").pack()

        # Mostres crítiques
        col3 = ttk.Frame(stats_frame, style="Card.TFrame")
        col3.pack(side=tk.LEFT, expand=True)
        self.lbl_critical = ttk.Label(col3, text="--", style="Big.TLabel",
                                      foreground=SEVERITY_COLORS["CRITICAL"])
        self.lbl_critical.pack()
        ttk.Label(col3, text="Mostres crítiques", style="CardMuted.TLabel").pack()

        # Mostres warning
        col4 = ttk.Frame(stats_frame, style="Card.TFrame")
        col4.pack(side=tk.LEFT, expand=True)
        self.lbl_warning = ttk.Label(col4, text="--", style="Big.TLabel",
                                     foreground=SEVERITY_COLORS["WARNING"])
        self.lbl_warning.pack()
        ttk.Label(col4, text="Mostres warning", style="CardMuted.TLabel").pack()

    def _build_results_notebook(self, parent):
        """Notebook de resultats."""
        self.notebook = ttk.Notebook(parent)
        self.notebook.grid(row=1, column=0, sticky="nsew")

        # Tab Timeline
        self.tab_timeline = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_timeline, text="Timeline")
        self._build_timeline_tab()

        # Tab Llista
        self.tab_list = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_list, text="Llista Injeccions")
        self._build_list_tab()

        # Tab Suggeriments
        self.tab_suggestions = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_suggestions, text="Suggeriments")
        self._build_suggestions_tab()

        # Tab Comparació COLUMN vs BP
        self.tab_compare = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_compare, text="COLUMN vs BP")
        self._build_compare_tab()

    def _build_timeline_tab(self):
        """Tab amb timeline visual."""
        if not HAS_MATPLOTLIB:
            ttk.Label(self.tab_timeline, text="Matplotlib no disponible").pack(expand=True)
            return

        self.fig = Figure(figsize=(12, 6), dpi=100, facecolor=COLORS["bg"])
        self.ax = self.fig.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.tab_timeline)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Toolbar
        toolbar_frame = ttk.Frame(self.tab_timeline)
        toolbar_frame.pack(fill=tk.X)
        NavigationToolbar2Tk(self.canvas, toolbar_frame)

        # Llegenda
        legend_frame = ttk.Frame(self.tab_timeline, style="Card.TFrame")
        legend_frame.pack(fill=tk.X, pady=5)

        for name, color in INJECTION_COLORS.items():
            ttk.Label(legend_frame, text=f"● {name}", foreground=color,
                     font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=10)

        ttk.Label(legend_frame, text="   |   ", foreground="gray").pack(side=tk.LEFT)
        ttk.Label(legend_frame, text="▲ Timeout", foreground="red",
                 font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=10)

    def _build_list_tab(self):
        """Tab amb llista d'injeccions."""
        columns = ("pos", "name", "type", "duration", "start", "end", "timeout", "zone", "status")

        tree_frame = ttk.Frame(self.tab_list)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=25)

        headers = {
            "pos": ("#", 40),
            "name": ("Nom", 100),
            "type": ("Tipus", 80),
            "duration": ("Duració", 70),
            "start": ("Inici", 70),
            "end": ("Final", 70),
            "timeout": ("Timeout", 70),
            "zone": ("Zona", 80),
            "status": ("Estat", 80),
        }

        for col, (text, width) in headers.items():
            self.tree.heading(col, text=text)
            self.tree.column(col, width=width, anchor="center")

        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Tags per colors
        self.tree.tag_configure("sample_crit", background="#FFEBEE")
        self.tree.tag_configure("sample_warn", background="#FFF8E1")
        self.tree.tag_configure("sample_ok", background="#E3F2FD")
        self.tree.tag_configure("khp", background="#F3E5F5")
        self.tree.tag_configure("control", background="#E0F2F1")

    def _build_suggestions_tab(self):
        """Tab amb suggeriments."""
        self.txt_suggestions = tk.Text(self.tab_suggestions, wrap=tk.WORD,
                                       font=("Segoe UI", 10), padx=15, pady=15)
        self.txt_suggestions.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.txt_suggestions.tag_configure("title", font=("Segoe UI", 12, "bold"))
        self.txt_suggestions.tag_configure("warning", foreground=SEVERITY_COLORS["WARNING"])
        self.txt_suggestions.tag_configure("critical", foreground=SEVERITY_COLORS["CRITICAL"])
        self.txt_suggestions.tag_configure("ok", foreground=SEVERITY_COLORS["OK"])

    def _build_compare_tab(self):
        """Tab de comparació COLUMN vs BP."""
        self.txt_compare = tk.Text(self.tab_compare, wrap=tk.WORD,
                                   font=("Consolas", 10), padx=15, pady=15)
        self.txt_compare.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.txt_compare.tag_configure("title", font=("Segoe UI", 14, "bold"))
        self.txt_compare.tag_configure("subtitle", font=("Segoe UI", 11, "bold"))
        self.txt_compare.tag_configure("column", foreground="#3498DB")
        self.txt_compare.tag_configure("bp", foreground="#9B59B6")
        self.txt_compare.tag_configure("ok", foreground=SEVERITY_COLORS["OK"])
        self.txt_compare.tag_configure("warning", foreground=SEVERITY_COLORS["WARNING"])
        self.txt_compare.tag_configure("critical", foreground=SEVERITY_COLORS["CRITICAL"])

    def _build_bottom_bar(self, parent):
        """Barra inferior."""
        bar = ttk.Frame(parent)
        bar.grid(row=2, column=0, sticky="ew", pady=(10, 0))

        # ESQUERRA: Botons principals
        left_frame = ttk.Frame(bar)
        left_frame.pack(side=tk.LEFT)

        ttk.Button(left_frame, text="CALCULAR",
                  command=self._calculate).pack(side=tk.LEFT, padx=5, ipadx=10, ipady=3)
        ttk.Button(left_frame, text="COLUMN + BP",
                  command=self._generate_both_modes).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(left_frame, text="Auto",
                       variable=self.var_auto_calc).pack(side=tk.LEFT, padx=10)

        # DRETA: Export
        ttk.Button(bar, text="Exportar CSV", command=self._export_csv).pack(side=tk.RIGHT, padx=5)
        ttk.Button(bar, text="Exportar PDF", command=self._export_pdf).pack(side=tk.RIGHT, padx=5)
        ttk.Button(bar, text="Guardar Config", command=self._save_config).pack(side=tk.RIGHT, padx=5)

    # =========================================================================
    # ACTUALITZACIONS
    # =========================================================================

    def _parse_positions(self, text: str) -> List[int]:
        """Parseja string de posicions (ex: '5,10,15') a llista d'enters."""
        if not text.strip():
            return []
        try:
            positions = []
            for part in text.replace(" ", "").split(","):
                if part:
                    positions.append(int(part))
            return positions
        except ValueError:
            return []

    def _on_mode_change(self):
        """Quan canvia el mode."""
        mode = self.var_mode.get()
        self.generator = SequenceGenerator(mode)
        self.var_duration.set(DURATIONS[mode]["sample"])
        self._update_mode_info()
        self._update_duration_buttons()
        self._on_param_change()

    def _update_mode_info(self):
        """Actualitza info del mode."""
        mode = self.var_mode.get()
        dur = DURATIONS[mode]["sample"]
        opt = DURATIONS[mode]["optimal"]
        self.lbl_mode_info.configure(text=f"Duració: {dur} min (òptim: {opt} min)")

    def _set_duration(self, value):
        """Estableix duració."""
        self.var_duration.set(value)
        self._on_param_change()

    def _on_param_change(self):
        """Quan canvia un paràmetre."""
        if self.var_auto_calc.get():
            self._schedule_calculate()

    def _schedule_calculate(self):
        """Programa càlcul amb debounce."""
        if hasattr(self, '_calc_after'):
            self.root.after_cancel(self._calc_after)
        self._calc_after = self.root.after(200, self._calculate)

    def _calculate(self):
        """Calcula la seqüència."""
        try:
            mode = self.var_mode.get()
            self.generator = SequenceGenerator(mode)

            # Parsejar posicions MQ i NaOH
            mq_positions = self._parse_positions(self.var_mq_positions.get())
            naoh_positions = self._parse_positions(self.var_naoh_positions.get())

            self.plan = self.generator.generate_sequence(
                n_samples=self.var_samples.get(),
                khp_start=self.var_khp_start.get(),
                khp_end=self.var_khp_end.get(),
                mq_positions=mq_positions,
                naoh_positions=naoh_positions,
                mq_at_start=self.var_mq_start.get(),
                naoh_at_end=self.var_naoh_end.get(),
                t0=self.var_t0.get(),
                sample_duration=self.var_duration.get(),
            )

            self._update_summary()
            self._update_timeline()
            self._update_list()
            self._update_suggestions()

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

    def _update_summary(self):
        """Actualitza resum."""
        if not self.plan:
            return

        self.lbl_total_inj.configure(text=str(self.plan.total_injections))
        self.lbl_total_time.configure(text=f"{self.plan.total_duration / 60:.1f}")
        self.lbl_critical.configure(text=str(self.plan.samples_in_critical))
        self.lbl_warning.configure(text=str(self.plan.samples_in_warning))

        # Color segons severitat
        if self.plan.samples_in_critical > 0:
            self.lbl_critical.configure(foreground=SEVERITY_COLORS["CRITICAL"])
        else:
            self.lbl_critical.configure(foreground=SEVERITY_COLORS["OK"])

    def _update_timeline(self):
        """Actualitza timeline."""
        if not HAS_MATPLOTLIB or not self.plan:
            return

        self.ax.clear()

        mode = self.var_mode.get()
        zones = get_zones(mode)
        max_y = 75 if mode == "COLUMN" else 20

        # Dibuixar zones de fons
        for zname, zdata in zones.items():
            if zdata["end"] > max_y:
                continue
            color = ZONE_COLORS.get(zname, "#CCCCCC")
            self.ax.axhspan(zdata["start"], zdata["end"], alpha=0.1, color=color)
            # Label zona
            self.ax.text(-0.5, (zdata["start"] + zdata["end"]) / 2, zname,
                        fontsize=8, ha="right", va="center", color=color)

        # Línia límit crític
        crit_start, crit_end = get_critical_zone(mode)
        self.ax.axhline(y=crit_end, color=SEVERITY_COLORS["CRITICAL"],
                       linestyle="--", linewidth=2, alpha=0.7)

        # Dibuixar injeccions
        for inj in self.plan.injections:
            x = inj.position
            color = INJECTION_COLORS.get(inj.inj_type.value, "#999999")

            # Barra vertical representant la injecció
            self.ax.bar(x, max_y, bottom=0, width=0.8, color=color, alpha=0.3, edgecolor=color)

            # Timeout (si hi ha)
            if inj.timeout_pos is not None:
                # Triangle vermell per timeout
                marker_color = SEVERITY_COLORS.get(inj.timeout_severity, "gray")
                self.ax.scatter(x, inj.timeout_pos, marker="^", s=100,
                              c=marker_color, edgecolors="white", linewidth=1, zorder=5)

                # Warning especial per mostres crítiques
                if inj.inj_type == InjectionType.SAMPLE and inj.in_critical:
                    self.ax.scatter(x, inj.timeout_pos, marker="o", s=200,
                                  facecolors="none", edgecolors="red", linewidth=2, zorder=4)

        # Configuració
        self.ax.set_xlabel("Injecció #", fontsize=10)
        self.ax.set_ylabel("Temps cromatograma (min)", fontsize=10)
        self.ax.set_title(f"Timeline Seqüència - {mode} ({self.plan.total_injections} injeccions, "
                         f"{self.plan.samples_in_critical} mostres crítiques)", fontsize=11)
        self.ax.set_xlim(0, self.plan.total_injections + 1)
        self.ax.set_ylim(-2, max_y + 2)
        self.ax.grid(True, alpha=0.3, axis="y")

        self.fig.tight_layout()
        self.canvas.draw()

    def _update_list(self):
        """Actualitza llista."""
        for item in self.tree.get_children():
            self.tree.delete(item)

        if not self.plan:
            return

        for inj in self.plan.injections:
            timeout_str = f"{inj.timeout_pos:.1f}" if inj.timeout_pos else "-"
            zone_str = inj.timeout_zone or "-"
            status = inj.timeout_severity

            # Determinar tag
            if inj.inj_type == InjectionType.SAMPLE:
                if inj.in_critical:
                    tag = "sample_crit"
                elif inj.timeout_severity == "WARNING":
                    tag = "sample_warn"
                else:
                    tag = "sample_ok"
            elif inj.inj_type == InjectionType.KHP:
                tag = "khp"
            else:
                tag = "control"

            self.tree.insert("", "end", values=(
                inj.position,
                inj.name,
                inj.inj_type.value,
                f"{inj.duration:.1f}",
                f"{inj.start_time:.1f}",
                f"{inj.end_time:.1f}",
                timeout_str,
                zone_str,
                status
            ), tags=(tag,))

    def _update_suggestions(self):
        """Actualitza suggeriments."""
        self.txt_suggestions.delete(1.0, tk.END)

        if not self.plan:
            return

        # Resum
        self.txt_suggestions.insert(tk.END, "RESUM DE LA SEQÜÈNCIA\n", "title")
        self.txt_suggestions.insert(tk.END, "=" * 50 + "\n\n")

        self.txt_suggestions.insert(tk.END, f"Total injeccions: {self.plan.total_injections}\n")
        self.txt_suggestions.insert(tk.END, f"Durada total: {self.plan.total_duration:.1f} min ")
        self.txt_suggestions.insert(tk.END, f"({self.plan.total_duration/60:.1f} h)\n\n")

        # Warnings
        if self.plan.samples_in_critical > 0:
            self.txt_suggestions.insert(tk.END, f"[!] {self.plan.samples_in_critical} MOSTRES EN ZONA CRITICA\n", "critical")

            # Llistar mostres afectades
            critical_samples = [i for i in self.plan.injections
                               if i.inj_type == InjectionType.SAMPLE and i.in_critical]
            for inj in critical_samples[:10]:
                self.txt_suggestions.insert(tk.END, f"   - {inj.name}: timeout a {inj.timeout_pos:.1f} min ({inj.timeout_zone})\n")
            if len(critical_samples) > 10:
                self.txt_suggestions.insert(tk.END, f"   ... i {len(critical_samples)-10} més\n")

            self.txt_suggestions.insert(tk.END, "\n")

        if self.plan.samples_in_warning > 0:
            self.txt_suggestions.insert(tk.END, f"[*] {self.plan.samples_in_warning} mostres en zona warning\n", "warning")
            self.txt_suggestions.insert(tk.END, "\n")

        if self.plan.samples_in_critical == 0 and self.plan.samples_in_warning == 0:
            self.txt_suggestions.insert(tk.END, "[OK] Cap mostra en zona critica o warning!\n", "ok")
            self.txt_suggestions.insert(tk.END, "\n")

        # Suggeriments
        suggestions = self.generator.suggest_control_positions(self.plan)

        if suggestions:
            self.txt_suggestions.insert(tk.END, "\nSUGGERIMENTS\n", "title")
            self.txt_suggestions.insert(tk.END, "=" * 50 + "\n\n")

            for i, sug in enumerate(suggestions, 1):
                self.txt_suggestions.insert(tk.END, f"{i}. Inserir {sug['type']} abans de #{sug['insert_before']} ({sug['injection_name']})\n")
                self.txt_suggestions.insert(tk.END, f"   Raó: {sug['reason']}\n")
                self.txt_suggestions.insert(tk.END, f"   Efecte: {sug['effect']}\n")
                self.txt_suggestions.insert(tk.END, f"   Afecta: {', '.join(sug['affected'][:5])}\n\n")

        # Recomanacions generals
        self.txt_suggestions.insert(tk.END, "\nRECOMANACIONS GENERALS\n", "title")
        self.txt_suggestions.insert(tk.END, "=" * 50 + "\n\n")

        duration = self.var_duration.get()
        if abs(duration - 77.2) > 0.1:
            self.txt_suggestions.insert(tk.END, "* Considera usar duracio 77.2 min (elimina deriva)\n")

        self.txt_suggestions.insert(tk.END, "* Espera 3-5 min despres del flush abans d'iniciar\n")

        mode = self.var_mode.get()
        _, crit_end = get_critical_zone(mode)
        self.txt_suggestions.insert(tk.END, f"* T0 recomanat: >{crit_end} min per evitar zona critica\n")

    def _generate_both_modes(self):
        """Genera plans per ambdos modes (COLUMN i BP) amb les mateixes mostres."""
        try:
            n_samples = self.var_samples.get()
            t0 = self.var_t0.get()

            # Parametres comuns
            khp_start = self.var_khp_start.get()
            khp_end = self.var_khp_end.get()
            mq_positions = self._parse_positions(self.var_mq_positions.get())
            naoh_positions = self._parse_positions(self.var_naoh_positions.get())
            mq_at_start = self.var_mq_start.get()
            naoh_at_end = self.var_naoh_end.get()

            # Generar COLUMN
            gen_col = SequenceGenerator("COLUMN")
            self.plan_column = gen_col.generate_sequence(
                n_samples=n_samples,
                khp_start=khp_start,
                khp_end=khp_end,
                mq_positions=mq_positions,
                naoh_positions=naoh_positions,
                mq_at_start=mq_at_start,
                naoh_at_end=naoh_at_end,
                t0=t0,
                sample_duration=DURATIONS["COLUMN"]["sample"],
            )

            # Generar BP
            gen_bp = SequenceGenerator("BP")
            self.plan_bp = gen_bp.generate_sequence(
                n_samples=n_samples,
                khp_start=khp_start,
                khp_end=khp_end,
                mq_positions=mq_positions,
                naoh_positions=naoh_positions,
                mq_at_start=mq_at_start,
                naoh_at_end=naoh_at_end,
                t0=t0,
                sample_duration=DURATIONS["BP"]["sample"],
            )

            # Actualitzar vista comparació
            self._update_compare()

            # Anar a la pestanya de comparació
            self.notebook.select(self.tab_compare)

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

    def _update_compare(self):
        """Actualitza la vista de comparació COLUMN vs BP."""
        self.txt_compare.delete(1.0, tk.END)

        if not self.plan_column or not self.plan_bp:
            self.txt_compare.insert(tk.END, "Prem 'Generar COLUMN + BP' per comparar ambdós modes.\n")
            return

        n_samples = self.plan_column.n_samples

        # Títol
        self.txt_compare.insert(tk.END, f"COMPARACIÓ COLUMN vs BP - {n_samples} mostres\n", "title")
        self.txt_compare.insert(tk.END, "=" * 60 + "\n\n")

        # Taula comparativa
        self.txt_compare.insert(tk.END, f"{'Paràmetre':<25} {'COLUMN':>15} {'BP':>15}\n")
        self.txt_compare.insert(tk.END, "-" * 60 + "\n")

        # Injeccions
        self.txt_compare.insert(tk.END, f"{'Total injeccions':<25} ")
        self.txt_compare.insert(tk.END, f"{self.plan_column.total_injections:>15}", "column")
        self.txt_compare.insert(tk.END, f"{self.plan_bp.total_injections:>15}\n", "bp")

        # Duració
        col_h = self.plan_column.total_duration / 60
        bp_h = self.plan_bp.total_duration / 60
        self.txt_compare.insert(tk.END, f"{'Duració total (h)':<25} ")
        self.txt_compare.insert(tk.END, f"{col_h:>15.1f}", "column")
        self.txt_compare.insert(tk.END, f"{bp_h:>15.1f}\n", "bp")

        # Mostres crítiques
        self.txt_compare.insert(tk.END, f"{'Mostres crítiques':<25} ")
        tag_col = "critical" if self.plan_column.samples_in_critical > 0 else "ok"
        tag_bp = "critical" if self.plan_bp.samples_in_critical > 0 else "ok"
        self.txt_compare.insert(tk.END, f"{self.plan_column.samples_in_critical:>15}", tag_col)
        self.txt_compare.insert(tk.END, f"{self.plan_bp.samples_in_critical:>15}\n", tag_bp)

        # Mostres warning
        self.txt_compare.insert(tk.END, f"{'Mostres warning':<25} ")
        tag_col = "warning" if self.plan_column.samples_in_warning > 0 else "ok"
        tag_bp = "warning" if self.plan_bp.samples_in_warning > 0 else "ok"
        self.txt_compare.insert(tk.END, f"{self.plan_column.samples_in_warning:>15}", tag_col)
        self.txt_compare.insert(tk.END, f"{self.plan_bp.samples_in_warning:>15}\n", tag_bp)

        # KHP crític
        self.txt_compare.insert(tk.END, f"{'KHP en zona crítica':<25} ")
        self.txt_compare.insert(tk.END, f"{self.plan_column.khp_in_critical:>15}")
        self.txt_compare.insert(tk.END, f"{self.plan_bp.khp_in_critical:>15}\n")

        self.txt_compare.insert(tk.END, "\n")

        # Recomanacions
        self.txt_compare.insert(tk.END, "RECOMANACIONS\n", "subtitle")
        self.txt_compare.insert(tk.END, "-" * 60 + "\n\n")

        # Ordre recomanat
        if bp_h < col_h:
            self.txt_compare.insert(tk.END, "* Recomanat executar BP primer (més curt)\n", "ok")
            self.txt_compare.insert(tk.END, f"  - BP: {bp_h:.1f}h, COLUMN: {col_h:.1f}h\n\n")
        else:
            self.txt_compare.insert(tk.END, "* BP i COLUMN tenen duracions similars\n")

        # Warnings específics
        if self.plan_column.samples_in_critical > 0:
            self.txt_compare.insert(tk.END, f"* COLUMN: {self.plan_column.samples_in_critical} mostres en zona critica - revisar T0\n", "warning")
        if self.plan_bp.samples_in_critical > 0:
            self.txt_compare.insert(tk.END, f"* BP: {self.plan_bp.samples_in_critical} mostres en zona critica - revisar T0\n", "warning")

        # Llista mostres
        self.txt_compare.insert(tk.END, "\n\nLLISTA DE MOSTRES\n", "subtitle")
        self.txt_compare.insert(tk.END, "-" * 60 + "\n\n")

        samples = [i for i in self.plan_column.injections if i.inj_type == InjectionType.SAMPLE]
        unique_samples = sorted(set(i.sample_num for i in samples if i.sample_num))

        for s in unique_samples[:20]:
            self.txt_compare.insert(tk.END, f"  M{s}: R1 + R2\n")
        if len(unique_samples) > 20:
            self.txt_compare.insert(tk.END, f"  ... i {len(unique_samples)-20} més\n")

    # =========================================================================
    # EXPORTACIÓ
    # =========================================================================

    def _export_csv(self):
        """Exporta a CSV."""
        if not self.plan:
            messagebox.showwarning("Avís", "Primer calcula una seqüència")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile=f"sequence_plan_{datetime.now():%Y%m%d_%H%M}.csv"
        )

        if not filepath:
            return

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("position,name,type,replica,sample_num,duration,start_time,end_time,timeout_pos,timeout_zone,severity,in_critical\n")
                for inj in self.plan.injections:
                    f.write(f"{inj.position},{inj.name},{inj.inj_type.value},{inj.replica or ''},{inj.sample_num or ''},")
                    f.write(f"{inj.duration:.2f},{inj.start_time:.2f},{inj.end_time:.2f},")
                    f.write(f"{inj.timeout_pos:.2f if inj.timeout_pos else ''},{inj.timeout_zone or ''},")
                    f.write(f"{inj.timeout_severity},{inj.in_critical}\n")

            messagebox.showinfo("Exportat", f"Guardat: {filepath}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _export_pdf(self):
        """Exporta a PDF."""
        if not HAS_MATPLOTLIB or not self.plan:
            messagebox.showwarning("Avís", "Cal matplotlib i dades")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF", "*.pdf")],
            initialfile=f"sequence_plan_{datetime.now():%Y%m%d_%H%M}.pdf"
        )

        if not filepath:
            return

        try:
            self.fig.savefig(filepath, dpi=150, bbox_inches='tight')
            messagebox.showinfo("Exportat", f"Guardat: {filepath}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _save_config(self):
        """Guarda configuració."""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
            initialfile="sequence_config.json"
        )

        if not filepath:
            return

        config = {
            "mode": self.var_mode.get(),
            "samples": self.var_samples.get(),
            "duration": self.var_duration.get(),
            "t0": self.var_t0.get(),
            "khp_start": self.var_khp_start.get(),
            "khp_end": self.var_khp_end.get(),
            "mq_positions": self.var_mq_positions.get(),
            "naoh_positions": self.var_naoh_positions.get(),
            "mq_start": self.var_mq_start.get(),
            "naoh_end": self.var_naoh_end.get(),
        }

        try:
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=2)
            messagebox.showinfo("Guardat", f"Configuració guardada: {filepath}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def run(self):
        """Executa mainloop."""
        if self.standalone:
            self.root.mainloop()


# =============================================================================
# API PÚBLICA
# =============================================================================

def open_planner(parent=None):
    """Obre el planificador."""
    gui = SequencePlannerGUI(parent=parent)
    if parent is None:
        gui.run()
    return gui


if __name__ == "__main__":
    app = SequencePlannerGUI()
    app.run()
