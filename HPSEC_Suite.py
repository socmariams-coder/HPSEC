#!/usr/bin/env python3
"""
HPSEC Suite v2.0
================
Wizard multi-fase per processament de dades HPSEC.

Fases:
1. IMPORT    - Seleccionar SEQ, detectar mostres
2. CALIBRATE - Analitzar KHP, calcular factors
3. PROCESS   - Detectar anomalies, processar mostres
4. REVIEW    - Seleccionar repliques, validar
5. EXPORT    - Generar Excel/PDF finals

Cada fase permet aturar, revisar i modificar abans de continuar.
"""

__version__ = "2.0.0"

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from datetime import datetime
import threading

import traceback
import numpy as np
import pandas as pd

# Pipeline modules
from hpsec_import import (
    import_sequence, generate_import_manifest, save_import_manifest,
    load_manifest, import_from_manifest
)
from hpsec_calibrate import calibrate_from_import
from hpsec_process import process_sequence, write_consolidated_excel
from hpsec_config import get_config

# Optional: matplotlib per grafics
try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# =============================================================================
# CONSTANTS
# =============================================================================

PHASES = ["IMPORT", "CALIBRATE", "PROCESS", "REVIEW", "EXPORT"]
PHASE_TITLES = {
    "IMPORT": "1. Importar Sequencia",
    "CALIBRATE": "2. Calibracio KHP",
    "PROCESS": "3. Processar Mostres",
    "REVIEW": "4. Revisar Repliques",
    "EXPORT": "5. Exportar Resultats",
}

WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800


# =============================================================================
# MAIN APPLICATION
# =============================================================================

class HPSECSuite(tk.Tk):
    """Aplicacio principal wizard multi-fase."""

    def __init__(self):
        super().__init__()

        self.title(f"HPSEC Suite v{__version__}")
        self.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")

        # Estat
        self.current_phase = 0
        self.seq_path = None
        self.imported_data = None
        self.calibration_data = None
        self.processed_data = None
        self.review_data = None

        # Config - obtenir el dict, no el ConfigManager
        config_manager = get_config()
        self.config_data = config_manager.config if hasattr(config_manager, 'config') else config_manager

        # Setup UI
        self._setup_ui()
        self._show_phase(0)

    def _setup_ui(self):
        """Configura la UI principal."""
        # Frame superior: navegacio fases
        self.nav_frame = ttk.Frame(self)
        self.nav_frame.pack(fill=tk.X, padx=10, pady=5)

        self.phase_buttons = []
        for i, phase in enumerate(PHASES):
            btn = ttk.Button(
                self.nav_frame,
                text=f"{i+1}. {phase}",
                command=lambda idx=i: self._show_phase(idx),
                state=tk.DISABLED if i > 0 else tk.NORMAL
            )
            btn.pack(side=tk.LEFT, padx=2)
            self.phase_buttons.append(btn)

        # Separador
        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=10, pady=5)

        # Frame central: contingut de cada fase
        self.content_frame = ttk.Frame(self)
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Frame inferior: botons navegacio
        self.button_frame = ttk.Frame(self)
        self.button_frame.pack(fill=tk.X, padx=10, pady=10)

        self.btn_back = ttk.Button(self.button_frame, text="< Enrere", command=self._go_back)
        self.btn_back.pack(side=tk.LEFT)

        self.btn_next = ttk.Button(self.button_frame, text="Seguent >", command=self._go_next)
        self.btn_next.pack(side=tk.RIGHT)

        # Barra de progres
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            self.button_frame,
            variable=self.progress_var,
            maximum=100
        )
        self.progress_bar.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=20)

        self.status_var = tk.StringVar(value="Llest")
        self.status_label = ttk.Label(self.button_frame, textvariable=self.status_var)
        self.status_label.pack(side=tk.RIGHT, padx=10)

    def _clear_content(self):
        """Neteja el frame de contingut."""
        for widget in self.content_frame.winfo_children():
            widget.destroy()

    def _show_phase(self, phase_idx):
        """Mostra la fase indicada."""
        self.current_phase = phase_idx
        self._clear_content()

        # Actualitzar botons navegacio
        self.btn_back.config(state=tk.NORMAL if phase_idx > 0 else tk.DISABLED)
        self.btn_next.config(text="Finalitzar" if phase_idx == len(PHASES)-1 else "Seguent >")

        # Highlight fase actual
        for i, btn in enumerate(self.phase_buttons):
            if i == phase_idx:
                btn.config(style="Accent.TButton")
            else:
                btn.config(style="TButton")

        # Mostrar contingut de la fase
        phase = PHASES[phase_idx]

        # Titol
        title = ttk.Label(
            self.content_frame,
            text=PHASE_TITLES[phase],
            font=("Helvetica", 16, "bold")
        )
        title.pack(pady=10)

        # Contingut especific
        if phase == "IMPORT":
            self._show_import_phase()
        elif phase == "CALIBRATE":
            self._show_calibrate_phase()
        elif phase == "PROCESS":
            self._show_process_phase()
        elif phase == "REVIEW":
            self._show_review_phase()
        elif phase == "EXPORT":
            self._show_export_phase()

    def _go_back(self):
        """Torna a la fase anterior."""
        if self.current_phase > 0:
            self._show_phase(self.current_phase - 1)

    def _go_next(self):
        """Avanca a la seguent fase."""
        # Validar fase actual
        if not self._validate_current_phase():
            return

        # Executar fase si cal
        if not self._execute_current_phase():
            return

        if self.current_phase < len(PHASES) - 1:
            # Habilitar seguent fase
            self.phase_buttons[self.current_phase + 1].config(state=tk.NORMAL)
            self._show_phase(self.current_phase + 1)
        else:
            # Finalitzar
            messagebox.showinfo("Completat", "Processament finalitzat!")

    def _validate_current_phase(self):
        """Valida que la fase actual esta completa."""
        phase = PHASES[self.current_phase]

        if phase == "IMPORT":
            if not self.seq_path:
                messagebox.showwarning("Avis", "Selecciona una carpeta SEQ")
                return False

        return True

    def _execute_current_phase(self):
        """Executa la fase actual si cal processar."""
        phase = PHASES[self.current_phase]

        try:
            if phase == "IMPORT" and self.imported_data is None:
                self._run_import()
            elif phase == "CALIBRATE" and self.calibration_data is None:
                self._run_calibrate()
            elif phase == "PROCESS" and self.processed_data is None:
                self._run_process()
            elif phase == "EXPORT":
                self._run_export()
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Error a fase {phase}:\n{str(e)}")
            return False

    # =========================================================================
    # FASE 1: IMPORT
    # =========================================================================

    def _show_import_phase(self):
        """Mostra UI per importar sequencia."""
        frame = ttk.Frame(self.content_frame)
        frame.pack(fill=tk.BOTH, expand=True)

        # Seleccio carpeta
        sel_frame = ttk.Frame(frame)
        sel_frame.pack(fill=tk.X, pady=10)

        ttk.Label(sel_frame, text="Carpeta SEQ:").pack(side=tk.LEFT)

        self.seq_path_var = tk.StringVar(value=self.seq_path or "")
        entry = ttk.Entry(sel_frame, textvariable=self.seq_path_var, width=60)
        entry.pack(side=tk.LEFT, padx=5)

        ttk.Button(sel_frame, text="Buscar...", command=self._browse_seq).pack(side=tk.LEFT)

        # Frame per info manifest/importacio
        self.import_info_frame = ttk.Frame(frame)
        self.import_info_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Mostrar info segons estat
        if self.imported_data:
            self._show_import_summary(self.import_info_frame)
        elif self.seq_path:
            self._check_existing_manifest()

    def _browse_seq(self):
        """Obre dialeg per seleccionar carpeta SEQ."""
        path = filedialog.askdirectory(title="Selecciona carpeta SEQ")
        if path:
            self.seq_path = path
            self.seq_path_var.set(path)
            # Reset dades posteriors
            self.imported_data = None
            self.calibration_data = None
            self.processed_data = None
            self.existing_manifest = None
            # Comprovar manifest i refrescar vista
            self._show_phase(0)

    def _check_existing_manifest(self):
        """Comprova si existeix manifest i mostra opcions."""
        # Netejar frame
        for widget in self.import_info_frame.winfo_children():
            widget.destroy()

        self.existing_manifest = load_manifest(self.seq_path)

        if self.existing_manifest:
            self._show_manifest_options()
        else:
            ttk.Label(
                self.import_info_frame,
                text="Prem 'Seguent' per importar la sequencia",
                font=("Helvetica", 10)
            ).pack(pady=20)

    def _show_manifest_options(self):
        """Mostra opcions quan existeix un manifest."""
        manifest = self.existing_manifest

        # Info del manifest existent
        info_frame = ttk.LabelFrame(self.import_info_frame, text="Importacio Anterior Detectada")
        info_frame.pack(fill=tk.X, pady=10, padx=10)

        seq = manifest.get("sequence", {})
        summary = manifest.get("summary", {})
        generated = manifest.get("generated_at", "")[:19].replace("T", " ")

        info_text = f"""
Sequencia: {seq.get('name', 'N/A')}
Importat el: {generated}
Mode: {seq.get('method', 'N/A')} / {seq.get('data_mode', 'N/A')}

Mostres: {summary.get('total_samples', 0)} | KHP: {summary.get('total_khp', 0)} | Controls: {summary.get('total_controls', 0)}
Repliques: {summary.get('total_replicas', 0)} (Direct: {summary.get('replicas_with_direct', 0)}, UIB: {summary.get('replicas_with_uib', 0)}, DAD: {summary.get('replicas_with_dad', 0)})
        """

        ttk.Label(info_frame, text=info_text.strip(), justify=tk.LEFT).pack(padx=10, pady=10)

        # Opcions
        opt_frame = ttk.Frame(self.import_info_frame)
        opt_frame.pack(fill=tk.X, pady=10, padx=10)

        self.import_mode_var = tk.StringVar(value="manifest")

        ttk.Radiobutton(
            opt_frame,
            text="Usar importacio anterior (rapid - llegeix dades segons manifest)",
            variable=self.import_mode_var,
            value="manifest"
        ).pack(anchor=tk.W, pady=2)

        ttk.Radiobutton(
            opt_frame,
            text="Reimportar completament (detectar i aparellar de nou)",
            variable=self.import_mode_var,
            value="full"
        ).pack(anchor=tk.W, pady=2)

        ttk.Label(
            self.import_info_frame,
            text="Prem 'Seguent' per continuar",
            font=("Helvetica", 9, "italic")
        ).pack(pady=10)

    def _run_import(self):
        """Executa importacio (normal o des de manifest)."""
        self.status_var.set("Important...")
        self.progress_var.set(0)
        self.update()

        def progress_cb(pct, msg):
            self.progress_var.set(pct)
            self.status_var.set(msg)
            self.update()

        # Decidir mode d'importacio
        use_manifest = (
            hasattr(self, 'existing_manifest') and
            self.existing_manifest and
            hasattr(self, 'import_mode_var') and
            self.import_mode_var.get() == "manifest"
        )

        if use_manifest:
            self.status_var.set("Important des de manifest...")
            self.imported_data = import_from_manifest(
                self.seq_path,
                manifest=self.existing_manifest,
                progress_callback=progress_cb
            )
        else:
            self.status_var.set("Important (deteccio completa)...")
            self.imported_data = import_sequence(self.seq_path, progress_callback=progress_cb)

        if not self.imported_data.get("success"):
            error_msg = self.imported_data.get("error")
            if not error_msg:
                errors_list = self.imported_data.get("errors", [])
                error_msg = errors_list[0] if errors_list else "No s'han pogut importar dades"
            raise Exception(error_msg)

        self.progress_var.set(100)
        mode_str = "des de manifest" if use_manifest else "completa"
        self.status_var.set(f"Importacio {mode_str} completada")

    def _show_import_summary(self, parent):
        """Mostra resum complet de la importacio amb manifest."""
        # Generar manifest
        self.import_manifest = generate_import_manifest(self.imported_data)
        manifest = self.import_manifest

        # Frame principal amb scroll
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # === SECCIÓ 1: Info general ===
        info_frame = ttk.LabelFrame(main_frame, text="Informacio General")
        info_frame.pack(fill=tk.X, pady=5)

        seq = manifest["sequence"]
        mf = manifest["master_file"]
        summary = manifest["summary"]

        info_grid = ttk.Frame(info_frame)
        info_grid.pack(fill=tk.X, padx=10, pady=5)

        labels = [
            ("Sequencia:", seq["name"]),
            ("Data:", seq["date"][:10] if seq["date"] else "N/A"),
            ("Metode:", seq["method"]),
            ("Mode dades:", seq["data_mode"]),
            ("MasterFile:", mf["filename"]),
            ("Format:", mf["format"]),
        ]
        for i, (label, value) in enumerate(labels):
            ttk.Label(info_grid, text=label, font=("Helvetica", 9, "bold")).grid(row=i//3, column=(i%3)*2, sticky="e", padx=2)
            ttk.Label(info_grid, text=str(value)).grid(row=i//3, column=(i%3)*2+1, sticky="w", padx=(0,15))

        # === SECCIÓ 2: Resum estadístic ===
        stats_frame = ttk.LabelFrame(main_frame, text="Resum Dades Detectades")
        stats_frame.pack(fill=tk.X, pady=5)

        stats_grid = ttk.Frame(stats_frame)
        stats_grid.pack(fill=tk.X, padx=10, pady=5)

        stats_labels = [
            ("Mostres:", summary["total_samples"]),
            ("KHP:", summary["total_khp"]),
            ("Controls:", summary["total_controls"]),
            ("Repliques:", summary["total_replicas"]),
            ("Amb DOC Direct:", summary["replicas_with_direct"]),
            ("Amb DOC UIB:", summary["replicas_with_uib"]),
            ("Amb DAD:", summary["replicas_with_dad"]),
        ]
        for i, (label, value) in enumerate(stats_labels):
            col = i % 4
            row = i // 4
            ttk.Label(stats_grid, text=label).grid(row=row, column=col*2, sticky="e", padx=2)
            ttk.Label(stats_grid, text=str(value), font=("Helvetica", 9, "bold")).grid(row=row, column=col*2+1, sticky="w", padx=(0,15))

        # === SECCIÓ 3: Taula de mostres ===
        table_frame = ttk.LabelFrame(main_frame, text="Detall Mostres")
        table_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        columns = ("Mostra", "Tipus", "Rep", "Direct", "UIB", "DAD")
        tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=10)

        tree.heading("Mostra", text="Mostra")
        tree.heading("Tipus", text="Tipus")
        tree.heading("Rep", text="Rep")
        tree.heading("Direct", text="DOC Direct")
        tree.heading("UIB", text="DOC UIB")
        tree.heading("DAD", text="DAD")

        tree.column("Mostra", width=120)
        tree.column("Tipus", width=70)
        tree.column("Rep", width=40)
        tree.column("Direct", width=150)
        tree.column("UIB", width=150)
        tree.column("DAD", width=120)

        for sample in manifest["samples"]:
            for rep in sample["replicas"]:
                # Direct info
                d = rep.get("direct", {})
                direct_str = f"{d['n_points']} pts ({d['t_min']:.0f}-{d['t_max']:.0f}min)" if d else "-"

                # UIB info
                u = rep.get("uib", {})
                uib_str = f"{u['n_points']} pts ({u['t_min']:.0f}-{u['t_max']:.0f}min)" if u else "-"

                # DAD info
                dad = rep.get("dad", {})
                dad_str = f"{dad['n_points']} pts x {dad['n_wavelengths']} wl" if dad else "-"

                tree.insert("", tk.END, values=(
                    sample["name"],
                    sample["type"],
                    rep["replica"],
                    direct_str,
                    uib_str,
                    dad_str
                ))

        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # === SECCIÓ 4: Avisos ===
        if manifest["warnings"] or manifest["orphan_files"]["uib"] or manifest["orphan_files"]["dad"]:
            warn_frame = ttk.LabelFrame(main_frame, text="Avisos")
            warn_frame.pack(fill=tk.X, pady=5)

            warn_text = ""
            if manifest["warnings"]:
                warn_text += "\n".join(manifest["warnings"]) + "\n"
            if manifest["orphan_files"]["uib"]:
                warn_text += f"Fitxers UIB orfes: {', '.join(manifest['orphan_files']['uib'])}\n"
            if manifest["orphan_files"]["dad"]:
                warn_text += f"Fitxers DAD orfes: {', '.join(manifest['orphan_files']['dad'])}\n"

            ttk.Label(warn_frame, text=warn_text.strip(), foreground="orange").pack(padx=10, pady=5)

        # === Botó guardar manifest ===
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        ttk.Button(btn_frame, text="Guardar Manifest JSON", command=self._save_manifest).pack(side=tk.RIGHT)

    def _save_manifest(self):
        """Guarda el manifest JSON."""
        if not hasattr(self, 'import_manifest') or not self.import_manifest:
            messagebox.showwarning("Avis", "No hi ha manifest per guardar")
            return

        output_path = save_import_manifest(self.imported_data)
        messagebox.showinfo("Guardat", f"Manifest guardat a:\n{output_path}")

    # =========================================================================
    # FASE 2: CALIBRATE
    # =========================================================================

    def _show_calibrate_phase(self):
        """Mostra UI per calibracio."""
        frame = ttk.Frame(self.content_frame)
        frame.pack(fill=tk.BOTH, expand=True)

        if self.calibration_data:
            self._show_calibration_summary(frame)
        else:
            ttk.Label(frame, text="Prem 'Seguent' per executar calibracio").pack(pady=20)

    def _run_calibrate(self):
        """Executa calibracio."""
        self.status_var.set("Calibrant...")
        self.progress_var.set(0)
        self.update()

        def progress_cb(pct, msg):
            self.progress_var.set(pct)
            self.status_var.set(msg)
            self.update()

        self.calibration_data = calibrate_from_import(
            self.imported_data,
            config=self.config_data,
            progress_callback=progress_cb
        )

        self.progress_var.set(100)

        if self.calibration_data.get("success"):
            self.status_var.set("Calibracio completada")
        else:
            self.status_var.set("Calibracio: usant defaults")
            # No fallar - continuar amb defaults

    def _show_calibration_summary(self, parent):
        """Mostra resum de calibracio."""
        info_frame = ttk.LabelFrame(parent, text="Resum Calibracio")
        info_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        cal = self.calibration_data

        info_text = f"""
Mode: {cal.get('mode', 'N/A')}
Font KHP: {cal.get('khp_source', 'N/A')}

Factor Direct: {cal.get('factor_direct', 0):.6f}
Factor UIB: {cal.get('factor_uib', 0):.6f}
Shift UIB: {cal.get('shift_uib', 0):.3f} min

KHP Concentracio: {cal.get('khp_conc', 0):.1f} ppm
        """

        if cal.get("errors"):
            info_text += f"\nAvisos: {', '.join(cal['errors'])}"

        text = tk.Text(info_frame, height=12, width=60)
        text.insert(tk.END, info_text)
        text.config(state=tk.DISABLED)
        text.pack(padx=10, pady=10)

    # =========================================================================
    # FASE 3: PROCESS
    # =========================================================================

    def _show_process_phase(self):
        """Mostra UI per processament."""
        frame = ttk.Frame(self.content_frame)
        frame.pack(fill=tk.BOTH, expand=True)

        if self.processed_data:
            self._show_process_summary(frame)
        else:
            ttk.Label(frame, text="Prem 'Seguent' per processar mostres").pack(pady=20)

    def _run_process(self):
        """Executa processament."""
        self.status_var.set("Processant...")
        self.progress_var.set(0)
        self.update()

        def progress_cb(msg, pct):
            self.progress_var.set(pct)
            self.status_var.set(msg)
            self.update()

        # Nota: process_sequence usa DEFAULT_PROCESS_CONFIG intern
        # No passar config extern perque te estructura diferent (niuada)
        self.processed_data = process_sequence(
            self.imported_data,
            self.calibration_data,
            config=None,
            progress_callback=progress_cb
        )

        self.progress_var.set(100)
        self.status_var.set("Processament completat")

    def _show_process_summary(self, parent):
        """Mostra resum de processament."""
        info_frame = ttk.LabelFrame(parent, text="Resum Processament")
        info_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        proc = self.processed_data
        samples = proc.get("samples", [])

        n_valid = sum(1 for s in samples if s.get("processed") and not s.get("anomalies"))
        n_anomalies = sum(1 for s in samples if s.get("anomalies"))

        info_text = f"""
Mostres processades: {len(samples)}
Valides: {n_valid}
Amb anomalies: {n_anomalies}
        """

        if proc.get("errors"):
            info_text += f"\nErrors: {len(proc['errors'])}"

        text = tk.Text(info_frame, height=8, width=60)
        text.insert(tk.END, info_text)
        text.config(state=tk.DISABLED)
        text.pack(padx=10, pady=10)

        # Taula mostres
        self._show_samples_table(parent, samples)

    def _show_samples_table(self, parent, samples):
        """Mostra taula de mostres."""
        table_frame = ttk.LabelFrame(parent, text="Mostres")
        table_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Treeview
        columns = ("Mostra", "Rep", "Status", "Anomalies", "R2", "Area")
        tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=10)

        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=100)

        for s in samples:
            status = "OK" if s.get("processed") and not s.get("anomalies") else "CHECK"
            anomalies = ", ".join(s.get("anomalies", [])) or "-"
            r2 = f"{s.get('peak_info', {}).get('r2', 0):.4f}"
            area = f"{s.get('peak_info', {}).get('area', 0):.1f}"

            tree.insert("", tk.END, values=(
                s.get("name", "?"),
                s.get("replica", "?"),
                status,
                anomalies,
                r2,
                area
            ))

        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)

        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # =========================================================================
    # FASE 4: REVIEW
    # =========================================================================

    def _show_review_phase(self):
        """Mostra UI per revisio i seleccio de repliques."""
        frame = ttk.Frame(self.content_frame)
        frame.pack(fill=tk.BOTH, expand=True)

        if not self.processed_data:
            ttk.Label(frame, text="Cal processar primer les mostres").pack(pady=20)
            return

        # Inicialitzar review_data si no existeix
        if self.review_data is None:
            self._init_review_data()

        # Instruccions
        ttk.Label(
            frame,
            text="Selecciona la replica preferida per cada mostra (o AUTO per seleccio automatica)",
            font=("Helvetica", 10)
        ).pack(pady=5)

        # Frame amb scroll per la llista de mostres
        canvas = tk.Canvas(frame)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Headers
        header_frame = ttk.Frame(scrollable_frame)
        header_frame.pack(fill=tk.X, pady=5)

        headers = ["Mostra", "Repliques", "Seleccio", "R² millor", "Anomalies"]
        widths = [150, 200, 120, 100, 200]
        for h, w in zip(headers, widths):
            ttk.Label(header_frame, text=h, font=("Helvetica", 9, "bold"), width=w//8).pack(side=tk.LEFT, padx=5)

        ttk.Separator(scrollable_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=2)

        # Agrupar mostres per nom
        samples_by_name = {}
        for s in self.processed_data.get("samples", []):
            name = s.get("name", "?")
            if name not in samples_by_name:
                samples_by_name[name] = []
            samples_by_name[name].append(s)

        # Mostrar cada mostra amb les seves repliques
        self.replica_vars = {}
        for name, replicas in sorted(samples_by_name.items()):
            row_frame = ttk.Frame(scrollable_frame)
            row_frame.pack(fill=tk.X, pady=2)

            # Nom mostra
            ttk.Label(row_frame, text=name, width=18).pack(side=tk.LEFT, padx=5)

            # Repliques disponibles
            rep_info = []
            for r in replicas:
                rep_num = r.get("replica", "?")
                r2 = r.get("peak_info", {}).get("r2", 0)
                anom = r.get("anomalies", [])
                status = "OK" if not anom else f"({len(anom)} anom)"
                rep_info.append(f"{rep_num}: R²={r2:.4f} {status}")

            ttk.Label(row_frame, text=", ".join([f"Rep {r.get('replica', '?')}" for r in replicas]), width=25).pack(side=tk.LEFT, padx=5)

            # Selector de replica
            options = ["AUTO"] + [str(r.get("replica", i+1)) for i, r in enumerate(replicas)]
            var = tk.StringVar(value=self.review_data.get(name, {}).get("selected", "AUTO"))
            self.replica_vars[name] = var

            combo = ttk.Combobox(row_frame, textvariable=var, values=options, width=10, state="readonly")
            combo.pack(side=tk.LEFT, padx=5)
            combo.bind("<<ComboboxSelected>>", lambda e, n=name: self._on_replica_selected(n))

            # Millor R²
            best_r2 = max(r.get("peak_info", {}).get("r2", 0) for r in replicas)
            r2_label = ttk.Label(row_frame, text=f"{best_r2:.4f}", width=12)
            r2_label.pack(side=tk.LEFT, padx=5)

            # Anomalies de la replica seleccionada
            selected_rep = self._get_selected_replica(name, replicas)
            anomalies = selected_rep.get("anomalies", []) if selected_rep else []
            anom_text = ", ".join(anomalies) if anomalies else "-"
            ttk.Label(row_frame, text=anom_text[:30], width=25).pack(side=tk.LEFT, padx=5)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Resum
        summary_frame = ttk.LabelFrame(frame, text="Resum Seleccio")
        summary_frame.pack(fill=tk.X, pady=10)

        n_auto = sum(1 for v in self.replica_vars.values() if v.get() == "AUTO")
        n_manual = len(self.replica_vars) - n_auto
        ttk.Label(summary_frame, text=f"Total mostres: {len(samples_by_name)} | AUTO: {n_auto} | Manual: {n_manual}").pack(pady=5)

    def _init_review_data(self):
        """Inicialitza dades de revisio amb seleccio automatica."""
        self.review_data = {}

        samples_by_name = {}
        for s in self.processed_data.get("samples", []):
            name = s.get("name", "?")
            if name not in samples_by_name:
                samples_by_name[name] = []
            samples_by_name[name].append(s)

        for name, replicas in samples_by_name.items():
            # Seleccio automatica: replica amb millor R² i sense anomalies
            best = None
            best_score = -1
            for r in replicas:
                r2 = r.get("peak_info", {}).get("r2", 0)
                has_anomalies = bool(r.get("anomalies"))
                # Score: R² penalitzat si te anomalies
                score = r2 if not has_anomalies else r2 * 0.5
                if score > best_score:
                    best_score = score
                    best = r

            self.review_data[name] = {
                "selected": "AUTO",
                "auto_replica": str(best.get("replica", 1)) if best else "1",
                "replicas": replicas
            }

    def _on_replica_selected(self, sample_name):
        """Handler quan es selecciona una replica."""
        if sample_name in self.replica_vars:
            selected = self.replica_vars[sample_name].get()
            if sample_name not in self.review_data:
                self.review_data[sample_name] = {}
            self.review_data[sample_name]["selected"] = selected

    def _get_selected_replica(self, sample_name, replicas):
        """Retorna la replica seleccionada (o la auto)."""
        if sample_name not in self.review_data:
            return replicas[0] if replicas else None

        selection = self.review_data[sample_name].get("selected", "AUTO")
        if selection == "AUTO":
            auto_rep = self.review_data[sample_name].get("auto_replica", "1")
            selection = auto_rep

        for r in replicas:
            if str(r.get("replica")) == str(selection):
                return r
        return replicas[0] if replicas else None

    # =========================================================================
    # FASE 5: EXPORT
    # =========================================================================

    def _show_export_phase(self):
        """Mostra UI per exportacio."""
        frame = ttk.Frame(self.content_frame)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="Opcions d'exportacio:").pack(pady=10)

        # Opcions
        self.export_excel_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="Generar Excels consolidats",
                       variable=self.export_excel_var).pack(anchor=tk.W, padx=20)

        self.export_pdf_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="Generar informe PDF (TODO)",
                       variable=self.export_pdf_var, state=tk.DISABLED).pack(anchor=tk.W, padx=20)

        self.export_only_selected_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="Només repliques seleccionades",
                       variable=self.export_only_selected_var).pack(anchor=tk.W, padx=20)

        # Carpeta sortida
        out_frame = ttk.Frame(frame)
        out_frame.pack(fill=tk.X, pady=10, padx=20)

        ttk.Label(out_frame, text="Carpeta sortida:").pack(side=tk.LEFT)

        default_out = str(Path(self.seq_path) / "Dades_Consolidades") if self.seq_path else ""
        self.output_path_var = tk.StringVar(value=default_out)
        ttk.Entry(out_frame, textvariable=self.output_path_var, width=50).pack(side=tk.LEFT, padx=5)

        ttk.Button(out_frame, text="Canviar...", command=self._browse_output).pack(side=tk.LEFT)

        # Resum del que s'exportara
        if self.review_data:
            summary_frame = ttk.LabelFrame(frame, text="Resum Exportacio")
            summary_frame.pack(fill=tk.X, pady=10, padx=20)

            n_samples = len(self.review_data)
            ttk.Label(summary_frame, text=f"Mostres a exportar: {n_samples}").pack(pady=5)

    def _browse_output(self):
        """Selecciona carpeta sortida."""
        path = filedialog.askdirectory(title="Selecciona carpeta sortida")
        if path:
            self.output_path_var.set(path)

    def _run_export(self):
        """Executa exportacio de fitxers Excel consolidats."""
        if not self.export_excel_var.get():
            self.status_var.set("Cap opcio d'exportacio seleccionada")
            return

        self.status_var.set("Exportant...")
        self.progress_var.set(0)
        self.update()

        output_path = Path(self.output_path_var.get())
        output_path.mkdir(parents=True, exist_ok=True)

        # Obtenir dades necessaries
        seq_name = self.processed_data.get("seq_name", Path(self.seq_path).name)
        method = self.processed_data.get("method", "COLUMN")
        master_info = self.imported_data.get("master_info", {})
        date_master = master_info.get("date", datetime.now().strftime("%Y-%m-%d"))

        shift_uib = self.calibration_data.get("shift_uib", 0) if self.calibration_data else 0
        shift_direct = self.calibration_data.get("shift_direct", 0) if self.calibration_data else 0

        # Agrupar mostres per nom
        samples_by_name = {}
        all_samples = (
            self.processed_data.get("samples", []) +
            self.processed_data.get("khp_samples", []) +
            self.processed_data.get("control_samples", [])
        )
        for s in all_samples:
            name = s.get("name", "?")
            if name not in samples_by_name:
                samples_by_name[name] = []
            samples_by_name[name].append(s)

        excels_written = 0
        errors = []
        total = len(samples_by_name)

        for i, (name, replicas) in enumerate(sorted(samples_by_name.items())):
            self.progress_var.set(int((i / total) * 100))
            self.status_var.set(f"Exportant {name}...")
            self.update()

            # Determinar quines repliques exportar
            if self.export_only_selected_var.get() and self.review_data:
                # Només la seleccionada
                selected = self._get_selected_replica(name, replicas)
                replicas_to_export = [selected] if selected else replicas[:1]
            else:
                # Totes
                replicas_to_export = replicas

            for sample in replicas_to_export:
                if not sample or not sample.get("processed"):
                    continue

                try:
                    n_written = self._write_sample_excel(
                        output_path, sample, seq_name, method,
                        date_master, master_info, shift_uib, shift_direct
                    )
                    excels_written += n_written
                except Exception as e:
                    errors.append(f"{name}: {str(e)}")

        self.progress_var.set(100)
        self.status_var.set(f"Exportacio completada: {excels_written} fitxers")

        msg = f"Fitxers exportats a:\n{output_path}\n\nExcels generats: {excels_written}"
        if errors:
            msg += f"\n\nErrors ({len(errors)}):\n" + "\n".join(errors[:5])
        messagebox.showinfo("Completat", msg)

    def _write_sample_excel(self, output_path, sample, seq_name, method, date_master, master_info, shift_uib, shift_direct):
        """Escriu Excel consolidat per una mostra."""
        mostra = sample.get("name", "UNKNOWN")
        rep = sample.get("replica", "1")

        # Convertir replica a lletra
        try:
            rep_int = int(rep)
            rep_letter = chr(ord('A') + rep_int - 1)
        except (ValueError, TypeError):
            rep_letter = str(rep)

        out_name = f"{seq_name}_{mostra}_{rep_letter}.xlsx"
        out_path = output_path / out_name

        # Extreure dades del sample processat
        peak_info = sample.get("peak_info", {})
        timeout_info = sample.get("timeout_info", {})
        snr_info = sample.get("snr_info", {})
        df_dad = sample.get("df_dad")

        # Dades DOC
        data_mode = sample.get("data_mode", "DIRECT")
        t_doc = sample.get("t_doc")
        y_doc_raw = sample.get("y_doc_raw")
        y_doc_net = sample.get("y_doc_net")
        baseline = sample.get("baseline", 0)
        fitxer_doc = sample.get("fitxer_doc", "")

        # Dades UIB (si DUAL)
        y_doc_uib = sample.get("y_doc_uib")
        y_doc_uib_raw = sample.get("y_doc_uib_raw")
        baseline_uib = sample.get("baseline_uib")
        fitxer_doc_uib = sample.get("fitxer_doc_uib", "")

        # Determinar doc_mode per l'Excel
        if data_mode == "DUAL":
            doc_mode = "DUAL"
        elif data_mode == "DIRECT":
            doc_mode = "DIRECT"
        elif data_mode == "UIB":
            doc_mode = "UIB"
        else:
            doc_mode = "DAD_ONLY" if df_dad is not None else "UNKNOWN"

        # Fitxer DAD
        fitxer_dad = sample.get("fitxer_dad", "")

        # Escriure Excel
        write_consolidated_excel(
            out_path=str(out_path),
            mostra=mostra,
            rep=rep_letter,
            seq_out=seq_name,
            date_master=date_master,
            method=method,
            doc_mode=doc_mode,
            fitxer_doc=fitxer_doc,
            fitxer_dad=fitxer_dad,
            st_doc="",
            st_dad="",
            t_doc=t_doc,
            y_doc_raw=y_doc_raw,
            y_doc_net=y_doc_net,
            baseline=baseline,
            df_dad=df_dad,
            peak_info=peak_info,
            sample_analysis=sample,
            master_file=str(self.imported_data.get("master_file", "")),
            y_doc_uib=y_doc_uib,
            y_doc_uib_raw=y_doc_uib_raw,
            baseline_uib=baseline_uib,
            fitxer_doc_uib=fitxer_doc_uib,
            st_doc_uib="",
            shift_uib=shift_uib,
            shift_direct=shift_direct,
            master_info=master_info,
            timeout_info=timeout_info,
            snr_info=snr_info,
        )

        return 1


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    """Punt d'entrada principal."""
    app = HPSECSuite()
    app.mainloop()


if __name__ == "__main__":
    main()
