# -*- coding: utf-8 -*-
"""
DOCtor GUI - Eina d'Etiquetatge de Pics HPSEC
=============================================

GUI interactiva per:
- Visualitzar cromatogrames (rèpliques)
- Etiquetar pics com OK/NO
- Exportar dades per anàlisi estadístic
"""

import os
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Importar funcions d'anàlisi de DOCtor_v2
from DOCtor_v2 import (
    read_excel_doc,
    analyze_peaks,
    detect_batman_signal,
    detect_ears,
    calc_pearson_replicates,
    sample_key_from_filename,
    NOISE_THRESHOLD
)


class DOCtorGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("DOCtor GUI - Etiquetatge de Pics")
        self.root.geometry("1400x900")

        # Dades
        self.samples = []  # Llista de mostres processades
        self.current_idx = 0
        self.labels = {}  # {sample_key: {pic_idx: {"r1": "OK"/"NO", "r2": "OK"/"NO"}}}
        self.labels_file = None

        # Construir GUI
        self._build_gui()

    def _build_gui(self):
        """Construeix la interfície."""

        # Frame principal
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        # === HEADER ===
        header = ttk.Frame(main)
        header.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(header, text="Carregar Carpeta", command=self._load_folder).pack(side=tk.LEFT)

        self.lbl_progress = ttk.Label(header, text="Cap dada carregada", font=("Segoe UI", 11))
        self.lbl_progress.pack(side=tk.LEFT, padx=20)

        ttk.Button(header, text="Exportar CSV", command=self._export_csv).pack(side=tk.RIGHT)
        ttk.Button(header, text="Desar Etiquetes", command=self._save_labels).pack(side=tk.RIGHT, padx=5)

        # === GRÀFICS ===
        graphs_frame = ttk.Frame(main)
        graphs_frame.pack(fill=tk.BOTH, expand=True)

        # Frame R1
        self.frame_r1 = ttk.LabelFrame(graphs_frame, text="Rèplica 1", padding=5)
        self.frame_r1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        # Frame R2
        self.frame_r2 = ttk.LabelFrame(graphs_frame, text="Rèplica 2", padding=5)
        self.frame_r2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))

        # Crear figures matplotlib
        self.fig1, self.ax1 = plt.subplots(figsize=(6, 3))
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=self.frame_r1)
        self.canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.fig2, self.ax2 = plt.subplots(figsize=(6, 3))
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=self.frame_r2)
        self.canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # === INFO MOSTRA ===
        info_frame = ttk.Frame(main)
        info_frame.pack(fill=tk.X, pady=10)

        self.lbl_sample_info = ttk.Label(info_frame, text="", font=("Segoe UI", 10))
        self.lbl_sample_info.pack(side=tk.LEFT)

        self.lbl_suggestion = ttk.Label(info_frame, text="", font=("Segoe UI", 10, "bold"))
        self.lbl_suggestion.pack(side=tk.RIGHT)

        # === TAULES DE PICS (una per rèplica) ===
        tables_container = ttk.Frame(main)
        tables_container.pack(fill=tk.BOTH, expand=True, pady=10)

        # Taula R1
        self.table_frame_r1 = ttk.LabelFrame(tables_container, text="Pics Rèplica 1", padding=5)
        self.table_frame_r1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        # Taula R2
        self.table_frame_r2 = ttk.LabelFrame(tables_container, text="Pics Rèplica 2", padding=5)
        self.table_frame_r2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))

        # Amplades fixes per columnes (OK, NO, DOBLE, GHOST, SMALL)
        self.col_widths = [3, 5, 4, 4, 4, 4, 4, 3, 3, 3, 4, 4]  # Pic, t, A, As33, As66, M%, S%, OK, NO, DBL, GHO, SML

        # Crear taules buides (es popularan amb _update_table)
        self.table_r1_container = None
        self.table_r2_container = None

        # Variables per radiobuttons
        self.radio_vars_r1 = []
        self.radio_vars_r2 = []

        # === NAVEGACIÓ ===
        nav_frame = ttk.Frame(main)
        nav_frame.pack(fill=tk.X, pady=10)

        ttk.Button(nav_frame, text="◄ Anterior", command=self._prev_sample).pack(side=tk.LEFT)

        self.lbl_nav = ttk.Label(nav_frame, text="0 / 0", font=("Segoe UI", 12, "bold"))
        self.lbl_nav.pack(side=tk.LEFT, padx=20)

        ttk.Button(nav_frame, text="Següent ►", command=self._next_sample).pack(side=tk.LEFT)

        # Saltar a mostra
        ttk.Label(nav_frame, text="  Anar a:").pack(side=tk.LEFT, padx=(30, 5))
        self.entry_goto = ttk.Entry(nav_frame, width=6)
        self.entry_goto.pack(side=tk.LEFT)
        self.entry_goto.bind("<Return>", lambda e: self._goto_sample())
        ttk.Button(nav_frame, text="Anar", command=self._goto_sample).pack(side=tk.LEFT, padx=5)

        # Filtre
        ttk.Label(nav_frame, text="  Filtre:").pack(side=tk.LEFT, padx=(30, 5))
        self.filter_var = tk.StringVar(value="Totes")
        filter_combo = ttk.Combobox(nav_frame, textvariable=self.filter_var,
                                    values=["Totes", "Sense etiquetar", "Amb problemes", "Peak mismatch", "Nomes BP", "Nomes COLUMN"],
                                    width=15, state="readonly")
        filter_combo.pack(side=tk.LEFT)
        filter_combo.bind("<<ComboboxSelected>>", lambda e: self._apply_filter())

    def _load_folder(self):
        """Carrega carpeta amb SEQs."""
        base_dir = filedialog.askdirectory(title="Selecciona carpeta amb SEQs")
        if not base_dir:
            return

        self.base_dir = base_dir
        self.samples = []
        self.labels = {}

        # Buscar carpetes SEQ
        seq_folders = sorted([
            os.path.join(base_dir, d) for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d, "Resultats_Consolidats"))
        ])

        if not seq_folders:
            messagebox.showwarning("Avís", "No s'han trobat SEQs amb Resultats_Consolidats")
            return

        # Processar
        self.lbl_progress.config(text="Processant...")
        self.root.update()

        for seq_path in seq_folders:
            seq_name = os.path.basename(seq_path)
            results_folder = os.path.join(seq_path, "Resultats_Consolidats")

            files = [
                os.path.join(results_folder, f) for f in os.listdir(results_folder)
                if f.lower().endswith(".xlsx") and "mq" not in f.lower() and "naoh" not in f.lower()
            ]

            if not files:
                continue

            # Agrupar per mostra
            groups = {}
            for f in files:
                key, rep = sample_key_from_filename(os.path.basename(f))
                groups.setdefault(key, []).append((rep or "?", f))

            for k in groups:
                groups[k].sort(key=lambda x: (x[0] != "1", x[0] != "2", x[0]))

            # Processar cada mostra
            for sample_key, items in groups.items():
                sample = self._process_sample(seq_name, sample_key, items)
                if sample:
                    self.samples.append(sample)

        # Carregar etiquetes si existeixen
        self.labels_file = os.path.join(base_dir, "doctor_labels.json")
        if os.path.exists(self.labels_file):
            with open(self.labels_file, "r", encoding="utf-8") as f:
                self.labels = json.load(f)
            messagebox.showinfo("Info", f"Carregades {len(self.labels)} etiquetes existents")

        self.current_idx = 0
        self.filtered_indices = list(range(len(self.samples)))
        self._update_display()

        self.lbl_progress.config(text=f"Carregades {len(self.samples)} mostres de {len(seq_folders)} SEQs")

    def _process_sample(self, seq_name, sample_key, items):
        """Processa una mostra i retorna dict amb tota la info."""
        sample = {
            "seq": seq_name,
            "key": sample_key,
            "unique_key": f"{seq_name}_{sample_key}",
            "type": "?",  # BP o COLUMN
            "reps": [],
            "pearson": np.nan,
            "n_peaks": [0, 0],
            "peak_mismatch": False
        }

        for rep_id, fpath in items[:2]:
            t, y, stype = read_excel_doc(fpath)
            sample["type"] = stype  # Guardar tipus (BP/COLUMN)

            if t is None or len(t) < 10:
                continue

            max_y = float(np.max(y))

            if max_y < NOISE_THRESHOLD:
                continue

            # Analitzar
            peak_analysis = analyze_peaks(t, y)
            bat = detect_batman_signal(t, y)
            ears = detect_ears(t, y)

            # Calcular soroll baseline (primers 10% dels punts)
            baseline_end = max(10, len(y) // 10)
            baseline_noise = float(np.std(y[:baseline_end]))

            rep = {
                "rep_id": rep_id,
                "t": t,
                "y": y,
                "max_y": max_y,
                "peaks": peak_analysis.get("peaks", []) if peak_analysis else [],
                "n_peaks": peak_analysis.get("n_peaks", 0) if peak_analysis else 0,
                "batman": bat,
                "ears": ears,
                "baseline_noise": baseline_noise
            }
            sample["reps"].append(rep)

        if len(sample["reps"]) == 0:
            return None

        # Pearson entre rèpliques
        if len(sample["reps"]) >= 2:
            r1, r2 = sample["reps"][0], sample["reps"][1]
            sample["pearson"], _ = calc_pearson_replicates(r1["t"], r1["y"], r2["t"], r2["y"])
            sample["n_peaks"] = [r1["n_peaks"], r2["n_peaks"]]
            sample["peak_mismatch"] = r1["n_peaks"] != r2["n_peaks"]
        else:
            sample["n_peaks"] = [sample["reps"][0]["n_peaks"], 0]

        return sample

    def _update_display(self):
        """Actualitza la visualització amb la mostra actual."""
        if not self.samples or not self.filtered_indices:
            return

        # Índex real
        real_idx = self.filtered_indices[self.current_idx]
        sample = self.samples[real_idx]

        # Actualitzar navegació
        self.lbl_nav.config(text=f"{self.current_idx + 1} / {len(self.filtered_indices)}")

        # Info mostra
        type_str = f"[{sample['type']}]"
        pears_str = f"ρ={sample['pearson']:.4f}" if not np.isnan(sample['pearson']) else "1 rèplica"
        peaks_str = f"Pics: {sample['n_peaks'][0]}/{sample['n_peaks'][1]}" if len(sample["reps"]) > 1 else f"Pics: {sample['n_peaks'][0]}"
        mismatch_str = " MISMATCH!" if sample["peak_mismatch"] else ""

        noise_strs = []
        for i, rep in enumerate(sample["reps"]):
            noise_strs.append(f"R{i+1}={rep['baseline_noise']:.1f}")
        noise_str = f"Soroll: {', '.join(noise_strs)} mAU"

        self.lbl_sample_info.config(text=f"{sample['seq']} | {sample['key']} {type_str} | {pears_str} | {peaks_str}{mismatch_str} | {noise_str}")

        # Dibuixar gràfics
        self._draw_graphs(sample)

        # Actualitzar taula
        self._update_table(sample)

        # Suggeriment
        self._update_suggestion(sample)

    def _draw_graphs(self, sample):
        """Dibuixa els gràfics de les rèpliques."""
        # Gràfic R1
        self.ax1.clear()
        if len(sample["reps"]) >= 1:
            rep = sample["reps"][0]
            self._draw_single_graph(self.ax1, rep, "R1")
        else:
            self.ax1.text(0.5, 0.5, "Sense dades", ha="center", va="center")
        self.canvas1.draw()

        # Gràfic R2
        self.ax2.clear()
        if len(sample["reps"]) >= 2:
            rep = sample["reps"][1]
            self._draw_single_graph(self.ax2, rep, "R2")
        else:
            self.ax2.text(0.5, 0.5, "Sense rèplica", ha="center", va="center", color="gray")
            self.ax2.set_xticks([])
            self.ax2.set_yticks([])
        self.canvas2.draw()

    def _draw_single_graph(self, ax, rep, title):
        """Dibuixa un sol gràfic."""
        ax.plot(rep["t"], rep["y"], color="#2E86AB", linewidth=1)

        # Ordenar pics per temps per numerar correctament
        peaks_sorted = sorted(enumerate(rep["peaks"]), key=lambda x: x[1]["t_peak"])

        # Calcular offset vertical
        y_range = rep["max_y"] - min(rep["y"])
        y_offset_num = y_range * 0.12  # Per al número
        y_offset_txt = y_range * 0.02  # Per a les mètriques (més baix)

        # Numerar pics i mostrar mètriques
        for display_idx, (orig_idx, pk) in enumerate(peaks_sorted):
            t_pk = pk["t_peak"]
            h_pk = pk["height"]

            # Cercle amb número, més amunt del pic
            ax.annotate(f"{display_idx+1}", (t_pk, h_pk + y_offset_num), fontsize=9, fontweight="bold",
                       ha="center", va="bottom", color="white",
                       bbox=dict(boxstyle="circle,pad=0.3", facecolor="#2E86AB", edgecolor="none"))

            # Mètriques al costat dret del pic (petit, sense recuadre)
            as66 = pk.get("as_66", 0)
            as33 = pk.get("as_33", 0)
            as_div = abs(as66 - as33)

            # Color segons si té problema
            is_prob = pk.get("is_problem", False)
            txt_color = "red" if is_prob else "gray"

            # Text petit al costat dret
            metrics_txt = f"As:{as66:.1f}\nΔ:{as_div:.2f}"
            ax.annotate(metrics_txt, (t_pk + 0.3, h_pk - y_offset_txt),
                       fontsize=6, ha="left", va="top", color=txt_color, alpha=0.8)

        # Marcar Batman
        if rep["batman"]:
            b = rep["batman"]
            ax.scatter([b["t_left"], b["t_right"]], [b["height_left"], b["height_right"]],
                      c="red", s=80, zorder=5, marker="o", edgecolors="darkred", linewidths=2)
            ax.scatter([b["t_valley"]], [b["height_valley"]], c="red", s=80, zorder=5,
                      marker="v", edgecolors="darkred", linewidths=2)
            # Etiqueta BATMAN
            ax.annotate("BATMAN", (b["t_valley"], b["height_valley"]),
                       fontsize=7, ha="center", va="top", color="red", fontweight="bold",
                       xytext=(0, -10), textcoords="offset points")

        # Marcar Orelletes
        for e in rep["ears"][:5]:
            ax.scatter([e["t"]], [e["height"]], c="orange", s=60, zorder=5, marker="^",
                      edgecolors="darkorange", linewidths=1.5)

        ax.set_xlabel("min", fontsize=8)
        ax.set_ylabel("mAU", fontsize=8)
        ax.set_title(f"{title} ({rep['n_peaks']} pics)", fontsize=10)
        ax.grid(True, alpha=0.3)

    def _update_table(self, sample):
        """Actualitza les taules de pics (una per rèplica)."""
        unique_key = sample["unique_key"]

        # Netejar taules anteriors
        for widget in self.table_frame_r1.winfo_children():
            widget.destroy()
        for widget in self.table_frame_r2.winfo_children():
            widget.destroy()

        self.radio_vars_r1 = []
        self.radio_vars_r2 = []

        # Crear taula R1
        if len(sample["reps"]) >= 1:
            peaks_r1 = sample["reps"][0]["peaks"]
            # Ordenar per temps d'aparició
            peaks_r1_sorted = sorted(enumerate(peaks_r1), key=lambda x: x[1]["t_peak"])
            self._create_replica_table(
                self.table_frame_r1, peaks_r1_sorted, unique_key, "r1", self.radio_vars_r1
            )
        else:
            ttk.Label(self.table_frame_r1, text="Sense dades", foreground="gray").pack()

        # Crear taula R2
        if len(sample["reps"]) >= 2:
            peaks_r2 = sample["reps"][1]["peaks"]
            # Ordenar per temps d'aparició
            peaks_r2_sorted = sorted(enumerate(peaks_r2), key=lambda x: x[1]["t_peak"])
            self._create_replica_table(
                self.table_frame_r2, peaks_r2_sorted, unique_key, "r2", self.radio_vars_r2
            )
        else:
            ttk.Label(self.table_frame_r2, text="Sense rèplica", foreground="gray").pack()

    def _create_replica_table(self, parent_frame, peaks_sorted, unique_key, rep_key, radio_vars_list):
        """Crea una taula per una rèplica."""
        # Capçalera
        header_frame = ttk.Frame(parent_frame)
        header_frame.pack(fill=tk.X)

        headers = ["Pic", "t(min)", "A", "As33", "As66", "M%", "S%", "OK", "NO", "DBL", "GHO", "SML"]
        for col, (h, w) in enumerate(zip(headers, self.col_widths)):
            lbl = ttk.Label(header_frame, text=h, width=w, font=("Segoe UI", 9, "bold"),
                           anchor="center", relief="raised", padding=2)
            lbl.grid(row=0, column=col, sticky="ew", padx=1)

        # Container amb scroll
        container = ttk.Frame(parent_frame)
        container.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(container, height=180)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollable = ttk.Frame(canvas)

        scrollable.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        if not peaks_sorted:
            ttk.Label(scrollable, text="Cap pic detectat", foreground="gray").grid(row=0, column=0)
            return

        # Files de dades
        for display_idx, (orig_idx, pk) in enumerate(peaks_sorted):
            # Color de fons segons flags
            bg_color = "#ffffff"
            if pk.get("flags"):
                bg_color = "#fff3cd"  # Groc clar si té problemes

            # Valors
            t_val = f"{pk['t_peak']:.2f}"
            a_val = f"{pk['sym_area']:.2f}"
            as33_val = f"{pk['as_33']:.2f}"
            as66_val = f"{pk['as_66']:.2f}"
            m_val = f"{pk['mono_total']:.0f}"
            s_val = f"{pk.get('smooth_total', 100):.0f}"

            row_frame = tk.Frame(scrollable, bg=bg_color)
            row_frame.grid(row=display_idx, column=0, sticky="ew", pady=1)

            # Número de pic (per ordre d'aparició, 1-indexed)
            tk.Label(row_frame, text=f"{display_idx + 1}", width=self.col_widths[0],
                    anchor="center", bg=bg_color, font=("Segoe UI", 9, "bold")).grid(row=0, column=0, padx=1)
            tk.Label(row_frame, text=t_val, width=self.col_widths[1],
                    anchor="center", bg=bg_color).grid(row=0, column=1, padx=1)
            tk.Label(row_frame, text=a_val, width=self.col_widths[2],
                    anchor="center", bg=bg_color).grid(row=0, column=2, padx=1)
            tk.Label(row_frame, text=as33_val, width=self.col_widths[3],
                    anchor="center", bg=bg_color).grid(row=0, column=3, padx=1)
            tk.Label(row_frame, text=as66_val, width=self.col_widths[4],
                    anchor="center", bg=bg_color).grid(row=0, column=4, padx=1)
            tk.Label(row_frame, text=m_val, width=self.col_widths[5],
                    anchor="center", bg=bg_color).grid(row=0, column=5, padx=1)
            tk.Label(row_frame, text=s_val, width=self.col_widths[6],
                    anchor="center", bg=bg_color).grid(row=0, column=6, padx=1)

            # Radiobuttons OK/NO/DOBLE/GHOST/SMALL
            # Per defecte: OK (l'usuari només canvia els que no ho son)
            var = tk.StringVar(value="OK")

            # Carregar etiqueta existent (usem display_idx com a clau)
            label_key = f"{rep_key}_{display_idx}"
            if unique_key in self.labels and label_key in self.labels[unique_key]:
                var.set(self.labels[unique_key][label_key])

            rb_ok = ttk.Radiobutton(row_frame, text="", variable=var, value="OK",
                                    command=lambda uk=unique_key, rk=rep_key, di=display_idx:
                                    self._on_label_change_new(uk, rk, di))
            rb_ok.grid(row=0, column=7, padx=1)

            rb_no = ttk.Radiobutton(row_frame, text="", variable=var, value="NO",
                                    command=lambda uk=unique_key, rk=rep_key, di=display_idx:
                                    self._on_label_change_new(uk, rk, di))
            rb_no.grid(row=0, column=8, padx=1)

            rb_doble = ttk.Radiobutton(row_frame, text="", variable=var, value="DOBLE",
                                       command=lambda uk=unique_key, rk=rep_key, di=display_idx:
                                       self._on_label_change_new(uk, rk, di))
            rb_doble.grid(row=0, column=9, padx=1)

            rb_ghost = ttk.Radiobutton(row_frame, text="", variable=var, value="GHOST",
                                       command=lambda uk=unique_key, rk=rep_key, di=display_idx:
                                       self._on_label_change_new(uk, rk, di))
            rb_ghost.grid(row=0, column=10, padx=1)

            rb_small = ttk.Radiobutton(row_frame, text="", variable=var, value="SMALL",
                                       command=lambda uk=unique_key, rk=rep_key, di=display_idx:
                                       self._on_label_change_new(uk, rk, di))
            rb_small.grid(row=0, column=11, padx=1)

            radio_vars_list.append((var, orig_idx, display_idx))

    def _on_label_change_new(self, unique_key, rep_key, display_idx):
        """Callback quan canvia una etiqueta (nou format)."""
        if unique_key not in self.labels:
            self.labels[unique_key] = {}

        label_key = f"{rep_key}_{display_idx}"

        # Obtenir el valor del radiobutton corresponent
        if rep_key == "r1":
            for var, orig_idx, di in self.radio_vars_r1:
                if di == display_idx:
                    self.labels[unique_key][label_key] = var.get()
                    break
        else:
            for var, orig_idx, di in self.radio_vars_r2:
                if di == display_idx:
                    self.labels[unique_key][label_key] = var.get()
                    break

        # Actualitzar suggeriment
        real_idx = self.filtered_indices[self.current_idx]
        self._update_suggestion(self.samples[real_idx])

    def _update_suggestion(self, sample):
        """Actualitza el suggeriment de millor rèplica amb detecció automàtica."""
        unique_key = sample["unique_key"]

        # Prioritat 1: Mostrar BATMAN si detectat
        batman_r1 = sample["reps"][0].get("batman") if len(sample["reps"]) >= 1 else None
        batman_r2 = sample["reps"][1].get("batman") if len(sample["reps"]) >= 2 else None

        if batman_r1 and batman_r2:
            self.lbl_suggestion.config(text="!! BATMAN EN AMBDUES REPLIQUES !!", foreground="red")
            return
        elif batman_r1:
            self.lbl_suggestion.config(text="!! BATMAN en R1 - Triar R2 !!", foreground="red")
            return
        elif batman_r2:
            self.lbl_suggestion.config(text="!! BATMAN en R2 - Triar R1 !!", foreground="red")
            return

        # Prioritat 2: Warning de selecció (del DOCtor_v2)
        selection_warning = sample.get("selection_warning")
        if selection_warning:
            self.lbl_suggestion.config(text=selection_warning, foreground="darkorange")
            return

        # Prioritat 3: Mostrar motiu de selecció automàtica
        if len(sample["reps"]) >= 1:
            rep1 = sample["reps"][0]
            motiu_r1 = rep1.get("_motiu", "")
            triada_r1 = rep1.get("triada", False)
            dominated_r1 = rep1.get("_dominated", False)

        if len(sample["reps"]) >= 2:
            rep2 = sample["reps"][1]
            motiu_r2 = rep2.get("_motiu", "")
            triada_r2 = rep2.get("triada", False)
            dominated_r2 = rep2.get("_dominated", False)

            # Mostrar selecció automàtica
            if triada_r1 and motiu_r1:
                color = "red" if dominated_r2 else "green"
                self.lbl_suggestion.config(text=f"Auto: R1 | {motiu_r1}", foreground=color)
            elif triada_r2 and motiu_r2:
                color = "red" if dominated_r1 else "green"
                self.lbl_suggestion.config(text=f"Auto: R2 | {motiu_r2}", foreground=color)
            else:
                self.lbl_suggestion.config(text="Empat automàtic", foreground="blue")
            return

        elif len(sample["reps"]) == 1:
            rep1 = sample["reps"][0]
            motiu = rep1.get("_motiu", "")
            dominated = rep1.get("_dominated", False)

            if dominated:
                self.lbl_suggestion.config(text=f"WARNING: {motiu}", foreground="orange")
            else:
                self.lbl_suggestion.config(text=f"Única rèplica OK", foreground="gray")
            return

        self.lbl_suggestion.config(text="Sense dades", foreground="gray")

    def _prev_sample(self):
        """Mostra anterior."""
        if self.current_idx > 0:
            self._auto_save()  # Guardar abans de canviar
            self.current_idx -= 1
            self._update_display()

    def _next_sample(self):
        """Mostra següent."""
        if self.current_idx < len(self.filtered_indices) - 1:
            self._auto_save()  # Guardar abans de canviar
            self.current_idx += 1
            self._update_display()

    def _auto_save(self):
        """Guarda etiquetes automàticament (sense missatge)."""
        if self.labels_file and self.labels:
            try:
                with open(self.labels_file, "w", encoding="utf-8") as f:
                    json.dump(self.labels, f, indent=2, ensure_ascii=False)
            except:
                pass  # Silenciós per no molestar

    def _goto_sample(self):
        """Salta a una mostra específica."""
        try:
            idx = int(self.entry_goto.get()) - 1
            if 0 <= idx < len(self.filtered_indices):
                self._auto_save()  # Guardar abans de canviar
                self.current_idx = idx
                self._update_display()
        except ValueError:
            pass

    def _apply_filter(self):
        """Aplica filtre a les mostres."""
        self._auto_save()  # Guardar abans de canviar filtre
        filter_type = self.filter_var.get()

        self.filtered_indices = []

        for i, sample in enumerate(self.samples):
            include = False

            if filter_type == "Totes":
                include = True
            elif filter_type == "Sense etiquetar":
                # No té cap etiqueta
                include = sample["unique_key"] not in self.labels
            elif filter_type == "Amb problemes":
                # Algun pic té flags
                for rep in sample["reps"]:
                    for pk in rep["peaks"]:
                        if pk.get("flags"):
                            include = True
                            break
            elif filter_type == "Peak mismatch":
                include = sample["peak_mismatch"]
            elif filter_type == "Nomes BP":
                include = sample.get("type") == "BP"
            elif filter_type == "Nomes COLUMN":
                include = sample.get("type") == "COLUMN"

            if include:
                self.filtered_indices.append(i)

        self.current_idx = 0
        if self.filtered_indices:
            self._update_display()
        else:
            messagebox.showinfo("Info", "Cap mostra coincideix amb el filtre")
            self.filter_var.set("Totes")
            self._apply_filter()

    def _save_labels(self):
        """Desa les etiquetes a JSON."""
        if not self.labels_file:
            messagebox.showwarning("Avís", "Primer carrega una carpeta")
            return

        with open(self.labels_file, "w", encoding="utf-8") as f:
            json.dump(self.labels, f, indent=2, ensure_ascii=False)

        n_samples = len(self.labels)
        n_peaks = sum(len(peaks) for peaks in self.labels.values())
        messagebox.showinfo("Desat", f"Etiquetes desades:\n{n_samples} mostres\n{n_peaks} pics etiquetats")

    def _export_csv(self):
        """Exporta totes les dades a CSV."""
        if not self.samples:
            messagebox.showwarning("Avís", "No hi ha dades")
            return

        # Preparar dades
        rows = []

        for sample in self.samples:
            unique_key = sample["unique_key"]

            for rep_idx, rep in enumerate(sample["reps"]):
                rep_key = f"r{rep_idx + 1}"
                rep_name = f"R{rep_idx + 1}"

                # Ordenar pics per temps
                peaks_sorted = sorted(enumerate(rep["peaks"]), key=lambda x: x[1]["t_peak"])

                for display_idx, (orig_idx, pk) in enumerate(peaks_sorted):
                    # Obtenir etiqueta si existeix (nou format)
                    label_key = f"{rep_key}_{display_idx}"
                    label = self.labels.get(unique_key, {}).get(label_key, "")

                    rows.append({
                        "seq": sample["seq"],
                        "sample": sample["key"],
                        "type": sample.get("type", ""),
                        "replica": rep_name,
                        "peak_num": display_idx + 1,
                        "t_peak": pk["t_peak"],
                        "height": pk["height"],
                        "area_total": pk.get("total_area", 0),
                        "area_left": pk.get("area_left", 0),
                        "area_right": pk.get("area_right", 0),
                        "A": pk["sym_area"],
                        "As33": pk["as_33"],
                        "As66": pk["as_66"],
                        "As_divergence": pk["as_divergence"],
                        "M_total": pk["mono_total"],
                        "M_left": pk["mono_left"],
                        "M_right": pk["mono_right"],
                        "S_total": pk.get("smooth_total", 100),
                        "S_left": pk.get("smooth_left", 100),
                        "S_right": pk.get("smooth_right", 100),
                        "flags": "|".join(pk.get("flags", [])),
                        "label": label,
                        "pearson": sample["pearson"],
                        "baseline_noise": rep["baseline_noise"],
                        "batman": "Yes" if rep["batman"] else "No",
                        "n_ears": len(rep["ears"])
                    })

        # Desar
        df = pd.DataFrame(rows)
        csv_path = os.path.join(self.base_dir, f"doctor_peaks_{datetime.now().strftime('%Y%m%d_%H%M')}.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        messagebox.showinfo("Exportat", f"CSV desat:\n{csv_path}\n\n{len(rows)} pics exportats")

    def run(self):
        """Inicia l'aplicació."""
        # Guardar en tancar
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()

    def _on_close(self):
        """Guarda i tanca."""
        self._auto_save()
        self.root.destroy()


if __name__ == "__main__":
    app = DOCtorGUI()
    app.run()
