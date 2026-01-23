# -*- coding: utf-8 -*-
"""
DOCtor_Diag v1.4 - Diagnòstic i Validació d'Anomalies HPSEC
===========================================================

GUI per etiquetar manualment pics i validar deteccions.
Genera CSV amb etiquetes per entrenar/ajustar llindars.

Part de la SUITE HPSEC.
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from scipy.signal import find_peaks

# Importar funcions compartides
from hpsec_core import (
    detect_batman, calc_top_smoothness
)

# =============================================================================
# PARÀMETRES
# =============================================================================
ANALYSIS_T_START = 10.0
ANALYSIS_T_END = 70.0
MIN_PEAK_PROMINENCE_PCT = 3.0

# Regió HS (Humic Substances) - Pic Principal
# Intervals definits a HPSEC_Suite: BioP[0-18], HS[18-23], BB[23-30], Carbonats[30-40]
HS_REGION_START = 18.0  # min
HS_REGION_END = 23.0    # min

LABELS = ["", "OK", "BAT", "IRR", "STOP", "FP_BAT", "FP_IRR", "FP_STOP", "OTHER"]


# =============================================================================
# FUNCIONS AUXILIARS
# =============================================================================
def read_excel_doc(filepath):
    """Llegeix fitxer Excel i retorna t, y."""
    try:
        xls = pd.ExcelFile(filepath, engine="openpyxl")
        sheet = "DOC" if "DOC" in xls.sheet_names else xls.sheet_names[0]
        df = pd.read_excel(xls, sheet_name=sheet)
        df.columns = [str(c).strip().lower() for c in df.columns]

        t_col = next((c for c in df.columns if "time" in c or "min" in c), None)
        y_col = next((c for c in df.columns if "doc" in c or "mau" in c or "signal" in c), None)

        if t_col is None or y_col is None:
            return None, None

        t = pd.to_numeric(df[t_col], errors="coerce").values
        y = pd.to_numeric(df[y_col], errors="coerce").values

        mask = np.isfinite(t) & np.isfinite(y)
        return t[mask], y[mask]
    except Exception as e:
        print(f"Error llegint {filepath}: {e}")
        return None, None


def calc_asymmetry_factor(t, y, peak_idx, baseline, height_pct=50):
    """Calcula factor d'asimetria a una altura donada."""
    if peak_idx <= 0 or peak_idx >= len(y) - 1:
        return 1.0

    peak_height = y[peak_idx] - baseline
    if peak_height <= 0:
        return 1.0

    target_y = baseline + peak_height * (height_pct / 100.0)

    left_t = None
    for i in range(peak_idx, 0, -1):
        if y[i] <= target_y:
            if y[i+1] != y[i]:
                frac = (target_y - y[i]) / (y[i+1] - y[i])
                left_t = t[i] + frac * (t[i+1] - t[i])
            else:
                left_t = t[i]
            break

    right_t = None
    for i in range(peak_idx, len(y) - 1):
        if y[i] <= target_y:
            if y[i-1] != y[i]:
                frac = (target_y - y[i]) / (y[i-1] - y[i])
                right_t = t[i] + frac * (t[i-1] - t[i])
            else:
                right_t = t[i]
            break

    if left_t is None or right_t is None:
        return 1.0

    t_peak = t[peak_idx]
    width_left = t_peak - left_t
    width_right = right_t - t_peak

    if width_left <= 0:
        return 1.0

    return width_right / width_left


def detect_linear_segments(t, y, segment_size=5, min_r2=0.98, min_angle_deg=45, min_angle_changes=3):
    """
    Detecta si un pic té segments excessivament lineals (rectes) amb angles bruscos.

    Pics amorfes sovint tenen trams rectes amb canvis d'angle pronunciats,
    en lloc de corbes suaus.

    Parameters:
        t, y: temps i senyal del segment del pic
        segment_size: mida de cada segment a analitzar (punts)
        min_r2: R² mínim per considerar un segment lineal (0.98 = molt recte)
        min_angle_deg: angle mínim (graus) per considerar un canvi brusc (default: 45°)
        min_angle_changes: nombre mínim de canvis bruscos per considerar amorf

    Returns:
        dict amb is_linear, n_linear_segments, max_r2, angles, n_angle_changes
    """
    from scipy.stats import linregress

    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(t) < segment_size * 2:
        return {"is_linear": False, "n_linear_segments": 0, "max_r2": 0,
                "n_angle_changes": 0, "reason": "too_short"}

    # SUBMOSTREJAR: 1 de cada 5 punts per reduir sensibilitat al soroll
    step = 5
    t_sub = t[::step]
    y_sub = y[::step]

    if len(t_sub) < segment_size * 2:
        # Si després de submostrar és massa curt, usar dades originals
        t_sub = t
        y_sub = y

    # Dividir en segments solapats
    linear_segments = []
    slopes = []

    for i in range(0, len(t_sub) - segment_size, segment_size // 2):
        t_seg = t_sub[i:i+segment_size]
        y_seg = y_sub[i:i+segment_size]

        if len(t_seg) < 3:
            continue

        # Regressió lineal
        slope, intercept, r_value, _, _ = linregress(t_seg, y_seg)
        r2 = r_value ** 2

        if r2 >= min_r2:
            linear_segments.append({
                "start_idx": i,
                "end_idx": i + segment_size,
                "r2": r2,
                "slope": slope,
                "t_start": t_seg[0],
                "t_end": t_seg[-1]
            })
            slopes.append(slope)

    # Calcular ANGLES entre segments lineals consecutius
    angles = []
    if len(slopes) > 1:
        for i in range(len(slopes) - 1):
            m1 = slopes[i]
            m2 = slopes[i+1]

            # Angle entre dues rectes: arctan(|(m2-m1)/(1+m1*m2)|)
            denominator = 1 + m1 * m2
            if abs(denominator) < 1e-6:
                # Rectes perpendiculars (angle = 90°)
                angle_deg = 90.0
            else:
                angle_rad = np.arctan(abs((m2 - m1) / denominator))
                angle_deg = np.degrees(angle_rad)

            angles.append(angle_deg)

    # Comptar canvis d'angle > threshold
    n_angle_changes = sum(1 for angle in angles if angle > min_angle_deg)

    max_r2 = max([seg["r2"] for seg in linear_segments]) if linear_segments else 0
    n_linear = len(linear_segments)

    # CRITERI: Múltiples canvis d'angle bruscos (>45°)
    is_linear = n_angle_changes >= min_angle_changes

    return {
        "is_linear": is_linear,
        "n_linear_segments": n_linear,
        "max_r2": max_r2,
        "angles": angles,
        "n_angle_changes": n_angle_changes,
        "max_angle": max(angles) if angles else 0,
        "segments": linear_segments,
        "reason": f"angle_changes={n_angle_changes}" if is_linear else "smooth_curve"
    }


def detect_stop(t, y, top_pct=0.30, max_cv=0.1, min_duration=0.3):
    """
    Detecta STOP/TimeOUT (meseta/saturació) al cim del pic.

    Una meseta és una zona plana al cim (baixa variabilitat).

    Parameters:
        t, y: temps i senyal del segment del pic
        top_pct: percentatge del cim a analitzar (0.30 = top 30%)
        max_cv: CV màxim (%) per considerar pla (0.1% = meseta real)
        min_duration: duració mínima de la meseta (min)

    Returns:
        dict amb is_stop, cv, duration, info
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(y) < 10:
        return {"is_stop": False, "cv": 100.0, "duration": 0, "reason": "too_short"}

    baseline = float(np.percentile(y, 10))
    y_max = float(np.max(y))
    height = y_max - baseline

    if height <= 0:
        return {"is_stop": False, "cv": 100.0, "duration": 0, "reason": "no_height"}

    # Extreure zona del cim
    threshold = baseline + height * (1 - top_pct)
    top_mask = y >= threshold
    t_top = t[top_mask]
    y_top = y[top_mask]

    if len(y_top) < 5:
        return {"is_stop": False, "cv": 100.0, "duration": 0, "reason": "top_too_small"}

    # Calcular CV (Coefficient of Variation) = (std / mean) * 100
    y_mean = np.mean(y_top)
    y_std = np.std(y_top)

    if y_mean > 0:
        cv = (y_std / y_mean) * 100
    else:
        cv = 100.0

    # Duració de la zona del cim
    duration = float(t_top[-1] - t_top[0])

    # Criteri: CV baix i duració suficient
    is_stop = (cv <= max_cv) and (duration >= min_duration)

    return {
        "is_stop": is_stop,
        "cv": cv,
        "duration": duration,
        "y_mean": y_mean,
        "y_std": y_std,
        "t_top": t_top,
        "y_top": y_top,
        "threshold": threshold,
        "reason": "stop_detected" if is_stop else "not_flat_enough"
    }


def analyze_peaks_for_diag(t, y):
    """Analitza pics per diagnòstic."""
    try:
        if t is None or len(t) < 20:
            print(f"  AVIS: Dades insuficients per anàlisi (len={len(t) if t is not None else 0})")
            return []

        mask = (t >= ANALYSIS_T_START) & (t <= ANALYSIS_T_END)
        t_win = t[mask]
        y_win = y[mask]

        if len(t_win) < 20:
            print(f"  AVIS: Dades insuficients a finestra [{ANALYSIS_T_START}-{ANALYSIS_T_END}] (len={len(t_win)})")
            return []

        y_max = float(np.max(y_win))
        min_prominence = y_max * (MIN_PEAK_PROMINENCE_PCT / 100.0)

        print(f"  - Cercant pics (prominence >= {min_prominence:.1f} mAU)...")
        peaks, props = find_peaks(y_win, prominence=min_prominence, width=3)

        if len(peaks) == 0:
            print(f"  AVIS: Cap pic detectat")
            return []

        print(f"  - {len(peaks)} pics trobats")

        # LÍMIT DE SEGURETAT: màxim 50 pics
        if len(peaks) > 50:
            print(f"  AVIS: Massa pics detectats ({len(peaks)}). Limitant a 50 primers.")
            peaks = peaks[:50]
            for key in props:
                props[key] = props[key][:50]

        baseline = float(np.percentile(y_win, 10))

        # IDENTIFICAR PIC PRINCIPAL (més alt a la regió HS 18-22 min)
        main_peak_idx = None
        max_height_in_hs = 0
        for i, pk_idx in enumerate(peaks):
            pk_t = float(t_win[pk_idx])
            pk_height = float(y_win[pk_idx])

            # Comprovar si està a la regió HS
            if HS_REGION_START <= pk_t <= HS_REGION_END:
                if pk_height > max_height_in_hs:
                    max_height_in_hs = pk_height
                    main_peak_idx = i

        if main_peak_idx is not None:
            main_peak_t = float(t_win[peaks[main_peak_idx]])
            print(f"  OK: Pic principal identificat: P{main_peak_idx+1} a t={main_peak_t:.2f} min (altura={max_height_in_hs:.1f} mAU)")
        else:
            print(f"  AVIS: Cap pic a la regió HS ({HS_REGION_START}-{HS_REGION_END} min)")

        results = []
        for i, pk_idx in enumerate(peaks):
            if i % 10 == 0 and i > 0:
                print(f"    Processant pic {i}/{len(peaks)}...")

            pk_t = float(t_win[pk_idx])
            pk_height = float(y_win[pk_idx])

            left_base = int(props["left_bases"][i])
            right_base = int(props["right_bases"][i])

            as_33 = calc_asymmetry_factor(t_win, y_win, pk_idx, baseline, height_pct=33)
            as_66 = calc_asymmetry_factor(t_win, y_win, pk_idx, baseline, height_pct=66)
            as_div = abs(as_66 - as_33)

            t_seg = t_win[left_base:right_base+1]
            y_seg = y_win[left_base:right_base+1]

            batman_pct = 0.0
            irr_pct = 100.0
            stop_cv = 100.0
            is_stop = False
            is_linear = False
            n_linear_segments = 0
            n_angle_changes = 0

            if len(t_seg) >= 10:
                try:
                    # BATMAN: NOMÉS al pic principal a la regió HS
                    if i == main_peak_idx:
                        bat_result = detect_batman(t_seg, y_seg, top_pct=0.15, min_valley_depth=0.01)
                        if bat_result.get("is_batman"):
                            batman_pct = bat_result.get("max_depth", 0) * 100

                    # IRR: a tots els pics
                    smooth_result = calc_top_smoothness(t_seg, y_seg, top_pct=0.15)
                    irr_pct = smooth_result.get("smoothness", 100.0)

                    # STOP (meseta/saturació): CV < 0.1% = meseta real
                    stop_result = detect_stop(t_seg, y_seg, top_pct=0.30, max_cv=0.1, min_duration=0.3)
                    is_stop = stop_result.get("is_stop", False)
                    stop_cv = stop_result.get("cv", 100.0)

                    # LINEAR (segments rectes amb angles bruscos): Pics/valls amorfes
                    linear_result = detect_linear_segments(t_seg, y_seg, segment_size=5, min_r2=0.98,
                                                          min_angle_deg=45, min_angle_changes=3)
                    is_linear = linear_result.get("is_linear", False)
                    n_linear_segments = linear_result.get("n_linear_segments", 0)
                    n_angle_changes = linear_result.get("n_angle_changes", 0)
                except Exception as e:
                    print(f"    AVIS: Error analitzant pic {i+1}: {e}")

            results.append({
                "num": i + 1,
                "t_peak": pk_t,
                "height": pk_height,
                "as_33": as_33,
                "as_66": as_66,
                "as_div": as_div,
                "batman_pct": batman_pct,
                "irr_pct": irr_pct,
                "is_stop": is_stop,
                "stop_cv": stop_cv,
                "is_linear": is_linear,
                "n_linear_segments": n_linear_segments,
                "n_angle_changes": n_angle_changes,
                "left_base": left_base,
                "right_base": right_base,
                "label": "",
                "notes": "",
                "is_main_peak": (i == main_peak_idx)
            })

        return results

    except Exception as e:
        print(f"  ERROR a analyze_peaks_for_diag: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return []


# =============================================================================
# GUI PRINCIPAL
# =============================================================================
class DiagApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("DOCtor_Diag v1.5 - Validació d'Anomalies")
        self.root.geometry("1400x950")
        self.root.state('zoomed')  # Maximitzar finestra

        # Gestió de SEQs
        self.dades_folder = None
        self.seq_list = []  # Llista de (seq_name, results_path)
        self.current_seq_idx = 0

        # Gestió de fitxers dins de la SEQ actual
        self.files = []
        self.file_info = []
        self.current_file_idx = 0

        # Dades actuals
        self.current_t = None
        self.current_y = None
        self.current_peaks = []
        self.all_results = []
        self.selected_peak_num = None

        self._build_gui()

    def _build_gui(self):
        # Frame principal
        main = tk.Frame(self.root, padx=10, pady=10)
        main.pack(fill=tk.BOTH, expand=True)

        # === TOP: Botons navegació ===
        top_frame = tk.Frame(main)
        top_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Button(top_frame, text="Carregar DADES", command=self._load_folder,
                  bg="#4CAF50", fg="white", padx=10).pack(side=tk.LEFT, padx=5)

        tk.Button(top_frame, text="Skip", command=self._skip_file,
                  bg="#FF5722", fg="white", padx=8, font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=5)

        # Navegació SEQ
        tk.Label(top_frame, text="SEQ:").pack(side=tk.LEFT, padx=(15, 5))
        self.btn_prev_seq = tk.Button(top_frame, text="<", command=self._prev_seq, padx=5, state=tk.DISABLED)
        self.btn_prev_seq.pack(side=tk.LEFT, padx=2)

        self.seq_var = tk.StringVar(value="---")
        self.seq_combo = ttk.Combobox(top_frame, textvariable=self.seq_var, width=15, state="readonly")
        self.seq_combo.pack(side=tk.LEFT, padx=2)
        self.seq_combo.bind("<<ComboboxSelected>>", self._on_seq_selected)

        self.btn_next_seq = tk.Button(top_frame, text=">", command=self._next_seq, padx=5, state=tk.DISABLED)
        self.btn_next_seq.pack(side=tk.LEFT, padx=2)

        # Navegació fitxers
        tk.Label(top_frame, text="  Fitxer:").pack(side=tk.LEFT, padx=(15, 5))
        self.btn_prev_file = tk.Button(top_frame, text="<", command=self._prev_file, padx=5, state=tk.DISABLED)
        self.btn_prev_file.pack(side=tk.LEFT, padx=2)
        self.btn_next_file = tk.Button(top_frame, text=">", command=self._next_file, padx=5, state=tk.DISABLED)
        self.btn_next_file.pack(side=tk.LEFT, padx=2)

        self.file_label = tk.Label(top_frame, text="Cap fitxer carregat", font=("Segoe UI", 9))
        self.file_label.pack(side=tk.LEFT, padx=10)

        self.progress_label = tk.Label(top_frame, text="", font=("Segoe UI", 9))
        self.progress_label.pack(side=tk.LEFT, padx=5)

        tk.Button(top_frame, text="Guardar CSV", command=self._save_csv,
                  bg="#2196F3", fg="white", padx=10).pack(side=tk.RIGHT, padx=5)

        # === EDICIÓ (ABANS del gràfic per ser visible) ===
        edit_frame = tk.LabelFrame(main, text="EDITAR PIC SELECCIONAT", padx=10, pady=5,
                                   font=("Segoe UI", 10, "bold"), fg="darkblue")
        edit_frame.pack(fill=tk.X, pady=(0, 10))

        # Fila única amb tot
        tk.Label(edit_frame, text="Pic:").pack(side=tk.LEFT, padx=(0, 5))
        self.selected_label = tk.Label(edit_frame, text="-", font=("Segoe UI", 11, "bold"),
                                       fg="blue", width=5)
        self.selected_label.pack(side=tk.LEFT, padx=(0, 15))

        tk.Label(edit_frame, text="Etiqueta:").pack(side=tk.LEFT, padx=(0, 5))

        self.label_var = tk.StringVar(value="")
        for lbl in LABELS:
            display = lbl if lbl else "·"
            bg_color = "#e0e0e0" if not lbl else None
            btn = tk.Radiobutton(edit_frame, text=display, variable=self.label_var,
                                value=lbl, indicatoron=0, width=6, padx=3, pady=2)
            btn.pack(side=tk.LEFT, padx=1)

        tk.Label(edit_frame, text="  Notes:").pack(side=tk.LEFT, padx=(15, 5))
        self.notes_entry = tk.Entry(edit_frame, width=35)
        self.notes_entry.pack(side=tk.LEFT, padx=(0, 10))

        tk.Button(edit_frame, text="APLICAR", command=self._apply_edit,
                  bg="#FF9800", fg="white", padx=15, font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=5)

        # === MIG: Gràfic ===
        self.fig = Figure(figsize=(12, 3.5), dpi=100)
        self.ax = self.fig.add_subplot(111)

        canvas_frame = tk.Frame(main)
        canvas_frame.pack(fill=tk.BOTH, expand=False, pady=(0, 5))

        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, canvas_frame)
        toolbar.update()

        # === BAIX: Taula ===
        table_frame = tk.LabelFrame(main, text="Pics Detectats", padx=5, pady=5)
        table_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        # Scrollbars
        scroll_y = tk.Scrollbar(table_frame, orient=tk.VERTICAL)
        scroll_x = tk.Scrollbar(table_frame, orient=tk.HORIZONTAL)

        columns = ("num", "t_peak", "height", "as33", "as66", "delta_as", "bat", "irr", "stop", "label", "notes")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=8,
                                  yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)

        scroll_y.config(command=self.tree.yview)
        scroll_x.config(command=self.tree.xview)

        self.tree.heading("num", text="#")
        self.tree.heading("t_peak", text="t_pic")
        self.tree.heading("height", text="Altura")
        self.tree.heading("as33", text="AS33")
        self.tree.heading("as66", text="AS66")
        self.tree.heading("delta_as", text="ΔAs")
        self.tree.heading("bat", text="BAT%")
        self.tree.heading("irr", text="IRR%")
        self.tree.heading("stop", text="STOP(CV)")
        self.tree.heading("label", text="Etiqueta")
        self.tree.heading("notes", text="Notes")

        self.tree.column("num", width=40, anchor="center")
        self.tree.column("t_peak", width=70, anchor="center")
        self.tree.column("height", width=70, anchor="center")
        self.tree.column("as33", width=55, anchor="center")
        self.tree.column("as66", width=55, anchor="center")
        self.tree.column("delta_as", width=55, anchor="center")
        self.tree.column("bat", width=55, anchor="center")
        self.tree.column("irr", width=55, anchor="center")
        self.tree.column("stop", width=70, anchor="center")
        self.tree.column("label", width=75, anchor="center")
        self.tree.column("notes", width=180, anchor="w")

        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.tree.pack(fill=tk.BOTH, expand=True)

        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)

    def _load_folder(self):
        """Carrega la carpeta DADES i detecta totes les SEQs disponibles."""
        folder = filedialog.askdirectory(title="Selecciona carpeta DADES (arrel amb SEQs)")
        if not folder:
            return

        self.dades_folder = folder
        self.seq_list = []

        print(f"\n=== Cercant SEQs a: {folder} ===")

        # Buscar subcarpetes amb Resultats_Consolidats (excloent _BP)
        try:
            for item in os.listdir(folder):
                item_path = os.path.join(folder, item)

                # Excloure carpetes que acabin amb _BP
                if item.endswith("_BP"):
                    print(f"  Excloent (BP): {item}")
                    continue

                if os.path.isdir(item_path):
                    results_path = os.path.join(item_path, "Resultats_Consolidats")
                    if os.path.isdir(results_path):
                        self.seq_list.append((item, results_path))
                        print(f"  OK: SEQ trobada: {item}")
        except Exception as e:
            print(f"Error llistant carpeta: {e}")
            messagebox.showerror("Error", f"Error llegint carpeta:\n{e}")
            return

        # Si no hi ha SEQs, buscar Resultats_Consolidats directament
        results_direct = os.path.join(folder, "Resultats_Consolidats")
        if os.path.isdir(results_direct) and not self.seq_list:
            seq_name = os.path.basename(folder)
            if not seq_name.endswith("_BP"):
                self.seq_list.append((seq_name, results_direct))
                print(f"  OK: Resultats directes trobats: {seq_name}")

        if not self.seq_list:
            messagebox.showwarning("Avís", "No s'han trobat carpetes Resultats_Consolidats\n(o totes són _BP)")
            return

        # Ordenar SEQs alfabèticament
        self.seq_list.sort(key=lambda x: x[0])

        print(f"\n=== Total: {len(self.seq_list)} SEQs trobades ===")

        # Actualitzar combo box
        seq_names = [seq[0] for seq in self.seq_list]
        self.seq_combo['values'] = seq_names

        # Activar controls
        self.btn_prev_seq.config(state=tk.NORMAL if len(self.seq_list) > 1 else tk.DISABLED)
        self.btn_next_seq.config(state=tk.NORMAL if len(self.seq_list) > 1 else tk.DISABLED)

        # Carregar primera SEQ
        self.current_seq_idx = 0
        self.seq_combo.current(0)
        self.all_results = []

        messagebox.showinfo("Carregat", f"{len(self.seq_list)} SEQs trobades\n\nSeleccionada: {seq_names[0]}")

        self._load_seq_files()

    def _load_seq_files(self):
        """Carrega els fitxers de la SEQ actual."""
        if not self.seq_list or self.current_seq_idx >= len(self.seq_list):
            return

        seq_name, results_path = self.seq_list[self.current_seq_idx]
        self.files = []
        self.file_info = []

        print(f"\n=== Carregant fitxers de SEQ: {seq_name} ===")

        try:
            for f in os.listdir(results_path):
                if not f.lower().endswith(".xlsx") or f.startswith("~$"):
                    continue

                fname_lower = f.lower()

                # Excloure fitxers BP
                if "_bp" in fname_lower or fname_lower.endswith("bp.xlsx"):
                    continue

                # Excloure patrons
                if any(x in fname_lower for x in ["mq", "naoh", "khp", "no3", "na2co3", "blank", "blanco"]):
                    continue

                filepath = os.path.join(results_path, f)
                self.files.append(filepath)
                self.file_info.append({"seq": seq_name, "filename": f})
                print(f"  OK: {f}")
        except Exception as e:
            print(f"Error llegint {results_path}: {e}")
            messagebox.showerror("Error", f"Error llegint SEQ {seq_name}:\n{e}")
            return

        print(f"Total: {len(self.files)} fitxers")

        if not self.files:
            messagebox.showwarning("Avís", f"No s'han trobat fitxers XLSX vàlids a {seq_name}")
            self.file_label.config(text=f"[{seq_name}] Sense fitxers")
            self.progress_label.config(text="")
            self.btn_prev_file.config(state=tk.DISABLED)
            self.btn_next_file.config(state=tk.DISABLED)
            return

        # Activar botons de navegació de fitxers
        self.btn_prev_file.config(state=tk.NORMAL if len(self.files) > 1 else tk.DISABLED)
        self.btn_next_file.config(state=tk.NORMAL if len(self.files) > 1 else tk.DISABLED)

        # Carregar primer fitxer
        self.current_file_idx = 0
        self._load_current_file()

    def _prev_seq(self):
        """Navega a la SEQ anterior."""
        if self.current_seq_idx > 0:
            self._save_current_peaks()
            self.current_seq_idx -= 1
            self.seq_combo.current(self.current_seq_idx)
            self._load_seq_files()

    def _next_seq(self):
        """Navega a la SEQ següent."""
        if self.current_seq_idx < len(self.seq_list) - 1:
            self._save_current_peaks()
            self.current_seq_idx += 1
            self.seq_combo.current(self.current_seq_idx)
            self._load_seq_files()

    def _on_seq_selected(self, event):
        """Quan es selecciona una SEQ del combo box."""
        new_idx = self.seq_combo.current()
        if new_idx != self.current_seq_idx and new_idx >= 0:
            self._save_current_peaks()
            self.current_seq_idx = new_idx
            self._load_seq_files()

    def _prev_file(self):
        """Navega al fitxer anterior dins de la SEQ actual."""
        if self.current_file_idx > 0:
            self._save_current_peaks()
            self.current_file_idx -= 1
            self._load_current_file()

    def _next_file(self):
        """Navega al fitxer següent dins de la SEQ actual."""
        if self.current_file_idx < len(self.files) - 1:
            self._save_current_peaks()
            self.current_file_idx += 1
            self._load_current_file()

    def _skip_file(self):
        """Salta el fitxer actual (per errors o fitxers problemàtics)."""
        if not self.files:
            return

        info = self.file_info[self.current_file_idx]
        print(f"\nSALTANT FITXER: [{info['seq']}] {info['filename']}")

        # Si hi ha següent fitxer, anar-hi
        if self.current_file_idx < len(self.files) - 1:
            self.current_file_idx += 1
            self._load_current_file()
        # Si és l'últim de la SEQ i hi ha més SEQs
        elif self.current_seq_idx < len(self.seq_list) - 1:
            messagebox.showinfo("Últim fitxer", "Últim fitxer d'aquesta SEQ.\nPassant a la següent SEQ.")
            self._next_seq()
        else:
            messagebox.showinfo("Final", "Últim fitxer de l'última SEQ.")

    def _load_current_file(self):
        """Carrega el fitxer actual de la SEQ actual."""
        if not self.files:
            return

        filepath = self.files[self.current_file_idx]
        info = self.file_info[self.current_file_idx]

        self.file_label.config(text=f"[{info['seq']}] {info['filename']}")
        self.progress_label.config(text=f"Fitxer {self.current_file_idx + 1}/{len(self.files)}")

        print(f"\n{'='*70}")
        print(f"CARREGANT FITXER [{self.current_file_idx + 1}/{len(self.files)}]")
        print(f"SEQ: {info['seq']}")
        print(f"Fitxer: {info['filename']}")
        print(f"Path: {filepath}")
        print(f"{'='*70}")

        # Netejar dades anteriors
        self.current_t = None
        self.current_y = None
        self.current_peaks = []

        try:
            # STEP 1: Llegir Excel
            print(f"[1/4] Llegint Excel...")
            self.root.update_idletasks()  # Actualitzar GUI
            self.current_t, self.current_y = read_excel_doc(filepath)

            if self.current_t is None or len(self.current_t) == 0:
                print(f"  ERROR: Fitxer buit o sense dades vàlides")
                self._show_error(f"Error llegint fitxer:\n{info['filename']}\n\nFitxer buit o columnes no trobades")
                self._update_table()
                return

            print(f"  OK: Dades llegides: {len(self.current_t)} punts")
            print(f"  OK: t: [{self.current_t[0]:.2f} - {self.current_t[-1]:.2f}] min")
            print(f"  OK: y: [{np.min(self.current_y):.1f} - {np.max(self.current_y):.1f}] mAU")

            # STEP 2: Analitzar pics
            print(f"[2/4] Analitzant pics...")
            self.root.update_idletasks()
            self.current_peaks = analyze_peaks_for_diag(self.current_t, self.current_y)
            print(f"  OK: Pics detectats: {len(self.current_peaks)}")

            # STEP 3: Recuperar etiquetes guardades
            print(f"[3/4] Recuperant etiquetes...")
            for pk in self.current_peaks:
                for saved in self.all_results:
                    if (saved.get("seq") == info["seq"] and
                        saved.get("file") == info["filename"] and
                        saved.get("peak_num") == pk["num"]):
                        pk["label"] = saved.get("label", "")
                        pk["notes"] = saved.get("notes", "")
                        break

            # STEP 4: Dibuixar
            print(f"[4/4] Dibuixant gràfic...")
            self.root.update_idletasks()
            self._draw_plot()
            self._update_table()
            self._clear_edit_fields()

            print(f"OK: FITXER CARREGAT CORRECTAMENT")

        except Exception as e:
            print(f"\nERROR CRÍTIC carregant fitxer:")
            print(f"  Tipus: {type(e).__name__}")
            print(f"  Missatge: {str(e)}")
            import traceback
            traceback.print_exc()

            error_msg = f"Error carregant:\n{info['filename']}\n\n{type(e).__name__}: {str(e)}"
            self._show_error(error_msg)
            self.current_peaks = []
            self._update_table()

            messagebox.showerror("Error", error_msg + "\n\nUsa </> per saltar aquest fitxer")

    def _show_error(self, msg):
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        self.ax.text(0.5, 0.5, msg, ha="center", va="center", transform=self.ax.transAxes, fontsize=14, color="red")
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.fig.tight_layout()
        self.canvas.draw_idle()
        self.root.update_idletasks()

    def _draw_plot(self):
        print(f"  _draw_plot() cridada")

        # NETEJAR COMPLETAMENT: esborrar tot i recrear
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)

        if self.current_t is None or len(self.current_t) == 0:
            print(f"  -> Sense dades (current_t is None o buit)")
            self.ax.text(0.5, 0.5, "Sense dades", ha="center", va="center", transform=self.ax.transAxes)
            self.fig.tight_layout()
            self.canvas.draw_idle()
            self.root.update_idletasks()
            return

        mask = (self.current_t >= ANALYSIS_T_START) & (self.current_t <= ANALYSIS_T_END)
        t_plot = self.current_t[mask]
        y_plot = self.current_y[mask]

        print(f"  -> Filtre [{ANALYSIS_T_START}-{ANALYSIS_T_END}]: {len(t_plot)} punts de {len(self.current_t)}")

        if len(t_plot) == 0:
            print(f"  -> PROBLEMA: 0 punts després del filtre!")
            self.ax.text(0.5, 0.5, "Sense dades a la finestra", ha="center", va="center", transform=self.ax.transAxes)
            self.fig.tight_layout()
            self.canvas.draw_idle()
            self.root.update_idletasks()
            return

        self.ax.plot(t_plot, y_plot, 'b-', linewidth=0.8)

        # Marcar pics
        for pk in self.current_peaks:
            t_pk = pk["t_peak"]
            h_pk = pk["height"]
            num = pk["num"]

            # Prioritat: STOP > Batman > IRR > OK
            color = "green"
            marker = "o"
            label_suffix = ""

            if pk.get("is_stop", False):
                color = "#00BCD4"  # Cian/Turquesa
                marker = "D"  # Diamant
                label_suffix = " STOP"
            elif pk["batman_pct"] > 5:
                color = "red"
                marker = "v"
                label_suffix = " BAT"
            elif pk["irr_pct"] < 18:
                color = "orange"
                marker = "s"
                label_suffix = " IRR"

            self.ax.plot(t_pk, h_pk, marker=marker, color=color, markersize=8, zorder=5)
            self.ax.annotate(f"P{num}{label_suffix}", (t_pk, h_pk), textcoords="offset points",
                           xytext=(0, 10), ha="center", fontsize=9, fontweight="bold", color=color)

        self.ax.set_xlabel("Temps (min)")
        self.ax.set_ylabel("mAU")
        self.ax.set_xlim(ANALYSIS_T_START, ANALYSIS_T_END)

        y_min, y_max = float(np.min(y_plot)), float(np.max(y_plot))
        margin = (y_max - y_min) * 0.15
        self.ax.set_ylim(y_min - margin, y_max + margin)

        self.ax.grid(True, alpha=0.3)

        # Llegenda
        self.ax.plot([], [], 'go', label="Normal")
        self.ax.plot([], [], 'rv', label="Batman")
        self.ax.plot([], [], 'orange', marker='s', linestyle='', label="IRR<18%")
        self.ax.plot([], [], color='#00BCD4', marker='D', linestyle='', label="STOP (meseta)")
        self.ax.legend(loc="upper right", fontsize=8)

        self.fig.tight_layout()

        # FORÇAR REFRESH COMPLET amb draw_idle per evitar bloquejos
        self.canvas.draw_idle()
        self.root.update_idletasks()

    def _update_table(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

        for pk in self.current_peaks:
            # Formatació STOP
            stop_text = "-"
            if pk.get("is_stop", False):
                stop_text = f"OK ({pk.get('stop_cv', 0):.1f}%)"

            values = (
                f"P{pk['num']}",
                f"{pk['t_peak']:.2f}",
                f"{pk['height']:.1f}",
                f"{pk['as_33']:.2f}",
                f"{pk['as_66']:.2f}",
                f"{pk['as_div']:.2f}",
                f"{pk['batman_pct']:.1f}" if pk['batman_pct'] > 0 else "-",
                f"{pk['irr_pct']:.1f}",
                stop_text,
                pk['label'],
                pk['notes']
            )

            # Prioritat colors: STOP > Batman > IRR
            tag = ""
            if pk.get("is_stop", False):
                tag = "stop"
            elif pk["batman_pct"] > 5:
                tag = "batman"
            elif pk["irr_pct"] < 18:
                tag = "irr"

            self.tree.insert("", tk.END, values=values, tags=(tag,))

        self.tree.tag_configure("batman", background="#ffcccc")
        self.tree.tag_configure("irr", background="#fff3cd")
        self.tree.tag_configure("stop", background="#b2ebf2")  # Cian clar

    def _on_tree_select(self, event):
        selection = self.tree.selection()
        if not selection:
            return

        values = self.tree.item(selection[0], "values")
        if not values:
            return

        try:
            peak_num = int(values[0].replace("P", ""))
        except:
            return

        self.selected_peak_num = peak_num
        self.selected_label.config(text=f"P{peak_num}")

        for pk in self.current_peaks:
            if pk["num"] == peak_num:
                self.label_var.set(pk.get("label", ""))
                self.notes_entry.delete(0, tk.END)
                self.notes_entry.insert(0, pk.get("notes", ""))
                break

    def _clear_edit_fields(self):
        self.selected_peak_num = None
        self.selected_label.config(text="-")
        self.label_var.set("")
        self.notes_entry.delete(0, tk.END)

    def _apply_edit(self):
        if self.selected_peak_num is None:
            messagebox.showwarning("Avís", "Selecciona un pic primer")
            return

        new_label = self.label_var.get()
        new_notes = self.notes_entry.get().strip()

        info = self.file_info[self.current_file_idx]

        # Actualitzar pic actual
        for pk in self.current_peaks:
            if pk["num"] == self.selected_peak_num:
                pk["label"] = new_label
                pk["notes"] = new_notes

                # Guardar a resultats globals
                found = False
                for saved in self.all_results:
                    if (saved.get("seq") == info["seq"] and
                        saved.get("file") == info["filename"] and
                        saved.get("peak_num") == pk["num"]):
                        saved.update({
                            "label": new_label,
                            "notes": new_notes
                        })
                        found = True
                        break

                if not found:
                    self.all_results.append({
                        "seq": info["seq"],
                        "file": info["filename"],
                        "peak_num": pk["num"],
                        "t_peak": pk["t_peak"],
                        "height": pk["height"],
                        "as_33": pk["as_33"],
                        "as_66": pk["as_66"],
                        "as_div": pk["as_div"],
                        "batman_pct": pk["batman_pct"],
                        "irr_pct": pk["irr_pct"],
                        "label": new_label,
                        "notes": new_notes
                    })
                break

        self._update_table()
        print(f"Aplicat: P{self.selected_peak_num} -> {new_label} | {new_notes}")

    def _save_current_peaks(self):
        """Guarda tots els pics amb etiqueta a all_results."""
        if not self.files:
            return

        info = self.file_info[self.current_file_idx]

        for pk in self.current_peaks:
            if pk["label"] or pk["notes"]:
                found = False
                for saved in self.all_results:
                    if (saved.get("seq") == info["seq"] and
                        saved.get("file") == info["filename"] and
                        saved.get("peak_num") == pk["num"]):
                        saved["label"] = pk["label"]
                        saved["notes"] = pk["notes"]
                        found = True
                        break

                if not found:
                    self.all_results.append({
                        "seq": info["seq"],
                        "file": info["filename"],
                        "peak_num": pk["num"],
                        "t_peak": pk["t_peak"],
                        "height": pk["height"],
                        "as_33": pk["as_33"],
                        "as_66": pk["as_66"],
                        "as_div": pk["as_div"],
                        "batman_pct": pk["batman_pct"],
                        "irr_pct": pk["irr_pct"],
                        "label": pk["label"],
                        "notes": pk["notes"]
                    })

    def _save_csv(self):
        self._save_current_peaks()

        if not self.all_results:
            messagebox.showwarning("Avís", "No hi ha dades per guardar")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile=f"DOCtor_Diag_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )

        if not filepath:
            return

        # Aplicar etiqueta per defecte: buit = OK
        results_with_defaults = []
        for result in self.all_results:
            result_copy = result.copy()
            if not result_copy.get("label", "").strip():
                result_copy["label"] = "OK"
            results_with_defaults.append(result_copy)

        df = pd.DataFrame(results_with_defaults)
        df.to_csv(filepath, index=False)

        messagebox.showinfo("Guardat", f"CSV guardat: {os.path.basename(filepath)}\n{len(df)} pics etiquetats\n\nNota: Pics sense etiqueta = OK")

    def run(self):
        self.root.mainloop()


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    app = DiagApp()
    app.run()
