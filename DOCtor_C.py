# -*- coding: utf-8 -*-
"""
DOCtor_C v1.1 - Detecció d'Anomalies en Cromatogrames HPSEC (COLUMN)
====================================================================

Part de la SUITE HPSEC.
Per a anàlisi de BYPASS utilitzar DOCtor_BP.py

Detecció basada en:
- PEARSON entre rèpliques
- Batman estricte (pic-vall-pic al cim)
- IRR (irregularitats, smoothness < 18%)
- TimeOUT/STOP (saturació/mesetes) - MÈTODE PRINCIPAL via dt intervals
- Orelletes (pics secundaris propers)

v1.1: Eliminada detecció AMORPHOUS (precària, molts FP)
      - detect_amorphous() eliminada (~370 línies)
      - detect_amorphous_on_peak() eliminada (~70 línies)
      - validate_amorphous_with_replicates() eliminada (~190 línies)
      - calc_smoothness() eliminada (~55 línies)
      - Total: ~685 línies eliminades

PDF compacte: múltiples mostres per pàgina + taula resum.
"""

import os
import re
import sys
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import find_peaks
from scipy.stats import pearsonr
from scipy.optimize import curve_fit

from hpsec_utils import baseline_stats
from hpsec_core import (
    bigaussian, fit_bigaussian, detect_batman as detect_batman_top,
    detect_peak_anomaly, calc_top_smoothness,
    THRESH_R2_VALID, THRESH_R2_CHECK, ASYM_MIN, ASYM_MAX
)


# =============================================================================
# PARÀMETRES
# =============================================================================

# Soroll - sota aquest valor no s'analitza
NOISE_THRESHOLD = 20.0  # mAU

# Pearson - llindar per inspeccionar TimeOUT
PEARSON_INSPECT = 0.995

# TimeOUT - detecció per simetria i alçada
SYMMETRY_MIN = 0.7       # Ratio mínim àrea esq/dreta (1.0 = simètric)
HEIGHT_LOW_PCT = 20.0    # Alçada baixa si < 20% del màxim esperat

# Batman
BATMAN_MAX_SEP_MIN = 0.5
BATMAN_DROP_MIN = 0.02
BATMAN_DROP_MAX = 0.35
BATMAN_MIN_PROMINENCE = 0.05
BATMAN_MIN_HEIGHT_RANGE_PCT = 15
BATMAN_MIN_SIGMA = 3.0

# Orelletes - pics secundaris molt propers a un pic més alt
EAR_MAX_DISTANCE_MIN = 0.5      # Distància màxima al pic més alt (minuts)
EAR_MIN_HEIGHT_FRAC = 0.10      # Només pics per sobre del 10% de l'altura
EAR_MIN_PROMINENCE_PCT = 2.0    # Prominència mínima: 2% del pic principal

# =============================================================================
# REGLA ASIMETRIA - Detecció automàtica de pics problemàtics (Mode E)
# =============================================================================
# Regla: ((|As66 - As33| > 0.40) OR (As66 > 1.20)) AND (Pearson <= 0.997)
THRESH_PEARSON = 0.997          # Pearson <= 0.997 = necessari per marcar
THRESH_AS_DIVERGENCE = 0.40     # |As66 - As33| > 0.40 = divergència alta
THRESH_AS66 = 1.20              # As66 > 1.20 = asimetria alta

# =============================================================================
# ZONA HÚMICS - Limitar Batman i Orelletes a aquesta franja
# =============================================================================
HUMIC_ZONE_ENABLED = True       # Activar filtre de zona
HUMIC_ZONE_START = 18.0         # Inici zona húmics (minuts)
HUMIC_ZONE_END = 23.0           # Final zona húmics (minuts)

# =============================================================================
# TIMEOUT (SATURACIÓ) - Detecció de mesetes/plateaus
# =============================================================================
TIMEOUT_ENABLED = True          # Activar detecció TimeOUT
TIMEOUT_T_START = 15.0          # Inici finestra anàlisi (min)
TIMEOUT_T_END = 35.0            # Final finestra anàlisi (min)
TIMEOUT_FINE_RES = 0.1          # Resolució chunks (min)
TIMEOUT_MIN_DUR = 0.8           # Duració mínima meseta (min)
TIMEOUT_MAX_DUR = 3.5           # Duració màxima meseta (min)
TIMEOUT_PERCENTILE = 30         # Percentil per filtre inicial
TIMEOUT_EDGE_WINDOW = 0.3       # Finestra per calcular pendents bordes (min)
TIMEOUT_MIN_CONTRAST = 3.0      # Ratio mínim bord/meseta
TIMEOUT_MIN_EDGE_SLOPE = 6.0    # Pendent mínim als bordes
TIMEOUT_MIN_LEVEL_FACTOR = 1.30 # Senyal > baseline * 1.30
TIMEOUT_MAX_CV = 0.1            # CV màxim (%) - només parades reals (CV < 0.1%)

# NOTA: Detecció AMORPHOUS eliminada (v1.1) - era precària i generava molts FP
# Ara s'utilitza només detecció per timeout (dt intervals) que és més robusta


# =============================================================================
# AJUST GAUSSIÀ - Funcions per avaluar la forma dels pics
# =============================================================================
def gaussian(t, amplitude, mu, sigma, baseline):
    """Funció gaussiana per ajust de pics."""
    return amplitude * np.exp(-(t - mu)**2 / (2 * sigma**2)) + baseline


def fit_gaussian_to_peak(t, y, peak_idx, left_idx, right_idx, height_pct=33):
    """
    Ajusta una gaussiana al pic i retorna R².

    Paràmetres:
        t, y: arrays de temps i senyal
        peak_idx: índex del màxim del pic
        left_idx, right_idx: índexs dels límits del pic
        height_pct: percentatge d'altura mínim per l'ajust (evita solapaments)

    Returns:
        dict amb r2, paràmetres de l'ajust, i dades per visualització
    """
    # Extreure segment del pic
    t_seg = np.asarray(t[left_idx:right_idx+1], dtype=float)
    y_seg = np.asarray(y[left_idx:right_idx+1], dtype=float)

    if len(t_seg) < 5:
        return {"r2": 0.0, "error": "Segment massa curt"}

    # Calcular baseline i altura
    baseline_val = float(np.min(y_seg))
    peak_val = float(y[peak_idx])
    peak_height = peak_val - baseline_val

    if peak_height <= 0:
        return {"r2": 0.0, "error": "Altura zero o negativa"}

    # Truncar per sota del height_pct (evitar solapaments)
    threshold = baseline_val + peak_height * (height_pct / 100.0)

    # Trobar índexs on el senyal està per sobre del threshold
    mask = y_seg >= threshold
    if np.sum(mask) < 5:
        return {"r2": 0.0, "error": "Pocs punts sobre threshold"}

    # Usar només la part per sobre del threshold per l'ajust
    t_fit = t_seg[mask]
    y_fit = y_seg[mask]

    # Valors inicials per l'ajust
    amplitude_guess = peak_height
    mu_guess = float(t[peak_idx])
    # Estimar sigma a partir de l'amplada a mitja altura
    half_height = baseline_val + peak_height * 0.5
    above_half = t_seg[y_seg >= half_height]
    if len(above_half) >= 2:
        fwhm = float(above_half[-1] - above_half[0])
        sigma_guess = fwhm / 2.355  # FWHM = 2.355 * sigma
    else:
        sigma_guess = (t_seg[-1] - t_seg[0]) / 4

    sigma_guess = max(0.01, sigma_guess)  # Evitar sigma zero

    try:
        # Ajustar gaussiana
        popt, pcov = curve_fit(
            gaussian, t_fit, y_fit,
            p0=[amplitude_guess, mu_guess, sigma_guess, threshold],
            bounds=(
                [0, t_fit[0], 0.001, 0],  # mínims
                [np.inf, t_fit[-1], np.inf, np.inf]  # màxims
            ),
            maxfev=3000
        )

        # Calcular R² sobre el segment truncat
        y_pred = gaussian(t_fit, *popt)
        ss_res = np.sum((y_fit - y_pred)**2)
        ss_tot = np.sum((y_fit - np.mean(y_fit))**2)

        if ss_tot > 0:
            r2 = 1 - (ss_res / ss_tot)
        else:
            r2 = 0.0

        # Generar corba ajustada per visualització (tot el segment)
        t_plot = np.linspace(t_seg[0], t_seg[-1], 100)
        y_plot = gaussian(t_plot, *popt)

        return {
            "r2": max(0.0, r2),
            "amplitude": popt[0],
            "mu": popt[1],
            "sigma": popt[2],
            "baseline_fit": popt[3],
            "t_fit": t_plot,
            "y_fit": y_plot,
            "threshold": threshold,
            "height_pct": height_pct
        }

    except Exception as e:
        return {"r2": 0.0, "error": str(e)}


# =============================================================================
# LECTURA EXCEL
# =============================================================================
def read_excel_doc(file_path):
    """Llegeix DOC sheet, retorna t, y, sample_type."""
    try:
        df = pd.read_excel(file_path, sheet_name="DOC")
    except:
        df = pd.read_excel(file_path, sheet_name=0)

    df.columns = [str(c).strip().lower() for c in df.columns]

    tcol = next((c for c in df.columns if "time" in c), None)
    ycol = next((c for c in df.columns if c != tcol and ("doc" in c or "toc" in c)), None)

    if not tcol or not ycol:
        return None, None, "UNKNOWN"

    # Detectar tipus
    sample_type = "COLUMN"
    id_cols = [c for c in df.columns if "id" in c]
    if id_cols:
        blob = " ".join(df[id_cols[0]].dropna().astype(str).head(10)).upper()
        if "BP" in blob:
            sample_type = "BP"

    t = pd.to_numeric(df[tcol], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[ycol], errors="coerce").to_numpy(dtype=float)

    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]

    if t.size < 10:
        return None, None, sample_type

    return t[np.argsort(t)], y[np.argsort(t)], sample_type


# =============================================================================
# PEARSON ENTRE RÈPLIQUES
# =============================================================================
def calc_pearson_replicates(t1, y1, t2, y2, window=None):
    """Calcula Pearson entre dues rèpliques (interpolant si cal)."""
    if t1 is None or t2 is None or len(t1) < 10 or len(t2) < 10:
        return np.nan, np.nan

    # Rang comú
    t_min = max(t1.min(), t2.min())
    t_max = min(t1.max(), t2.max())

    if window:
        t_min = max(t_min, window[0])
        t_max = min(t_max, window[1])

    if t_max <= t_min:
        return np.nan, np.nan

    # Interpolar a punts comuns
    t_common = np.linspace(t_min, t_max, 500)
    y1_interp = np.interp(t_common, t1, y1)
    y2_interp = np.interp(t_common, t2, y2)

    try:
        r, _ = pearsonr(y1_interp, y2_interp)
        return r, t_max - t_min
    except:
        return np.nan, np.nan


# =============================================================================
# DETECCIÓ TIMEOUT (simetria àrea + factor asimetria) - PIC PRINCIPAL
# =============================================================================
MAX_PEAKS_TO_ANALYZE = 5  # Analitzar fins a 5 pics principals

# Altures per calcular As (factor asimetria)
AS_HEIGHTS = [33, 66]  # Percentatges de l'altura del pic

# Monotonia - % mínim de punts que han de ser monòtons
MONOTONIC_MIN_PCT = 85.0  # 85% dels punts han de ser monòtons


def calc_monotonicity(y, left_idx, peak_idx, right_idx, noise_threshold=0.0):
    """
    Calcula el % de monotonia a cada costat del pic, ajustat per soroll.

    - Costat esquerre: hauria de ser monòton creixent (y[i+1] >= y[i])
    - Costat dret: hauria de ser monòton decreixent (y[i+1] <= y[i])
    - Només es compta com a violació si supera el llindar de soroll

    Args:
        y: senyal
        left_idx, peak_idx, right_idx: índexs del pic
        noise_threshold: només violacions > aquest valor compten (típic: 2*baseline_noise)

    Returns:
        tuple: (mono_left_pct, mono_right_pct, mono_total_pct)
        - 100% = perfectament monòton
        - <85% = possibles problemes (orelletes, etc.)
    """
    # Costat esquerre (ha de créixer)
    left_segment = y[left_idx:peak_idx+1]
    if len(left_segment) > 1:
        diffs_left = np.diff(left_segment)
        # Violació: decreix més que el soroll (diff < -noise_threshold)
        n_ok = np.sum(diffs_left >= -noise_threshold)
        mono_left_pct = 100.0 * n_ok / len(diffs_left)
    else:
        mono_left_pct = 100.0

    # Costat dret (ha de decréixer)
    right_segment = y[peak_idx:right_idx+1]
    if len(right_segment) > 1:
        diffs_right = np.diff(right_segment)
        # Violació: creix més que el soroll (diff > noise_threshold)
        n_ok = np.sum(diffs_right <= noise_threshold)
        mono_right_pct = 100.0 * n_ok / len(diffs_right)
    else:
        mono_right_pct = 100.0

    # Total ponderat per nombre de punts
    n_left = len(left_segment) - 1
    n_right = len(right_segment) - 1
    total_points = n_left + n_right

    if total_points > 0:
        mono_total_pct = (n_left * mono_left_pct + n_right * mono_right_pct) / total_points
    else:
        mono_total_pct = 100.0

    return mono_left_pct, mono_right_pct, mono_total_pct


# NOTA: calc_smoothness() eliminada (v1.1) - no s'utilitza després de treure amorphous


def detect_peak_problem(as_33, as_66, as_divergence=None, pearson=1.0):
    """
    Detecta si un pic és problemàtic usant asimetria i Pearson.

    Regla: ((|As66 - As33| > 0.40) OR (As66 > 1.20)) AND (Pearson <= 0.997)

    Un pic és problemàtic si:
    1. Divergència alta (|As66-As33| > 0.40) O As66 alt (> 1.20)
    2. I les rèpliques no correlacionen bé (Pearson ≤ 0.997)

    Returns:
        tuple: (is_problem, reason)
    """
    if as_divergence is None:
        as_divergence = abs(as_66 - as_33)

    # Primer: Pearson ha de ser baix (<=0.997)
    if pearson > THRESH_PEARSON:
        return False, ""  # Rèpliques correlacionen bé, no marcar

    # Després: Δ alta O As66 alt
    crit_div = as_divergence > THRESH_AS_DIVERGENCE
    crit_as66 = as_66 > THRESH_AS66

    if crit_div or crit_as66:
        reasons = []
        if crit_div:
            reasons.append(f"D={as_divergence:.2f}")
        if crit_as66:
            reasons.append(f"As66={as_66:.2f}")
        reasons.append(f"Pearson={pearson:.4f}")
        return True, " + ".join(reasons)

    return False, ""


def select_best_replica(rep1, rep2):
    """
    Selecciona la millor rèplica basant-se en:
    1. Batman prioritari - si una té Batman, l'altra és millor
    2. Problemes detectats (Regla D)
    3. Alçada (la de més alçada és millor)
    4. Soroll (menys soroll és millor)

    Args:
        rep1, rep2: diccionaris amb info de rèplica (peaks, batman, baseline_noise, etc.)

    Returns:
        dict: {
            "best": "R1" o "R2",
            "reason": explicació,
            "r1_dominated": bool,  # R1 té problemes
            "r2_dominated": bool,  # R2 té problemes
            "warning": str o None
        }
    """
    result = {
        "best": None,
        "reason": "",
        "r1_dominated": False,
        "r2_dominated": False,
        "warning": None
    }

    # Si només hi ha una rèplica
    if rep1 is None and rep2 is None:
        result["warning"] = "CAP RÈPLICA DISPONIBLE"
        return result

    if rep1 is None:
        result["best"] = "R2"
        result["reason"] = "Única rèplica disponible"
        # Comprovar si té problemes
        if rep2.get("batman"):
            result["warning"] = "WARNING: Única rèplica amb BATMAN"
            result["r2_dominated"] = True
        elif any(pk.get("is_problem") for pk in rep2.get("peaks", [])):
            result["warning"] = "WARNING: Única rèplica amb pics problemàtics"
            result["r2_dominated"] = True
        return result

    if rep2 is None:
        result["best"] = "R1"
        result["reason"] = "Única rèplica disponible"
        if rep1.get("batman"):
            result["warning"] = "WARNING: Única rèplica amb BATMAN"
            result["r1_dominated"] = True
        elif any(pk.get("is_problem") for pk in rep1.get("peaks", [])):
            result["warning"] = "WARNING: Única rèplica amb pics problemàtics"
            result["r1_dominated"] = True
        return result

    # Ambdues rèpliques existeixen
    r1_batman = rep1.get("batman") is not None
    r2_batman = rep2.get("batman") is not None

    # 1. BATMAN PRIORITARI - si una té Batman, triar l'altra
    if r1_batman and not r2_batman:
        result["best"] = "R2"
        result["reason"] = "R1 té BATMAN"
        result["r1_dominated"] = True
        return result

    if r2_batman and not r1_batman:
        result["best"] = "R1"
        result["reason"] = "R2 té BATMAN"
        result["r2_dominated"] = True
        return result

    if r1_batman and r2_batman:
        # Ambdues tenen Batman - triar per alçada
        r1_height = max((pk.get("height", 0) for pk in rep1.get("peaks", [])), default=0)
        r2_height = max((pk.get("height", 0) for pk in rep2.get("peaks", [])), default=0)
        if r1_height >= r2_height:
            result["best"] = "R1"
        else:
            result["best"] = "R2"
        result["reason"] = "Ambdues BATMAN, triat per alçada"
        result["r1_dominated"] = True
        result["r2_dominated"] = True
        result["warning"] = "WARNING: Ambdues rèpliques amb BATMAN"
        return result

    # 2. Problemes detectats (Regla D)
    r1_problems = sum(1 for pk in rep1.get("peaks", []) if pk.get("is_problem"))
    r2_problems = sum(1 for pk in rep2.get("peaks", []) if pk.get("is_problem"))

    result["r1_dominated"] = r1_problems > 0
    result["r2_dominated"] = r2_problems > 0

    if r1_problems > 0 and r2_problems == 0:
        result["best"] = "R2"
        result["reason"] = f"R1 té {r1_problems} pics problemàtics"
        return result

    if r2_problems > 0 and r1_problems == 0:
        result["best"] = "R1"
        result["reason"] = f"R2 té {r2_problems} pics problemàtics"
        return result

    # 3. Ambdues amb problemes o cap - triar per alçada
    r1_height = max((pk.get("height", 0) for pk in rep1.get("peaks", [])), default=0)
    r2_height = max((pk.get("height", 0) for pk in rep2.get("peaks", [])), default=0)

    if r1_problems > 0 and r2_problems > 0:
        # Ambdues problemàtiques - triar la de més alçada
        if r1_height >= r2_height:
            result["best"] = "R1"
        else:
            result["best"] = "R2"
        result["reason"] = "Ambdues problemàtiques, triat per alçada"
        result["warning"] = "WARNING: Ambdues rèpliques amb problemes"
        return result

    # 4. Cap problema - triar per soroll (menys soroll = millor)
    r1_noise = rep1.get("baseline_noise", 0)
    r2_noise = rep2.get("baseline_noise", 0)

    if r1_noise <= r2_noise:
        result["best"] = "R1"
        result["reason"] = f"Menys soroll (R1:{r1_noise:.2f} vs R2:{r2_noise:.2f})"
    else:
        result["best"] = "R2"
        result["reason"] = f"Menys soroll (R2:{r2_noise:.2f} vs R1:{r1_noise:.2f})"

    return result


def calc_asymmetry_factor(t, y, peak_idx, baseline, height_pct=10):
    """
    Calcula el Factor d'Asimetria (As) a un % de l'altura del pic.

    As = b/a on:
    - a = distància del pic al punt esquerre (al height_pct% de l'altura)
    - b = distància del pic al punt dret

    Returns:
        As (float): 1.0 = simètric, >1 = tailing (cua dreta), <1 = fronting
    """
    peak_height = float(y[peak_idx])
    peak_time = float(t[peak_idx])

    # Altura al % especificat
    threshold_height = baseline + (peak_height - baseline) * (height_pct / 100.0)

    # Buscar punt esquerre on y creua threshold_height
    a = 0.0
    for i in range(peak_idx, -1, -1):
        if y[i] <= threshold_height:
            # Interpolar per trobar el punt exacte
            if i < peak_idx:
                y1, y2 = float(y[i]), float(y[i+1])
                t1, t2 = float(t[i]), float(t[i+1])
                if y2 != y1:
                    t_cross = t1 + (threshold_height - y1) * (t2 - t1) / (y2 - y1)
                    a = peak_time - t_cross
            break

    # Buscar punt dret on y creua threshold_height
    b = 0.0
    for i in range(peak_idx, len(y)):
        if y[i] <= threshold_height:
            # Interpolar per trobar el punt exacte
            if i > peak_idx:
                y1, y2 = float(y[i-1]), float(y[i])
                t1, t2 = float(t[i-1]), float(t[i])
                if y2 != y1:
                    t_cross = t1 + (threshold_height - y1) * (t2 - t1) / (y2 - y1)
                    b = t_cross - peak_time
            break

    # Calcular As
    if a > 0:
        return b / a
    return 1.0  # Si no es pot calcular, assumir simètric


def analyze_peaks(t, y):
    """
    Analitza fins a 5 pics principals amb ajust BI-GAUSSIÀ.
    Detecta Batman (valls al cim) per cada pic.

    Returns:
        dict amb info global, R² bi-gaussià, i llista de pics analitzats
    """
    from scipy.integrate import trapezoid

    if t is None or len(t) < 20:
        return None

    t = np.asarray(t, float)
    y = np.asarray(y, float)

    baseline = float(np.percentile(y, 10))
    max_signal = float(np.max(y))

    # Calcular soroll de baseline (primers 10% dels punts)
    baseline_end = max(10, len(y) // 10)
    baseline_noise = float(np.std(y[:baseline_end]))
    noise_threshold = 2.0 * baseline_noise  # Llindar per monotonia

    if max_signal < NOISE_THRESHOLD:
        return {"status": "LOW_SIGNAL", "max_mau": max_signal, "peaks": [], "baseline_noise": baseline_noise}

    # Trobar pics
    min_prom = (max_signal - baseline) * 0.05
    peaks, props = find_peaks(y, prominence=min_prom, width=2)

    if len(peaks) == 0:
        return {"status": "OK", "peaks": [], "n_peaks": 0, "n_with_issues": 0, "baseline_noise": baseline_noise}

    # Agafar fins a 5 pics més prominents, després ordenar per temps
    sorted_by_prom = np.argsort(props["prominences"])[::-1]
    top_n = min(MAX_PEAKS_TO_ANALYZE, len(peaks))
    top_indices_by_prom = sorted_by_prom[:top_n]

    # Ordenar per temps d'aparició (no per prominència)
    top_indices = sorted(top_indices_by_prom, key=lambda i: t[peaks[i]])

    # Detectar STOP/TimeOUT a nivell global (abans d'analitzar pics)
    global_timeouts = detect_timeout(t, y) if TIMEOUT_ENABLED else []

    analyzed_peaks = []
    any_timeout = False

    for rank, idx in enumerate(top_indices):
        pk_idx = int(peaks[idx])
        left_base = int(props["left_bases"][idx])
        right_base = int(props["right_bases"][idx])
        prominence = float(props["prominences"][idx])

        peak_height = float(y[pk_idx]) - baseline
        t_peak = float(t[pk_idx])
        t_left = float(t[left_base])
        t_right = float(t[right_base])

        # === DETECCIÓ HÍBRIDA: VALLS + SMOOTHNESS ===
        # Extreure segment del pic
        t_seg = t[left_base:right_base+1]
        y_seg = y[left_base:right_base+1]

        # Detectar anomalies (Batman + irregularitats)
        # NOMÉS A ZONA HÚMICA: Batman/irregularitats són típiques de substàncies húmiques
        if HUMIC_ZONE_ENABLED:
            in_humic_zone = (t_peak >= HUMIC_ZONE_START) and (t_peak <= HUMIC_ZONE_END)
        else:
            in_humic_zone = True  # Si zona desactivada, analitzar tots els pics

        if in_humic_zone:
            # Paràmetres ajustats per reduir FP: top_pct=0.15, min_valley_depth=0.05, smoothness=18%
            anomaly = detect_peak_anomaly(t_seg, y_seg, top_pct=0.15, min_valley_depth=0.05, smoothness_threshold=18.0)
            is_anomaly = anomaly.get("is_anomaly", False)
            anomaly_type = anomaly.get("anomaly_type", "OK")
            smoothness = anomaly.get("smoothness", 100.0)
            is_batman = anomaly.get("is_batman", False)
            is_irregular = anomaly.get("is_irregular", False)
        else:
            # Pic fora de zona húmica: NO analitzar Batman
            anomaly = {"is_anomaly": False, "anomaly_type": "OK", "smoothness": 100.0, "is_batman": False, "is_irregular": False}
            is_anomaly = False
            anomaly_type = "OK"
            smoothness = 100.0
            is_batman = False
            is_irregular = False

        # NOTA: Detecció AMORPHOUS eliminada (v1.1) - s'utilitza timeout per dt intervals

        # Comprovar si el pic conté o solapa amb mesetes STOP
        has_stop = False
        stop_cv = 100.0
        stop_info = None
        for timeout in global_timeouts:
            # Comprovar solapament: meseta dins del pic?
            if timeout["t_start"] >= t_left and timeout["t_end"] <= t_right:
                has_stop = True
                stop_cv = min(stop_cv, timeout.get("cv", 100.0))
                stop_info = timeout
                break
            # També comprovar solapament parcial
            elif (timeout["t_start"] <= t_right and timeout["t_end"] >= t_left):
                has_stop = True
                stop_cv = min(stop_cv, timeout.get("cv", 100.0))
                stop_info = timeout
                break

        # === SIMETRIA (àrees) - mantingut per compatibilitat ===
        left_t = t[left_base:pk_idx+1]
        left_y = y[left_base:pk_idx+1]
        area_left = trapezoid(left_y, left_t) if len(left_t) > 1 else 0

        right_t = t[pk_idx:right_base+1]
        right_y = y[pk_idx:right_base+1]
        area_right = trapezoid(right_y, right_t) if len(right_t) > 1 else 0

        if area_left > 0 and area_right > 0:
            sym_area = min(area_left, area_right) / max(area_left, area_right)
        else:
            sym_area = 0.0

        # Factor d'Asimetria (As) - mantingut per compatibilitat
        as_33 = calc_asymmetry_factor(t, y, pk_idx, baseline, height_pct=33)
        as_66 = calc_asymmetry_factor(t, y, pk_idx, baseline, height_pct=66)
        as_divergence = abs(as_33 - as_66)

        # Monotonia (ajustada per soroll)
        mono_left, mono_right, mono_total = calc_monotonicity(y, left_base, pk_idx, right_base, noise_threshold)

        # NOTA: calc_smoothness eliminada (v1.1) - no necessària sense amorphous

        # Àrea total del pic (per ponderar)
        total_area = area_left + area_right

        # Indicadors de qualitat
        flags = []

        # HÍBRID: Anomalies detectades (Batman, Irregular, STOP)
        if is_anomaly:
            if is_batman:
                n_valleys = anomaly.get("n_valleys", 0)
                max_depth = anomaly.get("max_valley_depth", 0)
                flags.append(f"BAT({n_valleys}v,{max_depth*100:.0f}%)")
            if is_irregular:
                flags.append(f"IRR({smoothness:.0f}%)")

        # NOTA: AMORF flag eliminat (v1.1) - detecció precària

        if has_stop:
            flags.append(f"STOP(CV={stop_cv:.1f}%)")

        if mono_total < MONOTONIC_MIN_PCT:
            flags.append("no-mono")

        # Té algun flag?
        has_issues = is_anomaly or has_stop or mono_total < MONOTONIC_MIN_PCT
        if has_issues:
            any_timeout = True

        analyzed_peaks.append({
            "rank": rank + 1,  # 1, 2, ...
            "t_peak": t_peak,
            "peak_idx": pk_idx,
            "left_base": left_base,
            "right_base": right_base,
            "height": peak_height,
            "prominence": prominence,
            "t_left": t_left,
            "t_right": t_right,
            "area_left": area_left,
            "area_right": area_right,
            "total_area": total_area,
            "sym_area": sym_area,
            "as_33": as_33,
            "as_66": as_66,
            "as_divergence": as_divergence,
            # Detecció híbrida (valls + smoothness)
            "anomaly": anomaly,
            "is_anomaly": is_anomaly,
            "anomaly_type": anomaly_type,
            "smoothness": smoothness,
            "is_batman": is_batman,
            "is_irregular": is_irregular,
            # NOTA: Camps amorphous eliminats (v1.1)
            # STOP/TimeOUT (mesetes en el pic)
            "has_stop": has_stop,
            "stop_cv": stop_cv,
            "stop_info": stop_info,
            # Monotonia
            "mono_left": mono_left,
            "mono_right": mono_right,
            "mono_total": mono_total,
            "flags": flags,
            "has_issues": has_issues
        })

    # Comptar pics amb problemes
    n_with_issues = sum(1 for p in analyzed_peaks if p["has_issues"])
    n_batman = sum(1 for p in analyzed_peaks if p.get("is_batman"))
    n_with_stop = sum(1 for p in analyzed_peaks if p.get("has_stop"))

    return {
        "status": "CHECK" if any_timeout else "OK",
        "peaks": analyzed_peaks,
        "n_peaks": len(analyzed_peaks),
        "n_with_issues": n_with_issues,
        "n_batman": n_batman,
        "n_with_stop": n_with_stop,
        "baseline_noise": baseline_noise,
        "global_timeouts": global_timeouts  # Per debug/visualització
    }


def analyze_main_peak(t, y):
    """Wrapper per compatibilitat."""
    return analyze_peaks(t, y)


# =============================================================================
# DETECCIÓ BATMAN (VALLS AL CIM DEL PIC)
# =============================================================================
# NOTA: detect_amorphous_on_peak() eliminada (v1.1) - precària i molts FP
# Ara s'utilitza només detecció per timeout (dt intervals)

def detect_batman_on_peak(t, y, peak_idx, left_idx, right_idx, top_pct=0.15, min_valley_depth=0.05):
    """
    Detecta Batman (valls al cim) en un pic individual.

    Utilitza detect_batman_top de hpsec_core que busca LOCAL MINIMA
    a la part superior del pic (artefactes del detector).

    Parameters:
        t, y: arrays de temps i senyal
        peak_idx: índex del màxim del pic
        left_idx, right_idx: límits del pic
        top_pct: fracció del cim a analitzar (0.15 = top 15%)
        min_valley_depth: profunditat mínima de vall (0.05 = 5%)

    Returns:
        dict amb info Batman o None si no es detecta
    """
    if t is None or len(t) < 10:
        return None

    # Extreure segment del pic
    t_seg = np.asarray(t[left_idx:right_idx+1], dtype=float)
    y_seg = np.asarray(y[left_idx:right_idx+1], dtype=float)

    if len(t_seg) < 10:
        return None

    # Detectar Batman al CIM del pic (paràmetres estrictes per reduir FP)
    result = detect_batman_top(t_seg, y_seg, top_pct=top_pct, min_valley_depth=min_valley_depth)

    if result and result.get("is_batman"):
        # Afegir info de context
        result["t_peak"] = float(t[peak_idx])
        result["peak_height"] = float(y[peak_idx]) - float(np.min(y_seg))
        return result

    return None


def detect_batman_signal(t, y):
    """
    [LEGACY] Wrapper per compatibilitat - ara analitza tots els pics.
    Retorna el primer Batman trobat o None.
    """
    if t is None or len(t) < 10:
        return None

    t = np.asarray(t, float)
    y = np.asarray(y, float)

    # Filtrar a zona dels húmics si està activat
    if HUMIC_ZONE_ENABLED:
        mask = (t >= HUMIC_ZONE_START) & (t <= HUMIC_ZONE_END)
        if not np.any(mask):
            return None
        t_zone = t[mask]
        y_zone = y[mask]
        # Ajustar índexs
        offset = np.where(mask)[0][0]
    else:
        t_zone = t
        y_zone = y
        offset = 0

    # Detectar pics a la zona
    baseline = float(np.percentile(y_zone, 10))
    max_signal = float(np.max(y_zone))
    if max_signal < NOISE_THRESHOLD:
        return None

    min_prom = (max_signal - baseline) * 0.05
    peaks, props = find_peaks(y_zone, prominence=min_prom, width=2)

    if len(peaks) == 0:
        return None

    # Analitzar cada pic per Batman
    for i, pk_idx in enumerate(peaks):
        left_base = int(props["left_bases"][i])
        right_base = int(props["right_bases"][i])

        batman = detect_batman_on_peak(t_zone, y_zone, pk_idx, left_base, right_base)
        if batman:
            return batman

    return None


# =============================================================================
# DETECCIÓ ORELLETES (pics secundaris propers a un pic més alt)
# =============================================================================
def detect_ears(t, y):
    """
    Detecta orelletes = pics secundaris molt propers a un pic MÉS ALT.
    Si HUMIC_ZONE_ENABLED, només busca a la zona dels húmics.

    Un pic és "orelleta" si:
    1. Hi ha un altre pic més alt a menys de EAR_MAX_DISTANCE_MIN minuts
    2. Té prominència suficient (no és soroll)
    3. Està per sobre del nivell mínim (% de l'altura màxima)

    Returns:
        list of dicts amb info de cada orelleta detectada
    """
    if t is None or len(t) < 20:
        return []

    t = np.asarray(t, float)
    y = np.asarray(y, float)

    # Filtrar a zona dels húmics si està activat
    if HUMIC_ZONE_ENABLED:
        mask = (t >= HUMIC_ZONE_START) & (t <= HUMIC_ZONE_END)
        if not np.any(mask):
            return []  # No hi ha dades a la zona húmica
        t = t[mask]
        y = y[mask]

    # Baseline i màxim
    baseline = float(np.percentile(y, 10))
    y_max = float(np.max(y))
    signal_range = y_max - baseline

    if signal_range <= 0:
        return []

    # Prominència mínima per considerar un pic (% del rang)
    min_prom = EAR_MIN_PROMINENCE_PCT / 100.0 * signal_range

    # Trobar TOTS els pics amb prominència significativa
    peaks, props = find_peaks(y, prominence=min_prom, width=2)

    if len(peaks) < 2:
        return []  # Cal almenys 2 pics per tenir orelletes

    # Nivell mínim d'altura
    min_level = baseline + EAR_MIN_HEIGHT_FRAC * signal_range

    ears = []

    # Per cada pic, mirar si té un pic més alt a prop
    for i, pk_idx in enumerate(peaks):
        pk_height = float(y[pk_idx])
        pk_time = float(t[pk_idx])

        if pk_height < min_level:
            continue  # Massa baix per ser rellevant

        # Buscar pics més alts dins la distància màxima
        has_higher_neighbor = False
        higher_peak_time = None

        for j, other_idx in enumerate(peaks):
            if i == j:
                continue

            other_height = float(y[other_idx])
            other_time = float(t[other_idx])
            distance = abs(pk_time - other_time)

            # Si hi ha un pic més alt a menys de la distància màxima
            if other_height > pk_height and distance <= EAR_MAX_DISTANCE_MIN:
                has_higher_neighbor = True
                higher_peak_time = other_time
                break

        if has_higher_neighbor:
            # Aquest pic és una orelleta!
            prom = float(props["prominences"][i])
            prom_pct = 100.0 * prom / signal_range

            ears.append({
                "t": pk_time,
                "idx": int(pk_idx),
                "height": pk_height,
                "prominence": prom,
                "prom_pct": prom_pct,
                "higher_peak_t": higher_peak_time,
                "distance": abs(pk_time - higher_peak_time)
            })

    # Ordenar per altura (més altes primer)
    ears.sort(key=lambda e: -e["height"])

    return ears


# =============================================================================
# NOTA: detect_amorphous() eliminada (v1.1) - ~370 línies
# Era precària i generava molts falsos positius
# Ara s'utilitza només detecció per timeout (dt intervals) que és més robusta
# =============================================================================

# =============================================================================
# DETECCIÓ TIMEOUT (SATURACIÓ/MESETES)
# =============================================================================
def detect_timeout(t, y):
    """
    Detecta events TimeOUT (mesetes/plateaus) que indiquen saturació del detector.

    Busca regions planes amb:
    1. Duració entre TIMEOUT_MIN_DUR i TIMEOUT_MAX_DUR
    2. Contrast alt entre bordes i meseta (ratio > TIMEOUT_MIN_CONTRAST)
    3. Pendents pronunciades als bordes (> TIMEOUT_MIN_EDGE_SLOPE)
    4. Nivell per sobre de baseline * TIMEOUT_MIN_LEVEL_FACTOR

    Returns:
        list of dicts amb info de cada TimeOUT detectat, o llista buida
    """
    if not TIMEOUT_ENABLED:
        return []

    if t is None or len(t) < 20:
        return []

    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    # Filtrar a finestra d'anàlisi
    mask = (t >= TIMEOUT_T_START) & (t <= TIMEOUT_T_END)
    if not np.any(mask):
        return []

    t_win = t[mask]
    y_win = y[mask]

    if len(t_win) < 20:
        return []

    # Baseline (percentil 5 per robustesa)
    baseline_val = float(np.percentile(y_win, 5))
    min_required_y = baseline_val * TIMEOUT_MIN_LEVEL_FACTOR

    # Helper per calcular pendent
    def get_slope(t_seg, y_seg):
        if len(t_seg) < 2:
            return 0.0
        slope, _ = np.polyfit(t_seg, y_seg, 1)
        return abs(slope)

    # Crear chunks de resolució fina
    num_steps = int((t_win[-1] - t_win[0]) / TIMEOUT_FINE_RES)
    chunks = []

    for i in range(num_steps):
        t_center = t_win[0] + i * TIMEOUT_FINE_RES
        mask_c = (t_win >= t_center) & (t_win < t_center + TIMEOUT_FINE_RES)
        if np.sum(mask_c) < 2:
            continue
        chunks.append({
            'idx': i,
            'slope_abs': get_slope(t_win[mask_c], y_win[mask_c]),
            'indices': np.where(mask_c)[0],
            't_start': float(t_win[mask_c][0]),
            't_end': float(t_win[mask_c][-1])
        })

    if not chunks:
        return []

    # Trobar illes de baixa pendent (mesetes)
    all_slopes = [c['slope_abs'] for c in chunks]
    thresh = np.percentile(all_slopes, TIMEOUT_PERCENTILE)

    islands = []
    current_island = []
    for c in chunks:
        if c['slope_abs'] <= thresh:
            current_island.append(c)
        else:
            if current_island:
                islands.append(current_island)
            current_island = []
    if current_island:
        islands.append(current_island)

    # Avaluar candidats
    candidates = []
    for isl in islands:
        first, last = isl[0], isl[-1]
        t_s, t_e = first['t_start'], last['t_end']
        dur = t_e - t_s

        # Filtre duració
        if not (TIMEOUT_MIN_DUR <= dur <= TIMEOUT_MAX_DUR):
            continue

        # Mètriques de la meseta
        idx_s = first['indices'][0]
        idx_e = last['indices'][-1]
        y_plat = y_win[idx_s:idx_e+1]
        t_plat = t_win[idx_s:idx_e+1]

        if len(y_plat) < 2:
            continue

        y_mean = float(np.mean(y_plat))
        slope_plat = get_slope(t_plat, y_plat)

        # Pendents dels bordes
        mask_pre = (t_win >= t_s - TIMEOUT_EDGE_WINDOW) & (t_win < t_s)
        mask_post = (t_win > t_e) & (t_win <= t_e + TIMEOUT_EDGE_WINDOW)

        slope_pre = get_slope(t_win[mask_pre], y_win[mask_pre]) if np.any(mask_pre) else 0
        slope_post = get_slope(t_win[mask_post], y_win[mask_post]) if np.any(mask_post) else 0
        max_edge = max(slope_pre, slope_post)

        # Ratio contrast
        ratio = max_edge / (slope_plat + 1e-6)

        # CV (Coeficient de Variació) = std / mean
        # CV baix = valors molt iguals = PLA
        # CV alt = valors variables
        y_std = float(np.std(y_plat))
        cv_plat = (y_std / y_mean * 100) if y_mean > 1e-6 else 0  # en %

        # Filtres
        passes_edge = max_edge > TIMEOUT_MIN_EDGE_SLOPE
        passes_level = y_mean > min_required_y
        passes_contrast = ratio > TIMEOUT_MIN_CONTRAST
        passes_cv = cv_plat < TIMEOUT_MAX_CV  # Només parades reals (CV molt baix)

        if passes_edge and passes_level and passes_contrast and passes_cv:
            candidates.append({
                't_start': t_s,
                't_end': t_e,
                'duration': dur,
                'y_mean': y_mean,
                'y_std': y_std,
                'cv': cv_plat,  # CV en % (baix = pla)
                'slope_plat': slope_plat,
                'max_edge': max_edge,
                'ratio': ratio,
                'baseline': baseline_val
            })

    # Ordenar per ratio (millor contrast primer)
    candidates.sort(key=lambda x: -x['ratio'])

    return candidates


# =============================================================================
# NOTA: validate_amorphous_with_replicates() eliminada (v1.1) - ~190 línies
# Era necessària per amorphous detection, que ara està eliminada
# =============================================================================

# =============================================================================
# AGRUPACIÓ RÈPLIQUES
# =============================================================================
_REP_PAT = re.compile(r"(?i)(?:_|\b)(rep(?:lica)?|r)\s*([12])(?:\b|_)")

def sample_key_from_filename(fname):
    base = os.path.splitext(os.path.basename(fname))[0]
    rep_id = None
    m = _REP_PAT.search(base)
    if m:
        rep_id = m.group(2)
        base = _REP_PAT.sub("", base)
    return re.sub(r"__+", "_", base).strip("_ "), rep_id


# =============================================================================
# GUI
# =============================================================================
class SeqSelectorDialog:
    def __init__(self, parent, seq_folders):
        self.result = None
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Seleccionar SEQs")
        self.dialog.geometry("450x400")
        self.dialog.transient(parent)
        self.dialog.grab_set()

        main = ttk.Frame(self.dialog, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main, text=f"{len(seq_folders)} SEQs trobades",
                  font=("Segoe UI", 11, "bold")).pack(pady=(0,10))

        # Lista con scroll
        frame = ttk.Frame(main)
        frame.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(frame, height=250)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        self.scroll_frame = ttk.Frame(canvas)

        self.scroll_frame.bind("<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0,0), window=self.scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.check_vars = {}
        for path in seq_folders:
            var = tk.BooleanVar(value=True)
            self.check_vars[path] = var
            ttk.Checkbutton(self.scroll_frame, text=os.path.basename(path),
                           variable=var).pack(anchor="w")

        # Botons
        btn_frame = ttk.Frame(main)
        btn_frame.pack(fill=tk.X, pady=10)
        ttk.Button(btn_frame, text="Tot", command=lambda: [v.set(True) for v in self.check_vars.values()]).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="Res", command=lambda: [v.set(False) for v in self.check_vars.values()]).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Processar", command=self._ok).pack(side=tk.RIGHT)
        ttk.Button(btn_frame, text="Cancel", command=self._cancel).pack(side=tk.RIGHT, padx=5)

        self.dialog.protocol("WM_DELETE_WINDOW", self._cancel)
        self.dialog.wait_window()

    def _ok(self):
        self.result = [p for p,v in self.check_vars.items() if v.get()]
        self.dialog.destroy()

    def _cancel(self):
        self.result = None
        self.dialog.destroy()


# =============================================================================
# PROCESSAMENT
# =============================================================================
def process_seqs(base_dir, selected_seqs, progress_cb=None):
    """Processa SEQs COLUMN i genera PDF compacte."""

    date_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    pdf_path = os.path.join(base_dir, f"REPORT_DOCtor_C_{date_str}.pdf")

    all_samples = []  # Per taula resum

    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        pass

    with PdfPages(pdf_path) as pdf:

        # === PROCESSAR CADA SEQ ===
        for seq_idx, seq_path in enumerate(selected_seqs):
            seq_name = os.path.basename(seq_path)
            if progress_cb:
                progress_cb(seq_idx + 1, len(selected_seqs), seq_name)

            results_folder = os.path.join(seq_path, "Resultats_Consolidats")
            if not os.path.isdir(results_folder):
                continue

            # Llistar fitxers COLUMN (excloure BP, blancs, patrons i controls)
            EXCLUDE_PATTERNS = ["mq", "naoh", "khp", "no3", "na2co3", "br", "blank", "blanco"]
            files = []
            for f in os.listdir(results_folder):
                if not f.lower().endswith(".xlsx"):
                    continue
                fname_lower = f.lower()
                if any(pat in fname_lower for pat in EXCLUDE_PATTERNS):
                    continue
                # Excloure fitxers BP (aquests van a DOCtor_BP.py)
                is_bp_file = "_bp" in fname_lower or fname_lower.endswith("bp.xlsx")
                if is_bp_file:
                    continue
                files.append(os.path.join(results_folder, f))

            if not files:
                continue

            # Agrupar per mostra
            groups = {}
            for f in files:
                key, rep = sample_key_from_filename(os.path.basename(f))
                groups.setdefault(key, []).append((rep or "?", f))

            for k in groups:
                groups[k].sort(key=lambda x: (x[0] != "1", x[0] != "2", x[0]))

            # Processar mostres
            seq_samples = []
            for sample_key, items in groups.items():
                sample_data = {
                    "seq": seq_name,
                    "sample": sample_key,
                    "type": "?",
                    "n_reps": len(items),
                    "pearson": np.nan,
                    "max_mau": 0,
                    "status": "OK",  # OK, LOW_SIGNAL, CHECK, PEAK_MISMATCH, SINGLE_REP
                    "n_peaks_r1": 0,
                    "n_peaks_r2": 0,
                    "peak_mismatch": False,
                    "batman": False,
                    "batman_drop": 0,
                    "ears": 0,
                    "timeout": False,
                    "timeout_count": 0,
                    # NOTA: camps amorphous eliminats (v1.1)
                    "reps": []
                }

                rep_data = []
                for rep_id, fpath in items[:2]:
                    t, y, stype = read_excel_doc(fpath)
                    sample_data["type"] = stype

                    if t is not None:
                        max_y = float(np.max(y))
                        sample_data["max_mau"] = max(sample_data["max_mau"], max_y)

                        # Batman
                        bat = detect_batman_signal(t, y)
                        if bat:
                            sample_data["batman"] = True
                            sample_data["batman_drop"] = max(sample_data["batman_drop"], bat.get("max_depth", 0)*100)

                        # Orelletes
                        ears = detect_ears(t, y)
                        sample_data["ears"] = max(sample_data["ears"], len(ears))

                        # TimeOUT (saturació/mesetes)
                        timeouts = detect_timeout(t, y)
                        if timeouts:
                            sample_data["timeout"] = True
                            sample_data["timeout_count"] = max(sample_data.get("timeout_count", 0), len(timeouts))

                        # NOTA: Detecció AMORPHOUS eliminada (v1.1) - precària

                        # Anàlisi de pics - només pic principal
                        peak_analysis = analyze_peaks(t, y)

                        # Extreure info
                        n_peaks = 0
                        n_with_issues = 0
                        peaks_with_issues = []

                        if peak_analysis and peak_analysis.get("status") != "LOW_SIGNAL":
                            n_peaks = peak_analysis.get("n_peaks", 0)
                            n_with_issues = peak_analysis.get("n_with_issues", 0)
                            # Guardar pics amb problemes per marcar al gràfic
                            peaks_with_issues = [p for p in peak_analysis.get("peaks", []) if p["has_issues"]]

                        rep_data.append({
                            "rep": rep_id, "t": t, "y": y,
                            "bat": bat, "ears": ears, "timeouts": timeouts,
                            "peak_analysis": peak_analysis,
                            "n_peaks": n_peaks,
                            "n_with_issues": n_with_issues,
                            "peaks_with_issues": peaks_with_issues,
                            "all_peaks": peak_analysis.get("peaks", []) if peak_analysis else []
                        })

                # === DETERMINAR ESTATUS ===

                # 1. Filtre soroll
                if sample_data["max_mau"] < NOISE_THRESHOLD:
                    sample_data["status"] = "LOW_SIGNAL"

                # 2. Dues rèpliques - comparar
                elif len(rep_data) >= 2:
                    t1, y1 = rep_data[0]["t"], rep_data[0]["y"]
                    t2, y2 = rep_data[1]["t"], rep_data[1]["y"]
                    sample_data["pearson"], _ = calc_pearson_replicates(t1, y1, t2, y2)

                    # NOTA: validate_amorphous_with_replicates eliminada (v1.1)

                    # Re-avaluar problemes de pics amb el Pearson real (Asimetria + Pearson)
                    pearson_val = sample_data["pearson"] if not np.isnan(sample_data["pearson"]) else 1.0
                    for rep in rep_data:
                        n_issues = 0
                        for pk in rep.get("all_peaks", []):
                            as_33 = pk.get("as_33", 1.0)
                            as_66 = pk.get("as_66", 1.0)
                            as_divergence = pk.get("as_divergence", 0.0)
                            is_problem, problem_reason = detect_peak_problem(as_33, as_66, as_divergence, pearson_val)
                            pk["is_problem"] = is_problem
                            pk["problem_reason"] = problem_reason
                            # Actualitzar flags
                            if is_problem and "PROBLEM" not in pk["flags"]:
                                pk["flags"].append("PROBLEM")
                                pk["has_issues"] = True
                            elif not is_problem and "PROBLEM" in pk["flags"]:
                                pk["flags"].remove("PROBLEM")
                                pk["has_issues"] = len(pk["flags"]) > 0
                            if pk["has_issues"]:
                                n_issues += 1
                        rep["n_with_issues"] = n_issues
                        rep["peaks_with_issues"] = [p for p in rep.get("all_peaks", []) if p["has_issues"]]
                        # Actualitzar peak_analysis si existeix
                        if rep.get("peak_analysis"):
                            rep["peak_analysis"]["n_with_issues"] = n_issues

                    # Detectar peak mismatch (diferent nombre de pics)
                    n_peaks_r1 = rep_data[0].get("n_peaks", 0)
                    n_peaks_r2 = rep_data[1].get("n_peaks", 0)
                    sample_data["n_peaks_r1"] = n_peaks_r1
                    sample_data["n_peaks_r2"] = n_peaks_r2
                    sample_data["peak_mismatch"] = n_peaks_r1 != n_peaks_r2

                    # Determinar status
                    if sample_data["peak_mismatch"]:
                        sample_data["status"] = "PEAK_MISMATCH"
                    elif not np.isnan(sample_data["pearson"]) and sample_data["pearson"] <= PEARSON_INSPECT:
                        sample_data["status"] = "CHECK"
                    else:
                        sample_data["status"] = "OK"

                # 3. Sense rèplica - warning
                elif len(rep_data) == 1:
                    sample_data["n_peaks_r1"] = rep_data[0].get("n_peaks", 0)
                    sample_data["n_peaks_r2"] = 0
                    sample_data["peak_mismatch"] = False
                    sample_data["status"] = "SINGLE_REP"  # Warning: només una rèplica

                # === DETERMINAR RÈPLICA TRIADA ===
                # Usar la funció select_best_replica amb la nova lògica:
                # 1. Batman prioritari
                # 2. Regla D (problemes detectats)
                # 3. Alçada
                # 4. Soroll

                if len(rep_data) >= 2:
                    # Preparar dades per select_best_replica
                    rep1_info = {
                        "batman": rep_data[0].get("bat"),
                        "peaks": rep_data[0].get("all_peaks", []),
                        "baseline_noise": rep_data[0].get("peak_analysis", {}).get("baseline_noise", 0)
                    }
                    rep2_info = {
                        "batman": rep_data[1].get("bat"),
                        "peaks": rep_data[1].get("all_peaks", []),
                        "baseline_noise": rep_data[1].get("peak_analysis", {}).get("baseline_noise", 0)
                    }

                    selection = select_best_replica(rep1_info, rep2_info)

                    best_idx = 0 if selection["best"] == "R1" else 1
                    motiu = selection["reason"]
                    warning = selection.get("warning")

                    for i, rep in enumerate(rep_data):
                        rep["triada"] = (i == best_idx)
                        rep["_motiu"] = motiu if i == best_idx else ""
                        rep["_dominated"] = selection.get(f"r{i+1}_dominated", False)

                    # Afegir warning a sample_data si cal
                    if warning:
                        sample_data["selection_warning"] = warning
                        if "WARNING" in warning:
                            sample_data["status"] = "CHECK"

                elif len(rep_data) == 1:
                    # Una sola rèplica - comprovar si té problemes
                    rep1_info = {
                        "batman": rep_data[0].get("bat"),
                        "peaks": rep_data[0].get("all_peaks", []),
                        "baseline_noise": rep_data[0].get("peak_analysis", {}).get("baseline_noise", 0)
                    }

                    selection = select_best_replica(rep1_info, None)

                    rep_data[0]["triada"] = True
                    rep_data[0]["_motiu"] = selection["reason"]
                    rep_data[0]["_dominated"] = selection.get("r1_dominated", False)

                    if selection.get("warning"):
                        sample_data["selection_warning"] = selection["warning"]
                        sample_data["status"] = "CHECK"

                sample_data["reps"] = rep_data
                seq_samples.append(sample_data)
                all_samples.append(sample_data)

            # === PÀGINES: 2 mostres per pàgina, rèpliques separades ===
            # Filtrar mostres amb senyal (excloure LOW_SIGNAL i sense dades)
            samples_with_signal = [s for s in seq_samples
                                   if s["status"] != "LOW_SIGNAL" and len(s["reps"]) > 0]

            samples_per_page = 2
            for page_start in range(0, len(samples_with_signal), samples_per_page):
                page_samples = samples_with_signal[page_start:page_start + samples_per_page]

                fig, axes = plt.subplots(len(page_samples), 2, figsize=(11, 4*len(page_samples)))
                fig.suptitle(f"{seq_name} [COLUMN]", fontsize=12, fontweight="bold")

                # Assegurar que axes és 2D
                if len(page_samples) == 1:
                    axes = axes.reshape(1, -1)

                for row_idx, sample in enumerate(page_samples):
                    # Color segons estatus
                    status = sample["status"]
                    status_colors = {
                        "OK": "green",
                        "PEAK_MISMATCH": "red",
                        "CHECK": "darkorange",
                        "SINGLE_REP": "purple",
                        "LOW_SIGNAL": "gray"
                    }
                    title_color = status_colors.get(status, "black")

                    # Info comuna
                    pears = sample["pearson"]
                    pears_str = f"ρ={pears:.3f}" if not np.isnan(pears) else "1 rèplica"

                    # Dibuixar cada rèplica en columna separada
                    reps = sample["reps"][:2]
                    for col_idx in range(2):
                        ax = axes[row_idx, col_idx]

                        if col_idx < len(reps):
                            rep = reps[col_idx]
                            color = "#2E86AB" if col_idx == 0 else "#A23B72"

                            # Plot senyal
                            ax.plot(rep["t"], rep["y"], color=color, linewidth=1)

                            # Marcar Batman (valls al cim del pic)
                            if rep["bat"]:
                                b = rep["bat"]
                                # Nova estructura: t_top, y_top, valleys (índexs)
                                if "t_top" in b and "valleys" in b:
                                    t_top = b["t_top"]
                                    y_top = b["y_top"]
                                    valleys = b["valleys"]
                                    # Marcar cada vall amb triangle vermell invertit
                                    for v_idx in valleys:
                                        if v_idx < len(t_top):
                                            ax.scatter([t_top[v_idx]], [y_top[v_idx]],
                                                      c="red", s=80, zorder=6, marker="v",
                                                      edgecolors="darkred", linewidths=1.5)
                                    # Línia horitzontal al threshold
                                    if "threshold" in b:
                                        ax.axhline(b["threshold"], color="red", linestyle=":",
                                                  alpha=0.3, linewidth=1)

                            # Marcar Orelletes (triangles taronges)
                            for e in rep["ears"][:5]:
                                ax.scatter([e["t"]], [e["height"]],
                                          c="orange", s=50, zorder=5, marker="^",
                                          edgecolors="darkorange", linewidths=1.5)

                            # Marcar TimeOUT (mesetes/saturació) - banda verda amb CV
                            for to in rep.get("timeouts", []):
                                ax.axvspan(to["t_start"], to["t_end"], color="#2ecc71", alpha=0.3, zorder=1)
                                # Mostrar CV petit a la part superior (baix = pla)
                                cv_val = to.get("cv", 0)
                                ax.text(to["t_start"] + 0.05, to["y_mean"] * 1.02,
                                       f"CV={cv_val:.1f}%", fontsize=6, color="#27ae60",
                                       verticalalignment="bottom")

                            # NOTA: Visualització AMORPHOUS eliminada (v1.1)

                            # Mostrar info de TOTS els pics amb detecció híbrida
                            # Recollir etiquetes per mostrar a la llegenda
                            peak_labels = []

                            for pk in rep.get("all_peaks", []):
                                t_pk = pk["t_peak"]
                                h_pk = pk["height"]
                                baseline_pk = float(np.min(rep["y"][pk["left_base"]:pk["right_base"]+1]))
                                is_anomaly = pk.get("is_anomaly", False)
                                is_batman = pk.get("is_batman", False)
                                is_irregular = pk.get("is_irregular", False)
                                has_stop = pk.get("has_stop", False)
                                smoothness = pk.get("smoothness", 100)
                                anomaly_info = pk.get("anomaly", {})

                                # Marcar Batman al pic (valls al cim)
                                if is_batman:
                                    batman_info = anomaly_info.get("batman_info", {})
                                    if "t_top" in batman_info and "valleys" in batman_info:
                                        t_top = batman_info["t_top"]
                                        y_top = batman_info["y_top"]
                                        for v_idx in batman_info["valleys"]:
                                            if v_idx < len(t_top):
                                                ax.scatter([t_top[v_idx]], [y_top[v_idx]],
                                                          c="red", s=60, zorder=7, marker="v",
                                                          edgecolors="darkred", linewidths=1.5)

                                # Color segons si té anomalia
                                has_any_anomaly = is_anomaly or has_stop
                                if has_any_anomaly:
                                    txt_color = "red"
                                    marker_style = "x"
                                else:
                                    txt_color = "darkgreen"
                                    marker_style = "."

                                # Marcador petit al màxim del pic
                                ax.scatter([t_pk], [h_pk + baseline_pk], c=txt_color, s=30, zorder=6,
                                          marker=marker_style, linewidths=1.5)

                                # Etiqueta A SOTA del pic (a la baseline) - prioritzar anomalies
                                labels = []
                                if is_batman:
                                    labels.append("BAT")
                                # NOTA: Etiquetes AMORF i replicate_rejected eliminades (v1.1)
                                if has_stop:
                                    stop_cv = pk.get("stop_cv", 0)
                                    labels.append(f"STOP({stop_cv:.1f}%)")
                                if is_irregular and not labels:
                                    labels.append(f"IRR:{smoothness:.0f}%")

                                if not labels:
                                    label = f"{smoothness:.0f}%"
                                else:
                                    label = "+".join(labels)

                                # Posició: sota el pic, centrat
                                ax.annotate(label, (t_pk, baseline_pk - 5), fontsize=6,
                                           color=txt_color, ha="center", va="top",
                                           fontweight="bold" if has_any_anomaly else "normal")

                                # Guardar info per zoom si és anomalia
                                if has_any_anomaly:
                                    anom_type = "BAT" if is_batman else ("STOP" if has_stop else "IRR")
                                    peak_labels.append({
                                        "t_pk": t_pk, "h_pk": h_pk + baseline_pk,
                                        "left": pk["left_base"], "right": pk["right_base"],
                                        "type": anom_type,
                                        "anomaly_info": anomaly_info
                                    })

                            # === ZOOM INSET per anomalies ===
                            if peak_labels:
                                # Mostrar zoom del primer pic amb anomalia
                                anom = peak_labels[0]
                                t_data = rep["t"]
                                y_data = rep["y"]

                                # Rang del zoom (±2 min al voltant del pic)
                                t_center = anom["t_pk"]
                                t_zoom_min = max(t_center - 2, t_data[anom["left"]])
                                t_zoom_max = min(t_center + 2, t_data[anom["right"]])

                                # Màscara per zoom
                                zoom_mask = (t_data >= t_zoom_min) & (t_data <= t_zoom_max)
                                t_zoom = t_data[zoom_mask]
                                y_zoom = y_data[zoom_mask]

                                if len(t_zoom) > 5:
                                    # Crear inset axes (cantonada superior dreta)
                                    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
                                    axins = inset_axes(ax, width="35%", height="35%", loc="upper right",
                                                      borderpad=1)

                                    # Plot zoom
                                    axins.plot(t_zoom, y_zoom, color=color, linewidth=1.5)

                                    # Marcar valls si és Batman
                                    if anom["type"] == "BAT":
                                        batman_info = anom["anomaly_info"].get("batman_info", {})
                                        if "t_top" in batman_info and "valleys" in batman_info:
                                            t_top = batman_info["t_top"]
                                            y_top = batman_info["y_top"]
                                            for v_idx in batman_info["valleys"]:
                                                if v_idx < len(t_top):
                                                    axins.scatter([t_top[v_idx]], [y_top[v_idx]],
                                                                c="red", s=40, zorder=7, marker="v",
                                                                edgecolors="darkred", linewidths=1)

                                    # Format inset
                                    axins.set_xlim(t_zoom_min, t_zoom_max)
                                    y_min_zoom = np.min(y_zoom)
                                    y_max_zoom = np.max(y_zoom)
                                    y_margin = (y_max_zoom - y_min_zoom) * 0.1
                                    axins.set_ylim(y_min_zoom - y_margin, y_max_zoom + y_margin)
                                    axins.set_title(f"ZOOM: {anom['type']}", fontsize=7, color="red")
                                    axins.tick_params(labelsize=5)
                                    axins.grid(True, alpha=0.3)

                                    # Marc vermell per destacar
                                    for spine in axins.spines.values():
                                        spine.set_edgecolor("red")
                                        spine.set_linewidth(1.5)

                            # Títol amb info de rèplica
                            n_peaks = rep.get("n_peaks", 0)
                            n_issues = rep.get("n_with_issues", 0)

                            rep_flags = []
                            rep_flags.append(f"{n_peaks} pics")
                            if n_issues > 0:
                                rep_flags.append(f"{n_issues} CHECK")
                            if rep["bat"]:
                                bat_depth = rep['bat'].get('max_depth', 0) * 100
                                rep_flags.append(f"BAT:{bat_depth:.0f}%")
                            if rep["ears"]:
                                rep_flags.append(f"EAR:{len(rep['ears'])}")
                            if rep.get("timeouts"):
                                # Mostrar CV del primer STOP (baix = pla)
                                cv_stop = rep['timeouts'][0].get('cv', 0) if rep['timeouts'] else 0
                                rep_flags.append(f"STOP(CV={cv_stop:.1f}%)")

                            rep_info = " | ".join(rep_flags)

                            # Marca de rèplica triada
                            triada_mark = " *" if rep.get("triada") else ""
                            title_color = "orange" if n_issues > 0 else "green"
                            if rep["bat"]:
                                title_color = "red"
                            if rep.get("timeouts"):
                                title_color = "#27ae60"  # Verd fosc per STOP

                            ax.set_title(f"R{rep['rep']}{triada_mark}: {rep_info}", fontsize=9,
                                        color=title_color, fontweight="bold" if rep.get("triada") else "normal")
                            ax.set_xlabel("min", fontsize=8)
                            ax.set_ylabel("mAU", fontsize=8)
                            ax.set_xlim(10, 70)  # Rang temporal estàndard
                            ax.tick_params(labelsize=7)
                            ax.grid(True, alpha=0.3)

                            # Marc verd per rèplica triada, taronja si té pics amb problemes
                            if rep.get("triada"):
                                has_issues = rep.get("n_with_issues", 0) > 0
                                border_color = "green" if not has_issues else "orange"
                                for spine in ax.spines.values():
                                    spine.set_edgecolor(border_color)
                                    spine.set_linewidth(3)
                        else:
                            # No hi ha segona rèplica
                            ax.text(0.5, 0.5, "Sense rèplica", ha="center", va="center",
                                   fontsize=12, color="gray", transform=ax.transAxes)
                            ax.axis("off")

                    # Títol de fila (nom mostra + status + pearson + pics)
                    n_p1 = sample.get("n_peaks_r1", 0)
                    n_p2 = sample.get("n_peaks_r2", 0)
                    peaks_str = f"Pics:{n_p1}/{n_p2}" if n_p2 > 0 else f"Pics:{n_p1}"
                    mismatch_str = " MISMATCH!" if sample.get("peak_mismatch") else ""
                    row_title = f"{sample['sample'][:30]}  [{status}]  {pears_str}  {peaks_str}{mismatch_str}"
                    axes[row_idx, 0].annotate(row_title, xy=(0, 1.15), xycoords='axes fraction',
                                              fontsize=10, fontweight="bold", color=title_color)

                plt.tight_layout()
                pdf.savefig(fig, facecolor='white')
                plt.close(fig)

        # === TAULA RESUM GLOBAL ===
        if all_samples:
            # Ordenar per status (problemàtics primer), després per Pearson
            status_order = {"PEAK_MISMATCH": 0, "CHECK": 1, "SINGLE_REP": 2, "LOW_SIGNAL": 3, "OK": 4}
            all_samples.sort(key=lambda x: (
                status_order.get(x["status"], 5),
                x["pearson"] if not np.isnan(x["pearson"]) else 1
            ))

            # Pàgines de taula
            rows_per_page = 30
            for page_start in range(0, len(all_samples), rows_per_page):
                page_samples = all_samples[page_start:page_start + rows_per_page]

                fig = plt.figure(figsize=(11, 8))
                ax = fig.add_subplot(111)
                ax.axis("off")

                if page_start == 0:
                    ax.set_title("RESUM - Ordenat per Status (problemàtics primer)",
                                fontsize=14, fontweight="bold", pad=20)

                cols = ["SEQ", "Mostra", "Status", "Pearson", "Pics", "mAU", "Bat", "Ear", "STOP"]
                rows = []
                cell_colors = []

                # Colors per status
                status_bg = {
                    "PEAK_MISMATCH": "#ffcccc",  # Vermell clar
                    "CHECK": "#fff3cd",          # Groc clar
                    "SINGLE_REP": "#e6ccff",     # Lila clar
                    "LOW_SIGNAL": "#e6e6e6",     # Gris clar
                    "TIMEOUT": "#d5f5e3",        # Verd clar
                    "OK": "white"
                }

                for s in page_samples:
                    pears = f"{s['pearson']:.3f}" if not np.isnan(s['pearson']) else "-"
                    n_p1 = str(s.get('n_peaks_r1', 0))
                    bat = f"{s['batman_drop']:.0f}%" if s['batman'] else "-"
                    ears = str(s['ears']) if s['ears'] > 0 else "-"
                    stop = str(s.get('timeout_count', 0)) if s.get('timeout') else "-"

                    rows.append([
                        s['seq'][:12],
                        s['sample'][:18],
                        s['status'],
                        pears,
                        n_p1,
                        f"{s['max_mau']:.0f}",
                        bat, ears, stop
                    ])

                    # Color per fila segons status
                    bg = status_bg.get(s['status'], "white")
                    cell_colors.append([bg] * len(cols))

                tbl = ax.table(cellText=rows, colLabels=cols, loc="center",
                              cellLoc="center", cellColours=cell_colors)
                tbl.auto_set_font_size(False)
                tbl.set_fontsize(8)
                tbl.scale(1.1, 1.25)

                # Capçalera
                for j in range(len(cols)):
                    tbl[(0, j)].set_facecolor("steelblue")
                    tbl[(0, j)].set_text_props(color="white", fontweight="bold")

                plt.tight_layout()
                pdf.savefig(fig, facecolor='white')
                plt.close(fig)

        # === ESTADÍSTIQUES FINALS ===
        fig = plt.figure(figsize=(11, 8))

        # Histograma Pearson
        ax1 = fig.add_subplot(221)
        pearson_vals = [s["pearson"] for s in all_samples if not np.isnan(s["pearson"])]
        if pearson_vals:
            ax1.hist(pearson_vals, bins=20, color="steelblue", edgecolor="white", alpha=0.8)
            ax1.axvline(PEARSON_INSPECT, color="red", linestyle="--", label=f"Inspecció ({PEARSON_INSPECT})")
            ax1.set_xlabel("Pearson")
            ax1.set_ylabel("Freqüència")
            ax1.set_title("Distribució Pearson")
            ax1.legend(fontsize=8)

        # Max mAU vs Anomalies (Batman/Ears)
        ax2 = fig.add_subplot(222)
        max_vals_all = [s["max_mau"] for s in all_samples]
        has_anomaly = [s["batman"] or s["ears"] > 0 for s in all_samples]
        if max_vals_all:
            colors = ["red" if a else "steelblue" for a in has_anomaly]
            y_pos = [1 if a else 0 for a in has_anomaly]
            ax2.scatter(max_vals_all, y_pos, c=colors, alpha=0.5, s=20)
            ax2.axvline(NOISE_THRESHOLD, color="gray", linestyle="--", label=f"Soroll ({NOISE_THRESHOLD})")
            ax2.set_xlabel("Max mAU")
            ax2.set_ylabel("Anomalia (Batman/Ears)")
            ax2.set_yticks([0, 1])
            ax2.set_yticklabels(["No", "Sí"])
            ax2.set_title("Senyal màxim vs Anomalies")
            ax2.legend(fontsize=8)

        # Resum numèric per STATUS
        ax3 = fig.add_subplot(223)
        ax3.axis("off")

        n_total = len(all_samples)
        n_mismatch = sum(1 for s in all_samples if s["status"] == "PEAK_MISMATCH")
        n_check = sum(1 for s in all_samples if s["status"] == "CHECK")
        n_single = sum(1 for s in all_samples if s["status"] == "SINGLE_REP")
        n_low = sum(1 for s in all_samples if s["status"] == "LOW_SIGNAL")
        n_ok = sum(1 for s in all_samples if s["status"] == "OK")
        n_batman = sum(1 for s in all_samples if s["batman"])
        n_ears = sum(1 for s in all_samples if s["ears"] > 0)
        n_timeout = sum(1 for s in all_samples if s.get("timeout"))

        pct = lambda n: f"{100*n/n_total:.1f}%" if n_total > 0 else "0%"

        summary = (
            f"RESUM PER STATUS [COLUMN]\n"
            f"════════════════════════════\n\n"
            f"  Mostres totals:     {n_total}\n"
            f"  SEQs processades:   {len(selected_seqs)}\n\n"
            f"  PEAK_MISMATCH:      {n_mismatch} ({pct(n_mismatch)})\n"
            f"  CHECK:              {n_check} ({pct(n_check)})\n"
            f"  SINGLE_REP:         {n_single} ({pct(n_single)})\n"
            f"  LOW_SIGNAL:         {n_low} ({pct(n_low)})\n"
            f"  OK:                 {n_ok} ({pct(n_ok)})\n\n"
            f"  Batman:             {n_batman} ({pct(n_batman)})\n"
            f"  Orelletes:          {n_ears} ({pct(n_ears)})\n"
            f"  TimeOUT (STOP):     {n_timeout} ({pct(n_timeout)})\n"
        )
        ax3.text(0.05, 0.5, summary, fontsize=11, family="monospace",
                verticalalignment="center", transform=ax3.transAxes,
                bbox=dict(boxstyle="round", facecolor="whitesmoke"))

        # Info generació
        ax4 = fig.add_subplot(224)
        ax4.axis("off")
        ax4.text(0.5, 0.5, f"DOCtor_C v1.0\nCOLUMN\n{datetime.now().strftime('%Y-%m-%d %H:%M')}",
                fontsize=14, ha="center", va="center", color="gray")

        plt.tight_layout()
        pdf.savefig(fig, facecolor='white')
        plt.close(fig)

        # === TAULA DETALL STOPS ===
        # Recollir tots els STOP de totes les mostres
        stop_rows = []
        for s in all_samples:
            for rep in s.get("reps", []):
                for to in rep.get("timeouts", []):
                    stop_rows.append({
                        "sample": s["sample"][:20],
                        "rep": rep.get("rep", "?"),
                        "t_start": to.get("t_start", 0),
                        "t_end": to.get("t_end", 0),
                        "dur": to.get("duration", 0),
                        "y_mean": to.get("y_mean", 0),
                        "y_std": to.get("y_std", 0),
                        "cv": to.get("cv", 0),
                        "baseline": to.get("baseline", 0),
                        "edge": to.get("max_edge", 0),
                        "ratio": to.get("ratio", 0)
                    })

        if stop_rows:
            # Crear pàgina amb taula STOP
            fig_stop = plt.figure(figsize=(11.69, 8.27))
            ax_stop = fig_stop.add_subplot(111)
            ax_stop.axis("off")
            ax_stop.set_title("DETALL STOPS - Valors per afinar llindars (CV baix = pla)", fontsize=14, fontweight="bold", pad=20)

            # Capçaleres
            cols = ["Mostra", "R", "t_ini", "t_fi", "Dur", "y_mean", "std", "CV%", "Base", "Edge", "Ratio"]

            # Dades
            table_data = []
            for r in stop_rows:
                table_data.append([
                    r["sample"],
                    r["rep"],
                    f"{r['t_start']:.2f}",
                    f"{r['t_end']:.2f}",
                    f"{r['dur']:.2f}",
                    f"{r['y_mean']:.1f}",
                    f"{r['y_std']:.2f}",
                    f"{r['cv']:.2f}",
                    f"{r['baseline']:.1f}",
                    f"{r['edge']:.2f}",
                    f"{r['ratio']:.1f}"
                ])

            # Crear taula
            tbl = ax_stop.table(cellText=table_data, colLabels=cols, loc="center", cellLoc="center")
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(8)
            tbl.scale(1.2, 1.4)

            # Estil capçalera
            for j in range(len(cols)):
                tbl[(0, j)].set_facecolor("#27ae60")
                tbl[(0, j)].set_text_props(color="white", fontweight="bold")

            # Afegir llindars actuals com a referència
            thresholds_text = (
                f"Llindars: DUR={TIMEOUT_MIN_DUR}-{TIMEOUT_MAX_DUR} | LEVEL>{TIMEOUT_MIN_LEVEL_FACTOR}x | "
                f"EDGE>{TIMEOUT_MIN_EDGE_SLOPE} | RATIO>{TIMEOUT_MIN_CONTRAST} | CV<{TIMEOUT_MAX_CV}%"
            )
            ax_stop.text(0.5, 0.02, thresholds_text, transform=ax_stop.transAxes,
                        ha="center", fontsize=8, color="gray")

            plt.tight_layout()
            pdf.savefig(fig_stop, facecolor='white')
            plt.close(fig_stop)

    return pdf_path, all_samples


# =============================================================================
# GUI PRINCIPAL
# =============================================================================
class DOCtorApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("DOCtor_C v1.0 - Column")
        self.root.geometry("400x220")
        self.root.resizable(False, False)

        main = ttk.Frame(self.root, padding=20)
        main.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main, text="DOCtor_C v1.0", font=("Segoe UI", 16, "bold")).pack()
        ttk.Label(main, text="Anàlisi COLUMN - SUITE HPSEC", foreground="gray").pack()
        ttk.Label(main, text="(Per BP utilitzar DOCtor_BP.py)", foreground="gray", font=("Segoe UI", 8)).pack(pady=(0,15))

        self.btn = ttk.Button(main, text="Seleccionar carpeta...", command=self._run)
        self.btn.pack(pady=10, ipadx=15, ipady=8)

        self.progress = tk.StringVar(value="")
        ttk.Label(main, textvariable=self.progress, font=("Segoe UI", 9)).pack(pady=10)

        self.pbar = ttk.Progressbar(main, length=300, mode='determinate')
        self.pbar.pack()

    def _run(self):
        base_dir = filedialog.askdirectory(title="Carpeta arrel amb SEQs")
        if not base_dir:
            return

        # Trobar SEQs
        seq_folders = sorted([os.path.join(base_dir, d) for d in os.listdir(base_dir)
                             if os.path.isdir(os.path.join(base_dir, d, "Resultats_Consolidats"))])

        if not seq_folders:
            messagebox.showwarning("Avís", "No s'han trobat SEQs amb Resultats_Consolidats")
            return

        selector = SeqSelectorDialog(self.root, seq_folders)
        if not selector.result:
            return

        selected = selector.result
        self.btn.config(state="disabled")
        self.pbar["maximum"] = len(selected)

        def progress(i, n, name):
            self.progress.set(f"{i}/{n}: {name}")
            self.pbar["value"] = i
            self.root.update()

        try:
            pdf_path, samples = process_seqs(base_dir, selected, progress)

            n_issues = sum(1 for s in samples if s["status"] not in ["OK", "LOW_SIGNAL"])

            self.progress.set(f"Fet! {len(samples)} mostres, {n_issues} amb anomalies")
            messagebox.showinfo("Completat", f"PDF: {os.path.basename(pdf_path)}\n{len(samples)} mostres processades")
            os.startfile(base_dir)
        except Exception as e:
            messagebox.showerror("Error", str(e))
        finally:
            self.btn.config(state="normal")
            self.pbar["value"] = 0

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    DOCtorApp().run()
