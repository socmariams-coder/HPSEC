"""
hpsec_utils.py
==============
Funciones compartidas entre los scripts HPSEC.

Contiene:
- baseline_stats: Estadístiques de baseline (mitjana, std, llindar 3σ)
- detect_main_peak: Detección de pico principal con scipy
- detect_batman: Detecció de doble pic (Batman) amb filtres robusts
- seleccionar_carpeta: GUI para selección de carpeta
- obtenir_seq: Extracción de ID de secuencia
- is_khp: Detección de muestras KHP
- normalize_key: Normalización de strings para matching
"""

import os
import re
import tkinter as tk
from tkinter import filedialog

import numpy as np
from scipy.signal import find_peaks
from scipy.integrate import trapezoid


# =============================================================================
# DETECCIÓN DE PICO PRINCIPAL
# =============================================================================
def detect_main_peak(t, y, min_prominence_pct=5.0):
    """
    Detecta el pico principal en una señal cromatográfica.

    Args:
        t: Array de tiempos (minutos)
        y: Array de intensidades
        min_prominence_pct: Prominencia mínima como % del máximo (default: 5.0)

    Returns:
        dict con keys:
            - valid: bool indicando si se encontró pico válido
            - area: área del pico (trapezoid)
            - t_start: tiempo inicio del pico
            - t_max: tiempo del máximo
            - t_end: tiempo fin del pico
            - height: altura del pico
            - prominence: prominencia del pico
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(t) < 10 or len(y) < 10:
        return {"valid": False}

    y_max = float(np.nanmax(y))
    if y_max < 1e-6:
        return {"valid": False}

    min_prominence = y_max * (min_prominence_pct / 100.0)
    peaks, props = find_peaks(y, prominence=min_prominence, width=3)

    if len(peaks) == 0:
        return {"valid": False}

    # Seleccionar pico con mayor prominencia
    idx = int(np.argmax(props["prominences"]))
    main_peak = int(peaks[idx])
    left_idx = int(props["left_bases"][idx])
    right_idx = int(props["right_bases"][idx])

    # Calcular área usando regla del trapecio
    area = float(trapezoid(y[left_idx:right_idx + 1], t[left_idx:right_idx + 1]))

    return {
        "valid": True,
        "area": area,
        "t_start": float(t[left_idx]),
        "t_max": float(t[main_peak]),
        "t_end": float(t[right_idx]),
        "height": float(y[main_peak]),
        "prominence": float(props["prominences"][idx]),
    }


# =============================================================================
# GUI - SELECCIÓN DE CARPETA
# =============================================================================
def seleccionar_carpeta(titulo="Selecciona carpeta SEQ"):
    """
    Muestra diálogo para seleccionar carpeta.

    Args:
        titulo: Título del diálogo

    Returns:
        str con la ruta de la carpeta seleccionada, o string vacío si se cancela
    """
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    folder = filedialog.askdirectory(title=titulo)
    root.destroy()
    return folder


# =============================================================================
# EXTRACCIÓN DE IDENTIFICADORES
# =============================================================================
def obtenir_seq(folder):
    """
    Extrae el ID de secuencia del nombre de carpeta.

    Args:
        folder: Ruta de la carpeta

    Returns:
        str con el ID de secuencia (ej: "123A") o "000" si no se encuentra
    """
    nom = os.path.basename(os.path.normpath(folder))
    m = re.search(r"(\d+[A-Za-z]?)", nom)
    return m.group(1) if m else "000"


def normalize_key(s):
    """
    Normaliza un string para matching (elimina caracteres especiales, uppercase).

    Args:
        s: String a normalizar

    Returns:
        str normalizado (solo alfanuméricos, uppercase)
    """
    return re.sub(r"[^A-Za-z0-9]+", "", str(s or "")).upper()


# =============================================================================
# DETECCIÓN DE MUESTRAS KHP
# =============================================================================
def is_khp(name):
    """
    Determina si un nombre corresponde a una muestra KHP.

    Args:
        name: Nombre de la muestra

    Returns:
        bool indicando si es KHP
    """
    return "KHP" in str(name).upper()


def extract_khp_conc(name):
    """
    Extrae la concentración de KHP del nombre.

    Args:
        name: Nombre que contiene la concentración (ej: "KHP50", "KHP 100")

    Returns:
        int con la concentración en ppm, o 0 si no se encuentra
    """
    m = re.search(r"KHP\s*(\d+)", str(name), re.IGNORECASE)
    if m:
        return int(m.group(1))
    return 0


# =============================================================================
# ESTADÍSTIQUES DE BASELINE
# =============================================================================
def baseline_stats(y, pct_low=10, pct_high=30, min_noise=0.01):
    """
    Calcula estadístiques de la baseline usant percentils.

    Selecciona els punts entre els percentils indicats (per defecte 10-30)
    per estimar la baseline sense pics ni soroll extrem.

    Args:
        y: Array de valors del senyal
        pct_low: Percentil inferior (default: 10)
        pct_high: Percentil superior (default: 30)
        min_noise: Soroll mínim (mAU) basat en precisió instrumental (default: 0.01)

    Returns:
        dict amb:
            - mean: mitjana de la baseline
            - std: desviació estàndard de la baseline (mínim min_noise)
            - threshold_3sigma: mean + 3*std (llindar per pics significatius)
    """
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]

    if len(y) < 10:
        return {"mean": 0.0, "std": min_noise, "threshold_3sigma": 3.0 * min_noise}

    p_low = np.percentile(y, pct_low)
    p_high = np.percentile(y, pct_high)

    # Seleccionar punts dins del rang de baseline
    mask = (y >= p_low) & (y <= p_high)
    baseline_points = y[mask]

    if len(baseline_points) < 5:
        # Fallback: usar percentil 10 com a baseline
        baseline_points = y[y <= p_high]

    if len(baseline_points) < 2:
        return {"mean": float(p_low), "std": min_noise, "threshold_3sigma": float(p_low) + 3.0 * min_noise}

    mean_val = float(np.mean(baseline_points))
    std_val = float(np.std(baseline_points))

    # Aplicar soroll mínim instrumental per evitar SNR artificials
    # DAD típic: precisió ~0.01 mAU
    std_val = max(std_val, min_noise)

    return {
        "mean": mean_val,
        "std": std_val,
        "threshold_3sigma": mean_val + 3.0 * std_val
    }


# =============================================================================
# DETECCIÓ DE BATMAN (DOBLE PIC)
# =============================================================================
# Paràmetres per defecte del detector Batman
BATMAN_MAX_SEP_MIN = 0.5       # Separació màxima entre pics (minuts)
BATMAN_DROP_MIN = 0.02         # Caiguda mínima entre pics (fracció)
BATMAN_DROP_MAX = 0.35         # Caiguda màxima entre pics (fracció)
BATMAN_MIN_PROMINENCE = 0.05   # Prominència mínima per detectar pics
BATMAN_DISTANCE = 15           # Distància mínima entre pics (punts)
BATMAN_MIN_HEIGHT_RANGE_PCT = 15.0  # Altura mínima com a % del rang (max-min)
BATMAN_MIN_SIGMA = 3.0         # Pics han d'estar N sigmes sobre baseline


def detect_batman(t, y, max_sep=None, drop_min=None, drop_max=None,
                  min_prominence=None, min_height_range_pct=None, min_sigma=None):
    """
    Detecta doble pic (Batman) en el senyal amb filtres robusts.

    Un doble pic es caracteritza per dos pics adjacents amb una vall
    entremig. Aquest patró indica problemes en la separació cromatogràfica.

    Filtres aplicats:
    1. Els pics han de tenir prominència mínima
    2. La separació temporal ha de ser petita (< max_sep minuts)
    3. La caiguda entre pics ha de ser significativa però no excessiva
    4. Els pics han d'estar almenys al min_height_range_pct del rang del senyal
    5. Els pics han d'estar almenys min_sigma sigmes per sobre de la baseline

    Args:
        t: Array de temps
        y: Array de valors
        max_sep: Separació màxima entre pics (minuts)
        drop_min: Caiguda mínima (fracció del pic)
        drop_max: Caiguda màxima (fracció del pic)
        min_prominence: Prominència mínima per find_peaks
        min_height_range_pct: Altura mínima com a % del rang
        min_sigma: Nombre de sigmes sobre baseline

    Returns:
        dict amb info del doble pic o None si no es detecta
    """
    # Usar valors per defecte si no s'especifiquen
    max_sep = max_sep if max_sep is not None else BATMAN_MAX_SEP_MIN
    drop_min = drop_min if drop_min is not None else BATMAN_DROP_MIN
    drop_max = drop_max if drop_max is not None else BATMAN_DROP_MAX
    min_prominence = min_prominence if min_prominence is not None else BATMAN_MIN_PROMINENCE
    min_height_range_pct = min_height_range_pct if min_height_range_pct is not None else BATMAN_MIN_HEIGHT_RANGE_PCT
    min_sigma = min_sigma if min_sigma is not None else BATMAN_MIN_SIGMA

    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    if t.size < 10 or y.size < 10:
        return None

    # Calcular estadístiques de baseline
    bl_stats = baseline_stats(y)
    bl_mean = bl_stats["mean"]
    bl_std = bl_stats["std"]
    threshold_sigma = bl_mean + min_sigma * bl_std

    # Calcular rang del senyal i llindar d'altura
    y_min = float(np.nanmin(y))
    y_max = float(np.nanmax(y))
    signal_range = y_max - y_min

    if signal_range < 1e-6:
        return None

    min_height = y_min + (min_height_range_pct / 100.0) * signal_range

    # Detectar pics
    peaks, props = find_peaks(y, prominence=min_prominence, distance=BATMAN_DISTANCE)

    if len(peaks) < 2:
        return None

    best = None
    best_drop = -np.inf

    for i in range(len(peaks) - 1):
        left = int(peaks[i])
        right = int(peaks[i + 1])

        if right <= left + 2:
            continue

        # Filtre 1: Separació temporal
        sep_min = float(t[right] - t[left])
        if sep_min > max_sep:
            continue

        # Filtre 2: Altura mínima (% del rang)
        avg_peak = float((y[left] + y[right]) / 2.0)
        if avg_peak < min_height:
            continue

        # Filtre 3: Pics han d'estar sobre el llindar de sigma
        if y[left] < threshold_sigma or y[right] < threshold_sigma:
            continue

        # Trobar la vall entre els dos pics
        seg = y[left:right + 1]
        idx_v = left + int(np.argmin(seg))
        val_v = float(y[idx_v])

        # Filtre 4: Caiguda dins del rang acceptable
        if avg_peak <= 0:
            continue

        drop_pct = float((avg_peak - val_v) / avg_peak)
        if drop_pct < drop_min or drop_pct > drop_max:
            continue

        # Guardar el millor candidat
        if drop_pct > best_drop:
            best_drop = drop_pct
            best = {
                "drop_pct": drop_pct,
                "t_left": float(t[left]),
                "t_right": float(t[right]),
                "t_valley": float(t[idx_v]),
                "height_left": float(y[left]),
                "height_right": float(y[right]),
                "height_valley": val_v,
                "separation_min": sep_min,
                "sigma_above_baseline": float((avg_peak - bl_mean) / bl_std) if bl_std > 0 else 0.0
            }

    return best


# =============================================================================
# UTILIDADES DE CÁLCULO
# =============================================================================
def mode_robust(data, bins=50):
    """
    Calcula la moda robusta de un array usando histograma.

    Args:
        data: Array de valores
        bins: Número de bins para el histograma

    Returns:
        float con el valor de la moda robusta
    """
    if data is None or len(data) == 0:
        return 0.0
    counts, edges = np.histogram(np.asarray(data), bins=bins)
    i = int(np.argmax(counts))
    return 0.5 * (edges[i] + edges[i + 1])


def t_at_max(t, y):
    """
    Obtiene el tiempo correspondiente al valor máximo.

    Args:
        t: Array de tiempos
        y: Array de valores

    Returns:
        float con el tiempo del máximo, o None si no es válido
    """
    if t is None or y is None or len(t) == 0 or len(y) == 0:
        return None
    if len(t) < 10 or len(y) < 10:
        return None
    i = int(np.nanargmax(y))
    return float(t[i])
