"""
hpsec_utils.py
==============
Funcions utilitàries compartides entre els scripts HPSEC.

Conté:
- baseline_stats: Estadístiques de baseline (mitjana, std, llindar 3σ)
- baseline_stats_windowed: Baseline amb finestres temporals
- seleccionar_carpeta: GUI per selecció de carpeta
- obtenir_seq: Extracció d'ID de seqüència
- is_khp: Detecció de mostres KHP
- extract_khp_conc: Extracció concentració KHP
- normalize_key: Normalització de strings per matching
- mode_robust: Càlcul moda robusta
- t_at_max: Temps al màxim

NOTA: detect_main_peak i detect_batman s'han mogut a hpsec_core.py (2026-01-29)
"""

import os
import re
import tkinter as tk
from tkinter import filedialog

import numpy as np


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


def baseline_stats_windowed(t, y, method="column", timeout_positions=None, config=None):
    """
    Calcula estadístiques de baseline usant finestres temporals específiques.

    Evita regions amb timeouts per obtenir estimacions de soroll consistents
    entre rèpliques.

    Args:
        t: Array de temps (minuts)
        y: Array de senyal (mAU)
        method: "column" o "bp" - determina quines finestres usar
        timeout_positions: Llista de posicions temporals (minuts) on hi ha timeouts
        config: ConfigManager instance (opcional, usa global si None)

    Returns:
        dict amb:
            - mean: mitjana de la baseline
            - std: desviació estàndard (mínim min_noise)
            - threshold_3sigma: mean + 3*std
            - window_used: nom de la finestra utilitzada (o "percentile_fallback")
    """
    # Carregar configuració
    if config is None:
        from hpsec_config import get_config
        config = get_config()

    baseline_cfg = config.get("baseline", default={})

    # Paràmetres
    timeout_margin = baseline_cfg.get("timeout_margin_min", 1.5)
    min_noise = baseline_cfg.get("min_noise_mau", 0.01)

    # Seleccionar finestres segons mètode
    if method.lower() == "bp":
        windows = baseline_cfg.get("windows_bp", [{"start": 5.0, "end": 10.0, "name": "post-peak"}])
    else:
        windows = baseline_cfg.get("windows_column", [
            {"start": 0.0, "end": 3.0, "name": "pre-peak"},
            {"start": 55.0, "end": 65.0, "name": "LMW-stable"}
        ])

    # Convertir a numpy
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    # Preparar llista de timeouts
    if timeout_positions is None:
        timeout_positions = []

    def window_has_timeout(w_start, w_end):
        """Comprova si una finestra conté algun timeout (amb marge)."""
        for t_pos in timeout_positions:
            # Timeout afecta si cau dins la finestra expandida pel marge
            if (t_pos - timeout_margin) < w_end and (t_pos + timeout_margin) > w_start:
                return True
        return False

    def get_window_data(w_start, w_end):
        """Extreu dades dins una finestra temporal."""
        mask = (t >= w_start) & (t <= w_end)
        return y[mask]

    # Intentar cada finestra en ordre
    for window in windows:
        w_start = window.get("start", 0)
        w_end = window.get("end", 10)
        w_name = window.get("name", f"{w_start}-{w_end}")

        # Verificar que la finestra no tingui timeout
        if window_has_timeout(w_start, w_end):
            continue

        # Verificar que tenim prou dades
        window_data = get_window_data(w_start, w_end)
        if len(window_data) < 10:
            continue

        # Calcular estadístiques de la finestra
        # Filtrar valors finits
        window_data = window_data[np.isfinite(window_data)]
        if len(window_data) < 5:
            continue

        mean_val = float(np.mean(window_data))
        std_val = float(np.std(window_data))

        # Aplicar soroll mínim
        std_val = max(std_val, min_noise)

        return {
            "mean": mean_val,
            "std": std_val,
            "threshold_3sigma": mean_val + 3.0 * std_val,
            "window_used": w_name
        }

    # Fallback: mètode percentil original
    pct_low = baseline_cfg.get("fallback_percentile_low", 10)
    pct_high = baseline_cfg.get("fallback_percentile_high", 30)

    result = baseline_stats(y, pct_low=pct_low, pct_high=pct_high, min_noise=min_noise)
    result["window_used"] = "percentile_fallback"

    return result


# =============================================================================
# UTILIDADES DE CÁLCULO
# =============================================================================
# NOTA: detect_batman s'ha mogut a hpsec_core.py (2026-01-29)
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
