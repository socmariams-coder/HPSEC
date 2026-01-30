"""
hpsec_process.py - Mòdul de processament de dades HPSEC (Fase 3)
================================================================

FASE 3 del pipeline de 5 fases:
- Aplicar alineació temporal (shifts calculats a Fase 2)
- Detecció de pics i timeouts
- Càlcul d'àrees per fraccions de temps
- Càlcul de SNR, LOD, LOQ

NOTA: La correcció de baseline es fa a Fase 1 (Import).
Si les dades inclouen y_net, s'usa directament.
Si no, es calcula baseline aquí per compatibilitat.

REQUEREIX:
- Fase 1: import_sequence() → dades amb y_net (baseline ja restada)
- Fase 2: calibrate_sequence() → shifts d'alineació

NO fa:
- Lectura de fitxers (Fase 1: IMPORTAR)
- Validació KHP (Fase 2: CALIBRAR)
- Selecció de rèpliques (Fase 4: REVISAR)
- Escriptura Excel finals (Fase 5: EXPORTAR)

Usat per HPSEC_Suite.py
"""

__version__ = "1.1.0"
__version_date__ = "2026-01-30"

import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
from scipy.integrate import trapezoid

# Import funcions de detecció des de hpsec_core (Single Source of Truth)
from hpsec_core import (
    detect_timeout,
    format_timeout_status,
    TIMEOUT_CONFIG,
    calc_snr,
    detect_batman,
    detect_main_peak,
    find_peak_boundaries,
)
from hpsec_utils import baseline_stats, baseline_stats_windowed, mode_robust
from hpsec_config import get_config


# =============================================================================
# CONFIGURACIÓ PER DEFECTE
# =============================================================================
DEFAULT_PROCESS_CONFIG = {
    # Baseline
    "bp_baseline_win": 1.0,          # Finestra baseline per BP (min)
    "col_baseline_start": 10.0,      # Inici finestra baseline per COLUMN (min)
    # DAD
    "target_wavelengths": [220, 252, 254, 272, 290, 362],
    "dad_subsample": 5,              # Submostreig DAD (cada N punts)
    # Peak detection
    "peak_min_prominence_pct": 5.0,  # Prominència mínima (% del màxim)
    # Límit temporal
    "max_time_min": 70.0,            # Temps màxim per truncar cromatogrames
    # Fraccions de temps per integració parcial (Column mode)
    "time_fractions": {
        "BioP": [0, 18],
        "HS": [18, 23],
        "BB": [23, 30],
        "SB": [30, 40],
        "LMW": [40, 70],
    },
}


# =============================================================================
# FUNCIONS UTILITAT
# =============================================================================
def truncate_chromatogram(t, y, max_time_min=None):
    """
    Trunca cromatograma a un temps màxim.

    Args:
        t: Array de temps en minuts
        y: Array de senyal (o llista d'arrays)
        max_time_min: Temps màxim (defecte: 70 min)

    Returns:
        t_trunc, y_trunc (o llista de y_trunc si y és llista)
    """
    if max_time_min is None:
        max_time_min = DEFAULT_PROCESS_CONFIG.get("max_time_min", 70.0)

    t = np.asarray(t)
    mask = t <= max_time_min

    if isinstance(y, (list, tuple)):
        return t[mask], [np.asarray(yi)[mask] if yi is not None else None for yi in y]
    else:
        y = np.asarray(y)
        return t[mask], y[mask]


# NOTA: mode_robust s'ha mogut a hpsec_utils.py (2026-01-29) - importat a dalt


# =============================================================================
# FUNCIONS BASELINE I SMOOTHING
# =============================================================================
def get_baseline_correction(t, y, mode_type="COL", config=None, use_end=False):
    """
    Correcció de baseline segons mode BP o COLUMN.

    Args:
        t: array de temps (min)
        y: array de senyal
        mode_type: "BP" o "COL"
        config: configuració
        use_end: si True, usa el final del cromatograma per baseline (recomanat per BP Direct)

    Returns:
        Array amb el valor de baseline (constant)
    """
    config = config or DEFAULT_PROCESS_CONFIG
    t = np.asarray(t)
    y = np.asarray(y)

    if mode_type == "BP":
        if use_end:
            # Per BP Direct: usar els últims punts (després del pic)
            n = len(y)
            n_edge = max(20, n // 5)  # Últim 20%
            val = float(np.median(y[-n_edge:]))
        else:
            # Per BP UIB: usar primers punts
            mask = t < config["bp_baseline_win"]
            val = mode_robust(y[mask]) if np.sum(mask) > 10 else float(np.nanmin(y))
        return np.full_like(y, val, dtype=float)

    # COLUMN: usar primers punts
    mask = t < config["col_baseline_start"]
    val = mode_robust(y[mask]) if np.sum(mask) > 10 else float(np.nanmin(y))
    return np.full_like(y, val, dtype=float)


def apply_smoothing(y, window_length=11, polyorder=3):
    """
    Aplica suavitzat Savgol.

    Args:
        y: Array de senyal
        window_length: Longitud de la finestra (imparell)
        polyorder: Ordre del polinomi

    Returns:
        Array suavitzat
    """
    y = np.asarray(y)
    if len(y) < window_length:
        return y
    return savgol_filter(y, window_length, polyorder)


# =============================================================================
# FUNCIONS ALINEACIÓ TEMPORAL
# =============================================================================
def align_signals_by_max(t_ref, y_ref, t_other, y_other):
    """
    Alinea dos senyals pel màxim i interpola el segon a l'escala de temps del primer.

    Args:
        t_ref: temps de referència
        y_ref: senyal de referència
        t_other: temps del senyal a alinear
        y_other: senyal a alinear

    Returns:
        y_aligned: senyal alineat i interpolat a t_ref
        shift: desplaçament aplicat (minuts)
    """
    t_ref = np.asarray(t_ref)
    y_ref = np.asarray(y_ref)
    t_other = np.asarray(t_other)
    y_other = np.asarray(y_other)

    # Trobar màxims
    idx_max_ref = np.argmax(y_ref)
    idx_max_other = np.argmax(y_other)

    t_max_ref = t_ref[idx_max_ref]
    t_max_other = t_other[idx_max_other]

    # Calcular shift necessari
    shift = t_max_ref - t_max_other

    # Aplicar shift al temps
    t_other_shifted = t_other + shift

    # Interpolar a l'escala de temps de referència
    y_aligned = np.interp(t_ref, t_other_shifted, y_other, left=0, right=0)

    return y_aligned, shift


def apply_shift(t_ref, t_signal, y_signal, shift):
    """
    Aplica un shift temporal i interpola a l'escala de referència.

    Args:
        t_ref: escala de temps de referència
        t_signal: escala de temps del senyal
        y_signal: senyal a desplaçar
        shift: desplaçament en minuts (positiu = avançar, negatiu = retardar)

    Returns:
        y_shifted: senyal desplaçat i interpolat a t_ref
    """
    t_ref = np.asarray(t_ref)
    t_signal = np.asarray(t_signal)
    y_signal = np.asarray(y_signal)

    t_shifted = t_signal + shift
    y_shifted = np.interp(t_ref, t_shifted, y_signal, left=0, right=0)
    return y_shifted


# =============================================================================
# FUNCIONS PROCESSAMENT DAD
# =============================================================================
def process_dad(df_dad, config=None):
    """
    Processa DAD: extreu wavelengths i submostreig.

    Args:
        df_dad: DataFrame amb columnes 'time (min)' i wavelengths
        config: Configuració

    Returns:
        DataFrame processat amb wavelengths d'interès i submostreig
    """
    config = config or DEFAULT_PROCESS_CONFIG
    if df_dad is None or df_dad.empty:
        return pd.DataFrame()

    target_wls = config["target_wavelengths"]
    cols_to_keep = ["time (min)"]

    for wl in target_wls:
        wl_str = str(wl)
        if wl_str in df_dad.columns:
            cols_to_keep.append(wl_str)

    if len(cols_to_keep) == 1:
        return pd.DataFrame()

    df_filtered = df_dad[cols_to_keep].copy()

    subsample = config["dad_subsample"]
    indices = [0] + list(range(subsample, len(df_filtered), subsample))
    df_sub = df_filtered.iloc[indices].reset_index(drop=True)

    return df_sub


# =============================================================================
# FUNCIONS DETECCIÓ PICS
# =============================================================================
# NOTA: find_peak_boundaries i detect_main_peak s'importen de hpsec_core.py
# Eliminades versions locals per evitar duplicació (2026-01-29)


# =============================================================================
# FUNCIONS CÀLCUL ÀREES
# =============================================================================
def calcular_fraccions_temps(t, y, config=None):
    """
    Calcula àrees per fraccions de temps (integració parcial).

    Args:
        t: Array de temps (minuts)
        y: Array de senyal (mAU, ja amb baseline restada)
        config: Configuració amb time_fractions

    Returns:
        Dict amb àrees per fracció: {BioP, HS, BB, SB, LMW, total, *_pct}
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(t) < 2 or len(y) < 2:
        return {"total": 0.0}

    # Assegurar que y no té valors negatius (baseline ja restada)
    y_clean = np.maximum(y, 0)

    # Àrea total del cromatograma
    kpis = {"total": float(trapezoid(y_clean, t))}

    # Obtenir fraccions de la config
    if config is None:
        config = DEFAULT_PROCESS_CONFIG
    fractions = config.get("time_fractions", DEFAULT_PROCESS_CONFIG["time_fractions"])

    # Calcular àrea per cada fracció
    for nom, (t_ini, t_fi) in fractions.items():
        mask = (t >= t_ini) & (t < t_fi)
        if np.sum(mask) > 1:
            kpis[nom] = float(trapezoid(y_clean[mask], t[mask]))
        else:
            kpis[nom] = 0.0

    # Calcular percentatges
    if kpis["total"] > 0:
        for nom in fractions.keys():
            kpis[f"{nom}_pct"] = 100.0 * kpis[nom] / kpis["total"]
    else:
        for nom in fractions.keys():
            kpis[f"{nom}_pct"] = 0.0

    return kpis


def detectar_tmax_senyals(t_doc, y_doc, df_dad, config=None):
    """
    Detecta el temps de retenció (tmax) per DOC i cada longitud d'ona DAD.

    Args:
        t_doc: Array temps DOC
        y_doc: Array senyal DOC (net, amb baseline restada)
        df_dad: DataFrame DAD amb columnes 'time (min)' i wavelengths
        config: Configuració

    Returns:
        Dict amb tmax per cada senyal: {DOC: x, A220: y, A254: z, ...}
    """
    if config is None:
        config = DEFAULT_PROCESS_CONFIG
    target_wls = config.get('target_wavelengths', [220, 252, 254, 272, 290, 362])

    result = {"DOC": 0.0}
    for wl in target_wls:
        result[f"A{wl}"] = 0.0

    # tmax DOC
    if t_doc is not None and y_doc is not None and len(t_doc) > 10:
        t_doc = np.asarray(t_doc).flatten()
        y_doc = np.asarray(y_doc).flatten()
        idx_max = np.argmax(y_doc)
        result["DOC"] = float(t_doc[idx_max])

    # tmax per cada wavelength DAD
    if df_dad is not None and not df_dad.empty and 'time (min)' in df_dad.columns:
        t_dad = pd.to_numeric(df_dad['time (min)'], errors='coerce').to_numpy()

        for wl in target_wls:
            wl_str = str(wl)
            if wl_str in df_dad.columns:
                y_wl = pd.to_numeric(df_dad[wl_str], errors='coerce').to_numpy()
                if len(y_wl) > 10 and not np.all(np.isnan(y_wl)):
                    # Restar baseline (percentil 5)
                    baseline = np.nanpercentile(y_wl, 5)
                    y_wl_net = y_wl - baseline
                    idx_max = np.nanargmax(y_wl_net)
                    result[f"A{wl}"] = float(t_dad[idx_max])

    return result


def calcular_arees_fraccions_complet(t_doc, y_doc, df_dad, config=None):
    """
    Calcula àrees per fraccions de temps per DOC i totes les wavelengths DAD.

    Args:
        t_doc: Array temps DOC
        y_doc: Array senyal DOC (net)
        df_dad: DataFrame DAD
        config: Configuració

    Returns:
        Dict amb estructura:
        {
            "DOC": {BioP: x, HS: y, ..., total: z},
            "A220": {BioP: x, HS: y, ..., total: z},
            ...
        }
    """
    if config is None:
        config = DEFAULT_PROCESS_CONFIG
    target_wls = config.get('target_wavelengths', [220, 252, 254, 272, 290, 362])

    result = {}

    # Fraccions DOC
    if t_doc is not None and y_doc is not None and len(t_doc) > 10:
        result["DOC"] = calcular_fraccions_temps(t_doc, y_doc, config)
    else:
        result["DOC"] = {"total": 0.0}

    # Fraccions per cada wavelength DAD
    if df_dad is not None and not df_dad.empty and 'time (min)' in df_dad.columns:
        t_dad = pd.to_numeric(df_dad['time (min)'], errors='coerce').to_numpy()

        for wl in target_wls:
            wl_str = str(wl)
            if wl_str in df_dad.columns:
                y_wl = pd.to_numeric(df_dad[wl_str], errors='coerce').to_numpy()
                if len(y_wl) > 10 and not np.all(np.isnan(y_wl)):
                    # Restar baseline
                    baseline = np.nanpercentile(y_wl, 5)
                    y_wl_net = np.maximum(y_wl - baseline, 0)
                    result[f"A{wl}"] = calcular_fraccions_temps(t_dad, y_wl_net, config)
                else:
                    result[f"A{wl}"] = {"total": 0.0}
            else:
                result[f"A{wl}"] = {"total": 0.0}
    else:
        for wl in target_wls:
            result[f"A{wl}"] = {"total": 0.0}

    return result


def analyze_sample_areas(t_doc, y_doc, df_dad, peak_info, config=None):
    """
    Analitza una mostra i calcula totes les àrees (DOC + DAD wavelengths).

    Args:
        t_doc: Array temps DOC
        y_doc: Array senyal DOC (net)
        df_dad: DataFrame DAD
        peak_info: Dict amb info del pic (output de detect_main_peak)
        config: Configuració

    Returns:
        dict amb:
            - doc_area: Àrea DOC
            - doc_t_retention: Temps de retenció
            - doc_t_start, doc_t_end: Límits del pic
            - a{wl}_area: Àrea per cada wavelength
            - dad_wavelengths: Dict amb àrees per wavelength
            - valid: True si s'ha pogut calcular
    """
    target_wls = config.get('target_wavelengths', [220, 252, 254, 272, 290, 362]) if config else [220, 252, 254, 272, 290, 362]

    result = {
        'doc_area': 0.0,
        'doc_t_retention': 0.0,
        'doc_t_start': 0.0,
        'doc_t_end': 0.0,
        'dad_wavelengths': {},
        'valid': False,
    }
    for wl in target_wls:
        result[f'a{wl}_area'] = 0.0

    if t_doc is None or y_doc is None or len(t_doc) < 10:
        return result

    t_doc = np.asarray(t_doc).flatten()
    y_doc = np.asarray(y_doc).flatten()

    if peak_info and peak_info.get('valid'):
        result['doc_area'] = peak_info.get('area', 0.0)
        result['doc_t_retention'] = peak_info.get('t_max', 0.0)
        result['doc_t_start'] = peak_info.get('t_start', 0.0)
        result['doc_t_end'] = peak_info.get('t_end', 0.0)
        result['valid'] = True

        t_start = result['doc_t_start']
        t_end = result['doc_t_end']

        if df_dad is not None and not df_dad.empty and 'time (min)' in df_dad.columns:
            t_dad = pd.to_numeric(df_dad['time (min)'], errors='coerce').to_numpy()

            if t_start > 0 and t_end > t_start:
                dad_left_idx = int(np.searchsorted(t_dad, t_start))
                dad_right_idx = int(np.searchsorted(t_dad, t_end))

                dad_left_idx = max(0, min(dad_left_idx, len(t_dad) - 1))
                dad_right_idx = max(0, min(dad_right_idx, len(t_dad) - 1))

                if dad_right_idx > dad_left_idx:
                    for col in df_dad.columns:
                        if col == 'time (min)':
                            continue
                        try:
                            wl_str = str(col)
                            y_wl = pd.to_numeric(df_dad[col], errors='coerce').to_numpy()
                            if len(y_wl) > dad_right_idx:
                                area_wl = float(trapezoid(
                                    y_wl[dad_left_idx:dad_right_idx+1],
                                    t_dad[dad_left_idx:dad_right_idx+1]
                                ))
                                try:
                                    wl_int = int(wl_str)
                                    if wl_int in target_wls:
                                        result[f'a{wl_int}_area'] = area_wl
                                except ValueError:
                                    pass
                                result['dad_wavelengths'][wl_str] = area_wl
                        except Exception:
                            continue

    return result


# =============================================================================
# CÀLCUL SNR I BASELINE NOISE
# =============================================================================
def calculate_snr_info(y_doc_net, peak_info, y_doc_uib=None,
                       t_min=None, method="column", timeout_positions=None):
    """
    Calcula SNR, LOD, LOQ i baseline noise per DOC Direct i UIB.

    Usa finestres temporals específiques per evitar regions amb timeouts,
    obtenint estimacions de soroll consistents entre rèpliques.

    Args:
        y_doc_net: Senyal DOC net (Direct)
        peak_info: Diccionari amb info del pic (height)
        y_doc_uib: Senyal DOC UIB (opcional, per DUAL)
        t_min: Array de temps en minuts (opcional, per càlcul windowed)
        method: "column" o "bp" - determina finestres de baseline
        timeout_positions: Llista de posicions temporals (min) dels timeouts

    Returns:
        dict amb:
            - snr_direct: SNR del senyal Direct
            - baseline_noise_direct: Desviació estàndard baseline Direct (mAU)
            - lod_direct: Limit of Detection = 3 × noise (mAU)
            - loq_direct: Limit of Quantification = 10 × noise (mAU)
            - baseline_window_direct: Finestra usada per calcular baseline
            - snr_uib, baseline_noise_uib, lod_uib, loq_uib (si DUAL)
    """
    result = {}

    # Determinar si podem usar el mètode windowed
    use_windowed = (t_min is not None and len(t_min) > 10)

    # Direct
    if y_doc_net is not None and len(y_doc_net) > 10:
        if use_windowed:
            bl_stats = baseline_stats_windowed(
                t_min, y_doc_net,
                method=method,
                timeout_positions=timeout_positions
            )
            result["baseline_window_direct"] = bl_stats.get("window_used", "unknown")
        else:
            bl_stats = baseline_stats(y_doc_net)
            result["baseline_window_direct"] = "percentile"

        noise_direct = bl_stats.get("std", 0.0)
        result["baseline_noise_direct"] = noise_direct
        result["lod_direct"] = 3.0 * noise_direct
        result["loq_direct"] = 10.0 * noise_direct

        if peak_info and peak_info.get("valid") and peak_info.get("height", 0) > 0:
            height = peak_info["height"]
            if noise_direct > 0:
                result["snr_direct"] = height / noise_direct
            else:
                result["snr_direct"] = calc_snr(y_doc_net, height)

    # UIB
    if y_doc_uib is not None and len(y_doc_uib) > 10:
        if use_windowed:
            bl_stats_uib = baseline_stats_windowed(
                t_min, y_doc_uib,
                method=method,
                timeout_positions=timeout_positions
            )
            result["baseline_window_uib"] = bl_stats_uib.get("window_used", "unknown")
        else:
            bl_stats_uib = baseline_stats(y_doc_uib)
            result["baseline_window_uib"] = "percentile"

        noise_uib = bl_stats_uib.get("std", 0.0)
        result["baseline_noise_uib"] = noise_uib
        result["lod_uib"] = 3.0 * noise_uib
        result["loq_uib"] = 10.0 * noise_uib

        if peak_info and peak_info.get("valid"):
            # Usar el màxim de UIB com a alçada del pic
            height_uib = float(np.max(y_doc_uib))
            if noise_uib > 0:
                result["snr_uib"] = height_uib / noise_uib
            else:
                result["snr_uib"] = calc_snr(y_doc_uib, height_uib)

    return result


# =============================================================================
# PROCESSAMENT D'UNA MOSTRA
# =============================================================================
def process_sample(sample_data, calibration_data=None, config=None):
    """
    Processa una mostra individual: alineació, pics, àrees.

    NOTA: Si les dades inclouen y_net (baseline ja restada per import),
    s'usa directament. Si no, es calcula baseline aquí (compatibilitat).

    Args:
        sample_data: Dict amb dades de la mostra (de import_sequence):
            - name: Nom de la mostra
            - replica: Número de rèplica
            - t_doc: Array de temps DOC
            - y_doc_direct: Senyal DOC Direct RAW (si DUAL)
            - y_doc_direct_net: Senyal DOC Direct NET (si disponible)
            - y_doc_uib: Senyal DOC UIB RAW (si DUAL)
            - y_doc_uib_net: Senyal DOC UIB NET (si disponible)
            - y_doc: Senyal DOC RAW (si simple)
            - y_doc_net: Senyal DOC NET (si disponible)
            - df_dad: DataFrame DAD
        calibration_data: Dict amb dades de calibració (de calibrate_sequence):
            - shift_uib: Shift per DOC_UIB (minuts)
            - shift_direct: Shift per DOC_Direct (minuts)
        config: Configuració

    Returns:
        dict amb:
            - name, replica: Identificació
            - t_doc, y_doc_net: Dades processades
            - peak_info: Info del pic principal
            - areas: Dict amb àrees per fraccions
            - tmax_signals: Dict amb tmax per senyal
            - snr_info: Dict amb SNR, LOD, LOQ
            - timeout_info: Info de timeouts
            - anomalies: Llista d'anomalies detectades
    """
    config = config or DEFAULT_PROCESS_CONFIG

    result = {
        "name": sample_data.get("name", "UNKNOWN"),
        "replica": sample_data.get("replica", "1"),
        "processed": False,
        "anomalies": [],
    }

    # Obtenir dades RAW
    t_doc = sample_data.get("t_doc")
    t_doc_uib = sample_data.get("t_doc_uib")  # Temps UIB (pot ser diferent de t_doc)
    y_doc_direct = sample_data.get("y_doc_direct")
    y_doc_uib = sample_data.get("y_doc_uib")
    y_doc = sample_data.get("y_doc")  # Mode simple
    df_dad = sample_data.get("df_dad")

    # Obtenir dades NET (si disponibles, calculades per import)
    y_doc_direct_net_precomp = sample_data.get("y_doc_direct_net")
    y_doc_uib_net_precomp = sample_data.get("y_doc_uib_net")
    y_doc_net_precomp = sample_data.get("y_doc_net")

    # Determinar mode (DUAL vs simple)
    is_dual = y_doc_direct is not None and y_doc_uib is not None

    if t_doc is None or (y_doc is None and not is_dual):
        result["error"] = "Missing DOC data"
        return result

    t_doc = np.asarray(t_doc).flatten()

    # NO truncar - mantenir dades completes per anàlisi post-run
    # Les visualitzacions limitaran a 70 min però les dades es preserven
    if is_dual:
        y_doc_direct = np.asarray(y_doc_direct).flatten() if y_doc_direct is not None else None
        y_doc_uib = np.asarray(y_doc_uib).flatten() if y_doc_uib is not None else None
        if y_doc_direct_net_precomp is not None:
            y_doc_direct_net_precomp = np.asarray(y_doc_direct_net_precomp).flatten()
        if y_doc_uib_net_precomp is not None:
            y_doc_uib_net_precomp = np.asarray(y_doc_uib_net_precomp).flatten()
    else:
        y_doc = np.asarray(y_doc).flatten() if y_doc is not None else None
        if y_doc_net_precomp is not None:
            y_doc_net_precomp = np.asarray(y_doc_net_precomp).flatten()

    # Detectar mode BP vs COLUMN
    t_max_chromato = float(np.max(t_doc))
    is_bp = t_max_chromato < 20
    mode_type = "BP" if is_bp else "COL"

    # Aplicar shifts d'alineació (si disponibles)
    # NOTA: Els shifts s'apliquen als senyals RAW, després es recalcula baseline si cal
    shifts_applied = False
    if calibration_data:
        shift_uib = calibration_data.get("shift_uib", 0.0)
        shift_direct = calibration_data.get("shift_direct", 0.0)

        if is_dual:
            # UIB: SEMPRE interpolar a escala t_doc (referencia Direct)
            # Això assegura que tots els arrays tinguin la mateixa longitud
            uib_interpolated = False
            if t_doc_uib is not None and len(y_doc_uib) > 0:
                t_uib_arr = np.asarray(t_doc_uib).flatten()
                if abs(shift_uib) > 0.001:
                    y_doc_uib = apply_shift(t_doc, t_uib_arr, y_doc_uib, shift_uib)
                    shifts_applied = True
                elif len(t_uib_arr) != len(t_doc):
                    # Sense shift pero cal interpolar UIB a escala Direct
                    y_doc_uib = np.interp(t_doc, t_uib_arr, y_doc_uib, left=0, right=0)
                uib_interpolated = True
            elif len(y_doc_uib) != len(t_doc):
                # t_doc_uib no disponible pero longitud diferent - no podem processar UIB
                y_doc_uib = None
                y_doc_uib_net_precomp = None
            # Si UIB s'ha interpolat, invalidar y_net precomp (calculat amb temps original)
            if uib_interpolated:
                y_doc_uib_net_precomp = None
            # Direct: ja esta a t_doc
            if abs(shift_direct) > 0.001:
                y_doc_direct = apply_shift(t_doc, t_doc, y_doc_direct, shift_direct)
                shifts_applied = True
        else:
            shift = calibration_data.get("shift", 0.0)
            if abs(shift) > 0.001:
                y_doc = apply_shift(t_doc, t_doc, y_doc, shift)
                shifts_applied = True

    # Correcció de baseline
    # Usar y_net precalculat si disponible I no s'han aplicat shifts
    # (els shifts invaliden el y_net precalculat)
    if is_dual:
        if y_doc_direct_net_precomp is not None and not shifts_applied:
            y_doc_direct_net = y_doc_direct_net_precomp
        else:
            baseline_direct = get_baseline_correction(t_doc, y_doc_direct, mode_type, config, use_end=is_bp)
            y_doc_direct_net = y_doc_direct - baseline_direct

        if y_doc_uib is not None and len(y_doc_uib) > 0:
            if y_doc_uib_net_precomp is not None and not shifts_applied and len(y_doc_uib_net_precomp) == len(t_doc):
                y_doc_uib_net = y_doc_uib_net_precomp
            else:
                baseline_uib = get_baseline_correction(t_doc, y_doc_uib, mode_type, config, use_end=False)
                y_doc_uib_net = y_doc_uib - baseline_uib
        else:
            y_doc_uib_net = None

        # Per processament principal, usar Direct
        y_doc_net = y_doc_direct_net
    else:
        if y_doc_net_precomp is not None and not shifts_applied:
            y_doc_net = y_doc_net_precomp
        else:
            baseline = get_baseline_correction(t_doc, y_doc, mode_type, config)
            y_doc_net = y_doc - baseline

    # Detectar timeouts
    timeout_info = detect_timeout(t_doc)
    timeout_positions = timeout_info.get("t_positions", [])

    if timeout_info.get("n_timeouts", 0) > 0:
        result["timeout_info"] = timeout_info

    # Detectar pic principal
    y_smooth = apply_smoothing(y_doc_net)
    peak_info = detect_main_peak(t_doc, y_smooth, config.get("peak_min_prominence_pct", 5.0), is_bp=is_bp)

    if not peak_info.get("valid"):
        result["anomalies"].append("NO_PEAK")
    else:
        # Detectar Batman
        batman_result = detect_batman(t_doc, y_smooth)
        if batman_result.get("is_batman"):
            result["anomalies"].append("BATMAN")

    result["peak_info"] = peak_info

    # Calcular àrees per fraccions
    if not is_bp:
        areas = calcular_arees_fraccions_complet(t_doc, y_doc_net, df_dad, config)
    else:
        # Per BP, només àrea total
        areas = {"DOC": calcular_fraccions_temps(t_doc, y_doc_net, config)}

    result["areas"] = areas

    # Detectar tmax senyals
    tmax_signals = detectar_tmax_senyals(t_doc, y_doc_net, df_dad, config)
    result["tmax_signals"] = tmax_signals

    # Calcular SNR info
    snr_info = calculate_snr_info(
        y_doc_net,
        peak_info,
        y_doc_uib=y_doc_uib_net if is_dual else None,
        t_min=t_doc,
        method="bp" if is_bp else "column",
        timeout_positions=timeout_positions
    )
    result["snr_info"] = snr_info

    # Guardar dades processades
    result["t_doc"] = t_doc
    result["y_doc_net"] = y_doc_net
    if is_dual:
        result["y_doc_direct_net"] = y_doc_direct_net
        result["y_doc_uib_net"] = y_doc_uib_net
    result["df_dad"] = df_dad
    result["is_bp"] = is_bp
    result["is_dual"] = is_dual
    result["processed"] = True

    return result


# =============================================================================
# HELPER: APLANAR ESTRUCTURA DE MOSTRES
# =============================================================================
def _flatten_samples_for_processing(imported_data, data_mode="DUAL"):
    """
    Converteix l'estructura nested de import_sequence a llista plana per process_sample.

    Args:
        imported_data: Dict retornat per import_sequence()
        data_mode: "DUAL", "UIB" o "DIRECT"

    Returns:
        tuple de 3 llistes: (samples, khp_samples, control_samples)
        Cada element és un dict amb:
            - name: Nom de la mostra
            - replica: Número de rèplica
            - t_doc: Array de temps
            - y_doc_direct / y_doc_uib / y_doc: Arrays de senyal RAW segons mode
            - y_doc_net / y_doc_uib_net: Arrays amb baseline restada (si disponible)
            - baseline_uib: Valor de baseline aplicat (si disponible)
            - df_dad: DataFrame DAD (si disponible)

    Nota: En mode DUAL, si direct no està disponible (encara no extret del master),
          s'usa UIB com a senyal principal per permetre processar igualment.
    """
    samples = []
    khp_samples = []
    control_samples = []

    all_samples = imported_data.get("samples", {})

    for sample_name, sample_info in all_samples.items():
        sample_type = sample_info.get("type", "SAMPLE")
        replicas = sample_info.get("replicas", {})

        for rep_num, rep_data in replicas.items():
            flat_sample = {
                "name": sample_name,
                "replica": str(rep_num),
                "sample_type": sample_type,
            }

            # Extreure dades segons data_mode (DUAL, DIRECT, UIB)
            uib = rep_data.get("uib", {})
            direct = rep_data.get("direct", {})
            has_uib = uib and "t" in uib and "y" in uib
            has_direct = direct and "t" in direct and "y" in direct

            if data_mode == "DUAL":
                # Mode DUAL: ambdos senyals separats, cada un amb el seu temps
                if has_direct:
                    flat_sample["t_doc"] = direct["t"]  # Temps principal (referencia)
                    flat_sample["t_doc_direct"] = direct["t"]
                    flat_sample["y_doc_direct"] = direct["y"]
                    if "y_net" in direct:
                        flat_sample["y_doc_direct_net"] = direct["y_net"]
                    if "baseline" in direct:
                        flat_sample["baseline_direct"] = direct["baseline"]

                if has_uib:
                    flat_sample["t_doc_uib"] = uib["t"]  # Temps propi UIB
                    if "t_doc" not in flat_sample:
                        flat_sample["t_doc"] = uib["t"]
                    flat_sample["y_doc_uib"] = uib["y"]
                    if "y_net" in uib:
                        flat_sample["y_doc_uib_net"] = uib["y_net"]
                    if "baseline" in uib:
                        flat_sample["baseline_uib"] = uib["baseline"]

                # Fallback DUAL: si nomes hi ha un senyal, convertir a mode simple
                if has_uib and not has_direct:
                    flat_sample["y_doc"] = uib["y"]
                    if "y_net" in uib:
                        flat_sample["y_doc_net"] = uib["y_net"]
                    if "baseline" in uib:
                        flat_sample["baseline"] = uib["baseline"]
                    flat_sample.pop("y_doc_uib", None)
                    flat_sample.pop("y_doc_uib_net", None)
                    flat_sample.pop("baseline_uib", None)
                elif has_direct and not has_uib:
                    flat_sample["y_doc"] = direct["y"]
                    if "y_net" in direct:
                        flat_sample["y_doc_net"] = direct["y_net"]
                    if "baseline" in direct:
                        flat_sample["baseline"] = direct["baseline"]
                    flat_sample.pop("y_doc_direct", None)
                    flat_sample.pop("y_doc_direct_net", None)
                    flat_sample.pop("baseline_direct", None)

            elif data_mode == "DIRECT":
                # Mode DIRECT: nomes usar Direct (ignorar UIB)
                if has_direct:
                    flat_sample["t_doc"] = direct["t"]
                    flat_sample["y_doc"] = direct["y"]
                    if "y_net" in direct:
                        flat_sample["y_doc_net"] = direct["y_net"]
                    if "baseline" in direct:
                        flat_sample["baseline"] = direct["baseline"]

            elif data_mode == "UIB":
                # Mode UIB: nomes usar UIB (ignorar Direct)
                if has_uib:
                    flat_sample["t_doc"] = uib["t"]
                    flat_sample["y_doc"] = uib["y"]
                    if "y_net" in uib:
                        flat_sample["y_doc_net"] = uib["y_net"]
                    if "baseline" in uib:
                        flat_sample["baseline"] = uib["baseline"]

            # Extreure dades DAD
            dad = rep_data.get("dad", {})
            if dad and "df" in dad:
                flat_sample["df_dad"] = dad["df"]

            # Classificar segons tipus
            if sample_type == "KHP":
                khp_samples.append(flat_sample)
            elif sample_type == "CONTROL":
                control_samples.append(flat_sample)
            else:
                samples.append(flat_sample)

    return samples, khp_samples, control_samples


# =============================================================================
# FUNCIÓ PRINCIPAL: PROCESSAR SEQÜÈNCIA
# =============================================================================
def process_sequence(imported_data, calibration_data=None, config=None, progress_callback=None):
    """
    FASE 3: Processa totes les mostres d'una seqüència.

    Args:
        imported_data: Dict retornat per import_sequence() (Fase 1)
        calibration_data: Dict retornat per calibrate_sequence() (Fase 2)
            - shift_uib: Shift per DOC_UIB (minuts)
            - shift_direct: Shift per DOC_Direct (minuts)
        config: Configuració
        progress_callback: Funció callback per reportar progrés

    Returns:
        dict amb:
            - success: True si s'ha processat correctament
            - seq_name: Nom de la seqüència
            - method: "BP" o "COLUMN"
            - samples: Llista de mostres processades
            - khp_samples: Llista de KHP processats
            - control_samples: Llista de controls processats
            - errors: Llista d'errors
            - warnings: Llista d'avisos
            - summary: Resum estadístic
    """
    config = config or DEFAULT_PROCESS_CONFIG

    result = {
        "success": False,
        "seq_name": imported_data.get("seq_name", "UNKNOWN"),
        "seq_path": imported_data.get("seq_path", ""),
        "method": imported_data.get("method", "UNKNOWN"),
        "data_mode": imported_data.get("data_mode", "UNKNOWN"),
        "samples": [],
        "khp_samples": [],
        "control_samples": [],
        "errors": [],
        "warnings": [],
    }

    # Verificar dades d'entrada
    if not imported_data.get("success"):
        result["errors"].append("Imported data is invalid")
        return result

    # Aplanar l'estructura de mostres
    data_mode = imported_data.get("data_mode", "UIB")
    all_samples, khp_samples, control_samples = _flatten_samples_for_processing(
        imported_data, data_mode=data_mode
    )

    total_samples = len(all_samples) + len(khp_samples) + len(control_samples)
    if total_samples == 0:
        result["errors"].append("No samples to process")
        return result

    # Processar mostres regulars
    processed_count = 0
    for i, sample in enumerate(all_samples):
        if progress_callback:
            progress_callback(f"Processing {sample.get('name', 'sample')}...", (i + 1) / total_samples * 100)

        try:
            processed = process_sample(sample, calibration_data, config)
            result["samples"].append(processed)

            if not processed.get("processed"):
                result["warnings"].append(f"{sample.get('name')}: {processed.get('error', 'Processing failed')}")

        except Exception as e:
            result["errors"].append(f"{sample.get('name')}: {str(e)}")

        processed_count += 1

    # Processar KHP
    for i, khp in enumerate(khp_samples):
        if progress_callback:
            progress_callback(f"Processing KHP {khp.get('name', '')}...", (processed_count + i + 1) / total_samples * 100)

        try:
            processed = process_sample(khp, calibration_data, config)
            result["khp_samples"].append(processed)
        except Exception as e:
            result["errors"].append(f"KHP {khp.get('name')}: {str(e)}")

        processed_count += 1

    # Processar controls
    for i, ctrl in enumerate(control_samples):
        if progress_callback:
            progress_callback(f"Processing {ctrl.get('name', '')}...", (processed_count + i + 1) / total_samples * 100)

        try:
            processed = process_sample(ctrl, calibration_data, config)
            result["control_samples"].append(processed)
        except Exception as e:
            result["errors"].append(f"Control {ctrl.get('name')}: {str(e)}")

    # Generar resum
    n_valid = sum(1 for s in result["samples"] if s.get("processed") and s.get("peak_info", {}).get("valid"))
    n_with_anomalies = sum(1 for s in result["samples"] if s.get("anomalies"))
    n_timeouts = sum(1 for s in result["samples"] if s.get("timeout_info", {}).get("n_timeouts", 0) > 0)

    result["summary"] = {
        "total_samples": len(result["samples"]),
        "valid_peaks": n_valid,
        "with_anomalies": n_with_anomalies,
        "with_timeouts": n_timeouts,
        "n_khp": len(result["khp_samples"]),
        "n_controls": len(result["control_samples"]),
    }

    result["success"] = len(result["errors"]) == 0

    if progress_callback:
        progress_callback("Processing complete", 100)

    return result


# =============================================================================
# ESCRIPTURA EXCEL CONSOLIDAT
# =============================================================================

def write_consolidated_excel(out_path, mostra, rep, seq_out, date_master,
                             method, doc_mode, fitxer_doc, fitxer_dad,
                             st_doc, st_dad, t_doc, y_doc_raw, y_doc_net,
                             baseline, df_dad, peak_info, sample_analysis=None,
                             master_file=None, row_start=None, row_end=None,
                             # Paràmetres dual protocol
                             y_doc_uib=None, y_doc_uib_raw=None, baseline_uib=None,
                             fitxer_doc_uib=None, st_doc_uib=None,
                             # Paràmetres de processament
                             shift_uib=None, shift_direct=None,
                             smoothing_applied=True,
                             # Info del MasterFile 0-INFO
                             master_info=None,
                             # Detecció de timeouts TOC
                             timeout_info=None,
                             # SNR i baseline noise
                             snr_info=None):
    """
    Escriu fitxer Excel consolidat amb àrees per fraccions de temps.

    Estructura fulls:
      ID: Identificació, fitxers, estat, processament
      TMAX: Temps de retenció per DOC i cada wavelength DAD
      AREAS: Àrees per fraccions de temps (BioP, HS, BB, SB, LMW, total)
      DOC: Cromatograma DOC (net, raw, baseline)
      DAD: Cromatograma DAD (totes les wavelengths)
    """
    from datetime import datetime

    sample_analysis = sample_analysis or {}
    master_info = master_info or {}
    is_dual = y_doc_uib is not None and hasattr(y_doc_uib, '__len__') and len(y_doc_uib) > 0

    # Convertir a arrays numpy si cal (no truncar - s'assumeix que les dades ja vénen truncades)
    t_doc = np.asarray(t_doc) if t_doc is not None else np.array([])
    y_doc_net = np.asarray(y_doc_net) if y_doc_net is not None else None
    y_doc_raw = np.asarray(y_doc_raw) if y_doc_raw is not None and hasattr(y_doc_raw, '__len__') and len(y_doc_raw) > 0 else None
    if is_dual:
        y_doc_uib = np.asarray(y_doc_uib) if y_doc_uib is not None else None
        y_doc_uib_raw = np.asarray(y_doc_uib_raw) if y_doc_uib_raw is not None and hasattr(y_doc_uib_raw, '__len__') and len(y_doc_uib_raw) > 0 else None

        # IMPORTANT: Interpolar UIB a la mida de t_doc (Direct) si tenen longituds diferents
        if y_doc_uib is not None and len(y_doc_uib) != len(t_doc) and len(t_doc) > 0:
            # Assumim que UIB té el seu propi array de temps amb rang similar
            # Creem un temps UIB aproximat basant-nos en el rang de t_doc
            t_uib_approx = np.linspace(t_doc.min(), t_doc.max(), len(y_doc_uib))
            y_doc_uib = np.interp(t_doc, t_uib_approx, y_doc_uib)

        if y_doc_uib_raw is not None and len(y_doc_uib_raw) != len(t_doc) and len(t_doc) > 0:
            t_uib_approx = np.linspace(t_doc.min(), t_doc.max(), len(y_doc_uib_raw))
            y_doc_uib_raw = np.interp(t_doc, t_uib_approx, y_doc_uib_raw)

    # NO truncar - mantenir dades completes

    # Handle baseline as scalar or array
    if baseline is not None:
        if hasattr(baseline, '__len__') and len(baseline) > 0:
            baseline_direct_val = float(np.mean(baseline))
        else:
            baseline_direct_val = float(baseline)
    else:
        baseline_direct_val = 0.0

    if baseline_uib is not None:
        if hasattr(baseline_uib, '__len__') and len(baseline_uib) > 0:
            baseline_uib_val = float(np.mean(baseline_uib))
        else:
            baseline_uib_val = float(baseline_uib)
    else:
        baseline_uib_val = 0.0

    # === ID SHEET ===
    inj_vol = master_info.get("Inj_Volume (uL)", "")
    uib_range = master_info.get("UIB_range")
    if pd.isna(inj_vol) or inj_vol is None:
        inj_vol = ""
    if pd.isna(uib_range) or uib_range is None:
        uib_range = "-"
    else:
        uib_range = str(uib_range)

    id_rows = [
        ("Script_Version", f"hpsec_process v{__version__}"),
        ("Consolidation_Date", datetime.now().strftime("%Y-%m-%d %H:%M")),
        ("---", "---"),
        ("Sample", mostra),
        ("Replica", rep),
        ("SEQ", seq_out),
        ("Method", method),
        ("Date", date_master),
        ("Inj_Volume_uL", inj_vol),
        ("UIB_range", uib_range),
    ]

    id_rows.append(("File_MasterFile", master_file or ""))
    id_rows.append(("File_DAD", fitxer_dad or ""))
    if is_dual and fitxer_doc_uib:
        id_rows.append(("File_DOC_UIB", fitxer_doc_uib))

    id_rows.extend([
        ("DOC_Mode", doc_mode),
        ("Status_DOC", st_doc),
        ("Status_DAD", st_dad),
        ("Peak_Valid", peak_info.get("valid", False) if peak_info else False),
    ])

    id_rows.append(("DOC_N_Points", len(t_doc) if t_doc is not None else 0))
    if row_start is not None:
        id_rows.append(("DOC_Row_Start", row_start))
    if row_end is not None:
        id_rows.append(("DOC_Row_End", row_end))

    id_rows.extend([
        ("DOC_Baseline_Method", "percentile"),
        ("DOC_Baseline_mAU", round(baseline_direct_val, 3)),
        ("DOC_Smoothing", "YES" if smoothing_applied else "NO"),
    ])
    if shift_direct is not None:
        id_rows.append(("DOC_Shift_sec", round(shift_direct * 60, 2)))

    if is_dual:
        id_rows.append(("UIB_Baseline_mAU", round(baseline_uib_val, 3)))
        if shift_uib is not None:
            id_rows.append(("UIB_Shift_sec", round(shift_uib * 60, 2)))
        id_rows.append(("UIB_Status", st_doc_uib or "OK"))

    if df_dad is not None and not df_dad.empty:
        id_rows.append(("DAD_N_Points", len(df_dad)))

    # Timeout info
    if timeout_info and timeout_info.get("n_timeouts", 0) > 0:
        id_rows.append(("---", "---"))
        id_rows.append(("TOC_Timeout_Detected", "YES"))
        id_rows.append(("TOC_Timeout_Count", str(timeout_info.get("n_timeouts", 0))))
        id_rows.append(("TOC_Timeout_Severity", timeout_info.get("severity", "OK")))
        timeouts = timeout_info.get("timeouts", [])
        for i, to in enumerate(timeouts[:3]):
            zone = to.get("zone", "?")
            t_min_to = to.get("t_start_min", 0)
            dur = to.get("duration_sec", 0)
            sev = to.get("severity", "INFO")
            id_rows.append((f"TOC_Timeout_{i+1}", f"{t_min_to:.1f} min ({dur:.0f}s) - {zone} [{sev}]"))
        zone_summary = timeout_info.get("zone_summary", {})
        if zone_summary:
            affected_zones = [f"{z}:{n}" for z, n in zone_summary.items() if n > 0]
            id_rows.append(("TOC_Zones_Affected", ", ".join(affected_zones)))
        id_rows.append(("TOC_Dt_Median_sec", timeout_info.get("dt_median_sec", 0)))
        id_rows.append(("TOC_Dt_Max_sec", timeout_info.get("dt_max_sec", 0)))
    else:
        id_rows.append(("TOC_Timeout_Detected", "NO"))

    # SNR info
    if snr_info:
        id_rows.append(("---", "---"))
        if snr_info.get("snr_direct") is not None:
            id_rows.append(("SNR_Direct", round(snr_info["snr_direct"], 1)))
        if snr_info.get("baseline_noise_direct") is not None:
            id_rows.append(("Baseline_Noise_Direct_mAU", round(snr_info["baseline_noise_direct"], 3)))
        if snr_info.get("lod_direct") is not None:
            id_rows.append(("LOD_Direct_mAU", round(snr_info["lod_direct"], 3)))
        if snr_info.get("loq_direct") is not None:
            id_rows.append(("LOQ_Direct_mAU", round(snr_info["loq_direct"], 3)))
        if is_dual and snr_info.get("snr_uib") is not None:
            id_rows.append(("SNR_UIB", round(snr_info["snr_uib"], 1)))
        if is_dual and snr_info.get("baseline_noise_uib") is not None:
            id_rows.append(("Baseline_Noise_UIB_mAU", round(snr_info["baseline_noise_uib"], 3)))
        if is_dual and snr_info.get("lod_uib") is not None:
            id_rows.append(("LOD_UIB_mAU", round(snr_info["lod_uib"], 3)))
        if is_dual and snr_info.get("loq_uib") is not None:
            id_rows.append(("LOQ_UIB_mAU", round(snr_info["loq_uib"], 3)))

    df_id = pd.DataFrame(id_rows, columns=["Field", "Value"])

    # === TMAX SHEET ===
    tmax_data = detectar_tmax_senyals(t_doc, y_doc_net, df_dad)
    tmax_rows = []
    for signal, tmax_val in tmax_data.items():
        tmax_rows.append((signal, round(tmax_val, 3) if tmax_val > 0 else "-"))
    df_tmax = pd.DataFrame(tmax_rows, columns=["Signal", "tmax (min)"])

    # === AREAS SHEET ===
    fraccions_data = calcular_arees_fraccions_complet(t_doc, y_doc_net, df_dad)
    fractions_config = DEFAULT_PROCESS_CONFIG.get("time_fractions", {})
    fraction_names = list(fractions_config.keys()) + ["total"]

    header = ["Fraction", "Range (min)", "DOC"]
    target_wls = DEFAULT_PROCESS_CONFIG.get("target_wavelengths", [220, 254, 280])
    for wl in target_wls:
        header.append(f"A{wl}")

    areas_rows = []
    for frac_name in fraction_names:
        if frac_name == "total":
            rang = "0-70"
        else:
            t_ini, t_fi = fractions_config.get(frac_name, [0, 0])
            rang = f"{t_ini}-{t_fi}"

        row = [frac_name, rang]
        doc_area = fraccions_data.get("DOC", {}).get(frac_name, 0.0)
        row.append(round(doc_area, 2) if doc_area > 0 else "-")
        for wl in target_wls:
            dad_area = fraccions_data.get(f"A{wl}", {}).get(frac_name, 0.0)
            row.append(round(dad_area, 2) if dad_area > 0 else "-")
        areas_rows.append(row)

    df_areas = pd.DataFrame(areas_rows, columns=header)

    # === DOC SHEET ===
    if is_dual:
        doc_data = {
            "time (min)": t_doc,
            "DOC_Direct (mAU)": y_doc_net,
            "DOC_Direct_RAW (mAU)": y_doc_raw,
            "BASELINE_Direct (mAU)": baseline_direct_val,  # Scalar value
            "DOC_UIB (mAU)": y_doc_uib,
        }
        if y_doc_uib_raw is not None and hasattr(y_doc_uib_raw, '__len__') and len(y_doc_uib_raw) > 0:
            doc_data["DOC_UIB_RAW (mAU)"] = y_doc_uib_raw
        if baseline_uib is not None:
            doc_data["BASELINE_UIB (mAU)"] = baseline_uib_val  # Scalar value
        df_doc_out = pd.DataFrame(doc_data)
    else:
        df_doc_out = pd.DataFrame({
            "time (min)": t_doc,
            "DOC (mAU)": y_doc_net,
            "DOC_RAW (mAU)": y_doc_raw,
            "BASELINE (mAU)": baseline_direct_val,  # Scalar value
        })

    # Escriure Excel
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_id.to_excel(writer, sheet_name="ID", index=False)
        df_tmax.to_excel(writer, sheet_name="TMAX", index=False)
        df_areas.to_excel(writer, sheet_name="AREAS", index=False)
        df_doc_out.to_excel(writer, sheet_name="DOC", index=False)
        if df_dad is not None and not df_dad.empty:
            df_dad.to_excel(writer, sheet_name="DAD", index=False)


# =============================================================================
# EXPORTS PER COMPATIBILITAT
# =============================================================================
__all__ = [
    # Config
    "DEFAULT_PROCESS_CONFIG",
    # Utilitats
    "truncate_chromatogram",
    "mode_robust",
    # Baseline/smoothing
    "get_baseline_correction",
    "apply_smoothing",
    # Alineació
    "align_signals_by_max",
    "apply_shift",
    # DAD
    "process_dad",
    # Pics
    "find_peak_boundaries",
    "detect_main_peak",
    # Àrees
    "calcular_fraccions_temps",
    "calcular_arees_fraccions_complet",
    "detectar_tmax_senyals",
    "analyze_sample_areas",
    # SNR
    "calculate_snr_info",
    # Escriptura
    "write_consolidated_excel",
    # Funcions principals
    "process_sample",
    "process_sequence",
]
