"""
hpsec_analyze.py - Mòdul d'anàlisi de mostres HPSEC (Fase 3: ANALITZAR)
=======================================================================

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

__version__ = "1.4.0"
__version_date__ = "2026-02-03"
# v1.4.0: Nova funció unificada analyze_signal() per qualsevol tipus de senyal
#         Nova funció analyze_signal_comparison() per comparar dos senyals
# v1.3.0: Afegides mètriques DUAL/COLUMN: batman_uib, pearson_direct_uib,
#         area_diff_pct, sb_hs_ratio, doc_254_ratio, n_peaks_254_HS
# v1.2.0: Afegit càlcul FWHM i simetria del pic (usat calculate_fwhm de hpsec_core)

import os
import json
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
from scipy.integrate import trapezoid
from scipy.stats import pearsonr

# Import funcions de detecció des de hpsec_core (Single Source of Truth)
from hpsec_core import (
    detect_timeout,
    format_timeout_status,
    TIMEOUT_CONFIG,
    calc_snr,
    detect_batman,
    detect_main_peak,
    find_peak_boundaries,
    calculate_fwhm,
    calculate_symmetry,
    # Funcions baseline (migrades de utils 2026-02-03)
    baseline_stats,
    baseline_stats_windowed,
    # Funcions alineació (migrades 2026-02-03)
    align_signals_by_max,
    apply_shift,
    # Funcions noves
    calc_snr_complete,
    compare_signals,
    # Constants
    THRESH_SNR,
)
from hpsec_config import get_config


# =============================================================================
# CONFIGURACIÓ PER DEFECTE
# =============================================================================
DEFAULT_PROCESS_CONFIG = {
    # NOTA: Baseline config ara a hpsec_config.py secció "baseline"
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


# NOTA: get_baseline_correction() eliminada (2026-02-02)
# La correcció de baseline es fa a import. Process requereix y_net.


# =============================================================================
# FUNCIONS DE SMOOTHING
# =============================================================================
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
# NOTA: align_signals_by_max i apply_shift s'han mogut a hpsec_core.py (2026-02-03)
# =============================================================================

# =============================================================================
# FUNCIONS PROCESSAMENT DAD
# =============================================================================
def analyze_dad(df_dad, config=None):
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


def calculate_dad_snr_info(df_dad, target_wavelengths=None):
    """
    Calcula SNR, LOD, LOQ per cada longitud d'ona DAD.

    Args:
        df_dad: DataFrame amb columnes 'time (min)' i wavelengths
        target_wavelengths: Llista de wavelengths a processar (default: [220, 252, 254, 272, 290, 362])

    Returns:
        dict amb estructura:
        {
            "A254": {"snr": x, "lod": y, "loq": z, "noise": n, "peak_height": h},
            "A220": {...},
            ...
        }
    """
    if target_wavelengths is None:
        target_wavelengths = [220, 252, 254, 272, 290, 362]

    result = {}

    if df_dad is None or df_dad.empty or 'time (min)' not in df_dad.columns:
        return result

    t_dad = pd.to_numeric(df_dad['time (min)'], errors='coerce').to_numpy()

    for wl in target_wavelengths:
        wl_str = str(wl)
        if wl_str not in df_dad.columns:
            continue

        y_wl = pd.to_numeric(df_dad[wl_str], errors='coerce').to_numpy()

        if len(y_wl) < 10 or np.all(np.isnan(y_wl)):
            continue

        # Calcular baseline (percentil 10)
        bl_stats = baseline_stats(y_wl)
        baseline_val = bl_stats.get("baseline", float(np.nanpercentile(y_wl, 10)))
        noise = bl_stats.get("std", 0.01)

        # Senyal net
        y_net = y_wl - baseline_val

        # Alçada del pic (màxim)
        peak_height = float(np.nanmax(y_net))

        # SNR
        snr = peak_height / noise if noise > 0 else 0.0

        result[f"A{wl}"] = {
            "snr": snr,
            "lod": 3.0 * noise,
            "loq": 10.0 * noise,
            "noise": noise,
            "peak_height": peak_height,
        }

    return result


# =============================================================================
# FUNCIÓ UNIFICADA: ANALITZAR UN SENYAL
# =============================================================================
def analyze_signal(t, y, signal_type="DOC", mode="COLUMN", timeout_positions=None,
                   config=None, baseline_precomputed=None):
    """
    Analitza un senyal individual de forma unificada (DOC Direct, DOC UIB, DAD 254, etc.).

    Aquesta funció centralitza el processament per qualsevol tipus de senyal,
    evitant duplicació de codi entre BP i COLUMN modes.

    Args:
        t: Array de temps (minuts)
        y: Array de senyal (mAU) - pot ser RAW o NET
        signal_type: Tipus de senyal:
            - "DOC_DIRECT", "DOC_UIB", "DOC" (per mode simple)
            - "DAD_254", "DAD_220", "DAD_272", etc.
        mode: "COLUMN" o "BP" - determina finestres de baseline i paràmetres
        timeout_positions: Llista de posicions temporals (min) dels timeouts
                          Propagat des del senyal DOC per coherència
        config: Configuració opcional
        baseline_precomputed: Si s'ha calculat el baseline prèviament (per y_net)

    Returns:
        dict amb:
            - valid: Bool si el senyal és vàlid per anàlisi
            - signal_type: Tipus de senyal processat
            - mode: Mode BP o COLUMN

            Baseline:
            - baseline_value: Valor de baseline (mAU)
            - baseline_std: Desviació estàndard del baseline (noise)
            - baseline_window: Finestra usada per calcular baseline

            Peak detection:
            - peak_valid: Bool si s'ha detectat pic principal
            - peak_idx: Índex del màxim
            - peak_height: Alçada del pic (mAU)
            - t_max: Temps del màxim (min)
            - t_start, t_end: Límits del pic (min)
            - area: Àrea integrada (mAU·min)
            - fwhm: Full Width at Half Maximum (min)
            - symmetry: Ratio d'asimetria (>1 = tailing)

            Quality metrics:
            - snr: Signal-to-Noise Ratio
            - lod: Limit of Detection = 3 × noise (mAU)
            - loq: Limit of Quantification = 10 × noise (mAU)

            Anomalies:
            - is_batman: Bool si detectat patró Batman
            - batman_info: Dict amb detalls Batman (si detectat)
            - anomalies: Llista d'anomalies detectades

            Timeouts (només DOC):
            - has_timeout: Bool si hi ha timeouts
            - timeout_in_peak: Bool si timeout afecta el pic principal
    """
    config = config or DEFAULT_PROCESS_CONFIG
    is_bp = mode.upper() == "BP"
    is_dad = signal_type.upper().startswith("DAD")
    is_doc = not is_dad

    result = {
        "valid": False,
        "signal_type": signal_type,
        "mode": mode,
        "anomalies": [],
    }

    # Validar entrada
    if t is None or y is None:
        result["error"] = "Missing data"
        return result

    t = np.asarray(t, dtype=float).flatten()
    y = np.asarray(y, dtype=float).flatten()

    if len(t) < 10 or len(y) < 10:
        result["error"] = "Insufficient data points"
        return result

    if len(t) != len(y):
        result["error"] = f"Array length mismatch: t={len(t)}, y={len(y)}"
        return result

    # =========================================================================
    # BASELINE
    # =========================================================================
    if baseline_precomputed is not None:
        # Senyal ja amb baseline restada (y_net)
        baseline_value = float(baseline_precomputed)
        y_net = y  # Ja és net
        # Calcular noise del senyal net
        bl_stats = baseline_stats_windowed(
            t, y_net,
            method="bp" if is_bp else "column",
            timeout_positions=timeout_positions
        )
        baseline_std = bl_stats.get("std", 0.01)
        baseline_window = bl_stats.get("window_used", "precomputed")
    else:
        # Calcular baseline
        bl_stats = baseline_stats_windowed(
            t, y,
            method="bp" if is_bp else "column",
            timeout_positions=timeout_positions,
            config=config
        )
        baseline_value = bl_stats.get("baseline", float(np.percentile(y, 10)))
        baseline_std = bl_stats.get("std", 0.01)
        baseline_window = bl_stats.get("window_used", "percentile")
        # Restar baseline
        y_net = y - baseline_value

    result["baseline_value"] = baseline_value
    result["baseline_std"] = baseline_std
    result["baseline_window"] = baseline_window

    # =========================================================================
    # PEAK DETECTION
    # =========================================================================
    y_smooth = apply_smoothing(y_net)
    peak_info = detect_main_peak(t, y_smooth, config.get("peak_min_prominence_pct", 5.0), is_bp=is_bp)

    result["peak_valid"] = peak_info.get("valid", False)

    if peak_info.get("valid"):
        result["peak_idx"] = peak_info.get("peak_idx", 0)
        result["peak_height"] = peak_info.get("height", 0.0)
        result["t_max"] = peak_info.get("t_max", 0.0)
        result["t_start"] = peak_info.get("t_start", 0.0)
        result["t_end"] = peak_info.get("t_end", 0.0)
        result["area"] = peak_info.get("area", 0.0)

        # FWHM i simetria
        peak_idx = peak_info["peak_idx"]
        left_idx = peak_info.get("left_idx", 0)
        right_idx = peak_info.get("right_idx", len(y_smooth) - 1)

        fwhm = calculate_fwhm(t, y_smooth, peak_idx, left_idx, right_idx)
        symmetry = calculate_symmetry(t, y_smooth, peak_idx, left_idx, right_idx)

        result["fwhm"] = fwhm
        result["symmetry"] = symmetry

        # =====================================================================
        # BATMAN DETECTION
        # =====================================================================
        batman_result = detect_batman(t, y_smooth)
        result["is_batman"] = batman_result.get("is_batman", False)
        if result["is_batman"]:
            result["batman_info"] = batman_result
            result["anomalies"].append("BATMAN")

        # =====================================================================
        # SNR / LOD / LOQ
        # =====================================================================
        noise = baseline_std if baseline_std > 0 else 0.01
        height = result["peak_height"]

        result["snr"] = height / noise if noise > 0 else 0.0
        result["lod"] = 3.0 * noise
        result["loq"] = 10.0 * noise

        # Check SNR threshold
        if result["snr"] < THRESH_SNR:
            result["anomalies"].append("LOW_SNR")

        result["valid"] = True
    else:
        result["anomalies"].append("NO_PEAK")

    # =========================================================================
    # TIMEOUT CHECK (només DOC)
    # =========================================================================
    if is_doc and timeout_positions:
        result["has_timeout"] = len(timeout_positions) > 0

        # Verificar si algun timeout afecta el pic principal
        if result["peak_valid"]:
            t_start = result.get("t_start", 0)
            t_end = result.get("t_end", float(np.max(t)))

            timeout_in_peak = any(
                t_start <= tp <= t_end
                for tp in timeout_positions
            )
            result["timeout_in_peak"] = timeout_in_peak

            if timeout_in_peak:
                result["anomalies"].append("TIMEOUT_IN_PEAK")
    else:
        result["has_timeout"] = False
        result["timeout_in_peak"] = False

    return result


def analyze_signal_comparison(signal1_result, signal2_result, t1=None, y1=None, t2=None, y2=None):
    """
    Compara dos senyals analitzats (ex: DOC Direct vs DOC UIB).

    Args:
        signal1_result: Resultat de analyze_signal() per senyal 1
        signal2_result: Resultat de analyze_signal() per senyal 2
        t1, y1: Arrays de temps i senyal 1 (opcionals, per Pearson)
        t2, y2: Arrays de temps i senyal 2 (opcionals, per Pearson)

    Returns:
        dict amb:
            - pearson: Correlació Pearson entre senyals
            - area_diff_pct: Diferència % d'àrees
            - height_diff_pct: Diferència % d'alçades de pic
            - t_max_diff: Diferència de temps de màxim (min)
            - both_valid: Bool si ambdós senyals són vàlids
    """
    result = {
        "both_valid": False,
        "pearson": np.nan,
        "area_diff_pct": np.nan,
        "height_diff_pct": np.nan,
        "t_max_diff": np.nan,
    }

    valid1 = signal1_result.get("valid", False)
    valid2 = signal2_result.get("valid", False)

    if not valid1 or not valid2:
        return result

    result["both_valid"] = True

    # Pearson correlation (si tenim les dades)
    if t1 is not None and y1 is not None and t2 is not None and y2 is not None:
        comparison = compare_signals(t1, y1, t2, y2, normalize=True)
        result["pearson"] = comparison.get("pearson", np.nan)

    # Area diff
    area1 = signal1_result.get("area", 0)
    area2 = signal2_result.get("area", 0)
    if max(area1, area2) > 0:
        result["area_diff_pct"] = abs(area1 - area2) / max(area1, area2) * 100

    # Height diff
    h1 = signal1_result.get("peak_height", 0)
    h2 = signal2_result.get("peak_height", 0)
    if max(h1, h2) > 0:
        result["height_diff_pct"] = abs(h1 - h2) / max(h1, h2) * 100

    # t_max diff
    t1_max = signal1_result.get("t_max", 0)
    t2_max = signal2_result.get("t_max", 0)
    result["t_max_diff"] = abs(t1_max - t2_max)

    return result


# =============================================================================
# PROCESSAMENT D'UNA MOSTRA
# =============================================================================
def analyze_sample(sample_data, calibration_data=None, config=None):
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
            - fwhm_doc: FWHM del pic DOC (minuts)
            - symmetry_doc: Simetria del pic DOC (ratio)
            - fwhm_uib: FWHM del pic UIB (minuts, només si DUAL)
            - batman_direct: Bool si detectat Batman a Direct
            - batman_uib: Bool si detectat Batman a UIB (només DUAL)
            - areas: Dict amb àrees per fraccions (DOC + DAD)
            - areas_uib: Dict àrees UIB per fraccions (només DUAL)
            - tmax_signals: Dict amb tmax per senyal
            - snr_info: Dict amb SNR, LOD, LOQ
            - timeout_info: Info de timeouts
            - anomalies: Llista d'anomalies detectades

            Només COLUMN mode:
            - pearson_direct_uib: Correlació Direct vs UIB
            - area_diff_pct: Dict amb diff % per fracció (Direct vs UIB)
            - sb_hs_ratio: Ratio àrea SB / àrea HS
            - doc_254_ratio: Dict amb ratio DOC/254 per fracció
            - n_peaks_254_HS: Nombre de pics a 254nm dins zona HS
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

    # Aplicar shifts d'alineació (si disponibles, venen de calibrate)
    # El shift és translació temporal - NO invalida la correcció de baseline.
    # S'aplica shift a TOTS els senyals (raw i net) per mantenir coherència.
    if calibration_data:
        shift_uib = calibration_data.get("shift_uib", 0.0)
        shift_direct = calibration_data.get("shift_direct", 0.0)

        if is_dual:
            # UIB: interpolar a escala t_doc (referencia Direct)
            if t_doc_uib is not None and y_doc_uib is not None and len(y_doc_uib) > 0:
                t_uib_arr = np.asarray(t_doc_uib).flatten()
                # Aplicar shift + interpolació a RAW
                if abs(shift_uib) > 0.001:
                    y_doc_uib = apply_shift(t_doc, t_uib_arr, y_doc_uib, shift_uib)
                elif len(t_uib_arr) != len(t_doc):
                    y_doc_uib = np.interp(t_doc, t_uib_arr, y_doc_uib, left=0, right=0)
                # Aplicar shift + interpolació a NET (si disponible)
                if y_doc_uib_net_precomp is not None:
                    if abs(shift_uib) > 0.001:
                        y_doc_uib_net_precomp = apply_shift(t_doc, t_uib_arr, y_doc_uib_net_precomp, shift_uib)
                    elif len(t_uib_arr) != len(t_doc):
                        y_doc_uib_net_precomp = np.interp(t_doc, t_uib_arr, y_doc_uib_net_precomp, left=0, right=0)
            elif y_doc_uib is not None and len(y_doc_uib) != len(t_doc):
                y_doc_uib = None
                y_doc_uib_net_precomp = None

            # Direct: ja està a t_doc, només shift si cal
            if abs(shift_direct) > 0.001:
                y_doc_direct = apply_shift(t_doc, t_doc, y_doc_direct, shift_direct)
                if y_doc_direct_net_precomp is not None:
                    y_doc_direct_net_precomp = apply_shift(t_doc, t_doc, y_doc_direct_net_precomp, shift_direct)
        else:
            shift = calibration_data.get("shift", 0.0)
            if abs(shift) > 0.001:
                y_doc = apply_shift(t_doc, t_doc, y_doc, shift)
                if y_doc_net_precomp is not None:
                    y_doc_net_precomp = apply_shift(t_doc, t_doc, y_doc_net_precomp, shift)

    # Interpolació UIB a escala Direct (sense calibration_data)
    # Necessari quan UIB té diferent resolució temporal que Direct
    if is_dual and t_doc_uib is not None and y_doc_uib is not None:
        t_uib_arr = np.asarray(t_doc_uib).flatten()
        if len(t_uib_arr) != len(t_doc):
            # Interpolar UIB RAW a escala Direct
            y_doc_uib = np.interp(t_doc, t_uib_arr, y_doc_uib, left=0, right=0)
            # Interpolar UIB NET si disponible
            if y_doc_uib_net_precomp is not None:
                y_doc_uib_net_precomp = np.interp(t_doc, t_uib_arr, y_doc_uib_net_precomp, left=0, right=0)

    # Correcció de baseline
    # REQUEREIX y_net precalculat per import. Si no disponible → error.
    # (La baseline NO es recalcula aquí, ha d'estar feta a import)
    if is_dual:
        if y_doc_direct_net_precomp is not None:
            y_doc_direct_net = y_doc_direct_net_precomp
        else:
            result["error"] = "BASELINE_MISSING"
            result["error_message"] = "Dades Direct sense correcció de baseline. Cal tornar a importar la seqüència."
            return result

        if y_doc_uib is not None and len(y_doc_uib) > 0:
            if y_doc_uib_net_precomp is not None and len(y_doc_uib_net_precomp) == len(t_doc):
                y_doc_uib_net = y_doc_uib_net_precomp
            else:
                # UIB sense baseline: warning però seguim amb Direct
                y_doc_uib_net = None
                result["anomalies"].append("UIB_NO_BASELINE")
        else:
            y_doc_uib_net = None

        # Per processament principal, usar Direct
        y_doc_net = y_doc_direct_net
    else:
        if y_doc_net_precomp is not None:
            y_doc_net = y_doc_net_precomp
        else:
            result["error"] = "BASELINE_MISSING"
            result["error_message"] = "Dades sense correcció de baseline. Cal tornar a importar la seqüència."
            return result

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
        # Detectar Batman DOC Direct
        batman_result = detect_batman(t_doc, y_smooth)
        if batman_result.get("is_batman"):
            result["anomalies"].append("BATMAN_DIRECT")
            result["batman_direct"] = True
        else:
            result["batman_direct"] = False

    # Detectar Batman UIB (si DUAL)
    if is_dual and y_doc_uib_net is not None and len(y_doc_uib_net) > 0:
        y_uib_smooth = apply_smoothing(y_doc_uib_net)
        batman_uib_result = detect_batman(t_doc, y_uib_smooth)
        if batman_uib_result.get("is_batman"):
            result["anomalies"].append("BATMAN_UIB")
            result["batman_uib"] = True
        else:
            result["batman_uib"] = False

    result["peak_info"] = peak_info

    # Calcular FWHM i simetria del pic principal
    if peak_info.get("valid"):
        peak_idx = peak_info.get("peak_idx", 0)
        left_idx = peak_info.get("left_idx", 0)
        right_idx = peak_info.get("right_idx", len(y_doc_net) - 1)

        fwhm_doc = calculate_fwhm(t_doc, y_smooth, peak_idx, left_idx, right_idx)
        sym_doc = calculate_symmetry(t_doc, y_smooth, peak_idx, left_idx, right_idx)

        result["fwhm_doc"] = fwhm_doc
        result["symmetry_doc"] = sym_doc

        # FWHM per UIB si és DUAL
        if is_dual and y_doc_uib_net is not None and len(y_doc_uib_net) > 0:
            y_uib_smooth = apply_smoothing(y_doc_uib_net)
            peak_uib = detect_main_peak(t_doc, y_uib_smooth, config.get("peak_min_prominence_pct", 5.0), is_bp=is_bp)
            if peak_uib.get("valid"):
                fwhm_uib = calculate_fwhm(
                    t_doc, y_uib_smooth,
                    peak_uib["peak_idx"],
                    peak_uib.get("left_idx", 0),
                    peak_uib.get("right_idx", len(t_doc) - 1)
                )
                result["fwhm_uib"] = fwhm_uib

        # FWHM i Symmetry per DAD 254 (BP mode)
        if is_bp and df_dad is not None and not df_dad.empty and '254' in df_dad.columns:
            try:
                t_dad = pd.to_numeric(df_dad['time (min)'], errors='coerce').to_numpy()
                y_254 = pd.to_numeric(df_dad['254'], errors='coerce').to_numpy()
                if len(y_254) > 10:
                    # Baseline i suavitzat
                    baseline_254 = float(np.nanpercentile(y_254, 10))
                    y_254_net = y_254 - baseline_254
                    y_254_smooth = apply_smoothing(y_254_net)
                    # Detectar pic
                    peak_254 = detect_main_peak(t_dad, y_254_smooth, 5.0, is_bp=True)
                    if peak_254.get("valid"):
                        peak_idx_254 = peak_254["peak_idx"]
                        left_idx_254 = peak_254.get("left_idx", 0)
                        right_idx_254 = peak_254.get("right_idx", len(t_dad) - 1)
                        # FWHM
                        fwhm_254 = calculate_fwhm(
                            t_dad, y_254_smooth,
                            peak_idx_254, left_idx_254, right_idx_254
                        )
                        result["fwhm_254"] = fwhm_254
                        # Symmetry (50% altura, estàndard)
                        sym_254 = calculate_symmetry(
                            t_dad, y_254_smooth,
                            peak_idx_254, left_idx_254, right_idx_254
                        )
                        result["symmetry_254"] = sym_254
            except Exception:
                pass

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

    # Calcular SNR info (DOC Direct i UIB)
    snr_info = calculate_snr_info(
        y_doc_net,
        peak_info,
        y_doc_uib=y_doc_uib_net if is_dual else None,
        t_min=t_doc,
        method="bp" if is_bp else "column",
        timeout_positions=timeout_positions
    )
    result["snr_info"] = snr_info

    # Calcular SNR info per DAD (totes les wavelengths)
    dad_snr_info = calculate_dad_snr_info(df_dad, config.get("target_wavelengths"))
    if dad_snr_info:
        result["snr_info_dad"] = dad_snr_info

    # =========================================================================
    # MÈTRIQUES ADDICIONALS (només COLUMN i DUAL)
    # =========================================================================
    if not is_bp:
        # --- Pearson Direct vs UIB ---
        if is_dual and y_doc_uib_net is not None and len(y_doc_uib_net) == len(y_doc_direct_net):
            try:
                pearson_val, _ = pearsonr(y_doc_direct_net, y_doc_uib_net)
                result["pearson_direct_uib"] = float(pearson_val)
            except Exception:
                result["pearson_direct_uib"] = np.nan

        # --- Area diff % Direct vs UIB per fracció ---
        if is_dual and "DOC" in areas:
            areas_direct = areas.get("DOC", {})
            # Calcular àrees UIB per fraccions
            areas_uib = calcular_fraccions_temps(t_doc, y_doc_uib_net, config) if y_doc_uib_net is not None else {}
            result["areas_uib"] = areas_uib

            area_diff_pct = {}
            for frac in ["BioP", "HS", "BB", "SB", "LMW", "total"]:
                a_d = areas_direct.get(frac, 0)
                a_u = areas_uib.get(frac, 0)
                if max(a_d, a_u) > 0:
                    area_diff_pct[frac] = abs(a_d - a_u) / max(a_d, a_u) * 100
                else:
                    area_diff_pct[frac] = 0.0
            result["area_diff_pct"] = area_diff_pct

        # --- SB/HS ratio (Direct) ---
        if "DOC" in areas:
            area_hs = areas["DOC"].get("HS", 0)
            area_sb = areas["DOC"].get("SB", 0)
            if area_hs > 0:
                result["sb_hs_ratio"] = float(area_sb / area_hs)
            else:
                result["sb_hs_ratio"] = np.nan

        # --- SB/HS ratio (UIB) ---
        if is_dual and "areas_uib" in result:
            area_hs_uib = result["areas_uib"].get("HS", 0)
            area_sb_uib = result["areas_uib"].get("SB", 0)
            if area_hs_uib > 0:
                result["sb_hs_ratio_uib"] = float(area_sb_uib / area_hs_uib)
            else:
                result["sb_hs_ratio_uib"] = np.nan

        # --- DOC/254 ratio per fracció ---
        if "DOC" in areas and "A254" in areas:
            doc_254_ratio = {}
            for frac in ["BioP", "HS", "BB", "SB", "LMW", "total"]:
                a_doc = areas["DOC"].get(frac, 0)
                a_254 = areas["A254"].get(frac, 0)
                if a_254 > 0:
                    doc_254_ratio[frac] = float(a_doc / a_254)
                else:
                    doc_254_ratio[frac] = np.nan
            result["doc_254_ratio"] = doc_254_ratio

        # --- Nombre de pics a 254nm dins zona HS (18-23 min) ---
        if df_dad is not None and not df_dad.empty and '254' in df_dad.columns:
            try:
                t_dad = df_dad['time (min)'].to_numpy()
                y_254 = df_dad['254'].to_numpy()

                # Filtrar zona HS
                hs_start, hs_end = 18.0, 23.0
                mask_hs = (t_dad >= hs_start) & (t_dad <= hs_end)

                if np.sum(mask_hs) > 10:
                    y_254_hs = y_254[mask_hs]
                    # Detectar pics amb prominència mínima (5% del rang)
                    y_range = np.max(y_254_hs) - np.min(y_254_hs)
                    min_prom = y_range * 0.05
                    peaks, _ = find_peaks(y_254_hs, prominence=min_prom, distance=3)
                    result["n_peaks_254_HS"] = len(peaks)
                else:
                    result["n_peaks_254_HS"] = 0
            except Exception:
                result["n_peaks_254_HS"] = None

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
            # Extreure informació d'injecció (per seleccionar calibració correcta)
            inj_info = rep_data.get("injection_info", {})

            flat_sample = {
                "name": sample_name,
                "replica": str(rep_num),
                "sample_type": sample_type,
                "inj_volume": inj_info.get("inj_volume"),  # Volum d'injecció en µL
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
def analyze_sequence(imported_data, calibration_data=None, config=None, progress_callback=None):
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
            processed = analyze_sample(sample, calibration_data, config)
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
            processed = analyze_sample(khp, calibration_data, config)
            result["khp_samples"].append(processed)
        except Exception as e:
            result["errors"].append(f"KHP {khp.get('name')}: {str(e)}")

        processed_count += 1

    # Processar controls
    for i, ctrl in enumerate(control_samples):
        if progress_callback:
            progress_callback(f"Processing {ctrl.get('name', '')}...", (processed_count + i + 1) / total_samples * 100)

        try:
            processed = analyze_sample(ctrl, calibration_data, config)
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
# GUARDAR RESULTAT ANÀLISI (JSON)
# =============================================================================

def get_data_folder(seq_path, create=False):
    """Retorna la carpeta CHECK/data d'una SEQ."""
    data_folder = os.path.join(seq_path, "CHECK", "data")
    if create:
        os.makedirs(data_folder, exist_ok=True)
    return data_folder


class NumpyEncoder(json.JSONEncoder):
    """Encoder JSON per tipus numpy."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if pd.isna(obj):
            return None
        return super().default(obj)


def save_analysis_result(analysis_data, output_path=None):
    """
    Guarda el resultat de l'anàlisi a JSON.

    Args:
        analysis_data: Dict retornat per analyze_sequence()
        output_path: Ruta de sortida (default: SEQ_PATH/CHECK/data/analysis_result.json)

    Returns:
        Path del fitxer generat o None si error
    """
    import json
    from datetime import datetime

    if not analysis_data:
        return None

    seq_path = analysis_data.get("seq_path", ".")

    if output_path is None:
        data_folder = get_data_folder(seq_path, create=True)
        output_path = os.path.join(data_folder, "analysis_result.json")

    # Preparar dades per serialitzar (eliminar arrays grans)
    result = {
        "version": "1.0",
        "timestamp": datetime.now().isoformat(),
        "seq_name": analysis_data.get("seq_name", ""),
        "seq_path": seq_path,
        "method": analysis_data.get("method", ""),
        "data_mode": analysis_data.get("data_mode", ""),
        "success": analysis_data.get("success", False),
        "errors": analysis_data.get("errors", []),
        "warnings": analysis_data.get("warnings", []),
        "summary": analysis_data.get("summary", {}),
        # Llista de mostres amb info resumida (sense arrays de dades)
        "samples": [],
        "khp_samples": [],
        "control_samples": [],
    }

    def summarize_sample(sample):
        """Extreu info resumida d'una mostra (sense arrays grans)."""
        return {
            "name": sample.get("name", ""),
            "replica": sample.get("replica", ""),
            "processed": sample.get("processed", False),
            "error": sample.get("error"),
            # Peak info
            "peak_info": sample.get("peak_info", {}),
            # Àrees
            "areas": sample.get("areas", {}),
            "areas_uib": sample.get("areas_uib", {}),
            # Anomalies
            "anomalies": sample.get("anomalies", []),
            "timeout_info": sample.get("timeout_info", {}),
            # SNR
            "snr_info": sample.get("snr_info", {}),
            # Mètriques DUAL
            "batman_uib": sample.get("batman_uib"),
            "pearson_direct_uib": sample.get("pearson_direct_uib"),
            "area_diff_pct": sample.get("area_diff_pct"),
        }

    # Resumir mostres
    for sample in analysis_data.get("samples", []):
        result["samples"].append(summarize_sample(sample))

    for khp in analysis_data.get("khp_samples", []):
        result["khp_samples"].append(summarize_sample(khp))

    for ctrl in analysis_data.get("control_samples", []):
        result["control_samples"].append(summarize_sample(ctrl))

    # Guardar
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        return output_path
    except Exception as e:
        print(f"Error guardant analysis_result.json: {e}")
        return None


def load_analysis_result(seq_path):
    """
    Carrega el resultat d'anàlisi si existeix.

    Args:
        seq_path: Path de la SEQ

    Returns:
        Dict amb el resultat o None si no existeix
    """
    import json

    data_folder = get_data_folder(seq_path)
    filepath = os.path.join(data_folder, "analysis_result.json")

    if not os.path.exists(filepath):
        return None

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error carregant analysis_result.json: {e}")
        return None


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
        ("Script_Version", f"hpsec_analyze v{__version__}"),
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
    # Smoothing
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
    "analyze_sample",
    "analyze_sequence",
    # Funcions unificades (v1.4.0)
    "analyze_signal",
    "analyze_signal_comparison",
]
