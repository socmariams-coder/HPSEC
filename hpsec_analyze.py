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

__version__ = "1.6.0"
__version_date__ = "2026-02-05"
# v1.6.0: Millores quantificació DUAL i comparació DAD
#         - areas_uib ara es calcula per DUAL (tant COLUMN com BP)
#         - quantify_sample(): retorna ppm_direct i ppm_uib separats
#         - compare_replicas(): R² DAD per les 6 λ (pearson_per_wavelength)
#         - Afegit pearson_min i wavelength_min per DAD
# v1.5.0: Comparació rèpliques, recomanació i quantificació
#         - compare_replicas(): Pearson + diff àrees per fracció
#         - recommend_replica(): Selecció automàtica DOC/DAD independent
#         - quantify_sample(): Aplicar calibració (àrea → ppm)
#         - analyze_sequence() ara genera samples_grouped amb tot
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
    # Bi-Gaussian fitting
    fit_bigaussian,
    THRESH_R2_VALID,
    THRESH_R2_CHECK,
    # Funcions baseline (migrades de utils 2026-02-03)
    get_baseline_value,
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

# Import funcions calibració per seleccionar calibració segons volum d'injecció
from hpsec_calibrate import (
    get_calibration_for_conditions,
    get_all_active_calibrations,
    get_rf_mass_cal,
    get_calibration_intercept,
    quantify_with_global_calibration
)

# Import sistema d'avisos estructurats
from hpsec_warnings import (
    create_warning, get_max_warning_level, WarningLevel,
    migrate_warnings_list,
    # Anomaly catalog system
    create_anomaly, get_anomaly_codes, has_anomaly, classify_anomalies,
    normalize_anomalies, mark_repaired, get_max_anomaly_severity,
    ANOMALY_CATALOG, CRITICAL_ANOMALIES,
)


# =============================================================================
# DEV NOTES - Logging per desenvolupament
# =============================================================================
_DEV_NOTES_ENABLED = os.environ.get("HPSEC_DEV_NOTES", "0") == "1"

def _log_detection_issue(seq_name: str, sample_name: str, issue_type: str,
                         signal: str, details: dict):
    """Log un problema de detecció a les dev notes (si actiu)."""
    if not _DEV_NOTES_ENABLED:
        return
    try:
        from hpsec_dev_notes import add_detection_issue
        add_detection_issue(
            seq_name=seq_name,
            sample_name=sample_name,
            issue_type=issue_type,
            signal=signal,
            details={
                "max_depth": details.get("max_depth", 0),
                "n_valleys": details.get("n_valleys", 0),
                "reason": details.get("reason", ""),
            },
            severity="warning"
        )
    except Exception:
        pass  # No fallar si dev_notes no disponible


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
    # Fraccions de temps: definides a hpsec_config.json, es carreguen dinàmicament
    "time_fractions": {},
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
def calcular_fraccions_temps(t, y, config=None, exclude_from_total=None):
    """
    Calcula àrees per fraccions de temps (integració parcial).

    Args:
        t: Array de temps (minuts)
        y: Array de senyal (mAU, ja amb baseline restada)
        config: Configuració amb time_fractions
        exclude_from_total: Llista de noms de fraccions a excloure del total
            (ex: ["LMW"] per COLUMN mostres reals). Les fraccions excloses
            segueixen reportant-se individualment però no compten al total.

    Returns:
        Dict amb àrees per fracció: {BioP, HS, BB, SB, LMW, total, total_all, *_pct}
        - total: àrea integrada (exclou fraccions de exclude_from_total si aplicable)
        - total_all: àrea total sense exclusions (sempre el cromatograma complet)
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(t) < 2 or len(y) < 2:
        return {"total": 0.0, "total_all": 0.0}

    exclude_from_total = set(exclude_from_total or [])

    # Assegurar que y no té valors negatius (baseline ja restada)
    y_clean = np.maximum(y, 0)

    # Àrea total del cromatograma (sense cap exclusió)
    total_all = float(trapezoid(y_clean, t))

    # Obtenir fraccions de la config (centralitzada a hpsec_config.json)
    if config is None:
        config = DEFAULT_PROCESS_CONFIG
    fractions = config.get("time_fractions", {})
    if not fractions:
        try:
            from hpsec_config import get_config
            cfg = get_config()
            for fname, finfo in cfg.get_all_fractions():
                fractions[fname] = [finfo["start"], finfo["end"]]
        except Exception:
            pass

    # Calcular àrea per cada fracció
    kpis = {}
    for nom, (t_ini, t_fi) in fractions.items():
        mask = (t >= t_ini) & (t < t_fi)
        if np.sum(mask) > 1:
            kpis[nom] = float(trapezoid(y_clean[mask], t[mask]))
        else:
            kpis[nom] = 0.0

    # Total operatiu: exclou fraccions indicades
    kpis["total_all"] = total_all
    if exclude_from_total:
        excluded_area = sum(kpis.get(nom, 0.0) for nom in exclude_from_total)
        kpis["total"] = total_all - excluded_area
        kpis["excluded_fractions"] = sorted(exclude_from_total)
    else:
        kpis["total"] = total_all

    # Calcular percentatges (sobre total operatiu, NO total_all)
    ref_total = kpis["total"]
    if ref_total > 0:
        for nom in fractions.keys():
            kpis[f"{nom}_pct"] = 100.0 * kpis[nom] / ref_total
    else:
        for nom in fractions.keys():
            kpis[f"{nom}_pct"] = 0.0

    return kpis


def detectar_tmax_senyals(t_doc, y_doc, df_dad, config=None, mode="COLUMN"):
    """
    Detecta el temps de retenció (tmax) per DOC i cada longitud d'ona DAD.

    Args:
        t_doc: Array temps DOC
        y_doc: Array senyal DOC (net, amb baseline restada)
        df_dad: DataFrame DAD amb columnes 'time (min)' i wavelengths
        config: Configuració
        mode: "BP" o "COLUMN" - per baseline DAD

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
                    # Baseline unificada
                    baseline = get_baseline_value(t_dad, y_wl, mode=mode)
                    y_wl_net = y_wl - baseline
                    idx_max = np.nanargmax(y_wl_net)
                    result[f"A{wl}"] = float(t_dad[idx_max])

    return result


def calcular_arees_fraccions_complet(t_doc, y_doc, df_dad, config=None,
                                     mode="COLUMN", exclude_from_total=None):
    """
    Calcula àrees per fraccions de temps per DOC i totes les wavelengths DAD.

    Args:
        t_doc: Array temps DOC
        y_doc: Array senyal DOC (net)
        df_dad: DataFrame DAD
        config: Configuració
        mode: "BP" o "COLUMN" - per càlcul de baseline DAD coherent
        exclude_from_total: Fraccions a excloure del total (ex: ["LMW"])

    Returns:
        Dict amb estructura:
        {
            "DOC": {BioP: x, HS: y, ..., total: z, total_all: w},
            "A220": {BioP: x, HS: y, ..., total: z, total_all: w},
            ...
        }
    """
    if config is None:
        config = DEFAULT_PROCESS_CONFIG
    target_wls = config.get('target_wavelengths', [220, 252, 254, 272, 290, 362])

    result = {}

    # Fraccions DOC
    if t_doc is not None and y_doc is not None and len(t_doc) > 10:
        result["DOC"] = calcular_fraccions_temps(t_doc, y_doc, config,
                                                  exclude_from_total=exclude_from_total)
    else:
        result["DOC"] = {"total": 0.0, "total_all": 0.0}

    # Fraccions per cada wavelength DAD
    if df_dad is not None and not df_dad.empty and 'time (min)' in df_dad.columns:
        t_dad = pd.to_numeric(df_dad['time (min)'], errors='coerce').to_numpy()

        for wl in target_wls:
            wl_str = str(wl)
            if wl_str in df_dad.columns:
                y_wl = pd.to_numeric(df_dad[wl_str], errors='coerce').to_numpy()
                if len(y_wl) > 10 and not np.all(np.isnan(y_wl)):
                    # Baseline unificada: usar get_baseline_value() del core
                    baseline = get_baseline_value(t_dad, y_wl, mode=mode)
                    y_wl_net = np.maximum(y_wl - baseline, 0)
                    result[f"A{wl}"] = calcular_fraccions_temps(
                        t_dad, y_wl_net, config,
                        exclude_from_total=exclude_from_total)
                else:
                    result[f"A{wl}"] = {"total": 0.0, "total_all": 0.0}
            else:
                result[f"A{wl}"] = {"total": 0.0, "total_all": 0.0}
    else:
        for wl in target_wls:
            result[f"A{wl}"] = {"total": 0.0, "total_all": 0.0}

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
            result["anomalies"].append(create_anomaly("BATMAN", details=batman_result))

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
            result["anomalies"].append(create_anomaly("LOW_SNR", details={"snr": result["snr"]}))

        result["valid"] = True
    else:
        result["anomalies"].append(create_anomaly("NO_PEAK"))

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
                result["anomalies"].append(create_anomaly("TIMEOUT_IN_PEAK"))
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
# COMPARACIÓ DE RÈPLIQUES
# =============================================================================

def _get_warning_thresholds():
    """Obté llindars de warnings des de config, amb defaults."""
    try:
        config = get_config()
        thresholds = config.get("warnings", {}).get("thresholds", {})
        return {
            "pearson": thresholds.get("replica_pearson", {}).get("warning", 0.995),
            "area_diff": thresholds.get("replica_area_diff_pct", {}).get("warning", 10.0),
            "frac_diff": thresholds.get("replica_fraction_diff_pct", {}).get("warning", 15.0),
        }
    except Exception:
        return {"pearson": 0.995, "area_diff": 10.0, "frac_diff": 15.0}

# Defaults (usats si config no disponible)
REPLICA_PEARSON_THRESHOLD = 0.995      # Warning si Pearson < 0.995
REPLICA_AREA_DIFF_THRESHOLD = 10.0     # Warning si diff àrea total > 10%
REPLICA_FRAC_DIFF_THRESHOLD = 15.0     # Warning si diff fracció > 15%


def compare_replicas(r1_result, r2_result, mode="COLUMN", config=None):
    """
    Compara dues rèpliques d'una mateixa mostra.

    Args:
        r1_result: Resultat de analyze_sample() per rèplica 1
        r2_result: Resultat de analyze_sample() per rèplica 2
        mode: "COLUMN" o "BP"
        config: Configuració opcional

    Returns:
        dict amb:
            - doc: comparació senyals DOC
                - pearson: correlació R1-R2
                - area_diff_pct: diferència àrea total (%)
                - fraction_diff_pct: dict amb diff per fracció (COLUMN)
                - warnings: llista de warnings
            - dad: comparació senyals DAD
                - pearson_254: correlació R1-R2 a 254nm
                - area_diff_pct: diferència àrea total (%)
                - warnings: llista de warnings
    """
    # Obtenir llindars de config (o usar defaults)
    thresholds = _get_warning_thresholds()
    pearson_threshold = thresholds["pearson"]
    area_diff_threshold = thresholds["area_diff"]
    frac_diff_threshold = thresholds["frac_diff"]

    result = {
        "doc": {
            "pearson": np.nan,
            "area_diff_pct": np.nan,
            "fraction_diff_pct": {},
            "warnings": []
        },
        "dad": {
            "pearson_254": np.nan,
            "pearson_per_wavelength": {},  # R² per cada λ
            "pearson_min": np.nan,         # R² mínim
            "wavelength_min": None,        # λ amb R² mínim
            "area_diff_pct": np.nan,
            "warnings": []
        }
    }

    # Verificar que ambdues rèpliques són vàlides
    if not r1_result.get("processed") or not r2_result.get("processed"):
        result["doc"]["warnings"].append(create_anomaly("REPLICA_NOT_PROCESSED"))
        return result

    is_column = mode.upper() == "COLUMN"

    # =========================================================================
    # COMPARACIÓ DOC
    # =========================================================================
    t1 = r1_result.get("t_doc")
    t2 = r2_result.get("t_doc")
    y1 = r1_result.get("y_doc_net")
    y2 = r2_result.get("y_doc_net")

    # Convertir a arrays i validar longituds
    if t1 is not None:
        t1 = np.asarray(t1).flatten()
    if t2 is not None:
        t2 = np.asarray(t2).flatten()
    if y1 is not None:
        y1 = np.asarray(y1).flatten()
    if y2 is not None:
        y2 = np.asarray(y2).flatten()

    # Validar que t i y tenen la mateixa longitud
    t1_valid = t1 is not None and y1 is not None and len(t1) == len(y1) and len(y1) > 10
    t2_valid = t2 is not None and y2 is not None and len(t2) == len(y2) and len(y2) > 10

    if t1_valid and t2_valid:
        # Interpolar si longituds diferents
        if len(y1) != len(y2):
            t_common = t1 if len(t1) <= len(t2) else t2
            y1_interp = np.interp(t_common, t1, y1)
            y2_interp = np.interp(t_common, t2, y2)
        else:
            y1_interp, y2_interp = y1, y2

        # Pearson
        try:
            pearson_val, _ = pearsonr(y1_interp, y2_interp)
            result["doc"]["pearson"] = float(pearson_val)
            if pearson_val < pearson_threshold:
                result["doc"]["warnings"].append(create_anomaly("LOW_CORRELATION",
                    details={"pearson": float(pearson_val), "threshold": pearson_threshold}))
        except Exception:
            pass

        # Diferència àrea total
        areas1 = r1_result.get("areas", {}).get("DOC", {})
        areas2 = r2_result.get("areas", {}).get("DOC", {})
        area1_total = areas1.get("total", 0)
        area2_total = areas2.get("total", 0)

        if max(area1_total, area2_total) > 0:
            diff_pct = abs(area1_total - area2_total) / max(area1_total, area2_total) * 100
            result["doc"]["area_diff_pct"] = diff_pct
            if diff_pct > area_diff_threshold:
                result["doc"]["warnings"].append(create_anomaly("AREA_DIFF_HIGH",
                    details={"diff_pct": diff_pct, "threshold": area_diff_threshold}))

        # Diferència per fracció (només COLUMN)
        if is_column:
            for frac in ["BioP", "HS", "BB", "SB", "LMW"]:
                a1 = areas1.get(frac, 0)
                a2 = areas2.get(frac, 0)
                if max(a1, a2) > 0:
                    frac_diff = abs(a1 - a2) / max(a1, a2) * 100
                    result["doc"]["fraction_diff_pct"][frac] = frac_diff
                    if frac_diff > frac_diff_threshold:
                        result["doc"]["warnings"].append(create_anomaly("FRACTION_DIFF_HIGH",
                            details={"fraction": frac, "diff_pct": frac_diff, "threshold": frac_diff_threshold}))
                else:
                    result["doc"]["fraction_diff_pct"][frac] = 0.0

    # =========================================================================
    # COMPARACIÓ DAD (totes les λ: 220, 252, 254, 272, 290, 362)
    # =========================================================================
    df_dad1 = r1_result.get("df_dad")
    df_dad2 = r2_result.get("df_dad")
    wavelengths = ['220', '252', '254', '272', '290', '362']

    if df_dad1 is not None and df_dad2 is not None:
        if not df_dad1.empty and not df_dad2.empty:
            try:
                t_dad1 = df_dad1['time (min)'].to_numpy()
                t_dad2 = df_dad2['time (min)'].to_numpy()

                # Calcular R² per cada λ
                pearson_per_wl = {}
                for wl in wavelengths:
                    if wl in df_dad1.columns and wl in df_dad2.columns:
                        y1_wl = df_dad1[wl].to_numpy()
                        y2_wl = df_dad2[wl].to_numpy()

                        # Validar longituds
                        if len(t_dad1) != len(y1_wl) or len(t_dad2) != len(y2_wl):
                            continue

                        # Interpolar si cal
                        if len(y1_wl) != len(y2_wl):
                            t_common = t_dad1 if len(t_dad1) <= len(t_dad2) else t_dad2
                            y1_wl = np.interp(t_common, t_dad1, y1_wl)
                            y2_wl = np.interp(t_common, t_dad2, y2_wl)

                        # Pearson per aquesta λ
                        try:
                            pearson_wl, _ = pearsonr(y1_wl, y2_wl)
                            pearson_per_wl[wl] = float(pearson_wl)
                        except Exception:
                            pass

                # Guardar resultats
                result["dad"]["pearson_per_wavelength"] = pearson_per_wl

                # Trobar mínim i la seva λ
                if pearson_per_wl:
                    min_wl = min(pearson_per_wl, key=pearson_per_wl.get)
                    result["dad"]["pearson_min"] = pearson_per_wl[min_wl]
                    result["dad"]["wavelength_min"] = min_wl

                    # Warning si mínim és baix
                    if pearson_per_wl[min_wl] < REPLICA_PEARSON_THRESHOLD:
                        result["dad"]["warnings"].append(create_anomaly("LOW_CORRELATION_254",
                            details={"wavelength": min_wl, "pearson": pearson_per_wl[min_wl],
                                     "threshold": REPLICA_PEARSON_THRESHOLD}))

                # Mantenir pearson_254 per compatibilitat
                if '254' in pearson_per_wl:
                    result["dad"]["pearson_254"] = pearson_per_wl['254']

                # Diferència àrea 254
                areas1_254 = r1_result.get("areas", {}).get("A254", {})
                areas2_254 = r2_result.get("areas", {}).get("A254", {})
                a1_254 = areas1_254.get("total", 0)
                a2_254 = areas2_254.get("total", 0)

                if max(a1_254, a2_254) > 0:
                    diff_254 = abs(a1_254 - a2_254) / max(a1_254, a2_254) * 100
                    result["dad"]["area_diff_pct"] = diff_254
                    if diff_254 > REPLICA_AREA_DIFF_THRESHOLD:
                        result["dad"]["warnings"].append(create_anomaly("AREA_DIFF_HIGH_254",
                            details={"diff_pct": diff_254, "threshold": REPLICA_AREA_DIFF_THRESHOLD}))

            except Exception:
                pass

    return result


def recommend_replica(r1_result, r2_result, comparison, mode="COLUMN"):
    """
    Recomana la millor rèplica per DOC i DAD independentment.

    Args:
        r1_result: Resultat de analyze_sample() per rèplica 1
        r2_result: Resultat de analyze_sample() per rèplica 2
        comparison: Resultat de compare_replicas()
        mode: "COLUMN" o "BP"

    Returns:
        dict amb:
            - doc: { replica: "1"|"2", score: float, reason: str }
            - dad: { replica: "1"|"2", score: float, reason: str }
    """
    result = {
        "doc": {"replica": "1", "score": 0.5, "reason": "Default"},
        "dad": {"replica": "1", "score": 0.5, "reason": "Default"}
    }

    is_bp = mode.upper() == "BP"

    # =========================================================================
    # RECOMANACIÓ DOC
    # =========================================================================
    # Criteri principal: sense anomalies
    # Criteri secundari: SNR més alt

    anom1 = r1_result.get("anomalies", [])
    anom2 = r2_result.get("anomalies", [])
    snr1 = r1_result.get("snr_info", {}).get("snr_direct", 0) or 0
    snr2 = r2_result.get("snr_info", {}).get("snr_direct", 0) or 0

    # Anomalies crítiques (del catàleg)
    codes1 = get_anomaly_codes(anom1)
    codes2 = get_anomaly_codes(anom2)
    has_critical1 = bool(codes1 & CRITICAL_ANOMALIES)
    has_critical2 = bool(codes2 & CRITICAL_ANOMALIES)

    # Sets derivats del catàleg per reparabilitat
    irreparable_codes = {c for c, e in ANOMALY_CATALOG.items()
                         if e.get("invalidates") and not e.get("repairable")}
    repairable_codes = {c for c, e in ANOMALY_CATALOG.items()
                        if e.get("severity") == WarningLevel.BLOCKER and e.get("repairable")}

    if has_critical1 and not has_critical2:
        result["doc"] = {"replica": "2", "score": 0.95, "reason": "R1 té anomalies crítiques"}
    elif has_critical2 and not has_critical1:
        result["doc"] = {"replica": "1", "score": 0.95, "reason": "R2 té anomalies crítiques"}
    elif has_critical1 and has_critical2:
        # Ambdues tenen anomalies crítiques
        has_irreparable1 = bool(codes1 & irreparable_codes)
        has_irreparable2 = bool(codes2 & irreparable_codes)
        has_batman1 = bool(codes1 & repairable_codes)
        has_batman2 = bool(codes2 & repairable_codes)

        if has_irreparable1 and has_irreparable2:
            # Ambdues amb anomalies no reparables → mostra no vàlida
            if snr2 > snr1:
                result["doc"] = {"replica": "2", "score": 0.3,
                                 "reason": "Ambdues amb anomalies no reparables",
                                 "valid": False}
            else:
                result["doc"] = {"replica": "1", "score": 0.3,
                                 "reason": "Ambdues amb anomalies no reparables",
                                 "valid": False}
        elif has_batman1 or has_batman2:
            # Almenys una té Batman (reparable) → triar la millor, suggerir reparació
            repairable = []
            if has_batman1 and not has_irreparable1:
                repairable.append("1")
            if has_batman2 and not has_irreparable2:
                repairable.append("2")

            if len(repairable) == 2:
                # Ambdues reparables, triar per SNR
                chosen = "2" if snr2 > snr1 else "1"
                result["doc"] = {"replica": chosen, "score": 0.4,
                                 "reason": "Ambdues amb Batman (reparable), triar per SNR",
                                 "repairable": True,
                                 "repairable_replicas": repairable}
            elif len(repairable) == 1:
                result["doc"] = {"replica": repairable[0], "score": 0.4,
                                 "reason": f"R{repairable[0]} amb Batman (reparable)",
                                 "repairable": True,
                                 "repairable_replicas": repairable}
            else:
                # Cap reparable (p.ex. ambdues amb irreparable + batman)
                if snr2 > snr1:
                    result["doc"] = {"replica": "2", "score": 0.3,
                                     "reason": "Ambdues amb anomalies crítiques",
                                     "valid": False}
                else:
                    result["doc"] = {"replica": "1", "score": 0.3,
                                     "reason": "Ambdues amb anomalies crítiques",
                                     "valid": False}
        else:
            # Anomalies crítiques sense Batman → no vàlida
            if snr2 > snr1:
                result["doc"] = {"replica": "2", "score": 0.3,
                                 "reason": "Ambdues amb anomalies crítiques",
                                 "valid": False}
            else:
                result["doc"] = {"replica": "1", "score": 0.3,
                                 "reason": "Ambdues amb anomalies crítiques",
                                 "valid": False}
    else:
        # Cap anomalia crítica, triar per SNR
        if snr2 > snr1 * 1.1:  # R2 ha de ser >10% millor
            result["doc"] = {"replica": "2", "score": 0.85, "reason": "SNR superior"}
        elif snr1 > snr2 * 1.1:
            result["doc"] = {"replica": "1", "score": 0.85, "reason": "SNR superior"}
        else:
            result["doc"] = {"replica": "1", "score": 0.75, "reason": "SNR similar, preferència R1"}

    # =========================================================================
    # RECOMANACIÓ DAD
    # =========================================================================
    # BP: SNR alt, FWHM consistent
    # COLUMN: SNR alt, deriva baseline baixa

    snr_dad1 = r1_result.get("snr_info_dad", {}).get("A254", {}).get("snr", 0) or 0
    snr_dad2 = r2_result.get("snr_info_dad", {}).get("A254", {}).get("snr", 0) or 0

    if is_bp:
        # BP: prioritzar SNR i FWHM
        fwhm1 = r1_result.get("fwhm_254", 0) or 0
        fwhm2 = r2_result.get("fwhm_254", 0) or 0

        if snr_dad2 > snr_dad1 * 1.1:
            result["dad"] = {"replica": "2", "score": 0.85, "reason": "SNR 254nm superior"}
        elif snr_dad1 > snr_dad2 * 1.1:
            result["dad"] = {"replica": "1", "score": 0.85, "reason": "SNR 254nm superior"}
        else:
            result["dad"] = {"replica": "1", "score": 0.75, "reason": "SNR similar, preferència R1"}
    else:
        # COLUMN: prioritzar SNR i deriva baseline
        # TODO: Implementar detecció deriva baseline DAD
        if snr_dad2 > snr_dad1 * 1.1:
            result["dad"] = {"replica": "2", "score": 0.85, "reason": "SNR 254nm superior"}
        elif snr_dad1 > snr_dad2 * 1.1:
            result["dad"] = {"replica": "1", "score": 0.85, "reason": "SNR 254nm superior"}
        else:
            result["dad"] = {"replica": "1", "score": 0.75, "reason": "SNR similar, preferència R1"}

    return result


def repair_batman_in_replica(sample_result, signal="direct"):
    """
    Repara Batman en una rèplica usant repair_with_parabola().

    Modifica in-place el sample_result: actualitza y_doc_net (o y_doc_uib_net),
    recalcula àrees, i guarda traçabilitat de la reparació.

    Args:
        sample_result: Dict retornat per analyze_sample()
        signal: "direct" o "uib"

    Returns:
        dict amb info de reparació:
            - repaired: True/False
            - signal: "direct"/"uib"
            - repair_info: detalls de la reparació
            - original_y: array original (backup)
            - original_areas: àrees originals (backup)
    """
    from hpsec_core import repair_with_parabola

    t = np.asarray(sample_result.get("t_doc", []))
    if len(t) == 0:
        return {"repaired": False, "reason": "No time data"}

    repair_result = {"repaired": False, "signal": signal}

    if signal == "direct":
        y_key = "y_doc_net"
        anom_key = "BATMAN_DIRECT"
        batman_key = "batman_direct"
        batman_info_key = "batman_direct_info"
        areas_key = "areas"
    else:
        y_key = "y_doc_uib_net"
        anom_key = "BATMAN_UIB"
        batman_key = "batman_uib"
        batman_info_key = "batman_uib_info"
        areas_key = "areas_uib"

    y_original = np.asarray(sample_result.get(y_key, []))
    if len(y_original) == 0:
        return {"repaired": False, "reason": f"No {signal} data"}

    # Guardar dades originals per traçabilitat
    repair_result["original_y"] = y_original.copy().tolist()
    repair_result["original_areas"] = sample_result.get(areas_key, {}).copy()

    # Aplicar reparació
    y_repaired, repair_info, was_repaired = repair_with_parabola(t, y_original)

    if not was_repaired:
        return {"repaired": False, "reason": "repair_with_parabola failed",
                "repair_info": repair_info}

    # Actualitzar senyal reparat
    sample_result[y_key] = y_repaired.tolist()
    sample_result[f"{y_key}_original"] = y_original.tolist()  # Backup

    # Marcar anomalia Batman com a reparada
    anomalies = sample_result.get("anomalies", [])
    if not mark_repaired(anomalies, anom_key, repair_info=repair_info):
        # Fallback per strings antics (backward compat)
        if anom_key in anomalies:
            anomalies.remove(anom_key)
            anomalies.append(f"{anom_key}_REPAIRED")
    sample_result[batman_key] = False
    sample_result[f"{batman_key}_repaired"] = True
    sample_result[f"{batman_key}_repair_info"] = {
        "t_max": repair_info.get("t_max"),
        "y_max_original": repair_info.get("y_max_original"),
        "y_max_theoretical": repair_info.get("y_max_theoretical"),
        "coeffs": repair_info.get("coeffs"),
    }

    # Recalcular àrees amb senyal reparat
    try:
        new_areas = calcular_fraccions_temps(t, y_repaired)
        if signal == "direct":
            if "areas" not in sample_result:
                sample_result["areas"] = {}
            sample_result["areas"]["DOC"] = new_areas
        else:
            sample_result["areas_uib"] = new_areas
    except Exception as e:
        repair_result["areas_recalc_error"] = str(e)

    repair_result["repaired"] = True
    repair_result["repair_info"] = {
        "t_max": repair_info.get("t_max"),
        "y_max_original": repair_info.get("y_max_original"),
        "y_max_theoretical": repair_info.get("y_max_theoretical"),
    }

    return repair_result


def quantify_sample(sample_result, calibration_data, mode="COLUMN", seq_date=None):
    """
    Aplica calibració GLOBAL per convertir àrees a concentracions.

    Utilitza rf_mass_cal de Calibration_Reference.json (calibració global versionada).

    Fórmules segons model:
    - origin:    ppm = Area × 1000 / (rf_mass_cal × volume_uL)
    - intercept: ppm = (Area - intercept) × 1000 / (rf_mass_cal × volume_uL)

    Args:
        sample_result: Resultat de analyze_sample()
        calibration_data: Dict amb dades de calibració local (per volum i shift)
        mode: "COLUMN" o "BP"
        seq_date: Data de la SEQ per seleccionar calibració (None = activa)

    Returns:
        dict amb:
            - concentration_ppm: concentració total DOC Direct (ppm) - compatibilitat
            - concentration_ppm_direct: concentració DOC Direct (ppm)
            - concentration_ppm_uib: concentració DOC UIB (ppm) - si DUAL
            - fractions: dict amb concentració per fracció (COLUMN)
            - fractions_uib: dict amb concentració UIB per fracció (COLUMN DUAL)
            - calibration_source: "GLOBAL" o "LOCAL" (fallback)
            - rf_mass_cal_used: rf_mass_cal utilitzat
            - intercept: valor intercept aplicat (0 si origin)
    """
    result = {
        "concentration_ppm": None,
        "concentration_ppm_direct": None,
        "concentration_ppm_uib": None,
        "fractions": {},
        "fractions_uib": {},
        "calibration_source": None,
        "rf_mass_cal_used": None,
        "intercept": 0
    }

    if not sample_result.get("processed"):
        return result

    # =========================================================================
    # OBTENIR VOLUM D'INJECCIÓ
    # =========================================================================
    volume_uL = sample_result.get("inj_volume")
    if volume_uL is None and calibration_data:
        volume_uL = calibration_data.get("volume_uL") or calibration_data.get("inj_volume")
    if volume_uL is None:
        # Default segons mode
        volume_uL = 100 if mode.upper() == "BP" else 400

    # =========================================================================
    # OBTENIR rf_mass_cal GLOBAL I INTERCEPT
    # =========================================================================
    mode_key = mode.lower()  # 'column' o 'bp'

    # Intentar usar calibració global
    rf_mass_direct = get_rf_mass_cal(signal='direct', mode=mode_key, seq_date=seq_date)
    rf_mass_uib = get_rf_mass_cal(signal='uib', mode=mode_key, seq_date=seq_date)

    # Obtenir intercept per signal/mode (0 si origin, ex: BP)
    intercept = get_calibration_intercept(signal='direct', mode=mode_key, seq_date=seq_date)

    use_global = rf_mass_direct is not None and rf_mass_direct > 0

    # Fórmula única: ppm = (Area - intercept) × 1000 / (rf_mass × volume)
    def apply_formula(area, rf_mass):
        area_corrected = max(0, area - intercept)
        return area_corrected * 1000 / (rf_mass * volume_uL)

    # =========================================================================
    # QUANTIFICACIÓ DOC DIRECT
    # =========================================================================
    areas_direct = sample_result.get("areas", {}).get("DOC", {})
    area_total_direct = areas_direct.get("total", 0)

    if area_total_direct > 0:
        if use_global:
            ppm_direct = apply_formula(area_total_direct, rf_mass_direct)
            result["concentration_ppm_direct"] = float(ppm_direct)
            result["concentration_ppm"] = float(ppm_direct)
            result["calibration_source"] = "GLOBAL"
            result["rf_mass_cal_used"] = rf_mass_direct
            result["intercept"] = intercept
        else:
            # Fallback: usar RF local (àrea/ppm) si disponible
            rf_local = None
            if calibration_data:
                rf_local = calibration_data.get("rf_direct") or calibration_data.get("rf")
                if rf_local and rf_local > 0:
                    ppm_direct = area_total_direct / rf_local
                    result["concentration_ppm_direct"] = float(ppm_direct)
                    result["concentration_ppm"] = float(ppm_direct)
                    result["calibration_source"] = "LOCAL"

        # Concentracions per fracció (només COLUMN)
        if mode.upper() == "COLUMN":
            if use_global:
                for frac in ["BioP", "HS", "BB", "SB", "LMW"]:
                    area_frac = areas_direct.get(frac, 0)
                    if area_frac > 0:
                        result["fractions"][frac] = float(apply_formula(area_frac, rf_mass_direct))
                    else:
                        result["fractions"][frac] = 0.0
            elif calibration_data and calibration_data.get("rf_direct"):
                rf_local = calibration_data.get("rf_direct") or calibration_data.get("rf")
                if rf_local and rf_local > 0:
                    for frac in ["BioP", "HS", "BB", "SB", "LMW"]:
                        area_frac = areas_direct.get(frac, 0)
                        if area_frac > 0:
                            result["fractions"][frac] = float(area_frac / rf_local)
                        else:
                            result["fractions"][frac] = 0.0

    # =========================================================================
    # QUANTIFICACIÓ DOC UIB (si DUAL i rf_mass_uib disponible)
    # =========================================================================
    areas_uib = sample_result.get("areas_uib", {})
    area_total_uib = areas_uib.get("total", 0)

    if area_total_uib > 0:
        if rf_mass_uib and rf_mass_uib > 0:
            # Usar fórmula global amb model (origin/intercept)
            ppm_uib = apply_formula(area_total_uib, rf_mass_uib)
            result["concentration_ppm_uib"] = float(ppm_uib)

            # Concentracions UIB per fracció (només COLUMN)
            if mode.upper() == "COLUMN":
                for frac in ["BioP", "HS", "BB", "SB", "LMW"]:
                    area_frac = areas_uib.get(frac, 0)
                    if area_frac > 0:
                        result["fractions_uib"][frac] = float(apply_formula(area_frac, rf_mass_uib))
                    else:
                        result["fractions_uib"][frac] = 0.0
        else:
            # Fallback: usar RF UIB local si disponible
            if calibration_data:
                rf_uib_local = calibration_data.get("rf_uib", 0)
                if rf_uib_local and rf_uib_local > 0:
                    result["concentration_ppm_uib"] = float(area_total_uib / rf_uib_local)

                    if mode.upper() == "COLUMN":
                        for frac in ["BioP", "HS", "BB", "SB", "LMW"]:
                            area_frac = areas_uib.get(frac, 0)
                            if area_frac > 0:
                                result["fractions_uib"][frac] = float(area_frac / rf_uib_local)
                            else:
                                result["fractions_uib"][frac] = 0.0

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

    sample_name = sample_data.get("name", "UNKNOWN")
    seq_name = sample_data.get("seq_name", "")

    result = {
        "name": sample_name,
        "replica": sample_data.get("replica", "1"),
        "inj_volume": sample_data.get("inj_volume"),  # Preservar per quantificació
        "injection_index": sample_data.get("injection_index"),  # Ordre d'injecció
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

    # Filtrar DAD a les 6 λ seleccionades + submostreig (reduir matriu)
    if df_dad is not None and hasattr(df_dad, 'columns') and len(df_dad.columns) > 8:
        df_dad = analyze_dad(df_dad, config)

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
        shift_uib = calibration_data.get("shift_uib") or calibration_data.get("shift_min_u") or 0.0
        shift_direct = calibration_data.get("shift_direct") or calibration_data.get("shift_min") or 0.0

        if is_dual:
            # UIB: interpolar a escala t_doc (referencia Direct)
            if t_doc_uib is not None and y_doc_uib is not None and len(y_doc_uib) > 0:
                t_uib_arr = np.asarray(t_doc_uib).flatten()
                # Validar que t_uib i y_uib tenen la mateixa longitud
                if len(t_uib_arr) == len(y_doc_uib):
                    # Aplicar shift + interpolació a RAW
                    if abs(shift_uib) > 0.001:
                        y_doc_uib = apply_shift(t_doc, t_uib_arr, y_doc_uib, shift_uib)
                    elif len(t_uib_arr) != len(t_doc):
                        y_doc_uib = np.interp(t_doc, t_uib_arr, y_doc_uib, left=0, right=0)
                    # Aplicar shift + interpolació a NET (si disponible)
                    if y_doc_uib_net_precomp is not None and len(t_uib_arr) == len(y_doc_uib_net_precomp):
                        if abs(shift_uib) > 0.001:
                            y_doc_uib_net_precomp = apply_shift(t_doc, t_uib_arr, y_doc_uib_net_precomp, shift_uib)
                        elif len(t_uib_arr) != len(t_doc):
                            y_doc_uib_net_precomp = np.interp(t_doc, t_uib_arr, y_doc_uib_net_precomp, left=0, right=0)
                    # Marcar UIB com a ja alineat amb t_doc
                    t_doc_uib = t_doc
                else:
                    # Longitud no coincideix - invalidar UIB
                    y_doc_uib = None
                    y_doc_uib_net_precomp = None
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
        # Validar que t i y tenen la mateixa longitud
        if len(t_uib_arr) == len(y_doc_uib) and len(t_uib_arr) != len(t_doc):
            # Interpolar UIB RAW a escala Direct
            y_doc_uib = np.interp(t_doc, t_uib_arr, y_doc_uib, left=0, right=0)
            # Interpolar UIB NET si disponible
            if y_doc_uib_net_precomp is not None and len(t_uib_arr) == len(y_doc_uib_net_precomp):
                y_doc_uib_net_precomp = np.interp(t_doc, t_uib_arr, y_doc_uib_net_precomp, left=0, right=0)
        elif len(t_uib_arr) != len(y_doc_uib):
            # Longitud no coincideix - invalidar UIB
            y_doc_uib = None
            y_doc_uib_net_precomp = None

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
                result["anomalies"].append(create_anomaly("UIB_NO_BASELINE"))
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
    timeout_info = detect_timeout(t_doc, is_bp=is_bp)
    timeout_positions = timeout_info.get("t_positions", [])

    if timeout_info.get("n_timeouts", 0) > 0:
        result["timeout_info"] = timeout_info

    # Detectar pic principal
    y_smooth = apply_smoothing(y_doc_net)
    peak_info = detect_main_peak(t_doc, y_smooth, config.get("peak_min_prominence_pct", 5.0), is_bp=is_bp)

    if not peak_info.get("valid"):
        result["anomalies"].append(create_anomaly("NO_PEAK"))
    else:
        # Detectar Batman DOC Direct
        batman_result = detect_batman(t_doc, y_smooth)
        if batman_result.get("is_batman"):
            result["anomalies"].append(create_anomaly("BATMAN_DIRECT", details=batman_result))
            result["batman_direct"] = True
            result["batman_direct_info"] = batman_result
            # Log per dev notes (si actiu)
            _log_detection_issue(seq_name, sample_name, "batman", "direct", batman_result)
        else:
            result["batman_direct"] = False

    # Detectar Batman UIB (si DUAL)
    if is_dual and y_doc_uib_net is not None and len(y_doc_uib_net) > 0:
        y_uib_smooth = apply_smoothing(y_doc_uib_net)
        batman_uib_result = detect_batman(t_doc, y_uib_smooth)
        if batman_uib_result.get("is_batman"):
            result["anomalies"].append(create_anomaly("BATMAN_UIB", details=batman_uib_result))
            result["batman_uib"] = True
            result["batman_uib_info"] = batman_uib_result
            # Log per dev notes (si actiu)
            _log_detection_issue(seq_name, sample_name, "batman", "uib", batman_uib_result)
        else:
            result["batman_uib"] = False

    result["peak_info"] = peak_info

    # Check TIMEOUT_IN_PEAK: timeout que afecta el pic principal DOC
    if timeout_positions and peak_info.get("valid"):
        t_start_peak = peak_info.get("t_start", t_doc[peak_info.get("left_idx", 0)])
        t_end_peak = peak_info.get("t_end", t_doc[peak_info.get("right_idx", len(t_doc) - 1)])
        # Comprovar si qualsevol timeout (inici o fi del gap) cau dins del pic
        timeout_details = timeout_info.get("timeouts", [])
        timeout_in_peak = any(
            (t_start_peak <= to.get("t_start_min", 0) <= t_end_peak) or
            (t_start_peak <= to.get("t_end_min", 0) <= t_end_peak) or
            (to.get("t_start_min", 0) <= t_start_peak and to.get("t_end_min", 0) >= t_end_peak)
            for to in timeout_details
        )
        if timeout_in_peak:
            result["anomalies"].append(create_anomaly("TIMEOUT_IN_PEAK", details={"timeout_info": timeout_info}))
            result["timeout_in_peak"] = True

    # Calcular FWHM i simetria del pic principal
    if peak_info.get("valid"):
        peak_idx = peak_info.get("peak_idx", 0)
        left_idx = peak_info.get("left_idx", 0)
        right_idx = peak_info.get("right_idx", len(y_doc_net) - 1)

        fwhm_doc = calculate_fwhm(t_doc, y_smooth, peak_idx, left_idx, right_idx)
        sym_doc = calculate_symmetry(t_doc, y_smooth, peak_idx, left_idx, right_idx)

        result["fwhm_doc"] = fwhm_doc
        result["symmetry_doc"] = sym_doc

        # Bi-Gaussian fit DOC (NOMÉS BP mode)
        if is_bp:
            try:
                # Ampliar límits per fit (necessita més punts que per integració)
                if fwhm_doc and fwhm_doc > 0:
                    dt = t_doc[1] - t_doc[0] if len(t_doc) > 1 else 0.07
                    n_points_fwhm = int(fwhm_doc * 3 / dt)
                    fit_left = max(0, peak_idx - max(n_points_fwhm, 30))
                    fit_right = min(len(t_doc) - 1, peak_idx + max(n_points_fwhm, 30))
                else:
                    n_fallback = max(30, len(t_doc) // 5)
                    fit_left = max(0, peak_idx - n_fallback)
                    fit_right = min(len(t_doc) - 1, peak_idx + n_fallback)

                bigauss_result = fit_bigaussian(t_doc, y_smooth, peak_idx, fit_left, fit_right)
                if bigauss_result.get("valid"):
                    result["bigaussian_doc"] = {
                        "r2": bigauss_result.get("r2", 0),
                        "amplitude": bigauss_result.get("amplitude", 0),
                        "mu": bigauss_result.get("mu", 0),
                        "sigma_left": bigauss_result.get("sigma_left", 0),
                        "sigma_right": bigauss_result.get("sigma_right", 0),
                        "asymmetry": bigauss_result.get("asymmetry", 1),
                        "valid": True,
                    }
                    r2 = bigauss_result.get("r2", 0)
                    if r2 >= THRESH_R2_VALID:
                        result["bigaussian_doc"]["quality"] = "VALID"
                    elif r2 >= THRESH_R2_CHECK:
                        result["bigaussian_doc"]["quality"] = "CHECK"
                    else:
                        result["bigaussian_doc"]["quality"] = "INVALID"
            except Exception:
                result["bigaussian_doc"] = {"valid": False, "r2": 0}

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
                    # Baseline unificada
                    baseline_254 = get_baseline_value(t_dad, y_254, mode="BP")
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

                        # Bi-Gaussian fit DAD 254 (BP mode)
                        try:
                            bigauss_254 = fit_bigaussian(t_dad, y_254_smooth, peak_idx_254, left_idx_254, right_idx_254)
                            if bigauss_254.get("valid"):
                                result["bigaussian_254"] = {
                                    "r2": bigauss_254.get("r2", 0),
                                    "asymmetry": bigauss_254.get("asymmetry", 1),
                                    "valid": True,
                                }
                                r2_254 = bigauss_254.get("r2", 0)
                                if r2_254 >= THRESH_R2_VALID:
                                    result["bigaussian_254"]["quality"] = "VALID"
                                elif r2_254 >= THRESH_R2_CHECK:
                                    result["bigaussian_254"]["quality"] = "CHECK"
                                else:
                                    result["bigaussian_254"]["quality"] = "INVALID"
                        except Exception:
                            pass
            except Exception:
                pass

    # Calcular àrees per fraccions (inclou DAD si disponible)
    # LMW s'inclou al total (és senyal real). mode passa per baseline DAD correcta.
    mode_type = "BP" if is_bp else "COLUMN"
    areas = calcular_arees_fraccions_complet(
        t_doc, y_doc_net, df_dad, config,
        mode=mode_type)

    result["areas"] = areas
    result["mode"] = mode_type

    # Detectar tmax senyals
    tmax_signals = detectar_tmax_senyals(t_doc, y_doc_net, df_dad, config,
                                         mode=mode_type)
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

    # --- LOD/LOQ flags basats en SNR ---
    snr_direct = snr_info.get("snr_direct", 0)
    if snr_direct and 0 < snr_direct < 3:
        result["anomalies"].append(create_anomaly("BELOW_LOD", details={"snr": snr_direct}))
    elif snr_direct and 3 <= snr_direct < 10:
        result["anomalies"].append(create_anomaly("BELOW_LOQ", details={"snr": snr_direct}))

    # Calcular SNR info per DAD (totes les wavelengths)
    dad_snr_info = calculate_dad_snr_info(df_dad, config.get("target_wavelengths"))
    if dad_snr_info:
        result["snr_info_dad"] = dad_snr_info

    # =========================================================================
    # ÀREES UIB (per DUAL o quan només hi ha UIB)
    # =========================================================================
    # A08: Calcular areas_uib si:
    #   - Mode DUAL amb y_doc_uib_net disponible
    #   - O mode simple però dades venen d'UIB (is_uib_only)
    is_uib_only = sample_data.get("is_uib_only", False)

    if is_dual and "DOC" in areas and y_doc_uib_net is not None:
        areas_uib = calcular_fraccions_temps(t_doc, y_doc_uib_net, config)
        result["areas_uib"] = areas_uib
    elif is_uib_only and "DOC" in areas:
        # Només UIB: les àrees DOC ja són d'UIB, copiar a areas_uib
        result["areas_uib"] = areas.get("DOC", {}).copy()

    # =========================================================================
    # MÈTRIQUES ADDICIONALS (només COLUMN)
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
        if is_dual and "areas_uib" in result:
            areas_direct = areas.get("DOC", {})
            areas_uib = result["areas_uib"]
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

        # --- Nombre de pics per senyal dins totes les zones ---
        # n_peaks_per_wl: {"DOC": {"BioP": 1, "HS": 2, ...}, "A254": {...}, ...}
        from hpsec_config import get_config as _get_cfg
        _cfg = _get_cfg()
        _fracs = _cfg.get_all_fractions("BP" if is_bp else "COLUMN")
        zones_detect = [(fn, fi["start"], fi["end"]) for fn, fi in _fracs]

        n_peaks_per_wl = {}

        def _count_peaks_in_zones(t_arr, y_arr, signal_key):
            sig_peaks = {}
            try:
                for zone_name, z_start, z_end in zones_detect:
                    mask = (t_arr >= z_start) & (t_arr <= z_end)
                    if np.sum(mask) > 10:
                        y_zone = y_arr[mask]
                        y_range = np.max(y_zone) - np.min(y_zone)
                        min_prom = max(y_range * 0.05, 0.01)
                        peaks, _ = find_peaks(y_zone, prominence=min_prom, distance=3)
                        sig_peaks[zone_name] = len(peaks)
                    else:
                        sig_peaks[zone_name] = 0
            except Exception:
                pass
            if sig_peaks:
                n_peaks_per_wl[signal_key] = sig_peaks

        # DOC
        if t_doc is not None and y_doc_net is not None and len(t_doc) > 0:
            _count_peaks_in_zones(np.asarray(t_doc), np.asarray(y_doc_net), "DOC")

        # UIB
        if is_dual and y_doc_uib_net is not None and len(y_doc_uib_net) > 0:
            _count_peaks_in_zones(np.asarray(t_doc), np.asarray(y_doc_uib_net), "UIB")

        # DAD wavelengths
        if df_dad is not None and not df_dad.empty and 'time (min)' in df_dad.columns:
            t_dad = df_dad['time (min)'].to_numpy()
            wl_cols_detect = [c for c in df_dad.columns if c != 'time (min)']
            for wl_col in wl_cols_detect:
                wl_key = f"A{wl_col}" if not str(wl_col).startswith('A') else wl_col
                _count_peaks_in_zones(t_dad, df_dad[wl_col].to_numpy(), wl_key)

        result["n_peaks_per_wl"] = n_peaks_per_wl
        # Backwards compat
        result["n_peaks_254_HS"] = n_peaks_per_wl.get("A254", {}).get("HS")

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
        tuple de 4 llistes: (samples, khp_samples, control_samples, light_samples)
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
    control_samples = []  # Kept empty for backward compat (result dict still has the key)
    light_samples = []    # BLANK + CONTROL → lightweight analysis

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
                "injection_index": inj_info.get("line_num"),  # Ordre d'injecció al MasterFile
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
                    if "y_net" in uib and uib["y_net"] is not None:
                        flat_sample["y_doc_uib_net"] = uib["y_net"]
                    if "baseline" in uib:
                        flat_sample["baseline_uib"] = uib["baseline"]

                # Fallback DUAL: si nomes hi ha un senyal, convertir a mode simple
                if has_uib and not has_direct:
                    # A08: Marcar que les dades venen d'UIB per calcular areas_uib
                    flat_sample["is_uib_only"] = True
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

            # Extreure dades DAD — filtrar a λ seleccionades des del principi
            dad = rep_data.get("dad", {})
            if dad and "df" in dad:
                df_dad_raw = dad["df"]
                if hasattr(df_dad_raw, 'columns') and len(df_dad_raw.columns) > 8:
                    df_dad_raw = analyze_dad(df_dad_raw)
                flat_sample["df_dad"] = df_dad_raw

            # Classificar segons tipus
            if sample_type == "KHP":
                khp_samples.append(flat_sample)
            elif sample_type in ("BLANK", "CONTROL"):
                light_samples.append(flat_sample)
            else:
                samples.append(flat_sample)

    return samples, khp_samples, control_samples, light_samples


# =============================================================================
# GENERACIÓ D'AVISOS ESTRUCTURATS PER ANÀLISI
# =============================================================================

def _generate_analysis_warnings(result: dict) -> list:
    """
    Genera avisos estructurats a partir del resultat d'anàlisi.

    Ara les anomalies ja són dicts estructurats des de l'origen (create_anomaly).
    Aquesta funció agrega totes les anomalies de totes les rèpliques + comparacions,
    i afegeix errors de nivell de seqüència.

    Args:
        result: Dict del resultat de analyze_sequence()

    Returns:
        Llista d'avisos estructurats (mix de anomaly dicts + warning dicts)
    """
    warnings = []

    # 1. Errors crítics de seqüència (BLOCKER) — no són anomalies per-rèplica
    for error in result.get("errors", []):
        if "calibr" in error.lower():
            warnings.append(create_warning(
                code="ANA_NO_CALIBRATION",
                stage="analyze",
            ))
        else:
            warnings.append(create_warning(
                code="ANA_ERROR",
                level=WarningLevel.BLOCKER,
                message=error,
                stage="analyze",
            ))

    # 2. Mostres buides
    n_empty = 0
    for sample_name, sample_group in result.get("samples_grouped", {}).items():
        replicas = sample_group.get("replicas", {})
        if not replicas or not any(r.get("processed") for r in replicas.values()):
            n_empty += 1

    if n_empty > 0:
        warnings.append(create_warning(
            code="ANA_EMPTY_SAMPLES",
            stage="analyze",
            details={"n": n_empty},
        ))

    # 3. Agregar anomalies de totes les rèpliques i comparacions
    all_anomalies = []
    for sg in result.get("samples_grouped", {}).values():
        for rep in sg.get("replicas", {}).values():
            all_anomalies.extend(rep.get("anomalies", []))
        comp = sg.get("comparison") or {}
        for domain in ("doc", "dad"):
            all_anomalies.extend(comp.get(domain, {}).get("warnings", []))

    warnings.extend(all_anomalies)
    return warnings


# =============================================================================
# ANÀLISI LIGHTWEIGHT PER BLANCS / CONTROLS
# =============================================================================
def _analyze_light_sample(sample):
    """
    Anàlisi lightweight per BLANK (MQ) i CONTROL (NaOH).

    Calcula àrea total DOC, SNR i àrea A254 — sense fraccions ni quantificació.

    Args:
        sample: Dict amb dades de la mostra (flat_sample de _flatten_samples_for_processing)

    Returns:
        dict amb: name, replica, sample_type, processed, analysis_type="light",
                  area_total, snr, area_254, t_doc, y_doc_net, inj_volume
    """
    name = sample.get("name", "UNKNOWN")
    replica = sample.get("replica", "1")
    sample_type = sample.get("sample_type", "BLANK")

    result = {
        "name": name,
        "replica": replica,
        "sample_type": sample_type,
        "processed": False,
        "analysis_type": "light",
        "area_total": 0,
        "snr": 0,
        "t_doc": sample.get("t_doc"),
        "y_doc_net": None,
        "inj_volume": sample.get("inj_volume"),
        "injection_index": sample.get("injection_index"),  # Ordre d'injecció
    }

    # Obtenir senyal DOC
    y_doc_net = sample.get("y_doc_net")
    if y_doc_net is None:
        y_doc_net = sample.get("y_doc_direct_net")
    if y_doc_net is None:
        y_doc_net = sample.get("y_doc")
    if y_doc_net is None:
        y_doc_net = sample.get("y_doc_direct")

    t_doc = sample.get("t_doc")

    if y_doc_net is None or t_doc is None:
        result["error"] = "No DOC signal available"
        return result

    y_doc_net = np.asarray(y_doc_net, dtype=float)
    t_doc = np.asarray(t_doc, dtype=float)

    # Àrea total DOC
    area_total = float(np.trapz(y_doc_net, x=t_doc))

    # SNR (simple: peak_height / noise)
    peak_height = float(np.max(y_doc_net))
    snr = calc_snr(y_doc_net, peak_height)

    result["area_total"] = area_total
    result["snr"] = snr
    result["t_doc"] = t_doc
    result["y_doc_net"] = y_doc_net
    result["processed"] = True

    # Extreure àrea A254 del DAD si disponible
    df_dad = sample.get("df_dad")
    if df_dad is not None and hasattr(df_dad, 'columns'):
        try:
            # Buscar columna 254 (pot ser "A254", "254", 254, etc.)
            col_254 = None
            for col in df_dad.columns:
                col_str = str(col).replace("A", "").strip()
                if col_str == "254":
                    col_254 = col
                    break
            if col_254 is not None:
                t_col = None
                for tc in df_dad.columns:
                    tc_str = str(tc).lower()
                    if "time" in tc_str or "min" in tc_str or tc_str == "t":
                        t_col = tc
                        break
                if t_col is not None:
                    t_dad = np.asarray(df_dad[t_col], dtype=float)
                    y_254 = np.asarray(df_dad[col_254], dtype=float)
                    result["area_254"] = float(np.trapz(y_254, x=t_dad))
        except Exception:
            pass

    return result


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
        "light_samples": [],
        "errors": [],
        "warnings": [],
    }

    # Verificar dades d'entrada
    if not imported_data.get("success"):
        result["errors"].append("Imported data is invalid")
        return result

    # Aplanar l'estructura de mostres
    data_mode = imported_data.get("data_mode", "UIB")
    all_samples, khp_samples, control_samples, light_samples = _flatten_samples_for_processing(
        imported_data, data_mode=data_mode
    )

    total_samples = len(all_samples) + len(khp_samples) + len(control_samples) + len(light_samples)
    if total_samples == 0:
        result["errors"].append("No samples to process")
        return result

    # =========================================================================
    # SUPORT MÚLTIPLES CALIBRACIONS: Seleccionar calibració segons inj_volume
    # =========================================================================
    # Si una SEQ té múltiples KHP amb diferents condicions (ex: KHP2@100µL i KHP2@50µL),
    # cada mostra usarà la calibració que coincideixi amb el seu volum d'injecció.
    seq_path = imported_data.get("seq_path", "")
    method = imported_data.get("method", "COLUMN")
    mode = "BP" if method.upper() == "BP" else "COLUMN"

    # Carregar totes les calibracions actives per aquesta SEQ
    multi_calibrations = {}  # Cache: volume -> calibration_data
    if seq_path:
        try:
            active_cals = get_all_active_calibrations(seq_path, mode)
            for cal in active_cals:
                vol = cal.get("volume_uL", 0)
                if vol > 0:
                    multi_calibrations[vol] = cal
        except Exception as e:
            result["warnings"].append(f"No s'han pogut carregar calibracions: {e}")

    def get_sample_calibration(sample):
        """Retorna la calibració correcta per una mostra segons el seu inj_volume."""
        # Si hi ha múltiples calibracions, buscar per volum
        if multi_calibrations:
            inj_vol = sample.get("inj_volume")
            if inj_vol and inj_vol in multi_calibrations:
                return multi_calibrations[inj_vol]
            # Fallback: usar la primera disponible
            if multi_calibrations:
                return list(multi_calibrations.values())[0]
        # Fallback: usar calibration_data passat (compatibilitat)
        return calibration_data

    # Processar mostres regulars
    processed_count = 0
    for i, sample in enumerate(all_samples):
        if progress_callback:
            progress_callback(f"Processing {sample.get('name', 'sample')}...", (i + 1) / total_samples * 100)

        try:
            # Seleccionar calibració segons volum d'injecció de la mostra
            sample_cal = get_sample_calibration(sample)
            processed = analyze_sample(sample, sample_cal, config)
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
            sample_cal = get_sample_calibration(khp)
            processed = analyze_sample(khp, sample_cal, config)
            result["khp_samples"].append(processed)
        except Exception as e:
            result["errors"].append(f"KHP {khp.get('name')}: {str(e)}")

        processed_count += 1

    # Processar controls (backward compat — control_samples queda buit, van a light)
    for i, ctrl in enumerate(control_samples):
        if progress_callback:
            progress_callback(f"Processing {ctrl.get('name', '')}...", (processed_count + i + 1) / total_samples * 100)

        try:
            sample_cal = get_sample_calibration(ctrl)
            processed = analyze_sample(ctrl, sample_cal, config)
            result["control_samples"].append(processed)
        except Exception as e:
            result["errors"].append(f"Control {ctrl.get('name')}: {str(e)}")

    # Processar light samples (BLANK + CONTROL) — anàlisi lleugera
    for i, light in enumerate(light_samples):
        if progress_callback:
            progress_callback(
                f"Processing {light.get('name', '')} (light)...",
                (processed_count + len(control_samples) + i + 1) / total_samples * 100
            )

        try:
            processed = _analyze_light_sample(light)
            result["light_samples"].append(processed)
        except Exception as e:
            result["errors"].append(f"Light {light.get('name')}: {str(e)}")

    # =========================================================================
    # AGRUPAR RÈPLIQUES, COMPARAR, RECOMANAR I QUANTIFICAR
    # =========================================================================
    if progress_callback:
        progress_callback("Comparing replicas...", 90)

    # is_bp ja definit a dalt (mode, method extrets per multi-calibració)
    is_bp = mode == "BP"

    # Agrupar mostres per nom
    samples_by_name = {}
    for sample in result["samples"]:
        name = sample.get("name", "UNKNOWN")
        replica = sample.get("replica", "1")
        if name not in samples_by_name:
            samples_by_name[name] = {}
        samples_by_name[name][replica] = sample

    # Per cada mostra amb múltiples rèpliques, comparar i recomanar
    result["samples_grouped"] = {}

    for sample_name, replicas in samples_by_name.items():
        sample_group = {
            "replicas": replicas,
            "comparison": None,
            "recommendation": None,
            "selected": {"doc": "1", "dad": "1"},
            "quantification": None
        }

        replica_keys = sorted(replicas.keys())

        if len(replica_keys) >= 2:
            # Comparar R1 vs R2
            r1 = replicas.get(replica_keys[0])
            r2 = replicas.get(replica_keys[1])

            if r1 and r2:
                # Comparació
                comparison = compare_replicas(r1, r2, mode=mode, config=config)
                sample_group["comparison"] = comparison

                # Recomanació
                recommendation = recommend_replica(r1, r2, comparison, mode=mode)
                sample_group["recommendation"] = recommendation

                # Selecció inicial = recomanació
                sample_group["selected"] = {
                    "doc": recommendation["doc"]["replica"],
                    "dad": recommendation["dad"]["replica"]
                }

                # Propagare flags de validesa i reparabilitat
                doc_rec = recommendation.get("doc", {})
                sample_group["sample_valid"] = doc_rec.get("valid", True)
                sample_group["repairable"] = doc_rec.get("repairable", False)
                sample_group["repairable_replicas"] = doc_rec.get("repairable_replicas", [])
                sample_group["repaired"] = False  # Es posarà True si l'usuari repara

                # Quantificació (saltar si mostra no vàlida)
                if sample_group["sample_valid"] is False:
                    sample_group["quantification"] = {
                        "concentration_ppm": None,
                        "concentration_ppm_direct": None,
                        "concentration_ppm_uib": None,
                        "area_total": None,
                        "valid": False,
                        "reason": doc_rec.get("reason", "Mostra no vàlida")
                    }
                else:
                    selected_replica = sample_group["selected"]["doc"]
                    selected_sample = replicas.get(selected_replica, r1)
                    # Usar calibració específica segons volum d'injecció
                    sample_cal = get_sample_calibration(selected_sample)
                    quantification = quantify_sample(selected_sample, sample_cal, mode=mode)
                    sample_group["quantification"] = quantification

        elif len(replica_keys) == 1:
            # Només una rèplica
            r1 = replicas.get(replica_keys[0])
            sample_group["selected"] = {"doc": replica_keys[0], "dad": replica_keys[0]}
            sample_group["sample_valid"] = True
            sample_group["repairable"] = False
            sample_group["repairable_replicas"] = []
            sample_group["repaired"] = False

            # Quantificació
            if r1:
                # Usar calibració específica segons volum d'injecció
                sample_cal = get_sample_calibration(r1)
                quantification = quantify_sample(r1, sample_cal, mode=mode)
                sample_group["quantification"] = quantification

        result["samples_grouped"][sample_name] = sample_group

    # =========================================================================
    # AGRUPAR LIGHT SAMPLES (BLANK + CONTROL) — després de regulars
    # =========================================================================
    light_by_name = {}
    for ls in result["light_samples"]:
        name = ls.get("name", "UNKNOWN")
        replica = ls.get("replica", "1")
        if name not in light_by_name:
            light_by_name[name] = {}
        light_by_name[name][replica] = ls

    for sample_name in sorted(light_by_name.keys()):
        replicas = light_by_name[sample_name]
        first_rep = next(iter(replicas.values()))
        result["samples_grouped"][sample_name] = {
            "analysis_type": "light",
            "sample_type": first_rep.get("sample_type", "BLANK"),
            "replicas": replicas,
            "selected": {"doc": sorted(replicas.keys())[0]},
            "sample_valid": True,
        }

    # =========================================================================
    # AGRUPAR KHP SAMPLES — després de light
    # =========================================================================
    khp_by_name = {}
    for khp in result["khp_samples"]:
        name = khp.get("name", "UNKNOWN")
        replica = khp.get("replica", "1")
        if name not in khp_by_name:
            khp_by_name[name] = {}
        khp_by_name[name][replica] = khp

    for sample_name in sorted(khp_by_name.keys()):
        replicas = khp_by_name[sample_name]
        result["samples_grouped"][sample_name] = {
            "analysis_type": "khp",
            "sample_type": "KHP",
            "replicas": replicas,
            "selected": {"doc": sorted(replicas.keys())[0]},
            "sample_valid": True,
        }

    # =========================================================================
    # GENERAR RESUM
    # =========================================================================
    n_valid = sum(1 for s in result["samples"] if s.get("processed") and s.get("peak_info", {}).get("valid"))
    n_with_anomalies = sum(1 for s in result["samples"] if s.get("anomalies"))
    n_timeouts = sum(1 for s in result["samples"] if s.get("timeout_info", {}).get("n_timeouts", 0) > 0)
    n_with_warnings = sum(
        1 for sg in result["samples_grouped"].values()
        if sg.get("analysis_type") not in ("light", "khp") and sg.get("comparison") and (
            sg["comparison"].get("doc", {}).get("warnings") or
            sg["comparison"].get("dad", {}).get("warnings")
        )
    )

    result["summary"] = {
        "total_samples": len(result["samples_grouped"]),
        "total_replicas": len(result["samples"]),
        "valid_peaks": n_valid,
        "with_anomalies": n_with_anomalies,
        "with_timeouts": n_timeouts,
        "with_replica_warnings": n_with_warnings,
        "n_khp": len(result["khp_samples"]),
        "n_controls": len(result["control_samples"]),
        "n_light": len(result["light_samples"]),
    }

    result["success"] = len(result["errors"]) == 0

    # Generar avisos estructurats (nou sistema)
    result["warnings_structured"] = _generate_analysis_warnings(result)
    # warning_level: prioritzar anomalies (blocker/warning) sobre warnings genèrics
    anomaly_sev = get_max_anomaly_severity(result["warnings_structured"])
    legacy_sev = get_max_warning_level(result["warnings_structured"])
    # Usar el més greu entre anomalies i warnings legacy
    sev_order = {"blocker": 3, "warning": 2, "info": 1, "none": 0}
    result["warning_level"] = anomaly_sev if sev_order.get(anomaly_sev, 0) >= sev_order.get(legacy_sev, 0) else legacy_sev

    # Registrar mostres a l'índex global (Sample Database)
    try:
        from hpsec_samples_db import register_samples_from_analysis
        register_samples_from_analysis(result)
    except Exception as e:
        # No bloquejar l'anàlisi si falla el registre
        print(f"[WARNING] Error registrant mostres a l'índex: {e}")

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
    """Encoder JSON per tipus numpy i pandas."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="list")
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
        "date_processed": datetime.now().isoformat(),
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
        "light_samples": [],
        # Mostres agrupades per nom (per GUI)
        "samples_grouped": {},
    }

    def summarize_light_sample(sample):
        """Extreu info d'una mostra light per serialitzar a JSON."""
        t_doc = sample.get("t_doc")
        y_doc_net = sample.get("y_doc_net")
        return {
            "name": sample.get("name", ""),
            "replica": sample.get("replica", ""),
            "sample_type": sample.get("sample_type", "BLANK"),
            "processed": sample.get("processed", False),
            "analysis_type": "light",
            "area_total": sample.get("area_total", 0),
            "snr": sample.get("snr", 0),
            "area_254": sample.get("area_254", 0),
            "inj_volume": sample.get("inj_volume"),
            "t_doc": t_doc,
            "y_doc_net": y_doc_net,
        }

    def summarize_sample(sample):
        """Extreu info d'una mostra per serialitzar a JSON."""
        # Convertir df_dad (DataFrame) a dict serialitzable
        # El df_dad ja arriba filtrat a 6 λ per analyze_dad()
        df_dad = sample.get("df_dad")
        df_dad_serializable = None
        if df_dad is not None and hasattr(df_dad, 'to_dict'):
            try:
                if not df_dad.empty:
                    df_dad_serializable = df_dad.to_dict(orient="list")
            except Exception:
                pass

        return {
            # --- Camps existents ---
            "name": sample.get("name", ""),
            "replica": sample.get("replica", ""),
            "processed": sample.get("processed", False),
            "error": sample.get("error"),
            "peak_info": sample.get("peak_info", {}),
            "areas": sample.get("areas", {}),
            "areas_uib": sample.get("areas_uib", {}),
            "anomalies": sample.get("anomalies", []),
            "timeout_info": sample.get("timeout_info", {}),
            "snr_info": sample.get("snr_info", {}),
            "snr_info_dad": sample.get("snr_info_dad", {}),
            "batman_uib": sample.get("batman_uib"),
            "pearson_direct_uib": sample.get("pearson_direct_uib"),
            "area_diff_pct": sample.get("area_diff_pct"),
            # --- Camps escalars nous ---
            "tmax_signals": sample.get("tmax_signals", {}),
            "n_peaks_254_HS": sample.get("n_peaks_254_HS"),
            "n_peaks_per_wl": sample.get("n_peaks_per_wl", {}),
            "is_bp": sample.get("is_bp", False),
            "is_dual": sample.get("is_dual", False),
            "batman_direct": sample.get("batman_direct"),
            "batman_direct_info": sample.get("batman_direct_info"),
            "bigaussian_doc": sample.get("bigaussian_doc"),
            "bigaussian_254": sample.get("bigaussian_254"),
            "fwhm_doc": sample.get("fwhm_doc"),
            "symmetry_doc": sample.get("symmetry_doc"),
            "inj_volume": sample.get("inj_volume"),
            # --- Camps cromatograma (arrays) ---
            "t_doc": sample.get("t_doc"),
            "y_doc_net": sample.get("y_doc_net"),
            "y_doc_uib_net": sample.get("y_doc_uib_net"),
            "y_doc_direct_net": sample.get("y_doc_direct_net"),
            "df_dad": df_dad_serializable,
        }

    # Resumir mostres
    for sample in analysis_data.get("samples", []):
        result["samples"].append(summarize_sample(sample))

    for khp in analysis_data.get("khp_samples", []):
        result["khp_samples"].append(summarize_sample(khp))

    for ctrl in analysis_data.get("control_samples", []):
        result["control_samples"].append(summarize_sample(ctrl))

    for light in analysis_data.get("light_samples", []):
        result["light_samples"].append(summarize_light_sample(light))

    # Guardar samples_grouped (estructura agrupada per GUI)
    samples_grouped = analysis_data.get("samples_grouped", {})
    if samples_grouped:
        for sample_name, sample_data in samples_grouped.items():
            is_light = sample_data.get("analysis_type") == "light"
            if is_light:
                grouped_entry = {
                    "analysis_type": "light",
                    "sample_type": sample_data.get("sample_type", "BLANK"),
                    "replicas": {},
                    "selected": sample_data.get("selected"),
                    "sample_valid": sample_data.get("sample_valid", True),
                }
                for rep_key, rep_data in sample_data.get("replicas", {}).items():
                    grouped_entry["replicas"][rep_key] = summarize_light_sample(rep_data)
            else:
                grouped_entry = {
                    "replicas": {},
                    "comparison": sample_data.get("comparison"),
                    "recommendation": sample_data.get("recommendation"),
                    "selected": sample_data.get("selected"),
                    "quantification": sample_data.get("quantification"),
                    "sample_valid": sample_data.get("sample_valid", True),
                    "repairable": sample_data.get("repairable", False),
                    "repaired": sample_data.get("repaired", False),
                    "repair_history": sample_data.get("repair_history", []),
                }
                for rep_key, rep_data in sample_data.get("replicas", {}).items():
                    grouped_entry["replicas"][rep_key] = summarize_sample(rep_data)
            result["samples_grouped"][sample_name] = grouped_entry

    # Guardar
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        return output_path
    except Exception as e:
        print(f"Error guardant analysis_result.json: {e}")
        return None


def _restore_dataframes(data):
    """Converteix df_dad dicts a DataFrames i llistes a numpy arrays en dades carregades de JSON."""
    _ARRAY_KEYS = ("t_doc", "y_doc_net", "y_doc_uib_net", "y_doc_direct_net")

    def _restore_sample(sample):
        # Restaurar df_dad: dict → DataFrame
        df_dad_dict = sample.get("df_dad")
        if df_dad_dict and isinstance(df_dad_dict, dict):
            try:
                sample["df_dad"] = pd.DataFrame(df_dad_dict)
            except Exception:
                sample["df_dad"] = None
        # Restaurar arrays numèrics: list → numpy array
        for key in _ARRAY_KEYS:
            val = sample.get(key)
            if val is not None and isinstance(val, list):
                sample[key] = np.array(val)

    for sample in data.get("samples", []):
        _restore_sample(sample)
    for sample in data.get("khp_samples", []):
        _restore_sample(sample)
    for sample in data.get("control_samples", []):
        _restore_sample(sample)
    for sample in data.get("light_samples", []):
        _restore_sample(sample)
    for sample_data in data.get("samples_grouped", {}).values():
        for rep_data in sample_data.get("replicas", {}).values():
            _restore_sample(rep_data)


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
            data = json.load(f)
        _restore_dataframes(data)

        # Normalitzar anomalies (backward compat: strings → dicts)
        for sg in data.get("samples_grouped", {}).values():
            for rep in sg.get("replicas", {}).values():
                raw = rep.get("anomalies", [])
                if raw and any(isinstance(a, str) for a in raw):
                    rep["anomalies"] = normalize_anomalies(raw)
            # Normalitzar també warnings de comparació
            comp = sg.get("comparison") or {}
            for domain in ("doc", "dad"):
                domain_data = comp.get(domain, {})
                warns = domain_data.get("warnings", [])
                if warns and any(isinstance(w, str) for w in warns):
                    domain_data["warnings"] = normalize_anomalies(warns)

        return data
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
    _is_bp = float(np.max(t_doc)) < 20 if t_doc is not None and len(t_doc) > 0 else False
    _mode = "BP" if _is_bp else "COLUMN"
    tmax_data = detectar_tmax_senyals(t_doc, y_doc_net, df_dad, mode=_mode)
    tmax_rows = []
    for signal, tmax_val in tmax_data.items():
        tmax_rows.append((signal, round(tmax_val, 3) if tmax_val > 0 else "-"))
    df_tmax = pd.DataFrame(tmax_rows, columns=["Signal", "tmax (min)"])

    # === AREAS SHEET ===
    fraccions_data = calcular_arees_fraccions_complet(
        t_doc, y_doc_net, df_dad, mode=_mode)
    fractions_config = DEFAULT_PROCESS_CONFIG.get("time_fractions", {})
    if not fractions_config:
        try:
            from hpsec_config import get_config
            _cfg = get_config()
            for _fn, _fi in _cfg.get_all_fractions():
                fractions_config[_fn] = [_fi["start"], _fi["end"]]
        except Exception:
            pass
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
    "calculate_dad_snr_info",
    # Escriptura
    "write_consolidated_excel",
    # Funcions principals
    "analyze_sample",
    "analyze_sequence",
    # Funcions unificades (v1.4.0)
    "analyze_signal",
    "analyze_signal_comparison",
    # Comparació rèpliques (v1.5.0)
    "compare_replicas",
    "recommend_replica",
    "quantify_sample",
    # Constants
    "REPLICA_PEARSON_THRESHOLD",
    "REPLICA_AREA_DIFF_THRESHOLD",
    "REPLICA_FRAC_DIFF_THRESHOLD",
]
