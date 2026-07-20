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
# v1.3.0: Afegides mètriques DUAL/COLUMN: irregular_top_uib, pearson_direct_uib,
#         area_diff_pct, sb_hs_ratio, doc_254_ratio, n_peaks_254_HS
# v1.2.0: Afegit càlcul FWHM i simetria del pic (usat calculate_fwhm de hpsec_core)

import os
import json
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
from scipy.signal import find_peaks
from scipy.integrate import trapezoid
from scipy.stats import pearsonr

# Import funcions de detecció des de hpsec_core (Single Source of Truth)
from hpsec_core import (
    format_timeout_status,
    TIMEOUT_CONFIG,
    calc_snr,
    area_to_ppm,
    detect_irregular_top,
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
    apply_smoothing,
)
from hpsec_config import get_config

# Import funcions calibració global (rf_mass_cal + intercept)
from hpsec_calibrate import (
    get_all_active_calibrations,
    get_rf_mass_cal,
    get_calibration_intercept,
)

# Import sistema d'avisos estructurats
from hpsec_warnings import (
    get_max_warning_level, WarningLevel,
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


# La correcció de baseline es fa a la importació; el processament requereix y_net.

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
# find_peak_boundaries i detect_main_peak s'importen de hpsec_core.py


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
    # Carregar fraccions del config si no venen al param
    subzones = {}  # {subzone_name: (parent, start, end)}
    if not fractions:
        try:
            from hpsec_config import get_config
            cfg = get_config()
            for fname, finfo in cfg.get_all_fractions():
                fractions[fname] = [finfo["start"], finfo["end"]]
            subzones = cfg.get_all_subzones()
        except Exception as e:
            logger.warning("Could not load time fractions from config: %s", e)
    else:
        # fractions ja injectat — provar de llegir subzones del config
        # (no està al param 'fractions' simple, només per compatibilitat)
        try:
            from hpsec_config import get_config
            subzones = get_config().get_all_subzones()
        except Exception:
            subzones = {}

    # Calcular àrea per cada fracció principal
    kpis = {}
    for nom, lim in fractions.items():
        t_ini, t_fi = lim[0], lim[1]
        mask = (t >= t_ini) & (t < t_fi)
        if np.sum(mask) > 1:
            kpis[nom] = float(trapezoid(y_clean[mask], t[mask]))
        else:
            kpis[nom] = 0.0

    # Calcular àrea per cada sub-zona (si n'hi ha)
    for sub_nom, (parent_nom, s_ini, s_fi) in subzones.items():
        if s_ini is None or s_fi is None:
            continue
        mask = (t >= s_ini) & (t < s_fi)
        if np.sum(mask) > 1:
            kpis[sub_nom] = float(trapezoid(y_clean[mask], t[mask]))
        else:
            kpis[sub_nom] = 0.0

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
        # Sub-zones: percentatge respecte el parent (no respecte total)
        for sub_nom, (parent_nom, _s, _e) in subzones.items():
            parent_area = kpis.get(parent_nom, 0.0)
            if parent_area > 0:
                kpis[f"{sub_nom}_pct"] = 100.0 * kpis[sub_nom] / parent_area
            else:
                kpis[f"{sub_nom}_pct"] = 0.0
    else:
        for nom in fractions.keys():
            kpis[f"{nom}_pct"] = 0.0
        for sub_nom in subzones.keys():
            kpis[f"{sub_nom}_pct"] = 0.0

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
        except (ValueError, TypeError) as e:
            logger.debug("DOC Pearson correlation failed: %s", e)

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
                        except (ValueError, TypeError) as e:
                            logger.debug("Pearson for wavelength %s failed: %s", wl, e)

                # Guardar resultats
                result["dad"]["pearson_per_wavelength"] = pearson_per_wl

                # Trobar mínim i la seva λ
                if pearson_per_wl:
                    min_wl = min(pearson_per_wl, key=pearson_per_wl.get)
                    result["dad"]["pearson_min"] = pearson_per_wl[min_wl]
                    result["dad"]["wavelength_min"] = min_wl

                    # Warning si mínim és baix (indicant la λ afectada)
                    if pearson_per_wl[min_wl] < REPLICA_PEARSON_THRESHOLD:
                        result["dad"]["warnings"].append(create_anomaly("LOW_CORRELATION_DAD",
                            details={"wavelength": min_wl, "pearson": pearson_per_wl[min_wl],
                                     "threshold": REPLICA_PEARSON_THRESHOLD},
                            override_label=f"Correlació baixa A{min_wl} (r={pearson_per_wl[min_wl]:.3f})"))

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
                        result["dad"]["warnings"].append(create_anomaly("AREA_DIFF_HIGH_DAD",
                            details={"wavelength": "254", "diff_pct": diff_254,
                                     "threshold": REPLICA_AREA_DIFF_THRESHOLD},
                            override_label=f"Diferència àrea alta A254 ({diff_254:.0f}%)"))

            except Exception as e:
                logger.warning("DAD replica comparison failed: %s", e)

    return result


def recommend_replica(r1_result, r2_result=None, comparison=None, mode="COLUMN"):
    """
    Recomana la millor rèplica per DOC i DAD independentment.
    Suporta 2 rèpliques (signatura clàssica) o N rèpliques.

    Args (clàssic, 2 rèpliques):
        r1_result: Resultat de analyze_sample() per rèplica 1
        r2_result: Resultat de analyze_sample() per rèplica 2
        comparison: Resultat de compare_replicas()
        mode: "COLUMN" o "BP"

    Args (N rèpliques):
        r1_result: dict {rep_key: analyze_result, ...} amb totes les rèpliques
        r2_result: None (no usat)
        comparison: dict {(ki,kj): compare_result, ...} pairwise comparisons
        mode: "COLUMN" o "BP"

    Returns:
        dict amb:
            - doc: { replica: str, score: float, reason: str }
            - dad: { replica: str, score: float, reason: str }
    """
    # Detectar mode: N-rèpliques vs 2-rèpliques
    if r2_result is None and isinstance(r1_result, dict):
        # Mode N: r1_result és dict de rèpliques
        return _recommend_replica_multi(r1_result, comparison or {}, mode)

    # Mode clàssic: 2 rèpliques → delegar a _recommend_replica_multi
    replicas_dict = {"1": r1_result, "2": r2_result}
    pairwise = {("1", "2"): comparison} if comparison else {}
    return _recommend_replica_multi(replicas_dict, pairwise, mode)


# Anomalies EXCLUSIVES del senyal UIB: no han d'invalidar la quantificació DOC
# Direct (que es mesura independentment). Excepció: mostres només-UIB, on el
# Direct s'estima a partir de l'UIB i per tant sí que en depèn.
UIB_ONLY_CODES = {"UIB_SATURATED", "IRREGULAR_TOP_UIB", "UIB_NO_BASELINE"}


def _score_replica_doc(rep_key, rep_result):
    """Puntua una rèplica per DOC: anomalies + SNR. Retorna (score, reason)."""
    anomalies = rep_result.get("anomalies", [])
    codes = get_anomaly_codes(anomalies)
    # El DOC Direct no s'invalida per problemes només-UIB (senyal independent)
    if not rep_result.get("is_uib_only"):
        codes = codes - UIB_ONLY_CODES
    snr = rep_result.get("snr_info", {}).get("snr_direct", 0) or 0

    irreparable_codes = {c for c, e in ANOMALY_CATALOG.items()
                         if e.get("invalidates") and not e.get("repairable")}
    repairable_codes = {c for c, e in ANOMALY_CATALOG.items()
                        if e.get("severity") == WarningLevel.BLOCKER and e.get("repairable")}

    has_critical = bool(codes & CRITICAL_ANOMALIES)
    has_irreparable = bool(codes & irreparable_codes)
    has_repairable = bool(codes & repairable_codes)

    if has_irreparable:
        return (0.1 + snr * 0.0001, "anomalies no reparables", False, False)
    elif has_critical and has_repairable:
        return (0.4 + snr * 0.0001, "cim irregular (reparable)", True, True)
    elif has_critical:
        return (0.2 + snr * 0.0001, "anomalies crítiques", False, False)
    else:
        # Sense anomalies crítiques — puntuar per SNR
        return (0.75 + min(snr / 1000, 0.20), "OK", True, False)


def _recommend_replica_multi(replicas_dict, pairwise_comparisons, mode="COLUMN"):
    """
    Recomana la millor rèplica entre N candidats per DOC i DAD.

    Args:
        replicas_dict: {rep_key: analyze_result}
        pairwise_comparisons: {(ki,kj): compare_result}
        mode: "COLUMN" o "BP"
    """
    keys = sorted(replicas_dict.keys())
    if not keys:
        return {
            "doc": {"replica": "1", "score": 0, "reason": "Sense rèpliques"},
            "dad": {"replica": "1", "score": 0, "reason": "Sense rèpliques"},
        }
    if len(keys) == 1:
        return {
            "doc": {"replica": keys[0], "score": 0.5, "reason": "Rèplica única"},
            "dad": {"replica": keys[0], "score": 0.5, "reason": "Rèplica única"},
        }

    # === DOC ===
    doc_scores = {}
    all_valid = True
    repairable_keys = []
    for k in keys:
        score, reason, valid, repairable = _score_replica_doc(k, replicas_dict[k])
        doc_scores[k] = {"score": score, "reason": reason, "valid": valid, "repairable": repairable}
        if not valid:
            all_valid = False
        if repairable:
            repairable_keys.append(k)

    best_doc_key = max(keys, key=lambda k: doc_scores[k]["score"])
    best_doc = doc_scores[best_doc_key]
    doc_result = {
        "replica": best_doc_key,
        "score": best_doc["score"],
        "reason": best_doc["reason"],
    }
    # valid=False si TOTES les rèpliques tenen anomalies no reparables
    any_valid = any(doc_scores[k]["valid"] for k in keys)
    if not any_valid:
        doc_result["valid"] = False
        doc_result["reason"] = "Totes les rèpliques amb anomalies no reparables"
    if repairable_keys:
        doc_result["repairable"] = True
        doc_result["repairable_replicas"] = repairable_keys

    # === DAD ===
    dad_scores = {}
    for k in keys:
        snr_dad = replicas_dict[k].get("snr_info_dad", {}).get("A254", {}).get("snr", 0) or 0
        dad_scores[k] = snr_dad

    best_dad_key = max(keys, key=lambda k: dad_scores[k])
    best_dad_snr = dad_scores[best_dad_key]
    # Preferir R1 si SNR similar (dins 10%)
    first_key = keys[0]
    if best_dad_key != first_key:
        if dad_scores[first_key] >= best_dad_snr * 0.9:
            best_dad_key = first_key  # preferència primera rèplica si similar

    dad_result = {
        "replica": best_dad_key,
        "score": 0.85 if best_dad_snr > 0 else 0.5,
        "reason": "SNR 254nm superior" if best_dad_key != first_key else "SNR similar, preferència R1",
    }

    return {"doc": doc_result, "dad": dad_result}


def repair_irregular_top_in_replica(sample_result, signal="direct", factor=None,
                                    anchor_left_t=None, anchor_right_t=None):
    """
    Repara cim irregular (jagged/batman) en una rèplica usant repair_with_parabola().

    Modifica in-place el sample_result: actualitza y_doc_net (o y_doc_uib_net),
    recalcula àrees, i guarda traçabilitat de la reparació.

    Args:
        sample_result: Dict retornat per analyze_sample()
        signal: "direct" o "uib"
        factor: correction factor per l'altura teòrica (None = default REPAIR_FACTOR)

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
        anom_key = "IRREGULAR_TOP_DIRECT"
        irr_key = "irregular_top_direct"
        areas_key = "areas"
    else:
        y_key = "y_doc_uib_net"
        anom_key = "IRREGULAR_TOP_UIB"
        irr_key = "irregular_top_uib"
        areas_key = "areas_uib"

    y_original = np.asarray(sample_result.get(y_key, []))
    if len(y_original) == 0:
        return {"repaired": False, "reason": f"No {signal} data"}

    # Guardar dades originals per traçabilitat
    repair_result["original_y"] = y_original.copy().tolist()
    repair_result["original_areas"] = sample_result.get(areas_key, {}).copy()

    # Extreure segment al voltant del pic (com detect_main_peak)
    # find_tangents_and_anchors està dissenyat per segments, no cromatogrames complets
    is_bp = sample_result.get("is_bp", False)
    peak_idx = int(np.argmax(y_original))
    t_peak = float(t[peak_idx])
    half_window = 3.0 if is_bp else 5.0
    seg_mask = (t >= t_peak - half_window) & (t <= t_peak + half_window)
    t_seg = t[seg_mask]
    y_seg = y_original[seg_mask]

    # Aplicar reparació sobre segment (force=True: l'anomalia ja ha estat detectada)
    repair_kwargs = {"force": True}
    if factor is not None:
        repair_kwargs["factor"] = factor
    if anchor_left_t is not None:
        repair_kwargs["anchor_left_t"] = anchor_left_t
    if anchor_right_t is not None:
        repair_kwargs["anchor_right_t"] = anchor_right_t
    y_seg_repaired, repair_info, was_repaired = repair_with_parabola(
        t_seg, y_seg, **repair_kwargs
    )

    if not was_repaired:
        return {"repaired": False, "reason": "repair_with_parabola failed",
                "repair_info": repair_info}

    # Mapejar reparació del segment al cromatograma complet
    y_repaired = y_original.copy()
    y_repaired[seg_mask] = y_seg_repaired

    # Actualitzar senyal reparat
    sample_result[y_key] = y_repaired.tolist()
    sample_result[f"{y_key}_original"] = y_original.tolist()  # Backup

    # Marcar anomalia cim irregular com a reparada
    anomalies = sample_result.get("anomalies", [])
    if not mark_repaired(anomalies, anom_key, repair_info=repair_info):
        # Fallback per strings antics (backward compat)
        _marked = False
        for old_key in [anom_key, "BATMAN_DIRECT" if signal == "direct" else "BATMAN_UIB"]:
            if old_key in anomalies:
                anomalies.remove(old_key)
                anomalies.append(f"{anom_key}_REPAIRED")
                _marked = True
                break
        if not _marked:
            # Reparació forçada sense anomalia prèvia (p.ex. des de calibració,
            # on l'adaptador arriba amb anomalies=[]): registrar l'entrada
            # perquè l'estat 'repaired' quedi reflectit i sigui consultable.
            anomalies.append({"code": anom_key, "repaired": True,
                              "repair_info": repair_info})
    sample_result["anomalies"] = anomalies
    sample_result[irr_key] = False
    sample_result[f"{irr_key}_repaired"] = True
    sample_result[f"{irr_key}_repair_info"] = {
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


# Backwards compatibility alias
repair_batman_in_replica = repair_irregular_top_in_replica


def undo_repair_in_replica(sample_result, signal="direct"):
    """
    Desfà la reparació de cim irregular: restaura y_doc_net_original i recalcula fraccions.

    Args:
        sample_result: Dict retornat per analyze_sample() (modificat in-place)
        signal: "direct" o "uib"

    Returns:
        dict amb info de l'undo: {"undone": True/False, "reason": ...}
    """
    from hpsec_warnings import mark_repaired

    if signal == "direct":
        y_key = "y_doc_net"
        orig_key = "y_doc_net_original"
        anom_key = "IRREGULAR_TOP_DIRECT"
        irr_key = "irregular_top_direct"
        areas_key = "areas"
    else:
        y_key = "y_doc_uib_net"
        orig_key = "y_doc_uib_net_original"
        anom_key = "IRREGULAR_TOP_UIB"
        irr_key = "irregular_top_uib"
        areas_key = "areas_uib"

    y_original = sample_result.get(orig_key)
    if y_original is None:
        return {"undone": False, "reason": "No hi ha backup de senyal original"}

    t = np.asarray(sample_result.get("t_doc", []))
    y_orig = np.asarray(y_original)

    # Restaurar senyal original
    sample_result[y_key] = y_orig.tolist()
    sample_result.pop(orig_key, None)

    # Desmarcar anomalia com a reparada
    anomalies = sample_result.get("anomalies", [])
    for a in anomalies:
        if isinstance(a, dict) and a.get("code") == anom_key and a.get("repaired"):
            a.pop("repaired", None)
            a.pop("repair_info", None)
            break

    # Restaurar flags
    sample_result[irr_key] = True
    sample_result.pop(f"{irr_key}_repaired", None)
    sample_result.pop(f"{irr_key}_repair_info", None)

    # Recalcular fraccions amb senyal original
    try:
        new_areas = calcular_fraccions_temps(t, y_orig)
        if signal == "direct":
            if "areas" not in sample_result:
                sample_result["areas"] = {}
            sample_result["areas"]["DOC"] = new_areas
        else:
            sample_result["areas_uib"] = new_areas
    except Exception as e:
        return {"undone": True, "areas_recalc_error": str(e)}

    return {"undone": True, "signal": signal}


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
        "area_total": None,        # v2.2.0+: àrea total Direct (per traçabilitat)
        "area_total_uib": None,
        "fractions": {},
        "fractions_uib": {},
        "calibration_source": None,
        "rf_mass_cal_used": None,  # RF Direct
        "rf_mass_cal_uib_used": None,  # v2.2.0+: RF UIB
        "intercept": 0,            # Intercept Direct
        "intercept_uib": 0,        # v2.2.0+: Intercept UIB
        "volume_uL": None,         # v2.2.0+: volum injecció emprat
        "below_lod": False,
        "below_loq": False,
        "lod_ppm": None,
        "loq_ppm": None
    }

    if not sample_result.get("processed"):
        return result

    # =========================================================================
    # OBTENIR VOLUM D'INJECCIÓ
    # =========================================================================
    volume_uL = sample_result.get("inj_volume")
    volume_source = "manifest" if volume_uL is not None else None
    if volume_uL is None and calibration_data:
        volume_uL = calibration_data.get("volume_uL") or calibration_data.get("inj_volume")
        if volume_uL is not None:
            volume_source = "calibration"
    if volume_uL is None:
        # NO assumir en silenci: el volum és divisor directe de la ppm. S'usa un
        # valor per defecte per no bloquejar, però es marca explícitament i s'emet
        # una anomalia perquè la GUI ho avisi (o demani el volum a l'usuari).
        volume_uL = 100 if mode.upper() == "BP" else 400
        volume_source = "assumed"
        logger.warning("quantify_sample: VOLUM NO TROBAT per '%s' mode=%s — assumit %d uL (marcat)",
                       sample_result.get("sample_name", "?"), mode, volume_uL)
        sample_result.setdefault("anomalies", []).append(
            create_anomaly("ANA_VOLUME_ASSUMED",
                           details={"assumed_uL": volume_uL, "mode": mode},
                           override_label=f"Volum assumit {volume_uL} µL (no trobat)"))
    result["volume_uL"] = volume_uL
    result["volume_source"] = volume_source

    # =========================================================================
    # OBTENIR rf_mass_cal GLOBAL I INTERCEPT
    # =========================================================================
    mode_key = mode.lower()  # 'column' o 'bp'

    # Intentar usar calibració global — calibracions independents per senyal
    uib_sensitivity = sample_result.get('uib_sensitivity')
    rf_mass_direct = get_rf_mass_cal(signal='direct', mode=mode_key, seq_date=seq_date)
    rf_mass_uib = get_rf_mass_cal(signal='uib', mode=mode_key, seq_date=seq_date,
                                   sensitivity=uib_sensitivity)

    # Obtenir intercept per signal/mode (0 si origin, ex: BP)
    intercept_direct = get_calibration_intercept(signal='direct', mode=mode_key, seq_date=seq_date)
    intercept_uib = get_calibration_intercept(signal='uib', mode=mode_key, seq_date=seq_date,
                                               sensitivity=uib_sensitivity)

    use_global = rf_mass_direct is not None and rf_mass_direct > 0

    # Fórmula única a hpsec_core.area_to_ppm; wrapper local només per fixar
    # volume_uL i el default d'intercept d'aquest àmbit (Direct).
    def apply_formula(area, rf_mass, intercept=intercept_direct):
        return area_to_ppm(area, rf_mass, volume_uL, intercept=intercept)

    # =========================================================================
    # QUANTIFICACIÓ DOC DIRECT
    # =========================================================================
    # Si is_uib_only + direct_estimated_from_uib: areas["DOC"] conté àrees estimades
    # (correctes, escalades per factor sensibilitat) → quantificar normalment.
    # Si is_uib_only SENSE estimació (JSONs antics pre-fix): areas["DOC"] = UIB → saltar.
    is_uib_only = sample_result.get("is_uib_only", False)
    has_direct_estimate = sample_result.get("direct_estimated_from_uib", False)
    areas_direct = sample_result.get("areas", {}).get("DOC", {})
    area_total_direct = areas_direct.get("total", 0)
    if is_uib_only and not has_direct_estimate:
        area_total_direct = 0  # JSON antic: àrees UIB sense corregir

    # Sempre exposar volum + àrea total per traçabilitat (encara que ppm sigui None)
    result["volume_uL"] = volume_uL
    result["area_total"] = float(area_total_direct) if area_total_direct else None

    if area_total_direct > 0:
        if use_global:
            ppm_direct = apply_formula(area_total_direct, rf_mass_direct)
            result["concentration_ppm_direct"] = float(ppm_direct)
            result["concentration_ppm"] = float(ppm_direct)
            result["calibration_source"] = "GLOBAL"
            result["rf_mass_cal_used"] = rf_mass_direct
            result["intercept"] = intercept_direct
        else:
            # Fallback: usar RF local (àrea/ppm) si disponible — SENSE INTERCEPT
            logger.warning("quantify_sample: calibració global no disponible per %s, "
                           "usant RF local (punt únic, sense intercept)",
                           sample_result.get("sample_name", "?"))
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
    # LOD/LOQ in ppm (from area-based LOD/LOQ in snr_info)
    # =========================================================================
    snr_info = sample_result.get("snr_info", {})
    if use_global and rf_mass_direct > 0:
        lod_area = snr_info.get("lod_direct", 0)
        loq_area = snr_info.get("loq_direct", 0)
        # LOD/LOQ: usar formula sense intercept (l'intercept corregeix biaix,
        # no afecta el limit de deteccio que es basat en soroll)
        if lod_area > 0:
            result["lod_ppm"] = float(area_to_ppm(lod_area, rf_mass_direct, volume_uL))
        if loq_area > 0:
            result["loq_ppm"] = float(area_to_ppm(loq_area, rf_mass_direct, volume_uL))
        ppm_d = result.get("concentration_ppm_direct")
        if ppm_d is not None and result["lod_ppm"]:
            result["below_lod"] = ppm_d < result["lod_ppm"]
        if ppm_d is not None and result["loq_ppm"]:
            result["below_loq"] = ppm_d < result["loq_ppm"]

    # =========================================================================
    # QUANTIFICACIÓ DOC UIB (si DUAL i rf_mass_uib disponible)
    # =========================================================================
    areas_uib = sample_result.get("areas_uib", {})
    area_total_uib = areas_uib.get("total", 0)
    if area_total_uib:
        result["area_total_uib"] = float(area_total_uib)

    if area_total_uib > 0:
        if rf_mass_uib and rf_mass_uib > 0:
            # Usar fórmula global amb intercept UIB independent
            ppm_uib = apply_formula(area_total_uib, rf_mass_uib, intercept=intercept_uib)
            result["concentration_ppm_uib"] = float(ppm_uib)
            result["rf_mass_cal_uib_used"] = rf_mass_uib
            result["intercept_uib"] = intercept_uib

            # Concentracions UIB per fracció (només COLUMN)
            if mode.upper() == "COLUMN":
                for frac in ["BioP", "HS", "BB", "SB", "LMW"]:
                    area_frac = areas_uib.get(frac, 0)
                    if area_frac > 0:
                        result["fractions_uib"][frac] = float(apply_formula(area_frac, rf_mass_uib,
                                                                             intercept=intercept_uib))
                    else:
                        result["fractions_uib"][frac] = 0.0
        else:
            # Fallback: usar RF UIB local si disponible — SENSE INTERCEPT
            logger.warning("quantify_sample: rf_mass_uib global no disponible, "
                           "usant RF UIB local per %s",
                           sample_result.get("sample_name", "?"))
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

    # Fallback: si no hi ha Direct (is_uib_only), usar UIB com a concentration_ppm principal
    if result["concentration_ppm"] is None and result["concentration_ppm_uib"] is not None:
        result["concentration_ppm"] = result["concentration_ppm_uib"]

    return result



# ---------------------------------------------------------------------------
# FASES D'analyze_sample
# ---------------------------------------------------------------------------
# analyze_sample() és una orquestració de fases; cada fase viu en un helper
# _asample_* que rep l'estat de senyals `sig` (dict) i anota el `result`.


def _asample_prepare_signals(sample_data, calibration_data, config, result):
    """Fase 1 — senyals: RAW→NET, filtre DAD, shifts, interpolació UIB, baseline.

    Retorna l'estat de senyals (dict) o None si falta informació essencial
    (l'error queda anotat a result).
    """
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
        return None

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
    # EXCEPCIÓ BP: les finestres ja estan alineades per reassign_bp_by_dad254,
    # el shift DOC-DAD no s'ha d'aplicar (ja incorporat a les finestres).
    if calibration_data and not is_bp:
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
            return None

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
            return None

    return {
        "t_doc": t_doc,
        "t_doc_uib": t_doc_uib,
        "y_doc": y_doc,
        "y_doc_uib": y_doc_uib,
        "df_dad": df_dad,
        "is_bp": is_bp,
        "is_dual": is_dual,
        "y_doc_net": y_doc_net,
        "y_doc_direct_net": y_doc_direct_net if is_dual else None,
        "y_doc_uib_net": y_doc_uib_net if is_dual else None,
    }


def _asample_collect_timeouts(sample_data, result, is_dual):
    """Fase 2 — timeouts detectats a import: propaga flags al result."""
    # Timeout info: ve d'import (detect_sequence_timeouts → map_timeouts_to_injection)
    timeout_info = sample_data.get("timeout_info", {})
    timeout_positions = timeout_info.get("t_positions", [])

    # Timeout al límit d'injecció
    if timeout_info.get("timeout_at_boundary"):
        result["timeout_at_boundary"] = True
        result["anomalies"].append(create_anomaly("TIMEOUT_AT_BOUNDARY"))

    # Propagar flag reparació timestamps TOC
    if sample_data.get("toc_minute_precision"):
        timeout_info["toc_minute_precision"] = True

    if timeout_info.get("n_timeouts", 0) > 0 or timeout_info.get("toc_minute_precision") or timeout_info.get("timeout_at_boundary"):
        result["timeout_info"] = timeout_info

    # Timeout UIB: propagat des de Direct a import (single source of truth)
    if is_dual:
        uib_timeout = sample_data.get("timeout_info_uib", {})
        if uib_timeout and uib_timeout.get("n_timeouts", 0) > 0:
            result["timeout_info_uib"] = uib_timeout

    return timeout_info, timeout_positions


def _asample_detect_peak_and_repairs(sig, sample_data, config, result,
                                     seq_name, sample_name):
    """Fase 3 — pic principal, cims irregulars (+auto-repair UIB) i saturació UIB."""
    t_doc = sig["t_doc"]
    t_doc_uib = sig["t_doc_uib"]
    y_doc = sig["y_doc"]
    y_doc_uib = sig["y_doc_uib"]
    y_doc_net = sig["y_doc_net"]
    y_doc_uib_net = sig["y_doc_uib_net"]
    is_bp = sig["is_bp"]
    is_dual = sig["is_dual"]

    # Detectar pic principal
    y_smooth = apply_smoothing(y_doc_net)
    peak_info = detect_main_peak(t_doc, y_smooth, config.get("peak_min_prominence_pct", 5.0), is_bp=is_bp)

    if not peak_info.get("valid"):
        result["anomalies"].append(create_anomaly("NO_PEAK"))
    else:
        # Cim irregular DOC Direct (ja detectat per detect_main_peak pre-repair)
        if peak_info.get("is_irregular_top", False):
            irr_info = peak_info.get("irregular_top_info", {})
            result["anomalies"].append(create_anomaly("IRREGULAR_TOP_DIRECT", details=irr_info))
            result["irregular_top_direct"] = True
            result["irregular_top_direct_info"] = irr_info
            result["irregular_top_direct_repaired"] = peak_info.get("irregular_top_repaired", False)
            _log_detection_issue(seq_name, sample_name, "irregular_top", "direct", irr_info)
        else:
            result["irregular_top_direct"] = False

    # Detectar cim irregular UIB (si DUAL) — auto-repair consistent amb Direct
    if is_dual and y_doc_uib_net is not None and len(y_doc_uib_net) > 0:
        y_uib_smooth = apply_smoothing(y_doc_uib_net)
        irr_uib_result = detect_irregular_top(t_doc, y_uib_smooth)
        if irr_uib_result.get("is_irregular_top"):
            result["anomalies"].append(create_anomaly("IRREGULAR_TOP_UIB", details=irr_uib_result))
            result["irregular_top_uib"] = True
            result["irregular_top_uib_info"] = irr_uib_result
            _log_detection_issue(seq_name, sample_name, "irregular_top", "uib", irr_uib_result)

            # Auto-repair UIB (com Direct a detect_main_peak)
            from hpsec_core import repair_with_parabola
            uib_peak_idx = int(np.argmax(y_uib_smooth))
            t_uib_peak = float(t_doc[uib_peak_idx])
            uib_hw = 3.0 if is_bp else 5.0
            uib_seg_mask = (t_doc >= t_uib_peak - uib_hw) & (t_doc <= t_uib_peak + uib_hw)
            t_uib_seg = t_doc[uib_seg_mask]
            y_uib_seg = y_doc_uib_net[uib_seg_mask]

            if len(y_uib_seg) > 20:
                y_uib_seg_rep, uib_repair_info, uib_was_repaired = repair_with_parabola(
                    t_uib_seg, y_uib_seg
                )
                if uib_was_repaired:
                    result["y_doc_uib_net_original"] = y_doc_uib_net.tolist()
                    y_doc_uib_net = y_doc_uib_net.copy()
                    y_doc_uib_net[uib_seg_mask] = y_uib_seg_rep
                    result["irregular_top_uib_repaired"] = True
                    result["irregular_top_uib_repair_info"] = {
                        "t_max": uib_repair_info.get("t_max"),
                        "y_max_original": uib_repair_info.get("y_max_original"),
                        "y_max_theoretical": uib_repair_info.get("y_max_theoretical"),
                    }
                    logger.info(f"{seq_name}/{sample_name}: UIB irregular_top auto-repaired")
                else:
                    result["irregular_top_uib_repaired"] = False
            else:
                result["irregular_top_uib_repaired"] = False
        else:
            result["irregular_top_uib"] = False

    # Detectar saturació UIB per forma del pic (Gaussian clipping)
    # Un pic saturat té el cim retallat: y_max << amplitud Gaussiana predita pels flancs.
    # No depèn de cap paràmetre de sensibilitat — detecta per la forma intrínseca.
    y_uib_for_sat = y_doc_uib if is_dual else (y_doc if sample_data.get("is_uib_only") else None)
    t_uib_for_sat = t_doc_uib if is_dual else (t_doc if sample_data.get("is_uib_only") else None)
    if y_uib_for_sat is not None and len(y_uib_for_sat) > 0:
        from hpsec_core import detect_peak_clipping
        clip_info = detect_peak_clipping(t_uib_for_sat, y_uib_for_sat)
        if clip_info["is_saturated"]:
            sat_details = {
                "y_max": clip_info["y_max_observed"],
                "plateau_ratio": clip_info["plateau_ratio"],
                "plateau_width_pts": clip_info["plateau_width_pts"],
                "fwhm_pts": clip_info["fwhm_pts"],
            }
            result["anomalies"].append(create_anomaly("UIB_SATURATED", details=sat_details))
            result["uib_saturated"] = True
            logger.warning(f"{seq_name}/{sample_name}: UIB SATURATED "
                          f"plateau_ratio={clip_info['plateau_ratio']:.3f}, "
                          f"plateau={clip_info['plateau_width_pts']} pts, "
                          f"y_max={clip_info['y_max_observed']:.1f}")
        else:
            result["uib_saturated"] = False
    else:
        result["uib_saturated"] = False

    result["peak_info"] = peak_info

    sig["y_doc_uib_net"] = y_doc_uib_net
    return peak_info, y_smooth


def _asample_flag_timeout_in_peak(sig, config, result, timeout_info,
                                  timeout_positions, peak_info):
    """Fase 4 — anomalies TIMEOUT_IN_PEAK (Direct i UIB)."""
    t_doc = sig["t_doc"]
    y_doc_uib_net = sig["y_doc_uib_net"]
    is_bp = sig["is_bp"]
    is_dual = sig["is_dual"]

    # Check TIMEOUT_IN_PEAK: timeout que afecta el pic principal DOC Direct
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

    # Check TIMEOUT_IN_PEAK per UIB: timeout estimat que afecta el pic UIB
    uib_timeout = result.get("timeout_info_uib")
    if uib_timeout and uib_timeout.get("n_timeouts", 0) > 0:
        # Usar peak_uib si ja s'ha calculat, sinó detectar-lo
        uib_peak_for_timeout = None
        if is_dual and y_doc_uib_net is not None and len(y_doc_uib_net) > 0:
            y_uib_sm = apply_smoothing(y_doc_uib_net)
            uib_peak_for_timeout = detect_main_peak(
                t_doc, y_uib_sm, config.get("peak_min_prominence_pct", 5.0), is_bp=is_bp)

        if uib_peak_for_timeout and uib_peak_for_timeout.get("valid"):
            t_start_uib = uib_peak_for_timeout.get("t_start", t_doc[uib_peak_for_timeout.get("left_idx", 0)])
            t_end_uib = uib_peak_for_timeout.get("t_end", t_doc[uib_peak_for_timeout.get("right_idx", len(t_doc) - 1)])
            # UIB timeout usa affected_start/affected_end (no gaps sinó zones pertorbades)
            uib_to_details = uib_timeout.get("timeouts", [])
            timeout_in_uib_peak = any(
                (t_start_uib <= to.get("affected_start_min", to.get("t_start_min", 0)) <= t_end_uib) or
                (t_start_uib <= to.get("affected_end_min", to.get("t_end_min", 0)) <= t_end_uib) or
                (to.get("affected_start_min", to.get("t_start_min", 0)) <= t_start_uib and
                 to.get("affected_end_min", to.get("t_end_min", 0)) >= t_end_uib)
                for to in uib_to_details
            )
            if timeout_in_uib_peak:
                result["anomalies"].append(create_anomaly("TIMEOUT_IN_PEAK", details={
                    "timeout_info": uib_timeout, "signal": "uib"
                }))
                result["timeout_in_peak_uib"] = True


def _asample_peak_metrics(sig, config, result, peak_info, y_smooth):
    """Fase 5 — FWHM, simetria i ajust bigaussià (DOC, UIB i DAD 254 a BP)."""
    t_doc = sig["t_doc"]
    y_doc_net = sig["y_doc_net"]
    y_doc_uib_net = sig["y_doc_uib_net"]
    df_dad = sig["df_dad"]
    is_bp = sig["is_bp"]
    is_dual = sig["is_dual"]

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
            except Exception as e:
                logger.debug("Bi-Gaussian DOC fit failed: %s", e)
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
                        except Exception as e:
                            logger.debug("Bi-Gaussian 254nm fit failed: %s", e)
            except Exception as e:
                logger.debug("DAD 254nm peak analysis failed: %s", e)


def _asample_areas_and_snr(sig, config, result, peak_info, timeout_positions):
    """Fase 6 — àrees per fraccions, tmax per senyal i SNR/LOD/LOQ."""
    t_doc = sig["t_doc"]
    y_doc_net = sig["y_doc_net"]
    y_doc_uib_net = sig["y_doc_uib_net"]
    df_dad = sig["df_dad"]
    is_bp = sig["is_bp"]
    is_dual = sig["is_dual"]

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

    return areas


def _asample_areas_uib(sig, sample_data, config, result, areas):
    """Fase 7 — àrees UIB (mode DUAL o dades només-UIB)."""
    t_doc = sig["t_doc"]
    y_doc_uib_net = sig["y_doc_uib_net"]
    is_dual = sig["is_dual"]

    # =========================================================================
    # ÀREES UIB (per DUAL o quan només hi ha UIB)
    # =========================================================================
    # A08: Calcular areas_uib si:
    #   - Mode DUAL amb y_doc_uib_net disponible
    #   - O mode simple però dades venen d'UIB (is_uib_only)
    is_uib_only = sample_data.get("is_uib_only", False)
    result["is_uib_only"] = is_uib_only

    if is_dual and "DOC" in areas and y_doc_uib_net is not None:
        areas_uib = calcular_fraccions_temps(t_doc, y_doc_uib_net, config)
        result["areas_uib"] = areas_uib
    elif is_uib_only and "DOC" in areas:
        # Només UIB: les àrees DOC ja són d'UIB, copiar a areas_uib
        result["areas_uib"] = areas.get("DOC", {}).copy()


def _asample_column_metrics(sig, config, result, areas):
    """Fase 8 — mètriques només-COLUMN: Pearson Direct↔UIB, ratios i recompte de pics."""
    t_doc = sig["t_doc"]
    y_doc_net = sig["y_doc_net"]
    y_doc_direct_net = sig["y_doc_direct_net"]
    y_doc_uib_net = sig["y_doc_uib_net"]
    df_dad = sig["df_dad"]
    is_bp = sig["is_bp"]
    is_dual = sig["is_dual"]

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
            except Exception as e:
                logger.debug("Peak count in timeout zone failed for %s: %s", signal_key, e)
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


def _asample_hci_and_pack(sig, sample_data, result):
    """Fase 9 — HCI (només COLUMN) i empaquetat de senyals al result."""
    t_doc = sig["t_doc"]
    y_doc_net = sig["y_doc_net"]
    y_doc_direct_net = sig["y_doc_direct_net"]
    y_doc_uib_net = sig["y_doc_uib_net"]
    df_dad = sig["df_dad"]
    is_bp = sig["is_bp"]
    is_dual = sig["is_dual"]

    # =========================================================================
    # HCI (Humic Character Index) — només COLUMN, mai BP/BLANK/CONTROL
    # =========================================================================
    dad_export3d_path = sample_data.get("dad_export3d_path")
    if dad_export3d_path and not is_bp:
        try:
            from hpsec_humic import compute_hci
            hci_result = compute_hci(dad_export3d_path)
            if hci_result:
                result["hci"] = hci_result["hci"]
                result["hci_character"] = hci_result["character"]
        except Exception as e:
            logger.debug("HCI computation skipped: %s", e)

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


def _asample_estimate_direct_from_uib(sig, sample_data, config, result):
    """Fase 10 — dades només-UIB: estimar el Direct des del net UIB (factor sens/1000)."""
    t_doc = sig["t_doc"]
    y_doc_net = sig["y_doc_net"]
    is_uib_only = result.get("is_uib_only", False)

    # =========================================================================
    # ESTIMACIÓ DIRECT DES D'UIB (is_uib_only)
    # =========================================================================
    # Quan no hi ha Direct, estimar el cromatograma Direct a partir del net UIB.
    # Factor = sensibilitat_UIB / 1000 (guany instrumental: UIB amplifica per 1000/sens).
    # Apliquem sobre y_doc_net (que conté UIB net) per generar y_doc_direct_net.
    # Les àrees Direct (areas["DOC"]) es recalculen sobre l'estimat.
    if is_uib_only:
        # NO assumir 700 en silenci: el factor d'estimació és sens/1000, així que
        # un valor erroni escala malament tot el Direct estimat. Comprovar None
        # explícitament (no 'or', que també capturaria 0) i marcar si falta.
        sens = sample_data.get("uib_sensitivity")
        if sens not in (700, 1000):
            result.setdefault("anomalies", []).append(
                create_anomaly("ANA_SENSITIVITY_ASSUMED",
                               details={"found": sens, "assumed": 700},
                               override_label=f"Sensibilitat UIB no vàlida ({sens}) — assumit 700 ppb"))
            sens = 700
        direct_factor = sens / 1000.0
        y_direct_est = y_doc_net * direct_factor
        result["y_doc_direct_net"] = y_direct_est
        result["y_doc_uib_net"] = y_doc_net  # Preservar UIB original
        result["direct_estimated_from_uib"] = True
        result["direct_estimation_factor"] = direct_factor
        # Recalcular àrees Direct sobre l'estimat
        areas_direct_est = calcular_fraccions_temps(t_doc, y_direct_est, config)
        result["areas"]["DOC"] = areas_direct_est
        logger.info("is_uib_only %s: Direct estimat (factor=%.3f, sens=%d ppb), "
                     "area_direct_est=%.0f, area_uib=%.0f",
                     sample_data.get("name", "?"), direct_factor, sens,
                     areas_direct_est.get("total", 0),
                     result.get("areas_uib", {}).get("total", 0))


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
            - irregular_top_direct: Bool si detectat cim irregular a Direct
            - irregular_top_uib: Bool si detectat cim irregular a UIB (només DUAL)
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
        "sample_type": sample_data.get("sample_type", "SAMPLE"),
        "inj_volume": sample_data.get("inj_volume"),
        "injection_index": sample_data.get("injection_index"),
        # Propagar la sensibilitat UIB perquè arribi a quantify_sample (routing
        # UIB 700/1000). Abans es perdia aquí i quantify la rebia None.
        "uib_sensitivity": sample_data.get("uib_sensitivity"),
        "processed": False,
        "anomalies": [],
    }
    # Traçabilitat sibling (packs)
    if sample_data.get("source_seq"):
        result["source_seq"] = sample_data["source_seq"]
    if sample_data.get("original_rep_num"):
        result["original_rep_num"] = sample_data["original_rep_num"]


    # Pipeline per fases (helpers _asample_*; el comportament és el mateix,
    # però cada fase és una funció curta i llegible)
    sig = _asample_prepare_signals(sample_data, calibration_data, config, result)
    if sig is None:
        return result

    timeout_info, timeout_positions = _asample_collect_timeouts(
        sample_data, result, sig["is_dual"])
    peak_info, y_smooth = _asample_detect_peak_and_repairs(
        sig, sample_data, config, result, seq_name, sample_name)
    _asample_flag_timeout_in_peak(
        sig, config, result, timeout_info, timeout_positions, peak_info)
    _asample_peak_metrics(sig, config, result, peak_info, y_smooth)
    areas = _asample_areas_and_snr(
        sig, config, result, peak_info, timeout_positions)
    _asample_areas_uib(sig, sample_data, config, result, areas)
    _asample_column_metrics(sig, config, result, areas)
    _asample_hci_and_pack(sig, sample_data, result)
    _asample_estimate_direct_from_uib(sig, sample_data, config, result)

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
        Llista única de flat_samples, cadascun amb sample_type per decidir
        analyze_sample (SAMPLE/BLANK/KHP/PR_*) o _analyze_light_sample (CONTROL).
    """
    all_flat = []

    # Sensibilitat UIB (700 o 1000 ppb) — per detectar saturació
    uib_sensitivity = imported_data.get("uib_sensitivity")

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
                "uib_sensitivity": uib_sensitivity,  # ppb (700/1000) per detecció saturació
            }

            # Traçabilitat sibling (packs): d'on ve cada rèplica
            if rep_data.get("source_seq"):
                flat_sample["source_seq"] = rep_data["source_seq"]
            if rep_data.get("original_rep_num"):
                flat_sample["original_rep_num"] = rep_data["original_rep_num"]

            # Propagar timeout_info complet (font única: import → map_timeouts_to_injection)
            direct = rep_data.get("direct", {})
            if isinstance(direct, dict):
                ti = direct.get("timeout_info", {})
                if isinstance(ti, dict):
                    flat_sample["timeout_info"] = ti
                    if ti.get("toc_minute_precision"):
                        flat_sample["toc_minute_precision"] = True
                    if ti.get("timeout_at_boundary"):
                        flat_sample["timeout_at_boundary"] = True

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
                    # Propagar timeout_info UIB (propagat des de Direct a import)
                    uib_ti = uib.get("timeout_info", {})
                    if isinstance(uib_ti, dict) and uib_ti:
                        flat_sample["timeout_info_uib"] = uib_ti

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
            dad = rep_data.get("dad") or {}
            if dad and "df" in dad:
                df_dad_raw = dad["df"]
                if hasattr(df_dad_raw, 'columns') and len(df_dad_raw.columns) > 8:
                    df_dad_raw = analyze_dad(df_dad_raw)
                flat_sample["df_dad"] = df_dad_raw

            # Propagar path Export3D original per HCI (necessita 101 wavelengths)
            if dad.get("path"):
                flat_sample["dad_export3d_path"] = dad["path"]

            all_flat.append(flat_sample)

    return all_flat


# =============================================================================
# EXCLUSIÓ DE QUANTIFICACIÓ PER TIPOLOGIA DE MOSTRA
# =============================================================================

# Mapping sample_type → config key per consultar el camp "quantify"
_PR_TYPE_TO_CONFIG = {
    "PR_C": "PATRÓ_REF_C",
    "PR_I": "PATRÓ_REF_I",
    "PR_N": "PATRÓ_REF_N",
}


def _should_skip_quantification(sample_name, config=None, sample_type=None):
    """
    Comprova si una mostra s'ha d'excloure de la quantificació ppm.

    Per defecte TOT es quantifica. S'exclou si el sample_type té
    ``quantify: false`` al config (p.ex. PATRÓ_REF_I per inorgànics).

    Args:
        sample_name: Nom de la mostra (per logging)
        config: Configuració (si None, es llegeix de get_config())
        sample_type: Tipologia assignada a import ("PR_C", "PR_I", "PR_N", "SAMPLE", ...)

    Returns:
        True si s'ha de saltar la quantificació
    """
    if sample_type is None:
        return False

    config_key = _PR_TYPE_TO_CONFIG.get(sample_type)
    if not config_key:
        return False  # SAMPLE, BLANK, KHP, etc. → sempre quantificar

    if config is None:
        config = get_config()

    pr_config = config.get("sample_types", config_key, default={})
    # Per defecte quantificar; només skip si explícitament quantify=false
    return pr_config.get("quantify", True) is False


# =============================================================================
# GENERACIÓ D'AVISOS ESTRUCTURATS PER ANÀLISI
# =============================================================================

def _generate_analysis_warnings(result: dict) -> list:
    """
    Genera avisos estructurats a partir del resultat d'anàlisi.

    Tots els avisos usen create_anomaly() (font única: ANOMALY_CATALOG).

    Args:
        result: Dict del resultat de analyze_sequence()

    Returns:
        Llista d'avisos estructurats (dicts ANOMALY_CATALOG)
    """
    warnings = []

    # 1. Errors crítics de seqüència (BLOCKER)
    for error in result.get("errors", []):
        if "calibr" in error.lower():
            warnings.append(create_anomaly("ANA_NO_CALIBRATION"))
        else:
            anomaly = create_anomaly(
                "ANA_NO_CALIBRATION",
                details={"message": error},
            )
            anomaly["message"] = error
            warnings.append(anomaly)

    # 2. Mostres buides
    n_empty = 0
    for sample_name, sample_group in result.get("samples_grouped", {}).items():
        replicas = sample_group.get("replicas", {})
        if not replicas or not any(r.get("processed") for r in replicas.values()):
            n_empty += 1

    if n_empty > 0:
        warnings.append(create_anomaly(
            "ANA_EMPTY_SAMPLES",
            details={"n": n_empty},
        ))

    # 3. Resum compacte d'anomalies per mostra (1 línia, no 22 duplicats)
    # Les anomalies individuals es mostren a la taula d'anàlisi.
    n_with_issues = 0
    for sg in result.get("samples_grouped", {}).values():
        has_issue = False
        for rep in sg.get("replicas", {}).values():
            for a in rep.get("anomalies", []):
                sev = a.get("severity", "info") if isinstance(a, dict) else "info"
                if sev in ("blocker", "warning"):
                    has_issue = True
                    break
            if has_issue:
                break
        if not has_issue:
            comp = sg.get("comparison") or {}
            for domain in ("doc", "dad"):
                for w in comp.get(domain, {}).get("warnings", []):
                    sev = w.get("severity", "info") if isinstance(w, dict) else "info"
                    if sev in ("blocker", "warning"):
                        has_issue = True
                        break
                if has_issue:
                    break
        if has_issue:
            n_with_issues += 1

    if n_with_issues > 0:
        anomaly = create_anomaly("ANA_SAMPLES_WITH_ISSUES",
                                  details={"n": n_with_issues})
        anomaly["label"] = f"{n_with_issues} mostres amb anomalies"
        anomaly["message"] = f"{n_with_issues} mostres amb anomalies (veure taula)"
        warnings.append(anomaly)

    return warnings


# =============================================================================
# ANÀLISI LIGHTWEIGHT PER BLANCS / CONTROLS
# =============================================================================
def _analyze_light_sample(sample):
    """
    Anàlisi lightweight per CONTROL (NaOH, WASH).

    Calcula àrea total DOC, SNR i àrea A254 — sense fraccions ni quantificació.
    NOTA: BLANK (MQ) ara va a anàlisi completa + quantificació.

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
    area_total = float(np.trapezoid(y_doc_net, x=t_doc))

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
                    result["area_254"] = float(np.trapezoid(y_254, x=t_dad))
        except Exception as e:
            logger.debug("Area 254nm calculation failed: %s", e)

    return result


def _estimate_uib_timeouts_from_sequence(result, is_bp=False):
    """
    Estima timeouts UIB per injeccions sense DOC Direct, a partir del patró
    observat a les injeccions que SÍ tenen DOC Direct.

    El timeout del Sievers (recàrrega xeringues) ocorre cada ~77.2 min i afecta
    tant DOC Direct com UIB. DOC Direct el detecta per gap temporal; UIB no,
    però el patró anòmal és idèntic (pic espuri ~1.8 min).

    Estratègia:
    1. Recollir t_positions de totes les injeccions amb DOC Direct timeout
    2. Calcular T0 (primer timeout) i deriva (Δt per injecció)
    3. Extrapolar per a les injeccions sense DOC Direct
    4. Assignar timeout_info_uib estimat
    """
    from hpsec_core import estimate_timeout_for_uib

    all_samples = result.get("samples", [])
    if not all_samples:
        return

    # 1. Recollir patró de timeouts des de DOC Direct
    # Cada sample té un 'line_num' (número d'injecció a la seqüència)
    observed = []  # (inj_num, t_position)
    no_direct = []  # (inj_num, sample_ref)
    is_dual = any(s.get("is_dual") for s in all_samples)

    if not is_dual:
        return  # Sense UIB, no cal estimar

    for sample in all_samples:
        inj_num = (sample.get("injection_index")
                   or sample.get("injection_num")
                   or sample.get("line_num"))
        if inj_num is None:
            continue

        ti = sample.get("timeout_info")
        if ti and ti.get("n_timeouts", 0) > 0 and ti.get("t_positions"):
            for t_pos in ti["t_positions"]:
                observed.append((int(inj_num), t_pos))
        elif sample.get("is_dual") and not sample.get("timeout_info_uib"):
            # Injecció DUAL sense timeout detectat i sense estimació UIB
            no_direct.append((int(inj_num), sample))

    if not observed or not no_direct:
        return  # No cal estimar (totes tenen DOC Direct o cap timeout)

    # 2. Calcular T0 i deriva
    # Ordenar per inj_num
    observed.sort(key=lambda x: x[0])

    if len(observed) >= 2:
        # Calcular T0 absolut (primer timeout a la primera injecció)
        # T_absolut = inj_num * sample_duration + t_position_dins_injecció
        # Però no necessitem sample_duration: podem fer regressió lineal
        # t_position(inj) = T0 + (inj - 1) * drift
        # on drift = sample_duration - 77.2 (negatiu si dur > 77.2)
        import numpy as np
        inj_nums = np.array([x[0] for x in observed])
        t_positions = np.array([x[1] for x in observed])

        # Regressió lineal: t_pos = a + b * inj_num
        # (robust: eliminar outliers amb |residual| > 3 min, típicament salts per blancs)
        coeffs = np.polyfit(inj_nums, t_positions, 1)
        drift_per_inj = coeffs[0]   # pendent (negatiu = timeout es desplaça enrere)
        t0_intercept = coeffs[1]    # t_pos a inj=0

        # Residuals
        predicted = np.polyval(coeffs, inj_nums)
        residuals = t_positions - predicted
        mask_ok = np.abs(residuals) < 3.0  # Tolerar fins a 3 min de desviació

        if mask_ok.sum() >= 2:
            # Recalcular sense outliers
            coeffs = np.polyfit(inj_nums[mask_ok], t_positions[mask_ok], 1)
            drift_per_inj = coeffs[0]
            t0_intercept = coeffs[1]

        logger.info(
            f"Timeout UIB estimation: {len(observed)} observed, "
            f"drift={drift_per_inj:.3f} min/inj, "
            f"t0_intercept={t0_intercept:.1f} min, "
            f"{len(no_direct)} to estimate"
        )

        # 3. Extrapolar per injeccions sense DOC Direct
        for inj_num, sample in no_direct:
            estimated_pos = t0_intercept + drift_per_inj * inj_num

            # Només assignar si la posició és raonable (0 < pos < durada cromatograma)
            max_t = 80.0 if not is_bp else 12.0
            if 0 < estimated_pos < max_t:
                uib_timeout = estimate_timeout_for_uib(
                    direct_timeout_info={
                        "n_timeouts": 1,
                        "t_positions": [estimated_pos],
                    },
                    is_bp=is_bp,
                )
                if uib_timeout.get("n_timeouts", 0) > 0:
                    uib_timeout["source"] = "sequence_extrapolation"
                    uib_timeout["extrapolation_info"] = {
                        "drift_per_inj": round(drift_per_inj, 3),
                        "t0_intercept": round(t0_intercept, 1),
                        "n_observed": len(observed),
                        "inj_num": inj_num,
                    }
                    sample["timeout_info_uib"] = uib_timeout

    elif len(observed) == 1:
        # Només una observació: usar com a referència sense drift
        ref_inj, ref_pos = observed[0]
        for inj_num, sample in no_direct:
            # Aproximar: drift = -1.45 min/inj (teòric per COLUMN 78.65 min)
            typical_drift = -0.20 if is_bp else -1.45
            estimated_pos = ref_pos + typical_drift * (inj_num - ref_inj)

            max_t = 80.0 if not is_bp else 12.0
            if 0 < estimated_pos < max_t:
                uib_timeout = estimate_timeout_for_uib(
                    direct_timeout_info={
                        "n_timeouts": 1,
                        "t_positions": [estimated_pos],
                    },
                    is_bp=is_bp,
                )
                if uib_timeout.get("n_timeouts", 0) > 0:
                    uib_timeout["source"] = "single_ref_extrapolation"
                    sample["timeout_info_uib"] = uib_timeout


# =============================================================================
# FUNCIÓ PRINCIPAL: PROCESSAR SEQÜÈNCIA
# =============================================================================
def analyze_sequence(imported_data, calibration_data=None, config=None,
                     progress_callback=None, do_quantify=True):
    """
    FASE 3: Processa totes les mostres d'una seqüència.

    Args:
        imported_data: Dict retornat per import_sequence() (Fase 1)
        calibration_data: Dict retornat per calibrate_sequence() (Fase 2)
            - shift_uib: Shift per DOC_UIB (minuts)
            - shift_direct: Shift per DOC_Direct (minuts)
        config: Configuració
        progress_callback: Funció callback per reportar progrés
        do_quantify: Si True (default), aplica calibració (àrea → ppm) inline.
            Si False, només qualitatiu (àrees, SNR, R², HCI, anomalies) i la
            quantificació queda pendent. La quantificació es pot aplicar després
            amb quantify_sequence(). [v1.7.0]

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
            - quantification_pending: True si do_quantify=False (Fase 4 separada)
    """
    config = config or DEFAULT_PROCESS_CONFIG

    result = {
        "success": False,
        "seq_name": imported_data.get("seq_name", "UNKNOWN"),
        "seq_path": imported_data.get("seq_path", ""),
        "method": imported_data.get("method", "UNKNOWN"),
        "data_mode": imported_data.get("data_mode", "UNKNOWN"),
        "samples": [],
        "errors": [],
        "warnings": [],
    }

    # Verificar dades d'entrada
    if not imported_data.get("success"):
        result["errors"].append("Imported data is invalid")
        return result

    # Aplanar l'estructura de mostres
    data_mode = imported_data.get("data_mode", "UIB")
    all_flat = _flatten_samples_for_processing(imported_data, data_mode=data_mode)

    if len(all_flat) == 0:
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

    # Mínim de punts DOC per considerar un cromatograma vàlid
    min_doc_points = 60 if mode == "BP" else 450  # BP: 4 min, COLUMN: 30 min (dt=4s)

    # Processar totes les mostres — un sol bucle
    total_samples = len(all_flat)
    for i, sample in enumerate(all_flat):
        if progress_callback:
            progress_callback(f"Processing {sample.get('name', 'sample')}...",
                              (i + 1) / total_samples * 100)

        sample_type = sample.get("sample_type", "SAMPLE")

        # Saltar mostres amb cromatograma truncat
        doc_n_pts = sample.get("n_points", 0) or len(sample.get("t", []))
        if 0 < doc_n_pts < min_doc_points:
            processed = {
                "name": sample.get("name", "UNKNOWN"),
                "replica": sample.get("replica", "1"),
                "sample_type": sample_type,
                "processed": False,
                "sample_valid": False,
                "error": f"Cromatograma truncat ({doc_n_pts} punts, mínim {min_doc_points})",
                "anomalies": [create_anomaly(
                    "IMP_SHORT_CHROMATOGRAM",
                    sample=sample.get("name"),
                    details={"n_points": doc_n_pts, "min_points": min_doc_points},
                )],
            }
            result["samples"].append(processed)
            continue

        try:
            if sample_type == "CONTROL":
                processed = _analyze_light_sample(sample)
            else:
                sample_cal = get_sample_calibration(sample)
                processed = analyze_sample(sample, sample_cal, config)
                if not processed.get("processed"):
                    result["warnings"].append(
                        f"{sample.get('name')}: {processed.get('error', 'Processing failed')}")
            result["samples"].append(processed)
        except Exception as e:
            result["errors"].append(f"{sample.get('name')}: {str(e)}")

    # =========================================================================
    # AGRUPAR RÈPLIQUES, COMPARAR, RECOMANAR I QUANTIFICAR
    # =========================================================================
    if progress_callback:
        progress_callback("Comparing replicas...", 90)

    is_bp = mode == "BP"

    # Agrupar mostres per nom
    samples_by_name = {}
    for sample in result["samples"]:
        name = sample.get("name", "UNKNOWN")
        replica = sample.get("replica", "1")
        if name not in samples_by_name:
            samples_by_name[name] = {}
        samples_by_name[name][replica] = sample

    result["samples_grouped"] = {}

    for sample_name, replicas in samples_by_name.items():
        first_rep = next(iter(replicas.values()), {})
        sample_type = first_rep.get("sample_type", "SAMPLE")
        analysis_type = first_rep.get("analysis_type")  # "light" per CONTROL, "khp" si KHP

        # KHP i CONTROL: group simple sense comparació ni quantificació
        if sample_type == "KHP" or analysis_type == "khp":
            result["samples_grouped"][sample_name] = {
                "analysis_type": "khp",
                "sample_type": "KHP",
                "replicas": replicas,
                "selected": {"doc": sorted(replicas.keys())[0]},
                "sample_valid": True,
            }
            continue

        if sample_type == "CONTROL" or analysis_type == "light":
            result["samples_grouped"][sample_name] = {
                "analysis_type": "light",
                "sample_type": sample_type,
                "replicas": replicas,
                "selected": {"doc": sorted(replicas.keys())[0]},
                "sample_valid": True,
            }
            continue

        # Resta (SAMPLE, BLANK, PR_*): grouping complet
        skip_quant = _should_skip_quantification(sample_name, config, sample_type=sample_type)

        sample_group = {
            "replicas": replicas,
            "sample_type": sample_type,
            "skip_quantification": skip_quant,
            "comparison": None,
            "recommendation": None,
            "selected": {"doc": "1", "dad": "1"},
            "quantification": None
        }

        replica_keys = sorted(replicas.keys())

        if len(replica_keys) >= 2:
            # Comparacions pairwise (totes les parelles)
            from itertools import combinations
            valid_replicas = {k: replicas[k] for k in replica_keys if replicas.get(k)}

            pairwise_comparisons = {}
            for ki, kj in combinations(sorted(valid_replicas.keys()), 2):
                comp = compare_replicas(valid_replicas[ki], valid_replicas[kj],
                                        mode=mode, config=config)
                pairwise_comparisons[(ki, kj)] = comp

            # Backward compat: "comparison" = primera parella
            first_pair = (replica_keys[0], replica_keys[1])
            sample_group["comparison"] = pairwise_comparisons.get(first_pair)
            if len(pairwise_comparisons) > 1:
                sample_group["pairwise_comparisons"] = pairwise_comparisons

            # Recomanació (N rèpliques)
            recommendation = recommend_replica(
                valid_replicas, None, pairwise_comparisons, mode=mode)
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

            # Cross-replica validation per irregular_top:
            # Si és real, TOTES les rèpliques DOC Direct l'han de presentar.
            # Si només algunes el tenen → fals positiu → descartar detecció.
            _irr_codes = {"IRREGULAR_TOP", "IRREGULAR_TOP_DIRECT"}
            _reps_with_irr = []
            _reps_without_irr = []
            for rk in replica_keys:
                rep = replicas.get(rk, {})
                rep_codes = get_anomaly_codes(rep.get("anomalies", []))
                if rep_codes & _irr_codes:
                    _reps_with_irr.append(rk)
                else:
                    _reps_without_irr.append(rk)
            if _reps_with_irr and _reps_without_irr and len(replica_keys) >= 2:
                # Fals positiu: no totes les rèpliques el presenten
                logger.info(
                    "Cross-replica validation %s: irregular_top en %s però no en %s → fals positiu",
                    sample_name, _reps_with_irr, _reps_without_irr)
                for rk in _reps_with_irr:
                    rep = replicas[rk]
                    # Reclassificar anomalies: BLOCKER → INFO (fals positiu)
                    new_anomalies = []
                    for a in rep.get("anomalies", []):
                        code = a.get("code", "") if isinstance(a, dict) else str(a)
                        if code in _irr_codes:
                            if isinstance(a, dict):
                                a = dict(a)
                                a["severity"] = "INFO"
                                a["false_positive"] = True
                                a["label"] = (a.get("label", "") or code) + " (FP)"
                            new_anomalies.append(a)
                        else:
                            new_anomalies.append(a)
                    rep["anomalies"] = new_anomalies
                    rep["is_irregular_top"] = False
                    rep["irregular_top_false_positive"] = True
                # Recalcular repairable (ja no cal)
                sample_group["repairable"] = False
                sample_group["repairable_replicas"] = []
                # Recalcular sample_valid (pot tornar a ser True)
                score, reason, valid, _ = _score_replica_doc(
                    replica_keys[0], replicas[replica_keys[0]])
                any_valid = any(
                    _score_replica_doc(k, replicas[k])[2] for k in replica_keys)
                sample_group["sample_valid"] = any_valid

            # Quantificació (saltar si mostra no vàlida, exclosa o do_quantify=False)
            if not do_quantify:
                sample_group["quantification"] = None
            elif sample_group["sample_valid"] is False:
                sample_group["quantification"] = {
                    "concentration_ppm": None,
                    "concentration_ppm_direct": None,
                    "concentration_ppm_uib": None,
                    "area_total": None,
                    "valid": False,
                    "reason": doc_rec.get("reason", "Mostra no vàlida")
                }
            elif skip_quant:
                sample_group["quantification"] = {
                    "concentration_ppm": None,
                    "concentration_ppm_direct": None,
                    "concentration_ppm_uib": None,
                    "valid": False,
                    "reason": "Patró de referència (sense quantificació)"
                }
            else:
                selected_replica = sample_group["selected"]["doc"]
                r1 = replicas.get(replica_keys[0])
                selected_sample = replicas.get(selected_replica, r1)
                # Usar calibració específica segons volum d'injecció
                sample_cal = get_sample_calibration(selected_sample)
                quantification = quantify_sample(selected_sample, sample_cal, mode=mode)
                # HCI del doc_replica seleccionat
                hci = selected_sample.get("hci")
                if hci is not None:
                    quantification["hci"] = hci
                    quantification["hci_character"] = selected_sample.get("hci_character", "")
                sample_group["quantification"] = quantification

        elif len(replica_keys) == 1:
            # Només una rèplica
            r1 = replicas.get(replica_keys[0])
            sample_group["selected"] = {"doc": replica_keys[0], "dad": replica_keys[0]}
            sample_group["sample_valid"] = True
            sample_group["repairable"] = False
            sample_group["repairable_replicas"] = []
            sample_group["repaired"] = False

            # Quantificació (saltar si exclosa o do_quantify=False)
            if r1:
                if not do_quantify:
                    sample_group["quantification"] = None
                elif skip_quant:
                    sample_group["quantification"] = {
                        "concentration_ppm": None,
                        "concentration_ppm_direct": None,
                        "concentration_ppm_uib": None,
                        "valid": False,
                        "reason": "Patró de referència (sense quantificació)"
                    }
                else:
                    # Usar calibració específica segons volum d'injecció
                    sample_cal = get_sample_calibration(r1)
                    quantification = quantify_sample(r1, sample_cal, mode=mode)
                    # HCI
                    hci = r1.get("hci")
                    if hci is not None:
                        quantification["hci"] = hci
                        quantification["hci_character"] = r1.get("hci_character", "")
                    sample_group["quantification"] = quantification

        # Detectar composabilitat de timeouts entre rèpliques
        rep_keys_list = sorted(replicas.keys())
        if len(rep_keys_list) >= 2:
            r1_ti = replicas[rep_keys_list[0]].get("timeout_info", {})
            r2_ti = replicas[rep_keys_list[1]].get("timeout_info", {})
            r1_has = (r1_ti.get("n_timeouts", 0) or len(r1_ti.get("timeouts", []))) > 0
            r2_has = (r2_ti.get("n_timeouts", 0) or len(r2_ti.get("timeouts", []))) > 0
            if r1_has and r2_has:
                from hpsec_core import check_timeout_composability
                run_dur = 12.0 if is_bp else 70.0
                sample_group["timeout_composability"] = check_timeout_composability(
                    r1_ti, r2_ti, run_duration_min=run_dur
                )

        result["samples_grouped"][sample_name] = sample_group

    # =========================================================================
    # ESTIMAR TIMEOUTS UIB (post-processat de seqüència)
    # =========================================================================
    # Timeout UIB: ja propagat des de Direct a import (single source of truth).
    # _estimate_uib_timeouts_from_sequence ja no és necessària.

    # =========================================================================
    # GENERAR RESUM
    # =========================================================================
    n_valid = sum(1 for s in result["samples"] if s.get("processed") and s.get("peak_info", {}).get("valid"))
    n_with_anomalies = sum(1 for s in result["samples"] if s.get("anomalies"))
    n_timeouts = sum(1 for s in result["samples"] if s.get("timeout_info", {}).get("n_timeouts", 0) > 0)
    n_boundary_timeouts = sum(1 for s in result["samples"] if s.get("timeout_at_boundary"))
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
        "with_boundary_timeouts": n_boundary_timeouts,
        "with_replica_warnings": n_with_warnings,
        "n_khp": sum(1 for s in result["samples"] if s.get("sample_type") == "KHP"),
        "n_controls": sum(1 for s in result["samples"] if s.get("sample_type") == "CONTROL"),
        "n_blank": sum(1 for s in result["samples"] if s.get("sample_type") == "BLANK"),
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
        logger.warning("Error registrant mostres a l'índex: %s", e)

    # Estampar config fingerprint per detectar obsolescència
    try:
        from hpsec_config import get_config
        result["config_fingerprint"] = get_config().compute_config_fingerprint()
    except Exception as e:
        logger.warning("Config fingerprint computation failed: %s", e)

    # Estampar calibration fingerprint per detectar canvis de calibració
    try:
        from hpsec_calibrate import compute_calibration_fingerprint
        result["calibration_fingerprint"] = compute_calibration_fingerprint()
    except Exception as e:
        logger.warning("Calibration fingerprint computation failed: %s", e)

    if progress_callback:
        progress_callback("Processing complete", 100)

    # Flag per consumidors: la quantificació està pendent (cal cridar quantify_sequence)
    result["quantification_pending"] = not do_quantify

    # Els fitxers per mostra per a ús extern ara els genera el dataset FAIR
    # (hpsec_fair, traces/ + results_SEC.csv) a l'exportació, no aquí.

    return result


# =============================================================================
# QUANTIFY SEQUENCE — Aplica calibració a un analysis_result ja calculat
# =============================================================================


def quantify_sequence(analysis_result, seq_path=None, mode=None, seq_date=None,
                      progress_callback=None):
    """
    FASE 4 (separada): Aplica calibració (àrea → ppm) a un analysis_result.

    Permet quantificar després d'analitzar sense reprocessar cromatogrames.
    Útil quan:
    - L'usuari canvia la rèplica seleccionada i vol re-quantificar
    - S'aplica una nova recta de calibració
    - El pipeline separa Analitzar (qualitatiu) de Quantificar

    Args:
        analysis_result: Dict retornat per analyze_sequence(do_quantify=False)
        seq_path: Path de la SEQ (per carregar calibracions actives). Si None,
            s'usa analysis_result["seq_path"].
        mode: "COLUMN" o "BP". Si None, s'usa analysis_result["method"].
        seq_date: Data de la SEQ per seleccionar calibració. Si None, activa.
        progress_callback: Funció callback per reportar progrés (msg, pct).

    Returns:
        analysis_result enriquit amb sample_group["quantification"] per cada mostra.
        També posa result["quantification_pending"] = False.
    """
    if not analysis_result or not analysis_result.get("samples_grouped"):
        logger.warning("quantify_sequence: analysis_result buit o sense samples_grouped")
        return analysis_result

    if seq_path is None:
        seq_path = analysis_result.get("seq_path", "")
    if mode is None:
        method = analysis_result.get("method", "COLUMN")
        mode = "BP" if method.upper() == "BP" else "COLUMN"

    # Carregar totes les calibracions actives per aquesta SEQ
    multi_calibrations = {}
    if seq_path:
        try:
            active_cals = get_all_active_calibrations(seq_path, mode)
            for cal in active_cals:
                vol = cal.get("volume_uL", 0)
                if vol > 0:
                    multi_calibrations[vol] = cal
        except Exception as e:
            logger.warning("quantify_sequence: error carregant calibracions: %s", e)

    def _get_sample_cal(sample):
        if multi_calibrations:
            inj_vol = sample.get("inj_volume")
            if inj_vol and inj_vol in multi_calibrations:
                return multi_calibrations[inj_vol]
            return next(iter(multi_calibrations.values()))
        return {}

    # Patrons exclusió quantificació (PR_I, etc.)
    try:
        from hpsec_config import get_config
        cfg = get_config()
        no_quant_patterns = cfg.get("no_quantification_patterns", []) or []
    except Exception:
        no_quant_patterns = []

    samples_grouped = analysis_result["samples_grouped"]
    total = len(samples_grouped)
    processed = 0

    for sample_name, sample_group in samples_grouped.items():
        processed += 1
        if progress_callback:
            progress_callback(f"Quantificant {sample_name}",
                              int(processed / total * 100) if total else 100)

        # Patró excloent quantificació?
        skip_quant = any(pat.lower() in sample_name.lower() for pat in no_quant_patterns)
        # Override per usuari (right-click → "Excloure quantificació")
        # v2.2.0: la clau dominant al codebase és `skip_quantification`
        if sample_group.get("skip_quantification") or sample_group.get("exclude_from_quantification"):
            skip_quant = True

        # Mostra no vàlida?
        sample_valid = sample_group.get("sample_valid", True)
        replicas = sample_group.get("replicas", {})
        selected_doc = (sample_group.get("selected") or {}).get("doc")

        if sample_valid is False:
            sample_group["quantification"] = {
                "concentration_ppm": None,
                "concentration_ppm_direct": None,
                "concentration_ppm_uib": None,
                "area_total": None,
                "valid": False,
                "reason": "Mostra no vàlida"
            }
            continue

        if skip_quant:
            sample_group["quantification"] = {
                "concentration_ppm": None,
                "concentration_ppm_direct": None,
                "concentration_ppm_uib": None,
                "valid": False,
                "reason": "Patró de referència (sense quantificació)"
            }
            continue

        # Buscar la rèplica seleccionada
        # v2.2.0: tractar "none"/"Cap"/None com a explicit-no-replica (usuari excloent)
        selected_sample = None
        if selected_doc in (None, "", "none", "Cap"):
            sample_group["quantification"] = {
                "concentration_ppm": None,
                "concentration_ppm_direct": None,
                "concentration_ppm_uib": None,
                "valid": False,
                "reason": "Rèplica explícitament exclosa per l'usuari"
            }
            continue

        if selected_doc and selected_doc in replicas:
            selected_sample = replicas[selected_doc]
        elif replicas:
            # Fallback: primera rèplica (només si no s'ha exclós explícitament)
            selected_sample = next(iter(replicas.values()))

        if not selected_sample:
            sample_group["quantification"] = {
                "concentration_ppm": None,
                "valid": False,
                "reason": "Sense rèplica disponible"
            }
            continue

        # ─── v2.2.0+: quantificació per CADA rèplica vàlida + estadística ───
        # Es quantifiquen totes les rèpliques (incloent siblings: R1A, R2A,
        # R1B, R2B…) que NO siguin outlier i NO portin anomalia amb
        # invalidates=True. Una anomalia 'blocker' però amb invalidates=False
        # (e.g. IRREGULAR_TOP_DIRECT, que és reparable) NO ha de bloquejar
        # la quantificació — la dada ja s'ha corregit al pas d'integració.
        try:
            from hpsec_warnings import ANOMALY_CATALOG
        except Exception:
            ANOMALY_CATALOG = {}

        def _is_invalidating(code):
            entry = ANOMALY_CATALOG.get(code, {})
            return bool(entry.get("invalidates", False))

        per_replica = {}
        for rk, rd in replicas.items():
            if not isinstance(rd, dict):
                continue
            if rd.get("is_outlier", False):
                continue
            anoms = rd.get("anomalies") or []
            has_invalidating = any(
                _is_invalidating(a.get("code")) if isinstance(a, dict)
                else False
                for a in anoms
            )
            if has_invalidating:
                continue
            rep_cal = _get_sample_cal(rd)
            try:
                q_rep = quantify_sample(rd, rep_cal, mode=mode, seq_date=seq_date)
            except Exception as e:
                logger.warning(
                    "Error quantificant rèplica %s de %s: %s", rk, sample_name, e)
                continue
            per_replica[str(rk)] = {
                "ppm_direct": q_rep.get("concentration_ppm_direct"),
                "ppm_uib": q_rep.get("concentration_ppm_uib"),
                "area_total": q_rep.get("area_total"),
                "area_total_uib": q_rep.get("area_total_uib"),
                "rf_mass_cal_used": q_rep.get("rf_mass_cal_used"),
                "rf_mass_cal_uib_used": q_rep.get("rf_mass_cal_uib_used"),
                "intercept": q_rep.get("intercept"),
                "intercept_uib": q_rep.get("intercept_uib"),
                "volume_uL": q_rep.get("volume_uL"),
                "source_label": rd.get("_source_label", ""),
                "injection_index": rd.get("injection_index"),
            }

        # Calcular estadística només amb les rèpliques vàlides
        import statistics as _stats
        direct_vals = [v["ppm_direct"] for v in per_replica.values()
                       if v["ppm_direct"] is not None]
        uib_vals = [v["ppm_uib"] for v in per_replica.values()
                    if v["ppm_uib"] is not None]

        def _stat(values):
            if not values:
                return {"n": 0, "mean": None, "sd": None, "rsd_pct": None}
            n = len(values)
            mean = sum(values) / n
            sd = _stats.stdev(values) if n > 1 else 0.0
            rsd = (100.0 * sd / mean) if mean else None
            return {"n": n, "mean": mean, "sd": sd, "rsd_pct": rsd}

        statistics_block = {
            "direct": _stat(direct_vals),
            "uib": _stat(uib_vals),
        }

        # Quantificació "final" de la mostra: la rèplica seleccionada
        sample_cal = _get_sample_cal(selected_sample)
        quantification = quantify_sample(selected_sample, sample_cal, mode=mode,
                                          seq_date=seq_date)
        # HCI
        hci = selected_sample.get("hci")
        if hci is not None:
            quantification["hci"] = hci
            quantification["hci_character"] = selected_sample.get("hci_character", "")

        # Compatibilitat: 'concentration_ppm_direct' és el de la seleccionada
        # (com fins ara). Els nous camps per_replica + statistics afegeixen
        # info per a estadística i UI.
        quantification["per_replica"] = per_replica
        quantification["statistics"] = statistics_block
        quantification["selected_replica"] = str(selected_doc) if selected_doc else None
        sample_group["quantification"] = quantification

    analysis_result["quantification_pending"] = False

    # Re-estampar calibration fingerprint (nova recta aplicada)
    try:
        from hpsec_calibrate import compute_calibration_fingerprint
        analysis_result["calibration_fingerprint"] = compute_calibration_fingerprint()
    except Exception as e:
        logger.warning("Calibration fingerprint computation failed: %s", e)

    # (Els fitxers per mostra ara els genera el dataset FAIR a l'exportació.)

    if progress_callback:
        progress_callback("Quantificació completa", 100)

    return analysis_result


# =============================================================================
# GUARDAR RESULTAT ANÀLISI (JSON)
# =============================================================================

# Font única: hpsec_import (mateixa carpeta CHECK/data)
from hpsec_import import get_data_folder


from hpsec_utils import NumpyEncoder, _atomic_write_json

# La llista plana 'samples' NO porta arrays a disc: la GUI redibuixa des de
# 'samples_grouped', que és l'única còpia persistida dels senyals. Qualsevol
# writer d'analysis_result.json ha de passar per strip_flat_sample_arrays()
# (el Quantificar reescrivia el dict cru i re-inflava el fitxer a 23 MB).
ANALYSIS_FLAT_ARRAY_KEYS = ("t_doc", "y_doc_net", "y_doc_uib_net",
                            "y_doc_direct_net", "df_dad",
                            "y_doc_net_pre_composition")


def strip_flat_sample_arrays(result):
    """
    Retorna una còpia superficial de `result` amb les entrades de la llista
    plana 'samples' sense els camps d'array (ANALYSIS_FLAT_ARRAY_KEYS).
    No muta l'original: les dades en memòria conserven els arrays.
    """
    samples = result.get("samples")
    if not samples:
        return result
    out = dict(result)
    out["samples"] = [
        {k: v for k, v in s.items() if k not in ANALYSIS_FLAT_ARRAY_KEYS}
        if isinstance(s, dict) else s
        for s in samples
    ]
    return out


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
    from hpsec_version import SUITE_VERSION
    result = {
        "suite_version": SUITE_VERSION,
        "analyze_module": __version__,
        "timestamp": datetime.now().isoformat(),
        "seq_name": analysis_data.get("seq_name", ""),
        "seq_path": os.path.basename(seq_path),  # Relatiu: només nom SEQ
        "method": analysis_data.get("method", ""),
        "data_mode": analysis_data.get("data_mode", ""),
        "success": analysis_data.get("success", False),
        "errors": analysis_data.get("errors", []),
        "warnings": analysis_data.get("warnings", []),
        "warning_level": analysis_data.get("warning_level", "none"),
        "config_fingerprint": analysis_data.get("config_fingerprint", ""),
        "calibration_fingerprint": analysis_data.get("calibration_fingerprint", ""),
        # v2.2.0: flag que indica si la quantificació encara està pendent.
        # Quan True, l'usuari ha d'anar al pas Quantificar per aplicar la recta.
        "quantification_pending": analysis_data.get("quantification_pending", True),
        "summary": analysis_data.get("summary", {}),
        "samples": [],
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
            "injection_index": sample.get("injection_index"),
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
            except Exception as e:
                logger.warning("Failed to serialize df_dad: %s", e)

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
            "timeout_info_uib": sample.get("timeout_info_uib", {}),
            "snr_info": sample.get("snr_info", {}),
            "snr_info_dad": sample.get("snr_info_dad", {}),
            "irregular_top_uib": sample.get("irregular_top_uib"),
            "pearson_direct_uib": sample.get("pearson_direct_uib"),
            "area_diff_pct": sample.get("area_diff_pct"),
            # --- Camps escalars nous ---
            "tmax_signals": sample.get("tmax_signals", {}),
            "n_peaks_254_HS": sample.get("n_peaks_254_HS"),
            "n_peaks_per_wl": sample.get("n_peaks_per_wl", {}),
            "is_bp": sample.get("is_bp", False),
            "is_dual": sample.get("is_dual", False),
            "is_uib_only": sample.get("is_uib_only", False),
            "direct_estimated_from_uib": sample.get("direct_estimated_from_uib", False),
            "direct_estimation_factor": sample.get("direct_estimation_factor"),
            "irregular_top_direct": sample.get("irregular_top_direct"),
            "irregular_top_direct_info": sample.get("irregular_top_direct_info"),
            "bigaussian_doc": sample.get("bigaussian_doc"),
            "bigaussian_254": sample.get("bigaussian_254"),
            "fwhm_doc": sample.get("fwhm_doc"),
            "symmetry_doc": sample.get("symmetry_doc"),
            "inj_volume": sample.get("inj_volume"),
            "injection_index": sample.get("injection_index"),
            # --- Camps cromatograma (arrays) ---
            "t_doc": sample.get("t_doc"),
            "y_doc_net": sample.get("y_doc_net"),
            "y_doc_uib_net": sample.get("y_doc_uib_net"),
            "y_doc_direct_net": sample.get("y_doc_direct_net"),
            "df_dad": df_dad_serializable,
            # --- Path Export3D per RAW export (FAIR) ---
            "dad_export3d_path": sample.get("dad_export3d_path"),
            # --- Composició timeout (persistir si aplicada) ---
            "timeout_composition": sample.get("timeout_composition"),
            "y_doc_net_pre_composition": sample.get("y_doc_net_pre_composition"),
        }

    # Resumir mostres. La llista plana 'samples' NO la redibuixa la GUI (usa
    # 'samples_grouped'); per no duplicar MB, la persistim SENSE els arrays de
    # senyal (única còpia: samples_grouped). Estalvi ~50%.
    for sample in analysis_data.get("samples", []):
        if sample.get("analysis_type") == "light":
            entry = summarize_light_sample(sample)
        else:
            entry = summarize_sample(sample)
        for _ak in ANALYSIS_FLAT_ARRAY_KEYS:
            entry.pop(_ak, None)
        result["samples"].append(entry)

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
                    "skip_quantification": sample_data.get("skip_quantification", False),
                    "sample_type": sample_data.get("sample_type", "SAMPLE"),
                    "repairable": sample_data.get("repairable", False),
                    "repaired": sample_data.get("repaired", False),
                    "repair_history": sample_data.get("repair_history", []),
                }
                for rep_key, rep_data in sample_data.get("replicas", {}).items():
                    grouped_entry["replicas"][rep_key] = summarize_sample(rep_data)

                # Composabilitat de timeouts (ja calculada a analyze_sequence, copiar)
                tc = sample_data.get("timeout_composability")
                if tc:
                    grouped_entry["timeout_composability"] = tc

            result["samples_grouped"][sample_name] = grouped_entry

    # Guardar (ATÒMIC; encoder NumpyEncoder únic per a aquest fitxer — abans
    # Quantificar el reescrivia amb un encoder diferent). Sense indentació:
    # el fitxer és intern (cap humà l'edita) i la indentació l'inflava un 40%.
    try:
        _atomic_write_json(output_path, result, indent=None, ensure_ascii=False,
                           cls=NumpyEncoder)
        return output_path
    except Exception as e:
        print(f"Error guardant analysis_result.json: {e}")
        return None


def _restore_dataframes(data):
    """Converteix df_dad dicts a DataFrames i llistes a numpy arrays en dades carregades de JSON."""
    _ARRAY_KEYS = ("t_doc", "y_doc_net", "y_doc_uib_net", "y_doc_direct_net",
                    "y_doc_net_pre_composition")

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

    data_folder = get_data_folder(seq_path, create=False)
    filepath = os.path.join(data_folder, "analysis_result.json")

    if not os.path.exists(filepath):
        return None

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        _restore_dataframes(data)

        # v2.2.0: resoldre seq_path a absolut (al JSON es desa com a basename)
        if data.get("seq_path") and not os.path.isabs(data["seq_path"]):
            data["seq_path"] = os.path.abspath(seq_path)

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
    # Funcions principals
    "analyze_sample",
    "analyze_sequence",
    # Comparació rèpliques (v1.5.0)
    "compare_replicas",
    "recommend_replica",
    "quantify_sample",
    "quantify_sequence",
    # Constants
    "REPLICA_PEARSON_THRESHOLD",
    "REPLICA_AREA_DIFF_THRESHOLD",
    "REPLICA_FRAC_DIFF_THRESHOLD",
]
