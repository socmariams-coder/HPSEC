# -*- coding: utf-8 -*-
"""
hpsec_replica.py - Selecció de Rèpliques HPSEC
===============================================

Mòdul unificat per avaluar i seleccionar la millor rèplica entre R1 i R2.
Single Source of Truth per a tota la Suite HPSEC.

Usat per:
- HPSEC_Suite.py (processament principal)
- DOCtor_C.py (anàlisi Column)
- DOCtor_BP.py (anàlisi Bypass)

Criteris de selecció:
- COLUMN:
  1. Anomalies (Batman > Timeout > IRR)
  2. SNR (si diferència > 1.5x)
  3. Pearson entre rèpliques (warning si < 0.990)
  4. Diferència d'àrea (warning si > 15%)
  5. Alçada (tiebreaker)

- BP:
  1. Anomalies (Batman)
  2. R² status (VALID > CHECK > INVALID)
  3. R² valor
  4. SNR (tiebreaker)

v1.0 - 2026-01-26 - Creació inicial
v1.1 - 2026-01-26 - Criteris unificats COLUMN/BP:
      - COLUMN: +SNR, +Pearson actiu, +Àrea diff warnings
      - BP: +Pearson/Àrea warnings
      - Nous llindars: SNR_RATIO_THRESHOLD, AREA_DIFF_WARNING/CRITICAL
v1.2 - 2026-01-26 - Avaluació DAD:
      - evaluate_dad(): deriva, soroll, SNR per wavelength
      - evaluate_dad_multi(): múltiples wavelengths
      - compare_doc_dad(): correlació DOC-DAD
      - compare_replicas_full(): comparació completa DOC+DAD
      - COLUMN: DAD quality i drift afecten selecció
      - BP: DAD warnings (no afecten decisió, només confiança)
      - Nous llindars: DAD_DRIFT_*, DAD_NOISE_*, DAD_DOC_CORR_*
"""

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import pearsonr
from scipy.integrate import trapezoid

from hpsec_core import (
    detect_timeout, detect_peak_anomaly, TIMEOUT_CONFIG,
    fit_bigaussian, calc_snr,
    THRESH_R2_VALID, THRESH_R2_CHECK, ASYM_MIN, ASYM_MAX
)


# =============================================================================
# CONFIGURACIÓ
# =============================================================================

# Llindars generals
NOISE_THRESHOLD = 20.0      # mAU - sota aquest valor, senyal massa baix
PEARSON_THRESHOLD = 0.995   # Correlació mínima acceptable entre rèpliques
PEARSON_WARNING = 0.990     # Pearson sota aquest valor genera warning
SNR_RATIO_THRESHOLD = 1.5   # Si SNR_r1/SNR_r2 > 1.5, preferir la de major SNR
AREA_DIFF_WARNING = 15.0    # % diferència d'àrea per generar warning
AREA_DIFF_CRITICAL = 30.0   # % diferència d'àrea per marcar com CHECK

# DAD quality thresholds
DAD_DRIFT_WARNING = 1.0     # mAU - deriva alta
DAD_DRIFT_CRITICAL = 3.0    # mAU - deriva molt alta
DAD_NOISE_WARNING = 0.5     # mAU - soroll alt
DAD_DOC_CORR_MIN = 0.90     # Correlació mínima DOC-DAD
DAD_DOC_CORR_WARNING = 0.95 # Warning si correlació < 0.95

# Zona húmics (per limitar detecció Batman a Column)
HUMIC_ZONE = (18.0, 23.0)   # minuts

# Pesos per score (només informatiu, la selecció usa prioritats)
SCORE_WEIGHTS = {
    "batman": 100,      # Penalització per Batman
    "timeout": 50,      # Penalització per Timeout
    "irr": 30,          # Penalització per irregularitat
    "low_signal": 20,   # Penalització per senyal baix
}


# =============================================================================
# FUNCIONS AUXILIARS
# =============================================================================

def _baseline_stats(y):
    """Calcula estadístiques de baseline (primers 10% de punts)."""
    y = np.asarray(y, dtype=float)
    n = max(10, len(y) // 10)
    baseline = y[:n]
    return {
        "mean": float(np.mean(baseline)),
        "std": float(np.std(baseline)),
        "percentile_10": float(np.percentile(y, 10))
    }


def _find_main_peak(t, y):
    """Troba el pic principal i retorna info bàsica."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(t) < 20:
        return {"valid": False, "reason": "Poques dades"}

    baseline = float(np.percentile(y, 10))
    max_val = float(np.max(y))

    if max_val - baseline < 1e-6:
        return {"valid": False, "reason": "Senyal pla"}

    # Trobar pics
    min_prom = (max_val - baseline) * 0.05
    peaks, props = find_peaks(y, prominence=min_prom, width=2)

    if len(peaks) == 0:
        return {"valid": False, "reason": "Cap pic trobat"}

    # Pic principal = màxima prominència
    main_idx = peaks[np.argmax(props["prominences"])]
    main_prom_idx = np.argmax(props["prominences"])

    left_base = int(props["left_bases"][main_prom_idx])
    right_base = int(props["right_bases"][main_prom_idx])

    height = float(y[main_idx]) - baseline
    t_peak = float(t[main_idx])

    # Àrea del pic
    t_seg = t[left_base:right_base+1]
    y_seg = y[left_base:right_base+1] - baseline
    y_seg = np.maximum(y_seg, 0)
    area = float(trapezoid(y_seg, t_seg)) if len(t_seg) > 1 else 0

    return {
        "valid": True,
        "peak_idx": int(main_idx),
        "left_base": left_base,
        "right_base": right_base,
        "t_peak": t_peak,
        "height": height,
        "area": area,
        "baseline": baseline
    }


def _calc_roughness(y):
    """Calcula rugositat del senyal (suma de |diffs| / àrea)."""
    y = np.asarray(y, dtype=float)
    if len(y) < 2:
        return 999999.0

    diffs = np.abs(np.diff(y))
    total = np.sum(np.abs(y))

    if total < 1e-6:
        return 999999.0

    return float(np.sum(diffs) / total)


# =============================================================================
# AVALUACIÓ DE RÈPLICA
# =============================================================================

def evaluate_replica(t, y, method="COLUMN", humic_zone=None):
    """
    Avalua la qualitat d'una rèplica.

    Parameters
    ----------
    t : array-like
        Temps (minuts)
    y : array-like
        Senyal (mAU)
    method : str
        "COLUMN" o "BP"
    humic_zone : tuple, optional
        (t_start, t_end) per limitar detecció Batman. Default: HUMIC_ZONE

    Returns
    -------
    dict
        Diccionari amb:
        - valid: bool
        - reason: str (si no vàlid)
        - height: float (alçada pic principal)
        - area: float (àrea pic principal)
        - snr: float (signal-to-noise ratio)
        - batman: bool (té Batman)
        - irr: bool (és irregular)
        - timeout: bool (té timeout)
        - timeout_info: dict (detalls timeout)
        - r2: float (només BP, qualitat ajust)
        - r2_status: str (només BP, "VALID"/"CHECK"/"INVALID")
        - anomaly_score: float (score penalització)
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(t) < 20 or len(y) < 20:
        return {"valid": False, "reason": "Poques dades"}

    # Filtrar NaN/Inf
    mask = np.isfinite(t) & np.isfinite(y)
    t, y = t[mask], y[mask]

    if len(t) < 20:
        return {"valid": False, "reason": "Massa NaN/Inf"}

    # Estadístiques baseline
    bl_stats = _baseline_stats(y)
    max_signal = float(np.max(y))

    # Senyal massa baix?
    if max_signal < NOISE_THRESHOLD:
        return {
            "valid": True,  # Vàlid però marcat
            "low_signal": True,
            "height": max_signal - bl_stats["percentile_10"],
            "area": 0,
            "snr": 0,
            "batman": False,
            "irr": False,
            "timeout": False,
            "timeout_info": None,
            "anomaly_score": SCORE_WEIGHTS["low_signal"],
            "reason": "LOW_SIGNAL"
        }

    # Trobar pic principal
    peak_info = _find_main_peak(t, y)
    if not peak_info["valid"]:
        return {"valid": False, "reason": peak_info["reason"]}

    height = peak_info["height"]
    area = peak_info["area"]
    t_peak = peak_info["t_peak"]

    # SNR
    snr = height / bl_stats["std"] if bl_stats["std"] > 0 else 0.0

    # === DETECCIÓ ANOMALIES ===

    # Timeout (via dt intervals - hpsec_core)
    timeout_info = detect_timeout(t)
    has_timeout = timeout_info.get("n_timeouts", 0) > 0

    # Batman i Irregularitat (via detect_peak_anomaly)
    has_batman = False
    has_irr = False
    smoothness = 100.0

    # Per Column, només detectar Batman a zona húmics
    hz = humic_zone or HUMIC_ZONE
    in_humic_zone = (method == "COLUMN" and hz[0] <= t_peak <= hz[1]) or method == "BP"

    if in_humic_zone or method == "BP":
        # Extreure segment del pic
        left_base = peak_info["left_base"]
        right_base = peak_info["right_base"]
        t_seg = t[left_base:right_base+1]
        y_seg = y[left_base:right_base+1]

        if len(t_seg) >= 10:
            anomaly = detect_peak_anomaly(
                t_seg, y_seg,
                top_pct=0.15,
                min_valley_depth=0.05,
                smoothness_threshold=18.0
            )
            has_batman = anomaly.get("is_batman", False)
            has_irr = anomaly.get("is_irregular", False)
            smoothness = anomaly.get("smoothness", 100.0)

    # === ESPECÍFIC PER BP: Ajust bi-gaussià ===
    r2 = None
    r2_status = None
    asymmetry = None

    if method == "BP":
        fit_result = fit_bigaussian(
            t, y,
            peak_info["peak_idx"],
            peak_info["left_base"],
            peak_info["right_base"]
        )

        # fit_bigaussian retorna "status" (VALID/CHECK/INVALID), no "valid"
        fit_status = fit_result.get("status", "INVALID")
        if fit_status in ("VALID", "CHECK"):
            r2 = fit_result.get("r2", 0)
            r2_status = fit_status
            asymmetry = fit_result.get("asymmetry", 1.0)
        else:
            r2 = fit_result.get("r2", 0.0)  # Mantenir r2 encara que INVALID
            r2_status = "INVALID"
            asymmetry = fit_result.get("asymmetry")

    # === CALCULAR SCORE ===
    anomaly_score = 0.0
    if has_batman:
        anomaly_score += SCORE_WEIGHTS["batman"]
    if has_timeout:
        anomaly_score += SCORE_WEIGHTS["timeout"]
    if has_irr:
        anomaly_score += SCORE_WEIGHTS["irr"]

    return {
        "valid": True,
        "low_signal": False,
        "height": height,
        "area": area,
        "t_peak": t_peak,
        "snr": snr,
        "baseline_noise": bl_stats["std"],
        "batman": has_batman,
        "irr": has_irr,
        "smoothness": smoothness,
        "timeout": has_timeout,
        "timeout_info": timeout_info,
        "r2": r2,
        "r2_status": r2_status,
        "asymmetry": asymmetry,
        "anomaly_score": anomaly_score,
        "peak_info": peak_info
    }


# =============================================================================
# AVALUACIÓ DAD
# =============================================================================

def evaluate_dad(t, y, wavelength="A254"):
    """
    Avalua la qualitat d'un senyal DAD.

    Parameters
    ----------
    t : array-like
        Temps (minuts)
    y : array-like
        Senyal DAD (mAU)
    wavelength : str
        Nom de la longitud d'ona (per informació)

    Returns
    -------
    dict
        - valid: bool
        - drift: float (mAU) - canvi baseline inici vs final
        - drift_pct: float (%) - deriva relativa al senyal màxim
        - noise: float (mAU) - soroll baseline
        - snr: float - signal-to-noise ratio
        - max_signal: float (mAU)
        - quality: str ("OK", "WARNING", "POOR")
        - issues: list[str] - problemes detectats
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(t) < 20 or len(y) < 20:
        return {"valid": False, "reason": "Poques dades", "quality": "POOR"}

    # Filtrar NaN/Inf
    mask = np.isfinite(t) & np.isfinite(y)
    t, y = t[mask], y[mask]

    if len(t) < 20:
        return {"valid": False, "reason": "Massa NaN/Inf", "quality": "POOR"}

    issues = []

    # === DERIVA ===
    # Comparar mitjana primers 5 min vs últims 5 min
    n_points = len(t)
    n_baseline = max(10, n_points // 20)  # 5% de punts

    # Zona inicial (primers punts, t < 5 min o primers 5%)
    mask_start = t < min(5.0, t[n_baseline])
    if np.sum(mask_start) < 5:
        mask_start = np.zeros(n_points, dtype=bool)
        mask_start[:n_baseline] = True

    # Zona final (últims punts, t > 60 min o últims 5%)
    mask_end = t > max(60.0, t[-n_baseline])
    if np.sum(mask_end) < 5:
        mask_end = np.zeros(n_points, dtype=bool)
        mask_end[-n_baseline:] = True

    baseline_start = float(np.mean(y[mask_start])) if np.any(mask_start) else 0
    baseline_end = float(np.mean(y[mask_end])) if np.any(mask_end) else 0

    drift = baseline_end - baseline_start
    max_signal = float(np.max(y))

    # Deriva relativa
    if max_signal > 1e-6:
        drift_pct = abs(drift) / max_signal * 100
    else:
        drift_pct = 0.0

    # === SOROLL BASELINE ===
    # Usar zona inicial per estimar soroll
    noise = float(np.std(y[mask_start])) if np.any(mask_start) else 0

    # === SNR ===
    peak_height = max_signal - baseline_start
    snr = peak_height / noise if noise > 1e-6 else 0

    # === AVALUAR QUALITAT ===
    quality = "OK"

    if abs(drift) >= DAD_DRIFT_CRITICAL:
        issues.append(f"Deriva ALTA ({drift:.2f} mAU)")
        quality = "POOR"
    elif abs(drift) >= DAD_DRIFT_WARNING:
        issues.append(f"Deriva ({drift:.2f} mAU)")
        if quality == "OK":
            quality = "WARNING"

    if noise >= DAD_NOISE_WARNING:
        issues.append(f"Soroll alt ({noise:.2f} mAU)")
        if quality == "OK":
            quality = "WARNING"

    if max_signal < NOISE_THRESHOLD:
        issues.append("Senyal baix")
        quality = "POOR"

    return {
        "valid": True,
        "wavelength": wavelength,
        "drift": float(drift),
        "drift_pct": float(drift_pct),
        "noise": float(noise),
        "snr": float(snr),
        "max_signal": float(max_signal),
        "baseline_start": float(baseline_start),
        "baseline_end": float(baseline_end),
        "quality": quality,
        "issues": issues
    }


def evaluate_dad_multi(t, dad_data, wavelengths=None):
    """
    Avalua múltiples wavelengths DAD.

    Parameters
    ----------
    t : array-like
        Temps (minuts)
    dad_data : dict or DataFrame
        Dades DAD per wavelength. Si és dict: {wavelength: y_array}
        Si és DataFrame: columnes són wavelengths
    wavelengths : list, optional
        Llista de wavelengths a avaluar. Per defecte: ["A254", "A280"]

    Returns
    -------
    dict
        - valid: bool
        - wavelengths: dict[str, dict] - avaluació per wavelength
        - overall_quality: str ("OK", "WARNING", "POOR")
        - worst_wavelength: str - wavelength amb pitjor qualitat
        - issues: list[str] - tots els problemes
    """
    if wavelengths is None:
        wavelengths = ["A254", "A280"]

    # Convertir DataFrame a dict si cal
    if hasattr(dad_data, 'columns'):
        # És un DataFrame
        data_dict = {}
        for wl in wavelengths:
            for col in dad_data.columns:
                if wl.lower() in col.lower() or wl.replace("A", "") in col:
                    data_dict[wl] = dad_data[col].values
                    break
    else:
        data_dict = dad_data

    results = {}
    all_issues = []
    quality_order = {"OK": 0, "WARNING": 1, "POOR": 2}
    worst_quality = "OK"
    worst_wl = None

    for wl in wavelengths:
        if wl in data_dict:
            y = data_dict[wl]
            eval_wl = evaluate_dad(t, y, wavelength=wl)
            results[wl] = eval_wl

            if eval_wl.get("issues"):
                all_issues.extend([f"{wl}: {issue}" for issue in eval_wl["issues"]])

            wl_quality = eval_wl.get("quality", "OK")
            if quality_order.get(wl_quality, 0) > quality_order.get(worst_quality, 0):
                worst_quality = wl_quality
                worst_wl = wl

    return {
        "valid": len(results) > 0,
        "wavelengths": results,
        "overall_quality": worst_quality,
        "worst_wavelength": worst_wl,
        "issues": all_issues
    }


def compare_doc_dad(t_doc, y_doc, t_dad, y_dad, window=None):
    """
    Compara DOC amb DAD (A254) per verificar coherència.

    Parameters
    ----------
    t_doc, y_doc : array-like
        Temps i senyal DOC
    t_dad, y_dad : array-like
        Temps i senyal DAD (normalment A254)
    window : tuple, optional
        (t_min, t_max) per limitar comparació

    Returns
    -------
    dict
        - valid: bool
        - pearson: float - correlació DOC-DAD
        - quality: str ("OK", "WARNING", "POOR")
        - issues: list[str]
    """
    t_doc = np.asarray(t_doc, dtype=float)
    y_doc = np.asarray(y_doc, dtype=float)
    t_dad = np.asarray(t_dad, dtype=float)
    y_dad = np.asarray(y_dad, dtype=float)

    if len(t_doc) < 10 or len(t_dad) < 10:
        return {"valid": False, "pearson": np.nan, "quality": "POOR"}

    # Rang comú
    t_min = max(t_doc.min(), t_dad.min())
    t_max = min(t_doc.max(), t_dad.max())

    if window:
        t_min = max(t_min, window[0])
        t_max = min(t_max, window[1])

    if t_max <= t_min:
        return {"valid": False, "pearson": np.nan, "quality": "POOR"}

    # Interpolar a punts comuns
    t_common = np.linspace(t_min, t_max, 500)
    y_doc_interp = np.interp(t_common, t_doc, y_doc)
    y_dad_interp = np.interp(t_common, t_dad, y_dad)

    # Pearson
    try:
        r, _ = pearsonr(y_doc_interp, y_dad_interp)
    except:
        r = np.nan

    issues = []
    quality = "OK"

    if np.isnan(r):
        quality = "POOR"
        issues.append("No es pot calcular correlació")
    elif r < DAD_DOC_CORR_MIN:
        quality = "POOR"
        issues.append(f"Correlació DOC-DAD molt baixa ({r:.3f})")
    elif r < DAD_DOC_CORR_WARNING:
        quality = "WARNING"
        issues.append(f"Correlació DOC-DAD baixa ({r:.3f})")

    return {
        "valid": True,
        "pearson": float(r) if not np.isnan(r) else np.nan,
        "quality": quality,
        "issues": issues
    }


# =============================================================================
# COMPARACIÓ ENTRE RÈPLIQUES
# =============================================================================

def compare_replicas(t1, y1, t2, y2, window=None):
    """
    Compara dues rèpliques calculant Pearson i diferència d'àrea.

    Parameters
    ----------
    t1, y1 : array-like
        Temps i senyal de rèplica 1
    t2, y2 : array-like
        Temps i senyal de rèplica 2
    window : tuple, optional
        (t_min, t_max) per limitar la comparació

    Returns
    -------
    dict
        - pearson: float (correlació)
        - area_diff_pct: float (diferència àrea en %)
        - valid: bool
    """
    t1 = np.asarray(t1, dtype=float)
    y1 = np.asarray(y1, dtype=float)
    t2 = np.asarray(t2, dtype=float)
    y2 = np.asarray(y2, dtype=float)

    if len(t1) < 10 or len(t2) < 10:
        return {"valid": False, "pearson": np.nan, "area_diff_pct": np.nan}

    # Rang comú
    t_min = max(t1.min(), t2.min())
    t_max = min(t1.max(), t2.max())

    if window:
        t_min = max(t_min, window[0])
        t_max = min(t_max, window[1])

    if t_max <= t_min:
        return {"valid": False, "pearson": np.nan, "area_diff_pct": np.nan}

    # Interpolar a punts comuns
    t_common = np.linspace(t_min, t_max, 500)
    y1_interp = np.interp(t_common, t1, y1)
    y2_interp = np.interp(t_common, t2, y2)

    # Pearson
    try:
        r, _ = pearsonr(y1_interp, y2_interp)
    except:
        r = np.nan

    # Diferència d'àrea
    area1 = trapezoid(y1_interp, t_common)
    area2 = trapezoid(y2_interp, t_common)

    if max(area1, area2) > 0:
        area_diff_pct = abs(area1 - area2) / max(area1, area2) * 100
    else:
        area_diff_pct = 0.0

    return {
        "valid": True,
        "pearson": float(r) if not np.isnan(r) else np.nan,
        "area_diff_pct": float(area_diff_pct),
        "area_r1": float(area1),
        "area_r2": float(area2)
    }


# Zones temporals per fraccions COLUMN (minuts)
FRACTION_ZONES = {
    "BioP": (0, 18),
    "HS": (18, 23),
    "BB": (23, 30),
    "SB": (30, 40),
    "LMW": (40, 70),
}


def compare_replicas_by_fraction(t1, y1, t2, y2):
    """
    Compara dues rèpliques per cada fracció temporal (COLUMN mode).

    Parameters
    ----------
    t1, y1 : array-like
        Temps i senyal de rèplica 1
    t2, y2 : array-like
        Temps i senyal de rèplica 2

    Returns
    -------
    dict
        {
            "BioP": {"pearson": 0.998, "area_diff_pct": 2.1},
            "HS": {"pearson": 0.983, "area_diff_pct": 15.3},
            ...
        }
    """
    result = {}

    for zone_name, (t_start, t_end) in FRACTION_ZONES.items():
        comparison = compare_replicas(t1, y1, t2, y2, window=(t_start, t_end))

        if comparison.get("valid", False):
            result[zone_name] = {
                "pearson": comparison.get("pearson"),
                "area_diff_pct": comparison.get("area_diff_pct"),
            }
        else:
            result[zone_name] = {
                "pearson": None,
                "area_diff_pct": None,
            }

    return result


def compare_replicas_full(t1, y1, t2, y2,
                          t_dad1=None, dad1=None,
                          t_dad2=None, dad2=None,
                          window=None):
    """
    Comparació completa de dues rèpliques: DOC + DAD.

    Parameters
    ----------
    t1, y1 : array-like
        Temps i senyal DOC de rèplica 1
    t2, y2 : array-like
        Temps i senyal DOC de rèplica 2
    t_dad1, dad1 : array-like, dict, optional
        Temps i dades DAD de rèplica 1. dad1 pot ser:
        - array: un sol wavelength (assumeix A254)
        - dict: {wavelength: array}
    t_dad2, dad2 : array-like, dict, optional
        Temps i dades DAD de rèplica 2
    window : tuple, optional
        (t_min, t_max) per limitar comparació

    Returns
    -------
    dict
        - doc: dict - comparació DOC (pearson, area_diff, etc.)
        - dad_r1: dict - avaluació DAD rèplica 1 (drift, noise, etc.)
        - dad_r2: dict - avaluació DAD rèplica 2
        - doc_dad_r1: dict - correlació DOC-DAD rèplica 1
        - doc_dad_r2: dict - correlació DOC-DAD rèplica 2
        - overall_quality: str ("OK", "WARNING", "POOR")
        - issues: list[str]
    """
    result = {
        "doc": None,
        "dad_r1": None,
        "dad_r2": None,
        "doc_dad_r1": None,
        "doc_dad_r2": None,
        "overall_quality": "OK",
        "issues": []
    }

    quality_order = {"OK": 0, "WARNING": 1, "POOR": 2}

    # === Comparació DOC ===
    result["doc"] = compare_replicas(t1, y1, t2, y2, window=window)

    # === Avaluació DAD ===
    if t_dad1 is not None and dad1 is not None:
        if isinstance(dad1, dict):
            result["dad_r1"] = evaluate_dad_multi(t_dad1, dad1)
        else:
            result["dad_r1"] = evaluate_dad(t_dad1, dad1, wavelength="A254")

        # Correlació DOC-DAD R1
        y_dad1 = dad1 if not isinstance(dad1, dict) else dad1.get("A254", list(dad1.values())[0])
        result["doc_dad_r1"] = compare_doc_dad(t1, y1, t_dad1, y_dad1, window=window)

        if result["dad_r1"].get("issues"):
            result["issues"].extend([f"R1 DAD: {i}" for i in result["dad_r1"]["issues"]])
        if result["doc_dad_r1"].get("issues"):
            result["issues"].extend([f"R1: {i}" for i in result["doc_dad_r1"]["issues"]])

    if t_dad2 is not None and dad2 is not None:
        if isinstance(dad2, dict):
            result["dad_r2"] = evaluate_dad_multi(t_dad2, dad2)
        else:
            result["dad_r2"] = evaluate_dad(t_dad2, dad2, wavelength="A254")

        # Correlació DOC-DAD R2
        y_dad2 = dad2 if not isinstance(dad2, dict) else dad2.get("A254", list(dad2.values())[0])
        result["doc_dad_r2"] = compare_doc_dad(t2, y2, t_dad2, y_dad2, window=window)

        if result["dad_r2"].get("issues"):
            result["issues"].extend([f"R2 DAD: {i}" for i in result["dad_r2"]["issues"]])
        if result["doc_dad_r2"].get("issues"):
            result["issues"].extend([f"R2: {i}" for i in result["doc_dad_r2"]["issues"]])

    # === Qualitat global ===
    qualities = []
    if result["dad_r1"]:
        qualities.append(result["dad_r1"].get("quality", "OK"))
    if result["dad_r2"]:
        qualities.append(result["dad_r2"].get("quality", "OK"))
    if result["doc_dad_r1"]:
        qualities.append(result["doc_dad_r1"].get("quality", "OK"))
    if result["doc_dad_r2"]:
        qualities.append(result["doc_dad_r2"].get("quality", "OK"))

    for q in qualities:
        if quality_order.get(q, 0) > quality_order.get(result["overall_quality"], 0):
            result["overall_quality"] = q

    return result


# =============================================================================
# SELECCIÓ DE MILLOR RÈPLICA
# =============================================================================

def select_best_replica(eval1, eval2, method="COLUMN", comparison=None,
                        dad_eval1=None, dad_eval2=None):
    """
    Selecciona la millor rèplica basant-se en l'avaluació.

    Criteris per mètode:
    - COLUMN:
      1. Anomalies (Batman > Timeout > IRR) - disqualificadores
      2. DAD quality (si una té POOR i l'altra no)
      3. SNR (si diferència > 1.5x)
      4. Pearson entre rèpliques (warning si < 0.990)
      5. Diferència d'àrea (warning si > 15%, check si > 30%)
      6. DAD drift (preferir menys deriva)
      7. Alçada (tiebreaker final)

    - BP:
      1. Anomalies (Batman)
      2. R² status (VALID > CHECK > INVALID)
      3. R² valor (bi-gaussiana, major = millor)
      4. SNR (tiebreaker)
      + Pearson, àrea i DAD afegeixen warnings

    Parameters
    ----------
    eval1, eval2 : dict or None
        Resultat de evaluate_replica() per cada rèplica (DOC)
    method : str
        "COLUMN" o "BP"
    comparison : dict, optional
        Resultat de compare_replicas() o compare_replicas_full()
    dad_eval1, dad_eval2 : dict, optional
        Resultat de evaluate_dad() o evaluate_dad_multi() per cada rèplica

    Returns
    -------
    dict
        - best: "R1" o "R2" o None
        - reason: str (motiu de la selecció)
        - warning: str o None (avisos)
        - confidence: float (0-1, confiança en la selecció)
    """
    result = {
        "best": None,
        "reason": "",
        "warning": None,
        "confidence": 1.0
    }

    # === CAS: Cap rèplica disponible ===
    if eval1 is None and eval2 is None:
        result["warning"] = "CAP RÈPLICA DISPONIBLE"
        result["confidence"] = 0.0
        return result

    # === CAS: Només una rèplica ===
    if eval1 is None:
        result["best"] = "R2"
        result["reason"] = "Única rèplica disponible"
        result["confidence"] = 0.5
        if eval2.get("batman") or eval2.get("timeout"):
            result["warning"] = "WARNING: Única rèplica amb anomalies"
        return result

    if eval2 is None:
        result["best"] = "R1"
        result["reason"] = "Única rèplica disponible"
        result["confidence"] = 0.5
        if eval1.get("batman") or eval1.get("timeout"):
            result["warning"] = "WARNING: Única rèplica amb anomalies"
        return result

    # === CAS: Cap vàlida ===
    if not eval1.get("valid") and not eval2.get("valid"):
        result["warning"] = "CAP RÈPLICA VÀLIDA"
        result["confidence"] = 0.0
        return result

    if not eval1.get("valid"):
        result["best"] = "R2"
        result["reason"] = f"R1 invàlida: {eval1.get('reason', '?')}"
        return result

    if not eval2.get("valid"):
        result["best"] = "R1"
        result["reason"] = f"R2 invàlida: {eval2.get('reason', '?')}"
        return result

    # === CAS: Ambdues vàlides - aplicar criteris segons mètode ===

    if method == "BP":
        return _select_best_bp(eval1, eval2, comparison, dad_eval1, dad_eval2)
    else:
        return _select_best_column(eval1, eval2, comparison, dad_eval1, dad_eval2)


def _select_best_column(eval1, eval2, comparison=None, dad_eval1=None, dad_eval2=None):
    """
    Selecció per COLUMN: Anomalies → DAD → SNR → Àrea/Pearson → Alçada.

    Criteris ordenats per prioritat:
    1. Anomalies disqualificadores (Batman, Timeout, IRR)
    2. DAD quality (si una és POOR i l'altra no)
    3. SNR (si diferència significativa > 1.5x)
    4. Pearson i diferència d'àrea (warnings)
    5. DAD drift (preferir menys deriva)
    6. Alçada (tiebreaker final)
    """
    result = {
        "best": None,
        "reason": "",
        "warning": None,
        "confidence": 1.0
    }

    # Extreure flags anomalies
    r1_batman = eval1.get("batman", False)
    r2_batman = eval2.get("batman", False)
    r1_timeout = eval1.get("timeout", False)
    r2_timeout = eval2.get("timeout", False)
    r1_irr = eval1.get("irr", False)
    r2_irr = eval2.get("irr", False)

    # Extreure mètriques DOC
    snr1 = eval1.get("snr", 0) or 0
    snr2 = eval2.get("snr", 0) or 0
    h1 = eval1.get("height", 0) or 0
    h2 = eval2.get("height", 0) or 0

    # Extreure mètriques DAD
    dad1_quality = dad_eval1.get("quality", "OK") if dad_eval1 else "OK"
    dad2_quality = dad_eval2.get("quality", "OK") if dad_eval2 else "OK"
    dad1_drift = abs(dad_eval1.get("drift", 0)) if dad_eval1 else 0
    dad2_drift = abs(dad_eval2.get("drift", 0)) if dad_eval2 else 0

    reasons = []
    warnings = []

    # === CRITERI 1: Anomalies DOC disqualificadores ===

    # 1a. Batman (prioritat màxima)
    if r1_batman and not r2_batman:
        result["best"] = "R2"
        result["reason"] = "R1 té BATMAN"
        return result

    if r2_batman and not r1_batman:
        result["best"] = "R1"
        result["reason"] = "R2 té BATMAN"
        return result

    if r1_batman and r2_batman:
        warnings.append("Ambdues rèpliques amb BATMAN")

    # 1b. Timeout
    if r1_timeout and not r2_timeout:
        result["best"] = "R2"
        result["reason"] = "R1 té TIMEOUT"
        return result

    if r2_timeout and not r1_timeout:
        result["best"] = "R1"
        result["reason"] = "R2 té TIMEOUT"
        return result

    if r1_timeout and r2_timeout:
        warnings.append("Ambdues rèpliques amb TIMEOUT")

    # 1c. Irregularitat
    if r1_irr and not r2_irr:
        result["best"] = "R2"
        result["reason"] = "R1 és irregular"
        return result

    if r2_irr and not r1_irr:
        result["best"] = "R1"
        result["reason"] = "R2 és irregular"
        return result

    # === CRITERI 2: DAD Quality (si una és POOR) ===
    if dad_eval1 and dad_eval2:
        if dad1_quality == "POOR" and dad2_quality != "POOR":
            result["best"] = "R2"
            result["reason"] = "R1 DAD qualitat POOR"
            _add_dad_warnings(result, dad_eval1, dad_eval2, warnings)
            return result

        if dad2_quality == "POOR" and dad1_quality != "POOR":
            result["best"] = "R1"
            result["reason"] = "R2 DAD qualitat POOR"
            _add_dad_warnings(result, dad_eval1, dad_eval2, warnings)
            return result

        # Afegir warnings DAD
        _add_dad_warnings(result, dad_eval1, dad_eval2, warnings)

    # === CRITERI 3: SNR (si diferència significativa) ===
    if snr1 > 0 and snr2 > 0:
        snr_ratio = max(snr1, snr2) / min(snr1, snr2)
        if snr_ratio >= SNR_RATIO_THRESHOLD:
            if snr1 > snr2:
                result["best"] = "R1"
                result["reason"] = f"Millor SNR (R1:{snr1:.1f} vs R2:{snr2:.1f})"
                _add_comparison_info(result, comparison, reasons)
                if warnings:
                    result["warning"] = "WARNING: " + "; ".join(warnings)
                return result
            else:
                result["best"] = "R2"
                result["reason"] = f"Millor SNR (R2:{snr2:.1f} vs R1:{snr1:.1f})"
                _add_comparison_info(result, comparison, reasons)
                if warnings:
                    result["warning"] = "WARNING: " + "; ".join(warnings)
                return result

    # === CRITERI 4: Pearson i diferència d'àrea (warnings) ===
    if comparison and comparison.get("valid"):
        pearson = comparison.get("pearson")
        area_diff = comparison.get("area_diff_pct", 0)

        # Pearson molt baix - problema greu
        if pearson is not None and not np.isnan(pearson):
            if pearson < PEARSON_WARNING:
                warnings.append(f"Pearson BAIX ({pearson:.3f})")
                result["confidence"] *= 0.6
            elif pearson < PEARSON_THRESHOLD:
                reasons.append(f"Pearson={pearson:.3f}")
                result["confidence"] *= 0.85

        # Diferència d'àrea alta
        if area_diff >= AREA_DIFF_CRITICAL:
            warnings.append(f"Àrea DIFF {area_diff:.1f}%")
            result["confidence"] *= 0.7
        elif area_diff >= AREA_DIFF_WARNING:
            reasons.append(f"Àrea diff={area_diff:.1f}%")
            result["confidence"] *= 0.9

    # === CRITERI 5: DAD drift (preferir menys deriva) ===
    if dad_eval1 and dad_eval2 and dad1_drift > 0 and dad2_drift > 0:
        drift_ratio = max(dad1_drift, dad2_drift) / max(min(dad1_drift, dad2_drift), 0.01)
        if drift_ratio >= 2.0:  # Una té 2x més deriva
            if dad1_drift < dad2_drift:
                reasons.append(f"DAD drift R1:{dad1_drift:.2f} < R2:{dad2_drift:.2f}")
            else:
                reasons.append(f"DAD drift R2:{dad2_drift:.2f} < R1:{dad1_drift:.2f}")

    # === CRITERI 6: SNR secundari (si no era prou diferent) ===
    if snr1 > snr2 * 1.1:
        reasons.append(f"SNR R1:{snr1:.1f} > R2:{snr2:.1f}")
    elif snr2 > snr1 * 1.1:
        reasons.append(f"SNR R2:{snr2:.1f} > R1:{snr1:.1f}")

    # === CRITERI 7: Alçada (tiebreaker final) ===
    # Considerar també DAD drift si SNR similar
    if dad_eval1 and dad_eval2 and dad1_drift != dad2_drift:
        if dad1_drift < dad2_drift and h1 >= h2 * 0.95:
            result["best"] = "R1"
            reasons.append(f"Menys deriva DAD")
        elif dad2_drift < dad1_drift and h2 >= h1 * 0.95:
            result["best"] = "R2"
            reasons.append(f"Menys deriva DAD")
        elif h1 >= h2:
            result["best"] = "R1"
            reasons.append(f"Alçada R1:{h1:.1f}")
        else:
            result["best"] = "R2"
            reasons.append(f"Alçada R2:{h2:.1f}")
    elif h1 >= h2:
        result["best"] = "R1"
        reasons.append(f"Alçada R1:{h1:.1f}")
    else:
        result["best"] = "R2"
        reasons.append(f"Alçada R2:{h2:.1f}")

    # Construir raó i warning
    if warnings:
        result["warning"] = "WARNING: " + "; ".join(warnings)
    result["reason"] = "; ".join(reasons) if reasons else f"Alçada similar"

    return result


def _add_dad_warnings(result, dad_eval1, dad_eval2, warnings):
    """Afegeix warnings de DAD al resultat."""
    if not dad_eval1 or not dad_eval2:
        return

    # Recollir issues de DAD
    if dad_eval1.get("issues"):
        for issue in dad_eval1["issues"]:
            warnings.append(f"R1 DAD: {issue}")

    if dad_eval2.get("issues"):
        for issue in dad_eval2["issues"]:
            warnings.append(f"R2 DAD: {issue}")

    # Reduir confiança segons qualitat
    if dad_eval1.get("quality") == "WARNING" or dad_eval2.get("quality") == "WARNING":
        result["confidence"] *= 0.9
    if dad_eval1.get("quality") == "POOR" or dad_eval2.get("quality") == "POOR":
        result["confidence"] *= 0.7


def _add_comparison_info(result, comparison, reasons):
    """Afegeix info de comparació (Pearson, àrea) al resultat."""
    if comparison and comparison.get("valid"):
        pearson = comparison.get("pearson")
        area_diff = comparison.get("area_diff_pct", 0)

        if pearson is not None and not np.isnan(pearson):
            if pearson < PEARSON_WARNING:
                if result.get("warning"):
                    result["warning"] += f"; Pearson={pearson:.3f}"
                else:
                    result["warning"] = f"Pearson BAIX ({pearson:.3f})"
                result["confidence"] *= 0.7

        if area_diff >= AREA_DIFF_WARNING:
            reasons.append(f"Àrea diff={area_diff:.1f}%")


def _select_best_bp(eval1, eval2, comparison=None, dad_eval1=None, dad_eval2=None):
    """
    Selecció per BP: Batman → R² → SNR.

    Criteris ordenats per prioritat:
    1. Anomalies (Batman)
    2. R² status (VALID > CHECK > INVALID)
    3. R² valor (bi-gaussiana)
    4. SNR (tiebreaker)
    + DAD warnings (no afecten decisió, només confiança)
    """
    result = {
        "best": None,
        "reason": "",
        "warning": None,
        "confidence": 1.0
    }

    # Extreure mètriques
    r2_1 = eval1.get("r2", 0) or 0
    r2_2 = eval2.get("r2", 0) or 0
    status_1 = eval1.get("r2_status", "INVALID")
    status_2 = eval2.get("r2_status", "INVALID")
    snr_1 = eval1.get("snr", 0) or 0
    snr_2 = eval2.get("snr", 0) or 0

    # Anomalies
    r1_batman = eval1.get("batman", False)
    r2_batman = eval2.get("batman", False)

    reasons = []
    warnings = []

    # Afegir DAD warnings si disponibles
    if dad_eval1 or dad_eval2:
        _add_dad_warnings(result, dad_eval1, dad_eval2, warnings)

    # === CRITERI 1: Batman (prioritat màxima) ===
    if r1_batman and not r2_batman:
        result["best"] = "R2"
        result["reason"] = "R1 té BATMAN"
        return result

    if r2_batman and not r1_batman:
        result["best"] = "R1"
        result["reason"] = "R2 té BATMAN"
        return result

    if r1_batman and r2_batman:
        warnings.append("Ambdues rèpliques amb BATMAN")

    # === CRITERI 2: R² status ===
    status_order = {"VALID": 0, "CHECK": 1, "INVALID": 2}
    order_1 = status_order.get(status_1, 2)
    order_2 = status_order.get(status_2, 2)

    if order_1 < order_2:
        result["best"] = "R1"
        result["reason"] = f"R1 {status_1} vs R2 {status_2}"
        _add_bp_comparison_info(result, comparison, warnings)
        return result

    if order_2 < order_1:
        result["best"] = "R2"
        result["reason"] = f"R2 {status_2} vs R1 {status_1}"
        _add_bp_comparison_info(result, comparison, warnings)
        return result

    # Ambdues INVALID?
    if status_1 == "INVALID" and status_2 == "INVALID":
        warnings.append("Ambdues rèpliques INVALID")
        result["confidence"] = 0.3

    # === CRITERI 3: R² valor (arrodonit a 3 decimals) ===
    r2_1_round = round(r2_1, 3)
    r2_2_round = round(r2_2, 3)

    if r2_1_round > r2_2_round:
        result["best"] = "R1"
        reasons.append(f"R²: R1={r2_1:.4f} > R2={r2_2:.4f}")
    elif r2_2_round > r2_1_round:
        result["best"] = "R2"
        reasons.append(f"R²: R2={r2_2:.4f} > R1={r2_1:.4f}")
    else:
        # === CRITERI 4: SNR (tiebreaker) ===
        if snr_1 >= snr_2:
            result["best"] = "R1"
            reasons.append(f"SNR: R1={snr_1:.1f} >= R2={snr_2:.1f}")
        else:
            result["best"] = "R2"
            reasons.append(f"SNR: R2={snr_2:.1f} > R1={snr_1:.1f}")

    # Afegir info comparació (Pearson, àrea)
    _add_bp_comparison_info(result, comparison, warnings)

    if warnings:
        result["warning"] = "WARNING: " + "; ".join(warnings)
    result["reason"] = "; ".join(reasons) if reasons else f"R²={r2_1:.3f}"

    return result


def _add_bp_comparison_info(result, comparison, warnings):
    """Afegeix info de comparació per BP (Pearson, àrea)."""
    if comparison and comparison.get("valid"):
        pearson = comparison.get("pearson")
        area_diff = comparison.get("area_diff_pct", 0)

        if pearson is not None and not np.isnan(pearson):
            if pearson < PEARSON_WARNING:
                warnings.append(f"Pearson={pearson:.3f}")
                result["confidence"] *= 0.7
            elif pearson < PEARSON_THRESHOLD:
                result["confidence"] *= 0.85

        if area_diff >= AREA_DIFF_CRITICAL:
            warnings.append(f"Àrea diff={area_diff:.1f}%")
            result["confidence"] *= 0.7


# =============================================================================
# FUNCIÓ COMBINADA (per conveniència)
# =============================================================================

def evaluate_and_select(t1, y1, t2, y2, method="COLUMN"):
    """
    Avalua dues rèpliques i selecciona la millor.

    Funció de conveniència que combina evaluate_replica,
    compare_replicas i select_best_replica.

    Parameters
    ----------
    t1, y1 : array-like or None
        Temps i senyal de rèplica 1 (None si no disponible)
    t2, y2 : array-like or None
        Temps i senyal de rèplica 2 (None si no disponible)
    method : str
        "COLUMN" o "BP"

    Returns
    -------
    dict
        - best: "R1" o "R2" o None
        - reason: str
        - warning: str o None
        - eval_r1: dict (avaluació R1)
        - eval_r2: dict (avaluació R2)
        - comparison: dict (comparació entre rèpliques)
    """
    # Avaluar cada rèplica
    eval1 = None
    eval2 = None

    if t1 is not None and y1 is not None and len(t1) > 0:
        eval1 = evaluate_replica(t1, y1, method=method)

    if t2 is not None and y2 is not None and len(t2) > 0:
        eval2 = evaluate_replica(t2, y2, method=method)

    # Comparar si ambdues disponibles
    comparison = None
    if eval1 is not None and eval2 is not None:
        if eval1.get("valid") and eval2.get("valid"):
            comparison = compare_replicas(t1, y1, t2, y2)

    # Seleccionar
    selection = select_best_replica(eval1, eval2, method=method, comparison=comparison)

    return {
        "best": selection["best"],
        "reason": selection["reason"],
        "warning": selection.get("warning"),
        "confidence": selection.get("confidence", 1.0),
        "eval_r1": eval1,
        "eval_r2": eval2,
        "comparison": comparison
    }


# =============================================================================
# TEST BÀSIC
# =============================================================================

if __name__ == "__main__":
    # Test amb dades sintètiques
    print("=== Test hpsec_replica.py ===\n")

    # Crear pic sintètic
    t = np.linspace(0, 70, 1000)
    y1 = 100 * np.exp(-((t - 20)**2) / (2 * 2**2)) + 5  # Pic a t=20
    y2 = 90 * np.exp(-((t - 20)**2) / (2 * 2**2)) + 5   # Pic lleugerament més baix

    print("Test COLUMN:")
    result = evaluate_and_select(t, y1, t, y2, method="COLUMN")
    print(f"  Best: R{result['best']}")
    print(f"  Reason: {result['reason']}")
    print(f"  Warning: {result['warning']}")
    if result['comparison']:
        print(f"  Pearson: {result['comparison']['pearson']:.4f}")

    print("\nTest BP:")
    result_bp = evaluate_and_select(t, y1, t, y2, method="BP")
    print(f"  Best: R{result_bp['best']}")
    print(f"  Reason: {result_bp['reason']}")

    print("\nTest amb una sola rèplica:")
    result_single = evaluate_and_select(t, y1, None, None, method="COLUMN")
    print(f"  Best: R{result_single['best']}")
    print(f"  Reason: {result_single['reason']}")
    print(f"  Warning: {result_single['warning']}")

    print("\n=== Tests completats ===")
