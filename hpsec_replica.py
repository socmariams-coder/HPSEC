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
- COLUMN: Anomalies (Batman, IRR, Timeout) → Alçada
- BP: R² (qualitat ajust) → SNR

v1.0 - 2026-01-26 - Creació inicial
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

        if fit_result.get("valid", False):
            r2 = fit_result.get("r2", 0)
            asymmetry = fit_result.get("asymmetry", 1.0)

            # Determinar status
            if r2 >= THRESH_R2_VALID:
                r2_status = "VALID"
            elif r2 >= THRESH_R2_CHECK:
                r2_status = "CHECK"
            else:
                r2_status = "INVALID"
        else:
            r2 = 0.0
            r2_status = "INVALID"

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


# =============================================================================
# SELECCIÓ DE MILLOR RÈPLICA
# =============================================================================

def select_best_replica(eval1, eval2, method="COLUMN", comparison=None):
    """
    Selecciona la millor rèplica basant-se en l'avaluació.

    Criteris per mètode:
    - COLUMN:
      1. Anomalies (Batman > Timeout > IRR) - si una en té i l'altra no
      2. Alçada del pic principal

    - BP:
      1. R² status (VALID > CHECK > INVALID)
      2. R² valor (major = millor)
      3. SNR (tiebreaker)

    Parameters
    ----------
    eval1, eval2 : dict or None
        Resultat de evaluate_replica() per cada rèplica
    method : str
        "COLUMN" o "BP"
    comparison : dict, optional
        Resultat de compare_replicas() (Pearson, àrea)

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
        return _select_best_bp(eval1, eval2, comparison)
    else:
        return _select_best_column(eval1, eval2, comparison)


def _select_best_column(eval1, eval2, comparison=None):
    """Selecció per COLUMN: Anomalies → Alçada."""
    result = {
        "best": None,
        "reason": "",
        "warning": None,
        "confidence": 1.0
    }

    # Extreure flags
    r1_batman = eval1.get("batman", False)
    r2_batman = eval2.get("batman", False)
    r1_timeout = eval1.get("timeout", False)
    r2_timeout = eval2.get("timeout", False)
    r1_irr = eval1.get("irr", False)
    r2_irr = eval2.get("irr", False)

    reasons = []

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
        result["warning"] = "WARNING: Ambdues rèpliques amb BATMAN"
        reasons.append("Ambdues BATMAN")

    # === CRITERI 2: Timeout ===
    if r1_timeout and not r2_timeout:
        result["best"] = "R2"
        result["reason"] = "R1 té TIMEOUT"
        return result

    if r2_timeout and not r1_timeout:
        result["best"] = "R1"
        result["reason"] = "R2 té TIMEOUT"
        return result

    # === CRITERI 3: Irregularitat ===
    if r1_irr and not r2_irr:
        result["best"] = "R2"
        result["reason"] = "R1 és irregular"
        return result

    if r2_irr and not r1_irr:
        result["best"] = "R1"
        result["reason"] = "R2 és irregular"
        return result

    # === CRITERI 4: Alçada (tiebreaker) ===
    h1 = eval1.get("height", 0)
    h2 = eval2.get("height", 0)

    if h1 >= h2:
        result["best"] = "R1"
        reasons.append(f"Més alçada (R1:{h1:.1f} vs R2:{h2:.1f})")
    else:
        result["best"] = "R2"
        reasons.append(f"Més alçada (R2:{h2:.1f} vs R1:{h1:.1f})")

    # Afegir info Pearson si disponible
    if comparison and comparison.get("valid"):
        pearson = comparison.get("pearson")
        if pearson is not None and not np.isnan(pearson):
            if pearson < PEARSON_THRESHOLD:
                reasons.append(f"Pearson={pearson:.3f} (baix)")
                result["confidence"] *= 0.8
            else:
                reasons.append(f"Pearson={pearson:.3f}")

    result["reason"] = "; ".join(reasons)
    return result


def _select_best_bp(eval1, eval2, comparison=None):
    """Selecció per BP: R² → SNR."""
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
    snr_1 = eval1.get("snr", 0)
    snr_2 = eval2.get("snr", 0)

    # Batman també és important per BP
    r1_batman = eval1.get("batman", False)
    r2_batman = eval2.get("batman", False)

    reasons = []

    # === CRITERI 0: Batman (prioritat màxima també per BP) ===
    if r1_batman and not r2_batman:
        result["best"] = "R2"
        result["reason"] = "R1 té BATMAN"
        return result

    if r2_batman and not r1_batman:
        result["best"] = "R1"
        result["reason"] = "R2 té BATMAN"
        return result

    # === CRITERI 1: R² status ===
    status_order = {"VALID": 0, "CHECK": 1, "INVALID": 2}
    order_1 = status_order.get(status_1, 2)
    order_2 = status_order.get(status_2, 2)

    if order_1 < order_2:
        result["best"] = "R1"
        result["reason"] = f"R1 {status_1} vs R2 {status_2}"
        return result

    if order_2 < order_1:
        result["best"] = "R2"
        result["reason"] = f"R2 {status_2} vs R1 {status_1}"
        return result

    # Ambdues INVALID?
    if status_1 == "INVALID" and status_2 == "INVALID":
        result["warning"] = "WARNING: Ambdues rèpliques INVALID"
        result["confidence"] = 0.3

    # === CRITERI 2: R² valor (arrodonit a 3 decimals) ===
    r2_1_round = round(r2_1, 3)
    r2_2_round = round(r2_2, 3)

    if r2_1_round > r2_2_round:
        result["best"] = "R1"
        reasons.append(f"R²: R1={r2_1:.4f} > R2={r2_2:.4f}")
    elif r2_2_round > r2_1_round:
        result["best"] = "R2"
        reasons.append(f"R²: R2={r2_2:.4f} > R1={r2_1:.4f}")
    else:
        # === CRITERI 3: SNR (tiebreaker) ===
        if snr_1 >= snr_2:
            result["best"] = "R1"
            reasons.append(f"SNR: R1={snr_1:.1f} >= R2={snr_2:.1f}")
        else:
            result["best"] = "R2"
            reasons.append(f"SNR: R2={snr_2:.1f} > R1={snr_1:.1f}")

    # Afegir info Pearson si disponible
    if comparison and comparison.get("valid"):
        pearson = comparison.get("pearson")
        if pearson is not None and not np.isnan(pearson):
            if pearson < PEARSON_THRESHOLD:
                reasons.append(f"Pearson={pearson:.3f} (baix)")
                result["confidence"] *= 0.8

    result["reason"] = "; ".join(reasons) if reasons else f"R²={r2_1:.3f}"
    return result


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
