# -*- coding: utf-8 -*-
"""
HPSEC Core - Funcions compartides per anàlisi de cromatogrames
==============================================================

Mòdul centralitzat amb TOTES les funcions de detecció i processament.
Single Source of Truth per evitar duplicació de codi.

Funcions principals:
- Bi-Gaussiana: fit_bigaussian, bigaussian, check_asymmetry
- Detecció cim irregular: detect_irregular_top, detect_peak_anomaly, calc_top_smoothness
- Reparació: repair_with_parabola, find_tangents_and_anchors
- Timeout TOC: detect_timeout, format_timeout_status (MILLOR MÈTODE: dt intervals)
- Detecció pics: detect_main_peak, detect_all_peaks
- Integració: integrate_chromatogram (mode='full'|'main_peak')
- Utilitats: calc_snr, calc_peak_area, calc_pearson, apply_smoothing
- Mètriques pics: calculate_fwhm, calculate_symmetry

v1.2 - 2026-02-02: Afegides calculate_fwhm i calculate_symmetry (migrades de calibrate)
v1.1 - 2026-01-26: Afegides funcions timeout i detecció pics (migrades)
v1.0 - 2026-01-22: Versió inicial
"""

import logging
import numpy as np
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import linregress, pearsonr
from scipy.optimize import curve_fit
from scipy.integrate import trapezoid

logger = logging.getLogger(__name__)


# =============================================================================
# LLINDARS GLOBALS
# =============================================================================

# R² thresholds for bi-Gaussian fit
THRESH_R2_VALID = 0.987      # R² >= 0.987 → VALID
THRESH_R2_CHECK = 0.980      # 0.980 <= R² < 0.987 → CHECK

# Repair thresholds
REPAIR_MIN_R2 = 0.980        # Don't repair if R² < this (garbage peak)
REPAIR_FACTOR = 0.85         # Tangent height correction factor

# Bi-Gaussian fit constants
GAUSSIAN_HW_TO_SIGMA = 1.177  # Half-width at half-max to sigma: sqrt(2*ln(2))
MIN_SIGMA_GUESS = 0.05        # Minimum sigma initial guess (minutes)
REPAIR_FACTOR_MIN = 0.5       # Tangent repair: minimum height ratio (real/tangent)
REPAIR_FACTOR_MAX = 1.2       # Tangent repair: maximum height ratio (real/tangent)

# Pearson correlation
PEARSON_INTERP_POINTS = 500   # Grid points for interpolated Pearson calculation

# Asymmetry thresholds
ASYM_MIN = 0.33              # sigma_right/sigma_left minimum
ASYM_MAX = 3.0               # sigma_right/sigma_left maximum

# Irregular top detection (jagged/batman artifact)
MIN_VALLEY_DEPTH = 0.01      # 1% of peak height to detect irregular top

# SNR
THRESH_SNR = 10.0            # Minimum acceptable SNR


# =============================================================================
# SMOOTHING
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
# DOWNSAMPLING
# =============================================================================

# Cadència objectiu per DOC: ~4s per punt (Sievers M9e)
DOC_TARGET_DT_MIN = 0.0667   # 4 segons en minuts


def downsample_to_cadence(t, y, target_dt=DOC_TARGET_DT_MIN):
    """
    Redueix la cadència d'un senyal mitjançant bin-average.

    Si la cadència original (dt_median) ja és >= target_dt * 0.8, retorna
    les dades originals sense modificar.

    Mètode: bins uniformes de mida target_dt. Cada bin calcula la mitjana
    dels punts que hi cauen. Això preserva l'àrea integrada i la forma
    del pic millor que interpolació o subsampling directe.

    Args:
        t: Array de temps (minuts)
        y: Array de senyal
        target_dt: Cadència objectiu en minuts (defecte: 0.0667 = 4s)

    Returns:
        (t_ds, y_ds): Arrays downsampled. Si no cal downsample, retorna (t, y).
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(t) < 3:
        return t, y

    dt_median = float(np.median(np.diff(t)))

    # No cal downsample si ja estem a la cadència objectiu (±20%)
    if dt_median >= target_dt * 0.8:
        return t, y

    # Crear bins uniformes
    t_min, t_max = t[0], t[-1]
    n_bins = max(1, int(np.ceil((t_max - t_min) / target_dt)))
    bin_edges = np.linspace(t_min, t_min + n_bins * target_dt, n_bins + 1)

    # Assignar cada punt al bin corresponent
    bin_idx = np.digitize(t, bin_edges) - 1  # 0-indexed
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    # Calcular mitjana per bin (bins buits s'ometen)
    t_ds = []
    y_ds = []
    for i in range(n_bins):
        mask = bin_idx == i
        if np.any(mask):
            t_ds.append(float(np.mean(t[mask])))
            y_ds.append(float(np.mean(y[mask])))

    t_ds = np.array(t_ds)
    y_ds = np.array(y_ds)

    ratio = len(t) / max(len(t_ds), 1)
    logger.debug(f"downsample_to_cadence: {len(t)} → {len(t_ds)} punts "
                 f"(ratio {ratio:.1f}x, dt {dt_median:.4f} → {target_dt:.4f} min)")

    return t_ds, y_ds


# =============================================================================
# BI-GAUSSIAN FUNCTIONS
# =============================================================================

def bigaussian(t, amplitude, mu, sigma_left, sigma_right, baseline):
    """
    Bi-Gaussian (asymmetric Gaussian) function.

    Uses sigma_left for t <= mu, sigma_right for t > mu.
    This captures chromatographic tailing (sigma_right > sigma_left).

    Parameters:
        t: time array
        amplitude: peak height above baseline
        mu: peak center position
        sigma_left: width parameter for left side
        sigma_right: width parameter for right side
        baseline: baseline offset

    Returns:
        y values for the bi-Gaussian
    """
    t = np.asarray(t)
    y = np.where(
        t <= mu,
        amplitude * np.exp(-(t - mu)**2 / (2 * sigma_left**2)),
        amplitude * np.exp(-(t - mu)**2 / (2 * sigma_right**2))
    )
    return y + baseline


def fit_bigaussian(t, y, peak_idx, left_idx, right_idx, r2_top_pct=0):
    """
    Fit a Bi-Gaussian (asymmetric) to a peak.

    Parameters:
        t, y: time and signal arrays (full chromatogram)
        peak_idx: index of peak maximum
        left_idx, right_idx: indices of peak boundaries
        r2_top_pct: percentage of peak height for R² calculation (0-100)
                    If > 0, R² is calculated only on points ABOVE this % of height.
                    Fit is always done on full segment (from baseline).
                    Ex: r2_top_pct=66 → R² calculated on top 34% of peak

    Returns:
        dict with r2, status, parameters, asymmetry, and visualization data
    """
    # Extract peak segment
    t_seg = np.asarray(t[left_idx:right_idx+1], dtype=float)
    y_seg = np.asarray(y[left_idx:right_idx+1], dtype=float)

    if len(t_seg) < 5:
        return {"r2": 0.0, "error": "Segment too short", "status": "INVALID"}

    # Baseline and height (from full segment)
    baseline_val = float(np.min(y_seg))
    peak_val = float(y[peak_idx])
    peak_height = peak_val - baseline_val

    if peak_height <= 0:
        return {"r2": 0.0, "error": "Zero height", "status": "INVALID"}

    # Always fit on FULL segment (from baseline)
    t_fit = t_seg
    y_fit = y_seg

    # Estimate sigmas from half-widths
    t_peak = float(t[peak_idx])
    half_height = baseline_val + peak_height * 0.5

    # Left half-width
    left_mask = (t_seg <= t_peak) & (y_seg >= half_height)
    if np.sum(left_mask) >= 1:
        t_left_half = t_seg[left_mask]
        hw_left = t_peak - t_left_half[0]
        sigma_left_guess = hw_left / GAUSSIAN_HW_TO_SIGMA
    else:
        sigma_left_guess = (t_peak - t_seg[0]) / 3

    # Right half-width
    right_mask = (t_seg >= t_peak) & (y_seg >= half_height)
    if np.sum(right_mask) >= 1:
        t_right_half = t_seg[right_mask]
        hw_right = t_right_half[-1] - t_peak
        sigma_right_guess = hw_right / GAUSSIAN_HW_TO_SIGMA
    else:
        sigma_right_guess = (t_seg[-1] - t_peak) / 3

    sigma_left_guess = max(MIN_SIGMA_GUESS, sigma_left_guess)
    sigma_right_guess = max(MIN_SIGMA_GUESS, sigma_right_guess)

    try:
        popt, _ = curve_fit(
            bigaussian, t_fit, y_fit,
            p0=[peak_height, t_peak, sigma_left_guess, sigma_right_guess, baseline_val],
            bounds=(
                [peak_height * 0.5, t_fit[0], 0.01, 0.01, min(0, baseline_val - abs(baseline_val) * 0.5)],
                [peak_height * 1.5, t_fit[-1], 10.0, 10.0, max(0, baseline_val + peak_height * 0.3)]
            ),
            maxfev=5000
        )

        # Calculate R² - optionally only on top portion
        y_pred_full = bigaussian(t_seg, *popt)

        if r2_top_pct > 0:
            # R² only on TOP portion (above threshold)
            r2_threshold = baseline_val + peak_height * (r2_top_pct / 100.0)
            top_mask = y_seg >= r2_threshold
            if np.sum(top_mask) >= 3:
                y_top = y_seg[top_mask]
                y_pred_top = y_pred_full[top_mask]
                ss_res = np.sum((y_top - y_pred_top)**2)
                ss_tot = np.sum((y_top - np.mean(y_top))**2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            else:
                r2 = 0.0
        else:
            # R² on full segment
            ss_res = np.sum((y_seg - y_pred_full)**2)
            ss_tot = np.sum((y_seg - np.mean(y_seg))**2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        r2 = max(0.0, r2)

        # Determine status
        if r2 >= THRESH_R2_VALID:
            status = "VALID"
        elif r2 >= THRESH_R2_CHECK:
            status = "CHECK"
        else:
            status = "INVALID"

        # Generate curve for visualization (full segment for context)
        t_plot = np.linspace(t_seg[0], t_seg[-1], 100)
        y_plot = bigaussian(t_plot, *popt)

        # Asymmetry ratio
        asymmetry = popt[3] / popt[2] if popt[2] > 0 else 1.0

        return {
            "r2": r2,
            "r2_top_pct": r2_top_pct,
            "status": status,
            "amplitude": popt[0],
            "mu": popt[1],
            "sigma_left": popt[2],
            "sigma_right": popt[3],
            "baseline_fit": popt[4],
            "asymmetry": asymmetry,
            "t_fit": t_plot,
            "y_fit": y_plot
        }

    except Exception as e:
        return {"r2": 0.0, "error": str(e), "status": "INVALID"}


def check_asymmetry(asymmetry):
    """
    Check if asymmetry is within acceptable range.

    Returns:
        bool: True if OK, False if extreme (detector error)
    """
    return ASYM_MIN <= asymmetry <= ASYM_MAX


# =============================================================================
# PEAK CLIPPING / SATURATION DETECTION (Gaussian fit)
# =============================================================================

def detect_peak_clipping(t, y, peak_idx=None):
    """
    Detect signal clipping/saturation by plateau/FWHM ratio.

    A normal chromatographic peak (Gaussian-like) has a very narrow top:
    the plateau at 98% of max spans ~0.17× the FWHM.
    A clipped/saturated peak has a flat top: plateau/FWHM >> 0.17.

    Threshold: plateau_ratio > 0.40 = saturated (~15% clipping).

    This method is independent of any sensitivity parameter — it only depends
    on the intrinsic shape of the signal.

    Args:
        t: Array de temps (minuts)
        y: Array de senyal (raw or baseline-subtracted)
        peak_idx: Índex del pic (si None, usa argmax)

    Returns:
        dict:
            is_saturated: bool
            plateau_ratio: float — plateau_width / FWHM (Gaussian ≈ 0.17)
            plateau_width_pts: int — punts consecutius prop del màxim (>98%)
            fwhm_pts: int — amplada a mitja alçada (punts)
            y_max_observed: float — màxim net (baseline-subtracted)
    """
    result = {
        "is_saturated": False,
        "plateau_ratio": 0.0,
        "plateau_width_pts": 1,
        "fwhm_pts": 0,
        "y_max_observed": 0.0,
    }

    if t is None or y is None:
        return result

    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(t) < 20:
        return result

    if peak_idx is None:
        peak_idx = int(np.argmax(y))

    # --- Baseline: median of bottom 20% ---
    y_finite = y[np.isfinite(y)]
    if len(y_finite) < 10:
        return result
    y_sorted = np.sort(y_finite)
    n_bottom = max(5, len(y_sorted) // 5)
    baseline = float(np.median(y_sorted[:n_bottom]))
    y_net = y - baseline

    y_max = float(y_net[peak_idx])
    result["y_max_observed"] = y_max

    if y_max < 1e-6:
        return result

    # --- Plateau: punts consecutius dins 98% de y_max ---
    near_max = y_net >= y_max * 0.98
    plateau_start = peak_idx
    plateau_end = peak_idx
    while plateau_start > 0 and near_max[plateau_start - 1]:
        plateau_start -= 1
    while plateau_end < len(y_net) - 1 and near_max[plateau_end + 1]:
        plateau_end += 1
    plateau_width = plateau_end - plateau_start + 1
    result["plateau_width_pts"] = plateau_width

    # --- FWHM: walk outward from peak to half-maximum ---
    half_max = y_max * 0.50
    left_hm = 0
    for i in range(peak_idx - 1, -1, -1):
        if y_net[i] <= half_max:
            left_hm = i
            break
    right_hm = len(y_net) - 1
    for i in range(peak_idx + 1, len(y_net)):
        if y_net[i] <= half_max:
            right_hm = i
            break
    fwhm_pts = right_hm - left_hm
    result["fwhm_pts"] = fwhm_pts

    if fwhm_pts < 3:
        return result

    # --- Plateau/FWHM ratio ---
    # Gaussian: plateau(98%)/FWHM ≈ 0.17
    # Threshold 0.40 ≈ 15% clipping
    plateau_ratio = plateau_width / fwhm_pts
    result["plateau_ratio"] = float(plateau_ratio)
    result["is_saturated"] = plateau_ratio > 0.40

    return result


# =============================================================================
# IRREGULAR TOP DETECTION (jagged/batman)
# =============================================================================

def detect_irregular_top(t, y, top_pct=0.20, min_valley_depth=MIN_VALLEY_DEPTH):
    """
    Detect irregular peak top: valleys/jagged artifact from detector (historically "batman").

    ESTRICTE: Una vall només compta si té un pic ABANS i un pic DESPRÉS.
    Patró pic-vall-pic al cim del cromatograma.

    Parameters:
        t, y: time and signal arrays (peak segment)
        top_pct: fraction of peak height to analyze (0.20 = top 20%)
        min_valley_depth: minimum depth (as fraction of height) to count

    Returns:
        dict with is_irregular_top, n_valleys, max_depth, valley info
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    baseline = float(np.percentile(y, 10))
    y_max = float(np.max(y))
    height = y_max - baseline

    if height <= 0:
        return {"is_irregular_top": False, "n_valleys": 0, "max_depth": 0, "reason": "no_height"}

    # Define TOP region threshold
    threshold = baseline + height * (1 - top_pct)

    # Extract top region
    top_mask = y >= threshold
    t_top = t[top_mask]
    y_top = y[top_mask]

    if len(y_top) < 5:
        return {"is_irregular_top": False, "n_valleys": 0, "max_depth": 0, "reason": "top_too_small"}

    # Find LOCAL MAXIMA (peaks) in top region
    peaks_top, _ = find_peaks(y_top, distance=2)

    # Find LOCAL MINIMA (valleys) by inverting signal
    y_inverted = -y_top
    valleys, _ = find_peaks(y_inverted, distance=2)

    if len(valleys) == 0:
        return {
            "is_irregular_top": False,
            "n_valleys": 0,
            "max_depth": 0,
            "t_top": t_top,
            "y_top": y_top,
            "threshold": threshold,
            "reason": "no_valleys"
        }

    # ESTRICTE: Només comptar valls amb pic abans I després
    valid_valleys = []
    for v in valleys:
        has_peak_before = any(p < v for p in peaks_top)
        has_peak_after = any(p > v for p in peaks_top)
        if has_peak_before and has_peak_after:
            valid_valleys.append(v)

    if len(valid_valleys) == 0:
        return {
            "is_irregular_top": False,
            "n_valleys": 0,
            "max_depth": 0,
            "t_top": t_top,
            "y_top": y_top,
            "valleys": [],
            "threshold": threshold,
            "reason": "no_valid_valleys"
        }

    # Calculate depth of each valid valley
    y_max_top = float(np.max(y_top))
    valley_depths = [(y_max_top - y_top[v]) / height for v in valid_valleys]
    max_depth = max(valley_depths) if valley_depths else 0
    n_significant = sum(d > min_valley_depth for d in valley_depths)

    is_irregular_top = n_significant > 0 and max_depth > min_valley_depth

    return {
        "is_irregular_top": is_irregular_top,
        "n_valleys": n_significant,
        "max_depth": max_depth,
        "t_top": t_top,
        "y_top": y_top,
        "valleys": valid_valleys,  # Només valls amb pic-vall-pic
        "valley_depths": valley_depths,
        "threshold": threshold,
        "reason": "irregular_top_detected" if is_irregular_top else "valleys_too_shallow"
    }


# Backwards compatibility alias
detect_batman = detect_irregular_top


def calc_top_smoothness(t, y, top_pct=0.30):
    """
    Calculate smoothness at the top of a peak using second derivative.

    A normal peak has a smooth, negative second derivative at the top.
    Anomalies (notches, Batman) create irregular spikes in the second derivative.

    Parameters:
        t, y: time and signal arrays (peak segment)
        top_pct: fraction of peak height to analyze (0.30 = top 30%)

    Returns:
        dict with smoothness score (0-100), irregularity info
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(y) < 10:
        return {"smoothness": 100.0, "is_irregular": False, "reason": "too_short"}

    baseline = float(np.percentile(y, 10))
    y_max = float(np.max(y))
    height = y_max - baseline

    if height <= 0:
        return {"smoothness": 100.0, "is_irregular": False, "reason": "no_height"}

    # Extract TOP region
    threshold = baseline + height * (1 - top_pct)
    top_mask = y >= threshold
    y_top = y[top_mask]
    t_top = t[top_mask]

    if len(y_top) < 5:
        return {"smoothness": 100.0, "is_irregular": False, "reason": "top_too_small"}

    # Calculate first and second derivatives
    dt = np.diff(t_top)
    if np.any(dt <= 0):
        dt = np.where(dt <= 0, 1e-6, dt)  # Avoid division by zero

    dy = np.diff(y_top)
    d1 = dy / dt  # First derivative

    if len(d1) < 3:
        return {"smoothness": 100.0, "is_irregular": False, "reason": "not_enough_points"}

    dt2 = (dt[:-1] + dt[1:]) / 2  # Average dt for second derivative
    d2 = np.diff(d1) / dt2  # Second derivative

    # For a normal peak top, second derivative should be:
    # - Mostly NEGATIVE (concave down)
    # - SMOOTH (low variance in changes)

    # Metric 1: What fraction of d2 is positive? (should be near 0)
    positive_frac = np.sum(d2 > 0) / len(d2)

    # Metric 2: Coefficient of variation of d2 (normalized variability)
    d2_std = np.std(d2)
    d2_mean = np.mean(np.abs(d2))
    cv = d2_std / d2_mean if d2_mean > 1e-10 else 0

    # Metric 3: Count sign changes in d2 (oscillations = bad)
    sign_changes = np.sum(np.diff(np.sign(d2)) != 0)
    sign_change_rate = sign_changes / len(d2)

    # Combine into smoothness score (0-100)
    # Lower positive_frac, lower cv, lower sign_change_rate = higher smoothness
    smoothness = 100.0 * (1 - positive_frac) * np.exp(-cv/2) * np.exp(-sign_change_rate*2)
    smoothness = max(0.0, min(100.0, smoothness))

    # Nota: is_irregular es determina a detect_peak_anomaly amb el threshold configurable

    return {
        "smoothness": smoothness,
        "positive_frac": positive_frac,
        "cv": cv,
        "sign_change_rate": sign_change_rate,
        "d2": d2,
        "t_top": t_top,
        "y_top": y_top,
        "threshold": threshold
    }


def detect_peak_anomaly(t, y, top_pct=0.25, min_valley_depth=0.02, smoothness_threshold=70.0):
    """
    HYBRID detection: Irregular top (jagged/batman) + Smoothness.

    Strategy:
    1. First check for valleys at top (clear irregular top pattern)
    2. Then check smoothness (subtler irregularities)
    3. Combine both for final assessment

    Parameters:
        t, y: time and signal arrays (peak segment)
        top_pct: fraction of peak to analyze (0.25 = top 25%)
        min_valley_depth: minimum valley depth to count as irregular top
        smoothness_threshold: below this = rough top

    Returns:
        dict with is_anomaly, anomaly_type, details
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    # 1. Check for valleys at top (jagged/batman artifact)
    irr_top = detect_irregular_top(t, y, top_pct=top_pct, min_valley_depth=min_valley_depth)

    # 2. Check smoothness
    smooth = calc_top_smoothness(t, y, top_pct=top_pct)

    # 3. Combine results
    is_irregular_top = irr_top.get("is_irregular_top", False)
    smoothness = smooth.get("smoothness", 100.0)

    # Rough top: només si smoothness < threshold
    is_rough_top = smoothness < smoothness_threshold

    # Determine anomaly type
    if is_irregular_top:
        anomaly_type = "IRREGULAR_TOP"
        is_anomaly = True
    elif is_rough_top:
        anomaly_type = "ROUGH_TOP"
        is_anomaly = True
    else:
        anomaly_type = "OK"
        is_anomaly = False

    return {
        "is_anomaly": is_anomaly,
        "anomaly_type": anomaly_type,
        "smoothness": smoothness,
        "is_irregular_top": is_irregular_top,
        "is_irregular": is_irregular_top or is_rough_top,  # compat: qualsevol irregularitat
        "n_valleys": irr_top.get("n_valleys", 0),
        "max_valley_depth": irr_top.get("max_depth", 0),
        "irregular_top_info": irr_top,
        "smoothness_info": smooth
    }


# =============================================================================
# REPAIR FUNCTIONS
# =============================================================================

def find_tangents_and_anchors(t, y, middle_zone=(0.30, 0.70), deviation_pct=0.02):
    """
    Calculate tangent lines from middle zone and find anchor points.

    Parameters:
        t, y: time and signal arrays (peak segment)
        middle_zone: tuple (low_pct, high_pct) defining middle zone
        deviation_pct: threshold for anchor point deviation (fraction of height)

    Returns:
        dict with tangent parameters, intersection point, and anchors
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    baseline = float(np.percentile(y, 10))
    y_max = float(np.max(y))
    height = y_max - baseline

    if height <= 0:
        return None

    # Middle zone heights
    h_low = baseline + height * middle_zone[0]
    h_high = baseline + height * middle_zone[1]

    peak_idx = int(np.argmax(y))
    t_peak = float(t[peak_idx])

    # LEFT FLANK
    left_mask = (t < t_peak) & (y >= h_low) & (y <= h_high)
    if np.sum(left_mask) < 3:
        return None

    t_left = t[left_mask]
    y_left = y[left_mask]
    slope_L, intercept_L, r_L, _, _ = linregress(t_left, y_left)

    # RIGHT FLANK
    right_mask = (t > t_peak) & (y >= h_low) & (y <= h_high)
    if np.sum(right_mask) < 3:
        return None

    t_right = t[right_mask]
    y_right = y[right_mask]
    slope_R, intercept_R, r_R, _, _ = linregress(t_right, y_right)

    # INTERSECTION
    if abs(slope_L - slope_R) < 1e-10:
        return None

    t_intersect = (intercept_R - intercept_L) / (slope_L - slope_R)
    y_intersect = slope_L * t_intersect + intercept_L

    # ANCHOR POINTS
    deviation_threshold = height * deviation_pct

    # Left anchor
    left_of_peak = t < t_peak
    t_left_all = t[left_of_peak]
    y_left_all = y[left_of_peak]
    y_tangent_left = slope_L * t_left_all + intercept_L
    deviation_left = np.abs(y_left_all - y_tangent_left)

    anchor_left_idx = None
    for i in range(len(t_left_all) - 1, -1, -1):
        if deviation_left[i] < deviation_threshold:
            anchor_left_idx = i
            break

    if anchor_left_idx is not None:
        t_anchor_left = float(t_left_all[anchor_left_idx])
        y_anchor_left = float(y_left_all[anchor_left_idx])
    else:
        t_anchor_left = float(t_left[-1])
        y_anchor_left = float(y_left[-1])

    # Right anchor
    right_of_peak = t > t_peak
    t_right_all = t[right_of_peak]
    y_right_all = y[right_of_peak]
    y_tangent_right = slope_R * t_right_all + intercept_R
    deviation_right = np.abs(y_right_all - y_tangent_right)

    anchor_right_idx = None
    for i in range(len(t_right_all)):
        if deviation_right[i] < deviation_threshold:
            anchor_right_idx = i
            break

    if anchor_right_idx is not None:
        t_anchor_right = float(t_right_all[anchor_right_idx])
        y_anchor_right = float(y_right_all[anchor_right_idx])
    else:
        t_anchor_right = float(t_right[0])
        y_anchor_right = float(y_right[0])

    return {
        "slope_L": slope_L,
        "intercept_L": intercept_L,
        "r_L": r_L,
        "slope_R": slope_R,
        "intercept_R": intercept_R,
        "r_R": r_R,
        "t_intersect": t_intersect,
        "y_intersect": y_intersect,
        "t_anchor_left": t_anchor_left,
        "y_anchor_left": y_anchor_left,
        "t_anchor_right": t_anchor_right,
        "y_anchor_right": y_anchor_right,
        "h_low": h_low,
        "h_high": h_high,
        "baseline": baseline,
        "height": height,
    }


def fit_parabola(t1, y1, t2, y2, t3, y3):
    """
    Fit parabola through 3 points: y = at² + bt + c

    Returns:
        tuple (a, b, c) coefficients or None if fails
    """
    T = np.array([
        [t1**2, t1, 1],
        [t2**2, t2, 1],
        [t3**2, t3, 1]
    ])
    Y = np.array([y1, y2, y3])
    try:
        coeffs = np.linalg.solve(T, Y)
        return coeffs
    except np.linalg.LinAlgError:
        return None


def repair_with_parabola(t, y, factor=REPAIR_FACTOR, force=False):
    """
    Repair irregular peak top (jagged/batman artifact) using parabola interpolation.

    1. Detect irregular top pattern (skip if force=True)
    2. Calculate tangent intersection for theoretical peak
    3. Apply correction factor
    4. Fit parabola through anchors and theoretical max
    5. Replace deformed top with parabola

    Parameters:
        t, y: time and signal arrays (peak segment)
        factor: correction factor for tangent height
        force: if True, skip detect_irregular_top check (caller already decided repair needed)

    Returns:
        tuple (y_repaired, repair_info, was_repaired)
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    # Detect irregular top (jagged/batman) — skip if caller forces repair
    irr_top = detect_irregular_top(t, y)

    if not force and not irr_top["is_irregular_top"]:
        return y.copy(), irr_top, False

    # Get tangent info
    tangent = find_tangents_and_anchors(t, y)

    if tangent is None:
        return y.copy(), {"error": "tangent_failed"}, False

    # Theoretical max
    t_max = tangent["t_intersect"]
    y_max_theoretical = tangent["y_intersect"] * factor

    # Anchor points
    t1 = tangent["t_anchor_left"]
    y1 = tangent["y_anchor_left"]
    t3 = tangent["t_anchor_right"]
    y3 = tangent["y_anchor_right"]

    # Fit parabola
    coeffs = fit_parabola(t1, y1, t_max, y_max_theoretical, t3, y3)

    if coeffs is None:
        return y.copy(), {"error": "parabola_failed"}, False

    a, b, c = coeffs

    # Region to repair
    repair_mask = (t >= t1) & (t <= t3)
    t_repair = t[repair_mask]

    # Calculate parabola values
    y_parabola = a * t_repair**2 + b * t_repair + c

    # Replace: use maximum of original and parabola
    y_repaired = y.copy()
    y_repaired[repair_mask] = np.maximum(y[repair_mask], y_parabola)

    # Check if the repair actually changed anything
    actually_changed = not np.allclose(y_repaired, y, atol=1e-10)

    repair_info = {
        "irregular_top": irr_top,
        "tangent": tangent,
        "t_max": t_max,
        "y_max_theoretical": y_max_theoretical,
        "y_max_original": float(np.max(y)),
        "coeffs": coeffs,
        "t_anchor_left": t1,
        "y_anchor_left": y1,
        "t_anchor_right": t3,
        "y_anchor_right": y3,
    }

    if not actually_changed:
        repair_info["reason"] = "parabola_below_signal"

    return y_repaired, repair_info, actually_changed


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def calc_snr(y, peak_height, n_baseline_points=50):
    """
    Calculate Signal-to-Noise Ratio.

    Handles cases where early data points are zeros (before injection).
    """
    y = np.asarray(y, dtype=float)

    # Find first non-zero region
    nonzero_mask = np.abs(y) > 0.1
    if not np.any(nonzero_mask):
        return 0.0

    first_nonzero = np.argmax(nonzero_mask)

    # Use baseline after signal starts
    baseline_start = first_nonzero
    baseline_end = min(baseline_start + n_baseline_points, len(y))
    baseline_segment = y[baseline_start:baseline_end]

    if len(baseline_segment) < 10:
        baseline_segment = y[-n_baseline_points:]

    noise = float(np.std(baseline_segment))

    # Fallback: use lower 10% of signal
    if noise < 0.1:
        sorted_y = np.sort(y[nonzero_mask])
        n_low = max(10, len(sorted_y) // 10)
        noise = float(np.std(sorted_y[:n_low]))

    if noise > 0:
        return peak_height / noise
    return 0.0


def calc_peak_area(t, y, left_idx, right_idx, baseline):
    """Calculate peak area above baseline using trapezoidal integration."""
    t_seg = t[left_idx:right_idx+1]
    y_seg = y[left_idx:right_idx+1] - baseline
    y_seg = np.maximum(y_seg, 0)
    return trapezoid(y_seg, t_seg) if len(t_seg) > 1 else 0.0


def calc_pearson(t1, y1, t2, y2):
    """Calculate Pearson correlation between two chromatograms."""
    if t1 is None or t2 is None or len(t1) < 10 or len(t2) < 10:
        return np.nan

    t_min = max(t1.min(), t2.min())
    t_max = min(t1.max(), t2.max())

    if t_max <= t_min:
        return np.nan

    t_common = np.linspace(t_min, t_max, PEARSON_INTERP_POINTS)
    y1_interp = np.interp(t_common, t1, y1)
    y2_interp = np.interp(t_common, t2, y2)

    try:
        r, _ = pearsonr(y1_interp, y2_interp)
        return r
    except (ValueError, TypeError):
        return np.nan


# =============================================================================
# CALIBRATION
# =============================================================================

def calibrate_factor(t, y):
    """
    For NORMAL peaks, calculate what the correction factor should be.

    Compares theoretical height (tangent intersection) with actual height.
    factor = y_real / y_tangent

    Returns:
        dict with factor and info, or None if not suitable
    """
    irr_top = detect_irregular_top(t, y)
    if irr_top["is_irregular_top"]:
        return None

    tangent = find_tangents_and_anchors(t, y)
    if tangent is None:
        return None

    y_tangent = tangent["y_intersect"]
    y_real = float(np.max(y))

    if y_tangent <= 0:
        return None

    factor = y_real / y_tangent

    if factor < REPAIR_FACTOR_MIN or factor > REPAIR_FACTOR_MAX:
        return None

    return {
        "factor": factor,
        "y_real": y_real,
        "y_tangent": y_tangent,
        "r_left": tangent["r_L"],
        "r_right": tangent["r_R"],
    }


# =============================================================================
# TIMEOUT DETECTION (TOC Syringe Reload)
# =============================================================================

# Configuració detecció de timeouts TOC
TIMEOUT_CONFIG = {
    "threshold_sec": 60,        # Considera timeout si dt > 60 segons
    "major_timeout_sec": 70,    # Timeout major (recàrrega xeringues ~74s)
    "affected_zone_pre": 0.5,   # Minuts afectats ABANS del gap (~0.5 min)
    "affected_zone_post": 1.0,  # Minuts afectats DESPRÉS del gap (~1.0 min)
    "zones": {
        "RUN_START": [0, 0],      # Abans de BioP (inici run)
        "BioP": [0, 18],          # Biopolímers - CRÍTIC
        "HS": [18, 23],           # Substàncies húmiques - MOLT CRÍTIC
        "BB": [23, 30],           # Building Blocks - CRÍTIC
        "SB": [30, 40],           # Small Building blocks - CRÍTIC
        "LMW": [40, 70],          # Low Molecular Weight - Acceptable
        "POST_RUN": [70, 100],    # Post-run - IDEAL
    },
    "severity": {
        "RUN_START": "INFO",      # Timeout a l'inici, abans de pics
        "BioP": "WARNING",        # Pèrdua de biopolímers
        "HS": "CRITICAL",         # Pèrdua de substàncies húmiques (zona més important)
        "BB": "WARNING",          # Pèrdua de building blocks
        "SB": "WARNING",          # Pèrdua de small building blocks
        "LMW": "INFO",            # Zona de baix pes molecular, acceptable
        "POST_RUN": "OK",         # Zona ideal, sense impacte
        "BP_PEAK": "CRITICAL",    # BP: timeout a zona del pic principal (0-5 min)
        "BP_TAIL": "WARNING",     # BP: timeout a la cua (5-10 min)
    },
    # Zones específiques per BP (pic a ~1 min, cromatograma fins ~10 min)
    "bp_zones": {
        "BP_PEAK": [0, 5],        # Zona del pic principal → CRÍTIC
        "BP_TAIL": [5, 10],       # Zona de la cua → WARNING
        "POST_RUN": [10, 100],    # Post-run → OK
    },
}


def detect_timeout(t_min, threshold_sec=None, major_threshold_sec=None, is_bp=False):
    """
    Detecta timeouts en dades DOC basant-se en la cadència temporal.

    MÈTODE MÉS ROBUST per detectar pauses del TOC (recàrrega xeringues).
    Un timeout es detecta quan l'interval entre mesures supera el llindar.

    Args:
        t_min: Array de temps en minuts
        threshold_sec: Llindar per considerar timeout (defecte: 60s)
        major_threshold_sec: Llindar per timeout major/recàrrega (defecte: 70s)
        is_bp: Si és mode BP (pic a ~1 min, RUN_START no aplica)

    Returns:
        dict amb:
            - n_timeouts: nombre de timeouts detectats
            - n_major_timeouts: nombre de timeouts majors (recàrrega xeringues)
            - timeouts: llista de dicts amb info de cada timeout
            - dt_median_sec: mediana d'intervals (cadència normal)
            - dt_max_sec: interval màxim detectat
            - zone_summary: resum per zones
            - severity: severitat màxima detectada
            - warning_message: missatge de warning formatat
            - total_affected_min: duració total zona afectada (~2.5-3 min)
    """
    if threshold_sec is None:
        threshold_sec = TIMEOUT_CONFIG["threshold_sec"]
    if major_threshold_sec is None:
        major_threshold_sec = TIMEOUT_CONFIG["major_timeout_sec"]

    t = np.asarray(t_min)
    if len(t) < 2:
        return {
            "n_timeouts": 0,
            "n_major_timeouts": 0,
            "timeouts": [],
            "dt_median_sec": 0,
            "dt_max_sec": 0,
            "zone_summary": {},
            "severity": "OK",
            "warning_message": "",
            "total_affected_min": 0
        }

    # Calcular intervals en segons
    dt_sec = np.diff(t) * 60.0

    # Estadístiques bàsiques
    dt_median = float(np.median(dt_sec))
    dt_max = float(np.max(dt_sec))

    # Detectar timeouts
    timeout_indices = np.where(dt_sec > threshold_sec)[0]
    timeouts = []
    # Seleccionar zones segons mode (BP vs COLUMN)
    active_zones = TIMEOUT_CONFIG["bp_zones"] if is_bp else TIMEOUT_CONFIG["zones"]
    zone_counts = {zone: 0 for zone in active_zones.keys()}
    max_severity = "OK"
    severity_order = ["OK", "INFO", "WARNING", "CRITICAL"]

    pre_zone = TIMEOUT_CONFIG["affected_zone_pre"]
    post_zone = TIMEOUT_CONFIG["affected_zone_post"]

    for idx in timeout_indices:
        t_start = float(t[idx])
        t_end = float(t[idx + 1])
        duration_sec = float(dt_sec[idx])
        is_major = duration_sec >= major_threshold_sec

        # Zona afectada completa (PRE + gap + POST)
        affected_start = t_start - pre_zone
        affected_end = t_end + post_zone
        affected_duration = affected_end - affected_start

        # Determinar zona
        zone = "POST_RUN"  # Per defecte
        for zone_name, (t_ini, t_fi) in active_zones.items():
            if zone_name == "RUN_START":
                continue  # Tractem apart (només COLUMN)
            if t_ini <= t_start < t_fi:
                zone = zone_name
                break

        # Cas especial COLUMN: timeout a l'inici del run (t < 1 min)
        # En BP NO s'aplica perquè el pic principal és a ~1 min
        if t_start < 1.0 and not is_bp:
            zone = "RUN_START"

        zone_counts[zone] += 1
        severity = TIMEOUT_CONFIG["severity"].get(zone, "INFO")

        if severity_order.index(severity) > severity_order.index(max_severity):
            max_severity = severity

        timeouts.append({
            "index": int(idx),
            "t_start_min": round(t_start, 2),
            "t_end_min": round(t_end, 2),
            "duration_sec": round(duration_sec, 1),
            "is_major": is_major,
            "zone": zone,
            "severity": severity,
            "affected_start_min": round(affected_start, 2),
            "affected_end_min": round(affected_end, 2),
            "affected_duration_min": round(affected_duration, 2),
        })

    # Calcular duració total afectada
    total_affected = sum(to["affected_duration_min"] for to in timeouts)

    # Generar missatge de warning
    warning_parts = []
    n_major = sum(1 for to in timeouts if to["is_major"])

    if timeouts:
        critical_zones = [to for to in timeouts if to["severity"] == "CRITICAL"]
        warning_zones = [to for to in timeouts if to["severity"] == "WARNING"]
        info_zones = [to for to in timeouts if to["severity"] in ["INFO", "OK"]]

        if critical_zones:
            for to in critical_zones:
                warning_parts.append(
                    f"CRITICAL: Timeout {to['duration_sec']:.0f}s at {to['t_start_min']:.1f} min (HS zone)"
                )

        if warning_zones:
            for to in warning_zones[:3]:
                warning_parts.append(
                    f"WARNING: Timeout {to['duration_sec']:.0f}s at {to['t_start_min']:.1f} min ({to['zone']})"
                )

        if info_zones and not critical_zones and not warning_zones:
            for to in info_zones[:2]:
                warning_parts.append(
                    f"INFO: Timeout {to['duration_sec']:.0f}s at {to['t_start_min']:.1f} min ({to['zone']})"
                )

    warning_message = "; ".join(warning_parts) if warning_parts else ""

    # Llista de posicions temporals (t_start) per facilitar check TIMEOUT_IN_PEAK
    t_positions = [to["t_start_min"] for to in timeouts]

    return {
        "n_timeouts": len(timeouts),
        "n_major_timeouts": n_major,
        "timeouts": timeouts,
        "t_positions": t_positions,
        "dt_median_sec": round(dt_median, 2),
        "dt_max_sec": round(dt_max, 2),
        "zone_summary": {k: v for k, v in zone_counts.items() if v > 0},
        "severity": max_severity,
        "warning_message": warning_message,
        "total_affected_min": round(total_affected, 2)
    }


def format_timeout_status(timeout_info):
    """
    Formata l'estat de timeout per al camp Status del consolidat.

    Args:
        timeout_info: Dict retornat per detect_timeout()

    Returns:
        str: Estat formatat (OK, INFO, WARNING, CRITICAL)
    """
    if not timeout_info or timeout_info.get("n_timeouts", 0) == 0:
        return "OK"

    severity = timeout_info.get("severity", "OK")
    n_timeouts = timeout_info.get("n_timeouts", 0)
    n_major = timeout_info.get("n_major_timeouts", 0)

    if severity == "OK":
        return "OK"
    elif severity == "INFO":
        return f"INFO: {n_timeouts} timeout(s) in safe zone"
    elif severity == "WARNING":
        zones = timeout_info.get("zone_summary", {})
        affected = [z for z in ["BioP", "BB", "SB"] if zones.get(z, 0) > 0]
        return f"WARNING: timeout in {', '.join(affected)}"
    elif severity == "CRITICAL":
        return f"CRITICAL: timeout in HS zone ({n_major} major)"

    return f"{severity}: {n_timeouts} timeout(s)"


def estimate_timeout_for_uib(direct_timeout_info=None, sample_num=None,
                              t0=None, sample_duration=None, is_bp=False):
    """
    Estima la posició de timeouts per al senyal UIB.

    El timeout (recàrrega xeringa Sievers) afecta el DOC tant Direct com UIB,
    però UIB no té gaps temporals detectables (CSV per-injecció amb temps relatiu).
    L'anomalia UIB dura ~1.8 min (pic espuri + estabilització).

    Estratègia:
    1. Si DOC Direct disponible: transferir t_positions directament
    2. Si no: usar model predictiu (hpsec_planner) amb T0 i sample_duration

    Args:
        direct_timeout_info: timeout_info de DOC Direct (si disponible)
        sample_num: Número de mostra a la seqüència (1-based, per predicció)
        t0: Temps del primer timeout (min), estimat o del planner
        sample_duration: Duració de cada mostra (min)
        is_bp: Mode BP

    Returns:
        dict compatible amb timeout_info, amb camp extra 'estimated': True
    """
    # Durada de l'anomalia UIB (verificat: ~1.8 min, des de -0.1 a +1.7 min)
    ANOMALY_PRE_MIN = 0.2    # UIB comença a desviar-se just abans del gap DOC
    ANOMALY_POST_MIN = 1.8   # Pic espuri + estabilització

    zones = TIMEOUT_CONFIG["bp_zones"] if is_bp else TIMEOUT_CONFIG["zones"]
    severity_map = TIMEOUT_CONFIG["severity"]

    def _classify_position(pos):
        """Classifica una posició de timeout per zona i severitat."""
        for zone_name, (z_start, z_end) in zones.items():
            if z_start <= pos < z_end:
                return zone_name, severity_map.get(zone_name, "OK")
        return "POST_RUN", "OK"

    # --- Font 1: Transferir des de DOC Direct ---
    if direct_timeout_info and direct_timeout_info.get("n_timeouts", 0) > 0:
        t_positions = direct_timeout_info.get("t_positions", [])
        if t_positions:
            # Construir timeout_info per UIB amb les mateixes posicions
            timeouts = []
            zone_counts = {}
            max_sev_order = {"OK": 0, "INFO": 1, "WARNING": 2, "CRITICAL": 3}
            max_severity = "OK"

            for t_pos in t_positions:
                zone, sev = _classify_position(t_pos)
                timeouts.append({
                    "position_min": round(t_pos, 2),
                    "zone": zone,
                    "severity": sev,
                    "affected_start": round(t_pos - ANOMALY_PRE_MIN, 2),
                    "affected_end": round(t_pos + ANOMALY_POST_MIN, 2),
                })
                zone_counts[zone] = zone_counts.get(zone, 0) + 1
                if max_sev_order.get(sev, 0) > max_sev_order.get(max_severity, 0):
                    max_severity = sev

            return {
                "estimated": True,
                "source": "direct_transfer",
                "n_timeouts": len(t_positions),
                "n_major_timeouts": len(t_positions),  # Tots són recàrrega
                "timeouts": timeouts,
                "t_positions": [round(p, 2) for p in t_positions],
                "zone_summary": {k: v for k, v in zone_counts.items() if v > 0},
                "severity": max_severity,
                "anomaly_duration_min": ANOMALY_PRE_MIN + ANOMALY_POST_MIN,
                "warning_message": (
                    f"Timeout estimat (de DOC Direct): {len(t_positions)} "
                    f"timeout(s), severitat {max_severity}"
                ),
            }

    # --- Font 2: Model predictiu (planner) ---
    if sample_num is not None and t0 is not None:
        if sample_duration is None:
            sample_duration = 12.0 if is_bp else 78.65  # Durades típiques

        try:
            from hpsec_planner import predict_timeout_position
            pos = predict_timeout_position(sample_num, t0, sample_duration)
        except ImportError:
            pos = None

        if pos is not None:
            zone, sev = _classify_position(pos)
            return {
                "estimated": True,
                "source": "planner_prediction",
                "n_timeouts": 1,
                "n_major_timeouts": 1,
                "timeouts": [{
                    "position_min": round(pos, 2),
                    "zone": zone,
                    "severity": sev,
                    "affected_start": round(pos - ANOMALY_PRE_MIN, 2),
                    "affected_end": round(pos + ANOMALY_POST_MIN, 2),
                }],
                "t_positions": [round(pos, 2)],
                "zone_summary": {zone: 1},
                "severity": sev,
                "anomaly_duration_min": ANOMALY_PRE_MIN + ANOMALY_POST_MIN,
                "warning_message": (
                    f"Timeout predit (model planner): t={pos:.1f} min, "
                    f"zona {zone} ({sev})"
                ),
            }

    # Sense informació suficient
    return {
        "estimated": True,
        "source": "unknown",
        "n_timeouts": 0,
        "n_major_timeouts": 0,
        "timeouts": [],
        "t_positions": [],
        "zone_summary": {},
        "severity": "OK",
        "warning_message": "",
    }


# =============================================================================
# PEAK DETECTION
# =============================================================================

def find_peak_boundaries(t, y, peak_idx, baseline_level=None, threshold_pct=5.0, is_bp=False):
    """
    Troba els límits d'integració d'un pic usant mètode de derivades (estil Agilent).

    MÈTODE (tangent projection):
    1. Suavitzar senyal amb Savitzky-Golay per calcular derivades netes
    2. Calcular 2a derivada → trobar punts d'inflexió (zero-crossings de d2)
    3. Als punts d'inflexió, calcular la tangent (pendent = d1)
    4. Projectar les tangents fins al nivell de baseline
       → on la tangent creua el baseline = límit d'integració
    5. Fallback a threshold si derivades no donen resultat fiable

    AVANTATGE vs threshold:
    - Independent del volum d'injecció (mateixa forma de pic → mateixos límits)
    - Captura ~95.5% de l'àrea (equivalent a ±2σ en gaussiana ideal)
    - Més robust amb baseline drift
    - Consistent amb integració Agilent/ChemStation

    Args:
        t: Array de temps
        y: Array de senyal
        peak_idx: Índex del màxim del pic
        baseline_level: Nivell de baseline (auto-calculat si None)
        threshold_pct: % de l'amplitud per fallback (defecte: 5%)
        is_bp: Si és mode BP (afecta càlcul baseline)

    Returns:
        tuple(left_idx, right_idx): Índexs dels límits d'integració
    """
    n = len(y)
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    if n < 15:
        return 0, n - 1

    # Calcular baseline si no proporcionat
    if baseline_level is None:
        if is_bp:
            n_edge = max(20, n // 5)
            baseline_level = float(np.median(y[-n_edge:]))
        else:
            n_edge = max(10, n // 10)
            baseline_level = float(min(np.median(y[:n_edge]), np.median(y[-n_edge:])))

    peak_height = y[peak_idx]
    peak_amplitude = peak_height - baseline_level

    if peak_amplitude <= 0:
        return 0, n - 1

    # --- MÈTODE DERIVADES AMB PROJECCIÓ TANGENT (Agilent-style) ---

    # 1. Suavitzar per calcular derivades netes
    #    Finestra SG basada en temps (~0.7 min), independent de la durada del cromatograma.
    #    Una finestra massa gran (ex: proporcional a n) aplana el pic i redueix les
    #    pendents al punt d'inflexió → la projecció tangent va massa lluny.
    dt_median = float(np.median(np.diff(t))) if n > 1 else 1.0
    sg_target_min = 0.7  # minuts — ~FWHM típic d'un pic HPLC-SEC
    sg_window = int(sg_target_min / dt_median) if dt_median > 0 else 11
    sg_window = sg_window if sg_window % 2 == 1 else sg_window + 1
    sg_window = max(7, min(sg_window, n if n % 2 == 1 else n - 1))
    if sg_window > n:
        return _find_peak_boundaries_threshold(
            t, y, peak_idx, baseline_level, threshold_pct)

    try:
        y_smooth = savgol_filter(y, window_length=sg_window, polyorder=3)
    except Exception:
        return _find_peak_boundaries_threshold(
            t, y, peak_idx, baseline_level, threshold_pct)

    # 2. Calcular 1a i 2a derivada del senyal suavitzat
    dt_arr = np.diff(t)
    dt_arr = np.where(dt_arr <= 0, 1e-8, dt_arr)

    d1 = np.diff(y_smooth) / dt_arr           # 1a derivada (n-1 punts)
    d2 = np.diff(d1) / dt_arr[:-1]            # 2a derivada (n-2 punts)

    if len(d2) < 5:
        return _find_peak_boundaries_threshold(
            t, y, peak_idx, baseline_level, threshold_pct)

    # 3. Trobar punts d'inflexió: zero-crossings de d2 a cada costat del pic
    peak_in_d2 = min(peak_idx, len(d2) - 1)

    # Punt d'inflexió esquerre (d2 passa de positiu a negatiu entrant al pic)
    infl_left = None
    for i in range(peak_in_d2 - 1, 0, -1):
        if d2[i] >= 0 and d2[min(i + 1, len(d2) - 1)] < 0:
            infl_left = i + 1  # Index en coordenades de y (d2 offset +1)
            break

    # Punt d'inflexió dret (d2 passa de negatiu a positiu sortint del pic)
    infl_right = None
    for i in range(peak_in_d2 + 1, len(d2) - 1):
        if d2[i] >= 0 and d2[max(i - 1, 0)] < 0:
            infl_right = i + 1  # Index en coordenades de y
            break

    if infl_left is None or infl_right is None:
        return _find_peak_boundaries_threshold(
            t, y, peak_idx, baseline_level, threshold_pct)

    # 4. Projecció tangent: des del punt d'inflexió fins al baseline
    #    Tangent: y_tang(t_x) = y_infl + slope * (t_x - t_infl)
    #    Cruïlla amb baseline: baseline = y_infl + slope * (t_x - t_infl)
    #    t_x = t_infl + (baseline - y_infl) / slope

    # Tangent esquerra (pendent positiva, pujant cap al pic)
    slope_left = float(d1[min(infl_left, len(d1) - 1)])
    t_infl_left = float(t[infl_left])
    y_infl_left = float(y_smooth[infl_left])

    if slope_left > 1e-10:  # Pendent positiva (pujada)
        t_baseline_left = t_infl_left + (baseline_level - y_infl_left) / slope_left
    else:
        t_baseline_left = t_infl_left  # Fallback

    # Tangent dreta (pendent negativa, baixant des del pic)
    slope_right = float(d1[min(infl_right, len(d1) - 1)])
    t_infl_right = float(t[infl_right])
    y_infl_right = float(y_smooth[infl_right])

    if slope_right < -1e-10:  # Pendent negativa (baixada)
        t_baseline_right = t_infl_right + (baseline_level - y_infl_right) / slope_right
    else:
        t_baseline_right = t_infl_right  # Fallback

    # 5. Convertir temps de projecció a índexs
    left_idx = int(np.searchsorted(t, t_baseline_left))
    right_idx = int(np.searchsorted(t, t_baseline_right))

    # Clamp als límits de l'array
    left_idx = max(0, min(left_idx, n - 1))
    right_idx = max(0, min(right_idx, n - 1))

    # 6. VALIDACIÓ
    width_idx = right_idx - left_idx
    min_width = max(10, n // 30)
    max_width = int(n * 0.90)

    if width_idx < min_width or width_idx > max_width:
        return _find_peak_boundaries_threshold(
            t, y, peak_idx, baseline_level, threshold_pct)

    if left_idx >= peak_idx or right_idx <= peak_idx:
        return _find_peak_boundaries_threshold(
            t, y, peak_idx, baseline_level, threshold_pct)

    return int(left_idx), int(right_idx)


def _find_peak_boundaries_threshold(t, y, peak_idx, baseline_level, threshold_pct=5.0):
    """
    Fallback: troba límits per threshold (mètode antic).

    Usat quan el mètode de derivades no és fiable (poc senyal, soroll, etc.)
    """
    n = len(y)
    threshold = baseline_level + (threshold_pct / 100.0) * (y[peak_idx] - baseline_level)

    # Buscar límit esquerre
    left_idx = peak_idx
    for i in range(peak_idx - 1, -1, -1):
        if y[i] <= threshold:
            left_idx = i
            break
        if i > 0 and y[i] < y[i-1] and y[i] < y[i+1]:
            left_idx = i
            break
        left_idx = i

    # Buscar límit dret
    right_idx = peak_idx
    for i in range(peak_idx + 1, n):
        if y[i] <= threshold:
            right_idx = i
            break
        if i < n - 1 and y[i] < y[i-1] and y[i] < y[i+1]:
            right_idx = i
            break
        right_idx = i

    return int(left_idx), int(right_idx)


def detect_main_peak(t, y, min_prominence_pct=5.0, is_bp=None):
    """
    Detecta el pic principal d'un cromatograma amb límits d'integració correctes.

    Versió unificada que combina:
    - Detecció de pics amb prominència
    - Càlcul de baseline adaptat a BP/COLUMN
    - Límits d'integració precisos via find_peak_boundaries()

    Args:
        t: Array de temps (minuts)
        y: Array de senyal (mAU)
        min_prominence_pct: Prominència mínima com a % del màxim (defecte: 5%)
        is_bp: Si és mode BP (auto-detectat si None basant-se en durada)

    Returns:
        Dict amb:
            - valid: bool - True si s'ha detectat pic
            - t_max: temps al màxim
            - t_start, t_end: temps dels límits
            - peak_idx, left_idx, right_idx: índexs
            - area: àrea del pic
            - height: altura del pic
            - prominence: prominència del pic
            - is_bp: mode BP detectat/usat
            - baseline_level: nivell de baseline calculat
            - fallback: True si s'ha usat fallback (sense pic detectat)
    """
    if t is None or y is None:
        return {"valid": False}

    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(t) < 10 or len(y) < 10:
        return {"valid": False}

    y_max = float(np.nanmax(y))
    if y_max < 1e-6:
        return {"valid": False}

    # Auto-detectar mode BP si no especificat
    if is_bp is None:
        t_max_chromato = float(np.max(t))
        is_bp = t_max_chromato < 20  # BP típicament < 20 min

    # Calcular prominència mínima
    min_prominence = y_max * (min_prominence_pct / 100.0)

    # Trobar pics
    peaks, props = find_peaks(y, prominence=min_prominence, width=3)

    if len(peaks) == 0:
        # Fallback: usar màxim global
        peak_idx = int(np.argmax(y))
        n = len(y)
        if is_bp:
            n_edge = max(20, n // 5)
            baseline_level = float(np.median(y[-n_edge:]))
        else:
            n_edge = max(10, n // 10)
            baseline_level = float(min(np.median(y[:n_edge]), np.median(y[-n_edge:])))

        return {
            "valid": True,
            "t_max": float(t[peak_idx]),
            "peak_idx": peak_idx,
            "left_idx": 0,
            "right_idx": len(y) - 1,
            "t_start": float(t[0]),
            "t_end": float(t[-1]),
            "area": float(trapezoid(y, t)),
            "height": float(y[peak_idx]),
            "prominence": float(y_max),
            "is_bp": is_bp,
            "baseline_level": baseline_level,
            "fallback": True,
        }

    # Seleccionar pic amb major prominència
    idx = int(np.argmax(props["prominences"]))
    main_peak = int(peaks[idx])

    # Calcular baseline
    n = len(y)
    if is_bp:
        n_edge = max(20, n // 5)
        baseline_level = float(np.median(y[-n_edge:]))
    else:
        n_edge = max(10, n // 10)
        baseline_level = float(min(np.median(y[:n_edge]), np.median(y[-n_edge:])))

    # =====================================================================
    # PRE-REPAIR: Detectar i reparar cim irregular ABANS de find_peak_boundaries
    # Si el cim del pic té artefactes (jagged/batman), la projecció tangent
    # dóna límits erronis. Reparem primer, integrem sobre senyal reparat.
    # =====================================================================
    irregular_top_info = None
    smoothness_info = None
    irregular_top_repaired = False
    y_for_boundaries = y  # per defecte, senyal original

    # Segment al voltant del pic per detecció (±5 min COLUMN, ±3 min BP)
    t_peak = float(t[main_peak])
    half_window = 3.0 if is_bp else 5.0
    seg_mask = (t >= t_peak - half_window) & (t <= t_peak + half_window)
    t_seg = t[seg_mask]
    y_seg = y[seg_mask]

    if len(y_seg) > 20:
        irregular_top_info = detect_irregular_top(t_seg, y_seg)
        smoothness_info = calc_top_smoothness(t_seg, y_seg)
        smoothness_val = smoothness_info.get("smoothness", 100.0)

        # Reparar NOMÉS si IRREGULAR_TOP (valls pic-vall-pic reals)
        # smoothness < 70% NO és criteri suficient — genera 95% falsos positius
        # (pics BP naturalment aplanats, pics bimodals, etc.)
        needs_repair = irregular_top_info.get("is_irregular_top", False)

        if needs_repair:
            y_seg_repaired, repair_info, was_repaired = repair_with_parabola(
                t_seg, y_seg
            )
            if was_repaired:
                irregular_top_repaired = True
                # Aplicar reparació al senyal complet per find_peak_boundaries
                y_for_boundaries = y.copy()
                y_for_boundaries[seg_mask] = y_seg_repaired

    # Trobar límits precisos (sobre senyal reparat si calia)
    left_idx, right_idx = find_peak_boundaries(
        t, y_for_boundaries, main_peak, baseline_level, threshold_pct=5.0, is_bp=is_bp
    )

    left_idx = max(0, left_idx)
    right_idx = min(len(y) - 1, right_idx)

    # Calcular àrea sobre senyal ORIGINAL (no reparat) amb els límits correctes
    # La reparació serveix per trobar els límits, l'àrea és sobre dades originals
    if right_idx > left_idx:
        area = float(trapezoid(y[left_idx:right_idx + 1], t[left_idx:right_idx + 1]))
    else:
        area = 0.0

    # Si reparat, calcular també àrea sobre senyal reparat (per calibració)
    area_repaired = None
    if irregular_top_repaired and right_idx > left_idx:
        area_repaired = float(trapezoid(
            y_for_boundaries[left_idx:right_idx + 1], t[left_idx:right_idx + 1]
        ))

    result = {
        "valid": True,
        "t_max": float(t[main_peak]),
        "t_start": float(t[left_idx]),
        "t_end": float(t[right_idx]),
        "peak_idx": main_peak,
        "left_idx": left_idx,
        "right_idx": right_idx,
        "area": area,
        "height": float(y[main_peak]),
        "prominence": float(props["prominences"][idx]),
        "is_bp": is_bp,
        "baseline_level": baseline_level,
        "fallback": False,
    }

    # Afegir info d'irregular top si detectat (IRREGULAR_TOP o ROUGH_TOP)
    has_irregular_top = (irregular_top_info is not None
                         and irregular_top_info.get("is_irregular_top", False))
    has_rough_top = (smoothness_info is not None
                     and smoothness_info.get("smoothness", 100.0) < 70.0)
    if has_irregular_top or has_rough_top or irregular_top_repaired:
        result["is_irregular_top"] = True
        result["irregular_top_repaired"] = irregular_top_repaired
        result["irregular_top_info"] = irregular_top_info
        if area_repaired is not None:
            result["area_original"] = area
            result["area_repaired"] = area_repaired
        if has_rough_top and not has_irregular_top:
            result["anomaly_subtype"] = "ROUGH_TOP"
    if smoothness_info is not None:
        result["smoothness"] = smoothness_info.get("smoothness", 100.0)

    return result


def detect_all_peaks(t, y, min_prominence_pct=5.0):
    """
    Detecta TOTS els pics significatius d'un cromatograma.

    Usat per:
    - Validació KHP (no hauria de tenir múltiples pics significatius)
    - Detecció d'interferències

    Args:
        t: Array de temps
        y: Array de senyal
        min_prominence_pct: Prominència mínima com a % del rang (defecte: 5%)

    Returns:
        Llista de dicts amb info de cada pic:
            - idx: índex del pic
            - t: temps del màxim
            - height: altura
            - left_idx, right_idx: límits
            - prominence: prominència del pic
    """
    if t is None or y is None or len(t) < 20:
        return []

    y = np.asarray(y)
    t = np.asarray(t)

    y_range = np.max(y) - np.min(y)
    min_prominence = y_range * min_prominence_pct / 100.0

    peaks, properties = find_peaks(y, prominence=min_prominence, width=3)

    if len(peaks) == 0:
        return []

    left_bases = properties.get("left_bases", [0] * len(peaks))
    right_bases = properties.get("right_bases", [len(y) - 1] * len(peaks))
    prominences = properties.get("prominences", [0] * len(peaks))

    all_peaks = []
    for i, peak_idx in enumerate(peaks):
        left_idx = int(left_bases[i])
        right_idx = int(right_bases[i])

        all_peaks.append({
            "idx": int(peak_idx),
            "t": float(t[peak_idx]),
            "height": float(y[peak_idx]),
            "left_idx": left_idx,
            "right_idx": right_idx,
            "prominence": float(prominences[i]),
        })

    return all_peaks


# =============================================================================
# INTEGRATION
# =============================================================================

def integrate_above_baseline(t, y, baseline_mean, baseline_std, threshold_sigma=3.0):
    """
    Integra només el senyal per sobre del baseline + threshold.

    Mètode estàndard en cromatografia per:
    - Evitar integrar soroll de baseline
    - Obtenir àrees comparables entre runs
    - Reduir variabilitat per baseline drift

    Args:
        t: Array de temps (min)
        y: Array de senyal (mAU)
        baseline_mean: Mitjana del baseline
        baseline_std: Desviació estàndard del baseline (soroll)
        threshold_sigma: Múltiple de sigma per sobre baseline (defecte: 3.0)

    Returns:
        Dict amb:
            - area: àrea integrada sobre el threshold
            - threshold: valor del threshold usat
            - pct_above: percentatge de punts per sobre del threshold
    """
    t = np.asarray(t)
    y = np.asarray(y)

    if len(t) < 5 or len(y) < 5:
        return {'area': 0.0, 'threshold': 0.0, 'pct_above': 0.0}

    # Calcular threshold
    threshold = baseline_mean + threshold_sigma * baseline_std

    # Restar threshold i quedar-se només amb valors positius
    y_above = np.maximum(y - threshold, 0)

    # Integrar
    area = float(trapezoid(y_above, t))

    # Estadístiques
    pct_above = float(np.sum(y > threshold) / len(y) * 100)

    return {
        'area': area,
        'threshold': float(threshold),
        'pct_above': pct_above,
        'n_points_above': int(np.sum(y > threshold)),
        'n_points_total': len(y)
    }


def integrate_chromatogram(t, y, left_idx=None, right_idx=None,
                           baseline=None, mode='full'):
    """
    Integra un cromatograma amb mode seleccionable.

    Args:
        t: Array de temps
        y: Array de senyal
        left_idx: Índex inicial (opcional, defecte: 0)
        right_idx: Índex final (opcional, defecte: len-1)
        baseline: Valor baseline (opcional, calcula si None)
        mode: Mode d'integració:
            - 'full': Integra tota l'àrea del segment
            - 'main_peak': Detecta i integra només el pic principal

    Returns:
        Dict amb:
            - area: àrea integrada
            - baseline: valor baseline usat
            - left_idx, right_idx: límits d'integració
            - t_start, t_end: temps dels límits
            - mode: mode utilitzat
            - peak_info: info del pic (si mode='main_peak')
    """
    t = np.asarray(t)
    y = np.asarray(y)

    if len(t) < 5 or len(y) < 5:
        return {
            'area': 0.0,
            'baseline': 0.0,
            'left_idx': 0,
            'right_idx': 0,
            't_start': 0.0,
            't_end': 0.0,
            'mode': mode,
            'error': 'insufficient_data'
        }

    if left_idx is None:
        left_idx = 0
    if right_idx is None:
        right_idx = len(y) - 1

    # Assegurar límits vàlids
    left_idx = max(0, min(left_idx, len(y) - 1))
    right_idx = max(0, min(right_idx, len(y) - 1))

    if right_idx <= left_idx:
        right_idx = len(y) - 1

    # Baseline
    if baseline is None:
        segment = y[left_idx:right_idx+1]
        baseline = float(np.percentile(segment, 10))

    if mode == 'full':
        # Integrar tot el segment
        t_seg = t[left_idx:right_idx+1]
        y_seg = y[left_idx:right_idx+1] - baseline
        y_seg = np.maximum(y_seg, 0)  # No negatius
        area = float(trapezoid(y_seg, t_seg))

        return {
            'area': area,
            'baseline': baseline,
            'left_idx': left_idx,
            'right_idx': right_idx,
            't_start': float(t[left_idx]),
            't_end': float(t[right_idx]),
            'mode': 'full'
        }

    elif mode == 'main_peak':
        # Detectar i integrar només pic principal
        peak_info = detect_main_peak(t, y)

        if not peak_info.get('valid', False):
            # Fallback a full
            return integrate_chromatogram(t, y, left_idx, right_idx,
                                          baseline, mode='full')

        p_left = peak_info['left_idx']
        p_right = peak_info['right_idx']

        # Calcular baseline específic del pic
        peak_segment = y[p_left:p_right+1]
        peak_baseline = float(np.percentile(peak_segment, 10))

        t_peak = t[p_left:p_right+1]
        y_peak = y[p_left:p_right+1] - peak_baseline
        y_peak = np.maximum(y_peak, 0)
        area = float(trapezoid(y_peak, t_peak))

        return {
            'area': area,
            'baseline': peak_baseline,
            'left_idx': p_left,
            'right_idx': p_right,
            't_start': float(t[p_left]),
            't_end': float(t[p_right]),
            'mode': 'main_peak',
            'peak_info': peak_info
        }

    else:
        raise ValueError(f"Mode desconegut: {mode}. Usar 'full' o 'main_peak'.")


# =============================================================================
# PEAK METRICS (FWHM, Symmetry)
# =============================================================================

def calculate_fwhm(t, y, peak_idx, left_idx=None, right_idx=None):
    """
    Calcula Full Width at Half Maximum (FWHM) d'un pic.

    FWHM és l'amplada del pic a mitja alçada, indicador de:
    - Resolució cromatogràfica
    - Degradació de columna (FWHM augmenta)
    - Qualitat del pic

    Args:
        t: Array de temps (minuts)
        y: Array de senyal
        peak_idx: Índex del màxim del pic
        left_idx: Límit esquerre opcional (si None, busca en tot l'array)
        right_idx: Límit dret opcional (si None, busca en tot l'array)

    Returns:
        FWHM en minuts, o np.nan si no es pot calcular
    """
    try:
        t = np.asarray(t, dtype=float)
        y = np.asarray(y, dtype=float)

        if len(t) < 3 or peak_idx < 0 or peak_idx >= len(y):
            return np.nan

        # Límits de cerca
        if left_idx is None:
            left_idx = 0
        if right_idx is None:
            right_idx = len(y) - 1

        left_idx = max(0, left_idx)
        right_idx = min(len(y) - 1, right_idx)

        if peak_idx <= left_idx or peak_idx >= right_idx:
            return np.nan

        # Altura del pic i mitja altura
        h_peak = y[peak_idx]
        h_half = h_peak / 2

        # Buscar punt esquerre on creua h_half
        t_left = None
        for i in range(peak_idx, left_idx - 1, -1):
            if y[i] <= h_half:
                # Interpolació lineal per més precisió
                if i < peak_idx:
                    frac = (h_half - y[i]) / (y[i + 1] - y[i]) if y[i + 1] != y[i] else 0
                    t_left = t[i] + frac * (t[i + 1] - t[i])
                else:
                    t_left = t[i]
                break

        # Buscar punt dret on creua h_half
        t_right = None
        for i in range(peak_idx, right_idx + 1):
            if y[i] <= h_half:
                # Interpolació lineal
                if i > peak_idx:
                    frac = (h_half - y[i - 1]) / (y[i] - y[i - 1]) if y[i] != y[i - 1] else 0
                    t_right = t[i - 1] + frac * (t[i] - t[i - 1])
                else:
                    t_right = t[i]
                break

        if t_left is None or t_right is None:
            return np.nan

        fwhm = t_right - t_left
        return float(fwhm) if fwhm > 0 else np.nan

    except Exception as e:
        logger.debug("FWHM calculation failed: %s", e)
        return np.nan


def calculate_symmetry(t, y, peak_idx, left_idx=None, right_idx=None):
    """
    Calcula la simetria d'un pic (ratio amplades esquerra/dreta a mitja altura).

    Args:
        t: Array de temps (minuts)
        y: Array de senyal
        peak_idx: Índex del màxim del pic
        left_idx, right_idx: Límits opcionals

    Returns:
        Symmetry ratio (ideal ≈ 1.0), o np.nan si no es pot calcular
    """
    try:
        t = np.asarray(t, dtype=float)
        y = np.asarray(y, dtype=float)

        if len(t) < 3 or peak_idx < 0 or peak_idx >= len(y):
            return np.nan

        if left_idx is None:
            left_idx = 0
        if right_idx is None:
            right_idx = len(y) - 1

        left_idx = max(0, left_idx)
        right_idx = min(len(y) - 1, right_idx)

        if peak_idx <= left_idx or peak_idx >= right_idx:
            return np.nan

        h_peak = y[peak_idx]
        h_half = h_peak / 2
        t_peak = t[peak_idx]

        # Buscar punt esquerre
        t_left = None
        for i in range(peak_idx, left_idx - 1, -1):
            if y[i] <= h_half:
                if i < peak_idx:
                    frac = (h_half - y[i]) / (y[i + 1] - y[i]) if y[i + 1] != y[i] else 0
                    t_left = t[i] + frac * (t[i + 1] - t[i])
                else:
                    t_left = t[i]
                break

        # Buscar punt dret
        t_right = None
        for i in range(peak_idx, right_idx + 1):
            if y[i] <= h_half:
                if i > peak_idx:
                    frac = (h_half - y[i - 1]) / (y[i] - y[i - 1]) if y[i] != y[i - 1] else 0
                    t_right = t[i - 1] + frac * (t[i] - t[i - 1])
                else:
                    t_right = t[i]
                break

        if t_left is None or t_right is None:
            return np.nan

        width_left = t_peak - t_left
        width_right = t_right - t_peak

        if width_right <= 0:
            return np.nan

        return float(width_left / width_right)

    except Exception as e:
        logger.debug("Symmetry calculation failed: %s", e)
        return np.nan


# =============================================================================
# BASELINE FUNCTIONS (migrades de hpsec_utils.py 2026-02-03)
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
    data = np.asarray(data)
    data = data[np.isfinite(data)]
    if len(data) == 0:
        return 0.0
    counts, edges = np.histogram(data, bins=bins)
    i = int(np.argmax(counts))
    return 0.5 * (edges[i] + edges[i + 1])


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
    # NOTA (2026-02-03): COLUMN usa 0-10 min (pre-peak), evitant zones amb peaks
    if method.lower() == "bp":
        windows = baseline_cfg.get("windows_bp", [{"start": 5.0, "end": 10.0, "name": "post-peak"}])
    else:
        windows = baseline_cfg.get("windows_column", [
            {"start": 0.0, "end": 10.0, "name": "pre-peak"}
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

        if window_has_timeout(w_start, w_end):
            continue

        window_data = get_window_data(w_start, w_end)
        if len(window_data) < 10:
            continue

        window_data = window_data[np.isfinite(window_data)]
        if len(window_data) < 5:
            continue

        mean_val = float(np.mean(window_data))
        std_val = float(np.std(window_data))
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


def _sigma_clip(data, n_sigma=3):
    """Elimina outliers de la zona baseline usant mediana + MAD."""
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    if len(data) < 10:
        return data
    med = np.median(data)
    mad = np.median(np.abs(data - med))
    if mad < 1e-10:
        return data
    sigma_est = 1.4826 * mad  # MAD → desviació estàndard equivalent
    mask = np.abs(data - med) <= n_sigma * sigma_est
    clipped = data[mask]
    return clipped if len(clipped) >= 5 else data


def get_baseline_value(t, y, mode="COLUMN", config=None):
    """
    Calcula el valor de baseline per restar del senyal.

    FUNCIÓ UNIFICADA per a tot el pipeline (import, calibrate, process).

    Lògica:
    - BP: usar FINAL del cromatograma (després del pic) - últim X%
    - COLUMN: usar INICI del cromatograma (abans dels pics) - primer X%

    Args:
        t: Array de temps (minuts)
        y: Array de senyal (mAU)
        mode: "BP" o "COLUMN"
        config: ConfigManager instance (opcional, usa global si None)

    Returns:
        float: valor de baseline a restar
    """
    if config is None:
        from hpsec_config import get_config
        config = get_config()

    baseline_cfg = config.get("baseline", default={})

    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(y) < 10:
        return float(np.nanmin(y)) if len(y) > 0 else 0.0

    n = len(y)

    if mode.upper() == "BP":
        end_pct = baseline_cfg.get("bp_end_pct", 20)
        n_points = max(10, int(n * end_pct / 100))
        baseline_data = y[-n_points:]
    else:
        start_pct = baseline_cfg.get("column_start_pct", 15)
        n_points = max(10, int(n * start_pct / 100))
        baseline_data = y[:n_points]

    method = baseline_cfg.get("method", "mode")

    if method == "median":
        return float(np.median(baseline_data))
    else:
        # Sigma-clip: eliminar outliers (pics espuris, artefactes injecció)
        # abans del histograma per evitar que expandeixin el rang de bins
        baseline_data = _sigma_clip(baseline_data)
        return mode_robust(baseline_data)


def get_baseline_stats(t, y, mode="COLUMN", config=None):
    """
    Calcula estadístiques de baseline (mean, std) per SNR.

    Usa la mateixa zona que get_baseline_value() per coherència.

    Args:
        t: Array de temps (minuts)
        y: Array de senyal (mAU)
        mode: "BP" o "COLUMN"
        config: ConfigManager instance (opcional)

    Returns:
        dict amb: mean, std, min, max
    """
    if config is None:
        from hpsec_config import get_config
        config = get_config()

    baseline_cfg = config.get("baseline", default={})

    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    min_noise = baseline_cfg.get("min_noise_mau", 0.01)

    if len(y) < 10:
        return {"mean": 0.0, "std": min_noise, "min": 0.0, "max": 0.0}

    n = len(y)

    if mode.upper() == "BP":
        end_pct = baseline_cfg.get("bp_end_pct", 20)
        n_points = max(10, int(n * end_pct / 100))
        baseline_data = y[-n_points:]
    else:
        start_pct = baseline_cfg.get("column_start_pct", 15)
        n_points = max(10, int(n * start_pct / 100))
        baseline_data = y[:n_points]

    baseline_data = baseline_data[np.isfinite(baseline_data)]

    if len(baseline_data) < 5:
        return {"mean": 0.0, "std": min_noise, "min": 0.0, "max": 0.0}

    pct_low = baseline_cfg.get("stats_percentile_low", 5)
    pct_high = baseline_cfg.get("stats_percentile_high", 40)

    p_low = np.percentile(baseline_data, pct_low)
    p_high = np.percentile(baseline_data, pct_high)
    filtered = baseline_data[(baseline_data >= p_low) & (baseline_data <= p_high)]

    if len(filtered) < 5:
        filtered = baseline_data

    mean_val = float(np.mean(filtered))
    std_val = max(float(np.std(filtered)), min_noise)

    return {
        "mean": mean_val,
        "std": std_val,
        "min": float(np.min(baseline_data)),
        "max": float(np.max(baseline_data)),
    }


# =============================================================================
# ALIGNMENT FUNCTIONS (migrades de hpsec_analyze.py 2026-02-03)
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

    idx_max_ref = np.argmax(y_ref)
    idx_max_other = np.argmax(y_other)

    t_max_ref = t_ref[idx_max_ref]
    t_max_other = t_other[idx_max_other]

    shift = t_max_ref - t_max_other

    t_other_shifted = t_other + shift

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
    t_ref = np.asarray(t_ref).flatten()
    t_signal = np.asarray(t_signal).flatten()
    y_signal = np.asarray(y_signal).flatten()

    # Validar que t_signal i y_signal tenen la mateixa longitud
    if len(t_signal) != len(y_signal):
        raise ValueError(f"apply_shift: t_signal ({len(t_signal)}) i y_signal ({len(y_signal)}) tenen longituds diferents")

    t_shifted = t_signal + shift
    y_shifted = np.interp(t_ref, t_shifted, y_signal, left=0, right=0)
    return y_shifted


# =============================================================================
# SNR COMPLETE (consolidació de calc_snr, calculate_peak_snr, calculate_snr_info)
# =============================================================================

def calc_snr_complete(t, y, peak_height=None, peak_idx=None, method="column",
                      timeout_positions=None, config=None):
    """
    Calcula SNR, LOD, LOQ i estadístiques de baseline.

    FUNCIÓ CONSOLIDADA que unifica:
    - calc_snr() (versió bàsica)
    - calculate_peak_snr() de calibrate
    - calculate_snr_info() de analyze

    Args:
        t: Array de temps (minuts)
        y: Array de senyal (mAU)
        peak_height: Altura del pic (si None, usa màxim - baseline)
        peak_idx: Índex del pic (opcional, per calcular altura)
        method: "column" o "bp" - determina finestres de baseline
        timeout_positions: Llista de posicions temporals amb timeouts
        config: ConfigManager instance (opcional)

    Returns:
        dict amb:
            - snr: Signal-to-Noise Ratio
            - baseline_noise: Desviació estàndard del baseline (mAU)
            - baseline_mean: Mitjana del baseline (mAU)
            - lod: Limit of Detection = 3 × noise (mAU)
            - loq: Limit of Quantification = 10 × noise (mAU)
            - window_used: Finestra usada per calcular baseline
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(y) < 10:
        return {
            "snr": 0.0,
            "baseline_noise": 0.0,
            "baseline_mean": 0.0,
            "lod": 0.0,
            "loq": 0.0,
            "window_used": "insufficient_data"
        }

    use_windowed = (t is not None and len(t) > 10)

    if use_windowed:
        bl_stats = baseline_stats_windowed(
            t, y,
            method=method,
            timeout_positions=timeout_positions,
            config=config
        )
    else:
        bl_stats = baseline_stats(y)
        bl_stats["window_used"] = "percentile"

    baseline_mean = bl_stats.get("mean", 0.0)
    baseline_noise = bl_stats.get("std", 0.01)
    window_used = bl_stats.get("window_used", "unknown")

    if peak_height is None:
        if peak_idx is not None and 0 <= peak_idx < len(y):
            peak_height = y[peak_idx] - baseline_mean
        else:
            peak_height = float(np.max(y)) - baseline_mean

    peak_height = max(0.0, peak_height)

    if baseline_noise > 0:
        snr = peak_height / baseline_noise
    else:
        snr = 0.0

    lod = 3.0 * baseline_noise
    loq = 10.0 * baseline_noise

    return {
        "snr": float(snr),
        "baseline_noise": float(baseline_noise),
        "baseline_mean": float(baseline_mean),
        "lod": float(lod),
        "loq": float(loq),
        "window_used": window_used
    }


# =============================================================================
# SIGNAL COMPARISON (nova funció)
# =============================================================================

def compare_signals(t1, y1, t2, y2, normalize=False):
    """
    Compara dos senyals calculant Pearson i diferència d'àrea.

    Útil per:
    - Comparar Direct vs UIB (dins mateixa mostra)
    - Comparar rèpliques (rep1 vs rep2)
    - Verificar qualitat d'alineació

    Args:
        t1, y1: Temps i senyal del primer senyal
        t2, y2: Temps i senyal del segon senyal
        normalize: Si True, normalitza senyals abans de comparar

    Returns:
        dict amb:
            - pearson: Coeficient de correlació Pearson (-1 a 1)
            - area_diff_pct: Diferència percentual d'àrees (%)
            - area_1: Àrea del primer senyal
            - area_2: Àrea del segon senyal
            - n_points: Punts usats per la comparació
            - valid: True si la comparació és vàlida
    """
    result = {
        "pearson": np.nan,
        "area_diff_pct": np.nan,
        "area_1": 0.0,
        "area_2": 0.0,
        "n_points": 0,
        "valid": False
    }

    if t1 is None or t2 is None or y1 is None or y2 is None:
        return result

    t1 = np.asarray(t1, dtype=float)
    t2 = np.asarray(t2, dtype=float)
    y1 = np.asarray(y1, dtype=float)
    y2 = np.asarray(y2, dtype=float)

    if len(t1) < 10 or len(t2) < 10:
        return result

    t_min = max(float(np.min(t1)), float(np.min(t2)))
    t_max = min(float(np.max(t1)), float(np.max(t2)))

    if t_max <= t_min:
        return result

    n_points = min(len(t1), len(t2), 1000)
    t_common = np.linspace(t_min, t_max, n_points)

    y1_interp = np.interp(t_common, t1, y1)
    y2_interp = np.interp(t_common, t2, y2)

    if normalize:
        y1_max = np.max(np.abs(y1_interp))
        y2_max = np.max(np.abs(y2_interp))
        if y1_max > 0:
            y1_interp = y1_interp / y1_max
        if y2_max > 0:
            y2_interp = y2_interp / y2_max

    try:
        pearson_val, _ = pearsonr(y1_interp, y2_interp)
        result["pearson"] = float(pearson_val)
    except (ValueError, TypeError) as e:
        logger.debug("Pearson correlation failed: %s", e)
        result["pearson"] = np.nan

    area_1 = float(trapezoid(np.maximum(y1_interp, 0), t_common))
    area_2 = float(trapezoid(np.maximum(y2_interp, 0), t_common))

    result["area_1"] = area_1
    result["area_2"] = area_2

    max_area = max(area_1, area_2)
    if max_area > 0:
        result["area_diff_pct"] = abs(area_1 - area_2) / max_area * 100
    else:
        result["area_diff_pct"] = 0.0

    result["n_points"] = n_points
    result["valid"] = True

    return result
