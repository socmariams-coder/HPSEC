# -*- coding: utf-8 -*-
"""
HPSEC Core - Funcions compartides per anàlisi de cromatogrames
==============================================================

Mòdul centralitzat amb TOTES les funcions de detecció i processament.
Single Source of Truth per evitar duplicació de codi.

Funcions principals:
- Bi-Gaussiana: fit_bigaussian, bigaussian, check_asymmetry
- Detecció Batman: detect_batman, detect_peak_anomaly, calc_top_smoothness
- Reparació: repair_with_parabola, find_tangents_and_anchors
- Timeout TOC: detect_timeout, format_timeout_status (MILLOR MÈTODE: dt intervals)
- Detecció pics: detect_main_peak, detect_all_peaks
- Integració: integrate_chromatogram (mode='full'|'main_peak')
- Utilitats: calc_snr, calc_peak_area, calc_pearson

v1.1 - 2026-01-26: Afegides funcions timeout i detecció pics (migrades)
v1.0 - 2026-01-22: Versió inicial
"""

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import linregress, pearsonr
from scipy.optimize import curve_fit
from scipy.integrate import trapezoid


# =============================================================================
# LLINDARS GLOBALS
# =============================================================================

# R² thresholds for bi-Gaussian fit
THRESH_R2_VALID = 0.987      # R² >= 0.987 → VALID
THRESH_R2_CHECK = 0.980      # 0.980 <= R² < 0.987 → CHECK

# Repair thresholds
REPAIR_MIN_R2 = 0.980        # Don't repair if R² < this (garbage peak)
REPAIR_FACTOR = 0.85         # Tangent height correction factor

# Asymmetry thresholds
ASYM_MIN = 0.33              # sigma_right/sigma_left minimum
ASYM_MAX = 3.0               # sigma_right/sigma_left maximum

# Batman detection
MIN_VALLEY_DEPTH = 0.01      # 1% of peak height to detect Batman

# SNR
THRESH_SNR = 10.0            # Minimum acceptable SNR


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
        sigma_left_guess = hw_left / 1.177
    else:
        sigma_left_guess = (t_peak - t_seg[0]) / 3

    # Right half-width
    right_mask = (t_seg >= t_peak) & (y_seg >= half_height)
    if np.sum(right_mask) >= 1:
        t_right_half = t_seg[right_mask]
        hw_right = t_right_half[-1] - t_peak
        sigma_right_guess = hw_right / 1.177
    else:
        sigma_right_guess = (t_seg[-1] - t_peak) / 3

    sigma_left_guess = max(0.05, sigma_left_guess)
    sigma_right_guess = max(0.05, sigma_right_guess)

    try:
        popt, _ = curve_fit(
            bigaussian, t_fit, y_fit,
            p0=[peak_height, t_peak, sigma_left_guess, sigma_right_guess, baseline_val],
            bounds=(
                [peak_height * 0.5, t_fit[0], 0.01, 0.01, 0],
                [peak_height * 1.5, t_fit[-1], 10.0, 10.0, baseline_val + peak_height * 0.3]
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
# BATMAN DETECTION
# =============================================================================

def detect_batman(t, y, top_pct=0.20, min_valley_depth=MIN_VALLEY_DEPTH):
    """
    Detect if peak has Batman shape (valleys at top = detector artifact).

    ESTRICTE: Una vall només compta si té un pic ABANS i un pic DESPRÉS.
    Patró Batman = pic-vall-pic (les "orelles" del Batman).

    Parameters:
        t, y: time and signal arrays (peak segment)
        top_pct: fraction of peak height to analyze (0.20 = top 20%)
        min_valley_depth: minimum depth (as fraction of height) to count

    Returns:
        dict with is_batman, n_valleys, max_depth, valley info
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    baseline = float(np.percentile(y, 10))
    y_max = float(np.max(y))
    height = y_max - baseline

    if height <= 0:
        return {"is_batman": False, "n_valleys": 0, "max_depth": 0, "reason": "no_height"}

    # Define TOP region threshold
    threshold = baseline + height * (1 - top_pct)

    # Extract top region
    top_mask = y >= threshold
    t_top = t[top_mask]
    y_top = y[top_mask]

    if len(y_top) < 5:
        return {"is_batman": False, "n_valleys": 0, "max_depth": 0, "reason": "top_too_small"}

    # Find LOCAL MAXIMA (peaks) in top region
    peaks_top, _ = find_peaks(y_top, distance=2)

    # Find LOCAL MINIMA (valleys) by inverting signal
    y_inverted = -y_top
    valleys, _ = find_peaks(y_inverted, distance=2)

    if len(valleys) == 0:
        return {
            "is_batman": False,
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
            "is_batman": False,
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

    is_batman = n_significant > 0 and max_depth > min_valley_depth

    return {
        "is_batman": is_batman,
        "n_valleys": n_significant,
        "max_depth": max_depth,
        "t_top": t_top,
        "y_top": y_top,
        "valleys": valid_valleys,  # Només valls amb pic-vall-pic
        "valley_depths": valley_depths,
        "threshold": threshold,
        "reason": "batman_detected" if is_batman else "valleys_too_shallow"
    }


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
    HYBRID detection: Valleys (Batman) + Smoothness.

    Strategy:
    1. First check for valleys at top (clear Batman pattern)
    2. Then check smoothness (subtler irregularities)
    3. Combine both for final assessment

    Parameters:
        t, y: time and signal arrays (peak segment)
        top_pct: fraction of peak to analyze (0.25 = top 25%)
        min_valley_depth: minimum valley depth to count as Batman
        smoothness_threshold: below this = irregular

    Returns:
        dict with is_anomaly, anomaly_type, details
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    # 1. Check for valleys (Batman)
    batman = detect_batman(t, y, top_pct=top_pct, min_valley_depth=min_valley_depth)

    # 2. Check smoothness
    smooth = calc_top_smoothness(t, y, top_pct=top_pct)

    # 3. Combine results
    is_batman = batman.get("is_batman", False)
    smoothness = smooth.get("smoothness", 100.0)

    # IRR: només si smoothness < threshold (ex: 18%)
    is_irregular = smoothness < smoothness_threshold

    # Determine anomaly type
    if is_batman:
        anomaly_type = "BATMAN"
        is_anomaly = True
    elif is_irregular:
        anomaly_type = "IRREGULAR"
        is_anomaly = True
    else:
        anomaly_type = "OK"
        is_anomaly = False

    return {
        "is_anomaly": is_anomaly,
        "anomaly_type": anomaly_type,
        "smoothness": smoothness,
        "is_batman": is_batman,
        "is_irregular": is_irregular,
        "n_valleys": batman.get("n_valleys", 0),
        "max_valley_depth": batman.get("max_depth", 0),
        "batman_info": batman,
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


def repair_with_parabola(t, y, factor=REPAIR_FACTOR):
    """
    Repair Batman peak using parabola interpolation.

    1. Detect Batman pattern
    2. Calculate tangent intersection for theoretical peak
    3. Apply correction factor
    4. Fit parabola through anchors and theoretical max
    5. Replace deformed top with parabola

    Parameters:
        t, y: time and signal arrays (peak segment)
        factor: correction factor for tangent height

    Returns:
        tuple (y_repaired, repair_info, was_repaired)
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    # Detect Batman
    batman = detect_batman(t, y)

    if not batman["is_batman"]:
        return y.copy(), batman, False

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

    repair_info = {
        "batman": batman,
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

    return y_repaired, repair_info, True


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

    t_common = np.linspace(t_min, t_max, 500)
    y1_interp = np.interp(t_common, t1, y1)
    y2_interp = np.interp(t_common, t2, y2)

    try:
        r, _ = pearsonr(y1_interp, y2_interp)
        return r
    except:
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
    batman = detect_batman(t, y)
    if batman["is_batman"]:
        return None

    tangent = find_tangents_and_anchors(t, y)
    if tangent is None:
        return None

    y_tangent = tangent["y_intersect"]
    y_real = float(np.max(y))

    if y_tangent <= 0:
        return None

    factor = y_real / y_tangent

    if factor < 0.5 or factor > 1.2:
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
    },
}


def detect_timeout(t_min, threshold_sec=None, major_threshold_sec=None):
    """
    Detecta timeouts en dades DOC basant-se en la cadència temporal.

    MÈTODE MÉS ROBUST per detectar pauses del TOC (recàrrega xeringues).
    Un timeout es detecta quan l'interval entre mesures supera el llindar.

    Args:
        t_min: Array de temps en minuts
        threshold_sec: Llindar per considerar timeout (defecte: 60s)
        major_threshold_sec: Llindar per timeout major/recàrrega (defecte: 70s)

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
    zone_counts = {zone: 0 for zone in TIMEOUT_CONFIG["zones"].keys()}
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
        for zone_name, (t_ini, t_fi) in TIMEOUT_CONFIG["zones"].items():
            if zone_name == "RUN_START":
                continue  # Tractem apart
            if t_ini <= t_start < t_fi:
                zone = zone_name
                break

        # Cas especial: timeout a l'inici del run (t < 1 min)
        if t_start < 1.0:
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

    return {
        "n_timeouts": len(timeouts),
        "n_major_timeouts": n_major,
        "timeouts": timeouts,
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


# =============================================================================
# PEAK DETECTION
# =============================================================================

def detect_main_peak(t, y, min_prominence_pct=5.0):
    """
    Detecta el pic principal d'un cromatograma.

    Troba el pic més alt amb prominència suficient i retorna els seus límits.
    Usat per:
    - KHP Column: integrar NOMÉS el pic principal
    - Identificar zona d'interès

    Args:
        t: Array de temps
        y: Array de senyal
        min_prominence_pct: Prominència mínima com a % del rang (defecte: 5%)

    Returns:
        Dict amb:
            - valid: bool
            - t_max: temps al màxim
            - peak_idx: índex del màxim
            - left_idx, right_idx: límits del pic
            - t_start, t_end: temps dels límits
            - area: àrea del pic
            - height: altura del pic
            - fallback: True si s'ha usat màxim global
    """
    if t is None or y is None or len(t) < 20:
        return {"valid": False}

    y = np.asarray(y)
    t = np.asarray(t)

    # Calcular prominència mínima
    y_range = np.max(y) - np.min(y)
    min_prominence = y_range * min_prominence_pct / 100.0

    # Trobar pics
    peaks, properties = find_peaks(y, prominence=min_prominence, width=3)

    if len(peaks) == 0:
        # Fallback: usar màxim global
        peak_idx = int(np.argmax(y))
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
            "fallback": True,
        }

    # Seleccionar pic més alt
    peak_heights = y[peaks]
    main_peak_local_idx = int(np.argmax(peak_heights))
    peak_idx = peaks[main_peak_local_idx]

    # Obtenir límits del pic
    left_bases = properties.get("left_bases", [0] * len(peaks))
    right_bases = properties.get("right_bases", [len(y) - 1] * len(peaks))

    left_idx = int(left_bases[main_peak_local_idx])
    right_idx = int(right_bases[main_peak_local_idx])

    # Assegurar límits vàlids
    left_idx = max(0, left_idx)
    right_idx = min(len(y) - 1, right_idx)

    # Calcular àrea
    area = float(trapezoid(y[left_idx:right_idx+1], t[left_idx:right_idx+1]))

    return {
        "valid": True,
        "t_max": float(t[peak_idx]),
        "t_start": float(t[left_idx]),
        "t_end": float(t[right_idx]),
        "peak_idx": peak_idx,
        "left_idx": left_idx,
        "right_idx": right_idx,
        "area": area,
        "height": float(y[peak_idx]),
        "fallback": False,
    }


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
