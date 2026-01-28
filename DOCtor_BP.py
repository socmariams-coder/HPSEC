# -*- coding: utf-8 -*-
"""
DOCtor_BP v1.7 - HPSEC Chromatogram Anomaly Detection for BP Samples
====================================================================

Validation criteria for BP (Bypass) samples:
1. PEARSON correlation between replicates (if duplicates exist)
2. NUMBER OF PEAKS (2 peaks = warning, mismatch = problem)
3. BI-GAUSSIAN FIT to baseline → R² thresholds:
   - R² ≥ 0.987 → VALID (green)
   - 0.980 ≤ R² < 0.987 → CHECK (orange)
   - R² < 0.980 → INVALID (red)
4. BATMAN PEAK detection and REPAIR:
   - Detects valleys at peak top (detector artifacts)
   - Applies parabola repair using tangent intersection
   - Reports dH before/after repair
5. AREA DIFFERENCE between replicates
6. SNR (Signal-to-Noise Ratio)
7. ASYMMETRY FILTER (0.33 - 3.0)

Selection: Replica with highest R² (3 decimal comparison), SNR as tiebreaker
If no valid replica (R² < 0.980): No selection

v1.7 - Uses hpsec_replica for replica selection
v1.6 - Uses hpsec_core for shared functions
"""

import os
import re
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import find_peaks

# Import shared functions from core
from hpsec_core import (
    bigaussian, fit_bigaussian, check_asymmetry,
    detect_batman, repair_with_parabola, find_tangents_and_anchors,
    calc_snr, calc_peak_area, calc_pearson, calibrate_factor,
    THRESH_R2_VALID, THRESH_R2_CHECK, REPAIR_MIN_R2, REPAIR_FACTOR,
    ASYM_MIN, ASYM_MAX, THRESH_SNR, MIN_VALLEY_DEPTH
)
from hpsec_replica import select_best_replica, compare_replicas


# =============================================================================
# BP-SPECIFIC PARAMETERS
# =============================================================================

# Noise - below this value, don't analyze
NOISE_THRESHOLD = 20.0  # mAU

# Pearson - threshold for problem
THRESH_PEARSON = 0.995

# Area difference between replicates (%) - above this = problem
THRESH_AREA_DIFF_PCT = 15.0


# =============================================================================
# BP-SPECIFIC HELPER FUNCTIONS (wrappers around hpsec_core)
# =============================================================================

def fit_bigaussian_to_baseline(t, y, peak_idx, left_idx, right_idx):
    """Wrapper for fit_bigaussian from hpsec_core."""
    return fit_bigaussian(t, y, peak_idx, left_idx, right_idx)


def detect_jagged_peak(t, y, peak_idx, left_idx, right_idx, height_pct=0.80):
    """
    Detect if a peak has irregular/jagged top (Batman pattern).

    Uses valley-based detection: looks for LOCAL MINIMA at the peak top.
    Valleys indicate "Batman ears" pattern caused by detector artifacts.

    Parameters:
        t, y: time and signal arrays
        peak_idx: index of main peak
        left_idx, right_idx: peak boundaries
        height_pct: fraction of peak height to analyze (0.80 = top 20%)

    Returns:
        dict with is_jagged, n_valleys, max_depth, and details
    """
    t_seg = np.asarray(t[left_idx:right_idx+1], dtype=float)
    y_seg = np.asarray(y[left_idx:right_idx+1], dtype=float)

    if len(t_seg) < 10:
        return {"is_jagged": False, "n_valleys": 0, "max_depth": 0.0, "reason": "too_short"}

    # Use Batman detection (valley-based)
    batman = detect_batman(t_seg, y_seg, top_pct=(1.0 - height_pct))

    # Also get calibration data for normal peaks
    calib = calibrate_factor(t_seg, y_seg) if not batman["is_batman"] else None

    return {
        "is_jagged": batman["is_batman"],
        "is_batman": batman["is_batman"],
        "n_valleys": batman.get("n_valleys", 0),
        "max_depth": batman.get("max_depth", 0.0),
        "reason": batman.get("reason", "unknown"),
        "batman_info": batman,
        "calibration": calib,
    }


# =============================================================================
# EXCEL READING
# =============================================================================

def read_excel_bp(file_path):
    """Read DOC sheet for BP samples."""
    try:
        df = pd.read_excel(file_path, sheet_name="DOC")
    except:
        try:
            df = pd.read_excel(file_path, sheet_name=0)
        except Exception as e:
            return None, None, str(e)

    df.columns = [str(c).strip().lower() for c in df.columns]

    tcol = next((c for c in df.columns if "time" in c), None)
    ycol = next((c for c in df.columns if c != tcol and ("doc" in c or "toc" in c)), None)

    if not tcol or not ycol:
        return None, None, "time/doc columns not found"

    t = pd.to_numeric(df[tcol], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[ycol], errors="coerce").to_numpy(dtype=float)

    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]

    if t.size < 10:
        return None, None, "Insufficient data"

    idx = np.argsort(t)
    return t[idx], y[idx], None


# =============================================================================
# MAIN PEAK ANALYSIS (BP = 1 peak expected)
# =============================================================================

def analyze_bp_peak(t, y, apply_repair=True):
    """
    Analyze main peak for BP samples.
    BP should have 1 single Gaussian peak.

    If Batman pattern is detected and apply_repair=True,
    the peak is repaired using parabola interpolation.

    Returns:
        dict with all peak info including original and repaired data
    """
    if t is None or len(t) < 20:
        return None

    t = np.asarray(t, float)
    y = np.asarray(y, float)
    y_original = y.copy()

    # Baseline
    baseline = float(np.percentile(y, 10))
    max_signal = float(np.max(y))

    if max_signal < NOISE_THRESHOLD:
        return {
            "status": "LOW_SIGNAL",
            "max_mau": max_signal,
            "n_peaks": 0,
            "r2": 0.0,
            "r2_status": "INVALID",
            "snr": 0.0
        }

    # Find peaks
    min_prom = (max_signal - baseline) * 0.05
    peaks, props = find_peaks(y, prominence=min_prom, width=2)

    n_peaks = len(peaks)

    if n_peaks == 0:
        return {
            "status": "NO_PEAK",
            "max_mau": max_signal,
            "n_peaks": 0,
            "r2": 0.0,
            "r2_status": "INVALID",
            "snr": 0.0
        }

    # Get the most prominent peak
    main_idx = np.argmax(props["prominences"])
    pk_idx = int(peaks[main_idx])
    left_base = int(props["left_bases"][main_idx])
    right_base = int(props["right_bases"][main_idx])

    # Extract peak segment for analysis
    t_seg = t[left_base:right_base+1]
    y_seg = y[left_base:right_base+1]

    # Detect Batman pattern (valleys at top) on original segment
    jagged = detect_jagged_peak(t, y, pk_idx, left_base, right_base)
    is_batman = jagged.get("is_batman", False)
    n_valleys = jagged.get("n_valleys", 0)
    max_depth = jagged.get("max_depth", 0.0)

    # === APPLY REPAIR IF BATMAN DETECTED ===
    was_repaired = False
    repair_info = None
    repair_skipped_reason = None
    y_repaired = None
    height_original = float(y[pk_idx]) - baseline
    height_repaired = height_original

    # First, get preliminary R² to decide if repair should be attempted
    gauss_preliminary = fit_bigaussian_to_baseline(t, y, pk_idx, left_base, right_base)
    r2_preliminary = gauss_preliminary.get("r2", 0.0)

    if is_batman and apply_repair:
        # CHECK: Don't repair if R² is too low (garbage peak)
        if r2_preliminary < REPAIR_MIN_R2:
            repair_skipped_reason = f"R²={r2_preliminary:.3f} < {REPAIR_MIN_R2}"
        else:
            # Apply parabola repair to the peak segment
            y_seg_repaired, repair_info, was_repaired = repair_with_parabola(t_seg, y_seg)

            if was_repaired:
                # Update y with repaired segment
                y_repaired = y.copy()
                y_repaired[left_base:right_base+1] = y_seg_repaired
                height_repaired = float(np.max(y_seg_repaired)) - baseline

                # Use repaired signal for subsequent analysis
                y = y_repaired

    # Peak info (use possibly repaired height)
    t_peak = float(t[pk_idx])
    height = height_repaired if was_repaired else height_original

    # SNR (calculated on original to avoid bias)
    snr = calc_snr(y_original, height_original)

    # Peak area (on possibly repaired data)
    area = calc_peak_area(t, y, left_base, right_base, baseline)
    area_original = calc_peak_area(t, y_original, left_base, right_base, baseline)

    # Bi-Gaussian fit on ORIGINAL data (to assess original quality)
    gauss_original = fit_bigaussian_to_baseline(t, y_original, pk_idx, left_base, right_base)
    r2_original = gauss_original.get("r2", 0.0)

    # Bi-Gaussian fit on repaired data (if repaired)
    if was_repaired:
        gauss_repaired = fit_bigaussian_to_baseline(t, y_repaired, pk_idx, left_base, right_base)
        r2_repaired = gauss_repaired.get("r2", 0.0)
        # Use the better R² for status determination
        r2 = max(r2_original, r2_repaired)
        gauss = gauss_repaired if r2_repaired >= r2_original else gauss_original
    else:
        r2 = r2_original
        gauss = gauss_original
        r2_repaired = None
        gauss_repaired = None

    # Check asymmetry from bi-Gaussian fit
    asymmetry = gauss.get("asymmetry", 1.0)
    asymmetry_ok = ASYM_MIN <= asymmetry <= ASYM_MAX

    # Determine R² status
    if r2 >= THRESH_R2_VALID:
        r2_status = "VALID"
    elif r2 >= THRESH_R2_CHECK:
        r2_status = "CHECK"
    else:
        r2_status = "INVALID"

    # Determine overall status
    warnings = []

    if n_peaks == 2:
        warnings.append("2_PEAKS")
    elif n_peaks > 2:
        warnings.append("MULTIPLE_PEAKS")

    if is_batman:
        if was_repaired:
            repair_str = "REP"
        elif repair_skipped_reason:
            repair_str = "SKIP"
        else:
            repair_str = "NO-REP"
        warnings.append(f"BATMAN({n_valleys}v,{max_depth*100:.1f}%,{repair_str})")

    if snr < THRESH_SNR:
        warnings.append("LOW_SNR")

    # Asymmetry check
    if not asymmetry_ok:
        warnings.append(f"ASYM={asymmetry:.2f}")

    # Determine final status
    if r2_status == "INVALID":
        status = "INVALID"
    elif not asymmetry_ok:
        status = "INVALID"  # Extreme asymmetry = detector error
    elif r2_status == "CHECK":
        status = "CHECK"
    elif warnings:
        status = "WARNING"
    else:
        status = "VALID"

    return {
        "status": status,
        "r2_status": r2_status,
        "warnings": warnings,
        "n_peaks": n_peaks,
        "t_peak": t_peak,
        "height": height,
        "height_original": height_original,
        "height_repaired": height_repaired if was_repaired else None,
        "area": area,
        "area_original": area_original,
        "r2": r2,
        "r2_original": r2_original,
        "r2_repaired": r2_repaired,
        "snr": snr,
        "gauss_fit": gauss,
        "gauss_fit_original": gauss_original,
        "gauss_fit_repaired": gauss_repaired,
        "baseline": baseline,
        "max_mau": max_signal,
        "left_base": left_base,
        "right_base": right_base,
        "peak_idx": pk_idx,
        "jagged": jagged,
        "is_batman": is_batman,
        "was_repaired": was_repaired,
        "repair_info": repair_info,
        "repair_skipped_reason": repair_skipped_reason,
        "y_original": y_original,
        "y_repaired": y_repaired,
        "asymmetry": asymmetry,
        "asymmetry_ok": asymmetry_ok,
        "calibration": jagged.get("calibration"),
    }


# =============================================================================
# REPLICATE GROUPING
# =============================================================================

_REP_PAT = re.compile(r"(?i)(?:_|\b)(rep(?:lica)?|r)\s*([12])(?:\b|_)")

def sample_key_from_filename(fname):
    """Extract sample name and replica ID from filename."""
    base = os.path.splitext(os.path.basename(fname))[0]
    rep_id = None
    m = _REP_PAT.search(base)
    if m:
        rep_id = m.group(2)
        base = _REP_PAT.sub("", base)
    return re.sub(r"__+", "_", base).strip("_ "), rep_id


# =============================================================================
# BP PROCESSING
# =============================================================================

def process_bp_samples(base_dir, selected_seqs, progress_cb=None):
    """Process BP samples and generate PDF."""

    date_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    pdf_path = os.path.join(base_dir, f"REPORT_BP_{date_str}.pdf")

    all_samples = []
    calibration_data = []  # Collect calibration factors from normal peaks

    with PdfPages(pdf_path) as pdf:

        for seq_idx, seq_path in enumerate(selected_seqs):
            seq_name = os.path.basename(seq_path)
            if progress_cb:
                progress_cb(seq_idx + 1, len(selected_seqs), seq_name)

            results_folder = os.path.join(seq_path, "Resultats_Consolidats")
            if not os.path.isdir(results_folder):
                continue

            # Only BP files
            files = [os.path.join(results_folder, f) for f in os.listdir(results_folder)
                     if f.lower().endswith(".xlsx") and "bp" in f.lower()]

            if not files:
                continue

            # Group by sample
            groups = {}
            for f in files:
                key, rep = sample_key_from_filename(os.path.basename(f))
                groups.setdefault(key, []).append((rep or "?", f))

            for k in groups:
                groups[k].sort(key=lambda x: (x[0] != "1", x[0] != "2", x[0]))

            # Process each sample
            for sample_key, items in groups.items():
                sample = {
                    "seq": seq_name,
                    "sample": sample_key,
                    "n_reps": len(items),
                    "pearson": np.nan,
                    "area_diff_pct": np.nan,
                    "status": "VALID",
                    "warnings": [],
                    "reps": [],
                    "best_rep": None
                }

                rep_data = []
                for rep_id, fpath in items[:2]:
                    t, y, err = read_excel_bp(fpath)

                    if err:
                        rep_data.append({
                            "rep": rep_id,
                            "error": err,
                            "r2": 0.0,
                            "r2_status": "INVALID"
                        })
                        continue

                    # Analyze BP peak
                    analysis = analyze_bp_peak(t, y)

                    rep_data.append({
                        "rep": rep_id,
                        "t": t,
                        "y": y,
                        "analysis": analysis,
                        "r2": analysis.get("r2", 0.0) if analysis else 0.0,
                        "r2_status": analysis.get("r2_status", "INVALID") if analysis else "INVALID"
                    })

                    # Collect calibration data from normal peaks
                    if analysis and analysis.get("calibration"):
                        calibration_data.append({
                            "seq": seq_name,
                            "sample": sample_key,
                            "rep": rep_id,
                            **analysis["calibration"]
                        })

                # === COMPARE REPLICATES ===
                if len(rep_data) >= 2 and all(r.get("t") is not None for r in rep_data[:2]):
                    # Pearson (on original signals, before repair)
                    y1_orig = rep_data[0].get("analysis", {}).get("y_original", rep_data[0]["y"])
                    y2_orig = rep_data[1].get("analysis", {}).get("y_original", rep_data[1]["y"])
                    sample["pearson"] = calc_pearson(
                        rep_data[0]["t"], y1_orig,
                        rep_data[1]["t"], y2_orig
                    )

                    # Height comparison (use repaired heights if available)
                    h1 = rep_data[0].get("analysis", {}).get("height", 0)
                    h2 = rep_data[1].get("analysis", {}).get("height", 0)
                    h1_orig = rep_data[0].get("analysis", {}).get("height_original", h1)
                    h2_orig = rep_data[1].get("analysis", {}).get("height_original", h2)

                    # Calculate dH original and repaired
                    if h1_orig > 0:
                        sample["dh_original"] = (h2_orig - h1_orig) / h1_orig * 100
                    else:
                        sample["dh_original"] = np.nan

                    if h1 > 0:
                        sample["dh_repaired"] = (h2 - h1) / h1 * 100
                    else:
                        sample["dh_repaired"] = sample["dh_original"]

                    # Area difference (use repaired area)
                    a1 = rep_data[0].get("analysis", {}).get("area", 0)
                    a2 = rep_data[1].get("analysis", {}).get("area", 0)
                    if a1 > 0 and a2 > 0:
                        avg_area = (a1 + a2) / 2
                        sample["area_diff_pct"] = abs(a1 - a2) / avg_area * 100

                    # Peak count
                    n1 = rep_data[0].get("analysis", {}).get("n_peaks", 0)
                    n2 = rep_data[1].get("analysis", {}).get("n_peaks", 0)
                    if n1 != n2:
                        sample["warnings"].append("PEAK_MISMATCH")

                    # Check if repair improved dH
                    if not np.isnan(sample.get("dh_original", np.nan)):
                        dh_orig = abs(sample["dh_original"])
                        dh_rep = abs(sample.get("dh_repaired", dh_orig))
                        if dh_rep < dh_orig:
                            sample["repair_improved"] = True
                        else:
                            sample["repair_improved"] = False

                # === CHECK FOR 2 PEAKS WARNING ===
                for rep in rep_data:
                    if rep.get("analysis", {}).get("n_peaks", 0) == 2:
                        if "2_PEAKS" not in sample["warnings"]:
                            sample["warnings"].append("2_PEAKS")

                # === CHECK FOR BATMAN PEAK WARNING ===
                for rep in rep_data:
                    analysis = rep.get("analysis", {})
                    if analysis.get("is_batman", False):
                        n_valleys = analysis.get("jagged", {}).get("n_valleys", 0)
                        warn = f"BATMAN_R{rep.get('rep', '?')}"
                        if warn not in sample["warnings"]:
                            sample["warnings"].append(warn)

                # === SELECT BEST REPLICA via hpsec_replica (v1.7) ===
                # Crear evals compatibles amb hpsec_replica
                def make_eval(rep):
                    analysis = rep.get("analysis", {})
                    r2_status = rep.get("r2_status", "INVALID")
                    return {
                        "valid": r2_status in ["VALID", "CHECK"],
                        "r2": rep.get("r2", 0),
                        "r2_status": r2_status,
                        "snr": analysis.get("snr", 0) if analysis else 0,
                        "height": analysis.get("height", 0) if analysis else 0,
                        "batman": analysis.get("is_batman", False) if analysis else False,
                        "timeout": False,
                        "irr": False
                    }

                eval1 = make_eval(rep_data[0]) if len(rep_data) > 0 else None
                eval2 = make_eval(rep_data[1]) if len(rep_data) > 1 else None

                selection = select_best_replica(eval1, eval2, method="BP")

                if selection["best"]:
                    sample["best_rep"] = selection["best"][1]  # "R1" -> "1"
                    sample["selection_reason"] = selection["reason"]
                else:
                    sample["best_rep"] = None
                    sample["selection_reason"] = selection.get("warning", "Cap rèplica vàlida")

                # === DETERMINE SAMPLE STATUS ===
                r2_statuses = [r.get("r2_status", "INVALID") for r in rep_data]
                n_reps = len(rep_data)

                # Single replica handling
                is_single_rep = n_reps == 1
                if is_single_rep:
                    sample["warnings"].append("SINGLE_REP")

                if all(s == "INVALID" for s in r2_statuses):
                    sample["status"] = "INVALID"
                elif "INVALID" in r2_statuses and n_reps >= 2:
                    sample["status"] = "ONE_INVALID"
                elif is_single_rep:
                    # Single replica: cannot be VALID without verification
                    # Best case is CHECK if R² is good
                    if r2_statuses[0] == "VALID":
                        sample["status"] = "CHECK"  # Downgrade from VALID to CHECK
                    else:
                        sample["status"] = r2_statuses[0]  # Keep INVALID or CHECK
                elif all(s == "CHECK" for s in r2_statuses):
                    sample["status"] = "CHECK"
                elif "CHECK" in r2_statuses:
                    sample["status"] = "CHECK"
                elif not np.isnan(sample["pearson"]) and sample["pearson"] < THRESH_PEARSON:
                    sample["status"] = "LOW_PEARSON"
                    sample["warnings"].append("LOW_PEARSON")
                elif not np.isnan(sample["area_diff_pct"]) and sample["area_diff_pct"] > THRESH_AREA_DIFF_PCT:
                    sample["status"] = "AREA_DIFF"
                    sample["warnings"].append("AREA_DIFF")
                elif sample["warnings"]:
                    sample["status"] = "WARNING"
                else:
                    sample["status"] = "VALID"

                sample["reps"] = rep_data
                all_samples.append(sample)

            # === GENERATE PAGES ===
            samples_to_plot = [s for s in all_samples if s["seq"] == seq_name]

            for i in range(0, len(samples_to_plot), 2):
                page_samples = samples_to_plot[i:i+2]

                fig, axes = plt.subplots(len(page_samples), 2, figsize=(11, 4*len(page_samples)))
                fig.suptitle(f"BP - {seq_name}", fontsize=12, fontweight="bold")

                if len(page_samples) == 1:
                    axes = axes.reshape(1, -1)

                for row, sample in enumerate(page_samples):
                    # Status color
                    status_colors = {
                        "VALID": "green",
                        "CHECK": "orange",
                        "WARNING": "orange",
                        "ONE_INVALID": "orange",
                        "INVALID": "red",
                        "LOW_PEARSON": "red",
                        "AREA_DIFF": "orange"
                    }
                    status_color = status_colors.get(sample["status"], "black")

                    # Sample info
                    pears_str = f"ρ={sample['pearson']:.3f}" if not np.isnan(sample['pearson']) else "-"
                    area_str = f"ΔA={sample['area_diff_pct']:.1f}%" if not np.isnan(sample['area_diff_pct']) else "-"
                    warn_str = " | ".join(sample["warnings"]) if sample["warnings"] else ""

                    for col in range(2):
                        ax = axes[row, col]

                        if col < len(sample["reps"]):
                            rep = sample["reps"][col]

                            if rep.get("t") is not None:
                                t_data = rep["t"]
                                y_data = rep["y"]
                                analysis = rep.get("analysis", {})
                                r2 = rep.get("r2", 0)
                                r2_status = rep.get("r2_status", "INVALID")
                                is_selected = rep["rep"] == sample["best_rep"]

                                # Check if repair was applied
                                was_repaired = analysis.get("was_repaired", False)
                                y_original = analysis.get("y_original", y_data)
                                y_repaired = analysis.get("y_repaired", None)

                                # Line color
                                line_color = "#2E86AB" if col == 0 else "#A23B72"

                                # Plot ORIGINAL signal (always)
                                if was_repaired:
                                    ax.plot(t_data, y_original, color=line_color, linewidth=0.8,
                                           alpha=0.5, linestyle=":", label="Original")
                                    # Plot REPAIRED signal
                                    ax.plot(t_data, y_repaired, color=line_color, linewidth=1.2,
                                           label="Repaired")
                                else:
                                    ax.plot(t_data, y_data, color=line_color, linewidth=1, label="Signal")

                                # Plot Gaussian fit
                                gauss = analysis.get("gauss_fit", {})
                                if gauss.get("t_fit") is not None:
                                    if r2_status == "VALID":
                                        fit_color = "green"
                                    elif r2_status == "CHECK":
                                        fit_color = "orange"
                                    else:
                                        fit_color = "red"

                                    ax.plot(gauss["t_fit"], gauss["y_fit"],
                                           color=fit_color, linestyle="--", linewidth=1.5,
                                           alpha=0.8, label=f"BiGauss R²={r2:.3f}")

                                # Mark repair zone if repaired
                                if was_repaired:
                                    repair_info = analysis.get("repair_info", {})
                                    if repair_info and "tangent" in repair_info:
                                        t1 = repair_info["t_anchor_left"]
                                        t3 = repair_info["t_anchor_right"]
                                        ax.axvspan(t1, t3, alpha=0.15, color="yellow",
                                                  label="Repair zone")
                                        # Mark anchor points
                                        ax.scatter([t1, t3],
                                                  [repair_info["y_anchor_left"], repair_info["y_anchor_right"]],
                                                  c="orange", s=40, zorder=5, marker="o",
                                                  edgecolors="darkorange", linewidths=1)

                                # Mark peak
                                if analysis.get("t_peak"):
                                    ax.axvline(analysis["t_peak"], color="gray",
                                              linestyle=":", alpha=0.5)

                                # Replica title with repair info
                                n_peaks = analysis.get("n_peaks", 0)
                                snr = analysis.get("snr", 0)
                                r2_orig = analysis.get("r2_original", r2)

                                # Selection mark
                                if is_selected:
                                    sel_mark = " * SELECTED"
                                else:
                                    sel_mark = ""

                                # Build title with BEFORE/AFTER info
                                if was_repaired:
                                    r2_before = analysis.get("r2_original", r2)
                                    r2_after = analysis.get("r2_repaired", r2)
                                    h_orig = analysis.get("height_original", 0)
                                    h_rep = analysis.get("height_repaired", 0)
                                    dh_height = (h_rep - h_orig) / h_orig * 100 if h_orig > 0 else 0
                                    title = f"R{rep['rep']}{sel_mark} | R²: {r2_before:.3f} -> {r2_after:.3f} | h: {dh_height:+.1f}%"
                                else:
                                    title = f"R{rep['rep']}{sel_mark} | R²={r2:.3f} [{r2_status}] | {n_peaks} peak(s) | SNR={snr:.1f}"

                                if r2_status == "VALID":
                                    title_color = "green"
                                elif r2_status == "CHECK":
                                    title_color = "orange"
                                else:
                                    title_color = "red"

                                ax.set_title(title, fontsize=9, color=title_color,
                                            fontweight="bold" if is_selected else "normal")
                                ax.set_xlabel("min", fontsize=8)
                                ax.set_ylabel("mAU", fontsize=8)
                                ax.tick_params(labelsize=7)
                                ax.grid(True, alpha=0.3)
                                ax.legend(fontsize=6, loc="upper right")

                                # Border color based on status and selection
                                if is_selected:
                                    border_color = "green" if r2_status == "VALID" else "orange"
                                    border_width = 3
                                else:
                                    border_color = "gray"
                                    border_width = 1

                                for spine in ax.spines.values():
                                    spine.set_edgecolor(border_color)
                                    spine.set_linewidth(border_width)
                            else:
                                ax.text(0.5, 0.5, f"Error: {rep.get('error', '?')}",
                                       ha="center", va="center", color="red",
                                       transform=ax.transAxes)
                                ax.axis("off")
                        else:
                            ax.text(0.5, 0.5, "No replicate", ha="center", va="center",
                                   fontsize=12, color="gray", transform=ax.transAxes)
                            ax.axis("off")

                    # Row title with dH info
                    selected_str = f"Selected: R{sample['best_rep']}" if sample['best_rep'] else "NO VALID REPLICA"

                    # dH info
                    dh_orig = sample.get("dh_original", np.nan)
                    dh_rep = sample.get("dh_repaired", np.nan)
                    if not np.isnan(dh_orig):
                        if not np.isnan(dh_rep) and abs(dh_rep) != abs(dh_orig):
                            dh_str = f"dH: {dh_orig:+.1f}% -> {dh_rep:+.1f}%"
                        else:
                            dh_str = f"dH: {dh_orig:+.1f}%"
                    else:
                        dh_str = ""

                    row_title = f"{sample['sample'][:25]}  [{sample['status']}]  {pears_str}  {dh_str}  {warn_str}"
                    axes[row, 0].annotate(row_title, xy=(0, 1.12), xycoords='axes fraction',
                                          fontsize=9, fontweight="bold", color=status_color)
                    axes[row, 1].annotate(selected_str, xy=(1, 1.12), xycoords='axes fraction',
                                          fontsize=9, ha="right",
                                          color="green" if sample['best_rep'] else "red")

                plt.tight_layout()
                pdf.savefig(fig, facecolor='white')
                plt.close(fig)

        # === SUMMARY TABLE ===
        if all_samples:
            # Sort: invalid first
            all_samples.sort(key=lambda x: (
                0 if x["status"] == "INVALID" else
                1 if x["status"] in ["CHECK", "WARNING", "ONE_INVALID"] else 2,
                x.get("best_rep") is None
            ))

            fig = plt.figure(figsize=(11, 8))
            ax = fig.add_subplot(111)
            ax.axis("off")
            ax.set_title("BP SUMMARY - Gaussian Validation", fontsize=14, fontweight="bold", pad=20)

            cols = ["SEQ", "Sample", "Sel", "Pearson", "dH_o", "dH_r", "R²1_o", "R²1_r", "R²2_o", "R²2_r"]
            rows = []
            colors = []

            for s in all_samples:
                # R² values before/after for each replicate
                rep1 = s["reps"][0] if len(s["reps"]) > 0 else {}
                rep2 = s["reps"][1] if len(s["reps"]) > 1 else {}

                r2_1_orig = rep1.get("analysis", {}).get("r2_original", rep1.get("r2", 0))
                r2_1_rep = rep1.get("analysis", {}).get("r2_repaired", None)
                r2_2_orig = rep2.get("analysis", {}).get("r2_original", rep2.get("r2", 0)) if rep2 else 0
                r2_2_rep = rep2.get("analysis", {}).get("r2_repaired", None) if rep2 else None

                # dH values
                dh_orig = s.get("dh_original", np.nan)
                dh_rep = s.get("dh_repaired", np.nan)

                # Format R² showing before->after if repaired
                def fmt_r2(r2_o, r2_r):
                    if r2_r is not None:
                        return f"{r2_o:.3f}->{r2_r:.3f}"
                    return f"{r2_o:.3f}"

                rows.append([
                    s["seq"][:8],
                    s["sample"][:12],
                    f"R{s['best_rep']}" if s['best_rep'] else "-",
                    f"{s['pearson']:.3f}" if not np.isnan(s['pearson']) else "-",
                    f"{dh_orig:+.1f}" if not np.isnan(dh_orig) else "-",
                    f"{dh_rep:+.1f}" if not np.isnan(dh_rep) else "-",
                    f"{r2_1_orig:.3f}",
                    f"{r2_1_rep:.3f}" if r2_1_rep is not None else "-",
                    f"{r2_2_orig:.3f}" if rep2 else "-",
                    f"{r2_2_rep:.3f}" if r2_2_rep is not None else "-",
                ])

                # Background color
                if s["status"] == "VALID":
                    bg = "#ccffcc"  # Green
                elif s["status"] in ["CHECK", "WARNING", "ONE_INVALID"]:
                    bg = "#fff3cd"  # Yellow/orange
                else:
                    bg = "#ffcccc"  # Red

                colors.append([bg] * len(cols))

            table = ax.table(cellText=rows, colLabels=cols, loc="center",
                            cellLoc="center", cellColours=colors)
            table.auto_set_font_size(False)
            table.set_fontsize(7)
            table.scale(1.1, 1.3)

            for j in range(len(cols)):
                table[(0, j)].set_facecolor("steelblue")
                table[(0, j)].set_text_props(color="white", fontweight="bold")

            plt.tight_layout()
            pdf.savefig(fig, facecolor='white')
            plt.close(fig)

        # === STATISTICS ===
        fig = plt.figure(figsize=(11, 6))

        ax1 = fig.add_subplot(121)
        n_valid = sum(1 for s in all_samples if s["status"] == "VALID")
        n_check = sum(1 for s in all_samples if s["status"] in ["CHECK", "WARNING", "ONE_INVALID"])
        n_invalid = sum(1 for s in all_samples if s["status"] == "INVALID")

        ax1.pie([n_valid, n_check, n_invalid],
               labels=["VALID", "CHECK/WARNING", "INVALID"],
               colors=["#66bb6a", "#ffb74d", "#ef5350"],
               autopct="%1.1f%%", startangle=90)
        ax1.set_title("BP Sample Validation")

        ax2 = fig.add_subplot(122)
        ax2.axis("off")

        # Summary by status
        status_counts = {}
        for s in all_samples:
            status_counts[s["status"]] = status_counts.get(s["status"], 0) + 1

        n_no_selection = sum(1 for s in all_samples if s["best_rep"] is None)

        # Repair statistics
        n_with_repair = sum(1 for s in all_samples
                          for r in s.get("reps", [])
                          if r.get("analysis", {}).get("was_repaired", False))
        n_repair_improved = sum(1 for s in all_samples if s.get("repair_improved") is True)
        n_repair_worsened = sum(1 for s in all_samples if s.get("repair_improved") is False)

        summary = f"BP SUMMARY\n{'='*35}\n\n"
        summary += f"Total samples: {len(all_samples)}\n"
        summary += f"VALID (R²≥0.987): {n_valid} ({100*n_valid/len(all_samples):.1f}%)\n"
        summary += f"CHECK/WARNING: {n_check} ({100*n_check/len(all_samples):.1f}%)\n"
        summary += f"INVALID (R²<0.980): {n_invalid} ({100*n_invalid/len(all_samples):.1f}%)\n\n"
        summary += f"No valid replica: {n_no_selection}\n\n"
        summary += f"REPAIR STATISTICS:\n"
        summary += f"  Peaks repaired: {n_with_repair}\n"
        summary += f"  dH improved:   {n_repair_improved}\n"
        summary += f"  dH worsened:   {n_repair_worsened}\n\n"
        summary += "By status:\n"
        for status, count in sorted(status_counts.items()):
            summary += f"  {status}: {count}\n"
        summary += f"\nThresholds:\n"
        summary += f"  R² VALID: ≥{THRESH_R2_VALID}\n"
        summary += f"  R² CHECK: ≥{THRESH_R2_CHECK}\n"
        summary += f"  Pearson: ≥{THRESH_PEARSON}\n"
        summary += f"  Repair factor: {REPAIR_FACTOR}\n"
        summary += f"\nSelection: R² (3 dec) + SNR tiebreaker\n"

        ax2.text(0.05, 0.5, summary, fontsize=10, family="monospace",
                verticalalignment="center", transform=ax2.transAxes,
                bbox=dict(boxstyle="round", facecolor="whitesmoke"))

        plt.tight_layout()
        pdf.savefig(fig, facecolor='white')
        plt.close(fig)

        # === CALIBRATION SUMMARY ===
        if calibration_data:
            fig = plt.figure(figsize=(11, 6))

            # Filter valid calibration data
            valid_calib = [c for c in calibration_data if 0.5 < c["factor"] < 1.2]

            if valid_calib:
                factors = [c["factor"] for c in valid_calib]

                ax1 = fig.add_subplot(121)
                ax1.hist(factors, bins=15, edgecolor='black', alpha=0.7, color='steelblue')
                ax1.axvline(np.mean(factors), color='red', ls='--', lw=2,
                           label=f'Mean: {np.mean(factors):.4f}')
                ax1.axvline(np.median(factors), color='green', ls='--', lw=2,
                           label=f'Median: {np.median(factors):.4f}')
                ax1.axvline(REPAIR_FACTOR, color='orange', ls=':', lw=2,
                           label=f'Current: {REPAIR_FACTOR}')
                ax1.set_xlabel('Factor (y_real / y_tangent)')
                ax1.set_ylabel('Frequency')
                ax1.set_title('Tangent Correction Factor\n(from normal peaks)')
                ax1.legend(fontsize=8)
                ax1.grid(True, alpha=0.3)

                ax2 = fig.add_subplot(122)
                ax2.axis("off")

                # Batman summary
                n_batman = sum(1 for s in all_samples
                              for r in s.get("reps", [])
                              if r.get("analysis", {}).get("is_batman", False))

                calib_summary = f"CALIBRATION & BATMAN\n{'='*35}\n\n"
                calib_summary += f"Normal peaks analyzed: {len(valid_calib)}\n"
                calib_summary += f"Batman peaks detected: {n_batman}\n\n"
                calib_summary += f"Correction Factor:\n"
                calib_summary += f"  Mean:    {np.mean(factors):.4f}\n"
                calib_summary += f"  Median:  {np.median(factors):.4f}\n"
                calib_summary += f"  Std:     {np.std(factors):.4f}\n"
                calib_summary += f"  Min:     {np.min(factors):.4f}\n"
                calib_summary += f"  Max:     {np.max(factors):.4f}\n\n"
                calib_summary += f"Current factor: {REPAIR_FACTOR}\n"
                calib_summary += f"Recommended:    {np.median(factors):.4f}\n\n"
                calib_summary += "Note: Factor = y_real / y_tangent\n"
                calib_summary += "Used to correct tangent intersection\n"
                calib_summary += "for theoretical peak height estimation."

                ax2.text(0.05, 0.5, calib_summary, fontsize=10, family="monospace",
                        verticalalignment="center", transform=ax2.transAxes,
                        bbox=dict(boxstyle="round", facecolor="lightyellow"))

                plt.suptitle("Peak Calibration Data", fontsize=12, fontweight="bold")
                plt.tight_layout()
                pdf.savefig(fig, facecolor='white')
                plt.close(fig)

    return pdf_path, all_samples


# =============================================================================
# GUI
# =============================================================================

class DOCtorBPApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("DOCtor BP v1.6 - BiGaussian + Batman Repair")
        self.root.geometry("420x300")
        self.root.resizable(False, False)

        main = ttk.Frame(self.root, padding=20)
        main.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main, text="DOCtor BP", font=("Segoe UI", 18, "bold")).pack()
        ttk.Label(main, text="Bypass Sample Validation", foreground="gray").pack()

        # Thresholds info
        thresh_frame = ttk.LabelFrame(main, text="R² Thresholds", padding=5)
        thresh_frame.pack(pady=10, fill="x")
        ttk.Label(thresh_frame, text=f"✓ VALID: R² ≥ {THRESH_R2_VALID}", foreground="green").pack(anchor="w")
        ttk.Label(thresh_frame, text=f"⚠ CHECK: {THRESH_R2_CHECK} ≤ R² < {THRESH_R2_VALID}", foreground="orange").pack(anchor="w")
        ttk.Label(thresh_frame, text=f"✗ INVALID: R² < {THRESH_R2_CHECK}", foreground="red").pack(anchor="w")

        self.btn = ttk.Button(main, text="Select folder...", command=self._run)
        self.btn.pack(pady=10, ipadx=15, ipady=8)

        self.progress = tk.StringVar(value="")
        ttk.Label(main, textvariable=self.progress, font=("Segoe UI", 9)).pack(pady=5)

        self.pbar = ttk.Progressbar(main, length=300, mode='determinate')
        self.pbar.pack()

    def _run(self):
        base_dir = filedialog.askdirectory(title="Root folder with SEQs")
        if not base_dir:
            return

        # Find SEQs
        seq_folders = sorted([
            os.path.join(base_dir, d) for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d, "Resultats_Consolidats"))
        ])

        if not seq_folders:
            messagebox.showwarning("Warning", "No SEQs found with Resultats_Consolidats")
            return

        self.btn.config(state="disabled")
        self.pbar["maximum"] = len(seq_folders)

        def progress(i, n, name):
            self.progress.set(f"{i}/{n}: {name}")
            self.pbar["value"] = i
            self.root.update()

        try:
            pdf_path, samples = process_bp_samples(base_dir, seq_folders, progress)

            n_valid = sum(1 for s in samples if s["status"] == "VALID")
            n_total = len(samples)
            n_no_sel = sum(1 for s in samples if s["best_rep"] is None)

            self.progress.set(f"Done! {n_valid}/{n_total} valid")
            messagebox.showinfo("Complete",
                f"PDF: {os.path.basename(pdf_path)}\n"
                f"{n_total} samples processed\n"
                f"{n_valid} VALID ({100*n_valid/n_total:.1f}%)\n"
                f"{n_no_sel} with no valid replica")
            os.startfile(base_dir)

        except Exception as e:
            messagebox.showerror("Error", str(e))
            import traceback
            traceback.print_exc()
        finally:
            self.btn.config(state="normal")
            self.pbar["value"] = 0

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    DOCtorBPApp().run()
