"""
Reprocessament KHP BP amb la pipeline d'integració actual de la Suite.

Llegeix MasterFiles BP, extreu DOC per cada injecció KHP via 4-TOC_CALC,
i integra amb detect_main_peak() + find_peak_boundaries().
"""
import sys
import os
import json
import re
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from hpsec_import import extract_khp_conc, is_khp, llegir_masterfile_nou
from hpsec_core import detect_main_peak, find_peak_boundaries, get_baseline_value, get_baseline_stats

DADES = "C:/Users/Lequia/Desktop/Dades3"

# Sequences to process
BP_SEQS = [
    "114_SEQ_BP_CAL",
    "152_SEQ_BP_CAL",
    "156_SEQ_BP_CAL",
    "205_SEQ_BP_CAL",
    "292_SEQ_CAL_BP",
    "268_SEQ_BP",
    "270_SEQ_BP",
    "273_SEQ_BP",
    "277_SEQ_BP",
    "279B_SEQ_BP",
    "281_SEQ_BP",
    "284_SEQ_BP",
    "286_SEQ_BP",
    "287_SEQ_BP",
    "289_SEQ_BP",
]


def _parse_conc_from_name(name):
    """Parse concentration from sample names like '5PPM100UL_R1', '0.5PPM_R1', '250PPB_R2'."""
    name_upper = name.upper().split("_R")[0]  # Remove replica suffix
    # Try: NUMBERppm
    m = re.search(r'(\d+\.?\d*)\s*PPM', name_upper)
    if m:
        return float(m.group(1))
    # Try: NUMBERppb
    m = re.search(r'(\d+\.?\d*)\s*PPB', name_upper)
    if m:
        return float(m.group(1)) / 1000.0
    # Try: just a number at start
    m = re.match(r'(\d+\.?\d*)', name_upper)
    if m:
        val = float(m.group(1))
        if val >= 100:
            return val / 1000.0  # Assume ppb
        return val
    return 0.0


def _parse_vol_from_name(name):
    """Parse volume from sample names like '5PPM100UL_R1' or 'KHP2_50_R1'."""
    # Explicit: NUMBERul
    m = re.search(r'(\d+)\s*UL', name.upper())
    if m:
        return float(m.group(1))
    # Convention: KHPx_VOL_Rn where VOL is 25/50/75/100/150/200/400
    m = re.search(r'KHP\d+[._](\d+)[._]R\d', name, re.IGNORECASE)
    if m:
        v = int(m.group(1))
        if v in (25, 50, 75, 100, 150, 200, 400):
            return float(v)
    return None


def find_masterfile(seq_dir):
    for f in os.listdir(seq_dir):
        if "MasterFile" in f and f.endswith(".xlsx"):
            return os.path.join(seq_dir, f)
    return None


def process_seq(seq_name, seq_dir):
    """Process a BP sequence using 4-TOC_CALC + 2-TOC."""
    mf_path = find_masterfile(seq_dir)
    if not mf_path:
        print(f"  SKIP: no MasterFile")
        return []

    sheets = llegir_masterfile_nou(mf_path)
    if not sheets or sheets.get("error"):
        print(f"  SKIP: {sheets.get('error', '?')}")
        return []

    info = sheets.get("info", {})
    toc_df = sheets.get("toc")
    toc_calc = sheets.get("toc_calc")

    if toc_df is None or toc_df.empty:
        print(f"  SKIP: no 2-TOC")
        return []
    if toc_calc is None or toc_calc.empty:
        print(f"  SKIP: no 4-TOC_CALC")
        return []

    # Find signal column in 2-TOC
    sig_col = None
    for c in toc_df.columns:
        cs = str(c).lower()
        if "toc" in cs and "ppb" in cs:
            sig_col = c
            break
    if sig_col is None:
        print(f"  SKIP: no TOC signal column")
        return []

    # Global volume from 0-INFO
    global_vol = info.get("Inj_Volume (uL)")
    if global_vol is not None:
        try:
            global_vol = float(global_vol)
        except (ValueError, TypeError):
            global_vol = None

    # Get unique KHP samples from 4-TOC_CALC
    sample_col = "Sample" if "Sample" in toc_calc.columns else toc_calc.columns[1]
    toc_row_col = "TOC_Row" if "TOC_Row" in toc_calc.columns else toc_calc.columns[0]
    t_rel_col = "Temps_Relatiu (min)" if "Temps_Relatiu (min)" in toc_calc.columns else toc_calc.columns[2]

    # _CAL convention: all non-control injections are KHP
    is_cal_seq = "_CAL" in seq_name.upper()
    CONTROL_NAMES = {"MQ", "NAOH", "BUFFER", "BLANC", "BLANK", "FI"}

    results = []

    for sample_name in toc_calc[sample_col].dropna().unique():
        sample_str = str(sample_name).strip()
        if not sample_str:
            continue

        # Determine if this is a KHP sample
        if is_khp(sample_str):
            conc = extract_khp_conc(sample_str)
        elif is_cal_seq:
            # _CAL: everything except controls is KHP
            base_name = sample_str.split("_R")[0].upper()
            if any(ctrl in base_name for ctrl in CONTROL_NAMES):
                continue
            conc = _parse_conc_from_name(sample_str)
        else:
            continue

        if conc <= 0:
            continue

        # Get rows for this sample
        mask = toc_calc[sample_col] == sample_name
        sample_rows = toc_calc[mask]
        if len(sample_rows) < 5:
            continue

        toc_indices = sample_rows[toc_row_col].dropna().astype(int).values
        t_rel = sample_rows[t_rel_col].dropna().values.astype(float)

        if len(toc_indices) < 5:
            continue

        # Extract DOC values from 2-TOC using TOC_Row indices
        # TOC_Row is 1-indexed relative to row 8 (header=6 means row 7 is header)
        # After pd.read_excel(header=6), index 0 = row 8 of Excel
        # So TOC_Row maps to toc_df.iloc[toc_row - 8] BUT let's check actual mapping
        # In the 152 example: first TOC_Row=235, first toc_df has 2080 rows (indices 0-2079)
        # toc_df index starts at 0, TOC_Row seems to be absolute Excel row
        # Let's try: idx = toc_row - 8 (since header is at row 7)
        # But safer: just match by position using the toc_row as offset from first data row

        # The data rows in 2-TOC start at row 8 (0-indexed: row 7 is header with header=6)
        # TOC_Row is the Excel row number, so: df_index = TOC_Row - 8
        min_idx = toc_indices.min() - 8
        max_idx = toc_indices.max() - 8

        # Bounds check
        if min_idx < 0 or max_idx >= len(toc_df):
            # Try without offset
            min_idx = toc_indices.min()
            max_idx = toc_indices.max()
            if max_idx >= len(toc_df):
                continue

        # Extract signal
        try:
            y_vals = toc_df[sig_col].iloc[min_idx:max_idx + 1].values.astype(float)
        except (IndexError, ValueError):
            continue

        # Use Temps_Relatiu as time axis (more reliable than computing from timestamps)
        if len(t_rel) == len(y_vals):
            t = t_rel
        elif len(t_rel) > len(y_vals):
            t = t_rel[:len(y_vals)]
        else:
            # Fallback: uniform spacing
            dt = 0.0667  # ~4 seconds
            t = np.arange(len(y_vals)) * dt
        y = y_vals

        # Remove NaN
        valid = ~(np.isnan(t) | np.isnan(y))
        t = t[valid]
        y = y[valid]

        if len(t) < 10:
            continue

        # Only positive time (trim pre-injection)
        pos_mask = t >= 0
        t = t[pos_mask]
        y = y[pos_mask]

        if len(t) < 10:
            continue

        # Get volume: from sample name first, then global
        vol_from_name = _parse_vol_from_name(sample_str)
        if vol_from_name:
            vol = vol_from_name
        elif global_vol:
            vol = global_vol
        else:
            vol = 100

        # --- INTEGRATION (same as Suite) ---

        # Baseline (BP: last 20%)
        bl = get_baseline_value(t, y, mode="BP")

        # Net signal
        y_net = np.maximum(y - bl, 0)

        # Detect main peak
        peak_result = detect_main_peak(t, y_net, is_bp=True)
        if peak_result is None or peak_result.get("peak_idx") is None:
            results.append({
                "seq": seq_name, "sample": sample_str, "conc": conc,
                "vol": vol, "status": "NO_PEAK"
            })
            continue

        peak_idx = peak_result["peak_idx"]
        t_max = t[peak_idx]

        # Find integration boundaries (Agilent tangent projection)
        try:
            left_idx, right_idx = find_peak_boundaries(t, y_net, peak_idx, is_bp=True)
        except Exception:
            results.append({
                "seq": seq_name, "sample": sample_str, "conc": conc,
                "vol": vol, "status": "NO_BOUNDS"
            })
            continue

        # Integrate (trapezoid on net signal)
        area = float(np.trapz(y_net[left_idx:right_idx + 1], t[left_idx:right_idx + 1]))

        # RF
        ug_doc = conc * vol / 1000
        rf_mass = area / ug_doc if ug_doc > 0 else 0

        # Peak height
        peak_h = float(np.max(y_net))

        # SNR
        bl_stats = get_baseline_stats(t, y, mode="BP")
        noise = bl_stats.get("noise", 1e-6) if bl_stats else 1e-6
        snr = peak_h / noise if noise > 0 else 0

        # FWHM
        half_max = peak_h / 2
        above_half = np.where(y_net >= half_max)[0]
        fwhm = t[above_half[-1]] - t[above_half[0]] if len(above_half) >= 2 else 0

        results.append({
            "seq": seq_name, "sample": sample_str, "conc": conc,
            "vol": vol, "area": area, "t_max": t_max,
            "t_left": t[left_idx], "t_right": t[right_idx],
            "width": t[right_idx] - t[left_idx],
            "baseline": bl, "snr": snr, "fwhm": fwhm,
            "rf_mass": rf_mass, "ug_doc": ug_doc,
            "peak_height": peak_h, "status": "OK",
        })

    return results


def linear_regression(x, y):
    n = len(x)
    if n < 2:
        return None
    sx, sy = np.sum(x), np.sum(y)
    sxx, sxy = np.sum(x * x), np.sum(x * y)
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-15:
        return None
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    rms = np.sqrt(np.mean((y - y_pred) ** 2))
    return {"slope": slope, "intercept": intercept, "r2": r2, "rms": rms, "n": n}


# =====================================================================
if __name__ == "__main__":
    all_results = []

    print("=" * 100)
    print("REPROCESSAMENT BP — Pipeline Suite actual (detect_main_peak + find_peak_boundaries)")
    print("=" * 100)

    for seq_name in BP_SEQS:
        seq_dir = os.path.join(DADES, seq_name)
        if not os.path.isdir(seq_dir):
            print(f"\n{seq_name}: NOT FOUND")
            continue

        print(f"\n--- {seq_name} ---")
        results = process_seq(seq_name, seq_dir)

        for r in results:
            if r["status"] == "OK":
                print(f"  {r['sample']:<20} {r['conc']:>5g}ppm  v={r['vol']:>4.0f}  "
                      f"area={r['area']:>8.1f}  t=[{r['t_left']:.1f}-{r['t_right']:.1f}]  "
                      f"t_max={r['t_max']:.2f}  RF={r['rf_mass']:>7.0f}  "
                      f"SNR={r['snr']:>6.0f}  bl={r['baseline']:.1f}")
            else:
                print(f"  {r['sample']:<20} {r['conc']:>5g}ppm  {r['status']}")
            all_results.append(r)

    # =====================================================================
    ok = [r for r in all_results if r["status"] == "OK"]

    # Group replicas: average per (seq, conc)
    groups = {}
    for r in ok:
        key = (r["seq"], r["conc"], r["vol"])
        if key not in groups:
            groups[key] = []
        groups[key].append(r)

    averaged = []
    for (seq, conc, vol), reps in groups.items():
        avg_area = np.mean([r["area"] for r in reps])
        ug = conc * vol / 1000
        averaged.append({
            "seq": seq, "conc": conc, "vol": vol,
            "area": avg_area, "ug_doc": ug,
            "rf_mass": avg_area / ug if ug > 0 else 0,
            "n_reps": len(reps),
            "rsd": np.std([r["area"] for r in reps]) / avg_area * 100 if avg_area > 0 and len(reps) > 1 else 0,
        })

    print("\n" + "=" * 100)
    print("REGRESSIO PER SEQ (repliques promitjades)")
    print("=" * 100)

    seq_groups = {}
    for a in averaged:
        if a["seq"] not in seq_groups:
            seq_groups[a["seq"]] = []
        seq_groups[a["seq"]].append(a)

    print(f"\n{'SEQ':<25} {'slope':>8} {'int':>8} {'R2':>10} {'RMS':>8} {'n':>4}  concs")
    print("-" * 90)

    for sn in BP_SEQS:
        if sn not in seq_groups:
            continue
        entries = sorted(seq_groups[sn], key=lambda x: x["conc"])
        x = np.array([e["ug_doc"] for e in entries])
        y = np.array([e["area"] for e in entries])

        if len(x) >= 2:
            reg = linear_regression(x, y)
            if reg:
                concs = [e["conc"] for e in entries]
                print(f"{sn:<25} {reg['slope']:>8.1f} {reg['intercept']:>8.1f} "
                      f"{reg['r2']:>10.6f} {reg['rms']:>8.2f} {reg['n']:>4}  {concs}")
        else:
            rf = entries[0]["rf_mass"]
            print(f"{sn:<25} RF_mass={rf:>8.0f}  ({entries[0]['conc']}ppm, n_reps={entries[0]['n_reps']})")

    # Detail table
    print("\n" + "=" * 100)
    print("DETALL PROMITJOS")
    print("=" * 100)
    print(f"{'SEQ':<25} {'conc':>6} {'vol':>5} {'ug':>6} {'area':>8} {'RF_mass':>8} {'reps':>5} {'RSD%':>6}")
    for a in sorted(averaged, key=lambda x: (x["seq"], x["conc"])):
        print(f"{a['seq']:<25} {a['conc']:>6g} {a['vol']:>5.0f} {a['ug_doc']:>6.3f} "
              f"{a['area']:>8.1f} {a['rf_mass']:>8.0f} {a['n_reps']:>5} {a['rsd']:>6.1f}")

    # 2ppm comparison
    twoppm = [a for a in averaged if abs(a["conc"] - 2.0) < 0.01]
    if twoppm:
        print(f"\n--- 2ppm: RF_mass = {np.mean([a['rf_mass'] for a in twoppm]):.0f} "
              f"+/- {np.std([a['rf_mass'] for a in twoppm]):.0f} (n={len(twoppm)}) ---")
