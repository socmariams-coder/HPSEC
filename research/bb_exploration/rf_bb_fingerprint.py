# -*- coding: utf-8 -*-
"""
BB Fingerprint — Building Blocks Spectral Characterization
============================================================
Standalone: separa contribucio organica i inorganica a la zona BB (23-26 min).

Principi de separacio:
  - Inorganics (Br-, NO3-, etc.) absorbeixen a <230nm, quasi res a >254nm
  - Organics (building blocks) absorbeixen a tot el rang UV (inclos 254, 280nm)
  - A254 i A280 a BB = senyal PURAMENT ORGANIC
  - A210 a BB = organic + inorganic barrejat
  - Exces A210 = A210_mesurat - A210_predit_per_organic = INORGANIC

Calibracio del ratio organic A210/A254:
  - HA a BB: A210/A254 = 6.7
  - FA a BB: A210/A254 = 25.8
  - Usem la mitjana geometrica (~13) o calibrem amb HA/FA de cada SEQ

Verificat: els inorganics elueixen realment a BB (no son artefactes).
  - Br-: -11 mAU constant a 210nm (23-26 min)
  - NO3: bimodal, undershoot a BB-early + segon pic a 26 min
  - Na2CO3: contribucio positiva (alcalinitat)
"""

import os, sys, re, glob, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
from scipy import stats

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

# =============================================================================
# CONFIG
# =============================================================================
DATA_FOLDERS = ["C:/Users/Lequia/Desktop/Dades"]
OUTPUT_DIR = "C:/Users/Lequia/Desktop/HPSEC/rf_bb_fingerprint"

BB_WINDOW = (23.0, 26.0)
HS_WINDOW = (18.0, 23.0)
SB_WINDOW = (26.0, 32.0)

# Organic reference ratio A210/A254 at BB zone (from HA and FA standards)
# HA: 8.64/1.29 = 6.7, FA: 17.82/0.69 = 25.8
# Geometric mean as default; can be calibrated per-sequence
ORG_A210_A254_RATIO = np.sqrt(6.7 * 25.8)  # ~13.1

TYPE_PATTERNS = [
    ("SRHA",     r"(?i)^(SRHA|3S101H)"),
    ("SRFA",     r"(?i)^(SRFA|3S101F)"),
    ("HA",       r"(?i)(?:^|[_\-\s])HA(?:[_\-\s\.]|$|\d)"),
    ("FA",       r"(?i)(?:^|[_\-\s])FA(?:[_\-\s\.]|$|\d)"),
    ("MIX",      r"(?i)(?:^|[_\-\s])MIX(?:[_\-\s\.]|$|\d)"),
    ("NO3",      r"(?i)^(NO3|NaNO3)"),
    ("Br",       r"(?i)^(Br[_\.\s\d\-]|KBr)"),
    ("CaCO3",    r"(?i)^CaCO3"),
    ("Na2CO3",   r"(?i)^Na2CO3"),
    ("FR",       r"(?i)^FR\d"),
    ("EX",       r"(?i)^EX\d"),
    ("EXT",      r"(?i)^EXT\d"),
    ("LQ",       r"(?i)^LQ\d"),
    ("SK",       r"(?i)^SK\d"),
    ("PTLL",     r"(?i)^PTLL[\-_]"),
    ("PTT",      r"(?i)^PTT[\-_]"),
    ("ATL_2024", r"(?i)^\d{4}-\d{2}-\d{2}"),
    ("PTLL_2025",r"(?i)^\d{4}-PTLL"),
    ("LAB",      r"(?i)^\d{4}-L-"),
    ("LAB_T",    r"(?i)^\d{4}-T-"),
    ("EP",       r"(?i)^EP[_\s\-]"),
    ("SP",       r"(?i)^SP[_\s\-]"),
    ("POST_O3",  r"(?i)^POST"),
    ("KHP",      r"(?i)^KHP"),
    ("BLANK",    r"(?i)^(MQ|BLANK|BLK|H2O)"),
]

TYPE_GROUPS = {
    "FR": ["FR"], "PTLL": ["PTLL", "PTLL_2025"], "PTT": ["PTT"],
    "LAB": ["LAB"], "LAB_T": ["LAB_T"],
    "EP": ["EP"], "SP": ["SP"], "POST_O3": ["POST_O3"],
    "HA": ["HA", "SRHA"], "FA": ["FA", "SRFA"],
    "EX": ["EX", "EXT"], "KHP": ["KHP"], "BLANK": ["BLANK"],
    "NO3": ["NO3"], "Br": ["Br"], "CaCO3": ["CaCO3"], "Na2CO3": ["Na2CO3"],
}


def classify_sample(filename):
    base = re.sub(r'[_\s]*(R?\d+)\.(csv|CSV)$', '', filename)
    for label, pattern in TYPE_PATTERNS:
        if re.search(pattern, base):
            return label
    return "OTHER"


def assign_group(sample_type):
    for group, types in TYPE_GROUPS.items():
        if sample_type in types:
            return group
    return "OTHER"


def load_export3d(filepath):
    for enc in ['utf-16', 'utf-8', 'latin-1']:
        try:
            df = pd.read_csv(filepath, encoding=enc, sep=',', header=None)
            if df.shape[1] < 50:
                continue
            wl = df.iloc[0, 1:].values.astype(float)
            t = df.iloc[1:, 0].values.astype(float)
            d = df.iloc[1:, 1:].values.astype(float)
            if len(t) < 100 or np.max(t) < 15:
                continue
            return t, wl, d
        except:
            continue
    return None, None, None


def get_val(spec, wl, target):
    idx = np.argmin(np.abs(wl - target))
    return spec[idx]


def fraction_spec(t, data, t0, t1):
    mask = (t >= t0) & (t <= t1)
    if np.sum(mask) < 3:
        return None
    return np.mean(data[mask, :], axis=0)


def simple_slope(spec, wl, wl1, wl2):
    idx1 = np.argmin(np.abs(wl - wl1))
    idx2 = np.argmin(np.abs(wl - wl2))
    return (spec[idx1] - spec[idx2]) / (wl2 - wl1)


# =============================================================================
# BB FINGERPRINT
# =============================================================================
def extract_bb_fingerprint(time_vals, wl, data):
    """
    BB fingerprint amb separacio organica/inorganica.

    Organic metrics: basades en A254, A280 (no interferits per inorganics)
    Inorganic metrics: exces A210 per sobre del que prediu la MON organica
    """
    fp = {}

    bb = fraction_spec(time_vals, data, *BB_WINDOW)
    hs = fraction_spec(time_vals, data, *HS_WINDOW)
    sb = fraction_spec(time_vals, data, *SB_WINDOW)

    if bb is None:
        return None

    # --- Raw intensities at BB ---
    for target in [200, 210, 220, 230, 240, 254, 260, 270, 280, 300, 350, 400]:
        fp[f"bb_A{target}"] = get_val(bb, wl, target)

    a254 = fp["bb_A254"]
    a280 = fp["bb_A280"]
    a210 = fp["bb_A210"]
    a220 = fp["bb_A220"]
    a200 = fp["bb_A200"]

    # =================================================================
    # ORGANIC COMPONENT (A254, A280 = pure organic)
    # =================================================================
    fp["org_A254"] = a254
    fp["org_A280"] = a280
    fp["org_A300"] = fp["bb_A300"]
    fp["org_A350"] = fp["bb_A350"]

    # Organic ratios (inorganic-free zone >240nm)
    if abs(a280) > 0.005:
        fp["org_A254_A280"] = a254 / a280
    if abs(a254) > 0.005:
        fp["org_A280_A254"] = a280 / a254
        fp["org_A300_A254"] = fp["bb_A300"] / a254
        fp["org_A350_A254"] = fp["bb_A350"] / a254 if abs(fp["bb_A350"]) > 0.001 else np.nan

    # Organic slopes (>240nm zone, inorganic-free)
    fp["org_S254_280"] = simple_slope(bb, wl, 254, 280)
    fp["org_S280_350"] = simple_slope(bb, wl, 280, 350)
    fp["org_S254_300"] = simple_slope(bb, wl, 254, 300)

    # Organic spectral slope (log-linear, 240-300nm)
    mask_org = (wl >= 240) & (wl <= 300)
    w_org = wl[mask_org]
    s_org = bb[mask_org]
    valid = s_org > 0
    if np.sum(valid) >= 4:
        sl, _, r, _, _ = stats.linregress(w_org[valid], np.log(s_org[valid]))
        fp["org_SL_240_300"] = sl
    else:
        fp["org_SL_240_300"] = np.nan

    # Organic shape: integrate 240-300 and 300-400
    if abs(a254) > 0.005:
        bb_n254 = bb / a254
        mask_mid = (wl >= 240) & (wl <= 300)
        mask_vis = (wl >= 300) & (wl <= 400)
        fp["org_area_mid"] = np.trapz(bb_n254[mask_mid], wl[mask_mid])
        fp["org_area_vis"] = np.trapz(bb_n254[mask_vis], wl[mask_vis])
        if fp["org_area_vis"] > 0.01:
            fp["org_mid_vis"] = fp["org_area_mid"] / fp["org_area_vis"]

    # =================================================================
    # INORGANIC COMPONENT (excess A210 over organic prediction)
    # =================================================================
    # Organic A210 predicted from A254 * reference ratio
    org_a210_pred = a254 * ORG_A210_A254_RATIO if a254 > 0 else 0
    fp["inorg_A210_excess"] = a210 - org_a210_pred  # mAU of inorganic signal
    fp["org_A210_predicted"] = org_a210_pred

    # Same for A220
    # HA at BB: A220/A254 = 5.01/1.29 = 3.88, FA: 9.24/0.69 = 13.4
    org_a220_ratio = np.sqrt(3.88 * 13.4)  # ~7.2
    org_a220_pred = a254 * org_a220_ratio if a254 > 0 else 0
    fp["inorg_A220_excess"] = a220 - org_a220_pred

    # And A200
    # HA: 10.05/1.29=7.8, FA: 21.0/0.69=30.4
    org_a200_ratio = np.sqrt(7.8 * 30.4)  # ~15.4
    org_a200_pred = a254 * org_a200_ratio if a254 > 0 else 0
    fp["inorg_A200_excess"] = a200 - org_a200_pred

    # Inorganic fraction of A210
    if abs(a210) > 0.01:
        fp["inorg_fraction_A210"] = fp["inorg_A210_excess"] / a210
    else:
        fp["inorg_fraction_A210"] = np.nan

    # Inorganic spectral slope (200-230nm, dominated by anions)
    fp["inorg_S200_210"] = simple_slope(bb, wl, 200, 210)
    fp["inorg_S210_220"] = simple_slope(bb, wl, 210, 220)
    fp["inorg_S210_230"] = simple_slope(bb, wl, 210, 230)

    # Inorganic shape: which anion dominates?
    # Br-: steep drop 200-230, NO3: broader, Na2CO3: flat
    if abs(a210) > 0.5:
        fp["inorg_A200_A210"] = a200 / a210
        fp["inorg_A220_A210"] = a220 / a210
        fp["inorg_A230_A210"] = fp["bb_A230"] / a210

    # =================================================================
    # CONTEXT: BB vs HS
    # =================================================================
    if hs is not None:
        a254_hs = get_val(hs, wl, 254)
        a210_hs = get_val(hs, wl, 210)
        a280_hs = get_val(hs, wl, 280)

        if a254_hs > 0.01:
            fp["ctx_org_bb_hs_254"] = a254 / a254_hs  # organic BB fraction relative to HS
        if a280_hs > 0.01:
            fp["ctx_org_bb_hs_280"] = a280 / a280_hs
        if abs(a210_hs) > 0.5:
            fp["ctx_total_bb_hs_210"] = a210 / a210_hs  # total (org+inorg) BB/HS

    # =================================================================
    # ELUTION PROFILE within BB
    # =================================================================
    bb_mask = (time_vals >= BB_WINDOW[0]) & (time_vals <= BB_WINDOW[1])
    bb_time = time_vals[bb_mask]
    bb_data = data[bb_mask, :]

    for target_wl in [210, 254, 280]:
        wl_idx = np.argmin(np.abs(wl - target_wl))
        profile = bb_data[:, wl_idx]
        if len(profile) < 5:
            continue

        # Monotonicity
        diffs = np.diff(profile)
        fp[f"elut_mono_{target_wl}"] = np.sum(diffs < 0) / len(diffs) if len(diffs) > 0 else 0.5

        # Temporal slope
        if np.std(profile) > 0.001 and len(profile) > 3:
            sl, _, r, _, _ = stats.linregress(bb_time, profile)
            fp[f"elut_slope_{target_wl}"] = sl

    # NaN cleanup
    for k in list(fp.keys()):
        v = fp[k]
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            fp[k] = np.nan

    return fp


# =============================================================================
# MAIN
# =============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # =========================================================
    # STEP 1: Scan all samples
    # =========================================================
    print("=" * 70)
    print("STEP 1: Scanning...")
    print("=" * 70)

    records = []
    for base_dir in DATA_FOLDERS:
        if not os.path.isdir(base_dir):
            continue
        for seq_dir in sorted(glob.glob(os.path.join(base_dir, "*_SEQ*"))):
            seq_name = os.path.basename(seq_dir)
            is_bp = "_BP" in seq_name.upper()
            e3d = os.path.join(seq_dir, "Export3D")
            if not os.path.isdir(e3d):
                continue
            for csv_path in sorted(glob.glob(os.path.join(e3d, "*.csv")) +
                                   glob.glob(os.path.join(e3d, "*.CSV"))):
                fname = os.path.basename(csv_path)
                stype = classify_sample(fname)
                group = assign_group(stype)

                t, wl, d = load_export3d(csv_path)
                if t is None:
                    continue
                fp = extract_bb_fingerprint(t, wl, d)
                if fp is None:
                    continue

                fp["_type"] = stype
                fp["_group"] = group
                fp["_seq"] = seq_name
                fp["_file"] = fname
                fp["_bp"] = is_bp
                records.append(fp)

    df = pd.DataFrame(records)
    print(f"  Total: {len(df)} samples")
    for g, c in df["_group"].value_counts().items():
        print(f"    {g:15s}: {c}")

    # =========================================================
    # STEP 2: Organic vs Inorganic separation
    # =========================================================
    print("\n" + "=" * 70)
    print("STEP 2: Organic vs Inorganic at BB zone")
    print("=" * 70)

    display_groups = ["FR", "EX", "LAB", "LAB_T", "PTLL", "POST_O3", "HA", "FA", "NO3", "Br", "CaCO3", "Na2CO3"]

    print(f"\n  {'Type':12s} {'n':>4s} | {'A210_tot':>9s} {'A210_org':>9s} {'A210_inorg':>10s} {'%inorg':>7s} | {'A254_org':>9s} {'A280_org':>9s}")
    print(f"  {'-'*12} {'-'*4} | {'-'*9} {'-'*9} {'-'*10} {'-'*7} | {'-'*9} {'-'*9}")

    for g in display_groups:
        gdf = df[df["_group"] == g]
        if len(gdf) == 0:
            continue
        a210 = gdf["bb_A210"].mean()
        org_pred = gdf["org_A210_predicted"].mean()
        inorg = gdf["inorg_A210_excess"].mean()
        pct = gdf["inorg_fraction_A210"].mean()
        a254 = gdf["org_A254"].mean()
        a280 = gdf["org_A280"].mean()

        pct_str = f"{pct:.0%}" if not np.isnan(pct) else "N/A"
        print(f"  {g:12s} {len(gdf):4d} | {a210:9.2f} {org_pred:9.2f} {inorg:10.2f} {pct_str:>7s} | {a254:9.3f} {a280:9.3f}")

    # =========================================================
    # STEP 3: Organic BB fingerprint
    # =========================================================
    print("\n" + "=" * 70)
    print("STEP 3: Organic BB Fingerprint (based on A254, A280)")
    print("=" * 70)

    org_cols = ["org_A254", "org_A280", "org_A300", "org_A350",
                "org_A254_A280", "org_S254_280", "org_S280_350",
                "org_SL_240_300", "org_mid_vis",
                "ctx_org_bb_hs_254", "ctx_org_bb_hs_280"]
    org_cols = [c for c in org_cols if c in df.columns]

    print(f"\n  {'Type':12s} {'n':>4s}", end="")
    for c in org_cols:
        short = c.replace("org_","").replace("ctx_org_","c:")[:10]
        print(f" {short:>10s}", end="")
    print()
    print(f"  {'-'*12} {'-'*4}" + f" {'-'*10}" * len(org_cols))

    for g in ["FR", "EX", "LAB", "LAB_T", "PTLL", "POST_O3", "HA", "FA"]:
        gdf = df[df["_group"] == g]
        if len(gdf) == 0:
            continue
        print(f"  {g:12s} {len(gdf):4d}", end="")
        for c in org_cols:
            v = gdf[c].mean()
            if np.isnan(v):
                print(f" {'nan':>10s}", end="")
            elif abs(v) > 10:
                print(f" {v:10.1f}", end="")
            elif abs(v) > 0.1:
                print(f" {v:10.3f}", end="")
            else:
                print(f" {v:10.4f}", end="")
        print()

    # =========================================================
    # STEP 4: Inorganic BB fingerprint
    # =========================================================
    print("\n" + "=" * 70)
    print("STEP 4: Inorganic BB Fingerprint (excess A210)")
    print("=" * 70)

    inorg_cols = ["inorg_A210_excess", "inorg_A220_excess", "inorg_A200_excess",
                  "inorg_S200_210", "inorg_S210_230",
                  "inorg_A200_A210", "inorg_A220_A210", "inorg_A230_A210"]
    inorg_cols = [c for c in inorg_cols if c in df.columns]

    print(f"\n  {'Type':12s} {'n':>4s}", end="")
    for c in inorg_cols:
        short = c.replace("inorg_","")[:12]
        print(f" {short:>12s}", end="")
    print()
    print(f"  {'-'*12} {'-'*4}" + f" {'-'*12}" * len(inorg_cols))

    for g in ["FR", "EX", "LAB", "LAB_T", "PTLL", "POST_O3", "NO3", "Br", "CaCO3", "Na2CO3"]:
        gdf = df[df["_group"] == g]
        if len(gdf) == 0:
            continue
        print(f"  {g:12s} {len(gdf):4d}", end="")
        for c in inorg_cols:
            v = gdf[c].mean()
            if np.isnan(v):
                print(f" {'nan':>12s}", end="")
            else:
                print(f" {v:12.2f}", end="")
        print()

    # =========================================================
    # STEP 5: LAB vs LAB_T
    # =========================================================
    print("\n" + "=" * 70)
    print("STEP 5: LAB (PTLL) vs LAB_T (PTT)")
    print("=" * 70)

    lab = df[df["_group"] == "LAB"]
    labt = df[df["_group"] == "LAB_T"]

    if len(lab) > 0 and len(labt) > 0:
        test_cols = [c for c in df.columns if not c.startswith("_")]
        significant = []
        for col in test_cols:
            v1 = lab[col].dropna().values
            v2 = labt[col].dropna().values
            if len(v1) < 3 or len(v2) < 3:
                continue
            if np.std(v1) < 1e-10 and np.std(v2) < 1e-10:
                continue
            t_stat, p_val = stats.ttest_ind(v1, v2, equal_var=False)
            significant.append((col, np.mean(v1), np.mean(v2), t_stat, p_val))

        significant.sort(key=lambda x: x[4])

        print(f"\n  LAB (PTLL): {len(lab)} | LAB_T (PTT): {len(labt)}")
        print(f"\n  {'Feature':35s} {'LAB(PTLL)':>12s} {'LABT(PTT)':>12s} {'Ratio':>8s} {'p-val':>10s}")
        print(f"  {'-'*35} {'-'*12} {'-'*12} {'-'*8} {'-'*10}")

        for col, m1, m2, t_stat, p_val in significant[:20]:
            if p_val > 0.05:
                break
            ratio = m1 / m2 if abs(m2) > 0.001 else np.nan
            ratio_s = f"{ratio:8.2f}" if not np.isnan(ratio) else "    inf"
            stars = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*"
            print(f"  {col:35s} {m1:12.3f} {m2:12.3f} {ratio_s} {p_val:10.2e} {stars}")

    # =========================================================
    # STEP 6: Plots
    # =========================================================
    print("\n" + "=" * 70)
    print("STEP 6: Plots")
    print("=" * 70)

    colors = {
        "FR": "#2ecc71", "EX": "#8B8000", "LAB": "#00CED1", "LAB_T": "#FF00FF",
        "PTLL": "#3498db", "POST_O3": "#8B4513", "HA": "#c0392b", "FA": "#2c3e50",
        "NO3": "#000000", "Br": "#808080", "CaCO3": "#FFB6C1", "Na2CO3": "#9B59B6",
    }

    # --- Plot 1: Organic vs Inorganic bar chart ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    groups_plot = [g for g in ["FR", "EX", "LAB", "LAB_T", "PTLL", "POST_O3"] if g in df["_group"].values]
    x = np.arange(len(groups_plot))

    # A210 stacked: organic + inorganic
    ax = axes[0]
    org_vals = [df[df["_group"]==g]["org_A210_predicted"].mean() for g in groups_plot]
    inorg_vals = [df[df["_group"]==g]["inorg_A210_excess"].mean() for g in groups_plot]
    total_vals = [df[df["_group"]==g]["bb_A210"].mean() for g in groups_plot]

    ax.bar(x, org_vals, color='forestgreen', label='Organic (predicted)', alpha=0.8)
    ax.bar(x, inorg_vals, bottom=org_vals, color='coral', label='Inorganic (excess)', alpha=0.8)
    ax.scatter(x, total_vals, color='black', zorder=5, s=50, label='Total measured')
    ax.set_xticks(x)
    ax.set_xticklabels(groups_plot, rotation=30)
    ax.set_ylabel("A210 at BB zone (mAU)")
    ax.set_title("A210 Decomposition: Organic vs Inorganic")
    ax.legend()
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.grid(True, alpha=0.3)

    # Pure organic metrics (A254, A280)
    ax = axes[1]
    w = 0.35
    a254_vals = [df[df["_group"]==g]["org_A254"].mean() for g in groups_plot]
    a280_vals = [df[df["_group"]==g]["org_A280"].mean() for g in groups_plot]
    ax.bar(x - w/2, a254_vals, w, color='steelblue', label='A254 (organic)', alpha=0.8)
    ax.bar(x + w/2, a280_vals, w, color='darkorange', label='A280 (organic)', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(groups_plot, rotation=30)
    ax.set_ylabel("Absorbance at BB zone (mAU)")
    ax.set_title("Pure Organic BB Signal (A254, A280)")
    ax.legend()
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "11_organic_vs_inorganic_A210.png"), dpi=150)
    plt.close()

    # --- Plot 2: Inorganic spectral shape comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # All water types: excess spectrum (200-240nm)
    ax = axes[0]
    wl_inorg = [200, 210, 220, 230, 240]
    for g in groups_plot:
        gdf = df[df["_group"] == g]
        excess = [gdf[f"inorg_A{w}_excess"].mean() if f"inorg_A{w}_excess" in gdf.columns
                  else gdf[f"bb_A{w}"].mean() - gdf["org_A254"].mean() * ORG_A210_A254_RATIO * (1 if w == 210 else 0)
                  for w in wl_inorg]
        # Manual calculation for 230, 240
        excess_vals = []
        for w in wl_inorg:
            col = f"inorg_A{w}_excess"
            if col in gdf.columns:
                excess_vals.append(gdf[col].mean())
            else:
                excess_vals.append(np.nan)
        excess_vals = [v for v in excess_vals if not np.isnan(v)]
        wl_valid = wl_inorg[:len(excess_vals)]
        if excess_vals:
            ax.plot(wl_valid, excess_vals, 'o-', color=colors.get(g, 'gray'),
                    label=f"{g} (n={len(gdf)})", linewidth=1.5)

    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Excess absorbance (mAU)")
    ax.set_title("Inorganic excess spectrum at BB zone")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Inorganic shape ratios (fingerprint of which anion dominates)
    ax = axes[1]
    shape_cols = ["inorg_A200_A210", "inorg_A220_A210", "inorg_A230_A210"]
    shape_cols = [c for c in shape_cols if c in df.columns]
    if shape_cols:
        groups_with_data = [g for g in groups_plot + ["NO3", "Br", "Na2CO3"]
                           if g in df["_group"].values and
                           df[df["_group"]==g][shape_cols[0]].notna().any()]
        x2 = np.arange(len(groups_with_data))
        w2 = 0.8 / len(shape_cols)
        for i, col in enumerate(shape_cols):
            vals = [df[df["_group"]==g][col].mean() for g in groups_with_data]
            short = col.replace("inorg_", "")
            ax.bar(x2 + i*w2, vals, w2, label=short, alpha=0.8)
        ax.set_xticks(x2 + w2)
        ax.set_xticklabels(groups_with_data, rotation=30)
        ax.set_ylabel("Ratio")
        ax.set_title("Inorganic spectral shape (ratios rel. A210)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "12_inorganic_fingerprint.png"), dpi=150)
    plt.close()

    # --- Plot 3: LAB vs LAB_T comparison ---
    if len(lab) > 0 and len(labt) > 0:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        key_metrics = [
            ("inorg_A210_excess", "Inorganic A210\n(excess mAU)"),
            ("org_A254", "Organic A254\n(mAU)"),
            ("org_A280", "Organic A280\n(mAU)"),
            ("org_A254_A280", "Organic A254/A280\nratio"),
            ("org_S254_280", "Organic slope\nS254-280 (mAU/nm)"),
            ("inorg_S210_230", "Inorganic slope\nS210-230 (mAU/nm)"),
        ]

        for i, (col, title) in enumerate(key_metrics):
            ax = axes[i//3, i%3]
            if col not in df.columns:
                ax.set_visible(False)
                continue
            v1 = lab[col].dropna().values
            v2 = labt[col].dropna().values
            if len(v1) == 0 or len(v2) == 0:
                ax.set_visible(False)
                continue

            bp = ax.boxplot([v1, v2], labels=["LAB\n(PTLL)", "LAB_T\n(PTT)"],
                           patch_artist=True, widths=0.5)
            bp['boxes'][0].set_facecolor('#AED6F1')
            bp['boxes'][1].set_facecolor('#F5B7B1')
            _, p = stats.ttest_ind(v1, v2, equal_var=False)
            ax.set_title(f"{title}\np={p:.2e}")
            ax.grid(True, alpha=0.3)

        plt.suptitle("LAB (PTLL) vs LAB_T (PTT): Organic + Inorganic BB Metrics", fontsize=14)
        plt.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "13_lab_vs_labt_comparison.png"), dpi=150)
        plt.close()

    # --- Plot 4: Full BB fingerprint summary ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Organic A254 at BB as fraction of HS A254
    ax = axes[0]
    col = "ctx_org_bb_hs_254"
    if col in df.columns:
        groups_ctx = [g for g in groups_plot if df[df["_group"]==g][col].notna().any()]
        vals = [df[df["_group"]==g][col].mean() for g in groups_ctx]
        bar_colors = [colors.get(g, 'gray') for g in groups_ctx]
        ax.bar(range(len(groups_ctx)), vals, color=bar_colors, alpha=0.8)
        ax.set_xticks(range(len(groups_ctx)))
        ax.set_xticklabels(groups_ctx, rotation=30)
        ax.set_ylabel("BB/HS ratio")
        ax.set_title("Organic BB/HS ratio at 254nm\n(fraction of organic matter at BB)")
        ax.grid(True, alpha=0.3)

    # Inorganic excess A210
    ax = axes[1]
    vals = [df[df["_group"]==g]["inorg_A210_excess"].mean() for g in groups_plot]
    bar_colors = [colors.get(g, 'gray') for g in groups_plot]
    ax.bar(range(len(groups_plot)), vals, color=bar_colors, alpha=0.8)
    ax.set_xticks(range(len(groups_plot)))
    ax.set_xticklabels(groups_plot, rotation=30)
    ax.set_ylabel("Excess A210 (mAU)")
    ax.set_title("Inorganic excess at BB zone (A210)\n(proxy for anion content)")
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.grid(True, alpha=0.3)

    # Combined: organic vs inorganic "fingerprint"
    ax = axes[2]
    for g in groups_plot:
        gdf = df[df["_group"] == g]
        org_x = gdf["org_A254"].mean()
        inorg_y = gdf["inorg_A210_excess"].mean()
        ax.scatter(org_x, inorg_y, color=colors.get(g, 'gray'), s=100,
                   label=g, edgecolors='black', linewidth=0.5, zorder=5)
        # Individual points
        ax.scatter(gdf["org_A254"].values, gdf["inorg_A210_excess"].values,
                   color=colors.get(g, 'gray'), s=15, alpha=0.3)
    ax.set_xlabel("Organic A254 at BB (mAU)")
    ax.set_ylabel("Inorganic excess A210 (mAU)")
    ax.set_title("BB Fingerprint: Organic vs Inorganic")
    ax.legend(fontsize=7, loc='best')
    ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
    ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "14_bb_fingerprint_summary.png"), dpi=150)
    plt.close()

    print(f"  Plots saved to {OUTPUT_DIR}/")

    # =========================================================
    # SUMMARY
    # =========================================================
    print("\n" + "=" * 70)
    print("BB FINGERPRINT SUMMARY")
    print("=" * 70)
    print(f"""
  Samples: {len(df)}
  Organic reference ratio A210/A254 at BB: {ORG_A210_A254_RATIO:.1f}
    (HA=6.7, FA=25.8, geometric mean={ORG_A210_A254_RATIO:.1f})

  SEPARACIO ORGANIC/INORGANIC A BB:
  +-----------+----------+-----------+----------+--------+
  | Tipus     | A210_tot | A210_org  | A210_ino | %inorg |
  +-----------+----------+-----------+----------+--------+""")

    for g in ["FR", "EX", "LAB", "LAB_T", "PTLL", "POST_O3"]:
        gdf = df[df["_group"] == g]
        if len(gdf) == 0: continue
        a210 = gdf["bb_A210"].mean()
        org = gdf["org_A210_predicted"].mean()
        ino = gdf["inorg_A210_excess"].mean()
        pct = gdf["inorg_fraction_A210"].mean()
        pct_s = f"{pct:.0%}" if not np.isnan(pct) else "N/A"
        print(f"  | {g:9s} | {a210:8.1f} | {org:9.1f} | {ino:8.1f} | {pct_s:>6s} |")

    print(f"""  +-----------+----------+-----------+----------+--------+

  ORGANIC BB FINGERPRINT (A254, A280 = inorganic-free):
    FR:      A254={df[df['_group']=='FR']['org_A254'].mean():.3f}, A280={df[df['_group']=='FR']['org_A280'].mean():.3f} mAU
    LAB:     A254={df[df['_group']=='LAB']['org_A254'].mean():.3f}, A280={df[df['_group']=='LAB']['org_A280'].mean():.3f} mAU
    LAB_T:   A254={df[df['_group']=='LAB_T']['org_A254'].mean():.3f}, A280={df[df['_group']=='LAB_T']['org_A280'].mean():.3f} mAU
    POST_O3: A254={df[df['_group']=='POST_O3']['org_A254'].mean():.3f}, A280={df[df['_group']=='POST_O3']['org_A280'].mean():.3f} mAU

  INORGANIC BB FINGERPRINT (excess A210):
    LAB (PTLL):  {df[df['_group']=='LAB']['inorg_A210_excess'].mean():.1f} mAU  <-- mes anions
    LAB_T (PTT): {df[df['_group']=='LAB_T']['inorg_A210_excess'].mean():.1f} mAU
    Ratio PTLL/PTT: {df[df['_group']=='LAB']['inorg_A210_excess'].mean() / df[df['_group']=='LAB_T']['inorg_A210_excess'].mean():.1f}x
""")


if __name__ == "__main__":
    main()
