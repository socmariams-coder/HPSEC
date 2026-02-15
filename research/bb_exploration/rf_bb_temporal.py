# -*- coding: utf-8 -*-
"""
BB Temporal Exploration — Sub-structure within BB zone (23-26 min)
=================================================================
Explora si hi ha variabilitat temporal dins la zona BB:
  - Perfils d'elucio a 200, 210, 220, 230, 254, 280 nm
  - Separacio organic/inorganic per sub-finestres (early/mid/late)
  - Heatmaps temps x wavelength per tipus de mostra
  - Correlacio temporal entre wavelengths
"""

import os, sys, re, glob, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy import stats

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

# =============================================================================
# CONFIG
# =============================================================================
DATA_FOLDERS = ["C:/Users/Lequia/Desktop/Dades"]
OUTPUT_DIR = "C:/Users/Lequia/Desktop/HPSEC/rf_bb_temporal"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BB_WINDOW = (23.0, 26.0)
HS_WINDOW = (18.0, 23.0)

# Sub-windows within BB (1 min each)
BB_EARLY = (23.0, 24.0)
BB_MID   = (24.0, 25.0)
BB_LATE  = (25.0, 26.0)

# Key wavelengths to track
KEY_WL = [200, 210, 220, 230, 254, 280]

# Organic reference ratio
ORG_A210_A254_RATIO = np.sqrt(6.7 * 25.8)  # ~13.1

TYPE_PATTERNS = [
    ("SRHA",     r"(?i)^(SRHA|3S101H)"),
    ("SRFA",     r"(?i)^(SRFA|3S101F)"),
    ("HA",       r"(?i)(?:^|[_\-\s])HA(?:[_\-\s\.]|$|\d)"),
    ("FA",       r"(?i)(?:^|[_\-\s])FA(?:[_\-\s\.]|$|\d)"),
    ("NO3",      r"(?i)^(NO3|NaNO3)"),
    ("Br",       r"(?i)^(Br[_\.\s\d\-]|KBr)"),
    ("CaCO3",    r"(?i)^CaCO3"),
    ("Na2CO3",   r"(?i)^Na2CO3"),
    ("FR",       r"(?i)^FR\d"),
    ("EX",       r"(?i)^EX\d"),
    ("EXT",      r"(?i)^EXT\d"),
    ("LAB",      r"(?i)^\d{4}-L-"),
    ("LAB_T",    r"(?i)^\d{4}-T-"),
    ("PTLL",     r"(?i)^PTLL[\-_]"),
    ("PTT",      r"(?i)^PTT[\-_]"),
    ("POST_O3",  r"(?i)^POST"),
    ("KHP",      r"(?i)^KHP"),
    ("BLANK",    r"(?i)^(MQ|BLANK|BLK|H2O)"),
]

WATER_GROUPS = {
    "FR": ["FR"], "EX": ["EX", "EXT"],
    "LAB": ["LAB"], "LAB_T": ["LAB_T"],
    "PTLL": ["PTLL"], "POST_O3": ["POST_O3"],
    "HA": ["HA", "SRHA"], "FA": ["FA", "SRFA"],
    "NO3": ["NO3"], "Br": ["Br"],
}

# Colors per group
GROUP_COLORS = {
    "FR": "#2ecc71", "EX": "#8B4513", "LAB": "#00bcd4", "LAB_T": "#e91e63",
    "PTLL": "#3f51b5", "POST_O3": "#ff9800", "HA": "#c0392b", "FA": "#9b59b6",
    "NO3": "#607d8b", "Br": "#795548",
}


def classify_sample(filename):
    base = re.sub(r'[_\s]*(R?\d+)\.(csv|CSV)$', '', filename)
    for label, pattern in TYPE_PATTERNS:
        if re.search(pattern, base):
            return label
    return "OTHER"


def assign_group(sample_type):
    for group, types in WATER_GROUPS.items():
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


def get_wl_idx(wl, target):
    return np.argmin(np.abs(wl - target))


def time_slice(t, data, t0, t1):
    """Return time and data arrays for a time window."""
    mask = (t >= t0) & (t <= t1)
    return t[mask], data[mask, :]


# =============================================================================
# SCAN
# =============================================================================
print("=" * 70)
print("STEP 1: Scanning and loading BB temporal data...")
print("=" * 70)

samples = []
for folder in DATA_FOLDERS:
    for csv_path in glob.glob(os.path.join(folder, "**", "Export3D", "*.csv"), recursive=True):
        fn = os.path.basename(csv_path)
        stype = classify_sample(fn)
        group = assign_group(stype)
        if group == "OTHER":
            continue
        samples.append({"path": csv_path, "filename": fn, "type": stype, "group": group})

print(f"  Found {len(samples)} classified samples")

# Load all samples — extract BB temporal profiles
records = []
bb_profiles = defaultdict(list)  # group -> list of (time, profile_dict)

for s in samples:
    t, wl, data = load_export3d(s["path"])
    if t is None:
        continue

    # Full BB window
    bb_mask = (t >= BB_WINDOW[0]) & (t <= BB_WINDOW[1])
    if np.sum(bb_mask) < 5:
        continue

    bb_t = t[bb_mask]
    bb_d = data[bb_mask, :]

    # Extract time series at key wavelengths
    profiles = {}
    for w in KEY_WL:
        idx = get_wl_idx(wl, w)
        profiles[w] = bb_d[:, idx]

    bb_profiles[s["group"]].append({
        "time": bb_t,
        "profiles": profiles,
        "wl": wl,
        "bb_data": bb_d,
        "filename": s["filename"],
    })

    # Sub-window means
    rec = {"group": s["group"], "type": s["type"], "filename": s["filename"]}
    for label, (t0, t1) in [("early", BB_EARLY), ("mid", BB_MID), ("late", BB_LATE), ("full", BB_WINDOW)]:
        mask = (t >= t0) & (t <= t1)
        if np.sum(mask) < 2:
            continue
        spec = np.mean(data[mask, :], axis=0)
        for w in KEY_WL:
            idx = get_wl_idx(wl, w)
            rec[f"{label}_{w}"] = spec[idx]

    # Compute organic/inorganic per sub-window
    for label in ["early", "mid", "late", "full"]:
        a254 = rec.get(f"{label}_254", 0)
        a210 = rec.get(f"{label}_210", 0)
        a210_org = a254 * ORG_A210_A254_RATIO
        a210_inorg = a210 - a210_org
        rec[f"{label}_org_A254"] = a254
        rec[f"{label}_inorg_A210"] = a210_inorg
        if abs(a210) > 0.01:
            rec[f"{label}_frac_inorg"] = a210_inorg / a210
        else:
            rec[f"{label}_frac_inorg"] = np.nan

    records.append(rec)

df = pd.DataFrame(records)
print(f"  Loaded {len(df)} samples with BB temporal data")
for g in sorted(df["group"].unique()):
    print(f"    {g:12s}: {len(df[df['group']==g])}")

# =============================================================================
# STEP 2: Mean elution profiles per group at key wavelengths
# =============================================================================
print("\n" + "=" * 70)
print("STEP 2: Elution profiles within BB (23-26 min)")
print("=" * 70)

plot_groups = ["FR", "EX", "LAB", "LAB_T", "PTLL", "POST_O3", "HA", "FA"]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Elution profiles within BB zone (23-26 min) by wavelength", fontsize=14)

for i, w in enumerate([200, 210, 220, 230, 254, 280]):
    ax = axes[i // 3, i % 3]
    ax.set_title(f"{w} nm", fontsize=13, fontweight='bold')

    for g in plot_groups:
        if g not in bb_profiles or not bb_profiles[g]:
            continue
        # Interpolate all to common time grid
        all_profiles = []
        common_t = np.linspace(23.0, 26.0, 50)
        for sample in bb_profiles[g]:
            st = sample["time"]
            sp = sample["profiles"][w]
            interp = np.interp(common_t, st, sp)
            all_profiles.append(interp)

        arr = np.array(all_profiles)
        mean = np.mean(arr, axis=0)
        std = np.std(arr, axis=0)
        color = GROUP_COLORS.get(g, "gray")
        ax.plot(common_t, mean, color=color, label=f"{g} (n={len(arr)})", linewidth=2)
        ax.fill_between(common_t, mean - std, mean + std, alpha=0.15, color=color)

    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Absorbance (mAU)")
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)
    # Mark sub-windows
    for x in [24.0, 25.0]:
        ax.axvline(x, color='gray', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_elution_profiles_by_wl.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Plot 01: Elution profiles by wavelength")

# =============================================================================
# STEP 3: Organic vs Inorganic elution within BB
# =============================================================================
print("\n" + "=" * 70)
print("STEP 3: Organic vs Inorganic temporal evolution within BB")
print("=" * 70)

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle("Organic (A254) vs Inorganic (excess A210) elution within BB", fontsize=14)

for j, g in enumerate(plot_groups):
    ax = axes[j // 4, j % 4]
    if g not in bb_profiles or not bb_profiles[g]:
        ax.set_title(f"{g} — no data")
        continue

    common_t = np.linspace(23.0, 26.0, 50)
    all_254 = []
    all_210_inorg = []

    for sample in bb_profiles[g]:
        st = sample["time"]
        s254 = np.interp(common_t, st, sample["profiles"][254])
        s210 = np.interp(common_t, st, sample["profiles"][210])
        s210_org = s254 * ORG_A210_A254_RATIO
        s210_inorg = s210 - s210_org
        all_254.append(s254)
        all_210_inorg.append(s210_inorg)

    arr_254 = np.array(all_254)
    arr_inorg = np.array(all_210_inorg)

    mean_254 = np.mean(arr_254, axis=0)
    mean_inorg = np.mean(arr_inorg, axis=0)

    ax.plot(common_t, mean_254, color='#2ecc71', linewidth=2, label='Organic (A254)')
    ax.fill_between(common_t, mean_254 - np.std(arr_254, axis=0),
                     mean_254 + np.std(arr_254, axis=0), alpha=0.15, color='#2ecc71')

    ax2 = ax.twinx()
    ax2.plot(common_t, mean_inorg, color='#e74c3c', linewidth=2, label='Inorganic (excess A210)')
    ax2.fill_between(common_t, mean_inorg - np.std(arr_inorg, axis=0),
                      mean_inorg + np.std(arr_inorg, axis=0), alpha=0.15, color='#e74c3c')

    ax.set_title(f"{g} (n={len(arr_254)})", fontweight='bold')
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("A254 organic (mAU)", color='#2ecc71')
    ax2.set_ylabel("Excess A210 inorg (mAU)", color='#e74c3c')
    ax.tick_params(axis='y', labelcolor='#2ecc71')
    ax2.tick_params(axis='y', labelcolor='#e74c3c')

    for x in [24.0, 25.0]:
        ax.axvline(x, color='gray', linestyle='--', alpha=0.4)

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='best')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_organic_vs_inorganic_elution.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Plot 02: Organic vs Inorganic elution")

# =============================================================================
# STEP 4: Sub-window comparison table
# =============================================================================
print("\n" + "=" * 70)
print("STEP 4: Sub-window decomposition (Early/Mid/Late)")
print("=" * 70)

print(f"\n  {'Type':12s} | {'------- EARLY (23-24) -------':^30s} | {'-------- MID (24-25) --------':^30s} | {'------- LATE (25-26) --------':^30s}")
print(f"  {'':12s} |  A254_org  A210_inorg %inorg |  A254_org  A210_inorg %inorg |  A254_org  A210_inorg %inorg")
print(f"  {'-'*12} | {'-'*28} | {'-'*28} | {'-'*28}")

for g in ["FR", "EX", "LAB", "LAB_T", "PTLL", "POST_O3", "HA", "FA"]:
    sub = df[df["group"] == g]
    if len(sub) == 0:
        continue

    parts = []
    for label in ["early", "mid", "late"]:
        a254 = sub[f"{label}_org_A254"].mean()
        inorg = sub[f"{label}_inorg_A210"].mean()
        frac = sub[f"{label}_frac_inorg"].mean()
        parts.append(f"  {a254:7.3f}  {inorg:10.1f}  {frac:5.0%}")

    print(f"  {g:12s} | {parts[0]} | {parts[1]} | {parts[2]}")

# =============================================================================
# STEP 5: Heatmaps time x wavelength per group
# =============================================================================
print("\n" + "=" * 70)
print("STEP 5: Heatmaps time x wavelength")
print("=" * 70)

heatmap_groups = ["FR", "LAB", "LAB_T", "POST_O3", "HA", "NO3", "Br"]
heatmap_groups = [g for g in heatmap_groups if g in bb_profiles and bb_profiles[g]]

fig, axes = plt.subplots(2, 4, figsize=(22, 10))
fig.suptitle("Mean BB heatmap: Time (23-26 min) × Wavelength (200-300 nm)", fontsize=14)

common_t = np.linspace(23.0, 26.0, 50)
wl_range = np.arange(200, 302, 2)  # 200-300nm, 2nm steps

for j, g in enumerate(heatmap_groups):
    ax = axes[j // 4, j % 4]

    all_heatmaps = []
    for sample in bb_profiles[g]:
        wl = sample["wl"]
        bb_t = sample["time"]
        bb_d = sample["bb_data"]

        # Interpolate to common grid
        hm = np.zeros((len(common_t), len(wl_range)))
        for k, target_wl in enumerate(wl_range):
            wl_idx = get_wl_idx(wl, target_wl)
            hm[:, k] = np.interp(common_t, bb_t, bb_d[:, wl_idx])
        all_heatmaps.append(hm)

    mean_hm = np.mean(all_heatmaps, axis=0)

    # Clip for visualization
    vmax = np.percentile(np.abs(mean_hm), 95)
    im = ax.imshow(mean_hm, aspect='auto', origin='lower',
                   extent=[200, 300, 23.0, 26.0],
                   cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_title(f"{g} (n={len(all_heatmaps)})", fontweight='bold')
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Time (min)")
    plt.colorbar(im, ax=ax, label="mAU")

    # Mark key wavelengths
    for w in [210, 230, 254, 280]:
        ax.axvline(w, color='black', linestyle=':', alpha=0.3, linewidth=0.5)

# Hide unused axes
for j in range(len(heatmap_groups), 8):
    axes[j // 4, j % 4].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "03_heatmaps_time_wl.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Plot 03: Heatmaps time x wavelength")

# =============================================================================
# STEP 6: Temporal gradient — where does signal change fastest?
# =============================================================================
print("\n" + "=" * 70)
print("STEP 6: Temporal gradient (dA/dt) within BB")
print("=" * 70)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Temporal gradient dA/dt within BB zone (mAU/min)", fontsize=14)

for i, w in enumerate([200, 210, 220, 230, 254, 280]):
    ax = axes[i // 3, i % 3]
    ax.set_title(f"{w} nm — dA/dt", fontsize=13, fontweight='bold')

    for g in ["FR", "LAB", "LAB_T", "POST_O3", "HA", "FA"]:
        if g not in bb_profiles or not bb_profiles[g]:
            continue

        common_t = np.linspace(23.0, 26.0, 50)
        all_grads = []

        for sample in bb_profiles[g]:
            st = sample["time"]
            sp = sample["profiles"][w]
            interp = np.interp(common_t, st, sp)
            grad = np.gradient(interp, common_t)
            all_grads.append(grad)

        arr = np.array(all_grads)
        mean_grad = np.mean(arr, axis=0)
        color = GROUP_COLORS.get(g, "gray")
        ax.plot(common_t, mean_grad, color=color, label=g, linewidth=2)

    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("dA/dt (mAU/min)")
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    for x in [24.0, 25.0]:
        ax.axvline(x, color='gray', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "04_temporal_gradient.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Plot 04: Temporal gradient dA/dt")

# =============================================================================
# STEP 7: A254/A280 ratio evolution within BB
# =============================================================================
print("\n" + "=" * 70)
print("STEP 7: Spectral ratios evolution within BB")
print("=" * 70)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ratio_pairs = [
    (254, 280, "A254/A280\n(aromaticity character)"),
    (210, 254, "A210/A254\n(inorganic/organic proxy)"),
    (220, 254, "A220/A254\n(intermediate UV)"),
]

for k, (w1, w2, title) in enumerate(ratio_pairs):
    ax = axes[k]
    ax.set_title(title, fontsize=12, fontweight='bold')

    for g in ["FR", "LAB", "LAB_T", "POST_O3", "HA", "FA"]:
        if g not in bb_profiles or not bb_profiles[g]:
            continue

        common_t = np.linspace(23.0, 26.0, 50)
        all_ratios = []

        for sample in bb_profiles[g]:
            st = sample["time"]
            s1 = np.interp(common_t, st, sample["profiles"][w1])
            s2 = np.interp(common_t, st, sample["profiles"][w2])
            # Avoid division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = np.where(np.abs(s2) > 0.01, s1 / s2, np.nan)
            all_ratios.append(ratio)

        arr = np.array(all_ratios)
        mean_ratio = np.nanmean(arr, axis=0)
        color = GROUP_COLORS.get(g, "gray")
        ax.plot(common_t, mean_ratio, color=color, label=g, linewidth=2)

    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Ratio")
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    for x in [24.0, 25.0]:
        ax.axvline(x, color='gray', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "05_ratio_evolution.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Plot 05: Spectral ratio evolution within BB")

# =============================================================================
# STEP 8: Inorganic spectral shape evolution within BB
# =============================================================================
print("\n" + "=" * 70)
print("STEP 8: Inorganic spectral shape at early/mid/late BB")
print("=" * 70)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
inorg_wl = [200, 202, 204, 206, 208, 210, 212, 214, 216, 218, 220]

for k, (label, window) in enumerate([("Early BB (23-24)", BB_EARLY),
                                       ("Mid BB (24-25)", BB_MID),
                                       ("Late BB (25-26)", BB_LATE)]):
    ax = axes[k]
    ax.set_title(label, fontsize=12, fontweight='bold')

    for g in ["FR", "LAB", "LAB_T", "NO3", "Br"]:
        if g not in bb_profiles or not bb_profiles[g]:
            continue

        all_spectra = []
        for sample in bb_profiles[g]:
            wl = sample["wl"]
            st = sample["time"]
            bb_d = sample["bb_data"]
            mask = (st >= window[0]) & (st <= window[1])
            if np.sum(mask) < 2:
                continue
            spec = np.mean(bb_d[mask, :], axis=0)

            # Compute inorganic excess spectrum
            a254_idx = get_wl_idx(wl, 254)
            a254_val = spec[a254_idx]

            inorg_spec = []
            for w in inorg_wl:
                idx = get_wl_idx(wl, w)
                org_predicted = a254_val * ORG_A210_A254_RATIO * (w <= 210) + a254_val * 5.0 * (w > 210)
                # Simpler: just plot raw spectrum in the 200-220 range
                inorg_spec.append(spec[idx])
            all_spectra.append(inorg_spec)

        if not all_spectra:
            continue
        arr = np.array(all_spectra)
        mean = np.mean(arr, axis=0)
        color = GROUP_COLORS.get(g, "gray")
        ax.plot(inorg_wl, mean, color=color, label=g, linewidth=2, marker='o', markersize=3)

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Absorbance (mAU)")
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "06_inorganic_shape_subwindows.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Plot 06: Raw spectra 200-220nm at sub-windows")

# =============================================================================
# STEP 9: Correlation between wavelengths within BB (temporal)
# =============================================================================
print("\n" + "=" * 70)
print("STEP 9: Temporal correlation matrix between wavelengths at BB")
print("=" * 70)

corr_groups = ["FR", "LAB", "LAB_T"]
fig, axes = plt.subplots(1, len(corr_groups), figsize=(6 * len(corr_groups), 5))

corr_wl = [200, 210, 220, 230, 240, 254, 260, 270, 280, 300]

for j, g in enumerate(corr_groups):
    ax = axes[j]

    # Collect temporal profiles at each wavelength, stack all samples
    wl_series = {w: [] for w in corr_wl}
    for sample in bb_profiles.get(g, []):
        wl = sample["wl"]
        st = sample["time"]
        bb_d = sample["bb_data"]
        common_t = np.linspace(23.0, 26.0, 30)
        for w in corr_wl:
            idx = get_wl_idx(wl, w)
            interp = np.interp(common_t, st, bb_d[:, idx])
            wl_series[w].append(interp)

    # Build correlation matrix from concatenated temporal series
    n_wl = len(corr_wl)
    corr_mat = np.zeros((n_wl, n_wl))
    for a in range(n_wl):
        for b in range(n_wl):
            series_a = np.concatenate(wl_series[corr_wl[a]])
            series_b = np.concatenate(wl_series[corr_wl[b]])
            corr_mat[a, b] = np.corrcoef(series_a, series_b)[0, 1]

    im = ax.imshow(corr_mat, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(n_wl))
    ax.set_yticks(range(n_wl))
    ax.set_xticklabels([str(w) for w in corr_wl], rotation=45, fontsize=8)
    ax.set_yticklabels([str(w) for w in corr_wl], fontsize=8)
    ax.set_title(f"{g}", fontweight='bold')
    plt.colorbar(im, ax=ax, label="Pearson r")

    # Annotate
    for a in range(n_wl):
        for b in range(n_wl):
            val = corr_mat[a, b]
            color = 'white' if abs(val) > 0.5 else 'black'
            ax.text(b, a, f"{val:.2f}", ha='center', va='center', fontsize=6, color=color)

plt.suptitle("Temporal correlation between wavelengths within BB zone", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "07_temporal_correlation.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Plot 07: Temporal correlation matrix")

# =============================================================================
# STEP 10: Early-Late ratio per wavelength (temporal asymmetry)
# =============================================================================
print("\n" + "=" * 70)
print("STEP 10: Early/Late asymmetry within BB")
print("=" * 70)

print(f"\n  Ratio Early(23-24)/Late(25-26) per wavelength (>1 = more signal early)")
print(f"  {'Type':12s} |  A200   A210   A220   A230   A254   A280")
print(f"  {'-'*12} | {'-'*42}")

for g in ["FR", "EX", "LAB", "LAB_T", "PTLL", "POST_O3", "HA", "FA"]:
    sub = df[df["group"] == g]
    if len(sub) == 0:
        continue
    vals = []
    for w in [200, 210, 220, 230, 254, 280]:
        early = sub[f"early_{w}"].mean()
        late = sub[f"late_{w}"].mean()
        if abs(late) > 0.01:
            vals.append(f"{early/late:6.2f}")
        else:
            vals.append(f"  {'inf':>4s}")
    print(f"  {g:12s} | {'  '.join(vals)}")

# =============================================================================
# STEP 11: Inorganic species elution timing
# =============================================================================
print("\n" + "=" * 70)
print("STEP 11: Inorganic reference elution within BB")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for j, g in enumerate(["NO3", "Br"]):
    ax = axes[j]
    if g not in bb_profiles or not bb_profiles[g]:
        ax.set_title(f"{g} — no data")
        continue

    ax.set_title(f"{g} elution within BB (23-26 min)", fontweight='bold')
    for sample in bb_profiles[g]:
        st = sample["time"]
        for w in [200, 210, 220, 230]:
            ax.plot(st, sample["profiles"][w], label=f"{w}nm — {sample['filename']}", linewidth=1.5)

    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Absorbance (mAU)")
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linewidth=0.5)
    for x in [24.0, 25.0]:
        ax.axvline(x, color='gray', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "08_inorganic_ref_elution.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Plot 08: Inorganic reference elution")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("BB TEMPORAL EXPLORATION SUMMARY")
print("=" * 70)
print(f"\n  Plots saved to {OUTPUT_DIR}/")
print(f"  01: Elution profiles at 6 key wavelengths")
print(f"  02: Organic (A254) vs Inorganic (excess A210) elution")
print(f"  03: Heatmaps time × wavelength")
print(f"  04: Temporal gradient dA/dt")
print(f"  05: Spectral ratio evolution (A254/A280, A210/A254, A220/A254)")
print(f"  06: Raw spectra 200-220nm at early/mid/late BB")
print(f"  07: Temporal correlation between wavelengths")
print(f"  08: Inorganic reference (NO3, Br) elution detail")
