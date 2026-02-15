# -*- coding: utf-8 -*-
"""
Humic Fraction Deep Analysis — HPSEC-DAD Export3D
==================================================
Anàlisi en profunditat de la fracció HS (18-23 min) per classificar
tipologia d'aigua basant-se en la signatura espectral dels húmics.

Compara patrons HA/FA de referència amb mostres reals per:
1. Caracteritzar la signatura espectral HA vs FA
2. Construir índexs espectrals de tipologia húmica
3. Estimar el caràcter HA-like vs FA-like de cada mostra
4. Classificar aigües per origen/tractament via empremta húmica

Autor: HPSEC Suite
"""

import os
import sys
import re
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from collections import Counter, defaultdict
from scipy import stats
from scipy.signal import savgol_filter
from scipy.spatial.distance import cosine

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, LeaveOneOut, cross_val_predict
from sklearn.metrics import silhouette_score, confusion_matrix

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_FOLDERS = [
    "C:/Users/Lequia/Desktop/Dades3",
    "C:/Users/Lequia/OneDrive - Universitat de Girona/UdG365_HPLC-DAD - General/Dades",
]
OUTPUT_DIR = "C:/Users/Lequia/Desktop/HPSEC/rf_humic_analysis"

# Fraction windows (min)
FRACTIONS = {
    "BioP": (10.8, 18.0),
    "HS":   (18.0, 23.0),
    "BB":   (23.0, 26.0),
    "SB":   (26.0, 32.0),
    "LMW":  (32.0, 60.0),
}

# HS sub-windows for finer analysis (within 18-23 min)
HS_SUBFRACTIONS = {
    "HS_early": (18.0, 19.5),   # Higher MW humics
    "HS_mid":   (19.5, 21.0),   # Medium MW humics
    "HS_late":  (21.0, 23.0),   # Lower MW humics (transition to BB)
}

# Key wavelength regions for humic characterization
WL_REGIONS = {
    "UV-C":     (200, 230),  # Protein-like, nitrate interference
    "UV-B":     (230, 280),  # Aromatic structures, SUVA
    "UV-A":     (280, 320),  # Phenolic, humic-like
    "Vis_near": (320, 400),  # Chromophoric, high MW humics
}

# Spectral indices (literature-based)
SPECTRAL_INDICES = {
    "SUVA254":    254,        # Specific UV absorbance (aromaticity)
    "SUVA280":    280,        # Phenolic content
    "E2_E3":     (250, 365),  # MW proxy (Peuravuori & Pihlaja, 1997)
    "E4_E6":     (465, 665),  # Humification (Kononova, 1966) — outside our range
    "A254_A280":  (254, 280),  # Aromaticity ratio
    "A210_A254":  (210, 254),  # UV-C contribution
    "A250_A365":  (250, 365),  # Same as E2/E3
    "S275_295":  (275, 295),  # Spectral slope (Helms et al. 2008)
    "S350_400":  (350, 400),  # Spectral slope high λ
    "SR":        None,        # Slope ratio S275-295 / S350-400
}

WL_ALL = list(range(200, 402, 2))

# Sample type patterns — HA/FA patterns are key references
TYPE_PATTERNS = [
    # Reference standards
    ("SRHA",     r"(?i)^(SRHA|3S101H)"),
    ("SRFA",     r"(?i)^(SRFA|3S101F)"),
    ("HA",       r"(?i)(?:^|[_\-\s])HA(?:[_\-\s\.]|$|\d)"),
    ("FA",       r"(?i)(?:^|[_\-\s])FA(?:[_\-\s\.]|$|\d)"),
    ("MIX",      r"(?i)(?:^|[_\-\s])MIX(?:[_\-\s\.]|$|\d)"),
    # Inorganics (for contrast)
    ("NO3",      r"(?i)^(NO3|NaNO3)"),
    ("Br",       r"(?i)^(Br[_\.\s\d]|KBr)"),
    ("CaCO3",    r"(?i)^CaCO3"),
    ("Na2CO3",   r"(?i)^Na2CO3"),
    ("Na2SO4",   r"(?i)^Na2SO4"),
    ("NaOH",     r"(?i)^NaOH"),
    # Real water samples
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
    ("EP_PTL",   r"(?i)^EP[_\-]?PTL"),
    ("EP_PTT",   r"(?i)^EP[_\-]?PTT"),
    ("EP",       r"(?i)^EP[_\s\-]"),
    ("SP",       r"(?i)^SP[_\s\-]"),
    ("POST_O3",  r"(?i)^POST"),
    ("FS",       r"(?i)^FS\d"),
    ("SFiltres", r"(?i)^SFiltres"),
    ("KHP",      r"(?i)^KHP"),
    ("BLANK",    r"(?i)^(MQ|BLANK|BLK|H2O)"),
]


def classify_sample(filename):
    """Classify sample by filename pattern."""
    base = re.sub(r'[_\s]*(R?\d+)\.(csv|CSV)$', '', filename)
    for label, pattern in TYPE_PATTERNS:
        if re.search(pattern, base):
            return label
    return "OTHER"


def load_export3d(filepath):
    """Load Export3D CSV. Returns (time_array, wavelength_array, data_matrix) or None."""
    try:
        df = pd.read_csv(filepath, encoding='utf-16', sep=',', header=None)
        if df.shape[1] < 50:
            return None
        wavelengths = df.iloc[0, 1:].values.astype(float)
        time_vals = df.iloc[1:, 0].values.astype(float)
        data = df.iloc[1:, 1:].values.astype(float)
        if len(time_vals) < 100 or np.max(time_vals) < 15:
            return None  # BP mode, skip
        return time_vals, wavelengths, data
    except:
        return None


def extract_fraction_spectrum(time_vals, wavelengths, data, t_start, t_end):
    """Extract mean spectrum for a time window."""
    mask = (time_vals >= t_start) & (time_vals <= t_end)
    if np.sum(mask) < 3:
        return None
    return np.mean(data[mask, :], axis=0)


def compute_spectral_slope(spectrum, wavelengths, wl_start, wl_end):
    """Compute spectral slope (log-linear) for a wavelength range."""
    mask = (wavelengths >= wl_start) & (wavelengths <= wl_end)
    wl_sub = wavelengths[mask]
    spec_sub = spectrum[mask]
    # Only positive values for log
    valid = spec_sub > 0
    if np.sum(valid) < 3:
        return np.nan
    slope, _, _, _, _ = stats.linregress(wl_sub[valid], np.log(spec_sub[valid]))
    return slope


def compute_humic_features(time_vals, wavelengths, data):
    """Extract comprehensive humic characterization features from Export3D."""
    features = {}
    wl = wavelengths.astype(float)

    # --- Full HS fraction spectrum ---
    hs_spec = extract_fraction_spectrum(time_vals, wl, data, 18.0, 23.0)
    if hs_spec is None:
        return None

    # --- HS sub-fraction spectra ---
    for name, (t0, t1) in HS_SUBFRACTIONS.items():
        sub_spec = extract_fraction_spectrum(time_vals, wl, data, t0, t1)
        if sub_spec is not None:
            for w in WL_ALL:
                idx = np.argmin(np.abs(wl - w))
                if idx < len(sub_spec):
                    features[f"spec_{name}_{int(wl[idx])}"] = sub_spec[idx]

    # --- HS full spectrum (for fingerprinting) ---
    for w in WL_ALL:
        idx = np.argmin(np.abs(wl - w))
        if idx < len(hs_spec):
            features[f"HS_{int(wl[idx])}"] = hs_spec[idx]

    # --- BB and BioP for context ---
    for frac_name, (t0, t1) in [("BB", (23, 26)), ("BioP", (10.8, 18))]:
        fspec = extract_fraction_spectrum(time_vals, wl, data, t0, t1)
        if fspec is not None:
            for w in [200, 210, 220, 230, 240, 250, 254, 260, 270, 280, 300, 330, 350, 365, 400]:
                idx = np.argmin(np.abs(wl - w))
                if idx < len(fspec):
                    features[f"{frac_name}_{int(wl[idx])}"] = fspec[idx]

    # --- Spectral ratios (on HS fraction) ---
    def get_wl_val(spec, target_wl):
        idx = np.argmin(np.abs(wl - target_wl))
        return spec[idx] if idx < len(spec) else np.nan

    a210 = get_wl_val(hs_spec, 210)
    a220 = get_wl_val(hs_spec, 220)
    a230 = get_wl_val(hs_spec, 230)
    a250 = get_wl_val(hs_spec, 250)
    a254 = get_wl_val(hs_spec, 254)
    a260 = get_wl_val(hs_spec, 260)
    a265 = get_wl_val(hs_spec, 265)
    a272 = get_wl_val(hs_spec, 272)
    a280 = get_wl_val(hs_spec, 280)
    a300 = get_wl_val(hs_spec, 300)
    a330 = get_wl_val(hs_spec, 330)
    a350 = get_wl_val(hs_spec, 350)
    a365 = get_wl_val(hs_spec, 365)
    a400 = get_wl_val(hs_spec, 400)

    # Classic ratios
    features["E2_E3"] = a250 / a365 if a365 > 0.01 else np.nan
    features["A254_A280"] = a254 / a280 if a280 > 0.01 else np.nan
    features["A210_A254"] = a210 / a254 if a254 > 0.01 else np.nan
    features["A254_A365"] = a254 / a365 if a365 > 0.01 else np.nan
    features["A280_A350"] = a280 / a350 if a350 > 0.01 else np.nan
    features["A250_A365"] = a250 / a365 if a365 > 0.01 else np.nan
    features["A272_A280"] = a272 / a280 if a280 > 0.01 else np.nan
    features["A300_A400"] = a300 / a400 if a400 > 0.01 else np.nan
    features["A220_A280"] = a220 / a280 if a280 > 0.01 else np.nan
    features["A230_A260"] = a230 / a260 if a260 > 0.01 else np.nan

    # Spectral slopes (Helms et al. 2008)
    S275_295 = compute_spectral_slope(hs_spec, wl, 275, 295)
    S350_400 = compute_spectral_slope(hs_spec, wl, 350, 400)
    S206_240 = compute_spectral_slope(hs_spec, wl, 206, 240)

    features["S275_295"] = S275_295
    features["S350_400"] = S350_400
    features["S206_240"] = S206_240
    features["SR"] = S275_295 / S350_400 if (S350_400 and abs(S350_400) > 1e-6) else np.nan

    # --- HS shape descriptors ---
    # Normalized spectrum (shape only, independent of concentration)
    if a254 > 0.01:
        norm_spec = hs_spec / a254
        features["shape_peak_wl"] = wl[np.argmax(norm_spec[:50])]  # Peak below 300nm
        features["shape_UV_area"] = np.trapz(norm_spec[(wl >= 200) & (wl <= 280)],
                                              wl[(wl >= 200) & (wl <= 280)])
        features["shape_Vis_area"] = np.trapz(norm_spec[(wl >= 320) & (wl <= 400)],
                                               wl[(wl >= 320) & (wl <= 400)])
        features["shape_UV_Vis_ratio"] = features["shape_UV_area"] / features["shape_Vis_area"] if features["shape_Vis_area"] > 0.01 else np.nan

    # --- HS elution profile at key wavelengths ---
    # How does the signal change across the HS window? (shape of the peak)
    hs_mask = (time_vals >= 18.0) & (time_vals <= 23.0)
    hs_time = time_vals[hs_mask]
    hs_data = data[hs_mask, :]

    for target_wl in [210, 254, 280, 350]:
        wl_idx = np.argmin(np.abs(wl - target_wl))
        profile = hs_data[:, wl_idx]
        if len(profile) > 10 and np.max(profile) > 0.01:
            # Normalize profile
            norm_profile = profile / np.max(profile)
            peak_time = hs_time[np.argmax(profile)]
            # Asymmetry: area before peak vs after peak
            peak_idx = np.argmax(profile)
            area_before = np.trapz(norm_profile[:peak_idx+1]) if peak_idx > 0 else 0
            area_after = np.trapz(norm_profile[peak_idx:]) if peak_idx < len(norm_profile)-1 else 0
            total = area_before + area_after
            features[f"elution_peak_t_{target_wl}"] = peak_time
            features[f"elution_asym_{target_wl}"] = area_before / total if total > 0 else 0.5
            # Width at half maximum
            half_max = 0.5
            above = norm_profile >= half_max
            if np.any(above):
                first = hs_time[above][0]
                last = hs_time[above][-1]
                features[f"elution_fwhm_{target_wl}"] = last - first

    # --- Sub-fraction ratios (early vs late HS) ---
    hs_early = extract_fraction_spectrum(time_vals, wl, data, 18.0, 19.5)
    hs_late = extract_fraction_spectrum(time_vals, wl, data, 21.0, 23.0)
    if hs_early is not None and hs_late is not None:
        for target_wl in [210, 230, 254, 280, 350]:
            idx = np.argmin(np.abs(wl - target_wl))
            e = hs_early[idx]
            l = hs_late[idx]
            features[f"HS_early_late_{target_wl}"] = e / l if l > 0.01 else np.nan

    return features


def scan_all_export3d(folders):
    """Scan all Export3D directories, deduplicate, return file list."""
    files = []
    seen = set()
    for base_folder in folders:
        if not os.path.isdir(base_folder):
            continue
        for entry in sorted(os.listdir(base_folder)):
            entry_path = os.path.join(base_folder, entry)
            if not os.path.isdir(entry_path) or '_SEQ' not in entry:
                continue
            if entry.startswith('_'):
                continue
            exp_dir = os.path.join(entry_path, 'Export3d')
            if not os.path.isdir(exp_dir):
                continue
            csvs = glob.glob(os.path.join(exp_dir, '*.csv')) + glob.glob(os.path.join(exp_dir, '*.CSV'))
            for f in sorted(csvs):
                fname = os.path.basename(f)
                key = (entry, fname.upper())
                if key not in seen:
                    seen.add(key)
                    sample_type = classify_sample(fname)
                    files.append({
                        'path': f,
                        'filename': fname,
                        'seq': entry.replace('_SEQ_BP', '_SEQ').replace('_SEQ', ''),
                        'seq_folder': entry,
                        'type': sample_type,
                    })
    return files


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # =====================================================================
    # STEP 1: Scan and load
    # =====================================================================
    print("=" * 70)
    print("STEP 1: Scanning Export3D files...")
    print("=" * 70)

    all_files = scan_all_export3d(DATA_FOLDERS)
    # Exclude KHP, BLANK, NaOH, and inorganics with no organic signal
    exclude = {"KHP", "BLANK", "NaOH"}
    files = [f for f in all_files if f['type'] not in exclude]

    type_counts = Counter(f['type'] for f in files)
    print(f"Total files (excluding KHP/BLANK/NaOH): {len(files)}")
    for t, n in type_counts.most_common():
        print(f"  {t:15s}: {n:4d}")

    # =====================================================================
    # STEP 2: Extract humic features
    # =====================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Extracting humic characterization features...")
    print("=" * 70)

    records = []
    for i, f in enumerate(files):
        if (i + 1) % 100 == 0:
            print(f"  Processing {i+1}/{len(files)}: {f['filename']}...")
        result = load_export3d(f['path'])
        if result is None:
            continue
        time_vals, wavelengths, data = result
        feats = compute_humic_features(time_vals, wavelengths, data)
        if feats is None:
            continue
        feats['type'] = f['type']
        feats['seq'] = f['seq']
        feats['filename'] = f['filename']
        records.append(feats)

    df = pd.DataFrame(records)
    print(f"\nSuccessfully processed: {len(df)} samples")
    print(f"Features: {len([c for c in df.columns if c not in ['type', 'seq', 'filename']])}")

    # Save
    df.to_csv(os.path.join(OUTPUT_DIR, "humic_features.csv"), index=False)

    # Separate metadata and features
    meta_cols = ['type', 'seq', 'filename']
    feat_cols = [c for c in df.columns if c not in meta_cols]
    X = df[feat_cols].values
    types = df['type'].values

    # Replace inf/nan
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    # =====================================================================
    # STEP 3: HA vs FA reference comparison
    # =====================================================================
    print("\n" + "=" * 70)
    print("STEP 3: HA vs FA reference spectral signatures")
    print("=" * 70)

    # Get HS spectra columns
    hs_cols = [c for c in feat_cols if c.startswith("HS_") and c[3:].isdigit()]
    hs_wl = np.array([int(c.split("_")[1]) for c in hs_cols])

    # Reference types
    ref_types = ["HA", "FA", "SRHA", "SRFA", "MIX"]
    # Water types for comparison
    water_types = ["FR", "PTLL", "PTT", "SK", "EX", "ATL_2024", "LQ", "EXT",
                   "EP", "SP", "POST_O3"]

    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    # 3a. Mean HS spectra: HA vs FA vs SRHA vs SRFA
    ax1 = fig.add_subplot(gs[0, 0:2])
    colors_ref = {'HA': '#E74C3C', 'FA': '#3498DB', 'SRHA': '#C0392B',
                  'SRFA': '#2980B9', 'MIX': '#9B59B6'}
    for ref_type in ref_types:
        mask = types == ref_type
        if np.sum(mask) < 1:
            continue
        mean_spec = df.loc[mask, hs_cols].mean().values
        std_spec = df.loc[mask, hs_cols].std().values
        ax1.plot(hs_wl, mean_spec, color=colors_ref.get(ref_type, 'gray'),
                 linewidth=2.5, label=f"{ref_type} (n={np.sum(mask)})")
        if np.sum(mask) > 1:
            ax1.fill_between(hs_wl, mean_spec - std_spec, mean_spec + std_spec,
                           alpha=0.15, color=colors_ref.get(ref_type, 'gray'))
    ax1.set_xlabel("Wavelength (nm)")
    ax1.set_ylabel("Absorbance (mAU)")
    ax1.set_title("HS Fraction Mean Spectra: Reference Standards")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 3b. Normalized spectra (shape comparison)
    ax2 = fig.add_subplot(gs[0, 2])
    for ref_type in ref_types:
        mask = types == ref_type
        if np.sum(mask) < 1:
            continue
        mean_spec = df.loc[mask, hs_cols].mean().values
        idx_254 = np.argmin(np.abs(hs_wl - 254))
        if mean_spec[idx_254] > 0.01:
            norm_spec = mean_spec / mean_spec[idx_254]
            ax2.plot(hs_wl, norm_spec, color=colors_ref.get(ref_type, 'gray'),
                     linewidth=2, label=ref_type)
    ax2.set_xlabel("Wavelength (nm)")
    ax2.set_ylabel("Normalized (A/A254)")
    ax2.set_title("Shape Comparison (normalized at 254nm)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # 3c. Key ratio comparison: HA vs FA
    ax3 = fig.add_subplot(gs[1, 0])
    ratio_features = ["E2_E3", "A254_A280", "A210_A254", "S275_295", "SR"]
    ratio_data = {}
    for ref_type in ["HA", "FA", "SRHA", "SRFA"]:
        mask = types == ref_type
        if np.sum(mask) < 1:
            continue
        vals = {}
        for rf in ratio_features:
            if rf in df.columns:
                v = df.loc[mask, rf].dropna().values
                if len(v) > 0:
                    vals[rf] = (np.mean(v), np.std(v))
        ratio_data[ref_type] = vals

    x_pos = np.arange(len(ratio_features))
    width = 0.2
    for i, ref_type in enumerate(["HA", "FA", "SRHA", "SRFA"]):
        if ref_type not in ratio_data:
            continue
        means = [ratio_data[ref_type].get(rf, (0, 0))[0] for rf in ratio_features]
        stds = [ratio_data[ref_type].get(rf, (0, 0))[1] for rf in ratio_features]
        ax3.bar(x_pos + i * width, means, width, yerr=stds,
                label=ref_type, color=colors_ref.get(ref_type, 'gray'), alpha=0.8)
    ax3.set_xticks(x_pos + 1.5 * width)
    ax3.set_xticklabels(ratio_features, rotation=45, ha='right', fontsize=8)
    ax3.set_title("Spectral Indices: HA vs FA")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, axis='y')

    # 3d. HS sub-fraction comparison (early vs mid vs late)
    ax4 = fig.add_subplot(gs[1, 1])
    for ref_type in ["HA", "FA"]:
        mask = types == ref_type
        if np.sum(mask) < 1:
            continue
        for subfrac, ls in [("HS_early", "-"), ("HS_mid", "--"), ("HS_late", ":")]:
            sub_cols = [c for c in feat_cols if c.startswith(f"spec_{subfrac}_")]
            if not sub_cols:
                continue
            sub_wl = np.array([int(c.split("_")[-1]) for c in sub_cols])
            mean_spec = df.loc[mask, sub_cols].mean().values
            ax4.plot(sub_wl, mean_spec, color=colors_ref[ref_type],
                     linestyle=ls, linewidth=1.5,
                     label=f"{ref_type} {subfrac.split('_')[1]}")
    ax4.set_xlabel("Wavelength (nm)")
    ax4.set_ylabel("Absorbance (mAU)")
    ax4.set_title("HS Sub-fractions: Early/Mid/Late")
    ax4.legend(fontsize=7, ncol=2)
    ax4.grid(True, alpha=0.3)

    # 3e. Elution profiles at 254nm
    ax5 = fig.add_subplot(gs[1, 2])
    elution_cols_254 = ["elution_peak_t_254", "elution_asym_254", "elution_fwhm_254"]
    for ref_type in ["HA", "FA", "SRHA", "SRFA"]:
        mask = types == ref_type
        if np.sum(mask) < 1:
            continue
        peak_t = df.loc[mask, "elution_peak_t_254"].dropna().values
        fwhm = df.loc[mask, "elution_fwhm_254"].dropna().values
        if len(peak_t) > 0 and len(fwhm) > 0:
            ax5.scatter(np.mean(peak_t), np.mean(fwhm),
                       s=150, marker='o', color=colors_ref.get(ref_type, 'gray'),
                       edgecolors='black', linewidth=1, zorder=5,
                       label=f"{ref_type} (t={np.mean(peak_t):.1f}, w={np.mean(fwhm):.1f})")
    # Add water types
    for wt in water_types:
        mask = types == wt
        if np.sum(mask) < 3:
            continue
        peak_t = df.loc[mask, "elution_peak_t_254"].dropna().values
        fwhm = df.loc[mask, "elution_fwhm_254"].dropna().values
        if len(peak_t) > 0 and len(fwhm) > 0:
            ax5.scatter(peak_t, fwhm, s=20, alpha=0.5, label=f"{wt} (n={len(peak_t)})")
    ax5.set_xlabel("Peak elution time (min)")
    ax5.set_ylabel("FWHM (min)")
    ax5.set_title("HS Elution Profile @ 254nm")
    ax5.legend(fontsize=6, ncol=2, loc='upper right')
    ax5.grid(True, alpha=0.3)

    plt.savefig(os.path.join(OUTPUT_DIR, "01_HA_vs_FA_reference.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 01_HA_vs_FA_reference.png")

    # =====================================================================
    # STEP 4: PCA on HS fraction only
    # =====================================================================
    print("\n" + "=" * 70)
    print("STEP 4: PCA on HS fraction spectra")
    print("=" * 70)

    X_hs = df[hs_cols].values
    X_hs = np.nan_to_num(X_hs, nan=0, posinf=0, neginf=0)

    scaler = StandardScaler()
    X_hs_scaled = scaler.fit_transform(X_hs)

    pca = PCA(n_components=min(10, X_hs_scaled.shape[1]))
    pca_scores = pca.fit_transform(X_hs_scaled)

    print(f"PCA on HS spectra ({X_hs.shape[1]} wavelengths):")
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    for i in range(min(5, len(cumvar))):
        print(f"  PC{i+1}: {pca.explained_variance_ratio_[i]*100:.1f}% (cumulative: {cumvar[i]:.1f}%)")

    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # 4a. PC1 vs PC2 colored by type
    ax1 = fig.add_subplot(gs[0, 0:2])

    # Color scheme: references bold, waters lighter
    all_types = sorted(set(types))
    cmap = plt.cm.get_cmap('tab20', len(all_types))
    type_colors = {t: cmap(i) for i, t in enumerate(all_types)}
    # Override references
    type_colors.update({'HA': '#E74C3C', 'FA': '#3498DB', 'SRHA': '#C0392B',
                        'SRFA': '#2980B9', 'MIX': '#9B59B6'})

    # Plot water types first (background)
    for t in all_types:
        if t in ref_types:
            continue
        mask = types == t
        if np.sum(mask) < 2:
            continue
        ax1.scatter(pca_scores[mask, 0], pca_scores[mask, 1],
                   s=15, alpha=0.4, color=type_colors[t], label=f"{t} ({np.sum(mask)})")

    # Plot references on top
    for t in ref_types:
        mask = types == t
        if np.sum(mask) < 1:
            continue
        ax1.scatter(pca_scores[mask, 0], pca_scores[mask, 1],
                   s=120, marker='*', color=type_colors[t], edgecolors='black',
                   linewidth=0.5, zorder=10, label=f"{t} ({np.sum(mask)})")

    ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax1.set_title("PCA on HS Fraction Spectra — All Samples")
    ax1.legend(fontsize=6, ncol=3, loc='upper right')
    ax1.grid(True, alpha=0.3)

    # 4b. Loadings — what wavelengths drive each PC
    ax2 = fig.add_subplot(gs[0, 2])
    for pc_idx, color in [(0, '#E74C3C'), (1, '#3498DB'), (2, '#2ECC71')]:
        ax2.plot(hs_wl, pca.components_[pc_idx], color=color,
                linewidth=1.5, label=f"PC{pc_idx+1}")
    ax2.axhline(0, color='gray', linewidth=0.5)
    ax2.set_xlabel("Wavelength (nm)")
    ax2.set_ylabel("Loading")
    ax2.set_title("PCA Loadings (HS fraction)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 4c. PC1 vs PC2 — only HA/FA families + nearest waters
    ax3 = fig.add_subplot(gs[1, 0:2])

    # Compute mean HA and FA positions in PCA space
    ha_mask = np.isin(types, ['HA', 'SRHA'])
    fa_mask = np.isin(types, ['FA', 'SRFA'])

    if np.sum(ha_mask) > 0 and np.sum(fa_mask) > 0:
        ha_center = np.mean(pca_scores[ha_mask, :2], axis=0)
        fa_center = np.mean(pca_scores[fa_mask, :2], axis=0)

        # For each water sample, compute HA-likeness vs FA-likeness
        ha_fa_scores = []
        for i in range(len(pca_scores)):
            d_ha = np.linalg.norm(pca_scores[i, :2] - ha_center)
            d_fa = np.linalg.norm(pca_scores[i, :2] - fa_center)
            # Score: 0 = pure FA-like, 1 = pure HA-like
            total = d_ha + d_fa
            ha_likeness = 1 - (d_ha / total) if total > 0 else 0.5
            ha_fa_scores.append(ha_likeness)
        ha_fa_scores = np.array(ha_fa_scores)

        # Color by HA-likeness
        scatter = ax3.scatter(pca_scores[:, 0], pca_scores[:, 1],
                             c=ha_fa_scores, cmap='RdYlBu_r', s=20, alpha=0.6,
                             vmin=0.3, vmax=0.7)
        plt.colorbar(scatter, ax=ax3, label="HA-like ← → FA-like", shrink=0.8)

        # Mark references
        ax3.scatter(pca_scores[ha_mask, 0], pca_scores[ha_mask, 1],
                   s=200, marker='*', c='red', edgecolors='black', zorder=10, label="HA/SRHA")
        ax3.scatter(pca_scores[fa_mask, 0], pca_scores[fa_mask, 1],
                   s=200, marker='*', c='blue', edgecolors='black', zorder=10, label="FA/SRFA")
        ax3.plot(*ha_center, 'rX', markersize=20, markeredgecolor='black', zorder=11)
        ax3.plot(*fa_center, 'bX', markersize=20, markeredgecolor='black', zorder=11)

    ax3.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax3.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax3.set_title("HA-likeness vs FA-likeness (PCA proximity)")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # 4d. Distribution of HA-likeness by water type
    ax4 = fig.add_subplot(gs[1, 2])
    if np.sum(ha_mask) > 0 and np.sum(fa_mask) > 0:
        water_ha_scores = {}
        for wt in water_types:
            mask = types == wt
            if np.sum(mask) >= 3:
                water_ha_scores[wt] = ha_fa_scores[mask]

        if water_ha_scores:
            sorted_types = sorted(water_ha_scores.keys(),
                                 key=lambda t: np.median(water_ha_scores[t]))
            positions = range(len(sorted_types))
            bp = ax4.boxplot([water_ha_scores[t] for t in sorted_types],
                            positions=positions, vert=True, widths=0.6,
                            patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('#AED6F1')
            ax4.set_xticks(positions)
            ax4.set_xticklabels(sorted_types, rotation=45, ha='right', fontsize=8)
            ax4.axhline(0.5, color='gray', linestyle='--', linewidth=1, label="HA/FA boundary")
            ax4.set_ylabel("HA-likeness score")
            ax4.set_title("Water Type HA/FA Character")
            ax4.legend(fontsize=8)
            ax4.grid(True, alpha=0.3, axis='y')

    plt.savefig(os.path.join(OUTPUT_DIR, "02_PCA_HS_fraction.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 02_PCA_HS_fraction.png")

    # =====================================================================
    # STEP 5: Cosine similarity to HA/FA templates
    # =====================================================================
    print("\n" + "=" * 70)
    print("STEP 5: Spectral similarity to HA/FA templates")
    print("=" * 70)

    # Build template spectra
    ha_types = ['HA', 'SRHA']
    fa_types = ['FA', 'SRFA']

    ha_template = df.loc[np.isin(types, ha_types), hs_cols].mean().values
    fa_template = df.loc[np.isin(types, fa_types), hs_cols].mean().values

    # Cosine similarity of each sample to HA and FA templates
    sim_ha = []
    sim_fa = []
    for i in range(len(df)):
        spec = df.iloc[i][hs_cols].values.astype(float)
        sim_ha.append(1 - cosine(spec, ha_template) if np.any(spec > 0) else 0)
        sim_fa.append(1 - cosine(spec, fa_template) if np.any(spec > 0) else 0)

    sim_ha = np.array(sim_ha)
    sim_fa = np.array(sim_fa)

    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # 5a. Sim(HA) vs Sim(FA) scatter
    ax1 = fig.add_subplot(gs[0, 0:2])
    for t in all_types:
        mask = types == t
        if np.sum(mask) < 2:
            continue
        marker = '*' if t in ref_types else 'o'
        size = 120 if t in ref_types else 20
        alpha = 0.9 if t in ref_types else 0.5
        ax1.scatter(sim_fa[mask], sim_ha[mask], s=size, alpha=alpha,
                   color=type_colors[t], marker=marker, label=f"{t} ({np.sum(mask)})")
    ax1.plot([0.9, 1.0], [0.9, 1.0], 'k--', alpha=0.3)
    ax1.set_xlabel("Cosine Similarity to FA template")
    ax1.set_ylabel("Cosine Similarity to HA template")
    ax1.set_title("Spectral Similarity: HA vs FA Templates")
    ax1.legend(fontsize=6, ncol=3, loc='lower left')
    ax1.grid(True, alpha=0.3)

    # 5b. HA/FA character index = sim_HA - sim_FA per water type
    ax2 = fig.add_subplot(gs[0, 2])
    ha_fa_index = sim_ha - sim_fa  # positive = more HA-like

    water_index = {}
    for wt in water_types:
        mask = types == wt
        if np.sum(mask) >= 3:
            water_index[wt] = ha_fa_index[mask]

    if water_index:
        sorted_wt = sorted(water_index.keys(), key=lambda t: np.median(water_index[t]))
        bp = ax2.boxplot([water_index[t] for t in sorted_wt],
                        vert=True, widths=0.6, patch_artist=True)
        colors_box = []
        for t in sorted_wt:
            med = np.median(water_index[t])
            if med > 0.005:
                colors_box.append('#FADBD8')  # HA-like
            elif med < -0.005:
                colors_box.append('#D6EAF8')  # FA-like
            else:
                colors_box.append('#D5F5E3')  # Neutral
        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)
        ax2.set_xticklabels(sorted_wt, rotation=45, ha='right', fontsize=8)
        ax2.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax2.set_ylabel("HA-FA index (>0 = HA-like)")
        ax2.set_title("Humic Character by Water Type")
        ax2.grid(True, alpha=0.3, axis='y')

    # 5c. HA/FA index temporal evolution for major types
    ax3 = fig.add_subplot(gs[1, 0:2])
    for wt in ["FR", "PTLL", "PTT", "ATL_2024"]:
        mask = types == wt
        if np.sum(mask) < 5:
            continue
        seqs = df.loc[mask, 'seq'].values
        # Extract numeric part
        seq_nums = []
        for s in seqs:
            m = re.match(r'(\d+)', s)
            seq_nums.append(int(m.group(1)) if m else 0)
        seq_nums = np.array(seq_nums)
        order = np.argsort(seq_nums)
        ax3.scatter(seq_nums[order], ha_fa_index[mask][order],
                   s=20, alpha=0.5, label=wt)
        # Rolling mean
        if len(order) > 5:
            from scipy.ndimage import uniform_filter1d
            smooth = uniform_filter1d(ha_fa_index[mask][order].astype(float), size=5)
            ax3.plot(seq_nums[order], smooth, linewidth=2, alpha=0.8)
    ax3.axhline(0, color='gray', linestyle='--')
    ax3.set_xlabel("SEQ number")
    ax3.set_ylabel("HA-FA index")
    ax3.set_title("Temporal Evolution of Humic Character")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # 5d. Spectral differences: HA-like vs FA-like waters
    ax4 = fig.add_subplot(gs[1, 2])
    # Take top/bottom 20% of water samples by HA-FA index
    water_mask = np.isin(types, water_types)
    water_indices = ha_fa_index[water_mask]
    water_specs = df.loc[water_mask, hs_cols].values

    if len(water_indices) > 20:
        p20 = np.percentile(water_indices, 20)
        p80 = np.percentile(water_indices, 80)
        ha_like = water_specs[water_indices >= p80]
        fa_like = water_specs[water_indices <= p20]
        ha_mean = np.mean(ha_like, axis=0)
        fa_mean = np.mean(fa_like, axis=0)
        # Normalize both
        ha_norm = ha_mean / ha_mean[np.argmin(np.abs(hs_wl - 254))]
        fa_norm = fa_mean / fa_mean[np.argmin(np.abs(hs_wl - 254))]
        ax4.plot(hs_wl, ha_norm, color='red', linewidth=2, label=f"HA-like waters (n={len(ha_like)})")
        ax4.plot(hs_wl, fa_norm, color='blue', linewidth=2, label=f"FA-like waters (n={len(fa_like)})")
        ax4.plot(hs_wl, ha_norm - fa_norm, color='green', linewidth=1.5,
                linestyle='--', label="Difference")
        ax4.axhline(0, color='gray', linewidth=0.5)
        ax4.set_xlabel("Wavelength (nm)")
        ax4.set_ylabel("Normalized absorbance (A/A254)")
        ax4.set_title("HA-like vs FA-like Water Spectra")
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)

    plt.savefig(os.path.join(OUTPUT_DIR, "03_HA_FA_similarity.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 03_HA_FA_similarity.png")

    # =====================================================================
    # STEP 6: RF classifier — can we predict water origin from HS alone?
    # =====================================================================
    print("\n" + "=" * 70)
    print("STEP 6: RF classification using ONLY HS fraction features")
    print("=" * 70)

    # Use only HS-related features
    hs_feature_cols = [c for c in feat_cols if any(c.startswith(p) for p in
                       ["HS_", "spec_HS", "E2_E3", "A254", "A210", "A250",
                        "A280", "A272", "A220", "A230", "A300", "S275", "S350",
                        "S206", "SR", "shape_", "elution_", "HS_early"])]

    X_hs_feats = df[hs_feature_cols].values
    X_hs_feats = np.nan_to_num(X_hs_feats, nan=0, posinf=0, neginf=0)

    # Filter to types with >=5 samples
    valid_mask = np.array([np.sum(types == t) >= 5 for t in types])
    X_rf = X_hs_feats[valid_mask]
    y_rf = types[valid_mask]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_rf)

    rf = RandomForestClassifier(n_estimators=500, class_weight='balanced',
                                 random_state=42, n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(rf, X_rf, y_encoded, cv=cv, scoring='accuracy')
    print(f"\nHS-only RF accuracy (CV): {scores.mean():.3f} ± {scores.std():.3f}")
    print(f"  (Compare with full-spectrum RF: 0.905)")

    # Fit final model for importance
    rf.fit(X_rf, y_encoded)
    importances = pd.Series(rf.feature_importances_, index=hs_feature_cols)
    importances = importances.sort_values(ascending=False)

    print(f"\nTop 20 HS features for water type classification:")
    for feat, imp in importances.head(20).items():
        print(f"  {feat:40s}: {imp:.4f}")

    # Confusion matrix
    from sklearn.model_selection import cross_val_predict
    y_pred = cross_val_predict(rf, X_rf, y_encoded, cv=cv)

    fig = plt.figure(figsize=(18, 8))
    gs = GridSpec(1, 2, figure=fig, wspace=0.3)

    # 6a. Feature importance
    ax1 = fig.add_subplot(gs[0, 0])
    top20 = importances.head(20)
    colors_imp = []
    for f in top20.index:
        if 'elution' in f or 'shape' in f:
            colors_imp.append('#E74C3C')  # Shape features
        elif any(f.startswith(p) for p in ['E2', 'A2', 'A3', 'S2', 'S3', 'SR']):
            colors_imp.append('#3498DB')  # Ratio features
        else:
            colors_imp.append('#2ECC71')  # Spectral features
    ax1.barh(range(len(top20)), top20.values, color=colors_imp)
    ax1.set_yticks(range(len(top20)))
    ax1.set_yticklabels(top20.index, fontsize=8)
    ax1.invert_yaxis()
    ax1.set_xlabel("Importance")
    ax1.set_title(f"HS-only RF: Top 20 Features\n(CV accuracy: {scores.mean():.3f})")
    # Legend
    from matplotlib.patches import Patch
    ax1.legend(handles=[
        Patch(color='#E74C3C', label='Shape/Elution'),
        Patch(color='#3498DB', label='Spectral Ratios'),
        Patch(color='#2ECC71', label='Spectral Abs.')
    ], fontsize=8, loc='lower right')

    # 6b. Confusion matrix
    ax2 = fig.add_subplot(gs[0, 1])
    labels = le.classes_
    cm = confusion_matrix(y_encoded, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    im = ax2.imshow(cm_norm, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax2.set_yticks(range(len(labels)))
    ax2.set_yticklabels(labels, fontsize=7)
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("True")
    ax2.set_title("Confusion Matrix (HS features only)")
    plt.colorbar(im, ax=ax2, shrink=0.8)

    plt.savefig(os.path.join(OUTPUT_DIR, "04_RF_HS_only.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 04_RF_HS_only.png")

    # =====================================================================
    # STEP 7: Humic quality indices — practical classification tool
    # =====================================================================
    print("\n" + "=" * 70)
    print("STEP 7: Practical humic quality indices")
    print("=" * 70)

    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

    # 7a. E2/E3 vs S275-295 (two key DOM quality indices)
    ax1 = fig.add_subplot(gs[0, 0])
    for t in all_types:
        mask = (types == t) & np.isfinite(df['E2_E3'].values) & np.isfinite(df['S275_295'].values)
        if np.sum(mask) < 2:
            continue
        marker = '*' if t in ref_types else 'o'
        size = 120 if t in ref_types else 15
        ax1.scatter(df.loc[mask, 'E2_E3'], df.loc[mask, 'S275_295'],
                   s=size, alpha=0.5, marker=marker, color=type_colors[t],
                   label=f"{t}" if np.sum(mask) >= 5 or t in ref_types else None)
    ax1.set_xlabel("E2/E3 (A250/A365)")
    ax1.set_ylabel("S275-295 (spectral slope)")
    ax1.set_title("DOM Quality Space\n(↑E2/E3 = lower MW, ↑S275 = less aromatic)")
    ax1.legend(fontsize=6, ncol=2)
    ax1.grid(True, alpha=0.3)

    # 7b. SR (slope ratio) by type
    ax2 = fig.add_subplot(gs[0, 1])
    sr_data = {}
    for t in water_types + ref_types:
        mask = (types == t) & np.isfinite(df['SR'].values)
        if np.sum(mask) >= 2:
            sr_data[t] = df.loc[mask, 'SR'].values
    if sr_data:
        sorted_t = sorted(sr_data.keys(), key=lambda t: np.median(sr_data[t]))
        bp = ax2.boxplot([sr_data[t] for t in sorted_t], vert=True,
                        widths=0.6, patch_artist=True)
        for i, (patch, t) in enumerate(zip(bp['boxes'], sorted_t)):
            if t in ref_types:
                patch.set_facecolor(colors_ref.get(t, '#AED6F1'))
                patch.set_alpha(0.8)
            else:
                patch.set_facecolor('#AED6F1')
        ax2.set_xticklabels(sorted_t, rotation=45, ha='right', fontsize=7)
        ax2.axhline(1.0, color='red', linestyle='--', alpha=0.5, label="SR=1 (marine/fresh boundary)")
        ax2.set_ylabel("Slope Ratio (SR)")
        ax2.set_title("Spectral Slope Ratio\n(SR<1: terrestrial, SR>1: autochthonous)")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3, axis='y')

    # 7c. A254/A280 vs E2/E3 — aromaticity vs MW
    ax3 = fig.add_subplot(gs[0, 2])
    for t in all_types:
        mask = (types == t) & np.isfinite(df['A254_A280'].values) & np.isfinite(df['E2_E3'].values)
        if np.sum(mask) < 2:
            continue
        marker = '*' if t in ref_types else 'o'
        size = 120 if t in ref_types else 15
        ax3.scatter(df.loc[mask, 'A254_A280'], df.loc[mask, 'E2_E3'],
                   s=size, alpha=0.5, marker=marker, color=type_colors[t],
                   label=f"{t}" if np.sum(mask) >= 5 or t in ref_types else None)
    ax3.set_xlabel("A254/A280 (aromaticity)")
    ax3.set_ylabel("E2/E3 (MW proxy)")
    ax3.set_title("Aromaticity vs Molecular Weight")
    ax3.legend(fontsize=6, ncol=2)
    ax3.grid(True, alpha=0.3)

    # 7d. HS early/late ratio at 254nm by type (MW distribution within HS)
    ax4 = fig.add_subplot(gs[1, 0])
    el_col = "HS_early_late_254"
    if el_col in df.columns:
        el_data = {}
        for t in water_types + ref_types:
            mask = (types == t) & np.isfinite(df[el_col].values)
            if np.sum(mask) >= 2:
                el_data[t] = df.loc[mask, el_col].values
        if el_data:
            sorted_t = sorted(el_data.keys(), key=lambda t: np.median(el_data[t]))
            bp = ax4.boxplot([el_data[t] for t in sorted_t], vert=True,
                            widths=0.6, patch_artist=True)
            for patch, t in zip(bp['boxes'], sorted_t):
                if t in ref_types:
                    patch.set_facecolor(colors_ref.get(t, '#AED6F1'))
                else:
                    patch.set_facecolor('#AED6F1')
            ax4.set_xticklabels(sorted_t, rotation=45, ha='right', fontsize=7)
            ax4.set_ylabel("Early/Late HS ratio @ 254nm")
            ax4.set_title("MW Distribution within HS\n(>1: more high-MW, <1: more low-MW)")
            ax4.grid(True, alpha=0.3, axis='y')

    # 7e. Shape: UV area / Vis area by type
    ax5 = fig.add_subplot(gs[1, 1])
    uv_vis_col = "shape_UV_Vis_ratio"
    if uv_vis_col in df.columns:
        uv_data = {}
        for t in water_types + ref_types:
            mask = (types == t) & np.isfinite(df[uv_vis_col].values)
            if np.sum(mask) >= 2:
                uv_data[t] = df.loc[mask, uv_vis_col].values
        if uv_data:
            sorted_t = sorted(uv_data.keys(), key=lambda t: np.median(uv_data[t]))
            bp = ax5.boxplot([uv_data[t] for t in sorted_t], vert=True,
                            widths=0.6, patch_artist=True)
            for patch, t in zip(bp['boxes'], sorted_t):
                if t in ref_types:
                    patch.set_facecolor(colors_ref.get(t, '#AED6F1'))
                else:
                    patch.set_facecolor('#AED6F1')
            ax5.set_xticklabels(sorted_t, rotation=45, ha='right', fontsize=7)
            ax5.set_ylabel("UV area / Vis area (norm.)")
            ax5.set_title("UV vs Visible Absorption\n(↑ = protein/nitrate-like, ↓ = chromophoric)")
            ax5.grid(True, alpha=0.3, axis='y')

    # 7f. Multi-index summary — heatmap of median indices per type
    ax6 = fig.add_subplot(gs[1, 2])
    index_cols = ["E2_E3", "A254_A280", "A210_A254", "S275_295", "SR",
                  "shape_UV_Vis_ratio", "HS_early_late_254"]
    index_labels = ["E2/E3", "A254/A280", "A210/A254", "S275-295", "SR",
                    "UV/Vis", "Early/Late"]
    available_idx = [(c, l) for c, l in zip(index_cols, index_labels) if c in df.columns]

    summary_types = [t for t in water_types + ref_types if np.sum(types == t) >= 3]
    if available_idx and summary_types:
        heatmap_data = np.zeros((len(summary_types), len(available_idx)))
        for i, t in enumerate(summary_types):
            mask = types == t
            for j, (col, _) in enumerate(available_idx):
                vals = df.loc[mask, col].dropna().values
                heatmap_data[i, j] = np.median(vals) if len(vals) > 0 else np.nan

        # Normalize each column to 0-1
        for j in range(heatmap_data.shape[1]):
            col_vals = heatmap_data[:, j]
            valid = np.isfinite(col_vals)
            if np.sum(valid) > 0:
                vmin, vmax = np.nanmin(col_vals), np.nanmax(col_vals)
                if vmax > vmin:
                    heatmap_data[:, j] = (col_vals - vmin) / (vmax - vmin)

        im = ax6.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
        ax6.set_xticks(range(len(available_idx)))
        ax6.set_xticklabels([l for _, l in available_idx], rotation=45, ha='right', fontsize=8)
        ax6.set_yticks(range(len(summary_types)))
        ax6.set_yticklabels(summary_types, fontsize=8)
        ax6.set_title("Humic Quality Index Summary\n(normalized 0-1 per index)")
        plt.colorbar(im, ax=ax6, shrink=0.8)

    # 7g. Clustering in index space
    ax7 = fig.add_subplot(gs[2, 0:2])
    cluster_cols = [c for c, _ in available_idx]
    X_cluster = df[cluster_cols].values
    X_cluster = np.nan_to_num(X_cluster, nan=0, posinf=0, neginf=0)
    scaler_c = StandardScaler()
    X_cluster_scaled = scaler_c.fit_transform(X_cluster)

    # K-means with 3-6 clusters, pick best silhouette
    best_k = 3
    best_sil = -1
    for k in range(3, 7):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_cluster_scaled)
        sil = silhouette_score(X_cluster_scaled, labels)
        if sil > best_sil:
            best_sil = sil
            best_k = k

    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = km.fit_predict(X_cluster_scaled)

    pca_cluster = PCA(n_components=2)
    scores_cluster = pca_cluster.fit_transform(X_cluster_scaled)

    scatter = ax7.scatter(scores_cluster[:, 0], scores_cluster[:, 1],
                         c=cluster_labels, cmap='Set2', s=15, alpha=0.6)
    # Mark references
    for t, marker_t, color_t in [("HA", '*', 'red'), ("FA", '*', 'blue'),
                                  ("SRHA", 'P', 'red'), ("SRFA", 'P', 'blue')]:
        mask = types == t
        if np.sum(mask) > 0:
            ax7.scatter(scores_cluster[mask, 0], scores_cluster[mask, 1],
                       s=200, marker=marker_t, color=color_t, edgecolors='black',
                       zorder=10, label=t)
    ax7.set_xlabel("Index PC1")
    ax7.set_ylabel("Index PC2")
    ax7.set_title(f"K-Means Clustering on Humic Indices (k={best_k}, silhouette={best_sil:.3f})")
    ax7.legend(fontsize=8)
    ax7.grid(True, alpha=0.3)

    # Print cluster composition
    print(f"\nK-Means clustering (k={best_k}, silhouette={best_sil:.3f}):")
    for k in range(best_k):
        mask = cluster_labels == k
        cluster_types = Counter(types[mask])
        print(f"  Cluster {k}: {np.sum(mask)} samples")
        for t, n in cluster_types.most_common(5):
            print(f"    {t}: {n}")

    # 7h. Summary table
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    summary_text = f"HUMIC CHARACTERIZATION SUMMARY\n{'='*35}\n\n"
    summary_text += f"Total samples: {len(df)}\n"
    summary_text += f"HS features: {len(hs_feature_cols)}\n"
    summary_text += f"HS-only RF accuracy: {scores.mean():.3f}\n"
    summary_text += f"K-Means clusters: {best_k} (sil={best_sil:.3f})\n\n"

    # HA vs FA discrimination
    if np.sum(ha_mask) > 0 and np.sum(fa_mask) > 0:
        ha_e2e3 = df.loc[ha_mask, 'E2_E3'].median()
        fa_e2e3 = df.loc[fa_mask, 'E2_E3'].median()
        ha_sr = df.loc[ha_mask, 'SR'].dropna().median() if 'SR' in df.columns else np.nan
        fa_sr = df.loc[fa_mask, 'SR'].dropna().median() if 'SR' in df.columns else np.nan
        summary_text += f"HA median E2/E3: {ha_e2e3:.2f}\n"
        summary_text += f"FA median E2/E3: {fa_e2e3:.2f}\n"
        summary_text += f"HA median SR: {ha_sr:.3f}\n"
        summary_text += f"FA median SR: {fa_sr:.3f}\n\n"

    summary_text += "Key findings:\n"
    summary_text += "- HA: lower E2/E3, higher MW\n"
    summary_text += "- FA: higher E2/E3, lower MW\n"
    summary_text += "- SR<1: terrestrial DOM\n"
    summary_text += "- SR>1: autochthonous/processed\n"

    ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.savefig(os.path.join(OUTPUT_DIR, "05_humic_quality_indices.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 05_humic_quality_indices.png")

    # =====================================================================
    # STEP 8: Per-plant detailed comparison
    # =====================================================================
    print("\n" + "=" * 70)
    print("STEP 8: Per-plant humic fingerprint comparison")
    print("=" * 70)

    plant_types = ["FR", "PTLL", "PTT", "SK", "ATL_2024", "EX", "LQ", "EXT"]
    plant_types = [t for t in plant_types if np.sum(types == t) >= 5]

    fig = plt.figure(figsize=(20, 5 * ((len(plant_types) + 2) // 3)))
    n_rows = (len(plant_types) + 2) // 3
    gs = GridSpec(n_rows, 3, figure=fig, hspace=0.4, wspace=0.3)

    for idx, plant in enumerate(plant_types):
        row, col = idx // 3, idx % 3
        ax = fig.add_subplot(gs[row, col])

        mask = types == plant
        specs = df.loc[mask, hs_cols].values
        mean_spec = np.mean(specs, axis=0)
        std_spec = np.std(specs, axis=0)

        # Normalize at 254
        idx254 = np.argmin(np.abs(hs_wl - 254))
        if mean_spec[idx254] > 0.01:
            norm_mean = mean_spec / mean_spec[idx254]
            norm_std = std_spec / mean_spec[idx254]
        else:
            norm_mean = mean_spec
            norm_std = std_spec

        ax.plot(hs_wl, norm_mean, 'b-', linewidth=2, label=f"{plant} (n={np.sum(mask)})")
        ax.fill_between(hs_wl, norm_mean - norm_std, norm_mean + norm_std,
                        alpha=0.2, color='blue')

        # Add HA/FA templates for comparison
        ha_spec_n = ha_template / ha_template[idx254] if ha_template[idx254] > 0.01 else ha_template
        fa_spec_n = fa_template / fa_template[idx254] if fa_template[idx254] > 0.01 else fa_template
        ax.plot(hs_wl, ha_spec_n, 'r--', linewidth=1, alpha=0.7, label="HA ref")
        ax.plot(hs_wl, fa_spec_n, 'b--', linewidth=1, alpha=0.7, label="FA ref")

        # HA/FA similarity
        cos_ha = np.mean([1 - cosine(s, ha_template) for s in specs if np.any(s > 0)])
        cos_fa = np.mean([1 - cosine(s, fa_template) for s in specs if np.any(s > 0)])

        ax.set_title(f"{plant}\ncos(HA)={cos_ha:.4f}, cos(FA)={cos_fa:.4f}")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Norm. Abs. (A/A254)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.savefig(os.path.join(OUTPUT_DIR, "06_plant_fingerprints.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 06_plant_fingerprints.png")

    # =====================================================================
    # FINAL SUMMARY
    # =====================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nTotal samples: {len(df)}")
    print(f"HS spectral features: {len(hs_cols)}")
    print(f"Total humic features: {len(hs_feature_cols)}")
    print(f"HS-only RF accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
    print(f"Full-spectrum RF accuracy: 0.905 (for comparison)")
    print(f"K-Means clusters: {best_k} (silhouette={best_sil:.3f})")
    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print("=" * 70)
    print("DONE")


if __name__ == "__main__":
    main()
