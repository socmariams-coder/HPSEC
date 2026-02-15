# -*- coding: utf-8 -*-
"""
RF Spectral Exploration — HPSEC-DAD Export3D
=============================================
Exploració a fons de les dades Export3D amb Random Forest i PCA.
Sense targets externs — només dades DAD + DOC existents.

Autor: HPSEC Suite
"""

import os
import sys
import re
import glob
import json
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

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, classification_report

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_FOLDERS = [
    "C:/Users/Lequia/Desktop/Dades3",  # Primary: Dades3 (includes reorganized Meritxell)
    "C:/Users/Lequia/OneDrive - Universitat de Girona/UdG365_HPLC-DAD - General/Dades",  # OneDrive
]
OUTPUT_DIR = "C:/Users/Lequia/Desktop/HPSEC/rf_results_v3"

FRACTIONS = {
    "BioP": (10.8, 18.0),
    "HS":   (18.0, 23.0),
    "BB":   (23.0, 26.0),
    "SB":   (26.0, 32.0),
    "LMW":  (32.0, 60.0),
}

# Wavelengths to use for feature extraction (all available in Export3D)
# Export3D has 200-400nm in 2nm steps = 101 wavelengths
WL_ALL = list(range(200, 402, 2))

# Key wavelengths for detailed analysis
WL_KEY = [200, 206, 210, 220, 230, 240, 250, 254, 260, 272, 280, 290, 300, 330, 350, 362, 400]

# Sample type classification patterns (order matters — first match wins)
# Patterns use ^ prefix matching or lookahead to avoid false positives
TYPE_PATTERNS = [
    # --- Calibrants & blanks (EXCLUDE) ---
    ("KHP",      r"(?i)^KHP"),
    ("BLANK",    r"(?i)^(MQ|BLANK|BLK|H2O)"),
    ("NaOH",     r"(?i)^NaOH"),                  # Reagent blank
    ("Buffer",   r"(?i)^Buffer"),

    # --- IHSS standards ---
    ("SRHA",     r"(?i)^(SRHA|3S101H)"),          # Suwannee River Humic Acid
    ("SRFA",     r"(?i)^(SRFA|3S101F)"),          # Suwannee River Fulvic Acid
    ("HA",       r"(?i)^HA[_\.\s\-]"),            # HA_1, HA_R1
    ("FA",       r"(?i)^FA[_\.\s\-]"),            # FA_1, FA_R1
    ("MIX",      r"(?i)^MIX[_\.\s\-]"),           # HA+FA mix

    # --- Inorganic standards ---
    ("NO3",      r"(?i)^(NO3|NaNO3)"),
    ("Br",       r"(?i)^(Br[_\.\s\d]|KBr)"),     # Br_1, Br05, KBr
    ("NH4",      r"(?i)^(NH4|AMONI)"),
    ("Na2CO3",   r"(?i)^Na2CO3"),
    ("CaCO3",    r"(?i)^CaCO3"),
    ("Na2SO4",   r"(?i)^Na2SO4"),
    ("CLORAT",   r"(?i)^clorat"),
    ("CLORIT",   r"(?i)^clorit"),
    ("SIXSO4",   r"(?i)^SIXSO4"),                 # SI + SO4 mix
    ("BSA",      r"(?i)^BSA"),

    # --- Water samples: treatment stages ---
    ("POST_O3",  r"(?i)^POST[\s_]?O[3z]"),       # POST_O3, PostO3, POST O3
    ("SFiltres", r"(?i)^SFiltre"),                # Sortida Filtres
    ("EP_PTL",   r"(?i)^EP[_\s]PTL"),             # EP specific plant
    ("EP_PTT",   r"(?i)^EP[_\s]PTT"),
    ("EP_PTLL",  r"(?i)^EP[_\s]PTLL"),
    ("EP",       r"(?i)^EP[_\.\s\-]"),            # EP_1, EP_R1
    ("SP",       r"(?i)^SP[_\.\s\-]"),

    # --- Water samples: plants (Meritxell 2024 ATL) ---
    ("PTLL",     r"(?i)^PTLL[\-_]"),              # PTLL-02-01, PTLL-04-01
    ("PTT",      r"(?i)^PTT[\-_]"),               # PTT-02-01, PTT-04-01

    # --- Water samples: monitoring & experimental ---
    ("FR",       r"(?i)^FR\d"),                   # FR2326, FR2512...
    ("EX",       r"(?i)^EX\d"),                   # EX1-L, EX10-L...
    ("EXT",      r"(?i)^EXT\d"),                  # EXT1, EXT10 (Meritxell juny 2024)
    ("LQ",       r"(?i)^LQ\d"),                   # LQ0149, LQ0303...
    ("SK",       r"(?i)^SK\d"),                   # SK077, SK095... (mostreig 2023)
    ("FS",       r"(?i)^FS\d"),                   # FS1, FS2 (FISERSA?)

    # --- Dated samples (2024-MM-DD format from ATL-GENER) ---
    ("ATL_2024", r"(?i)^\d{4}-\d{2}-\d{2}"),     # 2024-01-05, 2024-02-10...

    # --- Lab samples (2025-L, 2025-T) ---
    ("PTLL_2025",r"(?i)^\d{4}-PTLL-"),            # 2025-PTLL-03-06
    ("LAB",      r"(?i)^\d{4}-L-"),               # 2025-L-03-01
    ("LAB_T",    r"(?i)^\d{4}-T-"),               # 2025-T-03-01

    # --- Repetibilitat (003-P1-A2-EXT-ATL format) ---
    ("REPE",     r"(?i)^\d{3}-P\d-A\d"),          # 003-P1-A2-EXT-ATL-25
]

# Types to INCLUDE in the main RF analysis (real water samples)
WATER_TYPES = {
    "EP", "SP", "POST_O3", "SFiltres",           # Treatment stages
    "PTLL", "PTT", "PTLL_2025",                   # Plants (Meritxell)
    "EP_PTL", "EP_PTT", "EP_PTLL",               # EP per plant
    "FR", "EX", "EXT", "LQ",                     # Monitoring/experimental
    "SK", "FS",                                    # Mostreig 2023
    "ATL_2024",                                    # Dated ATL samples
    "LAB", "LAB_T",                                # Lab 2025
    "REPE",                                        # Repetibilitat
}

# Types to EXCLUDE completely
EXCLUDE_TYPES = {"KHP", "BLANK", "NaOH", "Buffer"}

# Types for a separate standards analysis
STANDARD_TYPES = {"HA", "FA", "SRHA", "SRFA", "MIX", "NO3", "Br", "NH4",
                  "Na2CO3", "CaCO3", "Na2SO4", "CLORAT", "CLORIT", "SIXSO4", "BSA"}


def classify_sample(filename):
    """Classify sample type from filename."""
    base = os.path.splitext(os.path.basename(filename))[0]
    for stype, pattern in TYPE_PATTERNS:
        if re.search(pattern, base):
            return stype
    return "OTHER"


def load_export3d(filepath):
    """Load a single Export3D file. Returns (time_array, wavelengths, data_matrix) or None."""
    for enc in ["utf-16", "utf-8", "latin1"]:
        try:
            df = pd.read_csv(filepath, sep=",", encoding=enc, engine="python")
            if df.shape[1] < 10:
                continue

            cols = list(df.columns)
            time_col = cols[0]
            t = pd.to_numeric(df[time_col], errors="coerce").values

            wavelengths = []
            data_cols = []
            for c in cols[1:]:
                try:
                    wl = float(str(c).strip())
                    wavelengths.append(int(wl) if wl == int(wl) else wl)
                    data_cols.append(c)
                except ValueError:
                    continue

            if len(wavelengths) < 10:
                continue

            data = df[data_cols].apply(pd.to_numeric, errors="coerce").values
            valid = np.isfinite(t)

            return t[valid], np.array(wavelengths), data[valid]
        except Exception:
            continue
    return None


def extract_features(t, wavelengths, data, fractions=FRACTIONS):
    """
    Extract features from a single Export3D matrix.

    Returns dict of features:
    - area_{frac}_{wl}: integrated area per fraction per wavelength
    - ratio_{frac}_{wl1}_{wl2}: spectral ratios per fraction
    - slope_{frac}_{wl1}_{wl2}: spectral slopes per fraction
    - max_{frac}_{wl}: max signal per fraction per wavelength
    """
    features = {}
    wl_list = list(wavelengths)

    for frac_name, (t_start, t_end) in fractions.items():
        mask = (t >= t_start) & (t <= t_end)
        if np.sum(mask) < 3:
            continue

        t_frac = t[mask]
        data_frac = data[mask]

        # Integrated area per wavelength (trapezoid)
        for i, wl in enumerate(wl_list):
            if i < data_frac.shape[1]:
                y = data_frac[:, i]
                valid = np.isfinite(y)
                if np.sum(valid) > 2:
                    area = np.trapezoid(y[valid], t_frac[valid])
                    features[f"area_{frac_name}_{wl}"] = area
                    features[f"max_{frac_name}_{wl}"] = np.nanmax(y)
                else:
                    features[f"area_{frac_name}_{wl}"] = 0.0
                    features[f"max_{frac_name}_{wl}"] = 0.0

        # Key spectral ratios per fraction
        for wl1, wl2 in [(200, 254), (210, 254), (220, 254), (280, 254), (254, 362)]:
            k1 = f"area_{frac_name}_{wl1}"
            k2 = f"area_{frac_name}_{wl2}"
            if k1 in features and k2 in features and features[k2] > 0.1:
                features[f"ratio_{frac_name}_{wl1}_{wl2}"] = features[k1] / features[k2]

        # Spectral slopes (S206-240, S275-295, S350-380 — from thesis)
        for wl_a, wl_b in [(206, 240), (275, 295), (350, 380)]:
            ka = f"area_{frac_name}_{wl_a}"
            kb = f"area_{frac_name}_{wl_b}"
            # Find closest available wavelengths
            idx_a = min(range(len(wl_list)), key=lambda i: abs(wl_list[i] - wl_a)) if wl_list else None
            idx_b = min(range(len(wl_list)), key=lambda i: abs(wl_list[i] - wl_b)) if wl_list else None
            if idx_a is not None and idx_b is not None:
                ka_key = f"area_{frac_name}_{wl_list[idx_a]}"
                kb_key = f"area_{frac_name}_{wl_list[idx_b]}"
                if ka_key in features and kb_key in features:
                    denom = abs(wl_list[idx_a] - wl_list[idx_b])
                    if denom > 0 and features[kb_key] != 0:
                        features[f"slope_{frac_name}_{wl_a}_{wl_b}"] = (
                            (features[ka_key] - features[kb_key]) / denom
                        )

    # Cross-fraction ratios (same wavelength, different fractions)
    for wl in [210, 254, 280]:
        for f1, f2 in [("BioP", "HS"), ("HS", "BB"), ("BB", "LMW"), ("BioP", "LMW")]:
            k1 = f"area_{f1}_{wl}"
            k2 = f"area_{f2}_{wl}"
            if k1 in features and k2 in features and abs(features[k2]) > 0.1:
                features[f"xratio_{f1}_{f2}_{wl}"] = features[k1] / features[k2]

    return features


def scan_all_export3d(data_folders):
    """Scan all SEQ folders across multiple data directories for Export3D files.
    Deduplicates by filename+seq to avoid counting same file from multiple sources."""
    samples = []
    seen = set()  # (seq_name, filename) to deduplicate

    total_dirs = 0
    for data_folder in data_folders:
        if not os.path.isdir(data_folder):
            continue

        # Scan both *_SEQ and *_SEQ_BP patterns
        seq_dirs = sorted(glob.glob(os.path.join(data_folder, "*_SEQ")))
        seq_dirs += sorted(glob.glob(os.path.join(data_folder, "*_SEQ_BP")))

        for seq_dir in seq_dirs:
            seq_name = os.path.basename(seq_dir)

            # Try both Export3d and Export3D
            export_dir = None
            for subdir in ["Export3d", "Export3D"]:
                candidate = os.path.join(seq_dir, subdir)
                if os.path.isdir(candidate):
                    export_dir = candidate
                    break

            if export_dir is None:
                continue

            total_dirs += 1
            csv_files = glob.glob(os.path.join(export_dir, "*.csv"))
            csv_files += glob.glob(os.path.join(export_dir, "*.CSV"))
            csv_files = sorted(set(csv_files))

            # Skip UIB files
            csv_files = [f for f in csv_files if "UIB" not in os.path.basename(f).upper()]

            for fpath in csv_files:
                fname = os.path.basename(fpath)
                key = (seq_name, fname.upper())
                if key in seen:
                    continue
                seen.add(key)

                stype = classify_sample(fpath)
                samples.append({
                    "path": fpath,
                    "filename": fname,
                    "seq": seq_name,
                    "type": stype,
                })

    print(f"Scanned {total_dirs} SEQ directories across {len(data_folders)} locations")
    return samples


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # =========================================================================
    # STEP 1: Scan and classify all samples
    # =========================================================================
    print("=" * 70)
    print("STEP 1: Scanning Export3D files...")
    print("=" * 70)

    samples = scan_all_export3d(DATA_FOLDERS)
    print(f"\nTotal files found: {len(samples)}")

    type_counts = Counter(s["type"] for s in samples)
    print("\nSample type distribution:")
    for stype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {stype:12s}: {count:4d}")

    # Filter: exclude KHP and BLANK
    samples = [s for s in samples if s["type"] not in EXCLUDE_TYPES]
    print(f"\nAfter excluding KHP/BLANK: {len(samples)} files")

    # =========================================================================
    # STEP 2: Extract features from all files
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Extracting spectral features...")
    print("=" * 70)

    all_features = []
    all_meta = []
    errors = 0

    for i, sample in enumerate(samples):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Processing {i+1}/{len(samples)}: {sample['filename'][:40]}...")

        result = load_export3d(sample["path"])
        if result is None:
            errors += 1
            continue

        t, wavelengths, data = result

        # Verify we have enough wavelengths
        if len(wavelengths) < 50:
            errors += 1
            continue

        feats = extract_features(t, wavelengths, data)
        if len(feats) < 100:  # Should have ~500+ features
            errors += 1
            continue

        all_features.append(feats)
        all_meta.append(sample)

    print(f"\nSuccessfully processed: {len(all_features)} files")
    print(f"Errors/skipped: {errors}")

    if len(all_features) < 10:
        print("ERROR: Too few samples loaded. Check data paths.")
        return

    # Build feature DataFrame
    df_feat = pd.DataFrame(all_features)
    df_feat = df_feat.fillna(0)

    df_meta = pd.DataFrame(all_meta)

    # Add type column to feature df
    df_feat["type"] = df_meta["type"].values
    df_feat["seq"] = df_meta["seq"].values
    df_feat["filename"] = df_meta["filename"].values

    print(f"\nFeature matrix shape: {df_feat.shape[0]} samples × {df_feat.shape[1] - 3} features")

    # Save feature matrix
    df_feat.to_csv(os.path.join(OUTPUT_DIR, "feature_matrix.csv"), index=False)
    print(f"Feature matrix saved to {OUTPUT_DIR}/feature_matrix.csv")

    # =========================================================================
    # STEP 3: Type distribution after feature extraction
    # =========================================================================
    type_counts_final = Counter(df_feat["type"])
    print("\nFinal type distribution:")
    for stype, count in sorted(type_counts_final.items(), key=lambda x: -x[1]):
        print(f"  {stype:12s}: {count:4d}")

    # =========================================================================
    # STEP 4: PCA — Unsupervised exploration
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: PCA — Unsupervised exploration")
    print("=" * 70)

    # Only use area features for PCA (consistent across samples)
    area_cols = [c for c in df_feat.columns if c.startswith("area_")]
    X_area = df_feat[area_cols].values

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_area)

    # Replace any remaining NaN/inf with 0
    X_scaled = np.nan_to_num(X_scaled, nan=0, posinf=0, neginf=0)

    # PCA
    pca = PCA(n_components=min(20, X_scaled.shape[1], X_scaled.shape[0]))
    X_pca = pca.fit_transform(X_scaled)

    print(f"\nPCA explained variance:")
    cumvar = 0
    for i in range(min(10, len(pca.explained_variance_ratio_))):
        cumvar += pca.explained_variance_ratio_[i] * 100
        print(f"  PC{i+1}: {pca.explained_variance_ratio_[i]*100:.1f}% (cumulative: {cumvar:.1f}%)")

    # --- PCA Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Color map per type
    all_types = sorted(df_feat["type"].unique())
    cmap = plt.cm.get_cmap("tab20", len(all_types))
    type_colors = {t: cmap(i) for i, t in enumerate(all_types)}

    # PC1 vs PC2
    ax = axes[0]
    for stype in all_types:
        mask = df_feat["type"] == stype
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                  c=[type_colors[stype]], label=stype, alpha=0.6, s=20, edgecolors='none')
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("PCA: PC1 vs PC2")
    ax.legend(fontsize=7, ncol=2, loc="best")
    ax.grid(True, alpha=0.3)

    # PC2 vs PC3
    ax = axes[1]
    for stype in all_types:
        mask = df_feat["type"] == stype
        if pca.n_components_ > 2:
            ax.scatter(X_pca[mask, 1], X_pca[mask, 2],
                      c=[type_colors[stype]], label=stype, alpha=0.6, s=20, edgecolors='none')
    ax.set_xlabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_ylabel(f"PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)" if pca.n_components_ > 2 else "PC3")
    ax.set_title("PCA: PC2 vs PC3")
    ax.legend(fontsize=7, ncol=2, loc="best")
    ax.grid(True, alpha=0.3)

    # Scree plot
    ax = axes[2]
    n_show = min(15, len(pca.explained_variance_ratio_))
    ax.bar(range(1, n_show + 1), pca.explained_variance_ratio_[:n_show] * 100,
           color="steelblue", alpha=0.7)
    ax.plot(range(1, n_show + 1), np.cumsum(pca.explained_variance_ratio_[:n_show]) * 100,
            "r-o", markersize=4)
    ax.set_xlabel("Component")
    ax.set_ylabel("Variance (%)")
    ax.set_title("Scree Plot")
    ax.axhline(y=95, color="grey", linestyle="--", alpha=0.5, label="95% cumul.")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "01_pca_overview.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 01_pca_overview.png")

    # --- PCA Loadings: which wavelengths/fractions drive each PC ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for pc_idx, ax in enumerate(axes.flat):
        if pc_idx >= pca.n_components_:
            break
        loadings = pca.components_[pc_idx]

        # Reshape loadings into fraction × wavelength heatmap
        # area_cols are like "area_BioP_200", "area_BioP_202", ...
        frac_names = list(FRACTIONS.keys())
        wl_vals = []
        loading_matrix = []

        for frac in frac_names:
            row = []
            wl_row = []
            for wl in WL_ALL:
                col_name = f"area_{frac}_{wl}"
                if col_name in area_cols:
                    idx = area_cols.index(col_name)
                    row.append(loadings[idx])
                    if frac == frac_names[0]:
                        wl_row.append(wl)
                else:
                    row.append(0)
                    if frac == frac_names[0]:
                        wl_row.append(wl)
            loading_matrix.append(row)
            if wl_row:
                wl_vals = wl_row

        loading_matrix = np.array(loading_matrix)

        # Plot heatmap
        im = ax.imshow(loading_matrix, aspect="auto", cmap="RdBu_r",
                       vmin=-np.abs(loading_matrix).max(), vmax=np.abs(loading_matrix).max())
        ax.set_yticks(range(len(frac_names)))
        ax.set_yticklabels(frac_names)

        # X-axis: every 10th wavelength
        tick_positions = list(range(0, len(wl_vals), 10))
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([wl_vals[i] for i in tick_positions], fontsize=8, rotation=45)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_title(f"PC{pc_idx+1} loadings ({pca.explained_variance_ratio_[pc_idx]*100:.1f}%)")
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "02_pca_loadings.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 02_pca_loadings.png")

    # =========================================================================
    # STEP 5: Random Forest Classification
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: Random Forest Classification")
    print("=" * 70)

    # Filter to types with enough samples (>= 5)
    valid_types = [t for t, c in type_counts_final.items() if c >= 5 and t not in EXCLUDE_TYPES]
    mask_valid = df_feat["type"].isin(valid_types)

    print(f"\nTypes with >= 5 samples: {valid_types}")
    print(f"Samples for RF: {mask_valid.sum()}")

    # Prepare X and y
    feature_cols = [c for c in df_feat.columns if c.startswith(("area_", "ratio_", "slope_", "max_", "xratio_"))]
    X = df_feat.loc[mask_valid, feature_cols].values
    y = df_feat.loc[mask_valid, "type"].values

    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    print(f"Feature matrix: {X.shape[0]} samples × {X.shape[1]} features")

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_classes = len(le.classes_)
    print(f"Classes: {list(le.classes_)} ({n_classes} classes)")

    # Train RF with cross-validation
    rf = RandomForestClassifier(n_estimators=500, max_depth=None,
                                 min_samples_leaf=2, random_state=42,
                                 n_jobs=-1, class_weight="balanced")

    n_splits = min(5, min(Counter(y).values()))
    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(rf, X, y_enc, cv=cv, scoring="accuracy")
        print(f"\nCross-validation accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
        print(f"  Per fold: {[f'{s:.3f}' for s in scores]}")

    # Fit on all data for feature importance
    rf.fit(X, y_enc)

    # Feature importance
    importances = rf.feature_importances_
    feat_imp = sorted(zip(feature_cols, importances), key=lambda x: -x[1])

    print(f"\nTop 30 most important features:")
    for feat, imp in feat_imp[:30]:
        print(f"  {feat:40s}: {imp:.4f}")

    # Save full importance ranking
    df_imp = pd.DataFrame(feat_imp, columns=["feature", "importance"])
    df_imp.to_csv(os.path.join(OUTPUT_DIR, "feature_importance_full.csv"), index=False)

    # --- Feature Importance Heatmap: fraction × wavelength ---
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))

    # 5a. Area feature importance heatmap
    ax = axes[0]
    frac_names = list(FRACTIONS.keys())
    imp_matrix = np.zeros((len(frac_names), len(WL_ALL)))

    for feat, imp in feat_imp:
        if feat.startswith("area_"):
            parts = feat.split("_")  # area_BioP_200
            if len(parts) >= 3:
                frac = parts[1]
                try:
                    wl = int(parts[2])
                    if frac in frac_names and wl in WL_ALL:
                        fi = frac_names.index(frac)
                        wi = WL_ALL.index(wl)
                        imp_matrix[fi, wi] = imp
                except ValueError:
                    pass

    im = ax.imshow(imp_matrix, aspect="auto", cmap="YlOrRd")
    ax.set_yticks(range(len(frac_names)))
    ax.set_yticklabels(frac_names, fontsize=11, fontweight="bold")
    tick_positions = list(range(0, len(WL_ALL), 5))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([WL_ALL[i] for i in tick_positions], fontsize=8, rotation=45)
    ax.set_xlabel("Wavelength (nm)", fontsize=12)
    ax.set_title("RF Feature Importance: Area per Fraction × Wavelength", fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Importance")

    # Mark key wavelengths
    for wl in [210, 254, 280, 362]:
        if wl in WL_ALL:
            wi = WL_ALL.index(wl)
            ax.axvline(x=wi, color="white", linestyle="--", alpha=0.5, linewidth=0.5)
            ax.text(wi, -0.7, str(wl), ha="center", fontsize=7, color="black")

    # 5b. Aggregated importance by wavelength (summed over fractions)
    ax = axes[1]
    imp_by_wl = imp_matrix.sum(axis=0)  # Sum over fractions
    imp_by_frac = imp_matrix.sum(axis=1)  # Sum over wavelengths

    ax.bar(range(len(WL_ALL)), imp_by_wl, color="steelblue", alpha=0.7, width=1.0)
    ax.set_xticks(list(range(0, len(WL_ALL), 5)))
    ax.set_xticklabels([WL_ALL[i] for i in range(0, len(WL_ALL), 5)], fontsize=8, rotation=45)
    ax.set_xlabel("Wavelength (nm)", fontsize=12)
    ax.set_ylabel("Summed Importance", fontsize=12)
    ax.set_title("Total Feature Importance by Wavelength", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # Mark key wavelengths
    for wl in [210, 254, 280, 362]:
        if wl in WL_ALL:
            wi = WL_ALL.index(wl)
            ax.axvline(x=wi, color="red", linestyle="--", alpha=0.5)
            ax.text(wi, ax.get_ylim()[1] * 0.95, str(wl), ha="center", fontsize=9, color="red")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "03_rf_importance_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 03_rf_importance_heatmap.png")

    # --- Importance by fraction (bar chart) ---
    fig, ax = plt.subplots(figsize=(8, 5))
    colors_frac = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#3498db"]
    ax.barh(range(len(frac_names)), imp_by_frac, color=colors_frac)
    ax.set_yticks(range(len(frac_names)))
    ax.set_yticklabels(frac_names, fontsize=12, fontweight="bold")
    ax.set_xlabel("Summed Feature Importance", fontsize=12)
    ax.set_title("RF Feature Importance by SEC Fraction", fontsize=14, fontweight="bold")
    for i, v in enumerate(imp_by_frac):
        ax.text(v + 0.001, i, f"{v:.3f}", va="center", fontsize=10)
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "04_rf_importance_by_fraction.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 04_rf_importance_by_fraction.png")

    # =========================================================================
    # STEP 6: Confusion matrix and per-class analysis
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 6: Per-class analysis")
    print("=" * 70)

    y_pred = rf.predict(X)
    y_pred_labels = le.inverse_transform(y_pred)

    print("\nClassification report (on training data — see CV for unbiased estimate):")
    print(classification_report(y, y_pred_labels))

    # Confusion matrix
    cm = confusion_matrix(y, y_pred_labels, labels=le.classes_)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(le.classes_, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(n_classes))
    ax.set_yticklabels(le.classes_, fontsize=10)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("RF Confusion Matrix (training data)", fontsize=14, fontweight="bold")

    # Add text annotations
    for i in range(n_classes):
        for j in range(n_classes):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=10)

    plt.colorbar(im, shrink=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "05_confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 05_confusion_matrix.png")

    # =========================================================================
    # STEP 7: What makes each water type unique? (per-class feature importance)
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 7: What makes each type spectrally unique?")
    print("=" * 70)

    # For each class, train a one-vs-rest RF and get feature importance
    fig, axes = plt.subplots(2, (len(valid_types) + 1) // 2, figsize=(24, 10))
    axes = axes.flatten()

    for idx, stype in enumerate(sorted(valid_types)):
        if idx >= len(axes):
            break

        y_binary = (y == stype).astype(int)
        if y_binary.sum() < 3 or y_binary.sum() == len(y_binary):
            continue

        rf_ovr = RandomForestClassifier(n_estimators=200, max_depth=10,
                                         random_state=42, n_jobs=-1,
                                         class_weight="balanced")
        rf_ovr.fit(X, y_binary)

        imp_ovr = rf_ovr.feature_importances_

        # Build importance heatmap for this type
        imp_m = np.zeros((len(frac_names), len(WL_ALL)))
        for fi_idx, feat in enumerate(feature_cols):
            if feat.startswith("area_"):
                parts = feat.split("_")
                if len(parts) >= 3:
                    frac = parts[1]
                    try:
                        wl = int(parts[2])
                        if frac in frac_names and wl in WL_ALL:
                            imp_m[frac_names.index(frac), WL_ALL.index(wl)] = imp_ovr[fi_idx]
                    except ValueError:
                        pass

        ax = axes[idx]
        im = ax.imshow(imp_m, aspect="auto", cmap="YlOrRd")
        ax.set_yticks(range(len(frac_names)))
        ax.set_yticklabels(frac_names, fontsize=9)
        tick_pos = list(range(0, len(WL_ALL), 10))
        ax.set_xticks(tick_pos)
        ax.set_xticklabels([WL_ALL[i] for i in tick_pos], fontsize=7, rotation=45)
        n_samples = (y == stype).sum()
        ax.set_title(f"{stype} (n={n_samples})", fontsize=12, fontweight="bold")

        # Print top 5 features for this type
        top5 = sorted(zip(feature_cols, imp_ovr), key=lambda x: -x[1])[:5]
        print(f"\n{stype} — top 5 discriminating features:")
        for feat, imp in top5:
            print(f"  {feat:40s}: {imp:.4f}")

    # Hide unused axes
    for idx in range(len(valid_types), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Per-Type Feature Importance (One-vs-Rest RF)", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "06_per_type_importance.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("\nSaved: 06_per_type_importance.png")

    # =========================================================================
    # STEP 8: Spectral fingerprints per type — mean spectra per fraction
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 8: Mean spectral profiles per type and fraction")
    print("=" * 70)

    fig, axes = plt.subplots(len(frac_names), 1, figsize=(14, 4 * len(frac_names)))

    water_types_present = sorted([t for t in WATER_TYPES if t in valid_types])
    cmap_water = plt.cm.get_cmap("Set1", len(water_types_present))
    water_colors = {t: cmap_water(i) for i, t in enumerate(water_types_present)}

    for fi, frac in enumerate(frac_names):
        ax = axes[fi]

        for stype in water_types_present:
            mask = df_feat["type"] == stype
            if mask.sum() == 0:
                continue

            wl_values = []
            mean_areas = []
            std_areas = []

            for wl in WL_ALL:
                col = f"area_{frac}_{wl}"
                if col in df_feat.columns:
                    vals = df_feat.loc[mask, col].values
                    wl_values.append(wl)
                    mean_areas.append(np.mean(vals))
                    std_areas.append(np.std(vals))

            if wl_values:
                wl_arr = np.array(wl_values)
                mean_arr = np.array(mean_areas)
                std_arr = np.array(std_areas)

                ax.plot(wl_arr, mean_arr, color=water_colors[stype], label=stype, linewidth=1.5)
                ax.fill_between(wl_arr, mean_arr - std_arr, mean_arr + std_arr,
                              color=water_colors[stype], alpha=0.1)

        ax.set_title(f"Fraction: {frac} ({FRACTIONS[frac][0]}-{FRACTIONS[frac][1]} min)",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Integrated Area (mAU·min)")
        ax.legend(fontsize=9, ncol=len(water_types_present), loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(200, 400)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "07_spectral_fingerprints.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 07_spectral_fingerprints.png")

    # =========================================================================
    # STEP 9: Isolation Forest — Anomaly Detection
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 9: Isolation Forest — Anomaly Detection")
    print("=" * 70)

    # Only on water types
    mask_water = df_feat["type"].isin(WATER_TYPES)
    X_water = df_feat.loc[mask_water, feature_cols].values
    X_water = np.nan_to_num(X_water, nan=0, posinf=0, neginf=0)

    iso = IsolationForest(n_estimators=300, contamination=0.05, random_state=42, n_jobs=-1)
    anomaly_scores = iso.fit_predict(X_water)  # -1 for anomalies, 1 for normal
    anomaly_raw = iso.decision_function(X_water)  # Continuous score

    n_anomalies = (anomaly_scores == -1).sum()
    print(f"\nAnomalies detected: {n_anomalies} / {len(anomaly_scores)} ({n_anomalies/len(anomaly_scores)*100:.1f}%)")

    # List anomalies
    anomaly_indices = np.where(anomaly_scores == -1)[0]
    water_meta = df_meta[mask_water.values].reset_index(drop=True)

    print("\nTop anomalous samples:")
    anomaly_data = []
    for idx in anomaly_indices:
        anomaly_data.append({
            "filename": water_meta.iloc[idx]["filename"],
            "seq": water_meta.iloc[idx]["seq"],
            "type": water_meta.iloc[idx]["type"],
            "score": anomaly_raw[idx],
        })

    anomaly_data.sort(key=lambda x: x["score"])
    for ad in anomaly_data[:15]:
        print(f"  {ad['type']:6s} {ad['seq']:12s} {ad['filename']:40s} score={ad['score']:.3f}")

    # Plot anomalies on PCA
    X_water_scaled = scaler.transform(df_feat.loc[mask_water, area_cols].values)
    X_water_scaled = np.nan_to_num(X_water_scaled, nan=0, posinf=0, neginf=0)
    X_water_pca = pca.transform(X_water_scaled)

    fig, ax = plt.subplots(figsize=(10, 8))
    normal_mask = anomaly_scores == 1
    anomaly_mask = anomaly_scores == -1

    # Plot normal points colored by type
    water_types_arr = df_feat.loc[mask_water, "type"].values
    for stype in sorted(set(water_types_arr)):
        type_mask = (water_types_arr == stype) & normal_mask
        if type_mask.sum() > 0:
            ax.scatter(X_water_pca[type_mask, 0], X_water_pca[type_mask, 1],
                      c=[type_colors.get(stype, "grey")], label=stype, alpha=0.5, s=15, edgecolors='none')

    # Plot anomalies with red circles
    ax.scatter(X_water_pca[anomaly_mask, 0], X_water_pca[anomaly_mask, 1],
              c="none", edgecolors="red", linewidths=1.5, s=60, label="Anomaly", zorder=5)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("Isolation Forest Anomalies on PCA", fontsize=14, fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "08_anomaly_detection.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 08_anomaly_detection.png")

    # =========================================================================
    # STEP 10: Key spectral indices comparison
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 10: Key spectral indices per type")
    print("=" * 70)

    # Calculate key indices
    indices = {}
    for stype in water_types_present:
        mask = df_feat["type"] == stype
        if mask.sum() == 0:
            continue

        idx_data = {}
        for frac in frac_names:
            # A210/A254 ratio
            col_210 = f"area_{frac}_210"
            col_254 = f"area_{frac}_254"
            if col_210 in df_feat.columns and col_254 in df_feat.columns:
                a210 = df_feat.loc[mask, col_210].values
                a254 = df_feat.loc[mask, col_254].values
                valid = a254 > 0.1
                if valid.sum() > 0:
                    ratios = a210[valid] / a254[valid]
                    idx_data[f"{frac}_A210/A254"] = (np.mean(ratios), np.std(ratios))

            # A280/A254 ratio (protein proxy)
            col_280 = f"area_{frac}_280"
            if col_280 in df_feat.columns and col_254 in df_feat.columns:
                a280 = df_feat.loc[mask, col_280].values
                a254 = df_feat.loc[mask, col_254].values
                valid = a254 > 0.1
                if valid.sum() > 0:
                    ratios = a280[valid] / a254[valid]
                    idx_data[f"{frac}_A280/A254"] = (np.mean(ratios), np.std(ratios))

        indices[stype] = idx_data

    # Plot A210/A254 per fraction per type
    fig, axes = plt.subplots(1, len(frac_names), figsize=(4 * len(frac_names), 6))

    for fi, frac in enumerate(frac_names):
        ax = axes[fi]
        key = f"{frac}_A210/A254"

        types_with_data = [t for t in water_types_present if key in indices.get(t, {})]
        means = [indices[t][key][0] for t in types_with_data]
        stds = [indices[t][key][1] for t in types_with_data]
        colors = [water_colors[t] for t in types_with_data]

        bars = ax.bar(range(len(types_with_data)), means, yerr=stds,
                      color=colors, alpha=0.7, capsize=3, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(types_with_data)))
        ax.set_xticklabels(types_with_data, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("A210/A254")
        ax.set_title(f"{frac}", fontsize=13, fontweight="bold")
        ax.axhline(y=1.7, color="green", linestyle="--", alpha=0.7, label="HA/FA ref (1.7)")
        ax.grid(True, alpha=0.3, axis="y")
        if fi == 0:
            ax.legend(fontsize=8)

    fig.suptitle("Ratio A210/A254 per Fraction and Water Type\n(>1.7 suggests non-humic / N-rich / inorganic contribution)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "09_spectral_indices.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 09_spectral_indices.png")

    # =========================================================================
    # STEP 11: Temporal evolution — do samples change over SEQs?
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 11: Temporal evolution across sequences")
    print("=" * 70)

    # Extract SEQ number for ordering
    def extract_seq_num(seq_name):
        match = re.search(r"(\d+)", seq_name)
        return int(match.group(1)) if match else 0

    df_feat_water = df_feat[mask_water].copy()
    df_feat_water["seq_num"] = df_feat_water["seq"].apply(extract_seq_num)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # For each key wavelength, plot area_HS over time (grouped by type)
    key_metrics = [
        ("area_HS_254", "HS Area @ 254nm"),
        ("area_HS_210", "HS Area @ 210nm"),
        ("area_BB_254", "BB Area @ 254nm"),
        ("area_BioP_254", "BioP Area @ 254nm"),
        ("area_LMW_254", "LMW Area @ 254nm"),
    ]

    # Add ratio if available
    if "ratio_HS_210_254" in df_feat_water.columns:
        key_metrics.append(("ratio_HS_210_254", "HS Ratio A210/A254"))

    for mi, (metric, title) in enumerate(key_metrics):
        if mi >= len(axes.flat):
            break
        ax = axes.flat[mi]

        if metric not in df_feat_water.columns:
            ax.set_visible(False)
            continue

        for stype in water_types_present:
            mask_t = df_feat_water["type"] == stype
            if mask_t.sum() < 2:
                continue

            data_t = df_feat_water.loc[mask_t].sort_values("seq_num")
            ax.scatter(data_t["seq_num"], data_t[metric],
                      c=[water_colors[stype]], label=stype, alpha=0.5, s=15, edgecolors='none')

            # Moving average per type
            if len(data_t) > 5:
                grouped = data_t.groupby("seq_num")[metric].mean()
                ax.plot(grouped.index, grouped.values, color=water_colors[stype], alpha=0.7, linewidth=1)

        ax.set_xlabel("SEQ number")
        ax.set_ylabel(metric.split("_", 1)[1] if "_" in metric else metric)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
        if mi == 0:
            ax.legend(fontsize=7, ncol=2)

    # Hide unused
    for mi in range(len(key_metrics), len(axes.flat)):
        axes.flat[mi].set_visible(False)

    fig.suptitle("Temporal Evolution of Key Spectral Features Across Sequences",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "10_temporal_evolution.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 10_temporal_evolution.png")

    # =========================================================================
    # STEP 12: Summary statistics
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 12: Summary")
    print("=" * 70)

    print(f"\nTotal samples analyzed: {len(all_features)}")
    print(f"Feature dimensions: {len(feature_cols)}")

    if n_splits >= 2:
        print(f"RF classification accuracy (CV): {scores.mean():.3f} ± {scores.std():.3f}")

    print(f"\nPCA: {np.sum(pca.explained_variance_ratio_[:3]*100):.1f}% variance in first 3 PCs")

    print(f"\nMost important fraction (RF): {frac_names[np.argmax(imp_by_frac)]}")
    print(f"Most important wavelength region: {WL_ALL[np.argmax(imp_by_wl)]}-{WL_ALL[min(np.argmax(imp_by_wl)+2, len(WL_ALL)-1)]}nm")

    print(f"\nAnomalies: {n_anomalies} samples ({n_anomalies/len(anomaly_scores)*100:.1f}%)")

    print(f"\nAll results saved to: {OUTPUT_DIR}/")
    print("Files generated:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"  {f}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
