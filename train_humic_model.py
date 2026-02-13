# -*- coding: utf-8 -*-
"""
train_humic_model.py — Serialize PCA+LDA model to JSON
=======================================================
One-shot script that:
  1. Runs the rf_humic_lda.py pipeline (scan, extract, PCA, LDA)
  2. Saves the trained model to humic_lda_model.json (~35KB)
     so that hpsec_humic.py can predict HCI without sklearn.

Usage:
    python train_humic_model.py
"""

import os, sys, json, glob, re, warnings
import numpy as np
import pandas as pd
from scipy import stats

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneOut, cross_val_predict

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

# Import feature extraction from rf_humic_lda
from rf_humic_lda import (
    DATA_FOLDERS, WL_ALL, HS_WINDOW, HS_SUBWINDOWS,
    TYPE_PATTERNS, classify_sample, load_export3d, extract_all_features,
)

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "humic_lda_model.json")


def main():
    # =================================================================
    # STEP 1: Scan Export3D files
    # =================================================================
    print("=" * 70)
    print("STEP 1: Scanning Export3D files...")
    print("=" * 70)

    all_files = []
    seen = set()
    for base_folder in DATA_FOLDERS:
        if not os.path.isdir(base_folder):
            continue
        for entry in sorted(os.listdir(base_folder)):
            entry_path = os.path.join(base_folder, entry)
            if not os.path.isdir(entry_path) or '_SEQ' not in entry or entry.startswith('_'):
                continue
            exp_dir = os.path.join(entry_path, 'Export3d')
            if not os.path.isdir(exp_dir):
                continue
            csvs = glob.glob(os.path.join(exp_dir, '*.csv')) + \
                   glob.glob(os.path.join(exp_dir, '*.CSV'))
            for f in sorted(csvs):
                fname = os.path.basename(f)
                key = (entry, fname.upper())
                if key not in seen:
                    seen.add(key)
                    all_files.append({
                        'path': f, 'filename': fname,
                        'seq': entry, 'type': classify_sample(fname),
                    })

    exclude = {"KHP", "BLANK", "NaOH", "Buffer"}
    files = [f for f in all_files if f['type'] not in exclude]
    print(f"Files found: {len(files)}")

    # =================================================================
    # STEP 2: Extract features
    # =================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Extracting features...")
    print("=" * 70)

    records = []
    for i, f in enumerate(files):
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(files)}...")
        result = load_export3d(f['path'])
        if result is None:
            continue
        time_vals, wavelengths, data = result
        feats = extract_all_features(time_vals, wavelengths, data)
        if feats is None:
            continue
        feats['_type'] = f['type']
        feats['_seq'] = f['seq']
        feats['_filename'] = f['filename']
        records.append(feats)

    df = pd.DataFrame(records)
    meta_cols = ['_type', '_seq', '_filename']
    feat_cols = [c for c in df.columns if not c.startswith('_')]

    # Clean features
    X_all = df[feat_cols].values.astype(float)
    X_all = np.nan_to_num(X_all, nan=0, posinf=0, neginf=0)

    # Remove zero-variance features
    variances = np.var(X_all, axis=0)
    good_feats = variances > 1e-10
    feat_cols = [c for c, g in zip(feat_cols, good_feats) if g]
    X_all = X_all[:, good_feats]

    types = df['_type'].values
    print(f"Samples: {len(df)}, Features: {X_all.shape[1]}")

    # =================================================================
    # STEP 3: Identify reference samples
    # =================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Reference standards for LDA training")
    print("=" * 70)

    ha_mask = np.isin(types, ['HA', 'SRHA'])
    fa_mask = np.isin(types, ['FA', 'SRFA'])
    ref_mask = ha_mask | fa_mask

    n_ha = int(np.sum(ha_mask))
    n_fa = int(np.sum(fa_mask))
    print(f"  HA references: {n_ha}")
    print(f"  FA references: {n_fa}")

    y_ref = np.zeros(np.sum(ref_mask), dtype=int)
    y_ref[ha_mask[ref_mask]] = 1

    X_ref = X_all[ref_mask]

    # =================================================================
    # STEP 4: PCA + LDA
    # =================================================================
    print("\n" + "=" * 70)
    print("STEP 4: PCA + LDA")
    print("=" * 70)

    scaler = StandardScaler()
    X_all_scaled = scaler.fit_transform(X_all)
    X_ref_scaled = X_all_scaled[ref_mask]

    max_pcs = min(n_ha + n_fa - 2, X_ref.shape[1])
    pca = PCA(n_components=max_pcs)
    pca.fit(X_ref_scaled)

    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    n_pcs = int(np.argmax(cumvar >= 99) + 1)
    n_pcs = max(n_pcs, 5)
    n_pcs = min(n_pcs, max_pcs)

    print(f"  PCA: {n_pcs} PCs (variance: {cumvar[n_pcs-1]:.1f}%)")

    pca_final = PCA(n_components=n_pcs)
    pca_final.fit(X_ref_scaled)

    Z_ref = pca_final.transform(X_ref_scaled)
    Z_all = pca_final.transform(X_all_scaled)

    lda = LinearDiscriminantAnalysis()
    lda.fit(Z_ref, y_ref)

    # LOO cross-validation
    loo = LeaveOneOut()
    y_pred_loo = cross_val_predict(lda, Z_ref, y_ref, cv=loo)
    loo_acc = float(np.mean(y_pred_loo == y_ref))
    print(f"  LOO accuracy: {loo_acc:.1%} ({int(loo_acc * len(y_ref))}/{len(y_ref)})")

    # Compute centers
    ha_ref_scores = lda.decision_function(Z_ref[y_ref == 1])
    fa_ref_scores = lda.decision_function(Z_ref[y_ref == 0])
    ha_center = float(np.mean(ha_ref_scores))
    fa_center = float(np.mean(fa_ref_scores))
    print(f"  HA center: {ha_center:.4f}")
    print(f"  FA center: {fa_center:.4f}")
    print(f"  Gap: {abs(ha_center - fa_center):.1f}")

    # =================================================================
    # STEP 5: Serialize to JSON
    # =================================================================
    print("\n" + "=" * 70)
    print("STEP 5: Serializing model to JSON...")
    print("=" * 70)

    model = {
        "version": "2.0",
        "n_ha": n_ha,
        "n_fa": n_fa,
        "n_pcs": n_pcs,
        "loo_accuracy": round(loo_acc, 4),
        "ha_center": round(ha_center, 6),
        "fa_center": round(fa_center, 6),
        "feat_cols": feat_cols,
        "scaler_mean": [round(float(v), 8) for v in scaler.mean_],
        "scaler_scale": [round(float(v), 8) for v in scaler.scale_],
        "pca_components": [[round(float(v), 8) for v in row]
                           for row in pca_final.components_],
        "pca_mean": [round(float(v), 8) for v in pca_final.mean_],
        "lda_coef": [round(float(v), 8) for v in lda.coef_[0]],
        "lda_intercept": round(float(lda.intercept_[0]), 8),
    }

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(model, f, indent=2, ensure_ascii=False)

    size_kb = os.path.getsize(OUTPUT_PATH) / 1024
    print(f"  Saved: {OUTPUT_PATH}")
    print(f"  Size: {size_kb:.1f} KB")
    print(f"  Features: {len(feat_cols)}")
    print(f"  PCA components: {n_pcs} x {len(feat_cols)}")

    # Verify by loading back
    with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
        loaded = json.load(f)
    assert loaded["version"] == "2.0"
    assert len(loaded["feat_cols"]) == len(feat_cols)
    assert len(loaded["scaler_mean"]) == len(feat_cols)
    assert len(loaded["pca_components"]) == n_pcs
    assert len(loaded["lda_coef"]) == n_pcs
    print("  Verification: OK")

    print(f"\n{'='*70}")
    print(f"DONE — Model saved to: {OUTPUT_PATH}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
