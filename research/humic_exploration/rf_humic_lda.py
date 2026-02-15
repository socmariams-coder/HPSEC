# -*- coding: utf-8 -*-
"""
Humic Character Index v2 — PCA + LDA
=====================================
Entrena amb patrons HA/FA de referència, projecta totes les mostres
sobre l'eix discriminant HA↔FA.

Pipeline:
  1. Extreu features espectrals + elució + ràtios + sub-fraccions
  2. PCA per reduir dimensionalitat (31 refs → ~15 PCs)
  3. LDA sobre PCs per trobar l'eix HA↔FA
  4. Projecta totes les mostres → score continu HA↔FA
  5. Valida amb LOO cross-validation sobre refs
"""

import os, sys, re, glob, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from collections import Counter
from scipy import stats
from scipy.spatial.distance import cosine

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneOut, cross_val_predict

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

# =============================================================================
# CONFIG
# =============================================================================
DATA_FOLDERS = [
    "C:/Users/Lequia/Desktop/Dades3",
    "C:/Users/Lequia/OneDrive - Universitat de Girona/UdG365_HPLC-DAD - General/Dades",
]
OUTPUT_DIR = "C:/Users/Lequia/Desktop/HPSEC/rf_humic_analysis"

HS_WINDOW = (18.0, 23.0)
HS_SUBWINDOWS = {
    "early": (18.0, 19.5),
    "mid":   (19.5, 21.0),
    "late":  (21.0, 23.0),
}
WL_ALL = list(range(200, 402, 2))

TYPE_PATTERNS = [
    ("SRHA",     r"(?i)^(SRHA|3S101H)"),
    ("SRFA",     r"(?i)^(SRFA|3S101F)"),
    ("HA",       r"(?i)(?:^|[_\-\s])HA(?:[_\-\s\.]|$|\d)"),
    ("FA",       r"(?i)(?:^|[_\-\s])FA(?:[_\-\s\.]|$|\d)"),
    ("MIX",      r"(?i)(?:^|[_\-\s])MIX(?:[_\-\s\.]|$|\d)"),
    ("NO3",      r"(?i)^(NO3|NaNO3)"),
    ("Br",       r"(?i)^(Br[_\.\s\d]|KBr)"),
    ("CaCO3",    r"(?i)^CaCO3"),
    ("Na2CO3",   r"(?i)^Na2CO3"),
    ("Na2SO4",   r"(?i)^Na2SO4"),
    ("NaOH",     r"(?i)^NaOH"),
    ("Buffer",   r"(?i)^Buffer"),
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
    base = re.sub(r'[_\s]*(R?\d+)\.(csv|CSV)$', '', filename)
    for label, pattern in TYPE_PATTERNS:
        if re.search(pattern, base):
            return label
    return "OTHER"


def load_export3d(filepath):
    try:
        df = pd.read_csv(filepath, encoding='utf-16', sep=',', header=None)
        if df.shape[1] < 50:
            return None
        wavelengths = df.iloc[0, 1:].values.astype(float)
        time_vals = df.iloc[1:, 0].values.astype(float)
        data = df.iloc[1:, 1:].values.astype(float)
        if len(time_vals) < 100 or np.max(time_vals) < 15:
            return None
        return time_vals, wavelengths, data
    except:
        return None


def spectral_slope(spectrum, wl, wl_start, wl_end):
    """Log-linear spectral slope."""
    mask = (wl >= wl_start) & (wl <= wl_end)
    w = wl[mask]
    s = spectrum[mask]
    valid = s > 0
    if np.sum(valid) < 3:
        return np.nan
    slope, _, _, _, _ = stats.linregress(w[valid], np.log(s[valid]))
    return slope


def extract_all_features(time_vals, wavelengths, data):
    """Extract comprehensive feature set for HA/FA discrimination."""
    wl = wavelengths.astype(float)
    features = {}

    def get_val(spec, target):
        idx = np.argmin(np.abs(wl - target))
        return spec[idx] if idx < len(spec) else np.nan

    def fraction_spec(t0, t1):
        mask = (time_vals >= t0) & (time_vals <= t1)
        if np.sum(mask) < 3:
            return None
        return np.mean(data[mask, :], axis=0)

    # --- HS full spectrum (normalized at 254) ---
    hs = fraction_spec(18.0, 23.0)
    if hs is None:
        return None

    a254 = get_val(hs, 254)
    if a254 < 0.01:
        return None  # Too weak

    # Normalized HS spectrum
    hs_norm = hs / a254
    for w in WL_ALL:
        features[f"hs_norm_{w}"] = get_val(hs_norm, w)

    # --- Sub-fraction spectra (normalized) ---
    for name, (t0, t1) in HS_SUBWINDOWS.items():
        sub = fraction_spec(t0, t1)
        if sub is not None:
            sub_norm = sub / get_val(sub, 254) if get_val(sub, 254) > 0.01 else sub
            for w in [200, 210, 220, 230, 240, 250, 254, 260, 270, 280, 300, 330, 350, 365, 400]:
                features[f"sub_{name}_{w}"] = get_val(sub_norm, w)

    # --- Sub-fraction ratios (early/late distribution of signal) ---
    early = fraction_spec(18.0, 19.5)
    mid = fraction_spec(19.5, 21.0)
    late = fraction_spec(21.0, 23.0)

    if early is not None and late is not None:
        for w in [210, 230, 254, 280, 300, 350]:
            e = get_val(early, w)
            l = get_val(late, w)
            m = get_val(mid, w) if mid is not None else np.nan
            features[f"ratio_el_{w}"] = e / l if l > 0.01 else np.nan
            if mid is not None and m > 0.01:
                features[f"ratio_em_{w}"] = e / m
                features[f"ratio_ml_{w}"] = m / l if l > 0.01 else np.nan

    # --- Spectral ratios (on HS) ---
    a210 = get_val(hs, 210)
    a220 = get_val(hs, 220)
    a230 = get_val(hs, 230)
    a250 = get_val(hs, 250)
    a260 = get_val(hs, 260)
    a272 = get_val(hs, 272)
    a280 = get_val(hs, 280)
    a300 = get_val(hs, 300)
    a330 = get_val(hs, 330)
    a350 = get_val(hs, 350)
    a365 = get_val(hs, 365)
    a400 = get_val(hs, 400)

    features["E2_E3"] = a250 / a365 if a365 > 0.01 else np.nan
    features["A254_A280"] = a254 / a280 if a280 > 0.01 else np.nan
    features["A210_A254"] = a210 / a254
    features["A220_A254"] = a220 / a254
    features["A280_A254"] = a280 / a254
    features["A300_A254"] = a300 / a254
    features["A350_A254"] = a350 / a254 if a350 > 0.001 else np.nan
    features["A365_A254"] = a365 / a254 if a365 > 0.001 else np.nan
    features["A400_A254"] = a400 / a254 if a400 > 0.001 else np.nan
    features["A250_A365"] = a250 / a365 if a365 > 0.01 else np.nan
    features["A230_A260"] = a230 / a260 if a260 > 0.01 else np.nan
    features["A272_A280"] = a272 / a280 if a280 > 0.01 else np.nan
    features["A280_A350"] = a280 / a350 if a350 > 0.01 else np.nan
    features["A300_A400"] = a300 / a400 if a400 > 0.01 else np.nan

    # --- Spectral slopes ---
    features["S206_240"] = spectral_slope(hs, wl, 206, 240)
    features["S240_280"] = spectral_slope(hs, wl, 240, 280)
    features["S275_295"] = spectral_slope(hs, wl, 275, 295)
    features["S300_350"] = spectral_slope(hs, wl, 300, 350)
    features["S350_400"] = spectral_slope(hs, wl, 350, 400)
    s275 = features["S275_295"]
    s350 = features["S350_400"]
    features["SR"] = s275 / s350 if (s350 and abs(s350) > 1e-6) else np.nan

    # --- Slopes on normalized spectrum ---
    features["S_norm_275_295"] = spectral_slope(hs_norm, wl, 275, 295)
    features["S_norm_350_400"] = spectral_slope(hs_norm, wl, 350, 400)

    # --- Shape descriptors ---
    uv_mask = (wl >= 200) & (wl <= 280)
    vis_mask = (wl >= 320) & (wl <= 400)
    features["shape_uv_area"] = np.trapz(hs_norm[uv_mask], wl[uv_mask])
    features["shape_vis_area"] = np.trapz(hs_norm[vis_mask], wl[vis_mask])
    features["shape_uv_vis"] = features["shape_uv_area"] / features["shape_vis_area"] \
        if features["shape_vis_area"] > 0.01 else np.nan

    # Peak wavelength of normalized spectrum
    features["shape_peak_wl"] = wl[np.argmax(hs_norm[:50])]

    # Curvature at 254nm (2nd derivative)
    idx254 = np.argmin(np.abs(wl - 254))
    if 1 < idx254 < len(hs_norm) - 1:
        features["curv_254"] = hs_norm[idx254-1] - 2*hs_norm[idx254] + hs_norm[idx254+1]

    # --- Elution profile features at key wavelengths ---
    hs_mask = (time_vals >= 18.0) & (time_vals <= 23.0)
    hs_time = time_vals[hs_mask]
    hs_data = data[hs_mask, :]

    for target_wl in [210, 230, 254, 280, 300, 350]:
        wl_idx = np.argmin(np.abs(wl - target_wl))
        profile = hs_data[:, wl_idx]
        if len(profile) < 10 or np.max(profile) < 0.01:
            continue

        norm_profile = profile / np.max(profile)
        peak_idx = np.argmax(profile)
        peak_time = hs_time[peak_idx]

        # Peak time (retention = MW)
        features[f"elut_peak_{target_wl}"] = peak_time

        # Asymmetry
        area_before = np.trapz(norm_profile[:peak_idx+1]) if peak_idx > 0 else 0
        area_after = np.trapz(norm_profile[peak_idx:]) if peak_idx < len(norm_profile)-1 else 0
        total = area_before + area_after
        features[f"elut_asym_{target_wl}"] = area_before / total if total > 0 else 0.5

        # FWHM
        above_half = norm_profile >= 0.5
        if np.any(above_half):
            features[f"elut_fwhm_{target_wl}"] = hs_time[above_half][-1] - hs_time[above_half][0]

        # Tailing factor (width at 10% height, before vs after peak)
        above_10 = norm_profile >= 0.1
        if np.any(above_10):
            t_start = hs_time[above_10][0]
            t_end = hs_time[above_10][-1]
            features[f"elut_tail_{target_wl}"] = (t_end - peak_time) / (peak_time - t_start) \
                if peak_time > t_start else np.nan

        # Peak-to-valley ratio (peak vs minimum within window)
        min_val = np.min(profile)
        features[f"elut_pv_{target_wl}"] = np.max(profile) / min_val if min_val > 0.001 else np.nan

    # --- Elution peak shift between wavelengths ---
    if "elut_peak_210" in features and "elut_peak_280" in features:
        features["peak_shift_210_280"] = features["elut_peak_210"] - features["elut_peak_280"]
    if "elut_peak_254" in features and "elut_peak_350" in features:
        features["peak_shift_254_350"] = features["elut_peak_254"] - features["elut_peak_350"]
    if "elut_peak_210" in features and "elut_peak_350" in features:
        features["peak_shift_210_350"] = features["elut_peak_210"] - features["elut_peak_350"]

    # --- BB and BioP context (normalized) ---
    for frac, (t0, t1) in [("BB", (23, 26)), ("BioP", (10.8, 18))]:
        fspec = fraction_spec(t0, t1)
        if fspec is not None:
            fa254 = get_val(fspec, 254)
            if fa254 > 0.01:
                for w in [210, 230, 254, 280, 300, 350]:
                    features[f"{frac}_norm_{w}"] = get_val(fspec, w) / fa254

            # HS/fraction ratio at key wavelengths
            for w in [210, 254, 280, 350]:
                fval = get_val(fspec, w)
                hval = get_val(hs, w)
                if fval > 0.01:
                    features[f"hs_{frac}_ratio_{w}"] = hval / fval

    return features


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # =================================================================
    # STEP 1: Scan
    # =================================================================
    print("=" * 70)
    print("STEP 1: Scanning Export3D...")
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
    print(f"Files: {len(files)}")

    # =================================================================
    # STEP 2: Extract features
    # =================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Extracting features...")
    print("=" * 70)

    records = []
    for i, f in enumerate(files):
        if (i+1) % 200 == 0:
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
    # STEP 3: Identify HA and FA reference samples
    # =================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Reference standards for LDA training")
    print("=" * 70)

    ha_mask = np.isin(types, ['HA', 'SRHA'])
    fa_mask = np.isin(types, ['FA', 'SRFA'])
    ref_mask = ha_mask | fa_mask

    n_ha = np.sum(ha_mask)
    n_fa = np.sum(fa_mask)
    print(f"  HA references (HA + SRHA): {n_ha}")
    print(f"  FA references (FA + SRFA): {n_fa}")
    print(f"  Total training: {n_ha + n_fa}")

    # Labels: HA=1, FA=0
    y_ref = np.zeros(np.sum(ref_mask), dtype=int)
    y_ref[ha_mask[ref_mask]] = 1

    X_ref = X_all[ref_mask]

    # Show which samples
    print("\n  HA samples:")
    for idx in np.where(ha_mask)[0]:
        print(f"    {df.iloc[idx]['_filename']:30s}  ({df.iloc[idx]['_seq']})")
    print("  FA samples:")
    for idx in np.where(fa_mask)[0]:
        print(f"    {df.iloc[idx]['_filename']:30s}  ({df.iloc[idx]['_seq']})")

    # =================================================================
    # STEP 4: PCA → LDA pipeline
    # =================================================================
    print("\n" + "=" * 70)
    print("STEP 4: PCA + LDA")
    print("=" * 70)

    # Standardize
    scaler = StandardScaler()
    X_all_scaled = scaler.fit_transform(X_all)
    X_ref_scaled = X_all_scaled[ref_mask]

    # PCA: keep enough PCs to explain 99% of ref variance
    # but max n_ref - 2 to leave room for LDA
    max_pcs = min(n_ha + n_fa - 2, X_ref.shape[1])
    pca = PCA(n_components=max_pcs)
    pca.fit(X_ref_scaled)

    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    n_pcs = np.argmax(cumvar >= 99) + 1
    n_pcs = max(n_pcs, 5)  # Minimum 5 PCs
    n_pcs = min(n_pcs, max_pcs)

    print(f"  PCA on reference samples: {max_pcs} max PCs")
    print(f"  Variance at {n_pcs} PCs: {cumvar[n_pcs-1]:.1f}%")
    for i in range(min(10, len(cumvar))):
        print(f"    PC{i+1}: {pca.explained_variance_ratio_[i]*100:.1f}% (cum: {cumvar[i]:.1f}%)")

    # Re-fit PCA with selected n_pcs on ALL data (ref defines the space)
    pca_final = PCA(n_components=n_pcs)
    pca_final.fit(X_ref_scaled)

    Z_ref = pca_final.transform(X_ref_scaled)
    Z_all = pca_final.transform(X_all_scaled)

    # LDA on PCA scores
    lda = LinearDiscriminantAnalysis()
    lda.fit(Z_ref, y_ref)

    # LOO cross-validation on references
    loo = LeaveOneOut()
    y_pred_loo = cross_val_predict(lda, Z_ref, y_ref, cv=loo)
    loo_acc = np.mean(y_pred_loo == y_ref)
    print(f"\n  LOO cross-validation accuracy: {loo_acc:.1%} ({int(loo_acc * len(y_ref))}/{len(y_ref)})")

    # Misclassified references
    misclass = y_pred_loo != y_ref
    if np.any(misclass):
        print("  Misclassified:")
        ref_indices = np.where(ref_mask)[0]
        for idx in np.where(misclass)[0]:
            orig_idx = ref_indices[idx]
            true_label = "HA" if y_ref[idx] == 1 else "FA"
            pred_label = "HA" if y_pred_loo[idx] == 1 else "FA"
            print(f"    {df.iloc[orig_idx]['_filename']:30s} true={true_label} pred={pred_label}")
    else:
        print("  All references correctly classified!")

    # =================================================================
    # STEP 5: Project all samples → HA/FA score
    # =================================================================
    print("\n" + "=" * 70)
    print("STEP 5: Computing HA/FA score for all samples")
    print("=" * 70)

    # LDA decision function: positive = HA, negative = FA
    lda_scores = lda.decision_function(Z_all)

    # LDA probability of HA
    lda_proba = lda.predict_proba(Z_all)[:, 1]  # P(HA)

    # Normalize score to 0-100 scale: 0=pure FA, 100=pure HA
    # Use reference means as anchors
    ha_ref_scores = lda.decision_function(Z_ref[y_ref == 1])
    fa_ref_scores = lda.decision_function(Z_ref[y_ref == 0])
    ha_center = np.mean(ha_ref_scores)
    fa_center = np.mean(fa_ref_scores)

    # Linear rescale: fa_center → 0, ha_center → 100
    score_range = ha_center - fa_center
    if abs(score_range) > 1e-6:
        ha_fa_score = (lda_scores - fa_center) / score_range * 100
    else:
        ha_fa_score = np.full_like(lda_scores, 50.0)

    # Classify
    def classify(score):
        if score > 60:
            return "HA-dominant"
        elif score < 40:
            return "FA-dominant"
        else:
            return "Mixed"

    characters = [classify(s) for s in ha_fa_score]

    # Build results DataFrame
    results = pd.DataFrame({
        'Sample': df['_filename'].values,
        'SEQ': df['_seq'].values,
        'Type': types,
        'LDA_score': np.round(lda_scores, 4),
        'P(HA)': np.round(lda_proba, 4),
        'HA_FA_index': np.round(ha_fa_score, 1),
        'Character': characters,
    })

    print(f"\nCharacter distribution:")
    print(results['Character'].value_counts().to_string())

    # Verify references
    print(f"\n  Reference verification:")
    for ref_type in ['HA', 'SRHA', 'FA', 'SRFA', 'MIX']:
        mask = results['Type'] == ref_type
        if mask.sum() > 0:
            sub = results[mask]
            print(f"    {ref_type:6s}: score={sub['HA_FA_index'].mean():6.1f} "
                  f"P(HA)={sub['P(HA)'].mean():.3f}  [{sub['Character'].mode().iloc[0]}]")

    # =================================================================
    # STEP 6: Summary per type
    # =================================================================
    print("\n" + "=" * 70)
    print("STEP 6: Summary per sample type")
    print("=" * 70)

    water_types = ["FR", "PTLL", "PTT", "SK", "ATL_2024", "EX", "LQ", "EXT",
                   "EP", "SP", "POST_O3", "EP_PTL", "EP_PTT", "LAB", "LAB_T"]

    print(f"\n{'Type':15s} {'n':>4s}  {'HA/FA idx':>9s}  {'P(HA)':>7s}  {'Character':>12s}")
    print("-" * 60)
    for t in sorted(results['Type'].unique(),
                    key=lambda t: results[results['Type']==t]['HA_FA_index'].median()):
        sub = results[results['Type'] == t]
        if len(sub) < 1:
            continue
        char = sub['Character'].mode().iloc[0]
        marker = "***" if t in ['HA', 'SRHA', 'FA', 'SRFA', 'MIX'] else "   "
        print(f"{marker}{t:12s} {len(sub):4d}  "
              f"{sub['HA_FA_index'].median():9.1f}  "
              f"{sub['P(HA)'].mean():7.3f}  "
              f"{char:>12s}")

    # =================================================================
    # STEP 7: Save Excel
    # =================================================================
    print("\n" + "=" * 70)
    print("STEP 7: Saving results...")
    print("=" * 70)

    xlsx_path = os.path.join(OUTPUT_DIR, "humic_index_v2_LDA.xlsx")
    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        results.sort_values(['Type', 'SEQ', 'Sample']).to_excel(
            writer, sheet_name='Per_Sample', index=False)

        # Per type
        type_sum = results.groupby('Type').agg({
            'HA_FA_index': ['median', 'mean', 'std', 'min', 'max', 'count'],
            'P(HA)': ['mean', 'std'],
        }).round(3)
        type_sum.columns = ['_'.join(c) for c in type_sum.columns]
        type_sum = type_sum.sort_values('HA_FA_index_median')
        type_sum.to_excel(writer, sheet_name='Per_Type')

        # Per SEQ
        seq_sum = results.groupby(['SEQ', 'Type']).agg({
            'HA_FA_index': ['median', 'mean', 'std', 'count'],
            'P(HA)': ['mean'],
        }).round(3)
        seq_sum.columns = ['_'.join(c) for c in seq_sum.columns]
        seq_sum.to_excel(writer, sheet_name='Per_SEQ')

        # LDA model info
        model_info = pd.DataFrame({
            'Parameter': ['n_HA_ref', 'n_FA_ref', 'n_PCs', 'PCA_var_explained',
                          'LOO_accuracy', 'HA_center', 'FA_center', 'n_features'],
            'Value': [n_ha, n_fa, n_pcs, f"{cumvar[n_pcs-1]:.1f}%",
                      f"{loo_acc:.1%}", f"{ha_center:.4f}", f"{fa_center:.4f}",
                      X_all.shape[1]],
        })
        model_info.to_excel(writer, sheet_name='Model_Info', index=False)

        # Feature importance (LDA coefficients through PCA loadings)
        lda_coef = lda.coef_[0]  # LDA weights on PCs
        # Project back to original features
        feat_importance = pca_final.components_.T @ lda_coef
        feat_imp_df = pd.DataFrame({
            'Feature': feat_cols,
            'LDA_weight': np.round(feat_importance, 6),
            'abs_weight': np.round(np.abs(feat_importance), 6),
        }).sort_values('abs_weight', ascending=False)
        feat_imp_df.to_excel(writer, sheet_name='Feature_Importance', index=False)

    print(f"Saved: {xlsx_path}")

    # =================================================================
    # STEP 8: Visualization
    # =================================================================
    print("\nGenerating plots...")

    fig = plt.figure(figsize=(24, 24))
    gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.3)

    # --- 8a. LDA training: HA vs FA reference separation ---
    ax = fig.add_subplot(gs[0, 0])
    ha_scores_ref = ha_fa_score[ref_mask][y_ref == 1]
    fa_scores_ref = ha_fa_score[ref_mask][y_ref == 0]
    bins = np.linspace(-20, 120, 30)
    ax.hist(ha_scores_ref, bins=bins, color='#E74C3C', alpha=0.7, label=f"HA refs (n={n_ha})")
    ax.hist(fa_scores_ref, bins=bins, color='#3498DB', alpha=0.7, label=f"FA refs (n={n_fa})")
    ax.axvline(50, color='gray', linestyle='--', linewidth=1, label="Boundary")
    ax.set_xlabel("HA/FA Index (0=FA, 100=HA)")
    ax.set_ylabel("Count")
    ax.set_title(f"LDA Training: HA vs FA Separation\nLOO accuracy: {loo_acc:.0%}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- 8b. PCA space with LDA direction ---
    ax = fig.add_subplot(gs[0, 1])
    Z_all_2d = Z_all[:, :2] if Z_all.shape[1] >= 2 else Z_all
    for t in ['HA', 'SRHA']:
        m = types == t
        if m.sum() > 0:
            ax.scatter(Z_all_2d[m, 0], Z_all_2d[m, 1], s=120, marker='*',
                      c='red', edgecolors='black', zorder=10, label=t)
    for t in ['FA', 'SRFA']:
        m = types == t
        if m.sum() > 0:
            ax.scatter(Z_all_2d[m, 0], Z_all_2d[m, 1], s=120, marker='*',
                      c='blue', edgecolors='black', zorder=10, label=t)
    # Water samples colored by score
    water_mask = ~np.isin(types, ['HA', 'SRHA', 'FA', 'SRFA', 'MIX',
                                   'NO3', 'Br', 'CaCO3', 'Na2CO3', 'Na2SO4'])
    sc = ax.scatter(Z_all_2d[water_mask, 0], Z_all_2d[water_mask, 1],
                   c=ha_fa_score[water_mask], cmap='RdYlBu_r', s=10, alpha=0.5,
                   vmin=0, vmax=100)
    plt.colorbar(sc, ax=ax, label="HA/FA Index", shrink=0.8)
    ax.set_xlabel(f"PC1 ({pca_final.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca_final.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("PCA Space + LDA HA/FA Score")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- 8c. Top 20 features ---
    ax = fig.add_subplot(gs[0, 2])
    top20 = feat_imp_df.head(20)
    colors_bar = []
    for f in top20['Feature']:
        if 'elut' in f or 'peak_shift' in f:
            colors_bar.append('#E74C3C')
        elif any(f.startswith(p) for p in ['E2', 'A2', 'A3', 'A4', 'S2', 'S3', 'SR',
                                           'shape', 'curv', 'ratio']):
            colors_bar.append('#3498DB')
        elif 'sub_' in f:
            colors_bar.append('#F39C12')
        else:
            colors_bar.append('#2ECC71')
    ax.barh(range(len(top20)), top20['abs_weight'].values, color=colors_bar)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20['Feature'].values, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("| LDA weight | (contribution to HA/FA discrimination)")
    ax.set_title("Top 20 Discriminant Features")
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color='#E74C3C', label='Elution profile'),
        Patch(color='#3498DB', label='Spectral ratios'),
        Patch(color='#F39C12', label='Sub-fractions'),
        Patch(color='#2ECC71', label='Spectral shape'),
    ], fontsize=7, loc='lower right')

    # --- 8d. HA/FA Index boxplot per type ---
    ax = fig.add_subplot(gs[1, 0:2])
    plot_types = [t for t in sorted(results['Type'].unique())
                  if results[results['Type']==t].shape[0] >= 3]
    plot_types = sorted(plot_types,
                       key=lambda t: results[results['Type']==t]['HA_FA_index'].median())
    box_data = [results[results['Type']==t]['HA_FA_index'].values for t in plot_types]
    bp = ax.boxplot(box_data, vert=True, widths=0.6, patch_artist=True)
    for patch, t in zip(bp['boxes'], plot_types):
        med = results[results['Type']==t]['HA_FA_index'].median()
        if med > 60:
            patch.set_facecolor('#FADBD8')
        elif med < 40:
            patch.set_facecolor('#D6EAF8')
        else:
            patch.set_facecolor('#D5F5E3')
        if t in ['HA', 'SRHA', 'FA', 'SRFA', 'MIX']:
            patch.set_edgecolor('black')
            patch.set_linewidth(2)
    ax.set_xticklabels(plot_types, rotation=45, ha='right', fontsize=8)
    ax.axhline(50, color='gray', linestyle='--', linewidth=1)
    ax.axhspan(40, 60, alpha=0.1, color='green', label="Mixed zone")
    ax.set_ylabel("HA/FA Index (0=FA, 100=HA)")
    ax.set_title("Humic Character by Sample Type (PCA+LDA)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # --- 8e. Character pie ---
    ax = fig.add_subplot(gs[1, 2])
    water_results = results[results['Type'].isin(water_types)]
    if len(water_results) > 0:
        char_counts = water_results['Character'].value_counts()
        colors_pie = {'HA-dominant': '#E74C3C', 'FA-dominant': '#3498DB', 'Mixed': '#2ECC71'}
        ax.pie(char_counts.values,
               labels=[f"{k}\n({v}, {v/len(water_results)*100:.0f}%)"
                       for k, v in char_counts.items()],
               colors=[colors_pie.get(k, 'gray') for k in char_counts.index],
               startangle=90, textprops={'fontsize': 10})
        ax.set_title(f"Water Samples (n={len(water_results)})")

    # --- 8f. Temporal evolution ---
    ax = fig.add_subplot(gs[2, 0:2])
    for wt, color in [("FR", "#E74C3C"), ("PTLL", "#3498DB"), ("PTT", "#2ECC71"),
                       ("ATL_2024", "#F39C12"), ("SK", "#9B59B6"), ("EX", "#1ABC9C")]:
        sub = results[results['Type'] == wt].copy()
        if len(sub) < 5:
            continue
        sub['seq_num'] = sub['SEQ'].apply(
            lambda s: int(re.match(r'(\d+)', s).group(1)) if re.match(r'(\d+)', s) else 0)
        sub = sub.sort_values('seq_num')
        ax.scatter(sub['seq_num'], sub['HA_FA_index'], s=12, alpha=0.3, color=color)
        if len(sub) > 5:
            from scipy.ndimage import uniform_filter1d
            smooth = uniform_filter1d(sub['HA_FA_index'].values.astype(float),
                                     size=min(7, len(sub)//2))
            ax.plot(sub['seq_num'].values, smooth, color=color, linewidth=2.5, label=wt)
    ax.axhline(50, color='gray', linestyle='--')
    ax.axhspan(40, 60, alpha=0.1, color='green')
    ax.set_xlabel("SEQ number (chronological)")
    ax.set_ylabel("HA/FA Index")
    ax.set_title("Temporal Evolution of Humic Character")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- 8g. Per-plant detail: spectrum + score ---
    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')
    txt = "HUMIC CHARACTER INDEX v2\n"
    txt += "=" * 30 + "\n\n"
    txt += "Method: PCA + LDA\n"
    txt += f"  Training: {n_ha} HA + {n_fa} FA refs\n"
    txt += f"  PCs: {n_pcs} ({cumvar[n_pcs-1]:.0f}% var)\n"
    txt += f"  LOO accuracy: {loo_acc:.0%}\n"
    txt += f"  Features: {X_all.shape[1]}\n\n"
    txt += "Score interpretation:\n"
    txt += "  0-40:  FA-dominant\n"
    txt += "  40-60: Mixed\n"
    txt += "  60-100: HA-dominant\n\n"
    txt += f"Results: {len(results)} samples\n"
    char_c = results['Character'].value_counts()
    for k, v in char_c.items():
        txt += f"  {k}: {v}\n"
    txt += f"\nOutput: humic_index_v2_LDA.xlsx"
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # --- 8h. Detailed per-plant fingerprints ---
    ax_row3 = []
    plant_list = [t for t in ["FR", "PTLL", "PTT", "SK", "ATL_2024", "EX"]
                  if results[results['Type']==t].shape[0] >= 5]
    for idx, plant in enumerate(plant_list[:3]):
        ax = fig.add_subplot(gs[3, idx])
        sub = results[results['Type'] == plant]
        scores = sub['HA_FA_index'].values
        ax.hist(scores, bins=20, color='#3498DB', alpha=0.7, edgecolor='white')
        ax.axvline(50, color='gray', linestyle='--')
        ax.axvline(np.median(scores), color='red', linewidth=2,
                  label=f"Median: {np.median(scores):.1f}")
        ax.set_xlabel("HA/FA Index")
        ax.set_ylabel("Count")
        ax.set_title(f"{plant} (n={len(sub)})")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.savefig(os.path.join(OUTPUT_DIR, "09_humic_LDA_overview.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 09_humic_LDA_overview.png")

    # === Extra plot: per-plant fingerprint comparison ===
    plant_all = [t for t in ["FR", "PTLL", "PTT", "SK", "ATL_2024", "EX", "LQ", "EXT"]
                 if results[results['Type']==t].shape[0] >= 5]
    n_p = len(plant_all)
    fig, axes = plt.subplots(2, (n_p+1)//2, figsize=(6*((n_p+1)//2), 10))
    if n_p > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    for idx, plant in enumerate(plant_all):
        ax = axes[idx]
        sub_res = results[results['Type'] == plant]
        scores = sub_res['HA_FA_index'].values

        ax.hist(scores, bins=25, color='#3498DB', alpha=0.7, edgecolor='white', density=True)
        ax.axvline(50, color='gray', linestyle='--', linewidth=1)
        ax.axvline(np.median(scores), color='red', linewidth=2)
        ax.axvspan(40, 60, alpha=0.1, color='green')

        med = np.median(scores)
        char = "HA-dom" if med > 60 else "FA-dom" if med < 40 else "Mixed"
        ax.set_title(f"{plant} (n={len(sub_res)})\nMedian={med:.1f} → {char}", fontsize=10)
        ax.set_xlabel("HA/FA Index")
        ax.set_xlim(-20, 120)

    for idx in range(n_p, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "10_plant_LDA_distributions.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 10_plant_LDA_distributions.png")

    print(f"\n{'='*70}")
    print(f"DONE — Results in: {OUTPUT_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
