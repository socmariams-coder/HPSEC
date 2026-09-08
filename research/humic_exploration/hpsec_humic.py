# -*- coding: utf-8 -*-
"""
hpsec_humic.py — Humic Character Index (HCI) runtime module
=============================================================
Computes HCI (0=FA, 100=HA) from Export3D DAD files using a
pre-trained PCA+LDA model serialized to JSON.

Dependencies: numpy, scipy.stats.linregress (no sklearn needed).

Usage:
    from hpsec_humic import compute_hci
    result = compute_hci("path/to/export3d.csv")
    # result = {"hci": 42.5, "character": "FA-dominant", "lda_score": -1.23}
    # or None on error

Model (humic_lda_model.json):
    - Training: 14 HA (HA + SRHA) + 17 FA (FA + SRFA), 10 SEQs
    - 242 features (spectre HS normalitzat + ratios + slopes + perfils elució)
    - PCA 11 PCs (99.2% variància) + LDA binari
    - LOO accuracy: 93.5% (29/31)

Validació independent amb estàndards IHSS (no inclosos al training):
    - SK097 (3S101F, SRFA III):  HCI ≈ -101  → FA correcte ✓
    - SK098 (2S101F, SRFA II):   HCI ≈  -56  → FA correcte ✓
      (lots DIFERENTS dels SRFA del training → generalitza bé per FA)
    - SK096 (2R101N, SRNOM II):  HCI ≈  -97  → FA-dominant
      Esperat: la SRNOM d'IHSS és ~80% FA per massa (Suwannee River).
      En HPSEC-DAD el component FA domina el senyal espectral.
    - SK099 (1R110N, Nordic NOM): HCI = -568  → FORA DE RANG
      Causa: senyal DAD massa feble (1.2 mAU a 254nm vs ~10 de les altres).
      La Nordic NOM és poc aromàtica (SUVA molt baix, origen lacustre nòrdic)
      i estava preparada a concentració insuficient per HPSEC-DAD.
      A senyal tan feble, ~50/242 features no es poden calcular → extrapolació.

Limitacions:
    - L'escala 0-100 reflecteix la variabilitat DINS les refs del training.
      Mostres espectralment diferents poden donar valors fora de rang.
    - El training usa ~6 productes únics (3 HA, 3 FA) repetits en 38 injeccions.
      La classificació qualitativa (FA/Mixed/HA) és fiable; el valor numèric
      és un índex relatiu, no un percentatge absolut.
    - Mostres amb senyal feble (a254 < 0.5 mAU a la zona HS) poden donar
      scores extrems per extrapolació. compute_hci() retorna None si
      a254 < 0.01 o si la distància al training és excessiva.
"""

import os
import json
import numpy as np
from scipy import stats

# =============================================================================
# CONFIG (must match rf_humic_lda.py)
# =============================================================================
HS_WINDOW = (18.0, 23.0)
HS_SUBWINDOWS = {
    "early": (18.0, 19.5),
    "mid":   (19.5, 21.0),
    "late":  (21.0, 23.0),
}
WL_ALL = list(range(200, 402, 2))

# Signal thresholds
MIN_SIGNAL_MAU = 0.01          # Minimum signal at 254nm to compute HCI (mAU)
MIN_RATIO_DENOMINATOR = 0.01   # Minimum denominator for spectral ratios

# Model confidence guards
MAX_MISSING_FEATURES_PCT = 0.10  # Max fraction of missing/NaN features (10%)
MAX_TRAINING_DISTANCE = 30       # Max scaled Euclidean distance from training center

# HCI classification thresholds
HCI_HA_THRESHOLD = 60    # HCI > 60 → "HA-dominant"
HCI_FA_THRESHOLD = 40    # HCI < 40 → "FA-dominant", else "Mixed"

# =============================================================================
# MODEL SINGLETON
# =============================================================================
_cached_model = None


def load_humic_model(path=None):
    """Load the serialized PCA+LDA model from JSON (cached singleton)."""
    global _cached_model
    if _cached_model is not None:
        return _cached_model

    if path is None:
        path = os.path.join(os.path.dirname(__file__), "humic_lda_model.json")

    if not os.path.exists(path):
        return None

    with open(path, 'r', encoding='utf-8') as f:
        model = json.load(f)

    # Convert lists to numpy arrays for efficient computation
    model["scaler_mean"] = np.array(model["scaler_mean"], dtype=np.float64)
    model["scaler_scale"] = np.array(model["scaler_scale"], dtype=np.float64)
    model["pca_components"] = np.array(model["pca_components"], dtype=np.float64)
    model["pca_mean"] = np.array(model["pca_mean"], dtype=np.float64)
    model["lda_coef"] = np.array(model["lda_coef"], dtype=np.float64)

    _cached_model = model
    return model


# =============================================================================
# FEATURE EXTRACTION (copy from rf_humic_lda.py, no sklearn needed)
# =============================================================================

def _spectral_slope(spectrum, wl, wl_start, wl_end):
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
    """Extract comprehensive feature set for HA/FA discrimination.

    Exact copy of rf_humic_lda.py:extract_all_features() to ensure
    feature alignment with the training pipeline.
    """
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
    if a254 < MIN_SIGNAL_MAU:
        return None  # Too weak

    # Normalized HS spectrum
    hs_norm = hs / a254
    for w in WL_ALL:
        features[f"hs_norm_{w}"] = get_val(hs_norm, w)

    # --- Sub-fraction spectra (normalized) ---
    for name, (t0, t1) in HS_SUBWINDOWS.items():
        sub = fraction_spec(t0, t1)
        if sub is not None:
            sub_norm = sub / get_val(sub, 254) if get_val(sub, 254) > MIN_RATIO_DENOMINATOR else sub
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
            features[f"ratio_el_{w}"] = e / l if l > MIN_RATIO_DENOMINATOR else np.nan
            if mid is not None and m > MIN_RATIO_DENOMINATOR:
                features[f"ratio_em_{w}"] = e / m
                features[f"ratio_ml_{w}"] = m / l if l > MIN_RATIO_DENOMINATOR else np.nan

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

    features["E2_E3"] = a250 / a365 if a365 > MIN_RATIO_DENOMINATOR else np.nan
    features["A254_A280"] = a254 / a280 if a280 > MIN_RATIO_DENOMINATOR else np.nan
    features["A210_A254"] = a210 / a254
    features["A220_A254"] = a220 / a254
    features["A280_A254"] = a280 / a254
    features["A300_A254"] = a300 / a254
    features["A350_A254"] = a350 / a254 if a350 > 0.001 else np.nan
    features["A365_A254"] = a365 / a254 if a365 > 0.001 else np.nan
    features["A400_A254"] = a400 / a254 if a400 > 0.001 else np.nan
    features["A250_A365"] = a250 / a365 if a365 > MIN_RATIO_DENOMINATOR else np.nan
    features["A230_A260"] = a230 / a260 if a260 > MIN_RATIO_DENOMINATOR else np.nan
    features["A272_A280"] = a272 / a280 if a280 > MIN_RATIO_DENOMINATOR else np.nan
    features["A280_A350"] = a280 / a350 if a350 > MIN_RATIO_DENOMINATOR else np.nan
    features["A300_A400"] = a300 / a400 if a400 > MIN_RATIO_DENOMINATOR else np.nan

    # --- Spectral slopes ---
    features["S206_240"] = _spectral_slope(hs, wl, 206, 240)
    features["S240_280"] = _spectral_slope(hs, wl, 240, 280)
    features["S275_295"] = _spectral_slope(hs, wl, 275, 295)
    features["S300_350"] = _spectral_slope(hs, wl, 300, 350)
    features["S350_400"] = _spectral_slope(hs, wl, 350, 400)
    s275 = features["S275_295"]
    s350 = features["S350_400"]
    features["SR"] = s275 / s350 if (s350 and abs(s350) > 1e-6) else np.nan

    # --- Slopes on normalized spectrum ---
    features["S_norm_275_295"] = _spectral_slope(hs_norm, wl, 275, 295)
    features["S_norm_350_400"] = _spectral_slope(hs_norm, wl, 350, 400)

    # --- Shape descriptors ---
    uv_mask = (wl >= 200) & (wl <= 280)
    vis_mask = (wl >= 320) & (wl <= 400)
    features["shape_uv_area"] = np.trapezoid(hs_norm[uv_mask], wl[uv_mask])
    features["shape_vis_area"] = np.trapezoid(hs_norm[vis_mask], wl[vis_mask])
    features["shape_uv_vis"] = features["shape_uv_area"] / features["shape_vis_area"] \
        if features["shape_vis_area"] > MIN_RATIO_DENOMINATOR else np.nan

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
        if len(profile) < 10 or np.max(profile) < MIN_SIGNAL_MAU:
            continue

        norm_profile = profile / np.max(profile)
        peak_idx = np.argmax(profile)
        peak_time = hs_time[peak_idx]

        features[f"elut_peak_{target_wl}"] = peak_time

        area_before = np.trapezoid(norm_profile[:peak_idx+1]) if peak_idx > 0 else 0
        area_after = np.trapezoid(norm_profile[peak_idx:]) if peak_idx < len(norm_profile)-1 else 0
        total = area_before + area_after
        features[f"elut_asym_{target_wl}"] = area_before / total if total > 0 else 0.5

        above_half = norm_profile >= 0.5
        if np.any(above_half):
            features[f"elut_fwhm_{target_wl}"] = hs_time[above_half][-1] - hs_time[above_half][0]

        above_10 = norm_profile >= 0.1
        if np.any(above_10):
            t_start = hs_time[above_10][0]
            t_end = hs_time[above_10][-1]
            features[f"elut_tail_{target_wl}"] = (t_end - peak_time) / (peak_time - t_start) \
                if peak_time > t_start else np.nan

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
            if fa254 > MIN_RATIO_DENOMINATOR:
                for w in [210, 230, 254, 280, 300, 350]:
                    features[f"{frac}_norm_{w}"] = get_val(fspec, w) / fa254

            for w in [210, 254, 280, 350]:
                fval = get_val(fspec, w)
                hval = get_val(hs, w)
                if fval > MIN_RATIO_DENOMINATOR:
                    features[f"hs_{frac}_ratio_{w}"] = hval / fval

    return features


# =============================================================================
# LOAD EXPORT3D (lightweight, no pandas dependency for prediction)
# =============================================================================

def _load_export3d_for_hci(filepath):
    """Load Export3D CSV file and return (time_vals, wavelengths, data) arrays.

    Uses pandas (already available in the project) for robust CSV parsing.
    """
    import pandas as pd

    for enc in ["utf-16", "utf-8"]:
        try:
            df = pd.read_csv(filepath, encoding=enc, sep=',', header=None)
            if df.shape[1] < 50:
                continue
            wavelengths = df.iloc[0, 1:].values.astype(float)
            time_vals = df.iloc[1:, 0].values.astype(float)
            data = df.iloc[1:, 1:].values.astype(float)
            if len(time_vals) < 100 or np.max(time_vals) < 15:
                continue
            return time_vals, wavelengths, data
        except Exception:
            continue
    return None


# =============================================================================
# MAIN API: compute_hci
# =============================================================================

def compute_hci(dad_export3d_path):
    """Compute Humic Character Index from an Export3D DAD file.

    Args:
        dad_export3d_path: Path to the Export3D CSV file

    Returns:
        dict with {"hci": float, "character": str, "lda_score": float}
        or None on error (file not found, weak signal, model missing)
    """
    if not dad_export3d_path or not os.path.exists(dad_export3d_path):
        return None

    # Load model
    model = load_humic_model()
    if model is None:
        return None

    # Read Export3D
    result = _load_export3d_for_hci(dad_export3d_path)
    if result is None:
        return None
    time_vals, wavelengths, data = result

    # Extract features
    feats = extract_all_features(time_vals, wavelengths, data)
    if feats is None:
        return None

    # Align features to model's feat_cols (missing features → 0.0)
    feat_cols = model["feat_cols"]
    n_missing = sum(1 for col in feat_cols if col not in feats)
    n_nan = sum(1 for col in feat_cols
                if col in feats and isinstance(feats[col], float) and np.isnan(feats[col]))

    # Guard: si falten >10% de features, el score serà poc fiable
    # (ex: Nordic NOM amb senyal feble → 50/242 features absents → HCI = -568)
    if (n_missing + n_nan) > len(feat_cols) * MAX_MISSING_FEATURES_PCT:
        return None

    X = np.array([feats.get(col, 0.0) for col in feat_cols], dtype=np.float64)
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    # StandardScaler: X_scaled = (X - mean) / scale
    X_scaled = (X - model["scaler_mean"]) / model["scaler_scale"]

    # Guard: distància escalada al centre del training
    # Mostres molt lluny del training donen scores per extrapolació (no fiables)
    # Ref: SRFA nova → dist ~9, Nordic NOM → dist ~82
    dist = float(np.sqrt(np.sum(X_scaled ** 2)))
    if dist > MAX_TRAINING_DISTANCE:
        return None

    # PCA: X_pca = (X_scaled - pca_mean) @ pca_components.T
    X_pca = (X_scaled - model["pca_mean"]) @ model["pca_components"].T

    # LDA: score = X_pca @ lda_coef + lda_intercept
    lda_score = float(X_pca @ model["lda_coef"] + model["lda_intercept"])

    # Normalize to HCI: fa_center → 0, ha_center → 100
    ha_center = model["ha_center"]
    fa_center = model["fa_center"]
    score_range = ha_center - fa_center

    if abs(score_range) > 1e-6:
        hci = (lda_score - fa_center) / score_range * 100
    else:
        hci = 50.0

    # Classify
    if hci > HCI_HA_THRESHOLD:
        character = "HA-dominant"
    elif hci < HCI_FA_THRESHOLD:
        character = "FA-dominant"
    else:
        character = "Mixed"

    return {
        "hci": round(float(hci), 1),
        "character": character,
        "lda_score": round(lda_score, 4),
    }
