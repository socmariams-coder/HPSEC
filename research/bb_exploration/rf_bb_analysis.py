# -*- coding: utf-8 -*-
"""
Building Blocks (BB) Zone Spectral Analysis — Random Forest
=============================================================
Explora la zona BB (23-26 min) amb DAD espectral complet.
A diferencia de HS (on discriminem HA/FA), aqui volem:
  1. Quines metriques espectrals discriminen tipus d'aigua a BB
  2. Feature importance RF: quins wavelengths/ratios son rellevants
  3. Comparar amb espectres purs inorganics (NO3, Br, CaCO3)
  4. Separar contribucio organica vs inorganica

Adaptat de rf_humic_lda.py pero enfocat a BB.
"""

import os, sys, re, glob, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from collections import Counter, defaultdict
from scipy import stats

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

# =============================================================================
# CONFIG
# =============================================================================
DATA_FOLDERS = [
    "C:/Users/Lequia/Desktop/Dades",
]
OUTPUT_DIR = "C:/Users/Lequia/Desktop/HPSEC/rf_bb_analysis"

# Fraccions temporals
BB_WINDOW = (23.0, 26.0)
BB_SUBWINDOWS = {
    "early": (23.0, 24.5),
    "late":  (24.5, 26.0),
}
HS_WINDOW = (18.0, 23.0)
SB_WINDOW = (26.0, 32.0)
LMW_WINDOW = (32.0, 50.0)
BIOP_WINDOW = (10.8, 18.0)

WL_ALL = list(range(200, 402, 2))  # 101 wavelengths
WL_KEY = [200, 210, 220, 230, 240, 250, 254, 260, 270, 280, 300, 330, 350, 365, 400]
WL_RATIO = [210, 220, 230, 254, 280, 300, 350]

# Tipus de mostra (adaptat de rf_humic_lda.py + nous)
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

# Grups per analisi RF (agrupar tipus similars)
TYPE_GROUPS = {
    # Aigues de xarxa
    "FR":       ["FR"],
    "PTLL":     ["PTLL", "PTLL_2025"],
    "PTT":      ["PTT"],
    "LAB":      ["LAB"],         # Laboratori (aigua PTL tractada)
    "LAB_T":    ["LAB_T"],       # Laboratori (aigua PTT tractada)
    "EP":       ["EP", "EP_PTL", "EP_PTT"],  # Entrada planta
    "SP":       ["SP"],          # Sortida planta
    "POST_O3":  ["POST_O3"],     # Post ozonitzacio
    # Referencies organiques
    "HA":       ["HA", "SRHA"],
    "FA":       ["FA", "SRFA"],
    # Inorganics purs
    "NO3":      ["NO3"],
    "Br":       ["Br"],
    "CaCO3":    ["CaCO3"],
    "Na2CO3":   ["Na2CO3"],
    # Altres
    "EX":       ["EX", "EXT"],
    "KHP":      ["KHP"],
    "BLANK":    ["BLANK"],
}


def classify_sample(filename):
    base = re.sub(r'[_\s]*(R?\d+)\.(csv|CSV)$', '', filename)
    for label, pattern in TYPE_PATTERNS:
        if re.search(pattern, base):
            return label
    return "OTHER"


def load_export3d(filepath):
    """Llegeix Export3D CSV (UTF-16 o UTF-8)."""
    for enc in ['utf-16', 'utf-8', 'latin-1']:
        try:
            df = pd.read_csv(filepath, encoding=enc, sep=',', header=None)
            if df.shape[1] < 50:
                continue
            wavelengths = df.iloc[0, 1:].values.astype(float)
            time_vals = df.iloc[1:, 0].values.astype(float)
            data = df.iloc[1:, 1:].values.astype(float)
            if len(time_vals) < 100 or np.max(time_vals) < 15:
                continue
            if len(wavelengths) != 101:
                continue
            return time_vals, wavelengths, data
        except:
            continue
    return None


def spectral_slope(spectrum, wl, wl_start, wl_end):
    """Log-linear spectral slope over a wavelength range."""
    mask = (wl >= wl_start) & (wl <= wl_end)
    w = wl[mask]
    s = spectrum[mask]
    valid = s > 0
    if np.sum(valid) < 3:
        return np.nan
    slope, _, _, _, _ = stats.linregress(w[valid], np.log(s[valid]))
    return slope


def simple_slope(spectrum, wl, wl1, wl2):
    """Slope simple (Meritxell): (A_wl1 - A_wl2) / (wl2 - wl1) en mAU/nm."""
    idx1 = np.argmin(np.abs(wl - wl1))
    idx2 = np.argmin(np.abs(wl - wl2))
    a1 = spectrum[idx1]
    a2 = spectrum[idx2]
    return (a1 - a2) / (wl2 - wl1)


# =============================================================================
# BB FEATURE EXTRACTION
# =============================================================================
def extract_bb_features(time_vals, wavelengths, data):
    """
    Extrau features de la zona BB (23-26 min) + context de zones adjacents.

    Categories de features:
    A. Espectre BB normalitzat a 210nm (101 wl) — referencia UV, no 254 (BB pot tenir poc 254)
    B. Espectre BB normalitzat a 254nm (15 wl) — per comparar amb HS
    C. Intensitats absolutes a BB (7 wl) — concentracio real
    D. Sub-fraccions BB early/late (14 wl)
    E. Ratios espectrals BB (15+)
    F. Slopes simples Meritxell (6)
    G. Slopes log-lineals (6)
    H. Descriptors de forma (5)
    I. Perfil d'elucio BB (30)
    J. Context inter-fraccio: BB/HS, BB/SB, BB/BioP (20+)
    K. Indicadors inorganics (5)
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

    # ======== ESPECTRES BASE ========
    bb = fraction_spec(*BB_WINDOW)
    if bb is None:
        return None

    hs = fraction_spec(*HS_WINDOW)
    sb = fraction_spec(*SB_WINDOW)
    biop = fraction_spec(*BIOP_WINDOW)
    lmw = fraction_spec(*LMW_WINDOW)

    # Intensitats de referencia
    a254_bb = get_val(bb, 254)
    a210_bb = get_val(bb, 210)
    a220_bb = get_val(bb, 220)

    # Guarda: senyal minim (a 210 perque BB pot no tenir 254)
    if a210_bb < 0.005 and a254_bb < 0.005:
        return None  # Senyal massa feble (blancs)

    # ======== A. ESPECTRE BB NORMALITZAT A 210nm (101 features) ========
    # 210nm com a referencia perque BB te molt mes senyal UV que VIS
    if a210_bb > 0.01:
        bb_norm210 = bb / a210_bb
        for w in WL_ALL:
            features[f"bb210_{w}"] = get_val(bb_norm210, w)

    # ======== B. ESPECTRE BB NORMALITZAT A 254nm (15 features) ========
    if a254_bb > 0.01:
        bb_norm254 = bb / a254_bb
        for w in WL_KEY:
            features[f"bb254_{w}"] = get_val(bb_norm254, w)

    # ======== C. INTENSITATS ABSOLUTES BB (7 features) ========
    for w in WL_RATIO:
        features[f"bb_abs_{w}"] = get_val(bb, w)

    # ======== D. SUB-FRACCIONS BB (28 features) ========
    for name, (t0, t1) in BB_SUBWINDOWS.items():
        sub = fraction_spec(t0, t1)
        if sub is not None:
            sub_a210 = get_val(sub, 210)
            if sub_a210 > 0.01:
                sub_norm = sub / sub_a210
            else:
                sub_norm = sub
            for w in [200, 210, 220, 230, 240, 254, 280, 300, 330, 350, 365, 400]:
                features[f"bbsub_{name}_{w}"] = get_val(sub_norm, w)
            # Intensitat absoluta a 210 i 254
            features[f"bbsub_{name}_abs210"] = get_val(sub, 210)
            features[f"bbsub_{name}_abs254"] = get_val(sub, 254)

    # Ratio early/late a BB
    bb_early = fraction_spec(*BB_SUBWINDOWS["early"])
    bb_late = fraction_spec(*BB_SUBWINDOWS["late"])
    if bb_early is not None and bb_late is not None:
        for w in WL_RATIO:
            e = get_val(bb_early, w)
            l = get_val(bb_late, w)
            features[f"bb_el_ratio_{w}"] = e / l if l > 0.01 else np.nan

    # ======== E. RATIOS ESPECTRALS BB (15 features) ========
    a200 = get_val(bb, 200)
    a230 = get_val(bb, 230)
    a240 = get_val(bb, 240)
    a250 = get_val(bb, 250)
    a260 = get_val(bb, 260)
    a280 = get_val(bb, 280)
    a300 = get_val(bb, 300)
    a350 = get_val(bb, 350)
    a365 = get_val(bb, 365)
    a400 = get_val(bb, 400)

    # Ratios classics adaptats a BB
    features["bb_A210_A254"] = a210_bb / a254_bb if a254_bb > 0.005 else np.nan
    features["bb_A220_A254"] = a220_bb / a254_bb if a254_bb > 0.005 else np.nan
    features["bb_A200_A254"] = a200 / a254_bb if a254_bb > 0.005 else np.nan
    features["bb_A230_A254"] = a230 / a254_bb if a254_bb > 0.005 else np.nan
    features["bb_A254_A280"] = a254_bb / a280 if a280 > 0.005 else np.nan
    features["bb_A280_A254"] = a280 / a254_bb if a254_bb > 0.005 else np.nan
    features["bb_A300_A254"] = a300 / a254_bb if a254_bb > 0.005 else np.nan
    features["bb_A210_A280"] = a210_bb / a280 if a280 > 0.005 else np.nan
    features["bb_A210_A220"] = a210_bb / a220_bb if a220_bb > 0.005 else np.nan
    features["bb_E2_E3"] = a250 / a365 if a365 > 0.005 else np.nan
    features["bb_A250_A365"] = a250 / a365 if a365 > 0.005 else np.nan
    features["bb_A280_A350"] = a280 / a350 if a350 > 0.005 else np.nan
    features["bb_A300_A400"] = a300 / a400 if a400 > 0.005 else np.nan
    features["bb_A200_A210"] = a200 / a210_bb if a210_bb > 0.005 else np.nan
    features["bb_A200_A240"] = a200 / a240 if a240 > 0.005 else np.nan

    # ======== F. SLOPES SIMPLES MERITXELL (8 features) ========
    features["bb_S200_210"] = simple_slope(bb, wl, 200, 210)
    features["bb_S210_220"] = simple_slope(bb, wl, 210, 220)
    features["bb_S210_230"] = simple_slope(bb, wl, 210, 230)
    features["bb_S210_254"] = simple_slope(bb, wl, 210, 254)
    features["bb_S220_254"] = simple_slope(bb, wl, 220, 254)
    features["bb_S254_280"] = simple_slope(bb, wl, 254, 280)
    features["bb_S280_350"] = simple_slope(bb, wl, 280, 350)
    features["bb_S200_254"] = simple_slope(bb, wl, 200, 254)

    # ======== G. SLOPES LOG-LINEALS (7 features) ========
    features["bb_SL_200_220"] = spectral_slope(bb, wl, 200, 220)
    features["bb_SL_206_240"] = spectral_slope(bb, wl, 206, 240)
    features["bb_SL_220_240"] = spectral_slope(bb, wl, 220, 240)
    features["bb_SL_240_280"] = spectral_slope(bb, wl, 240, 280)
    features["bb_SL_275_295"] = spectral_slope(bb, wl, 275, 295)
    features["bb_SL_300_350"] = spectral_slope(bb, wl, 300, 350)
    s275 = features["bb_SL_275_295"]
    s350 = features["bb_SL_300_350"]
    features["bb_SR"] = s275 / s350 if (s350 and abs(s350) > 1e-6) else np.nan

    # ======== H. DESCRIPTORS DE FORMA (6 features) ========
    if a210_bb > 0.01:
        bb_n = bb / a210_bb
        uv_mask = (wl >= 200) & (wl <= 240)
        mid_mask = (wl >= 240) & (wl <= 300)
        vis_mask = (wl >= 300) & (wl <= 400)

        features["bb_shape_deep_uv"] = np.trapz(bb_n[uv_mask], wl[uv_mask])
        features["bb_shape_mid_uv"] = np.trapz(bb_n[mid_mask], wl[mid_mask])
        features["bb_shape_vis"] = np.trapz(bb_n[vis_mask], wl[vis_mask])
        features["bb_shape_uv_vis"] = features["bb_shape_deep_uv"] / features["bb_shape_vis"] \
            if features["bb_shape_vis"] > 0.01 else np.nan
        features["bb_shape_mid_vis"] = features["bb_shape_mid_uv"] / features["bb_shape_vis"] \
            if features["bb_shape_vis"] > 0.01 else np.nan

        # Peak wavelength
        features["bb_shape_peak_wl"] = wl[np.argmax(bb_n[:50])]  # Buscar pic en 200-300

        # Curvatura a 210nm i 254nm
        idx210 = np.argmin(np.abs(wl - 210))
        if 1 < idx210 < len(bb_n) - 1:
            features["bb_curv_210"] = bb_n[idx210-1] - 2*bb_n[idx210] + bb_n[idx210+1]
        idx254 = np.argmin(np.abs(wl - 254))
        if 1 < idx254 < len(bb_n) - 1:
            features["bb_curv_254"] = bb_n[idx254-1] - 2*bb_n[idx254] + bb_n[idx254+1]

    # ======== I. PERFIL D'ELUCIO BB (30 features) ========
    bb_mask = (time_vals >= BB_WINDOW[0]) & (time_vals <= BB_WINDOW[1])
    bb_time = time_vals[bb_mask]
    bb_data = data[bb_mask, :]

    for target_wl in [210, 220, 230, 254, 280, 300]:
        wl_idx = np.argmin(np.abs(wl - target_wl))
        profile = bb_data[:, wl_idx]
        if len(profile) < 5 or np.max(np.abs(profile)) < 0.005:
            continue

        norm_profile = profile / np.max(np.abs(profile))
        peak_idx = np.argmax(np.abs(profile))
        peak_time = bb_time[peak_idx]

        features[f"bb_elut_peak_{target_wl}"] = peak_time

        # Asimetria
        area_before = np.trapz(np.abs(norm_profile[:peak_idx+1])) if peak_idx > 0 else 0
        area_after = np.trapz(np.abs(norm_profile[peak_idx:])) if peak_idx < len(norm_profile)-1 else 0
        total = area_before + area_after
        features[f"bb_elut_asym_{target_wl}"] = area_before / total if total > 0 else 0.5

        # FWHM
        above_half = np.abs(norm_profile) >= 0.5
        if np.any(above_half):
            features[f"bb_elut_fwhm_{target_wl}"] = bb_time[above_half][-1] - bb_time[above_half][0]

        # Monotonia (decreixent? = difusio; pic = component discret)
        if len(profile) > 3:
            diffs = np.diff(profile)
            features[f"bb_elut_mono_{target_wl}"] = np.sum(diffs < 0) / len(diffs)

        # Gradient (pendent del perfil temporal)
        if len(profile) > 3:
            sl, _, r, _, _ = stats.linregress(bb_time, profile)
            features[f"bb_elut_slope_{target_wl}"] = sl

    # ======== J. CONTEXT INTER-FRACCIO (20+ features) ========
    # BB vs HS: quant de senyal BB relatiu a HS?
    if hs is not None:
        a254_hs = get_val(hs, 254)
        a210_hs = get_val(hs, 210)
        for w in [210, 220, 230, 254, 280, 300, 350]:
            bb_val = get_val(bb, w)
            hs_val = get_val(hs, w)
            if hs_val > 0.01:
                features[f"bb_hs_ratio_{w}"] = bb_val / hs_val
        # Diferencia de forma normalitzada
        if a254_hs > 0.01 and a254_bb > 0.01:
            hs_norm = hs / a254_hs
            bb_norm = bb / a254_bb
            for w in [200, 210, 220, 230, 240, 254, 280]:
                features[f"bb_hs_shape_diff_{w}"] = get_val(bb_norm, w) - get_val(hs_norm, w)

    # BB vs SB
    if sb is not None:
        for w in [210, 254, 280]:
            sb_val = get_val(sb, w)
            bb_val = get_val(bb, w)
            if sb_val > 0.005:
                features[f"bb_sb_ratio_{w}"] = bb_val / sb_val

    # BB vs LMW
    if lmw is not None:
        for w in [210, 254]:
            lmw_val = get_val(lmw, w)
            bb_val = get_val(bb, w)
            if lmw_val > 0.005:
                features[f"bb_lmw_ratio_{w}"] = bb_val / lmw_val

    # ======== K. INDICADORS INORGANICS (5 features) ========
    # Ratio UV profund / UV aromatic: molt alt = inorganics dominants
    features["bb_inorg_A200_A254"] = a200 / a254_bb if a254_bb > 0.005 else np.nan
    features["bb_inorg_A210_A254"] = a210_bb / a254_bb if a254_bb > 0.005 else np.nan
    features["bb_inorg_A210_A280"] = a210_bb / a280 if a280 > 0.005 else np.nan

    # "Caiguda espectral": quant cau el senyal entre 210 i 240?
    # Nitrat cau bruscament, organics cauen mes suau
    if a210_bb > 0.01:
        features["bb_inorg_drop_210_240"] = (a210_bb - a240) / a210_bb
        features["bb_inorg_drop_210_254"] = (a210_bb - a254_bb) / a210_bb

    # Convexitat espectral 200-230: nitrat te un pic definit, organics son monotons
    if a210_bb > 0.01:
        bb_n = bb / a210_bb
        idx_200 = np.argmin(np.abs(wl - 200))
        idx_210 = np.argmin(np.abs(wl - 210))
        idx_230 = np.argmin(np.abs(wl - 230))
        features["bb_inorg_convex_200_230"] = 2 * bb_n[idx_210] - bb_n[idx_200] - bb_n[idx_230]

    # NaN cleanup
    for k, v in features.items():
        if isinstance(v, float) and np.isnan(v):
            features[k] = 0.0

    return features


# =============================================================================
# SCAN ALL EXPORT3D
# =============================================================================
def scan_all_export3d():
    """Scan all Export3D folders and extract BB features."""
    records = []

    for base_dir in DATA_FOLDERS:
        if not os.path.isdir(base_dir):
            continue
        seq_dirs = sorted(glob.glob(os.path.join(base_dir, "*_SEQ*")))

        for seq_dir in seq_dirs:
            seq_name = os.path.basename(seq_dir)
            is_bp = "_BP" in seq_name.upper()
            export3d_dir = os.path.join(seq_dir, "Export3D")

            if not os.path.isdir(export3d_dir):
                continue

            csv_files = sorted(glob.glob(os.path.join(export3d_dir, "*.csv")) +
                              glob.glob(os.path.join(export3d_dir, "*.CSV")))

            for csv_path in csv_files:
                fname = os.path.basename(csv_path)
                sample_type = classify_sample(fname)

                result = load_export3d(csv_path)
                if result is None:
                    continue

                time_vals, wavelengths, data_mat = result
                feats = extract_bb_features(time_vals, wavelengths, data_mat)
                if feats is None:
                    continue

                feats["_sample_type"] = sample_type
                feats["_seq"] = seq_name
                feats["_file"] = fname
                feats["_is_bp"] = is_bp
                feats["_path"] = csv_path

                records.append(feats)

    return records


def assign_group(sample_type):
    """Assign sample to a broader group for RF classification."""
    for group, types in TYPE_GROUPS.items():
        if sample_type in types:
            return group
    return "OTHER"


# =============================================================================
# MAIN ANALYSIS
# =============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # =========================================================
    # STEP 1: Scan
    # =========================================================
    print("=" * 70)
    print("STEP 1: Scanning all Export3D files...")
    print("=" * 70)

    records = scan_all_export3d()
    print(f"\nTotal samples extracted: {len(records)}")

    # Stats per type
    type_counts = Counter(r["_sample_type"] for r in records)
    print("\nSamples per type:")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t:15s}: {c}")

    # =========================================================
    # STEP 2: Build feature matrix
    # =========================================================
    print("\n" + "=" * 70)
    print("STEP 2: Building feature matrix...")
    print("=" * 70)

    meta_cols = [c for c in records[0] if c.startswith("_")]
    feat_cols = sorted([c for c in records[0] if not c.startswith("_")])

    df = pd.DataFrame(records)
    df["_group"] = df["_sample_type"].apply(assign_group)

    X_all = df[feat_cols].values.astype(float)

    # Replace inf
    X_all[~np.isfinite(X_all)] = 0.0

    print(f"Feature matrix: {X_all.shape[0]} samples x {X_all.shape[1]} features")
    print(f"\nSamples per group:")
    group_counts = df["_group"].value_counts()
    for g, c in group_counts.items():
        print(f"  {g:15s}: {c}")

    # =========================================================
    # STEP 3: Pure inorganic spectra reference
    # =========================================================
    print("\n" + "=" * 70)
    print("STEP 3: Pure inorganic spectra at BB zone")
    print("=" * 70)

    for inorg in ["NO3", "Br", "CaCO3", "Na2CO3"]:
        inorg_df = df[df["_sample_type"] == inorg]
        if len(inorg_df) == 0:
            print(f"\n  {inorg}: no samples found")
            continue

        print(f"\n  {inorg} ({len(inorg_df)} samples):")
        # BB absolute intensities
        for w in [200, 210, 220, 230, 254, 280, 300]:
            col = f"bb_abs_{w}" if f"bb_abs_{w}" in feat_cols else None
            if col and col in inorg_df.columns:
                vals = inorg_df[col].values
                print(f"    A{w}_BB = {np.mean(vals):8.3f} mAU (std={np.std(vals):.3f})")

        # Key ratios
        for ratio_col in ["bb_A210_A254", "bb_A200_A254", "bb_inorg_drop_210_254"]:
            if ratio_col in inorg_df.columns:
                vals = inorg_df[ratio_col].values
                print(f"    {ratio_col} = {np.mean(vals):.2f}")

    # =========================================================
    # STEP 4: RF Classification (water types only)
    # =========================================================
    print("\n" + "=" * 70)
    print("STEP 4: Random Forest — water type discrimination at BB")
    print("=" * 70)

    # Filter: only water sample types with enough data for RF
    water_groups = ["FR", "PTLL", "PTT", "LAB", "LAB_T", "EP", "SP", "POST_O3", "EX"]
    mask_water = df["_group"].isin(water_groups)

    # Also include HA, FA, inorganics as reference
    ref_groups = ["HA", "FA", "NO3", "Br", "CaCO3", "Na2CO3"]
    mask_ref = df["_group"].isin(ref_groups)

    # Combine
    mask = mask_water | mask_ref
    df_rf = df[mask].copy()
    X_rf = X_all[mask.values]
    y_rf = df_rf["_group"].values

    # Min samples per class
    min_samples = 4
    group_sizes = Counter(y_rf)
    valid_groups = {g for g, c in group_sizes.items() if c >= min_samples}
    mask_valid = np.array([g in valid_groups for g in y_rf])

    X_rf = X_rf[mask_valid]
    y_rf = y_rf[mask_valid]
    df_rf = df_rf[mask_valid]

    print(f"\nRF dataset: {X_rf.shape[0]} samples, {X_rf.shape[1]} features")
    print(f"Classes ({len(set(y_rf))}):")
    for g, c in sorted(Counter(y_rf).items(), key=lambda x: -x[1]):
        print(f"  {g:15s}: {c}")

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_rf)

    # RF with stratified CV
    n_splits = min(5, min(Counter(y_rf).values()))
    if n_splits < 2:
        n_splits = 2

    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_pred = cross_val_predict(rf, X_scaled, y_rf, cv=cv)

    accuracy = np.mean(y_pred == y_rf)
    print(f"\n  Stratified {n_splits}-fold CV accuracy: {accuracy:.1%}")

    print(f"\n  Classification report:")
    report = classification_report(y_rf, y_pred, zero_division=0)
    print(report)

    # Confusion matrix
    labels = sorted(set(y_rf))
    cm = confusion_matrix(y_rf, y_pred, labels=labels)
    print("  Confusion matrix:")
    print(f"  {'':15s} " + " ".join(f"{l:>8s}" for l in labels))
    for i, row in enumerate(cm):
        print(f"  {labels[i]:15s} " + " ".join(f"{v:8d}" for v in row))

    # =========================================================
    # STEP 5: Feature importance
    # =========================================================
    print("\n" + "=" * 70)
    print("STEP 5: Feature importance (RF trained on all data)")
    print("=" * 70)

    rf_full = RandomForestClassifier(
        n_estimators=500, max_depth=None, min_samples_leaf=2,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    rf_full.fit(X_scaled, y_rf)

    # Gini importance
    importances = rf_full.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    print(f"\n  Top 40 features (Gini importance):")
    print(f"  {'Rank':>4s} {'Feature':40s} {'Importance':>10s}")
    print(f"  {'-'*4} {'-'*40} {'-'*10}")
    for rank, idx in enumerate(sorted_idx[:40], 1):
        print(f"  {rank:4d} {feat_cols[idx]:40s} {importances[idx]:10.4f}")

    # =========================================================
    # STEP 6: Permutation importance (more robust)
    # =========================================================
    print("\n" + "=" * 70)
    print("STEP 6: Permutation importance")
    print("=" * 70)

    perm_imp = permutation_importance(
        rf_full, X_scaled, y_rf, n_repeats=10, random_state=42, n_jobs=-1
    )
    perm_sorted = np.argsort(perm_imp.importances_mean)[::-1]

    print(f"\n  Top 30 features (permutation importance):")
    print(f"  {'Rank':>4s} {'Feature':40s} {'Mean':>10s} {'Std':>10s}")
    print(f"  {'-'*4} {'-'*40} {'-'*10} {'-'*10}")
    for rank, idx in enumerate(perm_sorted[:30], 1):
        print(f"  {rank:4d} {feat_cols[idx]:40s} {perm_imp.importances_mean[idx]:10.4f} {perm_imp.importances_std[idx]:10.4f}")

    # =========================================================
    # STEP 7: Feature category analysis
    # =========================================================
    print("\n" + "=" * 70)
    print("STEP 7: Importance by feature category")
    print("=" * 70)

    categories = {
        "BB spectrum (norm 210)": [c for c in feat_cols if c.startswith("bb210_")],
        "BB spectrum (norm 254)": [c for c in feat_cols if c.startswith("bb254_")],
        "BB absolute intensity": [c for c in feat_cols if c.startswith("bb_abs_")],
        "BB sub-fractions": [c for c in feat_cols if c.startswith("bbsub_")],
        "BB early/late ratio": [c for c in feat_cols if c.startswith("bb_el_ratio_")],
        "Spectral ratios": [c for c in feat_cols if c.startswith("bb_A") or c.startswith("bb_E")],
        "Simple slopes (Meritxell)": [c for c in feat_cols if c.startswith("bb_S") and not c.startswith("bb_SL") and not c.startswith("bb_SR")],
        "Log-linear slopes": [c for c in feat_cols if c.startswith("bb_SL_") or c == "bb_SR"],
        "Shape descriptors": [c for c in feat_cols if c.startswith("bb_shape_") or c.startswith("bb_curv_")],
        "Elution profile": [c for c in feat_cols if c.startswith("bb_elut_")],
        "BB/HS context": [c for c in feat_cols if c.startswith("bb_hs_")],
        "BB/SB context": [c for c in feat_cols if c.startswith("bb_sb_")],
        "BB/LMW context": [c for c in feat_cols if c.startswith("bb_lmw_")],
        "Inorganic indicators": [c for c in feat_cols if c.startswith("bb_inorg_")],
    }

    print(f"\n  {'Category':35s} {'N_feats':>7s} {'Sum_Gini':>10s} {'Max_Gini':>10s} {'Sum_Perm':>10s}")
    print(f"  {'-'*35} {'-'*7} {'-'*10} {'-'*10} {'-'*10}")

    cat_importance = []
    for cat, cols in categories.items():
        if not cols:
            continue
        idxs = [feat_cols.index(c) for c in cols if c in feat_cols]
        gini_sum = sum(importances[i] for i in idxs)
        gini_max = max(importances[i] for i in idxs) if idxs else 0
        perm_sum = sum(perm_imp.importances_mean[i] for i in idxs)
        print(f"  {cat:35s} {len(idxs):7d} {gini_sum:10.4f} {gini_max:10.4f} {perm_sum:10.4f}")
        cat_importance.append((cat, gini_sum, perm_sum))

    # =========================================================
    # STEP 8: Water type profiles at BB
    # =========================================================
    print("\n" + "=" * 70)
    print("STEP 8: BB spectral profiles per water type")
    print("=" * 70)

    key_features = ["bb_abs_210", "bb_abs_254", "bb_abs_280",
                    "bb_A210_A254", "bb_A210_A280",
                    "bb_S210_254", "bb_S210_230",
                    "bb_inorg_drop_210_254", "bb_inorg_convex_200_230",
                    "bb_hs_ratio_210", "bb_hs_ratio_254"]

    print(f"\n  {'Type':15s}", end="")
    for f in key_features:
        short = f.replace("bb_", "").replace("inorg_", "I_")[:12]
        print(f" {short:>12s}", end="")
    print()
    print(f"  {'-'*15}" + f" {'-'*12}" * len(key_features))

    profile_types = ["PTLL", "PTT", "FR", "LAB", "LAB_T", "EP", "SP", "POST_O3",
                     "EX", "HA", "FA", "NO3", "Br", "CaCO3", "Na2CO3"]

    for gtype in profile_types:
        gdf = df_rf[df_rf["_group"] == gtype] if gtype in set(y_rf) else df[df["_group"] == gtype]
        if len(gdf) == 0:
            continue
        print(f"  {gtype:15s}", end="")
        for f in key_features:
            if f in gdf.columns:
                val = gdf[f].mean()
                print(f" {val:12.3f}", end="")
            else:
                print(f" {'N/A':>12s}", end="")
        print(f"  (n={len(gdf)})")

    # =========================================================
    # STEP 9: PTLL vs PTT specific comparison
    # =========================================================
    print("\n" + "=" * 70)
    print("STEP 9: PTLL vs PTT detailed comparison at BB")
    print("=" * 70)

    ptll_df = df[df["_group"] == "PTLL"]
    ptt_df = df[df["_group"] == "PTT"]

    if len(ptll_df) > 0 and len(ptt_df) > 0:
        print(f"\n  PTLL: {len(ptll_df)} samples | PTT: {len(ptt_df)} samples")

        print(f"\n  {'Feature':40s} {'PTLL':>10s} {'PTT':>10s} {'Ratio':>8s} {'t-stat':>8s} {'p-val':>10s}")
        print(f"  {'-'*40} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*10}")

        # Test all features
        significant = []
        for col in feat_cols:
            v1 = ptll_df[col].values
            v2 = ptt_df[col].values
            if np.std(v1) < 1e-10 and np.std(v2) < 1e-10:
                continue
            t_stat, p_val = stats.ttest_ind(v1, v2, equal_var=False)
            m1, m2 = np.mean(v1), np.mean(v2)
            ratio = m1 / m2 if abs(m2) > 0.001 else np.nan
            if p_val < 0.05:
                significant.append((col, m1, m2, ratio, t_stat, p_val))

        # Sort by p-value
        significant.sort(key=lambda x: x[5])

        print(f"\n  Top 30 significant differences (p<0.05, n={len(significant)} total):")
        for col, m1, m2, ratio, t_stat, p_val in significant[:30]:
            stars = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*"
            print(f"  {col:40s} {m1:10.3f} {m2:10.3f} {ratio:8.2f} {t_stat:8.2f} {p_val:10.2e} {stars}")

    # =========================================================
    # STEP 10: Plots
    # =========================================================
    print("\n" + "=" * 70)
    print("STEP 10: Generating plots...")
    print("=" * 70)

    # Plot 1: Feature importance top 30
    fig, ax = plt.subplots(figsize=(12, 10))
    top_n = 30
    top_idx = sorted_idx[:top_n]
    y_pos = np.arange(top_n)
    ax.barh(y_pos, importances[top_idx], align='center', color='steelblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feat_cols[i] for i in top_idx], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Gini Importance')
    ax.set_title(f'Top {top_n} BB Features (RF Gini Importance)')
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "01_feature_importance_gini.png"), dpi=150)
    plt.close()

    # Plot 2: Category importance
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    cat_names = [c[0] for c in cat_importance]
    gini_vals = [c[1] for c in cat_importance]
    perm_vals = [c[2] for c in cat_importance]

    ax1.barh(range(len(cat_names)), gini_vals, color='steelblue')
    ax1.set_yticks(range(len(cat_names)))
    ax1.set_yticklabels(cat_names, fontsize=8)
    ax1.invert_yaxis()
    ax1.set_xlabel('Sum Gini Importance')
    ax1.set_title('Gini by Category')

    ax2.barh(range(len(cat_names)), perm_vals, color='coral')
    ax2.set_yticks(range(len(cat_names)))
    ax2.set_yticklabels(cat_names, fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel('Sum Permutation Importance')
    ax2.set_title('Permutation by Category')

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "02_category_importance.png"), dpi=150)
    plt.close()

    # Plot 3: BB spectra per type (normalized at 210nm)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    wl_plot = WL_ALL

    type_colors = {
        "PTLL": "blue", "PTT": "red", "FR": "green", "LAB": "cyan",
        "LAB_T": "magenta", "EP": "orange", "SP": "purple", "POST_O3": "brown",
        "EX": "olive", "HA": "darkred", "FA": "darkblue",
        "NO3": "black", "Br": "gray", "CaCO3": "pink",
    }

    # Water types
    ax = axes[0, 0]
    for gtype in ["PTLL", "PTT", "FR", "LAB", "LAB_T"]:
        gdf = df[df["_group"] == gtype]
        if len(gdf) == 0:
            continue
        spec_cols = [f"bb210_{w}" for w in wl_plot]
        valid_cols = [c for c in spec_cols if c in gdf.columns]
        if not valid_cols:
            continue
        mean_spec = gdf[valid_cols].mean().values
        ax.plot(wl_plot[:len(mean_spec)], mean_spec,
                label=f"{gtype} (n={len(gdf)})", color=type_colors.get(gtype, "gray"))
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Absorbance (norm. at 210nm)")
    ax.set_title("BB Spectra — Water Types (norm. 210nm)")
    ax.legend(fontsize=7)
    ax.set_xlim(200, 400)
    ax.grid(True, alpha=0.3)

    # Treatment stages
    ax = axes[0, 1]
    for gtype in ["EP", "SP", "POST_O3", "FR"]:
        gdf = df[df["_group"] == gtype]
        if len(gdf) == 0:
            continue
        spec_cols = [f"bb210_{w}" for w in wl_plot]
        valid_cols = [c for c in spec_cols if c in gdf.columns]
        if not valid_cols:
            continue
        mean_spec = gdf[valid_cols].mean().values
        ax.plot(wl_plot[:len(mean_spec)], mean_spec,
                label=f"{gtype} (n={len(gdf)})", color=type_colors.get(gtype, "gray"))
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Absorbance (norm. at 210nm)")
    ax.set_title("BB Spectra — Treatment Stages (norm. 210nm)")
    ax.legend(fontsize=7)
    ax.set_xlim(200, 400)
    ax.grid(True, alpha=0.3)

    # Inorganics vs water
    ax = axes[1, 0]
    for gtype in ["NO3", "Br", "CaCO3", "Na2CO3", "PTLL", "PTT"]:
        gdf = df[df["_group"] == gtype]
        if len(gdf) == 0:
            continue
        spec_cols = [f"bb210_{w}" for w in wl_plot]
        valid_cols = [c for c in spec_cols if c in gdf.columns]
        if not valid_cols:
            continue
        mean_spec = gdf[valid_cols].mean().values
        ls = "--" if gtype in ["NO3", "Br", "CaCO3", "Na2CO3"] else "-"
        ax.plot(wl_plot[:len(mean_spec)], mean_spec,
                label=f"{gtype} (n={len(gdf)})", color=type_colors.get(gtype, "gray"),
                linestyle=ls, linewidth=2 if ls == "--" else 1)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Absorbance (norm. at 210nm)")
    ax.set_title("BB Spectra — Inorganics vs Water (norm. 210nm)")
    ax.legend(fontsize=7)
    ax.set_xlim(200, 400)
    ax.grid(True, alpha=0.3)

    # HA/FA vs water
    ax = axes[1, 1]
    for gtype in ["HA", "FA", "PTLL", "PTT", "FR"]:
        gdf = df[df["_group"] == gtype]
        if len(gdf) == 0:
            continue
        spec_cols = [f"bb210_{w}" for w in wl_plot]
        valid_cols = [c for c in spec_cols if c in gdf.columns]
        if not valid_cols:
            continue
        mean_spec = gdf[valid_cols].mean().values
        ls = "--" if gtype in ["HA", "FA"] else "-"
        ax.plot(wl_plot[:len(mean_spec)], mean_spec,
                label=f"{gtype} (n={len(gdf)})", color=type_colors.get(gtype, "gray"),
                linestyle=ls, linewidth=2 if ls == "--" else 1)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Absorbance (norm. at 210nm)")
    ax.set_title("BB Spectra — HA/FA vs Water (norm. 210nm)")
    ax.legend(fontsize=7)
    ax.set_xlim(200, 400)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "03_bb_spectra_comparison.png"), dpi=150)
    plt.close()

    # Plot 4: PTLL vs PTT key features boxplot
    if len(ptll_df) > 0 and len(ptt_df) > 0:
        top_sig = significant[:12] if len(significant) >= 12 else significant
        if top_sig:
            n_plots = len(top_sig)
            ncols = 4
            nrows = (n_plots + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4*nrows))
            axes = axes.flatten() if nrows > 1 else [axes] if ncols == 1 else axes.flatten()

            for i, (col, m1, m2, ratio, t_stat, p_val) in enumerate(top_sig):
                ax = axes[i]
                bp = ax.boxplot([ptll_df[col].values, ptt_df[col].values],
                               labels=["PTLL", "PTT"], patch_artist=True)
                bp['boxes'][0].set_facecolor('lightblue')
                bp['boxes'][1].set_facecolor('lightsalmon')
                short_name = col.replace("bb_", "").replace("inorg_", "I_")
                ax.set_title(f"{short_name}\np={p_val:.2e}", fontsize=9)
                ax.grid(True, alpha=0.3)

            for j in range(i+1, len(axes)):
                axes[j].set_visible(False)

            plt.suptitle("PTLL vs PTT: Top significant BB features", fontsize=13)
            plt.tight_layout()
            fig.savefig(os.path.join(OUTPUT_DIR, "04_ptll_vs_ptt_boxplots.png"), dpi=150)
            plt.close()

    # Plot 5: Confusion matrix heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color=color, fontsize=9)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix (CV accuracy: {accuracy:.1%})')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "05_confusion_matrix.png"), dpi=150)
    plt.close()

    print(f"\n  All plots saved to {OUTPUT_DIR}/")

    # =========================================================
    # STEP 11: PCA + LDA (same methodology as HS)
    # =========================================================
    print("\n" + "=" * 70)
    print("STEP 11: PCA + LDA (mateixa metodologia que HS)")
    print("=" * 70)

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import LeaveOneOut

    # PCA: reduce dimensionality
    n_components_pca = min(X_scaled.shape[0] - 1, X_scaled.shape[1], 50)
    pca = PCA(n_components=n_components_pca, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # Explained variance
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    n_95 = np.searchsorted(cum_var, 0.95) + 1
    n_99 = np.searchsorted(cum_var, 0.99) + 1
    print(f"\n  PCA: {n_components_pca} components")
    print(f"    95% variance: {n_95} PCs")
    print(f"    99% variance: {n_99} PCs")
    print(f"    Top 5 PCs explain: {cum_var[4]:.1%}")

    # Show top feature loadings per PC
    print(f"\n  Top 3 feature loadings per PC (first 5 PCs):")
    for pc_i in range(min(5, n_components_pca)):
        loadings = pca.components_[pc_i]
        top3 = np.argsort(np.abs(loadings))[::-1][:3]
        items = ", ".join(f"{feat_cols[j]}({loadings[j]:+.3f})" for j in top3)
        print(f"    PC{pc_i+1} ({pca.explained_variance_ratio_[pc_i]:.1%}): {items}")

    # LDA on PCA components — use n_99 PCs for stability
    n_pcs_lda = min(n_99, X_pca.shape[1])
    X_pca_lda = X_pca[:, :n_pcs_lda]

    n_classes = len(set(y_rf))
    n_ld = min(n_classes - 1, n_pcs_lda)  # max LDA components = n_classes - 1

    lda = LinearDiscriminantAnalysis(n_components=n_ld)
    X_lda = lda.fit_transform(X_pca_lda, y_rf)

    print(f"\n  LDA: {n_pcs_lda} PCs -> {n_ld} LDs")
    print(f"    LDA explained variance ratios: {', '.join(f'{r:.1%}' for r in lda.explained_variance_ratio_[:5])}")

    # LOO cross-validation
    print(f"\n  LOO Cross-Validation...")
    loo = LeaveOneOut()
    y_loo_pred = []
    y_loo_true = []

    for train_idx, test_idx in loo.split(X_pca_lda):
        lda_cv = LinearDiscriminantAnalysis(n_components=n_ld)
        lda_cv.fit(X_pca_lda[train_idx], y_rf[train_idx])
        pred = lda_cv.predict(X_pca_lda[test_idx])
        y_loo_pred.append(pred[0])
        y_loo_true.append(y_rf[test_idx[0]])

    y_loo_pred = np.array(y_loo_pred)
    y_loo_true = np.array(y_loo_true)
    loo_accuracy = np.mean(y_loo_pred == y_loo_true)

    print(f"    LOO accuracy: {loo_accuracy:.1%} ({np.sum(y_loo_pred == y_loo_true)}/{len(y_loo_true)})")

    # Misclassifications
    misclass = y_loo_pred != y_loo_true
    if np.any(misclass):
        print(f"\n    Misclassifications ({np.sum(misclass)}):")
        mis_idx = np.where(misclass)[0]
        for idx in mis_idx:
            print(f"      {df_rf.iloc[idx]['_file']:30s} True={y_loo_true[idx]:10s} Pred={y_loo_pred[idx]:10s}")

    # LDA confusion matrix
    cm_lda = confusion_matrix(y_loo_true, y_loo_pred, labels=labels)
    print(f"\n    LDA confusion matrix (LOO):")
    print(f"    {'':15s} " + " ".join(f"{l:>8s}" for l in labels))
    for i, row in enumerate(cm_lda):
        mismark = " <--" if np.sum(row) != row[i] else ""
        print(f"    {labels[i]:15s} " + " ".join(f"{v:8d}" for v in row) + mismark)

    # Feature importance via LDA coefficients (back-project to original features)
    # LDA coef in PCA space -> project to original feature space
    # lda.coef_ shape: (n_classes, n_pcs_lda) for multi-class
    lda_coef_pca = lda.coef_  # shape (n_classes, n_pcs_lda)
    pca_components = pca.components_[:n_pcs_lda]  # shape (n_pcs_lda, n_features)

    # Back-project each class's LDA coefficients to original features
    lda_coef_orig = lda_coef_pca @ pca_components  # shape (n_classes, n_features)

    # Aggregate importance: mean absolute coefficient across all class discriminants
    lda_feat_importance = np.mean(np.abs(lda_coef_orig), axis=0)
    lda_sorted_idx = np.argsort(lda_feat_importance)[::-1]

    print(f"\n  Top 30 features (LDA back-projected importance):")
    print(f"  {'Rank':>4s} {'Feature':40s} {'LDA_imp':>10s} {'RF_Gini':>10s}")
    print(f"  {'-'*4} {'-'*40} {'-'*10} {'-'*10}")
    for rank, idx in enumerate(lda_sorted_idx[:30], 1):
        print(f"  {rank:4d} {feat_cols[idx]:40s} {lda_feat_importance[idx]:10.4f} {importances[idx]:10.4f}")

    # LDA category importance
    print(f"\n  LDA importance by feature category:")
    print(f"  {'Category':35s} {'N_feats':>7s} {'Sum_LDA':>10s} {'Max_LDA':>10s}")
    print(f"  {'-'*35} {'-'*7} {'-'*10} {'-'*10}")
    for cat, cols in categories.items():
        if not cols:
            continue
        idxs = [feat_cols.index(c) for c in cols if c in feat_cols]
        lda_sum = sum(lda_feat_importance[i] for i in idxs)
        lda_max = max(lda_feat_importance[i] for i in idxs) if idxs else 0
        print(f"  {cat:35s} {len(idxs):7d} {lda_sum:10.4f} {lda_max:10.4f}")

    # =========================================================
    # STEP 12: LDA scatter plots
    # =========================================================
    print("\n" + "=" * 70)
    print("STEP 12: LDA scatter plots")
    print("=" * 70)

    # Plot 6: LD1 vs LD2
    if n_ld >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        # LD1 vs LD2 — all classes
        ax = axes[0]
        for gtype in labels:
            mask_g = y_rf == gtype
            color = type_colors.get(gtype, "gray")
            marker = "s" if gtype in ["NO3", "Br", "CaCO3", "Na2CO3"] else \
                     "^" if gtype in ["HA", "FA"] else "o"
            ax.scatter(X_lda[mask_g, 0], X_lda[mask_g, 1],
                      c=color, marker=marker, s=40, alpha=0.7,
                      label=f"{gtype} (n={np.sum(mask_g)})", edgecolors='k', linewidth=0.3)
        ax.set_xlabel(f"LD1 ({lda.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"LD2 ({lda.explained_variance_ratio_[1]:.1%})")
        ax.set_title("LDA: LD1 vs LD2 (all classes)")
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)

        # LD1 vs LD2 — water types only (zoomed)
        ax = axes[1]
        water_in_labels = [g for g in ["FR", "EX", "LAB", "LAB_T", "PTLL", "POST_O3"] if g in labels]
        for gtype in water_in_labels:
            mask_g = y_rf == gtype
            color = type_colors.get(gtype, "gray")
            ax.scatter(X_lda[mask_g, 0], X_lda[mask_g, 1],
                      c=color, marker='o', s=50, alpha=0.7,
                      label=f"{gtype} (n={np.sum(mask_g)})", edgecolors='k', linewidth=0.3)
        ax.set_xlabel(f"LD1 ({lda.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"LD2 ({lda.explained_variance_ratio_[1]:.1%})")
        ax.set_title("LDA: LD1 vs LD2 (water types only)")
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "06_lda_scatter_ld1_ld2.png"), dpi=150)
        plt.close()

    # Plot 7: LD2 vs LD3 (if available)
    if n_ld >= 3:
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        ax = axes[0]
        for gtype in labels:
            mask_g = y_rf == gtype
            color = type_colors.get(gtype, "gray")
            marker = "s" if gtype in ["NO3", "Br", "CaCO3", "Na2CO3"] else \
                     "^" if gtype in ["HA", "FA"] else "o"
            ax.scatter(X_lda[mask_g, 1], X_lda[mask_g, 2],
                      c=color, marker=marker, s=40, alpha=0.7,
                      label=f"{gtype} (n={np.sum(mask_g)})", edgecolors='k', linewidth=0.3)
        ax.set_xlabel(f"LD2 ({lda.explained_variance_ratio_[1]:.1%})")
        ax.set_ylabel(f"LD3 ({lda.explained_variance_ratio_[2]:.1%})")
        ax.set_title("LDA: LD2 vs LD3 (all classes)")
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        for gtype in water_in_labels:
            mask_g = y_rf == gtype
            color = type_colors.get(gtype, "gray")
            ax.scatter(X_lda[mask_g, 1], X_lda[mask_g, 2],
                      c=color, marker='o', s=50, alpha=0.7,
                      label=f"{gtype} (n={np.sum(mask_g)})", edgecolors='k', linewidth=0.3)
        ax.set_xlabel(f"LD2 ({lda.explained_variance_ratio_[1]:.1%})")
        ax.set_ylabel(f"LD3 ({lda.explained_variance_ratio_[2]:.1%})")
        ax.set_title("LDA: LD2 vs LD3 (water types)")
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "07_lda_scatter_ld2_ld3.png"), dpi=150)
        plt.close()

    # Plot 8: PCA variance explained
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.bar(range(1, min(21, n_components_pca+1)),
            pca.explained_variance_ratio_[:20], color='steelblue')
    ax1.set_xlabel("PC")
    ax1.set_ylabel("Explained Variance Ratio")
    ax1.set_title("PCA: Individual Variance per PC")
    ax1.grid(True, alpha=0.3)

    ax2.plot(range(1, n_components_pca+1), cum_var, 'b-o', markersize=3)
    ax2.axhline(y=0.95, color='r', linestyle='--', label='95%')
    ax2.axhline(y=0.99, color='g', linestyle='--', label='99%')
    ax2.axvline(x=n_95, color='r', linestyle=':', alpha=0.5)
    ax2.axvline(x=n_99, color='g', linestyle=':', alpha=0.5)
    ax2.set_xlabel("Number of PCs")
    ax2.set_ylabel("Cumulative Variance")
    ax2.set_title(f"PCA: Cumulative Variance (95%={n_95} PCs, 99%={n_99} PCs)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "08_pca_variance.png"), dpi=150)
    plt.close()

    # Plot 9: LDA feature importance comparison with RF
    fig, ax = plt.subplots(figsize=(14, 10))
    top_n_lda = 30
    top_lda_idx = lda_sorted_idx[:top_n_lda]
    y_pos = np.arange(top_n_lda)

    width = 0.35
    ax.barh(y_pos - width/2, lda_feat_importance[top_lda_idx], width,
            label='LDA importance', color='coral')
    # Normalize RF to same scale for comparison
    rf_norm = importances / importances.max() * lda_feat_importance[top_lda_idx[0]]
    ax.barh(y_pos + width/2, rf_norm[top_lda_idx], width,
            label='RF Gini (normalized)', color='steelblue', alpha=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([feat_cols[i] for i in top_lda_idx], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.set_title('Top 30 Features: LDA vs RF Gini')
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "09_lda_vs_rf_importance.png"), dpi=150)
    plt.close()

    # Plot 10: LDA category importance
    fig, ax = plt.subplots(figsize=(10, 6))
    cat_lda_imp = []
    for cat, cols in categories.items():
        if not cols:
            continue
        idxs = [feat_cols.index(c) for c in cols if c in feat_cols]
        lda_sum = sum(lda_feat_importance[i] for i in idxs)
        cat_lda_imp.append((cat, lda_sum))

    cat_lda_imp.sort(key=lambda x: x[1], reverse=True)
    cat_names_lda = [c[0] for c in cat_lda_imp]
    cat_vals_lda = [c[1] for c in cat_lda_imp]

    ax.barh(range(len(cat_names_lda)), cat_vals_lda, color='coral')
    ax.set_yticks(range(len(cat_names_lda)))
    ax.set_yticklabels(cat_names_lda, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Sum LDA Importance')
    ax.set_title('LDA: Feature Category Importance')
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "10_lda_category_importance.png"), dpi=150)
    plt.close()

    print(f"\n  LDA plots saved to {OUTPUT_DIR}/")

    # =========================================================
    # SUMMARY
    # =========================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Total samples: {len(records)}")
    print(f"  Features extracted: {len(feat_cols)}")
    print(f"  RF classes: {len(set(y_rf))}")
    print(f"  CV accuracy: {accuracy:.1%}")
    print(f"  Top 5 RF features:")
    for rank, idx in enumerate(sorted_idx[:5], 1):
        print(f"    {rank}. {feat_cols[idx]} (Gini={importances[idx]:.4f})")
    print(f"\n  PCA+LDA:")
    print(f"    PCs used: {n_pcs_lda} (99% var), LDs: {n_ld}")
    print(f"    LOO accuracy: {loo_accuracy:.1%}")
    print(f"  Top 5 LDA features:")
    for rank, idx in enumerate(lda_sorted_idx[:5], 1):
        print(f"    {rank}. {feat_cols[idx]} (LDA={lda_feat_importance[idx]:.4f})")
    print(f"\n  Outputs saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
