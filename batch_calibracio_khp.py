# -*- coding: utf-8 -*-
"""
Batch Calibració KHP — totes les seqüències de Dades3.

Utilitza el pipeline complet de la Suite (import + analyze) per obtenir
àrees DOC amb baseline correction, detecció d'anomalies i SNR.

Genera:
  - Excel amb totes les injeccions KHP i les seves àrees
  - Gràfic μg C vs Àrea DOC amb regressió (COLUMN i BP separats)
  - Resum estadístic per concentració

Ús:
    python batch_calibracio_khp.py
"""

import os, sys, warnings, time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import io, contextlib

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(encoding="utf-8")

from hpsec_import import import_sequence, extract_khp_conc
from hpsec_analyze import analyze_sequence


def _quiet_import(seq_path):
    """import_sequence sense DEBUG prints."""
    with contextlib.redirect_stdout(io.StringIO()):
        return import_sequence(seq_path)


def _quiet_analyze(imported, **kwargs):
    """analyze_sequence sense DEBUG prints."""
    with contextlib.redirect_stdout(io.StringIO()):
        return analyze_sequence(imported, **kwargs)

DATA_FOLDER = "C:/Users/Lequia/Desktop/Dades3"
OUTPUT_DIR = "C:/Users/Lequia/Desktop/HPSEC/analisi_khp"


# =============================================================================
# MAIN
# =============================================================================
print("=" * 70)
print("BATCH CALIBRACIÓ KHP — Pipeline Suite (Dades3)")
print("=" * 70)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Trobar totes les seqüències
sequences = sorted(Path(DATA_FOLDER).iterdir())
sequences = [s for s in sequences if s.is_dir() and "_SEQ" in s.name.upper()]
print(f"  Seqüències trobades: {len(sequences)}")

all_khp = []
seq_errors = []
t0 = time.time()

for i, seq_dir in enumerate(sequences):
    seq_name = seq_dir.name
    seq_path = str(seq_dir)
    is_bp = "_BP" in seq_name.upper()
    method = "BP" if is_bp else "COLUMN"

    # Verificar que té MasterFile
    mf_files = [f for f in seq_dir.glob("*MasterFile*.xlsx")
                if "backup" not in f.name.lower()]
    if not mf_files:
        continue

    # 1. Import
    try:
        imported = _quiet_import(seq_path)
    except Exception as e:
        seq_errors.append((seq_name, f"Import: {e}"))
        continue

    if not imported.get("success"):
        errs = imported.get("errors", [])
        if errs:
            seq_errors.append((seq_name, f"Import: {errs[0]}"))
        continue

    # Té KHP?
    has_khp = any(
        sdata["type"] == "KHP"
        for sdata in imported["samples"].values()
    )
    if not has_khp:
        continue

    # 2. Analyze
    try:
        analysis = _quiet_analyze(imported, calibration_data=None, config=None)
    except Exception as e:
        seq_errors.append((seq_name, f"Analyze: {e}"))
        continue

    if not analysis.get("success"):
        continue

    # 3. Extreure resultats KHP
    n_khp = 0
    inj_volume = None
    # Obtenir volum d'injecció del MasterFile
    master_info = imported.get("master_data", {}).get("info", {})
    for key, val in master_info.items():
        if "volume" in str(key).lower() or "vol" in str(key).lower():
            try:
                inj_volume = float(val)
            except (ValueError, TypeError):
                pass
            break
    if inj_volume is None:
        inj_volume = 100 if is_bp else 400

    for khp in analysis["khp_samples"]:
        name = khp.get("name", "?")
        replica = khp.get("replica", "?")
        processed = khp.get("processed", False)

        conc_ppm = extract_khp_conc(name)

        # μg C = ppm × volume_uL / 1000
        ug_c = conc_ppm * inj_volume / 1000.0 if conc_ppm > 0 else None

        # Àrees
        areas = khp.get("areas", {}) if processed else {}
        area_doc = areas.get("DOC", {}).get("total") if areas else None

        # Fraccions DOC
        doc_fracs = areas.get("DOC", {}) if areas else {}

        # SNR
        snr_info = khp.get("snr_info", {})
        snr = snr_info.get("snr_direct", snr_info.get("snr"))

        # Pic
        peak = khp.get("peak_info", {}) or {}
        t_peak = peak.get("t_peak")
        peak_valid = peak.get("valid", False)

        # Anomalies
        anomalies = khp.get("anomalies", [])
        anomaly_codes = [a.get("code", "?") for a in anomalies]

        # Error
        error = khp.get("error_message", khp.get("error"))

        entry = {
            "SEQ": seq_name,
            "Method": method,
            "Sample_Name": name,
            "Replica": replica,
            "Conc_ppm": conc_ppm,
            "Volume_uL": inj_volume,
            "ug_C": ug_c,
            "Area_DOC": area_doc,
            "Area_HS": doc_fracs.get("HS"),
            "Area_BB": doc_fracs.get("BB"),
            "Area_LMW": doc_fracs.get("LMW"),
            "SNR": snr,
            "Peak_valid": peak_valid,
            "t_peak": t_peak,
            "Anomalies": ", ".join(anomaly_codes) if anomaly_codes else "",
            "Processed": processed,
            "Error": error or "",
        }
        all_khp.append(entry)
        if conc_ppm > 0:
            n_khp += 1

    if n_khp > 0:
        elapsed = time.time() - t0
        print(f"  [{i+1:3d}/{len(sequences)}] {seq_name:25s}  {n_khp:2d} KHP  "
              f"vol={inj_volume:.0f}μL  {method:6s}  ({elapsed:.0f}s)",
              flush=True)

elapsed = time.time() - t0
print(f"\n  Total: {len(all_khp)} injeccions KHP en {elapsed:.0f}s")

# =============================================================================
# DATAFRAME
# =============================================================================
df = pd.DataFrame(all_khp)

if len(df) == 0:
    print("ERROR: Cap injecció KHP trobada!")
    sys.exit(1)

# Filtrar: conc > 0 i processades
df_valid = df[(df["Conc_ppm"] > 0) & (df["Processed"] == True)].copy()
n_total = len(df)
n_valid = len(df_valid)
n_with_area = df_valid["Area_DOC"].notna().sum()
n_anomaly = (df_valid["Anomalies"] != "").sum()

print(f"\n  Conc > 0 i processades: {n_valid}")
print(f"  Amb àrea DOC: {n_with_area}")
print(f"  Amb anomalies: {n_anomaly}")

# Separar COLUMN i BP
df_col = df_valid[df_valid["Method"] == "COLUMN"].copy()
df_bp = df_valid[df_valid["Method"] == "BP"].copy()
print(f"  COLUMN: {len(df_col)}  |  BP: {len(df_bp)}")

# =============================================================================
# GUARDAR EXCEL
# =============================================================================
excel_path = os.path.join(OUTPUT_DIR, "khp_batch_calibracio.xlsx")

with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    # Totes les dades
    df.to_excel(writer, sheet_name="ALL", index=False)

    # Només vàlides
    df_valid.to_excel(writer, sheet_name="VALID", index=False)

    # Resum per concentració COLUMN
    if len(df_col) > 0:
        summary_col = []
        for conc in sorted(df_col["Conc_ppm"].unique()):
            subset = df_col[df_col["Conc_ppm"] == conc].dropna(subset=["Area_DOC"])
            # Sense anomalies
            clean = subset[subset["Anomalies"] == ""]
            summary_col.append({
                "Conc_ppm": conc,
                "ug_C": subset["ug_C"].iloc[0] if len(subset) > 0 else None,
                "n_total": len(subset),
                "n_clean": len(clean),
                "Area_mean_all": subset["Area_DOC"].mean() if len(subset) > 0 else None,
                "Area_std_all": subset["Area_DOC"].std() if len(subset) > 1 else 0,
                "Area_mean_clean": clean["Area_DOC"].mean() if len(clean) > 0 else None,
                "Area_std_clean": clean["Area_DOC"].std() if len(clean) > 1 else 0,
                "CV_all": (subset["Area_DOC"].std() / subset["Area_DOC"].mean() * 100
                           if len(subset) > 1 and subset["Area_DOC"].mean() > 0 else 0),
                "CV_clean": (clean["Area_DOC"].std() / clean["Area_DOC"].mean() * 100
                             if len(clean) > 1 and clean["Area_DOC"].mean() > 0 else 0),
                "SNR_mean": subset["SNR"].mean() if len(subset) > 0 else None,
            })
        pd.DataFrame(summary_col).to_excel(writer, sheet_name="SUMMARY_COLUMN", index=False)

    # Resum per concentració BP
    if len(df_bp) > 0:
        summary_bp = []
        for conc in sorted(df_bp["Conc_ppm"].unique()):
            subset = df_bp[df_bp["Conc_ppm"] == conc].dropna(subset=["Area_DOC"])
            clean = subset[subset["Anomalies"] == ""]
            summary_bp.append({
                "Conc_ppm": conc,
                "ug_C": subset["ug_C"].iloc[0] if len(subset) > 0 else None,
                "n_total": len(subset),
                "n_clean": len(clean),
                "Area_mean_all": subset["Area_DOC"].mean() if len(subset) > 0 else None,
                "Area_std_all": subset["Area_DOC"].std() if len(subset) > 1 else 0,
                "Area_mean_clean": clean["Area_DOC"].mean() if len(clean) > 0 else None,
                "Area_std_clean": clean["Area_DOC"].std() if len(clean) > 1 else 0,
                "CV_all": (subset["Area_DOC"].std() / subset["Area_DOC"].mean() * 100
                           if len(subset) > 1 and subset["Area_DOC"].mean() > 0 else 0),
                "CV_clean": (clean["Area_DOC"].std() / clean["Area_DOC"].mean() * 100
                             if len(clean) > 1 and clean["Area_DOC"].mean() > 0 else 0),
                "SNR_mean": subset["SNR"].mean() if len(subset) > 0 else None,
            })
        pd.DataFrame(summary_bp).to_excel(writer, sheet_name="SUMMARY_BP", index=False)

print(f"\n  Excel: {excel_path}")


# =============================================================================
# GRÀFIC: μg C vs Àrea DOC — COLUMN (tots vs clean)
# =============================================================================
from scipy.stats import linregress

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("Batch Calibració KHP — Pipeline Suite", fontsize=13, fontweight="bold")

for ax, df_method, method_name in [
    (axes[0], df_col, "COLUMN"),
    (axes[1], df_bp, "BP"),
]:
    df_area = df_method.dropna(subset=["Area_DOC", "ug_C"])
    if len(df_area) == 0:
        ax.set_title(f"{method_name} — Sense dades")
        continue

    # Separar clean vs anomaly
    df_clean = df_area[df_area["Anomalies"] == ""]
    df_anom = df_area[df_area["Anomalies"] != ""]

    # Scatter clean
    ax.scatter(df_clean["ug_C"], df_clean["Area_DOC"],
               c="steelblue", s=25, alpha=0.6, label=f"Clean (n={len(df_clean)})", zorder=3)

    # Scatter anomaly
    if len(df_anom) > 0:
        ax.scatter(df_anom["ug_C"], df_anom["Area_DOC"],
                   c="tomato", s=25, alpha=0.4, marker="x",
                   label=f"Anomaly (n={len(df_anom)})", zorder=2)

    # Regressió amb clean
    if len(df_clean) > 2:
        x = df_clean["ug_C"].values
        y = df_clean["Area_DOC"].values

        # Regressió lliure
        slope, intercept, r_value, _, _ = linregress(x, y)
        x_fit = np.linspace(0, max(x) * 1.1, 100)
        ax.plot(x_fit, slope * x_fit + intercept, "r-", linewidth=1.5, alpha=0.8,
                label=f"y = {slope:.1f}x + {intercept:.1f}\nR² = {r_value**2:.4f}")

        # Regressió per origen
        slope_o = np.sum(x * y) / np.sum(x * x)
        ss_res = np.sum((y - slope_o * x) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2_o = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        ax.plot(x_fit, slope_o * x_fit, "b--", linewidth=1, alpha=0.6,
                label=f"y = {slope_o:.1f}x (origen)\nR² = {r2_o:.4f}")

    # Medians per concentració (clean)
    if len(df_clean) > 0:
        medians = df_clean.groupby("Conc_ppm").agg(
            ug_C=("ug_C", "first"),
            Area_DOC=("Area_DOC", "median"),
        ).reset_index()
        ax.scatter(medians["ug_C"], medians["Area_DOC"],
                   c="navy", s=80, marker="D", zorder=5, edgecolors="white", linewidth=0.5,
                   label=f"Medians (n={len(medians)})")

        # Regressió medians
        if len(medians) > 2:
            xm, ym = medians["ug_C"].values, medians["Area_DOC"].values
            sl_m, int_m, r_m, _, _ = linregress(xm, ym)
            ax.plot(x_fit, sl_m * x_fit + int_m, "g-", linewidth=1.5, alpha=0.7,
                    label=f"Medians: R² = {r_m**2:.4f}")

    ax.set_xlabel("μg C injectats")
    ax.set_ylabel("Àrea DOC (ppb·min)")
    ax.set_title(f"KHP — {method_name}")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "khp_batch_calibracio.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Gràfic: {plot_path}")


# =============================================================================
# RESUM CONSOLA
# =============================================================================
for method_name, df_method in [("COLUMN", df_col), ("BP", df_bp)]:
    df_area = df_method.dropna(subset=["Area_DOC", "ug_C"])
    df_clean = df_area[df_area["Anomalies"] == ""]

    if len(df_clean) == 0:
        continue

    print(f"\n{'=' * 70}")
    print(f"RESUM {method_name} — Clean (sense anomalies)")
    print(f"{'=' * 70}")
    print(f"  {'ppm':>6s}  {'μg C':>7s}  {'n':>4s}  {'Àrea med':>10s}  {'Àrea std':>10s}  "
          f"{'CV%':>6s}  {'SNR med':>8s}")
    print("-" * 70)

    for conc in sorted(df_clean["Conc_ppm"].unique()):
        subset = df_clean[df_clean["Conc_ppm"] == conc]
        ug = subset["ug_C"].iloc[0]
        area_mean = subset["Area_DOC"].mean()
        area_std = subset["Area_DOC"].std() if len(subset) > 1 else 0
        cv = area_std / area_mean * 100 if area_mean > 0 else 0
        snr_mean = subset["SNR"].mean()
        print(f"  {conc:6.3f}  {ug:7.3f}  {len(subset):4d}  {area_mean:10.1f}  "
              f"{area_std:10.1f}  {cv:6.1f}  {snr_mean:8.1f}")

    x = df_clean["ug_C"].values
    y = df_clean["Area_DOC"].values
    if len(x) > 2:
        sl, intc, r, _, _ = linregress(x, y)
        print(f"\n  Regressió: y = {sl:.1f}x + {intc:.1f}   R² = {r**2:.4f}  (n={len(x)})")

        sl_o = np.sum(x * y) / np.sum(x * x)
        ss_res = np.sum((y - sl_o * x) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2_o = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        print(f"  Per origen: y = {sl_o:.1f}x          R² = {r2_o:.4f}")

        # RF equivalent: slope = area/μgC; rf_mass_cal = area*1000/(ppm*vol)
        # area = rf_mass_cal * ppm * vol / 1000
        # area/μgC = area / (ppm * vol / 1000) = rf_mass_cal
        # Equivalent: slope (area/μgC) = rf_mass_cal
        print(f"\n  RF equivalent (slope): {sl_o:.1f} ppb·min/μgC")

# Errors
if seq_errors:
    print(f"\n  Errors ({len(seq_errors)}):")
    for name, err in seq_errors[:10]:
        print(f"    {name}: {err}")
    if len(seq_errors) > 10:
        print(f"    ... i {len(seq_errors)-10} més")

print(f"\n{'=' * 70}")
print("FET!")
