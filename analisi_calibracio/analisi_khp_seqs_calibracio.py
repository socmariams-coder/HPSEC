"""
Rectes de calibracio NOMES amb SEQs dedicades
==============================================
Compara:
  G) Recta de 153_SEQ (COLUMN) + 152+156_SEQ_BP (BP)
  H) Recta unificada amb totes les SEQs cal juntes
vs els models anteriors (A actual, B per-mode, E origen)

Afegit: plot_chromatograms() — cromatogrames KHP amb arees i fit bigaussian
"""
import json
import re
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from scipy import stats
from collections import OrderedDict

# Afegir path de la suite per imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from hpsec_import import import_sequence
from hpsec_config import get_config
from hpsec_core import (detect_main_peak, detect_all_peaks, fit_bigaussian, bigaussian,
                        detect_peak_anomaly, repair_with_parabola)
from hpsec_calibrate import expand_integration_limits_to_baseline, extract_khp_conc

DADES_DIR = Path(r"C:\Users\Lequia\Desktop\Dades3")
OUTPUT_DIR = Path(r"C:\Users\Lequia\Desktop\HPSEC\analisi_calibracio")
OUTPUT_DIR.mkdir(exist_ok=True)


def extract_seq_number(name):
    m = re.match(r"(\d+)", str(name))
    return int(m.group(1)) if m else 0


def load_cal_result(seq_name):
    """Carrega calibration_result.json d'una SEQ."""
    path = DADES_DIR / seq_name / "CHECK" / "data" / "calibration_result.json"
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "calibrations" in data:
        return data["calibrations"]
    return [data] if isinstance(data, dict) else []


def entries_to_df(entries, seq_name):
    """Converteix entrades de calibracio a DataFrame."""
    rows = []
    for e in entries:
        c = e.get("conc_ppm", 0)
        if c <= 0:
            continue
        v = e.get("volume_uL", 0)
        rows.append({
            "seq_name": seq_name,
            "mode": e.get("mode", ""),
            "conc_ppm": c,
            "volume_uL": v,
            "ug_C": c * v / 1000,
            "area": e.get("area", 0),
            "snr": e.get("snr", 0),
            "rsd": e.get("rsd", 0),
            "has_batman": e.get("has_batman", False),
            "n_replicas": e.get("n_replicas", 1),
        })
    return pd.DataFrame(rows)


def calc_ppm(area, rf, intercept, volume):
    return (area - intercept) * 1000 / (rf * volume)


def fit_line(x, y):
    """Retorna slope, intercept, r2."""
    if len(x) < 2:
        return 0, 0, 0
    s, i, r, _, _ = stats.linregress(x, y)
    return s, i, r ** 2


def fit_origin(x, y):
    """Slope per origen."""
    return float(np.sum(x * y) / np.sum(x ** 2))


def main():
    print("=" * 100)
    print("RECTES DE CALIBRACIO — NOMES SEQs DEDICADES")
    print("=" * 100)

    # =====================================================================
    # 1. CARREGAR SEQS DE CALIBRACIO
    # =====================================================================
    cal_seqs = {
        # COLUMN/400
        "153_SEQ": {"expected_mode": "COLUMN", "expected_vol": 400},
        "206_SEQ": {"expected_mode": "COLUMN", "expected_vol": 400},
        # BP/100
        "152_SEQ_BP": {"expected_mode": "BP", "expected_vol": 100},
        "156_SEQ_BP": {"expected_mode": "BP", "expected_vol": 100},
    }

    all_dfs = []
    for seq_name, info in cal_seqs.items():
        entries = load_cal_result(seq_name)
        df = entries_to_df(entries, seq_name)
        # Filtrar per vol esperat (algunes seqs tenen volums mixtos)
        df = df[df["volume_uL"] == info["expected_vol"]]
        if not df.empty:
            all_dfs.append(df)
            print(f"\n  {seq_name} ({info['expected_mode']}/{info['expected_vol']}uL):")
            for _, r in df.sort_values("conc_ppm").iterrows():
                flag = " *batman*" if r["has_batman"] else ""
                print(f"    {r['conc_ppm']:6.3f} ppm -> ugC={r['ug_C']:.3f}  "
                      f"area={r['area']:8.1f}  snr={r['snr']:.0f}  rsd={r['rsd']:.1f}%{flag}")

    df_all = pd.concat(all_dfs, ignore_index=True)

    # Separar per mode
    col_cal = df_all[(df_all["mode"] == "COLUMN") & (df_all["volume_uL"] == 400)].copy()
    bp_cal = df_all[(df_all["mode"] == "BP") & (df_all["volume_uL"] == 100)].copy()

    # Filtrar batman per ser nets
    col_clean = col_cal[~col_cal["has_batman"]].copy()
    bp_clean = bp_cal[~bp_cal["has_batman"]].copy()

    # Treure 5ppm de 152_SEQ_BP (area invertida: 238 < 271 del 3ppm)
    bad_mask = (bp_clean["seq_name"] == "152_SEQ_BP") & (bp_clean["conc_ppm"] == 5.0)
    n_removed = bad_mask.sum()
    bp_clean = bp_clean[~bad_mask].copy()
    print(f"\n  Filtrat: {n_removed} punt(s) 5ppm de 152_SEQ_BP (area invertida)")

    print(f"\nPunts nets (sense batman):")
    print(f"  COLUMN: {len(col_clean)} (de {len(col_cal)})")
    print(f"  BP: {len(bp_clean)} (de {len(bp_cal)})")

    # =====================================================================
    # 2. RECTES PER MODE (nomes seqs calibracio)
    # =====================================================================
    print("\n" + "=" * 100)
    print("2. RECTES DE CALIBRACIO")
    print("=" * 100)

    # COLUMN
    s_col, i_col, r2_col = fit_line(col_clean["ug_C"], col_clean["area"])
    s_col_o = fit_origin(col_clean["ug_C"], col_clean["area"])
    ss_res = np.sum((col_clean["area"] - s_col_o * col_clean["ug_C"]) ** 2)
    ss_tot = np.sum((col_clean["area"] - col_clean["area"].mean()) ** 2)
    r2_col_o = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    print(f"\n  COLUMN (153+206, n={len(col_clean)}):")
    print(f"    Regressio:  y = {s_col:.1f}x + {i_col:.1f}  R2={r2_col:.4f}")
    print(f"    Per origen: y = {s_col_o:.1f}x         R2={r2_col_o:.4f}")

    # BP
    s_bp, i_bp, r2_bp = fit_line(bp_clean["ug_C"], bp_clean["area"])
    s_bp_o = fit_origin(bp_clean["ug_C"], bp_clean["area"])
    ss_res = np.sum((bp_clean["area"] - s_bp_o * bp_clean["ug_C"]) ** 2)
    ss_tot = np.sum((bp_clean["area"] - bp_clean["area"].mean()) ** 2)
    r2_bp_o = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    print(f"\n  BP (152+156, n={len(bp_clean)}):")
    print(f"    Regressio:  y = {s_bp:.1f}x + {i_bp:.1f}  R2={r2_bp:.4f}")
    print(f"    Per origen: y = {s_bp_o:.1f}x         R2={r2_bp_o:.4f}")

    # UNIFICADA (totes juntes)
    uni = pd.concat([col_clean, bp_clean])
    s_uni, i_uni, r2_uni = fit_line(uni["ug_C"], uni["area"])
    s_uni_o = fit_origin(uni["ug_C"], uni["area"])

    print(f"\n  UNIFICADA (COL+BP, n={len(uni)}):")
    print(f"    Regressio:  y = {s_uni:.1f}x + {i_uni:.1f}  R2={r2_uni:.4f}")
    print(f"    Per origen: y = {s_uni_o:.1f}x")

    diff_slopes = (s_bp - s_col) / s_col * 100
    print(f"\n  Diff slopes COL vs BP: {diff_slopes:+.1f}%")

    # =====================================================================
    # 3. DEFINIR MODELS
    # =====================================================================
    models = OrderedDict()
    models["A) Actual (628/81, 915/0)"] = {
        "COLUMN": {"rf": 628, "i": 81, "vol": 400},
        "BP": {"rf": 915, "i": 0, "vol": 100},
        "short": "Actual",
    }
    models["B) Per-mode CalSeq"] = {
        "COLUMN": {"rf": round(s_col, 1), "i": round(i_col, 1), "vol": 400},
        "BP": {"rf": round(s_bp, 1), "i": round(i_bp, 1), "vol": 100},
        "short": "CalSeq sep",
    }
    models["C) RF unic CalSeq + i sep"] = {
        "COLUMN": {"rf": round(s_uni, 1), "i": round(i_col, 1), "vol": 400},
        "BP": {"rf": round(s_uni, 1), "i": round(i_bp, 1), "vol": 100},
        "short": "RF unic+i sep",
    }
    models["D) RF unic CalSeq + i unic"] = {
        "COLUMN": {"rf": round(s_uni, 1), "i": round(i_uni, 1), "vol": 400},
        "BP": {"rf": round(s_uni, 1), "i": round(i_uni, 1), "vol": 100},
        "short": "Unificat",
    }
    models["E) Per-mode origen"] = {
        "COLUMN": {"rf": round(s_col_o, 1), "i": 0, "vol": 400},
        "BP": {"rf": round(s_bp_o, 1), "i": 0, "vol": 100},
        "short": "Origen sep",
    }
    models["F) RF unic origen"] = {
        "COLUMN": {"rf": round(s_uni_o, 1), "i": 0, "vol": 400},
        "BP": {"rf": round(s_uni_o, 1), "i": 0, "vol": 100},
        "short": "Origen unic",
    }

    # =====================================================================
    # 4. EVALUAR CADA MODEL CONTRA DADES CALIBRACIO
    # =====================================================================
    print("\n" + "=" * 100)
    print("3. ERRORS DE QUANTIFICACIO — Dades de calibracio")
    print("=" * 100)

    # Taula detallada
    all_data = pd.concat([col_clean, bp_clean])

    for mname, mparams in models.items():
        short = mparams["short"]
        print(f"\n  --- {short} ---")

        errors = []
        for _, row in all_data.iterrows():
            mode = row["mode"]
            p = mparams[mode]
            ppm_calc = calc_ppm(row["area"], p["rf"], p["i"], row["volume_uL"])
            err = (ppm_calc - row["conc_ppm"]) / row["conc_ppm"] * 100
            errors.append({"mode": mode, "conc": row["conc_ppm"], "ppm_calc": ppm_calc, "err_pct": err})

        df_err = pd.DataFrame(errors)
        rmse = np.sqrt(np.mean(df_err["err_pct"] ** 2))
        mae = np.mean(df_err["err_pct"].abs())
        within_10 = (df_err["err_pct"].abs() < 10).sum() / len(df_err) * 100

        # Consistencia COL-BP
        concs_common = set(df_err[df_err["mode"] == "COLUMN"]["conc"]) & set(df_err[df_err["mode"] == "BP"]["conc"])
        diffs = []
        for c in concs_common:
            ppm_c = df_err[(df_err["mode"] == "COLUMN") & (df_err["conc"] == c)]["ppm_calc"].median()
            ppm_b = df_err[(df_err["mode"] == "BP") & (df_err["conc"] == c)]["ppm_calc"].median()
            diffs.append(abs(ppm_b - ppm_c) / c * 100)
        col_bp = np.median(diffs) if diffs else float("nan")

        p_c = mparams["COLUMN"]
        p_b = mparams["BP"]
        print(f"    COL: rf={p_c['rf']}, i={p_c['i']}  |  BP: rf={p_b['rf']}, i={p_b['i']}")
        print(f"    RMSE={rmse:.1f}%  MAE={mae:.1f}%  <10%: {within_10:.0f}%  COL-BP: {col_bp:.1f}%")

        for mode in ["COLUMN", "BP"]:
            sub = df_err[df_err["mode"] == mode]
            print(f"    {mode}:")
            for _, r in sub.sort_values("conc").iterrows():
                print(f"      {r['conc']:6.3f} ppm -> {r['ppm_calc']:6.3f} ppm  ({r['err_pct']:+6.1f}%)")

    # =====================================================================
    # 5. GRAFICS
    # =====================================================================

    # --- Grafic 1: Recta de calibracio amb punts ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle("Rectes de Calibracio — Nomes SEQs dedicades", fontsize=14, fontweight="bold")

    # Rectes separades
    ax = axes[0]
    for data, label, color, marker in [
        (col_clean, "COLUMN 153+206", "#3498db", "o"),
        (bp_clean, "BP 152+156", "#e67e22", "s"),
    ]:
        ax.scatter(data["ug_C"], data["area"], c=color, marker=marker, s=80, alpha=0.7,
                   label=f"{label} (n={len(data)})")

    x_fit = np.linspace(0, col_clean["ug_C"].max() * 1.1, 100)
    ax.plot(x_fit, s_col * x_fit + i_col, "#3498db", lw=2,
            label=f"COL: {s_col:.0f}x+{i_col:.0f} R2={r2_col:.4f}")
    ax.plot(x_fit, s_bp * x_fit + i_bp, "#e67e22", lw=2,
            label=f"BP: {s_bp:.0f}x+{i_bp:.0f} R2={r2_bp:.4f}")
    ax.plot(x_fit, s_uni * x_fit + i_uni, "k--", lw=2, alpha=0.5,
            label=f"UNIF: {s_uni:.0f}x+{i_uni:.0f} R2={r2_uni:.4f}")
    # Actual
    ax.plot(x_fit, 628 * x_fit + 81, "r:", lw=2, alpha=0.4,
            label="Actual COL: 628x+81")
    ax.plot(x_fit, 915 * x_fit + 0, "r--", lw=2, alpha=0.4,
            label="Actual BP: 915x+0")

    ax.set_xlabel("ug C injectat")
    ax.set_ylabel("Area DOC (ppb*min)")
    ax.set_title("Rectes de calibracio")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Per sequencia
    ax = axes[1]
    seq_colors = {"153_SEQ": "#2980b9", "206_SEQ": "#85c1e9",
                  "152_SEQ_BP": "#d35400", "156_SEQ_BP": "#f0b27a"}
    for seq_name, color in seq_colors.items():
        info = cal_seqs.get(seq_name)
        if info is None:
            continue
        df_seq = entries_to_df(load_cal_result(seq_name), seq_name)
        df_seq = df_seq[(df_seq["volume_uL"] == info["expected_vol"]) & (df_seq["conc_ppm"] > 0)]
        df_seq["ug_C"] = df_seq["conc_ppm"] * df_seq["volume_uL"] / 1000

        batman_mask = df_seq["has_batman"]
        clean = df_seq[~batman_mask]
        batm = df_seq[batman_mask]

        ax.scatter(clean["ug_C"], clean["area"], c=color, s=80, alpha=0.8, label=seq_name)
        if not batm.empty:
            ax.scatter(batm["ug_C"], batm["area"], c=color, s=80, alpha=0.4,
                       marker="x", linewidths=2)

        # Fit individual
        if len(clean) >= 3:
            s, i, r2 = fit_line(clean["ug_C"], clean["area"])
            x_f = np.linspace(0, clean["ug_C"].max() * 1.1, 50)
            ax.plot(x_f, s * x_f + i, color=color, lw=1.5, alpha=0.6,
                    label=f"  {s:.0f}x+{i:.0f} R2={r2:.3f}")

    ax.set_xlabel("ug C injectat")
    ax.set_ylabel("Area DOC")
    ax.set_title("Per sequencia (x = batman)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Error per model (barplot)
    ax = axes[2]
    model_shorts = [models[m]["short"] for m in models]
    model_colors = ["#e74c3c", "#2ecc71", "#3498db", "#9b59b6", "#f39c12", "#1abc9c"]
    rmses = []
    col_bp_diffs = []
    within_10s = []

    for mname, mparams in models.items():
        errs = []
        for _, row in all_data.iterrows():
            p = mparams[row["mode"]]
            ppm = calc_ppm(row["area"], p["rf"], p["i"], row["volume_uL"])
            errs.append((ppm - row["conc_ppm"]) / row["conc_ppm"] * 100)
        errs = np.array(errs)
        rmses.append(np.sqrt(np.mean(errs ** 2)))
        within_10s.append((np.abs(errs) < 10).sum() / len(errs) * 100)

        diffs_m = []
        for c in set(col_clean["conc_ppm"]) & set(bp_clean["conc_ppm"]):
            sub_c = col_clean[col_clean["conc_ppm"] == c]
            sub_b = bp_clean[bp_clean["conc_ppm"] == c]
            if not sub_c.empty and not sub_b.empty:
                p_c = mparams["COLUMN"]
                p_b = mparams["BP"]
                ppm_c = calc_ppm(sub_c["area"].median(), p_c["rf"], p_c["i"], p_c["vol"])
                ppm_b = calc_ppm(sub_b["area"].median(), p_b["rf"], p_b["i"], p_b["vol"])
                diffs_m.append(abs(ppm_b - ppm_c) / c * 100)
        col_bp_diffs.append(np.median(diffs_m) if diffs_m else 100)

    x = np.arange(len(model_shorts))
    width = 0.35
    bars1 = ax.bar(x - width / 2, rmses, width, color=model_colors, alpha=0.8, label="RMSE (%)")
    bars2 = ax.bar(x + width / 2, col_bp_diffs, width, color=model_colors, alpha=0.4,
                   edgecolor=model_colors, linewidth=2, label="COL-BP diff (%)")

    for bar, v in zip(bars1, rmses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{v:.0f}", ha="center", fontsize=7)
    for bar, v in zip(bars2, col_bp_diffs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{v:.0f}", ha="center", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(model_shorts, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Error (%)")
    ax.set_title("RMSE i Consistencia COL-BP")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "khp_seqs_calibracio.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Grafic: {OUTPUT_DIR / 'khp_seqs_calibracio.png'}")

    # --- Grafic 2: ppm calculat vs real per model ---
    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    fig.suptitle("ppm calculat vs real — SEQs Calibracio", fontsize=14, fontweight="bold")

    for idx, (mname, mparams) in enumerate(models.items()):
        ax = axes[idx // 3, idx % 3]
        short = mparams["short"]

        for data, mode, color, marker in [
            (col_clean, "COLUMN", "#3498db", "o"), (bp_clean, "BP", "#e67e22", "s")
        ]:
            ppms = []
            for _, row in data.iterrows():
                p = mparams[mode]
                ppms.append(calc_ppm(row["area"], p["rf"], p["i"], row["volume_uL"]))
            ax.scatter(data["conc_ppm"], ppms, c=color, marker=marker, s=80, alpha=0.7,
                       label=f"{mode}")

        maxc = max(col_clean["conc_ppm"].max(), bp_clean["conc_ppm"].max()) * 1.1
        ax.plot([0, maxc], [0, maxc], "k--", lw=1.5, alpha=0.5, label="Ideal")
        ax.fill_between([0, maxc], [0, maxc * 0.9], [0, maxc * 1.1],
                        alpha=0.1, color="green")

        p_c = mparams["COLUMN"]
        p_b = mparams["BP"]
        ax.set_title(f"{short}\nCOL: rf={p_c['rf']}, i={p_c['i']}  |  BP: rf={p_b['rf']}, i={p_b['i']}",
                     fontsize=9, fontweight="bold")
        ax.set_xlabel("Conc real (ppm)")
        ax.set_ylabel("ppm calculat")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.3, maxc)
        ax.set_ylim(-0.3, maxc)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "khp_seqs_ppm_vs_real.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Grafic: {OUTPUT_DIR / 'khp_seqs_ppm_vs_real.png'}")

    # =====================================================================
    # 6. RESUM FINAL
    # =====================================================================
    print("\n" + "=" * 100)
    print("RESUM FINAL — Rectes de SEQs calibracio")
    print("=" * 100)
    print(f"""
  RECTES (dades netes, sense batman):
    COLUMN (153+206, n={len(col_clean)}):  y = {s_col:.1f}x + {i_col:.1f}  R2={r2_col:.4f}
    BP (152+156, n={len(bp_clean)}):       y = {s_bp:.1f}x + {i_bp:.1f}  R2={r2_bp:.4f}
    UNIFICADA (n={len(uni)}):              y = {s_uni:.1f}x + {i_uni:.1f}  R2={r2_uni:.4f}

  SLOPES (=RF):
    COLUMN: {s_col:.1f}    BP: {s_bp:.1f}    UNIF: {s_uni:.1f}
    Diff COL vs BP: {diff_slopes:+.1f}%

  INTERCEPTS:
    COLUMN: {i_col:.1f}    BP: {i_bp:.1f}    UNIF: {i_uni:.1f}

  ACTUAL vs NOU:
    COLUMN RF: 628 -> {s_col:.0f} ({(s_col-628)/628*100:+.1f}%)
    COLUMN i:  81 -> {i_col:.0f} ({i_col-81:+.0f})
    BP RF:     915 -> {s_bp:.0f} ({(s_bp-915)/915*100:+.1f}%)
    BP i:      0 -> {i_bp:.0f} ({i_bp-0:+.0f})
""")

    print("=" * 100)
    print("FET!")


def plot_chromatograms():
    """
    Plotar cromatogrames KHP de les SEQs de calibracio.
    Utilitza import_sequence() per obtenir dades crues,
    detect_main_peak() i fit_bigaussian() per analisi.

    Genera un PDF multi-pagina amb tots els cromatogrames,
    arees d'integracio marcades i fits bi-gaussian superposats.
    """
    config = get_config()

    cal_seqs = {
        "153_SEQ": {"expected_mode": "COLUMN", "expected_vol": 400},
        "206_SEQ": {"expected_mode": "COLUMN", "expected_vol": 400},
        "152_SEQ_BP": {"expected_mode": "BP", "expected_vol": 100},
        "156_SEQ_BP": {"expected_mode": "BP", "expected_vol": 100},
    }

    # Recollir totes les repliques KHP
    all_chromatograms = []

    for seq_name, info in cal_seqs.items():
        seq_path = str(DADES_DIR / seq_name)
        print(f"\n  Importando {seq_name}...")

        try:
            imported = import_sequence(seq_path, config)
        except Exception as e:
            print(f"    ERROR important {seq_name}: {e}")
            continue

        if not imported or not imported.get("success"):
            print(f"    Import fallit per {seq_name}")
            continue

        samples = imported.get("samples", {})
        khp_names = imported.get("khp_samples", [])
        method = imported.get("method", info["expected_mode"])

        print(f"    KHP trobats: {khp_names}")

        for khp_name in khp_names:
            sample = samples.get(khp_name, {})
            replicas = sample.get("replicas", {})
            conc = extract_khp_conc(khp_name)

            for rep_num, rep_data in replicas.items():
                direct = rep_data.get("direct") or {}
                t = direct.get("t")
                y_net = direct.get("y_net")

                if t is None or y_net is None:
                    continue

                t = np.asarray(t)
                y_net = np.asarray(y_net)

                # Obtenir volum d'injeccio
                inj_info = rep_data.get("injection_info", {})
                volume = inj_info.get("inj_volume", info["expected_vol"])

                is_bp = (method == "BP")
                ug_c = conc * volume / 1000

                all_chromatograms.append({
                    "seq_name": seq_name,
                    "khp_name": khp_name,
                    "replica": rep_num,
                    "conc_ppm": conc,
                    "volume_uL": volume,
                    "ug_c": ug_c,
                    "mode": method,
                    "is_bp": is_bp,
                    "t": t,
                    "y_net": y_net,
                })

    print(f"\n  Total cromatogrames: {len(all_chromatograms)}")

    if not all_chromatograms:
        print("  Cap cromatograma trobat!")
        return

    # Ordenar: primer per seq, despres per concentracio, despres replica
    all_chromatograms.sort(key=lambda c: (c["seq_name"], c["conc_ppm"], c["replica"]))

    # Generar PDF
    pdf_path = OUTPUT_DIR / "cromatogrames_khp_calibracio.pdf"
    n_per_page = 6  # 3 files x 2 columnes
    n_pages = (len(all_chromatograms) + n_per_page - 1) // n_per_page

    print(f"  Generant PDF amb {n_pages} pagines...")

    with PdfPages(str(pdf_path)) as pdf:
        for page in range(n_pages):
            fig, axes = plt.subplots(3, 2, figsize=(16, 20))
            fig.suptitle(
                f"Cromatogrames KHP — Calibracio (pag {page+1}/{n_pages})",
                fontsize=14, fontweight="bold", y=0.98
            )

            start = page * n_per_page
            end = min(start + n_per_page, len(all_chromatograms))

            for i in range(n_per_page):
                ax = axes[i // 2, i % 2]
                idx = start + i

                if idx >= end:
                    ax.axis("off")
                    continue

                chrom = all_chromatograms[idx]
                t = chrom["t"]
                y_net = chrom["y_net"]
                is_bp = chrom["is_bp"]

                # Detectar pic principal
                peak_info = detect_main_peak(t, y_net)

                if not peak_info or not peak_info.get("valid"):
                    ax.plot(t, y_net, "b-", lw=0.8, alpha=0.7)
                    ax.set_title(
                        f"{chrom['seq_name']} | {chrom['khp_name']} R{chrom['replica']}\n"
                        f"{chrom['conc_ppm']} ppm | PIC NO DETECTAT",
                        fontsize=9, fontweight="bold", color="red"
                    )
                    ax.grid(True, alpha=0.3)
                    continue

                peak_idx = peak_info["peak_idx"]
                left_idx = peak_info.get("left_idx", 0)
                right_idx = peak_info.get("right_idx", len(y_net) - 1)

                # Buscar limits en all_peaks
                all_peaks = detect_all_peaks(t, y_net)
                for pk in all_peaks:
                    if abs(pk["t"] - peak_info["t_max"]) < 0.1:
                        left_idx = pk["left_idx"]
                        right_idx = pk["right_idx"]
                        break

                # Expandir limits
                expansion = expand_integration_limits_to_baseline(
                    t, y_net, left_idx, right_idx, peak_idx,
                    baseline_threshold_pct=15,
                    min_width_minutes=1.0,
                    max_width_minutes=6.0 if is_bp else 10.0,
                    is_bp=is_bp
                )
                left_idx = expansion["left_idx"]
                right_idx = expansion["right_idx"]

                t_start = float(t[left_idx])
                t_end = float(t[right_idx])
                t_max = peak_info["t_max"]
                area_orig = float(np.trapezoid(y_net[left_idx:right_idx+1], t[left_idx:right_idx+1]))

                # Detectar batman i reparar
                t_seg = t[left_idx:right_idx+1]
                y_seg = y_net[left_idx:right_idx+1]
                anomaly = detect_peak_anomaly(t_seg, y_seg)
                has_batman = anomaly.get("is_batman", False)

                y_repaired = None
                area_repaired = None
                batman_label = ""
                if has_batman:
                    y_rep_seg, repair_info, was_repaired = repair_with_parabola(t_seg, y_seg)
                    if was_repaired:
                        y_repaired = y_net.copy()
                        y_repaired[left_idx:right_idx+1] = y_rep_seg
                        area_repaired = float(np.trapezoid(y_rep_seg, t_seg))
                        batman_label = f" BATMAN! Area orig={area_orig:.1f} -> rep={area_repaired:.1f}"
                    else:
                        batman_label = " BATMAN (no reparat)"

                # Senyal per plotting i fit
                y_plot = y_repaired if y_repaired is not None else y_net
                area_final = area_repaired if area_repaired is not None else area_orig

                # Plot cromatograma original (gris si batman reparat)
                if has_batman and y_repaired is not None:
                    ax.plot(t, y_net, color="#CCCCCC", lw=0.8, alpha=0.6, label="Original (batman)")
                    ax.plot(t, y_repaired, "b-", lw=1.0, alpha=0.8, label="Reparat")
                    # Area reparada
                    ax.fill_between(
                        t[left_idx:right_idx+1], 0, y_repaired[left_idx:right_idx+1],
                        alpha=0.25, color="#E67E22", label=f"Area rep={area_repaired:.1f}"
                    )
                    # Area original en transparencia
                    ax.fill_between(
                        t[left_idx:right_idx+1], 0, y_net[left_idx:right_idx+1],
                        alpha=0.08, color="#95A5A6"
                    )
                else:
                    ax.plot(t, y_net, "b-", lw=0.8, alpha=0.7, label="DOC net")
                    ax.fill_between(
                        t[left_idx:right_idx+1], 0, y_net[left_idx:right_idx+1],
                        alpha=0.25, color="#3498DB", label=f"Area={area_orig:.1f}"
                    )

                # Limits d'integracio (linies verticals)
                ax.axvline(t_start, color="green", ls="--", lw=0.8, alpha=0.6)
                ax.axvline(t_end, color="green", ls="--", lw=0.8, alpha=0.6)

                # Fit bi-gaussian sobre senyal (reparat si batman)
                bigauss = fit_bigaussian(t, y_plot, peak_idx, left_idx, right_idx)
                bigauss_label = ""

                if bigauss and bigauss.get("status") != "INVALID":
                    t_fit = bigauss.get("t_fit")
                    y_fit = bigauss.get("y_fit")
                    r2 = bigauss.get("r2", 0)
                    asym = bigauss.get("asymmetry", 1)

                    if t_fit is not None and y_fit is not None:
                        color_fit = "#E74C3C" if r2 < 0.95 else "#27AE60"
                        ax.plot(t_fit, y_fit, color=color_fit, ls="--", lw=1.5, alpha=0.8,
                                label=f"BiGauss R\u00b2={r2:.3f}")

                    bigauss_label = f"  R\u00b2={r2:.3f}  asym={asym:.2f}"
                    if bigauss["status"] == "CHECK":
                        bigauss_label += " !"
                else:
                    bigauss_label = "  BiGauss: INVALID"

                # Mark t_max
                y_at_peak = float(y_plot[peak_idx])
                ax.plot(t_max, y_at_peak, "rv", ms=8, alpha=0.7)

                # Configurar eix - Zoom al pic
                margin = (t_end - t_start) * 0.5
                view_left = max(t[0], t_start - margin)
                view_right = min(t[-1], t_end + margin)
                ax.set_xlim(view_left, view_right)

                y_peak = float(y_plot[peak_idx])
                ax.set_ylim(-y_peak * 0.05, y_peak * 1.25)

                ax.set_xlabel("Temps (min)", fontsize=8)
                ax.set_ylabel("DOC net (mAU)", fontsize=8)
                ax.tick_params(axis="both", labelsize=7)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=6, loc="upper right")

                # Titol amb info
                title_color = "#E74C3C" if has_batman else "black"
                area_text = f"Area={area_final:.1f}"
                if has_batman and area_repaired is not None:
                    area_text = f"Area orig={area_orig:.1f} -> rep={area_repaired:.1f} ({(area_repaired-area_orig)/area_orig*100:+.1f}%)"
                ax.set_title(
                    f"{chrom['seq_name']} | {chrom['khp_name']} R{chrom['replica']} | "
                    f"{chrom['mode']}/{chrom['volume_uL']}uL\n"
                    f"{chrom['conc_ppm']} ppm ({chrom['ug_c']:.2f} ugC) | "
                    f"t_max={t_max:.2f} | {area_text}{bigauss_label}",
                    fontsize=7.5, fontweight="bold", color=title_color
                )

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig, dpi=150)
            plt.close(fig)

    print(f"\n  PDF generat: {pdf_path}")
    print(f"  {len(all_chromatograms)} cromatogrames plotats")


def compare_batman_repair():
    """
    Compara rectes de calibracio:
      A) Sense batman (actual)
      B) Amb batman reparat (repair_with_parabola)

    Importa les seqs, calcula arees originals i reparades,
    i fa la regressio per comparar R².
    """
    config = get_config()

    cal_seqs = {
        "153_SEQ": {"expected_mode": "COLUMN", "expected_vol": 400},
        "206_SEQ": {"expected_mode": "COLUMN", "expected_vol": 400},
        "152_SEQ_BP": {"expected_mode": "BP", "expected_vol": 100},
        "156_SEQ_BP": {"expected_mode": "BP", "expected_vol": 100},
    }

    # Recollir totes les dades KHP amb arees originals i reparades
    rows = []

    for seq_name, info in cal_seqs.items():
        seq_path = str(DADES_DIR / seq_name)
        print(f"\n  Importando {seq_name}...")

        try:
            imported = import_sequence(seq_path, config)
        except Exception as e:
            print(f"    ERROR: {e}")
            continue

        if not imported or not imported.get("success"):
            continue

        samples = imported.get("samples", {})
        khp_names = imported.get("khp_samples", [])
        method = imported.get("method", info["expected_mode"])
        is_bp = (method == "BP")

        for khp_name in khp_names:
            sample = samples.get(khp_name, {})
            replicas = sample.get("replicas", {})
            conc = extract_khp_conc(khp_name)
            if conc <= 0:
                continue

            rep_areas_orig = []
            rep_areas_rep = []
            rep_batman = []
            rep_snrs = []

            for rep_num, rep_data in sorted(replicas.items()):
                direct = rep_data.get("direct") or {}
                t = direct.get("t")
                y_net = direct.get("y_net")
                if t is None or y_net is None:
                    continue

                t = np.asarray(t)
                y_net = np.asarray(y_net)

                # Detectar pic
                peak_info = detect_main_peak(t, y_net)
                if not peak_info or not peak_info.get("valid"):
                    continue

                peak_idx = peak_info["peak_idx"]
                left_idx = peak_info.get("left_idx", 0)
                right_idx = peak_info.get("right_idx", len(y_net) - 1)

                all_peaks = detect_all_peaks(t, y_net)
                for pk in all_peaks:
                    if abs(pk["t"] - peak_info["t_max"]) < 0.1:
                        left_idx = pk["left_idx"]
                        right_idx = pk["right_idx"]
                        break

                expansion = expand_integration_limits_to_baseline(
                    t, y_net, left_idx, right_idx, peak_idx,
                    baseline_threshold_pct=15,
                    min_width_minutes=1.0,
                    max_width_minutes=6.0 if is_bp else 10.0,
                    is_bp=is_bp
                )
                left_idx = expansion["left_idx"]
                right_idx = expansion["right_idx"]

                # Area original
                area_orig = float(np.trapezoid(
                    y_net[left_idx:right_idx+1], t[left_idx:right_idx+1]
                ))

                # SNR
                from hpsec_core import get_baseline_stats
                bl_mode = "BP" if is_bp else "COLUMN"
                bl_stats = get_baseline_stats(t, y_net, mode=bl_mode)
                bl_std = max(bl_stats.get("std", 0.01), 0.5)
                snr = float((y_net[peak_idx] - bl_stats["mean"]) / bl_std)

                # Detectar i reparar batman
                t_seg = t[left_idx:right_idx+1]
                y_seg = y_net[left_idx:right_idx+1]
                anomaly = detect_peak_anomaly(t_seg, y_seg)
                has_batman = anomaly.get("is_batman", False)

                area_repaired = area_orig
                if has_batman:
                    y_rep_seg, _, was_repaired = repair_with_parabola(t_seg, y_seg)
                    if was_repaired:
                        area_repaired = float(np.trapezoid(y_rep_seg, t_seg))

                rep_areas_orig.append(area_orig)
                rep_areas_rep.append(area_repaired)
                rep_batman.append(has_batman)
                rep_snrs.append(snr)

            if not rep_areas_orig:
                continue

            # Volum d'injeccio
            first_rep = list(replicas.values())[0]
            inj_info = first_rep.get("injection_info", {})
            volume = inj_info.get("inj_volume", info["expected_vol"])

            ug_c = conc * volume / 1000
            any_batman = any(rep_batman)

            rows.append({
                "seq_name": seq_name,
                "khp_name": khp_name,
                "mode": method,
                "conc_ppm": conc,
                "volume_uL": volume,
                "ug_c": ug_c,
                "area_orig": np.mean(rep_areas_orig),
                "area_repaired": np.mean(rep_areas_rep),
                "has_batman": any_batman,
                "n_replicas": len(rep_areas_orig),
                "snr": np.mean(rep_snrs),
            })

    df = pd.DataFrame(rows)
    print(f"\n  Total punts: {len(df)} ({df['has_batman'].sum()} amb batman)")

    # Filtrar 5ppm de 152_SEQ_BP (area invertida)
    bad = (df["seq_name"] == "152_SEQ_BP") & (df["conc_ppm"] == 5.0)
    df = df[~bad].copy()

    # =====================================================================
    # TAULA COMPLETA DE DADES
    # =====================================================================
    print("\n  Dades completes (area repaired si batman, sino orig):")
    print(f"  {'Seq':15s} {'mode':>6s} {'KHP':8s} {'ppm':>5s} {'vol':>4s} {'ugC':>6s} "
          f"{'area_o':>8s} {'area_r':>8s} {'SNR':>6s} {'bat':>4s}")
    print("  " + "-" * 82)
    for _, r in df.sort_values(["ug_c"]).iterrows():
        bat = "BAT" if r["has_batman"] else ""
        print(f"  {r['seq_name']:15s} {r['mode']:>6s} {r['khp_name']:8s} {r['conc_ppm']:5.2f} "
              f"{r['volume_uL']:4.0f} {r['ug_c']:6.3f} "
              f"{r['area_orig']:8.1f} {r['area_repaired']:8.1f} "
              f"{r['snr']:6.0f} {bat:>4s}")

    # =====================================================================
    # COMPARACIO: amb i sense batman, amb i sense baixa conc
    # =====================================================================
    print("\n" + "=" * 80)
    print("COMPARACIO RECTES DE CALIBRACIO")
    print("=" * 80)

    # Usar area_repaired per tots (batman reparat, resta igual)
    df["area_best"] = df["area_repaired"]

    scenarios = OrderedDict()

    # 1) Tots punts (batman reparat) — referencia
    scenarios["Tots (bat rep)"] = {
        "df": df, "y_col": "area_best", "color": "#95A5A6",
    }

    # 2) ugC >= 0.05 (treure BP 0.25ppm = 0.025 ugC)
    df_005 = df[df["ug_c"] >= 0.05].copy()
    scenarios["ugC>=0.05"] = {
        "df": df_005, "y_col": "area_best", "color": "#3498DB",
    }

    # 3) ugC >= 0.10 (treure BP 0.25+0.5ppm i COL 0.25ppm)
    df_010 = df[df["ug_c"] >= 0.10].copy()
    scenarios["ugC>=0.10"] = {
        "df": df_010, "y_col": "area_best", "color": "#E67E22",
    }

    # 4) ugC >= 0.20 (nomes >= 0.5ppm COL o >= 2ppm BP)
    df_020 = df[df["ug_c"] >= 0.20].copy()
    scenarios["ugC>=0.20"] = {
        "df": df_020, "y_col": "area_best", "color": "#27AE60",
    }

    # 5) ugC >= 0.30 (nomes >= 1ppm COL o >= 3ppm BP)
    df_030 = df[df["ug_c"] >= 0.30].copy()
    scenarios["ugC>=0.30"] = {
        "df": df_030, "y_col": "area_best", "color": "#8E44AD",
    }

    for name, s in scenarios.items():
        sdf = s["df"]
        x = sdf["ug_c"].values
        y = sdf[s["y_col"]].values
        slope, intercept, r2 = fit_line(x, y)
        s["x"], s["y"] = x, y
        s["slope"], s["intercept"], s["r2"] = slope, intercept, r2
        s["n"] = len(sdf)

        # RMSE en ppm
        y_pred = slope * x + intercept
        res_pct = (y - y_pred) / np.maximum(y_pred, 1) * 100
        rmse = np.sqrt(np.mean(res_pct**2))
        within_10 = (np.abs(res_pct) < 10).sum()
        s["rmse"] = rmse
        s["within_10"] = within_10

        snr_min = sdf["snr"].min() if len(sdf) > 0 else 0
        ugc_min = sdf["ug_c"].min() if len(sdf) > 0 else 0
        n_col = (sdf["mode"] == "COLUMN").sum() if len(sdf) > 0 else 0
        n_bp = (sdf["mode"] == "BP").sum() if len(sdf) > 0 else 0
        s["n_col"], s["n_bp"] = n_col, n_bp

        print(f"\n  {name} (n={len(sdf)}: {n_col} COL + {n_bp} BP, ugC_min={ugc_min:.3f}, SNR_min={snr_min:.0f}):")
        print(f"    y = {slope:.1f}x + {intercept:.1f}   R\u00b2 = {r2:.6f}   "
              f"RMSE={rmse:.1f}%   <10%: {within_10}/{len(sdf)}")

    # =====================================================================
    # GRAFIC COMPARATIU: 2x3
    # =====================================================================
    n_sc = len(scenarios)
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(
        "Recta de calibracio unificada COL+BP: llindar minim de ugC (batman reparat)",
        fontsize=14, fontweight="bold"
    )

    for ax_idx, (name, s) in enumerate(scenarios.items()):
        ax = axes[ax_idx // 3, ax_idx % 3]
        x, y = s["x"], s["y"]
        slope, intercept, r2 = s["slope"], s["intercept"], s["r2"]
        sdf = s["df"]

        # Punts amb color per mode
        for _, r in sdf.iterrows():
            c = "#3498DB" if r["mode"] == "COLUMN" else "#E67E22"
            m = "D" if r["has_batman"] else "o"
            y_val = r[s["y_col"]]
            ax.scatter(r["ug_c"], y_val, c=c, s=90, marker=m,
                      alpha=0.8, edgecolors="white", zorder=3)

        # Recta
        x_fit = np.linspace(0, 2.2, 100)
        ax.plot(x_fit, slope * x_fit + intercept, color=s["color"], lw=2.5,
                label=f"y = {slope:.0f}x + {intercept:.0f}")

        # Banda ±10%
        ax.fill_between(x_fit,
                        (slope * x_fit + intercept) * 0.9,
                        (slope * x_fit + intercept) * 1.1,
                        alpha=0.1, color=s["color"])

        ax.set_xlabel("ug C injectat", fontsize=10)
        ax.set_ylabel("Area DOC", fontsize=10)
        ax.set_title(f"{name} (n={s['n']}: {s['n_col']}COL+{s['n_bp']}BP)\n"
                     f"y={slope:.0f}x+{intercept:.0f}  R\u00b2={r2:.6f}  RMSE={s['rmse']:.1f}%",
                     fontsize=10, fontweight="bold")
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, 2.2)
        ax.set_ylim(-50, 1800)

    # Ultim subplot: taula resum
    ax = axes[1, 2]
    ax.axis("off")
    table_data = []
    for name, s in scenarios.items():
        table_data.append([
            name, str(s["n"]), f"{s['slope']:.0f}",
            f"{s['intercept']:.0f}", f"{s['r2']:.5f}",
            f"{s['rmse']:.1f}%", f"{s['within_10']}/{s['n']}"
        ])
    table = ax.table(
        cellText=table_data,
        colLabels=["Escenari", "n", "slope", "intercept", "R\u00b2", "RMSE", "<10%"],
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.8)
    # Highlight best R²
    best_idx = max(range(len(table_data)), key=lambda i: scenarios[list(scenarios.keys())[i]]["r2"])
    for j in range(7):
        table[best_idx + 1, j].set_facecolor("#D5F5E3")
    ax.set_title("Resum comparatiu", fontsize=11, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = OUTPUT_DIR / "comparacio_filtres_calibracio.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Grafic: {out_path}")

    # =====================================================================
    # RESUM FINAL
    # =====================================================================
    print("\n" + "=" * 80)
    print("RESUM FINAL")
    print("=" * 80)
    print(f"\n  {'Escenari':18s} {'n':>3s} {'COL':>4s} {'BP':>3s} {'slope':>6s} "
          f"{'i':>5s} {'R2':>9s} {'RMSE':>7s} {'<10%':>6s}")
    print("  " + "-" * 70)
    for name, s in scenarios.items():
        print(f"  {name:18s} {s['n']:3d} {s['n_col']:4d} {s['n_bp']:3d} {s['slope']:6.0f} "
              f"{s['intercept']:5.0f} {s['r2']:9.6f} {s['rmse']:6.1f}% "
              f"{s['within_10']:2d}/{s['n']}")


def diagnose_bp_integration():
    """
    Diagnostic detallat de la integracio BP:
    Per cada pic KHP, mostra que integra exactament la suite:
    - Nivell y_net als limits d'integracio
    - Pedestal (area del rectangl base dins limits)
    - Area neta (pic menys pedestal)
    - Compara COL vs BP
    """
    config = get_config()

    cal_seqs = {
        "153_SEQ": {"expected_mode": "COLUMN", "expected_vol": 400},
        "206_SEQ": {"expected_mode": "COLUMN", "expected_vol": 400},
        "152_SEQ_BP": {"expected_mode": "BP", "expected_vol": 100},
        "156_SEQ_BP": {"expected_mode": "BP", "expected_vol": 100},
    }

    from hpsec_core import get_baseline_stats

    all_rows = []

    for seq_name, info in cal_seqs.items():
        seq_path = str(DADES_DIR / seq_name)
        print(f"\n  Importando {seq_name}...")

        try:
            imported = import_sequence(seq_path, config)
        except Exception as e:
            print(f"    ERROR: {e}")
            continue

        if not imported or not imported.get("success"):
            continue

        samples = imported.get("samples", {})
        khp_names = imported.get("khp_samples", [])
        method = imported.get("method", info["expected_mode"])
        is_bp = (method == "BP")

        for khp_name in khp_names:
            sample = samples.get(khp_name, {})
            replicas = sample.get("replicas", {})
            conc = extract_khp_conc(khp_name)
            if conc <= 0:
                continue

            for rep_num, rep_data in sorted(replicas.items()):
                direct = rep_data.get("direct") or {}
                t = direct.get("t")
                y_net = direct.get("y_net")
                y_raw = direct.get("y")  # senyal raw (sense restar baseline)
                baseline = direct.get("baseline")
                if t is None or y_net is None:
                    continue

                t = np.asarray(t)
                y_net = np.asarray(y_net)

                volume = info["expected_vol"]
                ug_c = conc * volume / 1000

                # Detectar pic
                peak_info = detect_main_peak(t, y_net)
                if not peak_info or not peak_info.get("valid"):
                    continue

                peak_idx = peak_info["peak_idx"]
                left_idx = peak_info.get("left_idx", 0)
                right_idx = peak_info.get("right_idx", len(y_net) - 1)

                all_peaks = detect_all_peaks(t, y_net)
                for pk in all_peaks:
                    if abs(pk["t"] - peak_info["t_max"]) < 0.1:
                        left_idx = pk["left_idx"]
                        right_idx = pk["right_idx"]
                        break

                # Expandir limits (exactament com fa la suite)
                expansion = expand_integration_limits_to_baseline(
                    t, y_net, left_idx, right_idx, peak_idx,
                    baseline_threshold_pct=15,
                    min_width_minutes=1.0,
                    max_width_minutes=6.0 if is_bp else 10.0,
                    is_bp=is_bp
                )
                left_idx = expansion["left_idx"]
                right_idx = expansion["right_idx"]

                t_start = float(t[left_idx])
                t_end = float(t[right_idx])
                width = t_end - t_start
                t_max = peak_info["t_max"]
                y_peak = float(y_net[peak_idx])

                # ====== DIAGNOSTIC CLAU ======
                # Nivells als limits
                y_left = float(y_net[left_idx])
                y_right = float(y_net[right_idx])
                y_min_segment = float(np.min(y_net[left_idx:right_idx+1]))

                # "Pedestal" = el minim del segment dins limits
                pedestal_level = max(0, min(y_left, y_right))
                # Area total integrada (com fa la suite: trapezoid sobre y_net)
                area_total = float(np.trapezoid(
                    y_net[left_idx:right_idx+1], t[left_idx:right_idx+1]
                ))
                # Area del pedestal (rectangle)
                area_pedestal = pedestal_level * width
                # Area neta (pic menys pedestal)
                area_net = area_total - area_pedestal

                # Percentatge pedestal
                pct_pedestal = (area_pedestal / area_total * 100) if area_total > 0 else 0

                # Que integra integrate_above_baseline?
                bl_mode = "BP" if is_bp else "COLUMN"
                bl_stats = get_baseline_stats(t, y_net, mode=bl_mode)
                iab = None
                try:
                    from hpsec_core import integrate_above_baseline
                    iab = integrate_above_baseline(
                        t[left_idx:right_idx+1],
                        y_net[left_idx:right_idx+1],
                        bl_stats["mean"], bl_stats["std"]
                    )
                except Exception:
                    pass

                # Batman
                t_seg = t[left_idx:right_idx+1]
                y_seg = y_net[left_idx:right_idx+1]
                anomaly = detect_peak_anomaly(t_seg, y_seg)
                has_batman = anomaly.get("is_batman", False)

                # SNR
                bl_std = max(bl_stats.get("std", 0.01), 0.5)
                snr = float((y_peak - bl_stats["mean"]) / bl_std)

                all_rows.append({
                    "seq": seq_name,
                    "khp": khp_name,
                    "rep": rep_num,
                    "mode": method,
                    "conc": conc,
                    "vol": volume,
                    "ug_c": ug_c,
                    "t_max": t_max,
                    "t_start": t_start,
                    "t_end": t_end,
                    "width": width,
                    "y_peak": y_peak,
                    "y_left": y_left,
                    "y_right": y_right,
                    "y_min_seg": y_min_segment,
                    "pedestal": pedestal_level,
                    "area_total": area_total,
                    "area_pedestal": area_pedestal,
                    "area_net": area_net,
                    "pct_pedestal": pct_pedestal,
                    "area_iab": iab["area"] if iab else None,
                    "iab_threshold": iab["threshold"] if iab else None,
                    "bl_mean": bl_stats["mean"],
                    "bl_std": bl_stats["std"],
                    "snr": snr,
                    "batman": has_batman,
                    # Guardar arrays per plots
                    "_t": t,
                    "_y_net": y_net,
                    "_left_idx": left_idx,
                    "_right_idx": right_idx,
                    "_peak_idx": peak_idx,
                })

    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in all_rows])

    # =====================================================================
    # TAULA DIAGNOSTIC
    # =====================================================================
    print("\n" + "=" * 120)
    print("DIAGNOSTIC INTEGRACIO BP vs COLUMN")
    print("=" * 120)

    # Agrupar per mode
    for mode in ["COLUMN", "BP"]:
        sub = df[df["mode"] == mode].sort_values(["seq", "ug_c", "rep"])
        print(f"\n  === {mode} ===")
        print(f"  {'Seq':15s} {'KHP':8s} {'R':>2s} {'ppm':>5s} {'ugC':>5s} "
              f"{'t_max':>5s} {'width':>5s} {'y_peak':>7s} "
              f"{'y_left':>7s} {'y_right':>7s} {'pedestal':>8s} "
              f"{'area_tot':>8s} {'a_pedes':>8s} {'a_net':>8s} {'%ped':>5s} "
              f"{'bl_mean':>7s} {'bl_std':>6s} {'SNR':>5s}")
        print("  " + "-" * 135)

        for _, r in sub.iterrows():
            bat = "*" if r["batman"] else ""
            print(f"  {r['seq']:15s} {r['khp']:8s} {r['rep']:>2} {r['conc']:5.2f} {r['ug_c']:5.3f} "
                  f"{r['t_max']:5.2f} {r['width']:5.2f} {r['y_peak']:7.1f} "
                  f"{r['y_left']:7.2f} {r['y_right']:7.2f} {r['pedestal']:8.2f} "
                  f"{r['area_total']:8.1f} {r['area_pedestal']:8.1f} {r['area_net']:8.1f} "
                  f"{r['pct_pedestal']:4.1f}% "
                  f"{r['bl_mean']:7.2f} {r['bl_std']:6.2f} {r['snr']:5.0f}{bat}")

    # =====================================================================
    # ESTADISTIQUES COMPARATIVES
    # =====================================================================
    print("\n" + "=" * 120)
    print("ESTADISTIQUES PER MODE")
    print("=" * 120)

    for mode in ["COLUMN", "BP"]:
        sub = df[df["mode"] == mode]
        print(f"\n  {mode}:")
        print(f"    y_left  mean={sub['y_left'].mean():6.2f}  std={sub['y_left'].std():6.2f}  "
              f"min={sub['y_left'].min():6.2f}  max={sub['y_left'].max():6.2f}")
        print(f"    y_right mean={sub['y_right'].mean():6.2f}  std={sub['y_right'].std():6.2f}  "
              f"min={sub['y_right'].min():6.2f}  max={sub['y_right'].max():6.2f}")
        print(f"    pedestal mean={sub['pedestal'].mean():6.2f}  std={sub['pedestal'].std():6.2f}")
        print(f"    %pedestal mean={sub['pct_pedestal'].mean():5.1f}%  "
              f"min={sub['pct_pedestal'].min():5.1f}%  max={sub['pct_pedestal'].max():5.1f}%")
        print(f"    width   mean={sub['width'].mean():5.2f}  std={sub['width'].std():5.2f}")
        print(f"    bl_mean mean={sub['bl_mean'].mean():6.2f}  std={sub['bl_mean'].std():6.2f}")
        print(f"    bl_std  mean={sub['bl_std'].mean():6.2f}  std={sub['bl_std'].std():6.2f}")

    # Per sequencia BP
    print("\n  Per sequencia BP:")
    for seq in ["152_SEQ_BP", "156_SEQ_BP"]:
        sub = df[df["seq"] == seq]
        if sub.empty:
            continue
        print(f"\n    {seq}:")
        print(f"      pedestal mean={sub['pedestal'].mean():6.2f}")
        print(f"      %pedestal mean={sub['pct_pedestal'].mean():5.1f}%")
        print(f"      width    mean={sub['width'].mean():5.2f}")
        print(f"      bl_mean  mean={sub['bl_mean'].mean():6.2f}")

    # =====================================================================
    # RECTA AMB AREA NETA vs TOTAL
    # =====================================================================
    print("\n" + "=" * 120)
    print("RECTA CALIBRACIO: area total vs area neta (sense pedestal)")
    print("=" * 120)

    # Filtrar 5ppm de 152_SEQ_BP
    df_clean = df[~((df["seq"] == "152_SEQ_BP") & (df["conc"] == 5.0))].copy()

    # Agrupar per mostra (mean de repliques)
    grouped = df_clean.groupby(["seq", "khp", "mode", "conc", "ug_c"]).agg({
        "area_total": "mean",
        "area_net": "mean",
        "area_pedestal": "mean",
        "pct_pedestal": "mean",
        "pedestal": "mean",
        "width": "mean",
    }).reset_index()

    for area_col, label in [("area_total", "TOTAL (com la suite)"), ("area_net", "NETA (sense pedestal)")]:
        x = grouped["ug_c"].values
        y = grouped[area_col].values
        s, i, r2 = fit_line(x, y)
        y_pred = s * x + i
        res_pct = (y - y_pred) / np.maximum(y_pred, 1) * 100
        rmse = np.sqrt(np.mean(res_pct**2))

        # Per mode
        for mode in ["COLUMN", "BP"]:
            sub = grouped[grouped["mode"] == mode]
            if len(sub) >= 2:
                s_m, i_m, r2_m = fit_line(sub["ug_c"].values, sub[area_col].values)
            else:
                s_m, i_m, r2_m = 0, 0, 0
            print(f"    {mode:8s}: slope={s_m:6.0f}  intercept={i_m:6.1f}  R2={r2_m:.5f}")

        print(f"\n  {label}:")
        print(f"    UNIFICADA: y = {s:.0f}x + {i:.0f}   R2={r2:.6f}  RMSE={rmse:.1f}%")

    # =====================================================================
    # GRAFIC DIAGNOSTIC
    # =====================================================================
    # 2x2: cromatogrames zoom + rectes
    n_bp = len([r for r in all_rows if r["mode"] == "BP"])
    n_col = len([r for r in all_rows if r["mode"] == "COLUMN"])

    # Plot 1: Exemples de BP amb pedestal marcat
    fig, axes = plt.subplots(3, 4, figsize=(24, 16))
    fig.suptitle("Diagnostic Integracio: y_net als limits d'integracio\n"
                 "(blau = y_net, verd = limits, vermell = pedestal)",
                 fontsize=14, fontweight="bold")

    # Seleccionar fins a 12 exemples: 6 BP + 6 COL
    bp_rows = [r for r in all_rows if r["mode"] == "BP"]
    col_rows = [r for r in all_rows if r["mode"] == "COLUMN"]

    # Ordenar per ugC per veure l'efecte a baixa conc
    bp_rows.sort(key=lambda r: r["ug_c"])
    col_rows.sort(key=lambda r: r["ug_c"])

    # Agafar 6 de cada (distribuïts)
    def select_representative(rows, n=6):
        if len(rows) <= n:
            return rows
        step = len(rows) / n
        return [rows[int(i * step)] for i in range(n)]

    selected_bp = select_representative(bp_rows, 6)
    selected_col = select_representative(col_rows, 6)

    for plot_idx, (row, col_offset) in enumerate(
        [(r, 0) for r in selected_bp[:6]] +
        [(r, 0) for r in selected_col[:6]]
    ):
        if plot_idx >= 12:
            break
        ax = axes[plot_idx // 4, plot_idx % 4]

        t = row["_t"]
        y_net = row["_y_net"]
        left_idx = row["_left_idx"]
        right_idx = row["_right_idx"]
        peak_idx = row["_peak_idx"]

        t_start = float(t[left_idx])
        t_end = float(t[right_idx])
        y_left = float(y_net[left_idx])
        y_right = float(y_net[right_idx])
        y_peak = float(y_net[peak_idx])
        pedestal = row["pedestal"]

        # Zoom al pic
        margin = (t_end - t_start) * 0.3
        view_left = max(t[0], t_start - margin)
        view_right = min(t[-1], t_end + margin)
        mask = (t >= view_left) & (t <= view_right)

        # Cromatograma
        ax.plot(t[mask], y_net[mask], "b-", lw=0.8, alpha=0.7)

        # Area total (blau clar)
        seg_mask = (np.arange(len(t)) >= left_idx) & (np.arange(len(t)) <= right_idx)
        ax.fill_between(t[seg_mask], 0, y_net[seg_mask], alpha=0.15, color="#3498DB")

        # Pedestal (vermell)
        if pedestal > 0.5:
            ax.fill_between(
                [t_start, t_end], 0, pedestal,
                alpha=0.3, color="#E74C3C", hatch="///",
                label=f"Pedestal={pedestal:.1f} ({row['pct_pedestal']:.1f}%)"
            )

        # Limits
        ax.axvline(t_start, color="green", ls="--", lw=1, alpha=0.7)
        ax.axvline(t_end, color="green", ls="--", lw=1, alpha=0.7)

        # Marcar y als limits
        ax.plot(t_start, y_left, "ro", ms=6, zorder=5)
        ax.plot(t_end, y_right, "ro", ms=6, zorder=5)
        ax.axhline(0, color="gray", ls=":", lw=0.5)

        # Títol
        mode_color = "#E67E22" if row["mode"] == "BP" else "#3498DB"
        ax.set_title(
            f"{row['seq']} | {row['khp']} R{row['rep']}\n"
            f"{row['mode']} | {row['conc']:.2f}ppm ({row['ug_c']:.3f}ugC) | "
            f"area={row['area_total']:.0f} | ped={row['pct_pedestal']:.0f}%",
            fontsize=7.5, fontweight="bold", color=mode_color
        )
        ax.set_xlabel("min", fontsize=7)
        ax.set_ylabel("y_net", fontsize=7)
        ax.tick_params(axis="both", labelsize=6)
        ax.grid(True, alpha=0.3)
        if pedestal > 0.5:
            ax.legend(fontsize=6, loc="upper right")
        ax.set_ylim(-y_peak * 0.05, y_peak * 1.2)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out1 = OUTPUT_DIR / "diagnostic_integracio_bp.png"
    plt.savefig(str(out1), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Grafic: {out1}")

    # =====================================================================
    # GRAFIC 2: Recta area_total vs area_net
    # =====================================================================
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    fig.suptitle("Efecte del pedestal en la recta de calibracio", fontsize=14, fontweight="bold")

    for ax_idx, (area_col, title) in enumerate([
        ("area_total", "Area TOTAL (com fa la suite)"),
        ("area_net", "Area NETA (sense pedestal)"),
    ]):
        ax = axes[ax_idx]
        for mode, color, marker in [("COLUMN", "#3498DB", "o"), ("BP", "#E67E22", "s")]:
            sub = grouped[grouped["mode"] == mode]
            ax.scatter(sub["ug_c"], sub[area_col], c=color, marker=marker, s=80,
                       alpha=0.8, label=mode, zorder=3)

        x = grouped["ug_c"].values
        y = grouped[area_col].values
        s, i, r2 = fit_line(x, y)
        x_fit = np.linspace(0, 2.2, 100)
        ax.plot(x_fit, s * x_fit + i, "k-", lw=2, label=f"y={s:.0f}x+{i:.0f} R²={r2:.5f}")
        ax.fill_between(x_fit, (s*x_fit+i)*0.9, (s*x_fit+i)*1.1, alpha=0.08, color="gray")

        # Per mode
        for mode, color in [("COLUMN", "#3498DB"), ("BP", "#E67E22")]:
            sub = grouped[grouped["mode"] == mode]
            if len(sub) >= 2:
                s_m, i_m, r2_m = fit_line(sub["ug_c"].values, sub[area_col].values)
                ax.plot(x_fit, s_m * x_fit + i_m, color=color, ls="--", lw=1.5, alpha=0.5,
                        label=f"{mode}: {s_m:.0f}x+{i_m:.0f}")

        ax.set_xlabel("ug C injectat")
        ax.set_ylabel("Area")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, 2.2)

    # Tercer subplot: %pedestal vs ugC
    ax = axes[2]
    for mode, color, marker in [("COLUMN", "#3498DB", "o"), ("BP", "#E67E22", "s")]:
        sub = grouped[grouped["mode"] == mode]
        ax.scatter(sub["ug_c"], sub["pct_pedestal"], c=color, marker=marker, s=80,
                   alpha=0.8, label=mode, zorder=3)

    ax.set_xlabel("ug C injectat")
    ax.set_ylabel("% Pedestal")
    ax.set_title("Pes del pedestal vs concentracio\n(mes alt = mes distorsio)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 2.2)
    ax.axhline(10, color="red", ls=":", lw=1, alpha=0.5, label="10%")

    plt.tight_layout()
    out2 = OUTPUT_DIR / "diagnostic_pedestal_recta.png"
    plt.savefig(str(out2), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Grafic: {out2}")

    return df, all_rows


def compare_integration_methods():
    """
    Compara metodes d'integracio:
    A) Actual (3s, drop-to-zero)
    B) 2s threshold, drop-to-zero
    C) 1s threshold, drop-to-zero
    D) Slope-based + V2V (ChemStation-like)
    E) Slope-based + drop-to-zero (hybrid)

    Per cada metode: recta calibracio, R², RMSE, consistencia COL-BP.
    """
    config = get_config()
    from hpsec_calibrate import mode_robust

    cal_seqs = {
        "153_SEQ": {"expected_mode": "COLUMN", "expected_vol": 400},
        "206_SEQ": {"expected_mode": "COLUMN", "expected_vol": 400},
        "152_SEQ_BP": {"expected_mode": "BP", "expected_vol": 100},
        "156_SEQ_BP": {"expected_mode": "BP", "expected_vol": 100},
    }

    def find_limits_sigma(t, y, peak_idx, sigma_mult, max_width_min, min_width_min=1.0):
        """Replica calculate_integration_limits amb sigma_mult configurable."""
        n = len(y)
        baseline = mode_robust(y)
        y_range = float(np.max(y) - baseline)
        if y_range <= 0:
            return 0, n - 1, baseline, 0

        mask_bl = y < (baseline + 0.2 * y_range)
        if np.sum(mask_bl) > 5:
            std_bl = float(np.std(y[mask_bl]))
        else:
            std_bl = float(np.std(y)) * 0.1
        if std_bl < 1e-6:
            std_bl = 0.01 * y_range

        threshold = baseline + sigma_mult * std_bl

        dt = np.mean(np.diff(t)) if n > 1 else 0.01
        max_w_idx = int(max_width_min / dt) if dt > 0 else 300
        min_w_idx = int(min_width_min / dt) if dt > 0 else 50

        # Esquerra
        li = peak_idx
        for i in range(peak_idx - 1, max(0, peak_idx - max_w_idx), -1):
            if y[i] <= threshold:
                li = i
                break
            li = i
        else:
            li = max(0, peak_idx - max_w_idx)

        # Dreta
        ri = peak_idx
        for i in range(peak_idx + 1, min(n, peak_idx + max_w_idx)):
            if y[i] <= threshold:
                ri = i
                break
            ri = i
        else:
            ri = min(n - 1, peak_idx + max_w_idx)

        # Min width
        if ri - li < min_w_idx:
            expand = (min_w_idx - (ri - li)) // 2 + 1
            li = max(0, li - expand)
            ri = min(n - 1, ri + expand)

        li = max(0, int(min(li, peak_idx - 3)))
        ri = min(n - 1, int(max(ri, peak_idx + 3)))

        return li, ri, baseline, threshold

    def find_limits_slope(t, y, peak_idx, max_width_min, noise_mult=3.0):
        """
        Metode ChemStation: limits basats en la primera derivada (pendent).

        Busca on la pendent del senyal cau per sota del llindar de soroll,
        es a dir, on el senyal deixa de canviar i es 'planer'.
        Despres traca baseline local (V2V) entre inici i fi.
        """
        n = len(y)
        dt_arr = np.diff(t)
        dy_arr = np.diff(y)
        # Primera derivada (slope)
        slope = dy_arr / np.maximum(dt_arr, 1e-6)

        # Noise level: std de la slope a les zones planes
        baseline = mode_robust(y)
        y_range = float(np.max(y) - baseline)
        mask_flat = y[:-1] < (baseline + 0.15 * y_range)
        if np.sum(mask_flat) > 10:
            slope_noise = float(np.std(slope[mask_flat]))
        else:
            slope_noise = float(np.std(slope)) * 0.1
        if slope_noise < 1e-6:
            slope_noise = 0.01 * float(np.max(np.abs(slope)))

        # Slope threshold
        slope_threshold = noise_mult * slope_noise

        dt = np.mean(dt_arr) if n > 1 else 0.01
        max_w_idx = int(max_width_min / dt) if dt > 0 else 300

        # Buscar esquerra: des del pic cap a l'esquerra,
        # on |slope| < threshold durant uns quants punts consecutius
        consec_needed = max(3, int(0.1 / dt))  # 0.1 min de senyal planer
        li = max(0, peak_idx - max_w_idx)

        consec = 0
        for i in range(peak_idx - 1, max(0, peak_idx - max_w_idx), -1):
            if i < len(slope) and abs(slope[i]) < slope_threshold:
                consec += 1
                if consec >= consec_needed:
                    li = i + consec_needed  # inici del pic (on slope comenca)
                    break
            else:
                consec = 0

        # Buscar dreta: des del pic cap a la dreta
        ri = min(n - 1, peak_idx + max_w_idx)
        consec = 0
        for i in range(peak_idx + 1, min(len(slope), peak_idx + max_w_idx)):
            if abs(slope[i]) < slope_threshold:
                consec += 1
                if consec >= consec_needed:
                    ri = i - consec_needed + 1  # fi del pic
                    break
            else:
                consec = 0

        # Assegurar minim
        li = max(0, int(min(li, peak_idx - 3)))
        ri = min(n - 1, int(max(ri, peak_idx + 3)))

        return li, ri, slope_threshold, slope_noise

    rows = []

    for seq_name, info in cal_seqs.items():
        seq_path = str(DADES_DIR / seq_name)
        print(f"\n  Importando {seq_name}...")

        try:
            imported = import_sequence(seq_path, config)
        except Exception as e:
            print(f"    ERROR: {e}")
            continue

        if not imported or not imported.get("success"):
            continue

        samples = imported.get("samples", {})
        khp_names = imported.get("khp_samples", [])
        method = imported.get("method", info["expected_mode"])
        is_bp = (method == "BP")

        for khp_name in khp_names:
            sample = samples.get(khp_name, {})
            replicas = sample.get("replicas", {})
            conc = extract_khp_conc(khp_name)
            if conc <= 0:
                continue

            for rep_num, rep_data in sorted(replicas.items()):
                direct = rep_data.get("direct") or {}
                t = direct.get("t")
                y_net = direct.get("y_net")
                if t is None or y_net is None:
                    continue

                t = np.asarray(t)
                y_net = np.asarray(y_net)

                volume = info["expected_vol"]
                ug_c = conc * volume / 1000
                max_w = 6.0 if is_bp else 10.0

                # Detectar pic
                peak_info = detect_main_peak(t, y_net)
                if not peak_info or not peak_info.get("valid"):
                    continue

                peak_idx = peak_info["peak_idx"]
                y_peak = float(y_net[peak_idx])
                t_max = peak_info["t_max"]

                # Batman repair (aplicar una vegada, usar per tots els metodes)
                # Primer: limits 3s per detectar batman
                li3, ri3, _, _ = find_limits_sigma(t, y_net, peak_idx, 3.0, max_w)
                t_seg = t[li3:ri3+1]
                y_seg = y_net[li3:ri3+1]
                anomaly = detect_peak_anomaly(t_seg, y_seg)
                has_batman = anomaly.get("is_batman", False)

                y_use = y_net.copy()
                if has_batman:
                    y_rep_seg, _, was_repaired = repair_with_parabola(t_seg, y_seg)
                    if was_repaired:
                        y_use[li3:ri3+1] = y_rep_seg

                row = {
                    "seq": seq_name, "khp": khp_name, "rep": rep_num,
                    "mode": method, "conc": conc, "vol": volume,
                    "ug_c": ug_c, "t_max": t_max, "y_peak": y_peak,
                    "batman": has_batman,
                }

                # ====== A) Actual 3s drop-to-zero ======
                li, ri, bl, th = find_limits_sigma(t, y_use, peak_idx, 3.0, max_w)
                area = float(np.trapezoid(y_use[li:ri+1], t[li:ri+1]))
                row["area_3s"] = area
                row["w_3s"] = float(t[ri] - t[li])
                row["yL_3s"] = float(y_use[li])
                row["yR_3s"] = float(y_use[ri])
                row["th_3s"] = th

                # ====== B) 2s drop-to-zero ======
                li, ri, bl, th = find_limits_sigma(t, y_use, peak_idx, 2.0, max_w)
                area = float(np.trapezoid(y_use[li:ri+1], t[li:ri+1]))
                row["area_2s"] = area
                row["w_2s"] = float(t[ri] - t[li])
                row["yL_2s"] = float(y_use[li])
                row["yR_2s"] = float(y_use[ri])

                # ====== C) 1s drop-to-zero ======
                li, ri, bl, th = find_limits_sigma(t, y_use, peak_idx, 1.0, max_w)
                area = float(np.trapezoid(y_use[li:ri+1], t[li:ri+1]))
                row["area_1s"] = area
                row["w_1s"] = float(t[ri] - t[li])
                row["yL_1s"] = float(y_use[li])
                row["yR_1s"] = float(y_use[ri])

                # ====== D) Slope-based (ChemStation) + V2V ======
                li_sl, ri_sl, sl_th, sl_noise = find_limits_slope(
                    t, y_use, peak_idx, max_w, noise_mult=3.0
                )
                t_seg = t[li_sl:ri_sl+1]
                y_seg = y_use[li_sl:ri_sl+1]
                # V2V baseline
                y_l = float(y_use[li_sl])
                y_r = float(y_use[ri_sl])
                bl_local = np.interp(t_seg, [t_seg[0], t_seg[-1]], [y_l, y_r])
                y_above = np.maximum(y_seg - bl_local, 0)
                area_slope_v2v = float(np.trapezoid(y_above, t_seg))
                row["area_slope_v2v"] = area_slope_v2v
                row["w_slope"] = float(t[ri_sl] - t[li_sl])
                row["yL_slope"] = y_l
                row["yR_slope"] = y_r

                # ====== E) Slope-based drop-to-zero ======
                area_slope_d2z = float(np.trapezoid(y_seg, t_seg))
                row["area_slope_d2z"] = area_slope_d2z

                # ====== F) 3s + V2V (referencia anterior) ======
                li3, ri3, _, _ = find_limits_sigma(t, y_use, peak_idx, 3.0, max_w)
                t_seg3 = t[li3:ri3+1]
                y_seg3 = y_use[li3:ri3+1]
                y_l3 = float(y_use[li3])
                y_r3 = float(y_use[ri3])
                bl3 = np.interp(t_seg3, [t_seg3[0], t_seg3[-1]], [y_l3, y_r3])
                y_above3 = np.maximum(y_seg3 - bl3, 0)
                row["area_3s_v2v"] = float(np.trapezoid(y_above3, t_seg3))

                rows.append(row)

    df = pd.DataFrame(rows)
    print(f"\n  Total repliques: {len(df)}")

    # Filtrar 5ppm 152_SEQ_BP
    df = df[~((df["seq"] == "152_SEQ_BP") & (df["conc"] == 5.0))].copy()

    # =====================================================================
    # TAULA DETALLADA: nivells als limits per cada metode
    # =====================================================================
    print("\n" + "=" * 140)
    print("NIVELLS ALS LIMITS D'INTEGRACIO PER METODE")
    print("=" * 140)

    for mode in ["BP"]:  # Focus en BP que es el critic
        sub = df[df["mode"] == mode].sort_values(["seq", "ug_c", "rep"])
        print(f"\n  === {mode} ===")
        print(f"  {'Seq':15s} {'KHP':8s} {'ugC':>5s} "
              f"{'3s':>8s} {'yL':>5s} {'yR':>5s} {'w':>4s}  "
              f"{'2s':>8s} {'yL':>5s} {'yR':>5s} {'w':>4s}  "
              f"{'1s':>8s} {'yL':>5s} {'yR':>5s} {'w':>4s}  "
              f"{'slope':>8s} {'yL':>5s} {'yR':>5s} {'w':>4s}")
        print("  " + "-" * 130)
        for _, r in sub.iterrows():
            print(f"  {r['seq']:15s} {r['khp']:8s} {r['ug_c']:5.3f} "
                  f"{r['area_3s']:8.1f} {r['yL_3s']:5.1f} {r['yR_3s']:5.1f} {r['w_3s']:4.1f}  "
                  f"{r['area_2s']:8.1f} {r['yL_2s']:5.1f} {r['yR_2s']:5.1f} {r['w_2s']:4.1f}  "
                  f"{r['area_1s']:8.1f} {r['yL_1s']:5.1f} {r['yR_1s']:5.1f} {r['w_1s']:4.1f}  "
                  f"{r['area_slope_v2v']:8.1f} {r['yL_slope']:5.1f} {r['yR_slope']:5.1f} {r['w_slope']:4.1f}")

    # =====================================================================
    # AGRUPAR PER MOSTRA (mean repliques)
    # =====================================================================
    area_cols = ["area_3s", "area_2s", "area_1s", "area_slope_v2v", "area_slope_d2z", "area_3s_v2v"]
    agg_dict = {c: "mean" for c in area_cols}
    agg_dict.update({"yL_3s": "mean", "yR_3s": "mean", "w_3s": "mean",
                     "yL_slope": "mean", "yR_slope": "mean", "w_slope": "mean"})
    grouped = df.groupby(["seq", "khp", "mode", "conc", "ug_c"]).agg(agg_dict).reset_index()

    # =====================================================================
    # RECTES PER METODE
    # =====================================================================
    methods = OrderedDict([
        ("A) 3s drop-zero (actual)", "area_3s"),
        ("B) 2s drop-zero", "area_2s"),
        ("C) 1s drop-zero", "area_1s"),
        ("D) Slope + V2V (ChemSt.)", "area_slope_v2v"),
        ("E) Slope + drop-zero", "area_slope_d2z"),
        ("F) 3s + V2V", "area_3s_v2v"),
    ])

    print("\n" + "=" * 130)
    print("RECTES DE CALIBRACIO PER METODE")
    print("=" * 130)

    results = {}
    for mname, area_col in methods.items():
        x = grouped["ug_c"].values
        y = grouped[area_col].values
        s, i, r2 = fit_line(x, y)
        y_pred = s * x + i
        res = (y - y_pred)
        res_pct = res / np.maximum(y_pred, 1) * 100
        rmse = np.sqrt(np.mean(res_pct**2))

        # Per mode
        mode_stats = {}
        for mode in ["COLUMN", "BP"]:
            sub = grouped[grouped["mode"] == mode]
            if len(sub) >= 2:
                s_m, i_m, r2_m = fit_line(sub["ug_c"].values, sub[area_col].values)
            else:
                s_m, i_m, r2_m = 0, 0, 0
            mode_stats[mode] = {"slope": s_m, "intercept": i_m, "r2": r2_m}

        # Consistencia COL-BP per concentracio comuna
        concs_common = set(grouped[grouped["mode"] == "COLUMN"]["conc"]) & \
                       set(grouped[grouped["mode"] == "BP"]["conc"])
        diffs_pct = []
        for c in sorted(concs_common):
            a_col = grouped[(grouped["mode"] == "COLUMN") & (grouped["conc"] == c)][area_col].mean()
            a_bp = grouped[(grouped["mode"] == "BP") & (grouped["conc"] == c)][area_col].mean()
            # Convertir a ppm amb la recta unificada
            ppm_col = (a_col - i) * 1000 / (s * 400) if s > 0 else 0
            ppm_bp = (a_bp - i) * 1000 / (s * 100) if s > 0 else 0
            diff = (ppm_bp - ppm_col) / c * 100 if c > 0 else 0
            diffs_pct.append({"conc": c, "ppm_col": ppm_col, "ppm_bp": ppm_bp, "diff_pct": diff})

        med_diff = np.median([abs(d["diff_pct"]) for d in diffs_pct]) if diffs_pct else 0
        slope_diff = (mode_stats["BP"]["slope"] - mode_stats["COLUMN"]["slope"]) / \
                     mode_stats["COLUMN"]["slope"] * 100 if mode_stats["COLUMN"]["slope"] > 0 else 0
        int_diff = mode_stats["BP"]["intercept"] - mode_stats["COLUMN"]["intercept"]

        results[mname] = {
            "slope": s, "intercept": i, "r2": r2, "rmse": rmse,
            "mode_stats": mode_stats, "diffs": diffs_pct,
            "med_col_bp_diff": med_diff, "slope_diff_pct": slope_diff,
            "intercept_diff": int_diff, "area_col": area_col,
        }

        print(f"\n  {mname}:")
        print(f"    UNIFICADA: y = {s:.1f}x + {i:.1f}   R²={r2:.6f}   RMSE={rmse:.1f}%")
        print(f"    COLUMN:    y = {mode_stats['COLUMN']['slope']:.1f}x + "
              f"{mode_stats['COLUMN']['intercept']:.1f}   R²={mode_stats['COLUMN']['r2']:.5f}")
        print(f"    BP:        y = {mode_stats['BP']['slope']:.1f}x + "
              f"{mode_stats['BP']['intercept']:.1f}   R²={mode_stats['BP']['r2']:.5f}")
        print(f"    Slope diff COL-BP: {slope_diff:+.1f}%   Intercept diff: {int_diff:+.1f}")
        print(f"    Consistencia COL-BP (mediana |diff ppm|): {med_diff:.1f}%")

        if diffs_pct:
            print(f"    Per concentracio:")
            for d in diffs_pct:
                print(f"      {d['conc']:.2f} ppm: COL={d['ppm_col']:.3f}  "
                      f"BP={d['ppm_bp']:.3f}  diff={d['diff_pct']:+.1f}%")

    # =====================================================================
    # ANALISI AMB CALIBRACIONS SEPARADES (el que realment importa)
    # =====================================================================
    print("\n" + "=" * 130)
    print("ANALISI AMB CALIBRACIONS SEPARADES PER MODE")
    print("(Cada mode usa la seva propia recta — com fa la suite)")
    print("=" * 130)

    for mname, res in results.items():
        area_col = res["area_col"]
        ms = res["mode_stats"]

        # RMSE per mode amb la seva propia calibracio
        mode_rmses = {}
        for mode in ["COLUMN", "BP"]:
            sub = grouped[grouped["mode"] == mode]
            if len(sub) < 2 or ms[mode]["slope"] <= 0:
                mode_rmses[mode] = {"rmse": 999, "errors": []}
                continue

            rf = ms[mode]["slope"]
            intercept = ms[mode]["intercept"]
            vol = 400 if mode == "COLUMN" else 100

            errors = []
            for _, r in sub.iterrows():
                ppm_calc = (r[area_col] - intercept) * 1000 / (rf * vol)
                err_pct = (ppm_calc - r["conc"]) / r["conc"] * 100
                errors.append({"conc": r["conc"], "ppm_calc": ppm_calc, "err_pct": err_pct})

            errs = [e["err_pct"] for e in errors]
            rmse = np.sqrt(np.mean(np.array(errs)**2))
            mae = np.mean(np.abs(errs))
            within_5 = sum(1 for e in errs if abs(e) < 5)
            within_10 = sum(1 for e in errs if abs(e) < 10)
            mode_rmses[mode] = {
                "rmse": rmse, "mae": mae, "errors": errors,
                "within_5": within_5, "within_10": within_10, "n": len(errors),
            }

        res["mode_rmses"] = mode_rmses

        # Consistencia COL-BP amb calibracions SEPARADES
        concs_common = sorted(
            set(grouped[grouped["mode"] == "COLUMN"]["conc"]) &
            set(grouped[grouped["mode"] == "BP"]["conc"])
        )
        sep_diffs = []
        for c in concs_common:
            # ppm amb cal de cada mode
            a_col = grouped[(grouped["mode"] == "COLUMN") & (grouped["conc"] == c)][area_col].mean()
            a_bp = grouped[(grouped["mode"] == "BP") & (grouped["conc"] == c)][area_col].mean()

            rf_c, i_c = ms["COLUMN"]["slope"], ms["COLUMN"]["intercept"]
            rf_b, i_b = ms["BP"]["slope"], ms["BP"]["intercept"]

            ppm_c = (a_col - i_c) * 1000 / (rf_c * 400) if rf_c > 0 else 0
            ppm_b = (a_bp - i_b) * 1000 / (rf_b * 100) if rf_b > 0 else 0

            diff = (ppm_b - ppm_c) / c * 100 if c > 0 else 0
            sep_diffs.append({"conc": c, "ppm_col": ppm_c, "ppm_bp": ppm_b, "diff_pct": diff})

        med_sep = np.median([abs(d["diff_pct"]) for d in sep_diffs]) if sep_diffs else 0
        res["sep_diffs"] = sep_diffs
        res["med_sep_diff"] = med_sep

        print(f"\n  {mname}:")
        print(f"    COLUMN (rf={ms['COLUMN']['slope']:.0f}, i={ms['COLUMN']['intercept']:.1f}): "
              f"RMSE={mode_rmses['COLUMN']['rmse']:.1f}%  MAE={mode_rmses['COLUMN'].get('mae',0):.1f}%  "
              f"<5%: {mode_rmses['COLUMN'].get('within_5',0)}/{mode_rmses['COLUMN'].get('n',0)}  "
              f"<10%: {mode_rmses['COLUMN'].get('within_10',0)}/{mode_rmses['COLUMN'].get('n',0)}")
        print(f"    BP     (rf={ms['BP']['slope']:.0f}, i={ms['BP']['intercept']:.1f}): "
              f"RMSE={mode_rmses['BP']['rmse']:.1f}%  MAE={mode_rmses['BP'].get('mae',0):.1f}%  "
              f"<5%: {mode_rmses['BP'].get('within_5',0)}/{mode_rmses['BP'].get('n',0)}  "
              f"<10%: {mode_rmses['BP'].get('within_10',0)}/{mode_rmses['BP'].get('n',0)}")

        print(f"    COL-BP amb cal separades (mediana |diff|): {med_sep:.1f}%")
        for d in sep_diffs:
            print(f"      {d['conc']:.2f} ppm: COL={d['ppm_col']:.3f}  "
                  f"BP={d['ppm_bp']:.3f}  diff={d['diff_pct']:+.1f}%")

    # =====================================================================
    # RESUM COMPARATIU
    # =====================================================================
    print("\n" + "=" * 130)
    print("RESUM COMPARATIU (calibracions separades per mode)")
    print("=" * 130)
    print(f"\n  {'Metode':30s} {'R2_COL':>7s} {'RMSE_C':>7s} {'i_COL':>6s} "
          f"{'R2_BP':>7s} {'RMSE_B':>7s} {'i_BP':>6s} {'COL-BP':>7s}")
    print("  " + "-" * 95)
    for mname, res in results.items():
        ms = res["mode_stats"]
        mr = res["mode_rmses"]
        print(f"  {mname:30s} "
              f"{ms['COLUMN']['r2']:7.5f} {mr['COLUMN']['rmse']:6.1f}% {ms['COLUMN']['intercept']:6.1f} "
              f"{ms['BP']['r2']:7.5f} {mr['BP']['rmse']:6.1f}% {ms['BP']['intercept']:6.1f} "
              f"{res['med_sep_diff']:6.1f}%")

    # =====================================================================
    # GRAFIC COMPARATIU 3x2
    # =====================================================================
    fig, axes = plt.subplots(2, 3, figsize=(24, 14))
    fig.suptitle("Comparacio metodes d'integracio — Recta de calibracio KHP",
                 fontsize=14, fontweight="bold")

    for ax_idx, (mname, res) in enumerate(results.items()):
        ax = axes[ax_idx // 3, ax_idx % 3]
        area_col = res["area_col"]
        s, i, r2 = res["slope"], res["intercept"], res["r2"]

        # Punts
        for mode, color, marker in [("COLUMN", "#3498DB", "o"), ("BP", "#E67E22", "s")]:
            sub = grouped[grouped["mode"] == mode]
            ax.scatter(sub["ug_c"], sub[area_col], c=color, marker=marker, s=80,
                       alpha=0.8, label=mode, edgecolors="white", zorder=3)

        # Recta unificada
        x_fit = np.linspace(0, 2.2, 100)
        ax.plot(x_fit, s * x_fit + i, "k-", lw=2.5,
                label=f"UNIF: {s:.0f}x+{i:.0f}")

        # Rectes per mode
        for mode, color in [("COLUMN", "#3498DB"), ("BP", "#E67E22")]:
            ms = res["mode_stats"][mode]
            ax.plot(x_fit, ms["slope"] * x_fit + ms["intercept"],
                    color=color, ls="--", lw=1.5, alpha=0.6,
                    label=f"{mode}: {ms['slope']:.0f}x+{ms['intercept']:.0f}")

        # Banda ±10%
        ax.fill_between(x_fit, (s*x_fit+i)*0.9, (s*x_fit+i)*1.1,
                        alpha=0.08, color="gray")

        # Highlight millor
        title_color = "#27AE60" if res["med_col_bp_diff"] == min(r["med_col_bp_diff"] for r in results.values()) else "black"

        ax.set_xlabel("ug C injectat", fontsize=10)
        ax.set_ylabel("Area", fontsize=10)
        ax.set_title(
            f"{mname}\n"
            f"y={s:.0f}x+{i:.0f} R²={r2:.5f} RMSE={res['rmse']:.1f}%\n"
            f"COL-BP: {res['med_col_bp_diff']:.1f}%  |  "
            f"Slope diff: {res['slope_diff_pct']:+.1f}%  |  "
            f"Int diff: {res['intercept_diff']:+.1f}",
            fontsize=8.5, fontweight="bold", color=title_color
        )
        ax.legend(fontsize=6.5, loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, 2.2)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = OUTPUT_DIR / "comparacio_metodes_integracio.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Grafic: {out}")

    return results


def validate_real_samples():
    """
    Validacio amb mostres REALS: mateixa mostra mesurada en COL i BP.
    Parelles: 230_SEQ (COL) + 231_SEQ_BP (BP) — 12 mostres comunes.
    Per cada metode d'integracio, calcula ppm amb cal separades i compara.
    """
    config = get_config()
    from hpsec_calibrate import mode_robust
    from hpsec_core import get_baseline_stats

    # Parelles COL-BP amb mostres comunes
    pairs = [
        {"col_seq": "230_SEQ", "bp_seq": "231_SEQ_BP", "label": "230/231 (PTLL+PTT)"},
        {"col_seq": "168_SEQ", "bp_seq": "169_SEQ_BP", "label": "168/169 (S1-S4)"},
    ]

    # Calibracions per mode (de la comparacio anterior amb KHP)
    # Format: {metode: {mode: {rf, intercept}}}
    # Usem els valors calculats a compare_integration_methods()
    # Pero com no els tenim guardats, els recalculem in-situ

    # --- Funcions d'integracio (copiades de compare_integration_methods) ---
    def find_limits_sigma(t, y, peak_idx, sigma_mult, max_width_min, min_width_min=1.0):
        n = len(y)
        baseline = mode_robust(y)
        y_range = float(np.max(y) - baseline)
        if y_range <= 0:
            return 0, n - 1
        mask_bl = y < (baseline + 0.2 * y_range)
        if np.sum(mask_bl) > 5:
            std_bl = float(np.std(y[mask_bl]))
        else:
            std_bl = float(np.std(y)) * 0.1
        if std_bl < 1e-6:
            std_bl = 0.01 * y_range
        threshold = baseline + sigma_mult * std_bl
        dt = np.mean(np.diff(t)) if n > 1 else 0.01
        max_w_idx = int(max_width_min / dt) if dt > 0 else 300
        min_w_idx = int(min_width_min / dt) if dt > 0 else 50
        li = peak_idx
        for i in range(peak_idx - 1, max(0, peak_idx - max_w_idx), -1):
            if y[i] <= threshold:
                li = i; break
            li = i
        else:
            li = max(0, peak_idx - max_w_idx)
        ri = peak_idx
        for i in range(peak_idx + 1, min(n, peak_idx + max_w_idx)):
            if y[i] <= threshold:
                ri = i; break
            ri = i
        else:
            ri = min(n - 1, peak_idx + max_w_idx)
        if ri - li < min_w_idx:
            expand = (min_w_idx - (ri - li)) // 2 + 1
            li = max(0, li - expand)
            ri = min(n - 1, ri + expand)
        li = max(0, int(min(li, peak_idx - 3)))
        ri = min(n - 1, int(max(ri, peak_idx + 3)))
        return li, ri

    def find_limits_slope(t, y, peak_idx, max_width_min, noise_mult=3.0):
        n = len(y)
        dt_arr = np.diff(t)
        dy_arr = np.diff(y)
        slope = dy_arr / np.maximum(dt_arr, 1e-6)
        baseline = mode_robust(y)
        y_range = float(np.max(y) - baseline)
        mask_flat = y[:-1] < (baseline + 0.15 * y_range)
        if np.sum(mask_flat) > 10:
            slope_noise = float(np.std(slope[mask_flat]))
        else:
            slope_noise = float(np.std(slope)) * 0.1
        if slope_noise < 1e-6:
            slope_noise = 0.01 * float(np.max(np.abs(slope)))
        slope_threshold = noise_mult * slope_noise
        dt = np.mean(dt_arr) if n > 1 else 0.01
        max_w_idx = int(max_width_min / dt) if dt > 0 else 300
        consec_needed = max(3, int(0.1 / dt))
        li = max(0, peak_idx - max_w_idx)
        consec = 0
        for i in range(peak_idx - 1, max(0, peak_idx - max_w_idx), -1):
            if i < len(slope) and abs(slope[i]) < slope_threshold:
                consec += 1
                if consec >= consec_needed:
                    li = i + consec_needed; break
            else:
                consec = 0
        ri = min(n - 1, peak_idx + max_w_idx)
        consec = 0
        for i in range(peak_idx + 1, min(len(slope), peak_idx + max_w_idx)):
            if abs(slope[i]) < slope_threshold:
                consec += 1
                if consec >= consec_needed:
                    ri = i - consec_needed + 1; break
            else:
                consec = 0
        li = max(0, int(min(li, peak_idx - 3)))
        ri = min(n - 1, int(max(ri, peak_idx + 3)))
        return li, ri

    def compute_area(t, y, peak_idx, method_key, is_bp):
        """Calcula area amb un metode donat."""
        max_w = 6.0 if is_bp else 10.0

        if method_key == "3s":
            li, ri = find_limits_sigma(t, y, peak_idx, 3.0, max_w)
            return float(np.trapezoid(y[li:ri+1], t[li:ri+1]))
        elif method_key == "2s":
            li, ri = find_limits_sigma(t, y, peak_idx, 2.0, max_w)
            return float(np.trapezoid(y[li:ri+1], t[li:ri+1]))
        elif method_key == "1s":
            li, ri = find_limits_sigma(t, y, peak_idx, 1.0, max_w)
            return float(np.trapezoid(y[li:ri+1], t[li:ri+1]))
        elif method_key == "slope_v2v":
            li, ri = find_limits_slope(t, y, peak_idx, max_w)
            t_s, y_s = t[li:ri+1], y[li:ri+1]
            bl = np.interp(t_s, [t_s[0], t_s[-1]], [float(y[li]), float(y[ri])])
            return float(np.trapezoid(np.maximum(y_s - bl, 0), t_s))
        elif method_key == "slope_d2z":
            li, ri = find_limits_slope(t, y, peak_idx, max_w)
            return float(np.trapezoid(y[li:ri+1], t[li:ri+1]))
        elif method_key == "3s_v2v":
            li, ri = find_limits_sigma(t, y, peak_idx, 3.0, max_w)
            t_s, y_s = t[li:ri+1], y[li:ri+1]
            bl = np.interp(t_s, [t_s[0], t_s[-1]], [float(y[li]), float(y[ri])])
            return float(np.trapezoid(np.maximum(y_s - bl, 0), t_s))
        return 0.0

    # Calibracions KHP per cada metode (hardcoded dels resultats anteriors)
    # Format: {method: {"COLUMN": (rf, intercept), "BP": (rf, intercept)}}
    cals = {
        "3s":        {"COLUMN": (730, 43.9), "BP": (704, 39.4)},
        "2s":        {"COLUMN": (733, 50.8), "BP": (709, 41.9)},
        "1s":        {"COLUMN": (733, 63.5), "BP": (709, 47.1)},
        "slope_v2v": {"COLUMN": (685, 6.6),  "BP": (606, 21.9)},
        "slope_d2z": {"COLUMN": (735, 34.6), "BP": (713, 33.4)},
        "3s_v2v":    {"COLUMN": (675, 3.8),  "BP": (591, 28.5)},
    }
    method_labels = {
        "3s": "A) 3s actual",
        "2s": "B) 2s",
        "1s": "C) 1s",
        "slope_v2v": "D) Slope+V2V (ChemSt)",
        "slope_d2z": "E) Slope+drop-zero",
        "3s_v2v": "F) 3s+V2V",
    }

    all_results = []

    for pair in pairs:
        print(f"\n{'='*130}")
        print(f"PARELLA: {pair['label']}")
        print(f"  COL: {pair['col_seq']}   BP: {pair['bp_seq']}")
        print(f"{'='*130}")

        # Importar ambdues seqs
        col_data, bp_data = None, None
        for seq_name, mode_label in [(pair["col_seq"], "COL"), (pair["bp_seq"], "BP")]:
            seq_path = str(DADES_DIR / seq_name)
            print(f"\n  Importando {seq_name} ({mode_label})...")
            try:
                imported = import_sequence(seq_path, config)
            except Exception as e:
                print(f"    ERROR: {e}")
                continue
            if not imported or not imported.get("success"):
                print(f"    FALLIT")
                continue
            if mode_label == "COL":
                col_data = imported
            else:
                bp_data = imported

        if not col_data or not bp_data:
            print("  No s'han pogut importar ambdues seqs")
            continue

        col_samples = col_data.get("samples", {})
        bp_samples = bp_data.get("samples", {})

        # Trobar mostres comunes (excloure KHP, BLANK, MQ, NAOH)
        exclude_patterns = ["KHP", "BLANK", "MQ", "NAOH", "CONTROL"]
        def is_real(name):
            upper = name.upper()
            return not any(p in upper for p in exclude_patterns)

        col_names = {n for n in col_samples if is_real(n)}
        bp_names = {n for n in bp_samples if is_real(n)}
        common = sorted(col_names & bp_names)
        print(f"\n  Mostres comunes: {len(common)}")

        if not common:
            continue

        # Per cada mostra comuna, calcular area amb cada metode
        for sample_name in common:
            col_sample = col_samples[sample_name]
            bp_sample = bp_samples[sample_name]

            # Obtenir repliques i fer mean
            for mode, sample, vol, is_bp in [
                ("COLUMN", col_sample, 400, False),
                ("BP", bp_sample, 100, True),
            ]:
                replicas = sample.get("replicas", {})
                rep_areas = {mk: [] for mk in cals}

                for rep_num, rep_data in sorted(replicas.items()):
                    direct = rep_data.get("direct") or {}
                    t = direct.get("t")
                    y_net = direct.get("y_net")
                    if t is None or y_net is None:
                        continue
                    t = np.asarray(t)
                    y_net = np.asarray(y_net)

                    peak_info = detect_main_peak(t, y_net)
                    if not peak_info or not peak_info.get("valid"):
                        continue
                    peak_idx = peak_info["peak_idx"]

                    # Batman repair
                    li3, ri3 = find_limits_sigma(t, y_net, peak_idx, 3.0, 6.0 if is_bp else 10.0)
                    t_seg = t[li3:ri3+1]
                    y_seg = y_net[li3:ri3+1]
                    anomaly = detect_peak_anomaly(t_seg, y_seg)
                    y_use = y_net.copy()
                    if anomaly.get("is_batman", False):
                        y_rep, _, was = repair_with_parabola(t_seg, y_seg)
                        if was:
                            y_use[li3:ri3+1] = y_rep

                    # Calcular area per cada metode
                    for mk in cals:
                        a = compute_area(t, y_use, peak_idx, mk, is_bp)
                        rep_areas[mk].append(a)

                for mk in cals:
                    if rep_areas[mk]:
                        area_mean = np.mean(rep_areas[mk])
                        rf, intercept = cals[mk][mode]
                        ppm = (area_mean - intercept) * 1000 / (rf * vol)
                        all_results.append({
                            "pair": pair["label"],
                            "sample": sample_name,
                            "mode": mode,
                            "method": mk,
                            "area": area_mean,
                            "ppm": ppm,
                            "n_rep": len(rep_areas[mk]),
                        })

    df = pd.DataFrame(all_results)
    if df.empty:
        print("  No hi ha resultats!")
        return

    # =====================================================================
    # COMPARACIO COL vs BP per mostra i metode
    # =====================================================================
    print("\n" + "=" * 130)
    print("RESULTATS: ppm COL vs BP per mostra (amb calibracions separades)")
    print("=" * 130)

    for mk in cals:
        label = method_labels[mk]
        sub = df[df["method"] == mk]

        col_sub = sub[sub["mode"] == "COLUMN"].set_index("sample")
        bp_sub = sub[sub["mode"] == "BP"].set_index("sample")

        common_samples = sorted(set(col_sub.index) & set(bp_sub.index))
        if not common_samples:
            continue

        diffs = []
        print(f"\n  {label}:")
        print(f"    {'Mostra':15s} {'ppm_COL':>8s} {'ppm_BP':>8s} {'diff':>8s} {'diff%':>7s}")
        print(f"    {'-'*50}")

        for s in common_samples:
            ppm_c = col_sub.loc[s, "ppm"]
            ppm_b = bp_sub.loc[s, "ppm"]
            if isinstance(ppm_c, pd.Series):
                ppm_c = ppm_c.iloc[0]
            if isinstance(ppm_b, pd.Series):
                ppm_b = ppm_b.iloc[0]
            diff = ppm_b - ppm_c
            diff_pct = diff / ppm_c * 100 if ppm_c != 0 else 0
            diffs.append({"sample": s, "col": ppm_c, "bp": ppm_b, "diff": diff, "pct": diff_pct})
            print(f"    {s:15s} {ppm_c:8.3f} {ppm_b:8.3f} {diff:+8.3f} {diff_pct:+6.1f}%")

        abs_pcts = [abs(d["pct"]) for d in diffs]
        print(f"    {'':15s} {'':8s} {'':8s} {'mediana':>8s} {np.median(abs_pcts):+6.1f}%")
        print(f"    {'':15s} {'':8s} {'':8s} {'mean':>8s} {np.mean(abs_pcts):+6.1f}%")

    # =====================================================================
    # RESUM FINAL
    # =====================================================================
    print("\n" + "=" * 130)
    print("RESUM: Consistencia COL-BP amb mostres REALS")
    print("=" * 130)
    print(f"\n  {'Metode':30s} {'Mediana |diff|':>15s} {'Mean |diff|':>12s} {'Max |diff|':>12s} {'n':>4s}")
    print(f"  {'-'*80}")

    summary = []
    for mk in cals:
        label = method_labels[mk]
        sub = df[df["method"] == mk]
        col_sub = sub[sub["mode"] == "COLUMN"].set_index("sample")
        bp_sub = sub[sub["mode"] == "BP"].set_index("sample")
        common_samples = sorted(set(col_sub.index) & set(bp_sub.index))

        pcts = []
        for s in common_samples:
            ppm_c = col_sub.loc[s, "ppm"]
            ppm_b = bp_sub.loc[s, "ppm"]
            if isinstance(ppm_c, pd.Series):
                ppm_c = ppm_c.iloc[0]
            if isinstance(ppm_b, pd.Series):
                ppm_b = ppm_b.iloc[0]
            if ppm_c != 0:
                pcts.append(abs((ppm_b - ppm_c) / ppm_c * 100))

        if pcts:
            med = np.median(pcts)
            mn = np.mean(pcts)
            mx = np.max(pcts)
            summary.append({"method": label, "median": med, "mean": mn, "max": mx, "n": len(pcts)})
            print(f"  {label:30s} {med:14.1f}% {mn:11.1f}% {mx:11.1f}% {len(pcts):4d}")

    # =====================================================================
    # GRAFIC: ppm COL vs BP per metode
    # =====================================================================
    n_methods = len(cals)
    fig, axes = plt.subplots(2, 3, figsize=(24, 14))
    fig.suptitle("Validacio amb mostres reals: ppm COLUMN vs ppm BP\n"
                 "(cada punt = una mostra real mesurada en ambdos modes)",
                 fontsize=14, fontweight="bold")

    for ax_idx, mk in enumerate(cals):
        ax = axes[ax_idx // 3, ax_idx % 3]
        label = method_labels[mk]
        sub = df[df["method"] == mk]

        col_sub = sub[sub["mode"] == "COLUMN"].set_index("sample")
        bp_sub = sub[sub["mode"] == "BP"].set_index("sample")
        common_samples = sorted(set(col_sub.index) & set(bp_sub.index))

        ppms_col, ppms_bp, labels = [], [], []
        for s in common_samples:
            pc = col_sub.loc[s, "ppm"]
            pb = bp_sub.loc[s, "ppm"]
            if isinstance(pc, pd.Series): pc = pc.iloc[0]
            if isinstance(pb, pd.Series): pb = pb.iloc[0]
            ppms_col.append(pc)
            ppms_bp.append(pb)
            labels.append(s)

        ppms_col = np.array(ppms_col)
        ppms_bp = np.array(ppms_bp)

        # Plot
        ax.scatter(ppms_col, ppms_bp, c="#3498DB", s=80, alpha=0.8, edgecolors="white", zorder=3)
        # Linia ideal
        max_v = max(ppms_col.max(), ppms_bp.max()) * 1.1
        min_v = min(0, min(ppms_col.min(), ppms_bp.min()) - 0.1)
        ax.plot([min_v, max_v], [min_v, max_v], "k--", lw=1.5, alpha=0.5, label="Ideal (1:1)")
        # Banda +-10%
        ax.fill_between([min_v, max_v], [min_v*0.9, max_v*0.9], [min_v*1.1, max_v*1.1],
                        alpha=0.08, color="green")

        # Stats
        diffs_pct = np.abs((ppms_bp - ppms_col) / np.maximum(np.abs(ppms_col), 0.001) * 100)
        med = np.median(diffs_pct)
        within_10 = np.sum(diffs_pct < 10)

        # Regression
        if len(ppms_col) > 2:
            sl, ic, r, _, _ = stats.linregress(ppms_col, ppms_bp)
            ax.plot([min_v, max_v], [sl*min_v+ic, sl*max_v+ic], "r-", lw=1.5, alpha=0.5,
                    label=f"Fit: {sl:.2f}x+{ic:.3f}")

        title_color = "#27AE60" if med == min(s["median"] for s in summary) else "black"
        ax.set_xlabel("ppm COLUMN", fontsize=10)
        ax.set_ylabel("ppm BP", fontsize=10)
        ax.set_title(f"{label}\nMediana |diff|={med:.1f}%  <10%: {within_10}/{len(ppms_col)}",
                     fontsize=10, fontweight="bold", color=title_color)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(min_v, max_v)
        ax.set_ylim(min_v, max_v)
        ax.set_aspect("equal")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = OUTPUT_DIR / "validacio_real_col_vs_bp.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Grafic: {out}")


def validate_total_area():
    """
    Validacio amb AREA TOTAL (com fa la Suite realment).
    No busca un pic — integra tot el cromatograma amb trapezoid(max(y_net, 0), t).
    Usa la calibracio real de la Suite: COL rf=628 int=81, BP rf=915 int=0.
    """
    config = get_config()

    # Calibracio activa de la Suite
    # COLUMN direct: rf=628, intercept=81, vol=400 uL
    # BP direct: rf=915, intercept=0, vol=100 uL
    cal_params = {
        "COLUMN": {"rf": 628, "intercept": 81, "vol": 400},
        "BP":     {"rf": 915, "intercept": 0,  "vol": 100},
    }

    pairs = [
        {"col_seq": "230_SEQ", "bp_seq": "231_SEQ_BP", "label": "230/231 (PTLL+PTT)"},
        {"col_seq": "168_SEQ", "bp_seq": "169_SEQ_BP", "label": "168/169 (S1-S4)"},
    ]

    all_results = []

    for pair in pairs:
        print(f"\n{'='*100}")
        print(f"PARELLA: {pair['label']}")
        print(f"{'='*100}")

        col_data, bp_data = None, None
        for seq_name, mode_label in [(pair["col_seq"], "COL"), (pair["bp_seq"], "BP")]:
            seq_path = str(DADES_DIR / seq_name)
            print(f"  Importando {seq_name} ({mode_label})...")
            try:
                imported = import_sequence(seq_path, config)
            except Exception as e:
                print(f"    ERROR: {e}")
                continue
            if not imported or not imported.get("success"):
                print(f"    FALLIT")
                continue
            if mode_label == "COL":
                col_data = imported
            else:
                bp_data = imported

        if not col_data or not bp_data:
            print("  No s'han pogut importar ambdues seqs")
            continue

        col_samples = col_data.get("samples", {})
        bp_samples = bp_data.get("samples", {})

        # Trobar mostres comunes (excloure KHP, BLANK, MQ, NAOH, CONTROL)
        exclude_patterns = ["KHP", "BLANK", "MQ", "NAOH", "CONTROL"]
        def is_real(name):
            upper = name.upper()
            return not any(p in upper for p in exclude_patterns)

        col_names = {n for n in col_samples if is_real(n)}
        bp_names = {n for n in bp_samples if is_real(n)}
        common = sorted(col_names & bp_names)
        print(f"  Mostres comunes: {len(common)} -> {common}")

        if not common:
            continue

        print(f"\n  {'Mostra':15s} {'area_COL':>10s} {'ppm_COL':>8s} {'area_BP':>10s} {'ppm_BP':>8s} {'diff%':>7s}")
        print(f"  {'-'*65}")

        for sample_name in common:
            for mode_key, samples_dict in [("COLUMN", col_samples), ("BP", bp_samples)]:
                sample = samples_dict[sample_name]
                replicas = sample.get("replicas", {})
                cp = cal_params[mode_key]

                rep_areas = []
                for rep_num, rep_data in sorted(replicas.items()):
                    direct = rep_data.get("direct") or {}
                    t = direct.get("t")
                    y_net = direct.get("y_net")
                    if t is None or y_net is None:
                        continue
                    t = np.asarray(t, dtype=float)
                    y_net = np.asarray(y_net, dtype=float)

                    # EXACTAMENT com la Suite: max(y, 0) + trapezoid
                    y_clean = np.maximum(y_net, 0)
                    area_total = float(np.trapezoid(y_clean, t))
                    rep_areas.append(area_total)

                if rep_areas:
                    area_mean = float(np.mean(rep_areas))
                    ppm = max(0, area_mean - cp["intercept"]) * 1000 / (cp["rf"] * cp["vol"])
                    all_results.append({
                        "pair": pair["label"],
                        "sample": sample_name,
                        "mode": mode_key,
                        "area": area_mean,
                        "ppm": ppm,
                        "n_rep": len(rep_areas),
                    })

        # Imprimir resultats d'aquesta parella
        for sample_name in common:
            r_col = [r for r in all_results if r["sample"] == sample_name and r["mode"] == "COLUMN"]
            r_bp = [r for r in all_results if r["sample"] == sample_name and r["mode"] == "BP"]
            if r_col and r_bp:
                rc, rb = r_col[-1], r_bp[-1]
                diff_pct = (rb["ppm"] - rc["ppm"]) / rc["ppm"] * 100 if rc["ppm"] > 0 else 0
                print(f"  {sample_name:15s} {rc['area']:10.1f} {rc['ppm']:8.3f} {rb['area']:10.1f} {rb['ppm']:8.3f} {diff_pct:+6.1f}%")

    df = pd.DataFrame(all_results)
    if df.empty:
        print("  No hi ha resultats!")
        return

    # =====================================================================
    # RESUM
    # =====================================================================
    col_df = df[df["mode"] == "COLUMN"].set_index("sample")
    bp_df = df[df["mode"] == "BP"].set_index("sample")
    common_all = sorted(set(col_df.index) & set(bp_df.index))

    ppms_col, ppms_bp, names = [], [], []
    for s in common_all:
        pc = col_df.loc[s, "ppm"]
        pb = bp_df.loc[s, "ppm"]
        if isinstance(pc, pd.Series): pc = pc.iloc[0]
        if isinstance(pb, pd.Series): pb = pb.iloc[0]
        ppms_col.append(pc)
        ppms_bp.append(pb)
        names.append(s)

    ppms_col = np.array(ppms_col)
    ppms_bp = np.array(ppms_bp)
    diffs_pct = np.abs((ppms_bp - ppms_col) / np.maximum(np.abs(ppms_col), 0.001) * 100)

    print(f"\n{'='*100}")
    print(f"RESUM: AREA TOTAL (metode Suite) — Calibracio real (COL rf=628/int=81, BP rf=915/int=0)")
    print(f"{'='*100}")
    print(f"  Mostres: {len(common_all)}")
    print(f"  Mediana |diff COL-BP|: {np.median(diffs_pct):.1f}%")
    print(f"  Mean |diff COL-BP|:    {np.mean(diffs_pct):.1f}%")
    print(f"  Max |diff COL-BP|:     {np.max(diffs_pct):.1f}%")
    print(f"  Min |diff COL-BP|:     {np.min(diffs_pct):.1f}%")
    within_10 = np.sum(diffs_pct < 10)
    within_20 = np.sum(diffs_pct < 20)
    within_30 = np.sum(diffs_pct < 30)
    print(f"  Dins +-10%: {within_10}/{len(common_all)}")
    print(f"  Dins +-20%: {within_20}/{len(common_all)}")
    print(f"  Dins +-30%: {within_30}/{len(common_all)}")

    # Regression COL vs BP
    if len(ppms_col) > 2:
        sl, ic, r, p, se = stats.linregress(ppms_col, ppms_bp)
        print(f"\n  Regressio ppm_BP = {sl:.3f} * ppm_COL + {ic:.3f}")
        print(f"  R2 = {r**2:.4f}, p = {p:.2e}")

    # =====================================================================
    # GRAFIC
    # =====================================================================
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    fig.suptitle("AREA TOTAL (metode Suite real)\nCalibracio: COL rf=628/int=81, BP rf=915/int=0",
                 fontsize=14, fontweight="bold")

    ax.scatter(ppms_col, ppms_bp, c="#3498DB", s=120, alpha=0.8, edgecolors="white", zorder=3)

    # Etiquetar cada punt
    for i, name in enumerate(names):
        ax.annotate(name, (ppms_col[i], ppms_bp[i]), fontsize=7,
                    textcoords="offset points", xytext=(5, 5), alpha=0.7)

    max_v = max(ppms_col.max(), ppms_bp.max()) * 1.15
    min_v = max(0, min(ppms_col.min(), ppms_bp.min()) * 0.85)

    # Linia ideal 1:1
    ax.plot([min_v, max_v], [min_v, max_v], "g--", lw=2, alpha=0.6, label="Ideal (1:1)")
    # Banda +-10%
    ax.fill_between([min_v, max_v], [min_v*0.9, max_v*0.9], [min_v*1.1, max_v*1.1],
                    alpha=0.08, color="green", label="+-10%")
    # Banda +-20%
    ax.fill_between([min_v, max_v], [min_v*0.8, max_v*0.8], [min_v*1.2, max_v*1.2],
                    alpha=0.05, color="orange", label="+-20%")

    # Regressio
    if len(ppms_col) > 2:
        sl, ic, r, _, _ = stats.linregress(ppms_col, ppms_bp)
        ax.plot([min_v, max_v], [sl*min_v+ic, sl*max_v+ic], "r-", lw=2, alpha=0.6,
                label=f"Fit: {sl:.2f}x + {ic:.3f} (R2={r**2:.3f})")

    med = np.median(diffs_pct)
    ax.set_xlabel("ppm COLUMN", fontsize=12)
    ax.set_ylabel("ppm BP", fontsize=12)
    ax.set_title(f"n={len(common_all)} mostres reals | Mediana |diff|={med:.1f}% | <10%: {within_10}/{len(common_all)} | <20%: {within_20}/{len(common_all)}",
                 fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(min_v, max_v)
    ax.set_ylim(min_v, max_v)
    ax.set_aspect("equal")

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    out = OUTPUT_DIR / "validacio_area_total_suite.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Grafic: {out}")


def validate_suite_analyze():
    """
    Validacio fent servir analyze_sequence() de la Suite directament.
    Compara ppm COL vs BP per mostres reals mesurades en ambdos modes.
    """
    from hpsec_analyze import analyze_sequence
    config = get_config()

    pairs = [
        {"col_seq": "230_SEQ", "bp_seq": "231_SEQ_BP", "label": "230/231 (PTLL+PTT)"},
        {"col_seq": "168_SEQ", "bp_seq": "169_SEQ_BP", "label": "168/169 (S1-S4)"},
    ]

    all_results = []

    for pair in pairs:
        print(f"\n{'='*100}")
        print(f"PARELLA: {pair['label']}")
        print(f"{'='*100}")

        analyzed = {}
        for seq_name, mode_label in [(pair["col_seq"], "COL"), (pair["bp_seq"], "BP")]:
            seq_path = str(DADES_DIR / seq_name)
            print(f"  Importando + analitzant {seq_name} ({mode_label})...")
            try:
                imported = import_sequence(seq_path, config)
                if not imported or not imported.get("success"):
                    print(f"    Import FALLIT")
                    continue
                result = analyze_sequence(imported, calibration_data=None, config=config)
                if not result or not result.get("success"):
                    print(f"    Analyze FALLIT")
                    continue
                analyzed[mode_label] = result
                n_grouped = len(result.get('samples_grouped', {}))
                print(f"    OK: {n_grouped} mostres agrupades, method={result.get('method')}")
            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback; traceback.print_exc()
                continue

        if "COL" not in analyzed or "BP" not in analyzed:
            print("  No s'han pogut analitzar ambdues seqs")
            continue

        col_grouped = analyzed["COL"].get("samples_grouped", {})
        bp_grouped = analyzed["BP"].get("samples_grouped", {})

        # Trobar mostres comunes (excloure KHP, BLANK, MQ, NAOH, CONTROL)
        exclude_patterns = ["KHP", "BLANK", "MQ", "NAOH", "CONTROL"]
        def is_real(name):
            upper = name.upper()
            return not any(p in upper for p in exclude_patterns)

        col_names = {n for n in col_grouped if is_real(n)}
        bp_names = {n for n in bp_grouped if is_real(n)}
        common = sorted(col_names & bp_names)
        print(f"\n  Mostres comunes: {len(common)} -> {common}")

        if not common:
            # Debug: mostrar noms disponibles
            print(f"  COL noms: {sorted(col_names)}")
            print(f"  BP noms:  {sorted(bp_names)}")
            continue

        print(f"\n  {'Mostra':15s} {'ppm_COL':>8s} {'area_COL':>10s} {'ppm_BP':>8s} {'area_BP':>10s} {'diff%':>7s} {'valid_C':>7s} {'valid_B':>7s}")
        print(f"  {'-'*85}")

        for sample_name in common:
            col_group = col_grouped[sample_name]
            bp_group = bp_grouped[sample_name]

            col_q = col_group.get("quantification") or {}
            bp_q = bp_group.get("quantification") or {}

            ppm_col = col_q.get("concentration_ppm")
            ppm_bp = bp_q.get("concentration_ppm")
            area_col = col_q.get("area_total")
            area_bp = bp_q.get("area_total")
            valid_c = col_group.get("sample_valid", True)
            valid_b = bp_group.get("sample_valid", True)

            if ppm_col is not None and ppm_bp is not None and ppm_col > 0:
                diff_pct = (ppm_bp - ppm_col) / ppm_col * 100
                print(f"  {sample_name:15s} {ppm_col:8.3f} {area_col or 0:10.1f} {ppm_bp:8.3f} {area_bp or 0:10.1f} {diff_pct:+6.1f}% {'OK' if valid_c else 'NO':>7s} {'OK' if valid_b else 'NO':>7s}")
                all_results.append({
                    "pair": pair["label"],
                    "sample": sample_name,
                    "ppm_col": ppm_col,
                    "ppm_bp": ppm_bp,
                    "area_col": area_col or 0,
                    "area_bp": area_bp or 0,
                    "valid_col": valid_c,
                    "valid_bp": valid_b,
                })
            else:
                ppm_c_str = f"{ppm_col:.3f}" if ppm_col is not None else "None"
                ppm_b_str = f"{ppm_bp:.3f}" if ppm_bp is not None else "None"
                print(f"  {sample_name:15s} {ppm_c_str:>8s} {'---':>10s} {ppm_b_str:>8s} {'---':>10s} {'---':>7s} {'OK' if valid_c else 'NO':>7s} {'OK' if valid_b else 'NO':>7s}")

    if not all_results:
        print("  No hi ha resultats!")
        return

    df = pd.DataFrame(all_results)

    # =====================================================================
    # RESUM
    # =====================================================================
    ppms_col = df["ppm_col"].values
    ppms_bp = df["ppm_bp"].values
    names = df["sample"].tolist()
    diffs_pct = np.abs((ppms_bp - ppms_col) / np.maximum(np.abs(ppms_col), 0.001) * 100)

    within_10 = int(np.sum(diffs_pct < 10))
    within_20 = int(np.sum(diffs_pct < 20))
    within_30 = int(np.sum(diffs_pct < 30))

    print(f"\n{'='*100}")
    print(f"RESUM: analyze_sequence() de la Suite (pipeline complet)")
    print(f"{'='*100}")
    print(f"  Mostres: {len(df)}")
    print(f"  Mediana |diff COL-BP|: {np.median(diffs_pct):.1f}%")
    print(f"  Mean |diff COL-BP|:    {np.mean(diffs_pct):.1f}%")
    print(f"  Max |diff COL-BP|:     {np.max(diffs_pct):.1f}%")
    print(f"  Min |diff COL-BP|:     {np.min(diffs_pct):.1f}%")
    print(f"  Dins +-10%: {within_10}/{len(df)}")
    print(f"  Dins +-20%: {within_20}/{len(df)}")
    print(f"  Dins +-30%: {within_30}/{len(df)}")

    if len(ppms_col) > 2:
        sl, ic, r, p, se = stats.linregress(ppms_col, ppms_bp)
        print(f"\n  Regressio ppm_BP = {sl:.3f} * ppm_COL + {ic:.3f}")
        print(f"  R2 = {r**2:.4f}, p = {p:.2e}")

    # =====================================================================
    # GRAFIC
    # =====================================================================
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    fig.suptitle("Validacio COL vs BP: analyze_sequence() de la Suite\n"
                 "(calibracio global: COL rf=628/int=81, BP rf=915/int=0)",
                 fontsize=13, fontweight="bold")

    ax.scatter(ppms_col, ppms_bp, c="#3498DB", s=120, alpha=0.8, edgecolors="white", zorder=3)

    for i, name in enumerate(names):
        ax.annotate(name, (ppms_col[i], ppms_bp[i]), fontsize=7,
                    textcoords="offset points", xytext=(5, 5), alpha=0.7)

    max_v = max(ppms_col.max(), ppms_bp.max()) * 1.15
    min_v = max(0, min(ppms_col.min(), ppms_bp.min()) * 0.85)

    ax.plot([min_v, max_v], [min_v, max_v], "g--", lw=2, alpha=0.6, label="Ideal (1:1)")
    ax.fill_between([min_v, max_v], [min_v*0.9, max_v*0.9], [min_v*1.1, max_v*1.1],
                    alpha=0.08, color="green", label="+-10%")
    ax.fill_between([min_v, max_v], [min_v*0.8, max_v*0.8], [min_v*1.2, max_v*1.2],
                    alpha=0.05, color="orange", label="+-20%")

    if len(ppms_col) > 2:
        sl, ic, r, _, _ = stats.linregress(ppms_col, ppms_bp)
        ax.plot([min_v, max_v], [sl*min_v+ic, sl*max_v+ic], "r-", lw=2, alpha=0.6,
                label=f"Fit: {sl:.2f}x + {ic:.3f} (R2={r**2:.3f})")

    med = np.median(diffs_pct)
    ax.set_xlabel("ppm COLUMN", fontsize=12)
    ax.set_ylabel("ppm BP", fontsize=12)
    ax.set_title(f"n={len(df)} mostres | Med |diff|={med:.1f}% | <10%: {within_10}/{len(df)} | <20%: {within_20}/{len(df)}",
                 fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(min_v, max_v)
    ax.set_ylim(min_v, max_v)
    ax.set_aspect("equal")

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    out = OUTPUT_DIR / "validacio_suite_analyze.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Grafic: {out}")


if __name__ == "__main__":
    print("=" * 130)
    print("VALIDACIO AMB analyze_sequence() DE LA SUITE")
    print("=" * 130)
    validate_suite_analyze()
