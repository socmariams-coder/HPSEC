"""
Anàlisi comparativa COLUMN vs BP per la mateixa mostra.

Carrega analysis_result.json de parells COLUMN/BP, aparella mostres
pel nom, i genera taules + gràfics comparatius.

Mètriques comparades:
- ppm calculat (àrea → ppm amb calibració vigent)
- Àrea DOC total normalitzada per volum (àrea/µL)
- Àrea A254 (DAD) normalitzada per volum
- SNR DOC i SNR 254
- Fraccions DOC (BioP, HS, BB, SB, LMW) en % — perfil cromatogràfic
- Reproducibilitat entre rèpliques (CV%)
- Correlació COLUMN vs BP

Aparellament per proximitat: COLUMN i BP amb ≤MAX_SEQ_GAP de diferència
en número de SEQ per evitar falsos matchs temporals.
"""

import json
import os
import re
import sys
import numpy as np

# Fix Windows console encoding
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ---------- config ----------
DATA_DIR = r"C:\Users\Lequia\Desktop\Dades3"
CAL_REF = os.path.join(DATA_DIR, "REGISTRY", "Calibration_Reference.json")
OUT_DIR = os.path.join(DATA_DIR, "REGISTRY", "review", "column_vs_bp")

MAX_SEQ_GAP = 5  # Màxim gap entre números de SEQ per considerar un parell

# Calibration defaults
RF_COL = 752.90
RF_BP = 646.82
INTERCEPT_COL = 18.11
INTERCEPT_BP = 2.87
VOL_COL = 400.0  # µL
VOL_BP = 100.0

# Exclusions — SEQs amb problemes coneguts
EXCLUDE_SEQS = {
    "107_SEQ_BP_CAL",   # saturació
    "271_SEQ_BP",       # àrees absurdes
}

# Exclude sample name patterns (blancs, KHP, controls)
EXCLUDE_PATTERNS = ("MQ", "NAOH", "BUFFER", "KHP", "NAN")


def load_calibration():
    """Llegir RF i intercept del Calibration_Reference.json."""
    global RF_COL, RF_BP, INTERCEPT_COL, INTERCEPT_BP
    if not os.path.isfile(CAL_REF):
        print(f"[WARN] No es troba {CAL_REF}, usant valors per defecte")
        return
    with open(CAL_REF, encoding="utf-8") as f:
        cal = json.load(f)
    aid = cal.get("active_calibration_ids", {}).get("direct")
    for c in cal.get("calibrations", []):
        if c.get("id") == aid:
            rf = c.get("rf_mass_cal", {})
            ic = c.get("intercept", {})
            RF_COL = rf.get("column", RF_COL)
            RF_BP = rf.get("bp", RF_BP)
            INTERCEPT_COL = ic.get("column", INTERCEPT_COL)
            INTERCEPT_BP = ic.get("bp", INTERCEPT_BP)
            print(f"Calibracio activa: RF_COL={RF_COL:.1f} IC_COL={INTERCEPT_COL:.1f}"
                  f" | RF_BP={RF_BP:.1f} IC_BP={INTERCEPT_BP:.1f}")
            return
    print("[WARN] No s'ha trobat la calibracio activa")


def area_to_ppm(area, method):
    """Convertir area DOC a ppm amb la calibracio vigent."""
    if method == "COLUMN":
        return (area - INTERCEPT_COL) * 1000 / (RF_COL * VOL_COL)
    else:
        return (area - INTERCEPT_BP) * 1000 / (RF_BP * VOL_BP)


def extract_seq_num(name):
    """Extreure número de SEQ del nom (e.g. '285_SEQ' -> 285)."""
    m = re.match(r"(\d+)", name)
    return int(m.group(1)) if m else 0


def load_all_results():
    """Carregar tots els analysis_result.json disponibles."""
    results = {}
    for d in sorted(os.listdir(DATA_DIR)):
        if d in EXCLUDE_SEQS:
            continue
        f = os.path.join(DATA_DIR, d, "CHECK", "data", "analysis_result.json")
        if os.path.isfile(f):
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)
            results[d] = {
                "method": data.get("method", "?"),
                "data_mode": data.get("data_mode", "?"),
                "samples": data.get("samples", []),
                "seq_num": extract_seq_num(d),
            }
    return results


def find_pairs(results):
    """Trobar parells COLUMN/BP amb mostres en comú dins proximitat temporal."""
    col_seqs = {k: v for k, v in results.items()
                if v["method"] == "COLUMN" and "_CAL" not in k.upper()}
    bp_seqs = {k: v for k, v in results.items()
               if v["method"] == "BP" and "_CAL" not in k.upper()}

    pairs = []
    for ck, cv in col_seqs.items():
        col_num = cv["seq_num"]
        col_names = set()
        for s in cv["samples"]:
            n = s["name"]
            if not n.upper().startswith(EXCLUDE_PATTERNS):
                col_names.add(n)

        # Trobar BP més propera amb mostres compartides
        best_bp = None
        best_shared = set()
        best_gap = 999

        for bk, bv in bp_seqs.items():
            gap = abs(bv["seq_num"] - col_num)
            if gap > MAX_SEQ_GAP:
                continue
            bp_names = set()
            for s in bv["samples"]:
                n = s["name"]
                if not n.upper().startswith(EXCLUDE_PATTERNS):
                    bp_names.add(n)
            shared = col_names & bp_names
            if shared and (gap < best_gap or (gap == best_gap and len(shared) > len(best_shared))):
                best_bp = bk
                best_shared = shared
                best_gap = gap

        if best_bp and best_shared:
            pairs.append((ck, best_bp, sorted(best_shared)))

    return pairs


def extract_sample_data(samples_list, sample_name):
    """Extreure repliques d'una mostra per nom."""
    reps = []
    for s in samples_list:
        if s["name"] == sample_name:
            doc = s.get("areas", {}).get("DOC", {})
            a254 = s.get("areas", {}).get("A254", {})
            pi = s.get("peak_info", {})
            snr = s.get("snr_info", {})
            snr254 = s.get("snr_info_dad", {}).get("A254", {})
            vol = s.get("inj_volume", None)
            area_total = doc.get("total", 0) or 0

            # Fraccions en % (respecte total)
            fracs = {}
            for frac_name in ["BioP", "HS", "BB", "SB", "LMW"]:
                frac_area = doc.get(frac_name, 0) or 0
                fracs[frac_name] = (frac_area / area_total * 100) if area_total > 0 else 0

            reps.append({
                "replica": s.get("replica", "?"),
                "area_doc": area_total,
                "area_a254": a254.get("total", 0) or 0,
                "t_max": pi.get("t_max", None),
                "height": pi.get("height", 0) or 0,
                "snr_doc": snr.get("snr_direct", 0) or 0,
                "snr_254": snr254.get("snr", 0) or 0,
                "inj_volume": vol,
                "BioP": doc.get("BioP", 0) or 0,
                "HS": doc.get("HS", 0) or 0,
                "BB": doc.get("BB", 0) or 0,
                "SB": doc.get("SB", 0) or 0,
                "LMW": doc.get("LMW", 0) or 0,
                "fracs_pct": fracs,
            })
    return reps


def mean_std(vals):
    arr = np.array([v for v in vals if v is not None and v > 0])
    if len(arr) == 0:
        return 0.0, 0.0
    return float(np.mean(arr)), float(np.std(arr))


def main():
    load_calibration()
    results = load_all_results()
    pairs = find_pairs(results)

    if not pairs:
        print("No s'han trobat parells COLUMN/BP amb mostres compartides.")
        return

    print(f"\n{'='*80}")
    print(f"COMPARACIO COLUMN vs BP  (proximitat <= {MAX_SEQ_GAP} SEQs)")
    print(f"{'='*80}")

    os.makedirs(OUT_DIR, exist_ok=True)

    all_rows = []

    for col_seq, bp_seq, shared_names in sorted(pairs, key=lambda p: extract_seq_num(p[0])):
        col_samples = results[col_seq]["samples"]
        bp_samples = results[bp_seq]["samples"]

        for name in shared_names:
            col_reps = extract_sample_data(col_samples, name)
            bp_reps = extract_sample_data(bp_samples, name)
            if not col_reps or not bp_reps:
                continue

            col_area_mean, col_area_std = mean_std([r["area_doc"] for r in col_reps])
            bp_area_mean, bp_area_std = mean_std([r["area_doc"] for r in bp_reps])
            col_a254_mean, _ = mean_std([r["area_a254"] for r in col_reps])
            bp_a254_mean, _ = mean_std([r["area_a254"] for r in bp_reps])
            col_snr_mean, _ = mean_std([r["snr_doc"] for r in col_reps])
            bp_snr_mean, _ = mean_std([r["snr_doc"] for r in bp_reps])

            # Skip if either area is zero
            if col_area_mean <= 0 or bp_area_mean <= 0:
                continue

            col_ppm = area_to_ppm(col_area_mean, "COLUMN")
            bp_ppm = area_to_ppm(bp_area_mean, "BP")

            col_area_norm = col_area_mean / VOL_COL
            bp_area_norm = bp_area_mean / VOL_BP
            ratio = bp_area_norm / col_area_norm if col_area_norm > 0 else float("nan")

            col_cv = (col_area_std / col_area_mean * 100) if col_area_mean > 0 else 0
            bp_cv = (bp_area_std / bp_area_mean * 100) if bp_area_mean > 0 else 0

            # Fraccions mitjanes
            fracs_col = {}
            fracs_bp = {}
            for fn in ["BioP", "HS", "BB", "SB", "LMW"]:
                fracs_col[fn] = np.mean([r["fracs_pct"][fn] for r in col_reps])
                fracs_bp[fn] = np.mean([r["fracs_pct"][fn] for r in bp_reps])

            all_rows.append({
                "name": name,
                "col_seq": col_seq,
                "bp_seq": bp_seq,
                "col_area": col_area_mean,
                "col_area_std": col_area_std,
                "col_cv": col_cv,
                "bp_area": bp_area_mean,
                "bp_area_std": bp_area_std,
                "bp_cv": bp_cv,
                "col_ppm": col_ppm,
                "bp_ppm": bp_ppm,
                "ratio_area_norm": ratio,
                "col_a254": col_a254_mean,
                "bp_a254": bp_a254_mean,
                "col_snr": col_snr_mean,
                "bp_snr": bp_snr_mean,
                "fracs_col": fracs_col,
                "fracs_bp": fracs_bp,
                "col_reps": col_reps,
                "bp_reps": bp_reps,
            })

    if not all_rows:
        print("Cap dada per comparar.")
        return

    # ---- Print table ----
    print(f"\nN mostres aparellades: {len(all_rows)}")
    print(f"\n{'Mostra':15s} {'COL SEQ':12s} {'BP SEQ':15s} {'ppm COL':>8s} {'ppm BP':>8s} {'Ratio':>6s} {'CV_C%':>6s} {'CV_B%':>6s}")
    print("-" * 95)
    for r in all_rows:
        print(f"{r['name']:15s} {r['col_seq']:12s} {r['bp_seq']:15s} "
              f"{r['col_ppm']:8.3f} {r['bp_ppm']:8.3f} {r['ratio_area_norm']:6.3f} "
              f"{r['col_cv']:6.1f} {r['bp_cv']:6.1f}")

    # ---- Resum estadístic ----
    print(f"\n{'='*80}")
    print("RESUM ESTADISTIC")
    print(f"{'='*80}")

    ratios = np.array([r["ratio_area_norm"] for r in all_rows])
    col_ppms = np.array([r["col_ppm"] for r in all_rows])
    bp_ppms = np.array([r["bp_ppm"] for r in all_rows])

    valid = np.isfinite(ratios)
    ratios_valid = ratios[valid]

    print(f"N mostres comparades: {len(all_rows)}")
    print(f"N parells COL/BP:     {len(pairs)}")
    print(f"\nRatio area normalitzada (BP/COL per uL):")
    print(f"  Mitjana: {np.mean(ratios_valid):.3f} +/- {np.std(ratios_valid):.3f}")
    print(f"  Mediana: {np.median(ratios_valid):.3f}")
    print(f"  P10-P90: {np.percentile(ratios_valid, 10):.3f} - {np.percentile(ratios_valid, 90):.3f}")

    print(f"\nReproducibilitat (CV% entre repliques):")
    col_cvs = [r["col_cv"] for r in all_rows]
    bp_cvs = [r["bp_cv"] for r in all_rows]
    print(f"  COLUMN: {np.mean(col_cvs):.1f}% +/- {np.std(col_cvs):.1f}%")
    print(f"  BP:     {np.mean(bp_cvs):.1f}% +/- {np.std(bp_cvs):.1f}%")

    # Correlació amb i sense outliers
    for label, mask_fn in [
        ("TOTES", lambda r: True),
        ("Ratio 0.5-3.0", lambda r: 0.5 <= r["ratio_area_norm"] <= 3.0),
        ("Ratio 0.8-2.0", lambda r: 0.8 <= r["ratio_area_norm"] <= 2.0),
    ]:
        subset = [r for r in all_rows if mask_fn(r)]
        if len(subset) < 3:
            continue
        cp = np.array([r["col_ppm"] for r in subset])
        bp = np.array([r["bp_ppm"] for r in subset])
        rp = np.corrcoef(cp, bp)[0, 1]
        sl, ic = np.polyfit(cp, bp, 1)
        res = bp - (sl * cp + ic)
        rms = np.sqrt(np.mean(res**2))
        print(f"\nCorrelacio ppm ({label}, n={len(subset)}):")
        print(f"  R2 = {rp**2:.4f}")
        print(f"  BP = {sl:.3f} * COLUMN + {ic:.3f}")
        print(f"  RMS = {rms:.4f} ppm")

    # Outliers
    n_high = np.sum(ratios_valid > 3.0)
    n_low = np.sum(ratios_valid < 0.5)
    print(f"\nOutliers: {n_high} amb ratio > 3.0, {n_low} amb ratio < 0.5")

    # ---- Fraccions DOC ----
    print(f"\n{'='*80}")
    print("FRACCIONS DOC MITJANES (% del total)")
    print(f"{'='*80}")
    print(f"{'Fraccio':>8s} {'COLUMN':>10s} {'BP':>10s} {'Diff':>10s}")
    print("-" * 42)
    for fn in ["BioP", "HS", "BB", "SB", "LMW"]:
        col_f = np.mean([r["fracs_col"][fn] for r in all_rows])
        bp_f = np.mean([r["fracs_bp"][fn] for r in all_rows])
        print(f"{fn:>8s} {col_f:10.1f}% {bp_f:10.1f}% {bp_f - col_f:+10.1f}%")

    # ---- Gràfics ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        # Filtre per als gràfics: ratio raonable
        clean = [r for r in all_rows if 0.3 <= r["ratio_area_norm"] <= 5.0]
        c_ppms = np.array([r["col_ppm"] for r in clean])
        b_ppms = np.array([r["bp_ppm"] for r in clean])

        fig = plt.figure(figsize=(18, 14))
        fig.suptitle(f"Comparacio COLUMN vs BP (n={len(clean)}, proximitat <= {MAX_SEQ_GAP} SEQs)",
                     fontsize=14, fontweight="bold")
        gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

        # 1. Scatter ppm COLUMN vs BP
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(c_ppms, b_ppms, s=30, c="#2196F3", edgecolors="k",
                    linewidth=0.5, alpha=0.7, zorder=5)
        lim = max(c_ppms.max(), b_ppms.max()) * 1.1
        ax1.plot([0, lim], [0, lim], "k--", alpha=0.3, label="1:1")
        if len(c_ppms) > 2:
            rp = np.corrcoef(c_ppms, b_ppms)[0, 1]
            sl, ic = np.polyfit(c_ppms, b_ppms, 1)
            x_fit = np.linspace(0, lim, 100)
            ax1.plot(x_fit, sl * x_fit + ic, "r-", alpha=0.7,
                     label=f"Fit: y={sl:.2f}x{ic:+.2f}\nR2={rp**2:.3f}")
        ax1.set_xlabel("ppm COLUMN")
        ax1.set_ylabel("ppm BP")
        ax1.set_title("Concentracio DOC (ppm)")
        ax1.legend(fontsize=8)
        ax1.set_xlim(0, lim)
        ax1.set_ylim(0, lim)

        # 2. Bland-Altman (diferència % vs mitjana)
        ax2 = fig.add_subplot(gs[0, 1])
        means = (c_ppms + b_ppms) / 2
        diffs_pct = (b_ppms - c_ppms) / means * 100
        ax2.scatter(means, diffs_pct, s=30, c="#FF9800", edgecolors="k",
                    linewidth=0.5, alpha=0.7, zorder=5)
        ax2.axhline(0, color="k", linestyle="--", alpha=0.3)
        mean_diff = np.mean(diffs_pct[np.isfinite(diffs_pct)])
        std_diff = np.std(diffs_pct[np.isfinite(diffs_pct)])
        ax2.axhline(mean_diff, color="r", linestyle="-", alpha=0.5,
                    label=f"Bias: {mean_diff:.1f}%")
        ax2.axhline(mean_diff + 1.96 * std_diff, color="r", linestyle=":", alpha=0.3)
        ax2.axhline(mean_diff - 1.96 * std_diff, color="r", linestyle=":", alpha=0.3)
        ax2.set_xlabel("Mitjana ppm (COL+BP)/2")
        ax2.set_ylabel("Diferencia % (BP-COL)/Mitjana")
        ax2.set_title("Bland-Altman")
        ax2.legend(fontsize=8)

        # 3. Histograma ratios
        ax3 = fig.add_subplot(gs[0, 2])
        clean_ratios = [r["ratio_area_norm"] for r in clean
                        if np.isfinite(r["ratio_area_norm"])]
        ax3.hist(clean_ratios, bins=30, color="#4CAF50", edgecolor="k", alpha=0.8)
        ax3.axvline(1.0, color="k", linestyle="--", alpha=0.5, label="1:1")
        ax3.axvline(np.median(clean_ratios), color="r", linestyle="-", alpha=0.7,
                    label=f"Mediana: {np.median(clean_ratios):.3f}")
        ax3.set_xlabel("Ratio (BP/COL) area per uL")
        ax3.set_ylabel("Comptatge")
        ax3.set_title("Distribucio ratios")
        ax3.legend(fontsize=8)

        # 4. Scatter A254 normalitzada
        ax4 = fig.add_subplot(gs[1, 0])
        c254 = np.array([r["col_a254"] / VOL_COL for r in clean])
        b254 = np.array([r["bp_a254"] / VOL_BP for r in clean])
        has_254 = (c254 > 0) & (b254 > 0)
        if np.sum(has_254) > 2:
            ax4.scatter(c254[has_254], b254[has_254], s=30, c="#9C27B0",
                        edgecolors="k", linewidth=0.5, alpha=0.7, zorder=5)
            lim254 = max(c254[has_254].max(), b254[has_254].max()) * 1.1
            ax4.plot([0, lim254], [0, lim254], "k--", alpha=0.3, label="1:1")
            r254 = np.corrcoef(c254[has_254], b254[has_254])[0, 1]
            sl254, ic254 = np.polyfit(c254[has_254], b254[has_254], 1)
            x254 = np.linspace(0, lim254, 100)
            ax4.plot(x254, sl254 * x254 + ic254, "r-", alpha=0.7,
                     label=f"R2={r254**2:.3f}")
            ax4.set_xlim(0, lim254)
            ax4.set_ylim(0, lim254)
            ax4.legend(fontsize=8)
        ax4.set_xlabel("A254/uL COLUMN")
        ax4.set_ylabel("A254/uL BP")
        ax4.set_title("Absorbancia 254nm")

        # 5. Fraccions DOC stacked bar (COLUMN vs BP)
        ax5 = fig.add_subplot(gs[1, 1])
        frac_names = ["BioP", "HS", "BB", "SB", "LMW"]
        frac_colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]
        col_fracs = [np.mean([r["fracs_col"][fn] for r in clean]) for fn in frac_names]
        bp_fracs = [np.mean([r["fracs_bp"][fn] for r in clean]) for fn in frac_names]

        x_bar = [0, 1]
        bottom_col, bottom_bp = 0, 0
        for i, fn in enumerate(frac_names):
            ax5.bar(0, col_fracs[i], bottom=bottom_col, color=frac_colors[i],
                    edgecolor="k", linewidth=0.5, label=fn)
            ax5.bar(1, bp_fracs[i], bottom=bottom_bp, color=frac_colors[i],
                    edgecolor="k", linewidth=0.5)
            # Label dins la barra
            if col_fracs[i] > 3:
                ax5.text(0, bottom_col + col_fracs[i]/2, f"{col_fracs[i]:.0f}%",
                         ha="center", va="center", fontsize=7)
            if bp_fracs[i] > 3:
                ax5.text(1, bottom_bp + bp_fracs[i]/2, f"{bp_fracs[i]:.0f}%",
                         ha="center", va="center", fontsize=7)
            bottom_col += col_fracs[i]
            bottom_bp += bp_fracs[i]
        ax5.set_xticks([0, 1])
        ax5.set_xticklabels(["COLUMN", "BP"])
        ax5.set_ylabel("% area total")
        ax5.set_title("Perfil fraccions DOC (mitjana)")
        ax5.legend(fontsize=7, loc="upper right")

        # 6. ppm per parell SEQ (boxplot-like)
        ax6 = fig.add_subplot(gs[1, 2])
        # Group by pair
        pair_data = {}
        for r in clean:
            key = f"{r['col_seq']}/{r['bp_seq']}"
            if key not in pair_data:
                pair_data[key] = {"col": [], "bp": [], "ratios": []}
            pair_data[key]["col"].append(r["col_ppm"])
            pair_data[key]["bp"].append(r["bp_ppm"])
            pair_data[key]["ratios"].append(r["ratio_area_norm"])

        pair_names = sorted(pair_data.keys(), key=lambda k: extract_seq_num(k.split("/")[0]))
        pair_medians = [np.median(pair_data[k]["ratios"]) for k in pair_names]
        pair_n = [len(pair_data[k]["ratios"]) for k in pair_names]
        colors_bar = ["#4CAF50" if 0.8 <= m <= 1.5 else "#FF9800" if 0.5 <= m <= 2.0 else "#F44336"
                      for m in pair_medians]
        ax6.barh(range(len(pair_names)), pair_medians, color=colors_bar,
                 edgecolor="k", linewidth=0.5, alpha=0.8)
        ax6.axvline(1.0, color="k", linestyle="--", alpha=0.5)
        ax6.set_yticks(range(len(pair_names)))
        short_names = [f"{k.split('/')[0].replace('_SEQ','')} (n={pair_n[i]})"
                       for i, k in enumerate(pair_names)]
        ax6.set_yticklabels(short_names, fontsize=7)
        ax6.set_xlabel("Ratio mediana BP/COL")
        ax6.set_title("Ratio per parell SEQ")

        # 7. Ratio vs concentracio (test linealitat)
        ax7 = fig.add_subplot(gs[2, 0])
        ax7.scatter(c_ppms, [r["ratio_area_norm"] for r in clean],
                    s=30, c="#2196F3", edgecolors="k", linewidth=0.5, alpha=0.7)
        ax7.axhline(1.0, color="k", linestyle="--", alpha=0.3)
        ax7.set_xlabel("ppm COLUMN")
        ax7.set_ylabel("Ratio BP/COL (area/uL)")
        ax7.set_title("Ratio vs concentracio")
        ax7.set_ylim(0, 4)

        # 8. SNR comparació
        ax8 = fig.add_subplot(gs[2, 1])
        col_snrs = np.array([r["col_snr"] for r in clean])
        bp_snrs = np.array([r["bp_snr"] for r in clean])
        ax8.scatter(col_snrs, bp_snrs, s=30, c="#FF5722", edgecolors="k",
                    linewidth=0.5, alpha=0.7)
        snr_lim = max(col_snrs.max(), bp_snrs.max()) * 1.1
        ax8.plot([0, snr_lim], [0, snr_lim], "k--", alpha=0.3, label="1:1")
        ax8.set_xlabel("SNR COLUMN")
        ax8.set_ylabel("SNR BP")
        ax8.set_title("Signal-to-Noise")
        ax8.legend(fontsize=8)

        # 9. Evolucio temporal ratio
        ax9 = fig.add_subplot(gs[2, 2])
        seq_nums = [extract_seq_num(r["col_seq"]) for r in clean]
        ax9.scatter(seq_nums, [r["ratio_area_norm"] for r in clean],
                    s=30, c="#009688", edgecolors="k", linewidth=0.5, alpha=0.7)
        ax9.axhline(1.0, color="k", linestyle="--", alpha=0.3)
        # Moving average
        if len(seq_nums) > 10:
            sorted_idx = np.argsort(seq_nums)
            sorted_nums = np.array(seq_nums)[sorted_idx]
            sorted_ratios = np.array([all_rows[i]["ratio_area_norm"]
                                      for i in sorted_idx
                                      if all_rows[i] in clean])[:len(sorted_nums)]
            window = min(20, len(sorted_ratios) // 3)
            if window > 2:
                ma = np.convolve(sorted_ratios, np.ones(window)/window, mode="valid")
                ma_x = sorted_nums[window//2:window//2+len(ma)]
                ax9.plot(ma_x, ma, "r-", alpha=0.7, label=f"MA({window})")
                ax9.legend(fontsize=8)
        ax9.set_xlabel("Numero SEQ")
        ax9.set_ylabel("Ratio BP/COL")
        ax9.set_title("Evolucio temporal")
        ax9.set_ylim(0, 4)

        plt.savefig(os.path.join(OUT_DIR, "column_vs_bp_comparison.png"),
                    dpi=150, bbox_inches="tight")
        print(f"\nGrafic guardat a: {OUT_DIR}/column_vs_bp_comparison.png")
        plt.close()

    except ImportError:
        print("[WARN] matplotlib no disponible, grafics no generats")
    except Exception as e:
        import traceback
        print(f"[WARN] Error generant grafics: {e}")
        traceback.print_exc()

    # ---- CSV ----
    csv_path = os.path.join(OUT_DIR, "column_vs_bp_comparison.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("Mostra,SEQ_COL,SEQ_BP,Area_COL,Area_BP,ppm_COL,ppm_BP,"
                "Ratio_area_norm,A254_COL,A254_BP,"
                "SNR_COL,SNR_BP,CV_COL_pct,CV_BP_pct,"
                "BioP_COL,HS_COL,BB_COL,SB_COL,LMW_COL,"
                "BioP_BP,HS_BP,BB_BP,SB_BP,LMW_BP\n")
        for r in all_rows:
            fc = r["fracs_col"]
            fb = r["fracs_bp"]
            f.write(f"{r['name']},{r['col_seq']},{r['bp_seq']},"
                    f"{r['col_area']:.1f},{r['bp_area']:.1f},"
                    f"{r['col_ppm']:.4f},{r['bp_ppm']:.4f},"
                    f"{r['ratio_area_norm']:.4f},"
                    f"{r['col_a254']:.2f},{r['bp_a254']:.2f},"
                    f"{r['col_snr']:.0f},{r['bp_snr']:.0f},"
                    f"{r['col_cv']:.1f},{r['bp_cv']:.1f},"
                    f"{fc['BioP']:.1f},{fc['HS']:.1f},{fc['BB']:.1f},{fc['SB']:.1f},{fc['LMW']:.1f},"
                    f"{fb['BioP']:.1f},{fb['HS']:.1f},{fb['BB']:.1f},{fb['SB']:.1f},{fb['LMW']:.1f}\n")
    print(f"CSV guardat a: {csv_path}")


if __name__ == "__main__":
    main()
