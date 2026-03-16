"""
Analisi: d'on ve la diferencia entre DOC COLUMN i DOC BP?

Descompon la diferencia en components:
- Baseline noise
- Overhead fraccions
- Concentracio
- Timeouts
- Evolucio temporal
"""

import json
import os
import re
import sys
import numpy as np

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

DATA_DIR = r"C:\Users\Lequia\Desktop\Dades3"
OUT_DIR = os.path.join(DATA_DIR, "REGISTRY", "review", "diff_analysis")

RF_COL = 752.90; RF_BP = 646.82
IC_COL = 18.11; IC_BP = 2.87
VOL_COL = 400.0; VOL_BP = 100.0
MAX_SEQ_GAP = 5
EXCL = ("MQ", "NAOH", "BUFFER", "KHP", "NAN")


def seq_num(name):
    m = re.match(r"(\d+)", name)
    return int(m.group(1)) if m else 0


def to_ppm(area, method):
    if method == "COLUMN":
        return (area - IC_COL) * 1000 / (RF_COL * VOL_COL)
    return (area - IC_BP) * 1000 / (RF_BP * VOL_BP)


def load_pairs():
    results = {}
    for d in sorted(os.listdir(DATA_DIR)):
        f = os.path.join(DATA_DIR, d, "CHECK", "data", "analysis_result.json")
        if not os.path.isfile(f) or "_CAL" in d.upper():
            continue
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
        method = data.get("method", "?")
        if method in ("COLUMN", "BP"):
            results[d] = {"method": method, "samples": data["samples"], "num": seq_num(d)}

    col = {k: v for k, v in results.items() if v["method"] == "COLUMN"}
    bp = {k: v for k, v in results.items() if v["method"] == "BP"}

    rows = []
    for ck, cv in col.items():
        best_bp, best_gap = None, 999
        for bk, bv in bp.items():
            gap = abs(bv["num"] - cv["num"])
            if gap <= MAX_SEQ_GAP and gap < best_gap:
                best_bp, best_gap = bk, gap
        if not best_bp:
            continue
        bv = bp[best_bp]

        c_by = {}
        for s in cv["samples"]:
            n = s["name"]
            if not n.upper().startswith(EXCL):
                c_by.setdefault(n, []).append(s)
        b_by = {}
        for s in bv["samples"]:
            n = s["name"]
            if not n.upper().startswith(EXCL):
                b_by.setdefault(n, []).append(s)

        for name in set(c_by) & set(b_by):
            cr, br = c_by[name], b_by[name]

            def avg_field(reps, *keys, default=0):
                vals = []
                for s in reps:
                    v = s
                    for k in keys:
                        v = v.get(k, {}) if isinstance(v, dict) else {}
                    if isinstance(v, (int, float)) and v > 0:
                        vals.append(v)
                return np.mean(vals) if vals else default

            c_total = avg_field(cr, "areas", "DOC", "total")
            b_total = avg_field(br, "areas", "DOC", "total")
            if c_total <= 0 or b_total <= 0:
                continue

            c_pi = avg_field(cr, "peak_info", "area")
            b_pi = avg_field(br, "peak_info", "area")
            c_bl = avg_field(cr, "peak_info", "baseline_level")
            b_bl = avg_field(br, "peak_info", "baseline_level")
            c_snr = avg_field(cr, "snr_info", "snr_direct")
            b_snr = avg_field(br, "snr_info", "snr_direct")
            c_window = np.mean([
                (s.get("peak_info", {}).get("t_end", 0) or 0) -
                (s.get("peak_info", {}).get("t_start", 0) or 0)
                for s in cr
            ])

            fracs = {}
            for fn in ["BioP", "HS", "BB", "SB", "LMW"]:
                fracs[fn] = avg_field(cr, "areas", "DOC", fn, default=0)

            c_anoms = sum(len(s.get("anomalies", [])) for s in cr)
            b_anoms = sum(len(s.get("anomalies", [])) for s in br)
            c_timeout = any(
                any("TIMEOUT" in a.get("code", "") for a in s.get("anomalies", []))
                for s in cr
            )

            ratio = (b_total / VOL_BP) / (c_total / VOL_COL)

            rows.append({
                "name": name, "col_seq": ck, "bp_seq": best_bp,
                "c_total": c_total, "b_total": b_total,
                "c_pi": c_pi, "b_pi": b_pi,
                "ratio": ratio,
                "c_ppm": to_ppm(c_total, "COLUMN"),
                "b_ppm": to_ppm(b_total, "BP"),
                "fracs": fracs,
                "frac_sum": sum(fracs.values()),
                "overhead_pct": (c_total - c_pi) / c_total * 100 if c_total > 0 else 0,
                "bp_overhead": (b_total - b_pi) / b_total * 100 if b_total > 0 else 0,
                "c_window": c_window,
                "c_snr": c_snr, "b_snr": b_snr,
                "c_bl": c_bl, "b_bl": b_bl,
                "c_anoms": c_anoms, "b_anoms": b_anoms,
                "c_timeout": c_timeout,
                "col_num": cv["num"],
            })

    return rows


def main():
    rows = load_pairs()
    print(f"Total parells aparellats: {len(rows)}")

    # NO filter by ratio — show all
    valid = [r for r in rows if r["c_ppm"] > 0 and r["b_ppm"] > 0]
    print(f"Amb ppm > 0: {len(valid)}")

    os.makedirs(OUT_DIR, exist_ok=True)

    ratios = np.array([r["ratio"] for r in valid])
    c_ppms = np.array([r["c_ppm"] for r in valid])
    b_ppms = np.array([r["b_ppm"] for r in valid])
    diffs = b_ppms - c_ppms
    diffs_pct = diffs / c_ppms * 100

    # ================================================================
    print(f"\n{'='*80}")
    print("1. DIFERENCIA GLOBAL (BP - COLUMN)")
    print(f"{'='*80}")
    print(f"  N = {len(valid)}")
    print(f"  Diferencia ppm:  {np.mean(diffs):+.3f} +/- {np.std(diffs):.3f} (mitjana)")
    print(f"                   {np.median(diffs):+.3f} (mediana)")
    print(f"  Diferencia %:    {np.mean(diffs_pct):+.1f}% +/- {np.std(diffs_pct):.1f}%")
    print(f"                   {np.median(diffs_pct):+.1f}% (mediana)")
    print(f"  Ratio BP/COL:    {np.mean(ratios):.3f} +/- {np.std(ratios):.3f}")
    print(f"                   {np.median(ratios):.3f} (mediana)")

    # Correlation
    finite = np.isfinite(c_ppms) & np.isfinite(b_ppms) & (c_ppms > 0)
    if np.sum(finite) > 3:
        rp = np.corrcoef(c_ppms[finite], b_ppms[finite])[0, 1]
        sl, ic = np.polyfit(c_ppms[finite], b_ppms[finite], 1)
        print(f"  R2 = {rp**2:.4f}")
        print(f"  BP = {sl:.3f} * COL + {ic:.3f}")

    # ================================================================
    print(f"\n{'='*80}")
    print("2. PER RANG DE CONCENTRACIO")
    print(f"{'='*80}")
    for lo, hi, label in [(0,1,"<1"),(1,2,"1-2"),(2,3,"2-3"),(3,5,"3-5"),
                           (5,10,"5-10"),(10,20,"10-20"),(20,999,">20")]:
        sub = [r for r in valid if lo <= r["c_ppm"] < hi]
        if len(sub) < 3:
            continue
        sr = [r["ratio"] for r in sub]
        sd = [r["b_ppm"] - r["c_ppm"] for r in sub]
        sdp = [(r["b_ppm"]-r["c_ppm"])/r["c_ppm"]*100 for r in sub]
        print(f"  {label:6s} ppm: n={len(sub):3d}  ratio={np.median(sr):.3f}  "
              f"diff={np.median(sd):+.2f} ppm  ({np.median(sdp):+.1f}%)")

    # ================================================================
    print(f"\n{'='*80}")
    print("3. BASELINE I SOROLL")
    print(f"{'='*80}")
    c_bls = [r["c_bl"] for r in valid]
    b_bls = [r["b_bl"] for r in valid]
    print(f"  Baseline COLUMN: {np.mean(c_bls):.1f} +/- {np.std(c_bls):.1f} ppb")
    print(f"  Baseline BP:     {np.mean(b_bls):.1f} +/- {np.std(b_bls):.1f} ppb")
    print(f"  SNR COLUMN:      {np.mean([r['c_snr'] for r in valid]):.0f}")
    print(f"  SNR BP:          {np.mean([r['b_snr'] for r in valid]):.0f}")

    # Correlation ratio vs baseline
    for label, vals in [("baseline COL", [r["c_bl"] for r in valid]),
                         ("SNR COL", [r["c_snr"] for r in valid]),
                         ("ppm COL", [r["c_ppm"] for r in valid]),
                         ("overhead% COL", [r["overhead_pct"] for r in valid]),
                         ("finestra COL", [r["c_window"] for r in valid])]:
        x = np.array(vals)
        mask = np.isfinite(x) & np.isfinite(ratios)
        if np.sum(mask) > 5:
            r = np.corrcoef(x[mask], ratios[mask])[0, 1]
            print(f"  Correlacio ratio vs {label:18s}: r={r:+.3f}")

    # ================================================================
    print(f"\n{'='*80}")
    print("4. OVERHEAD (area fora pic principal)")
    print(f"{'='*80}")
    oh_c = [r["overhead_pct"] for r in valid]
    oh_b = [r["bp_overhead"] for r in valid]
    print(f"  COLUMN: {np.mean(oh_c):.1f}% +/- {np.std(oh_c):.1f}% del DOC.total esta fora del pic HS")
    print(f"  BP:     {np.mean(oh_b):.1f}% +/- {np.std(oh_b):.1f}% del DOC.total esta fora peak_info")
    print(f"  Finestra peak_info COLUMN: {np.mean([r['c_window'] for r in valid]):.1f} min")

    # ================================================================
    print(f"\n{'='*80}")
    print("5. FRACCIONS COLUMN")
    print(f"{'='*80}")
    for fn in ["BioP", "HS", "BB", "SB", "LMW"]:
        pcts = [r["fracs"][fn] / r["c_total"] * 100 for r in valid if r["c_total"] > 0]
        abs_ppb = [r["fracs"][fn] for r in valid]
        print(f"  {fn:5s}: {np.mean(pcts):5.1f}% +/- {np.std(pcts):.1f}%  "
              f"(area: {np.mean(abs_ppb):.0f} +/- {np.std(abs_ppb):.0f})")

    non_hs = [(r["c_total"] - r["fracs"]["HS"]) for r in valid]
    print(f"\n  Area NON-HS: {np.mean(non_hs):.0f} +/- {np.std(non_hs):.0f}")
    print(f"  Si resta NON-HS de COLUMN, el ratio seria:")
    hs_ratios = [(r["b_total"]/VOL_BP) / (r["fracs"]["HS"]/VOL_COL)
                 for r in valid if r["fracs"]["HS"] > 0]
    print(f"    Ratio BP / HS_COLUMN: {np.median(hs_ratios):.3f}")

    # ================================================================
    print(f"\n{'='*80}")
    print("6. IMPACTE ANOMALIES")
    print(f"{'='*80}")
    for label, filt in [
        ("Sense anomalies", lambda r: r["c_anoms"] == 0 and r["b_anoms"] == 0),
        ("Amb timeout COL", lambda r: r["c_timeout"]),
        ("Sense timeout", lambda r: not r["c_timeout"]),
        ("TOTES", lambda r: True),
    ]:
        sub = [r for r in valid if filt(r)]
        if len(sub) < 3:
            print(f"  {label:25s}: n={len(sub)} (insuficient)")
            continue
        sr = [r["ratio"] for r in sub]
        sd = [(r["b_ppm"]-r["c_ppm"])/r["c_ppm"]*100 for r in sub]
        print(f"  {label:25s}: n={len(sub):3d}  ratio={np.median(sr):.3f}  "
              f"diff={np.median(sd):+.1f}%")

    # ================================================================
    print(f"\n{'='*80}")
    print("7. EVOLUCIO TEMPORAL")
    print(f"{'='*80}")
    for lo, hi, label in [(72,120,"072-120"),(120,180,"120-180"),(180,230,"180-230"),
                           (230,260,"230-260"),(260,300,"260-300")]:
        sub = [r for r in valid if lo <= r["col_num"] < hi]
        if len(sub) < 3:
            continue
        sr = [r["ratio"] for r in sub]
        sd = [(r["b_ppm"]-r["c_ppm"])/r["c_ppm"]*100 for r in sub]
        print(f"  SEQ {label}: n={len(sub):3d}  ratio={np.median(sr):.3f}  "
              f"diff={np.median(sd):+.1f}%")

    # ================================================================
    print(f"\n{'='*80}")
    print("8. CASOS EXTREMS")
    print(f"{'='*80}")
    sorted_by_ratio = sorted(valid, key=lambda r: r["ratio"])
    print("\n  TOP 10 ratio mes alta (BP >> COL):")
    for r in sorted_by_ratio[-10:]:
        print(f"    {r['name']:15s} {r['col_seq']:15s}/{r['bp_seq']:15s}  "
              f"COL={r['c_ppm']:.2f}  BP={r['b_ppm']:.2f}  ratio={r['ratio']:.2f}  "
              f"bl_COL={r['c_bl']:.0f}")
    print("\n  TOP 10 ratio mes baixa (COL >> BP):")
    for r in sorted_by_ratio[:10]:
        print(f"    {r['name']:15s} {r['col_seq']:15s}/{r['bp_seq']:15s}  "
              f"COL={r['c_ppm']:.2f}  BP={r['b_ppm']:.2f}  ratio={r['ratio']:.2f}  "
              f"bl_COL={r['c_bl']:.0f}")

    # ================================================================
    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(18, 11))
        fig.suptitle(f"Analisi diferencia COLUMN vs BP (n={len(valid)})",
                     fontsize=14, fontweight="bold")

        # 1. Diff% vs concentration
        ax = axes[0, 0]
        ax.scatter(c_ppms, diffs_pct, s=15, alpha=0.4, edgecolors="k", linewidth=0.2)
        ax.axhline(0, color="k", linestyle="--", alpha=0.3)
        ax.set_xlabel("ppm COLUMN")
        ax.set_ylabel("Diferencia % (BP-COL)/COL")
        ax.set_title("Diferencia vs concentracio")
        ax.set_ylim(-100, 200)
        ax.set_xlim(0, 25)

        # 2. Diff% vs baseline
        ax = axes[0, 1]
        ax.scatter([r["c_bl"] for r in valid], diffs_pct,
                   s=15, alpha=0.4, c="#FF5722", edgecolors="k", linewidth=0.2)
        ax.axhline(0, color="k", linestyle="--", alpha=0.3)
        ax.set_xlabel("Baseline COLUMN (ppb)")
        ax.set_ylabel("Diferencia %")
        ax.set_title("Diferencia vs baseline")
        ax.set_ylim(-100, 200)

        # 3. Scatter ppm COL vs BP
        ax = axes[0, 2]
        mask = (c_ppms < 25) & (b_ppms < 25) & (c_ppms > 0)
        ax.scatter(c_ppms[mask], b_ppms[mask], s=15, alpha=0.4,
                   edgecolors="k", linewidth=0.2)
        lim = max(c_ppms[mask].max(), b_ppms[mask].max()) * 1.1
        ax.plot([0, lim], [0, lim], "k--", alpha=0.3, label="1:1")
        if np.sum(mask) > 3:
            rp = np.corrcoef(c_ppms[mask], b_ppms[mask])[0, 1]
            sl, ic = np.polyfit(c_ppms[mask], b_ppms[mask], 1)
            ax.plot([0, lim], [ic, sl*lim+ic], "r-", alpha=0.7,
                    label=f"R2={rp**2:.3f}\ny={sl:.2f}x{ic:+.2f}")
        ax.set_xlabel("ppm COLUMN"); ax.set_ylabel("ppm BP")
        ax.set_title("Correlacio ppm")
        ax.legend(fontsize=8)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)

        # 4. Histogram of ratio
        ax = axes[1, 0]
        r_clean = ratios[(ratios > 0.2) & (ratios < 4)]
        ax.hist(r_clean, bins=40, color="#4CAF50", edgecolor="k", alpha=0.8)
        ax.axvline(1.0, color="k", linestyle="--", alpha=0.5)
        ax.axvline(np.median(r_clean), color="r", linestyle="-", alpha=0.7,
                   label=f"Mediana: {np.median(r_clean):.3f}")
        ax.set_xlabel("Ratio BP/COL (area/uL)")
        ax.set_ylabel("Comptatge")
        ax.set_title("Distribucio ratio")
        ax.legend(fontsize=8)

        # 5. Ratio vs SEQ number (temporal)
        ax = axes[1, 1]
        nums = [r["col_num"] for r in valid]
        ax.scatter(nums, ratios, s=15, alpha=0.4, c="#2196F3",
                   edgecolors="k", linewidth=0.2)
        ax.axhline(1.0, color="k", linestyle="--", alpha=0.3)
        ax.set_xlabel("Numero SEQ")
        ax.set_ylabel("Ratio BP/COL")
        ax.set_title("Evolucio temporal")
        ax.set_ylim(0, 4)

        # 6. Diff% vs overhead
        ax = axes[1, 2]
        ax.scatter([r["overhead_pct"] for r in valid], diffs_pct,
                   s=15, alpha=0.4, c="#9C27B0", edgecolors="k", linewidth=0.2)
        ax.axhline(0, color="k", linestyle="--", alpha=0.3)
        ax.set_xlabel("Overhead % (area fora pic HS)")
        ax.set_ylabel("Diferencia %")
        ax.set_title("Diferencia vs overhead")
        ax.set_ylim(-100, 200)

        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "diff_analysis.png"),
                    dpi=150, bbox_inches="tight")
        print(f"\nGrafic: {OUT_DIR}/diff_analysis.png")
        plt.close()

    except Exception as e:
        print(f"Error grafics: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
