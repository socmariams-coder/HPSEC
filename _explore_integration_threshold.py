"""
Exploració: efecte del metode d'integracio sobre la consistencia COLUMN vs BP.

Hipotesi: el soroll de baseline afegeix area espuria, especialment a COLUMN
(finestra mes ampla) vs BP (finestra estreta). Provem multiples metodes
per trobar el que dona ratio BP/COL mes consistent (proper a 1.0).
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
OUT_DIR = os.path.join(DATA_DIR, "REGISTRY", "review", "integration_threshold")

RF_COL = 752.90;  RF_BP = 646.82
INTERCEPT_COL = 18.11;  INTERCEPT_BP = 2.87
VOL_COL = 400.0;  VOL_BP = 100.0
MAX_SEQ_GAP = 5
MIN_SEQ_GOOD = 178  # SEQs >= 178 tenen dades fiables
EXCLUDE_PATTERNS = ("MQ", "NAOH", "BUFFER", "KHP", "NAN")


def extract_seq_num(name):
    m = re.match(r"(\d+)", name)
    return int(m.group(1)) if m else 0


def area_to_ppm(area, method):
    if method == "COLUMN":
        return (area - INTERCEPT_COL) * 1000 / (RF_COL * VOL_COL)
    else:
        return (area - INTERCEPT_BP) * 1000 / (RF_BP * VOL_BP)


# ================================================================
# Integration methods
# ================================================================

def estimate_noise(t, y, t_start, t_end):
    if not isinstance(y, np.ndarray) or y.ndim == 0 or len(y) < 10:
        return 1.0
    mask = (t < t_start - 1.0) | (t > t_end + 1.0)
    if np.sum(mask) < 20:
        tail = np.concatenate([y[:50], y[-50:]])
        return max(np.std(tail), 0.01)
    return max(np.std(y[mask]), 0.01)


def estimate_fwhm(t, y, t_max_val):
    idx_max = np.argmin(np.abs(t - t_max_val))
    half_max = y[idx_max] / 2.0
    if half_max <= 0:
        return None
    left_idx = idx_max
    while left_idx > 0 and y[left_idx] > half_max:
        left_idx -= 1
    right_idx = idx_max
    while right_idx < len(y) - 1 and y[right_idx] > half_max:
        right_idx += 1
    fwhm = t[right_idx] - t[left_idx]
    return fwhm if fwhm > 0 else None


def _trapz(t_seg, y_seg):
    if len(t_seg) < 2:
        return 0.0
    return float(np.trapezoid(y_seg, t_seg))


def integrate_full(t, y, t_lo, t_hi):
    mask = (t >= t_lo) & (t <= t_hi)
    return _trapz(t[mask], y[mask])


def integrate_positive(t, y, t_lo, t_hi):
    mask = (t >= t_lo) & (t <= t_hi)
    return _trapz(t[mask], np.maximum(y[mask], 0.0))


def integrate_threshold(t, y, t_lo, t_hi, threshold):
    mask = (t >= t_lo) & (t <= t_hi)
    y_cut = np.where(y[mask] > threshold, y[mask] - threshold, 0.0)
    return _trapz(t[mask], y_cut)


def compute_areas(sample):
    """Calcular area amb tots els metodes per una mostra."""
    t = np.array(sample["t_doc"])
    y_raw = sample.get("y_doc_direct_net", sample.get("y_doc_net"))
    if y_raw is None:
        return None
    y = np.array(y_raw, dtype=float)
    if y.ndim == 0 or len(y) < 10:
        return None

    pi = sample["peak_info"]
    t_start, t_end = pi["t_start"], pi["t_end"]
    t_max = pi["t_max"]
    idx_max = np.argmin(np.abs(t - t_max))
    peak_h = float(y[idx_max]) if idx_max < len(y) else 0

    noise = estimate_noise(t, y, t_start, t_end)
    fwhm = estimate_fwhm(t, y, t_max)

    areas = {}

    # 1. Original (de la Suite)
    areas["original"] = pi["area"]

    # 2. Limits originals, variants
    areas["net_full"] = integrate_full(t, y, t_start, t_end)
    areas["pos_only"] = integrate_positive(t, y, t_start, t_end)

    # 3. Threshold absolut (N * sigma_noise)
    for ns in [0.5, 1, 2, 3, 5]:
        areas[f"thr_{ns}s"] = integrate_threshold(t, y, t_start, t_end, ns * noise)

    # 4. Threshold relatiu (% alçada pic) amb finestra COMPLETA
    if peak_h > 0:
        for pct in [0.5, 1, 2, 5, 10]:
            thr = peak_h * pct / 100
            areas[f"pct_{pct}%"] = integrate_threshold(t, y, t[0], t[-1], thr)

    # 5. FWHM window
    if fwhm and fwhm > 0:
        for k in [1, 1.5, 2, 3, 4, 6]:
            t_lo = t_max - k * fwhm
            t_hi = t_max + k * fwhm
            areas[f"fwhm_{k}x"] = integrate_positive(t, y, t_lo, t_hi)

    # 6. Finestra fixa centrada en t_max
    for w in [3, 5, 8, 12, 20, 30]:
        t_lo = t_max - w / 3
        t_hi = t_max + 2 * w / 3
        areas[f"fix_{w}m"] = integrate_positive(t, y, t_lo, t_hi)

    # 7. Full chromatogram positive
    areas["full_pos"] = integrate_positive(t, y, t[0], t[-1])

    # 8. Full chromatogram with noise threshold
    areas["full_1s"] = integrate_threshold(t, y, t[0], t[-1], noise)
    areas["full_2s"] = integrate_threshold(t, y, t[0], t[-1], 2 * noise)

    # Metadata
    anomalies = sample.get("anomalies", [])
    return {
        "areas": areas,
        "noise": noise,
        "fwhm": fwhm or 0,
        "window": t_end - t_start,
        "peak_h": peak_h,
        "t_max": t_max,
        "t_range": t[-1] - t[0],
        "has_timeout": any("TIMEOUT" in a.get("code", "") for a in anomalies),
        "has_irregular": sample.get("irregular_top_direct", False),
        "n_anomalies": len(anomalies),
    }


# ================================================================
# Data loading and pairing
# ================================================================

def load_and_pair():
    results = {}
    for d in sorted(os.listdir(DATA_DIR)):
        f = os.path.join(DATA_DIR, d, "CHECK", "data", "analysis_result.json")
        if not os.path.isfile(f) or "_CAL" in d.upper():
            continue
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
        method = data.get("method", "?")
        if method not in ("COLUMN", "BP"):
            continue
        results[d] = {
            "method": method,
            "samples": data.get("samples", []),
            "seq_num": extract_seq_num(d),
        }

    col_seqs = {k: v for k, v in results.items() if v["method"] == "COLUMN"}
    bp_seqs = {k: v for k, v in results.items() if v["method"] == "BP"}

    paired = []
    for ck, cv in col_seqs.items():
        col_num = cv["seq_num"]
        best_bp, best_gap = None, 999
        for bk, bv in bp_seqs.items():
            gap = abs(bv["seq_num"] - col_num)
            if gap <= MAX_SEQ_GAP and gap < best_gap:
                best_bp, best_gap = bk, gap
        if not best_bp:
            continue

        col_names = {}
        for s in cv["samples"]:
            n = s["name"]
            if not n.upper().startswith(EXCLUDE_PATTERNS) and "t_doc" in s:
                if n not in col_names:
                    col_names[n] = s

        bp_names = {}
        for s in bp_seqs[best_bp]["samples"]:
            n = s["name"]
            if not n.upper().startswith(EXCLUDE_PATTERNS) and "t_doc" in s:
                if n not in bp_names:
                    bp_names[n] = s

        for name in set(col_names) & set(bp_names):
            cs, bs = col_names[name], bp_names[name]
            if cs.get("peak_info", {}).get("valid") and bs.get("peak_info", {}).get("valid"):
                paired.append({
                    "name": name, "col_seq": ck, "bp_seq": best_bp,
                    "col": cs, "bp": bs,
                    "col_num": col_num, "bp_num": bp_seqs[best_bp]["seq_num"],
                })

    return paired


# ================================================================
# Analysis
# ================================================================

def analyze_pairs(label, pairs):
    print(f"\n{'='*120}")
    print(f"  {label} ({len(pairs)} parells)")
    print(f"{'='*120}")

    by_method = {}

    for p in pairs:
        col_res = compute_areas(p["col"])
        bp_res = compute_areas(p["bp"])
        if col_res is None or bp_res is None:
            continue

        for mn in col_res["areas"]:
            ca = col_res["areas"].get(mn, 0)
            ba = bp_res["areas"].get(mn, 0)
            if not isinstance(ca, (int, float)) or not isinstance(ba, (int, float)):
                continue
            if ca <= 0 or ba <= 0:
                continue

            ratio = (ba / VOL_BP) / (ca / VOL_COL)
            if ratio > 10 or ratio < 0.1:
                continue

            if mn not in by_method:
                by_method[mn] = []
            by_method[mn].append({
                "name": p["name"],
                "ratio": ratio,
                "col_ppm": area_to_ppm(ca, "COLUMN"),
                "bp_ppm": area_to_ppm(ba, "BP"),
                "col_area": ca, "bp_area": ba,
                "col_noise": col_res["noise"],
                "bp_noise": bp_res["noise"],
                "col_window": col_res["window"],
                "bp_window": bp_res["window"],
                "col_timeout": col_res["has_timeout"],
                "col_irregular": col_res["has_irregular"],
                "col_anomalies": col_res["n_anomalies"],
                "bp_anomalies": bp_res["n_anomalies"],
            })

    if not by_method:
        print("  Cap dada!")
        return by_method

    # Sort methods by score
    scores = {}
    for mn, rows in by_method.items():
        if len(rows) < 5:
            continue
        ratios = np.array([r["ratio"] for r in rows])
        cp = np.array([r["col_ppm"] for r in rows])
        bp = np.array([r["bp_ppm"] for r in rows])
        med = np.median(ratios)
        sd = np.std(ratios)
        r2 = np.corrcoef(cp, bp)[0, 1] ** 2 if len(cp) > 2 else 0
        sl = np.polyfit(cp, bp, 1)[0] if len(cp) > 2 else 0
        bias = abs(med - 1.0)
        w10 = np.mean(np.abs(ratios - 1.0) < 0.10) * 100
        scores[mn] = {
            "n": len(rows), "med": med, "avg": np.mean(ratios), "sd": sd,
            "r2": r2, "sl": sl, "bias": bias, "w10": w10,
            "score": bias + sd * 0.5 + (1 - r2) * 0.3,
        }

    # Print sorted by score
    print(f"\n{'Metode':18s} {'N':>4s} {'Med':>7s} {'Avg':>7s} {'SD':>6s} "
          f"{'R2':>6s} {'Slope':>6s} {'|Bias|':>7s} {'<10%':>6s} {'Score':>6s}")
    print("-" * 95)
    for mn, sc in sorted(scores.items(), key=lambda x: x[1]["score"]):
        flag = " ***" if sc["score"] == min(s["score"] for s in scores.values()) else ""
        print(f"{mn:18s} {sc['n']:4d} {sc['med']:7.3f} {sc['avg']:7.3f} {sc['sd']:6.3f} "
              f"{sc['r2']:6.3f} {sc['sl']:6.3f} {sc['bias']:7.3f} {sc['w10']:5.1f}% "
              f"{sc['score']:6.3f}{flag}")

    # Best method
    best = min(scores.keys(), key=lambda k: scores[k]["score"])
    print(f"\nMillor: {best} (score={scores[best]['score']:.3f})")

    # Detail: original vs best
    for mn in ["original", best]:
        if mn not in by_method or mn not in scores:
            continue
        rows = by_method[mn]
        ratios = np.array([r["ratio"] for r in rows])
        cp = np.array([r["col_ppm"] for r in rows])
        bp = np.array([r["bp_ppm"] for r in rows])
        print(f"\n  --- {mn} (n={len(rows)}) ---")
        for pct in [10, 25, 50, 75, 90]:
            print(f"    P{pct}: {np.percentile(ratios, pct):.3f}")
        if len(cp) > 2:
            r = np.corrcoef(cp, bp)[0, 1]
            sl, ic = np.polyfit(cp, bp, 1)
            print(f"    R2={r**2:.4f}, BP = {sl:.3f}*COL + {ic:.3f}")
        for lo, hi, lbl in [(0,2,"<2ppm"),(2,5,"2-5ppm"),(5,15,"5-15ppm"),(15,999,">15ppm")]:
            sub = [r for r in rows if lo <= r["col_ppm"] < hi]
            if len(sub) >= 3:
                sr = [r["ratio"] for r in sub]
                print(f"    {lbl:10s}: n={len(sub):3d}, ratio={np.median(sr):.3f} "
                      f"(SD={np.std(sr):.3f})")

    # Anomaly impact
    rows = by_method.get("original", [])
    if len(rows) >= 5:
        print(f"\n  Impacte anomalies:")
        clean = [r for r in rows if not r["col_timeout"] and not r["col_irregular"]
                 and r["col_anomalies"] == 0 and r["bp_anomalies"] == 0]
        with_to = [r for r in rows if r["col_timeout"]]
        with_ir = [r for r in rows if r["col_irregular"]]
        for lbl, sub in [("Netes", clean), ("Timeout", with_to),
                          ("Irregular", with_ir), ("Totes", rows)]:
            if len(sub) >= 3:
                sr = [r["ratio"] for r in sub]
                print(f"    {lbl:12s}: n={len(sub):3d}, ratio={np.median(sr):.3f} "
                      f"(SD={np.std(sr):.3f})")

    # Noise stats
    print(f"\n  Soroll baseline:")
    print(f"    COLUMN: sigma={np.mean([r['col_noise'] for r in rows]):.1f} ppb, "
          f"finestra={np.mean([r['col_window'] for r in rows]):.1f} min")
    print(f"    BP:     sigma={np.mean([r['bp_noise'] for r in rows]):.1f} ppb, "
          f"finestra={np.mean([r['bp_window'] for r in rows]):.1f} min")

    return by_method


def main():
    print("Carregant i aparellant dades...")
    all_pairs = load_and_pair()
    print(f"Total parells: {len(all_pairs)}")

    os.makedirs(OUT_DIR, exist_ok=True)

    good = [p for p in all_pairs
            if p["col_num"] >= MIN_SEQ_GOOD and p["bp_num"] >= MIN_SEQ_GOOD]
    old = [p for p in all_pairs
           if p["col_num"] < MIN_SEQ_GOOD or p["bp_num"] < MIN_SEQ_GOOD]
    print(f"  Post-{MIN_SEQ_GOOD}: {len(good)}")
    print(f"  Pre-{MIN_SEQ_GOOD}: {len(old)}")

    # Analyze both eras
    by_all = analyze_pairs("TOTES LES SEQS", all_pairs)
    by_good = analyze_pairs(f"POST-{MIN_SEQ_GOOD} (fiables)", good)

    # Generate plot for good data
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not by_good:
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 11))
        fig.suptitle(f"Integracio: impacte sobre ratio BP/COL (SEQs >= {MIN_SEQ_GOOD})",
                     fontsize=14, fontweight="bold")

        # Collect method summaries for good data
        summaries = {}
        for mn, rows in by_good.items():
            if len(rows) < 5:
                continue
            ratios = [r["ratio"] for r in rows if 0.2 <= r["ratio"] <= 5]
            if len(ratios) < 5:
                continue
            summaries[mn] = {
                "med": np.median(ratios), "sd": np.std(ratios),
                "n": len(ratios), "ratios": ratios,
                "rows": rows,
            }

        # 1. Bar chart: median ratio per method (sorted)
        ax = axes[0, 0]
        sorted_methods = sorted(summaries.keys(),
                                key=lambda k: abs(summaries[k]["med"] - 1.0))[:20]
        meds = [summaries[m]["med"] for m in sorted_methods]
        sds = [summaries[m]["sd"] for m in sorted_methods]
        colors = ["#4CAF50" if abs(m-1) < 0.1 else "#FF9800" if abs(m-1) < 0.3
                  else "#F44336" for m in meds]
        ax.barh(range(len(sorted_methods)), meds, xerr=sds,
                color=colors, edgecolor="k", linewidth=0.5, alpha=0.8, capsize=2)
        ax.axvline(1.0, color="k", linestyle="--", alpha=0.5)
        ax.set_yticks(range(len(sorted_methods)))
        ax.set_yticklabels(sorted_methods, fontsize=7)
        ax.set_xlabel("Ratio mediana BP/COL")
        ax.set_title("Metodes ordenats per proximitat a 1.0")

        # 2. Scatter original
        ax = axes[0, 1]
        if "original" in by_good:
            rows = by_good["original"]
            cp = np.array([r["col_ppm"] for r in rows])
            bp = np.array([r["bp_ppm"] for r in rows])
            m = (cp > 0) & (bp > 0) & (cp < 50) & (bp < 50)
            ax.scatter(cp[m], bp[m], s=25, alpha=0.6, edgecolors="k", linewidth=0.3)
            lim = max(cp[m].max(), bp[m].max()) * 1.1
            ax.plot([0, lim], [0, lim], "k--", alpha=0.3)
            if np.sum(m) > 2:
                r = np.corrcoef(cp[m], bp[m])[0, 1]
                sl, ic = np.polyfit(cp[m], bp[m], 1)
                ax.plot(np.linspace(0, lim), sl * np.linspace(0, lim) + ic,
                        "r-", alpha=0.7, label=f"R2={r**2:.3f}")
                ax.legend(fontsize=8)
            ax.set_xlabel("ppm COLUMN"); ax.set_ylabel("ppm BP")
            ax.set_title("Original"); ax.set_xlim(0, lim); ax.set_ylim(0, lim)

        # 3. Scatter best method
        ax = axes[0, 2]
        if sorted_methods:
            best = sorted_methods[0]
            rows = by_good[best]
            cp = np.array([r["col_ppm"] for r in rows])
            bp = np.array([r["bp_ppm"] for r in rows])
            m = (cp > 0) & (bp > 0) & (cp < 50) & (bp < 50)
            ax.scatter(cp[m], bp[m], s=25, alpha=0.6, edgecolors="k", linewidth=0.3,
                       c="#4CAF50")
            lim = max(cp[m].max(), bp[m].max()) * 1.1
            ax.plot([0, lim], [0, lim], "k--", alpha=0.3)
            if np.sum(m) > 2:
                r = np.corrcoef(cp[m], bp[m])[0, 1]
                sl, ic = np.polyfit(cp[m], bp[m], 1)
                ax.plot(np.linspace(0, lim), sl * np.linspace(0, lim) + ic,
                        "r-", alpha=0.7, label=f"R2={r**2:.3f}")
                ax.legend(fontsize=8)
            ax.set_xlabel("ppm COLUMN"); ax.set_ylabel("ppm BP")
            ax.set_title(f"Millor: {best}"); ax.set_xlim(0, lim); ax.set_ylim(0, lim)

        # 4. Violin: top 6 methods
        ax = axes[1, 0]
        top6 = sorted_methods[:6]
        for i, mn in enumerate(top6):
            ratios = [r for r in summaries[mn]["ratios"] if 0.3 <= r <= 3.0]
            parts = ax.violinplot([ratios], positions=[i], showmedians=True)
            for pc in parts['bodies']:
                pc.set_facecolor(colors[i] if i < len(colors) else "#999")
                pc.set_alpha(0.6)
        ax.axhline(1.0, color="k", linestyle="--", alpha=0.3)
        ax.set_xticks(range(len(top6)))
        ax.set_xticklabels(top6, fontsize=7, rotation=30)
        ax.set_ylabel("Ratio BP/COL")
        ax.set_title("Top 6 metodes")
        ax.set_ylim(0.3, 2.5)

        # 5. Ratio vs ppm (original)
        ax = axes[1, 1]
        if "original" in by_good:
            rows = by_good["original"]
            ax.scatter([r["col_ppm"] for r in rows],
                       [r["ratio"] for r in rows],
                       s=25, alpha=0.6, edgecolors="k", linewidth=0.3)
            ax.axhline(1.0, color="k", linestyle="--", alpha=0.3)
            ax.set_xlabel("ppm COLUMN")
            ax.set_ylabel("Ratio BP/COL")
            ax.set_title("Original: ratio vs concentracio")
            ax.set_ylim(0, 3); ax.set_xlim(0, 20)

        # 6. Ratio vs ppm (best)
        ax = axes[1, 2]
        if sorted_methods:
            best = sorted_methods[0]
            rows = by_good[best]
            ax.scatter([r["col_ppm"] for r in rows],
                       [r["ratio"] for r in rows],
                       s=25, alpha=0.6, edgecolors="k", linewidth=0.3, c="#4CAF50")
            ax.axhline(1.0, color="k", linestyle="--", alpha=0.3)
            ax.set_xlabel("ppm COLUMN")
            ax.set_ylabel("Ratio BP/COL")
            ax.set_title(f"{best}: ratio vs concentracio")
            ax.set_ylim(0, 3); ax.set_xlim(0, 20)

        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "integration_exploration.png"),
                    dpi=150, bbox_inches="tight")
        print(f"\nGrafic: {OUT_DIR}/integration_exploration.png")
        plt.close()

    except Exception as e:
        print(f"Error grafics: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
