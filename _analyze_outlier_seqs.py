"""
Quines seqüències concretes causen el ratio alt a baixa concentració?

Per cada parell COL/BP amb ratio anòmal, identificar la SEQ i veure
si el problema és generalitzat o concentrat en unes poques SEQs.
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
OUT_DIR = os.path.join(DATA_DIR, "REGISTRY", "review", "outlier_seqs")

RF_COL = 752.90; RF_BP = 646.82
IC_COL = 18.11; IC_BP = 2.87
VOL_COL = 400.0; VOL_BP = 100.0
MAX_SEQ_GAP = 5
EXCL = ("MQ", "NAOH", "BUFFER", "KHP", "NAN")


def seq_num(name):
    m = re.match(r"(\d+)", name)
    return int(m.group(1)) if m else 0


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

        c_by, b_by = {}, {}
        for s in cv["samples"]:
            n = s.get("name", "")
            if not n.upper().startswith(EXCL):
                c_by.setdefault(n, []).append(s)
        for s in bv["samples"]:
            n = s.get("name", "")
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

            c_ppm = (c_total - IC_COL) * 1000 / (RF_COL * VOL_COL)
            b_ppm = (b_total - IC_BP) * 1000 / (RF_BP * VOL_BP)
            if c_ppm <= 0 or b_ppm <= 0:
                continue

            rows.append({
                "name": name, "col_seq": ck, "bp_seq": best_bp,
                "c_total": c_total, "b_total": b_total,
                "c_ppm": c_ppm, "b_ppm": b_ppm,
                "ratio": b_ppm / c_ppm,
                "col_num": cv["num"], "bp_num": bv["num"],
                "pair": f"{ck} / {best_bp}",
            })
    return rows


def main():
    rows = load_pairs()
    print(f"Total parells: {len(rows)}")

    # ================================================================
    # 1. PER PARELL SEQ: estadístiques
    # ================================================================
    print("\n" + "=" * 80)
    print("1. ESTADÍSTIQUES PER PARELL DE SEQÜÈNCIES")
    print("=" * 80)

    pairs = {}
    for r in rows:
        key = r["pair"]
        pairs.setdefault(key, []).append(r)

    print(f"\n  {'Parell COL/BP':>40s}  {'n':>3s}  {'ratio med':>10s}  "
          f"{'ppm_COL med':>11s}  {'ppm_BP med':>10s}  {'flags':>10s}")
    print("  " + "-" * 95)

    pair_stats = []
    for pair_key in sorted(pairs.keys(), key=lambda k: pairs[k][0]["col_num"]):
        pr = pairs[pair_key]
        ratios = [r["ratio"] for r in pr]
        c_ppms = [r["c_ppm"] for r in pr]
        b_ppms = [r["b_ppm"] for r in pr]
        med_ratio = np.median(ratios)
        med_c = np.median(c_ppms)
        med_b = np.median(b_ppms)

        flags = []
        if med_ratio > 2.0:
            flags.append("RATIO>>")
        elif med_ratio > 1.5:
            flags.append("RATIO>")
        elif med_ratio < 0.5:
            flags.append("RATIO<<")
        elif med_ratio < 0.7:
            flags.append("RATIO<")

        n_outlier = sum(1 for r in ratios if r > 2.0 or r < 0.5)
        if n_outlier > 0:
            flags.append(f"{n_outlier}out")

        flag_str = " ".join(flags) if flags else "ok"
        print(f"  {pair_key:>40s}  {len(pr):3d}  {med_ratio:10.3f}  "
              f"{med_c:11.2f}  {med_b:10.2f}  {flag_str:>10s}")

        pair_stats.append({
            "pair": pair_key, "n": len(pr),
            "med_ratio": med_ratio, "med_c": med_c, "med_b": med_b,
            "col_num": pr[0]["col_num"],
            "ratios": ratios, "c_ppms": c_ppms,
            "flag": flag_str,
        })

    # ================================================================
    # 2. SEQS PROBLEMÀTIQUES (ratio > 1.5 o < 0.5)
    # ================================================================
    print("\n" + "=" * 80)
    print("2. PARELLS AMB RATIO ANÒMAL (mediana > 1.5 o < 0.5)")
    print("=" * 80)

    bad_pairs = [p for p in pair_stats if p["med_ratio"] > 1.5 or p["med_ratio"] < 0.5]
    ok_pairs = [p for p in pair_stats if 0.7 <= p["med_ratio"] <= 1.5]

    n_bad_samples = sum(p["n"] for p in bad_pairs)
    n_ok_samples = sum(p["n"] for p in ok_pairs)
    n_total = sum(p["n"] for p in pair_stats)

    print(f"\n  Parells anòmals: {len(bad_pairs)} (amb {n_bad_samples} mostres)")
    print(f"  Parells OK:      {len(ok_pairs)} (amb {n_ok_samples} mostres)")
    print(f"  Total:           {len(pair_stats)} parells, {n_total} mostres")

    if bad_pairs:
        print(f"\n  Detall parells anòmals:")
        for p in sorted(bad_pairs, key=lambda x: x["med_ratio"], reverse=True):
            print(f"    {p['pair']:>40s}  n={p['n']:2d}  ratio={p['med_ratio']:.3f}  "
                  f"ppm_COL={p['med_c']:.2f}  ppm_BP={p['med_b']:.2f}")

    # ================================================================
    # 3. SENSE PARELLS ANÒMALS: quin és el ratio?
    # ================================================================
    print("\n" + "=" * 80)
    print("3. ESTADÍSTIQUES SENSE PARELLS ANÒMALS")
    print("=" * 80)

    ok_rows = [r for r in rows
               if any(r["pair"] == p["pair"] for p in ok_pairs)]
    if ok_rows:
        ratios_ok = [r["ratio"] for r in ok_rows]
        c_ppms_ok = np.array([r["c_ppm"] for r in ok_rows])
        b_ppms_ok = np.array([r["b_ppm"] for r in ok_rows])

        print(f"\n  N = {len(ok_rows)} mostres de {len(ok_pairs)} parells")
        print(f"  Ratio mediana: {np.median(ratios_ok):.3f}")
        print(f"  Ratio mitjana: {np.mean(ratios_ok):.3f} ± {np.std(ratios_ok):.3f}")

        # Regressió
        slope, intercept = np.polyfit(c_ppms_ok, b_ppms_ok, 1)
        ss_res = np.sum((b_ppms_ok - (slope * c_ppms_ok + intercept)) ** 2)
        ss_tot = np.sum((b_ppms_ok - np.mean(b_ppms_ok)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        print(f"  Regressió: BP = {slope:.3f} * COL + {intercept:.3f}  (R²={r2:.4f})")

        # Per rang concentració
        ranges = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 10), (10, 20), (20, 999)]
        print(f"\n  Per rang concentració (sense parells anòmals):")
        print(f"  {'Rang':>10s}  {'n':>4s}  {'ratio med':>10s}  {'ratio mit':>10s}")
        for lo, hi in ranges:
            m = (c_ppms_ok >= lo) & (c_ppms_ok < hi)
            if m.sum() < 2:
                continue
            rr = np.array(ratios_ok)[m]
            print(f"    {lo:>2}-{hi:<3} ppm  n={m.sum():3d}  {np.median(rr):10.3f}  "
                  f"{np.mean(rr):10.3f}")

    # ================================================================
    # 4. DETALL MOSTRES amb ratio > 2 (individual)
    # ================================================================
    print("\n" + "=" * 80)
    print("4. MOSTRES INDIVIDUALS AMB RATIO > 2.0")
    print("=" * 80)

    high_ratio = sorted([r for r in rows if r["ratio"] > 2.0],
                        key=lambda x: x["ratio"], reverse=True)
    print(f"\n  N = {len(high_ratio)} mostres amb ratio > 2.0")
    print(f"\n  {'Mostra':>15s}  {'COL SEQ':>15s}  {'BP SEQ':>15s}  "
          f"{'ppm_COL':>8s}  {'ppm_BP':>8s}  {'ratio':>7s}")
    for r in high_ratio[:30]:
        print(f"  {r['name']:>15s}  {r['col_seq']:>15s}  {r['bp_seq']:>15s}  "
              f"{r['c_ppm']:8.2f}  {r['b_ppm']:8.2f}  {r['ratio']:7.2f}")

    # ================================================================
    # 5. DETALL MOSTRES amb ratio < 0.5
    # ================================================================
    print("\n" + "=" * 80)
    print("5. MOSTRES INDIVIDUALS AMB RATIO < 0.5")
    print("=" * 80)

    low_ratio = sorted([r for r in rows if r["ratio"] < 0.5],
                       key=lambda x: x["ratio"])
    print(f"\n  N = {len(low_ratio)} mostres amb ratio < 0.5")
    print(f"\n  {'Mostra':>15s}  {'COL SEQ':>15s}  {'BP SEQ':>15s}  "
          f"{'ppm_COL':>8s}  {'ppm_BP':>8s}  {'ratio':>7s}")
    for r in low_ratio[:20]:
        print(f"  {r['name']:>15s}  {r['col_seq']:>15s}  {r['bp_seq']:>15s}  "
              f"{r['c_ppm']:8.2f}  {r['b_ppm']:8.2f}  {r['ratio']:7.2f}")

    # ================================================================
    # PLOT
    # ================================================================
    os.makedirs(OUT_DIR, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    c_ppms_all = np.array([r["c_ppm"] for r in rows])
    b_ppms_all = np.array([r["b_ppm"] for r in rows])
    ratios_all = np.array([r["ratio"] for r in rows])

    # Classificar punts
    is_bad = np.array([any(r["pair"] == p["pair"] for p in bad_pairs) for r in rows])
    is_ok = ~is_bad

    # 1. Scatter TOT amb color per ok/anòmal
    ax = axes[0, 0]
    lim = min(30, max(np.percentile(c_ppms_all, 95), np.percentile(b_ppms_all, 95)))
    ax.scatter(c_ppms_all[is_ok], b_ppms_all[is_ok], s=15, alpha=0.5,
               c='steelblue', label=f'OK ({is_ok.sum()})')
    ax.scatter(c_ppms_all[is_bad], b_ppms_all[is_bad], s=25, alpha=0.7,
               c='red', marker='x', label=f'Anòmal ({is_bad.sum()})')
    ax.plot([0, lim], [0, lim], 'k--', lw=1)
    ax.set_xlabel("ppm COLUMN")
    ax.set_ylabel("ppm BP")
    ax.set_title("Tots els parells")
    ax.legend(fontsize=8)
    ax.set_xlim(0, lim * 1.1)
    ax.set_ylim(0, lim * 1.1)

    # 2. Scatter NOMES OK
    ax = axes[0, 1]
    if ok_rows:
        c_ok = np.array([r["c_ppm"] for r in ok_rows])
        b_ok = np.array([r["b_ppm"] for r in ok_rows])
        ax.scatter(c_ok, b_ok, s=15, alpha=0.5, c='steelblue')
        ax.plot([0, lim], [0, lim], 'k--', lw=1)
        sl, ic = np.polyfit(c_ok, b_ok, 1)
        ax.plot([0, lim], [ic, sl * lim + ic], 'r-', lw=1.5,
                label=f'{sl:.3f}x + {ic:.2f} (R²={r2:.3f})')
        ax.set_xlabel("ppm COLUMN")
        ax.set_ylabel("ppm BP")
        ax.set_title(f"Sense parells anòmals (n={len(ok_rows)})")
        ax.legend(fontsize=8)
        ax.set_xlim(0, lim * 1.1)
        ax.set_ylim(0, lim * 1.1)

    # 3. Ratio vs concentració: ok vs anòmal
    ax = axes[0, 2]
    ax.scatter(c_ppms_all[is_ok], ratios_all[is_ok], s=15, alpha=0.5,
               c='steelblue', label='OK')
    ax.scatter(c_ppms_all[is_bad], ratios_all[is_bad], s=25, alpha=0.7,
               c='red', marker='x', label='Anòmal')
    ax.axhline(1.0, color='k', ls='--', lw=1)
    ax.set_xlabel("ppm COLUMN")
    ax.set_ylabel("Ratio BP/COL")
    ax.set_title("Ratio vs concentració")
    ax.set_ylim(0, 5)
    ax.set_xlim(0, min(25, np.percentile(c_ppms_all, 95)))
    ax.legend(fontsize=8)

    # 4. Ratio per parell (barres horitzontals)
    ax = axes[1, 0]
    sorted_ps = sorted(pair_stats, key=lambda x: x["med_ratio"])
    y_pos = range(len(sorted_ps))
    colors = ['red' if p["med_ratio"] > 1.5 or p["med_ratio"] < 0.5 else 'steelblue'
              for p in sorted_ps]
    ax.barh(list(y_pos), [p["med_ratio"] for p in sorted_ps], color=colors, alpha=0.7)
    ax.axvline(1.0, color='k', ls='--', lw=1)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels([p["pair"].replace(" / ", "\n") for p in sorted_ps], fontsize=5)
    ax.set_xlabel("Ratio mediana BP/COL")
    ax.set_title("Ratio per parell de seqüències")
    ax.set_xlim(0, min(5, max(p["med_ratio"] for p in sorted_ps) * 1.1))

    # 5. Histogram ratio OK
    ax = axes[1, 1]
    if ok_rows:
        r_ok = np.array([r["ratio"] for r in ok_rows])
        r_ok_clean = r_ok[(r_ok > 0.3) & (r_ok < 3)]
        ax.hist(r_ok_clean, bins=40, color='steelblue', alpha=0.7, edgecolor='white')
        ax.axvline(1.0, color='k', ls='--', lw=1)
        ax.axvline(np.median(r_ok_clean), color='red', ls='-', lw=2,
                   label=f'mediana={np.median(r_ok_clean):.3f}')
        ax.set_xlabel("Ratio BP/COL")
        ax.set_ylabel("Freqüència")
        ax.set_title(f"Distribució ratio (parells OK, n={len(r_ok_clean)})")
        ax.legend()

    # 6. Ratio per rang concentració: totes vs OK
    ax = axes[1, 2]
    ranges = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 10), (10, 20)]
    x_pos = np.arange(len(ranges))
    all_medians = []
    ok_medians = []
    labels = []
    for lo, hi in ranges:
        m_all = (c_ppms_all >= lo) & (c_ppms_all < hi)
        all_medians.append(np.median(ratios_all[m_all]) if m_all.sum() >= 2 else np.nan)
        if ok_rows:
            m_ok = (c_ok >= lo) & (c_ok < hi) if 'c_ok' in dir() else np.zeros(0, bool)
            r_filt = np.array([r["ratio"] for r in ok_rows])
            ok_medians.append(np.median(r_filt[m_ok]) if m_ok.sum() >= 2 else np.nan)
        else:
            ok_medians.append(np.nan)
        labels.append(f"{lo}-{hi}")

    w = 0.35
    ax.bar(x_pos - w / 2, all_medians, w, label='Totes', color='coral', alpha=0.7)
    ax.bar(x_pos + w / 2, ok_medians, w, label='Sense anòmals', color='steelblue', alpha=0.7)
    ax.axhline(1.0, color='k', ls='--', lw=1)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Rang ppm COLUMN")
    ax.set_ylabel("Ratio mediana BP/COL")
    ax.set_title("Impacte d'excloure parells anòmals")
    ax.legend(fontsize=8)
    ax.set_ylim(0, min(4, max(filter(lambda x: not np.isnan(x), all_medians)) * 1.2))

    fig.suptitle("Anàlisi outliers: quines SEQs causen el ratio anòmal?",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "outlier_seqs.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nGràfic: {path}")


if __name__ == "__main__":
    main()
