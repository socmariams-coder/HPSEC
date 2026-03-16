"""
Hipòtesi: a baixa concentració, les fraccions COLUMN petites cauen
per sota del llindar d'integració i es perden → COLUMN dóna MENYS que BP.

Verificació:
1. Per cada parell COL/BP, calcular quantes fraccions COLUMN són ≈0
2. Correlacionar "fraccions perdudes" amb el ratio BP/COL
3. Mostrar que a alta concentració totes les fraccions tenen àrea → ratio ≈ 1
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
OUT_DIR = os.path.join(DATA_DIR, "REGISTRY", "review", "frac_loss")

RF_COL = 752.90; RF_BP = 646.82
IC_COL = 18.11; IC_BP = 2.87
VOL_COL = 400.0; VOL_BP = 100.0
MAX_SEQ_GAP = 5
EXCL = ("MQ", "NAOH", "BUFFER", "KHP", "NAN")
FRAC_NAMES = ["BioP", "HS", "BB", "SB", "LMW"]
MIN_AREA = 5.0  # àrea mínima per considerar "detectada"


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
            n = s["name"]
            if not n.upper().startswith(EXCL):
                c_by.setdefault(n, []).append(s)
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

            c_ppm = (c_total - IC_COL) * 1000 / (RF_COL * VOL_COL)
            b_ppm = (b_total - IC_BP) * 1000 / (RF_BP * VOL_BP)
            if c_ppm <= 0 or b_ppm <= 0:
                continue

            # Fraccions
            fracs = {}
            for fn in FRAC_NAMES:
                fracs[fn] = avg_field(cr, "areas", "DOC", fn, default=0)

            n_detected = sum(1 for fn in FRAC_NAMES if fracs[fn] > MIN_AREA)
            n_zero = 5 - n_detected
            sum_fracs = sum(fracs.values())
            frac_pcts = {fn: fracs[fn] / sum_fracs * 100 if sum_fracs > 0 else 0
                         for fn in FRAC_NAMES}

            # Àrea BP equivalent a COLUMN (ajustant per volum)
            b_equiv = b_total * VOL_COL / VOL_BP  # àrea que BP donaria amb 400µL

            # "Pèrdua" COLUMN = àrea BP equivalent - àrea COLUMN total
            loss = b_equiv - c_total
            loss_pct = loss / b_equiv * 100 if b_equiv > 0 else 0

            rows.append({
                "name": name, "col_seq": ck, "bp_seq": best_bp,
                "c_total": c_total, "b_total": b_total,
                "c_ppm": c_ppm, "b_ppm": b_ppm,
                "ratio": b_ppm / c_ppm,
                "fracs": fracs, "frac_pcts": frac_pcts,
                "n_detected": n_detected, "n_zero": n_zero,
                "sum_fracs": sum_fracs,
                "loss": loss, "loss_pct": loss_pct,
                "b_equiv": b_equiv,
                "col_num": cv["num"],
            })
    return rows


def main():
    rows = load_pairs()
    print(f"Parells: {len(rows)}")

    c_ppms = np.array([r["c_ppm"] for r in rows])
    ratios = np.array([r["ratio"] for r in rows])
    n_zeros = np.array([r["n_zero"] for r in rows])
    losses = np.array([r["loss_pct"] for r in rows])

    # ====================================================================
    # 1. FRACCIONS "PERDUDES" vs CONCENTRACIÓ
    # ====================================================================
    print("\n" + "=" * 70)
    print("1. FRACCIONS DETECTADES vs CONCENTRACIÓ")
    print(f"   (llindar detecció: àrea > {MIN_AREA})")
    print("=" * 70)

    ranges = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 10), (10, 20), (20, 999)]
    print(f"\n  {'Rang':>10s}  {'n':>4s}  {'fracs det':>10s}  {'fracs=0':>8s}  {'ratio':>7s}  {'loss%':>7s}")
    for lo, hi in ranges:
        m = (c_ppms >= lo) & (c_ppms < hi)
        if m.sum() < 2:
            continue
        med_det = np.median(n_zeros[m])
        med_ratio = np.median(ratios[m])
        med_loss = np.median(losses[m])
        print(f"    {lo:>2}-{hi:<3} ppm  n={m.sum():3d}  "
              f"det={np.median(np.array([r['n_detected'] for r in rows])[m]):.0f}/5  "
              f"zero={med_det:.0f}  "
              f"ratio={med_ratio:.3f}  "
              f"loss={med_loss:+.1f}%")

    # ====================================================================
    # 2. DETALL PER FRACCIÓ — quina es perd primer?
    # ====================================================================
    print("\n" + "=" * 70)
    print("2. PERCENTATGE DE MOSTRES AMB FRACCIÓ DETECTADA, PER RANG")
    print("=" * 70)

    print(f"\n  {'Rang':>10s}  {'n':>4s}  {'BioP':>6s}  {'HS':>6s}  {'BB':>6s}  {'SB':>6s}  {'LMW':>6s}")
    for lo, hi in ranges:
        m = (c_ppms >= lo) & (c_ppms < hi)
        if m.sum() < 2:
            continue
        row_subset = [r for r, ok in zip(rows, m) if ok]
        pcts = {}
        for fn in FRAC_NAMES:
            detected = sum(1 for r in row_subset if r["fracs"][fn] > MIN_AREA)
            pcts[fn] = detected / len(row_subset) * 100
        print(f"    {lo:>2}-{hi:<3} ppm  n={len(row_subset):3d}  "
              + "  ".join(f"{pcts[fn]:5.0f}%" for fn in FRAC_NAMES))

    # ====================================================================
    # 3. "PÈRDUA" COLUMN = BP_equiv - COL_total
    # ====================================================================
    print("\n" + "=" * 70)
    print("3. PÈRDUA COLUMN vs BP (àrea)")
    print("   Pèrdua = àrea_BP*4 - àrea_COL (positiu = COL perd)")
    print("=" * 70)

    for lo, hi in ranges:
        m = (c_ppms >= lo) & (c_ppms < hi)
        if m.sum() < 2:
            continue
        row_subset = [r for r, ok in zip(rows, m) if ok]
        med_loss_area = np.median([r["loss"] for r in row_subset])
        med_loss_pct = np.median([r["loss_pct"] for r in row_subset])
        med_col = np.median([r["c_total"] for r in row_subset])
        med_bp_eq = np.median([r["b_equiv"] for r in row_subset])
        print(f"    {lo:>2}-{hi:<3} ppm: COL={med_col:.0f}  BP*4={med_bp_eq:.0f}  "
              f"Δ={med_loss_area:+.0f}  ({med_loss_pct:+.1f}%)")

    # ====================================================================
    # 4. CORRELACIÓ entre fraccions=0 i ratio
    # ====================================================================
    print("\n" + "=" * 70)
    print("4. CORRELACIÓ FRACCIONS PERDUDES ↔ RATIO")
    print("=" * 70)

    for nz in range(6):
        mask = n_zeros == nz
        if mask.sum() < 3:
            continue
        med_ratio = np.median(ratios[mask])
        med_ppm = np.median(c_ppms[mask])
        print(f"  {nz} fraccions ≈ 0: n={mask.sum():3d}  ratio mediana={med_ratio:.3f}  "
              f"ppm mediana={med_ppm:.2f}")

    # ====================================================================
    # 5. SI RESTEM OVERHEAD COLUMN, COINCIDEIX AMB BP?
    # ====================================================================
    print("\n" + "=" * 70)
    print("5. COMPARACIÓ: BP vs FRACCIÓ DOMINANT COLUMN")
    print("   (HS és la fracció principal — comparem BP amb HS sola)")
    print("=" * 70)

    for lo, hi in ranges:
        m = (c_ppms >= lo) & (c_ppms < hi)
        if m.sum() < 2:
            continue
        row_subset = [r for r, ok in zip(rows, m) if ok]
        # ppm de HS sola
        hs_ppms = [(r["fracs"]["HS"] - IC_COL) * 1000 / (RF_COL * VOL_COL)
                    for r in row_subset]
        bp_ppms = [r["b_ppm"] for r in row_subset]
        col_ppms = [r["c_ppm"] for r in row_subset]
        ratio_total = np.median(np.array(bp_ppms) / np.array(col_ppms))
        hs_valid = [h for h in hs_ppms if h > 0]
        bp_valid = [b for b, h in zip(bp_ppms, hs_ppms) if h > 0]
        if hs_valid:
            ratio_hs = np.median(np.array(bp_valid) / np.array(hs_valid))
        else:
            ratio_hs = float('nan')
        print(f"    {lo:>2}-{hi:<3} ppm: ratio BP/Total={ratio_total:.3f}  "
              f"ratio BP/HS={ratio_hs:.3f}")

    # ====================================================================
    # PLOT
    # ====================================================================
    os.makedirs(OUT_DIR, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # 1. Ratio vs concentració amb colors per #fraccions zero
    ax = axes[0, 0]
    colors_nz = {0: 'green', 1: 'blue', 2: 'orange', 3: 'red', 4: 'darkred', 5: 'black'}
    for nz in range(6):
        m = n_zeros == nz
        if m.sum() > 0:
            ax.scatter(c_ppms[m], ratios[m], s=15, alpha=0.5,
                       c=colors_nz.get(nz, 'grey'), label=f'{nz} fracs=0')
    ax.axhline(1.0, color='k', ls='--', lw=1)
    ax.set_xlabel("ppm COLUMN")
    ax.set_ylabel("Ratio BP/COL")
    ax.set_title("Ratio vs conc (color = fraccions perdudes)")
    ax.set_ylim(0, 5)
    ax.set_xlim(0, min(25, np.percentile(c_ppms, 95)))
    ax.legend(fontsize=7, loc='upper right')

    # 2. % fraccions detectades vs concentració
    ax = axes[0, 1]
    for fn, color in zip(FRAC_NAMES, ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#795548']):
        # Binned
        bin_edges = [0, 1, 2, 3, 5, 10, 20]
        bin_centers = []
        pct_detected = []
        for i in range(len(bin_edges) - 1):
            m = (c_ppms >= bin_edges[i]) & (c_ppms < bin_edges[i + 1])
            if m.sum() < 3:
                continue
            row_subset = [r for r, ok in zip(rows, m) if ok]
            detected = sum(1 for r in row_subset if r["fracs"][fn] > MIN_AREA)
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            pct_detected.append(detected / len(row_subset) * 100)
        ax.plot(bin_centers, pct_detected, 'o-', color=color, label=fn, lw=2, markersize=5)
    ax.set_xlabel("ppm COLUMN")
    ax.set_ylabel("% mostres amb fracció detectada")
    ax.set_title(f"Detecció fraccions vs concentració (llindar={MIN_AREA})")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 105)

    # 3. Loss% vs concentració
    ax = axes[0, 2]
    ax.scatter(c_ppms, losses, s=15, alpha=0.5, c='coral')
    ax.axhline(0, color='k', ls='--', lw=1)
    ax.set_xlabel("ppm COLUMN")
    ax.set_ylabel("Pèrdua COLUMN vs BP (%)")
    ax.set_title("Pèrdua COLUMN (positiu = COL < BP)")
    ax.set_xlim(0, min(25, np.percentile(c_ppms, 95)))
    ax.set_ylim(-100, 100)

    # 4. Scatter COL total vs BP*4
    ax = axes[1, 0]
    c_tots = np.array([r["c_total"] for r in rows])
    b_equivs = np.array([r["b_equiv"] for r in rows])
    lim = np.percentile(np.concatenate([c_tots, b_equivs]), 95)
    ax.scatter(c_tots, b_equivs, s=15, alpha=0.5, c='steelblue')
    ax.plot([0, lim], [0, lim], 'k--', lw=1, label='1:1')
    slope, intercept = np.polyfit(c_tots, b_equivs, 1)
    ax.plot([0, lim], [intercept, slope * lim + intercept], 'r-', lw=1.5,
            label=f'{slope:.2f}x + {intercept:.0f}')
    ax.set_xlabel("Àrea COLUMN total")
    ax.set_ylabel("Àrea BP × 4 (equivalent)")
    ax.set_title("Àrees equivalents (ajustades per volum)")
    ax.legend(fontsize=8)
    ax.set_xlim(0, lim * 1.1)
    ax.set_ylim(0, lim * 1.1)

    # 5. Stacked fractions vs concentració (binned)
    ax = axes[1, 1]
    bin_edges = [0, 1, 2, 3, 5, 10, 20, 999]
    bin_labels = ['<1', '1-2', '2-3', '3-5', '5-10', '10-20', '>20']
    frac_medians = {fn: [] for fn in FRAC_NAMES}
    valid_bins = []
    for i in range(len(bin_edges) - 1):
        m = (c_ppms >= bin_edges[i]) & (c_ppms < bin_edges[i + 1])
        if m.sum() < 3:
            continue
        valid_bins.append(bin_labels[i])
        row_subset = [r for r, ok in zip(rows, m) if ok]
        for fn in FRAC_NAMES:
            ppm_f = np.median([(r["fracs"][fn]) * 1000 / (RF_COL * VOL_COL)
                               for r in row_subset])
            frac_medians[fn].append(ppm_f)

    x = np.arange(len(valid_bins))
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#795548']
    bottoms = np.zeros(len(valid_bins))
    for fn, color in zip(FRAC_NAMES, colors):
        vals = frac_medians[fn]
        ax.bar(x, vals, bottom=bottoms, color=color, label=fn, alpha=0.8)
        bottoms += np.array(vals)
    ax.set_xticks(x)
    ax.set_xticklabels(valid_bins)
    ax.set_xlabel("Rang ppm")
    ax.set_ylabel("ppm per fracció (mediana)")
    ax.set_title("Distribució DOC per fracció vs concentració")
    ax.legend(fontsize=7, ncol=5)

    # 6. n_fracs detectades vs ratio
    ax = axes[1, 2]
    for nz in range(6):
        m = n_zeros == nz
        if m.sum() >= 3:
            ax.scatter([nz] * m.sum(), ratios[m], s=10, alpha=0.3,
                       c=colors_nz.get(nz, 'grey'))
            ax.plot(nz, np.median(ratios[m]), 'kD', markersize=8)
    ax.axhline(1.0, color='k', ls='--', lw=1)
    ax.set_xlabel("Nombre de fraccions ≈ 0")
    ax.set_ylabel("Ratio BP/COL")
    ax.set_title("Ratio vs fraccions perdudes")
    ax.set_ylim(0, 5)
    ax.set_xticks(range(6))

    fig.suptitle("Hipòtesi: COLUMN perd senyal a baixa concentració (fraccions sota llindar)",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "frac_loss.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nGràfic: {path}")


if __name__ == "__main__":
    main()
