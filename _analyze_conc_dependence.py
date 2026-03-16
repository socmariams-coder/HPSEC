"""
Anàlisi: per què la diferència COLUMN-BP depèn de la concentració?

Separa 3 nivells:
  1. Àrees crues (normalitzades per volum) — efecte REAL del detector/integració
  2. Àrees amb RF — efecte calibració (RF_COL vs RF_BP)
  3. ppm (àrees amb RF + intercept) — efecte complet

Si el patró concentració-dependent persisteix a nivell 1, és un efecte REAL.
Si apareix només al nivell 3, és un artefacte de calibració (intercept).
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
OUT_DIR = os.path.join(DATA_DIR, "REGISTRY", "review", "conc_dependence")

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

            c_area = avg_field(cr, "areas", "DOC", "total")
            b_area = avg_field(br, "areas", "DOC", "total")
            if c_area <= 0 or b_area <= 0:
                continue

            # ppm amb calibració actual
            c_ppm = (c_area - IC_COL) * 1000 / (RF_COL * VOL_COL)
            b_ppm = (b_area - IC_BP) * 1000 / (RF_BP * VOL_BP)

            rows.append({
                "name": name,
                "col_seq": ck, "bp_seq": best_bp,
                "c_area": c_area, "b_area": b_area,
                "c_ppm": c_ppm, "b_ppm": b_ppm,
                "col_num": cv["num"],
            })
    return rows


def main():
    rows = load_pairs()
    valid = [r for r in rows if r["c_ppm"] > 0 and r["b_ppm"] > 0]
    print(f"Parells: {len(valid)}")

    c_areas = np.array([r["c_area"] for r in valid])
    b_areas = np.array([r["b_area"] for r in valid])
    c_ppms = np.array([r["c_ppm"] for r in valid])
    b_ppms = np.array([r["b_ppm"] for r in valid])

    # ====================================================================
    # NIVELL 1: Àrees normalitzades per volum (sense RF, sense intercept)
    # ====================================================================
    # area/vol = proporcional a concentració real si detector lineal
    c_area_vol = c_areas / VOL_COL  # area per µL
    b_area_vol = b_areas / VOL_BP

    print("\n" + "=" * 70)
    print("NIVELL 1: ÀREES NORMALITZADES PER VOLUM (area/µL)")
    print("  (Si ratio ≠ 1.0, la diferència és REAL — no calibració)")
    print("=" * 70)

    ratio_area = b_area_vol / c_area_vol
    print(f"  Ratio global (BP/COL):  {np.median(ratio_area):.3f} (mediana)")
    print(f"                          {np.mean(ratio_area):.3f} (mitjana)")

    # Regressió lineal area/vol
    slope_a, intercept_a = np.polyfit(c_area_vol, b_area_vol, 1)
    ss_res = np.sum((b_area_vol - (slope_a * c_area_vol + intercept_a)) ** 2)
    ss_tot = np.sum((b_area_vol - np.mean(b_area_vol)) ** 2)
    r2_a = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    print(f"  Regressió: BP_area/vol = {slope_a:.3f} * COL_area/vol + {intercept_a:.3f}")
    print(f"  R² = {r2_a:.4f}")
    print(f"  Intercept en area/µL = {intercept_a:.3f}")
    print(f"  Intercept equivalent en ppm ≈ {intercept_a * 1000 / RF_BP:.2f} (via RF_BP)")

    # Per rang de concentració (usant c_ppm com a referència)
    print(f"\n  Per rang de concentració:")
    ranges = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 10), (10, 20), (20, 999)]
    for lo, hi in ranges:
        mask = (c_ppms >= lo) & (c_ppms < hi)
        if mask.sum() < 2:
            continue
        r = ratio_area[mask]
        print(f"    {lo:>2}-{hi:<3} ppm: n={mask.sum():3d}  "
              f"ratio area/vol = {np.median(r):.3f}  "
              f"(mitjana {np.mean(r):.3f})")

    # ====================================================================
    # NIVELL 2: Què passa si ajustem RF per igualar mitjana?
    # ====================================================================
    print("\n" + "=" * 70)
    print("NIVELL 2: CALIBRACIÓ — QUÈ HAURIA DE SER RF_BP?")
    print("=" * 70)

    # Si area/vol = proporcional, RF_BP/RF_COL hauria de ser = slope
    print(f"  RF_COL actual = {RF_COL:.1f}")
    print(f"  RF_BP actual  = {RF_BP:.1f}")
    print(f"  Ratio RF: RF_BP/RF_COL = {RF_BP / RF_COL:.4f}")
    print(f"  Ratio àrees/vol: slope = {slope_a:.4f}")
    print()

    # El RF ideal per BP seria aquell que fa ratio ppm = 1.0
    # ppm_col = (c_area - IC_COL) * 1000 / (RF_COL * VOL_COL)
    # ppm_bp  = (b_area - IC_BP_new) * 1000 / (RF_BP_new * VOL_BP)
    # Volem ppm_bp = ppm_col

    # Mètode 1: regressió àrees crues (area_COL vs area_BP)
    slope_raw, intercept_raw = np.polyfit(c_areas, b_areas, 1)
    ss_res2 = np.sum((b_areas - (slope_raw * c_areas + intercept_raw)) ** 2)
    ss_tot2 = np.sum((b_areas - np.mean(b_areas)) ** 2)
    r2_raw = 1 - ss_res2 / ss_tot2 if ss_tot2 > 0 else 0
    print(f"  Regressió àrees crues: area_BP = {slope_raw:.4f} * area_COL + {intercept_raw:.2f}")
    print(f"  R² = {r2_raw:.4f}")
    print(f"  → Interpretació:")
    print(f"    slope={slope_raw:.4f}: per cada unitat d'àrea COLUMN, BP dóna {slope_raw:.4f}")
    print(f"    La relació volumètrica teòrica és VOL_BP/VOL_COL = {VOL_BP/VOL_COL:.4f}")
    print(f"    Ratio slope/volumètric = {slope_raw / (VOL_BP/VOL_COL):.4f}")
    print(f"    Si fos 1.000 → detector idèntic. Si ≠ 1 → diferència real detector/integració")
    print(f"    intercept={intercept_raw:.2f}: offset constant entre àrees")

    # ====================================================================
    # NIVELL 3: Quin intercept_BP eliminaria la dependència concentració?
    # ====================================================================
    print("\n" + "=" * 70)
    print("NIVELL 3: QUIN INTERCEPT BP ELIMINARIA EL BIAIX?")
    print("=" * 70)

    # Busquem IC_BP_opt tal que mediana(ratio_ppm) = 1.0
    # ppm_bp = (b_area - IC_BP_opt) * 1000 / (RF_BP * VOL_BP)
    # ratio = ppm_bp / ppm_col
    # Volem mediana(ratio) = 1.0

    best_ic, best_dev = 0, 999
    for ic_test in np.arange(-50, 100, 0.5):
        ppm_bp_test = (b_areas - ic_test) * 1000 / (RF_BP * VOL_BP)
        mask = (ppm_bp_test > 0) & (c_ppms > 0)
        if mask.sum() < 50:
            continue
        ratio_test = ppm_bp_test[mask] / c_ppms[mask]
        dev = abs(np.median(ratio_test) - 1.0)
        if dev < best_dev:
            best_ic, best_dev = ic_test, dev

    print(f"  Intercept BP actual:  {IC_BP:.1f}")
    print(f"  Intercept BP òptim:   {best_ic:.1f}  (per mediana ratio = 1.0)")

    # Test amb intercept òptim
    ppm_bp_opt = (b_areas - best_ic) * 1000 / (RF_BP * VOL_BP)
    mask_opt = (ppm_bp_opt > 0) & (c_ppms > 0)
    c_ppms_opt = c_ppms[mask_opt]
    b_ppms_opt = b_ppms[mask_opt]
    ratio_opt = ppm_bp_opt[mask_opt] / c_ppms_opt
    print(f"  Amb IC_BP={best_ic:.1f}: mediana ratio = {np.median(ratio_opt):.3f}")
    print(f"\n  Per rang de concentració amb IC_BP={best_ic:.1f}:")
    for lo, hi in ranges:
        m_opt = (c_ppms_opt >= lo) & (c_ppms_opt < hi)
        m_old = (c_ppms >= lo) & (c_ppms < hi)
        if m_opt.sum() < 2:
            continue
        r = ratio_opt[m_opt]
        old_r = (b_ppms[m_old] / c_ppms[m_old])
        print(f"    {lo:>2}-{hi:<3} ppm: n={m_opt.sum():3d}  ratio = {np.median(r):.3f}  (era {np.median(old_r):.3f})")

    # ====================================================================
    # NIVELL 4: Optimitzar RF_BP + IC_BP conjuntament
    # ====================================================================
    print("\n" + "=" * 70)
    print("NIVELL 4: OPTIMITZAR RF_BP + IC_BP CONJUNTAMENT")
    print("=" * 70)

    # Si ppm_bp = ppm_col per a tots, llavors:
    # (b_area - IC_BP) / (RF_BP * VOL_BP) = (c_area - IC_COL) / (RF_COL * VOL_COL)
    # b_area - IC_BP = RF_BP * VOL_BP * (c_area - IC_COL) / (RF_COL * VOL_COL)
    # b_area = (RF_BP * VOL_BP / (RF_COL * VOL_COL)) * c_area
    #        - (RF_BP * VOL_BP / (RF_COL * VOL_COL)) * IC_COL + IC_BP
    # b_area = K * c_area + (IC_BP - K * IC_COL)
    # on K = RF_BP * VOL_BP / (RF_COL * VOL_COL)

    # De la regressió àrees crues: b_area = slope_raw * c_area + intercept_raw
    # Doncs: K = slope_raw → RF_BP_opt = slope_raw * RF_COL * VOL_COL / VOL_BP
    RF_BP_opt = slope_raw * RF_COL * VOL_COL / VOL_BP
    IC_BP_opt = intercept_raw + slope_raw * IC_COL

    print(f"  De la regressió àrees: area_BP = {slope_raw:.4f} * area_COL + {intercept_raw:.2f}")
    print(f"  RF_BP teòric:  {RF_BP_opt:.1f}  (actual: {RF_BP:.1f}, diferència: {(RF_BP_opt - RF_BP) / RF_BP * 100:+.1f}%)")
    print(f"  IC_BP teòric:  {IC_BP_opt:.2f}  (actual: {IC_BP:.1f})")

    # Verificar: amb RF i IC òptims
    ppm_bp_v4 = (b_areas - IC_BP_opt) * 1000 / (RF_BP_opt * VOL_BP)
    mask_v4 = (ppm_bp_v4 > 0) & (c_ppms > 0)
    c_ppms_v4 = c_ppms[mask_v4]
    ratio_v4 = ppm_bp_v4[mask_v4] / c_ppms_v4
    print(f"\n  Verificació amb RF={RF_BP_opt:.1f}, IC={IC_BP_opt:.2f}:")
    print(f"    Mediana ratio = {np.median(ratio_v4):.3f}")
    for lo, hi in ranges:
        m_v4 = (c_ppms_v4 >= lo) & (c_ppms_v4 < hi)
        m_old = (c_ppms >= lo) & (c_ppms < hi)
        if m_v4.sum() < 2:
            continue
        r = ratio_v4[m_v4]
        old_r = (b_ppms[m_old] / c_ppms[m_old])
        print(f"    {lo:>2}-{hi:<3} ppm: n={m_v4.sum():3d}  ratio = {np.median(r):.3f}  (era {np.median(old_r):.3f})")

    # ====================================================================
    # NIVELL 5: Àrees per fracció — quina fracció infla COLUMN?
    # ====================================================================
    print("\n" + "=" * 70)
    print("NIVELL 5: SI BP ≈ AREA HS COLUMN, ON VA LA RESTA?")
    print("=" * 70)

    # Carregar fraccions
    frac_data = []
    for r in valid:
        # Re-read analysis for fractions
        f = os.path.join(DATA_DIR, r["col_seq"], "CHECK", "data", "analysis_result.json")
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
        for s in data["samples"]:
            if s["name"] == r["name"]:
                fracs = s.get("areas", {}).get("DOC", {})
                total = fracs.get("total", 0)
                if total <= 0:
                    break
                frac_data.append({
                    "name": r["name"],
                    "total": total,
                    "HS": fracs.get("HS", 0),
                    "BB": fracs.get("BB", 0),
                    "SB": fracs.get("SB", 0),
                    "LMW": fracs.get("LMW", 0),
                    "BioP": fracs.get("BioP", 0),
                    "b_area": r["b_area"],
                    "c_ppm": r["c_ppm"],
                })
                break

    if frac_data:
        # Comparar area_BP/vol vs area_HS_COL/vol
        hs_col = np.array([f["HS"] for f in frac_data]) / VOL_COL
        bp_vol = np.array([f["b_area"] for f in frac_data]) / VOL_BP
        total_col = np.array([f["total"] for f in frac_data]) / VOL_COL
        c_ppms_f = np.array([f["c_ppm"] for f in frac_data])

        ratio_hs = bp_vol / np.where(hs_col > 0, hs_col, 0.001)
        ratio_total = bp_vol / total_col

        print(f"  N = {len(frac_data)}")
        print(f"\n  Ratio BP vs HS_COLUMN (area/vol):")
        print(f"    Mediana: {np.median(ratio_hs):.3f}")
        print(f"    → Si ≈ 1.0: BP mesura el MATEIX que la fracció HS")
        print(f"    → Si > 1.0: BP mesura MÉS (inclou fracció no-HS)")

        print(f"\n  Ratio BP vs TOTAL_COLUMN (area/vol):")
        print(f"    Mediana: {np.median(ratio_total):.3f}")

        print(f"\n  Per rang de concentració:")
        print(f"    {'Rang':>10s}  {'n':>4s}  {'BP/HS':>7s}  {'BP/Total':>9s}  {'HS/Total':>9s}")
        for lo, hi in ranges:
            m = (c_ppms_f >= lo) & (c_ppms_f < hi)
            if m.sum() < 2:
                continue
            rhs = np.median(ratio_hs[m])
            rtot = np.median(ratio_total[m])
            hs_frac = np.median(np.array([f["HS"] for f in frac_data])[m] /
                               np.array([f["total"] for f in frac_data])[m])
            print(f"    {lo:>2}-{hi:<3} ppm  n={m.sum():3d}  {rhs:7.3f}  {rtot:9.3f}  {hs_frac:9.1%}")

    # ====================================================================
    # PLOT
    # ====================================================================
    os.makedirs(OUT_DIR, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # 1. Scatter àrees/vol
    ax = axes[0, 0]
    ax.scatter(c_areas / VOL_COL, b_areas / VOL_BP, s=15, alpha=0.5, c='steelblue')
    lim = max(np.percentile(c_areas / VOL_COL, 95), np.percentile(b_areas / VOL_BP, 95))
    ax.plot([0, lim], [0, lim], 'k--', lw=1, label='1:1')
    ax.plot([0, lim], [intercept_a, slope_a * lim + intercept_a], 'r-', lw=1.5,
            label=f'fit: {slope_a:.3f}x + {intercept_a:.2f}')
    ax.set_xlabel("COLUMN area/µL")
    ax.set_ylabel("BP area/µL")
    ax.set_title(f"Àrees normalitzades per volum (R²={r2_a:.3f})")
    ax.legend(fontsize=8)
    ax.set_xlim(0, lim * 1.1)
    ax.set_ylim(0, lim * 1.1)

    # 2. Ratio vs concentració (actual)
    ax = axes[0, 1]
    ax.scatter(c_ppms, b_ppms / c_ppms, s=15, alpha=0.5, c='coral')
    ax.axhline(1.0, color='k', ls='--', lw=1)
    ax.set_xlabel("ppm COLUMN")
    ax.set_ylabel("Ratio BP/COL (ppm)")
    ax.set_title("Ratio vs concentració (calibració ACTUAL)")
    ax.set_ylim(0, 5)
    ax.set_xlim(0, min(30, np.percentile(c_ppms, 98)))

    # 3. Ratio vs concentració (optimitzat)
    ax = axes[0, 2]
    if mask_v4.sum() > 0:
        c_ppms_v4 = c_ppms[mask_v4]
        ax.scatter(c_ppms_v4, ratio_v4, s=15, alpha=0.5, c='mediumseagreen')
        ax.axhline(1.0, color='k', ls='--', lw=1)
        ax.set_xlabel("ppm COLUMN")
        ax.set_ylabel("Ratio BP/COL (ppm)")
        ax.set_title(f"Ratio amb RF_BP={RF_BP_opt:.0f}, IC_BP={IC_BP_opt:.1f}")
        ax.set_ylim(0, 5)
        ax.set_xlim(0, min(30, np.percentile(c_ppms_v4, 98)))

    # 4. Scatter ppm actual
    ax = axes[1, 0]
    ax.scatter(c_ppms, b_ppms, s=15, alpha=0.5, c='coral')
    lim_p = min(30, max(np.percentile(c_ppms, 95), np.percentile(b_ppms, 95)))
    ax.plot([0, lim_p], [0, lim_p], 'k--', lw=1)
    slope_p, intercept_p = np.polyfit(c_ppms, b_ppms, 1)
    ax.plot([0, lim_p], [intercept_p, slope_p * lim_p + intercept_p], 'r-', lw=1.5,
            label=f'{slope_p:.2f}x + {intercept_p:.2f}')
    ax.set_xlabel("ppm COLUMN")
    ax.set_ylabel("ppm BP")
    ax.set_title("ppm actual")
    ax.legend(fontsize=8)
    ax.set_xlim(0, lim_p * 1.1)
    ax.set_ylim(0, lim_p * 1.1)

    # 5. Scatter ppm optimitzat
    ax = axes[1, 1]
    if mask_v4.sum() > 0:
        ax.scatter(c_ppms[mask_v4], ppm_bp_v4[mask_v4], s=15, alpha=0.5, c='mediumseagreen')
        ax.plot([0, lim_p], [0, lim_p], 'k--', lw=1)
        sl2, ic2 = np.polyfit(c_ppms[mask_v4], ppm_bp_v4[mask_v4], 1)
        ax.plot([0, lim_p], [ic2, sl2 * lim_p + ic2], 'r-', lw=1.5,
                label=f'{sl2:.2f}x + {ic2:.2f}')
        ss_res_v4 = np.sum((ppm_bp_v4[mask_v4] - (sl2 * c_ppms[mask_v4] + ic2)) ** 2)
        ss_tot_v4 = np.sum((ppm_bp_v4[mask_v4] - np.mean(ppm_bp_v4[mask_v4])) ** 2)
        r2_v4 = 1 - ss_res_v4 / ss_tot_v4 if ss_tot_v4 > 0 else 0
        ax.set_xlabel("ppm COLUMN")
        ax.set_ylabel("ppm BP (optimitzat)")
        ax.set_title(f"ppm amb RF_BP={RF_BP_opt:.0f}, IC_BP={IC_BP_opt:.1f} (R²={r2_v4:.3f})")
        ax.legend(fontsize=8)
        ax.set_xlim(0, lim_p * 1.1)
        ax.set_ylim(0, lim_p * 1.1)

    # 6. Histograma ratio àrees/vol
    ax = axes[1, 2]
    ratio_clean = ratio_area[(ratio_area > 0.2) & (ratio_area < 5)]
    ax.hist(ratio_clean, bins=40, color='steelblue', alpha=0.7, edgecolor='white')
    ax.axvline(1.0, color='k', ls='--', lw=1)
    ax.axvline(np.median(ratio_clean), color='red', ls='-', lw=2,
               label=f'mediana={np.median(ratio_clean):.3f}')
    ax.set_xlabel("Ratio BP/COL (area/µL)")
    ax.set_ylabel("Freqüència")
    ax.set_title("Distribució ratio àrees/vol")
    ax.legend()

    fig.suptitle("Anàlisi dependència concentració: COLUMN vs BP", fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "conc_dependence.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nGràfic: {path}")


if __name__ == "__main__":
    main()
