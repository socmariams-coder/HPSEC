"""
Compara KHP entre COLUMN i BP.

KHP és pic únic i net — NO té fraccions. Si la relació COLUMN/BP
es manté estable al KHP, llavors la diferència en mostres reals
ve de l'acumulació de soroll a les fraccions.

També estima la contribució de soroll per fracció:
  - Àrea de soroll = baseline_std * window_min * punts_per_min
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
OUT_DIR = os.path.join(DATA_DIR, "REGISTRY", "review", "khp_col_bp")

RF_COL = 752.90; RF_BP = 646.82
IC_COL = 18.11; IC_BP = 2.87
VOL_COL = 400.0; VOL_BP = 100.0

# Fraccions COLUMN i les seves finestres típiques (min)
FRACTION_WINDOWS = {
    "BioP": (12.0, 16.5),   # ~4.5 min
    "HS":   (16.5, 22.0),   # ~5.5 min
    "BB":   (22.0, 27.0),   # ~5.0 min
    "SB":   (27.0, 32.0),   # ~5.0 min
    "LMW":  (32.0, 45.0),   # ~13.0 min
}


def seq_num(name):
    m = re.match(r"(\d+)", name)
    return int(m.group(1)) if m else 0


def main():
    # ====================================================================
    # PART 1: KHP des de analysis_result.json (àrees DOC.total)
    # ====================================================================
    print("=" * 70)
    print("PART 1: KHP — COMPARACIÓ ÀREES ENTRE MODES")
    print("=" * 70)

    khp_col = []
    khp_bp = []

    for d in sorted(os.listdir(DATA_DIR)):
        f = os.path.join(DATA_DIR, d, "CHECK", "data", "analysis_result.json")
        if not os.path.isfile(f):
            continue
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
        method = data.get("method", "?")
        if method not in ("COLUMN", "BP"):
            continue

        for s in data.get("samples", []):
            name = s.get("name", "")
            if not name.upper().startswith("KHP"):
                continue
            area = s.get("areas", {}).get("DOC", {}).get("total", 0)
            pi_area = s.get("peak_info", {}).get("area", 0)
            bl = s.get("peak_info", {}).get("baseline_level", 0)
            snr = s.get("snr_info", {}).get("snr_direct", 0)
            vol = s.get("injection_volume", VOL_COL if method == "COLUMN" else VOL_BP)

            # Extreure concentració del nom
            conc = 0
            m = re.search(r"(\d+\.?\d*)\s*(?:ppm|PPM)?", name)
            if m:
                conc = float(m.group(1))
            if conc <= 0:
                continue

            entry = {
                "seq": d, "name": name, "method": method,
                "area_total": area, "pi_area": pi_area,
                "baseline": bl, "snr": snr,
                "vol": vol, "conc": conc,
                "seq_num": seq_num(d),
            }

            if method == "COLUMN":
                # Extreure fraccions
                fracs = s.get("areas", {}).get("DOC", {})
                for fn in ["BioP", "HS", "BB", "SB", "LMW"]:
                    entry[f"frac_{fn}"] = fracs.get(fn, 0)
                khp_col.append(entry)
            else:
                khp_bp.append(entry)

    print(f"\n  KHP COLUMN: {len(khp_col)} entrades")
    print(f"  KHP BP:     {len(khp_bp)} entrades")

    if khp_col:
        # Area/µg per COLUMN
        area_per_ug_col = [(k["area_total"] / (k["conc"] * k["vol"] / 1000)) for k in khp_col if k["conc"] > 0]
        pi_per_ug_col = [(k["pi_area"] / (k["conc"] * k["vol"] / 1000)) for k in khp_col if k["conc"] > 0]
        print(f"\n  COLUMN RF (area_total / µg_DOC):")
        print(f"    Mediana: {np.median(area_per_ug_col):.1f}")
        print(f"    Mitjana: {np.mean(area_per_ug_col):.1f} ± {np.std(area_per_ug_col):.1f}")
        print(f"  COLUMN RF (peak_info / µg_DOC):")
        print(f"    Mediana: {np.median(pi_per_ug_col):.1f}")
        print(f"    Mitjana: {np.mean(pi_per_ug_col):.1f} ± {np.std(pi_per_ug_col):.1f}")
        print(f"  → Diferència area_total vs peak_info: {(np.median(area_per_ug_col) / np.median(pi_per_ug_col) - 1) * 100:+.1f}%")

    if khp_bp:
        area_per_ug_bp = [(k["area_total"] / (k["conc"] * k["vol"] / 1000)) for k in khp_bp if k["conc"] > 0]
        pi_per_ug_bp = [(k["pi_area"] / (k["conc"] * k["vol"] / 1000)) for k in khp_bp if k["conc"] > 0]
        print(f"\n  BP RF (area_total / µg_DOC):")
        print(f"    Mediana: {np.median(area_per_ug_bp):.1f}")
        print(f"    Mitjana: {np.mean(area_per_ug_bp):.1f} ± {np.std(area_per_ug_bp):.1f}")
        print(f"  BP RF (peak_info / µg_DOC):")
        print(f"    Mediana: {np.median(pi_per_ug_bp):.1f}")
        print(f"    Mitjana: {np.mean(pi_per_ug_bp):.1f} ± {np.std(pi_per_ug_bp):.1f}")
        print(f"  → Diferència area_total vs peak_info: {(np.median(area_per_ug_bp) / np.median(pi_per_ug_bp) - 1) * 100:+.1f}%")

    if khp_col and khp_bp:
        print(f"\n  COMPARACIÓ:")
        print(f"    RF COLUMN (total): {np.median(area_per_ug_col):.1f}")
        print(f"    RF BP (total):     {np.median(area_per_ug_bp):.1f}")
        print(f"    Ratio BP/COL:      {np.median(area_per_ug_bp) / np.median(area_per_ug_col):.3f}")
        print(f"    RF COLUMN (peak):  {np.median(pi_per_ug_col):.1f}")
        print(f"    RF BP (peak):      {np.median(pi_per_ug_bp):.1f}")
        print(f"    Ratio BP/COL peak: {np.median(pi_per_ug_bp) / np.median(pi_per_ug_col):.3f}")

    # ====================================================================
    # PART 2: Fraccions KHP COLUMN — quant "soroll" hi ha fora del pic?
    # ====================================================================
    print("\n" + "=" * 70)
    print("PART 2: KHP COLUMN — FRACCIONS (on va l'àrea fora del pic KHP?)")
    print("=" * 70)

    if khp_col:
        for fn in ["BioP", "HS", "BB", "SB", "LMW"]:
            vals = [k[f"frac_{fn}"] for k in khp_col]
            pcts = [k[f"frac_{fn}"] / k["area_total"] * 100 if k["area_total"] > 0 else 0
                    for k in khp_col]
            print(f"  {fn:5s}: àrea = {np.median(vals):7.1f}  ({np.median(pcts):5.1f}% del total)")

        overhead = [(k["area_total"] - k["pi_area"]) / k["area_total"] * 100
                    if k["area_total"] > 0 else 0 for k in khp_col]
        print(f"\n  Overhead (area fora peak_info): {np.median(overhead):.1f}% del total")
        print(f"  → Això és 'soroll d'integració': àrea integrada en finestres on no hi ha pic")

    # ====================================================================
    # PART 3: Estimació soroll per fracció en mostres reals
    # ====================================================================
    print("\n" + "=" * 70)
    print("PART 3: SOROLL PER FRACCIÓ — MOSTRES REALS")
    print("=" * 70)

    # Carregar mostres reals amb fraccions
    noise_by_frac = {fn: [] for fn in ["BioP", "HS", "BB", "SB", "LMW"]}
    all_baselines = []

    for d in sorted(os.listdir(DATA_DIR)):
        f = os.path.join(DATA_DIR, d, "CHECK", "data", "analysis_result.json")
        if not os.path.isfile(f) or "_CAL" in d.upper():
            continue
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
        if data.get("method") != "COLUMN":
            continue

        for s in data.get("samples", []):
            name = s.get("name", "")
            if name.upper().startswith(("MQ", "NAOH", "BUFFER", "KHP")):
                continue
            fracs = s.get("areas", {}).get("DOC", {})
            total = fracs.get("total", 0)
            if total <= 0:
                continue

            bl = s.get("peak_info", {}).get("baseline_level", 0)
            all_baselines.append(bl)

            for fn in ["BioP", "HS", "BB", "SB", "LMW"]:
                area = fracs.get(fn, 0)
                noise_by_frac[fn].append(area)

    print(f"  Mostres COLUMN analitzades: {len(all_baselines)}")
    print(f"  Baseline mediana: {np.median(all_baselines):.2f} ppb")
    print(f"\n  Àrea per fracció (mostres reals):")

    total_window = 0
    for fn, (t0, t1) in FRACTION_WINDOWS.items():
        w = t1 - t0
        total_window += w
        vals = noise_by_frac[fn]
        if vals:
            ppm_equiv = np.median(vals) * 1000 / (RF_COL * VOL_COL)
            print(f"    {fn:5s}: finestra {w:4.1f} min  àrea mediana = {np.median(vals):7.1f}  "
                  f"(≈{ppm_equiv:.2f} ppm equiv)")

    print(f"\n  Finestra total fraccions: {total_window:.1f} min")
    print(f"  Finestra BP típica:      ~3-5 min")
    print(f"  Factor amplificació soroll COLUMN/BP: ~{total_window / 4:.1f}x")

    # ====================================================================
    # PART 4: Simulació — si el soroll acumula, quant afecta?
    # ====================================================================
    print("\n" + "=" * 70)
    print("PART 4: SIMULACIÓ — CONTRIBUCIÓ SOROLL BASELINE A ppm")
    print("=" * 70)

    # Baseline noise en ppb → àrea noise per fracció
    bl_std = np.std(all_baselines) if all_baselines else 1.0
    bl_median = np.median(all_baselines) if all_baselines else 0.0
    dt = 0.0667  # min per punt (cadència TOC)

    print(f"  Baseline: mediana={bl_median:.2f}, std={bl_std:.2f} ppb")
    print(f"  Cadència DOC: {dt:.4f} min/punt ({1/dt:.1f} pts/min)")
    print(f"\n  Contribució baseline per fracció (si baseline = constant positiu):")

    total_noise_area = 0
    for fn, (t0, t1) in FRACTION_WINDOWS.items():
        w = t1 - t0
        n_pts = int(w / dt)
        # Àrea baseline = bl_median * n_pts (la baseline ja es resta, però si no fos perfecta...)
        # Soroll àrea = bl_std * sqrt(n_pts) * dt  (random walk)
        noise_area = bl_std * np.sqrt(n_pts)
        total_noise_area += noise_area
        ppm_noise = noise_area * 1000 / (RF_COL * VOL_COL)
        print(f"    {fn:5s}: {w:4.1f} min, {n_pts:4d} pts, noise àrea ≈ {noise_area:.1f}  (≈{ppm_noise:.3f} ppm)")

    ppm_total_noise = total_noise_area * 1000 / (RF_COL * VOL_COL)
    print(f"\n  Soroll TOTAL acumulat 5 fraccions: àrea ≈ {total_noise_area:.1f}  (≈{ppm_total_noise:.3f} ppm)")

    # Comparar amb BP
    bp_window = 4.0  # min
    bp_pts = int(bp_window / dt)
    bp_noise = bl_std * np.sqrt(bp_pts)
    bp_ppm_noise = bp_noise * 1000 / (RF_BP * VOL_BP)
    print(f"  Soroll BP (finestra {bp_window} min): àrea ≈ {bp_noise:.1f}  (≈{bp_ppm_noise:.3f} ppm)")
    print(f"\n  Ratio soroll COLUMN/BP: {total_noise_area / bp_noise:.1f}x en àrea")
    print(f"  Ratio soroll COLUMN/BP: {ppm_total_noise / bp_ppm_noise:.1f}x en ppm")

    # ====================================================================
    # PART 5: Comprovació — per conc reals, quant és el soroll vs senyal?
    # ====================================================================
    print("\n" + "=" * 70)
    print("PART 5: SOROLL vs SENYAL — PER RANG DE CONCENTRACIÓ")
    print("=" * 70)

    conc_ranges = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 10), (10, 999)]
    print(f"  Soroll COLUMN estimat: {ppm_total_noise:.3f} ppm")
    print(f"  Soroll BP estimat:     {bp_ppm_noise:.3f} ppm")
    print(f"\n  {'Rang':>10s}  {'ppm_COL':>8s}  {'soroll/senyal':>14s}  {'Δ esperat':>10s}")
    for lo, hi in conc_ranges:
        mid = (lo + hi) / 2 if hi < 999 else 15
        snr_ratio = ppm_total_noise / mid * 100  # % soroll respecte senyal
        delta_expected = (ppm_total_noise - bp_ppm_noise) / mid * 100  # diferència % per soroll
        print(f"    {lo:>2}-{hi:<3} ppm  {mid:8.1f}  {snr_ratio:13.1f}%  {delta_expected:+9.1f}%")

    # ====================================================================
    # PLOT
    # ====================================================================
    os.makedirs(OUT_DIR, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. RF KHP: area_total vs peak_info per mode
    ax = axes[0, 0]
    if khp_col:
        concs_col = [k["conc"] for k in khp_col]
        rf_total_col = [k["area_total"] / (k["conc"] * k["vol"] / 1000) for k in khp_col]
        rf_peak_col = [k["pi_area"] / (k["conc"] * k["vol"] / 1000) for k in khp_col]
        ax.scatter(concs_col, rf_total_col, s=20, alpha=0.5, c='steelblue', label='COL total')
        ax.scatter(concs_col, rf_peak_col, s=20, alpha=0.5, c='lightblue', marker='x', label='COL peak')
    if khp_bp:
        concs_bp = [k["conc"] for k in khp_bp]
        rf_total_bp = [k["area_total"] / (k["conc"] * k["vol"] / 1000) for k in khp_bp]
        rf_peak_bp = [k["pi_area"] / (k["conc"] * k["vol"] / 1000) for k in khp_bp]
        ax.scatter(concs_bp, rf_total_bp, s=20, alpha=0.5, c='coral', label='BP total')
        ax.scatter(concs_bp, rf_peak_bp, s=20, alpha=0.5, c='lightsalmon', marker='x', label='BP peak')
    ax.set_xlabel("Concentració KHP (ppm)")
    ax.set_ylabel("RF (area / µg_DOC)")
    ax.set_title("RF per concentració i mode")
    ax.legend(fontsize=8)
    ax.set_ylim(0, max(2000, np.percentile(rf_total_col + rf_total_bp, 95) * 1.2) if (khp_col and khp_bp) else 2000)

    # 2. Overhead KHP per concentració
    ax = axes[0, 1]
    if khp_col:
        concs = [k["conc"] for k in khp_col]
        oh = [(k["area_total"] - k["pi_area"]) for k in khp_col]
        oh_pct = [(k["area_total"] - k["pi_area"]) / k["area_total"] * 100
                  if k["area_total"] > 0 else 0 for k in khp_col]
        ax.scatter(concs, oh_pct, s=20, alpha=0.5, c='steelblue')
        ax.set_xlabel("Concentració KHP (ppm)")
        ax.set_ylabel("Overhead (%) — àrea fora pic principal")
        ax.set_title("KHP COLUMN: overhead vs concentració")
        ax.axhline(0, color='k', ls='--', lw=0.5)

    # 3. Àrea fraccions KHP COLUMN (stacked)
    ax = axes[1, 0]
    if khp_col:
        sorted_khp = sorted(khp_col, key=lambda k: k["conc"])
        labels = [f"{k['name']}\n{k['conc']}ppm" for k in sorted_khp[:20]]
        frac_names = ["BioP", "HS", "BB", "SB", "LMW"]
        colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#795548']
        bottoms = np.zeros(min(20, len(sorted_khp)))
        for fn, color in zip(frac_names, colors):
            vals = [k[f"frac_{fn}"] for k in sorted_khp[:20]]
            ax.bar(range(len(vals)), vals, bottom=bottoms, color=color, label=fn, alpha=0.8)
            bottoms += np.array(vals)
        ax.set_xlabel("KHP (ordenat per conc)")
        ax.set_ylabel("Àrea")
        ax.set_title("KHP COLUMN: fraccions (primers 20)")
        ax.legend(fontsize=7, ncol=5)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=6)

    # 4. Soroll estimat vs concentració
    ax = axes[1, 1]
    conc_range = np.linspace(0.5, 10, 50)
    noise_col_pct = ppm_total_noise / conc_range * 100
    noise_bp_pct = bp_ppm_noise / conc_range * 100
    ax.plot(conc_range, noise_col_pct, 'b-', lw=2, label=f'COLUMN ({total_window:.0f} min)')
    ax.plot(conc_range, noise_bp_pct, 'r-', lw=2, label=f'BP ({bp_window:.0f} min)')
    ax.fill_between(conc_range, noise_bp_pct, noise_col_pct, alpha=0.15, color='purple',
                    label=f'Diferència (Δ≈{ppm_total_noise - bp_ppm_noise:.3f} ppm)')
    ax.set_xlabel("Concentració (ppm)")
    ax.set_ylabel("Soroll / Senyal (%)")
    ax.set_title("Contribució soroll vs concentració")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 50)

    fig.suptitle("KHP: COLUMN vs BP — soroll i fraccions", fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "khp_col_bp.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nGràfic: {path}")


if __name__ == "__main__":
    main()
