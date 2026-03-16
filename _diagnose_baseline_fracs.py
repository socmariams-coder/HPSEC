"""
Diagnòstic: mirar cromatogrames DOC reals per veure l'efecte de la baseline
sobre la integració de fraccions.

Per cada mostra amb ratio BP/COL anòmal:
- Pintar el cromatograma DOC amb les finestres de fracció
- Mostrar la baseline global vs una baseline local per fracció
- Calcular àrees amb baseline global vs local
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

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DATA_DIR = r"C:\Users\Lequia\Desktop\Dades3"
OUT_DIR = os.path.join(DATA_DIR, "REGISTRY", "review", "baseline_diag")

from hpsec_import import import_from_manifest, ensure_data_loaded
from hpsec_config import get_config as load_config
from hpsec_core import get_baseline_value
from scipy.integrate import trapezoid

FRAC_WINDOWS = {
    "BioP": (10.8, 18.0),
    "HS":   (18.0, 23.0),
    "BB":   (23.0, 26.0),
    "SB":   (26.0, 32.0),
    "LMW":  (32.0, 70.0),
}

RF_COL = 752.90; VOL_COL = 400.0; IC_COL = 18.11


def integrate_global_baseline(t, y_raw, baseline):
    """Mètode actual: baseline global, clip a 0."""
    y_net = np.maximum(y_raw - baseline, 0)
    result = {}
    total = 0
    for fn, (t0, t1) in FRAC_WINDOWS.items():
        mask = (t >= t0) & (t < t1)
        if mask.sum() > 1:
            area = float(trapezoid(y_net[mask], t[mask]))
        else:
            area = 0.0
        result[fn] = area
        total += area
    result["total"] = total
    return result, y_net


def integrate_local_baseline(t, y_raw):
    """Mètode nou: baseline local per fracció (valls entre pics)."""
    # Definir punts d'ancoratge: les transicions entre fraccions
    # En aquests punts el senyal hauria de ser ~baseline
    boundaries = [10.8, 18.0, 23.0, 26.0, 32.0, 70.0]

    # Per cada boundary, agafar la mediana del senyal ±0.5 min
    anchor_t = []
    anchor_y = []
    for b in boundaries:
        mask = (t >= b - 0.3) & (t <= b + 0.3)
        if mask.sum() > 0:
            anchor_t.append(b)
            anchor_y.append(np.median(y_raw[mask]))

    # Interpolar baseline entre ancoratges
    if len(anchor_t) >= 2:
        baseline_interp = np.interp(t, anchor_t, anchor_y)
    else:
        baseline_interp = np.full_like(t, np.median(y_raw[:50]))

    y_net = np.maximum(y_raw - baseline_interp, 0)

    result = {}
    total = 0
    for fn, (t0, t1) in FRAC_WINDOWS.items():
        mask = (t >= t0) & (t < t1)
        if mask.sum() > 1:
            area = float(trapezoid(y_net[mask], t[mask]))
        else:
            area = 0.0
        result[fn] = area
        total += area
    result["total"] = total
    return result, y_net, baseline_interp


def integrate_valley_baseline(t, y_raw):
    """Mètode 3: baseline com a mínim local (rolling min amb finestra gran)."""
    from scipy.ndimage import minimum_filter1d
    # Rolling minimum amb finestra de ~5 min (75 punts a dt=0.067)
    dt = np.median(np.diff(t)) if len(t) > 1 else 0.067
    window = max(5, int(5.0 / dt))
    y_min = minimum_filter1d(y_raw, size=window)

    y_net = np.maximum(y_raw - y_min, 0)

    result = {}
    total = 0
    for fn, (t0, t1) in FRAC_WINDOWS.items():
        mask = (t >= t0) & (t < t1)
        if mask.sum() > 1:
            area = float(trapezoid(y_net[mask], t[mask]))
        else:
            area = 0.0
        result[fn] = area
        total += area
    result["total"] = total
    return result, y_net, y_min


def main():
    config = load_config()
    os.makedirs(OUT_DIR, exist_ok=True)

    # Trobar mostres amb ratio anòmal (seleccionar unes quantes representatives)
    # Primer carregar analysis_results per trobar parells
    target_seqs = [
        # Baixa concentració, ratio alt
        ("150_SEQ", "BLNK025"),   # ratio 10.57
        ("143_SEQ", "AA"),        # ratio 7.27
        # Concentració mitjana, ratio normal
        ("285_SEQ", "LQ0149"),    # ratio ~1.1
        ("285_SEQ", "LQ0150"),    # ratio ~1.0
        # Alta concentració
        ("282_SEQ", "FR2591"),    # ratio ~0.32
    ]

    results_summary = []

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(target_seqs), 3, figsize=(24, 5 * len(target_seqs)))
    if len(target_seqs) == 1:
        axes = axes.reshape(1, -1)

    for idx, (seq_name, sample_name) in enumerate(target_seqs):
        seq_path = os.path.join(DATA_DIR, seq_name)
        manifest_path = os.path.join(seq_path, "CHECK", "data", "import_manifest.json")

        if not os.path.isfile(manifest_path):
            print(f"  SKIP {seq_name}/{sample_name}: no manifest")
            continue

        # Importar seqüència
        try:
            with open(manifest_path, encoding="utf-8") as fh:
                manifest = json.load(fh)
            imported = import_from_manifest(seq_path, manifest=manifest, config=config, load_data=False)
            if imported.get("data_deferred"):
                ensure_data_loaded(imported, config=config)
        except Exception as e:
            print(f"  SKIP {seq_name}/{sample_name}: {e}")
            continue

        # Trobar mostra
        samples = imported.get("samples", {})
        if sample_name not in samples:
            print(f"  SKIP {seq_name}/{sample_name}: mostra no trobada")
            continue

        s = samples[sample_name]
        reps = s.get("replicas", {})

        # Agafar primera rèplica amb dades DOC direct
        t_doc, y_doc = None, None
        for rep_key, rep_data in reps.items():
            direct = rep_data.get("direct")
            if not direct:
                continue
            t_raw = direct.get("t")
            y_raw = direct.get("y")
            if t_raw is not None and y_raw is not None:
                t_doc = np.array(t_raw)
                y_doc = np.array(y_raw)
                break

        if t_doc is None or len(t_doc) < 50:
            print(f"  SKIP {seq_name}/{sample_name}: no DOC data")
            continue

        # Baseline global (mètode actual)
        baseline_global = get_baseline_value(t_doc, y_doc, mode="COLUMN", config=config)

        # 3 mètodes d'integració
        res_global, y_net_global = integrate_global_baseline(t_doc, y_doc, baseline_global)
        res_local, y_net_local, bl_interp = integrate_local_baseline(t_doc, y_doc)
        res_valley, y_net_valley, y_min = integrate_valley_baseline(t_doc, y_doc)

        # Llegir àrees de l'analysis_result per comparar
        ar_path = os.path.join(seq_path, "CHECK", "data", "analysis_result.json")
        ppm_col = 0
        if os.path.isfile(ar_path):
            with open(ar_path, encoding="utf-8") as fh:
                ar = json.load(fh)
            for s_ar in ar.get("samples", []):
                if s_ar["name"] == sample_name:
                    ppm_col = s_ar.get("areas", {}).get("DOC", {}).get("total", 0)
                    ppm_col = (ppm_col - IC_COL) * 1000 / (RF_COL * VOL_COL)
                    break

        ppm_g = (res_global["total"] - IC_COL) * 1000 / (RF_COL * VOL_COL)
        ppm_l = (res_local["total"] - IC_COL) * 1000 / (RF_COL * VOL_COL)
        ppm_v = (res_valley["total"] - IC_COL) * 1000 / (RF_COL * VOL_COL)

        print(f"\n{'='*60}")
        print(f"  {seq_name} / {sample_name}  (ppm analysis_result = {ppm_col:.2f})")
        print(f"{'='*60}")
        print(f"  Baseline global: {baseline_global:.2f}")
        print(f"  {'Fracció':>6s}  {'Global':>8s}  {'Local':>8s}  {'Valley':>8s}  {'Δ loc%':>8s}  {'Δ val%':>8s}")
        for fn in ["BioP", "HS", "BB", "SB", "LMW", "total"]:
            g = res_global[fn]
            l = res_local[fn]
            v = res_valley[fn]
            dl = (l - g) / g * 100 if g > 0 else 0
            dv = (v - g) / g * 100 if g > 0 else 0
            print(f"  {fn:>6s}  {g:8.1f}  {l:8.1f}  {v:8.1f}  {dl:+7.1f}%  {dv:+7.1f}%")
        print(f"  ppm:    {ppm_g:8.2f}  {ppm_l:8.2f}  {ppm_v:8.2f}")

        results_summary.append({
            "seq": seq_name, "sample": sample_name,
            "ppm_ar": ppm_col,
            "ppm_global": ppm_g, "ppm_local": ppm_l, "ppm_valley": ppm_v,
            "total_global": res_global["total"],
            "total_local": res_local["total"],
            "total_valley": res_valley["total"],
        })

        # PLOTS
        frac_colors = {
            "BioP": '#2196F3', "HS": '#4CAF50', "BB": '#FF9800',
            "SB": '#9C27B0', "LMW": '#795548'
        }

        # Plot 1: Cromatograma cru + baselines
        ax = axes[idx, 0]
        ax.plot(t_doc, y_doc, 'k-', lw=0.5, alpha=0.7, label='Senyal cru')
        ax.axhline(baseline_global, color='red', ls='--', lw=1.5, label=f'BL global ({baseline_global:.1f})')
        ax.plot(t_doc, bl_interp, 'blue', ls='-', lw=1.5, label='BL local (interp)')
        ax.plot(t_doc, y_min, 'green', ls='-', lw=1, alpha=0.7, label='BL valley (rolling min)')
        for fn, (t0, t1) in FRAC_WINDOWS.items():
            ax.axvspan(t0, t1, alpha=0.1, color=frac_colors[fn])
            ax.text((t0 + t1) / 2, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 10,
                    fn, ha='center', fontsize=7, color=frac_colors[fn])
        ax.set_xlim(8, 50)
        ax.set_xlabel("Temps (min)")
        ax.set_ylabel("Senyal DOC")
        ax.set_title(f"{seq_name}/{sample_name} — Cromatograma + baselines")
        ax.legend(fontsize=7, loc='upper right')

        # Plot 2: Senyal net (global vs local)
        ax = axes[idx, 1]
        ax.plot(t_doc, y_net_global, 'r-', lw=0.8, alpha=0.7, label='Net global')
        ax.plot(t_doc, y_net_local, 'b-', lw=0.8, alpha=0.7, label='Net local')
        ax.axhline(0, color='k', ls=':', lw=0.5)
        for fn, (t0, t1) in FRAC_WINDOWS.items():
            ax.axvspan(t0, t1, alpha=0.1, color=frac_colors[fn])
        ax.set_xlim(8, 50)
        ax.set_xlabel("Temps (min)")
        ax.set_ylabel("Senyal net")
        ax.set_title(f"Senyal net: global (vermell) vs local (blau)")
        ax.legend(fontsize=7)

        # Plot 3: Barres fraccions comparant mètodes
        ax = axes[idx, 2]
        x = np.arange(5)
        w = 0.25
        fnames = ["BioP", "HS", "BB", "SB", "LMW"]
        g_vals = [res_global[fn] for fn in fnames]
        l_vals = [res_local[fn] for fn in fnames]
        v_vals = [res_valley[fn] for fn in fnames]
        ax.bar(x - w, g_vals, w, label='Global', color='red', alpha=0.7)
        ax.bar(x, l_vals, w, label='Local', color='blue', alpha=0.7)
        ax.bar(x + w, v_vals, w, label='Valley', color='green', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(fnames)
        ax.set_ylabel("Àrea")
        ax.set_title(f"Fraccions: G={res_global['total']:.0f} L={res_local['total']:.0f} V={res_valley['total']:.0f}")
        ax.legend(fontsize=7)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "baseline_diagnosis.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nGràfic: {path}")

    # Resum
    print(f"\n{'='*70}")
    print(f"RESUM: IMPACTE BASELINE SOBRE ÀREA TOTAL")
    print(f"{'='*70}")
    print(f"  {'Mostra':>25s}  {'ppm AR':>7s}  {'Global':>7s}  {'Local':>7s}  {'Valley':>7s}  {'ΔL%':>6s}  {'ΔV%':>6s}")
    for r in results_summary:
        dl = (r["ppm_local"] - r["ppm_global"]) / r["ppm_global"] * 100 if r["ppm_global"] > 0 else 0
        dv = (r["ppm_valley"] - r["ppm_global"]) / r["ppm_global"] * 100 if r["ppm_global"] > 0 else 0
        print(f"  {r['seq']+'/'+r['sample']:>25s}  {r['ppm_ar']:7.2f}  "
              f"{r['ppm_global']:7.2f}  {r['ppm_local']:7.2f}  {r['ppm_valley']:7.2f}  "
              f"{dl:+5.1f}%  {dv:+5.1f}%")


if __name__ == "__main__":
    main()
