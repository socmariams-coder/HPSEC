"""
Diagnòstic: Investigar els valors a254_area de SEQ_CAL.

Objectiu: verificar si les àrees 254nm són proporcionals a la concentració
i si el mètode d'integració (detect_main_peak + all_peaks bounds) és adequat.

Compara:
1. a254_area tal com arriba de analizar_khp_data (mètode actual)
2. Integració amb find_peak_boundaries (derivada tangent)
3. Integració amb els mateixos límits que DOC (transferits)
4. Àrea total 254nm (sense retallar)

Genera taula + scatter plot.
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hpsec_core import (
    detect_main_peak, detect_all_peaks, find_peak_boundaries,
    get_baseline_stats,
)
from hpsec_calibrate import (
    DEFAULT_CONFIG, detect_seq_cal_data, extract_khp_conc,
    calibrate_from_import,
)
from hpsec_import import (
    import_from_manifest, load_manifest, ensure_data_loaded,
)


def process_seq_cal(seq_path):
    """Processa una SEQ_CAL i extreu dades DOC + DAD per cada KHP."""
    print(f"\n{'='*70}")
    print(f"  SEQ: {os.path.basename(seq_path)}")
    print(f"{'='*70}")

    # Importar
    manifest = load_manifest(seq_path)
    if not manifest:
        print("  ERROR: No manifest")
        return None

    imported = import_from_manifest(seq_path, manifest=manifest, load_data=True)
    if not imported or not imported.get("success"):
        print(f"  ERROR importació: {imported.get('errors', [])}")
        return None

    if imported.get("data_deferred"):
        ensure_data_loaded(imported)

    method = imported.get("method", "COLUMN")
    is_bp = "BP" in method.upper()
    samples = imported.get("samples", {})
    khp_samples = imported.get("khp_samples", [])

    print(f"  Mode: {method}, KHP samples: {khp_samples}")

    results = []

    for sample_name in khp_samples:
        sample = samples.get(sample_name, {})
        replicas = sample.get("replicas", {})
        conc = extract_khp_conc(sample_name)

        for rep_id, rep_data in replicas.items():
            # DOC Direct
            sig_direct = rep_data.get("direct")
            if sig_direct is None:
                continue
            t_doc = sig_direct.get("t")
            y_doc_net = sig_direct.get("y_net")
            if t_doc is None or y_doc_net is None:
                continue
            t_doc = np.asarray(t_doc)
            y_doc_net = np.asarray(y_doc_net)

            # DAD — pot ser dict {"df": DataFrame} o directament DataFrame
            dad_raw = rep_data.get("dad")
            if dad_raw is None:
                results.append({
                    'name': sample_name, 'replica': rep_id, 'conc': conc,
                    'has_dad': False,
                })
                continue
            if isinstance(dad_raw, dict):
                df_dad = dad_raw.get("df")
            else:
                df_dad = dad_raw
            if df_dad is None or (hasattr(df_dad, 'empty') and df_dad.empty):
                results.append({
                    'name': sample_name, 'replica': rep_id, 'conc': conc,
                    'has_dad': False,
                })
                continue

            # Extreure 254nm
            col_254 = None
            for c in df_dad.columns:
                if "254" in str(c):
                    col_254 = c
                    break

            if not col_254:
                results.append({
                    'name': sample_name, 'replica': rep_id, 'conc': conc,
                    'has_dad': False, 'reason': 'no_254_col',
                })
                continue

            t_dad = pd.to_numeric(df_dad["time (min)"], errors="coerce").to_numpy()
            dad_254 = pd.to_numeric(df_dad[col_254], errors="coerce").to_numpy()
            mask = np.isfinite(t_dad) & np.isfinite(dad_254)
            t_dad = t_dad[mask]
            dad_254 = dad_254[mask]

            if len(t_dad) < 20:
                results.append({
                    'name': sample_name, 'replica': rep_id, 'conc': conc,
                    'has_dad': False, 'reason': f'too_few_points ({len(t_dad)})',
                })
                continue

            config = DEFAULT_CONFIG.copy()

            # ---- Mètode 1: Actual (detect_main_peak + all_peaks bounds) ----
            dad_peak = detect_main_peak(t_dad, dad_254, config["peak_min_prominence_pct"])

            a254_m1 = 0.0
            t_max_254 = None
            dad_l1, dad_r1 = 0, len(t_dad) - 1
            if dad_peak and dad_peak.get('valid'):
                t_max_254 = dad_peak['t_max']
                dad_pk_idx = dad_peak['peak_idx']
                dad_l1 = dad_peak['left_idx']
                dad_r1 = dad_peak['right_idx']

                # all_peaks override (com fa el codi actual)
                dad_all = detect_all_peaks(t_dad, dad_254, config["peak_min_prominence_pct"])
                for pk in dad_all:
                    if pk['idx'] == dad_pk_idx or abs(pk['t'] - t_max_254) < 0.1:
                        dad_l1 = pk['left_idx']
                        dad_r1 = pk['right_idx']
                        break

                if dad_r1 > dad_l1:
                    a254_m1 = float(np.trapezoid(dad_254[dad_l1:dad_r1+1], t_dad[dad_l1:dad_r1+1]))

            # ---- Mètode 2: find_peak_boundaries (derivada tangent) ----
            a254_m2 = 0.0
            dad_l2, dad_r2 = 0, len(t_dad) - 1
            if t_max_254 is not None:
                bl_dad = get_baseline_stats(t_dad, dad_254, mode=method)
                bl_level_dad = bl_dad.get("mean", 0)
                dad_l2, dad_r2 = find_peak_boundaries(
                    t_dad, dad_254, dad_peak['peak_idx'], bl_level_dad, is_bp=is_bp)
                if dad_r2 > dad_l2:
                    a254_m2 = float(np.trapezoid(dad_254[dad_l2:dad_r2+1], t_dad[dad_l2:dad_r2+1]))

            # ---- Mètode 3: Mateixos límits temporals que DOC ----
            a254_m3 = 0.0
            doc_peak = detect_main_peak(t_doc, y_doc_net, config["peak_min_prominence_pct"],
                                        is_bp=is_bp)
            doc_t_start = None
            doc_t_end = None
            doc_area = 0.0
            if doc_peak and doc_peak.get('valid'):
                doc_t_start = doc_peak.get('t_start', t_doc[doc_peak['left_idx']])
                doc_t_end = doc_peak.get('t_end', t_doc[doc_peak['right_idx']])
                doc_area = doc_peak.get('area', 0)

                # Transferir límits DOC al senyal 254
                dad_mask_doc = (t_dad >= doc_t_start) & (t_dad <= doc_t_end)
                if np.any(dad_mask_doc):
                    a254_m3 = float(np.trapezoid(dad_254[dad_mask_doc], t_dad[dad_mask_doc]))

            # ---- Mètode 4: Àrea total (sense limits) ----
            a254_total = float(np.trapezoid(np.maximum(dad_254, 0), t_dad))

            # ---- Mètode 5: Baseline-subtracted 254nm + find_peak_boundaries ----
            a254_m5 = 0.0
            if t_max_254 is not None:
                # Baseline 254: primers/últims punts
                bl_254_val = float(np.median(dad_254[:20])) if len(dad_254) > 20 else 0.0
                dad_254_net = dad_254 - bl_254_val
                dad_254_net = np.maximum(dad_254_net, 0)
                if dad_r2 > dad_l2:
                    a254_m5 = float(np.trapezoid(dad_254_net[dad_l2:dad_r2+1], t_dad[dad_l2:dad_r2+1]))

            # ---- Mètode 6: detect_main_peak directe (sense all_peaks override) ----
            a254_m6 = 0.0
            if dad_peak and dad_peak.get('valid'):
                dl = dad_peak['left_idx']
                dr = dad_peak['right_idx']
                if dr > dl:
                    a254_m6 = float(np.trapezoid(dad_254[dl:dr+1], t_dad[dl:dr+1]))

            results.append({
                'name': sample_name,
                'replica': rep_id,
                'conc': conc,
                'has_dad': True,
                't_max_254': t_max_254,
                't_max_doc': doc_peak.get('t_max', 0) if doc_peak and doc_peak.get('valid') else 0,
                'doc_area': doc_area,
                'doc_t_start': doc_t_start,
                'doc_t_end': doc_t_end,
                # Mètodes integració 254
                'a254_m1_actual': a254_m1,          # detect_main_peak + all_peaks override
                'a254_m2_tangent': a254_m2,         # find_peak_boundaries (derivada)
                'a254_m3_doc_limits': a254_m3,      # mateixos límits DOC
                'a254_m4_total': a254_total,        # àrea total sense retallar
                'a254_m5_bl_sub': a254_m5,          # baseline-subtracted + tangent
                'a254_m6_dmp_only': a254_m6,        # detect_main_peak sense all_peaks
                # Límits
                'm1_left': float(t_dad[dad_l1]) if dad_l1 < len(t_dad) else 0,
                'm1_right': float(t_dad[dad_r1]) if dad_r1 < len(t_dad) else 0,
                'm2_left': float(t_dad[dad_l2]) if dad_l2 < len(t_dad) else 0,
                'm2_right': float(t_dad[dad_r2]) if dad_r2 < len(t_dad) else 0,
                # DAD stats
                'dad_254_max': float(np.max(dad_254)),
                'dad_254_min': float(np.min(dad_254)),
                'dad_254_baseline': float(np.median(dad_254[:20])) if len(dad_254) > 20 else 0,
                'dad_n_points': len(t_dad),
            })

    return results, method, samples


def print_results(results, method, samples=None):
    """Mostra taula comparativa."""
    if not results:
        print("  Cap resultat")
        return

    # Filtrar només els que tenen DAD
    dad_results = [r for r in results if r.get('has_dad')]
    no_dad = [r for r in results if not r.get('has_dad')]

    if no_dad:
        print(f"\n  Sense DAD 254nm: {len(no_dad)}")
        for r in no_dad:
            print(f"    {r['name']}_R{r['replica']} conc={r['conc']} — {r.get('reason', 'no DAD')}")

    if not dad_results:
        print("  Cap resultat amb DAD")
        return

    print(f"\n  {'Name':<20} {'R':>2} {'conc':>5} {'DOC area':>8} "
          f"{'M1 actual':>9} {'M2 tang':>9} {'M3 DOClim':>9} {'M5 blsub':>9} {'M6 dmp':>9} "
          f"{'M1 lim':>12} {'M2 lim':>12} {'bl_254':>6} {'max254':>7}")
    print(f"  {'-'*20} {'-'*2} {'-'*5} {'-'*8} "
          f"{'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*9} "
          f"{'-'*12} {'-'*12} {'-'*6} {'-'*7}")

    for r in sorted(dad_results, key=lambda x: (x['conc'], x['replica'])):
        m1_lim = f"{r['m1_left']:.1f}-{r['m1_right']:.1f}"
        m2_lim = f"{r['m2_left']:.1f}-{r['m2_right']:.1f}"
        print(f"  {r['name']:<20} {r['replica']:>2} {r['conc']:>5.2f} {r['doc_area']:>8.1f} "
              f"{r['a254_m1_actual']:>9.1f} {r['a254_m2_tangent']:>9.1f} "
              f"{r['a254_m3_doc_limits']:>9.1f} {r['a254_m5_bl_sub']:>9.1f} "
              f"{r['a254_m6_dmp_only']:>9.1f} "
              f"{m1_lim:>12} {m2_lim:>12} "
              f"{r['dad_254_baseline']:>6.1f} {r['dad_254_max']:>7.1f}")

    # Regressió per cada mètode
    print(f"\n  REGRESSIÓ Àrea vs µg DOC:")
    print(f"  {'Mètode':<20} {'slope':>8} {'intercept':>10} {'R²':>8} {'n':>3}")
    print(f"  {'-'*20} {'-'*8} {'-'*10} {'-'*8} {'-'*3}")

    from scipy.stats import linregress

    vol_default = 400 if "COLUMN" in method.upper() else 100

    for mname, mkey in [
        ("DOC", "doc_area"),
        ("M1 actual", "a254_m1_actual"),
        ("M2 tangent", "a254_m2_tangent"),
        ("M3 DOC limits", "a254_m3_doc_limits"),
        ("M5 bl-subtracted", "a254_m5_bl_sub"),
        ("M6 dmp only", "a254_m6_dmp_only"),
    ]:
        x_vals = []
        y_vals = []
        for r in dad_results:
            if r['conc'] > 0 and r.get(mkey, 0) > 0:
                ug = r['conc'] * vol_default / 1000.0
                x_vals.append(ug)
                y_vals.append(r[mkey])

        if len(x_vals) >= 2:
            X = np.array(x_vals)
            Y = np.array(y_vals)
            if not np.all(X == X[0]):
                lr = linregress(X, Y)
                print(f"  {mname:<20} {lr.slope:>8.1f} {lr.intercept:>10.1f} "
                      f"{lr.rvalue**2:>8.4f} {len(x_vals):>3}")
            else:
                print(f"  {mname:<20} {'(X constant)':>28} {len(x_vals):>3}")
        else:
            print(f"  {mname:<20} {'(insuf.)':>28} {len(x_vals):>3}")

    # Plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle(f"Integració 254nm — {os.path.basename(seq_path)} ({method})", fontsize=14)

        methods = [
            ("DOC", "doc_area"),
            ("M1: actual (dmp+all_peaks)", "a254_m1_actual"),
            ("M2: tangent (fpb)", "a254_m2_tangent"),
            ("M3: DOC limits", "a254_m3_doc_limits"),
            ("M5: bl-subtracted + tangent", "a254_m5_bl_sub"),
            ("M6: dmp only (sense all_peaks)", "a254_m6_dmp_only"),
        ]

        for ax, (mname, mkey) in zip(axes.flat, methods):
            x_vals, y_vals, concs = [], [], []
            for r in dad_results:
                if r['conc'] > 0 and r.get(mkey, 0) > 0:
                    ug = r['conc'] * vol_default / 1000.0
                    x_vals.append(ug)
                    y_vals.append(r[mkey])
                    concs.append(r['conc'])

            if len(x_vals) >= 2:
                X = np.array(x_vals)
                Y = np.array(y_vals)

                # Color per concentració
                unique_concs = sorted(set(concs))
                colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_concs), 1)))
                conc_color = {c: colors[i] for i, c in enumerate(unique_concs)}

                for xi, yi, ci in zip(x_vals, y_vals, concs):
                    ax.scatter(xi, yi, c=[conc_color[ci]], s=50, edgecolors='white', linewidths=0.5)

                if not np.all(X == X[0]):
                    lr = linregress(X, Y)
                    x_line = np.linspace(0, max(X) * 1.1, 50)
                    ax.plot(x_line, lr.slope * x_line + lr.intercept, 'r-', linewidth=1.5)
                    ax.text(0.05, 0.95,
                            f"slope={lr.slope:.0f}\nint={lr.intercept:.1f}\nR²={lr.rvalue**2:.4f}",
                            transform=ax.transAxes, fontsize=8, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                ax.set_xlim(left=0)
                ax.set_ylim(bottom=min(0, min(Y) * 0.9))

            ax.set_title(mname, fontsize=9, fontweight='bold')
            ax.set_xlabel('µg DOC', fontsize=8)
            ax.set_ylabel('Àrea', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)

        fig.tight_layout()
        out_path = os.path.join(os.path.dirname(seq_path), "CHECK", "diag_a254_integration.png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=150)
        print(f"\n  Gràfic guardat: {out_path}")
        plt.close(fig)

        # Segon plot: cromatograma 254nm amb límits d'integració per cada rèplica
        n_plots = len(dad_results)
        if n_plots > 0:
            ncols = min(3, n_plots)
            nrows = (n_plots + ncols - 1) // ncols
            fig2, axes2 = plt.subplots(nrows, ncols, figsize=(6*ncols, 3.5*nrows))
            if n_plots == 1:
                axes2 = np.array([axes2])
            axes2 = axes2.flatten()

            for idx, r in enumerate(sorted(dad_results, key=lambda x: (x['conc'], x['replica']))):
                ax = axes2[idx]
                # Re-read DAD data for this replica
                sample = (samples or {}).get(r['name'], {})
                rep_data = sample.get("replicas", {}).get(r['replica'])
                if not rep_data:
                    continue
                dad_raw2 = rep_data.get("dad")
                if dad_raw2 is None:
                    continue
                df_dad = dad_raw2.get("df") if isinstance(dad_raw2, dict) else dad_raw2
                if df_dad is None:
                    continue

                col_254 = None
                for c in df_dad.columns:
                    if "254" in str(c):
                        col_254 = c
                        break
                if not col_254:
                    continue

                t_d = pd.to_numeric(df_dad["time (min)"], errors="coerce").to_numpy()
                y_d = pd.to_numeric(df_dad[col_254], errors="coerce").to_numpy()
                mk = np.isfinite(t_d) & np.isfinite(y_d)
                t_d, y_d = t_d[mk], y_d[mk]

                ax.plot(t_d, y_d, 'b-', linewidth=0.8, label='254nm')

                # Baseline
                bl = r.get('dad_254_baseline', 0)
                ax.axhline(bl, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)

                # M1 limits (actual)
                ax.axvline(r['m1_left'], color='red', linewidth=1, linestyle='--', alpha=0.7, label=f"M1: {r['m1_left']:.1f}-{r['m1_right']:.1f}")
                ax.axvline(r['m1_right'], color='red', linewidth=1, linestyle='--', alpha=0.7)

                # M2 limits (tangent)
                ax.axvline(r['m2_left'], color='green', linewidth=1, linestyle='-.', alpha=0.7, label=f"M2: {r['m2_left']:.1f}-{r['m2_right']:.1f}")
                ax.axvline(r['m2_right'], color='green', linewidth=1, linestyle='-.', alpha=0.7)

                # DOC limits
                if r.get('doc_t_start') and r.get('doc_t_end'):
                    ax.axvline(r['doc_t_start'], color='orange', linewidth=0.8, linestyle=':', alpha=0.7, label=f"DOC: {r['doc_t_start']:.1f}-{r['doc_t_end']:.1f}")
                    ax.axvline(r['doc_t_end'], color='orange', linewidth=0.8, linestyle=':', alpha=0.7)

                ax.set_title(f"{r['name']} R{r['replica']} ({r['conc']:g} ppm)\n"
                             f"M1={r['a254_m1_actual']:.0f}, M2={r['a254_m2_tangent']:.0f}, "
                             f"M3={r['a254_m3_doc_limits']:.0f}",
                             fontsize=8)
                ax.legend(fontsize=6, loc='upper right')
                ax.tick_params(labelsize=6)
                ax.set_xlabel('min', fontsize=7)

                # Zoom al pic
                if r.get('t_max_254'):
                    margin = 5.0 if "COLUMN" in method.upper() else 3.0
                    ax.set_xlim(r['t_max_254'] - margin, r['t_max_254'] + margin)

            # Amagar eixos sobrants
            for idx in range(n_plots, len(axes2)):
                axes2[idx].set_visible(False)

            fig2.suptitle(f"Cromatogrames 254nm amb límits — {os.path.basename(seq_path)}", fontsize=12)
            fig2.tight_layout()
            out_path2 = os.path.join(os.path.dirname(seq_path), "CHECK", "diag_a254_chromatograms.png")
            fig2.savefig(out_path2, dpi=150)
            print(f"  Cromatogrames guardats: {out_path2}")
            plt.close(fig2)

    except Exception as e:
        print(f"  Error generant gràfics: {e}")
        import traceback
        traceback.print_exc()


def main():
    from hpsec_config import get_config
    cfg = get_config()
    data_folder = cfg.get("paths", "data_folder", default="")

    if not data_folder:
        print("ERROR: data_folder no configurat")
        return

    # Buscar SEQ_CAL
    seq_cals = []
    for item in sorted(Path(data_folder).iterdir()):
        if item.is_dir() and "_CAL" in item.name.upper() and "_SEQ" in item.name.upper():
            seq_cals.append(str(item))

    if not seq_cals:
        print(f"No s'han trobat SEQ_CAL a {data_folder}")
        return

    print(f"Trobades {len(seq_cals)} SEQ_CAL:")
    for s in seq_cals:
        print(f"  {os.path.basename(s)}")

    # Permetre selecció per argument
    if len(sys.argv) > 1:
        target = sys.argv[1]
        seq_cals = [s for s in seq_cals if target in os.path.basename(s)]
        if not seq_cals:
            print(f"No s'ha trobat SEQ_CAL amb '{target}'")
            return

    global seq_path
    for seq_path in seq_cals:
        result = process_seq_cal(seq_path)
        if result:
            results, method, samples_data = result
            print_results(results, method, samples_data)


if __name__ == "__main__":
    main()
