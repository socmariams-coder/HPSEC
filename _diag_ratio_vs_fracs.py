"""Correlation: ratio ppm BP/COL vs COLUMN fraction composition."""
import json, os
import numpy as np
from collections import defaultdict
from pathlib import Path
from hpsec_config import get_data_folders
from hpsec_consolidate import detect_seq_type

FRACS = ['BioP', 'HS', 'BB', 'SB']

def main():
    folders = get_data_folders()
    inventory = defaultdict(dict)

    for folder in folders:
        for d in sorted(os.listdir(folder)):
            full = os.path.join(folder, d)
            jp = os.path.join(full, 'CHECK', 'data', 'analysis_result.json')
            mp = os.path.join(full, 'CHECK', 'data', 'import_manifest.json')
            if not os.path.exists(jp):
                continue
            with open(jp, encoding='utf-8') as f:
                data = json.load(f)
            vol = None
            if os.path.exists(mp):
                with open(mp, encoding='utf-8') as f:
                    m = json.load(f)
                samps = m.get('samples', [])
                if isinstance(samps, list):
                    for s in samps:
                        for r in (s.get('replicas') or []):
                            v = (r.get('injection') or {}).get('inj_volume')
                            if v:
                                vol = v
                                break
                        if vol:
                            break
            method = detect_seq_type(d)
            sg = data.get('samples_grouped', {})
            for name, sdata in sg.items():
                if sdata.get('sample_type') != 'SAMPLE':
                    continue
                sel = (sdata.get('selected') or {}).get('doc', '1')
                reps = sdata.get('replicas') or {}
                doc_rep = reps.get(sel, {})
                if sel == 'comp':
                    for rv in reps.values():
                        if isinstance(rv, dict) and rv.get('timeout_composition'):
                            doc_rep = rv
                            break
                if not isinstance(doc_rep, dict):
                    doc_rep = {}
                areas = (doc_rep.get('areas') or {}).get('DOC') or {}
                quant = sdata.get('quantification') or {}
                key = 'bp' if method == 'BP' else 'col'
                inventory[name][f'{key}_seq'] = d
                inventory[name][f'{key}_ppm'] = (
                    quant.get('concentration_ppm_direct')
                    or quant.get('concentration_ppm'))
                inventory[name][f'{key}_area_total'] = areas.get('total', 0) or 0
                inventory[name][f'{key}_vol'] = vol
                for fr in FRACS + ['LMW']:
                    inventory[name][f'{key}_DOC_{fr}'] = areas.get(fr, 0) or 0

    # Filter: both modes, valid ppm, exclude 272/271 anomaly
    both = {}
    for n, v in inventory.items():
        if v.get('col_area_total', 0) <= 0 or v.get('bp_area_total', 0) <= 0:
            continue
        if not v.get('col_ppm') or not v.get('bp_ppm') or v['col_ppm'] <= 0:
            continue
        if '271' in v.get('bp_seq', '') or '272' in v.get('col_seq', ''):
            continue
        both[n] = v

    print(f"Mostres: {len(both)} (exclosos 272/271)")

    # Calculate percentages
    rows = []
    for n, v in both.items():
        total_col = v['col_area_total']
        pct = {}
        for fr in FRACS + ['LMW']:
            pct[fr] = v.get(f'col_DOC_{fr}', 0) / total_col * 100 if total_col > 0 else 0
        pct['HS_BB'] = pct['HS'] + pct['BB']
        pct['LMW_SB'] = pct['LMW'] + pct['SB']
        ratio = v['bp_ppm'] / v['col_ppm']
        rows.append({
            'name': n, 'ratio': ratio,
            'col_seq': v.get('col_seq', ''), 'bp_seq': v.get('bp_seq', ''),
            **{f'pct_{k}': v2 for k, v2 in pct.items()},
        })

    # === CORRELATIONS ===
    print("\n=== CORRELACIO ratio_ppm vs % fraccio COLUMN ===")
    for label, key in [('HS', 'pct_HS'), ('BB', 'pct_BB'), ('HS+BB', 'pct_HS_BB'),
                        ('SB', 'pct_SB'), ('LMW', 'pct_LMW'), ('LMW+SB', 'pct_LMW_SB')]:
        x = np.array([r[key] for r in rows])
        y = np.array([r['ratio'] for r in rows])
        r_val = np.corrcoef(x, y)[0, 1]
        print(f"  {label:6s} vs ratio: R={r_val:+.4f}  R2={r_val**2:.4f}")

    # === TERCILS LMW ===
    print("\n=== RATIO PPM PER TERCILS DE %LMW ===")
    lmw_vals = sorted([r['pct_LMW'] for r in rows])
    t1 = lmw_vals[len(lmw_vals) // 3]
    t2 = lmw_vals[2 * len(lmw_vals) // 3]
    for label, lo, hi in [('Baix', 0, t1), ('Mig', t1, t2), ('Alt', t2, 100)]:
        subset = [r for r in rows if lo <= r['pct_LMW'] < hi]
        if not subset:
            continue
        ratios = np.array([r['ratio'] for r in subset])
        lmw_m = np.mean([r['pct_LMW'] for r in subset])
        hs_m = np.mean([r['pct_HS_BB'] for r in subset])
        n_out = sum(1 for r in ratios if r < 0.85 or r > 1.15)
        print(f"  {label:4s} LMW ({lmw_m:4.0f}%, HS+BB={hs_m:.0f}%): "
              f"n={len(subset):2d}  ratio={np.median(ratios):.3f} +/- {np.std(ratios):.3f}  "
              f"outliers={n_out}/{len(subset)}")

    # === TERCILS SB ===
    print("\n=== RATIO PPM PER TERCILS DE %SB ===")
    sb_vals = sorted([r['pct_SB'] for r in rows])
    t1s = sb_vals[len(sb_vals) // 3]
    t2s = sb_vals[2 * len(sb_vals) // 3]
    for label, lo, hi in [('Baix', 0, t1s), ('Mig', t1s, t2s), ('Alt', t2s, 100)]:
        subset = [r for r in rows if lo <= r['pct_SB'] < hi]
        if not subset:
            continue
        ratios = np.array([r['ratio'] for r in subset])
        sb_m = np.mean([r['pct_SB'] for r in subset])
        n_out = sum(1 for r in ratios if r < 0.85 or r > 1.15)
        print(f"  {label:4s} SB ({sb_m:4.0f}%): n={len(subset):2d}  "
              f"ratio={np.median(ratios):.3f} +/- {np.std(ratios):.3f}  "
              f"outliers={n_out}/{len(subset)}")

    # === TERCILS HS+BB ===
    print("\n=== RATIO PPM PER TERCILS DE %HS+BB ===")
    hsbb_vals = sorted([r['pct_HS_BB'] for r in rows])
    t1h = hsbb_vals[len(hsbb_vals) // 3]
    t2h = hsbb_vals[2 * len(hsbb_vals) // 3]
    for label, lo, hi in [('Baix', 0, t1h), ('Mig', t1h, t2h), ('Alt', t2h, 100)]:
        subset = [r for r in rows if lo <= r['pct_HS_BB'] < hi]
        if not subset:
            continue
        ratios = np.array([r['ratio'] for r in subset])
        hsbb_m = np.mean([r['pct_HS_BB'] for r in subset])
        lmw_m = np.mean([r['pct_LMW'] for r in subset])
        n_out = sum(1 for r in ratios if r < 0.85 or r > 1.15)
        print(f"  {label:4s} HS+BB ({hsbb_m:4.0f}%, LMW={lmw_m:.0f}%): "
              f"n={len(subset):2d}  ratio={np.median(ratios):.3f} +/- {np.std(ratios):.3f}  "
              f"outliers={n_out}/{len(subset)}")

    # === SCATTER PER PARELL: pct_HS+BB vs ratio ===
    print("\n=== PER PARELL SEQ: HS+BB% vs ratio ===")
    pairs = defaultdict(list)
    for r in rows:
        pairs[(r['col_seq'], r['bp_seq'])].append(r)

    for (cs, bs), pr in sorted(pairs.items()):
        hsbb = np.array([r['pct_HS_BB'] for r in pr])
        ratios = np.array([r['ratio'] for r in pr])
        lmw = np.array([r['pct_LMW'] for r in pr])
        r_val = np.corrcoef(hsbb, ratios)[0, 1] if len(hsbb) >= 3 else 0
        print(f"  {cs:12s}/{bs:12s}: n={len(pr):2d}  "
              f"HS+BB={np.mean(hsbb):.0f}%  LMW={np.mean(lmw):.0f}%  "
              f"ratio={np.median(ratios):.3f}  R(HS+BB,ratio)={r_val:+.3f}")

    # === PLOT ===
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        'Ratio ppm BP/COL vs composicio fraccions COLUMN\n'
        '(exclosos 272/271)', fontsize=13, fontweight='bold')

    plot_specs = [
        ('pct_HS', 'HS', '#E74C3C'),
        ('pct_BB', 'BB', '#F39C12'),
        ('pct_HS_BB', 'HS+BB', '#C0392B'),
        ('pct_SB', 'SB', '#2ECC71'),
        ('pct_LMW', 'LMW', '#9B59B6'),
        ('pct_LMW_SB', 'LMW+SB', '#8E44AD'),
    ]

    # Color by pair
    pair_colors = {}
    cmap = plt.cm.get_cmap('tab10', len(pairs))
    for i, (cs, bs) in enumerate(sorted(pairs.keys())):
        pair_colors[(cs, bs)] = cmap(i)

    for i, (key, label, default_color) in enumerate(plot_specs):
        ax = axes[i // 3, i % 3]
        for (cs, bs), pr in pairs.items():
            x = [r[key] for r in pr]
            y = [r['ratio'] for r in pr]
            ax.scatter(x, y, c=[pair_colors[(cs, bs)]] * len(x),
                       s=20, alpha=0.7, edgecolors='none',
                       label=f"{cs[:6]}/{bs[:6]}" if i == 0 else None)
        ax.axhline(1.0, color='k', ls='--', alpha=0.3)
        ax.axhline(0.85, color='r', ls='--', alpha=0.15)
        ax.axhline(1.15, color='r', ls='--', alpha=0.15)
        x_all = np.array([r[key] for r in rows])
        y_all = np.array([r['ratio'] for r in rows])
        r_val = np.corrcoef(x_all, y_all)[0, 1]
        if len(x_all) >= 3:
            z = np.polyfit(x_all, y_all, 1)
            xf = np.linspace(x_all.min(), x_all.max(), 50)
            ax.plot(xf, np.polyval(z, xf), 'k-', alpha=0.4, lw=1.5)
        ax.set_title(f"%{label} vs ratio (R={r_val:+.3f})", fontsize=10)
        ax.set_xlabel(f"%{label} COLUMN")
        ax.set_ylabel("Ratio ppm BP/COL")
        ax.set_ylim(0, min(3.5, y_all.max() * 1.1))
        ax.tick_params(labelsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Legend on first plot
    axes[0, 0].legend(fontsize=6, loc='upper left')

    fig.tight_layout()
    out = Path('_results/diag_col_bp')
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out / 'ratio_vs_fractions.png'), dpi=150, bbox_inches='tight')
    print(f"\nPlot: {out / 'ratio_vs_fractions.png'}")
    plt.close(fig)
    print("Fet!")


if __name__ == "__main__":
    main()
