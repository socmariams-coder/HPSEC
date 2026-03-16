"""
Delay auto-detect for ALL BP SEQs + cross-reference with COL/BP ratio.
======================================================================
For each BP SEQ:
1. Find peaks in full TOC
2. Match with HPLC injection times
3. Calculate optimal delay vs MasterFile delay
4. Cross-reference: SEQs with wrong delay → bad COL/BP ratio?
"""
import os, json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d

from hpsec_config import get_data_folders
from hpsec_consolidate import extract_seq_number, detect_seq_type


def auto_detect_delay(mf_path, method="BP"):
    """Auto-detect optimal delay for a MasterFile."""
    # Load TOC signal
    try:
        df_toc = pd.read_excel(mf_path, sheet_name='2-TOC', header=None)
    except Exception as e:
        return {"error": f"no 2-TOC: {e}"}

    # Find data start
    data_start = None
    for i in range(min(20, len(df_toc))):
        try:
            float(df_toc.iloc[i, 0])
            data_start = i
            break
        except (ValueError, TypeError):
            continue
    if data_start is None or data_start < 1:
        return {"error": "no data rows in 2-TOC"}

    df_toc = pd.read_excel(mf_path, sheet_name='2-TOC', header=data_start - 1)

    # Find TOC and date columns
    toc_col = None
    date_col = None
    for c in df_toc.columns:
        cs = str(c).lower()
        if toc_col is None and 'toc' in cs and 'ppb' in cs:
            toc_col = c
        if toc_col is None and 'toc' in cs:
            toc_col = c
        if date_col is None and 'started' in cs:
            date_col = c

    if toc_col is None:
        return {"error": "no TOC column"}

    toc_vals = pd.to_numeric(df_toc[toc_col], errors='coerce').values

    # Time from TOC timestamps
    t0_toc = None
    if date_col:
        toc_dates = pd.to_datetime(df_toc[date_col], errors='coerce')
        valid_dates = toc_dates.dropna()
        if len(valid_dates) > 0:
            t0_toc = valid_dates.iloc[0]

    # Use row index for time (4 sec cadence)
    dt_min = 4.0 / 60.0
    toc_t = np.arange(len(toc_vals)) * dt_min

    # Load HPLC injection times
    try:
        df_hplc = pd.read_excel(mf_path, sheet_name='1-HPLC-SEQ')
    except Exception:
        return {"error": "no 1-HPLC-SEQ"}

    hplc_date_col = None
    name_col = None
    for c in df_hplc.columns:
        if hplc_date_col is None and 'acquired' in str(c).lower():
            hplc_date_col = c
        if name_col is None and 'sample' in str(c).lower() and 'name' in str(c).lower():
            name_col = c

    if hplc_date_col is None or name_col is None or t0_toc is None:
        return {"error": "missing HPLC date/name columns or TOC t0"}

    hplc_injs = []
    for _, row in df_hplc.iterrows():
        try:
            dt = pd.to_datetime(row[hplc_date_col])
            hplc_injs.append({
                'name': str(row[name_col]),
                't_min': (dt - t0_toc).total_seconds() / 60.0,
            })
        except:
            pass

    if len(hplc_injs) < 2:
        return {"error": "too few HPLC injections"}

    hplc_times = np.array([inj['t_min'] for inj in hplc_injs])

    # Find peaks
    y = np.nan_to_num(toc_vals, nan=np.nanmedian(toc_vals[~np.isnan(toc_vals)]))
    y_smooth = uniform_filter1d(y, size=5)
    baseline = np.percentile(y_smooth, 15)

    if method == "BP":
        min_dist = int(8.0 / dt_min)
    else:
        min_dist = int(60.0 / dt_min)

    peaks, _ = find_peaks(y_smooth, height=baseline + 10, distance=min_dist, prominence=8)
    peak_times = toc_t[peaks]

    if len(peaks) < 2:
        return {"error": f"only {len(peaks)} peaks found"}

    # Read MasterFile delay
    mf_delay = None
    try:
        df_info = pd.read_excel(mf_path, sheet_name='0-INFO', header=None)
        for _, row in df_info.iterrows():
            if 'net delay' in str(row.iloc[0]).lower():
                try:
                    mf_delay = float(row.iloc[1])
                except:
                    pass
    except:
        pass

    # Search optimal delay
    best_delay = 0
    best_matches = 0

    for delay_try in np.arange(-30, 40, 0.2):
        expected = hplc_times + delay_try
        matches = 0
        total_dist = 0
        for exp_t in expected:
            diffs = np.abs(peak_times - exp_t)
            if len(diffs) > 0 and np.min(diffs) < 1.5:
                matches += 1
                total_dist += np.min(diffs)
        if matches > best_matches or (matches == best_matches and
                                       total_dist < getattr(best_delay, '_dist', 1e9)):
            best_matches = matches
            best_delay = delay_try

    # Per-injection delay (for drift analysis)
    per_inj_delays = []
    for j, inj in enumerate(hplc_injs):
        t_exp = inj['t_min'] + best_delay
        diffs = np.abs(peak_times - t_exp)
        if len(diffs) > 0:
            best_pk = np.argmin(diffs)
            dist = diffs[best_pk]
            actual_delay = peak_times[best_pk] - inj['t_min']
            per_inj_delays.append({
                'inj': j + 1, 'name': inj['name'],
                'delay': actual_delay, 'dist': dist,
                'matched': dist < 1.5,
            })

    matched_delays = [d['delay'] for d in per_inj_delays if d['matched']]
    delay_drift = max(matched_delays) - min(matched_delays) if len(matched_delays) >= 2 else 0

    return {
        'n_injections': len(hplc_injs),
        'n_peaks': len(peaks),
        'mf_delay': mf_delay,
        'optimal_delay': best_delay,
        'delay_error': (best_delay - mf_delay) if mf_delay is not None else None,
        'matches': best_matches,
        'match_pct': best_matches / len(hplc_injs) * 100,
        'delay_drift': delay_drift,
        'per_inj': per_inj_delays,
    }


def main():
    folders = get_data_folders()

    # Scan all BP SEQs
    bp_seqs = []
    for folder in folders:
        for d in sorted(os.listdir(folder)):
            full = os.path.join(folder, d)
            if not os.path.isdir(full):
                continue
            if detect_seq_type(d) != 'BP':
                continue
            mfs = [f for f in os.listdir(full)
                   if 'MasterFile' in f and f.endswith('.xlsx')
                   and '~' not in f and 'pre_vol' not in f and 'backup' not in f]
            if not mfs:
                mfs = [f for f in os.listdir(full)
                       if 'MasterFile' in f and f.endswith('.xlsx')
                       and '~' not in f and 'pre_vol' not in f]
            if mfs:
                bp_seqs.append((d, full, os.path.join(full, mfs[0])))

    print(f"BP SEQs: {len(bp_seqs)}")

    # Also load COL/BP ratio data if available
    ratio_data = {}
    csv_path = Path('_results/diag_col_bp/col_bp_comparison.csv')
    if csv_path.exists():
        import csv
        with open(csv_path, encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                bp_seq = row.get('bp_seq', '')
                if bp_seq:
                    if bp_seq not in ratio_data:
                        ratio_data[bp_seq] = []
                    try:
                        ratio_data[bp_seq].append(float(row.get('ratio_doc', 0)))
                    except (ValueError, TypeError):
                        pass

    # Analyze each BP
    results = []
    for seq_name, seq_path, mf_path in bp_seqs:
        print(f"  {seq_name}...", end=" ", flush=True)
        try:
            r = auto_detect_delay(mf_path, method="BP")
            r['seq'] = seq_name
            if 'error' in r:
                print(f"ERROR: {r['error']}")
            else:
                error = r.get('delay_error')
                error_str = f"{error:+.1f}" if error is not None else "?"
                verdict = "OK" if error is not None and abs(error) < 2 else (
                    "SUSPECT" if error is not None and abs(error) >= 2 else "?")
                r['verdict'] = verdict

                # Add ratio info
                ratios = ratio_data.get(seq_name, [])
                r['ratio_median'] = float(np.median(ratios)) if ratios else None
                r['ratio_n'] = len(ratios)

                print(f"MF={r['mf_delay']}  opt={r['optimal_delay']:.1f}  "
                      f"err={error_str}  match={r['match_pct']:.0f}%  "
                      f"drift={r['delay_drift']:.1f}min  {verdict}"
                      + (f"  ratio={r['ratio_median']:.3f}(n={r['ratio_n']})" if r['ratio_median'] else ""))
            results.append(r)
        except Exception as e:
            print(f"EXCEPTION: {e}")
            results.append({'seq': seq_name, 'error': str(e)})

    # Summary
    valid = [r for r in results if 'error' not in r]
    ok = [r for r in valid if r.get('verdict') == 'OK']
    suspect = [r for r in valid if r.get('verdict') == 'SUSPECT']

    print(f"\n{'='*80}")
    print(f"RESUM — {len(valid)} SEQs analitzades")
    print(f"{'='*80}")
    print(f"OK (delay error <2 min): {len(ok)}")
    print(f"SUSPECT (delay error >=2 min): {len(suspect)}")

    if suspect:
        print(f"\n=== DELAY SUSPECT ===")
        print(f"{'SEQ':20s} {'MF_delay':>8s} {'Optimal':>8s} {'Error':>8s} {'Match%':>7s} {'Drift':>6s} {'Ratio':>8s}")
        for r in sorted(suspect, key=lambda x: abs(x.get('delay_error') or 0), reverse=True):
            mfd = f"{r['mf_delay']:.1f}" if r['mf_delay'] is not None else "nan"
            err = f"{r['delay_error']:+.1f}" if r['delay_error'] is not None else "?"
            ratio = f"{r['ratio_median']:.3f}" if r['ratio_median'] else "-"
            print(f"{r['seq']:20s} {mfd:>8s} {r['optimal_delay']:>8.1f} {err:>8s} "
                  f"{r['match_pct']:>6.0f}% {r['delay_drift']:>5.1f}m {ratio:>8s}")

    # Cross-reference: delay error vs ratio
    print(f"\n=== CORRELACIO DELAY ERROR vs RATIO COL/BP ===")
    paired = [r for r in valid if r.get('ratio_median') is not None and r.get('delay_error') is not None]
    if len(paired) >= 3:
        errors = np.array([abs(r['delay_error']) for r in paired])
        ratios = np.array([r['ratio_median'] for r in paired])
        ratio_dev = np.abs(ratios - 1.0)
        corr = np.corrcoef(errors, ratio_dev)[0, 1]
        print(f"n={len(paired)}  R(|delay_error| vs |ratio-1|)={corr:.3f}")
        print(f"\nDetall:")
        for r in sorted(paired, key=lambda x: abs(x.get('delay_error', 0)), reverse=True):
            err = abs(r['delay_error'])
            rdev = abs(r['ratio_median'] - 1.0)
            flag = "***" if err > 2 and rdev > 0.15 else ""
            print(f"  {r['seq']:20s}  delay_err={r['delay_error']:+6.1f}  "
                  f"ratio={r['ratio_median']:.3f}  |ratio-1|={rdev:.3f}  {flag}")
    else:
        print(f"Insuficients dades (n={len(paired)})")

    # Save
    out = Path('_results/diag_delay')
    out.mkdir(parents=True, exist_ok=True)

    import csv
    csv_path = out / 'bp_delay_analysis.csv'
    fields = ['seq', 'verdict', 'n_injections', 'n_peaks', 'mf_delay',
              'optimal_delay', 'delay_error', 'match_pct', 'delay_drift',
              'ratio_median', 'ratio_n']
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter=';', extrasaction='ignore')
        w.writeheader()
        for r in results:
            if 'error' not in r:
                w.writerow(r)
    print(f"\nCSV: {csv_path}")

    # Plot
    if paired:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle("Delay error vs COL/BP ratio", fontsize=13, fontweight='bold')

        errors_all = [r.get('delay_error', 0) for r in valid if r.get('delay_error') is not None]

        # 1. Histogram delay errors
        ax = axes[0]
        ax.hist(errors_all, bins=20, color='#228be6', alpha=0.7, edgecolor='white')
        ax.axvline(0, color='k', ls='--', alpha=0.3)
        ax.axvline(2, color='r', ls='--', alpha=0.3, label='+2 min')
        ax.axvline(-2, color='r', ls='--', alpha=0.3, label='-2 min')
        ax.set_title(f'Distribucio delay error ({len(errors_all)} SEQs)')
        ax.set_xlabel('Delay error (optimal - MF) [min]')
        ax.legend(fontsize=7)

        # 2. Scatter delay error vs ratio deviation
        ax = axes[1]
        de = np.array([r['delay_error'] for r in paired])
        rd = np.array([r['ratio_median'] for r in paired])
        colors = ['red' if abs(e) > 2 else '#228be6' for e in de]
        ax.scatter(de, rd, c=colors, s=30, alpha=0.7, edgecolors='white')
        ax.axhline(1.0, color='k', ls='--', alpha=0.3)
        ax.axvline(0, color='k', ls='--', alpha=0.3)
        for r in paired:
            if abs(r['delay_error']) > 2 or abs(r['ratio_median'] - 1.0) > 0.2:
                ax.annotate(r['seq'][:8], (r['delay_error'], r['ratio_median']),
                           fontsize=6, alpha=0.7)
        ax.set_xlabel('Delay error (min)')
        ax.set_ylabel('Ratio COL/BP (mediana)')
        ax.set_title(f'Delay error vs ratio (R={corr:.3f})')

        # 3. |delay error| vs |ratio - 1|
        ax = axes[2]
        ax.scatter(np.abs(de), np.abs(rd - 1), c=colors, s=30, alpha=0.7, edgecolors='white')
        if len(de) >= 3:
            z = np.polyfit(np.abs(de), np.abs(rd - 1), 1)
            xf = np.linspace(0, np.abs(de).max(), 50)
            ax.plot(xf, np.polyval(z, xf), 'r-', alpha=0.5)
        ax.set_xlabel('|Delay error| (min)')
        ax.set_ylabel('|Ratio - 1|')
        ax.set_title('Magnitud error delay vs desviacio ratio')

        for a in axes:
            a.tick_params(labelsize=7)
            a.spines['top'].set_visible(False)
            a.spines['right'].set_visible(False)

        fig.tight_layout()
        p = out / 'delay_vs_ratio.png'
        fig.savefig(str(p), dpi=150, bbox_inches='tight')
        print(f"Plot: {p}")

    print("\nFet!")


if __name__ == "__main__":
    main()
