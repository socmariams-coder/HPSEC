"""
TOC peak count vs injection count diagnostic.
==============================================
For each SEQ:
1. Read 2-TOC (continuous signal)
2. Count main peaks
3. Compare with injection count from 1-HPLC-SEQ
4. Show assignment quality from 4-TOC_CALC
5. Generate visual diagnostic plot

Usage:
    python _diag_toc_peaks.py                 # all SEQs
    python _diag_toc_peaks.py --seq 231       # specific SEQ
    python _diag_toc_peaks.py --plot          # with plots
    python _diag_toc_peaks.py --method BP     # only BP
"""
import os, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d

from hpsec_config import get_data_folders
from hpsec_consolidate import extract_seq_number, detect_seq_type


def read_toc_continuous(mf_path):
    """Read the full continuous TOC signal from 2-TOC."""
    try:
        df = pd.read_excel(mf_path, sheet_name='2-TOC', header=None)
    except Exception:
        return None

    # Find header row
    data_start = None
    for i in range(min(20, len(df))):
        try:
            float(df.iloc[i, 0])
            data_start = i
            break
        except (ValueError, TypeError):
            continue

    if data_start is None or data_start < 1:
        return None

    df = pd.read_excel(mf_path, sheet_name='2-TOC', header=data_start - 1)

    # Find TOC column
    toc_col = None
    for c in df.columns:
        cs = str(c).lower()
        if 'toc' in cs and 'ppb' in cs:
            toc_col = c
            break
    if toc_col is None:
        for c in df.columns:
            if 'toc' in str(c).lower():
                toc_col = c
                break
    if toc_col is None:
        return None

    y = pd.to_numeric(df[toc_col], errors='coerce').values
    # Time: 4 sec cadence
    t = np.arange(len(y)) * (4.0 / 60.0)

    return {'t': t, 'y': y, 'n_rows': len(y)}


def read_injections(mf_path):
    """Read injection list from 1-HPLC-SEQ."""
    try:
        df = pd.read_excel(mf_path, sheet_name='1-HPLC-SEQ')
    except Exception:
        return None

    name_col = None
    for c in df.columns:
        if 'sample' in str(c).lower() and 'name' in str(c).lower():
            name_col = c
            break
    if name_col is None:
        return None

    injs = []
    for _, row in df.iterrows():
        name = str(row[name_col]).strip()
        line = int(row.get('Line#', 0))
        inj_num = int(row.get('Inj#', 1))
        injs.append({'name': name, 'line': line, 'inj_num': inj_num})

    return injs


def read_toc_calc_assignments(mf_path):
    """Read 4-TOC_CALC injection-to-TOC-row mapping."""
    try:
        df = pd.read_excel(mf_path, sheet_name='4-TOC_CALC')
    except Exception:
        return None

    if 'Inj_Index' not in df.columns or 'TOC_Row' not in df.columns:
        return None

    assignments = []
    for idx in sorted(df['Inj_Index'].unique()):
        if idx <= 0:
            continue
        rows = df[df['Inj_Index'] == idx]
        toc_rows = rows['TOC_Row'].values
        sample = rows['Sample'].dropna().unique()
        sample_str = str(sample[0]) if len(sample) > 0 else '?'
        t_rel = rows['Temps_Relatiu (min)'].dropna().values if 'Temps_Relatiu (min)' in rows.columns else []

        assignments.append({
            'inj_index': int(idx),
            'toc_row_start': int(toc_rows[0]),
            'toc_row_end': int(toc_rows[-1]),
            'sample': sample_str,
            't_start': float(t_rel[0]) if len(t_rel) > 0 else None,
            't_end': float(t_rel[-1]) if len(t_rel) > 0 else None,
        })

    return assignments


def detect_toc_peaks(t, y, method='BP'):
    """Detect main peaks in continuous TOC signal."""
    y_clean = np.nan_to_num(y, nan=np.nanmedian(y[~np.isnan(y)]))
    dt = t[1] - t[0] if len(t) > 1 else 4.0 / 60.0

    # Smooth
    y_smooth = uniform_filter1d(y_clean, size=max(3, int(0.3 / dt)))

    # Baseline
    baseline = np.percentile(y_smooth, 15)

    # Height threshold: baseline + 5% of dynamic range
    dyn_range = np.max(y_smooth) - baseline
    height_thresh = baseline + dyn_range * 0.05

    # Minimum distance between peaks
    if method == 'BP':
        min_dist = int(8.0 / dt)     # BP: ~11 min per injection
    else:
        min_dist = int(50.0 / dt)    # COLUMN: ~78 min per injection

    peaks, props = find_peaks(y_smooth, height=height_thresh,
                               distance=min_dist, prominence=max(5, dyn_range * 0.02))

    # Classify peaks
    peak_info = []
    for i, pk in enumerate(peaks):
        val = float(y_clean[pk])
        prom = float(props['prominences'][i])

        # Is it likely a real sample peak or just noise/MQ?
        is_significant = prom > dyn_range * 0.05

        peak_info.append({
            'index': int(pk),
            't': float(t[pk]),
            'value': val,
            'prominence': prom,
            'significant': is_significant,
        })

    return {
        'peaks': peak_info,
        'n_peaks': len(peaks),
        'n_significant': sum(1 for p in peak_info if p['significant']),
        'baseline': float(baseline),
        'y_smooth': y_smooth,
    }


def check_assignment_quality(toc_data, assignments, peak_info):
    """Check if each assignment window contains a peak."""
    t = toc_data['t']
    y = toc_data['y']
    results = []

    for a in assignments:
        # Get signal in assignment window
        row_start = max(0, a['toc_row_start'] - 1)  # 1-indexed
        row_end = min(len(y), a['toc_row_end'])
        y_window = y[row_start:row_end]
        t_window = t[row_start:row_end]

        y_valid = y_window[~np.isnan(y_window)]
        if len(y_valid) < 5:
            results.append({**a, 'has_peak': False, 'reason': 'no_data'})
            continue

        baseline = np.percentile(y_valid, 20)
        y_net = y_valid - baseline
        peak_val = float(np.max(y_net))
        noise = float(np.std(y_net[y_net < np.percentile(y_net, 30)])) if len(y_net) > 5 else 1
        snr = peak_val / noise if noise > 0 else 0

        # Check if any detected peak falls in this window
        peaks_in_window = [p for p in peak_info['peaks']
                          if row_start <= p['index'] <= row_end]

        is_control = any(x in a['sample'].lower() for x in ['mq', 'naoh', 'blanc', 'blnc'])

        results.append({
            **a,
            'has_peak': snr > 5 and not is_control,
            'has_detected_peak': len(peaks_in_window) > 0,
            'snr': snr,
            'peak_val': peak_val,
            'baseline': baseline,
            'is_control': is_control,
            'n_peaks_in_window': len(peaks_in_window),
        })

    return results


def analyze_seq(seq_path, do_plot=False):
    """Full TOC peak analysis for one SEQ."""
    seq_name = os.path.basename(seq_path)
    method = detect_seq_type(seq_name)

    # Find MasterFile
    mfs = [f for f in os.listdir(seq_path)
           if 'MasterFile' in f and f.endswith('.xlsx')
           and '~' not in f and 'pre_vol' not in f and 'backup' not in f]
    if not mfs:
        mfs = [f for f in os.listdir(seq_path)
               if 'MasterFile' in f and f.endswith('.xlsx')
               and '~' not in f and 'pre_vol' not in f]
    if not mfs:
        return {'seq': seq_name, 'error': 'no_masterfile'}

    mf_path = os.path.join(seq_path, mfs[0])

    # Load data
    toc_data = read_toc_continuous(mf_path)
    if toc_data is None:
        return {'seq': seq_name, 'error': 'no_toc'}

    injections = read_injections(mf_path)
    if not injections:
        return {'seq': seq_name, 'error': 'no_hplc'}

    assignments = read_toc_calc_assignments(mf_path)

    # Detect peaks
    peak_info = detect_toc_peaks(toc_data['t'], toc_data['y'], method)

    # Count injections by type
    n_total = len(injections)
    n_samples = sum(1 for inj in injections
                    if not any(x in inj['name'].lower()
                              for x in ['mq', 'naoh', 'blanc', 'blnc']))
    n_controls = n_total - n_samples
    n_khp = sum(1 for inj in injections if 'khp' in inj['name'].lower())

    # Check assignment quality
    assignment_quality = None
    if assignments:
        assignment_quality = check_assignment_quality(toc_data, assignments, peak_info)

    # Diagnosis
    n_peaks = peak_info['n_peaks']
    diff = n_peaks - n_total
    if abs(diff) <= 1:
        verdict = 'OK'
    elif abs(diff) <= 3:
        verdict = 'CHECK'
    else:
        verdict = 'MISMATCH'

    result = {
        'seq': seq_name,
        'method': method,
        'n_injections': n_total,
        'n_samples': n_samples,
        'n_controls': n_controls,
        'n_khp': n_khp,
        'n_peaks_detected': n_peaks,
        'n_peaks_significant': peak_info['n_significant'],
        'diff': diff,
        'verdict': verdict,
        'has_4toc': assignments is not None,
    }

    if assignment_quality:
        n_with_peak = sum(1 for a in assignment_quality
                         if a.get('has_detected_peak') and not a.get('is_control'))
        n_sample_assignments = sum(1 for a in assignment_quality if not a.get('is_control'))
        result['n_correct_assignments'] = n_with_peak
        result['n_sample_assignments'] = n_sample_assignments
        result['assignment_pct'] = (n_with_peak / n_sample_assignments * 100
                                     if n_sample_assignments > 0 else 0)

    if do_plot:
        _make_plot(result, toc_data, peak_info, injections, assignments,
                   assignment_quality, seq_path)

    return result


def _make_plot(result, toc_data, peak_info, injections, assignments,
               assignment_quality, seq_path):
    """Generate diagnostic plot."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    seq_name = result['seq']
    t = toc_data['t']
    y = toc_data['y']

    fig, axes = plt.subplots(2, 1, figsize=(18, 10))
    fig.suptitle(
        f"{seq_name} ({result['method']}) — TOC peaks vs injeccions\n"
        f"Injeccions: {result['n_injections']} ({result['n_samples']} mostres + "
        f"{result['n_controls']} controls, {result['n_khp']} KHP) | "
        f"Pics: {result['n_peaks_detected']} | "
        f"Verdict: {result['verdict']}",
        fontsize=12, fontweight='bold')

    # A) Full TOC with detected peaks
    ax = axes[0]
    ax.plot(t, y, 'b-', lw=0.3, alpha=0.5, label='TOC raw')
    ax.plot(t, peak_info['y_smooth'], 'b-', lw=0.6, alpha=0.8, label='Smooth')
    ax.axhline(peak_info['baseline'], color='orange', ls='--', lw=0.5, alpha=0.5,
               label=f"Baseline {peak_info['baseline']:.0f}")

    # Detected peaks
    for i, pk in enumerate(peak_info['peaks']):
        color = '#E74C3C' if pk['significant'] else '#95A5A6'
        ax.plot(pk['t'], pk['value'], 'v', color=color, ms=6, zorder=5)
        ax.annotate(f"P{i+1}", (pk['t'], pk['value'] + 8),
                   fontsize=6, ha='center', color=color, fontweight='bold')

    # Assignment windows from 4-TOC_CALC
    if assignments:
        colors = plt.colormaps['tab20'](np.linspace(0, 1, max(len(assignments), 1)))
        for i, a in enumerate(assignments):
            rs = max(0, a['toc_row_start'] - 1)
            re = min(len(t) - 1, a['toc_row_end'])
            is_khp = 'khp' in a['sample'].lower()
            is_ctrl = any(x in a['sample'].lower() for x in ['mq', 'naoh', 'blanc'])
            alpha = 0.15 if is_khp else (0.05 if is_ctrl else 0.08)
            color = '#E74C3C' if is_khp else ('#95A5A6' if is_ctrl else colors[i % len(colors)])
            ax.axvspan(t[rs], t[re], alpha=alpha, color=color)
            # Label
            mid_t = (t[rs] + t[re]) / 2
            ax.annotate(f"{a['inj_index']}:{a['sample'][:8]}", (mid_t, peak_info['baseline']),
                       fontsize=4, rotation=90, va='bottom', ha='center',
                       color=color, alpha=0.7)

    ax.set_ylabel('ppb')
    ax.set_title('Senyal TOC continu + pics detectats + finestres assignacio')
    ax.legend(fontsize=7, loc='upper right')

    # B) Assignment quality: SNR per injection
    ax = axes[1]
    if assignment_quality:
        inj_indices = [a['inj_index'] for a in assignment_quality]
        snrs = [a.get('snr', 0) for a in assignment_quality]
        bar_colors = []
        for a in assignment_quality:
            if a.get('is_control'):
                bar_colors.append('#95A5A6')  # grey for controls
            elif 'khp' in a.get('sample', '').lower():
                bar_colors.append('#E74C3C')  # red for KHP
            elif a.get('has_detected_peak'):
                bar_colors.append('#27AE60')  # green = peak matches
            else:
                bar_colors.append('#F39C12')  # orange = no peak in window

        x = np.arange(len(inj_indices))
        ax.bar(x, snrs, color=bar_colors, alpha=0.7, edgecolor='white')
        ax.axhline(5, color='orange', ls='--', lw=0.8, label='SNR=5')
        ax.set_xticks(x)
        labels = [f"{a['inj_index']}\n{a['sample'][:6]}" for a in assignment_quality]
        ax.set_xticklabels(labels, fontsize=5, rotation=0)
        ax.set_ylabel('SNR')
        ax.set_title('SNR per assignacio (verd=pic OK, taronja=sense pic, gris=control, vermell=KHP)')
        ax.legend(fontsize=7)

        # Add assignment percentage
        if result.get('assignment_pct') is not None:
            ax.text(0.98, 0.95,
                    f"Assignacio: {result['assignment_pct']:.0f}% "
                    f"({result['n_correct_assignments']}/{result['n_sample_assignments']})",
                    transform=ax.transAxes, fontsize=10, ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        ax.text(0.5, 0.5, "Sense 4-TOC_CALC (no hi ha assignacio)",
               transform=ax.transAxes, ha='center', va='center', fontsize=14, color='#888')

    for a in axes:
        a.tick_params(labelsize=7)
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = Path('_results/diag_toc_peaks')
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out / f'{seq_name}.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='TOC peak count diagnostic')
    parser.add_argument('--seq', type=int, help='Specific SEQ number')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--method', type=str, help='Filter: COLUMN or BP')
    args = parser.parse_args()

    folders = get_data_folders()

    all_seqs = []
    for folder in folders:
        for d in sorted(os.listdir(folder)):
            full = os.path.join(folder, d)
            if not os.path.isdir(full):
                continue
            has_mf = any('MasterFile' in f and f.endswith('.xlsx') and '~' not in f
                         for f in os.listdir(full))
            if not has_mf:
                continue
            method = detect_seq_type(d)
            num = extract_seq_number(d)
            if args.seq and num != args.seq:
                continue
            if args.method and method != args.method.upper():
                continue
            all_seqs.append(full)

    print(f"SEQs: {len(all_seqs)}")

    results = []
    for seq_path in all_seqs:
        name = os.path.basename(seq_path)
        print(f"  {name:25s}", end=" ", flush=True)
        try:
            r = analyze_seq(seq_path, do_plot=args.plot)
            results.append(r)
            if 'error' in r:
                print(f"ERROR: {r['error']}")
            else:
                assign_str = ""
                if r.get('assignment_pct') is not None:
                    assign_str = f"  assign={r['assignment_pct']:.0f}%"
                print(f"inj={r['n_injections']:2d} ({r['n_samples']}S+{r['n_controls']}C) "
                      f"pics={r['n_peaks_detected']:2d} "
                      f"diff={r['diff']:+d}  {r['verdict']}{assign_str}")
        except Exception as e:
            print(f"EXCEPTION: {e}")
            results.append({'seq': name, 'error': str(e)})

    # Summary
    valid = [r for r in results if 'error' not in r]
    ok = [r for r in valid if r['verdict'] == 'OK']
    check = [r for r in valid if r['verdict'] == 'CHECK']
    mismatch = [r for r in valid if r['verdict'] == 'MISMATCH']

    print(f"\n{'='*60}")
    print(f"RESUM — {len(valid)} SEQs")
    print(f"{'='*60}")
    print(f"OK (diff <=1):       {len(ok)}")
    print(f"CHECK (diff 2-3):    {len(check)}")
    print(f"MISMATCH (diff >3):  {len(mismatch)}")

    if mismatch:
        print(f"\n=== MISMATCH ===")
        for r in sorted(mismatch, key=lambda x: abs(x['diff']), reverse=True):
            assign = f"assign={r['assignment_pct']:.0f}%" if r.get('assignment_pct') is not None else ""
            print(f"  {r['seq']:25s} inj={r['n_injections']:2d} pics={r['n_peaks_detected']:2d} "
                  f"diff={r['diff']:+d}  {assign}")

    if check:
        print(f"\n=== CHECK ===")
        for r in check:
            assign = f"assign={r['assignment_pct']:.0f}%" if r.get('assignment_pct') is not None else ""
            print(f"  {r['seq']:25s} inj={r['n_injections']:2d} pics={r['n_peaks_detected']:2d} "
                  f"diff={r['diff']:+d}  {assign}")

    if args.plot:
        print(f"\nPlots: _results/diag_toc_peaks/")

    print("\nFet!")


if __name__ == "__main__":
    main()
