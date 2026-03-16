"""
TOC alignment diagnostic — find true delay from baseline transition.
====================================================================
Strategy:
1. Read TOC continuous signal (2-TOC)
2. Find where baseline becomes non-flat (first real peak)
3. Read HPLC injection timestamps (1-HPLC-SEQ)
4. delay = t_first_peak_TOC - t_first_injection_HPLC
5. Validate: using this delay, partition TOC into windows per injection
6. Check each window has a peak (samples) or not (MQ/NaOH)

Usage:
    python _diag_toc_alignment.py --seq 231 --plot
    python _diag_toc_alignment.py --method BP --plot
    python _diag_toc_alignment.py --plot    # all
"""
import os, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.ndimage import uniform_filter1d

from hpsec_config import get_data_folders
from hpsec_consolidate import extract_seq_number, detect_seq_type


def read_toc_signal(mf_path):
    """Read TOC continuous signal with timestamps."""
    try:
        df = pd.read_excel(mf_path, sheet_name='2-TOC', header=None)
    except Exception:
        return None

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

    toc_col = None
    date_col = None
    for c in df.columns:
        cs = str(c).lower()
        if toc_col is None and 'toc' in cs:
            toc_col = c
        if date_col is None and 'started' in cs:
            date_col = c

    if toc_col is None:
        return None

    y = pd.to_numeric(df[toc_col], errors='coerce').values
    t = np.arange(len(y)) * (4.0 / 60.0)  # 4 sec cadence

    # Absolute timestamps
    t0 = None
    if date_col:
        dates = pd.to_datetime(df[date_col], errors='coerce')
        valid = dates.dropna()
        if len(valid) > 0:
            t0 = valid.iloc[0]

    return {'y': y, 't': t, 't0': t0, 'n': len(y)}


def read_hplc_injections(mf_path, t0_toc):
    """Read HPLC injection times relative to TOC start."""
    try:
        df = pd.read_excel(mf_path, sheet_name='1-HPLC-SEQ')
    except Exception:
        return None

    name_col = None
    date_col = None
    for c in df.columns:
        if name_col is None and 'sample' in str(c).lower() and 'name' in str(c).lower():
            name_col = c
        if date_col is None and 'acquired' in str(c).lower():
            date_col = c

    if name_col is None or date_col is None or t0_toc is None:
        return None

    injs = []
    for _, row in df.iterrows():
        try:
            name = str(row[name_col]).strip()
            dt = pd.to_datetime(row[date_col])
            t_min = (dt - t0_toc).total_seconds() / 60.0

            is_mq = any(x in name.lower() for x in ['mq', 'blanc', 'blnc'])
            is_naoh = 'naoh' in name.lower()
            is_khp = 'khp' in name.lower()
            is_control = is_mq or is_naoh

            injs.append({
                'name': name,
                't_min': t_min,
                'is_control': is_control,
                'is_mq': is_mq,
                'is_naoh': is_naoh,
                'is_khp': is_khp,
            })
        except Exception:
            pass

    return injs


def find_baseline_end(y, t, min_stable_min=5.0):
    """Find where the initial stable baseline ends (first significant change).

    Returns t_min of first departure from baseline.
    """
    dt = t[1] - t[0] if len(t) > 1 else 4.0 / 60.0
    y_clean = np.nan_to_num(y, nan=np.nanmedian(y[~np.isnan(y)]))
    y_smooth = uniform_filter1d(y_clean, size=max(3, int(0.5 / dt)))

    # Use first 'min_stable_min' as baseline reference
    n_baseline = int(min_stable_min / dt)
    if n_baseline > len(y_smooth) // 2:
        n_baseline = len(y_smooth) // 4

    baseline = np.median(y_smooth[:n_baseline])
    baseline_std = np.std(y_smooth[:n_baseline])

    # Threshold: baseline + 5*std (significant departure)
    threshold = baseline + max(5 * baseline_std, 10)

    # Scan forward to find first sustained rise above threshold
    window = int(0.5 / dt)  # 0.5 min window
    for i in range(n_baseline, len(y_smooth) - window):
        if np.mean(y_smooth[i:i + window]) > threshold:
            return float(t[i]), baseline, threshold

    return None, baseline, threshold


def find_first_peak(y, t, baseline_end_t, method='BP'):
    """Find the first real peak after baseline departure."""
    dt = t[1] - t[0]
    y_clean = np.nan_to_num(y, nan=np.nanmedian(y[~np.isnan(y)]))
    y_smooth = uniform_filter1d(y_clean, size=max(3, int(0.3 / dt)))

    # Search only after baseline end
    start_idx = int(baseline_end_t / dt) if baseline_end_t else 0
    y_search = y_smooth[start_idx:]
    t_search = t[start_idx:]

    if len(y_search) < 10:
        return None

    # Find first local maximum
    baseline = np.percentile(y_search[:min(50, len(y_search))], 20)
    from scipy.signal import find_peaks

    if method == 'BP':
        min_dist = int(3.0 / dt)  # min 3 min between peaks for BP
        prominence = 15
    else:
        min_dist = int(20.0 / dt)  # min 20 min for COLUMN
        prominence = 10

    peaks, props = find_peaks(y_search, distance=min_dist,
                               prominence=prominence, height=baseline + 10)

    if len(peaks) == 0:
        # Fallback: just find the maximum in the first injection window
        window = int(15.0 / dt) if method == 'BP' else int(80.0 / dt)
        pk = np.argmax(y_search[:min(window, len(y_search))])
        return float(t_search[pk])

    return float(t_search[peaks[0]])


def analyze_alignment(seq_path, do_plot=False):
    """Full alignment analysis for one SEQ."""
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
    toc = read_toc_signal(mf_path)
    if toc is None:
        return {'seq': seq_name, 'error': 'no_toc'}

    injs = read_hplc_injections(mf_path, toc['t0'])
    if not injs:
        return {'seq': seq_name, 'error': 'no_hplc'}

    # Cadence
    hplc_times = np.array([inj['t_min'] for inj in injs])
    if len(hplc_times) >= 2:
        cadence = np.median(np.diff(hplc_times))
    else:
        cadence = 11.0 if method == 'BP' else 78.0

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

    # Step 1: Find baseline end
    baseline_end_t, baseline_val, threshold = find_baseline_end(
        toc['y'], toc['t'])

    if baseline_end_t is None:
        return {'seq': seq_name, 'error': 'no_baseline_departure',
                'baseline': baseline_val}

    # Step 2: Find first peak
    first_peak_t = find_first_peak(toc['y'], toc['t'], baseline_end_t, method)

    if first_peak_t is None:
        return {'seq': seq_name, 'error': 'no_first_peak'}

    # Step 3: Calculate delay
    first_inj_t = hplc_times[0]
    estimated_delay = first_peak_t - first_inj_t

    # Step 4: Validate — partition TOC into windows and check peaks
    dt = toc['t'][1] - toc['t'][0]
    y_clean = np.nan_to_num(toc['y'], nan=np.nanmedian(toc['y'][~np.isnan(toc['y'])]))
    y_smooth = uniform_filter1d(y_clean, size=max(3, int(0.3 / dt)))

    half_window = cadence / 2
    window_results = []
    for j, inj in enumerate(injs):
        t_center = inj['t_min'] + estimated_delay
        t_start = t_center - half_window * 0.3  # asymmetric: peak near start
        t_end = t_center + half_window * 0.7

        mask = (toc['t'] >= t_start) & (toc['t'] <= t_end)
        y_window = y_smooth[mask]

        if len(y_window) < 5:
            window_results.append({
                **inj, 'has_peak': False, 'snr': 0, 'peak_val': 0,
                't_center': t_center, 't_start': t_start, 't_end': t_end})
            continue

        bl = np.percentile(y_window, 20)
        y_net = y_window - bl
        peak_val = float(np.max(y_net))
        noise = float(np.std(y_net[y_net < np.percentile(y_net, 30)])) if np.sum(y_net < np.percentile(y_net, 30)) > 3 else 1.0
        snr = peak_val / noise if noise > 0 else 0

        # Expected: controls (MQ/NaOH) should have low SNR, samples should have high
        has_peak = snr > 5

        window_results.append({
            **inj, 'has_peak': has_peak, 'snr': snr, 'peak_val': peak_val,
            'baseline': bl, 't_center': t_center, 't_start': t_start, 't_end': t_end})

    # Stats
    sample_windows = [w for w in window_results if not w['is_control']]
    control_windows = [w for w in window_results if w['is_control']]
    n_sample_with_peak = sum(1 for w in sample_windows if w['has_peak'])
    n_control_with_peak = sum(1 for w in control_windows if w['has_peak'])

    # Validate with cadence: check peak intervals match injection intervals
    # Find peaks in the smooth signal
    from scipy.signal import find_peaks as sp_find_peaks
    if method == 'BP':
        min_pk_dist = int(6.0 / dt)
    else:
        min_pk_dist = int(40.0 / dt)

    all_peaks, _ = sp_find_peaks(y_smooth, distance=min_pk_dist,
                                  prominence=max(5, (np.max(y_smooth) - baseline_val) * 0.02),
                                  height=baseline_val + 10)

    # Verify first peak matches cadence
    cadence_ok = False
    if len(all_peaks) >= 3:
        peak_t = toc['t'][all_peaks]
        spacings = np.diff(peak_t)
        median_spacing = np.median(spacings)
        cadence_ok = abs(median_spacing - cadence) < cadence * 0.15

    # Verdict
    sample_pct = n_sample_with_peak / len(sample_windows) * 100 if sample_windows else 0
    if sample_pct >= 80 and cadence_ok:
        verdict = 'GOOD'
    elif sample_pct >= 60:
        verdict = 'ACCEPTABLE'
    elif sample_pct >= 40:
        verdict = 'CHECK'
    else:
        verdict = 'BAD'

    result = {
        'seq': seq_name,
        'method': method,
        'n_injections': len(injs),
        'cadence': cadence,
        'mf_delay': mf_delay,
        'estimated_delay': estimated_delay,
        'delay_diff': (estimated_delay - mf_delay) if mf_delay is not None and not np.isnan(mf_delay) else None,
        'first_inj_t': first_inj_t,
        'first_peak_t': first_peak_t,
        'baseline_end_t': baseline_end_t,
        'baseline': baseline_val,
        'n_samples': len(sample_windows),
        'n_controls': len(control_windows),
        'n_sample_with_peak': n_sample_with_peak,
        'sample_peak_pct': sample_pct,
        'cadence_ok': cadence_ok,
        'n_peaks_total': len(all_peaks),
        'verdict': verdict,
        'windows': window_results,
    }

    if do_plot:
        _make_plot(result, toc, y_smooth, all_peaks, injs)

    return result


def _make_plot(result, toc, y_smooth, all_peaks, injs):
    """Generate alignment diagnostic plot."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    seq_name = result['seq']
    t = toc['t']
    y = toc['y']

    fig, ax = plt.subplots(figsize=(20, 6))

    # TOC signal
    ax.plot(t, y, 'b-', lw=0.2, alpha=0.3)
    ax.plot(t, y_smooth, 'b-', lw=0.5, alpha=0.8)

    # Baseline reference
    ax.axhline(result['baseline'], color='orange', ls='--', lw=0.5, alpha=0.5)

    # Baseline end marker
    if result.get('baseline_end_t'):
        ax.axvline(result['baseline_end_t'], color='purple', ls='-', lw=1, alpha=0.5,
                   label=f"Baseline end: {result['baseline_end_t']:.1f} min")

    # First peak marker
    if result.get('first_peak_t'):
        ax.axvline(result['first_peak_t'], color='red', ls='-', lw=1.5, alpha=0.7,
                   label=f"1r pic: {result['first_peak_t']:.1f} min")

    # Detected peaks
    peak_t = t[all_peaks]
    peak_v = y[all_peaks]
    ax.plot(peak_t, peak_v, 'gv', ms=4, alpha=0.5, zorder=4)

    # Injection windows with estimated delay
    for j, w in enumerate(result['windows']):
        t_s = w['t_start']
        t_e = w['t_end']
        t_c = w['t_center']

        if w['is_khp']:
            color = '#E74C3C'
            alpha = 0.15
        elif w['is_control']:
            color = '#95A5A6'
            alpha = 0.05
        elif w['has_peak']:
            color = '#27AE60'
            alpha = 0.08
        else:
            color = '#F39C12'
            alpha = 0.12

        ax.axvspan(t_s, t_e, alpha=alpha, color=color, zorder=1)

        # Label at top
        label_y = ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else np.nanmax(y) * 0.95
        short_name = w['name'][:7]
        fontsize = 4 if len(injs) > 25 else 5
        ax.annotate(f"{j+1}:{short_name}", (t_c, label_y),
                   fontsize=fontsize, ha='center', va='top', rotation=90,
                   color=color, alpha=0.7)

    # Title with key info
    delay_mf = f"{result['mf_delay']:.1f}" if result['mf_delay'] is not None and not (isinstance(result['mf_delay'], float) and np.isnan(result['mf_delay'])) else "nan"
    ax.set_title(
        f"{seq_name} ({result['method']}) — "
        f"Delay MF: {delay_mf} | Estimat: {result['estimated_delay']:.1f} min | "
        f"Diff: {result.get('delay_diff', '?')} min\n"
        f"{result['n_injections']} inj ({result['n_samples']}S+{result['n_controls']}C) | "
        f"Mostres amb pic: {result['n_sample_with_peak']}/{result['n_samples']} "
        f"({result['sample_peak_pct']:.0f}%) | "
        f"Cadencia {'OK' if result['cadence_ok'] else 'CHECK'} ({result['cadence']:.1f} min) | "
        f"Verdict: {result['verdict']}",
        fontsize=10, fontweight='bold')

    ax.set_xlabel('min (des de inici TOC)')
    ax.set_ylabel('ppb')
    ax.legend(fontsize=8, loc='upper right')
    ax.tick_params(labelsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    out = Path('_results/diag_toc_alignment')
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out / f'{seq_name}.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='TOC alignment diagnostic')
    parser.add_argument('--seq', type=int, help='Specific SEQ number')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--method', type=str, help='COLUMN or BP')
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
            r = analyze_alignment(seq_path, do_plot=args.plot)
            results.append(r)
            if 'error' in r:
                print(f"ERROR: {r['error']}")
            else:
                dd = f"{r['delay_diff']:+.1f}" if r['delay_diff'] is not None else "?"
                print(f"delay_MF={r.get('mf_delay','?'):>5}  est={r['estimated_delay']:>6.1f}  "
                      f"diff={dd:>5}  pics={r['n_sample_with_peak']}/{r['n_samples']}  "
                      f"({r['sample_peak_pct']:.0f}%)  {r['verdict']}")
        except Exception as e:
            print(f"EXCEPTION: {e}")
            results.append({'seq': name, 'error': str(e)})

    # Summary
    valid = [r for r in results if 'error' not in r]
    print(f"\n{'='*70}")
    print(f"RESUM — {len(valid)} SEQs")
    print(f"{'='*70}")

    for verdict in ['GOOD', 'ACCEPTABLE', 'CHECK', 'BAD']:
        subset = [r for r in valid if r['verdict'] == verdict]
        print(f"{verdict:12s}: {len(subset)}")

    # Delay discrepancies
    big_diff = [r for r in valid if r.get('delay_diff') is not None and abs(r['delay_diff']) > 2]
    if big_diff:
        print(f"\n=== DELAY DISCREPANCY > 2 min ({len(big_diff)} SEQs) ===")
        for r in sorted(big_diff, key=lambda x: abs(x['delay_diff']), reverse=True)[:20]:
            print(f"  {r['seq']:25s} MF={r['mf_delay']:>6}  est={r['estimated_delay']:>6.1f}  "
                  f"diff={r['delay_diff']:+6.1f}  {r['verdict']}")

    if args.plot:
        print(f"\nPlots: _results/diag_toc_alignment/")

    print("\nFet!")


if __name__ == "__main__":
    main()
