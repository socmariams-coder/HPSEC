"""
Diagnostic: Delay verification for all SEQs.
=============================================
For each SEQ:
1. Read full TOC signal from 2-TOC
2. Read injection mapping from 4-TOC_CALC
3. For each injection, check if there's a clear peak at expected position
4. Flag SEQs where delay seems wrong

Usage:
    python _diag_delay_check.py                 # all SEQs
    python _diag_delay_check.py --seq 231       # specific SEQ
    python _diag_delay_check.py --plot          # generate plots
"""
import os, sys, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

from hpsec_config import get_data_folders
from hpsec_consolidate import extract_seq_number, detect_seq_type


def load_toc_signal(mf_path):
    """Load raw TOC signal from 2-TOC sheet."""
    try:
        df = pd.read_excel(mf_path, sheet_name='2-TOC', header=None)
    except Exception:
        return None, None

    # Find data start (first row with numeric first column)
    data_start = None
    for i in range(min(20, len(df))):
        try:
            float(df.iloc[i, 0])
            data_start = i
            break
        except (ValueError, TypeError):
            continue

    if data_start is None or data_start < 1:
        return None, None

    # Re-read with proper header
    df = pd.read_excel(mf_path, sheet_name='2-TOC', header=data_start - 1)

    # Find TOC column
    toc_col = None
    for c in df.columns:
        if 'toc' in str(c).lower() and 'ppb' in str(c).lower():
            toc_col = c
            break
    if toc_col is None:
        for c in df.columns:
            if 'toc' in str(c).lower():
                toc_col = c
                break

    # Find date column for timestamps
    date_col = None
    for c in df.columns:
        if 'started' in str(c).lower() or 'date' in str(c).lower():
            try:
                vals = pd.to_datetime(df[c].head(5), errors='coerce')
                if vals.notna().sum() >= 3:
                    date_col = c
                    break
            except Exception:
                continue

    if toc_col is None:
        return None, None

    toc_vals = pd.to_numeric(df[toc_col], errors='coerce').values

    # Build time array from timestamps
    timestamps = None
    if date_col:
        try:
            ts = pd.to_datetime(df[date_col], errors='coerce')
            if ts.notna().sum() > len(ts) * 0.5:
                t0 = ts.dropna().iloc[0]
                t_min = (ts - t0).dt.total_seconds() / 60.0
                timestamps = t_min.values
        except Exception:
            pass

    return toc_vals, timestamps


def load_injection_mapping(mf_path):
    """Load 4-TOC_CALC injection mapping."""
    try:
        df = pd.read_excel(mf_path, sheet_name='4-TOC_CALC')
    except Exception:
        return None

    if 'Inj_Index' not in df.columns or 'TOC_Row' not in df.columns:
        return None

    injections = []
    for idx in sorted(df['Inj_Index'].unique()):
        if idx <= 0:
            continue
        rows = df[df['Inj_Index'] == idx]
        toc_rows = rows['TOC_Row'].values
        t_rel = rows['Temps_Relatiu (min)'].dropna().values if 'Temps_Relatiu (min)' in rows.columns else []
        sample = rows['Sample'].dropna().unique()
        sample_str = str(sample[0]) if len(sample) > 0 else '?'

        injections.append({
            'inj_index': int(idx),
            'toc_row_start': int(toc_rows[0]),
            'toc_row_end': int(toc_rows[-1]),
            'n_pts': len(toc_rows),
            't_start': float(t_rel[0]) if len(t_rel) > 0 else None,
            't_end': float(t_rel[-1]) if len(t_rel) > 0 else None,
            'sample': sample_str,
        })

    return injections


def load_hplc_times(mf_path):
    """Load HPLC injection acquisition times from 1-HPLC-SEQ."""
    try:
        df = pd.read_excel(mf_path, sheet_name='1-HPLC-SEQ')
    except Exception:
        return None

    date_col = None
    for c in df.columns:
        if 'acquired' in str(c).lower() or ('date' in str(c).lower() and 'injection' in str(c).lower()):
            date_col = c
            break

    name_col = None
    for c in df.columns:
        if 'sample' in str(c).lower() and 'name' in str(c).lower():
            name_col = c
            break

    if date_col is None or name_col is None:
        return None

    result = []
    for i, row in df.iterrows():
        try:
            dt = pd.to_datetime(row[date_col])
            result.append({
                'line': int(row.get('Line#', i + 1)),
                'name': str(row[name_col]),
                'datetime': dt,
            })
        except Exception:
            continue

    return result


def analyze_peak_in_window(toc_signal, toc_row_start, toc_row_end, t_rel=None):
    """Analyze if there's a clear peak in a TOC window."""
    # Extract signal (1-indexed rows)
    i_start = max(0, toc_row_start - 1)
    i_end = min(len(toc_signal), toc_row_end)
    y = toc_signal[i_start:i_end].astype(float)
    y = y[~np.isnan(y)]

    if len(y) < 10:
        return {'has_peak': False, 'reason': 'insufficient_data', 'n_pts': len(y)}

    # Baseline (bottom 20%)
    baseline = float(np.percentile(y, 20))
    y_net = y - baseline
    peak_val = float(np.max(y_net))
    peak_idx = int(np.argmax(y_net))
    noise = float(np.std(y_net[y_net < np.percentile(y_net, 30)])) if len(y_net) > 5 else 1.0
    snr = peak_val / noise if noise > 0 else 0

    # FWHM
    half_max = peak_val / 2
    above_hm = y_net >= half_max
    if above_hm.any():
        first = np.argmax(above_hm)
        last = len(above_hm) - 1 - np.argmax(above_hm[::-1])
        fwhm_pts = last - first
        dt = (t_rel[-1] - t_rel[0]) / len(t_rel) if t_rel is not None and len(t_rel) > 1 else 0.067
        fwhm_min = fwhm_pts * dt
    else:
        fwhm_min = 0
        fwhm_pts = 0

    # Peak position relative to window
    n = len(y)
    peak_pct = peak_idx / n  # 0=start, 1=end

    # Quality checks
    has_peak = snr > 5 and fwhm_min < 3.0  # Clear peak with reasonable width
    peak_centered = 0.1 < peak_pct < 0.5  # Peak should be in first half for BP

    return {
        'has_peak': has_peak,
        'peak_val': peak_val,
        'baseline': baseline,
        'snr': snr,
        'fwhm_min': fwhm_min,
        'peak_pct': peak_pct,
        'peak_centered': peak_centered,
        'peak_idx': peak_idx,
        'n_pts': len(y),
    }


def find_peaks_in_full_toc(toc_signal, method='BP'):
    """Find all peaks in the full TOC signal."""
    from scipy.signal import find_peaks as sp_find_peaks
    from scipy.ndimage import uniform_filter1d

    y = toc_signal.astype(float)
    y = np.nan_to_num(y, nan=np.nanmedian(y))

    # Smooth
    dt = 0.067  # 4 sec cadence
    window = max(3, int(0.3 / dt))
    y_smooth = uniform_filter1d(y, size=window)

    baseline = np.percentile(y_smooth, 15)
    height_thresh = baseline + (np.max(y_smooth) - baseline) * 0.03

    # BP: peaks every ~11-12 min; COLUMN: every ~78 min
    if method == 'BP':
        min_dist = int(8.0 / dt)
    else:
        min_dist = int(60.0 / dt)

    peaks, props = sp_find_peaks(y_smooth, height=height_thresh,
                                  distance=min_dist, prominence=5)

    return peaks, y_smooth, baseline


def analyze_seq(seq_path, do_plot=False):
    """Full delay analysis for one SEQ."""
    seq_name = os.path.basename(seq_path)
    method = detect_seq_type(seq_name)

    # Find MasterFile
    mfs = [f for f in os.listdir(seq_path)
           if 'MasterFile' in f and f.endswith('.xlsx')
           and 'pre_vol' not in f and 'backup' not in f and '~' not in f]
    if not mfs:
        mfs = [f for f in os.listdir(seq_path)
               if 'MasterFile' in f and f.endswith('.xlsx')
               and '~' not in f and 'pre_vol' not in f]
    if not mfs:
        return {'seq': seq_name, 'error': 'no_masterfile'}

    mf_path = os.path.join(seq_path, mfs[0])

    # Load data
    toc_signal, toc_timestamps = load_toc_signal(mf_path)
    if toc_signal is None:
        return {'seq': seq_name, 'error': 'no_toc_signal'}

    injections = load_injection_mapping(mf_path)
    if not injections:
        return {'seq': seq_name, 'error': 'no_4toc_calc'}

    hplc_times = load_hplc_times(mf_path)

    # Read delay from 0-INFO
    try:
        df_info = pd.read_excel(mf_path, sheet_name='0-INFO', header=None)
        delay = None
        for _, row in df_info.iterrows():
            if str(row.iloc[0]).strip().lower() in ('net delay (min)', 'net delay'):
                try:
                    delay = float(row.iloc[1])
                except (ValueError, TypeError):
                    pass
        volume = None
        for _, row in df_info.iterrows():
            if 'volume' in str(row.iloc[0]).lower():
                try:
                    volume = float(row.iloc[1])
                except (ValueError, TypeError):
                    pass
    except Exception:
        delay = None
        volume = None

    # Analyze each injection
    n_good = 0
    n_bad = 0
    n_khp = 0
    khp_peaks = []
    sample_peaks = []
    results = []

    for inj in injections:
        t_rel = None
        if inj['t_start'] is not None and inj['t_end'] is not None:
            t_rel = np.linspace(inj['t_start'], inj['t_end'], inj['n_pts'])

        pk = analyze_peak_in_window(toc_signal, inj['toc_row_start'],
                                     inj['toc_row_end'], t_rel)
        pk['inj_index'] = inj['inj_index']
        pk['sample'] = inj['sample']

        is_khp = 'khp' in inj['sample'].lower()
        is_mq = 'mq' in inj['sample'].lower() or 'blanc' in inj['sample'].lower()
        is_naoh = 'naoh' in inj['sample'].lower()

        if is_khp:
            n_khp += 1
            khp_peaks.append(pk)
        elif not is_mq and not is_naoh:
            sample_peaks.append(pk)

        if pk['has_peak']:
            n_good += 1
        else:
            n_bad += 1

        results.append(pk)

    # Find peaks in full TOC for comparison
    full_peaks, y_smooth, full_baseline = find_peaks_in_full_toc(toc_signal, method)

    # Expected peak position for BP: ~2-3 min from injection start
    # Expected peak position for COLUMN: ~20-22 min from injection start
    if method == 'BP':
        expected_peak_pct = 0.2  # peak at 20% of window (~2 min in 10 min window)
    else:
        expected_peak_pct = 0.3  # peak at 30% of window (~20 min in 70 min window)

    # Diagnosis
    diagnosis = []

    # Check KHP peaks
    if khp_peaks:
        khp_with_peak = [p for p in khp_peaks if p['has_peak']]
        khp_no_peak = [p for p in khp_peaks if not p['has_peak']]
        if khp_no_peak:
            diagnosis.append(f"KHP: {len(khp_no_peak)}/{len(khp_peaks)} sense pic clar")
        if khp_with_peak:
            avg_pct = np.mean([p['peak_pct'] for p in khp_with_peak])
            avg_fwhm = np.mean([p['fwhm_min'] for p in khp_with_peak])
            if method == 'BP' and avg_pct > 0.6:
                diagnosis.append(f"KHP pic massa tard (pct={avg_pct:.2f}) -> delay massa petit?")
            elif method == 'BP' and avg_pct < 0.05:
                diagnosis.append(f"KHP pic massa d'hora (pct={avg_pct:.2f}) -> delay massa gran?")
            if avg_fwhm > 3.0:
                diagnosis.append(f"KHP FWHM anormal ({avg_fwhm:.1f} min) -> baseline no torna")

    # Check sample peaks
    if sample_peaks:
        n_sample_ok = sum(1 for p in sample_peaks if p['has_peak'])
        if n_sample_ok < len(sample_peaks) * 0.5:
            diagnosis.append(f"Mostres: {n_sample_ok}/{len(sample_peaks)} amb pic clar")

    # Compare number of detected peaks vs expected injections
    n_expected = len(injections)
    n_detected_full = len(full_peaks)
    if abs(n_detected_full - n_expected) > n_expected * 0.2:
        diagnosis.append(f"Pics detectats ({n_detected_full}) vs injeccions ({n_expected}): discrepancia")

    # Overall verdict
    if not diagnosis:
        verdict = "OK"
    elif any('delay' in d.lower() for d in diagnosis):
        verdict = "DELAY_SUSPECT"
    elif any('sense pic' in d.lower() or 'fwhm anormal' in d.lower() for d in diagnosis):
        verdict = "CHECK"
    else:
        verdict = "WARNING"

    result = {
        'seq': seq_name,
        'method': method,
        'n_injections': n_expected,
        'n_peaks_full_toc': n_detected_full,
        'n_good_windows': n_good,
        'n_bad_windows': n_bad,
        'n_khp': n_khp,
        'delay': delay,
        'volume': volume,
        'verdict': verdict,
        'diagnosis': diagnosis,
        'injection_results': results,
    }

    # Plot
    if do_plot:
        _make_plot(result, toc_signal, toc_timestamps, y_smooth,
                   full_peaks, full_baseline, injections, seq_path)

    return result


def _make_plot(result, toc_signal, toc_timestamps, y_smooth,
               full_peaks, full_baseline, injections, seq_path):
    """Generate diagnostic plot."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    seq_name = result['seq']
    method = result['method']

    fig, axes = plt.subplots(2, 1, figsize=(18, 10))
    fig.suptitle(f"{seq_name} ({method}) — Delay check\n"
                 f"Verdict: {result['verdict']}  |  "
                 f"delay={result['delay']}min  vol={result['volume']}uL  |  "
                 f"{result['n_injections']} inj, {result['n_peaks_full_toc']} pics detectats",
                 fontsize=12, fontweight='bold')

    # Full TOC signal
    ax = axes[0]
    x = np.arange(len(toc_signal))
    ax.plot(x, toc_signal, 'b-', lw=0.3, alpha=0.5, label='TOC raw')
    ax.plot(x, y_smooth, 'r-', lw=0.5, alpha=0.7, label='Smooth')
    ax.plot(full_peaks, toc_signal[full_peaks], 'gv', ms=6, label=f'{len(full_peaks)} pics')
    ax.axhline(full_baseline, color='orange', ls='--', lw=0.5)

    # Mark injection windows
    colors = plt.colormaps['tab20'](np.linspace(0, 1, max(len(injections), 1)))
    for i, inj in enumerate(injections):
        rs = inj['toc_row_start'] - 1
        re = inj['toc_row_end']
        is_khp = 'khp' in inj['sample'].lower()
        ax.axvspan(rs, re, alpha=0.08 if not is_khp else 0.15,
                   color='red' if is_khp else colors[i % len(colors)])
        # Label at top
        mid = (rs + re) / 2
        ax.annotate(f"{inj['inj_index']}", (mid, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 500),
                    fontsize=5, ha='center', va='top', color='black', alpha=0.6)

    ax.set_xlabel('TOC row')
    ax.set_ylabel('ppb')
    ax.set_title('Senyal TOC complet amb finestres d\'injeccio')
    ax.legend(fontsize=7)

    # Per-injection peak quality
    ax = axes[1]
    inj_indices = [r['inj_index'] for r in result['injection_results']]
    snrs = [r.get('snr', 0) for r in result['injection_results']]
    peak_pcts = [r.get('peak_pct', 0.5) for r in result['injection_results']]
    fwhms = [r.get('fwhm_min', 0) for r in result['injection_results']]

    x_pos = np.arange(len(inj_indices))
    bar_colors = []
    for r in result['injection_results']:
        if 'khp' in r.get('sample', '').lower():
            bar_colors.append('#E74C3C')
        elif r.get('has_peak'):
            bar_colors.append('#27AE60')
        else:
            bar_colors.append('#95A5A6')

    ax.bar(x_pos, snrs, color=bar_colors, alpha=0.7, edgecolor='white')
    ax.axhline(5, color='orange', ls='--', lw=0.8, label='SNR threshold')
    ax.set_xticks(x_pos)
    labels = [f"{r['inj_index']}\n{r.get('sample', '?')[:8]}" for r in result['injection_results']]
    ax.set_xticklabels(labels, fontsize=6, rotation=0)
    ax.set_ylabel('SNR')
    ax.set_title('SNR per injeccio (vermell=KHP, verd=OK, gris=sense pic)')
    ax.legend(fontsize=7)

    # Add diagnosis text
    if result['diagnosis']:
        diag_text = '\n'.join(result['diagnosis'])
        fig.text(0.02, 0.02, diag_text, fontsize=9, color='red',
                 bbox=dict(boxstyle='round', facecolor='#FADBD8', alpha=0.8))

    fig.tight_layout(rect=[0, 0.05, 1, 0.93])
    out = Path('_results/diag_delay')
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out / f'{seq_name}_delay.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Delay verification diagnostic')
    parser.add_argument('--seq', type=int, help='Specific SEQ number')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--method', type=str, help='Filter by method (COLUMN/BP)')
    args = parser.parse_args()

    folders = get_data_folders()
    print(f"Data folders: {folders}")

    # Scan SEQs
    all_seqs = []
    for folder in folders:
        for d in sorted(os.listdir(folder)):
            full = os.path.join(folder, d)
            if not os.path.isdir(full):
                continue
            # Need MasterFile with 4-TOC_CALC
            has_mf = any('MasterFile' in f and f.endswith('.xlsx') and '~' not in f
                         for f in os.listdir(full))
            if has_mf:
                method = detect_seq_type(d)
                num = extract_seq_number(d)
                if args.seq and num != args.seq:
                    continue
                if args.method and method != args.method.upper():
                    continue
                all_seqs.append(full)

    print(f"SEQs a analitzar: {len(all_seqs)}")

    results = []
    for seq_path in all_seqs:
        print(f"  {os.path.basename(seq_path)}...", end=" ", flush=True)
        try:
            r = analyze_seq(seq_path, do_plot=args.plot)
            results.append(r)
            print(f"{r['verdict']}  ({r.get('n_injections', '?')} inj, "
                  f"delay={r.get('delay', '?')})")
            if r.get('diagnosis'):
                for d in r['diagnosis']:
                    print(f"    ! {d}")
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({'seq': os.path.basename(seq_path), 'error': str(e)})

    # Summary
    print(f"\n{'='*60}")
    print("RESUM")
    print(f"{'='*60}")
    ok = [r for r in results if r.get('verdict') == 'OK']
    suspect = [r for r in results if r.get('verdict') == 'DELAY_SUSPECT']
    check = [r for r in results if r.get('verdict') in ('CHECK', 'WARNING')]
    errors = [r for r in results if 'error' in r]

    print(f"OK: {len(ok)}")
    print(f"DELAY SUSPECT: {len(suspect)}")
    if suspect:
        for r in suspect:
            print(f"  {r['seq']}: {', '.join(r.get('diagnosis', []))}")
    print(f"CHECK/WARNING: {len(check)}")
    if check:
        for r in check:
            print(f"  {r['seq']}: {', '.join(r.get('diagnosis', []))}")
    print(f"ERRORS: {len(errors)}")
    if errors:
        for r in errors:
            print(f"  {r['seq']}: {r.get('error', '?')}")

    if args.plot:
        print(f"\nPlots a: _results/diag_delay/")

    print("\nFet!")


if __name__ == "__main__":
    main()
