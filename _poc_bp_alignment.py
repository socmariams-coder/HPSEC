"""
Proof of concept: BP peak alignment by DAD254 → TOC matching.
==============================================================
Per cada injecció:
1. Trobar màxim DAD 254 (temps relatiu dins el run)
2. Buscar màxim TOC dins finestra de cerca
3. Alinear: centrar la finestra TOC al pic
4. Integrar X min abans + X min després del màxim
5. Comparar amb integració antiga (delay fix)

Usage:
    python _poc_bp_alignment.py 295_SEQ_BP
    python _poc_bp_alignment.py 231
"""
import sys, os
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from pathlib import Path

from hpsec_config import get_data_folders
from hpsec_consolidate import extract_seq_number, detect_seq_type

FLUSH_TIME_MIN = 3.637


def find_seq_path(arg):
    if os.path.isdir(arg):
        return arg
    for folder in get_data_folders():
        full = os.path.join(folder, arg)
        if os.path.isdir(full):
            return full
        try:
            num = int(arg)
            for d in os.listdir(folder):
                if extract_seq_number(d) == num:
                    return os.path.join(folder, d)
        except ValueError:
            pass
    return None


def read_export3d_254(filepath):
    try:
        with open(filepath, encoding='utf-16') as fh:
            lines = fh.readlines()
        header = lines[0].strip().split(',')
        idx254 = next((i for i, h in enumerate(header) if '254' in h), None)
        if idx254 is None: return None, None
        times, vals = [], []
        for line in lines[1:]:
            fields = line.strip().split(',')
            if len(fields) > idx254:
                try:
                    times.append(float(fields[0]))
                    vals.append(float(fields[idx254]))
                except ValueError: pass
        return np.array(times), np.array(vals)
    except:
        return None, None


def main():
    if len(sys.argv) < 2:
        print("Usage: python _poc_bp_alignment.py <SEQ>")
        sys.exit(1)

    seq_path = find_seq_path(sys.argv[1])
    if not seq_path:
        print(f"ERROR: '{sys.argv[1]}' no trobat"); sys.exit(1)

    seq_name = os.path.basename(seq_path)
    method = detect_seq_type(seq_name)
    print(f"=== {seq_name} ({method}) ===\n")

    mfs = [f for f in os.listdir(seq_path)
           if 'MasterFile' in f and f.endswith('.xlsx')
           and '~' not in f and 'pre_vol' not in f and 'backup' not in f]
    if not mfs:
        mfs = [f for f in os.listdir(seq_path)
               if 'MasterFile' in f and f.endswith('.xlsx')
               and '~' not in f and 'pre_vol' not in f]
    mf_path = os.path.join(seq_path, mfs[0])

    # === TOC continu ===
    df_toc = pd.read_excel(mf_path, sheet_name='2-TOC', header=6)
    toc_col = [c for c in df_toc.columns if 'toc' in str(c).lower()][0]
    date_col_toc = [c for c in df_toc.columns if 'started' in str(c).lower()]
    y_toc_raw = pd.to_numeric(df_toc[toc_col], errors='coerce').values
    dt = 4.0 / 60.0
    t_toc = np.arange(len(y_toc_raw)) * dt
    y_toc = np.nan_to_num(y_toc_raw, nan=np.nanmedian(y_toc_raw[~np.isnan(y_toc_raw)]))

    t0_toc = None
    if date_col_toc:
        dates = pd.to_datetime(df_toc[date_col_toc[0]], errors='coerce')
        valid = dates.dropna()
        if len(valid) > 0: t0_toc = valid.iloc[0]

    # === HPLC injections ===
    df_hplc = pd.read_excel(mf_path, sheet_name='1-HPLC-SEQ')
    name_col = [c for c in df_hplc.columns
                if 'sample' in str(c).lower() and 'name' in str(c).lower()][0]
    hplc_date_col = [c for c in df_hplc.columns if 'acquired' in str(c).lower()][0]

    injs = []
    for _, row in df_hplc.iterrows():
        try:
            name = str(row[name_col]).strip()
            if pd.isna(name) or name == 'nan': continue
            dt_val = pd.to_datetime(row[hplc_date_col])
            t_min = (dt_val - t0_toc).total_seconds() / 60.0 if t0_toc else 0
            injs.append({'name': name, 't_hplc': t_min,
                         'is_khp': 'khp' in name.lower(),
                         'is_control': any(x in name.lower()
                                           for x in ['mq', 'naoh', 'blanc', 'blnc'])})
        except: pass

    # === Current delay (for old method comparison) ===
    current_delay = FLUSH_TIME_MIN
    df_info = pd.read_excel(mf_path, sheet_name='0-INFO', header=None)
    hora_hplc = hora_toc = None
    for _, row in df_info.iterrows():
        key = str(row.iloc[0]).strip().lower() if pd.notna(row.iloc[0]) else ''
        val = row.iloc[1] if len(row) > 1 else None
        if 'hora hplc' in key: hora_hplc = val
        elif 'hora toc' in key: hora_toc = val
        elif 'net delay' in key and 'suite' not in key:
            try:
                v = float(val)
                if not np.isnan(v): current_delay = v
            except: pass
    if hora_hplc and hora_toc:
        try:
            def to_min(t):
                if hasattr(t, 'hour'):
                    return t.hour * 60 + t.minute + t.second / 60
                return sum(int(p)*m for p, m in zip(str(t).split(':'), [60, 1]))
            calc = FLUSH_TIME_MIN - (to_min(hora_hplc) - to_min(hora_toc))
            if current_delay == FLUSH_TIME_MIN: current_delay = calc
        except: pass

    # === Export3D files ===
    e3d = os.path.join(seq_path, 'Export3d')
    e3d_files = sorted(os.listdir(e3d)) if os.path.isdir(e3d) else []

    # === Parameters ===
    cadence = np.median(np.diff([i['t_hplc'] for i in injs])) if len(injs) >= 2 else 11.0
    SEARCH_WINDOW = cadence * 0.8  # mai buscar mes enlla de la seguent injeccio
    INTEG_BEFORE = 5.0     # min abans del maxim DOC
    INTEG_AFTER = 5.0      # min despres del maxim DOC

    # === Process each injection ===
    print(f"Delay MasterFile: {current_delay:.2f} min")
    print(f"Finestra integracio: -{INTEG_BEFORE} / +{INTEG_AFTER} min del maxim\n")

    header = (f"{'Inj':>3} {'Nom':>12} {'t_HPLC':>7} {'DAD_pk':>7} {'TOC_pk':>7} "
              f"{'delay':>6} {'area_NEW':>9} {'area_OLD':>9} {'diff%':>6} {'status':>8}")
    print(header)
    print('-' * len(header))

    name_counter = {}
    results = []

    for j, inj in enumerate(injs):
        nk = ''.join(c for c in inj['name'].lower() if c.isalnum())
        name_counter[nk] = name_counter.get(nk, 0) + 1
        rep = name_counter[nk]

        # Find Export3D
        matched = None
        for f in e3d_files:
            fb = f.lower().replace('.csv', '')
            fb_alnum = ''.join(c for c in fb if c.isalnum())
            if nk in fb_alnum and (f'_r{rep}' in fb or f'_{rep}' in fb):
                if 'uib' not in fb:
                    matched = f; break

        # === NEW METHOD: align by DAD peak ===
        t_dad_max_abs = None
        t_toc_max_abs = None
        area_new = None
        delay_real = None

        if matched:
            t_rel, y_254 = read_export3d_254(os.path.join(e3d, matched))
            if t_rel is not None and len(t_rel) > 5:
                # DAD peak (relative to injection)
                pk254 = np.argmax(y_254)
                t_dad_max_rel = t_rel[pk254]
                t_dad_max_abs = inj['t_hplc'] + t_dad_max_rel

                # Search TOC peak near DAD position
                search_start = t_dad_max_abs
                search_end = t_dad_max_abs + SEARCH_WINDOW
                mask = (t_toc >= search_start) & (t_toc <= search_end)

                if mask.any():
                    y_w = y_toc[mask]
                    t_w = t_toc[mask]
                    bl = np.percentile(y_w, 20)
                    pk_idx = np.argmax(y_w - bl)
                    t_toc_max_abs = t_w[pk_idx]
                    delay_real = t_toc_max_abs - t_dad_max_abs

                    # Integrate: INTEG_BEFORE before peak, INTEG_AFTER after peak
                    integ_start = t_toc_max_abs - INTEG_BEFORE
                    integ_end = t_toc_max_abs + INTEG_AFTER
                    integ_mask = (t_toc >= integ_start) & (t_toc <= integ_end)

                    if integ_mask.any():
                        y_integ = y_toc[integ_mask]
                        t_integ = t_toc[integ_mask]
                        bl_integ = np.percentile(y_integ, 20)
                        y_net = np.maximum(y_integ - bl_integ, 0)
                        area_new = float(np.trapz(y_net, t_integ))

        # === OLD METHOD: fixed delay ===
        area_old = None
        cadence = np.median(np.diff([i['t_hplc'] for i in injs])) if len(injs) >= 2 else 11.0
        t_center_old = inj['t_hplc'] + current_delay + 1.0  # +1 for retention estimate
        old_start = t_center_old - cadence * 0.4
        old_end = t_center_old + cadence * 0.6
        old_mask = (t_toc >= old_start) & (t_toc <= old_end)

        if old_mask.any():
            y_old = y_toc[old_mask]
            t_old = t_toc[old_mask]
            bl_old = np.percentile(y_old, 20)
            y_net_old = np.maximum(y_old - bl_old, 0)
            area_old = float(np.trapz(y_net_old, t_old))

        # === Compare ===
        diff_pct = None
        if area_new and area_old and area_old > 0:
            diff_pct = (area_new - area_old) / area_old * 100

        status = ''
        if inj['is_control']:
            status = 'CTRL'
        elif area_new is None:
            status = 'NO_DAD'
        elif delay_real is not None and delay_real < 0:
            status = 'NEG_DLY'
        elif diff_pct is not None and abs(diff_pct) > 20:
            status = 'DIFF!'
        else:
            status = 'OK'

        # Format
        td = f"{t_dad_max_abs:.1f}" if t_dad_max_abs else '-'
        tt = f"{t_toc_max_abs:.1f}" if t_toc_max_abs else '-'
        dr = f"{delay_real:.1f}" if delay_real is not None else '-'
        an = f"{area_new:.0f}" if area_new else '-'
        ao = f"{area_old:.0f}" if area_old else '-'
        dp = f"{diff_pct:+.0f}%" if diff_pct is not None else '-'

        print(f"{j+1:>3} {inj['name']:>12} {inj['t_hplc']:>7.1f} {td:>7} {tt:>7} "
              f"{dr:>6} {an:>9} {ao:>9} {dp:>6} {status:>8}")

        results.append({
            'inj': j+1, 'name': inj['name'], 'is_control': inj['is_control'],
            'is_khp': inj['is_khp'], 'delay': delay_real,
            'area_new': area_new, 'area_old': area_old, 'diff_pct': diff_pct,
        })

    # === Summary ===
    samples = [r for r in results if not r['is_control'] and r['area_new'] and r['area_old']]

    if samples:
        diffs = [r['diff_pct'] for r in samples if r['diff_pct'] is not None]
        areas_new = [r['area_new'] for r in samples]
        areas_old = [r['area_old'] for r in samples]
        delays = [r['delay'] for r in samples if r['delay'] is not None]

        print(f"\n{'='*60}")
        print(f"RESUM: {len(samples)} mostres comparades")
        print(f"{'='*60}")
        print(f"Delay real: mediana={np.median(delays):.1f} rang={min(delays):.1f}-{max(delays):.1f}")
        print(f"Area NEW:   mediana={np.median(areas_new):.0f} mean={np.mean(areas_new):.0f}")
        print(f"Area OLD:   mediana={np.median(areas_old):.0f} mean={np.mean(areas_old):.0f}")
        print(f"Diferencia: mediana={np.median(diffs):.0f}% mean={np.mean(diffs):.0f}%")
        print(f"  |diff| > 20%: {sum(1 for d in diffs if abs(d) > 20)}/{len(diffs)}")
        print(f"  |diff| > 50%: {sum(1 for d in diffs if abs(d) > 50)}/{len(diffs)}")

        # KHP
        khps = [r for r in results if r['is_khp'] and r['area_new']]
        if khps:
            print(f"\nKHP:")
            for k in khps:
                rf_new = k['area_new'] * 1000 / (2.0 * 100) if k['area_new'] else 0  # assume 2ppm 100uL
                rf_old = k['area_old'] * 1000 / (2.0 * 100) if k['area_old'] else 0
                print(f"  {k['name']}: area_new={k['area_new']:.0f} "
                      f"(RF={rf_new:.0f}) vs old={k['area_old']:.0f} "
                      f"(RF={rf_old:.0f})")

    # Plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"{seq_name} — Comparacio area NEW (alineat) vs OLD (delay fix)",
                     fontsize=12, fontweight='bold')

        if samples:
            an = np.array([r['area_new'] for r in samples])
            ao = np.array([r['area_old'] for r in samples])
            dd = np.array([r['diff_pct'] for r in samples if r['diff_pct'] is not None])

            # Scatter new vs old
            ax = axes[0]
            ax.scatter(ao, an, c='#228be6', s=20, alpha=0.7)
            mx = max(ao.max(), an.max()) * 1.1
            ax.plot([0, mx], [0, mx], 'k--', alpha=0.3)
            ax.set_xlabel('Area OLD (delay fix)')
            ax.set_ylabel('Area NEW (alineat)')
            if len(ao) >= 3:
                z = np.polyfit(ao, an, 1)
                r2 = np.corrcoef(ao, an)[0, 1]**2
                ax.set_title(f'Scatter (slope={z[0]:.2f}, R2={r2:.3f})')

            # Histogram diff
            ax = axes[1]
            ax.hist(dd, bins=15, color='#228be6', alpha=0.7, edgecolor='white')
            ax.axvline(0, color='k', ls='--', alpha=0.3)
            ax.axvline(np.median(dd), color='red', ls='-', alpha=0.6,
                       label=f'med={np.median(dd):.0f}%')
            ax.set_xlabel('Diferencia %')
            ax.set_title('Distribucio diferencies')
            ax.legend()

            # Delay per injection
            ax = axes[2]
            valid_d = [r for r in results if r['delay'] is not None and not r['is_control']]
            if valid_d:
                ax.scatter([r['inj'] for r in valid_d],
                          [r['delay'] for r in valid_d],
                          c=['red' if r['is_khp'] else '#228be6' for r in valid_d],
                          s=20, alpha=0.7)
                ax.set_xlabel('Injeccio #')
                ax.set_ylabel('Delay real (min)')
                ax.set_title('Drift delay')
                ax.axhline(current_delay, color='orange', ls='--',
                           label=f'MF delay={current_delay:.1f}')
                ax.legend(fontsize=7)

        for a in axes:
            a.tick_params(labelsize=7)
            a.spines['top'].set_visible(False)
            a.spines['right'].set_visible(False)

        fig.tight_layout()
        out = Path('_results/poc_alignment')
        out.mkdir(parents=True, exist_ok=True)
        p = out / f'{seq_name}.png'
        fig.savefig(str(p), dpi=150, bbox_inches='tight')
        print(f"\nPlot: {p}")
    except ImportError:
        pass

    print("\nFet!")


if __name__ == "__main__":
    main()
