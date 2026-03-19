"""Comparativa BP vs COLUMN — tangent vs trapezoid.
Genera CSV traçable + gràfic amb TOTES les SEQs processades."""

import json, os, numpy as np
from hpsec_import import import_sequence
from hpsec_core import find_peak_boundaries
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

data_dir = 'C:/Users/Lequia/Desktop/Dades3'

# Totes les SEQs BP (no CAL)
bp_seqs = sorted([d for d in os.listdir(data_dir)
                  if 'BP' in d.upper() and '_CAL' not in d.upper()
                  and os.path.isdir(os.path.join(data_dir, d))])

print(f"SEQs BP a processar: {len(bp_seqs)}")
for s in bp_seqs:
    print(f"  {s}")

# Pas 1: BP — reimportar i calcular tangent + trapezoid
bp_by_name = {}
for seq in bp_seqs:
    path = os.path.join(data_dir, seq)
    print(f"Important {seq}...", end=" ", flush=True)
    try:
        result = import_sequence(path)
    except Exception as e:
        print(f"ERROR: {e}")
        continue
    samples = result.get('samples', {})
    n = 0
    for sn, sd in samples.items():
        if any(x in sn.lower() for x in ['mq', 'khp', 'naoh', 'blanc', 'blnc']):
            continue
        for rk, rd in sd.get('replicas', {}).items():
            d = rd.get('direct', {})
            t = np.array(d.get('t', []))
            y_net = np.array(d.get('y_net', []))
            if len(t) < 10:
                continue
            vol = 100
            inj = rd.get('injection_info', {})
            if inj and inj.get('inj_volume'):
                vol = inj['inj_volume']

            area_trap = float(np.trapezoid(np.maximum(y_net, 0), t))
            pk = int(np.argmax(y_net))
            try:
                li, ri = find_peak_boundaries(t, y_net, pk)
                area_tang = float(np.trapezoid(
                    np.maximum(y_net[li:ri+1], 0), t[li:ri+1]))
            except Exception:
                area_tang = area_trap

            if sn not in bp_by_name:
                bp_by_name[sn] = {'tang': [], 'trap': [], 'vol': vol, 'seq': seq}
            bp_by_name[sn]['tang'].append(area_tang)
            bp_by_name[sn]['trap'].append(area_trap)
            n += 1
    print(f"{n} mostres")

# Pas 2: COLUMN — llegir àrees dels JSONs
col_by_name = {}
for seq in sorted(os.listdir(data_dir)):
    if 'BP' in seq.upper() or '_CAL' in seq.upper():
        continue
    ana = os.path.join(data_dir, seq, 'CHECK', 'data', 'analysis_result.json')
    if not os.path.exists(ana):
        continue
    try:
        with open(ana, 'r', encoding='utf-8') as f:
            a = json.load(f)
    except Exception:
        continue
    for s in a.get('samples', []):
        name = s.get('name', '')
        if any(x in name.lower() for x in ['mq', 'khp', 'naoh', 'blanc', 'blnc']):
            continue
        areas = s.get('areas', {}).get('DOC', {})
        area = areas.get('total', 0)
        vol = s.get('inj_volume') or 400
        if area > 0:
            if name not in col_by_name:
                col_by_name[name] = {'area': [], 'vol': vol, 'seq': seq}
            col_by_name[name]['area'].append(area)

# Pas 3: Parells
pairs = sorted(set(bp_by_name.keys()) & set(col_by_name.keys()))
print(f"\nParells BP+COL: {len(pairs)}")

RF = 759.4
IC = 0.1

# CSV
csv_path = 'C:/Users/Lequia/Desktop/HPSEC/_comparativa_bp_col.csv'
with open(csv_path, 'w', encoding='utf-8') as f:
    f.write("Mostra,SEQ_BP,SEQ_COL,vol_BP,vol_COL,"
            "area_BP_tangent,area_BP_trapezoid,area_COL,"
            "ppm_tang,ppm_trap,ppm_COL,"
            "ratio_tang,ratio_trap\n")

    ratios_tang = []
    ratios_trap = []
    ppm_t_list = []
    ppm_r_list = []
    ppm_c_list = []

    for name in pairs:
        bp = bp_by_name[name]
        col = col_by_name[name]

        a_tang = np.mean(bp['tang'])
        a_trap = np.mean(bp['trap'])
        a_col = np.mean(col['area'])
        v_bp = bp['vol']
        v_col = col['vol']

        ppm_tang = (a_tang - IC) * 1000 / (RF * v_bp)
        ppm_trap = (a_trap - IC) * 1000 / (RF * v_bp)
        ppm_col = (a_col - IC) * 1000 / (RF * v_col)

        r_tang = ppm_tang / ppm_col if ppm_col > 0 else 0
        r_trap = ppm_trap / ppm_col if ppm_col > 0 else 0

        ratios_tang.append(r_tang)
        ratios_trap.append(r_trap)
        ppm_t_list.append(ppm_tang)
        ppm_r_list.append(ppm_trap)
        ppm_c_list.append(ppm_col)

        f.write(f"{name},{bp['seq']},{col['seq']},{v_bp},{v_col},"
                f"{a_tang:.1f},{a_trap:.1f},{a_col:.1f},"
                f"{ppm_tang:.3f},{ppm_trap:.3f},{ppm_col:.3f},"
                f"{r_tang:.3f},{r_trap:.3f}\n")

rt = np.array(ratios_tang)
rr = np.array(ratios_trap)
ppm_t = np.array(ppm_t_list)
ppm_r = np.array(ppm_r_list)
ppm_c = np.array(ppm_c_list)

print(f"CSV: {csv_path}")
print(f"\nTANGENT:   mediana={np.median(rt):.3f} mean={np.mean(rt):.3f} "
      f"+-10%={((rt>0.9)&(rt<1.1)).sum()}/{len(rt)} "
      f"+-20%={((rt>0.8)&(rt<1.2)).sum()}/{len(rt)}")
print(f"TRAPEZOID: mediana={np.median(rr):.3f} mean={np.mean(rr):.3f} "
      f"+-10%={((rr>0.9)&(rr<1.1)).sum()}/{len(rr)} "
      f"+-20%={((rr>0.8)&(rr<1.2)).sum()}/{len(rr)}")

# Gràfic
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
mx = max(ppm_c.max(), ppm_t.max(), ppm_r.max()) * 1.1

# 1. Scatter tangent
ax = axes[0][0]
ax.scatter(ppm_c, ppm_t, c='#27AE60', s=20, alpha=0.6)
ax.plot([0, mx], [0, mx], 'k-', lw=0.8, alpha=0.4)
ax.fill_between([0, mx], [0, mx*0.8], [0, mx*1.2], alpha=0.04, color='green')
sl, ic_r, r, _, _ = stats.linregress(ppm_c, ppm_t)
ax.plot([0, mx], [ic_r, sl*mx+ic_r], '-', c='#27AE60', lw=1.5, alpha=0.7)
ax.set_title(f'BP TANGENT vs COLUMN\nslope={sl:.3f} R2={r**2:.3f} med={np.median(rt):.3f}', fontsize=10)
ax.set_xlabel('ppm COLUMN'); ax.set_ylabel('ppm BP tangent')
ax.set_xlim(0, mx); ax.set_ylim(0, mx); ax.set_aspect('equal'); ax.grid(True, alpha=0.15)

# 2. Scatter trapezoid
ax = axes[0][1]
ax.scatter(ppm_c, ppm_r, c='#8E44AD', s=20, alpha=0.6)
ax.plot([0, mx], [0, mx], 'k-', lw=0.8, alpha=0.4)
ax.fill_between([0, mx], [0, mx*0.8], [0, mx*1.2], alpha=0.04, color='green')
sl2, ic2, r2, _, _ = stats.linregress(ppm_c, ppm_r)
ax.plot([0, mx], [ic2, sl2*mx+ic2], '-', c='#8E44AD', lw=1.5, alpha=0.7)
ax.set_title(f'BP TRAPEZOID vs COLUMN\nslope={sl2:.3f} R2={r2**2:.3f} med={np.median(rr):.3f}', fontsize=10)
ax.set_xlabel('ppm COLUMN'); ax.set_ylabel('ppm BP trapezoid')
ax.set_xlim(0, mx); ax.set_ylim(0, mx); ax.set_aspect('equal'); ax.grid(True, alpha=0.15)

# 3. Histograma ratios
ax = axes[1][0]
bins = np.linspace(0.3, 2.0, 40)
ax.hist(rt, bins=bins, alpha=0.5, color='#27AE60', label=f'Tangent med={np.median(rt):.3f}')
ax.hist(rr, bins=bins, alpha=0.5, color='#8E44AD', label=f'Trapezoid med={np.median(rr):.3f}')
ax.axvline(1.0, c='k', lw=1, alpha=0.5)
ax.axvspan(0.8, 1.2, alpha=0.05, color='green')
ax.set_xlabel('Ratio BP/COLUMN'); ax.set_ylabel('Freq')
ax.set_title('Distribucio ratios', fontsize=10)
ax.legend(fontsize=9); ax.grid(True, alpha=0.15)

# 4. Ratio per mostra
ax = axes[1][1]
order = np.argsort(rt)
idx = np.arange(len(pairs))
ax.scatter(idx, rt[order], c='#27AE60', s=15, alpha=0.6, label='Tangent')
ax.scatter(idx, rr[order], c='#8E44AD', s=15, alpha=0.6, label='Trapezoid')
ax.axhline(1.0, c='k', lw=0.8, alpha=0.4)
ax.axhspan(0.8, 1.2, alpha=0.05, color='green')
ax.set_xlabel('Mostra (ordenada)'); ax.set_ylabel('Ratio BP/COL')
ax.set_title(f'Ratio per mostra (n={len(pairs)})', fontsize=10)
ax.legend(fontsize=9); ax.grid(True, alpha=0.15)

plt.suptitle(
    f'Comparativa BP vs COLUMN — {len(pairs)} parells, {len(bp_seqs)} SEQs BP\n'
    f'Tangent: mediana={np.median(rt):.3f} | Trapezoid: mediana={np.median(rr):.3f} | '
    f'RF={RF} IC={IC}',
    fontsize=12, fontweight='bold')
plt.tight_layout()
out = 'C:/Users/Lequia/Desktop/HPSEC/_comparativa_final.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Grafic: {out}")
