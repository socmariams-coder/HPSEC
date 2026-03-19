"""234_SEQ: TOC+DAD254 continu + delay KHP inici vs final."""
import numpy as np, pandas as pd, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

seq_path = 'C:/Users/Lequia/Desktop/Dades3/234_SEQ'
mf = os.path.join(seq_path, '234_MasterFile.xlsx')

# TOC
toc = pd.read_excel(mf, sheet_name='2-TOC', header=6)
sig_col = date_col = None
for c in toc.columns:
    if 'toc' in str(c).lower() and 'ppb' in str(c).lower(): sig_col = c
    if 'started' in str(c).lower(): date_col = c
y_toc = pd.to_numeric(toc[sig_col], errors='coerce').values
dates_toc = pd.to_datetime(toc[date_col], errors='coerce')
t0 = dates_toc.dropna().iloc[0]
t_toc = (dates_toc - t0).dt.total_seconds().values / 60.0
y_toc_clean = np.nan_to_num(y_toc, nan=0)

# HPLC
hplc = pd.read_excel(mf, sheet_name='1-HPLC-SEQ', header=0)
hplc_date_col = hplc_name_col = None
for c in hplc.columns:
    cl = str(c).lower()
    if 'acquired' in cl or ('injection' in cl and 'date' in cl): hplc_date_col = c
    if 'sample' in cl and 'name' in cl: hplc_name_col = c

injs = []
for _, row in hplc.iterrows():
    name = str(row[hplc_name_col]).strip()
    if pd.isna(name) or name == 'nan': continue
    t_abs = (pd.to_datetime(row[hplc_date_col]) - t0).total_seconds() / 60.0
    injs.append({'name': name, 't': t_abs})

# DAD254 continu
e3d_dir = os.path.join(seq_path, 'Export3d')
e3d_files = sorted(os.listdir(e3d_dir)) if os.path.isdir(e3d_dir) else []
dad_t_all, dad_y_all = [], []
name_counter = {}
for inj in injs:
    nk = ''.join(c for c in inj['name'].lower() if c.isalnum())
    name_counter[nk] = name_counter.get(nk, 0) + 1
    rep = name_counter[nk]
    matched = None
    for f in e3d_files:
        fb = f.lower().replace('.csv', '')
        fb_alnum = ''.join(c for c in fb if c.isalnum())
        if nk in fb_alnum and 'uib' not in fb:
            matched = f
            break
    if not matched: continue
    try:
        fpath = os.path.join(e3d_dir, matched)
        with open(fpath, encoding='utf-16') as fh:
            lines = fh.readlines()
        header = lines[0].strip().split(',')
        idx254 = next((i for i, h in enumerate(header) if '254' in h), None)
        if not idx254: continue
        for line in lines[1:]:
            fields = line.strip().split(',')
            if len(fields) > idx254:
                try:
                    dad_t_all.append(float(fields[0]) + inj['t'])
                    dad_y_all.append(float(fields[idx254]))
                except ValueError: pass
    except Exception: pass

dad_t = np.array(dad_t_all) if dad_t_all else np.array([])
dad_y = np.array(dad_y_all) if dad_y_all else np.array([])

# KHP delay: pic DOC a ~21 min dins injecció per KHP COLUMN
khp_injs = [(i, inj) for i, inj in enumerate(injs) if 'khp' in inj['name'].lower()]
print(f"KHP injeccions: {len(khp_injs)}")

khp_delays = []
for ki, (inj_idx, kinj) in enumerate(khp_injs):
    # Pic DOC de KHP: buscar el maxim TOC dins la finestra d'aquesta injecció
    # Finestra: t_hplc + 15 a t_hplc + 35 (pic HS a ~21 min + delay)
    for delay_guess in range(2, 20):
        t_center = kinj['t'] + 21.0 + delay_guess
        mask = (t_toc >= t_center - 2) & (t_toc <= t_center + 2)
        idx = np.where(mask)[0]
        if len(idx) > 3:
            y_w = y_toc_clean[idx]
            bl = np.percentile(y_w, 20)
            pk = np.argmax(y_w - bl)
            if y_w[pk] - bl > 30:
                t_toc_peak = float(t_toc[idx[pk]])
                delay = t_toc_peak - (kinj['t'] + 21.0)
                position = 'INICI' if kinj['t'] < 500 else 'FINAL'
                khp_delays.append({
                    'pos': position, 'delay': delay,
                    't_hplc': kinj['t'], 't_toc_peak': t_toc_peak,
                    'inj_num': inj_idx + 1
                })
                print(f"  KHP inj {inj_idx+1} ({position}): delay={delay:.2f} min")
                break

# Timeouts
dt_sec = np.diff(t_toc) * 60.0
to_idx = np.where(dt_sec > 30)[0]

# Gràfic
fig, axes = plt.subplots(3, 1, figsize=(18, 12))

# Complet
ax = axes[0]
ax.plot(t_toc, y_toc_clean, 'b-', lw=0.3, alpha=0.7)
ax.set_ylabel('TOC ppb', color='blue', fontsize=9)
if len(dad_t) > 0:
    ax2 = ax.twinx()
    ax2.plot(dad_t, dad_y, 'g-', lw=0.2, alpha=0.4)
    ax2.set_ylabel('A254 mAU', color='green', fontsize=9)
for inj in injs:
    is_khp = 'khp' in inj['name'].lower()
    ax.axvline(inj['t'], c='#E74C3C' if is_khp else '#DDD',
               lw=1 if is_khp else 0.2, ls=':', alpha=0.6 if is_khp else 0.15)
for ti in to_idx:
    ax.axvline(t_toc[ti], c='red', lw=0.3, ls='--', alpha=0.15)
for d in khp_delays:
    yi = y_toc_clean[np.searchsorted(t_toc, d['t_toc_peak'])]
    ax.annotate(f"{d['pos']} d={d['delay']:.1f}",
                (d['t_toc_peak'], yi), fontsize=8, color='#E74C3C',
                fontweight='bold', xytext=(5, 10), textcoords='offset points')
ax.set_title(f"234_SEQ COMPLET - {len(to_idx)} timeouts", fontsize=10)
ax.set_xlim(0, t_toc[-1])
ax.grid(True, alpha=0.1)

# Zoom inici
ax = axes[1]
t_lo, t_hi = 0, 400
m = (t_toc >= t_lo) & (t_toc <= t_hi)
ax.plot(t_toc[m], y_toc_clean[m], 'b-', lw=0.5, alpha=0.8)
if len(dad_t) > 0:
    ax2 = ax.twinx()
    md = (dad_t >= t_lo) & (dad_t <= t_hi)
    if md.any(): ax2.plot(dad_t[md], dad_y[md], 'g-', lw=0.3, alpha=0.5)
    ax2.set_ylabel('A254', color='green', fontsize=8)
for inj in injs:
    if t_lo <= inj['t'] <= t_hi:
        is_khp = 'khp' in inj['name'].lower()
        ax.axvline(inj['t'], c='#E74C3C' if is_khp else '#999', lw=0.5, ls=':', alpha=0.4)
        ax.annotate(inj['name'][:10], (inj['t'] + 1, 200), fontsize=5, rotation=90, va='top')
for ti in to_idx:
    if t_lo <= t_toc[ti] <= t_hi:
        ax.axvline(t_toc[ti], c='red', lw=0.5, ls='--', alpha=0.3)
ax.set_title('ZOOM INICI', fontsize=10)
ax.set_xlim(t_lo, t_hi)
ax.grid(True, alpha=0.1)

# Zoom final
t_lo2 = max(0, khp_injs[-1][1]['t'] - 200) if khp_injs else 1800
t_hi2 = t_toc[-1]
ax = axes[2]
m = (t_toc >= t_lo2) & (t_toc <= t_hi2)
ax.plot(t_toc[m], y_toc_clean[m], 'b-', lw=0.5, alpha=0.8)
if len(dad_t) > 0:
    ax2 = ax.twinx()
    md = (dad_t >= t_lo2) & (dad_t <= t_hi2)
    if md.any(): ax2.plot(dad_t[md], dad_y[md], 'g-', lw=0.3, alpha=0.5)
    ax2.set_ylabel('A254', color='green', fontsize=8)
for inj in injs:
    if t_lo2 <= inj['t'] <= t_hi2:
        is_khp = 'khp' in inj['name'].lower()
        ax.axvline(inj['t'], c='#E74C3C' if is_khp else '#999', lw=0.5, ls=':', alpha=0.4)
        ax.annotate(inj['name'][:10], (inj['t'] + 1, 200), fontsize=5, rotation=90, va='top')
for ti in to_idx:
    if t_lo2 <= t_toc[ti] <= t_hi2:
        ax.axvline(t_toc[ti], c='red', lw=0.5, ls='--', alpha=0.3)
ax.set_title('ZOOM FINAL', fontsize=10)
ax.set_xlim(t_lo2, t_hi2)
ax.set_xlabel('min', fontsize=9)
ax.grid(True, alpha=0.1)

inici_d = [d['delay'] for d in khp_delays if d['pos'] == 'INICI']
final_d = [d['delay'] for d in khp_delays if d['pos'] == 'FINAL']
d_ini = np.mean(inici_d) if inici_d else 0
d_fin = np.mean(final_d) if final_d else 0
drift = d_fin - d_ini

plt.suptitle(
    f"234_SEQ: KHP inici delay={d_ini:.1f} | KHP final delay={d_fin:.1f} | drift={drift:.1f} min",
    fontsize=12, fontweight='bold')
plt.tight_layout()
out = 'C:/Users/Lequia/Desktop/HPSEC/_234_toc_dad254_khp.png'
plt.savefig(out, dpi=130, bbox_inches='tight')
print(f"Guardat: {out}")
