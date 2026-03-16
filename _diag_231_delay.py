"""Auto-detect delay for 231_SEQ_BP."""
import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d

mf = 'C:/Users/Lequia/Desktop/Dades3/231_SEQ_BP/231_MasterFile.xlsx'

# 1. Load TOC
df_toc = pd.read_excel(mf, sheet_name='2-TOC', header=6)
toc_vals = pd.to_numeric(df_toc['TOC(ppb)'], errors='coerce').values
toc_dates = pd.to_datetime(df_toc['Date Started'], errors='coerce')
t0_toc = toc_dates.dropna().iloc[0]

# Build time from row index (4 sec cadence = 0.0667 min)
dt_sec = 4.0
dt_min = dt_sec / 60.0
n_pts = len(toc_vals)
toc_t = np.arange(n_pts) * dt_min  # relative time from first TOC measurement
print(f"TOC: {n_pts} pts, {toc_t[-1]:.1f} min total, dt={dt_min:.4f} min")

# Also compute absolute offset: TOC t=0 is the first measurement
# HPLC times are absolute, so we need the offset between TOC start and HPLC start

# 2. HPLC times (absolute, converted to minutes from TOC t=0)
df_hplc = pd.read_excel(mf, sheet_name='1-HPLC-SEQ')
name_col = [c for c in df_hplc.columns if 'sample' in str(c).lower() and 'name' in str(c).lower()][0]
date_col = [c for c in df_hplc.columns if 'acquired' in str(c).lower()][0]

hplc_injs = []
for _, row in df_hplc.iterrows():
    try:
        dt = pd.to_datetime(row[date_col])
        hplc_injs.append({
            'name': str(row[name_col]),
            't_abs': dt,
            't_min_from_toc0': (dt - t0_toc).total_seconds() / 60.0,
            'line': int(row.get('Line#', 0)),
        })
    except:
        pass

hplc_times = np.array([inj['t_min_from_toc0'] for inj in hplc_injs])
inj_spacing = np.median(np.diff(hplc_times))
print(f"HPLC: {len(hplc_injs)} inj, spacing={inj_spacing:.1f} min")
print(f"  First: {hplc_injs[0]['name']} at t={hplc_times[0]:.1f} min from TOC start")
print(f"  Last: {hplc_injs[-1]['name']} at t={hplc_times[-1]:.1f} min from TOC start")

# 3. Find peaks in TOC
y = np.nan_to_num(toc_vals, nan=np.nanmedian(toc_vals[~np.isnan(toc_vals)]))
y_smooth = uniform_filter1d(y, size=5)
baseline = np.percentile(y_smooth, 15)
min_dist = int(8.0 / dt_min)
peaks, props = find_peaks(y_smooth, height=baseline + 15, distance=min_dist, prominence=10)
peak_times = toc_t[peaks]
peak_vals = y[peaks]
print(f"\nPics: {len(peaks)} (vs {len(hplc_injs)} injeccions)")

for i, (pt, pv) in enumerate(zip(peak_times, peak_vals)):
    print(f"  P{i+1:2d}: t={pt:7.1f} min  val={pv:.0f} ppb")

# 4. Auto-detect delay
# The HPLC injection at time t_hplc produces a peak in TOC at:
#   t_peak_in_toc = t_hplc + delay_net
# where delay_net includes the physical delay between injection and TOC detection
#
# We search for the delay that maximizes peak↔injection matches
print(f"\n=== SEARCHING OPTIMAL DELAY ===")

best_delay = 0
best_matches = 0

for delay_try in np.arange(-20, 40, 0.1):
    expected = hplc_times + delay_try
    matches = 0
    for exp_t in expected:
        diffs = np.abs(peak_times - exp_t)
        if len(diffs) > 0 and np.min(diffs) < 1.5:
            matches += 1
    if matches > best_matches:
        best_matches = matches
        best_delay = delay_try

print(f"Delay optim: {best_delay:.1f} min")
print(f"Matches: {best_matches}/{len(hplc_injs)} ({best_matches/len(hplc_injs)*100:.0f}%)")
print(f"Delay MasterFile: 2.13 min")
print(f"Diferencia: {best_delay - 2.13:+.1f} min")

# 5. Show matching
print(f"\n=== MATCHING (delay={best_delay:.1f}) ===")
for j, inj in enumerate(hplc_injs):
    t_exp = inj['t_min_from_toc0'] + best_delay
    diffs = np.abs(peak_times - t_exp)
    if len(diffs) > 0:
        best_pk = np.argmin(diffs)
        dist = diffs[best_pk]
        status = "OK" if dist < 1.5 else "MISS"
        print(f"  Inj {j+1:2d} {inj['name']:15s} t_hplc={inj['t_min_from_toc0']:6.1f} "
              f"t_exp={t_exp:6.1f}  P{best_pk+1:2d} t={peak_times[best_pk]:6.1f}  "
              f"dist={dist:.1f}min  {status}")

# 6. Plot
fig, axes = plt.subplots(3, 1, figsize=(18, 14))
fig.suptitle(f"231_SEQ_BP — Auto-detect delay\n"
             f"MasterFile: 2.13 min | Optim: {best_delay:.1f} min | "
             f"Matches: {best_matches}/{len(hplc_injs)}",
             fontsize=13, fontweight="bold")

# A) Full TOC with CURRENT delay
ax = axes[0]
ax.plot(toc_t, y, 'b-', lw=0.3, alpha=0.4)
ax.plot(toc_t, y_smooth, 'b-', lw=0.5, alpha=0.8)
ax.plot(peak_times, peak_vals, 'gv', ms=5, zorder=5)
for j, inj in enumerate(hplc_injs):
    t_exp = inj['t_min_from_toc0'] + 2.13  # current delay
    is_khp = 'khp' in inj['name'].lower()
    ax.axvline(t_exp, color='red' if is_khp else 'orange', ls='--', lw=0.3, alpha=0.4)
ax.set_title(f"Delay ACTUAL (2.13 min) — linies taronges NO coincideixen amb pics")
ax.set_ylabel('ppb')

# B) Full TOC with OPTIMAL delay
ax = axes[1]
ax.plot(toc_t, y, 'b-', lw=0.3, alpha=0.4)
ax.plot(toc_t, y_smooth, 'b-', lw=0.5, alpha=0.8)
ax.plot(peak_times, peak_vals, 'gv', ms=5, zorder=5)
for j, inj in enumerate(hplc_injs):
    t_exp = inj['t_min_from_toc0'] + best_delay
    is_khp = 'khp' in inj['name'].lower()
    ax.axvline(t_exp, color='red' if is_khp else '#27AE60', ls='--', lw=0.5, alpha=0.5)
ax.set_title(f"Delay OPTIM ({best_delay:.1f} min) — linies verdes coincideixen amb pics")
ax.set_ylabel('ppb')

# C) Zoom
ax = axes[2]
zoom_end = min(150, toc_t[-1])
mask = toc_t < zoom_end
ax.plot(toc_t[mask], y[mask], 'b-', lw=0.6, alpha=0.7)
pk_mask = peak_times < zoom_end
ax.plot(peak_times[pk_mask], peak_vals[pk_mask], 'gv', ms=8, zorder=5)
for i, (pt, pv) in enumerate(zip(peak_times, peak_vals)):
    if pt < zoom_end:
        ax.annotate(f'P{i+1}', (pt, pv+8), fontsize=7, ha='center', color='green', fontweight='bold')

for j, inj in enumerate(hplc_injs):
    t_exp = inj['t_min_from_toc0'] + best_delay
    if t_exp < zoom_end:
        is_khp = 'khp' in inj['name'].lower()
        color = 'red' if is_khp else '#27AE60'
        ax.axvline(t_exp, color=color, ls='--', lw=0.6, alpha=0.6)
        ax.annotate(f"I{j+1}:{inj['name'][:6]}", (t_exp, 12),
                   fontsize=5, rotation=90, va='bottom', color=color)
ax.set_title(f"Zoom primers {zoom_end:.0f} min")
ax.set_xlabel('min (des de inici TOC)'); ax.set_ylabel('ppb')

fig.tight_layout(rect=[0, 0, 1, 0.93])
out = Path('_results/diag_delay')
out.mkdir(parents=True, exist_ok=True)
p = out / '231_delay_autodetect.png'
fig.savefig(str(p), dpi=150, bbox_inches='tight')
print(f"\nPlot: {p}")
