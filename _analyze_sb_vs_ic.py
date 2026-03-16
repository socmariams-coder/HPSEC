"""Analyze correlation between SB peak height and IC(ppb) across sequences."""
import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import pearsonr, spearmanr

SB_START, SB_END = 26.0, 32.0
base = 'C:/Users/Lequia/Desktop/HPSEC/Dades3'


def find_mf(path):
    for f in os.listdir(path):
        if f.endswith('MasterFile.xlsx'):
            return os.path.join(path, f)


# COLUMN sequences only (BP has different time fractions)
seqs_to_check = [
    # IC alt (205-239)
    ('205_SEQ', 9.3), ('206_SEQ', 8.5), ('210_SEQ', 9.1), ('212_SEQ', 8.5),
    ('213_SEQ', 8.6), ('216_SEQ', 8.8), ('218_SEQ', 8.7), ('219_SEQ', 8.6),
    ('222_SEQ', 8.7), ('223_SEQ', 8.5), ('228_SEQ', 8.4), ('230_SEQ', 9.4),
    ('232_SEQ', 10.0), ('233_SEQ', 9.3), ('234_SEQ', 9.0),
    # Transicio (244-250)
    ('244_SEQ', 1.4), ('246_SEQ', 0.9), ('248_SEQ', 0.6), ('250_SEQ', 0.6),
    # IC zero
    ('272_SEQ', 0.0), ('274_SEQ', 0.0), ('275_SEQ', 0.0), ('276_SEQ', 0.0),
    ('278_SEQ', 0.0), ('282_SEQ', 0.0), ('283_SEQ', 0.0), ('285_SEQ', 0.0),
    # IC baix antic
    ('074_SEQ', 0.4), ('075_SEQ', 0.5), ('076_SEQ', 0.5), ('077_SEQ', 0.5),
]

results = []
for seq_name, ic_expected in seqs_to_check:
    seq_path = os.path.join(base, seq_name)
    if not os.path.isdir(seq_path):
        continue
    mf = find_mf(seq_path)
    if not mf:
        continue

    try:
        toc_df = pd.read_excel(mf, sheet_name='2-TOC', header=6)
        toc_v = toc_df['TOC(ppb)'].astype(float).values
        ic_v = toc_df['IC(ppb)'].astype(float).values

        dates = pd.to_datetime(toc_df['Date Started'], errors='coerce')
        if dates.notna().sum() > 10:
            t_min = (dates - dates.iloc[0]).dt.total_seconds().values / 60.0
        else:
            t_min = np.arange(len(toc_v)) * 4.06 / 60.0

        # Find main DOC peaks (HS region)
        peaks, _ = find_peaks(toc_v, height=60, distance=150, prominence=30)

        for pk in peaks:
            t_pk = t_min[pk]
            t_inj_start = t_pk - 20  # HS peak at ~20 min relative

            # SB zone
            sb_mask = (t_min >= t_inj_start + SB_START) & (t_min <= t_inj_start + SB_END)
            # Baseline zone (5-10 min relative)
            bl_mask = (t_min >= t_inj_start + 5) & (t_min <= t_inj_start + 10)

            if sb_mask.sum() < 3 or bl_mask.sum() < 3:
                continue

            toc_sb = toc_v[sb_mask]
            ic_sb = ic_v[sb_mask]
            toc_bl = toc_v[bl_mask]
            ic_bl = ic_v[bl_mask]

            sb_peak = np.max(toc_sb) - np.median(toc_bl)
            sb_mean = np.mean(toc_sb) - np.median(toc_bl)
            ic_mean = np.mean(ic_v)
            ic_at_sb = np.mean(ic_sb)
            ic_at_bl = np.mean(ic_bl)

            results.append({
                'seq': seq_name,
                'ic_global': ic_mean,
                'ic_at_sb': ic_at_sb,
                'ic_at_bl': ic_at_bl,
                'ic_delta_sb': ic_at_sb - ic_at_bl,
                'sb_peak': sb_peak,
                'sb_mean': sb_mean,
                'toc_peak': toc_v[pk],
            })
    except Exception as e:
        print('Error %s: %s' % (seq_name, str(e)[:80]))

df = pd.DataFrame(results)
print('Total injections analyzed:', len(df))

# Group by IC level
df['ic_group'] = 'zero'
df.loc[df['ic_global'] > 0.5, 'ic_group'] = 'baix'
df.loc[df['ic_global'] > 5, 'ic_group'] = 'alt'

print('\nPer group:')
for grp in ['alt', 'baix', 'zero']:
    sub = df[df['ic_group'] == grp]
    if len(sub) == 0:
        continue
    print('  %s (n=%d):' % (grp, len(sub)))
    print('    SB_peak  = %.1f +/- %.1f ppb' % (sub['sb_peak'].mean(), sub['sb_peak'].std()))
    print('    IC_at_SB = %.2f +/- %.2f ppb' % (sub['ic_at_sb'].mean(), sub['ic_at_sb'].std()))
    print('    IC_at_BL = %.2f +/- %.2f ppb' % (sub['ic_at_bl'].mean(), sub['ic_at_bl'].std()))
    print('    IC delta = %.3f +/- %.3f ppb' % (sub['ic_delta_sb'].mean(), sub['ic_delta_sb'].std()))

# Correlations
print('\n--- Correlacions ---')
if len(df) > 5:
    r_p, p_p = pearsonr(df['sb_peak'], df['ic_global'])
    r_s, p_s = spearmanr(df['sb_peak'], df['ic_global'])
    print('SB_peak vs IC_global:   Pearson r=%.3f (p=%.4f), Spearman r=%.3f (p=%.4f)' % (r_p, p_p, r_s, p_s))

    r2, p2 = pearsonr(df['sb_peak'], df['ic_delta_sb'])
    print('SB_peak vs IC_delta_SB: Pearson r=%.3f (p=%.4f)' % (r2, p2))

    # Within IC-high group only
    ic_alt = df[df['ic_group'] == 'alt']
    if len(ic_alt) > 5:
        r3, p3 = pearsonr(ic_alt['sb_peak'], ic_alt['ic_delta_sb'])
        r4, p4 = pearsonr(ic_alt['sb_peak'], ic_alt['ic_at_sb'])
        print('(Bloc IC alt) SB_peak vs IC_delta: r=%.3f (p=%.4f)' % (r3, p3))
        print('(Bloc IC alt) SB_peak vs IC_at_SB: r=%.3f (p=%.4f)' % (r4, p4))

# Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Correlacio SB peak vs IC(ppb)', fontsize=14, fontweight='bold')

# 1. SB peak vs IC global
ax = axes[0, 0]
for grp, color, marker in [('alt', 'red', 'o'), ('baix', 'orange', 's'), ('zero', 'blue', '^')]:
    sub = df[df['ic_group'] == grp]
    ax.scatter(sub['sb_peak'], sub['ic_global'], c=color, marker=marker,
               alpha=0.6, s=30, label='IC %s (n=%d)' % (grp, len(sub)))
ax.set_xlabel('SB peak height (ppb above baseline)')
ax.set_ylabel('IC global mean (ppb)')
ax.set_title('SB peak vs IC global')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 2. SB peak vs IC delta at SB (within IC-high group)
ax = axes[0, 1]
for grp, color, marker in [('alt', 'red', 'o'), ('baix', 'orange', 's'), ('zero', 'blue', '^')]:
    sub = df[df['ic_group'] == grp]
    ax.scatter(sub['sb_peak'], sub['ic_delta_sb'], c=color, marker=marker,
               alpha=0.6, s=30, label='IC %s' % grp)
ax.set_xlabel('SB peak height (ppb above baseline)')
ax.set_ylabel('IC delta (SB - baseline) (ppb)')
ax.set_title('SB peak vs IC increment a la zona SB')
ax.axhline(0, color='gray', ls='--', lw=0.8)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 3. Histogram IC at SB vs baseline (IC-high group)
ax = axes[1, 0]
ic_alt = df[df['ic_group'] == 'alt']
if len(ic_alt) > 0:
    ax.hist(ic_alt['ic_at_bl'], bins=20, alpha=0.5, color='blue', label='IC at baseline zone')
    ax.hist(ic_alt['ic_at_sb'], bins=20, alpha=0.5, color='red', label='IC at SB zone')
    ax.set_title('Bloc IC alt (205-239): IC a baseline vs SB')
    ax.set_xlabel('IC (ppb)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# 4. SB peak distribution by IC group
ax = axes[1, 1]
for grp, color in [('alt', 'red'), ('baix', 'orange'), ('zero', 'blue')]:
    sub = df[df['ic_group'] == grp]
    if len(sub) > 0:
        ax.hist(sub['sb_peak'], bins=15, alpha=0.4, color=color,
                label='IC %s (n=%d, SB mean=%.1f)' % (grp, len(sub), sub['sb_peak'].mean()))
ax.set_xlabel('SB peak height (ppb above baseline)')
ax.set_ylabel('Count')
ax.set_title('Distribucio SB peak per grup IC')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = 'C:/Users/Lequia/Desktop/HPSEC/SB_vs_IC_correlation.png'
plt.savefig(out, dpi=150)
print('\nSaved:', out)
