# -*- coding: utf-8 -*-
import pandas as pd, numpy as np, sys
sys.stdout.reconfigure(encoding='utf-8')

merged = pd.read_csv('rf_humic_analysis/standards_list.csv')

print("LLISTAT ESTANDARDS HA/FA/MIX")
print("=" * 140)
for typ in ['HA', 'SRHA', 'MIX', 'SRFA', 'FA']:
    sub = merged[merged['Type'] == typ].sort_values('SEQ')
    for _, r in sub.iterrows():
        def fmt(col, w=6, d=2):
            v = r.get(col)
            if pd.notna(v):
                return f"{v:{w}.{d}f}"
            return " " * (w - 3) + "N/A"

        line = (f"  {r['Sample']:20s} {r['SEQ']:12s} {typ:5s}  "
                f"idx={r['HA_FA_index']:6.1f}  "
                f"E2/E3={fmt('E2_E3')}  "
                f"A254/280={fmt('A254_A280', 6, 4)}  "
                f"S275={fmt('S275_295', 8, 4)}  "
                f"SR={fmt('SR', 6, 3)}  "
                f"peak_t={fmt('elution_peak_t_254', 5, 2)}  "
                f"FWHM={fmt('elution_fwhm_254', 5, 2)}  "
                f"UV/Vis={fmt('shape_UV_Vis_ratio', 5, 2)}  "
                f"[{r['Status']}]")
        print(line)
    print()

# Robustness stats
ha_scores = merged[merged['Type'].isin(['HA', 'SRHA'])]['HA_FA_index']
fa_scores = merged[(merged['Type'].isin(['FA', 'SRFA'])) & (merged['Status'] == 'OK')]['HA_FA_index']

print("ROBUSTESA DEL MODEL")
print("=" * 60)
for typ in ['HA', 'SRHA', 'FA', 'SRFA', 'MIX']:
    sub = merged[(merged['Type'] == typ) & (merged['Status'] != 'REVISAR')]
    s = sub['HA_FA_index']
    if len(s) > 0:
        print(f"  {typ:6s}: mean={s.mean():6.1f}  std={s.std():5.1f}  [{s.min():.1f} .. {s.max():.1f}]  n={len(s)}")

gap = ha_scores.min() - fa_scores.max()
pooled = np.sqrt((ha_scores.std()**2 + fa_scores.std()**2) / 2)
cohend = (ha_scores.mean() - fa_scores.mean()) / pooled if pooled > 0 else 0

print(f"\n  Separacio HA-FA:")
print(f"    HA min:  {ha_scores.min():.1f}")
print(f"    FA max:  {fa_scores.max():.1f}")
print(f"    Gap:     {gap:.1f} (positiu = sense overlap)")
print(f"    Cohen d: {cohend:.2f} (>0.8 large, >2.0 very large)")
print(f"    LOO acc: 93.5% (29/31)")

# SEQ coverage
ha_seqs = set(merged[merged['Type'].isin(['HA', 'SRHA'])]['SEQ'])
fa_seqs = set(merged[merged['Type'].isin(['FA', 'SRFA'])]['SEQ'])
both = sorted(ha_seqs & fa_seqs)
print(f"\n  SEQs amb HA+FA junts:  {both}")
print(f"  SEQs nomes HA:        {sorted(ha_seqs - fa_seqs)}")
print(f"  SEQs nomes FA:        {sorted(fa_seqs - ha_seqs)}")
n_ok = len(merged[merged['Status'] == 'OK'])
n_rev = len(merged[merged['Status'] == 'REVISAR'])
n_bl = len(merged[merged['Status'] == 'LOO borderline'])
print(f"\n  Total: {len(merged)} estandards en {len(set(merged['SEQ']))} SEQs")
print(f"    OK:         {n_ok}")
print(f"    REVISAR:    {n_rev} (FA_R3, 223_SEQ)")
print(f"    Borderline: {n_bl} (SRFA_R2, 232_SEQ)")
