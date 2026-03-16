"""Comparacio regressio KHP_History vs calibracio anterior (CAL_20260220)"""
import json
import numpy as np
from hpsec_calibrate import fit_calibration_from_history

with open('C:/Users/Lequia/Desktop/Dades3/REGISTRY/KHP_History.json', 'r', encoding='utf-8') as f:
    cals = json.load(f)['calibrations']

with open('C:/Users/Lequia/Desktop/Dades3/REGISTRY/Calibration_Reference.json', 'r', encoding='utf-8') as f:
    calref = json.load(f)

# Calibracio anterior (CAL_20260220_001)
cal_old = next(c for c in calref['calibrations'] if c['id'] == 'CAL_20260220_001')

# Calibracio vigent (293_SEQ_CAL)
cal_new = next(c for c in calref['calibrations'] if c.get('is_active'))

# Registry antic
with open('C:/Users/Lequia/Desktop/Dades3/REGISTRY/KHP_Calibration_Registry.json', 'r', encoding='utf-8') as f:
    registry = json.load(f)

print('=' * 75)
print('COMPARACIO: Regressio KHP_History vs Calibracio Anterior (CAL_20260220)')
print('=' * 75)

# --- COLUMN ---
col_clean = [e for e in cals if
    e.get('valid_for_calibration') and not e.get('is_bp') and not e.get('is_outlier')
    and e.get('volume_uL') == 400 and 1 <= e.get('conc_ppm', 0) <= 5]

r_col_d = fit_calibration_from_history(col_clean, mode='COLUMN', signal='direct', model='intercept')
r_col_u = fit_calibration_from_history(col_clean, mode='COLUMN', signal='uib', model='intercept')

old_d_col = cal_old['rf_mass_cal']['direct']['column']
old_i_d_col = cal_old['intercept']['direct']['column']
old_u_col = cal_old['rf_mass_cal']['uib']['column']
old_i_u_col = cal_old['intercept']['uib']['column']
new_d_col = cal_new['rf_mass_cal']['direct']['column']
new_i_d_col = cal_new['intercept']['direct']['column']
new_u_col = cal_new['rf_mass_cal']['uib']['column']
new_i_u_col = cal_new['intercept']['uib']['column']

print()
print('COLUMN DIRECT')
hdr = f"  {'':>25} {'RF':>8} {'Int':>8} {'R2':>8} {'n':>4}"
print(hdr)
print(f"  {'Registry (2026-02-08)':>25} {registry['bp_col_factor']['doc']['col_mean']:8.1f} {'--':>8} {'--':>8} {'--':>4}")
print(f"  {'CAL_20260220 (anterior)':>25} {old_d_col:8.1f} {old_i_d_col:8.1f} {cal_old['r2']['column']:8.4f} {cal_old['n_points']['column']:4d}")
print(f"  {'CAL_20260222 (vigent)':>25} {new_d_col:8.1f} {new_i_d_col:8.1f} {cal_new['r2']['column']:8.4f} {cal_new['n_points']:4d}")
print(f"  {'Regressio actual':>25} {r_col_d['rf_mass_cal']:8.1f} {r_col_d['intercept']:8.1f} {r_col_d['r2']:8.4f} {r_col_d['n_points']:4d}")
print()
d1 = (r_col_d['rf_mass_cal'] - old_d_col) / old_d_col * 100
print(f"  Delta vs anterior: RF {d1:+.1f}%, int {r_col_d['intercept'] - old_i_d_col:+.1f}")
d2 = (r_col_d['rf_mass_cal'] - new_d_col) / new_d_col * 100
print(f"  Delta vs vigent:   RF {d2:+.1f}%, int {r_col_d['intercept'] - new_i_d_col:+.1f}")

# Impacte en ppm per a mostra tipica (area=600, vol=400)
area_test = 600
vol_test = 400
ppm_old = (area_test - old_i_d_col) * 1000 / (old_d_col * vol_test)
ppm_new = (area_test - new_i_d_col) * 1000 / (new_d_col * vol_test)
ppm_reg = (area_test - r_col_d['intercept']) * 1000 / (r_col_d['rf_mass_cal'] * vol_test)
print(f"  Impacte ppm (area={area_test}, vol={vol_test}):")
print(f"    Anterior (628+81): {ppm_old:.3f} ppm")
print(f"    Vigent (752+18):   {ppm_new:.3f} ppm")
print(f"    Regressio (782+3): {ppm_reg:.3f} ppm")

print()
print('COLUMN UIB')
print(hdr)
print(f"  {'Registry (2026-02-08)':>25} {registry['bp_col_factor']['uib']['col_mean']:8.1f} {'--':>8} {'--':>8} {'--':>4}")
print(f"  {'CAL_20260220 (anterior)':>25} {old_u_col:8.1f} {old_i_u_col:8.1f} {'--':>8} {'--':>4}")
print(f"  {'CAL_20260222 (vigent)':>25} {new_u_col:8.1f} {new_i_u_col:8.1f} {'--':>8} {'--':>4}")
print(f"  {'Regressio actual':>25} {r_col_u['rf_mass_cal']:8.1f} {r_col_u['intercept']:8.1f} {r_col_u['r2']:8.4f} {r_col_u['n_points']:4d}")
print()
d1 = (r_col_u['rf_mass_cal'] - old_u_col) / old_u_col * 100
print(f"  Delta vs anterior: RF {d1:+.1f}%, int {r_col_u['intercept'] - old_i_u_col:+.1f}")
d2 = (r_col_u['rf_mass_cal'] - new_u_col) / new_u_col * 100
print(f"  Delta vs vigent:   RF {d2:+.1f}%, int {r_col_u['intercept'] - new_i_u_col:+.1f}")

ppm_old_u = (area_test - old_i_u_col) * 1000 / (old_u_col * vol_test)
ppm_new_u = (area_test - new_i_u_col) * 1000 / (new_u_col * vol_test)
ppm_reg_u = (area_test - r_col_u['intercept']) * 1000 / (r_col_u['rf_mass_cal'] * vol_test)
print(f"  Impacte ppm UIB (area={area_test}, vol={vol_test}):")
print(f"    Anterior (628+81): {ppm_old_u:.3f} ppm")
print(f"    Vigent (628+81):   {ppm_new_u:.3f} ppm")
print(f"    Regressio (775+29):{ppm_reg_u:.3f} ppm")

print()
print('-' * 75)

# --- BP ---
bp_clean = []
bp_valid = [e for e in cals if e.get('valid_for_calibration') and e.get('is_bp') and not e.get('is_outlier')]
for e in bp_valid:
    c = e.get('conc_ppm', 0)
    v = e.get('volume_uL', 0)
    a = e.get('area', 0)
    if c <= 0 or v <= 0 or a <= 0:
        continue
    ug = c * v / 1000
    rf = a / ug
    if 400 < rf < 1200:
        bp_clean.append(e)

r_bp = fit_calibration_from_history(bp_clean, mode='BP', signal='direct', model='intercept')

bp_cal = [e for e in cals if '_CAL' in e.get('seq_name', '').upper() and e.get('is_bp')]
r_bp_cal = fit_calibration_from_history(bp_cal, mode='BP', signal='direct', model='intercept')

old_d_bp = cal_old['rf_mass_cal']['direct']['bp']
old_i_d_bp = cal_old['intercept']['direct']['bp']
new_d_bp = cal_new['rf_mass_cal']['direct']['bp']
new_i_d_bp = cal_new['intercept']['direct']['bp']

print()
print('BP DIRECT')
print(hdr)
print(f"  {'Registry (2026-02-08)':>25} {registry['bp_col_factor']['doc']['bp_mean']:8.1f} {'--':>8} {'--':>8} {'--':>4}")
print(f"  {'CAL_20260220 (anterior)':>25} {old_d_bp:8.1f} {old_i_d_bp:8.1f} {cal_old['r2']['bp']:8.4f} {cal_old['n_points']['bp']:4d}")
print(f"  {'CAL_20260222 (vigent)':>25} {new_d_bp:8.1f} {new_i_d_bp:8.1f} {cal_new['r2']['bp']:8.4f} {'--':>4}")
print(f"  {'292_SEQ_CAL_BP':>25} {r_bp_cal['rf_mass_cal']:8.1f} {r_bp_cal['intercept']:8.1f} {r_bp_cal['r2']:8.4f} {r_bp_cal['n_points']:4d}")
print(f"  {'Regressio clean (n=21)':>25} {r_bp['rf_mass_cal']:8.1f} {r_bp['intercept']:8.1f} {r_bp['r2']:8.4f} {r_bp['n_points']:4d}")
print()
d1 = (r_bp_cal['rf_mass_cal'] - old_d_bp) / old_d_bp * 100
print(f"  Delta 292_CAL vs anterior:  RF {d1:+.1f}%")
d2 = (r_bp['rf_mass_cal'] - old_d_bp) / old_d_bp * 100
print(f"  Delta clean vs anterior:    RF {d2:+.1f}%")

# Impacte ppm BP (area=150, vol=100)
area_bp = 150
vol_bp = 100
ppm_old_bp = (area_bp - old_i_d_bp) * 1000 / (old_d_bp * vol_bp)
ppm_cal_bp = (area_bp - r_bp_cal['intercept']) * 1000 / (r_bp_cal['rf_mass_cal'] * vol_bp)
ppm_clean_bp = (area_bp - r_bp['intercept']) * 1000 / (r_bp['rf_mass_cal'] * vol_bp)
print(f"  Impacte ppm (area={area_bp}, vol={vol_bp}):")
print(f"    Anterior (817+0):  {ppm_old_bp:.3f} ppm")
print(f"    292_CAL (646+4):   {ppm_cal_bp:.3f} ppm")
print(f"    Clean (730+10):    {ppm_clean_bp:.3f} ppm")

print()
print('=' * 75)
print('RESUM COMPARATIU')
print('=' * 75)
print()
print(f"{'':>18} {'Anterior':>12} {'Vigent':>12} {'Regressio':>12} {'D vs ant':>10}")
rf = r_col_d['rf_mass_cal']
print(f"{'COL Direct RF':>18} {old_d_col:12.1f} {new_d_col:12.1f} {rf:12.1f} {(rf-old_d_col)/old_d_col*100:+9.1f}%")
it = r_col_d['intercept']
print(f"{'COL Direct int':>18} {old_i_d_col:12.1f} {new_i_d_col:12.1f} {it:12.1f} {it-old_i_d_col:+10.1f}")
rf = r_col_u['rf_mass_cal']
print(f"{'COL UIB RF':>18} {old_u_col:12.1f} {new_u_col:12.1f} {rf:12.1f} {(rf-old_u_col)/old_u_col*100:+9.1f}%")
it = r_col_u['intercept']
print(f"{'COL UIB int':>18} {old_i_u_col:12.1f} {new_i_u_col:12.1f} {it:12.1f} {it-old_i_u_col:+10.1f}")
rf = r_bp_cal['rf_mass_cal']
print(f"{'BP Direct RF':>18} {old_d_bp:12.1f} {new_d_bp:12.1f} {rf:12.1f} {(rf-old_d_bp)/old_d_bp*100:+9.1f}%")
it = r_bp_cal['intercept']
print(f"{'BP Direct int':>18} {old_i_d_bp:12.1f} {new_i_d_bp:12.1f} {it:12.1f} {it-old_i_d_bp:+10.1f}")

print()
print('IMPACTE EN PPM (mostra tipica):')
print(f"  COLUMN (area=600, vol=400):")
print(f"    Anterior: {ppm_old:.3f}   Vigent: {ppm_new:.3f}   Regressio: {ppm_reg:.3f}")
print(f"  BP (area=150, vol=100):")
print(f"    Anterior: {ppm_old_bp:.3f}   292_CAL: {ppm_cal_bp:.3f}   Clean: {ppm_clean_bp:.3f}")
