"""Anàlisi calibració COLUMN + BP des de KHP_History.json"""
import json
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

# Load KHP_History
with open('C:/Users/Lequia/Desktop/Dades3/REGISTRY/KHP_History.json', 'r', encoding='utf-8') as f:
    khp = json.load(f)
cals = khp['calibrations']

# Load current calibration
with open('C:/Users/Lequia/Desktop/Dades3/REGISTRY/Calibration_Reference.json', 'r', encoding='utf-8') as f:
    calref = json.load(f)
active_cal = next(c for c in calref['calibrations'] if c.get('is_active'))

from hpsec_calibrate import fit_calibration_from_history

print('=' * 80)
print('ANALISI CALIBRACIO COLUMN + BP')
print('=' * 80)

# =====================================================================
# 1. COLUMN DIRECT
# =====================================================================
print('\n' + '=' * 60)
print('1. COLUMN DIRECT (mode=COLUMN, signal=direct)')
print('=' * 60)
result_col_d = fit_calibration_from_history(cals, mode='COLUMN', signal='direct', model='intercept')
if result_col_d['success']:
    print(f"  RF (slope):    {result_col_d['rf_mass_cal']:.1f}")
    print(f"  Intercept:     {result_col_d['intercept']:.1f}")
    print(f"  R2:            {result_col_d['r2']:.6f}")
    print(f"  n_points:      {result_col_d['n_points']}")
    print(f"  RMS residuals: {result_col_d['residuals_rms']:.1f}")

    cur_rf = active_cal['rf_mass_cal']['direct']['column']
    cur_int = active_cal['intercept']['direct']['column']
    print(f"\n  Vigent:        RF={cur_rf:.1f}, intercept={cur_int:.1f}")
    delta_rf = (result_col_d['rf_mass_cal'] - cur_rf) / cur_rf * 100
    delta_int = result_col_d['intercept'] - cur_int
    print(f"  Delta:         RF {delta_rf:+.1f}%, intercept {delta_int:+.1f}")

    concs = {}
    for p in result_col_d['points']:
        c = p['conc_ppm']
        if c not in concs:
            concs[c] = []
        concs[c].append(p)
    print(f"\n  Per concentracio:")
    print(f"  {'Conc':>8} {'n':>3} {'RF_mean':>10} {'RF_std':>10} {'CV%':>8}")
    for c in sorted(concs.keys()):
        pts = concs[c]
        rfs = [p['rf_mass'] for p in pts]
        rf_mean = np.mean(rfs)
        rf_std = np.std(rfs) if len(rfs) > 1 else 0
        cv = rf_std / rf_mean * 100 if rf_mean > 0 else 0
        print(f"  {c:8.2f} {len(pts):3d} {rf_mean:10.1f} {rf_std:10.1f} {cv:8.1f}")
else:
    print(f"  ERROR: {result_col_d.get('error', 'unknown')}")

# =====================================================================
# 2. COLUMN UIB
# =====================================================================
print('\n' + '=' * 60)
print('2. COLUMN UIB (mode=COLUMN, signal=uib)')
print('=' * 60)
result_col_u = fit_calibration_from_history(cals, mode='COLUMN', signal='uib', model='intercept')
if result_col_u['success']:
    print(f"  RF (slope):    {result_col_u['rf_mass_cal']:.1f}")
    print(f"  Intercept:     {result_col_u['intercept']:.1f}")
    print(f"  R2:            {result_col_u['r2']:.6f}")
    print(f"  n_points:      {result_col_u['n_points']}")
    print(f"  RMS residuals: {result_col_u['residuals_rms']:.1f}")

    cur_rf = active_cal['rf_mass_cal']['uib']['column']
    cur_int = active_cal['intercept']['uib']['column']
    print(f"\n  Vigent:        RF={cur_rf:.1f}, intercept={cur_int:.1f}")
    delta_rf = (result_col_u['rf_mass_cal'] - cur_rf) / cur_rf * 100
    delta_int = result_col_u['intercept'] - cur_int
    print(f"  Delta:         RF {delta_rf:+.1f}%, intercept {delta_int:+.1f}")

    concs = {}
    for p in result_col_u['points']:
        c = p['conc_ppm']
        if c not in concs:
            concs[c] = []
        concs[c].append(p)
    print(f"\n  Per concentracio:")
    print(f"  {'Conc':>8} {'n':>3} {'RF_mean':>10} {'RF_std':>10} {'CV%':>8}")
    for c in sorted(concs.keys()):
        pts = concs[c]
        rfs = [p['rf_mass'] for p in pts]
        rf_mean = np.mean(rfs)
        rf_std = np.std(rfs) if len(rfs) > 1 else 0
        cv = rf_std / rf_mean * 100 if rf_mean > 0 else 0
        print(f"  {c:8.2f} {len(pts):3d} {rf_mean:10.1f} {rf_std:10.1f} {cv:8.1f}")
else:
    print(f"  ERROR: {result_col_u.get('error', 'unknown')}")

# =====================================================================
# 3. BP DIRECT
# =====================================================================
print('\n' + '=' * 60)
print('3. BP DIRECT (mode=BP, signal=direct)')
print('=' * 60)
result_bp_d = fit_calibration_from_history(cals, mode='BP', signal='direct', model='intercept')
if result_bp_d['success']:
    print(f"  RF (slope):    {result_bp_d['rf_mass_cal']:.1f}")
    print(f"  Intercept:     {result_bp_d['intercept']:.1f}")
    print(f"  R2:            {result_bp_d['r2']:.6f}")
    print(f"  n_points:      {result_bp_d['n_points']}")
    print(f"  RMS residuals: {result_bp_d['residuals_rms']:.1f}")

    cur_rf = active_cal['rf_mass_cal']['direct']['bp']
    cur_int = active_cal['intercept']['direct']['bp']
    print(f"\n  Vigent:        RF={cur_rf:.1f}, intercept={cur_int:.1f}")
    delta_rf = (result_bp_d['rf_mass_cal'] - cur_rf) / cur_rf * 100
    delta_int = result_bp_d['intercept'] - cur_int
    print(f"  Delta:         RF {delta_rf:+.1f}%, intercept {delta_int:+.1f}")

    concs = {}
    for p in result_bp_d['points']:
        c = p['conc_ppm']
        if c not in concs:
            concs[c] = []
        concs[c].append(p)
    print(f"\n  Per concentracio:")
    print(f"  {'Conc':>8} {'n':>3} {'RF_mean':>10} {'RF_std':>10} {'CV%':>8}")
    for c in sorted(concs.keys()):
        pts = concs[c]
        rfs = [p['rf_mass'] for p in pts]
        rf_mean = np.mean(rfs)
        rf_std = np.std(rfs) if len(rfs) > 1 else 0
        cv = rf_std / rf_mean * 100 if rf_mean > 0 else 0
        print(f"  {c:8.2f} {len(pts):3d} {rf_mean:10.1f} {rf_std:10.1f} {cv:8.1f}")
else:
    print(f"  ERROR: {result_bp_d.get('error', 'unknown')}")

# Also try origin model for BP
print('\n  --- BP DIRECT (origin model) ---')
result_bp_d_origin = fit_calibration_from_history(cals, mode='BP', signal='direct', model='origin')
if result_bp_d_origin['success']:
    print(f"  RF (slope):    {result_bp_d_origin['rf_mass_cal']:.1f}")
    print(f"  Intercept:     0 (forced)")
    print(f"  R2:            {result_bp_d_origin['r2']:.6f}")
    print(f"  n_points:      {result_bp_d_origin['n_points']}")
    print(f"  RMS residuals: {result_bp_d_origin['residuals_rms']:.1f}")

# =====================================================================
# 4. COMPARACIO: SEQ_CAL vs All vs Production
# =====================================================================
print('\n' + '=' * 60)
print('4. COMPARACIO: SEQ_CAL only vs Production vs All')
print('=' * 60)

cal_only = [e for e in cals if '_CAL' in e.get('seq_name', '').upper()]
prod_only = [e for e in cals if '_CAL' not in e.get('seq_name', '').upper()]

print(f"\nSEQ_CAL entries: {len(cal_only)}, Production: {len(prod_only)}, Total: {len(cals)}")

# COLUMN Direct variants
r_cal = fit_calibration_from_history(cal_only, mode='COLUMN', signal='direct', model='intercept')
r_prod = fit_calibration_from_history(prod_only, mode='COLUMN', signal='direct', model='intercept')
if r_cal['success']:
    print(f"\n  COLUMN Direct (SEQ_CAL only): RF={r_cal['rf_mass_cal']:.1f}, int={r_cal['intercept']:.1f}, R2={r_cal['r2']:.4f}, n={r_cal['n_points']}")
if r_prod['success']:
    print(f"  COLUMN Direct (Production):   RF={r_prod['rf_mass_cal']:.1f}, int={r_prod['intercept']:.1f}, R2={r_prod['r2']:.4f}, n={r_prod['n_points']}")

# BP Direct variants
r_cal_bp = fit_calibration_from_history(cal_only, mode='BP', signal='direct', model='intercept')
r_prod_bp = fit_calibration_from_history(prod_only, mode='BP', signal='direct', model='intercept')
if r_cal_bp['success']:
    print(f"\n  BP Direct (SEQ_CAL only):     RF={r_cal_bp['rf_mass_cal']:.1f}, int={r_cal_bp['intercept']:.1f}, R2={r_cal_bp['r2']:.4f}, n={r_cal_bp['n_points']}")
if r_prod_bp['success']:
    print(f"  BP Direct (Production):       RF={r_prod_bp['rf_mass_cal']:.1f}, int={r_prod_bp['intercept']:.1f}, R2={r_prod_bp['r2']:.4f}, n={r_prod_bp['n_points']}")

# =====================================================================
# 5. OUTLIER ANALYSIS
# =====================================================================
print('\n' + '=' * 60)
print('5. ANALISI OUTLIERS (> 3s residuals)')
print('=' * 60)

for label, result in [('COLUMN Direct', result_col_d), ('COLUMN UIB', result_col_u), ('BP Direct', result_bp_d)]:
    if not result['success']:
        continue
    rms = result['residuals_rms']
    outliers = [p for p in result['points'] if abs(p['residual']) > 3 * rms]
    high_res = sorted(result['points'], key=lambda p: abs(p['residual']), reverse=True)[:5]
    print(f"\n  {label}: RMS={rms:.1f}, 3s threshold={3 * rms:.1f}")
    if outliers:
        print(f"  Outliers ({len(outliers)}):")
        for o in outliers:
            print(f"    {o['seq_name']:30s} {o['conc_ppm']:5.2f}ppm  area={o['area']:8.1f}  residual={o['residual']:+8.1f}")
    else:
        print(f"  Cap outlier detectat")
    print(f"  Top 5 residuals:")
    for o in high_res:
        print(f"    {o['seq_name']:30s} {o['conc_ppm']:5.2f}ppm  area={o['area']:8.1f}  residual={o['residual']:+8.1f}")

# =====================================================================
# 6. ENTRADES ANOMALES
# =====================================================================
print('\n' + '=' * 60)
print('6. ENTRADES AMB conc=25 o 6.88 (COLUMN)')
print('=' * 60)
odd = [e for e in cals if e.get('conc_ppm', 0) in [25.0, 6.88] and not e.get('is_bp')]
for e in odd:
    print(f"  {e['seq_name']:30s} conc={e['conc_ppm']:6.2f} vol={e['volume_uL']} area={e.get('area', 0):.1f} valid={e.get('valid_for_calibration')}")

if not odd:
    print("  Cap entrada amb concentracions anomales")

# =====================================================================
# 7. RESUM I RECOMANACIO
# =====================================================================
print('\n' + '=' * 60)
print('7. RESUM I RECOMANACIO')
print('=' * 60)

print("\n  CALIBRACIO VIGENT:")
for sig in ['direct', 'uib']:
    for mode in ['column', 'bp']:
        rf = active_cal['rf_mass_cal'][sig][mode]
        intc = active_cal['intercept'][sig][mode]
        print(f"    {sig:8s} {mode:8s}: RF={rf:.1f}, intercept={intc:.1f}")

print("\n  RESULTATS REGRESSIO (all valid entries):")
results_map = {
    'COLUMN Direct': result_col_d,
    'COLUMN UIB': result_col_u,
    'BP Direct': result_bp_d,
}
for label, r in results_map.items():
    if r['success']:
        print(f"    {label:16s}: RF={r['rf_mass_cal']:.1f}, int={r['intercept']:.1f}, R2={r['r2']:.4f}, n={r['n_points']}")
    else:
        print(f"    {label:16s}: FAILED - {r.get('error', 'unknown')}")

print("\n  RECOMANACIONS:")
if result_col_d['success']:
    d = abs(result_col_d['rf_mass_cal'] - active_cal['rf_mass_cal']['direct']['column']) / active_cal['rf_mass_cal']['direct']['column'] * 100
    if d < 5:
        print(f"    COLUMN Direct: OK (delta {d:.1f}% < 5%) - no cal canviar")
    elif d < 15:
        print(f"    COLUMN Direct: REVISAR (delta {d:.1f}%) - considerar actualitzar")
    else:
        print(f"    COLUMN Direct: ACTUALITZAR (delta {d:.1f}% > 15%)")

if result_col_u['success']:
    d = abs(result_col_u['rf_mass_cal'] - active_cal['rf_mass_cal']['uib']['column']) / active_cal['rf_mass_cal']['uib']['column'] * 100
    if d < 5:
        print(f"    COLUMN UIB:    OK (delta {d:.1f}% < 5%) - no cal canviar")
    elif d < 15:
        print(f"    COLUMN UIB:    REVISAR (delta {d:.1f}%) - considerar actualitzar")
    else:
        print(f"    COLUMN UIB:    ACTUALITZAR (delta {d:.1f}% > 15%)")

if result_bp_d['success']:
    d = abs(result_bp_d['rf_mass_cal'] - active_cal['rf_mass_cal']['direct']['bp']) / active_cal['rf_mass_cal']['direct']['bp'] * 100
    if d < 5:
        print(f"    BP Direct:     OK (delta {d:.1f}% < 5%) - no cal canviar")
    elif d < 15:
        print(f"    BP Direct:     REVISAR (delta {d:.1f}%) - considerar actualitzar")
    else:
        print(f"    BP Direct:     ACTUALITZAR (delta {d:.1f}% > 15%)")
