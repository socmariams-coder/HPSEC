#!/usr/bin/env python3
"""
Diagnòstic UIB 1ppm — Analitza per què les dues rèpliques agafen pics/àrees diferents.

Llegeix calibration_result.json (ja processat pel wizard) i compara les rèpliques
UIB per cada concentració. Genera gràfics i taula comparativa.

Output: _diag_uib_1ppm/ (gràfics + CSV resum)
"""

import sys
import os
import json
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from hpsec_config import get_config

OUTPUT_DIR = Path("_diag_uib_1ppm")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Trobar SEQ_CAL ──────────────────────────────────────────────────────
cfg = get_config()
data_folder = Path(cfg.get("paths", "data_folder", default=""))

SEQ_CAL_NAME = None  # None = auto-detect última

if SEQ_CAL_NAME:
    seq_path = data_folder / SEQ_CAL_NAME
else:
    candidates = []
    for item in sorted(data_folder.iterdir(), reverse=True):
        if item.is_dir() and '_CAL' in item.name.upper() and '_SEQ' in item.name.upper():
            cal_json = item / "CHECK" / "data" / "calibration_result.json"
            if cal_json.exists():
                candidates.append(item)
    seq_path = candidates[0] if candidates else None
    if not seq_path:
        print("ERROR: No s'ha trobat cap SEQ_CAL amb calibration_result.json")
        sys.exit(1)

print(f"SEQ_CAL: {seq_path.name}")
print(f"Path: {seq_path}")

# ── Llegir calibration_result.json ──────────────────────────────────────
cal_path = seq_path / "CHECK" / "data" / "calibration_result.json"
with open(cal_path, 'r', encoding='utf-8') as f:
    cal_result = json.load(f)

method = cal_result.get("mode", cal_result.get("method", "COLUMN"))
print(f"Method: {method}")
print(f"Success: {cal_result.get('success')}")

# Buscar sensibilitat UIB al manifest
manifest_path = seq_path / "CHECK" / "data" / "import_manifest.json"
uib_sensitivity = None
if manifest_path.exists():
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    raw_sens = manifest.get("sequence", {}).get("uib_sensitivity")
    if raw_sens:
        m = re.search(r'(\d+)\s*$', str(raw_sens))
        uib_sensitivity = float(m.group(1)) if m else None
print(f"UIB sensitivity: {uib_sensitivity}")
print()

# ── Extreure dades per mostra ───────────────────────────────────────────
# calibration_result té "calibrations" amb cada mostra analitzada
calibrations = cal_result.get("calibrations", [])
print(f"Total calibrations: {len(calibrations)}")

# Separar per senyal (doc_source)
entries_direct = []
entries_uib = []
for cal in calibrations:
    source = cal.get("doc_source", "direct")
    name = cal.get("name", "")
    conc = cal.get("conc_ppm", 0)
    replica = cal.get("replica", "")
    if "uib" in source.lower():
        entries_uib.append(cal)
    else:
        entries_direct.append(cal)

print(f"Direct entries: {len(entries_direct)}")
print(f"UIB entries: {len(entries_uib)}")
print()

# ── Taula completa UIB ──────────────────────────────────────────────────
if entries_uib:
    print(f"{'='*90}")
    print("TOTES LES ENTRADES UIB")
    print(f"{'='*90}")
    print(f"{'Name':<18} {'Rep':>3} {'Conc':>5} {'t_ret':>7} {'Area':>8} {'RF':>7} "
          f"{'t_left':>7} {'t_right':>7} {'width':>6} {'CR':>5} {'Q':>4} {'A254':>7} {'Shift':>7}")
    print("-" * 110)

    for e in sorted(entries_uib, key=lambda x: (x.get('conc_ppm', 0), x.get('name', ''))):
        name = e.get('name', '')
        rep = e.get('replica', '')
        conc = e.get('conc_ppm', 0)
        t_ret = e.get('t_retention', 0)
        area = e.get('area', 0)
        rf = e.get('rf_mass', 0)
        t_left = e.get('t_left', 0)
        t_right = e.get('t_right', 0)
        width = t_right - t_left if t_right and t_left else 0
        cr = e.get('concentration_ratio', 0)
        q = e.get('quality_score', 0)
        a254 = e.get('a254_area', 0)
        shift = e.get('shift_khp', 0)

        print(f"{name:<18} {rep:>3} {conc:5.2f} {t_ret:7.3f} {area:8.1f} {rf:7.1f} "
              f"{t_left:7.3f} {t_right:7.3f} {width:6.3f} {cr:5.3f} {q:4d} {a254:7.1f} {shift:7.3f}")

        # Anomalies
        anomalies = e.get('calibration_anomalies', [])
        qi = e.get('quality_issues', [])
        if anomalies:
            for a in anomalies:
                code = a.get('code', '?')
                label = a.get('label', '')
                print(f"{'':>22}  ⚠ {code}: {label}")
        if qi:
            for q_str in qi[:3]:
                print(f"{'':>22}  · {q_str}")

# ── Agrupar rèpliques UIB per concentració ──────────────────────────────
print(f"\n{'='*90}")
print("COMPARACIÓ RÈPLIQUES UIB PER CONCENTRACIÓ")
print(f"{'='*90}")

from collections import defaultdict
conc_groups = defaultdict(list)
for e in entries_uib:
    conc = e.get('conc_ppm', 0)
    conc_groups[conc].append(e)

for conc in sorted(conc_groups.keys()):
    reps = conc_groups[conc]
    if len(reps) < 2:
        continue

    areas = [r.get('area', 0) for r in reps]
    t_rets = [r.get('t_retention', 0) for r in reps]
    a_mean = np.mean(areas)
    a_rsd = np.std(areas) / a_mean * 100 if a_mean > 0 else 0
    dt = max(t_rets) - min(t_rets)

    print(f"\n  {conc:g} ppm — {len(reps)} rèpliques")
    print(f"  Àrees: {', '.join(f'{a:.1f}' for a in areas)}")
    print(f"  RSD àrees: {a_rsd:.1f}%")
    print(f"  t_ret: {', '.join(f'{t:.3f}' for t in t_rets)}")
    print(f"  Δt_ret: {dt:.3f} min")

    # Detalls
    for r in reps:
        name = r.get('name', '')
        rep = r.get('replica', '')
        t_left = r.get('t_left', 0)
        t_right = r.get('t_right', 0)
        width = t_right - t_left
        shift = r.get('shift_khp', 0)
        cr = r.get('concentration_ratio', 0)
        print(f"    {name}_R{rep}: area={r.get('area',0):.1f}, "
              f"t=[{t_left:.3f}, {t_right:.3f}] ({width:.3f}), "
              f"shift254={shift:.3f}, CR={cr:.3f}")

# ── Comparació Direct vs UIB (mateixa mostra) ───────────────────────────
print(f"\n{'='*90}")
print("COMPARACIÓ DIRECT vs UIB (mateixes mostres)")
print(f"{'='*90}")

# Agrupar per (name, replica)
direct_by_key = {}
for e in entries_direct:
    key = (e.get('name', ''), str(e.get('replica', '')))
    direct_by_key[key] = e

print(f"{'Name':<18} {'Rep':>3} {'t_DOC':>7} {'A_DOC':>8} {'t_UIB':>7} {'A_UIB':>8} "
      f"{'Δt':>7} {'ΔA%':>7} {'t254':>7}")
print("-" * 90)

for e_uib in sorted(entries_uib, key=lambda x: (x.get('conc_ppm', 0), x.get('name', ''))):
    key = (e_uib.get('name', ''), str(e_uib.get('replica', '')))
    e_dir = direct_by_key.get(key)
    if not e_dir:
        continue

    name = e_uib.get('name', '')
    rep = e_uib.get('replica', '')
    t_d = e_dir.get('t_retention', 0)
    a_d = e_dir.get('area', 0)
    t_u = e_uib.get('t_retention', 0)
    a_u = e_uib.get('area', 0)
    dt = t_u - t_d
    da_pct = (a_u - a_d) / a_d * 100 if a_d > 0 else 0
    t_254 = e_uib.get('t_max_254', e_dir.get('t_max_254', 0))
    if not t_254:
        # Buscar al bigaussian
        bg = e_uib.get('bigaussian_254', {})
        if bg:
            t_254 = bg.get('t_center', 0)

    print(f"{name:<18} {rep:>3} {t_d:7.3f} {a_d:8.1f} {t_u:7.3f} {a_u:8.1f} "
          f"{dt:7.3f} {da_pct:7.1f}% {t_254:7.3f}")

# ── Scatter àrea Direct vs UIB ─────────────────────────────────────────
pairs = []
for e_uib in entries_uib:
    key = (e_uib.get('name', ''), str(e_uib.get('replica', '')))
    e_dir = direct_by_key.get(key)
    if e_dir:
        pairs.append({
            'conc': e_uib.get('conc_ppm', 0),
            'area_direct': e_dir.get('area', 0),
            'area_uib': e_uib.get('area', 0),
            'name': e_uib.get('name', ''),
        })

if pairs:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. Scatter àrea Direct vs UIB
    ax = axes[0]
    concs_unique = sorted(set(p['conc'] for p in pairs))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(concs_unique), 1)))
    for ci, conc in enumerate(concs_unique):
        ps = [p for p in pairs if p['conc'] == conc]
        ax.scatter([p['area_direct'] for p in ps], [p['area_uib'] for p in ps],
                   c=[colors[ci]], label=f'{conc:g} ppm', s=50, alpha=0.8, edgecolors='k', linewidths=0.5)

    all_areas = [p['area_direct'] for p in pairs] + [p['area_uib'] for p in pairs]
    lim = max(all_areas) * 1.1
    ax.plot([0, lim], [0, lim], 'k--', alpha=0.3, label='1:1')
    ax.set_xlabel('Àrea Direct', fontsize=10)
    ax.set_ylabel('Àrea UIB', fontsize=10)
    ax.set_title('Direct vs UIB', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)

    # 2. t_retention Direct vs UIB
    ax = axes[1]
    for e_uib in entries_uib:
        key = (e_uib.get('name', ''), str(e_uib.get('replica', '')))
        e_dir = direct_by_key.get(key)
        if e_dir:
            conc = e_uib.get('conc_ppm', 0)
            ci = concs_unique.index(conc) if conc in concs_unique else 0
            ax.scatter(e_dir.get('t_retention', 0), e_uib.get('t_retention', 0),
                       c=[colors[ci]], s=50, alpha=0.8, edgecolors='k', linewidths=0.5)

    # Referència 254nm
    for e_uib in entries_uib:
        t_254 = 0
        bg = e_uib.get('bigaussian_254', {})
        if bg:
            t_254 = bg.get('t_center', 0)
        if t_254 > 0:
            ax.axhline(t_254, color='purple', ls=':', alpha=0.2)
            ax.axvline(t_254, color='purple', ls=':', alpha=0.2)
            break  # 254 és el mateix per totes

    ax.set_xlabel('t_ret Direct (min)', fontsize=10)
    ax.set_ylabel('t_ret UIB (min)', fontsize=10)
    ax.set_title('Temps retenció Direct vs UIB', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 3. Regressió àrea vs µg_DOC per UIB
    ax = axes[2]
    for e in entries_uib:
        conc = e.get('conc_ppm', 0)
        vol = e.get('volume_uL', 0)
        area = e.get('area', 0)
        if conc > 0 and vol > 0:
            ug = conc * vol / 1000
            ci = concs_unique.index(conc) if conc in concs_unique else 0
            ax.scatter(ug, area, c=[colors[ci]], s=50, alpha=0.8,
                       edgecolors='k', linewidths=0.5, label=f'{conc:g}' if ci == 0 else '')

    # Regressió lineal
    x_all = []
    y_all = []
    for e in entries_uib:
        conc = e.get('conc_ppm', 0)
        vol = e.get('volume_uL', 0)
        area = e.get('area', 0)
        if conc > 0 and vol > 0 and area > 0:
            x_all.append(conc * vol / 1000)
            y_all.append(area)

    if len(x_all) >= 2:
        x_arr = np.array(x_all)
        y_arr = np.array(y_all)
        # OLS
        A = np.column_stack([x_arr, np.ones_like(x_arr)])
        result = np.linalg.lstsq(A, y_arr, rcond=None)
        slope, intercept = result[0]
        y_pred = slope * x_arr + intercept
        ss_res = np.sum((y_arr - y_pred) ** 2)
        ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        rms = np.sqrt(np.mean((y_arr - y_pred) ** 2))

        x_line = np.linspace(0, max(x_arr) * 1.1, 100)
        ax.plot(x_line, slope * x_line + intercept, 'b-', alpha=0.7,
                label=f'y={slope:.1f}x+{intercept:.1f}\nR²={r2:.4f}, RMS={rms:.1f}')

        # Residuals per punt
        print(f"\n{'='*90}")
        print(f"REGRESSIÓ UIB: slope={slope:.1f}, intercept={intercept:.1f}, R²={r2:.6f}, RMS={rms:.1f}")
        print(f"{'='*90}")
        print(f"{'Name':<18} {'µg':>6} {'area':>8} {'pred':>8} {'resid':>8} {'%dev':>7}")
        print("-" * 60)
        for e in sorted(entries_uib, key=lambda x: x.get('conc_ppm', 0)):
            conc = e.get('conc_ppm', 0)
            vol = e.get('volume_uL', 0)
            area = e.get('area', 0)
            if conc > 0 and vol > 0 and area > 0:
                ug = conc * vol / 1000
                pred = slope * ug + intercept
                resid = area - pred
                dev = resid / pred * 100 if pred > 0 else 0
                print(f"{e.get('name',''):<18} {ug:6.3f} {area:8.1f} {pred:8.1f} {resid:8.1f} {dev:7.1f}%")

    ax.set_xlabel('µg DOC injectat', fontsize=10)
    ax.set_ylabel('Àrea UIB', fontsize=10)
    ax.set_title('Recta UIB', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "uib_vs_direct_overview.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nGràfic: {OUTPUT_DIR / 'uib_vs_direct_overview.png'}")

# ── Gràfic per concentració: rèpliques UIB superposades ─────────────────
for conc in sorted(conc_groups.keys()):
    reps = conc_groups[conc]
    if len(reps) < 2:
        continue

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    colors_rep = ['#2196F3', '#E91E63', '#4CAF50', '#FF9800']

    for i, r in enumerate(reps):
        name = r.get('name', '')
        rep = r.get('replica', '')
        t_ret = r.get('t_retention', 0)
        area = r.get('area', 0)
        t_left = r.get('t_left', 0)
        t_right = r.get('t_right', 0)
        c = colors_rep[i % len(colors_rep)]

        # Marcar punt de retenció i límits
        ax.axvline(t_ret, color=c, ls='--', alpha=0.5,
                   label=f'{name}_R{rep}: t={t_ret:.3f}, A={area:.1f}')
        ax.axvspan(t_left, t_right, alpha=0.1, color=c)

    # 254nm reference
    t_254_vals = [r.get('shift_khp', 0) + r.get('t_retention', 0)
                  for r in reps if r.get('t_retention', 0) > 0]
    # Buscar t_max_254 directament
    for r in reps:
        bg254 = r.get('bigaussian_254', {})
        if bg254 and bg254.get('t_center', 0) > 0:
            ax.axvline(bg254['t_center'], color='purple', ls=':', alpha=0.5,
                       label=f'254nm: t={bg254["t_center"]:.3f}')
            break

    ax.set_title(f'{conc:g} ppm — Comparació rèpliques UIB (límits integració)',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Temps (min)')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"uib_{conc:g}ppm_replicas.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

# ── CSV resum ───────────────────────────────────────────────────────────
import csv
all_data = []
for e in entries_uib:
    all_data.append({
        'name': e.get('name', ''),
        'replica': e.get('replica', ''),
        'conc_ppm': e.get('conc_ppm', 0),
        'volume_uL': e.get('volume_uL', 0),
        'ug_doc': e.get('conc_ppm', 0) * e.get('volume_uL', 0) / 1000,
        'area': e.get('area', 0),
        'rf_mass': e.get('rf_mass', 0),
        't_retention': e.get('t_retention', 0),
        't_left': e.get('t_left', 0),
        't_right': e.get('t_right', 0),
        'width': e.get('t_right', 0) - e.get('t_left', 0),
        'shift_254': e.get('shift_khp', 0),
        'concentration_ratio': e.get('concentration_ratio', 0),
        'quality_score': e.get('quality_score', 0),
        'a254_area': e.get('a254_area', 0),
        'fwhm_doc': e.get('fwhm_doc', 0),
        'has_irregular_top': e.get('has_irregular_top', False),
        'irregular_top_repaired': e.get('irregular_top_repaired', False),
        'doc_source': e.get('doc_source', ''),
    })

if all_data:
    csv_path = OUTPUT_DIR / "uib_all_entries.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=all_data[0].keys())
        w.writeheader()
        w.writerows(all_data)
    print(f"\nCSV: {csv_path}")

print(f"\nOutput: {OUTPUT_DIR.absolute()}")
print("DONE")
