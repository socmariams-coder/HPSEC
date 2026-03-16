"""Analysis: COLUMN fractions vs BP total + KHP cross-check."""
import json, os
import numpy as np
from collections import defaultdict
from pathlib import Path
from hpsec_config import get_data_folders
from hpsec_consolidate import detect_seq_type

FRACS = ['BioP', 'HS', 'BB', 'SB']

def main():
    folders = get_data_folders()
    seqs = []
    for folder in folders:
        for d in sorted(os.listdir(folder)):
            full = os.path.join(folder, d)
            jp = os.path.join(full, 'CHECK', 'data', 'analysis_result.json')
            cp = os.path.join(full, 'CHECK', 'data', 'calibration_result.json')
            mp = os.path.join(full, 'CHECK', 'data', 'import_manifest.json')
            if not os.path.exists(jp):
                continue
            with open(jp, encoding='utf-8') as f:
                analysis = json.load(f)
            cal = None
            if os.path.exists(cp):
                with open(cp, encoding='utf-8') as f:
                    cal = json.load(f)
            vol = None
            if os.path.exists(mp):
                with open(mp, encoding='utf-8') as f:
                    m = json.load(f)
                samps = m.get('samples', [])
                if isinstance(samps, list):
                    for s in samps:
                        for r in (s.get('replicas') or []):
                            v = (r.get('injection') or {}).get('inj_volume')
                            if v:
                                vol = v
                                break
                        if vol:
                            break
            seqs.append({'name': d, 'path': full, 'method': detect_seq_type(d),
                         'analysis': analysis, 'calibration': cal, 'volume': vol})

    # Build inventory
    inventory = defaultdict(dict)
    for seq in seqs:
        sg = seq['analysis'].get('samples_grouped', {})
        for name, sdata in sg.items():
            st = sdata.get('sample_type', 'SAMPLE')
            if st not in ('SAMPLE', 'KHP'):
                continue
            selected = (sdata.get('selected') or {}).get('doc', '1')
            replicas = sdata.get('replicas') or {}
            doc_rep = replicas.get(selected, {})
            if selected == 'comp':
                for rv in replicas.values():
                    if isinstance(rv, dict) and rv.get('timeout_composition'):
                        doc_rep = rv
                        break
            if not isinstance(doc_rep, dict):
                doc_rep = {}
            areas = (doc_rep.get('areas') or {}).get('DOC') or {}
            quant = sdata.get('quantification') or {}
            ti = doc_rep.get('timeout_info') or {}
            dad_sel = (sdata.get('selected') or {}).get('dad', selected)
            dad_rep = replicas.get(dad_sel, {})
            if not isinstance(dad_rep, dict):
                dad_rep = {}
            a254 = (dad_rep.get('areas') or {}).get('A254') or {}

            key = 'bp' if seq['method'] == 'BP' else 'col'
            inventory[name][f'{key}_seq'] = seq['name']
            inventory[name][f'{key}_ppm'] = quant.get('concentration_ppm_direct') or quant.get('concentration_ppm')
            inventory[name][f'{key}_area_total'] = areas.get('total', 0) or 0
            inventory[name][f'{key}_vol'] = seq['volume']
            inventory[name][f'{key}_timeouts'] = ti.get('n_timeouts', 0)
            inventory[name]['sample_type'] = st
            for fr in FRACS + ['LMW']:
                inventory[name][f'{key}_DOC_{fr}'] = areas.get(fr, 0) or 0
            inventory[name][f'{key}_A254_total'] = a254.get('total', 0) or 0
            for fr in FRACS + ['LMW']:
                inventory[name][f'{key}_A254_{fr}'] = a254.get(fr, 0) or 0

    # Filter both modes, SAMPLE only
    both = {n: v for n, v in inventory.items()
            if v.get('col_area_total', 0) > 0 and v.get('bp_area_total', 0) > 0
            and v.get('sample_type') == 'SAMPLE'}

    print("=" * 70)
    print("ANALISI FRACCIONS COLUMN vs TOTAL BP")
    print("=" * 70)
    print(f"Mostres amb ambdos modes: {len(both)}")

    if not both:
        print("Cap mostra!")
        return

    # 1. Correlation: each COLUMN fraction vs BP total area
    print("\n--- CORRELACIO: FRACCIO COLUMN vs AREA TOTAL BP ---")
    print("Quina fraccio COLUMN prediu millor el total BP?\n")

    for fr in FRACS + ['LMW', 'CHROM', 'TOTAL']:
        col_vals = []
        bp_vals = []
        for n, v in both.items():
            if fr == 'CHROM':
                fc = sum(v.get(f'col_DOC_{f}', 0) for f in FRACS)
            elif fr == 'TOTAL':
                fc = v.get('col_area_total', 0)
            else:
                fc = v.get(f'col_DOC_{fr}', 0)
            bt = v.get('bp_area_total', 0)
            if fc > 0 and bt > 0:
                col_vals.append(fc)
                bp_vals.append(bt)
        if len(col_vals) >= 5:
            ca = np.array(col_vals)
            ba = np.array(bp_vals)
            r = np.corrcoef(ca, ba)[0, 1]
            z = np.polyfit(ca, ba, 1)
            print(f"  {fr:5s} vs BP: n={len(col_vals):3d}  R={r:.4f}  "
                  f"R2={r**2:.4f}  slope={z[0]:.3f}  int={z[1]:.1f}")

    # 2. Same for A254
    both_a254 = {n: v for n, v in both.items()
                 if v.get('col_A254_total', 0) > 0 and v.get('bp_A254_total', 0) > 0}

    if both_a254:
        print(f"\n--- CORRELACIO A254: FRACCIO COLUMN vs A254 TOTAL BP ---")
        print(f"Mostres amb A254: {len(both_a254)}\n")
        for fr in FRACS + ['LMW', 'CHROM', 'TOTAL']:
            col_vals = []
            bp_vals = []
            for n, v in both_a254.items():
                if fr == 'CHROM':
                    fc = sum(v.get(f'col_A254_{f}', 0) for f in FRACS)
                elif fr == 'TOTAL':
                    fc = v.get('col_A254_total', 0)
                else:
                    fc = v.get(f'col_A254_{fr}', 0)
                bt = v.get('bp_A254_total', 0)
                if fc > 0 and bt > 0:
                    col_vals.append(fc)
                    bp_vals.append(bt)
            if len(col_vals) >= 5:
                ca = np.array(col_vals)
                ba = np.array(bp_vals)
                r = np.corrcoef(ca, ba)[0, 1]
                z = np.polyfit(ca, ba, 1)
                print(f"  {fr:5s} vs BP_A254: n={len(col_vals):3d}  R={r:.4f}  "
                      f"R2={r**2:.4f}  slope={z[0]:.3f}")

    # 3. Per-pair: fraction distribution + ratio
    print("\n--- PER PARELL: DISTRIBUCIO FRACCIONS COLUMN + RATIO vs BP ---")
    pairs = defaultdict(list)
    for n, v in both.items():
        pairs[(v.get('col_seq', ''), v.get('bp_seq', ''))].append((n, v))

    for (cs, bs), samples in sorted(pairs.items()):
        n = len(samples)
        frac_pcts = {f: [] for f in FRACS + ['LMW']}
        ratios_ppm = []
        ratios_area = []
        for name, v in samples:
            total_col = v.get('col_area_total', 0)
            if total_col > 0:
                for f in FRACS + ['LMW']:
                    frac_pcts[f].append(v.get(f'col_DOC_{f}', 0) / total_col * 100)
            ppm_c = v.get('col_ppm')
            ppm_b = v.get('bp_ppm')
            if ppm_c and ppm_b and ppm_c > 0:
                ratios_ppm.append(ppm_b / ppm_c)
            area_c = v.get('col_area_total', 0)
            area_b = v.get('bp_area_total', 0)
            if area_c > 0 and area_b > 0:
                ratios_area.append(area_b / area_c)

        r_ppm = np.median(ratios_ppm) if ratios_ppm else 0
        r_area = np.median(ratios_area) if ratios_area else 0
        vol_c = samples[0][1].get('col_vol')
        vol_b = samples[0][1].get('bp_vol')
        vol_r = vol_b / vol_c if vol_c and vol_b and vol_c > 0 else None

        print(f"\n  {cs:15s} / {bs:15s}  n={n}  "
              f"ratio_ppm={r_ppm:.3f}  ratio_area={r_area:.3f}  "
              f"vol={vol_c}/{vol_b} (ratio={f'{vol_r:.3f}' if vol_r else '?'})")
        for f in FRACS + ['LMW']:
            if frac_pcts[f]:
                vals = np.array(frac_pcts[f])
                print(f"    {f:5s}: {np.mean(vals):5.1f}% +/- {np.std(vals):.1f}%")
        if ratios_area and vol_r:
            area_norm = np.median(ratios_area) / vol_r
            print(f"    Area BP/COL normalitzat per vol: {area_norm:.3f} (1.0 = perfecte)")

    # 4. KHP cross-check
    print("\n" + "=" * 70)
    print("KHP PER SEQ")
    print("=" * 70)
    for seq in seqs:
        cal = seq.get('calibration')
        if not cal:
            continue
        entries = cal.get('calibrations', [])
        if not entries:
            continue
        print(f"\n  {seq['name']:20s} ({seq['method']:6s}, vol={seq['volume']}uL)")
        for e in entries:
            conc = e.get('conc_ppm', 0)
            area = e.get('area', e.get('area_total', 0))
            area_u = e.get('area_u', 0)
            rf = e.get('rf', e.get('rf_mass', 0))
            rf_u = e.get('rf_u', e.get('rf_mass_u', 0))
            vol = e.get('volume_uL', 0)
            valid = e.get('valid_for_calibration', True)
            anoms = [a.get('code') if isinstance(a, dict) else str(a)
                     for a in (e.get('calibration_anomalies') or [])]
            # Replicas info
            reps = e.get('replicas_info', [])
            n_reps = len(reps) if reps else e.get('n_replicas', '?')
            rsd = e.get('rsd', 0)
            sym = e.get('symmetry', 0)
            fwhm = e.get('fwhm_doc', 0)
            shift = e.get('shift_min', 0)

            v_mark = "V" if valid else "X"
            anom_str = " ".join(anoms) if anoms else "OK"
            print(f"    [{v_mark}] {conc}ppm vol={vol}uL: "
                  f"area={area:.0f} RF={rf:.0f} "
                  f"reps={n_reps} RSD={rsd:.1f}% "
                  f"sym={sym:.2f} FWHM={fwhm:.2f} "
                  f"shift={shift:.2f}min  {anom_str}")

            # Replica details if available
            if reps and isinstance(reps, list):
                for ri, rep in enumerate(reps):
                    if isinstance(rep, dict):
                        ra = rep.get('area', 0)
                        rt = rep.get('t_retention', 0)
                        rrf = rep.get('rf', 0)
                        print(f"      R{ri+1}: area={ra:.0f} t_ret={rt:.2f}min RF={rrf:.0f}")

    # 5. KHP paired: same conc COL vs BP
    print("\n--- KHP PARELLS (mateixa concentracio COL vs BP) ---")
    # Group KHP by conc
    khp_by_conc = defaultdict(lambda: {'col': [], 'bp': []})
    for seq in seqs:
        cal = seq.get('calibration')
        if not cal:
            continue
        entries = cal.get('calibrations', [])
        for e in entries:
            conc = e.get('conc_ppm', 0)
            area = e.get('area', e.get('area_total', 0))
            vol = e.get('volume_uL', 0)
            rf = e.get('rf', e.get('rf_mass', 0))
            valid = e.get('valid_for_calibration', True)
            mode = 'bp' if seq['method'] == 'BP' else 'col'
            khp_by_conc[conc][mode].append({
                'seq': seq['name'], 'area': area, 'vol': vol,
                'rf': rf, 'valid': valid
            })

    for conc in sorted(khp_by_conc.keys()):
        data = khp_by_conc[conc]
        if data['col'] and data['bp']:
            print(f"\n  KHP {conc} ppm:")
            for mode, label in [('col', 'COLUMN'), ('bp', 'BP')]:
                entries = data[mode]
                areas = [e['area'] for e in entries if e['area'] > 0]
                rfs = [e['rf'] for e in entries if e['rf'] > 0]
                vols = [e['vol'] for e in entries]
                print(f"    {label:6s}: n={len(entries)}  "
                      f"area={np.mean(areas):.0f}+/-{np.std(areas):.0f}  "
                      f"RF={np.mean(rfs):.0f}+/-{np.std(rfs):.0f}  "
                      f"vol={set(vols)}")
            # Area ratio normalized by volume
            area_col = np.mean([e['area'] for e in data['col'] if e['area'] > 0])
            area_bp = np.mean([e['area'] for e in data['bp'] if e['area'] > 0])
            vol_col = data['col'][0]['vol']
            vol_bp = data['bp'][0]['vol']
            if area_col > 0 and vol_col > 0 and vol_bp > 0:
                ratio_area = area_bp / area_col
                ratio_norm = ratio_area / (vol_bp / vol_col)
                print(f"    Area BP/COL: {ratio_area:.3f}  "
                      f"Normalitzat (div vol {vol_bp}/{vol_col}): {ratio_norm:.3f}")

    # 6. Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle("Fraccions COLUMN vs Total BP", fontsize=14, fontweight="bold")

        colors_frac = {'BioP': '#3498DB', 'HS': '#E74C3C', 'BB': '#F39C12', 'SB': '#2ECC71'}

        # 1-4: Each fraction vs BP total
        for i, fr in enumerate(FRACS):
            ax = axes[i // 2, i % 2] if i < 4 else None
            if ax is None:
                break
            col_vals = []
            bp_vals = []
            for n, v in both.items():
                fc = v.get(f'col_DOC_{fr}', 0)
                bt = v.get('bp_area_total', 0)
                if fc > 0 and bt > 0:
                    col_vals.append(fc)
                    bp_vals.append(bt)
            if col_vals:
                ca = np.array(col_vals)
                ba = np.array(bp_vals)
                ax.scatter(ca, ba, c=colors_frac[fr], s=15, alpha=0.6, edgecolors='none')
                if len(ca) >= 3:
                    z = np.polyfit(ca, ba, 1)
                    r2 = np.corrcoef(ca, ba)[0, 1] ** 2
                    xf = np.linspace(0, ca.max() * 1.1, 50)
                    ax.plot(xf, np.polyval(z, xf), 'r-', alpha=0.5)
                    ax.set_title(f"COL {fr} vs BP total\nR2={r2:.3f} slope={z[0]:.2f}", fontsize=9)
            ax.set_xlabel(f"COL DOC {fr}")
            ax.set_ylabel("BP DOC total")

        # 5: CHROM (sum BioP+HS+BB+SB) vs BP total
        ax = axes[1, 0]
        col_vals = []
        bp_vals = []
        for n, v in both.items():
            cc = sum(v.get(f'col_DOC_{f}', 0) for f in FRACS)
            bt = v.get('bp_area_total', 0)
            if cc > 0 and bt > 0:
                col_vals.append(cc)
                bp_vals.append(bt)
        if col_vals:
            ca = np.array(col_vals)
            ba = np.array(bp_vals)
            ax.scatter(ca, ba, c='#228be6', s=15, alpha=0.6, edgecolors='none')
            if len(ca) >= 3:
                z = np.polyfit(ca, ba, 1)
                r2 = np.corrcoef(ca, ba)[0, 1] ** 2
                xf = np.linspace(0, ca.max() * 1.1, 50)
                ax.plot(xf, np.polyval(z, xf), 'r-', alpha=0.5)
                ax.set_title(f"COL chrom vs BP total\nR2={r2:.3f} slope={z[0]:.2f}", fontsize=9)
        ax.set_xlabel("COL DOC chrom (BioP+HS+BB+SB)")
        ax.set_ylabel("BP DOC total")

        # 6: KHP area COL vs BP
        ax = axes[1, 1]
        khp_col_areas = []
        khp_bp_areas = []
        khp_labels = []
        for seq in seqs:
            cal = seq.get('calibration')
            if not cal:
                continue
            for e in cal.get('calibrations', []):
                area = e.get('area', 0)
                if area > 0:
                    mode = 'bp' if seq['method'] == 'BP' else 'col'
                    if mode == 'col':
                        khp_col_areas.append(area)
                        khp_labels.append(seq['name'])
                    else:
                        khp_bp_areas.append(area)

        if khp_col_areas and khp_bp_areas:
            ax.bar(range(len(khp_col_areas)), khp_col_areas, alpha=0.7,
                   color='#228be6', label='COLUMN')
            ax.bar(range(len(khp_col_areas), len(khp_col_areas) + len(khp_bp_areas)),
                   khp_bp_areas, alpha=0.7, color='#E74C3C', label='BP')
            ax.set_title("KHP areas per SEQ", fontsize=9)
            ax.legend(fontsize=7)
            ax.set_ylabel("Area KHP")

        # 7: Fraction distribution stacked per pair
        ax = axes[1, 2]
        pair_names = []
        pair_data = defaultdict(list)
        for (cs, bs), samples in sorted(pairs.items()):
            pair_names.append(f"{cs[:6]}/{bs[:6]}")
            for f in FRACS + ['LMW']:
                pcts = []
                for name, v in samples:
                    total = v.get('col_area_total', 0)
                    if total > 0:
                        pcts.append(v.get(f'col_DOC_{f}', 0) / total * 100)
                pair_data[f].append(np.mean(pcts) if pcts else 0)

        x = np.arange(len(pair_names))
        bottom = np.zeros(len(pair_names))
        frac_colors = {'BioP': '#3498DB', 'HS': '#E74C3C', 'BB': '#F39C12',
                       'SB': '#2ECC71', 'LMW': '#9B59B6'}
        for f in FRACS + ['LMW']:
            vals = np.array(pair_data[f])
            ax.bar(x, vals, bottom=bottom, color=frac_colors[f], label=f, width=0.7)
            bottom += vals
        ax.set_xticks(x)
        ax.set_xticklabels(pair_names, rotation=45, ha='right', fontsize=7)
        ax.set_title("Distribucio fraccions COL per parell", fontsize=9)
        ax.legend(fontsize=6, loc='upper right')
        ax.set_ylabel("%")

        for a in axes.flat:
            a.tick_params(labelsize=7)
            a.spines["top"].set_visible(False)
            a.spines["right"].set_visible(False)

        fig.tight_layout()
        out = Path("_results/diag_col_bp")
        out.mkdir(parents=True, exist_ok=True)
        p = out / "fractions_vs_bp.png"
        fig.savefig(str(p), dpi=150, bbox_inches="tight")
        print(f"\nPlot: {p}")
        plt.close(fig)

    except ImportError:
        print("matplotlib no disponible")

    print("\nFet!")


if __name__ == "__main__":
    main()
