"""Deep diagnostic: COL vs BP using only chromatographic fractions (BioP+HS+BB+SB)."""
import os, json, csv
import numpy as np
from collections import defaultdict
from pathlib import Path
from hpsec_config import get_data_folders
from hpsec_consolidate import detect_seq_type

FRACS_CHROM = ['BioP', 'HS', 'BB', 'SB']

def main():
    folders = get_data_folders()

    # Scan SEQs
    seqs = []
    for folder in folders:
        for d in sorted(os.listdir(folder)):
            full = os.path.join(folder, d)
            jp = os.path.join(full, 'CHECK', 'data', 'analysis_result.json')
            if os.path.exists(jp):
                seqs.append({'name': d, 'path': full, 'method': detect_seq_type(d)})

    print(f"SEQs: {sum(1 for s in seqs if s['method']=='COLUMN')} COL, "
          f"{sum(1 for s in seqs if s['method']=='BP')} BP")

    # Build inventory
    inventory = defaultdict(dict)
    for seq in seqs:
        with open(os.path.join(seq['path'], 'CHECK', 'data', 'analysis_result.json'),
                  encoding='utf-8') as f:
            data = json.load(f)
        sg = data.get('samples_grouped', {})

        # Volume
        mp = os.path.join(seq['path'], 'CHECK', 'data', 'import_manifest.json')
        vol = None
        if os.path.exists(mp):
            with open(mp, encoding='utf-8') as f:
                m = json.load(f)
            vol = (m.get('sequence') or {}).get('injection_volume')
            if not vol:
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

        for name, sdata in sg.items():
            st = sdata.get('sample_type', 'SAMPLE')
            if st not in ('SAMPLE', 'KHP'):
                continue

            selected = sdata.get('selected') or {}
            doc_sel = selected.get('doc', '1')
            dad_sel = selected.get('dad', doc_sel)
            replicas = sdata.get('replicas') or {}
            doc_rep = replicas.get(doc_sel, {})
            if doc_sel == 'comp':
                for rv in replicas.values():
                    if isinstance(rv, dict) and rv.get('timeout_composition'):
                        doc_rep = rv
                        break
            dad_rep = replicas.get(dad_sel, {})
            if not isinstance(doc_rep, dict):
                doc_rep = {}
            if not isinstance(dad_rep, dict):
                dad_rep = {}

            areas_doc = (doc_rep.get('areas') or {}).get('DOC') or {}
            areas_a254 = (dad_rep.get('areas') or {}).get('A254') or {}
            quant = sdata.get('quantification') or {}
            timeout_info = doc_rep.get('timeout_info') or {}

            anomalies = []
            for rd in replicas.values():
                if isinstance(rd, dict):
                    for a in rd.get('anomalies', []):
                        code = a.get('code') if isinstance(a, dict) else str(a)
                        if code not in anomalies:
                            anomalies.append(code)

            key = 'bp' if seq['method'] == 'BP' else 'col'
            inventory[name][f'{key}_seq'] = seq['name']
            inventory[name][f'{key}_vol'] = vol
            inventory[name][f'{key}_ppm'] = (quant.get('concentration_ppm_direct')
                                              or quant.get('concentration_ppm'))
            inventory[name][f'{key}_timeouts'] = timeout_info.get('n_timeouts', 0)
            inventory[name][f'{key}_anomalies'] = '; '.join(anomalies)
            inventory[name]['sample_type'] = st

            for f in FRACS_CHROM + ['LMW']:
                inventory[name][f'{key}_DOC_{f}'] = areas_doc.get(f, 0) or 0
                inventory[name][f'{key}_A254_{f}'] = areas_a254.get(f, 0) or 0

            inventory[name][f'{key}_doc_total'] = areas_doc.get('total', 0) or 0
            inventory[name][f'{key}_a254_total'] = areas_a254.get('total', 0) or 0

    # Filter both modes
    both = {n: v for n, v in inventory.items()
            if 'col_doc_total' in v and 'bp_doc_total' in v}

    print(f"\nMostres amb ambdos modes: {len(both)}")
    print(f"  SAMPLE: {sum(1 for v in both.values() if v.get('sample_type')=='SAMPLE')}")
    print(f"  KHP: {sum(1 for v in both.values() if v.get('sample_type')=='KHP')}")

    # Build rows
    rows = []
    for name in sorted(both.keys()):
        v = both[name]

        doc_chrom_col = sum(v.get(f'col_DOC_{f}', 0) for f in FRACS_CHROM)
        doc_chrom_bp = sum(v.get(f'bp_DOC_{f}', 0) for f in FRACS_CHROM)
        a254_chrom_col = sum(v.get(f'col_A254_{f}', 0) for f in FRACS_CHROM)
        a254_chrom_bp = sum(v.get(f'bp_A254_{f}', 0) for f in FRACS_CHROM)

        ppm_col = v.get('col_ppm')
        ppm_bp = v.get('bp_ppm')
        vol_col = v.get('col_vol')
        vol_bp = v.get('bp_vol')
        vol_ratio = vol_bp / vol_col if vol_col and vol_bp and vol_col > 0 else None

        ratio_ppm = ppm_bp / ppm_col if ppm_col and ppm_bp and ppm_col > 0 else None
        ratio_area = doc_chrom_bp / doc_chrom_col if doc_chrom_col > 0 else None
        ratio_a254 = a254_chrom_bp / a254_chrom_col if a254_chrom_col > 0 else None
        ratio_a254_norm = ratio_a254 / vol_ratio if ratio_a254 and vol_ratio else None

        pct_lmw_col = (v.get('col_DOC_LMW', 0) / v['col_doc_total'] * 100
                       if v['col_doc_total'] > 0 else 0)
        pct_lmw_bp = (v.get('bp_DOC_LMW', 0) / v['bp_doc_total'] * 100
                      if v['bp_doc_total'] > 0 else 0)

        rows.append({
            'name': name, 'type': v.get('sample_type', 'SAMPLE'),
            'col_seq': v.get('col_seq', ''), 'bp_seq': v.get('bp_seq', ''),
            'ppm_col': ppm_col, 'ppm_bp': ppm_bp, 'ratio_ppm': ratio_ppm,
            'doc_chrom_col': doc_chrom_col, 'doc_chrom_bp': doc_chrom_bp,
            'ratio_area_chrom': ratio_area,
            'a254_chrom_col': a254_chrom_col, 'a254_chrom_bp': a254_chrom_bp,
            'ratio_a254_chrom': ratio_a254, 'ratio_a254_norm': ratio_a254_norm,
            'pct_lmw_col': pct_lmw_col, 'pct_lmw_bp': pct_lmw_bp,
            'vol_col': vol_col, 'vol_bp': vol_bp, 'vol_ratio': vol_ratio,
            'to_col': v.get('col_timeouts', 0), 'to_bp': v.get('bp_timeouts', 0),
            'anom_col': v.get('col_anomalies', ''), 'anom_bp': v.get('bp_anomalies', ''),
            **{f'DOC_{f}_col': v.get(f'col_DOC_{f}', 0) for f in FRACS_CHROM},
            **{f'DOC_{f}_bp': v.get(f'bp_DOC_{f}', 0) for f in FRACS_CHROM},
            **{f'A254_{f}_col': v.get(f'col_A254_{f}', 0) for f in FRACS_CHROM},
            **{f'A254_{f}_bp': v.get(f'bp_A254_{f}', 0) for f in FRACS_CHROM},
        })

    # Save CSV
    out = Path('_results/diag_col_bp')
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'col_bp_chrom_fracs.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys(), delimiter=';')
        w.writeheader()
        w.writerows(rows)
    print(f"CSV: {out / 'col_bp_chrom_fracs.csv'} ({len(rows)} files)")

    # === ANALYSIS ===
    valid = [r for r in rows if r['ratio_ppm'] is not None]
    if not valid:
        print("Cap mostra valida!")
        return

    ratios_ppm = np.array([r['ratio_ppm'] for r in valid])

    print(f"\n{'='*70}")
    print(f"ANALISI FRACCIONS CROMATOGRAFIQUES (BioP+HS+BB+SB, sense LMW)")
    print(f"{'='*70}")
    print(f"Mostres: {len(valid)}")

    # Global
    print(f"\n--- RATIOS GLOBALS ---")
    print(f"ppm BP/COL:          med={np.median(ratios_ppm):.3f}  mean={np.mean(ratios_ppm):.3f}  SD={np.std(ratios_ppm):.3f}")

    ra = [r['ratio_area_chrom'] for r in valid if r['ratio_area_chrom']]
    if ra:
        ra = np.array(ra)
        print(f"Area DOC chrom:      med={np.median(ra):.3f}  mean={np.mean(ra):.3f}  SD={np.std(ra):.3f}")

    ra254 = [r['ratio_a254_chrom'] for r in valid if r['ratio_a254_chrom']]
    if ra254:
        ra254 = np.array(ra254)
        print(f"Area A254 chrom:     med={np.median(ra254):.3f}  mean={np.mean(ra254):.3f}  SD={np.std(ra254):.3f}")

    ran = [r['ratio_a254_norm'] for r in valid if r['ratio_a254_norm']]
    if ran:
        ran = np.array(ran)
        print(f"A254 norm (div vol): med={np.median(ran):.3f}  mean={np.mean(ran):.3f}  SD={np.std(ran):.3f}")

    # LMW %
    lmw_c = np.array([r['pct_lmw_col'] for r in valid])
    lmw_b = np.array([r['pct_lmw_bp'] for r in valid])
    print(f"\n%LMW COLUMN: med={np.median(lmw_c):.1f}%  mean={np.mean(lmw_c):.1f}%  range={np.min(lmw_c):.1f}-{np.max(lmw_c):.1f}%")
    print(f"%LMW BP:     med={np.median(lmw_b):.1f}%  mean={np.mean(lmw_b):.1f}%  range={np.min(lmw_b):.1f}-{np.max(lmw_b):.1f}%")

    # Per fraction DOC
    print(f"\n--- RATIO PER FRACCIO DOC (area BP/COL) ---")
    for f in FRACS_CHROM:
        fc = np.array([r[f'DOC_{f}_col'] for r in valid])
        fb = np.array([r[f'DOC_{f}_bp'] for r in valid])
        mask = (fc > 0) & (fb > 0)
        if mask.any():
            rf = fb[mask] / fc[mask]
            print(f"  {f:5s}: n={mask.sum():3d}  med={np.median(rf):.3f}  mean={np.mean(rf):.3f}  SD={np.std(rf):.3f}  range={np.min(rf):.3f}-{np.max(rf):.3f}")

    # Per fraction A254 normalized
    print(f"\n--- RATIO PER FRACCIO A254 (normalitzat per volum) ---")
    for f in FRACS_CHROM:
        fc = np.array([r[f'A254_{f}_col'] for r in valid])
        fb = np.array([r[f'A254_{f}_bp'] for r in valid])
        vr = np.array([r['vol_ratio'] or 1 for r in valid])
        mask = (fc > 0) & (fb > 0) & (vr > 0)
        if mask.any():
            rf = (fb[mask] / fc[mask]) / vr[mask]
            print(f"  {f:5s}: n={mask.sum():3d}  med={np.median(rf):.3f}  mean={np.mean(rf):.3f}  SD={np.std(rf):.3f}")

    # Per SEQ pair
    print(f"\n--- PER PARELL SEQ ---")
    pairs = defaultdict(list)
    for r in valid:
        pairs[(r['col_seq'], r['bp_seq'])].append(r)

    for (cs, bs), pr in sorted(pairs.items()):
        pr_ppm = np.array([r['ratio_ppm'] for r in pr])
        pr_area = [r['ratio_area_chrom'] for r in pr if r['ratio_area_chrom']]
        pr_a254n = [r['ratio_a254_norm'] for r in pr if r['ratio_a254_norm']]
        n_out = sum(1 for r in pr if r['ratio_ppm'] < 0.85 or r['ratio_ppm'] > 1.15)
        lmw_c = np.mean([r['pct_lmw_col'] for r in pr])
        lmw_b = np.mean([r['pct_lmw_bp'] for r in pr])

        line = f"  {cs:15s}/{bs:15s}: n={len(pr):2d}"
        line += f"  ppm={np.median(pr_ppm):.3f}+/-{np.std(pr_ppm):.3f}"
        if pr_area:
            line += f"  area_chrom={np.median(pr_area):.3f}"
        if pr_a254n:
            line += f"  A254n={np.median(pr_a254n):.3f}"
        line += f"  LMW={lmw_c:.0f}/{lmw_b:.0f}%  out={n_out}"
        print(line)

    # Clean vs dirty
    clean = [r for r in valid if 0.85 <= r['ratio_ppm'] <= 1.15]
    dirty = [r for r in valid if r['ratio_ppm'] < 0.85 or r['ratio_ppm'] > 1.15]

    print(f"\n--- NETS vs OUTLIERS ---")
    print(f"Nets (0.85-1.15): {len(clean)}/{len(valid)} ({len(clean)/len(valid)*100:.0f}%)")
    print(f"Outliers: {len(dirty)}/{len(valid)} ({len(dirty)/len(valid)*100:.0f}%)")

    if clean:
        cr = np.array([r['ratio_ppm'] for r in clean])
        print(f"  Nets ppm: med={np.median(cr):.3f}  SD={np.std(cr):.3f}")

    # Anomaly correlation
    print(f"\n--- CORRELACIO ANOMALIES ---")
    for label, subset in [("Outliers", dirty), ("Nets", clean)]:
        n = len(subset)
        if n == 0:
            continue
        n_to_col = sum(1 for r in subset if r['to_col'] > 0)
        n_to_bp = sum(1 for r in subset if r['to_bp'] > 0)
        n_ir_col = sum(1 for r in subset if 'IRREGULAR_TOP' in r['anom_col'])
        n_ir_bp = sum(1 for r in subset if 'IRREGULAR_TOP' in r['anom_bp'])
        n_any_anom = sum(1 for r in subset if r['anom_col'] or r['anom_bp'])
        print(f"  {label:8s} (n={n}): TO_COL={n_to_col} TO_BP={n_to_bp} "
              f"IRREG_COL={n_ir_col} IRREG_BP={n_ir_bp} qualsevol={n_any_anom}")

    # Detailed outliers
    if dirty:
        print(f"\n--- DETALL OUTLIERS (top 20) ---")
        for r in sorted(dirty, key=lambda x: abs(x['ratio_ppm'] - 1.0), reverse=True)[:20]:
            flags = []
            if r['to_col'] > 0: flags.append(f"TO_C({r['to_col']})")
            if r['to_bp'] > 0: flags.append(f"TO_B({r['to_bp']})")
            if 'IRREGULAR_TOP' in r['anom_col']: flags.append("IRR_C")
            if 'IRREGULAR_TOP' in r['anom_bp']: flags.append("IRR_B")
            if 'TIMEOUT_IN_PEAK' in r['anom_col'] or 'TIMEOUT_IN_PEAK' in r['anom_bp']:
                flags.append("TO_PIC")
            print(f"  {r['name']:20s} [{r['type']:3s}] {r['col_seq']:12s}/{r['bp_seq']:12s}  "
                  f"ratio={r['ratio_ppm']:.3f}  ppm={r['ppm_col']:.2f}/{r['ppm_bp']:.2f}  "
                  f"LMW={r['pct_lmw_col']:.0f}/{r['pct_lmw_bp']:.0f}%  {' '.join(flags)}")

    # === PLOTS ===
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f"COL vs BP — Fraccions cromatografiques (BioP+HS+BB+SB)\n"
                     f"{len(valid)} mostres, sense LMW", fontsize=13, fontweight="bold")

        colors = ['#E74C3C' if r['type']=='KHP' else '#228be6' for r in valid]
        col_ppms = np.array([r['ppm_col'] or 0 for r in valid])
        bp_ppms = np.array([r['ppm_bp'] or 0 for r in valid])

        # 1. Scatter ppm
        ax = axes[0, 0]
        ax.scatter(col_ppms, bp_ppms, c=colors, s=20, alpha=0.6, edgecolors='none')
        mx = max(col_ppms.max(), bp_ppms.max()) * 1.1
        ax.plot([0, mx], [0, mx], "k--", alpha=0.3)
        if len(col_ppms) >= 3:
            z = np.polyfit(col_ppms, bp_ppms, 1)
            r2 = np.corrcoef(col_ppms, bp_ppms)[0,1]**2
            xf = np.linspace(0, mx, 50)
            ax.plot(xf, np.polyval(z, xf), "r-", alpha=0.5)
            ax.set_title(f"ppm (slope={z[0]:.3f}, R2={r2:.3f})", fontsize=9)
        ax.set_xlabel("ppm COL"); ax.set_ylabel("ppm BP")

        # 2. Scatter area DOC chrom
        ax = axes[0, 1]
        ac = np.array([r['doc_chrom_col'] for r in valid])
        ab = np.array([r['doc_chrom_bp'] for r in valid])
        mask = (ac > 0) & (ab > 0)
        if mask.any():
            ax.scatter(ac[mask], ab[mask], c=[c for c,m in zip(colors,mask) if m], s=20, alpha=0.6)
            mx = max(ac[mask].max(), ab[mask].max()) * 1.1
            ax.plot([0, mx], [0, mx], "k--", alpha=0.3)
            if mask.sum() >= 3:
                z = np.polyfit(ac[mask], ab[mask], 1)
                r2 = np.corrcoef(ac[mask], ab[mask])[0,1]**2
                ax.plot(np.linspace(0,mx,50), np.polyval(z,np.linspace(0,mx,50)), "r-", alpha=0.5)
                ax.set_title(f"DOC chrom (sl={z[0]:.3f}, R2={r2:.3f})", fontsize=9)
        ax.set_xlabel("DOC chrom COL"); ax.set_ylabel("DOC chrom BP")

        # 3. Scatter A254 chrom
        ax = axes[0, 2]
        ac254 = np.array([r['a254_chrom_col'] for r in valid])
        ab254 = np.array([r['a254_chrom_bp'] for r in valid])
        mask254 = (ac254 > 0) & (ab254 > 0)
        if mask254.any():
            ax.scatter(ac254[mask254], ab254[mask254],
                      c=[c for c,m in zip(colors,mask254) if m], s=20, alpha=0.6)
            mx = max(ac254[mask254].max(), ab254[mask254].max()) * 1.1
            ax.plot([0, mx], [0, mx * 0.25], "g--", alpha=0.4, label="vol ratio 0.25")
            if mask254.sum() >= 3:
                z = np.polyfit(ac254[mask254], ab254[mask254], 1)
                r2 = np.corrcoef(ac254[mask254], ab254[mask254])[0,1]**2
                ax.plot(np.linspace(0,mx,50), np.polyval(z,np.linspace(0,mx,50)), "r-", alpha=0.5)
                ax.set_title(f"A254 chrom (sl={z[0]:.3f}, R2={r2:.3f})", fontsize=9)
            ax.legend(fontsize=7)
        ax.set_xlabel("A254 chrom COL"); ax.set_ylabel("A254 chrom BP")

        # 4. Ratio ppm vs concentration
        ax = axes[0, 3]
        ax.scatter(col_ppms, ratios_ppm, c=colors, s=20, alpha=0.6, edgecolors='none')
        ax.axhline(1.0, color="k", ls="--", alpha=0.3)
        ax.axhline(0.85, color="r", ls="--", alpha=0.2)
        ax.axhline(1.15, color="r", ls="--", alpha=0.2)
        ax.set_title("Ratio ppm vs concentracio", fontsize=9)
        ax.set_xlabel("ppm COL"); ax.set_ylabel("Ratio BP/COL")
        ax.set_ylim(0, min(3.5, ratios_ppm.max() * 1.1))

        # 5-8. Per-fraction scatter DOC
        for i, f in enumerate(FRACS_CHROM):
            ax = axes[1, i]
            fc = np.array([r[f'DOC_{f}_col'] for r in valid])
            fb = np.array([r[f'DOC_{f}_bp'] for r in valid])
            mask = (fc > 0) & (fb > 0)
            if mask.any():
                ax.scatter(fc[mask], fb[mask],
                          c=[c for c,m in zip(colors,mask) if m], s=15, alpha=0.5)
                mx = max(fc[mask].max(), fb[mask].max()) * 1.1
                ax.plot([0, mx], [0, mx], "k--", alpha=0.3, lw=0.8)
                if mask.sum() >= 3:
                    z = np.polyfit(fc[mask], fb[mask], 1)
                    r2 = np.corrcoef(fc[mask], fb[mask])[0,1]**2
                    ax.set_title(f"DOC {f} (sl={z[0]:.2f}, R2={r2:.3f})", fontsize=9)
                else:
                    ax.set_title(f"DOC {f}", fontsize=9)
            else:
                ax.set_title(f"DOC {f} (no data)", fontsize=9)
            ax.set_xlabel("COL"); ax.set_ylabel("BP")

        for a in axes.flat:
            a.tick_params(labelsize=7)
            a.spines["top"].set_visible(False)
            a.spines["right"].set_visible(False)

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        p = out / "col_bp_chrom_fracs.png"
        fig.savefig(str(p), dpi=150, bbox_inches="tight")
        print(f"\nPlot: {p}")
        plt.close(fig)

        # A254 per fraction plot
        fig2, axes2 = plt.subplots(1, 4, figsize=(16, 4))
        fig2.suptitle("A254 per fraccio (normalitzat per volum)", fontsize=12, fontweight="bold")
        for i, f in enumerate(FRACS_CHROM):
            ax = axes2[i]
            fc = np.array([r[f'A254_{f}_col'] for r in valid])
            fb = np.array([r[f'A254_{f}_bp'] for r in valid])
            vr = np.array([r['vol_ratio'] or 0.25 for r in valid])
            mask = (fc > 0) & (fb > 0) & (vr > 0)
            if mask.any():
                rf = (fb[mask] / fc[mask]) / vr[mask]
                ax.hist(rf, bins=min(20, len(rf)), color="#40c057", alpha=0.7, edgecolor="white")
                ax.axvline(np.median(rf), color="red", ls="-", alpha=0.6,
                          label=f"med={np.median(rf):.2f}")
                ax.axvline(1.0, color="k", ls="--", alpha=0.3, label="teoric 1.0")
                ax.legend(fontsize=7)
            ax.set_title(f"A254 {f}", fontsize=9)
            ax.set_xlabel("Ratio norm")
            ax.tick_params(labelsize=7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        fig2.tight_layout()
        p2 = out / "col_bp_a254_fracs.png"
        fig2.savefig(str(p2), dpi=150, bbox_inches="tight")
        print(f"Plot A254: {p2}")
        plt.close(fig2)

    except ImportError:
        print("matplotlib no disponible")

    print("\nFet!")


if __name__ == "__main__":
    main()
