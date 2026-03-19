"""Genera PDF amb cromatogrames BP+COLUMN per cada parell.
BP: mostra tangent (verd) i trapezoid (lila) amb límits.
COLUMN: mostra fraccions colorades."""

import json, os, csv, numpy as np
from hpsec_import import import_sequence
from hpsec_core import find_peak_boundaries
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

data_dir = 'C:/Users/Lequia/Desktop/Dades3'
fracs_col = {
    'BioP': (10.8, 18, '#E74C3C'), 'HS': (18, 23, '#F39C12'),
    'BB': (23, 26, '#27AE60'), 'SB': (26, 32, '#3498DB'),
    'LMW': (32, 70, '#95A5A6'),
}

# Llegir CSV per saber quins parells i filtrar ppm<6
pairs = []
with open('C:/Users/Lequia/Desktop/HPSEC/_comparativa_bp_col.csv', 'r', encoding='utf-8') as f:
    for r in csv.DictReader(f):
        if float(r['ppm_COL']) <= 6 and float(r['ppm_trap']) <= 6:
            pairs.append(r)

print(f"{len(pairs)} parells a graficar")

# Agrupar per SEQ_BP per importar una sola vegada
bp_seqs = sorted(set(r['SEQ_BP'] for r in pairs))
bp_cache = {}  # seq -> {name -> {t, y_net, vol}}

for seq in bp_seqs:
    path = os.path.join(data_dir, seq)
    print(f"Important {seq}...", end=" ", flush=True)
    try:
        result = import_sequence(path)
    except Exception as e:
        print(f"ERROR: {e}")
        continue
    samples = result.get('samples', {})
    bp_cache[seq] = {}
    n = 0
    for sn, sd in samples.items():
        reps_data = []
        for rk, rd in sorted(sd.get('replicas', {}).items()):
            d = rd.get('direct', {})
            t = np.array(d.get('t', []))
            y_net = np.array(d.get('y_net', []))
            if len(t) < 10:
                continue
            vol = 100
            inj = rd.get('injection_info', {})
            if inj and inj.get('inj_volume'):
                vol = inj['inj_volume']
            reps_data.append({'t': t, 'y_net': y_net, 'vol': vol, 'rep': rk})
        if reps_data:
            bp_cache[seq][sn] = reps_data
            n += 1
    print(f"{n} mostres")

# Cache COLUMN: llegir analysis JSON
col_cache = {}  # seq -> {name -> {t_doc, y_doc_net, areas, vol}}
col_seqs = sorted(set(r['SEQ_COL'] for r in pairs))
for seq in col_seqs:
    ana = os.path.join(data_dir, seq, 'CHECK', 'data', 'analysis_result.json')
    if not os.path.exists(ana):
        continue
    with open(ana, 'r', encoding='utf-8') as f:
        a = json.load(f)
    col_cache[seq] = {}
    for s in a.get('samples', []):
        name = s.get('name', '')
        t = s.get('t_doc', [])
        y = s.get('y_doc_net', [])
        areas = s.get('areas', {}).get('DOC', {})
        vol = s.get('inj_volume') or 400
        if t and y and len(t) > 10:
            col_cache[seq][name] = {
                't': np.array(t), 'y': np.array(y),
                'areas': areas, 'vol': vol,
            }

# Generar PDF
pdf_path = 'C:/Users/Lequia/Desktop/HPSEC/_cromatogrames_bp_col.pdf'
with PdfPages(pdf_path) as pdf:
    for i, row in enumerate(pairs):
        name = row['Mostra']
        seq_bp = row['SEQ_BP']
        seq_col = row['SEQ_COL']
        r_tang = float(row['ratio_tang'])
        r_trap = float(row['ratio_trap'])

        bp_reps = bp_cache.get(seq_bp, {}).get(name, [])
        col_data = col_cache.get(seq_col, {}).get(name)

        if not bp_reps and not col_data:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        # Tag
        if r_trap < 0.8:
            tag, tc = f'BAIX {r_trap:.2f}', '#2E86C1'
        elif r_trap > 1.2:
            tag, tc = f'ALT {r_trap:.2f}', '#E74C3C'
        else:
            tag, tc = f'OK {r_trap:.2f}', '#27AE60'

        # BP
        ax = axes[0]
        for rd in bp_reps:
            t = rd['t']
            y = rd['y_net']
            pk = int(np.argmax(y))
            try:
                li, ri = find_peak_boundaries(t, y, pk)
            except:
                li, ri = 0, len(t) - 1

            area_tang = float(np.trapezoid(np.maximum(y[li:ri+1], 0), t[li:ri+1]))
            area_trap = float(np.trapezoid(np.maximum(y, 0), t))

            ax.plot(t, y, '-', lw=0.6, alpha=0.8)
            # Tangent area
            ax.fill_between(t[li:ri+1], 0, np.maximum(y[li:ri+1], 0),
                           alpha=0.2, color='#27AE60')
            # Fora tangent
            if li > 0:
                ax.fill_between(t[:li+1], 0, np.maximum(y[:li+1], 0),
                               alpha=0.1, color='#E74C3C')
            if ri < len(t) - 1:
                ax.fill_between(t[ri:], 0, np.maximum(y[ri:], 0),
                               alpha=0.1, color='#E74C3C')
            ax.axvline(t[li], c='#27AE60', lw=0.8, alpha=0.6)
            ax.axvline(t[ri], c='#27AE60', lw=0.8, alpha=0.6)

        ax.axhline(0, c='k', lw=0.3, alpha=0.3)
        ax.set_xlim(0, 10)
        ax.set_xlabel('min', fontsize=8)
        ax.set_ylabel('ppb', fontsize=8)
        a_tang = float(row['area_BP_tangent'])
        a_trap = float(row['area_BP_trapezoid'])
        ax.set_title(f'BP {name} ({seq_bp[:9]})\n'
                     f'tang={a_tang:.0f} trap={a_trap:.0f} '
                     f'diff={((a_trap-a_tang)/a_tang*100) if a_tang > 0 else 0:+.0f}%',
                     fontsize=8)
        ax.grid(True, alpha=0.1)
        ax.annotate(tag, (0.02, 0.92), xycoords='axes fraction', fontsize=10,
                   fontweight='bold', color=tc,
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                            edgecolor=tc, alpha=0.9))

        # COLUMN
        ax = axes[1]
        if col_data:
            t = col_data['t']
            y = col_data['y']
            areas = col_data['areas']
            total = areas.get('total', 0)
            vol = col_data['vol']

            ax.plot(t, y, 'b-', lw=0.5, alpha=0.8)
            ax.axhline(0, c='k', lw=0.3, alpha=0.3)
            for fn, (t0, t1, c) in fracs_col.items():
                mask = (t >= t0) & (t <= t1)
                if mask.any():
                    ax.fill_between(t[mask], 0, np.maximum(y[mask], 0),
                                   alpha=0.12, color=c)
                    ax.axvline(t0, c=c, lw=0.3, ls='--', alpha=0.3)

            chrom = areas.get('BioP', 0) + areas.get('HS', 0) + areas.get('BB', 0)
            lmw = areas.get('LMW', 0)
            sb = areas.get('SB', 0)
            ax.set_xlim(0, min(t[-1], 80))
            cp = chrom / total * 100 if total > 0 else 0
            sp = (sb + lmw) / total * 100 if total > 0 else 0
            ax.set_title(f'COL {name} ({seq_col[:9]})\n'
                        f'total={total:.0f} chrom={cp:.0f}% SB+LMW={sp:.0f}%',
                        fontsize=8)
        ax.set_xlabel('min', fontsize=8)
        ax.set_ylabel('ppb', fontsize=8)
        ax.grid(True, alpha=0.1)

        plt.suptitle(f'{i+1}/{len(pairs)}: {name} — tang={r_tang:.2f} trap={r_trap:.2f}',
                    fontsize=10)
        plt.tight_layout()
        pdf.savefig(fig, dpi=100)
        plt.close(fig)

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(pairs)}...")

print(f"\nPDF: {pdf_path} ({len(pairs)} pagines)")
