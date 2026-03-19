"""4 SEQs CAL: scatter tangent amb tots els punts individuals."""
import numpy as np
from scipy import stats
from hpsec_import import import_sequence
from hpsec_core import find_peak_boundaries
from hpsec_calibrate import extract_khp_conc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def get_khp(seq_path, vol):
    result = import_sequence(seq_path)
    pts = []
    for sn, sd in result.get('samples', {}).items():
        if 'khp' not in sn.lower(): continue
        conc = extract_khp_conc(sn)
        if conc <= 0: continue
        for rk, rd in sorted(sd.get('replicas', {}).items()):
            d = rd.get('direct')
            if d is None or not isinstance(d, dict): continue
            t = np.array(d.get('t', []))
            y = np.array(d.get('y_net', []))
            if len(t) < 10: continue
            area_trap = float(np.trapezoid(np.maximum(y, 0), t))
            pk = int(np.argmax(y))
            try:
                li, ri = find_peak_boundaries(t, y, pk)
                area_tang = float(np.trapezoid(np.maximum(y[li:ri+1], 0), t[li:ri+1]))
            except:
                area_tang = area_trap
            pts.append({'conc': conc, 'vol': vol, 'ug': conc*vol/1000,
                       'tang': area_tang, 'trap': area_trap,
                       'name': sn, 'rep': rk})
    return pts

print("Important...")
bp_205 = get_khp('C:/Users/Lequia/Desktop/Dades3/205_SEQ_BP_CAL', 100)
col_206 = get_khp('C:/Users/Lequia/Desktop/Dades3/206_SEQ_CAL', 400)
bp_292 = get_khp('C:/Users/Lequia/Desktop/Dades3/292_SEQ_CAL_BP', 100)
col_293 = get_khp('C:/Users/Lequia/Desktop/Dades3/293_SEQ_CAL', 400)

datasets = [
    ('205 BP (v=100)', bp_205, '#E74C3C', 'o'),
    ('206 COL (v=400)', col_206, '#2E86C1', 's'),
    ('292 BP (v=100)', bp_292, '#E74C3C', 'D'),
    ('293 COL (v=400)', col_293, '#2E86C1', '^'),
]

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for ax, method, method_name in [(axes[0], 'tang', 'TANGENT (pic)'), (axes[1], 'trap', 'TRAPEZOID (complet)')]:
    all_pts = []
    for label, pts, color, marker in datasets:
        ug = [p['ug'] for p in pts]
        area = [p[method] for p in pts]
        ax.scatter(ug, area, c=color, s=40, marker=marker, edgecolors='k', lw=0.3,
                  alpha=0.7, label=label, zorder=5)
        # Anotar anomals
        for p in pts:
            a = p[method]
            # Buscar si es anomal (area molt diferent de la seva concentracio)
            expected = p['ug'] * 700  # aprox
            if a < expected * 0.3 or a > expected * 2:
                ax.annotate(f"{p['name'][:6]}R{p['rep']}", (p['ug'], a),
                           fontsize=5, color='red', alpha=0.7, xytext=(3,3),
                           textcoords='offset points')
        all_pts.extend([{'ug': u, 'area': a} for u, a in zip(ug, area)])

    # Regressio de tots
    u = np.array([p['ug'] for p in all_pts])
    a = np.array([p['area'] for p in all_pts])
    sl, ic, r, _, _ = stats.linregress(u, a)
    x_line = np.linspace(0, u.max() * 1.1, 100)
    ax.plot(x_line, sl * x_line + ic, 'k-', lw=1.5, alpha=0.6,
           label=f'Combo: {sl:.0f}x+{ic:.0f} R2={r**2:.4f}')

    ax.set_xlabel('ug DOC injectat', fontsize=10)
    ax.set_ylabel('Area', fontsize=10)
    ax.set_title(f'{method_name}\nslope={sl:.0f} ic={ic:.0f} R2={r**2:.4f}', fontsize=11)
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.15)
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)

plt.suptitle('4 SEQs calibracio: 205 BP + 206 COL + 292 BP + 293 COL\n'
             'Vermell=BP(100uL) Blau=COL(400uL) | o/s=205/206 D/^=292/293',
             fontsize=12, fontweight='bold')
plt.tight_layout()
out = 'C:/Users/Lequia/Desktop/HPSEC/_4cal_scatter.png'
plt.savefig(out, dpi=130, bbox_inches='tight')
print(f"Guardat: {out}")
