"""
Prototips de disposició per gràfics stacked del SampleDetailDialog.
Genera PNGs amb diferents propostes per avaluar.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
import matplotlib.cm as mpl_cm

# ── Carregar dades reals (285_SEQ COLUMN) ──
with open(r"C:\Users\Lequia\Desktop\Dades3\285_SEQ\CHECK\data\analysis_result.json",
          "r", encoding="utf-8") as f:
    data = json.load(f)

method = data["method"]  # COLUMN
is_bp = method.upper() == "BP"

# Trobar una mostra amb 2 rèpliques
sg = data.get("samples_grouped", {})
sample_name = None
sample_data = None
for name, sdata in sg.items():
    reps = sdata.get("replicas", {})
    if len(reps) >= 2:
        sample_name = name
        sample_data = sdata
        break

if not sample_data:
    print("No s'ha trobat mostra amb 2 rèpliques")
    exit()

print(f"Mostra: {sample_name}, method={method}")

replicas = sample_data.get("replicas", {})
rep_keys = sorted(replicas.keys())
r1 = replicas[rep_keys[0]]
r2 = replicas[rep_keys[1]] if len(rep_keys) > 1 else None
comparison = sample_data.get("comparison", {})
doc_comp = comparison.get("doc", {})
dad_comp = comparison.get("dad", {})

# Dades DOC
t1 = np.asarray(r1.get("t_doc", []))
y1_d = np.asarray(r1.get("y_doc_net", []))
y1_u = r1.get("y_doc_uib_net")
if y1_u:
    y1_u = np.asarray(y1_u)

t2 = np.asarray(r2.get("t_doc", [])) if r2 else None
y2_d = np.asarray(r2.get("y_doc_net", [])) if r2 else None

# Dades DAD
df_dad1 = r1.get("df_dad")
if df_dad1 and isinstance(df_dad1, dict):
    df_dad1 = pd.DataFrame(df_dad1)
df_dad2 = r2.get("df_dad") if r2 else None
if df_dad2 and isinstance(df_dad2, dict):
    df_dad2 = pd.DataFrame(df_dad2)

wl_cols = [c for c in df_dad1.columns if c != 'time (min)'] if df_dad1 is not None else []
wl_cols.sort(key=lambda x: int(x) if str(x).isdigit() else 0)

# Fraccions COLUMN
FRACS = [
    ("BioP", 10.8, 18),
    ("HS", 18, 23),
    ("BB", 23, 30),
    ("SB", 30, 40),
    ("LMW", 40, 70),
]

# Àrees per fracció
sel = sample_data.get("selected", {})
rep_sel = sel.get("doc", rep_keys[0])
sel_rep = replicas.get(rep_sel, r1)
areas = sel_rep.get("areas", {})
doc_areas = areas.get("DOC", {})
doc_total = doc_areas.get("total", 0)

# R² values
pearson_doc = doc_comp.get("pearson", 0)
pearson_per_wl = dad_comp.get("pearson_per_wavelength", {})
n_peaks_hs = sel_rep.get("n_peaks_254_HS", 0)

# Quantificació
quant = sample_data.get("quantification", {})
ppm_direct = quant.get("concentration_ppm_direct") or quant.get("concentration_ppm", 0)
ppm_uib = quant.get("concentration_ppm_uib", 0)

# Colors
C1 = '#1565C0'
C2 = '#E65100'
C_UIB = '#2E7D32'
x_min, x_max = (0, 15) if is_bp else (0, 70)


def add_fraction_vlines(ax, fracs, x_max):
    """Línies verticals discontinues per separar fraccions."""
    for name, start, end in fracs:
        if start > 0 and start <= x_max:
            ax.axvline(start, color='#999', ls=':', lw=0.5, zorder=0)


def add_fraction_table(fig, fracs, doc_areas, doc_total, dad_areas_by_wl,
                       wl_cols, rect=(0.02, 0.01, 0.96, 0.05)):
    """Taula de fraccions incrustada a la part inferior de la figura."""
    ax_tbl = fig.add_axes(rect)
    ax_tbl.axis('off')

    # Header
    col_labels = ["Fracció", "Rang", "DOC %"]
    for wl in wl_cols[:4]:  # Max 4 wl per espai
        col_labels.append(f"A{wl} %")

    rows = []
    for name, start, end in fracs:
        row = [name, f"{start}-{end}", ""]
        frac_area = doc_areas.get(name, 0)
        pct = (frac_area / doc_total * 100) if doc_total > 0 else 0
        row[2] = f"{pct:.1f}%"
        for wl in wl_cols[:4]:
            wl_key = f"A{wl}" if not str(wl).startswith('A') else wl
            wl_areas = dad_areas_by_wl.get(wl_key, {})
            wl_total = wl_areas.get("total", 0)
            wl_frac = wl_areas.get(name, 0)
            wl_pct = (wl_frac / wl_total * 100) if wl_total > 0 else 0
            row.append(f"{wl_pct:.1f}%")
        rows.append(row)

    tbl = ax_tbl.table(
        cellText=rows,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(6.5)
    tbl.scale(1, 1.1)

    # Estil
    for key, cell in tbl.get_celld().items():
        cell.set_linewidth(0.3)
        if key[0] == 0:  # Header
            cell.set_facecolor('#E8EAF6')
            cell.set_text_props(fontweight='bold')
        else:
            cell.set_facecolor('white')


def add_info_box(ax, lines, loc='upper left', fontsize=6.5):
    """Caixa d'info amb múltiples línies."""
    text = "\n".join(lines)
    props = dict(boxstyle='round,pad=0.3', facecolor='white',
                 edgecolor='#ccc', alpha=0.9)
    x, y = (0.01, 0.97) if 'left' in loc else (0.99, 0.97)
    ha = 'left' if 'left' in loc else 'right'
    ax.text(x, y, text, transform=ax.transAxes, fontsize=fontsize,
            va='top', ha=ha, bbox=props, linespacing=1.4,
            fontfamily='monospace')


# DAD areas by wavelength
dad_areas_by_wl = {}
for wl in wl_cols:
    wl_key = f"A{wl}" if not str(wl).startswith('A') else wl
    dad_areas_by_wl[wl_key] = areas.get(wl_key, {})


# ══════════════════════════════════════════════════════════════
# PROPOSTA A: Estret, taula de fraccions a baix, info boxes
# ══════════════════════════════════════════════════════════════
def proposal_a():
    n = 1 + len(wl_cols)
    fig = plt.figure(figsize=(5.5, n * 1.2 + 1.2), dpi=120)

    # Espai per gràfics (dalt) i taula (baix)
    gs = fig.add_gridspec(n, 1, hspace=0.08,
                          top=0.95, bottom=0.18, left=0.14, right=0.97)
    axes = [fig.add_subplot(gs[i]) for i in range(n)]

    # Compartir eix X
    for ax in axes[:-1]:
        ax.sharex(axes[-1])

    # DOC
    ax0 = axes[0]
    ax0.plot(t1, y1_d, color=C1, lw=0.8, label=f'R{rep_keys[0]}')
    if y1_u is not None and len(y1_u) == len(t1):
        ax0.plot(t1, y1_u, color=C_UIB, lw=0.7, ls='--', label='UIB')
    if r2 and t2 is not None:
        ax0.plot(t2, y2_d, color=C2, lw=0.8, alpha=0.7, label=f'R{rep_keys[1]}')
    ax0.set_ylabel("DOC", fontsize=7)
    ax0.legend(loc='upper right', fontsize=5.5, ncol=3, framealpha=0.7)
    ax0.tick_params(labelsize=6, labelbottom=False)
    ax0.grid(True, alpha=0.2, lw=0.4)
    add_fraction_vlines(ax0, FRACS, x_max)

    # Info DOC
    info = []
    if ppm_direct:
        info.append(f"ppm Direct: {ppm_direct:.3f}")
    if ppm_uib:
        info.append(f"ppm UIB:    {ppm_uib:.3f}")
    if pearson_doc:
        info.append(f"R² DOC:     {pearson_doc:.4f}")
    if info:
        add_info_box(ax0, info)

    # DAD
    for i, wl in enumerate(wl_cols):
        ax = axes[i + 1]
        is_last = (i == len(wl_cols) - 1)

        if df_dad1 is not None and wl in df_dad1.columns:
            ax.plot(df_dad1['time (min)'].values, df_dad1[wl].values,
                    color=C1, lw=0.8)
        if df_dad2 is not None and wl in df_dad2.columns:
            ax.plot(df_dad2['time (min)'].values, df_dad2[wl].values,
                    color=C2, lw=0.8, alpha=0.7)

        wl_label = f"A{wl}" if not str(wl).startswith('A') else wl
        ax.set_ylabel(wl_label, fontsize=7)
        ax.grid(True, alpha=0.2, lw=0.4)
        ax.tick_params(labelsize=6, labelbottom=is_last)
        add_fraction_vlines(ax, FRACS, x_max)

        # R² per λ
        wl_key = f"A{wl}"
        r2v = pearson_per_wl.get(wl_key, 0)
        if r2v and r2v > 0:
            clr = '#C62828' if r2v < 0.990 else '#555'
            ax.text(0.99, 0.92, f"R²={r2v:.4f}",
                    transform=ax.transAxes, fontsize=5.5,
                    color=clr, ha='right', va='top')

        # Pics HS a 254
        if wl in ('254', 'A254') and n_peaks_hs:
            ax.text(0.99, 0.75, f"{n_peaks_hs} pics HS",
                    transform=ax.transAxes, fontsize=5.5,
                    color='#6A1B9A', ha='right', va='top')

    axes[-1].set_xlabel("Temps (min)", fontsize=8)
    axes[-1].set_xlim(x_min, x_max)

    # Taula de fraccions
    add_fraction_table(fig, FRACS, doc_areas, doc_total,
                       dad_areas_by_wl, wl_cols,
                       rect=(0.05, 0.01, 0.90, 0.14))

    fig.suptitle(f"{sample_name}  |  {method}", fontsize=9, fontweight='bold', y=0.98)
    fig.savefig("proposal_A.png", dpi=150)
    print("Saved proposal_A.png")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════
# PROPOSTA B: Més compacte, 2 columnes (DOC+DAD254 | taula+info)
# ══════════════════════════════════════════════════════════════
def proposal_b():
    n_wl = len(wl_cols)
    fig = plt.figure(figsize=(7, n_wl * 0.9 + 2.2), dpi=120)

    # Layout: gràfics a l'esquerra (70%), taula+info a la dreta (30%)
    gs = fig.add_gridspec(n_wl + 1, 2, width_ratios=[3, 1.2],
                          hspace=0.08, wspace=0.05,
                          top=0.94, bottom=0.06, left=0.10, right=0.98)

    # Gràfics (columna esquerra)
    axes = []
    for i in range(n_wl + 1):
        ax = fig.add_subplot(gs[i, 0])
        axes.append(ax)
    for ax in axes[:-1]:
        ax.sharex(axes[-1])

    # DOC
    ax0 = axes[0]
    ax0.plot(t1, y1_d, color=C1, lw=0.8)
    if y1_u is not None and len(y1_u) == len(t1):
        ax0.plot(t1, y1_u, color=C_UIB, lw=0.7, ls='--')
    if r2 and t2 is not None:
        ax0.plot(t2, y2_d, color=C2, lw=0.8, alpha=0.7)
    ax0.set_ylabel("DOC", fontsize=7)
    ax0.tick_params(labelsize=6, labelbottom=False)
    ax0.grid(True, alpha=0.2, lw=0.4)
    add_fraction_vlines(ax0, FRACS, x_max)

    # R² DOC
    if pearson_doc:
        ax0.text(0.99, 0.92, f"R²={pearson_doc:.4f}",
                 transform=ax0.transAxes, fontsize=5.5,
                 color='#555', ha='right', va='top')

    # DAD
    for i, wl in enumerate(wl_cols):
        ax = axes[i + 1]
        is_last = (i == len(wl_cols) - 1)
        if df_dad1 is not None and wl in df_dad1.columns:
            ax.plot(df_dad1['time (min)'].values, df_dad1[wl].values,
                    color=C1, lw=0.8)
        if df_dad2 is not None and wl in df_dad2.columns:
            ax.plot(df_dad2['time (min)'].values, df_dad2[wl].values,
                    color=C2, lw=0.8, alpha=0.7)
        ax.set_ylabel(f"A{wl}", fontsize=7)
        ax.grid(True, alpha=0.2, lw=0.4)
        ax.tick_params(labelsize=6, labelbottom=is_last)
        add_fraction_vlines(ax, FRACS, x_max)

        r2v = pearson_per_wl.get(f"A{wl}", 0)
        if r2v and r2v > 0:
            clr = '#C62828' if r2v < 0.990 else '#555'
            ax.text(0.99, 0.92, f"R²={r2v:.4f}",
                    transform=ax.transAxes, fontsize=5.5,
                    color=clr, ha='right', va='top')

    axes[-1].set_xlabel("Temps (min)", fontsize=8)
    axes[-1].set_xlim(x_min, x_max)

    # Panel dret: info + taula
    ax_info = fig.add_subplot(gs[:3, 1])
    ax_info.axis('off')
    info_lines = [
        f"Mostra: {sample_name}",
        f"Mode: {method}",
        f"Reps: R{rep_keys[0]}" + (f", R{rep_keys[1]}" if r2 else ""),
        "",
        f"ppm Direct: {ppm_direct:.3f}" if ppm_direct else "",
        f"ppm UIB: {ppm_uib:.3f}" if ppm_uib else "",
        "",
        f"R² DOC: {pearson_doc:.4f}" if pearson_doc else "",
    ]
    info_lines = [l for l in info_lines if l is not None]
    ax_info.text(0.05, 0.95, "\n".join(info_lines),
                 transform=ax_info.transAxes, fontsize=6.5,
                 va='top', fontfamily='monospace', linespacing=1.4)

    # Taula fraccions al panel dret (baix)
    ax_tbl = fig.add_subplot(gs[3:, 1])
    ax_tbl.axis('off')
    rows = []
    for name, start, end in FRACS:
        frac_a = doc_areas.get(name, 0)
        pct = (frac_a / doc_total * 100) if doc_total > 0 else 0
        rows.append([name, f"{start}-{end}", f"{pct:.1f}%"])
    tbl = ax_tbl.table(cellText=rows,
                       colLabels=["Frac", "Rang", "DOC %"],
                       loc='upper center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(6)
    tbl.scale(1, 1.1)
    for key, cell in tbl.get_celld().items():
        cell.set_linewidth(0.3)
        if key[0] == 0:
            cell.set_facecolor('#E8EAF6')
            cell.set_text_props(fontweight='bold')
        else:
            cell.set_facecolor('white')

    fig.suptitle(f"{sample_name}  |  {method}", fontsize=9, fontweight='bold', y=0.97)
    fig.savefig("proposal_B.png", dpi=150)
    print("Saved proposal_B.png")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════
# PROPOSTA C: Ultra-compacte, gràfics molt estrets,
#             taula fraccions completa integrada
# ══════════════════════════════════════════════════════════════
def proposal_c():
    n_wl = len(wl_cols)
    n_graph = 1 + n_wl
    # Alçades: gràfics molt estrets (0.8 cadascun), taula final (1.5)
    heights = [0.8] * n_graph + [1.8]
    fig, all_axes = plt.subplots(
        n_graph + 1, 1,
        figsize=(5, sum(heights) + 0.5), dpi=120,
        gridspec_kw={'height_ratios': heights, 'hspace': 0.08},
    )
    graph_axes = all_axes[:n_graph]
    ax_table = all_axes[n_graph]

    for ax in graph_axes[:-1]:
        ax.sharex(graph_axes[-1])

    # DOC
    ax0 = graph_axes[0]
    ax0.plot(t1, y1_d, color=C1, lw=0.7)
    if y1_u is not None and len(y1_u) == len(t1):
        ax0.plot(t1, y1_u, color=C_UIB, lw=0.6, ls='--')
    if r2 and t2 is not None:
        ax0.plot(t2, y2_d, color=C2, lw=0.7, alpha=0.7)
    ax0.set_ylabel("DOC", fontsize=6.5, labelpad=2)
    ax0.tick_params(labelsize=5.5, labelbottom=False, length=2, pad=1)
    ax0.grid(True, alpha=0.2, lw=0.3)
    add_fraction_vlines(ax0, FRACS, x_max)

    # Info compacta DOC
    parts = []
    if pearson_doc:
        parts.append(f"R²={pearson_doc:.4f}")
    if ppm_direct:
        parts.append(f"{ppm_direct:.2f} ppm")
    if parts:
        ax0.text(0.99, 0.90, "  ".join(parts),
                 transform=ax0.transAxes, fontsize=5.5,
                 ha='right', va='top', color='#444')

    # Llegenda simple
    from matplotlib.lines import Line2D
    leg_handles = [
        Line2D([0], [0], color=C1, lw=0.7, label=f'R{rep_keys[0]}'),
    ]
    if r2:
        leg_handles.append(Line2D([0], [0], color=C2, lw=0.7, label=f'R{rep_keys[1]}'))
    if y1_u is not None:
        leg_handles.append(Line2D([0], [0], color=C_UIB, lw=0.6, ls='--', label='UIB'))
    ax0.legend(handles=leg_handles, loc='upper left', fontsize=5,
               ncol=3, framealpha=0.7, handlelength=1.5)

    # DAD
    for i, wl in enumerate(wl_cols):
        ax = graph_axes[i + 1]
        is_last = (i == n_wl - 1)
        if df_dad1 is not None and wl in df_dad1.columns:
            ax.plot(df_dad1['time (min)'].values, df_dad1[wl].values,
                    color=C1, lw=0.7)
        if df_dad2 is not None and wl in df_dad2.columns:
            ax.plot(df_dad2['time (min)'].values, df_dad2[wl].values,
                    color=C2, lw=0.7, alpha=0.7)
        ax.set_ylabel(f"A{wl}", fontsize=6.5, labelpad=2)
        ax.grid(True, alpha=0.2, lw=0.3)
        ax.tick_params(labelsize=5.5, labelbottom=is_last, length=2, pad=1)
        add_fraction_vlines(ax, FRACS, x_max)

        # R² compact
        r2v = pearson_per_wl.get(f"A{wl}", 0)
        parts_wl = []
        if r2v and r2v > 0:
            clr = '#C62828' if r2v < 0.990 else '#555'
            parts_wl.append(f"R²={r2v:.4f}")
        if wl in ('254', 'A254') and n_peaks_hs:
            parts_wl.append(f"{n_peaks_hs}pics")
        if parts_wl:
            ax.text(0.99, 0.88, "  ".join(parts_wl),
                    transform=ax.transAxes, fontsize=5,
                    color=clr if r2v and r2v < 0.990 else '#555',
                    ha='right', va='top')

    graph_axes[-1].set_xlabel("Temps (min)", fontsize=7)
    graph_axes[-1].set_xlim(x_min, x_max)

    # ── Taula de fraccions completa (últim subplot) ──
    ax_table.axis('off')

    col_labels = ["Fracció", "Rang (min)"]
    col_labels += ["DOC %"]
    for wl in wl_cols:
        col_labels.append(f"A{wl} %")

    rows = []
    for name, start, end in FRACS:
        row = [name, f"{start}–{end}"]
        fa = doc_areas.get(name, 0)
        pct = (fa / doc_total * 100) if doc_total > 0 else 0
        row.append(f"{pct:.1f}")
        for wl in wl_cols:
            wl_key = f"A{wl}"
            wa = areas.get(wl_key, {})
            wt = wa.get("total", 0)
            wf = wa.get(name, 0)
            wp = (wf / wt * 100) if wt > 0 else 0
            row.append(f"{wp:.1f}")
        rows.append(row)

    # Fila total
    total_row = ["TOTAL", "0–70", "100"]
    for wl in wl_cols:
        total_row.append("100")
    rows.append(total_row)

    tbl = ax_table.table(cellText=rows, colLabels=col_labels,
                         loc='upper center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(6)
    tbl.scale(1, 1.15)
    for key, cell in tbl.get_celld().items():
        cell.set_linewidth(0.3)
        cell.set_height(0.12)
        if key[0] == 0:
            cell.set_facecolor('#E0E0E0')
            cell.set_text_props(fontweight='bold', fontsize=5.5)
        elif key[0] == len(rows):  # Total row
            cell.set_facecolor('#F5F5F5')
            cell.set_text_props(fontweight='bold')
        else:
            cell.set_facecolor('white')

    fig.suptitle(f"{sample_name}  |  {method}  |  R{rep_keys[0]}"
                 + (f"+R{rep_keys[1]}" if r2 else ""),
                 fontsize=8, fontweight='bold', y=0.99)

    fig.subplots_adjust(top=0.96, bottom=0.02, left=0.12, right=0.97)
    fig.savefig("proposal_C.png", dpi=150)
    print("Saved proposal_C.png")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════
# PROPOSTA D: Grid 2 columnes, meitat d'amplada per gràfic
#             Fila1: DOC | UIB, Fila2: A220 | A252, ...
#             Taula final: fraccions=columnes, senyals=files
# ══════════════════════════════════════════════════════════════
def proposal_d():
    from matplotlib.lines import Line2D

    # UIB data
    y1_u_loc = y1_u
    y2_u = None
    if r2:
        _y2u = r2.get("y_doc_uib_net")
        if _y2u:
            y2_u = np.asarray(_y2u)

    # UIB areas
    areas_uib = sel_rep.get("areas_uib", {})
    uib_total = areas_uib.get("total", 0)
    has_uib = y1_u_loc is not None and len(y1_u_loc) > 0

    # n_peaks_per_wl (pot no existir en JSONs antics, calculem des de dades)
    n_peaks_per_wl = sel_rep.get("n_peaks_per_wl", {})
    if not n_peaks_per_wl:
        # Calcular al vol des de les dades crues
        from scipy.signal import find_peaks as _fp
        def _count(t_arr, y_arr):
            result = {}
            for fname, fstart, fend in FRACS:
                mask = (t_arr >= fstart) & (t_arr <= fend)
                if np.sum(mask) > 10:
                    y_z = y_arr[mask]
                    yr = np.max(y_z) - np.min(y_z)
                    pks, _ = _fp(y_z, prominence=max(yr*0.05, 0.01), distance=3)
                    result[fname] = len(pks)
                else:
                    result[fname] = 0
            return result
        # DOC
        if len(t1) > 0 and len(y1_d) > 0:
            n_peaks_per_wl["DOC"] = _count(t1, y1_d)
        if has_uib:
            n_peaks_per_wl["UIB"] = _count(t1, y1_u_loc)
        if df_dad1 is not None:
            t_dad = df_dad1['time (min)'].values
            for wl in wl_cols:
                wl_key = f"A{wl}" if not str(wl).startswith('A') else wl
                n_peaks_per_wl[wl_key] = _count(t_dad, df_dad1[wl].values)

    # Helper: format peak annotation (nomes HS)
    def _peak_text(sig_key):
        pk = n_peaks_per_wl.get(sig_key, {})
        n_hs = pk.get("HS", 0)
        return f"{n_hs}p HS" if n_hs else ""

    # Helper: plot annotation (R² + ppm/peaks)
    def _annotate(ax, r2v=None, ppm=None, sig_key=None):
        line1_parts = []
        if r2v and r2v > 0:
            line1_parts.append(f"R²={r2v:.4f}")
        if ppm:
            line1_parts.append(f"{ppm:.2f} ppm")
        line2 = _peak_text(sig_key) if sig_key else ""
        lines = []
        if line1_parts:
            lines.append("  ".join(line1_parts))
        if line2:
            lines.append(line2)
        if lines:
            clr = '#C62828' if (r2v and r2v < 0.990) else '#555'
            ax.text(0.99, 0.92, "\n".join(lines),
                    transform=ax.transAxes, fontsize=4.5,
                    color=clr, ha='right', va='top', linespacing=1.3)

    # Parells DAD
    pairs = []
    for i in range(0, len(wl_cols), 2):
        if i + 1 < len(wl_cols):
            pairs.append((wl_cols[i], wl_cols[i + 1]))
        else:
            pairs.append((wl_cols[i], None))

    n_graph_rows = 1 + len(pairs)
    n_total_rows = n_graph_rows + 1  # + table

    h_graphs = [1.0] * n_graph_rows
    h_table = [2.5]
    heights = h_graphs + h_table

    fig = plt.figure(figsize=(7.5, sum(heights) * 0.9 + 0.8), dpi=120)
    gs = fig.add_gridspec(
        n_total_rows, 2,
        height_ratios=heights,
        hspace=0.30, wspace=0.22,
        top=0.94, bottom=0.03, left=0.08, right=0.97
    )

    all_graph_axes = []

    # ── Fila 0: DOC Direct | DOC UIB ──
    ax_doc = fig.add_subplot(gs[0, 0])
    ax_uib = fig.add_subplot(gs[0, 1])
    all_graph_axes.extend([ax_doc, ax_uib])

    ax_doc.plot(t1, y1_d, color=C1, lw=0.7, label=f'R{rep_keys[0]}')
    if r2 and t2 is not None and y2_d is not None:
        ax_doc.plot(t2, y2_d, color=C2, lw=0.7, alpha=0.7, label=f'R{rep_keys[1]}')
    ax_doc.set_ylabel("DOC", fontsize=6.5, labelpad=2)
    ax_doc.tick_params(labelsize=5.5, length=2, pad=1)
    ax_doc.grid(True, alpha=0.2, lw=0.3)
    ax_doc.set_xlim(x_min, x_max)
    add_fraction_vlines(ax_doc, FRACS, x_max)
    ax_doc.legend(loc='upper left', fontsize=5, ncol=2,
                  framealpha=0.7, handlelength=1.2)
    _annotate(ax_doc, r2v=pearson_doc, ppm=ppm_direct, sig_key="DOC")

    # DOC UIB
    if has_uib:
        ax_uib.plot(t1, y1_u_loc, color=C_UIB, lw=0.7, label=f'R{rep_keys[0]}')
        if y2_u is not None and t2 is not None:
            ax_uib.plot(t2, y2_u, color='#66BB6A', lw=0.7, alpha=0.7,
                        label=f'R{rep_keys[1]}')
        ax_uib.legend(loc='upper left', fontsize=5, ncol=2,
                      framealpha=0.7, handlelength=1.2)
        _annotate(ax_uib, ppm=ppm_uib, sig_key="UIB")
    else:
        ax_uib.text(0.5, 0.5, "UIB no disponible",
                    ha='center', va='center',
                    transform=ax_uib.transAxes, fontsize=8, color='#aaa')
    ax_uib.set_ylabel("UIB", fontsize=6.5, labelpad=2)
    ax_uib.tick_params(labelsize=5.5, length=2, pad=1)
    ax_uib.grid(True, alpha=0.2, lw=0.3)
    ax_uib.set_xlim(x_min, x_max)
    add_fraction_vlines(ax_uib, FRACS, x_max)

    # ── Files DAD ──
    for row_i, (wl_left, wl_right) in enumerate(pairs):
        for col_j, wl in enumerate([wl_left, wl_right]):
            if wl is None:
                ax = fig.add_subplot(gs[row_i + 1, col_j])
                ax.axis('off')
                all_graph_axes.append(ax)
                continue

            ax = fig.add_subplot(gs[row_i + 1, col_j])
            all_graph_axes.append(ax)

            if df_dad1 is not None and wl in df_dad1.columns:
                ax.plot(df_dad1['time (min)'].values, df_dad1[wl].values,
                        color=C1, lw=0.7)
            if df_dad2 is not None and wl in df_dad2.columns:
                ax.plot(df_dad2['time (min)'].values, df_dad2[wl].values,
                        color=C2, lw=0.7, alpha=0.7)

            wl_label = f"A{wl}" if not str(wl).startswith('A') else wl
            ax.set_ylabel(wl_label, fontsize=6.5, labelpad=2)
            ax.grid(True, alpha=0.2, lw=0.3)
            ax.tick_params(labelsize=5.5, length=2, pad=1)
            ax.set_xlim(x_min, x_max)
            add_fraction_vlines(ax, FRACS, x_max)

            wl_key = f"A{wl}"
            # R² pot estar amb clau "A254" o "254"
            r2v = pearson_per_wl.get(wl_key, 0) or pearson_per_wl.get(str(wl), 0)
            _annotate(ax, r2v=r2v, sig_key=wl_key)

    # X label bottom row
    bottom_row = n_graph_rows - 1
    for col_j in range(2):
        idx = 2 + bottom_row * 2 + col_j
        if idx < len(all_graph_axes):
            ax = all_graph_axes[idx]
            if ax.axison:
                ax.set_xlabel("Temps (min)", fontsize=6.5)

    # ── Taula: fraccions=columnes amb rang a capçalera, senyals=files ──
    ax_tbl = fig.add_subplot(gs[n_graph_rows, :])
    ax_tbl.axis('off')

    # Capçalera: Senyal | BioP (0-18) | HS (18-23) | ... | TOTAL (0-70)
    col_labels = ["Senyal"]
    for fname, fstart, fend in FRACS:
        col_labels.append(f"{fname} ({fstart:g}–{fend:g})")
    col_labels.append(f"TOTAL (0–{x_max:g})")

    # Files: DOC, UIB, A220..A362
    signal_names = ["DOC"]
    if has_uib:
        signal_names.append("UIB")
    for wl in wl_cols:
        wl_lbl = f"A{wl}" if not str(wl).startswith('A') else wl
        signal_names.append(wl_lbl)

    rows = []
    for sig in signal_names:
        row = [sig]
        if sig == "DOC":
            sig_areas = doc_areas
            sig_total = doc_total
        elif sig == "UIB":
            sig_areas = areas_uib
            sig_total = uib_total
        else:
            sig_areas = areas.get(sig, {})
            sig_total = sig_areas.get("total", 0)

        for fname, fstart, fend in FRACS:
            fval = sig_areas.get(fname, 0)
            pct = (fval / sig_total * 100) if sig_total > 0 else 0
            row.append(f"{pct:.1f}")
        row.append("100" if sig_total > 0 else "\u2013")
        rows.append(row)

    tbl = ax_tbl.table(cellText=rows, colLabels=col_labels,
                       loc='upper center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(6)
    tbl.scale(1, 1.2)

    for key, cell in tbl.get_celld().items():
        cell.set_linewidth(0.3)
        cell.set_height(0.08)
        if key[0] == 0:  # Header
            cell.set_facecolor('#E0E0E0')
            cell.set_text_props(fontweight='bold', fontsize=5.5)
        elif key[1] == 0:  # Signal name column
            cell.set_facecolor('#F5F5F5')
            cell.set_text_props(fontweight='bold', fontsize=6)
        else:
            cell.set_facecolor('white')

    # Title
    rep_label = f"R{rep_keys[0]}"
    if r2:
        rep_label += f"+R{rep_keys[1]}"
    fig.suptitle(f"{sample_name}  |  {method}  |  {rep_label}",
                 fontsize=9, fontweight='bold', y=0.98)

    fig.savefig("proposal_D.png", dpi=150)
    print("Saved proposal_D.png")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════
# Generar totes
# ══════════════════════════════════════════════════════════════
proposal_d()
print("\nDone! Revisa proposal_D.png")
