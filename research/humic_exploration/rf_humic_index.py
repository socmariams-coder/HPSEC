# -*- coding: utf-8 -*-
"""
Humic Character Index — HPSEC-DAD Export3D
==========================================
Per cada mostra, calcula la similitud cosinus amb els patrons
SRHA i SRFA de Suwannee River (IHSS) i classifica el caràcter húmic.

Output:
  - humic_index.xlsx: taula amb índex per mostra
  - humic_index_summary.png: visualització resum
"""

import os, sys, re, glob, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from collections import Counter, defaultdict
from scipy.spatial.distance import cosine

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

# =============================================================================
# CONFIG
# =============================================================================
DATA_FOLDERS = [
    "C:/Users/Lequia/Desktop/Dades3",
    "C:/Users/Lequia/OneDrive - Universitat de Girona/UdG365_HPLC-DAD - General/Dades",
]
OUTPUT_DIR = "C:/Users/Lequia/Desktop/HPSEC/rf_humic_analysis"

HS_WINDOW = (18.0, 23.0)
WL_ALL = list(range(200, 402, 2))

TYPE_PATTERNS = [
    ("SRHA",     r"(?i)^(SRHA|3S101H)"),
    ("SRFA",     r"(?i)^(SRFA|3S101F)"),
    ("HA",       r"(?i)(?:^|[_\-\s])HA(?:[_\-\s\.]|$|\d)"),
    ("FA",       r"(?i)(?:^|[_\-\s])FA(?:[_\-\s\.]|$|\d)"),
    ("MIX",      r"(?i)(?:^|[_\-\s])MIX(?:[_\-\s\.]|$|\d)"),
    ("NO3",      r"(?i)^(NO3|NaNO3)"),
    ("Br",       r"(?i)^(Br[_\.\s\d]|KBr)"),
    ("CaCO3",    r"(?i)^CaCO3"),
    ("Na2CO3",   r"(?i)^Na2CO3"),
    ("Na2SO4",   r"(?i)^Na2SO4"),
    ("NaOH",     r"(?i)^NaOH"),
    ("Buffer",   r"(?i)^Buffer"),
    ("FR",       r"(?i)^FR\d"),
    ("EX",       r"(?i)^EX\d"),
    ("EXT",      r"(?i)^EXT\d"),
    ("LQ",       r"(?i)^LQ\d"),
    ("SK",       r"(?i)^SK\d"),
    ("PTLL",     r"(?i)^PTLL[\-_]"),
    ("PTT",      r"(?i)^PTT[\-_]"),
    ("ATL_2024", r"(?i)^\d{4}-\d{2}-\d{2}"),
    ("PTLL_2025",r"(?i)^\d{4}-PTLL"),
    ("LAB",      r"(?i)^\d{4}-L-"),
    ("LAB_T",    r"(?i)^\d{4}-T-"),
    ("EP_PTL",   r"(?i)^EP[_\-]?PTL"),
    ("EP_PTT",   r"(?i)^EP[_\-]?PTT"),
    ("EP",       r"(?i)^EP[_\s\-]"),
    ("SP",       r"(?i)^SP[_\s\-]"),
    ("POST_O3",  r"(?i)^POST"),
    ("FS",       r"(?i)^FS\d"),
    ("SFiltres", r"(?i)^SFiltres"),
    ("KHP",      r"(?i)^KHP"),
    ("BLANK",    r"(?i)^(MQ|BLANK|BLK|H2O)"),
]

def classify_sample(filename):
    base = re.sub(r'[_\s]*(R?\d+)\.(csv|CSV)$', '', filename)
    for label, pattern in TYPE_PATTERNS:
        if re.search(pattern, base):
            return label
    return "OTHER"

def load_export3d(filepath):
    try:
        df = pd.read_csv(filepath, encoding='utf-16', sep=',', header=None)
        if df.shape[1] < 50:
            return None
        wavelengths = df.iloc[0, 1:].values.astype(float)
        time_vals = df.iloc[1:, 0].values.astype(float)
        data = df.iloc[1:, 1:].values.astype(float)
        if len(time_vals) < 100 or np.max(time_vals) < 15:
            return None
        return time_vals, wavelengths, data
    except:
        return None

def extract_hs_spectrum(time_vals, wavelengths, data):
    """Extract mean HS fraction spectrum, interpolated to standard WL grid."""
    mask = (time_vals >= HS_WINDOW[0]) & (time_vals <= HS_WINDOW[1])
    if np.sum(mask) < 3:
        return None
    mean_spec = np.mean(data[mask, :], axis=0)
    # Interpolate to standard grid
    result = np.interp(WL_ALL, wavelengths, mean_spec)
    return result

def cosine_sim(a, b):
    """Cosine similarity, handles zero vectors."""
    if np.all(a == 0) or np.all(b == 0):
        return 0.0
    return 1 - cosine(a, b)

def classify_character(sim_ha, sim_fa, threshold=0.005):
    """Classify humic character based on similarity difference."""
    diff = sim_ha - sim_fa
    if diff > threshold:
        return "HA-dominant"
    elif diff < -threshold:
        return "FA-dominant"
    else:
        return "Mixed"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # === STEP 1: Scan files ===
    print("=" * 70)
    print("STEP 1: Scanning Export3D...")
    print("=" * 70)

    all_files = []
    seen = set()
    for base_folder in DATA_FOLDERS:
        if not os.path.isdir(base_folder):
            continue
        for entry in sorted(os.listdir(base_folder)):
            entry_path = os.path.join(base_folder, entry)
            if not os.path.isdir(entry_path) or '_SEQ' not in entry or entry.startswith('_'):
                continue
            exp_dir = os.path.join(entry_path, 'Export3d')
            if not os.path.isdir(exp_dir):
                continue
            csvs = glob.glob(os.path.join(exp_dir, '*.csv')) + \
                   glob.glob(os.path.join(exp_dir, '*.CSV'))
            for f in sorted(csvs):
                fname = os.path.basename(f)
                key = (entry, fname.upper())
                if key not in seen:
                    seen.add(key)
                    all_files.append({
                        'path': f, 'filename': fname,
                        'seq': entry, 'type': classify_sample(fname),
                    })

    exclude = {"KHP", "BLANK", "NaOH", "Buffer"}
    files = [f for f in all_files if f['type'] not in exclude]
    print(f"Files to process: {len(files)}")

    # === STEP 2: Extract HS spectra ===
    print("\n" + "=" * 70)
    print("STEP 2: Extracting HS spectra...")
    print("=" * 70)

    records = []
    for i, f in enumerate(files):
        if (i+1) % 200 == 0:
            print(f"  {i+1}/{len(files)}...")
        result = load_export3d(f['path'])
        if result is None:
            continue
        time_vals, wavelengths, data = result
        hs_spec = extract_hs_spectrum(time_vals, wavelengths, data)
        if hs_spec is None:
            continue
        records.append({
            'filename': f['filename'],
            'seq': f['seq'],
            'type': f['type'],
            'hs_spectrum': hs_spec,
        })

    print(f"Successfully extracted: {len(records)} samples")

    # === STEP 3: Build SRHA/SRFA templates ===
    print("\n" + "=" * 70)
    print("STEP 3: Building reference templates...")
    print("=" * 70)

    srha_specs = [r['hs_spectrum'] for r in records if r['type'] == 'SRHA']
    srfa_specs = [r['hs_spectrum'] for r in records if r['type'] == 'SRFA']
    ha_specs = [r['hs_spectrum'] for r in records if r['type'] in ('HA', 'SRHA')]
    fa_specs = [r['hs_spectrum'] for r in records if r['type'] in ('FA', 'SRFA')]

    print(f"  SRHA samples: {len(srha_specs)}")
    print(f"  SRFA samples: {len(srfa_specs)}")
    print(f"  All HA (incl SRHA): {len(ha_specs)}")
    print(f"  All FA (incl SRFA): {len(fa_specs)}")

    template_srha = np.mean(srha_specs, axis=0) if srha_specs else None
    template_srfa = np.mean(srfa_specs, axis=0) if srfa_specs else None
    template_ha = np.mean(ha_specs, axis=0) if ha_specs else None
    template_fa = np.mean(fa_specs, axis=0) if fa_specs else None

    if template_srha is None or template_srfa is None:
        print("ERROR: No SRHA or SRFA samples found!")
        return

    # === STEP 4: Compute similarity for every sample ===
    print("\n" + "=" * 70)
    print("STEP 4: Computing humic character index...")
    print("=" * 70)

    results = []
    for r in records:
        spec = r['hs_spectrum']

        sim_srha = cosine_sim(spec, template_srha)
        sim_srfa = cosine_sim(spec, template_srfa)
        sim_ha = cosine_sim(spec, template_ha)
        sim_fa = cosine_sim(spec, template_fa)

        # HA-FA index: positive = more HA-like
        delta_sr = sim_srha - sim_srfa
        delta_all = sim_ha - sim_fa

        # Percentage: 0% = pure FA, 100% = pure HA
        # Based on where it falls between SRFA and SRHA similarity
        total = sim_srha + sim_srfa
        pct_ha = (sim_srha / total * 100) if total > 0 else 50.0

        character = classify_character(sim_srha, sim_srfa)

        results.append({
            'Sample': r['filename'],
            'SEQ': r['seq'],
            'Type': r['type'],
            'cos_SRHA': round(sim_srha, 5),
            'cos_SRFA': round(sim_srfa, 5),
            'cos_HA_all': round(sim_ha, 5),
            'cos_FA_all': round(sim_fa, 5),
            'HA-FA_index': round(delta_sr, 5),
            '%HA_character': round(pct_ha, 1),
            'Character': character,
        })

    df = pd.DataFrame(results)
    print(f"\nResults: {len(df)} samples")
    print(f"\nCharacter distribution:")
    print(df['Character'].value_counts().to_string())

    # === STEP 5: Summary per type ===
    print("\n" + "=" * 70)
    print("STEP 5: Summary per sample type")
    print("=" * 70)

    summary = df.groupby('Type').agg({
        'cos_SRHA': ['mean', 'std', 'count'],
        'cos_SRFA': ['mean', 'std'],
        'HA-FA_index': ['mean', 'std'],
        '%HA_character': ['mean', 'std'],
    }).round(4)

    print(f"\n{'Type':15s} {'n':>4s}  {'cos(SRHA)':>10s}  {'cos(SRFA)':>10s}  {'HA-FA idx':>10s}  {'%HA':>6s}  {'Character':>12s}")
    print("-" * 80)
    for t in df['Type'].unique():
        mask = df['Type'] == t
        sub = df[mask]
        n = len(sub)
        if n < 1:
            continue
        char_mode = sub['Character'].mode().iloc[0]
        print(f"{t:15s} {n:4d}  "
              f"{sub['cos_SRHA'].mean():10.5f}  "
              f"{sub['cos_SRFA'].mean():10.5f}  "
              f"{sub['HA-FA_index'].mean():+10.5f}  "
              f"{sub['%HA_character'].mean():5.1f}%  "
              f"{char_mode:>12s}")

    # === STEP 6: Save Excel ===
    print("\n" + "=" * 70)
    print("STEP 6: Saving results...")
    print("=" * 70)

    xlsx_path = os.path.join(OUTPUT_DIR, "humic_index.xlsx")
    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        # Sheet 1: All samples
        df.sort_values(['Type', 'SEQ', 'Sample']).to_excel(
            writer, sheet_name='Per_Sample', index=False)

        # Sheet 2: Summary per type
        type_summary = df.groupby('Type').agg({
            'cos_SRHA': ['mean', 'std', 'min', 'max', 'count'],
            'cos_SRFA': ['mean', 'std', 'min', 'max'],
            'HA-FA_index': ['mean', 'std', 'min', 'max'],
            '%HA_character': ['mean', 'std', 'min', 'max'],
        }).round(5)
        type_summary.columns = ['_'.join(c) for c in type_summary.columns]
        type_summary.to_excel(writer, sheet_name='Per_Type')

        # Sheet 3: Summary per SEQ
        seq_summary = df.groupby(['SEQ', 'Type']).agg({
            'cos_SRHA': ['mean', 'std', 'count'],
            'cos_SRFA': ['mean', 'std'],
            'HA-FA_index': ['mean', 'std'],
            '%HA_character': ['mean'],
        }).round(5)
        seq_summary.columns = ['_'.join(c) for c in seq_summary.columns]
        seq_summary.to_excel(writer, sheet_name='Per_SEQ')

        # Sheet 4: Templates info
        template_info = pd.DataFrame({
            'Wavelength': WL_ALL,
            'SRHA_template': template_srha,
            'SRFA_template': template_srfa,
            'HA_all_template': template_ha,
            'FA_all_template': template_fa,
        })
        template_info.to_excel(writer, sheet_name='Templates', index=False)

    print(f"Saved: {xlsx_path}")

    # === STEP 7: Visualization ===
    print("\nGenerating plots...")

    fig = plt.figure(figsize=(24, 20))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    wl = np.array(WL_ALL)

    # --- 7a. Reference templates ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(wl, template_srha, 'r-', linewidth=2.5, label='SRHA (Suwannee)')
    ax1.plot(wl, template_srfa, 'b-', linewidth=2.5, label='SRFA (Suwannee)')
    if template_ha is not None:
        ax1.plot(wl, template_ha, 'r--', linewidth=1.5, alpha=0.6, label='HA (all)')
    if template_fa is not None:
        ax1.plot(wl, template_fa, 'b--', linewidth=1.5, alpha=0.6, label='FA (all)')
    ax1.set_xlabel("Wavelength (nm)")
    ax1.set_ylabel("Absorbance (mAU)")
    ax1.set_title("Reference Templates (HS fraction)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- 7b. Normalized templates ---
    ax1b = fig.add_subplot(gs[0, 1])
    idx254 = np.argmin(np.abs(wl - 254))
    for tmpl, name, color, ls in [(template_srha, 'SRHA', 'red', '-'),
                                   (template_srfa, 'SRFA', 'blue', '-'),
                                   (template_ha, 'HA all', 'red', '--'),
                                   (template_fa, 'FA all', 'blue', '--')]:
        if tmpl is not None and tmpl[idx254] > 0.01:
            ax1b.plot(wl, tmpl / tmpl[idx254], color=color, linestyle=ls,
                     linewidth=2 if ls == '-' else 1.5, alpha=0.8 if ls == '-' else 0.5,
                     label=name)
    ax1b.set_xlabel("Wavelength (nm)")
    ax1b.set_ylabel("A / A254")
    ax1b.set_title("Normalized Templates (shape)")
    ax1b.legend()
    ax1b.grid(True, alpha=0.3)

    # --- 7c. cos(SRHA) vs cos(SRFA) scatter ---
    ax2 = fig.add_subplot(gs[0, 2])
    # Water types
    water_types = ["FR", "PTLL", "PTT", "SK", "ATL_2024", "EX", "LQ", "EXT",
                   "EP", "SP", "POST_O3", "EP_PTL", "EP_PTT"]
    ref_types = ["SRHA", "SRFA", "HA", "FA", "MIX"]
    cmap = plt.cm.get_cmap('tab20', 30)
    type_list = sorted(df['Type'].unique())
    type_colors = {t: cmap(i) for i, t in enumerate(type_list)}
    type_colors.update({'HA': '#E74C3C', 'FA': '#3498DB', 'SRHA': '#C0392B',
                        'SRFA': '#2980B9', 'MIX': '#9B59B6'})

    for t in type_list:
        if t in ref_types:
            continue
        mask = df['Type'] == t
        if mask.sum() < 2:
            continue
        ax2.scatter(df.loc[mask, 'cos_SRFA'], df.loc[mask, 'cos_SRHA'],
                   s=12, alpha=0.4, color=type_colors.get(t, 'gray'),
                   label=f"{t} ({mask.sum()})")
    for t in ref_types:
        mask = df['Type'] == t
        if mask.sum() < 1:
            continue
        ax2.scatter(df.loc[mask, 'cos_SRFA'], df.loc[mask, 'cos_SRHA'],
                   s=150, marker='*', color=type_colors.get(t, 'gray'),
                   edgecolors='black', zorder=10, label=f"{t} ({mask.sum()})")
    # Diagonal
    lims = [min(ax2.get_xlim()[0], ax2.get_ylim()[0]),
            max(ax2.get_xlim()[1], ax2.get_ylim()[1])]
    ax2.plot(lims, lims, 'k--', alpha=0.3, label="Equal similarity")
    ax2.set_xlabel("cos(SRFA) →  more FA-like")
    ax2.set_ylabel("cos(SRHA) →  more HA-like")
    ax2.set_title("Humic Character Space")
    ax2.legend(fontsize=5, ncol=2, loc='lower right')
    ax2.grid(True, alpha=0.3)

    # --- 7d. %HA boxplot per type ---
    ax3 = fig.add_subplot(gs[1, 0:2])
    plot_types = [t for t in type_list if df[df['Type']==t].shape[0] >= 3]
    plot_types = sorted(plot_types, key=lambda t: df[df['Type']==t]['%HA_character'].median())
    box_data = [df[df['Type']==t]['%HA_character'].values for t in plot_types]
    bp = ax3.boxplot(box_data, vert=True, widths=0.6, patch_artist=True)
    for i, (patch, t) in enumerate(zip(bp['boxes'], plot_types)):
        med = df[df['Type']==t]['%HA_character'].median()
        if med > 50.2:
            patch.set_facecolor('#FADBD8')
        elif med < 49.8:
            patch.set_facecolor('#D6EAF8')
        else:
            patch.set_facecolor('#D5F5E3')
        if t in ref_types:
            patch.set_edgecolor('black')
            patch.set_linewidth(2)
    ax3.set_xticklabels(plot_types, rotation=45, ha='right', fontsize=8)
    ax3.axhline(50, color='gray', linestyle='--', linewidth=1, label="50% (neutral)")
    ax3.set_ylabel("% HA character")
    ax3.set_title("Humic Character by Sample Type\n(>50% = HA-dominant, <50% = FA-dominant)")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, axis='y')

    # --- 7e. Character pie chart ---
    ax4 = fig.add_subplot(gs[1, 2])
    # Only for water samples
    water_df = df[df['Type'].isin(water_types)]
    char_counts = water_df['Character'].value_counts()
    colors_pie = {'HA-dominant': '#E74C3C', 'FA-dominant': '#3498DB', 'Mixed': '#2ECC71'}
    ax4.pie(char_counts.values,
            labels=[f"{k}\n({v}, {v/len(water_df)*100:.0f}%)" for k, v in char_counts.items()],
            colors=[colors_pie.get(k, 'gray') for k in char_counts.index],
            autopct='', startangle=90, textprops={'fontsize': 10})
    ax4.set_title(f"Water Samples Character (n={len(water_df)})")

    # --- 7f. Temporal evolution per plant ---
    ax5 = fig.add_subplot(gs[2, 0:2])
    for wt, color in [("FR", "#E74C3C"), ("PTLL", "#3498DB"), ("PTT", "#2ECC71"),
                       ("ATL_2024", "#F39C12"), ("SK", "#9B59B6")]:
        mask = df['Type'] == wt
        if mask.sum() < 5:
            continue
        sub = df[mask].copy()
        # Extract SEQ number
        sub['seq_num'] = sub['SEQ'].apply(
            lambda s: int(re.match(r'(\d+)', s).group(1)) if re.match(r'(\d+)', s) else 0)
        sub = sub.sort_values('seq_num')
        ax5.scatter(sub['seq_num'], sub['%HA_character'], s=15, alpha=0.4, color=color)
        # Rolling mean
        if len(sub) > 5:
            from scipy.ndimage import uniform_filter1d
            smooth = uniform_filter1d(sub['%HA_character'].values.astype(float), size=min(7, len(sub)//2))
            ax5.plot(sub['seq_num'].values, smooth, color=color, linewidth=2.5, label=wt)
        else:
            ax5.plot(sub['seq_num'].values, sub['%HA_character'].values, color=color,
                    linewidth=1.5, label=wt)
    ax5.axhline(50, color='gray', linestyle='--', linewidth=1)
    ax5.set_xlabel("SEQ number (chronological)")
    ax5.set_ylabel("% HA character")
    ax5.set_title("Temporal Evolution of Humic Character")
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)

    # --- 7g. Summary text ---
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')
    txt = "HUMIC CHARACTER INDEX\n"
    txt += "=" * 30 + "\n\n"
    txt += "Reference: IHSS Suwannee River\n"
    txt += f"  SRHA: {len(srha_specs)} spectra\n"
    txt += f"  SRFA: {len(srfa_specs)} spectra\n\n"
    txt += f"Total samples: {len(df)}\n"
    txt += f"  HA-dominant: {(df['Character']=='HA-dominant').sum()}\n"
    txt += f"  FA-dominant: {(df['Character']=='FA-dominant').sum()}\n"
    txt += f"  Mixed:       {(df['Character']=='Mixed').sum()}\n\n"
    txt += "Interpretation:\n"
    txt += "  cos(SRHA) > cos(SRFA): HA-like\n"
    txt += "    → Higher MW, more aromatic\n"
    txt += "    → Terrestrial/allochthonous\n\n"
    txt += "  cos(SRFA) > cos(SRHA): FA-like\n"
    txt += "    → Lower MW, less aromatic\n"
    txt += "    → Autochthonous/microbial\n\n"
    txt += "  Mixed: similar to both\n"
    txt += "    → Balanced NOM composition\n\n"
    txt += f"Output: humic_index.xlsx\n"
    txt += f"  Per_Sample: all {len(df)} results\n"
    txt += f"  Per_Type: summary by type\n"
    txt += f"  Per_SEQ: summary by sequence\n"
    txt += f"  Templates: reference spectra"

    ax6.text(0.05, 0.95, txt, transform=ax6.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.savefig(os.path.join(OUTPUT_DIR, "07_humic_index_overview.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 07_humic_index_overview.png")

    # === STEP 8: Detailed per-plant comparison ===
    print("\nGenerating per-plant detail...")

    plant_types = [t for t in ["FR", "PTLL", "PTT", "SK", "ATL_2024", "EX", "LQ", "EXT"]
                   if df[df['Type']==t].shape[0] >= 5]
    n_plants = len(plant_types)
    fig, axes = plt.subplots(2, (n_plants+1)//2, figsize=(6*((n_plants+1)//2), 10))
    axes = axes.flatten()

    for idx, plant in enumerate(plant_types):
        ax = axes[idx]
        sub = df[df['Type'] == plant]
        specs = [r['hs_spectrum'] for r in records if r['type'] == plant]

        if not specs:
            continue

        mean_spec = np.mean(specs, axis=0)
        std_spec = np.std(specs, axis=0)

        # Normalized
        if mean_spec[idx254] > 0.01:
            norm_mean = mean_spec / mean_spec[idx254]
            norm_std = std_spec / mean_spec[idx254]
        else:
            norm_mean = mean_spec
            norm_std = std_spec

        srha_norm = template_srha / template_srha[idx254]
        srfa_norm = template_srfa / template_srfa[idx254]

        ax.fill_between(wl, norm_mean - norm_std, norm_mean + norm_std,
                        alpha=0.2, color='#3498DB')
        ax.plot(wl, norm_mean, '#3498DB', linewidth=2.5, label=f"{plant} (n={len(sub)})")
        ax.plot(wl, srha_norm, 'r--', linewidth=1.5, alpha=0.7, label="SRHA")
        ax.plot(wl, srfa_norm, 'b--', linewidth=1.5, alpha=0.7, label="SRFA")

        med_ha = sub['%HA_character'].median()
        cos_h = sub['cos_SRHA'].mean()
        cos_f = sub['cos_SRFA'].mean()
        char = sub['Character'].mode().iloc[0]

        ax.set_title(f"{plant}\n%HA={med_ha:.1f}% | cos(SRHA)={cos_h:.4f} cos(SRFA)={cos_f:.4f}\n→ {char}",
                     fontsize=9)
        ax.set_xlabel("λ (nm)", fontsize=8)
        ax.set_ylabel("A/A254", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(200, 400)

    # Hide unused axes
    for idx in range(n_plants, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "08_plant_humic_detail.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 08_plant_humic_detail.png")

    print(f"\n{'='*70}")
    print("DONE — All results in:", OUTPUT_DIR)
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
