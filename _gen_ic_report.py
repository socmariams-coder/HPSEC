"""Generate PDF report for IC(ppb) analysis across all MasterFiles."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Data collected from the scan (seq, n_nonzero, n_total, pct, ic_max, ic_mean_nz)
results_ic = [
    ('072B_SEQ_BP', 3653, 3653, 100.0, 0.3, 0.2),
    ('072_SEQ', 20861, 24607, 84.8, 0.3, 0.1),
    ('073B_SEQ_BP', 2164, 2164, 100.0, 0.3, 0.2),
    ('073_SEQ', 14409, 14409, 100.0, 0.5, 0.3),
    ('074B_SEQ_BP', 2465, 2465, 100.0, 0.4, 0.3),
    ('074_SEQ', 19110, 19110, 100.0, 0.5, 0.4),
    ('075B_SEQ_BP', 3299, 3299, 100.0, 0.5, 0.3),
    ('075_SEQ', 17088, 17088, 100.0, 0.7, 0.5),
    ('076_SEQ', 18289, 18289, 100.0, 0.6, 0.5),
    ('077B_SEQ_BP', 958, 958, 100.0, 0.4, 0.3),
    ('077_SEQ', 16628, 16628, 100.0, 0.8, 0.5),
    ('079_SEQ', 4364, 4364, 100.0, 0.7, 0.5),
    ('080_SEQ', 19550, 19550, 100.0, 7.3, 1.5),
    ('107_SEQ_BP', 2526, 2526, 100.0, 11.2, 2.6),
    ('108_SEQ_BP', 1290, 2964, 43.5, 2.6, 0.3),
    ('111_SEQ_BP_CAL', 908, 7856, 11.6, 4.1, 0.8),
    ('112_SEQ', 498, 24463, 2.0, 14.2, 2.4),
    ('114_SEQ_BP_CAL', 143, 3511, 4.1, 3.1, 1.0),
    ('124_SEQ', 2307, 8377, 27.5, 0.5, 0.2),
    ('125_SEQ', 3909, 20448, 19.1, 0.3, 0.1),
    ('127_SEQ', 85, 23184, 0.4, 0.5, 0.2),
    ('128_SEQ', 1567, 9388, 16.7, 5.7, 1.5),
    ('129_SEQ', 29, 4343, 0.7, 0.6, 0.3),
    ('136_SEQ', 1746, 25118, 7.0, 3.0, 0.8),
    ('142_SEQ', 616, 5047, 12.2, 10.6, 1.4),
    ('144_SEQ', 569, 17234, 3.3, 8.5, 1.3),
    ('145_SEQ_BP', 65, 3153, 2.1, 2.0, 0.8),
    ('147_SEQ', 1572, 20311, 7.7, 7.8, 1.7),
    ('152_SEQ_BP', 19, 2080, 0.9, 0.1, 0.0),
    ('156_SEQ_BP', 35, 6479, 0.5, 0.2, 0.1),
    ('164_SEQ', 914, 15424, 5.9, 0.5, 0.1),
    ('167_SEQ_BP', 277, 3315, 8.4, 0.4, 0.2),
    ('169_SEQ_BP', 2295, 2295, 100.0, 2.9, 0.5),
    ('173_SEQ_BP', 2069, 2069, 100.0, 15.4, 3.9),
    ('187_SEQ', 199, 5703, 3.5, 5.3, 2.3),
    ('195_SEQ', 2562, 14196, 18.0, 7.6, 1.1),
    ('199_SEQ', 14454, 14454, 100.0, 0.6, 0.3),
    ('200_SEQ', 24682, 24682, 100.0, 0.4, 0.2),
    ('201_SEQ', 29984, 30699, 97.7, 0.4, 0.2),
    ('203_SEQ', 13022, 13022, 100.0, 0.7, 0.4),
    ('205_SEQ', 4375, 4375, 100.0, 11.4, 9.3),
    ('206_SEQ', 19492, 19492, 100.0, 10.0, 8.5),
    ('208_SEQ_BP', 4431, 4431, 100.0, 11.1, 9.7),
    ('209B_SEQ', 22321, 22321, 100.0, 10.6, 8.3),
    ('209_SEQ', 22321, 22321, 100.0, 10.6, 8.3),
    ('210_SEQ', 16666, 16666, 100.0, 13.2, 9.1),
    ('212_SEQ', 13827, 13827, 100.0, 22.2, 8.5),
    ('213_SEQ', 23536, 23536, 100.0, 11.6, 8.6),
    ('214_SEQ_BP', 5460, 5460, 100.0, 11.3, 9.2),
    ('216_SEQ', 28746, 28746, 100.0, 11.9, 8.8),
    ('217_SEQ_BP', 4946, 4946, 100.0, 13.9, 11.8),
    ('218_SEQ', 36552, 36552, 100.0, 15.8, 8.7),
    ('219_SEQ', 20208, 20208, 100.0, 13.1, 8.6),
    ('220_SEQ_BP', 6168, 6168, 100.0, 12.9, 10.2),
    ('221_SEQ_BP', 3605, 3605, 100.0, 12.0, 10.9),
    ('222_SEQ', 15847, 15847, 100.0, 13.7, 8.7),
    ('223_SEQ', 18462, 18462, 100.0, 14.3, 8.5),
    ('224_SEQ', 5906, 5906, 100.0, 12.9, 10.8),
    ('225_SEQ_BP', 1815, 1815, 100.0, 12.0, 8.9),
    ('226_SEQ_BP', 4018, 4018, 100.0, 18.7, 10.0),
    ('227_SEQ', 13305, 13305, 100.0, 9.8, 8.4),
    ('228_SEQ', 28108, 28108, 100.0, 10.9, 8.4),
    ('229_SEQ_BP', 2147, 2147, 100.0, 10.9, 10.1),
    ('230_SEQ', 41832, 41832, 100.0, 15.3, 9.4),
    ('231_SEQ_BP', 5511, 5511, 100.0, 11.7, 9.5),
    ('232_SEQ', 18524, 18524, 100.0, 12.5, 10.0),
    ('233_SEQ', 22231, 22231, 100.0, 11.0, 9.3),
    ('234_SEQ', 40712, 40712, 100.0, 17.4, 9.0),
    ('235_SEQ_BP', 3943, 3943, 100.0, 11.0, 9.0),
    ('236B_SEQ', 4263, 4263, 100.0, 11.4, 8.8),
    ('236_SEQ', 22572, 22572, 100.0, 10.1, 8.2),
    ('239B_SEQ', 5906, 5906, 100.0, 10.4, 8.9),
    ('239_SEQ', 18294, 18294, 100.0, 11.6, 8.5),
    ('244_SEQ', 15196, 15196, 100.0, 1.7, 1.4),
    ('246_SEQ', 41997, 41997, 100.0, 4.4, 0.9),
    ('247_SEQ_BP', 5461, 5461, 100.0, 0.7, 0.5),
    ('248_SEQ', 15121, 15121, 100.0, 0.8, 0.6),
    ('249B_SEQ', 3919, 3919, 100.0, 0.8, 0.5),
    ('249C_SEQ', 3919, 3919, 100.0, 0.8, 0.5),
    ('249_SEQ_BP', 3919, 3919, 100.0, 0.8, 0.5),
    ('250_SEQ', 17072, 17072, 100.0, 0.8, 0.6),
    ('262B_SEQ', 5, 19651, 0.0, 0.1, 0.1),
    ('262_SEQ', 1, 19119, 0.0, 0.0, 0.0),
    ('266_SEQ', 2, 38870, 0.0, 0.0, 0.0),
    ('267_SEQ', 1, 34058, 0.0, 0.0, 0.0),
    ('273_SEQ_BP', 1741, 4923, 35.4, 0.8, 0.3),
    ('274_SEQ', 2, 39206, 0.0, 0.0, 0.0),
    ('275_SEQ', 86, 36875, 0.2, 0.2, 0.1),
    ('276B_SEQ', 83, 25671, 0.3, 0.3, 0.1),
    ('276_SEQ', 70, 15987, 0.4, 0.2, 0.1),
    ('278_SEQ', 10221, 37508, 27.3, 0.8, 0.1),
    ('279B_SEQ_BP', 3, 4368, 0.1, 0.1, 0.1),
    ('281_SEQ_BP', 8808, 16811, 52.4, 0.3, 0.0),
    ('282B_SEQ', 4882, 22107, 22.1, 0.3, 0.0),
    ('282_SEQ', 16332, 21623, 75.5, 1.0, 0.2),
    ('283_SEQ', 2034, 23991, 8.5, 0.4, 0.0),
    ('284_SEQ_BP', 1, 13715, 0.0, 0.0, 0.0),
    ('285_SEQ', 5260, 37068, 14.2, 0.7, 0.1),
    ('288_SEQ', 20, 39357, 0.1, 0.1, 0.1),
    ('289_SEQ_BP', 3, 17108, 0.0, 0.0, 0.0),
    ('290_SEQ', 37, 20571, 0.2, 0.2, 0.1),
    ('291_SEQ', 24, 22114, 0.1, 0.1, 0.0),
]

no_ic_seqs = [
    '081_SEQ','082_SEQ','083_SEQ','084_SEQ','085_SEQ','086_SEQ','087_SEQ',
    '088_SEQ','089_SEQ','090_SEQ','091_SEQ','093_SEQ','094_SEQ','095_SEQ',
    '096_SEQ','097_SEQ','098_SEQ','099_SEQ','100_SEQ','101_SEQ','102_SEQ',
    '103_SEQ','104_SEQ','105_SEQ','106_SEQ','109_SEQ_BP','113_SEQ_BP_CAL',
    '115_SEQ','116_SEQ','117_SEQ','118_SEQ','119_SEQ','120_SEQ','121_SEQ',
    '122_SEQ','123_SEQ','126_SEQ','131_SEQ','132_SEQ','133_SEQ','134_SEQ',
    '135_SEQ','139_SEQ','140_SEQ','141_SEQ','143_SEQ','146_SEQ','148_SEQ',
    '149_SEQ_BP','153_SEQ_BP','154_SEQ','157_SEQ','158_SEQ','159_SEQ',
    '160_SEQ_BP','161_SEQ','162_SEQ','163_SEQ','165_SEQ','168_SEQ_BP',
    '170_SEQ_BP','171_SEQ_BP','172_SEQ_BP','251_SEQ_BP','252_SEQ',
]

error_seqs = ['130_SEQ', '137_SEQ', '138_SEQ', '166_SEQ', '204_SEQ']


def seq_num(name):
    n = name.split('_')[0].replace('B', '').replace('C', '').replace('D', '')
    try:
        return int(n)
    except ValueError:
        return 0


# Categorize
significant = [(s, nz, nt, p, mx, mn) for s, nz, nt, p, mx, mn in results_ic if mn >= 1.0]
low_signal = [(s, nz, nt, p, mx, mn) for s, nz, nt, p, mx, mn in results_ic if 0 < mn < 1.0 and p > 5]
negligible = [(s, nz, nt, p, mx, mn) for s, nz, nt, p, mx, mn in results_ic if mn < 1.0 and p <= 5]

pdf_path = 'C:/Users/Lequia/Desktop/HPSEC/IC_ppb_analysis_report.pdf'
with PdfPages(pdf_path) as pdf:

    # === PAGE 1: Executive summary ===
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis('off')

    ax.text(0.5, 0.95, 'IC(ppb) — Columna G del full 2-TOC dels MasterFiles',
            transform=ax.transAxes, fontsize=16, fontweight='bold', ha='center', va='top')
    ax.text(0.5, 0.91, 'HPSEC Suite — Febrer 2026',
            transform=ax.transAxes, fontsize=11, ha='center', va='top', color='gray')

    summary = (
        "RESUM EXECUTIU\n"
        "\n"
        f"Total MasterFiles analitzats:    172\n"
        f"  - Amb dades IC(ppb) != 0:      102  (59.3%)\n"
        f"  - Sense dades IC (tot zeros):    65  (37.8%)\n"
        f"  - Errors (sense columna IC):      5  (2.9%)\n"
        "\n"
        "CATEGORIES:\n"
        "\n"
        f"1. IC SIGNIFICATIU (mean >= 1 ppb):  {len(significant)} SEQs\n"
        f"   Rang IC mitja: {min(s[5] for s in significant):.1f} - {max(s[5] for s in significant):.1f} ppb\n"
        f"   Bloc principal: SEQs 205-239 (IC mitja ~8-11 ppb, 100% files nonzero)\n"
        f"   Altres: 080, 107, 112, 128, 136, 142, 144, 147, 169, 173, 187, 195\n"
        "\n"
        f"2. SENYAL BAIX (mean < 1 ppb, >5% nonzero):  {len(low_signal)} SEQs\n"
        f"   Principalment SEQs 072-079 (IC 0.1-0.5) i 244-250 (IC 0.5-1.4)\n"
        f"   i algunes recents (273, 278, 281-285)\n"
        "\n"
        f"3. NEGLIGIBLE (mean < 1 ppb, <=5% files nonzero):  {len(negligible)} SEQs\n"
        f"   Valors esporadics, probablement soroll del detector\n"
        "\n"
        f"4. TOT ZEROS:  65 SEQs  (081-106, 109-123, 126-135, 139-165, 168-172, 251-252)\n"
        "\n"
        f"5. ERRORS (sense columna IC):  {', '.join(error_seqs)}\n"
        "\n"
        "OBSERVACIO CLAU:\n"
        "El bloc SEQ 205-239 mostra IC consistentment alt (~8-11 ppb mitja).\n"
        "Abans i despres d'aquest bloc, IC es baix o zero.\n"
        "Possible canvi de configuracio del TOC Sievers M9e en aquest periode."
    )

    ax.text(0.05, 0.84, summary, transform=ax.transAxes, fontsize=9.5,
            fontfamily='monospace', va='top')

    pdf.savefig(fig)
    plt.close()

    # === PAGE 2: Timeline plot ===
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11.69, 8.27),
                                    gridspec_kw={'height_ratios': [2, 1]})
    fig.suptitle('IC(ppb) — Evolucio temporal per seqüencia', fontsize=14, fontweight='bold')

    sorted_data = sorted(results_ic, key=lambda x: seq_num(x[0]))
    x_nums = [seq_num(s[0]) for s in sorted_data]
    y_mean = [s[5] for s in sorted_data]
    y_max = [s[4] for s in sorted_data]
    y_pct = [s[3] for s in sorted_data]

    colors = []
    for s in sorted_data:
        if s[5] >= 1.0:
            colors.append('red')
        elif s[3] > 5:
            colors.append('orange')
        else:
            colors.append('lightblue')

    ax1.bar(x_nums, y_mean, color=colors, width=0.8, alpha=0.8, label='IC mitja (nonzero)')
    ax1.scatter(x_nums, y_max, color='darkred', s=10, zorder=5, label='IC max')
    ax1.set_ylabel('IC (ppb)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax1.annotate('Bloc 205-239\n(IC alt)', xy=(222, 10), fontsize=9, ha='center',
                 color='red', fontweight='bold')

    ax2.bar(x_nums, y_pct, color=colors, width=0.8, alpha=0.8)
    ax2.set_ylabel('% files nonzero')
    ax2.set_xlabel('SEQ number')
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(50, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

    # === PAGE 3: Table - Significant IC ===
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis('off')
    ax.text(0.5, 0.97, 'SEQs amb IC significatiu (mean >= 1 ppb)',
            fontsize=14, fontweight='bold', ha='center', va='top', transform=ax.transAxes)

    col_labels = ['SEQ', 'N nonzero', 'N total', '% nonzero', 'IC max', 'IC mean (nz)']
    cell_data = [[s[0], f'{s[1]:,}', f'{s[2]:,}', f'{s[3]:.1f}%', f'{s[4]:.1f}', f'{s[5]:.1f}']
                 for s in sorted(significant, key=lambda x: seq_num(x[0]))]

    table = ax.table(cellText=cell_data, colLabels=col_labels, loc='center',
                     cellLoc='center', colWidths=[0.22, 0.13, 0.13, 0.13, 0.13, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)

    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#4472C4')
        table[0, j].set_text_props(color='white', fontweight='bold')

    for i, row in enumerate(cell_data):
        mean_val = float(row[5])
        if mean_val >= 8:
            for j in range(len(col_labels)):
                table[i + 1, j].set_facecolor('#FCE4EC')
        elif mean_val >= 3:
            for j in range(len(col_labels)):
                table[i + 1, j].set_facecolor('#FFF3E0')

    pdf.savefig(fig)
    plt.close()

    # === PAGE 4: Low/trace/zero ===
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis('off')
    ax.text(0.5, 0.97, 'SEQs amb IC baix / negligible / zero',
            fontsize=14, fontweight='bold', ha='center', va='top', transform=ax.transAxes)

    y = 0.90
    ax.text(0.05, y, f'IC baix (mean < 1 ppb, >5% nonzero): {len(low_signal)} SEQs',
            fontsize=10, fontweight='bold', transform=ax.transAxes)
    y -= 0.04
    low_text = ', '.join(sorted([s[0] for s in low_signal], key=lambda x: seq_num(x)))
    ax.text(0.05, y, low_text, fontsize=7.5, transform=ax.transAxes, va='top',
            wrap=True)

    y -= 0.12
    ax.text(0.05, y, f'IC negligible (mean < 1 ppb, <=5% nonzero): {len(negligible)} SEQs',
            fontsize=10, fontweight='bold', transform=ax.transAxes)
    y -= 0.04
    neg_text = ', '.join(sorted([s[0] for s in negligible], key=lambda x: seq_num(x)))
    ax.text(0.05, y, neg_text, fontsize=7.5, transform=ax.transAxes, va='top',
            wrap=True)

    y -= 0.12
    ax.text(0.05, y, f'IC = 0 (tot zeros): {len(no_ic_seqs)} SEQs',
            fontsize=10, fontweight='bold', transform=ax.transAxes)
    y -= 0.04
    zero_text = ', '.join(sorted(no_ic_seqs, key=lambda x: seq_num(x)))
    ax.text(0.05, y, zero_text, fontsize=7.5, transform=ax.transAxes, va='top',
            wrap=True)

    y -= 0.16
    ax.text(0.05, y, f'Errors (sense columna IC al full 2-TOC): {len(error_seqs)} SEQs',
            fontsize=10, fontweight='bold', transform=ax.transAxes)
    y -= 0.04
    ax.text(0.05, y, ', '.join(error_seqs), fontsize=8, transform=ax.transAxes)

    y -= 0.08
    ax.text(0.05, y, 'NOTES:', fontsize=10, fontweight='bold', transform=ax.transAxes)
    y -= 0.04
    notes = (
        "- El bloc SEQs 205-239 te IC consistentment alt (8-11 ppb mitja). Possible canvi config TOC.\n"
        "- Les SEQs 072-080 tenen IC baix pero present (0.1-0.5 ppb, 100% nonzero).\n"
        "- SEQs 244-250 mostren IC intermedi (0.5-1.4 ppb, 100% nonzero) - possible transicio.\n"
        "- SEQs >=262 tornen a IC ~ 0 (o traces negligibles).\n"
        "- 5 SEQs no tenen columna IC al 2-TOC (format MasterFile antic?)."
    )
    ax.text(0.05, y, notes, fontsize=8.5, transform=ax.transAxes, va='top')

    pdf.savefig(fig)
    plt.close()

    # === PAGE 5: IC intra-sequence variability ===
    intra_seqs = [
        ('205_SEQ', 'C:/Users/Lequia/Desktop/HPSEC/Dades3/205_SEQ', 'IC alt (gen 2024)'),
        ('212_SEQ', 'C:/Users/Lequia/Desktop/HPSEC/Dades3/212_SEQ', 'IC alt, pics fins 22 ppb'),
        ('226_SEQ_BP', 'C:/Users/Lequia/Desktop/HPSEC/Dades3/226_SEQ_BP', 'IC alt, BP'),
        ('080_SEQ', 'C:/Users/Lequia/Desktop/HPSEC/Dades3/080_SEQ', 'IC baix variable (2021)'),
        ('282_SEQ', 'C:/Users/Lequia/Desktop/HPSEC/Dades3/282_SEQ', 'IC quasi zero (2025)'),
    ]

    fig, axes = plt.subplots(len(intra_seqs), 1, figsize=(11.69, 2.8 * len(intra_seqs)))
    fig.suptitle('IC(ppb) intra-sequencia: varia amb les mostres?', fontsize=14, fontweight='bold')

    for idx, (name, path, desc) in enumerate(intra_seqs):
        mf = None
        for f in os.listdir(path):
            if f.endswith('MasterFile.xlsx'):
                mf = os.path.join(path, f)
                break

        toc = pd.read_excel(mf, sheet_name='2-TOC', header=6)
        ic = toc['IC(ppb)'].astype(float).values
        toc_ppb = toc['TOC(ppb)'].astype(float).values
        x = np.arange(len(ic))

        ax = axes[idx]
        ax.plot(x, toc_ppb, 'b-', linewidth=0.3, alpha=0.5, label='TOC(ppb)')
        ax2 = ax.twinx()
        ax2.plot(x, ic, 'r-', linewidth=0.5, alpha=0.8, label='IC(ppb)')
        title = '%s -- %s  (mean=%.1f, std=%.1f, min=%.1f, max=%.1f, CV=%.0f%%)' % (
            name, desc, ic.mean(), ic.std(), ic.min(), ic.max(),
            ic.std() / max(ic.mean(), 0.001) * 100)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel('TOC (ppb)', color='blue', fontsize=8)
        ax2.set_ylabel('IC (ppb)', color='red', fontsize=8)
        ax.legend(loc='upper left', fontsize=7)
        ax2.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.2)
        if idx == len(intra_seqs) - 1:
            ax.set_xlabel('Fila 2-TOC (cada punt = 1 mesura TOC, ~4 s)')

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

    # === PAGE 6: Firmware + conclusions ===
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis('off')
    ax.text(0.5, 0.95, 'Firmware TOC i conclusions',
            fontsize=16, fontweight='bold', ha='center', va='top', transform=ax.transAxes)

    text = (
        "FIRMWARE TOC SIEVERS M9e\n"
        "\n"
        "  - Firmware 1.11 Rev 9:   SEQs 072-080  (2021)\n"
        "  - Firmware 1.13 Rev 42:  SEQs 107-293  (gen 2022 - feb 2026)\n"
        "  - Canvi firmware:        entre SEQ 080 i 107 (entre 2021 i gen 2022)\n"
        "\n"
        "  El canvi de firmware NO explica el patro IC:\n"
        "  - El firmware 1.13 cobreix des de 2022 fins avui\n"
        "  - El bloc IC alt (205-239) es de gen-jun 2024, 2 anys despres del canvi\n"
        "  - SEQs amb firmware identic (1.13) tenen IC=0 o IC=9 segons el periode\n"
        "\n"
        "\n"
        "VARIABILITAT INTRA-SEQUENCIA\n"
        "\n"
        "  IC NO es constant dins una sequencia -- canvia amb les mostres:\n"
        "\n"
        "  - Bloc 205-239 (IC alt):\n"
        "      IC basal ~8-10 ppb amb pics fins 22 ppb\n"
        "      Els pics IC correlacionen amb pics TOC (mostres amb C inorganic)\n"
        "      CV intra-seq: 11-14%\n"
        "\n"
        "  - 080_SEQ (IC baix):\n"
        "      IC basal ~0.5 ppb amb pics fins 7.3 ppb\n"
        "      Pics IC coincideixen amb injeccions de mostres riques en carbonats\n"
        "      CV intra-seq: 51% (molt variable)\n"
        "\n"
        "  - 282_SEQ (IC quasi zero):\n"
        "      IC ~ 0 amb punxes esporadiques fins 1 ppb\n"
        "      CV intra-seq: 116% (basicament soroll)\n"
        "\n"
        "\n"
        "HIPOTESIS CANVI IC\n"
        "\n"
        "  El nivell basal d'IC al bloc 205-239 (~8-10 ppb) suggereix:\n"
        "\n"
        "  1. Canvi de mode TOC: TC -> NPOC (o invers)\n"
        "     En mode TC, el Sievers mesura TOC = TC - IC, i reporta IC separat.\n"
        "     En mode NPOC (purgant IC amb acid), IC reportat pot ser zero.\n"
        "\n"
        "  2. Canvi en la linia d'acid o en el purge del Sievers M9e\n"
        "     Si el purge d'IC no funciona be, IC queda al senyal.\n"
        "\n"
        "  3. Canvi en l'aigua MQ del carrier\n"
        "     Si l'aigua MQ te carbonats, IC basal puja.\n"
        "\n"
        "  Periode afectat: gen 2024 (SEQ 205) - jun 2024 (SEQ 239)\n"
        "  Transicio: jul-ago 2024 (SEQs 244-250, IC decreixent)\n"
        "  Retorn a zero: ago 2024 (SEQ 251+)\n"
        "\n"
        "\n"
        "IMPACTE EN L'ANALISI DOC\n"
        "\n"
        "  La Suite utilitza la columna TOC(ppb) (F), NO IC(ppb) (G).\n"
        "  Si el Sievers reporta TOC = TC - IC correctament, no hi ha impacte.\n"
        "  Pero si IC NO es restava del TC en aquell periode, el TOC reportat\n"
        "  estaria inflat en ~8-10 ppb (nivell basal IC), afectant baselines\n"
        "  i potencialment les areas integrades."
    )

    ax.text(0.05, 0.88, text, transform=ax.transAxes, fontsize=9,
            fontfamily='monospace', va='top')

    pdf.savefig(fig)
    plt.close()

    # === Helper: find MasterFile in a seq folder ===
    def find_masterfile(seq_path):
        for f in os.listdir(seq_path):
            if f.endswith('MasterFile.xlsx'):
                return os.path.join(seq_path, f)
        return None

    def plot_injections_page(pdf, seq_name, seq_path, title_suffix=''):
        mf = find_masterfile(seq_path)
        if mf is None:
            return

        toc_data = pd.read_excel(mf, sheet_name='2-TOC', header=6)
        calc_data = pd.read_excel(mf, sheet_name='4-TOC_CALC')

        toc_v = toc_data['TOC(ppb)'].astype(float).values
        ic_v = toc_data['IC(ppb)'].astype(float).values

        inj_list = sorted([i for i in calc_data['Inj_Index'].unique() if i > 0])
        if len(inj_list) == 0:
            return

        n = len(inj_list)
        step = max(1, (n - 1) // 5)
        picked = [inj_list[min(i * step, n - 1)] for i in range(6)]
        # Deduplicate while preserving order
        seen = set()
        picked = [x for x in picked if not (x in seen or seen.add(x))]
        # Pad if needed
        while len(picked) < 6:
            for candidate in inj_list:
                if candidate not in picked:
                    picked.append(candidate)
                    break
            else:
                break

        fig, axes_grid = plt.subplots(3, 2, figsize=(11.69, 8.27))
        fig.suptitle('%s -- Perfils DOC i IC per injeccio %s' % (seq_name, title_suffix),
                     fontsize=13, fontweight='bold')
        axes_flat = axes_grid.flatten()

        for ax_idx in range(6):
            ax = axes_flat[ax_idx]
            if ax_idx >= len(picked):
                ax.axis('off')
                continue

            inj_num = picked[ax_idx]
            mask = calc_data['Inj_Index'] == inj_num
            sub = calc_data[mask]
            if len(sub) == 0:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
                continue

            rows = sub['TOC_Row'].values
            # Clip to valid range
            rows = rows[rows < len(toc_v)]
            if len(rows) == 0:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
                continue

            t_rel = sub['Temps_Relatiu (min)'].astype(float).values[:len(rows)]
            y_toc = toc_v[rows]
            y_ic = ic_v[rows]

            sname = sub['Sample'].iloc[0] if 'Sample' in sub.columns else ''
            if pd.isna(sname) or sname is None:
                sname = ''

            ax.plot(t_rel, y_toc, 'b-', linewidth=0.8, alpha=0.8, label='TOC(ppb)')
            ax2_twin = ax.twinx()
            ax2_twin.plot(t_rel, y_ic, 'r-', linewidth=0.8, alpha=0.8, label='IC(ppb)')
            ax.set_title('Inj %d: %s' % (inj_num, sname), fontsize=9)
            ax.set_ylabel('TOC (ppb)', color='blue', fontsize=8)
            ax2_twin.set_ylabel('IC (ppb)', color='red', fontsize=8)
            ax.set_xlabel('Temps relatiu (min)', fontsize=7)
            ax.legend(loc='upper left', fontsize=6)
            ax2_twin.legend(loc='upper right', fontsize=6)
            ax.grid(True, alpha=0.2)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

    # === PAGE 7: 212_SEQ (high IC) ===
    plot_injections_page(pdf, '212_SEQ',
                         'C:/Users/Lequia/Desktop/HPSEC/Dades3/212_SEQ',
                         '(IC alt, gen 2024)')

    # === PAGE 8: 282_SEQ (zero IC, contrast) ===
    plot_injections_page(pdf, '282_SEQ',
                         'C:/Users/Lequia/Desktop/HPSEC/Dades3/282_SEQ',
                         '(IC ~ 0, contrast)')

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

print(f'PDF saved: {pdf_path}')
