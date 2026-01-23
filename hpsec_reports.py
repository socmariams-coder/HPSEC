"""
hpsec_reports.py - Generació de PDFs i informes HPSEC
======================================================

Mòdul centralitzat per generar informes professionals:
- REPORT_Consolidacio: Resum de consolidació amb punts DOC/DAD
- REPORT_Cromatogrames: Gràfics agrupats per rèplica
- REPORT_Calibracio: Informe de calibració KHP
- REPORT_Processament: Resultats QA/QC

Format: Estil científic, minimalista, optimitzat per impressió A4.
"""

import os
import re
import numpy as np
import pandas as pd
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


# =============================================================================
# CONFIGURACIÓ I ESTILS
# =============================================================================
LOGO_FILENAME = "logo STRs.png"  # Nom real del fitxer

# Colors corporatius
COLORS = {
    "primary": "#2E86AB",      # Blau principal
    "secondary": "#A23B72",    # Magenta
    "success": "#28a745",      # Verd
    "warning": "#ffc107",      # Groc
    "danger": "#dc3545",       # Vermell
    "light": "#f8f9fa",        # Gris clar
    "dark": "#343a40",         # Gris fosc
    "doc_direct": "#1f77b4",   # Blau DOC Direct
    "doc_uib": "#2ca02c",      # Verd DOC UIB
    "dad_254": "#d62728",      # Vermell A254
}

# Estils matplotlib per format científic
STYLE_CONFIG = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.titlesize": 12,
    "axes.linewidth": 0.5,
    "grid.linewidth": 0.3,
    "lines.linewidth": 0.8,
}

# Patrons per detectar mostres de control
CONTROL_PATTERNS = [
    r"^MQ",           # MilliQ
    r"^NaOH",         # NaOH
    r"^BLANK",        # Blank
    r"^BLK",          # Blank abreviat
    r"^H2O",          # Aigua
    r"^WATER",        # Water
    r"^STD",          # Standard (no KHP)
]


def get_logo_path():
    """Retorna el path al logo si existeix."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    logo_path = os.path.join(base_dir, LOGO_FILENAME)
    if os.path.exists(logo_path):
        return logo_path
    # Fallback: buscar variants
    for variant in ["Logo_STRs.png", "logo_strs.png", "LOGO_STRS.png"]:
        alt_path = os.path.join(base_dir, variant)
        if os.path.exists(alt_path):
            return alt_path
    return None


def apply_style():
    """Aplica estil científic a matplotlib."""
    plt.rcParams.update(STYLE_CONFIG)


def is_khp(name):
    """Detecta si és mostra KHP."""
    return "KHP" in str(name).upper()


def is_control(name):
    """Detecta si és mostra de control (MQ, NaOH, Blank, etc.)."""
    name_upper = str(name).upper().strip()
    for pattern in CONTROL_PATTERNS:
        if re.match(pattern, name_upper, re.IGNORECASE):
            return True
    return False


def extract_sample_base(name):
    """Extreu nom base de mostra (sense rèplica)."""
    name = str(name).strip()
    # Treure sufixos de rèplica
    match = re.match(r"^(.+?)(?:[_\-\s]?R\d+)?$", name, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return name


def sort_samples_for_report(samples):
    """
    Ordena mostres per l'informe: Normals → KHP → Controls.

    Args:
        samples: Llista de dicts amb clau 'mostra'

    Returns:
        Llista ordenada
    """
    normals = []
    khps = []
    controls = []

    for s in samples:
        mostra = s.get('mostra', '')
        if is_khp(mostra):
            khps.append(s)
        elif is_control(mostra):
            controls.append(s)
        else:
            normals.append(s)

    # Ordenar cada grup alfabèticament
    normals.sort(key=lambda x: x.get('mostra', ''))
    khps.sort(key=lambda x: x.get('mostra', ''))
    controls.sort(key=lambda x: x.get('mostra', ''))

    return normals + khps + controls


def group_replicates(samples):
    """
    Agrupa mostres per rèpliques.

    Args:
        samples: Llista de dicts amb 'mostra' i 'rep'

    Returns:
        Dict {base_name: [rep1_data, rep2_data, ...]}
    """
    groups = {}

    for s in samples:
        base = extract_sample_base(s.get('mostra', ''))
        if base not in groups:
            groups[base] = []
        groups[base].append(s)

    # Ordenar rèpliques dins cada grup
    for base in groups:
        groups[base].sort(key=lambda x: int(x.get('rep', 0) or 0))

    return groups


# =============================================================================
# FUNCIONS DE DIBUIX
# =============================================================================
def draw_header(fig, title, subtitle="", seq_name="", page_num=None, total_pages=None):
    """
    Dibuixa capçalera minimalista.
    """
    logo_path = get_logo_path()

    # Logo a l'esquerra (més gran)
    if logo_path:
        try:
            logo = plt.imread(logo_path)
            ax_logo = fig.add_axes([0.02, 0.91, 0.18, 0.08])
            ax_logo.imshow(logo)
            ax_logo.axis('off')
        except Exception:
            pass

    # Capçalera principal
    fig.text(0.5, 0.97, "HPSEC_Suite", ha='center', va='top',
             fontsize=12, fontweight='bold', color=COLORS["primary"])
    fig.text(0.5, 0.955, "Anàlisi de Matèria Orgànica per HPSEC-DAD-DOC", ha='center', va='top',
             fontsize=9, color=COLORS["dark"])

    # Títol del report
    fig.text(0.5, 0.925, title, ha='center', va='top',
             fontsize=11, fontweight='bold', color=COLORS["dark"])

    if subtitle:
        fig.text(0.5, 0.905, subtitle, ha='center', va='top',
                 fontsize=9, color='gray')

    # SEQ i paginació a la dreta
    if seq_name:
        fig.text(0.98, 0.97, f"SEQ {seq_name}", ha='right', va='top',
                 fontsize=9, fontweight='bold', color=COLORS["primary"])

    if page_num and total_pages:
        fig.text(0.98, 0.95, f"Pàg. {page_num}/{total_pages}", ha='right', va='top',
                 fontsize=8, color='gray')


def draw_footer(fig, text=""):
    """Dibuixa peu de pàgina minimalista."""
    # Línia separadora
    fig.add_artist(plt.Line2D([0.05, 0.95], [0.02, 0.02],
                              color='lightgray', linewidth=0.5,
                              transform=fig.transFigure))

    # Data i text
    date_str = datetime.now().strftime("%d/%m/%Y %H:%M")
    fig.text(0.05, 0.01, f"Generat: {date_str}", ha='left', va='bottom',
             fontsize=6, color='gray')

    if text:
        fig.text(0.95, 0.01, text, ha='right', va='bottom',
                 fontsize=6, color='gray')


# =============================================================================
# REPORT CONSOLIDACIÓ
# =============================================================================
def generate_consolidation_report(seq_path, xlsx_files, info, output_path=None):
    """
    Genera PDF de consolidació amb recompte de punts.

    Args:
        seq_path: Ruta a la carpeta SEQ
        xlsx_files: Llista de fitxers Excel consolidats
        info: Dict amb info de consolidació (mode, bp, seq, date, file_check)
        output_path: Path de sortida (opcional, per defecte CHECK/)

    Returns:
        Path del PDF generat
    """
    apply_style()

    # Preparar paths
    if output_path is None:
        check_path = os.path.join(seq_path, "CHECK")
        os.makedirs(check_path, exist_ok=True)
        output_path = check_path

    seq_name = info.get('seq', os.path.basename(seq_path))
    pdf_path = os.path.join(output_path, f"REPORT_Consolidacio_{seq_name}.pdf")

    # Llegir dades de tots els fitxers
    samples_data = []
    total_pts_direct = 0
    total_pts_uib = 0
    total_pts_dad = 0

    for f in sorted(xlsx_files):
        try:
            df_id = pd.read_excel(f, "ID", engine="openpyxl")
            id_dict = dict(zip(df_id["Camp"], df_id["Valor"]))

            mostra = str(id_dict.get("Mostra", ""))
            rep = str(id_dict.get("Rèplica", "-"))
            doc_mode = str(id_dict.get("DOC_MODE", ""))

            # Punts
            n_doc = int(id_dict.get("DOC_N_POINTS", 0) or 0)
            n_dad = int(id_dict.get("DAD_N_POINTS", 0) or 0)

            # Per DUAL, llegir punts separats si disponibles
            n_direct = n_doc  # Per defecte
            n_uib = 0

            # Llegir DOC sheet per comptar punts reals
            try:
                df_doc = pd.read_excel(f, "DOC", engine="openpyxl")
                if "DOC_Direct (mAU)" in df_doc.columns:
                    n_direct = df_doc["DOC_Direct (mAU)"].notna().sum()
                if "DOC_UIB (mAU)" in df_doc.columns:
                    n_uib = df_doc["DOC_UIB (mAU)"].notna().sum()
                elif "DOC (mAU)" in df_doc.columns and doc_mode == "UIB":
                    n_uib = df_doc["DOC (mAU)"].notna().sum()
                    n_direct = 0
            except:
                pass

            total_pts_direct += n_direct
            total_pts_uib += n_uib
            total_pts_dad += n_dad

            samples_data.append({
                'mostra': mostra,
                'rep': rep,
                'doc_mode': doc_mode,
                'n_direct': n_direct,
                'n_uib': n_uib,
                'n_dad': n_dad,
                'fitxer': os.path.basename(f),
            })
        except Exception as e:
            continue

    # Ordenar mostres
    samples_data = sort_samples_for_report(samples_data)

    with PdfPages(pdf_path) as pdf:
        # =================================================================
        # PÀGINA 1: Resum executiu
        # =================================================================
        fig = plt.figure(figsize=(8.27, 11.69))  # A4
        fig.patch.set_facecolor('white')

        draw_header(fig, "INFORME DE CONSOLIDACIÓ",
                   f"Seqüència {seq_name}", seq_name, 1, 2)

        # Informació general (taula compacta)
        ax_info = fig.add_axes([0.05, 0.68, 0.9, 0.18])
        ax_info.axis('off')

        mode_str = info.get('mode', 'N/A')
        method_str = "By-Pass (BP)" if info.get('bp', False) else "COLUMN"
        date_str = str(info.get('date', 'N/A'))

        khp_count = sum(1 for s in samples_data if is_khp(s['mostra']))
        control_count = sum(1 for s in samples_data if is_control(s['mostra']))
        sample_count = len(samples_data) - khp_count - control_count

        info_data = [
            ["PARÀMETRE", "VALOR", "PARÀMETRE", "VALOR"],
            ["Mode DOC", mode_str, "Mostres", str(sample_count)],
            ["Mètode", method_str, "KHP (estàndard)", str(khp_count)],
            ["Data SEQ", date_str[:10] if len(date_str) > 10 else date_str,
             "Controls (MQ/NaOH)", str(control_count)],
            ["Total fitxers", str(len(xlsx_files)), "─", "─"],
        ]

        tbl_info = ax_info.table(cellText=info_data, loc='center', cellLoc='center',
                                  colWidths=[0.22, 0.28, 0.22, 0.28])
        tbl_info.auto_set_font_size(False)
        tbl_info.set_fontsize(9)
        tbl_info.scale(1.0, 1.8)

        # Estil capçalera
        for j in range(4):
            tbl_info[(0, j)].set_facecolor(COLORS["primary"])
            tbl_info[(0, j)].set_text_props(color='white', fontweight='bold')

        # Recompte de punts (destacat)
        ax_pts = fig.add_axes([0.1, 0.48, 0.8, 0.15])
        ax_pts.axis('off')

        pts_data = [
            ["RECOMPTE DE PUNTS", "TOTAL", "MITJANA/MOSTRA"],
            ["DOC Direct", f"{total_pts_direct:,}",
             f"{total_pts_direct/max(len(samples_data),1):,.0f}" if total_pts_direct > 0 else "─"],
            ["DOC UIB", f"{total_pts_uib:,}",
             f"{total_pts_uib/max(len(samples_data),1):,.0f}" if total_pts_uib > 0 else "─"],
            ["DAD (espectral)", f"{total_pts_dad:,}",
             f"{total_pts_dad/max(len(samples_data),1):,.0f}" if total_pts_dad > 0 else "─"],
        ]

        tbl_pts = ax_pts.table(cellText=pts_data, loc='center', cellLoc='center',
                               colWidths=[0.4, 0.3, 0.3])
        tbl_pts.auto_set_font_size(False)
        tbl_pts.set_fontsize(10)
        tbl_pts.scale(1.0, 2.0)

        for j in range(3):
            tbl_pts[(0, j)].set_facecolor(COLORS["dark"])
            tbl_pts[(0, j)].set_text_props(color='white', fontweight='bold')

        # Verificació de fitxers (si hi ha info)
        file_check = info.get('file_check', {})
        if file_check:
            ax_check = fig.add_axes([0.1, 0.31, 0.8, 0.12])
            ax_check.axis('off')

            has_issues = file_check.get('has_issues', False)
            status_color = COLORS["danger"] if has_issues else COLORS["success"]
            status_text = "DISCREPÀNCIES DETECTADES" if has_issues else "VERIFICACIÓ CORRECTA"

            check_data = [
                ["VERIFICACIÓ FITXERS", "TROBATS", "USATS", "ORFES"],
                ["UIB", str(file_check.get('uib_found', 0)),
                 str(file_check.get('uib_used', 0)), str(file_check.get('uib_orphan', 0))],
                ["DAD", str(file_check.get('dad_found', 0)),
                 str(file_check.get('dad_used', 0)), str(file_check.get('dad_orphan', 0))],
            ]

            tbl_check = ax_check.table(cellText=check_data, loc='center', cellLoc='center',
                                       colWidths=[0.4, 0.2, 0.2, 0.2])
            tbl_check.auto_set_font_size(False)
            tbl_check.set_fontsize(9)
            tbl_check.scale(1.0, 1.6)

            for j in range(4):
                tbl_check[(0, j)].set_facecolor(COLORS["primary"])
                tbl_check[(0, j)].set_text_props(color='white', fontweight='bold')

            # Marcar orfes en vermell
            if file_check.get('uib_orphan', 0) > 0:
                tbl_check[(1, 3)].set_facecolor('#f8d7da')
            if file_check.get('dad_orphan', 0) > 0:
                tbl_check[(2, 3)].set_facecolor('#f8d7da')

            fig.text(0.5, 0.28, status_text, ha='center', fontsize=10,
                    fontweight='bold', color=status_color)

        # Processament aplicat
        fig.text(0.1, 0.20, "PROCESSAMENT APLICAT:", fontsize=9, fontweight='bold')
        fig.text(0.1, 0.17, "• Correcció baseline: Moda robusta (finestra inicial)", fontsize=8)
        fig.text(0.1, 0.14, "• Suavitzat: Savitzky-Golay (finestra=11, ordre=3)", fontsize=8)
        fig.text(0.1, 0.11, "• Alineació temporal: " +
                ("Pel màxim (BP)" if info.get('bp') else "KHP + A254 (COLUMN)"), fontsize=8)

        draw_footer(fig, "Serveis Tècnics de Recerca")
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # =================================================================
        # PÀGINA 2: Taula detallada de mostres
        # =================================================================
        rows_per_page = 40
        n_pages = (len(samples_data) + rows_per_page - 1) // rows_per_page

        for page_idx in range(n_pages):
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.patch.set_facecolor('white')

            draw_header(fig, "DETALL DE MOSTRES",
                       f"Punts per mostra", seq_name, page_idx + 2, n_pages + 1)

            ax = fig.add_axes([0.02, 0.05, 0.96, 0.87])
            ax.axis('off')

            # Subset de mostres per aquesta pàgina
            start_idx = page_idx * rows_per_page
            end_idx = min(start_idx + rows_per_page, len(samples_data))
            page_samples = samples_data[start_idx:end_idx]

            headers = ["#", "Mostra", "Rep", "Mode", "Pts Direct", "Pts UIB", "Pts DAD"]
            rows = []

            for i, s in enumerate(page_samples, start=start_idx + 1):
                mostra = s['mostra'][:18]
                if is_khp(s['mostra']):
                    mostra = f"● {mostra}"
                elif is_control(s['mostra']):
                    mostra = f"○ {mostra}"

                rows.append([
                    str(i),
                    mostra,
                    s['rep'],
                    s['doc_mode'],
                    str(s['n_direct']) if s['n_direct'] > 0 else "─",
                    str(s['n_uib']) if s['n_uib'] > 0 else "─",
                    str(s['n_dad']) if s['n_dad'] > 0 else "─",
                ])

            table_data = [headers] + rows
            tbl = ax.table(cellText=table_data, loc='upper center', cellLoc='center',
                          colWidths=[0.05, 0.30, 0.07, 0.12, 0.14, 0.14, 0.14])
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(7)
            tbl.scale(1.0, 1.2)

            # Estil capçalera
            for j in range(len(headers)):
                tbl[(0, j)].set_facecolor(COLORS["primary"])
                tbl[(0, j)].set_text_props(color='white', fontweight='bold')

            # Colorar files segons tipus
            for i, s in enumerate(page_samples, start=1):
                if is_khp(s['mostra']):
                    for j in range(len(headers)):
                        tbl[(i, j)].set_facecolor('#d4edda')  # Verd clar
                elif is_control(s['mostra']):
                    for j in range(len(headers)):
                        tbl[(i, j)].set_facecolor('#fff3cd')  # Groc clar

                # Marcar valors 0 o absents
                if s['n_direct'] == 0 and s['doc_mode'] in ('DUAL', 'DIRECT'):
                    tbl[(i, 4)].set_facecolor('#f8d7da')
                if s['n_dad'] == 0:
                    tbl[(i, 6)].set_facecolor('#f8d7da')

            # Llegenda
            fig.text(0.1, 0.02, "● KHP (estàndard)  ○ Control (MQ/NaOH)  Vermell = Dades absents",
                    fontsize=7, style='italic')

            draw_footer(fig)
            pdf.savefig(fig, dpi=150)
            plt.close(fig)

    return pdf_path


# =============================================================================
# REPORT CROMATOGRAMES
# =============================================================================
def generate_chromatograms_report(seq_path, xlsx_files, info, output_path=None):
    """
    Genera PDF amb cromatogrames agrupats per rèplica.

    Format: 4 parells (R1|R2) per pàgina, mostres → KHP → controls.

    Args:
        seq_path: Ruta a la carpeta SEQ
        xlsx_files: Llista de fitxers Excel consolidats
        info: Dict amb info (bp, seq)
        output_path: Path de sortida (opcional)

    Returns:
        Path del PDF generat
    """
    apply_style()

    # Preparar paths
    if output_path is None:
        check_path = os.path.join(seq_path, "CHECK")
        os.makedirs(check_path, exist_ok=True)
        output_path = check_path

    seq_name = info.get('seq', os.path.basename(seq_path))
    is_bp = info.get('bp', False)
    pdf_path = os.path.join(output_path, f"REPORT_Cromatogrames_{seq_name}.pdf")

    # Escala X segons mode
    if is_bp:
        x_min, x_max = -1, 5
    else:
        x_min, x_max = 0, 70

    # Llegir dades de tots els fitxers
    samples = []

    for f in sorted(xlsx_files):
        try:
            df_id = pd.read_excel(f, "ID", engine="openpyxl")
            id_dict = dict(zip(df_id["Camp"], df_id["Valor"]))

            mostra = str(id_dict.get("Mostra", ""))
            rep = str(id_dict.get("Rèplica", "1"))
            doc_mode = str(id_dict.get("DOC_MODE", ""))

            # Llegir DOC
            df_doc = pd.read_excel(f, "DOC", engine="openpyxl")
            t_doc = df_doc["time (min)"].values if "time (min)" in df_doc.columns else np.array([])

            # DOC Direct o DOC general
            y_direct = None
            y_uib = None

            if "DOC_Direct (mAU)" in df_doc.columns:
                y_direct = df_doc["DOC_Direct (mAU)"].values
            elif "DOC (mAU)" in df_doc.columns:
                y_direct = df_doc["DOC (mAU)"].values

            if "DOC_UIB (mAU)" in df_doc.columns:
                y_uib = df_doc["DOC_UIB (mAU)"].values

            # Llegir DAD
            t_dad = np.array([])
            y_dad_254 = np.array([])
            try:
                df_dad = pd.read_excel(f, "DAD", engine="openpyxl")
                if not df_dad.empty and "time (min)" in df_dad.columns:
                    t_dad = df_dad["time (min)"].values
                    for col in df_dad.columns:
                        if "254" in str(col):
                            y_dad_254 = df_dad[col].values
                            break
            except:
                pass

            if len(t_doc) > 10:
                samples.append({
                    'mostra': mostra,
                    'rep': rep,
                    't_doc': t_doc,
                    'y_direct': y_direct,
                    'y_uib': y_uib,
                    't_dad': t_dad,
                    'y_dad_254': y_dad_254,
                    'doc_mode': doc_mode,
                })
        except Exception:
            continue

    if not samples:
        return None

    # Ordenar i agrupar per rèplica
    samples = sort_samples_for_report(samples)
    groups = group_replicates(samples)

    # Ordenar grups: normals → KHP → controls
    group_order = []
    for base in groups:
        sample = groups[base][0]
        if is_khp(sample['mostra']):
            priority = 1
        elif is_control(sample['mostra']):
            priority = 2
        else:
            priority = 0
        group_order.append((priority, base))

    group_order.sort(key=lambda x: (x[0], x[1]))
    ordered_bases = [x[1] for x in group_order]

    # Generar PDF: 4 parells per pàgina
    pairs_per_page = 4
    n_pages = (len(ordered_bases) + pairs_per_page - 1) // pairs_per_page

    with PdfPages(pdf_path) as pdf:
        for page_idx in range(n_pages):
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.patch.set_facecolor('white')

            method_str = "BP" if is_bp else "COLUMN"
            draw_header(fig, f"CROMATOGRAMES {method_str}",
                       f"Seqüència {seq_name}", seq_name, page_idx + 1, n_pages)

            # 4 files x 2 columnes
            gs = GridSpec(4, 2, figure=fig,
                         left=0.08, right=0.92, top=0.86, bottom=0.06,
                         hspace=0.35, wspace=0.25)

            start_idx = page_idx * pairs_per_page
            end_idx = min(start_idx + pairs_per_page, len(ordered_bases))

            for row, base in enumerate(ordered_bases[start_idx:end_idx]):
                reps = groups[base]

                # Color del títol segons tipus
                if is_khp(base):
                    title_color = COLORS["success"]
                    title_suffix = " [KHP]"
                elif is_control(base):
                    title_color = COLORS["warning"]
                    title_suffix = " [CTRL]"
                else:
                    title_color = COLORS["dark"]
                    title_suffix = ""

                for col, rep_data in enumerate(reps[:2]):  # Màxim 2 rèpliques
                    ax = fig.add_subplot(gs[row, col])

                    t = rep_data['t_doc']

                    # Plot DOC Direct (blau)
                    if rep_data['y_direct'] is not None and len(rep_data['y_direct']) > 0:
                        ax.plot(t, rep_data['y_direct'], '-',
                               color=COLORS["doc_direct"], linewidth=0.7,
                               label='DOC Direct')

                    # Plot DOC UIB (verd) si disponible i diferent
                    if rep_data['y_uib'] is not None and len(rep_data['y_uib']) > 0:
                        ax.plot(t, rep_data['y_uib'], '-',
                               color=COLORS["doc_uib"], linewidth=0.5, alpha=0.7,
                               label='DOC UIB')

                    ax.set_xlim(x_min, x_max)
                    ax.set_xlabel('Temps (min)')
                    ax.set_ylabel('mAU', color=COLORS["doc_direct"])
                    ax.tick_params(axis='y', colors=COLORS["doc_direct"])

                    # Plot A254 (vermell) en eix secundari
                    if len(rep_data['t_dad']) > 10 and len(rep_data['y_dad_254']) > 10:
                        ax2 = ax.twinx()
                        ax2.plot(rep_data['t_dad'], rep_data['y_dad_254'], '-',
                                color=COLORS["dad_254"], linewidth=0.5, alpha=0.6)
                        ax2.set_ylabel('A254', color=COLORS["dad_254"], fontsize=7)
                        ax2.tick_params(axis='y', colors=COLORS["dad_254"], labelsize=6)

                    # Títol
                    rep_num = rep_data.get('rep', '?')
                    ax.set_title(f"{base} R{rep_num}{title_suffix}",
                                fontsize=8, fontweight='bold', color=title_color)

                    ax.grid(True, alpha=0.3, linewidth=0.3)

                # Si només hi ha 1 rèplica, deixar espai buit a la dreta
                if len(reps) < 2:
                    ax_empty = fig.add_subplot(gs[row, 1])
                    ax_empty.text(0.5, 0.5, f"(Sense R2)", ha='center', va='center',
                                 fontsize=9, color='lightgray', style='italic')
                    ax_empty.axis('off')

            # Llegenda al peu
            legend_elements = [
                mpatches.Patch(color=COLORS["doc_direct"], label='DOC Direct'),
                mpatches.Patch(color=COLORS["doc_uib"], label='DOC UIB'),
                mpatches.Patch(color=COLORS["dad_254"], label='A254 (DAD)'),
            ]
            fig.legend(handles=legend_elements, loc='lower center', ncol=3,
                      fontsize=7, frameon=False, bbox_to_anchor=(0.5, 0.01))

            pdf.savefig(fig, dpi=150)
            plt.close(fig)

    return pdf_path


# =============================================================================
# FUNCIÓ PRINCIPAL: GENERAR TOTS ELS REPORTS
# =============================================================================
def generate_all_reports(seq_path, xlsx_files, info, output_path=None):
    """
    Genera tots els informes de consolidació.

    Args:
        seq_path: Ruta a la carpeta SEQ
        xlsx_files: Llista de fitxers Excel consolidats
        info: Dict amb info de consolidació
        output_path: Path de sortida (opcional)

    Returns:
        Dict amb paths dels PDFs generats
    """
    results = {}

    try:
        results['consolidation'] = generate_consolidation_report(
            seq_path, xlsx_files, info, output_path)
    except Exception as e:
        results['consolidation_error'] = str(e)

    try:
        results['chromatograms'] = generate_chromatograms_report(
            seq_path, xlsx_files, info, output_path)
    except Exception as e:
        results['chromatograms_error'] = str(e)

    return results


# =============================================================================
# TEST STANDALONE
# =============================================================================
if __name__ == "__main__":
    import sys
    import glob

    if len(sys.argv) > 1:
        seq_path = sys.argv[1]
        print(f"Generant reports per: {seq_path}")

        # Buscar fitxers consolidats
        res_cons = os.path.join(seq_path, "Resultats_Consolidats")
        xlsx_files = glob.glob(os.path.join(res_cons, "*.xlsx"))
        xlsx_files = [f for f in xlsx_files if not os.path.basename(f).startswith("~$")]

        if xlsx_files:
            # Info bàsica
            info = {
                'seq': os.path.basename(seq_path).split('_')[0],
                'bp': 'BP' in os.path.basename(seq_path).upper(),
                'mode': 'DUAL',
                'date': datetime.now().strftime('%Y-%m-%d'),
            }

            results = generate_all_reports(seq_path, xlsx_files, info)

            print(f"\nResultats:")
            for k, v in results.items():
                print(f"  {k}: {v}")
        else:
            print(f"No s'han trobat fitxers consolidats a {res_cons}")
    else:
        print("Ús: python hpsec_reports.py <seq_folder>")
