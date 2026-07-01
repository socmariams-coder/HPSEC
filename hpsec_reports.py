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

# Import funcions d'identificació des de hpsec_import (Single Source of Truth)
from hpsec_import import is_khp


# =============================================================================
# CONFIGURACIÓ I ESTILS
# =============================================================================
LOGO_FILENAME = "logo STRs.png"  # Nom real del fitxer
INSTITUTION_LINE = "Serveis Tècnics de Recerca — Universitat de Girona · Developed by LEQUIA"

# Colors corporatius
COLORS = {
    "primary": "#2E86AB",      # Blau principal
    "primary_dark": "#1A5276",
    "secondary": "#A23B72",    # Magenta
    "accent": "#27AE60",       # Verd accent (reports)
    "success": "#28a745",      # Verd
    "warning": "#F39C12",      # Taronja (reports unified)
    "danger": "#E74C3C",       # Vermell (reports unified)
    "light": "#f8f9fa",        # Gris clar
    "dark": "#343a40",         # Gris fosc
    "text": "#2C3E50",
    "text_secondary": "#7F8C8D",
    "background": "#FFFFFF",
    "surface": "#F8F9FA",
    "border": "#E5E7EB",
    "table_header": "#2E86AB",
    "table_row_alt": "#F8FAFC",
    "doc_direct": "#1f77b4",   # Blau DOC Direct
    "doc_uib": "#2ca02c",      # Verd DOC UIB
    "dad_254": "#d62728",      # Vermell A254
    "khp_row": "#E8F8E8",
    "control_row": "#FFF9E6",
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

# Fonts per informes PDF
REPORT_FONTS = {
    "title": {"family": "sans-serif", "size": 18, "weight": "bold"},
    "subtitle": {"family": "sans-serif", "size": 12, "weight": "normal"},
    "section": {"family": "sans-serif", "size": 11, "weight": "bold"},
    "body": {"family": "sans-serif", "size": 9, "weight": "normal"},
    "small": {"family": "sans-serif", "size": 8, "weight": "normal"},
    "mono": {"family": "monospace", "size": 9, "weight": "normal"},
}

# Colors cromatogrames (analysis report)
C1 = '#1565C0'       # R1 Direct (blue)
C2 = '#E65100'       # R2 Direct (orange)
C_UIB = '#2E7D32'    # R1 UIB (dark green)
C_UIB2 = '#66BB6A'   # R2 UIB (light green)
LW = 0.7


def format_value(val, fmt=".2f", suffix="", default="-"):
    """Formata un valor numeric per a taules de reports."""
    if val is None:
        return default
    if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
        return default
    try:
        if val == 0 and fmt != ".0f":
            return default
        return f"{val:{fmt}}{suffix}"
    except Exception:
        return str(val)


def draw_minimal_table(ax, data, col_widths=None, header_color=None,
                       row_colors=None, font_size=8, max_row_height=None):
    """
    Dibuixa una taula minimalista sense divisions verticals.
    Nomes linies horitzontals a capcalera i final.
    """
    if header_color is None:
        header_color = COLORS["table_header"]

    n_rows = len(data)
    n_cols = len(data[0]) if data else 0

    if col_widths is None:
        col_widths = [1.0 / n_cols] * n_cols

    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    row_height = 0.9 / n_rows
    if max_row_height is not None:
        row_height = min(row_height, max_row_height)
    y_start = 0.95

    for i, row in enumerate(data):
        y = y_start - i * row_height
        x = 0.02
        is_header = (i == 0)

        if not is_header and row_colors and i in row_colors:
            rect = plt.Rectangle((0.02, y - row_height), 0.96, row_height,
                                 facecolor=row_colors[i], edgecolor='none',
                                 alpha=0.5)
            ax.add_patch(rect)

        for j, cell in enumerate(row):
            cell_width = col_widths[j] * 0.96

            if is_header:
                ax.text(x + cell_width / 2, y - row_height / 2, str(cell),
                        ha='center', va='center', fontsize=font_size,
                        fontweight='bold', color='white')
            else:
                ax.text(x + cell_width / 2, y - row_height / 2, str(cell),
                        ha='center', va='center', fontsize=font_size,
                        color=COLORS["text"])
            x += cell_width

        if is_header:
            rect = plt.Rectangle((0.02, y - row_height), 0.96, row_height,
                                 facecolor=header_color, edgecolor='none')
            ax.add_patch(rect)
            ax.axhline(y - row_height, xmin=0.02, xmax=0.98,
                        color=header_color, linewidth=1.5)

    y_final = y_start - n_rows * row_height
    ax.axhline(y_final, xmin=0.02, xmax=0.98,
               color=COLORS["border"], linewidth=1)


def draw_report_footer(fig, page_num):
    """Peu de pagina estandard per informes PDF."""
    fig.text(0.5, 0.02, INSTITUTION_LINE,
             ha='center', fontsize=8, color=COLORS["text_secondary"])
    fig.text(0.95, 0.02, str(page_num), ha='right', fontsize=8,
             color=COLORS["text_secondary"])


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
    # Treure sufixos de rèplica (requereix separador _ - o espai abans de R)
    # Això evita que FR2586 es confongui amb F + R2586
    match = re.match(r"^(.+?)(?:[_\-\s]R\d{1,2})?$", name, re.IGNORECASE)
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
    from hpsec_version import SUITE_FULL

    logo_path = get_logo_path()

    # Logo a l'esquerra
    if logo_path:
        try:
            logo = plt.imread(logo_path)
            ax_logo = fig.add_axes([0.02, 0.91, 0.18, 0.08])
            ax_logo.imshow(logo)
            ax_logo.axis('off')
        except Exception:
            pass

    # Capçalera: nom + versió
    fig.text(0.5, 0.97, SUITE_FULL, ha='center', va='top',
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
    from hpsec_version import SUITE_FULL

    # Línia separadora
    fig.add_artist(plt.Line2D([0.05, 0.95], [0.02, 0.02],
                              color='lightgray', linewidth=0.5,
                              transform=fig.transFigure))

    # Data i versió (sempre)
    date_str = datetime.now().strftime("%d/%m/%Y %H:%M")
    footer_left = f"Generat: {date_str}  |  {SUITE_FULL}"
    fig.text(0.05, 0.01, footer_left, ha='left', va='bottom',
             fontsize=6, color='gray')

    if text:
        fig.text(0.95, 0.01, text, ha='right', va='bottom',
                 fontsize=6, color='gray')


# =============================================================================
# REPORT CONSOLIDACIÓ
# =============================================================================
def generate_consolidation_report(seq_path, xlsx_files, info, output_path=None):
    """
    Genera PDF de consolidació amb taula completa de fitxers, punts, timeouts i SNR.

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

    # Intentar llegir consolidation.json per info global
    json_path = os.path.join(output_path, "consolidation.json")
    summary = {}
    if os.path.exists(json_path):
        try:
            import json
            with open(json_path, 'r', encoding='utf-8') as f:
                summary = json.load(f)
        except:
            pass

    # Extreure metadades de consolidació
    meta = summary.get('meta', {})
    script_version = meta.get('script_version', '')
    consolidation_date = meta.get('generated_at', '')

    # Llegir dades de tots els fitxers Excel
    samples_data = []
    total_pts_direct = 0
    total_pts_uib = 0
    total_pts_dad = 0

    for f in sorted(xlsx_files):
        try:
            df_id = pd.read_excel(f, "ID", engine="openpyxl")
            # Suportar ambdós formats (català antic / anglès nou)
            if "Field" in df_id.columns:
                id_dict = dict(zip(df_id["Field"], df_id["Value"]))
            else:
                id_dict = dict(zip(df_id["Camp"], df_id["Valor"]))

            mostra = str(id_dict.get("Sample", id_dict.get("Mostra", "")))
            rep = str(id_dict.get("Replica", id_dict.get("Rèplica", "-")))
            doc_mode = str(id_dict.get("DOC_Mode", id_dict.get("DOC_MODE", "")))

            # Fitxers origen
            file_uib = str(id_dict.get("File_DOC_UIB", ""))
            file_dad = str(id_dict.get("File_DAD", ""))

            # Punts
            n_doc = 0
            n_dad = 0
            try:
                n_doc = int(float(id_dict.get("DOC_N_Points", 0) or 0))
                n_dad = int(float(id_dict.get("DAD_N_Points", 0) or 0))
            except:
                pass

            # Timeout info
            timeout_detected = str(id_dict.get("TOC_Timeout_Detected", "NO")).upper() == "YES"
            timeout_severity = str(id_dict.get("TOC_Timeout_Severity", "OK"))
            timeout_detail = str(id_dict.get("TOC_Timeout_1", ""))

            # SNR
            snr_direct = None
            try:
                snr_direct = float(id_dict.get("SNR_Direct", 0) or 0)
            except:
                pass

            # Comptar punts per DOC Direct/UIB des del sheet DOC
            n_direct = n_doc
            n_uib = 0
            try:
                df_doc = pd.read_excel(f, "DOC", engine="openpyxl")
                if "DOC_Direct (mAU)" in df_doc.columns:
                    n_direct = int(df_doc["DOC_Direct (mAU)"].notna().sum())
                if "DOC_UIB (mAU)" in df_doc.columns:
                    n_uib = int(df_doc["DOC_UIB (mAU)"].notna().sum())
                elif "DOC (mAU)" in df_doc.columns and doc_mode == "UIB":
                    n_uib = int(df_doc["DOC (mAU)"].notna().sum())
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
                'file_uib': file_uib,
                'file_dad': file_dad,
                'n_direct': n_direct,
                'n_uib': n_uib,
                'n_dad': n_dad,
                'timeout': timeout_severity if timeout_detected else "OK",
                'timeout_detail': timeout_detail,
                'snr': snr_direct,
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
        ax_info = fig.add_axes([0.05, 0.70, 0.9, 0.16])
        ax_info.axis('off')

        mode_str = info.get('mode', 'N/A')
        method_str = "By-Pass (BP)" if info.get('bp', False) else "COLUMN"
        date_str = str(info.get('date', 'N/A'))

        khp_count = sum(1 for s in samples_data if is_khp(s['mostra']))
        control_count = sum(1 for s in samples_data if is_control(s['mostra']))
        sample_count = len(samples_data) - khp_count - control_count

        # Formatar data consolidació (de ISO a DD/MM/YYYY HH:MM)
        cons_date_str = "─"
        if consolidation_date:
            try:
                from datetime import datetime as dt
                cons_dt = dt.fromisoformat(consolidation_date)
                cons_date_str = cons_dt.strftime("%d/%m/%Y %H:%M")
            except:
                cons_date_str = consolidation_date[:16] if len(consolidation_date) > 16 else consolidation_date

        info_data = [
            ["PARÀMETRE", "VALOR", "PARÀMETRE", "VALOR"],
            ["Mode DOC", mode_str, "Mostres", str(sample_count)],
            ["Mètode", method_str, "KHP (estàndard)", str(khp_count)],
            ["Data SEQ", date_str[:10] if len(date_str) > 10 else date_str,
             "Controls (MQ/NaOH)", str(control_count)],
            ["Total injeccions", str(len(xlsx_files)), "Consolidat", cons_date_str],
        ]

        tbl_info = ax_info.table(cellText=info_data, loc='center', cellLoc='center',
                                  colWidths=[0.22, 0.28, 0.22, 0.28])
        tbl_info.auto_set_font_size(False)
        tbl_info.set_fontsize(9)
        tbl_info.scale(1.0, 1.8)

        for j in range(4):
            tbl_info[(0, j)].set_facecolor(COLORS["primary"])
            tbl_info[(0, j)].set_text_props(color='white', fontweight='bold')

        # Recompte de punts
        ax_pts = fig.add_axes([0.05, 0.54, 0.42, 0.12])
        ax_pts.axis('off')

        pts_data = [
            ["PUNTS", "TOTAL", "MITJANA"],
            ["DOC Direct", f"{total_pts_direct:,}",
             f"{total_pts_direct/max(len(samples_data),1):,.0f}" if total_pts_direct > 0 else "─"],
            ["DOC UIB", f"{total_pts_uib:,}",
             f"{total_pts_uib/max(len(samples_data),1):,.0f}" if total_pts_uib > 0 else "─"],
            ["DAD", f"{total_pts_dad:,}",
             f"{total_pts_dad/max(len(samples_data),1):,.0f}" if total_pts_dad > 0 else "─"],
        ]

        tbl_pts = ax_pts.table(cellText=pts_data, loc='center', cellLoc='center',
                               colWidths=[0.4, 0.3, 0.3])
        tbl_pts.auto_set_font_size(False)
        tbl_pts.set_fontsize(8)
        tbl_pts.scale(1.0, 1.6)

        for j in range(3):
            tbl_pts[(0, j)].set_facecolor(COLORS["dark"])
            tbl_pts[(0, j)].set_text_props(color='white', fontweight='bold')

        # Timeouts per severitat
        ax_to = fig.add_axes([0.53, 0.54, 0.42, 0.12])
        ax_to.axis('off')

        n_ok = sum(1 for s in samples_data if s['timeout'] == "OK")
        n_info = sum(1 for s in samples_data if s['timeout'] == "INFO")
        n_warn = sum(1 for s in samples_data if s['timeout'] == "WARNING")
        n_crit = sum(1 for s in samples_data if s['timeout'] == "CRITICAL")

        to_data = [
            ["TIMEOUTS TOC", "N", "%"],
            ["OK", str(n_ok), f"{100*n_ok/max(len(samples_data),1):.0f}%"],
            ["INFO", str(n_info), f"{100*n_info/max(len(samples_data),1):.0f}%"],
            ["WARNING", str(n_warn), f"{100*n_warn/max(len(samples_data),1):.0f}%"],
            ["CRITICAL", str(n_crit), f"{100*n_crit/max(len(samples_data),1):.0f}%"],
        ]

        tbl_to = ax_to.table(cellText=to_data, loc='center', cellLoc='center',
                             colWidths=[0.4, 0.3, 0.3])
        tbl_to.auto_set_font_size(False)
        tbl_to.set_fontsize(8)
        tbl_to.scale(1.0, 1.4)  # Ajustat per 5 files

        for j in range(3):
            tbl_to[(0, j)].set_facecolor(COLORS["dark"])
            tbl_to[(0, j)].set_text_props(color='white', fontweight='bold')
        # Colorar segons severitat
        tbl_to[(1, 0)].set_facecolor('#c6efce')  # OK - verd
        if n_info > 0:
            tbl_to[(2, 0)].set_facecolor('#cce5ff')  # INFO - blau clar
        if n_warn > 0:
            tbl_to[(3, 0)].set_facecolor('#fff3cd')  # WARNING - groc
        if n_crit > 0:
            tbl_to[(4, 0)].set_facecolor('#f8d7da')  # CRITICAL - vermell

        # Qualitat SNR/LOD (si tenim summary)
        quality = summary.get('quality', {})
        snr_info = quality.get('snr_direct', {})
        lod_direct = quality.get('lod_direct_mau')

        if snr_info or lod_direct:
            ax_qual = fig.add_axes([0.05, 0.42, 0.9, 0.08])
            ax_qual.axis('off')

            qual_data = [
                ["QUALITAT", "SNR min", "SNR mediana", "SNR max", "LOD (mAU)"],
                ["DOC Direct",
                 f"{snr_info.get('min', '─')}" if snr_info else "─",
                 f"{snr_info.get('median', '─')}" if snr_info else "─",
                 f"{snr_info.get('max', '─')}" if snr_info else "─",
                 f"{lod_direct:.2f}" if lod_direct else "─"],
            ]

            tbl_qual = ax_qual.table(cellText=qual_data, loc='center', cellLoc='center',
                                     colWidths=[0.2, 0.2, 0.2, 0.2, 0.2])
            tbl_qual.auto_set_font_size(False)
            tbl_qual.set_fontsize(8)
            tbl_qual.scale(1.0, 1.5)

            for j in range(5):
                tbl_qual[(0, j)].set_facecolor(COLORS["primary"])
                tbl_qual[(0, j)].set_text_props(color='white', fontweight='bold')

        # Verificació de fitxers
        file_check = info.get('file_check', {})
        if file_check:
            ax_check = fig.add_axes([0.1, 0.28, 0.8, 0.10])
            ax_check.axis('off')

            has_issues = file_check.get('has_issues', False)
            status_color = COLORS["danger"] if has_issues else COLORS["success"]
            status_text = "DISCREPÀNCIES DETECTADES" if has_issues else "VERIFICACIÓ CORRECTA"

            check_data = [
                ["FITXERS", "TROBATS", "USATS", "ORFES"],
                ["UIB", str(file_check.get('uib_found', 0)),
                 str(file_check.get('uib_used', 0)), str(file_check.get('uib_orphan', 0))],
                ["DAD", str(file_check.get('dad_found', 0)),
                 str(file_check.get('dad_used', 0)), str(file_check.get('dad_orphan', 0))],
            ]

            tbl_check = ax_check.table(cellText=check_data, loc='center', cellLoc='center',
                                       colWidths=[0.25, 0.25, 0.25, 0.25])
            tbl_check.auto_set_font_size(False)
            tbl_check.set_fontsize(8)
            tbl_check.scale(1.0, 1.5)

            for j in range(4):
                tbl_check[(0, j)].set_facecolor(COLORS["primary"])
                tbl_check[(0, j)].set_text_props(color='white', fontweight='bold')

            if file_check.get('uib_orphan', 0) > 0:
                tbl_check[(1, 3)].set_facecolor('#f8d7da')
            if file_check.get('dad_orphan', 0) > 0:
                tbl_check[(2, 3)].set_facecolor('#f8d7da')

            fig.text(0.5, 0.25, status_text, ha='center', fontsize=9,
                    fontweight='bold', color=status_color)

        # Alineació (si tenim summary)
        alignment = summary.get('alignment', {})
        if alignment:
            shift_uib = alignment.get('shift_uib', 0)
            shift_direct = alignment.get('shift_direct', 0)
            source = alignment.get('source', 'N/A')
            if shift_uib or shift_direct:
                fig.text(0.1, 0.20, f"ALINEACIÓ: UIB={shift_uib*60:.1f}s, Direct={shift_direct*60:.1f}s ({source})",
                        fontsize=8, fontweight='bold')

        # Processament aplicat
        fig.text(0.1, 0.15, "PROCESSAMENT:", fontsize=8, fontweight='bold')
        fig.text(0.1, 0.12, "• Baseline: Finestres temporals (evita timeouts) | Suavitzat: Savitzky-Golay (11,3) | "
                f"Alineació: {'Pel màxim (BP)' if info.get('bp') else 'KHP + A254'}", fontsize=7)

        draw_footer(fig, INSTITUTION_LINE)
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # =================================================================
        # PÀGINES 2+: Taula detallada de mostres
        # =================================================================
        rows_per_page = 35
        n_pages = (len(samples_data) + rows_per_page - 1) // rows_per_page

        for page_idx in range(n_pages):
            fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
            fig.patch.set_facecolor('white')

            draw_header(fig, "DETALL D'INJECCIONS",
                       f"Fitxers i punts per injecció", seq_name, page_idx + 2, n_pages + 1)

            ax = fig.add_axes([0.02, 0.08, 0.96, 0.82])
            ax.axis('off')

            start_idx = page_idx * rows_per_page
            end_idx = min(start_idx + rows_per_page, len(samples_data))
            page_samples = samples_data[start_idx:end_idx]

            headers = ["#", "Mostra", "R", "Fitxer UIB", "Fitxer DAD", "Pts DOC", "Pts DAD", "Timeout", "SNR"]
            rows = []

            for i, s in enumerate(page_samples, start=start_idx + 1):
                mostra = s['mostra']
                if is_khp(s['mostra']):
                    mostra = f"● {mostra}"
                elif is_control(s['mostra']):
                    mostra = f"○ {mostra}"

                # Timeout amb zona si hi ha detall
                timeout_str = s['timeout']
                if s['timeout_detail'] and s['timeout'] != "OK":
                    # Extreure zona del detall (ex: "11.5 min (74s) - BioP [WARNING]")
                    import re
                    match = re.search(r'- (\w+)', s['timeout_detail'])
                    if match:
                        timeout_str = f"{s['timeout']} ({match.group(1)})"

                snr_str = f"{s['snr']:.0f}" if s['snr'] and s['snr'] > 0 else "─"

                rows.append([
                    str(i),
                    mostra,
                    s['rep'],
                    s['file_uib'] if s['file_uib'] else "─",
                    s['file_dad'] if s['file_dad'] else "─",
                    str(s['n_direct']) if s['n_direct'] > 0 else (str(s['n_uib']) if s['n_uib'] > 0 else "─"),
                    str(s['n_dad']) if s['n_dad'] > 0 else "─",
                    timeout_str,
                    snr_str,
                ])

            table_data = [headers] + rows
            col_widths = [0.04, 0.18, 0.03, 0.20, 0.15, 0.10, 0.10, 0.12, 0.08]
            tbl = ax.table(cellText=table_data, loc='upper center', cellLoc='center',
                          colWidths=col_widths)
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(7)
            tbl.scale(1.0, 1.3)

            for j in range(len(headers)):
                tbl[(0, j)].set_facecolor(COLORS["primary"])
                tbl[(0, j)].set_text_props(color='white', fontweight='bold')

            for i, s in enumerate(page_samples, start=1):
                if is_khp(s['mostra']):
                    for j in range(len(headers)):
                        tbl[(i, j)].set_facecolor('#d4edda')
                elif is_control(s['mostra']):
                    for j in range(len(headers)):
                        tbl[(i, j)].set_facecolor('#fff3cd')

                # Marcar timeouts
                if s['timeout'] == "WARNING":
                    tbl[(i, 7)].set_facecolor('#fff3cd')
                elif s['timeout'] == "CRITICAL":
                    tbl[(i, 7)].set_facecolor('#f8d7da')

                # Marcar dades absents
                if s['n_direct'] == 0 and s['n_uib'] == 0:
                    tbl[(i, 5)].set_facecolor('#f8d7da')
                if s['n_dad'] == 0:
                    tbl[(i, 6)].set_facecolor('#f8d7da')

            fig.text(0.02, 0.03, "● KHP  ○ Control  Groc=Warning  Vermell=Critical/Absent",
                    fontsize=6, style='italic')

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

    # Llegir consolidation.json per obtenir versió
    json_path = os.path.join(output_path, "consolidation.json")
    script_version = ""
    if os.path.exists(json_path):
        try:
            import json
            with open(json_path, 'r', encoding='utf-8') as f:
                summary = json.load(f)
            script_version = summary.get('meta', {}).get('script_version', '')
        except:
            pass

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
            # Suportar ambdós formats (català antic / anglès nou)
            if "Field" in df_id.columns:
                id_dict = dict(zip(df_id["Field"], df_id["Value"]))
            else:
                id_dict = dict(zip(df_id["Camp"], df_id["Valor"]))

            mostra = str(id_dict.get("Sample", id_dict.get("Mostra", "")))
            rep = str(id_dict.get("Replica", id_dict.get("Rèplica", "1")))
            doc_mode = str(id_dict.get("DOC_Mode", id_dict.get("DOC_MODE", "")))

            # Timeout i warnings
            timeout_detected = str(id_dict.get("TOC_Timeout_Detected", "NO")).upper() == "YES"
            timeout_severity = str(id_dict.get("TOC_Timeout_Severity", "OK"))
            timeout_detail = str(id_dict.get("TOC_Timeout_1", ""))

            # Extreure posicions dels timeouts per marcar a la gràfica
            timeout_positions = []
            if timeout_detected:
                for i in range(1, 10):  # Màxim 9 timeouts
                    to_str = str(id_dict.get(f"TOC_Timeout_{i}", ""))
                    if to_str:
                        # Format: "11.5 min (74s) - BioP [WARNING]"
                        match_pos = re.match(r'([\d.]+)\s*min\s*\((\d+)s\)', to_str)
                        if match_pos:
                            t_min = float(match_pos.group(1))
                            dur_s = float(match_pos.group(2))
                            timeout_positions.append({
                                't_start': t_min,
                                't_end': t_min + dur_s / 60.0  # Convertir segons a minuts
                            })

            # Construir llista de warnings (sense text, només per color títol)
            warnings = []
            if timeout_detected and timeout_severity in ("WARNING", "CRITICAL"):
                warnings.append("TIMEOUT")

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
                    'warnings': warnings,
                    'timeout_severity': timeout_severity,
                    'timeout_positions': timeout_positions,
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

                    # Marcar zones de timeout amb patró 'x' subtil (ABANS de les línies)
                    timeout_positions = rep_data.get('timeout_positions', [])
                    for to in timeout_positions:
                        ax.axvspan(to['t_start'], to['t_end'],
                                  alpha=0.15, color='gray', hatch='xx', linewidth=0)

                    # Determinar si hi ha dades Direct i/o UIB
                    has_direct = rep_data['y_direct'] is not None and len(rep_data['y_direct']) > 0
                    has_uib = rep_data['y_uib'] is not None and len(rep_data['y_uib']) > 0

                    if has_direct and has_uib:
                        # DUAL: Ambdós disponibles - Direct principal, UIB secundari
                        ax.plot(t, rep_data['y_direct'], '-',
                               color=COLORS["doc_direct"], linewidth=0.8,
                               label='Direct')
                        ax.plot(t, rep_data['y_uib'], '--',
                               color=COLORS["doc_uib"], linewidth=0.6,
                               label='UIB')
                    elif has_direct:
                        # Només Direct
                        ax.plot(t, rep_data['y_direct'], '-',
                               color=COLORS["doc_direct"], linewidth=0.8,
                               label='Direct')
                    elif has_uib:
                        # Només UIB - usar com a principal (blau, sòlida)
                        ax.plot(t, rep_data['y_uib'], '-',
                               color=COLORS["doc_direct"], linewidth=0.8,
                               label='DOC (UIB)')

                    ax.set_xlim(x_min, x_max)
                    ax.set_xlabel('Temps (min)', fontsize=7)
                    ax.set_ylabel('mAU', color=COLORS["doc_direct"], fontsize=7)
                    ax.tick_params(axis='y', colors=COLORS["doc_direct"])

                    # Plot A254 (vermell) en eix secundari
                    ax2 = None
                    if len(rep_data['t_dad']) > 10 and len(rep_data['y_dad_254']) > 10:
                        ax2 = ax.twinx()
                        ax2.plot(rep_data['t_dad'], rep_data['y_dad_254'], '-',
                                color=COLORS["dad_254"], linewidth=0.5, alpha=0.6,
                                label='A254')
                        ax2.set_ylabel('A254', color=COLORS["dad_254"], fontsize=6)
                        ax2.tick_params(axis='y', colors=COLORS["dad_254"], labelsize=5)

                    # Títol amb nom complet (color segons severitat timeout)
                    rep_num = rep_data.get('rep', '?')
                    title_text = f"{base} R{rep_num}{title_suffix}"
                    ax.set_title(title_text, fontsize=7, fontweight='bold', color=title_color)

                    # Llegenda combinada (ax + ax2) dins la gràfica
                    handles, labels = ax.get_legend_handles_labels()
                    if ax2 is not None:
                        handles2, labels2 = ax2.get_legend_handles_labels()
                        handles += handles2
                        labels += labels2
                    ax.legend(handles, labels, loc='upper right', fontsize=5,
                             framealpha=0.7, handlelength=1.5, handletextpad=0.3)

                    ax.grid(True, alpha=0.3, linewidth=0.3)

                # Si només hi ha 1 rèplica, deixar espai buit a la dreta
                if len(reps) < 2:
                    ax_empty = fig.add_subplot(gs[row, 1])
                    ax_empty.text(0.5, 0.5, f"(Sense R2)", ha='center', va='center',
                                 fontsize=9, color='lightgray', style='italic')
                    ax_empty.axis('off')

            draw_footer(fig)
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
# FUNCIONS DE GENERACIÓ DE GRÀFICS PER FASE
# =============================================================================

def generate_import_plots(seq_path, import_result):
    """
    Genera gràfics de la fase d'importació (cromatogrames raw en grid).

    Args:
        seq_path: Ruta a la carpeta SEQ
        import_result: Dict amb el resultat de la importació

    Returns:
        Path del PDF generat o None si hi ha error
    """
    if not import_result or not import_result.get('success'):
        return None

    apply_style()

    # Preparar paths
    check_path = os.path.join(seq_path, "CHECK", "data")
    os.makedirs(check_path, exist_ok=True)

    seq_name = os.path.basename(seq_path)
    pdf_path = os.path.join(check_path, f"PLOTS_Import_{seq_name}.pdf")

    samples = import_result.get("samples", {})
    if not samples:
        return None

    # Ordenar mostres (excloure controls)
    sample_names = [n for n in sorted(samples.keys()) if not is_control(n)]

    with PdfPages(pdf_path) as pdf:
        # Grid: 3 columnes x 4 files per pàgina = 12 gràfics
        samples_per_page = 12
        n_cols, n_rows = 3, 4

        for page_start in range(0, len(sample_names), samples_per_page):
            page_samples = sample_names[page_start:page_start + samples_per_page]

            fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
            fig.patch.set_facecolor('white')

            page_num = (page_start // samples_per_page) + 1
            total_pages = (len(sample_names) + samples_per_page - 1) // samples_per_page

            draw_header(fig, "CROMATOGRAMES RAW - IMPORTACIÓ",
                       f"Seqüència: {seq_name}", seq_name, page_num, total_pages)

            gs = GridSpec(n_rows, n_cols, figure=fig,
                         left=0.06, right=0.98, top=0.88, bottom=0.06,
                         hspace=0.35, wspace=0.25)

            for idx, sample_name in enumerate(page_samples):
                row = idx // n_cols
                col = idx % n_cols
                ax = fig.add_subplot(gs[row, col])

                sample_data = samples.get(sample_name, {})
                replicas = sample_data.get("replicas", {})

                # Plotar primera rèplica (o totes si n'hi ha poques)
                _plot_sample_chromatogram(ax, sample_name, replicas)

            pdf.savefig(fig, dpi=150)
            plt.close(fig)

    return pdf_path


def _plot_sample_chromatogram(ax, sample_name, replicas):
    """Plotar cromatograma d'una mostra amb DOC i DAD."""
    if not replicas:
        ax.text(0.5, 0.5, "Sense dades", ha='center', va='center',
               fontsize=8, color='gray')
        ax.set_title(sample_name, fontsize=8, fontweight='bold')
        ax.axis('off')
        return

    # Usar primera rèplica
    rep_key = sorted(replicas.keys())[0]
    rep_data = replicas[rep_key]

    has_data = False

    # DOC Direct
    direct = rep_data.get("direct", {})
    if direct and direct.get("t") is not None:
        t = np.asarray(direct["t"])
        y = direct.get("y") if direct.get("y") is not None else direct.get("y_raw")
        if y is not None:
            y = np.asarray(y)
            if len(t) > 0 and len(y) > 0:
                ax.plot(t, y, color=COLORS["doc_direct"], linewidth=0.7, label="DOC")
                has_data = True

    # DOC UIB (si disponible)
    uib = rep_data.get("uib", {})
    if uib and uib.get("t") is not None:
        t = np.asarray(uib["t"])
        y = uib.get("y") if uib.get("y") is not None else uib.get("y_raw")
        if y is not None:
            y = np.asarray(y)
            if len(t) > 0 and len(y) > 0:
                ax.plot(t, y, color=COLORS["doc_uib"], linewidth=0.7,
                       alpha=0.8, label="UIB")
                has_data = True

    # DAD 254nm (eix secundari)
    dad = rep_data.get("dad", {})
    if dad:
        t_dad = None
        y254 = None

        # Format DataFrame
        df_dad = dad.get("df")
        if df_dad is not None:
            try:
                if "time (min)" in df_dad.columns:
                    t_dad = df_dad["time (min)"].values
                for col in df_dad.columns:
                    if "254" in str(col):
                        y254 = df_dad[col].values
                        break
            except:
                pass

        # Format arrays
        if t_dad is None and dad.get("t") is not None:
            t_dad = np.asarray(dad["t"])
            wavelengths = dad.get("wavelengths", {})
            y254 = wavelengths.get(254) or wavelengths.get("254")
            if y254 is not None:
                y254 = np.asarray(y254)

        if t_dad is not None and y254 is not None and len(t_dad) > 0:
            ax2 = ax.twinx()
            ax2.plot(t_dad, y254, color=COLORS["dad_254"], linewidth=0.5,
                    linestyle="--", alpha=0.6, label="254nm")
            ax2.tick_params(axis='y', labelsize=6, colors=COLORS["dad_254"])
            ax2.set_ylabel("254nm", fontsize=6, color=COLORS["dad_254"])
            has_data = True

    if not has_data:
        ax.text(0.5, 0.5, "Sense dades", ha='center', va='center',
               fontsize=8, color='gray')
        ax.axis('off')
    else:
        ax.set_xlabel("min", fontsize=6)
        ax.set_ylabel("DOC (mAU)", fontsize=6)
        ax.tick_params(axis='both', labelsize=6)
        ax.grid(True, alpha=0.3, linewidth=0.3)

    # Truncar nom si és massa llarg
    display_name = sample_name[:20] + "..." if len(sample_name) > 20 else sample_name
    ax.set_title(display_name, fontsize=7, fontweight='bold')


def generate_calibration_plots(seq_path, calibration_result, imported_data=None):
    """
    Genera gràfics de la fase de calibració (KHP).

    Args:
        seq_path: Ruta a la carpeta SEQ
        calibration_result: Dict amb el resultat de la calibració
        imported_data: Dict amb les dades importades (opcional)

    Returns:
        Path del PDF generat o None si hi ha error
    """
    if not calibration_result or not calibration_result.get('success'):
        return None

    apply_style()

    # Preparar paths
    check_path = os.path.join(seq_path, "CHECK", "data")
    os.makedirs(check_path, exist_ok=True)

    seq_name = os.path.basename(seq_path)
    pdf_path = os.path.join(check_path, f"PLOTS_Calibration_{seq_name}.pdf")

    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
        fig.patch.set_facecolor('white')

        draw_header(fig, "CALIBRACIÓ KHP",
                   f"Seqüència: {seq_name}", seq_name, 1, 1)

        # Layout: 2x2 grid
        gs = GridSpec(2, 2, figure=fig,
                     left=0.08, right=0.95, top=0.85, bottom=0.10,
                     hspace=0.30, wspace=0.25)

        # Obtenir dades KHP
        khp_data = calibration_result.get("khp_data", {})
        replicas_direct = khp_data.get("replicas_direct", [])
        replicas_uib = khp_data.get("replicas_uib", [])
        factor = calibration_result.get("factor", 1.0)
        khp_source = calibration_result.get("khp_source", "")
        khp_conc = calibration_result.get("khp_conc_ppm", 0)

        # Plot 1: Rèpliques Direct
        ax1 = fig.add_subplot(gs[0, 0])
        _plot_khp_replicas(ax1, replicas_direct, "KHP Direct", COLORS["doc_direct"])

        # Plot 2: Rèpliques UIB (si existeixen)
        ax2 = fig.add_subplot(gs[0, 1])
        if replicas_uib:
            _plot_khp_replicas(ax2, replicas_uib, "KHP UIB", COLORS["doc_uib"])
        else:
            ax2.text(0.5, 0.5, "Sense dades UIB", ha='center', va='center',
                    fontsize=10, color='gray')
            ax2.axis('off')
            ax2.set_title("KHP UIB", fontsize=9, fontweight='bold')

        # Plot 3: Àrees i factor
        ax3 = fig.add_subplot(gs[1, 0])
        _plot_calibration_summary(ax3, calibration_result)

        # Plot 4: Info textual
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')

        info_text = f"""
RESUM CALIBRACIÓ
================
Font KHP: {khp_source}
Concentració: {khp_conc:.1f} ppm

Factor calibració: {factor:.4f}

Rèpliques Direct: {len(replicas_direct)}
Rèpliques UIB: {len(replicas_uib)}
"""
        ax4.text(0.1, 0.9, info_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace')

        pdf.savefig(fig, dpi=150)
        plt.close(fig)

    return pdf_path


def _plot_khp_replicas(ax, replicas, title, color):
    """Plotar rèpliques KHP superposades."""
    if not replicas:
        ax.text(0.5, 0.5, "Sense dades", ha='center', va='center',
               fontsize=10, color='gray')
        ax.axis('off')
        ax.set_title(title, fontsize=9, fontweight='bold')
        return

    for i, rep in enumerate(replicas):
        t = rep.get('t_doc')
        y = rep.get('y_doc')
        area = rep.get('area', 0)

        if t is not None and y is not None:
            t = np.asarray(t)
            y = np.asarray(y)
            alpha = 0.7 if i > 0 else 1.0
            ax.plot(t, y, color=color, linewidth=0.8, alpha=alpha,
                   label=f"R{i+1} (A={area:.0f})")

    ax.set_xlabel("Temps (min)", fontsize=8)
    ax.set_ylabel("DOC (mAU)", fontsize=8)
    ax.tick_params(axis='both', labelsize=7)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc='upper right')
    ax.set_title(title, fontsize=9, fontweight='bold')


def _plot_calibration_summary(ax, calibration_result):
    """Plotar resum de calibració (àrees per rèplica)."""
    khp_data = calibration_result.get("khp_data", {})
    replicas_direct = khp_data.get("replicas_direct", [])
    replicas_uib = khp_data.get("replicas_uib", [])

    areas_direct = [r.get('area', 0) for r in replicas_direct]
    areas_uib = [r.get('area', 0) for r in replicas_uib]

    x = np.arange(max(len(areas_direct), len(areas_uib)))
    width = 0.35

    if areas_direct:
        ax.bar(x[:len(areas_direct)] - width/2, areas_direct, width,
              label='Direct', color=COLORS["doc_direct"], alpha=0.8)
    if areas_uib:
        ax.bar(x[:len(areas_uib)] + width/2, areas_uib, width,
              label='UIB', color=COLORS["doc_uib"], alpha=0.8)

    ax.set_xlabel("Rèplica", fontsize=8)
    ax.set_ylabel("Àrea", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"R{i+1}" for i in x], fontsize=7)
    ax.tick_params(axis='y', labelsize=7)
    ax.legend(fontsize=7)
    ax.set_title("Àrees KHP per rèplica", fontsize=9, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')


def generate_analysis_plots(seq_path, analysis_result):
    """
    Genera gràfics de la fase d'anàlisi (cromatogrames processats en grid).

    Args:
        seq_path: Ruta a la carpeta SEQ
        analysis_result: Dict amb el resultat de l'anàlisi

    Returns:
        Path del PDF generat o None si hi ha error
    """
    if not analysis_result or not analysis_result.get('success'):
        return None

    apply_style()

    # Preparar paths
    check_path = os.path.join(seq_path, "CHECK", "data")
    os.makedirs(check_path, exist_ok=True)

    seq_name = os.path.basename(seq_path)
    pdf_path = os.path.join(check_path, f"PLOTS_Analysis_{seq_name}.pdf")

    samples_grouped = analysis_result.get("samples_grouped", {})
    if not samples_grouped:
        return None

    # Ordenar mostres (excloure controls i KHP)
    sample_names = [n for n in sorted(samples_grouped.keys())
                   if not is_control(n) and not is_khp(n)]

    with PdfPages(pdf_path) as pdf:
        # Grid: 3 columnes x 3 files per pàgina = 9 mostres
        # Cada mostra mostra totes les rèpliques
        samples_per_page = 9
        n_cols, n_rows = 3, 3

        for page_start in range(0, len(sample_names), samples_per_page):
            page_samples = sample_names[page_start:page_start + samples_per_page]

            fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
            fig.patch.set_facecolor('white')

            page_num = (page_start // samples_per_page) + 1
            total_pages = (len(sample_names) + samples_per_page - 1) // samples_per_page

            draw_header(fig, "CROMATOGRAMES ANALITZATS",
                       f"Seqüència: {seq_name}", seq_name, page_num, total_pages)

            gs = GridSpec(n_rows, n_cols, figure=fig,
                         left=0.06, right=0.98, top=0.88, bottom=0.06,
                         hspace=0.35, wspace=0.25)

            for idx, sample_name in enumerate(page_samples):
                row = idx // n_cols
                col = idx % n_cols
                ax = fig.add_subplot(gs[row, col])

                sample_data = samples_grouped.get(sample_name, {})
                _plot_analyzed_sample(ax, sample_name, sample_data)

            pdf.savefig(fig, dpi=150)
            plt.close(fig)

    return pdf_path


def _plot_analyzed_sample(ax, sample_name, sample_data):
    """Plotar mostra analitzada amb totes les rèpliques."""
    replicas = sample_data.get("replicas", {})

    if not replicas:
        ax.text(0.5, 0.5, "Sense dades", ha='center', va='center',
               fontsize=8, color='gray')
        ax.set_title(sample_name, fontsize=7, fontweight='bold')
        ax.axis('off')
        return

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    has_data = False

    for i, (rep_key, rep_data) in enumerate(sorted(replicas.items())):
        color = colors[i % len(colors)]

        # Obtenir cromatograma (preferir DOC_final, després direct)
        t = rep_data.get("t")
        y = rep_data.get("y_final") or rep_data.get("y") or rep_data.get("y_raw")

        # Format alternatiu: dins de "direct"
        if t is None:
            direct = rep_data.get("direct", {})
            t = direct.get("t")
            y = direct.get("y") or direct.get("y_raw")

        if t is not None and y is not None:
            t = np.asarray(t)
            y = np.asarray(y)
            if len(t) > 0 and len(y) > 0:
                area = rep_data.get("area_total", 0)
                label = f"R{rep_key}"
                if area > 0:
                    label += f" ({area:.0f})"
                ax.plot(t, y, color=color, linewidth=0.6, alpha=0.8, label=label)
                has_data = True

    if not has_data:
        ax.text(0.5, 0.5, "Sense dades", ha='center', va='center',
               fontsize=8, color='gray')
        ax.axis('off')
    else:
        ax.set_xlabel("min", fontsize=6)
        ax.set_ylabel("mAU", fontsize=6)
        ax.tick_params(axis='both', labelsize=6)
        ax.grid(True, alpha=0.3, linewidth=0.3)
        ax.legend(fontsize=5, loc='upper right', framealpha=0.7)

    # Truncar nom si és massa llarg
    display_name = sample_name[:18] + "..." if len(sample_name) > 18 else sample_name
    ax.set_title(display_name, fontsize=7, fontweight='bold')


# =============================================================================
# REPORT CALIBRACIÓ
# =============================================================================

def generate_calibration_report(calibration=None, output_path=None):
    """
    Genera informe PDF de calibració.

    Llegeix TOTES les dades del Calibration_Reference.json i KHP_History.json.
    NO recalcula regressions — usa les dades emmagatzemades.

    Args:
        calibration: dict de calibració específica (None = activa).
                     Ha de contenir 'regression_data' amb punts i stats.
        output_path: carpeta de sortida (None = REGISTRY/)

    Returns:
        str: path del PDF generat, o None si error
    """
    import json as _json
    from hpsec_calibrate import (
        load_calibration_reference, get_active_global_calibration,
        get_registry_folder, compute_calibration_fingerprint,
        load_khp_history
    )

    apply_style()

    # Obtenir calibració
    if calibration is None:
        calibration = get_active_global_calibration()
    if not calibration:
        logger.error("No hi ha calibració per generar informe")
        return None

    # Dades de regressió (guardades al JSON)
    reg = calibration.get('regression_data', {})
    points = reg.get('points', [])
    stats_conc = reg.get('stats_per_concentration', {})

    # Output path
    if output_path is None:
        registry = get_registry_folder()
        if registry:
            output_path = registry
        else:
            output_path = os.path.dirname(os.path.abspath(__file__))

    os.makedirs(output_path, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    cal_id = calibration.get('id', 'unknown')
    pdf_path = os.path.join(output_path, f"REPORT_Calibracio_{ts}.pdf")

    # Dades bàsiques (protegir contra None i dict — calibracions antigues pre-regression_data)
    from hpsec_calibrate import _extract_rf_from_cal, _extract_intercept_from_cal

    mode = (reg.get('mode') or
            calibration.get('source', {}).get('mode') or
            'COLUMN')
    signal = reg.get('signal') or calibration.get('signal_scope') or 'direct'
    rf_cal = reg.get('rf_mass_cal') or 0
    intercept_raw = reg.get('intercept') or 0
    # Si rf_cal/intercept són dict, extreure valor escalar (suporta v2.0 nested i v3.0 planer)
    if isinstance(rf_cal, dict):
        rf_cal = rf_cal.get(mode.lower(), 0) or rf_cal.get(signal, {}).get(mode.lower(), 0) if isinstance(rf_cal, dict) else 0
    if isinstance(intercept_raw, dict):
        intercept_raw = intercept_raw.get(mode.lower(), 0) or intercept_raw.get(signal, {}).get(mode.lower(), 0) if isinstance(intercept_raw, dict) else 0
    intercept = float(intercept_raw) if intercept_raw else 0
    rf_cal = float(rf_cal) if rf_cal else 0
    r2 = reg.get('r2', calibration.get('r2', 0)) or 0
    if isinstance(r2, dict):
        r2 = r2.get(mode.lower(), 0) or 0
    r2 = float(r2)
    n_pts = reg.get('n_points', calibration.get('n_points', 0)) or 0
    rms = reg.get('residuals_rms', 0) or 0
    model_type = reg.get('model', calibration.get('model', 'intercept')) or 'intercept'
    fingerprint = compute_calibration_fingerprint(calibration)

    # Extreure RF per mode (suporta v3.0 planer i v2.0 nested)
    def _extract_rf(sig, mod):
        return _extract_rf_from_cal(calibration, mod, sig) or 0

    def _extract_int(sig, mod):
        return _extract_intercept_from_cal(calibration, mod, sig) or 0

    # Àmbit (signal_scope) de la calibració — usat a tots els títols
    cal_signal = calibration.get('signal_scope', signal).lower()
    cal_signal_label = cal_signal.upper()
    sens = calibration.get('uib_sensitivity')
    if sens:
        cal_signal_label = f"UIB {int(sens)}"

    # Historial KHP per gràfic temporal i QC (seq_path=None → usa REGISTRY global)
    khp_history = load_khp_history(None) or []

    # All calibrations for history context
    ref = load_calibration_reference() or {}
    all_calibrations = ref.get('calibrations', [])

    with PdfPages(pdf_path) as pdf:
        # =====================================================================
        # PÀGINA 1: Resum executiu
        # =====================================================================
        fig = plt.figure(figsize=(8.27, 11.69))  # A4
        fig.patch.set_facecolor('white')

        draw_header(fig, "INFORME DE CALIBRACIÓ",
                   f"Calibració {cal_id}", page_num=1, total_pages=5)

        # Taula resum
        ax_info = fig.add_axes([0.05, 0.68, 0.9, 0.16])
        ax_info.axis('off')

        valid_from = calibration.get('valid_from', '—')
        valid_to = calibration.get('valid_to', 'Vigent')
        created = calibration.get('metadata', {}).get('created_date', '—')
        source_desc = calibration.get('source', {}).get('description', '—')
        seq_refs = ", ".join(calibration.get('source', {}).get('seq_references', []))

        info_data = [
            ["PARÀMETRE", "VALOR", "PARÀMETRE", "VALOR"],
            ["ID", cal_id, "Vigent des de", str(valid_from)],
            ["Model", model_type, "Vigent fins", str(valid_to) if valid_to else "Vigent"],
            ["Mode", mode, "Creat", str(created)],
            ["Font", source_desc[:30], "SEQs referència", seq_refs[:30] if seq_refs else "—"],
            ["Fingerprint", fingerprint[:16], "Motiu", calibration.get('metadata', {}).get('reason', '—')[:30]],
        ]

        tbl = ax_info.table(cellText=info_data, loc='center', cellLoc='center',
                            colWidths=[0.18, 0.32, 0.18, 0.32])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1.0, 1.8)
        for j in range(4):
            tbl[(0, j)].set_facecolor(COLORS["primary"])
            tbl[(0, j)].set_text_props(color='white', fontweight='bold')

        # Taula paràmetres per mode
        ax_params = fig.add_axes([0.05, 0.50, 0.9, 0.14])
        ax_params.axis('off')

        params_data = [
            ["SENYAL / MODE", "RF (slope)", "Intercept", "R²", "N punts", "RMS"],
            [f"{cal_signal_label} / COLUMN",
             f"{_extract_rf(cal_signal, 'column'):.1f}",
             f"{_extract_int(cal_signal, 'column'):.1f}", "", "", ""],
            [f"{cal_signal_label} / BP",
             f"{_extract_rf(cal_signal, 'bp'):.1f}",
             f"{_extract_int(cal_signal, 'bp'):.1f}", "", "", ""],
        ]

        # Afegir R² i RMS a la fila corresponent al mode de la regressió
        mode_key = mode.upper() if mode else ""
        for i, row in enumerate(params_data[1:], 1):
            row_mode = "COLUMN" if "COLUMN" in row[0] else "BP"
            if row_mode == mode_key:
                row[3] = f"{r2:.6f}" if r2 else "—"
                row[4] = str(n_pts)
                row[5] = f"{rms:.2f}" if rms else "—"

        tbl2 = ax_params.table(cellText=params_data, loc='center', cellLoc='center',
                               colWidths=[0.25, 0.15, 0.15, 0.15, 0.15, 0.15])
        tbl2.auto_set_font_size(False)
        tbl2.set_fontsize(8)
        tbl2.scale(1.0, 1.8)
        for j in range(6):
            tbl2[(0, j)].set_facecolor(COLORS["dark"])
            tbl2[(0, j)].set_text_props(color='white', fontweight='bold')

        # Equació
        if model_type == 'origin':
            eq_str = f"Àrea = {rf_cal:.1f} × µg_DOC   (R² = {r2:.6f})"
        else:
            eq_str = f"Àrea = {rf_cal:.1f} × µg_DOC + {intercept:.1f}   (R² = {r2:.6f})"

        fig.text(0.5, 0.44, eq_str, ha='center', va='top',
                fontsize=12, fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#e8f4fd', edgecolor='#2980B9'))

        # Stats per concentració (si disponible)
        if stats_conc:
            ax_stats = fig.add_axes([0.05, 0.15, 0.9, 0.24])
            ax_stats.axis('off')

            stats_header = ["Conc (ppm)", "N", "Àrea mitj", "Àrea σ", "RF mitj", "RF σ", "RF CV%"]
            stats_rows = []
            for conc_str, st in sorted(stats_conc.items(), key=lambda x: float(x[0])):
                stats_rows.append([
                    f"{float(conc_str):g}",
                    str(st.get('n', 0)),
                    f"{st.get('area_mean', 0):.1f}",
                    f"{st.get('area_std', 0):.1f}",
                    f"{st.get('rf_mean', 0):.1f}",
                    f"{st.get('rf_std', 0):.1f}",
                    f"{st.get('rf_cv_pct', 0):.1f}%",
                ])

            stats_data = [stats_header] + stats_rows
            tbl3 = ax_stats.table(cellText=stats_data, loc='upper center', cellLoc='center',
                                  colWidths=[0.12, 0.08, 0.16, 0.14, 0.16, 0.14, 0.12])
            tbl3.auto_set_font_size(False)
            tbl3.set_fontsize(8)
            tbl3.scale(1.0, 1.6)
            for j in range(7):
                tbl3[(0, j)].set_facecolor(COLORS["primary"])
                tbl3[(0, j)].set_text_props(color='white', fontweight='bold')

            # Colorar CV alts
            for i, row in enumerate(stats_rows, 1):
                cv = float(row[6].rstrip('%'))
                if cv > 20:
                    tbl3[(i, 6)].set_facecolor('#f8d7da')
                elif cv > 10:
                    tbl3[(i, 6)].set_facecolor('#fff3cd')

        draw_footer(fig, INSTITUTION_LINE)
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # =====================================================================
        # PÀGINA 2: Scatter regressió + residuals
        # =====================================================================
        fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
        fig.patch.set_facecolor('white')

        draw_header(fig, "REGRESSIÓ LINEAL", page_num=2, total_pages=5)

        if points:
            gs = GridSpec(1, 2, figure=fig,
                         left=0.08, right=0.95, top=0.85, bottom=0.12,
                         wspace=0.25, width_ratios=[2, 1])

            # Scatter principal
            ax1 = fig.add_subplot(gs[0])

            x_inc = [p['ug_doc'] for p in points if not p.get('excluded')]
            y_inc = [p['area'] for p in points if not p.get('excluded')]
            x_exc = [p['ug_doc'] for p in points if p.get('excluded')]
            y_exc = [p['area'] for p in points if p.get('excluded')]

            if x_inc:
                ax1.scatter(x_inc, y_inc, c=COLORS["primary"], s=50, zorder=5,
                           edgecolors='white', linewidth=0.5, label='Inclòs')
            if x_exc:
                ax1.scatter(x_exc, y_exc, c=COLORS["danger"], s=50, marker='x',
                           zorder=5, linewidth=1.5, label='Exclòs')

            # Recta de regressió
            all_x = [p['ug_doc'] for p in points]
            if all_x and rf_cal > 0:
                x_line = np.linspace(0, max(all_x) * 1.1, 100)
                y_line = rf_cal * x_line + intercept
                ax1.plot(x_line, y_line, '-', color=COLORS["success"], linewidth=2,
                        label=f'Regressió (RF={rf_cal:.1f})')

                # Banda de predicció 95% (si ≥3 punts)
                n = len(x_inc)
                if n >= 3:
                    try:
                        from scipy.stats import t as t_dist
                        x_arr = np.array(x_inc)
                        y_arr = np.array(y_inc)
                        x_mean = np.mean(x_arr)
                        Sxx = np.sum((x_arr - x_mean)**2)
                        mse = np.sum((y_arr - (rf_cal * x_arr + intercept))**2) / (n - 2)
                        t_val = t_dist.ppf(0.975, n - 2)

                        se_pred = np.sqrt(mse * (1 + 1/n + (x_line - x_mean)**2 / Sxx))
                        ax1.fill_between(x_line,
                                        y_line - t_val * se_pred,
                                        y_line + t_val * se_pred,
                                        alpha=0.12, color=COLORS["primary"],
                                        label='Predicció 95%')
                    except Exception:
                        pass

            ax1.set_xlabel('µg DOC injectat', fontsize=10)
            ax1.set_ylabel('Àrea', fontsize=10)
            ax1.set_title(f'Regressió {mode} — {cal_signal_label}', fontsize=11, fontweight='bold')
            ax1.legend(fontsize=8, loc='upper left')
            ax1.grid(True, alpha=0.3)
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)

            # Etiquetes dels punts (seq_name abreujat)
            for p in points:
                if not p.get('excluded'):
                    label = p.get('seq_name', '')[:6]
                    ax1.annotate(label, (p['ug_doc'], p['area']),
                               fontsize=5, alpha=0.5, ha='center', va='bottom',
                               xytext=(0, 4), textcoords='offset points')

            # Residuals
            ax2 = fig.add_subplot(gs[1])
            inc_pts = [p for p in points if not p.get('excluded')]
            if inc_pts:
                x_res = [p['ug_doc'] for p in inc_pts]
                y_res = [p['residual'] for p in inc_pts]
                colors_res = [COLORS["success"] if abs(r) < rms * 2 else COLORS["warning"]
                             for r in y_res]
                ax2.bar(range(len(x_res)), y_res, color=colors_res, alpha=0.7)
                ax2.axhline(0, color='black', linewidth=0.5)
                ax2.axhline(rms, color=COLORS["warning"], linewidth=0.5, linestyle='--', alpha=0.5)
                ax2.axhline(-rms, color=COLORS["warning"], linewidth=0.5, linestyle='--', alpha=0.5)

                # Etiquetes x: conc ppm
                ax2.set_xticks(range(len(x_res)))
                ax2.set_xticklabels([f"{p['conc_ppm']:g}" for p in inc_pts],
                                    fontsize=7, rotation=45)
                ax2.set_xlabel('Concentració (ppm)', fontsize=8)
            ax2.set_ylabel('Residual', fontsize=8)
            ax2.set_title('Residuals', fontsize=10, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)

        else:
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.7])
            ax.text(0.5, 0.5,
                   "Sense dades de regressió\n(regression_data no disponible al JSON)",
                   ha='center', va='center', fontsize=14, color='gray')
            ax.axis('off')

        draw_footer(fig, INSTITUTION_LINE)
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # =====================================================================
        # PÀGINA 3: Evolució temporal RF (historial KHP)
        # =====================================================================
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.patch.set_facecolor('white')

        draw_header(fig, "EVOLUCIÓ TEMPORAL RF", page_num=3, total_pages=5)

        ax = fig.add_axes([0.08, 0.12, 0.84, 0.72])

        # Filtrar entrades vàlides del historial (per mode I senyal)
        mode_upper = mode.upper()
        hist_entries = []
        for entry in khp_history:
            if entry.get('mode', '').upper() != mode_upper:
                continue
            if not entry.get('valid_for_calibration', True):
                continue
            # Filtrar per senyal (doc_mode): Direct, UIB o DUAL
            entry_doc_mode = entry.get('doc_mode', 'Direct')
            if cal_signal == 'direct' and entry_doc_mode not in ('Direct', 'DUAL'):
                continue
            if cal_signal == 'uib' and entry_doc_mode not in ('UIB', 'DUAL'):
                continue
            rf = entry.get('rf_mass', 0)
            if rf <= 0:
                continue
            hist_entries.append(entry)

        if hist_entries:
            # Separar CAL vs producció
            cal_entries = [e for e in hist_entries if '_CAL' in e.get('seq_name', '').upper()]
            prod_entries = [e for e in hist_entries if '_CAL' not in e.get('seq_name', '').upper()]

            # X = índex seqüencial
            for i, entries in enumerate([prod_entries, cal_entries]):
                if not entries:
                    continue
                x_vals = list(range(len(entries)))
                y_vals = [e['rf_mass'] for e in entries]
                labels = [e.get('seq_name', '')[:8] for e in entries]

                color = COLORS["doc_direct"] if i == 0 else COLORS["success"]
                marker = 'o' if i == 0 else 's'
                label = 'Producció' if i == 0 else 'SEQ_CAL'
                ax.scatter(x_vals, y_vals, c=color, s=25, marker=marker,
                          alpha=0.6, label=label, zorder=3)

            # Línia RF vigent
            if rf_cal > 0:
                ax.axhline(rf_cal, color=COLORS["success"], linewidth=1.5,
                          linestyle='-', alpha=0.8, label=f'RF vigent ({rf_cal:.0f})')
                ax.axhspan(rf_cal * 0.9, rf_cal * 1.1, alpha=0.08, color=COLORS["success"])

            ax.set_xlabel('Entrada (ordre cronològic)', fontsize=9)
            ax.set_ylabel('RF mass (Àrea / µg DOC)', fontsize=9)
            ax.set_title(f'Evolució RF — {mode} {cal_signal_label}', fontsize=11, fontweight='bold')
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        else:
            ax.text(0.5, 0.5, f"Sense historial KHP per mode {mode}",
                   ha='center', va='center', fontsize=14, color='gray')
            ax.axis('off')

        draw_footer(fig, INSTITUTION_LINE)
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # =====================================================================
        # PÀGINA 4: QC Levey-Jennings
        # =====================================================================
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.patch.set_facecolor('white')

        draw_header(fig, "CONTROL DE QUALITAT (LEVEY-JENNINGS)", page_num=4, total_pages=5)

        ax = fig.add_axes([0.08, 0.12, 0.84, 0.72])

        # Producció: calcular desviació % vs recta vigent
        prod_entries = [e for e in hist_entries if '_CAL' not in e.get('seq_name', '').upper()]
        if prod_entries and rf_cal > 0:
            dev_data = []
            for e in prod_entries:
                area = e.get('area', 0)
                conc = e.get('conc_ppm', 0)
                vol = e.get('volume_uL', 0)
                if conc > 0 and vol > 0 and area > 0:
                    ug_expected = conc * vol / 1000.0
                    area_expected = rf_cal * ug_expected + intercept
                    dev_pct = (area - area_expected) / area_expected * 100 if area_expected > 0 else 0
                    dev_data.append({
                        'seq': e.get('seq_name', '')[:10],
                        'dev_pct': dev_pct,
                        'conc': conc,
                    })

            if dev_data:
                x = list(range(len(dev_data)))
                y = [d['dev_pct'] for d in dev_data]

                # Barres colorades per desviació
                colors_bars = []
                for d in y:
                    if abs(d) <= 10:
                        colors_bars.append(COLORS["success"])
                    elif abs(d) <= 20:
                        colors_bars.append(COLORS["warning"])
                    else:
                        colors_bars.append(COLORS["danger"])

                ax.bar(x, y, color=colors_bars, alpha=0.7, width=0.8)
                ax.axhline(0, color='black', linewidth=0.8)

                # Línies de control
                for lim, style, label in [(10, '--', '±10%'), (20, ':', '±20%')]:
                    ax.axhline(lim, color=COLORS["warning"], linewidth=0.8, linestyle=style, alpha=0.7)
                    ax.axhline(-lim, color=COLORS["warning"], linewidth=0.8, linestyle=style, alpha=0.7)

                ax.axhspan(-10, 10, alpha=0.05, color=COLORS["success"])
                ax.axhspan(-20, -10, alpha=0.03, color=COLORS["warning"])
                ax.axhspan(10, 20, alpha=0.03, color=COLORS["warning"])

                # Estadístiques
                n_ok = sum(1 for d in y if abs(d) <= 10)
                n_attn = sum(1 for d in y if 10 < abs(d) <= 20)
                n_out = sum(1 for d in y if abs(d) > 20)
                total = len(y)

                status_text = f"EN CONTROL: {n_ok}/{total}  |  ATENCIÓ: {n_attn}  |  FORA: {n_out}"
                status_color = COLORS["success"] if n_out == 0 and n_attn <= total * 0.1 else (
                    COLORS["warning"] if n_out <= 1 else COLORS["danger"])
                fig.text(0.5, 0.88, status_text, ha='center', fontsize=10,
                        fontweight='bold', color=status_color)

                ax.set_xlabel('Entrada KHP producció', fontsize=9)
                ax.set_ylabel('Desviació vs recta vigent (%)', fontsize=9)
                ax.set_title(f'Levey-Jennings — {mode} {cal_signal_label}', fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
            else:
                ax.text(0.5, 0.5, "No s'han pogut calcular desviacions",
                       ha='center', va='center', fontsize=14, color='gray')
                ax.axis('off')
        else:
            msg = "Sense dades de producció" if not prod_entries else "RF vigent = 0"
            ax.text(0.5, 0.5, msg, ha='center', va='center', fontsize=14, color='gray')
            ax.axis('off')

        draw_footer(fig, INSTITUTION_LINE)
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # =====================================================================
        # PÀGINA 5: Historial de calibracions
        # =====================================================================
        fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
        fig.patch.set_facecolor('white')

        draw_header(fig, "HISTORIAL DE CALIBRACIONS", page_num=5, total_pages=5)

        ax_hist = fig.add_axes([0.05, 0.15, 0.9, 0.70])
        ax_hist.axis('off')

        # Taula amb totes les calibracions
        hist_header = ["ID", "Àmbit", "Des de", "Fins", "RF COL", "RF BP", "R²", "Punts", "Font"]
        hist_rows = []
        for cal_entry in all_calibrations[:20]:  # Màxim 20 entrades
            scope_h = cal_entry.get('signal_scope', '?')
            rf_col_val = _extract_rf_from_cal(cal_entry, 'column', scope_h) or 0
            rf_bp_val = _extract_rf_from_cal(cal_entry, 'bp', scope_h) or 0

            sens_h = cal_entry.get('uib_sensitivity')
            scope_label = scope_h.upper()
            if sens_h:
                scope_label = f"UIB {int(sens_h)}"

            r2_h = cal_entry.get('r2', 0) or 0
            if isinstance(r2_h, dict):
                # Mostrar la R² del mode que tingui valor
                r2_h = r2_h.get('column', 0) or r2_h.get('bp', 0) or 0
            r2_h = float(r2_h) if r2_h else 0

            n_pts_h = cal_entry.get('n_points', '—')
            if isinstance(n_pts_h, dict):
                n_pts_h = n_pts_h.get('column', 0) or n_pts_h.get('bp', 0) or '—'

            src = cal_entry.get('source', {}).get('type', '')
            hist_rows.append([
                cal_entry.get('id', '—')[-18:],
                scope_label,
                str(cal_entry.get('valid_from', '—'))[:10],
                str(cal_entry.get('valid_to', 'Vigent'))[:10] if cal_entry.get('valid_to') else 'Vigent',
                f"{rf_col_val:.0f}" if rf_col_val else "—",
                f"{rf_bp_val:.0f}" if rf_bp_val else "—",
                f"{r2_h:.4f}" if r2_h else "—",
                str(n_pts_h),
                src[:12],
            ])

        if hist_rows:
            hist_data = [hist_header] + hist_rows
            tbl4 = ax_hist.table(cellText=hist_data, loc='upper center', cellLoc='center',
                                 colWidths=[0.16, 0.08, 0.10, 0.10, 0.09, 0.09, 0.10, 0.08, 0.10])
            tbl4.auto_set_font_size(False)
            tbl4.set_fontsize(7)
            tbl4.scale(1.0, 1.5)
            for j in range(9):
                tbl4[(0, j)].set_facecolor(COLORS["primary"])
                tbl4[(0, j)].set_text_props(color='white', fontweight='bold')

            # Marcar activa
            for i, cal_entry in enumerate(all_calibrations[:20], 1):
                if cal_entry.get('is_active'):
                    for j in range(9):
                        tbl4[(i, j)].set_facecolor('#d4edda')

        draw_footer(fig, INSTITUTION_LINE)
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # =====================================================================
        # PÀGINES EXTRA: Cromatogrames KHP (si disponibles)
        # =====================================================================
        chromatogram_pngs = _find_khp_chromatogram_pngs(calibration, reg)
        if chromatogram_pngs:
            plots_per_page = 6  # GridSpec(3, 2)
            n_chrom_pages = (len(chromatogram_pngs) + plots_per_page - 1) // plots_per_page

            for page_idx in range(n_chrom_pages):
                fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
                fig.patch.set_facecolor('white')

                start = page_idx * plots_per_page
                end = min(start + plots_per_page, len(chromatogram_pngs))
                page_pngs = chromatogram_pngs[start:end]

                fig.text(0.5, 0.97,
                         f"CROMATOGRAMES KHP — Pàg. {page_idx + 1}/{n_chrom_pages}",
                         ha='center', va='top', fontsize=12, fontweight='bold',
                         color='#2C3E50')

                gs = GridSpec(3, 2, figure=fig, left=0.04, right=0.96,
                              top=0.93, bottom=0.04, hspace=0.3, wspace=0.15)

                for plot_idx, png_path in enumerate(page_pngs):
                    row = plot_idx // 2
                    col_idx = plot_idx % 2
                    ax_chr = fig.add_subplot(gs[row, col_idx])
                    try:
                        img = plt.imread(png_path)
                        ax_chr.imshow(img)
                        ax_chr.axis('off')
                        # Subtítol del fitxer
                        fname = os.path.basename(png_path).replace('.png', '')
                        ax_chr.set_title(fname, fontsize=7, pad=2)
                    except Exception as e_img:
                        ax_chr.text(0.5, 0.5, f"Error: {e_img}", ha='center',
                                   va='center', fontsize=8, color='#E74C3C')
                        ax_chr.axis('off')

                # Amagar subplots buits
                for plot_idx in range(len(page_pngs), plots_per_page):
                    row = plot_idx // 2
                    col_idx = plot_idx % 2
                    ax_empty = fig.add_subplot(gs[row, col_idx])
                    ax_empty.axis('off')

                pdf.savefig(fig, dpi=120)
                plt.close(fig)

    import logging
    logging.getLogger(__name__).info(f"Informe calibració generat: {pdf_path}")
    return pdf_path


def generate_dual_calibration_report(output_path=None):
    """
    Genera informe PDF de calibració amb AMBDÓS senyals (Direct + UIB).

    Produeix un sol PDF combinat:
    - Pàg 1: Resum executiu amb paràmetres Direct + UIB
    - Pàg 2: Regressió Direct
    - Pàg 3: Regressió UIB
    - Pàg 4: Evolució temporal (2 subplots: Direct + UIB)
    - Pàg 5: QC Levey-Jennings (2 subplots: Direct + UIB)
    - Pàg 6: Historial calibracions
    - Pàg 7+: Cromatogrames KHP

    Args:
        output_path: carpeta de sortida (None = REGISTRY/)

    Returns:
        str: path del PDF generat, o None si error
    """
    import json as _json
    from hpsec_calibrate import (
        load_calibration_reference, get_active_global_calibration,
        get_registry_folder, compute_calibration_fingerprint,
        load_khp_history, _extract_rf_from_cal, _extract_intercept_from_cal
    )

    apply_style()

    # Obtenir calibracions Direct i UIB
    cal_direct = get_active_global_calibration(signal='direct')
    cal_uib = get_active_global_calibration(signal='uib')

    if not cal_direct and not cal_uib:
        logger.error("No hi ha cap calibració activa per generar informe dual")
        return None

    # Output path
    if output_path is None:
        registry = get_registry_folder()
        if registry:
            output_path = registry
        else:
            output_path = os.path.dirname(os.path.abspath(__file__))

    os.makedirs(output_path, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    pdf_path = os.path.join(output_path, f"REPORT_Calibracio_Dual_{ts}.pdf")

    # Preparar dades per cada senyal
    signals_data = []
    for sig_name, cal in [('direct', cal_direct), ('uib', cal_uib)]:
        if not cal:
            continue
        reg = cal.get('regression_data', {})
        mode = (reg.get('mode') or
                cal.get('source', {}).get('mode') or 'COLUMN')
        rf_cal = reg.get('rf_mass_cal') or 0
        intercept_raw = reg.get('intercept') or 0
        if isinstance(rf_cal, dict):
            rf_cal = rf_cal.get(mode.lower(), 0) or 0
        if isinstance(intercept_raw, dict):
            intercept_raw = intercept_raw.get(mode.lower(), 0) or 0
        intercept = float(intercept_raw) if intercept_raw else 0
        rf_cal = float(rf_cal) if rf_cal else 0
        r2 = reg.get('r2', cal.get('r2', 0)) or 0
        if isinstance(r2, dict):
            r2 = r2.get(mode.lower(), 0) or 0
        r2 = float(r2)
        n_pts = reg.get('n_points', cal.get('n_points', 0)) or 0
        rms = reg.get('residuals_rms', 0) or 0
        model_type = reg.get('model', cal.get('model', 'intercept')) or 'intercept'

        cal_signal_label = sig_name.upper()
        sens = cal.get('uib_sensitivity')
        if sens:
            cal_signal_label = f"UIB {int(sens)}"

        signals_data.append({
            'signal': sig_name,
            'cal': cal,
            'reg': reg,
            'mode': mode,
            'rf_cal': rf_cal,
            'intercept': intercept,
            'r2': r2,
            'n_pts': n_pts,
            'rms': rms,
            'model_type': model_type,
            'points': reg.get('points', []),
            'stats_conc': reg.get('stats_per_concentration', {}),
            'label': cal_signal_label,
            'sens': sens,
        })

    if not signals_data:
        logger.error("Cap senyal amb dades de calibració vàlides")
        return None

    # Historial i referència
    khp_history = load_khp_history(None) or []
    ref = load_calibration_reference() or {}
    all_calibrations = ref.get('calibrations', [])
    fingerprint = compute_calibration_fingerprint()

    n_signal_pages = len(signals_data)  # 1 o 2 pàgines de regressió
    total_pages = 3 + n_signal_pages  # resum + n_regressions + temporal + QC + historial
    # total_pages: 1 (resum) + n_signal_pages (regressió) + 1 (temporal) + 1 (QC) + 1 (historial)
    total_pages = 2 + n_signal_pages + 2  # = resum + regressions + temporal+QC combinats + historial

    with PdfPages(pdf_path) as pdf:
        # =================================================================
        # PÀGINA 1: Resum executiu combinat
        # =================================================================
        fig = plt.figure(figsize=(8.27, 11.69))  # A4
        fig.patch.set_facecolor('white')

        cal_ids = ", ".join(sd['cal'].get('id', '?')[-18:] for sd in signals_data)
        draw_header(fig, "INFORME DE CALIBRACIÓ",
                    f"Dual: Direct + UIB" if len(signals_data) > 1 else signals_data[0]['label'],
                    page_num=1, total_pages=total_pages)

        # Taula resum (dades de la primera calibració per les metadades generals)
        first_cal = signals_data[0]['cal']
        ax_info = fig.add_axes([0.05, 0.72, 0.9, 0.14])
        ax_info.axis('off')

        valid_from = first_cal.get('valid_from', '—')
        valid_to = first_cal.get('valid_to', 'Vigent')
        created = first_cal.get('metadata', {}).get('created_date', '—')
        source_desc = first_cal.get('source', {}).get('description', '—')
        seq_refs_all = set()
        for sd in signals_data:
            seq_refs_all.update(sd['cal'].get('source', {}).get('seq_references', []))
        seq_refs = ", ".join(sorted(seq_refs_all))

        info_data = [
            ["PARÀMETRE", "VALOR", "PARÀMETRE", "VALOR"],
            ["IDs", cal_ids, "Vigent des de", str(valid_from)],
            ["Senyals", " + ".join(sd['label'] for sd in signals_data),
             "Vigent fins", str(valid_to) if valid_to else "Vigent"],
            ["Mode", signals_data[0]['mode'], "Creat", str(created)],
            ["Font", source_desc[:30], "SEQs referència", seq_refs[:30] if seq_refs else "—"],
            ["Fingerprint", fingerprint[:16], "Motiu",
             first_cal.get('metadata', {}).get('reason', '—')[:30]],
        ]

        tbl = ax_info.table(cellText=info_data, loc='center', cellLoc='center',
                            colWidths=[0.18, 0.32, 0.18, 0.32])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1.0, 1.8)
        for j in range(4):
            tbl[(0, j)].set_facecolor(COLORS["primary"])
            tbl[(0, j)].set_text_props(color='white', fontweight='bold')

        # Taula paràmetres — files per cada senyal x mode
        y_params = 0.52 if len(signals_data) > 1 else 0.56
        h_params = 0.16 if len(signals_data) > 1 else 0.12
        ax_params = fig.add_axes([0.05, y_params, 0.9, h_params])
        ax_params.axis('off')

        params_data = [["SENYAL / MODE", "RF (slope)", "Intercept", "R²", "N punts", "RMS"]]
        for sd in signals_data:
            for mode_name in ['COLUMN', 'BP']:
                rf_v = _extract_rf_from_cal(sd['cal'], mode_name.lower(), sd['signal']) or 0
                int_v = _extract_intercept_from_cal(sd['cal'], mode_name.lower(), sd['signal']) or 0
                r2_v = ""
                n_v = ""
                rms_v = ""
                if sd['mode'].upper() == mode_name:
                    r2_v = f"{sd['r2']:.6f}" if sd['r2'] else "—"
                    n_v = str(sd['n_pts'])
                    rms_v = f"{sd['rms']:.2f}" if sd['rms'] else "—"
                params_data.append([
                    f"{sd['label']} / {mode_name}",
                    f"{rf_v:.1f}" if rf_v else "—",
                    f"{int_v:.1f}" if int_v else "—",
                    r2_v, n_v, rms_v,
                ])

        tbl2 = ax_params.table(cellText=params_data, loc='center', cellLoc='center',
                                colWidths=[0.25, 0.15, 0.15, 0.15, 0.15, 0.15])
        tbl2.auto_set_font_size(False)
        tbl2.set_fontsize(8)
        tbl2.scale(1.0, 1.6)
        for j in range(6):
            tbl2[(0, j)].set_facecolor(COLORS["dark"])
            tbl2[(0, j)].set_text_props(color='white', fontweight='bold')

        # Equacions per cada senyal
        eq_y = 0.46
        for sd in signals_data:
            if sd['model_type'] == 'origin':
                eq_str = f"{sd['label']}: Àrea = {sd['rf_cal']:.1f} × µg_DOC   (R² = {sd['r2']:.6f})"
            else:
                eq_str = (f"{sd['label']}: Àrea = {sd['rf_cal']:.1f} × µg_DOC + "
                          f"{sd['intercept']:.1f}   (R² = {sd['r2']:.6f})")
            fig.text(0.5, eq_y, eq_str, ha='center', va='top',
                     fontsize=10, fontfamily='monospace',
                     bbox=dict(boxstyle='round,pad=0.4', facecolor='#e8f4fd',
                               edgecolor='#2980B9'))
            eq_y -= 0.04

        # Stats per concentració — per cada senyal
        stats_y = eq_y - 0.02
        for sd in signals_data:
            if not sd['stats_conc']:
                continue
            # Títol senyal
            fig.text(0.05, stats_y, f"Estadístiques per concentració — {sd['label']}",
                     fontsize=9, fontweight='bold', color=COLORS["dark"])
            stats_y -= 0.01

            ax_stats = fig.add_axes([0.05, max(stats_y - 0.13, 0.05), 0.9, 0.13])
            ax_stats.axis('off')

            stats_header = ["Conc (ppm)", "N", "Àrea mitj", "Àrea σ",
                            "RF mitj", "RF σ", "RF CV%"]
            stats_rows = []
            for conc_str, st in sorted(sd['stats_conc'].items(),
                                        key=lambda x: float(x[0])):
                stats_rows.append([
                    f"{float(conc_str):g}",
                    str(st.get('n', 0)),
                    f"{st.get('area_mean', 0):.1f}",
                    f"{st.get('area_std', 0):.1f}",
                    f"{st.get('rf_mean', 0):.1f}",
                    f"{st.get('rf_std', 0):.1f}",
                    f"{st.get('rf_cv_pct', 0):.1f}%",
                ])

            stats_data_tbl = [stats_header] + stats_rows
            tbl3 = ax_stats.table(cellText=stats_data_tbl, loc='upper center',
                                   cellLoc='center',
                                   colWidths=[0.12, 0.08, 0.16, 0.14, 0.16, 0.14, 0.12])
            tbl3.auto_set_font_size(False)
            tbl3.set_fontsize(7)
            tbl3.scale(1.0, 1.4)
            for j in range(7):
                tbl3[(0, j)].set_facecolor(COLORS["primary"])
                tbl3[(0, j)].set_text_props(color='white', fontweight='bold')
            for i, row in enumerate(stats_rows, 1):
                cv = float(row[6].rstrip('%'))
                if cv > 20:
                    tbl3[(i, 6)].set_facecolor('#f8d7da')
                elif cv > 10:
                    tbl3[(i, 6)].set_facecolor('#fff3cd')

            stats_y -= 0.16

        draw_footer(fig, INSTITUTION_LINE)
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # =================================================================
        # PÀGINES 2-(2+n): Regressió per senyal
        # =================================================================
        for page_i, sd in enumerate(signals_data, 2):
            fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
            fig.patch.set_facecolor('white')

            draw_header(fig, f"REGRESSIÓ LINEAL — {sd['label']}",
                        f"Mode: {sd['mode']}",
                        page_num=page_i, total_pages=total_pages)

            points = sd['points']
            rf_cal = sd['rf_cal']
            intercept = sd['intercept']
            rms = sd['rms']

            if points:
                gs = GridSpec(1, 2, figure=fig,
                              left=0.08, right=0.95, top=0.85, bottom=0.12,
                              wspace=0.25, width_ratios=[2, 1])

                ax1 = fig.add_subplot(gs[0])
                x_inc = [p['ug_doc'] for p in points if not p.get('excluded')]
                y_inc = [p['area'] for p in points if not p.get('excluded')]
                x_exc = [p['ug_doc'] for p in points if p.get('excluded')]
                y_exc = [p['area'] for p in points if p.get('excluded')]

                if x_inc:
                    ax1.scatter(x_inc, y_inc, c=COLORS["primary"], s=50, zorder=5,
                                edgecolors='white', linewidth=0.5, label='Inclòs')
                if x_exc:
                    ax1.scatter(x_exc, y_exc, c=COLORS["danger"], s=50, marker='x',
                                zorder=5, linewidth=1.5, label='Exclòs')

                all_x = [p['ug_doc'] for p in points]
                if all_x and rf_cal > 0:
                    x_line = np.linspace(0, max(all_x) * 1.1, 100)
                    y_line = rf_cal * x_line + intercept
                    ax1.plot(x_line, y_line, '-', color=COLORS["success"], linewidth=2,
                             label=f'Regressió (RF={rf_cal:.1f})')

                    n = len(x_inc)
                    if n >= 3:
                        try:
                            from scipy.stats import t as t_dist
                            x_arr = np.array(x_inc)
                            y_arr = np.array(y_inc)
                            x_mean = np.mean(x_arr)
                            Sxx = np.sum((x_arr - x_mean) ** 2)
                            mse = np.sum((y_arr - (rf_cal * x_arr + intercept)) ** 2) / (n - 2)
                            t_val = t_dist.ppf(0.975, n - 2)
                            se_pred = np.sqrt(mse * (1 + 1 / n + (x_line - x_mean) ** 2 / Sxx))
                            ax1.fill_between(x_line,
                                             y_line - t_val * se_pred,
                                             y_line + t_val * se_pred,
                                             alpha=0.12, color=COLORS["primary"],
                                             label='Predicció 95%')
                        except Exception:
                            pass

                ax1.set_xlabel('µg DOC injectat', fontsize=10)
                ax1.set_ylabel('Àrea', fontsize=10)
                ax1.set_title(f'Regressió {sd["mode"]} — {sd["label"]}',
                              fontsize=11, fontweight='bold')
                ax1.legend(fontsize=8, loc='upper left')
                ax1.grid(True, alpha=0.3)
                ax1.spines['top'].set_visible(False)
                ax1.spines['right'].set_visible(False)

                for p in points:
                    if not p.get('excluded'):
                        label = p.get('seq_name', '')[:6]
                        ax1.annotate(label, (p['ug_doc'], p['area']),
                                     fontsize=5, alpha=0.5, ha='center', va='bottom',
                                     xytext=(0, 4), textcoords='offset points')

                ax2 = fig.add_subplot(gs[1])
                inc_pts = [p for p in points if not p.get('excluded')]
                if inc_pts:
                    x_res = [p['ug_doc'] for p in inc_pts]
                    y_res = [p['residual'] for p in inc_pts]
                    colors_res = [COLORS["success"] if abs(r) < rms * 2 else COLORS["warning"]
                                  for r in y_res]
                    ax2.bar(range(len(x_res)), y_res, color=colors_res, alpha=0.7)
                    ax2.axhline(0, color='black', linewidth=0.5)
                    ax2.axhline(rms, color=COLORS["warning"], linewidth=0.5,
                                linestyle='--', alpha=0.5)
                    ax2.axhline(-rms, color=COLORS["warning"], linewidth=0.5,
                                linestyle='--', alpha=0.5)
                    ax2.set_xticks(range(len(x_res)))
                    ax2.set_xticklabels([f"{p['conc_ppm']:g}" for p in inc_pts],
                                        fontsize=7, rotation=45)
                    ax2.set_xlabel('Concentració (ppm)', fontsize=8)
                ax2.set_ylabel('Residual', fontsize=8)
                ax2.set_title('Residuals', fontsize=10, fontweight='bold')
                ax2.grid(True, alpha=0.3, axis='y')
                ax2.spines['top'].set_visible(False)
                ax2.spines['right'].set_visible(False)
            else:
                ax = fig.add_axes([0.1, 0.1, 0.8, 0.7])
                ax.text(0.5, 0.5,
                        f"Sense dades de regressió per {sd['label']}\n"
                        "(regression_data no disponible al JSON)",
                        ha='center', va='center', fontsize=14, color='gray')
                ax.axis('off')

            draw_footer(fig, INSTITUTION_LINE)
            pdf.savefig(fig, dpi=150)
            plt.close(fig)

        # =================================================================
        # PÀGINA: Evolució temporal RF (2 subplots si dual)
        # =================================================================
        page_temporal = 2 + n_signal_pages
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.patch.set_facecolor('white')
        draw_header(fig, "EVOLUCIÓ TEMPORAL RF", page_num=page_temporal,
                    total_pages=total_pages)

        n_subplots = len(signals_data)
        gs = GridSpec(n_subplots, 1, figure=fig,
                      left=0.08, right=0.95,
                      top=0.85, bottom=0.08,
                      hspace=0.35)

        for subplot_i, sd in enumerate(signals_data):
            ax = fig.add_subplot(gs[subplot_i])
            mode_upper = sd['mode'].upper()
            cal_signal = sd['signal']

            hist_entries = []
            for entry in khp_history:
                if entry.get('mode', '').upper() != mode_upper:
                    continue
                if not entry.get('valid_for_calibration', True):
                    continue
                entry_doc_mode = entry.get('doc_mode', 'Direct')
                if cal_signal == 'direct' and entry_doc_mode not in ('Direct', 'DUAL'):
                    continue
                if cal_signal == 'uib' and entry_doc_mode not in ('UIB', 'DUAL'):
                    continue
                rf = entry.get('rf_mass', 0)
                if rf <= 0:
                    continue
                hist_entries.append(entry)

            if hist_entries:
                cal_entries = [e for e in hist_entries
                               if '_CAL' in e.get('seq_name', '').upper()]
                prod_entries = [e for e in hist_entries
                                if '_CAL' not in e.get('seq_name', '').upper()]

                for i, entries in enumerate([prod_entries, cal_entries]):
                    if not entries:
                        continue
                    x_vals = list(range(len(entries)))
                    y_vals = [e['rf_mass'] for e in entries]
                    color = COLORS["doc_direct"] if i == 0 else COLORS["success"]
                    marker = 'o' if i == 0 else 's'
                    label = 'Producció' if i == 0 else 'SEQ_CAL'
                    ax.scatter(x_vals, y_vals, c=color, s=20, marker=marker,
                               alpha=0.6, label=label, zorder=3)

                if sd['rf_cal'] > 0:
                    ax.axhline(sd['rf_cal'], color=COLORS["success"], linewidth=1.5,
                               linestyle='-', alpha=0.8,
                               label=f'RF vigent ({sd["rf_cal"]:.0f})')
                    ax.axhspan(sd['rf_cal'] * 0.9, sd['rf_cal'] * 1.1,
                               alpha=0.08, color=COLORS["success"])

                ax.set_ylabel('RF mass', fontsize=8)
                ax.set_title(f'{sd["label"]} — {sd["mode"]}', fontsize=10,
                             fontweight='bold')
                ax.legend(fontsize=7, loc='upper right')
                ax.grid(True, alpha=0.3)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
            else:
                ax.text(0.5, 0.5, f"Sense historial KHP per {sd['label']}",
                        ha='center', va='center', fontsize=12, color='gray')
                ax.axis('off')

            if subplot_i == n_subplots - 1:
                ax.set_xlabel('Entrada (ordre cronològic)', fontsize=8)

        draw_footer(fig, INSTITUTION_LINE)
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # =================================================================
        # PÀGINA: QC Levey-Jennings (2 subplots si dual)
        # =================================================================
        page_qc = page_temporal + 1
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.patch.set_facecolor('white')
        draw_header(fig, "CONTROL DE QUALITAT (LEVEY-JENNINGS)",
                    page_num=page_qc, total_pages=total_pages)

        gs = GridSpec(n_subplots, 1, figure=fig,
                      left=0.08, right=0.95,
                      top=0.85, bottom=0.08,
                      hspace=0.35)

        for subplot_i, sd in enumerate(signals_data):
            ax = fig.add_subplot(gs[subplot_i])
            mode_upper = sd['mode'].upper()
            cal_signal = sd['signal']
            rf_cal = sd['rf_cal']
            intercept = sd['intercept']

            # Re-filtrar historial per senyal
            hist_entries = []
            for entry in khp_history:
                if entry.get('mode', '').upper() != mode_upper:
                    continue
                if not entry.get('valid_for_calibration', True):
                    continue
                entry_doc_mode = entry.get('doc_mode', 'Direct')
                if cal_signal == 'direct' and entry_doc_mode not in ('Direct', 'DUAL'):
                    continue
                if cal_signal == 'uib' and entry_doc_mode not in ('UIB', 'DUAL'):
                    continue
                if entry.get('rf_mass', 0) <= 0:
                    continue
                hist_entries.append(entry)

            prod_entries = [e for e in hist_entries
                            if '_CAL' not in e.get('seq_name', '').upper()]

            if prod_entries and rf_cal > 0:
                dev_data = []
                for e in prod_entries:
                    area = e.get('area', 0)
                    conc = e.get('conc_ppm', 0)
                    vol = e.get('volume_uL', 0)
                    if conc > 0 and vol > 0 and area > 0:
                        ug_expected = conc * vol / 1000.0
                        area_expected = rf_cal * ug_expected + intercept
                        dev_pct = ((area - area_expected) / area_expected * 100
                                   if area_expected > 0 else 0)
                        dev_data.append({'dev_pct': dev_pct})

                if dev_data:
                    x = list(range(len(dev_data)))
                    y = [d['dev_pct'] for d in dev_data]

                    colors_bars = []
                    for d in y:
                        if abs(d) <= 10:
                            colors_bars.append(COLORS["success"])
                        elif abs(d) <= 20:
                            colors_bars.append(COLORS["warning"])
                        else:
                            colors_bars.append(COLORS["danger"])

                    ax.bar(x, y, color=colors_bars, alpha=0.7, width=0.8)
                    ax.axhline(0, color='black', linewidth=0.8)

                    for lim, style in [(10, '--'), (20, ':')]:
                        ax.axhline(lim, color=COLORS["warning"], linewidth=0.8,
                                   linestyle=style, alpha=0.7)
                        ax.axhline(-lim, color=COLORS["warning"], linewidth=0.8,
                                   linestyle=style, alpha=0.7)

                    ax.axhspan(-10, 10, alpha=0.05, color=COLORS["success"])

                    n_ok = sum(1 for d in y if abs(d) <= 10)
                    n_out = sum(1 for d in y if abs(d) > 20)
                    status_color = (COLORS["success"] if n_out == 0
                                    else COLORS["danger"])
                    ax.set_title(
                        f'{sd["label"]} — EN CONTROL: {n_ok}/{len(y)}, '
                        f'FORA: {n_out}',
                        fontsize=10, fontweight='bold', color=status_color)
                    ax.set_ylabel('Desviació %', fontsize=8)
                    ax.grid(True, alpha=0.3, axis='y')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                else:
                    ax.text(0.5, 0.5, f"Sense desviacions per {sd['label']}",
                            ha='center', va='center', fontsize=12, color='gray')
                    ax.axis('off')
            else:
                ax.text(0.5, 0.5, f"Sense dades producció per {sd['label']}",
                        ha='center', va='center', fontsize=12, color='gray')
                ax.axis('off')

            if subplot_i == n_subplots - 1:
                ax.set_xlabel('Entrada KHP producció', fontsize=8)

        draw_footer(fig, INSTITUTION_LINE)
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # =================================================================
        # PÀGINA: Historial de calibracions
        # =================================================================
        page_hist = page_qc + 1
        fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
        fig.patch.set_facecolor('white')
        draw_header(fig, "HISTORIAL DE CALIBRACIONS",
                    page_num=page_hist, total_pages=total_pages)

        ax_hist = fig.add_axes([0.05, 0.15, 0.9, 0.70])
        ax_hist.axis('off')

        hist_header = ["ID", "Àmbit", "Des de", "Fins", "RF COL", "RF BP",
                       "R²", "Punts", "Font"]
        hist_rows = []
        for cal_entry in all_calibrations[:20]:
            scope_h = cal_entry.get('signal_scope', '?')
            rf_col_val = _extract_rf_from_cal(cal_entry, 'column', scope_h) or 0
            rf_bp_val = _extract_rf_from_cal(cal_entry, 'bp', scope_h) or 0
            sens_h = cal_entry.get('uib_sensitivity')
            scope_label = scope_h.upper()
            if sens_h:
                scope_label = f"UIB {int(sens_h)}"
            r2_h = cal_entry.get('r2', 0) or 0
            if isinstance(r2_h, dict):
                r2_h = r2_h.get('column', 0) or r2_h.get('bp', 0) or 0
            r2_h = float(r2_h) if r2_h else 0
            n_pts_h = cal_entry.get('n_points', '—')
            if isinstance(n_pts_h, dict):
                n_pts_h = n_pts_h.get('column', 0) or n_pts_h.get('bp', 0) or '—'
            src = cal_entry.get('source', {}).get('type', '')
            hist_rows.append([
                cal_entry.get('id', '—')[-18:],
                scope_label,
                str(cal_entry.get('valid_from', '—'))[:10],
                (str(cal_entry.get('valid_to', 'Vigent'))[:10]
                 if cal_entry.get('valid_to') else 'Vigent'),
                f"{rf_col_val:.0f}" if rf_col_val else "—",
                f"{rf_bp_val:.0f}" if rf_bp_val else "—",
                f"{r2_h:.4f}" if r2_h else "—",
                str(n_pts_h),
                src[:12],
            ])

        if hist_rows:
            hist_data = [hist_header] + hist_rows
            tbl4 = ax_hist.table(cellText=hist_data, loc='upper center',
                                  cellLoc='center',
                                  colWidths=[0.16, 0.08, 0.10, 0.10, 0.09,
                                             0.09, 0.10, 0.08, 0.10])
            tbl4.auto_set_font_size(False)
            tbl4.set_fontsize(7)
            tbl4.scale(1.0, 1.5)
            for j in range(9):
                tbl4[(0, j)].set_facecolor(COLORS["primary"])
                tbl4[(0, j)].set_text_props(color='white', fontweight='bold')
            for i, cal_entry in enumerate(all_calibrations[:20], 1):
                if cal_entry.get('is_active'):
                    for j in range(9):
                        tbl4[(i, j)].set_facecolor('#d4edda')

        draw_footer(fig, INSTITUTION_LINE)
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # =================================================================
        # PÀGINES EXTRA: Cromatogrames KHP (ambdós senyals)
        # =================================================================
        all_pngs = []
        for sd in signals_data:
            pngs = _find_khp_chromatogram_pngs(sd['cal'], sd['reg'])
            all_pngs.extend(pngs)
        # Dedup preserving order
        seen = set()
        unique_pngs = []
        for p in all_pngs:
            if p not in seen:
                seen.add(p)
                unique_pngs.append(p)

        if unique_pngs:
            plots_per_page = 6
            n_chrom_pages = (len(unique_pngs) + plots_per_page - 1) // plots_per_page
            for page_idx in range(n_chrom_pages):
                fig = plt.figure(figsize=(11.69, 8.27))
                fig.patch.set_facecolor('white')
                start = page_idx * plots_per_page
                end = min(start + plots_per_page, len(unique_pngs))
                page_pngs = unique_pngs[start:end]
                fig.text(0.5, 0.97,
                         f"CROMATOGRAMES KHP — Pàg. {page_idx + 1}/{n_chrom_pages}",
                         ha='center', va='top', fontsize=12, fontweight='bold',
                         color='#2C3E50')
                gs_chr = GridSpec(3, 2, figure=fig, left=0.04, right=0.96,
                                  top=0.93, bottom=0.04, hspace=0.3, wspace=0.15)
                for plot_idx, png_path in enumerate(page_pngs):
                    row = plot_idx // 2
                    col_idx = plot_idx % 2
                    ax_chr = fig.add_subplot(gs_chr[row, col_idx])
                    try:
                        img = plt.imread(png_path)
                        ax_chr.imshow(img)
                        ax_chr.axis('off')
                        fname = os.path.basename(png_path).replace('.png', '')
                        ax_chr.set_title(fname, fontsize=7, pad=2)
                    except Exception as e_img:
                        ax_chr.text(0.5, 0.5, f"Error: {e_img}", ha='center',
                                    va='center', fontsize=8, color='#E74C3C')
                        ax_chr.axis('off')
                for plot_idx in range(len(page_pngs), plots_per_page):
                    row = plot_idx // 2
                    col_idx = plot_idx % 2
                    ax_empty = fig.add_subplot(gs_chr[row, col_idx])
                    ax_empty.axis('off')
                pdf.savefig(fig, dpi=120)
                plt.close(fig)

    import logging
    logging.getLogger(__name__).info(f"Informe calibració dual generat: {pdf_path}")
    return pdf_path


def _find_khp_chromatogram_pngs(calibration, reg):
    """Busca PNGs de cromatogrames KHP associats a la calibració."""
    import glob as _glob

    # 1. Buscar a regression_data.chromatogram_plots_dir
    plots_dir = reg.get('chromatogram_plots_dir')
    if plots_dir and os.path.isdir(plots_dir):
        pngs = sorted(_glob.glob(os.path.join(plots_dir, "khp_*.png")))
        if pngs:
            return pngs

    # 2. Buscar a source.seq_references (escanejar CHECK/data/khp_plots/)
    seq_refs = calibration.get('source', {}).get('seq_references', [])
    if seq_refs:
        from hpsec_config import get_config
        try:
            config = get_config()
            data_folder = config.get('general', 'data_folder', default='')
        except Exception:
            data_folder = ''

        if data_folder:
            for seq_name in seq_refs:
                seq_dir = os.path.join(data_folder, seq_name)
                chrom_dir = os.path.join(seq_dir, "CHECK", "data", "khp_plots")
                if os.path.isdir(chrom_dir):
                    pngs = sorted(_glob.glob(os.path.join(chrom_dir, "khp_*.png")))
                    if pngs:
                        return pngs

    return []


# =============================================================================
# KHP CHROMATOGRAM PLOTS — PNG per rèplica
# =============================================================================

def save_khp_chromatogram_plot(rep_data, seq_name, conc_ppm, replica_num,
                               signal_name, mode, output_dir):
    """
    Guarda un cromatograma KHP com a PNG (backend Agg, sense GUI).

    Args:
        rep_data: dict amb t_doc, y_doc, baseline, peak_info, t_dad, y_dad_254
        seq_name: Nom de la seqüència
        conc_ppm: Concentració del KHP
        replica_num: Número de rèplica (1, 2, ...)
        signal_name: 'direct' o 'uib'
        mode: 'COLUMN' o 'BP'
        output_dir: Carpeta de sortida

    Returns:
        str: path del PNG generat, o None si error
    """
    import logging
    _log = logging.getLogger(__name__)

    t_doc = rep_data.get('t_doc')
    if t_doc is None:
        t_doc = rep_data.get('t')
    y_doc = rep_data.get('y_doc')
    if y_doc is None:
        y_doc = rep_data.get('y')
    if t_doc is None or y_doc is None:
        return None

    t_doc = np.asarray(t_doc)
    y_doc = np.asarray(y_doc)
    if len(t_doc) < 5:
        return None

    os.makedirs(output_dir, exist_ok=True)

    fname = f"khp_{conc_ppm:g}ppm_R{replica_num}_{signal_name}.png"
    png_path = os.path.join(output_dir, fname)

    try:
        import matplotlib
        matplotlib.use('Agg')
        fig, ax = plt.subplots(figsize=(5, 3), dpi=120)
        fig.set_facecolor('white')

        # DOC signal
        ax.plot(t_doc, y_doc, color='#2196F3', linewidth=1.0,
                label=f'{signal_name.upper()} DOC')

        # Repaired signal
        y_repaired = rep_data.get('y_doc_repaired')
        if y_repaired is not None:
            y_repaired = np.asarray(y_repaired)
            ax.plot(t_doc, y_repaired, color='#E74C3C', linewidth=0.8,
                    linestyle='--', label='Reparat', alpha=0.7)

        # Baseline
        baseline = rep_data.get('baseline')
        if baseline is not None:
            if isinstance(baseline, (int, float)):
                ax.axhline(baseline, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)
            else:
                baseline = np.asarray(baseline)
                if len(baseline) == len(t_doc):
                    ax.plot(t_doc, baseline, color='gray', linewidth=0.5,
                            linestyle=':', alpha=0.5, label='Baseline')

        # Integration limits + shaded area
        peak_info = rep_data.get('peak_info', {})
        t_start = peak_info.get('t_start')
        t_end = peak_info.get('t_end')
        if t_start is not None and t_end is not None:
            mask = (t_doc >= t_start) & (t_doc <= t_end)
            if np.any(mask):
                y_fill = y_repaired[mask] if y_repaired is not None else y_doc[mask]
                ax.fill_between(t_doc[mask], 0, y_fill, color='#2196F3', alpha=0.15)
            ax.axvline(t_start, color='gray', linewidth=0.5, linestyle=':', alpha=0.6)
            ax.axvline(t_end, color='gray', linewidth=0.5, linestyle=':', alpha=0.6)

        # Area annotation
        area = rep_data.get('area', 0)
        if area > 0:
            ax.annotate(f"A={area:.1f}", xy=(0.98, 0.92), xycoords='axes fraction',
                        ha='right', fontsize=8, color='#2C3E50',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='#EBF5FB', alpha=0.8))

        # 254nm secondary axis
        t_dad = rep_data.get('t_dad')
        y_254 = rep_data.get('y_dad_254')
        if t_dad is not None and y_254 is not None:
            t_dad = np.asarray(t_dad)
            y_254 = np.asarray(y_254)
            ax2 = ax.twinx()
            ax2.plot(t_dad, y_254, color='#9B59B6', linewidth=0.6, alpha=0.5, label='254nm')
            ax2.set_ylabel('254nm', color='#9B59B6', fontsize=7)
            ax2.tick_params(axis='y', labelcolor='#9B59B6', labelsize=6)

        ax.set_title(f"{seq_name} — {conc_ppm:g} ppm R{replica_num} ({signal_name.upper()}, {mode})",
                     fontsize=8, fontweight='bold')
        ax.set_xlabel('Temps (min)', fontsize=7)
        ax.set_ylabel('Senyal DOC', fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6, loc='upper left')

        fig.tight_layout()
        fig.savefig(png_path, dpi=120, bbox_inches='tight')
        plt.close(fig)

        return png_path

    except Exception as e:
        _log.warning(f"Error guardant cromatograma KHP: {e}")
        plt.close('all')
        return None


def save_all_khp_chromatograms(calibration_result, seq_path):
    """
    Guarda cromatogrames PNG per totes les rèpliques KHP d'un resultat de calibració.

    Args:
        calibration_result: dict retornat per calibrate_from_import()
        seq_path: Path de la seqüència

    Returns:
        list[str]: paths dels PNGs generats
    """
    import logging
    _log = logging.getLogger(__name__)

    output_dir = os.path.join(seq_path, "CHECK", "data", "khp_plots")
    seq_name = os.path.basename(seq_path)
    method = calibration_result.get('method', 'COLUMN')
    saved_paths = []

    # Iterar sobre entrades de calibració (directe i UIB)
    for signal_key in ['calibrations_direct', 'calibrations_uib']:
        signal_name = 'direct' if 'direct' in signal_key else 'uib'
        entries = calibration_result.get(signal_key, [])

        for entry in entries:
            conc = entry.get('conc_ppm', 0)
            replicas = entry.get('replicas', [])

            for r_idx, rep in enumerate(replicas):
                path = save_khp_chromatogram_plot(
                    rep_data=rep,
                    seq_name=seq_name,
                    conc_ppm=conc,
                    replica_num=r_idx + 1,
                    signal_name=signal_name,
                    mode=method,
                    output_dir=output_dir,
                )
                if path:
                    saved_paths.append(path)

    if saved_paths:
        _log.info(f"Guardats {len(saved_paths)} cromatogrames KHP a {output_dir}")

    return saved_paths


# =============================================================================
# INFORME QA/QC CALIBRACIÓ PER-SEQ (PDF)
# =============================================================================

def _khp_load_result(seq_path):
    """Carrega calibration_result.json de la SEQ."""
    from pathlib import Path
    import json as _json
    p = Path(seq_path) / "CHECK" / "data" / "calibration_result.json"
    if not p.exists():
        return None
    with open(p, 'r', encoding='utf-8') as f:
        return _json.load(f)


def _khp_load_reference(seq_path):
    """Carrega Calibration_Reference.json des de REGISTRY/."""
    from pathlib import Path
    import json as _json
    registry = Path(seq_path).parent / "REGISTRY"
    p = registry / "Calibration_Reference.json"
    if not p.exists():
        return None
    with open(p, 'r', encoding='utf-8') as f:
        return _json.load(f)


def _khp_load_khp_history(seq_path):
    """Carrega KHP_History.json des de REGISTRY/."""
    from pathlib import Path
    import json as _json
    registry = Path(seq_path).parent / "REGISTRY"
    p = registry / "KHP_History.json"
    if not p.exists():
        return None
    with open(p, 'r', encoding='utf-8') as f:
        return _json.load(f)


def _khp_load_graphs(seq_path):
    """Carrega PNGs existents de grafics de calibracio."""
    from pathlib import Path
    graphs_path = Path(seq_path) / "CHECK" / "Graphs"
    seq_name = Path(seq_path).name
    graphs = {}
    for key, pattern in [('replicas', f"calibration_replicas_{seq_name}.png"),
                          ('history', f"calibration_history_{seq_name}.png")]:
        f = graphs_path / pattern
        if f.exists():
            graphs[key] = str(f)
    return graphs


def _khp_get_active_ref(cal_ref):
    """Retorna la calibracio activa."""
    if not cal_ref:
        return None
    active_id = cal_ref.get("active_calibration_id")
    for c in cal_ref.get("calibrations", []):
        if c.get("id") == active_id or c.get("is_active"):
            return c
    cals = cal_ref.get("calibrations", [])
    return cals[0] if cals else None


def _khp_fmt_bigaussian(bg):
    """Formata info bigaussiana."""
    if not bg or not isinstance(bg, dict):
        return "N/D"
    sigma = bg.get("sigma") or bg.get("sigma1")
    r2 = bg.get("r2") or bg.get("r_squared")
    if sigma is None and r2 is None:
        return "N/D"
    parts = []
    if sigma is not None:
        parts.append(f"\u03c3={sigma:.3f}")
    if r2 is not None:
        parts.append(f"R\u00b2={r2:.3f}")
    return ", ".join(parts)


def _khp_build_condition_label(cal):
    conc = cal.get('conc_ppm', 0)
    vol = cal.get('volume_uL', 0)
    return f"KHP {conc:.0f}ppm @ {vol:.0f}\u00b5L"


def _khp_filter_history_outliers(calibrations, mode):
    """Filtra historic per mode, exclou outliers (flag + IQR)."""
    candidates = []
    n_flagged = 0
    for h in calibrations:
        if h.get('mode') != mode:
            continue
        if h.get('is_outlier'):
            n_flagged += 1
            continue
        if h.get('area', 0) > 0:
            candidates.append(h)

    if len(candidates) < 4:
        return candidates, n_flagged

    vol_groups = {}
    for h in candidates:
        vol_groups.setdefault(h.get('volume_uL', 0), []).append(h)

    clean = []
    n_iqr = 0
    for vol, group in vol_groups.items():
        if len(group) < 4:
            clean.extend(group)
            continue
        current = list(group)
        while True:
            areas = [h.get('area', 0) for h in current]
            q1 = np.percentile(areas, 25)
            q3 = np.percentile(areas, 75)
            iqr = q3 - q1
            filtered = [h for h in current
                        if q1 - 1.5 * iqr <= h.get('area', 0) <= q3 + 1.5 * iqr]
            removed = len(current) - len(filtered)
            n_iqr += removed
            current = filtered
            if removed == 0 or len(current) < 4:
                break
        clean.extend(current)

    return clean, n_flagged + n_iqr


def _khp_draw_page1(pdf, data, cal_ref, seq_name):
    """Pagina 1: resum general + metriques per condicio."""
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor('white')

    logo_path = get_logo_path()
    if logo_path:
        try:
            logo = plt.imread(logo_path)
            ax_logo = fig.add_axes([0.05, 0.88, 0.25, 0.08])
            ax_logo.imshow(logo)
            ax_logo.axis('off')
        except Exception:
            pass

    fig.text(0.95, 0.94, "HPSEC Suite", ha='right', va='top',
             fontsize=10, color=COLORS["text_secondary"])
    fig.text(0.5, 0.85, "INFORME QA/QC", ha='center', va='top',
             fontsize=18, fontweight='bold', color=COLORS["primary"])
    fig.text(0.5, 0.81, f"Sequencia {seq_name}", ha='center', va='top',
             fontsize=12, color=COLORS["text"])
    date_str = datetime.now().strftime("%d/%m/%Y %H:%M")
    fig.text(0.5, 0.775, f"Generat: {date_str}", ha='center', va='top',
             fontsize=9, color=COLORS["text_secondary"])
    fig.add_artist(plt.Line2D([0.1, 0.9], [0.75, 0.75],
                              color=COLORS["primary"], linewidth=2,
                              transform=fig.transFigure))

    calibrations = data.get('calibrations', [])
    active_cals = [c for c in calibrations if c.get('is_active', False)]
    if not active_cals:
        active_cals = calibrations[:2]

    ref = _khp_get_active_ref(cal_ref)
    ref_rf = ref_r2 = ref_n = "-"
    if ref:
        rf_obj = ref.get("rf_mass_cal", {})
        if isinstance(rf_obj, dict):
            ref_rf = str(rf_obj.get("direct", {}).get("bp",
                         rf_obj.get("direct", {}).get("column", "-")))
        else:
            ref_rf = str(rf_obj)
        r2_val = ref.get("r2")
        if isinstance(r2_val, dict):
            r2_val = next((v for v in r2_val.values() if v), None)
        ref_r2 = f"{r2_val:.4f}" if r2_val else "-"
        ref_n = str(ref.get("n_points", "-"))

    mode = active_cals[0].get('mode', '-') if active_cals else '-'
    conditions_str = ", ".join(
        f"{c.get('volume_uL', 0):.0f}\u00b5L @ {c.get('conc_ppm', 0):.0f}ppm"
        for c in active_cals)
    statuses = [c.get('status', 'OK') for c in active_cals]
    global_status = "OK" if all(s == "OK" for s in statuses) else "WARNING"
    seq_date = active_cals[0].get('seq_date', '-') if active_cals else '-'
    if isinstance(seq_date, str) and len(seq_date) > 10:
        seq_date = seq_date[:10]

    summary_data = [
        ["Parametre", "Valor", "Parametre", "Valor"],
        ["Mode", f"{mode} ({conditions_str})", "Estat global", global_status],
        ["Calibracio ref.", f"RF={ref_rf}, R2={ref_r2} (n={ref_n})",
         "N cond. actives", str(len(active_cals))],
        ["Data SEQ", str(seq_date), "Data processament", date_str],
    ]
    ax_sum = fig.add_axes([0.08, 0.64, 0.84, 0.09])
    draw_minimal_table(ax_sum, summary_data,
                       col_widths=[0.18, 0.35, 0.18, 0.29], font_size=9)

    y_pos = 0.60
    for cal in active_cals:
        if y_pos < 0.08:
            break
        cond_label = _khp_build_condition_label(cal)
        status = cal.get('status', 'OK')
        status_color = COLORS["accent"] if status == "OK" else COLORS["warning"]
        fig.text(0.08, y_pos, cond_label,
                 fontsize=11, fontweight='bold', color=COLORS["text"])
        fig.text(0.50, y_pos, f"[{cal.get('condition_key', '')}]",
                 fontsize=9, color=COLORS["text_secondary"])
        bbox_props = dict(boxstyle="round,pad=0.3", facecolor=status_color,
                          edgecolor='none', alpha=0.9)
        fig.text(0.90, y_pos, status, ha='center', va='center',
                 fontsize=9, fontweight='bold', color='white', bbox=bbox_props)
        y_pos -= 0.025

        area_d = cal.get('area', 0)
        bg_doc = _khp_fmt_bigaussian(cal.get('bigaussian_doc'))
        area_u = cal.get('area_u', 0)
        bg_uib = _khp_fmt_bigaussian(cal.get('bigaussian_uib'))
        has_uib = area_u and area_u > 0

        metrics_data = [
            ["Metrica", "Direct", "UIB"] if has_uib else ["Metrica", "Valor"],
        ]
        def _row(label, val_d, val_u=None):
            return [label, val_d, val_u if val_u else "-"] if has_uib else [label, val_d]

        metrics_data.append(_row("Area", format_value(area_d, ".1f"),
                                 format_value(area_u, ".1f") if has_uib else None))
        metrics_data.append(_row("RF mass", format_value(cal.get('rf_mass', 0), ".1f"),
                                 format_value(cal.get('rf_mass_u', 0), ".1f") if has_uib else None))
        metrics_data.append(_row("RSD", format_value(cal.get('rsd', 0), ".1f", "%")))
        metrics_data.append(_row("SNR", format_value(cal.get('snr', 0), ".0f"),
                                 format_value(cal.get('snr_u', 0), ".0f") if has_uib else None))
        metrics_data.append(_row("Simetria", format_value(cal.get('symmetry', 0), ".2f")))
        metrics_data.append(_row("FWHM", format_value(cal.get('fwhm_doc', 0), ".2f", " min")))
        metrics_data.append(_row("Gaussiana", bg_doc, bg_uib if has_uib else None))
        bg_254 = _khp_fmt_bigaussian(cal.get('bigaussian_254'))
        if bg_254 != "N/D":
            metrics_data.append(_row("Gauss. 254", bg_254))

        table_h = 0.018 * (len(metrics_data) + 0.5)
        cw = [0.30, 0.35, 0.35] if has_uib else [0.35, 0.65]
        ax_met = fig.add_axes([0.08, y_pos - table_h, 0.50, table_h])
        draw_minimal_table(ax_met, metrics_data, col_widths=cw, font_size=8)

        info_x, info_y = 0.62, y_pos - 0.005
        fig.text(info_x, info_y, "Time shifts:", fontsize=9,
                 fontweight='bold', color=COLORS["text"])
        info_y -= 0.018
        fig.text(info_x, info_y, f"t DOC = {format_value(cal.get('t_retention', 0), '.2f', ' min')}",
                 fontsize=8, color=COLORS["text"])
        info_y -= 0.015
        fig.text(info_x, info_y, f"t 254 = {format_value(cal.get('t_dad_max', 0), '.3f', ' min')}",
                 fontsize=8, color=COLORS["text"])
        info_y -= 0.015
        fig.text(info_x, info_y, f"\u0394t = {format_value(cal.get('shift_sec', 0), '.1f', 's')}",
                 fontsize=8, color=COLORS["primary"])

        val_det = cal.get('validation_details', {})
        hist_comp = val_det.get('historical_comparison', {})
        hist_stats = hist_comp.get('historical_stats')
        info_y -= 0.022
        fig.text(info_x, info_y, "Historic:", fontsize=9,
                 fontweight='bold', color=COLORS["text"])
        info_y -= 0.018
        if hist_stats:
            mean_a = hist_stats.get('mean_area', 0)
            std_a = hist_stats.get('std_area', 0)
            dev_pct = hist_comp.get('area_deviation_pct', 0)
            n_hist = hist_stats.get('n_calibrations', 0)
            fig.text(info_x, info_y, f"Mitjana: {mean_a:.1f} +/- {std_a:.1f} (n={n_hist})",
                     fontsize=8, color=COLORS["text"])
            info_y -= 0.015
            dev_color = COLORS["accent"] if abs(dev_pct) < 15 else COLORS["warning"]
            fig.text(info_x, info_y, f"Actual: {area_d:.1f}  ({dev_pct:+.1f}%)",
                     fontsize=8, fontweight='bold', color=dev_color)
        else:
            fig.text(info_x, info_y, f"Estat: {hist_comp.get('status', '-')}",
                     fontsize=8, color=COLORS["text_secondary"])

        all_issues = (cal.get('quality_issues', []) + cal.get('calibration_warnings', [])
                      + hist_comp.get('warnings', []))
        if all_issues:
            info_y -= 0.020
            for issue in all_issues[:2]:
                fig.text(info_x, info_y, f"[!] {issue[:55]}",
                         fontsize=7, color=COLORS["warning"])
                info_y -= 0.013

        y_pos -= table_h + 0.03
        fig.add_artist(plt.Line2D([0.08, 0.92], [y_pos + 0.01, y_pos + 0.01],
                                  color=COLORS["border"], linewidth=0.5,
                                  transform=fig.transFigure))
        y_pos -= 0.01

    draw_report_footer(fig, 1)
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def _khp_draw_page2(pdf, data, cal_ref, khp_history, seq_name):
    """Pagina 2: grafic Area vs Mass + taula resum punts."""
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor('white')

    fig.text(0.5, 0.96, "RECTA DE CALIBRACIO GLOBAL",
             ha='center', va='top', fontsize=14, fontweight='bold',
             color=COLORS["primary"])
    fig.text(0.5, 0.93, f"{seq_name} - Area vs Mass (ug)",
             ha='center', va='top', fontsize=10, color=COLORS["text_secondary"])
    fig.add_artist(plt.Line2D([0.1, 0.9], [0.91, 0.91],
                              color=COLORS["primary"], linewidth=1,
                              transform=fig.transFigure))

    ref = _khp_get_active_ref(cal_ref)
    rf_mass_cal, r2_ref, tolerance_pct, warning_pct = 682, 0.99, 20, 15
    if ref:
        rf_obj = ref.get("rf_mass_cal", {})
        if isinstance(rf_obj, dict):
            rf_mass_cal = rf_obj.get("direct", {}).get("bp",
                          rf_obj.get("direct", {}).get("column", 682))
        elif isinstance(rf_obj, (int, float)):
            rf_mass_cal = rf_obj
        r2_ref = ref.get("r2", 0.99) or 0.99
        val = ref.get("validation", {})
        tolerance_pct = val.get("tolerance_pct", 20)
        warning_pct = val.get("warning_pct", 15)

    calibrations = data.get('calibrations', [])
    active_cals = [c for c in calibrations if c.get('is_active', False)]
    if not active_cals:
        active_cals = calibrations[:2]
    current_mode = active_cals[0].get('mode', 'BP') if active_cals else 'BP'

    hist_masses, hist_areas = [], []
    n_hist_excluded = 0
    if khp_history:
        clean_hist, n_hist_excluded = _khp_filter_history_outliers(
            khp_history.get('calibrations', []), current_mode)
        for h in clean_hist:
            conc, vol, area = h.get('conc_ppm', 0), h.get('volume_uL', 0), h.get('area', 0)
            if conc > 0 and vol > 0 and area > 0:
                hist_masses.append(conc * vol / 1000.0)
                hist_areas.append(area)

    curr_masses, curr_areas, curr_labels, curr_ok = [], [], [], []
    for cal in active_cals:
        conc, vol, area = cal.get('conc_ppm', 0), cal.get('volume_uL', 0), cal.get('area', 0)
        if conc > 0 and vol > 0 and area > 0:
            mass = conc * vol / 1000.0
            expected = rf_mass_cal * mass
            dev = abs(area - expected) / expected * 100 if expected > 0 else 0
            curr_masses.append(mass)
            curr_areas.append(area)
            curr_labels.append(cal.get('condition_key', ''))
            curr_ok.append(dev <= tolerance_pct)

    ax = fig.add_axes([0.10, 0.30, 0.80, 0.58])
    all_masses = hist_masses + curr_masses
    x_max = max(all_masses) * 1.15 if all_masses else 0.3
    x_line = np.linspace(0, x_max, 200)
    y_line = rf_mass_cal * x_line

    ax.plot(x_line, y_line, color=COLORS["primary"], linewidth=1.5,
            label=f"Recta: Area = {rf_mass_cal} x Mass")
    ax.fill_between(x_line, y_line * (1 - tolerance_pct / 100),
                    y_line * (1 + tolerance_pct / 100),
                    alpha=0.08, color=COLORS["danger"], label=f"+/-{tolerance_pct}% tolerancia")
    ax.fill_between(x_line, y_line * (1 - warning_pct / 100),
                    y_line * (1 + warning_pct / 100),
                    alpha=0.10, color=COLORS["warning"], label=f"+/-{warning_pct}% warning")

    if hist_masses:
        lbl = f"Historic ({len(hist_masses)} pts)"
        if n_hist_excluded > 0:
            lbl += f", {n_hist_excluded} exclosos"
        ax.scatter(hist_masses, hist_areas, c='#BBBBBB', s=25, alpha=0.6,
                   edgecolors='#999999', linewidths=0.5, zorder=3, label=lbl)

    for i, (m, a, lbl, ok) in enumerate(zip(curr_masses, curr_areas, curr_labels, curr_ok)):
        color = COLORS["accent"] if ok else COLORS["danger"]
        ax.scatter([m], [a], c=color, s=100, edgecolors='white',
                   linewidths=1.5, zorder=5, marker='o' if ok else 'X',
                   label=f"Actual: {lbl}" if i < 3 else None)
        ax.annotate(lbl, (m, a), textcoords="offset points",
                    xytext=(8, 8), fontsize=7, color=color, fontweight='bold')

    ax.text(0.03, 0.97, f"Area = {rf_mass_cal} x Mass (R2 = {r2_ref:.4f})",
            transform=ax.transAxes, fontsize=8, fontweight='bold', color=COLORS["primary"],
            va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=COLORS["border"], alpha=0.9))
    ax.set_xlabel("Mass (ug)", fontsize=9)
    ax.set_ylabel("Area (mAU*min)", fontsize=9)
    ax.set_xlim(0, x_max)
    y_all = hist_areas + curr_areas + [0]
    if y_all:
        ax.set_ylim(0, max(y_all) * 1.2)
    ax.grid(True, alpha=0.3, linewidth=0.3)
    ax.legend(loc='lower right', fontsize=7, frameon=True,
              framealpha=0.9, edgecolor=COLORS["border"])

    # Taula resum punts
    table_data = [["Condicio", "Vol (uL)", "Conc (ppm)", "Mass (ug)",
                   "Area", "RF mass", "Esperat", "Dev %", "Estat"]]
    for cal in active_cals:
        conc, vol, area = cal.get('conc_ppm', 0), cal.get('volume_uL', 0), cal.get('area', 0)
        rf_m = cal.get('rf_mass', 0)
        mass = conc * vol / 1000.0 if conc > 0 and vol > 0 else 0
        expected = rf_mass_cal * mass
        dev = (area - expected) / expected * 100 if expected > 0 else 0
        status_str = "WARN" if warning_pct < abs(dev) <= tolerance_pct else (
            "FAIL" if abs(dev) > tolerance_pct else "OK")
        table_data.append([cal.get('condition_key', '-'),
            format_value(vol, ".0f"), format_value(conc, ".1f"),
            format_value(mass, ".3f"), format_value(area, ".1f"),
            format_value(rf_m, ".1f"), format_value(expected, ".1f"),
            format_value(dev, "+.1f", "%"), status_str])

    table_height = 0.015 + 0.016 * len(table_data)
    ax_table = fig.add_axes([0.08, 0.05, 0.84, min(table_height, 0.18)])
    draw_minimal_table(ax_table, table_data,
                       col_widths=[0.14, 0.08, 0.09, 0.09, 0.10,
                                   0.10, 0.10, 0.10, 0.10], font_size=8)

    draw_report_footer(fig, 2)
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def _khp_draw_page3(pdf, seq_path, data, khp_history, seq_name):
    """Pagina 3: PNGs existents + historic generat."""
    graphs = _khp_load_graphs(seq_path)
    has_replicas = 'replicas' in graphs

    calibrations = data.get('calibrations', [])
    active_cals = [c for c in calibrations if c.get('is_active', False)]
    current_mode = active_cals[0].get('mode', 'BP') if active_cals else 'BP'

    mode_history, n_outliers = [], 0
    if khp_history:
        mode_history, n_outliers = _khp_filter_history_outliers(
            khp_history.get('calibrations', []), current_mode)

    if not has_replicas and not mode_history:
        return

    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor('white')
    fig.text(0.5, 0.96, "GRAFICS DE CALIBRACIO",
             ha='center', va='top', fontsize=14, fontweight='bold',
             color=COLORS["primary"])
    fig.text(0.5, 0.93, seq_name, ha='center', va='top',
             fontsize=10, color=COLORS["text_secondary"])
    fig.add_artist(plt.Line2D([0.1, 0.9], [0.91, 0.91],
                              color=COLORS["primary"], linewidth=1,
                              transform=fig.transFigure))

    if has_replicas:
        try:
            img = plt.imread(graphs['replicas'])
            ax = fig.add_axes([0.05, 0.48, 0.90, 0.40])
            ax.imshow(img)
            ax.axis('off')
            fig.text(0.5, 0.89, "Repliques KHP (DOC + DAD 254nm)",
                     ha='center', fontsize=10, fontweight='bold', color=COLORS["text"])
        except Exception:
            pass

    if mode_history:
        display_cals = mode_history[-15:]
        y_bottom = 0.05 if has_replicas else 0.30
        h = 0.38 if has_replicas else 0.55
        ax = fig.add_axes([0.10, y_bottom, 0.80, h])

        seq_labels, areas, colors_bars = [], [], []
        for cal in display_cals:
            name = cal.get('seq_name', 'N/A').replace('_SEQ', '').replace('_BP', '')
            seq_labels.append(name)
            areas.append(cal.get('area', 0))
            colors_bars.append(COLORS["accent"] if cal.get('seq_name') == seq_name
                               else COLORS["primary"])

        x = range(len(seq_labels))
        ax.bar(x, areas, color=colors_bars, edgecolor='white', linewidth=0.5)
        if areas:
            mean_area = np.mean(areas)
            std_area = np.std(areas) if len(areas) > 1 else 0
            lbl = f'Mitjana: {mean_area:.0f}'
            if n_outliers > 0:
                lbl += f' ({n_outliers} outliers exclosos)'
            ax.axhline(mean_area, color=COLORS["accent"], linestyle='--',
                       linewidth=1.5, label=lbl)
            if std_area > 0:
                ax.axhspan(mean_area - std_area, mean_area + std_area,
                           alpha=0.1, color=COLORS["accent"])
            ax.legend(loc='upper right', fontsize=7, frameon=False)

        ax.set_xticks(list(x))
        ax.set_xticklabels(seq_labels, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel("Area", fontsize=9)
        ax.set_title(f"Evolucio de l'area KHP ({current_mode})",
                     fontsize=11, fontweight='bold', color=COLORS["text"], pad=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    draw_report_footer(fig, 3)
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def generate_khp_report(seq_path, output_path=None):
    """
    Genera PDF QA/QC de calibracio per-SEQ (3 pagines).

    Diferent de generate_calibration_report() que genera l'informe global
    des de Calibration_Reference.json.

    Args:
        seq_path: Ruta a la carpeta SEQ
        output_path: Ruta de sortida (default: CHECK/)

    Returns:
        Path del PDF generat o None si error
    """
    from pathlib import Path

    data = _khp_load_result(seq_path)
    if not data:
        print(f"No s'han trobat dades de calibracio a {seq_path}")
        return None

    seq_name = data.get('seq_name', Path(seq_path).name)
    calibrations = data.get('calibrations', [])
    if not calibrations:
        print(f"No hi ha calibracions a {seq_path}")
        return None

    cal_ref = _khp_load_reference(seq_path)
    khp_history = _khp_load_khp_history(seq_path)

    if output_path is None:
        output_path = Path(seq_path) / "CHECK"
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    pdf_path = output_path / f"REPORT_Calibration_{seq_name}.pdf"
    if pdf_path.exists():
        try:
            with open(pdf_path, 'a'):
                pass
        except PermissionError:
            timestamp = datetime.now().strftime("%H%M%S")
            pdf_path = output_path / f"REPORT_Calibration_{seq_name}_{timestamp}.pdf"

    print(f"Generant {pdf_path}...")

    with PdfPages(pdf_path) as pdf:
        _khp_draw_page1(pdf, data, cal_ref, seq_name)
        _khp_draw_page2(pdf, data, cal_ref, khp_history, seq_name)
        _khp_draw_page3(pdf, seq_path, data, khp_history, seq_name)

    print(f"  [OK] PDF generat: {pdf_path}")
    return str(pdf_path)


# =============================================================================
# INFORME D'IMPORTACIÓ (PDF)
# =============================================================================

def _imp_load_manifest(seq_path):
    """Carrega dades d'importacio des del JSON."""
    from pathlib import Path
    json_path = Path(seq_path) / "CHECK" / "data" / "import_manifest.json"
    if not json_path.exists():
        return None
    import json as _json
    with open(json_path, 'r', encoding='utf-8') as f:
        return _json.load(f)


def _imp_reclassify_samples(samples):
    """Reclassifica tipus de mostres segons regles actuals de config."""
    try:
        from hpsec_import import is_khp, is_blank_injection, is_control_injection, is_reference_standard
        from hpsec_config import get_config
        config = get_config()
    except ImportError:
        return

    for s in samples:
        name = s.get("original_name", s.get("name", ""))
        if not name:
            continue
        if is_khp(name):
            s["type"] = "KHP"
        elif is_reference_standard(name, config):
            s["type"] = "PR"
        elif is_blank_injection(name, config):
            s["type"] = "BLANK"
        elif is_control_injection(name, config):
            s["type"] = "CONTROL"
        elif "test" in name.lower():
            s["type"] = "TEST"
        else:
            s["type"] = "SAMPLE"


def _imp_compute_t0(samples):
    """Calcula t0: temps (min) des de connexió TOC fins primera injecció."""
    min_row_start = None
    time_per_point = None

    for s in samples:
        for r in s.get("replicas", []):
            d = r.get("direct", {})
            row_start = d.get("row_start")
            n_points = d.get("n_points", 0)
            t_max = d.get("t_max", 0) or 0

            if row_start is not None and n_points > 0 and t_max > 0:
                if min_row_start is None or row_start < min_row_start:
                    min_row_start = row_start
                    time_per_point = t_max / n_points

    if min_row_start is not None and time_per_point is not None and min_row_start > 0:
        return min_row_start * time_per_point
    return 0


def _imp_count_by_type(samples):
    """Compta mostres i injeccions per tipus."""
    counts = {
        "SAMPLE": {"samples": 0, "injections": 0},
        "KHP": {"samples": 0, "injections": 0},
        "PR": {"samples": 0, "injections": 0},
        "CONTROL": {"samples": 0, "injections": 0},
        "BLANK": {"samples": 0, "injections": 0},
    }
    for s in samples:
        t = s.get("type", "SAMPLE")
        if t not in counts:
            t = "SAMPLE"
        counts[t]["samples"] += 1
        counts[t]["injections"] += len(s.get("replicas", []))
    return counts


def _imp_draw_page_summary(pdf, manifest, seq_name):
    """Dibuixa pagina 1 amb resum d'importacio."""
    import statistics
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor('white')

    logo_path = get_logo_path()
    if logo_path:
        try:
            logo = plt.imread(logo_path)
            ax_logo = fig.add_axes([0.05, 0.88, 0.25, 0.08])
            ax_logo.imshow(logo)
            ax_logo.axis('off')
        except Exception:
            pass

    fig.text(0.95, 0.94, "HPSEC Suite", ha='right', va='top',
             fontsize=10, color=COLORS["text_secondary"])
    fig.text(0.5, 0.85, "INFORME D'IMPORTACIO", ha='center', va='top',
             fontsize=18, fontweight='bold', color=COLORS["primary"])
    fig.text(0.5, 0.81, f"Sequencia {seq_name}", ha='center', va='top',
             fontsize=12, color=COLORS["text"])

    date_str = datetime.now().strftime("%d/%m/%Y %H:%M")
    fig.text(0.5, 0.775, f"Generat: {date_str}", ha='center', va='top',
             fontsize=9, color=COLORS["text_secondary"])
    fig.add_artist(plt.Line2D([0.1, 0.9], [0.75, 0.75],
                              color=COLORS["primary"], linewidth=2,
                              transform=fig.transFigure))

    seq_info = manifest.get("sequence", {})
    summary = manifest.get("summary", {})
    master = manifest.get("master_file", {})
    samples = manifest.get("samples", [])

    data_mode = seq_info.get("data_mode", "DUAL")
    method = seq_info.get("method", "COLUMN")
    seq_date = seq_info.get("date", "-")
    uib_sens = seq_info.get("uib_sensitivity", "-")
    master_name = master.get("filename", "-")

    type_counts = _imp_count_by_type(samples)
    total_injections = sum(c["injections"] for c in type_counts.values())

    summary_data = [
        ["Parametre", "Valor", "Parametre", "Valor"],
        ["Mode DOC", data_mode, "Metode", method],
        ["Data SEQ", str(seq_date)[:10], "MasterFile", master_name[:28]],
        ["Sensibilitat UIB",
         str(uib_sens) if uib_sens and uib_sens != "None" else "-",
         "Injeccions totals", str(total_injections)],
    ]

    ax_summary = fig.add_axes([0.08, 0.63, 0.84, 0.10])
    draw_minimal_table(ax_summary, summary_data,
                       col_widths=[0.22, 0.28, 0.22, 0.28], font_size=9)

    # Desglossament per tipus
    type_table = [["Tipus", "Mostres", "Injeccions", "% del total"]]
    label_map = {
        "SAMPLE": "Mostres", "KHP": "Patrons QC/QA (KHP)",
        "PR": "Patrons de referencia", "CONTROL": "Controls", "BLANK": "Blancs",
    }
    for type_key in ["SAMPLE", "KHP", "PR", "CONTROL", "BLANK"]:
        c = type_counts[type_key]
        pct = f"{c['injections'] / total_injections * 100:.0f}%" if total_injections > 0 and c["injections"] > 0 else "0%"
        type_table.append([label_map[type_key], str(c["samples"]),
                           str(c["injections"]), pct])

    n_type_rows = len(type_table)
    type_height = 0.02 + 0.022 * n_type_rows
    ax_types = fig.add_axes([0.08, 0.59 - type_height, 0.84, type_height])
    draw_minimal_table(ax_types, type_table,
                       col_widths=[0.40, 0.18, 0.22, 0.20], font_size=9)

    # Completesa dades
    y_comp_top = 0.59 - type_height - 0.025
    rep_direct = summary.get("replicas_with_direct", 0)
    rep_uib = summary.get("replicas_with_uib", 0)
    rep_dad = summary.get("replicas_with_dad", 0)

    if rep_direct == 0 and total_injections > 0:
        for s in samples:
            for r in s.get("replicas", []):
                if (r.get("direct", {}) or {}).get("n_points", 0) > 0:
                    rep_direct += 1
                if (r.get("uib", {}) or {}).get("n_points", 0) > 0:
                    rep_uib += 1
                if (r.get("dad", {}) or {}).get("n_points", 0) > 0:
                    rep_dad += 1

    def pct_str(n, total):
        return f"{n}/{total} ({n / total * 100:.0f}%)" if total > 0 else "0%"

    completeness_data = [["Senyal", "Dades"],
                         ["DOC Direct", pct_str(rep_direct, total_injections)]]
    if data_mode in ["DUAL", "UIB"]:
        completeness_data.append(["DOC UIB", pct_str(rep_uib, total_injections)])
    completeness_data.append(["DAD", pct_str(rep_dad, total_injections)])

    n_comp_rows = len(completeness_data)
    comp_height = 0.02 + 0.022 * n_comp_rows
    ax_comp = fig.add_axes([0.08, y_comp_top - comp_height, 0.84, comp_height])
    draw_minimal_table(ax_comp, completeness_data,
                       col_widths=[0.40, 0.60], font_size=9)

    # Estadistiques injeccio
    y_stats_top = y_comp_top - comp_height - 0.025
    all_tmax = []
    n_timeouts_warn = 0
    n_timeouts_crit = 0
    for s in samples:
        for r in s.get("replicas", []):
            d = r.get("direct", {})
            inj = r.get("injection", {})
            t_max_val = d.get("t_max", 0) or 0
            line_num = inj.get("line_num", 999999)
            if t_max_val > 0:
                all_tmax.append((line_num, t_max_val))
            has_to = d.get("has_timeout", False)
            if has_to:
                sev = d.get("timeout_severity", "WARNING")
                if sev == "CRITICAL":
                    n_timeouts_crit += 1
                else:
                    n_timeouts_warn += 1

    all_tmax.sort(key=lambda x: x[0])
    tmax_values = [t for _, t in all_tmax]
    tmax_no_last = tmax_values[:-1] if len(tmax_values) > 1 else tmax_values
    median_tmax = statistics.median(tmax_no_last) if tmax_no_last else 0
    t0 = _imp_compute_t0(samples)

    timeout_parts = []
    if n_timeouts_crit > 0:
        timeout_parts.append(f"{n_timeouts_crit} CRITICAL")
    if n_timeouts_warn > 0:
        timeout_parts.append(f"{n_timeouts_warn} WARNING")
    timeout_str = ", ".join(timeout_parts) if timeout_parts else "0"

    stats_data = [
        ["Parametre", "Valor", "Parametre", "Valor"],
        ["Mediana t acq (min)", f"{median_tmax:.1f}" if median_tmax > 0 else "-",
         "t0 (min)", f"{t0:.1f}" if t0 > 0 else "-"],
        ["Timeouts", timeout_str, "", ""],
    ]

    stats_height = 0.02 + 0.022 * len(stats_data)
    ax_stats = fig.add_axes([0.08, y_stats_top - stats_height, 0.84, stats_height])
    draw_minimal_table(ax_stats, stats_data,
                       col_widths=[0.22, 0.28, 0.22, 0.28], font_size=9)

    y_pos = y_stats_top - stats_height - 0.03

    # Avisos
    fig.text(0.08, y_pos, "Avisos:",
             fontsize=10, fontweight='bold', color=COLORS["text"])
    y_pos -= 0.025
    has_any_warning = False

    orphan_uib = manifest.get("orphan_files", {}).get("uib", [])
    orphan_dad = manifest.get("orphan_files", {}).get("dad", [])
    if orphan_uib:
        has_any_warning = True
        fig.text(0.10, y_pos,
                 f"[!] {len(orphan_uib)} fitxers UIB sense assignar: {', '.join(orphan_uib[:4])}",
                 fontsize=8, color=COLORS["warning"])
        y_pos -= 0.018
    if orphan_dad:
        has_any_warning = True
        fig.text(0.10, y_pos,
                 f"[!] {len(orphan_dad)} fitxers DAD sense assignar: {', '.join(orphan_dad[:4])}",
                 fontsize=8, color=COLORS["warning"])
        y_pos -= 0.018

    internal_msgs = {"importat des de manifest existent", "importat des de manifest"}
    raw_warnings = manifest.get("warnings", [])
    for w in raw_warnings:
        clean_w = w.replace("\u26a0\ufe0f", "").replace("\u26a0", "").strip()
        if clean_w and clean_w.lower() not in internal_msgs:
            has_any_warning = True
            fig.text(0.10, y_pos, f"[!] {clean_w[:80]}",
                     fontsize=8, color=COLORS["warning"])
            y_pos -= 0.018

    if not has_any_warning:
        fig.text(0.10, y_pos, "Res a reportar.",
                 fontsize=9, color=COLORS["accent"])

    draw_report_footer(fig, 1)
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def _imp_build_sorted_injections(samples):
    """Construeix llista plana d'injeccions ordenada."""
    type_order = {"SAMPLE": 0, "KHP": 1, "PR": 1, "CONTROL": 2, "BLANK": 3}
    injections = []
    for sample in samples:
        sample_type = sample.get("type", "SAMPLE")
        for rep in sample.get("replicas", []):
            inj_info = rep.get("injection", {})
            line_num = inj_info.get("line_num", 999999)
            injections.append({
                "name": sample.get("original_name", sample.get("name", "")),
                "type": sample_type,
                "replica": rep.get("replica", "?"),
                "direct": rep.get("direct", {}),
                "uib": rep.get("uib", {}),
                "dad": rep.get("dad", {}),
                "injection": inj_info,
                "line_num": line_num,
                "type_order": type_order.get(sample_type, 0),
            })
    injections.sort(key=lambda x: (x["type_order"], x["line_num"]))
    return injections


def _imp_draw_detail_pages(pdf, manifest, page_counter):
    """Dibuixa pagines amb taula detallada de mostres (landscape)."""
    samples = manifest.get("samples", [])
    seq_info = manifest.get("sequence", {})
    data_mode = seq_info.get("data_mode", "DUAL")
    seq_name = seq_info.get("name", "")
    injections = _imp_build_sorted_injections(samples)
    if not injections:
        return page_counter

    ROWS_PER_PAGE = 35
    if data_mode == "DUAL":
        headers = ["#", "Mostra", "R", "Pts DOC", "Pts UIB", "Pts DAD",
                    "Fitxer UIB", "Fitxer DAD", "Vol", "t acq (min)"]
        col_widths = [0.04, 0.17, 0.03, 0.07, 0.07, 0.07, 0.17, 0.17, 0.05, 0.10]
    elif data_mode == "UIB":
        headers = ["#", "Mostra", "R", "Pts DOC", "Pts UIB",
                    "Fitxer UIB", "Vol", "t acq (min)"]
        col_widths = [0.04, 0.22, 0.04, 0.08, 0.08, 0.22, 0.06, 0.10]
    else:
        headers = ["#", "Mostra", "R", "Pts DOC", "Pts DAD",
                    "Fitxer DAD", "Vol", "t acq (min)"]
        col_widths = [0.04, 0.22, 0.04, 0.08, 0.08, 0.22, 0.06, 0.10]

    for page_start in range(0, len(injections), ROWS_PER_PAGE):
        page_counter += 1
        page_injections = injections[page_start:page_start + ROWS_PER_PAGE]

        fig = plt.figure(figsize=(11.69, 8.27))
        fig.patch.set_facecolor('white')
        fig.text(0.5, 0.96, "DETALL D'INJECCIONS", ha='center', va='top',
                 fontsize=14, fontweight='bold', color=COLORS["primary"])
        fig.text(0.5, 0.93, f"{seq_name} - Pagina {page_counter}",
                 ha='center', va='top', fontsize=10, color=COLORS["text_secondary"])
        fig.add_artist(plt.Line2D([0.05, 0.95], [0.91, 0.91],
                                  color=COLORS["primary"], linewidth=1,
                                  transform=fig.transFigure))

        table_data = [headers]
        row_colors = {}
        for idx, inj in enumerate(page_injections, 1):
            row_num = page_start + idx
            direct = inj.get("direct", {})
            uib = inj.get("uib", {})
            dad = inj.get("dad", {})
            injection = inj.get("injection", {})

            pts_doc = direct.get("n_points", 0)
            pts_uib = uib.get("n_points", 0)
            pts_dad = dad.get("n_points", 0)

            uib_file = uib.get("file", "-") or "-"
            dad_file = dad.get("file", "-") or "-"
            if len(uib_file) > 22:
                uib_file = uib_file[:20] + ".."
            if len(dad_file) > 22:
                dad_file = dad_file[:20] + ".."

            vol = injection.get("inj_volume")
            vol_str = f"{int(vol)}" if vol else "-"
            t_max = direct.get("t_max", 0) or 0
            n_timeouts = direct.get("n_timeouts", 0)
            has_timeout = direct.get("has_timeout", False) or n_timeouts > 0

            if t_max > 0:
                t_str = f"{t_max:.1f}"
                if has_timeout:
                    t_str += f" [T:{n_timeouts}]"
            else:
                t_str = "-"

            if data_mode == "DUAL":
                row = [str(row_num), inj["name"][:22], str(inj["replica"]),
                       str(pts_doc) if pts_doc else "-",
                       str(pts_uib) if pts_uib else "-",
                       str(pts_dad) if pts_dad else "-",
                       uib_file, dad_file, vol_str, t_str]
            elif data_mode == "UIB":
                row = [str(row_num), inj["name"][:26], str(inj["replica"]),
                       str(pts_doc) if pts_doc else "-",
                       str(pts_uib) if pts_uib else "-",
                       uib_file, vol_str, t_str]
            else:
                row = [str(row_num), inj["name"][:26], str(inj["replica"]),
                       str(pts_doc) if pts_doc else "-",
                       str(pts_dad) if pts_dad else "-",
                       dad_file, vol_str, t_str]

            table_data.append(row)
            data_row_idx = idx
            if inj["type"] == "KHP":
                row_colors[data_row_idx] = COLORS["khp_row"]
            elif inj["type"] in ("CONTROL", "BLANK"):
                row_colors[data_row_idx] = COLORS["control_row"]

        ax_table = fig.add_axes([0.03, 0.06, 0.94, 0.83])
        draw_minimal_table(ax_table, table_data, col_widths,
                           row_colors=row_colors, font_size=7,
                           max_row_height=0.9 / 36)

        draw_report_footer(fig, page_counter)
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

    return page_counter


def _imp_load_data_for_chromatograms(seq_path):
    """Carrega dades importades (amb arrays) per generar cromatogrames."""
    try:
        from hpsec_import import load_manifest, import_from_manifest
        manifest = load_manifest(seq_path)
        if manifest:
            result = import_from_manifest(seq_path, manifest)
            if result and result.get("success"):
                return result
    except Exception as e:
        print(f"  [WARNING] No s'han pogut carregar dades per cromatogrames: {e}")
    return None


def _imp_plot_sample_chromatogram(ax, title, rep_data, show_legend=True):
    """Plotar cromatograma d'una replica amb DOC Direct + DOC UIB + DAD 254nm."""
    if rep_data is None:
        ax.text(0.5, 0.5, "Sense replica", ha='center', va='center',
                fontsize=8, color='#BBBBBB', transform=ax.transAxes)
        ax.text(0.03, 0.95, title, transform=ax.transAxes,
                fontsize=6.5, fontweight='bold', color=COLORS["text_secondary"],
                va='top', ha='left')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color('#EEEEEE')
        return

    has_data = False
    all_handles = []
    all_labels = []

    # DOC Direct
    direct = rep_data.get("direct", {})
    if direct and direct.get("t") is not None:
        t = np.asarray(direct["t"])
        y = direct.get("y") if direct.get("y") is not None else direct.get("y_raw")
        if y is not None:
            y = np.asarray(y)
            if len(t) > 0 and len(y) > 0:
                line, = ax.plot(t, y, color=COLORS["doc_direct"], linewidth=0.8)
                all_handles.append(line)
                all_labels.append("DOC")
                has_data = True

    # DOC UIB
    uib = rep_data.get("uib", {})
    if uib and uib.get("t") is not None:
        t = np.asarray(uib["t"])
        y = uib.get("y") if uib.get("y") is not None else uib.get("y_raw")
        if y is not None:
            y = np.asarray(y)
            if len(t) > 0 and len(y) > 0:
                line, = ax.plot(t, y, color=COLORS["doc_uib"], linewidth=0.8, alpha=0.8)
                all_handles.append(line)
                all_labels.append("UIB")
                has_data = True

    # DAD 254nm (eix secundari)
    dad = rep_data.get("dad", {})
    if dad:
        t_dad = None
        y254 = None
        df_dad = dad.get("df")
        if df_dad is not None:
            try:
                if "time (min)" in df_dad.columns:
                    t_dad = df_dad["time (min)"].values
                for col in df_dad.columns:
                    if "254" in str(col):
                        y254 = df_dad[col].values
                        break
            except Exception:
                pass
        if t_dad is None and dad.get("t") is not None:
            t_dad = np.asarray(dad["t"])
            wavelengths = dad.get("wavelengths", {})
            y254 = wavelengths.get(254) or wavelengths.get("254")
            if y254 is not None:
                y254 = np.asarray(y254)

        if t_dad is not None and y254 is not None and len(t_dad) > 0:
            ax2 = ax.twinx()
            line, = ax2.plot(t_dad, y254, color=COLORS["dad_254"],
                             linewidth=0.6, linestyle="--", alpha=0.6)
            all_handles.append(line)
            all_labels.append("254nm")
            ax2.tick_params(axis='y', labelsize=5, colors=COLORS["dad_254"])
            ax2.set_ylabel("254nm", fontsize=5, color=COLORS["dad_254"], labelpad=2)
            ax2.spines['right'].set_visible(True)
            ax2.spines['right'].set_color(COLORS["dad_254"])
            ax2.spines['right'].set_linewidth(0.5)
            has_data = True

    if not has_data:
        ax.text(0.5, 0.5, "Sense dades", ha='center', va='center',
                fontsize=8, color='#BBBBBB', transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color('#EEEEEE')
    else:
        ax.set_xlabel("min", fontsize=5, labelpad=1)
        ax.set_ylabel("DOC (mAU)", fontsize=5, labelpad=1)
        ax.tick_params(axis='both', labelsize=5, pad=1)
        ax.grid(True, alpha=0.3, linewidth=0.3)
        if show_legend and all_handles:
            ax.legend(all_handles, all_labels, loc='upper right',
                      fontsize=5, frameon=True, framealpha=0.85,
                      edgecolor='none', fancybox=False, handlelength=1.2)

    ax.text(0.03, 0.95, title, transform=ax.transAxes,
            fontsize=6.5, fontweight='bold', color=COLORS["text"],
            va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                      edgecolor='none', alpha=0.8))


def _imp_draw_chromatogram_pages(pdf, seq_path, manifest, page_counter):
    """Dibuixa pagines amb cromatogrames: 2 per fila (R1+R2), 1 fila per mostra."""
    from pathlib import Path
    imported = _imp_load_data_for_chromatograms(seq_path)
    if not imported:
        return page_counter

    samples_dict = imported.get("samples", {})
    if not samples_dict:
        return page_counter

    seq_name = manifest.get("sequence", {}).get("name", "")
    manifest_samples = manifest.get("samples", [])

    sample_items = []
    for ms in manifest_samples:
        sample_name = ms.get("name", "")
        if sample_name not in samples_dict:
            continue
        rep_dict = samples_dict[sample_name].get("replicas", {})
        if not rep_dict:
            continue
        sorted_keys = sorted(rep_dict.keys())
        r1_data = rep_dict.get(sorted_keys[0]) if len(sorted_keys) >= 1 else None
        r2_data = rep_dict.get(sorted_keys[1]) if len(sorted_keys) >= 2 else None
        has_any = False
        for rd in [r1_data, r2_data]:
            if rd is None:
                continue
            if ((rd.get("direct", {}) or {}).get("t") is not None or
                    (rd.get("uib", {}) or {}).get("t") is not None or
                    (rd.get("dad", {}) or {}).get("df") is not None):
                has_any = True
                break
        if has_any:
            sample_items.append((sample_name, r1_data, r2_data))

    if not sample_items:
        return page_counter

    ROWS_PER_PAGE = 4
    graphs_dir = Path(seq_path) / "CHECK" / "Graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)

    for page_start in range(0, len(sample_items), ROWS_PER_PAGE):
        page_counter += 1
        page_items = sample_items[page_start:page_start + ROWS_PER_PAGE]

        fig = plt.figure(figsize=(11.69, 8.27))
        fig.patch.set_facecolor('white')
        fig.text(0.5, 0.96, "CROMATOGRAMES RAW", ha='center', va='top',
                 fontsize=14, fontweight='bold', color=COLORS["primary"])
        fig.text(0.5, 0.93, f"{seq_name} - Pagina {page_counter}",
                 ha='center', va='top', fontsize=10, color=COLORS["text_secondary"])
        fig.add_artist(plt.Line2D([0.05, 0.95], [0.91, 0.91],
                                  color=COLORS["primary"], linewidth=1,
                                  transform=fig.transFigure))

        row_height = 0.20
        col_width = 0.39
        left_margins = [0.07, 0.54]
        top_start = 0.70

        for row_idx, (name, r1, r2) in enumerate(page_items):
            bottom = top_start - row_idx * row_height
            ax1 = fig.add_axes([left_margins[0], bottom + 0.025,
                                col_width, row_height - 0.045])
            _imp_plot_sample_chromatogram(ax1, f"{name} R1", r1)
            ax2 = fig.add_axes([left_margins[1], bottom + 0.025,
                                col_width, row_height - 0.045])
            _imp_plot_sample_chromatogram(ax2, f"{name} R2", r2)

        draw_report_footer(fig, page_counter)
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

    # Guardar PNG primera pagina
    try:
        first_items = sample_items[:ROWS_PER_PAGE]
        if first_items:
            save_fig = plt.figure(figsize=(11.69, 8.27))
            save_fig.patch.set_facecolor('white')
            save_fig.text(0.5, 0.96, "CROMATOGRAMES RAW", ha='center',
                          va='top', fontsize=14, fontweight='bold',
                          color=COLORS["primary"])
            save_fig.text(0.5, 0.93, seq_name, ha='center', va='top',
                          fontsize=10, color=COLORS["text_secondary"])

            for row_idx, (name, r1, r2) in enumerate(first_items):
                bottom = 0.70 - row_idx * 0.20
                ax1 = save_fig.add_axes([0.07, bottom + 0.025, 0.39, 0.155])
                _imp_plot_sample_chromatogram(ax1, f"{name} R1", r1)
                ax2 = save_fig.add_axes([0.54, bottom + 0.025, 0.39, 0.155])
                _imp_plot_sample_chromatogram(ax2, f"{name} R2", r2)

            png_path = graphs_dir / f"import_chromatograms_{seq_name}.png"
            save_fig.savefig(str(png_path), dpi=150, bbox_inches='tight',
                             facecolor='white')
            plt.close(save_fig)
    except Exception as e:
        print(f"  [WARNING] No s'ha pogut guardar PNG de cromatogrames: {e}")

    return page_counter


def generate_import_report(seq_path, output_path=None):
    """
    Genera PDF d'importacio amb estil professional.

    Args:
        seq_path: Ruta a la carpeta SEQ
        output_path: Ruta de sortida (default: CHECK/)

    Returns:
        Path del PDF generat o None si hi ha error
    """
    from pathlib import Path

    manifest = _imp_load_manifest(seq_path)
    if not manifest:
        print(f"No s'han trobat dades d'importacio a {seq_path}")
        return None

    seq_name = manifest.get("sequence", {}).get("name", Path(seq_path).name)
    _imp_reclassify_samples(manifest.get("samples", []))

    if output_path is None:
        output_path = Path(seq_path) / "CHECK"
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    pdf_path = output_path / f"REPORT_Import_{seq_name}.pdf"
    if pdf_path.exists():
        try:
            with open(pdf_path, 'a'):
                pass
        except PermissionError:
            timestamp = datetime.now().strftime("%H%M%S")
            pdf_path = output_path / f"REPORT_Import_{seq_name}_{timestamp}.pdf"
            print(f"  Fitxer bloquejat, usant: {pdf_path.name}")

    print(f"Generant {pdf_path}...")

    with PdfPages(pdf_path) as pdf_doc:
        _imp_draw_page_summary(pdf_doc, manifest, seq_name)
        page_counter = 1
        page_counter = _imp_draw_detail_pages(pdf_doc, manifest, page_counter)
        page_counter = _imp_draw_chromatogram_pages(
            pdf_doc, seq_path, manifest, page_counter)

    print(f"  [OK] PDF generat: {pdf_path}")
    return str(pdf_path)


# =============================================================================
# INFORME D'ANÀLISI (PDF)
# =============================================================================

def _ana_load_result(seq_path):
    """Carrega analysis_result.json de la SEQ."""
    from pathlib import Path
    import json as _json
    p = Path(seq_path) / "CHECK" / "data" / "analysis_result.json"
    if not p.exists():
        return None
    try:
        with open(p, 'r', encoding='utf-8') as f:
            data = _json.load(f)
        _ana_restore_arrays(data)
        return data
    except Exception as e:
        print(f"Error carregant analysis_result: {e}")
        return None


def _ana_restore_arrays(data):
    """Restaura arrays numpy i DataFrames des de JSON."""
    _ARRAY_KEYS = ("t_doc", "y_doc_net", "y_doc_uib_net", "y_doc_direct_net")

    def _restore_sample(sample):
        df_dad_dict = sample.get("df_dad")
        if df_dad_dict and isinstance(df_dad_dict, dict):
            try:
                sample["df_dad"] = pd.DataFrame(df_dad_dict)
            except Exception:
                sample["df_dad"] = None
        for key in _ARRAY_KEYS:
            val = sample.get(key)
            if val is not None and isinstance(val, list):
                sample[key] = np.array(val)

    for sample in data.get("samples", []):
        _restore_sample(sample)
    for sample_data in data.get("samples_grouped", {}).values():
        for rep_data in sample_data.get("replicas", {}).values():
            _restore_sample(rep_data)


def _ana_get_status_color(anomalies, score=1.0):
    """Retorna color per una mostra segons anomalies."""
    from hpsec_warnings import classify_anomalies
    classified = classify_anomalies(anomalies)
    if classified["blocker"]:
        return COLORS["danger"]
    elif classified["repaired"] or classified["warning"] or score < 0.7:
        return COLORS["warning"]
    return COLORS["accent"]


def _ana_status_text(anomalies, sample_valid=True):
    """Text curt d'estat per una mostra."""
    if not sample_valid:
        return "NO VAL"
    from hpsec_warnings import ANOMALY_CATALOG as _AC
    parts = []
    seen_icons = set()
    for a in anomalies:
        if isinstance(a, dict):
            code = a.get("code", "")
            repaired = a.get("repaired", False)
        else:
            repaired = "_REPAIRED" in str(a)
            code = str(a).replace("_REPAIRED", "")
        entry = _AC.get(code, {})
        icon = entry.get("icon", "")
        if icon and icon not in seen_icons:
            seen_icons.add(icon)
            parts.append(f"{icon}*" if repaired else icon)
    return " ".join(parts) if parts else "OK"


def _ana_draw_page1_summary(pdf, data, seq_name):
    """Pagina 1: resum general de la sequencia analitzada."""
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor('white')

    logo_path = get_logo_path()
    if logo_path:
        try:
            logo = plt.imread(logo_path)
            ax_logo = fig.add_axes([0.05, 0.88, 0.25, 0.08])
            ax_logo.imshow(logo)
            ax_logo.axis('off')
        except Exception:
            pass

    fig.text(0.95, 0.94, "HPSEC Suite", ha='right', va='top',
             fontsize=10, color=COLORS["text_secondary"])
    fig.text(0.5, 0.85, "INFORME D'ANALISI", ha='center', va='top',
             fontsize=18, fontweight='bold', color=COLORS["primary"])
    fig.text(0.5, 0.81, f"Sequencia {seq_name}", ha='center', va='top',
             fontsize=12, color=COLORS["text"])

    date_str = datetime.now().strftime("%d/%m/%Y %H:%M")
    date_processed = data.get("date_processed", "")
    if date_processed:
        try:
            dt = datetime.fromisoformat(date_processed)
            date_processed = dt.strftime("%d/%m/%Y %H:%M")
        except Exception:
            pass

    fig.text(0.5, 0.775, f"Generat: {date_str}", ha='center', va='top',
             fontsize=9, color=COLORS["text_secondary"])

    # v2.2.0: avís si la quantificació encara està pendent
    if data.get("quantification_pending"):
        fig.text(0.5, 0.755,
                 "⚠ QUANTIFICACIÓ PENDENT — les columnes ppm es mostren buides.\n"
                 "Genera el report després d'aplicar la recta al pas Quantificar.",
                 ha='center', va='top',
                 fontsize=9, color="#c0392b", style='italic',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='#fdecea',
                           edgecolor='#c0392b', linewidth=0.8))

    fig.add_artist(plt.Line2D([0.1, 0.9], [0.75, 0.75],
                              color=COLORS["primary"], linewidth=2,
                              transform=fig.transFigure))

    method = data.get("method", "-")
    data_mode = data.get("data_mode", "-")
    summary = data.get("summary", {})

    # Compute meaningful stats from samples_grouped
    from hpsec_warnings import ANOMALY_CATALOG as _AC_p1
    _sg_p1 = data.get("samples_grouped", {})
    n_samples = 0
    n_valid = 0
    n_invalid = 0
    n_blocker = 0
    n_batman = 0
    n_batman_repaired = 0
    for _sn, _sd in _sg_p1.items():
        if _sd.get("analysis_type") == "light":
            continue
        n_samples += 1
        if not _sd.get("sample_valid", True):
            n_invalid += 1
            continue
        n_valid += 1
        _sel = (_sd.get("selected") or {}).get("doc", "1")
        _rd = (_sd.get("replicas") or {}).get(_sel, {})
        has_blocker = False
        for _a in _rd.get("anomalies", []):
            if isinstance(_a, dict):
                _code = _a.get("code", "")
                _repaired = _a.get("repaired", False)
            else:
                _code = str(_a).replace("_REPAIRED", "")
                _repaired = "_REPAIRED" in str(_a)
            _entry = _AC_p1.get(_code, {})
            _sev = _entry.get("severity")
            if "IRREGULAR_TOP" in _code:
                if _repaired:
                    n_batman_repaired += 1
                else:
                    n_batman += 1
            if _sev and _sev.value == "blocker" and not _repaired:
                has_blocker = True
        if has_blocker:
            n_blocker += 1

    n_light = sum(1 for _sd in _sg_p1.values()
                  if _sd.get("analysis_type") == "light")

    info_data = [
        ["Parametre", "Valor", "Parametre", "Valor"],
        ["Metode", method, "Mode dades", data_mode],
        ["Data processament", date_processed or "-",
         "Mostres analitzades", str(n_samples)],
        ["Mostres valides", str(n_valid),
         "No valides", str(n_invalid) if n_invalid else "-"],
        ["Amb anomalies critiques", str(n_blocker) if n_blocker else "-",
         "Irregular top (batman)", str(n_batman) if n_batman else "-"],
        ["Batman reparat", str(n_batman_repaired) if n_batman_repaired else "-",
         "BLANC/CONTROL", str(n_light) if n_light else "-"],
    ]

    ax_info = fig.add_axes([0.08, 0.62, 0.84, 0.11])
    draw_minimal_table(ax_info, info_data,
                       col_widths=[0.20, 0.30, 0.20, 0.30], font_size=9)

    # Distribucio anomalies
    fig.text(0.08, 0.60, "Distribucio d'anomalies", fontsize=11,
             fontweight='bold', color=COLORS["text"])

    from hpsec_warnings import ANOMALY_CATALOG as _AC
    samples_grouped = data.get("samples_grouped", {})
    anomaly_counts = {}
    for sg_name, sg_data in samples_grouped.items():
        for rep_key, rep_data in (sg_data.get("replicas") or {}).items():
            for anom in rep_data.get("anomalies", []):
                if isinstance(anom, dict):
                    code = anom.get("code", "?")
                    repaired = anom.get("repaired", False)
                    key = f"{code}_REPAIRED" if repaired else code
                else:
                    key = str(anom)
                anomaly_counts[key] = anomaly_counts.get(key, 0) + 1

    if anomaly_counts:
        anom_table = [["Anomalia", "Vegades", "Severitat"]]
        for anom_key, count in sorted(anomaly_counts.items(), key=lambda x: -x[1]):
            repaired = "_REPAIRED" in anom_key
            code = anom_key.replace("_REPAIRED", "")
            entry = _AC.get(code, {})
            label = entry.get("label", code)
            if repaired:
                severity = "Reparat"
                label += " (reparat)"
            elif entry.get("severity") and entry["severity"].value == "blocker":
                severity = "CRITIC"
            elif entry.get("severity") and entry["severity"].value == "warning":
                severity = "Avis"
            else:
                severity = "Info"
            anom_table.append([label, str(count), severity])

        n_anom_rows = len(anom_table)
        table_h = min(0.015 * (n_anom_rows + 1), 0.20)
        ax_anom = fig.add_axes([0.08, 0.58 - table_h, 0.84, table_h])
        draw_minimal_table(ax_anom, anom_table,
                           col_widths=[0.50, 0.20, 0.30], font_size=8)
    else:
        fig.text(0.08, 0.56, "Cap anomalia detectada",
                 fontsize=9, color=COLORS["accent"])

    draw_report_footer(fig, 1)
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def _ana_draw_results_pages(pdf, data, page_start=2):
    """Pagines landscape amb taula de resultats (13 columnes).

    Mostres regulars primer, controls (MQ/KHP/BLANK/NaOH) al final amb fons gris.
    Per KHP: mostra area_peak (integració pic principal) en lloc d'area_total.
    """
    samples_grouped = data.get("samples_grouped", {})
    if not samples_grouped:
        return page_start

    regular_names, control_names = _ana_classify_samples(samples_grouped)
    # Ordre: regulars primer, després controls (separador visual via color)
    ordered_names = regular_names + control_names
    control_set = set(control_names)

    def _build_row(sample_name):
        sg = samples_grouped[sample_name]
        selected = sg.get("selected") or {}
        quant = sg.get("quantification") or {}
        comparison = sg.get("comparison") or {}
        doc_sel = selected.get("doc", "1")
        dad_sel = selected.get("dad", "1")
        doc_rep = (sg.get("replicas") or {}).get(doc_sel, {})
        dad_rep = (sg.get("replicas") or {}).get(dad_sel, {})
        sample_valid = sg.get("sample_valid", True)
        is_control = sample_name in control_set

        # Area DOC: per KHP usa area_peak (coherent amb calibració)
        areas = doc_rep.get("areas") or {}
        doc_areas = areas.get("DOC") or {}
        peak_info = doc_rep.get("peak_info") or {}
        if is_khp(sample_name) and peak_info.get("area"):
            area_direct = peak_info["area"]
            area_label = f"{format_value(area_direct, '.0f')}*"
        else:
            area_direct = doc_areas.get("total", 0)
            area_label = format_value(area_direct, ".0f")

        areas_uib = doc_rep.get("areas_uib") or {}
        area_uib = areas_uib.get("total", 0)

        ppm_d = quant.get("concentration_ppm_direct") or quant.get("concentration_ppm")
        ppm_u = quant.get("concentration_ppm_uib")
        snr_info = doc_rep.get("snr_info") or {}
        snr_d = snr_info.get("snr_direct", 0)

        dad_areas = (dad_rep.get("areas") or {})
        area_254 = dad_areas.get("A254", {}).get("total", 0)
        snr_254 = dad_rep.get("snr_info_dad", {}).get("A254", {}).get("snr", 0)

        r2_doc = comparison.get("doc", {}).get("pearson", 0) if comparison else 0
        r2_dad = comparison.get("dad", {}).get("pearson_min", 0) if comparison else 0

        anomalies = doc_rep.get("anomalies", [])
        status = _ana_status_text(anomalies, sample_valid)
        color = _ana_get_status_color(anomalies)

        return {
            "row": [
                sample_name[:20],
                f"R{doc_sel}" if doc_sel != "none" else "Cap",
                f"R{dad_sel}" if dad_sel != "none" else "Cap",
                area_label,
                format_value(ppm_d, ".2f") if sample_valid and ppm_d else "-",
                format_value(area_uib, ".0f"),
                format_value(ppm_u, ".2f") if ppm_u else "-",
                format_value(snr_d, ".0f"),
                format_value(area_254, ".1f"),
                format_value(snr_254, ".0f"),
                format_value(r2_doc, ".4f") if r2_doc > 0 else "-",
                format_value(r2_dad, ".4f") if r2_dad > 0 else "-",
                status,
            ],
            "color": color,
            "is_control": is_control,
        }

    table_rows = [_build_row(name) for name in ordered_names]

    rows_per_page = 20
    headers = ["Mostra", "DOC", "DAD", "A_DOC", "ppm",
               "A_UIB", "ppm_U", "SNR", "A_254", "SNR_254",
               "R2_DOC", "R2_DAD", "Estat"]
    col_widths = [0.14, 0.04, 0.04, 0.08, 0.07,
                  0.08, 0.07, 0.06, 0.08, 0.06,
                  0.08, 0.08, 0.08]

    page_num = page_start
    for page_start_idx in range(0, len(table_rows), rows_per_page):
        page_rows = table_rows[page_start_idx:page_start_idx + rows_per_page]

        fig = plt.figure(figsize=(11.69, 8.27))
        fig.patch.set_facecolor('white')
        fig.text(0.5, 0.96, "RESULTATS D'ANALISI",
                 ha='center', va='top', fontsize=14, fontweight='bold',
                 color=COLORS["primary"])
        fig.text(0.5, 0.93,
                 f"Mostres {page_start_idx + 1}-{page_start_idx + len(page_rows)} "
                 f"de {len(table_rows)}",
                 ha='center', va='top', fontsize=9, color=COLORS["text_secondary"])

        table_data = [headers]
        row_colors = {}
        has_khp_star = False
        for idx, entry in enumerate(page_rows):
            table_data.append(entry["row"])
            if entry.get("is_control"):
                row_colors[idx + 1] = "#F0F0F0"  # gris controls
            elif entry["color"] == COLORS["danger"]:
                row_colors[idx + 1] = "#FDEDEC"
            elif entry["color"] == COLORS["warning"]:
                row_colors[idx + 1] = "#FEF9E7"
            # Detectar si alguna fila té l'asterisc KHP
            if any("*" in str(c) for c in entry["row"]):
                has_khp_star = True

        n_rows = len(table_data)
        table_h = min(0.85, 0.035 * n_rows + 0.02)
        ax = fig.add_axes([0.03, 0.06, 0.94, table_h])
        draw_minimal_table(ax, table_data, col_widths=col_widths,
                           row_colors=row_colors, font_size=7.5)

        # Nota al peu per KHP area_peak
        if has_khp_star:
            fig.text(0.03, 0.03,
                     "* Àrea del pic principal (integració tangent). "
                     "Les mostres mostren àrea total del cromatograma.",
                     fontsize=6.5, color=COLORS["text_secondary"],
                     style='italic')

        draw_report_footer(fig, page_num)
        pdf.savefig(fig, dpi=150)
        plt.close(fig)
        page_num += 1

    return page_num


def _ana_get_fractions(method):
    """Retorna llista de fraccions per al metode."""
    try:
        from hpsec_config import get_config
        cfg = get_config()
        mode = "BP" if method.upper() == "BP" else "COLUMN"
        return cfg.get_all_fractions(mode)
    except Exception:
        return []


def _ana_draw_fraction_table(ax, sg, is_bp, fracs, wl_cols):
    """Dibuixa taula de fraccions a un axes."""
    ax.axis('off')
    selected = sg.get("selected") or {}
    rep_sel = selected.get("doc", "1")
    rep_data = (sg.get("replicas") or {}).get(rep_sel, {})

    sel_areas = rep_data.get("areas") or {}
    areas_uib = rep_data.get("areas_uib") or {}
    doc_areas = sel_areas.get("DOC", {})
    doc_total = doc_areas.get("total", 0)
    uib_total = areas_uib.get("total", 0)

    if is_bp:
        col_labels = ["Senyal", "Area Total"]
        rows = [["DOC", f"{doc_total:.1f}" if doc_total else "-"]]
        if uib_total:
            rows.append(["UIB", f"{uib_total:.1f}"])
        for wl in wl_cols:
            wl_lbl = f"A{wl}" if not str(wl).startswith('A') else wl
            wl_area = sel_areas.get(wl_lbl, sel_areas.get(wl, {}))
            total = wl_area.get("total", 0) if isinstance(wl_area, dict) else 0
            rows.append([wl_lbl, f"{total:.1f}" if total else "-"])
    else:
        x_max = 70
        col_labels = ["Senyal"]
        for fname, finfo in fracs:
            col_labels.append(f"{fname}\n({finfo['start']:g}-{finfo['end']:g})")
        col_labels.append(f"TOTAL\n(0-{x_max:g})")

        signal_names = ["DOC"]
        if uib_total > 0:
            signal_names.append("UIB")
        for wl in wl_cols:
            wl_lbl = f"A{wl}" if not str(wl).startswith('A') else wl
            signal_names.append(wl_lbl)

        rows = []
        for sig in signal_names:
            row = [sig]
            if sig == "DOC":
                sig_areas, sig_total = doc_areas, doc_total
            elif sig == "UIB":
                sig_areas, sig_total = areas_uib, uib_total
            else:
                sig_areas = sel_areas.get(sig, {})
                sig_total = sig_areas.get("total", 0) if isinstance(sig_areas, dict) else 0
            for fname, _finfo in fracs:
                fval = sig_areas.get(fname, 0) if isinstance(sig_areas, dict) else 0
                pct = (fval / sig_total * 100) if sig_total > 0 else 0
                row.append(f"{pct:.1f}%")
            row.append(f"{sig_total:.1f}" if sig_total > 0 else "-")
            rows.append(row)

    if not rows:
        return

    tbl = ax.table(cellText=rows, colLabels=col_labels,
                   loc='upper center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(5.5)
    tbl.scale(1, 1.2)
    for key, cell in tbl.get_celld().items():
        cell.set_linewidth(0.3)
        cell.set_height(0.12)
        if key[0] == 0:
            cell.set_facecolor('#E0E0E0')
            cell.set_text_props(fontweight='bold', fontsize=5)
        elif key[1] == 0:
            cell.set_facecolor('#F5F5F5')
            cell.set_text_props(fontweight='bold', fontsize=5.5)
        else:
            cell.set_facecolor('white')


def _ana_draw_chromatogram_pages(pdf, data, page_start):
    """Genera pagines de cromatogrames per cada mostra."""
    from matplotlib.gridspec import GridSpec as _GS
    samples_grouped = data.get("samples_grouped", {})
    method = data.get("method", "COLUMN")
    is_bp = method.upper() == "BP"
    x_min, x_max = (0, 10) if is_bp else (0, 70)
    fracs = _ana_get_fractions(method)
    page_num = page_start

    def _add_vlines(ax):
        if not is_bp and fracs:
            for _fn, fi in fracs:
                s = fi['start']
                if s > 0 and s <= x_max:
                    ax.axvline(s, color='#999', ls=':', lw=0.4, zorder=0)

    def _add_timeout_zones(ax, timeout_info):
        """Afegeix zones de timeout com a bandes vermelles translúcides."""
        if not timeout_info:
            return
        for to in timeout_info.get("timeouts", []):
            t_s = to.get("t_start_min")
            t_e = to.get("t_end_min")
            if t_s is not None and t_e is not None:
                ax.axvspan(t_s, t_e, alpha=0.12, color='#E74C3C', zorder=0)

    for sample_name in sorted(samples_grouped.keys()):
        sg = samples_grouped[sample_name]
        replicas = sg.get("replicas", {})
        if not replicas:
            continue

        selected = sg.get("selected") or {}
        comparison = sg.get("comparison") or {}
        doc_comp = comparison.get("doc") or {}
        dad_comp = comparison.get("dad") or {}
        quant = sg.get("quantification") or {}
        is_light = sg.get("analysis_type") == "light"

        rep_keys = sorted(replicas.keys())
        r1 = replicas.get(rep_keys[0], {})
        r2 = replicas.get(rep_keys[1], {}) if len(rep_keys) > 1 else None

        t1 = r1.get("t_doc")
        y1_d = r1.get("y_doc_net")
        y1_u = r1.get("y_doc_uib_net")
        has_chromatogram_data = t1 is not None and y1_d is not None

        # Timeout info from selected replica
        doc_sel = selected.get("doc", rep_keys[0])
        sel_rep_data = replicas.get(doc_sel, r1)
        timeout_info = sel_rep_data.get("timeout_info") or {}

        # BLANK/CONTROL: simplified page (DOC only, full width)
        if is_light and has_chromatogram_data:
            fig = plt.figure(figsize=(11.69, 8.27))
            fig.patch.set_facecolor('white')
            _ana_draw_sample_header(fig, sg, sample_name)

            ax = fig.add_axes([0.08, 0.10, 0.84, 0.75])
            t1 = np.asarray(t1)
            y1_d = np.asarray(y1_d)
            ax.plot(t1, y1_d, color='#888', lw=LW, label=f'R{rep_keys[0]}')
            if r2:
                t2 = r2.get("t_doc")
                y2_d = r2.get("y_doc_net")
                if t2 is not None and y2_d is not None:
                    ax.plot(np.asarray(t2), np.asarray(y2_d),
                            color='#bbb', lw=LW, alpha=0.7, label=f'R{rep_keys[1]}')
            ax.set_ylabel("DOC Direct", fontsize=8)
            ax.set_xlabel("Temps (min)", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.2, lw=0.3)
            ax.set_xlim(x_min, x_max)
            _add_vlines(ax)
            _add_timeout_zones(ax, timeout_info)
            if ax.get_legend_handles_labels()[1]:
                ax.legend(loc='upper right', fontsize=6, framealpha=0.7)
            # SNR annotation
            snr_info = sel_rep_data.get("snr_info") or {}
            snr_d = snr_info.get("snr_direct", 0)
            if snr_d > 0:
                ax.text(0.99, 0.95, f"SNR={snr_d:.0f}",
                        transform=ax.transAxes, fontsize=6,
                        color='#555', ha='right', va='top')

            draw_report_footer(fig, page_num)
            pdf.savefig(fig, dpi=150)
            plt.close(fig)
            page_num += 1
            continue

        df_dad1 = r1.get("df_dad")
        wl_cols = []
        if df_dad1 is not None and hasattr(df_dad1, 'columns'):
            wl_cols = [c for c in df_dad1.columns if c != 'time (min)']
            wl_cols.sort(key=lambda x: int(x) if str(x).isdigit() else 0)

        pairs = []
        for i in range(0, len(wl_cols), 2):
            if i + 1 < len(wl_cols):
                pairs.append((wl_cols[i], wl_cols[i + 1]))
            else:
                pairs.append((wl_cols[i], None))

        n_graph_rows = 1 + len(pairs)

        fig = plt.figure(figsize=(11.69, 8.27))
        fig.patch.set_facecolor('white')

        # Enhanced header (replaces old title+subtitle)
        _ana_draw_sample_header(fig, sg, sample_name)

        pearson_doc = doc_comp.get("pearson", 0)
        pearson_per_wl = dad_comp.get("pearson_per_wavelength", {})
        ppm_d = quant.get("concentration_ppm_direct") or quant.get("concentration_ppm")
        ppm_u = quant.get("concentration_ppm_uib")

        if not has_chromatogram_data:
            ax_msg = fig.add_axes([0.1, 0.35, 0.8, 0.5])
            ax_msg.axis('off')
            ax_msg.text(0.5, 0.6, "Dades cromatografiques no disponibles",
                        ha='center', va='center', fontsize=14,
                        color=COLORS["text_secondary"])
            sel_areas = r1.get("areas") or {}
            if sel_areas:
                ax_tbl = fig.add_axes([0.07, 0.08, 0.86, 0.22])
                _ana_draw_fraction_table(ax_tbl, sg, is_bp, fracs, wl_cols)
            draw_report_footer(fig, page_num)
            pdf.savefig(fig, dpi=150)
            plt.close(fig)
            page_num += 1
            continue

        h_graphs = [1.0] * n_graph_rows
        h_table = [1.8]
        heights = h_graphs + h_table
        n_total_rows = n_graph_rows + 1

        gs = fig.add_gridspec(
            n_total_rows, 2, height_ratios=heights,
            hspace=0.35, wspace=0.25,
            top=0.85, bottom=0.07, left=0.08, right=0.92)

        # Row 0: DOC Direct | DOC UIB
        ax_doc = fig.add_subplot(gs[0, 0])
        ax_uib = fig.add_subplot(gs[0, 1])

        t1 = np.asarray(t1)
        y1_d = np.asarray(y1_d)
        ax_doc.plot(t1, y1_d, color=C1, lw=LW, label=f'R{rep_keys[0]}')
        if r2:
            t2 = r2.get("t_doc")
            y2_d = r2.get("y_doc_net")
            if t2 is not None and y2_d is not None:
                ax_doc.plot(np.asarray(t2), np.asarray(y2_d),
                            color=C2, lw=LW, alpha=0.7, label=f'R{rep_keys[1]}')

        ax_doc.set_ylabel("DOC Direct", fontsize=7)
        ax_doc.tick_params(labelsize=6, length=2, pad=1)
        ax_doc.grid(True, alpha=0.2, lw=0.3)
        ax_doc.set_xlim(x_min, x_max)
        _add_vlines(ax_doc)
        _add_timeout_zones(ax_doc, timeout_info)
        if ax_doc.get_legend_handles_labels()[1]:
            ax_doc.legend(loc='upper right', fontsize=5.5, ncol=1,
                          framealpha=0.7, handlelength=1.2)

        if pearson_doc > 0:
            ann_parts = [f"R2={pearson_doc:.4f}"]
            if ppm_d:
                ann_parts.append(f"{ppm_d:.2f} ppm")
            clr = '#C62828' if pearson_doc < 0.990 else '#555'
            ax_doc.text(0.99, 0.72, "  ".join(ann_parts),
                        transform=ax_doc.transAxes, fontsize=5,
                        color=clr, ha='right', va='top')

        has_uib = False
        if y1_u is not None:
            y1_u = np.asarray(y1_u)
            if len(y1_u) == len(t1):
                has_uib = True
                ax_uib.plot(t1, y1_u, color=C_UIB, lw=LW, label=f'R{rep_keys[0]}')
                if r2:
                    y2_u = r2.get("y_doc_uib_net")
                    t2 = r2.get("t_doc")
                    if y2_u is not None and t2 is not None:
                        y2_u = np.asarray(y2_u)
                        t2 = np.asarray(t2)
                        if len(y2_u) == len(t2):
                            ax_uib.plot(t2, y2_u, color=C_UIB2, lw=LW,
                                        alpha=0.7, label=f'R{rep_keys[1]}')
                ax_uib.legend(loc='upper right', fontsize=5.5, ncol=1,
                              framealpha=0.7, handlelength=1.2)
                if ppm_u:
                    ax_uib.text(0.99, 0.72, f"{ppm_u:.2f} ppm",
                                transform=ax_uib.transAxes, fontsize=5,
                                color='#555', ha='right', va='top')
                _add_timeout_zones(ax_uib, timeout_info)

        if not has_uib:
            ax_uib.text(0.5, 0.5, "UIB no disponible",
                        ha='center', va='center',
                        transform=ax_uib.transAxes, fontsize=8, color='#aaa')

        ax_uib.set_ylabel("DOC UIB", fontsize=7)
        ax_uib.tick_params(labelsize=6, length=2, pad=1)
        ax_uib.grid(True, alpha=0.2, lw=0.3)
        ax_uib.set_xlim(x_min, x_max)
        if has_uib:
            _add_vlines(ax_uib)

        # DAD rows
        for row_i, (wl_left, wl_right) in enumerate(pairs):
            for col_j, wl in enumerate([wl_left, wl_right]):
                if wl is None:
                    ax = fig.add_subplot(gs[row_i + 1, col_j])
                    ax.axis('off')
                    continue
                ax = fig.add_subplot(gs[row_i + 1, col_j])
                if (df_dad1 is not None and 'time (min)' in df_dad1.columns
                        and wl in df_dad1.columns):
                    ax.plot(df_dad1['time (min)'].values,
                            df_dad1[wl].values, color=C1, lw=LW)
                    if r2:
                        df_dad2 = r2.get("df_dad")
                        if (df_dad2 is not None
                                and hasattr(df_dad2, 'columns')
                                and wl in df_dad2.columns):
                            ax.plot(df_dad2['time (min)'].values,
                                    df_dad2[wl].values,
                                    color=C2, lw=LW, alpha=0.7)

                wl_label = f"A{wl}" if not str(wl).startswith('A') else wl
                ax.set_ylabel(wl_label, fontsize=7)
                ax.grid(True, alpha=0.2, lw=0.3)
                ax.tick_params(labelsize=6, length=2, pad=1)
                ax.set_xlim(x_min, x_max)
                _add_vlines(ax)

                wl_key = f"A{wl}" if not str(wl).startswith('A') else wl
                r2v = pearson_per_wl.get(wl_key, 0) or pearson_per_wl.get(str(wl), 0)
                if r2v and r2v > 0:
                    clr = '#C62828' if r2v < 0.990 else '#555'
                    ax.text(0.99, 0.92, f"R2={r2v:.4f}",
                            transform=ax.transAxes, fontsize=5,
                            color=clr, ha='right', va='top')

        # Fraction table
        ax_tbl = fig.add_subplot(gs[n_graph_rows, :])
        _ana_draw_fraction_table(ax_tbl, sg, is_bp, fracs, wl_cols)

        draw_report_footer(fig, page_num)
        pdf.savefig(fig, dpi=150)
        plt.close(fig)
        page_num += 1

        # Heatmap DAD page (if Export3D available)
        page_num = _ana_draw_heatmap_page(pdf, sg, sample_name, method, page_num)

    return page_num


def _ana_draw_anomalies_page(pdf, data, page_num):
    """Pagina final amb detall d'anomalies i warnings per mostra."""
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor('white')
    fig.text(0.5, 0.96, "ANOMALIES I WARNINGS",
             ha='center', va='top', fontsize=14, fontweight='bold',
             color=COLORS["primary"])

    samples_grouped = data.get("samples_grouped", {})
    anomaly_rows = []

    for sample_name in sorted(samples_grouped.keys()):
        sg = samples_grouped[sample_name]
        sample_valid = sg.get("sample_valid", True)
        from hpsec_warnings import ANOMALY_CATALOG as _AC

        for rep_key, rep_data in (sg.get("replicas") or {}).items():
            anomalies = rep_data.get("anomalies", [])
            timeout_info = rep_data.get("timeout_info") or {}

            for anom in anomalies:
                if isinstance(anom, dict):
                    anom_code = anom.get("code", "")
                    anom_repaired = anom.get("repaired", False)
                    anom_details = anom.get("details", {})
                else:
                    anom_repaired = "_REPAIRED" in str(anom)
                    anom_code = str(anom).replace("_REPAIRED", "")
                    anom_details = {}

                entry = _AC.get(anom_code, {})
                sev = entry.get("severity")
                if anom_repaired:
                    severity = "Reparat"
                elif sev and sev.value == "blocker":
                    severity = "CRITIC"
                elif sev and sev.value == "warning":
                    severity = "Avis"
                else:
                    severity = "Info"

                detail = ""
                if "BATMAN" in anom_code:
                    batman_info = anom_details if anom_details else rep_data.get("batman_direct_info", {})
                    if batman_info:
                        depth = batman_info.get("max_depth", 0)
                        if depth:
                            detail = f"depth={depth:.2f}"
                elif "TIMEOUT" in anom_code:
                    n_t = timeout_info.get("n_timeouts", 0)
                    zone_summary = timeout_info.get("zone_summary", {})
                    zones_str = ",".join(zone_summary.keys()) if zone_summary else ""
                    timeouts_list = timeout_info.get("timeouts", [])
                    if timeouts_list:
                        to0 = timeouts_list[0]
                        detail = f"{to0.get('duration_sec',0):.0f}s at {to0.get('t_start_min',0):.1f}min ({to0.get('zone','')})"
                    else:
                        detail = f"n={n_t}, zones={zones_str}"

                anom_label = entry.get("label", anom_code) if entry else anom_code
                anomaly_rows.append([
                    sample_name[:18], f"R{rep_key}", anom_label, severity, detail[:30]
                ])

        # Comparison warnings
        comparison = sg.get("comparison") or {}
        rep_keys_list = sorted((sg.get("replicas") or {}).keys())
        rep_label = f"R{rep_keys_list[0]} vs R{rep_keys_list[1]}" if len(rep_keys_list) >= 2 else "-"
        for signal_key in ("doc", "dad"):
            comp_data = comparison.get(signal_key) or {}
            for warn in comp_data.get("warnings", []):
                if isinstance(warn, dict):
                    w_code = warn.get("code", "")
                    w_entry = _AC.get(w_code, {})
                    w_label = w_entry.get("label", w_code)
                    w_sev = w_entry.get("severity")
                    severity = "Avis" if w_sev and w_sev.value == "warning" else "Info"
                    w_details = warn.get("details", {})
                    detail_str = ""
                    if w_details.get("pearson"):
                        detail_str = f"r={w_details['pearson']:.3f}"
                    elif w_details.get("diff_pct"):
                        detail_str = f"diff={w_details['diff_pct']:.1f}%"
                    elif w_details.get("fraction"):
                        detail_str = f"{w_details['fraction']}: diff={w_details.get('diff_pct', 0):.1f}%"
                else:
                    w_label = str(warn)
                    if "362" in w_label and "CORRELATION" in w_label.upper():
                        severity = "Info"
                    elif "CORRELATION" in w_label.upper():
                        severity = "Avis"
                    elif "AREA_DIFF" in w_label.upper():
                        severity = "Avis"
                    else:
                        severity = "Info"
                    detail_str = ""

                source = "DOC" if signal_key == "doc" else "DAD"
                detail = detail_str or f"Comparacio repliques ({source})"
                anomaly_rows.append([
                    sample_name[:18], rep_label, w_label, severity, detail[:30]
                ])

        if not sample_valid:
            reason = ((sg.get("recommendation") or {})
                      .get("doc") or {}).get("reason", "")
            anomaly_rows.append([
                sample_name[:18], "-", "MOSTRA NO VALIDA", "CRITIC",
                reason[:30]
            ])

    if anomaly_rows:
        headers = ["Mostra", "Rep", "Anomalia", "Severitat", "Detalls"]
        col_widths = [0.20, 0.06, 0.30, 0.14, 0.30]
        max_rows = 30

        for start in range(0, len(anomaly_rows), max_rows):
            if start > 0:
                draw_report_footer(fig, page_num)
                pdf.savefig(fig, dpi=150)
                plt.close(fig)
                page_num += 1
                fig = plt.figure(figsize=(8.27, 11.69))
                fig.patch.set_facecolor('white')
                fig.text(0.5, 0.96, "ANOMALIES I WARNINGS (cont.)",
                         ha='center', va='top', fontsize=14, fontweight='bold',
                         color=COLORS["primary"])

            page_anom = anomaly_rows[start:start + max_rows]
            table_data = [headers] + page_anom
            n_rows = len(table_data)
            table_h = min(0.85, 0.022 * n_rows + 0.02)
            ax = fig.add_axes([0.06, 0.90 - table_h, 0.88, table_h])

            row_colors = {}
            for i, row in enumerate(page_anom):
                if row[3] == "CRITIC":
                    row_colors[i + 1] = "#FDEDEC"
                elif row[3] == "Reparat":
                    row_colors[i + 1] = "#FEF9E7"
                elif row[3] == "Info":
                    row_colors[i + 1] = "#EBF5FB"

            draw_minimal_table(ax, table_data, col_widths=col_widths,
                               row_colors=row_colors, font_size=7.5)

        n_critical = sum(1 for r in anomaly_rows if r[3] == "CRITIC")
        n_warning = sum(1 for r in anomaly_rows if r[3] == "Avis")
        n_repaired = sum(1 for r in anomaly_rows if r[3] == "Reparat")
        n_info = sum(1 for r in anomaly_rows if r[3] == "Info")

        summary_y = 0.90 - table_h - 0.04
        fig.text(0.08, summary_y,
                 f"Total: {len(anomaly_rows)}  |  "
                 f"CRITIC: {n_critical}  |  Avis: {n_warning}  |  "
                 f"Reparat: {n_repaired}  |  Info: {n_info}",
                 fontsize=9, fontweight='bold', color=COLORS["text"])
    else:
        fig.text(0.5, 0.5, "Cap anomalia detectada",
                 ha='center', va='center', fontsize=14,
                 fontweight='bold', color=COLORS["accent"])

    draw_report_footer(fig, page_num)
    pdf.savefig(fig, dpi=150)
    plt.close(fig)
    return page_num + 1


# =============================================================================
# OVERLAY & ENHANCED PAGES — Funcions noves per PDF d'anàlisi millorat
# =============================================================================

def _ana_classify_samples(samples_grouped):
    """Classifica mostres en regulars vs controls (MQ/BLANK/NaOH/KHP).

    Returns:
        (regular_names, control_names) — llistes ordenades
    """
    regular = []
    control = []
    for name in sorted(samples_grouped.keys()):
        sd = samples_grouped[name]
        if sd.get("analysis_type") == "light":
            control.append(name)
            continue
        stype = (sd.get("sample_type") or "").upper()
        if stype in ("BLANK", "CONTROL", "KHP"):
            control.append(name)
            continue
        is_ctrl = any(re.match(p, name, re.IGNORECASE) for p in CONTROL_PATTERNS)
        if is_ctrl or is_khp(name):
            control.append(name)
        else:
            regular.append(name)
    return regular, control


_FRAC_COLORS = {
    "BioP": "#3498DB", "HS": "#E74C3C", "BB": "#F39C12",
    "SB": "#2ECC71", "LMW": "#9B59B6",
}
_FRAC_ORDER = ["BioP", "HS", "BB", "SB", "LMW"]


def _ana_draw_doc_overlay_page(pdf, data, page_num):
    """Pàgina overlay: tots els cromatogrames DOC superposats."""
    samples_grouped = data.get("samples_grouped", {})
    method = data.get("method", "COLUMN")
    is_bp = method.upper() == "BP"
    x_min, x_max = (0, 10) if is_bp else (0, 70)
    fracs = _ana_get_fractions(method)
    seq_name = data.get("seq_name", "")

    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor('white')
    fig.text(0.5, 0.97, "CROMATOGRAMES DOC SUPERPOSATS", ha='center', va='top',
             fontsize=14, fontweight='bold', color=COLORS["primary"])
    fig.text(0.5, 0.94, f"{seq_name} — {len(samples_grouped)} mostres",
             ha='center', va='top', fontsize=9, color=COLORS["text_secondary"])

    ax = fig.add_axes([0.08, 0.10, 0.65, 0.80])
    cmap = plt.cm.tab20
    regular_names, control_names = _ana_classify_samples(samples_grouped)

    for idx, name in enumerate(regular_names):
        sg = samples_grouped[name]
        sel = (sg.get("selected") or {}).get("doc", "1")
        if sel == "none":
            continue
        rep = (sg.get("replicas") or {}).get(sel, {})
        t = rep.get("t_doc")
        y = rep.get("y_doc_net")
        if t is None or y is None:
            continue
        t, y = np.asarray(t), np.asarray(y)
        color = cmap(idx % 20)
        ax.plot(t, y, color=color, lw=0.6, label=name[:18])

    for name in control_names:
        sg = samples_grouped[name]
        sel = (sg.get("selected") or {}).get("doc", "1")
        if sel == "none":
            continue
        rep = (sg.get("replicas") or {}).get(sel, {})
        t = rep.get("t_doc")
        y = rep.get("y_doc_net")
        if t is None or y is None:
            continue
        t, y = np.asarray(t), np.asarray(y)
        ax.plot(t, y, color='#999', lw=0.4, ls='--', label=name[:18])

    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("Temps (min)", fontsize=8)
    ax.set_ylabel("DOC (ppb)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.2, lw=0.3)

    if not is_bp and fracs:
        for _fn, fi in fracs:
            s = fi['start']
            if 0 < s <= x_max:
                ax.axvline(s, color='#999', ls=':', lw=0.4, zorder=0)

    n_total = len(regular_names) + len(control_names)
    ncol = 2 if n_total > 20 else 1
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=5,
              ncol=ncol, framealpha=0.7, handlelength=1.2, borderaxespad=0)

    draw_report_footer(fig, page_num)
    pdf.savefig(fig, dpi=150)
    plt.close(fig)
    return page_num + 1


def _ana_draw_a254_overlay_page(pdf, data, page_num):
    """Pàgina overlay: tots els cromatogrames A254 superposats."""
    samples_grouped = data.get("samples_grouped", {})
    method = data.get("method", "COLUMN")
    is_bp = method.upper() == "BP"
    x_min, x_max = (0, 10) if is_bp else (0, 70)
    fracs = _ana_get_fractions(method)
    seq_name = data.get("seq_name", "")

    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor('white')
    fig.text(0.5, 0.97, "CROMATOGRAMES A254 SUPERPOSATS", ha='center', va='top',
             fontsize=14, fontweight='bold', color=COLORS["primary"])
    fig.text(0.5, 0.94, f"{seq_name} — {len(samples_grouped)} mostres",
             ha='center', va='top', fontsize=9, color=COLORS["text_secondary"])

    ax = fig.add_axes([0.08, 0.10, 0.65, 0.80])
    cmap = plt.cm.tab20
    regular_names, control_names = _ana_classify_samples(samples_grouped)

    any_plotted = False
    for idx, name in enumerate(regular_names):
        sg = samples_grouped[name]
        sel = (sg.get("selected") or {}).get("dad", (sg.get("selected") or {}).get("doc", "1"))
        if sel == "none":
            continue
        rep = (sg.get("replicas") or {}).get(sel, {})
        df_dad = rep.get("df_dad")
        if df_dad is None or not hasattr(df_dad, 'columns'):
            continue
        wl_col = None
        for candidate in ['254', 254, 'A254']:
            if candidate in df_dad.columns:
                wl_col = candidate
                break
        if wl_col is None:
            continue
        t_col = 'time (min)' if 'time (min)' in df_dad.columns else df_dad.columns[0]
        color = cmap(idx % 20)
        ax.plot(df_dad[t_col].values, df_dad[wl_col].values,
                color=color, lw=0.6, label=name[:18])
        any_plotted = True

    for name in control_names:
        sg = samples_grouped[name]
        sel = (sg.get("selected") or {}).get("dad", (sg.get("selected") or {}).get("doc", "1"))
        if sel == "none":
            continue
        rep = (sg.get("replicas") or {}).get(sel, {})
        df_dad = rep.get("df_dad")
        if df_dad is None or not hasattr(df_dad, 'columns'):
            continue
        wl_col = None
        for candidate in ['254', 254, 'A254']:
            if candidate in df_dad.columns:
                wl_col = candidate
                break
        if wl_col is None:
            continue
        t_col = 'time (min)' if 'time (min)' in df_dad.columns else df_dad.columns[0]
        ax.plot(df_dad[t_col].values, df_dad[wl_col].values,
                color='#999', lw=0.4, ls='--', label=name[:18])
        any_plotted = True

    if not any_plotted:
        ax.text(0.5, 0.5, "Dades A254 no disponibles",
                ha='center', va='center', transform=ax.transAxes,
                fontsize=14, color=COLORS["text_secondary"])

    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("Temps (min)", fontsize=8)
    ax.set_ylabel("Absorbància 254 nm (mAU)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.2, lw=0.3)

    if not is_bp and fracs:
        for _fn, fi in fracs:
            s = fi['start']
            if 0 < s <= x_max:
                ax.axvline(s, color='#999', ls=':', lw=0.4, zorder=0)

    n_total = len(regular_names) + len(control_names)
    ncol = 2 if n_total > 20 else 1
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=5,
              ncol=ncol, framealpha=0.7, handlelength=1.2, borderaxespad=0)

    draw_report_footer(fig, page_num)
    pdf.savefig(fig, dpi=150)
    plt.close(fig)
    return page_num + 1


def _ana_draw_doc_areas_page(pdf, data, page_num):
    """Pàgina barres apilades: àrees DOC per fracció i mostra."""
    samples_grouped = data.get("samples_grouped", {})
    method = data.get("method", "COLUMN")
    is_bp = method.upper() == "BP"
    fracs = _ana_get_fractions(method)
    seq_name = data.get("seq_name", "")

    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor('white')
    fig.text(0.5, 0.97, "ÀREES DOC PER FRACCIÓ", ha='center', va='top',
             fontsize=14, fontweight='bold', color=COLORS["primary"])
    fig.text(0.5, 0.94, f"{seq_name} — {len(samples_grouped)} mostres",
             ha='center', va='top', fontsize=9, color=COLORS["text_secondary"])

    ax = fig.add_axes([0.08, 0.15, 0.84, 0.73])

    regular_names, control_names = _ana_classify_samples(samples_grouped)
    ordered_names = regular_names + control_names

    x_positions = np.arange(len(ordered_names))
    bar_width = 0.7

    if not is_bp and fracs:
        # Stacked bars per fraction
        frac_names = [fn for fn, _ in fracs]
        bottoms = np.zeros(len(ordered_names))

        for frac_name in _FRAC_ORDER:
            if frac_name not in frac_names:
                continue
            values = []
            for name in ordered_names:
                sg = samples_grouped[name]
                sel = (sg.get("selected") or {}).get("doc", "1")
                if sel == "none":
                    values.append(0)
                    continue
                rep = (sg.get("replicas") or {}).get(sel, {})
                doc_areas = (rep.get("areas") or {}).get("DOC", {})
                values.append(doc_areas.get(frac_name, 0))
            values = np.array(values, dtype=float)
            color = _FRAC_COLORS.get(frac_name, '#888')
            is_light_mask = [n in control_names for n in ordered_names]
            bar_colors = [('#C0C0C0' if il else color) for il in is_light_mask]
            ax.bar(x_positions, values, bar_width, bottom=bottoms,
                   color=bar_colors, edgecolor='white', linewidth=0.3,
                   label=frac_name)
            bottoms += values

        handles = [mpatches.Patch(color=_FRAC_COLORS.get(fn, '#888'), label=fn)
                   for fn in _FRAC_ORDER if fn in frac_names]
        ax.legend(handles=handles, loc='upper right', fontsize=7, framealpha=0.8)
    else:
        # Simple bars for BP
        values = []
        for name in ordered_names:
            sg = samples_grouped[name]
            sel = (sg.get("selected") or {}).get("doc", "1")
            if sel == "none":
                values.append(0)
                continue
            rep = (sg.get("replicas") or {}).get(sel, {})
            doc_areas = (rep.get("areas") or {}).get("DOC", {})
            values.append(doc_areas.get("total", 0))
        values = np.array(values, dtype=float)
        bar_colors = ['#C0C0C0' if n in control_names else COLORS["primary"]
                      for n in ordered_names]
        ax.bar(x_positions, values, bar_width, color=bar_colors,
               edgecolor='white', linewidth=0.3)
        bottoms = values

    # ppm annotation above each bar
    for i, name in enumerate(ordered_names):
        sg = samples_grouped[name]
        quant = sg.get("quantification") or {}
        ppm = quant.get("concentration_ppm_direct") or quant.get("concentration_ppm")
        if ppm and bottoms[i] > 0:
            ax.text(i, bottoms[i] + bottoms.max() * 0.01, f"{ppm:.1f}",
                    ha='center', va='bottom', fontsize=4.5, color=COLORS["text"])

    n_samples = len(ordered_names)
    label_size = 8 if n_samples <= 15 else (6 if n_samples <= 30 else 5)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([n[:15] for n in ordered_names], rotation=45,
                       ha='right', fontsize=label_size)
    ax.set_ylabel("Àrea DOC", fontsize=8)
    ax.tick_params(axis='y', labelsize=7)
    ax.grid(True, axis='y', alpha=0.2, lw=0.3)

    draw_report_footer(fig, page_num)
    pdf.savefig(fig, dpi=150)
    plt.close(fig)
    return page_num + 1


def _ana_draw_sample_header(fig, sg, sample_name):
    """Bloc informatiu a la part superior de la pàgina per mostra (y=0.97→0.88)."""
    from hpsec_warnings import ANOMALY_CATALOG as _AC

    selected = sg.get("selected") or {}
    quant = sg.get("quantification") or {}
    comparison = sg.get("comparison") or {}
    sample_valid = sg.get("sample_valid", True)

    # Line 1: name | ppm Direct | ppm UIB | HCI
    title_color = COLORS["danger"] if not sample_valid else COLORS["primary"]
    fig.text(0.04, 0.97, sample_name, va='top', fontsize=12,
             fontweight='bold', color=title_color)

    line1_parts = []
    ppm_d = quant.get("concentration_ppm_direct") or quant.get("concentration_ppm")
    ppm_u = quant.get("concentration_ppm_uib")
    if ppm_d is not None:
        line1_parts.append(f"DOC: {ppm_d:.2f} ppm")
    if ppm_u is not None:
        line1_parts.append(f"UIB: {ppm_u:.2f} ppm")
    hci = quant.get("hci")
    hci_char = quant.get("hci_character", "")
    if hci is not None:
        line1_parts.append(f"HCI={hci:.2f} ({hci_char})")
    if not sample_valid:
        line1_parts.append("NO VÀLIDA")
    if line1_parts:
        fig.text(0.96, 0.97, "  |  ".join(line1_parts), ha='right', va='top',
                 fontsize=8, color=COLORS["text"])

    # Line 2: anomaly badges (compact, max 6)
    doc_sel = selected.get("doc", "1")
    rep_data = (sg.get("replicas") or {}).get(doc_sel, {})
    anomalies = rep_data.get("anomalies", [])

    badge_parts = []
    for anom in anomalies[:6]:
        if isinstance(anom, dict):
            code = anom.get("code", "")
            repaired = anom.get("repaired", False)
        else:
            repaired = "_REPAIRED" in str(anom)
            code = str(anom).replace("_REPAIRED", "")
        entry = _AC.get(code, {})
        icon = entry.get("icon", code[:3])
        label = entry.get("label", code)
        sev = entry.get("severity")
        if repaired:
            badge_parts.append((f"{icon}* {label}", '#888'))
        elif sev and sev.value == "blocker":
            badge_parts.append((f"{icon} {label}", COLORS["danger"]))
        elif sev and sev.value == "warning":
            badge_parts.append((f"{icon} {label}", COLORS["warning"]))
        else:
            badge_parts.append((f"{icon} {label}", COLORS["primary"]))

    if badge_parts:
        x_pos = 0.04
        for text, color in badge_parts:
            bbox = dict(boxstyle="round,pad=0.15", facecolor=color,
                        edgecolor='none', alpha=0.15)
            fig.text(x_pos, 0.935, text, va='top', fontsize=5.5,
                     color=color, bbox=bbox)
            x_pos += len(text) * 0.005 + 0.015
            if x_pos > 0.90:
                break

    # Line 3: metrics
    doc_comp = comparison.get("doc") or {}
    dad_comp = comparison.get("dad") or {}
    r2_doc = doc_comp.get("pearson", 0)
    r2_dad = dad_comp.get("pearson_min", 0)
    snr_info = rep_data.get("snr_info") or {}
    snr_d = snr_info.get("snr_direct", 0)
    snr_info_dad = rep_data.get("snr_info_dad") or {}
    snr_254 = snr_info_dad.get("A254", {}).get("snr", 0)
    area_diff_raw = rep_data.get("area_diff_pct")
    area_diff = area_diff_raw.get("total") if isinstance(area_diff_raw, dict) else area_diff_raw

    metrics = []
    if r2_doc > 0:
        clr = '#C62828' if r2_doc < 0.990 else COLORS["text_secondary"]
        metrics.append((f"R²_DOC={r2_doc:.4f}", clr))
    if r2_dad > 0:
        clr = '#C62828' if r2_dad < 0.990 else COLORS["text_secondary"]
        metrics.append((f"R²_DAD={r2_dad:.4f}", clr))
    if area_diff is not None:
        metrics.append((f"ΔA={area_diff:.1f}%", COLORS["text_secondary"]))
    if snr_d > 0:
        clr = '#C62828' if snr_d < 10 else COLORS["text_secondary"]
        metrics.append((f"SNR_DOC={snr_d:.0f}", clr))
    if snr_254 > 0:
        clr = '#C62828' if snr_254 < 10 else COLORS["text_secondary"]
        metrics.append((f"SNR_254={snr_254:.0f}", clr))
    metrics.append((f"R_DOC={doc_sel}  R_DAD={selected.get('dad', '?')}",
                    COLORS["text_secondary"]))

    if metrics:
        x_pos = 0.04
        for text, color in metrics:
            fig.text(x_pos, 0.905, text, va='top', fontsize=6, color=color)
            x_pos += len(text) * 0.005 + 0.01


def _ana_downsample_2d(t, data_2d, target_dt):
    """Downsample matriu 2D per bin-average (còpia local de hpsec_export)."""
    t = np.asarray(t, dtype=float)
    if hasattr(data_2d, 'values'):
        data_2d = data_2d.values
    data_2d = np.asarray(data_2d, dtype=float)

    dt_median = np.median(np.diff(t))
    if dt_median >= target_dt * 0.8:
        return t, data_2d

    t_min, t_max_val = t[0], t[-1]
    bins = np.arange(t_min, t_max_val + target_dt, target_dt)
    n_bins = len(bins) - 1
    if n_bins < 2:
        return t, data_2d

    t_new = np.zeros(n_bins)
    data_new = np.zeros((n_bins, data_2d.shape[1] if data_2d.ndim > 1 else 1))
    if data_2d.ndim == 1:
        data_2d = data_2d.reshape(-1, 1)

    indices = np.digitize(t, bins) - 1
    indices = np.clip(indices, 0, n_bins - 1)

    for b in range(n_bins):
        mask = indices == b
        if mask.any():
            t_new[b] = np.mean(t[mask])
            data_new[b] = np.mean(data_2d[mask], axis=0)
        else:
            t_new[b] = (bins[b] + bins[b + 1]) / 2
            nearest = np.argmin(np.abs(t - t_new[b]))
            data_new[b] = data_2d[nearest]

    return t_new, data_new


def _ana_draw_heatmap_page(pdf, sg, sample_name, method, page_num):
    """Pàgina heatmap DAD espectral: RAW + normalitzat per λ."""
    selected = sg.get("selected") or {}
    doc_sel = selected.get("doc", "1")
    rep_data = (sg.get("replicas") or {}).get(doc_sel, {})
    dad_path = rep_data.get("dad_export3d_path")
    if not dad_path or not os.path.exists(dad_path):
        return page_num

    try:
        from hpsec_import import llegir_dad_export3d
        df_full, status = llegir_dad_export3d(dad_path, wavelengths_to_keep=None)
        if df_full is None or df_full.empty:
            return page_num
    except Exception:
        return page_num

    is_bp = method.upper() == "BP"
    fracs = _ana_get_fractions(method)
    x_min, x_max_plot = (0, 10) if is_bp else (0, 70)

    # Extract time and wavelength data
    t_col = 'time (min)' if 'time (min)' in df_full.columns else df_full.columns[0]
    t_full = df_full[t_col].values
    wl_cols = [c for c in df_full.columns if c != t_col]
    wl_vals = []
    for c in wl_cols:
        try:
            wl_vals.append(float(c))
        except (ValueError, TypeError):
            pass
    if len(wl_vals) < 5:
        return page_num

    wl_cols_numeric = [c for c in wl_cols if _is_numeric_col(c)]
    wl_floats = np.array([float(c) for c in wl_cols_numeric])
    data_2d = df_full[wl_cols_numeric].values

    # Downsample to dt=0.1 min for PDF
    t_ds, data_ds = _ana_downsample_2d(t_full, data_2d, 0.1)

    # Crop to plot range
    mask = (t_ds >= x_min) & (t_ds <= x_max_plot)
    t_ds = t_ds[mask]
    data_ds = data_ds[mask]

    if len(t_ds) < 3:
        return page_num

    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor('white')
    fig.text(0.5, 0.97, f"HEATMAP DAD ESPECTRAL — {sample_name}",
             ha='center', va='top', fontsize=12, fontweight='bold',
             color=COLORS["primary"])

    gs = fig.add_gridspec(1, 2, wspace=0.30, left=0.08, right=0.92,
                          top=0.90, bottom=0.10)

    # Left: RAW heatmap
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.pcolormesh(t_ds, wl_floats, data_ds.T,
                         shading='auto', cmap='viridis', rasterized=True)
    ax1.set_xlabel("Temps (min)", fontsize=8)
    ax1.set_ylabel("λ (nm)", fontsize=8)
    ax1.set_title("Absorbància RAW (mAU)", fontsize=9)
    ax1.tick_params(labelsize=6)
    cb1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cb1.ax.tick_params(labelsize=6)

    if not is_bp and fracs:
        for _fn, fi in fracs:
            s = fi['start']
            if 0 < s <= x_max_plot:
                ax1.axvline(s, color='white', ls=':', lw=0.5, alpha=0.6)

    # Right: normalized per-λ heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    data_norm = data_ds.copy()
    for j in range(data_norm.shape[1]):
        col_max = np.max(np.abs(data_norm[:, j]))
        if col_max > 0:
            data_norm[:, j] /= col_max

    im2 = ax2.pcolormesh(t_ds, wl_floats, data_norm.T,
                         shading='auto', cmap='inferno', vmin=0, vmax=1,
                         rasterized=True)
    ax2.set_xlabel("Temps (min)", fontsize=8)
    ax2.set_ylabel("λ (nm)", fontsize=8)
    ax2.set_title("Normalitzat per λ (0-1)", fontsize=9)
    ax2.tick_params(labelsize=6)
    cb2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cb2.ax.tick_params(labelsize=6)

    if not is_bp and fracs:
        for _fn, fi in fracs:
            s = fi['start']
            if 0 < s <= x_max_plot:
                ax2.axvline(s, color='white', ls=':', lw=0.5, alpha=0.6)

    draw_report_footer(fig, page_num)
    pdf.savefig(fig, dpi=150)
    plt.close(fig)
    return page_num + 1


def _is_numeric_col(col_name):
    """Comprova si un nom de columna és numèric (wavelength)."""
    try:
        float(col_name)
        return True
    except (ValueError, TypeError):
        return False


def generate_analysis_report(seq_path, output_path=None, analysis_data=None):
    """
    Genera PDF d'analisi complet.

    Args:
        seq_path: Ruta a la carpeta SEQ
        output_path: Ruta de sortida (default: CHECK/)
        analysis_data: Dades d'analisi en memoria (opcional)

    Returns:
        Path del PDF generat o None si error
    """
    from pathlib import Path

    if analysis_data is None:
        data = _ana_load_result(seq_path)
    else:
        data = analysis_data

    if not data:
        print(f"No s'han trobat dades d'analisi a {seq_path}")
        return None

    seq_name = data.get("seq_name", Path(seq_path).name)
    samples_grouped = data.get("samples_grouped", {})

    if not samples_grouped:
        print(f"No hi ha mostres agrupades a {seq_path}")
        return None

    if output_path is None:
        output_path = Path(seq_path) / "CHECK"
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    pdf_path = output_path / f"REPORT_Analysis_{seq_name}.pdf"
    if pdf_path.exists():
        try:
            with open(pdf_path, 'a'):
                pass
        except PermissionError:
            timestamp = datetime.now().strftime("%H%M%S")
            pdf_path = output_path / f"REPORT_Analysis_{seq_name}_{timestamp}.pdf"
            print(f"  Fitxer bloquejat, usant: {pdf_path.name}")

    print(f"Generant {pdf_path}...")
    print(f"  Mostres: {len(samples_grouped)}")

    with PdfPages(pdf_path) as pdf:
        _ana_draw_page1_summary(pdf, data, seq_name)
        next_page = _ana_draw_results_pages(pdf, data, page_start=2)
        # Overlay pages (sequence-level)
        next_page = _ana_draw_doc_overlay_page(pdf, data, next_page)
        next_page = _ana_draw_a254_overlay_page(pdf, data, next_page)
        next_page = _ana_draw_doc_areas_page(pdf, data, next_page)
        # Per-sample chromatograms (enhanced + heatmaps)
        next_page = _ana_draw_chromatogram_pages(pdf, data, page_start=next_page)
        _ana_draw_anomalies_page(pdf, data, page_num=next_page)

    print(f"  [OK] PDF generat: {pdf_path}")
    return str(pdf_path)


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
