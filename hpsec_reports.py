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


# NOTA: is_khp s'ha mogut a hpsec_import.py (2026-01-29) - importat a dalt


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


def draw_footer(fig, text="", version=""):
    """Dibuixa peu de pàgina minimalista."""
    # Línia separadora
    fig.add_artist(plt.Line2D([0.05, 0.95], [0.02, 0.02],
                              color='lightgray', linewidth=0.5,
                              transform=fig.transFigure))

    # Data i versió
    date_str = datetime.now().strftime("%d/%m/%Y %H:%M")
    footer_left = f"Generat: {date_str}"
    if version:
        footer_left += f"  |  HPSEC Suite v{version}"
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

        draw_footer(fig, "LEQUIA · UdG", script_version)
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

            draw_footer(fig, version=script_version)
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

            draw_footer(fig, version=script_version)
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

        draw_footer(fig, "LEQUIA · UdG")
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

        draw_footer(fig, "LEQUIA · UdG")
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # =====================================================================
        # PÀGINA 3: Evolució temporal RF (historial KHP)
        # =====================================================================
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.patch.set_facecolor('white')

        draw_header(fig, "EVOLUCIÓ TEMPORAL RF", page_num=3, total_pages=5)

        ax = fig.add_axes([0.08, 0.12, 0.84, 0.72])

        # Filtrar entrades vàlides del historial
        mode_upper = mode.upper()
        hist_entries = []
        for entry in khp_history:
            if entry.get('mode', '').upper() != mode_upper:
                continue
            if not entry.get('valid_for_calibration', True):
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

        draw_footer(fig, "LEQUIA · UdG")
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

        draw_footer(fig, "LEQUIA · UdG")
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

        draw_footer(fig, "LEQUIA · UdG")
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

    t_doc = rep_data.get('t_doc') or rep_data.get('t')
    y_doc = rep_data.get('y_doc') or rep_data.get('y')
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
