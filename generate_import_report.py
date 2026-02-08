# -*- coding: utf-8 -*-
"""
generate_import_report.py - Genera PDF d'importacio
====================================================

Format professional, minimalista, estil GUI HPSEC Suite.
- Taules sense divisions verticals
- Horitzontals nomes a capcalera i final
- Marges nets, bona alineacio
- Logo STRs a primera pagina
- Cromatogrames raw integrats al PDF (R1+R2 per mostra)

Llegeix CHECK/data/import_manifest.json
Genera CHECK/REPORT_Import_{seq_name}.pdf
"""

import os
import json
import statistics
import numpy as np
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# =============================================================================
# CONFIGURACIO D'ESTIL - Copia de generate_calibration_report.py
# =============================================================================

COLORS = {
    "primary": "#2E86AB",
    "primary_dark": "#1A5276",
    "accent": "#27AE60",
    "warning": "#F39C12",
    "danger": "#E74C3C",
    "text": "#2C3E50",
    "text_secondary": "#7F8C8D",
    "background": "#FFFFFF",
    "surface": "#F8F9FA",
    "border": "#E5E7EB",
    "table_header": "#2E86AB",
    "table_row_alt": "#F8FAFC",
    # Cromatogrames
    "doc_direct": "#1f77b4",
    "doc_uib": "#2ca02c",
    "dad_254": "#d62728",
    # Files especials
    "khp_row": "#E8F8E8",
    "control_row": "#FFF9E6",
}

FONTS = {
    "title": {"family": "sans-serif", "size": 18, "weight": "bold"},
    "subtitle": {"family": "sans-serif", "size": 12, "weight": "normal"},
    "section": {"family": "sans-serif", "size": 11, "weight": "bold"},
    "body": {"family": "sans-serif", "size": 9, "weight": "normal"},
    "small": {"family": "sans-serif", "size": 8, "weight": "normal"},
    "mono": {"family": "monospace", "size": 9, "weight": "normal"},
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "grid.linewidth": 0.3,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.0,
})


# =============================================================================
# FUNCIONS AUXILIARS
# =============================================================================

def get_logo_path():
    """Retorna path al logo STRs."""
    base_dir = Path(__file__).parent
    logo_path = base_dir / "logo STRs.png"
    if logo_path.exists():
        return str(logo_path)
    return None


def format_value(val, fmt=".2f", suffix="", default="-"):
    """Formata un valor numeric."""
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

    row_colors: dict {row_index: color_hex} per files especials (KHP, Control)
    max_row_height: si definit, limita l'altura maxima de cada fila
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

        # Fons especial per files no-header
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

        # Fons capcalera
        if is_header:
            rect = plt.Rectangle((0.02, y - row_height), 0.96, row_height,
                                 facecolor=header_color, edgecolor='none')
            ax.add_patch(rect)

        # Linia sota capcalera
        if is_header:
            ax.axhline(y - row_height, xmin=0.02, xmax=0.98,
                        color=header_color, linewidth=1.5)

    # Linia final
    y_final = y_start - n_rows * row_height
    ax.axhline(y_final, xmin=0.02, xmax=0.98,
               color=COLORS["border"], linewidth=1)


def _draw_footer(fig, page_num):
    """Dibuixa peu de pagina estandard."""
    fig.text(0.5, 0.02, "Serveis Tecnics de Recerca - Universitat de Girona",
             ha='center', fontsize=8, color=COLORS["text_secondary"])
    fig.text(0.95, 0.02, str(page_num), ha='right', fontsize=8,
             color=COLORS["text_secondary"])


# =============================================================================
# CARREGAR DADES
# =============================================================================

def load_import_manifest(seq_path):
    """Carrega dades d'importacio des del JSON."""
    json_path = Path(seq_path) / "CHECK" / "data" / "import_manifest.json"
    if not json_path.exists():
        return None
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# =============================================================================
# HELPERS
# =============================================================================

def _reclassify_samples(samples):
    """Reclassifica tipus de mostres segons regles actuals de config.
    Necessari perquè manifests antics poden tenir classificacions obsoletes
    (ex: MQ com CONTROL en lloc de BLANK)."""
    try:
        from hpsec_import import is_khp, is_blank_injection, is_control_injection, is_reference_standard
        from hpsec_config import get_config
        config = get_config()
    except ImportError:
        return  # Si no es pot importar, mantenir tipus existents

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


def _compute_t0(samples):
    """Calcula t0: temps (min) des de connexió TOC fins primera injecció.
    Estimat a partir del row_start mínim i la resolució temporal."""
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


def _count_by_type(samples):
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


# =============================================================================
# PAGINA 1: RESUM (A4 portrait)
# =============================================================================

def _draw_page_summary(pdf, manifest, seq_name):
    """Dibuixa pagina 1 amb resum d'importacio."""
    fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
    fig.patch.set_facecolor('white')

    # --- CAPCALERA AMB LOGO ---
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

    # Linia separadora
    fig.add_artist(plt.Line2D([0.1, 0.9], [0.75, 0.75],
                              color=COLORS["primary"], linewidth=2,
                              transform=fig.transFigure))

    # --- DADES ---
    seq_info = manifest.get("sequence", {})
    summary = manifest.get("summary", {})
    master = manifest.get("master_file", {})
    samples = manifest.get("samples", [])

    data_mode = seq_info.get("data_mode", "DUAL")
    method = seq_info.get("method", "COLUMN")
    seq_date = seq_info.get("date", "-")
    uib_sens = seq_info.get("uib_sensitivity", "-")
    master_name = master.get("filename", "-")

    # Comptar per tipus
    type_counts = _count_by_type(samples)
    total_injections = sum(c["injections"] for c in type_counts.values())

    # --- TAULA RESUM ---
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
                       col_widths=[0.22, 0.28, 0.22, 0.28],
                       font_size=9)

    # --- TAULA DESGLOSSAMENT PER TIPUS ---
    type_table = [
        ["Tipus", "Mostres", "Injeccions", "% del total"],
    ]
    label_map = {
        "SAMPLE": "Mostres",
        "KHP": "Patrons QC/QA (KHP)",
        "PR": "Patrons de referencia",
        "CONTROL": "Controls",
        "BLANK": "Blancs",
    }
    for type_key in ["SAMPLE", "KHP", "PR", "CONTROL", "BLANK"]:
        c = type_counts[type_key]
        pct = f"{c['injections'] / total_injections * 100:.0f}%" if total_injections > 0 and c["injections"] > 0 else "0%"
        type_table.append([
            label_map[type_key], str(c["samples"]),
            str(c["injections"]), pct
        ])

    n_type_rows = len(type_table)
    type_height = 0.02 + 0.022 * n_type_rows
    ax_types = fig.add_axes([0.08, 0.59 - type_height, 0.84, type_height])
    draw_minimal_table(ax_types, type_table,
                       col_widths=[0.40, 0.18, 0.22, 0.20],
                       font_size=9)

    # --- TAULA COMPLETESA DADES ---
    y_comp_top = 0.59 - type_height - 0.025

    rep_direct = summary.get("replicas_with_direct", 0)
    rep_uib = summary.get("replicas_with_uib", 0)
    rep_dad = summary.get("replicas_with_dad", 0)

    # Recalcular si summary no te dades
    if rep_direct == 0 and total_injections > 0:
        for s in samples:
            for r in s.get("replicas", []):
                d = r.get("direct", {})
                if d and d.get("n_points", 0) > 0:
                    rep_direct += 1
                u = r.get("uib", {})
                if u and u.get("n_points", 0) > 0:
                    rep_uib += 1
                dd = r.get("dad", {})
                if dd and dd.get("n_points", 0) > 0:
                    rep_dad += 1

    def pct_str(n, total):
        if total == 0:
            return "0%"
        return f"{n}/{total} ({n / total * 100:.0f}%)"

    completeness_data = [
        ["Senyal", "Dades"],
        ["DOC Direct", pct_str(rep_direct, total_injections)],
    ]
    if data_mode in ["DUAL", "UIB"]:
        completeness_data.append(
            ["DOC UIB", pct_str(rep_uib, total_injections)]
        )
    completeness_data.append(
        ["DAD", pct_str(rep_dad, total_injections)]
    )

    n_comp_rows = len(completeness_data)
    comp_height = 0.02 + 0.022 * n_comp_rows
    ax_comp = fig.add_axes([0.08, y_comp_top - comp_height, 0.84, comp_height])
    draw_minimal_table(ax_comp, completeness_data,
                       col_widths=[0.40, 0.60],
                       font_size=9)

    # --- ESTADISTIQUES INJECCIO ---
    y_stats_top = y_comp_top - comp_height - 0.025

    # Recollir t_max de totes les injeccions (DOC direct), ordenades per line_num
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
            # Comptar timeouts
            has_to = d.get("has_timeout", False)
            if has_to:
                sev = d.get("timeout_severity", "WARNING")
                if sev == "CRITICAL":
                    n_timeouts_crit += 1
                else:
                    n_timeouts_warn += 1

    # Ordenar per line_num i excloure la darrera injecció
    all_tmax.sort(key=lambda x: x[0])
    tmax_values = [t for _, t in all_tmax]
    tmax_no_last = tmax_values[:-1] if len(tmax_values) > 1 else tmax_values

    median_tmax = statistics.median(tmax_no_last) if tmax_no_last else 0

    # t0: temps des de connexio TOC fins primera injeccio
    t0 = _compute_t0(samples)

    # Formatar timeouts
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
                       col_widths=[0.22, 0.28, 0.22, 0.28],
                       font_size=9)

    y_pos = y_stats_top - stats_height - 0.03

    # --- AVISOS (inclou orfes) ---
    fig.text(0.08, y_pos, "Avisos:",
             fontsize=10, fontweight='bold', color=COLORS["text"])
    y_pos -= 0.025

    has_any_warning = False

    # Fitxers orfes
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

    # Warnings reals (filtrar missatges interns irrellevants)
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

    # Peu de pagina
    _draw_footer(fig, 1)

    pdf.savefig(fig, dpi=150)
    plt.close(fig)


# =============================================================================
# PAGINES 2+: TAULA DETALLADA (A4 landscape)
# =============================================================================

def _build_sorted_injections(samples):
    """
    Construeix llista plana d'injeccions ordenada:
    Mostres normals -> KHP -> Controls
    """
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


def _draw_detail_pages(pdf, manifest, page_counter):
    """Dibuixa pagines amb taula detallada de mostres (landscape).
    La taula sempre ocupa tota la pagina (row_height s'adapta)."""
    samples = manifest.get("samples", [])
    seq_info = manifest.get("sequence", {})
    data_mode = seq_info.get("data_mode", "DUAL")
    seq_name = seq_info.get("name", "")

    injections = _build_sorted_injections(samples)

    if not injections:
        return page_counter

    ROWS_PER_PAGE = 35

    # Definir columnes segons mode
    if data_mode == "DUAL":
        headers = ["#", "Mostra", "R", "Pts DOC", "Pts UIB", "Pts DAD",
                    "Fitxer UIB", "Fitxer DAD", "Vol", "t acq (min)"]
        col_widths = [0.04, 0.17, 0.03, 0.07, 0.07, 0.07, 0.17, 0.17, 0.05, 0.10]
    elif data_mode == "UIB":
        headers = ["#", "Mostra", "R", "Pts DOC", "Pts UIB",
                    "Fitxer UIB", "Vol", "t acq (min)"]
        col_widths = [0.04, 0.22, 0.04, 0.08, 0.08, 0.22, 0.06, 0.10]
    else:  # DIRECT
        headers = ["#", "Mostra", "R", "Pts DOC", "Pts DAD",
                    "Fitxer DAD", "Vol", "t acq (min)"]
        col_widths = [0.04, 0.22, 0.04, 0.08, 0.08, 0.22, 0.06, 0.10]

    # Paginar
    for page_start in range(0, len(injections), ROWS_PER_PAGE):
        page_counter += 1
        page_injections = injections[page_start:page_start + ROWS_PER_PAGE]

        fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
        fig.patch.set_facecolor('white')

        # Capcalera
        fig.text(0.5, 0.96, "DETALL D'INJECCIONS", ha='center', va='top',
                 fontsize=14, fontweight='bold', color=COLORS["primary"])
        fig.text(0.5, 0.93, f"{seq_name} - Pagina {page_counter}",
                 ha='center', va='top', fontsize=10,
                 color=COLORS["text_secondary"])
        fig.add_artist(plt.Line2D([0.05, 0.95], [0.91, 0.91],
                                  color=COLORS["primary"], linewidth=1,
                                  transform=fig.transFigure))

        # Construir dades de la taula
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

            # Temps d'adquisicio (t_max del DOC direct)
            t_max = direct.get("t_max", 0) or 0
            # Detectar timeout: n_timeouts del JSON (o recalcular via has_timeout)
            n_timeouts = direct.get("n_timeouts", 0)
            has_timeout = direct.get("has_timeout", False) or n_timeouts > 0

            if t_max > 0:
                t_str = f"{t_max:.1f}"
                if has_timeout:
                    t_str += f" [T:{n_timeouts}]"
            else:
                t_str = "-"

            if data_mode == "DUAL":
                row = [
                    str(row_num),
                    inj["name"][:22],
                    str(inj["replica"]),
                    str(pts_doc) if pts_doc else "-",
                    str(pts_uib) if pts_uib else "-",
                    str(pts_dad) if pts_dad else "-",
                    uib_file,
                    dad_file,
                    vol_str,
                    t_str,
                ]
            elif data_mode == "UIB":
                row = [
                    str(row_num),
                    inj["name"][:26],
                    str(inj["replica"]),
                    str(pts_doc) if pts_doc else "-",
                    str(pts_uib) if pts_uib else "-",
                    uib_file,
                    vol_str,
                    t_str,
                ]
            else:  # DIRECT
                row = [
                    str(row_num),
                    inj["name"][:26],
                    str(inj["replica"]),
                    str(pts_doc) if pts_doc else "-",
                    str(pts_dad) if pts_dad else "-",
                    dad_file,
                    vol_str,
                    t_str,
                ]

            table_data.append(row)

            # Colors per tipus
            data_row_idx = idx  # index dins table_data (header=0)
            if inj["type"] == "KHP":
                row_colors[data_row_idx] = COLORS["khp_row"]
            elif inj["type"] in ("CONTROL", "BLANK"):
                row_colors[data_row_idx] = COLORS["control_row"]

        # Dibuixar taula - cap row_height per evitar files gegants
        ax_table = fig.add_axes([0.03, 0.06, 0.94, 0.83])
        draw_minimal_table(ax_table, table_data, col_widths,
                           row_colors=row_colors, font_size=7,
                           max_row_height=0.9 / 36)

        _draw_footer(fig, page_counter)
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

    return page_counter


# =============================================================================
# PAGINES FINALS: CROMATOGRAMES RAW (A4 landscape)
# Layout: 2 columnes (R1 | R2), 1 fila per mostra
# =============================================================================

def _load_imported_data_for_chromatograms(seq_path):
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


def _plot_sample_chromatogram(ax, title, rep_data, show_legend=True):
    """
    Plotar cromatograma d'una replica amb DOC Direct + DOC UIB + DAD 254nm.
    Titol dins la grafica (upper-left). Llegenda unificada dins.
    Eix secundari DAD amb spine visible.
    """
    if rep_data is None:
        ax.text(0.5, 0.5, "Sense replica", ha='center', va='center',
                fontsize=8, color='#BBBBBB', transform=ax.transAxes)
        # Titol dins
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
                line, = ax.plot(t, y, color=COLORS["doc_uib"], linewidth=0.8,
                                alpha=0.8)
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
            ax2.set_ylabel("254nm", fontsize=5, color=COLORS["dad_254"],
                           labelpad=2)
            # Spine dret visible per DAD
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

        # Llegenda unificada dins la grafica (inclou DAD)
        if show_legend and all_handles:
            ax.legend(all_handles, all_labels, loc='upper right',
                      fontsize=5, frameon=True, framealpha=0.85,
                      edgecolor='none', fancybox=False, handlelength=1.2)

    # Titol dins la grafica (upper-left, al costat de l'eix Y)
    ax.text(0.03, 0.95, title, transform=ax.transAxes,
            fontsize=6.5, fontweight='bold', color=COLORS["text"],
            va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                      edgecolor='none', alpha=0.8))


def _draw_chromatogram_pages(pdf, seq_path, manifest, page_counter):
    """Dibuixa pagines amb cromatogrames: 2 per fila (R1+R2), 1 fila per mostra."""
    imported = _load_imported_data_for_chromatograms(seq_path)
    if not imported:
        return page_counter

    samples_dict = imported.get("samples", {})
    if not samples_dict:
        return page_counter

    seq_name = manifest.get("sequence", {}).get("name", "")
    manifest_samples = manifest.get("samples", [])

    # Recollir R1 i R2 per cada mostra
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

        # Nomes si te alguna dada en alguna replica
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

    ROWS_PER_PAGE = 4  # 4 mostres per pagina (4 files x 2 columnes)

    # Carpeta per guardar PNGs
    graphs_dir = Path(seq_path) / "CHECK" / "Graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)

    for page_start in range(0, len(sample_items), ROWS_PER_PAGE):
        page_counter += 1
        page_items = sample_items[page_start:page_start + ROWS_PER_PAGE]

        fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
        fig.patch.set_facecolor('white')

        # Capcalera
        fig.text(0.5, 0.96, "CROMATOGRAMES RAW", ha='center', va='top',
                 fontsize=14, fontweight='bold', color=COLORS["primary"])
        fig.text(0.5, 0.93, f"{seq_name} - Pagina {page_counter}",
                 ha='center', va='top', fontsize=10,
                 color=COLORS["text_secondary"])
        fig.add_artist(plt.Line2D([0.05, 0.95], [0.91, 0.91],
                                  color=COLORS["primary"], linewidth=1,
                                  transform=fig.transFigure))

        # Grid: 4 files x 2 columnes, gap entre R1 i R2
        row_height = 0.20
        col_width = 0.39
        left_margins = [0.07, 0.54]
        top_start = 0.70

        for row_idx, (name, r1, r2) in enumerate(page_items):
            bottom = top_start - row_idx * row_height

            # R1
            ax1 = fig.add_axes([left_margins[0], bottom + 0.025,
                                col_width, row_height - 0.045])
            _plot_sample_chromatogram(ax1, f"{name} R1", r1)

            # R2
            ax2 = fig.add_axes([left_margins[1], bottom + 0.025,
                                col_width, row_height - 0.045])
            _plot_sample_chromatogram(ax2, f"{name} R2", r2)

        _draw_footer(fig, page_counter)
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

    # Guardar PNG amb tots els cromatogrames (primera pagina)
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

            row_height = 0.20
            col_width = 0.39
            left_margins = [0.07, 0.54]
            top_start = 0.70

            for row_idx, (name, r1, r2) in enumerate(first_items):
                bottom = top_start - row_idx * row_height
                ax1 = save_fig.add_axes([left_margins[0], bottom + 0.025,
                                         col_width, row_height - 0.045])
                _plot_sample_chromatogram(ax1, f"{name} R1", r1)
                ax2 = save_fig.add_axes([left_margins[1], bottom + 0.025,
                                         col_width, row_height - 0.045])
                _plot_sample_chromatogram(ax2, f"{name} R2", r2)

            png_path = graphs_dir / f"import_chromatograms_{seq_name}.png"
            save_fig.savefig(str(png_path), dpi=150, bbox_inches='tight',
                             facecolor='white')
            plt.close(save_fig)
    except Exception as e:
        print(f"  [WARNING] No s'ha pogut guardar PNG de cromatogrames: {e}")

    return page_counter


# =============================================================================
# GENERACIO DEL PDF
# =============================================================================

def generate_import_report(seq_path, output_path=None):
    """
    Genera PDF d'importacio amb estil professional.

    Args:
        seq_path: Ruta a la carpeta SEQ
        output_path: Ruta de sortida (default: CHECK/)

    Returns:
        Path del PDF generat o None si hi ha error
    """
    manifest = load_import_manifest(seq_path)
    if not manifest:
        print(f"No s'han trobat dades d'importacio a {seq_path}")
        return None

    seq_name = manifest.get("sequence", {}).get("name", Path(seq_path).name)

    # Reclassificar mostres segons regles actuals (BLANK vs CONTROL)
    _reclassify_samples(manifest.get("samples", []))

    # Path de sortida
    if output_path is None:
        output_path = Path(seq_path) / "CHECK"
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    pdf_path = output_path / f"REPORT_Import_{seq_name}.pdf"

    # Si el fitxer esta bloquejat, usar nom amb timestamp
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
        # Pagina 1: Resum
        _draw_page_summary(pdf_doc, manifest, seq_name)

        # Pagines 2+: Taula detallada
        page_counter = 1
        page_counter = _draw_detail_pages(pdf_doc, manifest, page_counter)

        # Pagines finals: Cromatogrames raw (R1+R2 per mostra)
        page_counter = _draw_chromatogram_pages(
            pdf_doc, seq_path, manifest, page_counter
        )

    print(f"  [OK] PDF generat: {pdf_path}")
    return str(pdf_path)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Test amb SEQ real."""
    base_path = Path("C:/Users/Lequia/Desktop/Dades3")

    test_seqs = [
        "286_SEQ_BP",
        "285_SEQ",
    ]

    results = []
    for seq_name in test_seqs:
        seq_path = base_path / seq_name
        if seq_path.exists():
            print(f"\n{'=' * 60}")
            print(f"Processant: {seq_name}")
            print('=' * 60)
            result = generate_import_report(seq_path)
            if result:
                results.append(result)
        else:
            print(f"No trobat: {seq_path}")

    print(f"\n{'=' * 60}")
    print(f"COMPLETAT: {len(results)} PDFs generats")
    print('=' * 60)
    for r in results:
        print(f"  {r}")

    if results:
        import subprocess
        subprocess.Popen(['start', '', results[0]], shell=True)


if __name__ == "__main__":
    main()
