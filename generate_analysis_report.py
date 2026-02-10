# -*- coding: utf-8 -*-
"""
generate_analysis_report.py - INFORME D'ANÀLISI PDF
=====================================================

PDF professional multi-pàgina:
  P1 (portrait)  - Resum seqüència: info, estadístiques, SNR
  P2+ (landscape) - Taula de resultats (13 columnes del panel)
  P3+ (landscape) - Cromatogrames per mostra (DOC|UIB + DAD parells)
  P final (portrait) - Anomalies i warnings

Estil consistent amb generate_import_report.py i generate_calibration_report.py.
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# =============================================================================
# CONFIGURACIO D'ESTIL - Identic a generate_calibration_report.py
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

# Graph colors (same as dialogs.py)
C1 = '#1565C0'      # R1 Direct (blue)
C2 = '#E65100'      # R2 Direct (orange)
C_UIB = '#2E7D32'   # R1 UIB (dark green)
C_UIB2 = '#66BB6A'  # R2 UIB (light green)
LW = 0.7


# =============================================================================
# FUNCIONS AUXILIARS (reutilitzades de generate_calibration_report.py)
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
    """Taula minimalista amb nomes linies horitzontals."""
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


def _draw_footer(fig, page_num):
    """Peu de pagina estandard."""
    fig.text(0.5, 0.02, "Serveis Tecnics de Recerca - Universitat de Girona",
             ha='center', fontsize=8, color=COLORS["text_secondary"])
    fig.text(0.95, 0.02, str(page_num), ha='right', fontsize=8,
             color=COLORS["text_secondary"])


# =============================================================================
# CARREGAR DADES
# =============================================================================

def _load_analysis_result(seq_path):
    """Carrega analysis_result.json de la SEQ."""
    p = Path(seq_path) / "CHECK" / "data" / "analysis_result.json"
    if not p.exists():
        return None
    try:
        with open(p, 'r', encoding='utf-8') as f:
            data = json.load(f)
        _restore_arrays(data)
        return data
    except Exception as e:
        print(f"Error carregant analysis_result: {e}")
        return None


def _restore_arrays(data):
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
    for sample in data.get("khp_samples", []):
        _restore_sample(sample)
    for sample in data.get("control_samples", []):
        _restore_sample(sample)
    for sample_data in data.get("samples_grouped", {}).values():
        for rep_data in sample_data.get("replicas", {}).values():
            _restore_sample(rep_data)


def _get_status_color(anomalies, score=1.0):
    """Retorna color per una mostra segons anomalies."""
    critical = ["BATMAN_DIRECT", "BATMAN_UIB", "NO_PEAK", "TIMEOUT_IN_PEAK"]
    has_critical = any(a in anomalies for a in critical)
    has_repaired = any("_REPAIRED" in a for a in anomalies)

    if has_critical:
        return COLORS["danger"]
    elif has_repaired or score < 0.7:
        return COLORS["warning"]
    return COLORS["accent"]


def _status_text(anomalies, sample_valid=True):
    """Text curt d'estat per una mostra."""
    if not sample_valid:
        return "NO VÀL"
    parts = []
    if any(a in anomalies for a in ["BATMAN_DIRECT", "BATMAN_UIB"]):
        parts.append("B")
    if any("_REPAIRED" in a for a in anomalies):
        parts.append("B*")
    if "NO_PEAK" in anomalies:
        parts.append("!")
    if "TIMEOUT_IN_PEAK" in anomalies:
        parts.append("T")
    if "BELOW_LOD" in anomalies:
        parts.append("<LOD")
    elif "BELOW_LOQ" in anomalies:
        parts.append("<LOQ")
    return " ".join(parts) if parts else "OK"


# =============================================================================
# PAGINA 1: RESUM SEQÜÈNCIA (A4 portrait)
# =============================================================================

def _draw_page1_summary(pdf, data, seq_name):
    """Pagina 1: resum general de la seqüència analitzada."""
    fig = plt.figure(figsize=(8.27, 11.69))
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

    fig.text(0.5, 0.85, "INFORME D'ANÀLISI", ha='center', va='top',
             fontsize=18, fontweight='bold', color=COLORS["primary"])

    fig.text(0.5, 0.81, f"Seqüència {seq_name}", ha='center', va='top',
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

    fig.add_artist(plt.Line2D([0.1, 0.9], [0.75, 0.75],
                              color=COLORS["primary"], linewidth=2,
                              transform=fig.transFigure))

    # --- INFO SEQÜÈNCIA ---
    method = data.get("method", "-")
    data_mode = data.get("data_mode", "-")
    summary = data.get("summary", {})

    info_data = [
        ["Paràmetre", "Valor", "Paràmetre", "Valor"],
        ["Mètode", method, "Mode dades", data_mode],
        ["Data processament", date_processed or "-",
         "Mostres", str(summary.get("total_samples", 0))],
        ["Rèpliques totals", str(summary.get("total_replicas", 0)),
         "Pics vàlids", str(summary.get("valid_peaks", 0))],
        ["Amb anomalies", str(summary.get("with_anomalies", 0)),
         "Amb timeouts", str(summary.get("with_timeouts", 0))],
        ["Warnings rèpliques", str(summary.get("with_replica_warnings", 0)),
         "KHP processats", str(summary.get("n_khp", 0))],
    ]

    ax_info = fig.add_axes([0.08, 0.62, 0.84, 0.11])
    draw_minimal_table(ax_info, info_data,
                       col_widths=[0.20, 0.30, 0.20, 0.30],
                       font_size=9)

    # --- ESTADÍSTIQUES SNR ---
    fig.text(0.08, 0.60, "Estadístiques SNR", fontsize=11,
             fontweight='bold', color=COLORS["text"])

    # Recollir SNR de totes les mostres
    snr_directs = []
    snr_uibs = []
    snr_254s = []

    samples_grouped = data.get("samples_grouped", {})
    for sg_name, sg_data in samples_grouped.items():
        selected = sg_data.get("selected", {})
        doc_sel = selected.get("doc", "1")
        if doc_sel == "none":
            continue  # Saltar mostres sense selecció
        rep_data = sg_data.get("replicas", {}).get(doc_sel, {})

        snr_info = rep_data.get("snr_info", {})
        snr_d = snr_info.get("snr_direct")
        if snr_d and snr_d > 0:
            snr_directs.append(snr_d)
        snr_u = snr_info.get("snr_uib")
        if snr_u and snr_u > 0:
            snr_uibs.append(snr_u)

        dad_sel = selected.get("dad", "1")
        if dad_sel == "none":
            continue
        dad_data = sg_data.get("replicas", {}).get(dad_sel, {})
        snr_dad = dad_data.get("snr_info_dad", {}).get("A254", {}).get("snr")
        if snr_dad and snr_dad > 0:
            snr_254s.append(snr_dad)

    snr_table = [
        ["Senyal", "n", "Mitjana", "Min", "Max", "< LOQ (n)"],
    ]

    def _snr_row(label, vals):
        n = len(vals)
        if n == 0:
            return [label, "0", "-", "-", "-", "-"]
        arr = np.array(vals)
        n_below_loq = int(np.sum(arr < 10))
        return [label, str(n), f"{np.mean(arr):.0f}", f"{np.min(arr):.0f}",
                f"{np.max(arr):.0f}", str(n_below_loq)]

    snr_table.append(_snr_row("DOC Direct", snr_directs))
    if snr_uibs:
        snr_table.append(_snr_row("DOC UIB", snr_uibs))
    snr_table.append(_snr_row("A254 (DAD)", snr_254s))

    ax_snr = fig.add_axes([0.08, 0.48, 0.84, 0.10])
    draw_minimal_table(ax_snr, snr_table,
                       col_widths=[0.18, 0.10, 0.18, 0.18, 0.18, 0.18],
                       font_size=9)

    # --- DISTRIBUCIÓ D'ANOMALIES ---
    fig.text(0.08, 0.46, "Distribució d'anomalies", fontsize=11,
             fontweight='bold', color=COLORS["text"])

    anomaly_counts = {}
    for sg_name, sg_data in samples_grouped.items():
        for rep_key, rep_data in sg_data.get("replicas", {}).items():
            for anom in rep_data.get("anomalies", []):
                anomaly_counts[anom] = anomaly_counts.get(anom, 0) + 1

    if anomaly_counts:
        anom_table = [["Anomalia", "Vegades", "Severitat"]]
        critical_set = {"BATMAN_DIRECT", "BATMAN_UIB", "NO_PEAK", "TIMEOUT_IN_PEAK"}
        for anom, count in sorted(anomaly_counts.items(), key=lambda x: -x[1]):
            severity = "CRÍTIC" if anom in critical_set else "Avís"
            if "_REPAIRED" in anom:
                severity = "Reparat"
            anom_table.append([anom, str(count), severity])

        n_anom_rows = len(anom_table)
        table_h = min(0.015 * (n_anom_rows + 1), 0.20)
        ax_anom = fig.add_axes([0.08, 0.44 - table_h, 0.84, table_h])
        draw_minimal_table(ax_anom, anom_table,
                           col_widths=[0.50, 0.20, 0.30],
                           font_size=8)
    else:
        fig.text(0.08, 0.42, "Cap anomalia detectada",
                 fontsize=9, color=COLORS["accent"])

    # --- ESTAT GLOBAL ---
    warning_level = data.get("warning_level", "OK")
    badge_color = COLORS["accent"] if warning_level == "OK" else (
        COLORS["warning"] if warning_level == "WARNING" else COLORS["danger"])

    bbox_props = dict(boxstyle="round,pad=0.4", facecolor=badge_color,
                      edgecolor='none', alpha=0.9)
    fig.text(0.90, 0.60, warning_level, ha='center', va='center',
             fontsize=11, fontweight='bold', color='white',
             bbox=bbox_props)

    _draw_footer(fig, 1)
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


# =============================================================================
# PÀGINA 2+: TAULA DE RESULTATS (A4 landscape)
# =============================================================================

def _draw_results_pages(pdf, data, page_start=2):
    """Pàgines landscape amb taula de resultats (13 columnes)."""
    samples_grouped = data.get("samples_grouped", {})
    if not samples_grouped:
        return page_start

    # Preparar files
    table_rows = []
    for sample_name in sorted(samples_grouped.keys()):
        sg = samples_grouped[sample_name]
        selected = sg.get("selected", {})
        quant = sg.get("quantification", {})
        comparison = sg.get("comparison", {})
        doc_sel = selected.get("doc", "1")
        dad_sel = selected.get("dad", "1")
        doc_rep = sg.get("replicas", {}).get(doc_sel, {})
        dad_rep = sg.get("replicas", {}).get(dad_sel, {})
        sample_valid = sg.get("sample_valid", True)

        # DOC data
        areas = doc_rep.get("areas") or {}
        doc_areas = areas.get("DOC") or {}
        area_direct = doc_areas.get("total", 0)
        areas_uib = doc_rep.get("areas_uib") or {}
        area_uib = areas_uib.get("total", 0)

        ppm_d = quant.get("concentration_ppm_direct") or quant.get("concentration_ppm")
        ppm_u = quant.get("concentration_ppm_uib")

        snr_info = doc_rep.get("snr_info") or {}
        snr_d = snr_info.get("snr_direct", 0)

        # DAD data
        dad_areas = (dad_rep.get("areas") or {})
        area_254 = dad_areas.get("A254", {}).get("total", 0)
        snr_254 = dad_rep.get("snr_info_dad", {}).get("A254", {}).get("snr", 0)

        # R²
        r2_doc = comparison.get("doc", {}).get("pearson", 0) if comparison else 0
        r2_dad = comparison.get("dad", {}).get("pearson_min", 0) if comparison else 0

        # Anomalies
        anomalies = doc_rep.get("anomalies", [])
        status = _status_text(anomalies, sample_valid)
        color = _get_status_color(anomalies)

        table_rows.append({
            "row": [
                sample_name[:20],
                f"R{doc_sel}" if doc_sel != "none" else "Cap",
                f"R{dad_sel}" if dad_sel != "none" else "Cap",
                format_value(area_direct, ".0f"),
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
        })

    # Paginar: ~20 files per pàgina
    rows_per_page = 20
    headers = ["Mostra", "DOC", "DAD", "A_DOC", "ppm",
               "A_UIB", "ppm_U", "SNR", "A_254", "SNR_254",
               "R²_DOC", "R²_DAD", "Estat"]

    col_widths = [0.14, 0.04, 0.04, 0.08, 0.07,
                  0.08, 0.07, 0.06, 0.08, 0.06,
                  0.08, 0.08, 0.08]

    page_num = page_start
    for page_start_idx in range(0, len(table_rows), rows_per_page):
        page_rows = table_rows[page_start_idx:page_start_idx + rows_per_page]

        fig = plt.figure(figsize=(11.69, 8.27))  # landscape
        fig.patch.set_facecolor('white')

        fig.text(0.5, 0.96, "RESULTATS D'ANÀLISI",
                 ha='center', va='top', fontsize=14, fontweight='bold',
                 color=COLORS["primary"])
        fig.text(0.5, 0.93,
                 f"Mostres {page_start_idx + 1}-{page_start_idx + len(page_rows)} "
                 f"de {len(table_rows)}",
                 ha='center', va='top', fontsize=9,
                 color=COLORS["text_secondary"])

        # Muntar dades taula
        table_data = [headers]
        row_colors = {}
        for idx, entry in enumerate(page_rows):
            table_data.append(entry["row"])
            # Color coding per fila
            if entry["color"] == COLORS["danger"]:
                row_colors[idx + 1] = "#FDEDEC"
            elif entry["color"] == COLORS["warning"]:
                row_colors[idx + 1] = "#FEF9E7"

        n_rows = len(table_data)
        table_h = min(0.85, 0.035 * n_rows + 0.02)

        ax = fig.add_axes([0.03, 0.06, 0.94, table_h])
        draw_minimal_table(ax, table_data, col_widths=col_widths,
                           row_colors=row_colors, font_size=7.5)

        _draw_footer(fig, page_num)
        pdf.savefig(fig, dpi=150)
        plt.close(fig)
        page_num += 1

    return page_num


# =============================================================================
# PÀGINES CROMATOGRAMES (A4 landscape) — 1 mostra per pàgina
# =============================================================================

def _get_fractions(method):
    """Retorna llista de fraccions per al mètode."""
    try:
        from hpsec_config import get_config
        cfg = get_config()
        mode = "BP" if method.upper() == "BP" else "COLUMN"
        return cfg.get_all_fractions(mode)
    except Exception:
        return []


def _draw_fraction_table(ax, sg, is_bp, fracs, wl_cols):
    """Dibuixa taula de fraccions a un axes (matplotlib table)."""
    ax.axis('off')

    selected = sg.get("selected", {})
    rep_sel = selected.get("doc", "1")
    rep_data = sg.get("replicas", {}).get(rep_sel, {})

    sel_areas = rep_data.get("areas") or {}
    areas_uib = rep_data.get("areas_uib") or {}
    doc_areas = sel_areas.get("DOC", {})
    doc_total = doc_areas.get("total", 0)
    uib_total = areas_uib.get("total", 0)

    if is_bp:
        # BP: just total per signal
        col_labels = ["Senyal", "Àrea Total"]
        rows = []
        rows.append(["DOC", f"{doc_total:.1f}" if doc_total else "-"])
        if uib_total:
            rows.append(["UIB", f"{uib_total:.1f}"])
        for wl in wl_cols:
            wl_lbl = f"A{wl}" if not str(wl).startswith('A') else wl
            wl_area = sel_areas.get(wl_lbl, sel_areas.get(wl, {}))
            total = wl_area.get("total", 0) if isinstance(wl_area, dict) else 0
            rows.append([wl_lbl, f"{total:.1f}" if total else "-"])
    else:
        # COLUMN: per-fraction breakdown (% of total)
        x_max = 70
        col_labels = ["Senyal"]
        for fname, finfo in fracs:
            col_labels.append(f"{fname}\n({finfo['start']:g}-{finfo['end']:g})")
        col_labels.append(f"TOTAL\n(0-{x_max:g})")

        signal_names = ["DOC"]
        has_uib = uib_total > 0
        if has_uib:
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


def _draw_chromatogram_pages(pdf, data, page_start):
    """Genera pàgines de cromatogrames per cada mostra."""
    samples_grouped = data.get("samples_grouped", {})
    method = data.get("method", "COLUMN")
    is_bp = method.upper() == "BP"
    x_min, x_max = (0, 10) if is_bp else (0, 70)
    fracs = _get_fractions(method)

    page_num = page_start

    # Helper: add fraction vlines
    def _add_vlines(ax):
        if not is_bp and fracs:
            for _fn, fi in fracs:
                s = fi['start']
                if s > 0 and s <= x_max:
                    ax.axvline(s, color='#999', ls=':', lw=0.4, zorder=0)

    for sample_name in sorted(samples_grouped.keys()):
        sg = samples_grouped[sample_name]
        replicas = sg.get("replicas", {})
        if not replicas:
            continue

        selected = sg.get("selected", {})
        comparison = sg.get("comparison", {})
        doc_comp = comparison.get("doc", {}) if comparison else {}
        dad_comp = comparison.get("dad", {}) if comparison else {}
        quant = sg.get("quantification", {})

        rep_keys = sorted(replicas.keys())
        r1 = replicas.get(rep_keys[0], {})
        r2 = replicas.get(rep_keys[1], {}) if len(rep_keys) > 1 else None

        # Check if chromatogram data is available
        t1 = r1.get("t_doc")
        y1_d = r1.get("y_doc_net")
        y1_u = r1.get("y_doc_uib_net")
        has_chromatogram_data = t1 is not None and y1_d is not None

        # DAD wavelengths
        df_dad1 = r1.get("df_dad")
        wl_cols = []
        if df_dad1 is not None and hasattr(df_dad1, 'columns'):
            wl_cols = [c for c in df_dad1.columns if c != 'time (min)']
            wl_cols.sort(key=lambda x: int(x) if str(x).isdigit() else 0)

        # Pairs for grid
        pairs = []
        for i in range(0, len(wl_cols), 2):
            if i + 1 < len(wl_cols):
                pairs.append((wl_cols[i], wl_cols[i + 1]))
            else:
                pairs.append((wl_cols[i], None))

        n_graph_rows = 1 + len(pairs)  # DOC/UIB + DAD pairs

        # Create landscape figure
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.patch.set_facecolor('white')

        # Title
        sample_valid = sg.get("sample_valid", True)
        title_color = COLORS["danger"] if not sample_valid else COLORS["primary"]
        fig.text(0.5, 0.97, sample_name, ha='center', va='top',
                 fontsize=12, fontweight='bold', color=title_color)

        ppm_d = quant.get("concentration_ppm_direct") or quant.get("concentration_ppm")
        ppm_u = quant.get("concentration_ppm_uib")
        subtitle_parts = [f"R_DOC={selected.get('doc', '?')}",
                          f"R_DAD={selected.get('dad', '?')}"]
        if ppm_d:
            subtitle_parts.append(f"{ppm_d:.2f} ppm")
        if ppm_u:
            subtitle_parts.append(f"UIB: {ppm_u:.2f} ppm")

        # Bigaussian info (BP only)
        sel_rep = replicas.get(selected.get("doc", rep_keys[0]), r1)
        bg_doc = sel_rep.get("bigaussian_doc")
        if is_bp and bg_doc and bg_doc.get("valid"):
            bg_r2 = bg_doc.get("r2", 0)
            bg_asym = bg_doc.get("asymmetry", 1)
            bg_qual = bg_doc.get("quality", "?")
            subtitle_parts.append(
                f"BiG: R²={bg_r2:.3f} asym={bg_asym:.2f} ({bg_qual})")

        if not sample_valid:
            subtitle_parts.append("NO VÀLIDA")

        fig.text(0.5, 0.94, "  |  ".join(subtitle_parts),
                 ha='center', va='top', fontsize=8,
                 color=COLORS["text_secondary"])

        # R² values
        pearson_doc = doc_comp.get("pearson", 0)
        pearson_per_wl = dad_comp.get("pearson_per_wavelength", {})

        # If no chromatogram data, show message and fraction table only
        if not has_chromatogram_data:
            ax_msg = fig.add_axes([0.1, 0.35, 0.8, 0.5])
            ax_msg.axis('off')
            ax_msg.text(0.5, 0.6,
                        "Dades cromatogràfiques no disponibles",
                        ha='center', va='center', fontsize=14,
                        color=COLORS["text_secondary"])
            ax_msg.text(0.5, 0.45,
                        "(JSON antic sense arrays t_doc/y_doc_net/df_dad)",
                        ha='center', va='center', fontsize=9,
                        color='#bbb')

            # Still show fraction table if areas exist
            sel_areas = r1.get("areas") or {}
            if sel_areas:
                ax_tbl = fig.add_axes([0.07, 0.08, 0.86, 0.22])
                _draw_fraction_table(ax_tbl, sg, is_bp, fracs, wl_cols)

            _draw_footer(fig, page_num)
            pdf.savefig(fig, dpi=150)
            plt.close(fig)
            page_num += 1
            continue

        # GridSpec: graphs + fraction table
        h_graphs = [1.0] * n_graph_rows
        h_table = [1.8]  # fraction table row
        heights = h_graphs + h_table
        n_total_rows = n_graph_rows + 1

        gs = fig.add_gridspec(
            n_total_rows, 2,
            height_ratios=heights,
            hspace=0.35, wspace=0.25,
            top=0.90, bottom=0.07, left=0.08, right=0.92
        )

        # --- Row 0: DOC Direct | DOC UIB ---
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
                            color=C2, lw=LW, alpha=0.7,
                            label=f'R{rep_keys[1]}')

        ax_doc.set_ylabel("DOC Direct", fontsize=7)
        ax_doc.tick_params(labelsize=6, length=2, pad=1)
        ax_doc.grid(True, alpha=0.2, lw=0.3)
        ax_doc.set_xlim(x_min, x_max)
        _add_vlines(ax_doc)
        if ax_doc.get_legend_handles_labels()[1]:
            ax_doc.legend(loc='upper right', fontsize=5.5, ncol=1,
                          framealpha=0.7, handlelength=1.2)

        # R² + ppm annotation DOC (right, below legend)
        if pearson_doc > 0:
            ann_parts = [f"R²={pearson_doc:.4f}"]
            if ppm_d:
                ann_parts.append(f"{ppm_d:.2f} ppm")
            clr = '#C62828' if pearson_doc < 0.990 else '#555'
            ax_doc.text(0.99, 0.72, "  ".join(ann_parts),
                        transform=ax_doc.transAxes, fontsize=5,
                        color=clr, ha='right', va='top')

        # UIB
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

        # --- DAD rows ---
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

                # R² per wavelength (right side)
                wl_key = f"A{wl}" if not str(wl).startswith('A') else wl
                r2v = pearson_per_wl.get(wl_key, 0) or pearson_per_wl.get(str(wl), 0)
                if r2v and r2v > 0:
                    clr = '#C62828' if r2v < 0.990 else '#555'
                    ax.text(0.99, 0.92, f"R²={r2v:.4f}",
                            transform=ax.transAxes, fontsize=5,
                            color=clr, ha='right', va='top')

        # --- Fraction table (bottom, spans 2 columns) ---
        ax_tbl = fig.add_subplot(gs[n_graph_rows, :])
        _draw_fraction_table(ax_tbl, sg, is_bp, fracs, wl_cols)

        _draw_footer(fig, page_num)
        pdf.savefig(fig, dpi=150)
        plt.close(fig)
        page_num += 1

    return page_num


# =============================================================================
# PÀGINA FINAL: ANOMALIES I WARNINGS (A4 portrait)
# =============================================================================

def _draw_anomalies_page(pdf, data, page_num):
    """Pàgina final amb detall d'anomalies i warnings per mostra."""
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor('white')

    fig.text(0.5, 0.96, "ANOMALIES I WARNINGS",
             ha='center', va='top', fontsize=14, fontweight='bold',
             color=COLORS["primary"])

    # Recollir anomalies per mostra
    samples_grouped = data.get("samples_grouped", {})
    anomaly_rows = []

    for sample_name in sorted(samples_grouped.keys()):
        sg = samples_grouped[sample_name]
        sample_valid = sg.get("sample_valid", True)

        for rep_key, rep_data in sg.get("replicas", {}).items():
            anomalies = rep_data.get("anomalies", [])
            timeout_info = rep_data.get("timeout_info", {})

            for anom in anomalies:
                critical_set = {"BATMAN_DIRECT", "BATMAN_UIB",
                                "NO_PEAK", "TIMEOUT_IN_PEAK"}
                if anom in critical_set:
                    severity = "CRÍTIC"
                elif "_REPAIRED" in anom:
                    severity = "Reparat"
                else:
                    severity = "Avís"

                detail = ""
                if "BATMAN" in anom:
                    batman_info = rep_data.get("batman_direct_info", {})
                    if batman_info:
                        depth = batman_info.get("max_depth", 0)
                        detail = f"depth={depth:.2f}"
                elif "TIMEOUT" in anom:
                    n_t = timeout_info.get("n_timeouts", 0)
                    zone_summary = timeout_info.get("zone_summary", {})
                    zones_str = ",".join(zone_summary.keys()) if zone_summary else ""
                    timeouts_list = timeout_info.get("timeouts", [])
                    if timeouts_list:
                        to0 = timeouts_list[0]
                        detail = f"{to0.get('duration_sec',0):.0f}s at {to0.get('t_start_min',0):.1f}min ({to0.get('zone','')})"
                    else:
                        detail = f"n={n_t}, zones={zones_str}"

                anomaly_rows.append([
                    sample_name[:18], f"R{rep_key}", anom, severity, detail[:30]
                ])

        # Comparison warnings (replica correlation/area diff)
        comparison = sg.get("comparison") or {}
        rep_keys_list = sorted(sg.get("replicas", {}).keys())
        rep_label = f"R{rep_keys_list[0]} vs R{rep_keys_list[1]}" if len(rep_keys_list) >= 2 else "-"
        for signal_key in ("doc", "dad"):
            comp_data = comparison.get(signal_key, {})
            for warn in comp_data.get("warnings", []):
                # LOW_CORRELATION_362 is informative (always low)
                if "362" in warn and "CORRELATION" in warn.upper():
                    severity = "Info"
                elif "CORRELATION" in warn.upper():
                    severity = "Avís"
                elif "AREA_DIFF" in warn.upper():
                    severity = "Avís"
                else:
                    severity = "Info"

                source = "DOC" if signal_key == "doc" else "DAD"
                detail = f"Comparació rèpliques ({source})"
                anomaly_rows.append([
                    sample_name[:18], rep_label, warn, severity, detail[:30]
                ])

        # Mostra no vàlida
        if not sample_valid:
            reason = (sg.get("recommendation", {})
                      .get("doc", {}).get("reason", ""))
            anomaly_rows.append([
                sample_name[:18], "-", "MOSTRA NO VÀLIDA", "CRÍTIC",
                reason[:30]
            ])

    if anomaly_rows:
        # Paginar anomalies
        headers = ["Mostra", "Rep", "Anomalia", "Severitat", "Detalls"]
        col_widths = [0.20, 0.06, 0.30, 0.14, 0.30]

        # Máxim ~30 files per pàgina
        max_rows = 30
        for start in range(0, len(anomaly_rows), max_rows):
            if start > 0:
                # Nova pàgina si necessari
                _draw_footer(fig, page_num)
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

            # Color rows per severitat
            row_colors = {}
            for i, row in enumerate(page_anom):
                if row[3] == "CRÍTIC":
                    row_colors[i + 1] = "#FDEDEC"
                elif row[3] == "Reparat":
                    row_colors[i + 1] = "#FEF9E7"
                elif row[3] == "Info":
                    row_colors[i + 1] = "#EBF5FB"

            draw_minimal_table(ax, table_data, col_widths=col_widths,
                               row_colors=row_colors, font_size=7.5)

        # Resum
        n_critical = sum(1 for r in anomaly_rows if r[3] == "CRÍTIC")
        n_warning = sum(1 for r in anomaly_rows if r[3] == "Avís")
        n_repaired = sum(1 for r in anomaly_rows if r[3] == "Reparat")
        n_info = sum(1 for r in anomaly_rows if r[3] == "Info")

        summary_y = 0.90 - table_h - 0.04
        fig.text(0.08, summary_y,
                 f"Total: {len(anomaly_rows)}  |  "
                 f"CRÍTIC: {n_critical}  |  Avís: {n_warning}  |  "
                 f"Reparat: {n_repaired}  |  Info: {n_info}",
                 fontsize=9, fontweight='bold', color=COLORS["text"])
    else:
        fig.text(0.5, 0.5, "Cap anomalia detectada",
                 ha='center', va='center', fontsize=14,
                 fontweight='bold', color=COLORS["accent"])

    _draw_footer(fig, page_num)
    pdf.savefig(fig, dpi=150)
    plt.close(fig)
    return page_num + 1


# =============================================================================
# FUNCIÓ PRINCIPAL
# =============================================================================

def generate_analysis_report(seq_path, output_path=None, analysis_data=None):
    """
    Genera PDF d'anàlisi complet.

    Args:
        seq_path: Ruta a la carpeta SEQ
        output_path: Ruta de sortida (default: CHECK/)
        analysis_data: Dades d'anàlisi en memòria (opcional, si no es carrega de JSON)

    Returns:
        Path del PDF generat o None si error
    """
    # Carregar dades
    if analysis_data is None:
        data = _load_analysis_result(seq_path)
    else:
        data = analysis_data

    if not data:
        print(f"No s'han trobat dades d'anàlisi a {seq_path}")
        return None

    seq_name = data.get("seq_name", Path(seq_path).name)
    samples_grouped = data.get("samples_grouped", {})

    if not samples_grouped:
        print(f"No hi ha mostres agrupades a {seq_path}")
        return None

    # Path de sortida
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
        # P1: Resum seqüència
        _draw_page1_summary(pdf, data, seq_name)

        # P2+: Taula resultats
        next_page = _draw_results_pages(pdf, data, page_start=2)

        # P3+: Cromatogrames per mostra
        next_page = _draw_chromatogram_pages(pdf, data, page_start=next_page)

        # P final: Anomalies i warnings
        _draw_anomalies_page(pdf, data, page_num=next_page)

    print(f"  [OK] PDF generat: {pdf_path}")
    return str(pdf_path)


# =============================================================================
# MAIN (test standalone)
# =============================================================================

def main():
    """Test amb una SEQ."""
    import sys
    if len(sys.argv) > 1:
        seq_path = sys.argv[1]
    else:
        base_path = Path("C:/Users/Lequia/Desktop/Dades3")
        seqs = sorted(base_path.glob("*_SEQ"))
        if not seqs:
            print("No s'han trobat SEQs")
            return
        seq_path = str(seqs[0])

    print(f"Generant report d'anàlisi per: {seq_path}")
    result = generate_analysis_report(seq_path)
    if result:
        print(f"Report generat: {result}")
    else:
        print("Error generant report")


if __name__ == "__main__":
    main()
