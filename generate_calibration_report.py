# -*- coding: utf-8 -*-
"""
generate_calibration_report.py - INFORME QA/QC de Calibracio
=============================================================

PDF professional de 3 pagines:
  P1 (portrait)  - Summary QA/QC: metriques, gaussiana, shifts, historic
  P2 (landscape) - Recta de calibracio global (Area vs Mass)
  P3 (portrait)  - Grafics de repliques i historic (PNGs existents)

Estil consistent amb generate_import_report.py.
"""

import json
import numpy as np
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# =============================================================================
# CONFIGURACIO D'ESTIL - Identic a generate_import_report.py
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


def _draw_footer(fig, page_num):
    """Peu de pagina estandard."""
    fig.text(0.5, 0.02, "Serveis Tecnics de Recerca - Universitat de Girona",
             ha='center', fontsize=8, color=COLORS["text_secondary"])
    fig.text(0.95, 0.02, str(page_num), ha='right', fontsize=8,
             color=COLORS["text_secondary"])


# =============================================================================
# CARREGAR DADES
# =============================================================================

def _load_calibration_result(seq_path):
    """Carrega calibration_result.json de la SEQ."""
    p = Path(seq_path) / "CHECK" / "data" / "calibration_result.json"
    if not p.exists():
        return None
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)


def _load_calibration_reference(seq_path):
    """Carrega Calibration_Reference.json des de REGISTRY/."""
    registry = Path(seq_path).parent / "REGISTRY"
    p = registry / "Calibration_Reference.json"
    if not p.exists():
        return None
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)


def _load_khp_history(seq_path):
    """Carrega KHP_History.json des de REGISTRY/."""
    registry = Path(seq_path).parent / "REGISTRY"
    p = registry / "KHP_History.json"
    if not p.exists():
        return None
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)


def _load_calibration_graphs(seq_path):
    """Carrega PNGs existents de grafics de calibracio."""
    graphs_path = Path(seq_path) / "CHECK" / "Graphs"
    seq_name = Path(seq_path).name
    graphs = {}
    for key, pattern in [('replicas', f"calibration_replicas_{seq_name}.png"),
                          ('history', f"calibration_history_{seq_name}.png")]:
        f = graphs_path / pattern
        if f.exists():
            graphs[key] = str(f)
    return graphs


def _get_active_ref(cal_ref):
    """Retorna la calibracio activa de Calibration_Reference.json."""
    if not cal_ref:
        return None
    active_id = cal_ref.get("active_calibration_id")
    for c in cal_ref.get("calibrations", []):
        if c.get("id") == active_id or c.get("is_active"):
            return c
    cals = cal_ref.get("calibrations", [])
    return cals[0] if cals else None


# =============================================================================
# HELPERS - formatat de metriques
# =============================================================================

def _fmt_bigaussian(bg):
    """Formata info bigaussiana: sigma=X, R2=Y o N/D."""
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


def _build_condition_label(cal):
    """Ex: 'KHP 2ppm @ 100uL'."""
    conc = cal.get('conc_ppm', 0)
    vol = cal.get('volume_uL', 0)
    return f"KHP {conc:.0f}ppm @ {vol:.0f}\u00b5L"


def _filter_history_outliers(calibrations, mode):
    """Filtra historic per mode, exclou is_outlier i outliers estadistics (IQR).
    Retorna (clean_list, n_excluded).
    """
    # Primer filtre: mode + flag is_outlier
    candidates = []
    n_flagged = 0
    for h in calibrations:
        if h.get('mode') != mode:
            continue
        if h.get('is_outlier'):
            n_flagged += 1
            continue
        area = h.get('area', 0)
        if area > 0:
            candidates.append(h)

    # Segon filtre: IQR iteratiu sobre area (agrupa per volume_uL)
    if len(candidates) < 4:
        return candidates, n_flagged

    vol_groups = {}
    for h in candidates:
        vol = h.get('volume_uL', 0)
        vol_groups.setdefault(vol, []).append(h)

    clean = []
    n_iqr = 0
    for vol, group in vol_groups.items():
        if len(group) < 4:
            clean.extend(group)
            continue
        # IQR iteratiu: repetir fins que no es treguin mes punts
        current = list(group)
        while True:
            areas = [h.get('area', 0) for h in current]
            q1 = np.percentile(areas, 25)
            q3 = np.percentile(areas, 75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            filtered = [h for h in current if lower <= h.get('area', 0) <= upper]
            removed = len(current) - len(filtered)
            n_iqr += removed
            current = filtered
            if removed == 0 or len(current) < 4:
                break
        clean.extend(current)

    return clean, n_flagged + n_iqr


# =============================================================================
# PAGINA 1: SUMMARY QA/QC (A4 portrait)
# =============================================================================

def _draw_page1_summary(pdf, data, cal_ref, seq_name):
    """Pagina 1: resum general + metriques per condicio."""
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

    fig.text(0.5, 0.85, "INFORME QA/QC", ha='center', va='top',
             fontsize=18, fontweight='bold', color=COLORS["primary"])

    fig.text(0.5, 0.81, f"Sequ\u00e8ncia {seq_name}", ha='center', va='top',
             fontsize=12, color=COLORS["text"])

    date_str = datetime.now().strftime("%d/%m/%Y %H:%M")
    fig.text(0.5, 0.775, f"Generat: {date_str}", ha='center', va='top',
             fontsize=9, color=COLORS["text_secondary"])

    fig.add_artist(plt.Line2D([0.1, 0.9], [0.75, 0.75],
                              color=COLORS["primary"], linewidth=2,
                              transform=fig.transFigure))

    # --- DADES ---
    calibrations = data.get('calibrations', [])
    active_cals = [c for c in calibrations if c.get('is_active', False)]
    if not active_cals:
        active_cals = calibrations[:2]

    # Referencia
    ref = _get_active_ref(cal_ref)
    ref_rf = "-"
    ref_r2 = "-"
    ref_n = "-"
    if ref:
        rf_obj = ref.get("rf_mass_cal", {})
        if isinstance(rf_obj, dict):
            ref_rf = str(rf_obj.get("direct", {}).get("bp",
                         rf_obj.get("direct", {}).get("column", "-")))
        else:
            ref_rf = str(rf_obj)
        r2_val = ref.get("r2")
        ref_r2 = f"{r2_val:.4f}" if r2_val else "-"
        ref_n = str(ref.get("n_points", "-"))

    # Mode
    mode = active_cals[0].get('mode', '-') if active_cals else '-'
    conditions_str = ", ".join(
        f"{c.get('volume_uL', 0):.0f}\u00b5L @ {c.get('conc_ppm', 0):.0f}ppm"
        for c in active_cals
    )

    # Estat global
    statuses = [c.get('status', 'OK') for c in active_cals]
    global_status = "OK" if all(s == "OK" for s in statuses) else "WARNING"

    seq_date = active_cals[0].get('seq_date', '-') if active_cals else '-'
    if isinstance(seq_date, str) and len(seq_date) > 10:
        seq_date = seq_date[:10]

    # --- TAULA RESUM GENERAL ---
    summary_data = [
        ["Par\u00e0metre", "Valor", "Par\u00e0metre", "Valor"],
        ["Mode", f"{mode} ({conditions_str})", "Estat global", global_status],
        ["Calibraci\u00f3 ref.", f"RF={ref_rf}, R\u00b2={ref_r2} (n={ref_n})",
         "N cond. actives", str(len(active_cals))],
        ["Data SEQ", str(seq_date), "Data processament", date_str],
    ]

    ax_sum = fig.add_axes([0.08, 0.64, 0.84, 0.09])
    draw_minimal_table(ax_sum, summary_data,
                       col_widths=[0.18, 0.35, 0.18, 0.29],
                       font_size=9)

    # --- BLOC PER CADA CONDICIO ---
    y_pos = 0.60

    for cal in active_cals:
        if y_pos < 0.08:
            break

        cond_label = _build_condition_label(cal)
        status = cal.get('status', 'OK')
        status_color = COLORS["accent"] if status == "OK" else COLORS["warning"]

        # Titol condicio + badge
        fig.text(0.08, y_pos, cond_label,
                 fontsize=11, fontweight='bold', color=COLORS["text"])
        fig.text(0.50, y_pos, f"[{cal.get('condition_key', '')}]",
                 fontsize=9, color=COLORS["text_secondary"])

        bbox_props = dict(boxstyle="round,pad=0.3", facecolor=status_color,
                          edgecolor='none', alpha=0.9)
        fig.text(0.90, y_pos, status, ha='center', va='center',
                 fontsize=9, fontweight='bold', color='white',
                 bbox=bbox_props)

        y_pos -= 0.025

        # --- TAULA METRIQUES COMPACTA ---
        area_d = cal.get('area', 0)
        rf_mass_d = cal.get('rf_mass', 0)
        rsd = cal.get('rsd', 0)
        snr_d = cal.get('snr', 0)
        sym_d = cal.get('symmetry', 0)
        fwhm_d = cal.get('fwhm_doc', 0)
        bg_doc = _fmt_bigaussian(cal.get('bigaussian_doc'))

        area_u = cal.get('area_u', 0)
        rf_mass_u = cal.get('rf_mass_u', 0)
        snr_u = cal.get('snr_u', 0)
        bg_uib = _fmt_bigaussian(cal.get('bigaussian_uib'))

        has_uib = area_u and area_u > 0

        metrics_data = [
            ["M\u00e8trica", "Direct", "UIB"] if has_uib else ["M\u00e8trica", "Valor"],
        ]

        def _row(label, val_d, val_u=None):
            if has_uib:
                return [label, val_d, val_u if val_u else "-"]
            return [label, val_d]

        metrics_data.append(_row("\u00c0rea", format_value(area_d, ".1f"),
                                 format_value(area_u, ".1f") if has_uib else None))
        metrics_data.append(_row("RF mass", format_value(rf_mass_d, ".1f"),
                                 format_value(rf_mass_u, ".1f") if has_uib else None))
        metrics_data.append(_row("RSD", format_value(rsd, ".1f", "%")))
        metrics_data.append(_row("SNR", format_value(snr_d, ".0f"),
                                 format_value(snr_u, ".0f") if has_uib else None))
        metrics_data.append(_row("Simetria", format_value(sym_d, ".2f")))
        metrics_data.append(_row("FWHM", format_value(fwhm_d, ".2f", " min")))
        metrics_data.append(_row("Gaussiana", bg_doc,
                                 bg_uib if has_uib else None))

        # 254nm gaussiana
        bg_254 = _fmt_bigaussian(cal.get('bigaussian_254'))
        if bg_254 != "N/D":
            metrics_data.append(_row("Gauss. 254", bg_254))

        n_rows = len(metrics_data)
        table_h = 0.018 * (n_rows + 0.5)

        if has_uib:
            cw = [0.30, 0.35, 0.35]
        else:
            cw = [0.35, 0.65]

        ax_met = fig.add_axes([0.08, y_pos - table_h, 0.50, table_h])
        draw_minimal_table(ax_met, metrics_data, col_widths=cw, font_size=8)

        # --- INFO LATERAL: time shifts + historic ---
        info_x = 0.62
        info_y = y_pos - 0.005

        # Time shifts
        t_doc = cal.get('t_retention', 0)
        t_254 = cal.get('t_dad_max', 0)
        shift_sec = cal.get('shift_sec', 0)

        fig.text(info_x, info_y, "Time shifts:", fontsize=9,
                 fontweight='bold', color=COLORS["text"])
        info_y -= 0.018
        fig.text(info_x, info_y,
                 f"t DOC = {format_value(t_doc, '.2f', ' min')}",
                 fontsize=8, color=COLORS["text"])
        info_y -= 0.015
        fig.text(info_x, info_y,
                 f"t 254 = {format_value(t_254, '.3f', ' min')}",
                 fontsize=8, color=COLORS["text"])
        info_y -= 0.015
        fig.text(info_x, info_y,
                 f"\u0394t = {format_value(shift_sec, '.1f', 's')}",
                 fontsize=8, color=COLORS["primary"])

        # Comparacio historica
        val_det = cal.get('validation_details', {})
        hist_comp = val_det.get('historical_comparison', {})
        hist_stats = hist_comp.get('historical_stats')

        info_y -= 0.022
        fig.text(info_x, info_y, "Hist\u00f2ric:", fontsize=9,
                 fontweight='bold', color=COLORS["text"])
        info_y -= 0.018

        if hist_stats:
            mean_a = hist_stats.get('mean_area', 0)
            std_a = hist_stats.get('std_area', 0)
            dev_pct = hist_comp.get('area_deviation_pct', 0)
            n_hist = hist_stats.get('n_calibrations', 0)

            fig.text(info_x, info_y,
                     f"Mitjana: {mean_a:.1f} \u00b1 {std_a:.1f} (n={n_hist})",
                     fontsize=8, color=COLORS["text"])
            info_y -= 0.015

            dev_color = COLORS["accent"] if abs(dev_pct) < 15 else COLORS["warning"]
            fig.text(info_x, info_y,
                     f"Actual: {area_d:.1f}  ({dev_pct:+.1f}%)",
                     fontsize=8, fontweight='bold', color=dev_color)
            info_y -= 0.015
        else:
            hist_status = hist_comp.get('status', '-')
            fig.text(info_x, info_y, f"Estat: {hist_status}",
                     fontsize=8, color=COLORS["text_secondary"])
            info_y -= 0.015

        # Warnings
        issues = cal.get('quality_issues', [])
        warnings = cal.get('calibration_warnings', [])
        hist_warnings = hist_comp.get('warnings', [])
        all_issues = issues + warnings + hist_warnings

        if all_issues:
            info_y -= 0.005
            for issue in all_issues[:2]:
                fig.text(info_x, info_y, f"[!] {issue[:55]}",
                         fontsize=7, color=COLORS["warning"])
                info_y -= 0.013

        y_pos -= table_h + 0.03

        # Separador entre condicions
        fig.add_artist(plt.Line2D([0.08, 0.92], [y_pos + 0.01, y_pos + 0.01],
                                  color=COLORS["border"], linewidth=0.5,
                                  transform=fig.transFigure))
        y_pos -= 0.01

    _draw_footer(fig, 1)
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


# =============================================================================
# PAGINA 2: RECTA DE CALIBRACIO GLOBAL (A4 portrait)
# =============================================================================

def _draw_page2_calibration_line(pdf, data, cal_ref, khp_history, seq_name):
    """Pagina 2: grafic Area vs Mass + taula resum punts."""
    fig = plt.figure(figsize=(8.27, 11.69))  # portrait
    fig.patch.set_facecolor('white')

    # Capcalera
    fig.text(0.5, 0.96, "RECTA DE CALIBRACI\u00d3 GLOBAL",
             ha='center', va='top', fontsize=14, fontweight='bold',
             color=COLORS["primary"])
    fig.text(0.5, 0.93, f"{seq_name} - Area vs Mass (\u00b5g)",
             ha='center', va='top', fontsize=10,
             color=COLORS["text_secondary"])
    fig.add_artist(plt.Line2D([0.1, 0.9], [0.91, 0.91],
                              color=COLORS["primary"], linewidth=1,
                              transform=fig.transFigure))

    # Obtenir referencia
    ref = _get_active_ref(cal_ref)
    rf_mass_cal = 682  # default
    r2_ref = 0.99
    tolerance_pct = 20
    warning_pct = 15

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

    # Calibracions actives de la SEQ
    calibrations = data.get('calibrations', [])
    active_cals = [c for c in calibrations if c.get('is_active', False)]
    if not active_cals:
        active_cals = calibrations[:2]

    current_mode = active_cals[0].get('mode', 'BP') if active_cals else 'BP'

    # Recollir punts historics (filtre IQR)
    hist_masses = []
    hist_areas = []
    hist_labels = []
    n_hist_excluded = 0

    if khp_history:
        clean_hist, n_hist_excluded = _filter_history_outliers(
            khp_history.get('calibrations', []), current_mode)
        for h in clean_hist:
            conc = h.get('conc_ppm', 0)
            vol = h.get('volume_uL', 0)
            area = h.get('area', 0)
            if conc > 0 and vol > 0 and area > 0:
                mass = conc * vol / 1000.0
                hist_masses.append(mass)
                hist_areas.append(area)
                hist_labels.append(h.get('seq_name', ''))

    if n_hist_excluded > 0:
        print(f"  Recta: {n_hist_excluded} outliers exclosos (flag + IQR)")

    # Punts actuals
    curr_masses = []
    curr_areas = []
    curr_labels = []
    curr_within_tolerance = []

    for cal in active_cals:
        conc = cal.get('conc_ppm', 0)
        vol = cal.get('volume_uL', 0)
        area = cal.get('area', 0)
        if conc > 0 and vol > 0 and area > 0:
            mass = conc * vol / 1000.0
            expected = rf_mass_cal * mass
            dev = abs(area - expected) / expected * 100 if expected > 0 else 0
            curr_masses.append(mass)
            curr_areas.append(area)
            curr_labels.append(cal.get('condition_key', ''))
            curr_within_tolerance.append(dev <= tolerance_pct)

    # --- GRAFIC ---
    ax = fig.add_axes([0.10, 0.30, 0.80, 0.58])

    # Rang X
    all_masses = hist_masses + curr_masses
    if all_masses:
        x_max = max(all_masses) * 1.15
    else:
        x_max = 0.3
    x_line = np.linspace(0, x_max, 200)

    # Recta
    y_line = rf_mass_cal * x_line
    ax.plot(x_line, y_line, color=COLORS["primary"], linewidth=1.5,
            label=f"Recta: Area = {rf_mass_cal} \u00d7 Mass")

    # Bandes tolerancia
    y_warn_upper = y_line * (1 + warning_pct / 100)
    y_warn_lower = y_line * (1 - warning_pct / 100)
    y_tol_upper = y_line * (1 + tolerance_pct / 100)
    y_tol_lower = y_line * (1 - tolerance_pct / 100)

    ax.fill_between(x_line, y_tol_lower, y_tol_upper,
                    alpha=0.08, color=COLORS["danger"],
                    label=f"\u00b1{tolerance_pct}% toler\u00e0ncia")
    ax.fill_between(x_line, y_warn_lower, y_warn_upper,
                    alpha=0.10, color=COLORS["warning"],
                    label=f"\u00b1{warning_pct}% warning")

    # Punts historics
    if hist_masses:
        hist_label = f"Hist\u00f2ric ({len(hist_masses)} pts)"
        if n_hist_excluded > 0:
            hist_label += f", {n_hist_excluded} exclosos"
        ax.scatter(hist_masses, hist_areas, c='#BBBBBB', s=25, alpha=0.6,
                   edgecolors='#999999', linewidths=0.5, zorder=3,
                   label=hist_label)

    # Punts actuals
    for i, (m, a, lbl, ok) in enumerate(zip(curr_masses, curr_areas,
                                             curr_labels, curr_within_tolerance)):
        color = COLORS["accent"] if ok else COLORS["danger"]
        marker = 'o' if ok else 'X'
        ax.scatter([m], [a], c=color, s=100, edgecolors='white',
                   linewidths=1.5, zorder=5, marker=marker,
                   label=f"Actual: {lbl}" if i < 3 else None)
        ax.annotate(lbl, (m, a), textcoords="offset points",
                    xytext=(8, 8), fontsize=7, color=color, fontweight='bold')

    # Equacio + R2
    eq_text = f"Area = {rf_mass_cal} \u00d7 Mass (R\u00b2 = {r2_ref:.4f})"
    ax.text(0.03, 0.97, eq_text, transform=ax.transAxes,
            fontsize=8, fontweight='bold', color=COLORS["primary"],
            va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=COLORS["border"], alpha=0.9))

    ax.set_xlabel("Mass (\u00b5g)", fontsize=9)
    ax.set_ylabel("Area (mAU\u00b7min)", fontsize=9)
    ax.set_xlim(0, x_max)
    y_all = hist_areas + curr_areas + [0]
    if y_all:
        ax.set_ylim(0, max(y_all) * 1.2)
    ax.grid(True, alpha=0.3, linewidth=0.3)
    ax.legend(loc='lower right', fontsize=7, frameon=True,
              framealpha=0.9, edgecolor=COLORS["border"])

    # --- TAULA RESUM PUNTS ---
    table_headers = ["Condici\u00f3", "Vol (\u00b5L)", "Conc (ppm)",
                     "Mass (\u00b5g)", "\u00c0rea", "RF mass",
                     "Esperat", "Dev %", "Estat"]
    table_data = [table_headers]

    for cal in active_cals:
        conc = cal.get('conc_ppm', 0)
        vol = cal.get('volume_uL', 0)
        area = cal.get('area', 0)
        rf_m = cal.get('rf_mass', 0)
        mass = conc * vol / 1000.0 if conc > 0 and vol > 0 else 0
        expected = rf_mass_cal * mass
        dev = (area - expected) / expected * 100 if expected > 0 else 0

        status_str = "OK" if abs(dev) <= tolerance_pct else "FAIL"
        if abs(dev) > warning_pct and abs(dev) <= tolerance_pct:
            status_str = "WARN"

        table_data.append([
            cal.get('condition_key', '-'),
            format_value(vol, ".0f"),
            format_value(conc, ".1f"),
            format_value(mass, ".3f"),
            format_value(area, ".1f"),
            format_value(rf_m, ".1f"),
            format_value(expected, ".1f"),
            format_value(dev, "+.1f", "%"),
            status_str,
        ])

    n_rows = len(table_data)
    table_height = 0.015 + 0.016 * n_rows
    ax_table = fig.add_axes([0.08, 0.05, 0.84, min(table_height, 0.18)])
    draw_minimal_table(ax_table, table_data,
                       col_widths=[0.14, 0.08, 0.09, 0.09, 0.10,
                                   0.10, 0.10, 0.10, 0.10],
                       font_size=8)

    _draw_footer(fig, 2)
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


# =============================================================================
# PAGINA 3: GRAFICS DE REPLIQUES I HISTORIC (A4 portrait)
# =============================================================================

def _draw_page3_graphs(pdf, seq_path, data, khp_history, seq_name):
    """Pagina 3: PNGs existents o grafic generat.
    Historic sempre generat des de dades (no PNG) per controlar outliers.
    """
    # TODO: Flexibilitzar criteri cal in/out. Ara es ~5% (RSD threshold).
    #       Provar amb 10% i veure com queda. Fer-ho configurable.

    graphs = _load_calibration_graphs(seq_path)

    has_replicas = 'replicas' in graphs

    # Determinar si tenim dades per generar historic
    calibrations = data.get('calibrations', [])
    active_cals = [c for c in calibrations if c.get('is_active', False)]
    current_mode = active_cals[0].get('mode', 'BP') if active_cals else 'BP'

    # Filtrar historic: excloure outliers (flag + IQR)
    mode_history = []
    n_outliers = 0
    if khp_history:
        mode_history, n_outliers = _filter_history_outliers(
            khp_history.get('calibrations', []), current_mode)

    has_history_data = len(mode_history) > 0

    if not has_replicas and not has_history_data:
        return  # res a mostrar

    fig = plt.figure(figsize=(8.27, 11.69))  # portrait
    fig.patch.set_facecolor('white')

    # Capcalera
    fig.text(0.5, 0.96, "GR\u00c0FICS DE CALIBRACI\u00d3",
             ha='center', va='top', fontsize=14, fontweight='bold',
             color=COLORS["primary"])
    fig.text(0.5, 0.93, seq_name, ha='center', va='top',
             fontsize=10, color=COLORS["text_secondary"])
    fig.add_artist(plt.Line2D([0.1, 0.9], [0.91, 0.91],
                              color=COLORS["primary"], linewidth=1,
                              transform=fig.transFigure))

    # --- GRAFIC REPLIQUES (PNG) ---
    if has_replicas:
        try:
            img = plt.imread(graphs['replicas'])
            ax = fig.add_axes([0.05, 0.48, 0.90, 0.40])
            ax.imshow(img)
            ax.axis('off')
            fig.text(0.5, 0.89, "R\u00e8pliques KHP (DOC + DAD 254nm)",
                     ha='center', fontsize=10, fontweight='bold',
                     color=COLORS["text"])
        except Exception as e:
            print(f"  Error carregant imatge repliques: {e}")

    # --- GRAFIC HISTORIC (sempre des de dades, sense outliers) ---
    if has_history_data:
        display_cals = mode_history[-15:]

        y_bottom = 0.05 if has_replicas else 0.30
        h = 0.38 if has_replicas else 0.55
        ax = fig.add_axes([0.10, y_bottom, 0.80, h])

        seq_labels = []
        areas = []
        colors_bars = []

        for cal in display_cals:
            name = cal.get('seq_name', 'N/A')
            name = name.replace('_SEQ', '').replace('_BP', '')
            seq_labels.append(name)
            areas.append(cal.get('area', 0))
            if cal.get('seq_name') == seq_name:
                colors_bars.append(COLORS["accent"])
            else:
                colors_bars.append(COLORS["primary"])

        x = range(len(seq_labels))
        ax.bar(x, areas, color=colors_bars, edgecolor='white',
               linewidth=0.5)

        if areas:
            mean_area = np.mean(areas)
            std_area = np.std(areas) if len(areas) > 1 else 0
            legend_label = f'Mitjana: {mean_area:.0f}'
            if n_outliers > 0:
                legend_label += f' ({n_outliers} outliers exclosos)'
            ax.axhline(mean_area, color=COLORS["accent"], linestyle='--',
                       linewidth=1.5, label=legend_label)
            if std_area > 0:
                ax.axhspan(mean_area - std_area, mean_area + std_area,
                           alpha=0.1, color=COLORS["accent"])
            ax.legend(loc='upper right', fontsize=7, frameon=False)

        ax.set_xticks(list(x))
        ax.set_xticklabels(seq_labels, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel("Area", fontsize=9)

        title = f"Evoluci\u00f3 de l'\u00e0rea KHP ({current_mode})"
        ax.set_title(title, fontsize=11, fontweight='bold',
                     color=COLORS["text"], pad=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if n_outliers > 0:
            print(f"  Historic: {n_outliers} outliers exclosos del grafic")

    _draw_footer(fig, 3)
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


# =============================================================================
# GENERACIO DEL PDF
# =============================================================================

def generate_calibration_report(seq_path, output_path=None):
    """
    Genera PDF QA/QC de calibracio (3 pagines).

    Args:
        seq_path: Ruta a la carpeta SEQ
        output_path: Ruta de sortida (default: CHECK/)

    Returns:
        Path del PDF generat o None si error
    """
    # Carregar dades
    data = _load_calibration_result(seq_path)
    if not data:
        print(f"No s'han trobat dades de calibracio a {seq_path}")
        return None

    seq_name = data.get('seq_name', Path(seq_path).name)
    calibrations = data.get('calibrations', [])

    if not calibrations:
        print(f"No hi ha calibracions a {seq_path}")
        return None

    # Carregar dades externes
    cal_ref = _load_calibration_reference(seq_path)
    khp_history = _load_khp_history(seq_path)

    graphs = _load_calibration_graphs(seq_path)
    print(f"  Grafics trobats: {list(graphs.keys()) if graphs else 'cap'}")
    if cal_ref:
        print(f"  Calibration_Reference carregada")
    if khp_history:
        n_hist = len(khp_history.get('calibrations', []))
        print(f"  KHP_History: {n_hist} entrades")

    # Path de sortida
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
            print(f"  Fitxer bloquejat, usant: {pdf_path.name}")

    print(f"Generant {pdf_path}...")

    active_cals = [c for c in calibrations if c.get('is_active', False)]
    print(f"  Calibracions actives: {len(active_cals)}")

    with PdfPages(pdf_path) as pdf:
        # Pagina 1: Summary QA/QC
        _draw_page1_summary(pdf, data, cal_ref, seq_name)

        # Pagina 2: Recta de calibracio global
        _draw_page2_calibration_line(pdf, data, cal_ref, khp_history, seq_name)

        # Pagina 3: Grafics repliques + historic
        _draw_page3_graphs(pdf, seq_path, data, khp_history, seq_name)

    print(f"  [OK] PDF generat: {pdf_path}")
    return str(pdf_path)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Test amb multiples SEQs."""
    base_path = Path("C:/Users/Lequia/Desktop/Dades3")

    test_seqs = [
        "286_SEQ_BP",
        "285_SEQ",
        "282_SEQ",
        "277_SEQ_BP",
    ]

    results = []
    for seq_name in test_seqs:
        seq_path = base_path / seq_name
        if seq_path.exists():
            print(f"\n{'='*60}")
            print(f"Processant: {seq_name}")
            print('='*60)
            result = generate_calibration_report(seq_path)
            if result:
                results.append(result)
        else:
            print(f"No trobat: {seq_path}")

    print(f"\n{'='*60}")
    print(f"COMPLETAT: {len(results)} PDFs generats")
    print('='*60)
    for r in results:
        print(f"  {r}")

    if results:
        import subprocess
        subprocess.Popen(['start', '', results[0]], shell=True)


if __name__ == "__main__":
    main()
