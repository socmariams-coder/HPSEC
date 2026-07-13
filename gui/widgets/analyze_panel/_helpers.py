"""
Shared helper functions for analyze_panel package.

Provides reusable table population and styling functions used by
panel.py (main panel) and dialogs.py (detail dialog).
"""

from PySide6.QtWidgets import (
    QTableWidgetItem, QHeaderView, QAbstractItemView
)
from PySide6.QtGui import QFont, QColor, QBrush

from gui.widgets.styles import COLOR_SUCCESS, COLOR_WARNING, COLOR_ERROR
from ._constants import FRACTION_NAMES, FRACTION_RANGES
from hpsec_warnings import (
    has_anomaly, get_anomaly_codes, classify_anomalies,
    ANOMALY_CATALOG,
)


# ---------------------------------------------------------------------------
# Table styling
# ---------------------------------------------------------------------------

def configure_table_style(table, compact=False):
    """Apply standard table styling.

    Args:
        table: QTableWidget to style
        compact: True for smaller fonts/padding (detail dialogs)
    """
    table.setEditTriggers(QAbstractItemView.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectRows)
    table.setSelectionMode(QAbstractItemView.SingleSelection)
    table.setAlternatingRowColors(True)
    table.verticalHeader().setVisible(False)

    if compact:
        table.setStyleSheet("""
            QTableWidget {
                gridline-color: #ddd;
                background-color: white;
                alternate-background-color: #f9f9f9;
                font-size: 11px;
            }
            QTableWidget::item { padding: 2px 4px; }
            QHeaderView::section {
                background-color: #f5f5f5;
                padding: 4px;
                border: none;
                border-bottom: 2px solid #ddd;
                font-weight: bold;
                font-size: 10px;
            }
        """)
    else:
        table.setStyleSheet("""
            QTableWidget {
                gridline-color: #ddd;
                background-color: white;
                alternate-background-color: #f9f9f9;
            }
            QTableWidget::item { padding: 4px 6px; }
            QTableWidget::item:selected {
                background-color: #E3F2FD;
                color: black;
            }
            QHeaderView::section {
                background-color: #f5f5f5;
                padding: 6px;
                border: none;
                border-bottom: 2px solid #ddd;
                font-weight: bold;
                font-size: 11px;
            }
        """)


# ---------------------------------------------------------------------------
# Signal summary table
# ---------------------------------------------------------------------------

def populate_signal_summary(table, rep_data, signal_keys, show_timeouts=True):
    """Populate a signal summary table from replica data.

    Args:
        table: QTableWidget (4 or 5 columns depending on show_timeouts)
        rep_data: Replica data dict
        signal_keys: List of (label, key) tuples
        show_timeouts: Whether to populate the Timeouts column (col 4)
    """
    table.setRowCount(0)

    tmax_signals = rep_data.get("tmax_signals") or {}
    areas = rep_data.get("areas") or {}
    areas_uib = rep_data.get("areas_uib") or {}
    snr_info = rep_data.get("snr_info") or {}
    snr_dad = rep_data.get("snr_info_dad") or {}
    timeout_info = rep_data.get("timeout_info") or {}
    n_timeouts = timeout_info.get("n_timeouts", 0)
    timeout_zones = timeout_info.get("zones", [])
    timeout_severity = timeout_info.get("severity", "OK")

    for label, key in signal_keys:
        row = table.rowCount()
        table.insertRow(row)
        table.setItem(row, 0, QTableWidgetItem(label))

        if key == "DOC":
            tmax = tmax_signals.get("DOC", 0)
            area = (areas.get("DOC") or {}).get("total", 0)
            snr = snr_info.get("snr_direct", 0)
            timeout_text = ""
            if n_timeouts > 0:
                timeout_text = f"T({n_timeouts}) {timeout_severity}"
                if timeout_zones:
                    timeout_text += f" zona {', '.join(timeout_zones)}"
        elif key == "UIB":
            tmax = tmax_signals.get("DOC", 0)  # Same detector
            area = areas_uib.get("total", 0)
            snr = snr_info.get("snr_uib", 0)
            timeout_text = "propagat de Direct" if n_timeouts > 0 else ""
        else:
            tmax = tmax_signals.get(key, 0)
            area = (areas.get(key) or {}).get("total", 0)
            snr_entry = snr_dad.get(key) or {}
            snr = snr_entry.get("snr", 0) if isinstance(snr_entry, dict) else 0
            timeout_text = ""

        table.setItem(row, 1, QTableWidgetItem(f"{tmax:.2f}" if tmax else "-"))
        table.setItem(row, 2, QTableWidgetItem(f"{area:.1f}" if area else "-"))
        table.setItem(row, 3, QTableWidgetItem(f"{snr:.0f}" if snr else "-"))

        if show_timeouts:
            timeout_item = QTableWidgetItem(timeout_text if timeout_text else "-")
            if timeout_text and "propagat" not in timeout_text:
                timeout_item.setForeground(QBrush(QColor(COLOR_WARNING)))
            table.setItem(row, 4, timeout_item)


# ---------------------------------------------------------------------------
# Fractions table
# ---------------------------------------------------------------------------

def populate_fractions_table(table, rep_data, is_bp, wavelengths,
                             show_ratio=False):
    """Populate a fractions table from replica data.

    Args:
        table: QTableWidget
        rep_data: Replica data dict
        is_bp: True for BP mode (single Total row), False for COLUMN
        wavelengths: List of wavelength keys (e.g. DAD_WL_MAIN or DAD_WL_ALL)
        show_ratio: Whether to append DOC/A254 ratio rows
    """
    table.setRowCount(0)

    areas = rep_data.get("areas") or {}
    areas_uib = rep_data.get("areas_uib") or {}

    headers = ["Fracció", "Rang (min)", "DOC Direct", "DOC UIB"] + wavelengths
    table.setColumnCount(len(headers))
    table.setHorizontalHeaderLabels(headers)
    header = table.horizontalHeader()
    for i in range(table.columnCount()):
        header.setSectionResizeMode(i, QHeaderView.ResizeToContents)

    doc_areas = areas.get("DOC") or {}
    uib_areas = areas_uib or {}

    if is_bp:
        fracs = [("Total", "0-70")]
    else:
        fracs = [(f, FRACTION_RANGES[f]) for f in FRACTION_NAMES]

    # Compute totals for percentage calculation
    totals = {"DOC_D": 0, "DOC_U": 0}
    for wl in wavelengths:
        totals[wl] = 0

    for frac_name, _ in fracs:
        if frac_name == "Total":
            totals["DOC_D"] = doc_areas.get("total", 0)
            totals["DOC_U"] = uib_areas.get("total", 0)
            for wl in wavelengths:
                totals[wl] = (areas.get(wl) or {}).get("total", 0)
        else:
            totals["DOC_D"] += doc_areas.get(frac_name, 0)
            totals["DOC_U"] += uib_areas.get(frac_name, 0)
            for wl in wavelengths:
                totals[wl] += (areas.get(wl) or {}).get(frac_name, 0)

    # --- Area rows ---
    for frac_name, rang in fracs:
        row = table.rowCount()
        table.insertRow(row)
        table.setItem(row, 0, QTableWidgetItem(frac_name))
        table.setItem(row, 1, QTableWidgetItem(rang))

        key = "total" if frac_name == "Total" else frac_name
        _set_area_cell(table, row, 2, doc_areas.get(key, 0))
        _set_area_cell(table, row, 3, uib_areas.get(key, 0))

        for j, wl in enumerate(wavelengths):
            _set_area_cell(table, row, 4 + j, (areas.get(wl) or {}).get(key, 0))

    # --- Total + percentage rows (COLUMN only) ---
    if not is_bp:
        bold = QFont()
        bold.setBold(True)

        # Total row
        row = table.rowCount()
        table.insertRow(row)
        _set_font_cell(table, row, 0, "Total", bold)
        table.setItem(row, 1, QTableWidgetItem("0-70"))

        for col_idx, key in enumerate(["DOC_D", "DOC_U"] + wavelengths):
            val = totals[key]
            item = QTableWidgetItem(f"{val:.1f}" if val else "-")
            item.setFont(bold)
            table.setItem(row, 2 + col_idx, item)

        # % separator row
        row_sep = table.rowCount()
        table.insertRow(row_sep)
        _set_font_cell(table, row_sep, 0, "%", bold)
        for c in range(1, table.columnCount()):
            table.setItem(row_sep, c, QTableWidgetItem(""))

        # % rows
        for frac_name, _ in fracs:
            row = table.rowCount()
            table.insertRow(row)
            table.setItem(row, 0, QTableWidgetItem(frac_name))
            table.setItem(row, 1, QTableWidgetItem(""))

            _set_pct_cell(table, row, 2, doc_areas.get(frac_name, 0), totals.get("DOC_D", 0))
            _set_pct_cell(table, row, 3, uib_areas.get(frac_name, 0), totals.get("DOC_U", 0))

            for j, wl in enumerate(wavelengths):
                val = (areas.get(wl) or {}).get(frac_name, 0)
                _set_pct_cell(table, row, 4 + j, val, totals.get(wl, 0))

        # --- DOC/A254 ratio (optional, for detail dialog) ---
        if show_ratio:
            italic = QFont()
            italic.setItalic(True)

            row_hdr = table.rowCount()
            table.insertRow(row_hdr)
            _set_font_cell(table, row_hdr, 0, "DOC/A254", italic)
            for c in range(1, table.columnCount()):
                table.setItem(row_hdr, c, QTableWidgetItem(""))

            for frac_name, _ in fracs:
                row = table.rowCount()
                table.insertRow(row)
                _set_font_cell(table, row, 0, frac_name, italic)
                table.setItem(row, 1, QTableWidgetItem(""))

                val_d = doc_areas.get(frac_name, 0)
                val_254 = (areas.get("A254") or {}).get(frac_name, 0)
                ratio = (val_d / val_254) if val_254 > 0 else 0
                ratio_item = QTableWidgetItem(f"{ratio:.2f}" if ratio > 0 else "-")
                ratio_item.setFont(italic)
                table.setItem(row, 2, ratio_item)
                for c in range(3, table.columnCount()):
                    table.setItem(row, c, QTableWidgetItem(""))


# ---------------------------------------------------------------------------
# Cell helpers
# ---------------------------------------------------------------------------

def _set_area_cell(table, row, col, value):
    table.setItem(row, col, QTableWidgetItem(f"{value:.1f}" if value else "-"))


def _set_font_cell(table, row, col, text, font):
    item = QTableWidgetItem(text)
    item.setFont(font)
    table.setItem(row, col, item)


def _set_pct_cell(table, row, col, value, total):
    pct = (value / total * 100) if total > 0 else 0
    table.setItem(row, col, QTableWidgetItem(f"{pct:.1f}%"))


# ---------------------------------------------------------------------------
# Timeout zone drawing (shared between detail dialog, results tab, etc.)
# ---------------------------------------------------------------------------

def draw_timeout_zones_on_ax(ax, timeout_info_r1, timeout_info_r2=None,
                              color_r1='#E74C3C', color_r2='#F39C12'):
    """Draw semi-transparent timeout zone rectangles on a matplotlib Axes.

    Args:
        ax: matplotlib Axes
        timeout_info_r1: timeout_info dict for replica 1 (or None)
        timeout_info_r2: timeout_info dict for replica 2 (or None)
        color_r1: color for R1 timeout zones
        color_r2: color for R2 timeout zones
    """
    drawn_labels = set()

    def _draw(timeout_info, color, label_prefix):
        if not timeout_info:
            return
        timeouts_list = timeout_info.get('timeouts', [])
        for to in timeouts_list:
            t_start = to.get('t_start_min', 0)
            t_end = to.get('t_end_min', 0)
            aff_start = to.get('affected_start_min', t_start - 0.5)
            aff_end = to.get('affected_end_min', t_end + 1.0)
            ax.axvspan(aff_start, aff_end, alpha=0.15, color=color, zorder=0)
            ax.axvline(t_start, color=color, ls='--', lw=0.8, alpha=0.6)
            # Label (una sola per rèplica)
            key = f"{label_prefix}_{t_start:.1f}"
            if key not in drawn_labels:
                drawn_labels.add(key)
                zone = to.get('zone', '')
                ax.annotate(f'TO {label_prefix}',
                            xy=(t_start, 0.95), xycoords=('data', 'axes fraction'),
                            fontsize=6, color=color, alpha=0.8, ha='center')

    _draw(timeout_info_r1, color_r1, 'R1')
    _draw(timeout_info_r2, color_r2, 'R2')


# ---------------------------------------------------------------------------
# Calibration comparison helpers
# ---------------------------------------------------------------------------

def format_calibration_comparison_html(rf_vigent, int_vigent, rf_new, int_new,
                                        r2_new=None, r2_vigent=None,
                                        show_equation=False,
                                        model_type="intercept"):
    """
    Genera HTML per taula comparació calibració vigent vs nova.

    Args:
        rf_vigent: RF actual
        int_vigent: Intercept actual
        rf_new: RF nou
        int_new: Intercept nou
        r2_new: R² de la nova regressió
        r2_vigent: R² de la calibració vigent
        show_equation: Si True, afegeix fila amb equació
        model_type: "intercept" o "origin"

    Returns:
        str: HTML amb taula estilitzada
    """
    # Colors per deltas
    def _delta_color(pct):
        if abs(pct) < 5:
            return "#27AE60"  # verd
        elif abs(pct) < 15:
            return "#E67E22"  # taronja
        return "#E74C3C"  # vermell

    # Delta RF
    delta_rf = (rf_new - rf_vigent) / rf_vigent * 100 if rf_vigent and rf_vigent > 0 else 0

    # Delta intercept (absolut, no relatiu)
    delta_int = int_new - int_vigent if int_vigent is not None else 0

    html = """
    <table style='font-size: 11px; border-collapse: collapse; width: 100%;'>
    <tr style='background-color: #2980B9; color: white;'>
        <th style='padding: 4px 8px;'></th>
        <th style='padding: 4px 8px;'>Vigent</th>
        <th style='padding: 4px 8px;'>Nova</th>
        <th style='padding: 4px 8px;'>Δ</th>
    </tr>
    """

    # RF
    html += f"""
    <tr style='border-bottom: 1px solid #eee;'>
        <td style='padding: 3px 8px;'><b>RF (slope)</b></td>
        <td style='padding: 3px 8px; text-align: center;'>{rf_vigent:.1f}</td>
        <td style='padding: 3px 8px; text-align: center;'><b>{rf_new:.1f}</b></td>
        <td style='padding: 3px 8px; text-align: center; color: {_delta_color(delta_rf)};'>
            {delta_rf:+.1f}%</td>
    </tr>
    """

    # Intercept
    html += f"""
    <tr style='border-bottom: 1px solid #eee;'>
        <td style='padding: 3px 8px;'><b>Intercept</b></td>
        <td style='padding: 3px 8px; text-align: center;'>{int_vigent:.1f}</td>
        <td style='padding: 3px 8px; text-align: center;'><b>{int_new:.1f}</b></td>
        <td style='padding: 3px 8px; text-align: center;'>{delta_int:+.1f}</td>
    </tr>
    """

    # R²
    if r2_new is not None:
        r2_color = "#27AE60" if r2_new >= 0.99 else ("#E67E22" if r2_new >= 0.95 else "#E74C3C")
        r2_vig_str = f"{r2_vigent:.6f}" if r2_vigent else "—"
        html += f"""
        <tr style='border-bottom: 1px solid #eee;'>
            <td style='padding: 3px 8px;'><b>R²</b></td>
            <td style='padding: 3px 8px; text-align: center;'>{r2_vig_str}</td>
            <td style='padding: 3px 8px; text-align: center;'>
                <b style='color: {r2_color};'>{r2_new:.6f}</b></td>
            <td style='padding: 3px 8px; text-align: center;'>—</td>
        </tr>
        """

    # Equació (opcional)
    if show_equation:
        if model_type == 'origin':
            eq = f"Àrea = {rf_new:.1f} × µg_DOC"
        else:
            eq = f"Àrea = {rf_new:.1f} × µg_DOC + {int_new:.1f}"
        html += f"""
        <tr style='background-color: #f8f9fa;'>
            <td colspan='4' style='padding: 4px 8px; font-family: monospace; text-align: center;'>
                {eq}</td>
        </tr>
        """

    html += "</table>"
    return html.strip()


def compute_prediction_band(x_fit, rf, intercept, x_data, y_data, confidence=0.95):
    """
    Calcula la banda de predicció per una regressió lineal.

    Args:
        x_fit: array de punts x per la banda (np.linspace)
        rf: slope de la regressió
        intercept: intercept de la regressió
        x_data: array de punts x observats
        y_data: array de punts y observats
        confidence: nivell de confiança (default 0.95)

    Returns:
        tuple (y_lower, y_upper) o None si error
    """
    import numpy as np
    try:
        from scipy.stats import t as t_dist
    except ImportError:
        return None

    n = len(x_data)
    if n < 3:
        return None

    x_arr = np.asarray(x_data, dtype=float)
    y_arr = np.asarray(y_data, dtype=float)

    y_pred_data = rf * x_arr + intercept
    mse = np.sum((y_arr - y_pred_data) ** 2) / (n - 2)
    x_mean = np.mean(x_arr)
    Sxx = np.sum((x_arr - x_mean) ** 2)

    if Sxx <= 0 or mse < 0:
        return None

    alpha = 1 - confidence
    t_val = t_dist.ppf(1 - alpha / 2, n - 2)

    y_fit = rf * x_fit + intercept
    se_pred = np.sqrt(mse * (1 + 1.0 / n + (x_fit - x_mean) ** 2 / Sxx))

    return (y_fit - t_val * se_pred, y_fit + t_val * se_pred)


# ---------------------------------------------------------------------------
# Sample status classification (extracted from AnalyzePanel)
# ---------------------------------------------------------------------------

def classify_sample_status(doc_rep_data, dad_rep_data, comparison,
                           sample_data=None):
    """Classifica l'estat d'una mostra: anomalies (col 7) + reparacio (per Accio col 8).

    Returns (status_color, status_text, status_tooltip,
             repair_color, repair_text, repair_tooltip).
    """
    # --- Defaults reparacio ---
    repair_color = "#888"
    repair_text = ""
    repair_tooltip = ""

    # Comprovar si l'usuari ha seleccionat "Cap"
    if sample_data:
        selected = sample_data.get("selected", {})
        if selected.get("doc") == "none":
            return ("#888888", "\u2014",
                    "Usuari ha seleccionat 'Cap' \u2014 No es quantificara ni exportara",
                    repair_color, repair_text, repair_tooltip)
        if sample_data.get("sample_valid") is False and not sample_data.get("repaired"):
            reason = (sample_data.get("recommendation", {})
                      .get("doc", {}).get("reason", "Ambdues repliques amb anomalies critiques"))
            return (COLOR_ERROR, "\u2718",
                    f"Mostra no valida \u2014 {reason}\nSeleccionar 'Cap' o generar noves dades",
                    repair_color, repair_text, repair_tooltip)

    # Merge anomalies from both replicas (deduplicate by code)
    doc_anomalies = doc_rep_data.get("anomalies", [])
    dad_anomalies = dad_rep_data.get("anomalies", [])
    all_anomalies = list(doc_anomalies)
    existing_codes = get_anomaly_codes(all_anomalies)
    for a in dad_anomalies:
        code = a.get("code") if isinstance(a, dict) else str(a).replace("_REPAIRED", "")
        if code not in existing_codes:
            all_anomalies.append(a)
            existing_codes.add(code)

    # Separar anomalies de reparacio (IRREGULAR_TOP) de la resta
    repair_codes = {"IRREGULAR_TOP", "IRREGULAR_TOP_DIRECT", "IRREGULAR_TOP_UIB"}
    anomalies_general = [a for a in all_anomalies
                         if (a.get("code") if isinstance(a, dict) else str(a).split("_REPAIRED")[0])
                         not in repair_codes]
    anomalies_repair = [a for a in all_anomalies
                        if (a.get("code") if isinstance(a, dict) else str(a).split("_REPAIRED")[0])
                        in repair_codes]

    # --- COLUMNA ESTAT: anomalies generals ---
    classified = classify_anomalies(anomalies_general)
    timeout_info = doc_rep_data.get("timeout_info", {})
    timeout_severity = timeout_info.get("severity", "OK")
    n_timeouts = timeout_info.get("n_timeouts", 0)
    replica_warnings = []
    if comparison:
        for domain in ("doc", "dad"):
            replica_warnings.extend((comparison.get(domain) or {}).get("warnings", []))

    has_blocker = bool(classified["blocker"])
    has_warn = bool(classified["warning"]
                    or (timeout_severity in ("WARNING", "CRITICAL"))
                    or replica_warnings)

    n_blocker = len(classified["blocker"])
    n_warn = len(classified["warning"])

    # Check LOD/LOQ from quantification
    # v2.2.0: quantification pot ser None quan do_quantify=False (pipeline
    # separat). Tractar None com a dict buit per no crashejar.
    quantification = (sample_data.get("quantification") or {}) if sample_data else {}
    below_lod = quantification.get("below_lod", False)
    below_loq = quantification.get("below_loq", False)
    lod_ppm = quantification.get("lod_ppm")
    loq_ppm = quantification.get("loq_ppm")

    # Check timeout composition
    timeout_composed = False
    if sample_data:
        sel_key = (sample_data.get("selected", {}) or {}).get("doc", "1")
        sel_rep = (sample_data.get("replicas", {}) or {}).get(sel_key, {})
        timeout_composed = bool(sel_rep.get("timeout_composition"))

    if has_blocker:
        status_color = COLOR_ERROR
        status_text = "\u2718"  # X
    elif below_lod and not has_warn:
        status_color = COLOR_ERROR
        status_text = "<LOD"
    elif below_loq and not has_warn:
        status_color = COLOR_WARNING
        status_text = "<LOQ"
    elif n_timeouts > 0 and not has_warn and not timeout_composed:
        status_color = COLOR_WARNING
        status_text = "\u23f1"  # timer
    elif n_timeouts > 0 and timeout_composed:
        status_color = COLOR_SUCCESS
        status_text = "\u23f1\u2713"  # timer+check
    elif has_warn:
        status_color = COLOR_WARNING
        n_total_warn = n_warn + (1 if n_timeouts > 0 else 0)
        status_text = f"\u26a0 {n_total_warn}"  # warning N
    else:
        status_color = COLOR_SUCCESS
        status_text = "\u2713"  # check

    # Tooltip anomalies
    tooltip_parts = []
    for key, label_prefix in [("blocker", "CRITIC"), ("warning", "Avis"), ("info", "Info")]:
        for a in classified[key]:
            code = a.get("code") if isinstance(a, dict) else str(a)
            entry = ANOMALY_CATALOG.get(code, {})
            lbl = (a.get("label") if isinstance(a, dict) else None) or entry.get("label", code)
            det = a.get("details", {}) if isinstance(a, dict) else {}
            if det.get("snr"):
                lbl += f" (SNR={det['snr']:.1f})"
            line = f"{label_prefix}: {lbl}"
            action = entry.get("action", "")
            if action:
                line += f"\n   \u2192 {action}"
            tooltip_parts.append(line)
    if n_timeouts > 0:
        zone_summary = timeout_info.get("zone_summary", {})
        zones_str = ", ".join(zone_summary.keys()) if zone_summary else "?"
        tooltip_parts.append(
            f"Timeouts Direct: {n_timeouts} ({timeout_severity}) \u2014 zones: {zones_str}")
        uib_ti = doc_rep_data.get("timeout_info_uib") or {}
        if uib_ti.get("n_timeouts", 0) > 0:
            uib_zone_summary = uib_ti.get("zone_summary", {})
            uib_in_peak = doc_rep_data.get("timeout_in_peak_uib", False)
            uib_zones_str = ", ".join(uib_zone_summary.keys()) if uib_zone_summary else "?"
            uib_tip = f"Timeouts UIB: {uib_ti['n_timeouts']} \u2014 zones: {uib_zones_str}"
            if uib_in_peak:
                uib_tip += " \u2014 DINS DEL PIC UIB!"
            tooltip_parts.append(uib_tip)
    if replica_warnings:
        for rw in replica_warnings:
            tooltip_parts.append(rw.get("label", rw.get("code", str(rw))) if isinstance(rw, dict) else str(rw))
    if below_lod and lod_ppm is not None:
        tooltip_parts.append(f"Sota LOD ({lod_ppm:.3f} ppm)")
    elif below_loq and loq_ppm is not None:
        tooltip_parts.append(f"Sota LOQ ({loq_ppm:.3f} ppm)")
    status_tooltip = "\n".join(tooltip_parts) if tooltip_parts else "OK"

    # --- COLUMNA REPARACIO: irregular_top + timeout composition ---
    if anomalies_repair:
        classified_r = classify_anomalies(anomalies_repair)
        n_repaired = len(classified_r["repaired"])
        n_pending = len(classified_r["blocker"]) + len(classified_r["warning"])
        can_repair = (sample_data and sample_data.get("repairable")
                      and not sample_data.get("repaired"))

        if n_repaired > 0 and n_pending == 0:
            repair_color = COLOR_SUCCESS
            repair_text = f"R\u2713 ({n_repaired})"
            repair_tooltip = "Reparaci\u00f3 aplicada \u2014 Clic per desfer o veure detalls"
        elif n_pending > 0:
            repair_color = COLOR_ERROR
            repair_text = "R" if can_repair else "\u26a0"
            repair_tooltip = "Cim irregular detectat \u2014 Clic per revisar i reparar"
        # Afegir detalls per cada replica
        rp_details = []
        for a in anomalies_repair:
            code = a.get("code", "") if isinstance(a, dict) else str(a)
            repaired = a.get("repaired", False) if isinstance(a, dict) else "_REPAIRED" in str(a)
            det = a.get("details", {}) if isinstance(a, dict) else {}
            depth = det.get("max_depth", 0)
            n_v = det.get("n_valleys", 0)
            signal = "Direct" if "DIRECT" in code else ("UIB" if "UIB" in code else "DOC")
            state = "reparat" if repaired else "pendent"
            rp_details.append(f"{signal}: {n_v} valls (prof. {depth:.1%}) \u2014 {state}")
        if rp_details:
            repair_tooltip = "\n".join(rp_details) + "\n\nClic per obrir dialeg de reparacio"

    # Timeout composable
    if sample_data:
        tc = sample_data.get("timeout_composability", {})
        if tc.get("composable"):
            repair_text = repair_text + " TC" if repair_text else "TC"
            repair_color = repair_color or "#3498DB"
            coverage = tc.get("coverage_pct", 100)
            unrep = tc.get("unrepairable_min", 0)
            if coverage < 100 and unrep > 0:
                tc_tip = (
                    f"\n\nTC: Composable ({coverage:.0f}% cobertura, "
                    f"{unrep:.1f} min solapament)\n"
                    "   \u2192 A la zona de solapament, s'usara la replica menys degradada\n"
                    "   Clic per composar repliques"
                )
            else:
                tc_tip = "\n\nTC: Timeouts composables \u2014 Clic per composar repliques"
            repair_tooltip = (repair_tooltip or "") + tc_tip
        # Already composed
        sel_key = (sample_data.get("selected", {}) or {}).get("doc", "1")
        sel_rep = (sample_data.get("replicas", {}) or {}).get(sel_key, {})
        if sel_rep.get("timeout_composition"):
            repair_text = repair_text.replace("TC", "TC\u2713") if "TC" in (repair_text or "") else "TC\u2713"
            repair_color = COLOR_SUCCESS

    return (status_color, status_text, status_tooltip,
            repair_color, repair_text, repair_tooltip)


def resolve_doc_replica(sample_data):
    """Resol la replica DOC real (gestiona 'comp' -> replica amb timeout_composition).

    Returns (replica_key, replica_data_dict).
    """
    doc_sel = (sample_data.get("selected", {}) or {}).get("doc", "1")
    replicas = sample_data.get("replicas", {})
    if doc_sel == "comp":
        for rk, rv in replicas.items():
            if rv.get("timeout_composition"):
                return rk, rv
        return doc_sel, {}
    return doc_sel, replicas.get(doc_sel, {})


def find_repair_targets(sample_name, samples_grouped):
    """Busca repliques/senyals amb anomalies de cim irregular (pendents, reparades o dismissed).

    Returns list of (rep_key, signal_type) tuples.
    """
    sample_data = samples_grouped.get(sample_name, {})
    replicas = sample_data.get("replicas", {})
    targets = []

    for rep_key, rep_data in replicas.items():
        anomalies = rep_data.get("anomalies", [])
        for signal_type, anom_key in [
            ("direct", "IRREGULAR_TOP_DIRECT"),
            ("uib", "IRREGULAR_TOP_UIB"),
        ]:
            for a in anomalies:
                if isinstance(a, dict) and a.get("code") == anom_key:
                    targets.append((rep_key, signal_type))
                    break
                elif isinstance(a, str) and anom_key in a:
                    targets.append((rep_key, signal_type))
                    break

    return targets
