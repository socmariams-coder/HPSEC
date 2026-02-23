"""
Shared helper functions for analyze_panel package.

Provides reusable table population and styling functions used by both
panel.py (main panel) and dialogs.py (detail dialog).
"""

from PySide6.QtWidgets import (
    QTableWidgetItem, QHeaderView, QAbstractItemView
)
from PySide6.QtGui import QFont, QColor, QBrush

from gui.widgets.styles import COLOR_WARNING
from ._constants import FRACTION_NAMES, FRACTION_RANGES


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
            area = areas.get("DOC", {}).get("total", 0)
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
            area = areas.get(key, {}).get("total", 0)
            snr_entry = snr_dad.get(key, {})
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
                totals[wl] = areas.get(wl, {}).get("total", 0)
        else:
            totals["DOC_D"] += doc_areas.get(frac_name, 0)
            totals["DOC_U"] += uib_areas.get(frac_name, 0)
            for wl in wavelengths:
                totals[wl] += areas.get(wl, {}).get(frac_name, 0)

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
            _set_area_cell(table, row, 4 + j, areas.get(wl, {}).get(key, 0))

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
                val = areas.get(wl, {}).get(frac_name, 0)
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
                val_254 = areas.get("A254", {}).get(frac_name, 0)
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
# Calibration comparison helpers (shared between AnalyzePanel and ReviewPanel)
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
