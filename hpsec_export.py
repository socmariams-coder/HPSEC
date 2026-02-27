"""
hpsec_export.py - Mòdul d'exportació HPSEC (Fase 4: EXPORTAR)
==============================================================

FASE 4 del pipeline de 4 fases:
- Generar Excels finals amb estructura estandarditzada
- Fulls: ID (traçabilitat), ID_BP (si COLUMN+BP), DOC, DAD, RESULTS
- Integració automàtica de dades BP quan es processa COLUMN
- Usar seleccions de rèpliques de la Fase 3 (Analitzar)

REQUEREIX:
- Fase 3: processed_data amb seleccions DOC/DAD per mostra

Usat per HPSEC_Suite.py
"""

__version__ = "2.0.0"
__version_date__ = "2026-02-10"

import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.integrate import trapezoid

logger = logging.getLogger(__name__)


def _get_max_severity(anomalies):
    """Retorna severitat màxima d'una llista d'anomalies (strings o dicts)."""
    try:
        from hpsec_warnings import get_max_anomaly_severity
        return get_max_anomaly_severity(anomalies)
    except ImportError:
        return ""


def _load_export_config():
    """Carrega configuració d'exportació des de hpsec_config.json.

    Retorna dict amb target_wavelengths i time_fractions en format [start, end].
    Fallback a valors hardcoded si la config no està disponible.
    """
    fallback = {
        "target_wavelengths": [220, 252, 254, 272, 290, 362],
        "time_fractions": {
            "BioP": [0, 18],
            "HS": [18, 23],
            "BB": [23, 30],
            "SB": [30, 40],
            "LMW": [40, 70],
        },
    }
    try:
        from hpsec_config import get_config
        cfg = get_config()

        # Wavelengths
        wls = cfg.get("wavelengths", "selected", default=None)
        if wls:
            fallback["target_wavelengths"] = wls

        # Time fractions: config format {"BioP": {"start": 10.8, "end": 18}} → [start, end]
        fractions_raw = cfg.get("time_fractions", default={})
        if fractions_raw:
            fractions = {}
            for name, info in fractions_raw.items():
                if isinstance(info, dict) and "start" in info and "end" in info:
                    fractions[name] = [info["start"], info["end"]]
                elif isinstance(info, (list, tuple)) and len(info) == 2:
                    fractions[name] = list(info)
            if fractions:
                fallback["time_fractions"] = fractions

    except Exception as e:
        logger.warning("Could not load export fractions from config: %s", e)

    return fallback


# Configuració per defecte (carregada dinàmicament)
DEFAULT_EXPORT_CONFIG = _load_export_config()


def write_final_excel(
    out_path: str,
    sample_name: str,
    sample_data: dict,
    calibration_data: dict = None,
    mode: str = "COLUMN",
    config: dict = None,
    bp_data: dict = None,
):
    """
    Escriu Excel final amb estructura estandarditzada.

    Fulls:
        ID: Traçabilitat completa (fitxers, shifts, quantificació)
        ID_BP: Traçabilitat BP (si COLUMN amb dades BP vinculades)
        DOC: Cromatogrames DOC (final, raw Direct, raw UIB)
        DAD: 6 longituds d'ona seleccionades
        RESULTS: Integracions per fraccions (+ BP_total/BP_ppm si disponible)

    Args:
        out_path: Camí del fitxer Excel
        sample_name: Nom de la mostra
        sample_data: Dict amb les dades de la rèplica seleccionada
        calibration_data: Dict amb dades de calibració (opcional)
        mode: "BP" o "COLUMN"
        config: Configuració (opcional)
        bp_data: Dades BP vinculades per aquesta mostra (opcional)

    Returns:
        dict amb info d'exportació
    """
    config = config or DEFAULT_EXPORT_CONFIG
    calibration_data = calibration_data or {}

    # Extreure dades de la rèplica
    t_doc = sample_data.get("t_doc")
    y_doc_net = sample_data.get("y_doc_net")
    y_doc_raw = sample_data.get("y_doc_raw")
    y_doc_uib_net = sample_data.get("y_doc_uib_net")
    y_doc_uib_raw = sample_data.get("y_doc_uib_raw")
    df_dad = sample_data.get("df_dad")

    # Convertir a numpy arrays
    if t_doc is not None:
        t_doc = np.asarray(t_doc)
    if y_doc_net is not None:
        y_doc_net = np.asarray(y_doc_net)
    if y_doc_raw is not None:
        y_doc_raw = np.asarray(y_doc_raw)
    if y_doc_uib_net is not None:
        y_doc_uib_net = np.asarray(y_doc_uib_net)
    if y_doc_uib_raw is not None:
        y_doc_uib_raw = np.asarray(y_doc_uib_raw)

    is_dual = y_doc_uib_net is not None and len(y_doc_uib_net) > 0

    # === FULL ID: Traçabilitat ===
    id_rows = _build_id_sheet(sample_name, sample_data, calibration_data, mode, is_dual)
    df_id = pd.DataFrame(id_rows, columns=["Field", "Value"])

    # === FULL ID_BP: Traçabilitat BP (si COLUMN amb BP) ===
    df_id_bp = None
    if bp_data and mode == "COLUMN":
        bp_id_rows = _build_bp_id_sheet(sample_name, bp_data, calibration_data)
        df_id_bp = pd.DataFrame(bp_id_rows, columns=["Field", "Value"])

    # === FULL DOC: Cromatogrames ===
    df_doc = _build_doc_sheet(t_doc, y_doc_net, y_doc_raw, y_doc_uib_net, y_doc_uib_raw, is_dual)

    # === FULL DAD: 6 longituds d'ona ===
    df_dad_export = _build_dad_sheet(df_dad, config)

    # === FULL RESULTS: Integracions ===
    df_results = _build_results_sheet(t_doc, y_doc_net, df_dad, mode, config, bp_data=bp_data)

    # Escriure Excel: ID | [ID_BP] | DOC | DAD | RESULTS
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_id.to_excel(writer, sheet_name="ID", index=False)
        if df_id_bp is not None and not df_id_bp.empty:
            df_id_bp.to_excel(writer, sheet_name="ID_BP", index=False)
        df_doc.to_excel(writer, sheet_name="DOC", index=False)
        if df_dad_export is not None and not df_dad_export.empty:
            df_dad_export.to_excel(writer, sheet_name="DAD", index=False)
        df_results.to_excel(writer, sheet_name="RESULTS", index=False)

    return {
        "success": True,
        "path": out_path,
        "sample": sample_name,
        "n_doc_points": len(t_doc) if t_doc is not None else 0,
        "n_dad_points": len(df_dad) if df_dad is not None else 0,
        "is_dual": is_dual,
        "has_bp": bp_data is not None,
    }


def _build_id_sheet(sample_name, sample_data, calibration_data, mode, is_dual):
    """Construeix les files del full ID (traçabilitat)."""
    rows = [
        ("Export_Version", f"hpsec_export v{__version__}"),
        ("Export_Date", datetime.now().strftime("%Y-%m-%d %H:%M")),
        ("---", "---"),
        # Identificació mostra
        ("Sample", sample_name),
        ("Replica_DOC", sample_data.get("replica", "?")),
        ("Replica_DAD", sample_data.get("replica_dad", sample_data.get("replica", "?"))),
        ("SEQ", sample_data.get("seq_name", "")),
        ("Method", mode),
        ("Sample_Type", sample_data.get("sample_type", "SAMPLE")),
        ("---", "---"),
    ]

    # Fitxers origen
    rows.append(("File_DOC_Direct", sample_data.get("file_doc", "")))
    if is_dual:
        rows.append(("File_DOC_UIB", sample_data.get("file_doc_uib", sample_data.get("file_doc", ""))))
    rows.append(("File_DAD", sample_data.get("file_dad", "")))
    rows.append(("File_MasterFile", sample_data.get("master_file", "")))
    rows.append(("---", "---"))

    # Shifts aplicats
    shift_direct = sample_data.get("shift_direct")
    shift_uib = sample_data.get("shift_uib")
    if shift_direct is not None:
        rows.append(("Shift_Direct_sec", round(shift_direct * 60, 2)))
    if is_dual and shift_uib is not None:
        rows.append(("Shift_UIB_sec", round(shift_uib * 60, 2)))
    rows.append(("---", "---"))

    # Baseline
    baseline_direct = sample_data.get("baseline_direct")
    baseline_uib = sample_data.get("baseline_uib")
    if baseline_direct is not None:
        rows.append(("Baseline_Direct_mAU", round(float(baseline_direct), 3)))
    if is_dual and baseline_uib is not None:
        rows.append(("Baseline_UIB_mAU", round(float(baseline_uib), 3)))
    rows.append(("---", "---"))

    # SNR info
    snr_info = sample_data.get("snr_info", {})
    if snr_info:
        if snr_info.get("snr_direct") is not None:
            rows.append(("SNR_Direct", round(snr_info["snr_direct"], 1)))
        if snr_info.get("lod_direct") is not None:
            rows.append(("LOD_Direct_mAU", round(snr_info["lod_direct"], 3)))
        if snr_info.get("loq_direct") is not None:
            rows.append(("LOQ_Direct_mAU", round(snr_info["loq_direct"], 3)))
        if is_dual:
            if snr_info.get("snr_uib") is not None:
                rows.append(("SNR_UIB", round(snr_info["snr_uib"], 1)))
            if snr_info.get("lod_uib") is not None:
                rows.append(("LOD_UIB_mAU", round(snr_info["lod_uib"], 3)))
        rows.append(("---", "---"))

    # Injecció i mode
    inj_volume = sample_data.get("inj_volume")
    data_mode = sample_data.get("data_mode", "")
    seq_date = sample_data.get("seq_date", "")
    if seq_date:
        rows.append(("Date", seq_date))
    if inj_volume:
        rows.append(("INJ_Volume_uL", inj_volume))
    inj_idx = sample_data.get("injection_index")
    if inj_idx is not None:
        rows.append(("Injection_Index", inj_idx))
    if data_mode:
        rows.append(("Data_Mode", data_mode))
    rows.append(("---", "---"))

    # Quantificació (si hi ha calibració)
    quantification = sample_data.get("quantification", {})
    if quantification:
        conc = quantification.get("concentration_ppm")
        if conc is not None:
            rows.append(("Concentration_ppm", round(conc, 3)))
        conc_d = quantification.get("concentration_ppm_direct")
        if conc_d is not None and conc_d != conc:
            rows.append(("Concentration_ppm_Direct", round(conc_d, 3)))
        conc_u = quantification.get("concentration_ppm_uib")
        if conc_u is not None:
            rows.append(("Concentration_ppm_UIB", round(conc_u, 3)))
        cr = quantification.get("calibration_ratio")
        if cr is not None:
            rows.append(("Calibration_Ratio", round(cr, 4)))
        area_total = quantification.get("area_total")
        if area_total is not None:
            rows.append(("Area_DOC_total", round(area_total, 2)))
        rows.append(("---", "---"))

    # Info calibració
    if calibration_data:
        cal_date = calibration_data.get("date", "")
        cal_khp = calibration_data.get("khp_conc_ppm", "")
        rows.append(("Calibration_Date", cal_date))
        rows.append(("Calibration_KHP_ppm", cal_khp))
        # RF i Intercept usats
        rf_obj = calibration_data.get("rf_mass_cal", {})
        if isinstance(rf_obj, dict):
            for sig_type in ("direct", "uib"):
                sig_rf = rf_obj.get(sig_type, {})
                if isinstance(sig_rf, dict):
                    for mode_key, val in sig_rf.items():
                        rows.append((f"RF_mass_cal_{sig_type}_{mode_key}", val))
        elif rf_obj:
            rows.append(("RF_mass_cal", rf_obj))
        intercept_obj = calibration_data.get("intercept", {})
        if isinstance(intercept_obj, dict):
            for sig_type in ("direct", "uib"):
                sig_int = intercept_obj.get(sig_type, {})
                if isinstance(sig_int, dict):
                    for mode_key, val in sig_int.items():
                        rows.append((f"Intercept_{sig_type}_{mode_key}", val))
        elif intercept_obj:
            rows.append(("Intercept", intercept_obj))
        cal_seq = calibration_data.get("seq_name", "")
        if cal_seq:
            rows.append(("Calibration_SEQ", cal_seq))
        rows.append(("---", "---"))

    # Anomalies detectades
    anomalies = sample_data.get("anomalies", [])
    if anomalies:
        for a in anomalies:
            if isinstance(a, dict):
                code = a.get("code", "")
                sev = a.get("severity", "info")
                rep = " [REPAIRED]" if a.get("repaired") else ""
                rows.append((f"Anomaly_{code}", f"{sev}{rep}"))
            else:
                rows.append(("Anomaly", str(a)))

    # Bi-Gaussian info (BP mode)
    bigaussian = sample_data.get("bigaussian_doc")
    if bigaussian and isinstance(bigaussian, dict):
        r2_bg = bigaussian.get("r2") or bigaussian.get("r_squared")
        if r2_bg is not None:
            rows.append(("Bigaussian_R2", round(r2_bg, 4)))
        asym = bigaussian.get("asymmetry")
        if asym is not None:
            rows.append(("Bigaussian_Asymmetry", round(asym, 3)))

    # Timeout info
    timeout_info = sample_data.get("timeout_info", {})
    if timeout_info.get("n_timeouts", 0) > 0:
        rows.append(("Timeout_Severity", timeout_info.get("severity", "")))
        zone_summary = timeout_info.get("zone_summary", {})
        if zone_summary:
            zones_str = "; ".join(f"{zone}: {count}" for zone, count in zone_summary.items())
            rows.append(("Timeout_Zones", zones_str))
    if timeout_info.get("toc_minute_precision"):
        rows.append(("TOC_Timestamp_Repair", "YES — cadencia 4s reconstruida (timestamps originals arrodonits al minut)"))

    # Irregular top repair info (jagged/batman artifact)
    irregular_top_repaired = sample_data.get("irregular_top_direct_repaired",
                                              sample_data.get("batman_direct_repaired"))
    if irregular_top_repaired:
        rows.append(("Irregular_Top_Direct_Repaired", "YES"))
        repair_info = sample_data.get("irregular_top_direct_repair_info",
                                       sample_data.get("batman_direct_repair_info", {}))
        if repair_info:
            rows.append(("Irregular_Top_Y_Max_Original", repair_info.get("y_max_original", "")))
            rows.append(("Irregular_Top_Y_Max_Theoretical", repair_info.get("y_max_theoretical", "")))

    return rows


def _build_bp_id_sheet(sample_name, bp_data, calibration_data):
    """Construeix les files del full ID_BP (traçabilitat BP vinculada).

    Args:
        sample_name: Nom de la mostra
        bp_data: Dict amb dades BP de la mostra
        calibration_data: Dict amb dades de calibració

    Returns:
        list de tuples (Field, Value)
    """
    rows = [
        ("Export_Version", f"hpsec_export v{__version__}"),
        ("Export_Date", datetime.now().strftime("%Y-%m-%d %H:%M")),
        ("---", "---"),
        # Identificació
        ("Sample", sample_name),
        ("Method", "BP"),
        ("SEQ_BP", bp_data.get("seq_name", bp_data.get("bp_seq", ""))),
        ("Date_BP", bp_data.get("seq_date", bp_data.get("date", ""))),
        ("---", "---"),
    ]

    # Rèplica seleccionada
    replica = bp_data.get("replica", bp_data.get("selected_replica", ""))
    if replica:
        rows.append(("Replica_BP", replica))

    # SNR
    snr = bp_data.get("snr_direct", bp_data.get("snr", None))
    if snr is not None:
        rows.append(("SNR_BP", round(float(snr), 1)))

    # Volum injecció
    inj_vol = bp_data.get("inj_volume")
    if inj_vol:
        rows.append(("INJ_Volume_uL_BP", inj_vol))
    rows.append(("---", "---"))

    # Quantificació
    area_total = bp_data.get("area_total")
    if area_total is not None:
        rows.append(("Area_BP_total", round(float(area_total), 2)))
    conc_ppm = bp_data.get("concentration_ppm")
    if conc_ppm is not None:
        rows.append(("Concentration_ppm_BP", round(float(conc_ppm), 3)))
    rows.append(("---", "---"))

    # Calibració BP usada
    if calibration_data:
        rf_obj = calibration_data.get("rf_mass_cal", {})
        if isinstance(rf_obj, dict):
            rf_bp = rf_obj.get("direct", {}).get("bp")
            if rf_bp is not None:
                rows.append(("RF_mass_cal_BP", rf_bp))
        intercept_obj = calibration_data.get("intercept", {})
        if isinstance(intercept_obj, dict):
            int_bp = intercept_obj.get("direct", {}).get("bp")
            if int_bp is not None:
                rows.append(("Intercept_BP", int_bp))
        rows.append(("---", "---"))

    # Anomalies
    anomalies = bp_data.get("anomalies", [])
    if anomalies:
        for a in anomalies:
            if isinstance(a, dict):
                code = a.get("code", "")
                sev = a.get("severity", "info")
                rep = " [REPAIRED]" if a.get("repaired") else ""
                rows.append((f"Anomaly_BP_{code}", f"{sev}{rep}"))
            else:
                rows.append(("Anomaly_BP", str(a)))

    # Bigaussian info
    bigaussian = bp_data.get("bigaussian_doc", bp_data.get("bigaussian", {}))
    if bigaussian and isinstance(bigaussian, dict):
        r2_bg = bigaussian.get("r2") or bigaussian.get("r_squared")
        if r2_bg is not None:
            rows.append(("Bigaussian_R2_BP", round(float(r2_bg), 4)))
        asym = bigaussian.get("asymmetry")
        if asym is not None:
            rows.append(("Bigaussian_Asymmetry_BP", round(float(asym), 3)))

    # Timeout info
    timeout_info = bp_data.get("timeout_info", {})
    if timeout_info.get("n_timeouts", 0) > 0:
        rows.append(("Timeout_Severity_BP", timeout_info.get("severity", "")))
        zone_summary = timeout_info.get("zone_summary", {})
        if zone_summary:
            zones_str = "; ".join(f"{zone}: {count}" for zone, count in zone_summary.items())
            rows.append(("Timeout_Zones_BP", zones_str))

    return rows


def _build_doc_sheet(t_doc, y_doc_net, y_doc_raw, y_doc_uib_net, y_doc_uib_raw, is_dual):
    """Construeix el DataFrame del full DOC."""
    if t_doc is None or len(t_doc) == 0:
        return pd.DataFrame({"time (min)": [], "DOC (mAU)": []})

    data = {"time (min)": t_doc}

    # Senyal final (corregit)
    if y_doc_net is not None:
        data["DOC (mAU)"] = y_doc_net
    else:
        data["DOC (mAU)"] = np.zeros_like(t_doc)

    # Raw Direct
    if y_doc_raw is not None and len(y_doc_raw) == len(t_doc):
        data["DOC_Direct_RAW (mAU)"] = y_doc_raw

    # UIB (si DUAL)
    if is_dual:
        if y_doc_uib_net is not None and len(y_doc_uib_net) == len(t_doc):
            data["DOC_UIB (mAU)"] = y_doc_uib_net
        if y_doc_uib_raw is not None and len(y_doc_uib_raw) == len(t_doc):
            data["DOC_UIB_RAW (mAU)"] = y_doc_uib_raw

    return pd.DataFrame(data)


def _build_dad_sheet(df_dad, config):
    """Construeix el DataFrame del full DAD amb 6 longituds d'ona."""
    if df_dad is None or df_dad.empty:
        return None

    target_wls = config.get("target_wavelengths", [220, 252, 254, 272, 290, 362])

    # Columna temps
    if "time (min)" in df_dad.columns:
        result = {"time (min)": df_dad["time (min)"].values}
    elif "Time" in df_dad.columns:
        result = {"time (min)": df_dad["Time"].values}
    else:
        return None

    # Afegir cada longitud d'ona
    for wl in target_wls:
        col_name = str(wl)
        if col_name in df_dad.columns:
            result[f"A{wl}"] = df_dad[col_name].values
        elif f"A{wl}" in df_dad.columns:
            result[f"A{wl}"] = df_dad[f"A{wl}"].values
        else:
            # Buscar columna aproximada
            for col in df_dad.columns:
                try:
                    if abs(float(col) - wl) < 1:
                        result[f"A{wl}"] = df_dad[col].values
                        break
                except (ValueError, TypeError):
                    pass

    return pd.DataFrame(result)


def _build_results_sheet(t_doc, y_doc_net, df_dad, mode, config, bp_data=None):
    """Construeix el DataFrame del full RESULTS amb integracions.

    Si bp_data és proporcionat, afegeix files BP_total i BP_ppm.
    """
    target_wls = config.get("target_wavelengths", [220, 252, 254, 272, 290, 362])
    fractions = config.get("time_fractions", {})

    # Header
    header = ["Fraction", "Range (min)", "DOC"]
    for wl in target_wls:
        header.append(f"A{wl}")

    rows = []

    # Calcular àrees per cada fracció
    fraction_names = list(fractions.keys())
    if mode == "BP":
        # Per BP o mostres sense fraccions, només total
        fraction_names = ["total"]
    else:
        fraction_names.append("total")

    for frac_name in fraction_names:
        if frac_name == "total":
            t_ini, t_fi = 0, 70
            rang = "0-70"
        else:
            t_ini, t_fi = fractions.get(frac_name, [0, 0])
            rang = f"{t_ini}-{t_fi}"

        row = [frac_name, rang]

        # Àrea DOC
        area_doc = _integrate_fraction(t_doc, y_doc_net, t_ini, t_fi)
        row.append(round(area_doc, 2) if area_doc > 0 else "-")

        # Àrees DAD
        for wl in target_wls:
            area_dad = _integrate_dad_fraction(df_dad, wl, t_ini, t_fi)
            row.append(round(area_dad, 2) if area_dad > 0 else "-")

        rows.append(row)

    # Afegir files BP si disponible
    if bp_data and mode == "COLUMN":
        # Fila separador
        rows.append(["---"] + ["---"] * (len(header) - 1))

        # BP_total: àrea total BP
        bp_area = bp_data.get("area_total")
        bp_row = ["BP_total", "BP", round(float(bp_area), 2) if bp_area else "-"]
        bp_row.extend(["-"] * len(target_wls))
        rows.append(bp_row)

        # BP_ppm: concentració BP
        bp_ppm = bp_data.get("concentration_ppm")
        ppm_row = ["BP_ppm", "BP", round(float(bp_ppm), 3) if bp_ppm else "-"]
        ppm_row.extend(["-"] * len(target_wls))
        rows.append(ppm_row)

    return pd.DataFrame(rows, columns=header)


def _integrate_fraction(t, y, t_ini, t_fi):
    """Integra una fracció temporal."""
    if t is None or y is None or len(t) == 0:
        return 0.0

    mask = (t >= t_ini) & (t <= t_fi)
    if not np.any(mask):
        return 0.0

    t_frac = t[mask]
    y_frac = y[mask]

    if len(t_frac) < 2:
        return 0.0

    return float(trapezoid(y_frac, t_frac))


def _integrate_dad_fraction(df_dad, wl, t_ini, t_fi):
    """Integra una fracció temporal per una longitud d'ona DAD."""
    if df_dad is None or df_dad.empty:
        return 0.0

    # Trobar columna temps
    if "time (min)" in df_dad.columns:
        t_col = "time (min)"
    elif "Time" in df_dad.columns:
        t_col = "Time"
    else:
        return 0.0

    # Trobar columna WL
    wl_col = None
    for col in [str(wl), f"A{wl}"]:
        if col in df_dad.columns:
            wl_col = col
            break

    if wl_col is None:
        # Buscar aproximada
        for col in df_dad.columns:
            try:
                if abs(float(col) - wl) < 1:
                    wl_col = col
                    break
            except (ValueError, TypeError):
                pass

    if wl_col is None:
        return 0.0

    t = df_dad[t_col].values
    y = df_dad[wl_col].values

    return _integrate_fraction(t, y, t_ini, t_fi)


# =============================================================================
# INTEGRACIÓ BP
# =============================================================================

def _find_and_load_bp_data(seq_path, samples_grouped):
    """Cerca SEQ BP i carrega dades BP per cada mostra.

    Args:
        seq_path: Path de la seqüència COLUMN
        samples_grouped: Dict de mostres agrupades

    Returns:
        dict: {
            "bp_seq_path": str o None,
            "bp_seq_name": str o None,
            "samples": {sample_name: bp_data_dict, ...}
        }
    """
    result = {
        "bp_seq_path": None,
        "bp_seq_name": None,
        "samples": {},
    }

    try:
        from hpsec_consolidate import find_matching_bp_sequence, load_bp_data_for_sample

        bp_path = find_matching_bp_sequence(seq_path)
        if not bp_path:
            return result

        result["bp_seq_path"] = bp_path
        result["bp_seq_name"] = Path(bp_path).name

        for sample_name in samples_grouped:
            bp_data = load_bp_data_for_sample(bp_path, sample_name)
            if bp_data:
                # Afegir info SEQ BP
                bp_data["bp_seq"] = result["bp_seq_name"]
                result["samples"][sample_name] = bp_data

    except Exception as e:
        print(f"[WARNING] Error carregant dades BP: {e}")

    return result


# =============================================================================
# EXPORTACIÓ LIGHTWEIGHT PER BLANCS / CONTROLS
# =============================================================================

def write_light_excel(
    out_path: str,
    sample_name: str,
    sample_data: dict,
    config: dict = None,
):
    """
    Escriu Excel lightweight per BLANK/CONTROL (només ID + DOC).

    Args:
        out_path: Camí del fitxer Excel
        sample_name: Nom de la mostra
        sample_data: Dict amb dades de la rèplica light
        config: Configuració (opcional)

    Returns:
        dict amb info d'exportació
    """
    # === FULL ID: Traçabilitat ===
    id_rows = [
        ("Export_Version", f"hpsec_export v{__version__}"),
        ("Export_Date", datetime.now().strftime("%Y-%m-%d %H:%M")),
        ("---", "---"),
        ("Sample", sample_name),
        ("Sample_Type", sample_data.get("sample_type", "BLANK")),
        ("Replica", sample_data.get("replica", "?")),
        ("Analysis_Type", "light"),
        ("---", "---"),
        ("Area_Total_DOC", sample_data.get("area_total", 0)),
        ("SNR", sample_data.get("snr", 0)),
        ("Inj_Volume_uL", sample_data.get("inj_volume", "")),
    ]
    df_id = pd.DataFrame(id_rows, columns=["Field", "Value"])

    # === FULL DOC: Cromatograma ===
    t_doc = sample_data.get("t_doc")
    y_doc_net = sample_data.get("y_doc_net")

    doc_data = {}
    if t_doc is not None:
        doc_data["Time_min"] = np.asarray(t_doc)
    if y_doc_net is not None:
        doc_data["DOC_net"] = np.asarray(y_doc_net)

    df_doc = pd.DataFrame(doc_data) if doc_data else pd.DataFrame()

    # Escriure Excel: ID | DOC
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_id.to_excel(writer, sheet_name="ID", index=False)
        if not df_doc.empty:
            df_doc.to_excel(writer, sheet_name="DOC", index=False)

    return {
        "success": True,
        "path": out_path,
        "sample": sample_name,
        "analysis_type": "light",
        "n_doc_points": len(t_doc) if t_doc is not None else 0,
    }


# =============================================================================
# EXPORTACIÓ DE SEQÜÈNCIES
# =============================================================================

def export_sequence(
    samples_grouped: dict,
    output_dir: str,
    calibration_data: dict = None,
    mode: str = "COLUMN",
    config: dict = None,
    progress_callback=None,
    seq_path: str = None,
    bp_resolved: dict = None,
):
    """
    Exporta totes les mostres d'una seqüència.

    Args:
        samples_grouped: Dict amb mostres agrupades
        output_dir: Directori de sortida
        calibration_data: Dades de calibració
        mode: "BP" o "COLUMN"
        config: Configuració
        progress_callback: Funció per reportar progrés (pct, msg)
        seq_path: Path de la seqüència (per cercar BP automàticament)
        bp_resolved: Dades BP pre-resoltes des del wizard (evita re-descobriment)

    Returns:
        dict amb resultats d'exportació
    """
    config = config or DEFAULT_EXPORT_CONFIG
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "success": True,
        "n_exported": 0,
        "n_errors": 0,
        "files": [],
        "errors": [],
        "bp_info": None,
    }

    # Cercar i carregar dades BP si mode COLUMN
    bp_all = {}
    if mode == "COLUMN":
        if bp_resolved:
            # Usar dades pre-resoltes (wizard)
            for name, sdata in bp_resolved.get("samples", {}).items():
                if sdata and sdata.get("bp_data"):
                    bp_all[name] = sdata["bp_data"]
            bp_info = bp_resolved.get("primary_bp")
            if bp_info:
                results["bp_info"] = {
                    "bp_seq_path": bp_info.get("path"),
                    "bp_seq_name": bp_info.get("name"),
                    "n_linked": len(bp_all),
                }
        elif seq_path:
            # Fallback: descobriment automàtic (batch, backwards compat)
            if progress_callback:
                progress_callback(0, "Cercant dades BP vinculades...")
            bp_result = _find_and_load_bp_data(seq_path, samples_grouped)
            bp_all = bp_result.get("samples", {})
            if bp_result.get("bp_seq_path"):
                results["bp_info"] = {
                    "bp_seq_path": bp_result["bp_seq_path"],
                    "bp_seq_name": bp_result["bp_seq_name"],
                    "n_linked": len(bp_all),
                }

    total = len(samples_grouped)
    n_skipped = 0
    for i, (sample_name, sample_info) in enumerate(samples_grouped.items()):
        if progress_callback:
            pct = 5 + int((i / total) * 90)
            progress_callback(pct, f"Exportant {sample_name}...")

        try:
            # Light samples (CONTROL) — exportació lightweight
            if sample_info.get("analysis_type") == "light":
                selected = sample_info.get("selected", {})
                doc_replica = selected.get("doc", "1")
                replicas = sample_info.get("replicas", {})
                doc_data = replicas.get(doc_replica, {})

                sample_type = sample_info.get("sample_type", "BLANK")
                filename = f"{sample_name}_R{doc_replica}.xlsx"
                filepath = output_path / filename

                result = write_light_excel(
                    str(filepath),
                    sample_name,
                    doc_data,
                    config,
                )

                results["files"].append({
                    "sample": sample_name,
                    "path": str(filepath),
                    "doc_replica": doc_replica,
                    "analysis_type": "light",
                })
                results["n_exported"] += 1
                continue

            # Saltar mostres no vàlides o amb selecció "cap"
            selected = sample_info.get("selected", {})
            doc_replica = selected.get("doc", "1")
            dad_replica = selected.get("dad", doc_replica)

            if doc_replica == "none" or sample_info.get("sample_valid") is False:
                n_skipped += 1
                continue

            replicas = sample_info.get("replicas", {})
            doc_data = replicas.get(doc_replica, {})
            dad_data = replicas.get(dad_replica, {})

            # Combinar dades (DOC de doc_replica, DAD de dad_replica)
            export_data = dict(doc_data)  # Copiar DOC data
            export_data["replica"] = doc_replica
            export_data["replica_dad"] = dad_replica

            # Si DAD ve d'altra rèplica, substituir
            if dad_replica != doc_replica and "df_dad" in dad_data:
                export_data["df_dad"] = dad_data["df_dad"]

            # Afegir quantificació si existeix
            if "quantification" in sample_info:
                export_data["quantification"] = sample_info["quantification"]

            # BP data per aquesta mostra
            bp_data = bp_all.get(sample_name)

            # Propagar bp_linked a sample_info per generate_summary_excel
            if bp_data:
                sample_info["bp_linked"] = bp_data

            # Nom del fitxer
            filename = f"{sample_name}_R{doc_replica}.xlsx"
            if dad_replica != doc_replica:
                filename = f"{sample_name}_DOC-R{doc_replica}_DAD-R{dad_replica}.xlsx"

            filepath = output_path / filename

            # Exportar
            result = write_final_excel(
                str(filepath),
                sample_name,
                export_data,
                calibration_data,
                mode,
                config,
                bp_data=bp_data,
            )

            results["files"].append({
                "sample": sample_name,
                "path": str(filepath),
                "doc_replica": doc_replica,
                "dad_replica": dad_replica,
                "has_bp": bp_data is not None,
            })
            results["n_exported"] += 1

        except Exception as e:
            results["errors"].append(f"{sample_name}: {str(e)}")
            results["n_errors"] += 1

    if progress_callback:
        progress_callback(100, "Exportació completada")

    results["n_skipped"] = n_skipped
    results["success"] = results["n_errors"] == 0
    return results


# =============================================================================
# GENERACIÓ DE RESUM
# =============================================================================

def generate_summary_excel(
    samples_grouped: dict,
    output_path: str,
    calibration_data: dict = None,
    mode: str = "COLUMN",
    config: dict = None,
):
    """
    Genera un Excel resum amb totes les mostres.

    Fulls:
        SUMMARY: Una fila per mostra amb concentració, SNR, warnings, BP info
        CALIBRATION: Info de calibració usada

    Args:
        samples_grouped: Dict amb mostres agrupades
        output_path: Camí del fitxer Excel
        calibration_data: Dades de calibració
        mode: "BP" o "COLUMN"
        config: Configuració

    Returns:
        dict amb info d'exportació
    """
    config = config or DEFAULT_EXPORT_CONFIG
    target_wls = config.get("target_wavelengths", [220, 252, 254, 272, 290, 362])

    # === FULL SUMMARY ===
    summary_rows = []
    for sample_name in sorted(samples_grouped.keys()):
        sample_info = samples_grouped[sample_name]

        # Light samples (BLANK/CONTROL) — fila simplificada
        if sample_info.get("analysis_type") == "light":
            selected = sample_info.get("selected", {})
            doc_replica = selected.get("doc", "1")
            replicas = sample_info.get("replicas", {})
            doc_data = replicas.get(doc_replica, {})
            sample_type = sample_info.get("sample_type", "BLANK")

            row = {
                "Sample": sample_name,
                "Type": sample_type,
                "Inj_Index": doc_data.get("injection_index", ""),
                "DOC_Replica": f"R{doc_replica}",
                "DAD_Replica": None,
                "Conc_ppm": None,
                "Area_total": doc_data.get("area_total"),
                "A_UIB": None,
                "ppm_UIB": None,
                "A_254": None,
                "SNR_Direct": doc_data.get("snr"),
                "SNR_UIB": None,
                "SNR_254": None,
                "R2_DOC": None,
                "R2_DAD": None,
                "Anomalies": "",
                "Warnings": "",
                "HCI": None,
                "HCI_Character": "",
            }
            summary_rows.append(row)
            continue

        selected = sample_info.get("selected", {})
        quantification = sample_info.get("quantification", {})
        comparison = sample_info.get("comparison", {})

        doc_replica = selected.get("doc", "1")
        dad_replica = selected.get("dad", "1")

        is_invalid = (doc_replica == "none"
                      or sample_info.get("sample_valid") is False
                      or sample_info.get("skip_quantification", False))

        doc_data = sample_info.get("replicas", {}).get(doc_replica, {})
        snr_info = doc_data.get("snr_info", {})

        # Determine sample type
        sample_type = doc_data.get("sample_type", "SAMPLE")

        # Warnings
        doc_warnings = comparison.get("doc", {}).get("warnings", []) if comparison else []
        dad_warnings = comparison.get("dad", {}).get("warnings", []) if comparison else []
        all_warnings = doc_warnings + dad_warnings
        # Anomalies de la rèplica seleccionada
        anomalies = doc_data.get("anomalies", [])

        # Dades DAD (pot ser diferent rèplica)
        dad_data = sample_info.get("replicas", {}).get(dad_replica, {}) if dad_replica != "none" else {}
        dad_areas = (dad_data.get("areas") or {})
        area_254 = dad_areas.get("A254", {}).get("total", 0)
        snr_info_dad = dad_data.get("snr_info_dad") or {}
        snr_254 = snr_info_dad.get("A254", {}).get("snr", 0)

        # R² values
        r2_doc = comparison.get("doc", {}).get("pearson", 0) if comparison else 0
        r2_dad = comparison.get("dad", {}).get("pearson_min", 0) if comparison else 0

        # Àrees UIB
        areas_uib = doc_data.get("areas_uib") or {}
        area_uib = areas_uib.get("total", 0)
        ppm_uib = quantification.get("concentration_ppm_uib")

        row = {
            "Sample": sample_name,
            "Type": sample_type,
            "Inj_Index": doc_data.get("injection_index", ""),
            "DOC_Replica": "Cap" if doc_replica == "none" else f"R{doc_replica}",
            "DAD_Replica": "Cap" if dad_replica == "none" else f"R{dad_replica}",
            "Conc_ppm": "NO VÀLIDA" if is_invalid else quantification.get("concentration_ppm"),
            "Area_total": quantification.get("area_total"),
            "A_UIB": area_uib if area_uib else None,
            "ppm_UIB": ppm_uib,
            "A_254": area_254 if area_254 else None,
            "SNR_Direct": snr_info.get("snr_direct"),
            "SNR_UIB": snr_info.get("snr_uib"),
            "SNR_254": snr_254 if snr_254 else None,
            "R2_DOC": round(r2_doc, 4) if r2_doc > 0 else None,
            "R2_DAD": round(r2_dad, 4) if r2_dad > 0 else None,
            "Anomalies": "; ".join(
                (a.get("code", "") + ("[R]" if a.get("repaired") else "")) if isinstance(a, dict)
                else str(a) for a in anomalies
            ) if anomalies else "",
            "Max_Severity": _get_max_severity(anomalies),
            "Warnings": "; ".join(
                (w.get("code", "") if isinstance(w, dict) else str(w)) for w in all_warnings
            ) if all_warnings else "",
            "HCI": quantification.get("hci"),
            "HCI_Character": quantification.get("hci_character", ""),
        }

        # BP linked info
        bp_info = sample_info.get("bp_linked", {})
        if bp_info:
            row["BP_SEQ"] = bp_info.get("bp_seq", bp_info.get("seq_name", ""))
            row["BP_Area"] = bp_info.get("area_total")
            row["BP_ppm"] = bp_info.get("concentration_ppm")

        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)

    # === FULL CALIBRATION ===
    cal_rows = []
    if calibration_data:
        for key, value in calibration_data.items():
            if not isinstance(value, (list, dict)):
                cal_rows.append((key, value))

    df_cal = pd.DataFrame(cal_rows, columns=["Field", "Value"]) if cal_rows else pd.DataFrame()

    # Escriure Excel
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="SUMMARY", index=False)
        if not df_cal.empty:
            df_cal.to_excel(writer, sheet_name="CALIBRATION", index=False)

    return {
        "success": True,
        "path": output_path,
        "n_samples": len(summary_rows),
    }


# =============================================================================
# PATCH EXCEL PER RECALIBRACIÓ
# =============================================================================

def patch_excel_calibration(excel_path, new_rf, new_intercept, mode="COLUMN",
                             new_rf_uib=None, new_intercept_uib=None):
    """
    Patch quirúrgic d'un Excel individual (per mostra) amb nous RF/intercept.

    Modifica NOMÉS les cel·les que depenen de la calibració:
    - Full ID: Concentration_ppm*, RF_mass_cal_*, Intercept_*
    - Full RESULTS: fila BP_ppm (si existeix)

    No regenera tot l'Excel — patch via openpyxl.

    Args:
        excel_path: Path al fitxer Excel (.xlsx)
        new_rf: Nou RF mass cal per direct
        new_intercept: Nou intercept per direct
        mode: "COLUMN" o "BP"
        new_rf_uib: RF per UIB (None = no tocar)
        new_intercept_uib: Intercept per UIB (None = no tocar)

    Returns:
        dict amb {success, patched_fields, errors}
    """
    import openpyxl

    result = {"success": False, "patched_fields": 0, "errors": []}

    try:
        wb = openpyxl.load_workbook(excel_path)
    except Exception as e:
        result["errors"].append(f"Error obrint Excel: {e}")
        return result

    n_patched = 0
    mode_key = mode.lower()
    signal_key = "direct"

    # === FULL ID ===
    if "ID" in wb.sheetnames:
        ws = wb["ID"]
        # El full ID és clau-valor: col A = nom camp, col B = valor
        # Primer llegir àrea i volum per recalcular concentració
        area_total = None
        volume_uL = None
        area_uib = None

        for row in ws.iter_rows(min_col=1, max_col=2):
            key = row[0].value
            if key == "Area_DOC_total":
                area_total = row[1].value
            elif key == "INJ_Volume_uL":
                volume_uL = row[1].value

        if area_total is not None and volume_uL and volume_uL > 0 and new_rf > 0:
            # Recalcular concentracions
            ppm = max(0, float(area_total) - new_intercept) * 1000 / (new_rf * float(volume_uL))

            for row in ws.iter_rows(min_col=1, max_col=2):
                key = row[0].value
                if key == "Concentration_ppm":
                    row[1].value = round(ppm, 3)
                    n_patched += 1
                elif key == "Concentration_ppm_Direct":
                    row[1].value = round(ppm, 3)
                    n_patched += 1
                elif key == "Concentration_ppm_UIB" and new_rf_uib and new_rf_uib > 0:
                    # Cal llegir àrea UIB — buscar Area_UIB_total o recalcular
                    # Per ara, marcar com a pendent
                    pass
                elif key == f"RF_mass_cal_{signal_key}_{mode_key}":
                    row[1].value = new_rf
                    n_patched += 1
                elif key == f"Intercept_{signal_key}_{mode_key}":
                    row[1].value = new_intercept
                    n_patched += 1

    # === FULL ID_BP (si existeix, per COLUMN amb BP vinculada) ===
    if "ID_BP" in wb.sheetnames:
        ws_bp = wb["ID_BP"]
        bp_area = None
        bp_volume = None
        for row in ws_bp.iter_rows(min_col=1, max_col=2):
            key = row[0].value
            if key == "Area_BP_total":
                bp_area = row[1].value
            elif key == "INJ_Volume_uL":
                bp_volume = row[1].value

        # BP usa el seu propi RF (no el COLUMN)
        # No patchejar BP des d'aquí — caldria RF_BP separat

    # === FULL RESULTS (fila BP_ppm) ===
    if "RESULTS" in wb.sheetnames:
        ws_res = wb["RESULTS"]
        for row in ws_res.iter_rows(min_col=1, max_col=3):
            if row[0].value == "BP_ppm":
                # Buscar àrea BP a la fila anterior (BP_total)
                bp_total_val = None
                for r2 in ws_res.iter_rows(min_col=1, max_col=3):
                    if r2[0].value == "BP_total" and r2[2].value not in (None, "-"):
                        bp_total_val = r2[2].value
                        break
                # BP_ppm es calcula amb RF_BP, no amb RF_COLUMN
                # No patchejar aquí si no tenim RF_BP
                break

    try:
        wb.save(excel_path)
        result["success"] = True
        result["patched_fields"] = n_patched
    except Exception as e:
        result["errors"].append(f"Error guardant Excel: {e}")

    return result


# =============================================================================
# EXPORTS
# =============================================================================
__all__ = [
    "write_final_excel",
    "write_light_excel",
    "export_sequence",
    "generate_summary_excel",
    "patch_excel_calibration",
    "DEFAULT_EXPORT_CONFIG",
    "_find_and_load_bp_data",
]
