"""
hpsec_export.py - Mòdul d'exportació HPSEC (Fase 5: EXPORTAR)
==============================================================

FASE 5 del pipeline de 5 fases:
- Generar Excels finals amb estructura estandarditzada
- Fulls: ID (traçabilitat), DOC (cromatogrames), DAD (6 WL), RESULTS (integracions)
- Usar seleccions de rèpliques de la Fase 4 (Revisar)

REQUEREIX:
- Fase 4: review_data amb seleccions DOC/DAD per mostra

Usat per HPSEC_Suite.py
"""

__version__ = "1.0.0"
__version_date__ = "2026-02-03"

import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.integrate import trapezoid

# Configuració per defecte
DEFAULT_EXPORT_CONFIG = {
    "target_wavelengths": [220, 252, 254, 272, 290, 362],
    "time_fractions": {
        "BioP": [0, 18],
        "HS": [18, 23],
        "BB": [23, 30],
        "SB": [30, 40],
        "LMW": [40, 70],
    },
}


def write_final_excel(
    out_path: str,
    sample_name: str,
    sample_data: dict,
    calibration_data: dict = None,
    mode: str = "COLUMN",
    config: dict = None,
):
    """
    Escriu Excel final amb estructura estandarditzada.

    Fulls:
        ID: Traçabilitat completa (fitxers, shifts, quantificació)
        DOC: Cromatogrames DOC (final, raw Direct, raw UIB)
        DAD: 6 longituds d'ona seleccionades
        RESULTS: Integracions per fraccions

    Args:
        out_path: Camí del fitxer Excel
        sample_name: Nom de la mostra
        sample_data: Dict amb les dades de la rèplica seleccionada
        calibration_data: Dict amb dades de calibració (opcional)
        mode: "BP" o "COLUMN"
        config: Configuració (opcional)

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

    # === FULL DOC: Cromatogrames ===
    df_doc = _build_doc_sheet(t_doc, y_doc_net, y_doc_raw, y_doc_uib_net, y_doc_uib_raw, is_dual)

    # === FULL DAD: 6 longituds d'ona ===
    df_dad_export = _build_dad_sheet(df_dad, config)

    # === FULL RESULTS: Integracions ===
    df_results = _build_results_sheet(t_doc, y_doc_net, df_dad, mode, config)

    # Escriure Excel
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_id.to_excel(writer, sheet_name="ID", index=False)
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

    # Quantificació (si hi ha calibració)
    quantification = sample_data.get("quantification", {})
    if quantification:
        conc = quantification.get("concentration_ppm")
        if conc is not None:
            rows.append(("Concentration_ppm", round(conc, 3)))
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


def _build_results_sheet(t_doc, y_doc_net, df_dad, mode, config):
    """Construeix el DataFrame del full RESULTS amb integracions."""
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
        # Per BP, només total
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
# EXPORTACIÓ DE SEQÜÈNCIES
# =============================================================================

def export_sequence(
    samples_grouped: dict,
    output_dir: str,
    calibration_data: dict = None,
    mode: str = "COLUMN",
    config: dict = None,
    progress_callback=None,
):
    """
    Exporta totes les mostres d'una seqüència.

    Args:
        samples_grouped: Dict amb mostres agrupades (de ReviewPanel)
        output_dir: Directori de sortida
        calibration_data: Dades de calibració
        mode: "BP" o "COLUMN"
        config: Configuració
        progress_callback: Funció per reportar progrés (pct, msg)

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
    }

    total = len(samples_grouped)
    for i, (sample_name, sample_info) in enumerate(samples_grouped.items()):
        if progress_callback:
            pct = int((i / total) * 100)
            progress_callback(pct, f"Exportant {sample_name}...")

        try:
            # Obtenir rèplica seleccionada per DOC
            selected = sample_info.get("selected", {})
            doc_replica = selected.get("doc", "1")
            dad_replica = selected.get("dad", doc_replica)

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
            )

            results["files"].append({
                "sample": sample_name,
                "path": str(filepath),
                "doc_replica": doc_replica,
                "dad_replica": dad_replica,
            })
            results["n_exported"] += 1

        except Exception as e:
            results["errors"].append(f"{sample_name}: {str(e)}")
            results["n_errors"] += 1

    if progress_callback:
        progress_callback(100, "Exportació completada")

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
        SUMMARY: Una fila per mostra amb concentració, SNR, warnings
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
        selected = sample_info.get("selected", {})
        quantification = sample_info.get("quantification", {})
        comparison = sample_info.get("comparison", {})

        doc_replica = selected.get("doc", "1")
        dad_replica = selected.get("dad", "1")

        doc_data = sample_info.get("replicas", {}).get(doc_replica, {})
        snr_info = doc_data.get("snr_info", {})

        # Warnings
        doc_warnings = comparison.get("doc", {}).get("warnings", []) if comparison else []
        dad_warnings = comparison.get("dad", {}).get("warnings", []) if comparison else []
        all_warnings = doc_warnings + dad_warnings

        row = {
            "Sample": sample_name,
            "DOC_Replica": f"R{doc_replica}",
            "DAD_Replica": f"R{dad_replica}",
            "Conc_ppm": quantification.get("concentration_ppm"),
            "Area_total": quantification.get("area_total"),
            "SNR_Direct": snr_info.get("snr_direct"),
            "SNR_UIB": snr_info.get("snr_uib"),
            "Warnings": "; ".join(all_warnings) if all_warnings else "",
        }

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
# EXPORTS
# =============================================================================
__all__ = [
    "write_final_excel",
    "export_sequence",
    "generate_summary_excel",
    "DEFAULT_EXPORT_CONFIG",
]
