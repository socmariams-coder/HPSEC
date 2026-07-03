# -*- coding: utf-8 -*-
"""
HPSEC Suite — FAIR open-data export (Frictionless Data Package)
================================================================

Generates a self-describing, machine-readable open dataset from a processed
sequence, following the Frictionless "Tabular Data Package" spec:

    RESULTATS/
    ├── results_SEC.csv            one row per sample: identity + ppm + DOC/DAD fractions
    │                              + processing provenance (injection, smoothing, baseline,
    │                              delay, calibration, volume, anomalies)
    ├── traces/{sample}_{date}_SEC.csv
    │                              time-resolved chromatogram. COLUMN: time + DOC + DAD
    │                              wavelengths. BP: DAD spectrum at the peak maximum.
    ├── datapackage.json           Frictionless descriptor: resources + field schema
    │                              (name/type/unit/description) + provenance + metadata
    └── README.txt                 human-readable overview (English)

All identifiers, column names, descriptions and documentation are in ENGLISH, as
required for internationally reusable open data. Encoding UTF-8, decimal point
'.', dates ISO 8601, CSV delimiter ','.
"""
import os
import csv
import json
import tempfile
from datetime import datetime, timezone

import numpy as np

from hpsec_version import SUITE_VERSION

TECHNIQUE = "HPSEC"                # High-Performance Size-Exclusion Chromatography
FRACTIONS = ["BioP", "HS", "BB", "SB", "LMW"]   # LC-OCD-style DOC fractions
FRACTION_LABEL = {
    "BioP": "Biopolymers", "HS": "Humic substances", "BB": "Building blocks",
    "SB": "Low-MW acids", "LMW": "Low-MW neutrals",
}
CSV_SEP = ","


# --------------------------------------------------------------------------- #
def _atomic_write(path, text):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _num(v, nd=None):
    """Return a JSON/CSV-safe number or empty string (never text-in-number)."""
    if v is None:
        return ""
    try:
        f = float(v)
        if np.isnan(f) or np.isinf(f):
            return ""
        return round(f, nd) if nd is not None else f
    except (TypeError, ValueError):
        return ""


def _sel_doc_replica(sample):
    sel = sample.get("selected") or {}
    return (sample.get("replicas") or {}).get(sel.get("doc", "1"), {}) or {}


def _dad_columns(df_dad):
    """Return (columns_list, is_dataframe) for a df_dad that may be a DataFrame or dict."""
    if df_dad is None:
        return None, False
    if hasattr(df_dad, "columns"):
        return list(df_dad.columns), True
    if isinstance(df_dad, dict):
        return list(df_dad.keys()), False
    return None, False


def _col_series(df_dad, col, is_df):
    return np.asarray(df_dad[col] if is_df else df_dad[col], dtype=float)


def _find_time_col(cols):
    return next((c for c in cols if "time" in str(c).lower()), None)


def _find_wl_col(cols, wl):
    for c in cols:
        cs = str(c)
        if cs == str(wl) or cs == f"A{wl}" or cs == f"{wl}.0":
            return c
    for c in cols:
        try:
            if abs(float(str(c)) - wl) < 1.0:
                return c
        except (ValueError, TypeError):
            pass
    return None


# --------------------------------------------------------------------------- #
# Provenance (per-sample processing recipe)
# --------------------------------------------------------------------------- #
def _provenance(sample, config, net_delay_min):
    rep = _sel_doc_replica(sample)
    sel = sample.get("selected") or {}
    q = sample.get("quantification") or {}
    chrom = {}
    try:
        chrom = config.get_section("chromatogram") if hasattr(config, "get_section") else {}
    except Exception:
        chrom = {}
    anomalies = rep.get("anomalies") or []
    anom_codes = ";".join(
        (a.get("code", "") if isinstance(a, dict) else str(a)) for a in anomalies)
    return {
        "injection_index": rep.get("injection_index"),
        "doc_replica": sel.get("doc"),
        "dad_replica": sel.get("dad"),
        "smoothing_method": "Savitzky-Golay",
        "smoothing_window": chrom.get("smoothing_window"),
        "smoothing_order": chrom.get("smoothing_order"),
        "baseline_method": "statistical-mode",
        "net_delay_min": _num(net_delay_min, 3),
        "rf_mass_cal_doc": q.get("rf_mass_cal_used"),
        "intercept_doc": q.get("intercept"),
        "volume_uL": q.get("volume_uL") or rep.get("inj_volume"),
        "calibration_source": q.get("calibration_source"),
        "repaired": bool(sample.get("repaired")),
        "anomalies": anom_codes,
    }


# --------------------------------------------------------------------------- #
# results_SEC.csv  (one row per sample)
# --------------------------------------------------------------------------- #
def _results_columns(target_wls):
    """Ordered list of (name, type, unit, description) for the results resource."""
    cols = [
        ("sequence", "string", "", "Sequence (run) identifier"),
        ("analysis_date", "date", "", "Date of instrumental analysis (ISO 8601)"),
        ("technique", "string", "", "Analytical technique"),
        ("mode", "string", "", "Chromatographic mode: COLUMN (size separation) or BP (bypass)"),
        ("sample", "string", "", "Sample name"),
        ("sample_type", "string", "", "SAMPLE, BLANK, CONTROL or KHP (standard)"),
        ("valid", "boolean", "", "Whether the selected replica passed quality checks"),
        ("conc_ppm_doc", "number", "mg C/L", "DOC concentration from the direct TOC signal"),
        ("conc_ppm_uib", "number", "mg C/L", "DOC concentration from the UV-persulfate (UIB) signal"),
        ("hci", "number", "", "Humic Character Index"),
        ("hci_character", "string", "", "Qualitative humic character label"),
        ("doc_area_total", "number", "a.u.*min", "Total integrated DOC peak area"),
    ]
    for fr in FRACTIONS:
        cols.append((f"doc_{fr.lower()}_area", "number", "a.u.*min",
                     f"DOC area of the {FRACTION_LABEL[fr]} fraction"))
        cols.append((f"doc_{fr.lower()}_pct", "number", "%",
                     f"DOC {FRACTION_LABEL[fr]} as percent of total DOC area"))
    for fr in FRACTIONS:
        cols.append((f"a254_{fr.lower()}_area", "number", "mAU*min",
                     f"UV254 area of the {FRACTION_LABEL[fr]} fraction"))
    for wl in target_wls:
        cols.append((f"a{wl}_total", "number", "mAU*min",
                     f"Total integrated UV{wl} nm peak area"))
    # provenance
    prov = [
        ("injection_index", "integer", "", "Index of the injection used"),
        ("doc_replica", "string", "", "Selected DOC replica"),
        ("dad_replica", "string", "", "Selected DAD (UV) replica"),
        ("smoothing_method", "string", "", "Signal smoothing method"),
        ("smoothing_window", "integer", "points", "Savitzky-Golay window length"),
        ("smoothing_order", "integer", "", "Savitzky-Golay polynomial order"),
        ("baseline_method", "string", "", "Baseline estimation method"),
        ("net_delay_min", "number", "min", "Applied HPLC-to-TOC time offset"),
        ("rf_mass_cal_doc", "number", "a.u.*min/ug", "Response factor used for DOC quantification"),
        ("intercept_doc", "number", "a.u.*min", "Calibration intercept used for DOC"),
        ("volume_uL", "number", "uL", "Injection volume used for quantification"),
        ("calibration_source", "string", "", "Origin of the calibration applied"),
        ("repaired", "boolean", "", "Whether the peak apex was repaired"),
        ("anomalies", "string", "", "Semicolon-separated quality anomaly codes"),
    ]
    return cols + prov


def _results_row(sample_name, sample, seq_name, seq_date, mode, target_wls,
                 config, net_delay_min):
    is_light = sample.get("analysis_type") == "light"
    rep = _sel_doc_replica(sample)
    q = sample.get("quantification") or {}
    sel = sample.get("selected") or {}
    is_invalid = (sel.get("doc") == "none"
                  or sample.get("sample_valid") is False
                  or sample.get("skip_quantification", False))
    areas = rep.get("areas") or {}
    doc = areas.get("DOC") or {}
    a254 = areas.get("A254") or {}
    stype = (sample.get("sample_type")
             or rep.get("sample_type", "BLANK" if is_light else "SAMPLE"))

    row = {
        "sequence": seq_name, "analysis_date": seq_date, "technique": TECHNIQUE,
        "mode": mode, "sample": sample_name, "sample_type": stype,
        "valid": ("" if is_light else (not is_invalid)),
        "conc_ppm_doc": _num(q.get("concentration_ppm_direct")
                             or q.get("concentration_ppm"), 4),
        "conc_ppm_uib": _num(q.get("concentration_ppm_uib"), 4),
        "hci": _num(q.get("hci"), 4),
        "hci_character": q.get("hci_character", ""),
        "doc_area_total": _num(doc.get("total"), 3),
    }
    total_doc = doc.get("total") or 0
    for fr in FRACTIONS:
        row[f"doc_{fr.lower()}_area"] = _num(doc.get(fr), 3)
        pct = doc.get(f"{fr}_pct")
        if pct is None and total_doc and doc.get(fr) is not None:
            pct = 100.0 * doc.get(fr) / total_doc
        row[f"doc_{fr.lower()}_pct"] = _num(pct, 2)
    for fr in FRACTIONS:
        row[f"a254_{fr.lower()}_area"] = _num(a254.get(fr), 3)
    for wl in target_wls:
        row[f"a{wl}_total"] = _num((areas.get(f"A{wl}") or {}).get("total"), 3)
    row.update(_provenance(sample, config, net_delay_min))
    return row


# --------------------------------------------------------------------------- #
# Per-sample trace CSV
# --------------------------------------------------------------------------- #
def _write_trace(path, sample, mode, target_wls):
    """COLUMN: time + DOC + one column per DAD wavelength (interpolated to DOC time).
    BP: DAD spectrum at the peak maximum (wavelength_nm, absorbance)."""
    rep = _sel_doc_replica(sample)
    sel = sample.get("selected") or {}
    dad_rep = (sample.get("replicas") or {}).get(sel.get("dad", sel.get("doc", "1")), {}) or {}
    df_dad = dad_rep.get("df_dad")
    cols, is_df = _dad_columns(df_dad)

    if mode == "BP":
        # DAD spectrum at the peak apex
        rows = [["wavelength_nm", "absorbance_mAU"]]
        if cols:
            tcol = _find_time_col(cols)
            pk = rep.get("peak_info") or {}
            tmax = pk.get("t_max")
            if tcol is not None:
                t = _col_series(df_dad, tcol, is_df)
                imax = int(np.argmin(np.abs(t - tmax))) if tmax is not None else int(len(t) // 2)
                for wl in target_wls:
                    wc = _find_wl_col(cols, wl)
                    if wc is not None:
                        y = _col_series(df_dad, wc, is_df)
                        rows.append([wl, _num(y[imax], 4)])
        _write_rows(path, rows)
        return path

    # COLUMN: time + DOC + wavelengths
    t_doc = rep.get("t_doc")
    y_doc = rep.get("y_doc_net")
    if t_doc is None or y_doc is None:
        return None
    t_doc = np.asarray(t_doc, dtype=float)
    header = ["time_min", "doc"]
    data = [t_doc, np.asarray(y_doc, dtype=float)]
    if cols:
        tcol = _find_time_col(cols)
        t_dad = _col_series(df_dad, tcol, is_df) if tcol is not None else None
        for wl in target_wls:
            wc = _find_wl_col(cols, wl)
            if wc is not None and t_dad is not None:
                y = _col_series(df_dad, wc, is_df)
                yi = np.interp(t_doc, t_dad, y, left=np.nan, right=np.nan)
                header.append(f"a{wl}")
                data.append(yi)
    rows = [header]
    for i in range(len(t_doc)):
        rows.append([_num(col[i], 5) for col in data])
    _write_rows(path, rows)
    return path


def _write_rows(path, rows):
    import io
    buf = io.StringIO()
    w = csv.writer(buf, delimiter=CSV_SEP, lineterminator="\n")
    w.writerows(rows)
    _atomic_write(path, buf.getvalue())


# --------------------------------------------------------------------------- #
# datapackage.json + README
# --------------------------------------------------------------------------- #
def _iso_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _slug(s):
    return "".join(c.lower() if c.isalnum() else "-" for c in str(s)).strip("-")


def _build_datapackage(seq_name, seq_date, mode, target_wls, results_cols,
                       trace_resources, calibration_data, config):
    def fields(colspec):
        out = []
        for name, typ, unit, desc in colspec:
            f = {"name": name, "type": typ, "description": desc}
            if unit:
                f["unit"] = unit
            out.append(f)
        return out

    trace_field_spec = [("time_min", "number", "min", "Elution time")]
    trace_field_spec.append(("doc", "number", "a.u.", "Baseline-corrected DOC signal"))
    for wl in target_wls:
        trace_field_spec.append((f"a{wl}", "number", "mAU", f"UV absorbance at {wl} nm"))
    bp_field_spec = [
        ("wavelength_nm", "integer", "nm", "UV wavelength"),
        ("absorbance_mAU", "number", "mAU", "UV absorbance at the peak maximum"),
    ]

    resources = [{
        "name": "results",
        "path": "results_SEC.csv",
        "format": "csv", "mediatype": "text/csv", "encoding": "utf-8",
        "dialect": {"delimiter": CSV_SEP},
        "schema": {"fields": fields(results_cols),
                   "primaryKey": ["sample"]},
        "description": ("One row per sample: identity, DOC/UIB concentrations, DOC and "
                        "UV fraction areas, and full processing provenance."),
    }]
    for res in trace_resources:
        resources.append({
            "name": res["name"], "path": res["path"],
            "format": "csv", "mediatype": "text/csv", "encoding": "utf-8",
            "dialect": {"delimiter": CSV_SEP},
            "schema": {"fields": fields(bp_field_spec if mode == "BP" else trace_field_spec)},
            "description": res["description"],
        })

    cal = {}
    if calibration_data:
        for k in ("rf_mass_cal", "intercept", "r2", "n_points", "calibration_date"):
            if k in calibration_data:
                cal[k] = calibration_data[k]

    return {
        "name": _slug(f"hpsec-sec-{seq_name}-{seq_date}"),
        "title": f"HPSEC-SEC dataset — {seq_name}",
        "description": (
            "Open dataset of a High-Performance Size-Exclusion Chromatography (HPSEC) "
            "run with DOC (TOC) and DAD (UV) detection. Contains per-sample DOC/UV "
            "concentrations and fraction areas, time-resolved chromatograms, and the "
            "full processing provenance for each sample."),
        "created": _iso_now(),
        "keywords": ["HPSEC", "SEC", "DOC", "TOC", "DAD", "UV", "chromatography",
                     "dissolved organic carbon", "water"],
        "licenses": [{"name": "CC-BY-4.0",
                      "title": "Creative Commons Attribution 4.0",
                      "path": "https://creativecommons.org/licenses/by/4.0/"}],
        "hpsec": {
            "suite_version": SUITE_VERSION,
            "sequence": seq_name,
            "analysis_date": seq_date,
            "mode": mode,
            "calibration": cal,
        },
        "resources": resources,
    }


def _readme_text(seq_name, seq_date, mode, n_samples):
    return f"""HPSEC-SEC OPEN DATASET
======================

Sequence : {seq_name}
Date     : {seq_date}
Mode     : {mode}
Samples  : {n_samples}
Generated: {_iso_now()} by HPSEC Suite v{SUITE_VERSION}

WHAT THIS IS
------------
An open, machine-readable dataset from a High-Performance Size-Exclusion
Chromatography (HPSEC) analysis with DOC (TOC) and DAD (UV) detection.
It follows the Frictionless "Tabular Data Package" specification: every table
is a plain CSV and 'datapackage.json' describes each column (name, type, unit,
description) plus the processing provenance and calibration used.

FILES
-----
  results_SEC.csv        One row per sample. Sample identity, DOC/UIB
                         concentrations (mg C/L), DOC and UV fraction areas,
                         and the processing recipe (injection, smoothing,
                         baseline, delay, calibration, volume, anomalies).
  traces/                One CSV per sample with the time-resolved chromatogram.
                         COLUMN mode: columns are time_min, doc and one column
                         per UV wavelength (a220 ... a362).
                         BP mode: the UV spectrum at the peak maximum
                         (wavelength_nm, absorbance_mAU).
  datapackage.json       Frictionless descriptor: field schema, units, provenance
                         and dataset metadata (license, version, calibration).
  README.txt             This file.

CONVENTIONS
-----------
  Encoding UTF-8 . Decimal separator '.' . CSV delimiter ',' . Dates ISO 8601.
  Empty cell = not available / not applicable (never a text placeholder).

HOW TO LOAD
-----------
  Python : import pandas as pd; pd.read_csv("results_SEC.csv")
           # or, with the descriptor:
           from frictionless import Package; Package("datapackage.json")
  R      : read.csv("results_SEC.csv")

LICENSE
-------
  CC-BY-4.0 (https://creativecommons.org/licenses/by/4.0/)
"""


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def generate_data_package(samples_grouped, output_dir, mode="COLUMN",
                          calibration_data=None, config=None, seq_name="",
                          seq_date="", net_delay_min=None):
    """Write the Frictionless Data Package (results + traces + datapackage.json + README).

    Returns a dict with the paths written and counts.
    """
    from hpsec_config import get_config
    config = config or get_config()
    try:
        target_wls = config.get("chromatogram", "target_wavelengths")
    except Exception:
        target_wls = None
    if not target_wls:
        target_wls = [220, 252, 254, 272, 290, 362]

    os.makedirs(output_dir, exist_ok=True)
    traces_dir = os.path.join(output_dir, "traces")

    # normalise date for filenames (YYYYMMDD)
    date_tag = "".join(ch for ch in str(seq_date)[:10] if ch.isdigit()) or "NA"

    # --- results_SEC.csv ---
    results_cols = _results_columns(target_wls)
    col_names = [c[0] for c in results_cols]
    rows = [col_names]
    trace_resources = []
    n = 0
    for name in sorted(samples_grouped.keys()):
        sample = samples_grouped[name]
        r = _results_row(name, sample, seq_name, seq_date, mode, target_wls,
                         config, net_delay_min)
        rows.append([r.get(c, "") for c in col_names])
        n += 1
        # trace (skip KHP standards from the open dataset traces? keep all real+light)
        fname = f"{name}_{date_tag}_SEC.csv"
        tpath = os.path.join(traces_dir, fname)
        written = _write_trace(tpath, sample, mode, target_wls)
        if written:
            trace_resources.append({
                "name": _slug(f"trace-{name}"),
                "path": f"traces/{fname}",
                "description": f"Time-resolved chromatogram for sample {name}.",
            })
    _write_rows(os.path.join(output_dir, "results_SEC.csv"), rows)

    # --- datapackage.json ---
    dp = _build_datapackage(seq_name, seq_date, mode, target_wls, results_cols,
                            trace_resources, calibration_data, config)
    _atomic_write(os.path.join(output_dir, "datapackage.json"),
                  json.dumps(dp, indent=2, ensure_ascii=False))

    # --- README.txt ---
    _atomic_write(os.path.join(output_dir, "README.txt"),
                  _readme_text(seq_name, seq_date, mode, n))

    return {
        "success": True,
        "results": os.path.join(output_dir, "results_SEC.csv"),
        "datapackage": os.path.join(output_dir, "datapackage.json"),
        "readme": os.path.join(output_dir, "README.txt"),
        "n_samples": n,
        "n_traces": len(trace_resources),
    }
