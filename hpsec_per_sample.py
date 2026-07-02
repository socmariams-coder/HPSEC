"""
HPSEC Suite — per-sample file export
=====================================

Genera dos fitxers autocontinguts per cada mostra analitzada, dissenyats per
treballar amb scripts externs (pandas, R, MATLAB, Excel) sense reprocessar
res ni dependre del JSON central de la seqüència.

Estructura per mostra (a `SEQ/RESULTATS/PER_SAMPLE/` — lliurament per a fora):
    {sample}.csv   ← cromatograma tabular (totes les rèpliques + DAD + final)
    {sample}.json  ← metadades estructurades (anomalies, àrees, selecció, ppm…)

Punts d'entrada:
    write_sample_files(name, sample_data, processed_data, output_dir, *, method)
        → genera CSV + JSON per una mostra (post-Analitzar).
    write_all_samples(processed_data, output_dir)
        → en lot per tota la seqüència.
    update_sample_quantification(name, quantification, output_dir, calibration)
        → actualitza el JSON existent amb ppm un cop quantificat.

El CSV utilitza el t_doc de la rèplica seleccionada com a temps de referència;
les altres rèpliques s'interpolen a aquest temps per quedar alineades. La
fidelitat dels arrays crus es preserva al JSON `replicas.{rk}.peak_info`.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

JSON_SCHEMA_VERSION = "1.0"


def _atomic_write_json(path, data, **dump_kwargs):
    """Escriu JSON atòmicament (temp + fsync + os.replace)."""
    import tempfile
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, **dump_kwargs)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

# ============================================================================
# Helpers
# ============================================================================


def _to_jsonable(value):
    """Converteix arrays numpy / objectes a tipus JSON-natius (recursiu)."""
    if value is None:
        return None
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        v = float(value)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    if isinstance(value, np.ndarray):
        return [_to_jsonable(x) for x in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return str(value)


def _get_dad_254_series(rep_data):
    """Extreu (t, y_254) d'una rèplica si té df_dad. Retorna (None, None) si no."""
    df_dad = rep_data.get("df_dad") if isinstance(rep_data, dict) else None
    if df_dad is None:
        return None, None
    try:
        if hasattr(df_dad, "columns"):
            cols = list(df_dad.columns)
        elif isinstance(df_dad, dict):
            cols = list(df_dad.keys())
        else:
            return None, None
        t_col = next((c for c in cols if "time" in str(c).lower()), None)
        wl_col = next((c for c in cols if "254" in str(c)), None)
        if t_col is None or wl_col is None:
            return None, None
        if hasattr(df_dad, "columns"):
            return (np.asarray(df_dad[t_col], dtype=float),
                    np.asarray(df_dad[wl_col], dtype=float))
        else:
            return (np.asarray(df_dad[t_col], dtype=float),
                    np.asarray(df_dad[wl_col], dtype=float))
    except Exception as e:
        logger.debug("_get_dad_254_series failed: %s", e)
        return None, None


def _interpolate_to(t_ref, t_src, y_src):
    """Interpola y_src(t_src) → y(t_ref). Fora del rang: NaN."""
    if t_src is None or y_src is None:
        return None
    t_src = np.asarray(t_src, dtype=float)
    y_src = np.asarray(y_src, dtype=float)
    if len(t_src) == 0 or len(y_src) == 0:
        return None
    y = np.interp(t_ref, t_src, y_src, left=np.nan, right=np.nan)
    return y


# ============================================================================
# Build chromatogram DataFrame (returned as plain dict-of-arrays)
# ============================================================================


def _build_chromatogram_table(sample_data):
    """Construeix el dict columnar per al CSV.

    Columnes:
        time         — t_doc de la rèplica seleccionada (o R1 si no n'hi ha)
        R1, R2, ...  — y_doc_net de cada rèplica interpolada a time
        Compost      — y composat (si timeout_composition aplicada a la sel.)
        DAD_254      — DAD A254 de la rèplica DAD seleccionada
        Selected_DOC — còpia de la rèplica/compost DOC final (per scripts)
    """
    replicas = sample_data.get("replicas") or {}
    selected = sample_data.get("selected") or {}
    sel_doc = selected.get("doc", "1")
    sel_dad = selected.get("dad", sel_doc)

    # Triar temps de referència: rèplica DOC seleccionada (o primera disponible)
    rep_keys = sorted(
        [k for k in replicas.keys() if k not in (None, "")],
        key=lambda x: (int(x) if str(x).isdigit() else 999))
    if not rep_keys:
        return None

    ref_key = sel_doc if sel_doc in replicas else rep_keys[0]
    ref_rep = replicas.get(ref_key) or {}
    t_ref = ref_rep.get("t_doc")
    if t_ref is None:
        # Fallback: primer t_doc disponible
        for k in rep_keys:
            t_alt = (replicas.get(k) or {}).get("t_doc")
            if t_alt is not None and len(t_alt) > 0:
                t_ref = t_alt
                ref_key = k
                break
    if t_ref is None or len(t_ref) == 0:
        return None

    t_ref = np.asarray(t_ref, dtype=float)
    table = {"time": t_ref.tolist()}

    # Una columna per rèplica
    for k in rep_keys:
        rd = replicas.get(k) or {}
        t_src = rd.get("t_doc")
        y_src = rd.get("y_doc_net")
        if t_src is None or y_src is None:
            continue
        if k == ref_key:
            y = np.asarray(y_src, dtype=float)
        else:
            y = _interpolate_to(t_ref, t_src, y_src)
        if y is None:
            continue
        table[f"R{k}"] = y.tolist()

    # Compost: si la rèplica seleccionada té timeout_composition aplicada,
    # y_doc_net ja és el compost; y_doc_net_pre_composition és l'original.
    sel_rep = replicas.get(sel_doc) or {}
    if sel_rep.get("timeout_composition"):
        # La 'final' és el compost; afegim columna explícita
        y_comp = np.asarray(sel_rep.get("y_doc_net"), dtype=float)
        table["Compost"] = y_comp.tolist()

    # DAD A254 interpolat a t_ref
    dad_rep = replicas.get(sel_dad) or {}
    t_dad, y_dad = _get_dad_254_series(dad_rep)
    if t_dad is not None and y_dad is not None:
        y_dad_interp = _interpolate_to(t_ref, t_dad, y_dad)
        if y_dad_interp is not None:
            table["DAD_254"] = y_dad_interp.tolist()

    # Selected_DOC: còpia explícita de la rèplica final agafada
    # (la seleccionada, post-reparació i post-composició)
    if sel_doc in replicas:
        sel_y = (replicas.get(sel_doc) or {}).get("y_doc_net")
        if sel_y is not None:
            if sel_doc == ref_key:
                table["Selected_DOC"] = list(sel_y)
            else:
                yi = _interpolate_to(
                    t_ref,
                    (replicas.get(sel_doc) or {}).get("t_doc"),
                    sel_y)
                if yi is not None:
                    table["Selected_DOC"] = yi.tolist()

    return table


def _write_csv(path, table):
    """Escriu un dict-of-arrays a CSV (sense pandas, stdlib only)."""
    import csv
    cols = list(table.keys())
    n = len(table[cols[0]])
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i in range(n):
            row = []
            for c in cols:
                v = table[c][i] if i < len(table[c]) else None
                if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                    row.append("")
                elif isinstance(v, float):
                    row.append(f"{v:.6g}")
                else:
                    row.append(str(v))
            w.writerow(row)


# ============================================================================
# Build metadata JSON
# ============================================================================


def _build_metadata(sample_name, sample_data, processed_data, method):
    """Construeix el dict de metadades estructurades."""
    from hpsec_version import SUITE_VERSION

    selected = sample_data.get("selected") or {}
    recommendation = sample_data.get("recommendation") or {}
    replicas = sample_data.get("replicas") or {}
    quantification = sample_data.get("quantification") or {}

    # ─── sample ───
    siblings = sorted({
        (rd or {}).get("_source_path", "")
        for rd in replicas.values()
        if isinstance(rd, dict) and (rd or {}).get("_source_path")
    })
    sample_block = {
        "name": sample_name,
        "seq_name": processed_data.get("seq_name", ""),
        "seq_date": processed_data.get("seq_date") or processed_data.get("timestamp", ""),
        "method": method,
        "type": sample_data.get("sample_type", "SAMPLE"),
        "siblings": [os.path.basename(p) for p in siblings if p],
    }

    # ─── replicas ───
    rep_meta = {}
    for rk in sorted(replicas.keys(),
                     key=lambda x: (int(x) if str(x).isdigit() else 999)):
        rd = replicas.get(rk) or {}
        rep_meta[str(rk)] = {
            "source_seq": os.path.basename(rd.get("_source_path", "")),
            "source_label": rd.get("_source_label", ""),
            "injection_index": rd.get("injection_index"),
            "inj_volume_uL": rd.get("inj_volume"),
            "peak_info": rd.get("peak_info") or {},
            "areas": rd.get("areas") or {},
            "areas_uib": rd.get("areas_uib") or {},
            "snr_doc": (rd.get("snr_info") or {}).get("snr_direct"),
            "snr_dad": (rd.get("snr_info_dad") or {}),
            "bigaussian_doc": rd.get("bigaussian_doc"),
            "bigaussian_254": rd.get("bigaussian_254"),
            "fwhm_doc": rd.get("fwhm_doc"),
            "symmetry_doc": rd.get("symmetry_doc"),
            "anomalies": rd.get("anomalies") or [],
            "timeout_info": rd.get("timeout_info") or {},
            "uib_saturated": rd.get("uib_saturated", False),
        }

    # ─── selection ───
    doc_rec = (recommendation.get("doc") or {}).get("replica")
    dad_rec = (recommendation.get("dad") or {}).get("replica")
    user_modified = (
        (selected.get("doc") and str(selected.get("doc")) != str(doc_rec))
        or (selected.get("dad") and str(selected.get("dad")) != str(dad_rec))
    )
    selection_block = {
        "doc": selected.get("doc"),
        "dad": selected.get("dad"),
        "doc_recommended": doc_rec,
        "dad_recommended": dad_rec,
        "reason": (recommendation.get("doc") or {}).get("reason", ""),
        "user_modified": bool(user_modified),
    }

    # ─── repair / composition ───
    repair_block = {
        "applied": bool(sample_data.get("repaired")),
        "history": sample_data.get("repair_history") or [],
    }

    sel_rep_data = replicas.get(selected.get("doc")) or {}
    composition_block = {
        "applied": bool(sel_rep_data.get("timeout_composition")),
        "details": sel_rep_data.get("timeout_composition") or {},
    }

    # ─── results (areas + ppm si quantificat) ───
    sel_rep_areas = (sel_rep_data.get("areas") or {}).get("DOC") or {}
    # Fraccions: cada clau de areas["DOC"] que no sigui 'total'.
    # Inclou tant principals (BioP/HS/BB/SB/LMW) com sub-zones (HS-1..4, BB-1..2)
    # que calcular_fraccions_temps ja deixa al dict 'areas'.
    fractions = {k: v for k, v in sel_rep_areas.items()
                 if k not in ("total",) and isinstance(v, (int, float))}
    # Percentatges per fracció respecte total
    total_doc = sel_rep_areas.get("total") or 0
    fractions_pct = {
        f"{k}_pct": (100.0 * v / total_doc if total_doc > 0 else 0.0)
        for k, v in fractions.items()
    } if total_doc > 0 else {}

    results_block = {
        "areas_per_fraction": fractions,
        "fractions_pct": fractions_pct,
        "area_total": sel_rep_areas.get("total"),
        "concentration_ppm": quantification.get("concentration_ppm_direct"),
        "concentration_ppm_uib": quantification.get("concentration_ppm_uib"),
        "hci": quantification.get("hci"),
        "hci_character": quantification.get("hci_character"),
        # v2.2.0+: estadística entre rèpliques vàlides
        "per_replica": quantification.get("per_replica") or {},
        "statistics": quantification.get("statistics") or {},
        "selected_replica": quantification.get("selected_replica"),
    }

    # ─── calibration (només si quantificat) ───
    quant_pending = processed_data.get("quantification_pending", True)
    cal_block = None
    if not quant_pending and quantification:
        cal_block = {
            "fingerprint": processed_data.get("calibration_fingerprint", ""),
            "rf_mass_cal_direct": quantification.get("rf_mass_cal_used"),
            "rf_mass_cal_uib": quantification.get("rf_mass_cal_uib_used"),
            "intercept_direct": quantification.get("intercept"),
            "intercept_uib": quantification.get("intercept_uib"),
            "volume_uL": quantification.get("volume_uL"),
            "calibration_source": quantification.get("calibration_source"),
        }

    metadata = {
        "$schema_version": JSON_SCHEMA_VERSION,
        "suite_version": SUITE_VERSION,
        "timestamp": datetime.now().isoformat(),
        "quantified": not quant_pending,
        "sample": sample_block,
        "replicas": rep_meta,
        "selection": selection_block,
        "repair": repair_block,
        "composition": composition_block,
        "results": results_block,
        "calibration": cal_block,
        "fingerprints": {
            "config": processed_data.get("config_fingerprint", ""),
            "calibration": processed_data.get("calibration_fingerprint", ""),
        },
        "csv_companion": f"{sample_name}.csv",
    }
    return _to_jsonable(metadata)


# ============================================================================
# Public API
# ============================================================================


def write_sample_files(sample_name, sample_data, processed_data, output_dir,
                        method=None):
    """Escriu {sample_name}.csv + {sample_name}.json a output_dir.

    Args:
        sample_name: nom mostra (clau a samples_grouped).
        sample_data: dict de la mostra (samples_grouped[sample_name]).
        processed_data: result complet (per llegir method, fingerprints…).
        output_dir: carpeta destí (es crea si no existeix).
        method: 'COLUMN' | 'BP'. Si None, es llegeix de processed_data.

    Skip mostres de tipus KHP (no és resultat exportable) i BLANK/CONTROL
    (anàlisi light: poca info útil; es manté el JSON però amb camps simples).
    """
    if not sample_name or not isinstance(sample_data, dict):
        return None
    sample_type = sample_data.get("sample_type", "SAMPLE")
    if sample_type == "KHP":
        return None

    if method is None:
        method = processed_data.get("method", "COLUMN")

    os.makedirs(output_dir, exist_ok=True)

    # Nom sanitizat (evitar / \ : * ? " < > |)
    safe = "".join(c if c.isalnum() or c in ("_", "-", ".") else "_"
                   for c in str(sample_name))
    csv_path = os.path.join(output_dir, f"{safe}.csv")
    json_path = os.path.join(output_dir, f"{safe}.json")

    # CSV (només si hi ha cromatograma)
    if sample_type in ("SAMPLE", "PR"):
        table = _build_chromatogram_table(sample_data)
        if table is not None:
            try:
                _write_csv(csv_path, table)
            except Exception as e:
                logger.warning("Error writing CSV for %s: %s", sample_name, e)

    # JSON (sempre)
    try:
        metadata = _build_metadata(sample_name, sample_data, processed_data, method)
        _atomic_write_json(json_path, metadata, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning("Error writing JSON for %s: %s", sample_name, e)

    return json_path


def write_all_samples(processed_data, output_dir=None):
    """Genera fitxers per mostra per a tota la seqüència.

    Args:
        processed_data: result d'analyze_sequence (o load_analysis_result).
        output_dir: carpeta destí. Si None, usa {seq_path}/PER_SAMPLE/.

    Returns:
        dict {sample_name: json_path}.
    """
    if not processed_data:
        return {}

    seq_path = processed_data.get("seq_path", "")
    if output_dir is None:
        if not seq_path:
            logger.warning("write_all_samples: cap seq_path; sense output_dir")
            return {}
        output_dir = os.path.join(seq_path, "RESULTATS", "PER_SAMPLE")

    samples_grouped = processed_data.get("samples_grouped") or {}
    method = processed_data.get("method", "COLUMN")

    written = {}
    for name, sd in samples_grouped.items():
        path = write_sample_files(name, sd, processed_data, output_dir, method)
        if path:
            written[name] = path
    return written


def update_sample_quantification(sample_name, quantification, output_dir,
                                  calibration_fingerprint=None,
                                  config_fingerprint=None):
    """Actualitza el JSON existent d'una mostra amb la quantificació aplicada.

    No toca el CSV (les àrees no canvien amb la recta, només el ppm).
    """
    if not sample_name or not quantification:
        return None
    safe = "".join(c if c.isalnum() or c in ("_", "-", ".") else "_"
                   for c in str(sample_name))
    json_path = os.path.join(output_dir, f"{safe}.json")
    if not os.path.exists(json_path):
        return None

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.warning("Error reading JSON for %s: %s", sample_name, e)
        return None

    data.setdefault("results", {})
    data["results"]["concentration_ppm"] = _to_jsonable(
        quantification.get("concentration_ppm_direct"))
    data["results"]["concentration_ppm_uib"] = _to_jsonable(
        quantification.get("concentration_ppm_uib"))
    data["results"]["hci"] = _to_jsonable(quantification.get("hci"))
    data["results"]["hci_character"] = quantification.get("hci_character")
    # v2.2.0+: estadística entre rèpliques
    data["results"]["per_replica"] = _to_jsonable(
        quantification.get("per_replica") or {})
    data["results"]["statistics"] = _to_jsonable(
        quantification.get("statistics") or {})
    data["results"]["selected_replica"] = quantification.get("selected_replica")

    data["calibration"] = _to_jsonable({
        "fingerprint": calibration_fingerprint or "",
        "rf_mass_cal_direct": quantification.get("rf_mass_cal_used"),
        "rf_mass_cal_uib": quantification.get("rf_mass_cal_uib_used"),
        "intercept_direct": quantification.get("intercept"),
        "intercept_uib": quantification.get("intercept_uib"),
        "volume_uL": quantification.get("volume_uL"),
        "calibration_source": quantification.get("calibration_source"),
    })

    fps = data.setdefault("fingerprints", {})
    if calibration_fingerprint:
        fps["calibration"] = calibration_fingerprint
    if config_fingerprint:
        fps["config"] = config_fingerprint

    data["quantified"] = True
    data["timestamp"] = datetime.now().isoformat()

    try:
        _atomic_write_json(json_path, data, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning("Error writing JSON for %s: %s", sample_name, e)
        return None
    return json_path


def update_all_quantifications(processed_data, output_dir=None):
    """Actualitza tots els JSON per mostra amb les quantificacions actuals."""
    if not processed_data:
        return 0
    seq_path = processed_data.get("seq_path", "")
    if output_dir is None:
        if not seq_path:
            return 0
        output_dir = os.path.join(seq_path, "RESULTATS", "PER_SAMPLE")
    if not os.path.isdir(output_dir):
        return 0

    cal_fp = processed_data.get("calibration_fingerprint", "")
    cfg_fp = processed_data.get("config_fingerprint", "")
    samples_grouped = processed_data.get("samples_grouped") or {}
    n = 0
    for name, sd in samples_grouped.items():
        q = sd.get("quantification")
        if q:
            res = update_sample_quantification(name, q, output_dir, cal_fp, cfg_fp)
            if res:
                n += 1
    return n


__all__ = [
    "write_sample_files",
    "write_all_samples",
    "update_sample_quantification",
    "update_all_quantifications",
    "JSON_SCHEMA_VERSION",
]
