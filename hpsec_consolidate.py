"""
hpsec_consolidate.py - Mòdul de consolidació de dades HPSEC
===========================================================

NOTA: Aquest mòdul està en procés de refactorització.
Les funcions de lectura de dades s'han mogut a hpsec_import.py (Fase 1).
Les funcions de processament s'han mogut a hpsec_process.py (Fase 3).
Aquest fitxer manté compatibilitat amb codi existent.

Conté tota la lògica per:
- Llegir fitxers DOC (UIB CSV o Excel mestre) → hpsec_import.py (MIGRAT)
- Llegir fitxers DAD (Export3D) → hpsec_import.py (MIGRAT)
- Processar i consolidar dades → hpsec_process.py (MIGRAT)
- Generar Excels consolidats → hpsec_export.py (pendent)

Usat per HPSEC_Suite.py i batch_process.py
"""

__version__ = "1.11.0"
__version_date__ = "2026-01-29"
# v1.11.0: Refactorització - funcions alineació mogudes a hpsec_calibrate.py
# v1.10.0: Refactorització - funcions processament mogudes a hpsec_process.py
# v1.9.0: Refactorització - funcions lectura mogudes a hpsec_import.py
# v1.8.0: Càlcul baseline amb finestres temporals (evita artefactes timeout)
# v1.7.0: Estadístiques timing (HPLC + TOC + toc_settings) per planificació/QC
# v1.6.0: Filtre injeccions control (MQ, NaOH) - configurable a hpsec_config.py
# v1.5.0: Sistema de matching amb confiança (EXACT/NORMALIZED/FUZZY) + detecció orfes
# v1.4.0: Validació mostres contra 1-HPLC-SEQ, detecció fitxers orfes
# v1.3.0: Afegit SNR, LOD, LOQ i baseline noise als consolidats
# v1.2.0: Refactor - timeout detection mogut a hpsec_core.py

import os
import re
import glob
import json
from datetime import datetime
from difflib import SequenceMatcher
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
from scipy.integrate import trapezoid

# Import funcions de detecció des de hpsec_core (Single Source of Truth)
from hpsec_core import (
    detect_timeout,
    format_timeout_status,
    TIMEOUT_CONFIG,
    calc_snr,
    detect_batman,
    repair_with_parabola,
    detect_main_peak,
)
from hpsec_utils import baseline_stats, baseline_stats_windowed
from hpsec_config import get_config

# Import validació KHP i alineació des de hpsec_calibrate (Single Source of Truth)
from hpsec_calibrate import (
    validate_khp_for_alignment,
    extract_khp_conc,
    get_injection_volume,
    # Funcions QAQC i alineació (migrades des de consolidate)
    QAQC_FOLDER,
    ALIGNMENT_LOG_FILE,
    get_qaqc_folder,
    ensure_qaqc_folder,
    get_alignment_log_path,
    load_alignment_log,
    save_alignment_log,
    add_alignment_entry,
    find_sibling_alignment,
    find_nearest_alignment,
    find_khp_for_alignment,
    calculate_column_alignment_shifts,
    get_a254_for_alignment,
)

# =============================================================================
# IMPORTS DES DE hpsec_import.py (Fase 1 - Lectura de dades)
# Mantinguts aquí per compatibilitat amb codi existent
# =============================================================================
from hpsec_import import (
    # Utilitats
    normalize_key,
    normalize_rep,
    is_khp,
    is_control_injection,
    obtenir_seq,
    seq_tag,
    split_sample_rep,
    clean_sample_name,
    skip_sample_direct,
    # Detecció mode
    is_bp_seq,
    detect_mode_from_folder,
    # Lectura master
    detect_master_format,
    trobar_excel_mestre,
    read_master_date,
    llegir_masterfile_nou,
    llegir_master_direct,
    # Lectura UIB
    llegir_doc_uib,
    netejar_nom_uib,
    # Lectura DAD
    llegir_dad_export3d,
    llegir_dad_1a,
    llegir_dad_amb_fallback,
    list_dad_files,
    # Matching
    get_valid_samples_from_hplc_seq,
    match_sample_confidence,
    is_sample_in_seq,
    CONFIDENCE_THRESHOLD,
    # DAD matching
    detect_dad_rep_style,
    dad_sample_rep_from_path,
    target_keys_from_desc,
    choose_best_candidate,
    # Verificació
    detect_replica_anomalies,
    check_sequence_files,
    # Nova funció principal
    import_sequence,
)


# =============================================================================
# IMPORTS DES DE hpsec_process.py (Fase 3 - Processament)
# Mantinguts aquí per compatibilitat amb codi existent
# =============================================================================
from hpsec_process import (
    # Config
    DEFAULT_PROCESS_CONFIG,
    # Utilitats
    truncate_chromatogram,
    mode_robust,
    # Baseline/smoothing
    get_baseline_correction,
    apply_smoothing,
    # Alineació
    align_signals_by_max,
    apply_shift,
    # DAD
    process_dad,
    # Pics
    find_peak_boundaries,
    # Nota: detect_main_peak no s'importa aquí perquè ja existeix local
    # Àrees
    calcular_fraccions_temps,
    calcular_arees_fraccions_complet,
    detectar_tmax_senyals,
    analyze_sample_areas,
    # SNR
    calculate_snr_info,
    # Funcions principals
    process_sample,
    process_sequence,
)


# NOTA: Les constants QAQC_FOLDER i ALIGNMENT_LOG_FILE ara s'importen de hpsec_calibrate.py
# NOTA: Les funcions d'alineació (get_qaqc_folder, load_alignment_log, etc.)
#       ara s'importen de hpsec_calibrate.py


def is_control_injection(sample_name, config=None):
    """
    Verifica si una mostra és una injecció de control (MQ, NaOH, etc.).

    Args:
        sample_name: Nom de la mostra
        config: Configuració (si None, es llegeix de get_config())

    Returns:
        True si és una injecció de control
    """
    if config is None:
        config = get_config()

    # ConfigManager.get() rep claus com arguments separats
    control_patterns = config.get("control_injections", "patterns", default=[])
    sample_upper = sample_name.upper()

    for pattern in control_patterns:
        if pattern.upper() in sample_upper:
            return True
    return False


# NOTA: Funcions d'alineació (get_qaqc_folder, ensure_qaqc_folder, get_alignment_log_path,
#       load_alignment_log, save_alignment_log, add_alignment_entry, find_sibling_alignment,
#       find_nearest_alignment) ara s'importen de hpsec_calibrate.py


# =============================================================================
# HISTORIAL KHP (TRACKING QUALITAT)
# =============================================================================
KHP_HISTORY_FILE = "KHP_History.json"


def get_khp_history_path(base_folder):
    """Retorna el path al fitxer d'historial KHP."""
    qaqc_path = get_qaqc_folder(base_folder)
    return os.path.join(qaqc_path, KHP_HISTORY_FILE)


def load_khp_history(base_folder):
    """Carrega l'historial KHP."""
    hist_path = get_khp_history_path(base_folder)
    if os.path.exists(hist_path):
        try:
            with open(hist_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {"entries": [], "version": "1.0", "stats": {}}


def save_khp_history(base_folder, history_data):
    """Guarda l'historial KHP."""
    ensure_qaqc_folder(base_folder)
    hist_path = get_khp_history_path(base_folder)
    try:
        with open(hist_path, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False


def add_khp_validation_entry(base_folder, khp_file, seq_name, method, validation_result, shift_info=None):
    """
    Afegeix una entrada a l'historial KHP.

    Args:
        base_folder: Carpeta de la SEQ
        khp_file: Nom del fitxer KHP
        seq_name: Nom de la SEQ
        method: "BP" o "COLUMN"
        validation_result: Dict retornat per validate_khp_for_alignment()
        shift_info: Dict amb shifts calculats (opcional)

    Returns:
        Dict amb l'entrada creada
    """
    history = load_khp_history(base_folder)

    entry = {
        "timestamp": datetime.now().isoformat(),
        "khp_file": khp_file,
        "seq_name": seq_name,
        "method": method,
        "valid": validation_result.get("valid", False),
        "issues": validation_result.get("issues", []),
        "warnings": validation_result.get("warnings", []),
        "metrics": validation_result.get("metrics", {}),
    }

    if shift_info:
        entry["shifts"] = shift_info

    # Evitar duplicats: actualitzar si ja existeix entrada per aquest KHP+SEQ
    existing_idx = None
    for i, e in enumerate(history["entries"]):
        if e.get("khp_file") == khp_file and e.get("seq_name") == seq_name:
            existing_idx = i
            break

    if existing_idx is not None:
        history["entries"][existing_idx] = entry
    else:
        history["entries"].append(entry)

    # Actualitzar estadístiques
    _update_khp_stats(history)

    save_khp_history(base_folder, history)
    return entry


def _update_khp_stats(history):
    """Actualitza estadístiques globals de l'historial KHP."""
    entries = history.get("entries", [])
    if not entries:
        history["stats"] = {}
        return

    valid_count = sum(1 for e in entries if e.get("valid", False))
    invalid_count = len(entries) - valid_count

    # Comptar issues
    issue_counts = {}
    for e in entries:
        for issue in e.get("issues", []):
            issue_type = issue.split(":")[0].strip()
            issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1

    # Estadístiques de mètriques
    ratios = [e.get("metrics", {}).get("ratio_a254_doc") for e in entries
              if e.get("metrics", {}).get("ratio_a254_doc") is not None]

    history["stats"] = {
        "total": len(entries),
        "valid": valid_count,
        "invalid": invalid_count,
        "valid_percent": round(100 * valid_count / len(entries), 1) if entries else 0,
        "issue_counts": issue_counts,
        "ratio_mean": round(np.mean(ratios), 4) if ratios else None,
        "ratio_std": round(np.std(ratios), 4) if ratios else None,
        "last_updated": datetime.now().isoformat()
    }


def get_khp_stats_summary(base_folder):
    """Retorna un resum de les estadístiques KHP."""
    history = load_khp_history(base_folder)
    return history.get("stats", {})


def find_valid_khp_for_method(base_folder, method, seq_date=None):
    """
    Busca un KHP vàlid per al mètode especificat en l'historial.

    Args:
        base_folder: Carpeta de la SEQ
        method: "BP" o "COLUMN"
        seq_date: Data de la SEQ (per trobar el més proper)

    Returns:
        Dict amb informació del KHP més proper vàlid, o None
    """
    history = load_khp_history(base_folder)

    # Filtrar per mètode i validesa
    candidates = [e for e in history.get("entries", [])
                  if e.get("method") == method and e.get("valid", False)]

    if not candidates:
        return None

    # Si tenim data, ordenar per proximitat temporal
    if seq_date:
        try:
            target_date = pd.to_datetime(seq_date)
            for c in candidates:
                c_timestamp = pd.to_datetime(c.get("timestamp", ""), errors='coerce')
                if pd.notna(c_timestamp):
                    c["_date_diff"] = abs((target_date - c_timestamp).days)
                else:
                    c["_date_diff"] = 9999
            candidates.sort(key=lambda x: x.get("_date_diff", 9999))
        except Exception:
            pass

    if candidates:
        best = candidates[0]
        return {
            "khp_file": best.get("khp_file", ""),
            "seq_name": best.get("seq_name", ""),
            "valid": best.get("valid", False),
            "metrics": best.get("metrics", {}),
            "shifts": best.get("shifts", {})
        }

    return None


# =============================================================================
# VERIFICACIÓ DE FITXERS (QA/QC)
# =============================================================================
FILE_CHECK_PREFIX = "File_Check_"


def detect_replica_anomalies(filenames):
    """
    Detecta anomalies en la numeració de rèpliques.

    Args:
        filenames: Llista de noms de fitxers

    Returns:
        List of dicts amb anomalies detectades
    """
    anomalies = []

    # Agrupar per mostra base
    samples = {}
    for fname in filenames:
        stem = os.path.splitext(fname)[0]
        # Treure sufixos UIB1B
        stem_clean = re.sub(r"_?UIB1B\d*", "", stem, flags=re.IGNORECASE)

        # Buscar patró de rèplica
        match_r = re.search(r"[_\-]R(\d+)$", stem_clean, flags=re.IGNORECASE)
        match_us = re.search(r"_(\d+)$", stem_clean)

        if match_r:
            base = stem_clean[:match_r.start()]
            rep = int(match_r.group(1))
            style = "R"
        elif match_us:
            base = stem_clean[:match_us.start()]
            rep = int(match_us.group(1))
            style = "UNDERSCORE"
        else:
            base = stem_clean
            rep = 1
            style = "NONE"

        base_key = re.sub(r"[^A-Za-z0-9]+", "", base).upper()
        if base_key not in samples:
            samples[base_key] = []
        samples[base_key].append({
            "filename": fname,
            "base": base,
            "rep": rep,
            "style": style
        })

    # Detectar anomalies
    for base_key, reps in samples.items():
        if len(reps) < 2:
            continue

        rep_nums = sorted([r["rep"] for r in reps])

        # Cas 1: Salts en numeració (ex: R1, R12 en lloc de R1, R2)
        for i, rep_num in enumerate(rep_nums):
            expected = i + 1
            if rep_num != expected and rep_num > 3:  # R12 en lloc de R2
                # Trobar el fitxer amb aquest número
                for r in reps:
                    if r["rep"] == rep_num:
                        # Proposar correcció
                        proposed_rep = expected
                        original = r["filename"]

                        # Construir nom proposat
                        if r["style"] == "R":
                            proposed = re.sub(
                                r"([_\-])R\d+(\.[^.]+)$",
                                rf"\1R{proposed_rep}\2",
                                original,
                                flags=re.IGNORECASE
                            )
                        elif r["style"] == "UNDERSCORE":
                            proposed = re.sub(
                                r"_\d+(\.[^.]+)$",
                                rf"_{proposed_rep}\1",
                                original
                            )
                        else:
                            proposed = original

                        if proposed != original:
                            anomalies.append({
                                "original": original,
                                "proposed": proposed,
                                "reason": f"R{rep_num} sembla error tipogràfic (esperat R{proposed_rep})",
                                "confidence": "HIGH" if rep_num > 9 else "MEDIUM",
                                "apply": False  # Per defecte no aplicar automàticament
                            })

    return anomalies


def check_sequence_files(seq_path, used_uib_files=None, used_dad_files=None):
    """
    Verifica els fitxers d'una seqüència i detecta discrepàncies.

    Args:
        seq_path: Ruta a la carpeta SEQ
        used_uib_files: Set de fitxers UIB utilitzats (paths complets)
        used_dad_files: Set de fitxers DAD utilitzats (paths complets)

    Returns:
        Dict amb info de verificació
    """
    used_uib_files = used_uib_files or set()
    used_dad_files = used_dad_files or set()

    path_csv = os.path.join(seq_path, "CSV")
    path_3d = os.path.join(seq_path, "Export3d")
    if not os.path.isdir(path_3d):
        path_3d = os.path.join(seq_path, "Export3D")

    result = {
        "seq_path": seq_path,
        "seq_name": os.path.basename(seq_path),
        "uib": {
            "found": [],
            "used": [],
            "orphan": [],
            "count_found": 0,
            "count_used": 0,
            "count_orphan": 0,
        },
        "dad": {
            "found": [],
            "used": [],
            "orphan": [],
            "count_found": 0,
            "count_used": 0,
            "count_orphan": 0,
        },
        "anomalies": [],
        "proposed_renames": [],
        "has_issues": False,
    }

    # Fitxers UIB
    if os.path.isdir(path_csv):
        uib_files = []
        for ext in ("*.csv", "*.CSV", "*.txt", "*.TXT"):
            uib_files.extend(glob.glob(os.path.join(path_csv, f"*UIB1B*{ext}")))
        uib_files = sorted(set(uib_files))

        result["uib"]["found"] = [os.path.basename(f) for f in uib_files]
        result["uib"]["count_found"] = len(uib_files)

        for f in uib_files:
            if f in used_uib_files:
                result["uib"]["used"].append(os.path.basename(f))
            else:
                result["uib"]["orphan"].append(os.path.basename(f))

        result["uib"]["count_used"] = len(result["uib"]["used"])
        result["uib"]["count_orphan"] = len(result["uib"]["orphan"])

        # Detectar anomalies UIB
        uib_anomalies = detect_replica_anomalies(result["uib"]["found"])
        result["proposed_renames"].extend(uib_anomalies)

    # Fitxers DAD
    if os.path.isdir(path_3d):
        dad_files = []
        for ext in ("*.csv", "*.CSV"):
            dad_files.extend(glob.glob(os.path.join(path_3d, ext)))
        # Excloure fitxers UIB que puguin estar aquí
        dad_files = [f for f in dad_files if "UIB1B" not in os.path.basename(f).upper()]
        dad_files = sorted(set(dad_files))

        result["dad"]["found"] = [os.path.basename(f) for f in dad_files]
        result["dad"]["count_found"] = len(dad_files)

        for f in dad_files:
            if f in used_dad_files:
                result["dad"]["used"].append(os.path.basename(f))
            else:
                result["dad"]["orphan"].append(os.path.basename(f))

        result["dad"]["count_used"] = len(result["dad"]["used"])
        result["dad"]["count_orphan"] = len(result["dad"]["orphan"])

        # Detectar anomalies DAD
        dad_anomalies = detect_replica_anomalies(result["dad"]["found"])
        result["proposed_renames"].extend(dad_anomalies)

    # Determinar si hi ha problemes
    result["has_issues"] = (
        result["uib"]["count_orphan"] > 0 or
        result["dad"]["count_orphan"] > 0 or
        len(result["proposed_renames"]) > 0
    )

    # Detectar mode (BP vs COLUMN)
    folder_name = os.path.basename(seq_path).upper()
    if "_BP" in folder_name or folder_name.endswith("BP"):
        result["detected_mode"] = "BP"
    else:
        result["detected_mode"] = "COLUMN"

    # Comptar mostres (basat en fitxers UIB o DAD únics)
    # Extreure noms base de mostres (sense rèplica)
    sample_names = set()

    for f in result["uib"]["found"]:
        # Extreure nom mostra de fitxers UIB (format: MOSTRA_UIB1B_Rx.csv)
        parts = f.replace("_UIB1B", "").replace(".csv", "").replace(".CSV", "").replace(".txt", "").replace(".TXT", "")
        # Treure _R1, _R2, etc.
        base = re.sub(r'_R\d+$', '', parts)
        if base and not is_khp(base):
            sample_names.add(base)

    for f in result["dad"]["found"]:
        # Extreure nom mostra de fitxers DAD
        parts = os.path.splitext(f)[0]
        base = re.sub(r'_R\d+$', '', parts)
        if base and not is_khp(base):
            sample_names.add(base)

    result["total_samples"] = len(sample_names)
    result["sample_names"] = sorted(sample_names)

    # Detectar si té KHP
    all_files = result["uib"]["found"] + result["dad"]["found"]
    result["has_khp"] = any(is_khp(f) for f in all_files)

    return result


def save_file_check_report(seq_path, check_result):
    """Guarda l'informe de verificació de fitxers al QAQC."""
    qaqc_path = ensure_qaqc_folder(seq_path)
    seq_name = os.path.basename(seq_path)

    report_path = os.path.join(qaqc_path, f"{FILE_CHECK_PREFIX}{seq_name}.json")

    report = {
        "timestamp": datetime.now().isoformat(),
        "seq_name": check_result["seq_name"],
        "seq_path": check_result["seq_path"],
        "summary": {
            "uib_found": check_result["uib"]["count_found"],
            "uib_used": check_result["uib"]["count_used"],
            "uib_orphan": check_result["uib"]["count_orphan"],
            "dad_found": check_result["dad"]["count_found"],
            "dad_used": check_result["dad"]["count_used"],
            "dad_orphan": check_result["dad"]["count_orphan"],
            "has_issues": check_result["has_issues"],
        },
        "orphan_files": {
            "uib": check_result["uib"]["orphan"],
            "dad": check_result["dad"]["orphan"],
        },
        "proposed_renames": check_result["proposed_renames"],
    }

    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        return report_path
    except Exception:
        return None


def apply_renames(seq_path, renames_to_apply):
    """
    Aplica els renames especificats.

    Args:
        seq_path: Ruta a la carpeta SEQ
        renames_to_apply: Llista de dicts amb {"original": ..., "proposed": ...}

    Returns:
        Dict amb resultats
    """
    path_csv = os.path.join(seq_path, "CSV")
    path_3d = os.path.join(seq_path, "Export3d")
    if not os.path.isdir(path_3d):
        path_3d = os.path.join(seq_path, "Export3D")

    results = {
        "success": [],
        "failed": [],
    }

    for rename in renames_to_apply:
        original = rename.get("original", "")
        proposed = rename.get("proposed", "")

        if not original or not proposed or original == proposed:
            continue

        # Buscar el fitxer original
        found_path = None
        for folder in [path_csv, path_3d]:
            if folder and os.path.isdir(folder):
                candidate = os.path.join(folder, original)
                if os.path.exists(candidate):
                    found_path = candidate
                    break

        if found_path:
            new_path = os.path.join(os.path.dirname(found_path), proposed)
            try:
                os.rename(found_path, new_path)
                results["success"].append({
                    "original": original,
                    "proposed": proposed,
                    "path": new_path
                })
            except Exception as e:
                results["failed"].append({
                    "original": original,
                    "proposed": proposed,
                    "error": str(e)
                })
        else:
            results["failed"].append({
                "original": original,
                "proposed": proposed,
                "error": "Fitxer no trobat"
            })

    return results


# =============================================================================
# CONFIGURACIÓ PER DEFECTE (CONSOLIDACIÓ)
# =============================================================================
DEFAULT_CONSOLIDATE_CONFIG = {
    "bp_baseline_win": 1.0,
    "col_baseline_start": 10.0,
    "target_wavelengths": [220, 252, 254, 272, 290, 362],
    "dad_subsample": 5,
    "peak_min_prominence_pct": 5.0,
    # Límit temporal màxim per cromatogrames (truncar dades posteriors)
    "max_time_min": 70.0,
    # Fraccions de temps per integració parcial (Column mode)
    "time_fractions": {
        "BioP": [0, 18],
        "HS": [18, 23],
        "BB": [23, 30],
        "SB": [30, 40],
        "LMW": [40, 70],
    },
}

# NOTA: TIMEOUT_CONFIG, detect_timeout i format_timeout_status ara estan a hpsec_core.py
# (Single Source of Truth per evitar duplicació)


# =============================================================================
# FUNCIONS UTILITAT
# =============================================================================
def truncate_chromatogram(t, y, max_time_min=None):
    """
    Trunca cromatograma a un temps màxim.

    Args:
        t: Array de temps en minuts
        y: Array de senyal (o llista d'arrays)
        max_time_min: Temps màxim (defecte: 70 min)

    Returns:
        t_trunc, y_trunc (o llista de y_trunc si y és llista)
    """
    if max_time_min is None:
        max_time_min = DEFAULT_CONSOLIDATE_CONFIG.get("max_time_min", 70.0)

    t = np.asarray(t)
    mask = t <= max_time_min

    if isinstance(y, (list, tuple)):
        return t[mask], [np.asarray(yi)[mask] if yi is not None else None for yi in y]
    else:
        y = np.asarray(y)
        return t[mask], y[mask]


def normalize_key(s):
    """Normalitza string per matching."""
    return re.sub(r"[^A-Za-z0-9]+", "", str(s or "")).upper()


def is_khp(name):
    """Detecta si és mostra KHP."""
    return "KHP" in str(name).upper()


# NOTA: validate_khp_for_alignment ara està a hpsec_calibrate.py
# Importat a l'inici del fitxer


def mode_robust(data, bins=50):
    """Calcula moda robusta amb histograma."""
    if data is None or len(data) == 0:
        return 0.0
    counts, edges = np.histogram(np.asarray(data), bins=bins)
    i = int(np.argmax(counts))
    return 0.5 * (edges[i] + edges[i + 1])


def obtenir_seq(folder):
    """Extreu ID de seqüència del nom de carpeta."""
    nom = os.path.basename(os.path.normpath(folder))
    m = re.search(r"(\d+[A-Za-z]?)", nom)
    return m.group(1) if m else "000"


def seq_tag(seq, bp):
    """Afegeix _BP si és seqüència BP."""
    if bp and "BP" not in seq.upper():
        return f"{seq}_BP"
    return seq


def is_bp_seq(folder, sample_dad_files=None):
    """Detecta si és seqüència BP."""
    name = os.path.basename(os.path.normpath(folder)).upper()
    if re.search(r"(^|[_\-\s])BP($|[_\-\s])", name):
        return True

    if sample_dad_files:
        for p in sample_dad_files[:3]:
            try:
                df, st = llegir_dad_export3d(p)
                if st.startswith("OK") and not df.empty and "time (min)" in df.columns:
                    tmax = pd.to_numeric(df["time (min)"], errors="coerce").max()
                    if pd.notna(tmax) and float(tmax) <= 18.0:
                        return True
            except Exception:
                continue
    return False


def skip_sample_direct(sample_desc):
    """Determina si s'ha d'ignorar una mostra."""
    d = str(sample_desc or "").strip().upper()
    if d in ("", "0"):
        return True
    if "PRE-HPLC" in d or ("POST" in d and "HPLC" in d):
        return True
    return False


def split_sample_rep(sample_desc):
    """Separa nom de mostra i rèplica."""
    s = str(sample_desc or "").strip()
    if not s:
        return "", None
    m = re.match(r"^(.*?)(?:[_\-\s]?R(\d+))\s*$", s, flags=re.IGNORECASE)
    if m and m.group(1).strip():
        return m.group(1).strip(), m.group(2)
    return s, None


def clean_sample_name(sample_desc):
    """Neteja nom de mostra."""
    s = str(sample_desc or "").strip()
    return re.sub(r"[^A-Za-z0-9]+", "", s) or "SAMPLE"


def normalize_rep(rep):
    """Normalitza número de rèplica."""
    if rep is None:
        return ""
    s = str(rep).strip()
    if not s:
        return ""
    try:
        return str(int(s))
    except Exception:
        return s


# =============================================================================
# FUNCIONS ALINEACIÓ TEMPORAL
# =============================================================================
def align_signals_by_max(t_ref, y_ref, t_other, y_other):
    """
    Alinea dos senyals pel màxim i interpola el segon a l'escala de temps del primer.

    Args:
        t_ref: temps de referència
        y_ref: senyal de referència
        t_other: temps del senyal a alinear
        y_other: senyal a alinear

    Returns:
        y_aligned: senyal alineat i interpolat a t_ref
        shift: desplaçament aplicat (minuts)
    """
    # Trobar màxims
    idx_max_ref = np.argmax(y_ref)
    idx_max_other = np.argmax(y_other)

    t_max_ref = t_ref[idx_max_ref]
    t_max_other = t_other[idx_max_other]

    # Calcular shift necessari
    shift = t_max_ref - t_max_other

    # Aplicar shift al temps
    t_other_shifted = t_other + shift

    # Interpolar a l'escala de temps de referència
    y_aligned = np.interp(t_ref, t_other_shifted, y_other, left=0, right=0)

    return y_aligned, shift


def apply_shift(t_ref, t_signal, y_signal, shift):
    """
    Aplica un shift temporal i interpola a l'escala de referència.

    Args:
        t_ref: escala de temps de referència
        t_signal: escala de temps del senyal
        y_signal: senyal a desplaçar
        shift: desplaçament en minuts (positiu = avançar, negatiu = retardar)

    Returns:
        y_shifted: senyal desplaçat i interpolat a t_ref
    """
    t_shifted = t_signal + shift
    y_shifted = np.interp(t_ref, t_shifted, y_signal, left=0, right=0)
    return y_shifted


# =============================================================================
# FUNCIONS CERCA KHP PER ALINEACIÓ
# =============================================================================
def find_khp_in_folder(folder):
    """Cerca fitxers KHP consolidats en una carpeta."""
    if not folder or not os.path.exists(folder):
        return []
    files = glob.glob(os.path.join(folder, "*KHP*.xlsx"))
    files = [f for f in files if not os.path.basename(f).startswith("~$")]
    return files


# NOTA: find_khp_for_alignment i calculate_column_alignment_shifts
#       ara s'importen de hpsec_calibrate.py


# =============================================================================
# FUNCIONS BASELINE I SMOOTHING
# =============================================================================
def get_baseline_correction(t, y, mode_type="COL", config=None, use_end=False):
    """
    Correcció de baseline segons mode BP o COLUMN.

    Args:
        t: array de temps (min)
        y: array de senyal
        mode_type: "BP" o "COL"
        config: configuració
        use_end: si True, usa el final del cromatograma per baseline (recomanat per BP Direct)

    Returns:
        Array amb el valor de baseline
    """
    config = config or DEFAULT_CONSOLIDATE_CONFIG
    t = np.asarray(t)
    y = np.asarray(y)

    if mode_type == "BP":
        if use_end:
            # Per BP Direct: usar els últims punts (després del pic)
            n = len(y)
            n_edge = max(20, n // 5)  # Últim 20%
            val = float(np.median(y[-n_edge:]))
        else:
            # Per BP UIB: usar primers punts
            mask = t < config["bp_baseline_win"]
            val = mode_robust(y[mask]) if np.sum(mask) > 10 else float(np.nanmin(y))
        return np.full_like(y, val, dtype=float)

    # COLUMN: usar primers punts
    mask = t < config["col_baseline_start"]
    val = mode_robust(y[mask]) if np.sum(mask) > 10 else float(np.nanmin(y))
    return np.full_like(y, val, dtype=float)


def apply_smoothing(y):
    """Aplica suavitzat Savgol."""
    y = np.asarray(y)
    if len(y) < 11:
        return y
    return savgol_filter(y, 11, 3)


# =============================================================================
# FUNCIONS LECTURA FITXERS
# =============================================================================

def detect_master_format(filepath):
    """
    Detecta el format del fitxer mestre.

    Returns:
        "NEW" si és nou MasterFile (0-INFO, 1-HPLC-SEQ, 2-TOC, 4-TOC_CALC)
        "OLD" si és format antic (0-CHECK, 2-TOC, 4-SEQ_DATA)
        None si no és reconegut
    """
    try:
        xl = pd.ExcelFile(filepath, engine="openpyxl")
        sheets = set(xl.sheet_names)

        # Format nou: té 0-INFO i 4-TOC_CALC (o 1-HPLC-SEQ)
        if "0-INFO" in sheets and ("4-TOC_CALC" in sheets or "1-HPLC-SEQ" in sheets):
            return "NEW"

        # Format antic: té 0-CHECK i 4-SEQ_DATA
        if any(s.lower() == "0-check" for s in sheets) and "4-SEQ_DATA" in sheets:
            return "OLD"

        # Compatibilitat: si té 2-TOC i 4-SEQ_DATA, és antic
        if "2-TOC" in sheets and "4-SEQ_DATA" in sheets:
            return "OLD"

        return None
    except Exception:
        return None


def read_master_date(seq_folder):
    """Llegeix data del fitxer mestre (compatible amb format antic i nou)."""
    xls = glob.glob(os.path.join(seq_folder, "*.xlsx"))
    for f in xls:
        if "~$" in os.path.basename(f):
            continue
        try:
            xf = pd.ExcelFile(f, engine="openpyxl")

            # Primer intentar format nou (0-INFO)
            if "0-INFO" in xf.sheet_names:
                df = pd.read_excel(xf, sheet_name="0-INFO", header=None, engine="openpyxl")
                # Buscar fila "Date"
                for i, row in df.iterrows():
                    if str(row.iloc[0]).strip().lower() == "date":
                        val = row.iloc[1]
                        if pd.notna(val):
                            return str(val)
                continue

            # Fallback format antic (0-CHECK)
            sheet = None
            for s in xf.sheet_names:
                if str(s).strip().lower() == "0-check":
                    sheet = s
                    break
            if sheet is None:
                sheet = 0
            df = pd.read_excel(xf, sheet_name=sheet, header=None, engine="openpyxl")
            val = df.iloc[1, 1]
            if pd.notna(val):
                return str(val)
        except Exception:
            pass
    return ""


def llegir_doc_uib(path):
    """Llegeix fitxer DOC format UIB (CSV tab-separated)."""
    for enc in ["utf-16", "utf-8"]:
        try:
            df = pd.read_csv(path, sep="\t", encoding=enc, header=None, engine="python")
            df = df.iloc[:, [0, 1]]
            df.columns = ["time (min)", "DOC"]
            return df, f"OK{' (UTF-8)' if enc == 'utf-8' else ''}"
        except Exception:
            continue
    return pd.DataFrame(), "Error"


def llegir_dad_export3d(path):
    """Llegeix fitxer DAD Export3D (CSV comma-separated)."""
    for enc in ["utf-16", "utf-8"]:
        try:
            df = pd.read_csv(path, sep=",", encoding=enc, engine="python")
            if df.shape[1] == 0:
                return pd.DataFrame(), "Buit"
            cols = list(df.columns)
            cols[0] = "time (min)"
            out_cols = [cols[0]]
            for c in cols[1:]:
                sc = str(c).strip()
                try:
                    v = float(sc)
                    out_cols.append(str(int(v)) if v.is_integer() else str(v))
                except Exception:
                    out_cols.append(sc)
            df.columns = out_cols
            return df, f"OK{' (UTF-8)' if enc == 'utf-8' else ''}"
        except Exception:
            continue
    return pd.DataFrame(), "Error"


def llegir_dad_1a(path, wavelength="254"):
    """
    Llegeix fitxer DAD1A (format Agilent: UTF-16, tab-separated, sense capçalera).

    El canal 1A correspon a 254nm. Altres canals:
    - DAD1A = 254nm
    - DAD1B = 220nm (si existeix)

    Args:
        path: Camí al fitxer DAD1A
        wavelength: Nom de la columna de wavelength (default "254")

    Returns:
        (DataFrame, status): DataFrame amb columns ['time (min)', wavelength], status string
    """
    for enc in ["utf-16", "utf-16-le", "utf-8", "latin1"]:
        try:
            df = pd.read_csv(path, encoding=enc, sep="\t", header=None,
                           names=["time (min)", wavelength])
            # Verificar que tenim dades numèriques
            df["time (min)"] = pd.to_numeric(df["time (min)"], errors="coerce")
            df[wavelength] = pd.to_numeric(df[wavelength], errors="coerce")

            # Filtrar NaN
            df = df.dropna()

            if len(df) < 10:
                continue

            return df, f"OK (DAD1A, {enc})"
        except Exception:
            continue
    return pd.DataFrame(), "Error"


def llegir_dad_amb_fallback(path_export3d, path_dad1a=None, wavelength="254"):
    """
    Llegeix DAD amb fallback: primer Export3D, després DAD1A.

    Args:
        path_export3d: Camí al fitxer Export3D (pot ser None)
        path_dad1a: Camí al fitxer DAD1A (pot ser None)
        wavelength: Wavelength a extreure (default "254")

    Returns:
        (t, y, source): Arrays de temps i senyal, i string indicant la font
    """
    # Primer intentar Export3D
    if path_export3d and os.path.exists(path_export3d):
        df, status = llegir_dad_export3d(path_export3d)
        if status.startswith("OK") and not df.empty:
            if wavelength in df.columns:
                t = pd.to_numeric(df["time (min)"], errors="coerce").to_numpy()
                y = pd.to_numeric(df[wavelength], errors="coerce").to_numpy()
                valid = np.isfinite(t) & np.isfinite(y)
                if np.sum(valid) > 10:
                    return t[valid], y[valid], "Export3D"

    # Fallback a DAD1A
    if path_dad1a and os.path.exists(path_dad1a):
        df, status = llegir_dad_1a(path_dad1a, wavelength)
        if status.startswith("OK") and not df.empty:
            t = df["time (min)"].to_numpy()
            y = df[wavelength].to_numpy()
            if len(t) > 10:
                return t, y, "DAD1A"

    return None, None, "NOT_FOUND"


# NOTA: get_a254_for_alignment ara s'importa de hpsec_calibrate.py


def netejar_nom_uib(nom_fitxer):
    """Neteja nom de fitxer UIB i extreu mostra/rèplica."""
    stem = os.path.splitext(nom_fitxer)[0]
    stem = re.sub(r"UIB1B\d*", "", stem, flags=re.IGNORECASE)
    tokens = [t for t in re.split(r"[_\-\s]+", stem) if t]

    rep = None
    if tokens:
        last = tokens[-1]
        if last.isdigit():
            rep = last
            tokens = tokens[:-1]
        else:
            m = re.fullmatch(r"R(\d+)", last, flags=re.IGNORECASE)
            if m:
                rep = m.group(1)
                tokens = tokens[:-1]

    if rep is None:
        rep = "1"

    mostra = "_".join(tokens).strip("_")
    if not mostra:
        mostra = os.path.splitext(nom_fitxer)[0]
    return mostra, rep


def trobar_excel_mestre(folder_seq, prefer_new=True):
    """
    Troba Excel mestre (compatible amb format antic i nou).

    Args:
        folder_seq: Carpeta de la seqüència
        prefer_new: Si True, prioritza MasterFile nou sobre format antic

    Returns:
        Tuple (path, format) on format és "NEW" o "OLD", o (None, None) si no es troba
    """
    candidats = []
    for ext in ("*.xlsx", "*.XLSX", "*.xlsm", "*.XLSM"):
        candidats.extend(glob.glob(os.path.join(folder_seq, ext)))
    candidats = [p for p in candidats if not os.path.basename(p).startswith("~$")]

    # Separar per tipus
    new_format = []
    old_format = []

    for p in sorted(candidats):
        fmt = detect_master_format(p)
        if fmt == "NEW":
            new_format.append(p)
        elif fmt == "OLD":
            old_format.append(p)

    # Prioritzar segons preferència
    if prefer_new and new_format:
        # Prioritzar fitxers amb "MasterFile" al nom
        for p in new_format:
            if "MasterFile" in os.path.basename(p):
                return p, "NEW"
        return new_format[0], "NEW"

    if old_format:
        return old_format[0], "OLD"

    if new_format:
        return new_format[0], "NEW"

    return None, None


def llegir_masterfile_nou(filepath):
    """
    Llegeix el nou format MasterFile.

    Returns:
        dict amb:
            - info: dict amb metadata de 0-INFO
            - hplc_seq: DataFrame de 1-HPLC-SEQ
            - toc: DataFrame de 2-TOC
            - toc_calc: DataFrame de 4-TOC_CALC
            - dad_khp: DataFrame de 3-DAD_KHP (si existeix)
    """
    result = {
        "info": {},
        "hplc_seq": None,
        "toc": None,
        "toc_calc": None,
        "dad_khp": None,
    }

    try:
        xl = pd.ExcelFile(filepath, engine="openpyxl")

        # 0-INFO
        if "0-INFO" in xl.sheet_names:
            df_info = pd.read_excel(xl, sheet_name="0-INFO", header=None, engine="openpyxl")
            for _, row in df_info.iterrows():
                key = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ""
                val = row.iloc[1] if len(row) > 1 and pd.notna(row.iloc[1]) else None
                if key:
                    result["info"][key] = val

        # 1-HPLC-SEQ
        if "1-HPLC-SEQ" in xl.sheet_names:
            result["hplc_seq"] = pd.read_excel(xl, sheet_name="1-HPLC-SEQ", engine="openpyxl")

        # 2-TOC
        if "2-TOC" in xl.sheet_names:
            result["toc"] = pd.read_excel(xl, sheet_name="2-TOC", header=6, engine="openpyxl")

        # 4-TOC_CALC
        if "4-TOC_CALC" in xl.sheet_names:
            result["toc_calc"] = pd.read_excel(xl, sheet_name="4-TOC_CALC", engine="openpyxl")

        # 3-DAD_KHP (opcional)
        if "3-DAD_KHP" in xl.sheet_names:
            result["dad_khp"] = pd.read_excel(xl, sheet_name="3-DAD_KHP", header=1, engine="openpyxl")

    except Exception as e:
        result["error"] = str(e)

    return result


def get_valid_samples_from_hplc_seq(master_data):
    """
    Extreu la llista de mostres vàlides de 1-HPLC-SEQ.

    Aquesta és la font de veritat per saber quines mostres pertanyen a una SEQ.
    Els fitxers UIB/DAD que no coincideixin amb aquestes mostres són orfes
    (exportats per error d'una altra seqüència).

    Args:
        master_data: Dict retornat per llegir_masterfile_nou()

    Returns:
        set de noms de mostres (normalitzats) que pertanyen a la SEQ
    """
    df_hplc = master_data.get("hplc_seq")
    if df_hplc is None or df_hplc.empty:
        return set()

    valid_samples = set()

    # Buscar columna "Sample Name" (pot tenir espais extra)
    sample_col = None
    for col in df_hplc.columns:
        if 'sample' in str(col).lower() and 'name' in str(col).lower():
            sample_col = col
            break

    if sample_col is None:
        return set()

    # Extreure noms únics i normalitzar
    for val in df_hplc[sample_col].dropna().unique():
        name = str(val).strip()
        if name:
            # Guardar nom original i normalitzat
            valid_samples.add(name)
            valid_samples.add(normalize_key(name))

    return valid_samples


def validate_hplc_seq(master_data, uib_files=None):
    """
    Valida la integritat de 1-HPLC-SEQ i creua amb fitxers existents.

    Quality checks:
    1. Fulla no buida
    2. Line# seqüencial (1, 2, 3... N sense gaps)
    3. Creuar amb fitxers UIB per detectar discrepàncies

    Args:
        master_data: Dict retornat per llegir_masterfile_nou()
        uib_files: Llista de fitxers UIB existents (opcional)

    Returns:
        dict amb:
            - valid: bool, si passa tots els checks
            - issues: list de problemes detectats
            - missing_lines: list de Line# que falten
            - expected_samples: int, mostres esperades segons Line#
            - actual_samples: int, mostres al MasterFile
            - orphan_files: list de fitxers sense entrada al MasterFile
            - missing_files: list d'entrades MasterFile sense fitxer
    """
    result = {
        "valid": True,
        "issues": [],
        "missing_lines": [],
        "expected_samples": 0,
        "actual_samples": 0,
        "orphan_files": [],
        "missing_files": [],
    }

    df_hplc = master_data.get("hplc_seq")

    # Check 1: Fulla buida
    if df_hplc is None or df_hplc.empty:
        result["valid"] = False
        result["issues"].append({
            "type": "EMPTY_HPLC_SEQ",
            "severity": "CRITICAL",
            "message": "Fulla 1-HPLC-SEQ buida o no existeix"
        })
        return result

    # Buscar columna Line#
    line_col = None
    for col in df_hplc.columns:
        col_str = str(col).lower()
        if 'line' in col_str and '#' in str(col):
            line_col = col
            break
        if col_str == 'line#':
            line_col = col
            break

    if line_col is None:
        # Intentar primera columna
        line_col = df_hplc.columns[0]

    # Check 2: Line# seqüencial
    try:
        lines = df_hplc[line_col].dropna().astype(int).tolist()
        if lines:
            min_line = min(lines)
            max_line = max(lines)
            expected_lines = set(range(min_line, max_line + 1))
            actual_lines = set(lines)
            missing = sorted(expected_lines - actual_lines)

            result["expected_samples"] = len(expected_lines)
            result["actual_samples"] = len(actual_lines)

            if missing:
                result["valid"] = False
                result["missing_lines"] = missing
                result["issues"].append({
                    "type": "MISSING_LINES",
                    "severity": "ERROR",
                    "message": f"Falten {len(missing)} línies a 1-HPLC-SEQ: {missing[:10]}{'...' if len(missing) > 10 else ''}",
                    "missing": missing
                })
    except Exception as e:
        result["issues"].append({
            "type": "LINE_CHECK_ERROR",
            "severity": "WARNING",
            "message": f"No s'ha pogut verificar seqüència Line#: {e}"
        })

    # Check 3: Creuar amb fitxers UIB
    if uib_files:
        # Obtenir Sample_Rep del MasterFile
        sample_rep_col = None
        for col in df_hplc.columns:
            if 'sample_rep' in str(col).lower():
                sample_rep_col = col
                break

        master_samples = set()
        if sample_rep_col:
            for val in df_hplc[sample_rep_col].dropna():
                master_samples.add(normalize_key(str(val).strip()))
        else:
            # Fallback: usar Sample Name + Inj#
            sample_col = None
            inj_col = None
            for col in df_hplc.columns:
                if 'sample' in str(col).lower() and 'name' in str(col).lower():
                    sample_col = col
                if 'inj' in str(col).lower() and '#' in str(col):
                    inj_col = col
            if sample_col and inj_col:
                for _, row in df_hplc.iterrows():
                    sample = str(row[sample_col]).strip() if pd.notna(row[sample_col]) else ""
                    inj = int(row[inj_col]) if pd.notna(row[inj_col]) else 1
                    if sample:
                        master_samples.add(normalize_key(f"{sample}_R{inj}"))

        # Obtenir samples dels fitxers UIB
        file_samples = set()
        for f in uib_files:
            nom = os.path.basename(f)
            mostra, rep = netejar_nom_uib(nom)
            file_samples.add(normalize_key(f"{mostra}_R{rep}"))

        # Fitxers sense entrada al MasterFile
        orphans = file_samples - master_samples
        if orphans:
            result["orphan_files"] = sorted(orphans)
            # No és error crític, pot ser normal (fitxers d'altra SEQ)

        # Entrades MasterFile sense fitxer
        missing = master_samples - file_samples
        if missing:
            result["missing_files"] = sorted(missing)
            result["issues"].append({
                "type": "MISSING_FILES",
                "severity": "WARNING",
                "message": f"{len(missing)} mostres al MasterFile sense fitxer UIB: {sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}"
            })

    # Check 4: Verificar dades DOC primeres mostres (memòria DOC esgotada)
    # Si la memòria del TOC s'esgota, sobreescriu els primers DOCs
    toc_calc_df = master_data.get("toc_calc")
    if toc_calc_df is not None and not toc_calc_df.empty:
        # Trobar columnes
        sample_col = None
        toc_row_col = None
        for col in toc_calc_df.columns:
            col_lower = str(col).lower()
            if 'sample' in col_lower:
                sample_col = col
            elif 'toc_row' in col_lower or 'toc row' in col_lower:
                toc_row_col = col

        if sample_col and toc_row_col:
            # Agrupar per sample i comptar punts
            sample_points = toc_calc_df.groupby(sample_col).size().reset_index(name='npts')
            sample_points = sample_points[sample_points[sample_col].notna()]

            if len(sample_points) >= 3:
                # Comparar punts de les primeres mostres amb la mediana
                first_samples = sample_points.head(3)
                median_pts = sample_points['npts'].median()

                incomplete_samples = []
                for _, row in first_samples.iterrows():
                    if row['npts'] < median_pts * 0.5:  # Menys del 50% de punts
                        incomplete_samples.append({
                            "sample": str(row[sample_col]),
                            "npts": int(row['npts']),
                            "expected": int(median_pts)
                        })

                if incomplete_samples:
                    result["incomplete_first_samples"] = incomplete_samples
                    result["issues"].append({
                        "type": "INCOMPLETE_FIRST_SAMPLES",
                        "severity": "WARNING",
                        "message": f"Primeres mostres amb dades incompletes (memòria DOC esgotada?): {[s['sample'] for s in incomplete_samples]}"
                    })

    return result


def match_sample_confidence(sample_name, valid_samples):
    """
    Cerca la millor coincidència d'una mostra amb la llista vàlida i retorna la confiança.

    Args:
        sample_name: Nom de la mostra (del fitxer UIB)
        valid_samples: Set de mostres vàlides (de get_valid_samples_from_hplc_seq)

    Returns:
        dict amb:
            - matched: bool, si s'ha trobat coincidència
            - best_match: str, nom de la mostra vàlida que coincideix (o None)
            - confidence: float 0-100, nivell de confiança
            - match_type: str, tipus de coincidència
    """
    result = {
        "matched": False,
        "best_match": None,
        "confidence": 0.0,
        "match_type": "NOT_FOUND"
    }

    if not valid_samples:
        # Si no tenim info de HPLC-SEQ, assumim match perfecte
        return {
            "matched": True,
            "best_match": sample_name,
            "confidence": 100.0,
            "match_type": "NO_VALIDATION"
        }

    sample_norm = normalize_key(sample_name)
    sample_upper = sample_name.upper()

    # 1. Coincidència exacta (100%)
    if sample_name in valid_samples:
        return {
            "matched": True,
            "best_match": sample_name,
            "confidence": 100.0,
            "match_type": "EXACT"
        }

    # 2. Coincidència normalitzada (95%)
    for vs in valid_samples:
        if normalize_key(vs) == sample_norm:
            return {
                "matched": True,
                "best_match": vs,
                "confidence": 95.0,
                "match_type": "NORMALIZED"
            }

    # 3. Case-insensitive (90%)
    for vs in valid_samples:
        if vs.upper() == sample_upper:
            return {
                "matched": True,
                "best_match": vs,
                "confidence": 90.0,
                "match_type": "CASE_INSENSITIVE"
            }

    # 4. Variants (espais, guions, underscores) (85%)
    variants = [
        sample_name.replace("_", " "),
        sample_name.replace(" ", "_"),
        sample_name.replace("-", "_"),
        sample_name.replace("_", "-"),
        sample_name.replace("_", ""),
        sample_name.replace(" ", ""),
    ]
    for v in variants:
        v_norm = normalize_key(v)
        v_upper = v.upper()
        for vs in valid_samples:
            if vs == v or normalize_key(vs) == v_norm or vs.upper() == v_upper:
                return {
                    "matched": True,
                    "best_match": vs,
                    "confidence": 85.0,
                    "match_type": "VARIANT"
                }

    # 5. Fuzzy matching amb SequenceMatcher
    best_ratio = 0.0
    best_match = None

    for vs in valid_samples:
        # Comparar amb nom original i normalitzat
        ratio1 = SequenceMatcher(None, sample_upper, vs.upper()).ratio()
        ratio2 = SequenceMatcher(None, sample_norm, normalize_key(vs)).ratio()
        ratio = max(ratio1, ratio2)

        if ratio > best_ratio:
            best_ratio = ratio
            best_match = vs

    # NOTA: Fuzzy matching desactivat per a IDs científics
    # FR2606 vs FR2608 són mostres DIFERENTS (el número és l'ID)
    # Només suggerim si la diferència és molt alta (>95%) i no és numèrica

    # Comprovar si la diferència és només en la part numèrica final
    # En aquest cas, NO suggerir - són mostres diferents
    import re
    sample_base = re.sub(r'\d+$', '', sample_name)  # "FR" de "FR2606"

    if best_ratio >= 0.95:  # Molt similar (probablement error tipogràfic real)
        # Verificar que no és només diferència numèrica
        best_base = re.sub(r'\d+$', '', best_match) if best_match else ""
        if sample_base == best_base:
            # Mateixa base, diferent número = mostres DIFERENTS, no suggerir
            return result

        confidence = best_ratio * 100
        return {
            "matched": True,
            "best_match": best_match,
            "confidence": confidence,
            "match_type": "FUZZY"
        }

    # 6. No trobat - sense suggeriment per evitar confusió
    return result


# Llindar de confiança per acceptar automàticament
CONFIDENCE_THRESHOLD = 85.0


def is_sample_in_seq(sample_name, valid_samples):
    """
    Verifica si una mostra pertany a la seqüència.

    Args:
        sample_name: Nom de la mostra (del fitxer UIB)
        valid_samples: Set de mostres vàlides (de get_valid_samples_from_hplc_seq)

    Returns:
        True si la mostra pertany a la SEQ, False si és orfe
    """
    match_info = match_sample_confidence(sample_name, valid_samples)
    return match_info["matched"] and match_info["confidence"] >= CONFIDENCE_THRESHOLD


def calculate_timing_stats(master_data, timeout_info_list=None):
    """
    Calcula estadístiques de timing de la seqüència per a planificació.

    Extreu timing de:
    - 1-HPLC-SEQ: Intervals entre injeccions HPLC (per planificar futures SEQs)
    - 2-TOC: Intervals entre mesures TOC (comportament real)

    Args:
        master_data: Dict retornat per llegir_masterfile_nou()
        timeout_info_list: Llista de timeout_info per cada mostra (opcional)

    Returns:
        dict amb timing HPLC i TOC
    """
    result = {
        "hplc": None,  # Timing injeccions HPLC
        "toc": None,   # Timing mesures TOC
        "t0_first_timeout_min": None,  # Posició primer timeout
    }

    # === TIMING HPLC (1-HPLC-SEQ) ===
    df_hplc = master_data.get("hplc_seq")
    if df_hplc is not None and not df_hplc.empty:
        # Trobar columna de data d'injecció
        date_col = None
        for col in df_hplc.columns:
            col_str = str(col).lower()
            if 'injection' in col_str and 'date' in col_str:
                date_col = col
                break

        if date_col:
            timestamps = []
            for _, row in df_hplc.iterrows():
                date_val = row[date_col]
                if pd.notna(date_val):
                    try:
                        timestamps.append(pd.to_datetime(date_val))
                    except:
                        pass

            if len(timestamps) >= 2:
                timestamps = sorted(timestamps)
                intervals_min = [(timestamps[i] - timestamps[i-1]).total_seconds() / 60.0
                                 for i in range(1, len(timestamps))]

                result["hplc"] = {
                    "t0_first": timestamps[0].isoformat(),
                    "t0_last": timestamps[-1].isoformat(),
                    "total_duration_min": round((timestamps[-1] - timestamps[0]).total_seconds() / 60.0, 2),
                    "interval_mean_min": round(float(np.mean(intervals_min)), 2),
                    "interval_std_min": round(float(np.std(intervals_min)), 2),
                    "interval_min_min": round(min(intervals_min), 2),
                    "interval_max_min": round(max(intervals_min), 2),
                    "n_injections": len(timestamps),
                }

    # === TIMING TOC (2-TOC) ===
    df_toc = master_data.get("toc")
    if df_toc is not None and not df_toc.empty:
        # Trobar columna de temps
        time_col = None
        for col in df_toc.columns:
            col_str = str(col).lower()
            if 'date' in col_str and 'start' in col_str:
                time_col = col
                break
        if time_col is None and len(df_toc.columns) > 3:
            time_col = df_toc.columns[3]

        if time_col:
            toc_times = []
            for _, row in df_toc.iterrows():
                time_val = row[time_col]
                if pd.notna(time_val):
                    try:
                        toc_times.append(pd.to_datetime(time_val))
                    except:
                        pass

            if len(toc_times) >= 2:
                toc_times = sorted(toc_times)
                # Calcular intervals (en segons per detectar timeouts)
                intervals_sec = [(toc_times[i] - toc_times[i-1]).total_seconds()
                                 for i in range(1, len(toc_times))]

                # Separar intervals normals de timeouts
                normal_intervals = [i for i in intervals_sec if i < 60]  # < 60s és normal
                timeout_intervals = [i for i in intervals_sec if i >= 60]  # >= 60s és timeout

                result["toc"] = {
                    "t0_first": toc_times[0].isoformat(),
                    "t0_last": toc_times[-1].isoformat(),
                    "total_duration_min": round((toc_times[-1] - toc_times[0]).total_seconds() / 60.0, 2),
                    "n_measurements": len(toc_times),
                    "interval_normal_mean_sec": round(float(np.mean(normal_intervals)), 2) if normal_intervals else None,
                    "n_timeouts_detected": len(timeout_intervals),
                    "timeout_duration_mean_sec": round(float(np.mean(timeout_intervals)), 1) if timeout_intervals else None,
                }

        # === TOC SETTINGS (acid/oxidizer flow) ===
        # Buscar columnes de flow rate
        acid_col = oxidizer_col = None
        for col in df_toc.columns:
            col_str = str(col).lower()
            if 'acid' in col_str and 'flow' in col_str:
                acid_col = col
            elif 'oxid' in col_str and 'flow' in col_str:
                oxidizer_col = col

        toc_settings = {}
        if acid_col is not None:
            acid_vals = pd.to_numeric(df_toc[acid_col], errors='coerce').dropna()
            if len(acid_vals) > 0:
                toc_settings["acid_flow_ul_min"] = round(float(acid_vals.median()), 2)
                # Verificar si constant
                if acid_vals.std() > 0.1:
                    toc_settings["acid_flow_warning"] = f"Variabilitat detectada (std={acid_vals.std():.2f})"

        if oxidizer_col is not None:
            ox_vals = pd.to_numeric(df_toc[oxidizer_col], errors='coerce').dropna()
            if len(ox_vals) > 0:
                toc_settings["oxidizer_flow_ul_min"] = round(float(ox_vals.median()), 2)
                if ox_vals.std() > 0.1:
                    toc_settings["oxidizer_flow_warning"] = f"Variabilitat detectada (std={ox_vals.std():.2f})"

        if toc_settings:
            result["toc_settings"] = toc_settings

    # === POSICIÓ PRIMER TIMEOUT ===
    if timeout_info_list:
        for to_info in timeout_info_list:
            if to_info and to_info.get("detected", False):
                timeouts = to_info.get("timeouts", [])
                if timeouts:
                    result["t0_first_timeout_min"] = round(timeouts[0].get("t_start_min", 0), 2)
                    break

    # === TEMPS PRE-INJECCIÓ (TOC abans de primera injecció HPLC) ===
    if result["hplc"] and result["toc"]:
        try:
            toc_start = pd.to_datetime(result["toc"]["t0_first"])
            hplc_start = pd.to_datetime(result["hplc"]["t0_first"])
            pre_injection_min = (hplc_start - toc_start).total_seconds() / 60.0
            result["pre_injection_time_min"] = round(pre_injection_min, 2)
        except:
            result["pre_injection_time_min"] = None

    return result if (result["hplc"] or result["toc"]) else None


def calculate_toc_calc_on_the_fly(master_data):
    """
    Calcula 4-TOC_CALC al vol a partir de 1-HPLC-SEQ i 2-TOC.

    Args:
        master_data: Dict retornat per llegir_masterfile_nou()

    Returns:
        DataFrame amb columns TOC_Row, Sample, Temps_Relatiu, Inj_Index
    """
    FLUSH_TIME_MIN = 3.637
    TOC_DATA_START_ROW = 8

    df_hplc = master_data.get("hplc_seq")
    df_toc = master_data.get("toc")

    if df_hplc is None or df_hplc.empty or df_toc is None or df_toc.empty:
        return None

    # Trobar columnes HPLC
    date_col = sample_col = sample_rep_col = None
    for col in df_hplc.columns:
        col_str = str(col).lower()
        if 'injection' in col_str and 'date' in col_str:
            date_col = col
        elif 'sample' in col_str and 'name' in col_str:
            sample_col = col
        elif 'sample_rep' in col_str:
            sample_rep_col = col

    if not date_col or not sample_col:
        return None

    # Llegir HPLC
    hplc_data = []
    for _, row in df_hplc.iterrows():
        date_val = row[date_col]
        sample_val = row[sample_rep_col] if sample_rep_col and pd.notna(row.get(sample_rep_col)) else row[sample_col]
        if pd.notna(date_val) and pd.notna(sample_val):
            hplc_data.append({'date': pd.to_datetime(date_val), 'sample': str(sample_val)})

    if not hplc_data:
        return None

    hplc_data = sorted(hplc_data, key=lambda x: x['date'])
    hplc_times = [h['date'] for h in hplc_data]
    hplc_samples = [h['sample'] for h in hplc_data]

    # Trobar columna de temps a TOC
    time_col = None
    for col in df_toc.columns:
        if 'date' in str(col).lower() and 'start' in str(col).lower():
            time_col = col
            break
    if time_col is None and len(df_toc.columns) > 3:
        time_col = df_toc.columns[3]

    if time_col is None:
        return None

    # Llegir timestamps TOC
    toc_data = []
    for idx, row in df_toc.iterrows():
        toc_row = TOC_DATA_START_ROW + idx  # Número de fila a l'Excel original
        toc_time_val = row[time_col]
        if pd.notna(toc_time_val):
            try:
                toc_time = pd.to_datetime(toc_time_val)
                toc_data.append((toc_row, toc_time))
            except:
                pass

    if not toc_data:
        return None

    # Calcular TOC_CALC
    calc_rows = []
    for toc_row, toc_time in toc_data:
        hora_hplc = toc_time + pd.Timedelta(minutes=FLUSH_TIME_MIN)
        inj_index = sum(1 for t in hplc_times if t <= hora_hplc)

        if 0 < inj_index <= len(hplc_samples):
            sample = hplc_samples[inj_index - 1]
            temps_rel = (hora_hplc - hplc_times[inj_index - 1]).total_seconds() / 60.0
        else:
            sample = ''
            temps_rel = None

        calc_rows.append({
            'TOC_Row': toc_row,
            'Sample': sample,
            'Temps_Relatiu (min)': round(temps_rel, 3) if temps_rel is not None else None,
            'Inj_Index': inj_index
        })

    return pd.DataFrame(calc_rows)


def _save_toc_calc_to_masterfile(masterfile_path, df_toc_calc):
    """
    Afegeix la fulla 4-TOC_CALC a un MasterFile existent.

    Args:
        masterfile_path: Ruta al MasterFile
        df_toc_calc: DataFrame amb les dades calculades
    """
    from openpyxl import load_workbook

    wb = load_workbook(masterfile_path)

    # Eliminar fulla existent si hi és
    if "4-TOC_CALC" in wb.sheetnames:
        del wb["4-TOC_CALC"]

    # Crear nova fulla al final
    ws = wb.create_sheet("4-TOC_CALC")

    # Escriure capçaleres
    headers = ["TOC_Row", "Sample", "Temps_Relatiu (min)", "Inj_Index"]
    for col, header in enumerate(headers, 1):
        ws.cell(row=1, column=col, value=header)

    # Escriure dades
    for row_idx, row in df_toc_calc.iterrows():
        ws.cell(row=row_idx + 2, column=1, value=row.get("TOC_Row"))
        ws.cell(row=row_idx + 2, column=2, value=row.get("Sample"))
        ws.cell(row=row_idx + 2, column=3, value=row.get("Temps_Relatiu (min)"))
        ws.cell(row=row_idx + 2, column=4, value=row.get("Inj_Index"))

    wb.save(masterfile_path)
    wb.close()


def build_sample_ranges_from_toc_calc(toc_calc_df, toc_df):
    """
    Construeix els rangs de files per cada mostra a partir de 4-TOC_CALC.

    Args:
        toc_calc_df: DataFrame de 4-TOC_CALC (TOC_Row, Sample, Temps_Relatiu, Inj_Index)
        toc_df: DataFrame de 2-TOC

    Returns:
        Dict amb Sample_Rep -> {row_start, row_end, inj_index, t_start}
    """
    if toc_calc_df is None or toc_calc_df.empty:
        return {}

    # Trobar columnes
    sample_col = None
    toc_row_col = None
    inj_idx_col = None

    for col in toc_calc_df.columns:
        col_lower = str(col).lower()
        if 'sample' in col_lower:
            sample_col = col
        elif 'toc_row' in col_lower or 'toc row' in col_lower:
            toc_row_col = col
        elif 'inj' in col_lower and 'index' in col_lower:
            inj_idx_col = col

    if sample_col is None:
        return {}

    # Agrupar per Sample
    sample_ranges = {}
    current_sample = None
    current_start = None

    for _, row in toc_calc_df.iterrows():
        sample = row.get(sample_col)
        toc_row = row.get(toc_row_col) if toc_row_col else None
        inj_idx = row.get(inj_idx_col) if inj_idx_col else None

        if pd.isna(sample) or str(sample).strip() == "":
            continue

        sample = str(sample).strip()

        if sample != current_sample:
            # Tancar rang anterior
            if current_sample is not None and current_start is not None:
                sample_ranges[current_sample]["row_end"] = prev_toc_row

            # Iniciar nou rang
            current_sample = sample
            current_start = toc_row
            sample_ranges[sample] = {
                "row_start": int(toc_row) if pd.notna(toc_row) else None,
                "row_end": None,
                "inj_index": int(inj_idx) if pd.notna(inj_idx) else None,
            }

        prev_toc_row = int(toc_row) if pd.notna(toc_row) else None

    # Tancar últim rang
    if current_sample is not None and current_start is not None:
        sample_ranges[current_sample]["row_end"] = prev_toc_row

    # Afegir t_start (timestamp de la primera fila) si tenim TOC
    if toc_df is not None and not toc_df.empty:
        time_col = None
        for col in toc_df.columns:
            if 'date' in str(col).lower() and 'start' in str(col).lower():
                time_col = col
                break
        if time_col is None and len(toc_df.columns) > 3:
            time_col = toc_df.columns[3]  # Columna D típicament

        if time_col:
            for sample, info in sample_ranges.items():
                row_start = info.get("row_start")
                if row_start is not None:
                    # row_start és 1-indexed des de la fila 8 (després de header)
                    idx = row_start - 8  # Ajustar a índex 0-based del DataFrame
                    if 0 <= idx < len(toc_df):
                        t_start = toc_df.iloc[idx][time_col]
                        info["t_start"] = t_start

    return sample_ranges


def extract_doc_from_masterfile(toc_df, row_start, row_end, t_start=None, detect_timeouts=True):
    """
    Extreu segment DOC del nou MasterFile.

    Args:
        toc_df: DataFrame de 2-TOC
        row_start: Fila inicial (1-indexed, relatiu a fila 8 del TOC original)
        row_end: Fila final
        t_start: Timestamp d'inici (opcional, per calcular temps relatiu)
        detect_timeouts: Si True, també detecta i retorna info de timeouts

    Returns:
        Si detect_timeouts=False: DataFrame amb columns "time (min)" i "DOC"
        Si detect_timeouts=True: tuple (DataFrame, timeout_info dict)
    """
    empty_timeout = {
        "n_timeouts": 0, "n_major_timeouts": 0, "timeouts": [],
        "dt_median_sec": 0, "dt_max_sec": 0, "zone_summary": {},
        "severity": "OK", "warning_message": ""
    }

    if toc_df is None or toc_df.empty:
        if detect_timeouts:
            return pd.DataFrame(), empty_timeout
        return pd.DataFrame()

    # Trobar columnes de temps i senyal
    time_col = None
    sig_col = None

    for col in toc_df.columns:
        col_str = str(col).lower()
        if 'date' in col_str and 'start' in col_str:
            time_col = col
        elif 'toc' in col_str and 'ppb' in col_str:
            sig_col = col
        elif 'tc' in col_str and 'ppb' in col_str and sig_col is None:
            sig_col = col

    # Fallback a posició si no es troben
    if time_col is None and len(toc_df.columns) > 3:
        time_col = toc_df.columns[3]
    if sig_col is None and len(toc_df.columns) > 5:
        sig_col = toc_df.columns[5]

    if time_col is None or sig_col is None:
        if detect_timeouts:
            return pd.DataFrame(), empty_timeout
        return pd.DataFrame()

    # Extreure segment (ajustar índexs)
    # row_start/row_end són 1-indexed des de fila 8 del Excel
    start_idx = max(int(row_start) - 8, 0)
    end_idx = max(int(row_end) - 8, start_idx)

    seg = toc_df.iloc[start_idx:end_idx + 1].copy()

    # Calcular temps relatiu
    seg_time = pd.to_datetime(seg[time_col], errors="coerce")
    if t_start is not None:
        t0 = pd.to_datetime(t_start, errors="coerce")
    else:
        t0 = seg_time.iloc[0] if len(seg_time) > 0 else None

    if t0 is not None:
        t_min = (seg_time - t0).dt.total_seconds() / 60.0
    else:
        t_min = pd.Series(range(len(seg))) * 0.1  # Fallback

    df_doc = pd.DataFrame({
        "time (min)": t_min.values,
        "DOC": pd.to_numeric(seg[sig_col], errors="coerce").values
    })
    df_doc = df_doc.dropna(subset=["time (min)", "DOC"])

    # Detectar timeouts si s'ha sol·licitat
    if detect_timeouts:
        timeout_info = detect_timeout(df_doc["time (min)"].values)
        return df_doc, timeout_info

    return df_doc


def llegir_master_direct(mestre):
    """Llegeix TOC i SEQ_DATA del mestre."""
    df_toc = pd.read_excel(mestre, sheet_name="2-TOC", header=6, engine="openpyxl")
    df_seq = pd.read_excel(mestre, sheet_name="4-SEQ_DATA", header=0, engine="openpyxl")
    return df_toc, df_seq


def extract_doc_from_master(df_toc, row_ini, row_fi, start_dt, detect_timeouts=True):
    """
    Extreu segment DOC del mestre (format antic).

    Args:
        df_toc: DataFrame de 2-TOC
        row_ini: Fila inicial
        row_fi: Fila final
        start_dt: Timestamp d'inici
        detect_timeouts: Si True, també detecta i retorna info de timeouts

    Returns:
        Si detect_timeouts=False: DataFrame amb columns "time (min)" i "DOC"
        Si detect_timeouts=True: tuple (DataFrame, timeout_info dict)
    """
    empty_timeout = {
        "n_timeouts": 0, "n_major_timeouts": 0, "timeouts": [],
        "dt_median_sec": 0, "dt_max_sec": 0, "zone_summary": {},
        "severity": "OK", "warning_message": ""
    }

    time_col = "Date Started" if "Date Started" in df_toc.columns else df_toc.columns[3]
    if "TOC(ppb)" in df_toc.columns:
        sig_col = "TOC(ppb)"
    elif "TC(ppb)" in df_toc.columns:
        sig_col = "TC(ppb)"
    else:
        sig_col = df_toc.columns[5]

    start_idx = max(int(row_ini) - 1, 0)
    end_idx = max(int(row_fi), start_idx)
    seg = df_toc.iloc[start_idx:end_idx].copy()

    seg_time = pd.to_datetime(seg[time_col], errors="coerce")
    t0 = pd.to_datetime(start_dt, errors="coerce")
    t_min = (seg_time - t0).dt.total_seconds() / 60.0

    df_doc = pd.DataFrame({"time (min)": t_min, "DOC": pd.to_numeric(seg[sig_col], errors="coerce")})
    df_doc = df_doc.dropna(subset=["time (min)", "DOC"])

    # Detectar timeouts si s'ha sol·licitat
    if detect_timeouts:
        timeout_info = detect_timeout(df_doc["time (min)"].values)
        return df_doc, timeout_info

    return df_doc


# =============================================================================
# FUNCIONS MATCHING DAD
# =============================================================================
def list_dad_files(folder_export3d, folder_csv):
    """Llista fitxers DAD disponibles."""
    out = []
    for folder in [folder_export3d, folder_csv]:
        if not folder or not os.path.isdir(folder):
            continue
        for ext in ("*.csv", "*.CSV", "*.txt", "*.TXT"):
            out.extend(glob.glob(os.path.join(folder, ext)))

    out2 = []
    for p in sorted(set(out)):
        bn = os.path.basename(p)
        if bn.startswith("~$"):
            continue
        if re.search(r"UIB1B", bn, flags=re.IGNORECASE):
            continue
        out2.append(p)
    return out2


def detect_dad_rep_style(dad_files):
    """Detecta estil de nomenclatura de rèpliques."""
    cnt_us = 0
    cnt_r = 0
    for p in dad_files or []:
        stem = os.path.splitext(os.path.basename(p))[0]
        if re.search(r"_[0-9]+$", stem):
            cnt_us += 1
        if re.search(r"[-_\s]R[0-9]+$", stem, flags=re.IGNORECASE):
            cnt_r += 1

    if cnt_us == 0 and cnt_r == 0:
        return ""
    return "UNDERSCORE" if cnt_us >= cnt_r else "R"


def dad_sample_rep_from_path(path):
    """Extreu mostra i rèplica del path DAD."""
    stem = os.path.splitext(os.path.basename(path))[0]

    m = re.match(r"^(.*)_([0-9]+)$", stem)
    if m:
        base = m.group(1).rstrip("-_ ")
        return base, m.group(2), "UNDERSCORE"

    m = re.match(r"^(.*?)[-_\s]R([0-9]+)$", stem, flags=re.IGNORECASE)
    if m:
        base = m.group(1).rstrip("-_ ")
        return base, m.group(2), "R"

    return stem, None, ""


def target_keys_from_desc(sample_desc):
    """Genera claus de matching per una mostra."""
    s = str(sample_desc or "").strip()
    keys = []
    if not s:
        return keys

    keys.append(normalize_key(s))

    tokens = [t for t in re.split(r"\s+", s) if t]
    if tokens:
        last = re.sub(r"[^A-Za-z0-9]+", "", tokens[-1])
        if last and not last.isdigit():
            keys.append(normalize_key(last))

    out = []
    seen = set()
    for k in keys:
        if k and k not in seen:
            out.append(k)
            seen.add(k)
    return out


def choose_best_candidate(target_key, rep, dad_files, used_files=None):
    """Tria el millor fitxer DAD per una mostra."""
    if not dad_files:
        return None, "NO_DATA"

    used = used_files if used_files is not None else set()
    rep_norm = normalize_rep(rep)
    keys = target_keys_from_desc(target_key)
    if not keys:
        return None, "NO_DATA"

    preferred_style = detect_dad_rep_style(dad_files)

    def iter_candidates(style_only):
        for p in dad_files:
            if p in used:
                continue
            s_part, r_part, style = dad_sample_rep_from_path(p)
            if style_only and preferred_style and style and style != preferred_style:
                continue
            yield p, s_part, r_part

    def exact_pass(style_only):
        out = []
        for p, s_part, r_part in iter_candidates(style_only):
            if not r_part:
                continue
            if rep_norm and normalize_rep(r_part) != rep_norm:
                continue
            sk = normalize_key(s_part)
            if sk in keys:
                out.append(p)
        return out

    exact = exact_pass(True) if preferred_style else exact_pass(False)
    if not exact and preferred_style:
        exact = exact_pass(False)

    if len(exact) == 1:
        return exact[0], "OK"
    if len(exact) > 1:
        return None, "NO_DATA"

    def name_only_pass(style_only):
        out = []
        for p, s_part, r_part in iter_candidates(style_only):
            sk = normalize_key(s_part)
            if sk in keys:
                out.append(p)
        return out

    name_only = name_only_pass(True) if preferred_style else name_only_pass(False)
    if len(name_only) == 0 and preferred_style:
        name_only = name_only_pass(False)

    if len(name_only) == 1:
        return name_only[0], "OK"

    return None, "NO_DATA"


# =============================================================================
# FUNCIONS PROCESSAMENT DAD
# =============================================================================
def process_dad(df_dad, config=None):
    """Processa DAD: extreu wavelengths i submostreig."""
    config = config or DEFAULT_CONSOLIDATE_CONFIG
    if df_dad is None or df_dad.empty:
        return pd.DataFrame()

    target_wls = config["target_wavelengths"]
    cols_to_keep = ["time (min)"]

    for wl in target_wls:
        wl_str = str(wl)
        if wl_str in df_dad.columns:
            cols_to_keep.append(wl_str)

    if len(cols_to_keep) == 1:
        return pd.DataFrame()

    df_filtered = df_dad[cols_to_keep].copy()

    subsample = config["dad_subsample"]
    indices = [0] + list(range(subsample, len(df_filtered), subsample))
    df_sub = df_filtered.iloc[indices].reset_index(drop=True)

    return df_sub


# =============================================================================
# FUNCIONS DETECCIÓ PICS
# =============================================================================
def find_peak_boundaries(t, y, peak_idx, baseline_level=None, threshold_pct=5.0, is_bp=False):
    """
    Troba els límits reals d'un pic per integració.
    """
    n = len(y)
    t = np.asarray(t)

    if baseline_level is None:
        if is_bp:
            n_edge = max(20, n // 5)
            baseline_level = np.median(y[-n_edge:])
        else:
            n_edge = max(10, n // 10)
            baseline_level = min(np.median(y[:n_edge]), np.median(y[-n_edge:]))

    peak_height = y[peak_idx]
    peak_amplitude = peak_height - baseline_level

    if peak_amplitude <= 0:
        return 0, n - 1

    threshold = baseline_level + (threshold_pct / 100.0) * peak_amplitude

    # Buscar límit esquerre
    left_idx = peak_idx
    for i in range(peak_idx - 1, -1, -1):
        if y[i] <= threshold:
            left_idx = i
            break
        if i > 0 and y[i] < y[i-1] and y[i] < y[i+1]:
            left_idx = i
            break
        left_idx = i

    # Buscar límit dret
    right_idx = peak_idx
    for i in range(peak_idx + 1, n):
        if y[i] <= threshold:
            right_idx = i
            break
        if i < n - 1 and y[i] < y[i-1] and y[i] < y[i+1]:
            right_idx = i
            break
        right_idx = i

    return left_idx, right_idx


def detect_main_peak(t, y, min_prominence_pct=5.0, is_bp=None):
    """Detecta el pic principal en el senyal amb límits d'integració correctes."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(t) < 10 or len(y) < 10:
        return {"valid": False}

    y_max = float(np.nanmax(y))
    if y_max < 1e-6:
        return {"valid": False}

    if is_bp is None:
        t_max_chromato = float(np.max(t))
        is_bp = t_max_chromato < 20

    min_prominence = y_max * (min_prominence_pct / 100.0)
    peaks, props = find_peaks(y, prominence=min_prominence, width=3)

    if len(peaks) == 0:
        return {"valid": False}

    idx = int(np.argmax(props["prominences"]))
    main_peak = int(peaks[idx])

    n = len(y)
    if is_bp:
        n_edge = max(20, n // 5)
        baseline_level = np.median(y[-n_edge:])
    else:
        n_edge = max(10, n // 10)
        baseline_level = min(np.median(y[:n_edge]), np.median(y[-n_edge:]))

    left_idx, right_idx = find_peak_boundaries(t, y, main_peak, baseline_level, threshold_pct=5.0, is_bp=is_bp)

    left_idx = max(0, left_idx)
    right_idx = min(len(y) - 1, right_idx)

    if right_idx > left_idx:
        area = float(trapezoid(y[left_idx:right_idx + 1], t[left_idx:right_idx + 1]))
    else:
        area = 0.0

    return {
        "valid": True,
        "area": area,
        "t_start": float(t[left_idx]),
        "t_max": float(t[main_peak]),
        "t_end": float(t[right_idx]),
        "height": float(y[main_peak]),
        "prominence": float(props["prominences"][idx]),
        "left_idx": left_idx,
        "right_idx": right_idx,
        "peak_idx": main_peak,
        "is_bp": is_bp,
        "baseline_level": float(baseline_level),
    }


def calcular_fraccions_temps(t, y, config=None):
    """
    Calcula àrees per fraccions de temps (integració parcial).

    Args:
        t: Array de temps (minuts)
        y: Array de senyal (mAU, ja amb baseline restada)
        config: Configuració amb time_fractions

    Returns:
        Dict amb àrees per fracció: {BioP, HS, BB, SB, LMW, total, *_pct}
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(t) < 2 or len(y) < 2:
        return {"total": 0.0}

    # Assegurar que y no té valors negatius (baseline ja restada)
    y_clean = np.maximum(y, 0)

    # Àrea total del cromatograma
    kpis = {"total": float(trapezoid(y_clean, t))}

    # Obtenir fraccions de la config
    if config is None:
        config = DEFAULT_CONSOLIDATE_CONFIG
    fractions = config.get("time_fractions", DEFAULT_CONSOLIDATE_CONFIG["time_fractions"])

    # Calcular àrea per cada fracció
    for nom, (t_ini, t_fi) in fractions.items():
        mask = (t >= t_ini) & (t < t_fi)
        if np.sum(mask) > 1:
            kpis[nom] = float(trapezoid(y_clean[mask], t[mask]))
        else:
            kpis[nom] = 0.0

    # Calcular percentatges
    if kpis["total"] > 0:
        for nom in fractions.keys():
            kpis[f"{nom}_pct"] = 100.0 * kpis[nom] / kpis["total"]
    else:
        for nom in fractions.keys():
            kpis[f"{nom}_pct"] = 0.0

    return kpis


def detectar_tmax_senyals(t_doc, y_doc, df_dad, config=None):
    """
    Detecta el temps de retenció (tmax) per DOC i cada longitud d'ona DAD.

    Args:
        t_doc: Array temps DOC
        y_doc: Array senyal DOC (net, amb baseline restada)
        df_dad: DataFrame DAD amb columnes 'time (min)' i wavelengths
        config: Configuració

    Returns:
        Dict amb tmax per cada senyal: {DOC: x, A220: y, A254: z, ...}
    """
    if config is None:
        config = DEFAULT_CONSOLIDATE_CONFIG
    target_wls = config.get('target_wavelengths', [220, 252, 254, 272, 290, 362])

    result = {"DOC": 0.0}
    for wl in target_wls:
        result[f"A{wl}"] = 0.0

    # tmax DOC
    if t_doc is not None and y_doc is not None and len(t_doc) > 10:
        t_doc = np.asarray(t_doc).flatten()
        y_doc = np.asarray(y_doc).flatten()
        idx_max = np.argmax(y_doc)
        result["DOC"] = float(t_doc[idx_max])

    # tmax per cada wavelength DAD
    if df_dad is not None and not df_dad.empty and 'time (min)' in df_dad.columns:
        t_dad = pd.to_numeric(df_dad['time (min)'], errors='coerce').to_numpy()

        for wl in target_wls:
            wl_str = str(wl)
            if wl_str in df_dad.columns:
                y_wl = pd.to_numeric(df_dad[wl_str], errors='coerce').to_numpy()
                if len(y_wl) > 10 and not np.all(np.isnan(y_wl)):
                    # Restar baseline (percentil 5)
                    baseline = np.nanpercentile(y_wl, 5)
                    y_wl_net = y_wl - baseline
                    idx_max = np.nanargmax(y_wl_net)
                    result[f"A{wl}"] = float(t_dad[idx_max])

    return result


def calcular_arees_fraccions_complet(t_doc, y_doc, df_dad, config=None):
    """
    Calcula àrees per fraccions de temps per DOC i totes les wavelengths DAD.

    Args:
        t_doc: Array temps DOC
        y_doc: Array senyal DOC (net)
        df_dad: DataFrame DAD
        config: Configuració

    Returns:
        Dict amb estructura:
        {
            "DOC": {BioP: x, HS: y, ..., total: z},
            "A220": {BioP: x, HS: y, ..., total: z},
            ...
        }
    """
    if config is None:
        config = DEFAULT_CONSOLIDATE_CONFIG
    target_wls = config.get('target_wavelengths', [220, 252, 254, 272, 290, 362])

    result = {}

    # Fraccions DOC
    if t_doc is not None and y_doc is not None and len(t_doc) > 10:
        result["DOC"] = calcular_fraccions_temps(t_doc, y_doc, config)
    else:
        result["DOC"] = {"total": 0.0}

    # Fraccions per cada wavelength DAD
    if df_dad is not None and not df_dad.empty and 'time (min)' in df_dad.columns:
        t_dad = pd.to_numeric(df_dad['time (min)'], errors='coerce').to_numpy()

        for wl in target_wls:
            wl_str = str(wl)
            if wl_str in df_dad.columns:
                y_wl = pd.to_numeric(df_dad[wl_str], errors='coerce').to_numpy()
                if len(y_wl) > 10 and not np.all(np.isnan(y_wl)):
                    # Restar baseline
                    baseline = np.nanpercentile(y_wl, 5)
                    y_wl_net = np.maximum(y_wl - baseline, 0)
                    result[f"A{wl}"] = calcular_fraccions_temps(t_dad, y_wl_net, config)
                else:
                    result[f"A{wl}"] = {"total": 0.0}
            else:
                result[f"A{wl}"] = {"total": 0.0}
    else:
        for wl in target_wls:
            result[f"A{wl}"] = {"total": 0.0}

    return result


def analyze_sample_areas(t_doc, y_doc, df_dad, peak_info, config=None):
    """
    Analitza una mostra i calcula totes les àrees (DOC + DAD wavelengths).
    """
    target_wls = config.get('target_wavelengths', [220, 252, 254, 272, 290, 362]) if config else [220, 252, 254, 272, 290, 362]

    result = {
        'doc_area': 0.0,
        'doc_t_retention': 0.0,
        'doc_t_start': 0.0,
        'doc_t_end': 0.0,
        'dad_wavelengths': {},
        'valid': False,
    }
    for wl in target_wls:
        result[f'a{wl}_area'] = 0.0

    if t_doc is None or y_doc is None or len(t_doc) < 10:
        return result

    t_doc = np.asarray(t_doc).flatten()
    y_doc = np.asarray(y_doc).flatten()

    if peak_info and peak_info.get('valid'):
        result['doc_area'] = peak_info.get('area', 0.0)
        result['doc_t_retention'] = peak_info.get('t_max', 0.0)
        result['doc_t_start'] = peak_info.get('t_start', 0.0)
        result['doc_t_end'] = peak_info.get('t_end', 0.0)
        result['valid'] = True

        t_start = result['doc_t_start']
        t_end = result['doc_t_end']

        if df_dad is not None and not df_dad.empty and 'time (min)' in df_dad.columns:
            t_dad = pd.to_numeric(df_dad['time (min)'], errors='coerce').to_numpy()

            if t_start > 0 and t_end > t_start:
                dad_left_idx = int(np.searchsorted(t_dad, t_start))
                dad_right_idx = int(np.searchsorted(t_dad, t_end))

                dad_left_idx = max(0, min(dad_left_idx, len(t_dad) - 1))
                dad_right_idx = max(0, min(dad_right_idx, len(t_dad) - 1))

                if dad_right_idx > dad_left_idx:
                    for col in df_dad.columns:
                        if col == 'time (min)':
                            continue
                        try:
                            wl_str = str(col)
                            y_wl = pd.to_numeric(df_dad[col], errors='coerce').to_numpy()
                            if len(y_wl) > dad_right_idx:
                                area_wl = float(trapezoid(
                                    y_wl[dad_left_idx:dad_right_idx+1],
                                    t_dad[dad_left_idx:dad_right_idx+1]
                                ))
                                try:
                                    wl_int = int(wl_str)
                                    if wl_int in target_wls:
                                        result[f'a{wl_int}_area'] = area_wl
                                except ValueError:
                                    pass
                                result['dad_wavelengths'][wl_str] = area_wl
                        except:
                            continue

    return result


# =============================================================================
# FUNCIÓ MATCHING UIB -> MESTRE
# =============================================================================
def find_master_row_for_sample(df_seq, mostra_uib, rep_uib):
    """
    Troba la fila del mestre que correspon a una mostra UIB.

    Args:
        df_seq: DataFrame del sheet 4-SEQ_DATA del mestre
        mostra_uib: Nom de la mostra extret del fitxer UIB
        rep_uib: Rèplica extreta del fitxer UIB

    Returns:
        Series amb la fila trobada, o None si no es troba
    """
    mostra_key = normalize_key(mostra_uib)
    rep_norm = normalize_rep(rep_uib)

    for _, row in df_seq.iterrows():
        sample_desc = str(row.get("SAMPLE_DESC", "")).strip()
        if skip_sample_direct(sample_desc):
            continue

        base_name, rep_explicit = split_sample_rep(sample_desc)
        base_key = normalize_key(base_name)

        # Matching per nom
        if base_key == mostra_key or mostra_key in base_key or base_key in mostra_key:
            # Si tenim rèplica explícita, verificar que coincideix
            if rep_explicit and rep_norm:
                if normalize_rep(rep_explicit) == rep_norm:
                    return row
            else:
                # Sense rèplica explícita, acceptar si el nom coincideix
                return row

    return None


def build_master_sample_index(df_seq):
    """
    Construeix un índex de mostres del mestre per facilitar el matching.

    Args:
        df_seq: DataFrame del sheet 4-SEQ_DATA del mestre

    Returns:
        Dict amb clau normalitzada -> llista de (idx, row, base_name, rep)
    """
    index = {}
    rep_counter = {}

    for idx, row in df_seq.iterrows():
        sample_desc = str(row.get("SAMPLE_DESC", "")).strip()
        if skip_sample_direct(sample_desc):
            continue

        base_name, rep_explicit = split_sample_rep(sample_desc)
        base_key = normalize_key(base_name)

        if rep_explicit is None:
            rep_counter[base_key] = rep_counter.get(base_key, 0) + 1
            rep = str(rep_counter[base_key])
        else:
            rep = str(rep_explicit)

        if base_key not in index:
            index[base_key] = []
        index[base_key].append({
            "idx": idx,
            "row": row,
            "base_name": base_name,
            "rep": rep,
            "sample_desc": sample_desc
        })

    return index


def match_uib_to_master(mostra_uib, rep_uib, master_index):
    """
    Busca la fila del mestre que correspon a una mostra UIB usant l'índex.

    Args:
        mostra_uib: Nom de la mostra extret del fitxer UIB
        rep_uib: Rèplica extreta del fitxer UIB
        master_index: Índex construït amb build_master_sample_index()

    Returns:
        Dict amb info de la fila trobada, o None si no es troba
    """
    mostra_key = normalize_key(mostra_uib)
    rep_norm = normalize_rep(rep_uib)

    # Buscar coincidència exacta
    if mostra_key in master_index:
        for entry in master_index[mostra_key]:
            if entry["rep"] == rep_norm:
                return entry
        # Si no hi ha coincidència de rèplica, retornar la primera
        if master_index[mostra_key]:
            return master_index[mostra_key][0]

    # Buscar coincidència parcial
    for key, entries in master_index.items():
        if mostra_key in key or key in mostra_key:
            for entry in entries:
                if entry["rep"] == rep_norm:
                    return entry
            if entries:
                return entries[0]

    return None


# =============================================================================
# CÀLCUL SNR I BASELINE NOISE
# =============================================================================
def calculate_snr_info(y_doc_net, peak_info, y_doc_uib=None,
                       t_min=None, method="column", timeout_positions=None):
    """
    Calcula SNR, LOD, LOQ i baseline noise per DOC Direct i UIB.

    Usa finestres temporals específiques per evitar regions amb timeouts,
    obtenint estimacions de soroll consistents entre rèpliques.

    Args:
        y_doc_net: Senyal DOC net (Direct)
        peak_info: Diccionari amb info del pic (height)
        y_doc_uib: Senyal DOC UIB (opcional, per DUAL)
        t_min: Array de temps en minuts (opcional, per càlcul windowed)
        method: "column" o "bp" - determina finestres de baseline
        timeout_positions: Llista de posicions temporals (min) dels timeouts

    Returns:
        dict amb:
            - snr_direct: SNR del senyal Direct
            - baseline_noise_direct: Desviació estàndard baseline Direct (mAU)
            - lod_direct: Limit of Detection = 3 × noise (mAU)
            - loq_direct: Limit of Quantification = 10 × noise (mAU)
            - baseline_window_direct: Finestra usada per calcular baseline
            - snr_uib, baseline_noise_uib, lod_uib, loq_uib (si DUAL)
    """
    result = {}

    # Determinar si podem usar el mètode windowed
    use_windowed = (t_min is not None and len(t_min) > 10)

    # Direct
    if y_doc_net is not None and len(y_doc_net) > 10:
        if use_windowed:
            bl_stats = baseline_stats_windowed(
                t_min, y_doc_net,
                method=method,
                timeout_positions=timeout_positions
            )
            result["baseline_window_direct"] = bl_stats.get("window_used", "unknown")
        else:
            bl_stats = baseline_stats(y_doc_net)
            result["baseline_window_direct"] = "percentile"

        noise_direct = bl_stats.get("std", 0.0)
        result["baseline_noise_direct"] = noise_direct
        result["lod_direct"] = 3.0 * noise_direct
        result["loq_direct"] = 10.0 * noise_direct

        if peak_info and peak_info.get("valid") and peak_info.get("height", 0) > 0:
            height = peak_info["height"]
            if noise_direct > 0:
                result["snr_direct"] = height / noise_direct
            else:
                result["snr_direct"] = calc_snr(y_doc_net, height)

    # UIB
    if y_doc_uib is not None and len(y_doc_uib) > 10:
        if use_windowed:
            bl_stats_uib = baseline_stats_windowed(
                t_min, y_doc_uib,
                method=method,
                timeout_positions=timeout_positions
            )
            result["baseline_window_uib"] = bl_stats_uib.get("window_used", "unknown")
        else:
            bl_stats_uib = baseline_stats(y_doc_uib)
            result["baseline_window_uib"] = "percentile"

        noise_uib = bl_stats_uib.get("std", 0.0)
        result["baseline_noise_uib"] = noise_uib
        result["lod_uib"] = 3.0 * noise_uib
        result["loq_uib"] = 10.0 * noise_uib

        if peak_info and peak_info.get("valid"):
            # Usar el màxim de UIB com a alçada del pic
            height_uib = float(np.max(y_doc_uib))
            if noise_uib > 0:
                result["snr_uib"] = height_uib / noise_uib
            else:
                result["snr_uib"] = calc_snr(y_doc_uib, height_uib)

    return result


# =============================================================================
# FUNCIONS RESUM CONSOLIDACIÓ
# =============================================================================

def _collect_sample_stats(mostra, rep, timeout_info, snr_info, peak_info, doc_mode, file_info=None, doc_direct_status=None):
    """
    Recull estadístiques d'una mostra per al resum de consolidació.

    Args:
        mostra: Nom de la mostra
        rep: Número de rèplica
        timeout_info: Info de timeouts TOC
        snr_info: Info de SNR i baseline
        peak_info: Info del pic principal
        doc_mode: Mode DOC (DUAL, DIRECT, UIB)
        file_info: Dict amb info dels fitxers (opcional):
            - file_dad: Nom fitxer DAD/Direct
            - file_uib: Nom fitxer UIB
            - row_start: Fila inicial
            - row_end: Fila final
            - npts: Nombre de punts
        doc_direct_status: Dict amb estat del DOC Direct (opcional):
            - status: "OK", "FALTA_HPLC_SEQ", "SENSE_MATCH", "SOBREESCRIT"
            - message: Missatge explicatiu

    Returns:
        dict amb estadístiques de la mostra
    """
    stats = {
        "name": f"{mostra}_R{rep}",
        "mostra": mostra,
        "replica": rep,
        "doc_mode": doc_mode,
        "peak_valid": peak_info.get("valid", False) if peak_info else False,
    }

    # File info (per GUI eficient)
    if file_info:
        stats["file_dad"] = file_info.get("file_dad", "")
        stats["file_uib"] = file_info.get("file_uib", "")
        stats["row_start"] = file_info.get("row_start", 0)
        stats["row_end"] = file_info.get("row_end", 0)
        stats["npts"] = file_info.get("npts", 0)
        stats["match_confidence"] = file_info.get("match_confidence", 100.0)
        stats["match_type"] = file_info.get("match_type", "exact")

    # DOC Direct status (per GUI - mostrar errors)
    if doc_direct_status:
        stats["doc_direct_status"] = doc_direct_status.get("status", "OK")
        stats["doc_direct_message"] = doc_direct_status.get("message", "")
    else:
        # Determinar status automàticament basant-se en file_info
        if file_info and file_info.get("row_start") and file_info.get("row_end"):
            stats["doc_direct_status"] = "OK"
            stats["doc_direct_message"] = ""
        else:
            stats["doc_direct_status"] = "SENSE_DADES"
            stats["doc_direct_message"] = "Sense dades DOC Direct"

    # Timeout info
    if timeout_info:
        stats["timeout_detected"] = timeout_info.get("n_timeouts", 0) > 0
        stats["timeout_count"] = timeout_info.get("n_timeouts", 0)
        stats["timeout_severity"] = timeout_info.get("severity", "OK")
        # Zones afectades
        zone_summary = timeout_info.get("zone_summary", {})
        if zone_summary:
            stats["timeout_zones"] = [z for z, n in zone_summary.items() if n > 0]
    else:
        stats["timeout_detected"] = False
        stats["timeout_count"] = 0
        stats["timeout_severity"] = "OK"
        stats["timeout_zones"] = []

    # SNR info
    if snr_info:
        stats["snr_direct"] = snr_info.get("snr_direct")
        stats["snr_uib"] = snr_info.get("snr_uib")
        stats["baseline_noise_direct"] = snr_info.get("baseline_noise_direct")
        stats["baseline_noise_uib"] = snr_info.get("baseline_noise_uib")
        stats["lod_direct"] = snr_info.get("lod_direct")
        stats["lod_uib"] = snr_info.get("lod_uib")
        # Finestra de baseline usada (per diagnòstic)
        stats["baseline_window_direct"] = snr_info.get("baseline_window_direct")
        stats["baseline_window_uib"] = snr_info.get("baseline_window_uib")

    return stats


def generate_consolidation_summary(result, sample_stats, timing_stats=None):
    """
    Genera un resum JSON de la consolidació.

    Args:
        result: Diccionari resultat de consolidate_sequence
        sample_stats: Llista d'estadístiques per mostra
        timing_stats: Dict amb estadístiques de timing (opcional)

    Returns:
        dict amb el resum complet
    """
    from hpsec_utils import is_khp

    # Comptar tipus de mostres
    n_khp = sum(1 for s in sample_stats if is_khp(s.get("mostra", "")))
    n_control = sum(1 for s in sample_stats
                    if any(ctrl in s.get("mostra", "").upper()
                           for ctrl in ["MQ", "NAOH", "BLANK", "CONTROL"]))
    n_samples = len(sample_stats) - n_khp - n_control

    # Estadístiques de timeouts
    timeout_counts = {"OK": 0, "INFO": 0, "WARNING": 0, "CRITICAL": 0}
    critical_samples = []
    for s in sample_stats:
        sev = s.get("timeout_severity", "OK")
        timeout_counts[sev] = timeout_counts.get(sev, 0) + 1
        if sev == "CRITICAL":
            critical_samples.append(s.get("name", ""))

    samples_with_timeout = sum(1 for s in sample_stats if s.get("timeout_detected", False))

    # Estadístiques SNR (filtrar valors extrems: SNR > 10000 o noise < 0.01 indica valors no fiables)
    MAX_SNR = 10000.0
    MIN_NOISE = 0.01
    snr_direct_vals = [s.get("snr_direct") for s in sample_stats
                       if s.get("snr_direct") is not None and s.get("snr_direct") < MAX_SNR]
    snr_uib_vals = [s.get("snr_uib") for s in sample_stats
                    if s.get("snr_uib") is not None and s.get("snr_uib") < MAX_SNR]
    noise_direct_vals = [s.get("baseline_noise_direct") for s in sample_stats
                         if s.get("baseline_noise_direct") is not None and s.get("baseline_noise_direct") > MIN_NOISE]
    noise_uib_vals = [s.get("baseline_noise_uib") for s in sample_stats
                      if s.get("baseline_noise_uib") is not None and s.get("baseline_noise_uib") > MIN_NOISE]

    def stats_summary(vals):
        if not vals:
            return None
        return {
            "min": round(min(vals), 2),
            "median": round(float(np.median(vals)), 2),
            "max": round(max(vals), 2),
            "mean": round(float(np.mean(vals)), 2),
        }

    # LOD (usar mediana del soroll)
    lod_direct = round(3.0 * float(np.median(noise_direct_vals)), 3) if noise_direct_vals else None
    lod_uib = round(3.0 * float(np.median(noise_uib_vals)), 3) if noise_uib_vals else None

    # Pics vàlids
    n_peak_valid = sum(1 for s in sample_stats if s.get("peak_valid", False))
    peak_valid_pct = round(100.0 * n_peak_valid / len(sample_stats), 1) if sample_stats else 0

    # === QUALITY CHECK: Detectar files duplicades i mostres duplicades ===
    row_usage = {}  # {(row_start, row_end): [sample_names]}
    sample_count = {}  # {sample_name: count}
    duplicate_rows = []
    duplicate_samples = []

    for s in sample_stats:
        row_start = s.get("row_start") or 0
        row_end = s.get("row_end") or 0
        sample_name = s.get("name", s.get("mostra", "unknown"))

        # Comptar mostres duplicades
        sample_count[sample_name] = sample_count.get(sample_name, 0) + 1

        # Files duplicades (només si tenim files vàlides)
        if row_start and row_end and row_start > 0 and row_end > 0 and row_start != row_end:
            key = (row_start, row_end)
            if key not in row_usage:
                row_usage[key] = []
            row_usage[key].append(sample_name)

    # Detectar mostres duplicades
    for sample_name, count in sample_count.items():
        if count > 1:
            duplicate_samples.append({
                "sample": sample_name,
                "count": count
            })

    # Detectar files compartides entre mostres DIFERENTS
    for (row_start, row_end), samples in row_usage.items():
        unique_names = set(samples)
        if len(unique_names) > 1:
            duplicate_rows.append({
                "rows": f"{row_start}-{row_end}",
                "samples": list(unique_names),
                "count": len(unique_names)
            })

    duplicate_rows.sort(key=lambda x: x["count"], reverse=True)
    duplicate_samples.sort(key=lambda x: x["count"], reverse=True)

    # Quality issues array
    quality_issues = []

    # Mostres duplicades (WARNING - cal revisar blocs B1/B2)
    for dup in duplicate_samples:
        quality_issues.append({
            "type": "DUPLICATE_SAMPLE",
            "severity": "WARNING",
            "message": f"Mostra '{dup['sample']}' apareix {dup['count']} cops (revisar blocs B1/B2)",
            "sample": dup["sample"],
            "count": dup["count"]
        })

    # Files duplicades (ERROR - problema de MasterFile)
    for dup in duplicate_rows:
        quality_issues.append({
            "type": "DUPLICATE_ROWS",
            "severity": "ERROR",
            "message": f"Files {dup['rows']} usades per mostres diferents: {', '.join(dup['samples'])}",
            "samples": dup["samples"],
            "rows": dup["rows"]
        })

    # Issues de validació 1-HPLC-SEQ (del resultat de consolidació)
    hplc_validation = result.get("hplc_seq_validation", {})
    for issue in hplc_validation.get("issues", []):
        quality_issues.append(issue)

    # Construir resum
    summary = {
        "meta": {
            "seq": result.get("seq", ""),
            "date": result.get("date", ""),
            "method": "BP" if result.get("bp", False) else "COLUMN",
            "mode": result.get("mode", ""),
            "generated_at": datetime.now().isoformat(),
            "script_version": __version__,
        },
        "counts": {
            "total_samples": len(sample_stats),
            "total_files": result.get("processed_count", 0),
            "khp_samples": n_khp,
            "control_samples": n_control,
            "regular_samples": n_samples,
        },
        "alignment": result.get("alignment", {}),
        "timeouts": {
            "samples_with_timeout": samples_with_timeout,
            "severity_counts": timeout_counts,
            "critical_samples": critical_samples[:10],  # Màxim 10
        },
        "quality": {
            "snr_direct": stats_summary(snr_direct_vals),
            "snr_uib": stats_summary(snr_uib_vals),
            "lod_direct_mau": lod_direct,
            "lod_uib_mau": lod_uib,
            "peak_valid_count": n_peak_valid,
            "peak_valid_pct": peak_valid_pct,
            "duplicate_rows": duplicate_rows,
            "duplicate_row_count": len(duplicate_rows),
            "duplicate_samples": duplicate_samples,
            "duplicate_sample_count": len(duplicate_samples),
        },
        "quality_issues": quality_issues,
        "hplc_seq_validation": hplc_validation if hplc_validation else {},
        "file_check": result.get("file_check", {}),
        "warnings": result.get("warnings", []) + [
            f"AVÍS DUPLICAT: Mostra '{dup['sample']}' apareix {dup['count']} cops (revisar blocs B1/B2)"
            for dup in duplicate_samples
        ],
        "errors": [e for e in result.get("errors", []) if not e.startswith("WARN:")] + [
            f"ERROR DUPLICAT: Files {dup['rows']} usades per {len(dup['samples'])} mostres: {', '.join(dup['samples'])}"
            for dup in duplicate_rows
        ],
        "samples": sample_stats,
    }

    # Afegir timing stats si disponibles (per planificació seqüències)
    if timing_stats:
        summary["timing"] = timing_stats

    return summary


def save_consolidation_summary(seq_path, summary):
    """
    Guarda el resum de consolidació a CHECK/consolidation.json.

    Args:
        seq_path: Ruta de la seqüència
        summary: Diccionari amb el resum

    Returns:
        Path del fitxer guardat o None si error
    """
    check_folder = os.path.join(seq_path, "CHECK")
    os.makedirs(check_folder, exist_ok=True)

    json_path = os.path.join(check_folder, "consolidation.json")

    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        return json_path
    except Exception as e:
        print(f"Error guardant consolidation.json: {e}")
        return None


# =============================================================================
# FUNCIÓ ESCRIPTURA EXCEL CONSOLIDAT (DUAL PROTOCOL)
# =============================================================================
def write_consolidated_excel(out_path, mostra, rep, seq_out, date_master,
                             method, doc_mode, fitxer_doc, fitxer_dad,
                             st_doc, st_dad, t_doc, y_doc_raw, y_doc_net,
                             baseline, df_dad, peak_info, sample_analysis=None,
                             master_file=None, row_start=None, row_end=None,
                             # Nous paràmetres per dual protocol
                             y_doc_uib=None, y_doc_uib_raw=None, baseline_uib=None,
                             fitxer_doc_uib=None, st_doc_uib=None,
                             # Paràmetres de processament
                             shift_uib=None, shift_direct=None,
                             smoothing_applied=True,
                             # Info del MasterFile 0-INFO
                             master_info=None,
                             # Detecció de timeouts TOC
                             timeout_info=None,
                             # SNR i baseline noise
                             snr_info=None):
    """
    Escriu fitxer Excel consolidat amb àrees per fraccions de temps.

    Suporta dual protocol: DOC (Direct) + DOC_UIB quan ambdós estan disponibles.
    Sempre guarda dades RAW i documenta el processament aplicat.

    Estructura fulls:
      ID: Identificació, fitxers, estat, processament (sense àrees)
      TMAX: Temps de retenció per DOC i cada wavelength DAD
      AREAS: Àrees per fraccions de temps (BioP, HS, BB, SB, LMW, total)
      DOC: Cromatograma DOC (net, raw, baseline)
      DAD: Cromatograma DAD (totes les wavelengths)

    Fraccions de temps:
      - BioP: 0-18 min (Biopolímers)
      - HS: 18-23 min (Àcids húmics)
      - BB: 23-30 min (Building Blocks)
      - SB: 30-40 min (Small Building blocks)
      - LMW: 40-70 min (Low Molecular Weight)
    """
    sample_analysis = sample_analysis or {}
    master_info = master_info or {}
    is_dual = y_doc_uib is not None and len(y_doc_uib) > 0

    # Truncar cromatogrames a 70 min per consistència
    max_time = DEFAULT_CONSOLIDATE_CONFIG.get("max_time_min", 70.0)
    if t_doc is not None and len(t_doc) > 0:
        mask = np.asarray(t_doc) <= max_time
        t_doc = np.asarray(t_doc)[mask]
        y_doc_raw = np.asarray(y_doc_raw)[mask] if y_doc_raw is not None else None
        y_doc_net = np.asarray(y_doc_net)[mask] if y_doc_net is not None else None
        baseline = np.asarray(baseline)[mask] if baseline is not None else None
        if is_dual:
            y_doc_uib = np.asarray(y_doc_uib)[mask] if y_doc_uib is not None else None
            y_doc_uib_raw = np.asarray(y_doc_uib_raw)[mask] if y_doc_uib_raw is not None else None
            baseline_uib = np.asarray(baseline_uib)[mask] if baseline_uib is not None else None

    # Truncar DAD també
    if df_dad is not None and not df_dad.empty and "time (min)" in df_dad.columns:
        df_dad = df_dad[df_dad["time (min)"] <= max_time].copy()

    # Calcular valors de baseline per documentar
    baseline_direct_val = float(np.mean(baseline)) if baseline is not None and len(baseline) > 0 else 0.0
    baseline_uib_val = float(np.mean(baseline_uib)) if baseline_uib is not None and len(baseline_uib) > 0 else 0.0

    # === 1. IDENTIFICACIÓ ===
    inj_vol = master_info.get("Inj_Volume (uL)", "")
    uib_range = master_info.get("UIB_range")
    # Convertir NaN/None a valors per defecte
    # Nota: "None" com a string es converteix a NaN per pandas, usem "-"
    if pd.isna(inj_vol) or inj_vol is None:
        inj_vol = ""
    if pd.isna(uib_range) or uib_range is None:
        uib_range = "-"
    else:
        uib_range = str(uib_range)

    id_rows = [
        ("Script_Version", f"hpsec_consolidate v{__version__}"),
        ("Consolidation_Date", datetime.now().strftime("%Y-%m-%d %H:%M")),
        ("---", "---"),
        ("Sample", mostra),
        ("Replica", rep),
        ("SEQ", seq_out),
        ("Method", method),
        ("Date", date_master),
        ("Inj_Volume_uL", inj_vol),
        ("UIB_range", uib_range),
    ]

    # === 2. FITXERS ===
    id_rows.append(("File_MasterFile", master_file or ""))
    id_rows.append(("File_DAD", fitxer_dad or ""))
    if is_dual and fitxer_doc_uib:
        id_rows.append(("File_DOC_UIB", fitxer_doc_uib))

    # === 3. ESTAT ===
    id_rows.extend([
        ("DOC_Mode", doc_mode),
        ("Status_DOC", st_doc),
        ("Status_DAD", st_dad),
        ("Peak_Valid", peak_info.get("valid", False)),
    ])

    # === 4. MESURES DOC (simplificat - àrees i tmax van a fulls separats) ===
    id_rows.append(("DOC_N_Points", len(t_doc)))
    if row_start is not None:
        id_rows.append(("DOC_Row_Start", row_start))
    if row_end is not None:
        id_rows.append(("DOC_Row_End", row_end))

    # === 5. PROCESSAMENT DOC ===
    id_rows.extend([
        ("DOC_Baseline_Method", "percentile"),
        ("DOC_Baseline_mAU", round(baseline_direct_val, 3)),
        ("DOC_Smoothing", "YES" if smoothing_applied else "NO"),
    ])
    if shift_direct is not None:
        id_rows.append(("DOC_Shift_sec", round(shift_direct * 60, 2)))

    # === 6. UIB (només si DUAL) ===
    if is_dual:
        id_rows.append(("UIB_Baseline_mAU", round(baseline_uib_val, 3)))
        if shift_uib is not None:
            id_rows.append(("UIB_Shift_sec", round(shift_uib * 60, 2)))
        id_rows.append(("UIB_Status", st_doc_uib or "OK"))

    # === 7. DAD N_Points (àrees van al full AREAS) ===
    if not df_dad.empty:
        id_rows.append(("DAD_N_Points", len(df_dad)))

    # === 8. TIMEOUT DETECTION (TOC syringe reload) ===
    if timeout_info and timeout_info.get("n_timeouts", 0) > 0:
        id_rows.append(("---", "---"))  # Separador visual
        id_rows.append(("TOC_Timeout_Detected", "YES"))
        id_rows.append(("TOC_Timeout_Count", str(timeout_info.get("n_timeouts", 0))))  # Forçar string per evitar conversió a bool
        id_rows.append(("TOC_Timeout_Severity", timeout_info.get("severity", "OK")))

        # Detalls dels timeouts
        timeouts = timeout_info.get("timeouts", [])
        for i, to in enumerate(timeouts[:3]):  # Màxim 3 timeouts detallats
            zone = to.get("zone", "?")
            t_min = to.get("t_start_min", 0)
            dur = to.get("duration_sec", 0)
            sev = to.get("severity", "INFO")
            id_rows.append((f"TOC_Timeout_{i+1}", f"{t_min:.1f} min ({dur:.0f}s) - {zone} [{sev}]"))

        # Zones afectades (resum)
        zone_summary = timeout_info.get("zone_summary", {})
        if zone_summary:
            affected_zones = [f"{z}:{n}" for z, n in zone_summary.items() if n > 0]
            id_rows.append(("TOC_Zones_Affected", ", ".join(affected_zones)))

        # Cadència de mesures
        id_rows.append(("TOC_Dt_Median_sec", timeout_info.get("dt_median_sec", 0)))
        id_rows.append(("TOC_Dt_Max_sec", timeout_info.get("dt_max_sec", 0)))
    else:
        id_rows.append(("TOC_Timeout_Detected", "NO"))

    # === 9. SNR, LOD, LOQ I BASELINE NOISE ===
    if snr_info:
        id_rows.append(("---", "---"))  # Separador visual
        if snr_info.get("snr_direct") is not None:
            id_rows.append(("SNR_Direct", round(snr_info["snr_direct"], 1)))
        if snr_info.get("baseline_noise_direct") is not None:
            id_rows.append(("Baseline_Noise_Direct_mAU", round(snr_info["baseline_noise_direct"], 3)))
        if snr_info.get("lod_direct") is not None:
            id_rows.append(("LOD_Direct_mAU", round(snr_info["lod_direct"], 3)))
        if snr_info.get("loq_direct") is not None:
            id_rows.append(("LOQ_Direct_mAU", round(snr_info["loq_direct"], 3)))
        if is_dual and snr_info.get("snr_uib") is not None:
            id_rows.append(("SNR_UIB", round(snr_info["snr_uib"], 1)))
        if is_dual and snr_info.get("baseline_noise_uib") is not None:
            id_rows.append(("Baseline_Noise_UIB_mAU", round(snr_info["baseline_noise_uib"], 3)))
        if is_dual and snr_info.get("lod_uib") is not None:
            id_rows.append(("LOD_UIB_mAU", round(snr_info["lod_uib"], 3)))
        if is_dual and snr_info.get("loq_uib") is not None:
            id_rows.append(("LOQ_UIB_mAU", round(snr_info["loq_uib"], 3)))

    df_id = pd.DataFrame(id_rows, columns=["Field", "Value"])

    # === TMAX SHEET: retention times for DOC and DAD ===
    tmax_data = detectar_tmax_senyals(t_doc, y_doc_net, df_dad)
    tmax_rows = [("Signal", "tmax (min)")]
    for signal, tmax_val in tmax_data.items():
        tmax_rows.append((signal, round(tmax_val, 3) if tmax_val > 0 else "-"))
    df_tmax = pd.DataFrame(tmax_rows[1:], columns=tmax_rows[0])

    # === AREAS SHEET: areas by time fractions ===
    fraccions_data = calcular_arees_fraccions_complet(t_doc, y_doc_net, df_dad)
    fractions_config = DEFAULT_CONSOLIDATE_CONFIG.get("time_fractions", {})
    fraction_names = list(fractions_config.keys()) + ["total"]

    # Build areas table
    areas_rows = []
    # Header: Fraction, Range, DOC, A220, A254, ...
    header = ["Fraction", "Range (min)", "DOC"]
    target_wls = DEFAULT_CONSOLIDATE_CONFIG.get("target_wavelengths", [220, 252, 254, 272, 290, 362])
    for wl in target_wls:
        header.append(f"A{wl}")

    for frac_name in fraction_names:
        if frac_name == "total":
            rang = "0-70"
        else:
            t_ini, t_fi = fractions_config.get(frac_name, [0, 0])
            rang = f"{t_ini}-{t_fi}"

        row = [frac_name, rang]
        # DOC
        doc_area = fraccions_data.get("DOC", {}).get(frac_name, 0.0)
        row.append(round(doc_area, 2) if doc_area > 0 else "-")
        # DAD wavelengths
        for wl in target_wls:
            dad_area = fraccions_data.get(f"A{wl}", {}).get(frac_name, 0.0)
            row.append(round(dad_area, 2) if dad_area > 0 else "-")
        areas_rows.append(row)

    df_areas = pd.DataFrame(areas_rows, columns=header)

    # Construir DataFrame DOC segons mode
    if is_dual:
        # Mode DUAL: RAW + processat per ambdós
        doc_data = {
            "time (min)": t_doc,
            "DOC_Direct (mAU)": y_doc_net,
            "DOC_Direct_RAW (mAU)": y_doc_raw,
            "BASELINE_Direct (mAU)": baseline,
            "DOC_UIB (mAU)": y_doc_uib,
        }
        # Afegir UIB RAW si disponible
        if y_doc_uib_raw is not None and len(y_doc_uib_raw) > 0:
            doc_data["DOC_UIB_RAW (mAU)"] = y_doc_uib_raw
        if baseline_uib is not None and len(baseline_uib) > 0:
            doc_data["BASELINE_UIB (mAU)"] = baseline_uib
        df_doc_out = pd.DataFrame(doc_data)
    else:
        # Mode simple: time, DOC, DOC_RAW, BASELINE
        df_doc_out = pd.DataFrame({
            "time (min)": t_doc,
            "DOC (mAU)": y_doc_net,
            "DOC_RAW (mAU)": y_doc_raw,
            "BASELINE (mAU)": baseline,
        })

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_id.to_excel(writer, sheet_name="ID", index=False)
        df_tmax.to_excel(writer, sheet_name="TMAX", index=False)
        df_areas.to_excel(writer, sheet_name="AREAS", index=False)
        df_doc_out.to_excel(writer, sheet_name="DOC", index=False)
        if not df_dad.empty:
            df_dad.to_excel(writer, sheet_name="DAD", index=False)


# =============================================================================
# FUNCIÓ PRINCIPAL: CONSOLIDATE_SEQUENCE
# =============================================================================
def consolidate_sequence(seq_path, config=None, progress_callback=None):
    """
    Consolida una seqüència completa.

    Args:
        seq_path: Ruta a la carpeta SEQ
        config: Configuració (opcional, usa DEFAULT_CONSOLIDATE_CONFIG)
        progress_callback: Funció callback(progress_pct, sample_name) per actualitzar progrés

    Returns:
        dict amb:
            - success: bool
            - mode: "UIB" o "DIRECT"
            - bp: bool
            - seq: nom seqüència
            - date: data
            - processed_count: nombre de mostres processades
            - output_path: ruta carpeta sortida
            - files: llista de fitxers generats
            - errors: llista d'errors
    """
    config = config or DEFAULT_CONSOLIDATE_CONFIG

    result = {
        "success": False,
        "mode": None,
        "bp": False,
        "seq": "",
        "date": "",
        "processed_count": 0,
        "output_path": "",
        "files": [],
        "errors": [],
        "warnings": [],
        "sample_stats": [],  # Estadístiques per mostra (timeout, SNR, etc.)
    }

    try:
        input_folder = seq_path

        # === AUTO-MIGRACIÓ: Si no hi ha MasterFile, intentar crear-lo ===
        master_files = glob.glob(os.path.join(input_folder, "*MasterFile*.xlsx"))
        master_files = [f for f in master_files if "~$" not in f and "backup" not in f.lower()]

        if not master_files:
            # Intentar migrar des de fitxers antics (v11/v12)
            try:
                from hpsec_migrate_master import migrate_single
                if progress_callback:
                    progress_callback(0, "Migrant format antic...")
                migrate_result = migrate_single(input_folder, force=False)

                if migrate_result.get('status') == 'ok':
                    result["warnings"].append(
                        f"AUTO-MIGRATE: MasterFile creat des de {migrate_result.get('rawdata', 'rawdata')}"
                    )
                elif migrate_result.get('status') == 'need_input':
                    result["warnings"].append(
                        "AVÍS: No s'ha trobat MasterFile ni rawdata v11/v12 per migrar"
                    )
                elif migrate_result.get('status') == 'exists':
                    pass  # MasterFile ja existeix (potser creat ara)
                else:
                    result["warnings"].append(
                        f"AVÍS: Migració fallida - {migrate_result.get('message', 'error desconegut')}"
                    )
            except ImportError:
                result["warnings"].append("AVÍS: Mòdul hpsec_migrate_master no disponible")
            except Exception as e:
                result["warnings"].append(f"AVÍS: Error durant migració automàtica: {e}")
        path_csv = os.path.join(input_folder, "CSV")
        path_3d = os.path.join(input_folder, "Export3d")

        # Fallback a Export3D si Export3d no existeix
        if not os.path.isdir(path_3d):
            path_3d = os.path.join(input_folder, "Export3D")

        # === AVISOS PUNTS CRÍTICS ===
        has_export3d = os.path.isdir(path_3d)
        has_csv = os.path.isdir(path_csv)

        if not has_export3d:
            result["warnings"].append("AVÍS: No s'ha trobat carpeta Export3d/Export3D (dades DAD)")

        if not has_csv:
            result["warnings"].append("AVÍS: No s'ha trobat carpeta CSV (fitxers UIB)")

        path_out = os.path.join(input_folder, "Resultats_Consolidats")
        os.makedirs(path_out, exist_ok=True)
        result["output_path"] = path_out

        # Detectar mode UIB vs DIRECT
        uib_files = []
        if has_csv:
            for ext in ("*.csv", "*.CSV", "*.txt", "*.TXT"):
                uib_files.extend(glob.glob(os.path.join(path_csv, f"*UIB1B*{ext}")))

        mode = "UIB" if uib_files else "DIRECT"
        result["mode"] = mode

        if mode == "DIRECT" and has_csv:
            result["warnings"].append("AVÍS: Carpeta CSV existeix però no s'han trobat fitxers UIB1B")

        # Detectar BP
        dad_pool = list_dad_files(
            path_3d if has_export3d else None,
            path_csv if has_csv else None,
        )

        if not dad_pool:
            result["warnings"].append("AVÍS: No s'han trobat fitxers DAD (Export3d ni CSV)")

        bp_flag = is_bp_seq(input_folder, dad_pool)
        method = "BP" if bp_flag else "COLUMN"
        result["bp"] = bp_flag

        # Obtenir SEQ i data
        seq = obtenir_seq(input_folder)
        seq_out = seq_tag(seq, bp_flag)
        date_master = read_master_date(input_folder)
        result["seq"] = seq_out
        result["date"] = date_master

        used_dad_files = set()
        used_uib_files = set()
        processed_count = 0

        if mode == "UIB":
            # MODE UIB/DUAL: DOC des de CSV UIB1B + opcionalment del mestre
            uib_files = sorted(set(uib_files))
            total = len(uib_files)

            # Intentar llegir el mestre per dual protocol
            mestre, master_format = trobar_excel_mestre(input_folder)
            df_toc = None
            df_seq = None
            master_index = None
            sample_ranges_new = None  # Per format nou
            df_dad_khp = None  # A254 del KHP per alineament (de 3-DAD_KHP)
            has_master = False
            master_info = {}  # Info de 0-INFO per passar a write_consolidated_excel
            valid_samples = set()  # Mostres vàlides de 1-HPLC-SEQ
            orphan_files = []  # Fitxers que no pertanyen a aquesta SEQ
            low_confidence_matches = []  # Matches amb confiança baixa (< CONFIDENCE_THRESHOLD)
            match_details = {}  # Detalls de matching per cada fitxer

            if mestre:
                result["master_format"] = master_format

                if master_format == "NEW":
                    # FORMAT NOU: MasterFile amb 4-TOC_CALC
                    try:
                        master_data = llegir_masterfile_nou(mestre)
                        if "error" not in master_data:
                            df_toc = master_data["toc"]
                            df_toc_calc = master_data["toc_calc"]
                            df_dad_khp = master_data.get("dad_khp")  # A254 del KHP (per alineament)
                            master_info = master_data.get("info", {})  # 0-INFO

                            # Obtenir mostres vàlides de 1-HPLC-SEQ
                            valid_samples = get_valid_samples_from_hplc_seq(master_data)
                            if valid_samples:
                                result["valid_samples"] = list(valid_samples)[:20]  # Guardar per debug

                            # === VALIDACIÓ CRÍTICA: Integritat 1-HPLC-SEQ ===
                            hplc_validation = validate_hplc_seq(master_data, uib_files)
                            result["hplc_seq_validation"] = hplc_validation

                            if not hplc_validation["valid"]:
                                # Afegir errors/warnings al resultat
                                for issue in hplc_validation["issues"]:
                                    if issue["severity"] == "CRITICAL":
                                        result["errors"].append(f"CRÍTIC: {issue['message']}")
                                    elif issue["severity"] == "ERROR":
                                        result["errors"].append(f"ERROR: {issue['message']}")
                                    else:
                                        result["warnings"].append(f"AVÍS: {issue['message']}")

                            # Calcular timing stats (per planificació futures SEQs)
                            timing_stats = calculate_timing_stats(master_data)
                            if timing_stats:
                                result["timing_stats"] = timing_stats

                            if df_toc is not None:
                                # Si no té TOC_CALC, calcular al vol
                                if df_toc_calc is None or df_toc_calc.empty:
                                    result["warnings"].append("MasterFile sense fulla 4-TOC_CALC, calculant...")
                                    df_toc_calc = calculate_toc_calc_on_the_fly(master_data)
                                    if df_toc_calc is not None and not df_toc_calc.empty:
                                        # Guardar al MasterFile
                                        try:
                                            _save_toc_calc_to_masterfile(mestre, df_toc_calc)
                                            result["warnings"].append(f"Fulla 4-TOC_CALC afegida al MasterFile ({len(df_toc_calc)} files)")
                                        except Exception as e:
                                            result["warnings"].append(f"No s'ha pogut guardar 4-TOC_CALC: {e}")

                                if df_toc_calc is not None and not df_toc_calc.empty:
                                    sample_ranges_new = build_sample_ranges_from_toc_calc(df_toc_calc, df_toc)
                                    if sample_ranges_new:
                                        has_master = True
                                        result["mode"] = "DUAL"
                    except Exception as e:
                        result["errors"].append(f"WARN: No s'ha pogut llegir MasterFile nou per dual: {e}")

                else:
                    # FORMAT ANTIC: Mestre amb 4-SEQ_DATA
                    try:
                        df_toc, df_seq = llegir_master_direct(mestre)
                        needed = ["SAMPLE_DESC", "START", "Row initial", "Row Final"]
                        if all(col in df_seq.columns for col in needed):
                            master_index = build_master_sample_index(df_seq)
                            has_master = True
                            result["mode"] = "DUAL"  # Canviar a DUAL si tenim mestre
                    except Exception as e:
                        result["errors"].append(f"WARN: No s'ha pogut llegir mestre per dual: {e}")

            # ================================================================
            # PROTOCOL ALINEACIÓ COLUMN: Pre-processar KHP per calcular shifts
            # Funciona tant per DUAL (has_master=True) com SINGLE (has_master=False)
            # - DUAL: calcula shift_uib + shift_direct
            # - SINGLE: calcula només shift_uib (DOC des de CSV)
            # ================================================================
            column_shift_uib = 0.0
            column_shift_direct = 0.0
            alignment_info = None

            if method == "COLUMN":  # Alineament per COLUMN (DUAL i SINGLE)
                # Identificar fitxer KHP UIB
                khp_uib_files = [f for f in uib_files if is_khp(os.path.basename(f))]

                if khp_uib_files:
                    khp_uib_file = khp_uib_files[0]
                    khp_nom = os.path.basename(khp_uib_file)
                    khp_mostra, khp_rep = netejar_nom_uib(khp_nom)

                    # Processar KHP per calcular shifts
                    try:
                        # Llegir DOC UIB del KHP
                        df_khp_uib, _ = llegir_doc_uib(khp_uib_file)
                        t_khp_uib = pd.to_numeric(df_khp_uib["time (min)"], errors="coerce").to_numpy()
                        y_khp_uib = pd.to_numeric(df_khp_uib["DOC"], errors="coerce").to_numpy()
                        base_khp_uib = get_baseline_correction(t_khp_uib, y_khp_uib, method, config)
                        y_khp_uib_net = apply_smoothing(y_khp_uib - base_khp_uib)
                        y_khp_uib_net[y_khp_uib_net < 0] = 0

                        # Llegir DOC Direct del KHP (suporta format antic i nou)
                        t_khp_direct = None
                        y_khp_direct_net = None
                        df_khp_direct = pd.DataFrame()

                        if master_index is not None:
                            # FORMAT ANTIC: usar match_uib_to_master
                            khp_match = match_uib_to_master(khp_mostra, khp_rep, master_index)
                            if khp_match:
                                khp_row = khp_match["row"]
                                khp_row_ini = int(float(khp_row["Row initial"]))
                                khp_row_fi = int(float(khp_row["Row Final"]))
                                df_khp_direct = extract_doc_from_master(df_toc, khp_row_ini, khp_row_fi, khp_row["START"], detect_timeouts=False)

                        elif sample_ranges_new is not None:
                            # FORMAT NOU: buscar en sample_ranges_new
                            khp_key = f"{khp_mostra}_R{khp_rep}"
                            khp_key_alt = f"{khp_mostra}_{khp_rep}"
                            khp_range = sample_ranges_new.get(khp_key) or sample_ranges_new.get(khp_key_alt)

                            # Buscar variants de KHP
                            if not khp_range:
                                for key in sample_ranges_new:
                                    if is_khp(key) and (f"_{khp_rep}" in key or f"_R{khp_rep}" in key):
                                        khp_range = sample_ranges_new[key]
                                        break

                            if khp_range:
                                khp_row_ini = int(khp_range.get("row_start", 0))
                                khp_row_fi = int(khp_range.get("row_end", 0))
                                t_start = khp_range.get("t_start", 0)
                                df_khp_direct = extract_doc_from_master(df_toc, khp_row_ini, khp_row_fi, t_start, detect_timeouts=False)

                        # Processar DOC Direct si s'ha trobat
                        if not df_khp_direct.empty:
                            t_khp_direct = pd.to_numeric(df_khp_direct["time (min)"], errors="coerce").to_numpy()
                            y_khp_direct = pd.to_numeric(df_khp_direct["DOC"], errors="coerce").to_numpy()
                            base_khp_direct = get_baseline_correction(t_khp_direct, y_khp_direct, method, config)
                            y_khp_direct_net = apply_smoothing(y_khp_direct - base_khp_direct)
                            y_khp_direct_net[y_khp_direct_net < 0] = 0

                        # Buscar A254 del KHP per alineament
                        # Prioritat: 1) MasterFile 3-DAD_KHP, 2) Export3D, 3) DAD1A
                        t_khp_a254 = None
                        y_khp_a254 = None
                        dad_source = "NOT_FOUND"

                        # Opció 2: Export3D (totes les wavelengths)
                        khp_dad_path, _ = choose_best_candidate(khp_mostra, khp_rep, dad_pool, set())

                        # Opció 3: DAD1A (només 254nm) - buscar a CSV folder
                        khp_dad1a_path = None
                        if path_csv and os.path.isdir(path_csv):
                            import glob as glob_mod
                            dad1a_candidates = glob_mod.glob(os.path.join(path_csv, f"{khp_mostra}*_DAD1A.*"))
                            if not dad1a_candidates:
                                dad1a_candidates = glob_mod.glob(os.path.join(path_csv, f"{khp_mostra}_{khp_rep}_DAD1A.*"))
                            if dad1a_candidates:
                                khp_dad1a_path = dad1a_candidates[0]

                        # Obtenir A254 amb prioritat: MasterFile > Export3D > DAD1A
                        t_khp_a254, y_khp_a254, dad_source = get_a254_for_alignment(
                            df_dad_khp=df_dad_khp,
                            path_export3d=khp_dad_path,
                            path_dad1a=khp_dad1a_path
                        )

                        # === VALIDAR QUALITAT KHP ABANS DE CALCULAR SHIFTS ===
                        # Obtenir paràmetres per comparació històrica
                        khp_conc = extract_khp_conc(khp_nom)
                        khp_volume = get_injection_volume(input_folder, is_bp=(method == "BP"))

                        khp_validation = validate_khp_for_alignment(
                            t_doc=t_khp_uib,
                            y_doc=y_khp_uib_net,
                            t_dad=t_khp_a254,
                            y_a254=y_khp_a254,
                            t_uib=t_khp_uib,  # El mateix per UIB
                            y_uib=y_khp_uib_net,
                            method=method,
                            seq_path=input_folder,
                            conc_ppm=khp_conc,
                            volume_uL=khp_volume
                        )

                        # Guardar validació a l'historial KHP
                        add_khp_validation_entry(
                            base_folder=input_folder,
                            khp_file=khp_nom,
                            seq_name=seq_out,
                            method=method,
                            validation_result=khp_validation
                        )

                        # Verificar si KHP és vàlid per usar en alineament
                        khp_is_valid = khp_validation.get("valid", False)
                        khp_issues = khp_validation.get("issues", [])

                        if not khp_is_valid:
                            # KHP INVALID: NO usar per alineament, buscar SIBLING o NO_KHP
                            issues_str = ", ".join(khp_issues)
                            result["warnings"].append(
                                f"WARN: KHP {khp_nom} INVALID - no s'usarà per alineament: {issues_str}"
                            )
                            result["warnings"].append(
                                f"INFO: Buscant SIBLING o usant shift=0"
                            )
                            # NO assignar alignment_info -> caurà al fallback SIBLING/NO_KHP

                        elif t_khp_a254 is not None and y_khp_a254 is not None:
                            # KHP VALID: Calcular shifts
                            tolerance_min = 2.0 / 60.0  # 2 segons
                            t_max_a254 = t_khp_a254[np.argmax(y_khp_a254)]

                            # Shift per DOC_UIB
                            # Usar t_max corregit si Batman va ser reparat
                            if khp_validation.get("t_max_corrected") is not None:
                                t_max_uib = khp_validation["t_max_corrected"]
                                result["warnings"].append(
                                    f"INFO: Usant t_max corregit ({t_max_uib:.3f} min) per KHP amb Batman reparat"
                                )
                            else:
                                t_max_uib = t_khp_uib[np.argmax(y_khp_uib_net)]

                            shift_uib = t_max_a254 - t_max_uib
                            if abs(shift_uib) > tolerance_min:
                                column_shift_uib = shift_uib

                            # Shift per DOC_Direct
                            t_max_direct = None
                            if t_khp_direct is not None and y_khp_direct_net is not None:
                                t_max_direct = t_khp_direct[np.argmax(y_khp_direct_net)]
                                shift_direct = t_max_a254 - t_max_direct
                                if abs(shift_direct) > tolerance_min:
                                    column_shift_direct = shift_direct

                            # Determinar mode: DUAL si tenim DOC_Direct, SINGLE si només UIB
                            align_mode = "DUAL" if t_khp_direct is not None else "SINGLE"

                            # Determinar status de validació per tracking
                            # (només arribem aquí si khp_is_valid == True)
                            khp_status = "VALID_WITH_WARNINGS" if khp_validation.get("warnings") else "VALID"

                            alignment_info = {
                                "khp_file": khp_nom,
                                "t_max_a254": t_max_a254,
                                "t_max_uib": t_max_uib,
                                "t_max_direct": t_max_direct,
                                "shift_uib": column_shift_uib,
                                "shift_direct": column_shift_direct,
                                "source": "KHP_LOCAL",
                                "dad_source": dad_source,
                                "align_mode": align_mode,
                                "khp_validation": khp_status,
                                "khp_issues": khp_validation.get("issues", []),
                                "khp_warnings": khp_validation.get("warnings", []),
                                "khp_metrics": khp_validation.get("metrics", {}),
                            }
                            result["alignment"] = alignment_info

                            # Actualitzar entrada historial amb shifts
                            add_khp_validation_entry(
                                base_folder=input_folder,
                                khp_file=khp_nom,
                                seq_name=seq_out,
                                method=method,
                                validation_result=khp_validation,
                                shift_info={
                                    "shift_uib": column_shift_uib,
                                    "shift_direct": column_shift_direct,
                                    "t_max_a254": float(t_max_a254),
                                    "t_max_uib": float(t_max_uib),
                                    "t_max_direct": float(t_max_direct) if t_max_direct else None,
                                }
                            )

                            # Guardar al log QAQC d'alineament
                            add_alignment_entry(
                                base_folder=input_folder,
                                seq_name=seq_out,
                                seq_date=date_master,
                                method=method,
                                shift_uib=column_shift_uib,
                                shift_direct=column_shift_direct,
                                khp_file=khp_nom,
                                details={
                                "t_max_a254": float(t_max_a254),
                                "t_max_uib": float(t_max_uib),
                                "t_max_direct": float(t_max_direct) if t_max_direct else None,
                                "align_mode": align_mode,
                                "khp_validation": khp_status,
                                "khp_issues": khp_validation.get("issues", []),
                                }
                            )

                            # Log informatiu del mode d'alineament
                            validation_note = ""
                            if not khp_is_valid:
                                validation_note = " [KHP INVALID - usant igualment]"
                            elif khp_validation.get("warnings"):
                                validation_note = " [KHP amb warnings]"

                            if align_mode == "SINGLE":
                                result["errors"].append(
                                    f"INFO: Alineament SINGLE (UIB→A254): shift={column_shift_uib*60:.1f}s{validation_note}"
                                )
                            else:
                                result["errors"].append(
                                    f"INFO: Alineament DUAL: UIB={column_shift_uib*60:.1f}s, Direct={column_shift_direct*60:.1f}s{validation_note}"
                                )

                    except Exception as e:
                        result["errors"].append(f"WARN: Error calculant alignment amb KHP: {e}")

                # Si no s'ha pogut calcular alignment amb KHP local, buscar SIBLING
                # NOTA: NO usar històric genèric per shifts
                if alignment_info is None:
                    # Buscar sibling amb KHP (mateixa SEQ numèrica)
                    sibling = find_sibling_alignment(input_folder, seq_out, method)

                    if sibling:
                        # Usar shifts del sibling
                        column_shift_uib = sibling.get("shift_uib", 0.0)
                        column_shift_direct = sibling.get("shift_direct", 0.0)
                        alignment_info = {
                            "khp_file": sibling.get("source_khp", ""),
                            "shift_uib": column_shift_uib,
                            "shift_direct": column_shift_direct,
                            "source": f"SIBLING:{sibling.get('source_seq', '')}",
                            "source_date": sibling.get("source_date", ""),
                            "khp_validation": sibling.get("khp_validation", "UNKNOWN"),
                            "khp_issues": sibling.get("khp_issues", []),
                        }
                        result["alignment"] = alignment_info
                        result["errors"].append(
                            f"INFO: Usant alignment de sibling SEQ {sibling.get('source_seq', '')} "
                            f"(shift_uib={column_shift_uib*60:.1f}s, shift_direct={column_shift_direct*60:.1f}s) "
                            f"[KHP: {sibling.get('khp_validation', 'UNKNOWN')}]"
                        )
                    else:
                        # SENSE KHP (local ni sibling) -> shift = 0
                        # Confiar en les hores dels equips + delay
                        column_shift_uib = 0.0
                        column_shift_direct = 0.0
                        alignment_info = {
                            "khp_file": "",
                            "shift_uib": 0.0,
                            "shift_direct": 0.0,
                            "source": "NO_KHP",
                            "khp_validation": "NO_KHP",
                        }
                        result["alignment"] = alignment_info
                        result["warnings"].append(
                            "AVÍS: No s'ha trobat KHP (local ni sibling). "
                            "Usant shift=0 (confiant en hores equips + delay)."
                        )

            # ================================================================
            # VALIDACIÓ KHP PER BP (només QC, no s'usa per alineament)
            # BP fa alineament per-sample, però validem KHP per detectar
            # anomalies de concentració
            # ================================================================
            if method == "BP":
                khp_uib_files_bp = [f for f in uib_files if is_khp(os.path.basename(f))]

                if khp_uib_files_bp:
                    khp_uib_file_bp = khp_uib_files_bp[0]
                    khp_nom_bp = os.path.basename(khp_uib_file_bp)

                    try:
                        # Llegir DOC UIB del KHP
                        df_khp_bp, _ = llegir_doc_uib(khp_uib_file_bp)
                        t_khp_bp = pd.to_numeric(df_khp_bp["time (min)"], errors="coerce").to_numpy()
                        y_khp_bp = pd.to_numeric(df_khp_bp["DOC"], errors="coerce").to_numpy()
                        base_khp_bp = get_baseline_correction(t_khp_bp, y_khp_bp, method, config)
                        y_khp_bp_net = apply_smoothing(y_khp_bp - base_khp_bp)
                        y_khp_bp_net[y_khp_bp_net < 0] = 0

                        # Buscar A254 del KHP
                        khp_mostra_bp, khp_rep_bp = netejar_nom_uib(khp_nom_bp)
                        khp_dad_path_bp, _ = choose_best_candidate(khp_mostra_bp, khp_rep_bp, dad_pool, set())

                        t_khp_a254_bp = None
                        y_khp_a254_bp = None

                        if khp_dad_path_bp:
                            df_dad_raw_bp, st_read_bp = llegir_dad_export3d(khp_dad_path_bp)
                            if st_read_bp.startswith("OK") and not df_dad_raw_bp.empty:
                                df_dad_bp = process_dad(df_dad_raw_bp, config)
                                if not df_dad_bp.empty and '254' in df_dad_bp.columns:
                                    t_khp_a254_bp = df_dad_bp['time (min)'].values
                                    y_khp_a254_bp = pd.to_numeric(df_dad_bp['254'], errors='coerce').to_numpy()

                        # Validar KHP BP (amb comparació històrica)
                        khp_conc_bp = extract_khp_conc(khp_nom_bp)
                        khp_volume_bp = get_injection_volume(input_folder, is_bp=True)

                        khp_validation_bp = validate_khp_for_alignment(
                            t_doc=t_khp_bp,
                            y_doc=y_khp_bp_net,
                            t_dad=t_khp_a254_bp if t_khp_a254_bp is not None else t_khp_bp,
                            y_a254=y_khp_a254_bp if y_khp_a254_bp is not None else np.zeros_like(y_khp_bp_net),
                            method="BP",
                            repair_batman=True,  # Reparar Batman també per BP
                            seq_path=input_folder,
                            conc_ppm=khp_conc_bp,
                            volume_uL=khp_volume_bp
                        )

                        # Guardar validació a l'historial
                        add_khp_validation_entry(
                            base_folder=input_folder,
                            khp_file=khp_nom_bp,
                            seq_name=seq_out,
                            method="BP",
                            validation_result=khp_validation_bp
                        )

                        # Reportar problemes
                        if not khp_validation_bp.get("valid", False):
                            issues_bp = ", ".join(khp_validation_bp.get("issues", []))
                            result["warnings"].append(
                                f"WARN: KHP {khp_nom_bp} INVALID (QC): {issues_bp}"
                            )
                            result["alignment"] = {
                                "khp_file": khp_nom_bp,
                                "source": "KHP_QC_ONLY",
                                "khp_validation": "INVALID",
                                "khp_issues": khp_validation_bp.get("issues", []),
                                "khp_metrics": khp_validation_bp.get("metrics", {}),
                            }
                        elif khp_validation_bp.get("warnings"):
                            warnings_bp = ", ".join(khp_validation_bp.get("warnings", []))
                            result["warnings"].append(
                                f"INFO: KHP {khp_nom_bp} OK amb warnings: {warnings_bp}"
                            )
                            result["alignment"] = {
                                "khp_file": khp_nom_bp,
                                "source": "KHP_QC_ONLY",
                                "khp_validation": "VALID_WITH_WARNINGS",
                                "khp_warnings": khp_validation_bp.get("warnings", []),
                                "khp_metrics": khp_validation_bp.get("metrics", {}),
                            }
                        else:
                            result["alignment"] = {
                                "khp_file": khp_nom_bp,
                                "source": "KHP_QC_ONLY",
                                "khp_validation": "VALID",
                                "khp_metrics": khp_validation_bp.get("metrics", {}),
                            }

                    except Exception as e:
                        result["warnings"].append(f"WARN: Error validant KHP BP: {e}")

            # Tracking de sample_ranges usats (per evitar duplicats amb blocs)
            used_sample_ranges = set()

            for i, f_doc_uib in enumerate(uib_files):
                progress = int(100 * (i + 1) / max(total, 1))
                nom_doc_uib = os.path.basename(f_doc_uib)
                mostra, rep = netejar_nom_uib(nom_doc_uib)

                if progress_callback:
                    progress_callback(progress, mostra)

                # === VERIFICAR SI LA MOSTRA PERTANY A AQUESTA SEQ ===
                if valid_samples:
                    match_info = match_sample_confidence(mostra, valid_samples)
                    match_details[nom_doc_uib] = match_info

                    if not match_info["matched"]:
                        # Fitxer orfe - existeix però no està a 1-HPLC-SEQ
                        orphan_files.append(nom_doc_uib)
                        best_guess = f" (similar a: {match_info['best_match']})" if match_info.get("best_match") else ""

                        # Afegir a sample_stats amb error (NO saltar - mostrar a la taula)
                        doc_direct_status = {
                            "status": "ORFE_HPLC_SEQ",
                            "message": f"Fitxer existeix, falta a 1-HPLC-SEQ{best_guess}"
                        }
                        file_info = {
                            "file_dad": "",
                            "file_uib": nom_doc_uib,
                            "row_start": None,
                            "row_end": None,
                            "npts": 0,
                            "match_confidence": 0.0,
                            "match_type": "orphan",
                        }
                        sample_stat = _collect_sample_stats(
                            mostra, rep, None, None, None, "UIB", file_info, doc_direct_status
                        )
                        result["sample_stats"].append(sample_stat)
                        result["warnings"].append(
                            f"ORFE: {nom_doc_uib} no pertany a aquesta SEQ (no està a 1-HPLC-SEQ){best_guess}"
                        )
                        continue  # Processat com a orfe, continuar amb el següent
                    elif match_info["confidence"] < CONFIDENCE_THRESHOLD:
                        # Match amb confiança baixa - processar però marcar per revisió
                        low_confidence_matches.append({
                            "file": nom_doc_uib,
                            "sample": mostra,
                            "matched_to": match_info["best_match"],
                            "confidence": match_info["confidence"],
                            "match_type": match_info["match_type"]
                        })
                        result["warnings"].append(
                            f"BAIXA_CONFIANÇA: {nom_doc_uib} → {match_info['best_match']} "
                            f"({match_info['confidence']:.0f}% {match_info['match_type']})"
                        )

                # Llegir DOC UIB
                df_doc_uib, st_doc_uib = llegir_doc_uib(f_doc_uib)

                # Tracking UIB files utilitzats
                used_uib_files.add(f_doc_uib)

                # Determinar si tenim dades UIB
                if df_doc_uib.empty or len(df_doc_uib) < 10:
                    st_doc_uib = "NO_DATA"
                    t_uib = np.array([0.0])
                    y_uib_raw = np.array([0.0])
                    y_uib_net = np.array([0.0])
                    base_uib = np.array([0.0])
                else:
                    t_uib = pd.to_numeric(df_doc_uib["time (min)"], errors="coerce").to_numpy()
                    y_uib_raw = pd.to_numeric(df_doc_uib["DOC"], errors="coerce").to_numpy()
                    base_uib = get_baseline_correction(t_uib, y_uib_raw, method, config)
                    y_uib_net = apply_smoothing(y_uib_raw - base_uib)
                    y_uib_net[y_uib_net < 0] = 0

                # Variables per DOC Direct (del mestre)
                t_direct = None
                y_direct_raw = None
                y_direct_net = None
                base_direct = None
                nom_doc_direct = "NO_DATA"
                st_doc_direct = "NO_DATA"
                row_ini = None
                row_fi = None
                timeout_info = None  # Info de timeouts detectats en DOC Direct
                doc_direct_status = {"status": "OK", "message": ""}  # Estat per GUI

                # Intentar obtenir DOC Direct del mestre (dual protocol)
                if has_master:
                    matched = False

                    # FORMAT NOU: Usar sample_ranges_new
                    if sample_ranges_new and not matched:
                        # Construir Sample_Rep per buscar
                        sample_rep_key = f"{mostra}_R{rep}"
                        sample_rep_normalized = normalize_key(sample_rep_key)
                        # Normalitzar base sample (sense _R#)
                        mostra_normalized = normalize_key(mostra)

                        # Buscar coincidència: exacta, normalitzada, o per base+bloc
                        match_info = None
                        matched_sample_rep = None  # Sample_Rep oficial del MasterFile
                        block_matches = []  # Per samples amb blocs (B1, B2)

                        for sr, info in sample_ranges_new.items():
                            # Saltar si ja s'ha usat aquest entry
                            if sr in used_sample_ranges:
                                continue

                            # 1. Coincidència exacta
                            if sr == sample_rep_key:
                                match_info = info
                                matched_sample_rep = sr
                                break

                            # 2. Coincidència normalitzada (ignorar espais, guions, etc.)
                            if normalize_key(sr) == sample_rep_normalized:
                                match_info = info
                                matched_sample_rep = sr
                                break

                            # 3. Samples amb blocs: SAMPLE_B#_R# → extreure base i comparar
                            if "_B" in sr and "_R" in sr:
                                # Extreure base: NAOH01MM_B1_R1 → NAOH01MM
                                sr_parts = sr.split("_B")
                                sr_base = sr_parts[0]
                                sr_rest = sr_parts[1] if len(sr_parts) > 1 else ""
                                sr_rep = sr_rest.split("_R")[1] if "_R" in sr_rest else "1"

                                if normalize_key(sr_base) == mostra_normalized and sr_rep == str(rep):
                                    block_matches.append((sr, info))

                            # 4. Coincidència parcial sense bloc
                            else:
                                sr_base = sr.rsplit("_R", 1)[0] if "_R" in sr else sr
                                sr_rep = sr.rsplit("_R", 1)[1] if "_R" in sr else "1"
                                if normalize_key(sr_base) == mostra_normalized and sr_rep == str(rep):
                                    match_info = info
                                    matched_sample_rep = sr
                                    break

                        # Si no hi ha match exacte però hi ha matches amb blocs, agafar el primer disponible
                        if match_info is None and block_matches:
                            matched_sample_rep, match_info = block_matches[0]

                        # Si encara no hi ha match, registrar error
                        if match_info is None:
                            doc_direct_status = {
                                "status": "FALTA_HPLC_SEQ",
                                "message": f"Rèplica {rep} absent a 1-HPLC-SEQ"
                            }
                            result["warnings"].append(
                                f"SENSE_MATCH: {sample_rep_key} no trobat a MasterFile (rèplica {rep} absent)"
                            )

                        # Marcar entry com a usat i actualitzar mostra/rep amb Sample_Rep oficial
                        if matched_sample_rep:
                            used_sample_ranges.add(matched_sample_rep)
                            # Parsejar Sample_Rep: pot ser "NAOH01MM_B1_R1" o "FR2608_R1"
                            if "_B" in matched_sample_rep and "_R" in matched_sample_rep:
                                # Format amb bloc: SAMPLE_B#_R#
                                parts = matched_sample_rep.rsplit("_R", 1)
                                mostra = parts[0]  # NAOH01MM_B1
                                rep = int(parts[1]) if len(parts) > 1 else rep
                            else:
                                # Format sense bloc: SAMPLE_R#
                                parts = matched_sample_rep.rsplit("_R", 1)
                                mostra = parts[0]
                                rep = int(parts[1]) if len(parts) > 1 else rep

                        if match_info:
                            try:
                                row_ini = match_info.get("row_start")
                                row_fi = match_info.get("row_end")
                                t_start = match_info.get("t_start")

                                if row_ini is not None and row_fi is not None:
                                    df_doc_direct, timeout_info = extract_doc_from_masterfile(df_toc, row_ini, row_fi, t_start, detect_timeouts=True)

                                    if not df_doc_direct.empty:
                                        t_direct = pd.to_numeric(df_doc_direct["time (min)"], errors="coerce").to_numpy()
                                        y_direct_raw = pd.to_numeric(df_doc_direct["DOC"], errors="coerce").to_numpy()

                                        base_direct = get_baseline_correction(t_direct, y_direct_raw, method, config, use_end=(method == "BP"))
                                        y_direct_net = apply_smoothing(y_direct_raw - base_direct)
                                        y_direct_net[y_direct_net < 0] = 0

                                        nom_doc_direct = os.path.basename(mestre)
                                        st_doc_direct = "OK"
                                        matched = True

                                        # Afegir warning si hi ha timeout en zona crítica
                                        if timeout_info and timeout_info.get("severity") in ["WARNING", "CRITICAL"]:
                                            result["errors"].append(f"TIMEOUT: {mostra} - {timeout_info.get('warning_message', '')}")
                            except Exception as e:
                                result["errors"].append(f"WARN: Error extraient DOC Direct (NEW) per {mostra}: {e}")

                    # FORMAT ANTIC: Usar master_index
                    if master_index and not matched:
                        match = match_uib_to_master(mostra, rep, master_index)
                        if match:
                            row = match["row"]
                            try:
                                row_ini = int(float(row["Row initial"]))
                                row_fi = int(float(row["Row Final"]))
                                df_doc_direct, timeout_info = extract_doc_from_master(df_toc, row_ini, row_fi, row["START"], detect_timeouts=True)

                                if not df_doc_direct.empty:
                                    t_direct = pd.to_numeric(df_doc_direct["time (min)"], errors="coerce").to_numpy()
                                    y_direct_raw = pd.to_numeric(df_doc_direct["DOC"], errors="coerce").to_numpy()

                                    # Per BP Direct, usar el final del cromatograma per baseline
                                    base_direct = get_baseline_correction(t_direct, y_direct_raw, method, config, use_end=(method == "BP"))
                                    y_direct_net = apply_smoothing(y_direct_raw - base_direct)
                                    y_direct_net[y_direct_net < 0] = 0

                                    nom_doc_direct = os.path.basename(mestre)
                                    st_doc_direct = "OK"

                                    # Afegir warning si hi ha timeout en zona crítica
                                    if timeout_info and timeout_info.get("severity") in ["WARNING", "CRITICAL"]:
                                        result["errors"].append(f"TIMEOUT: {mostra} - {timeout_info.get('warning_message', '')}")
                            except Exception as e:
                                result["errors"].append(f"WARN: Error extraient DOC Direct per {mostra}: {e}")

                # Decidir quin temps i DOC usar per l'anàlisi principal
                # Si tenim dual, usar el Direct com a principal (és el "oficial")
                if t_direct is not None and len(t_direct) > 1:
                    t_doc = t_direct
                    y_doc_net = y_direct_net
                    y_doc_raw = y_direct_raw
                    base = base_direct
                    nom_doc = nom_doc_direct
                    st_doc = st_doc_direct
                    doc_mode = "DUAL"
                elif st_doc_uib != "NO_DATA":
                    # Fallback a UIB si no hi ha Direct però UIB és vàlid
                    t_doc = t_uib.copy()  # Copiar per no modificar l'original
                    y_doc_net = y_uib_net
                    y_doc_raw = y_uib_raw
                    base = base_uib
                    nom_doc = nom_doc_uib
                    st_doc = st_doc_uib
                    doc_mode = "UIB"

                    # APLICAR SHIFT D'ALINEAMENT per MODE SINGLE (UIB sense Direct)
                    # El shift alinea UIB amb A254: t_corregit = t_original + shift
                    if method == "COLUMN" and column_shift_uib != 0.0:
                        t_doc = t_doc + column_shift_uib
                else:
                    # Ni Direct ni UIB tenen dades
                    t_doc = t_uib
                    y_doc_net = y_uib_net
                    y_doc_raw = y_uib_raw
                    base = base_uib
                    nom_doc = nom_doc_uib
                    st_doc = "NO_DATA"
                    doc_mode = "UIB"

                # Buscar DAD
                dad_path, st_dad = choose_best_candidate(mostra, rep, dad_pool, used_dad_files)
                df_dad = pd.DataFrame()
                nom_dad = "NO_DATA"

                if dad_path:
                    used_dad_files.add(dad_path)
                    nom_dad = os.path.basename(dad_path)
                    df_dad_raw, st_read = llegir_dad_export3d(dad_path)
                    if st_read.startswith("OK") and not df_dad_raw.empty:
                        df_dad = process_dad(df_dad_raw, config)
                        st_dad = "OK" if not df_dad.empty else "NO_DATA"

                # Detectar pic i calcular àrees (només si tenim dades)
                if st_doc != "NO_DATA":
                    peak_info = detect_main_peak(t_doc, y_doc_net, config["peak_min_prominence_pct"])
                    sample_analysis = analyze_sample_areas(t_doc, y_doc_net, df_dad, peak_info, config)
                else:
                    peak_info = {"valid": False, "t_max": 0, "area": 0, "t_start": 0, "t_end": 0}
                    sample_analysis = {}

                # Guardar Excel
                out_name = f"{mostra}_{seq_out}_R{rep}.xlsx"
                out_path = os.path.join(path_out, out_name)

                # =============================================================
                # ALINEAMENT TEMPORAL - A254 (DAD) ÉS LA REFERÈNCIA ABSOLUTA
                # - MAI moure A254 ni DAD
                # - DOC_Direct s'alinea a A254
                # - DOC_UIB s'alinea a A254
                # =============================================================
                y_doc_uib_aligned = None
                applied_shift_direct_sample = 0.0
                applied_shift_uib_sample = 0.0

                # Obtenir t_max de A254 si tenim DAD
                t_max_a254_sample = None
                if not df_dad.empty and '254' in df_dad.columns:
                    try:
                        t_dad_sample = df_dad['time (min)'].values
                        y_a254_sample = pd.to_numeric(df_dad['254'], errors='coerce').to_numpy()
                        if len(y_a254_sample) > 10 and np.nanmax(y_a254_sample) > 0:
                            t_max_a254_sample = t_dad_sample[np.argmax(y_a254_sample)]
                    except Exception:
                        pass

                if doc_mode == "DUAL" and len(y_uib_net) > 0:
                    try:
                        if method == "BP":
                            # =========================================
                            # BP: Alinear cada mostra individualment a A254
                            # =========================================
                            if t_max_a254_sample is not None:
                                # Trobar màxims de DOC
                                t_max_direct_sample = t_doc[np.argmax(y_doc_net)]
                                t_max_uib_sample = t_uib[np.argmax(y_uib_net)]

                                # Calcular shifts necessaris per alinear a A254
                                shift_direct_bp = t_max_a254_sample - t_max_direct_sample
                                shift_uib_bp = t_max_a254_sample - t_max_uib_sample

                                tolerance_min = 2.0 / 60.0  # 2 segons

                                # Aplicar shift a DOC_Direct si cal
                                if abs(shift_direct_bp) > tolerance_min:
                                    # Crear nova escala de temps alineada a A254
                                    t_doc_shifted = t_doc + shift_direct_bp
                                    # Interpolar a l'escala de temps de DAD
                                    y_doc_net = np.interp(t_dad_sample, t_doc_shifted, y_doc_net, left=0, right=0)
                                    y_doc_raw = np.interp(t_dad_sample, t_doc_shifted, y_doc_raw, left=0, right=0)
                                    base = np.interp(t_dad_sample, t_doc_shifted, base, left=base[0], right=base[-1])
                                    t_doc = t_dad_sample
                                    applied_shift_direct_sample = shift_direct_bp

                                # Aplicar shift a DOC_UIB si cal
                                if abs(shift_uib_bp) > tolerance_min:
                                    t_uib_shifted = t_uib + shift_uib_bp
                                    y_doc_uib_aligned = np.interp(t_doc, t_uib_shifted, y_uib_net, left=0, right=0)
                                    applied_shift_uib_sample = shift_uib_bp
                                else:
                                    # Sense shift, només interpolar a escala de temps Direct
                                    y_doc_uib_aligned = np.interp(t_doc, t_uib, y_uib_net, left=0, right=0)
                            else:
                                # Sense A254: fallback a alinear UIB amb Direct (comportament anterior)
                                # NOTA: Això no és ideal, caldria revisar si DAD està disponible
                                y_doc_uib_aligned = np.interp(t_doc, t_uib, y_uib_net, left=0, right=0)
                                result["warnings"].append(f"WARN: {mostra} sense A254 per alineament BP")
                        else:
                            # =========================================
                            # COLUMN: Usar shifts calculats del KHP
                            # Interpolar a l'eix de temps DAD (com BP)
                            # =========================================
                            # Obtenir eix de temps DAD
                            t_dad_sample = None
                            if not df_dad.empty and 'time (min)' in df_dad.columns:
                                t_dad_sample = df_dad['time (min)'].values

                            # Aplicar shift a DOC_Direct i interpolar a DAD
                            if t_direct is not None and t_dad_sample is not None:
                                t_direct_shifted = t_direct + column_shift_direct
                                y_doc_net = np.interp(t_dad_sample, t_direct_shifted, y_direct_net, left=0, right=0)
                                y_doc_raw = np.interp(t_dad_sample, t_direct_shifted, y_direct_raw, left=0, right=0)
                                base = np.interp(t_dad_sample, t_direct_shifted, base_direct, left=base_direct[0], right=base_direct[-1])
                                t_doc = t_dad_sample  # Usar eix DAD com a referència
                                if column_shift_direct != 0.0:
                                    applied_shift_direct_sample = column_shift_direct

                            # Aplicar shift a DOC_UIB i interpolar a l'eix de temps actual (DAD o Direct)
                            t_uib_shifted = t_uib + column_shift_uib
                            y_doc_uib_aligned = np.interp(t_doc, t_uib_shifted, y_uib_net, left=0, right=0)
                            if column_shift_uib != 0.0:
                                applied_shift_uib_sample = column_shift_uib

                    except Exception as e:
                        # Fallback: interpolació simple sense shift
                        try:
                            y_doc_uib_aligned = np.interp(t_doc, t_uib, y_uib_net, left=0, right=0)
                            result["warnings"].append(f"WARN: Error alineament {mostra}: {e}")
                        except Exception:
                            y_doc_uib_aligned = None

                # Preparar UIB RAW interpolat a l'escala de temps (amb shift si s'ha aplicat)
                y_uib_raw_interp = None
                base_uib_interp = None
                if doc_mode == "DUAL" and len(y_uib_raw) > 0:
                    try:
                        # Aplicar el mateix shift que a y_uib_net
                        if applied_shift_uib_sample != 0.0:
                            t_uib_shifted = t_uib + applied_shift_uib_sample
                            y_uib_raw_interp = np.interp(t_doc, t_uib_shifted, y_uib_raw, left=0, right=0)
                            base_uib_interp = np.interp(t_doc, t_uib_shifted, base_uib, left=base_uib[0], right=base_uib[-1])
                        else:
                            y_uib_raw_interp = np.interp(t_doc, t_uib, y_uib_raw, left=0, right=0)
                            base_uib_interp = np.interp(t_doc, t_uib, base_uib, left=base_uib[0], right=base_uib[-1])
                    except Exception:
                        pass

                # Determinar shifts aplicats (usant els valors calculats durant l'alineament)
                applied_shift_uib = None
                applied_shift_direct = None
                if doc_mode == "DUAL":
                    # Usar els shifts calculats per aquesta mostra específica
                    if applied_shift_uib_sample != 0.0:
                        applied_shift_uib = applied_shift_uib_sample
                    if applied_shift_direct_sample != 0.0:
                        applied_shift_direct = applied_shift_direct_sample

                # Calcular SNR i baseline noise (amb finestres per evitar timeouts)
                timeout_positions = []
                if timeout_info and timeout_info.get("timeouts"):
                    timeout_positions = [to["t_start_min"] for to in timeout_info["timeouts"]]

                snr_info = calculate_snr_info(
                    y_doc_net, peak_info,
                    y_doc_uib=y_doc_uib_aligned if doc_mode == "DUAL" else None,
                    t_min=t_doc,
                    method=method,
                    timeout_positions=timeout_positions
                )

                write_consolidated_excel(
                    out_path, mostra, rep, seq_out, date_master, method, doc_mode,
                    nom_doc, nom_dad, st_doc, st_dad, t_doc, y_doc_raw, y_doc_net, base,
                    df_dad, peak_info, sample_analysis,
                    master_file=mestre if doc_mode == "DUAL" else None,
                    row_start=row_ini, row_end=row_fi,
                    # Paràmetres dual protocol
                    y_doc_uib=y_doc_uib_aligned,
                    y_doc_uib_raw=y_uib_raw_interp,
                    baseline_uib=base_uib_interp,
                    fitxer_doc_uib=nom_doc_uib if doc_mode == "DUAL" else None,
                    st_doc_uib=st_doc_uib if doc_mode == "DUAL" else None,
                    # Paràmetres de processament
                    shift_uib=applied_shift_uib,
                    shift_direct=applied_shift_direct,
                    smoothing_applied=True,
                    # Info del MasterFile
                    master_info=master_info,
                    # Detecció de timeouts TOC
                    timeout_info=timeout_info,
                    # SNR i baseline noise
                    snr_info=snr_info
                )

                result["files"].append(out_path)
                processed_count += 1

                # Recollir estadístiques de la mostra
                # Obtenir confiança del match (si available)
                match_conf = 100.0  # Default alta confiança
                match_type = "exact"
                if nom_doc_uib in match_details:
                    mi = match_details[nom_doc_uib]
                    match_conf = mi.get("confidence", 100.0)
                    match_type = mi.get("match_type", "exact")

                file_info = {
                    "file_dad": nom_doc,
                    "file_uib": nom_doc_uib if doc_mode == "DUAL" else "",
                    "row_start": row_ini,
                    "row_end": row_fi,
                    "npts": len(t_doc) if t_doc is not None else 0,
                    "match_confidence": match_conf,
                    "match_type": match_type,
                }
                sample_stat = _collect_sample_stats(mostra, rep, timeout_info, snr_info, peak_info, doc_mode, file_info, doc_direct_status)
                result["sample_stats"].append(sample_stat)

        else:
            # MODE DIRECT: DOC des d'Excel mestre
            master_info = {}  # Inicialitzar per MODE DIRECT
            mestre, master_format = trobar_excel_mestre(input_folder)
            if not mestre:
                result["errors"].append("No s'ha trobat Excel mestre")
                return result

            result["master_format"] = master_format

            if master_format == "NEW":
                # ============================================================
                # FORMAT NOU: MasterFile amb 4-TOC_CALC
                # ============================================================
                master_data = llegir_masterfile_nou(mestre)
                if "error" in master_data:
                    result["errors"].append(f"Error llegint MasterFile: {master_data['error']}")
                    return result

                df_toc = master_data["toc"]
                df_toc_calc = master_data["toc_calc"]
                df_dad_khp_direct = master_data.get("dad_khp")  # A254 del KHP (per alineament)
                master_info = master_data.get("info", {})  # 0-INFO

                # Calcular timing stats (per planificació futures SEQs)
                timing_stats = calculate_timing_stats(master_data)
                if timing_stats:
                    result["timing_stats"] = timing_stats

                if df_toc is None or df_toc.empty:
                    result["errors"].append("MasterFile sense dades TOC")
                    return result

                # Si TOC_CALC buit, calcular al vol des de HPLC-SEQ i TOC
                if df_toc_calc is None or df_toc_calc.empty:
                    result["warnings"].append("MasterFile sense fulla 4-TOC_CALC, calculant...")
                    df_toc_calc = calculate_toc_calc_on_the_fly(master_data)
                    if df_toc_calc is None or df_toc_calc.empty:
                        result["errors"].append("No s'ha pogut calcular TOC_CALC: falten dades HPLC-SEQ o TOC")
                        return result

                    # Guardar la fulla calculada al MasterFile
                    try:
                        _save_toc_calc_to_masterfile(mestre, df_toc_calc)
                        result["warnings"].append(f"Fulla 4-TOC_CALC afegida al MasterFile ({len(df_toc_calc)} files)")
                    except Exception as e:
                        result["warnings"].append(f"No s'ha pogut guardar 4-TOC_CALC al MasterFile: {e}")

                # Construir rangs de mostres
                sample_ranges = build_sample_ranges_from_toc_calc(df_toc_calc, df_toc)

                if not sample_ranges:
                    result["errors"].append("No s'han pogut extreure mostres de 4-TOC_CALC")
                    return result

                # ================================================================
                # ALINEAMENT MODE DIRECT: Calcular shift DOC→A254 usant KHP
                # ================================================================
                direct_shift = 0.0
                alignment_info_direct = None

                if method == "COLUMN":
                    # Buscar KHP a sample_ranges
                    khp_keys = [k for k in sample_ranges.keys() if is_khp(k)]

                    if khp_keys:
                        khp_key = khp_keys[0]
                        khp_range = sample_ranges[khp_key]
                        khp_mostra, khp_rep = split_sample_rep(khp_key)
                        if khp_rep is None:
                            khp_rep = "1"

                        try:
                            # Extreure DOC del KHP des del MasterFile
                            khp_row_ini = int(khp_range.get("row_start", 0))
                            khp_row_fi = int(khp_range.get("row_end", 0))
                            khp_t_start = khp_range.get("t_start", 0)
                            df_khp_doc = extract_doc_from_masterfile(df_toc, khp_row_ini, khp_row_fi, khp_t_start, detect_timeouts=False)

                            if not df_khp_doc.empty:
                                t_khp_doc = pd.to_numeric(df_khp_doc["time (min)"], errors="coerce").to_numpy()
                                y_khp_doc = pd.to_numeric(df_khp_doc["DOC"], errors="coerce").to_numpy()
                                base_khp = get_baseline_correction(t_khp_doc, y_khp_doc, method, config)
                                y_khp_net = apply_smoothing(y_khp_doc - base_khp)
                                y_khp_net[y_khp_net < 0] = 0

                                # Buscar A254 del KHP per alineament
                                # Prioritat: 1) MasterFile 3-DAD_KHP, 2) Export3D, 3) DAD1A
                                khp_dad_path, _ = choose_best_candidate(khp_mostra, khp_rep, dad_pool, set())

                                # Buscar DAD1A com a fallback
                                khp_dad1a_path = None
                                if path_csv and os.path.isdir(path_csv):
                                    import glob as glob_mod
                                    dad1a_candidates = glob_mod.glob(os.path.join(path_csv, f"{khp_mostra}*_DAD1A.*"))
                                    if dad1a_candidates:
                                        khp_dad1a_path = dad1a_candidates[0]

                                # Obtenir A254 amb prioritat: MasterFile > Export3D > DAD1A
                                t_khp_a254, y_khp_a254, dad_source = get_a254_for_alignment(
                                    df_dad_khp=df_dad_khp_direct,
                                    path_export3d=khp_dad_path,
                                    path_dad1a=khp_dad1a_path
                                )

                                # === VALIDAR QUALITAT KHP (MODE DIRECT) ===
                                # Obtenir paràmetres per comparació històrica
                                khp_conc_direct = extract_khp_conc(khp_key)
                                khp_volume_direct = get_injection_volume(input_folder, is_bp=(method == "BP"))

                                khp_validation_direct = validate_khp_for_alignment(
                                    t_doc=t_khp_doc,
                                    y_doc=y_khp_net,
                                    t_dad=t_khp_a254,
                                    y_a254=y_khp_a254,
                                    method=method,
                                    repair_batman=True,
                                    seq_path=input_folder,
                                    conc_ppm=khp_conc_direct,
                                    volume_uL=khp_volume_direct
                                )

                                # Guardar validació a l'historial KHP
                                add_khp_validation_entry(
                                    base_folder=input_folder,
                                    khp_file=khp_key,
                                    seq_name=seq_out,
                                    method=method,
                                    validation_result=khp_validation_direct
                                )

                                khp_is_valid_direct = khp_validation_direct.get("valid", False)

                                if not khp_is_valid_direct:
                                    # KHP INVALID: NO usar per alineament
                                    issues_str = ", ".join(khp_validation_direct.get("issues", []))
                                    result["warnings"].append(
                                        f"WARN: KHP {khp_key} INVALID - no s'usarà per alineament: {issues_str}"
                                    )
                                    result["warnings"].append(
                                        f"INFO: Buscant SIBLING o usant shift=0"
                                    )
                                    # NO calcular shift ni assignar alignment_info_direct
                                    # -> caurà al fallback SIBLING/NO_KHP

                                elif t_khp_a254 is not None and y_khp_a254 is not None:
                                    # KHP VALID: Calcular shift
                                    tolerance_min = 2.0 / 60.0  # 2 segons
                                    t_max_a254 = t_khp_a254[np.argmax(y_khp_a254)]
                                    t_max_doc = t_khp_doc[np.argmax(y_khp_net)]
                                    shift_calc = t_max_a254 - t_max_doc

                                    if abs(shift_calc) > tolerance_min:
                                        direct_shift = shift_calc

                                    # Determinar status validació
                                    # (només arribem aquí si khp_is_valid_direct == True)
                                    khp_status_direct = "VALID_WITH_WARNINGS" if khp_validation_direct.get("warnings") else "VALID"

                                    alignment_info_direct = {
                                        "khp_file": khp_key,
                                        "t_max_a254": t_max_a254,
                                        "t_max_doc": t_max_doc,
                                        "shift_direct": direct_shift,
                                        "source": "KHP_LOCAL",
                                        "dad_source": dad_source,
                                        "align_mode": "DIRECT",
                                        "khp_validation": khp_status_direct,
                                        "khp_issues": khp_validation_direct.get("issues", []),
                                        "khp_warnings": khp_validation_direct.get("warnings", []),
                                        "khp_metrics": khp_validation_direct.get("metrics", {}),
                                    }
                                    result["alignment"] = alignment_info_direct

                                    # Actualitzar historial amb shifts
                                    add_khp_validation_entry(
                                        base_folder=input_folder,
                                        khp_file=khp_key,
                                        seq_name=seq_out,
                                        method=method,
                                        validation_result=khp_validation_direct,
                                        shift_info={
                                            "shift_direct": direct_shift,
                                            "t_max_a254": float(t_max_a254),
                                            "t_max_doc": float(t_max_doc),
                                        }
                                    )

                                    # Log QAQC
                                    add_alignment_entry(
                                        base_folder=input_folder,
                                        seq_name=seq_out,
                                        seq_date=date_master,
                                        method=method,
                                        shift_uib=0.0,
                                        shift_direct=direct_shift,
                                        khp_file=khp_key,
                                        details={
                                            "t_max_a254": float(t_max_a254),
                                            "t_max_doc": float(t_max_doc),
                                            "align_mode": "DIRECT",
                                            "khp_validation": khp_status_direct,
                                            "khp_issues": khp_validation_direct.get("issues", []),
                                        }
                                    )

                                    result["errors"].append(
                                        f"INFO: Alineament DIRECT (DOC→A254): shift={direct_shift*60:.1f}s"
                                    )

                        except Exception as e:
                            result["errors"].append(f"WARN: Error calculant alignment DIRECT amb KHP: {e}")

                    # Fallback a SIBLING si no s'ha pogut calcular
                    # NOTA: NO usar històric genèric per shifts
                    if alignment_info_direct is None:
                        sibling = find_sibling_alignment(input_folder, seq_out, method)

                        if sibling:
                            # Usar shifts del sibling
                            direct_shift = sibling.get("shift_direct", 0.0)
                            alignment_info_direct = {
                                "shift_direct": direct_shift,
                                "source": f"SIBLING:{sibling.get('source_seq', '')}",
                                "khp_validation": sibling.get("khp_validation", "UNKNOWN"),
                                "khp_issues": sibling.get("khp_issues", []),
                            }
                            result["alignment"] = alignment_info_direct
                            result["errors"].append(
                                f"INFO: Usant alignment de sibling SEQ {sibling.get('source_seq', '')} "
                                f"(shift_direct={direct_shift*60:.1f}s) "
                                f"[KHP: {sibling.get('khp_validation', 'UNKNOWN')}]"
                            )
                        else:
                            # SENSE KHP (local ni sibling) -> shift = 0
                            direct_shift = 0.0
                            alignment_info_direct = {
                                "shift_direct": 0.0,
                                "source": "NO_KHP",
                                "khp_validation": "NO_KHP",
                            }
                            result["alignment"] = alignment_info_direct
                            result["warnings"].append(
                                "AVÍS: No s'ha trobat KHP (local ni sibling). "
                                "Usant shift=0 (confiant en hores equips + delay)."
                            )

                # Filtrar mostres a processar (excloure controls)
                samples_to_process = {}
                for sample_rep, info in sample_ranges.items():
                    if skip_sample_direct(sample_rep):
                        continue
                    samples_to_process[sample_rep] = info

                total = len(samples_to_process)

                for i, (sample_rep, info) in enumerate(samples_to_process.items()):
                    progress = int(100 * (i + 1) / max(total, 1))

                    # Separar mostra i rèplica del Sample_Rep (ex: "FR2606_R1")
                    base_name, rep = split_sample_rep(sample_rep)
                    if rep is None:
                        rep = "1"

                    mostra_clean = clean_sample_name(base_name)

                    if progress_callback:
                        progress_callback(progress, mostra_clean)

                    row_ini = info.get("row_start")
                    row_fi = info.get("row_end")
                    t_start = info.get("t_start")

                    if row_ini is None or row_fi is None:
                        continue

                    df_doc, timeout_info = extract_doc_from_masterfile(df_toc, row_ini, row_fi, t_start, detect_timeouts=True)

                    # Afegir warning si hi ha timeout en zona crítica
                    if timeout_info and timeout_info.get("severity") in ["WARNING", "CRITICAL"]:
                        result["errors"].append(f"TIMEOUT: {mostra_clean} - {timeout_info.get('warning_message', '')}")

                    # Determinar si tenim dades DOC
                    if df_doc.empty or len(df_doc) < 10:
                        # No hi ha dades DOC directe - crear fitxer amb NO_DATA
                        st_doc = "NO_DATA"
                        t_doc = np.array([0.0])
                        y_doc = np.array([0.0])
                        doc_net = np.array([0.0])
                        base_arr = np.array([0.0])
                        peak_info = {"valid": False, "t_max": 0, "area": 0, "t_start": 0, "t_end": 0}
                        sample_analysis = {}
                    else:
                        st_doc = "OK"
                        t_doc = pd.to_numeric(df_doc["time (min)"], errors="coerce").to_numpy()
                        y_doc = pd.to_numeric(df_doc["DOC"], errors="coerce").to_numpy()

                        # APLICAR SHIFT D'ALINEAMENT (si calculat)
                        # El shift alinea DOC amb A254: t_corregit = t_original + shift
                        if direct_shift != 0.0:
                            t_doc = t_doc + direct_shift

                        base_arr = get_baseline_correction(t_doc, y_doc, method, config)
                        doc_net = apply_smoothing(y_doc - base_arr)
                        doc_net[doc_net < 0] = 0
                        peak_info = detect_main_peak(t_doc, doc_net, config["peak_min_prominence_pct"])
                        sample_analysis = {}  # Es calcularà després amb DAD

                    # Buscar DAD
                    dad_path, st_dad = choose_best_candidate(base_name, rep, dad_pool, used_dad_files)
                    df_dad = pd.DataFrame()
                    nom_dad = "NO_DATA"

                    if dad_path:
                        used_dad_files.add(dad_path)
                        nom_dad = os.path.basename(dad_path)
                        df_dad_raw, st_read = llegir_dad_export3d(dad_path)
                        if st_read.startswith("OK") and not df_dad_raw.empty:
                            df_dad = process_dad(df_dad_raw, config)
                            st_dad = "OK" if not df_dad.empty else "NO_DATA"

                    # =========================================
                    # BP MODE DIRECT (NEW): Alinear cada mostra a A254
                    # =========================================
                    if method == "BP" and st_doc == "OK" and not df_dad.empty and '254' in df_dad.columns:
                        try:
                            t_dad_sample = df_dad['time (min)'].values
                            y_a254_sample = pd.to_numeric(df_dad['254'], errors='coerce').to_numpy()

                            if len(y_a254_sample) > 10 and np.nanmax(y_a254_sample) > 0:
                                t_max_a254_sample = t_dad_sample[np.argmax(y_a254_sample)]
                                t_max_doc_sample = t_doc[np.argmax(doc_net)]

                                shift_bp = t_max_a254_sample - t_max_doc_sample
                                tolerance_min = 2.0 / 60.0  # 2 segons

                                if abs(shift_bp) > tolerance_min:
                                    # Aplicar shift: interpolar DOC a eix DAD
                                    t_doc_shifted = t_doc + shift_bp
                                    doc_net = np.interp(t_dad_sample, t_doc_shifted, doc_net, left=0, right=0)
                                    y_doc = np.interp(t_dad_sample, t_doc_shifted, y_doc, left=0, right=0)
                                    base_arr = np.interp(t_dad_sample, t_doc_shifted, base_arr, left=base_arr[0], right=base_arr[-1])
                                    t_doc = t_dad_sample

                                    # Recalcular peak_info amb dades alineades
                                    peak_info = detect_main_peak(t_doc, doc_net, config["peak_min_prominence_pct"])
                        except Exception as e:
                            result["warnings"].append(f"WARN: Error alineament BP per {mostra_clean}: {e}")

                    # Calcular àrees (només si tenim DOC)
                    if st_doc == "OK":
                        sample_analysis = analyze_sample_areas(t_doc, doc_net, df_dad, peak_info, config)

                    # Guardar Excel
                    out_name = f"{mostra_clean}_{seq_out}_R{rep}.xlsx"
                    out_path = os.path.join(path_out, out_name)

                    # Calcular SNR i baseline noise (amb finestres per evitar timeouts)
                    timeout_positions = []
                    if timeout_info and timeout_info.get("timeouts"):
                        timeout_positions = [to["t_start_min"] for to in timeout_info["timeouts"]]

                    snr_info = calculate_snr_info(
                        doc_net, peak_info,
                        t_min=t_doc,
                        method=method,
                        timeout_positions=timeout_positions
                    )

                    write_consolidated_excel(
                        out_path, mostra_clean, rep, seq_out, date_master, method, "DIRECT",
                        os.path.basename(mestre), nom_dad, st_doc, st_dad, t_doc, y_doc, doc_net, base_arr,
                        df_dad, peak_info, sample_analysis,
                        master_file=mestre, row_start=row_ini, row_end=row_fi,
                        smoothing_applied=True,
                        master_info=master_info,
                        timeout_info=timeout_info,
                        snr_info=snr_info
                    )

                    result["files"].append(out_path)
                    processed_count += 1

                    # Recollir estadístiques de la mostra
                    file_info = {
                        "file_dad": nom_dad if nom_dad else os.path.basename(mestre),
                        "file_uib": "",
                        "row_start": row_ini,
                        "row_end": row_fi,
                        "npts": len(t_doc) if t_doc is not None else 0,
                        "match_confidence": 100.0,  # DIRECT: match exacte des de MasterFile
                        "match_type": "masterfile",
                    }
                    sample_stat = _collect_sample_stats(mostra_clean, rep, timeout_info, snr_info, peak_info, "DIRECT", file_info)
                    result["sample_stats"].append(sample_stat)

            else:
                # ============================================================
                # FORMAT ANTIC: Mestre amb 4-SEQ_DATA (master_info buit)
                # ============================================================
                df_toc, df_seq = llegir_master_direct(mestre)

                needed = ["SAMPLE_DESC", "START", "Row initial", "Row Final"]
                for col in needed:
                    if col not in df_seq.columns:
                        result["errors"].append(f"Falta columna: {col}")
                        return result

                df_seq2 = df_seq.copy()
                df_seq2 = df_seq2[df_seq2["SAMPLE_DESC"].apply(lambda x: not skip_sample_direct(x))]
                df_seq2 = df_seq2.dropna(subset=["Row initial", "Row Final"])

                total = len(df_seq2)
                rep_counter = {}

                for i, (_, row) in enumerate(df_seq2.iterrows()):
                    progress = int(100 * (i + 1) / max(total, 1))
                    sample_desc = str(row["SAMPLE_DESC"]).strip()
                    base_name, rep_explicit = split_sample_rep(sample_desc)
                    base_key = normalize_key(base_name)

                    if rep_explicit is None:
                        rep_counter[base_key] = rep_counter.get(base_key, 0) + 1
                        rep = str(rep_counter[base_key])
                    else:
                        rep = str(rep_explicit)

                    mostra_clean = clean_sample_name(base_name)

                    if progress_callback:
                        progress_callback(progress, mostra_clean)

                    try:
                        row_ini = int(float(row["Row initial"]))
                        row_fi = int(float(row["Row Final"]))
                    except Exception:
                        continue

                    df_doc, timeout_info = extract_doc_from_master(df_toc, row_ini, row_fi, row["START"], detect_timeouts=True)

                    # Afegir warning si hi ha timeout en zona crítica
                    if timeout_info and timeout_info.get("severity") in ["WARNING", "CRITICAL"]:
                        result["errors"].append(f"TIMEOUT: {mostra_clean} - {timeout_info.get('warning_message', '')}")

                    # Determinar si tenim dades DOC
                    if df_doc.empty or len(df_doc) < 10:
                        st_doc = "NO_DATA"
                        t_doc = np.array([0.0])
                        y_doc = np.array([0.0])
                        doc_net = np.array([0.0])
                        base_arr = np.array([0.0])
                        peak_info = {"valid": False, "t_max": 0, "area": 0, "t_start": 0, "t_end": 0}
                        sample_analysis = {}
                    else:
                        st_doc = "OK"
                        t_doc = pd.to_numeric(df_doc["time (min)"], errors="coerce").to_numpy()
                        y_doc = pd.to_numeric(df_doc["DOC"], errors="coerce").to_numpy()
                        base_arr = get_baseline_correction(t_doc, y_doc, method, config)
                        doc_net = apply_smoothing(y_doc - base_arr)
                        doc_net[doc_net < 0] = 0
                        peak_info = detect_main_peak(t_doc, doc_net, config["peak_min_prominence_pct"])
                        sample_analysis = {}

                    # Buscar DAD
                    dad_path, st_dad = choose_best_candidate(base_name, rep, dad_pool, used_dad_files)
                    df_dad = pd.DataFrame()
                    nom_dad = "NO_DATA"

                    if dad_path:
                        used_dad_files.add(dad_path)
                        nom_dad = os.path.basename(dad_path)
                        df_dad_raw, st_read = llegir_dad_export3d(dad_path)
                        if st_read.startswith("OK") and not df_dad_raw.empty:
                            df_dad = process_dad(df_dad_raw, config)
                            st_dad = "OK" if not df_dad.empty else "NO_DATA"

                    # =========================================
                    # BP MODE DIRECT (ANTIC): Alinear cada mostra a A254
                    # =========================================
                    if method == "BP" and st_doc == "OK" and not df_dad.empty and '254' in df_dad.columns:
                        try:
                            t_dad_sample = df_dad['time (min)'].values
                            y_a254_sample = pd.to_numeric(df_dad['254'], errors='coerce').to_numpy()

                            if len(y_a254_sample) > 10 and np.nanmax(y_a254_sample) > 0:
                                t_max_a254_sample = t_dad_sample[np.argmax(y_a254_sample)]
                                t_max_doc_sample = t_doc[np.argmax(doc_net)]

                                shift_bp = t_max_a254_sample - t_max_doc_sample
                                tolerance_min = 2.0 / 60.0  # 2 segons

                                if abs(shift_bp) > tolerance_min:
                                    # Aplicar shift: interpolar DOC a eix DAD
                                    t_doc_shifted = t_doc + shift_bp
                                    doc_net = np.interp(t_dad_sample, t_doc_shifted, doc_net, left=0, right=0)
                                    y_doc = np.interp(t_dad_sample, t_doc_shifted, y_doc, left=0, right=0)
                                    base_arr = np.interp(t_dad_sample, t_doc_shifted, base_arr, left=base_arr[0], right=base_arr[-1])
                                    t_doc = t_dad_sample

                                    # Recalcular peak_info amb dades alineades
                                    peak_info = detect_main_peak(t_doc, doc_net, config["peak_min_prominence_pct"])
                        except Exception as e:
                            result["warnings"].append(f"WARN: Error alineament BP per {mostra_clean}: {e}")

                    # Calcular àrees (només si tenim DOC)
                    if st_doc == "OK":
                        sample_analysis = analyze_sample_areas(t_doc, doc_net, df_dad, peak_info, config)

                    # Guardar Excel
                    out_name = f"{mostra_clean}_{seq_out}_R{rep}.xlsx"
                    out_path = os.path.join(path_out, out_name)

                    # Calcular SNR i baseline noise (amb finestres per evitar timeouts)
                    timeout_positions = []
                    if timeout_info and timeout_info.get("timeouts"):
                        timeout_positions = [to["t_start_min"] for to in timeout_info["timeouts"]]

                    snr_info = calculate_snr_info(
                        doc_net, peak_info,
                        t_min=t_doc,
                        method=method,
                        timeout_positions=timeout_positions
                    )

                    write_consolidated_excel(
                        out_path, mostra_clean, rep, seq_out, date_master, method, "DIRECT",
                        os.path.basename(mestre), nom_dad, st_doc, st_dad, t_doc, y_doc, doc_net, base_arr,
                        df_dad, peak_info, sample_analysis,
                        master_file=mestre, row_start=row_ini, row_end=row_fi,
                        smoothing_applied=True,
                        master_info={},  # Format antic no té 0-INFO
                        timeout_info=timeout_info,
                        snr_info=snr_info
                    )

                    result["files"].append(out_path)
                    processed_count += 1

                    # Recollir estadístiques de la mostra
                    file_info = {
                        "file_dad": nom_dad if nom_dad else os.path.basename(mestre),
                        "file_uib": "",
                        "row_start": row_ini,
                        "row_end": row_fi,
                        "npts": len(t_doc) if t_doc is not None else 0,
                        "match_confidence": 100.0,  # DIRECT: match exacte des de MasterFile
                        "match_type": "masterfile",
                    }
                    sample_stat = _collect_sample_stats(mostra_clean, rep, timeout_info, snr_info, peak_info, "DIRECT", file_info)
                    result["sample_stats"].append(sample_stat)

        result["processed_count"] = processed_count
        result["success"] = processed_count > 0

        # Guardar fitxers orfes (no pertanyen a la SEQ)
        if 'orphan_files' in dir() and orphan_files:
            result["orphan_files_seq"] = orphan_files

        # Guardar matches amb confiança baixa
        if 'low_confidence_matches' in dir() and low_confidence_matches:
            result["low_confidence_matches"] = low_confidence_matches

        # Guardar detalls de matching
        if 'match_details' in dir() and match_details:
            result["match_details"] = match_details

        # Verificació de fitxers (QA/QC)
        file_check = check_sequence_files(input_folder, used_uib_files, used_dad_files)
        # Fitxers orfes (no pertanyen a la SEQ segons 1-HPLC-SEQ)
        seq_orphan_files = result.get("orphan_files_seq", [])

        # Fitxers amb confiança baixa
        low_conf_matches = result.get("low_confidence_matches", [])

        result["file_check"] = {
            "uib_found": file_check["uib"]["count_found"],
            "uib_used": file_check["uib"]["count_used"],
            "uib_orphan": file_check["uib"]["count_orphan"],
            "dad_found": file_check["dad"]["count_found"],
            "dad_used": file_check["dad"]["count_used"],
            "dad_orphan": file_check["dad"]["count_orphan"],
            "has_issues": file_check["has_issues"] or len(seq_orphan_files) > 0 or len(low_conf_matches) > 0,
            "orphan_files": file_check["uib"]["orphan"] + file_check["dad"]["orphan"],
            "seq_orphan_files": seq_orphan_files,  # Fitxers que no pertanyen a la SEQ
            "low_confidence_matches": low_conf_matches,  # Matches amb confiança < 85%
            "needs_review": len(seq_orphan_files) > 0 or len(low_conf_matches) > 0,
            "proposed_renames": file_check["proposed_renames"],
        }

        # Guardar informe si hi ha problemes
        if file_check["has_issues"]:
            report_path = save_file_check_report(input_folder, file_check)
            if report_path:
                result["file_check"]["report_path"] = report_path

            # Afegir warnings al resultat
            if file_check["uib"]["count_orphan"] > 0:
                result["errors"].append(
                    f"WARN: {file_check['uib']['count_orphan']} fitxers UIB no processats: "
                    f"{', '.join(file_check['uib']['orphan'][:3])}{'...' if file_check['uib']['count_orphan'] > 3 else ''}"
                )
            if file_check["dad"]["count_orphan"] > 0:
                result["errors"].append(
                    f"WARN: {file_check['dad']['count_orphan']} fitxers DAD no processats: "
                    f"{', '.join(file_check['dad']['orphan'][:3])}{'...' if file_check['dad']['count_orphan'] > 3 else ''}"
                )
            if file_check["proposed_renames"]:
                result["errors"].append(
                    f"WARN: {len(file_check['proposed_renames'])} possibles errors de nomenclatura detectats"
                )

        # Warnings per problemes de matching
        if len(seq_orphan_files) > 0:
            result["errors"].append(
                f"REVIEW: {len(seq_orphan_files)} fitxers no pertanyen a aquesta SEQ (segons 1-HPLC-SEQ)"
            )
        if len(low_conf_matches) > 0:
            result["errors"].append(
                f"REVIEW: {len(low_conf_matches)} matches amb confiança baixa - cal verificar assignacions"
            )

        # === GENERAR PDFs DE CONSOLIDACIÓ ===
        if result["processed_count"] > 0:
            result["pdf_paths"] = []

            # Recollir fitxers consolidats
            xlsx_files = glob.glob(os.path.join(path_out, "*.xlsx"))
            xlsx_files = [f for f in xlsx_files if "~$" not in f]

            if xlsx_files:
                # Crear carpeta CHECK per als PDFs
                check_folder = os.path.join(input_folder, "CHECK")
                os.makedirs(check_folder, exist_ok=True)

                info = {
                    "seq": result.get("seq", ""),
                    "date": result.get("date", ""),
                    "mode": result.get("mode", ""),
                    "bp": result.get("bp", False),
                }

                # 1. PDF Consolidació (recompte punts)
                try:
                    from hpsec_reports import generate_consolidation_report
                    pdf_path = generate_consolidation_report(input_folder, xlsx_files, info, check_folder)
                    if pdf_path and os.path.exists(pdf_path):
                        result["pdf_paths"].append(pdf_path)
                        result["warnings"].append(f"PDF generat: {os.path.basename(pdf_path)}")
                except Exception as e:
                    result["warnings"].append(f"No s'ha pogut generar PDF consolidació: {e}")

                # 2. PDF Cromatogrames
                try:
                    from hpsec_reports import generate_chromatograms_report
                    pdf_path = generate_chromatograms_report(input_folder, xlsx_files, info, check_folder)
                    if pdf_path and os.path.exists(pdf_path):
                        result["pdf_paths"].append(pdf_path)
                        result["warnings"].append(f"PDF generat: {os.path.basename(pdf_path)}")
                except Exception as e:
                    result["warnings"].append(f"No s'ha pogut generar PDF cromatogrames: {e}")

        # === GENERAR JSON RESUM CONSOLIDACIÓ ===
        if result.get("sample_stats"):
            try:
                timing_stats = result.get("timing_stats")
                summary = generate_consolidation_summary(result, result["sample_stats"], timing_stats)
                json_path = save_consolidation_summary(input_folder, summary)
                if json_path:
                    result["consolidation_json"] = json_path
                    result["consolidation_summary"] = summary
            except Exception as e:
                result["warnings"].append(f"No s'ha pogut generar consolidation.json: {e}")

    except Exception as e:
        import traceback
        result["errors"].append(str(e))
        result["errors"].append(traceback.format_exc())

    return result


# =============================================================================
# TEST STANDALONE
# =============================================================================
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        seq_path = sys.argv[1]
        print(f"Consolidant: {seq_path}")

        def progress(pct, sample):
            print(f"  [{pct:3d}%] {sample}")

        result = consolidate_sequence(seq_path, progress_callback=progress)

        print(f"\nResultat:")
        print(f"  Success: {result['success']}")
        print(f"  Mode: {result['mode']}")
        print(f"  BP: {result['bp']}")
        print(f"  SEQ: {result['seq']}")
        print(f"  Processats: {result['processed_count']}")
        print(f"  Fitxers: {len(result['files'])}")
        if result['errors']:
            print(f"  Errors: {result['errors']}")
    else:
        print("Ús: python hpsec_consolidate.py <seq_folder>")
