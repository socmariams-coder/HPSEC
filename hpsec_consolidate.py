"""
hpsec_consolidate.py - Mòdul de consolidació de dades HPSEC
===========================================================

Conté tota la lògica per:
- Llegir fitxers DOC (UIB CSV o Excel mestre)
- Llegir fitxers DAD (Export3D)
- Processar i consolidar dades
- Generar Excels consolidats

Usat per HPSEC_Suite.py i batch_process.py
"""

import os
import re
import glob
import json
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
from scipy.integrate import trapezoid


# =============================================================================
# LOG D'ALINEACIÓ (QA/QC)
# =============================================================================
QAQC_FOLDER = "HPSEC_QAQC"
ALIGNMENT_LOG_FILE = "Alignment_History.json"


def get_qaqc_folder(base_folder):
    """Retorna el path a la carpeta QAQC (a la carpeta pare de les SEQs)."""
    parent = os.path.dirname(os.path.normpath(base_folder))
    qaqc_path = os.path.join(parent, QAQC_FOLDER)
    return qaqc_path


def ensure_qaqc_folder(base_folder):
    """Crea la carpeta QAQC si no existeix i retorna el path."""
    qaqc_path = get_qaqc_folder(base_folder)
    os.makedirs(qaqc_path, exist_ok=True)
    return qaqc_path


def get_alignment_log_path(base_folder):
    """Retorna el path al fitxer de log d'alineació."""
    qaqc_path = get_qaqc_folder(base_folder)
    return os.path.join(qaqc_path, ALIGNMENT_LOG_FILE)


def load_alignment_log(base_folder):
    """Carrega el log d'alineació."""
    log_path = get_alignment_log_path(base_folder)
    if os.path.exists(log_path):
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {"entries": [], "version": "1.0"}


def save_alignment_log(base_folder, log_data):
    """Guarda el log d'alineació."""
    ensure_qaqc_folder(base_folder)  # Crear carpeta si no existeix
    log_path = get_alignment_log_path(base_folder)
    try:
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False


def add_alignment_entry(base_folder, seq_name, seq_date, method, shift_uib, shift_direct, khp_file, details=None):
    """
    Afegeix una entrada al log d'alineació.

    Args:
        base_folder: Carpeta de la SEQ
        seq_name: Nom de la SEQ (ex: "275", "284_BP")
        seq_date: Data de la SEQ
        method: "BP" o "COLUMN"
        shift_uib: Shift aplicat a DOC_UIB (minuts)
        shift_direct: Shift aplicat a DOC_Direct (minuts)
        khp_file: Nom del fitxer KHP usat
        details: Dict amb detalls adicionals
    """
    log_data = load_alignment_log(base_folder)

    entry = {
        "timestamp": datetime.now().isoformat(),
        "seq_name": seq_name,
        "seq_date": str(seq_date) if seq_date else "",
        "method": method,
        "shift_uib_min": shift_uib,
        "shift_uib_sec": shift_uib * 60,
        "shift_direct_min": shift_direct,
        "shift_direct_sec": shift_direct * 60,
        "khp_file": khp_file,
        "details": details or {}
    }

    # Evitar duplicats: actualitzar si ja existeix entrada per aquesta SEQ
    existing_idx = None
    for i, e in enumerate(log_data["entries"]):
        if e.get("seq_name") == seq_name:
            existing_idx = i
            break

    if existing_idx is not None:
        log_data["entries"][existing_idx] = entry
    else:
        log_data["entries"].append(entry)

    save_alignment_log(base_folder, log_data)
    return entry


def find_nearest_alignment(base_folder, method, seq_date=None):
    """
    Busca l'alineació més propera en el temps per usar quan no hi ha KHP.

    Args:
        base_folder: Carpeta de la SEQ
        method: "BP" o "COLUMN" (només busca del mateix mètode)
        seq_date: Data de la SEQ actual (per trobar la més propera)

    Returns:
        Dict amb shift_uib, shift_direct, source_seq o None si no es troba
    """
    log_data = load_alignment_log(base_folder)

    # Filtrar per mètode
    candidates = [e for e in log_data.get("entries", []) if e.get("method") == method]

    if not candidates:
        return None

    # Si tenim data, ordenar per proximitat temporal
    if seq_date:
        try:
            target_date = pd.to_datetime(seq_date)
            for c in candidates:
                c_date = pd.to_datetime(c.get("seq_date", ""), errors='coerce')
                if pd.notna(c_date):
                    c["_date_diff"] = abs((target_date - c_date).days)
                else:
                    c["_date_diff"] = 9999
            candidates.sort(key=lambda x: x.get("_date_diff", 9999))
        except Exception:
            pass

    # Retornar el més proper (o el més recent si no hi ha dates)
    if candidates:
        best = candidates[0]
        return {
            "shift_uib": best.get("shift_uib_min", 0.0),
            "shift_direct": best.get("shift_direct_min", 0.0),
            "source_seq": best.get("seq_name", ""),
            "source_date": best.get("seq_date", ""),
            "source_khp": best.get("khp_file", "")
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
}


# =============================================================================
# FUNCIONS UTILITAT
# =============================================================================
def normalize_key(s):
    """Normalitza string per matching."""
    return re.sub(r"[^A-Za-z0-9]+", "", str(s or "")).upper()


def is_khp(name):
    """Detecta si és mostra KHP."""
    return "KHP" in str(name).upper()


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


def find_khp_for_alignment(seq_folder):
    """
    Cerca KHP per alineació: LOCAL → SIBLINGS.

    Args:
        seq_folder: Carpeta de la SEQ

    Returns:
        (khp_path, source) o (None, None) si no es troba
    """
    # FASE 1: LOCAL
    res_cons = os.path.join(seq_folder, "Resultats_Consolidats")
    khp_files = find_khp_in_folder(res_cons)
    if khp_files:
        return khp_files[0], "LOCAL"

    # FASE 2: SIBLINGS
    folder_name = os.path.basename(seq_folder)
    parent_dir = os.path.dirname(seq_folder)

    match = re.match(r'^(\d+)', folder_name)
    if match:
        seq_id = match.group(1)
        try:
            all_folders = [d for d in os.listdir(parent_dir)
                          if os.path.isdir(os.path.join(parent_dir, d))]
            siblings = [d for d in all_folders
                       if d.startswith(seq_id) and d != folder_name]

            for sib in siblings:
                sib_res_cons = os.path.join(parent_dir, sib, "Resultats_Consolidats")
                khp_files = find_khp_in_folder(sib_res_cons)
                if khp_files:
                    return khp_files[0], f"SIBLING:{sib}"
        except Exception:
            pass

    return None, None


def calculate_column_alignment_shifts(khp_path, config=None, tolerance_sec=2.0):
    """
    Calcula els shifts d'alineació per COLUMN usant KHP com a referència.

    Protocol:
    1. A254 (DAD) és la referència absoluta
    2. Calcular shift_uib = t_max(A254) - t_max(DOC_UIB)
    3. Calcular shift_direct = t_max(A254) - t_max(DOC_Direct)

    Args:
        khp_path: Path al fitxer KHP consolidat
        config: Configuració
        tolerance_sec: Tolerància en segons per considerar alineat (default 2s)

    Returns:
        dict amb:
            - shift_uib: shift per DOC_UIB (minuts)
            - shift_direct: shift per DOC_Direct (minuts)
            - aligned: True si ja estaven alineats (shifts < tolerance)
            - details: dict amb detalls dels càlculs
    """
    result = {
        "shift_uib": 0.0,
        "shift_direct": 0.0,
        "aligned": True,
        "details": {}
    }

    tolerance_min = tolerance_sec / 60.0

    try:
        xls = pd.ExcelFile(khp_path, engine="openpyxl")

        # Llegir DOC
        if "DOC" not in xls.sheet_names:
            return result

        df_doc = pd.read_excel(xls, "DOC", engine="openpyxl")

        t = df_doc["time (min)"].values

        # Verificar que tenim les columnes necessàries
        has_direct = "DOC (mAU)" in df_doc.columns
        has_uib = "DOC_UIB (mAU)" in df_doc.columns

        if not has_direct and not has_uib:
            return result

        # Llegir DAD per A254
        if "DAD" not in xls.sheet_names:
            return result

        df_dad = pd.read_excel(xls, "DAD", engine="openpyxl")

        if "254" not in df_dad.columns and "254.0" not in df_dad.columns:
            return result

        t_dad = df_dad["time (min)"].values
        a254_col = "254" if "254" in df_dad.columns else "254.0"
        y_a254 = df_dad[a254_col].values

        # Trobar màxim A254 (referència)
        idx_max_a254 = np.argmax(y_a254)
        t_max_a254 = t_dad[idx_max_a254]

        result["details"]["t_max_a254"] = t_max_a254

        # Calcular shift per DOC_UIB
        if has_uib:
            y_uib = df_doc["DOC_UIB (mAU)"].values
            idx_max_uib = np.argmax(y_uib)
            t_max_uib = t[idx_max_uib]

            shift_uib = t_max_a254 - t_max_uib
            result["details"]["t_max_uib"] = t_max_uib
            result["details"]["shift_uib_raw"] = shift_uib

            if abs(shift_uib) > tolerance_min:
                result["shift_uib"] = shift_uib
                result["aligned"] = False

        # Calcular shift per DOC_Direct
        if has_direct:
            y_direct = df_doc["DOC (mAU)"].values
            idx_max_direct = np.argmax(y_direct)
            t_max_direct = t[idx_max_direct]

            shift_direct = t_max_a254 - t_max_direct
            result["details"]["t_max_direct"] = t_max_direct
            result["details"]["shift_direct_raw"] = shift_direct

            if abs(shift_direct) > tolerance_min:
                result["shift_direct"] = shift_direct
                result["aligned"] = False

    except Exception as e:
        result["details"]["error"] = str(e)

    return result


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


def extract_doc_from_masterfile(toc_df, row_start, row_end, t_start=None):
    """
    Extreu segment DOC del nou MasterFile.

    Args:
        toc_df: DataFrame de 2-TOC
        row_start: Fila inicial (1-indexed, relatiu a fila 8 del TOC original)
        row_end: Fila final
        t_start: Timestamp d'inici (opcional, per calcular temps relatiu)

    Returns:
        DataFrame amb columns "time (min)" i "DOC"
    """
    if toc_df is None or toc_df.empty:
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

    return df_doc


def llegir_master_direct(mestre):
    """Llegeix TOC i SEQ_DATA del mestre."""
    df_toc = pd.read_excel(mestre, sheet_name="2-TOC", header=6, engine="openpyxl")
    df_seq = pd.read_excel(mestre, sheet_name="4-SEQ_DATA", header=0, engine="openpyxl")
    return df_toc, df_seq


def extract_doc_from_master(df_toc, row_ini, row_fi, start_dt):
    """Extreu segment DOC del mestre."""
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
                             master_info=None):
    """
    Escriu fitxer Excel consolidat amb àrees calculades.

    Suporta dual protocol: DOC (Direct) + DOC_UIB quan ambdós estan disponibles.
    Sempre guarda dades RAW i documenta el processament aplicat.

    Estructura fulla ID:
      1. IDENTIFICACIÓ: Sample, Replica, SEQ, Method, Date, Inj_Volume, UIB_range
      2. FITXERS: File_MasterFile, File_DAD, File_DOC_UIB
      3. ESTAT: DOC_Mode, Status_DOC, Status_DAD, Peak_Valid
      4. DOC: N_Points, tmax, t_start, t_end, Area, Row_Start, Row_End
      5. PROCESSAMENT: Baseline_Method, Baseline_mAU, Smoothing, Shift_sec
      6. UIB (si DUAL): Baseline_mAU, Shift_sec, Status
      7. DAD (si processat): N_Points, àrees per longitud d'ona
    """
    sample_analysis = sample_analysis or {}
    master_info = master_info or {}
    is_dual = y_doc_uib is not None and len(y_doc_uib) > 0

    # Calcular valors de baseline per documentar
    baseline_direct_val = float(np.mean(baseline)) if baseline is not None and len(baseline) > 0 else 0.0
    baseline_uib_val = float(np.mean(baseline_uib)) if baseline_uib is not None and len(baseline_uib) > 0 else 0.0

    # === 1. IDENTIFICACIÓ ===
    id_rows = [
        ("Sample", mostra),
        ("Replica", rep),
        ("SEQ", seq_out),
        ("Method", method),
        ("Date", date_master),
        ("Inj_Volume_uL", master_info.get("Inj_Volume (uL)", "")),
        ("UIB_range", master_info.get("UIB_range", "None")),
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

    # === 4. MESURES DOC ===
    id_rows.extend([
        ("DOC_N_Points", len(t_doc)),
        ("DOC_tmax", round(sample_analysis.get("doc_t_retention", peak_info.get("t_max", 0.0)), 3)),
        ("DOC_t_start", round(sample_analysis.get("doc_t_start", peak_info.get("t_start", 0.0)), 3)),
        ("DOC_t_end", round(sample_analysis.get("doc_t_end", peak_info.get("t_end", 0.0)), 3)),
        ("DOC_Area", round(sample_analysis.get("doc_area", peak_info.get("area", 0.0)), 3)),
    ])
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

    # === 7. DAD (només si processat i té àrees) ===
    dad_processed = not df_dad.empty and any(
        sample_analysis.get(f"a{wl}_area", 0.0) > 0
        for wl in [220, 252, 254, 272, 290, 362]
    )
    if dad_processed:
        id_rows.append(("DAD_N_Points", len(df_dad)))
        for wl in [220, 252, 254, 272, 290, 362]:
            area = sample_analysis.get(f"a{wl}_area", 0.0)
            if area > 0:
                id_rows.append((f"DAD_A{wl}_Area", round(area, 3)))

    df_id = pd.DataFrame(id_rows, columns=["Camp", "Valor"])

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
    }

    try:
        input_folder = seq_path
        path_csv = os.path.join(input_folder, "CSV")
        path_3d = os.path.join(input_folder, "Export3d")

        # Fallback a Export3D si Export3d no existeix
        if not os.path.isdir(path_3d):
            path_3d = os.path.join(input_folder, "Export3D")

        path_out = os.path.join(input_folder, "Resultats_Consolidats")
        os.makedirs(path_out, exist_ok=True)
        result["output_path"] = path_out

        # Detectar mode UIB vs DIRECT
        uib_files = []
        if os.path.isdir(path_csv):
            for ext in ("*.csv", "*.CSV", "*.txt", "*.TXT"):
                uib_files.extend(glob.glob(os.path.join(path_csv, f"*UIB1B*{ext}")))

        mode = "UIB" if uib_files else "DIRECT"
        result["mode"] = mode

        # Detectar BP
        dad_pool = list_dad_files(
            path_3d if os.path.isdir(path_3d) else None,
            path_csv if os.path.isdir(path_csv) else None,
        )
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
            has_master = False
            master_info = {}  # Info de 0-INFO per passar a write_consolidated_excel

            if mestre:
                result["master_format"] = master_format

                if master_format == "NEW":
                    # FORMAT NOU: MasterFile amb 4-TOC_CALC
                    try:
                        master_data = llegir_masterfile_nou(mestre)
                        if "error" not in master_data:
                            df_toc = master_data["toc"]
                            df_toc_calc = master_data["toc_calc"]
                            master_info = master_data.get("info", {})  # 0-INFO

                            if df_toc is not None and df_toc_calc is not None:
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
            # ================================================================
            column_shift_uib = 0.0
            column_shift_direct = 0.0
            alignment_info = None

            if method == "COLUMN" and has_master:
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

                        # Llegir DOC Direct del KHP
                        khp_match = match_uib_to_master(khp_mostra, khp_rep, master_index)
                        t_khp_direct = None
                        y_khp_direct_net = None

                        if khp_match:
                            khp_row = khp_match["row"]
                            khp_row_ini = int(float(khp_row["Row initial"]))
                            khp_row_fi = int(float(khp_row["Row Final"]))
                            df_khp_direct = extract_doc_from_master(df_toc, khp_row_ini, khp_row_fi, khp_row["START"])

                            if not df_khp_direct.empty:
                                t_khp_direct = pd.to_numeric(df_khp_direct["time (min)"], errors="coerce").to_numpy()
                                y_khp_direct = pd.to_numeric(df_khp_direct["DOC"], errors="coerce").to_numpy()
                                base_khp_direct = get_baseline_correction(t_khp_direct, y_khp_direct, method, config)
                                y_khp_direct_net = apply_smoothing(y_khp_direct - base_khp_direct)
                                y_khp_direct_net[y_khp_direct_net < 0] = 0

                        # Buscar DAD del KHP per A254
                        khp_dad_path, _ = choose_best_candidate(khp_mostra, khp_rep, dad_pool, set())
                        t_khp_a254 = None
                        y_khp_a254 = None

                        if khp_dad_path:
                            df_khp_dad, st = llegir_dad_export3d(khp_dad_path)
                            if st.startswith("OK") and not df_khp_dad.empty:
                                t_khp_a254 = pd.to_numeric(df_khp_dad["time (min)"], errors="coerce").to_numpy()
                                a254_col = "254" if "254" in df_khp_dad.columns else "254.0" if "254.0" in df_khp_dad.columns else None
                                if a254_col:
                                    y_khp_a254 = pd.to_numeric(df_khp_dad[a254_col], errors="coerce").to_numpy()

                        # Calcular shifts: A254 és la referència
                        tolerance_min = 2.0 / 60.0  # 2 segons

                        if t_khp_a254 is not None and y_khp_a254 is not None:
                            t_max_a254 = t_khp_a254[np.argmax(y_khp_a254)]

                            # Shift per DOC_UIB
                            t_max_uib = t_khp_uib[np.argmax(y_khp_uib_net)]
                            shift_uib = t_max_a254 - t_max_uib
                            if abs(shift_uib) > tolerance_min:
                                column_shift_uib = shift_uib

                            # Shift per DOC_Direct
                            if t_khp_direct is not None and y_khp_direct_net is not None:
                                t_max_direct = t_khp_direct[np.argmax(y_khp_direct_net)]
                                shift_direct = t_max_a254 - t_max_direct
                                if abs(shift_direct) > tolerance_min:
                                    column_shift_direct = shift_direct

                            alignment_info = {
                                "khp_file": khp_nom,
                                "t_max_a254": t_max_a254,
                                "t_max_uib": t_max_uib,
                                "t_max_direct": t_max_direct if t_khp_direct is not None else None,
                                "shift_uib": column_shift_uib,
                                "shift_direct": column_shift_direct,
                                "source": "KHP_LOCAL",
                            }
                            result["alignment"] = alignment_info

                            # Guardar al log QAQC
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
                                    "t_max_direct": float(t_max_direct) if t_khp_direct is not None else None,
                                }
                            )

                    except Exception as e:
                        result["errors"].append(f"WARN: Error calculant alignment amb KHP: {e}")

                # Si no s'ha pogut calcular alignment amb KHP local, buscar en el log històric
                if alignment_info is None:
                    nearest = find_nearest_alignment(input_folder, method, date_master)
                    if nearest:
                        column_shift_uib = nearest.get("shift_uib", 0.0)
                        column_shift_direct = nearest.get("shift_direct", 0.0)
                        alignment_info = {
                            "khp_file": nearest.get("source_khp", ""),
                            "shift_uib": column_shift_uib,
                            "shift_direct": column_shift_direct,
                            "source": f"HISTORIC:{nearest.get('source_seq', '')}",
                            "source_date": nearest.get("source_date", ""),
                        }
                        result["alignment"] = alignment_info
                        result["errors"].append(
                            f"INFO: Usant alignment històric de SEQ {nearest.get('source_seq', '')} "
                            f"(shift_uib={column_shift_uib*60:.1f}s, shift_direct={column_shift_direct*60:.1f}s)"
                        )

            for i, f_doc_uib in enumerate(uib_files):
                progress = int(100 * (i + 1) / max(total, 1))
                nom_doc_uib = os.path.basename(f_doc_uib)
                mostra, rep = netejar_nom_uib(nom_doc_uib)

                if progress_callback:
                    progress_callback(progress, mostra)

                # Llegir DOC UIB
                df_doc_uib, st_doc_uib = llegir_doc_uib(f_doc_uib)
                if df_doc_uib.empty:
                    result["errors"].append(f"Error llegint DOC UIB: {nom_doc_uib}")
                    continue

                # Tracking UIB files utilitzats
                used_uib_files.add(f_doc_uib)

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

                # Intentar obtenir DOC Direct del mestre (dual protocol)
                if has_master:
                    matched = False

                    # FORMAT NOU: Usar sample_ranges_new
                    if sample_ranges_new and not matched:
                        # Construir Sample_Rep per buscar
                        sample_rep_key = f"{mostra}_R{rep}"
                        # Buscar coincidència exacta o parcial
                        match_info = None
                        for sr, info in sample_ranges_new.items():
                            if sr == sample_rep_key:
                                match_info = info
                                break
                            # Buscar per coincidència parcial (mostra sense rep)
                            if normalize_key(sr.split("_R")[0]) == normalize_key(mostra):
                                if match_info is None:
                                    match_info = info

                        if match_info:
                            try:
                                row_ini = match_info.get("row_start")
                                row_fi = match_info.get("row_end")
                                t_start = match_info.get("t_start")

                                if row_ini is not None and row_fi is not None:
                                    df_doc_direct = extract_doc_from_masterfile(df_toc, row_ini, row_fi, t_start)

                                    if not df_doc_direct.empty:
                                        t_direct = pd.to_numeric(df_doc_direct["time (min)"], errors="coerce").to_numpy()
                                        y_direct_raw = pd.to_numeric(df_doc_direct["DOC"], errors="coerce").to_numpy()

                                        base_direct = get_baseline_correction(t_direct, y_direct_raw, method, config, use_end=(method == "BP"))
                                        y_direct_net = apply_smoothing(y_direct_raw - base_direct)
                                        y_direct_net[y_direct_net < 0] = 0

                                        nom_doc_direct = os.path.basename(mestre)
                                        st_doc_direct = "OK"
                                        matched = True
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
                                df_doc_direct = extract_doc_from_master(df_toc, row_ini, row_fi, row["START"])

                                if not df_doc_direct.empty:
                                    t_direct = pd.to_numeric(df_doc_direct["time (min)"], errors="coerce").to_numpy()
                                    y_direct_raw = pd.to_numeric(df_doc_direct["DOC"], errors="coerce").to_numpy()

                                    # Per BP Direct, usar el final del cromatograma per baseline
                                    base_direct = get_baseline_correction(t_direct, y_direct_raw, method, config, use_end=(method == "BP"))
                                    y_direct_net = apply_smoothing(y_direct_raw - base_direct)
                                    y_direct_net[y_direct_net < 0] = 0

                                    nom_doc_direct = os.path.basename(mestre)
                                    st_doc_direct = "OK"
                            except Exception as e:
                                result["errors"].append(f"WARN: Error extraient DOC Direct per {mostra}: {e}")

                # Decidir quin temps i DOC usar per l'anàlisi principal
                # Si tenim dual, usar el Direct com a principal (és el "oficial")
                if t_direct is not None and len(t_direct) > 0:
                    t_doc = t_direct
                    y_doc_net = y_direct_net
                    y_doc_raw = y_direct_raw
                    base = base_direct
                    nom_doc = nom_doc_direct
                    st_doc = st_doc_direct
                    doc_mode = "DUAL"
                else:
                    # Fallback a UIB si no hi ha Direct
                    t_doc = t_uib
                    y_doc_net = y_uib_net
                    y_doc_raw = y_uib_raw
                    base = base_uib
                    nom_doc = nom_doc_uib
                    st_doc = st_doc_uib
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

                # Detectar pic (usar DOC principal)
                peak_info = detect_main_peak(t_doc, y_doc_net, config["peak_min_prominence_pct"])

                # Calcular àrees
                sample_analysis = analyze_sample_areas(t_doc, y_doc_net, df_dad, peak_info, config)

                # Guardar Excel
                out_name = f"{mostra}_{seq_out}_R{rep}.xlsx"
                out_path = os.path.join(path_out, out_name)

                # Preparar DOC_UIB per dual (alinear i interpolar a escala de temps Direct)
                y_doc_uib_aligned = None
                time_shift = 0.0
                if doc_mode == "DUAL" and len(y_uib_net) > 0:
                    try:
                        if method == "BP":
                            # BP: alinear pel màxim de cada mostra
                            y_doc_uib_aligned, time_shift = align_signals_by_max(
                                t_doc, y_doc_net, t_uib, y_uib_net
                            )
                        else:
                            # COLUMN: aplicar shifts calculats amb KHP (si n'hi ha)
                            if column_shift_uib != 0.0:
                                # Aplicar shift a UIB
                                y_doc_uib_aligned = apply_shift(t_doc, t_uib, y_uib_net, column_shift_uib)
                                time_shift = column_shift_uib
                            else:
                                # Sense shift, només interpolar
                                y_doc_uib_aligned = np.interp(t_doc, t_uib, y_uib_net)

                            # Aplicar shift a Direct si cal (modifica y_doc_net)
                            if column_shift_direct != 0.0 and t_direct is not None:
                                y_doc_net = apply_shift(t_doc, t_direct, y_direct_net, column_shift_direct)
                    except Exception:
                        # Fallback: interpolació simple
                        try:
                            y_doc_uib_aligned = np.interp(t_doc, t_uib, y_uib_net)
                        except Exception:
                            y_doc_uib_aligned = None

                # Preparar UIB RAW interpolat a l'escala de temps Direct
                y_uib_raw_interp = None
                base_uib_interp = None
                if doc_mode == "DUAL" and len(y_uib_raw) > 0:
                    try:
                        y_uib_raw_interp = np.interp(t_doc, t_uib, y_uib_raw)
                        base_uib_interp = np.interp(t_doc, t_uib, base_uib)
                    except Exception:
                        pass

                # Determinar shifts aplicats
                applied_shift_uib = None
                applied_shift_direct = None
                if doc_mode == "DUAL":
                    if method == "BP":
                        applied_shift_uib = time_shift  # shift calculat per align_signals_by_max
                    else:
                        applied_shift_uib = column_shift_uib if column_shift_uib != 0.0 else None
                        applied_shift_direct = column_shift_direct if column_shift_direct != 0.0 else None

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
                    master_info=master_info
                )

                result["files"].append(out_path)
                processed_count += 1

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
                master_info = master_data.get("info", {})  # 0-INFO

                if df_toc is None or df_toc.empty:
                    result["errors"].append("MasterFile sense dades TOC")
                    return result

                # Si TOC_CALC buit, calcular al vol des de HPLC-SEQ i TOC
                if df_toc_calc is None or df_toc_calc.empty:
                    result["errors"].append("INFO: Calculant 4-TOC_CALC al vol")
                    df_toc_calc = calculate_toc_calc_on_the_fly(master_data)
                    if df_toc_calc is None or df_toc_calc.empty:
                        result["errors"].append("No s'ha pogut calcular TOC_CALC")
                        return result

                # Construir rangs de mostres
                sample_ranges = build_sample_ranges_from_toc_calc(df_toc_calc, df_toc)

                if not sample_ranges:
                    result["errors"].append("No s'han pogut extreure mostres de 4-TOC_CALC")
                    return result

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

                    df_doc = extract_doc_from_masterfile(df_toc, row_ini, row_fi, t_start)
                    if df_doc.empty:
                        continue

                    t_doc = pd.to_numeric(df_doc["time (min)"], errors="coerce").to_numpy()
                    y_doc = pd.to_numeric(df_doc["DOC"], errors="coerce").to_numpy()

                    base_arr = get_baseline_correction(t_doc, y_doc, method, config)
                    doc_net = apply_smoothing(y_doc - base_arr)
                    doc_net[doc_net < 0] = 0

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

                    # Detectar pic
                    peak_info = detect_main_peak(t_doc, doc_net, config["peak_min_prominence_pct"])

                    # Calcular àrees
                    sample_analysis = analyze_sample_areas(t_doc, doc_net, df_dad, peak_info, config)

                    # Guardar Excel
                    out_name = f"{mostra_clean}_{seq_out}_R{rep}.xlsx"
                    out_path = os.path.join(path_out, out_name)

                    write_consolidated_excel(
                        out_path, mostra_clean, rep, seq_out, date_master, method, "DIRECT",
                        os.path.basename(mestre), nom_dad, "OK", st_dad, t_doc, y_doc, doc_net, base_arr,
                        df_dad, peak_info, sample_analysis,
                        master_file=mestre, row_start=row_ini, row_end=row_fi,
                        smoothing_applied=True,
                        master_info=master_info
                    )

                    result["files"].append(out_path)
                    processed_count += 1

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

                    df_doc = extract_doc_from_master(df_toc, row_ini, row_fi, row["START"])
                    if df_doc.empty:
                        continue

                    t_doc = pd.to_numeric(df_doc["time (min)"], errors="coerce").to_numpy()
                    y_doc = pd.to_numeric(df_doc["DOC"], errors="coerce").to_numpy()

                    base_arr = get_baseline_correction(t_doc, y_doc, method, config)
                    doc_net = apply_smoothing(y_doc - base_arr)
                    doc_net[doc_net < 0] = 0

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

                    # Detectar pic
                    peak_info = detect_main_peak(t_doc, doc_net, config["peak_min_prominence_pct"])

                    # Calcular àrees
                    sample_analysis = analyze_sample_areas(t_doc, doc_net, df_dad, peak_info, config)

                    # Guardar Excel
                    out_name = f"{mostra_clean}_{seq_out}_R{rep}.xlsx"
                    out_path = os.path.join(path_out, out_name)

                    write_consolidated_excel(
                        out_path, mostra_clean, rep, seq_out, date_master, method, "DIRECT",
                        os.path.basename(mestre), nom_dad, "OK", st_dad, t_doc, y_doc, doc_net, base_arr,
                        df_dad, peak_info, sample_analysis,
                        master_file=mestre, row_start=row_ini, row_end=row_fi,
                        smoothing_applied=True,
                        master_info={}  # Format antic no té 0-INFO
                    )

                    result["files"].append(out_path)
                    processed_count += 1

        result["processed_count"] = processed_count
        result["success"] = processed_count > 0

        # Verificació de fitxers (QA/QC)
        file_check = check_sequence_files(input_folder, used_uib_files, used_dad_files)
        result["file_check"] = {
            "uib_found": file_check["uib"]["count_found"],
            "uib_used": file_check["uib"]["count_used"],
            "uib_orphan": file_check["uib"]["count_orphan"],
            "dad_found": file_check["dad"]["count_found"],
            "dad_used": file_check["dad"]["count_used"],
            "dad_orphan": file_check["dad"]["count_orphan"],
            "has_issues": file_check["has_issues"],
            "orphan_files": file_check["uib"]["orphan"] + file_check["dad"]["orphan"],
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
