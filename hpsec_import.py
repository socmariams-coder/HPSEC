"""
hpsec_import.py - Mòdul d'importació de dades HPSEC (Fase 1)
=============================================================

FASE 1 del pipeline de 5 fases:
- Crear MasterFile si no existeix (des de rawdata v11/v12)
- Llegir fitxers font (.D, UIB, DAD)
- Aparellar mostres (matching)
- Identificar tipus (KHP, Control, Mostra)
- Aplicar correcció de baseline (preprocessament bàsic)
- Guardar dades RAW i NET (amb baseline restada)

NO fa:
- Validació KHP (Fase 2: CALIBRAR)
- Alineació ni càlcul d'àrees (Fase 3: PROCESSAR)
- Comparació rèpliques (Fase 4: REVISAR)
- Escriptura Excel finals (Fase 5: EXPORTAR)

NOTA: La correcció de baseline s'aplica aquí per permetre que
Calibrate (Fase 2) treballi sobre dades netes per detectar pics KHP.

Usat per HPSEC_Suite.py
"""

__version__ = "1.2.0"
__version_date__ = "2026-01-30"

import os
import re
import glob
import json
from datetime import datetime
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

from hpsec_config import get_config
from hpsec_utils import get_baseline_value
from hpsec_core import detect_timeout
from hpsec_migrate_master import migrate_single

# Import sistema d'avisos estructurats
from hpsec_warnings import (
    create_warning, get_max_warning_level, WarningLevel,
)

# =============================================================================
# CONSTANTS
# =============================================================================
CONFIDENCE_THRESHOLD = 85.0  # Llindar per acceptar match automàticament
DATA_FOLDER_NAME = "data"  # Subcarpeta dins CHECK per JSONs


# =============================================================================
# CARPETA DADES (JSONs)
# =============================================================================
def get_data_folder(seq_path, create=True):
    """
    Retorna la carpeta on guardar JSONs i dades processades.

    Ubicació: SEQ_PATH/CHECK/data/

    Args:
        seq_path: Ruta a la carpeta SEQ
        create: Si True, crea la carpeta si no existeix

    Returns:
        Path a la carpeta de dades
    """
    check_folder = os.path.join(seq_path, "CHECK")
    data_folder = os.path.join(check_folder, DATA_FOLDER_NAME)

    if create:
        os.makedirs(data_folder, exist_ok=True)

    return data_folder

# NOTA: Funcions de baseline centralitzades a hpsec_utils.py
# Usar get_baseline_value() i get_baseline_stats() directament


# =============================================================================
# UTILITATS BÀSIQUES
# =============================================================================

def normalize_key(s):
    """
    Normalitza string per matching.

    - Elimina guions (-) i espais
    - MANTÉ underscores (_) perquè són significatius en noms de fitxers
    - Converteix a majúscules

    Exemples:
        "MQ-1" -> "MQ1"
        "MQ_1" -> "MQ_1"  (underscore mantingut)
        "NaOH 0.1mM" -> "NAOH0.1MM"
    """
    result = re.sub(r"[\-\s]+", "", str(s or ""))  # Només elimina guions i espais
    return result.upper()


def generate_agilent_control_name(base_name, occurrence_num):
    """
    Genera nom segons lògica Agilent per controls repetits.

    Agilent/ChemStation assigna noms així quan una mostra es repeteix:
    - 1a aparició: nom base (ex: "MQ")
    - 2a aparició: nom base + "1" (ex: "MQ1")
    - 3a aparició: nom base + "2" (ex: "MQ2")

    Args:
        base_name: Nom base del control (ex: "MQ", "NaOH")
        occurrence_num: Número d'aparició (1, 2, 3, ...)

    Returns:
        str: Nom segons convenció Agilent

    Exemples:
        generate_agilent_control_name("MQ", 1) -> "MQ"
        generate_agilent_control_name("MQ", 2) -> "MQ1"
        generate_agilent_control_name("MQ", 3) -> "MQ2"
    """
    if occurrence_num <= 1:
        return base_name
    else:
        return f"{base_name}{occurrence_num - 1}"


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


def is_khp(name):
    """Detecta si és mostra KHP."""
    return "KHP" in str(name).upper()


def extract_khp_conc(filename):
    """
    Extreu la concentració de KHP del nom del fitxer.

    Patrons suportats:
    - KHP2, KHP_2, KHP-2 -> 2.0 ppm
    - KHP2.5 -> 2.5 ppm
    - KHP 10 -> 10.0 ppm

    Args:
        filename: Nom del fitxer o mostra

    Returns:
        float: Concentració en ppm, o 0.0 si no trobada
    """
    name = os.path.basename(str(filename)).upper()

    patterns = [
        r"KHP[_\-]?(\d+(?:\.\d+)?)",  # KHP2, KHP_2, KHP-2, KHP2.5
        r"KHP\s+(\d+(?:\.\d+)?)",      # KHP 2
    ]

    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue

    return 0.0


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

    control_patterns = config.get("control_injections", "patterns", default=[])
    sample_upper = sample_name.upper()

    for pattern in control_patterns:
        if pattern.upper() in sample_upper:
            return True
    return False


def is_reference_standard(sample_name, config=None):
    """
    Verifica si una mostra és un Patró de Referència (PR).

    Patrons: Br, NO3, CaCO3, SUWANNEE, SRNOM, o que acaben en HA/FA.

    Args:
        sample_name: Nom de la mostra
        config: Configuració (si None, es llegeix de get_config())

    Returns:
        True si és un patró de referència
    """
    if config is None:
        config = get_config()

    pr_config = config.get("sample_types", "PATRÓ_REF", default={})
    patterns = pr_config.get("patterns", [])
    suffixes = pr_config.get("suffixes", [])

    sample_upper = sample_name.upper().strip()

    # Comprovar patrons (conté el patró)
    for pattern in patterns:
        if pattern.upper() in sample_upper:
            return True

    # Comprovar sufixos (acaba amb el sufix)
    for suffix in suffixes:
        if sample_upper.endswith(suffix.upper()):
            return True

    return False


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


def skip_sample_direct(sample_desc):
    """Determina si s'ha d'ignorar una mostra (PRE-HPLC, POST-HPLC, etc.)."""
    d = str(sample_desc or "").strip().upper()
    if d in ("", "0"):
        return True
    if "PRE-HPLC" in d or ("POST" in d and "HPLC" in d):
        return True
    return False


# =============================================================================
# DETECCIÓ MODE (BP vs COLUMN)
# =============================================================================

def is_bp_seq(folder, sample_dad_files=None):
    """
    Detecta si és seqüència BP basant-se en:
    1. Nom de la carpeta (conté _BP)
    2. Durada dels cromatogrames DAD (≤18 min = BP)
    """
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


def detect_mode_from_folder(seq_path):
    """
    Detecta el mode de la seqüència (BP o COLUMN) i el tipus de dades (UIB/DIRECT/DUAL).

    Returns:
        dict amb method ("BP"/"COLUMN") i data_mode ("UIB"/"DIRECT"/"DUAL")
    """
    result = {
        "method": "COLUMN",
        "data_mode": "UIB",
    }

    # Detectar BP pel nom
    folder_name = os.path.basename(seq_path).upper()
    if "_BP" in folder_name or folder_name.endswith("BP"):
        result["method"] = "BP"

    # Detectar tipus de dades
    path_csv = os.path.join(seq_path, "CSV")
    has_uib = False
    has_direct = False

    if os.path.isdir(path_csv):
        uib_files = glob.glob(os.path.join(path_csv, "*UIB1B*"))
        has_uib = len(uib_files) > 0

    # Buscar master file per Direct
    master_path, master_format = trobar_excel_mestre(seq_path)
    if master_path and master_format == "NEW":
        master_data = llegir_masterfile_nou(master_path)
        if master_data.get("toc") is not None:
            has_direct = True

    if has_uib and has_direct:
        result["data_mode"] = "DUAL"
    elif has_direct:
        result["data_mode"] = "DIRECT"
    else:
        result["data_mode"] = "UIB"

    return result


# =============================================================================
# LECTURA FITXERS MASTER
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

    # Prioritzar format NEW (MasterFile)
    # NO retornar format OLD (v11/v12) - cal migrar-los primer
    if new_format:
        # Prioritzar fitxers amb "MasterFile" al nom
        for p in new_format:
            if "MasterFile" in os.path.basename(p):
                return p, "NEW"
        return new_format[0], "NEW"

    # Si només hi ha format OLD (v11/v12), retornar None per disparar migrate
    # El migrate crearà el MasterFile a partir del v11/v12
    return None, None


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
        "filepath": filepath,
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

            # Sensibilitat UIB a B5 (fila 4, columna 1 en 0-indexed)
            if len(df_info) > 4 and len(df_info.columns) > 1:
                uib_sens = df_info.iloc[4, 1]
                if pd.notna(uib_sens):
                    result["info"]["uib_sensitivity"] = uib_sens

        # 1-HPLC-SEQ (o 1-HPLC-SEQ_RAW per format antic v11)
        if "1-HPLC-SEQ" in xl.sheet_names:
            result["hplc_seq"] = pd.read_excel(xl, sheet_name="1-HPLC-SEQ", engine="openpyxl")
        elif "1-HPLC-SEQ_RAW" in xl.sheet_names:
            result["hplc_seq"] = pd.read_excel(xl, sheet_name="1-HPLC-SEQ_RAW", engine="openpyxl")

        # 2-TOC
        if "2-TOC" in xl.sheet_names:
            result["toc"] = pd.read_excel(xl, sheet_name="2-TOC", header=6, engine="openpyxl")

        # 4-TOC_CALC (o 4-SEQ_DATA per format antic v11)
        if "4-TOC_CALC" in xl.sheet_names:
            result["toc_calc"] = pd.read_excel(xl, sheet_name="4-TOC_CALC", engine="openpyxl")
        elif "4-SEQ_DATA" in xl.sheet_names:
            result["toc_calc"] = pd.read_excel(xl, sheet_name="4-SEQ_DATA", engine="openpyxl")

        # 3-DAD_KHP (opcional)
        if "3-DAD_KHP" in xl.sheet_names:
            result["dad_khp"] = pd.read_excel(xl, sheet_name="3-DAD_KHP", header=1, engine="openpyxl")

    except Exception as e:
        result["error"] = str(e)

    return result


def llegir_master_direct(mestre):
    """
    Llegeix format antic de master (2-TOC & 4-SEQ_DATA).

    Returns:
        (df_toc, df_seq) o (None, None) si error
    """
    try:
        xl = pd.ExcelFile(mestre, engine="openpyxl")

        df_toc = None
        df_seq = None

        # Llegir 2-TOC
        if "2-TOC" in xl.sheet_names:
            df_toc = pd.read_excel(xl, sheet_name="2-TOC", header=6, engine="openpyxl")

        # Llegir 4-SEQ_DATA
        if "4-SEQ_DATA" in xl.sheet_names:
            df_seq = pd.read_excel(xl, sheet_name="4-SEQ_DATA", engine="openpyxl")

        return df_toc, df_seq
    except Exception:
        return None, None


def extract_doc_from_masterfile(toc_df, row_start, row_end, t_start=None, detect_timeouts=True,
                                  max_duration_min=None):
    """
    Extreu segment DOC del nou MasterFile.

    Args:
        toc_df: DataFrame de 2-TOC
        row_start: Fila inicial (1-indexed, relatiu a fila 8 del TOC original)
        row_end: Fila final
        t_start: Timestamp d'inici (opcional, per calcular temps relatiu)
        detect_timeouts: Si True, també detecta i retorna info de timeouts
        max_duration_min: Durada màxima en minuts (per truncar última mostra)
                         Si None, no trunca. Típic: 75-80 min per COLUMN

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

    # Truncar si supera la durada màxima (útil per última mostra)
    if max_duration_min is not None and len(df_doc) > 0:
        max_time = df_doc["time (min)"].max()
        if max_time > max_duration_min:
            df_doc = df_doc[df_doc["time (min)"] <= max_duration_min].copy()

    # Detectar timeouts si s'ha sol·licitat
    if detect_timeouts:
        timeout_info = detect_timeout(df_doc["time (min)"].values)
        return df_doc, timeout_info

    return df_doc


def extract_doc_from_master(df_toc, row_ini, row_fi, start_dt, detect_timeouts=True,
                            max_duration_min=None):
    """
    Extreu segment DOC del format antic de master.

    Args:
        df_toc: DataFrame de 2-TOC
        row_ini: Fila inicial
        row_fi: Fila final
        start_dt: Datetime d'inici
        detect_timeouts: Si True, detecta timeouts
        max_duration_min: Durada màxima en minuts (per truncar última mostra)

    Returns:
        Si detect_timeouts=False: DataFrame amb columns "time (min)" i "DOC"
        Si detect_timeouts=True: tuple (DataFrame, timeout_info dict)
    """
    # Reutilitzar la funció del nou format
    return extract_doc_from_masterfile(df_toc, row_ini, row_fi, start_dt, detect_timeouts,
                                       max_duration_min)


# =============================================================================
# LECTURA FITXERS UIB (DOC)
# =============================================================================

def llegir_doc_uib(path):
    """
    Llegeix fitxer DOC format UIB (CSV tab-separated, UTF-16 o UTF-8).

    Returns:
        (DataFrame, status): DataFrame amb columns ['time (min)', 'DOC'], status string
    """
    for enc in ["utf-16", "utf-8"]:
        try:
            df = pd.read_csv(path, sep="\t", encoding=enc, header=None, engine="python")
            df = df.iloc[:, [0, 1]]
            df.columns = ["time (min)", "DOC"]
            return df, f"OK{' (UTF-8)' if enc == 'utf-8' else ''}"
        except Exception:
            continue
    return pd.DataFrame(), "Error"


def netejar_nom_uib(nom_fitxer):
    """
    Neteja nom de fitxer UIB i extreu mostra/rèplica.

    Returns:
        (mostra, rep): Tuple amb nom de mostra i número de rèplica
    """
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


# =============================================================================
# LECTURA FITXERS DAD
# =============================================================================

def llegir_dad_export3d(path):
    """
    Llegeix fitxer DAD Export3D (CSV comma-separated).

    Returns:
        (DataFrame, status): DataFrame amb temps i wavelengths, status string
    """
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
            df["time (min)"] = pd.to_numeric(df["time (min)"], errors="coerce")
            df[wavelength] = pd.to_numeric(df[wavelength], errors="coerce")
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


def list_dad_files(folder_export3d, folder_csv=None):
    """
    Llista tots els fitxers DAD (matriu 3D completa).

    NOMÉS busca a Export3d - els fitxers CSV són UIB (DOC), no DAD.

    Returns:
        Llista de paths a fitxers DAD
    """
    dad_files = []

    # Només Export3d conté fitxers DAD (matriu 3D)
    if folder_export3d and os.path.isdir(folder_export3d):
        for ext in ("*.csv", "*.CSV"):
            dad_files.extend(glob.glob(os.path.join(folder_export3d, ext)))

    # Excloure fitxers UIB (contenen "UIB1B") per si de cas
    dad_files = [f for f in dad_files if "UIB1B" not in os.path.basename(f).upper()]

    return sorted(set(dad_files))


# =============================================================================
# MATCHING MOSTRES (UIB ↔ HPLC-SEQ)
# =============================================================================

def get_valid_samples_from_hplc_seq(master_data):
    """
    Extreu la llista de mostres vàlides de 1-HPLC-SEQ.

    Args:
        master_data: Dict retornat per llegir_masterfile_nou()

    Returns:
        set de noms de mostres (normalitzats) que pertanyen a la SEQ
    """
    df_hplc = master_data.get("hplc_seq")
    if df_hplc is None or df_hplc.empty:
        return set()

    valid_samples = set()

    # Buscar columna "Sample Name"
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
            valid_samples.add(name)
            valid_samples.add(normalize_key(name))

    return valid_samples


def match_sample_confidence(sample_name, valid_samples):
    """
    Cerca la millor coincidència d'una mostra amb la llista vàlida.

    Args:
        sample_name: Nom de la mostra (del fitxer UIB)
        valid_samples: Set de mostres vàlides

    Returns:
        dict amb matched, best_match, confidence (0-100), match_type
    """
    result = {
        "matched": False,
        "best_match": None,
        "confidence": 0.0,
        "match_type": "NOT_FOUND"
    }

    if not valid_samples:
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

    # 4. Variants (85%)
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

    # 5. Fuzzy matching
    best_ratio = 0.0
    best_match = None

    for vs in valid_samples:
        ratio1 = SequenceMatcher(None, sample_upper, vs.upper()).ratio()
        ratio2 = SequenceMatcher(None, sample_norm, normalize_key(vs)).ratio()
        ratio = max(ratio1, ratio2)

        if ratio > best_ratio:
            best_ratio = ratio
            best_match = vs

    # Només suggerir si >95% i no és diferència numèrica
    sample_base = re.sub(r'\d+$', '', sample_name)

    if best_ratio >= 0.95:
        best_base = re.sub(r'\d+$', '', best_match) if best_match else ""
        if sample_base == best_base:
            return result

        confidence = best_ratio * 100
        return {
            "matched": True,
            "best_match": best_match,
            "confidence": confidence,
            "match_type": "FUZZY"
        }

    return result


def is_sample_in_seq(sample_name, valid_samples):
    """Verifica si una mostra pertany a la seqüència."""
    match_info = match_sample_confidence(sample_name, valid_samples)
    return match_info["matched"] and match_info["confidence"] >= CONFIDENCE_THRESHOLD


# =============================================================================
# MATCHING INTEL·LIGENT D'ORFES
# =============================================================================

def extract_sample_from_filename(filename):
    """
    Extreu el nom de mostra i rèplica d'un nom de fitxer.

    Exemples:
        "MOSTRA_A_R1.csv" → ("MOSTRA_A", 1)
        "KHP2_R2_UIB1B.csv" → ("KHP2", 2)
        "MOSTRA-B-F_1.csv" → ("MOSTRA-B-F", 1)
    """
    stem = os.path.splitext(os.path.basename(filename))[0]

    # Eliminar sufixos comuns (UIB1B, etc.)
    stem_clean = re.sub(r"_?UIB1B\d*", "", stem, flags=re.IGNORECASE)

    # Buscar patró _R# o _#
    match_r = re.search(r"[_\-]R(\d+)$", stem_clean, flags=re.IGNORECASE)
    match_us = re.search(r"_(\d+)$", stem_clean)

    if match_r:
        sample = stem_clean[:match_r.start()]
        rep = int(match_r.group(1))
    elif match_us:
        sample = stem_clean[:match_us.start()]
        rep = int(match_us.group(1))
    else:
        sample = stem_clean
        rep = 1

    return sample, rep


def compute_orphan_suggestions(samples, orphan_files, file_type="dad"):
    """
    Calcula suggeriments de matching entre mostres sense dades i fitxers orfes.

    Args:
        samples: Dict de mostres {nom: {type, replicas: {rep: {dad, uib, ...}}}}
        orphan_files: Llista de paths de fitxers orfes
        file_type: "dad" o "uib"

    Returns:
        Dict de suggeriments:
        {
            (sample_name, replica): {
                "suggested_file": path,
                "suggested_filename": nom_fitxer,
                "confidence": 0-100,
                "match_type": "FUZZY" | "NORMALIZED" | "EXACT",
                "auto_assign": True si confidence > 85
            }
        }
    """
    suggestions = {}

    if not orphan_files:
        return suggestions

    # Construir índex d'orfes: {(sample_norm, rep): [files]}
    orphan_index = {}
    for fpath in orphan_files:
        sample, rep = extract_sample_from_filename(fpath)
        sample_norm = normalize_key(sample)
        key = (sample_norm, rep)
        if key not in orphan_index:
            orphan_index[key] = []
        orphan_index[key].append({
            "path": fpath,
            "filename": os.path.basename(fpath),
            "sample_extracted": sample,
        })

    # Per cada mostra/rèplica sense dades del tipus especificat
    for sample_name, sample_info in samples.items():
        for rep_str, rep_data in sample_info.get("replicas", {}).items():
            rep_num = int(rep_str) if rep_str.isdigit() else 1

            # Comprovar si falta el tipus de dades
            has_data = False
            if file_type == "dad":
                has_data = rep_data.get("dad") is not None
            elif file_type == "uib":
                has_data = rep_data.get("uib") is not None and rep_data["uib"].get("t") is not None

            if has_data:
                continue  # Ja té dades, no cal suggeriment

            sample_norm = normalize_key(sample_name)

            # Buscar el millor match entre els orfes
            sample_norm = normalize_key(sample_name)
            best_match = None
            best_confidence = 0
            best_match_type = "NONE"

            for (orphan_sample_norm, orphan_rep), orphan_list in orphan_index.items():
                # Primer, comprovar si la rèplica coincideix
                rep_matches = (orphan_rep == rep_num)
                orphan_sample_orig = orphan_list[0]["sample_extracted"].upper()

                # Calcular similitud del nom amb diverses estratègies
                if sample_norm == orphan_sample_norm:
                    # Match exacte (normalitzat)
                    confidence = 100 if rep_matches else 80
                    match_type = "EXACT"
                elif sample_norm.endswith(orphan_sample_norm) or sample_name.upper().endswith(orphan_sample_orig):
                    # La mostra ACABA amb el nom de l'orfe (ex: "3S101H HA" acaba amb "HA")
                    confidence = 95 if rep_matches else 75
                    match_type = "SUFFIX"
                elif orphan_sample_norm in sample_norm or orphan_sample_orig in sample_name.upper():
                    # L'orfe està CONTINGUT dins la mostra
                    confidence = 85 if rep_matches else 65
                    match_type = "CONTAINS"
                else:
                    # Fuzzy match
                    ratio1 = SequenceMatcher(None, sample_norm, orphan_sample_norm).ratio()
                    ratio2 = SequenceMatcher(None, sample_name.upper(), orphan_sample_orig).ratio()
                    ratio = max(ratio1, ratio2)

                    # Penalitzar si la rèplica no coincideix
                    if not rep_matches:
                        ratio *= 0.7

                    confidence = ratio * 100
                    match_type = "FUZZY"


                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = orphan_list[0]  # Primer fitxer de la llista
                    best_match_type = match_type

            # Guardar suggeriment si la confiança és suficient
            if best_match and best_confidence >= 60:
                suggestions[(sample_name, rep_num)] = {
                    "suggested_file": best_match["path"],
                    "suggested_filename": best_match["filename"],
                    "confidence": round(best_confidence, 1),
                    "match_type": best_match_type,
                    "auto_assign": best_confidence >= 85,
                }

    return suggestions


def apply_orphan_suggestions(samples, suggestions, orphan_files, file_type="dad"):
    """
    Aplica els suggeriments d'alta confiança automàticament.

    Args:
        samples: Dict de mostres (es modifica in-place)
        suggestions: Dict de suggeriments de compute_orphan_suggestions
        orphan_files: Llista d'orfes (es modifica per treure els assignats)
        file_type: "dad" o "uib"

    Returns:
        Tuple (applied_count, remaining_orphans)
    """
    applied = []

    for (sample_name, rep_num), suggestion in suggestions.items():
        if not suggestion.get("auto_assign"):
            continue

        # Trobar la mostra i rèplica
        if sample_name not in samples:
            continue

        rep_str = str(rep_num)
        if rep_str not in samples[sample_name].get("replicas", {}):
            continue

        rep_data = samples[sample_name]["replicas"][rep_str]

        # Marcar com a assignació suggerida (no carregar dades encara)
        if file_type == "dad":
            rep_data["dad_suggestion"] = {
                "file": suggestion["suggested_file"],
                "filename": suggestion["suggested_filename"],
                "confidence": suggestion["confidence"],
                "match_type": suggestion["match_type"],
                "status": "SUGGESTED",  # Per revisar a la GUI
            }
        elif file_type == "uib":
            rep_data["uib_suggestion"] = {
                "file": suggestion["suggested_file"],
                "filename": suggestion["suggested_filename"],
                "confidence": suggestion["confidence"],
                "match_type": suggestion["match_type"],
                "status": "SUGGESTED",
            }

        applied.append(suggestion["suggested_file"])

    # Treure els fitxers aplicats de la llista d'orfes
    remaining = [f for f in orphan_files if f not in applied]

    return len(applied), remaining


# =============================================================================
# MATCHING DAD FILES
# =============================================================================

def detect_dad_rep_style(dad_files):
    """
    Detecta l'estil de numeració de rèpliques als fitxers DAD.

    Returns:
        "R" si usa _R1, _R2
        "UNDERSCORE" si usa _1, _2
        "" si no es detecta
    """
    r_count = 0
    us_count = 0

    for f in dad_files:
        stem = os.path.splitext(os.path.basename(f))[0]
        if re.search(r'_R\d+$', stem, re.IGNORECASE):
            r_count += 1
        elif re.search(r'_\d+$', stem):
            us_count += 1

    if r_count > us_count:
        return "R"
    elif us_count > r_count:
        return "UNDERSCORE"
    return ""


def dad_sample_rep_from_path(path):
    """
    Extreu nom de mostra i rèplica d'un path DAD.

    Returns:
        (sample, rep, style): Nom mostra, número rèplica, estil detectat
    """
    stem = os.path.splitext(os.path.basename(path))[0]

    # Estil R
    match_r = re.search(r'^(.+?)_R(\d+)$', stem, re.IGNORECASE)
    if match_r:
        return match_r.group(1), match_r.group(2), "R"

    # Estil underscore
    match_us = re.search(r'^(.+?)_(\d+)$', stem)
    if match_us:
        return match_us.group(1), match_us.group(2), "UNDERSCORE"

    return stem, "1", "NONE"


def target_keys_from_desc(sample_desc):
    """Genera claus de matching a partir d'una descripció de mostra."""
    keys = []
    s = str(sample_desc).strip()
    if s:
        keys.append(s)
        keys.append(normalize_key(s))
        keys.append(s.upper())
        keys.append(s.replace("_", ""))
        keys.append(s.replace("-", "_"))
    return keys


def choose_best_candidate(target_key, rep, dad_files, used_files=None):
    """
    Selecciona el millor fitxer DAD per una mostra.

    Returns:
        (path, status) o (None, status)
    """
    used_files = used_files or set()
    target_norm = normalize_key(target_key)
    rep_str = str(rep) if rep else "1"

    candidates = []
    for f in dad_files:
        if f in used_files:
            continue
        sample, file_rep, style = dad_sample_rep_from_path(f)
        if normalize_key(sample) == target_norm and file_rep == rep_str:
            candidates.append((f, 100))  # Match exacte
        elif normalize_key(sample) == target_norm:
            candidates.append((f, 80))   # Match mostra però no rèplica

    if candidates:
        candidates.sort(key=lambda x: -x[1])
        return candidates[0][0], "OK"

    return None, "NOT_FOUND"


# =============================================================================
# MATCHING MASTER ↔ UIB
# =============================================================================

def find_master_row_for_sample(df_seq, mostra_uib, rep_uib):
    """
    Troba la fila del mestre que correspon a una mostra UIB.

    Args:
        df_seq: DataFrame del sheet 1-HPLC-SEQ del mestre
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
        df_seq: DataFrame del sheet 1-HPLC-SEQ del mestre

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
# VERIFICACIÓ FITXERS (QA/QC)
# =============================================================================

def detect_replica_anomalies(filenames):
    """
    Detecta anomalies en la numeració de rèpliques (ex: R12 en lloc de R2).

    Returns:
        List of dicts amb anomalies detectades
    """
    anomalies = []
    samples = {}

    for fname in filenames:
        stem = os.path.splitext(fname)[0]
        stem_clean = re.sub(r"_?UIB1B\d*", "", stem, flags=re.IGNORECASE)

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

        for i, rep_num in enumerate(rep_nums):
            expected = i + 1
            if rep_num != expected and rep_num > 3:
                for r in reps:
                    if r["rep"] == rep_num:
                        proposed_rep = expected
                        original = r["filename"]

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
                                "apply": False
                            })

    return anomalies


def check_sequence_files(seq_path, used_uib_files=None, used_dad_files=None):
    """
    Verifica els fitxers d'una seqüència i detecta discrepàncies.

    Returns:
        Dict amb info de verificació (orfes, anomalies, etc.)
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
        "uib": {"found": [], "used": [], "orphan": [], "count_found": 0, "count_used": 0, "count_orphan": 0},
        "dad": {"found": [], "used": [], "orphan": [], "count_found": 0, "count_used": 0, "count_orphan": 0},
        "anomalies": [],
        "proposed_renames": [],
        "has_issues": False,
        "low_confidence_matches": [],
        "seq_orphan_files": [],
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

        uib_anomalies = detect_replica_anomalies(result["uib"]["found"])
        result["proposed_renames"].extend(uib_anomalies)

    # Fitxers DAD
    if os.path.isdir(path_3d):
        dad_files = []
        for ext in ("*.csv", "*.CSV"):
            dad_files.extend(glob.glob(os.path.join(path_3d, ext)))
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

        dad_anomalies = detect_replica_anomalies(result["dad"]["found"])
        result["proposed_renames"].extend(dad_anomalies)

    # Determinar si hi ha problemes
    result["has_issues"] = (
        result["uib"]["count_orphan"] > 0 or
        result["dad"]["count_orphan"] > 0 or
        len(result["proposed_renames"]) > 0
    )

    return result


# =============================================================================
# FUNCIÓ PRINCIPAL: import_sequence() v2
# =============================================================================

def parse_injections_from_masterfile(master_data, config=None):
    """
    Parseja les injeccions del MasterFile (fulla 1-HPLC-SEQ).

    Gestiona controls repetits (NaOH, MQ) assignant-los _1, _2, etc.
    per ordre d'aparició.

    Returns:
        list of dict: Llista d'injeccions amb info normalitzada
        list of str: Warnings (números no correlatius, etc.)
    """
    config = config or get_config()
    injections = []
    warnings = []

    # Obtenir DataFrame de HPLC-SEQ (pot ser 'hplc_seq' o 'seq')
    df_seq = master_data.get("hplc_seq")
    if df_seq is None:
        df_seq = master_data.get("seq")
    if df_seq is None or (hasattr(df_seq, 'empty') and df_seq.empty):
        return [], ["ERROR: No s'ha trobat fulla 1-HPLC-SEQ al MasterFile"], 0

    # ==========================================================================
    # VALIDACIÓ MASTERFILE: Columnes A-F (0-5) + N (13) han de tenir dades
    # ==========================================================================
    errors = []
    col_list = list(df_seq.columns)
    required_indices = [0, 1, 2, 3, 4, 5]  # A-F
    if len(col_list) > 13:
        required_indices.append(13)  # N (volum)

    # Noms de columnes per missatges
    col_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']

    # Trobar files amb dades (Line# no buit)
    line_col_idx = 0  # Columna A = Line#
    if len(col_list) > line_col_idx:
        line_col_name = col_list[line_col_idx]
        # Files amb Line# vàlid
        valid_rows = df_seq[line_col_name].notna() & (df_seq[line_col_name].astype(str).str.strip() != '')
        n_data_rows = valid_rows.sum()

        if n_data_rows == 0:
            return [], ["ERROR: Fulla 1-HPLC-SEQ buida - no hi ha cap injecció"], 0

        # Verificar cel·les buides en les files amb dades
        empty_cells = []
        for idx in df_seq[valid_rows].index:
            row_num = idx + 2  # +2 perquè Excel comença a 1 i té header
            line_val = df_seq.loc[idx, line_col_name]
            try:
                line_num = int(line_val)
            except (ValueError, TypeError):
                line_num = row_num

            for col_idx in required_indices:
                if col_idx < len(col_list):
                    col_name = col_list[col_idx]
                    val = df_seq.loc[idx, col_name]
                    if pd.isna(val) or str(val).strip() in ['', 'nan', 'NaN']:
                        col_letter = col_letters[col_idx] if col_idx < len(col_letters) else f'Col{col_idx}'
                        empty_cells.append(f"Line {line_num}: {col_letter}")

        if empty_cells:
            # Agrupar per mostrar màxim 5 exemples
            examples = empty_cells[:5]
            more = f" (+{len(empty_cells)-5} més)" if len(empty_cells) > 5 else ""
            errors.append(f"ERROR: Cel·les buides al MasterFile 1-HPLC-SEQ: {', '.join(examples)}{more}")

    # Identificar columnes
    sample_col = None
    inj_col = None
    line_col = None
    volume_col = None
    sample_rep_col = None  # Columna Sample_Rep del migrate (si existeix)

    for col in df_seq.columns:
        col_lower = str(col).lower().strip()
        if col_lower == "sample_rep":
            sample_rep_col = col
        elif "sample" in col_lower and "name" in col_lower:
            sample_col = col
        elif col_lower == "inj#" or col_lower == "inj":
            inj_col = col
        elif col_lower == "line#" or col_lower == "line":
            line_col = col
        elif "volume" in col_lower or "vol" in col_lower:
            volume_col = col

    # Columna N (índex 13) per volum si no té etiqueta
    # En Excel v11/v12, el volum està a la columna N sense etiqueta
    if volume_col is None:
        col_list = list(df_seq.columns)
        # Columna N = índex 13 (A=0, B=1, ..., N=13)
        if len(col_list) > 13:
            potential_vol_col = col_list[13]
            # Verificar que sembla una columna de volum (valors numèrics típics: 100, 400, etc.)
            try:
                sample_vals = df_seq[potential_vol_col].dropna().head(5)
                if len(sample_vals) > 0:
                    # Verificar que són valors numèrics raonables per volum (50-1000 µL)
                    numeric_vals = pd.to_numeric(sample_vals, errors='coerce').dropna()
                    if len(numeric_vals) > 0 and all(50 <= v <= 1000 for v in numeric_vals):
                        volume_col = potential_vol_col
            except Exception:
                pass

    if sample_col is None:
        # Fallback: buscar columna amb "sample"
        for col in df_seq.columns:
            if "sample" in str(col).lower():
                sample_col = col
                break

    if sample_col is None:
        return [], ["No s'ha trobat columna 'Sample Name' al MasterFile"], 0

    # Comptador per controls repetits
    control_counts = {}  # base_name -> total count
    control_sets = {}    # base_name -> current set number
    control_last_inj = {} # base_name -> last inj_num (per detectar nou set)
    control_patterns = ["naoh", "mq", "blank", "buffer", "wash"]

    # Pre-calcular tots els noms de mostra per detectar duplicats exactes
    all_sample_names = [
        str(row.get(sample_col, "")).strip()
        for _, row in df_seq.iterrows()
        if str(row.get(sample_col, "")).strip() not in ["", "nan"]
    ]

    # Identificar columnes crítiques per validació
    rt_col = None
    area_col = None
    for col in df_seq.columns:
        col_lower = str(col).lower().strip()
        if col_lower == "rt" or "retention" in col_lower:
            rt_col = col
        elif col_lower == "area":
            area_col = col

    # Columnes crítiques que NO haurien d'estar buides: A-F + N
    # A=Line#, B=Inj#, C=Location, D=Sample Name, E=Method, F=Date, N=Volume
    col_list = list(df_seq.columns)

    # Identificar columnes per índex (A=0, B=1, ..., F=5, N=13)
    location_col = col_list[2] if len(col_list) > 2 else None  # C
    method_col = col_list[4] if len(col_list) > 4 else None    # E
    date_col = col_list[5] if len(col_list) > 5 else None      # F

    critical_cols = {
        'Line#': line_col,           # A
        'Inj#': inj_col,             # B
        'Location': location_col,     # C
        'Sample Name': sample_col,    # D
        'Method': method_col,         # E
        'Date': date_col,             # F
        'Volume': volume_col,         # N
    }
    # Filtrar None
    critical_cols = {k: v for k, v in critical_cols.items() if v is not None}

    # Comptar total de files amb dades (per validació posterior)
    total_rows_with_line = 0
    skipped_rows = []
    incomplete_rows = []  # Files amb cel·les crítiques buides

    # Processar cada fila
    prev_line = 0
    for idx, row in df_seq.iterrows():
        # Detectar si la fila té número de línia (indica que és una injecció real)
        line_val = row.get(line_col) if line_col else None
        has_line_num = line_val is not None and str(line_val).strip() not in ["", "nan"]
        if has_line_num:
            total_rows_with_line += 1

            # Validar cel·les crítiques buides
            missing_cols = []
            for col_name, col_ref in critical_cols.items():
                if col_ref is not None:
                    val = row.get(col_ref)
                    if val is None or str(val).strip() in ["", "nan", "NaN"]:
                        missing_cols.append(col_name)
            if missing_cols:
                try:
                    line_num_val = int(line_val)
                    incomplete_rows.append((line_num_val, missing_cols))
                except (ValueError, TypeError):
                    pass

        sample_name = str(row.get(sample_col, "")).strip()
        if not sample_name or sample_name.lower() in ["nan", ""]:
            # Si té número de línia però no té nom, és un error al MasterFile
            if has_line_num:
                try:
                    line_num_val = int(line_val)
                    skipped_rows.append(line_num_val)
                    warnings.append(f"⚠️ Línia {line_num_val}: falta nom de mostra (columna buida)")
                except (ValueError, TypeError):
                    pass
            continue

        # Obtenir número de línia/injecció
        line_num = row.get(line_col) if line_col else idx + 1
        inj_num = row.get(inj_col) if inj_col else 1

        try:
            line_num = int(line_num)
        except (ValueError, TypeError):
            line_num = idx + 1

        try:
            inj_num = int(inj_num)
        except (ValueError, TypeError):
            inj_num = 1

        # Check correlativitat
        if prev_line > 0 and line_num != prev_line + 1:
            warnings.append(f"Injeccions no correlatives: {prev_line} -> {line_num}")
        prev_line = line_num

        sample_lower = sample_name.lower()

        # Inicialitzar variables que s'usen després del if/else
        is_repeated_control = False
        original_name_count = 1

        # =====================================================================
        # ASSIGNAR NOM ÚNIC: Prioritzar Sample_Rep del MasterFile (migrate)
        # =====================================================================
        if sample_rep_col is not None:
            sample_rep_val = row.get(sample_rep_col)
            if sample_rep_val and str(sample_rep_val).strip() not in ["", "nan"]:
                # Sample_Rep format: "NOM_R1" o "NOM1_R2" → extreure NOM sense _R{num}
                sample_rep = str(sample_rep_val).strip()
                # Extreure part del nom (sense _R{num} al final)
                match = re.match(r'^(.+)_R(\d+)$', sample_rep)
                if match:
                    unique_name = match.group(1)
                    effective_inj_num = int(match.group(2))
                else:
                    unique_name = sample_rep
                    effective_inj_num = inj_num

                # Determinar tipus
                if is_khp(sample_name):
                    sample_type = "KHP"
                elif is_reference_standard(sample_name, config):
                    sample_type = "PR"
                elif is_control_injection(sample_name, config):
                    sample_type = "CONTROL"
                elif "test" in sample_lower:
                    sample_type = "TEST"
                else:
                    sample_type = "SAMPLE"
            else:
                # Sample_Rep buit - fallback a nom original
                unique_name = sample_name
                effective_inj_num = inj_num
                if is_khp(sample_name):
                    sample_type = "KHP"
                elif is_reference_standard(sample_name, config):
                    sample_type = "PR"
                elif is_control_injection(sample_name, config):
                    sample_type = "CONTROL"
                elif "test" in sample_lower:
                    sample_type = "TEST"
                else:
                    sample_type = "SAMPLE"
        else:
            # =====================================================================
            # FALLBACK: MasterFile antic sense Sample_Rep - generar nom al vol
            # =====================================================================
            is_repeated_control = any(p in sample_lower for p in control_patterns)

            if is_repeated_control:
                # Només eliminar espais i guions alts, mantenir xifres i underscores
                base_name = re.sub(r'[\s\-]+', '', sample_name).strip()
                if not base_name:
                    base_name = sample_name

                if base_name not in control_sets:
                    control_sets[base_name] = 1
                    control_last_inj[base_name] = 0

                if inj_num <= control_last_inj.get(base_name, 0):
                    control_sets[base_name] += 1

                control_last_inj[base_name] = inj_num
                current_set = control_sets[base_name]

                if base_name not in control_counts:
                    control_counts[base_name] = 0
                control_counts[base_name] += 1

                original_name_count = all_sample_names.count(sample_name)
                if original_name_count > 1:
                    unique_name = generate_agilent_control_name(base_name, current_set)
                else:
                    unique_name = sample_name
                sample_type = "CONTROL"
                effective_inj_num = inj_num
            else:
                unique_name = sample_name
                effective_inj_num = inj_num
                if is_khp(sample_name):
                    sample_type = "KHP"
                elif is_reference_standard(sample_name, config):
                    sample_type = "PR"
                elif is_control_injection(sample_name, config):
                    sample_type = "CONTROL"
                elif "test" in sample_lower:
                    sample_type = "TEST"
                else:
                    sample_type = "SAMPLE"

        # Obtenir volum d'injecció (si disponible)
        inj_volume = None
        if volume_col is not None:
            try:
                vol_val = row.get(volume_col)
                if vol_val is not None and str(vol_val).strip() not in ["", "nan"]:
                    inj_volume = float(vol_val)
            except (ValueError, TypeError):
                pass

        # Info de set per controls (per matching amb fitxers MQ1_R1, MQ2_R1, etc.)
        # NOMÉS usar set/rep si hi ha duplicats exactes - si els noms ja són únics, fer match directe
        if is_repeated_control and original_name_count > 1:
            control_set_num = control_sets.get(base_name)
            control_rep_in_set = inj_num
        else:
            control_set_num = None
            control_rep_in_set = None


        injections.append({
            "line_num": line_num,
            "inj_num": effective_inj_num,  # Replica efectiva
            "inj_num_original": inj_num,   # Replica original del MasterFile
            "sample_name_original": sample_name,
            "sample_name": unique_name,
            "sample_type": sample_type,
            "inj_volume": inj_volume,  # Volum d'injecció en µL (pot ser None)
            "control_set": control_set_num,  # Número de set per controls (MQ1, MQ2, ...)
            "control_rep": control_rep_in_set,  # Rèplica dins del set (R1, R2, ...)
            "row_data": row.to_dict(),
        })

    # Validar: comparar total de files amb Line# vs injeccions processades
    if total_rows_with_line > len(injections):
        missing = total_rows_with_line - len(injections)
        warnings.insert(0, f"⚠️ ATENCIÓ: {missing} injecció(ns) no processada(es) - revisar MasterFile (files: {skipped_rows})")

    # Validar: files amb cel·les crítiques buides
    if incomplete_rows:
        # Agrupar per columnes que falten
        by_missing = {}
        for line_num, cols in incomplete_rows:
            key = tuple(sorted(cols))
            if key not in by_missing:
                by_missing[key] = []
            by_missing[key].append(line_num)

        for cols, lines in by_missing.items():
            lines_str = ", ".join(str(l) for l in lines[:5])
            if len(lines) > 5:
                lines_str += f"... (+{len(lines)-5})"
            warnings.append(
                f"⚠️ FILES INCOMPLETES: Línies {lines_str} - falta: {', '.join(cols)}"
            )

    # Afegir errors de validació MasterFile al principi dels warnings
    if errors:
        warnings = errors + warnings

    # Retornar també el total de línies del MasterFile per validació posterior
    return injections, warnings, total_rows_with_line


def find_data_for_injection(injection, seq_path, uib_files, dad_files, dad_csv_files,
                            master_khp_data, used_files, config=None,
                            toc_df=None, toc_calc_df=None,
                            valid_sample_names=None):
    """
    Busca dades per una injecció des de múltiples fonts.

    Prioritat:
    - DOC Direct: MasterFile 2-TOC via 4-TOC_CALC (SEMPRE)
    - DOC UIB: CSV UIB (si disponible)
    - DAD 254 (KHP): Export3d > CSV DAD > MasterFile 3-DAD_KHP

    Args:
        valid_sample_names: Set de noms de mostra normalitzats vàlids (per evitar
                           fallback matching quan el nom del fitxer és una mostra vàlida)

    Returns:
        dict amb dades trobades
    """
    config = config or get_config()
    result = {
        "has_data": False,
        "direct": None,  # DOC Direct del MasterFile
        "uib": None,     # DOC UIB del CSV
        "dad": None,
        "dad_source": None,  # "export3d", "csv", "masterfile"
    }

    sample_name = injection["sample_name"]
    inj_num = injection["inj_num"]
    line_num = injection["line_num"]
    sample_type = injection["sample_type"]
    original_name = injection["sample_name_original"]
    inj_num_original = injection.get("inj_num_original", inj_num)
    control_set = injection.get("control_set")  # Número de set per controls (MQ1, MQ2, ...)
    control_rep = injection.get("control_rep")  # Rèplica dins del set (R1, R2, ...)

    # 0. Extreure DOC Direct del MasterFile (SEMPRE si disponible)
    if toc_df is not None and toc_calc_df is not None and not toc_calc_df.empty:
        row_start = None
        row_end = None

        # Detectar format (nou vs antic)
        is_new_format = "Sample" in toc_calc_df.columns and "TOC_Row" in toc_calc_df.columns
        is_old_format = "SAMPLE_ID" in toc_calc_df.columns and "Row initial" in toc_calc_df.columns

        if is_new_format:
            # Format nou: 4-TOC_CALC amb Sample, TOC_Row, Inj_Index
            # Prioritzar cerca per line_num (Inj_Index) que és més fiable per controls
            mask = None

            # 1. Primer intentar per Inj_Index (line_num) - més fiable
            if "Inj_Index" in toc_calc_df.columns:
                mask = (toc_calc_df["Inj_Index"] == line_num)
                if not mask.any():
                    mask = None

            # 2. Si no, intentar per nom exacte amb rèplica
            if mask is None:
                sample_key_r = f"{original_name}_R{inj_num_original}"
                mask = (toc_calc_df["Sample"] == sample_key_r)
                if not mask.any():
                    # 3. Fallback: nom parcial + line_num
                    # Treure sufixos _1, _2 del original_name per matching més flexible
                    original_base = re.sub(r'[_\-]?\d+$', '', original_name).strip()
                    mask = (toc_calc_df["Sample"].str.contains(original_base, case=False, na=False))

            sample_rows = toc_calc_df[mask] if mask is not None else pd.DataFrame()
            if not sample_rows.empty:
                toc_rows = sample_rows["TOC_Row"].dropna()
                if len(toc_rows) > 0:
                    row_start = int(toc_rows.min())
                    row_end = int(toc_rows.max())

        elif is_old_format:
            # Format antic: 4-SEQ_DATA amb SAMPLE_ID, Row initial, Row Final
            # SAMPLE_ID pot ser numero (1, 2, 3...) o text (Pre, Post, MQ)
            mask = (toc_calc_df["SAMPLE_ID"].astype(str) == str(original_name))
            if not mask.any():
                # Provar amb Line# si existeix
                if "Line#" in toc_calc_df.columns:
                    mask = (toc_calc_df["Line#"] == line_num)

            sample_rows = toc_calc_df[mask]
            if not sample_rows.empty:
                # Usar primera fila que coincideixi
                row_data = sample_rows.iloc[0]
                try:
                    row_start = int(row_data.get("Row initial", 0))
                    row_end = int(row_data.get("Row Final", 0))
                except (ValueError, TypeError):
                    row_start = None
                    row_end = None

        if row_start is not None and row_end is not None and row_start > 0:
            # Extreure DOC Direct (amb truncat per última mostra)
            max_dur = config.get("max_duration_min", 80.0)
            df_doc, timeout_info = extract_doc_from_masterfile(
                toc_df, row_start, row_end, detect_timeouts=True,
                max_duration_min=max_dur
            )

            if not df_doc.empty:
                t_direct = df_doc["time (min)"].values
                y_direct = df_doc["DOC"].values

                result["direct"] = {
                    "path": "MasterFile:2-TOC",
                    "df": df_doc,
                    "t": t_direct,
                    "y": y_direct,
                    "row_start": row_start,
                    "row_end": row_end,
                    "timeout_info": timeout_info,
                }
                result["has_data"] = True

    # 1. Buscar UIB a CSV
    for uib_path in uib_files:
        if uib_path in used_files.get("uib", set()):
            continue

        filename = os.path.basename(uib_path)
        file_sample, file_rep = netejar_nom_uib(filename)

        # Match per nom original o nom únic
        file_sample_norm = normalize_key(file_sample)
        original_norm = normalize_key(original_name)
        unique_norm = normalize_key(sample_name)

        # També comparar amb el nom base sense sufix _1, _2, etc.
        # (El MasterFile pot tenir "NaOH 0.1mM_1" però el fitxer "NAOH 0.1MM")
        original_base_norm = re.sub(r'\d+$', '', original_norm)  # Treure números del final

        # Match directe
        name_match = (file_sample_norm == original_norm or
                      file_sample_norm == unique_norm or
                      file_sample_norm == original_base_norm)

        # Match per controls amb lògica Agilent:
        # - Set 1: unique_name="MQ", fitxers MQ_1.CSV, MQ_2.CSV
        # - Set 2: unique_name="MQ1", fitxers MQ1_1.CSV, MQ1_2.CSV
        # Fallback: Si no hi ha match directe, extreure base+número del fitxer
        # IMPORTANT: No usar fallback si el nom del fitxer és una mostra vàlida
        # (ex: no fer match MQ1 file amb MQ injection si MQ1 existeix com a mostra)
        file_control_num = None
        if not name_match:
            # Verificar si el nom del fitxer és una mostra vàlida (no usar fallback)
            file_is_valid_sample = (valid_sample_names and file_sample_norm in valid_sample_names)

            if not file_is_valid_sample:
                # Extreure base i número del nom del fitxer (fallback per dades antigues)
                match = re.match(r'^(.+?)(\d+)$', file_sample_norm)
                if match:
                    file_sample_base = match.group(1)
                    file_control_num = int(match.group(2))
                    # Comparar base amb original
                    if file_sample_base == original_norm:
                        name_match = True

        if name_match:
            # Verificar rèplica
            try:
                file_rep_int = int(file_rep) if file_rep else 1
            except ValueError:
                file_rep_int = 1

            # Lògica de matching rèplica:
            # - Si tenim control_set i file_control_num (fallback): comparar ambdós
            # - Altrament: comparar rèplica del fitxer amb inj_num
            if control_set is not None and file_control_num is not None:
                # Control: fitxer MQ1_R2 → set 1, rep 2
                rep_match = (file_control_num == control_set and file_rep_int == control_rep)
            elif file_control_num is not None:
                # Fallback: comparar número del nom amb inj_num
                rep_match = (file_control_num == inj_num)
            else:
                # Mostra normal: comparar rèplica del fitxer amb inj_num
                rep_match = (file_rep_int == inj_num)

            if rep_match:
                # Llegir UIB
                df_uib, status = llegir_doc_uib(uib_path)
                if status.startswith("OK"):
                    used_files.setdefault("uib", set()).add(uib_path)

                    t_uib = df_uib["time (min)"].values if not df_uib.empty else None
                    y_uib = df_uib["DOC"].values if not df_uib.empty else None

                    result["uib"] = {
                        "path": uib_path,
                        "df": df_uib,
                        "t": t_uib,
                        "y": y_uib,
                    }
                    result["has_data"] = True
                break

    # 2. Buscar DAD a Export3d (prioritat 1)
    for dad_path in dad_files:
        if dad_path in used_files.get("dad", set()):
            continue

        filename = os.path.basename(dad_path)
        file_sample, file_rep, _ = dad_sample_rep_from_path(dad_path)

        file_sample_norm = normalize_key(file_sample)
        original_norm = normalize_key(original_name)
        unique_norm = normalize_key(sample_name)

        # També comparar amb el nom base sense sufix numèric
        original_base_norm = re.sub(r'\d+$', '', original_norm)

        # Match directe
        name_match = (file_sample_norm == original_norm or
                      file_sample_norm == unique_norm or
                      file_sample_norm == original_base_norm)

        # Match per controls amb lògica Agilent (fallback per dades antigues)
        # IMPORTANT: No usar fallback si el nom del fitxer és una mostra vàlida
        file_control_num = None
        file_sample_base = None
        if not name_match:
            # Verificar si el nom del fitxer és una mostra vàlida (no usar fallback)
            file_is_valid_sample = (valid_sample_names and file_sample_norm in valid_sample_names)

            if not file_is_valid_sample:
                # Extreure base i número del nom del fitxer (MQ1 → base=MQ, num=1)
                match = re.match(r'^(.+?)(\d+)$', file_sample_norm)
                if match:
                    file_sample_base = match.group(1)
                    file_control_num = int(match.group(2))
                    # Comparar base amb original
                    if file_sample_base == original_norm or file_sample_base == original_base_norm:
                        name_match = True

        if name_match:
            try:
                file_rep_int = int(file_rep) if file_rep else 1
            except ValueError:
                file_rep_int = 1

            # Lògica de matching rèplica:
            # - Si tenim control_set i file_control_num (fallback): comparar ambdós
            # - Altrament: comparar rèplica del fitxer amb inj_num
            if control_set is not None and file_control_num is not None:
                # Control: fitxer MQ1_R2 → set 1, rep 2
                rep_match = (file_control_num == control_set and file_rep_int == control_rep)
            elif file_control_num is not None:
                # Fallback: comparar número del nom amb inj_num
                rep_match = (file_control_num == inj_num)
            else:
                # Mostra normal: comparar rèplica del fitxer amb inj_num
                rep_match = (file_rep_int == inj_num)

            if rep_match:
                df_dad, status = llegir_dad_export3d(dad_path)
                if status.startswith("OK"):
                    used_files.setdefault("dad", set()).add(dad_path)
                    result["dad"] = {
                        "path": dad_path,
                        "df": df_dad,
                    }
                    result["dad_source"] = "export3d"
                    result["has_data"] = True
                break

    # 2b. Fallback: buscar DAD a CSV (DAD1A format) si no hem trobat Export3d
    if result["dad"] is None and dad_csv_files:
        for dad_path in dad_csv_files:
            if dad_path in used_files.get("dad", set()):
                continue

            filename = os.path.basename(dad_path)
            # Extreure nom mostra del fitxer CSV (format típic: MOSTRA_DAD1A.csv)
            file_sample = filename.replace("_DAD1A", "").replace("_DAD", "")
            file_sample = os.path.splitext(file_sample)[0]

            # Intentar extreure rèplica del nom
            match = re.match(r'^(.+?)_R(\d+)$', file_sample, re.IGNORECASE)
            if match:
                file_sample_base = match.group(1)
                file_rep = int(match.group(2))
            else:
                file_sample_base = file_sample
                file_rep = 1

            file_sample_norm = normalize_key(file_sample_base)
            original_norm = normalize_key(original_name)

            if file_sample_norm == original_norm and file_rep == inj_num:
                df_dad, status = llegir_dad_1a(dad_path)
                if status.startswith("OK") and not df_dad.empty:
                    used_files.setdefault("dad", set()).add(dad_path)
                    result["dad"] = {
                        "path": dad_path,
                        "df": df_dad,
                    }
                    result["dad_source"] = "csv"
                    result["has_data"] = True
                    break

    # 3. Si és KHP i no tenim DAD, buscar a MasterFile 3-DAD_KHP
    if sample_type == "KHP" and result["dad"] is None and master_khp_data is not None:
        # Buscar columna corresponent al KHP
        # Format MasterFile 3-DAD_KHP: {SAMPLE}_{INJ}_R{REP} (e.g., KHP5_1_R1, KHP5_2_R2)
        # Per KHP controls: injection number = replica number
        khp_keys = [
            # Format correcte: SAMPLE_INJ_R{REP} (prioritari)
            f"{sample_name}_{inj_num}_R{inj_num}",  # KHP5_1_R1
            f"{original_name}_{inj_num}_R{inj_num}",
            # Formats alternatius
            f"{original_name}_R{inj_num}",
            f"{sample_name}_R{inj_num}",
            f"{original_name}_{inj_num}",
            f"{sample_name}_{inj_num}",
        ]
        # DEBUG: mostrar què busquem
        print(f"[DEBUG 3-DAD_KHP] Buscant KHP: sample={sample_name}, original={original_name}, inj={inj_num}")
        print(f"[DEBUG 3-DAD_KHP] Claus a buscar: {khp_keys}")
        all_cols = list(master_khp_data.columns)
        print(f"[DEBUG 3-DAD_KHP] Totes les columnes ({len(all_cols)}): {all_cols}")

        found = False
        for khp_key in khp_keys:
            if found:
                break
            khp_key_norm = normalize_key(khp_key)
            for col in master_khp_data.columns:
                col_norm = normalize_key(str(col))
                # Match exacte o parcial (la columna conté la clau)
                if col_norm == khp_key_norm or khp_key_norm in col_norm:
                    col_idx = master_khp_data.columns.get_loc(col)
                    print(f"[DEBUG 3-DAD_KHP] MATCH! col='{col}', idx={col_idx}, key='{khp_key}'")

                    # Format 3-DAD_KHP: columna KHP té temps, següent té valors
                    # Estructura: KHP5_1_R1 | Unnamed:1 | NaN | KHP5_2_R2 | Unnamed:4
                    #             time      | value     | NaN | time      | value
                    if col_idx + 1 < len(master_khp_data.columns):
                        time_col = col
                        value_col = master_khp_data.columns[col_idx + 1]

                        # Llegir dades saltant la primera fila (capçalera "time (min)", "value (mAU)")
                        time_data = master_khp_data[time_col].iloc[1:] if len(master_khp_data) > 1 else master_khp_data[time_col]
                        value_data = master_khp_data[value_col].iloc[1:] if len(master_khp_data) > 1 else master_khp_data[value_col]

                        df_khp = pd.DataFrame({
                            "time (min)": pd.to_numeric(time_data, errors="coerce"),
                            "254": pd.to_numeric(value_data, errors="coerce"),
                        }).dropna().reset_index(drop=True)

                        print(f"[DEBUG 3-DAD_KHP] df_khp len={len(df_khp)}, empty={df_khp.empty}")
                        if len(df_khp) > 0:
                            print(f"[DEBUG 3-DAD_KHP] Primeres files: {df_khp.head(3).to_dict()}")

                        if not df_khp.empty and len(df_khp) > 5:
                            result["dad"] = {
                                "path": "MasterFile:3-DAD_KHP",
                                "df": df_khp,
                                "source": "masterfile",
                            }
                            result["dad_source"] = "masterfile"
                            result["has_data"] = True
                            found = True
                            print(f"[DEBUG 3-DAD_KHP] SUCCESS! Carregades {len(df_khp)} files de 3-DAD_KHP")
                    break

        if not found:
            print(f"[DEBUG 3-DAD_KHP] NO MATCH per {sample_name} (inj {inj_num})")

    return result


# =============================================================================
# GENERACIÓ D'AVISOS ESTRUCTURATS PER IMPORTACIÓ
# =============================================================================

def _generate_import_warnings(result: dict) -> list:
    """
    Genera avisos estructurats a partir del resultat d'importació.

    Args:
        result: Dict del resultat de import_sequence()

    Returns:
        Llista d'avisos estructurats
    """
    warnings = []

    # 1. Errors crítics (BLOCKER)
    for error in result.get("errors", []):
        if "no data" in error.lower() or "buida" in error.lower() or "empty" in error.lower():
            warnings.append(create_warning(
                code="IMP_NO_DATA",
                stage="import",
            ))
        elif "uib" in error.lower() and ("missing" in error.lower() or "falt" in error.lower()):
            warnings.append(create_warning(
                code="IMP_MISSING_UIB",
                stage="import",
            ))
        elif "dad" in error.lower() and ("missing" in error.lower() or "falt" in error.lower()):
            warnings.append(create_warning(
                code="IMP_MISSING_DAD",
                stage="import",
            ))
        else:
            warnings.append(create_warning(
                code="IMP_ERROR",
                level=WarningLevel.BLOCKER,
                message=error,
                stage="import",
            ))

    # 2. Fitxers orfes
    orphan_uib = result.get("orphan_files", {}).get("uib", [])
    orphan_dad = result.get("orphan_files", {}).get("dad", [])
    n_orphan = len(orphan_uib) + len(orphan_dad)

    if n_orphan > 0:
        warnings.append(create_warning(
            code="IMP_ORPHAN_FILES",
            stage="import",
            details={"n": n_orphan, "uib": orphan_uib, "dad": orphan_dad},
        ))

    # 3. CSV buits
    n_empty_csv = 0
    for sample_name, sample_data in result.get("samples", {}).items():
        for rep_key, rep_data in sample_data.get("replicas", {}).items():
            uib_data = rep_data.get("uib") or {}
            if uib_data.get("t") is None and uib_data.get("y_raw") is None:
                n_empty_csv += 1

    if n_empty_csv > 0:
        warnings.append(create_warning(
            code="IMP_EMPTY_CSV",
            stage="import",
            details={"n": n_empty_csv},
        ))

    # 4. Fallback DAD (INFO)
    if result.get("dad_source") == "masterfile":
        warnings.append(create_warning(
            code="IMP_FALLBACK_DAD",
            stage="import",
        ))

    return warnings


def import_sequence(seq_path, config=None, progress_callback=None):
    """
    FASE 1: Importar dades RAW d'una seqüència (v2).

    FONT DE VERITAT: MasterFile fulla 1-HPLC-SEQ

    1. Llegeix MasterFile per obtenir llista completa d'injeccions
    2. Verifica correlativitat (warning si falten números)
    3. Per cada injecció, busca dades a CSV/Export3d/MasterFile
    4. Cross-check fitxers orfes

    Args:
        seq_path: Ruta a la carpeta de la seqüència
        config: Configuració (opcional)
        progress_callback: Funció callback(pct, msg) per progrés

    Returns:
        dict amb:
        - success: bool
        - seq_path: path original
        - seq_name: nom de la seqüència
        - method: "BP" o "COLUMN"
        - data_mode: "UIB", "DIRECT", o "DUAL"
        - injections: llista d'injeccions del MasterFile
        - samples: {nom: {type, replicas: {rep: {uib, dad}}}}
        - khp_samples, control_samples: llistes
        - orphan_files: fitxers no assignats
        - errors, warnings: llistes
    """
    config = config or get_config()

    result = {
        "success": False,
        "seq_path": seq_path,
        "seq_name": os.path.basename(seq_path),
        "method": "COLUMN",
        "data_mode": "UIB",
        "date": "",
        "master_data": None,
        "master_file": None,
        "master_format": None,
        "injections": [],
        "samples": {},
        "khp_samples": [],
        "control_samples": [],
        "valid_samples": set(),
        "orphan_files": {"uib": [], "dad": []},
        "file_check": None,
        "errors": [],
        "warnings": [],
    }

    def report_progress(pct, msg):
        if progress_callback:
            progress_callback(pct, msg)

    try:
        report_progress(5, "Detectant mode...")

        # 1. Detectar mode (BP/COLUMN)
        mode_info = detect_mode_from_folder(seq_path)
        result["method"] = mode_info["method"]
        result["data_mode"] = mode_info["data_mode"]

        report_progress(10, "Llegint MasterFile...")

        # 2. Llegir MasterFile (FONT DE VERITAT)
        master_path, master_format = trobar_excel_mestre(seq_path)
        result["master_format"] = master_format
        result["master_file"] = master_path

        if not master_path:
            report_progress(12, "MasterFile no trobat, intentant crear...")
            migrate_result = migrate_single(seq_path)

            if migrate_result.get("status") == "ok":
                master_path = migrate_result.get("file")
                master_format = "NEW"
                result["master_format"] = master_format
                result["master_file"] = master_path
                result["warnings"].append(
                    f"MasterFile creat des de rawdata ({migrate_result.get('rows', 0)} files TOC)"
                )
            elif migrate_result.get("status") == "exists":
                master_path = migrate_result.get("file")
                master_format = "NEW"
                result["master_format"] = master_format
                result["master_file"] = master_path

        if not master_path:
            result["errors"].append("No s'ha pogut obtenir MasterFile - font de veritat necessària")
            return result

        # Llegir MasterFile
        if master_format == "NEW":
            result["master_data"] = llegir_masterfile_nou(master_path)
        else:
            df_toc, df_seq = llegir_master_direct(master_path)
            result["master_data"] = {
                "toc": df_toc,
                "seq": df_seq,
                "filepath": master_path,
            }

        result["date"] = read_master_date(seq_path)

        # Extreure sensibilitat UIB si disponible (de 0-INFO B5)
        master_info = result["master_data"].get("info", {})
        uib_sensitivity = master_info.get("uib_sensitivity")
        if uib_sensitivity is not None:
            result["uib_sensitivity"] = uib_sensitivity

        report_progress(20, "Parsejant injeccions del MasterFile...")

        # 3. Parsejar injeccions del MasterFile (FONT DE VERITAT)
        injections, parse_warnings, master_line_count = parse_injections_from_masterfile(result["master_data"], config)
        result["injections"] = injections
        result["master_line_count"] = master_line_count  # Total línies al MasterFile (Line#)
        result["warnings"].extend(parse_warnings)

        if not injections:
            result["errors"].append("No s'han trobat injeccions al MasterFile")
            return result

        # Crear set de mostres vàlides
        for inj in injections:
            result["valid_samples"].add(inj["sample_name"])
            result["valid_samples"].add(inj["sample_name_original"])

        report_progress(30, "Llistant fitxers disponibles...")

        # 4. Llistar fitxers disponibles
        path_csv = os.path.join(seq_path, "CSV")
        uib_files = []
        dad_csv_files = []
        if os.path.isdir(path_csv):
            for ext in ("*.csv", "*.CSV", "*.txt", "*.TXT"):
                uib_files.extend(glob.glob(os.path.join(path_csv, f"*UIB1B*{ext}")))
                dad_csv_files.extend(glob.glob(os.path.join(path_csv, f"*DAD*{ext}")))
            uib_files = sorted(set(uib_files))
            dad_csv_files = sorted(set(dad_csv_files))

        path_3d = os.path.join(seq_path, "Export3d")
        if not os.path.isdir(path_3d):
            path_3d = os.path.join(seq_path, "Export3D")
        dad_files = list_dad_files(path_3d) if os.path.isdir(path_3d) else []

        # Llegir dades addicionals del MasterFile
        master_khp_data = None
        toc_df = None
        toc_calc_df = None
        has_3dad_khp_sheet = False
        try:
            xl = pd.ExcelFile(master_path)
            if "3-DAD_KHP" in xl.sheet_names:
                master_khp_data = pd.read_excel(master_path, sheet_name="3-DAD_KHP")
                has_3dad_khp_sheet = True
                print(f"[DEBUG] Full 3-DAD_KHP trobat: {master_khp_data.shape if master_khp_data is not None else 'None'}")
                if master_khp_data is not None:
                    print(f"[DEBUG] Columnes 3-DAD_KHP: {list(master_khp_data.columns)}")
            if "2-TOC" in xl.sheet_names:
                toc_df = pd.read_excel(master_path, sheet_name="2-TOC", header=6)
            if "4-TOC_CALC" in xl.sheet_names:
                toc_calc_df = pd.read_excel(master_path, sheet_name="4-TOC_CALC")
        except Exception as e:
            print(f"[WARNING] Error llegint fulls addicionals del MasterFile: {e}")

        # Warning si no hi ha Export3D ni 3-DAD_KHP però hi ha KHP
        has_export3d = os.path.isdir(path_3d) and len(dad_files) > 0
        has_khp = any(inj.get("sample_type") == "KHP" for inj in injections)
        if has_khp and not has_export3d and not has_3dad_khp_sheet:
            result["warnings"].append(
                "⚠️ ATENCIÓ: No s'ha trobat Export3D ni full 3-DAD_KHP al MasterFile. "
                "Les mostres KHP no tindran dades DAD 254nm."
            )
            print("[WARNING] KHP sense Export3D ni 3-DAD_KHP - DAD 254nm no disponible")

        report_progress(40, "Processant injeccions...")

        # 5. Processar cada injecció
        used_files = {"uib": set(), "dad": set()}
        total_inj = len(injections)

        # Crear set de noms de mostra normalitzats per evitar fallback matching incorrecte
        # Això evita que fitxer "MQ1" faci match amb injecció "MQ" quan "MQ1" és mostra vàlida
        valid_sample_names = {normalize_key(inj["sample_name"]) for inj in injections}

        for i, inj in enumerate(injections):
            pct = 40 + int(45 * (i + 1) / max(total_inj, 1))
            report_progress(pct, f"Processant {inj['sample_name']}...")

            sample_name = inj["sample_name"]
            sample_type = inj["sample_type"]
            inj_num = inj["inj_num"]

            # Buscar dades per aquesta injecció
            data = find_data_for_injection(
                inj, seq_path, uib_files, dad_files, dad_csv_files,
                master_khp_data, used_files, config,
                toc_df=toc_df, toc_calc_df=toc_calc_df,
                valid_sample_names=valid_sample_names
            )

            # Classificar mostra
            if sample_type == "KHP":
                if sample_name not in result["khp_samples"]:
                    result["khp_samples"].append(sample_name)
            elif sample_type == "CONTROL":
                if sample_name not in result["control_samples"]:
                    result["control_samples"].append(sample_name)

            # Crear entrada a samples
            if sample_name not in result["samples"]:
                result["samples"][sample_name] = {
                    "type": sample_type,
                    "original_name": inj["sample_name_original"],
                    "replicas": {}
                }

            # Processar dades DOC Direct si existeixen
            direct_data = data.get("direct")
            if direct_data and direct_data.get("t") is not None:
                t_direct = direct_data["t"]
                y_direct = direct_data["y"]
                y_direct_net = None
                baseline_direct = None

                if len(t_direct) > 10:
                    mode = "BP" if result["method"] == "BP" else "COLUMN"
                    baseline_direct = get_baseline_value(t_direct, y_direct, mode=mode)
                    y_direct_net = y_direct - baseline_direct

                direct_data["y_net"] = y_direct_net
                direct_data["baseline"] = baseline_direct

            # Processar dades UIB si existeixen
            uib_data = data.get("uib")
            if uib_data and uib_data.get("t") is not None:
                t_uib = uib_data["t"]
                y_uib = uib_data["y"]
                y_uib_net = None
                baseline_uib = None

                if len(t_uib) > 10:
                    mode = "BP" if result["method"] == "BP" else "COLUMN"
                    baseline_uib = get_baseline_value(t_uib, y_uib, mode=mode)
                    y_uib_net = y_uib - baseline_uib

                uib_data["y_net"] = y_uib_net
                uib_data["baseline"] = baseline_uib

            # Guardar rèplica
            rep_key = str(inj_num)
            # Detectar sobreescriptura (duplicats al MasterFile)
            if rep_key in result["samples"][sample_name]["replicas"]:
                result["warnings"].append(
                    f"⚠️ ATENCIÓ: '{sample_name}' rèplica {rep_key} duplicada (línia {inj['line_num']}) - revisar MasterFile"
                )
            result["samples"][sample_name]["replicas"][rep_key] = {
                "direct": direct_data,  # DOC Direct del MasterFile
                "uib": uib_data,        # DOC UIB del CSV (si disponible)
                "dad": data.get("dad"),
                "dad_source": data.get("dad_source"),
                "has_data": data.get("has_data", False),
                "injection_info": {
                    "line_num": inj["line_num"],
                    "inj_num": inj_num,
                    "inj_volume": inj.get("inj_volume"),  # Volum d'injecció en µL
                },
            }

        report_progress(90, "Verificant fitxers orfes...")

        # 6. Identificar fitxers orfes (no assignats a cap mostra)
        orphan_uib = [f for f in uib_files if f not in used_files["uib"]]
        orphan_dad = [f for f in dad_files if f not in used_files["dad"]]


        # 7. Matching intel·ligent: suggerir assignacions per orfes
        report_progress(92, "Calculant suggeriments de matching...")

        # Guardar tots els orfes ABANS d'aplicar suggeriments (per comptar punts)
        all_orphan_uib = orphan_uib.copy()
        all_orphan_dad = orphan_dad.copy()

        dad_suggestions = compute_orphan_suggestions(
            result["samples"], orphan_dad, file_type="dad"
        )
        uib_suggestions = compute_orphan_suggestions(
            result["samples"], orphan_uib, file_type="uib"
        )

        # Aplicar suggeriments d'alta confiança (>= 85%)
        dad_applied, orphan_dad = apply_orphan_suggestions(
            result["samples"], dad_suggestions, orphan_dad, file_type="dad"
        )
        uib_applied, orphan_uib = apply_orphan_suggestions(
            result["samples"], uib_suggestions, orphan_uib, file_type="uib"
        )

        # Guardar orfes restants (sense suggeriments aplicats)
        result["orphan_files"] = {
            "uib": orphan_uib,
            "dad": orphan_dad,
        }

        # Guardar TOTS els orfes originals (per comptar punts)
        result["all_orphan_files"] = {
            "uib": all_orphan_uib,
            "dad": all_orphan_dad,
        }

        result["orphan_suggestions"] = {
            "dad": {f"{k[0]}_R{k[1]}": v for k, v in dad_suggestions.items()},
            "uib": {f"{k[0]}_R{k[1]}": v for k, v in uib_suggestions.items()},
        }

        if dad_applied:
            result["warnings"].append(
                f"Suggerits {dad_applied} fitxers DAD per revisar (matching automàtic)"
            )
        if uib_applied:
            result["warnings"].append(
                f"Suggerits {uib_applied} fitxers UIB per revisar (matching automàtic)"
            )

        if orphan_uib:
            result["warnings"].append(
                f"Fitxers UIB orfes (no assignats): {len(orphan_uib)}"
            )
        if orphan_dad:
            result["warnings"].append(
                f"Fitxers DAD orfes (no assignats): {len(orphan_dad)}"
            )

        report_progress(95, "Finalitzant...")

        # 7. Estadístiques
        samples_with_data = sum(
            1 for s in result["samples"].values()
            for r in s["replicas"].values()
            if r.get("has_data")
        )
        samples_without_data = sum(
            1 for s in result["samples"].values()
            for r in s["replicas"].values()
            if not r.get("has_data")
        )

        result["stats"] = {
            "master_line_count": result.get("master_line_count", len(injections)),  # Line# al MasterFile
            "total_injections": len(injections),
            "total_samples": len(result["samples"]),
            "samples_with_data": samples_with_data,
            "samples_without_data": samples_without_data,
            "khp_count": len(result["khp_samples"]),
            "control_count": len(result["control_samples"]),
            "uib_files_used": len(used_files["uib"]),
            "dad_files_used": len(used_files["dad"]),
            "orphan_uib": len(orphan_uib),
            "orphan_dad": len(orphan_dad),
        }

        # Validar que els KHP tenen dades DOC (necessari per calibrar)
        khp_without_doc = []
        for khp_name in result["khp_samples"]:
            sample = result["samples"].get(khp_name, {})
            replicas = sample.get("replicas", {})
            has_any_doc = False
            for rep_data in replicas.values():
                direct = rep_data.get("direct") if rep_data else None
                uib = rep_data.get("uib") if rep_data else None
                direct_t = direct.get("t") if direct else None
                uib_t = uib.get("t") if uib else None
                if (direct_t is not None and len(direct_t) > 0) or \
                   (uib_t is not None and len(uib_t) > 0):
                    has_any_doc = True
                    break
            if not has_any_doc:
                khp_without_doc.append(khp_name)

        if khp_without_doc:
            result["warnings"].append(
                f"⚠️ KHP sense dades DOC: {', '.join(khp_without_doc)} - no es podrà calibrar (falten fitxers TOC?)"
            )
            result["khp_without_doc"] = khp_without_doc

        # Detectar si les PRIMERES mostres no tenen dades (DOC sobreescrit)
        # Ordenar per line_num per veure l'ordre d'injecció
        injection_order = []
        for sample_name, sample in result["samples"].items():
            for rep_key, rep_data in sample.get("replicas", {}).items():
                inj = rep_data.get("injection", {})
                line_num = inj.get("line_num", 999)
                has_data = rep_data.get("has_data", False)
                injection_order.append((line_num, sample_name, rep_key, has_data))

        injection_order.sort(key=lambda x: x[0])

        # Comptar primeres injeccions sense dades consecutives
        first_without_data = []
        found_with_data = False
        for line_num, sample_name, rep_key, has_data in injection_order:
            if has_data:
                found_with_data = True
                break
            else:
                first_without_data.append(f"{sample_name}_R{rep_key}")

        # Si hi ha primeres sense dades PERÒ després n'hi ha amb dades = sobreescrit
        if first_without_data and found_with_data:
            n_lost = len(first_without_data)
            samples_lost = list(dict.fromkeys([x.rsplit('_R', 1)[0] for x in first_without_data]))
            result["warnings"].append(
                f"⚠️ DOC SOBREESCRIT: Les primeres {n_lost} injeccions ({', '.join(samples_lost[:5])}"
                f"{'...' if len(samples_lost) > 5 else ''}) no tenen dades. "
                f"Tip: Esborra registre TOC abans de llençar SEQ o revisa durada."
            )
            result["doc_overwritten"] = {
                "n_lost": n_lost,
                "samples_lost": samples_lost,
                "first_with_data": injection_order[n_lost][1] if n_lost < len(injection_order) else None
            }

        # Detectar i TALLAR mostres amb dades ANÒMALES
        # 1. Calcular durada mitjana de les injeccions normals
        is_bp = result.get("method", "").upper() == "BP"
        min_points = 50 if is_bp else 200
        min_duration = 5.0 if is_bp else 30.0
        max_expected = 15.0 if is_bp else 85.0  # Límit inicial

        all_durations = []
        for sample_name, sample in result["samples"].items():
            for rep_key, rep_data in sample.get("replicas", {}).items():
                direct = rep_data.get("direct", {})
                if not direct:
                    continue
                t_array = direct.get("t")
                if t_array is not None and len(t_array) > min_points:
                    t_max = float(t_array[-1])
                    # Només incloure duracions "normals" per calcular mitjana
                    if min_duration < t_max < max_expected * 2:
                        all_durations.append(t_max)

        # Calcular durada referència (mediana de les normals)
        if all_durations:
            all_durations.sort()
            median_duration = all_durations[len(all_durations) // 2]
            # Permetre 5% extra per variabilitat
            trim_threshold = median_duration * 1.05
        else:
            trim_threshold = max_expected

        truncated_samples = []
        postrun_samples = []
        trimmed_samples = []

        for sample_name, sample in result["samples"].items():
            for rep_key, rep_data in sample.get("replicas", {}).items():
                direct = rep_data.get("direct", {})
                if not direct:
                    continue

                t_array = direct.get("t")
                y_array = direct.get("y")
                y_net = direct.get("y_net")

                if t_array is None or len(t_array) == 0:
                    continue

                import numpy as np
                t_array = np.asarray(t_array)
                n_points = len(t_array)
                t_max = float(t_array[-1])
                sample_id = f"{sample_name}_R{rep_key}"

                # Detectar dades truncades (massa poques)
                if n_points > 0 and (n_points < min_points or t_max < min_duration):
                    truncated_samples.append(f"{sample_id} ({n_points}pts, {t_max:.1f}min)")

                # Detectar i TALLAR postrun
                if t_max > trim_threshold * 1.5:  # >50% més llarg que normal
                    original_duration = t_max
                    original_points = n_points

                    # Trobar índex on tallar
                    trim_idx = np.searchsorted(t_array, trim_threshold)
                    if trim_idx < len(t_array) - 10:  # Deixar mínim 10 punts marge
                        # Tallar arrays
                        direct["t"] = t_array[:trim_idx].tolist()
                        if y_array is not None:
                            direct["y"] = np.asarray(y_array)[:trim_idx].tolist()
                        if y_net is not None:
                            direct["y_net"] = np.asarray(y_net)[:trim_idx].tolist()

                        # Actualitzar metadata
                        direct["t_max_original"] = original_duration
                        direct["n_points_original"] = original_points
                        direct["trimmed_at"] = trim_threshold
                        direct["postrun_removed"] = True

                        trimmed_samples.append(
                            f"{sample_id} ({original_duration:.0f}→{trim_threshold:.0f}min)"
                        )
                    else:
                        # No es pot tallar prou, només warning
                        postrun_samples.append(
                            f"{sample_id} ({t_max:.0f}min = {t_max/60:.1f}h)"
                        )

        if truncated_samples:
            result["warnings"].append(
                f"⚠️ DADES TRUNCADES: {', '.join(truncated_samples[:3])}"
                f"{'...' if len(truncated_samples) > 3 else ''} - poques dades, run interromput?"
            )
            result["truncated_samples"] = truncated_samples

        if trimmed_samples:
            # No warning - el tall és automàtic i transparent
            # Només guardem metadata per traçabilitat
            result["trimmed_samples"] = trimmed_samples
            result["trim_threshold"] = trim_threshold

        if postrun_samples:
            result["warnings"].append(
                f"⚠️ POSTRUN NO TALLAT: {', '.join(postrun_samples[:3])}"
                f"{'...' if len(postrun_samples) > 3 else ''} - revisar manualment"
            )
            result["postrun_samples"] = postrun_samples

        result["success"] = True
        report_progress(100, "Importació completada")

    except Exception as e:
        import traceback
        result["errors"].append(str(e))
        result["errors"].append(traceback.format_exc())

    # Generar avisos estructurats (nou sistema)
    result["warnings_structured"] = _generate_import_warnings(result)
    result["warning_level"] = get_max_warning_level(result["warnings_structured"])

    return result


# =============================================================================
# IMPORT PACK (SIBLINGS)
# =============================================================================

def extract_seq_num(seq_name):
    """Extreu el número de SEQ del nom (ex: '282B_SEQ' → 282)."""
    import re
    match = re.search(r'^(\d+)', os.path.basename(seq_name).replace('_SEQ', '').replace('_BP', ''))
    return int(match.group(1)) if match else 0


def detect_sibling_packs(seq_paths):
    """
    Agrupa carpetes per número de SEQ (siblings).

    Args:
        seq_paths: Llista de paths a carpetes SEQ

    Returns:
        dict: {seq_num: [paths]} on seq_num és el número base

    Example:
        Input: ['282_SEQ', '282B_SEQ', '283_SEQ']
        Output: {282: ['282_SEQ', '282B_SEQ'], 283: ['283_SEQ']}
    """
    packs = {}
    for path in seq_paths:
        seq_num = extract_seq_num(path)
        if seq_num not in packs:
            packs[seq_num] = []
        packs[seq_num].append(path)

    # Ordenar cada pack: base primer (282_SEQ), després amb lletra (282B_SEQ, 282C_SEQ)
    def sort_key(path):
        name = os.path.basename(path).replace('_SEQ', '').replace('_BP', '')
        # Extreure número i lletra (282 vs 282B)
        import re
        match = re.match(r'^(\d+)([A-Z]?)$', name, re.IGNORECASE)
        if match:
            num = int(match.group(1))
            letter = match.group(2).upper() if match.group(2) else ''
            # '' < 'A' < 'B' < 'C' ...
            return (num, letter)
        return (0, name)

    for seq_num in packs:
        packs[seq_num].sort(key=sort_key)

    return packs


def import_sequence_pack(seq_paths, config=None, progress_callback=None):
    """
    Importa múltiples carpetes siblings com un pack unificat.

    Quan l'equip s'atura i es reinicia, es creen carpetes com:
    282_SEQ, 282B_SEQ, 282C_SEQ

    Aquesta funció les importa totes i fusiona els resultats.

    Args:
        seq_paths: Llista de paths (siblings) o path únic
        config: Configuració (opcional)
        progress_callback: Callback(pct, msg)

    Returns:
        dict unificat amb totes les mostres dels siblings
    """
    # Si és un sol path, convertir a llista
    if isinstance(seq_paths, str):
        seq_paths = [seq_paths]

    if not seq_paths:
        return {"success": False, "errors": ["Cap path proporcionat"]}

    # Si només hi ha un path, importar directament
    if len(seq_paths) == 1:
        return import_sequence(seq_paths[0], config, progress_callback)

    def report_progress(pct, msg):
        if progress_callback:
            progress_callback(pct, msg)

    # Ordenar paths (282_SEQ abans de 282B_SEQ)
    seq_paths = sorted(seq_paths)
    primary_path = seq_paths[0]  # El principal és el primer (sense lletra)

    report_progress(0, f"Importació pack: {len(seq_paths)} siblings")

    # Importar cada sibling
    imported_results = []
    for i, path in enumerate(seq_paths):
        pct_start = int(100 * i / len(seq_paths))
        pct_end = int(100 * (i + 1) / len(seq_paths))

        def sub_progress(pct, msg):
            real_pct = pct_start + int((pct_end - pct_start) * pct / 100)
            report_progress(real_pct, f"[{os.path.basename(path)}] {msg}")

        result = import_sequence(path, config, sub_progress)
        imported_results.append(result)

    # Fusionar resultats
    report_progress(95, "Fusionant resultats...")
    merged = _merge_import_results(imported_results, primary_path)

    report_progress(100, f"Pack importat: {len(seq_paths)} siblings fusionats")

    return merged


def _merge_import_results(results, primary_path):
    """
    Fusiona múltiples resultats d'importació en un de sol.

    Args:
        results: Llista de dicts retornats per import_sequence()
        primary_path: Path principal (on es guarden els resultats)

    Returns:
        dict fusionat
    """
    if not results:
        return {"success": False, "errors": ["Cap resultat per fusionar"]}

    if len(results) == 1:
        return results[0]

    # Usar el primer com a base
    primary = results[0]

    merged = {
        "success": True,
        "seq_path": primary_path,
        "seq_name": os.path.basename(primary_path),
        "method": primary.get("method", "COLUMN"),
        "data_mode": primary.get("data_mode", "DUAL"),
        "date": primary.get("date", ""),
        "is_pack": True,
        "pack_sources": [r.get("seq_path") for r in results],
        "pack_count": len(results),
        "master_data": primary.get("master_data"),
        "master_file": primary.get("master_file"),
        "master_format": primary.get("master_format"),
        "injections": [],
        "samples": {},
        "khp_samples": [],
        "control_samples": [],
        "valid_samples": set(),
        "orphan_files": {"uib": [], "dad": []},
        "errors": [],
        "warnings": [],
    }

    # Afegir info de pack als warnings
    sibling_names = [os.path.basename(r.get("seq_path", "")) for r in results]
    merged["warnings"].append(
        f"📦 PACK FUSIONAT: {', '.join(sibling_names)}"
    )

    # Fusionar dades de cada resultat
    seen_samples = {}  # Per detectar duplicats

    for idx, result in enumerate(results):
        source_name = os.path.basename(result.get("seq_path", f"sibling_{idx}"))

        # Fusionar errors/warnings
        for err in result.get("errors", []):
            merged["errors"].append(f"[{source_name}] {err}")
        for warn in result.get("warnings", []):
            # No duplicar el warning de pack
            if "PACK FUSIONAT" not in warn:
                merged["warnings"].append(f"[{source_name}] {warn}")

        # Fusionar injeccions
        for inj in result.get("injections", []):
            inj_copy = inj.copy()
            inj_copy["source_seq"] = source_name
            merged["injections"].append(inj_copy)

        # Fusionar samples
        for sample_name, sample_data in result.get("samples", {}).items():
            if sample_name not in merged["samples"]:
                merged["samples"][sample_name] = {
                    "type": sample_data.get("type", "SAMPLE"),
                    "original_name": sample_data.get("original_name", sample_name),
                    "replicas": {},
                    "sources": []
                }

            # Afegir font
            merged["samples"][sample_name]["sources"].append(source_name)

            # Fusionar rèpliques (evitar sobreescriure)
            for rep_key, rep_data in sample_data.get("replicas", {}).items():
                # Crear clau única si ja existeix
                unique_key = rep_key
                base_key = rep_key
                counter = 1
                while unique_key in merged["samples"][sample_name]["replicas"]:
                    # Comprovar si són dades idèntiques (mateixa mostra processada 2 cops)
                    existing = merged["samples"][sample_name]["replicas"][unique_key]
                    if _replicas_identical(existing, rep_data):
                        break  # No afegir duplicat
                    unique_key = f"{base_key}_{source_name}"
                    counter += 1

                if unique_key not in merged["samples"][sample_name]["replicas"]:
                    rep_copy = rep_data.copy() if rep_data else {}
                    rep_copy["source_seq"] = source_name
                    merged["samples"][sample_name]["replicas"][unique_key] = rep_copy

        # Fusionar llistes KHP/Control
        for khp in result.get("khp_samples", []):
            if khp not in merged["khp_samples"]:
                merged["khp_samples"].append(khp)

        for ctrl in result.get("control_samples", []):
            if ctrl not in merged["control_samples"]:
                merged["control_samples"].append(ctrl)

        # Fusionar valid_samples
        merged["valid_samples"].update(result.get("valid_samples", set()))

        # Fusionar orphan_files
        merged["orphan_files"]["uib"].extend(result.get("orphan_files", {}).get("uib", []))
        merged["orphan_files"]["dad"].extend(result.get("orphan_files", {}).get("dad", []))

    # Estadístiques fusionades
    merged["stats"] = {
        "total_injections": len(merged["injections"]),
        "total_samples": len(merged["samples"]),
        "khp_count": len(merged["khp_samples"]),
        "control_count": len(merged["control_samples"]),
        "pack_sources": len(results),
    }

    # Check success
    merged["success"] = not any("error" in str(e).lower() for e in merged["errors"])

    return merged


def _replicas_identical(rep1, rep2):
    """Comprova si dues rèpliques tenen les mateixes dades (evitar duplicats)."""
    if not rep1 or not rep2:
        return False

    # Comparar per injection_info
    inj1 = rep1.get("injection_info", {})
    inj2 = rep2.get("injection_info", {})

    if inj1.get("line_num") == inj2.get("line_num") and \
       inj1.get("inj_num") == inj2.get("inj_num"):
        return True

    # Comparar per dades DOC
    direct1 = rep1.get("direct", {})
    direct2 = rep2.get("direct", {})

    t1 = direct1.get("t")
    t2 = direct2.get("t")

    if t1 is not None and t2 is not None:
        if len(t1) == len(t2) and len(t1) > 0:
            # Comparar primer i últim punt
            try:
                if abs(t1[0] - t2[0]) < 0.001 and abs(t1[-1] - t2[-1]) < 0.001:
                    return True
            except (IndexError, TypeError):
                pass

    return False


# =============================================================================
# GENERACIÓ DE MANIFEST JSON
# =============================================================================

def generate_import_manifest(imported_data, include_injection_details=True):
    """
    Genera un manifest JSON amb metadata de la importació.

    Útil per:
    - Traçabilitat: saber d'on venen les dades
    - Verificació: comprovar què s'ha detectat
    - Debugging: diagnosticar problemes d'importació

    Args:
        imported_data: Dict retornat per import_sequence()
        include_injection_details: Si True, inclou detall per cada injecció

    Returns:
        dict JSON-serializable amb tota la metadata
    """
    manifest = {
        "manifest_version": "1.0",
        "generated_at": datetime.now().isoformat(),
        "generator": f"hpsec_import v{__version__}",

        # Info seqüència
        "sequence": {
            "name": imported_data.get("seq_name", ""),
            "path": imported_data.get("seq_path", ""),
            "date": str(imported_data.get("date", "")),
            "method": imported_data.get("method", ""),  # COLUMN o BP
            "data_mode": imported_data.get("data_mode", ""),  # DUAL, DIRECT, UIB
            "uib_sensitivity": imported_data.get("uib_sensitivity"),  # Sensibilitat UIB (700, 1000, etc.)
        },

        # MasterFile
        "master_file": {
            "path": imported_data.get("master_file", ""),
            "format": imported_data.get("master_format", ""),  # NEW o OLD
            "filename": os.path.basename(imported_data.get("master_file", "")) if imported_data.get("master_file") else "",
        },

        # Estadístiques globals
        "stats": imported_data.get("stats", {}),

        # Avisos i errors
        "warnings": imported_data.get("warnings", []),
        "errors": imported_data.get("errors", []),

        # Fitxers orfes
        "orphan_files": {
            "uib": [os.path.basename(f) for f in imported_data.get("orphan_files", {}).get("uib", [])],
            "dad": [os.path.basename(f) for f in imported_data.get("orphan_files", {}).get("dad", [])],
        },

        # Suggeriments de matching (orfes → mostres)
        "orphan_suggestions": imported_data.get("orphan_suggestions", {"dad": {}, "uib": {}}),

        # Estat de revisió d'avisos
        "orphan_warning_dismissed": imported_data.get("orphan_warning_dismissed", False),
        "warnings_confirmed": imported_data.get("warnings_confirmed", False),
    }

    # Detall per mostra
    samples_detail = []
    for sample_name, sample_info in imported_data.get("samples", {}).items():
        sample_entry = {
            "name": sample_name,
            "original_name": sample_info.get("original_name", sample_name),  # Nom original del MasterFile
            "type": sample_info.get("type", "SAMPLE"),
            "replicas": [],
        }

        for rep_num, rep_data in sample_info.get("replicas", {}).items():
            replica_entry = {
                "replica": rep_num,
                "has_data": rep_data.get("has_data", False),
            }

            # DOC Direct
            direct = rep_data.get("direct") or {}
            if direct.get("t") is not None:
                t_arr = direct["t"]
                replica_entry["direct"] = {
                    "source": "MasterFile:2-TOC",
                    "row_start": direct.get("row_start"),
                    "row_end": direct.get("row_end"),
                    "n_points": len(t_arr),
                    "t_min": float(min(t_arr)),
                    "t_max": float(max(t_arr)),
                    "baseline": direct.get("baseline"),
                    "has_timeout": direct.get("timeout_info", {}).get("has_timeout", False),
                }
                if direct.get("timeout_info", {}).get("has_timeout"):
                    replica_entry["direct"]["timeout_ranges"] = direct["timeout_info"].get("timeout_ranges", [])
            elif direct.get("row_start") is not None or direct.get("n_points"):
                # Preservar metadades encara que no hi hagi dades reals
                replica_entry["direct"] = {
                    "source": "MasterFile:2-TOC",
                    "row_start": direct.get("row_start"),
                    "row_end": direct.get("row_end"),
                    "n_points": direct.get("n_points", 0),
                    "baseline": direct.get("baseline"),
                    "has_timeout": False,
                }

            # DOC UIB
            uib = rep_data.get("uib") or {}
            if uib.get("t") is not None:
                t_arr = uib["t"]
                replica_entry["uib"] = {
                    "source": "CSV",
                    "file": os.path.basename(uib.get("path", "") or uib.get("file", "")),
                    "n_points": len(t_arr),
                    "t_min": float(min(t_arr)),
                    "t_max": float(max(t_arr)),
                    "baseline": uib.get("baseline"),
                }
                # Afegir info d'assignació manual si existeix
                if uib.get("manual_assignment"):
                    replica_entry["uib"]["manual_assignment"] = True
                    replica_entry["uib"]["manual_file"] = uib.get("manual_file", "")
            elif uib.get("file") or uib.get("n_points") or uib.get("manual_file"):
                # Preservar metadades encara que no hi hagi dades reals
                replica_entry["uib"] = {
                    "source": "CSV",
                    "file": uib.get("file", ""),
                    "n_points": uib.get("n_points", 0),
                    "baseline": uib.get("baseline"),
                }
                if uib.get("manual_assignment"):
                    replica_entry["uib"]["manual_assignment"] = True
                    replica_entry["uib"]["manual_file"] = uib.get("manual_file", "")

            # DAD
            dad = rep_data.get("dad") or {}
            df = dad.get("df")
            if df is not None:
                t_col = df.columns[0]
                replica_entry["dad"] = {
                    "source": rep_data.get("dad_source", "unknown"),
                    "file": os.path.basename(dad.get("path", "") or dad.get("file", "")),
                    "n_points": len(df),
                    "n_wavelengths": len(df.columns) - 1,  # -1 per columna temps
                    "t_min": float(df[t_col].min()),
                    "t_max": float(df[t_col].max()),
                    "wavelengths_range": f"{df.columns[1]}-{df.columns[-1]}",
                }
                # Afegir info d'assignació manual si existeix
                if dad.get("manual_assignment"):
                    replica_entry["dad"]["manual_assignment"] = True
                    replica_entry["dad"]["manual_file"] = dad.get("manual_file") or dad.get("file", "")
            elif dad.get("file") or dad.get("manual_file"):
                # Preservar metadades encara que no hi hagi DataFrame
                replica_entry["dad"] = {
                    "source": rep_data.get("dad_source", "unknown"),
                    "file": dad.get("file", ""),
                    "n_points": 0,
                }
                if dad.get("manual_assignment"):
                    replica_entry["dad"]["manual_assignment"] = True
                    replica_entry["dad"]["manual_file"] = dad.get("manual_file") or dad.get("file", "")

            # Suggeriments de matching (si n'hi ha)
            dad_suggestion = rep_data.get("dad_suggestion")
            if dad_suggestion:
                replica_entry["dad_suggestion"] = {
                    "file": dad_suggestion.get("filename", ""),
                    "confidence": dad_suggestion.get("confidence", 0),
                    "match_type": dad_suggestion.get("match_type", ""),
                    "status": dad_suggestion.get("status", "SUGGESTED"),
                }

            uib_suggestion = rep_data.get("uib_suggestion")
            if uib_suggestion:
                replica_entry["uib_suggestion"] = {
                    "file": uib_suggestion.get("filename", ""),
                    "confidence": uib_suggestion.get("confidence", 0),
                    "match_type": uib_suggestion.get("match_type", ""),
                    "status": uib_suggestion.get("status", "SUGGESTED"),
                }

            # Info injecció original (si disponible)
            inj_info = rep_data.get("injection_info")
            if inj_info and include_injection_details:
                replica_entry["injection"] = {
                    "line_num": inj_info.get("line_num"),
                    "inj_num": inj_info.get("inj_num"),
                    "inj_volume": inj_info.get("inj_volume"),  # Volum d'injecció en µL
                    "location": inj_info.get("row_data", {}).get("Location"),
                    "acq_date": str(inj_info.get("row_data", {}).get("Injection Acquired Date", "")),
                    "method": inj_info.get("row_data", {}).get("Injection Acq Method Name"),
                }

            sample_entry["replicas"].append(replica_entry)

        samples_detail.append(sample_entry)

    manifest["samples"] = samples_detail

    # Resum per tipus
    manifest["summary"] = {
        "total_samples": len([s for s in samples_detail if s["type"] == "SAMPLE"]),
        "total_khp": len([s for s in samples_detail if s["type"] == "KHP"]),
        "total_pr": len([s for s in samples_detail if s["type"] == "PR"]),
        "total_replicas": sum(len(s["replicas"]) for s in samples_detail),
        "replicas_with_direct": sum(
            1 for s in samples_detail
            for r in s["replicas"]
            if "direct" in r
        ),
        "replicas_with_uib": sum(
            1 for s in samples_detail
            for r in s["replicas"]
            if "uib" in r
        ),
        "replicas_with_dad": sum(
            1 for s in samples_detail
            for r in s["replicas"]
            if "dad" in r
        ),
    }

    return manifest


def save_import_manifest(imported_data, output_path=None):
    """
    Genera i guarda el manifest JSON a un fitxer.

    Args:
        imported_data: Dict retornat per import_sequence()
        output_path: Ruta de sortida (default: SEQ_PATH/CHECK/data/import_manifest.json)

    Returns:
        Path del fitxer generat
    """
    manifest = generate_import_manifest(imported_data)

    if output_path is None:
        seq_path = imported_data.get("seq_path", ".")
        data_folder = get_data_folder(seq_path, create=True)
        output_path = os.path.join(data_folder, "import_manifest.json")

    # Assegurar que la carpeta existeix
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return output_path


def load_manifest(seq_path):
    """
    Carrega el manifest JSON si existeix.

    Busca primer a la nova ubicació (CHECK/data/) i després a l'antiga (arrel SEQ).

    Args:
        seq_path: Ruta a la carpeta SEQ

    Returns:
        dict amb manifest o None si no existeix
    """
    # Nova ubicació: CHECK/data/import_manifest.json
    data_folder = get_data_folder(seq_path, create=False)
    new_path = os.path.join(data_folder, "import_manifest.json")

    # Antiga ubicació: SEQ/import_manifest.json (compatibilitat)
    old_path = os.path.join(seq_path, "import_manifest.json")

    # Prioritzar nova ubicació
    for manifest_path in [new_path, old_path]:
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                continue
    return None


def import_from_manifest(seq_path, manifest=None, config=None, progress_callback=None):
    """
    Reimporta dades guiat pel manifest (molt més ràpid).

    En lloc de detectar/matching, llegeix directament d'on indica el manifest:
    - DOC Direct: files row_start-row_end del MasterFile
    - UIB: fitxer CSV indicat
    - DAD: segons source (export3d o dad1a)

    Args:
        seq_path: Ruta a la carpeta SEQ
        manifest: Manifest carregat (si None, el carrega)
        config: Configuració
        progress_callback: Funció callback(pct, msg)

    Returns:
        dict equivalent a import_sequence()
    """
    config = config or get_config()

    def report_progress(pct, msg):
        if progress_callback:
            progress_callback(pct, msg)

    # Carregar manifest si no proporcionat
    if manifest is None:
        manifest = load_manifest(seq_path)
        if manifest is None:
            # Fallback a import normal
            report_progress(0, "Manifest no trobat, important normalment...")
            return import_sequence(seq_path, config, progress_callback)

    report_progress(5, "Llegint manifest...")

    # Verificar si el manifest és antic (no té informació DOC Direct)
    manifest_samples = manifest.get("samples", [])
    has_direct_info = any(
        rep.get("direct") is not None
        for sample in manifest_samples
        for rep in sample.get("replicas", [])
    )

    if not has_direct_info and manifest_samples:
        # Manifest antic sense informació DOC Direct - reimportar completament
        report_progress(0, "Manifest antic detectat, reimportant...")
        print("[INFO] Manifest antic sense DOC Direct, fent reimportació completa")
        return import_sequence(seq_path, config, progress_callback)

    # Inicialitzar resultat
    seq_info = manifest.get("sequence", {})
    mf_info = manifest.get("master_file", {})

    result = {
        "success": False,
        "seq_path": seq_path,
        "seq_name": seq_info.get("name", os.path.basename(seq_path)),
        "method": seq_info.get("method", "COLUMN"),
        "data_mode": seq_info.get("data_mode", "DUAL"),
        "uib_sensitivity": seq_info.get("uib_sensitivity"),  # Restaurar sensibilitat UIB
        "date": seq_info.get("date", ""),
        "master_file": mf_info.get("path", ""),
        "master_format": mf_info.get("format", "NEW"),
        "master_data": None,
        "injections": [],
        "samples": {},
        "khp_samples": [],
        "control_samples": [],
        "valid_samples": set(),
        "orphan_files": {"uib": [], "dad": []},
        "errors": [],
        "warnings": ["Importat des de manifest existent"],
        "from_manifest": True,
        "manifest_date": manifest.get("generated_at", ""),
        "orphan_warning_dismissed": manifest.get("orphan_warning_dismissed", False),
        "warnings_confirmed": manifest.get("warnings_confirmed", False),
    }

    # Verificar que MasterFile existeix
    master_path = mf_info.get("path", "")
    if not master_path or not os.path.exists(master_path):
        # Intentar trobar-lo
        master_path_new, _ = trobar_excel_mestre(seq_path)
        if master_path_new:
            master_path = master_path_new
            result["master_file"] = master_path
        else:
            result["errors"].append("MasterFile no trobat")
            return result

    report_progress(10, "Llegint MasterFile...")

    # Llegir MasterFile per obtenir dades DOC Direct
    try:
        toc_df = None
        xl = pd.ExcelFile(master_path, engine="openpyxl")
        if "2-TOC" in xl.sheet_names:
            toc_df = pd.read_excel(xl, sheet_name="2-TOC", header=6, engine="openpyxl")
    except Exception as e:
        result["errors"].append(f"Error llegint MasterFile: {e}")
        return result

    report_progress(20, "Processant mostres del manifest...")

    # Processar cada mostra del manifest
    manifest_samples = manifest.get("samples", [])
    total = len(manifest_samples)

    for i, sample_info in enumerate(manifest_samples):
        pct = 20 + int((i / total) * 70) if total > 0 else 90
        report_progress(pct, f"Llegint {sample_info.get('name', '?')}...")

        sample_name = sample_info.get("name", "")
        sample_type = sample_info.get("type", "SAMPLE")

        if sample_name not in result["samples"]:
            result["samples"][sample_name] = {
                "type": sample_type,
                "original_name": sample_info.get("original_name", sample_name),  # Nom original del MasterFile
                "replicas": {},
            }

        # Classificar
        if sample_type == "KHP" and sample_name not in result["khp_samples"]:
            result["khp_samples"].append(sample_name)
        elif sample_type == "CONTROL" and sample_name not in result["control_samples"]:
            result["control_samples"].append(sample_name)

        result["valid_samples"].add(sample_name)

        # Processar cada rèplica
        for rep_info in sample_info.get("replicas", []):
            rep_num = str(rep_info.get("replica", "1"))

            rep_data = {
                "direct": None,
                "uib": None,
                "dad": None,
                "dad_source": None,
                "has_data": False,
                "injection_info": rep_info.get("injection"),
            }

            # === DOC Direct ===
            direct_info = rep_info.get("direct")
            if direct_info:
                row_start = direct_info.get("row_start")
                row_end = direct_info.get("row_end")

                # Preservar metadades del manifest encara que no es puguin llegir les dades
                rep_data["direct"] = {
                    "path": f"MasterFile:2-TOC",
                    "df": None,
                    "t": None,
                    "y": None,
                    "row_start": row_start,
                    "row_end": row_end,
                    "n_points": direct_info.get("n_points", 0),
                    "timeout_info": {},
                    "y_net": None,
                    "baseline": None,
                }

                # Intentar llegir les dades reals si tenim MasterFile
                if toc_df is not None and row_start is not None and row_end is not None:
                    try:
                        max_dur = config.get("max_duration_min", 80.0)
                        df_doc, timeout_info = extract_doc_from_masterfile(
                            toc_df, row_start, row_end,
                            t_start=None, detect_timeouts=True,
                            max_duration_min=max_dur
                        )
                        if df_doc is not None and not df_doc.empty:
                            t_direct = df_doc["time (min)"].values
                            y_direct = df_doc["DOC"].values

                            # Calcular baseline
                            mode = "BP" if result["method"] == "BP" else "COLUMN"
                            baseline = get_baseline_value(t_direct, y_direct, mode=mode)
                            y_net = np.array(y_direct) - baseline

                            rep_data["direct"] = {
                                "path": f"MasterFile:2-TOC",
                                "df": df_doc,
                                "t": t_direct,
                                "y": y_direct,
                                "row_start": row_start,
                                "row_end": row_end,
                                "n_points": len(t_direct),
                                "timeout_info": timeout_info,
                                "y_net": y_net,
                                "baseline": baseline,
                            }
                            rep_data["has_data"] = True
                    except Exception as e:
                        result["warnings"].append(f"{sample_name} rep {rep_num} Direct: {e}")

            # === DOC UIB ===
            uib_info = rep_info.get("uib")
            if uib_info:
                # Prioritzar manual_file si existeix (assignació manual de l'usuari)
                uib_file = uib_info.get("manual_file") or uib_info.get("file", "")

                # Preservar metadades del manifest encara que no es puguin llegir les dades
                rep_data["uib"] = {
                    "path": uib_file,
                    "df": None,
                    "t": None,
                    "y": None,
                    "file": uib_file,
                    "n_points": uib_info.get("n_points", 0),
                    "y_net": None,
                    "baseline": None,
                    "manual_assignment": uib_info.get("manual_assignment", False),
                }

                if uib_file:
                    # Buscar fitxer UIB
                    uib_path = os.path.join(seq_path, "CSV", uib_file)
                    if not os.path.exists(uib_path):
                        # Provar altres ubicacions
                        for subdir in ["", "CSV", "csv"]:
                            test_path = os.path.join(seq_path, subdir, uib_file) if subdir else os.path.join(seq_path, uib_file)
                            if os.path.exists(test_path):
                                uib_path = test_path
                                break

                    if os.path.exists(uib_path):
                        try:
                            df_uib, status = llegir_doc_uib(uib_path)
                            if not df_uib.empty and "OK" in status:
                                t_uib = df_uib["time (min)"].values
                                y_uib = df_uib["DOC"].values

                                # Baseline
                                mode = "BP" if result["method"] == "BP" else "COLUMN"
                                baseline = get_baseline_value(t_uib, y_uib, mode=mode)
                                y_net = np.array(y_uib) - baseline

                                rep_data["uib"] = {
                                    "path": uib_path,
                                    "df": df_uib,
                                    "t": t_uib,
                                    "y": y_uib,
                                    "file": uib_file,
                                    "n_points": len(t_uib),
                                    "y_net": y_net,
                                    "baseline": baseline,
                                }
                                rep_data["has_data"] = True
                        except Exception as e:
                            result["warnings"].append(f"{sample_name} rep {rep_num} UIB: {e}")

            # === DAD ===
            dad_info = rep_info.get("dad")
            if dad_info:
                dad_source = dad_info.get("source", "export3d")
                rep_data["dad_source"] = dad_source

                # Prioritzar manual_file si existeix (assignació manual de l'usuari)
                manual_dad_file = dad_info.get("manual_file")
                dad_file_from_manifest = dad_info.get("file", "")

                dad_loaded = False

                # 1. Intentar carregar des de manual_file
                if manual_dad_file and not dad_loaded:
                    # Buscar el fitxer manual
                    dad_dirs = ["Export3d", "Export3D", "CSV", "csv", ""]
                    for subdir in dad_dirs:
                        test_path = os.path.join(seq_path, subdir, manual_dad_file) if subdir else os.path.join(seq_path, manual_dad_file)
                        if os.path.exists(test_path):
                            try:
                                df_dad, status = llegir_dad_export3d(test_path)
                                if df_dad is not None and status.startswith("OK"):
                                    rep_data["dad"] = {
                                        "df": df_dad,
                                        "path": test_path,
                                        "file": manual_dad_file,
                                        "manual_assignment": True,
                                    }
                                    rep_data["dad_source"] = "export3d"
                                    rep_data["has_data"] = True
                                    dad_loaded = True
                                    break
                            except Exception:
                                pass
                    if not dad_loaded:
                        # Intentar amb dad1a
                        for subdir in dad_dirs:
                            test_path = os.path.join(seq_path, subdir, manual_dad_file) if subdir else os.path.join(seq_path, manual_dad_file)
                            if os.path.exists(test_path):
                                try:
                                    df_dad, status = llegir_dad_1a(test_path)
                                    if df_dad is not None and status.startswith("OK"):
                                        rep_data["dad"] = {
                                            "df": df_dad,
                                            "path": test_path,
                                            "file": manual_dad_file,
                                            "manual_assignment": True,
                                        }
                                        rep_data["dad_source"] = "dad1a"
                                        rep_data["has_data"] = True
                                        dad_loaded = True
                                        break
                                except Exception:
                                    pass

                # 2. Intentar carregar des de file guardat al manifest
                if dad_file_from_manifest and not dad_loaded:
                    dad_dirs = ["Export3d", "Export3D", "CSV", "csv", ""]
                    for subdir in dad_dirs:
                        test_path = os.path.join(seq_path, subdir, dad_file_from_manifest) if subdir else os.path.join(seq_path, dad_file_from_manifest)
                        if os.path.exists(test_path):
                            try:
                                df_dad, status = llegir_dad_export3d(test_path)
                                if df_dad is not None and status.startswith("OK"):
                                    rep_data["dad"] = {"df": df_dad, "path": test_path, "file": dad_file_from_manifest}
                                    rep_data["dad_source"] = "export3d"
                                    rep_data["has_data"] = True
                                    dad_loaded = True
                                    break
                            except Exception:
                                pass

                # 3. Fallback: buscar per nom de mostra (comportament original)
                if not dad_loaded and dad_source == "export3d":
                    path_3d = os.path.join(seq_path, "Export3d")
                    if not os.path.isdir(path_3d):
                        path_3d = os.path.join(seq_path, "Export3D")

                    if os.path.isdir(path_3d):
                        # Buscar fitxer per aquesta mostra
                        dad_files = list_dad_files(path_3d)
                        for df_path in dad_files:
                            # Match per nom mostra
                            fname = os.path.basename(df_path).upper()
                            sname = sample_name.upper().replace(" ", "").replace("_", "")
                            if sname in fname.replace(" ", "").replace("_", ""):
                                try:
                                    df_dad, status = llegir_dad_export3d(df_path)
                                    if df_dad is not None and status.startswith("OK"):
                                        rep_data["dad"] = {"df": df_dad, "path": df_path}
                                        rep_data["dad_source"] = "export3d"
                                        rep_data["has_data"] = True
                                        break
                                except Exception:
                                    pass

            result["samples"][sample_name]["replicas"][rep_num] = rep_data

    # Stats
    samples_with_data = sum(
        1 for s in result["samples"].values()
        for r in s.get("replicas", {}).values()
        if r.get("has_data")
    )

    result["stats"] = {
        "total_samples": len(result["samples"]),
        "samples_with_data": samples_with_data,
        "from_manifest": True,
    }

    # Recuperar informació d'orfes i suggeriments del manifest
    orphan_info = manifest.get("orphan_files", {})

    # Reconstruir paths complets per fitxers orfes
    def find_full_path(filename, search_dirs):
        """Busca el path complet d'un fitxer."""
        for subdir in search_dirs:
            test_path = os.path.join(seq_path, subdir, filename) if subdir else os.path.join(seq_path, filename)
            if os.path.exists(test_path):
                return test_path
        return None

    # Directoris on buscar cada tipus
    dad_dirs = ["Export3d", "Export3D", "CSV", "csv", ""]
    uib_dirs = ["CSV", "csv", ""]

    # Reconstruir paths per orfes UIB
    orphan_uib_full = []
    for fname in orphan_info.get("uib", []):
        full_path = find_full_path(fname, uib_dirs)
        if full_path:
            orphan_uib_full.append(full_path)

    # Reconstruir paths per orfes DAD
    orphan_dad_full = []
    for fname in orphan_info.get("dad", []):
        full_path = find_full_path(fname, dad_dirs)
        if full_path:
            orphan_dad_full.append(full_path)

    result["orphan_files"] = {
        "uib": orphan_uib_full,
        "dad": orphan_dad_full,
    }

    # Llista completa d'orfes (incloent suggeriments)
    all_orphan_uib = list(orphan_uib_full)
    all_orphan_dad = list(orphan_dad_full)

    # Afegir suggeriments des del manifest a cada rèplica
    for sample_info in manifest_samples:
        sample_name = sample_info.get("name", "")
        if sample_name not in result["samples"]:
            continue

        for rep_info in sample_info.get("replicas", []):
            rep_num = str(rep_info.get("replica", "1"))
            rep_data = result["samples"][sample_name]["replicas"].get(rep_num)
            if not rep_data:
                continue

            # Recuperar suggeriments DAD
            dad_suggestion = rep_info.get("dad_suggestion")
            if dad_suggestion:
                suggested_file = dad_suggestion.get("file", "")
                rep_data["dad_suggestion"] = {
                    "file": suggested_file,
                    "filename": suggested_file,
                    "confidence": dad_suggestion.get("confidence", 0),
                    "match_type": dad_suggestion.get("match_type", "SUGGESTED"),
                    "status": dad_suggestion.get("status", "SUGGESTED"),
                }
                # Afegir a llista completa d'orfes si no hi és
                if suggested_file:
                    full_path = find_full_path(suggested_file, dad_dirs)
                    if full_path and full_path not in all_orphan_dad:
                        all_orphan_dad.append(full_path)

            # Recuperar suggeriments UIB
            uib_suggestion = rep_info.get("uib_suggestion")
            if uib_suggestion:
                suggested_file = uib_suggestion.get("file", "")
                rep_data["uib_suggestion"] = {
                    "file": suggested_file,
                    "filename": suggested_file,
                    "confidence": uib_suggestion.get("confidence", 0),
                    "match_type": uib_suggestion.get("match_type", "SUGGESTED"),
                    "status": uib_suggestion.get("status", "SUGGESTED"),
                }
                # Afegir a llista completa d'orfes si no hi és
                if suggested_file:
                    full_path = find_full_path(suggested_file, uib_dirs)
                    if full_path and full_path not in all_orphan_uib:
                        all_orphan_uib.append(full_path)

    result["all_orphan_files"] = {
        "uib": all_orphan_uib,
        "dad": all_orphan_dad,
    }

    result["success"] = True
    report_progress(100, "Importació des de manifest completada")

    return result


# =============================================================================
# FUNCIONS AUXILIARS PER COMPATIBILITAT
# =============================================================================

# Re-exportar funcions que altres mòduls poden necessitar
__all__ = [
    # Carpeta dades
    "get_data_folder",
    "DATA_FOLDER_NAME",
    # Utilitats
    "normalize_key",
    "normalize_rep",
    "is_khp",
    "is_control_injection",
    "obtenir_seq",
    "seq_tag",
    "split_sample_rep",
    "clean_sample_name",
    "skip_sample_direct",
    # Detecció mode
    "is_bp_seq",
    "detect_mode_from_folder",
    # Baseline (usar get_baseline_value de hpsec_utils directament)
    # Lectura/creació master
    "migrate_single",
    "detect_master_format",
    "trobar_excel_mestre",
    "read_master_date",
    "llegir_masterfile_nou",
    "llegir_master_direct",
    # Lectura UIB
    "llegir_doc_uib",
    "netejar_nom_uib",
    # Lectura DAD
    "llegir_dad_export3d",
    "llegir_dad_1a",
    "llegir_dad_amb_fallback",
    "list_dad_files",
    # Matching
    "get_valid_samples_from_hplc_seq",
    "match_sample_confidence",
    "is_sample_in_seq",
    "CONFIDENCE_THRESHOLD",
    # Matching intel·ligent d'orfes
    "extract_sample_from_filename",
    "compute_orphan_suggestions",
    "apply_orphan_suggestions",
    # DAD matching
    "detect_dad_rep_style",
    "dad_sample_rep_from_path",
    "target_keys_from_desc",
    "choose_best_candidate",
    # Verificació
    "detect_replica_anomalies",
    "check_sequence_files",
    # Principal
    "import_sequence",
    # Manifest
    "generate_import_manifest",
    "save_import_manifest",
    "load_manifest",
    "import_from_manifest",
]
