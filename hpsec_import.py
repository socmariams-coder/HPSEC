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
from hpsec_utils import mode_robust
from hpsec_core import detect_timeout
from hpsec_migrate_master import migrate_single

# =============================================================================
# CONSTANTS
# =============================================================================
CONFIDENCE_THRESHOLD = 85.0  # Llindar per acceptar match automàticament

# Configuració baseline per defecte
DEFAULT_BASELINE_CONFIG = {
    "bp_baseline_win": 1.0,          # Finestra baseline per BP (min)
    "col_baseline_start": 10.0,      # Inici finestra baseline per COLUMN (min)
}


# =============================================================================
# CORRECCIÓ BASELINE
# =============================================================================
def get_baseline_correction(t, y, mode_type="COL", config=None, use_end=False):
    """
    Correcció de baseline segons mode BP o COLUMN.

    Args:
        t: array de temps (min)
        y: array de senyal
        mode_type: "BP" o "COL"
        config: configuració amb bp_baseline_win i col_baseline_start
        use_end: si True, usa el final del cromatograma per baseline (recomanat per BP)

    Returns:
        float: valor de baseline (constant)
    """
    config = config or DEFAULT_BASELINE_CONFIG
    t = np.asarray(t)
    y = np.asarray(y)

    if len(y) < 10:
        return float(np.nanmin(y)) if len(y) > 0 else 0.0

    if mode_type == "BP":
        if use_end:
            # Per BP: usar els últims punts (després del pic)
            n = len(y)
            n_edge = max(20, n // 5)  # Últim 20%
            return float(np.median(y[-n_edge:]))
        else:
            # Per BP: usar finestra inicial
            mask = t < config.get("bp_baseline_win", 1.0)
            if np.sum(mask) > 10:
                return float(mode_robust(y[mask]))
            return float(np.nanmin(y))

    # COLUMN: usar primers punts
    mask = t < config.get("col_baseline_start", 10.0)
    if np.sum(mask) > 10:
        return float(mode_robust(y[mask]))
    return float(np.nanmin(y))


# =============================================================================
# UTILITATS BÀSIQUES
# =============================================================================

def normalize_key(s):
    """Normalitza string per matching (elimina caràcters especials, majúscules)."""
    return re.sub(r"[^A-Za-z0-9]+", "", str(s or "")).upper()


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


def extract_doc_from_master(df_toc, row_ini, row_fi, start_dt, detect_timeouts=True):
    """
    Extreu segment DOC del format antic de master.

    Args:
        df_toc: DataFrame de 2-TOC
        row_ini: Fila inicial
        row_fi: Fila final
        start_dt: Datetime d'inici
        detect_timeouts: Si True, detecta timeouts

    Returns:
        Si detect_timeouts=False: DataFrame amb columns "time (min)" i "DOC"
        Si detect_timeouts=True: tuple (DataFrame, timeout_info dict)
    """
    # Reutilitzar la funció del nou format
    return extract_doc_from_masterfile(df_toc, row_ini, row_fi, start_dt, detect_timeouts)


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
    Llista tots els fitxers DAD (excloent UIB).

    Returns:
        Llista de paths a fitxers DAD
    """
    dad_files = []

    if folder_export3d and os.path.isdir(folder_export3d):
        for ext in ("*.csv", "*.CSV"):
            dad_files.extend(glob.glob(os.path.join(folder_export3d, ext)))

    # Excloure fitxers UIB
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
        return [], ["No s'ha trobat fulla 1-HPLC-SEQ al MasterFile"]

    # Identificar columnes
    sample_col = None
    inj_col = None
    line_col = None

    for col in df_seq.columns:
        col_lower = str(col).lower().strip()
        if "sample" in col_lower and "name" in col_lower:
            sample_col = col
        elif col_lower == "inj#" or col_lower == "inj":
            inj_col = col
        elif col_lower == "line#" or col_lower == "line":
            line_col = col

    if sample_col is None:
        # Fallback: buscar columna amb "sample"
        for col in df_seq.columns:
            if "sample" in str(col).lower():
                sample_col = col
                break

    if sample_col is None:
        return [], ["No s'ha trobat columna 'Sample Name' al MasterFile"]

    # Comptador per controls repetits
    control_counts = {}
    control_patterns = ["naoh", "mq", "blank", "buffer", "wash"]

    # Processar cada fila
    prev_line = 0
    for idx, row in df_seq.iterrows():
        sample_name = str(row.get(sample_col, "")).strip()
        if not sample_name or sample_name.lower() in ["nan", ""]:
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

        # Detectar si és control repetit
        sample_lower = sample_name.lower()
        is_repeated_control = any(p in sample_lower for p in control_patterns)

        # Assignar nom únic per controls repetits
        if is_repeated_control:
            # Normalitzar nom base (sense _1, _2 si ja existeix)
            base_name = re.sub(r'[_\-]?\d+$', '', sample_name).strip()
            if not base_name:
                base_name = sample_name

            # Incrementar comptador
            if base_name not in control_counts:
                control_counts[base_name] = 0
            control_counts[base_name] += 1

            unique_name = f"{base_name}_{control_counts[base_name]}"
            sample_type = "CONTROL"
            # Controls repetits: cada un és una "mostra" independent amb replica 1
            effective_inj_num = 1
        else:
            unique_name = sample_name
            effective_inj_num = inj_num
            # Determinar tipus
            if is_khp(sample_name):
                sample_type = "KHP"
            elif is_control_injection(sample_name, config):
                sample_type = "CONTROL"
            elif "test" in sample_lower:
                sample_type = "TEST"
            else:
                sample_type = "SAMPLE"

        injections.append({
            "line_num": line_num,
            "inj_num": effective_inj_num,  # Replica efectiva
            "inj_num_original": inj_num,   # Replica original del MasterFile
            "sample_name_original": sample_name,
            "sample_name": unique_name,
            "sample_type": sample_type,
            "row_data": row.to_dict(),
        })

    return injections, warnings


def find_data_for_injection(injection, seq_path, uib_files, dad_files, dad_csv_files,
                            master_khp_data, used_files, config=None,
                            toc_df=None, toc_calc_df=None):
    """
    Busca dades per una injecció des de múltiples fonts.

    Prioritat:
    - DOC Direct: MasterFile 2-TOC via 4-TOC_CALC (SEMPRE)
    - DOC UIB: CSV UIB (si disponible)
    - DAD 254 (KHP): Export3d > CSV DAD > MasterFile 3-DAD_KHP

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

    # 0. Extreure DOC Direct del MasterFile (SEMPRE si disponible)
    if toc_df is not None and toc_calc_df is not None and not toc_calc_df.empty:
        row_start = None
        row_end = None

        # Detectar format (nou vs antic)
        is_new_format = "Sample" in toc_calc_df.columns and "TOC_Row" in toc_calc_df.columns
        is_old_format = "SAMPLE_ID" in toc_calc_df.columns and "Row initial" in toc_calc_df.columns

        if is_new_format:
            # Format nou: 4-TOC_CALC amb Sample, TOC_Row, Inj_Index
            sample_key_r = f"{original_name}_R{inj_num_original}"
            mask = (toc_calc_df["Sample"] == sample_key_r)
            if not mask.any():
                mask = (toc_calc_df["Sample"].str.contains(original_name, case=False, na=False)) & \
                       (toc_calc_df["Inj_Index"] == line_num)

            sample_rows = toc_calc_df[mask]
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
            # Extreure DOC Direct
            df_doc, timeout_info = extract_doc_from_masterfile(
                toc_df, row_start, row_end, detect_timeouts=True
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
        if (normalize_key(file_sample) == normalize_key(original_name) or
            normalize_key(file_sample) == normalize_key(sample_name)):
            # Verificar replica
            try:
                file_rep_int = int(file_rep) if file_rep else 1
            except ValueError:
                file_rep_int = 1

            if file_rep_int == inj_num:
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

        if (normalize_key(file_sample) == normalize_key(original_name) or
            normalize_key(file_sample) == normalize_key(sample_name)):
            try:
                file_rep_int = int(file_rep) if file_rep else 1
            except ValueError:
                file_rep_int = 1

            if file_rep_int == inj_num:
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

    # 3. Si és KHP i no tenim DAD, buscar a MasterFile 3-DAD_KHP
    if sample_type == "KHP" and result["dad"] is None and master_khp_data is not None:
        # Buscar columna corresponent al KHP
        khp_key = f"{original_name}_R{inj_num}"

        for col in master_khp_data.columns:
            if normalize_key(col) == normalize_key(khp_key):
                # Trobar columna de temps (anterior)
                col_idx = master_khp_data.columns.get_loc(col)
                if col_idx > 0:
                    time_col = master_khp_data.columns[col_idx - 1]
                    # Construir DataFrame
                    df_khp = pd.DataFrame({
                        "time (min)": pd.to_numeric(master_khp_data[time_col], errors="coerce"),
                        "254nm": pd.to_numeric(master_khp_data[col], errors="coerce"),
                    }).dropna()

                    if not df_khp.empty:
                        result["dad"] = {
                            "path": "MasterFile:3-DAD_KHP",
                            "df": df_khp,
                        }
                        result["dad_source"] = "masterfile"
                        result["has_data"] = True
                break

    return result


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

        report_progress(20, "Parsejant injeccions del MasterFile...")

        # 3. Parsejar injeccions del MasterFile (FONT DE VERITAT)
        injections, parse_warnings = parse_injections_from_masterfile(result["master_data"], config)
        result["injections"] = injections
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
        try:
            xl = pd.ExcelFile(master_path)
            if "3-DAD_KHP" in xl.sheet_names:
                master_khp_data = pd.read_excel(master_path, sheet_name="3-DAD_KHP")
            if "2-TOC" in xl.sheet_names:
                toc_df = pd.read_excel(master_path, sheet_name="2-TOC", header=6)
            if "4-TOC_CALC" in xl.sheet_names:
                toc_calc_df = pd.read_excel(master_path, sheet_name="4-TOC_CALC")
        except Exception:
            pass

        report_progress(40, "Processant injeccions...")

        # 5. Processar cada injecció
        used_files = {"uib": set(), "dad": set()}
        total_inj = len(injections)

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
                toc_df=toc_df, toc_calc_df=toc_calc_df
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
                    mode_type = "BP" if result["method"] == "BP" else "COL"
                    use_end = (mode_type == "BP")
                    baseline_direct = get_baseline_correction(t_direct, y_direct, mode_type, use_end=use_end)
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
                    mode_type = "BP" if result["method"] == "BP" else "COL"
                    use_end = (mode_type == "BP")
                    baseline_uib = get_baseline_correction(t_uib, y_uib, mode_type, use_end=use_end)
                    y_uib_net = y_uib - baseline_uib

                uib_data["y_net"] = y_uib_net
                uib_data["baseline"] = baseline_uib

            # Guardar rèplica
            rep_key = str(inj_num)
            result["samples"][sample_name]["replicas"][rep_key] = {
                "direct": direct_data,  # DOC Direct del MasterFile
                "uib": uib_data,        # DOC UIB del CSV (si disponible)
                "dad": data.get("dad"),
                "dad_source": data.get("dad_source"),
                "has_data": data.get("has_data", False),
                "injection_info": {
                    "line_num": inj["line_num"],
                    "inj_num": inj_num,
                },
            }

        report_progress(90, "Verificant fitxers orfes...")

        # 6. Identificar fitxers orfes (no assignats a cap mostra)
        orphan_uib = [f for f in uib_files if f not in used_files["uib"]]
        orphan_dad = [f for f in dad_files if f not in used_files["dad"]]

        result["orphan_files"] = {
            "uib": orphan_uib,
            "dad": orphan_dad,
        }

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

        result["success"] = True
        report_progress(100, "Importació completada")

    except Exception as e:
        import traceback
        result["errors"].append(str(e))
        result["errors"].append(traceback.format_exc())

    return result


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
    }

    # Detall per mostra
    samples_detail = []
    for sample_name, sample_info in imported_data.get("samples", {}).items():
        sample_entry = {
            "name": sample_name,
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

            # DOC UIB
            uib = rep_data.get("uib") or {}
            if uib.get("t") is not None:
                t_arr = uib["t"]
                replica_entry["uib"] = {
                    "source": "CSV",
                    "file": os.path.basename(uib.get("path", "")),
                    "n_points": len(t_arr),
                    "t_min": float(min(t_arr)),
                    "t_max": float(max(t_arr)),
                    "baseline": uib.get("baseline"),
                }

            # DAD
            dad = rep_data.get("dad") or {}
            df = dad.get("df")
            if df is not None:
                t_col = df.columns[0]
                replica_entry["dad"] = {
                    "source": rep_data.get("dad_source", "unknown"),
                    "n_points": len(df),
                    "n_wavelengths": len(df.columns) - 1,  # -1 per columna temps
                    "t_min": float(df[t_col].min()),
                    "t_max": float(df[t_col].max()),
                    "wavelengths_range": f"{df.columns[1]}-{df.columns[-1]}",
                }

            # Info injecció original (si disponible)
            inj_info = rep_data.get("injection_info")
            if inj_info and include_injection_details:
                replica_entry["injection"] = {
                    "line_num": inj_info.get("line_num"),
                    "inj_num": inj_info.get("inj_num"),
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
        "total_controls": len([s for s in samples_detail if s["type"] == "CONTROL"]),
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
        output_path: Ruta de sortida (default: SEQ_PATH/import_manifest.json)

    Returns:
        Path del fitxer generat
    """
    manifest = generate_import_manifest(imported_data)

    if output_path is None:
        seq_path = imported_data.get("seq_path", ".")
        output_path = os.path.join(seq_path, "import_manifest.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return output_path


def load_manifest(seq_path):
    """
    Carrega el manifest JSON si existeix.

    Args:
        seq_path: Ruta a la carpeta SEQ

    Returns:
        dict amb manifest o None si no existeix
    """
    manifest_path = os.path.join(seq_path, "import_manifest.json")
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
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

    # Inicialitzar resultat
    seq_info = manifest.get("sequence", {})
    mf_info = manifest.get("master_file", {})

    result = {
        "success": False,
        "seq_path": seq_path,
        "seq_name": seq_info.get("name", os.path.basename(seq_path)),
        "method": seq_info.get("method", "COLUMN"),
        "data_mode": seq_info.get("data_mode", "DUAL"),
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
            if direct_info and toc_df is not None:
                row_start = direct_info.get("row_start")
                row_end = direct_info.get("row_end")
                if row_start is not None and row_end is not None:
                    try:
                        doc_result = extract_doc_from_masterfile(
                            toc_df, row_start, row_end,
                            t_start=None, detect_timeouts=True
                        )
                        if doc_result and doc_result.get("t") is not None:
                            # Calcular baseline
                            mode_type = "BP" if result["method"] == "BP" else "COL"
                            baseline = get_baseline_correction(
                                doc_result["t"], doc_result["y"], mode_type, config
                            )
                            y_net = np.array(doc_result["y"]) - baseline

                            rep_data["direct"] = {
                                "path": f"MasterFile:2-TOC",
                                "df": None,
                                "t": doc_result["t"],
                                "y": doc_result["y"],
                                "row_start": row_start,
                                "row_end": row_end,
                                "timeout_info": doc_result.get("timeout_info", {}),
                                "y_net": y_net,
                                "baseline": baseline,
                            }
                            rep_data["has_data"] = True
                    except Exception as e:
                        result["warnings"].append(f"{sample_name} rep {rep_num} Direct: {e}")

            # === DOC UIB ===
            uib_info = rep_info.get("uib")
            if uib_info:
                uib_file = uib_info.get("file", "")
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
                            uib_result = llegir_doc_uib(uib_path)
                            if uib_result and uib_result.get("t") is not None:
                                t_uib = uib_result["t"]
                                y_uib = uib_result["y"]

                                # Baseline
                                mode_type = "BP" if result["method"] == "BP" else "COL"
                                baseline = get_baseline_correction(t_uib, y_uib, mode_type, config)
                                y_net = np.array(y_uib) - baseline

                                rep_data["uib"] = {
                                    "path": uib_path,
                                    "df": uib_result.get("df"),
                                    "t": t_uib,
                                    "y": y_uib,
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

                # Buscar fitxer DAD
                if dad_source == "export3d":
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
                                    df_dad = llegir_dad_export3d(df_path)
                                    if df_dad is not None:
                                        rep_data["dad"] = {"df": df_dad, "path": df_path}
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

    result["success"] = True
    report_progress(100, "Importació des de manifest completada")

    return result


# =============================================================================
# FUNCIONS AUXILIARS PER COMPATIBILITAT
# =============================================================================

# Re-exportar funcions que altres mòduls poden necessitar
__all__ = [
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
    # Baseline
    "get_baseline_correction",
    "DEFAULT_BASELINE_CONFIG",
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
