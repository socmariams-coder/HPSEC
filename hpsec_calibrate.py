"""
hpsec_calibrate.py - Mòdul de calibració HPSEC

Funcions principals:
- calibrate_from_import(): Calibració des de dades importades (import_manifest.json)
- analizar_khp_data(): Anàlisi de dades KHP en memòria
- register_calibration(): Registre de calibracions amb suport múltiples condicions
- get_all_active_calibrations(): Obté totes les calibracions actives (una per condition_key)
- validate_khp_quality(): Validació de qualitat KHP

Suport múltiples condicions:
- Cada combinació (mode, volume, conc) genera un condition_key únic
- Una SEQ pot tenir N calibracions actives (una per condition_key)
- Ex: KHP2@100µL i KHP2@50µL poden coexistir

v1.5.1 - 2026-02-03: Eliminat calculate_peak_symmetry/snr locals, usar funcions de core
v1.5.0 - 2026-02-03: Suport múltiples calibracions per SEQ (condition_key)
v1.4.0 - 2026-01-30: Eliminat codi obsolet (fitxers consolidats Excel)
v1.3.0 - 2026-01-29: Migrades funcions alineació des de hpsec_consolidate.py
v1.1.0 - 2026-01-26: Refactor - funcions detecció mogudes a hpsec_core.py
v1.0.0 - Versió inicial
"""

__version__ = "1.5.1"
__version_date__ = "2026-02-03"

import os
import re
import glob
import json
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.signal import savgol_filter
from scipy.integrate import trapezoid
from hpsec_config import get_registry_path

# Import funcions de detecció des de hpsec_core (Single Source of Truth)
from hpsec_core import (
    detect_timeout,
    detect_batman,
    detect_peak_anomaly,
    detect_main_peak,
    detect_all_peaks,
    integrate_chromatogram,
    integrate_above_baseline,
    calculate_fwhm,
    calculate_symmetry,
    TIMEOUT_CONFIG
)

# Import funcions d'identificació des de hpsec_import (Single Source of Truth)
from hpsec_import import is_khp, extract_khp_conc, obtenir_seq

# Import funcions utilitàries
# NOTA (2026-02-03): Baseline functions ara a hpsec_core.py
from hpsec_core import mode_robust, get_baseline_value, get_baseline_stats
from hpsec_utils import t_at_max

# =============================================================================
# JSON ENCODER PER NUMPY TYPES
# =============================================================================

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


# =============================================================================
# CONSTANTS
# =============================================================================

# Fitxers d'historial GLOBAL (a REGISTRY/)
REGISTRY_FOLDER = "REGISTRY"
KHP_HISTORY_FILENAME = "KHP_History.json"
KHP_HISTORY_EXCEL_FILENAME = "KHP_History.xlsx"
SAMPLES_HISTORY_FILENAME = "Samples_History.json"

# Fitxers locals per SEQ (a CHECK/data/)
LOCAL_DATA_FOLDER = "data"  # Subcarpeta dins CHECK
CALIBRATION_FILENAME = "calibration_result.json"


# Configuració per defecte
DEFAULT_CONFIG = {
    # Consolidació
    "bp_baseline_win": 1.0,
    "col_baseline_start": 10.0,

    # Calibració
    "khp_pattern": "KHP",
    "peak_min_prominence_pct": 5.0,
    "alignment_threshold_sec": 4.0,

    # Processament
    "timeout_min_height_frac": 0.30,
    "batman_max_sep_min": 0.5,
    "batman_min_height_pct": 15.0,
    "batman_min_sigma": 3.0,
}

# Volums d'injecció (µL)
INJECTION_VOLUME_BP = 100
INJECTION_VOLUME_COLUMN = 400
INJECTION_VOLUME_COLUMN_OLD = 100  # SEQ 256-274

# =============================================================================
# THRESHOLDS CONCENTRATION RATIO (CR) - Basats en anàlisi de 26 calibracions
# =============================================================================
# CR = Àrea pic principal / Àrea total cromatograma (sobre baseline)
# Mesura quina fracció del senyal està al pic KHP (compost pur = ~100%)
#
# BP Mode: CR no és útil (SNR ~1.5, senyal sota threshold → CR artificial 100%)
# Column Mode: CR varia segons volum d'injecció
#   - 400µL (protocol actual): CR = 75.2% ± 3.9%, rang 70.6-81.9%
#   - 100µL (protocol antic):  CR = 69.9% ± 20.0%, alta variabilitat
#
CR_THRESHOLDS = {
    'BP': {
        # BP: No usar CR per validació (SNR massa baix)
        'skip_validation': True,
        'min_snr_for_cr': 5.0,  # Només validar CR si SNR > 5
    },
    'COLUMN_400': {
        # Protocol actual (SEQ >= 275): molt consistent
        'fail': 0.65,      # CR < 65% → FAIL
        'warning': 0.70,   # CR 65-70% → WARNING
        'ok': 0.70,        # CR >= 70% → OK
    },
    'COLUMN_100': {
        # Protocol antic (SEQ 256-274): més variable
        'fail': 0.45,      # CR < 45% → FAIL
        'warning': 0.55,   # CR 45-55% → WARNING
        'ok': 0.55,        # CR >= 55% → OK
    },
}


# =============================================================================
# FUNCIONS AUXILIARS
# =============================================================================

def get_injection_volume(seq_path, is_bp):
    """
    Retorna el volum d'injecció en µL segons el mode i la seqüència.

    - BP: sempre 100 µL
    - COLUMN SEQ 256-274: 100 µL (protocol antic)
    - COLUMN SEQ >= 275: 400 µL (protocol actual)
    """
    if is_bp:
        return INJECTION_VOLUME_BP

    # Extreure número de seqüència
    seq_num = extract_seq_number(seq_path)

    if seq_num and 256 <= seq_num <= 274:
        return INJECTION_VOLUME_COLUMN_OLD

    return INJECTION_VOLUME_COLUMN


def extract_seq_number(seq_path):
    """Extreu el número de seqüència del path."""
    if not seq_path:
        return None
    folder_name = os.path.basename(os.path.normpath(seq_path))
    match = re.search(r'^(\d+)', folder_name)
    if match:
        return int(match.group(1))
    return None


# NOTA: obtenir_seq s'ha mogut a hpsec_import.py (2026-01-29)


def get_condition_key(mode: str, volume_uL: int, conc_ppm: float = None) -> str:
    """
    Genera clau única per identificar condicions de calibració.

    Permet tenir múltiples calibracions actives per SEQ amb diferents condicions
    (ex: KHP2@100µL i KHP2@50µL).

    Args:
        mode: "COLUMN" o "BP"
        volume_uL: Volum d'injecció en µL
        conc_ppm: Concentració KHP en ppm (opcional)

    Returns:
        Clau única format: "{mode}_{volume}_{conc}"
        Ex: "COLUMN_400_5", "BP_100_2", "BP_50_2"
    """
    vol = int(volume_uL) if volume_uL else 0
    conc = int(conc_ppm) if conc_ppm else 0
    return f"{mode}_{vol}_{conc}"


def get_cr_thresholds(is_bp, volume_uL):
    """
    Retorna els thresholds de CR segons mode i volum.

    Args:
        is_bp: True si és mode BP
        volume_uL: Volum d'injecció en µL

    Returns:
        Dict amb:
            - fail: threshold per FAIL
            - warning: threshold per WARNING
            - ok: threshold per OK
            - protocol: descripció del protocol
            - skip: True si no cal validar CR
            - explanation: explicació dels thresholds
    """
    if is_bp:
        return {
            'fail': None,
            'warning': None,
            'ok': None,
            'protocol': 'BP',
            'skip': True,
            'explanation': (
                "Mode BP: CR no és fiable perquè el senyal és molt baix (SNR ~1.5). "
                "L'àrea total queda sota el threshold de baseline, resultant en CR=100% artificial."
            )
        }
    elif volume_uL >= 400:
        cfg = CR_THRESHOLDS['COLUMN_400']
        return {
            'fail': cfg['fail'],
            'warning': cfg['warning'],
            'ok': cfg['ok'],
            'protocol': 'Column 400µL',
            'skip': False,
            'explanation': (
                f"Protocol actual (400µL): CR molt consistent (75.2% ± 3.9%). "
                f"FAIL si CR < {cfg['fail']:.0%}, WARNING si < {cfg['warning']:.0%}."
            )
        }
    else:
        cfg = CR_THRESHOLDS['COLUMN_100']
        return {
            'fail': cfg['fail'],
            'warning': cfg['warning'],
            'ok': cfg['ok'],
            'protocol': 'Column 100µL',
            'skip': False,
            'explanation': (
                f"Protocol antic (100µL): CR més variable (69.9% ± 20.0%). "
                f"FAIL si CR < {cfg['fail']:.0%}, WARNING si < {cfg['warning']:.0%}."
            )
        }


# NOTA: is_khp, extract_khp_conc s'han mogut a hpsec_import.py (2026-01-29)
# NOTA: mode_robust, t_at_max s'han mogut a hpsec_utils.py (2026-01-29)


# =============================================================================
# NOTA: Funcions de baseline/stats ara estan a hpsec_utils.py (Single Source of Truth)
# Usar get_baseline_stats() i get_baseline_value() directament

# NOTA: detect_main_peak i detect_all_peaks ara estan a hpsec_core.py
# (Single Source of Truth per evitar duplicació)


def timeout_affects_peak(timeout_info, t_doc, left_idx, right_idx):
    """
    Verifica si algun timeout afecta l'interval d'integració del pic principal.

    Per KHP, només interessa si el timeout cau dins l'interval del pic,
    NO la nomenclatura de zones (HS, BioP, etc.) que és per mostres.

    Args:
        timeout_info: Dict retornat per detect_timeout()
        t_doc: Array de temps DOC
        left_idx, right_idx: Índexs d'integració del pic

    Returns:
        dict amb:
            - affects_peak: bool - si algun timeout afecta el pic
            - overlap_pct: float - percentatge del pic afectat
            - affected_timeouts: list - timeouts que afecten el pic
    """
    if not timeout_info or not timeout_info.get('timeouts'):
        return {'affects_peak': False, 'overlap_pct': 0, 'affected_timeouts': []}

    # Límits temporals del pic
    t_peak_start = t_doc[left_idx]
    t_peak_end = t_doc[right_idx]
    peak_duration = t_peak_end - t_peak_start

    affected_timeouts = []
    total_overlap = 0

    for to in timeout_info['timeouts']:
        # Zona afectada pel timeout (inclou PRE i POST)
        to_start = to.get('affected_start_min', to['t_start_min'] - 0.5)
        to_end = to.get('affected_end_min', to['t_end_min'] + 1.0)

        # Calcular overlap amb el pic
        overlap_start = max(t_peak_start, to_start)
        overlap_end = min(t_peak_end, to_end)
        overlap = max(0, overlap_end - overlap_start)

        if overlap > 0:
            affected_timeouts.append({
                't_start': to['t_start_min'],
                't_end': to['t_end_min'],
                'overlap_min': overlap,
                'is_major': to.get('is_major', False)
            })
            total_overlap += overlap

    overlap_pct = (total_overlap / peak_duration * 100) if peak_duration > 0 else 0

    return {
        'affects_peak': len(affected_timeouts) > 0,
        'overlap_pct': overlap_pct,
        'affected_timeouts': affected_timeouts
    }


# =============================================================================
# QUALITAT DEL PIC
# =============================================================================
# NOTA (2026-02-03): calculate_peak_symmetry() i calculate_peak_snr() eliminades.
# Usar calculate_symmetry() de hpsec_core.py (50% d'altura, estàndard cromatogràfic).
# SNR es calcula inline: (peak_height - baseline_mean) / baseline_std


def calculate_integration_limits(t, y, peak_idx, min_width_min=1.0, max_width_min=6.0):
    """
    Calcula els límits d'integració d'un pic KHP usant mètode simplificat.

    LÒGICA SIMPLIFICADA:
    1. Baseline = MODA de tot el senyal (valor més freqüent)
    2. Threshold = baseline + 3*sigma (estadístic)
    3. Límits = des del pic fins que senyal ≤ threshold

    Args:
        t: Array de temps (minuts)
        y: Array de senyal
        peak_idx: Índex del màxim del pic
        min_width_min: Amplada mínima en minuts (default 1.0)
        max_width_min: Amplada màxima en minuts (default 6.0)

    Returns:
        Dict amb left_idx, right_idx, baseline, threshold, width_minutes, etc.
    """
    try:
        t = np.asarray(t, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(y)

        if n < 10:
            return {
                "left_idx": 0, "right_idx": n - 1,
                "baseline": 0, "threshold": 0,
                "width_minutes": float(t[-1] - t[0]) if n > 1 else 0,
                "valid": False, "message": "Senyal massa curt"
            }

        # === 1. BASELINE = MODA de tot el senyal ===
        baseline = mode_robust(y)

        # === 2. STD dels punts propers a baseline ===
        y_range = float(np.max(y) - baseline)
        if y_range <= 0:
            return {
                "left_idx": 0, "right_idx": n - 1,
                "baseline": baseline, "threshold": baseline,
                "width_minutes": float(t[-1] - t[0]),
                "valid": False, "message": "Senyal sense pic"
            }

        mask_baseline = y < (baseline + 0.2 * y_range)
        if np.sum(mask_baseline) > 5:
            std_baseline = float(np.std(y[mask_baseline]))
        else:
            std_baseline = float(np.std(y)) * 0.1

        if std_baseline < 1e-6:
            std_baseline = 0.01 * y_range

        # === 3. THRESHOLD = baseline + 3*sigma ===
        threshold = baseline + 3 * std_baseline

        # Calcular límits en índexs basats en temps
        dt = np.mean(np.diff(t)) if n > 1 else 0.01
        max_width_idx = int(max_width_min / dt) if dt > 0 else 300
        min_width_idx = int(min_width_min / dt) if dt > 0 else 50

        # === 4. BUSCAR LÍMIT ESQUERRE ===
        left_idx = peak_idx
        for i in range(peak_idx - 1, max(0, peak_idx - max_width_idx), -1):
            if y[i] <= threshold:
                left_idx = i
                break
            left_idx = i
        else:
            left_idx = max(0, peak_idx - max_width_idx)

        # === 5. BUSCAR LÍMIT DRET ===
        right_idx = peak_idx
        for i in range(peak_idx + 1, min(n, peak_idx + max_width_idx)):
            if y[i] <= threshold:
                right_idx = i
                break
            right_idx = i
        else:
            right_idx = min(n - 1, peak_idx + max_width_idx)

        # === 6. VALIDAR AMPLADA MÍNIMA ===
        current_width_idx = right_idx - left_idx
        if current_width_idx < min_width_idx:
            expand_needed = (min_width_idx - current_width_idx) // 2 + 1
            left_idx = max(0, left_idx - expand_needed)
            right_idx = min(n - 1, right_idx + expand_needed)

        # Assegurar que el pic està dins dels límits
        left_idx = int(min(left_idx, peak_idx - 3))
        right_idx = int(max(right_idx, peak_idx + 3))
        left_idx = max(0, left_idx)
        right_idx = min(n - 1, right_idx)

        # Verificar si els límits arriben a baseline
        left_at_baseline = y[left_idx] <= threshold
        right_at_baseline = y[right_idx] <= threshold
        width_minutes = float(t[right_idx] - t[left_idx])

        return {
            "left_idx": left_idx,
            "right_idx": right_idx,
            "baseline": baseline,
            "std_baseline": std_baseline,
            "threshold": threshold,
            "width_minutes": width_minutes,
            "left_at_baseline": left_at_baseline,
            "right_at_baseline": right_at_baseline,
            "valid": left_at_baseline and right_at_baseline,
            "message": "OK" if (left_at_baseline and right_at_baseline) else "Límits no arriben a baseline"
        }

    except Exception as e:
        return {
            "left_idx": 0, "right_idx": len(y) - 1 if len(y) > 0 else 0,
            "baseline": 0, "threshold": 0,
            "width_minutes": 0, "valid": False,
            "message": f"Error: {e}"
        }


def expand_integration_limits_to_baseline(t, y, left_idx, right_idx, peak_idx,
                                          baseline_threshold_pct=15,
                                          min_width_minutes=1.0,
                                          max_width_minutes=6.0,
                                          is_bp=False):
    """
    Wrapper per compatibilitat - crida calculate_integration_limits.

    Manté la signatura antiga per codi que l'usa.
    """
    result = calculate_integration_limits(t, y, peak_idx, min_width_minutes, max_width_minutes)

    # Adaptar format de retorn per compatibilitat
    expanded_left = max(0, left_idx - result["left_idx"])
    expanded_right = max(0, result["right_idx"] - right_idx)

    return {
        "left_idx": result["left_idx"],
        "right_idx": result["right_idx"],
        "expanded_left": expanded_left,
        "expanded_right": expanded_right,
        "baseline": result["baseline"],
        "threshold_value": result["threshold"],
        "original_valid": (expanded_left == 0 and expanded_right == 0),
        "left_at_baseline": result.get("left_at_baseline", True),
        "right_at_baseline": result.get("right_at_baseline", True),
        "width_minutes": result["width_minutes"],
    }


def validate_integration_baseline(t, y, left_idx, right_idx, peak_idx, baseline_threshold_pct=15):
    """
    Valida que els límits d'integració arribin a valors propers a la línia base.

    Args:
        t: Array de temps
        y: Array de senyal
        left_idx: Índex límit esquerre
        right_idx: Índex límit dret
        peak_idx: Índex del pic
        baseline_threshold_pct: Percentatge màxim permès respecte l'altura del pic

    Returns:
        Dict amb valid, message, i detalls dels límits
    """
    try:
        y = np.asarray(y)
        peak_height = y[peak_idx]

        # Calcular baseline
        search_range = max(50, (right_idx - left_idx))
        local_region = y[max(0, left_idx - search_range):min(len(y), right_idx + search_range)]
        baseline = np.percentile(local_region, 5)

        effective_height = peak_height - baseline
        if effective_height <= 0:
            return {"valid": True, "message": "Pic no detectat correctament"}

        # Valors als límits
        left_value = y[left_idx] - baseline
        right_value = y[right_idx] - baseline

        left_pct = (left_value / effective_height) * 100 if effective_height > 0 else 0
        right_pct = (right_value / effective_height) * 100 if effective_height > 0 else 0

        left_at_baseline = left_pct <= baseline_threshold_pct
        right_at_baseline = right_pct <= baseline_threshold_pct
        valid = left_at_baseline and right_at_baseline

        if valid:
            message = "OK"
        else:
            issues = []
            if not left_at_baseline:
                issues.append(f"límit esquerre alt ({left_pct:.0f}%)")
            if not right_at_baseline:
                issues.append(f"límit dret alt ({right_pct:.0f}%)")
            message = "Límits integració estrets: " + ", ".join(issues)

        return {
            "valid": valid,
            "left_at_baseline": left_at_baseline,
            "right_at_baseline": right_at_baseline,
            "left_value_pct": left_pct,
            "right_value_pct": right_pct,
            "message": message,
        }

    except Exception as e:
        return {"valid": True, "message": f"Error validació: {e}"}


# NOTA: detect_batman i detect_timeout ara estan a hpsec_core.py
# (Single Source of Truth per evitar duplicació)
# Usar: from hpsec_core import detect_batman, detect_timeout, detect_peak_anomaly


# =============================================================================
# HELPER: QUANTIFICACIÓ AMB CALIBRACIONS
# =============================================================================

def get_calibration_for_conditions(calibration_data, volume_uL, signal="direct"):
    """
    Troba la calibració que coincideix amb les condicions donades.

    Args:
        calibration_data: Resultat de calibrate_sequence()
        volume_uL: Volum d'injecció de la mostra (µL)
        signal: "direct", "uib", o "primary"

    Returns:
        dict: Calibració amb les mateixes condicions, o la principal si no hi ha match
    """
    if not calibration_data:
        return None

    # Seleccionar llista de calibracions segons senyal
    if signal == "uib":
        calibrations = calibration_data.get("calibrations_uib", [])
    elif signal == "direct":
        calibrations = calibration_data.get("calibrations_direct", [])
    else:
        calibrations = calibration_data.get("calibrations", [])

    if not calibrations:
        # Fallback: retornar khp_data si existeix (compatibilitat)
        return calibration_data.get("khp_data")

    # Buscar calibració pel volum
    vol_int = int(volume_uL) if volume_uL else 0
    for cal in calibrations:
        if cal.get('volume_uL') == vol_int:
            return cal

    # Fallback: primera calibració (la de major conc/vol)
    return calibrations[0]


def quantify_sample(area, volume_uL, calibration_data, signal="direct"):
    """
    Calcula la concentració d'una mostra usant la calibració de les seves condicions.

    Args:
        area: Àrea integrada del cromatograma (mAU·min)
        volume_uL: Volum d'injecció de la mostra (µL)
        calibration_data: Resultat de calibrate_sequence()
        signal: "direct", "uib", o "primary"

    Returns:
        float: Concentració en ppm (mg/L)

    Fórmula:
        conc_ppm = Area × 1000 / (RF_mass × volume_µL)

    Example:
        >>> cal = calibrate_sequence(seq_path, samples)
        >>> for sample in samples:
        ...     area = sample.get("area", 0)
        ...     vol = sample.get("inj_volume", 400)
        ...     conc = quantify_sample(area, vol, cal, signal="direct")
    """
    cal = get_calibration_for_conditions(calibration_data, volume_uL, signal)
    if not cal:
        return 0.0

    rf_mass = cal.get('rf_mass', 0)
    if rf_mass <= 0 or volume_uL <= 0:
        return 0.0
    return area * 1000 / (rf_mass * volume_uL)


def get_all_calibrations(calibration_data, signal="direct"):
    """
    Retorna totes les calibracions disponibles per un senyal.

    Cada calibració correspon a una combinació de condicions (nom, volum, conc).

    Args:
        calibration_data: Resultat de calibrate_sequence()
        signal: "direct", "uib", o "primary"

    Returns:
        list of dict: Llista de calibracions amb estructura idèntica
    """
    if not calibration_data:
        return []

    # Seleccionar llista segons senyal
    if signal == "uib":
        calibrations = calibration_data.get("calibrations_uib", [])
    elif signal == "direct":
        calibrations = calibration_data.get("calibrations_direct", [])
    else:
        calibrations = calibration_data.get("calibrations", [])

    # Si no hi ha llista, intentar compatibilitat amb format antic
    if not calibrations:
        khp_data = calibration_data.get("khp_data", {})
        if khp_data:
            return [khp_data]
        return []

    return calibrations


# =============================================================================
# COMPARACIÓ HISTÒRICA KHP
# =============================================================================

def get_historical_khp_stats(seq_path, mode="COLUMN", conc_ppm=None, volume_uL=None,
                             doc_mode=None, uib_sensitivity=None,
                             n_recent=10, exclude_outliers=True):
    """
    Obté estadístiques de les calibracions KHP històriques.

    Filtra per mode, concentració, volum, doc_mode i sensibilitat UIB
    per comparar "pomes amb pomes".

    Args:
        seq_path: Path de la SEQ actual (per trobar KHP_History.json)
        mode: "COLUMN" o "BP"
        conc_ppm: Concentració KHP en ppm (ex: 2 per KHP2). REQUERIT per comparació vàlida.
        volume_uL: Volum d'injecció en µL (ex: 100, 400). REQUERIT per comparació vàlida.
        doc_mode: "Direct", "UIB" o "DUAL". Si None, no filtra.
        uib_sensitivity: 700 o 1000 (ppb). Només aplica si doc_mode conté UIB.
        n_recent: Nombre de calibracions recents a considerar
        exclude_outliers: Excloure calibracions marcades com outlier

    Returns:
        Dict amb estadístiques o None si no hi ha prou dades:
        {
            'mean_area': float,
            'std_area': float,
            'mean_concentration_ratio': float,
            'n_calibrations': int,
            'conc_ppm': float,
            'volume_uL': float,
            'doc_mode': str,
            'uib_sensitivity': int or None,
            'calibrations': list  # Les calibracions usades
        }
    """
    history = load_khp_history(seq_path)
    if not history:
        return None

    # Filtrar per mode, concentració, volum, doc_mode i sensibilitat
    valid_cals = []
    for cal in history:
        if cal.get('mode') != mode:
            continue
        if exclude_outliers and cal.get('is_outlier', False):
            continue
        if cal.get('area', 0) <= 0:
            continue
        # Filtrar per concentració si s'especifica
        if conc_ppm is not None:
            cal_conc = cal.get('conc_ppm', 0)
            if abs(cal_conc - conc_ppm) > 0.5:  # Tolerància de 0.5 ppm
                continue
        # Filtrar per volum si s'especifica
        if volume_uL is not None:
            cal_vol = cal.get('volume_uL', 0)
            if cal_vol != volume_uL:  # Volum ha de ser exacte
                continue
        # Filtrar per doc_mode si s'especifica
        if doc_mode is not None:
            cal_doc_mode = cal.get('doc_mode', 'N/A')
            # N/A és comodí (calibracions antigues) - acceptar sempre
            if cal_doc_mode != 'N/A':
                # Si doc_mode actual és DUAL, acceptar DUAL o el mateix senyal
                # Si doc_mode actual és Direct/UIB, només acceptar exacte o DUAL
                if doc_mode == "DUAL":
                    if cal_doc_mode not in ["DUAL", "Direct", "UIB"]:
                        continue
                else:
                    if cal_doc_mode != doc_mode and cal_doc_mode != "DUAL":
                        continue
        # Filtrar per sensibilitat UIB si s'especifica i és UIB
        if uib_sensitivity is not None and doc_mode in ["UIB", "DUAL"]:
            cal_sensitivity = cal.get('uib_sensitivity')
            if cal_sensitivity is not None and cal_sensitivity != uib_sensitivity:
                continue
        valid_cals.append(cal)

    if len(valid_cals) < 3:
        return None

    # Ordenar per data (més recent primer) i agafar n_recent
    valid_cals.sort(key=lambda x: x.get('date', ''), reverse=True)
    recent_cals = valid_cals[:n_recent]

    # Filtrar outliers estadístics (IQR) si tenim prou dades
    if exclude_outliers and len(recent_cals) >= 5:
        areas_raw = np.array([cal['area'] for cal in recent_cals])
        q1 = np.percentile(areas_raw, 25)
        q3 = np.percentile(areas_raw, 75)
        iqr = q3 - q1
        lower_bound = q1 - 3.0 * iqr  # Usar 3x IQR per ser menys agressiu
        upper_bound = q3 + 3.0 * iqr
        recent_cals = [cal for cal in recent_cals if lower_bound <= cal['area'] <= upper_bound]

    # Calcular estadístiques
    areas = [cal['area'] for cal in recent_cals]
    concentration_ratios = [cal.get('concentration_ratio', 1.0) for cal in recent_cals]

    # Només guardar resum de calibracions (no objectes complets per evitar bloat)
    calibrations_summary = [
        {'seq_name': cal.get('seq_name'), 'area': cal.get('area'), 'date': cal.get('date', '')[:10]}
        for cal in recent_cals
    ]

    return {
        'mean_area': float(np.mean(areas)),
        'std_area': float(np.std(areas)),
        'cv_area_pct': float(np.std(areas) / np.mean(areas) * 100) if np.mean(areas) > 0 else 0,
        'mean_concentration_ratio': float(np.mean(concentration_ratios)),
        'min_area': float(np.min(areas)),
        'max_area': float(np.max(areas)),
        'n_calibrations': len(recent_cals),
        'conc_ppm': conc_ppm,
        'volume_uL': volume_uL,
        'calibrations_summary': calibrations_summary  # Només resum, no objectes complets
    }


def compare_khp_historical(current_area, current_concentration_ratio, seq_path, mode="COLUMN",
                          conc_ppm=None, volume_uL=None, doc_mode=None, uib_sensitivity=None,
                          exclude_outliers=False):
    """
    Compara el KHP actual amb l'històric.

    IMPORTANT: Filtra per concentració, volum i doc_mode per comparar correctament.
    No es pot comparar KHP2 amb KHP5, ni 100µL amb 400µL, ni Direct amb UIB.

    Args:
        current_area: Àrea del pic principal del KHP actual
        current_concentration_ratio: Ratio àrea_pic_principal / àrea_total
        seq_path: Path de la SEQ
        mode: "COLUMN" o "BP"
        conc_ppm: Concentració KHP (ex: 2 per KHP2)
        volume_uL: Volum d'injecció en µL
        doc_mode: "Direct", "UIB" o "DUAL"
        uib_sensitivity: 700 o 1000 (ppb) - només si UIB
        exclude_outliers: Si True, exclou calibracions marcades com outlier

    Returns:
        Dict amb resultat de la comparació:
        {
            'status': 'OK' | 'WARNING' | 'INVALID' | 'INSUFFICIENT_DATA',
            'area_deviation_pct': float,
            'concentration_ratio_deviation_pct': float,
            'historical_stats': dict,
            'issues': list,
            'warnings': list
        }
    """
    stats = get_historical_khp_stats(seq_path, mode, conc_ppm=conc_ppm, volume_uL=volume_uL,
                                     doc_mode=doc_mode, uib_sensitivity=uib_sensitivity,
                                     exclude_outliers=exclude_outliers)

    result = {
        'status': 'OK',
        'area_deviation_pct': 0,
        'concentration_ratio_deviation_pct': 0,
        'historical_stats': stats,
        'issues': [],
        'warnings': []
    }

    if stats is None or stats['n_calibrations'] < 3:
        result['status'] = 'INSUFFICIENT_DATA'
        result['warnings'].append(f"Històric insuficient (<3 calibracions {mode})")
        return result

    # Comparar àrea
    # BP mode: senyal molt variable (SNR ~1.5), usar thresholds més permissius
    mean_area = stats['mean_area']
    is_bp = (mode == "BP")

    if current_area <= 0 and is_bp:
        # BP amb àrea=0 és esperat (senyal baix)
        result['warnings'].append("BP: àrea_main_peak=0 (senyal sota threshold)")
    elif mean_area > 0:
        area_deviation_pct = abs(current_area - mean_area) / mean_area * 100
        result['area_deviation_pct'] = area_deviation_pct

        if is_bp:
            # BP: thresholds més permissius (senyal molt variable)
            # Només FAIL si àrea desvia >100%, WARNING >50%
            if area_deviation_pct > 100:
                result['status'] = 'INVALID'
                result['issues'].append(f"BP: Desviació àrea {area_deviation_pct:.1f}% (>100%)")
            elif area_deviation_pct > 50:
                if result['status'] == 'OK':
                    result['status'] = 'WARNING'
                result['warnings'].append(f"BP: Desviació àrea {area_deviation_pct:.1f}% (>50%)")
        else:
            # Column: thresholds estrictes
            if area_deviation_pct > 20:
                result['status'] = 'INVALID'
                result['issues'].append(f"Desviació àrea {area_deviation_pct:.1f}% vs històric (>20%)")
            elif area_deviation_pct > 10:
                if result['status'] == 'OK':
                    result['status'] = 'WARNING'
                result['warnings'].append(f"Desviació àrea {area_deviation_pct:.1f}% vs històric (>10%)")

    # Comparar concentration_ratio - Usar thresholds segons mode/volum
    mean_cr = stats['mean_concentration_ratio']
    is_bp = (mode == "BP")
    vol = volume_uL if volume_uL else 400
    cr_config = get_cr_thresholds(is_bp, vol)

    if cr_config.get('skip', False):
        # BP mode: no validar CR
        pass
    elif mean_cr > 0 and current_concentration_ratio > 0:
        cr_deviation_pct = abs(current_concentration_ratio - mean_cr) / mean_cr * 100
        result['concentration_ratio_deviation_pct'] = cr_deviation_pct

        fail_threshold = cr_config['fail']
        warn_threshold = cr_config['warning']

        # Concentration ratio baix és més problemàtic que alt
        if current_concentration_ratio < fail_threshold:
            result['status'] = 'INVALID'
            result['issues'].append(
                f"CR baix: {current_concentration_ratio:.1%} < {fail_threshold:.0%} "
                f"({cr_config['protocol']})"
            )
        elif current_concentration_ratio < warn_threshold:
            if result['status'] == 'OK':
                result['status'] = 'WARNING'
            result['warnings'].append(
                f"CR moderat: {current_concentration_ratio:.1%} < {warn_threshold:.0%} "
                f"({cr_config['protocol']})"
            )

    return result


# =============================================================================
# VALIDACIÓ QUALITAT KHP
# =============================================================================

def validate_khp_quality(khp_data, all_peaks, timeout_info, anomaly_info=None, seq_path=None):
    """
    Validació específica per KHP (no aplicable a mostres normals).

    Criteris:
    1. No múltiples pics significatius (>2 pics amb prominència >10%)
    2. No timeout en zona del pic
    3. No batman/irregular
    4. Simetria acceptable (0.5-2.0)
    5. SNR > 50 (KHP ha de ser senyal fort)
    6. Límits no expandits excessivament
    7. Concentration ratio >= 0.90 (pic principal / total)
    8. Comparació històrica (desviació àrea < 20%)

    Args:
        khp_data: Dict amb dades del KHP analitzat
        all_peaks: Llista de pics detectats
        timeout_info: Dict retornat per detect_timeout()
        anomaly_info: Dict retornat per detect_peak_anomaly() (opcional)
        seq_path: Path de la SEQ per comparació històrica (opcional)

    Returns:
        Dict amb:
            - is_valid: bool (True si KHP és vàlid per calibració)
            - issues: llista d'issues detectats
            - warnings: llista de warnings (no invalidants)
            - quality_score: puntuació (0 = perfecte, >100 = invàlid)
            - historical_comparison: dict amb comparació històrica (si disponible)
    """
    issues = []
    warnings = []
    quality_score = 0

    # 1. Multi-pic
    if all_peaks and len(all_peaks) > 2:
        significant_peaks = [p for p in all_peaks if p.get('prominence', 0) > khp_data.get('height', 1) * 0.1]
        if len(significant_peaks) > 2:
            issues.append(f"MULTI_PEAK: {len(significant_peaks)} pics significatius detectats")
            quality_score += 30 * (len(significant_peaks) - 2)

    # 2. Timeout
    if timeout_info:
        severity = timeout_info.get('severity', 'OK')
        if severity == 'CRITICAL':
            issues.append(f"TIMEOUT_CRITICAL: timeout en zona HS")
            quality_score += 150
        elif severity == 'WARNING':
            zones = timeout_info.get('zone_summary', {})
            affected = [z for z in zones.keys() if zones[z] > 0]
            warnings.append(f"TIMEOUT_WARNING: timeout en {', '.join(affected)}")
            quality_score += 50
        elif severity == 'INFO':
            warnings.append(f"TIMEOUT_INFO: timeout en zona segura")
            quality_score += 10

    # 3. Anomalia de forma (batman/irregular)
    if anomaly_info:
        if anomaly_info.get('is_batman', False):
            issues.append("BATMAN: detectat artefacte batman al pic")
            quality_score += 50
        if anomaly_info.get('is_irregular', False):
            smoothness = anomaly_info.get('smoothness', 100)
            warnings.append(f"IRREGULAR: smoothness baixa ({smoothness:.0f}%)")
            quality_score += 30

    # 4. Simetria
    symmetry = khp_data.get('symmetry', 1.0)
    if symmetry < 0.5 or symmetry > 2.0:
        issues.append(f"ASYMMETRY: simetria fora de rang ({symmetry:.2f})")
        quality_score += 20
    elif symmetry < 0.7 or symmetry > 1.5:
        warnings.append(f"ASYMMETRY_WARN: simetria límit ({symmetry:.2f})")
        quality_score += 5

    # 5. SNR - Thresholds depenen del mode
    # BP: SNR ~1.5 és normal (senyal baix), no ha de ser issue
    # Column: SNR > 50 esperat
    snr = khp_data.get('snr', 0)
    is_bp_mode = khp_data.get('is_bp', False)

    # Només validar SNR si realment s'ha calculat (>0)
    # Calibracions antigues poden tenir SNR=0 (no calculat)
    if snr > 0:
        if is_bp_mode:
            # BP mode: SNR baix és esperat, només warning si molt baix
            if snr < 1.0:
                issues.append(f"LOW_SNR_BP: SNR extremadament baix ({snr:.2f})")
                quality_score += 40
            elif snr < 1.2:
                warnings.append(f"SNR_BP_WARN: SNR molt baix ({snr:.2f})")
                quality_score += 10
            # SNR > 1.2 és normal per BP
        else:
            # Column mode: SNR ha de ser alt
            if snr < 20:
                issues.append(f"LOW_SNR: SNR massa baix ({snr:.1f})")
                quality_score += 40
            elif snr < 50:
                warnings.append(f"SNR_WARN: SNR moderat ({snr:.1f})")
                quality_score += 10
    # Si SNR=0 (no calculat), no penalitzar

    # 6. Límits expandits excessivament
    if khp_data.get('limits_expanded', False):
        expansion_info = khp_data.get('expansion_info', {})
        area_inc = expansion_info.get('area_increase_pct', 0)
        if area_inc > 30:
            issues.append(f"EXPANSION: límits expandits excessivament (+{area_inc:.0f}%)")
            quality_score += 25
        elif area_inc > 15:
            warnings.append(f"EXPANSION_WARN: límits expandits (+{area_inc:.0f}%)")
            quality_score += 10

    # 7. Concentration ratio (pic principal vs total)
    # Thresholds depenen del mode i volum d'injecció
    concentration_ratio = khp_data.get('concentration_ratio', 1.0)
    is_bp = khp_data.get('is_bp', False)
    volume_uL = khp_data.get('volume_uL', 400)

    if is_bp:
        # BP Mode: CR no és fiable (SNR massa baix)
        cr_config = CR_THRESHOLDS['BP']
        if cr_config.get('skip_validation', False) and snr < cr_config.get('min_snr_for_cr', 5.0):
            # No validar CR per BP amb SNR baix
            pass
        else:
            # Si SNR és acceptable, validar amb thresholds estàndard
            if concentration_ratio < 0.70:
                warnings.append(f"CR_BP: {concentration_ratio:.1%} al pic principal")
                quality_score += 10
    else:
        # Column Mode: thresholds segons volum
        if volume_uL >= 400:
            cr_config = CR_THRESHOLDS['COLUMN_400']
            protocol_desc = "400µL"
        else:
            cr_config = CR_THRESHOLDS['COLUMN_100']
            protocol_desc = "100µL"

        fail_threshold = cr_config['fail']
        warn_threshold = cr_config['warning']

        if concentration_ratio < fail_threshold:
            issues.append(
                f"CR_FAIL: {concentration_ratio:.1%} < {fail_threshold:.0%} "
                f"(protocol {protocol_desc})"
            )
            quality_score += 50
        elif concentration_ratio < warn_threshold:
            warnings.append(
                f"CR_WARNING: {concentration_ratio:.1%} < {warn_threshold:.0%} "
                f"(protocol {protocol_desc})"
            )
            quality_score += 15

    # 8. Comparació històrica (si tenim seq_path)
    # Filtrar per concentració i volum per comparar "pomes amb pomes"
    historical_comparison = None
    reference_comparison = None
    current_area = khp_data.get('area', 0)

    if seq_path:
        mode = "BP" if khp_data.get('is_bp', False) else "COLUMN"
        conc_ppm = khp_data.get('conc_ppm', None)
        volume_uL = khp_data.get('volume_uL', None)
        doc_mode = khp_data.get('doc_mode', None)
        uib_sensitivity = khp_data.get('uib_sensitivity', None)

        historical_comparison = compare_khp_historical(
            current_area=current_area,
            current_concentration_ratio=concentration_ratio,
            seq_path=seq_path,
            mode=mode,
            conc_ppm=conc_ppm,
            volume_uL=volume_uL,
            doc_mode=doc_mode,
            uib_sensitivity=uib_sensitivity
        )

        if historical_comparison['status'] == 'INVALID':
            for issue in historical_comparison['issues']:
                issues.append(f"HISTORICAL: {issue}")
            quality_score += 100
        elif historical_comparison['status'] == 'WARNING':
            for warn in historical_comparison['warnings']:
                warnings.append(f"HISTORICAL: {warn}")
            quality_score += 20
        elif historical_comparison['status'] == 'INSUFFICIENT_DATA':
            # FALLBACK: Usar valors de referència de config
            ref = _get_reference_area(mode, conc_ppm, volume_uL, doc_mode, uib_sensitivity)
            if ref and ref['area_mean'] > 0:
                ref_mean = ref['area_mean']
                ref_std = ref['area_std']
                area_deviation_pct = abs(current_area - ref_mean) / ref_mean * 100

                # Thresholds segons mode
                threshold = 100.0 if mode == "BP" else 20.0

                reference_comparison = {
                    'source': ref['source'],
                    'ref_mean': ref_mean,
                    'ref_std': ref_std,
                    'area_deviation_pct': area_deviation_pct,
                    'threshold': threshold
                }

                if area_deviation_pct > threshold:
                    issues.append(
                        f"REFERENCE: Desviació àrea {area_deviation_pct:.0f}% > {threshold:.0f}% "
                        f"(vs {ref['source']}: {ref_mean}±{ref_std})"
                    )
                    quality_score += 100
                elif area_deviation_pct > threshold * 0.5:
                    warnings.append(
                        f"REFERENCE: Desviació àrea {area_deviation_pct:.0f}% "
                        f"(vs {ref['source']}: {ref_mean}±{ref_std})"
                    )
                    quality_score += 20

    # Determinar validesa
    is_valid = quality_score < 100 and len(issues) == 0

    return {
        'is_valid': is_valid,
        'issues': issues,
        'warnings': warnings,
        'quality_score': quality_score,
        'concentration_ratio': concentration_ratio,
        'historical_comparison': historical_comparison,
        'reference_comparison': reference_comparison
    }


def validate_khp_for_alignment(t_doc, y_doc, t_dad, y_a254, t_uib=None, y_uib=None,
                               method="COLUMN", repair_batman=True,
                               seq_path=None, conc_ppm=None, volume_uL=None,
                               doc_mode=None, uib_sensitivity=None):
    """
    Valida si el KHP és adequat per calcular shifts d'alineament.

    Aquesta funció ha de cridar-se ABANS de calcular els shifts per assegurar
    que les dades KHP són fiables.

    Criteris de validació:
    1. RATIO_LOW: ratio A254/DOC < 0.015 indica contaminació
    2. TIMEOUT_HS: timeout detectat a zona HS (18-23 min per COLUMN)
    3. NO_PEAK: no es pot identificar pic clar
    4. INTENSITY_EXTREME: intensitat molt diferent de l'esperat
    5. BATMAN: pic amb valley al cim (artefacte detector)
    6. HISTORICAL_DEVIATION: àrea desvia significativament de l'històric

    Args:
        t_doc, y_doc: Senyal DOC (Direct o UIB)
        t_dad, y_a254: Senyal A254
        t_uib, y_uib: Senyal UIB (opcional)
        method: "COLUMN" o "BP"
        repair_batman: Si True, repara pics Batman per millorar precisió t_max
        seq_path: Path de la SEQ (per comparació històrica)
        conc_ppm: Concentració KHP en ppm (per comparació històrica)
        volume_uL: Volum d'injecció en µL (per comparació històrica)

    Returns:
        dict amb:
            - valid: bool
            - issues: list de problemes detectats
            - warnings: list d'avisos
            - metrics: dict amb mètriques calculades
            - y_doc_clean: senyal DOC netejat (si Batman reparat)
            - t_max_corrected: t_max corregit si Batman reparat
    """
    result = {
        "valid": True,
        "issues": [],
        "warnings": [],
        "metrics": {},
        "y_doc_clean": None,
        "t_max_corrected": None,
    }

    # Verificar dades mínimes
    if t_doc is None or y_doc is None or len(t_doc) < 50:
        result["valid"] = False
        result["issues"].append("INSUFFICIENT_DOC_DATA")
        return result

    if t_dad is None or y_a254 is None or len(t_dad) < 50:
        result["valid"] = False
        result["issues"].append("INSUFFICIENT_DAD_DATA")
        return result

    # Netejar dades
    t_doc = np.asarray(t_doc, dtype=float)
    y_doc = np.asarray(y_doc, dtype=float)
    y_a254 = np.asarray(y_a254, dtype=float)

    # === 0. DETECTAR I REPARAR BATMAN (si activat) ===
    if method == "COLUMN":
        peak_zone = (t_doc >= 15) & (t_doc <= 30)
    else:
        peak_zone = (t_doc >= 0) & (t_doc <= 5)

    t_peak_zone = t_doc[peak_zone]
    y_peak_zone = y_doc[peak_zone]

    batman_info = None
    y_doc_working = y_doc.copy()

    if len(t_peak_zone) > 20:
        batman_info = detect_batman(t_peak_zone, y_peak_zone, top_pct=0.20, min_valley_depth=0.02)
        result["metrics"]["batman_detected"] = batman_info.get("is_batman", False)

        if batman_info.get("is_batman", False):
            result["warnings"].append(
                f"BATMAN: Detectat patró pic-vall-pic (profunditat {batman_info.get('max_depth', 0)*100:.1f}%)"
            )

            if repair_batman:
                try:
                    from hpsec_core import repair_with_parabola
                    y_repaired, repair_info, was_repaired = repair_with_parabola(t_peak_zone, y_peak_zone)
                    if was_repaired:
                        y_doc_working[peak_zone] = y_repaired
                        result["y_doc_clean"] = y_doc_working
                        result["metrics"]["batman_repaired"] = True
                        result["warnings"].append("BATMAN_REPAIRED: Pic reparat amb paràbola")

                        idx_max_repaired = np.argmax(y_repaired)
                        t_max_corrected = t_peak_zone[idx_max_repaired]
                        result["t_max_corrected"] = float(t_max_corrected)
                except Exception as e:
                    result["metrics"]["batman_repair_error"] = str(e)

    # Trobar pics - usar y_doc_working (reparat si Batman)
    idx_max_doc = np.argmax(y_doc_working)
    idx_max_a254 = np.argmax(y_a254)
    t_max_doc = t_doc[idx_max_doc]
    t_max_a254 = t_dad[idx_max_a254]

    result["metrics"]["t_max_doc"] = float(t_max_doc)
    result["metrics"]["t_max_a254"] = float(t_max_a254)
    result["metrics"]["intensity_doc"] = float(np.max(y_doc_working))
    result["metrics"]["intensity_a254"] = float(np.max(y_a254))

    # === 1. VERIFICAR POSICIÓ PIC ===
    if method == "COLUMN":
        if not (15 <= t_max_doc <= 28):
            result["warnings"].append(f"PEAK_POSITION_UNUSUAL: t_max={t_max_doc:.1f} min (esperat 18-25)")
        if not (15 <= t_max_a254 <= 28):
            result["warnings"].append(f"A254_PEAK_POSITION_UNUSUAL: t_max={t_max_a254:.1f} min")
    else:
        if not (0.3 <= t_max_doc <= 5):
            result["warnings"].append(f"PEAK_POSITION_UNUSUAL: t_max={t_max_doc:.1f} min (esperat 0.5-3)")

    # === 2. CALCULAR RATIO A254/DOC ===
    if method == "COLUMN":
        t_start = max(0, t_max_doc - 5)
        t_end = t_max_doc + 8
    else:
        t_start = max(0, t_max_doc - 1)
        t_end = t_max_doc + 2

    # Àrea DOC - usar y_doc_working (reparat si Batman)
    mask_doc = (t_doc >= t_start) & (t_doc <= t_end)
    if np.sum(mask_doc) > 5:
        baseline_doc = np.percentile(y_doc_working[mask_doc], 5)
        y_doc_corr = y_doc_working[mask_doc] - baseline_doc
        y_doc_corr[y_doc_corr < 0] = 0
        area_doc = np.trapezoid(y_doc_corr, t_doc[mask_doc])
    else:
        area_doc = 0

    # Àrea A254
    mask_a254 = (t_dad >= t_start) & (t_dad <= t_end)
    if np.sum(mask_a254) > 5:
        baseline_a254 = np.percentile(y_a254[mask_a254], 5)
        y_a254_corr = y_a254[mask_a254] - baseline_a254
        y_a254_corr[y_a254_corr < 0] = 0
        area_a254 = np.trapezoid(y_a254_corr, t_dad[mask_a254])
    else:
        area_a254 = 0

    # Calcular ratio
    if area_doc > 0:
        ratio = area_a254 / area_doc
        result["metrics"]["ratio_a254_doc"] = float(ratio)
        result["metrics"]["area_doc"] = float(area_doc)
        result["metrics"]["area_a254"] = float(area_a254)

        if ratio < 0.015:
            result["valid"] = False
            result["issues"].append(f"RATIO_LOW: {ratio:.4f} < 0.015 (possible contaminació)")
        elif ratio < 0.020:
            result["warnings"].append(f"RATIO_BORDERLINE: {ratio:.4f}")
    else:
        result["valid"] = False
        result["issues"].append("NO_DOC_AREA: No s'ha pogut calcular àrea DOC")

    # === 3. DETECTAR TIMEOUT A ZONA HS (només COLUMN) ===
    if method == "COLUMN":
        timeout_info = detect_timeout(t_doc)
        result["metrics"]["timeout_info"] = {
            "has_timeout": timeout_info.get("has_timeout", False),
            "count": timeout_info.get("count", 0),
        }

        for to in timeout_info.get("timeouts", []):
            t_start_to = to.get("t_start_min", 0)
            t_end_to = to.get("t_end_min", 0)
            if t_start_to <= 23 and t_end_to >= 18:
                result["valid"] = False
                result["issues"].append(
                    f"TIMEOUT_HS: Timeout {to.get('duration_sec', 0):.0f}s a {t_start_to:.1f}-{t_end_to:.1f} min"
                )
                break

    # === 4. VERIFICAR INTENSITAT ===
    # Thresholds basats en valors típics per cada mode:
    # COLUMN: KHP típic 400-800 mAU (volum 400µL), 100-200 mAU (volum 100µL)
    # BP: KHP típic 150-300 mAU (volum 100µL)
    intensity = np.max(y_doc)
    result["metrics"]["intensity_doc"] = float(intensity)

    if method == "COLUMN":
        # COLUMN: rang normal 100-1500, extrem >3000 o <30
        if intensity < 30:
            result["valid"] = False
            result["issues"].append(f"INTENSITY_TOO_LOW: {intensity:.0f} mAU (min 30)")
        elif intensity < 80:
            result["warnings"].append(f"INTENSITY_LOW: {intensity:.0f} mAU (típic >100)")
        elif intensity > 3000:
            result["valid"] = False
            result["issues"].append(f"INTENSITY_EXTREME: {intensity:.0f} mAU (>3x normal, possible error concentració)")
        elif intensity > 1500:
            result["warnings"].append(f"INTENSITY_HIGH: {intensity:.0f} mAU")
    else:
        # BP: rang normal 100-600, extrem >1500 o <30
        if intensity < 30:
            result["valid"] = False
            result["issues"].append(f"INTENSITY_TOO_LOW: {intensity:.0f} mAU (min 30)")
        elif intensity < 80:
            result["warnings"].append(f"INTENSITY_LOW: {intensity:.0f} mAU (típic >100)")
        elif intensity > 1500:
            result["valid"] = False
            result["issues"].append(f"INTENSITY_EXTREME: {intensity:.0f} mAU (>3x normal, possible error concentració)")
        elif intensity > 800:
            result["warnings"].append(f"INTENSITY_HIGH: {intensity:.0f} mAU")

    # === 5. VERIFICAR COHERÈNCIA PICS ===
    diff_peaks = abs(t_max_doc - t_max_a254)
    result["metrics"]["peak_diff_min"] = float(diff_peaks)

    if diff_peaks > 2.0:
        result["warnings"].append(f"PEAK_MISMATCH: DOC i A254 difereixen {diff_peaks:.1f} min")

    # === 6. COMPARACIÓ HISTÒRICA (si tenim paràmetres) ===
    if seq_path and conc_ppm is not None and volume_uL is not None:
        try:
            # Calcular àrea del pic principal per comparar
            # Usar y_doc_working (reparat si Batman)
            area_doc = result["metrics"].get("area_doc", 0)
            area_total = np.trapezoid(np.maximum(y_doc_working - np.percentile(y_doc_working, 5), 0), t_doc) if len(y_doc_working) > 5 else 0
            concentration_ratio = area_doc / area_total if area_total > 0 else 0

            historical = compare_khp_historical(
                current_area=area_doc,
                current_concentration_ratio=concentration_ratio,
                seq_path=seq_path,
                mode=method,
                conc_ppm=conc_ppm,
                volume_uL=volume_uL,
                doc_mode=doc_mode,
                uib_sensitivity=uib_sensitivity
            )

            result["metrics"]["historical_comparison"] = {
                "status": historical.get("status", "UNKNOWN"),
                "area_deviation_pct": historical.get("area_deviation_pct", 0),
                "n_calibrations": historical.get("historical_stats", {}).get("n_calibrations", 0) if historical.get("historical_stats") else 0,
            }

            if historical.get("status") == "INVALID":
                for issue in historical.get("issues", []):
                    result["valid"] = False
                    result["issues"].append(f"HISTORICAL: {issue}")
            elif historical.get("status") == "WARNING":
                for warn in historical.get("warnings", []):
                    result["warnings"].append(f"HISTORICAL: {warn}")
            elif historical.get("status") == "INSUFFICIENT_DATA":
                result["warnings"].append(f"HISTORICAL: Dades insuficients per comparar ({method}, {conc_ppm}ppm, {volume_uL}µL)")

        except Exception as e:
            result["warnings"].append(f"HISTORICAL: Error en comparació: {e}")

    return result


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

        # Verificar columnes
        has_direct = "DOC (mAU)" in df_doc.columns or "DOC_Direct (mAU)" in df_doc.columns
        has_uib = "DOC_UIB (mAU)" in df_doc.columns

        if not has_direct and not has_uib:
            return result

        direct_col = "DOC_Direct (mAU)" if "DOC_Direct (mAU)" in df_doc.columns else "DOC (mAU)"

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
        result["details"]["t_max_a254"] = float(t_max_a254)

        # Calcular shift per DOC_UIB
        if has_uib:
            y_uib = df_doc["DOC_UIB (mAU)"].values
            idx_max_uib = np.argmax(y_uib)
            t_max_uib = t[idx_max_uib]

            shift_uib = t_max_a254 - t_max_uib
            result["details"]["t_max_uib"] = float(t_max_uib)
            result["details"]["shift_uib_raw"] = float(shift_uib)

            if abs(shift_uib) > tolerance_min:
                result["shift_uib"] = float(shift_uib)
                result["aligned"] = False

        # Calcular shift per DOC_Direct
        if has_direct:
            y_direct = df_doc[direct_col].values
            idx_max_direct = np.argmax(y_direct)
            t_max_direct = t[idx_max_direct]

            shift_direct = t_max_a254 - t_max_direct
            result["details"]["t_max_direct"] = float(t_max_direct)
            result["details"]["shift_direct_raw"] = float(shift_direct)

            if abs(shift_direct) > tolerance_min:
                result["shift_direct"] = float(shift_direct)
                result["aligned"] = False

    except Exception as e:
        result["details"]["error"] = str(e)

    return result


def get_a254_for_alignment(df_dad_khp=None, path_export3d=None, path_dad1a=None):
    """
    Obté senyal A254 per alineament, amb prioritat:
    1. MasterFile 3-DAD_KHP (si df_dad_khp proporcionat)
    2. Export3D
    3. DAD1A

    Args:
        df_dad_khp: DataFrame de la fulla 3-DAD_KHP del MasterFile (pot ser None)
        path_export3d: Camí al fitxer Export3D (pot ser None)
        path_dad1a: Camí al fitxer DAD1A (pot ser None)

    Returns:
        (t, y, source): Arrays de temps i senyal A254, i string indicant la font
    """
    # Prioritat 1: MasterFile 3-DAD_KHP
    if df_dad_khp is not None and not df_dad_khp.empty:
        try:
            cols = df_dad_khp.columns.tolist()
            t_col = None
            y_col = None

            for c in cols:
                c_lower = str(c).lower()
                if 'time' in c_lower and t_col is None:
                    t_col = c
                elif ('value' in c_lower or 'mau' in c_lower) and y_col is None:
                    y_col = c

            if t_col and y_col:
                t = pd.to_numeric(df_dad_khp[t_col], errors="coerce").to_numpy()
                y = pd.to_numeric(df_dad_khp[y_col], errors="coerce").to_numpy()
                valid = np.isfinite(t) & np.isfinite(y)
                if np.sum(valid) > 10:
                    return t[valid], y[valid], "MasterFile_3-DAD_KHP"
        except Exception:
            pass

    # Prioritat 2 i 3: Export3D o DAD1A
    # Import local per evitar dependència circular
    from hpsec_import import llegir_dad_amb_fallback
    return llegir_dad_amb_fallback(path_export3d, path_dad1a, wavelength="254")


# =============================================================================
# GESTIÓ CALIBRACIONS LOCALS (CHECK/data/)
# =============================================================================

def get_local_data_path(seq_path):
    """
    Retorna el path de la carpeta CHECK/data/ d'una SEQ.
    Nova ubicació unificada per tots els JSONs locals.
    """
    if not seq_path:
        return None
    return os.path.join(seq_path, "CHECK", LOCAL_DATA_FOLDER)


def ensure_local_data_folder(seq_path):
    """Crea la carpeta CHECK/data/ si no existeix."""
    data_path = get_local_data_path(seq_path)
    if data_path and not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)
    return data_path




def load_local_calibrations(seq_path):
    """
    Carrega l'historial LOCAL de calibracions d'una SEQ.
    Ubicació: CHECK/data/calibration_result.json
    """
    data_path = get_local_data_path(seq_path)
    if not data_path:
        return []

    filepath = os.path.join(data_path, CALIBRATION_FILENAME)
    if not os.path.exists(filepath):
        return []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get("calibrations", [])
    except Exception as e:
        print(f"Error carregant calibracions: {e}")
        return []


def save_local_calibrations(seq_path, calibrations):
    """
    Guarda l'historial LOCAL de calibracions d'una SEQ a CHECK/data/.
    """
    data_path = ensure_local_data_folder(seq_path)
    if not data_path:
        return False

    filepath = os.path.join(data_path, CALIBRATION_FILENAME)

    try:
        data = {
            "version": "2.0",
            "seq_name": os.path.basename(seq_path),
            "updated": datetime.now().isoformat(),
            "calibrations": calibrations
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        return True
    except Exception as e:
        print(f"Error guardant CHECK/data: {e}")
        return False


def get_active_calibration(seq_path, mode=None):
    """
    Retorna la calibració activa d'una SEQ.
    """
    calibrations = load_local_calibrations(seq_path)

    for cal in calibrations:
        if cal.get("is_active", False) and not cal.get("is_outlier", False):
            if mode is None or cal.get("mode") == mode:
                return cal

    # Si no hi ha cap activa, retornar la més recent no-outlier
    for cal in calibrations:
        if not cal.get("is_outlier", False):
            if mode is None or cal.get("mode") == mode:
                return cal

    return None


def get_all_active_calibrations(seq_path, mode=None):
    """
    Retorna TOTES les calibracions actives d'una SEQ (una per condition_key).

    Permet tenir múltiples calibracions actives per a diferents condicions
    (ex: KHP2@100µL i KHP2@50µL).

    Args:
        seq_path: Path de la seqüència
        mode: Filtre opcional per mode ("COLUMN" o "BP")

    Returns:
        Llista de calibracions actives, una per cada condition_key única.
    """
    calibrations = load_local_calibrations(seq_path)
    active_by_condition = {}

    for cal in calibrations:
        # Ignorar outliers i inactives
        if not cal.get("is_active", False) or cal.get("is_outlier", False):
            continue
        # Filtre de mode si especificat
        if mode and cal.get("mode") != mode:
            continue

        # Clau única per condició
        key = get_condition_key(
            cal.get("mode", ""),
            cal.get("volume_uL", 0),
            cal.get("conc_ppm", 0)
        )

        # Només guardar la primera (més recent) per cada condition_key
        if key not in active_by_condition:
            active_by_condition[key] = cal

    return list(active_by_condition.values())


def get_calibration_for_conditions(seq_path, volume_uL, mode=None, conc_ppm=None):
    """
    Retorna la calibració que coincideix amb les condicions especificades.

    Args:
        seq_path: Path de la seqüència
        volume_uL: Volum d'injecció de la mostra
        mode: Mode opcional ("COLUMN" o "BP")
        conc_ppm: Concentració KHP opcional (si None, busca qualsevol conc)

    Returns:
        Calibració que coincideix o None si no es troba.
    """
    active_cals = get_all_active_calibrations(seq_path, mode)

    # Buscar calibració amb volum coincident
    for cal in active_cals:
        cal_volume = cal.get("volume_uL", 0)
        if cal_volume == volume_uL:
            if conc_ppm is None or cal.get("conc_ppm", 0) == conc_ppm:
                return cal

    # Si no es troba exacta, retornar la primera activa del mode (fallback)
    if active_cals:
        return active_cals[0]

    return None


def is_seq_calibrated(seq_path, mode=None):
    """
    Comprova si una SEQ ja té calibració.

    Returns:
        (bool, dict): (té calibració, calibració activa si existeix)
    """
    active = get_active_calibration(seq_path, mode)
    return (active is not None, active)


def generate_calibration_id():
    """Genera un ID únic per una calibració."""
    return f"CAL_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


# =============================================================================
# GESTIÓ KHP_HISTORY (GLOBAL)
# =============================================================================

def get_registry_folder(seq_path=None):
    """
    Retorna la carpeta REGISTRY on es guarden els històrics globals.
    Ubicació: Definida a hpsec_config.json (paths.registry_folder)

    El paràmetre seq_path es manté per compatibilitat però s'ignora.
    """
    return get_registry_path()


def get_history_path(seq_path):
    """
    Retorna el path del fitxer d'històric KHP.
    Ubicació: PARENT_FOLDER/REGISTRY/KHP_History.json
    """
    registry = get_registry_folder(seq_path)
    if not registry:
        return None
    return os.path.join(registry, KHP_HISTORY_FILENAME)


def get_samples_history_path(seq_path):
    """
    Retorna el path del fitxer d'històric de mostres.
    Ubicació: PARENT_FOLDER/REGISTRY/Samples_History.json
    """
    registry = get_registry_folder(seq_path)
    if not registry:
        return None
    return os.path.join(registry, SAMPLES_HISTORY_FILENAME)


def load_khp_history(seq_path):
    """
    Carrega l'històric de calibracions KHP.
    Ubicació: PARENT_FOLDER/REGISTRY/KHP_History.json
    """
    history_path = get_history_path(seq_path)
    if not history_path or not os.path.exists(history_path):
        return []

    try:
        with open(history_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get("calibrations", [])
    except Exception as e:
        print(f"Error carregant històric KHP: {e}")
        return []


def save_khp_history(seq_path, calibrations):
    """
    Guarda l'històric de calibracions KHP.
    """
    history_path = get_history_path(seq_path)
    if not history_path:
        return False

    try:
        data = {
            "version": "1.0",
            "updated": datetime.now().isoformat(),
            "calibrations": calibrations
        }
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        return True
    except Exception as e:
        print(f"Error guardant històric KHP: {e}")
        return False


def get_khp_from_history(seq_path, target_seq_name, mode="COLUMN"):
    """Obté una calibració específica de l'històric."""
    history = load_khp_history(seq_path)
    for cal in history:
        if cal.get("seq_name") == target_seq_name and cal.get("mode") == mode:
            return cal
    return None


def get_best_khp_from_history(seq_path, mode="COLUMN", exclude_current=True):
    """
    Obté la millor calibració de l'històric (no outlier).
    """
    history = load_khp_history(seq_path)
    current_seq = os.path.basename(seq_path)

    for cal in history:
        if cal.get("is_outlier", False):
            continue
        if cal.get("mode") != mode:
            continue
        if exclude_current and cal.get("seq_name") == current_seq:
            continue
        return cal

    return None


def migrate_history_to_v2(seq_path, dry_run=False):
    """
    Migra calibracions de KHP_History.json al format v2:
    1. factor (obsolet) → RF (Response Factor)
    2. Afegeix camps de traçabilitat de selecció de rèpliques

    factor = conc / area (obsolet)
    RF = area / conc (nou format)

    Args:
        seq_path: Path de la seqüència (per localitzar Dades3)
        dry_run: Si True, només mostra què faria sense modificar

    Returns:
        dict amb:
            - migrated_rf: llista de seq_names on s'ha migrat factor→RF
            - migrated_selection: llista de seq_names on s'han afegit camps de selecció
            - skipped: llista de seq_names ja actualitzades
            - errors: llista d'errors
    """
    result = {'migrated_rf': [], 'migrated_selection': [], 'skipped': [], 'errors': []}

    history = load_khp_history(seq_path)
    if not history:
        return result

    modified = False
    for cal in history:
        seq_name = cal.get('seq_name', 'unknown')
        was_modified = False

        # 1. Migrar factor → RF
        if cal.get('rf', 0) == 0:
            old_factor = cal.get('factor', 0)
            area = cal.get('area', 0)
            conc = cal.get('conc_ppm', 0)

            rf = 0
            if old_factor > 0:
                rf = 1.0 / old_factor
            elif area > 0 and conc > 0:
                rf = area / conc

            if rf > 0:
                if not dry_run:
                    cal['rf'] = rf
                    if 'rf_doc' not in cal:
                        cal['rf_doc'] = rf
                    if 'rf_direct' not in cal:
                        cal['rf_direct'] = rf
                    if 'factor' in cal:
                        del cal['factor']
                result['migrated_rf'].append(seq_name)
                was_modified = True

        # 2. Afegir camps de selecció si no existeixen
        if 'selection' not in cal:
            n_replicas = cal.get('n_replicas', 1)
            if not dry_run:
                cal['selection'] = {
                    'method': 'average' if n_replicas > 1 else 'single',
                    'reason': 'migrated_from_legacy',
                    'selected_replicas': list(range(1, n_replicas + 1)),
                    'n_replicas_available': n_replicas,
                    'is_manual': False,
                }
            result['migrated_selection'].append(seq_name)
            was_modified = True

        if was_modified:
            modified = True
        elif seq_name not in result['migrated_rf'] and seq_name not in result['migrated_selection']:
            result['skipped'].append(seq_name)

    # Guardar canvis
    if modified and not dry_run:
        save_khp_history(seq_path, history)

    return result


# =============================================================================
# ANÀLISI KHP
# =============================================================================

def analizar_khp_data(t_doc, y_doc_net, metadata, df_dad=None, config=None):
    """
    Analitza dades KHP en memòria (sense llegir Excel).

    Versió de analizar_khp_consolidado que rep dades directament.
    Usada per calibrate_from_import().

    Args:
        t_doc: Array de temps (min)
        y_doc_net: Array de senyal DOC (amb baseline restada)
        metadata: Dict amb:
            - name: Nom de la mostra (ex: "KHP2")
            - conc_ppm: Concentració en ppm
            - replica: Número de rèplica
            - method: "BP" o "COLUMN"
            - seq_path: Path de la SEQ (per volum injecció)
        df_dad: DataFrame DAD opcional (amb "time (min)" i columnes wavelength)
        config: Configuració

    Returns:
        Dict amb dades d'anàlisi (igual que analizar_khp_consolidado)
    """
    config = {**DEFAULT_CONFIG, **(config or {})}

    # Extreure metadata
    name = metadata.get("name", "KHP")
    conc = metadata.get("conc_ppm", 0)
    replica = metadata.get("replica", "1")
    method = metadata.get("method", "COLUMN")
    seq_path = metadata.get("seq_path", "")
    volume_uL_meta = metadata.get("volume_uL")  # Volum del metadata (si disponible)

    if conc == 0:
        # Intentar extreure de nom
        conc = extract_khp_conc(name)

    if conc == 0:
        return None

    # Netejar NaN
    t_doc = np.asarray(t_doc)
    y_doc_net = np.asarray(y_doc_net)
    mask = np.isfinite(t_doc) & np.isfinite(y_doc_net)
    t_doc, y_doc_net = t_doc[mask], y_doc_net[mask]

    if len(t_doc) < 10:
        return None

    # Detectar si és BP
    t_max_chromato = float(np.max(t_doc))
    is_bp_chromato = (method == "BP") or t_max_chromato < 20

    # Detectar pics
    all_peaks = detect_all_peaks(t_doc, y_doc_net, config["peak_min_prominence_pct"])
    peak_info = detect_main_peak(t_doc, y_doc_net, config["peak_min_prominence_pct"])

    if not peak_info.get('valid', False):
        return None

    t_retention = peak_info.get('t_max', 0)

    # Baseline stats
    mode = "BP" if is_bp_chromato else "COLUMN"
    bl_stats = get_baseline_stats(t_doc, y_doc_net, mode=mode)

    # Límits del pic
    peak_idx = peak_info.get('peak_idx', int(np.argmax(y_doc_net)))
    left_idx = peak_info.get('left_idx', 0)
    right_idx = peak_info.get('right_idx', len(y_doc_net) - 1)

    # Buscar límits en all_peaks
    for pk in all_peaks:
        if pk['idx'] == peak_idx or abs(pk['t'] - peak_info['t_max']) < 0.1:
            left_idx = pk['left_idx']
            right_idx = pk['right_idx']
            break

    # Expandir límits si cal
    original_left_idx = left_idx
    original_right_idx = right_idx

    expansion = expand_integration_limits_to_baseline(
        t_doc, y_doc_net, left_idx, right_idx, peak_idx,
        baseline_threshold_pct=15,
        min_width_minutes=1.0,
        max_width_minutes=6.0 if is_bp_chromato else 10.0,
        is_bp=is_bp_chromato
    )

    left_idx = expansion['left_idx']
    right_idx = expansion['right_idx']
    limits_expanded = not expansion['original_valid']

    if limits_expanded:
        new_area = float(trapezoid(y_doc_net[left_idx:right_idx+1], t_doc[left_idx:right_idx+1]))
        old_area = peak_info.get('area', 0)
        peak_info['area'] = new_area
        peak_info['left_idx'] = left_idx
        peak_info['right_idx'] = right_idx
        peak_info['t_start'] = float(t_doc[left_idx])
        peak_info['t_end'] = float(t_doc[right_idx])
        peak_info['limits_expanded'] = True

    # Simetria i SNR (usant funcions de hpsec_core)
    symmetry = calculate_symmetry(t_doc, y_doc_net, peak_idx, left_idx, right_idx)
    # SNR inline: garantir mínim noise per evitar divisions per zero
    baseline_std = bl_stats.get("std", 0.01)
    signal_range = float(np.max(y_doc_net) - np.min(y_doc_net)) if len(y_doc_net) > 0 else 1.0
    min_std = max(0.5, signal_range * 0.001)
    noise = max(baseline_std, min_std)
    snr = float((y_doc_net[peak_idx] - bl_stats["mean"]) / noise)

    # Timeout detection
    timeout_info = detect_timeout(t_doc)
    has_timeout = timeout_info['n_timeouts'] > 0
    timeout_severity = timeout_info.get('severity', 'OK')

    # Batman/Anomalies
    t_peak_seg = t_doc[left_idx:right_idx+1]
    y_peak_seg = y_doc_net[left_idx:right_idx+1]
    anomaly_info = detect_peak_anomaly(t_peak_seg, y_peak_seg)
    has_batman = anomaly_info.get('is_batman', False)
    has_irregular = anomaly_info.get('is_irregular', False)
    smoothness = anomaly_info.get('smoothness', 100.0)

    # DAD 254nm
    shift_khp = 0.0
    has_dad = df_dad is not None and not df_dad.empty
    t_dad = None
    dad_254 = None
    t_dad_max = None
    dad_peak_info = None
    a254_area = 0.0
    a254_doc_ratio = 0.0

    if has_dad and "time (min)" in df_dad.columns:
        # Buscar columna 254
        col_254 = None
        for c in df_dad.columns:
            if "254" in str(c):
                col_254 = c
                break

        if col_254:
            t_dad = pd.to_numeric(df_dad["time (min)"], errors="coerce").to_numpy()
            dad_254 = pd.to_numeric(df_dad[col_254], errors="coerce").to_numpy()

            dad_mask = np.isfinite(t_dad) & np.isfinite(dad_254)
            t_dad, dad_254 = t_dad[dad_mask], dad_254[dad_mask]

            if len(t_dad) > 10:
                t_doc_max = t_at_max(t_doc, y_doc_net)
                t_dad_max = t_at_max(t_dad, dad_254)

                if t_doc_max and t_dad_max:
                    shift_khp = t_dad_max - t_doc_max

                dad_peak_info = detect_main_peak(t_dad, dad_254, config["peak_min_prominence_pct"])

                if dad_peak_info and dad_peak_info.get('valid'):
                    t_start_doc = peak_info.get('t_start', t_doc[left_idx])
                    t_end_doc = peak_info.get('t_end', t_doc[right_idx])
                    t_start_dad = t_start_doc + shift_khp
                    t_end_dad = t_end_doc + shift_khp

                    dad_left_idx = int(np.searchsorted(t_dad, t_start_dad))
                    dad_right_idx = int(np.searchsorted(t_dad, t_end_dad))
                    dad_left_idx = max(0, min(dad_left_idx, len(t_dad) - 1))
                    dad_right_idx = max(0, min(dad_right_idx, len(t_dad) - 1))

                    if dad_right_idx > dad_left_idx:
                        a254_area = float(trapezoid(dad_254[dad_left_idx:dad_right_idx+1],
                                                   t_dad[dad_left_idx:dad_right_idx+1]))
                        if a254_area > 0:
                            a254_doc_ratio = peak_info['area'] / a254_area

    # Àrees amb integració sobre baseline
    baseline_mean = bl_stats.get('mean', 0)
    baseline_std = bl_stats.get('std', 0.1)

    t_peak = t_doc[left_idx:right_idx+1]
    y_peak = y_doc_net[left_idx:right_idx+1]
    peak_integration = integrate_above_baseline(
        t_peak, y_peak,
        baseline_mean=baseline_mean,
        baseline_std=baseline_std,
        threshold_sigma=3.0
    )
    area_main_peak = peak_integration['area']

    total_integration = integrate_above_baseline(
        t_doc, y_doc_net,
        baseline_mean=baseline_mean,
        baseline_std=baseline_std,
        threshold_sigma=3.0
    )
    area_total = total_integration['area']
    concentration_ratio = area_main_peak / area_total if area_total > 0 else 1.0

    # Volum injecció (prioritza metadata, després calcula)
    if volume_uL_meta is not None:
        volume_uL = volume_uL_meta
    elif seq_path:
        volume_uL = get_injection_volume(seq_path, is_bp_chromato)
    else:
        volume_uL = 100

    # Qualitat
    quality_score = 0
    quality_issues = []

    # Simetria alta (>1.5) NO es penalitza per KHP - és normal per un patró
    if symmetry < 0.8:
        quality_score += 10
        quality_issues.append(f"Simetria baixa ({symmetry:.2f})")
    if snr < 10:
        quality_score += 20
        quality_issues.append(f"SNR baix ({snr:.1f})")
    if has_batman:
        quality_score += 50
        quality_issues.append("Doble pic (Batman)")
    if has_irregular:
        quality_score += 30
        quality_issues.append(f"Pic irregular (smoothness={smoothness:.0f}%)")
    # Timeout: només penalitza si afecta l'interval d'integració del pic
    if has_timeout:
        peak_timeout = timeout_affects_peak(timeout_info, t_doc, left_idx, right_idx)
        if peak_timeout['affects_peak']:
            overlap = peak_timeout['overlap_pct']
            if overlap > 30:  # >30% del pic afectat
                quality_score += 150
                quality_issues.append(f"TIMEOUT CRÍTIC (afecta {overlap:.0f}% del pic)")
            elif overlap > 10:  # 10-30% afectat
                quality_score += 100
                quality_issues.append(f"TimeOUT en pic ({overlap:.0f}% afectat)")
            else:  # <10% afectat
                quality_score += 20
                quality_issues.append(f"TimeOUT marginal ({overlap:.0f}% afectat)")
        # else: Timeout fora del pic - NO penalitza ni mostra warning
    if len(all_peaks) > 3:
        quality_score += 5 * (len(all_peaks) - 3)
        quality_issues.append(f"Múltiples pics ({len(all_peaks)})")
    if limits_expanded:
        quality_issues.append("Límits expandits")

    # =========================================================================
    # NOVES MÈTRIQUES: FWHM, RF, RF_V, CR per tots els senyals
    # =========================================================================

    # FWHM per DOC
    fwhm_doc = calculate_fwhm(t_doc, y_doc_net, peak_idx, left_idx, right_idx)

    # FWHM per 254nm
    fwhm_254 = np.nan
    if has_dad and dad_peak_info and dad_peak_info.get('valid'):
        dad_peak_idx = dad_peak_info.get('peak_idx', 0)
        dad_left_idx = dad_peak_info.get('left_idx', 0)
        dad_right_idx = dad_peak_info.get('right_idx', len(t_dad) - 1 if t_dad is not None else 0)
        if t_dad is not None and dad_254 is not None:
            fwhm_254 = calculate_fwhm(t_dad, dad_254, dad_peak_idx, dad_left_idx, dad_right_idx)

    # RF = Area / Concentració (ppm)
    rf_doc = peak_info['area'] / conc if conc > 0 else 0.0

    # RF_V = Area / (Concentració × Volum) - normalitzat per condicions
    rf_v_doc = peak_info['area'] / (conc * volume_uL) if conc > 0 and volume_uL > 0 else 0.0

    # RF i RF_V per 254nm
    rf_254 = a254_area / conc if conc > 0 and a254_area > 0 else 0.0
    rf_v_254 = a254_area / (conc * volume_uL) if conc > 0 and volume_uL > 0 and a254_area > 0 else 0.0

    # CR per 254nm (àrea pic principal / àrea total)
    # Necessitem calcular l'àrea total de 254nm
    cr_254 = np.nan
    a254_area_total = 0.0
    if has_dad and t_dad is not None and dad_254 is not None and len(t_dad) > 10:
        # Integrar tot el cromatograma 254nm
        a254_area_total = float(trapezoid(dad_254, t_dad))
        if a254_area_total > 0 and a254_area > 0:
            cr_254 = a254_area / a254_area_total

    return {
        'name': name,  # Nom del KHP (ex: "KHP2", "KHP2_50")
        'filename': f"{name}_R{replica}",
        'filepath': seq_path,
        'conc_ppm': conc,
        'area': peak_info['area'],
        'shift_min': shift_khp,
        'shift_sec': shift_khp * 60,
        'peak_info': peak_info,
        'has_dad': has_dad,
        't_doc_max': t_at_max(t_doc, y_doc_net),
        't_dad_max': t_dad_max,
        't_doc': t_doc,
        'y_doc': y_doc_net,
        't_dad': t_dad,
        'y_dad_254': dad_254,
        'symmetry': symmetry,
        'snr': snr,
        'baseline_stats': bl_stats,
        'all_peaks_count': len(all_peaks),
        'all_peaks': all_peaks,
        'quality_score': quality_score,
        'quality_issues': quality_issues,
        'has_batman': has_batman,
        'has_timeout': has_timeout,
        'timeout_info': timeout_info,
        'timeout_severity': timeout_severity,
        'anomaly_info': anomaly_info,
        'has_irregular': has_irregular,
        'smoothness': smoothness,
        'dad_peak_info': dad_peak_info,
        'peak_left_idx': left_idx,
        'peak_right_idx': right_idx,
        'is_bp': is_bp_chromato,
        'doc_mode': "MEMORY",
        'seq_date': '',
        't_retention': t_retention,
        'baseline_valid': True,
        'limits_expanded': limits_expanded,
        'expansion_info': expansion,
        'a254_area': a254_area,
        'a254_doc_ratio': a254_doc_ratio,
        'height': float(y_doc_net[peak_idx]),
        'area_total': area_total,
        'area_main_peak': area_main_peak,
        'concentration_ratio': concentration_ratio,
        'volume_uL': volume_uL,
        'uib_sensitivity': None,
        # Noves mètriques per anàlisi de qualitat
        'fwhm_doc': fwhm_doc,
        'fwhm_254': fwhm_254,
        'rf_doc': rf_doc,
        'rf_v_doc': rf_v_doc,
        'rf_254': rf_254,
        'rf_v_254': rf_v_254,
        'cr_254': cr_254,
        'a254_area_total': a254_area_total,
    }


# =============================================================================
# REGISTRE DE CALIBRACIONS
# =============================================================================

def _extract_replicas_info(khp_data):
    """
    Extreu informació resumida de cada replicat per guardar a CHECK/data.
    """
    replicas = khp_data.get('replicas', khp_data.get('all_khp_data', []))
    if not replicas:
        peak_info = khp_data.get('peak_info', {})
        return [{
            "filename": khp_data.get('filename', 'N/A'),
            "area": khp_data.get('area', 0),
            "t_start": peak_info.get('t_start', 0),
            "t_end": peak_info.get('t_end', 0),
            "t_max": peak_info.get('t_max', khp_data.get('t_retention', 0)),
            "baseline": khp_data.get('baseline_stats', {}).get('mean', 0),
        }]

    replicas_info = []
    for rep in replicas:
        peak_info = rep.get('peak_info', {})
        replicas_info.append({
            "filename": rep.get('filename', 'N/A'),
            "area": rep.get('area', peak_info.get('area', 0)),
            "t_start": peak_info.get('t_start', 0),
            "t_end": peak_info.get('t_end', 0),
            "t_max": peak_info.get('t_max', rep.get('t_retention', 0)),
            "baseline": rep.get('baseline_stats', {}).get('mean', 0),
            "symmetry": rep.get('symmetry', 1.0),
            "snr": rep.get('snr', 0),
        })

    return replicas_info


def _get_reference_area(mode, conc_ppm, volume_uL, doc_mode, uib_sensitivity):
    """
    Obté valors de referència de la config quan no hi ha històric.

    Returns:
        dict amb 'area_mean', 'area_std', 'source' o None si no hi ha referència
    """
    try:
        config_path = os.path.join(os.path.dirname(__file__), "hpsec_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            ref_values = config.get('calibration', {}).get('reference_values', {})

            # Construir clau de cerca
            # Format: MODE_KHPx_VOLuL_DOCMODE
            conc_str = f"KHP{int(conc_ppm)}" if conc_ppm else "KHP"
            vol_str = f"{int(volume_uL)}uL" if volume_uL else ""

            # Provar diferents combinacions
            keys_to_try = []
            if doc_mode and 'UIB' in doc_mode and uib_sensitivity:
                keys_to_try.append(f"{mode}_{conc_str}_{vol_str}_UIB_{uib_sensitivity}")
            if doc_mode:
                keys_to_try.append(f"{mode}_{conc_str}_{vol_str}_{doc_mode}")
            keys_to_try.append(f"{mode}_{conc_str}_{vol_str}")

            for key in keys_to_try:
                if key in ref_values:
                    ref = ref_values[key]
                    return {
                        'area_mean': ref.get('area_mean', 0),
                        'area_std': ref.get('area_std', 0),
                        'source': f"config:{key}"
                    }
    except Exception:
        pass
    return None


def register_calibration(seq_path, khp_data, khp_source, mode="COLUMN"):
    """
    Registra una nova calibració a l'històric.

    Guarda a DOS llocs:
    1. LOCAL (CHECK/data/calibrations.json) - Historial complet de la SEQ
    2. GLOBAL (KHP_History.json) - Una entrada per SEQ per comparacions

    VALIDACIÓ COMPLETA amb validate_khp_quality():
    - valid_for_shift: Pic clar, sense timeout crític, batman reparat
    - valid_for_calibration: Tots els criteris de qualitat (8 criteris)
    """
    seq_name = os.path.basename(seq_path)

    # Calcular RF (Response Factor) = Àrea / Concentració
    area = khp_data.get('area', 0)
    conc = khp_data.get('conc_ppm', 0)
    rf = area / conc if conc > 0 else 0

    seq_date = khp_data.get('seq_date', '')
    if not seq_date:
        seq_date = datetime.now().isoformat()

    is_bp = khp_data.get('is_bp', False)
    # Usar volum de khp_data si disponible (per múltiples condicions), sino default
    volume = khp_data.get('volume_uL') or get_injection_volume(seq_path, is_bp)
    khp_name = khp_data.get('name', f"KHP{conc}")  # Nom del KHP (ex: "KHP2", "KHP2_50")
    doc_mode = khp_data.get('doc_mode', 'N/A')
    uib_sensitivity = khp_data.get('uib_sensitivity')

    # =========================================================================
    # PARÀMETRES PER SENYAL: Direct (D), UIB (U), 254
    # Veure docs/PARAMETRES_CALIBRACIO.md per definicions
    # =========================================================================

    # --- DIRECT ---
    rf_d = khp_data.get('rf_doc', rf)
    rf_v_d = khp_data.get('rf_v_doc', rf_d / volume * 100 if volume > 0 else 0)
    t_max_d = khp_data.get('t_doc_max', khp_data.get('t_retention', 0))
    fwhm_d = khp_data.get('fwhm_doc', 0)
    snr_d = khp_data.get('snr', 0)
    sym_d = khp_data.get('symmetry', 1.0)
    ar_d = khp_data.get('concentration_ratio', khp_data.get('area_ratio', 1.0))
    d254_d = khp_data.get('a254_doc_ratio', 0)

    # --- UIB ---
    area_u = khp_data.get('area_uib', 0)
    rf_u = khp_data.get('rf_uib', 0)
    if rf_u == 0 and area_u > 0 and conc > 0:
        rf_u = area_u / conc
    rf_v_u = khp_data.get('rf_v_uib', rf_u / volume * 100 if rf_u > 0 and volume > 0 else 0)
    snr_u = khp_data.get('snr_uib', 0)
    d254_u = khp_data.get('a254_doc_ratio_uib', 0)

    # --- 254nm ---
    area_254 = khp_data.get('a254_area', 0)
    rf_254 = khp_data.get('rf_254', 0)
    if rf_254 == 0 and area_254 > 0 and conc > 0:
        rf_254 = area_254 / conc
    rf_v_254 = khp_data.get('rf_v_254', rf_254 / volume * 100 if rf_254 > 0 and volume > 0 else 0)
    fwhm_254 = khp_data.get('fwhm_254', 0)
    ar_254 = khp_data.get('cr_254', khp_data.get('area_ratio_254', 0))
    t_dad_max = khp_data.get('t_dad_max', 0)  # t_max del senyal 254nm (referència)

    peak_info = khp_data.get('peak_info', {})
    t_retention = khp_data.get('t_retention', khp_data.get('t_doc_max', 0))

    # =========================================================================
    # VALIDACIÓ COMPLETA amb validate_khp_quality()
    # Criteris: multi-pic, timeout, batman, simetria, SNR, límits, CR, històric
    # =========================================================================
    all_peaks = khp_data.get('all_peaks', [])
    timeout_info = khp_data.get('timeout_info', {})
    anomaly_info = khp_data.get('anomaly_info', {})

    validation_result = validate_khp_quality(
        khp_data=khp_data,
        all_peaks=all_peaks,
        timeout_info=timeout_info,
        anomaly_info=anomaly_info,
        seq_path=seq_path
    )

    # Extreure resultats de validació
    valid_for_calibration = validation_result.get('is_valid', True)
    calibration_issues = validation_result.get('issues', [])
    calibration_warnings = validation_result.get('warnings', [])
    quality_score = validation_result.get('quality_score', 0)
    quality_issues = calibration_issues + calibration_warnings

    # =========================================================================
    # VALIDACIÓ PER SHIFT (alineació temporal)
    # Criteris més relaxats: només necessitem posició pic fiable
    # =========================================================================
    valid_for_shift = True
    shift_issues = []

    if not t_retention or t_retention <= 0:
        valid_for_shift = False
        shift_issues.append("No s'ha detectat pic")

    if timeout_info.get('severity') == 'CRITICAL':
        valid_for_shift = False
        shift_issues.append("Timeout crític a zona pic")

    if khp_data.get('has_batman', False) and not khp_data.get('batman_repaired', False):
        shift_issues.append("Batman no reparat (shift imprecís)")

    # Alias per compatibilitat
    is_outlier = not valid_for_calibration

    # Obtenir info històrica per registre (no per validació, ja feta a validate_khp_quality)
    historical_comparison = validation_result.get('historical_comparison', {})

    # rf_mass = Area / µg DOC (normalitzat per massa injectada)
    rf_mass = khp_data.get('rf_mass', 0)
    if rf_mass == 0 and area > 0 and conc > 0 and volume > 0:
        rf_mass = area * 1000 / (conc * volume)

    entry = {
        "cal_id": generate_calibration_id(),
        "seq_name": seq_name,
        "seq_path": seq_path,
        "date": seq_date,
        "seq_date": seq_date,
        "date_processed": datetime.now().isoformat(),
        "mode": mode,
        "khp_name": khp_name,  # Nom del KHP (ex: "KHP2", "KHP2_50")
        "khp_file": khp_data.get('filename', 'N/A'),
        "khp_source": khp_source,
        "doc_mode": doc_mode,
        "conc_ppm": conc,
        "volume_uL": volume,
        "uib_sensitivity": uib_sensitivity,
        "is_bp": is_bp,
        "condition_key": get_condition_key(mode, volume, conc),  # Clau única per condició

        # =====================================================================
        # PARÀMETRES DIRECT (D) - Senyal principal
        # =====================================================================
        "area": area,                      # Àrea pic principal Direct
        "area_total": khp_data.get('area_total', 0),
        "rf": rf,                          # Response Factor Direct (àrea/conc)
        "rf_mass": rf_mass,                # RF normalitzat per massa (àrea/µg DOC)
        "rf_v": rf_v_d,                    # RF normalitzat per volum
        "t_retention": t_max_d,            # Temps pic màxim
        "fwhm_doc": fwhm_d,                # FWHM Direct
        "snr": snr_d,                      # SNR Direct
        "symmetry": sym_d,                 # Simetria
        "area_ratio": ar_d,                # Àrea pic / Àrea total (antic concentration_ratio)
        "n_peaks": khp_data.get('all_peaks_count', 1),  # Nombre de pics (antic all_peaks_count)
        "shift_sec": khp_data.get('shift_sec', khp_data.get('shift_min', 0) * 60),
        "shift_min": khp_data.get('shift_min', 0),

        # =====================================================================
        # PARÀMETRES UIB (U) - Senyal alternatiu
        # =====================================================================
        "area_u": area_u,                  # Àrea pic principal UIB
        "rf_u": rf_u,                      # Response Factor UIB
        "rf_v_u": rf_v_u,                  # RF UIB normalitzat per volum
        "snr_u": snr_u,                    # SNR UIB
        "d254_u": d254_u,                  # Ratio DOC/254 amb UIB

        # =====================================================================
        # PARÀMETRES 254nm - Senyal DAD
        # =====================================================================
        "area_254": area_254,              # Àrea pic 254nm
        "rf_254": rf_254,                  # Response Factor 254nm
        "rf_v_254": rf_v_254,              # RF 254nm normalitzat per volum
        "fwhm_254": fwhm_254,              # FWHM 254nm
        "ar_254": ar_254,                  # Area Ratio 254nm
        "t_dad_max": t_dad_max,            # t_max del 254nm (referència per shift)
        "d254_d": d254_d,                  # Ratio DOC/254 amb Direct

        # =====================================================================
        # TRAÇABILITAT I QUALITAT
        # =====================================================================
        "n_replicas": khp_data.get('n_replicas', 1),
        "rsd": khp_data.get('rsd', 0),
        "selection": khp_data.get('selection', {
            'method': 'legacy',
            'reason': 'pre_v2.1',
            'selected_replicas': list(range(1, khp_data.get('n_replicas', 1) + 1)),
            'n_replicas_available': khp_data.get('n_replicas', 1),
            'is_manual': False,
        }),
        "replica_comparison": khp_data.get('replica_comparison', {}),
        "quality_score": quality_score,
        "quality_issues": quality_issues,
        "has_batman": khp_data.get('has_batman', False),
        "has_timeout": khp_data.get('has_timeout', False),

        # =====================================================================
        # VALIDACIÓ
        # =====================================================================
        "valid_for_shift": valid_for_shift,
        "shift_issues": shift_issues,
        "valid_for_calibration": valid_for_calibration,
        "calibration_issues": calibration_issues,
        "calibration_warnings": calibration_warnings,

        # Override manual
        "manual_override": None,
        "manual_override_reason": "",
        "manual_override_by": "",
        "manual_override_date": None,

        # Estat
        "is_outlier": is_outlier,
        "is_active": valid_for_calibration,
        "status": "INVALID_CAL" if not valid_for_calibration else (
            "INVALID_SHIFT" if not valid_for_shift else "OK"
        ),

        # =====================================================================
        # ALTRES (detall / compatibilitat)
        # =====================================================================
        "baseline_valid": khp_data.get('baseline_valid', True),
        "limits_expanded": khp_data.get('limits_expanded', False),
        "t_start": peak_info.get('t_start', 0),
        "t_end": peak_info.get('t_end', 0),
        "peak_left_idx": khp_data.get('peak_left_idx', peak_info.get('left_idx', 0)),
        "peak_right_idx": khp_data.get('peak_right_idx', peak_info.get('right_idx', 0)),
        "baseline": khp_data.get('baseline_stats', {}).get('mean', 0),
        "baseline_std": khp_data.get('baseline_stats', {}).get('std', 0),
        "replicas_info": _extract_replicas_info(khp_data),
        "validation_details": {
            "quality_score": quality_score,
            "issues": calibration_issues,
            "warnings": calibration_warnings,
            "historical_comparison": historical_comparison,
        },

        # Compatibilitat amb codi antic (DEPRECAT - usar els nous noms)
        "rf_doc": rf_d,
        "rf_uib": rf_u,
        "a254_area": area_254,
        "a254_doc_ratio": d254_d,
        "area_main_peak": khp_data.get('area_main_peak', area),
        "concentration_ratio": ar_d,  # Deprecat, usar area_ratio
        "all_peaks_count": khp_data.get('all_peaks_count', 1),  # Deprecat, usar n_peaks
    }

    # 1. GUARDAR A LOCAL (CHECK/data)
    local_cals = load_local_calibrations(seq_path)

    # Clau única per aquesta calibració (permet múltiples condicions actives)
    new_condition_key = get_condition_key(mode, volume, conc)

    if not valid_for_calibration:
        # Calibració invàlida per quantitatiu: NO desactivar l'anterior vàlida
        # S'afegeix al registre per traçabilitat però no s'activa
        pass
    else:
        # Calibració vàlida: desactivar les anteriors amb MATEIXA CONDICIÓ
        # (no desactivem calibracions d'altres condicions, ex: KHP2@100µL no afecta KHP2@50µL)
        for cal in local_cals:
            cal_condition_key = get_condition_key(
                cal.get("mode", ""),
                cal.get("volume_uL", 0),
                cal.get("conc_ppm", 0)
            )
            if cal_condition_key == new_condition_key:
                cal["is_active"] = False

    local_cals.insert(0, entry)
    save_local_calibrations(seq_path, local_cals)

    # 2. GUARDAR A GLOBAL (KHP_History.json)
    global_cals = load_khp_history(seq_path)

    # Actualitzar o afegir entrada al global (una per seq+condition_key)
    updated = False
    for i, cal in enumerate(global_cals):
        cal_condition_key = get_condition_key(
            cal.get("mode", ""),
            cal.get("volume_uL", 0),
            cal.get("conc_ppm", 0)
        )
        if cal.get("seq_name") == seq_name and cal_condition_key == new_condition_key:
            global_cals[i] = entry
            updated = True
            break

    if not updated:
        global_cals.append(entry)

    global_cals.sort(key=lambda x: x.get("date", ""), reverse=True)
    save_khp_history(seq_path, global_cals)

    return entry


def set_calibration_override(seq_path, cal_id, override_value, reason="", user="manual"):
    """
    Aplica un override manual a una calibració.

    Permet forçar una calibració com a vàlida o invàlida, independentment
    de la validació automàtica.

    Args:
        seq_path: Path de la seqüència
        cal_id: ID de la calibració a modificar
        override_value: True (forçar vàlid), False (forçar invàlid), None (tornar a automàtic)
        reason: Motiu de l'override
        user: Usuari que fa l'override

    Returns:
        dict amb resultat: {"success": bool, "message": str, "entry": dict}
    """
    from datetime import datetime

    # LOCAL
    local_cals = load_local_calibrations(seq_path)
    entry_found = None
    mode = None

    for cal in local_cals:
        if cal.get("cal_id") == cal_id:
            cal["manual_override"] = override_value
            cal["manual_override_reason"] = reason
            cal["manual_override_by"] = user
            cal["manual_override_date"] = datetime.now().isoformat() if override_value is not None else None

            # Actualitzar is_active segons override
            if override_value is not None:
                cal["is_active"] = override_value
                cal["status"] = "MANUAL_VALID" if override_value else "MANUAL_INVALID"
            else:
                # Tornar a validació automàtica
                cal["is_active"] = cal.get("valid_for_calibration", True)
                if not cal.get("valid_for_calibration", True):
                    cal["status"] = "INVALID_CAL"
                elif not cal.get("valid_for_shift", True):
                    cal["status"] = "INVALID_SHIFT"
                else:
                    cal["status"] = "OK"

            entry_found = cal.copy()
            break

    if not entry_found:
        return {"success": False, "message": f"No s'ha trobat calibració amb ID {cal_id}", "entry": None}

    # Si s'activa manualment, desactivar les altres amb MATEIXA CONDICIÓ
    # (no afectem altres condicions, ex: KHP2@100µL no afecta KHP2@50µL)
    if override_value is True:
        target_condition_key = get_condition_key(
            entry_found.get("mode", ""),
            entry_found.get("volume_uL", 0),
            entry_found.get("conc_ppm", 0)
        )
        for cal in local_cals:
            if cal.get("cal_id") != cal_id:
                cal_condition_key = get_condition_key(
                    cal.get("mode", ""),
                    cal.get("volume_uL", 0),
                    cal.get("conc_ppm", 0)
                )
                if cal_condition_key == target_condition_key:
                    cal["is_active"] = False

    save_local_calibrations(seq_path, local_cals)

    # GLOBAL
    global_cals = load_khp_history(seq_path)
    seq_name = os.path.basename(seq_path)

    for cal in global_cals:
        if cal.get("cal_id") == cal_id:
            cal["manual_override"] = override_value
            cal["manual_override_reason"] = reason
            cal["manual_override_by"] = user
            cal["manual_override_date"] = entry_found.get("manual_override_date")
            cal["is_active"] = entry_found.get("is_active")
            cal["status"] = entry_found.get("status")
            break

    save_khp_history(seq_path, global_cals)

    action = "validat" if override_value else ("invalidat" if override_value is False else "retornat a automàtic")
    return {
        "success": True,
        "message": f"Calibració {cal_id} {action} manualment",
        "entry": entry_found
    }


def set_replica_selection(seq_path, cal_id, selection_method, user="manual"):
    """
    Canvia la selecció de rèpliques d'una calibració.

    Permet seleccionar manualment quines rèpliques usar per la calibració,
    recalculant els valors segons la nova selecció.

    Args:
        seq_path: Path de la seqüència
        cal_id: ID de la calibració a modificar
        selection_method: "average", "R1", "R2", "best_quality"
        user: Qui fa el canvi

    Returns:
        dict amb success, message, entry actualitzada
    """
    # Carregar calibracions locals
    local_cals = load_local_calibrations(seq_path)

    entry_found = None
    entry_idx = None

    for i, cal in enumerate(local_cals):
        if cal.get("cal_id") == cal_id:
            entry_found = cal
            entry_idx = i
            break

    if not entry_found:
        return {"success": False, "message": f"No s'ha trobat calibració amb ID {cal_id}", "entry": None}

    # Obtenir rèpliques originals
    replicas_info = entry_found.get('replicas_info', [])
    replica_comparison = entry_found.get('replica_comparison', {})
    replica_details = replica_comparison.get('replica_details', [])

    # Si no tenim dades de rèpliques, no podem canviar
    if not replica_details and not replicas_info:
        return {"success": False, "message": "No hi ha dades de rèpliques per recalcular", "entry": None}

    # Usar replica_details si disponible, sinó replicas_info
    replicas = replica_details if replica_details else replicas_info
    n_replicas = len(replicas)

    if n_replicas < 1:
        return {"success": False, "message": "No hi ha rèpliques disponibles", "entry": None}

    # Validar selecció
    if selection_method.startswith('R'):
        rep_num = int(selection_method[1:])
        if rep_num > n_replicas:
            return {"success": False, "message": f"Rèplica {rep_num} no existeix (només {n_replicas} disponibles)", "entry": None}

    # Calcular nous valors segons selecció
    old_selection = entry_found.get('selection', {})

    if selection_method == 'average':
        # Mitjana de totes
        areas = [r.get('area', 0) for r in replicas]
        shifts = [r.get('shift_sec', 0) for r in replicas]
        a254_ratios = [r.get('a254_doc_ratio', 0) for r in replicas if r.get('a254_doc_ratio', 0) > 0]

        new_area = float(np.mean(areas)) if areas else entry_found.get('area', 0)
        new_shift_sec = float(np.mean(shifts)) if shifts else entry_found.get('shift_sec', 0)
        new_a254_ratio = float(np.mean(a254_ratios)) if a254_ratios else entry_found.get('a254_doc_ratio', 0)
        selected_replicas = list(range(1, n_replicas + 1))
        status_text = f"Mitjana R{'+R'.join(map(str, selected_replicas))}"

    elif selection_method.startswith('R'):
        # Rèplica específica
        rep_num = int(selection_method[1:])
        rep_idx = rep_num - 1
        rep = replicas[rep_idx]

        new_area = rep.get('area', entry_found.get('area', 0))
        new_shift_sec = rep.get('shift_sec', entry_found.get('shift_sec', 0))
        new_a254_ratio = rep.get('a254_doc_ratio', entry_found.get('a254_doc_ratio', 0))
        selected_replicas = [rep_num]
        status_text = f"Manual R{rep_num}"

    elif selection_method == 'best_quality':
        # Millor per quality_score
        sorted_reps = sorted(replicas, key=lambda x: x.get('quality_score', 0))
        best = sorted_reps[0]
        best_num = best.get('replica_num', 1)

        new_area = best.get('area', entry_found.get('area', 0))
        new_shift_sec = best.get('shift_sec', entry_found.get('shift_sec', 0))
        new_a254_ratio = best.get('a254_doc_ratio', entry_found.get('a254_doc_ratio', 0))
        selected_replicas = [best_num]
        status_text = f"Millor qualitat R{best_num}"

    else:
        return {"success": False, "message": f"Mètode de selecció desconegut: {selection_method}", "entry": None}

    # Recalcular RF (Response Factor = area / conc)
    conc = entry_found.get('conc_ppm', 0)
    new_rf = new_area / conc if conc > 0 else 0

    # Actualitzar entrada
    entry_found['area'] = new_area
    entry_found['rf'] = new_rf
    # Eliminar factor obsolet si existeix
    if 'factor' in entry_found:
        del entry_found['factor']
    entry_found['shift_sec'] = new_shift_sec
    entry_found['shift_min'] = new_shift_sec / 60.0
    entry_found['a254_doc_ratio'] = new_a254_ratio

    # Actualitzar traçabilitat
    entry_found['selection'] = {
        'method': selection_method,
        'reason': 'manual',
        'selected_replicas': selected_replicas,
        'n_replicas_available': n_replicas,
        'is_manual': True,
        'previous_method': old_selection.get('method', 'unknown'),
        'changed_by': user,
        'changed_date': datetime.now().isoformat(),
    }

    # Guardar
    local_cals[entry_idx] = entry_found
    save_local_calibrations(seq_path, local_cals)

    # Actualitzar GLOBAL
    global_cals = load_khp_history(seq_path)
    for i, cal in enumerate(global_cals):
        if cal.get("cal_id") == cal_id:
            global_cals[i] = entry_found
            break
    save_khp_history(seq_path, global_cals)

    return {
        "success": True,
        "message": f"Selecció canviada a {status_text}",
        "entry": entry_found,
        "changes": {
            "old_method": old_selection.get('method', 'unknown'),
            "new_method": selection_method,
            "old_area": old_selection.get('area', 0),
            "new_area": new_area,
        }
    }


def mark_calibration_as_outlier(seq_path, seq_name, mode="COLUMN", is_outlier=True, cal_id=None):
    """
    Marca/desmarca una calibració com a outlier.
    DEPRECAT: Usar set_calibration_override() per overrides manuals.
    """
    # LOCAL
    local_cals = load_local_calibrations(seq_path)

    for cal in local_cals:
        if cal.get("mode") == mode:
            if cal_id is None or cal.get("cal_id") == cal_id:
                cal["is_outlier"] = is_outlier
                if is_outlier:
                    cal["is_active"] = False

    # Si es marca com outlier, activar la següent vàlida
    if is_outlier:
        for cal in local_cals:
            if cal.get("mode") == mode and not cal.get("is_outlier", False):
                cal["is_active"] = True
                break

    save_local_calibrations(seq_path, local_cals)

    # GLOBAL
    global_cals = load_khp_history(seq_path)

    for cal in global_cals:
        if cal.get("seq_name") == seq_name and cal.get("mode") == mode:
            cal["is_outlier"] = is_outlier
            if is_outlier:
                cal["is_active"] = False
            break

    save_khp_history(seq_path, global_cals)


def update_calibration_validation(seq_path=None, update_global=True):
    """
    Actualitza les calibracions existents amb els nous camps de validació.

    Llegeix cada calibració del JSON, executa validate_khp_quality() i
    actualitza els camps:
    - valid_for_shift
    - valid_for_calibration
    - calibration_issues
    - calibration_warnings
    - quality_score
    - shift_issues
    - reference_comparison

    Args:
        seq_path: Path d'una SEQ específica, o None per actualitzar global
        update_global: Si True, actualitza KHP_History.json global

    Returns:
        dict amb resum: {"updated": int, "errors": list, "details": list}
    """
    result = {"updated": 0, "errors": [], "details": []}

    if seq_path:
        # Actualitzar només una SEQ
        paths_to_update = [seq_path]
    else:
        # Actualitzar totes les SEQ del global
        paths_to_update = []

    if update_global:
        # Carregar històric global
        # Necessitem un seq_path vàlid per trobar el JSON
        base_path = r"C:\Users\Lequia\Desktop\Dades2"

        # Trobar una SEQ vàlida per obtenir el path correcte
        sample_seq = None
        for item in os.listdir(base_path):
            if "_SEQ" in item:
                sample_seq = os.path.join(base_path, item)
                break

        if not sample_seq:
            result['errors'].append("No s'ha trobat cap SEQ a Dades2")
            return result

        global_cals = load_khp_history(sample_seq)

        for cal in global_cals:
            cal_seq_path = cal.get('seq_path', '')
            if cal_seq_path and os.path.isdir(cal_seq_path):
                if cal_seq_path not in paths_to_update:
                    paths_to_update.append(cal_seq_path)

        # Revalidar cada calibració
        updated_cals = []
        for cal in global_cals:
            try:
                # Crear khp_data simulat des de la calibració existent
                khp_data = {
                    'area': cal.get('area', 0),
                    'conc_ppm': cal.get('conc_ppm', 0),
                    'volume_uL': cal.get('volume_uL', 0),
                    'doc_mode': cal.get('doc_mode', 'N/A'),
                    'is_bp': cal.get('is_bp', False),
                    'symmetry': cal.get('symmetry', 1.0),
                    'snr': cal.get('snr', 0),
                    'concentration_ratio': cal.get('concentration_ratio', 1.0),
                    'has_batman': cal.get('has_batman', False),
                    'limits_expanded': cal.get('limits_expanded', False),
                    'uib_sensitivity': cal.get('uib_sensitivity'),
                    't_retention': cal.get('t_retention', 0),
                }

                # Timeout i anomaly info (reconstruir si possible)
                timeout_info = {'severity': 'CRITICAL' if cal.get('has_timeout') else 'OK'}
                anomaly_info = {'is_batman': cal.get('has_batman', False), 'is_irregular': False}

                # Buscar quality_issues existents per detectar irregular
                existing_issues = cal.get('quality_issues', [])
                for issue in existing_issues:
                    if 'irregular' in issue.lower() or 'smoothness' in issue.lower():
                        anomaly_info['is_irregular'] = True
                        # Intentar extreure smoothness
                        import re
                        match = re.search(r'smoothness[=:\s]*(\d+)', issue, re.I)
                        if match:
                            anomaly_info['smoothness'] = int(match.group(1))

                all_peaks = []

                # Executar validació
                validation = validate_khp_quality(
                    khp_data=khp_data,
                    all_peaks=all_peaks,
                    timeout_info=timeout_info,
                    anomaly_info=anomaly_info,
                    seq_path=cal.get('seq_path', base_path)
                )

                # Actualitzar camps
                cal['valid_for_calibration'] = validation.get('is_valid', True)
                cal['calibration_issues'] = validation.get('issues', [])
                cal['calibration_warnings'] = validation.get('warnings', [])
                cal['quality_score'] = validation.get('quality_score', 0)

                # Validació shift
                t_retention = cal.get('t_retention', 0)
                valid_for_shift = True
                shift_issues = []

                if not t_retention or t_retention <= 0:
                    valid_for_shift = False
                    shift_issues.append("No s'ha detectat pic")

                if cal.get('has_timeout') and timeout_info.get('severity') == 'CRITICAL':
                    valid_for_shift = False
                    shift_issues.append("Timeout crític")

                if cal.get('has_batman', False):
                    shift_issues.append("Batman detectat")

                cal['valid_for_shift'] = valid_for_shift
                cal['shift_issues'] = shift_issues

                # Camps d'override manual (inicialitzar si no existeixen)
                if 'manual_override' not in cal:
                    cal['manual_override'] = None
                if 'manual_override_reason' not in cal:
                    cal['manual_override_reason'] = ""
                if 'manual_override_by' not in cal:
                    cal['manual_override_by'] = ""
                if 'manual_override_date' not in cal:
                    cal['manual_override_date'] = None

                # Actualitzar is_outlier i is_active segons validació
                if cal['manual_override'] is None:
                    cal['is_outlier'] = not cal['valid_for_calibration']
                    cal['is_active'] = cal['valid_for_calibration']

                # Actualitzar status
                if cal['manual_override'] is True:
                    cal['status'] = "MANUAL_VALID"
                elif cal['manual_override'] is False:
                    cal['status'] = "MANUAL_INVALID"
                elif not cal['valid_for_calibration']:
                    cal['status'] = "INVALID_CAL"
                elif not cal['valid_for_shift']:
                    cal['status'] = "INVALID_SHIFT"
                else:
                    cal['status'] = "OK"

                # Reference comparison
                if validation.get('reference_comparison'):
                    cal['reference_comparison'] = validation['reference_comparison']

                result['details'].append({
                    'seq_name': cal.get('seq_name'),
                    'valid_for_cal': cal['valid_for_calibration'],
                    'valid_for_shift': cal['valid_for_shift'],
                    'issues': cal['calibration_issues']
                })
                result['updated'] += 1

            except Exception as e:
                result['errors'].append(f"{cal.get('seq_name', 'unknown')}: {e}")

            updated_cals.append(cal)

        # Guardar JSON actualitzat
        save_khp_history(sample_seq, updated_cals)

    return result


# =============================================================================
# HELPERS PER FALLBACK SIBLING/HISTORY
# =============================================================================

def _try_sibling_calibration(seq_path, method, config, report_progress=None):
    """
    Busca KHP en SEQ siblings (mateix prefix numeric).

    Returns:
        Dict amb resultats de calibracio o None si no trobat
    """
    import re
    from pathlib import Path

    # Importar aqui per evitar circular import
    try:
        from hpsec_import import import_sequence
    except ImportError:
        return None

    seq_path = Path(seq_path)
    folder_name = seq_path.name
    parent_dir = seq_path.parent

    # Extreure prefix numeric (256, 257, etc.)
    match = re.match(r'^(\d+)', folder_name)
    if not match:
        return None

    seq_id = match.group(1)

    try:
        all_folders = [d for d in parent_dir.iterdir() if d.is_dir()]
        siblings = [d for d in all_folders
                   if d.name.startswith(seq_id) and d.name != folder_name]

        for sib in siblings:
            if report_progress:
                report_progress(16, f"Provant sibling {sib.name}...")

            # Importar sibling
            sib_imported = import_sequence(str(sib))
            if not sib_imported.get("success"):
                continue

            sib_khp = sib_imported.get("khp_samples", [])
            if not sib_khp:
                continue

            # Calibrar amb dades del sibling
            sib_cal = calibrate_from_import(sib_imported, config=config)
            if sib_cal.get("success"):
                sib_cal["khp_source"] = f"SIBLING:{sib.name}"
                return sib_cal

    except Exception as e:
        pass

    return None


def _try_history_calibration(seq_path, method):
    """
    Busca calibracio a KHP_History.json.

    Returns:
        Dict amb rf, khp_conc i history_entry o None
    """
    history = load_khp_history(seq_path)
    if not history:
        return None

    # Filtrar per metode
    method_history = [h for h in history if h.get("mode", "").upper() == method.upper()]
    if not method_history:
        method_history = history  # Usar qualsevol si no hi ha del metode

    if not method_history:
        return None

    # Usar la calibracio mes recent
    latest = method_history[-1]
    khp_conc = latest.get("conc_ppm", 0)
    khp_area = latest.get("area", 0)

    # Obtenir RF (nou format) o calcular des de factor/area (format antic)
    rf = latest.get("rf", 0)
    if rf == 0 and khp_conc > 0 and khp_area > 0:
        rf = khp_area / khp_conc
    elif rf == 0:
        # Fallback: intentar convertir des de factor antic
        old_factor = latest.get("factor", 0)
        if old_factor > 0:
            rf = 1.0 / old_factor  # factor = conc/area, RF = area/conc = 1/factor

    if rf > 0:
        return {
            "rf": rf,
            "rf_direct": latest.get("rf_direct", rf),
            "rf_uib": latest.get("rf_uib", 0),
            "khp_conc": khp_conc,
            "history_entry": latest,
        }

    return None


# =============================================================================
# CALIBRACIÓ DES D'IMPORT (NOVA API)
# =============================================================================

def calibrate_from_import(imported_data, config=None, progress_callback=None):
    """
    Calibra una seqüència usant dades d'import_sequence() (en memòria).

    Versió moderna de calibrate_sequence que NO llegeix Excels.
    Usa les dades y_net calculades per import_sequence().

    IMPORTANT: Calibra AMBDÓS senyals (DOC Direct i DOC UIB) de forma independent
    quan estiguin disponibles.

    Args:
        imported_data: Dict retornat per import_sequence() amb:
            - samples: Dict de mostres
            - khp_samples: Llista de noms de KHP
            - method: "BP" o "COLUMN"
            - seq_path: Path de la SEQ
        config: Configuració opcional
        progress_callback: Funció(pct, msg) per reportar progrés

    Returns:
        Dict amb:
        - success: bool
        - mode: "DUAL", "DIRECT", "UIB" (quins senyals s'han calibrat)
        # Calibració DOC Direct:
        - rf_direct: Response Factor Direct (àrea/conc)
        - shift_direct: Shift temporal Direct vs 254nm (min)
        - khp_area_direct: Àrea KHP amb DOC Direct
        - khp_data_direct: Dades KHP completes per Direct
        # Calibració DOC UIB:
        - rf_uib: Response Factor UIB (àrea/conc)
        - shift_uib: Shift temporal UIB (min)
        - khp_area_uib: Àrea KHP amb DOC UIB
        - khp_data_uib: Dades KHP completes per UIB
        # Calibració 254nm:
        - rf_254: Response Factor 254nm
        # Principal (usa Direct si disponible, sino UIB):
        - rf: Response Factor principal (Direct > UIB)
        - khp_area: Àrea principal
        - khp_conc: Concentració del KHP (ppm)
        - khp_data: Dades KHP principals
        - khp_source: "LOCAL"
        - errors: Llista d'errors
    """
    config = {**DEFAULT_CONFIG, **(config or {})}

    def report_progress(pct, msg):
        if progress_callback:
            progress_callback(pct, msg)

    result = {
        "success": False,
        "mode": None,
        # Calibració Direct
        "rf_direct": 0.0,
        "shift_direct": 0.0,
        "khp_area_direct": 0.0,
        "khp_data_direct": None,
        # Calibració UIB
        "rf_uib": 0.0,
        "shift_uib": 0.0,
        "khp_area_uib": 0.0,
        "khp_data_uib": None,
        # Calibració 254nm
        "rf_254": 0.0,
        # Principal
        "rf": 0.0,
        "khp_area": 0.0,
        "khp_conc": 0.0,
        "khp_data": None,
        "khp_source": "LOCAL",
        "calibration": None,
        "errors": [],
        "warnings": [],
    }

    if not imported_data or not imported_data.get("success", False):
        result["errors"].append("Dades d'import no valides")
        return result

    report_progress(10, "Buscant KHP a les dades importades...")

    # Obtenir info de la seqüència
    seq_path = imported_data.get("seq_path", "")
    method = imported_data.get("method", "COLUMN")
    samples = imported_data.get("samples", {})
    khp_names = imported_data.get("khp_samples", [])

    # Si khp_names està buit, buscar KHP en els noms de mostres
    if not khp_names:
        khp_names = [name for name in samples.keys() if "KHP" in name.upper()]

    if not khp_names:
        # FALLBACK 1: Buscar KHP en SEQ sibling
        report_progress(15, "No KHP local, buscant siblings...")
        sibling_result = _try_sibling_calibration(seq_path, method, config, report_progress)
        if sibling_result:
            # Copiar resultats del sibling
            for key, value in sibling_result.items():
                result[key] = value
            khp_source = sibling_result.get("khp_source", "SIBLING")
            result["khp_source"] = khp_source

            # IMPORTANT: Registrar calibració sibling a KHP_History
            if result.get("khp_data"):
                report_progress(95, "Registrant calibració sibling...")
                calibration = register_calibration(seq_path, result["khp_data"], khp_source, method)
                result["calibration"] = calibration
            result["success"] = True
            return result

        # FALLBACK 2: Usar KHP_History.json
        report_progress(18, "No sibling, buscant historial...")
        history_result = _try_history_calibration(seq_path, method)
        if history_result:
            result["rf"] = history_result.get("rf", 0)
            result["rf_direct"] = history_result.get("rf_direct", 0)
            result["rf_uib"] = history_result.get("rf_uib", 0)
            result["khp_conc"] = history_result.get("khp_conc", 0)
            result["khp_source"] = "HISTORY"
            result["success"] = True
            result["errors"].append("WARN: Usant RF d'historial (només quantificació, no shift)")
            # NOTA: No registrem calibracions HISTORY perquè no tenen khp_data complet
            return result

        result["errors"].append("No s'han trobat mostres KHP (local, sibling, ni historial)")
        return result

    report_progress(20, f"Analitzant {len(khp_names)} KHP...")

    # Analitzar cada KHP per AMBDÓS senyals separadament
    khp_data_direct_list = []
    khp_data_uib_list = []

    for khp_name in khp_names:
        sample = samples.get(khp_name, {})
        replicas = sample.get("replicas", {})

        for rep_num, rep_data in replicas.items():
            direct = rep_data.get("direct") or {}
            uib = rep_data.get("uib") or {}

            # Obtenir DAD si disponible
            dad_data = rep_data.get("dad", {})
            df_dad = dad_data.get("df") if dad_data else None

            # Obtenir volum d'injecció
            injection_info = rep_data.get("injection_info", {})
            inj_volume = injection_info.get("inj_volume", 100)  # Default 100µL

            # Preparar metadata base
            base_metadata = {
                "name": khp_name,
                "conc_ppm": extract_khp_conc(khp_name),
                "volume_uL": inj_volume,
                "replica": str(rep_num),
                "method": method,
                "seq_path": seq_path,
            }

            # Analitzar DOC DIRECT si disponible
            if direct.get("t") is not None and direct.get("y_net") is not None:
                t_direct = direct.get("t")
                y_net_direct = direct.get("y_net")

                metadata_direct = {**base_metadata, "doc_source": "direct"}
                khp_result_direct = analizar_khp_data(t_direct, y_net_direct, metadata_direct, df_dad, config)

                if khp_result_direct:
                    khp_result_direct["doc_source"] = "direct"
                    khp_data_direct_list.append(khp_result_direct)

            # Analitzar DOC UIB si disponible
            if uib.get("t") is not None and uib.get("y_net") is not None:
                t_uib = uib.get("t")
                y_net_uib = uib.get("y_net")

                metadata_uib = {**base_metadata, "doc_source": "uib"}
                khp_result_uib = analizar_khp_data(t_uib, y_net_uib, metadata_uib, df_dad, config)

                if khp_result_uib:
                    khp_result_uib["doc_source"] = "uib"
                    khp_data_uib_list.append(khp_result_uib)

    has_direct = len(khp_data_direct_list) > 0
    has_uib = len(khp_data_uib_list) > 0

    if not has_direct and not has_uib:
        # Diagnòstic: per què no hi ha dades KHP?
        khp_without_doc = []
        for khp_name in khp_names:
            sample = samples.get(khp_name, {})
            replicas = sample.get("replicas", {})
            has_any_doc = False
            for rep_num, rep_data in replicas.items():
                direct = rep_data.get("direct") if rep_data else None
                uib = rep_data.get("uib") if rep_data else None
                if (direct and direct.get("t")) or (uib and uib.get("t")):
                    has_any_doc = True
                    break
            if not has_any_doc:
                khp_without_doc.append(khp_name)

        if khp_without_doc:
            result["errors"].append(f"KHP sense dades DOC: {', '.join(khp_without_doc)} (SEQ sense fitxers TOC?)")
        else:
            result["errors"].append("No s'ha pogut analitzar cap KHP (dades invàlides o pic no detectat)")
        return result

    # Determinar mode
    if has_direct and has_uib:
        result["mode"] = "DUAL"
    elif has_direct:
        result["mode"] = "DIRECT"
    else:
        result["mode"] = "UIB"

    report_progress(50, f"Mode calibracio: {result['mode']}")

    def compare_replicas(replicas):
        """
        Compara rèpliques i calcula mètriques de diferència.

        Returns:
            dict amb mètriques de comparació entre rèpliques
        """
        if len(replicas) < 2:
            return {
                'n_replicas': len(replicas),
                'comparable': False,
                'reason': 'single_replica' if len(replicas) == 1 else 'no_replicas'
            }

        # Extreure mètriques de cada rèplica
        areas = [r.get('area', 0) for r in replicas]
        t_maxs = [r.get('peak_info', {}).get('t_max', 0) or r.get('t_doc_max', 0) for r in replicas]
        snrs = [r.get('snr', 0) for r in replicas]
        symmetries = [r.get('symmetry', 0) for r in replicas]
        a254_ratios = [r.get('a254_doc_ratio', 0) for r in replicas]
        shift_secs = [r.get('shift_sec', 0) for r in replicas]
        quality_scores = [r.get('quality_score', 0) for r in replicas]

        # Calcular estadístiques
        mean_area = float(np.mean(areas)) if areas else 0
        std_area = float(np.std(areas)) if len(areas) > 1 else 0
        rsd_area = (std_area / mean_area * 100) if mean_area > 0 else 0

        # Diferències entre rèpliques (per a 2 rèpliques)
        diff_area_pct = abs(areas[0] - areas[1]) / mean_area * 100 if mean_area > 0 and len(areas) >= 2 else 0
        diff_t_max_sec = abs(t_maxs[0] - t_maxs[1]) * 60 if len(t_maxs) >= 2 else 0  # en segons
        diff_snr = abs(snrs[0] - snrs[1]) if len(snrs) >= 2 else 0
        diff_shift_sec = abs(shift_secs[0] - shift_secs[1]) if len(shift_secs) >= 2 else 0

        # Correlació de perfils (si disponible)
        pearson_profiles = None
        if len(replicas) >= 2:
            # Intentar correlacionar perfils DOC si tenim les dades
            y1 = replicas[0].get('y_doc_net')
            y2 = replicas[1].get('y_doc_net')
            if y1 is not None and y2 is not None and len(y1) == len(y2) and len(y1) > 10:
                try:
                    from scipy.stats import pearsonr
                    pearson_profiles, _ = pearsonr(y1, y2)
                    pearson_profiles = float(pearson_profiles)
                except:
                    pass

        return {
            'n_replicas': len(replicas),
            'comparable': True,
            # Estadístiques globals
            'mean_area': mean_area,
            'std_area': std_area,
            'rsd_area': rsd_area,
            # Diferències entre rèpliques
            'diff_area_pct': diff_area_pct,
            'diff_t_max_sec': diff_t_max_sec,
            'diff_snr': diff_snr,
            'diff_shift_sec': diff_shift_sec,
            'pearson_profiles': pearson_profiles,
            # Valors per rèplica (per mostrar a GUI)
            'replica_details': [
                {
                    'replica_num': i + 1,
                    'area': areas[i] if i < len(areas) else 0,
                    't_max': t_maxs[i] if i < len(t_maxs) else 0,
                    'snr': snrs[i] if i < len(snrs) else 0,
                    'symmetry': symmetries[i] if i < len(symmetries) else 0,
                    'a254_doc_ratio': a254_ratios[i] if i < len(a254_ratios) else 0,
                    'shift_sec': shift_secs[i] if i < len(shift_secs) else 0,
                    'quality_score': quality_scores[i] if i < len(quality_scores) else 0,
                }
                for i in range(len(replicas))
            ]
        }

    def select_best_khp(khp_list, manual_selection=None):
        """
        Processa KHPs agrupant per condicions (nom + volum).

        Retorna LLISTA de calibracions, una per cada combinació de condicions.
        Igual que tenim COLUMN/BP o KHP1/KHP2/KHP5, cada condició genera
        una calibració independent amb estructura idèntica.

        Args:
            khp_list: Llista de resultats d'anàlisi KHP
            manual_selection: None (automàtic) o dict amb:
                - method: "R1", "R2", "average", "best_quality"

        Returns:
            list of dict: Llista de calibracions, cada una amb:
                - name, volume_uL, conc_ppm: Condicions
                - area, rf, rf_mass: Paràmetres de calibració
                - n_replicas, rsd, selection: Traçabilitat
        """
        if not khp_list:
            return []

        # Agrupar per condicions analítiques: (concentració, volum)
        by_key = {}
        for d in khp_list:
            conc = d.get('conc_ppm', 0)
            volume = d.get('volume_uL', 100)
            key = (conc, volume)
            if key not in by_key:
                by_key[key] = []
            by_key[key].append(d)

        # Processar cada grup → una calibració per grup
        calibrations = []

        for key, group_replicas in by_key.items():
            group_conc, group_volume = key
            cal = _process_khp_group(group_replicas, group_conc, group_volume, manual_selection)
            if cal:
                # Calcular RF_mass = Area × 1000 / (conc × vol) = Area / µg DOC
                conc = cal['conc_ppm']
                vol = cal['volume_uL']
                if conc > 0 and vol > 0:
                    cal['rf'] = cal['area'] / conc  # RF tradicional
                    cal['rf_mass'] = cal['area'] * 1000 / (conc * vol)  # RF normalitzat
                calibrations.append(cal)

        # Ordenar per concentració (desc) i volum (desc)
        calibrations.sort(key=lambda c: (-c.get('conc_ppm', 0), -c.get('volume_uL', 0)))

        return calibrations

    def _process_khp_group(replicas, group_conc, group_volume, manual_selection=None):
        """Processa un grup de rèpliques KHP amb les mateixes condicions (conc, vol)."""
        if not replicas:
            return None

        # Assignar número de rèplica si no existeix
        for i, rep in enumerate(replicas):
            if 'replica_num' not in rep:
                rep['replica_num'] = i + 1

        # Comparar rèpliques
        comparison = compare_replicas(replicas)

        # Nom: sempre "KHP" (el patró), els números són atributs (conc, vol)
        group_name = "KHP"

        # Estadístiques bàsiques
        areas = [r['area'] for r in replicas]
        shifts = [r['shift_min'] for r in replicas]
        mean_area = float(np.mean(areas))
        std_area = float(np.std(areas))
        mean_shift = float(np.mean(shifts))
        rsd = float((std_area / mean_area) * 100.0) if mean_area > 0 else 100.0

        # Mètriques addicionals (promig)
        a254_ratios = [r.get('a254_doc_ratio', 0) for r in replicas if r.get('a254_doc_ratio', 0) > 0]
        mean_a254_ratio = float(np.mean(a254_ratios)) if a254_ratios else 0.0
        a254_areas = [r.get('a254_area', 0) for r in replicas if r.get('a254_area', 0) > 0]
        mean_a254_area = float(np.mean(a254_areas)) if a254_areas else 0.0
        shift_secs = [r.get('shift_sec', 0) for r in replicas if r.get('shift_sec', 0) != 0]
        mean_shift_sec = float(np.mean(shift_secs)) if shift_secs else 0.0

        # Mètriques de qualitat (promig de rèpliques)
        snrs = [r.get('snr', 0) for r in replicas if r.get('snr', 0) > 0]
        mean_snr = float(np.mean(snrs)) if snrs else 0.0
        t_retentions = [r.get('t_retention', 0) or r.get('t_doc_max', 0) for r in replicas]
        t_retentions = [t for t in t_retentions if t > 0]
        mean_t_retention = float(np.mean(t_retentions)) if t_retentions else 0.0
        fwhms = [r.get('fwhm_doc', 0) for r in replicas if r.get('fwhm_doc', 0) > 0]
        mean_fwhm = float(np.mean(fwhms)) if fwhms else 0.0
        symmetries = [r.get('symmetry', 1.0) for r in replicas if r.get('symmetry', 0) > 0]
        mean_symmetry = float(np.mean(symmetries)) if symmetries else 1.0
        volumes = [r.get('volume_uL', 0) for r in replicas if r.get('volume_uL', 0) > 0]
        volume_uL = int(volumes[0]) if volumes else 100
        t_dad_maxs = [r.get('t_dad_max', 0) for r in replicas if r.get('t_dad_max', 0) > 0]
        mean_t_dad_max = float(np.mean(t_dad_maxs)) if t_dad_maxs else 0.0

        # Determinar mètode de selecció
        if manual_selection:
            selection_method = manual_selection.get('method', 'average')
            selection_reason = 'manual'
        elif len(replicas) == 1:
            selection_method = 'single'
            selection_reason = 'only_one_replica'
        elif rsd < 10.0:
            selection_method = 'average'
            selection_reason = f'rsd_ok ({rsd:.1f}% < 10%)'
        else:
            selection_method = 'best_quality'
            selection_reason = f'rsd_high ({rsd:.1f}% >= 10%)'

        # Aplicar selecció
        if selection_method == 'average' or selection_method == 'single':
            selected_area = mean_area
            selected_shift_min = mean_shift
            selected_shift_sec = mean_shift_sec
            selected_a254_ratio = mean_a254_ratio
            selected_a254_area = mean_a254_area
            selected_replicas = [r['replica_num'] for r in replicas]
            status = f"Promig R{'+R'.join(map(str, selected_replicas))}" if len(replicas) > 1 else "Única rèplica R1"

        elif selection_method.startswith('R'):
            # Selecció manual d'una rèplica específica
            rep_num = int(selection_method[1:])
            selected_rep = next((r for r in replicas if r.get('replica_num') == rep_num), replicas[0])
            selected_area = selected_rep['area']
            selected_shift_min = selected_rep['shift_min']
            selected_shift_sec = selected_rep.get('shift_sec', 0)
            selected_a254_ratio = selected_rep.get('a254_doc_ratio', 0)
            selected_a254_area = selected_rep.get('a254_area', 0)
            selected_replicas = [rep_num]
            status = f"Manual R{rep_num}"

        else:  # best_quality
            sorted_replicas = sorted(replicas, key=lambda x: x.get('quality_score', 0))
            best = sorted_replicas[0]
            selected_area = best['area']
            selected_shift_min = best['shift_min']
            selected_shift_sec = best.get('shift_sec', 0)
            selected_a254_ratio = best.get('a254_doc_ratio', 0)
            selected_a254_area = best.get('a254_area', 0)
            selected_replicas = [best.get('replica_num', 1)]
            status = f"Millor qualitat R{selected_replicas[0]} (RSD {rsd:.1f}%)"

        return {
            # Valors seleccionats
            'name': group_name,  # Nom del KHP (ex: "KHP2", "KHP2_50")
            'name_full': f"KHP{group_conc}@{group_volume}µL",  # Condicions: conc + volum
            'conc_ppm': group_conc,
            'area': selected_area,
            'shift_min': selected_shift_min,
            'shift_sec': selected_shift_sec,
            'a254_doc_ratio': selected_a254_ratio,
            'a254_area': selected_a254_area,
            'is_bp': replicas[0].get('is_bp', False),

            # Mètriques de qualitat (promig de rèpliques)
            'snr': mean_snr,
            't_retention': mean_t_retention,
            't_doc_max': mean_t_retention,
            't_dad_max': mean_t_dad_max,  # t_max del 254nm (referència)
            'fwhm_doc': mean_fwhm,
            'symmetry': mean_symmetry,
            'volume_uL': group_volume,  # Volum d'aquest grup

            # Traçabilitat de selecció
            'selection': {
                'method': selection_method,          # 'average', 'single', 'best_quality', 'R1', 'R2', etc.
                'reason': selection_reason,          # 'rsd_ok', 'rsd_high', 'manual', 'only_one_replica'
                'selected_replicas': selected_replicas,  # [1, 2] o [1] o [2]
                'n_replicas_available': len(replicas),
                'is_manual': manual_selection is not None,
                'khp_name': group_name,  # Nom del KHP
            },

            # Comparació entre rèpliques
            'replica_comparison': comparison,

            # Estadístiques globals
            'n_replicas': len(replicas),
            'rsd': rsd,
            'mean_area': mean_area,
            'std_area': std_area,

            # Totes les rèpliques (per GUI i recàlcul)
            'replicas': replicas,

            # Status llegible
            'status': status,
        }

    report_progress(60, "Processant calibracions KHP...")

    # Calibrar DOC DIRECT - retorna llista de calibracions (una per condició)
    calibrations_direct = []
    if has_direct:
        calibrations_direct = select_best_khp(khp_data_direct_list)
        if calibrations_direct:
            result["calibrations_direct"] = calibrations_direct
            # Valors principals (primera calibració = major conc/vol)
            primary = calibrations_direct[0]
            result["khp_data_direct"] = primary  # Compatibilitat
            result["khp_area_direct"] = primary['area']
            result["shift_direct"] = primary['shift_min']
            result["rf_direct"] = primary.get('rf', 0)
            result["rf_mass_direct"] = primary.get('rf_mass', 0)
            # Info si hi ha múltiples condicions
            if len(calibrations_direct) > 1:
                all_conditions = [f"KHP{c['conc_ppm']}@{c['volume_uL']}µL" for c in calibrations_direct]
                result["warnings"].append(
                    f"ℹ️ MÚLTIPLES CONDICIONS KHP: {', '.join(all_conditions)}. "
                    f"Cada mostra usarà la calibració amb les seves condicions (conc, vol)."
                )

    # Calibrar DOC UIB - retorna llista de calibracions
    calibrations_uib = []
    if has_uib:
        calibrations_uib = select_best_khp(khp_data_uib_list)
        if calibrations_uib:
            result["calibrations_uib"] = calibrations_uib
            # Valors principals
            primary = calibrations_uib[0]
            result["khp_data_uib"] = primary  # Compatibilitat
            result["khp_area_uib"] = primary['area']
            result["shift_uib"] = primary['shift_min']
            result["rf_uib"] = primary.get('rf', 0)
            result["rf_mass_uib"] = primary.get('rf_mass', 0)

    report_progress(80, "Calculant RF...")

    # Usar Direct com a principal, sino UIB
    if calibrations_direct:
        primary = calibrations_direct[0]
        result["calibrations"] = calibrations_direct  # Llista principal
        result["khp_data"] = primary
        result["khp_area"] = primary['area']
        result["khp_conc"] = primary['conc_ppm']
        result["rf"] = primary.get('rf', 0)
        result["rf_mass"] = primary.get('rf_mass', 0)
    elif calibrations_uib:
        primary = calibrations_uib[0]
        result["calibrations"] = calibrations_uib  # Llista principal
        result["khp_data"] = primary
        result["khp_area"] = primary['area']
        result["khp_conc"] = primary['conc_ppm']
        result["rf"] = primary.get('rf', 0)
        result["rf_mass"] = primary.get('rf_mass', 0)

    if result["rf"] == 0:
        result["errors"].append("WARN: RF és zero (àrea o concentració invàlides)")

    # Afegir comparació històrica
    report_progress(85, "Comparant amb històric...")

    def add_historical_comparison(khp_data, signal_name):
        """Afegeix comparació històrica a khp_data."""
        if not khp_data:
            return
        mode = "BP" if khp_data.get('is_bp', False) else "COLUMN"
        conc_ppm = khp_data.get('conc_ppm')

        # Obtenir volume_uL - primer del khp_data, després de les rèpliques
        volume_uL = khp_data.get('volume_uL')
        if volume_uL is None:
            replicas = khp_data.get('replicas', [])
            if replicas:
                volume_uL = replicas[0].get('volume_uL')
        if volume_uL is None:
            volume_uL = 400 if mode == "COLUMN" else 100

        area = khp_data.get('area', 0)

        # Calcular concentration_ratio
        area_total = khp_data.get('area_total', area)
        concentration_ratio = area / area_total if area_total > 0 else 1.0

        hist_comparison = compare_khp_historical(
            current_area=area,
            current_concentration_ratio=concentration_ratio,
            seq_path=seq_path,
            mode=mode,
            conc_ppm=conc_ppm,
            volume_uL=volume_uL,
            doc_mode=None,  # No filtrar per doc_mode (calibracions antigues són N/A)
            uib_sensitivity=None,
            exclude_outliers=False  # Incloure totes les calibracions
        )
        khp_data['historical_comparison'] = hist_comparison

        # També afegir a cada rèplica
        for rep in khp_data.get('replicas', []):
            rep_area = rep.get('area', 0)
            rep_area_total = rep.get('area_total', rep_area)
            rep_cr = rep_area / rep_area_total if rep_area_total > 0 else 1.0
            rep['historical_comparison'] = compare_khp_historical(
                current_area=rep_area,
                current_concentration_ratio=rep_cr,
                seq_path=seq_path,
                mode=mode,
                conc_ppm=conc_ppm,
                volume_uL=volume_uL,
                doc_mode=None,  # No filtrar per doc_mode
                uib_sensitivity=None,
                exclude_outliers=False
            )

    if result.get("khp_data_direct"):
        add_historical_comparison(result["khp_data_direct"], "Direct")

    if result.get("khp_data_uib"):
        add_historical_comparison(result["khp_data_uib"], "UIB")

    report_progress(90, "Registrant calibracions...")

    # Registrar TOTES les calibracions (una per cada condició)
    calibrations_list = result.get("calibrations", [])
    if calibrations_list:
        registered = []
        for cal_data in calibrations_list:
            mode = "BP" if cal_data.get('is_bp', False) else "COLUMN"
            calibration = register_calibration(seq_path, cal_data, "LOCAL", mode)
            registered.append(calibration)
        result["registered_calibrations"] = registered
        # Compatibilitat: la primera és la principal
        if registered:
            result["calibration"] = registered[0]
    elif result.get("khp_data"):
        # Fallback: format antic amb només khp_data
        mode = "BP" if result["khp_data"].get('is_bp', False) else "COLUMN"
        calibration = register_calibration(seq_path, result["khp_data"], "LOCAL", mode)
        result["calibration"] = calibration

    report_progress(100, "Calibracio completada")

    result["success"] = True
    return result


# =============================================================================
# TEST STANDALONE
# =============================================================================

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("HPSEC Calibrate - Test Standalone")
    print("=" * 60)

    if len(sys.argv) > 1:
        seq_path = sys.argv[1]
    else:
        # Demanar path
        seq_path = input("Introdueix path de la SEQ: ").strip()

    if not os.path.exists(seq_path):
        print(f"ERROR: Path no existeix: {seq_path}")
        sys.exit(1)

    print(f"\nCalibrant: {seq_path}")
    print("-" * 60)

    def progress(pct, msg):
        print(f"  [{pct:3d}%] {msg}")

    result = calibrate_sequence(seq_path, progress_callback=progress)

    print("-" * 60)

    if result["success"]:
        cal = result["calibration"]
        print(f"CALIBRACIÓ OK!")
        print(f"  KHP: {cal['khp_file']} ({result['khp_source']})")
        print(f"  Concentració: {cal['conc_ppm']} ppm")
        print(f"  Àrea: {cal['area']:.2f}")
        print(f"  RF: {cal.get('rf', 0):.2f}")
        print(f"  Mode: {cal['mode']}")
        print(f"  Shift: {cal['shift_sec']:.1f} s")
    else:
        print(f"ERROR: {result['errors']}")
