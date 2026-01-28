"""
hpsec_calibrate.py - Mòdul de calibració HPSEC standalone

Funcions per:
- Anàlisi de KHP (analizar_khp_consolidado, analizar_khp_lote)
- Cerca de KHP (buscar_khp_consolidados, find_khp_in_folder)
- Gestió historial calibracions (CALDATA local, KHP_History global)
- Càlcul de factors de calibració
- Validació qualitat KHP (validate_khp_quality)

Pot usar-se standalone o importat des de HPSEC_Suite.

v1.1 - 2026-01-26: Refactor - funcions detecció mogudes a hpsec_core.py
v1.0 - Versió inicial
"""

__version__ = "1.2.0"
__version_date__ = "2026-01-28"

import os
import re
import glob
import json
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.signal import savgol_filter
from scipy.integrate import trapezoid

# Import funcions de detecció des de hpsec_core (Single Source of Truth)
from hpsec_core import (
    detect_timeout,
    detect_batman,
    detect_peak_anomaly,
    detect_main_peak,
    detect_all_peaks,
    integrate_chromatogram,
    integrate_above_baseline,
    TIMEOUT_CONFIG
)

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

# Fitxers d'historial
KHP_HISTORY_FILENAME = "KHP_History.json"
KHP_HISTORY_EXCEL_FILENAME = "KHP_History.xlsx"
SAMPLES_HISTORY_FILENAME = "Samples_History.json"

# Sistema CALDATA - Carpeta local per cada SEQ
CALDATA_FOLDER = "CALDATA"
CALDATA_FILENAME = "calibrations.json"

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


def obtenir_seq(folder):
    """Extreu ID de seqüència del nom de carpeta."""
    nom = os.path.basename(os.path.normpath(folder))
    m = re.search(r"(\d+[A-Za-z]?)", nom)
    return m.group(1) if m else "000"


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


def is_khp(name):
    """Detecta si és mostra KHP."""
    return "KHP" in str(name).upper()


def extract_khp_conc(filename):
    """
    Extreu la concentració de KHP del nom del fitxer.

    Patrons:
    - KHP2 -> 2 ppm
    - KHP_2 -> 2 ppm
    - KHP-2 -> 2 ppm
    - KHP2_xxx -> 2 ppm
    """
    name = os.path.basename(filename).upper()

    # Patró principal: KHP seguit de número
    patterns = [
        r"KHP[_\-]?(\d+(?:\.\d+)?)",  # KHP2, KHP_2, KHP-2, KHP2.5
        r"KHP\s*(\d+(?:\.\d+)?)",      # KHP 2
    ]

    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue

    return 0


def mode_robust(data, bins=50):
    """Calcula moda robusta amb histograma."""
    if data is None or len(data) == 0:
        return 0.0
    counts, edges = np.histogram(np.asarray(data), bins=bins)
    i = int(np.argmax(counts))
    return 0.5 * (edges[i] + edges[i + 1])


def t_at_max(t, y):
    """Retorna el temps al màxim del senyal."""
    if t is None or y is None or len(t) == 0 or len(y) == 0:
        return None
    idx = int(np.argmax(y))
    return float(t[idx])


# =============================================================================
# FUNCIONS DE BASELINE I ESTADÍSTIQUES
# =============================================================================

def baseline_stats_time(t, y, t_start=0, t_end=2.0, fallback_pct=10):
    """
    Calcula estadístiques de baseline en una finestra temporal.

    Args:
        t: Array de temps
        y: Array de senyal
        t_start: Temps inicial de la finestra
        t_end: Temps final de la finestra
        fallback_pct: Percentatge de dades a usar si la finestra és buida

    Returns:
        Dict amb mean, std, min, max
    """
    mask = (t >= t_start) & (t <= t_end)

    if np.sum(mask) < 10:
        # Fallback: usar primers X% de dades
        n_points = max(10, int(len(y) * fallback_pct / 100))
        baseline_data = y[:n_points]
    else:
        baseline_data = y[mask]

    return {
        "mean": float(np.mean(baseline_data)),
        "std": float(np.std(baseline_data)),
        "min": float(np.min(baseline_data)),
        "max": float(np.max(baseline_data)),
    }


# NOTA: detect_main_peak i detect_all_peaks ara estan a hpsec_core.py
# (Single Source of Truth per evitar duplicació)


# =============================================================================
# QUALITAT DEL PIC
# =============================================================================

def calculate_peak_symmetry(t, y, peak_idx, left_idx, right_idx):
    """
    Calcula la simetria (tailing factor) d'un pic.

    Returns:
        Factor de simetria (1.0 = simètric, >1 = tailing, <1 = fronting)
    """
    if peak_idx <= left_idx or peak_idx >= right_idx:
        return 1.0

    t_peak = t[peak_idx]

    # Amplada esquerra i dreta al 10% de l'altura
    h_peak = y[peak_idx]
    h_10 = h_peak * 0.1

    # Trobar punts al 10% d'altura
    left_10_idx = peak_idx
    for i in range(peak_idx, left_idx - 1, -1):
        if y[i] <= h_10:
            left_10_idx = i
            break

    right_10_idx = peak_idx
    for i in range(peak_idx, right_idx + 1):
        if y[i] <= h_10:
            right_10_idx = i
            break

    w_left = t_peak - t[left_10_idx]
    w_right = t[right_10_idx] - t_peak

    if w_left <= 0:
        return 1.0

    return float(w_right / w_left)


def calculate_peak_snr(y, peak_idx, baseline_mean, baseline_std):
    """
    Calcula el Signal-to-Noise Ratio d'un pic.
    """
    if baseline_std <= 0:
        baseline_std = 0.001

    signal = y[peak_idx] - baseline_mean
    return float(signal / baseline_std)


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
# COMPARACIÓ HISTÒRICA KHP
# =============================================================================

def get_historical_khp_stats(seq_path, mode="COLUMN", conc_ppm=None, volume_uL=None, n_recent=10, exclude_outliers=True):
    """
    Obté estadístiques de les calibracions KHP històriques.

    Filtra per mode, concentració i volum d'injecció per comparar "pomes amb pomes".

    Args:
        seq_path: Path de la SEQ actual (per trobar KHP_History.json)
        mode: "COLUMN" o "BP"
        conc_ppm: Concentració KHP en ppm (ex: 2 per KHP2). REQUERIT per comparació vàlida.
        volume_uL: Volum d'injecció en µL (ex: 100, 400). REQUERIT per comparació vàlida.
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
            'calibrations': list  # Les calibracions usades
        }
    """
    history = load_khp_history(seq_path)
    if not history:
        return None

    # Filtrar per mode, concentració, volum i excloure outliers si cal
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
        'calibrations': recent_cals
    }


def compare_khp_historical(current_area, current_concentration_ratio, seq_path, mode="COLUMN", conc_ppm=None, volume_uL=None):
    """
    Compara el KHP actual amb l'històric.

    IMPORTANT: Filtra per concentració i volum per comparar correctament.
    No es pot comparar KHP2 amb KHP5, ni 100µL amb 400µL.

    Args:
        current_area: Àrea del pic principal del KHP actual
        current_concentration_ratio: Ratio àrea_pic_principal / àrea_total
        seq_path: Path de la SEQ
        mode: "COLUMN" o "BP"
        conc_ppm: Concentració KHP (ex: 2 per KHP2)
        volume_uL: Volum d'injecció en µL

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
    stats = get_historical_khp_stats(seq_path, mode, conc_ppm=conc_ppm, volume_uL=volume_uL)

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
    if seq_path:
        mode = "BP" if khp_data.get('is_bp', False) else "COLUMN"
        conc_ppm = khp_data.get('conc_ppm', None)
        volume_uL = khp_data.get('volume_uL', None)

        historical_comparison = compare_khp_historical(
            current_area=khp_data.get('area', 0),
            current_concentration_ratio=concentration_ratio,
            seq_path=seq_path,
            mode=mode,
            conc_ppm=conc_ppm,
            volume_uL=volume_uL
        )

        if historical_comparison['status'] == 'INVALID':
            for issue in historical_comparison['issues']:
                issues.append(f"HISTORICAL: {issue}")
            quality_score += 100
        elif historical_comparison['status'] == 'WARNING':
            for warn in historical_comparison['warnings']:
                warnings.append(f"HISTORICAL: {warn}")
            quality_score += 20

    # Determinar validesa
    is_valid = quality_score < 100 and len(issues) == 0

    return {
        'is_valid': is_valid,
        'issues': issues,
        'warnings': warnings,
        'quality_score': quality_score,
        'concentration_ratio': concentration_ratio,
        'historical_comparison': historical_comparison
    }


def validate_khp_for_alignment(t_doc, y_doc, t_dad, y_a254, t_uib=None, y_uib=None,
                               method="COLUMN", repair_batman=True,
                               seq_path=None, conc_ppm=None, volume_uL=None):
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

    # Trobar pics
    idx_max_doc = np.argmax(y_doc)
    idx_max_a254 = np.argmax(y_a254)
    t_max_doc = t_doc[idx_max_doc]
    t_max_a254 = t_dad[idx_max_a254]

    result["metrics"]["t_max_doc"] = float(t_max_doc)
    result["metrics"]["t_max_a254"] = float(t_max_a254)
    result["metrics"]["intensity_doc"] = float(np.max(y_doc))
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

    # Àrea DOC
    mask_doc = (t_doc >= t_start) & (t_doc <= t_end)
    if np.sum(mask_doc) > 5:
        baseline_doc = np.percentile(y_doc[mask_doc], 5)
        y_doc_corr = y_doc[mask_doc] - baseline_doc
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
            # Usar mateixa lògica que a la secció 2 (ratio)
            area_doc = result["metrics"].get("area_doc", 0)
            area_total = np.trapezoid(np.maximum(y_doc - np.percentile(y_doc, 5), 0), t_doc) if len(y_doc) > 5 else 0
            concentration_ratio = area_doc / area_total if area_total > 0 else 0

            historical = compare_khp_historical(
                current_area=area_doc,
                current_concentration_ratio=concentration_ratio,
                seq_path=seq_path,
                mode=method,
                conc_ppm=conc_ppm,
                volume_uL=volume_uL
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


# =============================================================================
# GESTIÓ CALDATA (LOCAL)
# =============================================================================

def get_caldata_path(seq_path):
    """Retorna el path de la carpeta CALDATA d'una SEQ."""
    if not seq_path:
        return None
    return os.path.join(seq_path, CALDATA_FOLDER)


def ensure_caldata_folder(seq_path):
    """Crea la carpeta CALDATA si no existeix."""
    caldata_path = get_caldata_path(seq_path)
    if caldata_path and not os.path.exists(caldata_path):
        os.makedirs(caldata_path, exist_ok=True)
    return caldata_path


def load_local_calibrations(seq_path):
    """
    Carrega l'historial LOCAL de calibracions d'una SEQ (CALDATA).
    """
    caldata_path = get_caldata_path(seq_path)
    if not caldata_path:
        return []

    filepath = os.path.join(caldata_path, CALDATA_FILENAME)
    if not os.path.exists(filepath):
        return []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get("calibrations", [])
    except Exception as e:
        print(f"Error carregant CALDATA: {e}")
        return []


def save_local_calibrations(seq_path, calibrations):
    """
    Guarda l'historial LOCAL de calibracions d'una SEQ (CALDATA).
    """
    caldata_path = ensure_caldata_folder(seq_path)
    if not caldata_path:
        return False

    filepath = os.path.join(caldata_path, CALDATA_FILENAME)

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
        print(f"Error guardant CALDATA: {e}")
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

def get_history_path(seq_path):
    """
    Retorna el path del fitxer d'històric KHP.
    Cerca a la carpeta pare (carpeta compartida) on estan totes les SEQ.
    """
    if not seq_path:
        return None
    parent_folder = os.path.dirname(seq_path)
    return os.path.join(parent_folder, KHP_HISTORY_FILENAME)


def load_khp_history(seq_path):
    """
    Carrega l'històric de calibracions KHP.
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


# =============================================================================
# CERCA DE KHP
# =============================================================================

def find_khp_in_folder(folder):
    """Cerca fitxers KHP en una carpeta."""
    if not os.path.exists(folder):
        return []
    files = glob.glob(os.path.join(folder, "*KHP*.xlsx"))
    files = [f for f in files if not os.path.basename(f).startswith("~$")]
    return files


def buscar_khp_consolidados(input_folder, allow_manual=True, gui_callbacks=None):
    """
    Cerca KHP en tres fases: LOCAL → SIBLINGS → MANUAL

    Args:
        input_folder: Carpeta SEQ on buscar
        allow_manual: Si True, permet selecció manual si no es troba
        gui_callbacks: Dict amb funcions GUI (messagebox, filedialog)

    Returns:
        (llista_fitxers, origen)
    """
    # FASE 1: LOCAL (Resultats_Consolidats o Consolidat)
    res_cons = os.path.join(input_folder, "Resultats_Consolidats")
    khp_files = find_khp_in_folder(res_cons)

    if khp_files:
        return khp_files, "LOCAL"

    # Fallback a carpeta antiga "Consolidat"
    consolidat_old = os.path.join(input_folder, "Consolidat")
    khp_files = find_khp_in_folder(consolidat_old)

    if khp_files:
        return khp_files, "LOCAL"

    # FASE 2: SIBLINGS (carpetes germanes amb mateix prefix numèric)
    folder_name = os.path.basename(input_folder)
    parent_dir = os.path.dirname(input_folder)

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
                    return khp_files, f"SIBLING:{sib}"
        except Exception:
            pass

    # FASE 3: MANUAL (si permès i hi ha GUI)
    if allow_manual and gui_callbacks:
        messagebox = gui_callbacks.get('messagebox')
        filedialog = gui_callbacks.get('filedialog')

        if messagebox and filedialog:
            if messagebox.askyesno("KHP NO TROBAT",
                                   "No s'ha trobat cap KHP.\n\nVols seleccionar manualment una carpeta amb KHP?"):
                d = filedialog.askdirectory(title="Selecciona Carpeta SEQ amb KHP")
                if d:
                    manual_res = os.path.join(d, "Resultats_Consolidats")
                    if os.path.exists(manual_res):
                        khp_files = find_khp_in_folder(manual_res)
                        if khp_files:
                            return khp_files, "MANUAL"
                    khp_files = find_khp_in_folder(d)
                    if khp_files:
                        return khp_files, "MANUAL"

    return [], "NONE"


# =============================================================================
# ANÀLISI KHP
# =============================================================================

def analizar_khp_consolidado(khp_file, config=None):
    """
    Analitza un fitxer KHP consolidat amb detall complet.

    Returns:
        Dict amb conc, area, shift_min, qualitat, dades per gràfics, etc.
    """
    # Merge config amb DEFAULT_CONFIG
    config = {**DEFAULT_CONFIG, **(config or {})}
    fname = os.path.basename(khp_file)

    conc = extract_khp_conc(fname)
    if conc == 0:
        return None

    try:
        xls = pd.ExcelFile(khp_file, engine="openpyxl")

        # Llegir metadades del full ID
        doc_mode = "N/A"
        seq_date = ""
        method_from_file = None

        if "ID" in xls.sheet_names:
            df_id = pd.read_excel(xls, "ID", engine="openpyxl")
            # Compatible amb format antic (Camp/Valor) i nou (Field/Value)
            if "Field" in df_id.columns:
                id_dict = dict(zip(df_id["Field"], df_id["Value"]))
            elif "Camp" in df_id.columns:
                id_dict = dict(zip(df_id["Camp"], df_id["Valor"]))
            else:
                id_dict = {}
            doc_mode = str(id_dict.get("DOC_MODE", id_dict.get("Source", "N/A")))
            seq_date = str(id_dict.get("Date", ""))
            method_from_file = str(id_dict.get("Method", id_dict.get("MODE", ""))).upper()
            if method_from_file not in ["BP", "COLUMN", "COL"]:
                method_from_file = None

        df_doc = pd.read_excel(xls, "DOC", engine="openpyxl")
        t_doc = pd.to_numeric(df_doc["time (min)"], errors="coerce").to_numpy()

        # Buscar columna DOC
        doc_col = None
        for c in df_doc.columns:
            if 'doc' in str(c).lower() and 'raw' not in str(c).lower():
                doc_col = c
                break
        if doc_col is None:
            doc_col = df_doc.columns[1]

        doc_net = pd.to_numeric(df_doc[doc_col], errors="coerce").to_numpy()

        # Netejar NaN
        mask = np.isfinite(t_doc) & np.isfinite(doc_net)
        t_doc, doc_net = t_doc[mask], doc_net[mask]

        # Detectar TOTS els pics
        all_peaks = detect_all_peaks(t_doc, doc_net, config["peak_min_prominence_pct"])

        # Detectar pic principal
        peak_info = detect_main_peak(t_doc, doc_net, config["peak_min_prominence_pct"])

        if not peak_info['valid']:
            return None

        t_retention = peak_info.get('t_max', 0)

        # Detectar si és BP
        t_max_chromato = float(np.max(t_doc))
        t_peak = peak_info.get('t_max', 10)

        if method_from_file == "BP":
            is_bp_chromato = True
        elif method_from_file in ["COLUMN", "COL"]:
            is_bp_chromato = False
        else:
            is_bp_by_name = "BP" in fname.upper() or "_BP" in fname.upper()
            is_bp_by_chromato = t_max_chromato < 20 or t_peak < 10
            is_bp_chromato = is_bp_by_name or is_bp_by_chromato

        # Baseline stats
        if is_bp_chromato:
            bl_stats = baseline_stats_time(t_doc, doc_net, t_start=0, t_end=2.0)
        else:
            bl_stats = baseline_stats_time(t_doc, doc_net, t_start=0, t_end=10.0)

        # Límits del pic
        peak_idx = peak_info.get('peak_idx', int(np.argmax(doc_net)))
        left_idx = peak_info.get('left_idx', 0)
        right_idx = peak_info.get('right_idx', len(doc_net) - 1)

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
            t_doc, doc_net, left_idx, right_idx, peak_idx,
            baseline_threshold_pct=15,
            min_width_minutes=1.0,
            max_width_minutes=6.0 if is_bp_chromato else 10.0,
            is_bp=is_bp_chromato
        )

        left_idx = expansion['left_idx']
        right_idx = expansion['right_idx']
        limits_expanded = not expansion['original_valid']

        if limits_expanded:
            new_area = float(trapezoid(doc_net[left_idx:right_idx+1], t_doc[left_idx:right_idx+1]))
            old_area = peak_info.get('area', 0)

            peak_info['area'] = new_area
            peak_info['left_idx'] = left_idx
            peak_info['right_idx'] = right_idx
            peak_info['t_start'] = float(t_doc[left_idx])
            peak_info['t_end'] = float(t_doc[right_idx])
            peak_info['limits_expanded'] = True
            peak_info['expansion_info'] = {
                'original_left': original_left_idx,
                'original_right': original_right_idx,
                'old_area': old_area,
                'new_area': new_area,
                'area_increase_pct': ((new_area - old_area) / old_area * 100) if old_area > 0 else 0
            }

        # Simetria i SNR
        symmetry = calculate_peak_symmetry(t_doc, doc_net, peak_idx, left_idx, right_idx)
        snr = calculate_peak_snr(doc_net, peak_idx, bl_stats["mean"], bl_stats["std"])

        # Anomalies - Usar funcions de hpsec_core
        # Timeout (MILLOR MÈTODE: basat en intervals dt)
        timeout_info = detect_timeout(t_doc)
        has_timeout = timeout_info['n_timeouts'] > 0
        timeout_severity = timeout_info.get('severity', 'OK')

        # Batman/Anomalies de forma (usar segment del pic)
        t_peak_seg = t_doc[left_idx:right_idx+1]
        y_peak_seg = doc_net[left_idx:right_idx+1]
        anomaly_info = detect_peak_anomaly(t_peak_seg, y_peak_seg)
        has_batman = anomaly_info.get('is_batman', False)
        has_irregular = anomaly_info.get('is_irregular', False)
        smoothness = anomaly_info.get('smoothness', 100.0)

        # DAD 254nm
        shift_khp = 0.0
        has_dad = "DAD" in xls.sheet_names
        t_dad = None
        dad_254 = None
        t_dad_max = None
        dad_peak_info = None
        a254_area = 0.0
        a254_doc_ratio = 0.0

        if has_dad:
            df_dad = pd.read_excel(xls, "DAD", engine="openpyxl")
            if "254" in df_dad.columns:
                t_dad = pd.to_numeric(df_dad["time (min)"], errors="coerce").to_numpy()
                dad_254 = pd.to_numeric(df_dad["254"], errors="coerce").to_numpy()

                dad_mask = np.isfinite(t_dad) & np.isfinite(dad_254)
                t_dad, dad_254 = t_dad[dad_mask], dad_254[dad_mask]

                t_doc_max = t_at_max(t_doc, doc_net)
                t_dad_max = t_at_max(t_dad, dad_254)

                if t_doc_max and t_dad_max:
                    shift_khp = t_dad_max - t_doc_max

                dad_peak_info = detect_main_peak(t_dad, dad_254, config["peak_min_prominence_pct"])

                if dad_peak_info and dad_peak_info.get('valid') and len(t_dad) > 10:
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

                        dad_peak_info['a254_area'] = a254_area
                        dad_peak_info['a254_left_idx'] = dad_left_idx
                        dad_peak_info['a254_right_idx'] = dad_right_idx

        # Calcular àrea total i concentration ratio
        # (QC important: quina fracció del DOC total està al pic principal)
        # IMPORTANT: Usar integració sobre baseline per AMBDUES àrees (consistència)
        baseline_mean = bl_stats.get('mean', 0)
        baseline_std = bl_stats.get('std', 0.1)

        # Àrea pic principal = integració sobre baseline dins els límits del pic
        # (NO usar peak_info['area'] que és sense correcció)
        t_peak = t_doc[left_idx:right_idx+1]
        y_peak = doc_net[left_idx:right_idx+1]
        peak_integration = integrate_above_baseline(
            t_peak, y_peak,
            baseline_mean=baseline_mean,
            baseline_std=baseline_std,
            threshold_sigma=3.0
        )
        area_main_peak = peak_integration['area']

        # Àrea total = integració sobre baseline de TOT el cromatograma
        total_integration = integrate_above_baseline(
            t_doc, doc_net,
            baseline_mean=baseline_mean,
            baseline_std=baseline_std,
            threshold_sigma=3.0
        )
        area_total = total_integration['area']

        # Concentration ratio: pic principal vs total (ambdós baseline-corregits)
        concentration_ratio = area_main_peak / area_total if area_total > 0 else 1.0

        # Calcular volum d'injecció (necessari per comparació històrica)
        # Derivar seq_path del khp_file (puja 2 nivells: Resultats_Consolidats -> SEQ)
        seq_path_derived = os.path.dirname(os.path.dirname(khp_file))
        volume_uL = get_injection_volume(seq_path_derived, is_bp_chromato)

        # Qualitat - Sistema millorat amb severitats
        quality_score = 0
        quality_issues = []

        if symmetry > 1.5:
            quality_score += 10
            quality_issues.append(f"Simetria alta ({symmetry:.2f})")
        if symmetry < 0.8:
            quality_score += 10
            quality_issues.append(f"Simetria baixa ({symmetry:.2f})")
        if snr < 10:
            quality_score += 20
            quality_issues.append(f"SNR baix ({snr:.1f})")

        # Batman/Irregular (de hpsec_core.detect_peak_anomaly)
        if has_batman:
            quality_score += 50
            quality_issues.append("Doble pic (Batman)")
        if has_irregular:
            quality_score += 30
            quality_issues.append(f"Pic irregular (smoothness={smoothness:.0f}%)")

        # Timeout amb severitat (de hpsec_core.detect_timeout)
        if has_timeout:
            if timeout_severity == 'CRITICAL':
                quality_score += 150
                quality_issues.append("TIMEOUT CRÍTIC en zona HS")
            elif timeout_severity == 'WARNING':
                quality_score += 100
                quality_issues.append(f"TimeOUT en zona crítica ({timeout_info.get('zone_summary', {})})")
            else:
                quality_score += 20
                quality_issues.append("TimeOUT en zona segura")

        if len(all_peaks) > 3:
            quality_score += 5 * (len(all_peaks) - 3)
            quality_issues.append(f"Múltiples pics ({len(all_peaks)})")

        if limits_expanded:
            exp_info = peak_info.get('expansion_info', {})
            area_inc = exp_info.get('area_increase_pct', 0)
            quality_issues.append(f"Límits expandits (+{area_inc:.0f}% àrea)")

        return {
            'filename': fname,
            'filepath': khp_file,
            'conc_ppm': conc,
            'area': peak_info['area'],
            'shift_min': shift_khp,
            'shift_sec': shift_khp * 60,
            'peak_info': peak_info,
            'has_dad': has_dad,
            't_doc_max': t_at_max(t_doc, doc_net),
            't_dad_max': t_dad_max,
            't_doc': t_doc,
            'y_doc': doc_net,
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
            'doc_mode': doc_mode,
            'seq_date': seq_date,
            't_retention': t_retention,
            'baseline_valid': True,
            'limits_expanded': limits_expanded,
            'expansion_info': expansion,
            'a254_area': a254_area,
            'a254_doc_ratio': a254_doc_ratio,
            'height': float(doc_net[peak_idx]),  # Afegit per validate_khp_quality
            'area_total': area_total,
            'area_main_peak': area_main_peak,
            'concentration_ratio': concentration_ratio,
            'volume_uL': volume_uL,
        }

    except Exception as e:
        print(f"Error analitzant KHP {fname}: {e}")
        import traceback
        traceback.print_exc()
        return None


def analizar_khp_lote(khp_files, config=None):
    """
    Analitza un lot de fitxers KHP i selecciona el millor.

    Returns:
        Dict amb dades detallades + selecció final
    """
    all_data = []
    for f in khp_files:
        d = analizar_khp_consolidado(f, config)
        if d:
            all_data.append(d)

    if not all_data:
        return None

    # Agrupar per concentració
    by_conc = {}
    for d in all_data:
        conc = d['conc_ppm']
        if conc not in by_conc:
            by_conc[conc] = []
        by_conc[conc].append(d)

    # Seleccionar concentració més alta
    selected_conc = max(by_conc.keys())
    replicas = by_conc[selected_conc]

    # Estadístiques
    areas = [r['area'] for r in replicas]
    shifts = [r['shift_min'] for r in replicas]
    symmetries = [r.get('symmetry', 1.0) for r in replicas]
    snrs = [r.get('snr', 0) for r in replicas]
    quality_scores = [r.get('quality_score', 0) for r in replicas]

    mean_area = float(np.mean(areas))
    std_area = float(np.std(areas))
    mean_shift = float(np.mean(shifts))
    std_shift = float(np.std(shifts))
    rsd = float((std_area / mean_area) * 100.0) if mean_area > 0 else 100.0

    stats = {
        'n_replicas': len(replicas),
        'mean_area': mean_area,
        'std_area': std_area,
        'rsd_area': rsd,
        'mean_shift': mean_shift,
        'std_shift': std_shift,
        'mean_symmetry': float(np.mean(symmetries)),
        'mean_snr': float(np.mean(snrs)),
        'min_quality_score': min(quality_scores),
        'concentrations': list(by_conc.keys()),
    }

    if len(replicas) == 1:
        result = replicas[0].copy()
        result['status'] = "Única rèplica"
        result['n_replicas'] = 1
        result['all_khp_data'] = all_data
        result['stats'] = stats
        result['selected_idx'] = 0
        return result

    if rsd < 10.0:
        # RSD acceptable: usar promig
        result = {
            'filename': f"KHP{selected_conc}_PROMIG",
            'filepath': replicas[0]['filepath'],
            'conc_ppm': selected_conc,
            'area': mean_area,
            'shift_min': mean_shift,
            'shift_sec': mean_shift * 60,
            'status': f"Promig {len(replicas)} rèpliques (RSD {rsd:.1f}%)",
            'has_dad': replicas[0]['has_dad'],
            'n_replicas': len(replicas),
            'rsd': rsd,
            'all_khp_data': all_data,
            'replicas': replicas,
            'stats': stats,
            'is_average': True,
            'selected_idx': -1,
            't_doc': replicas[0].get('t_doc'),
            'y_doc': replicas[0].get('y_doc'),
            't_dad': replicas[0].get('t_dad'),
            'y_dad_254': replicas[0].get('y_dad_254'),
            'peak_info': replicas[0].get('peak_info'),
            'symmetry': stats['mean_symmetry'],
            'snr': stats['mean_snr'],
            'doc_mode': replicas[0].get('doc_mode', 'N/A'),
            'seq_date': replicas[0].get('seq_date', ''),
            't_retention': float(np.mean([r.get('t_retention', r.get('t_doc_max', 0)) for r in replicas])),
            't_doc_max': replicas[0].get('t_doc_max', 0),
            'is_bp': replicas[0].get('is_bp', False),
            'peak_left_idx': replicas[0].get('peak_left_idx', 0),
            'peak_right_idx': replicas[0].get('peak_right_idx', 0),
            'limits_expanded': replicas[0].get('limits_expanded', False),
            'quality_issues': replicas[0].get('quality_issues', []),
            'quality_score': stats['min_quality_score'],
            'baseline_valid': replicas[0].get('baseline_valid', True),
            'baseline_stats': replicas[0].get('baseline_stats', {}),
            'a254_area': float(np.mean([r.get('a254_area', 0) for r in replicas if r.get('a254_area', 0) > 0])) if any(r.get('a254_area', 0) > 0 for r in replicas) else 0,
            'a254_doc_ratio': float(np.mean([r.get('a254_doc_ratio', 0) for r in replicas if r.get('a254_doc_ratio', 0) > 0])) if any(r.get('a254_doc_ratio', 0) > 0 for r in replicas) else 0,
            'dad_peak_info': replicas[0].get('dad_peak_info'),
        }
        return result
    else:
        # RSD alt: seleccionar millor per qualitat
        sorted_replicas = sorted(replicas, key=lambda x: (x.get('quality_score', 0), -x['area']))
        best = sorted_replicas[0]
        best_idx = replicas.index(best)

        result = best.copy()
        result['status'] = f"Millor qualitat (RSD {rsd:.1f}% > 10%)"
        result['n_replicas'] = len(replicas)
        result['rsd'] = rsd
        result['all_khp_data'] = all_data
        result['replicas'] = replicas
        result['stats'] = stats
        result['is_average'] = False
        result['selected_idx'] = best_idx
        return result


# =============================================================================
# REGISTRE DE CALIBRACIONS
# =============================================================================

def _extract_replicas_info(khp_data):
    """
    Extreu informació resumida de cada replicat per guardar a CALDATA.
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


def register_calibration(seq_path, khp_data, khp_source, mode="COLUMN"):
    """
    Registra una nova calibració a l'històric.

    Guarda a DOS llocs:
    1. LOCAL (CALDATA/calibrations.json) - Historial complet de la SEQ
    2. GLOBAL (KHP_History.json) - Una entrada per SEQ per comparacions
    """
    seq_name = os.path.basename(seq_path)

    # Calcular factor de calibració
    area = khp_data.get('area', 0)
    conc = khp_data.get('conc_ppm', 0)
    factor = conc / area if area > 0 else 0

    seq_date = khp_data.get('seq_date', '')
    if not seq_date:
        seq_date = datetime.now().isoformat()

    is_bp = khp_data.get('is_bp', False)
    volume = get_injection_volume(seq_path, is_bp)

    peak_info = khp_data.get('peak_info', {})

    entry = {
        "cal_id": generate_calibration_id(),
        "seq_name": seq_name,
        "seq_path": seq_path,
        "date": seq_date,
        "seq_date": seq_date,
        "date_processed": datetime.now().isoformat(),
        "mode": mode,
        "khp_file": khp_data.get('filename', 'N/A'),
        "khp_source": khp_source,
        "doc_mode": khp_data.get('doc_mode', 'N/A'),
        "conc_ppm": conc,
        "volume_uL": volume,
        "area": area,
        "factor": factor,
        "shift_min": khp_data.get('shift_min', 0),
        "shift_sec": khp_data.get('shift_sec', khp_data.get('shift_min', 0) * 60),
        "symmetry": khp_data.get('symmetry', 1.0),
        "snr": khp_data.get('snr', 0),
        "all_peaks_count": khp_data.get('all_peaks_count', 1),
        "has_batman": khp_data.get('has_batman', False),
        "has_timeout": khp_data.get('has_timeout', False),
        "n_replicas": khp_data.get('n_replicas', 1),
        "rsd": khp_data.get('rsd', 0),
        "status": khp_data.get('status', 'OK'),
        "quality_issues": khp_data.get('quality_issues', []),
        "is_bp": is_bp,
        "is_outlier": False,
        "is_active": True,
        "baseline_valid": khp_data.get('baseline_valid', True),
        "t_retention": khp_data.get('t_retention', khp_data.get('t_doc_max', 0)),
        "limits_expanded": khp_data.get('limits_expanded', False),
        "t_start": peak_info.get('t_start', 0),
        "t_end": peak_info.get('t_end', 0),
        "peak_left_idx": khp_data.get('peak_left_idx', peak_info.get('left_idx', 0)),
        "peak_right_idx": khp_data.get('peak_right_idx', peak_info.get('right_idx', 0)),
        "baseline": khp_data.get('baseline_stats', {}).get('mean', 0),
        "baseline_std": khp_data.get('baseline_stats', {}).get('std', 0),
        "replicas_info": _extract_replicas_info(khp_data),
        "a254_area": khp_data.get('a254_area', 0),
        "a254_doc_ratio": khp_data.get('a254_doc_ratio', 0),
        "area_total": khp_data.get('area_total', 0),
        "area_main_peak": khp_data.get('area_main_peak', khp_data.get('area', 0)),
        "concentration_ratio": khp_data.get('concentration_ratio', 1.0),
    }

    # 1. GUARDAR A LOCAL (CALDATA)
    local_cals = load_local_calibrations(seq_path)

    for cal in local_cals:
        if cal.get("mode") == mode:
            cal["is_active"] = False

    local_cals.insert(0, entry)
    save_local_calibrations(seq_path, local_cals)

    # 2. GUARDAR A GLOBAL (KHP_History.json)
    global_cals = load_khp_history(seq_path)

    updated = False
    for i, cal in enumerate(global_cals):
        if cal.get("seq_name") == seq_name and cal.get("mode") == mode:
            global_cals[i] = entry
            updated = True
            break

    if not updated:
        global_cals.append(entry)

    global_cals.sort(key=lambda x: x.get("date", ""), reverse=True)
    save_khp_history(seq_path, global_cals)

    return entry


def mark_calibration_as_outlier(seq_path, seq_name, mode="COLUMN", is_outlier=True, cal_id=None):
    """
    Marca/desmarca una calibració com a outlier.
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


# =============================================================================
# FUNCIÓ PRINCIPAL STANDALONE
# =============================================================================

def calibrate_sequence(seq_path, config=None, progress_callback=None, gui_callbacks=None):
    """
    Funció principal de calibració standalone.

    Args:
        seq_path: Path de la seqüència a calibrar
        config: Configuració (DEFAULT_CONFIG si None)
        progress_callback: Funció per reportar progrés (pct, msg)
        gui_callbacks: Dict amb messagebox i filedialog per interacció GUI

    Returns:
        Dict amb:
        - success: bool
        - khp_data: dades del KHP analitzat
        - khp_source: origen del KHP (LOCAL, SIBLING, MANUAL)
        - calibration: entrada de calibració registrada
        - errors: llista d'errors
    """
    # Merge config amb DEFAULT_CONFIG (default té prioritat per claus que falten)
    merged_config = {**DEFAULT_CONFIG, **(config or {})}
    config = merged_config

    result = {
        "success": False,
        "khp_data": None,
        "khp_source": None,
        "calibration": None,
        "errors": [],
    }

    if progress_callback:
        progress_callback(10, "Cercant KHP...")

    # Cercar KHP
    khp_files, khp_source = buscar_khp_consolidados(
        seq_path,
        allow_manual=(gui_callbacks is not None),
        gui_callbacks=gui_callbacks
    )

    if not khp_files:
        result["errors"].append("No s'ha trobat cap fitxer KHP")
        return result

    result["khp_source"] = khp_source

    if progress_callback:
        progress_callback(30, f"Analitzant {len(khp_files)} KHP...")

    # Analitzar KHP
    khp_data = analizar_khp_lote(khp_files, config)

    if not khp_data:
        result["errors"].append("No s'ha pogut analitzar cap KHP")
        return result

    result["khp_data"] = khp_data

    if progress_callback:
        progress_callback(70, "Registrant calibració...")

    # Determinar mode
    mode = "BP" if khp_data.get('is_bp', False) else "COLUMN"

    # Registrar calibració
    calibration = register_calibration(seq_path, khp_data, khp_source, mode)
    result["calibration"] = calibration

    if progress_callback:
        progress_callback(100, "Calibració completada")

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
        print(f"  Factor: {cal['factor']:.6f}")
        print(f"  Mode: {cal['mode']}")
        print(f"  Shift: {cal['shift_sec']:.1f} s")
    else:
        print(f"ERROR: {result['errors']}")
