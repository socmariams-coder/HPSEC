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
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.signal import savgol_filter
from scipy.integrate import trapezoid
from hpsec_config import get_registry_path, get_config

logger = logging.getLogger(__name__)

# Import funcions de detecció des de hpsec_core (Single Source of Truth)
from hpsec_core import (
    detect_timeout,
    detect_irregular_top,
    detect_peak_anomaly,
    calc_top_smoothness,
    detect_main_peak,
    detect_all_peaks,
    integrate_chromatogram,
    integrate_above_baseline,
    calculate_fwhm,
    calculate_symmetry,
    fit_bigaussian,
    repair_with_parabola,
    TIMEOUT_CONFIG
)

# Import funcions d'identificació des de hpsec_import (Single Source of Truth)
from hpsec_import import is_khp, extract_khp_conc, obtenir_seq

# Import funcions utilitàries
# NOTA (2026-02-03): Baseline functions ara a hpsec_core.py
from hpsec_core import mode_robust, get_baseline_value, get_baseline_stats
from hpsec_utils import t_at_max

# Import sistema d'avisos estructurats
from hpsec_warnings import (
    create_warning, get_max_warning_level, WarningLevel,
    migrate_warnings_list
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

# Fitxers d'historial GLOBAL (a REGISTRY/)
REGISTRY_FOLDER = "REGISTRY"
KHP_HISTORY_FILENAME = "KHP_History.json"
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
    "irregular_top_max_sep_min": 0.5,   # formerly batman_max_sep_min
    "irregular_top_min_height_pct": 15.0,  # formerly batman_min_height_pct
    "irregular_top_min_sigma": 3.0,    # formerly batman_min_sigma
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
# CALIBRACIÓ DE REFERÈNCIA GLOBAL (VERSIONADA)
# =============================================================================
#
# Sistema de calibració global amb:
# - Calibracions versionades (valid_from / valid_to)
# - rf_mass_cal separat per senyal (direct/uib) i mode (column/bp)
# - QC check del KHP local vs calibració global
# - Batch reprocessing usa calibració vigent per data SEQ
#
# Estructura rf_mass_cal:
#   {
#     "direct": {"column": 682, "bp": 682},
#     "uib": {"column": 682, "bp": 682}
#   }
#
# =============================================================================

CALIBRATION_REFERENCE_FILENAME = "Calibration_Reference.json"
QC_HISTORY_FILENAME = "QC_History.json"


def get_calibration_reference_path():
    """Retorna el path al fitxer de calibració de referència global."""
    registry = get_registry_path()
    if registry:
        return os.path.join(registry, CALIBRATION_REFERENCE_FILENAME)
    return None


def get_qc_history_path():
    """Retorna el path al fitxer d'historial QC."""
    registry = get_registry_path()
    if registry:
        return os.path.join(registry, QC_HISTORY_FILENAME)
    return None


def load_qc_history():
    """
    Carrega l'historial QC.

    Returns:
        list d'entrades QC o [] si no existeix
    """
    qc_path = get_qc_history_path()
    if not qc_path or not os.path.exists(qc_path):
        return []

    try:
        with open(qc_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('entries', [])
    except Exception as e:
        logger.error(f"Error carregant QC History: {e}")
        return []


def load_calibration_reference():
    """
    Carrega la calibració de referència global.

    Returns:
        dict amb les dades de calibració o None si no existeix
    """
    ref_path = get_calibration_reference_path()
    if not ref_path or not os.path.exists(ref_path):
        return None

    try:
        with open(ref_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error carregant calibració de referència: {e}")
        return None


def save_calibration_reference(data):
    """
    Guarda la calibració de referència global.

    Args:
        data: dict amb les dades de calibració

    Returns:
        bool indicant èxit
    """
    ref_path = get_calibration_reference_path()
    if not ref_path:
        return False

    try:
        data['updated'] = datetime.now().strftime('%Y-%m-%d')
        with open(ref_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"Error guardant calibració de referència: {e}")
        return False


def get_calibration_for_date(seq_date):
    """
    Retorna la calibració vigent per una data donada.

    Args:
        seq_date: Data de la SEQ (YYYY-MM-DD string o datetime)

    Returns:
        dict amb la calibració vigent o None
    """
    ref = load_calibration_reference()
    if not ref or 'calibrations' not in ref:
        return None

    # Normalitzar data a string YYYY-MM-DD
    if hasattr(seq_date, 'strftime'):
        seq_date_str = seq_date.strftime('%Y-%m-%d')
    else:
        seq_date_str = str(seq_date)[:10]

    # Buscar calibració vigent per aquesta data
    for cal in ref['calibrations']:
        valid_from = cal.get('valid_from', '1900-01-01')
        valid_to = cal.get('valid_to')

        if seq_date_str >= valid_from:
            if valid_to is None or seq_date_str <= valid_to:
                return cal

    return None


def get_active_global_calibration():
    """
    Retorna la calibració global activa actual (Calibration_Reference.json).

    Nota: Diferent de get_active_calibration() que obté calibracions locals per SEQ.

    Returns:
        dict amb la calibració activa o None
    """
    ref = load_calibration_reference()
    if not ref:
        return None

    active_id = ref.get('active_calibration_id')
    if active_id:
        for cal in ref.get('calibrations', []):
            if cal.get('id') == active_id:
                return cal

    # Fallback: primera calibració activa
    for cal in ref.get('calibrations', []):
        if cal.get('is_active', False):
            return cal

    return None


def get_rf_mass_cal(signal='direct', mode='column', seq_date=None):
    """
    Obté el RF_mass de calibració global per senyal i mode.

    Args:
        signal: 'direct' o 'uib'
        mode: 'column' o 'bp'
        seq_date: Data SEQ per seleccionar calibració (None = activa)

    Returns:
        float: rf_mass_cal o None si no està definit
    """
    if seq_date:
        cal = get_calibration_for_date(seq_date)
    else:
        cal = get_active_global_calibration()

    if not cal:
        return None

    rf_mass_cal = cal.get('rf_mass_cal', {})

    # Estructura: {"direct": {"column": X, "bp": Y}, "uib": {...}}
    signal_data = rf_mass_cal.get(signal.lower(), {})
    if isinstance(signal_data, dict):
        return signal_data.get(mode.lower())

    # Fallback: valor simple (compatibilitat)
    if isinstance(rf_mass_cal, (int, float)):
        return rf_mass_cal

    return None


def get_calibration_intercept(signal='direct', mode='column', seq_date=None):
    """
    Obté l'intercept de la calibració per signal/mode (0 si forçada a origen).

    Suporta intercept escalar (retrocompatible) o dict nested per-mode:
        {"direct": {"column": 81, "bp": 0}, "uib": {"column": 0, "bp": 0}}

    Args:
        signal: 'direct' o 'uib'
        mode: 'column' o 'bp'
        seq_date: Data SEQ (None = calibració activa)

    Returns:
        float: intercept (0 si model origin o no trobat)
    """
    if seq_date:
        cal = get_calibration_for_date(seq_date)
    else:
        cal = get_active_global_calibration()

    if not cal:
        return 0

    intercept = cal.get('intercept', 0)
    if isinstance(intercept, dict):
        signal_data = intercept.get(signal.lower(), {})
        if isinstance(signal_data, dict):
            return signal_data.get(mode.lower(), 0)
        return 0
    return float(intercept) if intercept else 0


def compute_calibration_fingerprint(calibration=None):
    """
    SHA-256[:16] dels paràmetres de calibració que afecten la quantificació.

    Permet detectar si una anàlisi ha quedat obsoleta després d'un canvi de calibració.
    Segueix el patró de compute_config_fingerprint() de hpsec_config.py.

    Args:
        calibration: Dict de calibració (None = activa)

    Returns:
        str: Hash hex de 16 caràcters, o "" si no hi ha calibració
    """
    import hashlib
    if calibration is None:
        calibration = get_active_global_calibration()
    if not calibration:
        return ""
    # Camps que afecten la quantificació: rf_mass_cal, intercept, id
    data = {
        "rf_mass_cal": calibration.get("rf_mass_cal"),
        "intercept": calibration.get("intercept"),
        "id": calibration.get("id"),
    }
    raw = json.dumps(data, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]


def quantify_with_global_calibration(area, volume_uL, signal='direct', mode='column', seq_date=None):
    """
    Quantifica una mostra usant rf_mass_cal global (Calibration_Reference.json).

    Nota: Aquesta funció és la recomanada per quantificació estàndard.
    Usa rf_mass_cal global (Calibration_Reference.json) amb intercept.

    Fórmules segons model:
    - origin:    ppm = Area × 1000 / (rf_mass_cal × volume_uL)
    - intercept: ppm = (Area - intercept) × 1000 / (rf_mass_cal × volume_uL)

    Args:
        area: Àrea del pic (mAU·min)
        volume_uL: Volum d'injecció (µL)
        signal: 'direct' o 'uib'
        mode: 'column' o 'bp'
        seq_date: Data SEQ per seleccionar calibració correcta

    Returns:
        dict amb:
            - concentration_ppm: concentració calculada
            - rf_mass_cal_used: valor rf_mass_cal utilitzat
            - calibration_id: ID de la calibració usada
            - model: 'origin' o 'intercept'
            - intercept: valor intercept (0 si origin)
            - success: bool
    """
    if seq_date:
        cal = get_calibration_for_date(seq_date)
    else:
        cal = get_active_global_calibration()

    if not cal:
        return {
            'success': False,
            'concentration_ppm': None,
            'rf_mass_cal_used': None,
            'calibration_id': None,
            'error': 'No hi ha calibració disponible'
        }

    rf_mass_cal = get_rf_mass_cal(signal, mode, seq_date)
    if rf_mass_cal is None or rf_mass_cal <= 0:
        return {
            'success': False,
            'concentration_ppm': None,
            'rf_mass_cal_used': None,
            'calibration_id': cal.get('id'),
            'error': f'No hi ha rf_mass_cal per signal={signal}, mode={mode}'
        }

    if volume_uL <= 0:
        return {
            'success': False,
            'concentration_ppm': None,
            'rf_mass_cal_used': rf_mass_cal,
            'calibration_id': cal.get('id'),
            'error': 'Volum invàlid'
        }

    # Obtenir intercept per-mode (0 si origin)
    intercept = get_calibration_intercept(signal, mode, seq_date)

    # Fórmula única: ppm = (Area - intercept) × 1000 / (rf_mass_cal × volume)
    # Si intercept = 0 (origin), queda: ppm = Area × 1000 / (rf_mass_cal × volume)
    area_corrected = max(0, area - intercept)  # No permetre àrees negatives
    concentration_ppm = area_corrected * 1000 / (rf_mass_cal * volume_uL)

    return {
        'success': True,
        'concentration_ppm': concentration_ppm,
        'rf_mass_cal_used': rf_mass_cal,
        'calibration_id': cal.get('id'),
        'intercept': intercept
    }


def validate_khp_qc(khp_data, seq_date=None, signal='direct', mode='column'):
    """
    Valida el KHP d'una SEQ com a QC check.

    Compara el rf_mass mesurat del KHP vs rf_mass_cal vigent.

    Args:
        khp_data: dict amb dades del KHP (area, conc_ppm, volume_uL, rf_mass)
        seq_date: Data de la SEQ
        signal: 'direct' o 'uib'
        mode: 'column' o 'bp'

    Returns:
        dict amb:
            - status: 'PASS', 'WARNING', 'FAIL'
            - deviation_pct: desviació percentual
            - rf_mass_measured: valor mesurat
            - rf_mass_expected: valor esperat (rf_mass_cal)
            - calibration_id: ID calibració usada
            - message: missatge descriptiu
    """
    # Llegir thresholds del config (permet ajustar sense tocar codi)
    config = get_config()
    warning_pct = config.get('calibration', 'qc_thresholds', 'warning_pct', default=5.0)
    fail_pct = config.get('calibration', 'qc_thresholds', 'fail_pct', default=10.0)

    rf_mass_cal = get_rf_mass_cal(signal, mode, seq_date)
    if rf_mass_cal is None:
        return {
            'status': 'UNKNOWN',
            'deviation_pct': None,
            'rf_mass_measured': None,
            'rf_mass_expected': None,
            'calibration_id': None,
            'message': 'No hi ha calibració de referència'
        }

    cal = get_calibration_for_date(seq_date) if seq_date else get_active_global_calibration()
    cal_id = cal.get('id') if cal else None

    # Obtenir rf_mass mesurat del KHP
    rf_mass_measured = khp_data.get('rf_mass', 0)
    if rf_mass_measured <= 0:
        # Calcular si no està disponible
        area = khp_data.get('area', 0)
        conc = khp_data.get('conc_ppm', 0)
        volume = khp_data.get('volume_uL', 0)
        if area > 0 and conc > 0 and volume > 0:
            rf_mass_measured = area * 1000 / (conc * volume)

    if rf_mass_measured <= 0:
        return {
            'status': 'FAIL',
            'deviation_pct': None,
            'rf_mass_measured': None,
            'rf_mass_expected': rf_mass_cal,
            'calibration_id': cal_id,
            'message': 'KHP sense rf_mass vàlid'
        }

    deviation_pct = abs(rf_mass_measured - rf_mass_cal) / rf_mass_cal * 100

    if deviation_pct <= warning_pct:
        status = 'PASS'
        message = f"QC PASS: RF_mass {rf_mass_measured:.0f} dins tolerància ({deviation_pct:.1f}%)"
    elif deviation_pct <= fail_pct:
        status = 'WARNING'
        message = f"QC WARNING: RF_mass {rf_mass_measured:.0f} desviació {deviation_pct:.1f}% (limit {warning_pct}%)"
    else:
        status = 'FAIL'
        message = f"QC FAIL: RF_mass {rf_mass_measured:.0f} fora rang ({deviation_pct:.1f}% > {fail_pct}%)"

    return {
        'status': status,
        'deviation_pct': deviation_pct,
        'rf_mass_measured': rf_mass_measured,
        'rf_mass_expected': rf_mass_cal,
        'calibration_id': cal_id,
        'message': message
    }


def register_qc_result(seq_name, seq_date, qc_result, khp_data):
    """
    Registra el resultat QC a QC_History.json.

    Args:
        seq_name: Nom de la SEQ
        seq_date: Data de la SEQ
        qc_result: Resultat de validate_khp_qc()
        khp_data: Dades del KHP

    Returns:
        bool indicant èxit
    """
    qc_path = get_qc_history_path()
    if not qc_path:
        return False

    # Carregar historial existent
    if os.path.exists(qc_path):
        try:
            with open(qc_path, 'r', encoding='utf-8') as f:
                history = json.load(f)
        except Exception:
            history = {'version': '1.0', 'entries': []}
    else:
        history = {'version': '1.0', 'entries': []}

    # Crear entrada
    entry = {
        'seq_name': seq_name,
        'seq_date': str(seq_date)[:10] if seq_date else None,
        'timestamp': datetime.now().isoformat(),
        'calibration_id': qc_result.get('calibration_id'),
        'khp_name': khp_data.get('name', khp_data.get('khp_name', 'KHP')),
        'khp_conc_ppm': khp_data.get('conc_ppm'),
        'volume_uL': khp_data.get('volume_uL'),
        'measured': {
            'area': khp_data.get('area'),
            'rf_mass': qc_result.get('rf_mass_measured')
        },
        'expected': {
            'rf_mass_cal': qc_result.get('rf_mass_expected')
        },
        'qc_result': {
            'status': qc_result.get('status'),
            'deviation_pct': qc_result.get('deviation_pct'),
            'message': qc_result.get('message')
        }
    }

    history['entries'].insert(0, entry)
    history['updated'] = datetime.now().isoformat()

    # Guardar
    try:
        with open(qc_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"Error guardant QC History: {e}")
        return False


def add_calibration(rf_mass_cal_values, source, valid_from, r2=None, n_points=None,
                    conditions=None, reason="", intercept_values=None,
                    regression_data=None):
    """
    Afegeix una nova calibració i tanca l'anterior.

    Args:
        rf_mass_cal_values: dict amb estructura {"direct": {"column": X, "bp": Y}, "uib": {...}}
        source: dict amb info de la font (type, description, seq_references)
        valid_from: Data inici vigència (YYYY-MM-DD)
        r2: Coeficient de determinació
        n_points: Nombre de punts usats
        conditions: dict amb condicions (column_type, flow_rate, etc.)
        reason: Motiu del canvi
        intercept_values: dict amb intercepts per senyal/mode
        regression_data: dict complet de la regressió (punts, residuals, stats).
            Estructura esperada (de fit_calibration_from_history):
            {
                "rf_mass_cal": float, "intercept": float, "r2": float,
                "n_points": int, "residuals_rms": float,
                "model": str,  # "intercept" o "origin"
                "signal": str,  # "direct", "uib", "254"
                "mode": str,  # "COLUMN", "BP"
                "points": [
                    {"seq_name": str, "date": str, "conc_ppm": float,
                     "volume_uL": float, "ug_doc": float, "area": float,
                     "rf_mass": float, "residual": float, "y_pred": float,
                     "excluded": bool (opcional)}
                ]
            }

    Returns:
        str: ID de la nova calibració o None si error
    """
    ref = load_calibration_reference()
    if not ref:
        ref = {
            'version': '2.0',
            'created': datetime.now().strftime('%Y-%m-%d'),
            'calibrations': [],
            'qc_thresholds': {
                'rf_mass_deviation_warning_pct': 15,
                'rf_mass_deviation_fail_pct': 25,
                'min_r2_new_calibration': 0.98,
                'min_points_new_calibration': 5
            }
        }

    # Generar ID
    cal_id = f"CAL_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Tancar calibració anterior
    valid_from_date = str(valid_from)[:10]
    for cal in ref['calibrations']:
        if cal.get('is_active', False):
            # Calcular dia anterior
            from datetime import timedelta
            try:
                vf = datetime.strptime(valid_from_date, '%Y-%m-%d')
                valid_to = (vf - timedelta(days=1)).strftime('%Y-%m-%d')
                cal['valid_to'] = valid_to
            except (ValueError, TypeError) as e:
                logger.warning("Could not parse valid_from_date '%s': %s", valid_from_date, e)
            cal['is_active'] = False

    # Crear nova calibració
    new_cal = {
        'id': cal_id,
        'rf_mass_cal': rf_mass_cal_values,
        'model': 'intercept' if intercept_values else 'origin',
        'intercept': intercept_values if intercept_values is not None else 0,
        'r2': r2,
        'n_points': n_points,
        'valid_from': valid_from_date,
        'valid_to': None,
        'is_active': True,
        'source': source or {},
        'conditions': conditions or {},
        'validation': ref.get('qc_thresholds', {}),
        'metadata': {
            'created_date': datetime.now().strftime('%Y-%m-%d'),
            'created_by': 'user',
            'reason': reason
        }
    }

    # Guardar dades completes de regressió (punts, residuals, stats)
    if regression_data:
        new_cal['regression_data'] = _sanitize_regression_data(regression_data)

    ref['calibrations'].insert(0, new_cal)
    ref['active_calibration_id'] = cal_id

    if save_calibration_reference(ref):
        return cal_id
    return None


def _sanitize_regression_data(reg_data):
    """
    Neteja les dades de regressió per serialització JSON.

    Converteix numpy a Python natiu, selecciona camps rellevants dels punts,
    i afegeix stats per concentració.

    Args:
        reg_data: dict retornat per fit_calibration_from_history()

    Returns:
        dict net per a JSON (sense numpy, sense camps innecessaris)
    """
    def _to_python(val):
        """Converteix numpy scalar a Python natiu."""
        if hasattr(val, 'item'):
            return val.item()
        return val

    sanitized = {
        'rf_mass_cal': _to_python(reg_data.get('rf_mass_cal', 0)),
        'intercept': _to_python(reg_data.get('intercept', 0)),
        'r2': _to_python(reg_data.get('r2', 0)),
        'n_points': int(reg_data.get('n_points', 0)),
        'residuals_rms': _to_python(reg_data.get('residuals_rms', 0)),
        'model': reg_data.get('model', 'intercept'),
        'signal': reg_data.get('signal', 'direct'),
        'mode': reg_data.get('mode', ''),
    }

    # Punts individuals
    points = reg_data.get('points', [])
    clean_points = []
    for p in points:
        clean_points.append({
            'seq_name': str(p.get('seq_name', '')),
            'date': str(p.get('date', '')),
            'conc_ppm': _to_python(p.get('conc_ppm', 0)),
            'volume_uL': _to_python(p.get('volume_uL', 0)),
            'ug_doc': _to_python(p.get('ug_doc', 0)),
            'area': _to_python(p.get('area', 0)),
            'rf_mass': _to_python(p.get('rf_mass', 0)),
            'residual': _to_python(p.get('residual', 0)),
            'y_pred': _to_python(p.get('y_pred', 0)),
            'excluded': bool(p.get('excluded', False)),
        })
    sanitized['points'] = clean_points

    # Stats per concentració (per la taula de l'informe)
    conc_stats = {}
    from collections import defaultdict
    by_conc = defaultdict(list)
    for p in clean_points:
        if not p.get('excluded'):
            by_conc[p['conc_ppm']].append(p)

    for conc, pts in sorted(by_conc.items()):
        areas = [p['area'] for p in pts]
        rfs = [p['rf_mass'] for p in pts]
        conc_stats[str(conc)] = {
            'n': len(pts),
            'area_mean': sum(areas) / len(areas),
            'area_std': float(np.std(areas)) if len(areas) > 1 else 0,
            'rf_mean': sum(rfs) / len(rfs),
            'rf_std': float(np.std(rfs)) if len(rfs) > 1 else 0,
            'rf_cv_pct': float(np.std(rfs) / np.mean(rfs) * 100) if len(rfs) > 1 and np.mean(rfs) > 0 else 0,
        }
    sanitized['stats_per_concentration'] = conc_stats

    return sanitized


# =============================================================================
# REQUANTIFICACIÓ RETROACTIVA
# =============================================================================

def requantify_analysis_json(json_path, new_rf_direct, new_intercept_direct,
                              new_rf_uib=None, new_intercept_uib=None,
                              new_rf_bp=None, new_intercept_bp=None):
    """
    Re-quantifica un analysis_result.json amb nous RF/intercept.

    Recalcula NOMES els camps ppm de quantification a partir de les àrees
    existents (que no canvien). No reprocessa cromatogrames.

    Args:
        json_path: Path a analysis_result.json
        new_rf_direct: Nou RF mass cal per direct
        new_intercept_direct: Nou intercept per direct
        new_rf_uib: RF per UIB (None = usar same as direct)
        new_intercept_uib: Intercept per UIB (None = usar same as direct)
        new_rf_bp: RF per BP direct (None = no tocar si és BP)
        new_intercept_bp: Intercept per BP (None = 0)

    Returns:
        dict amb {success, samples_updated, errors, mode}
    """
    import json as _json
    from datetime import datetime

    result = {"success": False, "samples_updated": 0, "errors": [], "mode": ""}

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = _json.load(f)
    except Exception as e:
        result["errors"].append(f"Error llegint JSON: {e}")
        return result

    # Determinar mode de la SEQ
    method = data.get("method", "").upper()
    is_bp = "BP" in method
    result["mode"] = "BP" if is_bp else "COLUMN"

    # Seleccionar RF/intercept segons mode
    if is_bp:
        rf_direct = new_rf_bp if new_rf_bp is not None else new_rf_direct
        intercept_direct = new_intercept_bp if new_intercept_bp is not None else 0
        rf_uib = new_rf_uib or rf_direct
        intercept_uib = new_intercept_uib if new_intercept_uib is not None else intercept_direct
    else:
        rf_direct = new_rf_direct
        intercept_direct = new_intercept_direct
        rf_uib = new_rf_uib or rf_direct
        intercept_uib = new_intercept_uib if new_intercept_uib is not None else intercept_direct

    if rf_direct <= 0:
        result["errors"].append(f"RF direct invàlid: {rf_direct}")
        return result

    def _apply_formula(area, rf, intercept):
        """Calcula ppm des d'àrea: ppm = max(0, area - intercept) * 1000 / (rf * vol)."""
        area_corrected = max(0, area - intercept)
        return area_corrected * 1000 / (rf * volume_uL) if volume_uL > 0 and rf > 0 else 0

    samples_grouped = data.get("samples_grouped", {})
    n_updated = 0

    for sample_name, sg in samples_grouped.items():
        # Saltar mostres light (BLANK/CONTROL) — no tenen quantificació
        if sg.get("analysis_type") == "light":
            continue

        quantification = sg.get("quantification")
        if not quantification:
            continue

        # Obtenir rèplica seleccionada per llegir àrees i volum
        selected = sg.get("selected", {})
        doc_rep_key = selected.get("doc", "1")
        if doc_rep_key == "none":
            continue

        replicas = sg.get("replicas", {})
        doc_replica = replicas.get(doc_rep_key, {})

        # Àrees DOC direct (no canvien)
        areas_doc = doc_replica.get("areas", {}).get("DOC", {})
        area_total = areas_doc.get("total", 0)

        # Volum d'injecció
        volume_uL = doc_replica.get("inj_volume")
        if not volume_uL or volume_uL <= 0:
            volume_uL = 100 if is_bp else 400  # fallback heurístic

        # === Requantificar DOC Direct ===
        if area_total > 0:
            ppm_direct = _apply_formula(area_total, rf_direct, intercept_direct)
            quantification["concentration_ppm"] = float(ppm_direct)
            quantification["concentration_ppm_direct"] = float(ppm_direct)

            # Fraccions (només COLUMN)
            if not is_bp:
                fractions = quantification.get("fractions", {})
                for frac_name in ["BioP", "HS", "BB", "SB", "LMW"]:
                    area_frac = areas_doc.get(frac_name, 0)
                    if area_frac > 0:
                        fractions[frac_name] = float(_apply_formula(area_frac, rf_direct, intercept_direct))
                    else:
                        fractions[frac_name] = 0.0
                quantification["fractions"] = fractions

        # === Requantificar DOC UIB ===
        areas_uib = doc_replica.get("areas_uib", {})
        area_total_uib = areas_uib.get("total", 0)

        if area_total_uib > 0 and rf_uib and rf_uib > 0:
            ppm_uib = _apply_formula(area_total_uib, rf_uib, intercept_uib)
            quantification["concentration_ppm_uib"] = float(ppm_uib)

            # Fraccions UIB (només COLUMN)
            if not is_bp:
                fractions_uib = quantification.get("fractions_uib", {})
                for frac_name in ["BioP", "HS", "BB", "SB", "LMW"]:
                    area_frac = areas_uib.get(frac_name, 0)
                    if area_frac > 0:
                        fractions_uib[frac_name] = float(_apply_formula(area_frac, rf_uib, intercept_uib))
                    else:
                        fractions_uib[frac_name] = 0.0
                quantification["fractions_uib"] = fractions_uib

        # Metadata de calibració
        quantification["rf_mass_cal_used"] = rf_direct
        quantification["intercept"] = intercept_direct
        quantification["calibration_source"] = "GLOBAL"

        sg["quantification"] = quantification
        n_updated += 1

    # Actualitzar fingerprint i timestamp de requantificació
    data["calibration_fingerprint"] = compute_calibration_fingerprint()
    data["requantified_at"] = datetime.now().isoformat()

    # Guardar
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            _json.dump(data, f, indent=2, ensure_ascii=False)
        result["success"] = True
        result["samples_updated"] = n_updated
    except Exception as e:
        result["errors"].append(f"Error guardant JSON: {e}")

    return result


def fit_calibration_from_history(calibrations, mode="COLUMN", signal="direct",
                                  model="intercept"):
    """
    Regressió lineal sobre dades KHP: Area = rf_mass_cal * ug_DOC + intercept.

    Args:
        calibrations: llista d'entrades KHP_History (dicts amb area, conc_ppm, volume_uL, etc.)
        mode: "COLUMN", "BP", o "ALL" (unificada COL+BP)
        signal: "direct", "uib", o "254"
        model: "intercept" (lliure) o "origin" (intercept=0)

    Returns:
        dict: rf_mass_cal, intercept, r2, n_points, points[], residuals_rms, success
    """
    from scipy.stats import linregress

    # Filtrar per mode, descartar outliers i no vàlids
    filtered = []
    for cal in calibrations:
        cal_mode = cal.get('mode', '').upper()
        if mode.upper() != "ALL" and cal_mode != mode.upper():
            continue
        if cal.get('is_outlier', False):
            continue
        if not cal.get('valid_for_calibration', True):
            continue

        conc = cal.get('conc_ppm', 0)
        vol = cal.get('volume_uL', 0)
        if conc <= 0 or vol <= 0:
            continue

        # Àrea segons senyal
        if signal.lower() == 'uib':
            area = cal.get('area_u', 0)
        elif signal.lower() == '254':
            area = cal.get('area_254', 0) or cal.get('a254_area', 0) or 0
        else:
            area = cal.get('area', 0)

        if area <= 0:
            continue

        ug_doc = conc * vol / 1000.0  # µg DOC injectat
        filtered.append({
            'seq_name': cal.get('seq_name', ''),
            'date': cal.get('date', ''),
            'mode': cal_mode,
            'conc_ppm': conc,
            'volume_uL': vol,
            'ug_doc': ug_doc,
            'area': area,
            'rf_mass': area / ug_doc if ug_doc > 0 else 0,
            'is_outlier': cal.get('is_outlier', False),
        })

    if len(filtered) < 2:
        return {
            'rf_mass_cal': None, 'intercept': 0, 'r2': None,
            'n_points': len(filtered), 'points': filtered,
            'residuals_rms': None, 'success': False,
            'error': f'Insuficients punts ({len(filtered)}), mínim 2'
        }

    X = np.array([p['ug_doc'] for p in filtered])
    Y = np.array([p['area'] for p in filtered])

    if model == "origin":
        # Regressió per l'origen: slope = Σ(XY)/Σ(X²)
        slope = np.sum(X * Y) / np.sum(X ** 2)
        intercept_val = 0.0
        Y_pred = slope * X
        ss_res = np.sum((Y - Y_pred) ** 2)
        ss_tot = np.sum((Y - np.mean(Y)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    else:
        # Regressió lliure amb intercept
        # Si tots els X són iguals, linregress falla → fallback a origin
        if np.all(X == X[0]):
            slope = np.sum(X * Y) / np.sum(X ** 2)
            intercept_val = 0.0
            Y_pred = slope * X
            ss_res = np.sum((Y - Y_pred) ** 2)
            ss_tot = np.sum((Y - np.mean(Y)) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        else:
            lr = linregress(X, Y)
            slope = lr.slope
            intercept_val = lr.intercept
            r2 = lr.rvalue ** 2
        Y_pred = slope * X + intercept_val

    residuals = Y - Y_pred
    rms = float(np.sqrt(np.mean(residuals ** 2)))

    # Afegir residual a cada punt
    for i, p in enumerate(filtered):
        p['residual'] = float(residuals[i])
        p['y_pred'] = float(Y_pred[i])

    return {
        'rf_mass_cal': float(slope),
        'intercept': float(intercept_val),
        'r2': float(r2),
        'n_points': len(filtered),
        'points': filtered,
        'residuals_rms': rms,
        'success': True,
    }


# =============================================================================
# FUNCIONS AUXILIARS
# =============================================================================

def get_injection_volume(seq_path, is_bp, manifest_volume=None):
    """
    Retorna el volum d'injecció en µL.

    Prioritat:
    1) manifest_volume — valor llegit del MasterFile (font de veritat)
    2) Heurístic per mode/SEQ — NOMÉS si manifest no té dades

    Si cap font proporciona volum, retorna None (NO suposar).

    Args:
        seq_path: Path de la SEQ
        is_bp: True si mode BP
        manifest_volume: Volum llegit del manifest (pot ser None)

    Returns:
        float o None
    """
    # 1) Manifest té prioritat absoluta
    if manifest_volume is not None:
        return float(manifest_volume)

    # 2) Heurístic com a fallback (amb warning al log)
    logger.warning("get_injection_volume: No manifest volume for %s (is_bp=%s) — using heuristic",
                   os.path.basename(seq_path) if seq_path else "?", is_bp)

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
    # Usar precisió decimal per distingir 0.1/0.25/0.5/1/2/5 ppm
    if conc_ppm and conc_ppm != int(conc_ppm):
        conc_str = f"{conc_ppm:.2f}".rstrip('0').rstrip('.')
    else:
        conc_str = str(int(conc_ppm)) if conc_ppm else "0"
    return f"{mode}_{vol}_{conc_str}"


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
    Calcula els límits d'integració d'un pic KHP usant mètode de derivades (Agilent-style).

    LÒGICA:
    1. Delegar a find_peak_boundaries() de hpsec_core.py (mètode derivades)
    2. Calcular baseline com a MODA del senyal
    3. Validar amplada mínima/màxima
    4. Retornar info completa per compatibilitat

    Args:
        t: Array de temps (minuts)
        y: Array de senyal
        peak_idx: Índex del màxim del pic
        min_width_min: Amplada mínima en minuts (default 1.0)
        max_width_min: Amplada màxima en minuts (default 6.0)

    Returns:
        Dict amb left_idx, right_idx, baseline, threshold, width_minutes, etc.
    """
    from hpsec_core import find_peak_boundaries

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

        threshold = baseline + 3 * std_baseline

        # === 3. DELIMITAR per DERIVADES (mètode Agilent) ===
        left_idx, right_idx = find_peak_boundaries(
            t, y, peak_idx, baseline_level=baseline, is_bp=False
        )

        # === 4. VALIDAR AMPLADA ===
        dt = np.mean(np.diff(t)) if n > 1 else 0.01
        max_width_idx = int(max_width_min / dt) if dt > 0 else 300
        min_width_idx = int(min_width_min / dt) if dt > 0 else 50

        current_width_idx = right_idx - left_idx

        # Expandir si massa estret
        if current_width_idx < min_width_idx:
            expand_needed = (min_width_idx - current_width_idx) // 2 + 1
            left_idx = max(0, left_idx - expand_needed)
            right_idx = min(n - 1, right_idx + expand_needed)

        # Retallar si massa ample
        if current_width_idx > max_width_idx:
            excess = (current_width_idx - max_width_idx) // 2
            left_idx = min(left_idx + excess, peak_idx - 3)
            right_idx = max(right_idx - excess, peak_idx + 3)

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
            "message": "OK" if (left_at_baseline and right_at_baseline) else "Limits no arriben a baseline",
            "method": "derivative"
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


# NOTA: detect_irregular_top i detect_timeout ara estan a hpsec_core.py
# (Single Source of Truth per evitar duplicació)
# Usar: from hpsec_core import detect_irregular_top, detect_timeout, detect_peak_anomaly


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
        # Filtrar per concentració si s'especifica (tolerància relativa 10%)
        if conc_ppm is not None:
            cal_conc = cal.get('conc_ppm', 0)
            tol = max(0.01, conc_ppm * 0.1)
            if abs(cal_conc - conc_ppm) > tol:
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
                          exclude_outliers=True):  # C12/C17: SEMPRE excloure outliers
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
            # C12/C17: Thresholds MOLT estrictes per calibració (iguals que Column)
            # FAIL si àrea desvia >20%, WARNING >10%
            if area_deviation_pct > 20:
                result['status'] = 'INVALID'
                result['issues'].append(f"BP: Desviació àrea {area_deviation_pct:.1f}% (>20%)")
            elif area_deviation_pct > 10:
                if result['status'] == 'OK':
                    result['status'] = 'WARNING'
                result['warnings'].append(f"BP: Desviació àrea {area_deviation_pct:.1f}% (>10%)")
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
    3. No cim irregular
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

    # 0. Guard: conc i area han de ser > 0 (entrades blanques/invàlides)
    conc = khp_data.get('conc_ppm', khp_data.get('conc', 0))
    area = khp_data.get('area', khp_data.get('area_total', 0))
    if not conc or conc <= 0:
        issues.append("ZERO_CONC: concentració KHP és 0 o absent")
        quality_score += 200
    if not area or area <= 0:
        issues.append("ZERO_AREA: àrea KHP és 0 o absent")
        quality_score += 200

    # 1. Multi-pic
    if all_peaks and len(all_peaks) > 2:
        significant_peaks = [p for p in all_peaks if p.get('prominence', 0) > khp_data.get('height', 1) * 0.1]
        if len(significant_peaks) > 2:
            issues.append(f"MULTI_PEAK: {len(significant_peaks)} pics significatius detectats")
            quality_score += 30 * (len(significant_peaks) - 2)

    # 2. Timeout — severity WARNING o CRITICAL invalida directament (+100)
    # INFO: timeout en zona segura, no penalitza
    if timeout_info:
        severity = timeout_info.get('severity', 'OK')
        if severity == 'CRITICAL':
            issues.append(f"TIMEOUT_CRITICAL: timeout en zona HS")
            quality_score += 100
        elif severity == 'WARNING':
            zones = timeout_info.get('zone_summary', {})
            affected = [z for z in zones.keys() if zones[z] > 0]
            issues.append(f"TIMEOUT_WARNING: timeout en {', '.join(affected)}")
            quality_score += 100
        # INFO: No penalitzar - timeout en zona segura

    # 3. Anomalia de forma (cim irregular / jagged)
    if anomaly_info:
        if anomaly_info.get('is_irregular_top', False):
            issues.append("IRREGULAR_TOP: detectat cim irregular al pic (jagged/batman)")
            quality_score += 50
        if anomaly_info.get('is_irregular', False) and not anomaly_info.get('is_irregular_top', False):
            smoothness = anomaly_info.get('smoothness', 100)
            warnings.append(f"ROUGH_TOP: smoothness baixa ({smoothness:.0f}%)")
            quality_score += 30

    # 4. Simetria (C11: irrellevant per BP)
    # NOTA: Per KHP, el rang acceptable és més ampli (0.5-2.0 és normal)
    # Només alerta severa si és molt asimètric (<0.4 o >2.5)
    symmetry = khp_data.get('symmetry', 1.0)
    is_bp_mode = khp_data.get('is_bp', False)
    if not is_bp_mode:  # Simetria no es valida per BP
        if symmetry < 0.4 or symmetry > 2.5:
            issues.append(f"ASYMMETRY: simetria fora de rang ({symmetry:.2f})")
            quality_score += 20
        elif symmetry < 0.5 or symmetry > 2.0:
            # Warning lleu - no crític per KHP
            warnings.append(f"ASYMMETRY_WARN: simetria límit ({symmetry:.2f})")
            quality_score += 3

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

    # 7b. Bigaussian shape quality (R² > 0.98 requerit)
    bigaussian_doc = khp_data.get('bigaussian_doc')
    bigaussian_254 = khp_data.get('bigaussian_254')

    if bigaussian_doc and bigaussian_doc.get('status') not in ('ERROR', None):
        r2_doc = bigaussian_doc.get('r2', 0)
        if r2_doc < 0.95:
            issues.append(f"PEAK_SHAPE_POOR: bigaussian DOC R²={r2_doc:.3f} < 0.95")
            quality_score += 40
        elif r2_doc < 0.98:
            warnings.append(f"PEAK_SHAPE_WARN: bigaussian DOC R²={r2_doc:.3f} < 0.98")
            quality_score += 15

    if bigaussian_254 and bigaussian_254.get('status') not in ('ERROR', None):
        r2_254 = bigaussian_254.get('r2', 0)
        if r2_254 < 0.95:
            issues.append(f"PEAK_SHAPE_POOR_254: bigaussian 254nm R²={r2_254:.3f} < 0.95")
            quality_score += 30
        elif r2_254 < 0.98:
            warnings.append(f"PEAK_SHAPE_WARN_254: bigaussian 254nm R²={r2_254:.3f} < 0.98")
            quality_score += 10

    # 7c. t_retention anomal (checks basats en t_max_254 o t_max_DOC)
    t_retention = khp_data.get('t_retention', 0)
    t_dad_max = khp_data.get('t_dad_max', 0)
    t_ref = t_dad_max if t_dad_max and t_dad_max > 0 else t_retention

    if t_ref > 0:
        if is_bp_mode:
            if t_ref > 3.5:
                issues.append(f"T_RETENTION_ANOMAL: t={t_ref:.2f} min (BP esperat <3.5)")
                quality_score += 50
        else:
            if t_ref < 18 or t_ref > 25:
                issues.append(f"T_RETENTION_ANOMAL: t={t_ref:.2f} min (COLUMN esperat 18-25)")
                quality_score += 50

    # 7d. t_retention mismatch DOC vs 254
    if t_retention > 0 and t_dad_max and t_dad_max > 0:
        dt = abs(t_retention - t_dad_max)
        if dt > 2.0:
            issues.append(f"T_MISMATCH: |DOC({t_retention:.2f})-254({t_dad_max:.2f})|={dt:.2f} min >2")
            quality_score += 60
        elif dt > 1.0:
            warnings.append(f"T_MISMATCH_WARN: |DOC({t_retention:.2f})-254({t_dad_max:.2f})|={dt:.2f} min >1")
            quality_score += 20

    # 7e. NO_DAD_254_REFERENCE
    has_dad = khp_data.get('a254_area', 0) > 0 or (t_dad_max and t_dad_max > 0)
    if not has_dad:
        warnings.append("NO_DAD_254: sense senyal 254nm per verificar alineació")
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

        # C12/C17: Outliers exclosos per defecte (exclude_outliers=True)
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
                               method="COLUMN", repair_irregular_top=True,
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
    5. IRREGULAR_TOP: pic amb cim irregular (jagged/batman, artefacte detector)
    6. HISTORICAL_DEVIATION: àrea desvia significativament de l'històric

    Args:
        t_doc, y_doc: Senyal DOC (Direct o UIB)
        t_dad, y_a254: Senyal A254
        t_uib, y_uib: Senyal UIB (opcional)
        method: "COLUMN" o "BP"
        repair_irregular_top: Si True, repara pics amb cim irregular per millorar precisió t_max
        seq_path: Path de la SEQ (per comparació històrica)
        conc_ppm: Concentració KHP en ppm (per comparació històrica)
        volume_uL: Volum d'injecció en µL (per comparació històrica)

    Returns:
        dict amb:
            - valid: bool
            - issues: list de problemes detectats
            - warnings: list d'avisos
            - metrics: dict amb mètriques calculades
            - y_doc_clean: senyal DOC netejat (si irregular top reparat)
            - t_max_corrected: t_max corregit si irregular top reparat
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

    # === 0. DETECTAR I REPARAR CIM IRREGULAR (jagged/batman, si activat) ===
    if method == "COLUMN":
        peak_zone = (t_doc >= 15) & (t_doc <= 30)
    else:
        peak_zone = (t_doc >= 0) & (t_doc <= 5)

    t_peak_zone = t_doc[peak_zone]
    y_peak_zone = y_doc[peak_zone]

    irregular_top_info = None
    y_doc_working = y_doc.copy()

    if len(t_peak_zone) > 20:
        irregular_top_info = detect_irregular_top(t_peak_zone, y_peak_zone, top_pct=0.20, min_valley_depth=0.02)
        smoothness_info = calc_top_smoothness(t_peak_zone, y_peak_zone)
        smoothness_val = smoothness_info.get("smoothness", 100.0)
        result["metrics"]["irregular_top_detected"] = irregular_top_info.get("is_irregular_top", False)
        result["metrics"]["smoothness"] = smoothness_val

        is_irregular = irregular_top_info.get("is_irregular_top", False)
        is_rough = smoothness_val < 70.0
        needs_repair = is_irregular or is_rough

        if needs_repair:
            anomaly_type = "IRREGULAR_TOP" if is_irregular else "ROUGH_TOP"
            if is_irregular:
                result["warnings"].append(
                    f"IRREGULAR_TOP: Detectat cim irregular (profunditat {irregular_top_info.get('max_depth', 0)*100:.1f}%)"
                )
            else:
                result["warnings"].append(
                    f"ROUGH_TOP: Cim rugós (smoothness={smoothness_val:.1f}%)"
                )

            if repair_irregular_top:
                try:
                    force_repair = is_rough and not is_irregular
                    y_repaired, repair_info, was_repaired = repair_with_parabola(
                        t_peak_zone, y_peak_zone, force=force_repair
                    )
                    if was_repaired:
                        y_doc_working[peak_zone] = y_repaired
                        result["y_doc_clean"] = y_doc_working
                        result["metrics"]["irregular_top_repaired"] = True
                        result["warnings"].append("IRREGULAR_TOP_REPAIRED: Pic reparat amb paràbola")

                        idx_max_repaired = np.argmax(y_repaired)
                        t_max_corrected = t_peak_zone[idx_max_repaired]
                        result["t_max_corrected"] = float(t_max_corrected)
                except Exception as e:
                    result["metrics"]["irregular_top_repair_error"] = str(e)

    # Trobar pics - usar y_doc_working (reparat si cim irregular)
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

    # Àrea DOC - usar y_doc_working (reparat si cim irregular)
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
            # Usar y_doc_working (reparat si cim irregular)
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
        logger.error(f"Error carregant calibracions: {e}")
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
        logger.error(f"Error guardant CHECK/data: {e}")
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
        logger.error(f"Error carregant històric KHP: {e}")
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
        logger.error(f"Error guardant històric KHP: {e}")
        return False


def clean_khp_history(seq_path, dry_run=True):
    """
    Neteja entrades invàlides del KHP_History.json.

    Elimina:
    - Entrades amb conc_ppm=0 (blancs invàlids)
    - Entrades amb area=0 (sense dades DOC)
    - Entrades amb rf=0 i rf_mass_cal=0 (sense RF calculat)

    Args:
        seq_path: Path per localitzar KHP_History.json
        dry_run: Si True, només reporta sense modificar

    Returns:
        dict amb {removed: int, kept: int, total: int, removed_entries: [...]}
    """
    history = load_khp_history(seq_path)
    if not history:
        return {"removed": 0, "kept": 0, "total": 0, "removed_entries": []}

    kept = []
    removed = []

    for entry in history:
        conc = entry.get("conc_ppm", 0)
        area = entry.get("area", 0)
        rf = entry.get("rf", 0)
        rf_mass = entry.get("rf_mass_cal", 0)
        seq = entry.get("seq_name", "?")

        reasons = []
        if not conc or conc <= 0:
            reasons.append("conc=0")
        if not area or area <= 0:
            reasons.append("area=0")
        if (not rf or rf <= 0) and (not rf_mass or rf_mass <= 0):
            reasons.append("rf=0")

        if reasons:
            entry["_removal_reasons"] = reasons
            removed.append(entry)
            logger.info("clean_khp_history: REMOVE %s — %s", seq, ", ".join(reasons))
        else:
            kept.append(entry)

    result = {
        "removed": len(removed),
        "kept": len(kept),
        "total": len(history),
        "removed_entries": [
            {"seq_name": e.get("seq_name", "?"), "conc": e.get("conc_ppm", 0),
             "area": e.get("area", 0), "reasons": e.get("_removal_reasons", [])}
            for e in removed
        ],
    }

    if not dry_run and removed:
        save_khp_history(seq_path, kept)
        logger.info("clean_khp_history: eliminades %d/%d entrades, %d vàlides conservades",
                     len(removed), len(history), len(kept))

    return result


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

    # conc=0 és acceptable: shift, àrees i detecció d'anomalies no necessiten concentració.
    # Només RF (response factor) no es pot calcular sense conc (ja protegit amb if conc > 0).

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
    mode = "BP" if is_bp_chromato else "COLUMN"

    # =========================================================================
    # STEP 0: Integrar DAD 254nm PRIMER (referència temporal)
    # El 254nm defineix la posició del pic KHP. El DOC s'alinea després.
    # =========================================================================
    has_dad = df_dad is not None and not df_dad.empty
    t_dad = None
    dad_254 = None
    t_max_254 = None  # Referència temporal del pic KHP
    dad_peak_info = None
    a254_area = 0.0
    a254_area_total = 0.0
    cr_254 = np.nan
    dad_quality_warnings = []

    if has_dad and "time (min)" in df_dad.columns:
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
                # Integrar 254nm amb derivades (find_peak_boundaries)
                dad_peak_info = detect_main_peak(t_dad, dad_254, config["peak_min_prominence_pct"])

                if dad_peak_info and dad_peak_info.get('valid'):
                    t_max_254 = dad_peak_info.get('t_max', 0)
                    dad_pk_idx = dad_peak_info.get('peak_idx', 0)
                    dad_l_idx = dad_peak_info.get('left_idx', 0)
                    dad_r_idx = dad_peak_info.get('right_idx', len(t_dad) - 1)

                    # Buscar límits en all_peaks DAD
                    dad_all_peaks = detect_all_peaks(t_dad, dad_254, config["peak_min_prominence_pct"])
                    for pk in dad_all_peaks:
                        if pk['idx'] == dad_pk_idx or abs(pk['t'] - t_max_254) < 0.1:
                            dad_l_idx = pk['left_idx']
                            dad_r_idx = pk['right_idx']
                            break

                    # Àrea 254nm
                    if dad_r_idx > dad_l_idx:
                        a254_area = float(trapezoid(dad_254[dad_l_idx:dad_r_idx+1],
                                                    t_dad[dad_l_idx:dad_r_idx+1]))

                    # Àrea total 254nm
                    a254_area_total = float(trapezoid(np.maximum(dad_254, 0), t_dad))

                    # Check 90%: àrea pic principal vs total
                    if a254_area_total > 0 and a254_area > 0:
                        cr_254 = a254_area / a254_area_total
                        if cr_254 < 0.90:
                            dad_quality_warnings.append(
                                f"MULTI_PEAK_254: pic principal={cr_254:.0%} de l'àrea total 254nm")

                    # t_ret check per 254
                    if is_bp_chromato and t_max_254 > 3.5:
                        dad_quality_warnings.append(
                            f"T_RETENTION_254_ANOMAL: t_max={t_max_254:.2f} min (BP esperat <3.5)")
                    elif not is_bp_chromato and (t_max_254 < 18 or t_max_254 > 25):
                        dad_quality_warnings.append(
                            f"T_RETENTION_254_ANOMAL: t_max={t_max_254:.2f} min (COLUMN esperat 18-25)")

                    logger.debug("analizar_khp_data: 254nm integrat independent: t_max=%.2f, area=%.2f, CR=%.2f",
                                 t_max_254, a254_area, cr_254 if not np.isnan(cr_254) else 0)
                else:
                    dad_quality_warnings.append("NO_254_PEAK: pic 254nm no detectat o invàlid")
                    logger.warning("analizar_khp_data: pic DAD 254nm no vàlid")
    else:
        dad_quality_warnings.append("NO_DAD_254_REFERENCE: no hi ha senyal 254nm, no es pot verificar alineació")

    # =========================================================================
    # STEP 1: Integrar DOC alineat a t_max_254 (o independent si no hi ha 254)
    # =========================================================================
    all_peaks = detect_all_peaks(t_doc, y_doc_net, config["peak_min_prominence_pct"])

    if t_max_254 is not None and len(all_peaks) > 0:
        # Buscar pic DOC més proper a t_max_254
        best_peak = min(all_peaks, key=lambda pk: abs(pk['t'] - t_max_254))
        # Si el pic més proper és massa lluny (>2 min), usar detect_main_peak normal
        if abs(best_peak['t'] - t_max_254) > 2.0:
            dad_quality_warnings.append(
                f"T_RETENTION_MISMATCH: pic DOC a {best_peak['t']:.2f} vs 254 a {t_max_254:.2f} min")
            peak_info = detect_main_peak(t_doc, y_doc_net, config["peak_min_prominence_pct"])
        else:
            # Usar el pic DOC alineat al 254
            peak_info = detect_main_peak(t_doc, y_doc_net, config["peak_min_prominence_pct"])
            # Si detect_main_peak no ha trobat el pic alineat, forçar-lo
            if peak_info.get('valid') and abs(peak_info['t_max'] - best_peak['t']) > 0.5:
                # El pic principal DOC no coincideix amb el 254 → usar el proper al 254
                peak_info['peak_idx'] = best_peak['idx']
                peak_info['t_max'] = best_peak['t']
                peak_info['left_idx'] = best_peak['left_idx']
                peak_info['right_idx'] = best_peak['right_idx']
                peak_info['area'] = float(trapezoid(
                    y_doc_net[best_peak['left_idx']:best_peak['right_idx']+1],
                    t_doc[best_peak['left_idx']:best_peak['right_idx']+1]))
                dad_quality_warnings.append(
                    f"DOC_ALIGNED_TO_254: pic DOC seleccionat per alineació amb 254nm (t={best_peak['t']:.2f})")
    else:
        peak_info = detect_main_peak(t_doc, y_doc_net, config["peak_min_prominence_pct"])

    if not peak_info.get('valid', False):
        return None

    t_retention = peak_info.get('t_max', 0)

    # Shift temporal DOC vs 254
    shift_khp = (t_max_254 - t_retention) if t_max_254 is not None else 0.0

    # Check t_mismatch DOC vs 254
    if t_max_254 is not None and abs(shift_khp) > 1.0:
        dad_quality_warnings.append(
            f"T_RETENTION_MISMATCH: |DOC({t_retention:.2f}) - 254({t_max_254:.2f})| = {abs(shift_khp):.2f} min")

    # Baseline stats
    bl_stats = get_baseline_stats(t_doc, y_doc_net, mode=mode)

    # Límits del pic DOC
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
    # has_timeout only for WARNING/CRITICAL — INFO (safe zones like SB) is not relevant
    timeout_info = detect_timeout(t_doc, is_bp=is_bp_chromato)
    timeout_severity = timeout_info.get('severity', 'OK')
    has_timeout = timeout_info['n_timeouts'] > 0 and timeout_severity in ('WARNING', 'CRITICAL')

    # Cim irregular (jagged/batman) / Anomalies
    t_peak_seg = t_doc[left_idx:right_idx+1]
    y_peak_seg = y_doc_net[left_idx:right_idx+1]
    anomaly_info = detect_peak_anomaly(t_peak_seg, y_peak_seg)
    has_irregular_top = anomaly_info.get('is_irregular_top', False)
    has_irregular = anomaly_info.get('is_irregular', False)
    smoothness = anomaly_info.get('smoothness', 100.0)

    # Reparació cim irregular: intentar reparar amb paràbola si detectat
    # Aplica tant per IRREGULAR_TOP (pic-vall-pic) com ROUGH_TOP (smoothness baixa)
    irregular_top_repaired = False
    repair_info = None
    area_original = peak_info['area']
    if has_irregular_top or has_irregular:
        try:
            # force=True si ROUGH_TOP sense valls profundes (has_irregular sense has_irregular_top)
            force_repair = has_irregular and not has_irregular_top
            y_repaired_seg, repair_info, was_repaired = repair_with_parabola(
                t_peak_seg, y_peak_seg, force=force_repair
            )
            if was_repaired:
                irregular_top_repaired = True
                # Recalcular àrea amb senyal reparat
                area_repaired = float(trapezoid(y_repaired_seg, t_peak_seg))
                peak_info['area_original'] = area_original
                peak_info['area'] = area_repaired
                peak_info['area_repaired'] = area_repaired
                # Actualitzar senyal al segment
                y_doc_net_repaired = y_doc_net.copy()
                y_doc_net_repaired[left_idx:right_idx+1] = y_repaired_seg
        except Exception:
            irregular_top_repaired = False

    # DAD 254nm: ratio DOC/254
    a254_doc_ratio = peak_info['area'] / a254_area if a254_area > 0 else 0.0

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

    # Baseline drift local
    n_bl = max(3, len(t_doc) // 10)
    bl_left = float(np.median(y_doc_net[max(0, left_idx - n_bl):left_idx])) if left_idx > 0 else 0.0
    bl_right_slice = y_doc_net[right_idx + 1:min(len(y_doc_net), right_idx + n_bl + 1)]
    bl_right = float(np.median(bl_right_slice)) if len(bl_right_slice) > 0 else 0.0
    bl_mean = (bl_left + bl_right) / 2
    bl_drift_pct = abs(bl_mean / float(y_doc_net[peak_idx]) * 100) if y_doc_net[peak_idx] > 0 else 0.0

    # Volum injecció (prioritza metadata del manifest, heurístic com a fallback)
    volume_uL = get_injection_volume(seq_path, is_bp_chromato, manifest_volume=volume_uL_meta)
    if volume_uL is None:
        logger.warning("analizar_khp_data: VOLUM DESCONEGUT per %s — usant 100 uL per evitar crash",
                       os.path.basename(seq_path) if seq_path else "?")
        volume_uL = 100  # Fallback últim recurs per no crashejar

    # Qualitat
    quality_score = 0
    quality_issues = []

    # Afegir warnings de DAD 254nm (detectats a STEP 0)
    for dw in dad_quality_warnings:
        quality_issues.append(dw)
        if "ANOMAL" in dw or "MISMATCH" in dw:
            quality_score += 50
        elif "MULTI_PEAK" in dw:
            quality_score += 30
        elif "NO_DAD_254" in dw or "NO_254_PEAK" in dw:
            quality_score += 20

    # Check DOC: àrea pic principal vs total (90% check)
    if concentration_ratio < 0.90 and area_total > 0:
        quality_issues.append(
            f"MULTI_PEAK_DOC: pic principal={concentration_ratio:.0%} de l'àrea total DOC")
        quality_score += 30

    # C11: Simetria i smoothness irrellevants per BP (senyal molt variable)
    if not is_bp_chromato:
        # Simetria alta (>1.5) NO es penalitza per KHP - és normal per un patró
        if symmetry < 0.8:
            quality_score += 10
            quality_issues.append(f"Simetria baixa ({symmetry:.2f})")
        if has_irregular:
            quality_score += 30
            quality_issues.append(f"Pic irregular (smoothness={smoothness:.0f}%)")

    if snr < 10:
        quality_score += 20
        quality_issues.append(f"SNR baix ({snr:.1f})")
    if has_irregular_top:
        quality_score += 100
        quality_issues.append("Pic amb cim irregular (jagged/batman)")
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
    if bl_drift_pct > 8:
        quality_score += 10
        quality_issues.append(f"Baseline drift alt ({bl_drift_pct:.0f}%)")

    # =========================================================================
    # NOVES MÈTRIQUES: FWHM, RF, RF_MASS, CR per tots els senyals
    # =========================================================================

    # FWHM per DOC
    fwhm_doc = calculate_fwhm(t_doc, y_doc_net, peak_idx, left_idx, right_idx)

    # FWHM per 254nm (usa límits propis del 254, no transferits del DOC)
    fwhm_254 = np.nan
    if has_dad and dad_peak_info and dad_peak_info.get('valid'):
        _dad_pk = dad_peak_info.get('peak_idx', 0)
        _dad_li = dad_peak_info.get('left_idx', 0)
        _dad_ri = dad_peak_info.get('right_idx', len(t_dad) - 1 if t_dad is not None else 0)
        if t_dad is not None and dad_254 is not None:
            fwhm_254 = calculate_fwhm(t_dad, dad_254, _dad_pk, _dad_li, _dad_ri)

    # =========================================================================
    # BIGAUSSIAN FIT (C05) - Sempre guardar (INVALID = info QC valuosa)
    # =========================================================================
    bigaussian_doc = None
    bigaussian_254 = None

    # Bigaussian per DOC
    try:
        if len(t_doc) > 20 and peak_idx > 5 and peak_idx < len(t_doc) - 5:
            bigauss_result = fit_bigaussian(t_doc, y_doc_net, peak_idx, left_idx, right_idx)
            bigaussian_doc = {
                "r2": bigauss_result.get("r2", 0),
                "amplitude": bigauss_result.get("amplitude", 0),
                "mu": bigauss_result.get("mu", 0),
                "sigma_left": bigauss_result.get("sigma_left", 0),
                "sigma_right": bigauss_result.get("sigma_right", 0),
                "asymmetry": bigauss_result.get("asymmetry", 1),
                "status": bigauss_result.get("status", "INVALID"),
            }
    except Exception as e:
        bigaussian_doc = {"r2": 0, "status": "ERROR", "error": str(e)}

    # Bigaussian per 254nm (usa límits propis del 254)
    try:
        if has_dad and dad_peak_info and dad_peak_info.get('valid'):
            _dad_pk = dad_peak_info.get('peak_idx', 0)
            _dad_li = dad_peak_info.get('left_idx', 0)
            _dad_ri = dad_peak_info.get('right_idx', len(t_dad) - 1 if t_dad is not None else 0)
            if t_dad is not None and dad_254 is not None and len(t_dad) > 20:
                bigauss_254 = fit_bigaussian(t_dad, dad_254, _dad_pk, _dad_li, _dad_ri)
                bigaussian_254 = {
                    "r2": bigauss_254.get("r2", 0),
                    "asymmetry": bigauss_254.get("asymmetry", 1),
                    "status": bigauss_254.get("status", "INVALID"),
                }
    except Exception as e:
        bigaussian_254 = {"r2": 0, "status": "ERROR", "error": str(e)}

    # RF = Area / Concentració (ppm)
    rf_doc = peak_info['area'] / conc if conc > 0 else 0.0

    # RF_MASS = Area × 1000 / (Concentració × Volum) = Area / µg DOC injectat
    rf_mass_doc = peak_info['area'] * 1000 / (conc * volume_uL) if conc > 0 and volume_uL > 0 else 0.0

    # RF i RF_MASS per 254nm
    rf_254 = a254_area / conc if conc > 0 and a254_area > 0 else 0.0
    rf_mass_254 = a254_area * 1000 / (conc * volume_uL) if conc > 0 and volume_uL > 0 and a254_area > 0 else 0.0

    # CR 254nm ja calculat a STEP 0 (cr_254, a254_area_total)

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
        't_dad_max': t_max_254,  # 254nm és la referència temporal
        't_doc': t_doc,
        'y_doc': y_doc_net,
        'y_doc_repaired': y_doc_net_repaired if irregular_top_repaired else None,
        't_dad': t_dad,
        'y_dad_254': dad_254,
        'symmetry': symmetry,
        'snr': snr,
        'baseline_stats': bl_stats,
        'all_peaks_count': len(all_peaks),
        'all_peaks': all_peaks,
        'quality_score': quality_score,
        'quality_issues': quality_issues,
        'has_irregular_top': has_irregular_top,
        'irregular_top_repaired': irregular_top_repaired,
        'repair_info': repair_info,
        'area_original': area_original if irregular_top_repaired else None,
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
        'rf_mass_doc': rf_mass_doc,
        'rf_254': rf_254,
        'rf_mass_254': rf_mass_254,
        'cr_254': cr_254,
        'a254_area_total': a254_area_total,
        # Bigaussian fit (C05 - per BP mode)
        'bigaussian_doc': bigaussian_doc,
        'bigaussian_254': bigaussian_254,
        # Baseline drift
        'bl_left': bl_left,
        'bl_right': bl_right,
        'bl_drift_pct': bl_drift_pct,
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
    except Exception as e:
        logger.warning("Failed to get KHP reference values: %s", e)
    return None


def register_calibration(seq_path, khp_data, khp_source, mode="COLUMN"):
    """
    Registra una nova calibració a l'històric.

    Guarda a DOS llocs:
    1. LOCAL (CHECK/data/calibrations.json) - Historial complet de la SEQ
    2. GLOBAL (KHP_History.json) - Una entrada per SEQ per comparacions

    VALIDACIÓ COMPLETA amb validate_khp_quality():
    - valid_for_shift: Pic clar, sense timeout crític, cim irregular reparat
    - valid_for_calibration: Tots els criteris de qualitat (8 criteris)
    """
    seq_name = os.path.basename(seq_path)

    # Calcular RF (Response Factor) = Àrea / Concentració
    area = khp_data.get('area', 0)
    conc = khp_data.get('conc_ppm', 0)

    # Guard: entrades blanques (conc=0 o area=0) no s'han de registrar
    if not conc or conc <= 0:
        logger.warning("register_calibration: skip entry with conc=0 for %s", seq_name)
        return {"success": False, "error": "KHP concentration is 0", "valid_for_calibration": False}
    if not area or area <= 0:
        logger.warning("register_calibration: skip entry with area=0 for %s", seq_name)
        return {"success": False, "error": "KHP area is 0", "valid_for_calibration": False}

    rf = area / conc

    seq_date = khp_data.get('seq_date', '')
    if not seq_date:
        seq_date = datetime.now().isoformat()

    is_bp = khp_data.get('is_bp', False)
    # Volum: prioritat manifest → heurístic → warning si cap
    volume_from_khp = khp_data.get('volume_uL')
    volume = get_injection_volume(seq_path, is_bp, manifest_volume=volume_from_khp)

    if volume is None:
        logger.warning("register_calibration: VOLUM DESCONEGUT per %s — no es pot calcular RF_mass", seq_name)
        return {"success": False, "error": "Volume unknown — cannot compute RF_mass",
                "valid_for_calibration": False}
    khp_name = khp_data.get('name', f"KHP{conc}")  # Nom del KHP (ex: "KHP2", "KHP2_50")
    doc_mode = khp_data.get('doc_mode', 'N/A')
    uib_sensitivity = khp_data.get('uib_sensitivity')

    # =========================================================================
    # PARÀMETRES PER SENYAL: Direct (D), UIB (U), 254
    # Veure docs/PARAMETRES_CALIBRACIO.md per definicions
    # =========================================================================

    # --- DIRECT ---
    rf_d = khp_data.get('rf_doc', rf)
    rf_mass_d = khp_data.get('rf_mass_doc', rf_d * 1000 / volume if volume > 0 else 0)  # rf_mass = area*1000/(conc*vol)
    t_max_d = khp_data.get('t_doc_max', khp_data.get('t_retention', 0))
    fwhm_d = khp_data.get('fwhm_doc', 0)
    snr_d = khp_data.get('snr', 0)
    sym_d = khp_data.get('symmetry', 1.0)
    ar_d = khp_data.get('concentration_ratio', khp_data.get('area_ratio', 1.0))
    d254_d = khp_data.get('a254_doc_ratio', 0)

    # --- UIB ---
    area_u = khp_data.get('area_uib', 0)
    area_total_u = khp_data.get('area_total_uib', 0)
    rf_u = khp_data.get('rf_uib', 0)
    if rf_u == 0 and area_u > 0 and conc > 0:
        rf_u = area_u / conc
    rf_mass_u = khp_data.get('rf_mass_uib', rf_u * 1000 / volume if rf_u > 0 and volume > 0 else 0)  # rf_mass = area*1000/(conc*vol)
    t_retention_u = khp_data.get('t_retention_uib', khp_data.get('t_doc_max_uib', 0))
    fwhm_u = khp_data.get('fwhm_uib', 0)
    snr_u = khp_data.get('snr_uib', 0)
    symmetry_u = khp_data.get('symmetry_uib', 1.0)
    ar_u = khp_data.get('area_ratio_uib', khp_data.get('concentration_ratio_uib', 1.0))
    n_peaks_u = khp_data.get('n_peaks_uib', khp_data.get('all_peaks_count_uib', 1))
    shift_sec_u = khp_data.get('shift_sec_uib', khp_data.get('shift_min_uib', 0) * 60)
    shift_min_u = khp_data.get('shift_min_uib', 0)
    d254_u = khp_data.get('a254_doc_ratio_uib', 0)

    # --- 254nm ---
    area_254 = khp_data.get('a254_area', 0)
    if area_254 > 0:
        logger.info("register_calibration: %s area_254=%.2f (QAQC disponible)", seq_name, area_254)
    else:
        logger.warning("register_calibration: %s area_254=0 — QAQC 254nm no disponible", seq_name)
    rf_254 = khp_data.get('rf_254', 0)
    if rf_254 == 0 and area_254 > 0 and conc > 0:
        rf_254 = area_254 / conc
    rf_mass_254 = khp_data.get('rf_mass_254', rf_254 * 1000 / volume if rf_254 > 0 and volume > 0 else 0)  # rf_mass = area*1000/(conc*vol)
    fwhm_254 = khp_data.get('fwhm_254', 0)
    ar_254 = khp_data.get('cr_254', khp_data.get('area_ratio_254', 0))
    t_dad_max = khp_data.get('t_dad_max', 0)  # t_max del senyal 254nm (referència)

    peak_info = khp_data.get('peak_info', {})
    t_retention = khp_data.get('t_retention', khp_data.get('t_doc_max', 0))

    # =========================================================================
    # VALIDACIÓ COMPLETA amb validate_khp_quality()
    # Criteris: multi-pic, timeout, cim irregular, simetria, SNR, límits, CR, històric
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

    if khp_data.get('has_irregular_top', False) and not khp_data.get('irregular_top_repaired', False):
        shift_issues.append("Cim irregular no reparat (shift imprecís)")

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
        "seq_path": os.path.basename(seq_path),  # Relatiu: només nom SEQ
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
        "t_retention": t_max_d,            # Temps pic màxim
        "fwhm_doc": fwhm_d,                # FWHM Direct
        "snr": snr_d,                      # SNR Direct
        "symmetry": sym_d,                 # Simetria
        "area_ratio": ar_d,                # Àrea pic / Àrea total (antic concentration_ratio)
        "n_peaks": khp_data.get('all_peaks_count', 1),  # Nombre de pics (antic all_peaks_count)
        "shift_sec": khp_data.get('shift_sec', khp_data.get('shift_min', 0) * 60),
        "shift_min": khp_data.get('shift_min', 0),
        "bigaussian_doc": khp_data.get('bigaussian_doc'),  # Bigaussian fit DOC

        # =====================================================================
        # PARÀMETRES UIB (U) - Senyal alternatiu
        # =====================================================================
        "area_u": area_u,                  # Àrea pic principal UIB
        "area_total_u": area_total_u,      # Àrea total UIB
        "rf_u": rf_u,                      # Response Factor UIB
        "rf_mass_u": rf_mass_u,            # RF UIB normalitzat per massa (àrea/µg DOC)
        "t_retention_u": t_retention_u,    # Temps pic màxim UIB
        "fwhm_u": fwhm_u,                  # FWHM UIB
        "snr_u": snr_u,                    # SNR UIB
        "symmetry_u": symmetry_u,          # Simetria UIB
        "area_ratio_u": ar_u,              # Àrea pic / Àrea total UIB
        "n_peaks_u": n_peaks_u,            # Nombre de pics UIB
        "shift_sec_u": shift_sec_u,        # Shift temporal UIB (segons)
        "shift_min_u": shift_min_u,        # Shift temporal UIB (minuts)
        "d254_u": d254_u,                  # Ratio DOC/254 amb UIB
        "bigaussian_uib": khp_data.get('bigaussian_uib'),  # Bigaussian fit UIB

        # =====================================================================
        # PARÀMETRES 254nm - Senyal DAD
        # =====================================================================
        "area_254": area_254,              # Àrea pic 254nm
        "rf_254": rf_254,                  # Response Factor 254nm
        "rf_mass_254": rf_mass_254,        # RF 254nm normalitzat per massa (àrea/µg DOC)
        "fwhm_254": fwhm_254,              # FWHM 254nm
        "ar_254": ar_254,                  # Area Ratio 254nm
        "t_dad_max": t_dad_max,            # t_max del 254nm (referència per shift)
        "d254_d": d254_d,                  # Ratio DOC/254 amb Direct
        "bigaussian_254": khp_data.get('bigaussian_254'),  # Bigaussian fit 254nm

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
        "has_irregular_top": khp_data.get('has_irregular_top', False),
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
        areas = [(r.get('area') or 0) for r in replicas]
        shifts = [(r.get('shift_sec') or 0) for r in replicas]
        a254_ratios = [(r.get('a254_doc_ratio') or 0) for r in replicas if (r.get('a254_doc_ratio') or 0) > 0]

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


# =============================================================================

# =============================================================================
# GENERACIÓ D'AVISOS ESTRUCTURATS PER CALIBRACIÓ
# =============================================================================

def _generate_calibration_warnings(result: dict, method: str = "COLUMN") -> list:
    """
    Genera avisos estructurats a partir del resultat de calibració.

    Analitza el resultat i crea avisos amb nivells (BLOCKER/WARNING/INFO)
    segons la jerarquia definida a docs/SISTEMA_AVISOS.md.

    Args:
        result: Dict del resultat de calibrate_from_import()
        method: "COLUMN" o "BP"

    Returns:
        Llista d'avisos estructurats
    """
    warnings = []

    # 1. Errors crítics (BLOCKER)
    if not result.get("success"):
        for error in result.get("errors", []):
            if "no s'han trobat" in error.lower() or "no khp" in error.lower():
                warnings.append(create_warning(
                    code="CAL_NO_KHP",
                    stage="calibrate",
                ))
            elif "invàlid" in error.lower() or "invalid" in error.lower():
                warnings.append(create_warning(
                    code="CAL_ALL_REPLICAS_INVALID",
                    stage="calibrate",
                ))
            else:
                # Error genèric
                warnings.append(create_warning(
                    code="CAL_ERROR",
                    level=WarningLevel.BLOCKER,
                    message=error,
                    stage="calibrate",
                ))

    # 2. Analitzar cada calibració
    for cal in result.get("calibrations", []) + result.get("calibrations_direct", []) + result.get("calibrations_uib", []):
        condition_key = f"{method}_{cal.get('conc_ppm', 0)}_{cal.get('volume_uL', 0)}"

        # Timeout
        if cal.get("timeout_info", {}).get("n_timeouts", 0) > 0:
            severity = cal.get("timeout_info", {}).get("severity", "CRITICAL")
            if severity == "CRITICAL":
                warnings.append(create_warning(
                    code="CAL_TIMEOUT",
                    stage="calibrate",
                    condition_key=condition_key,
                    details={
                        "n_timeouts": cal["timeout_info"]["n_timeouts"],
                        "zones": cal["timeout_info"].get("zone_summary", {}),
                    },
                ))
            elif severity == "WARNING":
                warnings.append(create_warning(
                    code="CAL_TIMEOUT",
                    level=WarningLevel.WARNING,
                    message=f"Timeout detectat (zones no crítiques)",
                    stage="calibrate",
                    condition_key=condition_key,
                ))

        # Cim irregular (jagged/batman)
        irr_top_info = (cal.get("irregular_top_info") or cal.get("anomalies", {}).get("irregular_top")
                        or cal.get("batman_info", {}) or cal.get("anomalies", {}).get("batman", {}))
        if irr_top_info and irr_top_info.get("is_irregular_top", irr_top_info.get("is_batman", False)):
            warnings.append(create_warning(
                code="CAL_IRREGULAR_TOP",
                stage="calibrate",
                condition_key=condition_key,
                details={
                    "valley_depth": irr_top_info.get("max_depth", 0),
                },
            ))

        # RSD alt
        rsd = cal.get("rsd", 0)
        if rsd > 10:
            warnings.append(create_warning(
                code="CAL_RSD_HIGH",
                stage="calibrate",
                condition_key=condition_key,
                details={"rsd": rsd, "threshold": 10},
            ))

        # Quality issues
        for issue in cal.get("calibration_issues", []):
            if "MULTI" in issue.upper():
                warnings.append(create_warning(
                    code="CAL_MULTI_PEAK",
                    stage="calibrate",
                    condition_key=condition_key,
                ))
            elif "SNR" in issue.upper():
                snr = cal.get("snr", 0)
                if "extrema" in issue.lower() or snr < 5:
                    warnings.append(create_warning(
                        code="CAL_SNR_VERY_LOW",
                        stage="calibrate",
                        condition_key=condition_key,
                        details={"snr": snr},
                    ))
                else:
                    warnings.append(create_warning(
                        code="CAL_SNR_LOW",
                        stage="calibrate",
                        condition_key=condition_key,
                        details={"snr": snr},
                    ))
            elif "CR" in issue.upper() or "CONCENTRATION_RATIO" in issue.upper():
                cr = cal.get("concentration_ratio", 0)
                if cr < 0.5:
                    warnings.append(create_warning(
                        code="CAL_CR_VERY_LOW",
                        stage="calibrate",
                        condition_key=condition_key,
                        details={"cr": cr},
                    ))
                else:
                    warnings.append(create_warning(
                        code="CAL_CR_LOW",
                        stage="calibrate",
                        condition_key=condition_key,
                        details={"cr": cr},
                    ))
            elif "INTENSITY" in issue.upper():
                intensity = cal.get("intensity_max", 0)
                if "extreme" in issue.lower():
                    warnings.append(create_warning(
                        code="CAL_INTENSITY_EXTREME",
                        stage="calibrate",
                        condition_key=condition_key,
                        details={"val": intensity},
                    ))
                else:
                    warnings.append(create_warning(
                        code="CAL_INTENSITY_LOW",
                        stage="calibrate",
                        condition_key=condition_key,
                        details={"val": intensity},
                    ))

        # Calibration warnings (INFO level mostly)
        for warn in cal.get("calibration_warnings", []):
            if "ASYMMETRY" in warn.upper() or "SIMETRIA" in warn.upper():
                sym = cal.get("symmetry", 0)
                warnings.append(create_warning(
                    code="CAL_SYMMETRY_LOW",
                    stage="calibrate",
                    condition_key=condition_key,
                    details={"sym": sym},
                ))
            elif "SMOOTHNESS" in warn.upper() or "IRREGULAR" in warn.upper():
                warnings.append(create_warning(
                    code="CAL_SMOOTHNESS_LOW",
                    stage="calibrate",
                    condition_key=condition_key,
                    details={"val": cal.get("smoothness", 0)},
                ))
            elif "EXPANSION" in warn.upper():
                warnings.append(create_warning(
                    code="CAL_EXPANSION_HIGH",
                    stage="calibrate",
                    condition_key=condition_key,
                    details={"pct": 0},  # TODO: extreure de warn
                ))
            elif "HISTORICAL" in warn.upper():
                hist = cal.get("historical_comparison", {})
                dev = hist.get("area_deviation_pct", 0)
                if dev > 15:
                    warnings.append(create_warning(
                        code="CAL_DEVIATION_HIGH",
                        stage="calibrate",
                        condition_key=condition_key,
                        details={"dev": dev},
                    ))
                elif dev > 5:
                    warnings.append(create_warning(
                        code="CAL_DEVIATION_MINOR",
                        stage="calibrate",
                        condition_key=condition_key,
                        details={"dev": dev},
                    ))
                else:
                    # Històric insuficient
                    warnings.append(create_warning(
                        code="CAL_HISTORICAL_INSUFFICIENT",
                        stage="calibrate",
                        condition_key=condition_key,
                    ))

    # 3. Sense KHP local — shift no verificable
    if result.get("khp_source") == "SENSE_KHP":
        warnings.append(create_warning(
            code="CAL_GLOBAL_ONLY",
            stage="calibrate",
            details={"msg": "Sense KHP local — shift no verificable"},
        ))

    # 4. Replicas amb outliers
    for cal in result.get("calibrations", []):
        selection = cal.get("selection", {})
        if selection.get("reason") == "rsd_high" and selection.get("n_replicas_available", 0) > 1:
            # Alguna rèplica exclosa
            selected = selection.get("selected_replicas", [])
            n_available = selection.get("n_replicas_available", 0)
            if len(selected) < n_available:
                for i in range(1, n_available + 1):
                    if i not in selected:
                        warnings.append(create_warning(
                            code="CAL_REPLICA_OUTLIER",
                            stage="calibrate",
                            details={"n": i},
                        ))

    return warnings


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
        - mode: "DUAL", "DIRECT", "UIB" (quins senyals s'han analitzat)
        # Verificació KHP DOC Direct (RF per QC, NO per quantificació):
        - rf_direct: Response Factor Direct local (per comparar amb recta global)
        - shift_direct: Shift temporal Direct vs 254nm (min)
        - khp_area_direct: Àrea KHP amb DOC Direct
        - khp_data_direct: Dades KHP completes per Direct
        # Verificació KHP DOC UIB:
        - rf_uib: Response Factor UIB local (per QC)
        - shift_uib: Shift temporal UIB (min)
        - khp_area_uib: Àrea KHP amb DOC UIB
        - khp_data_uib: Dades KHP completes per UIB
        # Verificació 254nm:
        - rf_254: Response Factor 254nm
        # Principal (usa Direct si disponible, sino UIB):
        - rf: Response Factor principal local (per QC, Direct > UIB)
        - khp_area: Àrea principal
        - khp_conc: Concentració del KHP (ppm)
        - khp_data: Dades KHP principals
        - khp_source: "LOCAL" (amb KHP) o "SENSE_KHP" (sense KHP)
        - qc_results: Validació KHP vs calibració global
        - errors: Llista d'errors

    NOTA: La quantificació de mostres usa SEMPRE la calibració global
    (rf_mass_cal + intercept de Calibration_Reference.json).
    El RF local calculat aquí serveix NOMÉS per verificació QC.
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
        # Sense KHP local: shift temporal no verificable.
        # La quantificació usa sempre la calibració global (rf_mass_cal).
        # El KHP local serveix per: (1) verificar shift DOC-DAD, (2) QC vs recta global.
        report_progress(15, "Sense KHP local — shift no verificable")
        result["khp_source"] = "SENSE_KHP"
        result["success"] = True
        result["warnings"].append("Sense KHP local — shift no verificable, quantificació amb calibració global (shift=0)")
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
            if df_dad is not None:
                logger.debug("calibrate_from_import: %s rep %s té DAD (%s, %d files)",
                             khp_name, rep_num, rep_data.get("dad_source", "?"),
                             len(df_dad))
            else:
                logger.warning("calibrate_from_import: %s rep %s SENSE DAD — area_254 serà 0",
                               khp_name, rep_num)

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
                "seq_path": os.path.basename(seq_path),  # Relatiu: només nom SEQ
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

    # Warnings per concentració i DAD 254
    for khp_result in khp_data_direct_list + khp_data_uib_list:
        kname = khp_result.get('name', 'KHP')
        if khp_result.get('conc_ppm', 0) == 0:
            w = f"⚠️ {kname}: concentració no detectada al nom — RF no calculable"
            if w not in result["warnings"]:
                result["warnings"].append(w)
        if khp_result.get('has_dad') and khp_result.get('a254_area', 0) == 0:
            w = f"⚠️ {kname}: DAD disponible però sense pic 254nm vàlid — 254 no registrat a històric"
            if w not in result["warnings"]:
                result["warnings"].append(w)

    # Propagar uib_sensitivity i doc_mode als resultats KHP
    uib_sensitivity = imported_data.get("uib_sensitivity")
    for entry in khp_data_direct_list + khp_data_uib_list:
        if uib_sensitivity is not None:
            entry['uib_sensitivity'] = uib_sensitivity

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
                if (direct and direct.get("t") is not None) or (uib and uib.get("t") is not None):
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
                except (ValueError, TypeError) as e:
                    logger.debug("Pearson profiles calculation failed: %s", e)

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
                    cal['rf'] = cal['area'] / conc  # RF tradicional (àrea/ppm)
                    cal['rf_mass'] = cal['area'] * 1000 / (conc * vol)  # RF normalitzat per massa (àrea/µg DOC)
                    # Crear camp específic segons doc_source (Direct o UIB)
                    doc_source = cal.get('doc_source', 'direct')
                    if doc_source == 'uib':
                        cal['rf_mass_uib'] = cal['rf_mass']
                    else:
                        cal['rf_mass_doc'] = cal['rf_mass']
                else:
                    # Sense concentració: shift i àrees sí, RF no
                    cal['rf'] = 0.0
                    cal['rf_mass'] = 0.0
                    cal['no_conc'] = True
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
        areas_original = [r.get('area_original') or r['area'] for r in replicas]
        shifts = [r['shift_min'] for r in replicas]
        mean_area = float(np.mean(areas))
        mean_area_original = float(np.mean(areas_original))
        std_area = float(np.std(areas))
        mean_shift = float(np.mean(shifts))
        rsd = float((std_area / mean_area) * 100.0) if mean_area > 0 else 100.0

        # Mètriques addicionals (promig)
        a254_ratios = [(r.get('a254_doc_ratio') or 0) for r in replicas if (r.get('a254_doc_ratio') or 0) > 0]
        mean_a254_ratio = float(np.mean(a254_ratios)) if a254_ratios else 0.0
        a254_areas = [(r.get('a254_area') or 0) for r in replicas if (r.get('a254_area') or 0) > 0]
        mean_a254_area = float(np.mean(a254_areas)) if a254_areas else 0.0
        shift_secs = [(r.get('shift_sec') or 0) for r in replicas if (r.get('shift_sec') or 0) != 0]
        mean_shift_sec = float(np.mean(shift_secs)) if shift_secs else 0.0

        # Mètriques de qualitat (promig de rèpliques)
        snrs = [(r.get('snr') or 0) for r in replicas if (r.get('snr') or 0) > 0]
        mean_snr = float(np.mean(snrs)) if snrs else 0.0
        t_retentions = [(r.get('t_retention') or r.get('t_doc_max') or 0) for r in replicas]
        t_retentions = [t for t in t_retentions if t and t > 0]
        mean_t_retention = float(np.mean(t_retentions)) if t_retentions else 0.0
        fwhms = [(r.get('fwhm_doc') or 0) for r in replicas if (r.get('fwhm_doc') or 0) > 0]
        mean_fwhm = float(np.mean(fwhms)) if fwhms else 0.0
        symmetries = [(r.get('symmetry') or 1.0) for r in replicas if (r.get('symmetry') or 0) > 0]
        mean_symmetry = float(np.mean(symmetries)) if symmetries else 1.0
        volumes = [(r.get('volume_uL') or 0) for r in replicas if (r.get('volume_uL') or 0) > 0]
        volume_uL = int(volumes[0]) if volumes else 100
        t_dad_maxs = [(r.get('t_dad_max') or 0) for r in replicas if (r.get('t_dad_max') or 0) > 0]
        mean_t_dad_max = float(np.mean(t_dad_maxs)) if t_dad_maxs else 0.0

        # Anomalies i qualitat (propagar de rèpliques: pitjor cas)
        quality_scores = [r.get('quality_score', 0) for r in replicas]
        max_quality_score = max(quality_scores) if quality_scores else 0
        all_quality_issues = []
        for r in replicas:
            for issue in r.get('quality_issues', []):
                if issue not in all_quality_issues:
                    all_quality_issues.append(issue)
        group_has_irregular_top = any(r.get('has_irregular_top', False) for r in replicas)
        group_has_irregular = any(r.get('has_irregular', False) for r in replicas)
        group_has_timeout = any(r.get('has_timeout', False) for r in replicas)
        group_smoothness = min((r.get('smoothness', 100.0) for r in replicas), default=100.0)
        group_irregular_top_repaired = any(r.get('irregular_top_repaired', False) for r in replicas)

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
            selected_area_original = mean_area_original
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
            selected_area_original = selected_rep.get('area_original') or selected_rep['area']
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
            selected_area_original = best.get('area_original') or best['area']
            selected_shift_min = best['shift_min']
            selected_shift_sec = best.get('shift_sec', 0)
            selected_a254_ratio = best.get('a254_doc_ratio', 0)
            selected_a254_area = best.get('a254_area', 0)
            selected_replicas = [best.get('replica_num', 1)]
            status = f"Millor qualitat R{selected_replicas[0]} (RSD {rsd:.1f}%)"

        # Bigaussian: agafar de la primera rèplica seleccionada (o millor qualitat)
        # Sempre propagar, encara que sigui INVALID (info QC valuosa)
        bg_source = replicas[0]
        if len(selected_replicas) == 1:
            bg_source = next((r for r in replicas if r.get('replica_num') == selected_replicas[0]), replicas[0])
        bigaussian_doc = bg_source.get('bigaussian_doc')
        bigaussian_uib = bg_source.get('bigaussian_uib')
        bigaussian_254 = bg_source.get('bigaussian_254')

        return {
            # Valors seleccionats
            'name': group_name,  # Nom del KHP (ex: "KHP2", "KHP2_50")
            'name_full': f"KHP{group_conc}@{group_volume}µL",  # Condicions: conc + volum
            'conc_ppm': group_conc,
            'area': selected_area,
            'area_original': selected_area_original if selected_area_original != selected_area else None,
            'area_repaired': selected_area if selected_area_original != selected_area else None,
            'shift_min': selected_shift_min,
            'shift_sec': selected_shift_sec,
            'a254_doc_ratio': selected_a254_ratio,
            'a254_area': selected_a254_area,
            'is_bp': replicas[0].get('is_bp', False),
            'doc_source': replicas[0].get('doc_source', 'direct'),  # 'direct' o 'uib'

            # Mètriques de qualitat (promig de rèpliques)
            'snr': mean_snr,
            't_retention': mean_t_retention,
            't_doc_max': mean_t_retention,
            't_dad_max': mean_t_dad_max,  # t_max del 254nm (referència)
            'fwhm_doc': mean_fwhm,
            'symmetry': mean_symmetry,
            'volume_uL': group_volume,  # Volum d'aquest grup

            # Bigaussian fit (sempre guardat, INVALID = info QC)
            'bigaussian_doc': bigaussian_doc,
            'bigaussian_uib': bigaussian_uib,
            'bigaussian_254': bigaussian_254,

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

            # Anomalies i qualitat (pitjor cas de totes les rèpliques)
            'quality_score': max_quality_score,
            'quality_issues': all_quality_issues,
            'has_irregular_top': group_has_irregular_top,
            'has_irregular': group_has_irregular,
            'has_timeout': group_has_timeout,
            'timeout_severity': max(
                (r.get('timeout_severity', 'OK') for r in replicas),
                key=lambda s: {'OK': 0, 'INFO': 1, 'WARNING': 2, 'CRITICAL': 3}.get(s, 0),
                default='OK'
            ),
            'smoothness': group_smoothness,
            'irregular_top_repaired': group_irregular_top_repaired,

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

    report_progress(80, "Verificant KHP vs calibració global...")

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

    # Comprovar si conc desconeguda
    has_unknown_conc = result.get("khp_conc", 0) == 0
    if has_unknown_conc:
        result["warnings"].append("KHP sense concentració coneguda: no es pot calcular RF ni comparar amb recta global")
        result["no_conc"] = True
    elif result["rf"] == 0:
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

    # Enriquir calibracions Direct amb dades UIB (perquè register_calibration les guardi)
    calibrations_list = result.get("calibrations", [])
    calibrations_uib_reg = result.get("calibrations_uib", [])
    if calibrations_uib_reg and calibrations_list:
        uib_primary = calibrations_uib_reg[0]
        for cal_data in calibrations_list:
            cal_data['area_uib'] = uib_primary.get('area', 0)
            cal_data['area_total_uib'] = uib_primary.get('area_total', 0)
            cal_data['rf_uib'] = uib_primary.get('rf_doc', uib_primary.get('rf', 0))
            cal_data['rf_mass_uib'] = uib_primary.get('rf_mass_doc', uib_primary.get('rf_mass', 0))
            cal_data['t_retention_uib'] = uib_primary.get('t_retention', 0)
            cal_data['t_doc_max_uib'] = uib_primary.get('t_doc_max', uib_primary.get('t_retention', 0))
            cal_data['fwhm_uib'] = uib_primary.get('fwhm_doc', 0)
            cal_data['snr_uib'] = uib_primary.get('snr', 0)
            cal_data['symmetry_uib'] = uib_primary.get('symmetry', 1.0)
            cal_data['concentration_ratio_uib'] = uib_primary.get('concentration_ratio', 1.0)
            cal_data['shift_sec_uib'] = uib_primary.get('shift_sec', 0)
            cal_data['shift_min_uib'] = uib_primary.get('shift_min', 0)
            cal_data['doc_mode'] = result.get('mode', 'DUAL')
            cal_data['bigaussian_uib'] = uib_primary.get('bigaussian_doc')

    # Registrar TOTES les calibracions (una per cada condició)
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

    report_progress(95, "Validant KHP vs calibracio global...")

    # =========================================================================
    # VALIDACIÓ QC: Comparar KHP local vs calibració global
    # =========================================================================
    qc_results = []

    # Obtenir data SEQ per seleccionar calibració correcta
    seq_date = imported_data.get("seq_date") or imported_data.get("date")
    if not seq_date and seq_path:
        # Intentar obtenir del manifest
        manifest_path = os.path.join(seq_path, "CHECK", "import_manifest.json")
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, 'r', encoding='utf-8') as f:
                    manifest = json.load(f)
                    seq_date = manifest.get("seq_date") or manifest.get("date")
            except Exception as e:
                logger.warning("Could not load manifest for seq_date: %s", e)

    # QC per DOC Direct
    if result.get("khp_data_direct"):
        khp_data = result["khp_data_direct"]
        qc = validate_khp_qc(khp_data, seq_date=seq_date, signal='direct', mode=method.lower())
        qc['signal'] = 'direct'
        qc_results.append(qc)

        # Afegir info QC al resultat
        result["qc_direct"] = qc

        # Registrar a QC_History
        if qc.get("status") != "UNKNOWN":
            seq_name = os.path.basename(seq_path) if seq_path else "UNKNOWN"
            register_qc_result(seq_name, seq_date, qc, khp_data)

    # QC per DOC UIB
    if result.get("khp_data_uib"):
        khp_data = result["khp_data_uib"]
        qc = validate_khp_qc(khp_data, seq_date=seq_date, signal='uib', mode=method.lower())
        qc['signal'] = 'uib'
        qc_results.append(qc)

        # Afegir info QC al resultat
        result["qc_uib"] = qc

        # Registrar a QC_History
        if qc.get("status") != "UNKNOWN":
            seq_name = os.path.basename(seq_path) if seq_path else "UNKNOWN"
            register_qc_result(seq_name, seq_date, qc, khp_data)

    result["qc_results"] = qc_results

    # Afegir warning si QC no passa
    for qc in qc_results:
        if qc.get("status") == "WARNING":
            result["warnings"].append(f"⚠️ QC {qc.get('signal', '').upper()}: {qc.get('message', '')}")
        elif qc.get("status") == "FAIL":
            result["warnings"].append(f"❌ QC {qc.get('signal', '').upper()}: {qc.get('message', '')}")

    report_progress(100, "Calibracio completada")

    result["success"] = True

    # Generar avisos estructurats (nou sistema)
    result["warnings_structured"] = _generate_calibration_warnings(result, method)
    result["warning_level"] = get_max_warning_level(result["warnings_structured"])

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
