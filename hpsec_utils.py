"""
hpsec_utils.py
==============
Funcions utilitàries compartides entre els scripts HPSEC.

Conté:
- NumpyEncoder: encoder JSON per tipus numpy/pandas (font única)
- _atomic_write_json: escriptura JSON atòmica temp+fsync+os.replace (font única)
- seleccionar_carpeta: GUI per selecció de carpeta
- t_at_max: Temps al màxim

NOTA (2026-02-03): Les funcions de baseline s'han mogut a hpsec_core.py:
  - baseline_stats, baseline_stats_windowed
  - get_baseline_value, get_baseline_stats
  - mode_robust
  Importar des de hpsec_core en lloc d'aquí.

NOTA: detect_main_peak i detect_irregular_top (formerly detect_batman) s'han mogut a hpsec_core.py (2026-01-29)
NOTA: obtenir_seq, is_khp, extract_khp_conc, normalize_key s'han mogut a hpsec_import.py (2026-01-29)
"""

import json
import os

import numpy as np
import pandas as pd

# Re-exportar funcions de baseline des de hpsec_core per compatibilitat enrere
from hpsec_core import (
    baseline_stats,
    baseline_stats_windowed,
    get_baseline_value,
    get_baseline_stats,
    mode_robust,
)


# =============================================================================
# JSON: ENCODER NUMPY/PANDAS + ESCRIPTURA ATÒMICA
# =============================================================================
class NumpyEncoder(json.JSONEncoder):
    """Encoder JSON per tipus numpy i pandas."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="list")
        if pd.isna(obj):
            return None
        return super().default(obj)


def _atomic_write_json(path, data, **dump_kwargs):
    """Escriu un JSON de forma atòmica: temp al mateix directori + os.replace.

    Evita deixar el fitxer bo corromput o a mitges si l'escriptura falla (disc
    ple, lock OneDrive/Excel, crash): el fitxer de destí o queda intacte (vell)
    o passa a ser el nou complet, mai un estat intermedi.
    """
    import tempfile
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix='.tmp_', suffix='.json')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(data, f, **dump_kwargs)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)  # atòmic dins el mateix sistema de fitxers
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


# =============================================================================
# GUI - SELECCIÓN DE CARPETA
# =============================================================================
def seleccionar_carpeta(titulo="Selecciona carpeta SEQ"):
    """
    Muestra diálogo para seleccionar carpeta.

    Args:
        titulo: Título del diálogo

    Returns:
        str con la ruta de la carpeta seleccionada, o string vacío si se cancela
    """
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    folder = filedialog.askdirectory(title=titulo)
    root.destroy()
    return folder


# =============================================================================
# ALTRES UTILITATS
# =============================================================================
def t_at_max(t, y):
    """
    Obtiene el tiempo correspondiente al valor máximo.

    Args:
        t: Array de tiempos
        y: Array de valores

    Returns:
        float con el tiempo del máximo, o None si no es válido
    """
    if t is None or y is None or len(t) == 0 or len(y) == 0:
        return None
    if len(t) < 10 or len(y) < 10:
        return None
    i = int(np.nanargmax(y))
    return float(t[i])
