"""
hpsec_utils.py
==============
Funcions utilitàries compartides entre els scripts HPSEC.

Conté:
- seleccionar_carpeta: GUI per selecció de carpeta
- t_at_max: Temps al màxim

NOTA (2026-02-03): Les funcions de baseline s'han mogut a hpsec_core.py:
  - baseline_stats, baseline_stats_windowed
  - get_baseline_value, get_baseline_stats
  - mode_robust
  Importar des de hpsec_core en lloc d'aquí.

NOTA: detect_main_peak i detect_batman s'han mogut a hpsec_core.py (2026-01-29)
NOTA: obtenir_seq, is_khp, extract_khp_conc, normalize_key s'han mogut a hpsec_import.py (2026-01-29)
"""

import numpy as np

# Re-exportar funcions de baseline des de hpsec_core per compatibilitat enrere
from hpsec_core import (
    baseline_stats,
    baseline_stats_windowed,
    get_baseline_value,
    get_baseline_stats,
    mode_robust,
)


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
