# -*- coding: utf-8 -*-
"""
hpsec_config.py - Gestió centralitzada de configuració HPSEC Suite
==================================================================

Centralitza tots els paràmetres configurables en un sol lloc.
Permet guardar/carregar configuracions personalitzades.

Autor: HPSEC Suite
Versió: 1.0
"""

import os
import json
from pathlib import Path
from datetime import datetime

# =============================================================================
# CONFIGURACIÓ PER DEFECTE
# =============================================================================

DEFAULT_CONFIG = {
    # --- PATHS ---
    "paths": {
        "data_folder": "C:/Users/Lequia/Desktop/Dades2",  # Carpeta base amb les SEQs
    },

    # --- FRACCIONS TEMPORALS (min) ---
    "time_fractions": {
        "BioP": {"start": 0.0, "end": 18.0, "name": "Biopolímers"},
        "HS": {"start": 18.0, "end": 23.0, "name": "Àcids Húmics"},
        "BB": {"start": 23.0, "end": 30.0, "name": "Building Blocks"},
        "SB": {"start": 30.0, "end": 40.0, "name": "Small Building Blocks"},
        "LMW": {"start": 40.0, "end": 70.0, "name": "Low Molecular Weight"},
    },

    # --- CROMATOGRAMA ---
    "chromatogram": {
        "max_duration_min": 78.65,
        "baseline_window_bp": 1.0,
        "baseline_window_column": 10.0,
        "smoothing_window": 11,
        "smoothing_order": 3,
    },

    # --- PLANIFICACIÓ SEQÜÈNCIES ---
    "sequence": {
        # Durada per mostra (cromatograma + post-run) en minuts
        "sample_duration_column": 78.65,  # COLUMN mode (70 min crom + post-run)
        "sample_duration_bp": 12.0,       # BP mode (bypass, pic al principi)
        # Flux fase mòbil (mL/min)
        "flow_rate_column": 0.75,         # COLUMN mode
        "flow_rate_bp": 0.75,             # BP mode
        # Cicle TOC
        "toc_cycle_min": 77.2,            # Cicle recàrrega xeringa TOC
        "toc_timeout_sec": 74,            # Duració timeout
    },

    # --- THRESHOLDS QUALITAT ---
    "quality": {
        "r2_valid": 0.987,
        "r2_check": 0.980,
        "pearson_min": 0.995,
        "pearson_warning": 0.990,
        "snr_min": 10.0,
        "snr_ratio_threshold": 1.5,
        "area_diff_warning": 15.0,  # %
        "area_diff_critical": 30.0,  # %
    },

    # --- DAD ---
    "dad": {
        "drift_warning": 1.0,  # mAU
        "drift_critical": 3.0,  # mAU
        "noise_warning": 0.5,  # mAU
        "doc_correlation_min": 0.90,
        "doc_correlation_warning": 0.95,
    },

    # --- WAVELENGTHS ---
    "wavelengths": {
        "selected": [220, 254, 272, 290, 362],
        "available": [210, 220, 230, 240, 250, 254, 260, 272, 280, 290, 300, 350, 362, 400],
        "primary": 254,  # Wavelength principal per visualització
    },

    # --- VOLUMS INJECCIÓ (referència, es llegeixen dels fitxers) ---
    "injection_volumes": {
        "bp_default": 100,  # µL
        "column_default": 400,  # µL
        "column_old": 100,  # µL (SEQ < 275)
        "column_change_seq": 275,
    },

    # --- DETECCIÓ ANOMALIES ---
    "detection": {
        "batman_max_sep_min": 0.5,
        "batman_drop_min": 0.05,
        "batman_drop_max": 0.50,
        "timeout_min_duration": 5.0,  # segons
        "timeout_major": 74.0,  # segons (recàrrega xeringa)
        "ears_threshold": 0.10,  # 10% height
        "ears_max_sep_min": 0.5,
        "irr_smoothness_threshold": 0.18,  # 18%
    },

    # --- CALIBRACIÓ ---
    "calibration": {
        "khp_pattern": "KHP",
        "peak_min_prominence_pct": 5.0,
        "symmetry_min": 0.5,
        "symmetry_max": 2.0,
        "snr_min_khp": 50.0,
    },

    # --- INJECCIONS CONTROL ---
    # Patrons de nom que identifiquen injeccions de control
    # Aquestes injeccions poden tenir múltiples rèpliques/blocs i NO s'han de considerar orfes
    "control_injections": {
        "patterns": ["MQ", "NAOH", "BLANK", "H2O", "WASH"],
        "ignore_orphan": True,  # No marcar com a orfes els controls no trobats a 1-HPLC-SEQ
    },

    # --- TIMEOUT TOC ---
    "timeout_zones": {
        "RUN_START": {"start": 0, "end": 1, "severity": "INFO"},
        "BioP": {"start": 0, "end": 18, "severity": "WARNING"},
        "HS": {"start": 18, "end": 23, "severity": "CRITICAL"},
        "BB": {"start": 23, "end": 30, "severity": "WARNING"},
        "SB": {"start": 30, "end": 40, "severity": "WARNING"},
        "LMW": {"start": 40, "end": 70, "severity": "INFO"},
        "POST_RUN": {"start": 70, "end": 999, "severity": "OK"},
    },

    # --- INTERFÍCIE ---
    "ui": {
        "theme": "clam",  # ttk theme: clam, alt, default, classic
        "font_family": "Segoe UI",
        "font_size": 10,
        "accent_color": "#2E86AB",
        "warning_color": "#F6AE2D",
        "error_color": "#E63946",
        "success_color": "#2A9D8F",
    },
}


# =============================================================================
# CLASSE CONFIGMANAGER
# =============================================================================

class ConfigManager:
    """Gestiona la configuració de l'aplicació."""

    CONFIG_FILENAME = "hpsec_config.json"

    def __init__(self, app_folder=None):
        """
        Inicialitza el gestor de configuració.

        Args:
            app_folder: Carpeta de l'aplicació (per defecte, carpeta del script)
        """
        if app_folder is None:
            app_folder = os.path.dirname(os.path.abspath(__file__))

        self.app_folder = app_folder
        self.config_path = os.path.join(app_folder, self.CONFIG_FILENAME)
        self.config = self._load_config()

    def _load_config(self):
        """Carrega la configuració des del fitxer o usa valors per defecte."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    saved_config = json.load(f)
                # Merge amb defaults (per si hi ha nous paràmetres)
                return self._merge_configs(DEFAULT_CONFIG, saved_config)
            except Exception as e:
                print(f"Error carregant configuració: {e}")
                return DEFAULT_CONFIG.copy()
        return DEFAULT_CONFIG.copy()

    def _merge_configs(self, default, saved):
        """Fusiona configuració guardada amb defaults (recursiu)."""
        result = default.copy()
        for key, value in saved.items():
            if key in result:
                if isinstance(value, dict) and isinstance(result[key], dict):
                    result[key] = self._merge_configs(result[key], value)
                else:
                    result[key] = value
        return result

    def save(self):
        """Guarda la configuració actual al fitxer."""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error guardant configuració: {e}")
            return False

    def reset_to_defaults(self):
        """Restaura la configuració per defecte."""
        self.config = DEFAULT_CONFIG.copy()
        self.save()

    def get(self, *keys, default=None):
        """
        Obté un valor de configuració.

        Args:
            *keys: Claus niuades (ex: get("quality", "r2_valid"))
            default: Valor per defecte si no existeix

        Returns:
            Valor de configuració o default
        """
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def set(self, *keys_and_value):
        """
        Estableix un valor de configuració.

        Args:
            *keys_and_value: Claus niuades + valor final
                Ex: set("quality", "r2_valid", 0.99)
        """
        if len(keys_and_value) < 2:
            raise ValueError("Cal almenys una clau i un valor")

        keys = keys_and_value[:-1]
        value = keys_and_value[-1]

        # Navegar fins al penúltim nivell
        current = self.config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        # Establir valor
        current[keys[-1]] = value

    def get_time_fraction(self, name):
        """Obté els límits d'una fracció temporal."""
        fractions = self.get("time_fractions", default={})
        if name in fractions:
            return fractions[name]["start"], fractions[name]["end"]
        return None, None

    def get_all_fractions(self):
        """Retorna totes les fraccions temporals ordenades."""
        fractions = self.get("time_fractions", default={})
        return sorted(fractions.items(), key=lambda x: x[1]["start"])

    def get_selected_wavelengths(self):
        """Retorna les wavelengths seleccionades."""
        return self.get("wavelengths", "selected", default=[254])

    def export_config(self, filepath):
        """Exporta la configuració a un fitxer."""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error exportant: {e}")
            return False

    def import_config(self, filepath):
        """Importa configuració des d'un fitxer."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                imported = json.load(f)
            self.config = self._merge_configs(DEFAULT_CONFIG, imported)
            self.save()
            return True
        except Exception as e:
            print(f"Error important: {e}")
            return False


# =============================================================================
# INSTÀNCIA GLOBAL
# =============================================================================

# Instància singleton per ús global
_config_manager = None

def get_config():
    """Obté la instància global del ConfigManager."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


# =============================================================================
# FUNCIONS HELPER
# =============================================================================

def get_threshold(name):
    """Obté un threshold de qualitat."""
    cfg = get_config()
    return cfg.get("quality", name)

def get_detection_param(name):
    """Obté un paràmetre de detecció."""
    cfg = get_config()
    return cfg.get("detection", name)

def get_ui_param(name):
    """Obté un paràmetre d'interfície."""
    cfg = get_config()
    return cfg.get("ui", name)


if __name__ == "__main__":
    # Test
    cfg = get_config()
    print("Configuració carregada:")
    print(f"  R² VALID: {cfg.get('quality', 'r2_valid')}")
    print(f"  Wavelengths: {cfg.get_selected_wavelengths()}")
    print(f"  Fraccions: {[f[0] for f in cfg.get_all_fractions()]}")
