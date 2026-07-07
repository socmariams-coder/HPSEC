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
import hashlib
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

# =============================================================================
# CLASSIFICACIÓ DE SECCIONS PER IMPACTE
# =============================================================================

# Seccions que afecten resultats ja processats (requereixen reprocessament)
REPROCESS_SECTIONS = frozenset([
    "detection", "quality", "baseline", "chromatogram",
    "time_fractions", "timeout_zones", "wavelengths", "warnings", "dad",
])

# Seccions que només afecten el proper processament
FUTURE_SECTIONS = frozenset([
    "calibration", "blank_injections", "control_injections",
    "sample_types", "sequence", "injection_volumes",
])

# Seccions amb efecte immediat (UI, paths)
IMMEDIATE_SECTIONS = frozenset(["paths", "ui"])


# =============================================================================
# CONFIGURACIÓ PER DEFECTE
# =============================================================================

DEFAULT_CONFIG = {
    # --- PATHS ---
    "paths": {
        "data_folders": [],       # Llista de carpetes amb SEQs
        "registry_folder": "",    # Carpeta REGISTRY compartit (explícit o derivat de 1a carpeta)
    },

    # --- FRACCIONS TEMPORALS (min) ---
    # Definides NOMÉS a hpsec_config.json (no hardcoded aquí)
    "time_fractions": {},

    # --- CROMATOGRAMA ---
    "chromatogram": {
        "max_duration_min": 78.65,
        "baseline_window_bp": 1.0,
        "baseline_window_column": 10.0,
        "smoothing_window": 11,
        "smoothing_order": 3,
    },

    # --- CÀLCUL BASELINE ---
    # Correcció de baseline: zona del cromatograma a usar
    "baseline": {
        # BP: usar FINAL del cromatograma (després del pic)
        "bp_end_pct": 20,           # Últim 20% del cromatograma
        # COLUMN: usar INICI del cromatograma (abans dels pics)
        "column_start_pct": 15,     # Primer 15% (~10 min en run de 70 min)
        # Mètode de càlcul
        "method": "mode",           # "mode" (robust) o "median"
        # Paràmetres per estadístiques (SNR, etc.)
        "stats_percentile_low": 5,  # Percentil baix per excloure outliers
        "stats_percentile_high": 40, # Percentil alt per zona "baixa"
        # Soroll mínim instrumental (mAU)
        "min_noise_mau": 0.01,
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
        # Marge pre-injecció per assignació TOC→HPLC (minuts)
        # El pic DOC s'eixampla per dispersió al reactor TOC: la pujada comença
        # ABANS de l'hora d'injecció HPLC. Aquest marge permet assignar files TOC
        # amb temps negatiu (fins a -X min) a la injecció correcta.
        "toc_pre_margin_min": 1.5,        # BP: dispersió reactor TOC
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
        "irregular_top_max_sep_min": 0.5,  # formerly batman_max_sep_min
        "irregular_top_drop_min": 0.05,  # formerly batman_drop_min
        "irregular_top_drop_max": 0.50,  # formerly batman_drop_max
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
        "guided_search_window_min": 2.5,  # Finestra ±min cerca dirigida 254nm→DOC
    },

    # --- INJECCIONS BLANC ---
    # Patrons de nom que identifiquen injeccions blanc (aigua MQ, etc.)
    "blank_injections": {
        "patterns": ["MQ", "BLANK", "BLK", "H2O", "WATER", "MILLI", "BLANC"],
    },

    # --- INJECCIONS CONTROL ---
    # Patrons de nom que identifiquen injeccions de control (neteja, verificació)
    # Aquestes injeccions poden tenir múltiples rèpliques/blocs i NO s'han de considerar orfes
    "control_injections": {
        "patterns": ["NAOH", "WASH", "CONTROL"],
        "ignore_orphan": True,  # No marcar com a orfes els controls no trobats a 1-HPLC-SEQ
    },

    # --- TIMEOUT TOC ---
    "timeout_zones": {
        "RUN_START": {"start": 0, "end": 1, "severity": "INFO"},
        "BioP": {"start": 0, "end": 18, "severity": "WARNING"},
        "HS": {"start": 18, "end": 23, "severity": "CRITICAL"},
        "BB": {"start": 23, "end": 26, "severity": "WARNING"},
        "SB": {"start": 26, "end": 32, "severity": "INFO"},
        "LMW": {"start": 32, "end": 70, "severity": "INFO"},
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
                config = self._merge_configs(DEFAULT_CONFIG, saved_config)
                return self._migrate_config(config)
            except Exception as e:
                logger.error(f"Error carregant configuració: {e}")
                return DEFAULT_CONFIG.copy()
        return DEFAULT_CONFIG.copy()

    def _migrate_config(self, config):
        """Migra claus obsoletes a les noves."""
        # data_folder (string) → data_folders (list)
        paths = config.get("paths", {})
        if "data_folder" in paths:
            old_folder = paths.pop("data_folder")
            if old_folder and not paths.get("data_folders"):
                paths["data_folders"] = [old_folder]
            config["paths"] = paths

        det = config.get("detection", {})
        # batman_max_sep → batman_max_sep_min → irregular_top_max_sep_min
        if "batman_max_sep" in det:
            if "irregular_top_max_sep_min" not in det:
                det["irregular_top_max_sep_min"] = det["batman_max_sep"]
            del det["batman_max_sep"]
        if "batman_max_sep_min" in det:
            if "irregular_top_max_sep_min" not in det:
                det["irregular_top_max_sep_min"] = det["batman_max_sep_min"]
            del det["batman_max_sep_min"]
        # batman_drop_min → irregular_top_drop_min
        if "batman_drop_min" in det:
            if "irregular_top_drop_min" not in det:
                det["irregular_top_drop_min"] = det["batman_drop_min"]
            del det["batman_drop_min"]
        # batman_drop_max → irregular_top_drop_max
        if "batman_drop_max" in det:
            if "irregular_top_drop_max" not in det:
                det["irregular_top_drop_max"] = det["batman_drop_max"]
            del det["batman_drop_max"]
        return config

    def compute_config_fingerprint(self):
        """SHA-256 de les seccions que afecten el processament. Returns hex[:16]."""
        data = {s: self.config.get(s, {}) for s in sorted(REPROCESS_SECTIONS)}
        raw = json.dumps(data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]

    def _merge_configs(self, default, saved):
        """Fusiona configuració guardada amb defaults (recursiu)."""
        result = default.copy()
        for key, value in saved.items():
            if key in result:
                if isinstance(value, dict) and isinstance(result[key], dict):
                    result[key] = self._merge_configs(result[key], value)
                else:
                    result[key] = value
            else:
                # Claus noves del fitxer guardat (no existeixen als defaults)
                result[key] = value
        return result

    def save(self):
        """Guarda la configuració actual al fitxer."""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Error guardant configuració: {e}")
            return False

    def reload(self):
        """Rellegeix la configuració des del fitxer."""
        self.config = self._load_config()

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

    def get_all_fractions(self, mode=None):
        """Retorna totes les fraccions temporals ordenades.

        Args:
            mode: "COLUMN", "BP" o None (default COLUMN).
              BP no té fraccions definides, retorna llista buida.
        """
        if mode and mode.upper() == "BP":
            return []
        fractions = self.get("time_fractions", default={})
        return sorted(fractions.items(), key=lambda x: x[1]["start"])

    def get_subzones(self, parent_name):
        """Retorna les sub-zones d'una fracció principal (ordenades).

        Args:
            parent_name: nom de la fracció principal (HS, BB, etc.)

        Returns:
            list[(subzone_name, start, end)]. Llista buida si no té sub-zones.
        """
        fractions = self.get("time_fractions", default={})
        parent = fractions.get(parent_name) or {}
        subs = parent.get("subzones") or {}
        items = [(name, info.get("start"), info.get("end"))
                 for name, info in subs.items()]
        return sorted(items, key=lambda x: x[1] if x[1] is not None else 0)

    def get_all_subzones(self, mode=None):
        """Retorna totes les sub-zones de totes les fraccions, indexades pel
        nom de la subzona.

        Returns:
            dict {subzone_name: (parent_name, start, end)}.
            Ordenats per inici. Buit si mode=BP.
        """
        if mode and mode.upper() == "BP":
            return {}
        fractions = self.get("time_fractions", default={})
        result = {}
        items = []
        for parent_name, parent in fractions.items():
            for sub_name, sub in (parent.get("subzones") or {}).items():
                items.append((sub_name, parent_name,
                              sub.get("start"), sub.get("end")))
        items.sort(key=lambda x: x[2] if x[2] is not None else 0)
        return {name: (parent, s, e) for name, parent, s, e in items}

    def get_selected_wavelengths(self):
        """Retorna les wavelengths seleccionades."""
        return self.get("wavelengths", "selected", default=[254])

    def get_section(self, section_name):
        """
        Obté una secció completa de configuració.

        Args:
            section_name: Nom de la secció (ex: "samples_db")

        Returns:
            Dict amb la secció o None si no existeix
        """
        return self.config.get(section_name)

    def set_section(self, section_name, section_data):
        """
        Estableix una secció completa de configuració.

        Args:
            section_name: Nom de la secció
            section_data: Dict amb les dades de la secció
        """
        self.config[section_name] = section_data

    def export_config(self, filepath):
        """Exporta la configuració a un fitxer."""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Error exportant: {e}")
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
            logger.error(f"Error important: {e}")
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

def get_data_folders():
    """Retorna llista de carpetes de dades configurades."""
    cfg = get_config()
    folders = cfg.get("paths", "data_folders") or []
    # Backward compat: si data_folder existeix (string antic), convertir
    if not folders:
        single = cfg.get("paths", "data_folder")
        if single:
            folders = [single]
    return [f for f in folders if f]


def get_data_folder():
    """Retorna la primera carpeta de dades (per REGISTRY si no configurat explícitament)."""
    folders = get_data_folders()
    return folders[0] if folders else ""

def get_sample_duration(mode="COLUMN"):
    """Durada per mostra (crom + post-run) en minuts, segons mode.

    Font única per als valors de sequence.sample_duration_* (evitar
    hardcodejar 78.65/12.0 pels mòduls).
    """
    cfg = get_config()
    if str(mode).upper() == "BP":
        return float(cfg.get("sequence", "sample_duration_bp", default=12.0))
    return float(cfg.get("sequence", "sample_duration_column", default=78.65))


def get_toc_cycle_min():
    """Cicle de recàrrega de xeringa del TOC (minuts). Font única."""
    return float(get_config().get("sequence", "toc_cycle_min", default=77.2))


def get_toc_timeout_sec():
    """Durada del timeout del TOC (segons). Font única."""
    return float(get_config().get("sequence", "toc_timeout_sec", default=74))


def get_registry_path():
    """
    Obté la carpeta REGISTRY per JSONs globals (KHP_History, Samples_History).
    Si registry_folder configurat explícitament, usar-lo.
    Si no, derivar de la primera carpeta de dades + /REGISTRY.
    La crea si no existeix.
    """
    cfg = get_config()
    registry = cfg.get("paths", "registry_folder")
    if not registry:
        data_folder = get_data_folder()
        if data_folder:
            registry = os.path.join(data_folder, "REGISTRY")
    if registry:
        os.makedirs(registry, exist_ok=True)
        return registry
    return None

def save_config(cfg=None):
    """
    Guarda la configuració actual.

    Args:
        cfg: Instància ConfigManager (o usa global si no es proporciona)

    Returns:
        True si s'ha guardat correctament
    """
    if cfg is None:
        cfg = get_config()
    return cfg.save()


if __name__ == "__main__":
    # Test
    cfg = get_config()
    print("Configuració carregada:")
    print(f"  R² VALID: {cfg.get('quality', 'r2_valid')}")
    print(f"  Wavelengths: {cfg.get_selected_wavelengths()}")
    print(f"  Fraccions: {[f[0] for f in cfg.get_all_fractions()]}")
