"""
Shared constants for analyze_panel package.

Derives fraction definitions and wavelengths from hpsec_config.
Anomaly severity classification is defined here (not configurable).
"""

from hpsec_config import get_config


# ---------------------------------------------------------------------------
# Anomaly severity classification (not configurable)
# ---------------------------------------------------------------------------
CRITICAL_ANOMALIES = {"BATMAN_DIRECT", "BATMAN_UIB", "TIMEOUT_IN_PEAK", "NO_PEAK"}
WARNING_ANOMALIES = {"LOW_SNR", "BASELINE_MISSING", "ASYMMETRIC_PEAK", "BELOW_LOD", "BELOW_LOQ"}
# Everything else (UIB_NO_BASELINE, EARS_DETECTED, etc.) → INFO


# ---------------------------------------------------------------------------
# Fractions from config
# ---------------------------------------------------------------------------
def _load_fractions():
    cfg = get_config()
    fracs = cfg.get_all_fractions()  # sorted by start
    names = [name for name, _ in fracs]
    ranges = {name: f"{info['start']:g}-{info['end']:g}" for name, info in fracs}
    return names, ranges


FRACTION_NAMES, FRACTION_RANGES = _load_fractions()


# ---------------------------------------------------------------------------
# Wavelengths from config
# ---------------------------------------------------------------------------
def _load_wavelengths():
    cfg = get_config()
    selected = cfg.get_selected_wavelengths()  # e.g. [220, 254, 272, 290, 362]
    primary = cfg.get("wavelengths", "primary", default=254)

    # Main: exclude 362 for compact panel view, primary wavelength first
    main_nums = [wl for wl in selected if wl != 362]
    if primary in main_nums:
        main_nums.remove(primary)
        main_nums.insert(0, primary)
    main = [f"A{wl}" for wl in main_nums]

    # All: include A252 (UV alternative) + all selected, primary first
    all_nums = sorted(set(selected + [252]))
    if primary in all_nums:
        all_nums.remove(primary)
        all_nums.insert(0, primary)
    all_wl = [f"A{wl}" for wl in all_nums]

    return main, all_wl


DAD_WL_MAIN, DAD_WL_ALL = _load_wavelengths()


# ---------------------------------------------------------------------------
# Signal keys for summary tables
# ---------------------------------------------------------------------------
SIGNAL_KEYS_MAIN = [
    ("DOC Direct", "DOC"),
    ("DOC UIB", "UIB"),
] + [(wl, wl) for wl in DAD_WL_MAIN]

SIGNAL_KEYS_ALL = [
    ("DOC Direct", "DOC"),
    ("DOC UIB", "UIB"),
] + [(wl, wl) for wl in DAD_WL_ALL]
