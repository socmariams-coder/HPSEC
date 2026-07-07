"""
Preferències persistents de la GUI (QSettings).

Guarda petites preferències d'ús entre sessions (última carpeta oberta,
últim destí d'exportació...) al registre d'usuari de Windows / fitxer
de config de l'usuari. No conté cap paràmetre científic: això viu a
hpsec_config.json.
"""

from PySide6.QtCore import QSettings

_ORG = "LEQUIA"
_APP = "HPSEC Suite"


def get_settings() -> QSettings:
    """Retorna el QSettings de l'aplicació (font única org/app)."""
    return QSettings(_ORG, _APP)


def remember_dir(key: str, path: str):
    """Desa una carpeta usada (silenciós si path buit)."""
    if path:
        get_settings().setValue(key, path)


def recall_dir(key: str, fallback: str = "") -> str:
    """Recupera una carpeta recordada; fallback si no n'hi ha o ja no existeix."""
    import os
    value = get_settings().value(key, "", type=str)
    if value and os.path.isdir(value):
        return value
    return fallback
