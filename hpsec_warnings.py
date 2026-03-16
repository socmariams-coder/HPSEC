"""
HPSEC Suite - Sistema d'Avisos Estructurats
============================================

Mòdul centralitzat per gestionar avisos amb nivells jeràrquics.
Font única: ANOMALY_CATALOG.

Nivells:
- BLOCKER: No es pot continuar. Cal resoldre primer.
- WARNING: Es pot continuar AMB nota obligatòria.
- INFO: Es pot continuar sense acció. Només informatiu.

Ús:
    from hpsec_warnings import create_anomaly, get_max_anomaly_severity, WarningLevel

    # Crear un avís
    warning = create_anomaly("CAL_NO_KHP")

    # Amb detalls
    warning = create_anomaly("IMP_ORPHAN_FILES", details={"n": 3, "files": "a.csv, b.csv"})

    # Obtenir nivell màxim
    max_level = get_max_anomaly_severity(warnings_list)
"""

from enum import Enum
from typing import Optional, Any


class WarningLevel(str, Enum):
    """Nivells d'avís ordenats per gravetat."""
    BLOCKER = "blocker"
    WARNING = "warning"
    INFO = "info"
    NONE = "none"

    def __lt__(self, other):
        order = {self.BLOCKER: 3, self.WARNING: 2, self.INFO: 1, self.NONE: 0}
        return order.get(self, 0) < order.get(other, 0)

    def __gt__(self, other):
        order = {self.BLOCKER: 3, self.WARNING: 2, self.INFO: 1, self.NONE: 0}
        return order.get(self, 0) > order.get(other, 0)


# =============================================================================
# CATÀLEG D'ANOMALIES (Single Source of Truth)
# =============================================================================

ANOMALY_CATALOG = {
    # === Avisos d'importació (nivell seqüència) ===
    "IMP_NO_DATA": {
        "severity": WarningLevel.BLOCKER,
        "label": "Sense dades",
        "icon": "!",
        "description": "Carpeta buida o sense CSV",
        "stage": "import",
        "repairable": False,
        "invalidates": True,
        "action": "Verificar carpeta i MasterFile",
    },
    "IMP_MISSING_UIB": {
        "severity": WarningLevel.BLOCKER,
        "label": "Falta UIB",
        "icon": "!",
        "description": "Falten dades UIB (mode DUAL)",
        "stage": "import",
        "repairable": False,
        "invalidates": True,
        "action": "Afegir CSV UIB a la carpeta",
    },
    "IMP_MISSING_DAD": {
        "severity": WarningLevel.BLOCKER,
        "label": "Falta DAD",
        "icon": "!",
        "description": "Falten dades DAD (mode DUAL)",
        "stage": "import",
        "repairable": False,
        "invalidates": True,
        "action": "Afegir Export3D o CSV DAD a la carpeta",
    },
    "IMP_ORPHAN_FILES": {
        "severity": WarningLevel.WARNING,
        "label": "Fitxers orfes",
        "icon": "?",
        "description": "Fitxers sense assignar detectats",
        "stage": "import",
        "repairable": False,
        "invalidates": False,
        "action": "Assignar fitxers a Importar",
    },
    "IMP_INCOMPLETE": {
        "severity": WarningLevel.WARNING,
        "label": "Importació incompleta",
        "icon": "!",
        "description": "Importació amb injeccions duplicades o dades parcials",
        "stage": "import",
        "repairable": False,
        "invalidates": False,
        "action": "Corregir MasterFile (Inj# duplicat)",
    },
    "IMP_SHORT_CHROMATOGRAM": {
        "severity": WarningLevel.BLOCKER,
        "label": "Cromatograma truncat",
        "icon": "✂",
        "description": "Cromatograma DOC amb menys de 30 min de dades",
        "stage": "import",
        "repairable": False,
        "invalidates": True,
        "action": "Excloure mostra (dades insuficients)",
    },
    "IMP_TOC_MINUTE_PRECISION": {
        "severity": WarningLevel.INFO,
        "label": "Timestamps TOC arrodonits",
        "icon": "⏱",
        "description": "2-TOC amb timestamps al minut (sense segons). Cadència 4s reconstruïda automàticament.",
        "stage": "import",
        "repairable": True,
        "invalidates": False,
        "action": "Automàtic — verificar cromatograma DOC visualment",
    },

    # === Avisos de calibració (nivell seqüència) ===
    "CAL_NO_KHP": {
        "severity": WarningLevel.BLOCKER,
        "label": "Sense KHP",
        "icon": "!",
        "description": "No s'ha trobat cap KHP a la seqüència",
        "stage": "calibrate",
        "repairable": False,
        "invalidates": False,
        "action": "Afegir KHP o usar calibració global",
    },
    "CAL_ALL_REPLICAS_INVALID": {
        "severity": WarningLevel.BLOCKER,
        "label": "Totes rèpliques invàlides",
        "icon": "!",
        "description": "Totes les rèpliques KHP són invàlides",
        "stage": "calibrate",
        "repairable": False,
        "invalidates": False,
        "action": "Revisar cromatogrames KHP",
    },
    "CAL_GLOBAL_ONLY": {
        "severity": WarningLevel.INFO,
        "label": "Calibració global",
        "icon": "ℹ",
        "description": "Sense KHP local — usant calibració global",
        "stage": "calibrate",
        "repairable": False,
        "invalidates": False,
        "action": "Informatiu — shift no verificable",
    },
    "CAL_REPLICA_OUTLIER": {
        "severity": WarningLevel.WARNING,
        "label": "Rèplica outlier",
        "icon": "⚠",
        "description": "Rèplica KHP marcada com outlier",
        "stage": "calibrate",
        "repairable": False,
        "invalidates": False,
        "action": "Revisar rèplica marcada",
    },

    # === Avisos d'anàlisi (nivell seqüència) ===
    "ANA_NO_CALIBRATION": {
        "severity": WarningLevel.BLOCKER,
        "label": "Sense calibració",
        "icon": "!",
        "description": "No hi ha calibració disponible",
        "stage": "analyze",
        "repairable": False,
        "invalidates": True,
        "action": "Executar Verificar primer",
    },
    "ANA_EMPTY_SAMPLES": {
        "severity": WarningLevel.WARNING,
        "label": "Mostres buides",
        "icon": "⚠",
        "description": "Mostres sense dades processades",
        "stage": "analyze",
        "repairable": False,
        "invalidates": False,
        "action": "Revisar mostres sense dades",
    },
    "ANA_SAMPLES_WITH_ISSUES": {
        "severity": WarningLevel.WARNING,
        "label": "Mostres amb anomalies",
        "icon": "\u26a0",
        "description": "Mostres amb anomalies detectades (veure taula)",
        "stage": "analyze",
        "repairable": False,
        "invalidates": False,
        "action": "Revisar la taula d'an\u00e0lisi",
    },

    # === Anomalies d'anàlisi (per rèplica) ===
    "IRREGULAR_TOP_DIRECT": {  # formerly BATMAN_DIRECT — jagged/batman artifact
        "severity": WarningLevel.BLOCKER,
        "label": "Irregular Top Direct",
        "icon": "B",
        "description": "Doble pic detectat en senyal DOC Direct",
        "stage": "analyze",
        "repairable": True,
        "invalidates": False,
        "action": "Revisar reparació paràbola al detall",
    },
    "IRREGULAR_TOP_UIB": {  # formerly BATMAN_UIB — jagged/batman artifact
        "severity": WarningLevel.BLOCKER,
        "label": "Irregular Top UIB",
        "icon": "B",
        "description": "Doble pic detectat en senyal DOC UIB",
        "stage": "analyze",
        "repairable": True,
        "invalidates": False,
        "action": "Revisar reparació paràbola al detall",
    },
    "IRREGULAR_TOP": {  # formerly BATMAN — jagged/batman artifact
        "severity": WarningLevel.BLOCKER,
        "label": "Irregular Top",
        "icon": "B",
        "description": "Doble pic detectat",
        "stage": "analyze",
        "repairable": True,
        "invalidates": False,
        "action": "Revisar reparació paràbola al detall",
    },
    "NO_PEAK": {
        "severity": WarningLevel.BLOCKER,
        "label": "Sense pic",
        "icon": "!",
        "description": "No s'ha detectat el pic DOC principal",
        "stage": "analyze",
        "repairable": False,
        "invalidates": True,
        "action": "Verificar que la injecció s'ha realitzat correctament",
    },
    "TIMEOUT_IN_PEAK": {
        "severity": WarningLevel.BLOCKER,
        "label": "Timeout al pic",
        "icon": "T!",
        "description": "Timeout del detector dins la zona del pic principal",
        "stage": "analyze",
        "repairable": False,
        "invalidates": True,
        "action": "Mostra invàlida — repetir injecció si possible",
    },
    "TIMEOUT_AT_BOUNDARY": {
        "severity": WarningLevel.BLOCKER,
        "label": "Timeout al límit d'injecció",
        "icon": "T↕",
        "description": "Timeout del detector just abans de la injecció — pre-margin reduït",
        "stage": "import",
        "repairable": False,
        "invalidates": True,
        "action": "Pre-margin insuficient per absorbir shift DOC — triar l'altra rèplica",
    },
    "LOW_SNR": {
        "severity": WarningLevel.WARNING,
        "label": "SNR baix",
        "icon": "SNR",
        "description": "Relació senyal-soroll per sota del llindar",
        "stage": "analyze",
        "repairable": False,
        "invalidates": False,
        "action": "Considerar concentrar la mostra o augmentar volum injecció",
    },
    "BELOW_LOD": {
        "severity": WarningLevel.WARNING,
        "label": "Sota LOD",
        "icon": "<LOD",
        "description": "SNR < 3: senyal no distingible del soroll",
        "stage": "analyze",
        "repairable": False,
        "invalidates": False,
        "action": "Senyal no fiable — reportar com <LOD",
    },
    "BELOW_LOQ": {
        "severity": WarningLevel.WARNING,
        "label": "Sota LOQ",
        "icon": "<LOQ",
        "description": "SNR < 10: quantificació poc fiable",
        "stage": "analyze",
        "repairable": False,
        "invalidates": False,
        "action": "Quantificació orientativa — reportar com <LOQ",
    },
    "UIB_SATURATED": {
        "severity": WarningLevel.BLOCKER,
        "label": "UIB saturat",
        "icon": "SAT",
        "description": "Senyal UIB saturat (y_max >= 95% sensibilitat)",
        "stage": "analyze",
        "repairable": False,
        "invalidates": True,
        "action": "Usar només senyal DOC Direct per aquesta mostra",
    },
    "UIB_NO_BASELINE": {
        "severity": WarningLevel.INFO,
        "label": "UIB sense baseline",
        "icon": "",
        "description": "Senyal UIB sense correcció de baseline precomputada",
        "stage": "analyze",
        "repairable": False,
        "invalidates": False,
        "action": "Informatiu — baseline estimada automàticament",
    },

    # === Timeout composició ===
    "TIMEOUT_COMPOSABLE": {
        "severity": WarningLevel.INFO,
        "label": "Timeouts composables",
        "icon": "TC",
        "description": "Les 2 rèpliques tenen timeouts a posicions diferents — composició possible",
        "stage": "analyze",
        "repairable": True,
        "invalidates": False,
        "action": "Composar rèpliques per obtenir un cromatograma net",
    },
    "TIMEOUT_COMPOSED": {
        "severity": WarningLevel.INFO,
        "label": "Timeout reparat (composició)",
        "icon": "TC✓",
        "description": "Cromatograma composat a partir de segments de les dues rèpliques",
        "stage": "analyze",
        "repairable": False,
        "invalidates": False,
        "action": "Informatiu — verificar visualment el resultat",
    },

    # === Comparació de rèpliques ===
    "LOW_CORRELATION": {
        "severity": WarningLevel.WARNING,
        "label": "Correlació baixa DOC",
        "icon": "r↓",
        "description": "Pearson entre rèpliques DOC per sota del llindar",
        "stage": "analyze",
        "repairable": False,
        "invalidates": False,
        "action": "Revisar rèpliques i seleccionar la més representativa",
    },
    "AREA_DIFF_HIGH": {
        "severity": WarningLevel.WARNING,
        "label": "Diferència àrea alta",
        "icon": "ΔA",
        "description": "Diferència d'àrea total entre rèpliques supera el llindar",
        "stage": "analyze",
        "repairable": False,
        "invalidates": False,
        "action": "Revisar rèpliques — possible problema d'injecció",
    },
    "FRACTION_DIFF_HIGH": {
        "severity": WarningLevel.WARNING,
        "label": "Diferència fracció alta",
        "icon": "ΔF",
        "description": "Diferència d'àrea per fracció entre rèpliques supera el llindar",
        "stage": "analyze",
        "repairable": False,
        "invalidates": False,
        "action": "Revisar distribució per fraccions de cada rèplica",
    },
    "REPLICA_NOT_PROCESSED": {
        "severity": WarningLevel.BLOCKER,
        "label": "Rèplica no processada",
        "icon": "!",
        "description": "Una o ambdues rèpliques no s'han processat",
        "stage": "analyze",
        "repairable": False,
        "invalidates": False,
        "action": "Tornar a executar l'anàlisi",
    },
    "LOW_CORRELATION_DAD": {
        "severity": WarningLevel.WARNING,
        "label": "Correlació baixa DAD",
        "icon": "r↓",
        "description": "Pearson entre rèpliques DAD per sota del llindar",
        "stage": "analyze",
        "repairable": False,
        "invalidates": False,
        "action": "Revisar perfils DAD de les rèpliques",
    },
    # Backward compat alias
    "LOW_CORRELATION_254": {
        "severity": WarningLevel.WARNING,
        "label": "Correlació baixa DAD",
        "icon": "r↓",
        "description": "Pearson entre rèpliques DAD per sota del llindar",
        "stage": "analyze",
        "repairable": False,
        "invalidates": False,
        "action": "Revisar perfils DAD de les rèpliques",
    },
    "AREA_DIFF_HIGH_DAD": {
        "severity": WarningLevel.WARNING,
        "label": "Diferència àrea alta DAD",
        "icon": "ΔA",
        "description": "Diferència d'àrea DAD entre rèpliques supera el llindar",
        "stage": "analyze",
        "repairable": False,
        "invalidates": False,
        "action": "Revisar perfils DAD — possible interferència",
    },
    # Backward compat alias
    "AREA_DIFF_HIGH_254": {
        "severity": WarningLevel.WARNING,
        "label": "Diferència àrea alta DAD",
        "icon": "ΔA",
        "description": "Diferència d'àrea DAD entre rèpliques supera el llindar",
        "stage": "analyze",
        "repairable": False,
        "invalidates": False,
        "action": "Revisar perfils DAD — possible interferència",
    },

    # === Anomalies de calibració KHP (per rèplica KHP) ===
    "KHP_TIMEOUT_PEAK": {
        "severity": WarningLevel.BLOCKER,
        "label": "KHP timeout al pic",
        "icon": "T!",
        "description": "Timeout del detector afecta el pic KHP",
        "stage": "calibrate",
        "repairable": False,
        "invalidates": True,
        "action": "Excloure rèplica — àrea compromesa",
    },
    "KHP_BIGAUSSIAN_LOW": {
        "severity": WarningLevel.WARNING,
        "label": "KHP forma pic anòmala",
        "icon": "R²",
        "description": "R² bigaussiana del pic KHP < 0.95 — forma no gaussiana",
        "stage": "calibrate",
        "repairable": False,
        "invalidates": False,
        "action": "Revisar cromatograma — possible interferència o pic deformat",
    },
    "KHP_RSD_HIGH": {
        "severity": WarningLevel.WARNING,
        "label": "KHP variabilitat alta entre rèpliques",
        "icon": "RSD",
        "description": "Desviació relativa (RSD) entre rèpliques KHP > 10%",
        "stage": "calibrate",
        "repairable": False,
        "invalidates": False,
        "action": "Revisar cromatogrames — possible error de preparació o injecció",
    },
}

# Codis KHP eliminats — presents en JSONs antics, ignorats en lectura
IGNORED_KHP_CODES = {
    'KHP_MULTI_PEAK', 'KHP_ASYMMETRY', 'KHP_SNR_LOW',
    'KHP_IRREGULAR_TOP', 'KHP_BASELINE_DRIFT', 'KHP_NO_DAD',
    'KHP_DOC_GUIDED_BY_254', 'KHP_FWHM_HIGH',
    'KHP_CR_LOW',
}

# Sets derivats del catàleg
CRITICAL_ANOMALIES = {code for code, e in ANOMALY_CATALOG.items()
                      if e.get("severity") == WarningLevel.BLOCKER}
WARNING_ANOMALIES = {code for code, e in ANOMALY_CATALOG.items()
                     if e.get("severity") == WarningLevel.WARNING}


def create_anomaly(code: str, details: dict = None, replica: str = None,
                   sample: str = None, override_label: str = None) -> dict:
    """Crea un dict d'anomalia estructurat des del catàleg.

    Args:
        override_label: Si donat, substitueix el label del catàleg
            (útil per anomalies dinàmiques com "Correlació baixa A220").
    """
    entry = ANOMALY_CATALOG.get(code, {})
    return {
        "code": code,
        "severity": entry.get("severity", WarningLevel.INFO).value,
        "label": override_label or entry.get("label", code),
        "icon": entry.get("icon", ""),
        "message": entry.get("description", code),
        "action": entry.get("action", ""),
        "repairable": entry.get("repairable", False),
        "repaired": False,
        "repair_info": None,
        "details": details or {},
        "replica": replica,
        "sample": sample,
    }


def get_anomaly_codes(anomalies: list) -> set:
    """Extreu codis d'una llista mixta (strings + dicts)."""
    codes = set()
    for a in anomalies:
        if isinstance(a, str):
            codes.add(a.replace("_REPAIRED", ""))
        elif isinstance(a, dict):
            codes.add(a.get("code", ""))
    return codes


def has_anomaly(anomalies: list, code: str) -> bool:
    """Comprova si un codi existeix a la llista (suporta strings i dicts)."""
    return code in get_anomaly_codes(anomalies)


def classify_anomalies(anomalies: list) -> dict:
    """Classifica per severitat: {blocker: [...], warning: [...], info: [...], repaired: [...], dismissed: [...]}."""
    result = {"blocker": [], "warning": [], "info": [], "repaired": [], "dismissed": []}
    for a in anomalies:
        if isinstance(a, dict):
            code = a.get("code", "")
            repaired = a.get("repaired", False)
            dismissed = a.get("dismissed", False)
        else:
            repaired = "_REPAIRED" in str(a)
            dismissed = False
            code = str(a).replace("_REPAIRED", "")
        if dismissed:
            result["dismissed"].append(a)
            continue
        if repaired:
            result["repaired"].append(a)
            continue
        sev = ANOMALY_CATALOG.get(code, {}).get("severity", WarningLevel.INFO)
        key = sev.value if isinstance(sev, WarningLevel) else sev
        if key == "blocker":
            result["blocker"].append(a)
        elif key == "warning":
            result["warning"].append(a)
        else:
            result["info"].append(a)
    return result


def normalize_anomalies(raw_list: list) -> list:
    """Converteix llista mixta (strings + dicts) a dicts estructurats. Per backward compat JSON."""
    result = []
    for item in raw_list:
        if isinstance(item, str):
            repaired = "_REPAIRED" in item
            code = item.replace("_REPAIRED", "")
            anomaly = create_anomaly(code)
            if repaired:
                anomaly["repaired"] = True
            result.append(anomaly)
        elif isinstance(item, dict) and "code" in item:
            result.append(item)
    return result


def mark_repaired(anomalies: list, code: str, repair_info: dict = None) -> bool:
    """Marca anomalia com a reparada. Retorna True si trobada."""
    for a in anomalies:
        if isinstance(a, dict) and a.get("code") == code and not a.get("repaired"):
            a["repaired"] = True
            a["repair_info"] = repair_info
            return True
    return False


def mark_dismissed(anomalies: list, code: str) -> bool:
    """Marca anomalia com a fals positiu (dismissed). Retorna True si trobada."""
    for a in anomalies:
        if isinstance(a, dict) and a.get("code") == code and not a.get("dismissed"):
            a["dismissed"] = True
            return True
    return False


def unmark_dismissed(anomalies: list, code: str) -> bool:
    """Reactiva anomalia dismissed. Retorna True si trobada."""
    for a in anomalies:
        if isinstance(a, dict) and a.get("code") == code and a.get("dismissed"):
            a.pop("dismissed", None)
            return True
    return False


def get_max_anomaly_severity(anomalies: list) -> str:
    """Retorna severitat màxima (ignorant repaired). Returns 'blocker'/'warning'/'info'/'none'."""
    classified = classify_anomalies(anomalies)
    if classified["blocker"]:
        return "blocker"
    if classified["warning"]:
        return "warning"
    if classified["info"] or classified["repaired"]:
        return "info"
    return "none"


# Backward compat alias — consumidors antics que llegeixen w.get("level")
# ara haurien de llegir w.get("severity"), però aquest wrapper manté compatibilitat
def get_max_warning_level(warnings: list) -> str:
    """Alias de get_max_anomaly_severity per backward compat.

    Suporta dicts amb clau "level" (antic) o "severity" (nou).
    """
    if not warnings:
        return "none"

    levels = {"blocker": 3, "warning": 2, "info": 1, "none": 0}
    max_level = "none"

    for w in warnings:
        if w.get("repaired", False) or w.get("dismissed", False):
            continue
        # Suporta ambdós formats
        level = w.get("severity", w.get("level", "info"))
        if levels.get(level, 0) > levels.get(max_level, 0):
            max_level = level

    return max_level
