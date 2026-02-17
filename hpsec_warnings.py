"""
HPSEC Suite - Sistema d'Avisos Estructurats
============================================

Mòdul centralitzat per gestionar avisos amb nivells jeràrquics.

Nivells:
- BLOCKER: No es pot continuar. Cal resoldre primer.
- WARNING: Es pot continuar AMB nota obligatòria.
- INFO: Es pot continuar sense acció. Només informatiu.

Ús:
    from hpsec_warnings import create_warning, get_max_warning_level, WarningLevel

    # Crear un avís
    warning = create_warning(
        code="CAL_RSD_HIGH",
        level=WarningLevel.WARNING,
        message="RSD de rèpliques alt (12.3%)",
        stage="calibrate",
        details={"rsd": 12.3, "threshold": 10}
    )

    # Obtenir nivell màxim
    max_level = get_max_warning_level(warnings_list)
"""

from datetime import datetime
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
# DEFINICIONS D'AVISOS PER ETAPA
# =============================================================================

# Format: code -> (level, message_template, dismissable)
WARNING_DEFINITIONS = {
    # === IMPORTAR ===
    "IMP_NO_DATA": (WarningLevel.BLOCKER, "Carpeta buida o sense CSV", False),
    "IMP_MISSING_UIB": (WarningLevel.BLOCKER, "Falten dades UIB (mode DUAL)", False),
    "IMP_MISSING_DAD": (WarningLevel.BLOCKER, "Falten dades DAD (mode DUAL)", False),
    "IMP_ORPHAN_FILES": (WarningLevel.WARNING, "{n} fitxers orfes no assignables", True),
    "IMP_EMPTY_CSV": (WarningLevel.WARNING, "{n} CSV buits detectats", True),
    "IMP_ASSIGNMENT_SUGGESTION": (WarningLevel.INFO, "Suggeriment d'assignació: {file} → {sample}", True),
    "IMP_FALLBACK_DAD": (WarningLevel.INFO, "Usant CSV DAD com a fallback (no Export3D)", True),

    # === CALIBRAR ===
    "CAL_NO_KHP": (WarningLevel.BLOCKER, "No s'ha trobat cap KHP", False),
    "CAL_ALL_REPLICAS_INVALID": (WarningLevel.BLOCKER, "Totes les rèpliques són invàlides", False),
    "CAL_TIMEOUT": (WarningLevel.BLOCKER, "Timeout detectat en KHP: {details}", False),
    "CAL_BATMAN": (WarningLevel.BLOCKER, "Pic Batman detectat (doble pic)", False),
    "CAL_RSD_HIGH": (WarningLevel.WARNING, "RSD de rèpliques alt ({rsd:.1f}% > {threshold}%)", True),
    "CAL_GLOBAL_ONLY": (WarningLevel.INFO, "Sense KHP local - usant calibració global ({msg})", True),
    "CAL_DEVIATION_HIGH": (WarningLevel.WARNING, "Desviació històrica alta ({dev:.1f}% > 15%)", True),
    "CAL_REPLICA_OUTLIER": (WarningLevel.WARNING, "Rèplica {n} marcada com outlier", True),
    "CAL_SYMMETRY_LOW": (WarningLevel.INFO, "Simetria fora de rang ({sym:.2f})", True),
    "CAL_SMOOTHNESS_LOW": (WarningLevel.INFO, "Smoothness baix ({val:.0f}%)", True),
    "CAL_DEVIATION_MINOR": (WarningLevel.INFO, "Desviació històrica menor ({dev:.1f}% < 15%)", True),
    "CAL_HISTORICAL_INSUFFICIENT": (WarningLevel.INFO, "Històric insuficient (< 3 calibracions)", True),
    "CAL_SNR_LOW": (WarningLevel.WARNING, "SNR baix ({snr:.1f})", True),
    "CAL_SNR_VERY_LOW": (WarningLevel.BLOCKER, "SNR extremadament baix ({snr:.1f})", False),
    "CAL_EXPANSION_HIGH": (WarningLevel.WARNING, "Límits expandits (+{pct:.0f}%)", True),
    "CAL_CR_LOW": (WarningLevel.WARNING, "Concentration ratio baix ({cr:.1%})", True),
    "CAL_CR_VERY_LOW": (WarningLevel.BLOCKER, "Concentration ratio crític ({cr:.1%})", False),
    "CAL_MULTI_PEAK": (WarningLevel.BLOCKER, "Múltiples pics detectats", False),
    "CAL_INTENSITY_LOW": (WarningLevel.WARNING, "Intensitat baixa ({val:.0f} mAU)", True),
    "CAL_INTENSITY_EXTREME": (WarningLevel.BLOCKER, "Intensitat extrema ({val:.0f} mAU)", False),

    # === ANALITZAR ===
    "ANA_NO_CALIBRATION": (WarningLevel.BLOCKER, "No hi ha calibració disponible", False),
    "ANA_TIMEOUT": (WarningLevel.BLOCKER, "Timeout detectat en mostra {sample}", False),
    "ANA_BATMAN": (WarningLevel.WARNING, "Pic Batman en mostra {sample}", True),  # WARNING en anàlisi
    "ANA_NO_PEAK": (WarningLevel.WARNING, "No s'ha detectat pic DOC en {sample}", True),
    "ANA_SNR_LOW": (WarningLevel.WARNING, "SNR baix en {sample} ({snr:.1f} < 10)", True),
    "ANA_AREA_NEGATIVE": (WarningLevel.WARNING, "Àrea negativa en {sample}", True),
    "ANA_SHIFT_HIGH": (WarningLevel.WARNING, "Shift alt en {sample} ({shift:.1f}s > 30s)", True),
    "ANA_EMPTY_SAMPLES": (WarningLevel.WARNING, "{n} mostres sense dades", True),
    "ANA_SECONDARY_PEAKS": (WarningLevel.INFO, "Pics secundaris detectats en {sample}", True),
    "ANA_CONCENTRATION_RATIO": (WarningLevel.INFO, "Concentration ratio: {ratio:.1%}", True),
    "ANA_REPLICA_CORRELATION_LOW": (WarningLevel.WARNING, "Correlació rèpliques baixa en {sample}", True),
    "ANA_REPLICA_AREA_DIFF": (WarningLevel.WARNING, "Diferència àrea rèpliques alta en {sample}", True),

    # === CONSOLIDAR ===
    "CON_NO_BP": (WarningLevel.BLOCKER, "No s'ha trobat BP associat", False),
    "CON_BP_INVALID": (WarningLevel.BLOCKER, "BP trobat però invàlid", False),
    "CON_RF_DEVIATION_HIGH": (WarningLevel.WARNING, "Diferència RF > 20% entre SEQ i BP", True),
    "CON_BP_OLD": (WarningLevel.WARNING, "BP antic (> 7 dies)", True),
    "CON_MISSING_SAMPLES": (WarningLevel.WARNING, "{n} mostres no consolidades", True),
    "CON_BP_REUSED": (WarningLevel.INFO, "BP reutilitzat de {date}", True),
}


def create_warning(
    code: str,
    level: WarningLevel = None,
    message: str = None,
    stage: str = None,
    details: dict = None,
    sample: str = None,
    condition_key: str = None,
) -> dict:
    """
    Crea un avís estructurat.

    Args:
        code: Codi únic de l'avís (ex: "CAL_RSD_HIGH")
        level: Nivell de gravetat. Si no s'especifica, s'usa el de la definició.
        message: Missatge personalitzat. Si no s'especifica, s'usa el template.
        stage: Etapa ("import", "calibrate", "analyze", "consolidate")
        details: Dict amb detalls específics (per formatting del message)
        sample: Nom de la mostra (si aplica)
        condition_key: Clau de condició (si aplica, ex: "BP_100_2")

    Returns:
        Dict amb l'avís estructurat
    """
    details = details or {}

    # Obtenir definició si existeix
    definition = WARNING_DEFINITIONS.get(code)
    if definition:
        default_level, message_template, dismissable = definition
        if level is None:
            level = default_level
        if message is None:
            try:
                message = message_template.format(**details, sample=sample or "")
            except KeyError:
                message = message_template
    else:
        # Avís personalitzat (no definit)
        if level is None:
            level = WarningLevel.INFO
        if message is None:
            message = code
        dismissable = True

    # Deduir stage del codi si no s'especifica
    if stage is None:
        prefix = code.split("_")[0] if "_" in code else ""
        stage_map = {"IMP": "import", "CAL": "calibrate", "ANA": "analyze", "CON": "consolidate"}
        stage = stage_map.get(prefix, "unknown")

    return {
        "code": code,
        "level": level.value if isinstance(level, WarningLevel) else level,
        "message": message,
        "stage": stage,
        "details": details,
        "timestamp": datetime.now().isoformat(),
        "sample": sample,
        "condition_key": condition_key,
        "dismissable": dismissable,
        "dismissed": False,
        "dismissed_by": None,
        "dismissed_note": None,
        "dismissed_at": None,
    }


def get_max_warning_level(warnings: list) -> str:
    """
    Retorna el nivell màxim d'avisos no dismissed.

    Args:
        warnings: Llista d'avisos estructurats

    Returns:
        "blocker", "warning", "info" o "none"
    """
    if not warnings:
        return "none"

    levels = {"blocker": 3, "warning": 2, "info": 1, "none": 0}
    max_level = "none"

    for w in warnings:
        if not w.get("dismissed", False):
            level = w.get("level", "info")
            if levels.get(level, 0) > levels.get(max_level, 0):
                max_level = level

    return max_level


def filter_warnings_by_level(warnings: list, level: str) -> list:
    """Filtra avisos per nivell."""
    return [w for w in warnings if w.get("level") == level and not w.get("dismissed", False)]


def filter_warnings_by_stage(warnings: list, stage: str) -> list:
    """Filtra avisos per etapa."""
    return [w for w in warnings if w.get("stage") == stage]


def has_blockers(warnings: list) -> bool:
    """Retorna True si hi ha avisos BLOCKER no dismissed."""
    return get_max_warning_level(warnings) == "blocker"


def dismiss_warning(warning: dict, reviewer: str, note: str) -> dict:
    """
    Marca un avís com a "dismissed" (revisat/acceptat).

    Args:
        warning: L'avís a marcar
        reviewer: Nom/inicials del revisor
        note: Nota explicativa (obligatòria per WARNING)

    Returns:
        L'avís actualitzat
    """
    if not warning.get("dismissable", True):
        raise ValueError(f"L'avís {warning.get('code')} no es pot ignorar (BLOCKER)")

    warning["dismissed"] = True
    warning["dismissed_by"] = reviewer
    warning["dismissed_note"] = note
    warning["dismissed_at"] = datetime.now().isoformat()
    return warning


def warnings_summary(warnings: list) -> dict:
    """
    Genera un resum dels avisos.

    Returns:
        Dict amb comptadors per nivell i etapa
    """
    summary = {
        "total": len(warnings),
        "by_level": {"blocker": 0, "warning": 0, "info": 0},
        "by_stage": {"import": 0, "calibrate": 0, "analyze": 0, "consolidate": 0},
        "dismissed": 0,
        "pending": 0,
    }

    for w in warnings:
        level = w.get("level", "info")
        stage = w.get("stage", "unknown")

        if level in summary["by_level"]:
            summary["by_level"][level] += 1
        if stage in summary["by_stage"]:
            summary["by_stage"][stage] += 1

        if w.get("dismissed", False):
            summary["dismissed"] += 1
        else:
            summary["pending"] += 1

    return summary


# =============================================================================
# MIGRACIÓ: Convertir warnings antics (strings) a nous (estructurats)
# =============================================================================

def migrate_legacy_warning(legacy_warning: str, stage: str = "unknown") -> dict:
    """
    Converteix un warning antic (string) al nou format estructurat.

    Intenta detectar el tipus d'avís pel contingut del missatge.
    """
    legacy = legacy_warning.upper()

    # Mapeig de patrons a codis
    patterns = [
        # Calibrar
        ("TIMEOUT", "CAL_TIMEOUT" if stage == "calibrate" else "ANA_TIMEOUT"),
        ("BATMAN", "CAL_BATMAN" if stage == "calibrate" else "ANA_BATMAN"),
        ("MULTI", "CAL_MULTI_PEAK"),
        ("LOW_SNR", "CAL_SNR_LOW" if stage == "calibrate" else "ANA_SNR_LOW"),
        ("SNR", "CAL_SNR_LOW" if stage == "calibrate" else "ANA_SNR_LOW"),
        ("ASYMMETRY", "CAL_SYMMETRY_LOW"),
        ("SIMETRIA", "CAL_SYMMETRY_LOW"),
        ("SMOOTHNESS", "CAL_SMOOTHNESS_LOW"),
        ("IRREGULAR", "CAL_SMOOTHNESS_LOW"),
        ("EXPANSION", "CAL_EXPANSION_HIGH"),
        ("CR_", "CAL_CR_LOW"),
        ("CONCENTRATION_RATIO", "CAL_CR_LOW"),
        ("HISTORICAL", "CAL_DEVIATION_MINOR"),
        ("DESVIACIÓ", "CAL_DEVIATION_HIGH"),
        ("DEVIATION", "CAL_DEVIATION_HIGH"),
        ("RSD", "CAL_RSD_HIGH"),
        ("INTENSITY", "CAL_INTENSITY_LOW"),
        ("PEAK_POSITION", "ANA_SHIFT_HIGH"),
        ("RATIO", "ANA_CONCENTRATION_RATIO"),
        ("PEAK_MISMATCH", "ANA_SHIFT_HIGH"),
        ("CORRELATION", "ANA_REPLICA_CORRELATION_LOW"),
        ("AREA_DIFF", "ANA_REPLICA_AREA_DIFF"),
    ]

    code = "UNKNOWN"
    for pattern, mapped_code in patterns:
        if pattern in legacy:
            code = mapped_code
            break

    # Determinar level basat en patrons
    if any(x in legacy for x in ["TIMEOUT", "EXTREME", "CRITICAL", "INVALID"]):
        level = WarningLevel.BLOCKER
    elif any(x in legacy for x in ["WARNING", "WARN", "LOW", "HIGH", "DIFF"]):
        level = WarningLevel.WARNING
    else:
        level = WarningLevel.INFO

    return create_warning(
        code=code,
        level=level,
        message=legacy_warning,  # Mantenir missatge original
        stage=stage,
        details={"legacy": True, "original": legacy_warning},
    )


def migrate_warnings_list(legacy_warnings: list, stage: str = "unknown") -> list:
    """Converteix una llista de warnings antics al nou format."""
    return [migrate_legacy_warning(w, stage) for w in legacy_warnings if isinstance(w, str)]


def normalize_warnings(warnings: list, stage: str = "unknown") -> list:
    """
    Normalitza una llista mixta de warnings (strings + dicts) al format estructurat.

    Strings es converteixen via migrate_legacy_warning().
    Dicts ja estructurats es retornen tal qual.

    Args:
        warnings: Llista mixta de strings i/o dicts
        stage: Etapa per defecte per warnings legacy

    Returns:
        Llista de dicts estructurats (format create_warning)
    """
    result = []
    for w in warnings:
        if isinstance(w, str):
            result.append(migrate_legacy_warning(w, stage))
        elif isinstance(w, dict) and "code" in w:
            result.append(w)
        elif isinstance(w, dict):
            result.append(create_warning(
                code="UNKNOWN",
                message=str(w),
                stage=stage,
                details={"legacy": True, "original": w},
            ))
    return result


# =============================================================================
# UTILITATS PER AVISOS
# =============================================================================

# =============================================================================
# CATÀLEG D'ANOMALIES (Single Source of Truth)
# =============================================================================

ANOMALY_CATALOG = {
    # === Anomalies d'anàlisi (per rèplica) ===
    "BATMAN_DIRECT": {
        "severity": WarningLevel.BLOCKER,
        "label": "Batman Direct",
        "icon": "B",
        "description": "Doble pic detectat en senyal DOC Direct",
        "stage": "analyze",
        "repairable": True,
        "invalidates": False,
    },
    "BATMAN_UIB": {
        "severity": WarningLevel.BLOCKER,
        "label": "Batman UIB",
        "icon": "B",
        "description": "Doble pic detectat en senyal DOC UIB",
        "stage": "analyze",
        "repairable": True,
        "invalidates": False,
    },
    "BATMAN": {
        "severity": WarningLevel.BLOCKER,
        "label": "Batman",
        "icon": "B",
        "description": "Doble pic detectat",
        "stage": "analyze",
        "repairable": True,
        "invalidates": False,
    },
    "NO_PEAK": {
        "severity": WarningLevel.BLOCKER,
        "label": "Sense pic",
        "icon": "!",
        "description": "No s'ha detectat el pic DOC principal",
        "stage": "analyze",
        "repairable": False,
        "invalidates": True,
    },
    "TIMEOUT_IN_PEAK": {
        "severity": WarningLevel.BLOCKER,
        "label": "Timeout al pic",
        "icon": "T!",
        "description": "Timeout del detector dins la zona del pic principal",
        "stage": "analyze",
        "repairable": False,
        "invalidates": True,
    },
    "LOW_SNR": {
        "severity": WarningLevel.WARNING,
        "label": "SNR baix",
        "icon": "SNR",
        "description": "Relació senyal-soroll per sota del llindar",
        "stage": "analyze",
        "repairable": False,
        "invalidates": False,
    },
    "BELOW_LOD": {
        "severity": WarningLevel.WARNING,
        "label": "Sota LOD",
        "icon": "<LOD",
        "description": "SNR < 3: senyal no distingible del soroll",
        "stage": "analyze",
        "repairable": False,
        "invalidates": False,
    },
    "BELOW_LOQ": {
        "severity": WarningLevel.WARNING,
        "label": "Sota LOQ",
        "icon": "<LOQ",
        "description": "SNR < 10: quantificació poc fiable",
        "stage": "analyze",
        "repairable": False,
        "invalidates": False,
    },
    "UIB_NO_BASELINE": {
        "severity": WarningLevel.INFO,
        "label": "UIB sense baseline",
        "icon": "",
        "description": "Senyal UIB sense correcció de baseline precomputada",
        "stage": "analyze",
        "repairable": False,
        "invalidates": False,
    },

    # === Comparació de rèpliques ===
    "LOW_CORRELATION": {
        "severity": WarningLevel.WARNING,
        "label": "Correlació baixa DOC",
        "icon": "r\u2193",
        "description": "Pearson entre rèpliques DOC per sota del llindar",
        "stage": "analyze",
        "repairable": False,
        "invalidates": False,
    },
    "AREA_DIFF_HIGH": {
        "severity": WarningLevel.WARNING,
        "label": "Diferència àrea alta",
        "icon": "\u0394A",
        "description": "Diferència d'àrea total entre rèpliques supera el llindar",
        "stage": "analyze",
        "repairable": False,
        "invalidates": False,
    },
    "FRACTION_DIFF_HIGH": {
        "severity": WarningLevel.WARNING,
        "label": "Diferència fracció alta",
        "icon": "\u0394F",
        "description": "Diferència d'àrea per fracció entre rèpliques supera el llindar",
        "stage": "analyze",
        "repairable": False,
        "invalidates": False,
    },
    "REPLICA_NOT_PROCESSED": {
        "severity": WarningLevel.BLOCKER,
        "label": "Rèplica no processada",
        "icon": "!",
        "description": "Una o ambdues rèpliques no s'han processat",
        "stage": "analyze",
        "repairable": False,
        "invalidates": False,
    },
    "LOW_CORRELATION_254": {
        "severity": WarningLevel.WARNING,
        "label": "Correlació baixa A254",
        "icon": "r\u2193",
        "description": "Pearson entre rèpliques a 254nm per sota del llindar",
        "stage": "analyze",
        "repairable": False,
        "invalidates": False,
    },
    "AREA_DIFF_HIGH_254": {
        "severity": WarningLevel.WARNING,
        "label": "Diferència àrea alta A254",
        "icon": "\u0394A",
        "description": "Diferència d'àrea a 254nm entre rèpliques supera el llindar",
        "stage": "analyze",
        "repairable": False,
        "invalidates": False,
    },
}

# Sets derivats del catàleg
CRITICAL_ANOMALIES = {code for code, e in ANOMALY_CATALOG.items()
                      if e.get("severity") == WarningLevel.BLOCKER}
WARNING_ANOMALIES = {code for code, e in ANOMALY_CATALOG.items()
                     if e.get("severity") == WarningLevel.WARNING}


def create_anomaly(code: str, details: dict = None, replica: str = None, sample: str = None) -> dict:
    """Crea un dict d'anomalia estructurat des del catàleg."""
    entry = ANOMALY_CATALOG.get(code, {})
    return {
        "code": code,
        "severity": entry.get("severity", WarningLevel.INFO).value,
        "label": entry.get("label", code),
        "icon": entry.get("icon", ""),
        "message": entry.get("description", code),
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
    """Classifica per severitat: {blocker: [...], warning: [...], info: [...], repaired: [...]}."""
    result = {"blocker": [], "warning": [], "info": [], "repaired": []}
    for a in anomalies:
        if isinstance(a, dict):
            code = a.get("code", "")
            repaired = a.get("repaired", False)
        else:
            repaired = "_REPAIRED" in str(a)
            code = str(a).replace("_REPAIRED", "")
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


# =============================================================================
# UTILITATS PER AVISOS
# =============================================================================

def create_warnings_from_timeout_info(timeout_info: dict, stage: str = "calibrate", sample: str = None) -> list:
    """
    Crea avisos estructurats a partir de la info de timeout de hpsec_core.detect_timeout().

    Args:
        timeout_info: Dict retornat per hpsec_core.detect_timeout()
        stage: "calibrate" o "analyze"
        sample: Nom de la mostra (opcional)

    Returns:
        Llista d'avisos estructurats
    """
    warnings = []

    if not timeout_info or timeout_info.get("n_timeouts", 0) == 0:
        return warnings

    severity = timeout_info.get("severity", "OK")

    if severity == "CRITICAL":
        code = "CAL_TIMEOUT" if stage == "calibrate" else "ANA_TIMEOUT"
        warnings.append(create_warning(
            code=code,
            stage=stage,
            sample=sample,
            details={
                "n_timeouts": timeout_info["n_timeouts"],
                "zones": timeout_info.get("zone_summary", {}),
                "severity": severity,
            },
        ))
    elif severity == "WARNING":
        code = "CAL_TIMEOUT" if stage == "calibrate" else "ANA_TIMEOUT"
        warnings.append(create_warning(
            code=code,
            level=WarningLevel.WARNING,
            message=f"Timeout detectat en zones no crítiques",
            stage=stage,
            sample=sample,
            details={
                "n_timeouts": timeout_info["n_timeouts"],
                "zones": timeout_info.get("zone_summary", {}),
                "severity": severity,
            },
        ))
    # INFO: No afegir avís - timeout en zona segura

    return warnings


def create_warnings_from_batman_info(batman_info: dict, stage: str = "calibrate", sample: str = None) -> list:
    """
    Crea avisos estructurats a partir de la info de batman de hpsec_core.detect_batman().

    Args:
        batman_info: Dict retornat per hpsec_core.detect_batman()
        stage: "calibrate" o "analyze"
        sample: Nom de la mostra (opcional)

    Returns:
        Llista d'avisos estructurats
    """
    warnings = []

    if not batman_info or not batman_info.get("is_batman", False):
        return warnings

    code = "CAL_BATMAN" if stage == "calibrate" else "ANA_BATMAN"
    warnings.append(create_warning(
        code=code,
        stage=stage,
        sample=sample,
        details={
            "valley_depth": batman_info.get("max_depth", 0),
            "n_valleys": batman_info.get("n_valleys", 0),
        },
    ))

    return warnings
