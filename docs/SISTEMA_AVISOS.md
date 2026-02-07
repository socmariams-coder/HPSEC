# Sistema d'Avisos HPSEC Suite

**Creat:** 2026-02-06
**Darrera actualització:** 2026-02-06

---

## Filosofia

Els avisos no són tots iguals. Un sistema intel·ligent ha de:
1. **Bloquejar** quan hi ha errors crítics que impedeixen continuar
2. **Avisar** quan cal revisió humana però es pot continuar
3. **Informar** quan és només context útil

---

## Jerarquia d'Avisos

| Nivell | Icona | Color | Comportament |
|--------|-------|-------|--------------|
| **BLOCKER** | :no_entry: | Vermell | NO es pot continuar. Cal resoldre primer. |
| **WARNING** | :warning: | Taronja | Es pot continuar AMB nota obligatòria. |
| **INFO** | :information_source: | Blau | Es pot continuar sense acció. Només informatiu. |

---

## Avisos per Etapa

### 1. IMPORTAR

| Codi | Nivell | Missatge | Justificació |
|------|--------|----------|--------------|
| `IMP_NO_DATA` | BLOCKER | Carpeta buida o sense CSV | No hi ha res a processar |
| `IMP_MISSING_UIB` | BLOCKER | Falten dades UIB (mode DUAL) | Dades incompletes |
| `IMP_MISSING_DAD` | BLOCKER | Falten dades DAD (mode DUAL) | Dades incompletes |
| `IMP_ORPHAN_FILES` | WARNING | {n} fitxers orfes no assignables | Cal documentar per què |
| `IMP_EMPTY_CSV` | WARNING | {n} CSV buits detectats | Pot ser normal (blancs) |
| `IMP_ASSIGNMENT_SUGGESTION` | INFO | Suggeriment d'assignació: {file} → {sample} | Només informatiu |
| `IMP_FALLBACK_DAD` | INFO | Usant CSV DAD com a fallback (no Export3D) | Només informatiu |

### 2. CALIBRAR

| Codi | Nivell | Missatge | Justificació |
|------|--------|----------|--------------|
| `CAL_NO_KHP` | BLOCKER | No s'ha trobat cap KHP | No es pot calibrar |
| `CAL_ALL_REPLICAS_INVALID` | BLOCKER | Totes les rèpliques són invàlides | No hi ha calibració vàlida |
| `CAL_TIMEOUT` | BLOCKER | Timeout detectat en KHP | Senyal saturat, cal reinjectar |
| `CAL_BATMAN` | BLOCKER | Pic Batman detectat (doble pic) | Problema d'injecció |
| `CAL_RSD_HIGH` | WARNING | RSD de rèpliques alt ({rsd}% > 10%) | Qualitat dubtosa, però usable |
| `CAL_SIBLING_KHP` | WARNING | Usant KHP de seqüència veïna ({seq}) | Cal confirmar acceptabilitat |
| `CAL_DEVIATION_HIGH` | WARNING | Desviació històrica alta ({dev}% > 15%) | Revisar tendència |
| `CAL_REPLICA_OUTLIER` | WARNING | Rèplica {n} marcada com outlier | Una rèplica exclosa |
| `CAL_SYMMETRY_LOW` | INFO | Simetria fora de rang ({sym}) | Pot ser normal en BP |
| `CAL_SMOOTHNESS_LOW` | INFO | Smoothness baix ({val}) | Backend no polit |
| `CAL_DEVIATION_MINOR` | INFO | Desviació històrica menor ({dev}% < 15%) | Dins tolerància |
| `CAL_HISTORICAL_INSUFFICIENT` | INFO | Històric insuficient (< 3 calibracions) | No es pot comparar |

### 3. ANALITZAR

| Codi | Nivell | Missatge | Justificació |
|------|--------|----------|--------------|
| `ANA_NO_CALIBRATION` | BLOCKER | No hi ha calibració disponible | No es pot analitzar |
| `ANA_TIMEOUT` | BLOCKER | Timeout detectat en mostra {sample} | Senyal saturat |
| `ANA_BATMAN` | BLOCKER | Pic Batman en mostra {sample} | Problema d'injecció |
| `ANA_NO_PEAK` | WARNING | No s'ha detectat pic DOC en {sample} | Mostra buida o problema |
| `ANA_SNR_LOW` | WARNING | SNR baix en {sample} ({snr} < 10) | Qualitat dubtosa |
| `ANA_AREA_NEGATIVE` | WARNING | Àrea negativa en {sample} | Problema de baseline |
| `ANA_SHIFT_HIGH` | WARNING | Shift alt en {sample} ({shift}s > 30s) | Possible deriva |
| `ANA_EMPTY_SAMPLES` | WARNING | {n} mostres sense dades | Algunes mostres buides |
| `ANA_SECONDARY_PEAKS` | INFO | Pics secundaris detectats en {sample} | Només informatiu |
| `ANA_CONCENTRATION_RATIO` | INFO | Concentration ratio: {ratio} | Només informatiu |

### 4. CONSOLIDAR

| Codi | Nivell | Missatge | Justificació |
|------|--------|----------|--------------|
| `CON_NO_BP` | BLOCKER | No s'ha trobat BP associat | Cal BP per consolidar |
| `CON_BP_INVALID` | BLOCKER | BP trobat però invàlid | Cal BP vàlid |
| `CON_RF_DEVIATION_HIGH` | WARNING | Diferència RF > 20% entre SEQ i BP | Revisar consistència |
| `CON_BP_OLD` | WARNING | BP antic (> 7 dies) | Considerar recalibrar |
| `CON_MISSING_SAMPLES` | WARNING | {n} mostres no consolidades | Algunes mostres falten |
| `CON_BP_REUSED` | INFO | BP reutilitzat de {date} | Només informatiu |

---

## Timeouts i Batman - Casos Especials

### Timeout (Senyal Saturat)

El timeout és un dels problemes més crítics. Indica que el detector s'ha saturat.

**Detecció:**
- Senyal constant al màxim durant > N segons
- Típicament al voltant del pic principal
- El valor màxim del senyal es manté "pla"

**Impacte:**
- L'àrea calculada és INCORRECTA (subestimada)
- El RF serà erroni
- TOTES les mostres analitzades amb aquest RF seran errònies

**Acció requerida:**
- BLOCKER en calibració (cal reinjectar amb menys concentració o volum)
- BLOCKER en anàlisi (cal reinjectar la mostra diluïda)

### Batman (Doble Pic)

El "Batman" és un doble pic causat per problemes d'injecció.

**Detecció:**
- Dos pics molt propers on hauria d'haver-n'hi un
- Forma característica de "orelles de Batman"
- Típicament per bombolla d'aire o injecció irregular

**Impacte:**
- L'àrea pot ser correcta (si s'integra tot)
- Però la forma del pic és anòmala
- El temps de retenció pot estar desplaçat

**Acció requerida:**
- BLOCKER en calibració (cal reinjectar)
- WARNING en anàlisi (revisar si l'àrea és acceptable)

---

## Comportament del Header

```
+------------------------------------------------------------------+
| NIVELL MAX  | INDICADOR        | BOTO AVISOS    | SEGUENT        |
+------------------------------------------------------------------+
| Cap avis    | [check] OK       | [check] (gris) | Enabled        |
| Nomes INFO  | [check] OK       | [i] Info       | Enabled        |
| WARNING     | [!] Revisar      | [!] Revisar    | Enabled* (nota)|
| BLOCKER     | [X] Bloquejat    | [X] Errors     | Disabled       |
+------------------------------------------------------------------+

* Quan hi ha WARNING i l'usuari prem "Seguent":
  -> Dialeg: "Hi ha avisos pendents. Afegeix una nota per continuar."
  -> Camp nota (obligatori) + Qui (obligatori)
  -> [Cancel] [Continuar amb nota]
```

---

## Estructura de Dades

Cada avís segueix aquesta estructura:

```python
{
    "code": "CAL_RSD_HIGH",           # Codi unic
    "level": "warning",                # "blocker", "warning", "info"
    "message": "RSD de repliques alt (12.3%)",
    "stage": "calibrate",              # "import", "calibrate", "analyze", "consolidate"
    "details": {                       # Detalls especifics
        "rsd": 12.3,
        "threshold": 10,
        "replicas": [1, 2]
    },
    "timestamp": "2026-02-06T10:30:00",
    "sample": null,                    # Si aplica a mostra especifica
    "condition_key": "BP_100_2",       # Si aplica a condicio especifica
    "dismissable": true,               # Pot ser ignorat amb nota?
    "dismissed": false,                # Ja s'ha ignorat?
    "dismissed_by": null,              # Qui l'ha ignorat
    "dismissed_note": null,            # Nota de dismissal
    "dismissed_at": null               # Quan s'ha ignorat
}
```

---

## Integració amb GUI

### Indicador d'Estat (Header)

L'indicador mostra el nivell màxim d'avisos pendents:

```python
def get_max_warning_level(warnings: list) -> str:
    """Retorna el nivell maxim d'avisos no dismissed."""
    levels = {"blocker": 3, "warning": 2, "info": 1, "none": 0}
    max_level = "none"

    for w in warnings:
        if not w.get("dismissed", False):
            level = w.get("level", "info")
            if levels.get(level, 0) > levels.get(max_level, 0):
                max_level = level

    return max_level
```

### Botó Avisos

El text i color del botó canvien segons l'estat:

| Estat | Text | Color | Acció al clicar |
|-------|------|-------|-----------------|
| Cap avís | "OK" | Verd (disabled) | - |
| INFO pendents | "i Info" | Blau | Mostra llista INFO |
| WARNING pendents | "! Revisar" | Taronja | Mostra llista + permet dismiss |
| BLOCKER pendents | "X Errors" | Vermell | Mostra llista (no dismiss) |
| Tots dismissed | "MMS" (revisor) | Verd | Permet revertir |

### Diàleg de Revisió

Quan l'usuari revisa avisos WARNING:

1. Mostra llista d'avisos pendents
2. Permet seleccionar quins "acceptar"
3. Requereix nota explicativa
4. Requereix nom/inicials del revisor
5. Guarda al JSON corresponent

---

## Implementació

### Fitxers

- **`hpsec_warnings.py`**: Mòdul centralitzat amb:
  - `create_warning()`: Crea avisos estructurats
  - `get_max_warning_level()`: Retorna nivell màxim (blocker/warning/info/none)
  - `WarningLevel`: Enum amb nivells
  - `WARNING_DEFINITIONS`: Definicions de tots els codis d'avís
  - `migrate_warnings_list()`: Converteix avisos antics (strings) al nou format
  - `create_warnings_from_timeout_info()`: Converteix info timeout a avisos
  - `create_warnings_from_batman_info()`: Converteix info batman a avisos

- **`hpsec_core.py`**: Funcions de detecció (ja existents):
  - `detect_timeout()`: Detecta pauses del TOC per intervals temporals
  - `detect_batman()`: Detecta patró doble pic (orelles de Batman)

- **`hpsec_import.py`**: `_generate_import_warnings()` genera avisos a Fase 1
- **`hpsec_calibrate.py`**: `_generate_calibration_warnings()` genera avisos a Fase 2
- **`hpsec_analyze.py`**: `_generate_analysis_warnings()` genera avisos a Fase 3

### Camps nous als resultats

Cada funció principal afegeix:
```python
result["warnings_structured"] = [...]  # Llista d'avisos estructurats
result["warning_level"] = "none"       # Nivell màxim: blocker/warning/info/none
```

### GUI

- `ProcessWizardPanel._get_warning_level()`: Llegeix nivell dels resultats
- `ProcessWizardPanel._get_warnings_list()`: Obté llista d'avisos
- `WarningReviewDialog`: Mostra avisos amb icones i colors per nivell

---

## Historial de Canvis

| Data | Canvi |
|------|-------|
| 2026-02-06 | Implementació completa: hpsec_warnings.py, integració backend i GUI |
| 2026-02-06 | Creació del document |

