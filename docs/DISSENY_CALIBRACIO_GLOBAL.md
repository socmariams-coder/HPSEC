# Disseny: Sistema de Calibració Global Versionat

## 1. Concepte

### Filosofia
- **rf_mass_cal** → Factor global per **quantificar** totes les mostres (separat BP/COLUMN)
- **KHP de cada SEQ** → Dues funcions:
  1. **QC check**: validar que rf_mass mesurat està dins rang
  2. **SHIFT temporal**: calcular desfase DOC-DAD per alinear pics
- **Calibracions versionades** → Cada canvi d'equip/mètode genera nova calibració
- **Traçabilitat** → Batch reprocessing usa la calibració vigent en la data original

### Fórmula de quantificació
```
ppm = Area × 1000 / (rf_mass_cal × volume_uL)

on:
  - Area: àrea del pic DOC (mAU·min)
  - rf_mass_cal: factor de calibració global vigent (mAU·min / µg DOC)
  - volume_uL: volum d'injecció (µL)
```

---

## 2. Estructura de Dades

### 2.1 Fitxer: `REGISTRY/Calibration_Reference.json`

```json
{
  "version": "2.0",
  "updated": "2026-02-07T15:30:00",

  "active_calibration_id": "CAL_20260207_001",

  "calibrations": [
    {
      "id": "CAL_20260207_001",
      "rf_mass_cal": {
        "direct": {"column": 682, "bp": 682},
        "uib": {"column": 682, "bp": 682}
      },
      "model": "origin",
      "r2": 0.9897,
      "n_points": 24,

      "valid_from": "2025-10-13",
      "valid_to": null,
      "is_active": true,

      "source": {
        "type": "historic_analysis",
        "description": "Anàlisi retrospectiu SEQs 275-285",
        "seq_references": ["275_SEQ", "276_SEQ", "278_SEQ", "282_SEQ", "283_SEQ", "285_SEQ"],
        "outliers_excluded": ["267_SEQ", "271_SEQ_BP", "272_SEQ"]
      },

      "conditions": {
        "method": "COLUMN",
        "volume_uL": 400,
        "khp_conc_ppm": [2.0, 5.0],
        "column_type": "Aquagel",
        "flow_rate": 0.75
      },

      "validation": {
        "tolerance_pct": 20,
        "warning_pct": 15
      },

      "metadata": {
        "created_date": "2026-02-07",
        "created_by": "system",
        "reason": "Calibració inicial basada en anàlisi històric",
        "notes": "Pendent validació amb SEQ dedicada"
      }
    },
    {
      "id": "CAL_HISTORIC_100uL",
      "rf_mass_cal": {
        "direct": {"column": 720, "bp": 720},
        "uib": {"column": 720, "bp": 720}
      },
      "model": "estimated",
      "r2": null,
      "n_points": 2,

      "valid_from": "2025-01-01",
      "valid_to": "2025-10-12",
      "is_active": false,

      "source": {
        "type": "estimated",
        "description": "Estimació per protocol antic 100µL",
        "seq_references": ["258_SEQ", "274_SEQ"],
        "notes": "Basat en SEQs amb volum confirmat 100µL"
      },

      "conditions": {
        "method": "COLUMN",
        "volume_uL": 100,
        "khp_conc_ppm": [1.0, 2.0, 5.0]
      },

      "metadata": {
        "created_date": "2026-02-07",
        "created_by": "system",
        "reason": "Calibració retroactiva per SEQs antigues"
      }
    }
  ],

  "qc_thresholds": {
    "rf_mass_deviation_warning_pct": 15,
    "rf_mass_deviation_fail_pct": 25,
    "min_r2_new_calibration": 0.98,
    "min_points_new_calibration": 5
  }
}
```

### 2.2 Fitxer: `REGISTRY/QC_History.json` (NOU)

Registre de QC checks de cada SEQ:

```json
{
  "version": "1.0",
  "entries": [
    {
      "seq_name": "285_SEQ",
      "seq_date": "2025-11-15",
      "calibration_id": "CAL_20260207_001",
      "khp_name": "KHP2",
      "khp_conc_ppm": 2.0,
      "volume_uL": 400,

      "measured": {
        "area": 575.6,
        "rf_mass": 719.4
      },

      "expected": {
        "rf_mass_cal": 682,
        "area_expected": 545.6
      },

      "qc_result": {
        "deviation_pct": 5.5,
        "status": "PASS",
        "message": "RF dins tolerància (5.5% vs ref 682)"
      },

      "timestamp": "2026-02-07T15:30:00"
    }
  ]
}
```

---

## 3. Funcions Backend

### 3.1 `hpsec_calibrate.py` - Funcions noves

```python
# === GESTIÓ CALIBRACIONS ===

def get_calibration_for_date(seq_date: str) -> dict:
    """
    Retorna la calibració vigent per una data donada.

    Args:
        seq_date: Data de la SEQ (YYYY-MM-DD o datetime)

    Returns:
        dict amb la calibració vigent o None
    """

def get_active_calibration() -> dict:
    """Retorna la calibració activa actual."""

def add_calibration(rf_mass_cal: float, source: dict,
                    valid_from: str, conditions: dict,
                    r2: float = None, n_points: int = None) -> str:
    """
    Afegeix una nova calibració i tanca l'anterior.

    Returns:
        ID de la nova calibració
    """

def close_calibration(cal_id: str, valid_to: str, reason: str) -> bool:
    """Tanca una calibració (defineix valid_to)."""

def list_calibrations() -> list:
    """Llista totes les calibracions ordenades per data."""


# === QUANTIFICACIÓ ===

def quantify_sample(area: float, volume_uL: float,
                    seq_date: str = None) -> dict:
    """
    Quantifica una mostra usant rf_mass_cal.

    Args:
        area: Àrea del pic
        volume_uL: Volum d'injecció
        seq_date: Data SEQ (per seleccionar calibració correcta)

    Returns:
        dict amb:
          - concentration_ppm
          - rf_mass_cal_used
          - calibration_id
    """


# === QC CHECK ===

def validate_khp_qc(khp_data: dict, seq_date: str) -> dict:
    """
    Valida el KHP d'una SEQ com a QC check.

    Compara el rf_mass mesurat vs rf_mass_cal vigent.

    Returns:
        dict amb:
          - status: "PASS", "WARNING", "FAIL"
          - deviation_pct
          - message
          - calibration_id
    """

def register_qc_result(seq_name: str, qc_result: dict) -> bool:
    """Registra el resultat QC a QC_History.json."""


# === CREAR NOVA CALIBRACIÓ DES DE SEQ ===

def create_calibration_from_seq(seq_path: str,
                                 khp_data_list: list) -> dict:
    """
    Crea una nova calibració a partir d'una SEQ de calibració.

    Args:
        seq_path: Path de la SEQ de calibració
        khp_data_list: Llista de dades KHP (múltiples concentracions)

    Returns:
        dict amb:
          - success: bool
          - calibration_id: ID nova calibració
          - rf_mass_cal: valor calculat
          - r2: coeficient determinació
          - plot_path: path al gràfic
    """
```

### 3.2 Simplificació `calibrate_from_import()`

```python
def calibrate_from_import(imported_data, config=None, progress_callback=None):
    """
    SIMPLIFICAT: Ja no busca siblings ni historial.

    Nou flux:
    1. Carregar rf_mass_cal vigent per la data de la SEQ
    2. Si hi ha KHP a la SEQ:
       a. Calcular rf_mass del KHP
       b. Comparar vs rf_mass_cal (QC check)
       c. Registrar resultat QC
    3. Retornar rf_mass_cal per quantificació

    El rf_mass_cal global s'usa per quantificar totes les mostres.
    El KHP local només serveix per validar l'estat de l'instrument.
    """
```

---

## 4. Flux de Processament

### 4.1 Processament SEQ Normal

```
1. Import SEQ
   ↓
2. Obtenir data SEQ
   ↓
3. get_calibration_for_date(seq_date) → rf_mass_cal
   ↓
4. Si SEQ té KHP:
   │  a. Analitzar KHP → rf_mass_measured
   │  b. validate_khp_qc() → QC status
   │  c. register_qc_result()
   │  d. Si FAIL → WARNING a l'usuari
   ↓
5. Quantificar mostres amb rf_mass_cal
   ↓
6. Guardar resultats
```

### 4.2 Processament SEQ de Calibració

```
1. Import SEQ (amb múltiples KHPs: 1ppm, 2ppm, 5ppm, etc.)
   ↓
2. Analitzar tots els KHPs
   ↓
3. Construir taula µg_DOC vs Area
   ↓
4. Regressió lineal (per origen)
   ↓
5. Validar:
   - R² >= 0.98?
   - n_points >= 5?
   - Residus acceptables?
   ↓
6. Si PASS:
   │  a. Mostrar gràfic a l'usuari
   │  b. Demanar confirmació
   │  c. add_calibration()
   │  d. close_calibration() anterior
   ↓
7. Retornar resultat
```

### 4.3 Batch Reprocessing

```
Per cada SEQ a reprocessar:
  1. Llegir data original de la SEQ
  2. get_calibration_for_date(data_original) → rf_mass_cal d'aquella època
  3. Reprocessar amb aquell rf_mass_cal
  4. Guardar resultats

Nota: NO usar la calibració actual, usar la vigent en la data de la SEQ
```

---

## 5. GUI

### 5.1 Nou Tab o Panel: "Calibració"

```
┌─────────────────────────────────────────────────────────────────────┐
│ 📊 Calibració Global                                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ CALIBRACIÓ ACTIVA                                                   │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │  rf_mass_cal = 682  mAU·min / µg DOC                           │ │
│ │  R² = 0.9897   |   n = 24 punts   |   Model: per origen        │ │
│ │  Vigent des de: 2025-10-13                                     │ │
│ │  Font: Anàlisi històric SEQs 275-285                           │ │
│ └─────────────────────────────────────────────────────────────────┘ │
│                                                                     │
│ VALIDACIÓ QC (últimes SEQs)                                         │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │ SEQ        Data       KHP    RF_mass   Dev%   Status           │ │
│ │ ────────────────────────────────────────────────────────────── │ │
│ │ 285_SEQ    2025-11-15 KHP2   719       +5.4%  ✓ PASS           │ │
│ │ 283_SEQ    2025-11-10 KHP2   788       +15.5% ⚠ WARNING        │ │
│ │ 282_SEQ    2025-11-08 KHP2   730       +7.0%  ✓ PASS           │ │
│ └─────────────────────────────────────────────────────────────────┘ │
│                                                                     │
│ [📈 Veure gràfic calibració]  [📜 Històric calibracions]           │
│                                                                     │
│ ─────────────────────────────────────────────────────────────────── │
│                                                                     │
│ CREAR NOVA CALIBRACIÓ                                               │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │ Per crear una nova calibració, processa una SEQ amb múltiples  │ │
│ │ concentracions de KHP (recomanat: 1, 2, 5 ppm).                 │ │
│ │                                                                 │ │
│ │ SEQ actual: [285_SEQ        ▼]  [🔬 Analitzar com calibració]  │ │
│ └─────────────────────────────────────────────────────────────────┘ │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 Diàleg: Històric de Calibracions

```
┌─────────────────────────────────────────────────────────────────────┐
│ Històric de Calibracions                                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ ID               RF_mass  R²      Vigent des de   Fins a    Status  │
│ ────────────────────────────────────────────────────────────────── │
│ CAL_20260207     682      0.990   2025-10-13      -         ACTIVA  │
│ CAL_HISTORIC     720      -       2025-01-01      2025-10-12 Tancada│
│                                                                     │
│ Detall seleccionat:                                                 │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │ ID: CAL_20260207                                                │ │
│ │ rf_mass_cal: 682 mAU·min/µg DOC                                 │ │
│ │ Model: Regressió per origen                                     │ │
│ │ R²: 0.9897                                                      │ │
│ │ Punts: 24                                                       │ │
│ │ Font: Anàlisi històric SEQs 275-285                             │ │
│ │ Condicions: COLUMN, 400µL, KHP 2-5ppm                           │ │
│ │ Creada: 2026-02-07 per system                                   │ │
│ │ Motiu: Calibració inicial basada en anàlisi històric            │ │
│ └─────────────────────────────────────────────────────────────────┘ │
│                                                                     │
│                                           [Tancar]                  │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.3 Diàleg: Nova Calibració

```
┌─────────────────────────────────────────────────────────────────────┐
│ Nova Calibració des de SEQ                                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ SEQ: 290_SEQ_CALIBRATION                                            │
│ Data: 2026-02-14                                                    │
│                                                                     │
│ KHPs detectats:                                                     │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │  ☑ KHP1 (1 ppm) - 2 rèpliques - Àrea: 145.2 ± 3.1              │ │
│ │  ☑ KHP2 (2 ppm) - 2 rèpliques - Àrea: 287.5 ± 5.2              │ │
│ │  ☑ KHP5 (5 ppm) - 2 rèpliques - Àrea: 712.8 ± 8.7              │ │
│ └─────────────────────────────────────────────────────────────────┘ │
│                                                                     │
│ Resultat regressió:                                                 │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │                                                                 │ │
│ │    [GRÀFIC µg DOC vs Area amb línia de regressió]              │ │
│ │                                                                 │ │
│ │    rf_mass_cal = 678 mAU·min / µg DOC                          │ │
│ │    R² = 0.9987  ✓                                              │ │
│ │    n = 6 punts  ✓                                              │ │
│ │                                                                 │ │
│ └─────────────────────────────────────────────────────────────────┘ │
│                                                                     │
│ Comparació amb calibració actual:                                   │
│   Actual: rf_mass_cal = 682                                         │
│   Nova:   rf_mass_cal = 678                                         │
│   Diferència: -0.6%  ✓ Dins tolerància                              │
│                                                                     │
│ Motiu del canvi: [____________________________________]             │
│                                                                     │
│ ⚠️ Crear nova calibració tancarà l'actual (vigent fins ahir)        │
│                                                                     │
│                    [Cancel·lar]  [✓ Crear calibració]               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 6. Migració

### 6.1 Passos de migració

1. **Crear `Calibration_Reference.json`** amb estructura v2.0
   - Calibració activa: rf_mass_cal = 682 (des de 2025-10-13)
   - Calibració històrica: rf_mass_cal = 720 (2025-01-01 a 2025-10-12)

2. **Crear `QC_History.json`** buit

3. **Actualitzar `hpsec_calibrate.py`**:
   - Afegir funcions gestió calibracions
   - Simplificar `calibrate_from_import()`
   - Afegir funcions QC

4. **Actualitzar GUI**:
   - Nou panel/tab calibració
   - Integrar QC check al processament

5. **Batch opcional**: Reprocessar SEQs existents amb nou sistema
   - Calcular QC retroactiu per SEQs amb KHP
   - Verificar consistència resultats

---

## 7. Casos d'Ús

### 7.1 Procesar SEQ normal (amb KHP)
```
1. Importar 286_SEQ (data: 2025-11-20)
2. Sistema detecta KHP2
3. get_calibration_for_date("2025-11-20") → CAL_20260207 (rf_mass_cal=682)
4. Analitzar KHP2 → rf_mass = 710
5. QC check: 710 vs 682 = +4.1% → PASS
6. Quantificar mostres amb rf_mass_cal = 682
7. Guardar QC result
```

### 7.2 Procesar SEQ normal (sense KHP)
```
1. Importar 287_SEQ (data: 2025-11-22, només mostres, sense KHP)
2. get_calibration_for_date("2025-11-22") → CAL_20260207 (rf_mass_cal=682)
3. No hi ha KHP → No QC check (warning informatiu)
4. Quantificar mostres amb rf_mass_cal = 682
```

### 7.3 Procesar SEQ antiga
```
1. Importar 260_SEQ (data: 2025-03-15)
2. get_calibration_for_date("2025-03-15") → CAL_HISTORIC (rf_mass_cal=720)
3. Quantificar mostres amb rf_mass_cal = 720 (no 682!)
```

### 7.4 Crear nova calibració
```
1. Importar 290_SEQ_CAL (SEQ dedicada amb KHP 1,2,5 ppm)
2. Analitzar tots els KHPs
3. Regressió → rf_mass_cal = 678, R² = 0.998
4. Mostrar gràfic i resultat a usuari
5. Usuari confirma
6. add_calibration(678, valid_from="2026-02-14")
7. close_calibration(CAL_20260207, valid_to="2026-02-13")
```

### 7.5 QC FAIL
```
1. Importar 291_SEQ (data: 2026-02-20)
2. Detectar KHP2
3. Analitzar KHP2 → rf_mass = 850
4. QC check: 850 vs 682 = +24.6% → FAIL
5. Mostrar WARNING: "⚠️ KHP fora de rang! Revisar instrument/mètode"
6. Quantificar mostres amb rf_mass_cal = 682 (però marcar com dubtoses?)
```

---

## 8. Consideracions

### 8.1 BP vs COLUMN
- **rf_mass_cal SEPARAT** per BP i COLUMN
- Més flexibilitat: si cal canviar un mode, no afecta l'altre
- Si són iguals, simplement es posa el mateix valor al registre
- L'anàlisi mostra que segueixen la mateixa tendència, però millor tenir-los separats per si de cas

### 8.2 Volum d'injecció
- El rf_mass_cal és independent del volum (és per µg DOC)
- El volum afecta el càlcul de µg DOC: `µg_DOC = conc × vol / 1000`
- La fórmula de quantificació inclou el volum

### 8.3 Canvis d'instrument/mètode
Situacions que requereixen nova calibració:
- Canvi de columna
- Canvi de detector
- Canvi de fase mòbil
- Canvi de flux
- Manteniment major

---

## 9. Resum Fitxers

| Fitxer | Ubicació | Funció |
|--------|----------|--------|
| `Calibration_Reference.json` | REGISTRY/ | Calibracions versionades |
| `QC_History.json` | REGISTRY/ | Historial QC checks |
| `hpsec_calibrate.py` | ./ | Backend calibració |
| `calibration_panel.py` | gui/widgets/ | GUI gestió calibració |

