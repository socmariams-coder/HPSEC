# Optimització de Seqüències HPSEC: Gestió de Timeouts TOC

## 1. Resum Executiu

L'analitzador TOC (Sievers M9e) realitza una recàrrega de xeringues cada **77.2 minuts**, causant un timeout de **74 segons**. Aquest document descriu com optimitzar les seqüències HPLC per garantir que els timeouts mai afectin la zona crítica del cromatograma.

### Solució Proposada
- Canviar la duració de mostra HPLC de 78.65 min a **77.2 min**
- Aplicar un temps d'espera de **3-5 minuts** després de l'estabilització/flush
- Resultat: **100% dels timeouts en zona post-run** (>70 min)

---

## 2. Paràmetres del Sistema

### 2.1 Paràmetres Verificats (6 seqüències)

| Paràmetre | Valor | Precisió | Font |
|-----------|-------|----------|------|
| Cicle recàrrega xeringa | **77.20 min** | ±0.009 min | Mesurat de 2-TOC |
| Mesures per cicle | 1140 | - | Manual equip |
| Interval entre mesures | ~4.06 s | - | Calculat |
| Duració timeout | **74 segons** | - | Observat |
| Duració mostra HPLC actual | **78.65 min** | ±0.01 min | Mesurat de 1-HPLC-SEQ |
| Deriva actual per mostra | **1.45 min** | ±0.01 min | 78.65 - 77.20 |

### 2.2 Zones del Cromatograma

| Zona | Rang (min) | Impacte d'un Timeout |
|------|------------|----------------------|
| BioP | 0-18 | CRÍTIC - Pèrdua de pics biopolímers |
| HS | 18-23 | CRÍTIC - Pèrdua de substàncies húmiques |
| BB | 23-30 | CRÍTIC - Pèrdua de building blocks |
| SB | 30-40 | CRÍTIC - Pèrdua de small building blocks |
| LMW | 40-70 | Acceptable - Zona de baix pes molecular |
| **Post-run** | **70-77** | **IDEAL - Sense pics d'interès** |

---

## 3. Model Predictiu de Timeouts

### 3.1 Fórmula de Predicció

```
Pos_timeout(N) = T0 - (N-1) × Deriva    (mod Cicle)
```

On:
- **T0** = temps del primer timeout dins la primera mostra (minuts)
- **N** = número de mostra (1, 2, 3...)
- **Deriva** = Duració_mostra - Cicle_recàrrega
- **Cicle** = 77.2 min

### 3.2 Verificació del Model

| Mètrica | Valor |
|---------|-------|
| Correlació model vs real | **r = 0.9982** |
| Error mitjà de predicció | **1.2%** |
| Seqüències analitzades | 6 |
| Total mostres | 150 |

### 3.3 Resultats per Seqüència

| Seq | Injeccions | T_mostra | Cicle | Delay | T0 | % Crític Real | % Crític Predit |
|-----|------------|----------|-------|-------|-----|---------------|-----------------|
| 272_SEQ | 29 | 78.7 min | 77.2 min | 9.2 min | 48.8 min | 79.3% | 75.9% |
| 274_SEQ | 29 | 78.7 min | 77.2 min | 7.4 min | 42.4 min | 93.1% | 93.1% |
| 275_SEQ | 29 | 78.7 min | 77.2 min | 10.7 min | 20.5 min | 51.7% | 51.7% |
| 278_SEQ | 29 | 78.7 min | 77.2 min | 12.0 min | 63.4 min | 41.4% | 44.8% |
| 282_SEQ | 19 | 78.7 min | 77.2 min | 12.4 min | 0.5 min | 5.6% | 5.3% |
| 283_SEQ | 15 | 78.7 min | 77.2 min | 12.1 min | 33.0 min | 100.0% | 100.0% |

---

## 4. Relació T0 vs % Timeouts Crítics

### 4.1 Corba Teòrica

La relació entre T0 i el percentatge de timeouts en zona crítica és **no-lineal** (parabòlica):

| Rang T0 (min) | % Timeouts Crítics | Avaluació |
|---------------|-------------------|-----------|
| 0-10 | 5-15% | EXCEL·LENT |
| 10-20 | 15-50% | Acceptable |
| 20-35 | 50-95% | DOLENT |
| **35-50** | **90-100%** | **PITJOR CAS** |
| 50-65 | 50-80% | Dolent |
| **65-77** | **10-40%** | **BON CAS** |

### 4.2 Gràfic de Referència

![T0 vs % Timeouts Crítics](analisi_completa_T0.png)

*El gràfic mostra les 6 seqüències analitzades superposades amb la corba teòrica del model.*

---

## 5. Escenari Optimitzat: Duració = 77.2 min

### 5.1 Canvi Proposat

| Paràmetre | Actual | Proposat |
|-----------|--------|----------|
| Duració mostra | 78.65 min | **77.2 min** |
| Deriva per mostra | 1.45 min | **0 min** |
| Control del timeout | Variable | **Total** |

### 5.2 Avantatges

1. **Deriva = 0**: El timeout cau sempre a la mateixa posició (T0) per a TOTES les mostres
2. **Control total**: Si T0 > 40 min → 0% timeouts crítics garantit
3. **Predictibilitat**: El tècnic pot decidir exactament on caurà el timeout

### 5.3 Gràfic Comparatiu

![Escenari 77.2 min vs 78.65 min](escenari_77min.png)

---

## 6. Protocol Optimitzat amb Flush Inicial

### 6.1 Principi

Segons el manual de l'equip, quan s'inicia una anàlisi es fa un **flush de xeringues**. Això significa:
- Les xeringues queden plenes (posició 0 del cicle)
- El primer timeout serà exactament a 77.2 min des del flush
- **T0 = 77.2 - X**, on X és el temps d'espera després del flush

### 6.2 Càlcul del Temps d'Espera

| Espera (X) | T0 = 77.2 - X | Zona | Resultat |
|------------|---------------|------|----------|
| 0 min | 77.2 min | Límit | ⚠️ Timeout solapa amb inici mostra 2 |
| **1 min** | **76.2 min** | **Post-run** | **IDEAL** |
| **3 min** | **74.2 min** | **Post-run** | **IDEAL** |
| **5 min** | **72.2 min** | **Post-run** | **IDEAL** |
| **7 min** | **70.2 min** | **Post-run** | **IDEAL** |
| 10 min | 67.2 min | LMW (final) | Acceptable |
| 37 min | 40.2 min | SB/LMW límit | ⚠️ Límit zona crítica |

### 6.3 Protocol Recomanat

```
╔════════════════════════════════════════════════════════════════╗
║                 PROTOCOL D'INICI DE SEQÜÈNCIA                  ║
╠════════════════════════════════════════════════════════════════╣
║                                                                ║
║  1. PREPARACIÓ                                                 ║
║     - Preparar mostres i seqüència HPLC                        ║
║     - Verificar que el mètode HPLC té duració = 77.2 min       ║
║                                                                ║
║  2. ESTABILITZACIÓ TOC                                         ║
║     - Iniciar estabilització/flush del TOC                     ║
║     - Esperar que completi el flush de xeringues               ║
║                                                                ║
║  3. TEMPS D'ESPERA                                             ║
║     - Esperar 3-5 minuts després del flush                     ║
║     - (Temps per verificacions finals)                         ║
║                                                                ║
║  4. INICI SEQÜÈNCIA                                            ║
║     - Iniciar seqüència HPLC                                   ║
║     - T0 serà ≈ 72-74 min (zona post-run)                      ║
║                                                                ║
║  RESULTAT: 100% dels timeouts en zona post-run (>70 min)       ║
║            per a TOTES les mostres de la seqüència             ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
```

---

## 7. Taula de Referència Ràpida

### 7.1 Si NO es pot canviar la duració (78.65 min)

| Situació | Acció | Resultat |
|----------|-------|----------|
| Seqüència curta (<15 mostres) | Observar timeout, esperar fins T0 > 60 min | ~85% mostres segures |
| Seqüència llarga (>15 mostres) | Posar mostres importants a posicions 1-14 | Minimitza impacte |
| T0 entre 35-50 min | Considerar esperar o dividir seqüència | Evita pitjor cas |

### 7.2 Si es canvia la duració a 77.2 min

| Situació | Acció | Resultat |
|----------|-------|----------|
| Qualsevol seqüència | Flush + esperar 3-5 min | **100% timeouts en post-run** |

---

## 8. Fitxers de Referència

### 8.1 Documents
| Fitxer | Descripció |
|--------|------------|
| `MODEL_PREDICTIU_TIMEOUT.md` | Model matemàtic detallat |
| `INFORME_TIMEOUT_RECARREGA.md` | Informe inicial de l'anàlisi |

### 8.2 Gràfics
| Fitxer | Descripció |
|--------|------------|
| `analisi_completa_T0.png` | T0 vs % Crítics (6 seqüències) |
| `escenari_77min.png` | Comparació 77.2 vs 78.65 min |
| `timeout_patterns_analysis.png` | Patrons de timeout |
| `timeout_by_concentration.png` | Anàlisi per concentració |

### 8.3 Dades
| Fitxer | Descripció |
|--------|------------|
| `comparacio_direct_uib_complet.pdf` | 331 pàgines de comparacions |
| `comparacio_direct_uib_complet_metrics.csv` | Mètriques de totes les mostres |

---

## 9. Conclusions

1. **El model és vàlid** (r = 0.9982) per predir la posició dels timeouts

2. **Amb duració actual (78.65 min)**:
   - Deriva de 1.45 min per mostra
   - 40-100% de mostres amb timeout crític (depenent de T0)

3. **Amb duració optimitzada (77.2 min)**:
   - Deriva = 0
   - Control total de la posició del timeout
   - **100% timeouts en post-run** amb protocol de flush + 3-5 min espera

4. **Recomanació**: Modificar la duració del mètode HPLC a **77.2 min** i aplicar el protocol d'espera post-flush.

---

## 10. Implementació: Detecció Automàtica de Timeouts

S'ha implementat la detecció automàtica de timeouts al mòdul de consolidació (`hpsec_consolidate.py`).

### 10.1 Funcionalitat

El sistema detecta automàticament els timeouts TOC analitzant la cadència de dades del DOC Direct:
- **Llindar timeout**: interval > 60 segons entre mesures consecutives
- **Timeout major**: interval > 70 segons (recàrrega de xeringues)

### 10.2 Nivells de Severitat

| Zona | Severitat | Descripció |
|------|-----------|------------|
| RUN_START | INFO | Timeout abans de BioP (t < 1 min) |
| BioP | WARNING | Timeout en zona de biopolímers (0-18 min) |
| **HS** | **CRITICAL** | Timeout en zona de substàncies húmiques (18-23 min) |
| BB | WARNING | Timeout en zona building blocks (23-30 min) |
| SB | WARNING | Timeout en zona small building blocks (30-40 min) |
| LMW | INFO | Timeout en zona low molecular weight (40-70 min) |
| POST_RUN | OK | Timeout en post-run (>70 min) - IDEAL |

### 10.3 Output al Consolidat

El full **ID** de cada fitxer consolidat ara inclou:

```
TOC_Timeout_Detected: YES/NO
TOC_Timeout_Count: [nombre de timeouts]
TOC_Timeout_Severity: OK/INFO/WARNING/CRITICAL
TOC_Timeout_1: [posició] min ([duració]s) - [zona] [severitat]
TOC_Zones_Affected: [llista de zones afectades]
TOC_Dt_Median_sec: [cadència normal en segons]
TOC_Dt_Max_sec: [interval màxim detectat]
```

### 10.4 Warnings a la Consolidació

Durant la consolidació, es generen warnings automàtics per timeouts en zones crítiques:

```
TIMEOUT: [mostra] - WARNING: Timeout 74s at 25.3 min (BB)
TIMEOUT: [mostra] - CRITICAL: Timeout 74s at 20.1 min (HS zone)
```

### 10.5 Fitxers Modificats

- `hpsec_consolidate.py`:
  - Afegida configuració `TIMEOUT_CONFIG`
  - Nova funció `detect_doc_timeouts()`
  - Nova funció `format_timeout_status()`
  - Modificades `extract_doc_from_masterfile()` i `extract_doc_from_master()`
  - Actualitzada `write_consolidated_excel()` per incloure info de timeout

---

*Document generat: 2025-01-26*
*Basat en l'anàlisi de 6 seqüències (150 mostres)*
*Model verificat amb correlació r = 0.9982*
*Detecció automàtica de timeouts implementada*
