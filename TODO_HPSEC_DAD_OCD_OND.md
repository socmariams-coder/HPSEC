# TODO: HPSEC-DAD-OCD-OND — Detecció de Nitrogen Orgànic Dissolt

## Context

L'evolució natural del mètode HPSEC-DAD-OCD és afegir un detector de nitrogen orgànic (OND)
per obtenir perfils DON fraccionats per pes molecular, complementant els perfils DOC existents.

### Per què?

1. **Els DBPs nitrogenats (N-DBPs)** són 10-100x més citotòxics i genotòxics que THMs/HAAs
   - Haloacetonitrils (HANs), nitrosamines (NDMA), halonitrometans
   - La Directiva EU 2020/2184 els monitorarà properament
2. **Saber quines fraccions de MW contenen nitrogen** permetria predir N-DBP formation
3. **El DAD ja conté informació sobre N** — proteïnes i aminoàcids absorbeixen a 200-220nm
4. **La tesi (Valentí-Quiroga 2024)** va demostrar que els models espectrals A254-S206-240
   predien THMs millor que DOC sol, però no va explorar DON ni predicció de N-DBPs

### Què hem après de l'anàlisi espectral

- PCA sobre dades Export3D mostra **3-4 components espectrals** distingibles pel DAD
- Component 1 (~94%): NOM aromàtic (húmics/fúlvics), espectre exponencial amb shoulder 270nm
- Component 2 (~6%): Senyal UV-C <220nm — barreja d'inorgànics (NO₃⁻, Br⁻) + matèria nitrogenada
- Component 3 (<0.5%): Variació espectral fina dins els húmics
- **El UV-C no pot distingir nitrogen orgànic de nitrogen inorgànic** — cal un detector químic (OND)
- Els anions elueixen a 23.2 min (NO₃⁻) i 23.2 min (Br⁻) amb el mètode actual (tampó fosfat pH 6.8 + 0.1M Na₂SO₄)
- Possible precolumna SAX de porus restringit (<30 Å) per eliminar anions inorgànics selectivament

### Arquitectura de detecció: DAD + Dual-λ post-OCD

L'addició d'un **detector dual-longitud d'ona a la sortida de l'OCD** obre una dimensió analítica
completament nova. L'OCD destrueix tota la matèria orgànica (UV-persulfat → CO₂), per tant:

```
                  [DAD]                    [Dual-λ post-OCD]           [Diferència]
                  (pre-oxidació)           (post-oxidació)             (calculada)
                  ─────────────            ─────────────────           ────────────
Orgànic:          Sí (HA, FA, prot...)     No (destruït)              = DAD - Dual-λ → ORGÀNIC PUR
Inorgànic:        Sí (NO₃⁻, Br⁻...)       Sí (intacte)              = 0 → confirmat inorgànic
```

**Senyals derivats (4 canals independents per cada λ del Dual):**

| Canal | Fórmula | Què mesura |
|-------|---------|-----------|
| **DAD(λ)** | Mesura directa | Absorbància total (orgànic + inorgànic) |
| **Post(λ)** | Mesura directa | Absorbància residual (només inorgànic) |
| **Org(λ)** | DAD(λ) − Post(λ) | Absorbància orgànica pura a λ |
| **Inorg(λ)** | Post(λ) | Absorbància inorgànica pura a λ |

**Selecció de les 2 longituds d'ona del detector dual:**

La selecció òptima depèn de què volem maximitzar:

| Objectiu | λ₁ | λ₂ | Justificació |
|----------|----|----|-------------|
| **Separar orgànic/inorgànic** | 210 | 254 | 210nm: màxima interferència inorgànica; 254nm: referència orgànica pura |
| **Detectar nitrogen orgànic** | 210 | 254 | Proteïnes/aminoàcids tenen ratio A210/A254 alt; post-OCD elimina la contribució orgànica |
| **Maximitzar info independent** | 208 | 330 | Segons anàlisi PCA: màxima informació no redundant amb DAD |
| **Compatibilitat amb tesi** | 220 | 254 | Centroides del clustering de Valentí-Quiroga (2024) |

**Recomanació: λ₁ = 210 nm, λ₂ = 254 nm**

Amb 210nm i 254nm al detector post-OCD obtenim:

1. **Post(210)**: senyal pur de NO₃⁻ + Br⁻ (anions que absorbeixen a 210nm, intactes post-oxidació)
2. **Post(254)**: hauria de ser ~0 (res d'inorgànic rellevant absorbeix a 254nm) → línia base de control
3. **Org(210) = DAD(210) − Post(210)**: absorbància orgànica a 210nm → proxy de material nitrogenat
4. **Org(254) = DAD(254) − Post(254)**: absorbància orgànica a 254nm ≈ DAD(254) → validació

**Nou índex: Ratio Org(210)/Org(254) per fracció SEC**

| Fracció | Org(210)/Org(254) esperat | Composició dominant |
|---------|---------------------------|---------------------|
| BioP | 3-5 (alt) | Proteïnes, polisacàrids amb N — fort a 210nm |
| HS | 1.5-2.0 | Aromàtics amb N heterocíclic moderat |
| BB | 1.2-1.5 | Àcids orgànics, fenols — poc N |
| LMW | 2-4 (alt) | Aminoàcids lliures — fort a 210nm |

Combinat amb DOC de l'OCD, es pot estimar DON indirectament:
- Si Org(210)/Org(254) > ratio de referència HA/FA (~1.7) → excés de material nitrogenat
- Excess_N_proxy = Org(210) − 1.7 × Org(254)
- Calibrar contra DON real (TN − DIN) per obtenir ppb N

### Avantatges del sistema DAD + Dual-λ post-OCD

1. **Separació directa orgànic/inorgànic** sense precolumna ni MCR-ALS
2. **Detecció de N-orgànic** via excess UV-C orgànic (Org(210) elevat sense contribució inorgànica)
3. **QC integrat**: Post(254) ≈ 0 serveix de control de qualitat del detector
4. **Retrocompatible**: no canvia res del mètode actual, només afegeix informació
5. **Validació creuada**: 4 canals × 5 fraccions = 20 paràmetres per mostra (vs 5 actuals amb DOC sol)

---

## Fase 0: Revisió bibliogràfica i viabilitat

### 0.1 Estat de l'art SEC-OND
- [ ] Revisar literatura sobre SEC acoblat a detectors de nitrogen:
  - Shimadzu TOC-L + TNM-L (Total Nitrogen Module via chemiluminescència)
  - Analytik Jena multi N/C
  - Sievers M9 + modificacions per TN
  - LC-OCD-OND (Huber et al., DOC-Labor — ja existeix comercialment?)
- [ ] Revisar papers de Her et al., Huber et al. sobre DON per HPSEC
- [ ] Contactar DOC-Labor (Karlsruhe) per disponibilitat OND acoblat a LC-OCD
- [ ] Revisar si el Sievers M9 actual pot generar senyal de nitrogen (subproducte de l'oxidació)

### 0.2 Principi de detecció OND
- [ ] Definir la química de detecció:
  - Oxidació UV-persulfat → N orgànic → NO → chemiluminescència (CLD)
  - O bé: oxidació → NO₃⁻ → reducció → NO → CLD
  - O bé: oxidació catalítica a alta T → N₂ → TCD
- [ ] Avaluar sensibilitat requerida (LOD objectiu: ~10 ppb N)
- [ ] Avaluar interferències: el tampó fosfat no conté N, però (NH₄)₂S₂O₈ de l'OCD sí!
  - **CRÍTIC**: L'oxidant del Sievers M9 és persulfat d'amoni → conté nitrogen
  - Caldrà canviar a persulfat de sodi (Na₂S₂O₈) o K₂S₂O₈ per l'OND
  - O posar l'OND en paral·lel amb oxidant diferent

### 0.3 Viabilitat amb instrumentació existent
- [ ] Avaluar si es pot afegir un detector TN en sèrie o paral·lel al Sievers M9
- [ ] Alternativa: detector standalone post-columna (split flow)
- [ ] Alternativa: modificar el Sievers M9 per capturar els gasos d'oxidació
- [ ] Pressupost estimat per cada opció

---

## Fase 1: Predicció espectral de DON (software, sense hardware nou)

### 1.1 Estimació indirecta de DON via DAD
Abans de tenir OND hardware, extreure el màxim del DAD per estimar contingut nitrogenat:

- [ ] Implementar MCR-ALS sobre dades Export3D per separar components espectrals
  - Component UV-C (<220nm) com a proxy de matèria nitrogenada + inorgànica
  - Amb correcció per NO₃⁻/Br⁻ coneguts (espectre de referència)
- [ ] Calcular índexs espectrals indicadors de nitrogen:
  - Ratio A210/A254 (proteïnes/aminoàcids absorbeixen preferentment a 210nm)
  - Absorció específica a 206nm (pic de NO₃⁻ i pèptids)
  - Ratio A280/A254 (triptòfan/tirosina vs aromàtics generals)
  - SUVA₂₁₀ per fracció com a proxy de contingut proteic
- [ ] Construir model predictiu DON vs DAD multi-λ:
  - Training set: mostres amb DON mesurat per mètode independent (TN - NO₃⁻ - NO₂⁻ - NH₄⁺)
  - Features: àrees integrades a 206, 210, 220, 254, 280nm per fracció SEC
  - Model PLS o Random Forest
- [ ] Validar amb estàndards: BSA (proteïna), aminoàcids (Trp, Phe, Glu), HA/FA (poc N)

### 1.2 Fingerprint nitrogen per fracció SEC
- [ ] Crear perfil "N-proxy" per cada fracció (BioP, HS, BB, LMW):
  - BioP: esperem alt N (proteïnes, polisacàrids amb N)
  - HS: N moderat (nitrogen heterocíclic, amines aromàtiques)
  - BB: N baix (àcids orgànics petits, pocs amb N)
  - LMW: aminoàcids lliures (alt N per unitat de C)
- [ ] Integrar el perfil N-proxy al software HPSEC Suite:
  - Nova columna "N-proxy" a la taula d'anàlisi
  - Ratio DOC/N-proxy per fracció (equivalent a C/N fraccionat)

---

## Fase 2: Hardware OND

### 2.1 Configuració instrumental
- [ ] Configuració de flux — TOT EN SÈRIE (sense splitter):

```
[Injector] → [Precolumna SAX?] → [SEC cols] → [DAD] → [OCD Sievers M9] → [Dual-λ]
                                                 │           │                  │
                                              101 λ        DOC(t)         210nm + 254nm
                                            pre-oxidació   C orgànic      post-oxidació
```

**5 senyals simultanis per cada punt temporal:**

| # | Senyal | Detector | Què conté |
|---|--------|----------|-----------|
| 1 | DAD(t, λ) | DAD pre-OCD | Espectre complet: orgànic + inorgànic |
| 2 | DOC(t) | Sievers M9 | Carboni orgànic dissolt (µg C/L) |
| 3 | Post210(t) | Dual-λ post-OCD | NO₃⁻(original) + NO₃⁻(de N-org oxidat) + Br⁻ |
| 4 | Post254(t) | Dual-λ post-OCD | ~0 (control qualitat: tot l'orgànic destruït) |

**Senyals derivats (calculats per software):**

| # | Senyal | Fórmula | Informació |
|---|--------|---------|-----------|
| 5 | Org210(t) | DAD(210) − Post210 | Pèrdua d'absorbància a 210nm per oxidació |
| 6 | Org254(t) | DAD(254) − Post254 | ≈ DAD(254) → validació |
| 7 | ΔNO₃(t) | Post210 − DAD_inorg(210) | NO₃⁻ NOU generat per oxidació de N-org → **proxy DON** |
| 8 | SUVA210(t) | Org210 / DOC | Absorbància orgànica específica a 210nm per unitat de C |
| 9 | N-index(t) | ΔNO₃ / DOC | Proxy de ratio N/C fraccionat per MW |

**La clau del senyal 7 (ΔNO₃):**

L'OCD oxida: R-NH₂ + persulfat + UV → CO₂ + NO₃⁻

Per tant:
- Pre-OCD a 210nm (DAD): NO₃⁻(mostra) + Org(210nm)
- Post-OCD a 210nm (Dual): NO₃⁻(mostra) + NO₃⁻(generat de N-org)
- La diferència NO ÉS simplement orgànic vs inorgànic!
- Post210 POT SER MÉS GRAN que DAD(210) si hi ha molt N orgànic convertit a NO₃⁻

Escenaris:

| Cas | DAD(210) | Post(210) | Post > DAD? | Interpretació |
|-----|----------|-----------|-------------|---------------|
| Mostra amb NOM aromàtic, poc N | Alt | Baix | No | Orgànic absorbia, destruït per OCD |
| Mostra amb molt NO₃⁻, poc NOM | Mitjà | Mitjà | ≈ Igual | Inorgànic intacte |
| Mostra amb proteïnes (alt N-org) | Alt | Alt | **Possible!** | N-org → NO₃⁻ nou compensa pèrdua orgànic |
| Blanc (MQ) | ~0 | ~0 | No | Referència |

**Experiment crític de validació:**
- Injectar BSA (proteïna, 16% N): Post210 hauria de ser > 0 (NO₃⁻ generat)
- Injectar HA (poc N, ~1%): Post210 ≈ 0 (poc NO₃⁻ generat)
- Injectar NO₃⁻ pur: Post210 ≈ DAD(210) (inorgànic intacte)

- [ ] Calibrar offset temporal DAD → OCD → Dual-λ (volum mort de l'OCD)
- [ ] Avaluar dispersió cromatogràfica (band broadening) dins l'OCD
  - L'OCD té reactor UV + mescla amb àcid i persulfat → dispersió inevitable
  - Cal deconvolucionar per recuperar resolució temporal
  - Comparar amplada de pic DAD vs Dual-λ amb estàndard PSS
- [ ] Verificar linealitat Post210 vs [NO₃⁻] a concentracions esperades (0.1-50 mg/L)
- [ ] Verificar que l'OCD converteix >95% del N-org a NO₃⁻ (rendiment d'oxidació)
  - Test amb aminoàcids de MW conegut: Trp, Phe, Glu
  - Comparar Post210 amb TN teòric

### 2.2 Selecció del detector OND
- [ ] Opció A: **Shimadzu TNM-L** acoblat a TOC-L
  - Pro: comercialment disponible, chemiluminescència, LOD ~5 ppb N
  - Contra: cal segon analitzador (no pot compartir Sievers M9)
- [ ] Opció B: **Analytik Jena multi N/C**
  - Pro: mesura simultània C i N
  - Contra: substitutiu del Sievers M9, no complementari
- [ ] Opció C: **Detector CLD standalone** post-oxidació
  - Pro: afegible al setup actual
  - Contra: cal desenvolupament custom
- [ ] Opció D: **DOC-Labor LC-OCD-OND** (si existeix)
  - Pro: solució integrada, provada per NOM
  - Contra: substitueix tot el setup

### 2.3 Eliminació d'interferència amoni del persulfat
- [ ] Si OND comparteix oxidació amb OCD:
  - Canviar (NH₄)₂S₂O₈ → Na₂S₂O₈ o K₂S₂O₈
  - Validar que el canvi d'oxidant no afecta la recuperació DOC del Sievers M9
  - Mesurar blanc de nitrogen amb el nou oxidant
- [ ] Si OND té oxidació independent:
  - Usar Na₂S₂O₈ només per l'OND
  - Mantenir (NH₄)₂S₂O₈ per l'OCD (no canviar res)

### 2.4 Precolumna SAX per anions
- [ ] Seleccionar cartutx guard SAX amb porus <30 Å:
  - Agilent PL-SAX guard?
  - Waters IC guard?
  - Thermo Dionex IonPac AG?
- [ ] Testejar amb estàndards:
  - NO₃⁻ 50 mg/L → ha de desaparèixer el pic a 23.2 min
  - HA/FA → perfil DOC idèntic amb i sense precolumna
  - EP real → comparar amb/sense precolumna
- [ ] Determinar capacitat i freqüència de regeneració
  - El Na₂SO₄ 0.1M de l'eluent hauria d'auto-regenerar entre injeccions
  - Monitorar capacitat cada 50 injeccions

---

## Fase 3: Calibratge i validació OND

### 3.1 Estàndards de nitrogen orgànic
- [ ] Preparar estàndards de calibratge DON:
  - BSA (Bovine Serum Albumin, 16% N) → BioP fracció
  - L-triptòfan (13.7% N, MW 204, Rt ~51 min segons tesi)
  - L-fenilalanina (8.5% N, MW 165, Rt ~30 min)
  - L-glutamina (19.2% N, MW 146, Rt ~24.5 min)
  - Suwannee HA (0.7-1.2% N) i FA (0.7% N) → referència NOM
  - Glicina (18.7% N, MW 75) → LMW
- [ ] Construir corba de calibratge DON per fracció
- [ ] Determinar LOD i LOQ per fracció

### 3.2 Ratio DOC/DON fraccionat
- [ ] Calcular C/N per fracció SEC com a indicador de composició:

| Fracció | DOC/DON esperat | Interpretació |
|---------|-----------------|---------------|
| BioP | 5-10 | Alt N: proteïnes, polipèptids |
| HS | 20-50 | N moderat: N heterocíclic |
| BB | 30-100 | Poc N: àcids orgànics, fenols |
| LMW-àcids | 10-30 | Variable: aminoàcids vs àcids orgànics |
| LMW-neutres | 5-20 | Aminoàcids, urea |

- [ ] Validar amb aigües reals (EP, SP, POST_O3)
- [ ] Comparar amb DON bulk (TN - DIN) per tancament de balanç

### 3.3 Validació creuada DAD vs OND
- [ ] Amb dades OND reals, validar els models predictius de Fase 1:
  - El proxy A210/A254 correla amb DON mesurat?
  - El MCR-ALS component UV-C correla amb DON?
  - Quina fracció SEC té millor/pitjor predicció?
- [ ] Quantificar el límit del DAD per predir DON
  - En quines condicions el proxy espectral és suficient?
  - Quan cal el detector OND real?

---

## Fase 4: Integració al software HPSEC Suite

### 4.1 Importació dades OND
- [ ] Definir format de fitxer OND (CSV? integrat amb OCD?)
- [ ] Afegir canal OND a hpsec_consolidate.py (al costat de DOC i DAD)
- [ ] Sincronitzar temporalment OND amb DOC i DAD (offset configurable)

### 4.2 Processament DON
- [ ] Calcular àrees DON per fracció (mateixa lògica que DOC)
- [ ] Calcular DOC/DON ratio per fracció
- [ ] Quantificar DON en ppb N per fracció (calibratge amb estàndards)
- [ ] Afegir columnes DON a la taula d'anàlisi i exports

### 4.3 Visualització
- [ ] Cromatograma triple overlay: DOC + DAD(254nm) + DON
- [ ] Gràfic DOC/DON ratio vs temps de retenció
- [ ] Stacked bar chart: DOC vs DON per fracció
- [ ] Dashboard: DON total, DON per fracció, C/N per fracció

### 4.4 Models predictius N-DBP
- [ ] Integrar models THMs-FP existents (tesi Valentí-Quiroga) amb DON
- [ ] Desenvolupar models N-DBPs-FP:
  - HANs-FP = f(DON_HS, DON_BB, Br⁻, Cl₂_dosi)
  - NDMA-FP = f(DON_BioP, DON_LMW, ...)
- [ ] Comparar poder predictiu: DOC-only vs DOC+DAD vs DOC+DAD+DON

---

## Fase 5: Publicació i difusió

### 5.1 Article mètode
- [ ] "HPSEC-DAD-OCD-OND: Simultaneous size-fractionated DOC and DON profiling
       for drinking water NOM characterization"
- [ ] Contingut: mètode, calibratge, validació, aplicació a 3 DWTPs (PTL/PTT/PTC)
- [ ] Revista objectiu: Water Research o Environmental Science & Technology

### 5.2 Article aplicació
- [ ] "Fractionated DON as predictor of nitrogenous DBP formation in chlorinated
       drinking waters"
- [ ] Contingut: DON per fracció → correlació amb HANs, NDMA, etc.
- [ ] Comparar amb models DOC-only de la tesi

### 5.3 Software
- [ ] Publicar HPSEC Suite amb mòdul OND com a open-source tool
- [ ] Documentar mètode d'anàlisi espectral (MCR-ALS, proxy DON per DAD)

---

## Dependències i riscos

| Risc | Impacte | Mitigació |
|------|---------|-----------|
| Persulfat d'amoni interfireix amb OND | Alt | Canviar a Na₂S₂O₈, validar OCD |
| Precolumna SAX retén húmics | Alt | Verificar porus <30 Å, comparar amb/sense |
| LOD OND insuficient per aigües tractades | Mitjà | Augmentar volum injecció, concentrar |
| DAD no pot predir DON acuradament | Baix | Per això cal Fase 2 (hardware OND) |
| Splitter degrada resolució SEC | Mitjà | Minimitzar volum mort, ajustar split ratio |

---

## Timeline estimat

| Fase | Durada | Prerequisits |
|------|--------|-------------|
| Fase 0: Revisió bibliogràfica | 1-2 mesos | — |
| Fase 1: Proxy DAD (software) | 2-3 mesos | Dades Export3D existents |
| Fase 2: Hardware OND | 6-12 mesos | Pressupost, compra detector |
| Fase 3: Calibratge | 3-4 mesos | Hardware instal·lat |
| Fase 4: Software | 2-3 mesos | Primeres dades OND |
| Fase 5: Publicació | 6-12 mesos | Dades completes |
