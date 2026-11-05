# VWD vs senyal actual — exploració externa (NO Suite)

Mètode **en desenvolupament**: la sortida del detector DOC (OCD/TOC) s'ha connectat
a un detector UV de longitud d'ona variable (VWD) a **dues longituds d'ona**
(canals `VWD1A`, `VWD1B`). La Suite **no** llegeix aquest detector i **no s'ha de
tocar**. Aquest script tracta el VWD a banda, però s'integra amb el que la Suite
**ja processa** (DOC net + DAD 254 del pipeline real) per comparar-lo a fons contra
el senyal actual.

## Ús
```
python vwd_vs_actual.py 308_SEQ_BP            # una seq
python vwd_vs_actual.py 307_SEQ_BP 308_SEQ_BP # diverses
```
Sortides a `out_<SEQ>/` (regenerables, fora de git):
`<mostra>_overlay.png` (3 panells: superposició normalitzada · magnitud real VWD ·
raó espectral VWD1A/VWD1B), `_SUMMARY_<SEQ>.csv`, `_PANORAMA_<SEQ>.png`.

## Informe complet (document amb gràfiques)
`make_informe.py` genera `informe_vwd_307_308_v1.html` (autocontingut, figures
incrustades). Per obtenir el PDF: `make_informe.py` + Chrome headless `--print-to-pdf`
(workflow premium HTML→Chrome PDF). El document recull objecte/context, mètode, resultats
(oscil·lació de base ~2,25 min, SNR finestra/base ≈1, VWD1B viu ~100× menor, localització
a la cel·la VWD per λ aparellada, cap retenció estable), la refutació de la hipòtesi del
reactiu i el diagnòstic + següents passos. El PDF és regenerable (no es versiona a git).

## Troballes 307/308 (03-11-2026) — primera passada
El VWD, tal com està connectat (aigües avall del DOC), **NO dona senyal d'analit
utilitzable** en aquestes dues seqüències:

1. **VWD1A** dominat per una **oscil·lació de línia de base periòdica** (~2,25 min,
   ±40–100 mAU), **idèntica en blancs (milliQ) i en mostres**. El "pic" dins la
   finestra d'elució del DOC té la mateixa amplitud que els lòbuls de fora → és un
   lòbul més de l'oscil·lació, no un pic. SNR finestra/base ≈ 1.
2. **λ confirmades: VWD1A = 220 nm, VWD1B = 254 nm.** VWD1B NO és un canal mort:
   a la seva pròpia escala dona ~0,5–1,4 mAU (≈100× més petit que el 220, coherent
   amb l'absortivitat 220>254) i porta LA MATEIXA oscil·lació. (Correcció: una
   lectura inicial el va donar per "pla" només perquè es dibuixava a l'escala del 220.)
3. **Cap relació de retenció estable** VWD↔DOC: els lags de correlació creuada
   dispersen de −3 a +3 min i no coincideixen ni entre rèpliques.
4. **Bases de temps no registrades**: VWD arrenca a −4,6 min, pic DAD ~1,1 min, pic
   DOC ~3,5 min → tres offsets de detector; cal alinear per injecció abans de res.

**Localització (prova amb λ aparellada — `vwd_representa.py`):** el DAD mesura els
MATEIXOS 220 i 254 en línia (amunt). Al mateix λ, el **DAD és net** (pic a ~1,1 min,
base plana) i el **VWD oscil·la** (±100 mAU a 220). En un **blanc de milliQ** el DAD
220/254 és pla (res a absorbir) però el VWD oscil·la igual → l'artefacte NO és
cromatografia ni analit ni λ: és **la cel·la del VWD aigües avall del DOC**.

Hipòtesi (a validar amb la tècnica): cel·la UV just després de l'etapa OCD/TOC →
pulsació de flux / bombolles / índex de refracció i pressió = base periòdica ~2,25 min.

`vwd_representa.py`: per mostra, 2 files (220 i 254 nm) amb DAD (amunt) vs VWD (avall)
en mAU reals + DOC de referència. Sortides a `out_<SEQ>/matched/`.

## Correlació amb el reactiu del DOC — hipòtesi REFUTADA per la dosi real
Hipòtesi inicial: l'oscil·lació VWD = cicle d'injecció d'oxidant/àcid. `vwd_correla_reactiu.py`
donava un senyal periòdic (~2,3 min, 220 i 254 en fase, present en blancs, fase-lock baix)
que SEMBLAVA suportar-ho. **Però amb la dosi REAL llegida del full de DOC, NO lliga.**

Dosi real (full `2-TOC`, Sievers M9e; cols Acid/Oxidizer Flow Rate + timestamps):
- **Cabal CONSTANT**: àcid 5 µL/min, oxidant 2 µL/min (setpoints, no modulats).
- **Recàrrega de xeringa**: 6 cops en tota la seq, **cada ~77 min** (~74 s d'aturada).
  És l'únic esdeveniment periòdic del reactiu (quadra amb buidar una xeringa a 5 µL/min).

**Conclusió:** la recàrrega (~77 min) NO pot generar l'oscil·lació VWD, que té diversos
cicles DINS de cada injecció (~15 min). L'oscil·lació NO és la dosificació de reactiu.
El ~2,3 min del FFT era una mesura poc robusta sobre finestra curta amb components
barrejats (deriva lenta + ripple); no coincideix amb cap cicle real del reactiu.

**Obert:** l'oscil·lació s'introdueix ENTRE el DAD (net) i el VWD → dins/després del
TOC o el nou tram. Candidats (NO confirmats per dades, per a la tècnica): ripple de
lliurament de la xeringa doble, bombolles de l'oxidació UV, cicle tèrmic/mesura del M9.
Sincronització disponible per a correlació fina: `0-INFO` (Hora HPLC / Hora TOC / flush).
