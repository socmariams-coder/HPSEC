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

## Troballes 307/308 (03-11-2026) — primera passada
El VWD, tal com està connectat (aigües avall del DOC), **NO dona senyal d'analit
utilitzable** en aquestes dues seqüències:

1. **VWD1A** dominat per una **oscil·lació de línia de base periòdica** (~2,25 min,
   ±40–100 mAU), **idèntica en blancs (milliQ) i en mostres**. El "pic" dins la
   finestra d'elució del DOC té la mateixa amplitud que els lòbuls de fora → és un
   lòbul més de l'oscil·lació, no un pic. SNR finestra/base ≈ 1.
2. **VWD1B pràcticament pla (~0)** — amb prou feines dona senyal al segon λ. La raó
   `VWD1A/VWD1B ≈ 100` és només A dividit per un canal gairebé nul, no redundància
   espectral.
3. **Cap relació de retenció estable** VWD↔DOC: els lags de correlació creuada
   dispersen de −3 a +3 min i no coincideixen ni entre rèpliques.
4. **Bases de temps no registrades**: VWD arrenca a −4,6 min, pic DAD ~1,1 min, pic
   DOC ~3,5 min → tres offsets de detector; cal alinear per injecció abans de res.

Hipòtesi (a validar amb la tècnica): cel·la UV aigües avall de l'etapa OCD/TOC →
pulsació de flux / bombolles / índex de refracció i pressió = base periòdica; VWD1B
~nul suggereix config/λ/guany del 2n canal.
