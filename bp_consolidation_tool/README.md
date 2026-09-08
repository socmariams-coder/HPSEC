# bp_consolidation_tool — Consolidació COLUMN↔BP (fora de la Suite)

Llavor d'un **programari a part** per comparar/consolidar mostres entre
seqüències **COLUMN** i **BP**.

## Per què està aquí (i no a la Suite)

Rumb de simplificació (2026-09): la HPSEC Suite processa COLUMN i BP com a
**modes independents**. El **mode BP** (processar una SEQ BP pel seu compte)
**es manté dins la Suite**. El que surt és **només la consolidació/comparació**
COLUMN↔BP — vincular una mostra de columna amb el seu BP, combinar-los i
exportar-ho junts. Aquesta lògica viu aquí.

## Estat

`bp_consolidation.py` és el codi extret de `hpsec_consolidate.py` (2026-09-08).
En el moment de l'extracció ja estava **orfe** dins la Suite (cap cridador: la
consolidació s'havia desconnectat en el refactor de simplificació de juliol).
Es preserva íntegre com a punt de partida; **no s'ha reescrit ni validat de nou**.

Autocontingut tret d'un import diferit opcional (`hpsec_import.load_manifest`).

## Entrada principal

`find_bp_for_samples(column_seq_path, sample_names, data_folder=None, search_range=2)`
→ per a un conjunt de mostres COLUMN, troba la BP per proximitat i/o per nom, i
retorna les dades BP + les BPs disponibles per a un selector.

## Pendent (quan es reprengui com a programa)

Definir com s'engega (CLI/GUI pròpia), la política de matching COLUMN↔BP, i el
format de sortida consolidat. No forma part del pipeline de la Suite.
