# HPSEC Suite — TODO General

Data: 2026-02-14

---

## Prioritat ALTA — Pendent

- [ ] **MasterFiles pendents a Dades3 (4 SEQs)**: Cal regenerar el rawdata des d'Agilent:
  - `087_SEQ` — no té rawdata (només Export3D)
  - `229_SEQ_BP` — rawdata corrupte (columnes HPLC-SEQ desplaçades, col A buida)
  - `237_SEQ` — no té rawdata (només Export3D)
  - `240_SEQ` — no té rawdata (només Export3D)

---

## Prioritat MITJA — Fase avançada

- [ ] **Control de concurrència al REGISTRY (SharePoint/OneDrive)**: Si les dades estan a una
  carpeta SharePoint/Teams sincronitzada via OneDrive, múltiples usuaris processant seqüències
  simultàniament poden generar conflictes als JSON compartits (KHP_History.json, QC_History.json,
  Samples_Index.json). Cal implementar un mecanisme de bloqueig (file lock o similar) per evitar
  corrupcions. Requisit previ: marcar la carpeta OneDrive com "sempre disponible en aquest dispositiu".

- [ ] **Verificar alineament temporal cromatogrames Export3D**: Abans d'integrar mètriques BB
  temporals (early/late ratios, pic timing) cal confirmar que els cromatogrames de diferents
  seqüències estan alineats temporalment. Possibles causes de desalineament: variacions de flux,
  temperatura columna, degradació columna entre seqüències. Si no estan alineats, caldria un pas
  de time-warping o normalització temporal previ.

---

## Prioritat BAIXA — Futur

- [ ] **BB fingerprint a la Suite**: Integrar mètriques BB per mostra individual:
  (1) A254/A280 a BB — caràcter orgànic, (2) fracció inorgànica A210, (3) shape inorgànic
  (A200/A210, A220/A210). Prerequisit: verificar alineament temporal.
  Codi exploratori a `research/bb_exploration/`.

---

## Recerca completada (research/)

- [x] **Humic Character Index (HCI)** — PCA+LDA HA/FA, integrat a Suite (hpsec_humic.py)
- [x] **BB RF + PCA+LDA** — RF 100% CV, LDA 94.4% LOO, 249 features, 11 classes
- [x] **BB org/inorg separació** — A254=orgànic, excess A210=inorgànic, ratio PTLL/PTT 3x
- [x] **BB temporal** — Sub-estructura dins BB, timing aniònic diferent per tipus d'aigua
  - CAVEAT: alineament temporal no verificat
