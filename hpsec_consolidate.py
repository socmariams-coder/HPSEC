"""
hpsec_consolidate.py - Utilitats d'identitat de seqüència HPSEC
================================================================

Utilitats petites i genèriques per identificar seqüències pel nom:
- extract_seq_number(): número de SEQ del nom
- detect_seq_type():    COLUMN / BP / UNKNOWN

Usat per: history_panel, samples_db (dashboard + backend), hpsec_calibrate.

NOTA (2026-09-08): la consolidació/comparació COLUMN↔BP (find_bp_for_samples,
find_matching_bp_sequence, load_bp_data_for_sample, add_bp_sheet_to_excel...)
s'ha EXTRET a `bp_consolidation_tool/` (programari a part). La Suite processa
COLUMN i BP com a modes independents i no els consolida. Vegeu el README d'aquell
directori. El mode BP (processar una SEQ BP sola) es manté dins la Suite.
"""

__version__ = "2.0.0"
__version_date__ = "2026-09-08"

import re
from typing import Optional


def extract_seq_number(seq_name: str) -> Optional[int]:
    """
    Extreu el número de seqüència del nom.

    Examples:
        "286_SEQ_COLUMN" -> 286
        "285_SEQ_BP" -> 285
        "SEQ_287_TEST" -> 287
    """
    # Patró: número al principi (seguit de _ o final de nom)
    match = re.match(r'^(\d+)(?:_|$)', seq_name)
    if match:
        return int(match.group(1))

    # Patró: _número_ o _número al final
    match = re.search(r'_(\d+)(?:_|$)', seq_name)
    if match:
        return int(match.group(1))

    return None


def detect_seq_type(seq_name: str) -> str:
    """
    Detecta el tipus de seqüència (COLUMN o BP).

    Convenció:
    - *_SEQ_BP, *_BP -> BP mode
    - *_SEQ (sense sufix), *_SEQ_COLUMN, *_C -> COLUMN mode
    """
    name_upper = seq_name.upper()

    # Primer comprovar BP (més específic)
    if "_BP" in name_upper or "BYPASS" in name_upper:
        return "BP"
    # Després COLUMN (o sense sufix = COLUMN per defecte)
    elif "COLUMN" in name_upper or "_C_" in name_upper or name_upper.endswith("_C"):
        return "COLUMN"
    elif name_upper.endswith("_SEQ") or re.match(r'^\d+[A-Z]?_SEQ$', name_upper):
        # Pattern: 256_SEQ, 256B_SEQ -> COLUMN (sense sufix = columna)
        return "COLUMN"
    else:
        return "UNKNOWN"


__all__ = [
    "extract_seq_number",
    "detect_seq_type",
]
