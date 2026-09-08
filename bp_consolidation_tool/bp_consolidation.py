"""
bp_consolidation.py — Consolidació COLUMN↔BP (mòdul standalone)
================================================================

Aquest mòdul és la LLAVOR d'un programari a part per a la consolidació/
comparació de mostres entre seqüències COLUMN i BP (vincular una mostra de
columna amb el seu BP, combinar-los, i afegir un full BP a un Excel).

Es va EXTREURE de la HPSEC Suite el 2026-09-08 (rumb de simplificació): la
Suite processa COLUMN i BP com a modes independents, però NO els compara ni
els consolida — això viu aquí, fora del pipeline. El mode BP (processar una
SEQ BP pel seu compte) es manté dins la Suite.

Estat: codi preservat tal com estava a `hpsec_consolidate.py` (ja estava
orfe dins la Suite en el moment de l'extracció). Autocontingut: no depèn de
la Suite tret de `hpsec_import.load_manifest` (import diferit, opcional).

Funcions públiques:
- find_related_sequences   — COLUMN i BP del mateix lot (±rang)
- find_matching_bp_sequence — BP que correspon a una COLUMN (proximitat)
- load_bp_data_for_sample  — dades BP d'una mostra concreta
- consolidate_sample_data  — combina COLUMN + BP en una estructura
- find_all_sample_versions — totes les versions d'una mostra a diferents SEQ
- find_bp_for_samples      — cerca BP per a un conjunt de mostres COLUMN
- add_bp_sheet_to_excel    — afegeix un full BP a un Excel existent
"""

__version__ = "1.0.0"
__extracted_from__ = "hpsec_consolidate.py @ HPSEC Suite (2026-09-08)"

import re
from pathlib import Path
from typing import List, Dict, Optional


# --------------------------------------------------------------------------- #
# Utilitats d'identitat de seqüència (còpia autònoma; també viuen a la Suite)
# --------------------------------------------------------------------------- #
def extract_seq_number(seq_name: str) -> Optional[int]:
    """Número de seqüència del nom (ex: '286_SEQ_COLUMN' -> 286), o None."""
    match = re.match(r'^(\d+)(?:_|$)', seq_name)
    if match:
        return int(match.group(1))
    match = re.search(r'_(\d+)(?:_|$)', seq_name)
    if match:
        return int(match.group(1))
    return None


def detect_seq_type(seq_name: str) -> str:
    """Tipus de seqüència: 'COLUMN', 'BP' o 'UNKNOWN'."""
    name_upper = seq_name.upper()
    if "_BP" in name_upper or "BYPASS" in name_upper:
        return "BP"
    elif "COLUMN" in name_upper or "_C_" in name_upper or name_upper.endswith("_C"):
        return "COLUMN"
    elif name_upper.endswith("_SEQ") or re.match(r'^\d+[A-Z]?_SEQ$', name_upper):
        return "COLUMN"
    else:
        return "UNKNOWN"


def _normalize_sample_name(name: str) -> str:
    """Normalitza nom de mostra per comparació."""
    normalized = name.strip().upper()
    normalized = re.sub(r'[\s_-]+', '_', normalized)
    return normalized


# --------------------------------------------------------------------------- #
# Cerca de seqüències relacionades
# --------------------------------------------------------------------------- #
def find_related_sequences(
    seq_path: str,
    search_range: int = 5,
) -> Dict[str, List[str]]:
    """
    Cerca seqüències relacionades (COLUMN i BP del mateix lot o propers).

    Returns dict amb: current, current_type, current_number, column_seqs, bp_seqs.
    """
    seq_path = Path(seq_path)
    parent_dir = seq_path.parent
    seq_name = seq_path.name

    current_number = extract_seq_number(seq_name)
    current_type = detect_seq_type(seq_name)

    result = {
        "current": str(seq_path),
        "current_type": current_type,
        "current_number": current_number,
        "column_seqs": [],
        "bp_seqs": [],
    }

    if current_number is None:
        return result
    if not parent_dir.exists():
        return result

    for item in parent_dir.iterdir():
        if not item.is_dir():
            continue
        item_name = item.name
        item_number = extract_seq_number(item_name)
        if item_number is None:
            continue
        if abs(item_number - current_number) > search_range:
            continue
        item_type = detect_seq_type(item_name)
        if item_type == "COLUMN":
            result["column_seqs"].append(str(item))
        elif item_type == "BP":
            result["bp_seqs"].append(str(item))

    result["column_seqs"].sort(key=lambda x: extract_seq_number(Path(x).name) or 0)
    result["bp_seqs"].sort(key=lambda x: extract_seq_number(Path(x).name) or 0)
    return result


def find_matching_bp_sequence(
    column_seq_path: str,
    search_range: int = 2,
) -> Optional[str]:
    """Troba la SEQ BP que correspon a una SEQ COLUMN (per proximitat de número)."""
    related = find_related_sequences(column_seq_path, search_range)
    if not related["bp_seqs"]:
        return None

    current_number = related["current_number"]
    if current_number is None:
        return related["bp_seqs"][0] if related["bp_seqs"] else None

    bp_with_distance = []
    for bp_path in related["bp_seqs"]:
        bp_number = extract_seq_number(Path(bp_path).name)
        if bp_number is not None:
            distance = abs(bp_number - current_number)
            bp_with_distance.append((distance, bp_number, bp_path))

    if not bp_with_distance:
        return None
    bp_with_distance.sort(key=lambda x: (x[0], -x[1]))
    return bp_with_distance[0][2]


def load_bp_data_for_sample(
    bp_seq_path: str,
    sample_name: str,
) -> Optional[Dict]:
    """Carrega les dades BP d'una mostra específica (analysis_result.json o manifest)."""
    import json
    try:
        from hpsec_import import load_manifest
    except Exception:
        load_manifest = None

    bp_path = Path(bp_seq_path)

    analysis_path = bp_path / "CHECK" / "data" / "analysis_result.json"
    if analysis_path.exists():
        try:
            with open(analysis_path, 'r', encoding='utf-8') as f:
                analysis = json.load(f)
            samples_grouped = analysis.get("samples_grouped", {})
            if sample_name in samples_grouped:
                return _extract_bp_summary(samples_grouped[sample_name], str(bp_path))
            for name, data in samples_grouped.items():
                if _normalize_sample_name(name) == _normalize_sample_name(sample_name):
                    return _extract_bp_summary(data, str(bp_path))
        except Exception:
            pass

    manifest = load_manifest(str(bp_path)) if load_manifest else None
    if manifest:
        samples = manifest.get("samples", {})
        for name, data in samples.items():
            if _normalize_sample_name(name) == _normalize_sample_name(sample_name):
                return {
                    "seq_path": str(bp_path),
                    "seq_name": bp_path.name,
                    "sample_name": name,
                    "found_in": "manifest",
                    "replicas": list(data.keys()) if isinstance(data, dict) else [],
                }
    return None


def _extract_bp_summary(sample_data: Dict, seq_path: str) -> Dict:
    """Extreu resum de dades BP d'una mostra."""
    selected = sample_data.get("selected", {})
    doc_replica = selected.get("doc", "1")
    replicas = sample_data.get("replicas", {})
    replica_data = replicas.get(doc_replica, {})
    quantification = sample_data.get("quantification", {})
    return {
        "seq_path": seq_path,
        "seq_name": Path(seq_path).name,
        "replica": doc_replica,
        "area_total": quantification.get("area_total"),
        "concentration_ppm": quantification.get("concentration_ppm"),
        "snr_direct": replica_data.get("snr_info", {}).get("snr_direct"),
        "snr_uib": replica_data.get("snr_info", {}).get("snr_uib"),
        "anomalies": replica_data.get("anomalies", []),
        "found_in": "analysis_result",
    }


def consolidate_sample_data(
    column_data: Dict,
    bp_data: Optional[Dict] = None,
) -> Dict:
    """Consolida dades COLUMN i BP d'una mostra en una sola estructura."""
    consolidated = {
        "column": column_data,
        "bp": bp_data,
        "has_bp": bp_data is not None,
    }
    if bp_data:
        consolidated["bp_concentration_ppm"] = bp_data.get("concentration_ppm")
        consolidated["bp_area_total"] = bp_data.get("area_total")
        consolidated["bp_seq"] = bp_data.get("seq_name")
    return consolidated


def find_all_sample_versions(
    sample_name: str,
    data_folder: str,
    seq_type: str = None,
) -> List[Dict]:
    """Troba totes les versions d'una mostra a diferents seqüències."""
    import json

    data_path = Path(data_folder)
    versions = []
    if not data_path.exists():
        return versions

    for seq_dir in data_path.iterdir():
        if not seq_dir.is_dir():
            continue
        seq_name = seq_dir.name
        detected_type = detect_seq_type(seq_name)
        if seq_type and detected_type != seq_type:
            continue
        analysis_path = seq_dir / "CHECK" / "data" / "analysis_result.json"
        if not analysis_path.exists():
            continue
        try:
            with open(analysis_path, 'r', encoding='utf-8') as f:
                analysis = json.load(f)
            samples_grouped = analysis.get("samples_grouped", {})
            normalized_target = _normalize_sample_name(sample_name)
            for name, data in samples_grouped.items():
                if _normalize_sample_name(name) == normalized_target:
                    versions.append({
                        "seq_path": str(seq_dir),
                        "seq_name": seq_name,
                        "seq_type": detected_type,
                        "seq_number": extract_seq_number(seq_name),
                        "sample_name": name,
                        "data": data,
                    })
                    break
        except Exception:
            continue

    versions.sort(key=lambda x: x.get("seq_number") or 0)
    return versions


def find_bp_for_samples(
    column_seq_path: str,
    sample_names: List[str],
    data_folder: str = None,
    search_range: int = 2,
) -> Dict:
    """
    Cerca dades BP per a un conjunt de mostres COLUMN.

    1. find_matching_bp_sequence() per proximitat (±search_range)
    2. Per cada mostra: load_bp_data_for_sample() des de la BP primària
    3. Mostres sense match: cerca ampliada per nom (si data_folder donat)
    4. find_related_sequences(search_range=10) per llistar BPs disponibles
    """
    result = {
        "primary_bp": None,
        "available_bps": [],
        "samples": {},
    }
    for name in sample_names:
        result["samples"][name] = {"bp_seq": None, "bp_data": None, "source": None}

    bp_path = find_matching_bp_sequence(column_seq_path, search_range)
    if bp_path:
        result["primary_bp"] = {"path": bp_path, "name": Path(bp_path).name}
        for name in sample_names:
            bp_data = load_bp_data_for_sample(bp_path, name)
            if bp_data:
                result["samples"][name] = {
                    "bp_seq": Path(bp_path).name,
                    "bp_data": bp_data,
                    "source": "proximity",
                }

    if data_folder is None:
        data_folder = str(Path(column_seq_path).parent)

    missing = [n for n in sample_names if result["samples"][n]["bp_data"] is None]
    if missing:
        for name in missing:
            versions = find_all_sample_versions(name, data_folder, seq_type="BP")
            if versions:
                col_num = extract_seq_number(Path(column_seq_path).name)
                if col_num is not None:
                    versions.sort(key=lambda v: abs((v.get("seq_number") or 0) - col_num))
                best = versions[0]
                bp_data = _extract_bp_summary(best["data"], best["seq_path"])
                result["samples"][name] = {
                    "bp_seq": best["seq_name"],
                    "bp_data": bp_data,
                    "source": "name_search",
                }

    related = find_related_sequences(column_seq_path, search_range=10)
    for bp in related.get("bp_seqs", []):
        bp_name = Path(bp).name
        result["available_bps"].append({
            "path": bp,
            "name": bp_name,
            "number": extract_seq_number(bp_name),
        })

    return result


def add_bp_sheet_to_excel(
    excel_path: str,
    bp_data: Dict,
) -> bool:
    """Afegeix un full 'BP' a un Excel existent."""
    import pandas as pd

    if not bp_data:
        return False
    try:
        bp_rows = [
            ("BP_SEQ", bp_data.get("seq_name", "")),
            ("BP_Replica", f"R{bp_data.get('replica', '?')}"),
            ("---", "---"),
            ("Area_Total", bp_data.get("area_total")),
            ("Concentration_ppm", bp_data.get("concentration_ppm")),
            ("SNR_Direct", bp_data.get("snr_direct")),
            ("SNR_UIB", bp_data.get("snr_uib")),
        ]
        anomalies = bp_data.get("anomalies", [])
        if anomalies:
            bp_rows.append(("---", "---"))
            bp_rows.append(("Anomalies", ", ".join(anomalies)))

        df_bp = pd.DataFrame(bp_rows, columns=["Field", "Value"])
        with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a',
                            if_sheet_exists='replace') as writer:
            df_bp.to_excel(writer, sheet_name="BP", index=False)
        return True
    except Exception as e:
        print(f"Error afegint full BP: {e}")
        return False


__all__ = [
    "extract_seq_number",
    "detect_seq_type",
    "find_related_sequences",
    "find_matching_bp_sequence",
    "load_bp_data_for_sample",
    "consolidate_sample_data",
    "find_all_sample_versions",
    "find_bp_for_samples",
    "add_bp_sheet_to_excel",
]
