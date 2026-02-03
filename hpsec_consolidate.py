"""
hpsec_consolidate.py - Consolidació multi-seqüència HPSEC
==========================================================

Funcionalitats:
- Cercar SEQs relacionades (COLUMN + BP del mateix lot)
- Vincular mostres per nom entre seqüències
- Combinar dades COLUMN + BP en una sola estructura
- Gestionar múltiples versions d'una mateixa mostra

Usat per:
- ExportPanel (afegir full BP a Excel COLUMN)
- Futur: Vista consolidada de mostres

"""

__version__ = "1.0.0"
__version_date__ = "2026-02-03"

import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple


def extract_seq_number(seq_name: str) -> Optional[int]:
    """
    Extreu el número de seqüència del nom.

    Args:
        seq_name: Nom de la seqüència (ex: "286_SEQ_COLUMN")

    Returns:
        Número de seqüència o None si no es pot extreure

    Examples:
        "286_SEQ_COLUMN" -> 286
        "285_SEQ_BP" -> 285
        "SEQ_287_TEST" -> 287
    """
    # Patró: número al principi seguit de _
    match = re.match(r'^(\d+)_', seq_name)
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

    Args:
        seq_name: Nom de la seqüència

    Returns:
        "COLUMN", "BP" o "UNKNOWN"
    """
    name_upper = seq_name.upper()

    if "COLUMN" in name_upper or "_C_" in name_upper or name_upper.endswith("_C"):
        return "COLUMN"
    elif "BP" in name_upper or "BYPASS" in name_upper:
        return "BP"
    else:
        return "UNKNOWN"


def find_related_sequences(
    seq_path: str,
    search_range: int = 5,
) -> Dict[str, List[str]]:
    """
    Cerca seqüències relacionades (COLUMN i BP del mateix lot o propers).

    Args:
        seq_path: Path de la seqüència actual
        search_range: Rang de números a cercar (±N)

    Returns:
        Dict amb:
            - "current": path actual
            - "current_type": "COLUMN" o "BP"
            - "current_number": número de seqüència
            - "column_seqs": llista de paths COLUMN trobades
            - "bp_seqs": llista de paths BP trobades
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

    # Cercar seqüències al directori pare
    if not parent_dir.exists():
        return result

    for item in parent_dir.iterdir():
        if not item.is_dir():
            continue

        item_name = item.name
        item_number = extract_seq_number(item_name)

        if item_number is None:
            continue

        # Filtrar per rang de números
        if abs(item_number - current_number) > search_range:
            continue

        item_type = detect_seq_type(item_name)

        if item_type == "COLUMN":
            result["column_seqs"].append(str(item))
        elif item_type == "BP":
            result["bp_seqs"].append(str(item))

    # Ordenar per número
    result["column_seqs"].sort(key=lambda x: extract_seq_number(Path(x).name) or 0)
    result["bp_seqs"].sort(key=lambda x: extract_seq_number(Path(x).name) or 0)

    return result


def find_matching_bp_sequence(
    column_seq_path: str,
    search_range: int = 2,
) -> Optional[str]:
    """
    Troba la SEQ BP que correspon a una SEQ COLUMN.

    Prioritat:
    1. Mateix número de lot
    2. Número adjacent (±1)
    3. Dins del rang de cerca

    Args:
        column_seq_path: Path de la seqüència COLUMN
        search_range: Rang màxim de cerca

    Returns:
        Path de la SEQ BP trobada o None
    """
    related = find_related_sequences(column_seq_path, search_range)

    if not related["bp_seqs"]:
        return None

    current_number = related["current_number"]
    if current_number is None:
        return related["bp_seqs"][0] if related["bp_seqs"] else None

    # Ordenar per proximitat al número actual
    bp_with_distance = []
    for bp_path in related["bp_seqs"]:
        bp_number = extract_seq_number(Path(bp_path).name)
        if bp_number is not None:
            distance = abs(bp_number - current_number)
            bp_with_distance.append((distance, bp_number, bp_path))

    if not bp_with_distance:
        return None

    # Prioritzar per: 1) distància, 2) número més proper (preferint anterior)
    bp_with_distance.sort(key=lambda x: (x[0], -x[1]))

    return bp_with_distance[0][2]


def load_bp_data_for_sample(
    bp_seq_path: str,
    sample_name: str,
) -> Optional[Dict]:
    """
    Carrega les dades BP d'una mostra específica.

    Args:
        bp_seq_path: Path de la seqüència BP
        sample_name: Nom de la mostra a cercar

    Returns:
        Dict amb dades BP o None si no es troba
    """
    from hpsec_import import load_manifest
    import json

    bp_path = Path(bp_seq_path)

    # Intent 1: Carregar des de analysis_result.json
    analysis_path = bp_path / "CHECK" / "data" / "analysis_result.json"
    if analysis_path.exists():
        try:
            with open(analysis_path, 'r', encoding='utf-8') as f:
                analysis = json.load(f)

            # Cercar la mostra
            samples_grouped = analysis.get("samples_grouped", {})
            if sample_name in samples_grouped:
                sample_data = samples_grouped[sample_name]
                return _extract_bp_summary(sample_data, str(bp_path))

            # Cercar amb nom normalitzat
            for name, data in samples_grouped.items():
                if _normalize_sample_name(name) == _normalize_sample_name(sample_name):
                    return _extract_bp_summary(data, str(bp_path))

        except Exception:
            pass

    # Intent 2: Carregar des del manifest
    manifest = load_manifest(str(bp_path))
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


def _normalize_sample_name(name: str) -> str:
    """Normalitza nom de mostra per comparació."""
    # Eliminar espais, guions baixos duplicats, convertir a majúscules
    normalized = name.strip().upper()
    normalized = re.sub(r'[\s_-]+', '_', normalized)
    return normalized


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
    """
    Consolida dades COLUMN i BP d'una mostra.

    Args:
        column_data: Dades COLUMN (obligatori)
        bp_data: Dades BP (opcional)

    Returns:
        Dict consolidat amb tota la informació
    """
    consolidated = {
        "column": column_data,
        "bp": bp_data,
        "has_bp": bp_data is not None,
    }

    # Afegir camps de resum
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
    """
    Troba totes les versions d'una mostra en diferents seqüències.

    Args:
        sample_name: Nom de la mostra
        data_folder: Carpeta arrel de dades
        seq_type: Filtrar per tipus ("COLUMN", "BP" o None per tots)

    Returns:
        Llista de dicts amb info de cada versió trobada
    """
    import json

    data_path = Path(data_folder)
    versions = []

    if not data_path.exists():
        return versions

    # Cercar en totes les carpetes SEQ
    for seq_dir in data_path.iterdir():
        if not seq_dir.is_dir():
            continue

        seq_name = seq_dir.name
        detected_type = detect_seq_type(seq_name)

        if seq_type and detected_type != seq_type:
            continue

        # Cercar analysis_result.json
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

    # Ordenar per número de seqüència
    versions.sort(key=lambda x: x.get("seq_number") or 0)

    return versions


# =============================================================================
# FUNCIONS D'EXPORTACIÓ CONSOLIDADA
# =============================================================================

def add_bp_sheet_to_excel(
    excel_path: str,
    bp_data: Dict,
) -> bool:
    """
    Afegeix un full BP a un Excel existent.

    Args:
        excel_path: Path del fitxer Excel
        bp_data: Dades BP a afegir

    Returns:
        True si s'ha afegit correctament
    """
    import pandas as pd
    from openpyxl import load_workbook

    if not bp_data:
        return False

    try:
        # Crear DataFrame amb dades BP
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

        # Afegir al Excel
        with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a',
                            if_sheet_exists='replace') as writer:
            df_bp.to_excel(writer, sheet_name="BP", index=False)

        return True

    except Exception as e:
        print(f"Error afegint full BP: {e}")
        return False


# =============================================================================
# EXPORTS
# =============================================================================
__all__ = [
    "extract_seq_number",
    "detect_seq_type",
    "find_related_sequences",
    "find_matching_bp_sequence",
    "load_bp_data_for_sample",
    "consolidate_sample_data",
    "find_all_sample_versions",
    "add_bp_sheet_to_excel",
]
