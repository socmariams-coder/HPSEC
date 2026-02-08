"""
hpsec_samples_db.py - Base de Dades de Mostres HPSEC
=====================================================

Gestiona un índex global de totes les mostres analitzades:
- Registra aparicions de mostres a través de totes les SEQs
- Permet cercar per nom o àlies (des d'Excel extern)
- Calcula tendències temporals
- Compara múltiples mostres

Usat per:
- SamplesDBPanel (nou tab "Mostres")
- Hook a hpsec_analyze.py per registre automàtic

"""

__version__ = "1.0.0"
__version_date__ = "2026-02-06"

import os
import re
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

# =============================================================================
# CONFIGURACIÓ I PATHS
# =============================================================================

def get_registry_path() -> Optional[Path]:
    """Retorna el path de la carpeta REGISTRY."""
    from hpsec_config import get_config
    cfg = get_config()
    registry = cfg.get("paths", "registry_folder")
    if registry and os.path.exists(registry):
        return Path(registry)
    return None


def get_samples_index_path() -> Optional[str]:
    """Retorna el path del fitxer Samples_Index.json."""
    registry = get_registry_path()
    if registry:
        return str(registry / "Samples_Index.json")
    return None


def get_data_folder() -> Optional[str]:
    """Retorna el path de la carpeta de dades."""
    from hpsec_config import get_config
    cfg = get_config()
    return cfg.get("paths", "data_folder")


# =============================================================================
# FUNCIONS CRUD - ÍNDEX DE MOSTRES
# =============================================================================

def load_samples_index() -> Dict:
    """
    Carrega l'índex de mostres global.

    Returns:
        Dict amb estructura de l'índex, o estructura buida si no existeix.
    """
    index_path = get_samples_index_path()
    if not index_path or not os.path.exists(index_path):
        return _create_empty_index()

    try:
        with open(index_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error carregant índex de mostres: {e}")
        return _create_empty_index()


def save_samples_index(index: Dict) -> bool:
    """
    Guarda l'índex de mostres.

    Args:
        index: Diccionari amb l'estructura de l'índex

    Returns:
        True si s'ha guardat correctament
    """
    index_path = get_samples_index_path()
    if not index_path:
        return False

    try:
        # Actualitzar timestamp
        index["updated"] = datetime.now().isoformat()

        # Actualitzar estadístiques
        index["statistics"] = _calculate_statistics(index)

        # Assegurar que existeix el directori
        os.makedirs(os.path.dirname(index_path), exist_ok=True)

        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error guardant índex de mostres: {e}")
        return False


def _create_empty_index() -> Dict:
    """Crea un índex buit."""
    return {
        "version": "1.0",
        "updated": datetime.now().isoformat(),
        "samples": {},
        "statistics": {
            "total_samples": 0,
            "total_appearances": 0
        }
    }


def _calculate_statistics(index: Dict) -> Dict:
    """Calcula estadístiques de l'índex."""
    samples = index.get("samples", {})

    total_samples = len(samples)
    total_appearances = sum(
        len(s.get("appearances", []))
        for s in samples.values()
    )

    # Comptador per tipus
    by_type = {}
    for s in samples.values():
        stype = s.get("sample_type", "UNKNOWN")
        by_type[stype] = by_type.get(stype, 0) + 1

    return {
        "total_samples": total_samples,
        "total_appearances": total_appearances,
        "by_type": by_type
    }


# =============================================================================
# REGISTRE DE MOSTRES
# =============================================================================

def register_sample_appearance(
    sample_name: str,
    seq_name: str,
    seq_path: str,
    sample_data: Dict,
    seq_type: str = "COLUMN"
) -> bool:
    """
    Registra una aparició d'una mostra a l'índex.

    Args:
        sample_name: Nom de la mostra
        seq_name: Nom de la seqüència
        seq_path: Path de la seqüència
        sample_data: Dades de la mostra (de samples_grouped)
        seq_type: "COLUMN" o "BP"

    Returns:
        True si s'ha registrat correctament
    """
    index = load_samples_index()

    # Normalitzar nom per clau
    normalized_name = _normalize_sample_name(sample_name)

    # Crear entrada si no existeix
    if normalized_name not in index["samples"]:
        index["samples"][normalized_name] = {
            "sample_name": sample_name,
            "sample_type": _detect_sample_type(sample_name),
            "first_seen": datetime.now().strftime("%Y-%m-%d"),
            "last_seen": datetime.now().strftime("%Y-%m-%d"),
            "n_appearances": 0,
            "appearances": []
        }

    sample_entry = index["samples"][normalized_name]

    # Crear aparició
    appearance = _create_appearance(
        seq_name, seq_path, sample_data, seq_type
    )

    # Comprovar si ja existeix aquesta SEQ
    existing_idx = None
    for i, app in enumerate(sample_entry["appearances"]):
        if app.get("seq_name") == seq_name and app.get("seq_type") == seq_type:
            existing_idx = i
            break

    if existing_idx is not None:
        # Actualitzar existent
        sample_entry["appearances"][existing_idx] = appearance
    else:
        # Afegir nova
        sample_entry["appearances"].append(appearance)

    # Actualitzar metadades
    sample_entry["n_appearances"] = len(sample_entry["appearances"])
    sample_entry["last_seen"] = datetime.now().strftime("%Y-%m-%d")

    # Guardar
    return save_samples_index(index)


def register_samples_from_analysis(analysis_result: Dict) -> int:
    """
    Registra totes les mostres d'un resultat d'anàlisi.

    Args:
        analysis_result: Resultat complet de analyze_sequence()

    Returns:
        Nombre de mostres registrades
    """
    samples_grouped = analysis_result.get("samples_grouped", {})
    seq_path = analysis_result.get("seq_path", "")
    seq_name = analysis_result.get("seq_name", os.path.basename(seq_path))
    seq_type = analysis_result.get("seq_type", "COLUMN")

    count = 0
    for sample_name, sample_data in samples_grouped.items():
        # Saltar controls i blancs si es vol (opcional)
        sample_type = _detect_sample_type(sample_name)
        if sample_type in ["BLANC"]:
            continue

        if register_sample_appearance(
            sample_name, seq_name, seq_path, sample_data, seq_type
        ):
            count += 1

    return count


def _create_appearance(
    seq_name: str,
    seq_path: str,
    sample_data: Dict,
    seq_type: str
) -> Dict:
    """Crea el registre d'una aparició."""
    # Extreure dades quantificació
    quantification = sample_data.get("quantification", {})
    selected = sample_data.get("selected", {})
    replicas = sample_data.get("replicas", {})

    # Àrees per fracció (extreure de DOC)
    fractions = {}
    selected_rep = selected.get("doc", "1")
    if selected_rep in replicas:
        rep_data = replicas[selected_rep]
        areas = rep_data.get("areas", {})
        if isinstance(areas, dict):
            # L'estructura és: {"DOC": {"BB": x, "BB_pct": y, ...}, "A220": {...}, ...}
            # Agafem les fraccions de DOC amb els valors _pct
            doc_areas = areas.get("DOC", {})
            if isinstance(doc_areas, dict):
                for key, value in doc_areas.items():
                    if key.endswith("_pct") and isinstance(value, (int, float)):
                        frac_name = key.replace("_pct", "")
                        fractions[frac_name] = round(value, 1)

    # Obtenir area_total de DOC
    area_total = None
    if selected_rep in replicas:
        rep_data = replicas[selected_rep]
        areas = rep_data.get("areas", {})
        doc_areas = areas.get("DOC", {})
        if isinstance(doc_areas, dict):
            area_total = doc_areas.get("total")

    # Cercar BP aparellat
    bp_paired = sample_data.get("bp_data") is not None
    bp_seq = sample_data.get("bp_data", {}).get("seq_name", "") if bp_paired else ""

    return {
        "seq_name": seq_name,
        "seq_path": seq_path,
        "seq_type": seq_type,
        "date_processed": datetime.now().isoformat(),
        "n_replicas": len(replicas),
        "selected_replica": selected_rep,
        "area_total": area_total,
        "concentration_ppm": quantification.get("concentration_ppm"),
        "fractions": fractions,
        "has_bp_paired": bp_paired,
        "bp_seq_name": bp_seq,
        "warnings": sample_data.get("warnings", []),
    }


# =============================================================================
# CERCA I FILTRES
# =============================================================================

def search_samples(
    query: str = None,
    sample_type: str = None,
    date_from: str = None,
    date_to: str = None,
    seq_type: str = None,
    has_bp: bool = None,
    min_appearances: int = None
) -> List[Dict]:
    """
    Cerca mostres amb filtres múltiples.

    Args:
        query: Text a cercar en nom o àlies
        sample_type: SAMPLE, CONTROL, KHP, BLANC
        date_from: Data mínima (YYYY-MM-DD)
        date_to: Data màxima (YYYY-MM-DD)
        seq_type: COLUMN o BP
        has_bp: True per mostres amb BP aparellat
        min_appearances: Mínim d'aparicions

    Returns:
        Llista de mostres que compleixen els filtres
    """
    index = load_samples_index()
    aliases = load_sample_aliases()

    results = []

    for key, sample in index.get("samples", {}).items():
        # Filtre per tipus
        if sample_type and sample.get("sample_type") != sample_type:
            continue

        # Filtre per dates
        if date_from:
            if sample.get("last_seen", "") < date_from:
                continue
        if date_to:
            if sample.get("first_seen", "") > date_to:
                continue

        # Filtre per aparicions
        if min_appearances:
            if sample.get("n_appearances", 0) < min_appearances:
                continue

        # Filtre per BP aparellat
        if has_bp is not None:
            has_any_bp = any(
                app.get("has_bp_paired")
                for app in sample.get("appearances", [])
            )
            if has_bp != has_any_bp:
                continue

        # Filtre per tipus de SEQ
        if seq_type:
            has_type = any(
                app.get("seq_type") == seq_type
                for app in sample.get("appearances", [])
            )
            if not has_type:
                continue

        # Filtre per query (nom o àlies)
        if query:
            query_upper = query.upper()
            name_match = query_upper in sample.get("sample_name", "").upper()
            alias = aliases.get(key, "")
            alias_match = query_upper in alias.upper() if alias else False
            if not (name_match or alias_match):
                continue

        # Afegir àlies al resultat
        result = sample.copy()
        result["alias"] = aliases.get(key, "")
        result["key"] = key
        results.append(result)

    # Ordenar per últim vist
    results.sort(key=lambda x: x.get("last_seen", ""), reverse=True)

    return results


def get_sample_history(sample_name: str) -> List[Dict]:
    """
    Retorna totes les aparicions d'una mostra ordenades per data.

    Args:
        sample_name: Nom de la mostra (original o normalitzat)

    Returns:
        Llista d'aparicions ordenades cronològicament
    """
    index = load_samples_index()

    # Cercar per nom normalitzat
    normalized = _normalize_sample_name(sample_name)
    sample = index.get("samples", {}).get(normalized)

    if not sample:
        # Cercar per nom original
        for key, s in index.get("samples", {}).items():
            if s.get("sample_name", "").upper() == sample_name.upper():
                sample = s
                break

    if not sample:
        return []

    appearances = sample.get("appearances", [])

    # Ordenar per data
    sorted_apps = sorted(
        appearances,
        key=lambda x: x.get("date_processed", "")
    )

    return sorted_apps


def get_sample_trends(sample_name: str) -> Dict:
    """
    Calcula tendències temporals d'una mostra.

    Args:
        sample_name: Nom de la mostra

    Returns:
        Dict amb sèries temporals per cada mètrica
    """
    history = get_sample_history(sample_name)

    if not history:
        return {}

    # Preparar sèries
    dates = []
    area_total = []
    concentration = []
    fractions_series = {
        "BioP": [], "HS": [], "BB": [], "SB": [], "LMW": []
    }

    for app in history:
        date = app.get("date_processed", "")[:10]  # YYYY-MM-DD
        dates.append(date)

        area_total.append(app.get("area_total"))
        concentration.append(app.get("concentration_ppm"))

        fracs = app.get("fractions", {})
        for key in fractions_series:
            fractions_series[key].append(fracs.get(key))

    return {
        "dates": dates,
        "area_total": area_total,
        "concentration_ppm": concentration,
        "fractions": fractions_series
    }


def compare_samples(sample_names: List[str]) -> Dict:
    """
    Compara múltiples mostres.

    Args:
        sample_names: Llista de noms de mostres

    Returns:
        Dict amb estadístiques comparatives
    """
    comparison = {
        "samples": [],
        "metrics": {}
    }

    for name in sample_names:
        history = get_sample_history(name)
        if not history:
            continue

        # Última aparició
        last = history[-1] if history else {}

        # Estadístiques
        areas = [a.get("area_total") for a in history if a.get("area_total")]
        concs = [a.get("concentration_ppm") for a in history if a.get("concentration_ppm")]

        sample_stats = {
            "name": name,
            "n_appearances": len(history),
            "last_area": last.get("area_total"),
            "last_conc": last.get("concentration_ppm"),
            "avg_area": sum(areas) / len(areas) if areas else None,
            "avg_conc": sum(concs) / len(concs) if concs else None,
            "last_fractions": last.get("fractions", {})
        }

        comparison["samples"].append(sample_stats)

    return comparison


# =============================================================================
# RECONSTRUCCIÓ D'ÍNDEX
# =============================================================================

def rebuild_samples_index(
    data_folder: str = None,
    progress_callback=None
) -> Dict:
    """
    Reconstrueix l'índex escanejant tots els analysis_result.json.

    Args:
        data_folder: Carpeta arrel (o usa config per defecte)
        progress_callback: Funció(current, total, message) per progrés

    Returns:
        Nou índex complet
    """
    if data_folder is None:
        data_folder = get_data_folder()

    if not data_folder or not os.path.exists(data_folder):
        return _create_empty_index()

    # Crear índex nou
    index = _create_empty_index()

    data_path = Path(data_folder)

    # Llistar totes les carpetes SEQ
    seq_dirs = [d for d in data_path.iterdir() if d.is_dir()]
    total = len(seq_dirs)

    for i, seq_dir in enumerate(seq_dirs):
        if progress_callback:
            progress_callback(i, total, f"Processant {seq_dir.name}...")

        # Cercar analysis_result.json
        analysis_path = seq_dir / "CHECK" / "data" / "analysis_result.json"
        if not analysis_path.exists():
            continue

        try:
            with open(analysis_path, 'r', encoding='utf-8') as f:
                analysis = json.load(f)

            # Determinar tipus
            from hpsec_consolidate import detect_seq_type
            seq_type = detect_seq_type(seq_dir.name)

            # Afegir info necessària
            analysis["seq_path"] = str(seq_dir)
            analysis["seq_name"] = seq_dir.name
            analysis["seq_type"] = seq_type

            # Registrar mostres (directament a l'índex, no guardar encara)
            samples_grouped = analysis.get("samples_grouped", {})
            for sample_name, sample_data in samples_grouped.items():
                _add_to_index(index, sample_name, seq_dir.name,
                             str(seq_dir), sample_data, seq_type)

        except Exception as e:
            print(f"Error processant {seq_dir.name}: {e}")
            continue

    if progress_callback:
        progress_callback(total, total, "Guardant índex...")

    # Guardar
    save_samples_index(index)

    return index


def _add_to_index(
    index: Dict,
    sample_name: str,
    seq_name: str,
    seq_path: str,
    sample_data: Dict,
    seq_type: str
):
    """Afegeix una mostra a l'índex (sense guardar)."""
    normalized_name = _normalize_sample_name(sample_name)
    sample_type = _detect_sample_type(sample_name)

    # Saltar blancs
    if sample_type == "BLANC":
        return

    # Crear entrada si no existeix
    if normalized_name not in index["samples"]:
        index["samples"][normalized_name] = {
            "sample_name": sample_name,
            "sample_type": sample_type,
            "first_seen": datetime.now().strftime("%Y-%m-%d"),
            "last_seen": datetime.now().strftime("%Y-%m-%d"),
            "n_appearances": 0,
            "appearances": []
        }

    sample_entry = index["samples"][normalized_name]

    # Crear aparició
    appearance = _create_appearance(seq_name, seq_path, sample_data, seq_type)

    # Comprovar duplicats
    existing = False
    for app in sample_entry["appearances"]:
        if app.get("seq_name") == seq_name and app.get("seq_type") == seq_type:
            existing = True
            break

    if not existing:
        sample_entry["appearances"].append(appearance)
        sample_entry["n_appearances"] = len(sample_entry["appearances"])


# =============================================================================
# SISTEMA D'ÀLIES
# =============================================================================

_aliases_cache = None
_aliases_cache_time = None


def load_sample_aliases() -> Dict[str, str]:
    """
    Carrega el mapeig ID -> Àlies des del fitxer Excel configurat.

    Returns:
        Dict amb clau=nom_normalitzat, valor=àlies
    """
    global _aliases_cache, _aliases_cache_time

    # Usar cache si és recent (5 segons)
    if _aliases_cache is not None and _aliases_cache_time:
        if (datetime.now() - _aliases_cache_time).seconds < 5:
            return _aliases_cache

    from hpsec_config import get_config
    cfg = get_config()

    # Obtenir path i columnes
    excel_path = cfg.get("paths", "samples_alias_excel", "")
    samples_db_cfg = cfg.get_section("samples_db") or {}
    id_col = samples_db_cfg.get("alias_column_id", "")
    alias_col = samples_db_cfg.get("alias_column_alias", "")

    if not excel_path or not os.path.exists(excel_path):
        _aliases_cache = {}
        _aliases_cache_time = datetime.now()
        return {}

    if not id_col or not alias_col:
        _aliases_cache = {}
        _aliases_cache_time = datetime.now()
        return {}

    try:
        import pandas as pd
        df = pd.read_excel(excel_path)

        if id_col not in df.columns or alias_col not in df.columns:
            _aliases_cache = {}
            _aliases_cache_time = datetime.now()
            return {}

        # Crear mapeig
        aliases = {}
        for _, row in df.iterrows():
            sample_id = str(row[id_col]).strip()
            alias = str(row[alias_col]).strip()
            if sample_id and alias and alias != "nan":
                normalized = _normalize_sample_name(sample_id)
                aliases[normalized] = alias

        _aliases_cache = aliases
        _aliases_cache_time = datetime.now()
        return aliases

    except Exception as e:
        print(f"Error carregant àlies: {e}")
        _aliases_cache = {}
        _aliases_cache_time = datetime.now()
        return {}


def get_sample_alias(sample_name: str) -> Optional[str]:
    """
    Retorna l'àlies d'una mostra.

    Args:
        sample_name: Nom de la mostra

    Returns:
        Àlies o None si no en té
    """
    aliases = load_sample_aliases()
    normalized = _normalize_sample_name(sample_name)
    return aliases.get(normalized)


def get_excel_columns(excel_path: str) -> List[str]:
    """
    Llegeix les columnes disponibles d'un fitxer Excel.

    Args:
        excel_path: Path del fitxer Excel

    Returns:
        Llista de noms de columnes
    """
    if not excel_path or not os.path.exists(excel_path):
        return []

    try:
        import pandas as pd
        df = pd.read_excel(excel_path, nrows=0)  # Només capçaleres
        return list(df.columns)
    except Exception as e:
        print(f"Error llegint columnes Excel: {e}")
        return []


def get_excel_preview(
    excel_path: str,
    id_col: str,
    alias_col: str,
    n_rows: int = 5
) -> List[Dict]:
    """
    Retorna una vista prèvia del mapeig.

    Args:
        excel_path: Path del fitxer Excel
        id_col: Nom columna ID
        alias_col: Nom columna àlies
        n_rows: Nombre de files a mostrar

    Returns:
        Llista de dicts amb id i alias
    """
    if not excel_path or not os.path.exists(excel_path):
        return []

    try:
        import pandas as pd
        df = pd.read_excel(excel_path, nrows=n_rows)

        if id_col not in df.columns or alias_col not in df.columns:
            return []

        preview = []
        for _, row in df.iterrows():
            preview.append({
                "id": str(row[id_col]),
                "alias": str(row[alias_col])
            })
        return preview

    except Exception as e:
        print(f"Error llegint preview: {e}")
        return []


def configure_alias_mapping(
    excel_path: str,
    id_col: str,
    alias_col: str
) -> bool:
    """
    Guarda la configuració de mapeig d'àlies.

    Args:
        excel_path: Path del fitxer Excel
        id_col: Nom columna ID
        alias_col: Nom columna àlies

    Returns:
        True si s'ha guardat correctament
    """
    global _aliases_cache, _aliases_cache_time

    from hpsec_config import get_config, save_config
    cfg = get_config()

    # Actualitzar paths
    cfg.set("paths", "samples_alias_excel", excel_path)

    # Actualitzar samples_db
    samples_db = cfg.get_section("samples_db") or {}
    samples_db["alias_column_id"] = id_col
    samples_db["alias_column_alias"] = alias_col
    cfg.set_section("samples_db", samples_db)

    # Invalidar cache
    _aliases_cache = None
    _aliases_cache_time = None

    return save_config(cfg)


# =============================================================================
# HELPERS
# =============================================================================

def _normalize_sample_name(name: str) -> str:
    """Normalitza nom de mostra per comparació."""
    if not name:
        return ""
    normalized = name.strip().upper()
    normalized = re.sub(r'[\s_-]+', '_', normalized)
    return normalized


def _detect_sample_type(sample_name: str) -> str:
    """Detecta el tipus de mostra pel nom."""
    name_upper = sample_name.upper()

    # Patrons definits a config
    from hpsec_config import get_config
    cfg = get_config()
    sample_types = cfg.get_section("sample_types") or {}

    for type_key, type_cfg in sample_types.items():
        patterns = type_cfg.get("patterns", [])
        for pattern in patterns:
            if pattern.upper() in name_upper:
                return type_key

    # Default
    return "MOSTRA"


# =============================================================================
# EXPORTS
# =============================================================================
__all__ = [
    "load_samples_index",
    "save_samples_index",
    "rebuild_samples_index",
    "register_sample_appearance",
    "register_samples_from_analysis",
    "get_sample_history",
    "search_samples",
    "get_sample_trends",
    "compare_samples",
    "load_sample_aliases",
    "get_sample_alias",
    "get_excel_columns",
    "get_excel_preview",
    "configure_alias_mapping",
]
