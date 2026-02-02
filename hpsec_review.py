"""
hpsec_review.py - Mòdul de revisió de rèpliques HPSEC (Fase 4)
==============================================================

FASE 4 del pipeline de 5 fases:
- Comparar rèpliques d'una mostra
- Calcular estadístiques de qualitat
- Recomanar millor rèplica o promig
- Identificar mostres que requereixen atenció

REQUEREIX:
- Fase 3: process_sequence() → dades processades

NO fa:
- Lectura de fitxers (Fase 1: IMPORTAR)
- Validació KHP (Fase 2: CALIBRAR)
- Processament de senyals (Fase 3: PROCESSAR)
- Escriptura Excel finals (Fase 5: EXPORTAR)

Usat per HPSEC_Suite.py
"""

__version__ = "1.1.0"
__version_date__ = "2026-01-29"
# v1.1.0 - Selecció independent DOC/DAD

import numpy as np
from collections import defaultdict

# Import funcions de comparació de rèpliques (Single Source of Truth)
from hpsec_replica import (
    # Avaluació individual
    evaluate_replica,
    evaluate_dad,
    evaluate_dad_multi,
    # Comparació
    compare_replicas,
    compare_replicas_full,
    compare_doc_dad,
    # Selecció
    select_best_replica,
    evaluate_and_select,
    # Constants
    NOISE_THRESHOLD,
    PEARSON_THRESHOLD,
    PEARSON_WARNING,
    SNR_RATIO_THRESHOLD,
    AREA_DIFF_WARNING,
    AREA_DIFF_CRITICAL,
    HUMIC_ZONE,
)


# =============================================================================
# CONFIGURACIÓ
# =============================================================================
DEFAULT_REVIEW_CONFIG = {
    # Llindars de qualitat
    "pearson_threshold": PEARSON_THRESHOLD,
    "pearson_warning": PEARSON_WARNING,
    "area_diff_warning": AREA_DIFF_WARNING,
    "area_diff_critical": AREA_DIFF_CRITICAL,
    "snr_ratio_threshold": SNR_RATIO_THRESHOLD,
    "noise_threshold": NOISE_THRESHOLD,
    # Zona húmics (Column)
    "humic_zone": HUMIC_ZONE,
}

# Classificació de resultats
REVIEW_STATUS = {
    "OK": "Rèpliques consistents, selecció automàtica",
    "CHECK": "Revisar manualment - diferències significatives",
    "WARNING": "Anomalies detectades però seleccionable",
    "DISCARD": "Ambdues rèpliques invàlides",
}


# =============================================================================
# RECOLLIDA D'ESTADÍSTIQUES
# =============================================================================
def collect_sample_stats(sample_name, replica, timeout_info, snr_info, peak_info, data_mode):
    """
    Recull estadístiques d'una mostra processada.

    Args:
        sample_name: Nom de la mostra
        replica: Número de rèplica
        timeout_info: Dict amb info de timeouts
        snr_info: Dict amb SNR, LOD, LOQ
        peak_info: Dict amb info del pic principal
        data_mode: "UIB", "DIRECT" o "DUAL"

    Returns:
        dict amb estadístiques de la mostra
    """
    stats = {
        "name": f"{sample_name}_R{replica}",
        "sample": sample_name,
        "replica": replica,
        "data_mode": data_mode,
        "peak_valid": peak_info.get("valid", False) if peak_info else False,
    }

    # Timeout info
    if timeout_info:
        stats["timeout_detected"] = timeout_info.get("n_timeouts", 0) > 0
        stats["timeout_count"] = timeout_info.get("n_timeouts", 0)
        stats["timeout_severity"] = timeout_info.get("severity", "OK")
        zone_summary = timeout_info.get("zone_summary", {})
        if zone_summary:
            stats["timeout_zones"] = [z for z, n in zone_summary.items() if n > 0]
    else:
        stats["timeout_detected"] = False
        stats["timeout_count"] = 0
        stats["timeout_severity"] = "OK"
        stats["timeout_zones"] = []

    # SNR info
    if snr_info:
        stats["snr_direct"] = snr_info.get("snr_direct")
        stats["snr_uib"] = snr_info.get("snr_uib")
        stats["baseline_noise_direct"] = snr_info.get("baseline_noise_direct")
        stats["baseline_noise_uib"] = snr_info.get("baseline_noise_uib")
        stats["lod_direct"] = snr_info.get("lod_direct")
        stats["lod_uib"] = snr_info.get("lod_uib")
        stats["baseline_window_direct"] = snr_info.get("baseline_window_direct")
        stats["baseline_window_uib"] = snr_info.get("baseline_window_uib")

    # Peak info
    if peak_info and peak_info.get("valid"):
        stats["peak_area"] = peak_info.get("area", 0.0)
        stats["peak_height"] = peak_info.get("height", 0.0)
        stats["peak_tmax"] = peak_info.get("t_max", 0.0)

    return stats


# =============================================================================
# REVISIÓ D'UNA MOSTRA
# =============================================================================
def review_sample(replicas, method="COLUMN", config=None):
    """
    Revisa les rèpliques d'una mostra i recomana la millor.

    IMPORTANT: Selecció INDEPENDENT DOC i DAD.
    Pot recomanar R1 per DOC però R2 per DAD si la qualitat és diferent.

    Args:
        replicas: Llista de dicts amb dades processades de cada rèplica
                  Cada dict ha de tenir: t_doc, y_doc_net, df_dad, peak_info, snr_info, anomalies
        method: "COLUMN" o "BP"
        config: Configuració

    Returns:
        dict amb:
            - status: "OK", "CHECK", "WARNING", "DISCARD"
            - selected_doc: "R1", "R2", "AVERAGE", None (selecció per DOC)
            - selected_dad: "R1", "R2", "AVERAGE", None (selecció per DAD)
            - selected: "R1", "R2", "MIXED", None (resum - MIXED si DOC≠DAD)
            - comparison: dict amb mètriques de comparació
            - evaluations: llista d'avaluacions DOC per rèplica
            - dad_evaluations: llista d'avaluacions DAD per rèplica
            - recommendation: descripció de la recomanació
            - recommendation_doc: descripció selecció DOC
            - recommendation_dad: descripció selecció DAD
            - warnings: llista d'avisos
            - stats: estadístiques detallades
    """
    config = config or DEFAULT_REVIEW_CONFIG

    result = {
        "status": "DISCARD",
        "selected": None,
        "selected_doc": None,
        "selected_dad": None,
        "comparison": {},
        "evaluations": [],
        "dad_evaluations": [],
        "recommendation": "",
        "recommendation_doc": "",
        "recommendation_dad": "",
        "warnings": [],
        "stats": {},
    }

    if not replicas or len(replicas) == 0:
        result["recommendation"] = "No hi ha rèpliques disponibles"
        return result

    # Cas 1: Una sola rèplica
    if len(replicas) == 1:
        rep = replicas[0]
        t = rep.get("t_doc")
        y = rep.get("y_doc_net")

        if t is None or y is None or len(t) < 10:
            result["recommendation"] = "Rèplica única sense dades vàlides"
            return result

        # Avaluar DOC
        eval_result = evaluate_replica(t, y, method=method)
        result["evaluations"].append(eval_result)

        # Avaluar DAD si disponible
        df_dad = rep.get("df_dad")
        dad_eval = None
        if df_dad is not None and hasattr(df_dad, 'columns'):
            t_dad = df_dad.index.values if hasattr(df_dad.index, 'values') else np.arange(len(df_dad))
            dad_eval = evaluate_dad_multi(t_dad, df_dad)
            result["dad_evaluations"].append(dad_eval)
        else:
            result["dad_evaluations"].append(None)

        # Verificar anomalies
        anomalies = rep.get("anomalies", [])
        has_anomaly = eval_result.get("batman", False) or eval_result.get("timeout", False) or eval_result.get("irr", False)
        if has_anomaly or anomalies:
            result["status"] = "WARNING"
            result["warnings"].append("Rèplica única amb anomalies")
        else:
            result["status"] = "OK"

        result["selected"] = "R1"
        result["selected_doc"] = "R1"
        result["selected_dad"] = "R1"
        result["recommendation"] = "Rèplica única seleccionada"
        result["recommendation_doc"] = "Rèplica única"
        result["recommendation_dad"] = "Rèplica única"

        # Stats
        result["stats"] = {
            "n_replicas": 1,
            "snr": eval_result.get("snr"),
            "has_anomaly": has_anomaly,
        }

        return result

    # Cas 2: Dues o més rèpliques - comparar
    rep1 = replicas[0]
    rep2 = replicas[1] if len(replicas) > 1 else None

    t1 = rep1.get("t_doc")
    y1 = rep1.get("y_doc_net")
    t2 = rep2.get("t_doc") if rep2 else None
    y2 = rep2.get("y_doc_net") if rep2 else None

    # Verificar dades DOC
    valid1 = t1 is not None and y1 is not None and len(t1) > 10
    valid2 = t2 is not None and y2 is not None and len(t2) > 10

    if not valid1 and not valid2:
        result["recommendation"] = "Cap rèplica amb dades DOC vàlides"
        return result

    if not valid2:
        # Només R1 vàlida
        eval1 = evaluate_replica(t1, y1, method=method)
        result["evaluations"] = [eval1, None]

        # DAD R1
        df_dad1 = rep1.get("df_dad")
        if df_dad1 is not None and hasattr(df_dad1, 'columns'):
            t_dad1 = df_dad1.index.values if hasattr(df_dad1.index, 'values') else np.arange(len(df_dad1))
            dad_eval1 = evaluate_dad_multi(t_dad1, df_dad1)
            result["dad_evaluations"] = [dad_eval1, None]
        else:
            result["dad_evaluations"] = [None, None]

        result["selected"] = "R1"
        result["selected_doc"] = "R1"
        result["selected_dad"] = "R1"
        has_anomaly = eval1.get("batman", False) or eval1.get("timeout", False) or eval1.get("irr", False)
        result["status"] = "WARNING" if has_anomaly else "OK"
        result["recommendation"] = "R1 seleccionada (R2 invàlida)"
        result["recommendation_doc"] = "R1 (R2 invàlida)"
        result["recommendation_dad"] = "R1 (R2 invàlida)"
        return result

    if not valid1:
        # Només R2 vàlida
        eval2 = evaluate_replica(t2, y2, method=method)
        result["evaluations"] = [None, eval2]

        # DAD R2
        df_dad2 = rep2.get("df_dad")
        if df_dad2 is not None and hasattr(df_dad2, 'columns'):
            t_dad2 = df_dad2.index.values if hasattr(df_dad2.index, 'values') else np.arange(len(df_dad2))
            dad_eval2 = evaluate_dad_multi(t_dad2, df_dad2)
            result["dad_evaluations"] = [None, dad_eval2]
        else:
            result["dad_evaluations"] = [None, None]

        result["selected"] = "R2"
        result["selected_doc"] = "R2"
        result["selected_dad"] = "R2"
        has_anomaly = eval2.get("batman", False) or eval2.get("timeout", False) or eval2.get("irr", False)
        result["status"] = "WARNING" if has_anomaly else "OK"
        result["recommendation"] = "R2 seleccionada (R1 invàlida)"
        result["recommendation_doc"] = "R2 (R1 invàlida)"
        result["recommendation_dad"] = "R2 (R1 invàlida)"
        return result

    # === AMBDUES VÀLIDES - COMPARACIÓ INDEPENDENT DOC/DAD ===
    t1 = np.asarray(t1)
    y1 = np.asarray(y1)
    t2 = np.asarray(t2)
    y2 = np.asarray(y2)

    # Avaluar DOC cada rèplica
    eval1 = evaluate_replica(t1, y1, method=method)
    eval2 = evaluate_replica(t2, y2, method=method)
    result["evaluations"] = [eval1, eval2]

    # Avaluar DAD cada rèplica
    df_dad1 = rep1.get("df_dad")
    df_dad2 = rep2.get("df_dad")
    dad_eval1 = None
    dad_eval2 = None

    if df_dad1 is not None and hasattr(df_dad1, 'columns'):
        t_dad1 = df_dad1.index.values if hasattr(df_dad1.index, 'values') else np.arange(len(df_dad1))
        dad_eval1 = evaluate_dad_multi(t_dad1, df_dad1)

    if df_dad2 is not None and hasattr(df_dad2, 'columns'):
        t_dad2 = df_dad2.index.values if hasattr(df_dad2.index, 'values') else np.arange(len(df_dad2))
        dad_eval2 = evaluate_dad_multi(t_dad2, df_dad2)

    result["dad_evaluations"] = [dad_eval1, dad_eval2]

    # Comparar rèpliques DOC
    comparison = compare_replicas(t1, y1, t2, y2)
    result["comparison"] = comparison

    # === SELECCIÓ DOC ===
    selection_doc = select_best_replica(eval1, eval2, method=method, comparison=comparison)
    result["selected_doc"] = selection_doc.get("best")
    result["recommendation_doc"] = selection_doc.get("reason", "")

    # === SELECCIÓ DAD (independent) ===
    selection_dad = _select_best_dad(dad_eval1, dad_eval2)
    result["selected_dad"] = selection_dad.get("best")
    result["recommendation_dad"] = selection_dad.get("reason", "")

    # === DETERMINAR SELECCIÓ GLOBAL ===
    if result["selected_doc"] == result["selected_dad"]:
        result["selected"] = result["selected_doc"]
        result["recommendation"] = f"DOC i DAD: {result['selected_doc']} - {result['recommendation_doc']}"
    elif result["selected_doc"] is None:
        result["selected"] = result["selected_dad"]
        result["recommendation"] = f"DAD: {result['selected_dad']} (DOC descartable)"
    elif result["selected_dad"] is None:
        result["selected"] = result["selected_doc"]
        result["recommendation"] = f"DOC: {result['selected_doc']} (DAD descartable)"
    else:
        result["selected"] = "MIXED"
        result["recommendation"] = f"MIXED: DOC={result['selected_doc']}, DAD={result['selected_dad']}"
        result["warnings"].append(f"Seleccio diferent: DOC={result['selected_doc']}, DAD={result['selected_dad']}")

    # Afegir warnings de selecció DOC
    if selection_doc.get("warning"):
        result["warnings"].append(selection_doc["warning"])

    # Afegir warnings de selecció DAD
    if selection_dad.get("warning"):
        result["warnings"].append(selection_dad["warning"])

    # Determinar status
    pearson = comparison.get("pearson", 1.0)
    area_diff = comparison.get("area_diff_pct", 0.0)

    if result["selected"] is None or (result["selected_doc"] is None and result["selected_dad"] is None):
        result["status"] = "DISCARD"
    elif result["selected"] == "MIXED":
        result["status"] = "CHECK"  # MIXED sempre requereix revisió
    elif pearson < config["pearson_warning"] or area_diff > config["area_diff_critical"]:
        result["status"] = "CHECK"
    elif eval1.get("batman") or eval2.get("batman") or result["warnings"]:
        result["status"] = "WARNING"
    else:
        result["status"] = "OK"

    # Stats
    result["stats"] = {
        "n_replicas": 2,
        "pearson": pearson,
        "area_diff_pct": area_diff,
        "snr_r1": eval1.get("snr"),
        "snr_r2": eval2.get("snr"),
        "anomaly_r1": eval1.get("batman", False) or eval1.get("timeout", False),
        "anomaly_r2": eval2.get("batman", False) or eval2.get("timeout", False),
        "dad_quality_r1": dad_eval1.get("overall_quality") if dad_eval1 else None,
        "dad_quality_r2": dad_eval2.get("overall_quality") if dad_eval2 else None,
    }

    return result


def _select_best_dad(dad_eval1, dad_eval2):
    """
    Selecciona millor rèplica basant-se NOMÉS en criteris DAD.

    Criteris:
    1. Quality (OK > WARNING > POOR)
    2. Drift (menys deriva = millor)
    3. SNR (major SNR = millor)

    Args:
        dad_eval1, dad_eval2: Resultat de evaluate_dad_multi() o None

    Returns:
        dict amb: best, reason, warning
    """
    result = {
        "best": None,
        "reason": "",
        "warning": None,
    }

    # Cap DAD disponible
    if dad_eval1 is None and dad_eval2 is None:
        result["reason"] = "Cap DAD disponible"
        return result

    # Només una disponible
    if dad_eval1 is None:
        result["best"] = "R2"
        result["reason"] = "Única DAD disponible"
        return result

    if dad_eval2 is None:
        result["best"] = "R1"
        result["reason"] = "Única DAD disponible"
        return result

    # Extreure qualitat
    q1 = dad_eval1.get("overall_quality", "OK")
    q2 = dad_eval2.get("overall_quality", "OK")
    quality_order = {"OK": 0, "WARNING": 1, "POOR": 2}

    # Criteri 1: Quality
    if quality_order.get(q1, 0) < quality_order.get(q2, 0):
        result["best"] = "R1"
        result["reason"] = f"DAD quality R1={q1} > R2={q2}"
        return result

    if quality_order.get(q2, 0) < quality_order.get(q1, 0):
        result["best"] = "R2"
        result["reason"] = f"DAD quality R2={q2} > R1={q1}"
        return result

    # Criteri 2: Drift (A254)
    wl1 = dad_eval1.get("wavelengths", {})
    wl2 = dad_eval2.get("wavelengths", {})

    drift1 = abs(wl1.get("A254", {}).get("drift", 0))
    drift2 = abs(wl2.get("A254", {}).get("drift", 0))

    if drift1 < drift2 * 0.7:  # R1 té 30% menys deriva
        result["best"] = "R1"
        result["reason"] = f"DAD drift R1={drift1:.2f} < R2={drift2:.2f}"
        return result

    if drift2 < drift1 * 0.7:  # R2 té 30% menys deriva
        result["best"] = "R2"
        result["reason"] = f"DAD drift R2={drift2:.2f} < R1={drift1:.2f}"
        return result

    # Criteri 3: SNR (A254)
    snr1 = wl1.get("A254", {}).get("snr", 0)
    snr2 = wl2.get("A254", {}).get("snr", 0)

    if snr1 > snr2 * 1.2:  # R1 té 20% més SNR
        result["best"] = "R1"
        result["reason"] = f"DAD SNR R1={snr1:.1f} > R2={snr2:.1f}"
        return result

    if snr2 > snr1 * 1.2:  # R2 té 20% més SNR
        result["best"] = "R2"
        result["reason"] = f"DAD SNR R2={snr2:.1f} > R1={snr1:.1f}"
        return result

    # Tiebreaker: R1 per defecte
    result["best"] = "R1"
    result["reason"] = "DAD similar, R1 per defecte"

    return result


# =============================================================================
# REVISIÓ DE SEQÜÈNCIA
# =============================================================================
def review_sequence(processed_data, config=None, progress_callback=None):
    """
    FASE 4: Revisa tota la seqüència i genera recomanacions.

    Args:
        processed_data: Dict retornat per process_sequence() (Fase 3)
        config: Configuració
        progress_callback: Funció callback per reportar progrés

    Returns:
        dict amb:
            - success: True si s'ha revisat correctament
            - seq_name: Nom de la seqüència
            - method: "BP" o "COLUMN"
            - samples: Dict {nom_mostra: review_result}
            - summary: Resum estadístic
            - alerts: Llista de mostres que requereixen atenció
            - errors: Llista d'errors
    """
    config = config or DEFAULT_REVIEW_CONFIG

    result = {
        "success": False,
        "seq_name": processed_data.get("seq_name", "UNKNOWN"),
        "seq_path": processed_data.get("seq_path", ""),
        "method": processed_data.get("method", "UNKNOWN"),
        "samples": {},
        "summary": {},
        "alerts": [],
        "errors": [],
    }

    # Verificar dades d'entrada
    if not processed_data.get("success"):
        result["errors"].append("Processed data is invalid")
        return result

    method = processed_data.get("method", "COLUMN")
    all_samples = processed_data.get("samples", [])

    if len(all_samples) == 0:
        result["errors"].append("No samples to review")
        return result

    # Agrupar mostres per nom (per trobar rèpliques)
    samples_by_name = defaultdict(list)
    for sample in all_samples:
        name = sample.get("name", "UNKNOWN")
        samples_by_name[name].append(sample)

    # Revisar cada mostra
    total = len(samples_by_name)
    ok_count = 0
    check_count = 0
    warning_count = 0
    discard_count = 0

    for i, (name, replicas) in enumerate(samples_by_name.items()):
        if progress_callback:
            progress_callback(f"Reviewing {name}...", (i + 1) / total * 100)

        try:
            review = review_sample(replicas, method=method, config=config)
            result["samples"][name] = review

            # Comptar per status
            status = review.get("status", "DISCARD")
            if status == "OK":
                ok_count += 1
            elif status == "CHECK":
                check_count += 1
                result["alerts"].append({
                    "sample": name,
                    "status": status,
                    "reason": review.get("recommendation", ""),
                    "warnings": review.get("warnings", []),
                })
            elif status == "WARNING":
                warning_count += 1
                if review.get("warnings"):
                    result["alerts"].append({
                        "sample": name,
                        "status": status,
                        "reason": review.get("recommendation", ""),
                        "warnings": review.get("warnings", []),
                    })
            else:
                discard_count += 1
                result["alerts"].append({
                    "sample": name,
                    "status": status,
                    "reason": review.get("recommendation", ""),
                })

        except Exception as e:
            result["errors"].append(f"{name}: {str(e)}")

    # Generar resum
    result["summary"] = {
        "total_samples": total,
        "ok": ok_count,
        "check": check_count,
        "warning": warning_count,
        "discard": discard_count,
        "alerts_count": len(result["alerts"]),
    }

    result["success"] = len(result["errors"]) == 0

    if progress_callback:
        progress_callback("Review complete", 100)

    return result


# =============================================================================
# APLICAR SELECCIONS
# =============================================================================
def apply_selections(review_data, selections, processed_data):
    """
    Aplica les seleccions de l'usuari i genera dades finals.

    IMPORTANT: Suporta selecció INDEPENDENT DOC/DAD.

    Args:
        review_data: Dict retornat per review_sequence()
        selections: Dict amb seleccions. Pot ser:
            - Simple: {nom_mostra: "R1"/"R2"/"AVERAGE"/"DISCARD"}
            - Independent: {nom_mostra: {"doc": "R1", "dad": "R2"}}
            Si no especificat, usa la recomanació automàtica
        processed_data: Dict retornat per process_sequence()

    Returns:
        dict amb:
            - success: True si s'ha aplicat correctament
            - samples: Dict {nom_mostra: {selected_data}}
            - discarded: Llista de mostres descartades
            - mixed_samples: Llista de mostres amb selecció MIXED
            - summary: Resum d'aplicació
    """
    result = {
        "success": False,
        "samples": {},
        "discarded": [],
        "mixed_samples": [],
        "summary": {},
    }

    if not review_data.get("success"):
        return result

    # Agrupar mostres processades per nom
    samples_by_name = defaultdict(list)
    for sample in processed_data.get("samples", []):
        name = sample.get("name", "UNKNOWN")
        samples_by_name[name].append(sample)

    applied_count = 0
    discarded_count = 0
    averaged_count = 0
    mixed_count = 0

    for name, review in review_data.get("samples", {}).items():
        replicas = samples_by_name.get(name, [])
        if not replicas:
            result["discarded"].append(name)
            discarded_count += 1
            continue

        # Obtenir selecció (manual o automàtica)
        user_selection = selections.get(name)

        # Determinar selecció DOC i DAD
        if user_selection is None:
            # Usar recomanació automàtica
            sel_doc = review.get("selected_doc", review.get("selected"))
            sel_dad = review.get("selected_dad", review.get("selected"))
        elif isinstance(user_selection, dict):
            # Selecció independent
            sel_doc = user_selection.get("doc", review.get("selected_doc"))
            sel_dad = user_selection.get("dad", review.get("selected_dad"))
        else:
            # Selecció simple (mateixa per DOC i DAD)
            sel_doc = user_selection
            sel_dad = user_selection

        # Cas DISCARD
        if sel_doc == "DISCARD" and sel_dad == "DISCARD":
            result["discarded"].append(name)
            discarded_count += 1
            continue

        # Aplicar selecció
        sample_result = {
            "review": review,
            "source_doc": sel_doc,
            "source_dad": sel_dad,
            "is_mixed": sel_doc != sel_dad,
        }

        # Obtenir dades DOC
        if sel_doc == "AVERAGE" and len(replicas) >= 2:
            sample_result["data_doc"] = _average_replicas(replicas[0], replicas[1])
            sample_result["source_doc"] = "AVERAGE"
            averaged_count += 1
        elif sel_doc == "R2" and len(replicas) >= 2:
            sample_result["data_doc"] = replicas[1]
        elif sel_doc == "R1" and len(replicas) >= 1:
            sample_result["data_doc"] = replicas[0]
        elif len(replicas) >= 1:
            sample_result["data_doc"] = replicas[0]
            sample_result["source_doc"] = "R1"

        # Obtenir dades DAD
        if sel_dad == "AVERAGE" and len(replicas) >= 2:
            sample_result["data_dad"] = _average_dad(replicas[0], replicas[1])
            sample_result["source_dad"] = "AVERAGE"
        elif sel_dad == "R2" and len(replicas) >= 2:
            sample_result["data_dad"] = replicas[1]
        elif sel_dad == "R1" and len(replicas) >= 1:
            sample_result["data_dad"] = replicas[0]
        elif len(replicas) >= 1:
            sample_result["data_dad"] = replicas[0]
            sample_result["source_dad"] = "R1"

        # Compatibilitat: data és DOC per defecte
        sample_result["data"] = sample_result.get("data_doc", replicas[0] if replicas else None)
        sample_result["source"] = sel_doc if sel_doc == sel_dad else "MIXED"

        result["samples"][name] = sample_result

        if sel_doc != sel_dad:
            mixed_count += 1
            result["mixed_samples"].append({
                "name": name,
                "doc": sel_doc,
                "dad": sel_dad,
                "reason_doc": review.get("recommendation_doc", ""),
                "reason_dad": review.get("recommendation_dad", ""),
            })

        applied_count += 1

    result["summary"] = {
        "applied": applied_count,
        "discarded": discarded_count,
        "averaged": averaged_count,
        "mixed": mixed_count,
    }

    result["success"] = True
    return result


def _average_dad(rep1, rep2):
    """
    Calcula el promig de les dades DAD de dues rèpliques.

    Args:
        rep1, rep2: Dicts amb dades de rèplica (df_dad)

    Returns:
        Dict amb df_dad promitjat
    """
    import pandas as pd

    df1 = rep1.get("df_dad")
    df2 = rep2.get("df_dad")

    if df1 is None:
        return rep2

    if df2 is None:
        return rep1

    # Si són DataFrames, promitjar
    if hasattr(df1, 'columns') and hasattr(df2, 'columns'):
        try:
            # Interpolar df2 a índex de df1
            df2_interp = df2.reindex(df1.index).interpolate(method='linear')
            df_avg = (df1 + df2_interp) / 2.0

            result = rep1.copy()
            result["df_dad"] = df_avg
            result["replica"] = "AVG"
            result["is_averaged"] = True
            return result
        except Exception:
            return rep1

    return rep1


def _average_replicas(rep1, rep2):
    """
    Calcula el promig de dues rèpliques.

    Args:
        rep1, rep2: Dicts amb dades de rèplica (t_doc, y_doc_net, etc.)

    Returns:
        Dict amb dades promitjades
    """
    t1 = rep1.get("t_doc")
    y1 = rep1.get("y_doc_net")
    t2 = rep2.get("t_doc")
    y2 = rep2.get("y_doc_net")

    if t1 is None or y1 is None or t2 is None or y2 is None:
        return rep1  # Fallback

    t1 = np.asarray(t1)
    y1 = np.asarray(y1)
    t2 = np.asarray(t2)
    y2 = np.asarray(y2)

    # Interpolar y2 a l'escala de t1
    y2_interp = np.interp(t1, t2, y2)

    # Promig
    y_avg = (y1 + y2_interp) / 2.0

    # Crear resultat
    averaged = rep1.copy()
    averaged["t_doc"] = t1
    averaged["y_doc_net"] = y_avg
    averaged["replica"] = "AVG"
    averaged["is_averaged"] = True

    # Promitjar peak_info si existeix
    if rep1.get("peak_info") and rep2.get("peak_info"):
        pi1 = rep1["peak_info"]
        pi2 = rep2["peak_info"]
        if pi1.get("valid") and pi2.get("valid"):
            averaged["peak_info"] = {
                "valid": True,
                "area": (pi1.get("area", 0) + pi2.get("area", 0)) / 2,
                "height": (pi1.get("height", 0) + pi2.get("height", 0)) / 2,
                "t_max": (pi1.get("t_max", 0) + pi2.get("t_max", 0)) / 2,
            }

    return averaged


# =============================================================================
# UTILITATS
# =============================================================================
def get_review_summary_text(review_data):
    """
    Genera un text resum de la revisió.

    Args:
        review_data: Dict retornat per review_sequence()

    Returns:
        String amb resum formatat
    """
    summary = review_data.get("summary", {})
    alerts = review_data.get("alerts", [])

    lines = [
        f"=== RESUM REVISIÓ: {review_data.get('seq_name', 'UNKNOWN')} ===",
        f"Mètode: {review_data.get('method', 'UNKNOWN')}",
        f"Total mostres: {summary.get('total_samples', 0)}",
        f"  OK: {summary.get('ok', 0)}",
        f"  CHECK: {summary.get('check', 0)}",
        f"  WARNING: {summary.get('warning', 0)}",
        f"  DISCARD: {summary.get('discard', 0)}",
    ]

    # Comptar mostres MIXED (selecció DOC≠DAD)
    mixed_samples = []
    for name, review in review_data.get("samples", {}).items():
        if review.get("selected") == "MIXED":
            mixed_samples.append({
                "name": name,
                "doc": review.get("selected_doc"),
                "dad": review.get("selected_dad"),
            })

    if mixed_samples:
        lines.append("")
        lines.append(f"=== SELECCIÓ MIXTA DOC/DAD ({len(mixed_samples)}) ===")
        for m in mixed_samples[:5]:
            lines.append(f"  {m['name']}: DOC={m['doc']}, DAD={m['dad']}")
        if len(mixed_samples) > 5:
            lines.append(f"  ... i {len(mixed_samples) - 5} més")

    if alerts:
        lines.append("")
        lines.append(f"=== ALERTES ({len(alerts)}) ===")
        for alert in alerts[:10]:  # Màxim 10 alertes
            lines.append(f"  {alert.get('sample')}: {alert.get('status')} - {alert.get('reason', '')}")
        if len(alerts) > 10:
            lines.append(f"  ... i {len(alerts) - 10} més")

    return "\n".join(lines)


# =============================================================================
# EXPORTS
# =============================================================================
__all__ = [
    # Config
    "DEFAULT_REVIEW_CONFIG",
    "REVIEW_STATUS",
    # Estadístiques
    "collect_sample_stats",
    # Revisió
    "review_sample",
    "review_sequence",
    # Aplicació
    "apply_selections",
    # Utilitats
    "get_review_summary_text",
    # Re-exports des de hpsec_replica
    "evaluate_replica",
    "compare_replicas",
    "select_best_replica",
    "evaluate_and_select",
]
