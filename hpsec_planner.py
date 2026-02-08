# -*- coding: utf-8 -*-
"""
hpsec_planner.py - Planificador de Seqüències HPSEC
===================================================

Predicció i optimització de timeouts TOC per seqüències HPLC.

Model basat en:
- Cicle recàrrega xeringa TOC: 77.2 min
- Timeout: 74 segons
- Deriva = Duració_mostra - 77.2 min

API:
    predict_timeouts(n_samples, sample_duration, t0, mode) -> list[dict]
    optimize_sequence(n_samples, sample_duration, mode) -> dict
    get_critical_zone(mode) -> tuple

Autor: HPSEC Suite
Versió: 1.0
"""

import numpy as np
from typing import List, Dict, Tuple, Optional

# =============================================================================
# CONSTANTS
# =============================================================================

# Paràmetres del sistema TOC
TOC_CYCLE_MIN = 77.2          # Cicle recàrrega xeringa (minuts)
TOC_TIMEOUT_SEC = 74          # Duració del timeout (segons)
TOC_TIMEOUT_MIN = 74 / 60     # ~1.23 min

# Durades típiques de mostra
SAMPLE_DURATION_CURRENT = 78.65  # Duració actual (min)
SAMPLE_DURATION_OPTIMAL = 77.2   # Duració òptima (min)

# Zones cromatograma per mode
ZONES = {
    "COLUMN": {
        "EARLY":    {"start": 0,  "end": 5,  "severity": "WARNING",  "name": "Pre-pic"},
        "BioP":     {"start": 5,  "end": 18, "severity": "CRITICAL", "name": "Biopolimers"},
        "HS":       {"start": 18, "end": 23, "severity": "CRITICAL", "name": "Acids Humics"},
        "BB":       {"start": 23, "end": 30, "severity": "CRITICAL", "name": "Building Blocks"},
        "SB":       {"start": 30, "end": 40, "severity": "WARNING",  "name": "Small BB"},
        "LMW":      {"start": 40, "end": 70, "severity": "INFO",     "name": "Low MW"},
        "POST_RUN": {"start": 70, "end": 999,"severity": "OK",       "name": "Post-run"},
    },
    "BP": {
        "MAIN":     {"start": 0,  "end": 4,  "severity": "CRITICAL", "name": "Pic Principal"},
        "TAIL":     {"start": 4,  "end": 8,  "severity": "WARNING",  "name": "Cua"},
        "POST_RUN": {"start": 8,  "end": 999,"severity": "OK",       "name": "Post-run"},
    }
}

# Rang crític per mode (zona on NO volem timeouts)
CRITICAL_ZONE = {
    "COLUMN": (5, 30),   # 5-30 min és crític (BioP, HS, BB)
    "BP": (0, 4),        # 0-4 min és crític (pic principal)
}


# =============================================================================
# FUNCIONS PÚBLIQUES
# =============================================================================

def get_zones(mode: str = "COLUMN") -> Dict:
    """Retorna les zones per al mode especificat."""
    return ZONES.get(mode.upper(), ZONES["COLUMN"])


def get_critical_zone(mode: str = "COLUMN") -> Tuple[float, float]:
    """Retorna el rang crític (start, end) per al mode."""
    return CRITICAL_ZONE.get(mode.upper(), CRITICAL_ZONE["COLUMN"])


def predict_timeout_position(sample_num: int, t0: float,
                              sample_duration: float = SAMPLE_DURATION_CURRENT) -> Optional[float]:
    """
    Prediu la posició del timeout dins d'una mostra.

    Mètode: Calcula temps absoluts i troba si algun timeout cau dins la mostra.

    Args:
        sample_num: Número de mostra (1, 2, 3...)
        t0: Temps del primer timeout (min des de l'inici de la seqüència)
        sample_duration: Duració de cada mostra (min)

    Returns:
        Posició del timeout dins la mostra (min), o None si no hi ha timeout
    """
    # Temps d'inici i final d'aquesta mostra
    sample_start = (sample_num - 1) * sample_duration
    sample_end = sample_num * sample_duration

    # Buscar si algun timeout cau dins aquest rang
    # Els timeouts ocorren a: t0, t0 + 77.2, t0 + 2*77.2, ...
    # Primer timeout després de sample_start
    if t0 >= sample_start:
        first_timeout_after_start = t0
    else:
        # Quants cicles complets han passat?
        cycles_passed = int((sample_start - t0) / TOC_CYCLE_MIN)
        first_timeout_after_start = t0 + (cycles_passed + 1) * TOC_CYCLE_MIN

    # Si aquest timeout cau dins la mostra, retornar la posició relativa
    if first_timeout_after_start < sample_end:
        pos_in_sample = first_timeout_after_start - sample_start
        return pos_in_sample

    return None


def predict_timeouts(n_samples: int, sample_duration: float = SAMPLE_DURATION_CURRENT,
                     t0: float = 40.0, mode: str = "COLUMN") -> List[Dict]:
    """
    Prediu els timeouts per a tota una seqüència.

    Args:
        n_samples: Nombre de mostres a la seqüència
        sample_duration: Duració de cada mostra (min)
        t0: Temps del primer timeout dins la primera mostra (min)
        mode: "COLUMN" o "BP"

    Returns:
        Llista de dicts amb info de cada mostra:
        - sample: número de mostra
        - timeout_pos: posició del timeout (min) o None
        - zone: zona afectada
        - severity: severitat (CRITICAL, WARNING, INFO, OK)
        - in_critical: bool si està en zona crítica
    """
    results = []
    zones = get_zones(mode)
    critical_start, critical_end = get_critical_zone(mode)

    for i in range(1, n_samples + 1):
        pos = predict_timeout_position(i, t0, sample_duration)

        # Determinar zona
        zone_name = "POST_RUN"
        severity = "OK"

        if pos is not None:
            for zname, zdata in zones.items():
                if zdata["start"] <= pos < zdata["end"]:
                    zone_name = zname
                    severity = zdata["severity"]
                    break

        in_critical = pos is not None and critical_start <= pos < critical_end

        results.append({
            "sample": i,
            "timeout_pos": pos,
            "zone": zone_name,
            "severity": severity,
            "in_critical": in_critical,
        })

    return results


def calculate_statistics(predictions: List[Dict]) -> Dict:
    """
    Calcula estadístiques d'una predicció.

    Returns:
        Dict amb: total, critical_count, critical_pct, by_zone, by_severity
    """
    total = len(predictions)
    critical = sum(1 for p in predictions if p["in_critical"])

    by_zone = {}
    by_severity = {"CRITICAL": 0, "WARNING": 0, "INFO": 0, "OK": 0}

    for p in predictions:
        zone = p["zone"]
        by_zone[zone] = by_zone.get(zone, 0) + 1
        by_severity[p["severity"]] += 1

    return {
        "total": total,
        "critical_count": critical,
        "critical_pct": (critical / total * 100) if total > 0 else 0,
        "ok_count": total - critical,
        "ok_pct": ((total - critical) / total * 100) if total > 0 else 0,
        "by_zone": by_zone,
        "by_severity": by_severity,
    }


def recommend_blank_injections(predictions: List[Dict], mode: str = "COLUMN") -> List[Dict]:
    """
    Recomana injeccions de MQ/NaOH per evitar timeouts en zona crítica.

    La idea és: si el timeout cau en zona crítica a la mostra X,
    inserir un blanc (MQ o NaOH) abans desplaça el timeout.

    Returns:
        Llista de recomanacions amb: sample_before, reason, action
    """
    recommendations = []
    critical_samples = [p for p in predictions if p["in_critical"]]

    if not critical_samples:
        return recommendations

    # Agrupar consecutius
    groups = []
    current_group = []

    for p in critical_samples:
        if not current_group or p["sample"] == current_group[-1]["sample"] + 1:
            current_group.append(p)
        else:
            if current_group:
                groups.append(current_group)
            current_group = [p]
    if current_group:
        groups.append(current_group)

    # Per cada grup, recomanar injecció de blanc abans del primer
    for group in groups:
        first_sample = group[0]["sample"]
        last_sample = group[-1]["sample"]

        if len(group) == 1:
            reason = f"Timeout en zona crítica a mostra {first_sample}"
        else:
            reason = f"Timeouts en zona crítica mostres {first_sample}-{last_sample}"

        recommendations.append({
            "insert_before": first_sample,
            "samples_affected": [p["sample"] for p in group],
            "reason": reason,
            "action": f"Injectar MQ o NaOH abans de mostra {first_sample}",
            "effect": "Desplaça timeout fora de zona crítica"
        })

    return recommendations


def find_optimal_t0(n_samples: int, sample_duration: float = SAMPLE_DURATION_CURRENT,
                    mode: str = "COLUMN") -> Dict:
    """
    Troba el T0 òptim que minimitza timeouts en zona crítica.

    Returns:
        Dict amb: best_t0, best_critical_pct, worst_t0, worst_critical_pct
    """
    best_t0 = 0
    best_critical = 100
    worst_t0 = 0
    worst_critical = 0

    # Escanejar tots els T0 possibles
    for t0 in np.arange(0, TOC_CYCLE_MIN, 0.5):
        predictions = predict_timeouts(n_samples, sample_duration, t0, mode)
        stats = calculate_statistics(predictions)

        if stats["critical_pct"] < best_critical:
            best_critical = stats["critical_pct"]
            best_t0 = t0

        if stats["critical_pct"] > worst_critical:
            worst_critical = stats["critical_pct"]
            worst_t0 = t0

    return {
        "best_t0": best_t0,
        "best_critical_pct": best_critical,
        "worst_t0": worst_t0,
        "worst_critical_pct": worst_critical,
    }


def optimize_sequence(n_samples: int, sample_duration: float = SAMPLE_DURATION_CURRENT,
                      mode: str = "COLUMN", current_t0: Optional[float] = None) -> Dict:
    """
    Proporciona recomanacions d'optimització per una seqüència.

    Args:
        n_samples: Nombre de mostres
        sample_duration: Duració actual de mostra (min)
        mode: "COLUMN" o "BP"
        current_t0: T0 actual (si es coneix)

    Returns:
        Dict amb recomanacions i prediccions
    """
    # Predicció amb paràmetres actuals
    current_predictions = None
    current_stats = None
    if current_t0 is not None:
        current_predictions = predict_timeouts(n_samples, sample_duration, current_t0, mode)
        current_stats = calculate_statistics(current_predictions)

    # Trobar T0 òptim amb duració actual
    optimal_current = find_optimal_t0(n_samples, sample_duration, mode)

    # Predicció amb duració òptima (77.2 min)
    optimal_duration = find_optimal_t0(n_samples, SAMPLE_DURATION_OPTIMAL, mode)

    # Calcular delay recomanat
    # Si usem 77.2 min, el timeout sempre cau a T0 (deriva = 0)
    # Volem T0 > zona_crítica
    critical_start, critical_end = get_critical_zone(mode)
    recommended_t0 = critical_end + 5  # 5 min de marge

    # Delay necessari després del flush per aconseguir T0 desitjat
    # Això depèn de quan es va fer l'últim cicle TOC

    result = {
        "mode": mode,
        "n_samples": n_samples,
        "sample_duration": sample_duration,

        # Situació actual
        "current": {
            "t0": current_t0,
            "predictions": current_predictions,
            "stats": current_stats,
        } if current_t0 is not None else None,

        # Optimització amb duració actual
        "optimal_with_current_duration": {
            "best_t0": optimal_current["best_t0"],
            "critical_pct": optimal_current["best_critical_pct"],
            "worst_t0": optimal_current["worst_t0"],
            "worst_critical_pct": optimal_current["worst_critical_pct"],
        },

        # Optimització amb duració 77.2 min
        "optimal_with_77min": {
            "duration": SAMPLE_DURATION_OPTIMAL,
            "best_t0": optimal_duration["best_t0"],
            "critical_pct": optimal_duration["best_critical_pct"],
            "recommended_t0": recommended_t0,
            "deriva": 0,  # Amb 77.2 min, la deriva és 0
        },

        # Recomanacions
        "recommendations": [],
    }

    # Generar recomanacions
    if sample_duration != SAMPLE_DURATION_OPTIMAL:
        result["recommendations"].append({
            "type": "DURATION",
            "message": f"Canviar duració de mostra de {sample_duration:.1f} a {SAMPLE_DURATION_OPTIMAL} min",
            "benefit": "Elimina la deriva - timeout sempre a la mateixa posició",
            "priority": "HIGH",
        })

    if current_t0 is not None and current_stats and current_stats["critical_pct"] > 20:
        result["recommendations"].append({
            "type": "DELAY",
            "message": f"Ajustar delay inicial per obtenir T0 ≈ {recommended_t0:.0f} min",
            "benefit": f"Reduiria timeouts crítics de {current_stats['critical_pct']:.0f}% a ~0%",
            "priority": "HIGH",
        })

    result["recommendations"].append({
        "type": "WAIT",
        "message": "Esperar 3-5 min després del flush abans d'iniciar seqüència",
        "benefit": "Permet estabilització i control del T0",
        "priority": "MEDIUM",
    })

    return result


def format_prediction_table(predictions: List[Dict], max_rows: int = 20) -> str:
    """Formata les prediccions com a taula de text."""
    lines = []
    lines.append("=" * 60)
    lines.append(f"{'Mostra':>6} | {'Timeout (min)':>13} | {'Zona':>10} | {'Severitat':>10}")
    lines.append("-" * 60)

    for i, p in enumerate(predictions[:max_rows]):
        pos_str = f"{p['timeout_pos']:.1f}" if p['timeout_pos'] is not None else "-"
        sev_mark = "⚠️" if p['severity'] == "CRITICAL" else ("⚡" if p['severity'] == "WARNING" else "✓")
        lines.append(f"{p['sample']:>6} | {pos_str:>13} | {p['zone']:>10} | {sev_mark} {p['severity']:>8}")

    if len(predictions) > max_rows:
        lines.append(f"... i {len(predictions) - max_rows} mostres més")

    lines.append("=" * 60)

    return "\n".join(lines)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("PLANIFICADOR DE SEQÜÈNCIES HPSEC")
    print("=" * 60)

    # Exemple: seqüència de 29 mostres, duració actual
    n = 29
    duration = 78.65
    t0 = 40.0

    print(f"\nSeqüència: {n} mostres, {duration} min/mostra, T0={t0} min")
    print(f"Mode: COLUMN")

    # Predicció
    predictions = predict_timeouts(n, duration, t0, "COLUMN")
    stats = calculate_statistics(predictions)

    print(f"\nResultats:")
    print(f"  Timeouts en zona crítica: {stats['critical_count']}/{stats['total']} ({stats['critical_pct']:.1f}%)")
    print(f"  Timeouts OK: {stats['ok_count']}/{stats['total']} ({stats['ok_pct']:.1f}%)")

    print(f"\nPer zona:")
    for zone, count in stats['by_zone'].items():
        print(f"  {zone}: {count}")

    # Optimització
    print("\n" + "=" * 60)
    print("OPTIMITZACIÓ")
    opt = optimize_sequence(n, duration, "COLUMN", t0)

    print(f"\nAmb duració actual ({duration} min):")
    print(f"  Millor T0: {opt['optimal_with_current_duration']['best_t0']:.1f} min → {opt['optimal_with_current_duration']['critical_pct']:.1f}% crítics")
    print(f"  Pitjor T0: {opt['optimal_with_current_duration']['worst_t0']:.1f} min → {opt['optimal_with_current_duration']['worst_critical_pct']:.1f}% crítics")

    print(f"\nAmb duració òptima ({SAMPLE_DURATION_OPTIMAL} min):")
    print(f"  Millor T0: {opt['optimal_with_77min']['best_t0']:.1f} min → {opt['optimal_with_77min']['critical_pct']:.1f}% crítics")
    print(f"  T0 recomanat: {opt['optimal_with_77min']['recommended_t0']:.1f} min")

    print("\nRecomanacions:")
    for rec in opt['recommendations']:
        print(f"  [{rec['priority']}] {rec['message']}")
