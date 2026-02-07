# -*- coding: utf-8 -*-
"""
HPSEC Development Notes System
==============================

Sistema centralitzat per recopilar notes durant el processament.
Útil per:
- Detectar problemes de detecció (batman/jagged peaks)
- Registrar casos especials o edge cases
- Facilitar debugging entre sessions
- Guiar la següent iteració de desenvolupament

Ús:
    from hpsec_dev_notes import add_note, add_detection_issue, get_notes_summary

    # Afegir nota general
    add_note("283_SEQ", "import", "El fitxer UIB té format estrany")

    # Reportar problema de detecció
    add_detection_issue("283_SEQ", "FR2606_R1", "batman",
                       details={"t_max": 25.3, "expected": 22.0})
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any

# Path del fitxer de notes (REGISTRY)
def _get_notes_path() -> Path:
    """Retorna el path del fitxer de notes."""
    from hpsec_config import load_config
    config = load_config()
    registry = Path(config.get("paths", {}).get("registry", ""))
    if not registry.exists():
        registry = Path(__file__).parent / "REGISTRY"
    return registry / "Development_Notes.json"


def _load_notes() -> Dict:
    """Carrega les notes existents o crea estructura buida."""
    path = _get_notes_path()
    if path.exists():
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass

    return {
        "version": "1.0",
        "created": datetime.now().isoformat(),
        "updated": None,
        "description": "Notes de desenvolupament - detectar problemes i guiar millores",
        "sessions": [],
        "current_session": None,
        "detection_issues": [],
        "general_notes": [],
        "stats": {
            "total_notes": 0,
            "total_detection_issues": 0,
            "issues_by_type": {}
        }
    }


def _save_notes(notes: Dict):
    """Guarda les notes al fitxer."""
    path = _get_notes_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    notes["updated"] = datetime.now().isoformat()

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(notes, f, indent=2, ensure_ascii=False)


def start_session(description: str = "") -> str:
    """
    Inicia una nova sessió de desenvolupament.
    Retorna l'ID de la sessió.
    """
    notes = _load_notes()

    session_id = datetime.now().strftime("DEV_%Y%m%d_%H%M%S")
    session = {
        "id": session_id,
        "started": datetime.now().isoformat(),
        "ended": None,
        "description": description,
        "sequences_processed": [],
        "notes_count": 0,
        "issues_count": 0
    }

    notes["sessions"].append(session)
    notes["current_session"] = session_id

    _save_notes(notes)
    print(f"[DEV] Sessió iniciada: {session_id}")
    return session_id


def end_session(summary: str = ""):
    """Finalitza la sessió actual."""
    notes = _load_notes()

    session_id = notes.get("current_session")
    if not session_id:
        return

    for session in notes["sessions"]:
        if session["id"] == session_id:
            session["ended"] = datetime.now().isoformat()
            if summary:
                session["summary"] = summary
            break

    notes["current_session"] = None
    _save_notes(notes)
    print(f"[DEV] Sessió finalitzada: {session_id}")


def add_note(
    seq_name: str,
    stage: str,
    message: str,
    category: str = "general",
    details: Optional[Dict] = None
):
    """
    Afegeix una nota general.

    Args:
        seq_name: Nom de la seqüència (ex: "283_SEQ")
        stage: Etapa (import, calibrate, analyze, consolidate)
        message: Missatge descriptiu
        category: Categoria (general, warning, error, todo, question)
        details: Detalls addicionals (diccionari)
    """
    notes = _load_notes()

    note = {
        "timestamp": datetime.now().isoformat(),
        "session": notes.get("current_session"),
        "seq_name": seq_name,
        "stage": stage,
        "category": category,
        "message": message,
        "details": details or {}
    }

    notes["general_notes"].append(note)
    notes["stats"]["total_notes"] += 1

    # Actualitzar sessió actual
    if notes.get("current_session"):
        for session in notes["sessions"]:
            if session["id"] == notes["current_session"]:
                session["notes_count"] += 1
                if seq_name not in session["sequences_processed"]:
                    session["sequences_processed"].append(seq_name)
                break

    _save_notes(notes)
    print(f"[DEV] Nota ({category}): {seq_name}/{stage} - {message}")


def add_detection_issue(
    seq_name: str,
    sample_name: str,
    issue_type: str,
    signal: str = "direct",
    details: Optional[Dict] = None,
    severity: str = "warning"
):
    """
    Reporta un problema de detecció de pics.

    Args:
        seq_name: Nom de la seqüència
        sample_name: Nom de la mostra (amb rèplica, ex: "FR2606_R1")
        issue_type: Tipus de problema:
            - "batman": Pic amb forma batman/jagged
            - "no_peak": No s'ha detectat pic
            - "multiple_peaks": Múltiples pics inesperats
            - "wrong_position": Pic en posició incorrecta
            - "baseline_issue": Problema amb baseline
            - "asymmetry": Asimetria excessiva
            - "noise": Soroll excessiu
        signal: Senyal afectat (direct, uib, dad)
        details: Detalls (t_max, expected_t, area, etc.)
        severity: Gravetat (info, warning, error)
    """
    notes = _load_notes()

    issue = {
        "timestamp": datetime.now().isoformat(),
        "session": notes.get("current_session"),
        "seq_name": seq_name,
        "sample_name": sample_name,
        "issue_type": issue_type,
        "signal": signal,
        "severity": severity,
        "details": details or {},
        "resolved": False,
        "resolution": None
    }

    notes["detection_issues"].append(issue)
    notes["stats"]["total_detection_issues"] += 1

    # Comptar per tipus
    if issue_type not in notes["stats"]["issues_by_type"]:
        notes["stats"]["issues_by_type"][issue_type] = 0
    notes["stats"]["issues_by_type"][issue_type] += 1

    # Actualitzar sessió actual
    if notes.get("current_session"):
        for session in notes["sessions"]:
            if session["id"] == notes["current_session"]:
                session["issues_count"] += 1
                if seq_name not in session["sequences_processed"]:
                    session["sequences_processed"].append(seq_name)
                break

    _save_notes(notes)
    print(f"[DEV] Issue ({issue_type}): {seq_name}/{sample_name} [{signal}] - {severity}")


def add_calibration_note(
    seq_name: str,
    khp_name: str,
    message: str,
    metrics: Optional[Dict] = None
):
    """Nota específica per calibració."""
    add_note(
        seq_name=seq_name,
        stage="calibrate",
        message=message,
        category="calibration",
        details={
            "khp_name": khp_name,
            "metrics": metrics or {}
        }
    )


def add_analysis_note(
    seq_name: str,
    sample_name: str,
    message: str,
    metrics: Optional[Dict] = None
):
    """Nota específica per anàlisi."""
    add_note(
        seq_name=seq_name,
        stage="analyze",
        message=message,
        category="analysis",
        details={
            "sample_name": sample_name,
            "metrics": metrics or {}
        }
    )


def get_notes_summary() -> Dict:
    """Retorna un resum de totes les notes."""
    notes = _load_notes()

    # Agrupar issues per tipus
    issues_by_type = {}
    for issue in notes["detection_issues"]:
        itype = issue["issue_type"]
        if itype not in issues_by_type:
            issues_by_type[itype] = []
        issues_by_type[itype].append({
            "seq": issue["seq_name"],
            "sample": issue["sample_name"],
            "signal": issue["signal"],
            "details": issue["details"]
        })

    # Agrupar notes per categoria
    notes_by_category = {}
    for note in notes["general_notes"]:
        cat = note["category"]
        if cat not in notes_by_category:
            notes_by_category[cat] = []
        notes_by_category[cat].append({
            "seq": note["seq_name"],
            "stage": note["stage"],
            "message": note["message"]
        })

    return {
        "total_sessions": len(notes["sessions"]),
        "current_session": notes.get("current_session"),
        "total_notes": notes["stats"]["total_notes"],
        "total_detection_issues": notes["stats"]["total_detection_issues"],
        "issues_by_type": issues_by_type,
        "notes_by_category": notes_by_category,
        "last_updated": notes.get("updated")
    }


def get_issues_for_seq(seq_name: str) -> List[Dict]:
    """Retorna tots els issues d'una seqüència."""
    notes = _load_notes()
    return [
        issue for issue in notes["detection_issues"]
        if issue["seq_name"] == seq_name
    ]


def get_unresolved_issues() -> List[Dict]:
    """Retorna tots els issues no resolts."""
    notes = _load_notes()
    return [
        issue for issue in notes["detection_issues"]
        if not issue.get("resolved", False)
    ]


def mark_issue_resolved(seq_name: str, sample_name: str, resolution: str):
    """Marca un issue com a resolt."""
    notes = _load_notes()

    for issue in notes["detection_issues"]:
        if issue["seq_name"] == seq_name and issue["sample_name"] == sample_name:
            issue["resolved"] = True
            issue["resolution"] = resolution
            issue["resolved_at"] = datetime.now().isoformat()

    _save_notes(notes)


def clear_session_notes(session_id: Optional[str] = None):
    """Esborra les notes d'una sessió (o l'actual)."""
    notes = _load_notes()

    target_session = session_id or notes.get("current_session")
    if not target_session:
        return

    # Filtrar notes i issues
    notes["general_notes"] = [
        n for n in notes["general_notes"]
        if n.get("session") != target_session
    ]
    notes["detection_issues"] = [
        i for i in notes["detection_issues"]
        if i.get("session") != target_session
    ]

    # Recalcular stats
    notes["stats"]["total_notes"] = len(notes["general_notes"])
    notes["stats"]["total_detection_issues"] = len(notes["detection_issues"])
    notes["stats"]["issues_by_type"] = {}
    for issue in notes["detection_issues"]:
        itype = issue["issue_type"]
        if itype not in notes["stats"]["issues_by_type"]:
            notes["stats"]["issues_by_type"][itype] = 0
        notes["stats"]["issues_by_type"][itype] += 1

    _save_notes(notes)
    print(f"[DEV] Notes de sessió {target_session} esborrades")


def print_summary():
    """Imprimeix un resum de les notes."""
    summary = get_notes_summary()

    print("\n" + "="*60)
    print("DEVELOPMENT NOTES SUMMARY")
    print("="*60)
    print(f"Sessions: {summary['total_sessions']}")
    print(f"Current session: {summary['current_session'] or 'None'}")
    print(f"Total notes: {summary['total_notes']}")
    print(f"Total detection issues: {summary['total_detection_issues']}")

    if summary["issues_by_type"]:
        print("\nIssues by type:")
        for itype, issues in summary["issues_by_type"].items():
            print(f"  {itype}: {len(issues)}")
            for issue in issues[:3]:  # Mostrar màxim 3
                print(f"    - {issue['seq']}/{issue['sample']} [{issue['signal']}]")
            if len(issues) > 3:
                print(f"    ... i {len(issues)-3} més")

    if summary["notes_by_category"]:
        print("\nNotes by category:")
        for cat, cat_notes in summary["notes_by_category"].items():
            print(f"  {cat}: {len(cat_notes)}")

    print("="*60 + "\n")


# Per ús directe des de CLI
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        cmd = sys.argv[1]

        if cmd == "summary":
            print_summary()

        elif cmd == "start":
            desc = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else ""
            start_session(desc)

        elif cmd == "end":
            summary = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else ""
            end_session(summary)

        elif cmd == "unresolved":
            issues = get_unresolved_issues()
            print(f"\nUnresolved issues: {len(issues)}")
            for issue in issues:
                print(f"  [{issue['issue_type']}] {issue['seq_name']}/{issue['sample_name']}")

        else:
            print("Usage: python hpsec_dev_notes.py [summary|start|end|unresolved]")
    else:
        print_summary()
