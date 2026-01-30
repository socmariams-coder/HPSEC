#!/usr/bin/env python3
"""
Batch Processing de Dades3 amb el nou flux de 5 fases.

Executa: Import → Calibrate → Process per cada SEQ de Dades3.
Genera Excels consolidats a la carpeta Dades_Consolidades de cada SEQ.

Ús:
    python batch_process_dades3.py
"""

import os
import sys
import traceback
from pathlib import Path
from datetime import datetime

# Afegir directori HPSEC al path
sys.path.insert(0, str(Path(__file__).parent))

from hpsec_import import import_sequence
from hpsec_calibrate import calibrate_from_import
from hpsec_process import process_sequence, write_consolidated_excel
from hpsec_config import get_config

# Configuració
DADES3 = Path(r"C:\Users\Lequia\Desktop\Dades3")
LOG_FILE = DADES3 / "batch_process_log.txt"


def log_message(msg, log_file=None):
    """Escriu missatge a consola i fitxer."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{timestamp}] {msg}"
    print(full_msg)
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(full_msg + "\n")


def find_seq_folders(base_path):
    """Troba totes les carpetes SEQ dins de Dades3."""
    seq_folders = []
    for item in base_path.iterdir():
        if item.is_dir() and "SEQ" in item.name.upper():
            seq_folders.append(item)
    return sorted(seq_folders)


def write_excels_from_processed(seq_path, processed_data, imported_data, calibration_data):
    """
    Escriu Excels consolidats per cada mostra processada.

    Args:
        seq_path: Path de la seqüència
        processed_data: Resultat de process_sequence()
        imported_data: Resultat de import_sequence()
        calibration_data: Resultat de calibrate_from_import()
    """
    # Crear carpeta Dades_Consolidades
    check_dir = seq_path / "Dades_Consolidades"
    check_dir.mkdir(exist_ok=True)

    method = processed_data.get("method", "COLUMN")
    seq_name = processed_data.get("seq_name", seq_path.name)

    # Obtenir info del MasterFile
    master_info = imported_data.get("master_info", {})
    date_master = master_info.get("date", datetime.now().strftime("%Y-%m-%d"))

    # Shifts de calibració
    shift_uib = calibration_data.get("shift_uib", 0) if calibration_data else 0
    shift_direct = calibration_data.get("shift_direct", 0) if calibration_data else 0

    excels_written = 0

    # Combinar totes les mostres
    all_processed = (
        processed_data.get("samples", []) +
        processed_data.get("khp_samples", []) +
        processed_data.get("control_samples", [])
    )

    for sample in all_processed:
        if not sample.get("processed"):
            continue

        mostra = sample.get("name", "UNKNOWN")
        rep = sample.get("replica", "1")  # replica pot ser str o int

        # Construir nom fitxer (rep com A, B per compatibilitat)
        try:
            rep_int = int(rep)
            rep_letter = chr(ord('A') + rep_int - 1)
        except (ValueError, TypeError):
            rep_letter = str(rep)
        out_name = f"{seq_name}_{mostra}_{rep_letter}.xlsx"
        out_path = check_dir / out_name

        try:
            # Extreure dades per escriure
            peak_info = sample.get("peak_info", {})
            timeout_info = sample.get("timeout_info", {})
            snr_info = sample.get("snr_info", {})

            # Obtenir dades originals d'imported_data
            sample_data = None
            samples_dict = imported_data.get("samples", {})
            if mostra in samples_dict:
                replicas = samples_dict[mostra].get("replicas", {})
                # rep pot ser int o str, provar ambdós
                rep_str = str(rep)
                if rep_str in replicas:
                    sample_data = replicas[rep_str]
                elif rep in replicas:
                    sample_data = replicas[rep]

            # Dades del sample processat
            t_doc = sample.get("t_doc", [])
            y_doc_net = sample.get("y_doc_net", [])
            df_dad = sample.get("df_dad")

            # Dades de imported_data (nova estructura v2)
            direct_data = sample_data.get("direct") if sample_data else None
            uib_data = sample_data.get("uib") if sample_data else None
            dad_data = sample_data.get("dad") if sample_data else None

            # Determinar mode i dades principals
            has_direct = direct_data is not None and direct_data.get("t") is not None
            has_uib = uib_data is not None and uib_data.get("t") is not None
            is_dual = has_direct and has_uib

            if is_dual:
                doc_mode = "DUAL"
                # DOC Direct com a principal
                t_doc = direct_data["t"]
                y_doc_raw = direct_data["y"]
                y_doc_net = direct_data.get("y_net", direct_data["y"])
                baseline = direct_data.get("baseline", 0)
                fitxer_doc = direct_data.get("path", "")
                # UIB com a secundari
                y_doc_uib = uib_data.get("y_net", uib_data["y"])
                y_doc_uib_raw = uib_data["y"]
                baseline_uib = uib_data.get("baseline", 0)
                fitxer_doc_uib = uib_data.get("path", "")
            elif has_direct:
                doc_mode = "DIRECT"
                t_doc = direct_data["t"]
                y_doc_raw = direct_data["y"]
                y_doc_net = direct_data.get("y_net", direct_data["y"])
                baseline = direct_data.get("baseline", 0)
                fitxer_doc = direct_data.get("path", "")
                y_doc_uib = None
                y_doc_uib_raw = None
                baseline_uib = None
                fitxer_doc_uib = None
            elif has_uib:
                doc_mode = "UIB"
                t_doc = uib_data["t"]
                y_doc_raw = uib_data["y"]
                y_doc_net = uib_data.get("y_net", uib_data["y"])
                baseline = uib_data.get("baseline", 0)
                fitxer_doc = uib_data.get("path", "")
                y_doc_uib = None
                y_doc_uib_raw = None
                baseline_uib = None
                fitxer_doc_uib = None
            else:
                # Sense dades DOC, saltar
                print(f"  SKIP {mostra}: no DOC data")
                continue

            st_doc = ""
            st_doc_uib = ""

            # DAD
            fitxer_dad = dad_data.get("path", "") if dad_data else ""
            st_dad = ""

            # Escriure Excel
            write_consolidated_excel(
                out_path=str(out_path),
                mostra=mostra,
                rep=rep_letter,
                seq_out=seq_name,
                date_master=date_master,
                method=method,
                doc_mode=doc_mode,
                fitxer_doc=fitxer_doc,
                fitxer_dad=fitxer_dad,
                st_doc=st_doc,
                st_dad=st_dad,
                t_doc=t_doc,
                y_doc_raw=y_doc_raw,
                y_doc_net=y_doc_net,
                baseline=baseline,
                df_dad=df_dad,
                peak_info=peak_info,
                sample_analysis=sample,
                master_file=str(imported_data.get("master_file", "")),
                y_doc_uib=y_doc_uib,
                y_doc_uib_raw=y_doc_uib_raw,
                baseline_uib=baseline_uib,
                fitxer_doc_uib=fitxer_doc_uib,
                st_doc_uib=st_doc_uib,
                shift_uib=shift_uib,
                shift_direct=shift_direct,
                master_info=master_info,
                timeout_info=timeout_info,
                snr_info=snr_info,
            )
            excels_written += 1

        except Exception as e:
            print(f"  ERROR escrivint {out_name}: {e}")
            traceback.print_exc()

    return excels_written


def process_single_seq(seq_path, config=None):
    """
    Processa una seqüència completa: Import → Calibrate → Process → Write.

    Returns:
        dict amb resultat i estadístiques
    """
    result = {
        "seq_name": seq_path.name,
        "success": False,
        "import_success": False,
        "calibrate_success": False,
        "process_success": False,
        "n_samples": 0,
        "n_excels": 0,
        "error": None,
    }

    try:
        # FASE 1: IMPORT
        print(f"  1. Importing...")
        imported_data = import_sequence(str(seq_path))

        if not imported_data.get("success"):
            result["error"] = f"Import failed: {imported_data.get('error', 'Unknown')}"
            return result

        result["import_success"] = True
        result["n_samples"] = len(imported_data.get("samples", {}))
        print(f"     -> {result['n_samples']} samples imported")

        # FASE 2: CALIBRATE
        print(f"  2. Calibrating...")
        calibration_data = calibrate_from_import(imported_data, config=config)

        if calibration_data and calibration_data.get("success"):
            result["calibrate_success"] = True
            cal_mode = calibration_data.get("mode", "?")
            msg_parts = [f"Mode: {cal_mode}"]
            if calibration_data.get("factor_direct", 0) > 0:
                msg_parts.append(f"Direct={calibration_data['factor_direct']:.4f}")
            if calibration_data.get("factor_uib", 0) > 0:
                msg_parts.append(f"UIB={calibration_data['factor_uib']:.4f}")
            if calibration_data.get("shift_uib", 0) != 0:
                msg_parts.append(f"Shift UIB={calibration_data['shift_uib']:.2f}min")
            print(f"     -> {', '.join(msg_parts)}")
        else:
            # Calibracio pot fallar si no hi ha KHP, continuem amb shifts 0
            print(f"     -> No KHP calibration (using defaults)")
            calibration_data = {"shift_uib": 0, "shift_direct": 0, "factor": 1.0, "factor_direct": 0, "factor_uib": 0}

        # FASE 3: PROCESS
        print(f"  3. Processing...")
        processed_data = process_sequence(imported_data, calibration_data, config=config)

        if not processed_data.get("success") and len(processed_data.get("errors", [])) > 0:
            result["error"] = f"Process failed: {processed_data.get('errors', [])}"
            return result

        result["process_success"] = True
        n_processed = len(processed_data.get("samples", []))
        print(f"     -> {n_processed} samples processed")

        # ESCRIURE EXCELS
        print(f"  4. Writing Excels...")
        n_excels = write_excels_from_processed(seq_path, processed_data, imported_data, calibration_data)
        result["n_excels"] = n_excels
        print(f"     -> {n_excels} Excel files written to Dades_Consolidades/")

        result["success"] = True

    except Exception as e:
        result["error"] = str(e)
        traceback.print_exc()

    return result


def main():
    """Processa totes les seqüències de Dades3."""
    print("="*70)
    print("BATCH PROCESSING DE DADES3")
    print("="*70)
    print(f"Directori: {DADES3}")
    print()

    # Inicialitzar log
    log_message("Batch processing started", LOG_FILE)

    # Trobar SEQs
    seq_folders = find_seq_folders(DADES3)
    print(f"Trobades {len(seq_folders)} seqüències")
    print()

    # Processar cada SEQ
    results = []
    for i, seq_folder in enumerate(seq_folders):
        print(f"\n[{i+1}/{len(seq_folders)}] {seq_folder.name}")
        print("-"*50)

        result = process_single_seq(seq_folder)
        results.append(result)

        status = "OK" if result["success"] else f"FAILED: {result.get('error', 'Unknown')}"
        log_message(f"{seq_folder.name}: {status}", LOG_FILE)

    # Resum final
    print("\n" + "="*70)
    print("RESUM FINAL")
    print("="*70)

    n_success = sum(1 for r in results if r["success"])
    n_failed = len(results) - n_success
    total_excels = sum(r.get("n_excels", 0) for r in results)

    print(f"Seqüències processades: {len(results)}")
    print(f"  Correctes: {n_success}")
    print(f"  Amb errors: {n_failed}")
    print(f"Excels generats: {total_excels}")

    if n_failed > 0:
        print("\nSeqüències amb errors:")
        for r in results:
            if not r["success"]:
                print(f"  - {r['seq_name']}: {r.get('error', 'Unknown')}")

    log_message(f"Batch completed: {n_success}/{len(results)} OK, {total_excels} Excels", LOG_FILE)

    return n_failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
