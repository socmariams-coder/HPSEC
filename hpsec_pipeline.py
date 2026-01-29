# -*- coding: utf-8 -*-
"""
hpsec_pipeline.py - Pipeline integrat per processar seqüències HPSEC
====================================================================

Orquestra tot el flux de processament:
1. Consolidar - Llegir .D → Excel consolidats
2. Calibrar - Trobar KHP → calcular factor
3. Processar - Detectar anomalies → triar rèpliques
4. Exportar - Generar fitxers finals

API principal:
    process_sequence(seq_path, callbacks) -> PipelineResult

Autor: HPSEC Suite
Versió: 1.1
"""

# =============================================================================
# VERSIÓ DEL PIPELINE - Incrementar quan hi ha canvis significatius
# =============================================================================
PIPELINE_VERSION = "1.1.0"
PIPELINE_VERSION_DATE = "2025-01-26"

# Historial de versions:
# 1.0.0 - Versió inicial
# 1.1.0 - Afegit tracking de versió, detecció automàtica d'estat

import os
import json
import glob
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, List, Any
import pandas as pd
import numpy as np

# Imports dels mòduls HPSEC
from hpsec_consolidate import consolidate_sequence, check_sequence_files
from hpsec_calibrate import (
    calibrate_sequence, get_active_calibration,
    buscar_khp_consolidados, analizar_khp_consolidado
)
from hpsec_replica import (
    evaluate_replica, select_best_replica, compare_replicas,
    evaluate_dad, compare_replicas_full
)
from hpsec_review import review_sample
from hpsec_config import get_config
from hpsec_migrate_master import migrate_single, find_seq_folders


# =============================================================================
# DATACLASSES PER RESULTATS
# =============================================================================

@dataclass
class StepResult:
    """Resultat d'un pas del pipeline."""
    success: bool
    step: str
    message: str
    data: Dict = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class SampleResult:
    """Resultat del processament d'una mostra."""
    name: str
    replicas: Dict[str, Dict]  # R1, R2 -> evaluation
    best_replica: Optional[str]  # Legacy: mateixa que selected_doc
    selection_reason: str
    selection_confidence: float
    warnings: List[str] = field(default_factory=list)
    calibrated_conc: Optional[float] = None
    # Selecció independent DOC/DAD
    selected_doc: Optional[str] = None  # R1, R2, None
    selected_dad: Optional[str] = None  # R1, R2, None
    selected: Optional[str] = None  # R1, R2, MIXED, None
    selection_reason_doc: str = ""
    selection_reason_dad: str = ""
    # Comparativa entre rèpliques
    comparison: Dict = field(default_factory=dict)  # {pearson, area_diff_pct}
    # Àrees per fracció (de la rèplica seleccionada)
    areas_fraction: Dict = field(default_factory=dict)  # {BioP, HS, BB, SB, LMW, total}
    # Concentracions per fracció
    conc_fraction: Dict = field(default_factory=dict)  # {BioP, HS, BB, SB, LMW, total}
    # Paràmetres de transformació (per reproductibilitat)
    transformation: Dict = field(default_factory=dict)  # {shift_sec, baseline_mau, factor}


@dataclass
class PipelineResult:
    """Resultat complet del pipeline."""
    success: bool
    seq_path: str
    seq_name: str
    mode: str  # COLUMN o BP
    doc_mode: str  # UIB, DIRECT, DUAL

    # Resultats per pas
    migration: Optional[StepResult] = None  # Nou: migració de SEQ antiga
    consolidation: Optional[StepResult] = None
    calibration: Optional[StepResult] = None
    processing: Optional[StepResult] = None
    export: Optional[StepResult] = None

    # Detalls
    samples: List[SampleResult] = field(default_factory=list)
    summary: Dict = field(default_factory=dict)

    # Temps
    started_at: str = ""
    finished_at: str = ""
    duration_sec: float = 0


# =============================================================================
# CALLBACKS
# =============================================================================

@dataclass
class PipelineCallbacks:
    """Callbacks per reportar progrés."""
    on_step_start: Optional[Callable[[str, int, int], None]] = None  # step, current, total
    on_step_progress: Optional[Callable[[str, int, int, str], None]] = None  # step, current, total, item
    on_step_complete: Optional[Callable[[str, StepResult], None]] = None
    on_sample_processed: Optional[Callable[[str, SampleResult], None]] = None
    on_error: Optional[Callable[[str, str], None]] = None  # step, error_msg


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def process_sequence(
    seq_path: str,
    callbacks: Optional[PipelineCallbacks] = None,
    skip_consolidation: bool = False,
    skip_calibration: bool = False,
    skip_migration: bool = False,
    skip_processing: bool = False,
    force_overwrite: bool = False,
    config: Optional[Dict] = None
) -> PipelineResult:
    """
    Processa una seqüència HPSEC completa.

    Args:
        seq_path: Ruta a la carpeta SEQ
        callbacks: Callbacks per reportar progrés
        skip_consolidation: Si True, assumeix que ja està consolidat
        skip_calibration: Si True, usa calibració existent
        skip_migration: Si True, no intenta migrar seqüències antigues
        skip_processing: Si True, assumeix que ja està processat
        force_overwrite: Sobreescriure fitxers existents
        config: Configuració personalitzada

    Returns:
        PipelineResult amb tots els resultats
    """
    callbacks = callbacks or PipelineCallbacks()
    config = config or get_config().config

    seq_name = os.path.basename(seq_path)
    started_at = datetime.now()

    result = PipelineResult(
        success=False,
        seq_path=seq_path,
        seq_name=seq_name,
        mode="",
        doc_mode="",
        started_at=started_at.isoformat(),
    )

    try:
        # =====================================================================
        # PAS 0: MIGRAR (si necessari)
        # =====================================================================
        needs_migration = not skip_migration and _needs_migration(seq_path)
        total_steps = 5 if needs_migration else 4
        step_offset = 1 if needs_migration else 0

        if needs_migration:
            if callbacks.on_step_start:
                callbacks.on_step_start("migrar", 1, total_steps)

            mig_result = _step_migrate(seq_path, callbacks, force_overwrite)
            result.migration = mig_result

            if not mig_result.success:
                if callbacks.on_error:
                    callbacks.on_error("migrar", mig_result.message)
                return result

            # El method pot venir de la migració
            if mig_result.data.get('method'):
                result.mode = mig_result.data['method']

            if callbacks.on_step_complete:
                callbacks.on_step_complete("migrar", mig_result)
        else:
            result.migration = StepResult(
                success=True, step="migrar",
                message="No necessària (ja té MasterFile)"
            )

        # =====================================================================
        # PAS 1: CONSOLIDAR
        # =====================================================================
        if not skip_consolidation:
            if callbacks.on_step_start:
                callbacks.on_step_start("consolidar", 1 + step_offset, total_steps)

            con_result = _step_consolidate(seq_path, callbacks, force_overwrite, config)
            result.consolidation = con_result

            if not con_result.success:
                if callbacks.on_error:
                    callbacks.on_error("consolidar", con_result.message)
                return result

            result.mode = "BP" if con_result.data.get("bp") else "COLUMN"
            result.doc_mode = con_result.data.get("mode", "UNKNOWN")

            if callbacks.on_step_complete:
                callbacks.on_step_complete("consolidar", con_result)
        else:
            # Detectar mode dels fitxers existents
            result.mode, result.doc_mode = _detect_existing_mode(seq_path)
            result.consolidation = StepResult(
                success=True, step="consolidar",
                message="Omès (ja consolidat)"
            )

        # =====================================================================
        # PAS 2: CALIBRAR
        # =====================================================================
        if not skip_calibration:
            if callbacks.on_step_start:
                callbacks.on_step_start("calibrar", 2 + step_offset, total_steps)

            cal_result = _step_calibrate(seq_path, result.mode, callbacks, config)
            result.calibration = cal_result

            if not cal_result.success:
                # Calibració fallida no és fatal, continuem sense
                result.calibration.warnings.append("Processant sense calibració")

            if callbacks.on_step_complete:
                callbacks.on_step_complete("calibrar", cal_result)
        else:
            # Usar calibració existent
            existing_cal = get_active_calibration(seq_path, result.mode)
            if existing_cal:
                result.calibration = StepResult(
                    success=True, step="calibrar",
                    message=f"Usant calibració existent (factor={existing_cal.get('factor', 0):.4f})",
                    data={"calibration": existing_cal}
                )
            else:
                result.calibration = StepResult(
                    success=False, step="calibrar",
                    message="No hi ha calibració disponible"
                )

        # =====================================================================
        # PAS 3: PROCESSAR (detectar anomalies + triar rèpliques)
        # =====================================================================
        if not skip_processing:
            if callbacks.on_step_start:
                callbacks.on_step_start("qc", 3 + step_offset, total_steps)

            # Obtenir dades de calibració per calcular concentracions
            calibration_data = None
            if result.calibration and result.calibration.success:
                calibration_data = result.calibration.data.get("calibration")

            proc_result = _step_process(seq_path, result.mode, callbacks, config, calibration_data)
            result.processing = proc_result
            result.samples = proc_result.data.get("samples", [])

            if callbacks.on_step_complete:
                callbacks.on_step_complete("qc", proc_result)
        else:
            result.processing = StepResult(
                success=True, step="qc",
                message="Omès (ja processat)"
            )
            # Intentar carregar resultats existents
            results_file = os.path.join(seq_path, "CHECK", "processing_summary.json")
            if os.path.exists(results_file):
                try:
                    with open(results_file, 'r', encoding='utf-8') as f:
                        existing = json.load(f)
                    result.processing.data = {"loaded_from": results_file}
                except:
                    pass

        # =====================================================================
        # PAS 4: EXPORTAR
        # =====================================================================
        if callbacks.on_step_start:
            callbacks.on_step_start("exportar", 4 + step_offset, total_steps)

        calibration = result.calibration.data.get("calibration") if result.calibration else None
        exp_result = _step_export(seq_path, result.samples, calibration, callbacks, config)
        result.export = exp_result

        if callbacks.on_step_complete:
            callbacks.on_step_complete("exportar", exp_result)

        # =====================================================================
        # RESUM FINAL
        # =====================================================================
        result.success = True
        result.summary = _create_summary(result)

    except Exception as e:
        result.success = False
        if callbacks.on_error:
            callbacks.on_error("pipeline", str(e))
        # Guardar error al pas corresponent
        if result.migration is None:
            result.migration = StepResult(False, "migrar", str(e))
        elif result.consolidation is None:
            result.consolidation = StepResult(False, "consolidar", str(e))
        elif result.calibration is None:
            result.calibration = StepResult(False, "calibrar", str(e))
        elif result.processing is None:
            result.processing = StepResult(False, "qc", str(e))
        else:
            result.export = StepResult(False, "exportar", str(e))

    finally:
        finished_at = datetime.now()
        result.finished_at = finished_at.isoformat()
        result.duration_sec = (finished_at - started_at).total_seconds()

    return result


# =============================================================================
# PASSOS INDIVIDUALS
# =============================================================================

def _needs_migration(seq_path: str) -> bool:
    """
    Detecta si la seqüència necessita migració (no té MasterFile).

    Una SEQ necessita migració si:
    - No té cap fitxer *_MasterFile.xlsx
    - Té fitxer rawdata (*_v11*.xlsx o *_v12*.xlsx)

    Returns:
        True si necessita migració, False si ja té MasterFile o no té rawdata
    """
    # Buscar MasterFile existent
    masterfiles = glob.glob(os.path.join(seq_path, '*_MasterFile.xlsx'))
    masterfiles = [f for f in masterfiles if 'backup' not in f.lower() and '~$' not in f]

    if masterfiles:
        return False  # Ja té MasterFile

    # Verificar que té rawdata per migrar
    rawdata_patterns = ['*_v11*.xlsx', '*_v12*.xlsx', '*RAWDATA*.xlsx']
    for pattern in rawdata_patterns:
        matches = glob.glob(os.path.join(seq_path, pattern))
        matches = [m for m in matches if 'backup' not in m.lower() and '~$' not in m]
        if matches:
            return True  # Té rawdata per migrar

    return False  # No té rawdata (format desconegut)


def _step_migrate(seq_path: str, callbacks: PipelineCallbacks,
                  force_overwrite: bool) -> StepResult:
    """Pas 0: Migrar seqüència antiga a format MasterFile."""
    try:
        seq_name = os.path.basename(seq_path)

        def progress_cb(step, total, msg):
            if callbacks.on_step_progress:
                callbacks.on_step_progress("migrar", step, total, msg)

        result = migrate_single(seq_path, progress_callback=progress_cb, force=force_overwrite)

        if result['status'] == 'ok':
            return StepResult(
                success=True, step="migrar",
                message=f"MasterFile creat: {os.path.basename(result.get('file', ''))}",
                data={
                    'masterfile': result.get('file'),
                    'rows': result.get('rows', 0),
                    'seq_id': result.get('seq_id'),
                    'method': result.get('method'),
                    'has_khp': result.get('has_khp', False)
                }
            )
        elif result['status'] == 'skip':
            return StepResult(
                success=True, step="migrar",
                message=result.get('message', 'Omès'),
                data={'masterfile': result.get('file')}
            )
        else:
            return StepResult(
                success=False, step="migrar",
                message=result.get('message', 'Error desconegut')
            )

    except Exception as e:
        return StepResult(
            success=False, step="migrar",
            message=str(e)
        )


def _step_consolidate(seq_path: str, callbacks: PipelineCallbacks,
                      force_overwrite: bool, config: Dict) -> StepResult:
    """Pas 1: Consolidar la seqüència."""
    try:
        # Comptar fitxers font (UIB i DAD)
        file_check = check_sequence_files(seq_path)
        n_uib = file_check.get('uib', {}).get('count_found', 0)
        n_dad = file_check.get('dad', {}).get('count_found', 0)

        # Detectar mode
        mode = "UIB" if n_uib > 0 else "DIRECT"

        # Comptar fitxers ja consolidats
        consolidats_path = os.path.join(seq_path, "Resultats_Consolidats")
        n_consolidats = 0
        consolidated_files = []
        if os.path.isdir(consolidats_path):
            consolidated_files = glob.glob(os.path.join(consolidats_path, "*.xlsx"))
            # Excloure fitxers temporals i backups
            consolidated_files = [f for f in consolidated_files
                                 if not os.path.basename(f).startswith('~$')
                                 and 'backup' not in f.lower()]
            n_consolidats = len(consolidated_files)

        # Buscar MasterFile
        masterfile = None
        masterfiles = glob.glob(os.path.join(seq_path, '*_MasterFile.xlsx'))
        masterfiles = [f for f in masterfiles if '~$' not in f and 'backup' not in f.lower()]
        if masterfiles:
            masterfile = masterfiles[0]

        # Info detallada per data
        data = {
            'n_uib_files': n_uib,
            'n_dad_files': n_dad,
            'n_consolidated': n_consolidats,
            'mode': mode,
            'masterfile': os.path.basename(masterfile) if masterfile else None,
            'consolidated_files': [os.path.basename(f) for f in consolidated_files[:10]],  # Màx 10
        }

        # Cas 1: No hi ha fitxers font
        if n_uib == 0 and n_dad == 0:
            msg = "No s'han trobat fitxers per consolidar"
            details = [
                f"Fitxers UIB a CSV/: {n_uib}",
                f"Fitxers DAD a Export3D/: {n_dad}",
            ]
            if n_consolidats > 0:
                details.append(f"Fitxers ja consolidats: {n_consolidats}")
                msg = "No hi ha fitxers nous per consolidar"
            if masterfile:
                details.append(f"MasterFile: {os.path.basename(masterfile)}")

            return StepResult(
                success=False, step="consolidar",
                message=msg,
                data=data,
                warnings=details
            )

        # Cas 2: Ja hi ha consolidats i no forcem sobreescriptura
        if n_consolidats > 0 and not force_overwrite:
            data['already_consolidated'] = True
            return StepResult(
                success=True, step="consolidar",
                message=f"Ja consolidat ({n_consolidats} fitxers existents)",
                data=data,
                warnings=[
                    f"Fitxers UIB trobats: {n_uib}",
                    f"Fitxers DAD trobats: {n_dad}",
                    f"Consolidats existents: {n_consolidats}",
                    "Usa 'força sobreescriptura' per reconsolidar",
                ]
            )

        # Cas 3: Consolidar
        total = n_uib if mode == "UIB" else n_dad
        if total == 0:
            total = max(n_uib, n_dad)

        def progress_cb(pct, sample):
            if callbacks.on_step_progress:
                current = int(pct * total / 100)
                callbacks.on_step_progress("consolidar", current, total, sample)

        result = consolidate_sequence(
            seq_path,
            config=config,
            progress_callback=progress_cb
        )

        if result.get("success"):
            processed = result.get('processed_count', 0)
            data.update(result)
            data['n_files'] = n_uib if mode == "UIB" else n_dad
            data['n_samples'] = processed

            return StepResult(
                success=True, step="consolidar",
                message=f"{processed} mostres consolidades (mode {mode})",
                data=data
            )
        else:
            errors = result.get("errors", [])
            if not errors:
                errors = ["Error desconegut en la consolidació"]

            # Afegir context
            errors.insert(0, f"Fitxers UIB: {n_uib}, DAD: {n_dad}, Mode: {mode}")
            if masterfile:
                errors.insert(1, f"MasterFile: {os.path.basename(masterfile)}")

            return StepResult(
                success=False, step="consolidar",
                message=f"Error consolidant ({mode})",
                data=data,
                errors=errors
            )

    except Exception as e:
        import traceback
        return StepResult(
            success=False, step="consolidar",
            message=f"Error: {str(e)}",
            errors=[str(e), traceback.format_exc()]
        )


def _step_calibrate(seq_path: str, mode: str, callbacks: PipelineCallbacks,
                    config: Dict) -> StepResult:
    """Pas 2: Calibrar amb KHP."""
    try:
        # Buscar KHP
        khp_files, source = buscar_khp_consolidados(seq_path, allow_manual=False)

        if not khp_files:
            return StepResult(
                success=False, step="calibrar",
                message="No s'han trobat fitxers KHP",
                warnings=["Calibració no disponible"]
            )

        if callbacks.on_step_progress:
            callbacks.on_step_progress("calibrar", 1, 2, f"Analitzant {len(khp_files)} KHP...")

        # Calibrar
        cal_result = calibrate_sequence(seq_path, config=config)

        if callbacks.on_step_progress:
            callbacks.on_step_progress("calibrar", 2, 2, "Registrant calibració...")

        if cal_result.get("success"):
            cal_data = cal_result.get("calibration", {})
            factor = cal_data.get("factor", 0)
            return StepResult(
                success=True, step="calibrar",
                message=f"Factor = {factor:.6f} (font: {source})",
                data={"calibration": cal_data, "source": source}
            )
        else:
            return StepResult(
                success=False, step="calibrar",
                message="Error en calibració",
                errors=cal_result.get("errors", [])
            )

    except Exception as e:
        return StepResult(
            success=False, step="calibrar",
            message=str(e)
        )


def _step_process(seq_path: str, mode: str, callbacks: PipelineCallbacks,
                  config: Dict, calibration: Optional[Dict] = None) -> StepResult:
    """
    Pas 3: Processar mostres (detectar anomalies, triar rèpliques).

    Amb selecció independent DOC/DAD i càlcul de concentracions.
    """
    try:
        # Trobar fitxers consolidats
        con_folder = os.path.join(seq_path, "Resultats_Consolidats")
        if not os.path.isdir(con_folder):
            return StepResult(
                success=False, step="qc",
                message="No s'ha trobat carpeta Resultats_Consolidats"
            )

        con_files = glob.glob(os.path.join(con_folder, "*.xlsx"))
        con_files = [f for f in con_files if "~$" not in f]

        # Agrupar per mostra
        samples_dict = _group_by_sample(con_files)
        total_samples = len(samples_dict)

        if total_samples == 0:
            return StepResult(
                success=False, step="qc",
                message="No s'han trobat fitxers consolidats"
            )

        sample_results = []

        for i, (sample_name, files) in enumerate(samples_dict.items()):
            if callbacks.on_step_progress:
                callbacks.on_step_progress("qc", i + 1, total_samples, sample_name)

            sample_result = _process_sample(sample_name, files, mode, config, calibration)
            sample_results.append(sample_result)

            if callbacks.on_sample_processed:
                callbacks.on_sample_processed(sample_name, sample_result)

        # Estadístiques
        valid_selections = sum(1 for s in sample_results if s.best_replica)

        return StepResult(
            success=True, step="qc",
            message=f"{valid_selections}/{total_samples} mostres amb selecció vàlida",
            data={"samples": sample_results, "total": total_samples}
        )

    except Exception as e:
        return StepResult(
            success=False, step="qc",
            message=str(e)
        )


def _step_export(seq_path: str, samples: List[SampleResult],
                 calibration: Optional[Dict], callbacks: PipelineCallbacks,
                 config: Dict) -> StepResult:
    """Pas 4: Exportar resultats finals."""
    try:
        output_folder = os.path.join(seq_path, "CHECK")
        os.makedirs(output_folder, exist_ok=True)

        total = len(samples) + 2  # samples + summary + (optional) PDF

        # 1. Exportar resum JSON
        if callbacks.on_step_progress:
            callbacks.on_step_progress("exportar", 1, total, "Generant resum...")

        summary_path = os.path.join(output_folder, "processing_summary.json")
        summary_data = _create_export_summary(samples, calibration)
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False, default=str)

        # 2. Exportar Excel amb seleccions
        if callbacks.on_step_progress:
            callbacks.on_step_progress("exportar", 2, total, "Generant Excel...")

        excel_path = os.path.join(output_folder, "replica_selections.xlsx")
        _export_selections_excel(samples, calibration, excel_path)

        # 3. Copiar fitxers de rèpliques seleccionades
        selected_folder = os.path.join(output_folder, "Selected")
        os.makedirs(selected_folder, exist_ok=True)

        for i, sample in enumerate(samples):
            if callbacks.on_step_progress:
                callbacks.on_step_progress("exportar", 3 + i, total, f"Copiant {sample.name}...")

            if sample.best_replica and sample.best_replica in sample.replicas:
                src_file = sample.replicas[sample.best_replica].get("filepath")
                if src_file and os.path.exists(src_file):
                    import shutil
                    dst_file = os.path.join(selected_folder, os.path.basename(src_file))
                    shutil.copy2(src_file, dst_file)

        return StepResult(
            success=True, step="exportar",
            message=f"Exportat a {output_folder}",
            data={
                "output_folder": output_folder,
                "summary_file": summary_path,
                "excel_file": excel_path,
                "excel_generated": os.path.exists(excel_path),
                "pdf_file": None,  # PDF no implementat encara
                "pdf_generated": False,
                "selected_folder": selected_folder,
                "n_selected": sum(1 for s in samples if s.best_replica),
            }
        )

    except Exception as e:
        return StepResult(
            success=False, step="exportar",
            message=str(e)
        )


# =============================================================================
# HELPERS
# =============================================================================

def _detect_existing_mode(seq_path: str):
    """Detecta mode de fitxers consolidats existents."""
    con_folder = os.path.join(seq_path, "Resultats_Consolidats")
    if not os.path.isdir(con_folder):
        return "COLUMN", "UNKNOWN"

    # Llegir primer fitxer
    files = glob.glob(os.path.join(con_folder, "*.xlsx"))
    files = [f for f in files if "~$" not in f]

    if files:
        try:
            xls = pd.ExcelFile(files[0], engine='openpyxl')
            if 'ID' in xls.sheet_names:
                df_id = pd.read_excel(xls, 'ID')
                id_dict = dict(zip(df_id['Camp'], df_id['Valor']))
                mode = id_dict.get('Method', 'COLUMN')
                doc_mode = id_dict.get('DOC_MODE', 'UNKNOWN')
                return mode, doc_mode
        except:
            pass

    return "COLUMN", "UNKNOWN"


def _group_by_sample(files: List[str]) -> Dict[str, List[str]]:
    """Agrupa fitxers per nom de mostra."""
    samples = {}

    for f in files:
        basename = os.path.basename(f)
        # Format: SAMPLE_SEQ_R1.xlsx o SAMPLE_R1.xlsx
        parts = basename.replace('.xlsx', '').split('_')

        # Trobar la part Rx
        sample_parts = []
        for p in parts:
            if p.startswith('R') and p[1:].isdigit():
                break
            sample_parts.append(p)

        sample_name = '_'.join(sample_parts)

        if sample_name not in samples:
            samples[sample_name] = []
        samples[sample_name].append(f)

    return samples


def _process_sample(sample_name: str, files: List[str], mode: str,
                    config: Dict, calibration: Optional[Dict] = None) -> SampleResult:
    """
    Processa una mostra individual amb selecció independent DOC/DAD.

    Args:
        sample_name: Nom de la mostra
        files: Llista de fitxers consolidats (R1, R2, etc.)
        mode: "COLUMN" o "BP"
        config: Configuració
        calibration: Dades de calibració (per calcular concentracions)

    Returns:
        SampleResult amb selecció independent DOC/DAD i camps de reproductibilitat
    """
    replicas = {}
    areas_by_replica = {}  # {R1: {BioP: x, HS: y, ...}, R2: {...}}
    transformation_by_replica = {}  # {R1: {shift_sec, baseline}, R2: {...}}

    # Llegir cada rèplica
    for filepath in files:
        rep_id = _extract_replica_id(filepath)

        try:
            t, y, dad_data = _read_consolidated_file(filepath)

            # Avaluar rèplica DOC
            evaluation = evaluate_replica(t, y, method=mode)
            evaluation['filepath'] = filepath

            # Avaluar DAD si disponible
            if dad_data is not None:
                dad_eval = evaluate_dad(dad_data.get('t', []),
                                        dad_data.get('y_254', []),
                                        wavelength="A254")
                evaluation['dad_eval'] = dad_eval

            # Llegir àrees per fracció del full AREAS
            areas = _read_areas_from_file(filepath)
            if areas:
                areas_by_replica[rep_id] = areas
                evaluation['areas_fraction'] = areas
                evaluation['area'] = areas.get('total', 0)

            # Llegir paràmetres de transformació del full ID
            transform = _read_transformation_from_file(filepath)
            if transform:
                transformation_by_replica[rep_id] = transform
                evaluation['transformation'] = transform

            replicas[rep_id] = evaluation

        except Exception as e:
            replicas[rep_id] = {
                'valid': False,
                'error': str(e),
                'filepath': filepath
            }

    # Selecció independent DOC i DAD
    best = None
    selected_doc = None
    selected_dad = None
    selected = None
    reason = "Cap rèplica vàlida"
    reason_doc = ""
    reason_dad = ""
    confidence = 0.0
    warnings = []
    comparison_data = {}

    if len(replicas) >= 2:
        rep_ids = sorted(replicas.keys())
        eval1 = replicas.get(rep_ids[0], {})
        eval2 = replicas.get(rep_ids[1], {})

        # Comparar si ambdues són vàlides
        if eval1.get('valid') and eval2.get('valid'):
            try:
                t1, y1, _ = _read_consolidated_file(eval1['filepath'])
                t2, y2, _ = _read_consolidated_file(eval2['filepath'])

                comparison = compare_replicas(t1, y1, t2, y2)
                comparison_data = {
                    'pearson': comparison.get('pearson'),
                    'area_diff_pct': comparison.get('area_diff_pct'),
                    'area_r1': comparison.get('area_r1'),
                    'area_r2': comparison.get('area_r2'),
                }

                # Guardar comparació en les rèpliques
                eval1['comparison'] = comparison
                eval2['comparison'] = comparison

                # Selecció DOC
                selection_doc = select_best_replica(
                    eval1, eval2,
                    method=mode,
                    comparison=comparison
                )
                selected_doc = selection_doc.get('best')
                reason_doc = selection_doc.get('reason', '')
                confidence = selection_doc.get('confidence', 0.5)

                if selection_doc.get('warning'):
                    warnings.append(f"DOC: {selection_doc['warning']}")

                # Selecció DAD independent
                dad_eval1 = eval1.get('dad_eval', {})
                dad_eval2 = eval2.get('dad_eval', {})

                if dad_eval1.get('valid', True) and dad_eval2.get('valid', True):
                    # Comparar qualitat DAD
                    drift1 = abs(dad_eval1.get('drift', 0))
                    drift2 = abs(dad_eval2.get('drift', 0))
                    snr_dad1 = dad_eval1.get('snr', 0)
                    snr_dad2 = dad_eval2.get('snr', 0)

                    if dad_eval1.get('quality') == 'POOR' and dad_eval2.get('quality') != 'POOR':
                        selected_dad = rep_ids[1]
                        reason_dad = f"{rep_ids[0]} DAD qualitat baixa"
                    elif dad_eval2.get('quality') == 'POOR' and dad_eval1.get('quality') != 'POOR':
                        selected_dad = rep_ids[0]
                        reason_dad = f"{rep_ids[1]} DAD qualitat baixa"
                    elif drift1 < drift2 * 0.5:  # Drift significativament menor
                        selected_dad = rep_ids[0]
                        reason_dad = f"Menys deriva DAD ({drift1:.2f} vs {drift2:.2f})"
                    elif drift2 < drift1 * 0.5:
                        selected_dad = rep_ids[1]
                        reason_dad = f"Menys deriva DAD ({drift2:.2f} vs {drift1:.2f})"
                    elif snr_dad1 > snr_dad2 * 1.5:  # SNR DAD significativament major
                        selected_dad = rep_ids[0]
                        reason_dad = f"Millor SNR DAD ({snr_dad1:.0f} vs {snr_dad2:.0f})"
                    elif snr_dad2 > snr_dad1 * 1.5:
                        selected_dad = rep_ids[1]
                        reason_dad = f"Millor SNR DAD ({snr_dad2:.0f} vs {snr_dad1:.0f})"
                    else:
                        # Mateixa que DOC
                        selected_dad = selected_doc
                        reason_dad = "Mateixa que DOC"
                else:
                    # Mateixa que DOC si DAD no disponible/vàlid
                    selected_dad = selected_doc
                    reason_dad = "DAD no disponible, segueix DOC"

                # Determinar selected (MIXED si diferent)
                if selected_doc == selected_dad:
                    selected = selected_doc
                    reason = reason_doc
                else:
                    selected = "MIXED"
                    reason = f"DOC={selected_doc}, DAD={selected_dad}"
                    warnings.append(f"Selecció híbrida: DOC de {selected_doc}, DAD de {selected_dad}")

                best = selected_doc  # Legacy: best_replica = selected_doc

            except Exception as e:
                reason = f"Error comparant: {e}"

        elif eval1.get('valid'):
            best = selected_doc = selected_dad = selected = rep_ids[0]
            reason = reason_doc = reason_dad = f"{rep_ids[1]} no vàlida"
            confidence = 0.7
        elif eval2.get('valid'):
            best = selected_doc = selected_dad = selected = rep_ids[1]
            reason = reason_doc = reason_dad = f"{rep_ids[0]} no vàlida"
            confidence = 0.7

    elif len(replicas) == 1:
        rep_id = list(replicas.keys())[0]
        if replicas[rep_id].get('valid'):
            best = selected_doc = selected_dad = selected = rep_id
            reason = reason_doc = reason_dad = "Única rèplica"
            confidence = 0.5
            warnings.append("Només una rèplica disponible")

    # Obtenir àrees i transformació de la rèplica seleccionada (DOC)
    areas_fraction = {}
    transformation = {}
    if selected_doc and selected_doc in areas_by_replica:
        areas_fraction = areas_by_replica[selected_doc]
    if selected_doc and selected_doc in transformation_by_replica:
        transformation = transformation_by_replica[selected_doc]

    # Calcular concentracions si tenim calibració
    conc_fraction = {}
    calibrated_conc = None
    if calibration and areas_fraction:
        factor = calibration.get('factor', 0)
        if factor > 0:
            for frac, area in areas_fraction.items():
                if isinstance(area, (int, float)) and area > 0:
                    conc_fraction[frac] = area * factor
            calibrated_conc = conc_fraction.get('total')

            # Afegir info calibració a transformation
            transformation['factor_used'] = factor
            transformation['khp_source'] = calibration.get('seq_name', calibration.get('khp_source', 'N/A'))
            transformation['khp_area'] = calibration.get('area', 0)
            transformation['khp_conc'] = calibration.get('conc_ppm', 0)

    return SampleResult(
        name=sample_name,
        replicas=replicas,
        best_replica=best,
        selection_reason=reason,
        selection_confidence=confidence,
        warnings=warnings,
        calibrated_conc=calibrated_conc,
        # Selecció independent DOC/DAD
        selected_doc=selected_doc,
        selected_dad=selected_dad,
        selected=selected,
        selection_reason_doc=reason_doc,
        selection_reason_dad=reason_dad,
        # Comparativa
        comparison=comparison_data,
        # Àrees i concentracions
        areas_fraction=areas_fraction,
        conc_fraction=conc_fraction,
        # Transformació
        transformation=transformation,
    )


def _read_areas_from_file(filepath: str) -> Dict[str, float]:
    """Llegeix àrees per fracció del full AREAS d'un fitxer consolidat."""
    try:
        xls = pd.ExcelFile(filepath, engine='openpyxl')
        if 'AREAS' not in xls.sheet_names:
            return {}

        df = pd.read_excel(xls, 'AREAS')

        # Buscar columna de fraccions (pot ser 'Fraction', 'Fracció', etc.)
        frac_col = None
        for col in ['Fraction', 'Fracció', 'Fraccio']:
            if col in df.columns:
                frac_col = col
                break

        if frac_col is None:
            return {}

        # Buscar columna DOC
        doc_col = None
        for col in ['DOC', 'DOC (mAU·min)', 'DOC_Direct']:
            if col in df.columns:
                doc_col = col
                break

        if doc_col is None:
            return {}

        areas = {}
        for _, row in df.iterrows():
            frac = str(row[frac_col]).strip()
            area = row[doc_col]
            # Saltar valors buits o "-"
            if pd.notna(area) and str(area).strip() not in ['-', '', 'nan', 'NaN']:
                try:
                    areas[frac] = float(area)
                except (ValueError, TypeError):
                    pass  # Skip non-numeric values

        # Calcular total si no existeix
        if 'total' not in areas and areas:
            areas['total'] = sum(areas.values())

        return areas

    except Exception:
        return {}


def _read_transformation_from_file(filepath: str) -> Dict[str, Any]:
    """Llegeix paràmetres de transformació del full ID d'un fitxer consolidat."""
    try:
        xls = pd.ExcelFile(filepath, engine='openpyxl')
        if 'ID' not in xls.sheet_names:
            return {}

        df = pd.read_excel(xls, 'ID')

        # Crear dict de camps
        id_dict = {}
        for col_name in ['Field', 'Camp']:
            if col_name in df.columns:
                val_col = 'Value' if 'Value' in df.columns else 'Valor'
                if val_col in df.columns:
                    id_dict = dict(zip(df[col_name], df[val_col]))
                    break

        transform = {}

        # Shift
        for key in ['DOC_Shift_sec', 'Shift_sec', 'DOC_Shift']:
            if key in id_dict and pd.notna(id_dict[key]):
                transform['shift_sec'] = float(id_dict[key])
                break

        # Baseline
        for key in ['DOC_Baseline_mAU', 'Baseline_mAU']:
            if key in id_dict and pd.notna(id_dict[key]):
                transform['baseline_mau'] = float(id_dict[key])
                break

        # Baseline method
        for key in ['DOC_Baseline_Method', 'Baseline_Method']:
            if key in id_dict and pd.notna(id_dict[key]):
                transform['baseline_method'] = str(id_dict[key])
                break

        # DOC mode
        for key in ['DOC_Mode', 'Mode']:
            if key in id_dict and pd.notna(id_dict[key]):
                transform['doc_mode'] = str(id_dict[key])
                break

        # Smoothing
        for key in ['DOC_Smoothing', 'Smoothing']:
            if key in id_dict and pd.notna(id_dict[key]):
                transform['smoothing'] = str(id_dict[key])
                break

        return transform

    except Exception:
        return {}


def _extract_replica_id(filepath: str) -> str:
    """Extreu ID de rèplica del nom de fitxer."""
    basename = os.path.basename(filepath)
    parts = basename.replace('.xlsx', '').split('_')

    for p in parts:
        if p.startswith('R') and len(p) >= 2 and p[1:].isdigit():
            return p

    return "R1"  # Default


def _read_consolidated_file(filepath: str):
    """Llegeix fitxer consolidat i retorna t, y, dad_data."""
    xls = pd.ExcelFile(filepath, engine='openpyxl')

    # Llegir DOC
    df_doc = pd.read_excel(xls, 'DOC')
    t = df_doc['time (min)'].values

    # Buscar columna DOC (pot ser 'DOC (mAU)', 'DOC_Direct (mAU)', etc.)
    y_col = None
    for col in ['DOC (mAU)', 'DOC_Direct (mAU)', 'DOC_UIB (mAU)']:
        if col in df_doc.columns:
            y_col = col
            break

    if y_col is None:
        # Agafar primera columna numèrica després de time
        for col in df_doc.columns[1:]:
            if pd.api.types.is_numeric_dtype(df_doc[col]):
                y_col = col
                break

    y = df_doc[y_col].values if y_col else np.zeros_like(t)

    # Llegir DAD si existeix
    dad_data = None
    if 'DAD' in xls.sheet_names:
        try:
            df_dad = pd.read_excel(xls, 'DAD')
            if 'time (min)' in df_dad.columns and '254' in df_dad.columns:
                dad_data = {
                    't': df_dad['time (min)'].values,
                    'y_254': df_dad['254'].values
                }
        except:
            pass

    return t, y, dad_data


def _create_summary(result: PipelineResult) -> Dict:
    """Crea resum del processament."""
    samples = result.samples

    return {
        "total_samples": len(samples),
        "valid_selections": sum(1 for s in samples if s.best_replica),
        "high_confidence": sum(1 for s in samples if s.selection_confidence > 0.8),
        "with_warnings": sum(1 for s in samples if s.warnings),
        "by_replica": {
            "R1": sum(1 for s in samples if s.best_replica == "R1"),
            "R2": sum(1 for s in samples if s.best_replica == "R2"),
            "None": sum(1 for s in samples if not s.best_replica),
        }
    }


def _create_export_summary(samples: List[SampleResult],
                           calibration: Optional[Dict]) -> Dict:
    """
    Crea dades per exportar a JSON amb tots els camps de reproductibilitat.

    Inclou:
    - Selecció independent DOC/DAD
    - Comparativa entre rèpliques (Pearson, àrea)
    - Àrees per fracció
    - Concentracions calculades
    - Paràmetres de transformació
    """
    return {
        "pipeline_version": PIPELINE_VERSION,
        "pipeline_version_date": PIPELINE_VERSION_DATE,
        "generated_at": datetime.now().isoformat(),
        "calibration": calibration,
        "samples": [
            {
                "name": s.name,
                # Selecció independent DOC/DAD
                "selected_doc": s.selected_doc,
                "selected_dad": s.selected_dad,
                "selected": s.selected,  # R1, R2, MIXED, None
                "selection_reason_doc": s.selection_reason_doc,
                "selection_reason_dad": s.selection_reason_dad,
                # Legacy (per compatibilitat)
                "best_replica": s.best_replica,
                "reason": s.selection_reason,
                "confidence": s.selection_confidence,
                "warnings": s.warnings,
                # Comparativa entre rèpliques
                "comparison": s.comparison,  # {pearson, area_diff_pct, ...}
                # Àrees per fracció (de la rèplica seleccionada)
                "areas_fraction": s.areas_fraction,  # {BioP, HS, BB, SB, LMW, total}
                # Concentracions calculades
                "concentration_ppm": s.calibrated_conc,
                "conc_fraction": s.conc_fraction,  # {BioP, HS, BB, SB, LMW, total}
                # Paràmetres de transformació (per reproductibilitat)
                "transformation": s.transformation,  # {shift_sec, baseline, factor, ...}
                # Info de cada rèplica
                "replicas": {
                    rep_id: {
                        "valid": rep.get("valid"),
                        "snr": rep.get("snr"),
                        "batman": rep.get("batman"),
                        "timeout": rep.get("timeout"),
                        "irr": rep.get("irr"),
                        "area": rep.get("area"),
                        "areas_fraction": rep.get("areas_fraction"),
                        "dad_eval": {
                            "valid": rep.get("dad_eval", {}).get("valid"),
                            "quality": rep.get("dad_eval", {}).get("quality"),
                            "drift": rep.get("dad_eval", {}).get("drift"),
                            "snr": rep.get("dad_eval", {}).get("snr"),
                        } if rep.get("dad_eval") else None,
                    }
                    for rep_id, rep in s.replicas.items()
                }
            }
            for s in samples
        ]
    }


def _export_selections_excel(samples: List[SampleResult],
                             calibration: Optional[Dict],
                             filepath: str):
    """Exporta seleccions a Excel."""
    rows = []

    for s in samples:
        row = {
            "Mostra": s.name,
            "Selecció": s.best_replica or "-",
            "Motiu": s.selection_reason,
            "Confiança": f"{s.selection_confidence:.0%}",
            "Avisos": "; ".join(s.warnings) if s.warnings else "-",
        }

        # Afegir info de cada rèplica
        for rep_id in ["R1", "R2"]:
            if rep_id in s.replicas:
                rep = s.replicas[rep_id]
                row[f"{rep_id}_Valid"] = "✓" if rep.get("valid") else "✗"
                row[f"{rep_id}_SNR"] = f"{rep.get('snr', 0):.1f}" if rep.get('snr') else "-"
                row[f"{rep_id}_Batman"] = "⚠" if rep.get("batman") else "-"

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_excel(filepath, index=False, engine='openpyxl')


# =============================================================================
# FUNCIONS ADDICIONALS
# =============================================================================

def import_old_sequence(
    seq_path: str,
    callbacks: Optional[PipelineCallbacks] = None,
    force_overwrite: bool = False,
    config: Optional[Dict] = None
) -> PipelineResult:
    """
    Importa una seqüència antiga (pre-MasterFile) amb pipeline complet.

    Força la migració i executa tot el pipeline.
    Equivalent a process_sequence amb skip_migration=False i force=True per migrar.

    Args:
        seq_path: Ruta a la carpeta SEQ antiga
        callbacks: Callbacks per reportar progrés
        force_overwrite: Sobreescriure fitxers existents (inclou regenerar MasterFile)
        config: Configuració personalitzada

    Returns:
        PipelineResult amb tots els resultats
    """
    return process_sequence(
        seq_path,
        callbacks=callbacks,
        skip_consolidation=False,
        skip_calibration=False,
        skip_migration=False,  # Sempre intenta migrar
        force_overwrite=force_overwrite,
        config=config
    )


def check_sequence_status(seq_path: str) -> Dict[str, Any]:
    """
    Comprova l'estat d'una seqüència (què té i què li falta).

    Returns:
        Dict amb:
        - has_masterfile: bool
        - has_rawdata: bool
        - has_consolidation: bool
        - has_calibration: bool
        - has_results: bool
        - needs_migration: bool
        - needs_reprocess: bool (True si versió antiga)
        - processed_version: str o None
        - current_version: str
        - status: str ('ready', 'needs_migration', 'needs_processing', 'complete', 'outdated')
        - details: dict amb detalls addicionals
    """
    status = {
        'has_masterfile': False,
        'has_rawdata': False,
        'has_consolidation': False,
        'has_calibration': False,
        'has_results': False,
        'needs_migration': False,
        'needs_reprocess': False,
        'processed_version': None,
        'current_version': PIPELINE_VERSION,
        'status': 'unknown',
        'details': {}
    }

    # Check MasterFile
    masterfiles = glob.glob(os.path.join(seq_path, '*_MasterFile.xlsx'))
    masterfiles = [f for f in masterfiles if 'backup' not in f.lower() and '~$' not in f]
    status['has_masterfile'] = len(masterfiles) > 0
    if masterfiles:
        status['details']['masterfile'] = os.path.basename(masterfiles[0])

    # Check rawdata
    rawdata_patterns = ['*_v11*.xlsx', '*_v12*.xlsx', '*RAWDATA*.xlsx']
    for pattern in rawdata_patterns:
        matches = glob.glob(os.path.join(seq_path, pattern))
        matches = [m for m in matches if 'backup' not in m.lower() and '~$' not in m]
        if matches:
            status['has_rawdata'] = True
            status['details']['rawdata'] = os.path.basename(matches[0])
            break

    # Check consolidation
    con_folder = os.path.join(seq_path, "Resultats_Consolidats")
    if os.path.isdir(con_folder):
        con_files = glob.glob(os.path.join(con_folder, "*.xlsx"))
        con_files = [f for f in con_files if '~$' not in f]
        status['has_consolidation'] = len(con_files) > 0
        status['details']['consolidated_files'] = len(con_files)

    # Check calibration
    cal_file = os.path.join(seq_path, "CHECK", "calibration.json")
    status['has_calibration'] = os.path.exists(cal_file)

    # Check results and version
    # has_results = True NOMÉS si hi ha processing_summary.json AMB mostres processades
    results_file = os.path.join(seq_path, "CHECK", "processing_summary.json")
    status['has_results'] = False  # Per defecte, no processat

    if os.path.exists(results_file):
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                results_data = json.load(f)

            # Comprovar si realment té mostres processades
            samples = results_data.get('samples', [])
            if samples and len(samples) > 0:
                status['has_results'] = True
                status['processed_version'] = results_data.get('pipeline_version', '0.0.0')
                status['details']['processed_at'] = results_data.get('generated_at', 'unknown')
                status['details']['n_samples_processed'] = len(samples)

                # Comparar versions
                if _version_is_older(status['processed_version'], PIPELINE_VERSION):
                    status['needs_reprocess'] = True
                    status['details']['version_info'] = (
                        f"Processat amb v{status['processed_version']}, "
                        f"actual v{PIPELINE_VERSION}"
                    )
            else:
                # Fitxer existeix però sense mostres - procés incomplet
                status['details']['partial_processing'] = True
        except:
            # Error llegint - considerar com no processat
            status['details']['processing_error'] = True

    # Determine overall status
    status['needs_migration'] = not status['has_masterfile'] and status['has_rawdata']

    if status['has_results'] and status['needs_reprocess']:
        status['status'] = 'outdated'
    elif status['has_results']:
        status['status'] = 'complete'
    elif status['needs_migration']:
        status['status'] = 'needs_migration'
    elif status['has_masterfile'] or status['has_consolidation']:
        status['status'] = 'needs_processing'
    elif status['has_rawdata']:
        status['status'] = 'needs_migration'
    else:
        status['status'] = 'no_data'

    return status


def _version_is_older(old_version: str, new_version: str) -> bool:
    """Compara dues versions (format X.Y.Z). Retorna True si old < new."""
    try:
        def parse_version(v):
            # Gestionar versions amb format diferent
            if v is None or v == 'unknown':
                return (0, 0, 0)
            parts = str(v).split('.')
            return tuple(int(p) for p in parts[:3])

        old = parse_version(old_version)
        new = parse_version(new_version)
        return old < new
    except:
        return True  # En cas de dubte, reprocessar


def find_latest_seq(base_path: str) -> Optional[str]:
    """
    Troba la carpeta SEQ més recent (per data de modificació).

    Args:
        base_path: Ruta base (ex: Dades2)

    Returns:
        Ruta completa a la SEQ més recent o None
    """
    folders = find_seq_folders(base_path)
    if not folders:
        return None

    # Ordenar per data de modificació (més recent primer)
    folders_with_mtime = []
    for f in folders:
        try:
            mtime = os.path.getmtime(f)
            folders_with_mtime.append((f, mtime))
        except:
            pass

    if not folders_with_mtime:
        return None

    folders_with_mtime.sort(key=lambda x: x[1], reverse=True)
    return folders_with_mtime[0][0]


def get_all_sequences_status(base_path: str) -> List[Dict[str, Any]]:
    """
    Obté l'estat de totes les seqüències en un repositori.

    Args:
        base_path: Ruta base (ex: Dades2)

    Returns:
        Llista de dicts amb info de cada SEQ:
        - path, name, status, has_masterfile, has_consolidation,
          has_calibration, has_results, needs_reprocess, processed_version
    """
    folders = find_seq_folders(base_path)
    results = []

    for folder in folders:
        status = check_sequence_status(folder)
        results.append({
            'path': folder,
            'name': os.path.basename(folder),
            **status
        })

    return results


def get_pending_sequences(base_path: str) -> Dict[str, List[str]]:
    """
    Troba totes les seqüències i les classifica per estat.

    Args:
        base_path: Ruta base (ex: Dades2)

    Returns:
        Dict amb llistes de paths per cada estat:
        - needs_migration: SEQs que necessiten migrar
        - needs_processing: SEQs que necessiten processar
        - complete: SEQs completament processades
        - no_data: SEQs sense dades reconegudes
    """
    folders = find_seq_folders(base_path)

    result = {
        'needs_migration': [],
        'needs_processing': [],
        'complete': [],
        'no_data': []
    }

    for folder in folders:
        status = check_sequence_status(folder)
        result[status['status']].append(folder)

    return result


# =============================================================================
# FUNCIONS PÚBLIQUES PER EXECUCIÓ PAS A PAS
# =============================================================================

def run_migration_step(
    seq_path: str,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
    force_overwrite: bool = False
) -> StepResult:
    """
    Executa l'etapa de migració de forma independent.

    Args:
        seq_path: Ruta a la carpeta SEQ
        on_progress: Callback per reportar progrés (current, total, item)
        force_overwrite: Sobreescriure si existeix

    Returns:
        StepResult amb el resultat de la migració
    """
    try:
        callbacks = PipelineCallbacks()
        if on_progress:
            callbacks.on_step_progress = lambda step, cur, tot, item: on_progress(cur, tot, item)

        return _step_migrate(seq_path, callbacks, force_overwrite)
    except Exception as e:
        import traceback
        return StepResult(
            success=False,
            step="migrar",
            message=f"Error en migració: {str(e)}",
            errors=[str(e), traceback.format_exc()]
        )


def run_consolidation_step(
    seq_path: str,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
    force_overwrite: bool = False
) -> StepResult:
    """
    Executa l'etapa de consolidació de forma independent.

    Args:
        seq_path: Ruta a la carpeta SEQ
        on_progress: Callback per reportar progrés
        force_overwrite: Sobreescriure si existeix

    Returns:
        StepResult amb el resultat de la consolidació
    """
    try:
        callbacks = PipelineCallbacks()
        if on_progress:
            callbacks.on_step_progress = lambda step, cur, tot, item: on_progress(cur, tot, item)

        config = get_config().config
        return _step_consolidate(seq_path, callbacks, force_overwrite, config)
    except Exception as e:
        import traceback
        return StepResult(
            success=False,
            step="consolidar",
            message=f"Error en consolidació: {str(e)}",
            errors=[str(e), traceback.format_exc()]
        )


def run_calibration_step(
    seq_path: str,
    consolidation_file: Optional[str] = None,
    on_progress: Optional[Callable[[int, int, str], None]] = None
) -> StepResult:
    """
    Executa l'etapa de calibratge de forma independent.

    Args:
        seq_path: Ruta a la carpeta SEQ
        consolidation_file: Ruta al fitxer de consolidació (opcional)
        on_progress: Callback per reportar progrés

    Returns:
        StepResult amb el resultat del calibratge
    """
    try:
        callbacks = PipelineCallbacks()
        if on_progress:
            callbacks.on_step_progress = lambda step, cur, tot, item: on_progress(cur, tot, item)

        # Detectar mode
        mode, _ = _detect_existing_mode(seq_path)
        config = get_config().config

        return _step_calibrate(seq_path, mode, callbacks, config)
    except Exception as e:
        import traceback
        return StepResult(
            success=False,
            step="calibrar",
            message=f"Error en calibratge: {str(e)}",
            errors=[str(e), traceback.format_exc()]
        )


def run_processing_step(
    seq_path: str,
    consolidation_file: Optional[str] = None,
    correction_factor: Optional[float] = None,
    on_progress: Optional[Callable[[int, int, str], None]] = None
) -> StepResult:
    """
    Executa l'etapa de processament de forma independent.

    Args:
        seq_path: Ruta a la carpeta SEQ
        consolidation_file: Ruta al fitxer de consolidació (opcional)
        correction_factor: Factor de correcció del calibratge (opcional)
        on_progress: Callback per reportar progrés

    Returns:
        StepResult amb el resultat del processament
    """
    try:
        callbacks = PipelineCallbacks()
        if on_progress:
            callbacks.on_step_progress = lambda step, cur, tot, item: on_progress(cur, tot, item)

        # Detectar mode
        mode, _ = _detect_existing_mode(seq_path)
        config = get_config().config

        # Carregar calibració si existeix
        calibration = get_active_calibration(seq_path, mode)

        result = _step_process(seq_path, mode, callbacks, config, calibration)

        # Afegir detalls extra per la GUI
        if result.success and result.data.get('samples'):
            samples = result.data['samples']
            total = len(samples)
            valid = sum(1 for s in samples if s.best_replica)
            high_conf = sum(1 for s in samples if s.selection_confidence >= 0.8)
            anomalies = sum(1 for s in samples if s.warnings)

            result.data['total_samples'] = total
            result.data['valid_selections'] = valid
            result.data['high_confidence'] = high_conf
            result.data['anomalies_detected'] = anomalies

            # Detall per mostra (format per GUI QC table)
            sample_details = []
            for s in samples:
                # Determinar estat
                if s.best_replica and s.selection_confidence >= 0.8:
                    status = "OK"
                elif s.best_replica:
                    status = "CHECK"
                else:
                    status = "FAIL"

                # Extreure comparativa (Pearson, diff àrea)
                pearson_r = s.comparison.get('pearson') if s.comparison else None
                area_diff_pct = s.comparison.get('area_diff_pct') if s.comparison else None

                # Qualitat DAD - buscar en la rèplica seleccionada per DAD
                dad_quality = "?"
                if s.selected_dad and s.selected_dad in s.replicas:
                    rep_data = s.replicas[s.selected_dad]
                    if isinstance(rep_data, dict) and rep_data.get('dad_eval'):
                        dad_quality = rep_data['dad_eval'].get('quality', '?')

                sample_details.append({
                    'name': s.name,
                    'status': status,
                    'valid': s.best_replica is not None,
                    'pearson_r': pearson_r,
                    'area_diff_pct': area_diff_pct,
                    # Selecció independent DOC/DAD
                    'selected_doc': s.selected_doc,
                    'selected_dad': s.selected_dad,
                    'selected': s.selected,  # R1, R2, MIXED
                    'selection_reason_doc': s.selection_reason_doc,
                    'selection_reason_dad': s.selection_reason_dad,
                    'dad_quality': dad_quality,
                    'reason': s.selection_reason,
                    'confidence': s.selection_confidence,
                    'warnings': s.warnings,
                    # Àrees i concentracions
                    'areas_fraction': s.areas_fraction,
                    'concentration_ppm': s.calibrated_conc,
                    'conc_fraction': s.conc_fraction,
                })
            result.data['sample_details'] = sample_details
            # Also store as 'samples' for GUI compatibility
            result.data['samples'] = sample_details

        return result
    except Exception as e:
        import traceback
        return StepResult(
            success=False,
            step="qc",
            message=f"Error en processament: {str(e)}",
            errors=[str(e), traceback.format_exc()]
        )


def run_export_step(
    seq_path: str,
    processing_result: Optional[StepResult] = None,
    on_progress: Optional[Callable[[int, int, str], None]] = None
) -> StepResult:
    """
    Executa l'etapa d'exportació de forma independent.

    Args:
        seq_path: Ruta a la carpeta SEQ
        processing_result: Resultat del processament (opcional)
        on_progress: Callback per reportar progrés

    Returns:
        StepResult amb el resultat de l'exportació
    """
    try:
        callbacks = PipelineCallbacks()
        if on_progress:
            callbacks.on_step_progress = lambda step, cur, tot, item: on_progress(cur, tot, item)

        config = get_config().config

        # Obtenir mostres del resultat del processament
        samples = []
        if processing_result and processing_result.data.get('samples'):
            samples = processing_result.data['samples']

        # Obtenir calibració si existeix
        mode, _ = _detect_existing_mode(seq_path)
        calibration = get_active_calibration(seq_path, mode)

        return _step_export(seq_path, samples, calibration, callbacks, config)
    except Exception as e:
        import traceback
        return StepResult(
            success=False,
            step="exportar",
            message=f"Error en exportació: {str(e)}",
            errors=[str(e), traceback.format_exc()]
        )


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("HPSEC Pipeline - Test")
    print("=" * 60)

    # Test status check
    print("\n1. Comprovar estat d'una seqüència")
    test_path = input("Introdueix ruta SEQ (o Enter per ometre): ").strip()

    if test_path and os.path.isdir(test_path):
        # Comprovar estat
        status = check_sequence_status(test_path)
        print(f"\nEstat de {os.path.basename(test_path)}:")
        print(f"  MasterFile: {'✓' if status['has_masterfile'] else '✗'}")
        print(f"  Rawdata: {'✓' if status['has_rawdata'] else '✗'}")
        print(f"  Consolidat: {'✓' if status['has_consolidation'] else '✗'}")
        print(f"  Calibrat: {'✓' if status['has_calibration'] else '✗'}")
        print(f"  Resultats: {'✓' if status['has_results'] else '✗'}")
        print(f"  Status: {status['status']}")
        print(f"  Necessita migració: {'SÍ' if status['needs_migration'] else 'No'}")

        # Preguntar si processar
        if status['status'] != 'complete':
            proceed = input("\nProcessar aquesta seqüència? (s/n): ").strip().lower()

            if proceed == 's':
                def on_step_start(step, current, total):
                    print(f"\n[{current}/{total}] Iniciant {step.upper()}...")

                def on_progress(step, current, total, item):
                    print(f"  [{step}] {current}/{total}: {item}")

                def on_step_complete(step, result):
                    status_icon = "✓" if result.success else "✗"
                    print(f"  {status_icon} {result.message}")

                callbacks = PipelineCallbacks(
                    on_step_start=on_step_start,
                    on_step_progress=on_progress,
                    on_step_complete=on_step_complete
                )

                result = process_sequence(test_path, callbacks)

                print(f"\n{'=' * 60}")
                print(f"RESULTAT FINAL: {'OK' if result.success else 'ERROR'}")
                print(f"Mode: {result.mode} / {result.doc_mode}")
                print(f"Duració: {result.duration_sec:.1f}s")
                print(f"Mostres processades: {len(result.samples)}")

                if result.migration and result.migration.success and result.migration.data.get('masterfile'):
                    print(f"Migració: {result.migration.message}")

                if result.summary:
                    print(f"\nResum seleccions:")
                    for key, val in result.summary.items():
                        print(f"  {key}: {val}")
        else:
            print("\nLa seqüència ja està completament processada.")

    else:
        # Mostrar seqüències pendents
        base_path = 'C:/Users/Lequia/Desktop/Dades2'
        if os.path.isdir(base_path):
            print(f"\nBuscant seqüències a {base_path}...")
            pending = get_pending_sequences(base_path)

            print(f"\nResum:")
            print(f"  Necessiten migració: {len(pending['needs_migration'])}")
            print(f"  Necessiten processar: {len(pending['needs_processing'])}")
            print(f"  Completes: {len(pending['complete'])}")
            print(f"  Sense dades: {len(pending['no_data'])}")

            if pending['needs_migration']:
                print(f"\nPrimeres 5 SEQs que necessiten migració:")
                for p in pending['needs_migration'][:5]:
                    print(f"  - {os.path.basename(p)}")
