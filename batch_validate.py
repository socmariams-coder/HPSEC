"""
HPSEC Suite - Batch Validation Tool
====================================

Eina per processar TOTES les seqüències i generar informes de validació.
Permet identificar errors i problemes de forma sistemàtica.

Ús:
    python batch_validate.py D:/HPSEC/SEQS --stages import,calibrate
    python batch_validate.py D:/HPSEC/SEQS --stages import
    python batch_validate.py D:/HPSEC/SEQS --report-only
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import traceback

# Afegir path del projecte
sys.path.insert(0, str(Path(__file__).parent))

from hpsec_import import import_sequence
from hpsec_calibrate import calibrate_from_import, load_khp_history


class BatchValidator:
    """Validador batch per a totes les seqüències."""

    def __init__(self, base_path, output_dir=None):
        self.base_path = Path(base_path)
        self.output_dir = Path(output_dir) if output_dir else self.base_path / "BATCH_REPORTS"
        self.output_dir.mkdir(exist_ok=True)

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Resultats
        self.import_results = []
        self.calibration_results = []
        self.errors = []

        # Estadístiques
        self.stats = {
            'import': defaultdict(int),
            'calibrate': defaultdict(int),
        }

    def find_sequences(self, pattern="*_SEQ"):
        """Troba totes les carpetes SEQ."""
        seqs = []
        for item in self.base_path.iterdir():
            if item.is_dir() and "_SEQ" in item.name:
                seqs.append(item)

        # Ordenar per número de SEQ
        def get_seq_num(p):
            import re
            match = re.search(r'(\d+)', p.name)
            return int(match.group(1)) if match else 0

        seqs.sort(key=get_seq_num)
        return seqs

    def run_import(self, seq_path, save_json=True):
        """Executa import per una seqüència."""
        seq_name = seq_path.name
        result = {
            'seq_name': seq_name,
            'seq_path': str(seq_path),
            'success': False,
            'errors': [],
            'warnings': [],
            'metrics': {},
            'timestamp': datetime.now().isoformat(),
        }

        try:
            # Executar import
            import_data = import_sequence(str(seq_path))

            if import_data:
                result['success'] = True
                result['metrics'] = {
                    'n_samples': len(import_data.get('samples', [])),
                    'n_khp': len(import_data.get('khp_files', [])),
                    'n_blanks': len(import_data.get('blanks', [])),
                    'method': import_data.get('method', 'UNKNOWN'),
                    'doc_mode': import_data.get('doc_mode', 'UNKNOWN'),
                    'has_consolidates': import_data.get('has_consolidates', False),
                }

                # Extreure warnings i errors de import_data
                result['warnings'] = import_data.get('warnings', [])
                result['errors'] = import_data.get('errors', [])
                if import_data.get('errors'):
                    result['success'] = False

                # Guardar JSON a CHECK/data
                if save_json:
                    check_dir = seq_path / "CHECK" / "data"
                    check_dir.mkdir(parents=True, exist_ok=True)
                    json_path = check_dir / "import_result.json"

                    # Netejar dades no serialitzables
                    clean_data = self._clean_for_json(import_data)
                    clean_data['_validation'] = result

                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(clean_data, f, indent=2, ensure_ascii=False, default=str)

                    result['json_path'] = str(json_path)

                # Guardar import_data per calibració posterior
                result['_import_data'] = import_data

            else:
                result['errors'].append("Import retorna None")

        except Exception as e:
            result['errors'].append(f"EXCEPTION: {str(e)}")
            result['traceback'] = traceback.format_exc()
            self.errors.append({
                'stage': 'import',
                'seq_name': seq_name,
                'error': str(e),
                'traceback': traceback.format_exc(),
            })

        # Estadístiques
        self.stats['import']['total'] += 1
        if result['success']:
            self.stats['import']['success'] += 1
        else:
            self.stats['import']['failed'] += 1
        if result['warnings']:
            self.stats['import']['with_warnings'] += 1

        self.import_results.append(result)
        return result

    def run_calibration(self, seq_path, import_data=None, save_json=True):
        """Executa calibració per una seqüència."""
        seq_name = seq_path.name
        result = {
            'seq_name': seq_name,
            'seq_path': str(seq_path),
            'success': False,
            'errors': [],
            'warnings': [],
            'metrics': {},
            'timestamp': datetime.now().isoformat(),
        }

        try:
            # Si no tenim import_data, intentar importar
            if import_data is None:
                import_result = self.run_import(seq_path, save_json=save_json)
                if not import_result['success']:
                    result['errors'].append("Import fallit - no es pot calibrar")
                    result['errors'].extend(import_result['errors'])
                    return result
                import_data = import_result.get('_import_data')

            if not import_data:
                result['errors'].append("No hi ha dades d'import")
                return result

            # Executar calibració
            cal_result = calibrate_from_import(import_data)

            if cal_result:
                result['success'] = cal_result.get('success', False)
                result['errors'] = cal_result.get('errors', [])

                # Mètriques
                result['metrics'] = {
                    'mode': cal_result.get('mode', 'UNKNOWN'),
                    'factor_direct': cal_result.get('factor_direct', 0),
                    'factor_uib': cal_result.get('factor_uib', 0),
                    'shift_uib': cal_result.get('shift_uib', 0),
                    'khp_area': cal_result.get('khp_area_direct', 0) or cal_result.get('khp_area', 0),
                    'khp_conc': cal_result.get('khp_conc', 0),
                    'khp_source': cal_result.get('khp_source', 'UNKNOWN'),
                }

                # Extreure mètriques KHP detallades (des de replicas)
                khp_data = cal_result.get('khp_data_direct') or cal_result.get('khp_data')
                if khp_data:
                    replicas = khp_data.get('replicas', [])

                    # Agregar mètriques de totes les rèpliques
                    all_t_max = []
                    all_snr = []
                    all_ratio = []
                    all_cr = []
                    all_quality = []
                    all_height = []
                    all_symmetry = []
                    has_batman = False
                    has_timeout = False
                    all_issues = []

                    for rep in replicas:
                        t_ret = rep.get('t_retention', 0)
                        if t_ret and t_ret > 0:
                            all_t_max.append(t_ret)

                        snr = rep.get('snr', 0)
                        if snr and snr > 0:
                            all_snr.append(snr)

                        ratio = rep.get('a254_doc_ratio', 0)
                        if ratio and ratio > 0:
                            all_ratio.append(ratio)

                        cr = rep.get('concentration_ratio', 0)
                        if cr and cr > 0:
                            all_cr.append(cr)

                        q = rep.get('quality_score', 0)
                        if q is not None:
                            all_quality.append(q)

                        h = rep.get('height', 0)
                        if h and h > 0:
                            all_height.append(h)

                        sym = rep.get('symmetry', 0)
                        if sym and sym > 0:
                            all_symmetry.append(sym)

                        if rep.get('has_batman'):
                            has_batman = True
                        if rep.get('has_timeout'):
                            has_timeout = True

                        all_issues.extend(rep.get('quality_issues', []))

                    # Calcular mitjanes/estadístiques
                    import numpy as np
                    result['metrics'].update({
                        't_max': float(np.mean(all_t_max)) if all_t_max else 0,
                        't_max_std': float(np.std(all_t_max)) if len(all_t_max) > 1 else 0,
                        'snr': float(np.mean(all_snr)) if all_snr else 0,
                        'snr_min': float(np.min(all_snr)) if all_snr else 0,
                        'a254_doc_ratio': float(np.mean(all_ratio)) if all_ratio else 0,
                        'a254_doc_ratio_std': float(np.std(all_ratio)) if len(all_ratio) > 1 else 0,
                        'concentration_ratio': float(np.mean(all_cr)) if all_cr else 0,
                        'quality_score': int(np.max(all_quality)) if all_quality else 0,  # Pitjor cas
                        'quality_score_mean': float(np.mean(all_quality)) if all_quality else 0,
                        'height': float(np.mean(all_height)) if all_height else 0,
                        'symmetry': float(np.mean(all_symmetry)) if all_symmetry else 0,
                        'has_batman': has_batman,
                        'has_timeout': has_timeout,
                        'n_replicas': len(replicas),
                        'valid_for_calibration': khp_data.get('valid_for_calibration', True),
                    })

                    # Timeout details
                    for rep in replicas:
                        timeout_info = rep.get('timeout_info', {})
                        if timeout_info.get('timeouts'):
                            first_to = timeout_info['timeouts'][0]
                            result['metrics']['timeout_time'] = first_to.get('t_start_min', 0)
                            result['metrics']['timeout_severity'] = rep.get('timeout_severity', 'OK')
                            break

                    # Warnings de qualitat (únics)
                    unique_issues = list(set(all_issues))
                    result['warnings'].extend(unique_issues)

                # Guardar JSON a CHECK/data
                if save_json:
                    check_dir = seq_path / "CHECK" / "data"
                    check_dir.mkdir(parents=True, exist_ok=True)
                    json_path = check_dir / "calibration_result.json"

                    clean_result = self._clean_for_json(cal_result)
                    clean_result['_validation'] = {
                        'success': result['success'],
                        'errors': result['errors'],
                        'warnings': result['warnings'],
                        'metrics': result['metrics'],
                    }

                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(clean_result, f, indent=2, ensure_ascii=False, default=str)

                    result['json_path'] = str(json_path)

            else:
                result['errors'].append("Calibració retorna None")

        except Exception as e:
            result['errors'].append(f"EXCEPTION: {str(e)}")
            result['traceback'] = traceback.format_exc()
            self.errors.append({
                'stage': 'calibrate',
                'seq_name': seq_name,
                'error': str(e),
                'traceback': traceback.format_exc(),
            })

        # Estadístiques
        self.stats['calibrate']['total'] += 1
        if result['success']:
            self.stats['calibrate']['success'] += 1
        else:
            self.stats['calibrate']['failed'] += 1
        if result['warnings']:
            self.stats['calibrate']['with_warnings'] += 1

        # Comptar problemes específics
        for w in result['warnings']:
            if 'BATMAN' in w.upper():
                self.stats['calibrate']['batman'] += 1
            if 'TIMEOUT' in w.upper():
                self.stats['calibrate']['timeout'] += 1
            if 'SNR' in w.upper():
                self.stats['calibrate']['low_snr'] += 1

        self.calibration_results.append(result)
        return result

    def run_batch(self, stages=['import', 'calibrate'], limit=None):
        """Executa batch per totes les seqüències."""
        seqs = self.find_sequences()
        if limit:
            seqs = seqs[:limit]

        print(f"\n{'='*60}")
        print(f"BATCH VALIDATION - {len(seqs)} seqüències")
        print(f"Stages: {', '.join(stages)}")
        print(f"Output: {self.output_dir}")
        print(f"{'='*60}\n")

        for i, seq_path in enumerate(seqs):
            seq_name = seq_path.name
            print(f"[{i+1}/{len(seqs)}] {seq_name}...", end=" ", flush=True)

            import_data = None

            if 'import' in stages:
                import_result = self.run_import(seq_path)
                status_import = "OK" if import_result['success'] else "FAIL"
                warns_import = f"({len(import_result['warnings'])}w)" if import_result['warnings'] else ""
                print(f"Import:{status_import}{warns_import}", end=" ", flush=True)
                import_data = import_result.get('_import_data')

            if 'calibrate' in stages:
                cal_result = self.run_calibration(seq_path, import_data)
                status_cal = "OK" if cal_result['success'] else "FAIL"
                warns_cal = f"({len(cal_result['warnings'])}w)" if cal_result['warnings'] else ""
                print(f"Cal:{status_cal}{warns_cal}", end="", flush=True)

            print()

        print(f"\n{'='*60}")
        print("BATCH COMPLETAT")
        print(f"{'='*60}")

        # Generar informes
        self.generate_reports()

    def generate_reports(self):
        """Genera informes de resum."""
        # 1. Informe JSON complet
        full_report = {
            'timestamp': self.timestamp,
            'base_path': str(self.base_path),
            'stats': dict(self.stats),
            'import_results': self.import_results,
            'calibration_results': self.calibration_results,
            'errors': self.errors,
        }

        json_path = self.output_dir / f"batch_report_{self.timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(full_report, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nJSON report: {json_path}")

        # 2. Informe TXT resum
        txt_path = self.output_dir / f"batch_summary_{self.timestamp}.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            self._write_summary_report(f)
        print(f"Summary report: {txt_path}")

        # 3. CSV amb mètriques
        csv_path = self.output_dir / f"batch_metrics_{self.timestamp}.csv"
        self._write_csv_report(csv_path)
        print(f"CSV metrics: {csv_path}")

        # 4. Llista d'errors
        if self.errors:
            errors_path = self.output_dir / f"batch_errors_{self.timestamp}.txt"
            with open(errors_path, 'w', encoding='utf-8') as f:
                for err in self.errors:
                    f.write(f"\n{'='*60}\n")
                    f.write(f"Stage: {err['stage']}\n")
                    f.write(f"SEQ: {err['seq_name']}\n")
                    f.write(f"Error: {err['error']}\n")
                    if err.get('traceback'):
                        f.write(f"\nTraceback:\n{err['traceback']}\n")
            print(f"Errors log: {errors_path}")

    def _write_summary_report(self, f):
        """Escriu l'informe de resum amb warnings categoritzats."""
        f.write(f"HPSEC BATCH VALIDATION REPORT\n")
        f.write(f"{'='*60}\n")
        f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Base path: {self.base_path}\n\n")

        # === IMPORT ===
        f.write(f"\n{'='*60}\n")
        f.write(f"IMPORT SUMMARY\n")
        f.write(f"{'='*60}\n\n")

        stats_i = self.stats['import']
        f.write(f"Total: {stats_i['total']}\n")
        f.write(f"  OK Success: {stats_i['success']}\n")
        f.write(f"  FAIL Failed: {stats_i['failed']}\n")
        f.write(f"  WARN With warnings: {stats_i['with_warnings']}\n\n")

        # Categoritzar warnings d'import
        import_warn_categories = defaultdict(list)
        for r in self.import_results:
            for w in r.get('warnings', []):
                w_lower = w.lower()
                if 'orfe' in w_lower or 'orphan' in w_lower:
                    import_warn_categories['ORPHAN_FILES'].append((r['seq_name'], w))
                elif 'duplica' in w_lower:
                    import_warn_categories['DUPLICATES'].append((r['seq_name'], w))
                elif 'suggerit' in w_lower or 'matching' in w_lower or 'revisar' in w_lower:
                    import_warn_categories['MATCHING_REVIEW'].append((r['seq_name'], w))
                elif 'masterfile' in w_lower or 'creat des de' in w_lower:
                    import_warn_categories['MASTERFILE_AUTO'].append((r['seq_name'], w))
                else:
                    import_warn_categories['OTHER'].append((r['seq_name'], w))

        # Llista de fallits
        failed_imports = [r for r in self.import_results if not r['success']]
        if failed_imports:
            f.write(f"FAILED IMPORTS ({len(failed_imports)}):\n")
            for r in failed_imports:
                f.write(f"  - {r['seq_name']}: {'; '.join(r['errors'][:3])}\n")
            f.write("\n")

        # Warnings per categoria
        if import_warn_categories:
            f.write(f"IMPORT WARNINGS BY CATEGORY:\n\n")

            if import_warn_categories['ORPHAN_FILES']:
                f.write(f"  [ORPHAN] Fitxers orfes ({len(import_warn_categories['ORPHAN_FILES'])}):\n")
                for seq, w in import_warn_categories['ORPHAN_FILES'][:10]:
                    f.write(f"    - {seq}: {w[:60]}\n")
                if len(import_warn_categories['ORPHAN_FILES']) > 10:
                    f.write(f"    ... i {len(import_warn_categories['ORPHAN_FILES'])-10} més\n")
                f.write("\n")

            if import_warn_categories['DUPLICATES']:
                f.write(f"  [DUP] Duplicats detectats ({len(import_warn_categories['DUPLICATES'])}):\n")
                for seq, w in import_warn_categories['DUPLICATES']:
                    f.write(f"    - {seq}: {w[:60]}\n")
                f.write("\n")

            if import_warn_categories['MATCHING_REVIEW']:
                f.write(f"  [MATCH] Matching automàtic per revisar ({len(import_warn_categories['MATCHING_REVIEW'])}):\n")
                for seq, w in import_warn_categories['MATCHING_REVIEW'][:10]:
                    f.write(f"    - {seq}: {w[:60]}\n")
                if len(import_warn_categories['MATCHING_REVIEW']) > 10:
                    f.write(f"    ... i {len(import_warn_categories['MATCHING_REVIEW'])-10} més\n")
                f.write("\n")

            if import_warn_categories['MASTERFILE_AUTO']:
                f.write(f"  [MFAUTO] MasterFile generat automàticament ({len(import_warn_categories['MASTERFILE_AUTO'])}):\n")
                for seq, w in import_warn_categories['MASTERFILE_AUTO']:
                    f.write(f"    - {seq}: {w[:60]}\n")
                f.write("\n")

            if import_warn_categories['OTHER']:
                f.write(f"  [OTHER] Altres warnings ({len(import_warn_categories['OTHER'])}):\n")
                for seq, w in import_warn_categories['OTHER'][:10]:
                    f.write(f"    - {seq}: {w[:60]}\n")
                if len(import_warn_categories['OTHER']) > 10:
                    f.write(f"    ... i {len(import_warn_categories['OTHER'])-10} més\n")
                f.write("\n")

        # === CALIBRATION ===
        f.write(f"\n{'='*60}\n")
        f.write(f"CALIBRATION SUMMARY\n")
        f.write(f"{'='*60}\n\n")

        stats_c = self.stats['calibrate']
        f.write(f"Total: {stats_c['total']}\n")
        f.write(f"  OK Success: {stats_c['success']}\n")
        f.write(f"  FAIL Failed: {stats_c['failed']}\n")
        f.write(f"  WARN With warnings: {stats_c['with_warnings']}\n\n")

        # Comptar problemes de pics
        batman_seqs = []
        timeout_seqs = []
        low_snr_seqs = []
        asymmetry_seqs = []

        for r in self.calibration_results:
            m = r.get('metrics', {})
            if m.get('has_batman'):
                batman_seqs.append(r['seq_name'])
            if m.get('has_timeout'):
                timeout_seqs.append((r['seq_name'], m.get('timeout_time', 0), m.get('timeout_severity', '')))
            if m.get('snr', 999) < 10 and m.get('snr', 0) > 0:
                low_snr_seqs.append((r['seq_name'], m.get('snr', 0)))
            sym = m.get('symmetry', 1)
            if sym and (sym < 0.5 or sym > 2.0):
                asymmetry_seqs.append((r['seq_name'], sym))

        f.write(f"PEAK ISSUES DETECTED:\n\n")
        f.write(f"  [BAT] Batman artifacts: {len(batman_seqs)}\n")
        if batman_seqs:
            f.write(f"        SEQs: {', '.join(batman_seqs[:10])}\n")
            if len(batman_seqs) > 10:
                f.write(f"        ... i {len(batman_seqs)-10} més\n")

        f.write(f"  [TO] Timeout detected: {len(timeout_seqs)}\n")
        if timeout_seqs:
            for seq, t, sev in timeout_seqs[:5]:
                f.write(f"        - {seq}: t={t:.1f}min ({sev})\n")
            if len(timeout_seqs) > 5:
                f.write(f"        ... i {len(timeout_seqs)-5} més\n")

        f.write(f"  [SNR] Low SNR (<10): {len(low_snr_seqs)}\n")
        if low_snr_seqs:
            for seq, snr in sorted(low_snr_seqs, key=lambda x: x[1])[:5]:
                f.write(f"        - {seq}: SNR={snr:.0f}\n")

        f.write(f"  [ASYM] Asymmetry issues: {len(asymmetry_seqs)}\n")
        if asymmetry_seqs:
            for seq, sym in asymmetry_seqs[:5]:
                f.write(f"        - {seq}: sym={sym:.2f}\n")
        f.write("\n")

        # Llista de fallits
        failed_cals = [r for r in self.calibration_results if not r['success']]
        if failed_cals:
            f.write(f"FAILED CALIBRATIONS ({len(failed_cals)}):\n")
            for r in failed_cals:
                f.write(f"  - {r['seq_name']}: {'; '.join(r['errors'][:3])}\n")
            f.write("\n")

        # Calibracions amb qualitat baixa
        low_quality = [r for r in self.calibration_results
                      if r['metrics'].get('quality_score', 0) > 50]
        if low_quality:
            f.write(f"LOW QUALITY CALIBRATIONS (Q>50) ({len(low_quality)}):\n")
            for r in sorted(low_quality, key=lambda x: -x['metrics'].get('quality_score', 0))[:20]:
                q = r['metrics'].get('quality_score', 0)
                m = r['metrics']
                f.write(f"  - {r['seq_name']}: Q={q}")
                # Afegir detalls
                issues = []
                if m.get('has_batman'): issues.append('BAT')
                if m.get('has_timeout'): issues.append('TO')
                if m.get('snr', 999) < 10: issues.append('SNR')
                if issues:
                    f.write(f" [{','.join(issues)}]")
                f.write("\n")
            f.write("\n")

        # === MÈTRIQUES GLOBALS ===
        f.write(f"\n{'='*60}\n")
        f.write(f"GLOBAL METRICS\n")
        f.write(f"{'='*60}\n\n")

        # Calcular estadístiques de mètriques
        valid_cals = [r for r in self.calibration_results if r['success']]
        if valid_cals:
            import numpy as np

            # Separar per mètode
            bp_cals = [r for r in valid_cals if 'BP' in r['seq_name']]
            col_cals = [r for r in valid_cals if 'BP' not in r['seq_name']]

            areas = [r['metrics'].get('khp_area', 0) for r in valid_cals if r['metrics'].get('khp_area', 0) > 0]
            t_maxs = [r['metrics'].get('t_max', 0) for r in valid_cals if r['metrics'].get('t_max', 0) > 0]
            snrs = [r['metrics'].get('snr', 0) for r in valid_cals if r['metrics'].get('snr', 0) > 0]
            ratios = [r['metrics'].get('a254_doc_ratio', 0) for r in valid_cals if r['metrics'].get('a254_doc_ratio', 0) > 0]
            heights = [r['metrics'].get('height', 0) for r in valid_cals if r['metrics'].get('height', 0) > 0]
            symmetries = [r['metrics'].get('symmetry', 0) for r in valid_cals if r['metrics'].get('symmetry', 0) > 0]
            crs = [r['metrics'].get('concentration_ratio', 0) for r in valid_cals if r['metrics'].get('concentration_ratio', 0) > 0]

            f.write(f"All calibrations (n={len(valid_cals)}):\n")
            f.write(f"  BP mode: {len(bp_cals)}, COLUMN mode: {len(col_cals)}\n\n")

            f.write(f"Peak metrics:\n")
            if areas:
                f.write(f"  Area:     {np.mean(areas):>8.0f} +/- {np.std(areas):>6.0f}  (n={len(areas)}, range: {np.min(areas):.0f}-{np.max(areas):.0f})\n")
            if heights:
                f.write(f"  Height:   {np.mean(heights):>8.1f} +/- {np.std(heights):>6.1f}  (n={len(heights)}, range: {np.min(heights):.1f}-{np.max(heights):.1f})\n")
            if t_maxs:
                f.write(f"  t_max:    {np.mean(t_maxs):>8.2f} +/- {np.std(t_maxs):>6.2f}  (n={len(t_maxs)}, range: {np.min(t_maxs):.2f}-{np.max(t_maxs):.2f})\n")
            if symmetries:
                f.write(f"  Symmetry: {np.mean(symmetries):>8.2f} +/- {np.std(symmetries):>6.2f}  (n={len(symmetries)}, range: {np.min(symmetries):.2f}-{np.max(symmetries):.2f})\n")

            f.write(f"\nQuality metrics:\n")
            if snrs:
                f.write(f"  SNR:      {np.mean(snrs):>8.0f} +/- {np.std(snrs):>6.0f}  (n={len(snrs)}, min={np.min(snrs):.0f})\n")
            if ratios:
                f.write(f"  D/254:    {np.mean(ratios):>8.2f} +/- {np.std(ratios):>6.2f}  (n={len(ratios)}, range: {np.min(ratios):.2f}-{np.max(ratios):.2f})\n")
            if crs:
                f.write(f"  CR:       {np.mean(crs):>8.3f} +/- {np.std(crs):>6.3f}  (n={len(crs)}, range: {np.min(crs):.3f}-{np.max(crs):.3f})\n")

            # Estadístiques per mode
            if col_cals:
                col_areas = [r['metrics'].get('khp_area', 0) for r in col_cals if r['metrics'].get('khp_area', 0) > 0]
                col_tmax = [r['metrics'].get('t_max', 0) for r in col_cals if r['metrics'].get('t_max', 0) > 0]
                if col_areas:
                    f.write(f"\nCOLUMN mode (n={len(col_cals)}):\n")
                    f.write(f"  Area: {np.mean(col_areas):.0f} +/- {np.std(col_areas):.0f}\n")
                    if col_tmax:
                        f.write(f"  t_max: {np.mean(col_tmax):.2f} +/- {np.std(col_tmax):.2f}\n")

            if bp_cals:
                bp_areas = [r['metrics'].get('khp_area', 0) for r in bp_cals if r['metrics'].get('khp_area', 0) > 0]
                bp_tmax = [r['metrics'].get('t_max', 0) for r in bp_cals if r['metrics'].get('t_max', 0) > 0]
                if bp_areas:
                    f.write(f"\nBP mode (n={len(bp_cals)}):\n")
                    f.write(f"  Area: {np.mean(bp_areas):.0f} +/- {np.std(bp_areas):.0f}\n")
                    if bp_tmax:
                        f.write(f"  t_max: {np.mean(bp_tmax):.2f} +/- {np.std(bp_tmax):.2f}\n")

            # Detectar outliers
            if areas and len(areas) > 3:
                mean_area = np.mean(areas)
                std_area = np.std(areas)
                outliers = [(r['seq_name'], r['metrics'].get('khp_area', 0)) for r in valid_cals
                           if r['metrics'].get('khp_area', 0) > 0 and
                           abs(r['metrics'].get('khp_area', 0) - mean_area) > 2*std_area]
                if outliers:
                    f.write(f"\nPOTENTIAL OUTLIERS (>2 sigma):\n")
                    for seq, area in sorted(outliers, key=lambda x: -abs(x[1]-mean_area)):
                        sigma = (area - mean_area) / std_area
                        f.write(f"  - {seq}: Area={area:.0f} ({sigma:+.1f}σ)\n")

        # === EXCEPCIONS ===
        if self.errors:
            f.write(f"\n{'='*60}\n")
            f.write(f"EXCEPTIONS ({len(self.errors)})\n")
            f.write(f"{'='*60}\n\n")
            for err in self.errors:
                f.write(f"  - [{err['stage']}] {err['seq_name']}: {err['error'][:80]}\n")

    def _write_csv_report(self, path):
        """Escriu informe CSV amb mètriques completes."""
        import csv

        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Capçalera ampliada
            writer.writerow([
                'SEQ', 'Import_OK', 'Import_Warn', 'Cal_OK', 'Cal_Warn',
                'Method', 'Mode', 'N_Samples', 'N_KHP', 'N_Replicas',
                'Area', 'Height', 't_max', 't_max_std', 'Symmetry',
                'SNR', 'SNR_min', 'D/254', 'D/254_std', 'CR',
                'Quality', 'Quality_mean', 'Valid',
                'Batman', 'Timeout', 'TO_time', 'TO_severity',
                'Errors'
            ])

            # Combinar resultats
            import_dict = {r['seq_name']: r for r in self.import_results}
            cal_dict = {r['seq_name']: r for r in self.calibration_results}

            all_seqs = set(import_dict.keys()) | set(cal_dict.keys())

            for seq_name in sorted(all_seqs):
                imp = import_dict.get(seq_name, {})
                cal = cal_dict.get(seq_name, {})
                imp_m = imp.get('metrics', {})
                cal_m = cal.get('metrics', {})

                def fmt(val, decimals=2):
                    if val and val > 0:
                        return f"{val:.{decimals}f}"
                    return ''

                row = [
                    seq_name,
                    '1' if imp.get('success') else '0',
                    len(imp.get('warnings', [])),
                    '1' if cal.get('success') else '0',
                    len(cal.get('warnings', [])),
                    imp_m.get('method', ''),
                    cal_m.get('mode', ''),
                    imp_m.get('n_samples', ''),
                    imp_m.get('n_khp', ''),
                    cal_m.get('n_replicas', ''),
                    fmt(cal_m.get('khp_area'), 1),
                    fmt(cal_m.get('height'), 1),
                    fmt(cal_m.get('t_max'), 2),
                    fmt(cal_m.get('t_max_std'), 3),
                    fmt(cal_m.get('symmetry'), 2),
                    fmt(cal_m.get('snr'), 0),
                    fmt(cal_m.get('snr_min'), 0),
                    fmt(cal_m.get('a254_doc_ratio'), 2),
                    fmt(cal_m.get('a254_doc_ratio_std'), 3),
                    fmt(cal_m.get('concentration_ratio'), 3),
                    cal_m.get('quality_score', ''),
                    fmt(cal_m.get('quality_score_mean'), 1),
                    '1' if cal_m.get('valid_for_calibration', True) else '0',
                    '1' if cal_m.get('has_batman') else '0',
                    '1' if cal_m.get('has_timeout') else '0',
                    fmt(cal_m.get('timeout_time'), 1),
                    cal_m.get('timeout_severity', ''),
                    '; '.join((imp.get('errors', []) + cal.get('errors', []))[:2]),
                ]
                writer.writerow(row)

    def _clean_for_json(self, data):
        """Neteja dades per serialització JSON."""
        import numpy as np

        if isinstance(data, dict):
            return {k: self._clean_for_json(v) for k, v in data.items()
                   if not k.startswith('_') and k not in ['t_doc', 'y_doc', 't_dad', 'y_dad', 'y_dad_254']}
        elif isinstance(data, list):
            return [self._clean_for_json(v) for v in data]
        elif isinstance(data, np.ndarray):
            return None  # No guardar arrays grans
        elif isinstance(data, (np.integer, np.floating)):
            return float(data)
        elif isinstance(data, Path):
            return str(data)
        else:
            return data


def main():
    parser = argparse.ArgumentParser(description='HPSEC Batch Validation Tool')
    parser.add_argument('base_path', help='Directori base amb les SEQs')
    parser.add_argument('--stages', default='import,calibrate',
                       help='Etapes a executar (import,calibrate)')
    parser.add_argument('--output', '-o', help='Directori de sortida per informes')
    parser.add_argument('--limit', '-n', type=int, help='Limitar a N seqüències')
    parser.add_argument('--report-only', action='store_true',
                       help='Només generar informe dels JSONs existents')

    args = parser.parse_args()

    if not os.path.isdir(args.base_path):
        print(f"ERROR: {args.base_path} no és un directori vàlid")
        sys.exit(1)

    stages = [s.strip() for s in args.stages.split(',')]
    validator = BatchValidator(args.base_path, args.output)

    if args.report_only:
        # TODO: Implementar lectura de JSONs existents
        print("Mode report-only no implementat encara")
    else:
        validator.run_batch(stages=stages, limit=args.limit)


if __name__ == "__main__":
    main()
