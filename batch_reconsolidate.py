# -*- coding: utf-8 -*-
"""
batch_reconsolidate.py - Reconsolida totes les SEQs amb validació KHP
======================================================================

Regenera tots els fitxers de consolidació usant la nova validació KHP integrada.
Mostra informació detallada sobre:
- Validació KHP (RATIO_LOW, TIMEOUT_HS, BATMAN, etc.)
- Shifts calculats (UIB i Direct vs A254)
- Reparació Batman si s'aplica

Ús:
    python batch_reconsolidate.py [--dry-run] [--start SEQ] [--end SEQ] [--only-column] [--only-bp]

Autor: HPSEC Suite
"""

import os
import sys
import re
import json
import argparse
from pathlib import Path
from datetime import datetime

# Assegurar path HPSEC
sys.path.insert(0, r"C:\Users\Lequia\Desktop\HPSEC")


def find_seq_folders(base_path):
    """Troba totes les carpetes SEQ ordenades per número."""
    seq_pattern = re.compile(r'^(\d+)([A-Z]?)_SEQ(_BP)?$')
    seq_folders = []

    for item in os.listdir(base_path):
        full_path = os.path.join(base_path, item)
        if os.path.isdir(full_path):
            match = seq_pattern.match(item)
            if match:
                seq_num = int(match.group(1))
                suffix = match.group(2) or ''
                is_bp = match.group(3) is not None
                seq_folders.append({
                    'name': item,
                    'path': full_path,
                    'num': seq_num,
                    'suffix': suffix,
                    'is_bp': is_bp,
                    'sort_key': (seq_num, suffix, is_bp)
                })

    # Ordenar per número de SEQ
    seq_folders.sort(key=lambda x: x['sort_key'])
    return seq_folders


def reconsolidate_sequence(seq_path, dry_run=False):
    """
    Reconsolida una seqüència.

    Returns:
        dict amb resultat complet
    """
    from hpsec_consolidate import consolidate_sequence

    seq_name = os.path.basename(seq_path)
    result = {
        'seq_name': seq_name,
        'success': False,
        'message': '',
        'method': '?',
        'mode': '?',
        'n_samples': 0,
        'khp_validation': None,
        'alignment': None,
        'errors': [],
        'warnings': [],
    }

    # Verificar que existeix
    if not os.path.isdir(seq_path):
        result['message'] = "Folder not found"
        return result

    if dry_run:
        result['success'] = True
        result['message'] = "DRY-RUN"
        return result

    # Esborrar JSON existent per forçar reconsolidació
    check_folder = os.path.join(seq_path, "CHECK")
    json_path = os.path.join(check_folder, "consolidation.json")
    if os.path.exists(json_path):
        try:
            os.remove(json_path)
        except Exception as e:
            result['message'] = f"Cannot delete JSON: {e}"
            return result

    # Reconsolidar
    try:
        cons_result = consolidate_sequence(seq_path)

        if cons_result is None:
            result['message'] = "consolidate_sequence returned None"
            return result

        result['success'] = cons_result.get('success', False)
        result['method'] = cons_result.get('method', '?')
        result['mode'] = cons_result.get('mode', '?')
        result['n_samples'] = len(cons_result.get('sample_stats', []))
        result['errors'] = cons_result.get('errors', [])
        result['warnings'] = cons_result.get('warnings', [])

        # Extreure info alineament i validació KHP
        alignment = cons_result.get('alignment', {})
        if alignment:
            result['alignment'] = {
                'khp_file': alignment.get('khp_file', ''),
                'source': alignment.get('source', ''),
                'shift_uib_sec': alignment.get('shift_uib', 0) * 60,
                'shift_direct_sec': alignment.get('shift_direct', 0) * 60,
                'align_mode': alignment.get('align_mode', ''),
            }

            # Validació KHP
            result['khp_validation'] = {
                'status': alignment.get('khp_validation', 'UNKNOWN'),
                'issues': alignment.get('khp_issues', []),
                'warnings': alignment.get('khp_warnings', []),
                'metrics': alignment.get('khp_metrics', {}),
            }

        if result['success']:
            result['message'] = f"OK ({result['n_samples']} samples)"
        else:
            errors = result['errors']
            result['message'] = f"Failed: {errors[0] if errors else 'Unknown'}"

    except Exception as e:
        result['message'] = f"Exception: {str(e)[:80]}"

    return result


def format_khp_status(result):
    """Formata l'estat de validació KHP per mostrar."""
    if result.get('khp_validation') is None:
        return "- NO_KHP"

    kv = result['khp_validation']
    status = kv.get('status', '?')

    # Casos especials
    if status == "NO_KHP":
        return "- NO_KHP (shift=0)"

    # Simplificar status
    if 'VALID' in status and 'INVALID' not in status:
        icon = '[OK]'
        color_status = 'VALID'
    elif 'INVALID' in status:
        icon = '[ERR]'
        color_status = 'INVALID'
    else:
        icon = '?'
        color_status = status

    # Afegir issues si n'hi ha
    issues = kv.get('issues', [])
    if issues:
        issues_short = ', '.join([i.split(':')[0] for i in issues[:2]])
        return f"{icon} {color_status} ({issues_short})"

    return f"{icon} {color_status}"


def format_source(alignment):
    """Formata la font de l'alineament."""
    source = alignment.get('source', '')
    if source.startswith('SIBLING:'):
        return f"[SIBLING: {source[8:]}]"
    elif source == 'KHP_LOCAL':
        return "[LOCAL]"
    elif source == 'NO_KHP':
        return "[NO_KHP]"
    elif source.startswith('HISTORIC:'):
        return f"[HISTORIC: {source[9:]}]"
    return f"[{source}]" if source else ""


def main():
    parser = argparse.ArgumentParser(description='Reconsolida totes les SEQs amb validació KHP')
    parser.add_argument('--dry-run', action='store_true', help='Mostra què es faria sense executar')
    parser.add_argument('--start', type=int, default=0, help='SEQ inicial (número)')
    parser.add_argument('--end', type=int, default=9999, help='SEQ final (número)')
    parser.add_argument('--only-column', action='store_true', help='Només SEQs COLUMN')
    parser.add_argument('--only-bp', action='store_true', help='Només SEQs BP')
    parser.add_argument('--base-path', type=str, default=r'C:\Users\Lequia\Desktop\Dades2',
                        help='Carpeta base amb les SEQs')
    args = parser.parse_args()

    print("=" * 80)
    print("BATCH RECONSOLIDATE - Validació KHP integrada")
    print("=" * 80)
    print(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Base path: {args.base_path}")
    print(f"Range: SEQ {args.start} - {args.end}")
    if args.only_column:
        print("Filter: COLUMN only")
    if args.only_bp:
        print("Filter: BP only")
    if args.dry_run:
        print("MODE: DRY-RUN (no changes)")
    print()

    # Trobar SEQs
    seq_folders = find_seq_folders(args.base_path)

    # Filtrar per rang
    seq_folders = [s for s in seq_folders if args.start <= s['num'] <= args.end]

    # Filtrar per tipus
    if args.only_column:
        seq_folders = [s for s in seq_folders if not s['is_bp']]
    if args.only_bp:
        seq_folders = [s for s in seq_folders if s['is_bp']]

    n_column = sum(1 for s in seq_folders if not s['is_bp'])
    n_bp = sum(1 for s in seq_folders if s['is_bp'])

    print(f"SEQs a processar: {len(seq_folders)} (COLUMN: {n_column}, BP: {n_bp})")
    print("-" * 80)

    # Processar
    results = []
    stats = {'ok': 0, 'failed': 0}
    start_time = datetime.now()

    for i, seq in enumerate(seq_folders, 1):
        seq_name = seq['name']
        seq_path = seq['path']
        mode = "BP" if seq['is_bp'] else "COL"

        print(f"\n[{i:2d}/{len(seq_folders)}] {seq_name:<20} ({mode})")

        result = reconsolidate_sequence(seq_path, dry_run=args.dry_run)
        results.append(result)

        if result['success']:
            stats['ok'] += 1

            # Mostrar info alineament
            alignment = result.get('alignment', {})
            if alignment:
                shift_uib = alignment.get('shift_uib_sec', 0)
                shift_direct = alignment.get('shift_direct_sec', 0)
                khp_file = alignment.get('khp_file', '')[:25]
                source_str = format_source(alignment)

                print(f"       KHP: {khp_file} {source_str}")
                print(f"       Shifts: UIB={shift_uib:+.1f}s, Direct={shift_direct:+.1f}s")

                # Validació KHP
                khp_status = format_khp_status(result)
                print(f"       Validacio: {khp_status}")

                # Mostrar issues/warnings
                kv = result.get('khp_validation', {})
                if kv.get('issues'):
                    for issue in kv['issues'][:2]:
                        print(f"       [!] {issue[:60]}")

            print(f"       -> {result['message']}")
        else:
            stats['failed'] += 1
            print(f"       [ERR] {result['message']}")

    # Resum final
    elapsed = (datetime.now() - start_time).total_seconds()
    print("\n" + "=" * 80)
    print("RESUM")
    print("=" * 80)
    print(f"Temps: {elapsed:.1f}s")
    print(f"Total: {len(results)} | OK: {stats['ok']} | Errors: {stats['failed']}")

    # Estadístiques KHP
    print("\n--- Validacio KHP ---")
    khp_valid = sum(1 for r in results if (r.get('khp_validation') or {}).get('status', '').startswith('VALID'))
    khp_invalid = sum(1 for r in results if 'INVALID' in str((r.get('khp_validation') or {}).get('status', '')))
    khp_none = sum(1 for r in results if (r.get('khp_validation') or {}).get('status', '') == 'NO_KHP'
                   or r.get('khp_validation') is None)

    # Estadístiques per font
    khp_local = sum(1 for r in results if (r.get('alignment') or {}).get('source', '') == 'KHP_LOCAL')
    khp_sibling = sum(1 for r in results if (r.get('alignment') or {}).get('source', '').startswith('SIBLING:'))
    khp_no = sum(1 for r in results if (r.get('alignment') or {}).get('source', '') == 'NO_KHP')

    print(f"VALID: {khp_valid} | INVALID: {khp_invalid} | NO_KHP: {khp_none}")
    print(f"Font: LOCAL={khp_local} | SIBLING={khp_sibling} | SENSE_KHP={khp_no}")

    # Llistar KHPs problematics
    problematic = [r for r in results if (r.get('khp_validation') or {}).get('issues')]
    if problematic:
        print("\nKHPs amb problemes:")
        for r in problematic:
            issues = ', '.join([i.split(':')[0] for i in r['khp_validation']['issues']])
            print(f"  {r['seq_name']:<20} {issues}")

    # Guardar report JSON
    if not args.dry_run and results:
        output_path = r"C:\Users\Lequia\Desktop\HPSEC\PROVES"
        os.makedirs(output_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(output_path, f"batch_reconsolidate_{timestamp}.json")

        report = {
            'timestamp': timestamp,
            'base_path': args.base_path,
            'total': len(results),
            'ok': stats['ok'],
            'failed': stats['failed'],
            'elapsed_sec': elapsed,
            'khp_stats': {
                'valid': khp_valid,
                'invalid': khp_invalid,
                'no_khp': khp_none,
            },
            'results': results,
        }

        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            print(f"\nReport: {report_path}")
        except Exception as e:
            print(f"\nError guardant report: {e}")

    # Mostrar KHP_History stats
    try:
        from hpsec_consolidate import get_khp_stats_summary
        khp_stats = get_khp_stats_summary(args.base_path)
        if khp_stats:
            print("\n--- KHP_History.json ---")
            print(f"Total entrades: {khp_stats.get('total', 0)}")
            print(f"Vàlids: {khp_stats.get('valid', 0)} ({khp_stats.get('valid_percent', 0):.1f}%)")
            if khp_stats.get('issue_counts'):
                print(f"Issues: {khp_stats['issue_counts']}")
    except Exception:
        pass

    print("\n" + "=" * 80)

    return 0 if stats['failed'] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
