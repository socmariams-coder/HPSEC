# -*- coding: utf-8 -*-
"""
batch_reconsolidate.py - Reconsolida totes les SEQs amb el nou càlcul de baseline
==================================================================================

Regenera tots els fitxers de consolidació usant el nou mètode de finestres
temporals per al càlcul de soroll baseline (evita artefactes de timeout).

Ús:
    python batch_reconsolidate.py [--dry-run] [--start SEQ] [--end SEQ]

Autor: HPSEC Suite
"""

import os
import sys
import re
import argparse
from pathlib import Path
from datetime import datetime


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
        tuple: (success, message, stats)
    """
    from hpsec_consolidate import consolidate_sequence

    seq_name = os.path.basename(seq_path)
    check_folder = os.path.join(seq_path, "CHECK")
    json_path = os.path.join(check_folder, "consolidation.json")

    # Verificar que existeix
    if not os.path.isdir(seq_path):
        return False, "Folder not found", {}

    if dry_run:
        exists = os.path.exists(json_path)
        return True, f"Would reconsolidate (JSON exists: {exists})", {}

    # Esborrar JSON existent per forçar reconsolidació
    if os.path.exists(json_path):
        try:
            os.remove(json_path)
        except Exception as e:
            return False, f"Cannot delete JSON: {e}", {}

    # Reconsolidar
    try:
        result = consolidate_sequence(seq_path)

        if result.get('success'):
            n_samples = len(result.get('sample_stats', []))
            return True, f"OK ({n_samples} samples)", {
                'samples': n_samples,
                'mode': result.get('mode', '?'),
                'errors': len(result.get('errors', []))
            }
        else:
            errors = result.get('errors', ['Unknown error'])
            return False, f"Failed: {errors[0] if errors else 'Unknown'}", {}

    except Exception as e:
        return False, f"Exception: {str(e)[:50]}", {}


def main():
    parser = argparse.ArgumentParser(description='Reconsolida totes les SEQs amb nou càlcul baseline')
    parser.add_argument('--dry-run', action='store_true', help='Mostra què es faria sense executar')
    parser.add_argument('--start', type=int, default=0, help='SEQ inicial (número)')
    parser.add_argument('--end', type=int, default=9999, help='SEQ final (número)')
    parser.add_argument('--base-path', type=str, default=r'C:\Users\Lequia\Desktop\Dades2',
                        help='Carpeta base amb les SEQs')
    args = parser.parse_args()

    print("=" * 70)
    print("BATCH RECONSOLIDATE - Nou calcul baseline amb finestres temporals")
    print("=" * 70)
    print(f"Base path: {args.base_path}")
    print(f"Range: SEQ {args.start} - {args.end}")
    if args.dry_run:
        print("MODE: DRY-RUN (no changes)")
    print()

    # Trobar SEQs
    seq_folders = find_seq_folders(args.base_path)

    # Filtrar per rang
    seq_folders = [s for s in seq_folders if args.start <= s['num'] <= args.end]

    print(f"Found {len(seq_folders)} SEQ folders to process")
    print("-" * 70)

    # Processar
    stats = {'ok': 0, 'failed': 0, 'skipped': 0}
    start_time = datetime.now()

    for i, seq in enumerate(seq_folders, 1):
        seq_name = seq['name']
        seq_path = seq['path']

        print(f"[{i:3d}/{len(seq_folders)}] {seq_name:<25} ", end='', flush=True)

        success, message, _ = reconsolidate_sequence(seq_path, dry_run=args.dry_run)

        if success:
            print(f"OK  - {message}")
            stats['ok'] += 1
        else:
            print(f"ERR - {message}")
            stats['failed'] += 1

    # Resum
    elapsed = (datetime.now() - start_time).total_seconds()
    print("-" * 70)
    print(f"Completed in {elapsed:.1f}s")
    print(f"  OK:     {stats['ok']}")
    print(f"  Failed: {stats['failed']}")

    if not args.dry_run:
        print()
        print("All sequences now use windowed baseline calculation.")


if __name__ == "__main__":
    main()
