"""
Script per verificar que els JSON existents són IDÈNTICS als que genera el codi actual.
"""
import json
import random
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from hpsec_import import import_sequence, get_data_folder
from hpsec_calibrate import calibrate_from_import


def load_json_file(seq_path, filename):
    """Carrega un JSON de la carpeta data d'una seqüència."""
    data_folder = get_data_folder(seq_path)
    json_path = Path(data_folder) / filename
    if json_path.exists():
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def load_import_manifest(seq_path):
    return load_json_file(seq_path, 'import_manifest.json')


def load_calibration_result(seq_path):
    return load_json_file(seq_path, 'calibration_result.json')


def compare_dicts(d1, d2, path=''):
    """Compara dos dicts recursivament, retorna diferències."""
    diffs = []
    skip_keys = {'date_processed', 'timestamp', 'processing_time', 'seq_path'}

    all_keys = set(d1.keys()) | set(d2.keys())
    for key in all_keys:
        if key in skip_keys:
            continue
        full_path = f'{path}.{key}' if path else key

        if key not in d1:
            diffs.append(f'  FALTA en actual: {full_path}')
        elif key not in d2:
            diffs.append(f'  NOU en recalculat: {full_path}')
        elif type(d1[key]) is not type(d2[key]):
            diffs.append(f'  TIPUS DIFERENT {full_path}: {type(d1[key]).__name__} vs {type(d2[key]).__name__}')
        elif isinstance(d1[key], dict):
            diffs.extend(compare_dicts(d1[key], d2[key], full_path))
        elif isinstance(d1[key], (list, tuple)):
            if len(d1[key]) != len(d2[key]):
                diffs.append(f'  LLARGADA DIFERENT {full_path}: {len(d1[key])} vs {len(d2[key])}')
        elif isinstance(d1[key], float):
            if abs(d1[key] - d2[key]) > 0.001:
                diffs.append(f'  VALOR DIFERENT {full_path}: {d1[key]:.4f} vs {d2[key]:.4f}')
        elif d1[key] != d2[key]:
            v1 = str(d1[key])[:50]
            v2 = str(d2[key])[:50]
            diffs.append(f'  VALOR DIFERENT {full_path}: {v1} vs {v2}')

    return diffs


def main():
    dades3 = Path(r'C:\Users\Lequia\Desktop\Dades3')

    # Trobar SEQs amb JSON
    seqs = [d for d in dades3.iterdir() if d.is_dir() and 'SEQ' in d.name.upper()
            and (d / 'CHECK' / 'data' / 'import_manifest.json').exists()]

    # Seleccionar 3 random (amb seed fix per reproduïbilitat)
    random.seed(123)
    selected = random.sample(seqs, min(3, len(seqs)))

    print('VERIFICACIÓ IDENTITAT JSON')
    print('=' * 70)

    results = {'ok': 0, 'diff': 0, 'error': 0}

    for seq_path in selected:
        print(f'\n>>> {seq_path.name}')

        # 1. IMPORT
        existing_import = load_import_manifest(str(seq_path))
        print('  Recalculant import_sequence...')
        new_import = import_sequence(str(seq_path))

        if not existing_import or not new_import:
            print('  ERROR carregant dades')
            results['error'] += 1
            continue

        diffs = compare_dicts(existing_import, new_import)
        if diffs:
            print(f'  IMPORT: {len(diffs)} DIFERÈNCIES:')
            for d in diffs[:10]:
                print(d)
            if len(diffs) > 10:
                print(f'    ... i {len(diffs)-10} més')
            results['diff'] += 1
        else:
            print('  IMPORT: IDÈNTIC ✓')
            results['ok'] += 1

        # 2. CALIBRATE
        existing_calib = load_calibration_result(str(seq_path))
        if existing_calib:
            print('  Recalculant calibrate_from_import...')
            new_calib = calibrate_from_import(new_import)

            if new_calib:
                diffs = compare_dicts(existing_calib, new_calib)
                if diffs:
                    print(f'  CALIBRATE: {len(diffs)} DIFERÈNCIES:')
                    for d in diffs[:10]:
                        print(d)
                    if len(diffs) > 10:
                        print(f'    ... i {len(diffs)-10} més')
                    results['diff'] += 1
                else:
                    print('  CALIBRATE: IDÈNTIC ✓')
                    results['ok'] += 1
            else:
                print('  CALIBRATE: recalcul retorna None')
                results['error'] += 1
        else:
            print('  CALIBRATE: no hi ha JSON existent')

    print('\n' + '=' * 70)
    print(f"RESUM: {results['ok']} idèntics, {results['diff']} amb diferències, {results['error']} errors")


if __name__ == '__main__':
    main()
