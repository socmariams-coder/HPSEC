"""Test migració un a un amb timeout per detectar fitxers problemàtics."""
import os
import sys
import time
import subprocess

base_path = 'C:/Users/Lequia/Desktop/Dades2'
timeout_sec = 60  # 1 minut màxim per SEQ

# Trobar carpetes SEQ
folders = sorted([
    os.path.join(base_path, f)
    for f in os.listdir(base_path)
    if os.path.isdir(os.path.join(base_path, f)) and 'SEQ' in f.upper()
])

print(f"Trobades {len(folders)} carpetes SEQ")
print(f"Timeout: {timeout_sec}s per carpeta\n")

results = {'ok': [], 'error': [], 'timeout': [], 'skip': []}

for i, folder in enumerate(folders):
    name = os.path.basename(folder)
    print(f"[{i+1}/{len(folders)}] {name}... ", end='', flush=True)

    start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, '-u', 'hpsec_migrate_master.py', folder, '--force'],
            cwd='C:/Users/Lequia/Desktop/HPSEC',
            capture_output=True,
            text=True,
            timeout=timeout_sec
        )
        elapsed = time.time() - start

        if 'ok' in result.stdout.lower() or "'status': 'ok'" in result.stdout:
            print(f"OK ({elapsed:.1f}s)")
            results['ok'].append(name)
        elif 'skip' in result.stdout.lower():
            print(f"SKIP ({elapsed:.1f}s)")
            results['skip'].append(name)
        else:
            print(f"ERROR ({elapsed:.1f}s)")
            print(f"  stdout: {result.stdout[:200]}")
            print(f"  stderr: {result.stderr[:200]}")
            results['error'].append(name)

    except subprocess.TimeoutExpired:
        print(f"TIMEOUT (>{timeout_sec}s) ⚠️")
        results['timeout'].append(name)
    except Exception as e:
        print(f"EXCEPTION: {e}")
        results['error'].append(name)

print("\n" + "="*60)
print("RESUM:")
print(f"  OK:      {len(results['ok'])}")
print(f"  SKIP:    {len(results['skip'])}")
print(f"  ERROR:   {len(results['error'])}")
print(f"  TIMEOUT: {len(results['timeout'])}")

if results['timeout']:
    print(f"\nFitxers amb TIMEOUT:")
    for f in results['timeout']:
        print(f"  - {f}")

if results['error']:
    print(f"\nFitxers amb ERROR:")
    for f in results['error']:
        print(f"  - {f}")
