# -*- coding: utf-8 -*-
"""
Reorganitza les dades Export3D de Meritxell a l'estructura estàndard de Dades3.

Estructura objectiu:
    Dades3/XXX_SEQ/Export3d/*.csv
    Dades3/XXX_SEQ/XXX_SEQ_RAWDATA.xlsx (si existeix)

Mode DRY_RUN per defecte — mostra què faria sense tocar res.
"""

import os
import sys
import re
import shutil
import glob
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")

# =============================================================================
# CONFIGURACIÓ
# =============================================================================

DRY_RUN = False  # Canviar a False per executar realment

TARGET_DIR = "C:/Users/Lequia/Desktop/Dades3"
MERITXELL_BASE = "C:/Users/Lequia/Desktop/SEQ_antigues_Meritxell"
MERITXELL_3D = os.path.join(MERITXELL_BASE, "export 3D-Meritxell")

# =============================================================================
# FASE 1: Mapatge — Trobar totes les Export3D CSV i assignar-les a un SEQ
# =============================================================================

def scan_meritxell():
    """Scan all Meritxell Export3D data and map to SEQ numbers."""
    mapping = defaultdict(list)  # seq_num -> list of (src_path, notes)
    unmapped = []  # files that can't be assigned to a SEQ

    # --- 2024: Carpetes amb *_SEQ subfolders ---
    for campaign in ["ATL-GENER", "ATL-FEBRER", "ATL-ABRIL", "ATL-MAIG", "ATL-JUNY",
                     "FISERSA", "INORGANICS"]:
        campaign_dir = os.path.join(MERITXELL_3D, "2024", campaign)
        if not os.path.isdir(campaign_dir):
            continue
        for entry in os.listdir(campaign_dir):
            entry_path = os.path.join(campaign_dir, entry)
            if os.path.isdir(entry_path) and "_SEQ" in entry:
                # Extract SEQ number
                match = re.match(r"(\d+[A-Z]?)_SEQ", entry)
                if match:
                    seq_id = match.group(1)
                    csvs = glob.glob(os.path.join(entry_path, "*.csv"))
                    csvs += glob.glob(os.path.join(entry_path, "*.CSV"))
                    for f in csvs:
                        mapping[seq_id].append((f, f"2024/{campaign}/{entry}"))

    # --- 2024: CalibracioGener (BYPASS + COLUMNA, sense num SEQ) ---
    for mode in ["BYPASS", "COLUMNA"]:
        cal_dir = os.path.join(MERITXELL_3D, "2024", "CalibracioGener", mode)
        if os.path.isdir(cal_dir):
            csvs = glob.glob(os.path.join(cal_dir, "*.csv"))
            for f in csvs:
                unmapped.append((f, f"2024/CalibracioGener/{mode} (sense SEQ num)"))

    # --- 2024: Proves ---
    proves_dir = os.path.join(MERITXELL_3D, "2024", "Proves")
    if os.path.isdir(proves_dir):
        csvs = glob.glob(os.path.join(proves_dir, "*.csv"))
        for f in csvs:
            unmapped.append((f, "2024/Proves (sense SEQ num)"))

    # --- 2024: ATL-MAIG/232_PATRONS ---
    patrons_dir = os.path.join(MERITXELL_3D, "2024", "ATL-MAIG", "232_PATRONS")
    if os.path.isdir(patrons_dir):
        csvs = glob.glob(os.path.join(patrons_dir, "*.csv"))
        for f in csvs:
            mapping["232"].append((f, "2024/ATL-MAIG/232_PATRONS"))

    # --- 2023: 3D_BYPASS_MOSTREIG2 (SEQ178, SEQ181, SEQ189, SEQ193) ---
    bypass_dir = os.path.join(MERITXELL_3D, "2023", "3D_BYPASS_MOSTREIG2")
    if os.path.isdir(bypass_dir):
        for entry in os.listdir(bypass_dir):
            entry_path = os.path.join(bypass_dir, entry)
            if os.path.isdir(entry_path):
                match = re.match(r"SEQ(\d+)", entry)
                if match:
                    seq_id = match.group(1)
                    csvs = glob.glob(os.path.join(entry_path, "*.csv"))
                    suffix = entry.replace(f"SEQ{seq_id}", "").strip("_")
                    for f in csvs:
                        mapping[seq_id].append((f, f"2023/3D_BYPASS_MOSTREIG2/{entry} ({suffix})"))

    # --- 2023: Export3D_mostreig3 (Montfulla, PTLL, PTT — sense num SEQ) ---
    for subdir in ["Montfulla", "PTLL", "PTT"]:
        m3_dir = os.path.join(MERITXELL_3D, "2023", "Export3D_mostreig3", subdir)
        if os.path.isdir(m3_dir):
            csvs = glob.glob(os.path.join(m3_dir, "*.csv"))
            for f in csvs:
                unmapped.append((f, f"2023/Export3D_mostreig3/{subdir} (sense SEQ num)"))

    # Also root level of mostreig3
    m3_root = os.path.join(MERITXELL_3D, "2023", "Export3D_mostreig3")
    if os.path.isdir(m3_root):
        csvs = [f for f in glob.glob(os.path.join(m3_root, "*.csv"))
                if os.path.isfile(f)]
        for f in csvs:
            unmapped.append((f, "2023/Export3D_mostreig3/ (root, sense SEQ num)"))

    # --- 2023: Flat folders (sense SEQ) ---
    flat_dirs = {
        "ATL_Juny2023": "2023/ATL_Juny2023",
        "COLUMNES3_STANDARDS": "2023/COLUMNES3_STANDARDS",
        "Export3D_Inorganics": "2023/Export3D_Inorganics",
        "Export3D_Patrons": "2023/Export3D_Patrons",
        "Export_repe": "2023/Export_repe",
    }
    for dirname, note in flat_dirs.items():
        flat_dir = os.path.join(MERITXELL_3D, "2023", dirname)
        if os.path.isdir(flat_dir):
            csvs = glob.glob(os.path.join(flat_dir, "*.csv"))
            csvs += glob.glob(os.path.join(flat_dir, "*.CSV"))
            for f in csvs:
                unmapped.append((f, f"{note} (sense SEQ num)"))

    # --- IHSS_087_SEQ ---
    ihss_dir = os.path.join(MERITXELL_3D, "IHSS_087_SEQ")
    if os.path.isdir(ihss_dir):
        csvs = glob.glob(os.path.join(ihss_dir, "*.csv"))
        for f in csvs:
            mapping["087"].append((f, "IHSS_087_SEQ"))

    return mapping, unmapped


def scan_rawdata():
    """Find all RAWDATA xlsx files and map to SEQ numbers."""
    rawdata_map = {}
    for f in os.listdir(MERITXELL_BASE):
        if f.endswith(".xlsx") and "RAWDATA" in f:
            match = re.match(r"(\d+[A-Z]?)_SEQ_RAWDATA", f)
            if match:
                seq_id = match.group(1)
                rawdata_map[seq_id] = os.path.join(MERITXELL_BASE, f)
    return rawdata_map


def check_existing_dades3():
    """Check what already exists in Dades3."""
    existing = {}
    for entry in os.listdir(TARGET_DIR):
        entry_path = os.path.join(TARGET_DIR, entry)
        if os.path.isdir(entry_path) and "_SEQ" in entry:
            match = re.match(r"(\d+[A-Z]?)_SEQ", entry)
            if match:
                seq_id = match.group(1)
                has_export3d = os.path.isdir(os.path.join(entry_path, "Export3d"))
                n_csv = 0
                if has_export3d:
                    n_csv = len(glob.glob(os.path.join(entry_path, "Export3d", "*.csv")))
                    n_csv += len(glob.glob(os.path.join(entry_path, "Export3d", "*.CSV")))
                existing[seq_id] = {
                    "path": entry_path,
                    "has_export3d": has_export3d,
                    "n_csv": n_csv,
                    "folder_name": entry,
                }
    return existing


def main():
    print("=" * 70)
    print("REORGANITZACIÓ DADES MERITXELL → ESTRUCTURA DADES3")
    print(f"{'DRY RUN — no es mou res' if DRY_RUN else '⚠️  EXECUCIÓ REAL — es mouran fitxers!'}")
    print("=" * 70)

    # --- Scan ---
    print("\n📂 Escanejant dades Meritxell...")
    mapping, unmapped = scan_meritxell()
    rawdata_map = scan_rawdata()
    existing = check_existing_dades3()

    # --- Summary ---
    total_mapped = sum(len(v) for v in mapping.values())
    print(f"\n✅ CSV amb SEQ assignat: {total_mapped} fitxers en {len(mapping)} SEQs")
    print(f"❓ CSV sense SEQ:       {len(unmapped)} fitxers")
    print(f"📊 RAWDATA xlsx:        {len(rawdata_map)} fitxers")
    print(f"📁 SEQs existents a Dades3: {len(existing)}")

    # --- Detail per SEQ ---
    print("\n" + "=" * 70)
    print("DETALL PER SEQ (Export3D)")
    print("=" * 70)

    all_seq_ids = sorted(set(list(mapping.keys()) + list(rawdata_map.keys())),
                          key=lambda x: (int(re.match(r"\d+", x).group()), x))

    actions = []  # (action_type, src, dst, note)

    for seq_id in all_seq_ids:
        csv_files = mapping.get(seq_id, [])
        rawdata = rawdata_map.get(seq_id)
        exists = existing.get(seq_id)

        # Determine target folder name
        folder_name = f"{seq_id}_SEQ"
        if exists:
            folder_name = exists["folder_name"]

        target_seq = os.path.join(TARGET_DIR, folder_name)
        target_export3d = os.path.join(target_seq, "Export3d")

        status_parts = []
        if exists:
            if exists["has_export3d"]:
                status_parts.append(f"Dades3: JA EXISTEIX ({exists['n_csv']} CSV)")
            else:
                status_parts.append("Dades3: carpeta buida")
        else:
            status_parts.append("Dades3: NO EXISTEIX")

        if csv_files:
            sources = set(note for _, note in csv_files)
            status_parts.append(f"Meritxell: {len(csv_files)} CSV ({', '.join(sources)})")
        if rawdata:
            status_parts.append("RAWDATA: Sí")

        # Print status
        has_work = bool(csv_files) or bool(rawdata)
        marker = "→" if has_work else " "
        print(f"\n  {marker} {folder_name}")
        for sp in status_parts:
            print(f"      {sp}")

        # Plan actions
        if csv_files:
            if exists and exists["has_export3d"] and exists["n_csv"] > 0:
                print(f"      ⚠️  CONFLICTE: Dades3 ja té Export3d amb {exists['n_csv']} CSV")
                print(f"         Opció: SKIP (no sobreescriure)")
                actions.append(("SKIP_CONFLICT", None, target_export3d, folder_name))
            else:
                actions.append(("MKDIR_SEQ", None, target_seq, folder_name))
                actions.append(("MKDIR_EXPORT3D", None, target_export3d, folder_name))
                for src, note in csv_files:
                    dst = os.path.join(target_export3d, os.path.basename(src))
                    actions.append(("COPY_CSV", src, dst, note))

        if rawdata:
            dst = os.path.join(target_seq, os.path.basename(rawdata))
            if not os.path.exists(dst):
                actions.append(("COPY_RAWDATA", rawdata, dst, folder_name))

    # --- Unmapped files ---
    print("\n" + "=" * 70)
    print(f"FITXERS SENSE SEQ ASSIGNAT ({len(unmapped)})")
    print("=" * 70)

    unmapped_dir = os.path.join(TARGET_DIR, "_UNMAPPED_MERITXELL")
    by_source = defaultdict(list)
    for src, note in unmapped:
        by_source[note].append(src)

    for note, files in sorted(by_source.items()):
        print(f"\n  {note}: {len(files)} CSV")
        for f in files[:5]:
            print(f"    {os.path.basename(f)}")
        if len(files) > 5:
            print(f"    ... +{len(files)-5} més")

        # Create subfolder from note
        subfolder = note.replace(" (sense SEQ num)", "").replace("/", "_").replace(" ", "_")
        target_sub = os.path.join(unmapped_dir, subfolder)
        actions.append(("MKDIR_UNMAPPED", None, target_sub, note))
        for src in files:
            dst = os.path.join(target_sub, os.path.basename(src))
            actions.append(("COPY_UNMAPPED", src, dst, note))

    # --- Action summary ---
    print("\n" + "=" * 70)
    print("RESUM D'ACCIONS")
    print("=" * 70)

    action_counts = defaultdict(int)
    for action_type, *_ in actions:
        action_counts[action_type] += 1

    for at, count in sorted(action_counts.items()):
        print(f"  {at:20s}: {count}")

    total_copies = action_counts.get("COPY_CSV", 0) + action_counts.get("COPY_UNMAPPED", 0) + action_counts.get("COPY_RAWDATA", 0)
    print(f"\n  TOTAL COPIES: {total_copies}")
    print(f"  CONFLICTES SALTATS: {action_counts.get('SKIP_CONFLICT', 0)}")

    # --- Execute if not dry run ---
    if DRY_RUN:
        print(f"\n{'='*70}")
        print("DRY RUN COMPLETAT — Canvia DRY_RUN = False per executar")
        print(f"{'='*70}")
    else:
        print(f"\n{'='*70}")
        print("EXECUTANT...")
        print(f"{'='*70}")

        done = 0
        errors = 0
        for action_type, src, dst, note in actions:
            try:
                if action_type.startswith("MKDIR"):
                    if dst and not os.path.exists(dst):
                        os.makedirs(dst, exist_ok=True)
                elif action_type.startswith("COPY"):
                    if src and dst and not os.path.exists(dst):
                        shutil.copy2(src, dst)
                        done += 1
                elif action_type == "SKIP_CONFLICT":
                    pass
            except Exception as e:
                print(f"  ERROR: {action_type} {dst}: {e}")
                errors += 1

        print(f"\n  Fitxers copiats: {done}")
        print(f"  Errors: {errors}")
        print("  COMPLETAT!")


if __name__ == "__main__":
    main()
