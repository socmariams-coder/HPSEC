"""
Script per compilar HPSEC Suite a .exe

Requisits:
- Python 3.10+
- PyInstaller: pip install pyinstaller

Execució:
    python build_exe.py

El fitxer .exe es crearà a la carpeta dist/
"""

import subprocess
import sys
import os

def main():
    print("=" * 60)
    print("HPSEC Suite - Compilació a .exe")
    print("=" * 60)

    # Verificar PyInstaller
    try:
        import PyInstaller
        print(f"PyInstaller versió: {PyInstaller.__version__}")
    except ImportError:
        print("PyInstaller no instal·lat. Instal·lant...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

    # Opcions de PyInstaller
    script = "HPSEC_Suite.py"
    name = "HPSEC_Suite"

    # Comanda PyInstaller
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",                    # Un sol fitxer .exe
        "--windowed",                   # Sense consola
        f"--name={name}",               # Nom del executable
        "--clean",                      # Neteja fitxers temporals
        "--noconfirm",                  # Sobreescriu sense preguntar
        # Dades addicionals si cal
        # "--add-data=config.json;.",
        script
    ]

    print(f"\nExecutant: {' '.join(cmd)}\n")
    print("-" * 60)

    # Executar
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))

    print("-" * 60)

    if result.returncode == 0:
        exe_path = os.path.join("dist", f"{name}.exe")
        print(f"\nCompilació completada!")
        print(f"Executable: {exe_path}")
        print(f"\nPots distribuir aquest fitxer als tècnics.")
    else:
        print(f"\nError durant la compilació (codi: {result.returncode})")
        sys.exit(1)


if __name__ == "__main__":
    main()
