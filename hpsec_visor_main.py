#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Punt d'entrada per empaquetar el visor com a executable (PyInstaller).

Importa tot l'arbre del motor a dalt perquè PyInstaller el reculli, i arrenca
la finestra tkinter. Mode ocult `--build <carpeta|masterfile>` per provar el
bundle congelat des de la línia de comandes (genera l'HTML i surt).
"""

import os
import sys

# En congelat, el codi viu a sys._MEIPASS; afegim-lo al path per si de cas.
if getattr(sys, "frozen", False):
    base = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    if base not in sys.path:
        sys.path.insert(0, base)

# Imports de dalt -> PyInstaller segueix tota la cadena del motor
import numpy            # noqa: F401
import pandas           # noqa: F401
import scipy            # noqa: F401
import openpyxl         # noqa: F401
import hpsec_config     # noqa: F401
import hpsec_core       # noqa: F401
import hpsec_import     # noqa: F401
import hpsec_analyze    # noqa: F401
import hpsec_viewer_build as B  # noqa: E402
import hpsec_visor_gui          # noqa: F401,E402


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--build":
        out = B.build_one(sys.argv[2], open_after=False, progress=lambda m: print(m))
        print("OK:", out)
        return
    hpsec_visor_gui.main()


if __name__ == "__main__":
    main()
