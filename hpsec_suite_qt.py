#!/usr/bin/env python3
"""
HPSEC Suite v2.0 (PySide6)
===========================

Nova interfície gràfica moderna amb PySide6.

Executa:
    python hpsec_suite_qt.py

Requereix:
    pip install PySide6
"""

# Forçar backend Agg (sense finestres) ABANS de carregar res que importi matplotlib.
# Els informes (hpsec_reports.py) usen pyplot només per generar PDF/PNG; la GUI embeed
# les gràfiques via FigureCanvasQTAgg explícit, que no depèn del backend per defecte.
# Sense això, en mode interactiu (qtagg) cada figura de pyplot mostra una "finestra
# fantasma" grisa que parpelleja en generar informes.
import matplotlib
matplotlib.use("Agg")

from gui.main_window import main

if __name__ == "__main__":
    main()
