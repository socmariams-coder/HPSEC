#!/usr/bin/env pythonw
# -*- coding: utf-8 -*-
"""Arrencador del visor (per a PC amb Python + Suite). La GUI viu a
hpsec_visor_gui.py perquè també es pugui importar en empaquetar l'exe."""
import os
import sys

APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import hpsec_visor_gui

if __name__ == "__main__":
    hpsec_visor_gui.main()
