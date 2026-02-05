"""
HPSEC Suite - Calibrate Panel Package
=====================================

Mòdul de calibració KHP refactoritzat.

Exporta tots els components per compatibilitat:
- CalibratePanel: Widget principal
- CalibrateWorker: Thread de calibració
- KHPReplicaGraphWidget, HistoryBarWidget: Widgets de gràfics
"""

from .worker import CalibrateWorker
from .graph_widgets import KHPReplicaGraphWidget, HistoryBarWidget
from .panel import CalibratePanel

__all__ = [
    "CalibratePanel",
    "CalibrateWorker",
    "KHPReplicaGraphWidget",
    "HistoryBarWidget",
]
