"""
HPSEC Suite - Analyze Panel Package
====================================

Mòdul d'anàlisi refactoritzat amb taula unificada DOC+DAD.

Exporta:
- AnalyzePanel: Widget principal amb taula unificada i panel de fraccions
- AnalyzeWorker: Thread d'anàlisi
- SampleDetailDialog: Diàleg de detall amb gràfics i estadístiques
"""

from .worker import AnalyzeWorker
from .panel import AnalyzePanel
from .dialogs import SampleDetailDialog

__all__ = [
    "AnalyzePanel",
    "AnalyzeWorker",
    "SampleDetailDialog",
]
