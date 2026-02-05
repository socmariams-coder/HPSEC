"""
HPSEC Suite - Import Panel Package
==================================

Mòdul d'importació de seqüències refactoritzat.

Exporta tots els components per compatibilitat:
- ImportPanel: Widget principal
- ImportWorker: Thread d'importació
- OrphanFilesDialog, ChromatogramPreviewDialog: Diàlegs
- ComboBoxDelegate, FileAssignmentDelegate: Delegates
"""

from .delegates import ComboBoxDelegate, FileAssignmentDelegate
from .worker import ImportWorker
from .dialogs import OrphanFilesDialog, ChromatogramPreviewDialog
from .panel import ImportPanel

__all__ = [
    "ImportPanel",
    "ImportWorker",
    "OrphanFilesDialog",
    "ChromatogramPreviewDialog",
    "ComboBoxDelegate",
    "FileAssignmentDelegate",
]
