"""Widgets personalitzats per HPSEC Suite."""

from gui.widgets.dashboard_panel import DashboardPanel
from gui.widgets.process_wizard_panel import ProcessWizardPanel
from gui.widgets.export_panel import ExportPanel
from gui.widgets.samples_db_panel import SamplesDBPanel

# Panels individuals (usats pel wizard internament)
from gui.widgets.import_panel import ImportPanel
from gui.widgets.calibrate_panel import CalibratePanel
from gui.widgets.analyze_panel import AnalyzePanel
from gui.widgets.review_panel import ReviewPanel

__all__ = [
    "DashboardPanel",
    "ProcessWizardPanel",
    "ExportPanel",
    "SamplesDBPanel",
    "ImportPanel",
    "CalibratePanel",
    "AnalyzePanel",
    "ReviewPanel",
]
