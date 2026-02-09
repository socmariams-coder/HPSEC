"""
HPSEC Suite - Analyze Worker
=============================

Thread worker per anàlisi asíncrona.
"""

from PySide6.QtCore import Signal, QThread

from hpsec_analyze import analyze_sequence


class AnalyzeWorker(QThread):
    """Worker thread per anàlisi asíncrona."""
    progress = Signal(str, int)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, imported_data, calibration_data, config=None):
        super().__init__()
        self.imported_data = imported_data
        self.calibration_data = calibration_data
        self.config = config

    def run(self):
        try:
            def progress_cb(msg, pct):
                self.progress.emit(msg, int(pct))

            result = analyze_sequence(
                self.imported_data,
                self.calibration_data,
                config=self.config,
                progress_callback=progress_cb
            )
            self.finished.emit(result)

        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n\n{traceback.format_exc()}")
