"""
HPSEC Suite - Calibrate Worker
==============================

Thread worker per calibració asíncrona.
"""

from PySide6.QtCore import Signal, QThread

from hpsec_calibrate import calibrate_from_import


class CalibrateWorker(QThread):
    """Worker thread para calibración asíncrona."""
    progress = Signal(int, str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, imported_data, config=None):
        super().__init__()
        self.imported_data = imported_data
        self.config = config

    def run(self):
        try:
            def progress_cb(pct, msg):
                self.progress.emit(int(pct), msg)

            result = calibrate_from_import(
                self.imported_data,
                config=self.config,
                progress_callback=progress_cb
            )
            self.finished.emit(result)

        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n{traceback.format_exc()}")
