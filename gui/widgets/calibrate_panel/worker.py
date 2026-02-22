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

            # Assegurar dades carregades (pot venir amb data_deferred=True
            # des del preload de metadades). FER-HO AL THREAD per no
            # bloquejar la UI (llegir MasterFile + CSV + Export3D és lent).
            if self.imported_data and self.imported_data.get("data_deferred"):
                from hpsec_import import ensure_data_loaded
                progress_cb(2, "Carregant senyals des del disc...")
                ensure_data_loaded(
                    self.imported_data,
                    config=self.config,
                    progress_callback=lambda pct, msg: progress_cb(
                        2 + int(pct * 0.15), msg  # 2-17% del total
                    ),
                )

            result = calibrate_from_import(
                self.imported_data,
                config=self.config,
                progress_callback=progress_cb
            )
            self.finished.emit(result)

        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n{traceback.format_exc()}")
