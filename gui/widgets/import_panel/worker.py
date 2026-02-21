"""
HPSEC Suite - Import Worker
===========================

Thread worker per importació asíncrona.
"""

from PySide6.QtCore import Signal, QThread

from hpsec_import import import_sequence, import_from_manifest


class ImportWorker(QThread):
    """Worker thread per importació asíncrona."""
    progress = Signal(int, str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, seq_path, use_manifest=False, manifest=None, load_data=True):
        super().__init__()
        self.seq_path = seq_path
        self.use_manifest = use_manifest
        self.manifest = manifest
        self.load_data = load_data

    def run(self):
        try:
            def progress_cb(pct, msg):
                self.progress.emit(int(pct), msg)

            if self.use_manifest and self.manifest:
                result = import_from_manifest(
                    self.seq_path,
                    manifest=self.manifest,
                    progress_callback=progress_cb,
                    load_data=self.load_data
                )
            else:
                result = import_sequence(
                    self.seq_path,
                    progress_callback=progress_cb
                )
            self.finished.emit(result)
        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n\n{traceback.format_exc()}")
