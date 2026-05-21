"""
QuantifyWorker — thread separat per quantificar un analysis_result.
"""

from PySide6.QtCore import QThread, Signal
import logging

logger = logging.getLogger(__name__)


class QuantifyWorker(QThread):
    """Worker per aplicar quantify_sequence en un thread separat."""

    progress = Signal(str, int)      # (msg, pct)
    completed = Signal(dict)          # analysis_result enriquit
    error = Signal(str)

    def __init__(self, analysis_result: dict, seq_path: str = None,
                 mode: str = None):
        super().__init__()
        self._analysis_result = analysis_result
        self._seq_path = seq_path
        self._mode = mode

    def run(self):
        try:
            from hpsec_analyze import quantify_sequence
            result = quantify_sequence(
                self._analysis_result,
                seq_path=self._seq_path,
                mode=self._mode,
                progress_callback=lambda msg, pct: self.progress.emit(msg, pct),
            )
            self.completed.emit(result)
        except Exception as e:
            logger.exception("QuantifyWorker failed: %s", e)
            self.error.emit(str(e))
