"""
HPSEC Suite - Analyze Worker
=============================

Thread worker per anàlisi asíncrona.
"""

import os
import logging

from PySide6.QtCore import Signal, QThread

from hpsec_analyze import analyze_sequence

logger = logging.getLogger(__name__)


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

            # Assegurar dades carregades (pot venir amb data_deferred=True
            # des del preload de metadades). FER-HO AL THREAD per no
            # bloquejar la UI.
            if self.imported_data and self.imported_data.get("data_deferred"):
                from hpsec_import import ensure_data_loaded
                progress_cb("Carregant senyals des del disc...", 2)
                ensure_data_loaded(
                    self.imported_data,
                    config=self.config,
                    progress_callback=lambda pct, msg: progress_cb(
                        msg, 2 + int(pct * 0.15)
                    ),
                )

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


class SiblingAnalyzeWorker(QThread):
    """Worker per analitzar N siblings seqüencialment.

    Cada sibling s'analitza independentment amb analyze_sequence().
    """
    progress = Signal(str, int)           # msg, pct global
    sibling_finished = Signal(str, dict)  # path, result
    all_finished = Signal(dict)           # {path: result}
    error = Signal(str)

    def __init__(self, sibling_imported, sibling_calibrated, config=None):
        """
        Args:
            sibling_imported: dict {path: imported_data}
            sibling_calibrated: dict {path: calibration_data}
            config: config opcional
        """
        super().__init__()
        self.sibling_imported = sibling_imported
        self.sibling_calibrated = sibling_calibrated
        self.config = config

    def run(self):
        results = {}
        paths = list(self.sibling_imported.keys())
        n = len(paths)

        try:
            for i, path in enumerate(paths):
                name = os.path.basename(path)
                imported = self.sibling_imported[path]
                calibrated = self.sibling_calibrated.get(path, {})
                base_pct = int(i / n * 100)

                def progress_cb(msg, pct, _base=base_pct, _n=n, _name=name):
                    global_pct = _base + int(pct / _n)
                    self.progress.emit(f"[{_name}] {msg}", global_pct)

                # ensure_data_loaded si cal
                if imported and imported.get("data_deferred"):
                    from hpsec_import import ensure_data_loaded
                    progress_cb("Carregant senyals...", 2)
                    ensure_data_loaded(
                        imported,
                        config=self.config,
                        progress_callback=lambda pct, msg, _cb=progress_cb: _cb(
                            msg, 2 + int(pct * 0.15)
                        ),
                    )

                try:
                    from hpsec_analyze import save_analysis_result
                    result = analyze_sequence(
                        imported,
                        calibrated,
                        config=self.config,
                        progress_callback=progress_cb
                    )
                    # Guardar JSON individual per cada sibling
                    if result and result.get("success"):
                        save_analysis_result(result)
                except Exception as e:
                    logger.warning("Error analitzant %s: %s", name, e)
                    result = {
                        "success": False,
                        "error": str(e),
                        "samples_grouped": {},
                    }

                results[path] = result
                self.sibling_finished.emit(path, result)

            self.progress.emit(f"Anàlisi completada ({n} carpetes)", 100)
            self.all_finished.emit(results)

        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n\n{traceback.format_exc()}")
