# -*- coding: utf-8 -*-
"""
SequenceState - Gestió centralitzada de l'estat d'una seqüència
================================================================

Detecta automàticament l'estat de cada fase del pipeline basant-se
en els fitxers JSON existents a CHECK/data/.

Fases del pipeline:
1. IMPORT    → import_manifest.json
2. CALIBRATE → calibration_result.json
3. ANALYZE   → analysis_result.json
4. REVIEW    → review_result.json (mostres seleccionades)
5. EXPORT    → (fitxers generats a CHECK/)
"""

import os
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any

logger = logging.getLogger(__name__)
from datetime import datetime
from enum import Enum


class Phase(Enum):
    """Fases del pipeline."""
    IMPORT = "import"
    CALIBRATE = "calibrate"
    ANALYZE = "analyze"
    REVIEW = "review"
    EXPORT = "export"


@dataclass
class PhaseStatus:
    """Estat d'una fase."""
    completed: bool = False
    timestamp: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    errors: List[str] = field(default_factory=list)


@dataclass
class SequenceState:
    """
    Estat complet d'una seqüència.

    Detecta automàticament l'estat basant-se en els JSONs existents.
    Permet saber en quina fase està i quina és la següent acció.
    """
    seq_path: str
    seq_name: str = ""

    # Estat de cada fase
    import_status: PhaseStatus = field(default_factory=PhaseStatus)
    calibrate_status: PhaseStatus = field(default_factory=PhaseStatus)
    analyze_status: PhaseStatus = field(default_factory=PhaseStatus)
    review_status: PhaseStatus = field(default_factory=PhaseStatus)
    export_status: PhaseStatus = field(default_factory=PhaseStatus)

    # Info addicional
    has_khp: bool = False
    khp_source: str = ""  # "LOCAL", "SIBLING", "SENSE_KHP", "ALTERNATIU", "MITJANA HISTÒRICA"
    data_mode: str = ""   # "DUAL", "UIB", "DIRECT"
    method: str = ""      # "COLUMN", "BP"
    warnings: List[str] = field(default_factory=list)

    # Múltiples calibracions actives (una per condition_key)
    active_calibrations: List[Dict[str, Any]] = field(default_factory=list)
    n_calibration_conditions: int = 0

    # Comptadors de mostres (del manifest summary)
    n_samples: int = 0      # M = Mostres
    n_khp: int = 0          # PC = Patrons Calibració
    n_pr: int = 0           # PR = Patrons Referència
    n_blank: int = 0        # Blancs (MQ, etc.)
    n_control: int = 0      # Controls (NaOH, etc.)

    # Comptadors d'injeccions (del manifest stats)
    n_inj_master: int = 0   # Línies al MasterFile (1-HPLC-SEQ)
    n_inj_imported: int = 0 # Injeccions realment importades

    # Data de la seqüència
    seq_date: str = ""      # Format: "YYYY-MM-DD" o "DD/MM/YY"

    # Warnings específics per fase
    import_warnings: List[str] = field(default_factory=list)
    analyze_warnings: List[str] = field(default_factory=list)

    # Notes de l'usuari
    notes: str = ""

    # Notes unificades per al dashboard (font única de warnings/anomalies/notes)
    dashboard_notes: List[Dict[str, Any]] = field(default_factory=list)

    # Config fingerprint (per detectar obsolescència)
    config_fingerprint: str = ""

    # Versió de la Suite amb la qual es va processar
    suite_version: str = ""

    # Origen (multi-folder)
    source_folder: str = ""    # Nom curt de la carpeta d'origen (per filtre)
    source_path: str = ""      # Path complet de la carpeta d'origen

    # Siblings (carpetes germanes com 282B_SEQ, 282C_SEQ)
    siblings: List[str] = field(default_factory=list)  # Paths de siblings
    is_sibling: bool = False  # True si és sibling secundari (282B, 282C...)

    # Twin cross-method (COLUMN ↔ BP que analitzen les mateixes mostres)
    twin_seq_path: str = ""    # Path de la SEQ twin (mode complementari)
    twin_seq_name: str = ""    # Nom de la twin (ex: "288_SEQ_BP")
    twin_match_pct: float = 0  # % de mostres coincidents (0-100)

    # Paths dels JSONs
    _check_data_path: str = ""

    def __post_init__(self):
        """Inicialitza i detecta l'estat."""
        self.seq_name = os.path.basename(self.seq_path)
        self._check_data_path = os.path.join(self.seq_path, "CHECK", "data")
        self.refresh()

    def refresh(self):
        """Refresca l'estat llegint els JSONs."""
        self.import_status = self._check_phase("import_manifest.json")
        self.calibrate_status = self._check_phase("calibration_result.json")
        self.analyze_status = self._check_phase("analysis_result.json")
        self.review_status = self._check_phase("review_result.json")
        self.export_status = self._check_export()
        self._extract_metadata()

    # JSONs grans que NO cal carregar sencer per al dashboard
    _LARGE_JSONS = {"analysis_result.json"}

    def _check_phase(self, filename: str) -> PhaseStatus:
        """Comprova si una fase està completada."""
        filepath = os.path.join(self._check_data_path, filename)

        if not os.path.exists(filepath):
            return PhaseStatus(completed=False)

        try:
            # analysis_result.json pot ser >10MB — llegir només metadades
            if filename in self._LARGE_JSONS:
                data = self._read_json_metadata(filepath)
            else:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

            # Obtenir timestamp del fitxer o del contingut
            timestamp = data.get('timestamp') or data.get('date') or data.get('updated')
            if not timestamp:
                mtime = os.path.getmtime(filepath)
                timestamp = datetime.fromtimestamp(mtime).isoformat()

            return PhaseStatus(
                completed=True,
                timestamp=timestamp,
                data=data
            )
        except Exception as e:
            return PhaseStatus(
                completed=False,
                errors=[str(e)]
            )

    @staticmethod
    def _read_json_metadata(filepath: str, max_bytes: int = 4096) -> dict:
        """Llegeix només les primeres claus d'un JSON gran (sense carregar-lo sencer)."""
        import re
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                head = f.read(max_bytes)
            data = {}
            for key in ('success', 'timestamp', 'date', 'method', 'data_mode',
                        'seq_path', 'seq_name', 'warning_level',
                        'config_fingerprint'):
                pattern = rf'"{key}"\s*:\s*("([^"]*)"|(true|false|null|\d+[\.\d]*))'
                m = re.search(pattern, head)
                if m:
                    val = m.group(2) if m.group(2) is not None else m.group(3)
                    if val == 'true': val = True
                    elif val == 'false': val = False
                    elif val == 'null': val = None
                    else:
                        try: val = float(val) if '.' in str(val) else int(val)
                        except (ValueError, TypeError): pass
                    data[key] = val
            # Detectar si "warnings" és un array no buit (no podem parsejar-lo,
            # però podem veure si conté elements)
            m_warn = re.search(r'"warnings"\s*:\s*\[(\s*\]|\s*[^\]])', head)
            if m_warn:
                content = m_warn.group(1).strip()
                if content != ']':
                    # Hi ha warnings reals al JSON → parsejar-les si es pot
                    # (Intentar extreure el primer warning com a preview)
                    m_first = re.search(r'"warnings"\s*:\s*\[\s*"([^"]{1,100})"', head)
                    if m_first:
                        data['warnings'] = [m_first.group(1)]
                    else:
                        data['warnings'] = ['[metadata-only]']
                else:
                    data['warnings'] = []
            return data
        except Exception:
            return {}

    def _check_export(self) -> PhaseStatus:
        """Comprova si s'han exportat fitxers."""
        check_folder = os.path.join(self.seq_path, "CHECK")

        if not os.path.exists(check_folder):
            return PhaseStatus(completed=False)

        # Buscar PDFs o Excels exportats
        exported_files = []
        for f in os.listdir(check_folder):
            if f.endswith('.pdf') or f.endswith('.xlsx'):
                exported_files.append(f)

        if exported_files:
            return PhaseStatus(
                completed=True,
                data={'files': exported_files}
            )

        return PhaseStatus(completed=False)

    def _extract_metadata(self):
        """Extreu metadata dels JSONs per mostrar info addicional."""
        self.warnings = []
        self.import_warnings = []
        self.calibrate_warnings = []
        self.analyze_warnings = []
        self.review_warnings = []
        self.notes = ""

        # Del manifest d'importació
        if self.import_status.data:
            # Notes de l'usuari
            self.notes = self.import_status.data.get('notes', '')
            data = self.import_status.data
            self.suite_version = data.get('suite_version', '')
            self.method = data.get('sequence', {}).get('method', '')
            self.data_mode = data.get('sequence', {}).get('data_mode', '')
            # Warnings d'importació (masterfile, assignació fitxers, etc.)
            # Filtrar warnings no rellevants
            raw_warnings = data.get('warnings', [])
            raw_errors = data.get('errors', [])

            # Detectar errors (prefix "ERROR:" o errors explícits)
            import_errors = [
                w for w in raw_warnings + raw_errors
                if w.upper().startswith('ERROR:')
            ]
            if import_errors:
                self.import_status.errors = import_errors

            # Warnings = resta (no errors, no trivials)
            _trivial_import = {
                'manifest existent',
                '4-toc_calc no trobat',
                'calculant automàticament',
            }
            self.import_warnings = [
                w for w in raw_warnings
                if not any(t in w.lower() for t in _trivial_import)
                and not w.upper().startswith('ERROR:')
            ]
            # Enriquir avisos orfes antics (sense noms) amb noms del manifest
            orphan_files = data.get('orphan_files', {})
            orphan_uib = orphan_files.get('uib', [])
            orphan_dad = orphan_files.get('dad', [])
            if orphan_uib or orphan_dad:
                enriched = []
                for w in self.import_warnings:
                    if 'orfes' in w.lower() and '→' not in w:
                        # Format antic sense noms — substituir
                        if 'uib' in w.lower() and orphan_uib:
                            names = [os.path.basename(f) for f in orphan_uib]
                            w = f"UIB orfes: {', '.join(names)} → Assignar a Importar"
                        elif 'dad' in w.lower() and orphan_dad:
                            names = [os.path.basename(f) for f in orphan_dad]
                            w = f"DAD orfes: {', '.join(names)} → Assignar a Importar"
                    enriched.append(w)
                self.import_warnings = enriched
            # Comptadors del summary
            summary = data.get('summary', {})
            self.n_samples = summary.get('total_samples', 0)
            self.n_khp = summary.get('total_khp', 0)
            self.n_pr = summary.get('total_pr', 0)
            # Blank/Control: del summary si disponible, sinó comptar de samples[]
            if 'total_blank' in summary:
                self.n_blank = summary.get('total_blank', 0)
                self.n_control = summary.get('total_control', 0)
            else:
                samples_list = data.get('samples', [])
                self.n_blank = sum(1 for s in samples_list if s.get('type') == 'BLANK')
                self.n_control = sum(1 for s in samples_list if s.get('type') == 'CONTROL')
            # Comptadors d'injeccions (stats o summary)
            stats = data.get('stats', {})
            self.n_inj_master = stats.get('master_line_count', 0)
            # Prioritat: total_replicas_imported (nou) > summary.total_replicas (manifest)
            # NO usar total_injections perquè no reflecteix pèrdues per Inj# duplicat
            self.n_inj_imported = stats.get('total_replicas_imported', 0)
            if self.n_inj_imported == 0:
                self.n_inj_imported = summary.get('total_replicas', 0)
            # Últim fallback: total_injections (igual a master_line_count si no hi ha pèrdues)
            if self.n_inj_imported == 0:
                self.n_inj_imported = stats.get('total_injections', 0)
            # Data de la seqüència — normalitzar a DD/MM/YY
            seq_info = data.get('sequence', {})
            date_str = seq_info.get('date', '').strip()
            if date_str:
                from datetime import datetime
                dt = None
                # Intentar múltiples formats
                # 1. ISO: "2026-01-26 00:00:00" o "2026-01-26"
                try:
                    dt = datetime.fromisoformat(date_str.replace(' ', 'T').split('T')[0])
                except (ValueError, TypeError):
                    pass
                # 2. DD/MM/YYYY o DD/M/YYYY: "14/5/2021", "28/4/2022"
                if not dt:
                    for fmt in ('%d/%m/%Y', '%d/%m/%y', '%m/%d/%Y'):
                        try:
                            dt = datetime.strptime(date_str, fmt)
                            break
                        except (ValueError, TypeError):
                            pass
                self.seq_date = dt.strftime("%d/%m/%y") if dt else date_str

        # De la calibració
        if self.calibrate_status.data:
            data = self.calibrate_status.data
            calibrations = data.get('calibrations', [])

            # Recollir TOTES les calibracions actives (suport múltiples condicions)
            self.active_calibrations = [
                cal for cal in calibrations
                if cal.get('is_active', False) and not cal.get('is_outlier', False)
            ]
            self.n_calibration_conditions = len(self.active_calibrations)

            # Obtenir la primera activa (per compatibilitat amb codi antic)
            active_cal = self.active_calibrations[0] if self.active_calibrations else None
            if not active_cal and calibrations:
                active_cal = calibrations[0]

            # KHP info - LOCAL, SEQ, SIBLING significa que té KHP
            if active_cal:
                self.khp_source = active_cal.get('khp_source') or ''
                khp_upper = self.khp_source.upper()
                # Té KHP si és LOCAL, SEQ, DIRECT, UIB, DUAL o SIBLING
                self.has_khp = (
                    khp_upper in ('LOCAL', 'SEQ', 'DIRECT', 'UIB', 'DUAL') or
                    khp_upper.startswith('SIBLING')
                )

            # Extreure avisos de calibració (calibration_anomalies del catàleg)
            # Ignorar codis eliminats (sorollosos, presents en JSONs antics)
            from hpsec_warnings import IGNORED_KHP_CODES
            for cal in calibrations:
                for a in cal.get('calibration_anomalies', []):
                    if isinstance(a, dict):
                        if a.get('code', '') in IGNORED_KHP_CODES:
                            continue
                        sev = a.get('severity', 'info')
                        if sev in ('blocker', 'warning'):
                            label = a.get('label', a.get('code', ''))
                            sample = a.get('sample', '')
                            text = f"{label} ({sample})" if sample else label
                            self.calibrate_warnings.append(text)

        # De l'anàlisi
        if self.analyze_status.data:
            data = self.analyze_status.data
            raw_analyze_warnings = data.get('warnings', [])
            # Filtrar warnings trivials (metadata-only placeholders)
            if isinstance(raw_analyze_warnings, list):
                self.analyze_warnings = [
                    w for w in raw_analyze_warnings
                    if isinstance(w, str) and w.strip()
                    and w != '[metadata-only]'
                ]
            else:
                self.analyze_warnings = []
            # NO usar warning_level com a fallback per al triangle:
            # warning_level reflecteix anomalies per-mostra (IRREGULAR_TOP, etc.)
            # que es mostren dins l'anàlisi. El triangle del dashboard ha de
            # reservar-se per avisos reals de nivell superior (ex: "Missing DOC data").
            self.config_fingerprint = data.get('config_fingerprint', '')

        # De la revisió — BP info
        self.review_bp_name = None
        self.review_bp_mtime = None
        if self.review_status.data:
            bp_info = self.review_status.data.get('bp_info', {})
            self.review_bp_name = bp_info.get('bp_seq_name')
            self.review_bp_mtime = bp_info.get('bp_analysis_mtime')
            discarded = self.review_status.data.get('discarded_samples', [])
            if discarded:
                self.review_warnings.append(
                    f"{len(discarded)} mostr{'a' if len(discarded) == 1 else 'es'} descartad{'a' if len(discarded) == 1 else 'es'}"
                )
            if self.is_bp_stale:
                self.review_warnings.append(f"BP {self.review_bp_name} actualitzada")

        # Fallback: si no hi ha manifest, llegir inventari del MasterFile
        if self.n_samples == 0 and self.n_inj_master == 0:
            self._read_inventory_from_masterfile()

        # Construir notes unificades per al dashboard
        self._build_dashboard_notes()

    def _read_inventory_from_masterfile(self):
        """Llegeix inventari bàsic del MasterFile (1-HPLC-SEQ) sense importar."""
        import glob
        try:
            # Trobar MasterFile
            candidates = glob.glob(os.path.join(self.seq_path, "*MasterFile*.xlsx"))
            candidates = [c for c in candidates if 'backup' not in c.lower() and '~$' not in c]
            if not candidates:
                return

            import pandas as pd
            mf_path = candidates[0]
            xl = pd.ExcelFile(mf_path, engine='openpyxl')

            if '1-HPLC-SEQ' not in xl.sheet_names and '1-HPLC_SEQ' not in xl.sheet_names:
                return
            sheet = '1-HPLC-SEQ' if '1-HPLC-SEQ' in xl.sheet_names else '1-HPLC_SEQ'
            hplc = pd.read_excel(xl, sheet_name=sheet, header=0, engine='openpyxl')

            # Trobar columna de noms
            name_col = None
            for c in hplc.columns:
                if 'sample' in str(c).lower() and 'name' in str(c).lower():
                    name_col = c
                    break
            if name_col is None:
                return

            names = []
            for _, row in hplc.iterrows():
                n = str(row[name_col]).strip()
                if pd.isna(n) or n == 'nan' or not n:
                    continue
                names.append(n)

            self.n_inj_master = len(names)

            # Classificar per nom
            khp_patterns = ['khp', 'kph']
            mq_patterns = ['mq', 'blanc', 'blnc', 'blank']
            ctrl_patterns = ['naoh', 'buffer']

            n_khp = n_blank = n_ctrl = n_sample = 0
            unique_samples = set()
            for n in names:
                nl = n.lower()
                if any(p in nl for p in khp_patterns):
                    n_khp += 1
                elif any(p in nl for p in mq_patterns):
                    n_blank += 1
                elif any(p in nl for p in ctrl_patterns):
                    n_ctrl += 1
                else:
                    n_sample += 1
                    unique_samples.add(nl)

            # Dividir per 2 (rèpliques) per estimar mostres úniques
            self.n_khp = max(1, n_khp // 2) if n_khp > 0 else 0
            self.n_blank = max(1, n_blank // 2) if n_blank > 0 else 0
            self.n_control = max(1, n_ctrl // 2) if n_ctrl > 0 else 0
            self.n_samples = len(unique_samples)

            # Method
            if not self.method:
                self.method = "BP" if "BP" in self.seq_name.upper() else "COLUMN"

            # Data
            if not self.seq_date:
                date_col = None
                for c in hplc.columns:
                    cl = str(c).lower()
                    if 'acquired' in cl or ('injection' in cl and 'date' in cl):
                        date_col = c
                        break
                if date_col:
                    first_date = pd.to_datetime(hplc[date_col].iloc[0], errors='coerce')
                    if pd.notna(first_date):
                        self.seq_date = first_date.strftime("%d/%m/%y")

        except Exception as e:
            logger.debug("_read_inventory_from_masterfile %s: %s", self.seq_name, e)

    def _build_dashboard_notes(self):
        """
        Construeix la llista unificada de notes/warnings/anomalies per al dashboard.

        Font ÚNICA: usa les dades ja carregades per _check_phase() (PhaseStatus.data).
        El dashboard NO ha de rellegir JSONs — usa aquest camp directament.
        """
        notes = []

        _skip_warnings = {
            "Importat des de manifest existent",
            "4-TOC_CALC no trobat al MasterFile, calculant automàticament...",
        }

        # --- IMPORT ---
        if self.import_status.data:
            self._collect_warnings(self.import_status.data, "IMP", notes, _skip_warnings)
            self._collect_user_notes(self.import_status.data, "IMP", notes)

        # --- CALIBRACIÓ ---
        if self.calibrate_status.data:
            data = self.calibrate_status.data
            self._collect_warnings(data, "CAL", notes)
            # Calibration anomalies (ANOMALY_CATALOG) — deduplicar per codi
            from hpsec_warnings import IGNORED_KHP_CODES
            anom_counts = {}  # (code, label) → count
            for cal in data.get("calibrations", []):
                for anom in cal.get("calibration_anomalies", []):
                    if isinstance(anom, dict):
                        code = anom.get("code", "")
                        if code in IGNORED_KHP_CODES:
                            continue
                        sev = anom.get("severity", "info")
                        if sev in ("blocker", "warning"):
                            label = anom.get("label", code)
                            key = (code, label)
                            anom_counts[key] = anom_counts.get(key, 0) + 1
            for (code, label), count in anom_counts.items():
                content = f"{label} ({count})" if count > 1 else label
                notes.append({
                    "stage": "CAL", "type": "QUAL",
                    "content": content[:60],
                })
            self._collect_user_notes(data, "CAL", notes)

        # --- ANÀLISI (metadata-only, sense carregar JSON sencer) ---
        if self.analyze_status.data:
            wl = self.analyze_status.data.get("warning_level", "none")
            if wl in ("warning", "blocker"):
                notes.append({
                    "stage": "ANA",
                    "type": "WARN" if wl == "warning" else "ANOM",
                    "severity": wl,
                    "content": f"Avisos d'anàlisi ({wl})",
                    "icon": "⚠" if wl == "warning" else "!!",
                })

        # --- CONSOLIDACIÓ (si existeix, fitxer petit) ---
        con_path = os.path.join(self._check_data_path, "consolidation.json")
        if os.path.exists(con_path):
            try:
                with open(con_path, 'r', encoding='utf-8') as f:
                    con_data = json.load(f)
                self._collect_warnings(con_data, "CON", notes)
                self._collect_user_notes(con_data, "CON", notes)
            except Exception:
                pass

        # --- user_notes.json (notes sense etapa executada) ---
        notes_file = os.path.join(self._check_data_path, "user_notes.json")
        if os.path.exists(notes_file):
            try:
                with open(notes_file, 'r', encoding='utf-8') as f:
                    notes_data = json.load(f)
                for un in notes_data.get("notes", [])[-3:]:
                    stage = un.get("stage", "?")[:3].upper()
                    notes.append({
                        "stage": stage, "type": "USR",
                        "content": un.get("note", "")[:60],
                        "reviewer": un.get("reviewer", ""),
                    })
            except Exception:
                pass

        self.dashboard_notes = notes

    @staticmethod
    def _collect_warnings(data, stage, notes, skip_set=None):
        """Extreu warnings genèrics d'un JSON i els afegeix a notes."""
        warnings = data.get("warnings", [])
        if not isinstance(warnings, list):
            return
        for w in warnings[:3]:
            if isinstance(w, str) and w.strip():
                if skip_set and w.strip() in skip_set:
                    continue
                notes.append({"stage": stage, "type": "WARN", "content": w[:80]})
            elif isinstance(w, dict):
                msg = w.get("message", w.get("code", ""))
                if msg and (not skip_set or msg.strip() not in skip_set):
                    notes.append({"stage": stage, "type": "WARN", "content": msg[:80]})

    @staticmethod
    def _collect_user_notes(data, stage, notes):
        """Extreu warnings_confirmed i user_notes d'un JSON."""
        wc = data.get("warnings_confirmed")
        if isinstance(wc, dict):
            user_note = wc.get("user_note", "")
            if user_note:
                notes.append({
                    "stage": stage, "type": "NOTE",
                    "content": user_note[:80],
                    "reviewer": wc.get("reviewer", ""),
                })
        for un in data.get("user_notes", [])[-3:]:
            if isinstance(un, dict):
                notes.append({
                    "stage": stage, "type": "USR",
                    "content": un.get("note", "")[:60],
                    "reviewer": un.get("reviewer", ""),
                })

    @property
    def info_text(self) -> str:
        """Retorna text informatiu per mostrar al dashboard."""
        parts = []

        # Mètode i mode
        if self.method:
            parts.append(self.method)
        if self.data_mode:
            parts.append(self.data_mode)

        # KHP status - només mostrar si NO té KHP (problema)
        if self.calibrate_status.completed and not self.has_khp:
            parts.append("No KHP!")

        return " · ".join(parts) if parts else ""

    @property
    def is_config_stale(self) -> bool:
        """True si la config actual difereix de la usada en l'anàlisi."""
        if not self.config_fingerprint or not self.analyze_status.completed:
            return False
        from hpsec_config import get_config
        return self.config_fingerprint != get_config().compute_config_fingerprint()

    @property
    def has_warnings(self) -> bool:
        """Indica si hi ha warnings o problemes."""
        return self.calibrate_status.completed and not self.has_khp

    @property
    def calibration_conditions_text(self) -> str:
        """
        Retorna text amb les condicions de calibració actives.
        Ex: "KHP2@100µL, KHP2@50µL" si hi ha múltiples condicions.
        """
        if not self.active_calibrations:
            return ""

        conditions = []
        for cal in self.active_calibrations:
            conc = cal.get('conc_ppm', 0)
            vol = cal.get('volume_uL', 0)
            conditions.append(f"KHP{int(conc)}@{int(vol)}µL")

        return ", ".join(conditions)

    @property
    def samples_text(self) -> str:
        """
        Retorna text amb comptadors de mostres.
        Ex: "12M 2PC 7PR" o buit si no hi ha dades.
        """
        if not self.import_status.completed:
            return ""
        parts = []
        if self.n_samples > 0:
            parts.append(f"{self.n_samples}M")
        if self.n_khp > 0:
            parts.append(f"{self.n_khp}PC")
        if self.n_pr > 0:
            parts.append(f"{self.n_pr}PR")
        return " ".join(parts)

    @property
    def import_incomplete(self) -> bool:
        """True si el MasterFile té més injeccions que les importades."""
        if self.n_inj_master > 0 and self.n_inj_imported > 0:
            return self.n_inj_imported < self.n_inj_master
        return False

    @property
    def import_state(self) -> str:
        """
        Estat de la fase Import per determinar color.
        Returns: 'ok', 'warning', 'error', 'pending'
        """
        if not self.import_status.completed:
            return 'pending'
        if self.import_status.errors:
            return 'error'
        # Importació incompleta o warnings → warning (no error)
        if self.import_incomplete or self.import_warnings:
            return 'warning'
        return 'ok'

    @property
    def calibrate_state(self) -> str:
        """
        Estat de la fase Calibrar per determinar color.
        - ok: KHP local sense anomalies
        - warning: KHP sibling/sense KHP, o KHP local amb anomalies
        - error: Error real de calibració
        Returns: 'ok', 'warning', 'error', 'pending'
        """
        if not self.calibrate_status.completed:
            return 'pending'
        if self.calibrate_status.errors:
            return 'error'

        khp_upper = self.khp_source.upper()
        # KHP local = verd (amb triangle si té avisos)
        if khp_upper in ('LOCAL', 'SEQ', 'DIRECT', 'UIB', 'DUAL'):
            if self.calibrate_warnings:
                return 'warning'
            return 'ok'
        # KHP sibling o sense KHP = taronja (shift no verificable, quantificació OK)
        if khp_upper.startswith('SIBLING') or khp_upper == 'SENSE_KHP':
            return 'warning'
        # Alternatiu o mitjana històrica = taronja
        if khp_upper.startswith('ALTERNATIU') or khp_upper.startswith('MITJANA'):
            return 'warning'
        # Cas desconegut = vermell
        return 'error'

    @property
    def analyze_state(self) -> str:
        """
        Estat de la fase Analitzar per determinar color.
        Returns: 'ok', 'warning', 'error', 'pending'
        """
        if not self.analyze_status.completed:
            return 'pending'
        if self.analyze_status.errors:
            return 'error'
        if self.analyze_warnings:
            return 'warning'
        return 'ok'

    @property
    def review_state(self) -> str:
        """
        Estat de la fase Revisar per determinar color.
        Returns: 'ok', 'warning', 'error', 'pending'
        """
        if not self.review_status.completed:
            return 'pending'
        if self.review_status.errors:
            return 'error'
        # Comprovar si hi ha mostres descartades
        if self.review_status.data:
            discarded = self.review_status.data.get('discarded_samples', [])
            if discarded:
                return 'warning'
        if self.is_bp_stale:
            return 'warning'
        return 'ok'

    @property
    def is_bp_stale(self) -> bool:
        """True si la BP vinculada s'ha re-analitzat des de l'última revisió."""
        if not self.review_status.completed:
            return False
        bp_name = getattr(self, 'review_bp_name', None)
        bp_mtime = getattr(self, 'review_bp_mtime', None)
        if not bp_name or not bp_mtime:
            return False
        bp_path = os.path.join(os.path.dirname(self.seq_path), bp_name)
        bp_analysis = os.path.join(bp_path, "CHECK", "data", "analysis_result.json")
        if not os.path.exists(bp_analysis):
            return False
        current_mtime = os.path.getmtime(bp_analysis)
        return current_mtime > bp_mtime

    @property
    def current_phase(self) -> Phase:
        """Retorna la fase actual (primera no completada de les 4 principals)."""
        if not self.import_status.completed:
            return Phase.IMPORT
        if not self.calibrate_status.completed:
            return Phase.CALIBRATE
        if not self.analyze_status.completed:
            return Phase.ANALYZE
        if not self.review_status.completed:
            return Phase.REVIEW
        # Les 4 fases principals completades
        return Phase.EXPORT  # Indica que es pot exportar (opcional)

    @property
    def next_action(self) -> str:
        """Retorna la descripció de la següent acció."""
        phase = self.current_phase
        # Si les 4 fases principals estan completades
        if self.review_status.completed:
            return "Completat"
        actions = {
            Phase.IMPORT: "Importar",
            Phase.CALIBRATE: "Verificar",
            Phase.ANALYZE: "Analitzar",
            Phase.REVIEW: "Exportar",
        }
        return actions.get(phase, "Completat")

    @property
    def progress_pct(self) -> int:
        """Retorna el percentatge de progrés (0-100) de les 4 fases principals."""
        completed = sum([
            self.import_status.completed,
            self.calibrate_status.completed,
            self.analyze_status.completed,
            self.review_status.completed,
        ])
        return int(completed / 4 * 100)

    @property
    def status_icons(self) -> str:
        """Retorna icones d'estat per les 4 fases principals."""
        def icon(status: PhaseStatus) -> str:
            return "✓" if status.completed else "○"

        return f"{icon(self.import_status)}{icon(self.calibrate_status)}{icon(self.analyze_status)}{icon(self.review_status)}"

    def can_run_phase(self, phase: Phase) -> bool:
        """Comprova si es pot executar una fase."""
        if phase == Phase.IMPORT:
            return True
        if phase == Phase.CALIBRATE:
            return self.import_status.completed
        if phase == Phase.ANALYZE:
            return self.calibrate_status.completed
        if phase == Phase.REVIEW:
            return self.analyze_status.completed
        if phase == Phase.EXPORT:
            return self.review_status.completed
        return False

    def get_phase_status(self, phase: Phase) -> PhaseStatus:
        """Retorna l'estat d'una fase."""
        mapping = {
            Phase.IMPORT: self.import_status,
            Phase.CALIBRATE: self.calibrate_status,
            Phase.ANALYZE: self.analyze_status,
            Phase.REVIEW: self.review_status,
            Phase.EXPORT: self.export_status
        }
        return mapping[phase]

    def invalidate_from(self, phase: Phase):
        """
        Invalida una fase i totes les posteriors.

        Quan es refà una fase, les posteriors queden pendents.
        Els JSONs no s'esborren (es sobreescriuran quan es refacin).
        """
        phases = [Phase.IMPORT, Phase.CALIBRATE, Phase.ANALYZE, Phase.REVIEW, Phase.EXPORT]
        start_idx = phases.index(phase)

        for p in phases[start_idx:]:
            status = self.get_phase_status(p)
            status.completed = False
            status.data = None

    def save_notes(self, notes_text: str) -> bool:
        """
        Guarda les notes al manifest.json.

        Args:
            notes_text: Text de les notes

        Returns:
            True si s'ha guardat correctament
        """
        manifest_path = os.path.join(self._check_data_path, "import_manifest.json")

        if not os.path.exists(manifest_path):
            return False

        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            data['notes'] = notes_text
            data['notes_updated'] = datetime.now().isoformat()

            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            self.notes = notes_text
            return True
        except Exception as e:
            print(f"Error guardant notes: {e}")
            return False

    def to_dict(self) -> Dict[str, Any]:
        """Converteix l'estat a diccionari."""
        return {
            'seq_name': self.seq_name,
            'seq_path': self.seq_path,
            'current_phase': self.current_phase.value,
            'next_action': self.next_action,
            'progress_pct': self.progress_pct,
            'status_icons': self.status_icons,
            'phases': {
                'import': {
                    'completed': self.import_status.completed,
                    'timestamp': self.import_status.timestamp
                },
                'calibrate': {
                    'completed': self.calibrate_status.completed,
                    'timestamp': self.calibrate_status.timestamp
                },
                'analyze': {
                    'completed': self.analyze_status.completed,
                    'timestamp': self.analyze_status.timestamp
                },
                'review': {
                    'completed': self.review_status.completed,
                    'timestamp': self.review_status.timestamp
                },
                'export': {
                    'completed': self.export_status.completed,
                }
            }
        }

    def __repr__(self) -> str:
        return f"SequenceState({self.seq_name}: {self.status_icons} → {self.next_action})"


def _extract_seq_num_and_suffix(seq_name: str) -> tuple:
    """
    Extreu el número de SEQ i el sufix de lletra.

    Ex: '282_SEQ' → (282, '')
        '282B_SEQ' → (282, 'B')
        '282_SEQ_BP' → (282, '', True)
        '282B_SEQ_BP' → (282, 'B', True)

    Returns:
        (seq_num, suffix, is_bp)
    """
    import re
    name_upper = seq_name.upper()
    is_bp = '_BP' in name_upper

    # Netejar el nom: treure _SEQ, _BP, _CAL i altres sufixos
    clean = name_upper.replace('_SEQ', '').replace('_BP', '').replace('_CAL', '').strip('_')

    # Buscar patró: número + lletra opcional
    match = re.match(r'^(\d+)([A-Z]?)$', clean)
    if match:
        seq_num = int(match.group(1))
        suffix = match.group(2)  # '' o 'B', 'C', etc.
        return (seq_num, suffix, is_bp)

    return (0, '', is_bp)


def _scan_single_folder(data_folder: str, group_siblings: bool = True) -> List[SequenceState]:
    """
    Escaneja UNA carpeta i retorna les SequenceState trobades.

    Args:
        data_folder: Carpeta amb les SEQs (ex: Dades3)
        group_siblings: Si True, agrupa siblings (282_SEQ + 282B_SEQ)

    Returns:
        Llista de SequenceState (sense source_folder assignat)
    """
    sequences = []

    if not os.path.isdir(data_folder):
        return sequences

    # Primer pas: detectar totes les carpetes SEQ
    seq_folders = {}  # {(num, is_bp): [(path, suffix), ...]}

    for item in sorted(os.listdir(data_folder)):
        item_path = os.path.join(data_folder, item)
        if os.path.isdir(item_path) and "_SEQ" in item.upper():
            seq_num, suffix, is_bp = _extract_seq_num_and_suffix(item)
            if seq_num > 0:
                key = (seq_num, is_bp)
                if key not in seq_folders:
                    seq_folders[key] = []
                seq_folders[key].append((item_path, suffix))

    # Ordenar cada grup: base primer (sense sufix), després alfabètic
    for key in seq_folders:
        seq_folders[key].sort(key=lambda x: (x[1] != '', x[1]))

    # Segon pas: crear SequenceState
    for key, paths_suffixes in sorted(seq_folders.items()):
        # El primer és el principal (base o menor sufix)
        primary_path, _ = paths_suffixes[0]

        try:
            state = SequenceState(primary_path)

            if group_siblings and len(paths_suffixes) > 1:
                # Afegir siblings secundaris
                state.siblings = [p for p, _ in paths_suffixes[1:]]

            sequences.append(state)

            # Si NO agrupem, afegir també els siblings com a entrades separades
            if not group_siblings:
                for sibling_path, suffix in paths_suffixes[1:]:
                    try:
                        sibling_state = SequenceState(sibling_path)
                        sibling_state.is_sibling = True
                        sequences.append(sibling_state)
                    except Exception as e:
                        logger.debug("Could not load sibling %s: %s", sibling_path, e)

        except Exception as e:
            logger.debug("Skipping problematic folder %s: %s", primary_path, e)

    return sequences


def _get_sample_names_from_manifest(seq_path: str) -> set:
    """Llegeix els noms de mostres regulars (no blancs/controls) d'un manifest."""
    manifest_path = os.path.join(seq_path, "CHECK", "data", "import_manifest.json")
    if not os.path.isfile(manifest_path):
        return set()
    try:
        import json
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        samples = manifest.get("samples", [])
        # samples pot ser llista o dict
        if isinstance(samples, dict):
            names = set(samples.keys())
        elif isinstance(samples, list):
            names = {s.get("name", s.get("sample_name", "")) for s in samples}
        else:
            return set()
        # Filtrar blancs/controls/KHP
        exclude = {"MQ", "MQ1", "MQ2", "NAOH", "BUFFER", "NaOH"}
        result = set()
        for n in names:
            if not n:
                continue
            n_upper = n.upper()
            if any(tag in n_upper for tag in ["MQ", "NAOH", "BUFFER", "KHP", "BLANC"]):
                continue
            if n in exclude:
                continue
            result.add(n)
        return result
    except Exception as e:
        logger.debug("Could not read sample names from %s: %s", manifest_path, e)
        return set()


def _detect_twins(sequences: List[SequenceState]):
    """
    Detecta twins cross-method (COLUMN <-> BP) per coincidencia de mostres.

    Estrategia en 2 passos:
    1. Primer: match per num de SEQ (072_SEQ <-> 072_SEQ_BP) — fallback si no hi ha manifest
    2. Despres: match per noms de mostres (288_SEQ <-> 287_SEQ_BP) — prioritari

    Dos seqs son twins si comparteixen >= 50% de mostres regulars.
    """
    MIN_MATCH_PCT = 50

    # Separar COLUMN i BP
    columns = []
    bps = []
    for seq in sequences:
        _, _, is_bp = _extract_seq_num_and_suffix(seq.seq_name)
        if is_bp:
            bps.append(seq)
        else:
            columns.append(seq)

    if not columns or not bps:
        return

    # Pas 1: fallback per numero (seqs sense manifest)
    col_by_num = {}
    bp_by_num = {}
    for s in columns:
        num, _, _ = _extract_seq_num_and_suffix(s.seq_name)
        if num > 0:
            col_by_num.setdefault(num, []).append(s)
    for s in bps:
        num, _, _ = _extract_seq_num_and_suffix(s.seq_name)
        if num > 0:
            bp_by_num.setdefault(num, []).append(s)

    for num in col_by_num:
        if num in bp_by_num:
            col_seq = col_by_num[num][0]
            bp_seq = bp_by_num[num][0]
            col_seq.twin_seq_path = bp_seq.seq_path
            col_seq.twin_seq_name = bp_seq.seq_name
            bp_seq.twin_seq_path = col_seq.seq_path
            bp_seq.twin_seq_name = col_seq.seq_name
            logger.debug("Twins per numero: %s <-> %s", col_seq.seq_name, bp_seq.seq_name)

    # Pas 2: match per mostres (sobreescriu el fallback si millor)
    # Carregar mostres dels manifests (cache)
    samples_cache = {}
    for seq in columns + bps:
        names = _get_sample_names_from_manifest(seq.seq_path)
        if names:
            samples_cache[seq.seq_path] = names

    # Per cada COLUMN amb manifest, buscar la millor BP
    for col_seq in columns:
        col_samples = samples_cache.get(col_seq.seq_path)
        if not col_samples:
            continue

        best_bp = None
        best_pct = 0
        best_common = 0

        for bp_seq in bps:
            bp_samples = samples_cache.get(bp_seq.seq_path)
            if not bp_samples:
                continue
            common = col_samples & bp_samples
            min_count = min(len(col_samples), len(bp_samples))
            match_pct = len(common) / min_count * 100 if min_count > 0 else 0
            if match_pct >= MIN_MATCH_PCT and match_pct > best_pct:
                best_pct = match_pct
                best_bp = bp_seq
                best_common = len(common)

        if best_bp:
            col_seq.twin_seq_path = best_bp.seq_path
            col_seq.twin_seq_name = best_bp.seq_name
            col_seq.twin_match_pct = best_pct
            best_bp.twin_seq_path = col_seq.seq_path
            best_bp.twin_seq_name = col_seq.seq_name
            best_bp.twin_match_pct = best_pct
            logger.debug("Twins per mostres: %s <-> %s (%.0f%%, %d comunes)",
                        col_seq.seq_name, best_bp.seq_name, best_pct, best_common)


def get_all_sequences(data_folders, group_siblings: bool = True) -> List[SequenceState]:
    """
    Obté l'estat de totes les seqüències d'una o múltiples carpetes.

    Args:
        data_folders: Carpeta (str) o llista de carpetes (list) amb SEQs
        group_siblings: Si True, agrupa siblings (282_SEQ + 282B_SEQ)

    Returns:
        Llista de SequenceState ordenada per número de SEQ
    """
    # Backward compat: string → llista
    if isinstance(data_folders, str):
        data_folders = [data_folders]

    all_seqs = []
    for folder in data_folders:
        if not folder:
            continue
        folder_name = os.path.basename(folder)
        seqs = _scan_single_folder(folder, group_siblings)
        for seq in seqs:
            seq.source_folder = folder_name
            seq.source_path = folder
        all_seqs.extend(seqs)

    # Detectar twins cross-method (COLUMN ↔ BP)
    _detect_twins(all_seqs)

    # Ordenar globalment per (seq_num, is_bp)
    def _sort_key(s):
        seq_num, suffix, is_bp = _extract_seq_num_and_suffix(s.seq_name)
        return (seq_num, is_bp, suffix)

    all_seqs.sort(key=_sort_key)
    return all_seqs


if __name__ == "__main__":
    # Test
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    from hpsec_config import get_config, get_data_folders
    cfg = get_config()
    data_folders = get_data_folders()

    print(f"Analitzant: {data_folders}")
    print("=" * 60)

    sequences = get_all_sequences(data_folders)

    print(f"{'SEQ':<15} {'ESTAT':<8} {'PROGRES':<8} {'ACCIO':<12}")
    print("-" * 60)

    for seq in sequences:
        # Usar ASCII per consola Windows
        icons = seq.status_icons.replace("✓", "+").replace("○", "-")
        print(f"{seq.seq_name:<15} {icons:<8} {seq.progress_pct:>5}%   {seq.next_action:<12}")
