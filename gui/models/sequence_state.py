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
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
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
    khp_source: str = ""  # "SEQ", "SIBLING", "HISTORY"
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

    # Data de la seqüència
    seq_date: str = ""      # Format: "YYYY-MM-DD" o "DD/MM/YY"

    # Warnings específics per fase
    import_warnings: List[str] = field(default_factory=list)
    analyze_warnings: List[str] = field(default_factory=list)

    # Notes de l'usuari
    notes: str = ""

    # Siblings (carpetes germanes com 282B_SEQ, 282C_SEQ)
    siblings: List[str] = field(default_factory=list)  # Paths de siblings
    is_sibling: bool = False  # True si és sibling secundari (282B, 282C...)

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

    def _check_phase(self, filename: str) -> PhaseStatus:
        """Comprova si una fase està completada."""
        filepath = os.path.join(self._check_data_path, filename)

        if not os.path.exists(filepath):
            return PhaseStatus(completed=False)

        try:
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
        self.analyze_warnings = []
        self.notes = ""

        # Del manifest d'importació
        if self.import_status.data:
            # Notes de l'usuari
            self.notes = self.import_status.data.get('notes', '')
            data = self.import_status.data
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

            # Warnings = resta (no errors, no "manifest existent")
            self.import_warnings = [
                w for w in raw_warnings
                if 'manifest existent' not in w.lower()
                and not w.upper().startswith('ERROR:')
            ]
            # Comptadors del summary
            summary = data.get('summary', {})
            self.n_samples = summary.get('total_samples', 0)
            self.n_khp = summary.get('total_khp', 0)
            self.n_pr = summary.get('total_pr', 0)
            # Data de la seqüència
            seq_info = data.get('sequence', {})
            date_str = seq_info.get('date', '')
            if date_str:
                # Convertir a format curt DD/MM/YY
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(date_str.replace(' ', 'T').split('T')[0])
                    self.seq_date = dt.strftime("%d/%m/%y")
                except:
                    self.seq_date = date_str[:10] if len(date_str) >= 10 else date_str

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

        # De l'anàlisi
        if self.analyze_status.data:
            data = self.analyze_status.data
            self.analyze_warnings = data.get('warnings', [])

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
    def import_state(self) -> str:
        """
        Estat de la fase Import per determinar color.
        Returns: 'ok', 'warning', 'error', 'pending'
        """
        if not self.import_status.completed:
            return 'pending'
        if self.import_status.errors:
            return 'error'
        if self.import_warnings:
            return 'warning'
        return 'ok'

    @property
    def calibrate_state(self) -> str:
        """
        Estat de la fase Calibrar per determinar color.
        - ok: KHP local (SEQ/DIRECT/UIB/DUAL)
        - warning: KHP sibling (acceptable)
        - error: Només històric o cap KHP
        Returns: 'ok', 'warning', 'error', 'pending'
        """
        if not self.calibrate_status.completed:
            return 'pending'
        if self.calibrate_status.errors:
            return 'error'

        khp_upper = self.khp_source.upper()
        # KHP local = verd
        if khp_upper in ('LOCAL', 'SEQ', 'DIRECT', 'UIB', 'DUAL'):
            return 'ok'
        # KHP sibling = taronja (acceptable)
        if khp_upper.startswith('SIBLING'):
            return 'warning'
        # Històric o cap = vermell
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
        return 'ok'

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
            Phase.CALIBRATE: "Calibrar",
            Phase.ANALYZE: "Analitzar",
            Phase.REVIEW: "Revisar",
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

    # Netejar el nom: treure _SEQ, _BP
    clean = name_upper.replace('_SEQ', '').replace('_BP', '').strip('_')

    # Buscar patró: número + lletra opcional
    match = re.match(r'^(\d+)([A-Z]?)$', clean)
    if match:
        seq_num = int(match.group(1))
        suffix = match.group(2)  # '' o 'B', 'C', etc.
        return (seq_num, suffix, is_bp)

    return (0, '', is_bp)


def get_all_sequences(data_folder: str, group_siblings: bool = True) -> List[SequenceState]:
    """
    Obté l'estat de totes les seqüències d'una carpeta.

    Args:
        data_folder: Carpeta amb les SEQs (ex: Dades3)
        group_siblings: Si True, agrupa siblings (282_SEQ + 282B_SEQ) i només
                       retorna el principal amb la llista de siblings

    Returns:
        Llista de SequenceState ordenada per nom
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
                    except Exception:
                        pass

        except Exception:
            pass  # Ignorar carpetes problemàtiques

    return sequences


if __name__ == "__main__":
    # Test
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    from hpsec_config import get_config
    cfg = get_config()
    data_folder = cfg.get("paths", "data_folder")

    print(f"Analitzant: {data_folder}")
    print("=" * 60)

    sequences = get_all_sequences(data_folder)

    print(f"{'SEQ':<15} {'ESTAT':<8} {'PROGRES':<8} {'ACCIO':<12}")
    print("-" * 60)

    for seq in sequences:
        # Usar ASCII per consola Windows
        icons = seq.status_icons.replace("✓", "+").replace("○", "-")
        print(f"{seq.seq_name:<15} {icons:<8} {seq.progress_pct:>5}%   {seq.next_action:<12}")
