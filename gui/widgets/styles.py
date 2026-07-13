"""
HPSEC Suite - Shared UI Styles
==============================

Constants i estils compartits per tots els panels.
Garanteix consistència visual a tota l'aplicació.
"""

from PySide6.QtGui import QFont, QColor
from PySide6.QtCore import Qt

# Font ÚNICA de color: el tema modern (gui/styles/theme.py). Aquest mòdul manté
# els noms COLOR_* (molts panells hi depenen) però els valors surten del tema,
# perquè no hi hagi dues paletes en conflicte.
from gui.styles.theme import COLORS as _T


# =============================================================================
# LAYOUT CONSTANTS
# =============================================================================

# Marges principals del panel (left, top, right, bottom)
PANEL_MARGINS = (16, 12, 16, 12)

# Spacing entre elements
PANEL_SPACING = 12
SECTION_SPACING = 16
COMPACT_SPACING = 8

# Marges per seccions internes
SECTION_MARGINS = (12, 8, 12, 8)
COMPACT_MARGINS = (8, 6, 8, 6)
NO_MARGINS = (0, 0, 0, 0)


# =============================================================================
# TYPOGRAPHY
# =============================================================================

# Títol principal del panel
TITLE_FONT_SIZE = 16
TITLE_FONT_WEIGHT = QFont.Bold

# Subtítols i seccions
SUBTITLE_FONT_SIZE = 14
SUBTITLE_FONT_WEIGHT = QFont.DemiBold

# Text normal
BODY_FONT_SIZE = 12
SMALL_FONT_SIZE = 11
TINY_FONT_SIZE = 10


# =============================================================================
# COLORS
# =============================================================================

# Colors primaris (del tema)
COLOR_PRIMARY = _T["primary"]            # Blau principal
COLOR_PRIMARY_DARK = _T["primary_hover"] # Blau fosc (hover)
COLOR_SECONDARY = _T["info"]             # Accent secundari

# Colors d'estat (del tema — semàfor únic a tota l'app)
COLOR_SUCCESS = _T["success"]
COLOR_SUCCESS_LIGHT = _T["success_bg"]
COLOR_WARNING = _T["warning"]
COLOR_WARNING_LIGHT = _T["warning_bg"]
COLOR_ERROR = _T["error"]
COLOR_ERROR_LIGHT = _T["error_bg"]
COLOR_PENDING = _T["border_strong"]      # Gris pendent
COLOR_CURRENT = _T["primary"]            # Fase actual (= PRIMARY)
COLOR_CAL_BG = _T["primary_light"]       # Fons blau suau per SEQ_CAL
COLOR_CAL_TEXT = _T["primary_hover"]     # Text CAL

# Colors neutres (del tema)
COLOR_TEXT = _T["text_primary"]
COLOR_TEXT_SECONDARY = _T["text_secondary"]
COLOR_TEXT_MUTED = _T["text_muted"]
COLOR_BORDER = _T["border"]
COLOR_BACKGROUND = _T["surface"]


# =============================================================================
# STYLESHEETS
# =============================================================================

# Barra d'avisos (warnings) — tokens del tema
STYLE_WARNING_BAR = f"""
    QFrame {{
        background-color: {COLOR_WARNING_LIGHT};
        border: 1px solid {COLOR_WARNING};
        border-radius: 6px;
    }}
"""

STYLE_WARNING_TEXT = f"color: {COLOR_WARNING};"

# Barra d'error
STYLE_ERROR_BAR = f"""
    QFrame {{
        background-color: {COLOR_ERROR_LIGHT};
        border: 1px solid {COLOR_ERROR};
        border-radius: 6px;
    }}
"""

STYLE_ERROR_TEXT = f"color: {COLOR_ERROR};"

# Barra d'èxit
STYLE_SUCCESS_BAR = f"""
    QFrame {{
        background-color: {COLOR_SUCCESS_LIGHT};
        border: 1px solid {COLOR_SUCCESS};
        border-radius: 6px;
    }}
"""

STYLE_SUCCESS_TEXT = f"color: {COLOR_SUCCESS};"

# Placeholder text
STYLE_PLACEHOLDER = f"color: {COLOR_TEXT_MUTED}; font-size: 14px;"

# Labels
STYLE_LABEL_BOLD = f"font-weight: bold; color: {COLOR_TEXT};"
STYLE_LABEL_SECONDARY = f"color: {COLOR_TEXT_SECONDARY};"
STYLE_LABEL_MUTED = f"color: {COLOR_TEXT_MUTED}; font-size: {SMALL_FONT_SIZE}px;"

# GroupBox
STYLE_GROUPBOX = "QGroupBox { font-weight: bold; }"
STYLE_GROUPBOX_PRIMARY = f"QGroupBox {{ font-weight: bold; color: {COLOR_PRIMARY_DARK}; }}"

# Badges d'impacte per Config Panel
STYLE_BADGE_RETROACTIVE = """
    QLabel {
        background-color: #FEF3C7; color: #92400E;
        border: 1px solid #F59E0B; border-radius: 4px;
        padding: 2px 8px; font-size: 11px; font-weight: bold;
    }
"""

STYLE_BADGE_FUTURE = """
    QLabel {
        background-color: #DBEAFE; color: #1E40AF;
        border: 1px solid #3B82F6; border-radius: 4px;
        padding: 2px 8px; font-size: 11px; font-weight: bold;
    }
"""

STYLE_SECTION_CHANGED = """
    QFrame { border: 2px solid #F59E0B; background-color: #FFFBEB; }
"""


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_title_font():
    """Crea font per títols principals."""
    font = QFont()
    font.setPointSize(TITLE_FONT_SIZE)
    font.setWeight(TITLE_FONT_WEIGHT)
    return font


def create_subtitle_font():
    """Crea font per subtítols."""
    font = QFont()
    font.setPointSize(SUBTITLE_FONT_SIZE)
    font.setWeight(SUBTITLE_FONT_WEIGHT)
    return font


def apply_panel_layout(layout):
    """Aplica marges i spacing estàndard a un layout de panel."""
    layout.setContentsMargins(*PANEL_MARGINS)
    layout.setSpacing(PANEL_SPACING)


def apply_section_layout(layout):
    """Aplica marges i spacing per seccions internes."""
    layout.setContentsMargins(*SECTION_MARGINS)
    layout.setSpacing(COMPACT_SPACING)


# =============================================================================
# EMPTY STATE / PLACEHOLDER
# =============================================================================

STYLE_EMPTY_STATE = f"""
    QFrame {{
        background-color: #f8f9fa;
        border: 2px dashed {COLOR_BORDER};
        border-radius: 8px;
        padding: 24px;
    }}
"""

STYLE_EMPTY_STATE_TEXT = f"""
    color: {COLOR_TEXT_MUTED};
    font-size: 14px;
"""

STYLE_EMPTY_STATE_ICON = f"""
    color: {COLOR_TEXT_MUTED};
    font-size: 32px;
"""


def create_empty_state_widget(icon: str, title: str, description: str, parent=None):
    """
    Crea un widget d'estat buit amb icona, títol i descripció.

    Args:
        icon: Emoji o caràcter per l'icona (ex: "📁", "📊", "🔬")
        title: Títol breu (ex: "No hi ha dades")
        description: Descripció o instrucció (ex: "Importa una seqüència per començar")
        parent: Widget pare (opcional)

    Returns:
        QFrame configurat com a estat buit
    """
    from PySide6.QtWidgets import QFrame, QVBoxLayout, QLabel
    from PySide6.QtCore import Qt

    frame = QFrame(parent)
    frame.setStyleSheet(STYLE_EMPTY_STATE)

    layout = QVBoxLayout(frame)
    layout.setAlignment(Qt.AlignCenter)
    layout.setSpacing(8)

    icon_label = QLabel(icon)
    icon_label.setStyleSheet(STYLE_EMPTY_STATE_ICON)
    icon_label.setAlignment(Qt.AlignCenter)
    layout.addWidget(icon_label)

    title_label = QLabel(title)
    title_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #495057;")
    title_label.setAlignment(Qt.AlignCenter)
    layout.addWidget(title_label)

    desc_label = QLabel(description)
    desc_label.setStyleSheet(STYLE_EMPTY_STATE_TEXT)
    desc_label.setAlignment(Qt.AlignCenter)
    desc_label.setWordWrap(True)
    layout.addWidget(desc_label)

    return frame
