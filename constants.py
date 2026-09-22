"""constants.py — values shared across views."""

COLLEGES = ["Vivekananda Institute of Professional Studies - Technical Campus (VIPS-TC)", "Other"]
BRANCHES = ["AIDS","CSE","ECE","IT","ME","CE","EEE","Other"]
SECTIONS = ["A","B","C"]

# Internal risk levels -> what people actually read on screen
RISK_LABELS = {"High":"At Risk", "Medium":"Needs Attention", "Low":"Thriving"}

# Used when no counselor has set a check-in schedule yet
DEFAULT_CHECKIN_DAYS = 30

# Warm palette, matched to the stylesheet in ui.py
COLOR_RISK      = "#d4603a"
COLOR_ATTENTION = "#f5a623"
COLOR_THRIVING  = "#48b87a"
COLOR_TEXT      = "#6b4f3f"
COLOR_MUTED     = "#b09070"
COLOR_SURFACE   = "#fffaf6"
COLOR_GRID      = "#f0e6d8"
RISK_COLORS = {"High": COLOR_RISK, "Medium": COLOR_ATTENTION, "Low": COLOR_THRIVING}
