"""
database.py
Handles all Google Sheets read/write operations.
The sheet acts as a simple database — one row per student submission.

Sheet structure (columns):
timestamp | student_name | roll_number | email | [17 feature columns] |
burnout_risk | confidence_high | confidence_low | confidence_medium |
counselor_status | counselor_notes
"""

import json
import pandas as pd
from datetime import datetime
import streamlit as st

# ── Try to import gspread; give helpful error if not installed ─────────────────
try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSPREAD_AVAILABLE = True
except ImportError:
    GSPREAD_AVAILABLE = False

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

FEATURE_COLS = [
    "exams_per_month", "assignments_per_week", "attendance_pressure",
    "cgpa", "backlogs", "study_hours_per_day", "fomo_score",
    "peer_pressure", "family_expectations", "social_media_hrs",
    "rejection_sensitivity", "sleep_hours", "exercise_days",
    "diet_quality", "confidence", "support_system", "mental_health_visits",
]

ALL_COLS = (
    ["timestamp", "student_name", "roll_number", "email"]
    + FEATURE_COLS
    + ["burnout_risk", "confidence_high", "confidence_low", "confidence_medium",
       "counselor_status", "counselor_notes"]
)


def _get_client():
    """Return an authenticated gspread client using service account from secrets."""
    if not GSPREAD_AVAILABLE:
        return None
    try:
        creds_dict = dict(st.secrets["gcp_service_account"])
        creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
        return gspread.authorize(creds)
    except Exception:
        return None


def _get_sheet(client):
    """Open the spreadsheet and return the first worksheet."""
    try:
        sheet_id = st.secrets["google_sheets"]["sheet_id"]
        sh = client.open_by_key(sheet_id)
        return sh.sheet1
    except Exception:
        return None


def ensure_header(sheet):
    """Add header row if the sheet is empty."""
    try:
        values = sheet.get_all_values()
        if not values:
            sheet.append_row(ALL_COLS)
    except Exception:
        pass


def save_submission(student_info: dict, features: dict,
                    prediction: str, probabilities: dict) -> bool:
    """
    Save a student survey submission to Google Sheets.

    Parameters
    ----------
    student_info : dict  with keys: name, roll_number, email
    features     : dict  mapping feature name → value
    prediction   : str   "Low" | "Medium" | "High"
    probabilities: dict  {"High": 0.x, "Low": 0.x, "Medium": 0.x}

    Returns True on success, False on failure.
    """
    client = _get_client()
    if client is None:
        # Fallback: save locally to a CSV
        return _save_local(student_info, features, prediction, probabilities)

    sheet = _get_sheet(client)
    if sheet is None:
        return _save_local(student_info, features, prediction, probabilities)

    ensure_header(sheet)

    row = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        student_info.get("name", ""),
        student_info.get("roll_number", ""),
        student_info.get("email", ""),
    ]
    row += [features.get(f, "") for f in FEATURE_COLS]
    row += [
        prediction,
        round(probabilities.get("High", 0), 3),
        round(probabilities.get("Low", 0), 3),
        round(probabilities.get("Medium", 0), 3),
        "Pending",   # counselor_status
        "",          # counselor_notes
    ]
    try:
        sheet.append_row(row)
        return True
    except Exception:
        return _save_local(student_info, features, prediction, probabilities)


def _save_local(student_info, features, prediction, probabilities) -> bool:
    """Fallback: append submission to a local CSV file."""
    import os
    filepath = "submissions.csv"
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "student_name": student_info.get("name", ""),
        "roll_number": student_info.get("roll_number", ""),
        "email": student_info.get("email", ""),
        **features,
        "burnout_risk": prediction,
        "confidence_high": round(probabilities.get("High", 0), 3),
        "confidence_low": round(probabilities.get("Low", 0), 3),
        "confidence_medium": round(probabilities.get("Medium", 0), 3),
        "counselor_status": "Pending",
        "counselor_notes": "",
    }
    df_new = pd.DataFrame([row])
    if os.path.exists(filepath):
        df_new.to_csv(filepath, mode="a", header=False, index=False)
    else:
        df_new.to_csv(filepath, index=False)
    return True


def load_all_submissions() -> pd.DataFrame:
    """Load all student submissions. Returns a DataFrame."""
    client = _get_client()
    if client:
        sheet = _get_sheet(client)
        if sheet:
            try:
                data = sheet.get_all_records()
                if data:
                    return pd.DataFrame(data)
            except Exception:
                pass

    # Fallback to local CSV
    import os
    if os.path.exists("submissions.csv"):
        return pd.read_csv("submissions.csv")
    return pd.DataFrame(columns=ALL_COLS)


def update_counselor_status(roll_number: str, status: str, notes: str = "") -> bool:
    """Update a student's counselor_status and notes by roll number."""
    client = _get_client()
    if client:
        sheet = _get_sheet(client)
        if sheet:
            try:
                records = sheet.get_all_records()
                for i, rec in enumerate(records, start=2):  # row 1 is header
                    if str(rec.get("roll_number", "")) == str(roll_number):
                        status_col = ALL_COLS.index("counselor_status") + 1
                        notes_col  = ALL_COLS.index("counselor_notes") + 1
                        sheet.update_cell(i, status_col, status)
                        sheet.update_cell(i, notes_col, notes)
                        return True
            except Exception:
                pass

    # Fallback: update local CSV
    import os
    if os.path.exists("submissions.csv"):
        df = pd.read_csv("submissions.csv")
        mask = df["roll_number"].astype(str) == str(roll_number)
        df.loc[mask, "counselor_status"] = status
        df.loc[mask, "counselor_notes"] = notes
        df.to_csv("submissions.csv", index=False)
        return True
    return False
