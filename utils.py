"""utils.py — small data helpers shared by the views (no Streamlit UI in here)."""
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from constants import DEFAULT_CHECKIN_DAYS

def get_reminder_frequency():
    from database import get_reminder
    r = get_reminder()
    return int(r["frequency_days"]) if r else DEFAULT_CHECKIN_DAYS

def checkin_days_left(last_submission, freq_days):
    """Days until the next check-in is due (negative = overdue), measured from the
    student's own last submission. None if there is no valid submission time."""
    last = pd.to_datetime(last_submission, errors="coerce")
    if pd.isna(last): return None
    due = last.to_pydatetime() + timedelta(days=int(freq_days))
    return int(np.ceil((due - datetime.now()).total_seconds() / 86400))

def trajectory(scores):
    if len(scores)<2: return "First check-in", "#b09070"
    diff = int(scores[-1]) - int(scores[-2])
    if diff>5:    return f"Score up {diff} pts — let's work on this", "#d4603a"
    elif diff<-5: return f"Score improved {abs(diff)} pts — great progress!", "#48b87a"
    else:         return "Stable since last survey", "#f5a623"

def latest_per_student(df):
    """Most recent submission per roll number (whole row, not per-column last values)."""
    if "roll_number" not in df.columns or "timestamp" not in df.columns:
        return df.copy()
    return df.sort_values("timestamp").drop_duplicates("roll_number", keep="last").reset_index(drop=True)

