"""
database.py — GitHub-based persistent storage for Burnout Buster v5
Data is stored as CSV files in a private GitHub repository.
Never resets, never pauses, completely free forever.
"""
import hashlib, json, base64, os
from datetime import datetime, timedelta
import pandas as pd
import streamlit as st

FEATURES = [
    "exams_per_month","assignments_per_week","attendance_pressure","cgpa",
    "backlogs","study_hours_per_day","fomo_score","peer_pressure",
    "family_expectations","social_media_hrs","rejection_sensitivity",
    "sleep_hours","exercise_days","diet_quality","confidence",
    "support_system","mental_health_visits",
]

FILES = {
    "students":          "data/students.csv",
    "submissions":       "data/submissions.csv",
    "replies":           "data/replies.csv",
    "counselor_actions": "data/counselor_actions.csv",
    "reminders":         "data/reminders.csv",
    "college_records":   "data/college_records.csv",
}

def hash_password(pwd: str) -> str:
    return hashlib.sha256(str(pwd).strip().encode("utf-8")).hexdigest()

# ── GITHUB API ────────────────────────────────────────────────────────────────
def _gh_headers():
    try:
        token = st.secrets["GITHUB_TOKEN"]
        return {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
    except Exception:
        return {}

def _gh_repo():
    try:
        return st.secrets["GITHUB_REPO"]
    except Exception:
        return ""

def _get_file(path: str):
    """Get file content and SHA from GitHub."""
    import requests
    repo = _gh_repo()
    headers = _gh_headers()
    if not repo or not headers:
        return None, None
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            data = r.json()
            content = base64.b64decode(data["content"]).decode("utf-8")
            return content, data["sha"]
        return None, None
    except Exception:
        return None, None

def _put_file(path: str, content: str, sha: str = None, message: str = "update data"):
    """Create or update a file on GitHub."""
    import requests
    repo = _gh_repo()
    headers = _gh_headers()
    if not repo or not headers:
        return False
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    payload = {
        "message": message,
        "content": base64.b64encode(content.encode("utf-8")).decode("utf-8"),
    }
    if sha:
        payload["sha"] = sha
    try:
        r = requests.put(url, headers=headers, json=payload, timeout=15)
        return r.status_code in (200, 201)
    except Exception:
        return False


def _read_df(file_key: str) -> pd.DataFrame:
    """Read CSV — tries GitHub first, falls back to local file."""
    # Try local first (faster for local development)
    local_path = f"local_{file_key}.csv"
    if os.path.exists(local_path):
        try:
            return pd.read_csv(local_path, dtype=str)
        except Exception:
            pass
    # Try GitHub
    path = FILES[file_key]
    content, _ = _get_file(path)
    if content and content.strip():
        try:
            from io import StringIO
            return pd.read_csv(StringIO(content), dtype=str)
        except Exception:
            pass
    return pd.DataFrame()

def _write_df(file_key: str, df: pd.DataFrame, message: str = "update") -> bool:
    """Write DataFrame — saves locally AND to GitHub."""
    # Always save locally
    local_path = f"local_{file_key}.csv"
    df.to_csv(local_path, index=False)
    # Try GitHub
    path = FILES[file_key]
    _, sha = _get_file(path)
    content = df.to_csv(index=False)
    return _put_file(path, content, sha, message)

def _append_row(file_key: str, row: dict) -> bool:
    """Append a row — saves locally AND to GitHub."""
    df = _read_df(file_key)
    new_row = pd.DataFrame([row])
    df = pd.concat([df, new_row], ignore_index=True) if not df.empty else new_row
    # Save locally
    local_path = f"local_{file_key}.csv"
    df.to_csv(local_path, index=False)
    # Save to GitHub (for live site)
    _write_df(file_key, df, f"add {file_key} record")
    return True

# ── STUDENT AUTH ──────────────────────────────────────────────────────────────
def student_exists(roll: str, branch: str = "", section: str = "") -> bool:
    if not roll or not roll.strip():
        return False
    df = _read_df("students")
    if df.empty or "roll_number" not in df.columns:
        return False
    mask = df["roll_number"].str.strip() == roll.strip()
    if branch:
        mask = mask & (df.get("branch", pd.Series([""] * len(df))).str.strip() == branch.strip())
    if section:
        mask = mask & (df.get("section", pd.Series([""] * len(df))).str.strip() == section.strip())
    return mask.any()

def register_student(roll, name, email, college, branch, section, age, password) -> bool:
    if student_exists(roll.strip(), branch, section):
        return True
    row = {
        "roll_number": str(roll).strip(),
        "name": str(name).strip(),
        "email": str(email).strip(),
        "college": str(college),
        "branch": str(branch),
        "section": str(section),
        "age": str(age),
        "password_hash": hash_password(str(password)),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    return _append_row("students", row)

def verify_student(roll: str, password: str, branch: str = "", section: str = ""):
    if not roll or not password:
        return None
    roll = str(roll).strip()
    h = hash_password(str(password))
    df = _read_df("students")
    if df.empty or "roll_number" not in df.columns or "password_hash" not in df.columns:
        return None
    # Try with branch and section first
    mask = (df["roll_number"].str.strip() == roll) & (df["password_hash"].str.strip() == h)
    if branch:
        mask_full = mask & (df.get("branch", pd.Series([""] * len(df))).str.strip() == branch.strip())
        match = df[mask_full]
        if not match.empty:
            return match.iloc[0].to_dict()
    # Fallback — match by roll + password only
    match = df[mask]
    return match.iloc[0].to_dict() if not match.empty else None

def get_all_students() -> pd.DataFrame:
    return _read_df("students")

def get_student(roll: str):
    df = _read_df("students")
    if df.empty or "roll_number" not in df.columns:
        return None
    match = df[df["roll_number"].str.strip() == str(roll).strip()]
    return match.iloc[0].to_dict() if not match.empty else None

# ── SUBMISSIONS ───────────────────────────────────────────────────────────────
def save_submission(roll, name, branch, section, features: dict,
                    score: int, risk: str, proba: dict, note: str) -> bool:
    row = {
        "roll_number": str(roll).strip(),
        "student_name": str(name).strip(),
        "branch": str(branch),
        "section": str(section),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "burnout_score": str(score),
        "burnout_risk": risk,
        "student_note": str(note),
        "confidence_high": str(round(proba.get("High", 0), 3)),
        "confidence_medium": str(round(proba.get("Medium", 0), 3)),
        "confidence_low": str(round(proba.get("Low", 0), 3)),
    }
    for k, v in features.items():
        row[k] = str(v)
    return _append_row("submissions", row)

def get_student_submissions(roll: str) -> pd.DataFrame:
    df = _read_df("submissions")
    if df.empty or "roll_number" not in df.columns:
        return pd.DataFrame()
    result = df[df["roll_number"].str.strip() == str(roll).strip()].copy()
    if "burnout_score" in result.columns:
        result["burnout_score"] = pd.to_numeric(result["burnout_score"], errors="coerce").fillna(0).astype(int)
    return result.reset_index(drop=True)

def get_all_submissions() -> pd.DataFrame:
    df = _read_df("submissions")
    if not df.empty and "burnout_score" in df.columns:
        df["burnout_score"] = pd.to_numeric(df["burnout_score"], errors="coerce").fillna(0).astype(int)
    return df

# ── COUNSELOR ACTIONS ────────────────────────────────────────────────────────
def upsert_counselor_action(roll, status, notes, flagged=False) -> bool:
    df = _read_df("counselor_actions")
    roll = str(roll).strip()
    row = {
        "roll_number": roll,
        "status": status,
        "notes": str(notes),
        "flagged": str(flagged),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    if not df.empty and "roll_number" in df.columns:
        df = df[df["roll_number"].str.strip() != roll]
    new_row = pd.DataFrame([row])
    df = pd.concat([df, new_row], ignore_index=True) if not df.empty else new_row
    return _write_df("counselor_actions", df, f"update action {roll}")

def get_counselor_action(roll) -> dict:
    df = _read_df("counselor_actions")
    if df.empty or "roll_number" not in df.columns:
        return {}
    match = df[df["roll_number"].str.strip() == str(roll).strip()]
    return match.iloc[0].to_dict() if not match.empty else {}

# ── REPLIES ───────────────────────────────────────────────────────────────────
def save_reply(roll, message) -> bool:
    row = {
        "roll_number": str(roll).strip(),
        "counselor_message": str(message),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "read_by_student": "False",
    }
    return _append_row("replies", row)

def get_replies(roll) -> pd.DataFrame:
    df = _read_df("replies")
    if df.empty or "roll_number" not in df.columns:
        return pd.DataFrame()
    return df[df["roll_number"].str.strip() == str(roll).strip()].reset_index(drop=True)

def mark_replies_read(roll):
    df = _read_df("replies")
    if df.empty or "roll_number" not in df.columns:
        return
    df.loc[df["roll_number"].str.strip() == str(roll).strip(), "read_by_student"] = "True"
    _write_df("replies", df, f"mark replies read {roll}")

# ── REMINDERS ────────────────────────────────────────────────────────────────
def get_reminder() -> dict:
    df = _read_df("reminders")
    if df.empty:
        return {}
    return df.iloc[-1].to_dict()

def save_reminder(frequency_days: int) -> bool:
    next_due = (datetime.now() + timedelta(days=frequency_days)).strftime("%Y-%m-%d %H:%M:%S")
    row = {
        "frequency_days": str(frequency_days),
        "last_sent": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "next_due": next_due,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    return _append_row("reminders", row)

# ── COLLEGE RECORDS ───────────────────────────────────────────────────────────
def save_college_records(df_records: pd.DataFrame, branch: str, section: str) -> bool:
    """Save college academic records uploaded by counselor."""
    df_records["branch"] = branch
    df_records["section"] = section
    df_records["uploaded_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    existing = _read_df("college_records")
    if not existing.empty and "roll_number" in existing.columns and "roll_number" in df_records.columns:
        # Remove old records for this branch+section
        mask = ~((existing.get("branch", "") == branch) & (existing.get("section", "") == section))
        existing = existing[mask]
        combined = pd.concat([existing, df_records], ignore_index=True)
    else:
        combined = df_records
    return _write_df("college_records", combined, f"upload records {branch}-{section}")

def get_college_records(roll: str = "", branch: str = "", section: str = "") -> pd.DataFrame:
    df = _read_df("college_records")
    if df.empty:
        return pd.DataFrame()
    if roll:
        df = df[df.get("roll_number", pd.Series()).str.strip() == str(roll).strip()]
    if branch:
        df = df[df.get("branch", pd.Series()) == branch]
    if section:
        df = df[df.get("section", pd.Series()) == section]
    return df.reset_index(drop=True)
