"""
database.py — Local CSV data layer for Burnout Buster v4
Supabase is used when available, falls back to local CSV silently.
"""
import hashlib, os, json
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

STUDENTS_CSV    = "students.csv"
SUBMISSIONS_CSV = "submissions.csv"
REPLIES_CSV     = "replies.csv"
REMINDERS_CSV   = "reminders.csv"

def hash_password(pwd: str) -> str:
    return hashlib.sha256(str(pwd).strip().encode("utf-8")).hexdigest()

@st.cache_resource
def get_client():
    try:
        from supabase import create_client
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        return create_client(url, key)
    except Exception:
        return None

# ── LOCAL CSV HELPERS ─────────────────────────────────────────────────────────
def _read(path) -> pd.DataFrame:
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, dtype=str)
            return df
        except Exception:
            pass
    return pd.DataFrame()

def _append(path, row: dict) -> bool:
    try:
        df_new = pd.DataFrame([row])
        if os.path.exists(path):
            df_new.to_csv(path, mode="a", header=False, index=False)
        else:
            df_new.to_csv(path, index=False)
        return True
    except Exception:
        return False

# ── STUDENT AUTH ──────────────────────────────────────────────────────────────
def student_exists(roll: str) -> bool:
    if not roll or not roll.strip():
        return False
    client = get_client()
    if client:
        try:
            r = client.table("students").select("roll_number").eq("roll_number", roll.strip()).execute()
            return len(r.data) > 0
        except Exception:
            pass
    df = _read(STUDENTS_CSV)
    if df.empty or "roll_number" not in df.columns:
        return False
    return roll.strip() in df["roll_number"].str.strip().values

def register_student(roll, name, email, college, branch, section, age, password) -> bool:
    roll = str(roll).strip()
    if student_exists(roll):
        return True  # already registered
    data = {
        "roll_number":   roll,
        "name":          str(name).strip(),
        "email":         str(email).strip(),
        "college":       str(college),
        "branch":        str(branch),
        "section":       str(section),
        "age":           str(age),
        "password_hash": hash_password(str(password)),
        "created_at":    datetime.now().isoformat(),
    }
    client = get_client()
    if client:
        try:
            client.table("students").insert(data).execute()
            return True
        except Exception:
            pass
    return _append(STUDENTS_CSV, data)

def verify_student(roll: str, password: str):
    if not roll or not password:
        return None
    roll = str(roll).strip()
    h    = hash_password(str(password))
    client = get_client()
    if client:
        try:
            r = client.table("students").select("*").eq("roll_number", roll).execute()
            if r.data:
                s = r.data[0]
                if s.get("password_hash") == h:
                    return s
            return None
        except Exception:
            pass
    # Local CSV
    df = _read(STUDENTS_CSV)
    if df.empty or "roll_number" not in df.columns or "password_hash" not in df.columns:
        return None
    match = df[
        (df["roll_number"].str.strip() == roll) &
        (df["password_hash"].str.strip() == h)
    ]
    if match.empty:
        return None
    return match.iloc[0].to_dict()

def get_student(roll: str):
    if not roll:
        return None
    roll = str(roll).strip()
    client = get_client()
    if client:
        try:
            r = client.table("students").select("*").eq("roll_number", roll).execute()
            return r.data[0] if r.data else None
        except Exception:
            pass
    df = _read(STUDENTS_CSV)
    if df.empty or "roll_number" not in df.columns:
        return None
    match = df[df["roll_number"].str.strip() == roll]
    return match.iloc[0].to_dict() if not match.empty else None

def get_all_students() -> pd.DataFrame:
    client = get_client()
    if client:
        try:
            r = client.table("students").select("*").execute()
            return pd.DataFrame(r.data) if r.data else pd.DataFrame()
        except Exception:
            pass
    return _read(STUDENTS_CSV)

# ── SUBMISSIONS ───────────────────────────────────────────────────────────────
def save_submission(roll, name, features: dict, score: int, risk: str, proba: dict, note: str) -> bool:
    data = {
        "roll_number":       str(roll).strip(),
        "student_name":      str(name).strip(),
        "timestamp":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "burnout_score":     str(score),
        "burnout_risk":      risk,
        "student_note":      str(note),
        "confidence_high":   str(round(proba.get("High",   0), 3)),
        "confidence_medium": str(round(proba.get("Medium", 0), 3)),
        "confidence_low":    str(round(proba.get("Low",    0), 3)),
    }
    for k, v in features.items():
        data[k] = str(v)
    client = get_client()
    if client:
        try:
            supabase_data = {k: (float(v) if k in ["cgpa","confidence_high","confidence_medium","confidence_low"]
                                 else int(v) if k in FEATURES and k != "cgpa"
                                 else v)
                             for k, v in data.items()}
            client.table("submissions").insert(supabase_data).execute()
            return True
        except Exception:
            pass
    return _append(SUBMISSIONS_CSV, data)

def get_student_submissions(roll: str) -> pd.DataFrame:
    if not roll:
        return pd.DataFrame()
    roll = str(roll).strip()
    client = get_client()
    if client:
        try:
            r = client.table("submissions").select("*").eq("roll_number", roll).order("timestamp").execute()
            return pd.DataFrame(r.data) if r.data else pd.DataFrame()
        except Exception:
            pass
    df = _read(SUBMISSIONS_CSV)
    if df.empty or "roll_number" not in df.columns:
        return pd.DataFrame()
    result = df[df["roll_number"].str.strip() == roll].copy()
    if "burnout_score" in result.columns:
        result["burnout_score"] = pd.to_numeric(result["burnout_score"], errors="coerce").fillna(0).astype(int)
    return result

def get_all_submissions() -> pd.DataFrame:
    client = get_client()
    if client:
        try:
            r = client.table("submissions").select("*").order("timestamp", desc=True).execute()
            return pd.DataFrame(r.data) if r.data else pd.DataFrame()
        except Exception:
            pass
    df = _read(SUBMISSIONS_CSV)
    if not df.empty and "burnout_score" in df.columns:
        df["burnout_score"] = pd.to_numeric(df["burnout_score"], errors="coerce").fillna(0).astype(int)
    return df

# ── COUNSELOR ACTIONS ────────────────────────────────────────────────────────
def upsert_counselor_action(roll, status, notes, flagged=False) -> bool:
    roll = str(roll).strip()
    client = get_client()
    data = {"roll_number": roll, "status": status, "notes": notes,
            "flagged": str(flagged), "updated_at": datetime.now().isoformat()}
    if client:
        try:
            client.table("counselor_actions").upsert(data, on_conflict="roll_number").execute()
            return True
        except Exception:
            pass
    # Local: read, update/insert, write back
    path = "counselor_actions.csv"
    df = _read(path)
    if not df.empty and "roll_number" in df.columns:
        df = df[df["roll_number"].str.strip() != roll]
    new_row = pd.DataFrame([data])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(path, index=False)
    return True

def get_counselor_action(roll) -> dict:
    roll = str(roll).strip()
    client = get_client()
    if client:
        try:
            r = client.table("counselor_actions").select("*").eq("roll_number", roll).execute()
            return r.data[0] if r.data else {}
        except Exception:
            pass
    df = _read("counselor_actions.csv")
    if df.empty or "roll_number" not in df.columns:
        return {}
    match = df[df["roll_number"].str.strip() == roll]
    return match.iloc[0].to_dict() if not match.empty else {}

# ── REPLIES ───────────────────────────────────────────────────────────────────
def save_reply(roll, message) -> bool:
    data = {
        "roll_number":      str(roll).strip(),
        "counselor_message": str(message),
        "timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "read_by_student":  "False",
    }
    client = get_client()
    if client:
        try:
            client.table("replies").insert(data).execute()
            return True
        except Exception:
            pass
    return _append(REPLIES_CSV, data)

def get_replies(roll) -> pd.DataFrame:
    roll = str(roll).strip()
    client = get_client()
    if client:
        try:
            r = client.table("replies").select("*").eq("roll_number", roll).order("timestamp").execute()
            return pd.DataFrame(r.data) if r.data else pd.DataFrame()
        except Exception:
            pass
    df = _read(REPLIES_CSV)
    if df.empty or "roll_number" not in df.columns:
        return pd.DataFrame()
    return df[df["roll_number"].str.strip() == roll]

def mark_replies_read(roll):
    roll = str(roll).strip()
    client = get_client()
    if client:
        try:
            client.table("replies").update({"read_by_student": True}).eq("roll_number", roll).execute()
            return
        except Exception:
            pass
    df = _read(REPLIES_CSV)
    if not df.empty and "roll_number" in df.columns:
        df.loc[df["roll_number"].str.strip() == roll, "read_by_student"] = "True"
        df.to_csv(REPLIES_CSV, index=False)

# ── REMINDERS ────────────────────────────────────────────────────────────────
def get_reminder() -> dict:
    client = get_client()
    if client:
        try:
            r = client.table("reminders").select("*").order("created_at", desc=True).limit(1).execute()
            return r.data[0] if r.data else {}
        except Exception:
            pass
    df = _read(REMINDERS_CSV)
    if df.empty:
        return {}
    return df.iloc[-1].to_dict()

def save_reminder(frequency_days: int) -> bool:
    next_due = (datetime.now() + timedelta(days=frequency_days)).isoformat()
    data = {
        "frequency_days": str(frequency_days),
        "last_sent":      datetime.now().isoformat(),
        "next_due":       next_due,
        "created_at":     datetime.now().isoformat(),
    }
    client = get_client()
    if client:
        try:
            client.table("reminders").insert(data).execute()
            return True
        except Exception:
            pass
    return _append(REMINDERS_CSV, data)
