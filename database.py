"""
database.py — SQL storage for Burnout Buster.

Built on SQLAlchemy, so the same code runs on:
  • SQLite (default) — a local burnout.db file next to this module, zero setup.
  • PostgreSQL — set DATABASE_URL (env var or .streamlit/secrets.toml), e.g. a free
    Supabase/Neon database, so data survives Streamlit Cloud restarts.

Tables are created automatically on first use.
Old CSV data can be imported with `python migrate_csv_to_db.py`.
"""
import hashlib, hmac, os, uuid
import bcrypt
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import streamlit as st
from sqlalchemy import (create_engine, MetaData, Table, Column, Integer, Float, String,
                        Text, Boolean, DateTime, select, insert, update, delete)
from sqlalchemy.exc import SQLAlchemyError

from scoring import FEATURES

DEFAULT_DB_URL = "sqlite:///" + (Path(__file__).resolve().parent / "burnout.db").as_posix()

# ── SCHEMA ────────────────────────────────────────────────────────────────────
metadata = MetaData()

students = Table("students", metadata,
    Column("roll_number",   String(32),  primary_key=True),
    Column("name",          String(120), nullable=False),
    Column("email",         String(200), nullable=False, default=""),
    Column("college",       String(200), nullable=False, default=""),
    Column("branch",        String(20),  nullable=False),
    Column("section",       String(5),   nullable=False),
    Column("age",           Integer),
    Column("password_hash", String(200), nullable=False),
    Column("created_at",    DateTime,    nullable=False),
)

submissions = Table("submissions", metadata,
    Column("id",                Integer,     primary_key=True, autoincrement=True),
    Column("roll_number",       String(32),  nullable=False, index=True),
    Column("student_name",      String(120), nullable=False, default=""),
    Column("branch",            String(20),  nullable=False, default=""),
    Column("section",           String(5),   nullable=False, default=""),
    Column("timestamp",         DateTime,    nullable=False, index=True),
    Column("burnout_score",     Integer,     nullable=False),
    Column("burnout_risk",      String(10),  nullable=False),
    Column("student_note",      Text,        nullable=False, default=""),
    Column("confidence_high",   Float),
    Column("confidence_medium", Float),
    Column("confidence_low",    Float),
    *[Column(f, Float if f == "cgpa" else Integer) for f in FEATURES],
)

replies = Table("replies", metadata,
    Column("id",                Integer,    primary_key=True, autoincrement=True),
    Column("roll_number",       String(32), nullable=False, index=True),
    Column("counselor_message", Text,       nullable=False),
    Column("timestamp",         DateTime,   nullable=False),
    Column("read_by_student",   Boolean,    nullable=False, default=False),
)

counselor_actions = Table("counselor_actions", metadata,
    Column("roll_number", String(32), primary_key=True),
    Column("status",      String(30), nullable=False),
    Column("notes",       Text,       nullable=False, default=""),
    Column("flagged",     Boolean,    nullable=False, default=False),
    Column("updated_at",  DateTime,   nullable=False),
)

reminders = Table("reminders", metadata,
    Column("id",             Integer,  primary_key=True, autoincrement=True),
    Column("frequency_days", Integer,  nullable=False),
    Column("last_sent",      DateTime, nullable=False),
    Column("next_due",       DateTime, nullable=False),
    Column("created_at",     DateTime, nullable=False),
)

college_records = Table("college_records", metadata,
    Column("id",             Integer,     primary_key=True, autoincrement=True),
    Column("roll_number",    String(32),  nullable=False, index=True),
    Column("name",           String(120), nullable=False, default=""),
    Column("attendance_pct", Float),
    Column("marks_pct",      Float),
    Column("participation",  String(60),  nullable=False, default=""),
    Column("remarks",        Text,        nullable=False, default=""),
    Column("branch",         String(20),  nullable=False),
    Column("section",        String(5),   nullable=False),
    Column("uploaded_at",    DateTime,    nullable=False),
)

# Counselor alerts live in the database (not session state) so an alert raised by a
# student's submission is visible in the counselor's session.
notifications = Table("notifications", metadata,
    Column("id",          String(32),  primary_key=True),
    Column("created_at",  DateTime,    nullable=False, index=True),
    Column("name",        String(120), nullable=False, default=""),
    Column("roll_number", String(32),  nullable=False),
    Column("risk",        String(10),  nullable=False),
    Column("score",       Integer,     nullable=False),
    Column("flagged",     Boolean,     nullable=False, default=False),
    Column("read",        Boolean,     nullable=False, default=False),
)

# ── ENGINE ────────────────────────────────────────────────────────────────────
def database_url() -> str:
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        try: url = str(st.secrets.get("DATABASE_URL", ""))
        except Exception: url = ""
    url = url.strip() or DEFAULT_DB_URL
    # Hosted Postgres providers hand out postgres:// URLs; use the psycopg 3 driver
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix):]
    return url

def make_engine(url: str = None):
    engine = create_engine(url or database_url(), pool_pre_ping=True)
    metadata.create_all(engine)
    return engine

@st.cache_resource
def get_engine():
    return make_engine()

def storage_summary() -> dict:
    """What the app is actually storing data in — shown to counselors so a
    misconfigured DATABASE_URL can't silently leave the app on throwaway storage."""
    url = get_engine().url
    backend = url.get_backend_name()
    is_sqlite = backend == "sqlite"
    return {
        "backend": {"sqlite": "SQLite", "postgresql": "PostgreSQL"}.get(backend, backend.title()),
        "location": url.database if is_sqlite else (url.host or ""),
        "persistent": not is_sqlite,
    }

def _now() -> datetime:
    return datetime.now().replace(microsecond=0)

def _execute(*stmts) -> bool:
    """Run write statements in one transaction. Returns False if it failed."""
    try:
        with get_engine().begin() as conn:
            for stmt in stmts:
                conn.execute(stmt)
        return True
    except SQLAlchemyError:
        return False

def _df(stmt) -> pd.DataFrame:
    with get_engine().connect() as conn:
        return pd.read_sql(stmt, conn)

def _first(stmt) -> dict:
    with get_engine().connect() as conn:
        row = conn.execute(stmt).mappings().first()
    return dict(row) if row else {}

def _text(v) -> str:
    return "" if v is None or pd.isna(v) else str(v).strip()

def _number(v):
    n = pd.to_numeric(v, errors="coerce")
    return None if pd.isna(n) else float(n)

MIN_PASSWORD_LENGTH = 8

def _pwd_bytes(pwd: str) -> bytes:
    return str(pwd).strip().encode("utf-8")[:72]   # bcrypt ignores anything past 72 bytes

def hash_password(pwd: str) -> str:
    return bcrypt.hashpw(_pwd_bytes(pwd), bcrypt.gensalt()).decode()

def check_password(pwd: str, stored: str):
    """Returns (is_correct, needs_upgrade). Accounts created before bcrypt used plain
    SHA-256; those still verify and are re-hashed on the next successful login."""
    stored = str(stored or "").strip()
    if stored.startswith("$2"):
        try:
            return bcrypt.checkpw(_pwd_bytes(pwd), stored.encode()), False
        except ValueError:
            return False, False
    legacy = hashlib.sha256(_pwd_bytes(pwd)).hexdigest()
    return hmac.compare_digest(legacy, stored), True

# ── STUDENT AUTH ──────────────────────────────────────────────────────────────
def get_student(roll: str):
    if not roll or not str(roll).strip():
        return None
    return _first(select(students).where(students.c.roll_number == str(roll).strip())) or None

def student_exists(roll: str) -> bool:
    """Roll numbers are unique — one profile per roll number."""
    return get_student(roll) is not None

def register_student(roll, name, email, college, branch, section, age, password) -> bool:
    """Returns False if the roll number is already registered or the save failed."""
    try:
        with get_engine().begin() as conn:
            conn.execute(insert(students).values(
                roll_number=str(roll).strip(), name=str(name).strip(),
                email=str(email).strip(), college=str(college),
                branch=str(branch), section=str(section), age=int(age),
                password_hash=hash_password(password), created_at=_now()))
        return True
    except SQLAlchemyError:   # IntegrityError = roll number already registered
        return False

def verify_student(roll: str, password: str, branch: str, section: str):
    """Roll number, password, branch and section must all match."""
    if not roll or not password:
        return None
    student = get_student(roll)
    if not student:
        return None
    ok, needs_upgrade = check_password(password, student["password_hash"])
    if not ok:
        return None
    if student["branch"] != str(branch).strip() or student["section"] != str(section).strip():
        return None
    if needs_upgrade:   # re-hash a legacy SHA-256 password with bcrypt
        new_hash = hash_password(password)
        if _execute(update(students)
                    .where(students.c.roll_number == student["roll_number"])
                    .values(password_hash=new_hash)):
            student["password_hash"] = new_hash
    return student

def get_all_students() -> pd.DataFrame:
    return _df(select(*[c for c in students.c if c.name != "password_hash"]))

# ── SUBMISSIONS ───────────────────────────────────────────────────────────────
def save_submission(roll, name, branch, section, features: dict,
                    score: int, risk: str, proba: dict, note: str) -> bool:
    row = {
        "roll_number": str(roll).strip(),
        "student_name": str(name).strip(),
        "branch": str(branch),
        "section": str(section),
        "timestamp": _now(),
        "burnout_score": int(score),
        "burnout_risk": risk,
        "student_note": str(note or ""),
        "confidence_high": round(float(proba.get("High", 0)), 3),
        "confidence_medium": round(float(proba.get("Medium", 0)), 3),
        "confidence_low": round(float(proba.get("Low", 0)), 3),
    }
    for f in FEATURES:
        row[f] = float(features[f]) if f == "cgpa" else int(features[f])
    return _execute(insert(submissions).values(**row))

def get_student_submissions(roll: str) -> pd.DataFrame:
    return _df(select(submissions)
               .where(submissions.c.roll_number == str(roll).strip())
               .order_by(submissions.c.timestamp, submissions.c.id))

def get_all_submissions() -> pd.DataFrame:
    return _df(select(submissions).order_by(submissions.c.timestamp, submissions.c.id))

# ── COUNSELOR ACTIONS ────────────────────────────────────────────────────────
def upsert_counselor_action(roll, status, notes, flagged=False) -> bool:
    roll = str(roll).strip()
    return _execute(
        delete(counselor_actions).where(counselor_actions.c.roll_number == roll),
        insert(counselor_actions).values(roll_number=roll, status=status, notes=str(notes or ""),
                                         flagged=bool(flagged), updated_at=_now()),
    )

def get_counselor_action(roll) -> dict:
    return _first(select(counselor_actions).where(counselor_actions.c.roll_number == str(roll).strip()))

def get_all_counselor_actions() -> dict:
    """{roll_number: action} for every student — one query for the whole dashboard."""
    df = _df(select(counselor_actions))
    return {r["roll_number"]: r for r in df.to_dict("records")}

# ── REPLIES ───────────────────────────────────────────────────────────────────
def save_reply(roll, message) -> bool:
    return _execute(insert(replies).values(
        roll_number=str(roll).strip(), counselor_message=str(message),
        timestamp=_now(), read_by_student=False))

def get_replies(roll) -> pd.DataFrame:
    return _df(select(replies)
               .where(replies.c.roll_number == str(roll).strip())
               .order_by(replies.c.timestamp, replies.c.id))

def mark_replies_read(roll) -> bool:
    return _execute(update(replies)
                    .where(replies.c.roll_number == str(roll).strip(),
                           replies.c.read_by_student.is_(False))
                    .values(read_by_student=True))

# ── REMINDERS ────────────────────────────────────────────────────────────────
def get_reminder() -> dict:
    return _first(select(reminders).order_by(reminders.c.id.desc()).limit(1))

def save_reminder(frequency_days: int) -> bool:
    now = _now()
    return _execute(insert(reminders).values(
        frequency_days=int(frequency_days), last_sent=now,
        next_due=now + timedelta(days=int(frequency_days)), created_at=now))

# ── COLLEGE RECORDS ───────────────────────────────────────────────────────────
def save_college_records(df_records: pd.DataFrame, branch: str, section: str) -> bool:
    """Replace the records for this branch+section with the uploaded ones."""
    now = _now()
    rows = [{
        "roll_number": _text(r.get("roll_number")),
        "name": _text(r.get("name")),
        "attendance_pct": _number(r.get("attendance_pct")),
        "marks_pct": _number(r.get("marks_pct")),
        "participation": _text(r.get("participation")),
        "remarks": _text(r.get("remarks")),
        "branch": branch, "section": section, "uploaded_at": now,
    } for r in df_records.to_dict("records")]
    stmts = [delete(college_records).where(college_records.c.branch == branch,
                                           college_records.c.section == section)]
    if rows:
        stmts.append(insert(college_records).values(rows))
    return _execute(*stmts)

def get_college_records(roll: str = "", branch: str = "", section: str = "") -> pd.DataFrame:
    stmt = select(*[c for c in college_records.c if c.name != "id"])
    if roll:    stmt = stmt.where(college_records.c.roll_number == str(roll).strip())
    if branch:  stmt = stmt.where(college_records.c.branch == branch)
    if section: stmt = stmt.where(college_records.c.section == section)
    return _df(stmt)

# ── COUNSELOR ALERTS ─────────────────────────────────────────────────────────
def add_notification(name, roll, risk, score, flagged=False) -> bool:
    return _execute(insert(notifications).values(
        id=uuid.uuid4().hex[:12], created_at=_now(), name=str(name).strip(),
        roll_number=str(roll).strip(), risk=str(risk), score=int(score),
        flagged=bool(flagged), read=False))

def get_notifications(days: int = 30) -> pd.DataFrame:
    """Alerts from the last `days` days, newest first."""
    return _df(select(notifications)
               .where(notifications.c.created_at >= _now() - timedelta(days=days))
               .order_by(notifications.c.created_at.desc()))

def mark_notifications_read(ids=None) -> bool:
    """Mark the given alert ids read, or all alerts when ids is None."""
    stmt = update(notifications).values(read=True)
    if ids is not None:
        stmt = stmt.where(notifications.c.id.in_(list(ids)))
    return _execute(stmt)
