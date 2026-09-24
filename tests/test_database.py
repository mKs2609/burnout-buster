"""Storage layer: accounts, passwords, submissions, alerts, records, failures."""
import io

import pandas as pd
import pytest

from conftest import STUDENT_PASSWORD


# ── accounts and passwords ────────────────────────────────────────────────────
def test_registration_is_unique_per_roll_number(database, student):
    assert database.register_student("1001", "Someone Else", "", "VIPS-TC", "CSE", "B", 21, "another-pass") is False
    assert database.get_student("1001")["name"] == "Asha Rao"


def test_passwords_are_bcrypt_hashed_and_salted(database, student):
    stored = database.get_student("1001")["password_hash"]
    assert stored.startswith("$2")
    assert STUDENT_PASSWORD not in stored
    assert stored != database.hash_password(STUDENT_PASSWORD)   # different salt each time


def test_login_requires_password_branch_and_section(database, student):
    assert database.verify_student("1001", STUDENT_PASSWORD, "AIDS", "A")
    assert database.verify_student("1001", "wrong-password", "AIDS", "A") is None
    assert database.verify_student("1001", STUDENT_PASSWORD, "CSE", "A") is None
    assert database.verify_student("1001", STUDENT_PASSWORD, "AIDS", "B") is None


def test_legacy_sha256_passwords_still_work_and_upgrade(database, student):
    """Accounts created before bcrypt must keep working and be re-hashed on login."""
    import hashlib
    from sqlalchemy import update
    legacy = hashlib.sha256(b"old-password").hexdigest()
    with database.get_engine().begin() as conn:
        conn.execute(update(database.students)
                     .where(database.students.c.roll_number == "1001")
                     .values(password_hash=legacy))

    assert database.verify_student("1001", "old-password", "AIDS", "A")
    assert database.get_student("1001")["password_hash"].startswith("$2")
    assert database.verify_student("1001", "old-password", "AIDS", "A")   # still works after upgrade


def test_very_long_passwords_do_not_crash(database, student):
    assert database.verify_student("1001", "x" * 500, "AIDS", "A") is None


def test_student_list_never_exposes_password_hashes(database, student):
    assert "password_hash" not in database.get_all_students().columns


# ── submissions ───────────────────────────────────────────────────────────────
def _submit(database, roll="1001", score=50, risk="Medium", note=""):
    features = {f: 5 for f in database.FEATURES}
    features["cgpa"] = 7.5
    return database.save_submission(roll, "Asha Rao", "AIDS", "A", features, score, risk,
                                    {"Low": 0.2, "Medium": 0.6, "High": 0.2}, note)


def test_submissions_are_stored_and_returned_in_order(database, student):
    for score in (30, 50, 70):
        assert _submit(database, score=score)
    scores = database.get_student_submissions("1001")["burnout_score"].tolist()
    assert scores == [30, 50, 70]


def test_submissions_are_scoped_to_one_student(database, student):
    database.register_student("1002", "Other", "", "VIPS-TC", "CSE", "B", 20, "pass-word-2")
    _submit(database, roll="1001")
    _submit(database, roll="1002")
    assert len(database.get_student_submissions("1001")) == 1


# ── counselor alerts ──────────────────────────────────────────────────────────
def test_alerts_persist_outside_session_state(database):
    assert database.add_notification("Asha Rao", "1001", "High", 80, flagged=True)
    alerts = database.get_notifications()
    assert len(alerts) == 1
    assert alerts.iloc[0]["read"] in (False, 0)


def test_marking_alerts_read(database):
    database.add_notification("A", "1", "High", 80)
    database.add_notification("B", "2", "Medium", 50)
    ids = database.get_notifications()["id"].tolist()
    assert database.mark_notifications_read([ids[0]])
    read_flags = dict(zip(database.get_notifications()["id"], database.get_notifications()["read"]))
    assert bool(read_flags[ids[0]]) and not bool(read_flags[ids[1]])
    assert database.mark_notifications_read()
    assert all(bool(r) for r in database.get_notifications()["read"])


def test_old_alerts_drop_out_of_the_window(database):
    from datetime import datetime, timedelta
    from sqlalchemy import update
    database.add_notification("Old", "1", "High", 80)
    with database.get_engine().begin() as conn:
        conn.execute(update(database.notifications).values(created_at=datetime.now() - timedelta(days=45)))
    assert database.get_notifications(days=30).empty


# ── replies ───────────────────────────────────────────────────────────────────
def test_replies_are_marked_read_for_that_student_only(database):
    database.save_reply("1001", "Checking in")
    database.save_reply("1002", "Hello")
    database.mark_replies_read("1001")
    assert all(bool(r) for r in database.get_replies("1001")["read_by_student"])
    assert not any(bool(r) for r in database.get_replies("1002")["read_by_student"])


# ── college records ───────────────────────────────────────────────────────────
CSV = "roll_number,name,attendance_pct,marks_pct,participation,remarks\n01217711924,Ravi,90,80,Active,Good\n"


def test_upload_keeps_leading_zeros_in_roll_numbers(database):
    records = pd.read_csv(io.StringIO(CSV), dtype=str)
    assert database.save_college_records(records, "CSE", "A")
    found = database.get_college_records(roll="01217711924")
    assert len(found) == 1 and found.iloc[0]["attendance_pct"] == 90.0


def test_re_uploading_a_section_replaces_its_records(database):
    records = pd.read_csv(io.StringIO(CSV), dtype=str)
    database.save_college_records(records, "CSE", "A")
    database.save_college_records(records, "CSE", "A")
    assert len(database.get_college_records(branch="CSE", section="A")) == 1


# ── counselor actions and reminders ───────────────────────────────────────────
def test_counselor_action_is_upserted_not_duplicated(database):
    database.upsert_counselor_action("1001", "Pending", "first note")
    database.upsert_counselor_action("1001", "Contacted", "second note")
    action = database.get_counselor_action("1001")
    assert action["status"] == "Contacted" and action["notes"] == "second note"
    assert len(database.get_all_counselor_actions()) == 1


def test_latest_reminder_wins(database):
    database.save_reminder(30)
    database.save_reminder(14)
    assert database.get_reminder()["frequency_days"] == 14


# ── failure handling ──────────────────────────────────────────────────────────
@pytest.fixture
def broken_db(database, monkeypatch):
    """Point the engine at an unwritable path so every write fails."""
    from sqlalchemy import create_engine
    monkeypatch.setattr(database, "get_engine",
                        lambda: create_engine("sqlite:///Z:/nonexistent/dir/x.db"))
    return database


def test_writes_report_failure_instead_of_pretending(broken_db):
    features = {f: 5 for f in broken_db.FEATURES}
    assert broken_db.save_reply("1001", "hi") is False
    assert broken_db.save_submission("1001", "A", "AIDS", "A", features, 50, "Medium", {}, "") is False
    assert broken_db.upsert_counselor_action("1001", "Contacted", "") is False
    assert broken_db.add_notification("A", "1001", "High", 80) is False
    assert broken_db.register_student("9", "X", "", "c", "CSE", "A", 20, "pass-word") is False


def test_storage_summary_reports_the_backend_in_use(database):
    """Counselors are told when the app is on throwaway storage."""
    summary = database.storage_summary()
    assert summary["backend"] == "SQLite" and summary["persistent"] is False
