"""Small helpers: check-in due dates, latest-per-student, trajectory wording."""
from datetime import datetime, timedelta

import pandas as pd

from utils import checkin_days_left, latest_per_student, trajectory


def test_checkin_due_date_counts_from_the_students_own_survey():
    assert checkin_days_left(datetime.now() - timedelta(days=40), 30) == -10
    assert checkin_days_left(datetime.now() - timedelta(days=40), 60) == 20
    assert checkin_days_left(datetime.now(), 30) == 30


def test_checkin_handles_missing_or_unparseable_dates():
    assert checkin_days_left(None, 30) is None
    assert checkin_days_left("not a date", 30) is None


def test_latest_per_student_keeps_whole_rows():
    """groupby().last() would carry an old note onto the newest score — this must not."""
    df = pd.DataFrame([
        {"roll_number": "1", "timestamp": "2026-01-01 10:00:00", "burnout_score": 40, "student_note": "old worry"},
        {"roll_number": "1", "timestamp": "2026-02-01 10:00:00", "burnout_score": 20, "student_note": ""},
        {"roll_number": "2", "timestamp": "2026-01-15 10:00:00", "burnout_score": 70, "student_note": "help"},
    ])
    latest = latest_per_student(df)
    row = latest[latest.roll_number == "1"].iloc[0]
    assert len(latest) == 2
    assert row.burnout_score == 20 and row.student_note == ""


def test_trajectory_wording():
    assert "First check-in" in trajectory([50])[0]
    assert "up" in trajectory([40, 60])[0]
    assert "improved" in trajectory([60, 40])[0]
    assert "Stable" in trajectory([50, 52])[0]
