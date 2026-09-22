"""
Shared test setup.

Every test runs against a throwaway SQLite database in a temp directory, so tests
never touch burnout.db or the local_*.csv files.
"""
import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Must be set before database.py builds its engine
_TMP_DB = Path(tempfile.mkdtemp(prefix="burnout-tests-")) / "test.db"
os.environ["DATABASE_URL"] = "sqlite:///" + _TMP_DB.as_posix()

import database as db  # noqa: E402  (import after DATABASE_URL is set)

COUNSELOR_PASSWORD = "test-counselor-pw"
STUDENT_PASSWORD = "student-pass-1"


@pytest.fixture(autouse=True)
def clean_db():
    """Empty every table before each test."""
    from sqlalchemy import delete
    with db.get_engine().begin() as conn:
        for table in reversed(db.metadata.sorted_tables):
            conn.execute(delete(table))
    yield


@pytest.fixture
def database():
    return db


@pytest.fixture
def student(database):
    """A registered student: roll 1001, branch AIDS, section A."""
    assert database.register_student("1001", "Asha Rao", "asha@vips.edu", "VIPS-TC",
                                     "AIDS", "A", 20, STUDENT_PASSWORD)
    return {"roll": "1001", "password": STUDENT_PASSWORD, "branch": "AIDS", "section": "A"}


@pytest.fixture
def app():
    """Factory returning a freshly run AppTest with counselor secrets configured."""
    from streamlit.testing.v1 import AppTest

    def _run(**session_state):
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
        at.secrets["COUNSELOR_PASSWORD"] = COUNSELOR_PASSWORD
        for k, v in session_state.items():
            at.session_state[k] = v
        return at.run()

    return _run


@pytest.fixture(scope="session")
def model_and_meta():
    import json
    import joblib
    return joblib.load(ROOT / "burnout_model.pkl"), json.loads((ROOT / "model_meta.json").read_text())


@pytest.fixture
def healthy_answers(model_and_meta):
    """A typical thriving student's answers (the model's reference profile)."""
    _, meta = model_and_meta
    return dict(meta["reference_profile"])
