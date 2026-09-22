"""End-to-end flows through the Streamlit UI (streamlit.testing AppTest)."""
from datetime import datetime, timedelta

import pytest

from conftest import COUNSELOR_PASSWORD, STUDENT_PASSWORD

NEW_PASSWORD = "brand-new-pass"

# survey widget key per model feature, in the order they appear in the form
FEATURE_KEYS = {
    "exams_per_month": "q1", "assignments_per_week": "q2", "attendance_pressure": "q3",
    "cgpa": "q4", "backlogs": "q5", "study_hours_per_day": "q6", "fomo_score": "q7",
    "peer_pressure": "q8", "family_expectations": "q9", "social_media_hrs": "q10",
    "rejection_sensitivity": "q11", "mental_health_visits": "q12", "sleep_hours": "q13",
    "exercise_days": "q14", "diet_quality": "q15", "confidence": "q16", "support_system": "q17",
}


def answers_from(profile, **overrides):
    """Turn a {feature: value} profile into {widget key: value} for the survey form."""
    return {FEATURE_KEYS[f]: v for f, v in {**profile, **overrides}.items() if f in FEATURE_KEYS}


def _fill_survey(at, name="Test Student", roll="2001", password=None, existing_password=None,
                 branch="AIDS", section="A", answers=None):
    at.text_input(key="s_name").input(name)
    at.text_input(key="s_roll").input(roll).run()
    if password is not None:
        at.text_input(key="s_pwd").input(password)
        at.text_input(key="s_pwd2").input(password)
    if existing_password is not None:
        at.text_input(key="s_pwd_existing").input(existing_password)
    at.selectbox(key="s_branch").select(branch)
    at.selectbox(key="s_section").select(section)
    for key, value in (answers or {}).items():
        widget = next((w for w in at.select_slider if w.key == key), None) \
                 or next(w for w in at.slider if w.key == key)
        widget.set_value(value)
    at.checkbox[0].check()
    return at.button(key="submit_survey").click().run()


def _errors(at):
    return [e.value for e in at.error]


def _page_text(at):
    return " ".join(m.value for m in at.markdown)


def _sign_in_counselor(app, password=COUNSELOR_PASSWORD):
    at = app()
    at.text_input(key="c_pwd").input(password)
    return at.button(key="c_login").click().run()


# ── smoke ─────────────────────────────────────────────────────────────────────
def test_app_boots_with_all_tabs(app):
    at = app()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Home", "Take Survey", "My Portal",
                                          "Counselor", "Analytics", "About the Model"]


def test_home_makes_no_unsourced_claims(app):
    text = _page_text(app())
    for claim in ("60-70%", "25%", "2 weeks", "100% Confidential"):
        assert claim not in text


# ── survey ────────────────────────────────────────────────────────────────────
def test_new_student_can_register_and_submit(app, database):
    at = _fill_survey(app(), roll="2001", password=NEW_PASSWORD)
    assert not _errors(at) and not at.exception
    assert database.student_exists("2001")
    assert len(database.get_student_submissions("2001")) == 1


def test_short_passwords_are_rejected(app, database):
    at = _fill_survey(app(), roll="2002", password="short")
    assert any("at least 8" in e for e in _errors(at))
    assert not database.student_exists("2002")


def test_existing_roll_number_requires_the_password(app, database, student):
    at = _fill_survey(app(), roll="1001", name="Impostor")
    assert any("portal password" in e for e in _errors(at))
    assert len(database.get_student_submissions("1001")) == 0


def test_wrong_password_or_branch_is_rejected(app, database, student):
    for kwargs in ({"existing_password": "not-the-password"},
                   {"existing_password": STUDENT_PASSWORD, "branch": "CSE"}):
        at = _fill_survey(app(), roll="1001", name="Impostor", **kwargs)
        assert any("doesn't match" in e for e in _errors(at))
    assert len(database.get_student_submissions("1001")) == 0


def test_returning_student_keeps_their_profile_name(app, database, student):
    _fill_survey(app(), roll="1001", name="Impostor", existing_password=STUDENT_PASSWORD)
    assert database.get_student_submissions("1001").iloc[-1]["student_name"] == "Asha Rao"


def test_result_explains_which_answers_shaped_the_score(app, database, healthy_answers):
    at = _fill_survey(app(), roll="2003", password=NEW_PASSWORD,
                      answers=answers_from(healthy_answers, sleep_hours=3))
    text = _page_text(at)
    assert "What's Shaping Your Score" in text
    assert "Sleep" in text
    assert "SLEEP FIRST" in text.upper()        # the action plan follows the driver


def test_safety_rule_lifts_an_otherwise_healthy_profile(app, database, healthy_answers):
    """3h sleep on an otherwise thriving profile: the model alone scores it Low."""
    at = _fill_survey(app(), roll="2006", password=NEW_PASSWORD,
                      answers=answers_from(healthy_answers, sleep_hours=3))
    assert any("4 hours or less" in i.value for i in at.info)
    assert database.get_student_submissions("2006").iloc[-1]["burnout_risk"] != "Low"


def test_at_risk_result_shows_helplines(app, database):
    severe = {"q1": 8, "q2": 12, "q3": 10, "q5": 8, "q7": 10, "q8": 10, "q9": 10,
              "q10": 12, "q11": 10, "q13": 3, "q14": 0, "q15": 1, "q16": 1, "q17": 1}
    at = _fill_survey(app(), roll="2004", password=NEW_PASSWORD, answers=severe)
    assert "Helplines" in _page_text(at)
    assert database.get_student_submissions("2004").iloc[-1]["burnout_risk"] == "High"


# ── counselor ─────────────────────────────────────────────────────────────────
def test_counselor_login(app):
    assert not _sign_in_counselor(app, "wrong-password").session_state.counselor_logged_in
    at = _sign_in_counselor(app)
    assert at.session_state.counselor_logged_in and not at.exception


def test_the_old_hardcoded_password_no_longer_works(app):
    at = _sign_in_counselor(app, "ProjectAlpha01")
    assert not at.session_state.counselor_logged_in


def test_alert_raised_by_a_student_reaches_the_counselors_session(app, database):
    _fill_survey(app(), roll="2005", password=NEW_PASSWORD,
                 answers={"q13": 3, "q5": 6, "q16": 2, "q17": 2})
    assert len(database.get_notifications()) == 1
    at = _sign_in_counselor(app)
    assert any(e.label.startswith("Alerts") and "unread" in e.label for e in at.expander)


def test_counselor_counts_add_up_to_the_student_total(app, database, student):
    features = {f: 5 for f in database.FEATURES}
    for score, risk in [(20, "Low"), (50, "Medium"), (80, "High")]:   # same student, 3 surveys
        database.save_submission("1001", "Asha Rao", "AIDS", "A", features, score, risk, {}, "")
    at = _sign_in_counselor(app)
    text = _page_text(at)
    import re
    counts = [int(n) for n in re.findall(
        r'stat-num[^>]*>(\d+)</div><div class="stat-lbl">(?:Students|At Risk|Needs Attention|Thriving)', text)][:4]
    assert counts[0] == 1 and sum(counts[1:]) == 1


def test_counselor_can_set_status_and_send_a_message(app, database, student):
    features = {f: 5 for f in database.FEATURES}
    database.save_submission("1001", "Asha Rao", "AIDS", "A", features, 80, "High", {}, "")
    at = app(counselor_logged_in=True)     # already signed in, so the login form is gone
    at.selectbox(key="st_1001").select("Contacted")
    at = at.button(key="sv_1001").click().run()
    assert database.get_counselor_action("1001")["status"] == "Contacted"
    at.text_area(key="rp_1001").input("Hi Asha, checking in.")
    at = at.button(key="send_1001").click().run()
    assert len(database.get_replies("1001")) == 1


def test_student_names_and_notes_cannot_inject_html(app, database):
    features = {f: 5 for f in database.FEATURES}
    database.register_student("1003", "<img src=x onerror=alert(1)>", "", "VIPS-TC", "IT", "A", 20, "pass-word-3")
    database.save_submission("1003", "<img src=x onerror=alert(1)>", "IT", "A", features,
                             80, "High", {}, "<b>bold</b>")
    text = _page_text(_sign_in_counselor(app))
    assert "<img src=x" not in text and "&lt;img src=x" in text
    assert "<b>bold</b>" not in text


# ── portal ────────────────────────────────────────────────────────────────────
def _sign_in_student(app, roll="1001", password=STUDENT_PASSWORD, branch="AIDS"):
    at = app()
    at.text_input(key="p_roll").input(roll)
    at.text_input(key="p_pwd").input(password)
    at.selectbox(key="p_branch").select(branch)
    return at.button(key="portal_login").click().run()


def test_portal_login_requires_the_right_branch(app, student):
    assert not _sign_in_student(app, branch="CSE").session_state.student_logged_in
    assert _sign_in_student(app).session_state.student_logged_in


def test_portal_shows_counselor_messages_and_marks_them_read(app, database, student):
    database.save_reply("1001", "Hi Asha, checking in.")
    at = _sign_in_student(app)
    assert "checking in" in _page_text(at)
    assert all(bool(r) for r in database.get_replies("1001")["read_by_student"])


def test_overdue_check_in_is_flagged_for_that_student(app, database, student):
    from sqlalchemy import update
    features = {f: 5 for f in database.FEATURES}
    database.save_submission("1001", "Asha Rao", "AIDS", "A", features, 40, "Medium", {}, "")
    with database.get_engine().begin() as conn:
        conn.execute(update(database.submissions).values(timestamp=datetime.now() - timedelta(days=40)))
    at = _sign_in_student(app)
    assert any("was due 10 day(s) ago" in w.value for w in at.warning)


def test_fresh_submission_gets_no_reminder(app, database, student):
    features = {f: 5 for f in database.FEATURES}
    database.save_submission("1001", "Asha Rao", "AIDS", "A", features, 40, "Medium", {}, "")
    at = _sign_in_student(app)
    assert not any("check-in" in w.value for w in at.warning)


# ── model card ────────────────────────────────────────────────────────────────
def test_model_card_discloses_simulated_data_and_metrics(app):
    at = app()
    assert any("simulated" in w.value for w in at.warning)
    assert len(at.dataframe) >= 2      # confusion matrix + model comparison
