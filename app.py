"""
app.py — Burnout Buster v5
Design: tracker.gg inspired — dark navy, red accent, clean cards
Features: GitHub persistent storage, college records, improved analytics
"""
import streamlit as st
import joblib, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Burnout Buster",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS: tracker.gg inspired ───────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600;700;800&family=Barlow+Condensed:wght@600;700;800&family=Inter:wght@300;400;500&display=swap');

* { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1,h2,h3,h4 { font-family: 'Barlow', sans-serif !important; }

/* Dark gradient background like tracker.gg */
.stApp {
    background: linear-gradient(135deg, #0d1117 0%, #161b27 40%, #1a1f35 70%, #161b27 100%);
    min-height: 100vh;
}

/* Hide Streamlit default header completely */
[data-testid="stHeader"] { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
.stApp > header { display: none !important; }
header { display: none !important; }
#MainMenu { display: none !important; }
footer { display: none !important; }

.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* Tabs — tracker.gg navbar style */
.stTabs [data-baseweb="tab-list"] {
    background: #0d1117;
    border-radius: 0;
    padding: 0 24px;
    gap: 0;
    border-bottom: 2px solid #1e2333;
}
.stTabs [data-baseweb="tab"] {
    color: #8892a4 !important;
    border-radius: 0 !important;
    font-family: 'Barlow', sans-serif !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 16px 20px !important;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    border-bottom: 3px solid transparent !important;
    margin-bottom: -2px;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    background: transparent !important;
    color: #ff4655 !important;
    border-bottom: 3px solid #ff4655 !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #e2e8f0 !important;
    background: rgba(255,255,255,0.03) !important;
}
.stTabs [data-baseweb="tab-panel"] {
    padding: 0 !important;
}

/* Buttons */
div.stButton > button {
    background: linear-gradient(135deg, #ff4655, #c9304a);
    color: white;
    border: none;
    border-radius: 6px;
    padding: 10px 24px;
    font-family: 'Barlow', sans-serif;
    font-size: 14px;
    font-weight: 700;
    width: 100%;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    transition: all 0.2s ease;
    box-shadow: 0 4px 15px rgba(255,70,85,0.3);
}
div.stButton > button:hover {
    background: linear-gradient(135deg, #ff5f6d, #d63547);
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(255,70,85,0.4);
}

/* Inputs */
input, textarea, select {
    background: #1e2333 !important;
    border: 1px solid #2d3550 !important;
    border-radius: 6px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
}
input:focus, textarea:focus {
    border-color: #ff4655 !important;
    box-shadow: 0 0 0 2px rgba(255,70,85,0.15) !important;
}
input::placeholder { color: #4a5568 !important; }
input[type="text"], input[type="password"], input[type="number"] {
    color: #e2e8f0 !important;
}

/* Labels and text */
label { color: #8892a4 !important; font-size: 12px !important; font-weight: 600 !important; text-transform: uppercase; letter-spacing: 0.8px; }
p, li { color: #a0aec0; font-size: 14px; line-height: 1.6; }
h1 { color: #f7fafc !important; }
h2 { color: #e2e8f0 !important; }
h3 { color: #ff4655 !important; }

/* Cards */
.trk-card {
    background: #161b27;
    border: 1px solid #1e2333;
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 12px;
}
.trk-card-accent {
    background: linear-gradient(135deg, #161b27, #1a1f35);
    border: 1px solid #2d3550;
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 12px;
}

/* Stat boxes */
.stat-box {
    background: #161b27;
    border: 1px solid #1e2333;
    border-radius: 8px;
    padding: 20px;
    text-align: center;
}
.stat-num {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 36px;
    font-weight: 800;
    color: #f7fafc;
    line-height: 1;
}
.stat-lbl {
    font-size: 11px;
    color: #4a5568;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 6px;
}

/* Risk badges */
.badge-high   { background: rgba(255,70,85,0.15); color: #ff4655; border: 1px solid rgba(255,70,85,0.4); border-radius: 4px; padding: 2px 10px; font-size: 11px; font-weight: 700; letter-spacing: 0.5px; }
.badge-medium { background: rgba(245,166,35,0.15); color: #f5a623; border: 1px solid rgba(245,166,35,0.4); border-radius: 4px; padding: 2px 10px; font-size: 11px; font-weight: 700; letter-spacing: 0.5px; }
.badge-low    { background: rgba(72,199,142,0.15); color: #48c78e; border: 1px solid rgba(72,199,142,0.4); border-radius: 4px; padding: 2px 10px; font-size: 11px; font-weight: 700; letter-spacing: 0.5px; }

/* Section label */
.sec-label {
    font-family: 'Barlow', sans-serif;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #ff4655;
    margin: 24px 0 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.sec-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #1e2333;
}

/* Student row */
.student-row {
    background: #161b27;
    border: 1px solid #1e2333;
    border-left: 3px solid #2d3550;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin-bottom: 8px;
    transition: all 0.2s;
}
.student-row:hover { border-color: #2d3550; background: #1a1f35; }
.student-row-high   { border-left-color: #ff4655; }
.student-row-medium { border-left-color: #f5a623; }
.student-row-low    { border-left-color: #48c78e; }

/* Reply bubble */
.reply-bubble {
    background: #1a1f35;
    border-left: 3px solid #ff4655;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin-bottom: 10px;
    color: #a0aec0;
    font-size: 14px;
}

/* Score display */
.score-big {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 72px;
    font-weight: 800;
    line-height: 1;
}

/* Page wrapper */
.page { padding: 24px 32px; }

/* Expander */
.streamlit-expanderHeader {
    background: #161b27 !important;
    color: #a0aec0 !important;
    border: 1px solid #1e2333 !important;
    border-radius: 6px !important;
}
</style>
""", unsafe_allow_html=True)

# ── LOAD MODEL ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load("burnout_model.pkl")
    le    = joblib.load("label_encoder.pkl")
    with open("model_meta.json") as f:
        meta = json.load(f)
    return model, le, meta

model, le, meta = load_model()
FEATURES = meta["features"]

COLLEGES = ["Vivekananda Institute of Professional Studies - Technical Campus (VIPS-TC)", "Other"]
BRANCHES  = ["AIDS","CSE","ECE","IT","ME","CE","EEE","Other"]
SECTIONS  = ["A","B","C"]

# ── SESSION STATE ──────────────────────────────────────────────────────────────
for k, v in [
    ("counselor_logged_in", False),
    ("student_logged_in",   False),
    ("student_data",        {}),
    ("notifications",       []),
]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── HELPERS ────────────────────────────────────────────────────────────────────
def compute_score(proba_dict):
    return int(min(100, max(0,
        proba_dict.get("High",0)*100 +
        proba_dict.get("Medium",0)*50 +
        proba_dict.get("Low",0)*10)))

def score_info(score):
    if score <= 33:   return "Thriving",       "#48c78e", "low",    "badge-low"
    elif score <= 66: return "Needs Attention", "#f5a623", "medium", "badge-medium"
    else:             return "At Risk",         "#ff4655", "high",   "badge-high"

def trajectory(scores):
    if len(scores) < 2: return "—", "#4a5568", "First assessment"
    diff = int(scores[-1]) - int(scores[-2])
    if diff > 5:    return f"+{diff}", "#ff4655", "Score increased since last survey"
    elif diff < -5: return str(diff), "#48c78e", "Score improved since last survey"
    else:           return "~", "#f5a623", "Score is stable"

def predict(vals):
    arr   = np.array([vals])
    idx   = model.predict(arr)[0]
    proba = model.predict_proba(arr)[0]
    label = le.inverse_transform([idx])[0]
    return label, dict(zip(le.classes_, proba))

def push_notif(name, roll, risk, score, flagged=False):
    if risk not in ("High","Medium"): return
    st.session_state.notifications.insert(0, {
        "id":      len(st.session_state.notifications),
        "ts":      datetime.now().strftime("%d %b, %I:%M %p"),
        "name":    name,
        "roll":    roll,
        "risk":    risk,
        "score":   score,
        "flagged": flagged,
        "read":    False,
        "expires": (datetime.now() + timedelta(days=30)).isoformat(),
    })

def personalized_advice(feat_dict, score, label):
    """Generate unique advice based on student's specific weak areas."""
    tips = []
    if float(feat_dict.get("sleep_hours", 7)) < 6:
        tips.append(("Sleep Debt", "You're averaging under 6 hours. Try going to bed 30 min earlier for 2 weeks — it's the single highest-impact change you can make."))
    if float(feat_dict.get("social_media_hrs", 3)) > 5:
        tips.append(("Screen Overload", "5+ hours of social media daily is actively increasing your stress. Set a 2-hour limit using your phone's screen time settings."))
    if float(feat_dict.get("fomo_score", 5)) > 7:
        tips.append(("FOMO", "High FOMO is draining your mental energy. Practice intentional social media breaks — even 1 day off per week helps significantly."))
    if float(feat_dict.get("exercise_days", 3)) < 2:
        tips.append(("Physical Activity", "You're barely moving. Even a 20-minute walk 3x a week releases endorphins that directly reduce academic stress."))
    if float(feat_dict.get("confidence", 5)) < 4:
        tips.append(("Self Confidence", "Low confidence is amplifying all other stressors. Write down 3 things you did well each evening — it genuinely rewires your thinking."))
    if float(feat_dict.get("support_system", 5)) < 4:
        tips.append(("Isolation", "You seem to be going through this alone. Reach out to one person today — a friend, family member, or your college counselor."))
    if float(feat_dict.get("backlogs", 0)) > 2:
        tips.append(("Academic Backlog", "Multiple backlogs create compounding stress. Talk to your academic advisor about a realistic catch-up plan this week."))
    if float(feat_dict.get("peer_pressure", 5)) > 7:
        tips.append(("Peer Pressure", "High peer pressure is affecting your decisions. It's okay to say no — your path doesn't need to match everyone else's."))
    if not tips:
        if label == "Thriving":
            tips.append(("Keep It Up", "Your scores across all areas look healthy. Keep maintaining your current habits and check in monthly."))
        else:
            tips.append(("Overall Wellness", "Multiple factors are contributing to your stress. Focus on sleep and one social connection this week as first steps."))
    return tips[:3]

# ── NAVBAR ─────────────────────────────────────────────────────────────────────
unread = sum(1 for n in st.session_state.notifications if not n["read"])
alert_html = ""
if unread > 0:
    alert_html = f'<span style="background:#ff4655;color:white;font-size:11px;font-weight:700;padding:3px 8px;border-radius:12px;margin-left:12px;">{unread} NEW</span>'

student_name = ""
if st.session_state.student_logged_in:
    student_name = st.session_state.student_data.get("name","").split()[0]
    user_html = f'<span style="color:#48c78e;font-size:13px;font-weight:600;">● {student_name}</span>'
else:
    user_html = '<span style="color:#4a5568;font-size:13px;">Not logged in</span>'

navbar = """
<div style="background:#0d1117;padding:14px 32px;display:flex;align-items:center;
justify-content:space-between;border-bottom:1px solid #1e2333;position:sticky;top:0;z-index:999;">
  <div style="display:flex;align-items:center;gap:16px;">
    <div style="background:linear-gradient(135deg,#ff4655,#c9304a);width:32px;height:32px;
    border-radius:6px;display:flex;align-items:center;justify-content:center;
    font-size:16px;">🎯</div>
    <div>
      <div style="color:white;font-family:Barlow,sans-serif;font-weight:800;font-size:18px;
      letter-spacing:0.5px;line-height:1;">BURNOUT BUSTER</div>
      <div style="color:#4a5568;font-size:11px;letter-spacing:1px;">VIPS-TC WELLNESS PLATFORM</div>
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:20px;">
    ALERT_PLACEHOLDER
    USER_PLACEHOLDER
    <span style="color:#2d3550;font-size:18px;">|</span>
    <span style="color:#4a5568;font-size:12px;">Mohit Kumar</span>
  </div>
</div>"""

if unread > 0:
    alert_part = str(unread) + " NEW"
    alert_html = "<span style='background:#ff4655;color:white;font-size:11px;font-weight:700;padding:3px 8px;border-radius:12px;margin-left:12px;'>" + alert_part + "</span>"
else:
    alert_html = ""

if st.session_state.student_logged_in:
    user_html = "<span style='color:#48c78e;font-size:13px;font-weight:600;'>● " + student_name + "</span>"
else:
    user_html = "<span style='color:#4a5568;font-size:13px;'>Not logged in</span>"

navbar_html = "<div style='background:#0d1117;padding:14px 32px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid #1e2333;'>"
navbar_html += "<div style='display:flex;align-items:center;gap:16px;'>"
navbar_html += "<div style='background:linear-gradient(135deg,#ff4655,#c9304a);width:32px;height:32px;border-radius:6px;display:flex;align-items:center;justify-content:center;font-size:16px;'>🎯</div>"
navbar_html += "<div><div style='color:white;font-family:Barlow,sans-serif;font-weight:800;font-size:18px;letter-spacing:0.5px;line-height:1;'>BURNOUT BUSTER</div>"
navbar_html += "<div style='color:#4a5568;font-size:11px;letter-spacing:1px;'>VIPS-TC WELLNESS PLATFORM</div></div></div>"
navbar_html += "<div style='display:flex;align-items:center;gap:20px;'>"
navbar_html += alert_html
navbar_html += user_html
navbar_html += "<span style='color:#2d3550;font-size:18px;'>|</span>"
navbar_html += "<span style='color:#4a5568;font-size:12px;'>Mohit Kumar</span>"
navbar_html += "</div></div>"
st.markdown(navbar_html, unsafe_allow_html=True)

# ── TABS ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "HOME", "TAKE SURVEY", "MY PORTAL", "COUNSELOR", "ANALYTICS", "DATASET"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — HOME
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="page">', unsafe_allow_html=True)

    # Hero
    st.markdown("""
    <div style="background:linear-gradient(135deg,#161b27 0%,#1a1f35 100%);
    border:1px solid #1e2333;border-radius:12px;padding:48px 40px;
    margin-bottom:24px;position:relative;overflow:hidden;">
      <div style="position:absolute;top:0;right:0;width:300px;height:100%;
      background:radial-gradient(circle at 100% 50%,rgba(255,70,85,0.08),transparent 70%);"></div>
      <div style="color:#ff4655;font-family:Barlow,sans-serif;font-size:11px;
      font-weight:700;letter-spacing:3px;text-transform:uppercase;margin-bottom:12px;">
        VIPS-TC Wellness Initiative
      </div>
      <div style="color:#f7fafc;font-family:Barlow,sans-serif;font-size:40px;
      font-weight:800;line-height:1.1;margin-bottom:16px;">
        Know your burnout risk<br>before it hits.
      </div>
      <div style="color:#8892a4;font-size:16px;max-width:520px;line-height:1.7;margin-bottom:28px;">
        A 3-minute assessment powered by machine learning.
        Get your personal burnout score and actionable advice.
        Your counselor is notified if you need support.
      </div>
      <div style="display:flex;gap:16px;flex-wrap:wrap;">
        <div style="background:rgba(255,70,85,0.1);border:1px solid rgba(255,70,85,0.3);
        border-radius:6px;padding:8px 16px;color:#ff4655;font-size:13px;font-weight:600;">
          ML-Powered Prediction
        </div>
        <div style="background:rgba(72,199,142,0.1);border:1px solid rgba(72,199,142,0.3);
        border-radius:6px;padding:8px 16px;color:#48c78e;font-size:13px;font-weight:600;">
          100% Confidential
        </div>
        <div style="background:rgba(245,166,35,0.1);border:1px solid rgba(245,166,35,0.3);
        border-radius:6px;padding:8px 16px;color:#f5a623;font-size:13px;font-weight:600;">
          Early Intervention
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    cards = [
        ("TAKE THE SURVEY", "Answer 17 questions covering academics, lifestyle, and mental health. Takes 3 minutes.", "#ff4655"),
        ("TRACK PROGRESS", "Your personal portal shows score history, trend analysis, and direct messages from your counselor.", "#f5a623"),
        ("GET SUPPORT", "Results go directly to your counselor who will reach out if needed. You are not alone.", "#48c78e"),
    ]
    for col, (title, desc, color) in zip([c1,c2,c3], cards):
        with col:
            st.markdown(f"""
            <div class="trk-card" style="border-top:3px solid {color};height:140px;">
              <div style="color:{color};font-family:Barlow,sans-serif;font-size:12px;
              font-weight:700;letter-spacing:2px;margin-bottom:10px;">{title}</div>
              <div style="color:#8892a4;font-size:13px;line-height:1.6;">{desc}</div>
            </div>""", unsafe_allow_html=True)

    # Score guide
    st.markdown('<div class="sec-label">SCORE GUIDE</div>', unsafe_allow_html=True)
    g1, g2, g3 = st.columns(3)
    guide = [
        ("0 — 33", "THRIVING", "You're managing well. Keep your healthy habits going.", "#48c78e"),
        ("34 — 66", "NEEDS ATTENTION", "Some stress signals detected. Small changes matter.", "#f5a623"),
        ("67 — 100", "AT RISK", "Significant burnout indicators. Please reach out for support.", "#ff4655"),
    ]
    for col, (rng, lbl, desc, color) in zip([g1,g2,g3], guide):
        with col:
            st.markdown(f"""
            <div style="background:#161b27;border:1px solid #1e2333;border-radius:8px;
            padding:20px;text-align:center;border-top:3px solid {color};">
              <div style="font-family:Barlow Condensed,sans-serif;font-size:28px;
              font-weight:800;color:{color};">{rng}</div>
              <div style="color:white;font-family:Barlow,sans-serif;font-size:14px;
              font-weight:700;margin:6px 0;letter-spacing:1px;">{lbl}</div>
              <div style="color:#4a5568;font-size:12px;line-height:1.5;">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SURVEY
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    from database import student_exists, register_student, save_submission, get_student_submissions

    st.markdown('<div class="page">', unsafe_allow_html=True)
    st.markdown("## Student Wellness Assessment")
    st.markdown("Your responses are **confidential**. Honest answers help us support you better.")

    st.markdown('<div class="sec-label">ABOUT YOU</div>', unsafe_allow_html=True)
    i1, i2 = st.columns(2)
    with i1:
        s_name  = st.text_input("Full Name *",   placeholder="e.g. Ravi Sharma", key="s_name")
        s_roll  = st.text_input("Roll Number *", placeholder="e.g. 01217711924", key="s_roll")
        s_email = st.text_input("College Email", placeholder="e.g. ravi@vips.edu", key="s_email")
        s_age   = st.number_input("Age", min_value=16, max_value=30, value=20, key="s_age")
    with i2:
        s_college = st.selectbox("College *",  COLLEGES, key="s_college")
        s_branch  = st.selectbox("Branch *",   BRANCHES, key="s_branch")
        s_section = st.selectbox("Section *",  SECTIONS, key="s_section")
        if s_roll and s_roll.strip():
            is_new = not student_exists(s_roll.strip(), s_branch, s_section)
        else:
            is_new = True
        if is_new:
            st.markdown("**Create your portal password**")
            s_pwd  = st.text_input("Password *",         type="password", key="s_pwd")
            s_pwd2 = st.text_input("Confirm Password *", type="password", key="s_pwd2")
        else:
            s_pwd = s_pwd2 = None
            if s_roll and s_roll.strip():
                st.success("Welcome back! Your profile is already set up.")

    st.markdown('<div class="sec-label">ACADEMIC</div>', unsafe_allow_html=True)
    a1, a2, a3 = st.columns(3)
    with a1:
        q_exams   = st.select_slider("Exams per month",      options=list(range(1,9)),  value=4, key="q1")
        q_assign  = st.select_slider("Assignments per week", options=list(range(1,13)), value=5, key="q2")
    with a2:
        q_attend  = st.slider("Attendance pressure (1-10)",  1, 10, 6, key="q3")
        q_cgpa    = st.slider("Current CGPA", 4.0, 10.0, 7.0, step=0.1, key="q4")
    with a3:
        q_backlog = st.select_slider("Active backlogs",      options=list(range(0,9)),  value=0, key="q5")
        q_study   = st.select_slider("Study hours per day",  options=list(range(1,13)), value=5, key="q6")

    st.markdown('<div class="sec-label">SOCIAL & MENTAL</div>', unsafe_allow_html=True)
    b1, b2, b3 = st.columns(3)
    with b1:
        q_fomo   = st.slider("FOMO level (1-10)",            1, 10, 5, key="q7")
        q_peer   = st.slider("Peer pressure (1-10)",         1, 10, 5, key="q8")
    with b2:
        q_family = st.slider("Family expectations (1-10)",   1, 10, 6, key="q9")
        q_social = st.select_slider("Social media hrs/day",  options=list(range(0,13)), value=3, key="q10")
    with b3:
        q_reject = st.slider("Rejection sensitivity (1-10)", 1, 10, 5, key="q11")
        q_mhv    = st.select_slider("Counselor visits/month",options=list(range(0,6)),  value=0, key="q12")

    st.markdown('<div class="sec-label">LIFESTYLE</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        q_sleep  = st.select_slider("Sleep hours/night",     options=list(range(3,11)), value=6, key="q13")
        q_exer   = st.select_slider("Exercise days/week",    options=list(range(0,8)),  value=2, key="q14")
    with c2:
        q_diet   = st.slider("Diet quality (1-10)",          1, 10, 5, key="q15")
        q_conf   = st.slider("Self-confidence (1-10)",       1, 10, 5, key="q16")
    with c3:
        q_support= st.slider("Support from friends/family",  1, 10, 5, key="q17")

    st.markdown('<div class="sec-label">ANYTHING ELSE?</div>', unsafe_allow_html=True)
    s_note = st.text_area("Anything you want your counselor to know? (optional)",
                          placeholder="e.g. I've been feeling anxious before exams...",
                          height=80, key="s_note")

    consent = st.checkbox("I understand this is a wellness screening tool, not a clinical diagnosis. I consent to my counselor reviewing my responses.")
    st.markdown("")

    if st.button("SUBMIT ASSESSMENT", use_container_width=True, key="submit_survey"):
        errors = []
        if not s_name or not s_name.strip(): errors.append("Please enter your name")
        if not s_roll or not s_roll.strip(): errors.append("Please enter your roll number")
        if not consent:                       errors.append("Please check the consent box")
        if is_new:
            if not s_pwd:                     errors.append("Please create a password")
            elif s_pwd != s_pwd2:             errors.append("Passwords do not match")
        if errors:
            for e in errors: st.error(e)
        else:
            feat_vals = [q_exams, q_assign, q_attend, q_cgpa, q_backlog, q_study,
                         q_fomo, q_peer, q_family, q_social, q_reject,
                         q_sleep, q_exer, q_diet, q_conf, q_support, q_mhv]
            feat_dict = dict(zip(FEATURES, feat_vals))
            risk, proba = predict(feat_vals)
            score = compute_score(proba)
            slabel, color, risk_css, badge_css = score_info(score)

            if is_new and s_pwd:
                with st.spinner("Setting up your profile..."):
                    register_student(s_roll.strip(), s_name.strip(), s_email.strip(),
                                     s_college, s_branch, s_section, s_age, s_pwd)

            prev = get_student_submissions(s_roll.strip())
            flagged = False
            if not prev.empty and "burnout_risk" in prev.columns and len(prev) >= 1:
                if risk == "High" and prev.iloc[-1].get("burnout_risk","") == "High":
                    flagged = True

            with st.spinner("Saving your submission..."):
                save_submission(s_roll.strip(), s_name.strip(), s_branch, s_section,
                                feat_dict, score, risk, proba, s_note.strip() if s_note else "")

            push_notif(s_name.strip(), s_roll.strip(), risk, score, flagged)

            st.markdown("---")
            st.markdown(f"### Assessment Complete — Results for {s_name.strip()}")

            # Result card
            _, rc, _ = st.columns([1,2,1])
            with rc:
                st.markdown(f"""
                <div style="background:#161b27;border:1px solid #1e2333;border-radius:12px;
                padding:32px;text-align:center;border-top:4px solid {color};">
                  <div class="score-big" style="color:{color};">{score}</div>
                  <div style="color:#4a5568;font-size:14px;margin:-4px 0 12px;">out of 100</div>
                  <span class="badge-{risk_css}" style="font-size:13px;padding:4px 14px;">{slabel.upper()}</span>
                </div>""", unsafe_allow_html=True)

            st.markdown("")
            pc1, pc2, pc3 = st.columns(3)
            clrs = {"High":"#ff4655","Low":"#48c78e","Medium":"#f5a623"}
            lbls = {"High":"At Risk","Low":"Thriving","Medium":"Needs Attention"}
            for col, k in zip([pc1,pc2,pc3], le.classes_):
                with col:
                    st.markdown(f"""
                    <div class="stat-box">
                      <div style="font-family:Barlow Condensed,sans-serif;font-size:28px;
                      font-weight:800;color:{clrs[k]};">{proba[k]*100:.0f}%</div>
                      <div class="stat-lbl">{lbls[k]}</div>
                    </div>""", unsafe_allow_html=True)

            # Personalized advice
            st.markdown('<div class="sec-label">YOUR PERSONALIZED ACTION PLAN</div>', unsafe_allow_html=True)
            tips = personalized_advice(feat_dict, score, slabel)
            tip_cols = st.columns(len(tips))
            tip_colors = ["#ff4655","#f5a623","#48c78e"]
            for i, (col, (title, desc)) in enumerate(zip(tip_cols, tips)):
                with col:
                    st.markdown(f"""
                    <div class="trk-card" style="border-left:3px solid {tip_colors[i]};">
                      <div style="color:{tip_colors[i]};font-family:Barlow,sans-serif;
                      font-size:12px;font-weight:700;letter-spacing:1.5px;margin-bottom:8px;">
                        {title.upper()}
                      </div>
                      <div style="color:#8892a4;font-size:13px;line-height:1.6;">{desc}</div>
                    </div>""", unsafe_allow_html=True)

            if flagged:
                st.error("This is your second consecutive At Risk result. Your counselor has been specially alerted and will reach out soon.")

            if slabel == "At Risk":
                st.markdown("""
                <div style="background:rgba(255,70,85,0.08);border:1px solid rgba(255,70,85,0.2);
                border-radius:8px;padding:16px 20px;margin-top:16px;">
                  <div style="color:#ff4655;font-weight:700;margin-bottom:6px;">Free Helplines (Confidential)</div>
                  <div style="color:#8892a4;font-size:13px;">
                    iCall: 9152987821 &nbsp;|&nbsp; Vandrevala: 1860-2662-345 (24/7) &nbsp;|&nbsp; NIMHANS: 080-46110007
                  </div>
                </div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — STUDENT PORTAL
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    from database import verify_student, get_student_submissions, get_replies, mark_replies_read, get_reminder

    st.markdown('<div class="page">', unsafe_allow_html=True)
    st.markdown("## My Wellness Portal")

    if not st.session_state.student_logged_in:
        st.markdown("Log in with your roll number, branch, section, and the password you created during your first survey.")
        _, lc, _ = st.columns([1,1.2,1])
        with lc:
            st.markdown('<div class="trk-card">', unsafe_allow_html=True)
            p_roll    = st.text_input("Roll Number", placeholder="e.g. 01217711924", key="p_roll")
            p_branch  = st.selectbox("Branch", BRANCHES, key="p_branch")
            p_section = st.selectbox("Section", SECTIONS, key="p_section")
            p_pwd     = st.text_input("Password", type="password", key="p_pwd")
            if st.button("LOGIN TO PORTAL", use_container_width=True, key="portal_login"):
                if p_roll and p_pwd:
                    student = verify_student(p_roll.strip(), p_pwd.strip(), p_branch, p_section)
                    if student:
                        st.session_state.student_logged_in = True
                        st.session_state.student_data      = student
                        st.rerun()
                    else:
                        st.error("Incorrect credentials. Make sure your roll number, branch, section, and password all match what you used during the survey.")
                else:
                    st.warning("Please fill in all fields.")
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        sd   = st.session_state.student_data
        name = sd.get("name", "Student")
        roll = str(sd.get("roll_number",""))

        hc1, hc2 = st.columns([4,1])
        with hc1:
            st.markdown(f"## Welcome back, {name.split()[0]}")
        with hc2:
            if st.button("LOGOUT", key="s_out"):
                st.session_state.student_logged_in = False
                st.session_state.student_data = {}
                st.rerun()

        # Reminder
        try:
            reminder = get_reminder()
            if reminder and reminder.get("next_due"):
                next_due = datetime.strptime(str(reminder["next_due"]), "%Y-%m-%d %H:%M:%S")
                days_left = (next_due - datetime.now()).days
                if days_left <= 3:
                    st.warning(f"Your next wellness check-in is due {'today' if days_left<=0 else f'in {days_left} day(s)'}. Head to TAKE SURVEY.")
        except Exception:
            pass

        subs = get_student_submissions(roll)

        if subs.empty:
            st.markdown("""
            <div class="trk-card" style="text-align:center;padding:40px;">
              <div style="color:#4a5568;font-size:14px;">You have not taken the survey yet.</div>
              <div style="color:#8892a4;font-size:13px;margin-top:8px;">Head to TAKE SURVEY to get your first wellness score.</div>
            </div>""", unsafe_allow_html=True)
        else:
            scores = []
            if "burnout_score" in subs.columns:
                scores = pd.to_numeric(subs["burnout_score"], errors="coerce").fillna(0).astype(int).tolist()

            latest = scores[-1] if scores else 0
            slabel, color, risk_css, badge_css = score_info(latest)
            delta, dcolor, dmsg = trajectory(scores)

            s1,s2,s3,s4 = st.columns(4)
            with s1:
                st.markdown(f'<div class="stat-box" style="border-top:3px solid {color};"><div class="score-big" style="color:{color};font-size:48px;">{latest}</div><div class="stat-lbl">Latest Score</div></div>', unsafe_allow_html=True)
            with s2:
                st.markdown(f'<div class="stat-box"><div style="font-family:Barlow Condensed,sans-serif;font-size:40px;font-weight:800;color:{dcolor};">{delta}</div><div class="stat-lbl">Trend</div></div>', unsafe_allow_html=True)
            with s3:
                st.markdown(f'<div class="stat-box"><div class="score-big" style="font-size:48px;">{len(subs)}</div><div class="stat-lbl">Surveys Done</div></div>', unsafe_allow_html=True)
            with s4:
                st.markdown(f'<div class="stat-box"><br><span class="{badge_css}">{slabel.upper()}</span><div class="stat-lbl" style="margin-top:10px;">Current Status</div></div>', unsafe_allow_html=True)

            st.markdown(f'<div style="color:{dcolor};font-size:13px;margin:8px 0 20px;">{dmsg}</div>', unsafe_allow_html=True)

            if len(scores) > 1:
                st.markdown('<div class="sec-label">SCORE HISTORY</div>', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(10, 3.5))
                fig.patch.set_facecolor("#0d1117")
                ax.set_facecolor("#161b27")
                xs = list(range(1, len(scores)+1))
                ax.fill_between(xs, scores, alpha=0.15, color="#ff4655")
                ax.plot(xs, scores, color="#ff4655", linewidth=2.5, marker="o",
                        markersize=8, markerfacecolor="#0d1117",
                        markeredgewidth=2.5, markeredgecolor="#ff4655")
                ax.axhline(33, color="#48c78e", linewidth=1, linestyle="--", alpha=0.4, label="Thriving")
                ax.axhline(66, color="#f5a623", linewidth=1, linestyle="--", alpha=0.4, label="Needs Attention")
                for i, s in enumerate(scores):
                    ax.annotate(str(s), (xs[i], s), textcoords="offset points",
                                xytext=(0,12), ha="center", fontsize=10, color="#e2e8f0", fontweight="bold")
                ax.set_xlabel("Survey #", color="#4a5568", fontsize=11)
                ax.set_ylabel("Score", color="#4a5568", fontsize=11)
                ax.tick_params(colors="#4a5568")
                ax.spines[:].set_color("#1e2333")
                ax.set_ylim(0, 115)
                ax.legend(fontsize=9, facecolor="#161b27", edgecolor="#1e2333", labelcolor="#8892a4")
                plt.tight_layout()
                st.pyplot(fig)

            with st.expander("All Submissions"):
                show_cols = [c for c in ["timestamp","burnout_score","burnout_risk"] if c in subs.columns]
                st.dataframe(subs[show_cols], use_container_width=True)

        # Messages
        st.markdown('<div class="sec-label">MESSAGES FROM YOUR COUNSELOR</div>', unsafe_allow_html=True)
        replies = get_replies(roll)
        if replies.empty:
            st.markdown('<div class="trk-card" style="color:#4a5568;text-align:center;padding:24px;">No messages yet. Your counselor will reach out after reviewing your assessment.</div>', unsafe_allow_html=True)
        else:
            mark_replies_read(roll)
            for _, rep in replies.iterrows():
                ts  = rep.get("timestamp","")
                msg = rep.get("counselor_message","")
                st.markdown(f"""
                <div class="reply-bubble">
                  <div style="color:#4a5568;font-size:11px;margin-bottom:6px;letter-spacing:0.5px;">
                    COUNSELOR &nbsp;·&nbsp; {ts}
                  </div>
                  {msg}
                </div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — COUNSELOR DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    from database import (get_all_submissions, get_all_students, upsert_counselor_action,
                          get_counselor_action, save_reply, save_reminder,
                          save_college_records, get_college_records)

    st.markdown('<div class="page">', unsafe_allow_html=True)

    if not st.session_state.counselor_logged_in:
        st.markdown("## Counselor Login")
        st.markdown("Restricted to authorised counselors only.")
        _, lc, _ = st.columns([1,1.2,1])
        with lc:
            st.markdown('<div class="trk-card">', unsafe_allow_html=True)
            pwd = st.text_input("Password", type="password", placeholder="Enter counselor password", key="c_pwd")
            if st.button("LOGIN", use_container_width=True, key="c_login"):
                COUNSELOR_PASSWORD = "ProjectAlpha01"
                try:
                    COUNSELOR_PASSWORD = st.secrets["COUNSELOR_PASSWORD"]
                except Exception:
                    pass
                if pwd == COUNSELOR_PASSWORD:
                    st.session_state.counselor_logged_in = True
                    st.rerun()
                else:
                    st.error("Incorrect password.")
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        hc1, hc2 = st.columns([4,1])
        with hc1: st.markdown("## Counselor Dashboard")
        with hc2:
            if st.button("LOGOUT", key="c_out"):
                st.session_state.counselor_logged_in = False
                st.rerun()

        # Alerts
        notifs = [n for n in st.session_state.notifications
                  if datetime.fromisoformat(n["expires"]) > datetime.now()]
        st.session_state.notifications = notifs
        unread_c = sum(1 for n in notifs if not n["read"])

        with st.expander(f"ALERTS  {('· ' + str(unread_c) + ' UNREAD') if unread_c else '· ALL READ'}", expanded=unread_c>0):
            if not notifs:
                st.markdown('<div style="color:#4a5568;padding:12px;">No alerts yet.</div>', unsafe_allow_html=True)
            else:
                if st.button("Mark all read", key="mark_all"):
                    for n in st.session_state.notifications: n["read"] = True
                    st.rerun()
                for n in notifs:
                    brd  = "#ff4655" if n["risk"]=="High" else "#f5a623"
                    new_badge = '<span style="background:#ff4655;color:white;font-size:10px;padding:1px 6px;border-radius:10px;margin-left:6px;">NEW</span>' if not n["read"] else ""
                    flag_badge = '<span style="background:#1e2333;color:#ff4655;font-size:10px;padding:1px 6px;border-radius:10px;border:1px solid #ff4655;margin-left:6px;">PERSISTENT</span>' if n.get("flagged") else ""
                    st.markdown(f"""
                    <div style="background:#161b27;border:1px solid {brd};border-radius:6px;
                    padding:12px 16px;margin-bottom:8px;opacity:{'1' if not n['read'] else '0.5'};">
                      <strong style="color:#e2e8f0;">{n['name']}</strong>
                      <span style="color:#4a5568;font-size:12px;"> · {n['roll']}</span>
                      <span style="background:{brd};color:white;font-size:11px;font-weight:700;
                      padding:2px 8px;border-radius:4px;margin-left:8px;">{n['score']}/100 · {n['risk']}</span>
                      {new_badge}{flag_badge}
                      <span style="float:right;color:#4a5568;font-size:11px;">{n['ts']}</span>
                    </div>""", unsafe_allow_html=True)
                    if not n["read"]:
                        if st.button("Mark read", key=f"nr_{n['id']}"):
                            n["read"] = True
                            st.rerun()

        # Reminder settings
        with st.expander("SURVEY REMINDER SETTINGS"):
            freq = st.select_slider("Reminder frequency", options=[7,14,21,30,60], value=30,
                                    format_func=lambda x: f"Every {x} days")
            if st.button("Save Reminder Schedule", key="save_rem"):
                save_reminder(freq)
                st.success(f"Reminder set — students will be prompted every {freq} days.")

        # College Records Upload
        st.markdown('<div class="sec-label">COLLEGE ACADEMIC RECORDS</div>', unsafe_allow_html=True)
        st.markdown("Upload official college records per section. The system will cross-reference these with burnout scores to identify inconsistencies.")

        with st.expander("UPLOAD RECORDS"):
            ur1, ur2 = st.columns(2)
            with ur1:
                up_branch  = st.selectbox("Branch",  BRANCHES, key="up_branch")
            with ur2:
                up_section = st.selectbox("Section", SECTIONS, key="up_section")

            st.markdown("**Required CSV columns:** `roll_number, name, attendance_pct, marks_pct, participation, remarks`")
            st.markdown("Download template:")

            template_df = pd.DataFrame({
                "roll_number":    ["01217711924","01217711925"],
                "name":           ["Student Name","Student Name"],
                "attendance_pct": [85, 72],
                "marks_pct":      [78, 65],
                "participation":  ["Active","Low"],
                "remarks":        ["Good","Needs improvement"],
            })
            st.download_button("Download CSV Template",
                               data=template_df.to_csv(index=False).encode(),
                               file_name="college_records_template.csv",
                               mime="text/csv")

            uploaded = st.file_uploader("Upload Records CSV", type=["csv"], key="rec_upload")
            if uploaded:
                try:
                    df_up = pd.read_csv(uploaded)
                    required = ["roll_number","attendance_pct","marks_pct"]
                    missing = [c for c in required if c not in df_up.columns]
                    if missing:
                        st.error(f"Missing columns: {', '.join(missing)}")
                    else:
                        st.dataframe(df_up.head(), use_container_width=True)
                        if st.button("Confirm Upload", key="confirm_upload"):
                            with st.spinner("Saving records to database..."):
                                save_college_records(df_up, up_branch, up_section)
                            st.success(f"Saved {len(df_up)} records for {up_branch}-{up_section}!")
                except Exception as ex:
                    st.error(f"Error reading file: {ex}")

        # Student list
        st.markdown('<div class="sec-label">STUDENT SUBMISSIONS</div>', unsafe_allow_html=True)
        all_subs  = get_all_submissions()
        all_studs = get_all_students()

        if all_subs.empty:
            st.markdown('<div class="trk-card" style="color:#4a5568;text-align:center;padding:32px;">No student submissions yet.</div>', unsafe_allow_html=True)
        else:
            # Name lookup
            name_lookup = {}
            if not all_studs.empty and "roll_number" in all_studs.columns:
                for _, row in all_studs.iterrows():
                    name_lookup[str(row["roll_number"]).strip()] = str(row.get("name",""))
            if "student_name" in all_subs.columns:
                for _, row in all_subs.iterrows():
                    r = str(row.get("roll_number","")).strip()
                    n = str(row.get("student_name","")).strip()
                    if r and n and n != "nan" and r not in name_lookup:
                        name_lookup[r] = n

            unique_rolls = all_subs["roll_number"].nunique() if "roll_number" in all_subs.columns else 0
            high   = int((all_subs["burnout_risk"]=="High").sum())   if "burnout_risk" in all_subs.columns else 0
            medium = int((all_subs["burnout_risk"]=="Medium").sum()) if "burnout_risk" in all_subs.columns else 0
            low    = int((all_subs["burnout_risk"]=="Low").sum())    if "burnout_risk" in all_subs.columns else 0

            m1,m2,m3,m4 = st.columns(4)
            with m1: st.markdown(f'<div class="stat-box"><div class="stat-num">{unique_rolls}</div><div class="stat-lbl">Students</div></div>', unsafe_allow_html=True)
            with m2: st.markdown(f'<div class="stat-box" style="border-top:3px solid #ff4655;"><div class="stat-num" style="color:#ff4655;">{high}</div><div class="stat-lbl">At Risk</div></div>', unsafe_allow_html=True)
            with m3: st.markdown(f'<div class="stat-box" style="border-top:3px solid #f5a623;"><div class="stat-num" style="color:#f5a623;">{medium}</div><div class="stat-lbl">Needs Attention</div></div>', unsafe_allow_html=True)
            with m4: st.markdown(f'<div class="stat-box" style="border-top:3px solid #48c78e;"><div class="stat-num" style="color:#48c78e;">{low}</div><div class="stat-lbl">Thriving</div></div>', unsafe_allow_html=True)

            st.markdown("")
            fc1, fc2, fc3, fc4 = st.columns(4)
            with fc1: rf = st.multiselect("Risk", ["High","Medium","Low"], default=["High","Medium","Low"], key="rf")
            with fc2: fb = st.selectbox("Branch", ["All"]+BRANCHES, key="fb")
            with fc3: search = st.text_input("Search name/roll", "", key="srch")
            with fc4: show_flag = st.checkbox("Persistent At Risk only", key="sflag")

            if "roll_number" in all_subs.columns and "timestamp" in all_subs.columns:
                latest = all_subs.sort_values("timestamp").groupby("roll_number").last().reset_index()
            else:
                latest = all_subs.copy()

            if rf and "burnout_risk" in latest.columns:
                latest = latest[latest["burnout_risk"].isin(rf)]
            if fb != "All" and "branch" in latest.columns:
                latest = latest[latest["branch"] == fb]
            if search:
                roll_m = latest["roll_number"].astype(str).str.contains(search, na=False)
                name_m = pd.Series([search.lower() in name_lookup.get(str(r),"").lower() for r in latest["roll_number"]], index=latest.index)
                latest = latest[roll_m | name_m]

            st.markdown(f'<div style="color:#4a5568;font-size:13px;margin-bottom:12px;">Showing {len(latest)} students</div>', unsafe_allow_html=True)

            for _, row in latest.iterrows():
                roll_d  = str(row.get("roll_number","")).strip()
                risk_d  = row.get("burnout_risk","Unknown")
                score_d = int(row.get("burnout_score",0)) if str(row.get("burnout_score",0)).replace(".","").isdigit() else 0
                ts_d    = row.get("timestamp","")
                note_d  = str(row.get("student_note",""))
                branch_d= row.get("branch","")
                section_d= row.get("section","")
                sname_d = name_lookup.get(roll_d, roll_d)

                stud_subs = all_subs[all_subs["roll_number"].astype(str).str.strip()==roll_d] if "roll_number" in all_subs.columns else pd.DataFrame()
                flagged_d = False
                if len(stud_subs) >= 2 and "burnout_risk" in stud_subs.columns:
                    last2 = stud_subs.sort_values("timestamp").tail(2)["burnout_risk"].tolist()
                    flagged_d = all(r=="High" for r in last2)

                if show_flag and not flagged_d: continue

                action_d = get_counselor_action(roll_d)
                status_d = action_d.get("status","Pending")
                notes_d  = action_d.get("notes","")

                risk_css_d = "high" if risk_d=="High" else "medium" if risk_d=="Medium" else "low"

                # Check college records for this student
                records = get_college_records(roll=roll_d)
                has_records = not records.empty
                consistency_flag = ""
                if has_records and "attendance_pct" in records.columns:
                    try:
                        att = float(records.iloc[0]["attendance_pct"])
                        attend_score = float(row.get("attendance_pressure",5))
                        if att > 85 and attend_score > 8:
                            consistency_flag = '<span style="background:rgba(245,166,35,0.2);color:#f5a623;font-size:10px;padding:2px 8px;border-radius:4px;margin-left:8px;">VERIFY RESPONSE</span>'
                    except Exception:
                        pass

                flag_html = '<span style="background:rgba(255,70,85,0.15);color:#ff4655;font-size:10px;font-weight:700;padding:2px 8px;border-radius:4px;border:1px solid rgba(255,70,85,0.3);margin-left:8px;">PERSISTENT</span>' if flagged_d else ""

                left_part = "<strong style='color:#e2e8f0;font-size:15px;'>" + sname_d + "</strong>"
                left_part += "<span style='color:#4a5568;font-size:12px;margin-left:8px;'>Roll: " + roll_d + "</span>"
                left_part += "<span style='color:#2d3550;font-size:12px;margin-left:8px;'>" + str(branch_d) + "-" + str(section_d) + "</span>"
                left_part += flag_html + consistency_flag

                right_part = "<span class='badge-" + risk_css_d + "'>" + risk_d.upper() + "</span>"
                right_part += "<span style='color:#8892a4;font-size:13px;margin-left:8px;font-family:Barlow Condensed,sans-serif;font-weight:700;'>" + str(score_d) + "/100</span>"
                right_part += "<div style='color:#2d3550;font-size:11px;margin-top:3px;'>" + str(ts_d) + "</div>"

                note_part = ""
                if note_d and str(note_d).strip() and str(note_d).strip() != "nan":
                    note_part = "<div style='background:#1a1f35;border-left:3px solid #f5a623;border-radius:0 4px 4px 0;padding:8px 12px;margin-top:10px;color:#8892a4;font-size:13px;'><strong style='color:#f5a623;'>Student Note:</strong> " + str(note_d) + "</div>"

                card_html = "<div class='student-row student-row-" + risk_css_d + "'>"
                card_html += "<div style='display:flex;justify-content:space-between;align-items:center;'>"
                card_html += "<div>" + left_part + "</div>"
                card_html += "<div style='text-align:right;'>" + right_part + "</div>"
                card_html += "</div>" + note_part + "</div>"
                st.markdown(card_html, unsafe_allow_html=True)

                with st.expander(f"Actions — {sname_d} ({roll_d})"):
                    # College records view
                    if has_records:
                        st.markdown("**Academic Records from College:**")
                        st.dataframe(records.drop(columns=["branch","section","uploaded_at"], errors="ignore"),
                                     use_container_width=True)
                        st.markdown("---")

                    ac1, ac2 = st.columns([1,2])
                    with ac1:
                        if note_d and note_d != "nan" and note_d.strip():
                            st.markdown(f'<div style="background:#1a1f35;border-left:3px solid #f5a623;border-radius:0 4px 4px 0;padding:8px 12px;margin-bottom:12px;color:#8892a4;font-size:13px;"><strong style="color:#f5a623;">Student Note:</strong> {note_d}</div>', unsafe_allow_html=True)
                        new_status = st.selectbox("Status",
                            ["Pending","Contacted","No Action Needed"],
                            index=["Pending","Contacted","No Action Needed"].index(status_d)
                            if status_d in ["Pending","Contacted","No Action Needed"] else 0,
                            key=f"st_{roll_d}")
                        new_notes = st.text_area("Notes", value=notes_d if str(notes_d)!="nan" else "",
                                                 height=80, key=f"nt_{roll_d}")
                        if st.button("Save Status", key=f"sv_{roll_d}"):
                            upsert_counselor_action(roll_d, new_status, new_notes, flagged_d)
                            st.success("Status updated!")
                    with ac2:
                        st.markdown("**Send Private Message to Student**")
                        reply_msg = st.text_area("Message",
                            placeholder="Hi, I reviewed your assessment and wanted to check in...",
                            height=100, key=f"rp_{roll_d}")
                        if st.button("Send Message", key=f"send_{roll_d}"):
                            if reply_msg and reply_msg.strip():
                                save_reply(roll_d, reply_msg.strip())
                                st.success("Message sent! Student will see it in their portal.")
                            else:
                                st.warning("Please type a message first.")

            st.markdown("---")
            st.download_button("Export All Data (CSV)",
                               data=all_subs.to_csv(index=False).encode(),
                               file_name="burnout_submissions.csv", mime="text/csv")

    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    from database import get_all_submissions, get_all_students

    st.markdown('<div class="page">', unsafe_allow_html=True)
    st.markdown("## Analytics Dashboard")

    if not st.session_state.counselor_logged_in:
        st.warning("Please log in as counselor to view analytics.")
    else:
        all_subs  = get_all_submissions()
        all_studs = get_all_students()

        if all_subs.empty:
            st.markdown('<div class="trk-card" style="color:#4a5568;text-align:center;padding:40px;">No data yet. Analytics appear once students start submitting surveys.</div>', unsafe_allow_html=True)
        else:
            if not all_studs.empty and "roll_number" in all_studs.columns:
                merge_cols = [c for c in ["roll_number","branch","section","age","college"] if c in all_studs.columns]
                try:
                    all_subs = all_subs.merge(all_studs[merge_cols].astype(str), on="roll_number", how="left", suffixes=("","_student"))
                except Exception:
                    pass

            latest = all_subs.sort_values("timestamp").groupby("roll_number").last().reset_index() if "roll_number" in all_subs.columns else all_subs.copy()
            if "burnout_score" in latest.columns:
                latest["burnout_score"] = pd.to_numeric(latest["burnout_score"], errors="coerce").fillna(0)

            total   = len(latest)
            avg_sc  = int(latest["burnout_score"].mean()) if "burnout_score" in latest.columns and total > 0 else 0
            at_risk = int((latest["burnout_risk"]=="High").sum())  if "burnout_risk" in latest.columns else 0
            thriving= int((latest["burnout_risk"]=="Low").sum())   if "burnout_risk" in latest.columns else 0

            m1,m2,m3,m4 = st.columns(4)
            with m1: st.markdown(f'<div class="stat-box"><div class="stat-num">{total}</div><div class="stat-lbl">Total Students</div></div>', unsafe_allow_html=True)
            with m2: st.markdown(f'<div class="stat-box"><div class="stat-num">{avg_sc}/100</div><div class="stat-lbl">Avg Score</div></div>', unsafe_allow_html=True)
            with m3: st.markdown(f'<div class="stat-box" style="border-top:3px solid #ff4655;"><div class="stat-num" style="color:#ff4655;">{at_risk}</div><div class="stat-lbl">At Risk</div></div>', unsafe_allow_html=True)
            with m4: st.markdown(f'<div class="stat-box" style="border-top:3px solid #48c78e;"><div class="stat-num" style="color:#48c78e;">{thriving}</div><div class="stat-lbl">Thriving</div></div>', unsafe_allow_html=True)

            st.markdown("")

            def styled_fig():
                fig, ax = plt.subplots()
                fig.patch.set_facecolor("#0d1117")
                ax.set_facecolor("#161b27")
                ax.tick_params(colors="#4a5568")
                for spine in ax.spines.values(): spine.set_color("#1e2333")
                return fig, ax

            col1, col2 = st.columns(2)

            with col1:
                st.markdown('<div class="sec-label">RISK DISTRIBUTION</div>', unsafe_allow_html=True)
                if "burnout_risk" in latest.columns:
                    counts = latest["burnout_risk"].value_counts()
                    fig, ax = styled_fig()
                    fig.set_size_inches(5,4)
                    clrs = [{"High":"#ff4655","Medium":"#f5a623","Low":"#48c78e"}.get(l,"#2d3550") for l in counts.index]
                    wedges, texts, autotexts = ax.pie(
                        counts.values, labels=counts.index, autopct="%1.0f%%",
                        colors=clrs, startangle=90,
                        wedgeprops=dict(edgecolor="#0d1117", linewidth=2.5))
                    for t in texts: t.set_color("#8892a4"); t.set_fontsize(12)
                    for at in autotexts: at.set_color("white"); at.set_fontweight("bold"); at.set_fontsize(11)
                    ax.set_title("Students by Risk Level", color="#e2e8f0", fontweight="bold", pad=15)
                    plt.tight_layout()
                    st.pyplot(fig)

            with col2:
                st.markdown('<div class="sec-label">AVERAGE SCORE BY BRANCH</div>', unsafe_allow_html=True)
                branch_col = "branch_student" if "branch_student" in latest.columns else "branch"
                if branch_col in latest.columns and "burnout_score" in latest.columns:
                    branch_avg = latest.groupby(branch_col)["burnout_score"].mean().sort_values(ascending=False)
                    fig, ax = styled_fig()
                    fig.set_size_inches(5,4)
                    clrs = ["#ff4655" if v>66 else "#f5a623" if v>33 else "#48c78e" for v in branch_avg.values]
                    bars = ax.barh(branch_avg.index[::-1], branch_avg.values[::-1],
                                   color=clrs[::-1], height=0.55, edgecolor="none")
                    for bar, val in zip(bars, branch_avg.values[::-1]):
                        ax.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2,
                                f"{val:.0f}", va="center", color="#8892a4", fontsize=10, fontweight="bold")
                    ax.set_xlabel("Avg Score", color="#4a5568", fontsize=11)
                    ax.set_title("Burnout by Branch", color="#e2e8f0", fontweight="bold")
                    ax.set_xlim(0,115)
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info("Branch data appears once students submit with branch info.")

            # Trend over time
            st.markdown('<div class="sec-label">BURNOUT TREND OVER TIME</div>', unsafe_allow_html=True)
            if "timestamp" in all_subs.columns and "burnout_score" in all_subs.columns:
                try:
                    all_subs["date"] = pd.to_datetime(all_subs["timestamp"]).dt.date
                    all_subs["burnout_score"] = pd.to_numeric(all_subs["burnout_score"], errors="coerce")
                    trend = all_subs.groupby("date")["burnout_score"].mean().reset_index()
                    if len(trend) > 1:
                        fig, ax = styled_fig()
                        fig.set_size_inches(10,3.5)
                        ax.fill_between(range(len(trend)), trend["burnout_score"], alpha=0.12, color="#ff4655")
                        ax.plot(range(len(trend)), trend["burnout_score"], color="#ff4655",
                                linewidth=2.5, marker="o", markersize=6,
                                markerfacecolor="#0d1117", markeredgewidth=2, markeredgecolor="#ff4655")
                        ax.axhline(33, color="#48c78e", linewidth=1, linestyle="--", alpha=0.4)
                        ax.axhline(66, color="#f5a623", linewidth=1, linestyle="--", alpha=0.4)
                        ax.set_xticks(range(len(trend)))
                        ax.set_xticklabels([str(d) for d in trend["date"]], rotation=30, ha="right", fontsize=9)
                        ax.set_ylabel("Avg Score", color="#4a5568")
                        ax.set_title("Institution Burnout Trend", color="#e2e8f0", fontweight="bold")
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.info("Trend chart appears once submissions exist on multiple dates.")
                except Exception:
                    st.info("Trend data will appear as more submissions come in.")

            col3, col4 = st.columns(2)

            with col3:
                st.markdown('<div class="sec-label">SECTION COMPARISON</div>', unsafe_allow_html=True)
                sec_col = "section_student" if "section_student" in latest.columns else "section"
                if sec_col in latest.columns and "burnout_score" in latest.columns:
                    sec_avg = latest.groupby(sec_col)["burnout_score"].mean().sort_values(ascending=False)
                    fig, ax = styled_fig()
                    fig.set_size_inches(5,3.5)
                    clrs = ["#ff4655" if v>66 else "#f5a623" if v>33 else "#48c78e" for v in sec_avg.values]
                    bars = ax.bar(sec_avg.index, sec_avg.values, color=clrs, width=0.45, edgecolor="none")
                    for bar, val in zip(bars, sec_avg.values):
                        ax.text(bar.get_x()+bar.get_width()/2, val+1, f"{val:.0f}",
                                ha="center", color="#8892a4", fontweight="bold")
                    ax.set_ylabel("Avg Score", color="#4a5568")
                    ax.set_title("Score by Section", color="#e2e8f0", fontweight="bold")
                    ax.set_ylim(0,115)
                    plt.tight_layout()
                    st.pyplot(fig)

            with col4:
                st.markdown('<div class="sec-label">TOP BURNOUT DRIVERS</div>', unsafe_allow_html=True)
                feat_imp = meta["feature_importances"]
                top_feats = sorted(feat_imp.items(), key=lambda x:-x[1])[:8]
                labels_f  = [f[0].replace("_"," ").title() for f,_ in top_feats]
                values_f  = [v for _,v in top_feats]
                fig, ax = styled_fig()
                fig.set_size_inches(5,3.5)
                clrs = ["#ff4655" if i<2 else "#f5a623" if i<5 else "#2d3550" for i in range(len(labels_f))]
                ax.barh(labels_f[::-1], values_f[::-1], color=clrs[::-1], height=0.55, edgecolor="none")
                ax.set_xlabel("Importance", color="#4a5568")
                ax.set_title("Feature Importance", color="#e2e8f0", fontweight="bold")
                ax.set_xlim(0, max(values_f)*1.25)
                plt.tight_layout()
                st.pyplot(fig)

    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — DATASET
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown('<div class="page">', unsafe_allow_html=True)
    st.markdown("## Training Dataset")
    try:
        df_t = pd.read_csv("burnout_dataset.csv")
        m1,m2,m3,m4 = st.columns(4)
        with m1: st.markdown(f'<div class="stat-box"><div class="stat-num">{len(df_t)}</div><div class="stat-lbl">Records</div></div>', unsafe_allow_html=True)
        with m2: st.markdown(f'<div class="stat-box" style="border-top:3px solid #ff4655;"><div class="stat-num" style="color:#ff4655;">{int((df_t["burnout_risk"]=="High").sum())}</div><div class="stat-lbl">At Risk</div></div>', unsafe_allow_html=True)
        with m3: st.markdown(f'<div class="stat-box" style="border-top:3px solid #f5a623;"><div class="stat-num" style="color:#f5a623;">{int((df_t["burnout_risk"]=="Medium").sum())}</div><div class="stat-lbl">Needs Attention</div></div>', unsafe_allow_html=True)
        with m4: st.markdown(f'<div class="stat-box" style="border-top:3px solid #48c78e;"><div class="stat-num" style="color:#48c78e;">{int((df_t["burnout_risk"]=="Low").sum())}</div><div class="stat-lbl">Thriving</div></div>', unsafe_allow_html=True)
        st.markdown("")
        rf3 = st.multiselect("Filter", ["High","Medium","Low"], default=["High","Medium","Low"])
        st.dataframe(df_t[df_t["burnout_risk"].isin(rf3)], use_container_width=True, height=400)
        st.download_button("Download Dataset", data=df_t.to_csv(index=False).encode(),
                           file_name="burnout_dataset.csv", mime="text/csv")
    except FileNotFoundError:
        st.error("Run `python train_model.py` first.")
    st.markdown('</div>', unsafe_allow_html=True)

# ── FOOTER ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:#0d1117;border-top:1px solid #1e2333;padding:16px 32px;
text-align:center;margin-top:20px;">
  <span style="color:#2d3550;font-size:12px;letter-spacing:1px;">
    BURNOUT BUSTER &nbsp;·&nbsp; VIPS-TC &nbsp;·&nbsp; Mohit Kumar
  </span>
</div>""", unsafe_allow_html=True)
