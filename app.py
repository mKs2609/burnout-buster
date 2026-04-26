"""
app.py — Burnout Buster v4
Design: vlr.gg inspired — dark navbar, clean white content, structured cards
Bugs fixed: survey data showing in portal, counselor dashboard HTML rendering, student names
"""
import streamlit as st
import joblib, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Burnout Buster — VIPS-TC",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS: vlr.gg inspired ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

* { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: #f4f4f4;
}

/* Top navbar feel */
.stApp {
    background: #f0f0f0;
}

/* Hide default sidebar toggle on desktop */
[data-testid="collapsedControl"] { display: none; }

/* Tabs — navbar style like vlr.gg */
.stTabs [data-baseweb="tab-list"] {
    background: #1a1a2e;
    border-radius: 0;
    padding: 0 16px;
    gap: 0;
    border-bottom: 3px solid #ff4655;
}
.stTabs [data-baseweb="tab"] {
    color: #ffffff !important;
    border-radius: 0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 14px 18px !important;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    border-bottom: 3px solid transparent !important;
    margin-bottom: -3px;
}
.stTabs [aria-selected="true"] {
    background: transparent !important;
    color: white !important;
    border-bottom: 3px solid #ff4655 !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: white !important;
    background: rgba(255,255,255,0.05) !important;
}

/* Content area */
.block-container {
    padding: 0 !important;
    padding-top: 60px !important;
    max-width: 100% !important;
}

/* Cards */
.bb-card {
    background: white;
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 12px;
    border: 1px solid #e8e8e8;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.bb-card-dark {
    background: #1a1a2e;
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 12px;
    border: 1px solid #2d2d4e;
}
.risk-badge-high   { background:#fff0f0; color:#d63031; border:1.5px solid #ff7675; border-radius:6px; padding:3px 10px; font-weight:700; font-size:12px; }
.risk-badge-medium { background:#fffbf0; color:#e17055; border:1.5px solid #fdcb6e; border-radius:6px; padding:3px 10px; font-weight:700; font-size:12px; }
.risk-badge-low    { background:#f0fff4; color:#00b894; border:1.5px solid #55efc4; border-radius:6px; padding:3px 10px; font-weight:700; font-size:12px; }

.stat-box {
    background: white;
    border-radius: 8px;
    padding: 18px 20px;
    text-align: center;
    border: 1px solid #e8e8e8;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.stat-num { font-size: 32px; font-weight: 700; font-family: 'DM Mono', monospace; color: #1a1a2e; }
.stat-lbl { font-size: 12px; color: #888; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }

.section-title {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #ff4655;
    margin: 24px 0 12px;
}

/* Score ring */
.score-display {
    font-size: 52px;
    font-weight: 700;
    font-family: 'DM Mono', monospace;
}

/* Buttons */
div.stButton > button {
    background: #ff4655;
    color: white;
    border: none;
    border-radius: 6px;
    padding: 10px 24px;
    font-family: 'DM Sans', sans-serif;
    font-size: 14px;
    font-weight: 600;
    width: 100%;
    letter-spacing: 0.3px;
    transition: all 0.15s ease;
}
div.stButton > button:hover {
    background: #d63031;
    transform: translateY(-1px);
}

/* Inputs */
input, textarea {
    border-radius: 6px !important;
    border: 1.5px solid #ddd !important;
    background: white !important;
    color: #1a1a2e !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
}
input:focus, textarea:focus {
    border-color: #ff4655 !important;
    color: #1a1a2e !important;
}
input::placeholder { color: #aaa !important; }

/* Labels */
label { color: #444 !important; font-weight: 500 !important; font-size: 13px !important; }
p, li { color: #333; font-size: 14px; }
h1 { color: #1a1a2e !important; font-family: 'DM Sans', sans-serif !important; }
h2 { color: #1a1a2e !important; }
h3 { color: #ff4655 !important; }

/* Notification dot */
.notif-dot {
    display: inline-block;
    width: 8px; height: 8px;
    background: #ff4655;
    border-radius: 50%;
    margin-left: 6px;
    vertical-align: middle;
}

/* Student card in dashboard */
.student-row {
    background: white;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 8px;
    border: 1px solid #e8e8e8;
    border-left: 4px solid #ddd;
}
.student-row-high   { border-left-color: #ff4655; }
.student-row-medium { border-left-color: #fdcb6e; }
.student-row-low    { border-left-color: #00b894; }

/* Reply bubble */
.reply-msg {
    background: #f8f9fa;
    border-left: 3px solid #ff4655;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin-bottom: 10px;
    font-size: 14px;
    color: #333;
}

/* Padding wrapper */
.page-wrap { padding: 20px 32px; }
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

COLLEGES = [
    "Vivekananda Institute of Professional Studies - Technical Campus (VIPS-TC)",
    "Other",
]
BRANCHES = ["AIDS","CSE","ECE","IT","ME","CE","EEE","Other"]
SECTIONS = ["A","B","C"]

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
        proba_dict.get("Low",0)*10
    )))

def score_info(score):
    if score <= 33:   return "Thriving",       "#00b894", "risk-badge-low",    "🌱"
    elif score <= 66: return "Needs Attention", "#e17055", "risk-badge-medium", "🌤️"
    else:             return "At Risk",         "#d63031", "risk-badge-high",   "🌧️"

def trajectory(scores):
    if len(scores) < 2: return "→", "#888", "First submission"
    diff = scores[-1] - scores[-2]
    if diff > 5:    return "↑", "#d63031", f"+{diff} pts since last survey"
    elif diff < -5: return "↓", "#00b894", f"{diff} pts since last survey — great progress!"
    else:           return "→", "#e17055", "Stable since last survey"

def predict(vals):
    arr      = np.array([vals])
    idx      = model.predict(arr)[0]
    proba    = model.predict_proba(arr)[0]
    label    = le.inverse_transform([idx])[0]
    return label, dict(zip(le.classes_, proba))

def push_notif(name, roll, risk, score, flagged=False):
    if risk not in ("High","Medium"): return
    st.session_state.notifications.insert(0, {
        "id":        len(st.session_state.notifications),
        "ts":        datetime.now().strftime("%d %b %Y, %I:%M %p"),
        "name":      name,
        "roll":      roll,
        "risk":      risk,
        "score":     score,
        "flagged":   flagged,
        "read":      False,
        "expires":   (datetime.now() + timedelta(days=30)).isoformat(),
    })

# ── HEADER ─────────────────────────────────────────────────────────────────────
unread = sum(1 for n in st.session_state.notifications if not n["read"])
if unread > 0:
    alert_text = str(unread) + " new alert" + ("s" if unread > 1 else "")
    notif_part = "<span style='color:#ff4655;font-size:13px;font-weight:600;'>🔔 " + alert_text + "</span>"
else:
    notif_part = ""

header_html = """
<div style="background:#1a1a2e;padding:12px 32px;display:flex;align-items:center;
justify-content:space-between;border-bottom:3px solid #ff4655;">
  <div style="display:flex;align-items:center;gap:12px;">
    <span style="font-size:22px;">🌸</span>
    <span style="color:white;font-family:DM Sans,sans-serif;font-weight:700;font-size:20px;">Burnout Buster</span>
    <span style="color:#666;font-size:13px;margin-left:4px;">VIPS-TC Wellness Platform</span>
  </div>
  <div style="display:flex;align-items:center;gap:20px;">
    NOTIF_PLACEHOLDER
    <span style="color:#666;font-size:12px;">AIDS 260 Practicum · VIPS-TC</span>
  </div>
</div>"""

header_html = header_html.replace("NOTIF_PLACEHOLDER", notif_part)
st.markdown(header_html, unsafe_allow_html=True)

# ── TABS ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏠 Home",
    "📝 Take Survey",
    "🎓 My Portal",
    "🛡️ Counselor",
    "📊 Analytics",
    "📋 Dataset",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — HOME
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="page-wrap">', unsafe_allow_html=True)

    # Hero
    st.markdown("""
    <div style="background:#1a1a2e;border-radius:10px;padding:36px 40px;margin-bottom:24px;
    border-left:5px solid #ff4655;">
      <div style="color:#ff4655;font-size:11px;font-weight:700;letter-spacing:2px;
      text-transform:uppercase;margin-bottom:8px;">VIPS-TC Wellness Initiative</div>
      <div style="color:white;font-family:DM Sans,sans-serif;font-size:28px;font-weight:700;
      margin-bottom:10px;">Know your burnout risk before it hits.</div>
      <div style="color:#aaa;font-size:15px;max-width:580px;line-height:1.6;">
        A 3-minute survey that uses machine learning to predict your burnout risk level.
        Your counselor gets notified if you need support — confidentially.
      </div>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="bb-card">
          <div style="font-size:28px;margin-bottom:10px;">📝</div>
          <div style="font-weight:700;font-size:15px;color:#1a1a2e;margin-bottom:6px;">Take the Survey</div>
          <div style="color:#666;font-size:13px;line-height:1.5;">
            Answer 17 questions about academics, lifestyle, and mental health. Takes ~3 minutes.
          </div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="bb-card">
          <div style="font-size:28px;margin-bottom:10px;">📈</div>
          <div style="font-weight:700;font-size:15px;color:#1a1a2e;margin-bottom:6px;">Track Your Progress</div>
          <div style="color:#666;font-size:13px;line-height:1.5;">
            Log into your personal portal to see your score history and messages from your counselor.
          </div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="bb-card">
          <div style="font-size:28px;margin-bottom:10px;">🛡️</div>
          <div style="font-weight:700;font-size:15px;color:#1a1a2e;margin-bottom:6px;">Get Support</div>
          <div style="color:#666;font-size:13px;line-height:1.5;">
            Results go directly to your counselor who will reach out if you need help. You're not alone.
          </div>
        </div>""", unsafe_allow_html=True)

    # Score guide
    st.markdown('<div class="section-title">Score Guide</div>', unsafe_allow_html=True)
    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown("""
        <div style="background:white;border-radius:8px;padding:18px;border:1px solid #e8e8e8;
        border-top:4px solid #00b894;text-align:center;">
          <div style="font-size:28px;">🌱</div>
          <div style="font-family:DM Mono,monospace;font-size:22px;font-weight:700;color:#00b894;">0 – 33</div>
          <div style="font-weight:700;color:#1a1a2e;margin:4px 0;">Thriving</div>
          <div style="color:#666;font-size:12px;">You're doing great — keep it up!</div>
        </div>""", unsafe_allow_html=True)
    with g2:
        st.markdown("""
        <div style="background:white;border-radius:8px;padding:18px;border:1px solid #e8e8e8;
        border-top:4px solid #fdcb6e;text-align:center;">
          <div style="font-size:28px;">🌤️</div>
          <div style="font-family:DM Mono,monospace;font-size:22px;font-weight:700;color:#e17055;">34 – 66</div>
          <div style="font-weight:700;color:#1a1a2e;margin:4px 0;">Needs Attention</div>
          <div style="color:#666;font-size:12px;">Some areas to work on</div>
        </div>""", unsafe_allow_html=True)
    with g3:
        st.markdown("""
        <div style="background:white;border-radius:8px;padding:18px;border:1px solid #e8e8e8;
        border-top:4px solid #ff4655;text-align:center;">
          <div style="font-size:28px;">🌧️</div>
          <div style="font-family:DM Mono,monospace;font-size:22px;font-weight:700;color:#d63031;">67 – 100</div>
          <div style="font-weight:700;color:#1a1a2e;margin:4px 0;">At Risk</div>
          <div style="color:#666;font-size:12px;">Reach out for support</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SURVEY
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    from database import student_exists, register_student, save_submission, get_student_submissions

    st.markdown('<div class="page-wrap">', unsafe_allow_html=True)
    st.markdown("## 📝 Student Wellness Survey")
    st.markdown("Your responses are **confidential**. Please answer honestly — this helps us support you better.")

    st.markdown('<div class="section-title">About You</div>', unsafe_allow_html=True)
    i1, i2 = st.columns(2)
    with i1:
        s_name  = st.text_input("Full Name *",     placeholder="e.g. Ravi Sharma",     key="s_name")
        s_roll  = st.text_input("Roll Number *",   placeholder="e.g. 01217711924",     key="s_roll")
        s_email = st.text_input("College Email",   placeholder="e.g. ravi@vips.edu",   key="s_email")
        s_age   = st.number_input("Age", min_value=16, max_value=30, value=20,          key="s_age")
    with i2:
        s_college = st.selectbox("College *",  COLLEGES, key="s_college")
        s_branch  = st.selectbox("Branch *",   BRANCHES, key="s_branch")
        s_section = st.selectbox("Section *",  SECTIONS, key="s_section")

        # Password section
        if s_roll and s_roll.strip():
            is_new = not student_exists(s_roll.strip())
        else:
            is_new = True

        if is_new:
            st.markdown("**🔐 Create your portal password**")
            s_pwd  = st.text_input("Create Password *",  type="password", key="s_pwd")
            s_pwd2 = st.text_input("Confirm Password *", type="password", key="s_pwd2")
        else:
            s_pwd = s_pwd2 = None
            if s_roll and s_roll.strip():
                st.info("👋 Welcome back! Your profile is already set up.")

    st.markdown("---")
    st.markdown('<div class="section-title">Academic Life</div>', unsafe_allow_html=True)
    a1, a2, a3 = st.columns(3)
    with a1:
        q_exams   = st.select_slider("Exams per month",      options=list(range(1,9)),  value=4, key="q1")
        q_assign  = st.select_slider("Assignments per week", options=list(range(1,13)), value=5, key="q2")
    with a2:
        q_attend  = st.slider("Attendance pressure (1–10)",  1, 10, 6, key="q3")
        q_cgpa    = st.slider("Current CGPA",  4.0, 10.0, 7.0, step=0.1, key="q4")
    with a3:
        q_backlog = st.select_slider("Active backlogs",       options=list(range(0,9)), value=0, key="q5")
        q_study   = st.select_slider("Study hours per day",   options=list(range(1,13)),value=5, key="q6")

    st.markdown('<div class="section-title">Social & Mental</div>', unsafe_allow_html=True)
    b1, b2, b3 = st.columns(3)
    with b1:
        q_fomo   = st.slider("FOMO level (1–10)",             1, 10, 5, key="q7")
        q_peer   = st.slider("Peer pressure (1–10)",          1, 10, 5, key="q8")
    with b2:
        q_family = st.slider("Family expectations (1–10)",    1, 10, 6, key="q9")
        q_social = st.select_slider("Social media hrs/day",   options=list(range(0,13)),value=3, key="q10")
    with b3:
        q_reject = st.slider("Rejection sensitivity (1–10)",  1, 10, 5, key="q11")
        q_mhv    = st.select_slider("Counselor visits/month", options=list(range(0,6)), value=0, key="q12")

    st.markdown('<div class="section-title">Lifestyle</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        q_sleep  = st.select_slider("Sleep hours/night",      options=list(range(3,11)), value=6, key="q13")
        q_exer   = st.select_slider("Exercise days/week",     options=list(range(0,8)),  value=2, key="q14")
    with c2:
        q_diet   = st.slider("Diet quality (1–10)",           1, 10, 5, key="q15")
        q_conf   = st.slider("Self-confidence (1–10)",        1, 10, 5, key="q16")
    with c3:
        q_support= st.slider("Support from friends/family",   1, 10, 5, key="q17")

    st.markdown("---")
    st.markdown('<div class="section-title">Anything Else?</div>', unsafe_allow_html=True)
    s_note = st.text_area(
        "Is there anything you'd like your counselor to know? (optional)",
        placeholder="e.g. I've been feeling anxious before exams lately...",
        height=90, key="s_note"
    )

    consent = st.checkbox("✅ I understand this is a wellness screening tool, not a clinical diagnosis. I consent to my counselor reviewing my responses.")
    st.markdown("")

    if st.button("💛 Submit Survey & See My Result", use_container_width=True, key="submit_survey"):
        errors = []
        if not s_name or not s_name.strip():  errors.append("Please enter your name")
        if not s_roll or not s_roll.strip():  errors.append("Please enter your roll number")
        if not consent:                        errors.append("Please check the consent box")
        if is_new:
            if not s_pwd:                      errors.append("Please create a password")
            elif s_pwd != s_pwd2:              errors.append("Passwords don't match")

        if errors:
            for e in errors: st.error(e)
        else:
            feat_vals = [q_exams, q_assign, q_attend, q_cgpa, q_backlog, q_study,
                         q_fomo, q_peer, q_family, q_social, q_reject,
                         q_sleep, q_exer, q_diet, q_conf, q_support, q_mhv]
            feat_dict = dict(zip(FEATURES, feat_vals))
            risk, proba = predict(feat_vals)
            score = compute_score(proba)
            slabel, color, badge_css, icon = score_info(score)

            # Register student if new
            if is_new and s_pwd:
                register_student(
                    s_roll.strip(), s_name.strip(), s_email.strip(),
                    s_college, s_branch, s_section, s_age, s_pwd
                )

            # Check persistent flag
            prev = get_student_submissions(s_roll.strip())
            flagged = False
            if not prev.empty and "burnout_risk" in prev.columns and len(prev) >= 1:
                if risk == "High" and prev.iloc[-1].get("burnout_risk","") == "High":
                    flagged = True

            # Save submission — pass name too
            save_submission(s_roll.strip(), s_name.strip(), feat_dict, score, risk, proba, s_note.strip() if s_note else "")

            # Notify counselor
            push_notif(s_name.strip(), s_roll.strip(), risk, score, flagged)

            st.success(f"✅ Submitted! Hi **{s_name.strip()}**, here are your results:")
            st.markdown("---")

            # Result card
            _, rc, _ = st.columns([1,2,1])
            with rc:
                st.markdown(f"""
                <div class="bb-card" style="text-align:center;border-top:5px solid {color};">
                  <div style="font-size:48px;">{icon}</div>
                  <div class="score-display" style="color:{color};">{score}<span style="font-size:20px;font-weight:400;color:#888;">/100</span></div>
                  <div style="font-size:20px;font-weight:700;color:#1a1a2e;margin:8px 0;">{slabel}</div>
                  <span class="{badge_css}">{risk} Risk</span>
                  <div style="color:#666;font-size:13px;margin-top:12px;">
                    {'You\'re doing great — keep maintaining your healthy habits! 🌱' if slabel=='Thriving' else
                     'Some stress signals detected. Small changes can make a big difference.' if slabel=='Needs Attention' else
                     'We\'re here to support you. Please reach out to your counselor — you\'re not alone. 💛'}
                  </div>
                </div>""", unsafe_allow_html=True)

            # Probabilities
            st.markdown("#### Confidence Breakdown")
            pc1, pc2, pc3 = st.columns(3)
            clrs = {"High":"#d63031","Low":"#00b894","Medium":"#e17055"}
            lbls = {"High":"At Risk","Low":"Thriving","Medium":"Needs Attention"}
            for col, k in zip([pc1,pc2,pc3], le.classes_):
                with col:
                    st.markdown(f"""
                    <div class="stat-box">
                      <div style="color:{clrs[k]};font-family:DM Mono,monospace;font-size:24px;font-weight:700;">{proba[k]*100:.1f}%</div>
                      <div style="color:#888;font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:1px;">{lbls[k]}</div>
                    </div>""", unsafe_allow_html=True)

            # Recommendations
            st.markdown("---")
            st.markdown("#### 💡 What You Can Do")
            if slabel == "Thriving":
                st.success("✅ You're in great shape! Keep maintaining your healthy habits.")
            elif slabel == "Needs Attention":
                st.warning("⚠️ You're showing some stress signals. Try: sleeping 7+ hrs, exercising 20 min/day, reducing social media, talking to someone you trust.")
            else:
                st.error("🚨 You're going through a tough time. Please reach out to your counselor. You are not alone.")
                st.markdown("**Free helplines:** iCall: 9152987821 · Vandrevala: 1860-2662-345 (24/7) · NIMHANS: 080-46110007")

            if flagged:
                st.error("🚩 This is your second consecutive At Risk result. Your counselor has been specially alerted.")

    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — STUDENT PORTAL
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    from database import verify_student, get_student_submissions, get_replies, mark_replies_read, get_reminder

    st.markdown('<div class="page-wrap">', unsafe_allow_html=True)
    st.markdown("## 🎓 My Wellness Portal")

    if not st.session_state.student_logged_in:
        st.markdown("Log in with your roll number and the password you created when you took the survey.")
        _, lc, _ = st.columns([1,1.2,1])
        with lc:
            st.markdown('<div class="bb-card">', unsafe_allow_html=True)
            p_roll = st.text_input("Roll Number", placeholder="e.g. 01217711924", key="p_roll")
            p_pwd  = st.text_input("Password",    type="password",                key="p_pwd")
            if st.button("Log In to My Portal", use_container_width=True, key="portal_login"):
                if p_roll and p_pwd:
                    student = verify_student(p_roll.strip(), p_pwd.strip())
                    if student:
                        st.session_state.student_logged_in = True
                        st.session_state.student_data      = student
                        st.rerun()
                    else:
                        st.error("❌ Incorrect roll number or password. If you haven't taken the survey yet, please do so first to create your account.")
                else:
                    st.warning("Please enter both roll number and password.")
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        sd   = st.session_state.student_data
        name = sd.get("name", "Student")
        roll = str(sd.get("roll_number", ""))

        # Header
        col_h1, col_h2 = st.columns([4,1])
        with col_h1:
            st.markdown(f"## 👋 Welcome back, {name.split()[0]}!")
        with col_h2:
            if st.button("Logout", key="s_out"):
                st.session_state.student_logged_in = False
                st.session_state.student_data = {}
                st.rerun()

        # Reminder banner
        try:
            reminder = get_reminder()
            if reminder and reminder.get("next_due"):
                next_due  = datetime.fromisoformat(str(reminder["next_due"]))
                days_left = (next_due - datetime.now()).days
                if days_left <= 3:
                    st.warning(f"📅 Your next wellness survey is due {'today' if days_left<=0 else f'in {days_left} day(s)'}! Head to **📝 Take Survey**.")
                elif days_left <= 7:
                    st.info(f"📅 Your next wellness check-in is in {days_left} days.")
        except Exception:
            pass

        # Load submissions
        subs = get_student_submissions(roll)

        if subs.empty:
            st.markdown("""
            <div class="bb-card" style="text-align:center;padding:40px;">
              <div style="font-size:32px;">📝</div>
              <div style="font-weight:600;color:#1a1a2e;margin:10px 0;">You haven't taken the survey yet</div>
              <div style="color:#666;">Head to the <strong>📝 Take Survey</strong> tab to get your first wellness score.</div>
            </div>""", unsafe_allow_html=True)
        else:
            scores = []
            if "burnout_score" in subs.columns:
                scores = pd.to_numeric(subs["burnout_score"], errors="coerce").fillna(0).astype(int).tolist()

            latest_score = scores[-1] if scores else 0
            slabel, color, badge_css, icon = score_info(latest_score)
            arrow, tcolor, tmsg = trajectory(scores)

            # Stats row
            s1, s2, s3, s4 = st.columns(4)
            with s1:
                st.markdown(f"""
                <div class="stat-box" style="border-top:4px solid {color};">
                  <div class="stat-num" style="color:{color};">{latest_score}</div>
                  <div class="stat-lbl">Latest Score</div>
                </div>""", unsafe_allow_html=True)
            with s2:
                st.markdown(f"""
                <div class="stat-box">
                  <div style="font-size:32px;font-weight:700;color:{tcolor};">{arrow}</div>
                  <div class="stat-lbl">Trend</div>
                </div>""", unsafe_allow_html=True)
            with s3:
                st.markdown(f"""
                <div class="stat-box">
                  <div class="stat-num">{len(subs)}</div>
                  <div class="stat-lbl">Surveys Taken</div>
                </div>""", unsafe_allow_html=True)
            with s4:
                st.markdown(f"""
                <div class="stat-box">
                  <span class="{badge_css}">{slabel}</span>
                  <div class="stat-lbl" style="margin-top:8px;">Current Status</div>
                </div>""", unsafe_allow_html=True)

            st.markdown(f'<div style="color:{tcolor};font-size:13px;margin:8px 0 16px;">{tmsg}</div>', unsafe_allow_html=True)

            # Score chart
            if len(scores) > 1:
                st.markdown('<div class="section-title">Score History</div>', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(10, 3))
                fig.patch.set_facecolor("white")
                ax.set_facecolor("#f8f8f8")
                xs = list(range(1, len(scores)+1))
                ax.fill_between(xs, scores, alpha=0.1, color="#ff4655")
                ax.plot(xs, scores, color="#ff4655", linewidth=2.5, marker="o",
                        markersize=8, markerfacecolor="white",
                        markeredgewidth=2.5, markeredgecolor="#ff4655")
                ax.axhline(33, color="#00b894", linewidth=1, linestyle="--", alpha=0.5, label="Thriving")
                ax.axhline(66, color="#fdcb6e", linewidth=1, linestyle="--", alpha=0.5, label="Needs Attention")
                for i, s in enumerate(scores):
                    ax.annotate(str(s), (xs[i], s), textcoords="offset points",
                                xytext=(0,10), ha="center", fontsize=10,
                                color="#1a1a2e", fontweight="bold")
                ax.set_xlabel("Survey #", color="#888", fontsize=11)
                ax.set_ylabel("Score", color="#888", fontsize=11)
                ax.set_title("Your Wellness Journey", color="#1a1a2e", fontweight="bold")
                ax.tick_params(colors="#888"); ax.spines[:].set_visible(False)
                ax.set_ylim(0, 115); ax.legend(fontsize=9)
                plt.tight_layout()
                st.pyplot(fig)

            # Submission history table
            with st.expander("📋 All My Submissions"):
                show_cols = [c for c in ["timestamp","burnout_score","burnout_risk"] if c in subs.columns]
                st.dataframe(subs[show_cols].rename(columns={
                    "timestamp":"Date","burnout_score":"Score","burnout_risk":"Result"
                }), use_container_width=True)

        # Counselor messages
        st.markdown("---")
        st.markdown('<div class="section-title">Messages from Your Counselor</div>', unsafe_allow_html=True)
        replies = get_replies(roll)
        if replies.empty:
            st.markdown("""
            <div class="bb-card" style="color:#666;text-align:center;padding:24px;">
              No messages yet. Your counselor will reach out after reviewing your survey. 💛
            </div>""", unsafe_allow_html=True)
        else:
            mark_replies_read(roll)
            for _, rep in replies.iterrows():
                ts  = rep.get("timestamp","")
                msg = rep.get("counselor_message","")
                st.markdown(f"""
                <div class="reply-msg">
                  <div style="font-size:11px;color:#888;margin-bottom:6px;">Your Counselor · {ts}</div>
                  {msg}
                </div>""", unsafe_allow_html=True)

        # Focus areas
        if not subs.empty and scores:
            st.markdown("---")
            st.markdown('<div class="section-title">Areas to Focus On</div>', unsafe_allow_html=True)
            feat_imp = meta["feature_importances"]
            last_row = subs.iloc[-1]
            good_when_high = ["sleep_hours","exercise_days","diet_quality","confidence","support_system","cgpa","study_hours_per_day"]
            risky = []
            for feat, _ in sorted(feat_imp.items(), key=lambda x:-x[1])[:8]:
                if feat in last_row:
                    val = float(last_row[feat]) if last_row[feat] != "" else 0
                    is_risky = (val >= 7 and feat not in good_when_high) or (val <= 4 and feat in good_when_high)
                    if is_risky:
                        risky.append((feat.replace("_"," ").title(), val))
            if risky:
                for fname, fval in risky[:3]:
                    st.markdown(f"""
                    <div style="background:white;border-left:3px solid #ff4655;border-radius:0 6px 6px 0;
                    padding:10px 14px;margin-bottom:8px;border:1px solid #eee;border-left-width:3px;border-left-color:#ff4655;">
                      <strong style="color:#1a1a2e;">{fname}</strong>
                      <span style="color:#888;font-size:12px;"> — current value: {fval}</span>
                    </div>""", unsafe_allow_html=True)
            else:
                st.success("All key indicators are looking healthy! 🌱")

    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — COUNSELOR DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    from database import get_all_submissions, get_all_students, upsert_counselor_action, get_counselor_action, save_reply, save_reminder

    st.markdown('<div class="page-wrap">', unsafe_allow_html=True)

    if not st.session_state.counselor_logged_in:
        st.markdown("## 🔐 Counselor Login")
        st.markdown("This section is restricted to authorised counselors only.")
        _, lc, _ = st.columns([1,1.2,1])
        with lc:
            st.markdown('<div class="bb-card">', unsafe_allow_html=True)
            pwd = st.text_input("Password", type="password", placeholder="••••••••", key="c_pwd")
            if st.button("Login", use_container_width=True, key="c_login"):
                COUNSELOR_PASSWORD = "ProjectAlpha01"
                try:
                    COUNSELOR_PASSWORD = st.secrets["COUNSELOR_PASSWORD"]
                except Exception:
                    pass
                if pwd == COUNSELOR_PASSWORD:
                    st.session_state.counselor_logged_in = True
                    st.rerun()
                else:
                    st.error("❌ Incorrect password.")
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        hc1, hc2 = st.columns([4,1])
        with hc1: st.markdown("## 🛡️ Counselor Dashboard")
        with hc2:
            if st.button("Logout", key="c_out"):
                st.session_state.counselor_logged_in = False
                st.rerun()

        # ── Notifications ─────────────────────────────────────────────────────
        notifs = [n for n in st.session_state.notifications
                  if datetime.fromisoformat(n["expires"]) > datetime.now()]
        st.session_state.notifications = notifs
        unread_c = sum(1 for n in notifs if not n["read"])

        with st.expander(f"🔔 Alerts {'· ' + str(unread_c) + ' unread' if unread_c else '· all read'}", expanded=unread_c>0):
            if not notifs:
                st.info("No alerts yet. They appear when students submit the survey.")
            else:
                if st.button("Mark all read", key="mark_all"):
                    for n in st.session_state.notifications: n["read"] = True
                    st.rerun()
                for n in notifs:
                    brd  = "#ff4655" if n["risk"]=="High" else "#fdcb6e"
                    bg   = "#fff8f8" if n["risk"]=="High" else "#fffdf0"
                    icon = "🌧️" if n["risk"]=="High" else "🌤️"
                    new  = ' <span style="background:#ff4655;color:white;font-size:10px;padding:1px 6px;border-radius:10px;">NEW</span>' if not n["read"] else ""
                    flag = ' <span style="background:#1a1a2e;color:white;font-size:10px;padding:1px 6px;border-radius:10px;">🚩 PERSISTENT</span>' if n.get("flagged") else ""
                    st.markdown(f"""
                    <div style="background:{bg};border:1px solid {brd};border-radius:8px;
                    padding:12px 16px;margin-bottom:8px;opacity:{'1' if not n['read'] else '0.55'};">
                      <strong style="color:#1a1a2e;">{n['name']}</strong>
                      <span style="color:#888;font-size:12px;"> · {n['roll']}</span>
                      <span style="background:{brd};color:white;font-size:11px;font-weight:700;
                      padding:2px 8px;border-radius:12px;margin-left:8px;">{n['score']}/100 · {n['risk']}</span>
                      {new}{flag}
                      <span style="float:right;color:#aaa;font-size:11px;">{n['ts']}</span>
                    </div>""", unsafe_allow_html=True)
                    if not n["read"]:
                        if st.button("✓ Mark read", key=f"nr_{n['id']}"):
                            n["read"] = True
                            st.rerun()

        # ── Reminder settings ─────────────────────────────────────────────────
        with st.expander("📅 Survey Reminder Settings"):
            freq = st.select_slider("Reminder frequency", options=[7,14,21,30,60], value=30,
                                    format_func=lambda x: f"Every {x} days")
            if st.button("💾 Save Reminder Schedule", key="save_rem"):
                save_reminder(freq)
                st.success(f"✅ Students will be reminded every {freq} days.")

        st.markdown("---")

        # ── Load all data ──────────────────────────────────────────────────────
        all_subs  = get_all_submissions()
        all_studs = get_all_students()

        if all_subs.empty:
            st.info("📭 No student submissions yet.")
        else:
            # Build student name lookup
            name_lookup = {}
            if not all_studs.empty and "roll_number" in all_studs.columns and "name" in all_studs.columns:
                for _, row in all_studs.iterrows():
                    name_lookup[str(row["roll_number"]).strip()] = str(row.get("name",""))

            # Also grab names from submissions directly (saved there too)
            if "student_name" in all_subs.columns and "roll_number" in all_subs.columns:
                for _, row in all_subs.iterrows():
                    r = str(row.get("roll_number","")).strip()
                    n = str(row.get("student_name","")).strip()
                    if r and n and r not in name_lookup:
                        name_lookup[r] = n

            # Stats
            unique_rolls = all_subs["roll_number"].nunique() if "roll_number" in all_subs.columns else 0
            high   = int((all_subs["burnout_risk"]=="High").sum())   if "burnout_risk" in all_subs.columns else 0
            medium = int((all_subs["burnout_risk"]=="Medium").sum()) if "burnout_risk" in all_subs.columns else 0
            low    = int((all_subs["burnout_risk"]=="Low").sum())    if "burnout_risk" in all_subs.columns else 0

            m1,m2,m3,m4 = st.columns(4)
            with m1: st.markdown(f'<div class="stat-box"><div class="stat-num">{unique_rolls}</div><div class="stat-lbl">Students</div></div>', unsafe_allow_html=True)
            with m2: st.markdown(f'<div class="stat-box" style="border-top:3px solid #ff4655;"><div class="stat-num" style="color:#d63031;">{high}</div><div class="stat-lbl">At Risk</div></div>', unsafe_allow_html=True)
            with m3: st.markdown(f'<div class="stat-box" style="border-top:3px solid #fdcb6e;"><div class="stat-num" style="color:#e17055;">{medium}</div><div class="stat-lbl">Needs Attention</div></div>', unsafe_allow_html=True)
            with m4: st.markdown(f'<div class="stat-box" style="border-top:3px solid #00b894;"><div class="stat-num" style="color:#00b894;">{low}</div><div class="stat-lbl">Thriving</div></div>', unsafe_allow_html=True)

            st.markdown("")

            # Filters
            fc1, fc2, fc3 = st.columns(3)
            with fc1: rf = st.multiselect("Risk", ["High","Medium","Low"], default=["High","Medium","Low"], key="dash_rf")
            with fc2: search = st.text_input("🔎 Search name or roll", "", key="dash_search")
            with fc3: show_flag = st.checkbox("🚩 Persistent At Risk only", key="dash_flag")

            # Get latest submission per student
            if "roll_number" in all_subs.columns and "timestamp" in all_subs.columns:
                latest = all_subs.sort_values("timestamp").groupby("roll_number").last().reset_index()
            else:
                latest = all_subs.copy()

            if rf and "burnout_risk" in latest.columns:
                latest = latest[latest["burnout_risk"].isin(rf)]
            if search:
                roll_match = latest["roll_number"].astype(str).str.contains(search, na=False)
                name_match = pd.Series([search.lower() in name_lookup.get(str(r),"").lower()
                                        for r in latest["roll_number"]], index=latest.index)
                latest = latest[roll_match | name_match]

            st.markdown(f"Showing **{len(latest)}** students")
            st.markdown("")

            for _, row in latest.iterrows():
                roll_d  = str(row.get("roll_number","")).strip()
                risk_d  = row.get("burnout_risk","Unknown")
                score_d = int(row.get("burnout_score",0)) if str(row.get("burnout_score",0)).isdigit() else 0
                ts_d    = row.get("timestamp","")
                note_d  = row.get("student_note","")
                sname_d = name_lookup.get(roll_d, roll_d)

                # Check persistent flag
                stud_subs = all_subs[all_subs["roll_number"].astype(str).str.strip()==roll_d] if "roll_number" in all_subs.columns else pd.DataFrame()
                flagged_d = False
                if len(stud_subs) >= 2 and "burnout_risk" in stud_subs.columns:
                    last2 = stud_subs.sort_values("timestamp").tail(2)["burnout_risk"].tolist()
                    flagged_d = all(r=="High" for r in last2)

                if show_flag and not flagged_d: continue

                action_d = get_counselor_action(roll_d)
                status_d = action_d.get("status","Pending")
                notes_d  = action_d.get("notes","")

                brd_d = {"High":"#ff4655","Medium":"#fdcb6e","Low":"#00b894"}.get(risk_d,"#ddd")
                risk_css = "high" if risk_d=="High" else "medium" if risk_d=="Medium" else "low"
                badge_d = f'<span class="risk-badge-{risk_css}">{risk_d}</span>'
                flag_badge = ('<span style="background:#1a1a2e;color:#ff4655;font-size:10px;font-weight:700;padding:2px 8px;border-radius:10px;border:1px solid #ff4655;">🚩 PERSISTENT</span>' if flagged_d else "")

                left_html = f"""<strong style="color:#1a1a2e;font-size:15px;">{sname_d}</strong>
                <span style="color:#888;font-size:12px;margin-left:8px;">Roll: {roll_d}</span>
                {flag_badge}"""

                right_html = f"""{badge_d}
                <span style="color:#888;font-size:12px;margin-left:8px;">{score_d}/100</span>
                <div style="color:#aaa;font-size:11px;margin-top:2px;">{ts_d}</div>"""

                note_html = f"""<div style="background:#f8f8f8;border-radius:4px;padding:8px 10px;margin-top:10px;color:#555;font-size:13px;"><strong>Student note:</strong> {note_d}</div>""" if note_d and str(note_d).strip() and str(note_d).strip() != "nan" else ""

                st.markdown(f"""
                <div class="student-row student-row-{risk_css}">
                  <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div>{left_html}</div>
                    <div style="text-align:right;">{right_html}</div>
                  </div>
                  {note_html}
                </div>""", unsafe_allow_html=True)

                with st.expander(f"Actions for {sname_d} ({roll_d})"):
                    ac1, ac2 = st.columns([1,2])
                    with ac1:
                        new_status = st.selectbox("Status",
                            ["Pending","Contacted","No Action Needed"],
                            index=["Pending","Contacted","No Action Needed"].index(status_d)
                            if status_d in ["Pending","Contacted","No Action Needed"] else 0,
                            key=f"st_{roll_d}")
                        new_notes = st.text_area("Notes", value=notes_d if str(notes_d) != "nan" else "",
                                                 height=80, key=f"nt_{roll_d}")
                        if st.button("💾 Save Status", key=f"sv_{roll_d}"):
                            upsert_counselor_action(roll_d, new_status, new_notes, flagged_d)
                            st.success("✅ Status updated!")
                    with ac2:
                        st.markdown("**💌 Send a Private Message to Student**")
                        reply_msg = st.text_area("Your message",
                            placeholder="Hi! I reviewed your survey and I'd like to check in with you...",
                            height=100, key=f"rp_{roll_d}")
                        if st.button("📤 Send Message", key=f"send_{roll_d}"):
                            if reply_msg and reply_msg.strip():
                                save_reply(roll_d, reply_msg.strip())
                                st.success("✅ Message sent! Student will see it in their portal.")
                            else:
                                st.warning("Please type a message first.")

            st.markdown("---")
            st.download_button("⬇️ Export All Data (CSV)",
                               data=all_subs.to_csv(index=False).encode("utf-8"),
                               file_name="burnout_submissions.csv", mime="text/csv")

    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    from database import get_all_submissions, get_all_students

    st.markdown('<div class="page-wrap">', unsafe_allow_html=True)
    st.markdown("## 📊 Analytics")

    if not st.session_state.counselor_logged_in:
        st.warning("🔐 Please log in as a counselor to view analytics.")
    else:
        all_subs  = get_all_submissions()
        all_studs = get_all_students()

        if all_subs.empty:
            st.info("No data yet. Analytics will appear once students start submitting surveys.")
        else:
            if not all_studs.empty and "roll_number" in all_studs.columns:
                merge_cols = [c for c in ["roll_number","branch","section","age","college"] if c in all_studs.columns]
                all_subs = all_subs.merge(all_studs[merge_cols].astype(str), on="roll_number", how="left")

            latest = all_subs.sort_values("timestamp").groupby("roll_number").last().reset_index() if "roll_number" in all_subs.columns else all_subs
            if "burnout_score" in latest.columns:
                latest["burnout_score"] = pd.to_numeric(latest["burnout_score"], errors="coerce").fillna(0)

            total   = len(latest)
            avg_sc  = int(latest["burnout_score"].mean()) if "burnout_score" in latest.columns else 0
            at_risk = int((latest["burnout_risk"]=="High").sum())  if "burnout_risk" in latest.columns else 0
            thriving= int((latest["burnout_risk"]=="Low").sum())   if "burnout_risk" in latest.columns else 0

            m1,m2,m3,m4 = st.columns(4)
            with m1: st.markdown(f'<div class="stat-box"><div class="stat-num">{total}</div><div class="stat-lbl">Total Students</div></div>', unsafe_allow_html=True)
            with m2: st.markdown(f'<div class="stat-box"><div class="stat-num">{avg_sc}/100</div><div class="stat-lbl">Avg Score</div></div>', unsafe_allow_html=True)
            with m3: st.markdown(f'<div class="stat-box" style="border-top:3px solid #ff4655;"><div class="stat-num" style="color:#d63031;">{at_risk}</div><div class="stat-lbl">At Risk</div></div>', unsafe_allow_html=True)
            with m4: st.markdown(f'<div class="stat-box" style="border-top:3px solid #00b894;"><div class="stat-num" style="color:#00b894;">{thriving}</div><div class="stat-lbl">Thriving</div></div>', unsafe_allow_html=True)

            st.markdown("")
            fig_c1, fig_c2 = st.columns(2)

            # Risk pie
            with fig_c1:
                st.markdown('<div class="section-title">Risk Distribution</div>', unsafe_allow_html=True)
                if "burnout_risk" in latest.columns:
                    counts = latest["burnout_risk"].value_counts()
                    fig, ax = plt.subplots(figsize=(5,4))
                    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
                    clrs = [{"High":"#ff4655","Medium":"#fdcb6e","Low":"#00b894"}.get(l,"#ccc") for l in counts.index]
                    wedges, texts, autotexts = ax.pie(counts.values, labels=counts.index,
                        autopct="%1.0f%%", colors=clrs, startangle=90,
                        wedgeprops=dict(edgecolor="white",linewidth=2))
                    for at in autotexts: at.set_color("white"); at.set_fontweight("bold")
                    ax.set_title("Students by Risk Level", color="#1a1a2e", fontweight="bold")
                    plt.tight_layout(); st.pyplot(fig)

            # Branch avg
            with fig_c2:
                st.markdown('<div class="section-title">Average Score by Branch</div>', unsafe_allow_html=True)
                if "branch" in latest.columns and "burnout_score" in latest.columns:
                    branch_avg = latest.groupby("branch")["burnout_score"].mean().sort_values(ascending=False)
                    fig, ax = plt.subplots(figsize=(5,4))
                    fig.patch.set_facecolor("white"); ax.set_facecolor("#f8f8f8")
                    clrs = ["#ff4655" if v>66 else "#fdcb6e" if v>33 else "#00b894" for v in branch_avg.values]
                    bars = ax.barh(branch_avg.index[::-1], branch_avg.values[::-1], color=clrs[::-1], height=0.6)
                    for bar, val in zip(bars, branch_avg.values[::-1]):
                        ax.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2,
                                f"{val:.0f}", va="center", color="#333", fontsize=10, fontweight="bold")
                    ax.set_xlabel("Avg Score", color="#888"); ax.set_title("By Branch", color="#1a1a2e", fontweight="bold")
                    ax.tick_params(colors="#888"); ax.spines[:].set_visible(False); ax.set_xlim(0,115)
                    plt.tight_layout(); st.pyplot(fig)
                else:
                    st.info("Branch data appears once students submit with branch info.")

            # Trend over time
            st.markdown('<div class="section-title">Burnout Trend Over Time</div>', unsafe_allow_html=True)
            if "timestamp" in all_subs.columns and "burnout_score" in all_subs.columns:
                try:
                    all_subs["date"] = pd.to_datetime(all_subs["timestamp"]).dt.date
                    all_subs["burnout_score"] = pd.to_numeric(all_subs["burnout_score"], errors="coerce")
                    trend = all_subs.groupby("date")["burnout_score"].mean().reset_index()
                    if len(trend) > 1:
                        fig, ax = plt.subplots(figsize=(10,3.5))
                        fig.patch.set_facecolor("white"); ax.set_facecolor("#f8f8f8")
                        ax.fill_between(range(len(trend)), trend["burnout_score"], alpha=0.1, color="#ff4655")
                        ax.plot(range(len(trend)), trend["burnout_score"], color="#ff4655",
                                linewidth=2.5, marker="o", markersize=6,
                                markerfacecolor="white", markeredgewidth=2, markeredgecolor="#ff4655")
                        ax.axhline(33, color="#00b894", linewidth=1, linestyle="--", alpha=0.5)
                        ax.axhline(66, color="#fdcb6e", linewidth=1, linestyle="--", alpha=0.5)
                        ax.set_xticks(range(len(trend)))
                        ax.set_xticklabels([str(d) for d in trend["date"]], rotation=30, ha="right", fontsize=9)
                        ax.set_ylabel("Avg Score", color="#888")
                        ax.set_title("Average Burnout Score Over Time", color="#1a1a2e", fontweight="bold")
                        ax.tick_params(colors="#888"); ax.spines[:].set_visible(False)
                        plt.tight_layout(); st.pyplot(fig)
                    else:
                        st.info("Trend chart appears once there are submissions on multiple dates.")
                except Exception:
                    st.info("Trend data will appear as more submissions come in.")

            # Section comparison
            if "section" in latest.columns and "burnout_score" in latest.columns:
                st.markdown('<div class="section-title">Section Comparison</div>', unsafe_allow_html=True)
                sec_avg = latest.groupby("section")["burnout_score"].mean().sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(6,3))
                fig.patch.set_facecolor("white"); ax.set_facecolor("#f8f8f8")
                clrs = ["#ff4655" if v>66 else "#fdcb6e" if v>33 else "#00b894" for v in sec_avg.values]
                ax.bar(sec_avg.index, sec_avg.values, color=clrs, width=0.4, edgecolor="white", linewidth=2)
                for i,(sec,val) in enumerate(sec_avg.items()):
                    ax.text(i, val+1, f"{val:.0f}", ha="center", color="#333", fontweight="bold")
                ax.set_ylabel("Avg Score", color="#888")
                ax.set_title("Average Score by Section", color="#1a1a2e", fontweight="bold")
                ax.tick_params(colors="#888"); ax.spines[:].set_visible(False); ax.set_ylim(0,115)
                plt.tight_layout(); st.pyplot(fig)

    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — DATASET
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown('<div class="page-wrap">', unsafe_allow_html=True)
    st.markdown("## 📋 Training Dataset")
    try:
        df_t = pd.read_csv("burnout_dataset.csv")
        t1,t2,t3,t4 = st.columns(4)
        with t1: st.markdown(f'<div class="stat-box"><div class="stat-num">{len(df_t)}</div><div class="stat-lbl">Records</div></div>', unsafe_allow_html=True)
        with t2: st.markdown(f'<div class="stat-box" style="border-top:3px solid #ff4655;"><div class="stat-num" style="color:#d63031;">{int((df_t["burnout_risk"]=="High").sum())}</div><div class="stat-lbl">At Risk</div></div>', unsafe_allow_html=True)
        with t3: st.markdown(f'<div class="stat-box" style="border-top:3px solid #fdcb6e;"><div class="stat-num" style="color:#e17055;">{int((df_t["burnout_risk"]=="Medium").sum())}</div><div class="stat-lbl">Needs Attention</div></div>', unsafe_allow_html=True)
        with t4: st.markdown(f'<div class="stat-box" style="border-top:3px solid #00b894;"><div class="stat-num" style="color:#00b894;">{int((df_t["burnout_risk"]=="Low").sum())}</div><div class="stat-lbl">Thriving</div></div>', unsafe_allow_html=True)
        st.markdown("")
        rf3 = st.multiselect("Filter", ["High","Medium","Low"], default=["High","Medium","Low"])
        def c_risk(val):
            return {"High":"background-color:#fff0f0;color:#d63031",
                    "Medium":"background-color:#fffbf0;color:#e17055",
                    "Low":"background-color:#f0fff4;color:#00b894"}.get(val,"")
        st.dataframe(df_t[df_t["burnout_risk"].isin(rf3)].style.applymap(c_risk, subset=["burnout_risk"]),
                     use_container_width=True, height=400)
        st.download_button("⬇️ Download CSV", data=df_t.to_csv(index=False).encode(),
                           file_name="burnout_dataset.csv", mime="text/csv")
    except FileNotFoundError:
        st.error("Run `python train_model.py` first.")
    st.markdown('</div>', unsafe_allow_html=True)

# ── FOOTER ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:#1a1a2e;padding:16px 32px;margin-top:20px;
border-top:2px solid #ff4655;text-align:center;">
  <span style="color:#666;font-size:12px;">
    🌸 Burnout Buster v4 · AIDS 260 Practicum · VIPS-TC ·
    Yash Choudhary · Mohit Kumar · Lakshay · Supervisor: Dr Sapna Yadav
  </span>
</div>""", unsafe_allow_html=True)
