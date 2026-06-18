"""
app.py — Burnout Buster v6
Design: Warm & Energetic + Magazine Style + Fade/Slide Animations
Colors: Cream, Coral, Amber — humanized, real, editorial
"""
import streamlit as st
import joblib, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Burnout Buster",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700;900&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

/* ── Hide Streamlit chrome ── */
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="collapsedControl"],
.stApp > header, header, #MainMenu, footer { display: none !important; }

.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── Background — warm cream gradient ── */
.stApp {
    background: linear-gradient(160deg, #fdf6ee 0%, #fef9f3 35%, #fff5f0 65%, #fdf0e8 100%);
}

/* ── Animations ── */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(24px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeInLeft {
    from { opacity: 0; transform: translateX(-24px); }
    to   { opacity: 1; transform: translateX(0); }
}
@keyframes fadeInRight {
    from { opacity: 0; transform: translateX(24px); }
    to   { opacity: 1; transform: translateX(0); }
}
@keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}
@keyframes pulse-soft {
    0%, 100% { transform: scale(1); }
    50%       { transform: scale(1.03); }
}

.anim-up    { animation: fadeInUp   0.6s ease both; }
.anim-left  { animation: fadeInLeft 0.6s ease both; }
.anim-right { animation: fadeInRight 0.6s ease both; }
.anim-fade  { animation: fadeIn     0.5s ease both; }
.anim-up.d1 { animation-delay: 0.1s; }
.anim-up.d2 { animation-delay: 0.2s; }
.anim-up.d3 { animation-delay: 0.3s; }
.anim-up.d4 { animation-delay: 0.4s; }

/* ── Tabs — clean editorial nav ── */
.stTabs [data-baseweb="tab-list"] {
    background: white;
    border-radius: 0;
    padding: 0 32px;
    gap: 0;
    border-bottom: 2px solid #f0e6d8;
    box-shadow: 0 2px 12px rgba(200,140,80,0.08);
}
.stTabs [data-baseweb="tab"] {
    color: #b09070 !important;
    border-radius: 0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 18px 22px !important;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    border-bottom: 3px solid transparent !important;
    margin-bottom: -2px;
    transition: all 0.25s ease;
}
.stTabs [aria-selected="true"] {
    background: transparent !important;
    color: #d4603a !important;
    border-bottom: 3px solid #d4603a !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #d4603a !important;
    background: rgba(212,96,58,0.04) !important;
}
.stTabs [data-baseweb="tab-panel"] { padding: 0 !important; }

/* ── Buttons ── */
div.stButton > button {
    background: linear-gradient(135deg, #d4603a, #e8855a);
    color: white;
    border: none;
    border-radius: 50px;
    padding: 12px 32px;
    font-family: 'DM Sans', sans-serif;
    font-size: 14px;
    font-weight: 600;
    width: 100%;
    letter-spacing: 0.3px;
    transition: all 0.25s ease;
    box-shadow: 0 4px 20px rgba(212,96,58,0.25);
}
div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 28px rgba(212,96,58,0.35);
    background: linear-gradient(135deg, #c05530, #d4603a);
}

/* ── Inputs ── */
input, textarea, select {
    background: #fffaf6 !important;
    border: 1.5px solid #e8d5c0 !important;
    border-radius: 12px !important;
    color: #3d2b1f !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    transition: border-color 0.2s ease !important;
}
input:focus, textarea:focus {
    border-color: #d4603a !important;
    box-shadow: 0 0 0 3px rgba(212,96,58,0.12) !important;
    background: white !important;
}
input { color: #3d2b1f !important; }
label {
    color: #8a6a55 !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* ── Typography ── */
p, li { color: #6b4f3f; font-size: 15px; line-height: 1.7; }
h1 { color: #2d1a0e !important; font-family: 'Playfair Display', serif !important; }
h2 { color: #3d2b1f !important; font-family: 'Playfair Display', serif !important; }
h3 { color: #d4603a !important; font-family: 'DM Sans', sans-serif !important; }

/* ── Cards ── */
.mag-card {
    background: white;
    border-radius: 20px;
    padding: 28px 32px;
    margin-bottom: 16px;
    box-shadow: 0 4px 24px rgba(180,120,60,0.08);
    border: 1px solid #f0e6d8;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}
.mag-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 32px rgba(180,120,60,0.14);
}
.mag-card-coral  { border-top: 4px solid #d4603a; }
.mag-card-amber  { border-top: 4px solid #f5a623; }
.mag-card-green  { border-top: 4px solid #48b87a; }
.mag-card-cream  { background: linear-gradient(135deg, #fff8f0, #fff3e8); border: 1px solid #f0dcc8; }

/* ── Stat boxes ── */
.stat-pill {
    background: white;
    border-radius: 16px;
    padding: 22px 20px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(180,120,60,0.08);
    border: 1px solid #f0e6d8;
    transition: transform 0.2s ease;
}
.stat-pill:hover { transform: translateY(-2px); }
.stat-num {
    font-family: 'Playfair Display', serif;
    font-size: 42px;
    font-weight: 700;
    color: #2d1a0e;
    line-height: 1;
}
.stat-lbl {
    font-size: 11px;
    color: #b09070;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 8px;
}

/* ── Risk badges ── */
.badge-thriving  { background: #e8f7ef; color: #2a7d4f; border: 1.5px solid #a8dfc0; border-radius: 50px; padding: 4px 14px; font-size: 12px; font-weight: 700; letter-spacing: 0.5px; display: inline-block; }
.badge-attention { background: #fff8e8; color: #b07820; border: 1.5px solid #f5d890; border-radius: 50px; padding: 4px 14px; font-size: 12px; font-weight: 700; letter-spacing: 0.5px; display: inline-block; }
.badge-risk      { background: #fdf0ec; color: #c04020; border: 1.5px solid #f0b8a8; border-radius: 50px; padding: 4px 14px; font-size: 12px; font-weight: 700; letter-spacing: 0.5px; display: inline-block; }

/* ── Section divider ── */
.sec-divider {
    display: flex;
    align-items: center;
    gap: 16px;
    margin: 32px 0 20px;
}
.sec-divider-line { flex: 1; height: 1px; background: linear-gradient(90deg, #f0dcc8, transparent); }
.sec-divider-text {
    font-family: 'DM Sans', sans-serif;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #d4603a;
    white-space: nowrap;
}

/* ── Student row in dashboard ── */
.student-card {
    background: white;
    border-radius: 16px;
    padding: 18px 22px;
    margin-bottom: 10px;
    border: 1.5px solid #f0e6d8;
    border-left: 5px solid #e8d5c0;
    box-shadow: 0 2px 12px rgba(180,120,60,0.06);
    transition: all 0.2s ease;
}
.student-card:hover {
    box-shadow: 0 6px 24px rgba(180,120,60,0.12);
    transform: translateX(4px);
}
.student-card-risk     { border-left-color: #d4603a; }
.student-card-attention{ border-left-color: #f5a623; }
.student-card-thriving { border-left-color: #48b87a; }

/* ── Reply bubble ── */
.reply-card {
    background: linear-gradient(135deg, #fff8f0, #fff3e8);
    border-left: 4px solid #d4603a;
    border-radius: 0 16px 16px 0;
    padding: 16px 20px;
    margin-bottom: 12px;
    color: #6b4f3f;
    font-size: 14px;
    line-height: 1.6;
}

/* ── Score ring ── */
.score-ring {
    font-family: 'Playfair Display', serif;
    font-size: 80px;
    font-weight: 900;
    line-height: 1;
}

/* ── Page wrapper ── */
.page { padding: 28px 36px; }

/* ── Expander ── */
.streamlit-expanderHeader {
    background: white !important;
    border: 1.5px solid #f0e6d8 !important;
    border-radius: 12px !important;
    color: #6b4f3f !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Selectbox ── */
[data-baseweb="select"] > div {
    background: #fffaf6 !important;
    border: 1.5px solid #e8d5c0 !important;
    border-radius: 12px !important;
    color: #3d2b1f !important;
}

/* ── Slider ── */
[data-testid="stSlider"] [data-baseweb="slider"] div { background: #d4603a !important; }

/* ── Checkbox ── */
[data-testid="stCheckbox"] label { color: #6b4f3f !important; text-transform: none !important; font-size: 14px !important; letter-spacing: 0 !important; }

/* ── Multiselect ── */
[data-baseweb="tag"] { background: #fdf0ec !important; color: #d4603a !important; }

/* ── Progress bar ── */
.stProgress > div > div { background: linear-gradient(90deg, #d4603a, #f5a623) !important; border-radius: 50px !important; }
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
FEATURES  = meta["features"]
COLLEGES  = ["Vivekananda Institute of Professional Studies - Technical Campus (VIPS-TC)", "Other"]
BRANCHES  = ["AIDS","CSE","ECE","IT","ME","CE","EEE","Other"]
SECTIONS  = ["A","B","C"]

# ── SESSION STATE ──────────────────────────────────────────────────────────────
for k, v in [("counselor_logged_in",False),("student_logged_in",False),
             ("student_data",{}),("notifications",[])]:
    if k not in st.session_state: st.session_state[k] = v

# ── HELPERS ────────────────────────────────────────────────────────────────────
def compute_score(p):
    return int(min(100,max(0, p.get("High",0)*100 + p.get("Medium",0)*50 + p.get("Low",0)*10)))

def score_info(s):
    if s<=33:   return "Thriving",       "#48b87a", "thriving",  "badge-thriving"
    elif s<=66: return "Needs Attention", "#f5a623", "attention", "badge-attention"
    else:       return "At Risk",         "#d4603a", "risk",      "badge-risk"

def trajectory(scores):
    if len(scores)<2: return "First check-in", "#b09070"
    diff = int(scores[-1]) - int(scores[-2])
    if diff>5:    return f"Score up {diff} pts — let's work on this", "#d4603a"
    elif diff<-5: return f"Score improved {abs(diff)} pts — great progress!", "#48b87a"
    else:         return "Stable since last survey", "#f5a623"

def predict(vals):
    arr=np.array([vals]); idx=model.predict(arr)[0]
    proba=model.predict_proba(arr)[0]; label=le.inverse_transform([idx])[0]
    return label, dict(zip(le.classes_,proba))

def push_notif(name,roll,risk,score,flagged=False):
    if risk not in ("High","Medium"): return
    st.session_state.notifications.insert(0,{
        "id":len(st.session_state.notifications),
        "ts":datetime.now().strftime("%d %b, %I:%M %p"),
        "name":name,"roll":roll,"risk":risk,"score":score,
        "flagged":flagged,"read":False,
        "expires":(datetime.now()+timedelta(days=30)).isoformat(),
    })

def personalized_advice(feat_dict, label):
    tips = []
    if float(feat_dict.get("sleep_hours",7))<6:
        tips.append(("Sleep First", "You're running on under 6 hours. Even one extra hour changes everything — try a consistent bedtime for just 2 weeks."))
    if float(feat_dict.get("social_media_hrs",3))>5:
        tips.append(("Digital Reset", "5+ hours of scrolling daily is quietly draining you. Try a 2-hour daily limit using your phone's screen time — you'll feel it within days."))
    if float(feat_dict.get("fomo_score",5))>7:
        tips.append(("Let Go of FOMO", "Constantly feeling like you're missing out is exhausting. Remind yourself: you're on your own timeline, and that's okay."))
    if float(feat_dict.get("exercise_days",3))<2:
        tips.append(("Move Your Body", "Even a 20-minute walk 3 times a week releases enough endorphins to noticeably reduce academic stress."))
    if float(feat_dict.get("confidence",5))<4:
        tips.append(("Rebuild Confidence", "Low self-belief amplifies every other stressor. Write 3 small wins each evening — this genuinely rewires how your brain sees itself."))
    if float(feat_dict.get("support_system",5))<4:
        tips.append(("Reach Out", "You seem to be carrying this alone. One honest conversation with someone you trust can shift everything."))
    if float(feat_dict.get("backlogs",0))>2:
        tips.append(("Clear the Backlog", "Multiple pending subjects create a constant background anxiety. Talk to your academic advisor about a realistic plan this week."))
    if not tips:
        if label=="Thriving":
            tips.append(("Keep It Going", "All your indicators look healthy. Keep your current habits — you're doing better than you think."))
        else:
            tips.append(("Small Steps", "Multiple areas need attention. Start with just one thing this week — better sleep is always the highest-impact first step."))
    return tips[:3]

# ── NAVBAR ─────────────────────────────────────────────────────────────────────
unread = sum(1 for n in st.session_state.notifications if not n["read"])
student_first = st.session_state.student_data.get("name","").split()[0] if st.session_state.student_logged_in else ""

alert_html = ""
if unread > 0:
    alert_html = "<span style='background:#d4603a;color:white;font-size:11px;font-weight:700;padding:4px 10px;border-radius:50px;margin-left:8px;animation:pulse-soft 2s infinite;'>" + str(unread) + " Alert</span>"

user_html = ""
if st.session_state.student_logged_in:
    user_html = "<span style='color:#48b87a;font-size:13px;font-weight:600;background:#e8f7ef;padding:6px 14px;border-radius:50px;border:1.5px solid #a8dfc0;'>● " + student_first + "</span>"
else:
    user_html = "<span style='color:#b09070;font-size:13px;'>Not signed in</span>"

navbar = "<div style='background:white;padding:16px 36px;display:flex;align-items:center;"
navbar += "justify-content:space-between;border-bottom:1px solid #f0e6d8;"
navbar += "box-shadow:0 2px 16px rgba(180,120,60,0.08);position:sticky;top:0;z-index:999;'>"
navbar += "<div style='display:flex;align-items:center;gap:14px;'>"
navbar += "<div style='width:38px;height:38px;background:linear-gradient(135deg,#d4603a,#f5a623);"
navbar += "border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:18px;"
navbar += "box-shadow:0 4px 12px rgba(212,96,58,0.3);'>🌿</div>"
navbar += "<div><div style='color:#2d1a0e;font-family:Playfair Display,serif;font-weight:700;"
navbar += "font-size:20px;line-height:1;'>Burnout Buster</div>"
navbar += "<div style='color:#b09070;font-size:11px;letter-spacing:1px;font-weight:500;margin-top:1px;'>VIPS-TC Wellness</div></div></div>"
navbar += "<div style='display:flex;align-items:center;gap:16px;'>"
navbar += alert_html + user_html
navbar += "<span style='color:#e8d5c0;font-size:20px;'>|</span>"
navbar += "<span style='color:#b09070;font-size:13px;font-weight:500;'>Mohit Kumar</span>"
navbar += "</div></div>"
st.markdown(navbar, unsafe_allow_html=True)

# ── TABS ───────────────────────────────────────────────────────────────────────
tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs(["Home","Take Survey","My Portal","Counselor","Analytics","Dataset"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — HOME
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="page">', unsafe_allow_html=True)

    # Hero — magazine editorial style
    st.markdown("""
    <div class="anim-up" style="background:linear-gradient(135deg,#2d1a0e 0%,#4a2c18 60%,#6b3d20 100%);
    border-radius:24px;padding:56px 48px;margin-bottom:28px;position:relative;overflow:hidden;">
      <div style="position:absolute;top:-40px;right:-40px;width:280px;height:280px;
      background:radial-gradient(circle,rgba(212,96,58,0.25),transparent 70%);border-radius:50%;"></div>
      <div style="position:absolute;bottom:-60px;left:30%;width:200px;height:200px;
      background:radial-gradient(circle,rgba(245,166,35,0.15),transparent 70%);border-radius:50%;"></div>
      <div style="position:relative;z-index:1;max-width:600px;">
        <div style="display:inline-block;background:rgba(212,96,58,0.25);border:1px solid rgba(212,96,58,0.4);
        border-radius:50px;padding:6px 18px;margin-bottom:20px;">
          <span style="color:#f5a623;font-size:12px;font-weight:700;letter-spacing:2px;">VIPS-TC WELLNESS INITIATIVE</span>
        </div>
        <div style="color:white;font-family:Playfair Display,serif;font-size:46px;font-weight:900;
        line-height:1.15;margin-bottom:18px;">
          Catch burnout<br><em style="color:#f5a623;">before</em> it catches you.
        </div>
        <div style="color:#c8a888;font-size:16px;line-height:1.8;margin-bottom:32px;max-width:480px;">
          A 3-minute science-backed assessment that gives you a personal wellness score
          and connects you with support — quietly, confidentially, before things get hard.
        </div>
        <div style="display:flex;gap:12px;flex-wrap:wrap;">
          <div style="background:rgba(255,255,255,0.1);backdrop-filter:blur(10px);border:1px solid rgba(255,255,255,0.15);
          border-radius:50px;padding:10px 20px;color:white;font-size:13px;font-weight:600;">
            ML-Powered Prediction
          </div>
          <div style="background:rgba(72,184,122,0.2);border:1px solid rgba(72,184,122,0.4);
          border-radius:50px;padding:10px 20px;color:#7ddaa8;font-size:13px;font-weight:600;">
            100% Confidential
          </div>
          <div style="background:rgba(245,166,35,0.2);border:1px solid rgba(245,166,35,0.4);
          border-radius:50px;padding:10px 20px;color:#f5c842;font-size:13px;font-weight:600;">
            Early Support
          </div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    # Three feature cards — magazine layout
    c1,c2,c3 = st.columns([1.2,1,1])
    with c1:
        st.markdown("""
        <div class="mag-card mag-card-coral anim-left d1">
          <div style="font-size:32px;margin-bottom:14px;">📋</div>
          <div style="font-family:Playfair Display,serif;font-size:20px;font-weight:700;
          color:#2d1a0e;margin-bottom:10px;">Take the Survey</div>
          <div style="color:#8a6a55;font-size:14px;line-height:1.7;">
            17 questions covering academics, social pressures, lifestyle, and emotional wellbeing.
            Takes about 3 minutes. Completely honest answers give the most accurate results.
          </div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="mag-card mag-card-amber anim-up d2">
          <div style="font-size:32px;margin-bottom:14px;">📈</div>
          <div style="font-family:Playfair Display,serif;font-size:20px;font-weight:700;
          color:#2d1a0e;margin-bottom:10px;">Track Your Journey</div>
          <div style="color:#8a6a55;font-size:14px;line-height:1.7;">
            Your personal portal shows your wellness score history, trend over time,
            and private messages from your counselor.
          </div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="mag-card mag-card-green anim-right d3">
          <div style="font-size:32px;margin-bottom:14px;">🤝</div>
          <div style="font-family:Playfair Display,serif;font-size:20px;font-weight:700;
          color:#2d1a0e;margin-bottom:10px;">Get Real Support</div>
          <div style="color:#8a6a55;font-size:14px;line-height:1.7;">
            Your counselor sees your results and reaches out if needed.
            No judgment. Just support, before things get overwhelming.
          </div>
        </div>""", unsafe_allow_html=True)

    # Score guide — editorial style
    st.markdown("""
    <div class="sec-divider anim-up d2">
      <div class="sec-divider-line"></div>
      <div class="sec-divider-text">Understanding Your Score</div>
      <div class="sec-divider-line" style="background:linear-gradient(90deg,transparent,#f0dcc8);"></div>
    </div>""", unsafe_allow_html=True)

    g1,g2,g3 = st.columns(3)
    guide = [
        ("0 — 33","Thriving","You're managing well. Your habits are supporting your mental health.","#48b87a","#e8f7ef","#a8dfc0"),
        ("34 — 66","Needs Attention","Some stress signals showing. A few small changes can make a real difference.","#f5a623","#fff8e8","#f5d890"),
        ("67 — 100","At Risk","You're under significant strain. Please reach out — support is here for you.","#d4603a","#fdf0ec","#f0b8a8"),
    ]
    for col,(rng,lbl,desc,color,bg,brd) in zip([g1,g2,g3],guide):
        with col:
            st.markdown(f"""
            <div class="anim-up d{guide.index((rng,lbl,desc,color,bg,brd))+1}"
            style="background:{bg};border:1.5px solid {brd};border-radius:20px;
            padding:24px;text-align:center;">
              <div style="font-family:Playfair Display,serif;font-size:32px;
              font-weight:900;color:{color};margin-bottom:6px;">{rng}</div>
              <div style="color:#2d1a0e;font-family:Playfair Display,serif;
              font-size:16px;font-weight:700;margin-bottom:10px;">{lbl}</div>
              <div style="color:#8a6a55;font-size:13px;line-height:1.6;">{desc}</div>
            </div>""", unsafe_allow_html=True)

    # Stats strip
    st.markdown("""
    <div class="sec-divider anim-up" style="margin-top:32px;">
      <div class="sec-divider-line"></div>
      <div class="sec-divider-text">By the Numbers</div>
      <div class="sec-divider-line" style="background:linear-gradient(90deg,transparent,#f0dcc8);"></div>
    </div>""", unsafe_allow_html=True)

    s1,s2,s3,s4 = st.columns(4)
    stats = [("60-70%","of students report burnout symptoms annually"),
             ("25%","dropout rate linked to unaddressed burnout"),
             ("2 weeks","earlier detection with our ML model"),
             ("17","factors tracked across 4 life areas")]
    for col,(num,desc) in zip([s1,s2,s3,s4],stats):
        with col:
            st.markdown(f"""
            <div class="stat-pill anim-up">
              <div style="font-family:Playfair Display,serif;font-size:32px;
              font-weight:900;color:#d4603a;">{num}</div>
              <div style="color:#8a6a55;font-size:12px;line-height:1.5;margin-top:6px;">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SURVEY
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    from database import student_exists, register_student, save_submission, get_student_submissions

    st.markdown('<div class="page">', unsafe_allow_html=True)
    st.markdown("""
    <div class="anim-up">
      <div style="font-family:Playfair Display,serif;font-size:36px;font-weight:900;
      color:#2d1a0e;margin-bottom:8px;">Student Wellness Assessment</div>
      <div style="color:#8a6a55;font-size:15px;margin-bottom:8px;">
        Your responses are <strong>completely confidential</strong>.
        Honest answers give the most accurate results and help us support you better.
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="sec-divider"><div class="sec-divider-line"></div>
    <div class="sec-divider-text">About You</div>
    <div class="sec-divider-line" style="background:linear-gradient(90deg,transparent,#f0dcc8);"></div>
    </div>""", unsafe_allow_html=True)

    i1,i2 = st.columns(2)
    with i1:
        s_name  = st.text_input("Full Name *",   placeholder="e.g. Ravi Sharma",   key="s_name")
        s_roll  = st.text_input("Roll Number *", placeholder="e.g. 01217711924",   key="s_roll")
        s_email = st.text_input("College Email", placeholder="e.g. ravi@vips.edu", key="s_email")
        s_age   = st.number_input("Age", min_value=16, max_value=30, value=20,      key="s_age")
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

    st.markdown("""
    <div class="sec-divider"><div class="sec-divider-line"></div>
    <div class="sec-divider-text">Academic Life</div>
    <div class="sec-divider-line" style="background:linear-gradient(90deg,transparent,#f0dcc8);"></div>
    </div>""", unsafe_allow_html=True)
    a1,a2,a3 = st.columns(3)
    with a1:
        q_exams  = st.select_slider("Exams per month",      options=list(range(1,9)),  value=4, key="q1")
        q_assign = st.select_slider("Assignments per week", options=list(range(1,13)), value=5, key="q2")
    with a2:
        q_attend = st.slider("Attendance pressure (1-10)", 1, 10, 6, key="q3")
        q_cgpa   = st.slider("Current CGPA", 4.0, 10.0, 7.0, step=0.1, key="q4")
    with a3:
        q_backlog= st.select_slider("Active backlogs",     options=list(range(0,9)),  value=0, key="q5")
        q_study  = st.select_slider("Study hours/day",     options=list(range(1,13)), value=5, key="q6")

    st.markdown("""
    <div class="sec-divider"><div class="sec-divider-line"></div>
    <div class="sec-divider-text">Social & Mental</div>
    <div class="sec-divider-line" style="background:linear-gradient(90deg,transparent,#f0dcc8);"></div>
    </div>""", unsafe_allow_html=True)
    b1,b2,b3 = st.columns(3)
    with b1:
        q_fomo   = st.slider("FOMO level (1-10)",           1, 10, 5, key="q7")
        q_peer   = st.slider("Peer pressure (1-10)",        1, 10, 5, key="q8")
    with b2:
        q_family = st.slider("Family expectations (1-10)",  1, 10, 6, key="q9")
        q_social = st.select_slider("Social media hrs/day", options=list(range(0,13)), value=3, key="q10")
    with b3:
        q_reject = st.slider("Rejection sensitivity (1-10)",1, 10, 5, key="q11")
        q_mhv    = st.select_slider("Counselor visits/month",options=list(range(0,6)), value=0, key="q12")

    st.markdown("""
    <div class="sec-divider"><div class="sec-divider-line"></div>
    <div class="sec-divider-text">Lifestyle</div>
    <div class="sec-divider-line" style="background:linear-gradient(90deg,transparent,#f0dcc8);"></div>
    </div>""", unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    with c1:
        q_sleep  = st.select_slider("Sleep hours/night",   options=list(range(3,11)), value=6, key="q13")
        q_exer   = st.select_slider("Exercise days/week",  options=list(range(0,8)),  value=2, key="q14")
    with c2:
        q_diet   = st.slider("Diet quality (1-10)",        1, 10, 5, key="q15")
        q_conf   = st.slider("Self-confidence (1-10)",     1, 10, 5, key="q16")
    with c3:
        q_support= st.slider("Support from friends/family",1, 10, 5, key="q17")

    st.markdown("""
    <div class="sec-divider"><div class="sec-divider-line"></div>
    <div class="sec-divider-text">Anything Else?</div>
    <div class="sec-divider-line" style="background:linear-gradient(90deg,transparent,#f0dcc8);"></div>
    </div>""", unsafe_allow_html=True)
    s_note = st.text_area("Is there anything you'd like your counselor to know? (optional)",
                          placeholder="e.g. I've been feeling anxious lately, or I have a personal situation I'm dealing with...",
                          height=90, key="s_note")
    consent = st.checkbox("I understand this is a wellness screening tool, not a clinical diagnosis. I consent to my counselor reviewing my responses.")
    st.markdown("")

    if st.button("Submit My Assessment", use_container_width=True, key="submit_survey"):
        errors = []
        if not s_name or not s_name.strip(): errors.append("Please enter your name")
        if not s_roll or not s_roll.strip(): errors.append("Please enter your roll number")
        if not consent: errors.append("Please check the consent box")
        if is_new:
            if not s_pwd: errors.append("Please create a password")
            elif s_pwd != s_pwd2: errors.append("Passwords do not match")
        if errors:
            for e in errors: st.error(e)
        else:
            feat_vals = [q_exams,q_assign,q_attend,q_cgpa,q_backlog,q_study,
                         q_fomo,q_peer,q_family,q_social,q_reject,
                         q_sleep,q_exer,q_diet,q_conf,q_support,q_mhv]
            feat_dict = dict(zip(FEATURES,feat_vals))
            risk,proba = predict(feat_vals)
            score = compute_score(proba)
            slabel,color,risk_css,badge_css = score_info(score)

            if is_new and s_pwd:
                with st.spinner("Setting up your profile..."):
                    register_student(s_roll.strip(),s_name.strip(),s_email.strip(),
                                     s_college,s_branch,s_section,s_age,s_pwd)

            prev = get_student_submissions(s_roll.strip())
            flagged = False
            if not prev.empty and "burnout_risk" in prev.columns and len(prev)>=1:
                if risk=="High" and prev.iloc[-1].get("burnout_risk","")=="High":
                    flagged = True

            with st.spinner("Saving your assessment..."):
                save_submission(s_roll.strip(),s_name.strip(),s_branch,s_section,
                                feat_dict,score,risk,proba,s_note.strip() if s_note else "")
            push_notif(s_name.strip(),s_roll.strip(),risk,score,flagged)

            st.markdown("---")
            st.markdown(f"""
            <div class="anim-up" style="font-family:Playfair Display,serif;font-size:26px;
            font-weight:700;color:#2d1a0e;margin-bottom:20px;">
              Your results, {s_name.strip().split()[0]}
            </div>""", unsafe_allow_html=True)

            _,rc,_ = st.columns([1,2,1])
            with rc:
                bg_map = {"thriving":"linear-gradient(135deg,#e8f7ef,#d4f0e2)",
                          "attention":"linear-gradient(135deg,#fff8e8,#fef0cc)",
                          "risk":"linear-gradient(135deg,#fdf0ec,#fde0d8)"}
                st.markdown(f"""
                <div class="anim-up mag-card" style="text-align:center;
                background:{bg_map[risk_css]};border-top:5px solid {color};">
                  <div class="score-ring" style="color:{color};">{score}</div>
                  <div style="color:#8a6a55;font-size:14px;margin:-4px 0 16px;">out of 100</div>
                  <span class="{badge_css}" style="font-size:14px;padding:6px 20px;">{slabel}</span>
                  <div style="color:#6b4f3f;font-size:14px;margin-top:16px;line-height:1.6;">
                    {"You're managing really well — keep it up!" if slabel=='Thriving'
                     else "Some areas need a bit of attention. Small changes make a big difference." if slabel=='Needs Attention'
                     else "You're going through a tough time. Please reach out — you don't have to do this alone."}
                  </div>
                </div>""", unsafe_allow_html=True)

            pc1,pc2,pc3 = st.columns(3)
            clrs = {"High":"#d4603a","Low":"#48b87a","Medium":"#f5a623"}
            lbls = {"High":"At Risk","Low":"Thriving","Medium":"Needs Attention"}
            for col,k in zip([pc1,pc2,pc3],le.classes_):
                with col:
                    st.markdown(f"""
                    <div class="stat-pill anim-up">
                      <div style="color:{clrs[k]};font-family:Playfair Display,serif;
                      font-size:28px;font-weight:700;">{proba[k]*100:.0f}%</div>
                      <div class="stat-lbl">{lbls[k]}</div>
                    </div>""", unsafe_allow_html=True)

            # Personalized tips
            st.markdown("""
            <div class="sec-divider anim-up"><div class="sec-divider-line"></div>
            <div class="sec-divider-text">Your Personal Action Plan</div>
            <div class="sec-divider-line" style="background:linear-gradient(90deg,transparent,#f0dcc8);"></div>
            </div>""", unsafe_allow_html=True)

            tips = personalized_advice(feat_dict, slabel)
            tip_cols = st.columns(len(tips))
            tip_bgs = ["#fdf0ec","#fff8e8","#e8f7ef"]
            tip_clrs= ["#d4603a","#b07820","#2a7d4f"]
            tip_brds= ["#f0b8a8","#f5d890","#a8dfc0"]
            for i,(col,(title,desc)) in enumerate(zip(tip_cols,tips)):
                with col:
                    st.markdown(f"""
                    <div class="anim-up d{i+1}" style="background:{tip_bgs[i]};
                    border:1.5px solid {tip_brds[i]};border-radius:20px;padding:22px;">
                      <div style="color:{tip_clrs[i]};font-family:DM Sans,sans-serif;
                      font-size:12px;font-weight:700;letter-spacing:1.5px;
                      text-transform:uppercase;margin-bottom:10px;">{title}</div>
                      <div style="color:#6b4f3f;font-size:13px;line-height:1.7;">{desc}</div>
                    </div>""", unsafe_allow_html=True)

            if slabel == "At Risk":
                st.markdown("""
                <div class="anim-up" style="background:#fdf0ec;border:1.5px solid #f0b8a8;
                border-radius:16px;padding:18px 24px;margin-top:16px;">
                  <div style="color:#c04020;font-weight:700;margin-bottom:6px;font-size:15px;">
                    Free & Confidential Helplines
                  </div>
                  <div style="color:#8a6a55;font-size:13px;">
                    iCall: 9152987821 &nbsp;·&nbsp;
                    Vandrevala Foundation: 1860-2662-345 (24/7) &nbsp;·&nbsp;
                    NIMHANS: 080-46110007
                  </div>
                </div>""", unsafe_allow_html=True)

            if flagged:
                st.error("This is your second consecutive At Risk result. Your counselor has been specially alerted and will reach out soon.")

    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — STUDENT PORTAL
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    from database import verify_student, get_student_submissions, get_replies, mark_replies_read, get_reminder

    st.markdown('<div class="page">', unsafe_allow_html=True)

    if not st.session_state.student_logged_in:
        st.markdown("""
        <div class="anim-up">
          <div style="font-family:Playfair Display,serif;font-size:36px;font-weight:900;
          color:#2d1a0e;margin-bottom:8px;">My Wellness Portal</div>
          <div style="color:#8a6a55;font-size:15px;margin-bottom:24px;">
            Log in to see your score history, progress, and messages from your counselor.
          </div>
        </div>""", unsafe_allow_html=True)
        _,lc,_ = st.columns([1,1.2,1])
        with lc:
            st.markdown('<div class="mag-card">', unsafe_allow_html=True)
            p_roll    = st.text_input("Roll Number",  placeholder="e.g. 01217711924", key="p_roll")
            p_branch  = st.selectbox("Branch",  BRANCHES, key="p_branch")
            p_section = st.selectbox("Section", SECTIONS, key="p_section")
            p_pwd     = st.text_input("Password", type="password", key="p_pwd")
            if st.button("Sign In to My Portal", use_container_width=True, key="portal_login"):
                if p_roll and p_pwd:
                    student = verify_student(p_roll.strip(), p_pwd.strip(), p_branch, p_section)
                    if student:
                        st.session_state.student_logged_in = True
                        st.session_state.student_data = student
                        st.rerun()
                    else:
                        st.error("Incorrect credentials. Make sure roll number, branch, section, and password all match what you used during the survey.")
                else:
                    st.warning("Please fill in all fields.")
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        sd   = st.session_state.student_data
        name = sd.get("name","Student")
        roll = str(sd.get("roll_number",""))

        hc1,hc2 = st.columns([4,1])
        with hc1:
            st.markdown(f"""
            <div class="anim-up">
              <div style="font-family:Playfair Display,serif;font-size:36px;
              font-weight:900;color:#2d1a0e;">Hey, {name.split()[0]}.</div>
              <div style="color:#8a6a55;font-size:15px;">Here's your wellness overview.</div>
            </div>""", unsafe_allow_html=True)
        with hc2:
            if st.button("Sign Out", key="s_out"):
                st.session_state.student_logged_in = False
                st.session_state.student_data = {}
                st.rerun()

        try:
            reminder = get_reminder()
            if reminder and reminder.get("next_due"):
                next_due  = datetime.strptime(str(reminder["next_due"]),"%Y-%m-%d %H:%M:%S")
                days_left = (next_due - datetime.now()).days
                if days_left <= 3:
                    st.warning(f"Your next wellness check-in is due {'today' if days_left<=0 else f'in {days_left} day(s)'}. Head to Take Survey.")
        except Exception:
            pass

        subs = get_student_submissions(roll)

        if subs.empty:
            st.markdown("""
            <div class="mag-card mag-card-cream" style="text-align:center;padding:48px;">
              <div style="font-size:40px;margin-bottom:16px;">📋</div>
              <div style="font-family:Playfair Display,serif;font-size:20px;
              color:#2d1a0e;margin-bottom:8px;">No assessments yet</div>
              <div style="color:#8a6a55;font-size:14px;">
                Head to <strong>Take Survey</strong> to get your first wellness score.
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            scores = pd.to_numeric(subs.get("burnout_score",pd.Series()),errors="coerce").fillna(0).astype(int).tolist() if "burnout_score" in subs.columns else []
            latest = scores[-1] if scores else 0
            slabel,color,risk_css,badge_css = score_info(latest)
            tmsg,tcolor = trajectory(scores)

            s1,s2,s3,s4 = st.columns(4)
            with s1:
                st.markdown(f"""
                <div class="stat-pill anim-up" style="border-top:4px solid {color};">
                  <div class="score-ring" style="color:{color};font-size:52px;">{latest}</div>
                  <div class="stat-lbl">Latest Score</div>
                </div>""", unsafe_allow_html=True)
            with s2:
                st.markdown(f"""
                <div class="stat-pill anim-up d1">
                  <div style="font-family:Playfair Display,serif;font-size:36px;
                  font-weight:700;color:{tcolor};">{'↑' if 'up' in tmsg else '↓' if 'improved' in tmsg else '~'}</div>
                  <div class="stat-lbl">Trend</div>
                </div>""", unsafe_allow_html=True)
            with s3:
                st.markdown(f"""
                <div class="stat-pill anim-up d2">
                  <div class="score-ring" style="font-size:52px;">{len(subs)}</div>
                  <div class="stat-lbl">Surveys Done</div>
                </div>""", unsafe_allow_html=True)
            with s4:
                st.markdown(f"""
                <div class="stat-pill anim-up d3" style="padding-top:28px;">
                  <span class="{badge_css}" style="font-size:13px;padding:6px 18px;">{slabel}</span>
                  <div class="stat-lbl" style="margin-top:12px;">Current Status</div>
                </div>""", unsafe_allow_html=True)

            st.markdown(f'<div style="color:{tcolor};font-size:13px;margin:10px 0 20px;">{tmsg}</div>', unsafe_allow_html=True)

            if len(scores)>1:
                st.markdown("""
                <div class="sec-divider"><div class="sec-divider-line"></div>
                <div class="sec-divider-text">Score History</div>
                <div class="sec-divider-line" style="background:linear-gradient(90deg,transparent,#f0dcc8);"></div>
                </div>""", unsafe_allow_html=True)
                fig,ax = plt.subplots(figsize=(10,3.5))
                fig.patch.set_facecolor("#fffaf6")
                ax.set_facecolor("#fffaf6")
                xs = list(range(1,len(scores)+1))
                ax.fill_between(xs,scores,alpha=0.12,color="#d4603a")
                ax.plot(xs,scores,color="#d4603a",linewidth=2.5,marker="o",
                        markersize=9,markerfacecolor="white",
                        markeredgewidth=2.5,markeredgecolor="#d4603a")
                ax.axhline(33,color="#48b87a",linewidth=1,linestyle="--",alpha=0.5,label="Thriving")
                ax.axhline(66,color="#f5a623",linewidth=1,linestyle="--",alpha=0.5,label="Needs Attention")
                for i,s in enumerate(scores):
                    ax.annotate(str(s),(xs[i],s),textcoords="offset points",
                                xytext=(0,12),ha="center",fontsize=10,
                                color="#2d1a0e",fontweight="bold")
                ax.set_xlabel("Survey #",color="#b09070",fontsize=11)
                ax.set_ylabel("Score",color="#b09070",fontsize=11)
                ax.set_title("Your Wellness Journey",color="#2d1a0e",
                             fontweight="bold",fontfamily="serif")
                ax.tick_params(colors="#b09070")
                for spine in ax.spines.values(): spine.set_color("#f0e6d8")
                ax.set_ylim(0,115)
                ax.legend(fontsize=9,facecolor="#fffaf6",
                          edgecolor="#f0e6d8",labelcolor="#8a6a55")
                plt.tight_layout()
                st.pyplot(fig)

            with st.expander("View all submissions"):
                show_cols = [c for c in ["timestamp","burnout_score","burnout_risk"] if c in subs.columns]
                st.dataframe(subs[show_cols],use_container_width=True)

        # Messages
        st.markdown("""
        <div class="sec-divider"><div class="sec-divider-line"></div>
        <div class="sec-divider-text">Messages from Your Counselor</div>
        <div class="sec-divider-line" style="background:linear-gradient(90deg,transparent,#f0dcc8);"></div>
        </div>""", unsafe_allow_html=True)

        replies = get_replies(roll)
        if replies.empty:
            st.markdown("""
            <div class="mag-card mag-card-cream" style="text-align:center;padding:28px;color:#8a6a55;">
              No messages yet. Your counselor will reach out after reviewing your assessment.
            </div>""", unsafe_allow_html=True)
        else:
            mark_replies_read(roll)
            for _,rep in replies.iterrows():
                ts  = rep.get("timestamp","")
                msg = rep.get("counselor_message","")
                st.markdown("<div class='reply-card'>"
                            "<div style='color:#b09070;font-size:11px;font-weight:700;"
                            "letter-spacing:1px;text-transform:uppercase;margin-bottom:8px;'>"
                            "Your Counselor &nbsp;·&nbsp; " + str(ts) + "</div>"
                            + str(msg) + "</div>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — COUNSELOR
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    from database import (get_all_submissions, get_all_students, upsert_counselor_action,
                          get_counselor_action, save_reply, save_reminder,
                          save_college_records, get_college_records)

    st.markdown('<div class="page">', unsafe_allow_html=True)

    if not st.session_state.counselor_logged_in:
        st.markdown("""
        <div class="anim-up">
          <div style="font-family:Playfair Display,serif;font-size:36px;font-weight:900;
          color:#2d1a0e;margin-bottom:8px;">Counselor Login</div>
          <div style="color:#8a6a55;font-size:15px;">Restricted to authorised counselors only.</div>
        </div>""", unsafe_allow_html=True)
        _,lc,_ = st.columns([1,1.2,1])
        with lc:
            st.markdown('<div class="mag-card">', unsafe_allow_html=True)
            pwd = st.text_input("Password", type="password",
                                placeholder="Enter counselor password", key="c_pwd")
            if st.button("Sign In", use_container_width=True, key="c_login"):
                COUNSELOR_PASSWORD = "ProjectAlpha01"
                try: COUNSELOR_PASSWORD = st.secrets["COUNSELOR_PASSWORD"]
                except Exception: pass
                if pwd == COUNSELOR_PASSWORD:
                    st.session_state.counselor_logged_in = True
                    st.rerun()
                else:
                    st.error("Incorrect password.")
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        hc1,hc2 = st.columns([4,1])
        with hc1:
            st.markdown("""
            <div class="anim-up">
              <div style="font-family:Playfair Display,serif;font-size:36px;
              font-weight:900;color:#2d1a0e;">Counselor Dashboard</div>
            </div>""", unsafe_allow_html=True)
        with hc2:
            if st.button("Sign Out", key="c_out"):
                st.session_state.counselor_logged_in = False
                st.rerun()

        # Alerts
        notifs = [n for n in st.session_state.notifications
                  if datetime.fromisoformat(n["expires"]) > datetime.now()]
        st.session_state.notifications = notifs
        unread_c = sum(1 for n in notifs if not n["read"])

        with st.expander(f"Alerts  {'· ' + str(unread_c) + ' unread' if unread_c else '· all read'}", expanded=unread_c>0):
            if not notifs:
                st.markdown('<div style="color:#b09070;padding:12px;">No alerts yet.</div>', unsafe_allow_html=True)
            else:
                if st.button("Mark all read", key="mark_all"):
                    for n in st.session_state.notifications: n["read"] = True
                    st.rerun()
                for n in notifs:
                    color_n = "#d4603a" if n["risk"]=="High" else "#f5a623"
                    bg_n    = "#fdf0ec" if n["risk"]=="High" else "#fff8e8"
                    brd_n   = "#f0b8a8" if n["risk"]=="High" else "#f5d890"
                    new_b   = "<span style='background:#d4603a;color:white;font-size:10px;padding:2px 8px;border-radius:50px;margin-left:8px;'>NEW</span>" if not n["read"] else ""
                    flag_b  = "<span style='background:#fdf0ec;color:#c04020;font-size:10px;font-weight:700;padding:2px 8px;border-radius:50px;border:1px solid #f0b8a8;margin-left:6px;'>PERSISTENT</span>" if n.get("flagged") else ""
                    notif_html = "<div style='background:" + bg_n + ";border:1.5px solid " + brd_n + ";border-radius:14px;padding:14px 18px;margin-bottom:10px;opacity:" + ('1' if not n['read'] else '0.55') + ";'>"
                    notif_html += "<strong style='color:#2d1a0e;'>" + n['name'] + "</strong>"
                    notif_html += "<span style='color:#b09070;font-size:12px;margin-left:8px;'>· " + n['roll'] + "</span>"
                    notif_html += "<span style='background:" + color_n + ";color:white;font-size:11px;font-weight:700;padding:3px 10px;border-radius:50px;margin-left:8px;'>" + str(n['score']) + "/100 · " + n['risk'] + "</span>"
                    notif_html += new_b + flag_b
                    notif_html += "<span style='float:right;color:#b09070;font-size:11px;'>" + n['ts'] + "</span></div>"
                    st.markdown(notif_html, unsafe_allow_html=True)
                    if not n["read"]:
                        if st.button("Mark read", key="nr_"+str(n['id'])):
                            n["read"] = True
                            st.rerun()

        with st.expander("Survey Reminder Settings"):
            freq = st.select_slider("Reminder frequency",options=[7,14,21,30,60],value=30,
                                    format_func=lambda x: f"Every {x} days")
            if st.button("Save Schedule", key="save_rem"):
                save_reminder(freq)
                st.success(f"Reminder set — students prompted every {freq} days.")

        # College Records
        st.markdown("""
        <div class="sec-divider"><div class="sec-divider-line"></div>
        <div class="sec-divider-text">College Academic Records</div>
        <div class="sec-divider-line" style="background:linear-gradient(90deg,transparent,#f0dcc8);"></div>
        </div>""", unsafe_allow_html=True)

        with st.expander("Upload Records"):
            ur1,ur2 = st.columns(2)
            with ur1: up_branch  = st.selectbox("Branch",  BRANCHES, key="up_branch")
            with ur2: up_section = st.selectbox("Section", SECTIONS, key="up_section")
            st.markdown("Required columns: `roll_number, name, attendance_pct, marks_pct, participation, remarks`")
            tmpl = pd.DataFrame({"roll_number":["01217711924"],"name":["Student Name"],
                                  "attendance_pct":[85],"marks_pct":[78],
                                  "participation":["Active"],"remarks":["Good"]})
            st.download_button("Download Template",data=tmpl.to_csv(index=False).encode(),
                               file_name="template.csv",mime="text/csv")
            uploaded = st.file_uploader("Upload CSV",type=["csv"],key="rec_upload")
            if uploaded:
                try:
                    df_up = pd.read_csv(uploaded)
                    st.dataframe(df_up.head(),use_container_width=True)
                    if st.button("Confirm Upload",key="confirm_upload"):
                        with st.spinner("Saving..."):
                            save_college_records(df_up,up_branch,up_section)
                        st.success(f"Saved {len(df_up)} records for {up_branch}-{up_section}!")
                except Exception as ex:
                    st.error(f"Error: {ex}")

        # Student list
        st.markdown("""
        <div class="sec-divider"><div class="sec-divider-line"></div>
        <div class="sec-divider-text">Student Submissions</div>
        <div class="sec-divider-line" style="background:linear-gradient(90deg,transparent,#f0dcc8);"></div>
        </div>""", unsafe_allow_html=True)

        all_subs  = get_all_submissions()
        all_studs = get_all_students()

        if all_subs.empty:
            st.markdown('<div class="mag-card mag-card-cream" style="text-align:center;padding:32px;color:#8a6a55;">No student submissions yet.</div>', unsafe_allow_html=True)
        else:
            name_lookup = {}
            if not all_studs.empty and "roll_number" in all_studs.columns:
                for _,row in all_studs.iterrows():
                    name_lookup[str(row["roll_number"]).strip()] = str(row.get("name",""))
            if "student_name" in all_subs.columns:
                for _,row in all_subs.iterrows():
                    r=str(row.get("roll_number","")).strip(); n=str(row.get("student_name","")).strip()
                    if r and n and n!="nan" and r not in name_lookup: name_lookup[r]=n

            unique_rolls = all_subs["roll_number"].nunique() if "roll_number" in all_subs.columns else 0
            high   = int((all_subs["burnout_risk"]=="High").sum())   if "burnout_risk" in all_subs.columns else 0
            medium = int((all_subs["burnout_risk"]=="Medium").sum()) if "burnout_risk" in all_subs.columns else 0
            low    = int((all_subs["burnout_risk"]=="Low").sum())    if "burnout_risk" in all_subs.columns else 0

            m1,m2,m3,m4 = st.columns(4)
            with m1: st.markdown('<div class="stat-pill anim-up"><div class="stat-num">' + str(unique_rolls) + '</div><div class="stat-lbl">Students</div></div>', unsafe_allow_html=True)
            with m2: st.markdown('<div class="stat-pill anim-up d1" style="border-top:4px solid #d4603a;"><div class="stat-num" style="color:#d4603a;">' + str(high) + '</div><div class="stat-lbl">At Risk</div></div>', unsafe_allow_html=True)
            with m3: st.markdown('<div class="stat-pill anim-up d2" style="border-top:4px solid #f5a623;"><div class="stat-num" style="color:#f5a623;">' + str(medium) + '</div><div class="stat-lbl">Needs Attention</div></div>', unsafe_allow_html=True)
            with m4: st.markdown('<div class="stat-pill anim-up d3" style="border-top:4px solid #48b87a;"><div class="stat-num" style="color:#48b87a;">' + str(low) + '</div><div class="stat-lbl">Thriving</div></div>', unsafe_allow_html=True)

            st.markdown("")
            fc1,fc2,fc3,fc4 = st.columns(4)
            with fc1: rf = st.multiselect("Risk",["High","Medium","Low"],default=["High","Medium","Low"],key="rf")
            with fc2: fb = st.selectbox("Branch",["All"]+BRANCHES,key="fb")
            with fc3: search = st.text_input("Search name/roll","",key="srch")
            with fc4: show_flag = st.checkbox("Persistent At Risk only",key="sflag")

            if "roll_number" in all_subs.columns and "timestamp" in all_subs.columns:
                latest = all_subs.sort_values("timestamp").groupby("roll_number").last().reset_index()
            else:
                latest = all_subs.copy()

            if rf and "burnout_risk" in latest.columns:
                latest = latest[latest["burnout_risk"].isin(rf)]
            if fb!="All" and "branch" in latest.columns:
                latest = latest[latest["branch"]==fb]
            if search:
                rm = latest["roll_number"].astype(str).str.contains(search,na=False)
                nm = pd.Series([search.lower() in name_lookup.get(str(r),"").lower() for r in latest["roll_number"]],index=latest.index)
                latest = latest[rm|nm]

            st.markdown(f'<div style="color:#b09070;font-size:13px;margin-bottom:16px;">Showing {len(latest)} students</div>', unsafe_allow_html=True)

            for _,row in latest.iterrows():
                roll_d   = str(row.get("roll_number","")).strip()
                risk_d   = row.get("burnout_risk","Unknown")
                score_d  = int(row.get("burnout_score",0)) if str(row.get("burnout_score",0)).replace(".","").isdigit() else 0
                ts_d     = row.get("timestamp","")
                note_d   = str(row.get("student_note",""))
                branch_d = row.get("branch","")
                section_d= row.get("section","")
                sname_d  = name_lookup.get(roll_d, roll_d)

                stud_subs = all_subs[all_subs["roll_number"].astype(str).str.strip()==roll_d] if "roll_number" in all_subs.columns else pd.DataFrame()
                flagged_d = False
                if len(stud_subs)>=2 and "burnout_risk" in stud_subs.columns:
                    last2 = stud_subs.sort_values("timestamp").tail(2)["burnout_risk"].tolist()
                    flagged_d = all(r=="High" for r in last2)

                if show_flag and not flagged_d: continue

                action_d = get_counselor_action(roll_d)
                status_d = action_d.get("status","Pending")
                notes_d  = action_d.get("notes","")

                risk_css_d  = "risk" if risk_d=="High" else "attention" if risk_d=="Medium" else "thriving"
                badge_css_d = "badge-risk" if risk_d=="High" else "badge-attention" if risk_d=="Medium" else "badge-thriving"
                card_css_d  = "student-card-risk" if risk_d=="High" else "student-card-attention" if risk_d=="Medium" else "student-card-thriving"

                records = get_college_records(roll=roll_d)
                has_records = not records.empty
                consistency_flag = ""
                if has_records and "attendance_pct" in records.columns:
                    try:
                        att = float(records.iloc[0]["attendance_pct"])
                        attend_score = float(row.get("attendance_pressure",5))
                        if att>85 and attend_score>8:
                            consistency_flag = " <span style='background:#fff8e8;color:#b07820;font-size:10px;padding:2px 8px;border-radius:50px;border:1px solid #f5d890;'>Verify Response</span>"
                    except Exception: pass

                flag_html = ""
                if flagged_d:
                    flag_html = " <span style='background:#fdf0ec;color:#c04020;font-size:10px;font-weight:700;padding:2px 8px;border-radius:50px;border:1.5px solid #f0b8a8;'>Persistent</span>"

                left_part = "<strong style='color:#2d1a0e;font-size:15px;font-family:Playfair Display,serif;'>" + sname_d + "</strong>"
                left_part += "<span style='color:#b09070;font-size:12px;margin-left:10px;'>Roll: " + roll_d + "</span>"
                left_part += "<span style='color:#d4c0b0;font-size:12px;margin-left:8px;'>" + str(branch_d) + "-" + str(section_d) + "</span>"
                left_part += flag_html + consistency_flag

                right_part = "<span class='" + badge_css_d + "' style='font-size:12px;'>" + risk_d + "</span>"
                right_part += "<span style='color:#6b4f3f;font-size:14px;font-family:Playfair Display,serif;font-weight:700;margin-left:10px;'>" + str(score_d) + "/100</span>"
                right_part += "<div style='color:#b09070;font-size:11px;margin-top:4px;'>" + str(ts_d) + "</div>"

                note_part = ""
                if note_d and str(note_d).strip() and str(note_d).strip()!="nan":
                    note_part = "<div style='background:#fff8f0;border-left:3px solid #f5a623;border-radius:0 8px 8px 0;padding:8px 14px;margin-top:12px;color:#6b4f3f;font-size:13px;'><strong style='color:#b07820;'>Student note:</strong> " + str(note_d) + "</div>"

                card_html = "<div class='student-card " + card_css_d + "'>"
                card_html += "<div style='display:flex;justify-content:space-between;align-items:center;'>"
                card_html += "<div>" + left_part + "</div>"
                card_html += "<div style='text-align:right;'>" + right_part + "</div>"
                card_html += "</div>" + note_part + "</div>"
                st.markdown(card_html, unsafe_allow_html=True)

                with st.expander("Actions — " + sname_d + " (" + roll_d + ")"):
                    if has_records:
                        st.markdown("**Academic Records from College:**")
                        st.dataframe(records.drop(columns=["branch","section","uploaded_at"],errors="ignore"),use_container_width=True)
                        st.markdown("---")
                    ac1,ac2 = st.columns([1,2])
                    with ac1:
                        new_status = st.selectbox("Status",["Pending","Contacted","No Action Needed"],
                            index=["Pending","Contacted","No Action Needed"].index(status_d)
                            if status_d in ["Pending","Contacted","No Action Needed"] else 0,
                            key="st_"+roll_d)
                        new_notes = st.text_area("Notes",value=notes_d if str(notes_d)!="nan" else "",
                                                  height=80,key="nt_"+roll_d)
                        if st.button("Save Status",key="sv_"+roll_d):
                            upsert_counselor_action(roll_d,new_status,new_notes,flagged_d)
                            st.success("Status updated!")
                    with ac2:
                        st.markdown("**Send a Private Message**")
                        reply_msg = st.text_area("Your message",
                            placeholder="Hi, I reviewed your assessment and wanted to check in with you...",
                            height=100,key="rp_"+roll_d)
                        if st.button("Send Message",key="send_"+roll_d):
                            if reply_msg and reply_msg.strip():
                                save_reply(roll_d,reply_msg.strip())
                                st.success("Message sent!")
                            else:
                                st.warning("Please type a message first.")

            st.markdown("---")
            st.download_button("Export All Data (CSV)",
                               data=all_subs.to_csv(index=False).encode(),
                               file_name="burnout_submissions.csv",mime="text/csv")

    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    from database import get_all_submissions, get_all_students

    st.markdown('<div class="page">', unsafe_allow_html=True)
    st.markdown("""
    <div class="anim-up">
      <div style="font-family:Playfair Display,serif;font-size:36px;font-weight:900;
      color:#2d1a0e;margin-bottom:8px;">Analytics Dashboard</div>
      <div style="color:#8a6a55;font-size:15px;">Institution-wide wellness insights for counselors.</div>
    </div>""", unsafe_allow_html=True)

    if not st.session_state.counselor_logged_in:
        st.warning("Please sign in as counselor to view analytics.")
    else:
        all_subs  = get_all_submissions()
        all_studs = get_all_students()

        if all_subs.empty:
            st.markdown('<div class="mag-card mag-card-cream" style="text-align:center;padding:40px;color:#8a6a55;">No data yet. Analytics appear once students start submitting assessments.</div>', unsafe_allow_html=True)
        else:
            if not all_studs.empty and "roll_number" in all_studs.columns:
                merge_cols = [c for c in ["roll_number","branch","section","age"] if c in all_studs.columns]
                try:
                    all_subs = all_subs.merge(all_studs[merge_cols].astype(str),on="roll_number",how="left",suffixes=("","_s"))
                except Exception: pass

            latest = all_subs.sort_values("timestamp").groupby("roll_number").last().reset_index() if "roll_number" in all_subs.columns else all_subs.copy()
            if "burnout_score" in latest.columns:
                latest["burnout_score"] = pd.to_numeric(latest["burnout_score"],errors="coerce").fillna(0)

            total   = len(latest)
            avg_sc  = int(latest["burnout_score"].mean()) if "burnout_score" in latest.columns and total>0 else 0
            at_risk = int((latest["burnout_risk"]=="High").sum())  if "burnout_risk" in latest.columns else 0
            thriving= int((latest["burnout_risk"]=="Low").sum())   if "burnout_risk" in latest.columns else 0

            m1,m2,m3,m4 = st.columns(4)
            with m1: st.markdown('<div class="stat-pill anim-up"><div class="stat-num">' + str(total) + '</div><div class="stat-lbl">Total Students</div></div>', unsafe_allow_html=True)
            with m2: st.markdown('<div class="stat-pill anim-up d1"><div class="stat-num">' + str(avg_sc) + '/100</div><div class="stat-lbl">Avg Score</div></div>', unsafe_allow_html=True)
            with m3: st.markdown('<div class="stat-pill anim-up d2" style="border-top:4px solid #d4603a;"><div class="stat-num" style="color:#d4603a;">' + str(at_risk) + '</div><div class="stat-lbl">At Risk</div></div>', unsafe_allow_html=True)
            with m4: st.markdown('<div class="stat-pill anim-up d3" style="border-top:4px solid #48b87a;"><div class="stat-num" style="color:#48b87a;">' + str(thriving) + '</div><div class="stat-lbl">Thriving</div></div>', unsafe_allow_html=True)

            st.markdown("")

            def warm_fig(w=5,h=4):
                fig,ax = plt.subplots(figsize=(w,h))
                fig.patch.set_facecolor("#fffaf6")
                ax.set_facecolor("#fffaf6")
                ax.tick_params(colors="#b09070")
                for spine in ax.spines.values(): spine.set_color("#f0e6d8")
                return fig,ax

            col1,col2 = st.columns(2)

            with col1:
                st.markdown("""
                <div class="sec-divider"><div class="sec-divider-line"></div>
                <div class="sec-divider-text">Risk Distribution</div>
                <div class="sec-divider-line" style="background:linear-gradient(90deg,transparent,#f0dcc8);"></div>
                </div>""", unsafe_allow_html=True)
                if "burnout_risk" in latest.columns:
                    counts = latest["burnout_risk"].value_counts()
                    fig,ax = warm_fig()
                    clrs = [{"High":"#d4603a","Medium":"#f5a623","Low":"#48b87a"}.get(l,"#e8d5c0") for l in counts.index]
                    wedges,texts,autotexts = ax.pie(counts.values,labels=counts.index,
                        autopct="%1.0f%%",colors=clrs,startangle=90,
                        wedgeprops=dict(edgecolor="white",linewidth=3))
                    for t in texts: t.set_color("#6b4f3f"); t.set_fontsize(12)
                    for at in autotexts: at.set_color("white"); at.set_fontweight("bold"); at.set_fontsize(11)
                    ax.set_title("Students by Risk Level",color="#2d1a0e",fontweight="bold",fontfamily="serif")
                    plt.tight_layout(); st.pyplot(fig)

            with col2:
                st.markdown("""
                <div class="sec-divider"><div class="sec-divider-line"></div>
                <div class="sec-divider-text">Average Score by Branch</div>
                <div class="sec-divider-line" style="background:linear-gradient(90deg,transparent,#f0dcc8);"></div>
                </div>""", unsafe_allow_html=True)
                branch_col = "branch_s" if "branch_s" in latest.columns else "branch"
                if branch_col in latest.columns and "burnout_score" in latest.columns:
                    branch_avg = latest.groupby(branch_col)["burnout_score"].mean().sort_values(ascending=False)
                    fig,ax = warm_fig()
                    clrs = ["#d4603a" if v>66 else "#f5a623" if v>33 else "#48b87a" for v in branch_avg.values]
                    bars = ax.barh(branch_avg.index[::-1],branch_avg.values[::-1],
                                   color=clrs[::-1],height=0.55,edgecolor="none")
                    for bar,val in zip(bars,branch_avg.values[::-1]):
                        ax.text(bar.get_width()+0.5,bar.get_y()+bar.get_height()/2,
                                f"{val:.0f}",va="center",color="#6b4f3f",fontsize=10,fontweight="bold")
                    ax.set_xlabel("Average Score",color="#b09070",fontsize=11)
                    ax.set_title("Burnout by Branch",color="#2d1a0e",fontweight="bold",fontfamily="serif")
                    ax.set_xlim(0,115); plt.tight_layout(); st.pyplot(fig)
                else:
                    st.info("Branch data appears once students submit with branch info.")

            # Trend
            st.markdown("""
            <div class="sec-divider"><div class="sec-divider-line"></div>
            <div class="sec-divider-text">Burnout Trend Over Time</div>
            <div class="sec-divider-line" style="background:linear-gradient(90deg,transparent,#f0dcc8);"></div>
            </div>""", unsafe_allow_html=True)
            if "timestamp" in all_subs.columns and "burnout_score" in all_subs.columns:
                try:
                    all_subs["date"] = pd.to_datetime(all_subs["timestamp"]).dt.date
                    all_subs["burnout_score"] = pd.to_numeric(all_subs["burnout_score"],errors="coerce")
                    trend = all_subs.groupby("date")["burnout_score"].mean().reset_index()
                    if len(trend)>1:
                        fig,ax = warm_fig(10,3.5)
                        ax.fill_between(range(len(trend)),trend["burnout_score"],alpha=0.12,color="#d4603a")
                        ax.plot(range(len(trend)),trend["burnout_score"],color="#d4603a",
                                linewidth=2.5,marker="o",markersize=7,
                                markerfacecolor="white",markeredgewidth=2.5,markeredgecolor="#d4603a")
                        ax.axhline(33,color="#48b87a",linewidth=1,linestyle="--",alpha=0.5,label="Thriving")
                        ax.axhline(66,color="#f5a623",linewidth=1,linestyle="--",alpha=0.5,label="Needs Attention")
                        ax.set_xticks(range(len(trend)))
                        ax.set_xticklabels([str(d) for d in trend["date"]],rotation=30,ha="right",fontsize=9)
                        ax.set_ylabel("Avg Score",color="#b09070")
                        ax.set_title("Institution Wellness Trend",color="#2d1a0e",fontweight="bold",fontfamily="serif")
                        ax.legend(fontsize=9,facecolor="#fffaf6",edgecolor="#f0e6d8",labelcolor="#8a6a55")
                        plt.tight_layout(); st.pyplot(fig)
                    else:
                        st.info("Trend chart appears once there are submissions on multiple dates.")
                except Exception:
                    st.info("Trend data will appear as more submissions come in.")

            col3,col4 = st.columns(2)
            with col3:
                st.markdown("""
                <div class="sec-divider"><div class="sec-divider-line"></div>
                <div class="sec-divider-text">Section Comparison</div>
                <div class="sec-divider-line" style="background:linear-gradient(90deg,transparent,#f0dcc8);"></div>
                </div>""", unsafe_allow_html=True)
                sec_col = "section_s" if "section_s" in latest.columns else "section"
                if sec_col in latest.columns and "burnout_score" in latest.columns:
                    sec_avg = latest.groupby(sec_col)["burnout_score"].mean().sort_values(ascending=False)
                    fig,ax = warm_fig(5,3.5)
                    clrs = ["#d4603a" if v>66 else "#f5a623" if v>33 else "#48b87a" for v in sec_avg.values]
                    bars = ax.bar(sec_avg.index,sec_avg.values,color=clrs,width=0.45,edgecolor="none")
                    for bar,val in zip(bars,sec_avg.values):
                        ax.text(bar.get_x()+bar.get_width()/2,val+1,f"{val:.0f}",
                                ha="center",color="#6b4f3f",fontweight="bold")
                    ax.set_ylabel("Avg Score",color="#b09070")
                    ax.set_title("Score by Section",color="#2d1a0e",fontweight="bold",fontfamily="serif")
                    ax.set_ylim(0,115); plt.tight_layout(); st.pyplot(fig)

            with col4:
                st.markdown("""
                <div class="sec-divider"><div class="sec-divider-line"></div>
                <div class="sec-divider-text">Top Burnout Drivers</div>
                <div class="sec-divider-line" style="background:linear-gradient(90deg,transparent,#f0dcc8);"></div>
                </div>""", unsafe_allow_html=True)
                feat_imp = meta["feature_importances"]
                top_feats = sorted(feat_imp.items(),key=lambda x:-x[1])[:8]
                labels_f  = [f[0].replace("_"," ").title() for f,_ in top_feats]
                values_f  = [v for _,v in top_feats]
                fig,ax = warm_fig(5,3.5)
                clrs = ["#d4603a" if i<2 else "#f5a623" if i<5 else "#e8d5c0" for i in range(len(labels_f))]
                ax.barh(labels_f[::-1],values_f[::-1],color=clrs[::-1],height=0.55,edgecolor="none")
                ax.set_xlabel("Importance",color="#b09070")
                ax.set_title("Feature Importance",color="#2d1a0e",fontweight="bold",fontfamily="serif")
                ax.set_xlim(0,max(values_f)*1.25); plt.tight_layout(); st.pyplot(fig)

    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — DATASET
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown('<div class="page">', unsafe_allow_html=True)
    st.markdown("""
    <div class="anim-up">
      <div style="font-family:Playfair Display,serif;font-size:36px;font-weight:900;
      color:#2d1a0e;margin-bottom:8px;">Training Dataset</div>
      <div style="color:#8a6a55;font-size:15px;">300 student records used to train the ML model.</div>
    </div>""", unsafe_allow_html=True)
    try:
        df_t = pd.read_csv("burnout_dataset.csv")
        m1,m2,m3,m4 = st.columns(4)
        with m1: st.markdown('<div class="stat-pill anim-up"><div class="stat-num">' + str(len(df_t)) + '</div><div class="stat-lbl">Records</div></div>', unsafe_allow_html=True)
        with m2: st.markdown('<div class="stat-pill anim-up d1" style="border-top:4px solid #d4603a;"><div class="stat-num" style="color:#d4603a;">' + str(int((df_t["burnout_risk"]=="High").sum())) + '</div><div class="stat-lbl">At Risk</div></div>', unsafe_allow_html=True)
        with m3: st.markdown('<div class="stat-pill anim-up d2" style="border-top:4px solid #f5a623;"><div class="stat-num" style="color:#f5a623;">' + str(int((df_t["burnout_risk"]=="Medium").sum())) + '</div><div class="stat-lbl">Needs Attention</div></div>', unsafe_allow_html=True)
        with m4: st.markdown('<div class="stat-pill anim-up d3" style="border-top:4px solid #48b87a;"><div class="stat-num" style="color:#48b87a;">' + str(int((df_t["burnout_risk"]=="Low").sum())) + '</div><div class="stat-lbl">Thriving</div></div>', unsafe_allow_html=True)
        st.markdown("")
        rf3 = st.multiselect("Filter",["High","Medium","Low"],default=["High","Medium","Low"])
        st.dataframe(df_t[df_t["burnout_risk"].isin(rf3)],use_container_width=True,height=420)
        st.download_button("Download Dataset",data=df_t.to_csv(index=False).encode(),
                           file_name="burnout_dataset.csv",mime="text/csv")
    except FileNotFoundError:
        st.error("Run `python train_model.py` first.")
    st.markdown('</div>', unsafe_allow_html=True)

# ── FOOTER ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:white;border-top:1px solid #f0e6d8;padding:20px 36px;
margin-top:24px;display:flex;justify-content:space-between;align-items:center;">
  <div style="font-family:Playfair Display,serif;color:#2d1a0e;font-weight:700;font-size:16px;">
    Burnout Buster
  </div>
  <div style="color:#b09070;font-size:13px;">
    VIPS-TC Wellness Platform &nbsp;·&nbsp; Mohit Kumar
  </div>
</div>""", unsafe_allow_html=True)
