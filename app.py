"""
app.py  ──  Burnout Buster v2
Tabs:
  1. 📝 Student Survey   – students fill in their own details & get a result
  2. 🎯 Quick Predictor  – manual slider-based predictor (for demos)
  3. 📈 Feature Insights – what drives burnout
  4. 🔐 Counselor Login  – password-protected admin dashboard
  5. 📋 Dataset Explorer – browse training data

Run:  streamlit run app.py
"""

import streamlit as st
import joblib, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Burnout Buster",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Inter:wght@300;400;500&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
h1,h2,h3{font-family:'Space Grotesk',sans-serif!important;}
.stApp{background:linear-gradient(135deg,#0f0f1a 0%,#1a1a2e 50%,#16213e 100%);}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#1a1a2e,#0f0f1a);border-right:1px solid #2d2d4e;}
.risk-card{border-radius:16px;padding:24px;text-align:center;margin:8px 0;font-family:'Space Grotesk',sans-serif;}
.card-low{background:linear-gradient(135deg,#0d4f3c,#1a7a5e);border:1px solid #2dd4a0;}
.card-med{background:linear-gradient(135deg,#4f3800,#7a5e00);border:1px solid #f5c518;}
.card-high{background:linear-gradient(135deg,#4f1010,#7a1a1a);border:1px solid #f55252;}
.section-hdr{color:#a78bfa;font-family:'Space Grotesk',sans-serif;font-size:13px;font-weight:600;letter-spacing:2px;text-transform:uppercase;margin:20px 0 8px;}
.metric-box{background:#1e1e3a;border:1px solid #2d2d5e;border-radius:12px;padding:16px;text-align:center;margin:4px;}
div.stButton>button{background:linear-gradient(135deg,#7c3aed,#4f46e5);color:white;border:none;border-radius:12px;padding:14px 40px;font-family:'Space Grotesk',sans-serif;font-size:16px;font-weight:700;width:100%;transition:all .3s ease;}
div.stButton>button:hover{background:linear-gradient(135deg,#6d28d9,#4338ca);}
.stTabs [data-baseweb="tab-list"]{background:#1e1e3a;border-radius:12px;padding:4px;}
.stTabs [data-baseweb="tab"]{color:#94a3b8!important;border-radius:8px;}
.stTabs [aria-selected="true"]{background:#7c3aed!important;color:white!important;}
p,li{color:#cbd5e1;}h1{color:#f1f5f9!important;}h2{color:#e2e8f0!important;}h3{color:#a78bfa!important;}
.admin-card{background:#1e1e3a;border:1px solid #2d2d5e;border-radius:12px;padding:16px;margin-bottom:10px;}
input[type="text"],input[type="password"]{background:#1e1e3a!important;color:#f1f5f9!important;border:1px solid #4c4c7a!important;border-radius:8px!important;}
</style>
""", unsafe_allow_html=True)

# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load("burnout_model.pkl")
    le    = joblib.load("label_encoder.pkl")
    with open("model_meta.json") as f:
        meta = json.load(f)
    return model, le, meta

model, le, meta = load_model()
FEATURES = meta["features"]

# ── Session state defaults ─────────────────────────────────────────────────────
for key, default in [
    ("counselor_logged_in", False),
    ("notifications",       []),
    ("submissions",         []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Helper: run prediction ─────────────────────────────────────────────────────
def predict(feature_values: list):
    arr        = np.array([feature_values])
    pred_idx   = model.predict(arr)[0]
    pred_proba = model.predict_proba(arr)[0]
    pred_label = le.inverse_transform([pred_idx])[0]
    proba_dict = dict(zip(le.classes_, pred_proba))
    return pred_label, proba_dict

# ── Helper: render result card ─────────────────────────────────────────────────
def render_result(pred_label, proba_dict):
    labels_map = {
        "Low":    ("card-low",  "🟢", "#2dd4a0", "#a7f3d0", "LOW RISK",    "Student shows healthy patterns. Keep it up!"),
        "Medium": ("card-med",  "🟡", "#f5c518", "#fde68a", "MEDIUM RISK", "Early warning signs detected. Intervention recommended."),
        "High":   ("card-high", "🔴", "#f55252", "#fca5a5", "HIGH RISK",   "⚠️ Urgent counselor attention required!"),
    }
    css, icon, col1, col2, label, msg = labels_map[pred_label]
    _, c, _ = st.columns([1, 2, 1])
    with c:
        st.markdown(f"""
        <div class="risk-card {css}">
          <div style="font-size:52px;">{icon}</div>
          <div style="font-size:34px;font-weight:700;color:{col1};margin:8px 0;">{label}</div>
          <div style="color:{col2};font-size:15px;">{msg}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("#### Confidence Breakdown")
    cols = st.columns(3)
    colors = {"High":"#f55252","Low":"#2dd4a0","Medium":"#f5c518"}
    emojis = {"High":"🔴","Low":"🟢","Medium":"🟡"}
    for i, lbl in enumerate(le.classes_):
        with cols[i]:
            st.markdown(f"""
            <div class="metric-box">
              <div style="font-size:22px;">{emojis[lbl]}</div>
              <div style="color:{colors[lbl]};font-size:22px;font-weight:700;">{proba_dict[lbl]*100:.1f}%</div>
              <div style="color:#94a3b8;font-size:13px;">{lbl} Risk</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 💡 Recommendations")
    if pred_label == "Low":
        st.success("✅ Student appears mentally healthy. Maintain current habits.")
        with st.expander("Preventive tips"):
            st.markdown("- Maintain 7–8 hours sleep\n- Keep social media under 2 hrs/day\n- Build peer support networks\n- Take regular study breaks")
    elif pred_label == "Medium":
        st.warning("⚠️ Early burnout signals detected. Act now before it worsens.")
        with st.expander("Action plan"):
            st.markdown("**Academic:** Speak to mentor about workload, prioritise tasks\n\n**Lifestyle:** Aim for 7 hrs sleep, add 20 min daily exercise\n\n**Social:** Reduce social media, connect with a counselor")
    else:
        st.error("🚨 URGENT: Student at HIGH risk. Immediate counselor referral required.")
        with st.expander("Emergency action plan"):
            st.markdown("- Schedule a same-day counselor appointment\n- Inform class mentor/guardian\n- Consider temporary academic load reduction\n\n**Helplines:**\n- iCall: 9152987821\n- Vandrevala: 1860-2662-345 (24/7)\n- NIMHANS: 080-46110007")

# ── Helper: push notification ──────────────────────────────────────────────────
def push_notification(name, roll, risk, confidence):
    if risk not in ("High", "Medium"):
        return
    from datetime import datetime
    icon    = "🔴" if risk == "High" else "🟡"
    urgency = "URGENT" if risk == "High" else "Warning"
    st.session_state.notifications.insert(0, {
        "id":           len(st.session_state.notifications),
        "timestamp":    datetime.now().strftime("%d %b %Y, %I:%M %p"),
        "student_name": name,
        "roll_number":  roll,
        "risk_level":   risk,
        "confidence":   confidence,
        "icon":         icon,
        "urgency":      urgency,
        "read":         False,
    })

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔥 Burnout Buster")
    st.markdown("*AI-powered early warning system*")
    st.markdown("---")

    unread = sum(1 for n in st.session_state.notifications if not n["read"])
    if unread > 0:
        st.markdown(f"""
        <div style="background:#4f1010;border:1px solid #f55252;border-radius:10px;
        padding:10px 14px;margin-bottom:12px;">
          <span style="font-size:20px;">🔔</span>
          <strong style="color:#f55252;margin-left:8px;">{unread} new alert{'s' if unread>1 else ''}</strong>
          <div style="color:#fca5a5;font-size:12px;margin-top:2px;">Go to Counselor tab</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("**📊 Model Info**")
    st.markdown(f"- Algorithm: Random Forest\n- Accuracy: **{meta['accuracy']}%**\n- Features: {len(FEATURES)}\n- Training samples: 300")
    st.markdown("---")
    st.markdown("**🎓 VIPS-TC · AIDS 260**")
    st.markdown("Yash · Mohit · Lakshay\n\n*Supervisor: Dr Sapna Yadav*")

# ── HEADER ─────────────────────────────────────────────────────────────────────
st.markdown("# 🔥 Burnout Buster")
st.markdown("##### Student Burnout Risk Prediction System | VIPS-TC College of Engineering")
st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📝 Student Survey",
    "🎯 Quick Predictor",
    "📈 Feature Insights",
    "🔐 Counselor Dashboard",
    "📋 Dataset Explorer",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — STUDENT SURVEY (real data collection)
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### 📝 Student Self-Assessment Survey")
    st.markdown("Fill in your details honestly. Your response is confidential and helps identify if you need support.")
    st.markdown("")

    # ── Student identity ──────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">👤 Your Details</div>', unsafe_allow_html=True)
    id_col1, id_col2, id_col3 = st.columns(3)
    with id_col1:
        s_name = st.text_input("Full Name", placeholder="e.g. Ravi Sharma")
    with id_col2:
        s_roll = st.text_input("Roll Number", placeholder="e.g. 01217711924")
    with id_col3:
        s_email = st.text_input("College Email", placeholder="e.g. ravi@vips.edu")

    st.markdown("---")

    # ── Survey questions (3 columns) ──────────────────────────────────────────
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown('<div class="section-hdr">📚 Academic</div>', unsafe_allow_html=True)
        q_exams   = st.select_slider("How many exams do you have per month?",
                        options=list(range(1, 9)), value=4)
        q_assign  = st.select_slider("Assignments per week?",
                        options=list(range(1, 13)), value=5)
        q_attend  = st.slider("Attendance pressure (1=none, 10=extreme)", 1, 10, 6)
        q_cgpa    = st.slider("Your current CGPA", 4.0, 10.0, 7.0, step=0.1)
        q_backlog = st.select_slider("Number of active backlogs",
                        options=list(range(0, 9)), value=0)
        q_study   = st.select_slider("Study hours per day (avg)",
                        options=list(range(1, 13)), value=5)

    with c2:
        st.markdown('<div class="section-hdr">🧠 Social & Mental</div>', unsafe_allow_html=True)
        q_fomo    = st.slider("FOMO level (fear of missing out)", 1, 10, 5,
                        help="How much do you fear missing social events?")
        q_peer    = st.slider("Peer pressure intensity", 1, 10, 5)
        q_family  = st.slider("Family expectations pressure", 1, 10, 6)
        q_social  = st.select_slider("Social media hours per day",
                        options=list(range(0, 13)), value=3)
        q_reject  = st.slider("Rejection sensitivity (1=low, 10=very high)", 1, 10, 5,
                        help="How badly does failure or rejection affect your mood?")
        q_mhv     = st.select_slider("Counselor/therapist visits this month",
                        options=list(range(0, 6)), value=0)

    with c3:
        st.markdown('<div class="section-hdr">🌙 Lifestyle</div>', unsafe_allow_html=True)
        q_sleep   = st.select_slider("Average sleep hours per night",
                        options=list(range(3, 11)), value=6)
        q_exer    = st.select_slider("Exercise days per week",
                        options=list(range(0, 8)), value=2)
        q_diet    = st.slider("Diet quality (1=junk food, 10=very healthy)", 1, 10, 5)
        q_conf    = st.slider("Self-confidence level", 1, 10, 5)
        q_support = st.slider("Support system (friends/family/mentors)", 1, 10, 5,
                        help="How much support do you have around you?")

    st.markdown("")

    # ── Consent checkbox ──────────────────────────────────────────────────────
    consent = st.checkbox("✅ I understand this is a screening tool, not a clinical diagnosis. I consent to sharing my responses for counselor review.")

    submit_btn = st.button("📤 Submit Survey & See My Risk", use_container_width=True)

    if submit_btn:
        if not s_name.strip() or not s_roll.strip():
            st.error("Please enter your Name and Roll Number before submitting.")
        elif not consent:
            st.warning("Please check the consent box to proceed.")
        else:
            feature_values = [
                q_exams, q_assign, q_attend, q_cgpa, q_backlog, q_study,
                q_fomo, q_peer, q_family, q_social, q_reject,
                q_sleep, q_exer, q_diet, q_conf, q_support, q_mhv
            ]
            features_dict = dict(zip(FEATURES, feature_values))
            pred_label, proba_dict = predict(feature_values)

            # ── Save submission ───────────────────────────────────────────────
            from datetime import datetime
            submission = {
                "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "student_name": s_name.strip(),
                "roll_number":  s_roll.strip(),
                "email":        s_email.strip(),
                "burnout_risk": pred_label,
                "confidence_high":   round(proba_dict.get("High",   0), 3),
                "confidence_medium": round(proba_dict.get("Medium", 0), 3),
                "confidence_low":    round(proba_dict.get("Low",    0), 3),
                "counselor_status":  "Pending",
                "counselor_notes":   "",
                **features_dict,
            }
            st.session_state.submissions.append(submission)

            # ── Push notification to counselor ────────────────────────────────
            push_notification(
                s_name.strip(), s_roll.strip(),
                pred_label, proba_dict.get(pred_label, 0)
            )

            # ── Save to Google Sheets (if configured) ─────────────────────────
            try:
                from database import save_submission
                save_submission(
                    {"name": s_name, "roll_number": s_roll, "email": s_email},
                    features_dict, pred_label, proba_dict
                )
            except Exception:
                pass  # Local session_state already has the data

            # ── Log to MLflow (if installed) ──────────────────────────────────
            try:
                from mlflow_tracker import log_prediction
                log_prediction(s_name, s_roll, features_dict,
                               pred_label, proba_dict, meta["accuracy"])
            except Exception:
                pass

            st.success(f"✅ Survey submitted! Hi **{s_name.strip()}**, here are your results:")
            st.markdown("---")
            render_result(pred_label, proba_dict)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — QUICK PREDICTOR (demo / manual)
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 🎯 Quick Predictor")
    st.markdown("Manually adjust sliders to explore how different factors affect burnout risk. Great for demos.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="section-hdr">📚 Academic</div>', unsafe_allow_html=True)
        p_exams   = st.slider("Exams per month",          1, 8,  4,  key="p_exams")
        p_assign  = st.slider("Assignments per week",     1, 12, 5,  key="p_assign")
        p_attend  = st.slider("Attendance pressure",      1, 10, 6,  key="p_attend")
        p_cgpa    = st.slider("CGPA",                     4.0, 10.0, 7.0, step=0.1, key="p_cgpa")
        p_backlog = st.slider("Backlogs",                 0, 8,  1,  key="p_back")
        p_study   = st.slider("Study hours/day",          1, 12, 5,  key="p_study")
    with c2:
        st.markdown('<div class="section-hdr">🧠 Social & Mental</div>', unsafe_allow_html=True)
        p_fomo    = st.slider("FOMO level",               1, 10, 5, key="p_fomo")
        p_peer    = st.slider("Peer pressure",            1, 10, 5, key="p_peer")
        p_family  = st.slider("Family expectations",      1, 10, 6, key="p_fam")
        p_social  = st.slider("Social media hrs/day",     0, 12, 3, key="p_soc")
        p_reject  = st.slider("Rejection sensitivity",    1, 10, 5, key="p_rej")
        p_mhv     = st.slider("Counselor visits/month",   0, 5,  0, key="p_mhv")
    with c3:
        st.markdown('<div class="section-hdr">🌙 Lifestyle</div>', unsafe_allow_html=True)
        p_sleep   = st.slider("Sleep hrs/night",          3, 10, 6, key="p_sleep")
        p_exer    = st.slider("Exercise days/week",       0, 7,  2, key="p_exer")
        p_diet    = st.slider("Diet quality",             1, 10, 5, key="p_diet")
        p_conf    = st.slider("Self-confidence",          1, 10, 5, key="p_conf")
        p_support = st.slider("Support system",           1, 10, 5, key="p_sup")

    if st.button("🔍 Predict Risk", use_container_width=True, key="quick_pred_btn"):
        fvals = [p_exams, p_assign, p_attend, p_cgpa, p_backlog, p_study,
                 p_fomo, p_peer, p_family, p_social, p_reject,
                 p_sleep, p_exer, p_diet, p_conf, p_support, p_mhv]
        pred_label, proba_dict = predict(fvals)
        st.markdown("---")
        render_result(pred_label, proba_dict)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — FEATURE INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📊 What Drives Burnout?")
    st.markdown("Top features the model uses to predict burnout risk, ranked by importance score.")

    feat_imp = meta["feature_importances"]
    sorted_feats = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)
    labels_plot  = [f[0].replace("_", " ").title() for f, _ in sorted_feats]
    values_plot  = [v for _, v in sorted_feats]
    bar_colors   = ["#7c3aed" if i < 3 else "#4f46e5" if i < 6 else "#312e81"
                    for i in range(len(labels_plot))]

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#1e1e3a")
    bars = ax.barh(labels_plot[::-1], values_plot[::-1], color=bar_colors[::-1],
                   height=0.65, edgecolor="none")
    for bar, val in zip(bars, values_plot[::-1]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", color="#a78bfa", fontsize=10, fontweight="bold")
    ax.set_xlabel("Feature Importance Score", color="#94a3b8", fontsize=11)
    ax.set_title("Feature Importance — Burnout Risk Predictors",
                 color="#f1f5f9", fontsize=14, fontweight="bold", pad=15)
    ax.tick_params(colors="#cbd5e1", labelsize=10)
    ax.spines[:].set_visible(False)
    ax.set_xlim(0, max(values_plot) * 1.2)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Top 3 burnout drivers:**\n1. 🧠 Rejection Sensitivity\n2. 📊 CGPA\n3. 🤝 Support System")
    with c2:
        st.markdown("**Protective factors:**\n- 😴 More sleep → lower risk\n- 🏃 Regular exercise → buffer\n- 💪 Higher confidence → protective\n- 🍎 Better diet → lower score")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — COUNSELOR DASHBOARD (password protected)
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    if not st.session_state.counselor_logged_in:
        st.markdown("### 🔐 Counselor Login")
        st.markdown("This section is restricted to authorised counselors only.")
        st.markdown("")
        _, lc, _ = st.columns([1, 1.5, 1])
        with lc:
            pwd = st.text_input("Enter counselor password", type="password",
                                placeholder="••••••••", key="counselor_pwd")
            if st.button("🔓 Login", use_container_width=True, key="login_btn"):
                import os
                COUNSELOR_PASSWORD = st.secrets.get("COUNSELOR_PASSWORD", "changeme")
                if pwd == COUNSELOR_PASSWORD:
                    st.session_state.counselor_logged_in = True
                    st.rerun()
                else:
                    st.error("❌ Incorrect password.")
           
    else:
        # ── Logged in ──────────────────────────────────────────────────────────
        hdr1, hdr2 = st.columns([4, 1])
        with hdr1:
            st.markdown("### 🛡️ Counselor Admin Dashboard")
        with hdr2:
            if st.button("🚪 Logout", key="logout_btn"):
                st.session_state.counselor_logged_in = False
                st.rerun()

        # ── Notification panel ────────────────────────────────────────────────
        notifs = st.session_state.notifications
        unread = sum(1 for n in notifs if not n["read"])

        with st.expander(f"🔔 Notifications  {'🔴 ' + str(unread) + ' new' if unread else '(all read)'}",
                         expanded=unread > 0):
            if not notifs:
                st.info("No notifications yet. They appear when students submit the survey.")
            else:
                col_hdr, col_btn = st.columns([3, 1])
                with col_btn:
                    if st.button("Mark all read", key="mark_all"):
                        for n in st.session_state.notifications:
                            n["read"] = True
                        st.rerun()
                for notif in notifs:
                    bg     = "#4f1010" if notif["risk_level"] == "High" else "#4f3800"
                    border = "#f55252" if notif["risk_level"] == "High" else "#f5c518"
                    alpha  = "1.0" if not notif["read"] else "0.5"
                    new_badge = '<span style="background:#7c3aed;color:white;font-size:10px;padding:2px 6px;border-radius:20px;margin-left:6px;">NEW</span>' if not notif["read"] else ""
                    st.markdown(f"""
                    <div style="background:{bg};border:1px solid {border};border-radius:10px;
                    padding:12px 16px;margin-bottom:8px;opacity:{alpha};">
                      <span style="font-size:16px;">{notif['icon']}</span>
                      <strong style="color:#f1f5f9;margin-left:6px;">{notif['student_name']}</strong>
                      <span style="color:#94a3b8;font-size:12px;"> ({notif['roll_number']})</span>
                      <span style="background:{border};color:#0f0f1a;font-size:11px;font-weight:700;
                      padding:2px 8px;border-radius:20px;margin-left:8px;">{notif['urgency']}</span>
                      {new_badge}
                      <span style="float:right;color:#64748b;font-size:11px;">{notif['timestamp']}</span>
                      <div style="color:#94a3b8;font-size:12px;margin-top:4px;">
                        {notif['risk_level']} risk · {notif['confidence']*100:.0f}% confidence
                      </div>
                    </div>""", unsafe_allow_html=True)
                    if not notif["read"]:
                        if st.button("✓ Mark read", key=f"nr_{notif['id']}"):
                            notif["read"] = True
                            st.rerun()

        st.markdown("---")

        # ── Load submissions ───────────────────────────────────────────────────
        # Try Google Sheets first, then local session_state
        all_submissions = []
        try:
            from database import load_all_submissions
            df_sheets = load_all_submissions()
            if not df_sheets.empty:
                all_submissions = df_sheets.to_dict("records")
        except Exception:
            pass

        if not all_submissions:
            all_submissions = st.session_state.submissions

        if not all_submissions:
            st.info("📭 No student submissions yet. Students need to fill the survey in the **📝 Student Survey** tab.")
        else:
            df_sub = pd.DataFrame(all_submissions)

            # ── Summary metrics ────────────────────────────────────────────────
            total  = len(df_sub)
            high   = int((df_sub["burnout_risk"] == "High").sum())
            medium = int((df_sub["burnout_risk"] == "Medium").sum())
            low    = int((df_sub["burnout_risk"] == "Low").sum())
            pending = int((df_sub.get("counselor_status", pd.Series(["Pending"]*total)) == "Pending").sum())

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Total Students", total)
            m2.metric("🔴 High Risk",   high,   delta=f"{high/total*100:.0f}%"   if total else "0%")
            m3.metric("🟡 Medium Risk", medium, delta=f"{medium/total*100:.0f}%" if total else "0%")
            m4.metric("🟢 Low Risk",    low,    delta=f"{low/total*100:.0f}%"    if total else "0%")
            m5.metric("⏳ Pending Action", pending)

            # ── Filters ────────────────────────────────────────────────────────
            st.markdown("#### 🔍 Filter Students")
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                risk_filter = st.multiselect("Risk level", ["High","Medium","Low"],
                                             default=["High","Medium","Low"], key="rf")
            with fc2:
                status_options = list(df_sub["counselor_status"].unique()) if "counselor_status" in df_sub.columns else ["Pending"]
                status_filter  = st.multiselect("Counselor status", status_options,
                                                default=status_options, key="sf")
            with fc3:
                search = st.text_input("🔎 Search by name or roll number", "", key="srch")

            df_filtered = df_sub[df_sub["burnout_risk"].isin(risk_filter)]
            if "counselor_status" in df_filtered.columns:
                df_filtered = df_filtered[df_filtered["counselor_status"].isin(status_filter)]
            if search:
                mask = (
                    df_filtered["student_name"].str.contains(search, case=False, na=False)
                    | df_filtered["roll_number"].astype(str).str.contains(search, na=False)
                )
                df_filtered = df_filtered[mask]

            st.markdown(f"Showing **{len(df_filtered)}** of {total} students")

            # ── Student cards ──────────────────────────────────────────────────
            for _, row in df_filtered.iterrows():
                risk   = row.get("burnout_risk", "Unknown")
                name   = row.get("student_name", "Unknown")
                roll   = str(row.get("roll_number", ""))
                email  = row.get("email", "—")
                status = row.get("counselor_status", "Pending")
                notes  = row.get("counselor_notes", "")
                ts     = row.get("timestamp", "")

                bg_map     = {"High":"#4f1010","Medium":"#4f3800","Low":"#0d4f3c"}
                brd_map    = {"High":"#f55252","Medium":"#f5c518","Low":"#2dd4a0"}
                icon_map   = {"High":"🔴","Medium":"🟡","Low":"🟢"}
                status_bg  = {"Pending":"#312e81","Contacted":"#0d4f3c","No Action Needed":"#374151"}

                bg  = bg_map.get(risk, "#1e1e3a")
                brd = brd_map.get(risk, "#2d2d5e")

                with st.container():
                    st.markdown(f"""
                    <div style="background:{bg};border:1px solid {brd};border-radius:12px;
                    padding:16px 20px;margin-bottom:8px;">
                      <div style="display:flex;justify-content:space-between;align-items:center;">
                        <div>
                          <span style="font-size:20px;">{icon_map.get(risk,'⚪')}</span>
                          <strong style="color:#f1f5f9;font-size:16px;margin-left:8px;">{name}</strong>
                          <span style="color:#94a3b8;font-size:13px;margin-left:8px;">Roll: {roll}</span>
                          <span style="color:#64748b;font-size:12px;margin-left:12px;">✉ {email}</span>
                        </div>
                        <div style="text-align:right;">
                          <span style="background:{brd};color:#0f0f1a;font-size:12px;font-weight:700;
                          padding:3px 10px;border-radius:20px;">{risk} Risk</span>
                          <div style="color:#64748b;font-size:11px;margin-top:4px;">{ts}</div>
                        </div>
                      </div>
                    </div>""", unsafe_allow_html=True)

                    with st.expander(f"⚙️ Actions for {name} ({roll})"):
                        ac1, ac2 = st.columns([1, 2])
                        with ac1:
                            new_status = st.selectbox(
                                "Update status",
                                ["Pending", "Contacted", "No Action Needed"],
                                index=["Pending","Contacted","No Action Needed"].index(status)
                                if status in ["Pending","Contacted","No Action Needed"] else 0,
                                key=f"status_{roll}_{ts}"
                            )
                        with ac2:
                            new_notes = st.text_area("Counselor notes",
                                                     value=notes, height=80,
                                                     key=f"notes_{roll}_{ts}",
                                                     placeholder="e.g. Scheduled meeting on Monday...")
                        if st.button("💾 Save", key=f"save_{roll}_{ts}"):
                            # Update in session_state
                            for s in st.session_state.submissions:
                                if str(s.get("roll_number")) == str(roll):
                                    s["counselor_status"] = new_status
                                    s["counselor_notes"]  = new_notes
                            # Update in Google Sheets if connected
                            try:
                                from database import update_counselor_status
                                update_counselor_status(roll, new_status, new_notes)
                            except Exception:
                                pass
                            st.success(f"✅ Updated {name}'s status to **{new_status}**")

            # ── Export button ──────────────────────────────────────────────────
            st.markdown("---")
            st.download_button(
                "⬇️ Export All Data (CSV)",
                data=df_sub.to_csv(index=False).encode("utf-8"),
                file_name="burnout_submissions.csv",
                mime="text/csv",
            )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — DATASET EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### 📋 Training Dataset Explorer")
    st.markdown("The 300-student synthetic dataset used to train the model.")
    try:
        df_train = pd.read_csv("burnout_dataset.csv")
        t1, t2, t3, t4 = st.columns(4)
        t1.metric("Total Records",   len(df_train))
        t2.metric("High Risk",  int((df_train["burnout_risk"] == "High").sum()))
        t3.metric("Medium Risk",int((df_train["burnout_risk"] == "Medium").sum()))
        t4.metric("Low Risk",   int((df_train["burnout_risk"] == "Low").sum()))

        rf = st.multiselect("Filter", ["High","Medium","Low"], default=["High","Medium","Low"])
        def colour_risk(val):
            m = {"High":"background-color:#4f1010;color:#fca5a5",
                 "Medium":"background-color:#4f3800;color:#fde68a",
                 "Low":"background-color:#0d4f3c;color:#a7f3d0"}
            return m.get(val, "")
        styled = df_train[df_train["burnout_risk"].isin(rf)].style.applymap(
            colour_risk, subset=["burnout_risk"])
        st.dataframe(styled, use_container_width=True, height=400)
        st.download_button("⬇️ Download Dataset (CSV)",
                           data=df_train.to_csv(index=False).encode("utf-8"),
                           file_name="burnout_dataset.csv", mime="text/csv")
    except FileNotFoundError:
        st.error("Dataset not found. Run `python train_model.py` first.")

# ── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#475569;font-size:13px;'>"
    "🔥 Burnout Buster v2 | AIDS 260 Practicum | VIPS-TC | "
    "Yash Choudhary · Mohit Kumar · Lakshay | Supervisor: Dr Sapna Yadav"
    "</div>", unsafe_allow_html=True)
