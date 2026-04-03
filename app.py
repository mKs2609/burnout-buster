"""
app.py  ──  Burnout Buster: Student Burnout Risk Predictor
Run with:  streamlit run app.py
"""

import streamlit as st
import joblib, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Burnout Buster",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS Styling ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Inter:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; }

.main { background: #0f0f1a; }

.stApp { background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%); }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #0f0f1a 100%);
    border-right: 1px solid #2d2d4e;
}

/* Cards */
.risk-card {
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    margin: 8px 0;
    font-family: 'Space Grotesk', sans-serif;
}
.card-low    { background: linear-gradient(135deg, #0d4f3c, #1a7a5e); border: 1px solid #2dd4a0; }
.card-medium { background: linear-gradient(135deg, #4f3800, #7a5e00); border: 1px solid #f5c518; }
.card-high   { background: linear-gradient(135deg, #4f1010, #7a1a1a); border: 1px solid #f55252; }

.metric-box {
    background: #1e1e3a;
    border: 1px solid #2d2d5e;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    margin: 4px;
}

.section-header {
    color: #a78bfa;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 8px;
    margin-top: 20px;
}

/* Slider labels */
.stSlider label { color: #cbd5e1 !important; font-size: 14px !important; }

/* Predict button */
div.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 14px 40px;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 18px;
    font-weight: 700;
    width: 100%;
    cursor: pointer;
    transition: all 0.3s ease;
    letter-spacing: 0.5px;
}
div.stButton > button:hover {
    background: linear-gradient(135deg, #6d28d9, #4338ca);
    transform: translateY(-1px);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: #1e1e3a; border-radius: 12px; padding: 4px; }
.stTabs [data-baseweb="tab"] { color: #94a3b8 !important; border-radius: 8px; }
.stTabs [aria-selected="true"] { background: #7c3aed !important; color: white !important; }

/* Text */
p, li { color: #cbd5e1; }
h1 { color: #f1f5f9 !important; }
h2 { color: #e2e8f0 !important; }
h3 { color: #a78bfa !important; }
</style>
""", unsafe_allow_html=True)

# ── Load model & meta ──────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load("burnout_model.pkl")
    le    = joblib.load("label_encoder.pkl")
    with open("model_meta.json") as f:
        meta = json.load(f)
    return model, le, meta

model, le, meta = load_model()
FEATURES = meta["features"]

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔥 Burnout Buster")
    st.markdown("*AI-powered early warning system for student burnout*")
    st.markdown("---")
    st.markdown("**📊 Model Info**")
    st.markdown(f"- Algorithm: Random Forest")
    st.markdown(f"- Accuracy: **{meta['accuracy']}%**")
    st.markdown(f"- Training samples: 300")
    st.markdown(f"- Features tracked: {len(FEATURES)}")
    st.markdown("---")
    st.markdown("**🎓 Project Details**")
    st.markdown("- Course: AIDS 260 Practicum")
    st.markdown("- Institution: VIPS-TC")
    st.markdown("- Branch: AIDS-A")
    st.markdown("---")
    st.markdown("**👥 Team**")
    st.markdown("- Yash Choudhary")
    st.markdown("- Mohit Kumar")
    st.markdown("- Lakshay")
    st.markdown("*Supervisor: Dr Sapna Yadav*")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🔥 Burnout Buster")
st.markdown("##### Student Burnout Risk Prediction System | VIPS-TC College of Engineering")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🎯 Risk Predictor", "📈 Feature Insights", "📋 Dataset Explorer"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 – RISK PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Fill in the student profile below")
    st.markdown("Adjust each slider to match the student's current situation. The model will predict burnout risk instantly.")
    st.markdown("")

    col_a, col_b, col_c = st.columns(3)

    # ── Academic ──────────────────────────────────────────────────────────────
    with col_a:
        st.markdown('<div class="section-header">📚 Academic Factors</div>', unsafe_allow_html=True)
        exams_per_month      = st.slider("Exams per month",        1, 8,  4)
        assignments_per_week = st.slider("Assignments per week",   1, 12, 5)
        attendance_pressure  = st.slider("Attendance pressure",    1, 10, 6, help="How much pressure do they feel about attendance? (1=none, 10=extreme)")
        cgpa                 = st.slider("CGPA",                   4.0, 10.0, 7.0, step=0.1)
        backlogs             = st.slider("Number of backlogs",     0, 8,  1)
        study_hours_per_day  = st.slider("Study hours per day",    1, 12, 5)

    # ── Social / Psychological ────────────────────────────────────────────────
    with col_b:
        st.markdown('<div class="section-header">🧠 Social & Psychological</div>', unsafe_allow_html=True)
        fomo_score            = st.slider("FOMO level",               1, 10, 5, help="Fear Of Missing Out (1=none, 10=extreme)")
        peer_pressure         = st.slider("Peer pressure",            1, 10, 5)
        family_expectations   = st.slider("Family expectations",      1, 10, 6)
        social_media_hrs      = st.slider("Social media hrs/day",     0, 12, 4)
        rejection_sensitivity = st.slider("Rejection sensitivity",    1, 10, 5, help="How badly does rejection affect them?")
        mental_health_visits  = st.slider("Counselor visits (past month)", 0, 5, 0)

    # ── Lifestyle ────────────────────────────────────────────────────────────
    with col_c:
        st.markdown('<div class="section-header">🌙 Lifestyle & Emotional</div>', unsafe_allow_html=True)
        sleep_hours   = st.slider("Sleep hours per night",    3, 10, 6)
        exercise_days = st.slider("Exercise days per week",   0, 7,  2)
        diet_quality  = st.slider("Diet quality",             1, 10, 5, help="1=junk food only, 10=very healthy")
        confidence    = st.slider("Self-confidence",          1, 10, 5)
        support_system= st.slider("Support system strength",  1, 10, 5, help="Friends, family, mentors available?")

    st.markdown("")
    predict_btn = st.button("🔍 Predict Burnout Risk", use_container_width=True)

    if predict_btn:
        features = np.array([[
            exams_per_month, assignments_per_week, attendance_pressure,
            cgpa, backlogs, study_hours_per_day,
            fomo_score, peer_pressure, family_expectations,
            social_media_hrs, rejection_sensitivity,
            sleep_hours, exercise_days, diet_quality,
            confidence, support_system, mental_health_visits
        ]])

        pred_idx  = model.predict(features)[0]
        pred_prob = model.predict_proba(features)[0]
        pred_label = le.inverse_transform([pred_idx])[0]

        # ── Result Card ───────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("## 📊 Prediction Result")

        r1, r2, r3 = st.columns([1, 2, 1])
        with r2:
            if pred_label == "Low":
                st.markdown(f"""
                <div class="risk-card card-low">
                    <div style="font-size:56px;">🟢</div>
                    <div style="font-size:36px; font-weight:700; color:#2dd4a0; margin:8px 0;">LOW RISK</div>
                    <div style="color:#a7f3d0; font-size:16px;">Student shows healthy patterns. Keep it up!</div>
                </div>""", unsafe_allow_html=True)
            elif pred_label == "Medium":
                st.markdown(f"""
                <div class="risk-card card-medium">
                    <div style="font-size:56px;">🟡</div>
                    <div style="font-size:36px; font-weight:700; color:#f5c518; margin:8px 0;">MEDIUM RISK</div>
                    <div style="color:#fde68a; font-size:16px;">Early warning signs detected. Intervention recommended.</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-card card-high">
                    <div style="font-size:56px;">🔴</div>
                    <div style="font-size:36px; font-weight:700; color:#f55252; margin:8px 0;">HIGH RISK</div>
                    <div style="color:#fca5a5; font-size:16px;">⚠️ Urgent counselor attention required immediately!</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("")

        # ── Probability breakdown ─────────────────────────────────────────────
        st.markdown("#### Confidence Breakdown")
        prob_cols = st.columns(3)
        labels_ordered = le.classes_
        colors = {"High": "#f55252", "Low": "#2dd4a0", "Medium": "#f5c518"}
        emojis = {"High": "🔴", "Low": "🟢", "Medium": "🟡"}
        for i, (lbl, prob) in enumerate(zip(labels_ordered, pred_prob)):
            with prob_cols[i]:
                st.markdown(f"""
                <div class="metric-box">
                    <div style="font-size:24px;">{emojis[lbl]}</div>
                    <div style="color:{colors[lbl]}; font-size:22px; font-weight:700;">{prob*100:.1f}%</div>
                    <div style="color:#94a3b8; font-size:13px;">{lbl} Risk</div>
                </div>""", unsafe_allow_html=True)

        # ── Recommendations ───────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 💡 Recommendations")

        if pred_label == "Low":
            st.success("✅ Student appears mentally healthy. Continue current habits.")
            with st.expander("Preventive tips"):
                st.markdown("""
- Maintain consistent sleep schedule (7–8 hrs)
- Keep social media to under 2 hrs/day
- Build peer support networks proactively
- Schedule regular breaks during study sessions
                """)
        elif pred_label == "Medium":
            st.warning("⚠️ Student is showing early burnout signals. Act now before it worsens.")
            with st.expander("Action plan"):
                st.markdown("""
**Academic:**
- Speak to a mentor or academic advisor about workload
- Prioritise and drop non-essential commitments temporarily

**Lifestyle:**
- Aim for at least 7 hours of sleep
- Add 20–30 min of daily physical activity

**Social:**
- Reduce social media to under 1 hr/day
- Connect with a college counselor for a check-in session
                """)
        else:
            st.error("🚨 URGENT: Student is at HIGH risk. Immediate counselor referral required.")
            with st.expander("Emergency action plan"):
                st.markdown("""
**Immediate steps:**
- Schedule a same-day counselor appointment
- Inform the class mentor/guardian
- Consider temporary academic load reduction (contact examination cell)

**Support resources:**
- iCall Helpline: 9152987821
- Vandrevala Foundation: 1860-2662-345 (24/7)
- NIMHANS helpline: 080-46110007

**Follow-up:**
- Weekly check-ins for 4 weeks
- Progress monitoring using this tool
                """)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 – FEATURE INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📊 What Drives Burnout?")
    st.markdown("These are the most important features the model uses to predict burnout risk, ranked by importance.")

    feat_imp = meta["feature_importances"]
    sorted_feats = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)
    labels_plot  = [f[0].replace("_", " ").title() for f, _ in sorted_feats]
    values_plot  = [v for _, v in sorted_feats]

    bar_colors = ["#7c3aed" if i < 3 else "#4f46e5" if i < 6 else "#312e81" for i in range(len(labels_plot))]

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#1e1e3a")
    bars = ax.barh(labels_plot[::-1], values_plot[::-1], color=bar_colors[::-1], height=0.65, edgecolor="none")

    # Value labels on bars
    for bar, val in zip(bars, values_plot[::-1]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", color="#a78bfa", fontsize=10, fontweight="bold")

    ax.set_xlabel("Feature Importance Score", color="#94a3b8", fontsize=11)
    ax.set_title("Feature Importance – Burnout Risk Predictors", color="#f1f5f9", fontsize=14, fontweight="bold", pad=15)
    ax.tick_params(colors="#cbd5e1", labelsize=10)
    ax.spines[:].set_visible(False)
    ax.xaxis.set_tick_params(colors="#94a3b8")
    ax.set_xlim(0, max(values_plot) * 1.2)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")
    st.markdown("### 🔑 Key Insight Summary")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
**Top 3 burnout drivers:**
1. 🧠 **Rejection Sensitivity** — biggest psychological predictor
2. 📊 **CGPA** — academic performance directly linked to burnout
3. 🤝 **Support System** — lack of support amplifies all stressors
        """)
    with c2:
        st.markdown("""
**Protective factors:**
- 😴 More sleep hours → lower risk
- 🏃 Regular exercise → protective effect
- 🍎 Better diet → lower burnout score
- 💪 Higher confidence → significant buffer
        """)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 – DATASET EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📋 Training Dataset Explorer")
    st.markdown("Browse the 300-student dataset used to train the model.")

    try:
        df = pd.read_csv("burnout_dataset.csv")

        # Summary stats
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.metric("Total Records", len(df))
        with s2:
            st.metric("High Risk Students", int((df["burnout_risk"] == "High").sum()))
        with s3:
            st.metric("Medium Risk Students", int((df["burnout_risk"] == "Medium").sum()))
        with s4:
            st.metric("Low Risk Students", int((df["burnout_risk"] == "Low").sum()))

        st.markdown("#### Filter by risk level")
        risk_filter = st.multiselect("Show risk categories:", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
        filtered = df[df["burnout_risk"].isin(risk_filter)]

        # Color rows
        def color_risk(val):
            colors_map = {"High": "background-color:#4f1010; color:#fca5a5",
                          "Medium": "background-color:#4f3800; color:#fde68a",
                          "Low": "background-color:#0d4f3c; color:#a7f3d0"}
            return colors_map.get(val, "")

        styled = filtered.style.map(color_risk, subset=["burnout_risk"])
        st.write(styled, use_container_width=True, height=400)

        st.download_button(
            label="⬇️ Download Dataset (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="burnout_dataset.csv",
            mime="text/csv",
        )
    except FileNotFoundError:
        st.error("Dataset not found. Run `python generate_dataset.py` first.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#475569; font-size:13px;'>"
    "🔥 Burnout Buster | AIDS 260 Practicum | VIPS-TC College of Engineering | "
    "Team: Yash Choudhary, Mohit Kumar, Lakshay | Supervisor: Dr Sapna Yadav"
    "</div>",
    unsafe_allow_html=True
)
