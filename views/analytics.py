"""Analytics tab — institution-wide charts for counselors."""
import streamlit as st
import pandas as pd
from constants import RISK_LABELS
from scoring import FEATURE_LABELS
from model_loader import load_model
from ui import section_divider
from utils import latest_per_student
import charts

def render():
    model, meta = load_model()
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

            latest = latest_per_student(all_subs)
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

            col1,col2 = st.columns(2)
            with col1:
                section_divider("Risk Distribution")
                if "burnout_risk" in latest.columns:
                    st.plotly_chart(charts.risk_pie(latest["burnout_risk"].value_counts().to_dict()),
                                    use_container_width=True)

            with col2:
                section_divider("Average Score by Branch")
                branch_col = "branch_s" if "branch_s" in latest.columns else "branch"
                if branch_col in latest.columns and "burnout_score" in latest.columns:
                    branch_avg = latest.groupby(branch_col)["burnout_score"].mean().sort_values()
                    st.plotly_chart(charts.score_bars(list(branch_avg.index), list(branch_avg.values),
                                                      horizontal=True, title="Burnout by Branch"),
                                    use_container_width=True)
                else:
                    st.info("Branch data appears once students submit with branch info.")

            section_divider("Burnout Trend Over Time")
            if "timestamp" in all_subs.columns and "burnout_score" in all_subs.columns:
                try:
                    all_subs["date"] = pd.to_datetime(all_subs["timestamp"]).dt.date
                    all_subs["burnout_score"] = pd.to_numeric(all_subs["burnout_score"],errors="coerce")
                    trend = all_subs.groupby("date")["burnout_score"].agg(["mean","size"]).reset_index()
                    if len(trend)>1:
                        st.plotly_chart(charts.trend(trend["date"], trend["mean"], trend["size"]),
                                        use_container_width=True)
                    else:
                        st.info("Trend chart appears once there are submissions on multiple dates.")
                except Exception:
                    st.info("Trend data will appear as more submissions come in.")

            col3,col4 = st.columns(2)
            with col3:
                section_divider("Section Comparison")
                sec_col = "section_s" if "section_s" in latest.columns else "section"
                if sec_col in latest.columns and "burnout_score" in latest.columns:
                    sec_avg = latest.groupby(sec_col)["burnout_score"].mean().sort_values(ascending=False)
                    st.plotly_chart(charts.score_bars(list(sec_avg.index), list(sec_avg.values),
                                                      title="Score by Section"),
                                    use_container_width=True)

            with col4:
                section_divider("Model Insight")
                top_feats = sorted(meta["feature_importances"].items(), key=lambda x: -x[1])[:8]
                st.plotly_chart(charts.importance([FEATURE_LABELS.get(f,f) for f,_ in top_feats],
                                                  [v*100 for _,v in top_feats]),
                                use_container_width=True)
                st.caption("Permutation importance on held-out data. It shows what the model uses, "
                           "learned from simulated training data — not proven causes of burnout.")

    st.markdown('</div>', unsafe_allow_html=True)
