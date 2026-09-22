"""My Portal tab — a student's own history and counselor messages."""
import streamlit as st
import pandas as pd
from constants import BRANCHES, SECTIONS
from ui import esc, first_name, score_info, section_divider
from utils import checkin_days_left, get_reminder_frequency, trajectory
import charts

def render():
    from database import verify_student, get_student_submissions, get_replies, mark_replies_read

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
              font-weight:900;color:#2d1a0e;">Hey, {esc(first_name(name))}.</div>
              <div style="color:#8a6a55;font-size:15px;">Here's your wellness overview.</div>
            </div>""", unsafe_allow_html=True)
        with hc2:
            if st.button("Sign Out", key="s_out"):
                st.session_state.student_logged_in = False
                st.session_state.student_data = {}
                st.rerun()

        subs = get_student_submissions(roll)

        # Check-in reminder: due a set number of days after this student's own last survey
        if not subs.empty:
            days_left = checkin_days_left(subs["timestamp"].max(), get_reminder_frequency())
            if days_left is not None and days_left <= 3:
                if days_left < 0:   due_txt = f"was due {-days_left} day(s) ago"
                elif days_left == 0: due_txt = "is due today"
                else:               due_txt = f"is due in {days_left} day(s)"
                st.warning(f"Your next wellness check-in {due_txt}. Head to Take Survey when you have 3 minutes.")

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
                section_divider("Score History")
                dates = pd.to_datetime(subs["timestamp"], errors="coerce")
                st.plotly_chart(charts.score_history(dates, scores), use_container_width=True)
                st.caption("Hover a point for the exact score and date; drag to zoom.")

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
                            "Your Counselor &nbsp;·&nbsp; " + esc(ts) + "</div>"
                            + esc(msg).replace("\n","<br>") + "</div>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
