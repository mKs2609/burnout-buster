"""Take Survey tab — the questionnaire, scoring and result."""
import streamlit as st
import pandas as pd
from constants import BRANCHES, COLLEGES, SECTIONS
from scoring import FEATURES, FEATURE_LABELS, assess
from model_loader import load_model
from ui import esc, first_name, score_info, section_divider
from advice import personalized_advice

def render():
    model, meta = load_model()
    from database import (get_student, register_student, verify_student, save_submission,
                          get_student_submissions, add_notification, MIN_PASSWORD_LENGTH)

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
        existing = get_student(s_roll.strip()) if s_roll and s_roll.strip() else None
        is_new = existing is None
        if is_new:
            st.markdown("**Create your portal password**")
            s_pwd  = st.text_input("Password *",         type="password", key="s_pwd",
                                   help=f"At least {MIN_PASSWORD_LENGTH} characters. "
                                        "Don't reuse your college portal password.")
            s_pwd2 = st.text_input("Confirm Password *", type="password", key="s_pwd2")
        else:
            st.info("Welcome back! This roll number already has a profile — enter your portal password to continue.")
            s_pwd  = st.text_input("Portal Password *", type="password", key="s_pwd_existing")
            s_pwd2 = None

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
            elif len(s_pwd.strip()) < MIN_PASSWORD_LENGTH:
                errors.append(f"Password must be at least {MIN_PASSWORD_LENGTH} characters")
            elif s_pwd != s_pwd2: errors.append("Passwords do not match")
        else:
            if not s_pwd: errors.append("Please enter your portal password")
            elif not verify_student(s_roll.strip(), s_pwd, s_branch, s_section):
                errors.append("Password, branch, or section doesn't match the profile registered for this roll number.")
        if errors:
            for e in errors: st.error(e)
        else:
            feat_vals = [q_exams,q_assign,q_attend,q_cgpa,q_backlog,q_study,
                         q_fomo,q_peer,q_family,q_social,q_reject,
                         q_sleep,q_exer,q_diet,q_conf,q_support,q_mhv]
            feat_dict = dict(zip(FEATURES,feat_vals))
            result = assess(model, meta, feat_dict)
            score, risk, proba = result["score"], result["risk"], result["proba"]
            slabel,color,risk_css,badge_css = score_info(score)
            # Returning students keep the name on their profile
            student_name = s_name.strip() if is_new else str(existing.get("name", s_name)).strip()

            saved = True
            if is_new:
                with st.spinner("Setting up your profile..."):
                    saved = register_student(s_roll.strip(),student_name,s_email.strip(),
                                             s_college,s_branch,s_section,s_age,s_pwd)

            flagged = False
            if saved:
                prev = get_student_submissions(s_roll.strip())
                if not prev.empty and "burnout_risk" in prev.columns:
                    if risk=="High" and prev.iloc[-1].get("burnout_risk","")=="High":
                        flagged = True

                with st.spinner("Saving your assessment..."):
                    saved = save_submission(s_roll.strip(),student_name,s_branch,s_section,
                                            feat_dict,score,risk,proba,s_note.strip() if s_note else "")

            if saved:
                if risk in ("High","Medium"):
                    add_notification(student_name,s_roll.strip(),risk,score,flagged)
            else:
                st.error("We couldn't save your assessment, so your counselor won't see it yet. "
                         "Your results are below — please try submitting again in a minute.")

            st.markdown("---")
            st.markdown(f"""
            <div class="anim-up" style="font-family:Playfair Display,serif;font-size:26px;
            font-weight:700;color:#2d1a0e;margin-bottom:20px;">
              Your results, {esc(first_name(student_name))}
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

            # Why this score: answers that raise it / lower it vs a typical thriving student
            if result["drivers"] or result["protective"] or result["safety_floor_applied"]:
                st.markdown("""
                <div class="sec-divider anim-up"><div class="sec-divider-line"></div>
                <div class="sec-divider-text">What's Shaping Your Score</div>
                <div class="sec-divider-line" style="background:linear-gradient(90deg,transparent,#f0dcc8);"></div>
                </div>""", unsafe_allow_html=True)
                ref = meta.get("reference_profile", {})
                def factor_rows(items, color, bg, verb):
                    out = ""
                    for f,impact in items:
                        out += ("<div style='display:flex;justify-content:space-between;align-items:center;"
                                "padding:10px 0;border-bottom:1px solid #f5ebe0;'>"
                                "<div><strong style='color:#2d1a0e;'>" + FEATURE_LABELS[f] + "</strong>"
                                "<span style='color:#b09070;font-size:12px;margin-left:10px;'>you: " + str(feat_dict[f])
                                + " · typical thriving student: " + str(ref.get(f,"–")) + "</span></div>"
                                "<span style='background:" + bg + ";color:" + color + ";font-size:11px;font-weight:700;"
                                "padding:3px 10px;border-radius:50px;white-space:nowrap;'>" + impact + " " + verb + "</span></div>")
                    return out
                panels = []
                if result["drivers"]:
                    panels.append(("Raising your score", factor_rows(result["drivers"], "#c04020", "#fdf0ec", "impact")))
                if result["protective"]:
                    panels.append(("Helping you", factor_rows(result["protective"], "#2a7d4f", "#e8f7ef", "help")))
                for col,(title,rows) in zip(st.columns(len(panels)) if panels else [], panels):
                    with col:
                        st.markdown("<div class='mag-card anim-up' style='padding:20px 24px;'>"
                                    "<div class='stat-lbl' style='margin:0 0 6px;text-align:left;'>" + title + "</div>"
                                    + rows + "</div>", unsafe_allow_html=True)
                if result["safety_floor_applied"]:
                    st.info("Your score was raised to at least **" + slabel + "** because of: "
                            + "; ".join(result["safety_reasons"])
                            + ". These answers matter on their own, even when everything else looks fine.")

            # Personalized tips
            st.markdown("""
            <div class="sec-divider anim-up"><div class="sec-divider-line"></div>
            <div class="sec-divider-text">Your Personal Action Plan</div>
            <div class="sec-divider-line" style="background:linear-gradient(90deg,transparent,#f0dcc8);"></div>
            </div>""", unsafe_allow_html=True)

            tips = personalized_advice(feat_dict, slabel, result["drivers"])
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
