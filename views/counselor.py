"""Counselor tab — alerts, records, the student list and actions."""
import streamlit as st
import pandas as pd
import hmac, html
from constants import BRANCHES, RISK_LABELS, SECTIONS
from database import storage_summary
from ui import esc
from utils import checkin_days_left, format_ts, get_reminder_frequency, latest_per_student

def render():
    from database import (get_all_submissions, get_all_students, upsert_counselor_action,
                          get_all_counselor_actions, save_reply, save_reminder,
                          save_college_records, get_college_records,
                          get_notifications, mark_notifications_read)

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
                # Pasting into a secrets box easily picks up a trailing space or
                # newline; that shouldn't lock a counselor out of the dashboard.
                try: expected = str(st.secrets["COUNSELOR_PASSWORD"]).strip()
                except Exception: expected = ""
                if not expected:
                    st.error("Counselor login isn't configured. Set COUNSELOR_PASSWORD in .streamlit/secrets.toml.")
                elif hmac.compare_digest(pwd.strip().encode("utf-8"), expected.encode("utf-8")):
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

        # Where data is being stored — a misconfigured DATABASE_URL would otherwise
        # leave the app quietly writing to storage that is wiped on every restart.
        store = storage_summary()
        if store["persistent"]:
            st.caption(f"Storage: {store['backend']} · {store['location']} — submissions are saved permanently.")
        else:
            st.warning(f"**Temporary storage in use** ({store['backend']}: {store['location']}). "
                       "Everything here is erased when the app restarts. Set `DATABASE_URL` in the app's "
                       "secrets — it must sit above any `[section]` line — then reboot the app.")

        # Alerts
        notifs = get_notifications()
        unread_c = int((~notifs["read"].astype(bool)).sum()) if not notifs.empty else 0

        with st.expander(f"Alerts  {'· ' + str(unread_c) + ' unread' if unread_c else '· all read'}", expanded=unread_c>0):
            if notifs.empty:
                st.markdown('<div style="color:#b09070;padding:12px;">No alerts yet.</div>', unsafe_allow_html=True)
            else:
                if unread_c and st.button("Mark all read", key="mark_all"):
                    if mark_notifications_read(): st.rerun()
                    else: st.error("Couldn't update alerts. Please try again.")
                for _,n in notifs.iterrows():
                    is_high = n.get("risk")=="High"
                    is_read = bool(n.get("read"))
                    color_n = "#d4603a" if is_high else "#f5a623"
                    bg_n    = "#fdf0ec" if is_high else "#fff8e8"
                    brd_n   = "#f0b8a8" if is_high else "#f5d890"
                    ts_n    = pd.to_datetime(n.get("created_at"),errors="coerce")
                    ts_n    = ts_n.strftime("%d %b, %I:%M %p") if pd.notna(ts_n) else ""
                    new_b   = "<span style='background:#d4603a;color:white;font-size:10px;padding:2px 8px;border-radius:50px;margin-left:8px;'>NEW</span>" if not is_read else ""
                    flag_b  = "<span style='background:#fdf0ec;color:#c04020;font-size:10px;font-weight:700;padding:2px 8px;border-radius:50px;border:1px solid #f0b8a8;margin-left:6px;'>PERSISTENT</span>" if bool(n.get("flagged")) else ""
                    notif_html = "<div style='background:" + bg_n + ";border:1.5px solid " + brd_n + ";border-radius:14px;padding:14px 18px;margin-bottom:10px;opacity:" + ('1' if not is_read else '0.55') + ";'>"
                    notif_html += "<strong style='color:#2d1a0e;'>" + html.escape(str(n.get('name',''))) + "</strong>"
                    notif_html += "<span style='color:#b09070;font-size:12px;margin-left:8px;'>· " + html.escape(str(n.get('roll_number',''))) + "</span>"
                    notif_html += "<span style='background:" + color_n + ";color:white;font-size:11px;font-weight:700;padding:3px 10px;border-radius:50px;margin-left:8px;'>" + html.escape(str(n.get('score',''))) + "/100 · " + html.escape(str(n.get('risk',''))) + "</span>"
                    notif_html += new_b + flag_b
                    notif_html += "<span style='float:right;color:#b09070;font-size:11px;'>" + ts_n + "</span></div>"
                    st.markdown(notif_html, unsafe_allow_html=True)
                    if not is_read:
                        if st.button("Mark read", key="nr_"+str(n.get('id'))):
                            if mark_notifications_read([n.get("id")]): st.rerun()
                            else: st.error("Couldn't update this alert. Please try again.")

        with st.expander("Survey Reminder Settings"):
            freq_options = [7,14,21,30,60]
            current_freq = get_reminder_frequency()
            if current_freq not in freq_options: freq_options = sorted(freq_options+[current_freq])
            st.markdown(f"Students are reminded in their portal when a check-in is due — "
                        f"**{current_freq} days after their own last survey** (currently). "
                        f"Overdue students are marked on the list below.")
            freq = st.select_slider("Check-in every",options=freq_options,value=current_freq,
                                    format_func=lambda x: f"{x} days")
            if st.button("Save Schedule", key="save_rem"):
                if save_reminder(freq):
                    st.success(f"Saved — each student is due {freq} days after their last check-in.")
                else:
                    st.error("Couldn't save the reminder schedule. Please try again.")

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
                    # dtype=str keeps leading zeros in roll numbers (01217711924)
                    df_up = pd.read_csv(uploaded, dtype=str)
                    df_up.columns = df_up.columns.str.strip().str.lower()
                    required = ["roll_number","name","attendance_pct","marks_pct","participation","remarks"]
                    missing = [c for c in required if c not in df_up.columns]
                    if missing:
                        st.error("Missing required column(s): " + ", ".join(missing))
                    else:
                        df_up["roll_number"] = df_up["roll_number"].str.strip()
                        st.dataframe(df_up.head(),use_container_width=True)
                        if st.button("Confirm Upload",key="confirm_upload"):
                            with st.spinner("Saving..."):
                                ok = save_college_records(df_up,up_branch,up_section)
                            if ok: st.success(f"Saved {len(df_up)} records for {up_branch}-{up_section}!")
                            else:  st.error("Couldn't save the records. Please try again.")
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

            # Counts use each student's latest result so they add up to the student total
            latest = latest_per_student(all_subs)
            unique_rolls = len(latest)
            risk_counts = latest["burnout_risk"].value_counts() if "burnout_risk" in latest.columns else pd.Series(dtype=int)
            high   = int(risk_counts.get("High",0))
            medium = int(risk_counts.get("Medium",0))
            low    = int(risk_counts.get("Low",0))

            m1,m2,m3,m4 = st.columns(4)
            with m1: st.markdown('<div class="stat-pill anim-up"><div class="stat-num">' + str(unique_rolls) + '</div><div class="stat-lbl">Students</div></div>', unsafe_allow_html=True)
            with m2: st.markdown('<div class="stat-pill anim-up d1" style="border-top:4px solid #d4603a;"><div class="stat-num" style="color:#d4603a;">' + str(high) + '</div><div class="stat-lbl">At Risk</div></div>', unsafe_allow_html=True)
            with m3: st.markdown('<div class="stat-pill anim-up d2" style="border-top:4px solid #f5a623;"><div class="stat-num" style="color:#f5a623;">' + str(medium) + '</div><div class="stat-lbl">Needs Attention</div></div>', unsafe_allow_html=True)
            with m4: st.markdown('<div class="stat-pill anim-up d3" style="border-top:4px solid #48b87a;"><div class="stat-num" style="color:#48b87a;">' + str(low) + '</div><div class="stat-lbl">Thriving</div></div>', unsafe_allow_html=True)

            st.markdown("")
            fc1,fc2,fc3,fc4 = st.columns(4)
            with fc1: rf = st.multiselect("Risk",["High","Medium","Low"],default=["High","Medium","Low"],
                                          format_func=lambda r: RISK_LABELS[r],key="rf")
            with fc2: fb = st.selectbox("Branch",["All"]+BRANCHES,key="fb")
            with fc3: search = st.text_input("Search name/roll","",key="srch")
            with fc4: show_flag = st.checkbox("Persistent At Risk only",key="sflag")

            if rf and "burnout_risk" in latest.columns:
                latest = latest[latest["burnout_risk"].isin(rf)]
            if fb!="All" and "branch" in latest.columns:
                latest = latest[latest["branch"]==fb]
            if search:
                rm = latest["roll_number"].astype(str).str.contains(search,na=False)
                nm = pd.Series([search.lower() in name_lookup.get(str(r),"").lower() for r in latest["roll_number"]],index=latest.index)
                latest = latest[rm|nm]

            st.markdown(f'<div style="color:#b09070;font-size:13px;margin-bottom:16px;">Showing {len(latest)} students</div>', unsafe_allow_html=True)

            # Fetch once for the whole list instead of two queries per student
            all_actions = get_all_counselor_actions()
            all_records = get_college_records()
            reminder_freq = get_reminder_frequency()

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

                action_d = all_actions.get(roll_d, {})
                status_d = action_d.get("status","Pending")
                notes_d  = action_d.get("notes","")

                badge_css_d = "badge-risk" if risk_d=="High" else "badge-attention" if risk_d=="Medium" else "badge-thriving"
                card_css_d  = "student-card-risk" if risk_d=="High" else "student-card-attention" if risk_d=="Medium" else "student-card-thriving"

                due_days = checkin_days_left(ts_d, reminder_freq)
                due_html = ""
                if due_days is not None and due_days < 0:
                    due_html = " <span style='background:#f5efe8;color:#8a6a55;font-size:10px;font-weight:700;padding:2px 8px;border-radius:50px;border:1px solid #e8d5c0;'>Check-in overdue " + str(-due_days) + "d</span>"

                records = all_records[all_records["roll_number"]==roll_d]
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

                left_part = "<strong style='color:#2d1a0e;font-size:15px;font-family:Playfair Display,serif;'>" + esc(sname_d) + "</strong>"
                left_part += "<span style='color:#b09070;font-size:12px;margin-left:10px;'>Roll: " + esc(roll_d) + "</span>"
                left_part += "<span style='color:#d4c0b0;font-size:12px;margin-left:8px;'>" + esc(branch_d) + "-" + esc(section_d) + "</span>"
                left_part += flag_html + consistency_flag + due_html

                right_part = "<span class='" + badge_css_d + "' style='font-size:12px;'>" + esc(RISK_LABELS.get(risk_d, risk_d)) + "</span>"
                right_part += "<span style='color:#6b4f3f;font-size:14px;font-family:Playfair Display,serif;font-weight:700;margin-left:10px;'>" + str(score_d) + "/100</span>"
                right_part += "<div style='color:#b09070;font-size:11px;margin-top:4px;'>" + esc(format_ts(ts_d)) + "</div>"

                note_part = ""
                if note_d and str(note_d).strip() and str(note_d).strip()!="nan":
                    note_part = "<div style='background:#fff8f0;border-left:3px solid #f5a623;border-radius:0 8px 8px 0;padding:8px 14px;margin-top:12px;color:#6b4f3f;font-size:13px;'><strong style='color:#b07820;'>Student note:</strong> " + esc(note_d).replace("\n","<br>") + "</div>"

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
                            if upsert_counselor_action(roll_d,new_status,new_notes,flagged_d):
                                st.success("Status updated!")
                            else:
                                st.error("Couldn't save the status. Please try again.")
                    with ac2:
                        st.markdown("**Send a Private Message**")
                        reply_msg = st.text_area("Your message",
                            placeholder="Hi, I reviewed your assessment and wanted to check in with you...",
                            height=100,key="rp_"+roll_d)
                        if st.button("Send Message",key="send_"+roll_d):
                            if reply_msg and reply_msg.strip():
                                if save_reply(roll_d,reply_msg.strip()):
                                    st.success("Message sent!")
                                else:
                                    st.error("Message wasn't sent. Please try again.")
                            else:
                                st.warning("Please type a message first.")

            st.markdown("---")
            st.download_button("Export All Data (CSV)",
                               data=all_subs.to_csv(index=False).encode(),
                               file_name="burnout_submissions.csv",mime="text/csv")

    st.markdown('</div>', unsafe_allow_html=True)
