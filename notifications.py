"""
notifications.py
In-app notification system for counselors.
Stores notifications in st.session_state so they persist during a session.
High-risk student submissions trigger a notification automatically.
"""

import streamlit as st
from datetime import datetime


def _init():
    """Initialise notification list in session state."""
    if "notifications" not in st.session_state:
        st.session_state.notifications = []
    if "notif_read_count" not in st.session_state:
        st.session_state.notif_read_count = 0


def add_notification(student_name: str, roll_number: str,
                     risk_level: str, confidence: float):
    """
    Add a new notification when a student submits a survey.
    Only High and Medium risk levels generate notifications.
    """
    _init()
    if risk_level not in ("High", "Medium"):
        return

    icon = "🔴" if risk_level == "High" else "🟡"
    urgency = "URGENT" if risk_level == "High" else "Warning"

    notif = {
        "id": len(st.session_state.notifications),
        "timestamp": datetime.now().strftime("%d %b %Y, %I:%M %p"),
        "student_name": student_name,
        "roll_number": roll_number,
        "risk_level": risk_level,
        "confidence": confidence,
        "icon": icon,
        "urgency": urgency,
        "read": False,
    }
    # Prepend so newest is first
    st.session_state.notifications.insert(0, notif)


def get_unread_count() -> int:
    _init()
    return sum(1 for n in st.session_state.notifications if not n["read"])


def mark_all_read():
    _init()
    for n in st.session_state.notifications:
        n["read"] = True


def mark_read(notif_id: int):
    _init()
    for n in st.session_state.notifications:
        if n["id"] == notif_id:
            n["read"] = True
            break


def clear_all():
    _init()
    st.session_state.notifications = []


def render_notification_bell():
    """
    Render a notification bell icon with unread badge in the sidebar.
    Call this from within a sidebar block.
    """
    _init()
    unread = get_unread_count()
    total  = len(st.session_state.notifications)

    if unread > 0:
        st.markdown(
            f"""<div style="background:#4f1010;border:1px solid #f55252;
            border-radius:10px;padding:10px 14px;margin-bottom:12px;
            display:flex;align-items:center;gap:10px;">
            <span style="font-size:22px;">🔔</span>
            <div>
              <div style="color:#f55252;font-weight:700;font-size:14px;">
                {unread} new alert{'s' if unread>1 else ''}
              </div>
              <div style="color:#fca5a5;font-size:12px;">
                High/Medium risk submissions
              </div>
            </div></div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""<div style="background:#1e1e3a;border:1px solid #2d2d5e;
            border-radius:10px;padding:10px 14px;margin-bottom:12px;
            display:flex;align-items:center;gap:10px;">
            <span style="font-size:22px;">🔕</span>
            <div style="color:#64748b;font-size:13px;">
              No new alerts · {total} total
            </div></div>""",
            unsafe_allow_html=True,
        )


def render_notification_panel():
    """
    Render the full notification panel (for use in the admin dashboard).
    Shows all notifications with read/unread state and mark-read buttons.
    """
    _init()
    notifs = st.session_state.notifications

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"### 🔔 Notifications ({len(notifs)})")
    with col2:
        if notifs and st.button("Mark all read", key="mark_all_read_btn"):
            mark_all_read()
            st.rerun()

    if not notifs:
        st.info("No notifications yet. They appear here when students submit the survey.")
        return

    for notif in notifs:
        bg    = "#4f1010" if notif["risk_level"] == "High" else "#4f3800"
        border= "#f55252" if notif["risk_level"] == "High" else "#f5c518"
        alpha = "1.0" if not notif["read"] else "0.5"

        with st.container():
            st.markdown(
                f"""<div style="background:{bg};border:1px solid {border};
                border-radius:12px;padding:14px 18px;margin-bottom:10px;
                opacity:{alpha};">
                <div style="display:flex;justify-content:space-between;
                align-items:flex-start;">
                  <div>
                    <span style="font-size:18px;">{notif['icon']}</span>
                    <strong style="color:#f1f5f9;margin-left:8px;">
                      {notif['student_name']}
                    </strong>
                    <span style="color:#94a3b8;font-size:12px;margin-left:6px;">
                      ({notif['roll_number']})
                    </span>
                    <span style="background:{border};color:#0f0f1a;
                    font-size:11px;font-weight:700;padding:2px 8px;
                    border-radius:20px;margin-left:8px;">
                      {notif['urgency']}
                    </span>
                    {'<span style="background:#7c3aed;color:white;font-size:10px;padding:2px 6px;border-radius:20px;margin-left:6px;">NEW</span>' if not notif["read"] else ""}
                  </div>
                  <span style="color:#64748b;font-size:11px;">{notif['timestamp']}</span>
                </div>
                <div style="color:#94a3b8;font-size:13px;margin-top:6px;">
                  {notif['risk_level']} burnout risk ·
                  {notif['confidence']*100:.0f}% model confidence
                </div></div>""",
                unsafe_allow_html=True,
            )
            if not notif["read"]:
                if st.button("✓ Mark read", key=f"read_{notif['id']}"):
                    mark_read(notif["id"])
                    st.rerun()
