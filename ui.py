"""ui.py — shared look and feel: stylesheet, navbar, footer, small text helpers."""
import html
import streamlit as st
from constants import RISK_LABELS

def inject_css():
    """App-wide stylesheet. Colors live here, not in the view modules."""
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
    /* Style the thumb and labels only. A blanket "... slider div" rule paints the
       track, thumb and value labels the same orange, so the numbers disappear into
       one solid bar. The filled track already follows theme primaryColor. */
    [data-testid="stSlider"] [role="slider"],
    [data-testid="stSelectSlider"] [role="slider"] {
        background: #d4603a !important;
        border: 2px solid white !important;
        box-shadow: 0 2px 8px rgba(212,96,58,0.35) !important;
    }
    [data-testid="stSliderThumbValue"] {
        color: #d4603a !important;
        background: transparent !important;
        font-weight: 700 !important;
    }
    [data-testid="stSliderTickBarMin"],
    [data-testid="stSliderTickBarMax"] {
        color: #b09070 !important;
        background: transparent !important;
    }
    
    /* ── Checkbox ── */
    [data-testid="stCheckbox"] label { color: #6b4f3f !important; text-transform: none !important; font-size: 14px !important; letter-spacing: 0 !important; }
    
    /* ── Multiselect ── */
    [data-baseweb="tag"] { background: #fdf0ec !important; color: #d4603a !important; }
    
    /* ── Progress bar ── */
    .stProgress > div > div { background: linear-gradient(90deg, #d4603a, #f5a623) !important; border-radius: 50px !important; }
    """, unsafe_allow_html=True)


def esc(v):
    """Escape user-supplied text before it goes into unsafe_allow_html markup."""
    return html.escape("" if v is None else str(v))


def first_name(name):
    parts = str(name or "").split()
    return parts[0] if parts else "there"


def score_info(s):
    """(label, color, css key, badge class) for a 0-100 score."""
    if s <= 33:  return "Thriving",        "#48b87a", "thriving",  "badge-thriving"
    elif s <= 66: return "Needs Attention", "#f5a623", "attention", "badge-attention"
    else:        return "At Risk",         "#d4603a", "risk",      "badge-risk"


def risk_badge(risk, font_size="12px"):
    css = {"High":"badge-risk","Medium":"badge-attention","Low":"badge-thriving"}.get(risk,"badge-thriving")
    return f"<span class='{css}' style='font-size:{font_size};'>{esc(RISK_LABELS.get(risk, risk))}</span>"


def section_divider(text):
    st.markdown(
        '<div class="sec-divider anim-up"><div class="sec-divider-line"></div>'
        f'<div class="sec-divider-text">{esc(text)}</div>'
        '<div class="sec-divider-line" style="background:linear-gradient(90deg,transparent,#f0dcc8);"></div>'
        '</div>', unsafe_allow_html=True)


def stat_pill(value, label, color=None, delay=""):
    border = f'border-top:4px solid {color};' if color else ""
    value_color = f'color:{color};' if color else ""
    st.markdown(f'<div class="stat-pill anim-up {delay}" style="{border}">'
                f'<div class="stat-num" style="{value_color}">{value}</div>'
                f'<div class="stat-lbl">{esc(label)}</div></div>', unsafe_allow_html=True)


def page_title(title, subtitle=""):
    st.markdown(f'<div class="anim-up"><div style="font-family:Playfair Display,serif;font-size:36px;'
                f'font-weight:900;color:#2d1a0e;margin-bottom:8px;">{esc(title)}</div>'
                f'<div style="color:#8a6a55;font-size:15px;margin-bottom:8px;">{subtitle}</div></div>',
                unsafe_allow_html=True)


def render_navbar():
    """Sticky top bar. Shows the counselor's unread alert count once signed in."""
    unread = 0
    if st.session_state.counselor_logged_in:
        from database import get_notifications
        notifs = get_notifications()
        if not notifs.empty and "read" in notifs.columns:
            unread = int((~notifs["read"].astype(bool)).sum())

    alert_html = ""
    if unread > 0:
        alert_html = ("<span style='background:#d4603a;color:white;font-size:11px;font-weight:700;padding:4px 10px;"
                      "border-radius:50px;margin-left:8px;animation:pulse-soft 2s infinite;'>" + str(unread) + " Alert</span>")

    if st.session_state.student_logged_in:
        who = ("<span style='color:#48b87a;font-size:13px;font-weight:600;background:#e8f7ef;padding:6px 14px;"
               "border-radius:50px;border:1.5px solid #a8dfc0;'>● "
               + esc(first_name(st.session_state.student_data.get("name",""))) + "</span>")
    else:
        who = "<span style='color:#b09070;font-size:13px;'>Not signed in</span>"

    st.markdown(
        "<div style='background:white;padding:16px 36px;display:flex;align-items:center;"
        "justify-content:space-between;border-bottom:1px solid #f0e6d8;"
        "box-shadow:0 2px 16px rgba(180,120,60,0.08);position:sticky;top:0;z-index:999;'>"
        "<div style='display:flex;align-items:center;gap:14px;'>"
        "<div style='width:38px;height:38px;background:linear-gradient(135deg,#d4603a,#f5a623);"
        "border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:18px;"
        "box-shadow:0 4px 12px rgba(212,96,58,0.3);'>🌿</div>"
        "<div><div style='color:#2d1a0e;font-family:Playfair Display,serif;font-weight:700;"
        "font-size:20px;line-height:1;'>Burnout Buster</div>"
        "<div style='color:#b09070;font-size:11px;letter-spacing:1px;font-weight:500;margin-top:1px;'>VIPS-TC Wellness</div>"
        "</div></div><div style='display:flex;align-items:center;gap:16px;'>"
        + alert_html + who +
        "<span style='color:#e8d5c0;font-size:20px;'>|</span>"
        "<span style='color:#b09070;font-size:13px;font-weight:500;'>Mohit Kumar</span>"
        "</div></div>", unsafe_allow_html=True)


def render_footer():
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
