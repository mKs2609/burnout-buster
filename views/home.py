"""Home tab — what the tool is and how the score works."""
import streamlit as st
import pandas as pd

def render():
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
          A 3-minute check-in that gives you a personal wellness score, shows what's
          driving it, and connects you with your counselor — before things get hard.
        </div>
        <div style="display:flex;gap:12px;flex-wrap:wrap;">
          <div style="background:rgba(255,255,255,0.1);backdrop-filter:blur(10px);border:1px solid rgba(255,255,255,0.15);
          border-radius:50px;padding:10px 20px;color:white;font-size:13px;font-weight:600;">
            ML-Powered Prediction
          </div>
          <div style="background:rgba(72,184,122,0.2);border:1px solid rgba(72,184,122,0.4);
          border-radius:50px;padding:10px 20px;color:#7ddaa8;font-size:13px;font-weight:600;">
            Seen only by your counselor
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
      <div class="sec-divider-text">How It Works</div>
      <div class="sec-divider-line" style="background:linear-gradient(90deg,transparent,#f0dcc8);"></div>
    </div>""", unsafe_allow_html=True)

    s1,s2,s3,s4 = st.columns(4)
    # Only facts about the tool itself — no unsourced statistics
    stats = [("17","questions across academics, social life, lifestyle and wellbeing"),
             ("3 min","to complete — answer honestly for the most useful result"),
             ("Top 3","factors shown with every score, so you know what to work on"),
             ("1:1","your counselor follows up privately if you need support")]
    for col,(num,desc) in zip([s1,s2,s3,s4],stats):
        with col:
            st.markdown(f"""
            <div class="stat-pill anim-up">
              <div style="font-family:Playfair Display,serif;font-size:32px;
              font-weight:900;color:#d4603a;">{num}</div>
              <div style="color:#8a6a55;font-size:12px;line-height:1.5;margin-top:6px;">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
