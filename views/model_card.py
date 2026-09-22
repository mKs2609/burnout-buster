"""About the Model tab — model card, metrics and training data."""
import streamlit as st
import pandas as pd
from constants import RISK_LABELS
from scoring import LEVELS, SEVERE_SIGNALS
from model_loader import load_model

def render():
    model, meta = load_model()
    st.markdown('<div class="page">', unsafe_allow_html=True)
    st.markdown("""
    <div class="anim-up">
      <div style="font-family:Playfair Display,serif;font-size:36px;font-weight:900;
      color:#2d1a0e;margin-bottom:8px;">About the Model</div>
      <div style="color:#8a6a55;font-size:15px;">How scores are calculated, how well the model performs, and its limits.</div>
    </div>""", unsafe_allow_html=True)

    st.warning("**Trained on simulated data.** No real student records were available, so the model learned from "
               f"{meta.get('n_samples','')} simulated survey responses built to behave like real ones — correlated answers, "
               "overlapping risk levels and 5% careless responders (see `generate_dataset.py`). "
               "The numbers below describe performance on that data. Scores are a screening aid for counselors, "
               "not a diagnosis — retrain on real, consented survey data before relying on them.")

    k1,k2,k3,k4 = st.columns(4)
    for col,(num,lbl) in zip([k1,k2,k3,k4],[
            (f"{meta.get('accuracy','–')}%","Accuracy (held-out)"),
            (f"{meta.get('f1_macro','–')}%","Macro F1"),
            (f"{meta.get('low_high_confusion_pct','–')}%","Low ↔ High mix-ups"),
            (str(meta.get('n_test','–')),"Test students")]):
        with col:
            st.markdown('<div class="stat-pill anim-up"><div class="stat-num" style="font-size:34px;">' + num
                        + '</div><div class="stat-lbl">' + lbl + '</div></div>', unsafe_allow_html=True)
    st.markdown("")

    mc1,mc2 = st.columns(2)
    with mc1:
        st.markdown("**How the score works**")
        st.markdown(
            "- Two gradient-boosting models estimate *P(at least Needs Attention)* and *P(At Risk)*.\n"
            "- **Score = 50 × P(≥ Needs Attention) + 50 × P(At Risk)**, from 0 to 100.\n"
            "- Bands: 0–33 Thriving · 34–66 Needs Attention · 67–100 At Risk.\n"
            "- **Monotonic by design** — e.g. more sleep or support can never raise a score; more backlogs can never lower it.\n"
            "- Each result lists the answers raising or lowering it, compared with a typical thriving student.")
        st.markdown("**Safety rules** — can only raise a score:")
        st.markdown("\n".join(f"- {text}" for _,_,text in SEVERE_SIGNALS)
                    + "\n\nAny one → at least *Needs Attention*; three or more → *At Risk*.")
    with mc2:
        if meta.get("confusion_matrix"):
            st.markdown("**Confusion matrix** (held-out test set)")
            names = [RISK_LABELS[l] for l in LEVELS]
            st.dataframe(pd.DataFrame(meta["confusion_matrix"],
                                      index=["Actual: "+n for n in names],
                                      columns=["Predicted: "+n for n in names]),
                         use_container_width=True)
        if meta.get("comparison"):
            st.markdown("**Compared with other models** (5-fold cross-validation)")
            st.dataframe(pd.DataFrame(meta["comparison"]).T.rename(columns={
                             "cv_accuracy":"Accuracy %","cv_accuracy_std":"± std","cv_f1_macro":"Macro F1 %"}),
                         use_container_width=True)
            best = max(meta["comparison"], key=lambda k: meta["comparison"][k]["cv_accuracy"])
            if best != meta.get("model"):
                st.caption(f"{best} scores slightly higher here. The {meta.get('model','chosen model').lower()} "
                           "is used for its common-sense guarantees and per-student explanations.")
        if meta.get("sanity_checks"):
            st.markdown("**Sanity checks** (training fails if any break)")
            st.markdown("\n".join(f"- {'✅' if ok else '❌'} {name}" for name,ok in meta["sanity_checks"].items()))

    st.markdown("""
    <div class="sec-divider"><div class="sec-divider-line"></div>
    <div class="sec-divider-text">Training Data</div>
    <div class="sec-divider-line" style="background:linear-gradient(90deg,transparent,#f0dcc8);"></div>
    </div>""", unsafe_allow_html=True)
    try:
        df_t = pd.read_csv("burnout_dataset.csv")
        m1,m2,m3,m4 = st.columns(4)
        with m1: st.markdown('<div class="stat-pill anim-up"><div class="stat-num">' + str(len(df_t)) + '</div><div class="stat-lbl">Records</div></div>', unsafe_allow_html=True)
        with m2: st.markdown('<div class="stat-pill anim-up d1" style="border-top:4px solid #d4603a;"><div class="stat-num" style="color:#d4603a;">' + str(int((df_t["burnout_risk"]=="High").sum())) + '</div><div class="stat-lbl">At Risk</div></div>', unsafe_allow_html=True)
        with m3: st.markdown('<div class="stat-pill anim-up d2" style="border-top:4px solid #f5a623;"><div class="stat-num" style="color:#f5a623;">' + str(int((df_t["burnout_risk"]=="Medium").sum())) + '</div><div class="stat-lbl">Needs Attention</div></div>', unsafe_allow_html=True)
        with m4: st.markdown('<div class="stat-pill anim-up d3" style="border-top:4px solid #48b87a;"><div class="stat-num" style="color:#48b87a;">' + str(int((df_t["burnout_risk"]=="Low").sum())) + '</div><div class="stat-lbl">Thriving</div></div>', unsafe_allow_html=True)
        st.markdown("")
        with st.expander(f"Browse the simulated training data ({len(df_t):,} rows)"):
            st.caption("Simulated responses — not real students. Published so the model's claims can be checked.")
            rf3 = st.multiselect("Filter",["High","Medium","Low"],default=["High","Medium","Low"],
                                 format_func=lambda r: RISK_LABELS[r])
            st.dataframe(df_t[df_t["burnout_risk"].isin(rf3)],use_container_width=True,height=420)
            st.download_button("Download Dataset",data=df_t.to_csv(index=False).encode(),
                               file_name="burnout_dataset.csv",mime="text/csv")
    except FileNotFoundError:
        st.error("Run `python generate_dataset.py` then `python train_model.py` first.")
    st.markdown('</div>', unsafe_allow_html=True)
