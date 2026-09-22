# 🔥 Burnout Buster — Student Burnout Risk Screening

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B)](https://streamlit.io)
[![Model](https://img.shields.io/badge/Model-Ordinal%20Monotonic%20Boosting-brightgreen)]()
[![Tests](https://img.shields.io/badge/Tests-72%20passing-success)]()
[![Live Demo](https://img.shields.io/badge/Demo-Live-success)](https://burnout-buster-tp2wbhw5ctpsd3ggy8yzlc.streamlit.app/)
[![Status](https://img.shields.io/badge/Status-Prototype-orange)]()

A web app where students take a 3-minute check-in and get a **0–100 wellness score with the reasons behind it**, and counselors get a dashboard that flags who needs support — so someone can step in *before* a student hits clinical burnout.

**🔗 Live App:** [burnout-buster-tp2wbhw5ctpsd3ggy8yzlc.streamlit.app](https://burnout-buster-tp2wbhw5ctpsd3ggy8yzlc.streamlit.app/)

> Built at VIPS-TC College of Engineering by **Mohit Kumar** (AIDS-A, Batch 2024)

---

## Table of Contents

- [Overview](#overview)
- [Key Highlights](#key-highlights)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Model Details](#model-details)
- [Features Tracked](#features-tracked)
- [Storage](#storage)
- [Security & Privacy](#security--privacy)
- [Tests](#tests)
- [Deployment](#deployment)
- [SDG Alignment](#sdg-alignment)
- [Crisis Resources (India)](#crisis-resources-india)
- [Roadmap](#roadmap)
- [Disclaimer](#disclaimer)
- [Author](#author)

---

## Overview

Students answer 17 questions across academics, social pressure, lifestyle and emotional
wellbeing. The app returns a 0–100 score, the band it falls in, **which answers pushed it
up or down**, and a matching action plan. Counselors see alerts, a filterable student list,
private replies, academic-record cross-checks and institution-wide analytics.

- **Model:** ordinal monotonic gradient boosting (two binary boosters)
- **Inputs:** 17 features across 4 categories
- **Dataset:** 5,000 simulated responses (see [Model Details](#model-details))
- **Interface:** Streamlit web app with interactive Plotly charts
- **Storage:** SQL via SQLAlchemy (SQLite locally, PostgreSQL in production)
- **Validation:** held-out test set + 5-fold cross-validation + automated sanity checks

**Score bands:** 0–33 Thriving · 34–66 Needs Attention · 67–100 At Risk.
The band is the single source of truth — students and counselors always see the same level.

## Key Highlights

- ✅ **Explains every score** — each result names the answers raising and lowering it, compared with a typical thriving student
- ✅ **Common-sense guarantees** — monotonic constraints mean more sleep or support can *never* raise a score, and more backlogs can *never* lower one
- ✅ **Safety rules** — severe answers (≤ 4 h sleep, ≥ 4 backlogs, very low support) can only raise a score, and the student is told which rule applied
- ✅ **Honest evaluation** — ~83% accuracy on 1,000 held-out students, benchmarked against baseline models, with a model card inside the app
- ✅ **Real persistence** — SQL storage that survives restarts, with bcrypt-hashed passwords
- ✅ **Tested** — 72 pytest tests covering the model's guarantees, storage and end-to-end UI flows
- ✅ **Live, publicly deployed demo** — try it without installing anything
- ✅ **Social-impact framing** — aligned with UN Sustainable Development Goals and paired with real crisis-support resources

## Tech Stack

| Layer | Tool |
| --- | --- |
| Language | Python 3.9+ |
| ML | scikit-learn (`HistGradientBoostingClassifier`, monotonic constraints) |
| Data | pandas / NumPy / simulated data generator |
| App / UI | Streamlit + Plotly |
| Storage | SQLAlchemy — SQLite or PostgreSQL |
| Security | bcrypt password hashing |
| Tests | pytest (72 tests) |
| Dev environment | `.devcontainer` (Codespaces-ready) |

## Quick Start

### 1. Install Python
Python 3.9+ — [python.org](https://python.org)

### 2. Clone and install dependencies
```bash
git clone https://github.com/mKs2609/burnout-buster.git
cd burnout-buster
pip install -r requirements.txt
```

### 3. Add your secrets
Copy [`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example) to `.streamlit/secrets.toml` and set a counselor password (top-level keys must stay above any `[section]`):

```toml
COUNSELOR_PASSWORD = "choose-a-strong-password"
```

### 4. Generate the dataset
```bash
python generate_dataset.py
```
Creates `burnout_dataset.csv` with 5,000 simulated responses.

### 5. Train the model
```bash
python train_model.py
```
Creates `burnout_model.pkl` and `model_meta.json`, prints the full evaluation, and aborts if a sanity check fails.

### 6. Launch the app
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501). Database tables are created automatically. 🎉

Coming from the old CSV version? `python migrate_csv_to_db.py` imports `local_*.csv` into the database.

## Project Structure

```
burnout-buster/
├── app.py                  # Page shell + tab routing
├── views/                  # One module per tab
│   ├── home.py             #   what the tool is, score bands
│   ├── survey.py           #   questionnaire, scoring, results
│   ├── portal.py           #   student history + counselor messages
│   ├── counselor.py        #   alerts, student list, actions, records
│   ├── analytics.py        #   institution-wide charts
│   └── model_card.py       #   metrics, limits, training data
├── scoring.py              # Model class, safety rules, explanations
├── database.py             # SQLAlchemy schema + all data access
├── charts.py               # Interactive Plotly charts
├── ui.py                   # Stylesheet, navbar, footer, markup helpers
├── utils.py                # Check-in dates, latest-per-student, trajectory
├── advice.py               # Action-plan tips
├── constants.py            # Branches, sections, labels, palette
├── model_loader.py         # Cached model loading
├── generate_dataset.py     # Builds the simulated training set
├── train_model.py          # Trains, evaluates, writes the model card
├── migrate_csv_to_db.py    # One-time CSV → database import
├── tests/                  # 72 pytest tests
├── .streamlit/             # Theme + secrets template
├── .devcontainer/          # Codespaces / dev container config
└── README.md
```

## How It Works

1. `generate_dataset.py` simulates 5,000 student profiles across the 17 features.
2. `train_model.py` trains the ordinal model, compares it with baselines, runs sanity checks, and writes the model plus its model card.
3. `app.py` and `views/` serve the survey; `scoring.py` turns answers into a score, a band, the factors behind it and any safety rule.
4. `database.py` stores students, submissions, counselor actions, replies, alerts and uploaded academic records.

## Model Details

Burnout risk is **ordered** (Low < Medium < High), so instead of one multiclass model,
two binary gradient-boosting models estimate **P(≥ Needs Attention)** and **P(At Risk)**
(the Frank & Hall ordinal approach). Each carries monotonic constraints — which multiclass
boosting cannot express.

- **Score** = 50·P(≥ Needs Attention) + 50·P(At Risk), giving a full 0–100 range
- **Guarantees:** more sleep, support, exercise or CGPA can never raise a score; more backlogs, FOMO or social media can never lower one
- **Safety rules:** any severe answer forces at least *Needs Attention*; three or more force *At Risk*
- **Explanations:** each answer is compared with a typical thriving student's, in log-odds so factors still rank when the score saturates at 0 or 100
- **Accuracy:** ~83% on 1,000 held-out students, macro-F1 ~82%, **0.3% Low↔High mix-ups**
- **Benchmarked against:** majority-class baseline, logistic regression, random forest, plain gradient boosting (all reported in the app's *About the Model* tab)
- **Sanity checks:** 8 automated checks (monotonicity, 3 h sleep is never "Thriving", worst answers → At Risk …). Training refuses to save a model that fails one.

**The training data is simulated.** No real student records were available, so
`generate_dataset.py` builds 5,000 responses driven by hidden strain and workload factors
(so answers correlate as real ones do), a noisy hidden burnout index (so classes overlap),
and 5% careless responders (flat mid-scale answers, as in real survey data). Swap in real,
consented responses with counselor-confirmed levels and `train_model.py` works unchanged.

## Features Tracked

| Category | Features |
| --- | --- |
| Academic | Exams/month, Assignments/week, Attendance pressure, CGPA, Backlogs, Study hours |
| Social | FOMO score, Peer pressure, Family expectations, Social media hours, Rejection sensitivity |
| Lifestyle | Sleep hours, Exercise days, Diet quality |
| Emotional | Self-confidence, Support system, Counselor visits |
| **Target** | Burnout risk: Low / Medium / High |

## Storage

Tables (`students`, `submissions`, `replies`, `counselor_actions`, `reminders`,
`college_records`, `notifications`) are created on first run.

| Setup | When to use |
| --- | --- |
| **SQLite** (default, no config) — `burnout.db` | Local development and demos |
| **PostgreSQL** — set `DATABASE_URL` (e.g. free [Supabase](https://supabase.com) or [Neon](https://neon.tech)) | Any real deployment. Streamlit Cloud wipes local files on restart, so SQLite data would be lost there |

## Security & Privacy

- Passwords hashed with **bcrypt** (per-user salt); pre-bcrypt accounts upgrade on next login
- Minimum 8-character passwords; counselor password read from secrets only
- Portal login requires roll number + password + branch + section; one profile per roll number
- All student-supplied text is escaped before rendering
- `.gitignore` keeps secrets, the database and student CSVs out of the repository

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

72 tests covering the model's guarantees (monotonicity, safety rules, explanations), the
storage layer (accounts, bcrypt upgrades, alerts, failure handling) and end-to-end UI flows
(registration, impersonation attempts, counselor actions, HTML-injection attempts,
reminders). They run against a temporary database and never touch `burnout.db`.

## Deployment

Live app deployed via Streamlit Community Cloud:

**🔗 [burnout-buster-tp2wbhw5ctpsd3ggy8yzlc.streamlit.app](https://burnout-buster-tp2wbhw5ctpsd3ggy8yzlc.streamlit.app/)**

To deploy your own copy:

1. Create a free account at [streamlit.io/cloud](https://streamlit.io/cloud)
2. Push this repo to GitHub
3. On Streamlit Cloud, click **"New App"** → connect your repo, main file `app.py`
4. In **Settings → Secrets**, add `COUNSELOR_PASSWORD`, and `DATABASE_URL` if you want data to survive restarts
5. Deploy — you'll get a public URL in ~2 minutes

## SDG Alignment

| SDG | Connection |
| --- | --- |
| SDG 3 — Good Health & Well-being | Early mental health detection |
| SDG 4 — Quality Education | Reducing dropout due to burnout |
| SDG 10 — Reduced Inequalities | Supporting vulnerable students |

## Crisis Resources (India)

If you or someone you know is struggling, help is available:

| Helpline | Number |
| --- | --- |
| iCall (TISS) | 9152987821 |
| Vandrevala Foundation | 1860-2662-345 (24/7) |
| NIMHANS | 080-46110007 |
| Snehi | 044-24640050 |

## Roadmap

- [x] Add unit tests for the training pipeline and app logic
- [x] Add explainability to predictions (per-answer factors behind every score)
- [ ] Validate the model against real (anonymized) student data from a consented pilot
- [ ] Counselor accounts instead of one shared password
- [ ] Email / WhatsApp notifications for counselors
- [ ] Add screenshots / GIF walkthrough of the app
- [ ] Add a license file

## Disclaimer

This tool is a prototype trained on **simulated data** and is intended for educational and
demonstration purposes. It is a **screening aid, not a clinical diagnostic tool**, and should
never replace professional mental health evaluation. If you are in crisis, please contact one
of the helplines above or your local emergency services.

## Author

**Mohit Kumar**
AIDS-A, Batch 2024 — VIPS-TC College of Engineering
GitHub: [@mKs2609](https://github.com/mKs2609)

---

*Made with ❤️ at VIPS-TC | AIDS-A Batch 2024*
