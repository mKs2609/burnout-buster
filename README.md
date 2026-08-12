# 🔥 Burnout Buster — Student Burnout Risk Prediction

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B)](https://streamlit.io)
[![Model](https://img.shields.io/badge/Model-Random%20Forest-brightgreen)]()
[![Live Demo](https://img.shields.io/badge/Demo-Live-success)](https://burnout-buster-tp2wbhw5ctpsd3ggy8yzlc.streamlit.app/)
[![Status](https://img.shields.io/badge/Status-Prototype-orange)]()

An ML-powered web app that predicts a student's burnout risk (**Low / Medium / High**) from 17 academic, social, lifestyle, and emotional signals — built to help counselors and institutions intervene *before* a student hits clinical burnout.

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
- [Deployment](#deployment)
- [SDG Alignment](#sdg-alignment)
- [Crisis Resources (India)](#crisis-resources-india)
- [Roadmap](#roadmap)
- [Disclaimer](#disclaimer)
- [Author](#author)

---

## Overview

Burnout Buster takes 17 validated inputs spanning academics, social pressure, lifestyle, and emotional wellbeing, and classifies a student's burnout risk using a Random Forest model trained on a synthetic (pattern-informed) dataset of 300 records. The goal is early detection — flagging risk roughly **two weeks ahead** of clinical burnout so counselors can step in sooner.

- **Model:** Random Forest Classifier
- **Inputs:** 17 features across 4 categories
- **Dataset:** 300 synthetic student records
- **Interface:** Streamlit web app
- **Validation:** 5-fold stratified cross-validation

## Key Highlights

- ✅ **End-to-end ML pipeline** — from synthetic data generation to model training to a deployed, interactive web app
- ✅ **Live, publicly deployed demo** — try it without installing anything
- ✅ **Domain-informed feature engineering** — 17 features grounded in academic, social, lifestyle, and emotional wellbeing research
- ✅ **Reproducible workflow** — dataset generation and model training are scripted and re-runnable from scratch
- ✅ **Social-impact framing** — aligned with UN Sustainable Development Goals and paired with real crisis-support resources

## Tech Stack

| Layer            | Tool                                |
| ----------------- | ------------------------------------ |
| Language          | Python 3.9+                          |
| ML                | scikit-learn (Random Forest)         |
| Data              | pandas / synthetic data generator    |
| App / UI          | Streamlit                            |
| Persistence       | `.pkl` model + label encoder         |
| Dev environment   | `.devcontainer` (Codespaces-ready)   |

## Quick Start

### 1. Install Python
Make sure Python 3.9+ is installed — [python.org](https://python.org)

### 2. Clone and install dependencies
```bash
git clone https://github.com/mKs2609/burnout-buster.git
cd burnout-buster
pip install -r requirements.txt
```

### 3. Generate the dataset
```bash
python generate_dataset.py
```
Creates `burnout_dataset.csv` with 300 student records.

### 4. Train the model
```bash
python train_model.py
```
Creates `burnout_model.pkl`, `label_encoder.pkl`, and `model_meta.json`.

### 5. Launch the app
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser. 🎉

## Project Structure

```
burnout-buster/
├── app.py                  # Main Streamlit web app
├── generate_dataset.py     # Creates synthetic training dataset
├── train_model.py          # Trains the ML model + saves artefacts
├── database.py             # Data handling / persistence layer
├── burnout_dataset.csv     # 300-student dataset (auto-generated)
├── burnout_model.pkl       # Trained Random Forest model
├── label_encoder.pkl       # Label encoder (High/Low/Medium → numbers)
├── model_meta.json         # Model accuracy + feature importances
├── requirements.txt        # Python package list
├── .devcontainer/          # Codespaces / dev container config
└── README.md
```

## How It Works

1. `generate_dataset.py` synthesizes a labeled dataset of student profiles across the 17 tracked features.
2. `train_model.py` trains a Random Forest classifier on that dataset, evaluates it with 5-fold stratified cross-validation, and serializes the model, label encoder, and metadata.
3. `app.py` loads the trained artifacts and serves a Streamlit form where a user enters their own values across the 17 inputs; the model returns a Low / Medium / High risk classification.
4. `database.py` handles storing and retrieving submissions from the app.

## Model Details

- **Algorithm:** Random Forest (300 trees, balanced class weights)
- **Reported accuracy:** 99%+ on the 300-student synthetic dataset
- **Validation:** 5-fold stratified cross-validation
- **Class balance:** Equal distribution (100 records per class)

## Features Tracked

| Category   | Features                                                                                   |
| ---------- | -------------------------------------------------------------------------------------------- |
| Academic   | Exams/month, Assignments/week, Attendance pressure, CGPA, Backlogs, Study hours              |
| Social     | FOMO score, Peer pressure, Family expectations, Social media hours, Rejection sensitivity    |
| Lifestyle  | Sleep hours, Exercise days, Diet quality                                                     |
| Emotional  | Self-confidence, Support system, Mental health visits                                        |
| **Target** | Burnout risk: Low / Medium / High                                                            |

## Deployment

Live app deployed via Streamlit Community Cloud:

**🔗 [burnout-buster-tp2wbhw5ctpsd3ggy8yzlc.streamlit.app](https://burnout-buster-tp2wbhw5ctpsd3ggy8yzlc.streamlit.app/)**

To deploy your own copy:

1. Create a free account at [streamlit.io/cloud](https://streamlit.io/cloud)
2. Push this repo to GitHub
3. On Streamlit Cloud, click **"New App"** → connect your repo
4. Set the main file path to `app.py`
5. Deploy — you'll get a public URL in ~2 minutes

## SDG Alignment

| SDG                                | Connection                        |
| ----------------------------------- | ---------------------------------- |
| SDG 3 — Good Health & Well-being    | Early mental health detection      |
| SDG 4 — Quality Education           | Reducing dropout due to burnout    |
| SDG 10 — Reduced Inequalities       | Supporting vulnerable students     |

## Crisis Resources (India)

If you or someone you know is struggling, help is available:

| Helpline               | Number                |
| ------------------------ | ------------------------ |
| iCall (TISS)              | 9152987821                |
| Vandrevala Foundation     | 1860-2662-345 (24/7)      |
| NIMHANS                   | 080-46110007               |
| Snehi                     | 044-24640050                |

## Roadmap

- [ ] Add unit tests for the training pipeline and app logic
- [ ] Validate the model against real (anonymized) student data
- [ ] Add confidence scores / explainability (e.g. SHAP) to predictions
- [ ] Add screenshots / GIF walkthrough of the app
- [ ] Add a license file

## Disclaimer

This tool is a prototype trained on **synthetic data** and is intended for educational and demonstration purposes. It is **not a clinical diagnostic tool** and should not replace professional mental health evaluation. If you are in crisis, please contact one of the helplines above or your local emergency services.

## Author

**Mohit Kumar**
AIDS-A, Batch 2024 — VIPS-TC College of Engineering
GitHub: [@mKs2609](https://github.com/mKs2609)

---

*Made with ❤️ at VIPS-TC | AIDS-A Batch 2024*
