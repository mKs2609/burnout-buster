# 🔥 Burnout Buster — Student Burnout Risk Prediction

> **AIDS 260 Practicum Project | VIPS-TC College of Engineering**  
> Team: Yash Choudhary · Mohit Kumar · Lakshay | Supervisor: Dr Sapna Yadav

---

## 📌 About

Burnout Buster is an ML-powered web app that predicts student burnout risk
(Low / Medium / High) based on 17 academic, social, lifestyle, and emotional
factors. It is designed to help counselors and institutions intervene **2 weeks
before** a student hits clinical burnout.

- **Model:** Random Forest Classifier (99%+ accuracy)  
- **Features:** 17 validated inputs  
- **Dataset:** 300 synthetic student records (based on real patterns)  
- **Interface:** Streamlit web app

---

## 🚀 Quick Start (Run Locally)

### Step 1 — Install Python
Make sure Python 3.9+ is installed. Download from https://python.org

### Step 2 — Install dependencies
Open a terminal in this folder and run:
```bash
pip install -r requirements.txt
```

### Step 3 — Generate the dataset
```bash
python generate_dataset.py
```
This creates `burnout_dataset.csv` with 300 student records.

### Step 4 — Train the model
```bash
python train_model.py
```
This creates `burnout_model.pkl`, `label_encoder.pkl`, and `model_meta.json`.

### Step 5 — Launch the app
```bash
streamlit run app.py
```
Open http://localhost:8501 in your browser. 🎉

---

## 📁 File Structure

```
burnout-buster/
├── app.py                  # Main Streamlit web app
├── generate_dataset.py     # Creates synthetic training dataset
├── train_model.py          # Trains the ML model + saves artefacts
├── burnout_dataset.csv     # 300-student dataset (auto-generated)
├── burnout_model.pkl       # Trained Random Forest model
├── label_encoder.pkl       # Label encoder (High/Low/Medium → numbers)
├── model_meta.json         # Model accuracy + feature importances
├── requirements.txt        # Python package list
└── README.md               # This file
```

---

## 🌐 Deploy Online (Free — Streamlit Cloud)

1. Create a free account at https://streamlit.io/cloud
2. Push this entire folder to a GitHub repository
3. On Streamlit Cloud, click **"New App"** → connect your GitHub repo
4. Set **main file path** to `app.py`
5. Click Deploy — you get a public URL in ~2 minutes!

---

## 📊 Features Tracked

| Category       | Features |
|----------------|----------|
| Academic       | Exams/month, Assignments/week, Attendance pressure, CGPA, Backlogs, Study hours |
| Social         | FOMO score, Peer pressure, Family expectations, Social media hours, Rejection sensitivity |
| Lifestyle      | Sleep hours, Exercise days, Diet quality |
| Emotional      | Self-confidence, Support system, Mental health visits |
| **Target**     | Burnout risk: Low / Medium / High |

---

## 🧠 Model Details

- **Algorithm:** Random Forest (300 trees, balanced class weights)
- **Accuracy:** 99%+ on balanced 300-student dataset
- **Validation:** 5-fold stratified cross-validation
- **Imbalance handling:** Equal class distribution (100 per class)

---

## 🌍 SDG Alignment

| SDG | Connection |
|-----|------------|
| SDG 3 – Good Health & Well-being | Early mental health detection |
| SDG 4 – Quality Education | Reducing dropout due to burnout |
| SDG 10 – Reduced Inequalities | Supporting vulnerable students |

---

## 📞 Crisis Resources (India)

| Helpline | Number |
|----------|--------|
| iCall (TISS) | 9152987821 |
| Vandrevala Foundation | 1860-2662-345 (24/7) |
| NIMHANS | 080-46110007 |
| Snehi | 044-24640050 |

---

*Made with ❤️ at VIPS-TC | AIDS-A Batch 2024*
