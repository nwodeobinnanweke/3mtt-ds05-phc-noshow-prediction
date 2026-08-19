# 3MTT NextGen Capstone Project: Primary Healthcare (PHC) No-Show Risk Prediction

**Track:** Data Science  
**Assigned Brief ID:** DS-05 — PHC No-Show Prediction  
**Fellow Name:** Nwode Obinna Nweke  
**Cohort:** 3MTT NextGen Fellow (Ebonyi State)  
**Submission Date:** August 2026  

---

## 1. Executive Summary & Operational Context
In Nigerian Primary Healthcare Centres (PHCs), unexpected patient absenteeism disrupts continuous primary care delivery, causing wasted clinical personnel hours and unutilized vaccine batches.

This project develops an end-to-end machine learning pipeline to estimate individual appointment no-show probabilities strictly at the time of booking. The predicted risk score serves as a triage mechanism for targeted administrative interventions (such as automated SMS reminders and Community Health Extension Worker [CHEW] outreach).

---

## 2. Simulation Cohort & Data Provenance
* **Cohort Design:** A synthetic simulation cohort of **3,000 patient appointments** modeled after primary care dynamics in Ebonyi State (*Abakaliki, Ebonyi, Izzi, Ikwo, Afikpo North*), complying with Option 3 of the 3MTT Data Sources Guide.
* **Methodological Framing:** Data were generated under defined probabilistic parameters to benchmark model performance, test calibration, and evaluate decision thresholds prior to field deployment.
* **Leakage-Safe Prediction Point:** Predictors are restricted entirely to features observable at the booking timestamp (`Age`, `Gender`, `LGA`, `ClinicType`, `Hypertension`, `Diabetes`, `Distance_KM`, `LeadTimeDays`, `PastNoShows`). Post-booking interventions (e.g., SMS delivery) are excluded from feature space $X$ to prevent circular target leakage.
* **Observed Baseline:** Overall cohort baseline no-show rate is **52.97%**.

---

## 3. Preprocessing & Cross-Validation Architecture
Data was split into an 80% training set (2,400 records) and a 20% stratified held-out test set (600 records). Preprocessing was isolated inside `scikit-learn` pipelines using `StandardScaler` for continuous numerical features and `OneHotEncoder(drop='first')` for categorical features.

### 5-Fold Stratified Cross-Validation ($N=2,400$)
| Model Architecture | Mean ROC-AUC (± SD) | Mean PR-AUC (± SD) | Mean F1-Score (± SD) |
| :--- | :---: | :---: | :---: |
| **Dummy Baseline (Stratified)** | 0.5119 ± 0.0174 | 0.5359 ± 0.0095 | 0.5383 ± 0.0165 |
| **Logistic Regression (Balanced)** | **0.6862 ± 0.0165** | **0.7157 ± 0.0249** | **0.6304 ± 0.0065** |
| **Random Forest (Balanced)** | 0.6789 ± 0.0158 | 0.7016 ± 0.0214 | 0.6270 ± 0.0104 |

---

## 4. Held-Out Test Evaluation ($N=600$)
Both candidate models were evaluated on the unseen 600-record test set:

| Evaluation Metric | Logistic Regression | Random Forest |
| :--- | :---: | :---: |
| **Test ROC-AUC** | **0.6832** | 0.6709 |
| **Test PR-AUC** | **0.7103** | 0.6804 |
| **Brier Score (Calibration)** | **0.2238** | 0.2293 |
| **Log-Loss** | **0.6378** | 0.6509 |

---

## 5. Operational Decision Threshold Optimization
In public healthcare operations, failing to flag a no-show (False Negative) has a higher societal cost than sending an unnecessary reminder (False Positive).

| Threshold | Precision | Recall | F1-Score | Operational Deployment Note |
| :---: | :---: | :---: | :---: | :--- |
| `0.30` | 0.5310 | 0.9969 | 0.6929 | Aggressive screening; high reminder volume. |
| `0.35` | 0.5391 | 0.9748 | 0.6943 | High sensitivity regime. |
| **`0.40`** | **0.5660** | **0.8899** | **0.6919** | **Recommended Operating Point:** Flags ~89.0% of all missed visits. |
| `0.45` | 0.6089 | 0.7296 | 0.6638 | Moderate balance point. |
| `0.50` | 0.6735 | 0.6164 | 0.6437 | Default uncalibrated threshold (misses ~38.4% of no-shows). |
| `0.60` | 0.7517 | 0.3522 | 0.4797 | Conservative; identifies only highest-confidence absentees. |

---

## 6. Test-Set Permutation Feature Importance
Permutation feature importance was evaluated on the held-out test set over 10 random shuffles:

| Feature Name | Mean ROC-AUC Drop | Standard Deviation |
| :--- | :---: | :---: |
| **`LeadTimeDays`** | **+0.0694** | 0.0124 |
| **`PastNoShows`** | **+0.0543** | 0.0136 |
| **`Distance_KM`** | **+0.0395** | 0.0122 |
| **`ClinicType`** | **+0.0070** | 0.0025 |
| **`Diabetes`** | +0.0005 | 0.0010 |
| **`Gender`** | -0.0001 | 0.0014 |
| **`Age`** | -0.0001 | 0.0039 |
| **`LGA`** | -0.0015 | 0.0025 |
| **`Hypertension`** | -0.0025 | 0.0026 |

---

## 7. Actionable Healthcare Delivery Recommendations
1. **Tiered Intervention Protocol:** Trigger automated SMS reminders and CHEW voice calls for patients with predicted no-show risk $\ge 0.40$.
2. **Scheduling Window Optimization:** Restrict elective follow-up scheduling to within 5 to 7 days of booking where feasible, as lead time is the single strongest determinant of attendance.
3. **Transport Barrier Mitigation:** Integrate community transport subsidies or localized outreach clinics for individuals residing $>5\text{ km}$ from the PHC.

---

## 8. Reproducibility
```bash
git clone [https://github.com/nwodeobinnanweke/3mtt-ds05-phc-noshow-prediction.git](https://github.com/nwodeobinnanweke/3mtt-ds05-phc-noshow-prediction.git)
cd 3mtt-ds05-phc-noshow-prediction
pip install -r requirements.txt
python -c "import urllib.request; exec(open('notebooks/PHC_NoShow_Pipeline.ipynb').read())"
