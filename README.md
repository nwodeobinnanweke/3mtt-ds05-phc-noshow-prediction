# 3MTT NextGen Capstone Project: Primary Healthcare (PHC) Appointment No-Show Prediction Pipeline

**Track:** Data Science  
**Brief ID:** DS-05 — PHC No-Show Prediction  
**Fellow Name:** Nwode Obinna Nweke  
**Cohort:** 3MTT NextGen Fellow (Ebonyi State)  
**Submission Date:** August 2026  

---

## 1. Project Overview & Clinical Dilemma
In Nigerian Primary Healthcare Centres (PHCs), unpredictable patient absenteeism leads to severe operational inefficiencies:
* **Vaccine Wastage:** Multi-dose vials (e.g., measles, BCG) reconstituted in anticipation of scheduled sessions must be discarded after 6 hours if patient turnout is insufficient.
* **Resource Strain:** Frontline personnel cannot afford universal phone or home-visit follow-up for every registered patient.

This project delivers a machine-learning decision-support pipeline that estimates an individual patient's default probability strictly at the time of appointment booking. These risk scores enable facility administrators to rationally triage outreach interventions (such as automated SMS reminders and Community Health Extension Worker [CHEW] engagement).

---

## 2. Simulation Cohort & Leakage-Safe Design
* **Methodological Framing:** In compliance with Option 3 of the 3MTT Data Sources Guide, a synthetic simulation cohort of **3,000 patient records** was generated to benchmark model architectures and operational workflows without violating patient data privacy regulations (NDPR).
* **Leakage Prevention:** To avoid circular target leakage, post-booking variables (such as reminder delivery) are excluded from the feature space $X$. The feature set is strictly restricted to variables available at scheduling: `Age`, `Gender`, `LGA`, `ClinicType`, `Hypertension`, `Diabetes`, `Distance_KM`, `LeadTimeDays`, and `PastNoShows`.
* **Cohort Characteristics:** Modeled across Ebonyi State LGAs (*Abakaliki, Ebonyi, Izzi, Ikwo, Afikpo North*), exhibiting an overall baseline no-show rate of **52.97%**.

---

## 3. Experimental Evaluation & Model Comparison
Models were developed inside `scikit-learn` pipelines with isolated preprocessing (`StandardScaler` and `OneHotEncoder`). Evaluation was conducted across 5-fold stratified cross-validation ($N=2,400$) and a held-out test set ($N=600$):

### 5-Fold Stratified Cross-Validation
| Architecture | Mean ROC-AUC (± SD) | Mean PR-AUC (± SD) | Mean F1-Score (± SD) |
| :--- | :---: | :---: | :---: |
| **Dummy Baseline (Stratified)** | 0.5119 ± 0.0174 | 0.5359 ± 0.0095 | 0.5383 ± 0.0165 |
| **Random Forest (Balanced)** | 0.6789 ± 0.0158 | 0.7016 ± 0.0214 | 0.6270 ± 0.0104 |
| **Logistic Regression (Balanced)** | **0.6862 ± 0.0165** | **0.7157 ± 0.0249** | **0.6304 ± 0.0065** |

### Held-Out Test Performance ($N=600$)
| Evaluation Metric | Logistic Regression (Selected) | Random Forest |
| :--- | :---: | :---: |
| **Test ROC-AUC** | **0.6832** | 0.6709 |
| **Test PR-AUC** | **0.7103** | 0.6804 |
| **Brier Score (Calibration)** | **0.2238** | 0.2293 |
| **Log-Loss** | **0.6378** | 0.6509 |

**Model Selection:** Balanced Logistic Regression is designated as the primary model due to superior discrimination (ROC-AUC 0.6832), superior probability calibration (Brier score 0.2238), and transparent coefficient interpretability.

---

## 4. Asymmetric Cost Analysis & Threshold Optimization
In public healthcare triage, classification thresholds must reflect operational trade-offs rather than defaulting to an arbitrary 0.50 cutoff:
* **False Negative (High Cost):** An impending absentee is missed, leading to wasted clinical slots and spoiled vaccine vials.
* **False Positive (Low Cost):** A compliant patient receives an unnecessary automated reminder.

### Logistic Regression Decision Thresholds (Held-Out Test Set)
| Threshold | Precision | Recall (Sensitivity) | F1-Score | Operational Context |
| :---: | :---: | :---: | :---: | :--- |
| `0.30` | 0.5310 | 0.9969 | 0.6929 | Broad screening regime |
| **`0.40`** | **0.5660** | **0.8899** | **0.6919** | **Recommended Operating Point:** Identifies ~89.0% of all absentees |
| `0.50` | 0.6735 | 0.6164 | 0.6437 | Conventional uncalibrated cutoff (misses ~38.4% of no-shows) |
| `0.60` | 0.7517 | 0.3522 | 0.4797 | Conservative; targets only highest-risk cases |

---

## 5. Permutation Feature Importance
Permutation feature importance computed on the held-out test set indicates that scheduling dynamics and behavioral history dominate predictive importance:

| Feature Name | Mean ROC-AUC Drop | Standard Deviation |
| :--- | :---: | :---: |
| **`LeadTimeDays`** | **+0.0694** | 0.0124 |
| **`PastNoShows`** | **+0.0543** | 0.0136 |
| **`Distance_KM`** | **+0.0395** | 0.0122 |
| **`ClinicType`** | **+0.0070** | 0.0025 |

---

## 6. Methodological Limitations & Future Scope
* **Simulation Scope:** This study benchmarks an end-to-end machine learning methodology on synthetic data. These predictive associations demonstrate pipeline efficacy and must not be interpreted as proven clinical causation in Ebonyi State.
* **Prospective Validation:** Real-world facility deployment requires integration into primary care EMR platforms, prospective clinical validation, and health governance oversight.

---

## 7. How to Reproduce and Run

### Run the Interactive Streamlit Decision Prototype
```bash
git clone [https://github.com/nwodeobinnanweke/3mtt-ds05-phc-noshow-prediction.git](https://github.com/nwodeobinnanweke/3mtt-ds05-phc-noshow-prediction.git)
cd 3mtt-ds05-phc-noshow-prediction
pip install -r requirements.txt
streamlit run app.py
