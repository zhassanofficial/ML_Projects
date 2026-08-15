# 📊 Telco Customer Churn Prediction

![Dashboard Preview](images/dashboard_preview.png)

## 🧠 Project Overview

This project aims to predict **customer churn** for a telecom company using the popular **Telco Customer Churn** dataset from Kaggle.  
We built and compared multiple machine learning models and deployed the best one as an interactive **Streamlit** dashboard.

**Key Business Goal:** Help retention teams identify high‑risk customers early and take proactive action.

---

## 📁 Dataset

- **Source:** [Kaggle – Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Rows:** 7,043 customers  
- **Features:** 20 (demographic, account, and service information)
- **Target:** `Churn` – Yes/No

---

## 🚀 Workflow

1. **Exploratory Data Analysis (EDA)**  
   - Distribution of churn, tenure, monthly charges, contract types, etc.  
   - Correlation analysis and feature interactions.

2. **Data Preprocessing**  
   - Fixed `TotalCharges` (object → float).  
   - One‑hot encoding for categorical variables.  
   - Standard scaling for numerical features.  
   - Handled class imbalance using `class_weight='balanced'`.

3. **Model Training & Tuning**  
   - Tested: Logistic Regression, Random Forest, XGBoost.  
   - Best model: **Logistic Regression** (tuned with GridSearchCV).  
   - Selected threshold: **0.60** (best F1‑score).

4. **Model Performance**  
   | Metric          | Score  |
   |-----------------|--------|
   | **ROC‑AUC**     | 0.844  |
   | **Recall**      | 0.80   |
   | **Precision**   | 0.51   |
   | **F1‑Score**    | 0.62   |
   - *Recall is prioritised because missing a churner costs more than false alarms.*

5. **Deployment**  
   - Built a **Streamlit** dashboard for real‑time prediction.  
   - Users can input customer details and instantly get churn probability and recommendations.

---

## 🖥️ Live Demo



---

## 🛠️ Tech Stack

- **Python** – Data processing & modeling  
- **Pandas / NumPy** – Data manipulation  
- **Matplotlib / Seaborn** – Visualization  
- **Scikit‑learn** – Preprocessing, modeling, tuning  
- **XGBoost** – Additional model testing  
- **Streamlit** – Interactive dashboard  
- **Joblib** – Model serialization  

---

