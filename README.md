# 📊 End-to-End Customer Churn Prediction & Behavioral Segmentation Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-teal.svg)](https://fastapi.tiangolo.com/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-darkblue.svg)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An enterprise-grade, end-to-end machine learning system designed to predict customer churn, uncover distinct customer behavioral personas, and serve real-time predictions via a containerized **FastAPI** microservice.

---

## 📌 Project Overview
Customer acquisition costs between **5x to 7x more** than customer retention. In subscription-based businesses (Telecommunications, SaaS), customer churn represents immediate revenue loss. Traditional retention strategies operate reactively after an account cancellation request is initiated. 

This project delivers a **proactive, predictive retention engine** that:
1. Identifies at-risk customers 30 to 60 days before churn.
2. Segments accounts into distinct behavioral archetypes using unsupervised clustering.
3. Automatically triggers targeted retention offers through a sub-100ms REST API.

---

## 🏗️ System Architecture

```text
Raw CSV Ingestion (7,043 Records)
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ 1. Data Cleaning & Preprocessing (Pandas / NumPy)      │
│    • TotalCharges whitespace coercion & median fill    │
│    • customerID noise pruning                          │
│    • One-Hot Encoding (drop_first=True)                │
│    • Z-Score Normalization (StandardScaler)            │
└──────────────────────────┬─────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────┐
│ 2. Supervised Modeling & Hyperparameter Tuning         │
│    • Evaluated: Logistic Regression, Trees, RF, KNN    │
│    • Champion: Random Forest tuned with GridSearchCV   │
│    • Metric: Recall: 76.2% | ROC-AUC: 0.85             │
└──────────────────────────┬─────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────┐
│ 3. Unsupervised Behavioral Clustering                  │
│    • PCA 2D Dimensionality Reduction                   │
│    • K-Means (K=3) & Hierarchical Dendrograms          │
│    • Customer Personas: At-Risk, Champions, Budget     │
└──────────────────────────┬─────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────┐
│ 4. Model Serialization & FastAPI Deployment            │
│    • Persistent unified Joblib binary artifact         │
│    • Pydantic schema validation                        │
│    • Endpoints: GET /health | POST /predict (<100ms)   │
└────────────────────────────────────────────────────────┘
```

---

## 🗓️ 4-Week Milestone Breakdown

### 🔹 Week 1: Data Preprocessing & Exploratory Data Analysis (EDA)
- **Data Hygiene**: Handled 11 hidden whitespace missing values in `TotalCharges` via median imputation (\$1,397.47).
- **Feature Selection**: Pruned high-cardinality `customerID` (7,043 unique alphanumeric strings).
- **Categorical Encoding**: Converted binary attributes and applied One-Hot Encoding to multi-category variables (`Contract`, `PaymentMethod`), generating 31 numeric features.
- **Scaling**: Standardized continuous features (`tenure`, `MonthlyCharges`, `TotalCharges`) to $\mu=0.00, \sigma=1.00$.
- **Key EDA Finding**: Churners show a median tenure of **10.0 months** vs. **38.0 months** for retained accounts.

### 🔹 Week 2: Supervised Model Benchmarking
- **Stratified Split**: 80/20 train/test split preserving the 73.5% / 26.5% class balance.
- **Model Comparison**:
  | Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
  | :--- | :---: | :---: | :---: | :---: | :---: |
  | **Logistic Regression** | 80.06% | 65.48% | 54.28% | 0.5934 | 0.8418 |
  | **Decision Tree** | 78.78% | 62.42% | 51.07% | 0.5617 | 0.8172 |
  | **Random Forest (Champion)** | **80.20%** | **66.30%** | **54.81%** | **0.6001** | **0.8445** |
  | **K-Nearest Neighbors** | 77.22% | 57.73% | 50.80% | 0.5404 | 0.7902 |

### 🔹 Week 3: Unsupervised Clustering & Model Optimization
- **Dimensionality Reduction**: Compressed 30 features into 2 Principal Components (PCA).
- **K-Means Clustering**: Validated optimal $K=3$ using Elbow Method (Inertia) and Silhouette analysis.
- **Customer Personas**:
  - **Cluster 0 (High-Risk New Users)**: Short tenure (<12 mos), month-to-month contracts, churn > 45%. Action: First-year onboarding incentives.
  - **Cluster 1 (High-Value Champions)**: Long tenure (>40 mos), fiber & streaming bundles, churn < 10%. Action: VIP loyalty status.
  - **Cluster 2 (Budget Users)**: Basic phone/DSL plans, low spend. Action: Flexible digital billing options.
- **Hyperparameter Optimization**: `GridSearchCV` on Random Forest with `class_weight='balanced'` boosted **Recall from 54.8% to 76.2%** and **ROC-AUC to 0.85**.

### 🔹 Week 4: Model Serialization & FastAPI Microservice
- **Serialization**: Saved model, scaler, column schema, and K-Means clusters into `churn_pipeline_artifacts.joblib`.
- **API Development**: Created asynchronous FastAPI service with Pydantic data contracts.
- **Latency**: Sub-100ms real-time inference returning churn probability, risk level, cluster persona, and targeted recommendations.

---

## 💻 Installation & Local Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/customer-churn-ml-pipeline.git
cd customer-churn-ml-pipeline
```

### 2. Set Up a Virtual Environment & Install Dependencies
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Run the FastAPI Prediction Microservice
```bash
uvicorn app:app --reload --port 8000
```
- Open Swagger UI in your browser: `http://127.0.0.1:8000/docs`

---

## 📡 API Usage Example

### Endpoint: `POST /predict`

#### Sample JSON Request Payload:
```json
{
  "tenure": 2.0,
  "MonthlyCharges": 95.0,
  "TotalCharges": 190.0,
  "Contract_One_year": 0,
  "Contract_Two_year": 0,
  "InternetService_Fiber_optic": 1,
  "InternetService_No": 0,
  "PaymentMethod_Credit_card_automatic": 0,
  "PaymentMethod_Electronic_check": 1,
  "PaymentMethod_Mailed_check": 0
}
```

#### Sample JSON API Response:
```json
{
  "churn_prediction": "Churn",
  "churn_probability": 0.8245,
  "risk_level": "High",
  "assigned_cluster": "Cluster 0",
  "recommended_action": "Urgent retention outreach"
}
```

---

## 👥 Author
- **Name**: Kaivalya Krishna
- **Program**: Artificial Intelligence & Machine Learning Internship
- **Role**: Machine Learning Engineer