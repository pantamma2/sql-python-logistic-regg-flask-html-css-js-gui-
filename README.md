#  Credit Risk Assessment System

A machine learning web application that predicts whether a loan applicant is a credit risk, using Logistic Regression, a Flask backend, MySQL database, and an interactive GUI.
 
##  What It Does

A user enters loan applicant details through a web form. The app runs the data through a trained ML model and predicts whether the applicant is **high risk** or **low risk** for credit default — and stores every prediction in a MySQL database for record keeping.

##  ML Pipeline

```
Preprocessing Pipeline (sklearn)
  ├── SimpleImputer     → handles missing values
  ├── StandardScaler    → normalizes numerical features
  └── OneHotEncoder     → encodes categorical features
        │
        ▼
Logistic Regression Model (model.pkl)
        │
        ▼
Prediction: High Risk / Low Risk
        │
        ▼
Result stored in MySQL → Rendered in UI
```

---

##  Input Features

| Feature | Type | Description |
|---|---|---|
| `person_age` | int | Applicant's age |
| `person_income` | float | Annual income |
| `person_home_ownership` | categorical | RENT / OWN / MORTGAGE |
| `person_emp_length` | float | Years of employment |
| `loan_intent` | categorical | Purpose of loan |
| `loan_amnt` | float | Loan amount requested |
| `loan_int_rate` | float | Interest rate |
| `loan_percent_income` | float | Loan as % of income |
| `cb_person_default_on_file` | categorical | Prior default history |
| `cb_person_cred_hist_length` | int | Credit history length (years) |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| ML Model | Scikit-learn (Logistic Regression + Pipeline) |
| Backend | Python, Flask |
| Database | MySQL (stores all predictions) |
| Frontend | HTML, CSS, JavaScript |
| Model Serialization | Joblib (.pkl) |

---

##  Setup & Installation

### Prerequisites
- Python 3.8+
- MySQL Server running locally

### 1. Clone the repository
```bash
git clone https://github.com/pantamma2/sql-python-logistic-regg-flask-html-css-js-gui-.git
cd sql-python-logistic-regg-flask-html-css-js-gui-
```

### 2. Install dependencies
```bash
pip install flask mysql-connector-python scikit-learn pandas joblib
```

### 3. Set up MySQL database
```sql
CREATE DATABASE credit_risk_assessment;
USE credit_risk_assessment;

CREATE TABLE CreditRisk (
    id INT AUTO_INCREMENT PRIMARY KEY,
    person_age INT,
    person_income FLOAT,
    person_home_ownership VARCHAR(20),
    person_emp_length FLOAT,
    loan_intent VARCHAR(30),
    loan_amnt FLOAT,
    loan_int_rate FLOAT,
    loan_percent_income FLOAT,
    cb_person_default_on_file VARCHAR(5),
    cb_person_cred_hist_length INT,
    prediction INT
);
```

### 4. Update DB credentials in `app.py`
```python
db_config = {
    'user': 'your_username',
    'password': 'your_password',
    'host': 'localhost',
    'database': 'credit_risk_assessment'
}
```

### 5. Train the model (if needed)
```bash
python train_model.py
```

### 6. Run the app
```bash
python app.py
```

##  Author

**Suryanarayana Murthy Chilukuri**
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/suryanarayana-murthy-chilukuri-a88107235)
[![GitHub](https://img.shields.io/badge/GitHub-pantamma2-black?logo=github)](https://github.com/pantamma2)
