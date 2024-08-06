from flask import Flask, request, jsonify, render_template
import mysql.connector
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from joblib import load

app = Flask(__name__)

# Database configuration
db_config = {
    'user': 'root',
    'password': 'suryaa',
    'host': 'localhost',
    'database': 'credit_risk_assessment'
}

# Load the trained model
model = load('model.pkl')

# Define required fields
required_fields = ['person_age', 'person_income', 'person_home_ownership', 'person_emp_length',
                   'loan_intent', 'loan_amnt', 'loan_int_rate', 'loan_percent_income',
                   'cb_person_default_on_file', 'cb_person_cred_hist_length']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()

    # Check if all required fields are present
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'Missing field: {field}'}), 400

    # Convert data to appropriate types
    data['person_age'] = int(data['person_age'])
    data['person_income'] = float(data['person_income'])
    data['person_emp_length'] = float(data['person_emp_length'])
    data['loan_amnt'] = float(data['loan_amnt'])
    data['loan_int_rate'] = float(data['loan_int_rate'])
    data['loan_percent_income'] = float(data['loan_percent_income'])
    data['cb_person_cred_hist_length'] = int(data['cb_person_cred_hist_length'])

    input_data = [data['person_age'], data['person_income'], data['person_home_ownership'], data['person_emp_length'], data['loan_intent'], data['loan_amnt'], data['loan_int_rate'], data['loan_percent_income'], data['cb_person_default_on_file'], data['cb_person_cred_hist_length']]
    input_df = pd.DataFrame([input_data], columns=required_fields)

    # Ensure model is correctly loaded and used for prediction
    if isinstance(model, Pipeline):
        try:
            prediction = model.predict(input_df)[0]
        except Exception as e:
            return jsonify({'error': str(e)}), 500  # Handle any exceptions gracefully
    else:
        return jsonify({'error': 'Model is not a pipeline'}), 500

    # Connect to the database and store prediction
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    add_prediction = ("INSERT INTO CreditRisk "
                      "(person_age, person_income, person_home_ownership, person_emp_length, loan_intent, loan_amnt, loan_int_rate, loan_percent_income, cb_person_default_on_file, cb_person_cred_hist_length, prediction) "
                      "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)")
    prediction_data = (data['person_age'], data['person_income'], data['person_home_ownership'], data['person_emp_length'], data['loan_intent'], data['loan_amnt'], data['loan_int_rate'], data['loan_percent_income'], data['cb_person_default_on_file'], data['cb_person_cred_hist_length'], int(prediction))
    cursor.execute(add_prediction, prediction_data)
    conn.commit()
    cursor.close()
    conn.close()

    # Render prediction result and visualizations
    return render_template('prediction_result.html', prediction=int(prediction))

if __name__ == '__main__':
    app.run(debug=True)
