import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.impute import SimpleImputer
from joblib import dump

# Load the dataset
file_path = "C:\\Users\\chsur\\Downloads\\archive (4)\\credit_risk_dataset.csv"
data = pd.read_csv(file_path)

# Data Preprocessing
numeric_features = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']
categorical_features = ['person_home_ownership', 'loan_intent', 'cb_person_default_on_file']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Split the data
X = data.drop('loan_status', axis=1)
y = data['loan_status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LogisticRegression())])

clf.fit(X_train, y_train)

# Visualize coefficients
def plot_coefficients(classifier, feature_names):
    # Get feature names after one-hot encoding
    categorical_encoder = classifier.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
    feature_names_categorical = categorical_encoder.get_feature_names_out(categorical_features)
    
    # Combine numeric and categorical feature names
    all_feature_names = numeric_features + list(feature_names_categorical)
    
    coef = classifier.named_steps['classifier'].coef_.flatten()
    feature_importance = pd.DataFrame({'feature': all_feature_names, 'importance': coef})
    


plt.figure(figsize=(10, 8))
sns.boxplot(x='loan_intent', y='loan_amnt', data=data, palette='Set2')
plt.title('Loan Amount Distribution across Loan Intent')
plt.xlabel('Loan Intent')
plt.ylabel('Loan Amount')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('static/feature_importance.png.png')  # Save the plot to a file
plt.show()

# Model Evaluation
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'ROC-AUC: {roc_auc}')

# Save the model
dump(clf, 'model.pkl')
