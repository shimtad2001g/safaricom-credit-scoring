import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
import pickle

print("Generating synthetic Telco/FinTech dataset...")
np.random.seed(42)

# Simulate 10,000 customers
n_samples = 10000
data = {
    'age': np.random.randint(18, 70, n_samples),
    'monthly_income': np.random.randint(1000, 50000, n_samples),
    'loan_amount': np.random.randint(500, 20000, n_samples),
    'tenure_months': np.random.randint(1, 120, n_samples),
    'num_late_payments': np.random.randint(0, 10, n_samples),
    'data_usage_gb': np.random.uniform(1.0, 50.0, n_samples)
}
df = pd.DataFrame(data)

# Create target: Default (1 = Default, 0 = Good Payer)
# Logic: Higher late payments + High loan amount relative to income = Higher default risk
risk_score = (df['num_late_payments'] * 0.5) + (df['loan_amount'] / df['monthly_income']) - (df['tenure_months'] * 0.01)
df['default'] = (risk_score > 1.5).astype(int)

# Train-test split
X = df.drop('default', axis=1)
y = df['default']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)

print(f"Model Accuracy: {acc*100:.2f}%")
print(f"Model Precision: {prec*100:.2f}%")

# Save the model
with open('credit_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("[OK] Model trained and saved as credit_model.pkl")
