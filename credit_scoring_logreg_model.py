# credit_scoring_logreg_model.py

import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


#1. Loading dataset
df = pd.read_csv("german.data-numeric", delim_whitespace=True, header=None)

# Assign column names based on the documentation:
#  - Features 0–23 are input variables (some are numeric encodings of categorical attributes)
#  - Column 24 is the target variable: 1 = good credit risk, 2 = bad credit risk
df.columns = [
    "Status", "Duration", "CreditHistory", "Purpose", "Amount",
    "Savings", "EmploymentDuration", "InstallmentRate", "PersonalStatusSex",
    "OtherDebtors", "ResidenceDuration", "Property", "Age", "OtherInstallmentPlans",
    "Housing", "ExistingCredits", "Job", "PeopleLiable", "Telephone", "ForeignWorker",
    "Unknown1", "Unknown2", "Unknown3", "Unknown4", "Target"
]

# 2. Features and target
X = df.drop("Target", axis=1)
y = df["Target"]

# 3. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# 4. Train logistic regression model
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

# 5. Predict and evaluate
y_pred = model.predict(X_test)
print("Model evaluation:")
print(classification_report(y_test, y_pred))

# 6. Save the trained model
joblib.dump(model, "logreg_model.pkl")
print("Model saved as logreg_model.pkl")
