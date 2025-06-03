# credit_scoring_logreg_model.py

import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. Synthetic dataset
data = {
    'income': [3000, 1500, 5000, 1200, 2500, 7000, 800, 2200, 6000, 900],
    'loan_amount': [500, 1200, 2000, 800, 1500, 3000, 700, 1100, 2500, 1000],
    'has_children': [1, 0, 1, 1, 0, 0, 1, 1, 0, 0],
    'age': [35, 24, 45, 22, 38, 50, 20, 30, 40, 27],
    'default': [0, 1, 0, 1, 0, 0, 1, 0, 0, 1]
}
df = pd.DataFrame(data)

# 2. Features and target
X = df[['income', 'loan_amount', 'has_children', 'age']]
y = df['default']

# 3. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# 4. Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# 5. Predict and evaluate
y_pred = model.predict(X_test)
print("Model evaluation:")
print(classification_report(y_test, y_pred))

# 6. Save the trained model
joblib.dump(model, "logreg_model.pkl")
print("Model saved as logreg_model.pkl")
