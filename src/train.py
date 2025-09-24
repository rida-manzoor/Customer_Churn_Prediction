```python
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from preprocess import build_preprocessor

DATA_PATH = "../WA_Fn-UseC_-Telco-Customer-Churn.csv"
SAVE_DIR = "../saved_models/"

# Load data
df = pd.read_csv(DATA_PATH)
df.drop("customerID", axis=1, inplace=True)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

y = df["Churn"].map({"Yes":1, "No":0})
X = df.drop("Churn", axis=1)

# Binary replacements
binary_map = {"Yes":1,"No":0,"Male":1,"Female":0,"No internet service":0,"No phone service":0}
X = X.replace(binary_map)

num_cols = X.select_dtypes(include=["int64","float64"]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

preprocessor = build_preprocessor(num_cols, cat_cols)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Transform for SMOTE
X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep = preprocessor.transform(X_test)

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train_prep, y_train)

# Train XGBoost
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
model.fit(X_res, y_res)

# Save artifacts
joblib.dump(preprocessor, SAVE_DIR + "preprocessor.joblib")
joblib.dump(model, SAVE_DIR + "xgb_model.joblib")
print("Model and preprocessor saved.")
```
