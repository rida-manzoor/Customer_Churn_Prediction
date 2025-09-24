```python
import joblib
import shap
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

SAVE_DIR = "../saved_models/"
preprocessor = joblib.load(SAVE_DIR + "preprocessor.joblib")
model = joblib.load(SAVE_DIR + "xgb_model.joblib")

# Example: Load test set again or from pickle
# X_test, y_test = ...

# preds = model.predict(preprocessor.transform(X_test))
# print(classification_report(y_test, preds))
# print("ROC-AUC:", roc_auc_score(y_test, model.predict_proba(preprocessor.transform(X_test))[:,1]))

# SHAP values
# explainer = shap.Explainer(model, preprocessor.transform(X_test))
# shap_values = explainer(preprocessor.transform(X_test)[:100])
# shap.summary_plot(shap_values)
```
