# src/train_model.py
"""
Train and save a churn prediction model.

Usage (from project root):
    python src/train_model.py \
        --data_path data/WA_Fn-UseC_-Telco-Customer-Churn.csv \
        --save_dir saved_models \
        --model xgboost \
        --tune \
        --trials 20

Notes:
- Requires: pandas, numpy, scikit-learn, imbalanced-learn, xgboost, optuna (optional), joblib
- If optuna is not installed, the script will fall back to training with default hyperparameters.
"""
import os
import json
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib

# Models
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# Optional: optuna for tuning
try:
    import optuna
    OPTUNA_AVAILABLE = True
except Exception:
    OPTUNA_AVAILABLE = False

RANDOM_STATE = 42
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def load_data(path):
    df = pd.read_csv(path)
    logging.info("Loaded data shape: %s", df.shape)
    return df


def clean_and_split(df, target_col="Churn"):
    # Drop customerID if present
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Convert TotalCharges to numeric
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Map common binary-like strings to numeric where appropriate
    binary_map = {
        "Yes": 1,
        "No": 0,
        "Male": 1,
        "Female": 0,
        "No internet service": 0,
        "No phone service": 0,
    }
    df = df.replace(binary_map)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    y = df[target_col].map({"Yes": 1, "No": 0}) if df[target_col].dtype == "object" else df[target_col]
    X = df.drop(columns=[target_col])
    return X, y.astype(int)


def build_preprocessor(X):
    # identify numeric and categorical columns
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, num_cols), ("cat", categorical_transformer, cat_cols)]
    )

    logging.info("Preprocessor built. Numeric cols: %d, Categorical cols: %d", len(num_cols), len(cat_cols))
    return preprocessor, num_cols, cat_cols


def train_default_model(model_name="xgboost", random_state=RANDOM_STATE):
    if model_name == "xgboost":
        return xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=random_state, n_jobs=-1)
    elif model_name == "rf" or model_name == "random_forest":
        return RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
    else:
        raise ValueError("Unsupported model_name. Choose 'xgboost' or 'rf'.")


def optuna_tune_xgboost(X_train, y_train, n_trials=20, random_state=RANDOM_STATE):
    if not OPTUNA_AVAILABLE:
        raise RuntimeError("Optuna not available. Install optuna to run tuning.")

    def objective(trial):
        # sample hyperparameters
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 0.3),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
            "random_state": random_state,
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "n_jobs": -1,
        }

        # small holdout inside training data for validation
        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=random_state, stratify=y_train)
        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr, early_stopping_rounds=20, eval_set=[(X_val, y_val)], verbose=False)
        preds_proba = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, preds_proba)
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    logging.info("Optuna best trial: %s", study.best_trial.params)
    return study.best_trial.params


def evaluate_and_report(model, X_test, y_test):
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, preds, output_dict=True)
    roc = roc_auc_score(y_test, proba)
    cm = confusion_matrix(y_test, preds)
    logging.info("Evaluation results: ROC-AUC=%.4f", roc)
    return {"classification_report": report, "roc_auc": float(roc), "confusion_matrix": cm.tolist()}


def save_artifacts(preprocessor, model, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    preprocessor_path = os.path.join(save_dir, "preprocessor.joblib")
    model_path = os.path.join(save_dir, "model.joblib")
    joblib.dump(preprocessor, preprocessor_path)
    joblib.dump(model, model_path)
    logging.info("Saved preprocessor -> %s", preprocessor_path)
    logging.info("Saved model -> %s", model_path)
    return preprocessor_path, model_path


def main(args):
    df = load_data(args.data_path)
    X, y = clean_and_split(df)
    preprocessor, num_cols, cat_cols = build_preprocessor(X)

    # Split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=args.test_size, stratify=y, random_state=RANDOM_STATE)
    logging.info("Train/test split sizes: %s / %s", X_train_raw.shape, X_test_raw.shape)

    # Fit preprocessor on training data and transform
    X_train_prep = preprocessor.fit_transform(X_train_raw)
    X_test_prep = preprocessor.transform(X_test_raw)

    # Balance with SMOTE
    sm = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = sm.fit_resample(X_train_prep, y_train)
    logging.info("After SMOTE: X_res.shape=%s, positive_count=%d", X_res.shape, int(y_res.sum()))

    # Train: tuning or default
    if args.tune and args.model.lower() in ["xgboost", "xgb"]:
        if not OPTUNA_AVAILABLE:
            logging.warning("Optuna not installed; continuing without tuning.")
            model = train_default_model("xgboost")
            model.fit(X_res, y_res)
        else:
            best_params = optuna_tune_xgboost(X_res, y_res, n_trials=args.trials)
            # ensure required params for classifier
            model = xgb.XGBClassifier(**{k: v for k, v in best_params.items() if k not in ["random_state", "use_label_encoder", "eval_metric", "n_jobs"]},
                                      use_label_encoder=False, eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1)
            # include n_estimators if present
            if "n_estimators" in best_params:
                model.set_params(n_estimators=int(best_params["n_estimators"]))
            model.fit(X_res, y_res)
    else:
        model = train_default_model("xgboost" if args.model.lower() in ["xgboost", "xgb"] else "rf")
        model.fit(X_res, y_res)

    # Evaluate
    eval_metrics = evaluate_and_report(model, X_test_prep, y_test)
    logging.info("ROC-AUC on test: %.4f", eval_metrics["roc_auc"])
    logging.info("Classification report (test):\n%s", json.dumps(eval_metrics["classification_report"], indent=2))

    # Save artifacts and metrics
    preprocessor_path, model_path = save_artifacts(preprocessor, model, args.save_dir)
    metrics_path = os.path.join(args.save_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(eval_metrics, f, indent=2)
    logging.info("Saved metrics -> %s", metrics_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train churn prediction model")
    parser.add_argument("--data_path", type=str, default="data/WA_Fn-UseC_-Telco-Customer-Churn.csv", help="Path to CSV dataset")
    parser.add_argument("--save_dir", type=str, default="saved_models", help="Directory to save artifacts")
    parser.add_argument("--model", type=str, default="xgboost", help="Model type: xgboost | rf")
    parser.add_argument("--tune", action="store_true", help="Run Optuna tuning (xgboost only)")
    parser.add_argument("--trials", type=int, default=20, help="Optuna trials if tuning")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set fraction")
    args = parser.parse_args()
    main(args)
