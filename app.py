"""
Attrition Assessment — Flask backend.

Two independent features:

1. Assessment page (/assessment): takes a single employee's attributes from
   a form and scores them with your trained logistic regression model.
   Drop your pickled model into model/attrition_model.pkl (see FEATURE_FIELDS
   below for the expected inputs, and model/scaler.pkl if you scaled your
   training data separately).

2. Home page (/): shows attrition charts built from a CSV file that lives
   in the app itself, at data/attrition_data.csv. Replace that file with
   your own dataset and the charts update automatically — no upload step.
"""

import os
import pickle
import traceback
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
# If your model was trained on scaled data and the scaler is NOT already
# baked into a Pipeline, save the fitted scaler separately as model/scaler.pkl
# and it will be applied automatically before every prediction:
#
#   import pickle
#   with open("model/scaler.pkl", "wb") as f:
#       pickle.dump(your_fitted_scaler, f)
#
# If your model IS a Pipeline that already includes scaling, leave
# model/scaler.pkl absent — raw inputs will be passed straight through.

model = pickle.load(open("model/attrition_model.pkl", "rb"))

# ---------------------------------------------------------------------------
# Feature schema — matches the employee_data.csv columns (excluding the
# Attrition target). Order doesn't matter here since features are passed to
# the model as a named pandas DataFrame, not a positional array — your
# pipeline's ColumnTransformer/preprocessing matches by column name.
# ---------------------------------------------------------------------------
TRAINING_COLUMNS=[
    'Age', 
    'DistanceFromHome', 
    'EnvironmentSatisfaction', 
    'JobInvolvement', 
    'JobLevel', 
    'JobSatisfaction', 
    'MonthlyIncome', 
    'NumCompaniesWorked', 
    'StockOptionLevel', 
    'TotalWorkingYears', 
    'TrainingTimesLastYear', 
    'WorkLifeBalance', 
    'YearsAtCompany', 
    'YearsInCurrentRole', 
    'YearsSinceLastPromotion', 
    'YearsWithCurrManager', 
    'BusinessTravel_Non-Travel', 
    'BusinessTravel_Travel_Frequently', 
    'BusinessTravel_Travel_Rarely', 
    'Department_Human Resources', 
    'Department_Research & Development', 
    'Department_Sales', 
    'EducationField_Human Resources', 
    'EducationField_Life Sciences', 
    'EducationField_Marketing', 
    'EducationField_Medical', 
    'EducationField_Other', 
    'EducationField_Technical Degree', 
    'Gender_Female', 
    'Gender_Male', 
    'JobRole_Healthcare Representative', 
    'JobRole_Human Resources', 
    'JobRole_Laboratory Technician', 
    'JobRole_Manager', 
    'JobRole_Manufacturing Director', 
    'JobRole_Research Director', 
    'JobRole_Research Scientist', 
    'JobRole_Sales Executive', 
    'JobRole_Sales Representative', 
    'MaritalStatus_Divorced', 
    'MaritalStatus_Married', 
    'MaritalStatus_Single', 
    'OverTime_No', 
    'OverTime_Yes']

NUMERIC_FIELDS = [
    "Age",
    "DistanceFromHome",
    "EnvironmentSatisfaction",
    "JobInvolvement",
    "JobLevel",
    "JobSatisfaction",
    "MonthlyIncome",
    "NumCompaniesWorked",
    "StockOptionLevel",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "WorkLifeBalance",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
]

CATEGORICAL_FIELDS = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "OverTime",
]

FEATURE_FIELDS = NUMERIC_FIELDS + CATEGORICAL_FIELDS

# Dropdown choices for the assessment form, drawn from employee_data.csv
CATEGORICAL_CHOICES = {
    "BusinessTravel": ["Non-Travel", "Travel_Rarely", "Travel_Frequently"],
    "Department": ["Human Resources", "Research & Development", "Sales"],
    "EducationField": ["Human Resources", "Life Sciences", "Marketing", "Medical", "Other", "Technical Degree"],
    "Gender": ["Female", "Male"],
    "JobRole": [
        "Healthcare Representative", "Human Resources", "Laboratory Technician",
        "Manager", "Manufacturing Director", "Research Director",
        "Research Scientist", "Sales Executive", "Sales Representative",
    ],
    "MaritalStatus": ["Divorced", "Married", "Single"],
    "OverTime": ["No", "Yes"],
}


def build_feature_dataframe(payload):
    row = {}

    # Numeric fields
    for field in NUMERIC_FIELDS:
        row[field] = float(payload[field])

    # Initialize every dummy column to 0
    dummy_columns = []

    for field, choices in CATEGORICAL_CHOICES.items():
        for choice in choices:
            col = f"{field}_{choice}"
            row[col] = 0
            dummy_columns.append(col)

    # Set selected category to 1
    for field in CATEGORICAL_FIELDS:
        value = payload[field]
        col = f"{field}_{value}"

        if col in row:
            row[col] = 1

    features = pd.DataFrame([row])

    return features


# ---------------------------------------------------------------------------
# Dataset → chart helpers (home page)
# ---------------------------------------------------------------------------
# These look for common HR-attrition-style column names but are forgiving:
# matching is case-insensitive and any chart whose required column is
# missing is simply skipped (the front end hides that card).

# candidate column names -> canonical key, checked case-insensitively
COLUMN_CANDIDATES = {
    "attrition": ["attrition", "left", "churn", "target", "exited"],
    "department": ["department", "dept", "team"],
    "age": ["age"],
    "overtime": ["overtime", "over_time"],
    "job_satisfaction": ["jobsatisfaction", "job_satisfaction"],
    "years_at_company": ["yearsatcompany", "years_at_company", "tenure"],
    "monthly_income": ["monthlyincome", "monthly_income", "salary"],
    "gender": ["gender", "sex"],
}


def find_column(df: pd.DataFrame, candidates: list) -> str | None:
    lower_map = {c.lower().replace(" ", "").replace("-", "_"): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().replace(" ", "").replace("-", "_")
        if key in lower_map:
            return lower_map[key]
    return None


def to_binary_attrition(series: pd.Series) -> pd.Series:
    """Normalize a variety of attrition encodings to 0/1."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float).clip(0, 1).round().astype(int)
    mapped = series.astype(str).str.strip().str.lower().map({
        "yes": 1, "y": 1, "true": 1, "1": 1, "left": 1, "churned": 1,
        "no": 0, "n": 0, "false": 0, "0": 0, "stayed": 0, "retained": 0,
    })
    return mapped.fillna(0).astype(int)


def bucket_series(series: pd.Series, bins: list, labels: list) -> pd.Series:
    return pd.cut(series, bins=bins, labels=labels, right=False, include_lowest=True)


def build_charts_from_dataframe(df: pd.DataFrame) -> dict:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    cols = {key: find_column(df, candidates) for key, candidates in COLUMN_CANDIDATES.items()}
    charts = {}
    warnings = []

    if cols["attrition"] is None:
        raise ValueError(
            "Could not find an attrition column in this file. "
            "Expected a column named something like 'Attrition', 'Left', or 'Churn'."
        )

    df["_attrition"] = to_binary_attrition(df[cols["attrition"]])
    overall_rate = round(float(df["_attrition"].mean()) * 100, 1)
    total_rows = int(len(df))
    total_left = int(df["_attrition"].sum())

    charts["summary"] = {
        "total_employees": total_rows,
        "total_attrition": total_left,
        "attrition_rate": overall_rate,
    }

    charts["overall"] = {
        "type": "doughnut",
        "title": "Overall attrition",
        "labels": ["Left", "Stayed"],
        "data": [total_left, total_rows - total_left],
    }

    # --- by department ---
    if cols["department"]:
        grp = df.groupby(cols["department"])["_attrition"].mean().sort_values(ascending=False) * 100
        charts["department"] = {
            "type": "bar",
            "title": "Attrition rate by department",
            "labels": [str(i) for i in grp.index.tolist()],
            "data": [round(v, 1) for v in grp.values.tolist()],
        }
    else:
        warnings.append("No department column found — skipped department chart.")

    # --- by overtime ---
    if cols["overtime"]:
        grp = df.groupby(cols["overtime"])["_attrition"].mean().sort_values(ascending=False) * 100
        charts["overtime"] = {
            "type": "bar",
            "title": "Attrition rate by overtime status",
            "labels": [str(i) for i in grp.index.tolist()],
            "data": [round(v, 1) for v in grp.values.tolist()],
        }
    else:
        warnings.append("No overtime column found — skipped overtime chart.")

    # --- by age bucket ---
    if cols["age"]:
        age_series = pd.to_numeric(df[cols["age"]], errors="coerce")
        bins = [18, 25, 35, 45, 55, 100]
        labels = ["18-24", "25-34", "35-44", "45-54", "55+"]
        df["_age_bucket"] = bucket_series(age_series, bins, labels)
        grp = df.groupby("_age_bucket")["_attrition"].mean().reindex(labels) * 100
        charts["age"] = {
            "type": "line",
            "title": "Attrition rate by age group",
            "labels": labels,
            "data": [round(v, 1) if pd.notna(v) else 0 for v in grp.values.tolist()],
        }
    else:
        warnings.append("No age column found — skipped age chart.")

    # --- by job satisfaction ---
    if cols["job_satisfaction"]:
        grp = df.groupby(cols["job_satisfaction"])["_attrition"].mean().sort_index() * 100
        charts["job_satisfaction"] = {
            "type": "bar",
            "title": "Attrition rate by job satisfaction",
            "labels": [str(i) for i in grp.index.tolist()],
            "data": [round(v, 1) for v in grp.values.tolist()],
        }
    else:
        warnings.append("No job satisfaction column found — skipped that chart.")

    # --- by years at company ---
    if cols["years_at_company"]:
        yrs = pd.to_numeric(df[cols["years_at_company"]], errors="coerce")
        bins = [0, 2, 5, 10, 20, 100]
        labels = ["0-1", "2-4", "5-9", "10-19", "20+"]
        df["_tenure_bucket"] = bucket_series(yrs, bins, labels)
        grp = df.groupby("_tenure_bucket")["_attrition"].mean().reindex(labels) * 100
        charts["tenure"] = {
            "type": "bar",
            "title": "Attrition rate by years at company",
            "labels": labels,
            "data": [round(v, 1) if pd.notna(v) else 0 for v in grp.values.tolist()],
        }
    else:
        warnings.append("No tenure/years-at-company column found — skipped that chart.")

    charts["warnings"] = warnings
    charts["detected_columns"] = {k: v for k, v in cols.items() if v}
    return charts


# ---------------------------------------------------------------------------
# Routes — pages
# ---------------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/assessment")
def assessment():
    return render_template(
        "assessment.html",
        numeric_fields=NUMERIC_FIELDS,
        categorical_choices=CATEGORICAL_CHOICES,
    )


# ---------------------------------------------------------------------------
# Routes — API
# ---------------------------------------------------------------------------

def risk_bucket(probability):
    if probability < 0.33:
        return "low"
    elif probability < 0.66:
        return "moderate"
    else:
        return "high"

@app.route("/api/predict", methods=["POST"])
def predict():

    payload = request.get_json(silent=True) or {}

    try:
        features = build_feature_dataframe(payload)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    try:
    # Ensure the columns match those used during training
        for col in TRAINING_COLUMNS:
            if col not in features.columns:
                features[col] = 0

        features = features[TRAINING_COLUMNS]

        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(features)[0][1])
        else:
            probability = float(model.predict(features)[0])

    except Exception:
        traceback.print_exc()
        return jsonify({
            "error": traceback.format_exc()
        }), 500

    probability = max(0.0, min(1.0, probability))

    return jsonify({
        "probability": round(probability, 4),
        "percentage": round(probability * 100, 1),
        "risk_level": risk_bucket(probability),
    })


DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "attrition_data.csv")


@app.route("/api/charts")
def get_charts():
    if not os.path.exists(DATA_PATH):
        return jsonify({
            "error": (
                "No dataset found at data/attrition_data.csv. "
            ),
        }), 404

    try:
        df = pd.read_csv(DATA_PATH)
    except Exception as exc:
        return jsonify({"error": f"Could not read data/attrition_data.csv: {exc}"}), 500

    if df.empty:
        return jsonify({"error": "data/attrition_data.csv has no rows."}), 400

    try:
        charts = build_charts_from_dataframe(df)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover - defensive
        return jsonify({"error": f"Could not process this dataset: {exc}"}), 500

    return jsonify(charts)


@app.route("/api/health")
def health():
    return jsonify({
        "model_loaded": model is not None,
        "message": "model is none" if model is None else "Model ready.",
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
