"""Train and compare three student outcome classifiers."""
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "synthetic_student_performance.csv"
MODEL_PATH = BASE_DIR / "best_student_outcome_model.joblib"
MODEL_PATH = BASE_DIR / "random_forest_student_outcome_model.joblib"
METRICS_PATH = BASE_DIR / "model_comparison.csv"
TARGET = "academic_outcome"
FEATURES = [
    "attendance",
    "internal_marks",
    "assignment_score",
    "previous_gpa",
    "study_hours_per_week",
    "class_participation",
]


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    missing = [column for column in FEATURES + [TARGET] if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    df = df.dropna(subset=[TARGET]).copy()
    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    y = df[TARGET].astype(str).str.strip()
    valid = y.ne("") & X.notna().all(axis=1)
    X, y = X.loc[valid], y.loc[valid]
    if y.nunique() < 2:
        raise ValueError("At least two outcome classes are required.")

    print(f"Rows used: {len(y)}")
    print("Target distribution:")
    print(y.value_counts().to_string())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
        ]),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42, class_weight="balanced", min_samples_leaf=3
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1,
            min_samples_leaf=2,
        ),
    }

    results = []
    fitted = {}
    labels = sorted(y.unique())
    for name, model in models.items():
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        fitted[name] = model
        results.append({
            "model": name,
            "accuracy": accuracy_score(y_test, predicted),
            "macro_precision": precision_score(y_test, predicted, average="macro", zero_division=0),
            "macro_recall": recall_score(y_test, predicted, average="macro", zero_division=0),
            "macro_f1": f1_score(y_test, predicted, average="macro", zero_division=0),
        })
        print(f"\n{'=' * 72}\n{name}")
        print(classification_report(y_test, predicted, labels=labels, zero_division=0))
        matrix = confusion_matrix(y_test, predicted, labels=labels)
        print("Confusion matrix (rows=true, columns=predicted; label order:", labels, ")")
        print(pd.DataFrame(matrix, index=labels, columns=labels).to_string())

    comparison = pd.DataFrame(results).sort_values(
        ["macro_f1", "accuracy"], ascending=False
    ).reset_index(drop=True)
    comparison.to_csv(METRICS_PATH, index=False)
    print("\nModel comparison (ranked by macro F1, then accuracy):")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    winner_name = comparison.loc[0, "model"]
    # Refit the selected model on all labeled rows for deployment after unbiased comparison.
    best_model = fitted[winner_name]
    best_model.fit(X, y)
    artifact = {
        "model": best_model,
        "model_name": winner_name,
        "features": FEATURES,
        "target": TARGET,
        "classes": list(best_model.classes_),
    }
    joblib.dump(artifact, MODEL_PATH)
    summary = {
        "best_model": winner_name,
        "selection_metric": "macro_f1 (tie-break: accuracy)",
        "test_metrics": comparison.iloc[0].to_dict(),
        "features": FEATURES,
        "target": TARGET,
    }
    (BASE_DIR / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\nSelected model: {winner_name}")
    print(f"Saved model: {MODEL_PATH.name}")
    print(f"Saved comparison: {METRICS_PATH.name}")


if __name__ == "__main__":
    main()

