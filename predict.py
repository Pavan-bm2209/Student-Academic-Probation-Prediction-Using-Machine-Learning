"""Prompt for student details and predict with the saved Random Forest."""
from pathlib import Path
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "random_forest_student_outcome_model.joblib"


def predict_outcome(attendance, internal_marks, assignment_score, previous_gpa,
                    study_hours_per_week, class_participation):
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run train_model.py first.")
    artifact = joblib.load(MODEL_PATH)
    row = pd.DataFrame([{
        "attendance": attendance,
        "internal_marks": internal_marks,
        "assignment_score": assignment_score,
        "previous_gpa": previous_gpa,
        "study_hours_per_week": study_hours_per_week,
        "class_participation": class_participation,
    }], columns=artifact["features"])
    model = artifact["model"]
    label = model.predict(row)[0]
    result = {"prediction": str(label)}
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(row)[0]
        result["probabilities"] = {
            str(class_name): float(probability)
            for class_name, probability in zip(model.classes_, probabilities)
        }
    return result


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. First run: python train_model.py"
        )

    print("Student Academic Outcome Prediction (Random Forest)")
    print("Enter the student's details:")
    try:
        values = {
            "attendance": float(input("Attendance (%): ")),
            "internal_marks": float(input("Internal marks: ")),
            "assignment_score": float(input("Assignment score: ")),
            "previous_gpa": float(input("Previous semester GPA: ")),
            "study_hours_per_week": float(input("Study hours per week: ")),
            "class_participation": float(input("Class participation score: ")),
        }
    except ValueError as exc:
        raise SystemExit("Please enter a valid number for each value.") from exc

    result = predict_outcome(**values)
    print(f"\nPredicted academic outcome: {result['prediction']}")
    if "probabilities" in result:
        print("Prediction probabilities:")
        for label, probability in result["probabilities"].items():
            print(f"  {label}: {probability:.1%}")


if __name__ == "__main__":
    main()
