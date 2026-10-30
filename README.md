# Student Academic Outcome Prediction

This project predicts a student's academic outcome as **Pass**, **Fail**, or **Academic Support Required** using attendance, marks, GPA, study hours, and class participation.

It compares three machine-learning models:

- Logistic Regression
- Decision Tree
- Random Forest

The project reports evaluation metrics and confusion matrices for all three models, then uses Random Forest for predictions entered by the user.

## Dataset

Place `synthetic_student_performance.csv` in this project folder.

The program expects these columns:

- `attendance`
- `internal_marks`
- `assignment_score`
- `previous_gpa`
- `study_hours_per_week`
- `class_participation`
- `academic_outcome` — the target label

The program ignores `student_id`, `is_synthetic`, and `dataset_version`.

## Setup

Open a terminal in this folder and install the required packages:

```bash
python -m pip install -r requirements.txt