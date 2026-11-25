import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "student-mat.csv"
MODEL_DIR = BASE_DIR / "models"

NUMERIC_FEATURES = [
    "age",
    "Medu",
    "Fedu",
    "traveltime",
    "studytime",
    "failures",
    "famrel",
    "freetime",
    "goout",
    "Dalc",
    "Walc",
    "health",
    "absences",
    "G1",
    "G2",
]

CATEGORICAL_FEATURES = [
    "school",
    "sex",
    "address",
    "famsize",
    "Pstatus",
    "Mjob",
    "Fjob",
    "reason",
    "guardian",
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "nursery",
    "higher",
    "internet",
    "romantic",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "passed"
PASS_THRESHOLD = 10


def load_data():
    frame = pd.read_csv(DATA_PATH, sep=";")
    frame[TARGET] = (frame["G3"] >= PASS_THRESHOLD).astype(int)
    return frame


def build_model():
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    classifier = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline(
        [("preprocessor", preprocessor), ("classifier", classifier)]
    )


def build_metadata(frame):
    return {
        "target": TARGET,
        "positive_label": "Lulus",
        "negative_label": "Tidak Lulus",
        "pass_threshold": PASS_THRESHOLD,
        "numeric": NUMERIC_FEATURES,
        "categorical": CATEGORICAL_FEATURES,
        "categories": {
            name: sorted(frame[name].dropna().astype(str).unique().tolist())
            for name in CATEGORICAL_FEATURES
        },
        "ranges": {
            name: {
                "min": int(frame[name].min()),
                "max": int(frame[name].max()),
                "mean": round(float(frame[name].mean()), 2),
            }
            for name in NUMERIC_FEATURES
        },
        "dataset": {
            "file": DATA_PATH.name,
            "source": "UCI Machine Learning Repository",
            "title": "Student Performance Data Set",
            "url": "https://archive.ics.uci.edu/dataset/320/student+performance",
            "rows": int(len(frame)),
        },
    }


def main():
    frame = load_data()
    features = frame[FEATURES]
    target = frame[TARGET]

    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, stratify=target, random_state=42
    )

    model = build_model()
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, x_train, y_train, cv=folds, scoring="f1")

    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "cv_f1_mean": round(float(cv_scores.mean()), 4),
        "cv_f1_std": round(float(cv_scores.std()), 4),
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions)), 4),
        "recall": round(float(recall_score(y_test, predictions)), 4),
        "f1": round(float(f1_score(y_test, predictions)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "model.joblib")
    (MODEL_DIR / "metadata.json").write_text(
        json.dumps(build_metadata(frame), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (MODEL_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Dataset: {len(frame)} baris, {len(FEATURES)} fitur")
    print(f"CV F1: {metrics['cv_f1_mean']} +/- {metrics['cv_f1_std']}")
    print(f"Test accuracy: {metrics['accuracy']} | F1: {metrics['f1']} | AUC: {metrics['roc_auc']}")
    print(f"Model tersimpan di {MODEL_DIR / 'model.joblib'}")


if __name__ == "__main__":
    main()
