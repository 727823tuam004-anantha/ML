# ROLLNO: 727823TUAM004
from datetime import datetime
import os
import time
import json
import argparse
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(X):
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--model_output", type=str, required=True)
    parser.add_argument("--metrics_output", type=str, required=True)
    parser.add_argument("--student_name", type=str, required=True)
    parser.add_argument("--roll_number", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="HospitalReadmission")
    args = parser.parse_args()

    print(f"ROLLNO: 727823TUAM004 | TIMESTAMP: {datetime.now().isoformat()}")

    df = pd.read_csv(os.path.join(args.train_data, "train.csv"))
    X_train = df.drop(columns=["target"])
    y_train = df["target"]

    pipeline = Pipeline([
        ("preprocessor", build_preprocessor(X_train)),
        ("classifier", LogisticRegression(max_iter=1000, solver="liblinear", random_state=42))
    ])

    mlflow.set_experiment(f"SKCT_{args.roll_number}_{args.dataset_name}")

    with mlflow.start_run():
        mlflow.set_tags({
            "student_name": args.student_name,
            "roll_number": args.roll_number,
            "dataset": args.dataset_name
        })

        start = time.time()
        pipeline.fit(X_train, y_train)
        training_time = time.time() - start

        os.makedirs(args.model_output, exist_ok=True)
        model_path = os.path.join(args.model_output, "model.joblib")
        joblib.dump(pipeline, model_path)

        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)

        mlflow.log_metric("training_time_seconds", float(training_time))
        mlflow.log_metric("model_size_mb", float(model_size_mb))
        mlflow.log_metric("random_seed", 42.0)
        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        os.makedirs(args.metrics_output, exist_ok=True)
        with open(os.path.join(args.metrics_output, "train_info.json"), "w") as f:
            json.dump({"training_time_seconds": training_time, "model_size_mb": model_size_mb}, f, indent=4)

    print(f"Model saved to {model_path} | training_time={training_time:.2f}s")


if __name__ == "__main__":
    main()
