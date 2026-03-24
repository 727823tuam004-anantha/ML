import os
import sys
import time
import argparse
import tempfile
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from utils import load_data, create_target, basic_clean, build_preprocessor, ensure_dir

# Resolve project root (one level up from src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STUDENT_NAME = "Anantha Narayanan A B"
ROLL_NUMBER = "727823TUAM004"
DATASET_NAME = "HospitalReadmission"


def get_model(model_name, seed, params):
    if model_name == "logreg":
        return LogisticRegression(C=params.get("C", 1.0), max_iter=1000, solver="liblinear", random_state=seed)
    elif model_name == "rf":
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 10),
            min_samples_split=params.get("min_samples_split", 2),
            random_state=seed, n_jobs=-1
        )
    elif model_name == "gb":
        return GradientBoostingClassifier(
            n_estimators=params.get("n_estimators", 100),
            learning_rate=params.get("learning_rate", 0.1),
            max_depth=params.get("max_depth", 3),
            random_state=seed
        )
    raise ValueError(f"Unsupported model: {model_name}")


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
    }
    if y_prob is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob))
    return metrics


def get_model_size_mb(model):
    tmp_dir = tempfile.mkdtemp()
    mlflow.sklearn.save_model(model, os.path.join(tmp_dir, "model_dir"))
    total = sum(
        os.path.getsize(os.path.join(r, f))
        for r, _, files in os.walk(os.path.join(tmp_dir, "model_dir"))
        for f in files
    )
    return total / (1024 * 1024)


def train_one_run(data_path, model_name, seed, params):
    df = load_data(data_path)
    df = create_target(df)
    df = basic_clean(df)

    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)

    preprocessor, _, _ = build_preprocessor(X_train)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", get_model(model_name, seed, params))
    ])

    mlflow.set_experiment(f"SKCT_{ROLL_NUMBER}_{DATASET_NAME}")

    with mlflow.start_run():
        mlflow.set_tags({
            "student_name": STUDENT_NAME,
            "roll_number": ROLL_NUMBER,
            "dataset": DATASET_NAME,
            "algorithm": model_name
        })
        mlflow.log_params(params)
        mlflow.log_param("random_seed", seed)

        start = time.time()
        pipeline.fit(X_train, y_train)
        training_time = time.time() - start

        metrics = evaluate(pipeline, X_test, y_test)
        model_size_mb = get_model_size_mb(pipeline)

        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        mlflow.log_metric("training_time_seconds", float(training_time))
        mlflow.log_metric("model_size_mb", float(model_size_mb))
        mlflow.log_metric("random_seed", float(seed))

        mlflow.sklearn.log_model(pipeline, artifact_path="model")
        run_id = mlflow.active_run().info.run_id

    return {"run_id": run_id, "model_name": model_name, "seed": seed, "params": params,
            "metrics": metrics, "best_metric": metrics["f1_score"], "pipeline": pipeline}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=os.path.join(PROJECT_ROOT, "data", "diabetes_data.csv"))
    args = parser.parse_args()

    run_plan = [
        ("logreg", 42,  {"C": 0.1}),
        ("logreg",  7,  {"C": 1.0}),
        ("logreg", 21,  {"C": 10.0}),
        ("logreg", 99,  {"C": 0.5}),
        ("rf",     42,  {"n_estimators": 100, "max_depth": 8}),
        ("rf",      7,  {"n_estimators": 150, "max_depth": 10}),
        ("rf",     21,  {"n_estimators": 200, "max_depth": 12}),
        ("rf",     99,  {"n_estimators": 120, "max_depth": 6}),
        ("gb",     42,  {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 3}),
        ("gb",      7,  {"n_estimators": 150, "learning_rate": 0.1,  "max_depth": 3}),
        ("gb",     21,  {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 2}),
        ("gb",     99,  {"n_estimators": 120, "learning_rate": 0.2,  "max_depth": 2}),
    ]

    results = []
    best_result = None

    for model_name, seed, params in run_plan:
        print(f"Running {model_name} seed={seed} params={params}")
        result = train_one_run(args.data_path, model_name, seed, params)
        results.append(result)
        if best_result is None or result["best_metric"] > best_result["best_metric"]:
            best_result = result

    best_model_dir = os.path.join(PROJECT_ROOT, "outputs", "best_model")
    metrics_dir = os.path.join(PROJECT_ROOT, "outputs", "metrics")
    ensure_dir(best_model_dir)
    ensure_dir(metrics_dir)

    pd.DataFrame([
        {"run_id": r["run_id"], "algorithm": r["model_name"], "seed": r["seed"],
         **r["params"], **r["metrics"]}
        for r in results
    ]).to_csv(os.path.join(metrics_dir, "all_runs_results.csv"), index=False)

    import mlflow.sklearn as mls
    import shutil
    best_model_path = os.path.join(best_model_dir, "model")
    if os.path.exists(best_model_path):
        shutil.rmtree(best_model_path)
    mls.save_model(best_result["pipeline"], best_model_path)

    with open(os.path.join(best_model_dir, "best_run.txt"), "w") as f:
        f.write(f"best_run_id={best_result['run_id']}\n")
        f.write(f"best_algorithm={best_result['model_name']}\n")
        f.write(f"best_f1_score={best_result['best_metric']}\n")

    print("Best run:", best_result["run_id"], "| F1:", best_result["best_metric"])


if __name__ == "__main__":
    main()
