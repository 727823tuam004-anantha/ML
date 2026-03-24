# ROLLNO: 727823TUAM004
from datetime import datetime
import os
import json
import argparse
import joblib
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--model_input", type=str, required=True)
    parser.add_argument("--eval_output", type=str, required=True)
    args = parser.parse_args()

    print(f"ROLLNO: 727823TUAM004 | TIMESTAMP: {datetime.now().isoformat()}")

    df = pd.read_csv(os.path.join(args.test_data, "test.csv"))
    X_test = df.drop(columns=["target"])
    y_test = df["target"]

    model = joblib.load(os.path.join(args.model_input, "model.joblib"))

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob))
    }

    os.makedirs(args.eval_output, exist_ok=True)
    with open(os.path.join(args.eval_output, "evaluation.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    print(metrics)


if __name__ == "__main__":
    main()
