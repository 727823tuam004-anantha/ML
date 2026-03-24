# ROLLNO: 727823TUAM004
from datetime import datetime
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, required=True)
    parser.add_argument("--train_output", type=str, required=True)
    parser.add_argument("--test_output", type=str, required=True)
    args = parser.parse_args()

    print(f"ROLLNO: 727823TUAM004 | TIMESTAMP: {datetime.now().isoformat()}")

    df = pd.read_csv(args.input_data)
    df.replace("?", np.nan, inplace=True)
    df = df[df["readmitted"].notna()].copy()
    df["target"] = (df["readmitted"] == "<30").astype(int)

    drop_cols = [c for c in ["encounter_id", "patient_nbr", "readmitted"] if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)

    if "weight" in df.columns and df["weight"].isna().mean() > 0.9:
        df.drop(columns=["weight"], inplace=True)

    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["target"], random_state=42
    )

    os.makedirs(args.train_output, exist_ok=True)
    os.makedirs(args.test_output, exist_ok=True)

    train_df.to_csv(os.path.join(args.train_output, "train.csv"), index=False)
    test_df.to_csv(os.path.join(args.test_output, "test.csv"), index=False)

    print(f"Train rows: {len(train_df)} | Test rows: {len(test_df)}")


if __name__ == "__main__":
    main()
