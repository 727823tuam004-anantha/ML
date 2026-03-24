import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[df["readmitted"].notna()]
    df["target"] = (df["readmitted"] == "<30").astype(int)
    return df


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.replace("?", np.nan, inplace=True)
    drop_cols = [c for c in ["encounter_id", "patient_nbr", "readmitted"] if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)
    if "weight" in df.columns and df["weight"].isna().mean() > 0.9:
        df.drop(columns=["weight"], inplace=True)
    return df


def get_feature_lists(df: pd.DataFrame):
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in ["target"]:
        if col in numeric_cols:
            numeric_cols.remove(col)
        if col in categorical_cols:
            categorical_cols.remove(col)
    return categorical_cols, numeric_cols


def build_preprocessor(X: pd.DataFrame):
    categorical_cols, numeric_cols = get_feature_lists(X)
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ])
    return preprocessor, categorical_cols, numeric_cols


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_joblib(obj, path):
    joblib.dump(obj, path)


def load_joblib(path):
    return joblib.load(path)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
