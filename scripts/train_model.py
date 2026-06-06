from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "Life Expectancy Data (1).csv"
MODEL_PATH = ROOT / "models" / "life_expectancy_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"

TARGET = "Life expectancy "
STATUS_COLUMN = "Status"
OUTLIER_COLUMNS = [
    "Adult Mortality",
    "infant deaths",
    "Alcohol",
    "percentage expenditure",
    "Hepatitis B",
    "Measles ",
    " BMI ",
    "under-five deaths ",
    "Polio",
    "Total expenditure",
    "Diphtheria ",
    " HIV/AIDS",
    "GDP",
    "Population",
    " thinness  1-19 years",
    " thinness 5-9 years",
    "Income composition of resources",
    "Schooling",
]


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, LabelEncoder], dict[str, float]]:
    df = df.copy()
    numeric_columns = df.select_dtypes(include=["float64", "int64"]).columns
    imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    fill_values: dict[str, float] = {}

    for column in numeric_columns:
        if df[column].isnull().sum() > 0:
            df[column] = imputer.fit_transform(df[[column]])
            fill_values[column] = float(imputer.statistics_[0])
        else:
            fill_values[column] = float(df[column].mean())

    for column in OUTLIER_COLUMNS:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        mean_value = df[column].mean()
        df[column] = np.where(
            (df[column] > upper_bound) | (df[column] < lower_bound),
            mean_value,
            df[column],
        )

    encoders: dict[str, LabelEncoder] = {}
    for column in df.select_dtypes(include="object").columns:
        encoder = LabelEncoder()
        df[column] = encoder.fit_transform(df[column])
        encoders[column] = encoder

    return df, encoders, fill_values


def train() -> None:
    raw_df = pd.read_csv(DATASET_PATH)
    cleaned_df, encoders, fill_values = clean_dataset(raw_df)

    x = cleaned_df.drop(columns=TARGET)
    y = cleaned_df[TARGET]

    scaler = StandardScaler()
    columns_to_scale = x.drop(columns=STATUS_COLUMN).columns.tolist()
    x[columns_to_scale] = scaler.fit_transform(x[columns_to_scale])

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=30
    )

    model = XGBRegressor(
        n_estimators=350,
        learning_rate=0.04,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        objective="reg:squarederror",
    )
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)

    feature_importance = (
        pd.DataFrame(
            {
                "feature": x.columns,
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    artifact = {
        "model": model,
        "scaler": scaler,
        "encoders": encoders,
        "feature_columns": x.columns.tolist(),
        "columns_to_scale": columns_to_scale,
        "fill_values": fill_values,
        "medians": raw_df.select_dtypes(include=["float64", "int64"]).median().to_dict(),
        "means": raw_df.select_dtypes(include=["float64", "int64"]).mean().to_dict(),
        "dataset_rows": int(raw_df.shape[0]),
        "dataset_columns": int(raw_df.shape[1]),
        "target": TARGET,
        "feature_importance": feature_importance.to_dict(orient="records"),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)

    metrics = {
        "model": "XGBRegressor",
        "r2_score": float(r2_score(y_test, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, prediction))),
        "mae": float(mean_absolute_error(y_test, prediction)),
        "train_rows": int(x_train.shape[0]),
        "test_rows": int(x_test.shape[0]),
        "feature_count": int(x.shape[1]),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    train()
