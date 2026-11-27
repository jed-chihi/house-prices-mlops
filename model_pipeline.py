"""Modular ML pipeline for Tunisian house price prediction."""

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "price_tnd"
# price_eur is price_tnd * 0.31 (leakage); id and Unnamed: 0 carry no information
DROP_COLS = ["price_eur", "id", "Unnamed: 0"]


def prepare_data(csv_path, target=TARGET, test_size=0.2, random_state=42):
    """Load the CSV, clean it and return X_train, X_test, y_train, y_test."""
    df = pd.read_csv(csv_path).drop_duplicates()
    df = df.dropna(subset=[target])
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def _build_preprocessor(X):
    """Impute + scale numeric columns, impute + one-hot encode categorical ones."""
    num_cols = X.select_dtypes(include="number").columns
    cat_cols = X.select_dtypes(exclude="number").columns
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer([("num", numeric, num_cols), ("cat", categorical, cat_cols)])


def train_model(X_train, y_train, n_estimators=200, random_state=42):
    """Train a preprocessing + RandomForest pipeline and return it."""
    model = Pipeline(
        [
            ("preprocessor", _build_preprocessor(X_train)),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=n_estimators, random_state=random_state, n_jobs=-1
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Return RMSE, MAE and R2 on the test set."""
    preds = model.predict(X_test)
    return {
        "rmse": mean_squared_error(y_test, preds) ** 0.5,
        "mae": mean_absolute_error(y_test, preds),
        "r2": r2_score(y_test, preds),
    }


def save_model(model, path="model.joblib"):
    """Save the trained model with joblib."""
    joblib.dump(model, path)


def load_model(path="model.joblib"):
    """Load a saved model."""
    return joblib.load(path)
