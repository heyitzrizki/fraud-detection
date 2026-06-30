from pathlib import Path

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _build_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    scale_numeric: bool = False,
) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler(with_mean=False)))

    numeric_transformer = Pipeline(steps=numeric_steps)
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", _build_one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )


def calculate_scale_pos_weight(y_train) -> float:
    positives = np.sum(y_train == 1)
    negatives = np.sum(y_train == 0)
    return float(negatives / positives) if positives else 1.0


def train_model(X_train, y_train, model_type: str = "lightgbm", random_state: int = 42):
    if model_type == "lightgbm":
        model = LGBMClassifier(
            objective="binary",
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=calculate_scale_pos_weight(y_train),
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
    elif model_type == "logistic_regression":
        model = LogisticRegression(
            class_weight="balanced",
            max_iter=500,
            n_jobs=-1,
            random_state=random_state,
        )
    elif model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=150,
            class_weight="balanced_subsample",
            min_samples_leaf=10,
            n_jobs=-1,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    model.fit(X_train, y_train)
    return model


def save_artifact(artifact, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)


def load_artifact(path: str | Path):
    return joblib.load(path)
