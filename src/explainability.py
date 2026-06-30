import numpy as np
import pandas as pd
import shap


def calculate_shap_values(
    model,
    preprocessor,
    X_sample: pd.DataFrame,
):
    transformed_sample = preprocessor.transform(X_sample)
    feature_names = preprocessor.get_feature_names_out()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(transformed_sample)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    return np.asarray(shap_values), feature_names


def clean_feature_name(feature_name: str, categorical_features: list[str] | None = None) -> str:
    clean_name = feature_name.replace("num__", "").replace("cat__", "")

    if categorical_features:
        for column in categorical_features:
            if clean_name == column or clean_name.startswith(f"{column}_"):
                return column

    return clean_name


def get_top_shap_reasons(
    shap_values_row,
    feature_names,
    top_n: int = 3,
    categorical_features: list[str] | None = None,
) -> list[str]:
    ranked_indices = np.argsort(np.abs(shap_values_row))[::-1]
    reasons = []

    for index in ranked_indices:
        reason = clean_feature_name(feature_names[index], categorical_features)
        if reason not in reasons:
            reasons.append(reason)
        if len(reasons) == top_n:
            break

    return reasons


def build_reason_lookup(
    transaction_ids,
    shap_values,
    feature_names,
    top_n: int = 3,
    categorical_features: list[str] | None = None,
) -> dict:
    return {
        int(transaction_id): get_top_shap_reasons(
            shap_values_row,
            feature_names,
            top_n=top_n,
            categorical_features=categorical_features,
        )
        for transaction_id, shap_values_row in zip(transaction_ids, shap_values)
    }
