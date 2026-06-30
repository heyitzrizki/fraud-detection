import numpy as np
import pandas as pd


TARGET_COLUMN = "isFraud"
ID_COLUMN = "TransactionID"

BASE_FEATURE_COLUMNS = [
    "TransactionDT",
    "TransactionAmt",
    "ProductCD",
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "addr2",
    "dist1",
    "dist2",
    "P_emaildomain",
    "R_emaildomain",
]

ENGINEERED_COLUMNS = [
    "transaction_hour",
    "transaction_day",
    "log_transaction_amt",
    "is_high_amount",
    "missing_count",
    "email_domain_match",
    "card_missing_count",
    "address_missing_count",
]


def _range_columns(prefix: str, start: int, end: int) -> list[str]:
    return [f"{prefix}{number}" for number in range(start, end + 1)]


def select_features(data: pd.DataFrame, include_target: bool = True) -> pd.DataFrame:
    selected_columns = get_recommended_raw_columns(include_target=include_target)

    available_columns = [column for column in selected_columns if column in data.columns]
    return data.loc[:, available_columns].copy()


def get_recommended_raw_columns(include_target: bool = True) -> list[str]:
    selected_columns = (
        [ID_COLUMN]
        + BASE_FEATURE_COLUMNS
        + _range_columns("C", 1, 14)
        + _range_columns("D", 1, 15)
        + _range_columns("M", 1, 9)
        + _range_columns("V", 1, 50)
    )

    if include_target:
        selected_columns.append(TARGET_COLUMN)

    return selected_columns


def create_features(
    data: pd.DataFrame,
    amount_high_threshold: float | None = None,
) -> tuple[pd.DataFrame, float]:
    data = data.copy()

    if "TransactionDT" in data.columns:
        data["transaction_hour"] = (data["TransactionDT"] // 3600) % 24
        data["transaction_day"] = data["TransactionDT"] // (3600 * 24)

    if "TransactionAmt" in data.columns:
        amount_high_threshold = (
            float(data["TransactionAmt"].quantile(0.95))
            if amount_high_threshold is None
            else amount_high_threshold
        )
        data["log_transaction_amt"] = np.log1p(data["TransactionAmt"])
        data["is_high_amount"] = (data["TransactionAmt"] >= amount_high_threshold).astype(int)
    else:
        amount_high_threshold = 0.0 if amount_high_threshold is None else amount_high_threshold

    source_columns = [
        column
        for column in data.columns
        if column not in {TARGET_COLUMN, ID_COLUMN} and column not in ENGINEERED_COLUMNS
    ]
    data["missing_count"] = data[source_columns].isna().sum(axis=1)

    if {"P_emaildomain", "R_emaildomain"}.issubset(data.columns):
        data["email_domain_match"] = (
            data["P_emaildomain"].notna()
            & data["R_emaildomain"].notna()
            & (data["P_emaildomain"] == data["R_emaildomain"])
        ).astype(int)

    card_columns = [column for column in ["card1", "card2", "card3", "card4", "card5", "card6"] if column in data.columns]
    if card_columns:
        data["card_missing_count"] = data[card_columns].isna().sum(axis=1)

    address_columns = [column for column in ["addr1", "addr2", "dist1", "dist2"] if column in data.columns]
    if address_columns:
        data["address_missing_count"] = data[address_columns].isna().sum(axis=1)

    return data, amount_high_threshold


def identify_categorical_features(data: pd.DataFrame) -> list[str]:
    excluded_columns = {TARGET_COLUMN, ID_COLUMN}
    return [
        column
        for column in data.columns
        if column not in excluded_columns and pd.api.types.is_object_dtype(data[column])
    ]


def identify_numeric_features(data: pd.DataFrame) -> list[str]:
    excluded_columns = {TARGET_COLUMN, ID_COLUMN}
    return [
        column
        for column in data.columns
        if column not in excluded_columns and pd.api.types.is_numeric_dtype(data[column])
    ]


def get_model_feature_columns(data: pd.DataFrame) -> list[str]:
    return [column for column in data.columns if column not in {TARGET_COLUMN, ID_COLUMN}]
