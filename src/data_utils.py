from pathlib import Path

import numpy as np
import pandas as pd


def reduce_memory_usage(data: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns to reduce memory pressure on large CSV files."""
    data = data.copy()

    for column in data.columns:
        column_type = data[column].dtype

        if pd.api.types.is_integer_dtype(column_type):
            min_value = data[column].min()
            max_value = data[column].max()

            if min_value >= np.iinfo(np.int8).min and max_value <= np.iinfo(np.int8).max:
                data[column] = data[column].astype(np.int8)
            elif min_value >= np.iinfo(np.int16).min and max_value <= np.iinfo(np.int16).max:
                data[column] = data[column].astype(np.int16)
            elif min_value >= np.iinfo(np.int32).min and max_value <= np.iinfo(np.int32).max:
                data[column] = data[column].astype(np.int32)

        elif pd.api.types.is_float_dtype(column_type):
            data[column] = pd.to_numeric(data[column], downcast="float")

    return data


def optional_sample_data(
    data: pd.DataFrame,
    sample_size: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    if sample_size is None or sample_size >= len(data):
        return data

    return data.sample(n=sample_size, random_state=random_state).sort_values("TransactionDT")


def load_transaction_data(
    data_path: str | Path = "data/raw/train_transaction.csv",
    sample_size: int | None = None,
    random_state: int = 42,
    reduce_memory: bool = True,
    usecols: list[str] | None = None,
) -> pd.DataFrame:
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Could not find {data_path}. Place train_transaction.csv in data/raw/."
        )

    if usecols is not None:
        header = pd.read_csv(data_path, nrows=0).columns
        usecols = [column for column in usecols if column in header]

    data = pd.read_csv(data_path, usecols=usecols)

    if reduce_memory:
        data = reduce_memory_usage(data)

    data = optional_sample_data(data, sample_size=sample_size, random_state=random_state)
    return data.reset_index(drop=True)
