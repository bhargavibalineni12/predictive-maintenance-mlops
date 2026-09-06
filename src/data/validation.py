import pandas as pd


EXPECTED_COLUMNS = [
    "Engine rpm",
    "Lub oil pressure",
    "Fuel pressure",
    "Coolant pressure",
    "lub oil temp",
    "Coolant temp",
    "Engine Condition",
]


def validate_columns(df: pd.DataFrame) -> None:
    """
    Validate that all expected columns are present in the dataset.
    """

    missing_columns = [
        column for column in EXPECTED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    print("Column validation passed.")


def validate_missing_values(df: pd.DataFrame) -> None:
    """
    Check whether the dataset contains missing values.
    """

    missing_values = df.isnull().sum()

    columns_with_missing_values = missing_values[
        missing_values > 0
    ]

    if not columns_with_missing_values.empty:
        raise ValueError(
            f"Missing values found:\n{columns_with_missing_values}"
        )

    print("Missing value validation passed.")

def validate_target_values(df: pd.DataFrame) -> None:
    """
    Validate that the target column contains only expected binary values.
    """

    expected_values = {0, 1}

    actual_values = set(df["Engine Condition"].dropna().unique())

    unexpected_values = actual_values - expected_values

    if unexpected_values:
        raise ValueError(
            f"Unexpected target values found: {unexpected_values}"
        )

    print("Target value validation passed.")

def validate_numeric_features(df: pd.DataFrame) -> None:
    """
    Validate that all sensor feature columns are numeric.
    """

    feature_columns = [
        "Engine rpm",
        "Lub oil pressure",
        "Fuel pressure",
        "Coolant pressure",
        "lub oil temp",
        "Coolant temp",
    ]

    non_numeric_columns = [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(df[column])
    ]

    if non_numeric_columns:
        raise ValueError(
            f"Non-numeric feature columns found: {non_numeric_columns}"
        )

    print("Numeric feature validation passed.")

def validate_duplicates(df: pd.DataFrame) -> None:
    """
    Check whether the dataset contains duplicate rows.
    """

    duplicate_count = df.duplicated().sum()

    if duplicate_count > 0:
        print(f"Warning: {duplicate_count} duplicate rows found.")
    else:
        print("Duplicate validation passed. No duplicate rows found.")
