import pandas as pd
from sklearn.model_selection import train_test_split


TARGET_COLUMN = "Engine Condition"


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Split the dataset into training and testing sets.

    Parameters
    ----------
    df : pd.DataFrame
        Validated engine dataset.

    test_size : float
        Proportion of data to use for testing.

    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")

    return X_train, X_test, y_train, y_test