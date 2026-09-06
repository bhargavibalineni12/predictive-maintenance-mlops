from pathlib import Path
import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the predictive maintenance dataset from a CSV file.

    Parameters
    ----------
    file_path : str
        Path to the raw engine dataset.

    Returns
    -------
    pd.DataFrame
        Loaded engine sensor dataset.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    df = pd.read_csv(path)

    print(f"Dataset loaded successfully. Shape: {df.shape}")

    return df