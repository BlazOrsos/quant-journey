# Method for saving data to CSV from a DataFrame
"""Helper utilities for common DataFrame I/O and date operations.

This module provides:
- save_df_to_csv: Robust CSV writer with optional directory creation.
- load_df_from_csv: Flexible CSV reader with parse_dates and index support.
- find_missing_dates: Identify missing dates from the last date in data up to today (or end date).
"""

from __future__ import annotations

import os
from typing import Iterable, Optional, Union

import pandas as pd


# Method for saving data to CSV from a DataFrame
def save_df_to_csv(
    df: pd.DataFrame,
    file_path: str,
    *,
    index: bool = False,
    create_dirs: bool = True,
    date_format: Optional[str] = None,
    float_format: Optional[str] = None,
    mode: str = "w",
    na_rep: Optional[str] = None,
    **kwargs,
) -> None:
    """Save a DataFrame to CSV with a few safety/ergonomic improvements.

    Parameters
    - df: DataFrame to write
    - file_path: Destination CSV path
    - index: Whether to write the index
    - create_dirs: Create parent directories if missing
    - date_format: strftime format for datetimes
    - float_format: Format string for floats (e.g., '%.6f')
    - mode: File write mode ('w' to overwrite, 'a' to append)
    - na_rep: Representation for NaNs
    - kwargs: Passed through to pandas.DataFrame.to_csv

    Raises
    - ValueError: If df is not a pandas DataFrame
    - OSError: On I/O errors when writing the file
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")

    parent = os.path.dirname(os.path.abspath(file_path))
    if create_dirs and parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

    df.to_csv(
        file_path,
        index=index,
        date_format=date_format,
        float_format=float_format,
        mode=mode,
        na_rep=na_rep,
        **kwargs,
    )

# Method for loading data from CSV
def load_df_from_csv(
    file_path: str,
    *,
    parse_dates: Optional[Union[bool, list[int], list[str], dict[str, list[str]]]] = None,
    index_col: Optional[Union[int, str, list[int], list[str]]] = None,
    dtype: Optional[dict[str, str]] = None,
    engine: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    """Load a CSV into a DataFrame with optional parsing and index handling.

    Parameters
    - file_path: Source CSV path
    - parse_dates: Passed to pandas.read_csv for datetime parsing
    - index_col: Column(s) to set as index
    - dtype: Optional dtype mapping
    - engine: CSV engine (e.g., 'c' or 'python')
    - kwargs: Any additional pandas.read_csv arguments

    Returns
    - pd.DataFrame: The loaded dataframe

    Raises
    - FileNotFoundError: If the path does not exist
    - pd.errors.ParserError: If pandas fails to parse the CSV
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV not found at '{file_path}'")

    return pd.read_csv(
        file_path,
        parse_dates=parse_dates,
        index_col=index_col,
        dtype=dtype,
        engine=engine,
        **kwargs,
    )

# Method for getting the last date from a CSV file
def get_last_date_from_csv(
    file_path: str,
    *,
    date_col: Optional[str] = None,
    index_col: Optional[Union[int, str]] = None,
    parse_dates: Optional[Union[bool, list[int], list[str]]] = None,
) -> Optional[pd.Timestamp]:
        """Get the most recent date from a CSV file.

        Parameters
        - file_path: Path to the CSV file
        - date_col: Name of the date column to check (if not using index)
        - index_col: Column to use as index (typically the date column)
        - parse_dates: Columns to parse as dates

        Returns
        - pd.Timestamp: The most recent date, or None if no dates found

        Raises
        - FileNotFoundError: If the CSV file doesn't exist
        - ValueError: If no valid dates are found
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV not found at '{file_path}'")

        df = pd.read_csv(file_path, index_col=index_col, parse_dates=parse_dates)

        if df.empty:
            return None

        # Check if using a date column
        if date_col is not None:
            if date_col not in df.columns:
                raise ValueError(f"Column '{date_col}' not found in CSV")
            dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
        elif isinstance(df.index, pd.DatetimeIndex):
            dates = df.index.dropna()
        else:
            raise ValueError("Provide date_col or ensure index is a DatetimeIndex")

        if dates.empty:
            return None

        return pd.Timestamp(dates.max())