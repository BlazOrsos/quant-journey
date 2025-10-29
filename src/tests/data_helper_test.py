import pandas as pd
import numpy as np
import os

from src.helpers.data_helper import save_df_to_csv, load_df_from_csv, get_last_date_from_csv


def test_save_and_load_csv(tmp_path):
    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [0.1, 0.2, np.nan],
        "ts": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]) ,
    })

    out = tmp_path / "subdir" / "test.csv"
    # Save with directory auto-creation
    save_df_to_csv(df, str(out), index=False, create_dirs=True, date_format="%Y-%m-%d")
    assert out.exists()

    # Load with parse_dates
    df2 = load_df_from_csv(str(out), parse_dates=["ts"])
    # Ensure content roundtrips (types may differ for floats due to NaN)
    assert list(df2.columns) == ["a", "b", "ts"]
    assert len(df2) == 3
    assert pd.api.types.is_datetime64_any_dtype(df2["ts"])  # parsed as datetime

def test_get_last_date_from_csv(tmp_path):
    # Create a test CSV with dates
    df = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-04", "2020-01-06"]),
        "value": [100, 101, 103, 106]
    })
    
    csv_path = tmp_path / "test_dates.csv"
    save_df_to_csv(df, str(csv_path), index=False, create_dirs=True)
    
    # Test getting last date with date_col parameter
    last_date = get_last_date_from_csv(str(csv_path), date_col="date")
    assert last_date == pd.Timestamp("2020-01-06")
    
    # Test with index as date column
    df_indexed = df.set_index("date")
    csv_path_indexed = tmp_path / "test_dates_indexed.csv"
    save_df_to_csv(df_indexed, str(csv_path_indexed), index=True)
    
    last_date_indexed = get_last_date_from_csv(
        str(csv_path_indexed), 
        index_col=0, 
        parse_dates=True
    )
    assert last_date_indexed == pd.Timestamp("2020-01-06")
    
    # Test with empty CSV
    empty_df = pd.DataFrame({"date": pd.to_datetime([]), "value": []})
    empty_csv = tmp_path / "empty.csv"
    save_df_to_csv(empty_df, str(empty_csv), index=False)
    
    last_date_empty = get_last_date_from_csv(str(empty_csv), date_col="date")
    assert last_date_empty is None