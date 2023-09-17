import pandas as pd

def validate_actuals(df: pd.DataFrame, upload_date: str):
    expected_columns = [
        "Time", "Load (kW)", "Pressure_kpa", "Cloud Cover (%)", "Humidity (%)", 
        "Temperature (C)", "Wind Direction (deg)", "Wind Speed (kmh)"
    ]

    # 1. Date Validation
    df["Time"] = pd.to_datetime(df["Time"])
    start_date = pd.to_datetime(upload_date) - pd.Timedelta(hours=16)
    end_date = pd.to_datetime(upload_date) + pd.Timedelta(hours=7)
    assert df["Time"].min() == start_date and df["Time"].max() == end_date, f"Invalid date range in data. Expected: {start_date} to {end_date}"
    assert len(df) == 24, f"Data should have 24 hours worth of data. Found: {len(df)} hours"

    # 2. Column Name Validation
    assert set(df.columns) == set(expected_columns), f"Column mismatch. Found: {df.columns}, Expected: {expected_columns}"

    # 3. Value Range Validation
    valid_ranges = {
        "Temperature (C)": (-50, 50),
        "Pressure_kpa": (900, 1100),
        "Cloud Cover (%)": (0, 100),
        "Humidity (%)": (0, 100),
        "Wind Direction (deg)": (0, 360),
        "Wind Speed (kmh)": (0, 200),
        "Load (kW)": (0, float('inf'))
    }
    for col, (min_val, max_val) in valid_ranges.items():
        assert df[col].between(min_val, max_val).all(), f"Invalid values in {col}"

    # 4. Check for Extra Columns
    for col in df.columns:
        assert col in expected_columns, f"Unexpected column found: {col}"

    # 5. Check for Missing Values
    assert not df.isnull().any().any(), f"Missing values detected: {df.isnull().sum()}"

    # 6. Extreme Differences Between Hours
    diff_df = df.diff().dropna()
    for col in ["Temperature (C)", "Load (kW)", "Pressure_kpa"]:
        assert (diff_df[col].abs() < df[col].std() * 3).all(), f"Extreme change detected in {col} between consecutive hours."

    # 7. Check for Constant Values
    for col in expected_columns:
        assert len(df[col].unique()) > 1, f"All values in {col} are the same."

    return "Actuals data validation passed!"


def validate_forecasts(df: pd.DataFrame, upload_date: str):
    expected_columns = [
        "Time", "Pressure_kpa", "Cloud Cover (%)", 
        "Temperature (C)", "Wind Direction (deg)", "Wind Speed (kmh)"
    ]

    # 1. Date Validation
    df["Time"] = pd.to_datetime(df["Time"])
    start_date = pd.to_datetime(upload_date) + pd.Timedelta(hours=32)
    end_date = pd.to_datetime(upload_date) + pd.Timedelta(hours=55)
    assert df["Time"].min() == start_date and df["Time"].max() == end_date, f"Invalid date range in data. Expected: {start_date} to {end_date}"
    assert len(df) == 24, f"Data should have 24 hours worth of data. Found: {len(df)} hours"

    # 2. Column Name Validation
    assert set(df.columns) == set(expected_columns), f"Column mismatch. Found: {df.columns}, Expected: {expected_columns}"

    # 3. Value Range Validation
    valid_ranges = {
        "Temperature (C)": (-50, 50),
        "Pressure_kpa": (900, 1100),
        "Cloud Cover (%)": (0, 100),
        "Wind Direction (deg)": (0, 360),
        "Wind Speed (kmh)": (0, 200)
        }
    for col, (min_val, max_val) in valid_ranges.items():
        assert df[col].between(min_val, max_val).all(), f"Invalid values in {col}"

    # 4. Check for Extra Columns
    for col in df.columns:
        assert col in expected_columns, f"Unexpected column found: {col}"

    # 5. Check for Missing Values
    assert not df.isnull().any().any(), f"Missing values detected: {df.isnull().sum()}"

    # 6. Extreme Differences Between Hours
    diff_df = df.diff().dropna()
    for col in ["Temperature (C)", "Pressure_kpa"]:
        assert (diff_df[col].abs() < df[col].std() * 3).all(), f"Extreme change detected in {col} between consecutive hours."

    # 7. Check for Constant Values
    for col in expected_columns:
        assert len(df[col].unique()) > 1, f"All values in {col} are the same."

    return "Forecasts data validation passed!"