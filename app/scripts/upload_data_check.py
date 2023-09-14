import pandas as pd

def validate_actuals(df: pd.DataFrame, upload_date: str):
    # Expected columns
    expected_columns = ["Time", "Load (kW)", "Pressure_kpa", "Cloud Cover (%)", "Humidity (%)", "Temperature (C)", "Wind Direction (deg)", "Wind Speed (kmh)"]

    # 1. Check for typos in column names
    assert set(df.columns) == set(expected_columns), f"Column mismatch. Found: {df.columns}, Expected: {expected_columns}"

    # 2. Check for missing columns
    for col in expected_columns:
        assert col in df.columns, f"Missing column: {col}"

    # 3. Check for valid dates and 24 hours worth of data
    df["Time"] = pd.to_datetime(df["Time"])
    start_date = pd.to_datetime(upload_date) - pd.Timedelta(hours=16)
    end_date = start_date + pd.Timedelta(hours=23)
    assert df["Time"].min() == start_date and df["Time"].max() <= pd.to_datetime(upload_date) + pd.Timedelta(hours=7), f"Invalid date range in data. Expected: {start_date} to {end_date}"
    assert len(df) == 24, f"Data should have 24 hours worth of data. Found: {len(df)} hours"

    # 4. Check for invalid values in other columns
    assert df["Temperature (C)"].between(-50, 50).all(), "Invalid values in Temperature (C)"
    assert df["Pressure_kpa"].between(900, 1100).all(), "Invalid values in Pressure_kpa"
    assert df["Cloud Cover (%)"].between(0, 100).all(), "Invalid values in Cloud Cover (%)"
    assert df["Humidity (%)"].between(0, 100).all(), "Invalid values in Humidity (%)"
    assert df["Wind Direction (deg)"].between(0, 360).all(), "Invalid values in Wind Direction (deg)"
    assert df["Wind Speed (kmh)"].between(0, 200).all(), "Invalid values in Wind Speed (kmh)"
    assert df["Load (kW)"].ge(0).all(), "Invalid values in Load (kW)"

    # 5. Check for extra columns
    for col in df.columns:
        assert col in expected_columns, f"Unexpected column found: {col}"

    # 6. Check for missing values in any of the columns
    assert not df.isnull().any().any(), f"Missing values detected: {df.isnull().sum()}"

    
    # Additional Checks:
    # Check for duplicates
    assert not df.duplicated().any(), "Duplicated rows detected."
    # Check for any non-numeric values in the numeric columns
    numeric_cols = ["Load (kW)", "Pressure_kpa", "Cloud Cover (%)", "Humidity (%)", "Temperature (C)", "Wind Direction (deg)", "Wind Speed (kmh)"]
    for col in numeric_cols:
        assert pd.api.types.is_numeric_dtype(df[col]), f"Non-numeric values detected in {col}"
    # Past Data Check
    assert df["Time"].max() <= pd.to_datetime(upload_date), "Actuals data should not contain future data."
    # Outliers Check
    for col in numeric_cols:
        mean_val = df[col].mean()
        std_val = df[col].std()
        assert df[col].between(mean_val - 3*std_val, mean_val + 3*std_val).all(), f"Outliers detected in {col}"
    # Consistency Check for Wind
    assert not ((df["Wind Speed (kmh)"] == 0) & (df["Wind Direction (deg)"].notna())).any(), "Inconsistent wind data detected."

    # Extreme Changes Between Consecutive Hours
    diff_df = df.diff().dropna()
    for col in ["Temperature (C)", "Load (kW)", "Pressure_kpa"]:
        assert (diff_df[col].abs() < df[col].std() * 3).all(), f"Extreme change detected in {col} between consecutive hours."

    # Constant Values
    for col in expected_columns:
        assert len(df[col].unique()) > 1, f"All values in {col} are the same."
    
    return "Actuals data validation passed!"


def validate_forecasts(df: pd.DataFrame, upload_date: str):
    # Expected columns
    expected_columns = ["Time", "Pressure_kpa", "Cloud Cover (%)", "Humidity (%)", "Temperature (C)", "Wind Direction (deg)", "Wind Speed (kmh)"]

    # 1. Check for typos in column names
    assert set(df.columns) == set(expected_columns), f"Column mismatch. Found: {df.columns}, Expected: {expected_columns}"

    # 2. Check for missing columns
    for col in expected_columns:
        assert col in df.columns, f"Missing column: {col}"

    # 3. Check for valid dates and 24 hours worth of data
    df["Time"] = pd.to_datetime(df["Time"])
    start_date = pd.to_datetime(upload_date) + pd.Timedelta(hours=8)
    end_date = start_date + pd.Timedelta(hours=23)
    assert df["Time"].min() == start_date and df["Time"].max() == end_date, f"Invalid date range in data. Expected: {start_date} to {end_date}"
    assert len(df) == 24, f"Data should have 24 hours worth of data. Found: {len(df)} hours"

    # 4. Check for invalid values in other columns
    assert df["Temperature (C)"].between(-50, 50).all(), "Invalid values in Temperature (C)"
    assert df["Pressure_kpa"].between(900, 1100).all(), "Invalid values in Pressure_kpa"
    assert df["Cloud Cover (%)"].between(0, 100).all(), "Invalid values in Cloud Cover (%)"
    assert df["Humidity (%)"].between(0, 100).all(), "Invalid values in Humidity (%)"
    assert df["Wind Direction (deg)"].between(0, 360).all(), "Invalid values in Wind Direction (deg)"
    assert df["Wind Speed (kmh)"].between(0, 200).all(), "Invalid values in Wind Speed (kmh)"
    assert df["Load (kW)"].ge(0).all(), "Invalid values in Load (kW)"

    # 5. Check for extra columns
    for col in df.columns:
        assert col in expected_columns, f"Unexpected column found: {col}"

    # 6. Check for missing values in any of the columns
    assert not df.isnull().any().any(), f"Missing values detected: {df.isnull().sum()}"

    # Additional Checks:
    # Check for duplicates
    assert not df.duplicated().any(), "Duplicated rows detected."
    # Check for any non-numeric values in the numeric columns
    numeric_cols = ["Load (kW)", "Pressure_kpa", "Cloud Cover (%)", "Humidity (%)", "Temperature (C)", "Wind Direction (deg)", "Wind Speed (kmh)"]
    for col in numeric_cols:
        assert pd.api.types.is_numeric_dtype(df[col]), f"Non-numeric values detected in {col}"
    # Extreme Changes Between Consecutive Hours
    diff_df = df.diff().dropna()
    for col in ["Temperature (C)", "Load (kW)", "Pressure_kpa"]:
        assert (diff_df[col].abs() < df[col].std() * 3).all(), f"Extreme change detected in {col} between consecutive hours."
    # Constant Values
    for col in expected_columns:
        assert len(df[col].unique()) > 1, f"All values in {col} are the same."

    return "Forecasts data validation passed!"