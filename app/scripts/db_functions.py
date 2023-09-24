import csv
import psycopg2
import pandas as pd
import pytz
import zipfile
import os
import datetime 


class PostgreSQLUploader:

    def __init__(self):
        self.conn_params = {
            'dbname': 'elec_db',
            'user': "ds4user",
            'password': "FIT3163!",
            'host': 'ds4db.postgres.database.azure.com',
            'port': 5432
        }
        self.conn = None
        self.cur = None

    def _connect(self):
        """Create a database connection and cursor."""
        self.conn = psycopg2.connect(**self.conn_params)
        self.cur = self.conn.cursor()

    def _disconnect(self):
        """Close the cursor and connection."""
        if self.cur:
            self.cur.close()
        if self.conn:
            self.conn.close()

    def fetch_to_dataframe(self, table_name, query=None):
        """
        Fetch data from the database table and convert to a pandas DataFrame.

        table_name: the name of the table to fetch data from.
        query: optional custom SQL query to fetch data. If provided, table_name will be ignored.
        """
        self._connect()

        # If a custom query is provided, use it; otherwise, fetch all from the table.
        sql = query or f"SELECT * FROM {table_name}"

        df = pd.read_sql(sql, self.conn)
        self._disconnect()
        return df
    
    def unzip_for_files(zip_path):
        # List to store paths of unzipped files
        extracted_files = []
        
        # Ensure the zip file exists
        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"The specified zip file '{zip_path}' does not exist.")

        # Define the directory to unzip the files into (same as the zip file's directory)
        extract_dir = os.path.dirname(zip_path)

        # Open the zip file and extract all
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
            # Add the full paths of the extracted files to the list
            for file_name in zip_ref.namelist():
                extracted_files.append(os.path.join(extract_dir, file_name))
        
        return extracted_files

    def upload_raw_actuals_csv(self, csv_file, table_name="elec_actuals"):
        """
        Uploads data from a CSV to a given table in the database.

        csv_file: the CSV file path
        table_name: the database table name to upload to
        column_mapping: dictionary mapping CSV columns to DB columns
        """
        

        column_mapping = {
            "Time": "time",
            "Load (kW)": "load_kw",
            "Pressure_kpa": "pressure_kpa",
            "Cloud Cover (%)": "cloud_cover_pct",
            "Humidity (%)": "humidity_pct",
            "Temperature (C) ": "temperature_c",
            "Temperature (C)": "temperature_c",
            "Wind Direction (deg)": "wind_direction_deg",
            "Wind Speed (kmh)": "wind_speed_kmh"
        }

        self._connect()
        with open(csv_file, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Check if all values in the row are empty
                if row["Time"].strip():
                    # Convert the 'time' string to datetime
                    row["Time"] = datetime.datetime.strptime(row["Time"], "%Y-%m-%d %H:%M:%S")

                    # Convert the other columns to appropriate numeric types
                    row["Load (kW)"] = float(row["Load (kW)"])
                    row["Pressure_kpa"] = float(row["Pressure_kpa"])
                    row["Cloud Cover (%)"] = float(row["Cloud Cover (%)"])
                    row["Humidity (%)"] = float(row["Humidity (%)"])
                    row["Temperature (C)"] = float(row["Temperature (C)"])
                    row["Wind Direction (deg)"] = float(row["Wind Direction (deg)"])
                    row["Wind Speed (kmh)"] = float(row["Wind Speed (kmh)"])
                    
                    # Map CSV columns to DB columns
                    mapped_row = {column_mapping[key]: value for key, value in row.items()}
                    # Dynamic SQL to insert data into the desired table
                    columns = ', '.join(mapped_row.keys())
                    values = ', '.join(['%(' + col + ')s' for col in mapped_row.keys()])
                    sql = f'INSERT INTO {table_name} ({columns}) VALUES ({values})'
                    self.cur.execute(sql, mapped_row)
            self.conn.commit()
        self._disconnect()

    def upload_raw_forecasts_csv(self, csv_file, table_name="elec_forecasts"):
        """
        Uploads data from a CSV to a given table in the database.

        csv_file: the CSV file path
        table_name: the database table name to upload to
        column_mapping: dictionary mapping CSV columns to DB columns
        """
        

        column_mapping = {
            "Time": "time",
            "Pressure_kpa": "pressure_kpa",
            "Cloud Cover (%)": "cloud_cover_pct",
            "Temperature (C) ": "temperature_c",
            "Temperature (C)": "temperature_c",
            "Wind Direction (deg)": "wind_direction_deg",
            "Wind Speed (kmh)": "wind_speed_kmh"
        }

        self._connect()
        with open(csv_file, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Check if all values in the row are empty
                if row["Time"].strip():
                    # Convert the 'time' string to datetime
                    row["Time"] = datetime.datetime.strptime(row["Time"], "%Y-%m-%d %H:%M:%S")

                    # Convert the other columns to appropriate numeric types
                    row["Pressure_kpa"] = float(row["Pressure_kpa"])
                    row["Cloud Cover (%)"] = float(row["Cloud Cover (%)"])
                    row["Temperature (C)"] = float(row["Temperature (C)"])
                    row["Wind Direction (deg)"] = float(row["Wind Direction (deg)"])
                    row["Wind Speed (kmh)"] = float(row["Wind Speed (kmh)"])
                    
                    # Map CSV columns to DB columns
                    mapped_row = {column_mapping[key]: value for key, value in row.items()}
                    # Dynamic SQL to insert data into the desired table
                    columns = ', '.join(mapped_row.keys())
                    values = ', '.join(['%(' + col + ')s' for col in mapped_row.keys()])
                    sql = f'INSERT INTO {table_name} ({columns}) VALUES ({values})'
                    self.cur.execute(sql, mapped_row)
            self.conn.commit()
        self._disconnect()

    def upload_model_forecasts(self, df):
        """
        Upload forecasts from the dataframe into the elec_forecasts table.
        
        df: a DataFrame with columns time and forecast
        """
        self._connect()

        # Ensure the dataframe has exactly 48 rows
        if len(df) != 48:
            raise ValueError("Expected dataframe with 48 rows")

        # Split the dataframe into two: first 24 rows and next 24 rows
        df_forecast_1 = df.iloc[24:]
        df_forecast_2 = df.iloc[:24]

        # Update forecast_1 for the first 24 rows
        for _, row in df_forecast_1.iterrows():
            time, forecast = row['time'], row['forecasts']
            sql = """
            UPDATE elec_forecasts
            SET forecast_1 = %s
            WHERE time = %s
            """
            self.cur.execute(sql, (forecast, time))

        # Update forecast_2 for the next 24 rows
        for _, row in df_forecast_2.iterrows():
            time, forecast = row['time'], row['forecasts']
            sql = """
            UPDATE elec_forecasts
            SET forecast_2 = %s
            WHERE time = %s
            """
            self.cur.execute(sql, (forecast, time))
            
        # Commit the changes
        self.conn.commit()
        self._disconnect()


def fetch_actuals_from_db(reference_date):

        # Dates for 8AM one week ago and 7AM of the reference_date
        start_datetime = (pd.to_datetime(reference_date) - pd.Timedelta(days=7)).replace(hour=8, minute=0, second=0)
        end_datetime = pd.to_datetime(reference_date).replace(hour=7, minute=0, second=0)
        start_date = start_datetime.strftime('%Y-%m-%d %H:%M:%S')
        end_date = end_datetime.strftime('%Y-%m-%d %H:%M:%S')

        # Set your database connection parameters
        db_params = {
            'dbname': 'elec_db',
            'user': 'ds4user',
            'password': 'FIT3163!',
            'host': 'ds4db.postgres.database.azure.com',
            'port': '5432'
        }

        # Connect to the database
        conn = psycopg2.connect(**db_params)
        
        # Use pandas to fetch data and convert it to a DataFrame
        query = f"""
            SELECT time, load_kw, pressure_kpa, cloud_cover_pct, temperature_c, wind_direction_deg, wind_speed_kmh 
            FROM elec_actuals 
            WHERE time >= '{start_date}' 
            AND time <= '{end_date}'
            ORDER BY time ASC
        """
        df = pd.read_sql(query, conn)

        # Convert the 'time' column to pandas datetime
        df['time'] = pd.to_datetime(df['time'])

        # Close the database connection
        conn.close()

        return df

def fetch_actuals_from_db_for_retraining():

        # Set your database connection parameters
        db_params = {
            'dbname': 'elec_db',
            'user': 'ds4user',
            'password': 'FIT3163!',
            'host': 'ds4db.postgres.database.azure.com',
            'port': '5432'
        }

        # Connect to the database
        conn = psycopg2.connect(**db_params)
        
        # Use pandas to fetch data and convert it to a DataFrame
        query = f"""
            SELECT time, load_kw, pressure_kpa, cloud_cover_pct, temperature_c, wind_direction_deg, wind_speed_kmh 
            FROM elec_actuals 
            ORDER BY time ASC
        """
        df = pd.read_sql(query, conn)

        # Convert the 'time' column to pandas datetime
        df['time'] = pd.to_datetime(df['time'])

        # Close the database connection
        conn.close()

        return df

def fetch_forecasts_from_db(reference_date):

        # Dates for 8AM one week ago and 7AM of the reference_date
        start_datetime = pd.to_datetime(reference_date).replace(hour=8, minute=0, second=0)
        end_datetime = (pd.to_datetime(reference_date) + pd.Timedelta(days=2)).replace(hour=7, minute=0, second=0)

        start_date = start_datetime.strftime('%Y-%m-%d %H:%M:%S')
        end_date = end_datetime.strftime('%Y-%m-%d %H:%M:%S')

        # Set your database connection parameters
        db_params = {
            'dbname': 'elec_db',
            'user': 'ds4user',
            'password': 'FIT3163!',
            'host': 'ds4db.postgres.database.azure.com',
            'port': '5432'
        }

        # Connect to the database
        conn = psycopg2.connect(**db_params)
        
        # Use pandas to fetch data and convert it to a DataFrame
        query = f"""
            SELECT time, pressure_kpa, cloud_cover_pct, temperature_c, wind_direction_deg, wind_speed_kmh 
            FROM elec_forecasts
            WHERE time >= '{start_date}' 
            AND time <= '{end_date}'
            ORDER BY time ASC
        """
        df = pd.read_sql(query, conn)

        # Convert the 'time' column to pandas datetime
        df['time'] = pd.to_datetime(df['time'])

        # Close the database connection
        conn.close()

        return df

def fetch_actuals_from_db_for_insights(start_date, end_date):
    melbourne_tz = pytz.timezone('Australia/Melbourne')

    # Convert to Melbourne timezone for consistency
    start_datetime = pd.to_datetime(start_date).tz_localize(melbourne_tz)
    end_datetime = pd.to_datetime(end_date).tz_localize(melbourne_tz)

    # Set your database connection parameters
    db_params = {
        'dbname': 'elec_db',
        'user': 'ds4user',
        'password': 'FIT3163!',  # Please be cautious about hardcoding passwords in your code
        'host': 'ds4db.postgres.database.azure.com',
        'port': '5432'
    }

    # Connect to the database
    conn = psycopg2.connect(**db_params)

    # Use pandas to fetch data and convert it to a DataFrame
    query = f"""
        SELECT time, load_kw
        FROM elec_actuals 
        WHERE time >= '{start_datetime}' 
        AND time <= '{end_datetime}'
        ORDER BY time ASC
    """
    df = pd.read_sql(query, conn)

    # Convert the 'time' column to pandas datetime
    df['time'] = pd.to_datetime(df['time'])

    # Close the database connection
    conn.close()

    return df


def fetch_forecasts_from_db_for_insights(start_date, end_date):
    melbourne_tz = pytz.timezone('Australia/Melbourne')

    # Convert to Melbourne timezone for consistency
    start_datetime = pd.to_datetime(start_date).tz_localize(melbourne_tz)
    end_datetime = pd.to_datetime(end_date).tz_localize(melbourne_tz)

    # Set your database connection parameters
    db_params = {
        'dbname': 'elec_db',
        'user': 'ds4user',
        'password': 'FIT3163!',  # Please be cautious about hardcoding passwords in your code
        'host': 'ds4db.postgres.database.azure.com',
        'port': '5432'
    }

    # Connect to the database
    conn = psycopg2.connect(**db_params)

    # Use pandas to fetch data and convert it to a DataFrame
    query = f"""
        SELECT time, forecast_1, forecast_2
        FROM elec_forecasts
        WHERE time >= '{start_datetime}' 
        AND time <= '{end_datetime}'
        ORDER BY time ASC
    """
    df = pd.read_sql(query, conn)

    # Convert the 'time' column to pandas datetime
    df['time'] = pd.to_datetime(df['time'])

    # Close the database connection
    conn.close()

    return df

# Test
# data = fetch_data_from_actuals("2020-03-30")
# print(data)

uploader = PostgreSQLUploader()

#uploader.upload_raw_actuals_csv(f'upload_data/Actuals_Interim.csv', 'elec_actuals')
# uploader.upload_raw_forecasts_csv(f"upload_data/Forecasts_Feb {i} 8AM.csv")


