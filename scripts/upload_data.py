import csv
import psycopg2
import pandas as pd

class PostgreSQLUploader:

    def __init__(self, host, dbname, user, password, port='5432'):
        self.conn_params = {
            'dbname': dbname,
            'user': user,
            'password': password,
            'host': host,
            'port': port
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

    def upload_raw_actuals_csv(self, csv_file, table_name):
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
            "Wind Direction (deg)": "wind_direction_deg",
            "Wind Speed (kmh)": "wind_speed_kmh"
        }

        self._connect()
        with open(csv_file, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Map CSV columns to DB columns
                mapped_row = {column_mapping[key]: value for key, value in row.items()}
                # Dynamic SQL to insert data into the desired table
                columns = ', '.join(mapped_row.keys())
                placeholders = ', '.join(['%(' + col + ')s' for col in mapped_row.keys()])
                sql = f'INSERT INTO {table_name} ({columns}) VALUES ({placeholders})'
                self.cur.execute(sql, mapped_row)
            self.conn.commit()
        self._disconnect()

uploader = PostgreSQLUploader(
    host='ds4db.postgres.database.azure.com',
    dbname='elec_db',
    user='ds4user',
    password='FIT3163!'
)

uploader.upload_raw_actuals_csv('data/elec_p4_dataset/Train/merged_actuals.csv', 'elec_actuals')


