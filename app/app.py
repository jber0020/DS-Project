from flask import Flask, render_template, jsonify, request
import psycopg2
from werkzeug.utils import secure_filename
import os
import pandas as pd
import zipfile
from scripts.upload_data_check import validate_actuals, validate_forecasts
from datetime import datetime, timedelta  # Make sure to import this at the top of your file
from scripts.models import get_forecasts
from scripts.db_functions import PostgreSQLUploader, fetch_actuals_from_db, fetch_forecasts_from_db, fetch_actuals_from_db_for_insights, fetch_forecasts_from_db_for_insights, fetch_actuals_from_db_for_retraining
import numpy as np
from pytz import timezone
from scripts.models import retraining_required, retrain_model
from chatbot.chatbot import Chatbot


app = Flask(__name__)
# app.config.from_object('config')

host='ds4db.postgres.database.azure.com',
dbname='elec_db',
user='ds4user',
password='FIT3163!'

UPLOAD_FOLDER = 'upload_data/'  # replace with your desired upload folder
ALLOWED_EXTENSIONS = {'zip'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

db_manager = PostgreSQLUploader()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_date_from_filename(filename: str) -> str:
    # Extract the date part of the filename (e.g. 'Feb_05_8AM' from 'Actuals_Feb_05_8AM.zip')
    date_str = filename.split("_", 1)[1].rsplit('.', 1)[0]
    # Convert the extracted date to a datetime object
    date_obj = datetime.strptime(date_str, "%b_%d_%I%p")
    
    # Get the current year
    # current_year = datetime.now().year
    current_year = 2021
    
    # Set the correct year on the date_obj
    date_obj = date_obj.replace(year=current_year)
    
    # Format the datetime object to 'YYYY-MM-DD'
    formatted_date = date_obj.strftime('%Y-%m-%d')
    return formatted_date

def get_data_insights(forecasts, historical_df, current_time):
    if not isinstance(current_time, pd.Timestamp):
        current_time = pd.Timestamp(current_time)
    # Ensure the "time" column is of datetime64[ns] type
    historical_df["time"] = historical_df["time"].astype('datetime64[ns]')
    # Remove rows with NaN values in the "time" column
    historical_df = historical_df.dropna(subset=["time"])
    
    # Ensure the dataframes are sorted by time
    forecasts = forecasts.sort_values(by="time")
    historical_df = historical_df.sort_values(by="time")

    def safe_stat(data, stat_func):
        try:
            value = stat_func(data)
            # If the value is a number, round it to the nearest whole number
            if isinstance(value, (int, float)):
                return round(value)
            else:
                return value
        except:
            return "NA"

    # Use forecast_2 when available, else default to forecast_1
    forecasts["forecast"] = forecasts["forecast_2"].combine_first(forecasts["forecast_1"])

    # Define the period for "yesterday", "today", "tomorrow", based on 8AM to 7AM schedule
    start_time_today = pd.Timestamp(current_time.date(), tz=current_time.tz) + pd.Timedelta(hours=8)
    start_time_today = start_time_today.to_datetime64()  # Ensure datetime64[ns] type

    start_time_yesterday = start_time_today - pd.Timedelta(days=1)
    end_time_today = start_time_today + pd.Timedelta(days=1)

    # Extract statistics from yesterday
    yesterday_demand = historical_df[(historical_df["time"] >= start_time_yesterday) & (historical_df["time"] < start_time_today)]

    # Extract statistics for today and tomorrow from the forecasts
    today_demand = forecasts[(forecasts["time"] >= start_time_today) & (forecasts["time"] < end_time_today)]
    tomorrow_demand = forecasts[(forecasts["time"] >= end_time_today) & (forecasts["time"] < (end_time_today + pd.Timedelta(days=1)))]

    insights = {
        "yesterday_max": safe_stat(yesterday_demand["load_kw"], max),
        "yesterday_min": safe_stat(yesterday_demand["load_kw"], min),
        "yesterday_avg": safe_stat(yesterday_demand["load_kw"], pd.Series.mean),

        "today_max": safe_stat(today_demand["forecast"], max),
        "today_min": safe_stat(today_demand["forecast"], min),
        "today_avg": safe_stat(today_demand["forecast"], pd.Series.mean),

        "tomorrow_max": safe_stat(tomorrow_demand["forecast"], max),
        "tomorrow_min": safe_stat(tomorrow_demand["forecast"], min),
        "tomorrow_avg": safe_stat(tomorrow_demand["forecast"], pd.Series.mean)
    }

    # Define the period for "last week", "last month", and "last year"
    start_time_last_week = start_time_today - pd.Timedelta(weeks=1)
    start_time_last_month = start_time_today - pd.Timedelta(weeks=4)
    start_time_last_year = start_time_today - pd.Timedelta(weeks=52)

    # Calculate the average demand for the defined periods
    last_week_demand = historical_df[(historical_df["time"] >= start_time_last_week) & (historical_df["time"] < start_time_today)]
    last_month_demand = historical_df[(historical_df["time"] >= start_time_last_month) & (historical_df["time"] < start_time_today)]
    last_year_demand = historical_df[(historical_df["time"] >= start_time_last_year) & (historical_df["time"] < start_time_today)]

    insights["weekly_avg"] = safe_stat(last_week_demand["load_kw"], pd.Series.mean)
    insights["monthly_avg"] = safe_stat(last_month_demand["load_kw"], pd.Series.mean)
    insights["yearly_avg"] = safe_stat(last_year_demand["load_kw"], pd.Series.mean)

    return insights

def get_uploaded_file():
    if 'file' not in request.files:
        return None
    file = request.files['file']
    if file.filename == '':
        return None
    return file

def save_file(file, filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    return filepath

def process_zip_file(filepath, filename, extracted_date):
    if filename.endswith('.zip'):
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(app.config['UPLOAD_FOLDER'])

            actuals_filename = None
            forecasts_filename = None

            for name in zip_ref.namelist():
                if "Actuals_" in name:
                    actuals_filename = name
                elif "Forecasts_" in name:
                    forecasts_filename = name

            if not (actuals_filename and forecasts_filename):
                return jsonify({"error": "ZIP file doesn't contain the required files."})

            actuals_path = os.path.join(app.config['UPLOAD_FOLDER'], actuals_filename)
            forecasts_path = os.path.join(app.config['UPLOAD_FOLDER'], forecasts_filename)

            actuals = pd.read_csv(actuals_path)
            forecasts = pd.read_csv(forecasts_path)

            print(validate_actuals(actuals, extracted_date))
            print(validate_forecasts(forecasts, extracted_date))

            # Actuals file - Upload the latest actuals to db
            try:
                db_manager.upload_raw_actuals_csv(actuals_path)
            except psycopg2.errors.UniqueViolation:
                print("A row with a duplicate primary key was found and skipped.")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

            try:
                db_manager.upload_raw_forecasts_csv(forecasts_path)
            except psycopg2.errors.UniqueViolation:
                print("A row with a duplicate primary key was found and skipped.")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

def retrain_framework(extracted_date):
    start_datetime_actuals = (pd.to_datetime(extracted_date) - pd.Timedelta(days=35)).replace(hour=8, minute=0, second=0)
    end_datetime_actuals = pd.to_datetime(extracted_date).replace(hour=7, minute=0, second=0)

    start_datetime_forecasts = (pd.to_datetime(extracted_date) - pd.Timedelta(days=28)).replace(hour=8, minute=0, second=0)
    end_datetime_forecasts = pd.to_datetime(extracted_date).replace(hour=7, minute=0, second=0)

    retrain_check_data_actuals = fetch_actuals_from_db_for_insights(start_datetime_actuals, end_datetime_actuals)
    retrain_check_data_forecasts = fetch_forecasts_from_db_for_insights(start_datetime_forecasts, end_datetime_forecasts)
    retrain_check_data_forecasts = retrain_check_data_forecasts.rename(columns={'forecast_2': 'forecast'})

    # Convert 'time' column to datetime
    retrain_check_data_actuals['time'] = pd.to_datetime(retrain_check_data_actuals['time'])

    # Merge DataFrame with itself on the 'time' column offset by 7 days to get the lagged load
    merged_actual_df = retrain_check_data_actuals.merge(retrain_check_data_actuals, left_on='time', right_on=retrain_check_data_actuals['time'] + pd.Timedelta(days=7), suffixes=('', '_lag'))
    
    # Drop rows where lagged_load_kw is NaN
    merged_actual_df.dropna(subset=['load_kw_lag'], inplace=True)

    # Rename columns to more intuitive names
    final_df = merged_actual_df[['time', 'load_kw', 'load_kw_lag']].rename(columns={'load_kw_lag': 'lagged_load_kw'})
    
    # Join the two DataFrames on the 'time' column
    merged_data = pd.merge(final_df, retrain_check_data_forecasts, on='time', how='inner')

    # Order by 'time' in ascending order
    merged_data = merged_data.sort_values(by='time')

    final_retrain_data = merged_data.dropna(subset=['time', 'load_kw', 'lagged_load_kw', 'forecast'])

    # Check if there are at least 14 rows of data after the first 7 rows
    print(final_retrain_data.shape[0])
    if final_retrain_data.shape[0] >= (14 * 24) and retraining_required(final_retrain_data):
        print("RETRAINING REQUIRED!")
        retraining_data = fetch_actuals_from_db_for_retraining()
        retrain_model(retraining_data)

def get_and_upload_forecasts(extracted_date):
    actuals_df = fetch_actuals_from_db(extracted_date)
    forecasts_df = fetch_forecasts_from_db(extracted_date)

    combined_df = pd.concat([actuals_df, forecasts_df], ignore_index=True)
    two_day_forecasts = get_forecasts(combined_df)
    # Take the df and upload those forecasts to db
    db_manager.upload_model_forecasts(two_day_forecasts)
    return two_day_forecasts

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/upload_files', methods=['POST'])
def data_upload_endpoint():
    
    file = get_uploaded_file()

    if not file or not allowed_file(file.filename):
        return jsonify({"error": "Ensure you upload a zip file with the naming convention Data_Mon DAY 8AM.zip (eg Data_Feb 15 8AM.zip)"})

    filename = secure_filename(file.filename)
    filepath = save_file(file, filename)

    # Extract date from the filename
    extracted_date = extract_date_from_filename(filename)

    process_zip_file(filepath, filename, extracted_date)
    print("hello")
    retrain_framework(extracted_date)
    print("hello")
    two_day_forecasts = get_and_upload_forecasts(extracted_date)
    
    return jsonify({
        "message": "Data processed successfully",
        "data": two_day_forecasts.to_dict(orient='records')
    })


@app.route('/api/get_historical_actuals_and_forecasts', methods=['POST'])
def data_historical_upload_endpoint():
    # Assuming the dates are being sent as JSON payload
    data = request.get_json()

    start_date = data['startDate']
    end_date = data['endDate']

    actuals_df = fetch_actuals_from_db_for_insights(start_date, end_date)
    forecasts_df = fetch_forecasts_from_db_for_insights(start_date, end_date)

    # If there are any overlapping column names (apart from 'time'), you might want to rename them first
    actuals_df = actuals_df.add_prefix('actual_')
    forecasts_df = forecasts_df.add_prefix('forecast_')

    # Ensure the 'time' column retains its original name
    actuals_df.rename(columns={'actual_time': 'time'}, inplace=True)
    forecasts_df.rename(columns={'forecast_time': 'time'}, inplace=True)

    # Merging the two dataframes on the 'time' column using an outer join
    merged_df = pd.merge(actuals_df, forecasts_df, on='time', how='outer')

    # Sort by the 'time' column to ensure the merged data is in chronological order
    merged_df = merged_df.sort_values(by='time')

    merged_df.to_csv("merged.csv")

    # Replace NaN values with None
    merged_df.replace({np.nan: None}, inplace=True)

    # Convert the DataFrame to a dictionary and send it as a response
    return jsonify(merged_df.to_dict(orient='records'))

@app.route('/api/data-insights', methods=['GET'])
def get_insights():
    filename = request.args.get('filename')
    filename = secure_filename(filename)
    
    if not filename:
        return jsonify({"error": "Filename not provided."}), 400

    # Extract date string from the filename
    current_date_str = extract_date_from_filename(filename)

    # Convert the date string to a datetime object
    current_date = datetime.strptime(current_date_str, '%Y-%m-%d')

    # Calculate two_days_later and year_ago based on current_date
    two_days_later = current_date + timedelta(days=2)
    year_ago = current_date - timedelta(days=367)

    forecasts = fetch_forecasts_from_db_for_insights(current_date, two_days_later)
    historical_df = fetch_actuals_from_db_for_insights(year_ago, two_days_later)

    insights = get_data_insights(forecasts, historical_df, current_date)

    # Return the insights as a JSON object
    return jsonify(insights)

@app.route('/chatbot', methods=['POST'])
def chatbot_endpoint():
    data = request.get_json()  # Assume the message is being sent as a JSON payload
    
    if not data or 'message' not in data:
        return jsonify({"error": "Missing 'message' in request"}), 400  # Bad Request

    user_message = data['message']
    chatbot_instance = Chatbot()
    chatbot_response = chatbot_instance.run_chatbot(user_message)
    print(chatbot_response)
    
    return jsonify({"response": chatbot_response})

if __name__ == "__main__":
    app.run(debug=True)
