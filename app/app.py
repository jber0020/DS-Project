from flask import Flask, render_template, jsonify, request
import psycopg2
from werkzeug.utils import secure_filename
import os
import pandas as pd
import zipfile
from scripts.upload_data_check import validate_actuals, validate_forecasts
from datetime import datetime, timedelta  # Make sure to import this at the top of your file
from scripts.models import get_forecasts
from scripts.db_functions import PostgreSQLUploader, fetch_actuals_from_db, fetch_forecasts_from_db, fetch_actuals_from_db_for_insights, fetch_forecasts_from_db_for_insights
import numpy as np
from pytz import timezone


app = Flask(__name__)

host='ds4db.postgres.database.azure.com',
dbname='elec_db',
user='ds4user',
password='FIT3163!'

UPLOAD_FOLDER = 'upload_data/'  # replace with your desired upload folder
ALLOWED_EXTENSIONS = {'zip'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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

def get_data_insights(forecasts, historical_df):
    # Ensure the dataframes are sorted by time
    forecasts = forecasts.sort_values(by="time")
    historical_df = historical_df.sort_values(by="time")

    # Define a helper function to safely extract stats
    def safe_stat(data, stat_func):
        try:
            return stat_func(data)
        except:
            return "NA"

    # Get current time independently from data and set timezone to Melbourne
    current_time = pd.Timestamp.now(tz='Australia/Melbourne')

    # Define the period for "yesterday", "today", "tomorrow", "last week", and "last month" based on 8AM to 7AM schedule
    start_time_today = pd.Timestamp(current_time.date(), tz=current_time.tz) + pd.Timedelta(hours=8)
    start_time_yesterday = start_time_today - pd.Timedelta(days=1)
    end_time_today = start_time_today + pd.Timedelta(days=1)
    start_time_last_week = start_time_today - pd.Timedelta(weeks=1)
    start_time_last_month = start_time_today - pd.Timedelta(weeks=4)
    start_time_last_year = start_time_today - pd.Timedelta(weeks=52)

    # Extract statistics from yesterday, today, and tomorrow
    yesterday_demand = historical_df[(historical_df["time"] >= start_time_yesterday) & (historical_df["time"] < start_time_today)]
    today_demand = historical_df[(historical_df["time"] >= start_time_today) & (historical_df["time"] < end_time_today)]
    tomorrow_demand = forecasts.iloc[:24]  # Assuming the forecasts start from the current "day" and contain data for at least the next 24 hours

    insights = {
        "yesterday_max": safe_stat(yesterday_demand["load_kw"], max),
        "yesterday_min": safe_stat(yesterday_demand["load_kw"], min),
        "yesterday_avg": safe_stat(yesterday_demand["load_kw"], pd.Series.mean),

        "today_max": safe_stat(today_demand["load_kw"], max),
        "today_min": safe_stat(today_demand["load_kw"], min),
        "today_avg": safe_stat(today_demand["load_kw"], pd.Series.mean),

        "tomorrow_max": safe_stat(tomorrow_demand["load_kw"], max),
        "tomorrow_min": safe_stat(tomorrow_demand["load_kw"], min),
        "tomorrow_avg": safe_stat(tomorrow_demand["load_kw"], pd.Series.mean)
    }

    # Calculate the average demand for the defined periods
    last_week_demand = historical_df[(historical_df["time"] >= start_time_last_week) & (historical_df["time"] < start_time_today)]
    last_month_demand = historical_df[(historical_df["time"] >= start_time_last_month) & (historical_df["time"] < start_time_today)]
    last_year_demand = historical_df[(historical_df["time"] >= start_time_last_year) & (historical_df["time"] < start_time_today)]

    insights["weekly_avg"] = safe_stat(last_week_demand["load_kw"], pd.Series.mean)
    insights["monthly_avg"] = safe_stat(last_month_demand["load_kw"], pd.Series.mean)
    insights["yearly_avg"] = safe_stat(last_year_demand["load_kw"], pd.Series.mean)

    return insights




@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/upload_files', methods=['POST'])
def data_upload_endpoint():
    # check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"})
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        "Data_Jan 18 8AM.zip"
        file_ext = filename.rsplit('.', 1)[1].lower()

        # Extract date from the filename
        print("here works", filename)
        extracted_date = extract_date_from_filename(filename)
        print("here works", extracted_date)

        if file_ext == 'zip':
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

                #print("yoooo", extracted_date)
                print(validate_actuals(actuals, extracted_date))
                print(validate_forecasts(forecasts, extracted_date))

                # Actuals file - Upload the latest actuals to db
                db_manager = PostgreSQLUploader()
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
                # If retrain:
                    # run josh retraining script

                # Fetch last weeks worth of data - To feed into the forecast function we need to go and get the last weeks worth of data (Joshs model expects 1 week lag variables and 2 day lag variables)
                # Last week + 2 day forecasted variables (load will be null)
                actuals_df = fetch_actuals_from_db(extracted_date)
                forecasts_df = fetch_forecasts_from_db(extracted_date)

                combined_df = pd.concat([actuals_df, forecasts_df], ignore_index=True)
                two_day_forecasts = get_forecasts(combined_df)
                two_day_forecasts.to_csv("fopre.csv")
                # Take the df and upload those forecasts to db
                db_manager.upload_model_forecasts(two_day_forecasts)

                # data_insights = get_data_insights(two_day_forecasts)

                return jsonify({
                    "message": "Data processed successfully",
                    "data": two_day_forecasts.to_dict(orient='records')
                })

        # For direct file uploads (not ZIP)
        # else:
        #     if file_ext == 'csv':
        #         df = pd.read_csv(filepath)
        #     elif file_ext in ['xlsx', 'xls']:
        #         df = pd.read_excel(filepath)
        #     # You can extend this to handle other file types if necessary

        #     # Return success message after processing single file
        #     return jsonify({"message": "File uploaded and loaded into DataFrame successfully!"})

    # return jsonify({"error": "File type not allowed"})

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
    print(filename)
    
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

    insights = get_data_insights(forecasts, historical_df)

    # Return the insights as a JSON object
    return jsonify(insights)


def test(extracted_date):
    db_manager = PostgreSQLUploader()
    actuals_df = fetch_actuals_from_db(extracted_date)
    forecasts_df = fetch_forecasts_from_db(extracted_date)
    
    print("hey")
    combined_df = pd.concat([actuals_df, forecasts_df], ignore_index=True)
    combined_df.to_csv("tester.csv")
    two_day_forecasts = get_forecasts(combined_df)

    # Take the df and upload those forecasts to db
    db_manager.upload_model_forecasts(two_day_forecasts)


if __name__ == "__main__":
    app.run(debug=True)
    # print(test("2021-02-08"))
