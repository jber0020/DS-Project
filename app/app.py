from flask import Flask, render_template, jsonify, request
from werkzeug.utils import secure_filename
import os
import pandas as pd
import zipfile
from scripts.upload_data_check import validate_actuals, validate_forecasts
from datetime import datetime, timedelta  # Make sure to import this at the top of your file
from scripts.models import get_forecasts


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


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/upload_files', methods=['POST'])
def data_upload_endpoint():
    # check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files['uploaded_file']

    if file.filename == '':
        return jsonify({"error": "No selected file"})
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        "Data_Jan 18 8AM.zip"
        file_ext = filename.rsplit('.', 1)[1].lower()

        # Extract date from the filename
        extracted_date = extract_date_from_filename(filename)

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
                #print(validate_actuals(actuals, extracted_date))
                #print(validate_forecasts(forecasts, extracted_date))

                # Actuals file - Upload the latest actuals to db 
                
                # Forecasts - Upload forecasts to db

                # If retrain:
                    # run josh retraining script

                # Fetch last weeks worth of data - To feed into the forecast function we need to go and get the last weeks worth of data (Joshs model expects 1 week lag variables and 2 day lag variables)
                # Run josh forecast script - this will give us a df which has 48 hours worth of forecasts

                # Take the df and upload those forecasts to db


                return forecasts
                return jsonify({"message": "ZIP file uploaded and files loaded into DataFrames successfully and validated!"})

        # For direct file uploads (not ZIP)
        # else:
        #     if file_ext == 'csv':
        #         df = pd.read_csv(filepath)
        #     elif file_ext in ['xlsx', 'xls']:
        #         df = pd.read_excel(filepath)
        #     # You can extend this to handle other file types if necessary

        #     # Return success message after processing single file
        #     return jsonify({"message": "File uploaded and loaded into DataFrame successfully!"})

    return jsonify({"error": "File type not allowed"})




if __name__ == "__main__":
    app.run(debug=True)
