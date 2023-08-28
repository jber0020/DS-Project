
from flask import render_template, jsonify
from app import app
import sys
sys.path.append('../scripts')
# import your scripts here

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/sample_endpoint')
def sample_endpoint():
    return jsonify({"message": "Hello from the API!"})

@app.route('/api/upload_and_forecast')
def sample_endpoint():
    # Josh script number 1 - takes the uploaded file and stores in database
    return jsonify({"message": "Lets forecast"})

# @app.route('/api/upload_files')
# def data_upload_endpoint():
#     # Use josh assertion script - assertion.py
#     # cleaned_data = assertion.py
#     # # Josh cleaning and preprocessing script - preprocess.py
#     # cleaned_data = 
#     # # Joshs forecasting script - forecast.py
#     # return jsonify({"message": "Hello from the API!"})
