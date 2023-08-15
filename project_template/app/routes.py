
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
    return jsonify({"message": "Hello from the API!"}   )


# some change
