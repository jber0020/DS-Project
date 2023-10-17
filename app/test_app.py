import pytest
from app.app import app  # adjust the import based on your folder structure
import os
from werkzeug.datastructures import FileStorage
from unittest.mock import patch
import unittest
import os
from scripts.db_functions import PostgreSQLUploader
from scripts.models import upload_to_blob, download_from_blob, blob_service_client

import pandas as pd

# Create a sample dataframe
mock_df = pd.DataFrame({
    'column1': ['value1', 'value2'],
    'column2': ['value3', 'value4']
})

# This ensures that Flask runs in test mode and exceptions are propagated rather than handled by the app's error handlers
@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_invalid_file_upload(client):
    # with app.app_context():
    response = client.post('/api/upload_files')
    assert b"Ensure you upload a zip file with the naming convention Data_Mon DAY 8AM.zip (eg Data_Feb 15 8AM.zip)" in response.data

def test_valid_file_upload(client):
    # Create a mock file for testing
    with open("test.zip", "wb") as f:
        f.write(b"This is a test zip file")

    data = {
        'file': (open('test.zip', 'rb'), 'test.zip')
    }

    # Mocking methods to avoid side-effects and focus on endpoint functionality
    with patch('app.app.get_uploaded_file') as mock_get_uploaded_file, \
         patch('app.app.allowed_file', return_value=True), \
         patch('app.app.save_file'), \
         patch('app.app.extract_date_from_filename'), \
         patch('app.app.process_zip_file'), \
         patch('app.app.retrain_framework'), \
         patch('app.app.get_and_upload_forecasts', return_value=mock_df):
             
        mock_get_uploaded_file.return_value = FileStorage(stream=open("test.zip", "rb"))

        response = client.post('/api/upload_files', data=data, content_type='multipart/form-data')
        assert b"Data processed successfully" in response.data

    os.remove("test.zip")

def test_chatbot_valid_data(client):
    test_message = "Hello, Chatbot!"
    mock_response = "Hello, User!"  # or whatever you expect

    # Mock the Chatbot instance and its run_chatbot method
    with patch('app.app.Chatbot') as MockChatbot:
        mock_chatbot_instance = MockChatbot.return_value
        mock_chatbot_instance.run_chatbot.return_value = mock_response

        response = client.post('/chatbot', json={"message": test_message})

        assert response.status_code == 200
        json_data = response.get_json()
        assert json_data["response"] == mock_response

def test_chatbot_no_data(client):
    response = client.post('/chatbot', json={})
    assert response.status_code == 400
    json_data = response.get_json()
    assert json_data["error"] == "Missing 'message' in request"

def test_chatbot_no_message_key(client):
    response = client.post('/chatbot', json={"other_key": "value"})
    assert response.status_code == 400
    json_data = response.get_json()
    assert json_data["error"] == "Missing 'message' in request"

def test_database_connection():
    db = PostgreSQLUploader()
    db._connect()
    assert db.conn is not None
    assert db.cur is not None
    db._disconnect()

@pytest.fixture(scope="module")
def blob_file_setup_teardown():
    test_file = "test_blob.txt"
    with open(test_file, "w") as f:
        f.write("This is a test blob file.")
    
    yield test_file

    if os.path.exists(test_file):
        os.remove(test_file)

def test_blob_upload_download(blob_file_setup_teardown):
    test_file = blob_file_setup_teardown
    container_name = "test"
    
    upload_to_blob(test_file, container_name, blob_service_client)
    
    # Now, let's delete the file locally and try downloading it
    os.remove(test_file)
    assert not os.path.exists(test_file)
    
    download_from_blob(test_file, container_name, blob_service_client)
    assert os.path.exists(test_file)

if __name__ == "__main__":
    unittest.main()
