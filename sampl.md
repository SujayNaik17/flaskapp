from flask import Flask, request, jsonify
import os
import requests
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['POST'])
def upload_file():
    print("Received a request")
    print("Request content-type:", request.content_type)
    print("Request data:", request.data)
    print("Request files:", request.files)
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400

    # Log the size of the uploaded file
    print(f"Uploaded file size: {len(file.read())} bytes")
    
    # Reset the file stream position to the beginning after logging the size
    file.seek(0)

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    response = make_prediction(file_path)
    print(file_path)

    return jsonify(response)

def make_prediction(file_path):
    url = "http://127.0.0.1:5000/api/predict"

    with open(file_path, 'rb') as file:
        files = {'file': file}
        response = requests.post(url, files=files)

    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")

    return {
        'status_code': response.status_code,
        'response': response.json()
    }

if __name__ == '__main__':
    app.run(port=3100, debug=True)
