"""
Flask API of the SMS Spam detection model model.
"""
import joblib
import os
import urllib.request
import tarfile
import sys
from flask import Flask, jsonify, request
from flasgger import Swagger
import pandas as pd

from text_preprocessing import prepare, _extract_message_len, _text_process

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.environ.get('MODEL_DIR', 'output')
MODEL_URL = os.environ.get('MODEL_URL')
MODEL_FILE = os.path.join(MODEL_DIR, 'model.joblib')
PREPROCESSOR_FILE = os.path.join(MODEL_DIR, 'preprocessor.joblib')

app = Flask(__name__)
swagger = Swagger(app)

def download_and_extract_model():
    """
    Downloads and extracts the model if not already present.
    """
    try:
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # Check if files exist (local cache)
        if os.path.exists(MODEL_FILE) and os.path.exists(PREPROCESSOR_FILE):
            print(f"Model files found in {MODEL_DIR}, skipping download.")
            return

        print("Model files not found locally.")

        # If no local files, check for MODEL_URL
        if not MODEL_URL:
            print("ERROR: MODEL_URL environment variable is not set.", file=sys.stderr)
            print("Please set MODEL_URL to point to a model-release.tar.gz file.", file=sys.stderr)
            sys.exit(1) 

        print(f"Downloading model from {MODEL_URL}...")
        
        # Temp path for the downloaded tarball
        tmp_tar_path = os.path.join(MODEL_DIR, 'model-release.tar.gz')
        
        urllib.request.urlretrieve(MODEL_URL, tmp_tar_path)
        print("Download complete. Extracting...")

        # Extract the tarball
        with tarfile.open(tmp_tar_path, "r:gz") as tar:
            tar.extractall(path=MODEL_DIR)
        
        print(f"Extraction complete. Model files are in {MODEL_DIR}.")

        # Clean up the downloaded tarball
        os.remove(tmp_tar_path)

        if not os.path.exists(MODEL_FILE) or not os.path.exists(PREPROCESSOR_FILE):
            print(f"ERROR: Expected model files not found after extraction.", file=sys.stderr)
            sys.exit(1)

    except Exception as e:
        print(f"Error during model download/extraction: {e}", file=sys.stderr)
        sys.exit(1)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict whether an SMS is Spam.
    ---
    consumes:
      - application/json
    parameters:
        - name: input_data
          in: body
          description: message to be classified.
          required: True
          schema:
            type: object
            required: sms
            properties:
                sms:
                    type: string
                    example: This is an example of an SMS.
    responses:
      200:
        description: "The result of the classification: 'spam' or 'ham'."
    """
    input_data = request.get_json()
    sms = input_data.get('sms')
    
    processed_sms = prepare(sms)
    if processed_sms is None:
        return jsonify({"error": "Failed to process SMS"}), 500
        
    model = joblib.load(MODEL_FILE) # Load from the configurable path
    prediction = model.predict(processed_sms)[0]
    
    res = {
        "result": prediction,
        "classifier": "decision tree",
        "sms": sms
    }
    print(res)
    return jsonify(res)

@app.route('/version', methods=['GET'])
def get_version():
    """
    Get the version of the model service.
    ---
    responses:
      200:
        description: "The version of the model service."
        schema:
          type: object
          properties:
            version:
              type: string
              example: "v1.1.0"
    """
    try:
        # Path to version.txt relative to this file
        version_file = os.path.join(PROJECT_ROOT, 'version.txt')
        with open(version_file, 'r') as f:
            version = f.read().strip()
        return jsonify({"version": version})
    except Exception as e:
        return jsonify({"version": "unknown", "error": str(e)}), 500

if __name__ == '__main__':

    download_and_extract_model()     
    # Get port from environment variable, default to 8081
    app_port = int(os.environ.get('APP_PORT', 8081))
    app.run(host="0.0.0.0", port=app_port, debug=True)
    