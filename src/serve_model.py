"""
Flask API of the SMS Spam detection model model.
"""
import joblib
import os
import urllib.request
import tarfile
import sys
import hashlib
import time
from flask import Flask, jsonify, request, make_response
from flasgger import Swagger
import pandas as pd
import psutil

from text_preprocessing import prepare, _extract_message_len, _text_process

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.environ.get('MODEL_DIR', 'output')
MODEL_URL = os.environ.get('MODEL_URL')
MODEL_FILE = os.path.join(MODEL_DIR, 'model.joblib')
PREPROCESSOR_FILE = os.path.join(MODEL_DIR, 'preprocessor.joblib')

CACHE_MAX_SIZE = int(os.environ.get('CACHE_MAX_SIZE', '1000'))
CACHE_TTL_SECONDS = int(os.environ.get('CACHE_TTL_SECONDS', '3600'))  # 1 hour default
CACHE_FLAG_HEADER = 'X-Cache-Enabled'

app = Flask(__name__)
swagger = Swagger(app)

# Simple in-memory cache: {message_hash: (prediction, timestamp)}
prediction_cache = {}
cache_hits = 0
cache_misses = 0

def get_cache_key(message):
    """Generate a cache key from the message content."""
    return hashlib.sha256(message.encode('utf-8')).hexdigest()

def get_from_cache(cache_key):
    """Retrieve a prediction from the cache if it exists and is not expired."""
    global cache_hits, cache_misses

    if cache_key in prediction_cache:
        prediction, timestamp = prediction_cache[cache_key]
        age = time.time() - timestamp

        if age < CACHE_TTL_SECONDS:
            cache_hits += 1
            print(f"[CACHE HIT] Key: {cache_key[:8]}... Age: {age:.2f}s")
            return prediction
        else:
            # Entry expired, remove it. Doing it here for simplicity
            del prediction_cache[cache_key]
            print(f"[CACHE EXPIRED] Key: {cache_key[:8]}... Age: {age:.2f}s")

    cache_misses += 1
    return None

def add_to_cache(cache_key, prediction):
    """Add a prediction to the cache."""
    global prediction_cache

    # If cache is full, remove oldest entry (simple FIFO eviction)
    if len(prediction_cache) >= CACHE_MAX_SIZE:
        oldest_key = next(iter(prediction_cache))
        del prediction_cache[oldest_key]
        print(f"[CACHE EVICTION] Removed oldest entry.")

    prediction_cache[cache_key] = (prediction, time.time())
    print(f"[CACHE ADD] Key: {cache_key[:8]}")

def get_cache_stats():
    """Get current cache statistics."""
    total_requests = cache_hits + cache_misses
    hit_rate = (cache_hits / total_requests * 100) if total_requests > 0 else 0

    return {
        "cache_size": len(prediction_cache),
        "cache_max_size": CACHE_MAX_SIZE,
        "cache_ttl_seconds": CACHE_TTL_SECONDS,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "total_requests": total_requests,
        "hit_rate_percent": round(hit_rate, 2),
    }

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

@app.route('/metrics', methods=['GET'])
def metics():
    """
    Get service metrics.
    ---
    responses:
      200:
        description: "Service metrics."
    """
    mem = psutil.virtual_memory()
    cache_stats = get_cache_stats()
    body = (
        f"backend_cpu_usage_percent {psutil.cpu_percent()}\n"
        f"backend_memory_max_bytes {mem.total}\n"
        f"backend_memory_used_bytes {mem.used}\n"
    )
    for key in cache_stats.keys():
        body += f"backend_{key} {cache_stats[key]}\n"
    response = make_response(body, 200)
    response.mimetype = "text/plain"
    return response

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

    # Check if caching is enabled via feature flag header
    use_cache = request.headers.get(CACHE_FLAG_HEADER, 'false').lower() == 'true'
    if use_cache:
        cache_key = get_cache_key(sms)
        cached_prediction = get_from_cache(cache_key)
        if cached_prediction is not None:
            res = {
                "result": cached_prediction,
                "classifier": "decision tree",
                "sms": sms,
                "cached": True
            }
            print(res)
            return jsonify(res)

    processed_sms = prepare(sms)
    if processed_sms is None:
        return jsonify({"error": "Failed to process SMS"}), 500
        
    model = joblib.load(MODEL_FILE) # Load from the configurable path
    prediction = model.predict(processed_sms)[0]

    if use_cache:
        add_to_cache(cache_key, prediction)

    res = {
        "result": prediction,
        "classifier": "decision tree",
        "sms": sms,
        "cached": False
    }
    print(res)
    return jsonify(res)

@app.route('/cache', methods=['GET'])
def cache_stats():
    """
    Get cache statistics.
    ---
    responses:
      200:
        description: "Cache statistics."
    """
    return jsonify()

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
    