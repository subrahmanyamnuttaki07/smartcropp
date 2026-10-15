"""
Plant Disease Detection System - Flask Web Application
------------------------------------------------------
Serves web portal for plant leaf image upload, runs CNN model prediction,
and displays diagnosis, confidence, and treatment guidelines.
"""

import os
import json
import uuid
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
from predict import PlantDiseasePredictor

app = Flask(__name__)

# Upload configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16 MB max limit

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Lazy loading of Predictor instance
predictor_instance = None

def get_predictor():
    global predictor_instance
    if predictor_instance is None:
        try:
            predictor_instance = PlantDiseasePredictor()
        except Exception as e:
            print(f"[!] Warning: Could not initialize predictor on startup: {e}")
            return None
    return predictor_instance

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Render main application home page."""
    # Check model trained status
    model_trained = os.path.exists(os.path.join(MODEL_DIR, 'plant_disease_model.keras')) or \
                    os.path.exists(os.path.join(MODEL_DIR, 'plant_disease_model.h5'))
    
    # Load evaluation metrics if available
    metrics = None
    metrics_path = os.path.join(MODEL_DIR, 'evaluation_metrics.json')
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
        except Exception:
            metrics = None

    return render_template('index.html', model_trained=model_trained, metrics=metrics)

@app.route('/predict', methods=['POST'])
def predict_disease():
    """Handle image upload, run CNN inference, and return prediction results."""
    # 1. Validate file presence in request
    if 'image' not in request.files:
        return jsonify({'error': 'No file part provided in request.'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No image selected for upload.'}), 400

    if not allowed_file(file.filename):
        return jsonify({
            'error': f'Invalid file format. Allowed extensions are: {", ".join(ALLOWED_EXTENSIONS).upper()}'
        }), 400

    # 2. Check if CNN model is loaded
    predictor = get_predictor()
    if predictor is None:
        return jsonify({
            'error': 'CNN Model is not trained yet! Please run "python train.py" to train the model first.'
        }), 500

    try:
        # Save file with unique filename to prevent browser caching issues
        ext = file.filename.rsplit('.', 1)[1].lower()
        unique_filename = f"leaf_{uuid.uuid4().hex[:8]}.{ext}"
        saved_filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(saved_filepath)

        # 3. Perform prediction
        result = predictor.predict(saved_filepath)
        result['image_url'] = url_for('static', filename=f'uploads/{unique_filename}')

        # Handle AJAX request vs Form Submit
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.is_json:
            return jsonify(result)
        
        return render_template('index.html', result=result, model_trained=True)

    except Exception as e:
        return jsonify({'error': f'Failed to process image: {str(e)}'}), 500

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """API endpoint returning model evaluation metrics."""
    metrics_path = os.path.join(MODEL_DIR, 'evaluation_metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    return jsonify({'error': 'Evaluation metrics not found.'}), 404

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File size exceeds maximum limit of 16MB.'}), 413

@app.errorhandler(500)
def internal_server_error(error):
    return jsonify({'error': 'Internal server error occurred while processing request.'}), 500

if __name__ == '__main__':
    print("="*60)
    print("   Starting Plant Disease Detection Flask Web Server")
    print("   URL: http://127.0.0.1:5000")
    print("="*60)
    app.run(host='0.0.0.0', port=5000, debug=True)
