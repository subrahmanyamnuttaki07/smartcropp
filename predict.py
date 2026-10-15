"""
Plant Disease Detection - Single Image Prediction Utility
---------------------------------------------------------
Loads trained CNN model and class mappings to classify plant leaf images.
Dynamically reads model.input_shape to ensure prediction preprocessing
matches the exact input dimensions (128x128) expected by the saved model.
"""

import os
import sys
import json
import argparse
import numpy as np
from PIL import Image
import tensorflow as tf

CONFIDENCE_THRESHOLD = 60.0  # Percentage below which prediction is marked Low Confidence
MARGIN_THRESHOLD = 15.0      # Min margin between Top-1 and Top-2 class probabilities

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'plant_disease_model.keras')
H5_MODEL_PATH = os.path.join(BASE_DIR, 'model', 'plant_disease_model.h5')
CLASS_INDICES_PATH = os.path.join(BASE_DIR, 'model', 'class_indices.json')

# Educational Care Guidelines Database
DISEASE_REMEDIES = {
    'Tomato___Early_blight': {
        'status': 'Diseased',
        'severity': 'Moderate',
        'symptoms': 'Brown concentric spots on lower leaves forming target-like patterns.',
        'treatment': 'Apply copper-based fungicide sprays or chlorothalonil. Prune infected lower leaves and ensure good airflow.',
        'prevention': 'Practice crop rotation every 2-3 years, remove crop debris after harvest, and avoid overhead sprinkler irrigation.'
    },
    'Tomato___Late_blight': {
        'status': 'Diseased',
        'severity': 'Severe',
        'symptoms': 'Large dark watery leaf spots with white downy fungal growth on underside under humid conditions.',
        'treatment': 'Apply systemic fungicides (Mancozeb or Copper octanoate) immediately. Remove and dispose of severely infected plant foliage.',
        'prevention': 'Use certified disease-resistant seeds, maximize sunlight exposure, and space plants for optimal ventilation.'
    },
    'Tomato___healthy': {
        'status': 'Healthy',
        'severity': 'None',
        'symptoms': 'Foliage exhibits vibrant green color with smooth surface and no spots or decay.',
        'treatment': 'No chemical treatment required. Continue regular drip watering and balanced organic fertilization.',
        'prevention': 'Inspect foliage weekly for early signs of pests or fungal spores.'
    },
    'Potato___Early_blight': {
        'status': 'Diseased',
        'severity': 'Moderate',
        'symptoms': 'Dark brown spots on older potato leaves leading to foliage yellowing.',
        'treatment': 'Spray protective copper or chlorothalonil fungicides at initial symptom appearance.',
        'prevention': 'Ensure balanced soil nitrogen and potassium levels, avoid waterlogging, and rotate solanaceous crops.'
    },
    'Potato___healthy': {
        'status': 'Healthy',
        'severity': 'None',
        'symptoms': 'Potato foliage is uniform green, sturdy, and disease-free.',
        'treatment': 'No treatment required.',
        'prevention': 'Maintain weed control and proper hilling.'
    }
}

class PlantDiseasePredictor:
    def __init__(self):
        self.model = None
        self.class_indices = None
        self.target_size = (128, 128)
        self._load_resources()

    def _load_resources(self):
        """Load trained TensorFlow CNN model and class label mapping."""
        if not os.path.exists(CLASS_INDICES_PATH):
            raise FileNotFoundError(
                f"Class indices file not found at '{CLASS_INDICES_PATH}'. "
                "Please train the model first by running 'python train.py'."
            )
        with open(CLASS_INDICES_PATH, 'r') as f:
            raw_map = json.load(f)
            self.class_indices = {int(k): v for k, v in raw_map.items()}

        model_file = MODEL_PATH if os.path.exists(MODEL_PATH) else H5_MODEL_PATH
        if not os.path.exists(model_file):
            raise FileNotFoundError(
                f"Model file not found at '{MODEL_PATH}'. "
                "Please train the model first by running 'python train.py'."
            )
        
        try:
            self.model = tf.keras.models.load_model(model_file)
            input_shape = self.model.input_shape
            # Extract target size dynamically from model shape (batch, height, width, channels)
            if input_shape and len(input_shape) >= 3 and input_shape[1] is not None:
                self.target_size = (input_shape[1], input_shape[2])
            print(f"[+] Loaded Model from '{os.path.basename(model_file)}' | Expected Input Shape: {self.model.input_shape}")
        except Exception as e:
            raise RuntimeError(f"Failed to load CNN model: {e}")

    def preprocess_image(self, image_input):
        """
        Resizes input image to model's expected shape (128, 128) and converts to RGB float32.
        Passes raw [0, 255] float32 array so internal model preprocess_input layer handles scaling ONCE.
        """
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image file does not exist at '{image_input}'")
            img = Image.open(image_input)
        elif isinstance(image_input, Image.Image):
            img = image_input
        else:
            raise ValueError("Input must be a valid file path or PIL Image instance.")

        img = img.convert('RGB')
        img = img.resize(self.target_size)
        img_array = np.array(img, dtype=np.float32)
        img_batch = np.expand_dims(img_array, axis=0)
        return img_batch

    def predict(self, image_input):
        """
        Runs CNN inference on input image.
        Returns prediction details with top 3 probabilities, confidence, and uncertainty flags.
        """
        img_batch = self.preprocess_image(image_input)
        predictions = self.model.predict(img_batch, verbose=0)[0]
        
        sorted_indices = np.argsort(predictions)[::-1]
        top1_idx = int(sorted_indices[0])
        top2_idx = int(sorted_indices[1]) if len(sorted_indices) > 1 else top1_idx

        top1_conf = float(predictions[top1_idx]) * 100.0
        top2_conf = float(predictions[top2_idx]) * 100.0
        margin = top1_conf - top2_conf

        disease_name = self.class_indices.get(top1_idx, "Unknown Disease")
        formatted_name = disease_name.replace('___', ' - ').replace('_', ' ')
        parts = disease_name.split('___')
        crop_name = parts[0].replace('_', ' ')
        condition_name = parts[1].replace('_', ' ') if len(parts) > 1 else 'Unknown'

        remedy_info = DISEASE_REMEDIES.get(disease_name, {
            'status': 'Unknown',
            'severity': 'Unknown',
            'symptoms': 'N/A',
            'treatment': 'Consult local agricultural extension service.',
            'prevention': 'Maintain clean cultivation practices.'
        })

        # Top 3 classes and probabilities
        top3_list = []
        all_probs = {}
        for idx in sorted_indices:
            cls_raw = self.class_indices.get(idx, str(idx))
            cls_clean = cls_raw.replace('___', ' - ').replace('_', ' ')
            prob_pct = round(float(predictions[idx]) * 100.0, 2)
            all_probs[cls_clean] = prob_pct
            if len(top3_list) < 3:
                top3_list.append({'class_name': cls_clean, 'probability': prob_pct})

        # Confidence & Out-of-Distribution Uncertainty Flag
        is_low_confidence = (top1_conf < CONFIDENCE_THRESHOLD) or (margin < MARGIN_THRESHOLD)

        if is_low_confidence:
            status_text = "Low Confidence / Uncertain Identification"
            severity_text = "Uncertain"
            warning_msg = (
                f"Low Prediction Confidence ({top1_conf:.1f}%). The uploaded photo may contain background clutter, "
                "non-leaf objects, or an unsupported plant condition. Please upload a clear, centered leaf image."
            )
        else:
            status_text = remedy_info['status']
            severity_text = remedy_info['severity']
            warning_msg = None

        return {
            'raw_class': disease_name,
            'crop': crop_name,
            'condition': condition_name,
            'predicted_disease': formatted_name,
            'confidence': round(top1_conf, 2),
            'margin': round(margin, 2),
            'is_low_confidence': is_low_confidence,
            'status': status_text,
            'severity': severity_text,
            'symptoms': remedy_info['symptoms'],
            'treatment': remedy_info['treatment'],
            'prevention': remedy_info['prevention'],
            'warning': warning_msg,
            'disclaimer': 'Educational & preliminary identification only. Always consult a certified agricultural extension specialist before applying chemical field treatments.',
            'top3_predictions': top3_list,
            'all_probabilities': all_probs
        }

def main():
    parser = argparse.ArgumentParser(description="Predict plant disease from a leaf image")
    parser.add_argument('--image', type=str, required=True, help="Path to input plant leaf image")
    args = parser.parse_args()

    try:
        predictor = PlantDiseasePredictor()
        result = predictor.predict(args.image)

        print("\n" + "="*60)
        print("                PREDICTION RESULT SUMMARY")
        print("="*60)
        print(f" Image Input       : {args.image}")
        print(f" Crop Name         : {result['crop']}")
        print(f" Condition / State : {result['condition']}")
        print(f" Confidence Score  : {result['confidence']}%")
        print(f" Uncertainty Flag  : {'LOW CONFIDENCE' if result['is_low_confidence'] else 'HIGH CONFIDENCE'}")
        print(f" Plant Health      : {result['status']}")
        print(f" Educational Guide : {result['treatment']}")
        if result['warning']:
            print(f" Warning Alert     : {result['warning']}")
        print("="*60)
        print("\nTop 3 Predictions:")
        for item in result['top3_predictions']:
            print(f"  - {item['class_name']:<30}: {item['probability']:>6.2f}%")
        print("="*60)

    except Exception as e:
        print(f"[X] Prediction Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
