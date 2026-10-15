"""
Verification Test Script - Predict 5 Different Plant Leaf Images
------------------------------------------------------------------
Loads 1 image from each of the 5 classes in dataset/val/ and runs
PlantDiseasePredictor to print predicted index, name, confidence, and top 3 probabilities.
"""

import os
import json
import numpy as np
from predict import PlantDiseasePredictor

def test_5_different_images():
    print("="*90)
    print("           VERIFICATION TEST: PREDICTING 5 DIFFERENT LEAF IMAGES")
    print("="*90)

    predictor = PlantDiseasePredictor()
    val_dir = os.path.join(os.path.dirname(__file__), 'dataset', 'val')
    classes = sorted([c for c in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, c))])

    results = []

    for cls in classes:
        cls_path = os.path.join(val_dir, cls)
        img_name = os.listdir(cls_path)[0]
        img_path = os.path.join(cls_path, img_name)

        # Run prediction
        res = predictor.predict(img_path)

        # Extract top 3 probabilities
        probs = res['all_probabilities']
        top3 = list(probs.items())[:3]

        results.append({
            'filename': img_name,
            'true_class': cls,
            'crop': res['crop'],
            'condition': res['condition'],
            'predicted_class_name': res['predicted_disease'],
            'confidence': res['confidence'],
            'top3': top3
        })

        print(f"\n[IMAGE FILE] : {img_name}")
        print(f"  - Ground Truth Class : {cls}")
        print(f"  - Predicted Crop     : {res['crop']}")
        print(f"  - Predicted Condition: {res['condition']}")
        print(f"  - Confidence Score   : {res['confidence']}%")
        print(f"  - Top 3 Probabilities:")
        for name, prob in top3:
            print(f"     * {name:<28}: {prob:>6.2f}%")

    print("\n" + "="*90)
    print("                          SUMMARY TABLE")
    print("="*90)
    print(f" {'Filename':<32} | {'True Class':<22} | {'Predicted Class':<22} | {'Confidence':<10}")
    print("-" * 90)
    for r in results:
        print(f" {r['filename'][:30]:<32} | {r['true_class']:<22} | {r['predicted_class_name']:<22} | {r['confidence']:>8.2f}%")
    print("="*90)

if __name__ == "__main__":
    test_5_different_images()
