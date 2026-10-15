"""
Complete ML Pipeline Diagnostic Script
-------------------------------------
Inspects:
 1. Model architecture, layers, input shape, output shape
 2. Saved class mapping vs Keras directory class names
 3. Dataset distribution across classes (Train / Val / Test)
 4. Performance evaluation on 10+ validation images
 5. Bias detection across predictions
"""

import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'plant_disease_model.keras')
CLASS_MAP_PATH = os.path.join(BASE_DIR, 'model', 'class_indices.json')
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')

def run_diagnostics():
    print("="*80)
    print("                 ML PIPELINE DIAGNOSTIC REPORT")
    print("="*80)

    # 1. Inspect Model File
    if not os.path.exists(MODEL_PATH):
        print(f"[X] Model file missing at: {MODEL_PATH}")
        return
    
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"[+] Loaded Model Path  : {MODEL_PATH}")
    print(f"[+] Model Input Shape  : {model.input_shape}")
    print(f"[+] Model Output Shape : {model.output_shape}")
    print(f"[+] Number of Output Neurons: {model.output_shape[-1]}")

    # 2. Inspect Class Mapping
    if not os.path.exists(CLASS_MAP_PATH):
        print(f"[X] Class mapping missing at: {CLASS_MAP_PATH}")
        return

    with open(CLASS_MAP_PATH, 'r') as f:
        class_map = json.load(f)
    
    print("\n[*] Saved Class Mapping (JSON):")
    for k, v in sorted(class_map.items(), key=lambda x: int(x[0])):
        print(f"    Index {k}: {v}")

    # 3. Inspect Dataset Structure & Class Imbalance
    train_path = os.path.join(DATASET_DIR, 'train')
    val_path = os.path.join(DATASET_DIR, 'val')

    print("\n[*] Inspecting Dataset Folders:")
    train_classes = sorted([c for c in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, c))])
    print(f"    Train Classes Detected ({len(train_classes)}): {train_classes}")

    print("\n[*] Image Counts per Class:")
    print(f"    {'Class Name':<28} | {'Train Count':<12} | {'Val Count':<12}")
    print("    " + "-"*58)
    for cls in train_classes:
        tr_cnt = len(os.listdir(os.path.join(train_path, cls)))
        val_cnt = len(os.listdir(os.path.join(val_path, cls))) if os.path.exists(os.path.join(val_path, cls)) else 0
        print(f"    {cls:<28} | {tr_cnt:<12} | {val_cnt:<12}")

    # 4. Check for Class Mapping Consistency
    keras_class_names = train_classes
    mismatch = False
    for idx, name in enumerate(keras_class_names):
        json_name = class_map.get(str(idx), class_map.get(idx, None))
        if json_name != name:
            mismatch = True
            print(f"[X] MISMATCH at index {idx}: Keras folder '{name}' != JSON '{json_name}'")
    
    if not mismatch:
        print("\n[+] Class Index Mapping is 100% CONSTANT and MATCHES directory order!")

    # 5. Evaluate Predictions on 10 Unseen Images across classes
    print("\n" + "="*80)
    print("         TESTING PREDICTIONS ON 10 UNSEEN VALIDATION IMAGES")
    print("="*80)
    print(f" {'Filename':<30} | {'True Class':<22} | {'Predicted Class':<22} | {'Conf':<6}")
    print("-" * 80)

    predictions_distribution = {v: 0 for v in class_map.values()}
    total_tested = 0

    for cls in train_classes:
        cls_val_dir = os.path.join(val_path, cls)
        if not os.path.exists(cls_val_dir):
            continue
        imgs = [f for f in os.listdir(cls_val_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))][:2]
        
        for img_name in imgs:
            img_path = os.path.join(cls_val_dir, img_name)
            img = Image.open(img_path).convert('RGB').resize((128, 128))
            arr = np.array(img, dtype=np.float32)
            batch = np.expand_dims(arr, axis=0)

            raw_preds = model.predict(batch, verbose=0)[0]
            top_idx = int(np.argmax(raw_preds))
            top_cls = class_map[str(top_idx)]
            conf = float(raw_preds[top_idx]) * 100.0

            predictions_distribution[top_cls] += 1
            total_tested += 1

            print(f" {img_name[:28]:<30} | {cls:<22} | {top_cls:<22} | {conf:>5.1f}%")
            top3_idx = np.argsort(raw_preds)[::-1][:3]
            top3_str = ", ".join([f"{class_map[str(i)]}: {raw_preds[i]*100:.1f}%" for i in top3_idx])
            print(f"    Top 3: {top3_str}\n")

    print("="*80)
    print("Prediction Distribution Across Test Samples:")
    for k, v in predictions_distribution.items():
        pct = (v / total_tested) * 100 if total_tested > 0 else 0
        print(f"  - {k:<28}: {v} samples ({pct:.1f}%)")
    print("="*80)

if __name__ == "__main__":
    run_diagnostics()
