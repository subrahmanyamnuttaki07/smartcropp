import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image

model_path = 'model/plant_disease_model.keras'
model = tf.keras.models.load_model(model_path)
with open('model/class_indices.json') as f:
    class_map = {int(k): v for k, v in json.load(f).items()}

dataset_val = 'dataset/val'
classes = sorted([c for c in os.listdir(dataset_val) if os.path.isdir(os.path.join(dataset_val, c))])

print(f"Loaded model from: {model_path}")
print(f"Class map: {class_map}\n")

print("="*90)
print(f" {'Filename':<32} | {'True Class':<22} | {'Predicted Class':<22} | {'Conf':<6}")
print("="*90)

for cls in classes:
    cls_dir = os.path.join(dataset_val, cls)
    img_name = os.listdir(cls_dir)[0]
    img_path = os.path.join(cls_dir, img_name)
    
    img = Image.open(img_path).convert('RGB').resize((128, 128))
    arr = np.array(img, dtype=np.float32)
    batch = np.expand_dims(arr, axis=0)
    
    preds = model.predict(batch, verbose=0)[0]
    top3_idx = np.argsort(preds)[::-1][:3]
    top1_cls = class_map[top3_idx[0]]
    conf = float(preds[top3_idx[0]]) * 100
    
    print(f" {img_name[:30]:<32} | {cls:<22} | {top1_cls:<22} | {conf:>5.1f}%")
    top3_str = ", ".join([f"{class_map[i]}: {preds[i]*100:.1f}%" for i in top3_idx])
    print(f"    Top 3 probabilities: {top3_str}\n")
print("="*90)
