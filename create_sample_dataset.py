"""
Sample Dataset Generator for Plant Disease Detection System
------------------------------------------------------------
Generates synthetic leaf images across 5 classes to allow instant end-to-end
testing of dataset loading, CNN model training, evaluation metrics, and web UI.
"""

import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

DATASET_DIR = os.path.join(os.path.dirname(__file__), 'dataset')

CLASSES = [
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___healthy',
    'Potato___Early_blight',
    'Potato___healthy'
]

def generate_leaf_image(category_name, width=256, height=256):
    """Generate a realistic leaf image with disease pattern based on class name."""
    # Base green leaf background with texture
    base_color = (34, 139, 34) if 'healthy' in category_name else (46, 117, 46)
    img = Image.new('RGB', (width, height), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)

    # Draw leaf boundary (ellipse/polygon hybrid)
    leaf_shape = [
        (width // 4, height // 8),
        (3 * width // 4, height // 4),
        (7 * width // 8, height // 2),
        (3 * width // 4, 3 * height // 4),
        (width // 2, 7 * height // 8),
        (width // 4, 3 * height // 4),
        (width // 8, height // 2)
    ]
    
    # Slight color jitter per leaf
    r = max(0, min(255, base_color[0] + random.randint(-15, 15)))
    g = max(0, min(255, base_color[1] + random.randint(-15, 15)))
    b = max(0, min(255, base_color[2] + random.randint(-15, 15)))
    draw.polygon(leaf_shape, fill=(r, g, b), outline=(20, 80, 20))

    # Draw leaf veins
    draw.line([(width // 2, height // 8), (width // 2, 7 * height // 8)], fill=(70, 160, 70), width=3)
    for y in range(height // 4, 3 * height // 4, 25):
        draw.line([(width // 2, y), (width // 4, y - 15)], fill=(60, 140, 60), width=2)
        draw.line([(width // 2, y), (3 * width // 4, y - 15)], fill=(60, 140, 60), width=2)

    # Add disease features based on category
    if 'Early_blight' in category_name:
        # Concentric dark brown spots with halo
        num_spots = random.randint(6, 14)
        for _ in range(num_spots):
            cx = random.randint(width // 4, 3 * width // 4)
            cy = random.randint(height // 4, 3 * height // 4)
            rad = random.randint(8, 18)
            # Yellow halo
            draw.ellipse([cx - rad - 3, cy - rad - 3, cx + rad + 3, cy + rad + 3], fill=(180, 180, 50))
            # Dark brown center
            draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=(80, 40, 20))
            # Concentric ring inside
            draw.ellipse([cx - rad // 2, cy - rad // 2, cx + rad // 2, cy + rad // 2], fill=(40, 20, 10))

    elif 'Late_blight' in category_name:
        # Large irregular dark greyish-brown lesions
        num_lesions = random.randint(3, 7)
        for _ in range(num_lesions):
            cx = random.randint(width // 4, 3 * width // 4)
            cy = random.randint(height // 4, 3 * height // 4)
            rad_x = random.randint(20, 40)
            rad_y = random.randint(15, 35)
            draw.ellipse([cx - rad_x, cy - rad_y, cx + rad_x, cy + rad_y], fill=(60, 50, 40))

    # Apply light Gaussian blur for natural look
    img = img.filter(ImageFilter.GaussianBlur(radius=0.8))
    
    # Add random pixel noise
    arr = np.array(img, dtype=np.int16)
    noise = np.random.randint(-10, 10, arr.shape, dtype=np.int16)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def generate_dataset(images_per_class_train=30, images_per_class_val=10):
    """Create train and validation folder structure and save images."""
    print("[*] Generating synthetic sample leaf dataset...")
    
    train_dir = os.path.join(DATASET_DIR, 'train')
    val_dir = os.path.join(DATASET_DIR, 'val')
    
    for cls in CLASSES:
        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(val_dir, cls), exist_ok=True)

        # Train images
        for i in range(images_per_class_train):
            img = generate_leaf_image(cls)
            img.save(os.path.join(train_dir, cls, f"{cls}_train_{i+1}.jpg"))
            
        # Validation images
        for i in range(images_per_class_val):
            img = generate_leaf_image(cls)
            img.save(os.path.join(val_dir, cls, f"{cls}_val_{i+1}.jpg"))

    # Also save one sample leaf image for direct test prediction
    sample_test_img = generate_leaf_image('Tomato___Early_blight')
    sample_test_path = os.path.join(os.path.dirname(__file__), 'sample_leaf.jpg')
    sample_test_img.save(sample_test_path)

    print(f"[+] Sample dataset created under '{DATASET_DIR}'!")
    print(f"    - Classes: {len(CLASSES)}")
    print(f"    - Training images: {len(CLASSES) * images_per_class_train}")
    print(f"    - Validation images: {len(CLASSES) * images_per_class_val}")
    print(f"[+] Created sample test image at '{sample_test_path}'")

if __name__ == "__main__":
    generate_dataset()
