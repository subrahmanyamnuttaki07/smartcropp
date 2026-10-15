"""
Real PlantVillage Dataset Downloader
-------------------------------------
Downloads real plant leaf disease photographs directly from the official
PlantVillage GitHub repository (spMohanty/PlantVillage-Dataset).
"""

import os
import json
import urllib.request
import random

DATASET_DIR = os.path.join(os.path.dirname(__file__), 'dataset')

TARGET_CLASSES = [
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___healthy',
    'Potato___Early_blight',
    'Potato___healthy'
]

GITHUB_API_BASE = "https://api.github.com/repos/spMohanty/PlantVillage-Dataset/contents/raw/color/"

def fetch_folder_contents(class_name):
    """Fetch list of file metadata for a specific class folder from GitHub API."""
    url = GITHUB_API_BASE + class_name
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req) as res:
            items = json.loads(res.read())
            image_urls = [item['download_url'] for item in items if item['type'] == 'file' and item['name'].lower().endswith(('.jpg', '.jpeg', '.png'))]
            return image_urls
    except Exception as e:
        print(f"[!] Error fetching contents for {class_name}: {e}")
        return []

def download_real_dataset(samples_per_class=60, train_ratio=0.8):
    """Download real PlantVillage images and organize into train/val split."""
    print("="*60)
    print("      DOWNLOADING REAL PLANTVILLAGE DATASET FROM GITHUB")
    print("="*60)
    
    train_dir = os.path.join(DATASET_DIR, 'train')
    val_dir = os.path.join(DATASET_DIR, 'val')

    total_downloaded = 0

    for cls in TARGET_CLASSES:
        print(f"[*] Fetching real images for class: {cls}...")
        image_urls = fetch_folder_contents(cls)
        
        if not image_urls:
            print(f"[!] Warning: No images found for {cls}, skipping.")
            continue
            
        random.seed(42)
        random.shuffle(image_urls)
        selected_urls = image_urls[:samples_per_class]

        num_train = int(len(selected_urls) * train_ratio)
        train_urls = selected_urls[:num_train]
        val_urls = selected_urls[num_train:]

        cls_train_path = os.path.join(train_dir, cls)
        cls_val_path = os.path.join(val_dir, cls)
        
        # Clear old synthetic files
        os.makedirs(cls_train_path, exist_ok=True)
        os.makedirs(cls_val_path, exist_ok=True)
        
        for f in os.listdir(cls_train_path):
            os.remove(os.path.join(cls_train_path, f))
        for f in os.listdir(cls_val_path):
            os.remove(os.path.join(cls_val_path, f))

        # Download train images
        for i, url in enumerate(train_urls):
            filename = f"real_{cls}_train_{i+1}.jpg"
            dest = os.path.join(cls_train_path, filename)
            urllib.request.urlretrieve(url, dest)
            total_downloaded += 1

        # Download val images
        for i, url in enumerate(val_urls):
            filename = f"real_{cls}_val_{i+1}.jpg"
            dest = os.path.join(cls_val_path, filename)
            urllib.request.urlretrieve(url, dest)
            total_downloaded += 1

        print(f"[+] Downloaded {len(train_urls)} train and {len(val_urls)} val real photos for '{cls}'")

    # Download one real test leaf image for root directory
    test_url = selected_urls[0] if selected_urls else None
    if test_url:
        sample_path = os.path.join(os.path.dirname(__file__), 'sample_leaf.jpg')
        urllib.request.urlretrieve(test_url, sample_path)
        print(f"[+] Replaced 'sample_leaf.jpg' with real PlantVillage leaf photo!")

    print("="*60)
    print(f"[+] Real PlantVillage Dataset Download Complete! Total images: {total_downloaded}")
    print("="*60)

if __name__ == "__main__":
    download_real_dataset(samples_per_class=300, train_ratio=0.8)
