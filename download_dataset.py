"""
Plant Village Dataset Downloader & Organizer
--------------------------------------------
This script provides helper routines to download and organize the real 
PlantVillage dataset for training the CNN model.

PlantVillage is a publicly available dataset of ~54,000 images covering 38 plant disease classes.
"""

import os
import sys
import zipfile
import urllib.request

DATASET_DIR = os.path.join(os.path.dirname(__file__), 'dataset')
KAGGLE_DATASET_HANDLE = "emmarex/plantdisease"
PLANTVILLAGE_ZIP_URL = "https://github.com/spMohanty/PlantVillage-Dataset/archive/refs/heads/master.zip"

def download_via_kaggle():
    """Download dataset using kagglehub if available."""
    try:
        import kagglehub
        print("[*] Downloading PlantVillage dataset using kagglehub...")
        path = kagglehub.dataset_download(KAGGLE_DATASET_HANDLE)
        print(f"[✓] Dataset downloaded to: {path}")
        print("[*] Copy or link images to your 'dataset/' folder.")
        return True
    except ImportError:
        print("[!] kagglehub module not installed. Run 'pip install kagglehub' to use Kaggle automated download.")
        return False
    except Exception as e:
        print(f"[!] Kaggle download failed: {e}")
        return False

def show_dataset_instructions():
    """Print step-by-step instructions for getting real PlantVillage dataset."""
    print("=" * 70)
    print("           PLANTVILLAGE DATASET INSTRUCTIONS")
    print("=" * 70)
    print("Option 1: Automated Download via Kaggle")
    print("  1. Run: pip install kagglehub")
    print("  2. Execute this script: python download_dataset.py --kaggle")
    print("\nOption 2: Direct Download from Kaggle Web")
    print("  1. Go to: https://www.kaggle.com/datasets/emmarex/plantdisease")
    print("  2. Download 'plantdisease.zip'")
    print("  3. Extract contents into the 'dataset/' folder so structure looks like:")
    print("     dataset/")
    print("       ├── Tomato___Early_blight/")
    print("       ├── Tomato___Late_blight/")
    print("       ├── Tomato___healthy/")
    print("       ├── Potato___Early_blight/")
    print("       └── Potato___healthy/")
    print("\nOption 3: Use Synthetic Demo Dataset")
    print("  Run 'python create_sample_dataset.py' to generate test leaf images immediately!")
    print("=" * 70)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--kaggle":
        success = download_via_kaggle()
        if not success:
            show_dataset_instructions()
    else:
        show_dataset_instructions()
