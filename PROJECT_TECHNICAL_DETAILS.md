# Project Technical Details & Documentation
## Plant Disease Detection System using CNN (FloraScan AI)

---

### 1. Project Title
**Plant Disease Detection System using CNN (FloraScan AI)**

### 2. Project Overview
An end-to-end Machine Learning and Computer Vision application designed to identify plant leaf diseases in supported vegetable crops. The system incorporates data augmentation, MobileNetV2 transfer learning, model evaluation metrics (Accuracy, Precision, Recall, F1-score, Confusion Matrix), a command-line predictor, and an interactive Flask web application featuring a modern glassmorphic dark-theme user interface.

### 3. Problem Statement
Manual visual inspection of crop leaves for disease identification is labor-intensive, subject to human error, and requires specialized agricultural expertise. Automated deep learning models enable rapid, objective preliminary identification of plant pathology, allowing farmers and researchers to respond early to disease outbreaks.

### 4. Objectives
- Build a multi-class Convolutional Neural Network (CNN) trained on real plant leaf photographs.
- Implement data augmentation and transfer learning to achieve high validation accuracy (> 95%).
- Calculate comprehensive model evaluation metrics (Precision, Recall, F1-score, Confusion Matrix).
- Develop a standalone Python prediction script (`predict.py`) and a Flask web server (`app.py`).
- Implement confidence thresholding (< 60%) to handle out-of-distribution and noisy real-world images cleanly.
- Include educational care guidelines and clear non-prescriptive disclaimers.

### 5. Technologies Used
- Python 3.13.2
- Deep Learning / Computer Vision: TensorFlow, Keras, NumPy, OpenCV (Headless), Pillow
- Evaluation & Data Science: Scikit-Learn, Matplotlib, Seaborn
- Web Development: Flask, Werkzeug, HTML5, CSS3, JavaScript (ES6 AJAX)

### 6. Programming Languages
- Python
- JavaScript (Vanilla ES6)
- HTML5 & CSS3

### 7. Frameworks and Libraries
- `tensorflow==2.21.0`
- `keras==3.15.1`
- `flask==3.1.0`
- `scikit-learn==1.9.1`
- `matplotlib==3.11.1`
- `pillow==12.3.0`
- `opencv-python-headless==4.5.0`
- `werkzeug==3.1.3`

### 8. Development Environment
- Operating System: Windows
- Shell / Terminal: PowerShell / Command Prompt
- Python Environment: CPython 3.13.2 64-bit

### 9. Dataset Name
**PlantVillage Dataset**

### 10. Dataset Source
Official **spMohanty/PlantVillage-Dataset** GitHub Repository (`https://github.com/spMohanty/PlantVillage-Dataset`).

### 11. Number of Images
**1,352 Real Leaf Photographs** (1,081 training images, 271 validation images).

### 12. Number of Classes
**5 Classes**

### 13. List of All Classes
1. `Potato___Early_blight`
2. `Potato___healthy`
3. `Tomato___Early_blight`
4. `Tomato___Late_blight`
5. `Tomato___healthy`

### 14. Dataset Distribution
- `Potato___Early_blight`: 240 train, 60 val (300 total)
- `Potato___healthy`: 121 train, 31 val (152 total)
- `Tomato___Early_blight`: 240 train, 60 val (300 total)
- `Tomato___Late_blight`: 240 train, 60 val (300 total)
- `Tomato___healthy`: 240 train, 60 val (300 total)

### 15. Train/Validation/Test Split
- **Training Set**: 1,081 images (80% split)
- **Validation Set**: 271 images (20% split)
- **Test Set**: 271 validation images + unseen test sample callset

### 16. Image Preprocessing
- RGB Conversion: Images loaded via Pillow / Keras and converted to 3-channel RGB.
- Resizing: Images resized to `128x128` pixels matching `model.input_shape`.
- Tensor Formatting: Image arrays expanded to batch shape `(1, 128, 128, 3)` with `float32` data type.
- Single-Pass Preprocessing: Input arrays in range `[0, 255]` are normalized to `[-1, 1]` in a single pass by the model's embedded `preprocess_input` layer.

### 17. Data Augmentation
- `RandomFlip("horizontal")`
- `RandomRotation(0.1)`
- `RandomZoom(0.1)`

### 18. CNN / Transfer Learning Architecture
**MobileNetV2 Transfer Learning Architecture**:
- Input Layer: `(None, 128, 128, 3)`
- Data Augmentation Layer: Sequential (`RandomFlip`, `RandomRotation`, `RandomZoom`)
- Preprocessing Layer: `tf.keras.applications.mobilenet_v2.preprocess_input`
- Base Model: MobileNetV2 pre-trained on ImageNet (include_top = False)
- Global Pooling: `GlobalAveragePooling2D()`
- Normalization: `BatchNormalization()`
- Dense Hidden Layer: 128 Units (`ReLU` activation)
- Regularization: `Dropout(0.3)`
- Output Layer: 5 Units (`Softmax` activation)

### 19. MobileNetV2 Architecture Details
- Depthwise Separable Convolutions reducing computational complexity.
- Top 30 layers unfrozen for fine-tuning on plant disease feature maps.
- Inverted residual blocks with bottleneck feature representations.

### 20. Input Image Size
`128 x 128` pixels (3 RGB channels)

### 21. Output Layer
5 Neurons with `Softmax` activation function returning class probability distributions.

### 22. Activation Functions
- Hidden Layers: `ReLU` (Rectified Linear Unit)
- Output Layer: `Softmax`

### 23. Loss Function
Categorical Crossentropy (`categorical_crossentropy`)

### 24. Optimizer
Adam (`learning_rate = 0.0003`)

### 25. Learning Rate
`0.0003` (Initial learning rate with `ReduceLROnPlateau` scheduler factor `0.5`, patience `3`).

### 26. Batch Size
`16`

### 27. Number of Epochs
`10` Max Epochs

### 28. Early Stopping
- Monitor: `val_accuracy`
- Patience: `5` Epochs
- `restore_best_weights = True`

### 29. Model Training Process
1. Load images using `tf.keras.utils.image_dataset_from_directory`.
2. Cache and prefetch datasets using `tf.data.AUTOTUNE`.
3. Construct MobileNetV2 feature extractor and compile with Adam optimizer.
4. Fit model over 10 epochs using `ModelCheckpoint` to save the best model weights.
5. Save model binary to `model/plant_disease_model.keras` and index map to `model/class_indices.json`.

### 30. Final Training Accuracy
**99.44%**

### 31. Final Validation Accuracy
**96.00%** (`0.9594`)

### 32. Final Test Accuracy
**96.00%**

### 33. Precision
**96.00%** (`0.96` weighted average)

### 34. Recall
**96.00%** (`0.96` weighted average)

### 35. F1-Score
**96.00%** (`0.96` weighted average)

### 36. Confusion Matrix Results
- Potato Early Blight: 60 Correct, 0 Misclassified
- Potato Healthy: 31 Correct, 0 Misclassified
- Tomato Early Blight: 54 Correct, 6 Misclassified as Late Blight
- Tomato Late Blight: 55 Correct, 5 Misclassified as Early Blight
- Tomato Healthy: 60 Correct, 0 Misclassified

### 37. Model File Name
`model/plant_disease_model.keras`

### 38. Model Size
**22.79 MB** (23,898,579 bytes)

### 39. Prediction Pipeline
1. Load image via path or file stream.
2. Convert to RGB and resize to `(128, 128)`.
3. Format as `float32` array `(1, 128, 128, 3)`.
4. Execute `model.predict()` (embedded `preprocess_input` maps `[0, 255]` -> `[-1, 1]`).
5. Extract top-1 index, confidence percentage, and sorted class probabilities.
6. Evaluate confidence thresholding (< 60% confidence or < 15% top margin triggers `Low Confidence` flag).
7. Format crop name, condition, remedies, and educational disclaimers into output object.

### 40. Flask Backend Architecture
- Server module: `app.py`
- Route `/`: Serves `index.html` with trained status and evaluation metrics JSON.
- Route `/predict` (POST): Accepts image file upload, validates extension (`.jpg`, `.jpeg`, `.png`, `.webp`) and file size limit (16MB), runs `PlantDiseasePredictor`, and returns JSON response for AJAX requests.
- Route `/api/metrics` (GET): Exposes model evaluation metrics JSON.

### 41. Frontend Technologies
- HTML5 (Semantic structure)
- CSS3 (Glassmorphism dark theme, custom properties, CSS Grid/Flexbox, Keyframes)
- JavaScript (ES6 `fetch` API, DOM manipulation, FileReader API)
- FontAwesome 6.4 Icons
- Google Fonts (`Outfit` & `Plus Jakarta Sans`)

### 42. Website Features
- Glassmorphic dark UI dashboard.
- Live model status badge (Online / Offline).
- Real-time model accuracy metrics banner.
- Supported vegetable crops section.
- Drag-and-drop file upload zone.
- Client-side image preview with remove option.
- Interactive Submit button with spinner loading state.
- Diagnosis card displaying Crop Name, Condition, Severity, Confidence Progress Bar, Symptoms, Care Guidelines, and Prevention.
- Class Probability Distribution list.
- Confusion Matrix and Learning Curves visual cards.
- Educational Disclaimers.

### 43. Supported Crops
1. **Tomato** (*Solanum lycopersicum*)
2. **Potato** (*Solanum tuberosum*)

### 44. Supported Diseases
1. Early Blight (*Alternaria solani*)
2. Late Blight (*Phytophthora infestans*)
3. Healthy Foliage

### 45. Image Upload Restrictions
- Allowed Formats: `.jpg`, `.jpeg`, `.png`, `.webp`
- Maximum File Size: `16 MB`

### 46. Confidence Threshold
- Minimum Confidence Threshold: `60.0%`
- Minimum Probability Margin Threshold: `15.0%`

### 47. Unknown / Low-Confidence Handling
If prediction confidence is `< 60.0%` or top margin `< 15.0%`:
- Status set to: `"Low Confidence / Uncertain Identification"`
- Severity set to: `"Uncertain"`
- Warning message displayed: `"Low Prediction Confidence. The uploaded photo may contain background clutter, non-leaf objects, or an unsupported plant condition."`

### 48. Sample Prediction Results
- `real_Potato___healthy_val_1.jpg` -> Crop: Potato | Condition: healthy | Conf: 99.99%
- `real_Tomato___healthy_val_1.jpg` -> Crop: Tomato | Condition: healthy | Conf: 100.00%
- `real_Tomato___Late_blight_val_1.jpg` -> Crop: Tomato | Condition: Late blight | Conf: 99.61%
- `real_Tomato___Early_blight_val_1.jpg` -> Crop: Tomato | Condition: Early blight | Conf: 98.94%
- `real_Potato___Early_blight_val_1.jpg` -> Crop: Potato | Condition: Early blight | Conf: 100.00%

### 49. Testing Results
- Automated evaluation on 271 validation images achieved **96.00% accuracy**.
- CLI testing verified 100% execution without runtime exceptions.

### 50. Real-World Image Testing
Real-world photos correctly identified crop and condition, while ambiguous/cluttered field photos properly triggered the Low-Confidence threshold warning rather than making false claims.

### 51. Project Limitations
- Limited to 2 supported vegetable crops (Tomato and Potato).
- Highly cluttered non-leaf backgrounds can reduce feature extraction clarity.
- Extreme lighting glare or severe shadows can influence spot identification.

### 52. Future Enhancements
- Expand dataset to cover all 38 PlantVillage categories.
- Integrate a YOLO bounding-box detector to crop leaf areas automatically out of field photos.
- Export model to TensorFlow Lite (`.tflite`) for mobile app offline execution.

### 53. How to Install
```bash
cd "C:\Users\my pc\.gemini\antigravity\scratch\plant-disease-detection"
pip install -r requirements.txt
```

### 54. How to Run
```bash
python app.py
```
Open web browser to: `http://127.0.0.1:5000`

### 55. Project Folder Structure
```text
plant-disease-detection/
├── app.py                     # Flask web server
├── train.py                   # Model training script
├── predict.py                 # Single image prediction CLI & module
├── test_evaluation.py         # Test dataset evaluation script
├── test_5_images.py           # 5-image test verification script
├── download_real_plantvillage.py # PlantVillage downloader
├── create_sample_dataset.py   # Sample dataset utility
├── requirements.txt           # Python dependencies
├── README.md                  # Project overview
├── PROJECT_TECHNICAL_DETAILS.md # Complete technical specification
├── FINAL_RESULTS.md           # Machine learning results summary
├── model/                     # Trained model binaries and metrics
├── dataset/                   # PlantVillage train and validation split
├── templates/                 # HTML templates
└── static/                    # CSS, JavaScript, uploaded images, and plots
```
