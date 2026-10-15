"""
Plant Disease Detection System - 128x128 MobileNetV2 Training Pipeline
------------------------------------------------------------------------
Trains MobileNetV2 transfer learning model at 128x128 resolution.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# Set random seeds for reproducibility
tf.keras.utils.set_random_seed(42)

# Configuration Parameters - 128x128 input shape
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.0003

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
STATIC_DIR = os.path.join(BASE_DIR, 'static')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

def load_datasets():
    """Load train and validation datasets at 128x128 resolution."""
    train_path = os.path.join(DATASET_DIR, 'train')
    val_path = os.path.join(DATASET_DIR, 'val')

    print("[*] Loading training dataset from 'dataset/train' at 128x128...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_path,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=True,
        seed=42
    )
    print("[*] Loading validation dataset from 'dataset/val' at 128x128...")
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_path,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=False
    )
    class_names = train_ds.class_names
    print(f"[+] Detected {len(class_names)} classes: {class_names}")
    return train_ds, val_ds, class_names

def build_data_augmentation():
    """Data augmentation pipeline."""
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ], name="data_augmentation")

def build_mobilenet_model(num_classes):
    """
    Build MobileNetV2 Architecture at 128x128.
    Includes embedded preprocess_input inside model graph.
    Inputs must be raw float32 arrays in range [0, 255].
    """
    data_aug = build_data_augmentation()

    inputs = layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), name="raw_image_input")
    x = data_aug(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        include_top=False,
        weights='imagenet'
    )
    # Fine-tune top 30 layers
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    x = base_model(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name="output_probabilities")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="PlantVillage_MobileNetV2_128")
    return model

def plot_training_history(history):
    """Save loss and accuracy plot to static directory."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy', color='#10b981', linewidth=2)
    plt.plot(epochs_range, val_acc, label='Validation Accuracy', color='#3b82f6', linewidth=2)
    plt.title('Training & Validation Accuracy', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower right')

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss', color='#ef4444', linewidth=2)
    plt.plot(epochs_range, val_loss, label='Validation Loss', color='#f59e0b', linewidth=2)
    plt.title('Training & Validation Loss', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right')

    plt.tight_layout()
    plot_path = os.path.join(STATIC_DIR, 'training_history.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[+] Saved training curves to: {plot_path}")

def evaluate_model(model, val_ds, class_names):
    """Calculate accuracy, precision, recall, F1-score, and plot confusion matrix."""
    print("\n[*] Evaluating MobileNetV2 model on validation dataset...")
    
    y_true = []
    y_pred = []

    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    report_text = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)

    print("\n" + "="*60)
    print("                    MODEL EVALUATION REPORT")
    print("="*60)
    print(report_text)
    print("="*60)

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    accuracy = float(np.mean(y_true == y_pred))

    metrics_summary = {
        "accuracy": float(accuracy),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_score_weighted": float(f1),
        "per_class": report_dict
    }

    metrics_json_path = os.path.join(MODEL_DIR, 'evaluation_metrics.json')
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"[+] Saved evaluation metrics to: {metrics_json_path}")

    # Plot Confusion Matrix using Matplotlib
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(cm, cmap=plt.cm.Greens)
    fig.colorbar(cax)

    clean_labels = [c.replace('___', ' - ').replace('_', ' ') for c in class_names]
    ax.set_xticks(np.arange(len(clean_labels)))
    ax.set_yticks(np.arange(len(clean_labels)))
    ax.set_xticklabels(clean_labels, rotation=45, ha='left')
    ax.set_yticklabels(clean_labels)

    for i in range(len(clean_labels)):
        for j in range(len(clean_labels)):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black', fontweight='bold')

    plt.title('Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()

    cm_path = os.path.join(STATIC_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[+] Saved confusion matrix heatmap to: {cm_path}")

    return metrics_summary

def train_pipeline(epochs=EPOCHS):
    """Main training execution pipeline."""
    train_ds, val_ds, class_names = load_datasets()

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    class_indices = {i: name for i, name in enumerate(class_names)}
    class_indices_path = os.path.join(MODEL_DIR, 'class_indices.json')
    with open(class_indices_path, 'w') as f:
        json.dump(class_indices, f, indent=4)
    print(f"[+] Saved class mappings to: {class_indices_path}")

    model = build_mobilenet_model(num_classes=len(class_names))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    keras_model_path = os.path.join(MODEL_DIR, 'plant_disease_model.keras')

    cb_list = [
        callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
        callbacks.ModelCheckpoint(keras_model_path, monitor='val_accuracy', save_best_only=True, verbose=1)
    ]

    print(f"\n[*] Starting model training for {epochs} epochs at 128x128 resolution...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=cb_list
    )

    model.save(keras_model_path)
    print(f"[+] Model training complete! Saved trained model to '{keras_model_path}'")

    plot_training_history(history)
    metrics = evaluate_model(model, val_ds, class_names)

    return model, metrics

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train MobileNetV2 CNN for Plant Disease Detection")
    parser.add_argument('--epochs', type=int, default=EPOCHS, help="Number of training epochs")
    args = parser.parse_args()

    train_pipeline(epochs=args.epochs)
