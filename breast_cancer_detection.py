# ============================================================
# BREAST CANCER DETECTION - Complete Code
# Dataset: Dataset_BUSI_with_GT (saved on Desktop)
# Model: Pretrained VGG16 (Transfer Learning)
# ============================================================

# ── CELL 1: Imports ─────────────────────────────────────────
import os
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report,
                              roc_curve, auc)
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import VGG16 as VGG16_pretrained
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ── CELL 2: Dataset Path (Desktop) ──────────────────────────
import os

# Works for Windows Desktop automatically
desktop = os.path.join(os.path.expanduser("~"), "Desktop")
dataset = os.path.join(desktop, "Dataset_BUSI_with_GT")

print(f"Dataset path: {dataset}")
print(f"Exists: {os.path.exists(dataset)}")

# ── CELL 3: Config ──────────────────────────────────────────
imagesize   = (224, 224)
categories  = ["benign", "malignant"]
labels      = {"benign": 0, "malignant": 1}

# ── CELL 4: Load & Preprocess Images ────────────────────────
X = []
y = []

for cate in categories:
    folderpath = os.path.join(dataset, cate)
    for image_file in tqdm(os.listdir(folderpath), desc=cate):
        # Skip mask images (_mask files)
        if "mask" in image_file.lower():
            continue
        imgpath = os.path.join(folderpath, image_file)
        img = cv2.imread(imgpath)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, imagesize)
        img = img / 255.0
        X.append(img)
        y.append(labels[cate])

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

print(f"\nTotal images loaded: {len(X)}")
print(f"Benign: {np.sum(y == 0)}, Malignant: {np.sum(y == 1)}")

# ── CELL 5: Visualize Sample Images ─────────────────────────
def plot_images(X, y, categories, n=10):
    fig, axes = plt.subplots(1, n, figsize=(30, 10))
    for i in range(n):
        ax = axes[i]
        img = X[i]
        ax.imshow(img)
        if isinstance(y[i], (np.ndarray, list)):
            label_index = np.argmax(y[i])
        else:
            label_index = int(y[i])
        ax.set_title(categories[label_index], fontsize=12)
        ax.axis("off")
    plt.suptitle("Sample Dataset Images", fontsize=16)
    plt.tight_layout()
    plt.savefig("sample_images.png", dpi=100)
    plt.show()

plot_images(X, y, ["Benign", "Malignant"])

# ── CELL 6: One-hot encode & Train/Test Split ────────────────
y_cat = to_categorical(y, num_classes=2)
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y_cat, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {len(Xtrain)} | Test: {len(Xtest)}")

# ── CELL 7: Data Augmentation ───────────────────────────────
datagen = ImageDataGenerator(
    rotation_range=15,
    horizontal_flip=True,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    fill_mode='nearest'
)
datagen.fit(Xtrain)

# ── CELL 8: Build Pretrained VGG16 Model ────────────────────
def build_transfer_model(input_shape=(224, 224, 3), num_classes=2):
    base_model = VGG16_pretrained(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    # Freeze all base layers
    for layer in base_model.layers:
        layer.trainable = False

    # Unfreeze last 4 layers for fine-tuning
    for layer in base_model.layers[-4:]:
        layer.trainable = True

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

model = build_transfer_model()
model.summary()

# ── CELL 9: Callbacks ───────────────────────────────────────
early_stop = EarlyStopping(
    monitor='val_loss', patience=5,
    restore_best_weights=True, verbose=1
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5,
    patience=3, min_lr=1e-7, verbose=1
)

# ── CELL 10: Train Model ─────────────────────────────────────
history = model.fit(
    datagen.flow(Xtrain, ytrain, batch_size=16),
    epochs=25,
    validation_data=(Xtest, ytest),
    callbacks=[early_stop, reduce_lr]
)

# ── CELL 11: Plot Training History ──────────────────────────
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training & Validation Accuracy")

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training & Validation Loss")

plt.tight_layout()
plt.savefig("training_history.png", dpi=100)
plt.show()

# ── CELL 12: Evaluate Model ──────────────────────────────────
test_loss, test_acc = model.evaluate(Xtest, ytest, verbose=0)
print(f"\nTest Accuracy : {test_acc  * 100:.2f}%")
print(f"Test Loss     : {test_loss:.4f}")

# ── CELL 13: Predictions ─────────────────────────────────────
y_pred       = model.predict(Xtest)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true       = np.argmax(ytest, axis=1)

# ── CELL 14: Classification Report ──────────────────────────
print("\nClassification Report:")
print(classification_report(
    y_true, y_pred_classes,
    target_names=["Benign", "Malignant"]
))

# ── CELL 15: Confusion Matrix ────────────────────────────────
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Benign", "Malignant"],
            yticklabels=["Benign", "Malignant"])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=100)
plt.show()

# ── CELL 16: ROC Curve & AUC ─────────────────────────────────
y_pred_probs = model.predict(Xtest)[:, 1]
fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color="blue", label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=100)
plt.show()

print(f"ROC AUC Score: {roc_auc:.4f}")

# ── CELL 17: Grad-CAM Visualization ─────────────────────────
def grad_cam(model, img_array, layer_name='block5_conv3'):
    # Get the VGG16 base model layer
    base = model.layers[0]
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[base.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), predictions.numpy()


def show_gradcam(model, img_array, original_img, layer_name='block5_conv3'):
    heatmap, preds = grad_cam(model, img_array, layer_name)
    label = ["Benign", "Malignant"][np.argmax(preds)]
    confidence = np.max(preds) * 100

    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    original_uint8 = np.uint8(255 * original_img)
    superimposed = cv2.addWeighted(original_uint8, 0.6, heatmap_color, 0.4, 0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original_img)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    axes[1].imshow(heatmap_resized, cmap='jet')
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis('off')
    axes[2].imshow(superimposed)
    axes[2].set_title("Superimposed")
    axes[2].axis('off')
    plt.suptitle(f"Grad-CAM | Prediction: {label} ({confidence:.1f}%)", fontsize=14)
    plt.tight_layout()
    plt.savefig("gradcam_output.png", dpi=100)
    plt.show()
    print(f"Prediction: {label} | Confidence: {confidence:.2f}%")


# Test Grad-CAM on a sample test image
sample_img   = Xtest[5]
sample_input = np.expand_dims(sample_img, axis=0)
show_gradcam(model, sample_input, sample_img)

# ── CELL 18: Save Model ──────────────────────────────────────
model.save("breast_cancer_vgg16.h5")
print("Model saved as breast_cancer_vgg16.h5")
