import os
import numpy as np
import cv2
import tensorflow as tf
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Dataset
desktop  = os.path.join(os.path.expanduser("~"), "Desktop")
dataset  = os.path.join(desktop, "Dataset_BUSI_with_GT")
categories = ["benign", "malignant"]
labels     = {"benign": 0, "malignant": 1}

# Load images
X, y = [], []
for cate in categories:
    folderpath = os.path.join(dataset, cate)
    for image_file in tqdm(os.listdir(folderpath), desc=cate):
        if "mask" in image_file.lower():
            continue
        imgpath = os.path.join(folderpath, image_file)
        img = cv2.imread(imgpath)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        X.append(img)
        y.append(labels[cate])

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)
print(f"Images loaded: {len(X)}")

y_cat = to_categorical(y, num_classes=2)
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y_cat, test_size=0.2, random_state=42, stratify=y
)

# Augmentation
datagen = ImageDataGenerator(
    rotation_range=15, horizontal_flip=True,
    zoom_range=0.1, width_shift_range=0.1,
    height_shift_range=0.1, fill_mode='nearest'
)
datagen.fit(Xtrain)

# Build model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
for layer in base_model.layers:
    layer.trainable = False
for layer in base_model.layers[-4:]:
    layer.trainable = True

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(2, activation='softmax')
])
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy', metrics=['accuracy'])

# Train
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr  = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)

history = model.fit(
    datagen.flow(Xtrain, ytrain, batch_size=16),
    epochs=25,
    validation_data=(Xtest, ytest),
    callbacks=[early_stop, reduce_lr]
)

# Save
model.save("breast_cancer_vgg16.h5")
print("Model saved successfully as breast_cancer_vgg16.h5")

# Evaluate
test_loss, test_acc = model.evaluate(Xtest, ytest, verbose=0)
print(f"Test Accuracy: {test_acc * 100:.2f}%")
