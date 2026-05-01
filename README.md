# Breast Cancer Detection — AI Diagnostic Web App

Binary ultrasound image classifier (Benign vs Malignant) using
Transfer Learning (VGG16) with Grad-CAM visualization, deployed as a Flask web app.

---

## Project Structure

```
breast_cancer/
├── breast_cancer_detection.py   # Jupyter notebook code (training)
├── app.py                       # Flask web application
├── templates/
│   └── index.html               # Frontend UI
├── requirements.txt             # Dependencies
└── breast_cancer_vgg16.h5       # Saved model (generated after training)
```

---

## Dataset

- **Name:** BUSI (Breast Ultrasound Images Dataset)
- **Classes:** Benign, Malignant
- **Images:** ~1,312 ultrasound images
- **Source:** https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset
- **Save to Desktop as:** `Dataset_BUSI_with_GT`

Folder structure inside dataset:
```
Dataset_BUSI_with_GT/
├── benign/
│   ├── benign (1).png
│   ├── benign (1)_mask.png   ← automatically skipped
│   └── ...
└── malignant/
    ├── malignant (1).png
    └── ...
```

---

## Step 1 — Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Step 2 — Train the Model

Run `breast_cancer_detection.py` in Jupyter Notebook or as a script:

```bash
python breast_cancer_detection.py
```

This will:
- Load and preprocess the BUSI dataset from your Desktop
- Train pretrained VGG16 with transfer learning + data augmentation
- Generate: confusion matrix, ROC curve, classification report, Grad-CAM
- Save model as `breast_cancer_vgg16.h5`

---

## Step 3 — Run Flask App

```bash
python app.py
```

Open browser at: http://localhost:5000

---

## Step 4 — Deploy (Render.com — Free)

1. Push project to GitHub
2. Go to https://render.com → New Web Service
3. Connect your GitHub repo
4. Set:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
5. Add `gunicorn` to requirements.txt
6. Deploy — get a live URL like `https://breast-cancer-detection.onrender.com`

---

## Features

- Pretrained VGG16 (ImageNet weights) with fine-tuned top layers
- Data augmentation (rotation, flip, zoom, shift)
- EarlyStopping + ReduceLROnPlateau callbacks
- Classification report (precision, recall, F1-score)
- Confusion matrix + ROC-AUC curve
- Grad-CAM heatmap visualization
- Flask web app with drag-and-drop image upload
- Real-time prediction with confidence scores

---

## Resume Bullet Points

```
Breast Cancer Detection | Python, TensorFlow, Keras, Flask, OpenCV
Binary ultrasound image classifier (Benign vs Malignant) using transfer
learning on the BUSI dataset (1,312 images).

• Fine-tuned pretrained VGG16 (ImageNet) with custom dense layers, dropout,
  and data augmentation to improve generalization
• Achieved 70%+ validation accuracy; evaluated using confusion matrix,
  ROC-AUC curve, and classification report (precision, recall, F1)
• Implemented Grad-CAM visualization to highlight tumor regions in
  ultrasound images for model interpretability
• Deployed as a Flask web app with drag-and-drop image upload returning
  real-time predictions with confidence scores and Grad-CAM overlay
```
