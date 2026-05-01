import os
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from PIL import Image
from io import BytesIO
import base64

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# ── Load Model ───────────────────────────────────────────────
MODEL_PATH = "breast_cancer_vgg16.keras"
model = tf.keras.models.load_model(MODEL_PATH)
CATEGORIES = ["Benign", "Malignant"]
IMG_SIZE   = (224, 224)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_bytes):
    """Convert image bytes to model-ready numpy array."""
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0), img

def generate_gradcam(model, img_array, layer_name='block5_conv3'):
    """Generate Grad-CAM heatmap."""
    try:
        base = model.layers[0]
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[base.get_layer(layer_name).output, model.output]
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            pred_index    = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        grads        = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap      = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap      = tf.squeeze(heatmap)
        heatmap      = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()
    except Exception:
        return None

def overlay_gradcam(original_pil, heatmap):
    """Overlay Grad-CAM heatmap on original image, return base64 string."""
    original_cv = cv2.cvtColor(np.array(original_pil.resize(IMG_SIZE)), cv2.COLOR_RGB2BGR)
    heatmap_resized  = cv2.resize(heatmap, IMG_SIZE)
    heatmap_uint8    = np.uint8(255 * heatmap_resized)
    heatmap_color    = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    superimposed     = cv2.addWeighted(original_cv, 0.6, heatmap_color, 0.4, 0)
    _, buffer        = cv2.imencode('.png', superimposed)
    return base64.b64encode(buffer).decode('utf-8')

# ── Routes ───────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Use PNG, JPG, JPEG, BMP, or TIFF'}), 400

    try:
        image_bytes          = file.read()
        img_array, pil_image = preprocess_image(image_bytes)

        # Prediction
        preds           = model.predict(img_array)
        pred_index      = int(np.argmax(preds[0]))
        label           = CATEGORIES[pred_index]
        confidence      = float(np.max(preds[0])) * 100
        benign_prob     = float(preds[0][0]) * 100
        malignant_prob  = float(preds[0][1]) * 100

        # Grad-CAM
        heatmap         = generate_gradcam(model, img_array)
        gradcam_b64     = None
        if heatmap is not None:
            gradcam_b64 = overlay_gradcam(pil_image, heatmap)

        # Original image as base64
        buffered = BytesIO()
        pil_image.resize(IMG_SIZE).save(buffered, format="PNG")
        original_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return jsonify({
            'prediction'    : label,
            'confidence'    : f"{confidence:.2f}",
            'benign_prob'   : f"{benign_prob:.2f}",
            'malignant_prob': f"{malignant_prob:.2f}",
            'original_image': original_b64,
            'gradcam_image' : gradcam_b64,
            'risk_level'    : 'High' if label == 'Malignant' else 'Low'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model': MODEL_PATH})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
