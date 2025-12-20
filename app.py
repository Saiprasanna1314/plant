# app.py
import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Path to your model (must be in project root)
MODEL_PATH = os.path.join(os.getcwd(), "plantvillage_cnn_64.h5")
UPLOAD_FOLDER = os.path.join("static", "uploads")
IMG_SIZE = (64, 64)

CONFIDENCE_THRESHOLD = 0.7   # disease confidence threshold
GREEN_THRESHOLD = 0.02       # plant detection threshold

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
model = load_model(MODEL_PATH)

# Hardcoded class labels (replace with all your model's classes)
labels = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    # Add all other classes from your model
]

# 🌿 Plant (green) check
def is_plant_image(img_array):
    r = img_array[:, :, 0]
    g = img_array[:, :, 1]
    b = img_array[:, :, 2]

    green_pixels = (g > r) & (g > b) & (g > 0.3)
    green_ratio = np.sum(green_pixels) / green_pixels.size

    return green_ratio > GREEN_THRESHOLD

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    f = request.files['file']
    if f.filename == '':
        return "No file selected", 400

    # Save uploaded image
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
    f.save(save_path)

    # Load image
    img = load_img(save_path, target_size=IMG_SIZE)
    arr_raw = img_to_array(img) / 255.0

    # 🥇 STEP 1: PLANT CHECK
    if not is_plant_image(arr_raw):
        return render_template(
            'index.html',
            prediction="Not a plant image",
            img_path=save_path
        )

    # Prepare for model
    arr = np.expand_dims(arr_raw, axis=0)

    # 🥈 STEP 2: DISEASE PREDICTION
    preds = model.predict(arr)[0]
    idx = int(np.argmax(preds))
    prob = float(preds[idx])

    # 🥉 STEP 3: CONFIDENCE CHECK
    if prob < CONFIDENCE_THRESHOLD:
        label = "Not a plant image"
    else:
        label = labels[idx]

    return render_template(
        'index.html',
        prediction=label,
        confidence=f"{prob * 100:.2f}%",
        img_path=save_path
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
