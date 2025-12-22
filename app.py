# app.py
import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ---------------- CONFIG ----------------
MODEL_PATH = os.path.join(os.getcwd(), "plantvillage_cnn_64.h5")
UPLOAD_FOLDER = os.path.join("static", "uploads")
IMG_SIZE = (64, 64)

CONFIDENCE_THRESHOLD = 0.7   # Disease confidence threshold
GREEN_THRESHOLD = 0.02       # Plant detection threshold

# ---------------- APP ----------------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- LOAD MODEL ----------------
model = load_model(MODEL_PATH)
# Warm-up the model to avoid first-request delay
model.predict(np.zeros((1, 64, 64, 3)))

# ---------------- LABELS (FULL PLANTVILLAGE 38 CLASSES) ----------------
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
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

# ---------------- PLANT CHECK ----------------
def is_plant_image(img_array):
    r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
    green_pixels = (g > r) & (g > b) & (g > 0.3)
    green_ratio = np.sum(green_pixels) / green_pixels.size
    return green_ratio > GREEN_THRESHOLD

# ---------------- ROUTES ----------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return render_template("index.html", prediction="No file uploaded")

    file = request.files["file"]
    if file.filename == "":
        return render_template("index.html", prediction="No file selected")

    # Save uploaded image
    path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(path)

    # Preprocess image
    img = load_img(path, target_size=IMG_SIZE)
    arr_raw = img_to_array(img) / 255.0

    # STEP 1: Plant check
    if not is_plant_image(arr_raw):
        return render_template("index.html", prediction="Not a plant image", img_path=path)

    # STEP 2: Model prediction
    preds = model.predict(np.expand_dims(arr_raw, axis=0))[0]
    idx = int(np.argmax(preds))
    prob = float(preds[idx])

    # STEP 3: Confidence & safe label mapping
    if prob < CONFIDENCE_THRESHOLD:
        label = "Plant detected, disease not confident"
    elif idx >= len(labels):
        label = "Unknown plant disease"
    else:
        label = labels[idx]

    return render_template(
        "index.html",
        prediction=label,
        confidence=f"{prob*100:.2f}%",
        img_path=path
    )

# ---------------- MAIN ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
