from flask import Flask, render_template, request, send_file

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)


# Load model and get version
MODEL_PATH = "blood_cancer_model_best.h5"
model = tf.keras.models.load_model(MODEL_PATH)
model_version = None
try:
    import h5py
    with h5py.File(MODEL_PATH, 'r') as f:
        model_version = f.attrs.get('keras_version', 'Unknown')
except Exception:
    model_version = 'Unknown'

IMG_SIZE = 224

def is_valid_blood_sample(pil_img):
    try:
        # Step 2: MICROSCOPY IMAGE CHECK
        img = pil_img.convert('RGB')
        img_array = np.array(img)
        
        # 1. Brightness: mean pixel value must be > 50 (not too dark)
        mean_val = np.mean(img_array)
        if mean_val <= 50:
            return False, "Image is too dark to analyse."

        # 2. Texture: std deviation must be > 12 (not blank/solid)
        std_val = np.std(img_array)
        if std_val <= 12:
            return False, "Image appears blank or solid-colored."

        # 3. Color: RGB channel difference must be > 18 (not grayscale)
        # Using the sum of absolute differences between channel means
        mean_r = np.mean(img_array[:, :, 0])
        mean_g = np.mean(img_array[:, :, 1])
        mean_b = np.mean(img_array[:, :, 2])
        channel_diff = abs(mean_r - mean_g) + abs(mean_g - mean_b) + abs(mean_b - mean_r)
        if channel_diff <= 18:
            return False, "Grayscale images not accepted. H&E-stained color slides required."

        # 4. Background: at least 10% of pixels must be near-white (>200 value)
        # Check pixels where ALL R, G, B channels are above 200
        is_white_mask = np.all(img_array > 200, axis=2)
        white_pixels_ratio = np.sum(is_white_mask) / (img_array.shape[0] * img_array.shape[1])
        if white_pixels_ratio < 0.10:
            return False, "Does not appear to be a microscopy slide. Natural photos not accepted."

        # 5. H&E stain signature: at least 4% of pixels must show purple or pink
        # Purple: (blue + red) > green * 1.1 AND not over-white
        # Pink: red > (blue + green) * 1.1 AND not over-white
        is_not_white = ~is_white_mask
        purple_mask = ((img_array[:, :, 2].astype(float) + img_array[:, :, 0].astype(float)) > (img_array[:, :, 1].astype(float) * 1.1)) & is_not_white
        pink_mask = (img_array[:, :, 0].astype(float) > ((img_array[:, :, 2].astype(float) + img_array[:, :, 1].astype(float)) * 1.05)) & is_not_white
        he_signature = np.sum(purple_mask | pink_mask) / (img_array.shape[0] * img_array.shape[1])
        
        if he_signature < 0.04:
            return False, "No blood smear staining detected. Upload an H&E-stained microscopy image."

        # 6. Not overexposed: less than 85% of pixels near 255
        overexposed_pixels = np.sum(img_array > 250) / (img_array.size / 3.0)
        if overexposed_pixels >= 0.85:
            return False, "Image is overexposed and contains too little detail."

        # 7. Not a natural photo: reject if mid-tone pixels > 80% AND H&E signature < 8%
        # Mid-tones: 50-200
        midtone_pixels = np.sum((img_array > 50) & (img_array < 200)) / (img_array.size / 3.0)
        if midtone_pixels > 0.80 and he_signature < 0.08:
            return False, "Looks like a regular photograph. Microscopy slides only."

        return True, "Valid"
    except Exception as e:
        return False, f"Error analyzing image structure: {str(e)}"

def predict_image_pil(pil_img):
    img = pil_img.convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0][0]
    
    # Step 3: CANCER PREDICTION labels
    if prediction > 0.5:
        label = "Normal"
        confidence = prediction
    else:
        label = "Cancer (Leukemia)"
        confidence = 1 - prediction
    return label, round(float(confidence) * 100, 2)

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    return predict_image_pil(img)

@app.route("/", methods=["GET", "POST"])
def index():
    errors = []
    results = []
    
    if request.method == "POST":
        files = request.files.getlist("file")
        
        if not files or all(f.filename == '' for f in files):
            errors.append("No file provided. Please upload an image.")
        else:
            from flask import url_for
            for file in files:
                if file.filename == '':
                    continue
                filename = file.filename
                safe_name = filename.replace(" ", "_")
                ext = safe_name.rsplit('.', 1)[-1].lower()
                
                # Step 1: FILE VALIDATION
                if ext == 'pdf':
                    errors.append(f"{filename}: Only image files are accepted. PDFs are not supported.")
                    continue
                    
                allowed_extensions = {"png", "jpg", "jpeg", "bmp", "tiff"}
                if ext not in allowed_extensions:
                    errors.append(f"{filename}: Only image files are accepted. PDFs are not supported." if ext == 'pdf' else f"{filename}: Unsupported file type.")
                    continue
                    
                filepath = os.path.join("static", safe_name)
                file.save(filepath)
                try:
                    from PIL import Image
                    pil_img = Image.open(filepath)
                    is_valid, reason = is_valid_blood_sample(pil_img)
                    if not is_valid:
                        errors.append(f"{filename}: {reason}")
                        continue
                        
                    label, confidence = predict_image_pil(pil_img)
                    image_url = url_for('static', filename=safe_name)
                    results.append({
                        'filename': filename,
                        'label': label,
                        'confidence': confidence,
                        'image_path': image_url
                    })
                except Exception as e:
                    errors.append(f"{filename}: Error processing sample ({str(e)}).")
                        
        return render_template("index.html", results=results, errors=errors, model_version=model_version)
    return render_template("index.html", model_version=model_version)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
