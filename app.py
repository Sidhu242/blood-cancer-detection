from flask import Flask, render_template, request, send_file
from pdf2image import convert_from_bytes
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
        # Convert to RGB array
        img = pil_img.convert('RGB')
        # Resize to small image to speed up math and average out noise
        img.thumbnail((300, 300))
        img_array = np.array(img)
        
        # 1. Check if it's too dark
        mean_brightness = np.mean(img_array)
        if mean_brightness < 60: 
            return False, "Image is too dark to be a microscopic slide."
            
        std_dev = np.std(img_array)
        if std_dev < 10:
            return False, "Image lacks structural detail (possibly a graphic/screenshot)."

        # 2. Check for grayscale (x-rays, documents)
        mean_r = np.mean(img_array[:,:,0])
        mean_g = np.mean(img_array[:,:,1])
        mean_b = np.mean(img_array[:,:,2])
        channel_diff = abs(mean_r - mean_g) + abs(mean_g - mean_b) + abs(mean_r - mean_b)
        if channel_diff < 15:
             return False, "Image appears grayscale. A stained blood sample is required."

        # 3. Check color distribution: Blood smears rarely have green as the dominant channel.
        if mean_g > mean_r + 20 and mean_g > mean_b + 20:
             return False, "Unusual color profile (too green) for a blood smear."

        return True, "Valid"
    except Exception as e:
        return False, f"Could not analyze image structure: {str(e)}"

def predict_image_pil(pil_img):
    img = pil_img.convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0][0]
    if prediction > 0.5:
        label = "Normal"
        confidence = prediction
    else:
        label = "Cancer"
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
        allowed_extensions = {"png", "jpg", "jpeg", "gif", "bmp", "tiff", "pdf"}
        
        if not files or all(f.filename == '' for f in files):
            errors.append("No file provided. Please upload an image or PDF.")
        else:
            from flask import url_for
            for file in files:
                if file.filename == '':
                    continue
                filename = file.filename
                safe_name = filename.replace(" ", "_")
                ext = safe_name.rsplit('.', 1)[-1].lower()
                
                if ext not in allowed_extensions:
                    errors.append(f"{filename}: Invalid file type. Please upload an image or PDF.")
                    continue
                    
                if ext == 'pdf':
                    try:
                        pdf_bytes = file.read()
                        images = convert_from_bytes(pdf_bytes)
                        if not images:
                            errors.append(f"{filename}: No images found in PDF.")
                        else:
                            pil_img = images[0]
                            is_valid, reason = is_valid_blood_sample(pil_img)
                            if not is_valid:
                                errors.append(f"{filename} rejected: {reason}")
                                continue
                                
                            label, confidence = predict_image_pil(pil_img)
                            preview_filename = f"pdf_preview_{safe_name}.jpg"
                            pil_img.save(os.path.join("static", preview_filename))
                            image_url = url_for('static', filename=preview_filename)
                            results.append({
                                'filename': filename,
                                'label': label,
                                'confidence': confidence,
                                'image_path': image_url
                            })
                    except Exception as e:
                        errors.append(f"{filename}: Error processing PDF ({str(e)}).")
                else:
                    filepath = os.path.join("static", safe_name)
                    file.save(filepath)
                    try:
                        from PIL import Image
                        pil_img = Image.open(filepath)
                        is_valid, reason = is_valid_blood_sample(pil_img)
                        if not is_valid:
                            errors.append(f"{filename} rejected: {reason}")
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
                        errors.append(f"{filename}: Error processing image ({str(e)}).")
                        
        return render_template("index.html", results=results, errors=errors, model_version=model_version)
    return render_template("index.html", model_version=model_version)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
