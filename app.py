from flask import Flask, render_template, request, send_file
from pdf2image import convert_from_bytes
import requests
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

def predict_image(img_path):
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
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        label = "Normal"
        confidence = prediction
    else:
        label = "Cancer"
        confidence = 1 - prediction

    return label, round(float(confidence) * 100, 2)

@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    if request.method == "POST":
        file = request.files.get("file")
        allowed_extensions = {"png", "jpg", "jpeg", "gif", "bmp", "tiff", "pdf"}
        if file and file.filename != '':
            filename = file.filename
            ext = filename.rsplit('.', 1)[-1].lower()
            if ext not in allowed_extensions:
                error = "Invalid file type. Please upload an image or PDF."
            elif ext == 'pdf':
                try:
                    pdf_bytes = file.read()
                    images = convert_from_bytes(pdf_bytes)
                    if not images:
                        error = "No images found in PDF."
                    else:
                        pil_img = images[0]
                        label, confidence = predict_image_pil(pil_img)
                        pil_img.save(os.path.join("static", "pdf_preview.jpg"))
                        from flask import url_for
                        image_url = url_for('static', filename="pdf_preview.jpg")
                        return render_template("index.html", label=label, confidence=confidence, image_path=image_url, model_version=model_version)
                except Exception as e:
                    error = f"Error processing PDF: {str(e)}"
            else:
                filepath = os.path.join("static", filename)
                file.save(filepath)
                try:
                    label, confidence = predict_image(filepath)
                except Exception as e:
                    error = f"Error processing image: {str(e)}"
                else:
                    from flask import url_for
                    image_url = url_for('static', filename=filename)
                    return render_template("index.html", label=label, confidence=confidence, image_path=image_url, model_version=model_version)
        else:
            error = "No file provided. Please upload an image or PDF."
        return render_template("index.html", error=error, model_version=model_version)
    return render_template("index.html", model_version=model_version)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
