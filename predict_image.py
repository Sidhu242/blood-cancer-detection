import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load the trained model
model = load_model('blood_cancer_model_best.h5')

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

image_path = 'dataset/cancer/Sap_148 (1).jpg'  # Using a real image file
img = preprocess_image(image_path)
prediction = model.predict(img)

if prediction[0][0] > 0.5:
    print("Cancer detected")
else:
    print("Normal")
