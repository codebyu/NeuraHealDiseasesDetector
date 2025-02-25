from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename

# Initialize Flask App
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "uploads/"
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)  # Create folder if not exists

# Load Trained AI Model
model = tf.keras.models.load_model("disease_detection_model.h5")

# Allowed Image Formats
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 🖼 Render HTML Form for User Image Upload
@app.route('/')
def home():
    return render_template('index.html')

# 🚀 Process Image & Predict Disease
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Load & Process Image
        img = cv2.imread(file_path)
        img = cv2.resize(img, (224, 224)) / 255.0  # Normalize
        img_array = np.expand_dims(img, axis=0)

        # Make Prediction
        prediction = model.predict(img_array)
        disease_labels = ["Healthy", "Acne", "Rash", "Jaundice", "Stress"]
        detected_disease = disease_labels[np.argmax(prediction)]

        return jsonify({"Detected Disease": detected_disease})

    return jsonify({"error": "Invalid file type. Only PNG, JPG, JPEG allowed."}), 400

# 🌐 Handle GET Requests (Prevent Errors)
@app.route('/predict', methods=['GET'])
def handle_get_request():
    return jsonify({"error": "Use POST request with an image file"}), 405

if __name__ == '__main__':
    app.run(debug=True)
