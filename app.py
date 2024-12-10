from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
from tensorflow.keras.models import model_from_json
from flask_cors import CORS
import logging
import gc

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load emotion model
def load_emotion_model():
    try:
        with open("notebook/emothionDetectorGPU.json", "r") as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json)
        model.load_weights("notebook/emothionDetectorGPU.h5")
        logging.info("Emotion model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Error loading emotion model: {e}")
        raise

# Load Age and Gender DNN models
def load_dnn_models():
    try:
        age_net = cv2.dnn.readNetFromCaffe("./savedModel/age_deploy.prototxt", "./savedModel/age_net.caffemodel")
        gender_net = cv2.dnn.readNetFromCaffe("./savedModel/gender_deploy.prototxt", "./savedModel/gender_net.caffemodel")
        logging.info("Age and gender detection models loaded successfully.")
        return age_net, gender_net
    except Exception as e:
        logging.error(f"Error loading DNN models: {e}")
        raise

# Initialize models
try:
    emotion_model = load_emotion_model()
    age_net, gender_net = load_dnn_models()
except Exception as e:
    logging.error(f"Initialization failed: {e}")
    exit(1)

# Labels
emotion_labels = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprise"}
age_ranges = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
gender_labels = ["Male", "Female"]

# Face detection using Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Extract features for emotion detection
def extract_features(image):
    try:
        feature = np.array(image).reshape(1, 48, 48, 1)
        return feature / 255.0
    except Exception as e:
        logging.error(f"Error extracting features: {e}")
        return None

# Process frame
@app.route("/process_frame", methods=["POST"])
def process_frame():
    try:
        # Get the base64 image data from the POST request
        data = request.get_json()
        image_data = data.get("image")
        
        if not image_data:
            return jsonify({"error": "No image data provided"}), 400

        # Decode the base64 image
        img_bytes = base64.b64decode(image_data.split(",")[1])
        np_img = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Failed to decode image"}), 400

        # Convert to grayscale and detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        results = []
        for (x, y, w, h) in faces:
            try:
                face = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(face, (48, 48))
                emotion_features = extract_features(face_resized)
                if emotion_features is None:
                    continue

                emotion_pred = emotion_model.predict(emotion_features)
                emotion_label = emotion_labels[np.argmax(emotion_pred)]

                # Gender and Age detection
                face_rgb = cv2.resize(frame[y:y+h, x:x+w], (227, 227))
                blob = cv2.dnn.blobFromImage(face_rgb, 0.007843, (227, 227), 
                                             (78.4263377603, 87.7689143744, 114.895847746), swapRB=True)

                gender_net.setInput(blob)
                gender_label = gender_labels[np.argmax(gender_net.forward()[0])]

                age_net.setInput(blob)
                age_label = age_ranges[np.argmax(age_net.forward()[0])]

                # Append results
                results.append({
                    "emotion": emotion_label,
                    "gender": gender_label,
                    "age": age_label,
                    "face_coords": [int(x), int(y), int(w), int(h)]
                })

            except Exception as e:
                logging.error(f"Error during face processing: {e}")

        return jsonify(results)
    
      # Free memory

    except Exception as e:
        logging.error(f"Error processing frame: {e}")
        return jsonify({"error": "Failed to process frame"}), 400

# Serve index page
@app.route("/")
def index():
    return render_template("index.html")

# Run the app
if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, debug=True)
    except Exception as e:
        logging.error(f"Error running app: {e}")