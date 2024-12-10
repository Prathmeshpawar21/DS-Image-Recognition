from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
from tensorflow.keras.models import model_from_json
from flask_cors import CORS
import logging

app = Flask(__name__)
CORS(app)  # Enable CORS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@app.route('/')
def index():
    return render_template('index.html')
    
# Load emotion model
def load_emotion_model():
    try:
        json_file = open("emothionDetectorGPU.json", "r")
        model_json = json_file.read()
        json_file.close()
        model = model_from_json(model_json)
        model.load_weights("emothionDetectorGPU.h5")
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

# Emotion, Age, and Gender labels
emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}
age_ranges = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_labels = ['Male', 'Female']

# Face cascade for detecting faces
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Extract features for emotion detection
def extract_features(image):
    try:
        feature = np.array(image)
        feature = feature.reshape(1, 48, 48, 1)  # Reshape for the model input
        return feature / 255.0  # Normalize
    except Exception as e:
        logging.error(f"Error extracting features: {e}")
        return None

# Function to process the frame received from the client
@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        # Get the base64 image data from the POST request
        data = request.get_json()
        image_data = data['image']
        
        # Decode the base64 image
        img_bytes = base64.b64decode(image_data.split(',')[1])
        np_img = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_img, flags=1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        results = []
        
        for (x, y, w, h) in faces:
            try:
                face = gray[y:y + h, x:x + w]
                face_resized = cv2.resize(face, (48, 48))

                emotion_features = extract_features(face_resized)
                emotion_pred = emotion_model.predict(emotion_features)
                emotion_label = emotion_labels[np.argmax(emotion_pred)]

                face_rgb = frame[y:y + h, x:x + w]
                face_rgb = cv2.resize(face_rgb, (227, 227))

                blob = cv2.dnn.blobFromImage(face_rgb, 0.007843, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=True)
                gender_net.setInput(blob)
                gender_pred = gender_net.forward()
                gender_label = gender_labels[np.argmax(gender_pred[0])]

                age_net.setInput(blob)
                age_pred = age_net.forward()
                age_label = age_ranges[np.argmax(age_pred[0])]

                results.append({
                    'emotion': str(emotion_label),
                    'gender': str(gender_label),
                    'age': str(age_label),
                    'face_coords': [int(x), int(y), int(w), int(h)]
                })
            except Exception as e:
                logging.error(f"Error during face processing: {e}")

        return jsonify(results)
    except Exception as e:
        logging.error(f"Error processing frame: {e}")
        return jsonify({'error': 'Failed to process frame'}), 400



if __name__ == '__main__':
    # app.run(debug=True)
    try:
        app.run(debug=True, host="0.0.0.0", port=5000)
    except Exception as e:
        logging.error(f"Error running app: {e}")




















