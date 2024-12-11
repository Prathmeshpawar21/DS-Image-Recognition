from flask import Flask, render_template, request, jsonify
import base64
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
import logging
import os
import gc  # Garbage collector

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load models
def load_emotion_model():
    try:
        json_file = open("notebook/emothionDetectorGPU.json", "r")
        model_json = json_file.read()
        json_file.close()
        model = model_from_json(model_json)
        model.load_weights("notebook/emothionDetectorGPU.h5")
        logging.info("Emotion model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Error loading emotion model: {e}")
        raise

def load_dnn_models():
    try:
        age_net = cv2.dnn.readNetFromCaffe("./savedModel/age_deploy.prototxt", "./savedModel/age_net.caffemodel")
        gender_net = cv2.dnn.readNetFromCaffe("./savedModel/gender_deploy.prototxt", "./savedModel/gender_net.caffemodel")
        logging.info("Age and gender detection models loaded successfully.")
        return age_net, gender_net
    except Exception as e:
        logging.error(f"Error loading DNN models: {e}")
        raise

# Load models
emotion_model = load_emotion_model()
age_net, gender_net = load_dnn_models()

# Define labels and Haar cascade
emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}
age_ranges = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_labels = ['Male', 'Female']
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

def extract_features(image):
    try:
        feature = np.array(image)
        feature = feature.reshape(1, 48, 48, 1)
        return feature / 255.0
    except Exception as e:
        logging.error(f"Error extracting features: {e}")
        return None

def process_frame(frame):
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            face_resized = cv2.resize(face, (48, 48))
            face_features = extract_features(face_resized)
            if face_features is None:
                continue
            emotion_pred = emotion_model.predict(face_features)
            emotion_label = emotion_labels[emotion_pred.argmax()]
            face_rgb = frame[y:y + h, x:x + w]
            blob = cv2.dnn.blobFromImage(face_rgb, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
            gender_net.setInput(blob)
            gender_pred = gender_net.forward()
            gender_label = gender_labels[gender_pred[0].argmax()]
            age_net.setInput(blob)
            age_pred = age_net.forward()
            age_label = age_ranges[age_pred[0].argmax()]
            label = f"{emotion_label}, {gender_label}, {age_label}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes() if ret else None
    except Exception as e:
        logging.error(f"Error processing frame: {e}")
        return None
    finally:
        # Release memory using garbage collector
        gc.collect()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        # Get the base64 image from the request
        data = request.json
        dataURL = data.get('image')
        if not dataURL:
            return jsonify({'error': 'No image data received'}), 400

        # Decode the base64 image
        header, encoded = dataURL.split(',', 1)
        nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'error': 'Failed to decode image'}), 400

        # Process the image
        processed_image = process_frame(img)
        if processed_image is not None:
            # Encode the processed image back to base64
            processed_image_base64 = base64.b64encode(processed_image).decode('utf-8')
            return jsonify({'processed_image': processed_image_base64})
        else:
            return jsonify({'error': 'Failed to process image'}), 500
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        return jsonify({'error': 'Internal server error'}), 500
    finally:
        # Release memory using garbage collector
        gc.collect()

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=8000)  # for gunicorn deployment purpose
        # app.run(host='0.0.0.0', port=5000)  # for Localhost
    except Exception as e:
        logging.error(f"Error running app: {e}")