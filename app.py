from flask import Flask, render_template, Response
import cv2
from tensorflow.keras.models import model_from_json
import numpy as np
import os
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

def find_working_webcam():
    """Attempt to find a working webcam by checking indices 0 to 5."""
    for i in range(5):  # Check first 5 possible webcam indices
        webcam = cv2.VideoCapture(i)
        if webcam.isOpened():
            logging.info(f"Webcam found at index {i}")
            return webcam
    logging.error("No working webcam found. Please check your hardware.")
    return None

try:
    emotion_model = load_emotion_model()
    age_net, gender_net = load_dnn_models()
    webcam = find_working_webcam()
    if webcam is None:
        raise RuntimeError("Webcam not available. Make sure the server has a connected webcam.")
except Exception as e:
    logging.error(f"Initialization failed: {e}")
    webcam = None  # To prevent further use in the app

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

def detect_face_emotion_gender_age():
    if not webcam:
        logging.error("Webcam not initialized.")
        return

    while True:
        try:
            ret, frame = webcam.read()
            if not ret:
                logging.warning("Failed to read frame from webcam.")
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                try:
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
                except Exception as e:
                    logging.error(f"Error during face processing: {e}")

            ret, jpeg = cv2.imencode('.jpg', frame)
            if ret:
                frame = jpeg.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        except Exception as e:
            logging.error(f"Error in streaming: {e}")
            break

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(detect_face_emotion_gender_age(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        port = int(os.environ.get("PORT", 5000))
        app.run(host="0.0.0.0", port=port)
    except Exception as e:
        logging.error(f"Error running app: {e}")
