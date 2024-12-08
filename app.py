from flask import Flask, render_template, Response
import cv2
from keras.models import model_from_json
import numpy as np
import gunicorn
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

app = Flask(__name__)

# Load the emotion detection model
json_file = open("notebook/emothionDetectorGPU.json", "r")
model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(model_json)
emotion_model.load_weights("notebook/emothionDetectorGPU.h5")

# Labels for emotions
emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# Load the gender and age detection models
age_net = cv2.dnn.readNetFromCaffe("./savedModel/age_deploy.prototxt", "./savedModel/age_net.caffemodel")
gender_net = cv2.dnn.readNetFromCaffe("./savedModel/gender_deploy.prototxt", "./savedModel/gender_net.caffemodel")

# Age and gender labels
age_ranges = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_labels = ['Male', 'Female']

def find_working_webcam():
    """Attempt to find a working webcam by checking indices 0 to 5."""
    for i in range(5):  # Check first 5 possible webcam indices
        webcam = cv2.VideoCapture(i)
        if webcam.isOpened():
            print(f"Webcam found at index {i}")
            return webcam
    raise RuntimeError("No working webcam found. Please check your hardware.")

# Initialize the webcam dynamically
webcam = find_working_webcam()

haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)


def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0


def detect_face_emotion_gender_age():
    while True:
        ret, frame = webcam.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # Crop and preprocess the face for emotion detection
            face = gray[y:y + h, x:x + w]
            face_resized = cv2.resize(face, (48, 48))
            face_features = extract_features(face_resized)
            emotion_pred = emotion_model.predict(face_features)
            emotion_label = emotion_labels[emotion_pred.argmax()]

            # Prepare the face for age and gender detection
            face_rgb = frame[y:y + h, x:x + w]  # Crop the face in RGB format
            blob = cv2.dnn.blobFromImage(face_rgb, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

            # Predict gender
            gender_net.setInput(blob)
            gender_pred = gender_net.forward()
            gender_label = gender_labels[gender_pred[0].argmax()]

            # Predict age
            age_net.setInput(blob)
            age_pred = age_net.forward()
            age_label = age_ranges[age_pred[0].argmax()]

            # Draw rectangle and text
            label = f"{emotion_label}, {gender_label}, {age_label}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangle
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)  # Yellow text

        # Encode frame for streaming
        ret, jpeg = cv2.imencode('.jpg', frame)
        if ret:
            frame = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(detect_face_emotion_gender_age(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
    
    # app.run()
