import cv2
import requests

# Open webcam (change device index if needed)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Webcam not found!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame!")
        break
    
    # Optional: Resize or process the frame
    # Send frame to Flask server via HTTP request or WebSocket

    _, img_encoded = cv2.imencode('.jpg', frame)
    response = requests.post("http://localhost:5000/upload", files={"image": img_encoded.tobytes()})
    # You can send the frame to your Flask app via POST or WebSocket.

cap.release()
