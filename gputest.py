# import cv2
# import requests

# # Open webcam (change device index if needed)
# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     print("Webcam not found!")
#     exit()

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to grab frame!")
#         break
    
#     # Optional: Resize or process the frame
#     # Send frame to Flask server via HTTP request or WebSocket

#     _, img_encoded = cv2.imencode('.jpg', frame)
#     response = requests.post("http://localhost:5000/upload", files={"image": img_encoded.tobytes()})
#     # You can send the frame to your Flask app via POST or WebSocket.

# cap.release()


import cv2
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Cannot open camera")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cannot receive frame")
            break
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()