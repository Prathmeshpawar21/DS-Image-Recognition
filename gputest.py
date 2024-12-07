# import os
# os.environ["TF_ENABLE_DIRECTML"] = "1"


# import tensorflow as tf

# # Check for available GPUs
# print("Available GPUs:", tf.config.list_physical_devices('GPU'))

# # Perform a test computation
# a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
# b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
# c = tf.matmul(a, b)
# print("Matrix multiplication result:\n", c)



# Initialize the webcam
webcam = cv2.VideoCapture(1)  # Open the webcam; index 1 can be changed if you have multiple cameras or different devices
# You may need to use `0` if the default camera is being used

# Load the Haar Cascade for face detection (pre-trained model for detecting faces)
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'  # Load the Haar Cascade file for face detection
face_cascade = cv2.CascadeClassifier(haar_file)  # Create a classifier for face detection using the loaded file

# Function to preprocess the image and extract features for the model
def extract_features(image):
    feature = np.array(image)  # Convert the image into a numpy array
    feature = feature.reshape(1, 48, 48, 1)  # Reshape the image to match the input shape of the model (1, 48, 48, 1)
    return feature / 255.0  # Normalize the pixel values to a range of 0 to 1 for better model performance

# Function that captures frames from the webcam, detects faces, and predicts emotions
def detect_face_and_emotion():
    while True:  # Infinite loop to keep processing frames
        ret, frame = webcam.read()  # Capture a frame from the webcam
        if not ret:  # If reading the frame failed, continue to the next iteration
            continue

        # Convert the frame to grayscale, as face detection works better on grayscale images
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale image using the face_cascade
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # The scale factor (1.3) and minimum neighbors (5) help with detection accuracy

        # Loop through each detected face and predict the emotion
        for (x, y, w, h) in faces:  # Iterate through all the faces detected in the image
            face = gray[y:y + h, x:x + w]  # Extract the face region from the image
            face = cv2.resize(face, (48, 48))  # Resize the face to 48x48 as required by the model
            face = extract_features(face)  # Extract features from the face for prediction
            pred = model.predict(face)  # Make a prediction using the loaded model
            predicted_label = labels[pred.argmax()]  # Get the emotion label with the highest probability from the model's output

            # Draw a rectangle around the detected face and put the predicted emotion label above it
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw a blue rectangle around the face
            cv2.putText(frame, predicted_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # Put the predicted label in red text

        # Convert the processed frame into JPEG format to send to the frontend (browser)
        ret, jpeg = cv2.imencode('.jpg', frame)  # Encode the frame as JPEG
        if ret:  # If the encoding was successful
            frame = jpeg.tobytes()  # Convert the JPEG image to bytes

        # Yield the frame as a part of a multipart HTTP response (this is used for streaming the video to the frontend)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')  # This sends the image to the client as part of a video stream
