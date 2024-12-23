# Face Emotion, Gender, and Age Detection System 😄🧑‍🦱

## Project Overview 🌟

The **Face Emotion, Gender, and Age Detection System** aims to analyze faces in images or video streams and detect the person's emotional state, gender, and approximate age. This project uses convolutional neural networks (CNNs) and trained models for robust and efficient face analysis.

## Screenshots

![App Screenshot](https://raw.githubusercontent.com/Prathmeshpawar21/__resources/refs/heads/main/SS/face-modified_1.png)

## Features 🎯

- Real-time face emotion recognition ( **Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise** etc.).

- Gender prediction based on facial features.
- Age estimation based on facial appearance.
- Use of OpenCV for real-time image and video processing.
- Easy-to-use interface for quick analysis.

## Technologies Used 🛠️

- **Python**
- **Libraries:** OpenCV, TensorFlow, Keras, NumPy, Matplotlib
- **Trained Models:**  
  - **Emotion Detection Model**
  - **Gender Detection Model** 
  - **Age Detection Model**
  
## Project Structure 🗂️

```
├── notebook
│   ├── genderAge.ipynb
│   └── research.ipynb
├── savedModel
│   ├── age_deploy.prototxt
│   └── age_net.caffemodel
|        .
|        .
|        .
├── static
│   ├── script.js
│   └── style.css
│   └── logo.png
├── templates
│   ├── index.html
├── .gitignore
├── LICENSE
├── README.md
├── app.py
├── emotionDetectorGPU.h5
├── emotionDetectorGPU.json
├── gputest.py
├── index.py
├── pythonEnv.txt
├── requirements.txt
├── tensorcpu.txt
├── vercel.json
└── wsgi.py

```

## Setup Instructions ⚙️

1. Clone the repository:
   ```bash
   git clone https://github.com/Prathmeshpawar21/DS-Image-Recognition.git
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

   **OR**

3. Create and activate a virtual environment using Anaconda:
   ```bash
   conda create -n venv python=3.10.15 -y
   conda activate venv
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Run the application:
   ```bash
   python app.py
   ```

6. Open your browser and navigate to `http://127.0.0.1:<port>`.

## How It Works 🧠

1. **Input:** Live webcam 
2. **Face Detection:** The system detects faces using OpenCV and a Haar Cascade or Dlib-based face detector.
3. **Emotion, Gender, and Age Prediction:** The system uses trained deep learning models to predict the emotion, gender, and age of the detected faces.
4. **Output:**  
   - **Emotion:** Predicted emotion ( **Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise** etc.).  
   - **Gender:** Predicted gender (Male/Female).  
   - **Age:** Estimated age range (e.g., 25-35 years).

## Results 🏆

- **Accuracy:** High accuracy in real-time face emotion, gender, and age detection.
- **Efficiency:** Real-time processing with minimal delay.

## Future Improvements 🚀

- Improve emotion recognition with more facial expressions.
- Add support for recognizing multiple faces in a single frame.
- Implement voice-based emotion detection for further analysis.
- Integrate with mobile platforms for real-time mobile face analysis.

## Contributing 🤝

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch-name`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch-name`).
5. Open a Pull Request.


## 🔗 Links

[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://prathameshpawar-mu.vercel.app/)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/prathameshpawar21/)

## License 📜
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

  
