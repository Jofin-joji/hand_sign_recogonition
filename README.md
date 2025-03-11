# Real-Time Hand Gesture Recognition

This project implements real-time hand gesture recognition using **MediaPipe Hands** and a trained deep learning model. It captures video from a webcam, detects hand landmarks, and predicts gestures using a pre-trained TensorFlow model.

## Features

- **Real-time hand detection** using MediaPipe.
- **Hand landmark visualization**, including nodes and connections.
- **Gesture classification** using a deep learning model.
- **Box enclosing the detected hand** for better visualization.
- **Live gesture predictions** displayed on the video feed.

## Installation

To set up the environment, install the required dependencies:

```bash
pip install opencv-python numpy tensorflow joblib mediapipe


Usage
Run the following command to start the real-time gesture recognition:

bash
Copy
Edit
python real_time_detection.py
Controls:
Press q to exit the program.
Model & Preprocessing
The deep learning model (gesture_recognition_model.h5) is trained on hand landmarks.
scaler.pkl is used for normalizing the extracted landmarks.
label_encoder.pkl maps model predictions to gesture labels.
Expected Output
The webcam feed will show detected hands with landmark nodes and connections.
A bounding box will enclose the detected hand.
The predicted gesture will be displayed on the screen.
Troubleshooting
If gestures are misclassified, ensure that gesture_recognition_model.h5 is correctly trained and compatible with the dataset.
If the webcam doesn't start, check camera permissions or try using a different video source.
Acknowledgments
MediaPipe Hands for real-time hand tracking.
TensorFlow for deep learning-based gesture classification.
OpenCV for image processing.
```
