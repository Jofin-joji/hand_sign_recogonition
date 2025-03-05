import cv2
import numpy as np
import tensorflow as tf
import joblib
import mediapipe as mp
import requests
import json
import threading
import itertools

# Load the trained deep learning model
model = tf.keras.models.load_model('gesture_recognition_model.h5')

# Load the saved scaler and label encoder
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize MediaPipe Drawing
mp_drawing = mp.solutions.drawing_utils

# Initialize the video capture (webcam)
cap = cv2.VideoCapture(0)

# Google Gemini API configuration
API_KEY = 'AIzaSyBBRfLUx8CRebFItO20Uuvhz0zA6mMeYpE'  # Replace with your actual API key
API_URL = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={API_KEY}'

detected_words = []
api_response = ""

# Lock for thread synchronization
api_lock = threading.Lock()

def extract_landmarks(frame):
    """Function to extract hand landmarks from a webcam frame and draw them."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Draw the hand landmarks and connections on the frame
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
            
            hand_landmarks = []
            for lm in landmarks.landmark:
                hand_landmarks.extend([lm.x, lm.y, lm.z])  # Flatten the landmarks into a list
            return hand_landmarks
    return None

def call_gemini_api(words):
    """Function to call the Google Gemini API and update the shared API response."""
    global api_response
    payload = {
        "contents": [
            {
                "parts": [{"text": " ".join(words)}]
            }
        ]
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        # Send the API request
        response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
        
        # Debugging: Log the response status and content
        print("Response Status Code:", response.status_code)
        print("Response Text:", response.text)
        
        if response.status_code == 200:
            response_data = response.json()
            sentence = response_data.get("contents", [{"parts": [{"text": "No meaningful sentence returned."}]}])[0]["parts"][0]["text"]
        else:
            sentence = f"Error in generating sentence. Status Code: {response.status_code}"
    except Exception as e:
        sentence = f"API Error: {str(e)}"
    
    # Update the shared response with a lock
    with api_lock:
        api_response = sentence

def remove_adjacent_duplicates(words):
    """Function to remove adjacent duplicate words."""
    return [key for key, _ in itertools.groupby(words)]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    landmarks = extract_landmarks(frame)
    
    if landmarks:
        # Normalize landmarks using the loaded scaler
        landmarks = np.array(landmarks).reshape(1, -1)  # Reshape for model input
        landmarks_scaled = scaler.transform(landmarks)  # Normalize the landmarks
        
        # Reshape for CNN input
        landmarks_scaled = landmarks_scaled.reshape(landmarks_scaled.shape[0], landmarks_scaled.shape[1], 1)
        
        # Predict gesture using the trained deep learning model
        prediction = model.predict(landmarks_scaled)
        predicted_label = np.argmax(prediction)  # Get the index of the highest probability
        
        # Decode the predicted label using the label_encoder
        predicted_gesture = label_encoder.inverse_transform([predicted_label])[0]
        
        # Add the detected gesture to the words list
        detected_words.append(predicted_gesture)
        
        # Remove adjacent duplicate words
        detected_words = remove_adjacent_duplicates(detected_words)
        
        # Display the detected gesture on the frame
        cv2.putText(frame, f"Detected: {predicted_gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # If we have collected a certain number of words, call the API in a separate thread
        if len(detected_words) >= 5:  # Adjust the threshold as needed
            # Call the API in a separate thread
            threading.Thread(target=call_gemini_api, args=(detected_words,)).start()
            detected_words = []  # Clear the words after API call
        
        # Display the sentence from the API response
        with api_lock:
            cv2.putText(frame, f"Sentence: {api_response}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    # Show the frame with the prediction and hand landmarks
    cv2.imshow("Gesture Recognition with Grammar", frame)
    
    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
