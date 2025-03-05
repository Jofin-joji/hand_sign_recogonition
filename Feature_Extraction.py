import cv2
import mediapipe as mp
import csv
import os
import time  # For adding delay

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Function to extract landmarks
def extract_landmarks(image):
    """
    Extract normalized hand landmarks from the image.
    Returns a flattened list of 63 values (21 landmarks * 3 coordinates).
    """
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks:
        landmarks = []
        for lm in results.multi_hand_landmarks[0].landmark:
            landmarks.extend([lm.x, lm.y, lm.z])  # Flatten landmarks
        return landmarks
    return None

# Function to save landmarks to CSV
def save_landmarks_to_csv(output_file, landmarks, label):
    """
    Saves the landmark data and the associated label to a CSV file.
    """
    if not os.path.exists(output_file):
        # Write the header only if the file does not exist
        with open(output_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            header = [f"x{i+1}" for i in range(21)] + [f"y{i+1}" for i in range(21)] + [f"z{i+1}" for i in range(21)] + ["label"]
            writer.writerow(header)

    # Append the data
    with open(output_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(landmarks + [label])

# Main function to capture and save data
def record_gestures(output_file, gesture_label, num_samples=50):
    """
    Records hand gestures and saves the landmarks along with their label to a CSV file.
    """
    cap = cv2.VideoCapture(0)  # Open webcam
    sample_count = 0

    print(f"Recording gestures for: {gesture_label}")
    while cap.isOpened() and sample_count < num_samples:
        ret, frame = cap.read()
        if not ret:
            break

        landmarks = extract_landmarks(frame)
        if landmarks:
            # Save landmarks with the label
            save_landmarks_to_csv(output_file, landmarks, gesture_label)
            sample_count += 1
            print(f"Sample {sample_count}/{num_samples} saved for '{gesture_label}'")

            # Add delay between frame captures
            time.sleep(0.5)  # Delay of 0.5 seconds (adjust as needed)

        # Display the frame with instructions
        cv2.putText(frame, f"Recording: {gesture_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Press 'q' to quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Gesture Recording", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Finished recording {gesture_label}.")

# Run the recorder for specific gestures
if __name__ == "__main__":
    output_csv = "hand_landmarks.csv"
    gestures = [ "Hello" , "See you later" , "I or Me" , "Father" , "Mother" , "Yes" , "No" , "Help" , "Please" , "Thank You" , "Want" , "What?" , "Dog", "Cat","Again or Repeat", "Eat/Food", "Milk", "More", "Go To", " Bathroom","Fine","Like","Learn","Sign","Finish or Done"]  # Define the gestures to record

    for gesture in gestures:
        print(f"\nPrepare to record gesture: {gesture}")
        input("Press Enter when ready...")
        record_gestures(output_csv, gesture, num_samples=50)

    print("All gestures recorded and saved to:", output_csv)
