import cv2
import mediapipe as mp
import pandas as pd
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Directory containing the subfolders for each hand sign (update the path to your main folder)
base_folder_path = "C:\\Users\\JOFIN\\Desktop\\sign_language\\New folder" # Update path

# Check if the base folder exists and is a directory
if not os.path.exists(base_folder_path):
    print(f"Error: The folder {base_folder_path} does not exist!")
elif not os.path.isdir(base_folder_path):
    print(f"Error: {base_folder_path} is not a directory!")
else:
    print(f"Path is correct: {base_folder_path}")

# Output CSV file
output_csv = "hand_signs_landmarks.csv"

# Prepare output data
data = []

# Get the list of all subfolders (one folder per hand sign)
hand_sign_folders = [f for f in os.listdir(base_folder_path) if os.path.isdir(os.path.join(base_folder_path, f))]

# Debug: Print the folders found
print("Folders found:", hand_sign_folders)

# Process each hand sign folder
for folder in hand_sign_folders:
    folder_path = os.path.join(base_folder_path, folder)
    
    # Get all images inside the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    # Process each image in the folder
    for image_file in image_files:
        # Construct the image path
        image_path = os.path.join(folder_path, image_file)
        
        # Load the image
        image = cv2.imread(image_path)
        
        # Check if image is loaded properly
        if image is None:
            print(f"Error: Unable to load image {image_path}")
            continue
        
        # Convert the image to RGB (MediaPipe requires RGB)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image to extract landmarks
        result = hands.process(rgb_image)
        
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Extract landmarks (x, y, z for each point)
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                
                # The folder name represents the hand sign label
                label = folder
                
                # Append landmarks with the label
                data.append([*landmarks, label])
        else:
            print(f"No hands detected in image {image_file}")

# Release MediaPipe resources
hands.close()

# Check if data is populated
print(f"Number of data points: {len(data)}")
if len(data) > 0:
    print(f"Sample data point: {data[0]}")
else:
    print("No data extracted!")

# Convert data to a DataFrame
columns = [f"landmark_{i}" for i in range(63)] + ["label"]  # 63 because 21 landmarks * 3 coordinates (x, y, z)
df = pd.DataFrame(data, columns=columns)

# Save to CSV
df.to_csv(output_csv, index=False)

print(f"Dataset saved to {output_csv}")
