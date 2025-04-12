import os
import cv2
import numpy as np
import face_alignment
from tqdm import tqdm

# Initialize the face alignment model
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)


# Function to extract landmarks from images
def extract_landmarks(image_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for img_name in tqdm(os.listdir(image_folder)):
        img_path = os.path.join(image_folder, img_name)
        if img_path.lower().endswith(('png', 'jpg', 'jpeg')):
            image = cv2.imread(img_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB as required by the model

            # Get the landmarks using face-alignment
            landmarks = fa.get_landmarks(image_rgb)
            if landmarks:
                # Save the landmarks as a NumPy array (or you could save it in other formats like CSV)
                landmarks = landmarks[0]  # Assuming one face per image, take the first detected face
                np.save(os.path.join(output_folder, f'{os.path.splitext(img_name)[0]}.npy'), landmarks)
            else:
                print(f"No landmarks detected for {img_name}")


# Folder paths
image_folder = r"C:\Users\miner\OneDrive\Desktop\introtocs\Python_Project\training\archive_\Test\surprise"  # Update with your image folder path
output_folder = r"C:\Users\miner\OneDrive\Desktop\introtocs\Python_Project\training\archive_\Test_nyp\surprise"  # Update with your desired output folder for saving landmarks

# Extract landmarks
extract_landmarks(image_folder, output_folder)
