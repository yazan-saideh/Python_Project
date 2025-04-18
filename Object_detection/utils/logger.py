import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

# Initialize MediaPipe Face Detection and Face Mesh
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                                  min_detection_confidence=0.5)


# Function to extract landmarks and bounding box from images using MediaPipe
def extract_landmarks_and_bbox(image_folder, output_folder, target_size=(224, 224)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all images in the folder
    for img_name in tqdm(os.listdir(image_folder)):
        img_path = os.path.join(image_folder, img_name)

        if img_path.lower().endswith(('png', 'jpg', 'jpeg')):
            # Read the image
            image = cv2.imread(img_path)

            # Resize image to the target size (224x224)
            image_resized = cv2.resize(image, target_size)
            image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

            # Process the image with Face Detection
            detection_results = face_detection.process(image_rgb)

            if detection_results.detections:
                # Get the first face's bounding box (if detected)
                bbox = detection_results.detections[0].location_data.relative_bounding_box
                bbox_coords = {
                    'xmin': bbox.xmin * image_resized.shape[1],
                    'ymin': bbox.ymin * image_resized.shape[0],
                    'width': bbox.width * image_resized.shape[1],
                    'height': bbox.height * image_resized.shape[0]
                }

                # Process the image with Face Mesh
                mesh_results = face_mesh.process(image_rgb)

                if mesh_results.multi_face_landmarks:
                    # Get the first face's landmarks
                    landmarks = mesh_results.multi_face_landmarks[0]
                    landmark_coords = np.array([[lm.x, lm.y] for lm in landmarks.landmark])

                    # Scale landmarks to pixel coordinates for the resized image
                    landmark_coords[:, 0] *= image_resized.shape[1]  # x coordinates
                    landmark_coords[:, 1] *= image_resized.shape[0]  # y coordinates

                    # Save bounding box and landmarks as a dictionary
                    data = {
                        'bounding_box': bbox_coords,
                        'landmarks': landmark_coords
                    }

                    # Save data as a .npy file (bounding box + landmarks)
                    np.save(os.path.join(output_folder, f'{os.path.splitext(img_name)[0]}.npy'), data)
                else:
                    print(f"No landmarks detected for {img_name}")
                    os.remove(img_path)  # Optional: delete image if no landmarks detected
            else:
                print(f"No bounding box detected for {img_name}")
                os.remove(img_path)  # Optional: delete image if no bounding box detected


# Set your image folder and output folder paths
image_folder = r"C:\Users\miner\OneDrive\Desktop\introtocs\Python_Project\training\archive_\MMAFEDB\test\surprise"  # Replace with your image folder path
output_folder = r"C:\Users\miner\OneDrive\Desktop\introtocs\Python_Project1\training\archive_\Test_nyp\surprise"  # Replace with the folder where you want to save the data

# Extract landmarks and bounding box and save as .npy files
extract_landmarks_and_bbox(image_folder, output_folder, target_size=(224, 224))
