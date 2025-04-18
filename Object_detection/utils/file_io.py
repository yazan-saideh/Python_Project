import os
import numpy as np
import pandas as pd

# Base folder where all expression folders are stored (each contains .npy files of landmarks and bounding boxes)
base_folder = r""
img_root = r""

output_csv = r""

# Prepare list to hold all data rows
all_data = []

# Loop over each expression folder (e.g., anger, happy, etc.)
expression_folders = os.listdir(base_folder)
for expression in expression_folders:
    folder_path = os.path.join(base_folder, expression)
    if not os.path.isdir(folder_path):
        continue  # skip files, only process folders

    for file in os.listdir(folder_path):
        if file.endswith('.npy'):
            landmark_path = os.path.join(folder_path, file)
            data = np.load(landmark_path, allow_pickle=True).item()  # Load dictionary (landmarks and bbox)

            landmarks = data['landmarks']
            bbox = data['bounding_box']

            # Flatten landmarks to 1D array
            flattened_landmarks = landmarks.flatten()

            # Get bounding box information
            bbox_values = [
                bbox['xmin'], bbox['ymin'], bbox['width'], bbox['height']
            ]

            base_filename = os.path.splitext(file)[0]
            image_folder = os.path.join(img_root, expression)
            actual_filename = None
            for ext in ['.jpg', '.jpeg', '.png']:
                candidate_path = os.path.join(image_folder, base_filename + ext)
                if os.path.exists(candidate_path):
                    actual_filename = base_filename + ext
                    break

            if actual_filename is None:
                print(f"⚠️ Image not found for: {base_filename} in {image_folder}")
                continue  # Skip this file if the image doesn't exist

            # Create a row: [filename, x1, y1, ..., x68, y68, xmin, ymin, width, height, expression_label]
            row = [actual_filename] + flattened_landmarks.tolist() + bbox_values + [expression]
            all_data.append(row)

# Generate column names
landmark_columns = [f'x{i//2+1}' if i % 2 == 0 else f'y{i//2+1}' for i in range(478*2)]
bbox_columns = ['xmin', 'ymin', 'width', 'height']
columns = ['filename'] + landmark_columns + bbox_columns + ['label']

# Save as DataFrame
df = pd.DataFrame(all_data, columns=columns)

# Save to CSV
df.to_csv(output_csv, index=False)
print(f"✅ CSV saved at: {output_csv}")
