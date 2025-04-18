import os
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from PIL import Image
import torch


class FER2013Dataset(Dataset):
    def __init__(self, img_root, csv_file, transform=None):
        self.img_root = img_root
        self.csv_file = csv_file
        self.transform = transform

        # Read CSV
        self.data_frame = pd.read_csv(csv_file)

        # Load image folder for class to index mapping
        self.image_folder = datasets.ImageFolder(root=img_root)
        self.class_to_idx = self.image_folder.class_to_idx

        self.image_paths = []
        self.expression_labels = []
        self.landmarks = []
        self.bounding_boxes = []

        for idx, row in self.data_frame.iterrows():
            img_name = row['filename']
            label_str = row['label']
            label = self.class_to_idx.get(label_str)

            # Skip if label not found in class_to_idx
            if label is None:
                print(f"⚠️ Label '{label_str}' not in folder structure, skipping.")
                continue

            landmarks = []
            for i in range(1, 479):
                x = row[f'x{i}']
                y = row[f'y{i}']
                landmarks.append((x, y))

            # Read bounding box values
            if all(col in row for col in ['xmin', 'ymin', 'width', 'height']):
                bbox = [row['xmin'], row['ymin'], row['width'], row['height']]
            else:
                bbox = [0, 0, 0, 0]  # default dummy bbox if missing

            img_path = os.path.join(img_root, label_str, img_name)
            if os.path.exists(img_path):
                self.image_paths.append(img_path)
                self.expression_labels.append(label)
                self.landmarks.append(landmarks)
                self.bounding_boxes.append(bbox)
            else:
                print(f"⚠️ Image not found: {img_path}")
        print("🔍 class_to_idx mapping:", self.class_to_idx)
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        expression_label = torch.tensor(self.expression_labels[idx], dtype=torch.long)

        # Open image and get original size
        image = Image.open(img_path)
        orig_width, orig_height = image.size

        # Process landmarks
        landmarks = torch.tensor(self.landmarks[idx], dtype=torch.float).view(-1)

        # Get bbox info: xmin, ymin, width, height
        row = self.data_frame.iloc[idx]
        x_min = row['xmin']
        y_min = row['ymin']
        width = row['width']
        height = row['height']

        # Convert to x_max, y_max
        x_max = x_min + width
        y_max = y_min + height

        # Rescale bbox to match transformed image (224x224)


        bbox = torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float)

        if self.transform:
            image = self.transform(image)

        return image, expression_label, landmarks, bbox

