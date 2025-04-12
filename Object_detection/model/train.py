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

        for idx, row in self.data_frame.iterrows():
            img_name = row['filename']
            label_str = row['label']
            label = self.class_to_idx.get(label_str)

            # Skip if label not found in class_to_idx
            if label is None:
                print(f"⚠️ Label '{label_str}' not in folder structure, skipping.")
                continue

            landmarks = []
            for i in range(1, 69):
                x = row[f'x{i}']
                y = row[f'y{i}']
                landmarks.append((x, y))

            img_path = os.path.join(img_root, label_str, img_name)
            if os.path.exists(img_path):
                self.image_paths.append(img_path)
                self.expression_labels.append(label)
                self.landmarks.append(landmarks)
            else:
                print(f"⚠️ Image not found: {img_path}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        expression_label = torch.tensor(self.expression_labels[idx], dtype=torch.long)

        # Open the image *before transform* to get its original size
        image = Image.open(img_path)
        orig_width, orig_height = image.size  # Get original dimensions

        # Rescale landmarks from original image size to 224x224
        landmarks = torch.tensor(self.landmarks[idx], dtype=torch.float)
        landmarks[:, 0] = landmarks[:, 0] / orig_width * 224  # x coordinates
        landmarks[:, 1] = landmarks[:, 1] / orig_height * 224  # y coordinates
        landmarks = landmarks.view(-1)  # Flatten

        if self.transform:
            image = self.transform(image)

        return image, expression_label, landmarks

