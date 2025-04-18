import torch.nn as nn
import torch
import torch.optim
from torch import optim


class Object_detection(nn.Module):
    def __init__(self):
        super(Object_detection, self).__init__()

        # Convolutional layers with BatchNorm
        self.Conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.Conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.Conv3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.Conv4 = nn.Conv2d(64, 128, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.Conv5 = nn.Conv2d(128, 256, 3, 1, 1)
        self.bn5 = nn.BatchNorm2d(256)
        self.pool5 = nn.MaxPool2d(2, 2)
        self.Conv6 = nn.Conv2d(256, 512, 3, 1, 1)
        self.bn6 = nn.BatchNorm2d(512)

        # Fully connected layers
        self.fc1 = nn.Linear(512 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 256)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

        # Output layers
        self.fc_expression = nn.Linear(256, 7)  # Emotion classification (12 classes)
        self.fc_landmarks = nn.Linear(256, 478 * 2)  # Landmark prediction (468 landmarks with x, y)
        self.fc_bbox = nn.Linear(256, 4)  # Bounding box prediction (x_min, y_min, x_max, y_max)

    def forward(self, image):
        # Convolution layers with BatchNorm
        image = self.bn1(torch.relu(self.Conv1(image)))
        image = self.pool1(image)
        image = self.bn2(torch.relu(self.Conv2(image)))
        image = self.pool2(image)
        image = self.bn3(torch.relu(self.Conv3(image)))
        image = self.pool3(image)
        image = self.bn4(torch.relu(self.Conv4(image)))
        image = self.pool4(image)
        image = self.bn5(torch.relu(self.Conv5(image)))
        image = self.pool5(image)
        image = self.bn6(torch.relu(self.Conv6(image)))

        image = image.view(image.size(0), -1)

        image = torch.relu(self.fc1(image))
        image = self.dropout(image)
        image = torch.relu(self.fc2(image))

        # Emotion prediction
        expression_output = self.fc_expression(image)

        # Landmark prediction
        landmarks_output = torch.sigmoid(self.fc_landmarks(image)) * 224  # Rescale to image size

        # Bounding box prediction
        bbox_output = torch.sigmoid(self.fc_bbox(image)) * 224

        return expression_output, landmarks_output, bbox_output

    def compute_loss(self, expression_predictions, expression_truth,
                     landMark_predictions, landMark_truth,
                     bbox_predictions, bbox_truth,
                     expression_weights=None, alpha=1.0, beta=1.0, gamma=1.0):
        # Loss for facial expression prediction
        if expression_weights is not None:
            expression_cross_loss = nn.CrossEntropyLoss(weight=expression_weights)
        else:
            expression_cross_loss = nn.CrossEntropyLoss()
        expression_loss = expression_cross_loss(expression_predictions, expression_truth)

        # Loss for landmark prediction
        landmark_loss_fn = nn.SmoothL1Loss()
        landmark_loss = landmark_loss_fn(landMark_predictions, landMark_truth)

        # Loss for bounding box prediction
        bbox_loss_fn = nn.SmoothL1Loss()
        bbox_loss = bbox_loss_fn(bbox_predictions, bbox_truth)

        # Total loss
        total_loss = alpha * expression_loss + beta * landmark_loss + gamma * bbox_loss
        return total_loss


