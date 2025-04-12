import torch.nn as nn
import torch
import torch.optim
from torch import optim


class Object_detection(nn.Module):
    def __init__(self):
        super(Object_detection, self).__init__()

        # Convolutional layers
        self.Conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.Conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.Conv3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.Conv4 = nn.Conv2d(64, 128, 3, 1, 1)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.Conv5 = nn.Conv2d(128, 256, 3, 1, 1)
        self.pool5 = nn.MaxPool2d(2, 2)
        self.Conv6 = nn.Conv2d(256, 512, 3, 1, 1)

        # Fully connected layers
        self.fc1 = nn.Linear(512 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 256)

        # Output layers
        self.fc_expression = nn.Linear(256, 12)  # Expression classification (11 classes)
        self.fc_landmarks = nn.Linear(256, 136)  # Landmark prediction (6 pairs for eyes, nose, and ears)

    def forward(self, image):
        # Convolutional layers
        image = self.pool1(torch.relu(self.Conv1(image)))
        image = self.pool2(torch.relu(self.Conv2(image)))
        image = self.pool3(torch.relu(self.Conv3(image)))
        image = self.pool4(torch.relu(self.Conv4(image)))
        image = self.pool5(torch.relu(self.Conv5(image)))
        image = torch.relu(self.Conv6(image))

        image = image.view(image.size(0), -1)

        image = torch.relu(self.fc1(image))
        image = torch.relu(self.fc2(image))

        expression_output = self.fc_expression(image)

        # 🔧 Use sigmoid + scale to 224x224 for landmarks
        landmarks_output = torch.sigmoid(self.fc_landmarks(image)) * 224

        return expression_output, landmarks_output

    def compute_loss(self, expression_predictions=None, expression_truth=None,
                     landMark_predictions=None, landMark_truth=None,
                     expression_weights=None, alpha=1.0, beta=1.0):
        # Loss for facial expression prediction
        if expression_weights is not None:
            expression_cross_loss = nn.CrossEntropyLoss(weight=expression_weights)
        else:
            expression_cross_loss = nn.CrossEntropyLoss()
        expression_loss = expression_cross_loss(expression_predictions, expression_truth)

        # Loss for landmark prediction (using Smooth L1 Loss)
        landmark_loss_fn = nn.SmoothL1Loss()  # Smooth L1 Loss for landmark regression
        landmark_loss = landmark_loss_fn(landMark_predictions, landMark_truth)

        # Total loss is a weighted sum of both losses
        total_loss = alpha * expression_loss + beta * landmark_loss
        return total_loss


