import torch.nn as nn
import torch
class Object_detection(nn.Module):
    def __init__(self):
        super(Object_detection,self).__init__()
        self.Conv1 = nn.Conv2d(3,16,3,1,1)
        self.pool1 = nn.MaxPool2d(2,2)

        self.Conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.Conv3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.Conv4 = nn.Conv2d(64, 128, 3, 1, 1)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.Conv5 = nn.Conv2d(128, 256, 3, 1, 1)
        self.pool5 = nn.MaxPool2d(2, 2)

        self.Conv6 = nn.Conv2d(256, 512, 3, 1, 1)

        self.fc1 = nn.Linear(512 * 13 * 13, 1024)  # Flattening output from Conv6
        self.fc2 = nn.Linear(1024, 256)  # Intermediate fully connected layer
        self.fc3 = nn.Linear(256, 5)

    def fowrward(self, image):
        image = self.pool1(torch.relu(self.conv1(image)))
        image = self.pool2(torch.relu(self.conv2(image)))
        image = self.pool3(torch.relu(self.conv3(image)))
        image = self.pool4(torch.relu(self.conv4(image)))
        image = self.pool5(torch.relu(self.conv5(image)))
        image = self.torch.relu(self.conv6(image))

        image = image.view(-1, 512 * 13 * 13)  # Flattening

        # Pass through fully connected layers
        image = torch.relu(self.fc1(image))  # FC1
        image = torch.relu(self.fc2(image))  # FC2
        image = self.fc3(image)

        return image
