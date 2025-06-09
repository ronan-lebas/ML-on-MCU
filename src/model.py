import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioCNN(nn.Module):
    def __init__(self, num_classes, in_channels=1):
        super(AudioCNN, self).__init__()
        
        # Conv block 1
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)  # [B, 32, F, T]
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2)  # [B, 32, F/2, T/2]
        
        # Conv block 2
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2)  # [B, 64, F/4, T/4]
        
        # Conv block 3
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.AdaptiveAvgPool2d((1, 1))  # [B, 128, 1, 1]

        # Fully connected
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        
        # Block 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        
        # Block 3
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        
        # Classifier
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# class AudioCNN(nn.Module):
#     def __init__(self, num_classes, in_channels=1):
#         super(AudioCNN, self).__init__()

#         # Conv block 1
#         self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.pool1 = nn.MaxPool2d(kernel_size=2)

#         # Conv block 2
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.pool2 = nn.AdaptiveAvgPool2d((1, 1))

#         # Fully connected
#         self.fc1 = nn.Linear(64, num_classes)

#     def forward(self, x):
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = self.pool1(x)

#         x = F.relu(self.bn2(self.conv2(x)))
#         x = self.pool2(x)

#         x = x.view(x.size(0), -1)
#         x = self.fc1(x)
#         return x


# class AudioCNN(nn.Module):
#     def __init__(self, num_classes, in_channels=1):
#         super(AudioCNN, self).__init__()

#         # Conv block 1
#         self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(32)
#         self.pool1 = nn.MaxPool2d(kernel_size=2)

#         # Conv block 2
#         self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.bn3 = nn.BatchNorm2d(64)
#         self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
#         self.bn4 = nn.BatchNorm2d(64)
#         self.pool2 = nn.MaxPool2d(kernel_size=2)

#         # Conv block 3
#         self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.bn5 = nn.BatchNorm2d(128)
#         self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
#         self.bn6 = nn.BatchNorm2d(128)
#         self.pool3 = nn.MaxPool2d(kernel_size=2)

#         # Conv block 4
#         self.conv7 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
#         self.bn7 = nn.BatchNorm2d(256)
#         self.conv8 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         self.bn8 = nn.BatchNorm2d(256)
#         self.pool4 = nn.AdaptiveAvgPool2d((1, 1))

#         # Fully connected
#         self.fc1 = nn.Linear(256, 128)
#         self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(64, num_classes)

#     def forward(self, x):
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = self.pool1(x)

#         x = F.relu(self.bn3(self.conv3(x)))
#         x = F.relu(self.bn4(self.conv4(x)))
#         x = self.pool2(x)

#         x = F.relu(self.bn5(self.conv5(x)))
#         x = F.relu(self.bn6(self.conv6(x)))
#         x = self.pool3(x)

#         x = F.relu(self.bn7(self.conv7(x)))
#         x = F.relu(self.bn8(self.conv8(x)))
#         x = self.pool4(x)

#         x = x.view(x.size(0), -1)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
