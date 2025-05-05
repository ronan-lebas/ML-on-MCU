import torch
import torch.nn as nn

class AudioCNN(nn.Module):
    def __init__(self, num_classes, in_channels=1):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=0)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(32, 32, kernel_size=7, stride=2, padding=0)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv1d(32, 32, kernel_size=7, stride=2, padding=0)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu3 = nn.ReLU()
        self.fc1 = nn.Linear(7936, 128)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)


    def forward(self, x):
        x = self.relu1(self.pool1(self.conv1(x)))
        x = self.relu2(self.pool2(self.conv2(x)))
        x = self.relu3(self.pool3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return x