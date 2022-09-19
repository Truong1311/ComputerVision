import torch
import torch.nn as nn

## model : LetNet

class LeNet5(nn.Module):
    def __init__(self, num_classes):
        super(LeNet5, self).__init__()
        self.num_classes = num_classes
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        
        self.fc = nn.Linear(400, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, self.num_classes)
    def forward(self, x):
        y = self.layer1(x)
        y = self.layer2(y)
        y = y.view(y.shape[0], -1) ## flatten con2v
        y = self.fc(y)
        y = self.relu(y)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        return y