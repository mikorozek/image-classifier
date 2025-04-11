import torch.nn as nn
import torch.nn.functional as F

class CustomCNN(nn.Module):
    def __init__(self, num_classes=50):
        super(CustomCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 512, kernel_size=5, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(512, 512, kernel_size=5, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(512, 512, kernel_size=5, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.conv4 = nn.Conv2d(512, 512, kernel_size=5, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(512 * 16 * 16, 4096)
        self.drop1 = nn.Dropout(0.15)
        self.fc2 = nn.Linear(4096, 1024)
        self.drop2 = nn.Dropout(0.15)
        self.fc3 = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        
        x = F.leaky_relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        x = self.fc3(x)
        
        return x
