import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class CustomCNN(nn.Module):
    def __init__(self, num_classes=50, dropout_rate=0.0):
        super(CustomCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(64 * 16 * 16, 4096)
        self.drop1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(4096, 1024)
        self.drop2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(1024, num_classes)

        self._initialize_weights
        
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

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            
            elif isinstance(m, nn.Linear):
                if m == self.fc3:
                    init.xavier_normal_(m.weight)
                else:  
                    init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
