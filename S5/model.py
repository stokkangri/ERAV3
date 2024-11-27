import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(16)
        self.dropout1 = nn.Dropout(0.10)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(32)
        self.dropout2 = nn.Dropout(0.10)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(32)
        self.dropout3 = nn.Dropout(0.10)
        self.conv4 = nn.Conv2d(32, 10, kernel_size=3)
        self.bn4 = nn.BatchNorm2d(10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x))) 
        x = self.dropout1(x)
        x = F.relu(self.bn2(F.max_pool2d(self.conv2(x), 2)))
        x = self.dropout2(x)
        x = F.relu(self.bn3(F.max_pool2d(self.conv3(x), 2)))
        x = self.dropout3(x)
        x = F.relu(self.bn4(F.max_pool2d(self.conv4(x), 2)))
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)
    
