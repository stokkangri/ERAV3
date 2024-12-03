import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
      super(SimpleCNN, self).__init__()
      if False:
        # convolution block 1
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(1, 8, 3),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(0.10),

            nn.Conv2d(8, 16, 3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.10)
        )

        # convolution block 2
        self.conv_block_2 = nn.Sequential(

            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.10)

        )

        # transition block 1
        self.transition_1 = nn.Sequential(
            nn.Conv2d(in_channels= 32,out_channels= 16, kernel_size =1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # convolution block 3
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(16, 16, 3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.10),

            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.10)
        )
        # fully connected
        self.fc1 = nn.Linear(32,10)

      if False:
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.norm1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(0.10)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.norm2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout(0.10)
        self.conv3 = nn.Conv2d(32, 32, 3)
        self.conv4 = nn.Conv2d(32, 10, 3)
    
        
      if True:
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(16)
        self.dropout1 = nn.Dropout(0.01)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(32)
        self.dropout2 = nn.Dropout(0.01)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(32)
        self.dropout3 = nn.Dropout(0.00)
        self.conv4 = nn.Conv2d(32, 10, kernel_size=3)
        self.bn4 = nn.BatchNorm2d(10)
        self.conv1x1_reduce = nn.Conv2d(32, 16, kernel_size=1)
        #self.conv1x1_expand = nn.Conv2d(16, 32, kernel_size=3)
        #self.gap = nn.AdaptiveAvgPool2d((1, 1)) 
        self.fc = nn.Linear(16*5*5, 10)
        

    def forward(self, x):

      if False:
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.transition_1(x)
        x = self.conv_block_3(x)

        # GAP
        x = F.adaptive_avg_pool2d(x, (1, 1))  # Input - 7x7x32 -> Output - 1x1x32

        # reshape using flatten
        x = torch.flatten(x, 1)

        # Fully connected layer
        x = self.fc1(x) # 1x1x32 -> 1x1x10

        x = x.view(-1, 10)

        return F.log_softmax(x)
    
      if False:
        # After 1st conv -> n_in = 28, p = 0, s = 1, k = 3, n_out = 26, j_in = 1, j_out = 1, r_in = 1, r_out = 3
        # After 1st Max Pool -> n_in = 26, p = 0, s = 2, k = 2, n_out = 13, j_in = 1, j_out = 2, r_in = 3, r_out = 4
        x = self.drop1(self.pool1(self.norm1(F.relu(self.conv1(x)))))
        # After 2nd conv -> n_in = 13, p = 0, s = 1, k = 3, n_out = 11, j_in = 2, j_out = 2, r_in = 4, r_out = 8
        # After 2nd Max Pool -> n_in = 11, p = 0, s = 2, k = 2, n_out = 5, j_in = 2, j_out = 4, r_in = 8, r_out = 10
        x = self.drop2(self.pool2(self.norm2(F.relu(self.conv2(x)))))
        # After 3rd conv -> n_in = 5, p = 0, s = 1, k = 3, n_out = 3, j_in = 4, j_out = 4, r_in = 10, r_out = 18
        # After 4th conv -> n_in = 13, p = 0, s = 1, k = 3, n_out = 11, j_in = 2, j_out = 2, r_in = 4, r_out = 26
        x = F.relu(self.conv4(F.relu(self.conv3(x))))
        x = x.view(-1,10)
        return F.log_softmax(x, dim=1)
        
        
      if True:
        x = F.relu(self.bn1(self.conv1(x)))  # 26x26x16
        x = self.dropout1(x)
        x = F.relu(self.bn2(F.max_pool2d(self.conv2(x), 2))) # 12x12x32
        x = self.dropout2(x)
        x = F.relu(self.bn3(F.max_pool2d(self.conv3(x), 2))) # 5x5x32
        x = self.dropout3(x)
        #x = F.relu(self.bn4(F.max_pool2d(self.conv4(x), 2))) # 3x3x10
       
        x = self.conv1x1_reduce(x) # 5x5x16
        #x = self.conv1x1_expand(x) # 5x5x32
        #x= self.gap(x)
        
        x = x.view(-1, 16*5*5)
        x = self.fc(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)
        
