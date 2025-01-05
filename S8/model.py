import torch.nn.functional as F
import torch.nn as nn

dropout_value = 0.1

def calc_rf (rf_in, kernel_size, jump_in):
    # jump_out = jump_in * stride
    # jump_in_n = jump_out_n-1
    #nin    rin jin s   p   nout    rout    jout
    # 32	1	1	1	1	32	    3	    1
    # 32	3	1	2	0	15	    5	    2
    # 15	5	2	1	0	13	    9	    2
    # 13	9	2	2	1	7	    13	    4
    # 7	    13	4	1	1	7	    21	    4
    # 7	    21	4	2	0	3	    29	    8
    # 3	    29	8	1	0	1	    45	    8
    return rf_in + (kernel_size - 1) * jump_in

def calc_out_size(in_size, padding, stride, kernel_size):
    # nin: number of input features
    # nout : number of output features
    # k : conv kernel size
    # p : padding size
    # s : conv stride size
    return 1 + (in_size + 2 * padding - kernel_size) / stride

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Input Block (32x32 -> 32x32)
        self.input = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        # Bottleneck Block 0 (32x32 -> 16x16)
        self.bottleneck0_main = nn.Sequential(
            # Expand
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # Depthwise with stride 2 to reduce spatial dimensions
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, groups=64, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64), #
            # Reduce back
            # Expand further
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        )
        
        self.bottleneck0_shortcut = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(64)
        )
        
        # Bottleneck Block 1 (16x16 -> 8x8)
        self.bottleneck1_main = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64), #
            # Expand
            nn.Conv2d(in_channels=64, out_channels=96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            # Depthwise with stride 2
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=2, padding=1, groups=96, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            # Expand further
            nn.Conv2d(in_channels=96, out_channels=128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # Reduce back
            nn.Conv2d(in_channels=128, out_channels=96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        )
        
        self.bottleneck1_shortcut = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=96, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(96)
        )
        
        # Bottleneck Block 2 (8x8 -> 4x4)
        self.bottleneck2_main = nn.Sequential(
            # Expand
            nn.Conv2d(in_channels=96, out_channels=128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # Depthwise with stride 2
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # Point-wise
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        )
        
        self.bottleneck2_shortcut = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=128, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(128)
        )
        
        # Global Average Pooling
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1))
        )
        
        # Final Classifier
        self.linear = nn.Sequential(
            nn.Linear(128, 10, bias=False)
        )

    def forward(self, x):
        x = self.input(x)  # 32x32
        
        # Bottleneck Block 0
        identity = self.bottleneck0_shortcut(x)
        x = self.bottleneck0_main(x)
        x = x + identity  # 16x16
        
        # Bottleneck Block 1
        identity = self.bottleneck1_shortcut(x)
        x = self.bottleneck1_main(x)
        x = x + identity  # 8x8
        
        # Bottleneck Block 2
        identity = self.bottleneck2_shortcut(x)
        x = self.bottleneck2_main(x)
        x = x + identity  # 4x4
        
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return F.log_softmax(x, dim=1)