import torch
import torch.nn as nn
from torch.nn import Conv2d
from torch.nn import ReLU
from torch.nn import AdaptiveAvgPool2d
from torch.nn import Linear
import torch.nn.functional as F

####
# ResNet18-Based Architecture

class Mednet(nn.Module):

    def __init__(self):
        super().__init__()

        kernel_size = 3 # Assuming a kernel size of 3
        stride = 1
        padding = (kernel_size-1)//2

        self.conv11 = Conv2d(1, 64, kernel_size, stride, padding) # increase filter dimension
        self.conv12 = Conv2d(64, 64, kernel_size, stride, padding)
        self.block1 = self._block(64)

        self.conv21 = Conv2d(64, 128, kernel_size, stride, padding)
        self.conv22 = Conv2d(128, 128, kernel_size, stride, padding)
        self.block2 = self._block(128)

        self.conv31 = Conv2d(128, 256, kernel_size, stride, padding)
        self.conv32 = Conv2d(256, 256, kernel_size, stride, padding)
        self.block3 = self._block(256)

        self.conv41 = Conv2d(256, 512, kernel_size, stride, padding)
        self.conv42 = Conv2d(512, 512, kernel_size, stride, padding)
        self.block4 = self._block(512)

        self.conv5 = Conv2d(512, 1, kernel_size, stride, padding)

        out_size = 1
        self.pool = AdaptiveAvgPool2d(out_size) # Reduces (H x W), not C

        in_size = 1 # Input W
        out_size = 2 # Output W
        self.linear = Linear(in_size, out_size, bias = False)
        # Results in 2 weights
        # Results in (1 x 2) output

    def forward(self, img):

        img = F.relu(self.conv11(img))
        img = F.relu(self.conv12(img))
        img = F.relu(self.block1(img) + img)

        img = F.relu(self.conv21(img))
        img = F.relu(self.conv22(img))
        img = F.relu(self.block2(img) + img)

        img = F.relu(self.conv31(img))
        img = F.relu(self.conv32(img))
        img = F.relu(self.block3(img) + img)

        img = F.relu(self.conv41(img))
        img = F.relu(self.conv42(img))
        img = self.block4(img) + img

        img = self.conv5(img)

        pool = self.pool(img)
        pool = torch.squeeze(pool, dim = 1) # Dimensionality stuff
        pool = torch.squeeze(pool, dim = 1)
        logits = self.linear(pool)
        probs = F.softmax(logits, dim = 1)
        return probs, img

    def _block(self, channels, kernel_size = 3, stride = 1, padding = 0):
        return nn.Sequential(Conv2d(channels, channels, kernel_size, stride, (kernel_size-1)//2),
                             ReLU(),
                             Conv2d(channels, channels, kernel_size, stride, (kernel_size-1)//2))

    def __repr__(self):
        
        keys = list(self.state_dict().keys())
        for i in range(len(keys)):
            print(keys[i])

        return str(self.__class__)