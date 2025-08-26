import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, 
            in_channels, 
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class PSPModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # Fixed pooling: assumes input is divisible by pool kernel/stride
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.AvgPool2d(kernel_size=4, stride=4),
                nn.Conv2d(in_channels, 1, kernel_size=1),
            ),
            nn.Sequential(
                nn.AvgPool2d(kernel_size=8, stride=8),
                nn.Conv2d(in_channels, 1, kernel_size=1),
            ),
            nn.Sequential(
                nn.AvgPool2d(kernel_size=16, stride=16),
                nn.Conv2d(in_channels, 1, kernel_size=1),
            ),
            nn.Sequential(
                nn.AvgPool2d(kernel_size=32, stride=32),
                nn.Conv2d(in_channels, 1, kernel_size=1),
            ),
        ])
        self.conv = nn.Conv2d(in_channels + 4, 3, kernel_size=3, padding=1)

    def forward(self, x):
        features = [x]
        for branch in self.branches:
            pooled = branch(x)
            upsampled = F.interpolate(pooled, size=x.shape[2:], mode='bilinear', align_corners=False)
            features.append(upsampled)
        return self.conv(torch.cat(features, dim=1))

class FAODNet(nn.Module):
    def __init__(self):
        super().__init__()
        # DS-Conv1: 3->3 channels
        self.ds_conv1 = DepthwiseSeparableConv(3, 3, kernel_size=3)
        self.relu1 = nn.ReLU(inplace=True)
        
        # DS-Conv2: 3->6 channels
        self.ds_conv2 = DepthwiseSeparableConv(3, 6, kernel_size=3)
        self.relu2 = nn.ReLU(inplace=True)
        
        # DS-Conv3: 9->9 channels (concat1: 3+6=9)
        self.ds_conv3 = DepthwiseSeparableConv(9, 9, kernel_size=5, padding=2)
        self.relu3 = nn.ReLU(inplace=True)
        
        # DS-Conv4: 15->6 channels (concat2: 6+9=15)
        self.ds_conv4 = DepthwiseSeparableConv(15, 6, kernel_size=7, padding=3)
        self.relu4 = nn.ReLU(inplace=True)
        
        # DS-Conv5: 24->3 channels (concat3: 3+6+9+6=24)
        self.ds_conv5 = DepthwiseSeparableConv(24, 3, kernel_size=3)
        self.relu5 = nn.ReLU(inplace=True)
        
        # Pyramid Pooling Module (input=3 channels, output=3 channels)
        self.psp = PSPModule(3)
        
        # Final convolution for K(x) (3->3 channels)
        self.final_conv = nn.Conv2d(3, 3, kernel_size=3, padding=1)

        self.b = 1

    def forward(self, x):
        # Feature extraction
        f1 = self.relu1(self.ds_conv1(x))  # 3 channels
        f2 = self.relu2(self.ds_conv2(f1))  # 6 channels
        
        concat1 = torch.cat([f1, f2], dim=1)  # 9 channels
        f3 = self.relu3(self.ds_conv3(concat1))  # 9 channels
        
        concat2 = torch.cat([f2, f3], dim=1)  # 15 channels
        f4 = self.relu4(self.ds_conv4(concat2))  # 6 channels
        
        concat3 = torch.cat([f1, f2, f3, f4], dim=1) # 24 channels
        f5 = self.relu5(self.ds_conv5(concat3))  # 3 channels
        
        # Pyramid Pooling Module
        psp_out = self.psp(f5)  # 3 channels
        
        # Generate K(x)
        k = self.final_conv(psp_out)  # 3 channels
        
        # Dehazing formula: J(x) = K(x)·I(x) - K(x) + 1
        j = k * x - k + self.b
        return torch.clamp(j, 0, 1)  # Clamp to valid image range
