import torch
import torch.nn as nn

# ---------------------------
# AOD-Net Model Definition
# ---------------------------
class AODNet(nn.Module):
    def __init__(self):
        super(AODNet, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(3, 3, kernel_size=1)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(6, 3, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(6, 3, kernel_size=7, padding=3)
        self.conv5 = nn.Conv2d(12, 3, kernel_size=3, padding=1)
        self.b = 1

    def forward(self, x):
        conv1 = self.relu(self.conv1(x))
        conv2 = self.relu(self.conv2(conv1))
        concat1 = torch.cat((conv1, conv2), 1)
        conv3 = self.relu(self.conv3(concat1))
        concat2 = torch.cat((conv2, conv3), 1)
        conv4 = self.relu(self.conv4(concat2))
        concat3 = torch.cat((conv1, conv2, conv3, conv4), 1)
        k = self.relu(self.conv5(concat3))
        clean = k * x - k + self.b
        return torch.clamp(clean, 0, 1)
