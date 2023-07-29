from typing import List
import warnings
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F



class BasicBlock_res_3d(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, kernel_size = 3,stride=1, padding=1):
        super(BasicBlock_res_3d, self).__init__()
        self.conv1 = nn.Conv3d(
            in_planes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Dis_res_3d(nn.Module):
    def __init__(self, num_classes=3,channels=2):
        # ResNet-based 3D convolutional surrogate model
        super(Dis_res_3d, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv3d(channels, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.layer1 = self._make_layer(64, 4, stride=1)
        self.conv2 = nn.Conv3d(64, 64, kernel_size=3,
                               stride=1, bias=False)
        self.bn2 = nn.BatchNorm3d(64)
        self.layer2 = self._make_layer(128, 4, stride=1)
        self.bn3 = nn.BatchNorm3d(128)
        self.conv3 = nn.Conv3d(128, 128, kernel_size=3,
                               stride=2,padding=1, bias=False)
        self.layer3 = self._make_layer(256, 4, stride=1)
        self.conv4 = nn.Conv3d(256, 256, kernel_size=3,
                               stride=2,padding=1, bias=False)
        self.linear = nn.Linear(256, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock_res_3d(self.in_planes, planes, stride=stride))
            self.in_planes = planes 
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.layer2(out)
        out = self.conv3(out)
        out = self.layer3(out)
        out = self.conv4(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out