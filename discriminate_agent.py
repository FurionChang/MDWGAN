from typing import List
import warnings
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock_res(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, kernel_size = 3,stride=1, padding=1):
        super(BasicBlock_res, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BasicBlock_vgg(nn.Module):
    def __init__(self,  in_channels, out_channels, kernel_size = 3, stride=1, padding=1,alpha=0.2, **kwargs):
        super(BasicBlock_vgg, self).__init__(**kwargs)
        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(alpha, inplace=True)

    def forward(self, X):
        return self.activation(self.batch_norm(self.conv2d(X)))


class discriminator_agent(nn.Module):
    def __init__(self,type):
        super(discriminator_agent, self).__init__()
        self.features = self._make_layers(type)
        #last_layer_feature = config[len(config)-1]
        self.linear_1 = nn.Linear(128, 64)
        self.linear_2 = nn.Linear(64,32)
        self.linear_3 = nn.Linear(32,16)
        self.linear_4 = nn.Linear(16,1)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = F.relu(self.linear_1(out))
        out = F.relu(self.linear_2(out))
        out = F.relu(self.linear_3(out))
        out = self.linear_4(out)
        return out

    def _make_layers(self,type):
        layers = []
        if type == 'vgg':
            layers.append(BasicBlock_vgg(in_channels=4,out_channels=64))
            layers.append(BasicBlock_vgg(in_channels=64,out_channels=128))
            layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
            layers.append(BasicBlock_vgg(in_channels=128,out_channels=128))
            layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
        if type == 'res':
            layers.append(BasicBlock_res(in_planes=4,planes=64))
            layers.append(BasicBlock_res(in_planes=64,planes=128))
            layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
            layers.append(BasicBlock_res(in_planes=128,planes=128))
            layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
        
        return nn.Sequential(*layers)

class Dis_res(nn.Module):
    def __init__(self, num_classes=3):
        super(Dis_res, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(10, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 4, stride=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3,
                               stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.layer2 = self._make_layer(128, 4, stride=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.layer3 = self._make_layer(256, 4, stride=1)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.layer4 = self._make_layer(512, 4, stride=1)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock_res(self.in_planes, planes, stride=stride))
            self.in_planes = planes 
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.layer2(out)
        out = self.pool1(out)
        out = self.layer3(out)
        out = self.pool2(out)
        out = self.layer4(out)
        out = self.pool3(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


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
