from typing import List
import warnings
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from d2l import torch as d2l


class G_block_1d(nn.Module):
    def __init__(self, out_channels, in_channels=3,**kwargs):
        super(G_block_1d, self).__init__(**kwargs)
        self.conv1d_trans = nn.Linear(in_channels,out_channels)
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, X):
        return self.activation(self.batch_norm(self.conv1d_trans(X)))

class Generator_1d(nn.Module):
    def __init__(self,channel_list:List,out_channels):
        super(Generator_1d,self).__init__()
        layers = []
        for i in range(len(channel_list)-1):
            block = G_block_1d(in_channels=channel_list[i],out_channels=channel_list[i+1])
            layers.append(block)
        self.net = nn.Sequential(*layers)
        self.final = nn.Linear(channel_list[len(channel_list)-1], out_channels)
        self.tanh = nn.Tanh()
    
    def forward(self,X):
        out = self.net(X)
        out = self.final(out)
        out = self.tanh(out)
        return out


class G_block(nn.Module):
    def __init__(self,  in_channels, out_channels, kernel_size=4, strides=2,
                 padding=1, **kwargs):
        super(G_block, self).__init__(**kwargs)
        self.conv2d_trans = nn.ConvTranspose2d(in_channels, out_channels,
                                               kernel_size, strides, padding,
                                               bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, X):
        return self.activation(self.batch_norm(self.conv2d_trans(X)))

class Generator(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Generator, self).__init__()
        self.g1 = G_block(in_channels=in_channels,out_channels=128)
        self.g2 = G_block(in_channels=128,out_channels=128,kernel_size=3,strides=1,padding=1)
        self.g5 = G_block(in_channels=128,out_channels=128)
        self.g3 = G_block(in_channels=128,out_channels=64,kernel_size=3,strides=1,padding=1)
        self.g4 = G_block(in_channels=64,out_channels=out_channels,kernel_size=3,strides=1,padding=1)
        self.last = nn.Tanh()

    def forward(self, X):
        out = self.g1(X)
        #print(out.size())
        out = self.g2(out)
        out = self.g5(out)
        out = self.g3(out)
        out = self.g4(out)
        out = self.last(out)
        return out


class D_block_1d(nn.Module):
    def __init__(self, out_channels, in_channels=3, alpha=0.2, **kwargs):
        super(D_block_1d, self).__init__(**kwargs)
        self.conv1d = nn.Linear(in_channels,out_channels)
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.LeakyReLU(alpha, inplace=True)

    def forward(self, X):
        return self.activation(self.batch_norm(self.conv1d(X)))

class Discriminator_1d(nn.Module):
    def __init__(self,channel_list:List,out_channels):
        super(Discriminator_1d,self).__init__()
        layers = []
        for i in range(len(channel_list)-1):
            block = D_block_1d(in_channels=channel_list[i],out_channels=channel_list[i+1])
            layers.append(block)
        self.net = nn.Sequential(*layers)
        self.final = nn.Linear(channel_list[len(channel_list)-1], out_channels)
    
    def forward(self,X):
        out = self.net(X)
        out = self.final(out)
        return out

class G_block_3d(nn.Module):
    def __init__(self,  in_channels, out_channels, kernel_size=4, strides=2,
                 padding=1, **kwargs):
        super(G_block_3d, self).__init__(**kwargs)
        self.conv3d_trans = nn.ConvTranspose3d(in_channels, out_channels,
                                               kernel_size, strides, padding,
                                               bias=False)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, X):
        return self.activation(self.batch_norm(self.conv3d_trans(X)))


class Generator_3d(nn.Module):
    def __init__(self,in_channels,out_channels=2):
        super(Generator_3d, self).__init__()
        self.g1 = G_block_3d(in_channels=in_channels,out_channels=128)
        self.g2 = G_block_3d(in_channels=128,out_channels=128,kernel_size=3,strides=1,padding=1)
        self.g3 = G_block_3d(in_channels=128,out_channels=128,kernel_size=3,strides=1,padding=1)
        self.g4 = G_block_3d(in_channels=128,out_channels=128)
        self.g5 = G_block_3d(in_channels=128,out_channels=128,kernel_size=3,strides=1,padding=1)
        self.g6 = G_block_3d(in_channels=128,out_channels=128,kernel_size=3,strides=1,padding=1)
        self.g7 = G_block_3d(in_channels=128,out_channels=128)
        self.g8 = G_block_3d(in_channels=128,out_channels=128,kernel_size=3,strides=1,padding=1)
        self.g9 = G_block_3d(in_channels=128,out_channels=128,kernel_size=3,strides=1,padding=1)
        self.g10 = G_block_3d(in_channels=128,out_channels=128,kernel_size=3,strides=1,padding=0)
        self.g11 = G_block_3d(in_channels=128,out_channels=64,kernel_size=3,strides=1,padding=1)
        self.g12 = G_block_3d(in_channels=64,out_channels=out_channels,kernel_size=3,strides=1,padding=1)

    def forward(self, X):
        out = self.g1(X)
        out = self.g2(out)
        out = self.g3(out)
        out = self.g4(out)
        out = self.g5(out)
        out = self.g6(out)
        out = self.g7(out)
        out = self.g8(out)
        out = self.g9(out)
        out = self.g10(out)
        out = self.g11(out)
        out = self.g12(out)
        #print(out)
        out = F.softmax(out,dim=1)
        #print(out)
        return out