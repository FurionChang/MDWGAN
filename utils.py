import torch
from torch import nn
import numpy as np
from torch.utils import data
from io import StringIO
import random
from d2l import torch as d2l
from matplotlib import pyplot as plt
import os
import glob
import torch.nn.functional as F
import re
import heapq
import pandas as pd

def get_filename_structures(path,sort_num=3):
    res = glob.glob(path+'*.txt')
    res.sort(key=lambda x:(int(x.split('_')[sort_num].split('.txt')[0]),float(x.split('_')[sort_num-1])))
    return res

def get_filename_moduli(path,sort_num=2):
    res = glob.glob(path+'*.txt')
    res.sort(key=lambda x:(int(x.split('_')[sort_num]),float(x.split('_')[sort_num+1])))
    return res

def num_to_cor(size_ele, elems):
    # element number to coordinates
    x_cor = elems % size_ele[0]
    x_cor = torch.where(x_cor != 0, x_cor, size_ele[0])
    y_cor = torch.floor(elems/size_ele[1]) + 1
    y_cor = y_cor.long()
    y_cor = torch.where(elems % size_ele[1] != 0, y_cor, y_cor - 1)
    y_cor = y_cor % size_ele[1] 
    y_cor = torch.where(y_cor !=0, y_cor, size_ele[1])
    z_cor = torch.floor(elems/(size_ele[0]*size_ele[1]))+1
    z_cor = torch.where(elems%(size_ele[0]*size_ele[1])!=0, z_cor, z_cor-1)
    z_cor = z_cor.long()
    return x_cor, y_cor, z_cor

def transfer_1d_to_3d(X,size_ele):
    res = torch.zeros(X.size()[0],size_ele[0],size_ele[1],size_ele[2])
    x_cor, y_cor, z_cor = num_to_cor(torch.tensor(size_ele).float(), X.long().cpu())
    x_cor = x_cor.long()
    y_cor = y_cor.long()
    z_cor = z_cor.long()
    for batch in range(X.size()[0]):
        for i  in range(X.size()[1]):
            res[batch,x_cor[batch,i]-1,y_cor[batch,i]-1,z_cor[batch,i]-1] = 1
    return res

def transfer_1d_to_3d_new(X,size_ele):
    res = torch.zeros(size_ele[0],size_ele[1],size_ele[2])
    x_cor, y_cor, z_cor = num_to_cor(torch.tensor(size_ele).float(), X.long().cpu())
    x_cor = x_cor.long()
    y_cor = y_cor.long()
    z_cor = z_cor.long()
    for i  in range(X.size()[0]):
        res[x_cor[i]-1,y_cor[i]-1,z_cor[i]-1] = 1
    return res


def load_data_from_dir(size_ele,fstruture_dir0='./structures_10/',fmoduli_dir0 = './moduli_10/'):
    
    # weight = 
    # other properties
    structures_path = get_filename_structures(fstruture_dir0)
    moduli_path = get_filename_moduli(fmoduli_dir0)
    elems = torch.zeros((len(structures_path),size_ele[0],size_ele[1],size_ele[2])).cuda()
    moduli = np.empty(shape = (0, 3))

    for i in range (len(structures_path)):
        fstruture_dir = structures_path[i]
        fmoduli_dir = moduli_path[i]
        #load elements from fe
        Xread = np.loadtxt (fstruture_dir, dtype = float, delimiter = 'None')
        #load moduli from fe results
        #as long string
        Yread = np.genfromtxt (fmoduli_dir, dtype = str, delimiter = 'None')
        #split long string to string array
        num_1 = float(re.findall(r"\d+.\d+",Yread[0])[0])
        num_2 = float(re.findall(r"\d+.\d+",Yread[2])[1])
        num_3 = float(re.findall(r"\d+.\d+",Yread[4])[2])
        #string to float
        Yread = [num_1,num_2,num_3]
        #average x, y, z
        Yread_p = np.array(Yread)
        #stack data
        Xread = torch.as_tensor(Xread)
        elems[i,:,:,:] = transfer_1d_to_3d_new(Xread,size_ele)
        moduli = np.vstack((moduli,Yread_p))

    #numpy array to torch tensor
    #as a must, if not will lead to 'int' object not callable error
    moduli = torch.from_numpy(moduli)
    # data type
    elems = elems.long()
    moduli = moduli.float()
    return elems,moduli

def load_data_from_dir_3d(size_ele,fstruture_dir0='./structures_10/',fmoduli_dir0 = './moduli_10/'):
    
    # weight = 
    # other properties
    structures_path = get_filename_structures(fstruture_dir0)
    moduli_path = get_filename_moduli(fmoduli_dir0)
    elems = torch.zeros((len(structures_path),2,size_ele[0],size_ele[1],size_ele[2])).cuda()
    moduli = np.empty(shape = (0, 3))

    for i in range (len(structures_path)):
        fstruture_dir = structures_path[i]
        fmoduli_dir = moduli_path[i]
        #load elements from fe
        Xread = np.loadtxt (fstruture_dir, dtype = float, delimiter = 'None')
        #load moduli from fe results
        #as long string
        Yread = np.genfromtxt (fmoduli_dir, dtype = str, delimiter = '\t')
        #split long string to string array
        num_1 = float(re.findall(r"\d+.\d+",Yread[0])[0])
        num_2 = float(re.findall(r"\d+.\d+",Yread[2])[1])
        num_3 = float(re.findall(r"\d+.\d+",Yread[4])[2])
        #string to float
        Yread = [num_1,num_2,num_3]
        #average x, y, z
        Yread_p = np.array(Yread)
        #stack data
        Xread = torch.as_tensor(Xread)
        elems[i,:,:,:,:] = transfer_3d_to_channels_2(transfer_1d_to_3d_new(Xread,size_ele))
        moduli = np.vstack((moduli,Yread_p))

    #numpy array to torch tensor
    #as a must, if not will lead to 'int' object not callable error
    moduli = torch.from_numpy(moduli)
    # data type
    elems = elems.long()
    moduli = moduli.float()
    return elems,moduli



def load_array(data_arrays, batch_size):  #@save
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size,shuffle=True)

def transfer_gan_out(gan_out,batch_size=8,sample_size=32):
    res = torch.zeros((batch_size,sample_size), device=d2l.try_gpu())
    gan_out_np = np.array(gan_out.detach().cpu())
    median = np.median(gan_out_np,axis=1)
    for i in range(gan_out.size()[0]):
        temp = ((gan_out[i]>=median[i]).nonzero(as_tuple=True)[0])
        if temp.size()[0]>=sample_size:
            res[i,:]=temp[0:sample_size]
        else:
            res[i,0:temp.size()[0]]=temp
    res += 1
    return res


def transfer_3d_gan_out(x_out):
    x_out = x_out.transpose(1,2).transpose(2,3).transpose(3,4)
    sampler = torch.distributions.Categorical(x_out)
    idx = sampler.sample()
    sum_matrix = torch.zeros((x_out.size(0),)).cuda()
    for i in range(x_out.size(0)):
        sum_matrix[i] = torch.sum(idx.data[i,:,:,:])
    return idx.data,sum_matrix

def transfer_3d_gan_out_limit(x_out):
    batch_size = x_out.size(0)
    tensor_out = torch.zeros((batch_size,x_out.size(2)*x_out.size(3)*x_out.size(4))).cuda()
    for i in range(batch_size):
        prob = x_out[i,1,:,:,:].view(-1).cpu().detach()
        prob = list(prob)
        index = list(pd.Series(prob).sort_values().index[:56])
        tensor_out[i,index] += 1
    tensor_out = tensor_out.view(batch_size,x_out.size(2),x_out.size(3),x_out.size(4))
    return tensor_out



def transfer_3d_to_channels_2(x_out):
    x_out = x_out.cuda()
    x_inv = torch.zeros((x_out.size()),dtype=torch.float).cuda()
    x_inv[x_out==0] = 1
    x_out = torch.unsqueeze(x_out,dim=0)
    x_inv = torch.unsqueeze(x_inv,dim=0)
    out = torch.cat((x_inv,x_out),dim=0)
    return out




