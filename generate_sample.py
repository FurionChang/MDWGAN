import torch
from torch import nn
import numpy as np
from torch.utils import data
from io import StringIO
import random
from d2l import torch as d2l
from matplotlib import pyplot as plt
from GAN_network import Generator_1d,Discriminator_1d
from utils import load_data_from_dir,num_to_cor,elemsplot,load_array,transfer_gan_out,load_data_from_dir_3d,transfer_symmetry_1_8,transfer_3d_gan_out,transfer_3d_to_channels_2,transfer_3d_to_channels_2_for_batch,transfer_3d_gan_out_limit
from Gan_3d import Discriminator_3d,Generator_3d
from tqdm import tqdm
from simplejson import OrderedDict
import argparse


parser = argparse.ArgumentParser(description="Figure plot")
parser.add_argument('--dir', default=0, help="directory of model")
#parser.add_argument('--id', default=1, help="id of data")
#parser.add_argument('--num', default=1, help='total num of data')
args = vars(parser.parse_args())
f_dir = str(args['dir'])
index = [90,95,100]

#num = int(args['num'])

size_ele = [3,3,3]
size_ele_random = [6,6,6]
noise_dim = 100
num_epochs = 50
latent_dim = 100
batch_size = 8





moduli_cal_net = torch.load('./moduli_cal_model/total_50_lr_0.0005_epoch_45_moduli_cal_1.pt').cuda()
num = 3
for i in index:
    for j in range(3):
        num+=1
        G_path = './new_generator_model/'+f_dir+'/symm_total_100_lr_0.0005_epoch_'+str(i)+'_generator.pt'
        noise = torch.normal(0, 1, size=(batch_size, latent_dim,1,1,1)).cuda()
        G_net = torch.load(G_path).cuda()
        out_float = G_net(noise)
        out_binary= transfer_3d_gan_out_limit(out_float)
        out_binary = transfer_3d_to_channels_2_for_batch(out_binary)
        path_float = './new_generated_samples/'+f_dir+'/sample_epoch_'+str(i)+'_float_'+str(j)+'.pt'
        path_binary = './new_generated_samples/'+f_dir+'/sample_epoch_'+str(i)+'_binary_'+str(j)+'.pt'
        out_float += 1
        out_binary += 1
        torch.save(out_float,path_float)
        torch.save(out_binary,path_binary)
