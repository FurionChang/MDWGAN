import torch
from torch import nn
import numpy as np
from torch.utils import data
from io import StringIO
import random
from d2l import torch as d2l
from matplotlib import pyplot as plt
import argparse

parser = argparse.ArgumentParser(description="Figure plot")
parser.add_argument('--index', default=0, help="index of data")
parser.add_argument('--id', default=1, help="id of data")
parser.add_argument('--num', default=1, help='total num of data')
args = vars(parser.parse_args())

index = int(args['index'])
id = int(args['id'])
num = int(args['num'])

moduli_cal = torch.load('./moduli_cal_model/total_50_lr_0.0005_epoch_45_moduli_cal_1.pt').cuda()

path_1 = './generated_samples/symm_generated_0_'+str(id)+'.pt'
data = torch.load(path_1)
#data -= 1
moduli_total = moduli_cal(data)
print(moduli_total)
moduli = moduli_total[index,:]
data_choice = data[index,1,:,:,:]
path_2 = './sample_plot/sample_'+str(num)+'.pt'
torch.save(data_choice,path_2)
data -= 1
g_struct = data[index,1,:,:,:].cpu()
gs_np = g_struct.numpy()
gs_np_lgc_re = gs_np > 0
gs_np_lgc_ma = gs_np < 1

ax = plt.figure(figsize=(10,15)).add_subplot(projection='3d')
ax.voxels(gs_np_lgc_re, edgecolor='None')
ax.set_xlim(0,6)
ax.set_ylim(0,6)
ax.set_zlim(0,6)
plt.title(moduli)
path_3 = './sample_plot/sample_plot_'+str(num)+'.png'
plt.savefig(path_3)
plt.show()

'''
# plt.save()
ax = plt.figure(figsize=(10,15)).add_subplot(projection='3d')
ax.voxels(gs_np_lgc_ma, edgecolor='None')
ax.set_xlim(0,6)
ax.set_ylim(0,6)
ax.set_zlim(0,6)
'''
