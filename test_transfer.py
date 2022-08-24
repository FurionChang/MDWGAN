import torch
from torch import nn
import numpy as np
from torch.utils import data
from utils import load_data_from_dir,num_to_cor,elemsplot,load_array,transfer_gan_out,transfer_1d_to_3d,load_data_from_dir_3d
from discriminate_agent import discriminator_agent
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image

size_ele_random = [6,6,6]
lr = 0.001
batch_size = 8
num_epochs = 30


model = torch.load('./moduli_cal_model/total_50_lr_0.0005_epoch_45_moduli_cal_1.pt').cuda()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('data loading...')
print('load random data...')
elems_random = torch.load('./dataset/elems_random.pt')
moduli_random = torch.load('./dataset/moduli_random.pt')
elems = elems_random.float().cuda()
elems += 1
moduli = moduli_random.float().cuda()
data_iter = load_array((elems,moduli), batch_size)
test_loss = 0
num_test =0
criterion = nn.MSELoss()
fake_x_moduli = torch.zeros((20000,)).cuda()
real_x_moduli = torch.zeros((20000,)).cuda()
fake_y_moduli = torch.zeros((20000,)).cuda()
real_y_moduli = torch.zeros((20000,)).cuda()
fake_z_moduli = torch.zeros((20000,)).cuda()
real_z_moduli = torch.zeros((20000,)).cuda()


with torch.no_grad():
    for X,moduli in data_iter:
        num_test += 1
        moduli_out = model(X)
        loss = criterion(moduli_out,moduli)
        test_loss += loss.item()
        fake_x_moduli[8*(num_test-1):8*num_test] = moduli_out[:,0]
        fake_y_moduli[8*(num_test-1):8*num_test] = moduli_out[:,1]
        fake_z_moduli[8*(num_test-1):8*num_test] = moduli_out[:,2]
        real_x_moduli[8*(num_test-1):8*num_test] = moduli[:,0]
        real_y_moduli[8*(num_test-1):8*num_test] = moduli[:,1]
        real_z_moduli[8*(num_test-1):8*num_test] = moduli[:,2]
        

line_x = np.arange(1., 2., 0.01)
line_y = line_x
fig = plt.figure(figsize=(8,8),dpi=120)
plt.plot(real_x_moduli.cpu(),fake_x_moduli.cpu(),'o',markersize=1)
plt.plot(line_x,line_y)
#plt.title('Fem X moduli vs. Surrogate X moduli',fontsize = 18)
plt.xlabel('$E_{x(fem)}$',fontsize = 25)
plt.ylabel('$E_{x(surr)}$',fontsize = 25)
plt.xticks(size=20)
plt.yticks(size=20)
plt.savefig('X_moduli_plot_1.png')
plt.show()

fig = plt.figure(figsize=(8,8),dpi=120)
plt.plot(real_y_moduli.cpu(),fake_y_moduli.cpu(),'o',markersize=1)
plt.plot(line_x,line_y)
#plt.title('Real Y moduli vs. Calculated Y moduli',fontsize = 20)
plt.xlabel('$E_{y(fem)}$',fontsize = 25)
plt.ylabel('$E_{y(surr)}$',fontsize = 25)
plt.xticks(size=20)
plt.yticks(size=20)
plt.savefig('Y_moduli_plot_1.png')
plt.show()

fig = plt.figure(figsize=(8,8),dpi=120)
plt.plot(real_z_moduli.cpu(),fake_z_moduli.cpu(),'o',markersize=1)
plt.plot(line_x,line_y)
#plt.title('Real Z moduli vs. Calculated Z moduli',fontsize = 20)
plt.xlabel('$E_{z(fem)}$',fontsize = 25)
plt.ylabel('$E_{z(surr)}$',fontsize = 25)
plt.xticks(size=20)
plt.yticks(size=20)
plt.savefig('Z_moduli_plot_1.png')
plt.show()