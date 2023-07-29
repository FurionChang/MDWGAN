import torch
from torch import nn
import numpy as np
from torch.utils import data
from utils import load_array
import matplotlib.pyplot as plt

size_ele_random = [6,6,6]
lr = 0.001
batch_size = 32
num_epochs = 30


model = torch.load('./surr_model/total_100_lr_0.001_moduli_cal_2_channel_batch_32.pt').cuda()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('data loading...')
print('load random data...')
elems_random = torch.load('./dataset/elems_random.pt')
moduli_random = torch.load('./dataset/moduli_random.pt')
elems_symm = torch.load('./dataset/elems_symm.pt')
moduli_symm = torch.load('./dataset/moduli_symm.pt')
elems = torch.cat((elems_random,elems_symm),dim=0).float().cuda()
elems = elems[:,:,:,:,:]
elems += 1
moduli = torch.cat((moduli_random,moduli_symm),dim=0).float().cuda()
data_iter = load_array((elems,moduli), batch_size)
test_loss = 0
num_test =0
criterion = nn.MSELoss()
fake_x_moduli = torch.zeros((40000,)).cuda()
real_x_moduli = torch.zeros((40000,)).cuda()
fake_y_moduli = torch.zeros((40000,)).cuda()
real_y_moduli = torch.zeros((40000,)).cuda()
fake_z_moduli = torch.zeros((40000,)).cuda()
real_z_moduli = torch.zeros((40000,)).cuda()


with torch.no_grad():
    for X,moduli in data_iter:
        num_test += 1
        moduli_out = model(X)
        loss = criterion(moduli_out,moduli)
        test_loss += loss.item()
        fake_x_moduli[batch_size*(num_test-1):batch_size*num_test] = moduli_out[:,0]/0.6
        fake_y_moduli[batch_size*(num_test-1):batch_size*num_test] = moduli_out[:,1]/0.6
        fake_z_moduli[batch_size*(num_test-1):batch_size*num_test] = moduli_out[:,2]/0.6
        real_x_moduli[batch_size*(num_test-1):batch_size*num_test] = moduli[:,0]/0.6
        real_y_moduli[batch_size*(num_test-1):batch_size*num_test] = moduli[:,1]/0.6
        real_z_moduli[batch_size*(num_test-1):batch_size*num_test] = moduli[:,2]/0.6
        
real_moduli_var_x = torch.var(real_x_moduli)
mse_moduli_x = torch.mean((real_x_moduli-fake_x_moduli)**2)
real_moduli_var_y = torch.var(real_y_moduli)
mse_moduli_y = torch.mean((real_y_moduli-fake_y_moduli)**2)
real_moduli_var_z = torch.var(real_z_moduli)
mse_moduli_z = torch.mean((real_z_moduli-fake_z_moduli)**2)

print("Moduli variance for x axis is: ")
print(real_moduli_var_x)
print("MSE between real moduli and fake moduli on x axis is: ")
print(mse_moduli_x)
print("Moduli variance for y axis is: ")
print(real_moduli_var_y)
print("MSE between real moduli and fake moduli on y axis is: ")
print(mse_moduli_y)
print("Moduli variance for z axis is: ")
print(real_moduli_var_z)
print("MSE between real moduli and fake moduli on z axis is: ")
print(mse_moduli_z)

path_x = './figure/moduli_x_real.pt'
path_y = './figure/moduli_y_real.pt'
path_z = './figure/moduli_z_real.pt'
torch.save(real_x_moduli,path_x)
torch.save(real_y_moduli,path_y)
torch.save(real_z_moduli,path_z)
path_x_1 = './figure/moduli_x_fake.pt'
path_y_1 = './figure/moduli_y_fake.pt'
path_z_1 = './figure/moduli_z_fake.pt'
torch.save(fake_x_moduli,path_x_1)
torch.save(fake_y_moduli,path_y_1)
torch.save(fake_z_moduli,path_z_1)

line_x = np.arange(1.5, 3.2, 0.01)
line_y = line_x
fig = plt.figure(figsize=(11,11),dpi=200)
ax = plt.axes()
plt.plot(real_x_moduli.cpu(),fake_x_moduli.cpu(),'o',markersize=1)
plt.plot(line_x,line_y)
#plt.title('Fem X moduli vs. Surrogate X moduli',fontsize = 18)
plt.xlabel('$E_{x(fem)}$',fontsize = 60)
plt.ylabel('$E_{x(surr)}$',fontsize = 60)
plt.xticks(np.arange(1.45, 3.3, 0.3),size=25)
plt.yticks(np.arange(1.45, 3.3, 0.3),size=25)
ax.spines['bottom'].set_linewidth(5)
ax.spines['top'].set_linewidth(5)
ax.spines['left'].set_linewidth(5)
ax.spines['right'].set_linewidth(5)
path = 'X_moduli_plot_batch_'+str(batch_size)+'.png'
plt.savefig(path)
plt.show()

fig = plt.figure(figsize=(11,11),dpi=200)
ax = plt.axes()
plt.plot(real_y_moduli.cpu(),fake_y_moduli.cpu(),'o',markersize=1)
plt.plot(line_x,line_y)
#plt.title('Real Y moduli vs. Calculated Y moduli',fontsize = 20)
plt.xlabel('$E_{y(fem)}$',fontsize = 30)
plt.ylabel('$E_{y(surr)}$',fontsize = 30)
plt.xticks(np.arange(1.45, 3.3, 0.3),size=25)
plt.yticks(np.arange(1.45, 3.3, 0.3),size=25)
ax.spines['bottom'].set_linewidth(5)
ax.spines['top'].set_linewidth(5)
ax.spines['left'].set_linewidth(5)
ax.spines['right'].set_linewidth(5)
path = 'Y_moduli_plot_batch_'+str(batch_size)+'.png'
plt.savefig(path)
plt.show()

fig = plt.figure(figsize=(11,11),dpi=200)
ax = plt.axes()
plt.plot(real_z_moduli.cpu(),fake_z_moduli.cpu(),'o',markersize=1)
plt.plot(line_x,line_y)
#plt.title('Real Z moduli vs. Calculated Z moduli',fontsize = 20)
plt.xlabel('$E_{z(fem)}$',fontsize = 30)
plt.ylabel('$E_{z(surr)}$',fontsize = 30)
plt.xticks(np.arange(1.45, 3.3, 0.3),size=25)
plt.yticks(np.arange(1.45, 3.3, 0.3),size=25)
ax.spines['bottom'].set_linewidth(5)
ax.spines['top'].set_linewidth(5)
ax.spines['left'].set_linewidth(5)
ax.spines['right'].set_linewidth(5)
path = 'Z_moduli_plot_batch_'+str(batch_size)+'.png'
plt.savefig(path)
plt.show()