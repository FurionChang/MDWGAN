from WGAN_network import WGAN_GP
import torch
from utils import load_array

path_G = './w_gan_generator/pure_gan_symm_13_17_1_channel/'
path_D = './w_gan_discriminator/pure_gan_symm_13_17_1_channel/'
WGAN_net = WGAN_GP(path_G,path_D)
print('data loading...')

print('load symmetric data...')
elems_symm = torch.load('./dataset/elems_symm.pt')
moduli_symm = torch.load('./dataset/moduli_symm.pt')

#elems_symm[elems_symm==0] += 1.3
#elems_symm[elems_symm==1] += 0.7
'''
print('load random data...')

elems_symm = torch.load('./dataset/elems_random.pt')
moduli_symm = torch.load('./dataset/moduli_random.pt')
elems_symm = elems_symm.float().cuda()
elems_symm = elems_symm.float().cuda()
'''
elems_symm[elems_symm==0] += 1.3
elems_symm[elems_symm==1] += 0.7

#elems_symm += 1
elems_symm = elems_symm[:,1:,:,:,:]
print('Successfully load the data...')
batch_size = 8
data_iter = load_array((elems_symm,moduli_symm), batch_size)
WGAN_net.train(data_iter)
