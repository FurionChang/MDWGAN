from WGAN_network import WGAN_GP
import torch
from utils import load_array


exp_name = 'pure_gan_symm'
path_G = './w_gan_generator/'+exp_name+'/'
path_D = './w_gan_discriminator/'+exp_name+'/'
path_s = './w_gan_sample/'+exp_name+'/'
WGAN_net = WGAN_GP(path_G,path_D)
G_model_path = path_G +'generator_lr_0.0001_epoch_100.pt'
D_model_path = path_D +'discriminator_lr_0.0001_epoch_100.pt'
WGAN_net.load_model(D_model_path,G_model_path)
WGAN_net.generate_samples(5,path_s)
