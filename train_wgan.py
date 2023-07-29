from WGAN_network import WGAN_GP
import torch
from utils import load_array
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--lr", default=1e-3,help="Learning rate")
parser.add_argument("--bsz", default=32,help="Batch size")
parser.add_argument("--epochs", default=100,help="Num epochs")
parser.add_argument("--G_channels", default=100,help="number of noise channels")
parser.add_argument("--D_channels", default=2,help="number of output channels")
parser.add_argument("--alpha", default=30,help="multiplier for Loss G2")
parser.add_argument("--beta", default=30,help="multiplier for Loss G3")
parser.add_argument("--obj", type=str, default='lg',help="Goal of your training, can be lg, iso, all, none.")
parser.add_argument("--G_path", type=str, default='./model/generator/',help="path of generator")
parser.add_argument("--D_path", type=str, default='./model/discriminator/',help="path of discriminator")
parser.add_argument("--dataset", type=str, default='random',help="dataset")
parser.add_argument("--surr_path", type=str, default=None,help="path of pretrained surrogate model")
args = vars(parser.parse_args())
lr = float(args['lr'])
bsz = int(args['bsz'])
epochs = int(args['epochs'])
alpha = int(args['alpha'])
beta = int(args['beta'])
G_channels = int(args['G_channels'])
D_channels = int(args['D_channels'])
obj = args['obj']
dataset = args['dataset']
surr_path = args['surr_path']
path_G = args['G_path']+str(D_channels)+'_channel_model/'+obj+'_'+dataset+'.pt'
path_D = args['D_path']+str(D_channels)+'_channel_model/'+obj+'_'+dataset+'.pt'


WGAN_net = WGAN_GP(path_G,path_D,bsz,lr=lr,G_channels=G_channels,D_channels=D_channels,surr_path=surr_path)
print('data loading...')

print('load data...')

elems_path = './dataset/elems_'+dataset+'.pt'
moduli_path = './dataset/moduli_'+dataset+'.pt'
elems = torch.load(elems_path)
moduli = torch.load(moduli_path)
elems = elems.float().cuda()
moduli = moduli.float().cuda()


elems += 1
if D_channels == 1:
    elems = elems[:,1:,:,:,:]

print('Successfully load the data...')


data_iter = load_array((elems,moduli), bsz)
WGAN_net.train(data_iter,obj,alpha,beta,epochs=epochs)
