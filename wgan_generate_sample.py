from WGAN_network import WGAN_GP
import torch
from utils import load_array
import time
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--lr", default=1e-3,help="Learning rate")
parser.add_argument("--bsz", default=32,help="Batch size")
parser.add_argument("--epochs", default=100,help="Num epochs")
parser.add_argument("--num", default=1000,help="Num of generated samples")
parser.add_argument("--G_channels", default=100,help="number of noise channels")
parser.add_argument("--D_channels", default=2,help="number of output channels")
parser.add_argument("--obj", type=str, default='lg',help="Goal of your training, can be lg, iso, all, none.")
parser.add_argument("--G_path", type=str, default='./model/generator/',help="path of generator")
parser.add_argument("--D_path", type=str, default='./model/discriminator/',help="path of discriminator")
parser.add_argument("--S_path", type=str, default='./sample/',help="path of generated samples")
parser.add_argument("--dataset", type=str, default='random',help="dataset")
parser.add_argument("--surr_path", type=str, default=None,help="path of pretrained surrogate model")
args = vars(parser.parse_args())
lr = float(args['lr'])
bsz = int(args['bsz'])
epochs = int(args['epochs'])
G_channels = int(args['G_channels'])
D_channels = int(args['D_channels'])
obj = args['obj']
dataset = args['dataset']
surr_path = args['surr_path']
path_G = args['G_path']+str(D_channels)+'_channel_model/'
path_D = args['D_path']+str(D_channels)+'_channel_model/'
path_s = args['S_path']+'sample_'+obj+'_'+dataset+'.pt'
num_sample = int(args['num'])

WGAN_net = WGAN_GP(path_G,path_D,bsz,lr=lr,G_channels=G_channels,D_channels=D_channels,epochs=epochs,surr_path=surr_path)
G_model_path = path_G+obj+'_'+dataset+'.pt'
D_model_path = path_D+obj+'_'+dataset+'.pt'
WGAN_net.load_model(D_model_path,G_model_path)
out = WGAN_net.generate_samples(num_sample,path_s) 
