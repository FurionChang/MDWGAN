import torch
from torch import nn
import numpy as np
from torch.utils import data
from io import StringIO
import random
from d2l import torch as d2l
from matplotlib import pyplot as plt
from utils import load_data_from_dir,load_array,transfer_gan_out,load_data_from_dir_3d,transfer_symmetry_1_8,transfer_3d_gan_out,transfer_3d_to_channels_2,transfer_3d_to_channels_2_for_batch,transfer_3d_gan_out_max,transfer_3d_gan_out_limit
from Gan_3d import Discriminator_3d,Generator_3d,BinaryFunc_sample,BinaryFunc_limit
from tqdm import tqdm
from simplejson import OrderedDict
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts



size_ele = [3,3,3]
size_ele_random = [6,6,6]
noise_dim = 100
num_epochs = 100
latent_dim = 100
lr_D = 5e-3
lr_G = 5e-3
batch_size = 8
#Binary_layer_sample = BinaryFunc_sample()
#Binary_layer_limit = BinaryFunc_limit()

def update_D(X, Z, net_D, net_G, loss_G1, trainer_D):
    """Update discriminator."""
    batch_size = X.shape[0]
    ones = torch.ones((batch_size,), device=X.device).float()
    zeros = torch.zeros((batch_size,), device=X.device).float()
    #zeros -= 1
    #ones = ones.to(torch.int64)
    #zeros = zeros.to(torch.int64)
    trainer_D.zero_grad()
    real_Y = net_D(X)
    #fake_Y_goal = torch.cat((ones,zeros),dim=1)
    #real_Y_goal = torch.cat((zeros,ones),dim=1)
    fake_X = net_G(Z)
    #fake_X,_= transfer_3d_gan_out(fake_X)
    #fake_X = transfer_3d_to_channels_2_for_batch(fake_X)
    #fake_X = Binary_layer_sample.apply(fake_X)
    fake_X += 1
    # Do not need to compute gradient for `net_G`, detach it from
    # computing gradients.
    fake_Y = net_D(fake_X.detach())
    loss_D = (loss_G1(real_Y, ones.reshape(real_Y.shape)) +
              loss_G1(fake_Y, zeros.reshape(fake_Y.shape))) / 2
    #loss_D = (loss_G1(real_Y, ones.reshape(real_Y.shape)) +
    #          loss_G1(fake_Y, zeros.reshape(fake_Y.shape))) / 2
    loss_D.backward()
    trainer_D.step()
    return loss_D.item()

def update_G(Z, net_D, net_G, net_moduli, loss_G1, loss_G2_1, loss_G2_2, trainer_G):
    """Update generator."""
    batch_size = Z.shape[0]
    #ones_part = torch.ones((batch_size,2,size_ele_random[0],size_ele_random[1],size_ele_random[2])).float().cuda()
    ones = torch.ones((batch_size,), device=Z.device).float()
    #ones = ones.to(torch.int64)
    #zeros = torch.zeros((batch_size,1), device=Z.device).float()
    #fake_Y_goal = torch.cat((zeros,ones),dim=1)
    moduli_goal = torch.ones((batch_size,3), device=Z.device).float()
    moduli_goal += 0.8
    trainer_G.zero_grad()
    # We could reuse `fake_X` from `update_D` to save computation
    fake_X = net_G(Z)
    #fake_X = Binary_layer_sample.apply(fake_X)
    #fake_X,_= transfer_3d_gan_out(fake_X)
    #fake_X = transfer_3d_to_channels_2_for_batch(fake_X)
    fake_X = fake_X + 1
    # Recomputing `fake_Y` is needed since `net_D` is changed
    fake_Y = net_D(fake_X)
    #moduli_fake_Y = net_moduli(fake_X)
    #moduli_fake_Y_new_1 = torch.cat((moduli_fake_Y[:,1:],moduli_fake_Y[:,:1]),dim=1)
    #moduli_fake_Y_new_2 = torch.cat((moduli_fake_Y[:,2:],moduli_fake_Y[:,:2]),dim=1)
    loss_G_1 = loss_G1(fake_Y, ones.reshape(fake_Y.shape))
    #loss_G_1 = loss_G1(fake_Y, ones.reshape(fake_Y.shape))
    #loss_G_2 = loss_G2_1(moduli_fake_Y,moduli_fake_Y_new_1) + loss_G2_1(moduli_fake_Y,moduli_fake_Y_new_2)
    #loss_G_2 = loss_G2_1(moduli_fake_Y,moduli_goal.reshape(moduli_fake_Y.shape))
    loss_G = loss_G_1 
    #loss_G = 2*loss_G_1 + loss_G_2
    loss_G.backward()
    trainer_G.step()
    return loss_G.item()

def train(net_D, net_G, net_moduli, data_iter, num_epochs, lr_D, lr_G, latent_dim, num_plots=4):
    decay_rate = 0.96
    #loss_G1 = nn.CrossEntropyLoss(reduction='sum')
    loss_G1 = nn.BCEWithLogitsLoss(reduction='sum')
    #loss_G1 = nn.SoftMarginLoss(reduction='sum')
    loss_G2_1 = nn.MSELoss(reduction='sum')
    loss_G2_2 = nn.L1Loss(reduction='sum')
    for w in net_D.parameters():
        nn.init.normal_(w, 0, 0.2)
    for w in net_G.parameters():
        nn.init.normal_(w, 0, 0.2)
        
    trainer_D = torch.optim.Adam(net_D.parameters(), lr = lr_D)
    trainer_G = torch.optim.Adam(net_G.parameters(), lr = lr_G)
    lr_scheduler_D = CosineAnnealingWarmRestarts(trainer_D, T_0=4,T_mult=2)
    lr_scheduler_G = CosineAnnealingWarmRestarts(trainer_G, T_0=4,T_mult=2)
    
    #elemsplot(size_ele, x_cor, y_cor, z_cor, num_plots)
    best_loss_D = 100
    best_loss_G = 100
    for epoch in range(num_epochs):
        # Train one epoch.
        # loss_D, loss_G, num_examples
        loss_D = 0
        loss_G = 0
        num = 0
        _tqdm = tqdm(data_iter)
        for X,moduli in _tqdm:
            batch_size = X.shape[0]
            #print(X.shape)
            Z = torch.normal(0, 10, size=(batch_size, latent_dim,1,1,1)).cuda()
            loss_D += update_D(X, Z, net_D, net_G, loss_G1, trainer_D)
            loss_G += update_G(Z, net_D, net_G, net_moduli, loss_G1, loss_G2_1,loss_G2_2, trainer_G)
            num += batch_size
            _tqdm.set_postfix(OrderedDict(stage='train',epoch=epoch,loss=loss_G/num))

        loss_D_res = loss_D/num
        loss_G_res = loss_G/num
        lr_scheduler_D.step()
        lr_scheduler_G.step()
        '''
        sample replacement
        with torch.no_grad():
            test_count = 0 
            change_count = 0
            for X,moduli in _tqdm:
                if test_count == 20:
                    break
                batch_size = X.shape[0]
                Z = torch.normal(0, 10, size=(batch_size, latent_dim,1,1,1)).cuda()
                fake_X = transfer_3d_gan_out_limit(fake_X)
                fake_X = transfer_3d_to_channels_2_for_batch(fake_X)
                fake_Y = net_moduli(fake_X)
                for i in range(batch_size):
                    real_moduli_sum = torch.sum(moduli[i,:])
                    fake_moduli_sum = torch.sum(fake_Y[i,:])
                    if real_moduli_sum < fake_moduli_sum:
                        X[i,:,:,:,:]=fake_X[i,:,:,:,:]
                        change_count += 1
                test_count += 1
                _tqdm.set_postfix(OrderedDict(stage='test',epoch=epoch))
            output_sentence = "In this epoch, we change "+str(change_count)+" samples in the total 160 samples."
            print(output_sentence)
        '''
        if loss_D_res < best_loss_D or loss_G_res < best_loss_G:
            best_loss_D = loss_D_res
            path = './new_discriminator_model/pure_gan_13_17_float_random/symm_total_'+str(num_epochs)+'_lr_'+str(lr_D)+'_epoch_'+str(epoch)+'_discriminator.pt'
            torch.save(net_D,path)
            print('Discriminator model successfully saved in '+path+' ...\n')
        #if loss_G_res < best_loss_G:
            best_loss_G = loss_G_res
            path = './new_generator_model/pure_gan_13_17_float_random/symm_total_'+str(num_epochs)+'_lr_'+str(lr_G)+'_epoch_'+str(epoch)+'_generator.pt'
            torch.save(net_G,path)
            print('Generator model successfully saved in '+path+' ...\n')
        # Visualize generated examples
        if epoch % 5 == 0:
            with torch.no_grad():
                for i in range(10):
                    Z = torch.normal(0, 10, size=(batch_size, latent_dim,1,1,1)).cuda()
                    fake_X = net_G(Z).detach()
                    #print(fake_X)
                    fake_X = transfer_3d_gan_out_limit(fake_X)
                    fake_X = transfer_3d_to_channels_2_for_batch(fake_X)
                    #fake_X = Binary_layer_limit.apply(fake_X)
                    fake_X += 1
                    fake_Y = net_moduli(fake_X)
                    print(fake_Y)

            path = './new_discriminator_model/pure_gan_13_17_float_random/symm_total_'+str(num_epochs)+'_lr_'+str(lr_D)+'_epoch_'+str(epoch)+'_discriminator.pt'
            torch.save(net_D,path)
            print('Discriminator model successfully saved in '+path+' ...\n')
            path = './new_generator_model/pure_gan_13_17_float_random/symm_total_'+str(num_epochs)+'_lr_'+str(lr_G)+'_epoch_'+str(epoch)+'_generator.pt'
            torch.save(net_G,path)
            print('Generator model successfully saved in '+path+' ...\n')
        '''
        Z = torch.normal(0, 1, size=(batch_size, latent_dim)).cuda()
        fake_X = net_G(Z).detach()
        fake_X = transfer_gan_out(fake_X,batch_size=batch_size,sample_size=sample_size)
        
        animator.axes[1].cla()
        fkx_cor, fky_cor, fkz_cor = num_to_cor(torch.tensor(size_ele).float(), fake_X.long().cpu())
        # data type
        
        fkx_cor = fkx_cor.float()
        fky_cor = fky_cor.float()
        fkz_cor = fkz_cor.float()
        #print(fkx_cor,fky_cor,fkz_cor)
        #
        animator.axes[1].cla()
        elemsplot(size_ele, fkx_cor, fky_cor, fkz_cor, num_plots,epoch)
        # Show the losses
        loss_D, loss_G = metric[0] / metric[2], metric[1] / metric[2]
        animator.add(epoch + 1, (loss_D, loss_G))
        '''
        

    path = './new_discriminator_model/pure_gan_13_17_float_random/symm_total_'+str(num_epochs)+'_lr_'+str(lr_D)+'_epoch_'+str(num_epochs)+'_discriminator.pt'
    torch.save(net_D,path)
    print('Discriminator model successfully saved in '+path+' ...\n')
    path = './new_generator_model/pure_gan_13_17_float_random/symm_total_'+str(num_epochs)+'_lr_'+str(lr_G)+'_epoch_'+str(num_epochs)+'_generator.pt'
    torch.save(net_G,path)
    print('Generator model successfully saved in '+path+' ...\n')


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('data loading...')
'''
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

elems_symm[elems_symm==0] += 1.3
elems_symm[elems_symm==1] += 0.7
#elems_symm += 1
print('Successfully load the data...')
G_net = Generator_3d(in_channels=noise_dim,out_channels=2)
D_net = Discriminator_3d()
moduli_cal_net = torch.load('./moduli_cal_model/total_50_lr_0.0005_epoch_45_moduli_cal_1.pt').cuda()
G_net = G_net.to(device)
D_net = D_net.to(device)
moduli_cal_net = moduli_cal_net.to(device)
data_iter = load_array((elems_symm,moduli_symm), batch_size)
train(D_net, G_net, moduli_cal_net, data_iter, num_epochs, lr_D, lr_G, latent_dim)