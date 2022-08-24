from simplejson import OrderedDict
import torch
from torch import nn
import numpy as np
from torch.utils import data
from io import StringIO
import random
from tqdm import tqdm
from d2l import torch as d2l
from matplotlib import pyplot as plt
from GAN_network import Generator_1d,Discriminator_1d
from discriminate_agent import discriminator_agent,Dis_res,Dis_res_3d
from utils import num_to_cor,elemsplot,load_array,transfer_gan_out,transfer_1d_to_3d,load_data_from_dir,load_data_from_dir_3d,transfer_symmetry_1_8


size_ele_random = [6,6,6]
size_ele_symm = [3,3,3]
lr = 0.0005
batch_size = 8
num_epochs = 100


def train(net,data_iter_train,data_iter_test,num_epochs,lr,step_size = 5,gamma = 0.1):
    
    criterion = nn.MSELoss(reduction='sum')
    
    #for w in net.parameters():
        #nn.init.normal_(w, 0, 0.5)
    trainer= torch.optim.Adam(net.parameters(), lr = lr, weight_decay=0.00001)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer,step_size,gamma)
    
    #elemsplot(size_ele, x_cor, y_cor, z_cor, num_plots)
    best_loss = 100
    for epoch in range(num_epochs):
        # Train one epoch
        # # loss_D, loss_G, num_examples
        train_loss = 0
        _tqdm = tqdm(data_iter_train)
        num = 0
        for X,moduli in _tqdm:
            trainer.zero_grad()
            num += 1
            #X = transfer_1d_to_3d(X,size_ele).cuda()
            #print(X.shape)
            moduli_out = net(X)
            #moduli *= 10
            loss = criterion(moduli_out,moduli)
            train_loss += loss.item()
            loss.backward()
            trainer.step()
            _tqdm.set_postfix(OrderedDict(stage='train',epoch=epoch,loss=train_loss/num))

        test_loss = 0
        _tqdm_test = tqdm(data_iter_test)
        num_test =0
        with torch.no_grad():
            for X,moduli in _tqdm_test:
                num_test += 1
                #X = transfer_1d_to_3d(X,size_ele).cuda()
                #print(X.shape)
                moduli_out = net(X)
                #moduli *= 10
                loss = criterion(moduli_out,moduli)
                test_loss += loss.item()
                _tqdm_test.set_postfix(OrderedDict(stage='test',epoch=epoch,loss=test_loss/num_test))
            if test_loss/num_test < best_loss:
                best_loss = test_loss/num_test
                path = './moduli_cal_model/total_'+str(num_epochs)+'_lr_'+str(lr)+'_epoch_'+str(epoch)+'_moduli_cal_2.pt'
                torch.save(net,path)
                print('Model successfully saved in '+path+' ...\n')
            elif epoch%5==0:
                path = './moduli_cal_model/total_'+str(num_epochs)+'_lr_'+str(lr)+'_epoch_'+str(epoch)+'_moduli_cal_2.pt'
                torch.save(net,path)
                print('Model successfully saved in '+path+' ...\n')
        scheduler.step()


    path = './moduli_cal_model/total_'+str(num_epochs)+'_lr_'+str(lr)+'_moduli_cal_2.pt'
    torch.save(net,path)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('data loading...')
print('load random data...')
elems_random = torch.load('./dataset/elems_random.pt')
moduli_random = torch.load('./dataset/moduli_random.pt')
print('load symmetric data...')
elems_symm = torch.load('./dataset/elems_symm.pt')
moduli_symm = torch.load('./dataset/moduli_symm.pt')
print('Successfully load the data...')
#print(elems_random.size())
#print(moduli_random.size())
#print(elems_symm.size())
#print(moduli_symm.size())
elems_random_train = elems_random[0:18000,:,:,:,:]
elems_random_test = elems_random[18000:,:,:,:,:]
elems_symm_train = elems_symm[0:18000,:,:,:,:]
elems_symm_test = elems_symm[18000:,:,:,:,:]
elems_train = torch.cat((elems_random_train,elems_symm_train),dim=0)
elems_test = torch.cat((elems_random_test,elems_symm_test),dim=0)
elems_train += 1
elems_test += 1
moduli_random_train = moduli_random[0:18000,:]
moduli_random_test = moduli_random[18000:,:]
moduli_symm_train = moduli_symm[0:18000,:]
moduli_symm_test = moduli_symm[18000:,:]
moduli_train = torch.cat((moduli_random_train,moduli_symm_train),dim=0)
moduli_test = torch.cat((moduli_random_test,moduli_symm_test),dim=0)
print('train and test data set constructed...')
data_iter_train = load_array((elems_train,moduli_train), batch_size)
data_iter_test = load_array((elems_test,moduli_test), batch_size)
net = Dis_res_3d().to(device)
print('net constructed...')
print('train begin...')
train(net, data_iter_train,data_iter_test, num_epochs, lr)
