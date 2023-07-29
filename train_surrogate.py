from simplejson import OrderedDict
import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from discriminate_agent import Dis_res_3d
from utils import load_array
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--lr", default=1e-3,help="Learning rate")
parser.add_argument("--bsz", default=32,help="Batch size")
parser.add_argument("--epochs", default=100,help="Num epochs")
parser.add_argument("--gamma", default=0.9,help="Gamma of scheduler")
parser.add_argument("--weight_decay", default=1e-5,help="Weight decay of optimizer")
parser.add_argument("--model_path", type=str, default="./surr_model/2_channel_model/",help="address of saved model")
args = vars(parser.parse_args())
lr = float(args['lr'])
batch_size = int(args['bsz'])
num_epochs = int(args['epochs'])
weight_decay = float(args['weight_decay'])
gamma = float(args['gamma'])
model_path = args['model_path']

def train(net,data_iter_train,data_iter_test,num_epochs,lr,gamma, model_path):
    
    criterion = nn.MSELoss(reduction='sum')
    trainer= torch.optim.Adam(net.parameters(), lr = lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(trainer,gamma)
    
    best_loss = 100
    for epoch in range(num_epochs):
        train_loss = 0
        _tqdm = tqdm(data_iter_train)
        num = 0
        ### Training Loop per epoch
        for X,moduli in _tqdm:
            trainer.zero_grad()
            num += 1
            moduli_out = net(X)
            loss = criterion(moduli_out,moduli)
            train_loss += loss.item()
            loss.backward()
            trainer.step()
            _tqdm.set_postfix(OrderedDict(stage='train',epoch=epoch,loss=train_loss/num))

        test_loss = 0
        _tqdm_test = tqdm(data_iter_test)
        num_test =0
        ### Test Loop per epoch
        with torch.no_grad():
            for X,moduli in _tqdm_test:
                num_test += 1
                moduli_out = net(X)
                loss = criterion(moduli_out,moduli)
                test_loss += loss.item()
                _tqdm_test.set_postfix(OrderedDict(stage='test',epoch=epoch,loss=test_loss/num_test))
            if test_loss/num_test < best_loss:
                best_loss = test_loss/num_test
                path = model_path+str(num_epochs)+'_lr_'+str(lr)+'_epoch_'+str(epoch)+'_surr.pt'
                torch.save(net,path)
                print('Model successfully saved in '+path+' ...\n')
            elif epoch%5==0:
                path = model_path+str(num_epochs)+'_lr_'+str(lr)+'_epoch_'+str(epoch)+'_surr.pt'
                torch.save(net,path)
                print('Model successfully saved in '+path+' ...\n')
        scheduler.step()

    path = model_path+str(num_epochs)+'_lr_'+str(lr)+'_surr.pt'
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
elems_random_train = elems_random[0:18000,1:,:,:,:]
elems_random_test = elems_random[18000:,1:,:,:,:]
elems_symm_train = elems_symm[0:18000,1:,:,:,:]
elems_symm_test = elems_symm[18000:,1:,:,:,:]
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
net = Dis_res_3d(channels=2).to(device)
print('net constructed...')
print('train begin...')
train(net, data_iter_train,data_iter_test, num_epochs, lr,gamma,model_path)
