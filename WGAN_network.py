from tkinter.tix import X_REGION
from typing import List
import warnings
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Function
from torch.autograd import Variable
from d2l import torch as d2l
import pandas as pd
from torch import autograd
import time as t
import matplotlib.pyplot as plt
#plt.switch_backend('agg')
import os
from itertools import chain
from torchvision import utils
from tqdm import tqdm
from simplejson import OrderedDict


class WDBlock_res_3d(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, kernel_size = 3,stride=1, padding=1):
        super(WDBlock_res_3d, self).__init__()
        self.conv1 = nn.Conv3d(
            in_planes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class WD_block_3d(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, kernel_size = 3,strides=1, padding=1):
        super(WD_block_3d, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels,
                                               kernel_size, strides, padding,
                                               bias=False)
        self.batch_norm = nn.InstanceNorm3d(out_channels,affine = True)
        self.activation = nn.LeakyReLU(0.2,inplace = True)

    def forward(self, x):
        return self.activation(self.batch_norm(self.conv3d(x)))

class WDiscriminator_3d(nn.Module):
    def __init__(self,in_channels = 2):
        super(WDiscriminator_3d, self).__init__()
        # c*6*6
        self.d1 = WD_block_3d(in_channels=in_channels,out_channels=256,kernel_size=3,strides=1,padding=0) 
        # 256*4*4
        self.d2 = WD_block_3d(in_channels=256,out_channels=512,kernel_size=4,strides=2,padding=1) 
        # 512*2*2
        #self.d3 = WD_block_3d(in_channels=512,out_channels=1024,kernel_size=4,strides=2,padding=1)
        # 1024*1*1
        self.out = nn.Conv3d(512, 1, 4, 2, 1)


    def forward(self, x):
        out = self.d1(x)
        out = self.d2(out)
        #out = self.d3(out)
        out = self.out(out)
        return out

    def feature_extraction(self,x):
        out = self.d1(x)
        out = self.d2(out)
        #out = self.d3(out)
        return out.view(-1,512*2*2*2)



class WG_block_3d(nn.Module):
    def __init__(self,  in_channels, out_channels, kernel_size=4, strides=2,
                 padding=1, **kwargs):
        super(WG_block_3d, self).__init__(**kwargs)
        self.conv3d_trans = nn.ConvTranspose3d(in_channels, out_channels,
                                               kernel_size, strides, padding,
                                               bias=False)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.activation = nn.ReLU(True)

    def forward(self, x):
        return self.activation(self.batch_norm(self.conv3d_trans(x)))


class WGenerator_3d(nn.Module):
    def __init__(self,in_channels,out_channels=2):
        super(WGenerator_3d, self).__init__()
        self.g1 = WG_block_3d(in_channels=in_channels,out_channels=512,kernel_size=4,strides=2,padding=1)
        self.g2 = WG_block_3d(in_channels=512,out_channels=256,kernel_size=4,strides=2,padding=1)
        self.g3 = nn.ConvTranspose3d(in_channels=256, out_channels=out_channels,kernel_size=3,stride=1, padding=0,bias=False)
        #self.g3 = WG_block_3d(in_channels=256,out_channels=out_channels,kernel_size=3,strides=1,padding=0)
        self.output = nn.Tanh()

    def forward(self, X):
        out = self.g1(X)
        out = self.g2(out)
        out = self.g3(out)
        out = self.output(out)
        out = out/2+1.5
        return out


class WGAN_GP(object):
    def __init__(self,G_path,D_path, G_channels=100,D_channels=2,epochs=100):
        self.G_path = G_path
        self.D_path = D_path
        if not os.path.exists(self.G_path):
            os.mkdir(self.G_path)
        if not os.path.exists(self.D_path):
            os.mkdir(self.D_path)
        self.G = WGenerator_3d(G_channels,out_channels=D_channels)
        self.D = WDiscriminator_3d(D_channels)

        self.moduli_cal = torch.load('./moduli_cal_model/total_50_lr_0.0005_epoch_45_moduli_cal_1.pt').cuda()
        # Check if cuda is available
        self.cuda = True if torch.cuda.is_available() else False
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.G.to(self.device)
        self.D.to(self.device)

        # WGAN values from paper
        self.learning_rate = 1e-4
        self.b1 = 0.5
        self.b2 = 0.999
        self.batch_size = 8

        # WGAN_gradient penalty uses ADAM
        self.d_optimizer = optim.Adam(self.D.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))
        self.g_optimizer = optim.Adam(self.G.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))

        # Set the logger
        '''
        self.logger = Logger('./logs')
        self.logger.writer.flush()
        self.number_of_images = 10
        '''
        self.moduli_critic = nn.MSELoss(reduction='sum')
        self.epochs = epochs
        self.generator_iters = 2500
        self.critic_iter = 3
        self.lambda_term = 10

    def get_torch_variable(self, arg):
        if self.cuda:
            return Variable(arg).cuda()
        else:
            return Variable(arg)


    def train(self, train_loader):
        #self.t_begin = t.time()
        #self.file = open("inception_score_graph.txt", "w")

        # Now batches are callable self.data.next()
        self.data = self.get_infinite_batches(train_loader)

        one = torch.tensor(1, dtype=torch.float)
        mone = one * -1
        if self.cuda:
            one = one.cuda()
            mone = mone.cuda()
        
        for epoch in range(self.epochs):
            g_iter_tqdm = tqdm(range(self.generator_iters))
            for g_iter in g_iter_tqdm:
                # Requires grad, Generator requires_grad = False
                for p in self.D.parameters():
                    p.requires_grad = True

                d_loss_real = 0
                d_loss_fake = 0
                # Wasserstein_D = 0
                # Train Dicriminator forward-loss-backward-update self.critic_iter times while 1 Generator forward-loss-backward-update
                for d_iter in range(self.critic_iter):
                    self.D.zero_grad()

                    X, moduli = self.data.__next__()
                    # Check for batch to have full batch_size
                    if (X.size()[0] != self.batch_size):
                        continue

                    z = torch.rand((self.batch_size, 100, 1, 1, 1))

                    X, z = self.get_torch_variable(X), self.get_torch_variable(z)

                    # Train discriminator
                    # WGAN - Training discriminator more iterations than generator
                    # Train with real images
                    d_loss_real = self.D(X)
                    d_loss_real = d_loss_real.mean()
                    d_loss_real.backward(mone)

                    # Train with fake images
                    z = self.get_torch_variable(torch.randn(self.batch_size, 100, 1, 1, 1))

                    fake_X = self.G(z)
                    d_loss_fake = self.D(fake_X)
                    d_loss_fake = d_loss_fake.mean()
                    d_loss_fake.backward(one)

                    # Train with gradient penalty
                    gradient_penalty = self.calculate_gradient_penalty(X.data, fake_X.data)
                    gradient_penalty.backward()


                    #d_loss = d_loss_fake - d_loss_real + gradient_penalty
                    #Wasserstein_D = d_loss_real - d_loss_fake
                    self.d_optimizer.step()
                    
                    #print(f'  Discriminator iteration: {d_iter}/{self.critic_iter}, loss_fake: {d_loss_fake}, loss_real: {d_loss_real}')

                # Generator update
                for p in self.D.parameters():
                    p.requires_grad = False  # to avoid computation

                self.G.zero_grad()
                # train generator
                # compute loss with fake images
                z = self.get_torch_variable(torch.randn(self.batch_size, 100, 1, 1, 1))
                fake_images = self.G(z)
                g_loss = self.D(fake_images)
                g_loss = g_loss.mean()
                g_loss.backward(mone)
                
                '''
                # Loss G2: moduli loss
                z = self.get_torch_variable(torch.randn(self.batch_size, 100, 1, 1, 1))
                fake_images = self.G(z)
                m_loss_goal = torch.tensor(1.8, dtype=torch.float)
                m_loss = self.moduli_cal(fake_images)
                m_loss = m_loss.mean()
                m_loss.backward(m_loss_goal)
                
                # Loss G3: symm loss
                z = self.get_torch_variable(torch.randn(self.batch_size, 100, 1, 1, 1))
                fake_images = self.G(z)
                s_loss = self.moduli_cal(fake_images)
                s_loss = self.calculate_moduli_symm_loss(s_loss)
                s_loss.backward()
                '''
                
                #g_cost = -g_loss
                self.g_optimizer.step()
                g_iter_tqdm.set_postfix(OrderedDict(stage='train',epoch=epoch,loss=-g_loss.item()))
                # print(f'Generator iteration: {g_iter}/{self.generator_iters}, g_loss: {g_loss}')
                # Saving model and sampling images every 1000th generator iterations
            if epoch % 5 == 0:
                self.save_model(epoch)



        #self.t_end = t.time()
        #print('Time of training-{}'.format((self.t_end - self.t_begin)))
        #self.file.close()

        # Save the trained parameters
        self.save_model(self.epochs)

    def calculate_moduli_symm_loss(self,moduli):
        symm_moduli = torch.cat((moduli[:,1:],moduli[:,:1]),dim=1)
        return self.moduli_critic(moduli,symm_moduli)

    # TO BE MODIFIED
    def evaluate(self, test_loader, D_model_path, G_model_path):
        self.load_model(D_model_path, G_model_path)
        z = self.get_torch_variable(torch.randn(self.batch_size, 100, 1, 1, 1))
        samples = self.G(z)
        samples = samples.mul(0.5).add(0.5)
        samples = samples.data.cpu()
        grid = utils.make_grid(samples)
        print("Grid of 8x8 images saved to 'dgan_model_image.png'.")
        utils.save_image(grid, 'dgan_model_image.png')


    def calculate_gradient_penalty(self, real_images, fake_images):
        eta = torch.FloatTensor(self.batch_size,1,1,1,1).uniform_(0,1)
        eta = eta.expand(self.batch_size, real_images.size(1), real_images.size(2), real_images.size(3), real_images.size(4))
        if self.cuda:
            eta = eta.cuda()
        else:
            eta = eta

        interpolated = eta * real_images + ((1 - eta) * fake_images)

        if self.cuda:
            interpolated = interpolated.cuda()
        else:
            interpolated = interpolated

        # define it to calculate gradient
        interpolated = Variable(interpolated, requires_grad=True)

        # calculate probability of interpolated examples
        prob_interpolated = self.D(interpolated)

        # calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(
                                   prob_interpolated.size()).cuda() if self.cuda else torch.ones(
                                   prob_interpolated.size()),
                               create_graph=True, retain_graph=True)[0]

        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_term
        return grad_penalty

    def real_images(self, images, number_of_images):
        if (self.C == 3):
            return self.to_np(images.view(-1, self.C, 32, 32)[:self.number_of_images])
        else:
            return self.to_np(images.view(-1, 32, 32)[:self.number_of_images])

    def generate_img(self, z, number_of_images):
        samples = self.G(z).data.cpu().numpy()[:number_of_images]
        generated_images = []
        for sample in samples:
            if self.C == 3:
                generated_images.append(sample.reshape(self.C, 32, 32))
            else:
                generated_images.append(sample.reshape(32, 32))
        return generated_images

    def to_np(self, x):
        return x.data.cpu().numpy()

    def save_model(self,epoch):
        print('Save models...')
        path_G = self.G_path+'generator_lr_'+str(self.learning_rate)+'_epoch_'+str(epoch)+'.pt'
        path_D = self.D_path+'discriminator_lr_'+str(self.learning_rate)+'_epoch_'+str(epoch)+'.pt'
        torch.save(self.G.state_dict(), path_G)
        torch.save(self.D.state_dict(), path_D)
        print('Successfully saved...')

    def load_model(self, D_model_path, G_model_path):
        #D_model_path = os.path.join(os.getcwd(), D_model_filename)
        #G_model_path = os.path.join(os.getcwd(), G_model_filename)
        self.D.load_state_dict(torch.load(D_model_path))
        self.G.load_state_dict(torch.load(G_model_path))
        print('Generator model loaded from {}.'.format(G_model_path))
        print('Discriminator model loaded from {}-'.format(D_model_path))

    def get_infinite_batches(self, data_loader):
        while True:
            for i, (X, moduli) in enumerate(data_loader):
                yield X, moduli

    def generate_samples(self,num,path):
        if not os.path.exists(path):
            os.mkdir(path)
        for i in range(num):
            z = self.get_torch_variable(torch.randn(self.batch_size, 100, 1, 1, 1))
            sample = self.G(z)
            temp_path = path+'sample_'+str(i)+'.pt'
            print(sample)
            torch.save(sample,temp_path)

    def generate_latent_walk(self, number):
        if not os.path.exists('interpolated_images/'):
            os.makedirs('interpolated_images/')

        number_int = 10
        # interpolate between twe noise(z1, z2).
        z_intp = torch.FloatTensor(1, 100, 1, 1)
        z1 = torch.randn(1, 100, 1, 1)
        z2 = torch.randn(1, 100, 1, 1)
        if self.cuda:
            z_intp = z_intp.cuda()
            z1 = z1.cuda()
            z2 = z2.cuda()

        z_intp = Variable(z_intp)
        images = []
        alpha = 1.0 / float(number_int + 1)
        print(alpha)
        for i in range(1, number_int + 1):
            z_intp.data = z1*alpha + z2*(1.0 - alpha)
            alpha += alpha
            fake_im = self.G(z_intp)
            fake_im = fake_im.mul(0.5).add(0.5) #denormalize
            images.append(fake_im.view(self.C,32,32).data.cpu())

        grid = utils.make_grid(images, nrow=number_int )
        utils.save_image(grid, 'interpolated_images/interpolated_{}.png'.format(str(number).zfill(3)))
        print("Saved interpolated images.")
