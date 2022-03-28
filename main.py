# basic 
import os
import numpy as np 

# for MNIST download 
import torch
import torch.nn as nn 
from torchvision import datasets, transforms

# fro visualize
import matplotlib.pyplot as plt 
import seaborn as sns 

#--------------------------------
#        System variables
#--------------------------------

# find the current path
path = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(f'{path}/figures'):
    os.mkdir(f'{path}/figures')

# define some color 
Blue    = .85 * np.array([   9, 132, 227]) / 255
Green   = .85 * np.array([   0, 184, 148]) / 255
Red     = .85 * np.array([ 255, 118, 117]) / 255
Yellow  = .85 * np.array([ 253, 203, 110]) / 255
Purple  = .85 * np.array([ 108,  92, 231]) / 255
colors  = [ Blue, Red, Green, Yellow, Purple]
sfz, mfz, lfz = 11, 13, 15
dpi     = 250
sns.set_style("whitegrid", {'axes.grid' : False})

#---------------------------------
#        MNIST Dataloader 
#---------------------------------

def get_MNIST( batchSize=128):
    train_data = datasets.MNIST(f'{path}/../data', train=True, download=True,
                            transform=transforms.Compose([ transforms.ToTensor()]))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batchSize,
                                        shuffle=True)

    test_data = datasets.MNIST(f'{path}/../data', train=False, download=True,
                            transform=transforms.Compose([ transforms.ToTensor()]))
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batchSize, 
                                        shuffle=False)
    return train_loader, test_loader

#---------------------------------------------
#      Variational Inforamtion bottleneck 
#---------------------------------------------

def _initLayer( layer, mode='xavier', bias=True):
    if mode == 'constant':
        nn.init.constant_( layer.weight, 0)
    if mode == 'gauss':
        nn.init.normal_( layer.weight, mean=0.0, std=.2)
    elif mode == 'xavier':
        nn.init.xavier_uniform_( layer.weight)
    if bias:
        nn.init.constant_( layer.bias, 0)
    return layer

class VIB( nn.Module):

    def __init__( self, z_dim=128, n_latent=12, gpu=True):
        super().__init__()
        # choose device 
        if gpu and torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'
        self.fc1 = _initLayer(nn.Linear(784, 1024))
        self.fc2 = _initLayer(nn.Linear(1024,1024))
        self.enc_mean = _initLayer(nn.Linear(1024, z_dim))
        self.enc_std  = _initLayer(nn.Linear(1024, z_dim))
        self.dec = _initLayer(nn.Linear(z_dim, 10))
        self.n_latent = n_latent
    
    def forward( self, x):
        x = 2*x - 1 
        x = torch.relu( self.fc2( torch.relu( self.fc1( x))))
        mu, logsig = self.enc_mean(x), self.enc_std( x)
        z_dim = mu.unsqueeze(1) + \
                torch.randn( (tuple( logsig.shape)[0],)+( self.n_latent,
                )+tuple( mu.shape)[1:]).to( self.device) + \
                logsig.unsqueeze(1)

if __name__ == '__main__':


    train_loader, test_loader = get_MNIST()