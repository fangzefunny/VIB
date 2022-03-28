# basic 
import os
import numpy as np 

# for MNIST download 
import torch
import torch.nn as nn 
from torch.optim import Adam
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

ce = nn.CrossEntropyLoss()

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
        z = mu.unsqueeze(1) + \
                torch.randn( (tuple( logsig.shape)[0],)+( self.n_latent,
                )+tuple( mu.shape)[1:]).to( self.device) + \
                logsig.exp().unsqueeze(1)
        p_Y1Z = torch.softmax( self.dec(z), dim=2)
        p_Y1X = p_Y1Z.mean(dim=1)
        return p_Y1Z, p_Y1X, mu, logsig 

    def get_loss( self, p_Y1Z, Y, mu, logsig, beta=1e-2):
        '''Calculate the VIB object
        '''
        # reshape the target label Y to match dims (n_batch, n_samples)
        batchSize, n_samples = p_Y1Z.shape[0], p_Y1Z.shape[1]
        Y = Y.view(-1,1) * torch.ones( batchSize, n_samples, dtype=torch.long).to(self.device)
        err = ce( p_Y1Z.view( -1, 10), Y.view( -1))
        comp = .5 * (mu**2 + logsig.exp().pow() - 2*logsig - 1).sum(dim=1).mean()
        return err + beta * comp 

#--------------------------
#      Train the VIB 
#--------------------------

def trainVIB( z_dim=128, lr=1e-4, beta=1e-2, 
                batchSize=128, MaxEpoch=10, logfreq=1):

    # get train and test data 
    train_data, test_data = get_MNIST(batchSize)

    # get model
    model = VIB( z_dim=z_dim)

    # get optimizer
    optimizer = Adam( model.parameters, lr=lr, betas=(.5,.999))

    for epi in range(MaxEpoch):

        epoch_loss = 0
        for i, batch in enumerate(train_data):
            x_batch, y_batch = batch
            x_batch, y_batch = x_batch.to(model.device), y_batch.to(model.device)
            # forward 
            p_Y1Z, p_Y1X, mu, logsig = model.forward( x_batch)
            loss = model.get_loss( p_Y1Z, p_Y1X, mu, logsig, beta=beta)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (i+1) % logfreq ==0:
            print( f'Epoch: {epi}, Loss: {epoch_loss:.3f}')



if __name__ == '__main__':


    train_loader, test_loader = get_MNIST()
