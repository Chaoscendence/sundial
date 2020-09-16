import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from sundial.neural_models.encoder_decoder import UNet
from sundial.neural_models.encoder_decoder import HourGlassNet
from sundial.utils import save_checkpoint
from sundial.utils import smooth
import matplotlib.pyplot as plt
from sundial.neural_models.loss import DprtMseLoss
from meta_settings import *

DEVICE = torch.device("cuda")
NUM_CHANNELS = 10


def params_contraints(nkz_params):
    output = torch.clone(nkz_params)
    phi_arr = nkz_params[:,4:5,:]    
    d_arr = nkz_params[:,5:6,:]
    phi_arr = torch.mean(phi_arr, dim=2, keepdim=True)
    d_arr = torch.mean(d_arr, dim=2, keepdim=True)    
    output[:,4:5,:] = phi_arr
    output[:,5:6,:] = d_arr
    return output

class InverseNetwork(nn.Module):
    """
    Example of settings
    inverse_settings = {
        'ConvLayers': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M',
                    256, 256, 'U', 128, 128, 'U', 64, 64, 'U'],
        'InChannels': 2,
        'OutChannels': NUM_CHANNELS,
    }
    """
    def __init__(self, settings=None):
        super(InverseNetwork, self).__init__()
        assert (settings is not None), "settings is not defined."
        
        # Embedding network
        self.embedding_network = HourGlassNet(settings=settings)
        self.constraint = nn.ReLU(inplace=True)
        self.loss = nn.MSELoss() 
        self.initialize_weights()

    def initialize_weights(self):
        for index, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv1d):
                init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)            
                m.bias.data.zero_()

    def forward(self, input):
        x_t = self.embedding_network(input)
        x_th = torch.abs(x_t[:,1:3,:])
        out = torch.cat((x_t[:,0:1,:], x_th, x_t[:,3:,:]), dim=1)        
        return out

    def postprocessing(self, input):
        return params_contraints(input)


class ForwardNetwork(nn.Module):
    """
    Example of settings
    inverse_settings = {
        'ConvLayers': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M',
                    256, 256, 'U', 128, 128, 'U', 64, 64, 'U'],
        'InChannels': NUM_CHANNELS,
        'OutChannels': 2
    }
    """
    def __init__(self, settings=None):
        super(ForwardNetwork, self).__init__()
        assert (settings is not None), "settings is not defined."
        
        # Embedding network
        self.embedding_network = HourGlassNet(settings=settings)
        self.loss = nn.MSELoss() 
        self.initialize_weights()

    def initialize_weights(self):
        for index, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv1d):
                init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)            
                m.bias.data.zero_()

    def forward(self, input):
        out = self.embedding_network(input)
        return out

    def preprocessing(self, input):
        return params_contraints(input)


class ForwardBag(nn.Module):
    def __init__(self, dp_forward, rt_forward):
        super(ForwardBag, self).__init__()
        self.dp_forward = dp_forward
        self.rt_forward = rt_forward

    @property
    def dp_forward_network(self):
        return self.dp_forward

    @property
    def rt_forward_network(self):
        return self.rt_forward

    def forward(self, input):
        dp = self.dp_forward(input)
        rt = self.rt_forward(input)
        output = torch.cat([dp, rt], dim=1)
        return output
        

class ForwardNetBag(nn.Module):
    def __init__(self, forward_nets):
        super(ForwardNetBag, self).__init__()
        # We assume forward_nets is a python list
        self.forward_nets = nn.ModuleList(forward_nets)

    @property
    def forward_networks(self):
        return self.forward_nets

    def forward(self, input):
        output = []    
        for net_i in self.forward_nets:
            output_i = net_i(input)
            output.append(output_i)
        output = torch.cat(output, dim=1)
        return output


class InverseNetBag(nn.Module):
    def __init__(self, inverse_nets):
        super(InverseNetBag, self).__init__()
        self.inverse_nets = nn.ModuleList(inverse_nets)

    @property
    def inverse_networks(self):
        return self.inverse_nets

    def forward(self, input):
        output = 0
        for k, net in enumerate(self.inverse_nets):
            out_k = net(input[:,k*2:(k+1)*2,:])
            output += out_k
        output /= len(self.inverse_nets)
        return output


#-------------------------------------------------------------------------------
class MIMONet(nn.Module):
    """ Multiple inverse networks and multiple forward networks.
    """
    def __init__(self):
        super(MIMONet, self).__init__()

        # Sub models
        self._subnets = {'inverse': None, 'forward': None}
        self.loss = DprtMseLoss()

    def freeze(self, network=None, frozen=True):
        assert (network in self._subnets.keys())
        net = self._subnets[network]
        for param in net.parameters():
            param.requires_grad = not frozen

    def initialize(self, inverse_network, forward_network):
        self._subnets['inverse'] = inverse_network
        self._subnets['forward'] = forward_network

    def forward(self, input):
        nkz = self.inverse_network(input)
        output = self.forward_network(nkz)
        return output

    @property
    def inverse_network(self):
        return self._subnets['inverse']

    @property
    def forward_network(self):
        return self._subnets['forward']



