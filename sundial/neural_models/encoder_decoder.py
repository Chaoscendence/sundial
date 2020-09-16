import math
import numpy as np
import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, settings):
        super(UNet, self).__init__()
        self.settings = settings
        self.layers = []
        self.head = []
        self._build_models()

    def forward(self, x):
        count = 0
        pool_indexes = []
        add_joints = []
        y = x
        for i, elem in enumerate(self.settings['ConvLayers']):
            if elem == 'M':
                add_joints.append(y)
                y, pi = self.layers[count](y)
                pool_indexes.append(pi)                
            elif elem == 'U':
                y = self.layers[count](y, pool_indexes[-1])
                pool_indexes.pop()
                y += add_joints[-1]
                # y = torch.cat((y, add_joints[-1]), dim=1)
                add_joints.pop()
            else:
                y = self.layers[count](y) #Conv
                count += 1
                y = self.layers[count](y) #ReLU
            count += 1
        # Run the last layer
        y = self.head(y)
        return y

    def _build_models(self):
        layers = []
        in_channels = self.settings['InChannels']
        out_channels = self.settings['OutChannels']
        add_joints = []
        self.kernel_size = 3
        for x in self.settings['ConvLayers']:
            if x == 'M':
                layers.append(nn.MaxPool1d(kernel_size=2, stride=2, 
                                           return_indices=True))
                # self.kernel_size = int(math.floor(self.kernel_size/2.0))
                # self.kernel_size = np.max([self.kernel_size, 1])
                add_joints.append(in_channels)
            elif x == 'U':
                layers.append(nn.MaxUnpool1d(kernel_size=2, stride=2))
                # self.kernel_size = int(math.ceil(self.kernel_size*2.0)+1)
                # in_channels += add_joints[-1]
                add_joints.pop()
            else:
                layers.append(nn.Conv1d(in_channels, x, 
                                kernel_size=self.kernel_size, stride=1,
                                padding=math.floor(self.kernel_size*0.5)))
                # layers.append(nn.BatchNorm1d(x))
                # layers.append(nn.Dropout(0.2))
                layers.append(nn.ReLU(inplace=True))
                # layers.append(nn.LeakyReLU(0.1))
                in_channels = x
    
        # The last layer
        head = []
        head.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1))
        # head.append(nn.LeakyReLU(0.1))
        # head.append(nn.Linear(out_channels*256, out_channels*256))
        head.append(nn.PReLU(out_channels))
        # head.append(nn.AvgPool1d(kernel_size=2, stride=2, padding=0))
        self.head = nn.Sequential(*head)
        self.layers = nn.Sequential(*layers)


class HourGlassNet(nn.Module):
    def __init__(self, settings):
        super(HourGlassNet, self).__init__()
        self.settings = settings
        self.num_blocks = 3
        self.block1 = self._build_layers(in_channels=settings['InChannels'])
        self.blocks = nn.ModuleList([self.block1])
        in_channels = 64
        for i in range(self.num_blocks):
            self.blocks.append(self._build_layers(in_channels=in_channels))
            in_channels += 64
        # self.head = nn.Sequential(
        #     nn.Conv1d(64, settings['OutChannels'], kernel_size=3, stride=1, 
        #               padding=1),
        #     nn.LeakyReLU(0.5)
        #     # nn.PReLU(settings['OutChannels'])
        #     )
        self.head = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=1, 
                      stride=1),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(0.5),
            # nn.PReLU(settings['OutChannels'])
            nn.Conv1d(16, settings['OutChannels'], kernel_size=1, stride=1)
            )

    def block_forward(self, block, x):
        count = 0
        pool_indexes = []
        add_joints = []
        y = x
        for i, elem in enumerate(self.settings['ConvLayers']):
            if elem == 'M':
                add_joints.append(y)
                y, pi = block[count](y)
                pool_indexes.append(pi)                
            elif elem == 'U':
                y = block[count](y, pool_indexes[-1])
                pool_indexes.pop()
                y += add_joints[-1]
                # y = torch.cat((y, add_joints[-1]), dim=1)
                add_joints.pop()
            else:
                y = block[count](y) #Conv
                count += 1
                y = block[count](y) #ReLU                
            count += 1
        return y

    def forward(self, x):
        x_i = self.block_forward(self.blocks[0], x)
        for i in range(self.num_blocks):
            y_i = self.block_forward(self.blocks[i+1], x_i)
            x_i = torch.cat((y_i, x_i), dim=1)
        out = self.head(x_i)
        return out

    def _build_layers(self, in_channels):
        layers = []
        add_joints = []
        self.kernel_size = 3
        for x in self.settings['ConvLayers']:
            if x == 'M':
                layers.append(nn.MaxPool1d(kernel_size=2, stride=2, 
                                           return_indices=True))
                # self.kernel_size = int(math.floor(self.kernel_size/2.0))
                # self.kernel_size = np.max([self.kernel_size, 1])
                add_joints.append(in_channels)
            elif x == 'U':
                layers.append(nn.MaxUnpool1d(kernel_size=2, stride=2))
                # self.kernel_size = int(math.ceil(self.kernel_size*2.0)+1)
                # in_channels += add_joints[-1]
                add_joints.pop()
            else:
                layers.append(nn.Conv1d(in_channels, x, 
                                    kernel_size=self.kernel_size, stride=1,
                                    padding=math.floor(self.kernel_size*0.5)))
                # layers += [nn.BatchNorm1d(x)]
                # layers += [nn.Dropout(0.2)]
                layers.append(nn.ReLU(inplace=True))
                in_channels = x
        return nn.Sequential(*layers)
