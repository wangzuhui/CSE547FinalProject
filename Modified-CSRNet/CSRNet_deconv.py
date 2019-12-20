#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSRNet with Dense Connection 
                                     
Perspective Fusion Networks

Created on Thu Nov 21 12:20:03 2019

@author: UM-AD\zw5t8
"""

import torch.nn as nn
import torch
from sknet import SKUnit
from densenet import DenseBlock, BasicBlock
#from DenseNet import _DenseLayer, _DenseBlock
from torchvision import models
from utils import save_net,load_net
import collections
from torchsummary import summary

class PFNet(nn.Module):
    def __init__(self, load_weights=False):
        super(PFNet, self).__init__()
        self.seen = 0

        # CRSNet-Dense-dilation
        self.frontend_feat = [64, 64, 'M']
        self.dense_feat1 = self.Dense_make_layer(nb_layers=6, in_planes=64, growth_rate=12, block=BasicBlock)
        self.after_dense1 = [128, 'M']
        self.dense_feat2 = self.Dense_make_layer(nb_layers=6, in_planes=128, growth_rate=12, block=BasicBlock)
        self.after_dense2 = [512, 'M']
        self.afdense1 = make_layers(self.after_dense1, in_channels = 136, batch_norm=False) #in_channels=64
        self.afdense2 = make_layers(self.after_dense2, in_channels = 200, batch_norm=False) #in_channels=128
        #self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat  = [512, 512, 512, 256, 128, 64]    #original [512, 512, 512, 256, 128, 64]
        #self.tail_feat = ['U', 64]#, 'U', 128, 'U', 64]
        self.frontend = make_layers(self.frontend_feat, in_channels = 3, batch_norm=False)
        self.backend = make_layers(self.backend_feat,in_channels = 512,dilation = True)
        #self.tail = make_layers(self.tail_feat, in_channels = 128, batch_norm=False)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        self._initialize_weights()
        
        '''
        if not load_weights:
            mod = models.vgg16(pretrained = True)
            self._initialize_weights()
            #fsd=collections.OrderedDict()
            for i in range(len(self.frontend.state_dict().items())): 
                list(self.frontend.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i][1].data[:]
            # wzh:
            #for param in self.frontend.parameters():
            #    param.requires_grad = False
        '''
        
        
        '''
        # CRSNet original
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat  = [512, 512, 512,256,128,64]
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat,in_channels = 512,dilation = True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        if not load_weights:
            mod = models.vgg16(pretrained = True)
            self._initialize_weights()
            #fsd=collections.OrderedDict()
            for i in range(len(self.frontend.state_dict().items())): #10个卷积*（weight，bias）=20个参数
                list(self.frontend.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i][1].data[:]
        '''
        
        # Encoder-Decoder
        '''
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat  = [512, 512] #[512, 512, 512, 512, 512] for CSRNet
        self.tail_feat = ['U', 256, 'U', 128, 'U', 64]
        self.frontend = make_layers(self.frontend_feat, batch_norm=True)
        self.backend = make_layers(self.backend_feat,in_channels = 512, batch_norm=False, dilation = True)
        self.tail = make_layers(self.tail_feat, in_channels = 512, batch_norm=False)
        '''
        
        # FishLike: Econder-Decoder-FineCreate
        '''
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat  = [512, 512] #[512, 512, 512, 512, 512] for CSRNet
        self.tail_feat = ['U', 256, 'U', 128, 'U', 64]
        self.fefi_feat = [64, 'M', 64, 'M', 64]
        self.frontend = make_layers(self.frontend_feat, batch_norm=True)
        self.backend = make_layers(self.backend_feat,in_channels = 512, batch_norm=False, dilation = True)
        self.tail = make_layers(self.tail_feat, in_channels = 512, batch_norm=False)
        self.fefi = make_layers(self.fefi_feat, in_channels = 64, batch_norm=False)
        '''
        
        # Ecoder-Feature Selection-Decoder-FineCreate
        '''
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.tail_feat = ['U', 256, 'U', 128, 'U', 64]
        self.fefi_feat = [64, 'M', 64, 'M', 64]
        self.frontend = make_layers(self.frontend_feat, batch_norm=True)
        self.middle = self.SK_make_layer(512, 256, 512, nums_block=1)
        self.tail = make_layers(self.tail_feat, in_channels = 512, batch_norm=False)
        self.fefi = make_layers(self.fefi_feat, in_channels = 64, batch_norm=False)
        '''
        
        '''
        if not load_weights:
            mod = models.vgg16(pretrained = True)
            self._initialize_weights()
            #fsd=collections.OrderedDict()
            for i in range(len(self.frontend.state_dict().items())): #10个卷积*（weight，bias）=20个参数
                list(self.frontend.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i][1].data[:]
                #python 2.7: self.frontend.state_dict().items()[i][1].data[:] = mod.state_dict().items()[i][1].data[:]
                #temp_key=list(self.frontend.state_dict().items())[i][0]
                #fsd[temp_key]=list(mod.state_dict().items())[i][1]
            #self.frontend.load_state_dict(fsd)
            #print("Mine",list(self.frontend.state_dict().items())[0][1])
        '''
        
        '''
        # Encoder(DenseBlocks)-Feature Selection-Decoder-FinceCreate
        self.frontend_feat = [64, 'M']
        self.dense_feat1 = self.Dense_make_layer(nb_layers=3, in_planes=64, growth_rate=12, block=BasicBlock)
        self.after_dense1 = [128, 'M']
        self.dense_feat2 = self.Dense_make_layer(nb_layers=3, in_planes=128, growth_rate=12, block=BasicBlock)
        self.after_dense2 = [256, 'M']
        self.tail_feat = ['U', 256, 'U', 128, 'U', 64]
        self.fefi_feat = [64, 'M', 64, 'M', 64]
        self.frontend = make_layers(self.frontend_feat, batch_norm=True)
        self.afdense1 = make_layers(self.after_dense1, in_channels = 100, batch_norm=False) #in_channels=64
        self.afdense2 = make_layers(self.after_dense2, in_channels = 164, batch_norm=False) #in_channels=128
        self.middle = self.SK_make_layer(256, 128, 256, nums_block=1)
        self.tail = make_layers(self.tail_feat, in_channels = 256, batch_norm=False)
        self.fefi = make_layers(self.fefi_feat, in_channels = 64, batch_norm=False)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        self._initialize_weights()
        '''

    def forward(self,x):
        x = self.frontend(x)
        x = self.dense_feat1(x)
        x = self.afdense1(x)
        x = self.dense_feat2(x)
        x = self.afdense2(x)
        x = self.backend(x)
        #x = self.middle(x)
        #x = self.tail(x)
        #x = self.fefi(x)
        x = self.output_layer(x)
        return x
    
    '''version-1 does not work
    def Dense_make_layer(self, block_config, num_layers, num_features, bn_size, growth_rate, drop_rate, memory_efficient=False):
        layers = [_DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )]
        for _ in range(1, block_config):
            layers.append(_DenseBlock(num_layers, num_features, bn_size, growth_rate, drop_rate))
        return nn.Sequential(*layers)
    '''
    
    def Dense_make_layer(self, block, in_planes, growth_rate, nb_layers, dropRate=0.0):
        layers = []
        for i in range(nb_layers):
            layers.append(block(in_planes+i*growth_rate, growth_rate, dropRate))
        return nn.Sequential(*layers)
    
    # SKConv
    def SK_make_layer(self, in_feats, mid_feats, out_feats, nums_block, stride=1):
        layers=[SKUnit(in_feats, mid_feats, out_feats, M=3, stride=stride)]
        for _ in range(1, nums_block):
            layers.append(SKUnit(out_feats, mid_feats, out_feats))
        return nn.Sequential(*layers)
    
       
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
                
def make_layers(cfg, in_channels = 3,batch_norm=False,dilation = False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'U':
            #layers += [nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)]
            upsample = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation = d_rate)
            layers += [upsample, conv2d, nn.ReLU(inplace=True)]
                        
            #layers += [nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)] #also ok
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


if __name__ == '__main__':
    #in_data = torch.randint(0, 255, (1, 3, 224, 224), dtype=torch.float32)
    net = PFNet()
    summary(net, input_size=(3, 480, 480), device='cpu')






                
