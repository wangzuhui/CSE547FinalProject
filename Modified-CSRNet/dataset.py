import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from image import *
from torchvision import transforms
import torchvision.transforms.functional as F
import PIL

class listDataset(Dataset):
    def __init__(self, root, shape=None, shuffle=True, transform_list=None,  train=False, seen=0, batch_size=1, num_workers=4):
        if train:
            root = root *4
        random.shuffle(root)
        
        self.nSamples = len(root)
        self.lines = root
        self.transform_list = transform_list
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def __len__(self):
        return self.nSamples
    def __getitem__(self, index):
        assert index <= len(self), 'index range error' 
        
        img_path = self.lines[index]
        
        img,target = load_data(img_path,self.train)
        
        i_w, i_h = img.size
        #print('Dataset: img size before change {}'.format((i_w, i_h)))
        
        ''' no use
        if i_w % 8 != 0:
            i_w = i_w + (8 - (i_w % 8))
        
        if i_h % 8 != 0:
            i_h = i_h + (8 - (i_h % 8))
        '''
        #self.transform_list.insert(0, transforms.Resize((i_w,i_h), PIL.Image.BICUBIC))
        #print('Dataset: img size after change {}'.format((i_w, i_h)))
        
        #img = 255.0 * F.to_tensor(img)
        #img[0,:,:]=img[0,:,:]-92.8207477031
        #img[1,:,:]=img[1,:,:]-95.2757037428
        #img[2,:,:]=img[2,:,:]-104.877445883
       
        transform = transforms.Compose(self.transform_list)
        if transform is not None:
            img = transform(img)
            #print('Dataset: img size after change-transform {}'.format((img.shape)))
        return img,target