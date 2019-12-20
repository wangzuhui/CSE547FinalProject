import random
import os
from PIL import Image,ImageFilter,ImageDraw
import numpy as np
import h5py
from PIL import ImageStat
import cv2

def load_data(img_path,train = True):
    gt_path = img_path.replace('.jpg','.h5').replace('images','ground_truth')
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file['density'])
    
    if False:
        #x_c = 128 # int(img.size[0]/2)
        #y_c = 128 # int(img.size[1]/2)
        crop_size = (img.size[0]/2,img.size[1]/2)
        if random.randint(0,9)<= -1:
            #dx = int(random.randint(0,1)*img.size[0]*1./100)
            #dy = int(random.randint(0,1)*img.size[1]*1./100)
            dx = 0
            dy = 0
        else:
            #dx = int(random.random()*img.size[0]*1./100) # random.random() \in [0,1]
            #dy = int(random.random()*img.size[1]*1./100)
            dx = 0
            dy = 0
        img = img.crop((dx,dy,crop_size[0]+dx,crop_size[1]+dy))
        target = target[dy:crop_size[1]+dy,dx:crop_size[0]+dx]
       
        if random.random()>0.8:
            target = np.fliplr(target)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
    
    #target = cv2.resize(target,(28,28),interpolation = cv2.INTER_CUBIC)*64   # For fishcount 224/28 = 8, 8*8=64
    target = cv2.resize(target,(int(target.shape[1]/8),int(target.shape[0]/8)),interpolation = cv2.INTER_CUBIC)*64 # resolution 1/8, element value times 64.
    #target = cv2.resize(target,(int(target.shape[1]/4),int(target.shape[0]/4)),interpolation = cv2.INTER_CUBIC)*16 # resolution 1/8, element value times 64.
    # https://github.com/leeyeehoo/CSRNet-pytorch/issues/29
    # Because 8 * 8 = 64. Actually, resize + *64 should be a simplified method of the convolution that sums the 8x8 grid, which I did in my own implementation.
    
    return img,target
