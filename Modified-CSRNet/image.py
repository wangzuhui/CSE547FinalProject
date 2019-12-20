import random
import scipy.io as io
import os
from PIL import Image,ImageFilter,ImageDraw, ImageChops
import numpy as np
import h5py
from PIL import ImageStat
import cv2


def load_data(img_path,train = True):
    '''
    # preperation:
    root = '/usr/local/home/zw5t8/CrowdCounting/datasets/UCSD/'
    roi_file = root + 'vidf1_33_roi_mainwalkway.mat'
    pers_file = root + 'vidf1_33_dmap3.mat'
    
    roi_mat = io.loadmat(roi_file)
    roi_pos = roi_mat['roi']['mask'][0][0]
    
    pers_mat = io.loadmat(pers_file)
    pers_pos = pers_mat['dmap']['pmapxy'][0][0]

    mask = np.multiply(roi_pos, pers_pos)
    '''
    gt_path = img_path.replace('.jpg','.h5').replace('images','ground_truth') #UCSD: png
    #print(img_path)
    #print(gt_path)
    img_ary = np.array(Image.open(img_path))
    #print('{} and {}'.format(type(img_ary), img_path.split('/')[-1]))
    #img_aryt = np.multiply(img_ary, mask)
    #img = Image.fromarray(img_aryt).convert('RGB')
    img = Image.fromarray(img_ary).convert('RGB')
    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file['density'])
    
    
    if False:
        #x_c = 128 # int(img.size[0]/2)
        #y_c = 128 # int(img.size[1]/2)
        crop_size = (img.size[0]/2,img.size[1]/2)
        if random.randint(0,9)<= -1:
            dx = int(random.randint(0,1)*img.size[0]*1./100)
            dy = int(random.randint(0,1)*img.size[1]*1./100)
        else:
            dx = int(random.random()*img.size[0]*1./100) # random.random() \in [0,1]
            dy = int(random.random()*img.size[1]*1./100)
        img = img.crop((dx,dy,crop_size[0]+dx,crop_size[1]+dy))
        target = target[dy:crop_size[1]+dy,dx:crop_size[0]+dx]
       
        if random.random()>0.8:
            target = np.fliplr(target)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
    
    
    t_w, t_h = target.shape
    #i_w, i_h = img.size
    
   # print('BEFORE image size should divide by 8 {}'.format((t_w, t_h)))
    
    #if target.shape != img.size:
    #    print('***ERROR: Input and Target have different resolutions!\n')
    
    if t_w % 8 != 0: 
        t_w = t_w + (8 - (t_w % 8))
    if t_h % 8 != 0:
        t_h = t_h + (8 - (t_h % 8))
    
    size = (t_w, t_h)
    #print('current size should divide by 8 {}'.format((t_w, t_h)))
    #print('before')
    img = img.resize(size)
    #print('after')
    target = cv2.resize(target, (int(t_w/8.0), int(t_h/8.0)), interpolation = cv2.INTER_CUBIC)*64
    # For CSRNet-Deconv
    #target = cv2.resize(target, (int(t_w/4.0), int(t_h/4.0)), interpolation = cv2.INTER_CUBIC)*64
    #target = cv2.resize(target, (int(t_w), int(t_h)), interpolation = cv2.INTER_CUBIC)
    #target = cv2.resize(target,(int(target.shape[1]/8),int(target.shape[0]/8)),interpolation = cv2.INTER_CUBIC)*64 # resolution 1/8, element value times 64.
    #target = cv2.resize(target,(int(target.shape[1]/4),int(target.shape[0]/4)),interpolation = cv2.INTER_CUBIC)*16 # resolution 1/8, element value times 64.
    # https://github.com/leeyeehoo/CSRNet-pytorch/issues/29
    # Because 8 * 8 = 64. Actually, resize + *64 should be a simplified method of the convolution that sums the 8x8 grid, which I did in my own implementation.
    
    return img,target