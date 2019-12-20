#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test the model

Created on Thu Nov 21 10:40:53 2019

@author: UM-AD\zw5t8
"""

import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter 
import scipy
import json
import torchvision.transforms.functional as F
from matplotlib import cm as CM
from image import *
#from PFNet_EcDc import PFNet
from model import CSRNet
import torch
from torchvision import transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import cv2


# % matplotlib inline # for jupyter notebook only

from torchvision import datasets, transforms
transform=transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225]),])

root = '/home/zuwang/CrowdCounting/Datasets/ShanghaiTech/'

#now generate the ShanghaiA's ground truth
part_A_train = os.path.join(root,'part_A_final/train_data','images')
part_A_test = os.path.join(root,'part_A_final/test_data','images')
part_B_train = os.path.join(root,'part_B_final/train_data','images')
part_B_test = os.path.join(root,'part_B_final/test_data','images')

path_sets = [part_B_test]

img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)

model = CSRNet()
#model = PFNet()
#model = fishnet99()
model = model.cuda()

#checkpoint = torch.load('FishCount99_B_model_best_new.pth.tar') #B_model_best.pth.tar
# CRSNet:Crsnet_cuibic_A_model_best.pth.tar
#checkpoint = torch.load('CSRNet_B_model_best.pth.tar')
checkpoint = torch.load('./cse547_csrnet_B_model_best.pth.tar')


model.load_state_dict(checkpoint['state_dict'])

loader = transforms.Compose([transforms.ToTensor()])  

unloader = transforms.ToPILImage()

def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

fd_name = './generate_density_maps'
if not os.path.isdir(fd_name):
    os.mkdir('generate_density_maps')
    
mae = 0
mse = 0.0

cmap = plt.cm.jet

for i in range(len(img_paths)):
    img = 255.0 * F.to_tensor(Image.open(img_paths[i]).convert('RGB'))

    img[0,:,:]=img[0,:,:]-92.8207477031
    img[1,:,:]=img[1,:,:]-95.2757037428
    img[2,:,:]=img[2,:,:]-104.877445883
    
    file_path = img_paths[i]
    file_name = file_path.split('/')[-1].split('.')[0]
    image_name = os.path.join(fd_name, file_name + '_dmap.png')
    #print(file_name)
    
    img = img.cuda()
    img = transform(Image.open(img_paths[i]).convert('RGB')).cuda()
    gt_file = h5py.File(img_paths[i].replace('.jpg','.h5').replace('images','ground_truth'),'r')
    groundtruth = np.asarray(gt_file['density'])
    
    output = model(img.unsqueeze(0))
    imshow(output)
    
    image = output.cpu().clone()
    image = image.squeeze(0)
    img = transforms.ToPILImage()(image)
    
    plt.imsave(image_name, img, cmap=cmap)
    mae += abs(output.detach().cpu().sum().numpy()-np.sum(groundtruth))
    mse += ((output.detach().cpu().sum().numpy()-np.sum(groundtruth))*(output.detach().cpu().sum().numpy()-np.sum(groundtruth))) 
    
    with open('results_mae_mst_A.txt', 'a+') as f:
        f.write('the image name: {} with mae: {} and mse: {}\n'.format(image_name, abs(output.detach().cpu().sum().numpy()-np.sum(groundtruth)), ((output.detach().cpu().sum().numpy()-np.sum(groundtruth))*(output.detach().cpu().sum().numpy()-np.sum(groundtruth))) ))
        f.write('the ground_truth number: {} and estimated number is: {}\n'.format(np.sum(groundtruth), output.detach().cpu().sum().numpy()))
        f.write('\n')
    print (i,mae, mse)
print ('Average MAE: ', mae/len(img_paths))
print ('Average MSE: ', np.sqrt(mse/len(img_paths)))
