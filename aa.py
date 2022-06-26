import numpy as np
import pandas as pd
import pathlib, sys, os, random, time
import numba, cv2, gc
#from tqdm import tqdm_notebook
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.ion()
#get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold
import albumentations as A
import segmentation_models_pytorch as smp
import rasterio
from rasterio.windows import Window

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as D

import torchvision
from torchvision import transforms as T

from hausdorff import HausdorffDTLoss
from lovasz_loss import LovaszSoftmax
def rle_encode(im):
    '''
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = im.flatten(order = 'F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle_decode(mask_rle, shape=(512, 512)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')
subm=pd.read_csv('./subtt_b_2.csv', sep='\t', names=['name', 'mask'])
subm['name'] = subm['name'].apply(lambda x: './data/test_a/' + x)
plt.figure(figsize=(16,8))
plt.subplot(121)
plt.imshow(rle_decode(subm['mask'].fillna('').iloc[0]), cmap='gray')
plt.subplot(122)
plt.imshow(cv2.imread(subm['name'].iloc[0]))
plt.subplot(122)