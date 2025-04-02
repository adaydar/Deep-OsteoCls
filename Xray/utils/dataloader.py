import os
import shutil
import tempfile
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
from sklearn.metrics import classification_report
import torch
#!pip install monai
from monai.apps import download_and_extract
from monai.config import print_config
from monai.metrics import ROCAUCMetric
from monai.networks.nets import DenseNet121
from monai.transforms import *
from monai.data import Dataset, DataLoader
from monai.utils import set_determinism
from tqdm import tqdm
import pandas as pd
from torchvision import datasets,transforms,models
import sys
import torchvision.transforms as transforms
from utils.config import *

#pixel_mean, pixel_std = 0, 1
sharpen_filter=np.array([[-1,-1,-1],
                 [-1,9,-1],
                [-1,-1,-1]])

class MyResize(Transform):
    def __init__(self,size=(224,224)):
        self.size = size
    def __call__(self,inputs):
        sharp_image=cv2.filter2D(np.array(inputs),-1,sharpen_filter)
        image=cv2.resize(sharp_image,dsize=(self.size[1],self.size[0]),interpolation=cv2.INTER_CUBIC)        
        #image2=image[25:475,25:475]
        #smooth = cv2.GaussianBlur(image2,(3,3),0)
        return image #smooth
        
class FFT(Transform):
    def __init__(self, size=(224,224)):
         self.size = size
    def __call__(self, inputs):
         #image = cv2.imread(np.array(inputs))
         #gray = cv2.cvtColor(np.array(inputs), cv2.COLOR_BGR2GRAY)   
         gray = np.array(inputs)
         sharp_image=cv2.filter2D(gray,-1,sharpen_filter)
         f = np.fft.fft2(sharp_image)
         fshift = np.fft.fftshift(f)
         rows, cols = gray.shape
         crow, ccol = rows // 2, cols // 2
         mask = np.ones((rows, cols), np.uint8)
         r = 25  # Radius of the high-pass filter
         mask[crow - r:crow + r, ccol - r:ccol + r] = 0
         fshift_filtered = fshift * mask
         f_ishift = np.fft.ifftshift(fshift_filtered)
         filtered_image = np.fft.ifft2(f_ishift)
         filtered_image = np.abs(filtered_image)
         return filtered_image
         
    
#Image Transformation
from monai.transforms import HistogramNormalize
import torchvision
pixel_mean, pixel_std = 0.66133188, 0.21229856
train_transforms= transforms.Compose([
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
            transforms.RandomHorizontalFlip(p=0.5),
            MyResize(),
            #transforms.functional.equalize(),
            #Lambda(lambda x: HistogramNormalize(num_bins=10)(x)),
            #transforms.Lambda(histogram_equalization),
            #transforms.Lambda(lambda x: transforms.functional.adjust_sharpness(x, sharpness_factor=2.0)),
            #transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize([pixel_mean]*3, [pixel_std]*3)])

val_transforms= transforms.Compose([
            #transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
            #transforms.Grayscale(num_output_channels=1),
            #transforms.functional.equalize(),
            #Lambda(lambda x: HistogramNormalize(num_bins=10)(x)),
            #transforms.Lambda(histogram_equalization),
            #transforms.functional.adjust_sharpness(sharpness_factor=2.0),
            #transforms.Lambda(lambda x: transforms.functional.adjust_sharpness(x, sharpness_factor=2.0)),
            MyResize(),
            transforms.ToTensor(),
            transforms.Normalize([pixel_mean]*3, [pixel_std]*3)])
train_transforms1= transforms.Compose([
            #transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
            #transforms.functional.equalize(),
            #Lambda(lambda x: HistogramNormalize(num_bins=10)(x)),
            #transforms.Lambda(histogram_equalization),
            #transforms.Lambda(lambda x: transforms.functional.adjust_sharpness(x, sharpness_factor=2.0)),
            #transforms.Grayscale(num_output_channels=1),
            #transforms.RandomHorizontalFlip(p=0.5),
            FFT(),
            transforms.ToTensor(),
            #transforms.Lambda(lambda x: FFT(x))])
            #FFT()])
            #transforms.Normalize([pixel_mean]*3, [pixel_std]*3)])
            ])

val_transforms1= transforms.Compose([
            #transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
            #transforms.Grayscale(num_output_channels=1),
            #transforms.functional.equalize(),
            #Lambda(lambda x: HistogramNormalize(num_bins=10)(x)),
            #transforms.Lambda(histogram_equalization),
            #transforms.functional.adjust_sharpness(sharpness_factor=2.0),
            FFT(),
            #MyResize(),
            transforms.ToTensor(),
            #transforms.Lambda(lambda x: FFT(x))])
            #FFT()])
            #transforms.Normalize([pixel_mean]*3, [pixel_std]*3)])
            ])

class Xraydataset(Dataset):
     def __init__(self, image_file_list, label_file_list, transforms, transforms1):
         self.image_file_list = image_file_list
         self.label_file_list = label_file_list
         self.transforms = transforms
         self.transforms1 = transforms1
         
     def __len__(self):
         return len(self.image_file_list)
     def __getitem__(self,index):
          image = Image.open(self.image_file_list[index]).convert('RGB')
          image1 = Image.open(self.image_file_list[index])
          return self.transforms(image), self.transforms1(image1), self.label_file_list[index]
          
train_ds = Xraydataset(image_file_list, image_label_list, train_transforms, train_transforms1)
train_loader = DataLoader(train_ds, batch_size = cfg.batch_size_train, shuffle =True, num_workers = cfg.num_workers)
val_ds = Xraydataset(v_image_file_list, v_image_label_list, val_transforms,val_transforms1)
val_loader = DataLoader(val_ds, batch_size = cfg.batch_size_val, shuffle =False, num_workers = cfg.num_workers)
