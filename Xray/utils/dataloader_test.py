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
from torchsummary import summary
from torchvision import datasets,transforms,models
from monai.metrics import get_confusion_matrix
from monai.metrics import compute_roc_auc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import seaborn as sn
import sys
import torch.nn as nn
import torchmetrics
from numpy import save
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib
from matplotlib import cm
import csv
from monai.transforms import HistogramNormalize
import torchvision

from monai.visualize import (
    GradCAMpp,
    OcclusionSensitivity,
    SmoothGrad,
    GuidedBackpropGrad,
    GuidedBackpropSmoothGrad,
)
from scipy import ndimage as nd
from utils.config_test import *

class Test_CONFIG():

  dir_path = "/workspace/udit/Akshay/KneeOAxray2/"
  test_dir = dir_path + "test"
  device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model_name ="model_class5_xray" 
  save_path = "/workspace/udit/Akshay/classification_MRI/M/B/Multiclass_Quant_Analysis/"
  folder_path = save_path + model_name
  save_folder = folder_path+ "/" + model_name  + ".pth"
  save_folder2 = folder_path+ "/" + model_name + "_cm.png"
  save_folder3 =folder_path + "/" + model_name + "_class_report.csv"
  save_folder4 =folder_path + "/" + "/"+ "gradcam/"
  embeddings_path = folder_path + "/" + model_name + "embeddings.npy"
  embeddings_path_png = folder_path + "/" + model_name + "embeddings.png"
  pretrained = True

  num_classes = 5

  batch_size_test = 1

  max_epochs = 1

  num_workers = 1

test_cfg = Test_CONFIG()

#Image Transformation
class SumDimension(Transform):
    def __init__(self, dim=1):
        self.dim = dim

    def __call__(self, inputs):
        return inputs.sum(self.dim)

class Astype(Transform):
    def __init__(self, type='uint8'):
        self.type = type
    def __call__(self, inputs):
        return inputs.astype(self.type)

sharpen_filter=np.array([[-1,-1,-1],
                 [-1,9,-1],
                [-1,-1,-1]])

pixel_mean, pixel_std = 0.66133188, 0.21229856

class MyResize(Transform):
    def __init__(self,size=(224,224)):
        self.size = size
    def __call__(self,inputs):
        sharp_image=cv2.filter2D(np.array(inputs),-1,sharpen_filter)
        image=cv2.resize(sharp_image,dsize=(self.size[1],self.size[0]),interpolation=cv2.INTER_CUBIC)        
        #image2=image[25:475,25:475]
        #smooth = cv2.GaussianBlur(image2,(3,3),0)
        return image #smooth

testX=np.array(test_image_file_list)
testY=np.array(test_image_label_list)

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
         
test_transforms= transforms.Compose([
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

test_transforms1 = transforms.Compose([
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
          
test_ds = Xraydataset(test_image_file_list, test_image_label_list,test_transforms, test_transforms1)
test_loader = DataLoader(test_ds, batch_size=test_cfg.batch_size_test, num_workers=test_cfg.num_workers)



