import os
from PIL import Image
import numpy as np
import cv2
import torch
#!pip install monai
from monai.transforms import *
from monai.data import Dataset, DataLoader
import pandas as pd
from torchvision import datasets,transforms,models
import torchvision.transforms as datasets
import torch.nn.functional as F
from torchvision.transforms import functional as S
import torch.nn as nn
import torchvision.transforms as T

class MRIDataset(Dataset):
     def __init__(self, image_file_list, image_file_list1, label_file_list, transforms):
         self.image_file_list = image_file_list
         self.image_file_list1 = image_file_list1
         self.label_file_list = label_file_list
         self.transforms = transforms
         self.pd_data =pd.DataFrame(self.label_file_list)

     def __len__(self):
         return len(self.image_file_list)
     def __getitem__(self,index):
         image = np.load(self.image_file_list[index])
         image1 = np.load(self.image_file_list1[index])
         return self.transforms(image), self.transforms(image1), self.label_file_list[index]

transformation = transforms.Compose([transforms.ToTensor()])
                      
test_dataset = MRIDataset(test_image_file_list,test_image_file_list1, test_image_label_list, transforms=transformation)
