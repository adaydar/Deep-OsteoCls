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

from utils.config_test import *
from utils.Segmentation_model import *

import os
image_list=[]
for i,k in enumerate(test_image_file_list):
  _,j= os.path.split(k)
  image_list.append(j)
  
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
                   
train_dataset = MRIDataset(image_file_list, image_file_list1, image_label_list, transforms=transformation)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size_train, shuffle=True, num_workers=cfg.num_workers, drop_last=False)

validation_dataset = MRIDataset(v_image_file_list,v_image_file_list1, v_image_label_list, transforms=transformation)
val_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=cfg.batch_size_val, shuffle=-False, num_workers=cfg.num_workers, drop_last=False)
   
cls_weights = [len(train_dataset) / cls_count for cls_count in counts]
#print(cls_weights)
instance_weights = [cls_weights[label] for label in values_to_count]
print("class weights:",instance_weights)
sampler = torch.utils.data.WeightedRandomSampler(torch.Tensor(instance_weights), len(train_dataset))
