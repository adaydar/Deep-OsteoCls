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

# Fix all random seeds.
import random
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
seed_everything(1)

class CONFIG():
  dir_path = "./KneeOAxray2/"
  save_path = "./classification_xray/github/" 
  train_dir=dir_path + 'train' 
  val_dir=dir_path + 'val'

  device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model_name = "xxxx"
  folder_path = save_path + model_name
  save_folder = folder_path+ "/" + model_name  + ".pth"
  save_folder1 = folder_path+ "/" + model_name + "_loss.png"
  save_folder2 = folder_path+ "/" + model_name + "_cm.png"
  save_folder3 =folder_path + "/" + model_name + "_class_report.csv"
  save_folder4 =folder_path + "/" + model_name + "_Gradcam.png"
  save_folder5 =folder_path + "/" + model_name + "summary.txt"
  embeddings_path = folder_path + "/" + model_name + "embeddings.npy"
  pretrained = True

  num_classes = 5

  batch_size_train = 32
  batch_size_val = 32

  max_epochs = 50
  lr = 5e-4
  weight_decay = 5e-3
  lr_decay_epoch =5 

  num_workers = 1

cfg = CONFIG()

#import train
class_names0 = os.listdir(cfg.train_dir)
class_names = sorted(class_names0)
print(class_names)
num_class = len(class_names)
image_files = [[os.path.join(cfg.train_dir, class_name, x) 
               for x in os.listdir(os.path.join(cfg.train_dir, class_name))] 
               for class_name in class_names]

image_file_list = []
image_label_list = []
for i, class_name in enumerate(class_names):
    image_file_list.extend(image_files[i])
    image_label_list.extend([i] * len(image_files[i]))
    
#import valid
v_class_names0 = os.listdir(cfg.val_dir)
v_class_names = sorted(v_class_names0)
print(v_class_names)
v_num_class = len(v_class_names)
v_image_files = [[os.path.join(cfg.val_dir, v_class_name, x) 
               for x in os.listdir(os.path.join(cfg.val_dir, v_class_name))] 
               for v_class_name in v_class_names]

v_image_file_list = []
v_image_label_list = []
for i, class_name in enumerate(v_class_names):
    v_image_file_list.extend(v_image_files[i])
    v_image_label_list.extend([i]*len(v_image_files[i]))
    
#Save the file
k = pd.DataFrame(dict({"image_name":image_file_list}))
k.to_csv(cfg.folder_path+"/train_list.csv")
trainX=np.array(image_file_list)
trainY=np.array(image_label_list)
valX=np.array(v_image_file_list)
valY=np.array(v_image_label_list)
