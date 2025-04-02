import os
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import torch
import torch.nn as nn
import random
import torchvision.transforms as T
import torch.nn.functional as F

# Fix all random seeds.
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
seed_everything(1)

class Test_CONFIG():

  dir_path = "./classification_MRI/Data_Medial_Binary_2/" 
  dir_path1 = "./classification_MRI/Data_Lateral_Binary_2/"
  save_path = "./classification_MRI/exp/"
  test_dir=dir_path + 'Test' 
  test_dir1 =dir_path1 + 'Test' 
  device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model_name ="xxxx"   
  folder_path = save_path + model_name
  save_folder = folder_path+ "/" + model_name  + ".pth"
  save_folder2 = folder_path+ "/" + model_name + "_cm.png"
  save_folder3 = folder_path + "/" + model_name + "_class_report.csv"
  save_folder4 =folder_path + "/" + "/"+ "gradcam/"
  embeddings_path = folder_path + "/" + model_name + "embeddings.npy"
  embeddings_path_png = folder_path + "/" + model_name + "tSNE.png"
  pretrained = True

  num_classes = 2

  batch_size_test = 1

  max_epochs = 1

  num_workers = 1

test_cfg = Test_CONFIG()

test_dir = test_cfg.test_dir
test_class_names0 = os.listdir(test_dir)
test_class_names = sorted(test_class_names0)
print(test_class_names)
test_num_class = len(test_class_names)
test_image_files = [[os.path.join(test_dir, test_class_name, x) 
               for x in os.listdir(os.path.join(test_dir, test_class_name))] 
               for test_class_name in test_class_names]

test_image_files1 = [[os.path.join(test_cfg.test_dir1, test_class_name, x) 
               for x in os.listdir(os.path.join(test_cfg.test_dir1, test_class_name))] 
               for test_class_name in test_class_names]

test_image_file_list = []
test_image_label_list = []
for i, class_name in enumerate(test_class_names):
    test_image_file_list.extend(test_image_files[i])
    test_image_label_list.extend([i]*len(test_image_files[i]))

test_image_file_list1 = []
for i, class_name in enumerate(test_class_names):
     test_image_file_list1.extend(test_image_files1[i])
     
print(test_image_file_list)
print(test_image_file_list1)
