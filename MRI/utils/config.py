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

class CONFIG():
  dir_path = "./classification_MRI/Data_Medial_Binary_2/" 
  dir_path1 = "./classification_MRI/Data_Lateral_Binary_2/"
  save_path = "./classification_MRI/exp/"
  train_dir=dir_path + 'Train' 
  val_dir=dir_path + 'Val'
  
  train_dir1 =dir_path1 + 'Train' 
  val_dir1 = dir_path1 + 'Val'

  device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model_name = "xxxx"
  folder_path = save_path + model_name
  save_folder = folder_path+ "/" + model_name  + ".pth"
  save_folder1 = folder_path+ "/" + model_name + "_loss.png"
  save_folder2 = folder_path+ "/" + model_name + "_cm.png"
  save_folder3 =folder_path + "/" + model_name + "_class_report.csv"
  save_folder4 =folder_path + "/" + model_name + "_Gradcam.png"
  save_folder5 =folder_path + "/" + model_name + "summary.txt"
  save_folder6 = folder_path+ "/" + model_name  + ".pth"
  save_folder7 = folder_path+ "/" + model_name  + "_cs.csv"
  embeddings_path = folder_path + "/" + model_name + "embeddings.npy"
  log_root_folder = folder_path + "/log"
  pretrained = True

  num_classes = 2

  batch_size_train = 36
  batch_size_val = 36

  max_epochs = 30
  lr = 5e-4
  weight_decay = 5e-3
  lr_decay_epoch = 5 

  num_workers = 1
  patience = 5
  log_every = 100
  lr_scheduler = "plateau"

cfg = CONFIG()

#import train
class_names0 = os.listdir(cfg.train_dir)
class_names = sorted(class_names0)
print(class_names)
num_class = len(class_names)
image_files = [[os.path.join(cfg.train_dir, class_name, x) 
               for x in os.listdir(os.path.join(cfg.train_dir, class_name))] 
               for class_name in class_names]
print(num_class)

image_files1 = [[os.path.join(cfg.train_dir1, class_name, x) 
               for x in os.listdir(os.path.join(cfg.train_dir1, class_name))] 
               for class_name in class_names]

image_file_list = []
image_label_list = []
for i, class_name in enumerate(class_names):
    image_file_list.extend(image_files[i])
    image_label_list.extend([i] * len(image_files[i]))
values_to_count = [0, 1]
counts = [image_label_list.count(value) for value in values_to_count]
for value, count in zip(values_to_count, counts):
    print(f"Count of {value}: {count}")

image_file_list1 = []
for i, class_name in enumerate(class_names):
     image_file_list1.extend(image_files1[i])

#print(image_file_list)
        
#import valid
v_class_names0 = os.listdir(cfg.val_dir)
v_class_names = sorted(v_class_names0)
print(v_class_names)
v_num_class = len(v_class_names)
v_image_files = [[os.path.join(cfg.val_dir, v_class_name, x) 
               for x in os.listdir(os.path.join(cfg.val_dir, v_class_name))] 
               for v_class_name in v_class_names]

v_image_files1 = [[os.path.join(cfg.val_dir1, v_class_name, x) 
               for x in os.listdir(os.path.join(cfg.val_dir1, v_class_name))] 
               for v_class_name in v_class_names]

v_image_file_list = []
v_image_label_list = []
for i, class_name in enumerate(v_class_names):
    v_image_file_list.extend(v_image_files[i])
    v_image_label_list.extend([i]*len(v_image_files[i]))

v_image_file_list1 = []
for i, class_name in enumerate(class_names):
     v_image_file_list1.extend(v_image_files1[i])

#print(len(image_file_list))  
#print(len(image_file_list1))  
#print(len(v_image_file_list))
#print(len(v_image_file_list1))

#Save the file
k = pd.DataFrame(dict({"image_name":image_file_list, "edges": image_file_list1}))
k.to_csv(cfg.folder_path+"train_list.csv")
trainX=np.array(image_file_list)
trainY=np.array(image_label_list)
valX=np.array(v_image_file_list)
valY=np.array(v_image_label_list)

def tensor_permutation(x):
    # Permute the tensor dimensions
    return x.permute(1,0,2)  # x.permute(0,2,1,3) Permute dimensions from (3, 64, 64) to (64, 3, 64)
