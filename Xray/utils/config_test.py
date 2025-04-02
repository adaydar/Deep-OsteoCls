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

from monai.visualize import (
    GradCAMpp,
    OcclusionSensitivity,
    SmoothGrad,
    GuidedBackpropGrad,
    GuidedBackpropSmoothGrad,
)
from scipy import ndimage as nd

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

class Test_CONFIG():

  dir_path = "./KneeOAxray2/"
  test_dir = dir_path + "test"
  device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model_name ="xxxx" 
  save_path = "./classification_Xray/"
  folder_path = save_path + model_name
  save_folder = folder_path+ "/" + model_name  + ".pth"
  save_folder1 = folder_path+ "/" + model_name + "_loss.png"
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

test_dir=test_cfg.test_dir
test_class_names0 = os.listdir(test_dir)
test_class_names = sorted(test_class_names0)
print(test_class_names)
test_num_class = len(test_class_names)
test_image_files = [[os.path.join(test_dir, test_class_name, x) 
               for x in os.listdir(os.path.join(test_dir, test_class_name))] 
               for test_class_name in test_class_names]

test_image_file_list = []
test_image_label_list = []
for i, class_name in enumerate(test_class_names):
    test_image_file_list.extend(test_image_files[i])
    test_image_label_list.extend([i]*len(test_image_files[i]))


