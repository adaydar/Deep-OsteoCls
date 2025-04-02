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
from monai.networks.nets.resnet import ResNet
from monai.networks.nets.densenet import DenseNet201
from monai.networks.nets import SEResNet50,SEResNet101,HighResNet
from monai.transforms import *
from monai.data import Dataset, DataLoader
from monai.utils import set_determinism
from tqdm import tqdm
import pandas as pd
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets,transforms,models
import sys
import torch.nn as nn
from utils.config import *
from utils.dataloader import *

act = Activations(softmax=True)
to_onehot = AsDiscrete(to_onehot=num_class)# n_classes=num_class
loss_function = torch.nn.CrossEntropyLoss(weight=None) 
model1 = models.vgg19(pretrained=cfg.pretrained).to(cfg.device)
model3=  models.vgg19(pretrained=cfg.pretrained).to(cfg.device)
model3.features[0] = torch.nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).to(cfg.device)

torch.autograd.set_detect_anomaly(True)

class OsteoXRNet(nn.Module):
    def __init__(self, model1,model3, num_classes=cfg.num_classes):
        super(OsteoXRNet, self).__init__()
        self.vgg19 = model1
        self.VGG19 = model3
        self.avg_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=1024, nhead=8)
        encoder_norm = nn.LayerNorm(1024)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6, norm=encoder_norm)
        self.row_embed = nn.Parameter(torch.rand(50, 1024))
        self.col_embed = nn.Parameter(torch.rand(50,1024))
        
        self.classifier = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, num_classes),
        ).to(cfg.device)

    def forward(self, x, x1):
        x = self.vgg19.features(x)
        x1 = self.VGG19.features(x1)
        x2 = torch.cat((x, x1), dim=1)
        #print(x2.shape)
        x3 = nn.functional.adaptive_avg_pool2d(x2, (1, 1))  # Shape: (batch_size, hidden_dim, 1, 1)
        h, w = x3.shape[-2:]
        pos = torch.cat([
                         self.col_embed[:w].unsqueeze(0).repeat(h,1,1),
                         self.row_embed[:h].unsqueeze(0).repeat(1,w,1),
                         ], dim=1).flatten(0,1).unsqueeze(1)
                         
        x4 = self.transformer_encoder(pos+x3.flatten(2).permute(2,0,1))
        x5 = x4.permute(1,0,2)
        #print(x2.shape)
        #x2 = self.avg_pool(x2)
        x5 = torch.flatten(x5, 1)
        #print(x2.shape)
        x5 = self.classifier(x5)
        return x5, x2
        
num_classes = 5  # Adjust this according to your problem
model = OsteoXRNet(model1, model3, num_classes = cfg.num_classes).to(cfg.device)
