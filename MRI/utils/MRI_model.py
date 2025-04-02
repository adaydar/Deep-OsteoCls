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

from utils.config import *
from utils.Entropy_calc import *


from monai.networks.nets.densenet import DenseNet201
from torch.optim.lr_scheduler import StepLR
from monai.networks.nets.resnet import ResNet
from monai.networks.nets import SEResNet50,SEResNet101,HighResNet
import torch
from torchvision import datasets,transforms,models
import torch.nn as nn

#MRFF module
class cv1(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(cv1,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self,x):
        x = self.conv(x)
        return x
       
class cv2(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(cv2,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=5,stride=1,padding=2,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self,x):
        x = self.conv(x)
        return x
        
class cv3(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(cv3,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=7,stride=1,padding=3,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self,x):
        x = self.conv(x)
        return x 

class MRFF(nn.Module):
    def __init__(self,in_c):
      super(MRFF, self).__init__()
      self.conb1 = cv1(in_c,in_c)
      self.conb2 = cv2(in_c,in_c)
      self.conb3 = cv3(in_c,in_c)
      
    def forward(self,x):
        x_1 = self.conb1(x)
        x_2 = self.conb2(x)
        x_3 = self.conb3(x)
        y_12 = torch.cat((x_1,x_2,x_3),dim=1)     
        return y_12
        
#weights = torch.tensor([0.05,0.1,0.1,0.15,0.6],device=cfg.device)

torch.autograd.set_detect_anomaly(True)

model1 = models.densenet121(pretrained=cfg.pretrained).to(cfg.device)

model1.features.conv0 = torch.nn.Conv2d(21, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).to(cfg.device)

group_norm_layer = nn.GroupNorm(7,7)

mrff_layer = MRFF(7)

# Modify the first layer of VGG19 to include GroupNorm
model1.features.conv0 = nn.Sequential(
    group_norm_layer,
    mrff_layer,
    model1.features.conv0 
)

model2 = models.densenet121(pretrained=cfg.pretrained).to(cfg.device)
model2.features.conv0 = torch.nn.Conv2d(21, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).to(cfg.device)

group_norm_layer1 = nn.GroupNorm(7,7)

mrff_layer1 = MRFF(7)

# Modify the first layer of VGG19 to include GroupNorm
model2.features.conv0 = nn.Sequential(
    group_norm_layer1,
    mrff_layer1,
    model2.features.conv0 
)

class OsteoMRNet(nn.Module):
    def __init__(self, model1,model2, num_classes=cfg.num_classes):
        super(OsteoMRNet,self).__init__()
        self.densenet1 = model1
        self.densenet2 = model2
        #self.features1 = nn.Sequential(*list(self.vgg16.children())[:-1])
        #self.features2 = nn.Sequential(*list(self.VGG16.children())[:-1])
        self.avg_pool = nn.AdaptiveAvgPool2d((9, 9))
        #self.max_pool = nn.Conv2d(in_channels=14336, out_channels=14336//2,kernel_size=1,stride=1,padding=1)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=1024*2, nhead=8)
        encoder_norm = nn.LayerNorm(1024*2)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6, norm=encoder_norm)
        self.row_embed = nn.Parameter(torch.rand(50, 1024*2))
        self.col_embed = nn.Parameter(torch.rand(50, 1024*2))
        self.entropy_calc = EntropyCalculator()
       # self.weight = nn.Parameter(torch.Tensor([initial_weight]))
        
        self.classifier = nn.Sequential(
            nn.Linear(1024*4, 1024*4),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024*4, 1024*2),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024*2, num_classes),
        ).to(cfg.device)      

    def forward(self, x, x1):
         x = x.permute(0,2,1,3) 
         x1 = x1.permute(0,2,1,3)
         x = self.densenet1.features(x)         
         x1 = self.densenet2.features(x1) 
         E = self.entropy_calc(x) 
         E1 = self.entropy_calc(x1)
         D1 = E.item()/(E.item()+E1.item())
         D2 = E1.item()/(E.item()+E1.item())
         x2 = torch.cat((x,x1),dim=1)        
         x3 = nn.functional.adaptive_avg_pool2d(x2, (1, 1))  # Shape: (batch_size, hidden_dim, 1, 1)  
         #x1 = x1.view(x1.size(0), x1.size(1))   
         #print(x1.size())       
         h, w = x3.shape[-2:]
         pos = torch.cat([
                         self.col_embed[:w].unsqueeze(0).repeat(h,1,1),
                         self.row_embed[:h].unsqueeze(0).repeat(1,w,1),
                         ], dim=1).flatten(0,1).unsqueeze(1)
                         
         b = self.transformer_encoder(pos+x3.flatten(2).permute(2,0,1))
         b = b.permute(1,0,2)
         #print(x2.shape)
         #x2 = self.avg_pool(x2)
         b = torch.flatten(b, 1)
         #print(x2.shape)
         x5 = self.classifier(b)
         return x5,x2, D1, D2
        
num_classes = 2  # Adjust this according to your problem
model = OsteoMRNet(model1,model2,  num_classes=cfg.num_classes).to(cfg.device)

optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
