import os
from PIL import Image
import numpy as np
import cv2
import torch
#!pip install monai
from monai.transforms import *
from monai.data import Dataset, DataLoader
from monai.metrics import ROCAUCMetric
import pandas as pd
from torchvision import datasets,transforms,models
import torchvision.transforms as datasets
import torch.nn.functional as F
from torchvision.transforms import functional as S
import torch.nn as nn
from tqdm import tqdm

from utils.config import *
from utils.Segmentation_model import *
from utils.Entropy_calc import *
from utils.Dataloader import *
from utils.MRI_model import *

weights = torch.tensor(cls_weights).to(cfg.device)
loss_function = torch.nn.CrossEntropyLoss(weight=weights) 
act = Activations(softmax=True)
to_onehot = AsDiscrete(to_onehot=cfg.num_classes)# n_classes=num_class

val_interval = 1


def weighted_loss(outputs, labels):
    softmax_op = torch.nn.Softmax(1)
    prob_pred = softmax_op(outputs)

    def set_weights():
        init_weights = np.array([[1, 3],
                                 [3, 1]], dtype=np.float32)

        adjusted_weights = init_weights + 1.0
        np.fill_diagonal(adjusted_weights, 0)

        return adjusted_weights
    cls_weights = set_weights()

    batch_num, class_num = outputs.size()
    class_hot = np.zeros([batch_num, class_num], dtype=np.float32)
    labels_np = labels.data.cpu().numpy()
    for ind in range(batch_num):
        class_hot[ind, :] = cls_weights[labels_np[ind], :]
    class_hot = torch.from_numpy(class_hot)
    class_hot = torch.autograd.Variable(class_hot).cuda()

    loss = torch.sum((prob_pred * class_hot)**2) / batch_num
    return loss


#Scheduler

class LRScheduler():
    def __init__(self, init_lr=1.0e-4, lr_decay_epoch=10):
        self.init_lr = init_lr
        self.lr_decay_epoch = lr_decay_epoch

    def __call__(self, optimizer, epoch):
        '''Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs.'''
        lr = self.init_lr * (0.8 ** (epoch // self.lr_decay_epoch))
        lr = max(lr, 1e-8)
        if epoch % self.lr_decay_epoch == 0:
            print ('LR is set to {}'.format(lr))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        return optimizer
