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
from utils.config import *
from utils.Xray_model import *
from utils.dataloader import *

optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
val_interval = 1

#Loss Function
def weighted_loss(outputs, labels, args):
    softmax_op = torch.nn.Softmax(1)
    prob_pred = softmax_op(outputs)

    def set_weights():
        # # weight matrix 01 (wm01)
        # init_weights = np.array([[1, 2, 3, 4, 5],
        #                          [2, 1, 2, 3, 4],
        #                          [3, 2, 1, 2, 3],
        #                          [4, 3, 2, 1, 2],
        #                          [5, 4, 3, 2, 1]], dtype=np.float)

        # weight matrix 02 (wm02)
        init_weights = np.array([[1, 3, 5, 7, 9],
                                 [3, 1, 3, 5, 7],
                                 [5, 3, 1, 3, 5],
                                 [7, 5, 3, 1, 3],
                                 [9, 7, 5, 3, 1]], dtype=np.float32)

        # # weight matrix 03 (wm03)
        # init_weights = np.array([[1, 4, 7, 10, 13],
        #                          [4, 1, 4, 7, 10],
        #                          [7, 4, 1, 4, 7],
        #                          [10, 7, 4, 1, 4],
        #                          [13, 10, 7, 4, 1]], dtype=np.float)

        # # weight matrix 04 (wm04)
        # init_weights = np.array([[1, 3, 6, 7, 9],
        #                          [4, 1, 4, 5, 7],
        #                          [6, 4, 1, 3, 5],
        #                          [7, 5, 3, 1, 3],
        #                          [9, 7, 5, 3, 1]], dtype=np.float)

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
    # loss = torch.mean(prob_pred * class_hot)

    return loss

loss_function = torch.nn.CrossEntropyLoss(weight=None) 

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

