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
from torchsummary import summary
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
from utils.test_config import *
from utils.test_dataloader import *
from utils.Xray_model import *
from utils.loss_fn import *

model.load_state_dict(torch.load(test_cfg.save_folder))

#testing
y_true = list()
y_predicted = list()
cross_entropy_list = list()
auc_metric = ROCAUCMetric()
features = []

with torch.no_grad():
    model.eval()
    y_pred = torch.tensor([], dtype=torch.float32, device=test_cfg.device)
    y = torch.tensor([], dtype=torch.long, device=test_cfg.device)
    
    for k, test_data in tqdm(enumerate(test_loader)):
        test_images1, test_images2, test_labels = test_data[0].to(test_cfg.device), test_data[1].to(test_cfg.device), test_data[2].to(test_cfg.device)
        outputs,feature = model(test_images1.float(), test_images2.float())
        outputs1 = outputs.argmax(dim=1)
        y_pred = torch.cat([y_pred, outputs], dim=0)
        y = torch.cat([y, test_labels], dim=0)
        cross_entropy_f1 = [act(i) for i in y_pred]
        features.append(feature.cpu().detach().numpy().reshape(-1))
        for i in y_pred:
           cs = act(i)
           cs1 = torch.max(cs)
           cs1 = round(cs1.item(),3)
        cross_entropy_list.append(cs1)
        for i in range(len(outputs)):
            y_predicted.append(outputs1[i].item())
            y_true.append(test_labels[i].item())
    y_onehot = [to_onehot(i) for i in y]
    y_pred_act = [act(i) for i in y_pred]
    auc_metric(y_pred_act, y_onehot)
    auc_result = auc_metric.aggregate()
    test_features = np.array(features)    
#saving the confusion metrics       
dict1 = {"image_name":image_list, "value":cross_entropy_list, "y_true":y_true, "y_predicted": y_predicted}
print(len(image_list),len(cross_entropy_list))
dt= pd.DataFrame(dict1)
dt.to_csv(test_cfg.folder_path+ "/" + test_cfg.model_name +".csv") 
save(test_cfg.embeddings_path,test_features)

file_path = test_cfg.folder_path+ "/" + test_cfg.model_name +".csv"

#plotting the confusion metrics and related metrices
df = pd.read_csv(file_path) 
confusion_matrix = pd.crosstab(df['y_true'], df['y_predicted'], rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(confusion_matrix, cmap="crest", annot=True, fmt=".0f")
plt.savefig(test_cfg.save_folder2)


#Calculate the QWK
from torchmetrics import ConfusionMatrix


def plot_confusion_matrix(y_true, y_pred, num_classes):
    y_true = torch.tensor(y_true)
    y_pred = torch.tensor(y_pred)
    cm = ConfusionMatrix(num_classes=num_classes, task='multiclass')
    cm.update(y_pred, y_true)
    cm_matrix = cm.compute().detach().cpu().numpy()
    return cm_matrix
    
confusion_matrix = plot_confusion_matrix(y_true, y_predicted, test_cfg.num_classes) 
   
y_true= df['y_true'].astype(int).tolist()
y_predicted = df['y_predicted'].astype(int).tolist()

from sklearn.metrics import cohen_kappa_score
y_true1 = np.array(y_true)
y_pred1 = np.array(y_predicted)
QWK = cohen_kappa_score(y_true1, y_pred1, weights="quadratic")


#Calculate MCC
def calculate_mcc(confusion_matrix):
    tp = confusion_matrix.diagonal()
    fp = confusion_matrix.sum(axis=0) - tp
    fn = confusion_matrix.sum(axis=1) - tp
    tn = confusion_matrix.sum() - (tp + fp + fn)
    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = numerator / denominator
    mcc = np.mean(mcc)
    return mcc

mcc = calculate_mcc(confusion_matrix)

#Calculate MAE

def compute_mae(y_true, y_pred):
      y_true = torch.tensor(y_true, dtype=torch.float)
      y_pred = torch.tensor(y_pred, dtype=torch.float)
      mae = torch.mean(torch.abs(y_pred - y_true))
      return mae.item()

mae = compute_mae(y_true, y_predicted)
     
sys.stdout = open(test_cfg.save_folder3, "w")
print(classification_report(y_true, y_predicted, target_names=test_class_names, digits=4))
print("AUC:",auc_result)
print("QWK:", QWK)
print("MCC:", mcc)
print("MAE:",mae)
sys.stdout.close()

#Save tSNE Plot
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200)
embeddings = np.load(test_cfg.embeddings_path)
tsne_embeddings = tsne.fit_transform(embeddings)
test_predictions = np.array(y_predicted)
cmap = cm.get_cmap("Set1") #tab20
fig, ax = plt.subplots(figsize=(8,8))
num_categories = 5
for lab in range(num_categories):
    indices = test_predictions==lab
    ax.scatter(tsne_embeddings[indices,0],tsne_embeddings[indices,1], c=np.array(cmap(lab)).reshape(1,4), label = lab ,alpha=0.5)
ax.legend(fontsize='large', markerscale=2)
plt.savefig(test_cfg.embeddings_path_png, dpi=400)


#Gradcam
from torchvision.utils import make_grid, save_image
import torchvision.transforms as T
import torch.nn.functional as F


def find_vgg_layer(arch, target_layer_name):
    hierarchy = target_layer_name.split('_')

    if len(hierarchy) >= 1:
        target_layer = arch.vgg19.features

    if len(hierarchy) == 2:
        target_layer = target_layer[int(hierarchy[1])]

    return target_layer

def find_resnet_layer(arch, target_layer_name):
    """Find resnet layer to calculate GradCAM and GradCAM++
    
    Args:
        arch: default torchvision densenet models
        target_layer_name (str): the name of layer with its hierarchical information. please refer to usages below.
            target_layer_name = 'conv1'
            target_layer_name = 'layer1'
            target_layer_name = 'layer1_basicblock0'
            target_layer_name = 'layer1_basicblock0_relu'
            target_layer_name = 'layer1_bottleneck0'
            target_layer_name = 'layer1_bottleneck0_conv1'
            target_layer_name = 'layer1_bottleneck0_downsample'
            target_layer_name = 'layer1_bottleneck0_downsample_0'
            target_layer_name = 'avgpool'
            target_layer_name = 'fc'
            
    Return:
        target_layer: found layer. this layer will be hooked to get forward/backward pass information.
    """
    if 'layer' in target_layer_name:
        hierarchy = target_layer_name.split('_')
        layer_num = int(hierarchy[0].lstrip('layer'))
        if layer_num == 1:
            target_layer = arch.layer1
        elif layer_num == 2:
            target_layer = arch.layer2
        elif layer_num == 3:
            target_layer = arch.layer3
        elif layer_num == 4:
            target_layer = arch.layer4
        else:
            raise ValueError('unknown layer : {}'.format(target_layer_name))

        if len(hierarchy) >= 2:
            bottleneck_num = int(hierarchy[1].lower().lstrip('bottleneck').lstrip('basicblock'))
            target_layer = target_layer[bottleneck_num]

        if len(hierarchy) >= 3:
            target_layer = target_layer._modules[hierarchy[2]]
                
        if len(hierarchy) == 4:
            target_layer = target_layer._modules[hierarchy[3]]

    else:
        target_layer = arch._modules[target_layer_name]

    return target_layer 

def visualize_cam(mask, img):
    print(type(mask))
    print(type(img))
    #heatmap = cv2.applyColorMap(np.uint8(255 * mask.squeeze()), cv2.COLORMAP_JET).to('cpu')
    #print(heatmap)
    mask = mask.detach().cpu().data.numpy()
    #img = img.detach().cpu().data.numpy()
    heatmap = cv2.applyColorMap(np.uint8(255 * mask.squeeze()),cv2.COLORMAP_JET)
    #print(type(heatmap))
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
    #print(type(heatmap))
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b])
    print(type(img))
    result = 0.5 * heatmap + img.cpu()
    result = result.div(result.max()).squeeze()
    
    return heatmap, result
    
class GradCAM(object):
    def __init__(self, model_dict, verbose=False):
        model_type = model_dict['type']
        layer_name = model_dict['layer_name']
        self.model_arch = model_dict['arch']

        self.gradients = dict()
        self.activations = dict()
        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]
            return None
        def forward_hook(module, input, output):
            self.activations['value'] = output
            return None

        if 'vgg' in model_type.lower():
            target_layer = find_vgg_layer(self.model_arch, layer_name)


        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

        if verbose:
            try:
                input_size = model_dict['input_size']
            except KeyError:
                print("please specify size of input image in model_dict. e.g. {'input_size':(224, 224)}")
                pass
            else:
                device = 'cuda' if next(self.model_arch.parameters()).is_cuda else 'cpu'
                self.model_arch(torch.zeros(1, 3, *(input_size), device=test_cfg.device),torch.zeros(1, 1, *(input_size),device=test_cfg.device))
                #print('saliency_map size :', self.activations['value'].shape[2:])


    def forward(self, input1, input2, class_idx=None, retain_graph=False):
        """
        Args:
            input: input image with shape of (1, 3, H, W)
            class_idx (int): class index for calculating GradCAM.
                    If not specified, the class index that makes the highest model prediction score will be used.
        Return:
            mask: saliency map of the same spatial dimension with input
            logit: model output
        """
        b, c, h, w = input1.size()

        logit = self.model_arch(input1, input2)
        if class_idx is None:
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            score = logit[:, class_idx].squeeze()

        self.model_arch.zero_grad()
        score.backward(retain_graph=retain_graph)
        gradients = self.gradients['value']
        activations = self.activations['value']
        b, k, u, v = gradients.size()

        alpha = gradients.view(b, k, -1).mean(2)
        #alpha = F.relu(gradients.view(b, k, -1)).mean(2)
        weights = alpha.view(b, k, 1, 1)

        saliency_map = (weights*activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.upsample(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data

        return saliency_map, logit

    def __call__(self, input1,input2, class_idx=None, retain_graph=False):
        return self.forward(input1,input2, class_idx, retain_graph)


def denormalize(tensor, mean, std):
    if not tensor.ndimension() == 4:
        raise TypeError('tensor should be 4D')

    mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)

    return tensor.mul(std).add(mean)
    
def denormalize1(tensor, mean, std):
    if not tensor.ndimension() == 4:
        raise TypeError('tensor should be 4D')

    mean = torch.FloatTensor(mean).view(1, 1, 1, 1).expand_as(tensor).to(tensor.device)
    std = torch.FloatTensor(std).view(1, 1, 1, 1).expand_as(tensor).to(tensor.device)

    return tensor.mul(std).add(mean)

def normalize1(tensor, mean, std):
    if not tensor.ndimension() == 4:
        raise TypeError('tensor should be 4D')

    mean = torch.FloatTensor(mean).view(1, 1, 1, 1).expand_as(tensor).to(tensor.device)
    std = torch.FloatTensor(std).view(1, 1, 1, 1).expand_as(tensor).to(tensor.device)

    return tensor.sub(mean).div(std)
    
def normalize(tensor, mean, std):
    if not tensor.ndimension() == 4:
        raise TypeError('tensor should be 4D')

    mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)

    return tensor.sub(mean).div(std)
    
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return self.do(tensor)
    
    def do(self, tensor):
        return normalize(tensor, self.mean, self.std)
    
    def undo(self, tensor):
        return denormalize(tensor, self.mean, self.std)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
        
class Normalize1(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return self.do(tensor)
    
    def do(self, tensor):
        return normalize1(tensor, self.mean, self.std)
    
    def undo(self, tensor):
        return denormalize1(tensor, self.mean, self.std)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

normalizer = Normalize(mean=[0.66133188, 0.66133188, 0.66133188], std=[0.21229856, 0.21229856, 0.21229856])    
normalizer1 = Normalize1(mean=[0.66133188], std=[0.21229856])   
      
def gradcam_Plot(img):
    images_1=[]
    pil_img = Image.open(img).convert('RGB')
    pil_img1 = Image.open(img).convert('L')
    a = np.asarray(pil_img1)
    a = np.expand_dims(a, axis=2)
    print(a.shape)
    #fig, ax = plt.subplots(1, 1, facecolor='white')
    torch_img = torch.from_numpy(np.asarray(pil_img)).permute(2, 0, 1).unsqueeze(0).float().div(255).cuda()
    print(torch_img.size())
    torch_img1 = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).float().div(255).cuda()
    print(torch_img1.size())
    torch_img = F.upsample(torch_img, size=(224, 224), mode='bilinear', align_corners=False)
    normed_torch_img = normalizer(torch_img)
    normed_torch_img1 = normalizer1(torch_img1)
    cam_dict = dict()
    vgg_model_dict = dict(type='vgg', arch=model, layer_name='avgpool', input_size=(224, 224))
    vgg_gradcam = GradCAM(vgg_model_dict, True)
    #vgg_gradcampp = GradCAMpp(vgg_model_dict, True)
    cam_dict['vgg'] = [vgg_gradcam]
    for gradcam in cam_dict.values():
       mask, _ = vgg_gradcam(normed_torch_img, normed_torch_img1)
       heatmap, result = visualize_cam(mask, torch_img)
       images_1.append(torch.stack([result], 0))
    images = make_grid(torch.cat(images_1, 0), nrow=1)
    transform = T. ToPILImage()
    images = transform(images)
    #plt.imshow(images)
    images.save(test_cfg.save_folder4 + str((j)) + ".png")
    #plt.savefig(test_cfg.save_folder4 + str((j)) + ".png",bbox_inches='tight')
"""
for i,k in tqdm(enumerate(test_image_file_list)):
 p,s = os.path.split(k)
 j,m = os.path.splitext(s)
 #print(k)
 sample = test_image_file_list[i]
 gradcam_Plot(k)
"""



