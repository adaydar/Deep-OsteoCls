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

from utils.config import *
from utils.config_test import *
from utils.Segmentation_model import *
from utils.Entropy_calc import *
from utils.Dataloader import *
from utils.Dataloader_test import *
from utils.MRI_model import *

model.load_state_dict(torch.load(test_cfg.save_folder))

y_true = list()
y_predicted = list()
cross_entropy_list = list()
auc_metric = ROCAUCMetric()
features = []
w1=[]
w2 = []

with torch.no_grad():
    model.eval()
    y_pred = torch.tensor([], dtype=torch.float32, device = test_cfg.device)
    y = torch.tensor([], dtype=torch.long, device=test_cfg.device)
    
    for k, test_data in tqdm(enumerate(test_loader)):
        test_images1,test_images2, test_labels = test_data[0].to(test_cfg.device), test_data[1].to(test_cfg.device), test_data[2].to(test_cfg.device)
        outputs,feature, weightage1, weightage2 = model(test_images1.float(), test_images2.float())
        outputs1 = outputs.argmax(dim=1)
        y_pred = torch.cat([y_pred, outputs], dim=0)
        y = torch.cat([y, test_labels], dim=0)
        cross_entropy_f1 = [act(i) for i in y_pred]
        features.append(feature.cpu().detach().numpy().reshape(-1))
        w1.append(weightage1)#cpu().detach().numpy())
        w2.append(weightage2)

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
    
    #print(len(w))
  
#saving the confusion metrics       
dict1 = {"image_name":image_list, "value":cross_entropy_list, 'y_true':y_true, 'y_predicted': y_predicted,"weight_medial" : w1,"weight_lateral": w2 }
print(len(image_list),len(cross_entropy_list))
dt= pd.DataFrame(dict1)
dt.to_csv(test_cfg.folder_path+ "/" + test_cfg.model_name +".csv") 
save(test_cfg.embeddings_path,test_features)

file_path = test_cfg.folder_path+ "/" + test_cfg.model_name +".csv"
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
num_categories = 2
for lab in range(num_categories):
    indices = test_predictions==lab
    ax.scatter(tsne_embeddings[indices,0],tsne_embeddings[indices,1], c=np.array(cmap(lab)).reshape(1,4), label = lab ,alpha=0.5)
ax.legend(fontsize='large', markerscale=2)
plt.savefig(test_cfg.embeddings_path_png, dpi=400)
