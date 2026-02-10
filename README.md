This repo contains the Official Pytorch implementation of our work:

DeepOsteoCls:Deep Learning-Based Framework for Knee Osteoarthritis Classification With Qualitative Explanations from Radiographs and MRI Volumes (Published in Biomedical Signal Processing and Control Journal)

![DeepOsteoCls_architecture](./DeepOsteoCls_architecture.png) Figure: Schematic of DeepOsteoCls framework and its components;(a) feature extraction block and classifier,(b) proposed OAED (Osteoarthritis Edge Detection module), (c) GAP (Global average pooling) module [26], (d) transformer encoder [27], (e) The MRI slice preprocessing module, (b) feature extraction block and classifier,
(f) MRI postprocessing module/ Domain Knowledge Transfer and Entropy Regularization (DoKTER) training scheme, (g) Multi-Resolution Attentive-Unet (MtRA-Unet)[28] model for segmentation of femoral cartilages,(i) the Region of Interest (ROI) in knee MRI, and (j) proposed Multi-Resolution Feature Integration (MRFI) module.

To cite the article:

@article{DAYDAR2026109819,
title = {DeepOsteoCls: Deep learning-based framework for Knee Osteoarthritis Classification with qualitative explanations from radiographs and MRI volumes},
journal = {Biomedical Signal Processing and Control},
volume = {119},
pages = {109819},
year = {2026},
issn = {1746-8094},
doi = {https://doi.org/10.1016/j.bspc.2026.109819},
url = {https://www.sciencedirect.com/science/article/pii/S1746809426003733},
author = {Akshay Daydar and Arijit Sur and Subramani Kanagaraj and Hanif Laskar},
keywords = {Clinical manifestations, Deep learning, Edge-based features, Multi-scale regional features, Osteoarthritis Initiative (OAI)},
abstract = {Knee Osteoarthritis (KOA) is a degenerative joint disorder affecting middle-aged and elderly individuals, with its diagnosis facing challenges in achieving objective, transparent quantification and incorporating clinical manifestations, despite advances in deep-learning for medical imaging. To address these issues, in this paper, a deep learning-based hybrid (Convolutional Neural Network (CNN)-Transformer encoder) classification framework, DeepOsteoCls, is proposed to perform binary and multi-class classification of KOA from X-rays and MRI scans from OsteoXRNet and OsteoMRNet models separately, with Gradient-weighted Class Activation Mappings (Grad-CAMs). The Osteoarthritis Edge Detection (OAED) and Multi-Resolution Feature Integration (MRFI) modules are also introduced in the proposed framework to facilitate the extraction of edge-based features from X-ray images and multi-scale regional features from the MRI volume, respectively. Furthermore, a disorder-aware weakly supervised training scheme—Domain Knowledge Transfer and Entropy Regularization (DoKTER) is proposed to enhance the explainability of Radiological KOA (RKOA) diagnosis by predicting the region score and GradCAMs of MRI scans. Comprehensive experiments on the Osteoarthritis Initiative (OAI) dataset demonstrated that the proposed framework achieved a classification accuracy of 72.10% for X-ray and 53.16% for MRI in a multi-class classification task, and 85.74% for X-ray and 81.04% for MRI in a binary classification task, outperforming state-of-the-art models. The DoKTER scheme is found to accurately classify the affected region with 65.15% and 62.5% for the OAI and Multi-Hospital Knee Osteoarthritis (MHKOA) datasets, respectively. Additionally, Femoral Cartilage Thickness (FCT) in non-RKOA subjects can be effectively monitored using the region score, with distinct cut-offs values. The code is available at: https://github.com/adaydar/Deep-OsteoCls}
}

Requirements

    Linux
    Python3 3.8.10
    Pytorch 1.13.1
    train and test with A100 GPU

Prepare Dataset:

    1. Kindly check "https://github.com/adaydar/MtRA-Unet/tree/main" repository for dataset preparation.
    2. Then kindly check the "config" file before running the training and testing code.

Training and Testing:

Prepare the dataset and then run the following command for training:

    python3 train.py

For Testing, run

    python3 test.py
