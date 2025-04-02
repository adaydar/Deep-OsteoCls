This repo contains the Official Pytorch implementation of our work:

DeepOsteoCls:Deep Learning-Based Framework for Knee Osteoarthritis Classification With Qualitative Explanations from Radiographs and MRI Volumes (under Review at Biomedical Signal Processing and Control)

![DeepOsteoCls_architecture](./DeepOsteoCls.png) Schematic of DeepOsteoCls framework and its components;(a) feature extraction block and classifier,(b) proposed OAED (Osteoarthritis Edge Detection module), (c) GAP (Global average pooling) module [26], (d) transformer encoder [27], (e) The MRI slice preprocessing module, (b) feature extraction block and classifier,
(f) MRI postprocessing module/ Domain Knowledge Transfer and Entropy Regularization (DoKTER) training scheme, (g) Multi-Resolution Attentive-Unet (MtRA-Unet)[28] model for segmentation of femoral cartilages,
(i) the Region of Interest (ROI) in knee MRI, and (j) proposed Multi-Resolution Feature Integration (MRFI) module.

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
