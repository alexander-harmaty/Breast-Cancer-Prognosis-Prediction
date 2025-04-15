# Breast-Cancer-Prognosis-Prediction

## BU-CS-581A: Biomedicine Multimodal ML Course Project

This repository contains the source code, experiments, and documentation for the "Breast Cancer Prognosis Prediction" project. The goal of the project is to improve prognostic predictions for breast cancer patients by leveraging a multimodal deep learning approach. Specifically, the project integrates imaging and clinical data using a joint fusion strategy: a CNN (built on a ResNet framework) processes DICOM breast MRI images, while an RNN (using LSTM/GRU) handles clinical data. The features extracted from both modalities are combined via a fusion layer to predict patient outcomes. This work serves as a reference implementation for utilizing multimodal fusion techniques in medical applications, aiming to enhance accuracy and support precision oncology.

## Data Attribution
This project uses the Duke Breast Cancer MRI dataset, including the clinical and other features, which is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0). The dataset is provided by The Cancer Imaging Archive (TCIA).
For more details and to access the dataset, please visit:
https://www.cancerimagingarchive.net/collection/duke-breast-cancer-mri
DOI: 10.7937/TCIA.e3sv-re93
