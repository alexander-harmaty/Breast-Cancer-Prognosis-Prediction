# Breast Cancer Prognosis Prediction
This repository contains the source code, experiments, and documentation for the "Breast Cancer Prognosis Prediction" project. The goal of the project is to improve prognostic predictions for breast cancer patients by leveraging a multimodal deep learning approach. Specifically, the project integrates imaging and clinical data using a joint fusion strategy: a CNN processes DICOM breast MRI images with .nrrd segmentation files being applied to specific series, while an RNN (using LSTM/GRU) handles clinical data. The features extracted from both modalities are combined via a fusion layer to predict patient outcomes. This work serves as a reference implementation for utilizing multimodal fusion techniques in medical applications, aiming to enhance accuracy and support precision oncology.

## Project Overview  
We aim to predict breast cancer recurrence by fusing imaging features (from a CNN on MRI scans) with clinical features (via an RNN on electronic health record data). Our pipeline:
- **Extract CNN features** from DICOM MRI volumes (TumorFeatureCNN).  
- **Extract RNN features** from tabular clinical data (advanced bidirectional LSTM).  
- **Fuse** the two feature sets in a simple fully‑connected “fusion” model.  

This approach leverages contextual patient information alongside imaging to improve prognostic accuracy.

---

## Plan of Action  

1. **Literature Review & Architecture Design**  
   - Studied multimodal fusion in medical imaging.  
   - Chose a CNN to capture spatial tumor characteristics.  
   - Chose an RNN to model time‑series and tabular clinical features.  
   - Designed a fusion layer to join both modalities for final prediction.

2. **Implementation Steps**  
   - **CNN component** (Mason)  
     - Load & preprocess DICOM series + optional segmentation masks.  
     - Define `TumorFeatureCNN` (3 conv layers + adaptive pool).  
     - Serialize extracted features to `cnn_features.pkl`.  
   - **Baseline RNN & Fusion** (Alex)  
     - Preprocess & encode clinical Excel sheet.  
     - Train a SimpleRNN baseline.  
     - Fuse baseline CNN+RNN features in `fusion_layer.py`.  
   - **Advanced RNN & Fusion** (Austine)  
     - Build bidirectional LSTM with regularization.  
     - Improve fusion layer with threshold tuning & cross‑validation.  
     - Serialize final features to `rnn_features.pkl`.

---

## Languages, Dependencies & Setup  

- **Languages**: Python 3.8+  
- **Core Libraries**:  
  - TensorFlow 2.x / Keras  
  - PyTorch  
  - scikit‑learn, imbalanced‑learn  
  - pandas, numpy, matplotlib  
  - pydicom, SimpleITK (for DICOM/NRRD I/O)  
- **Optional**: Google Colab (for free GPU)  
- **Data Files**:  
  - `Clinical_and_Other_Features.xlsx` (included)  
  - DICOM MRI volumes & NRRD masks (download separately; see below)

**Install via**  
```bash
pip install tensorflow torch torchvision scikit-learn imbalanced-learn pandas numpy matplotlib pydicom SimpleITK
```

## Colab vs. Local Python  

- **Google Colab**  
  - We developed and tested the notebooks (`.ipynb`) on Colab to leverage free GPU/TPU resources, speeding up CNN training on large MRI volumes and RNN training on tabular data.  
  - Simply open the notebook in Colab, connect to a GPU runtime, install any missing packages, and run the cells in order.  

- **Local Python**  
  - All steps are mirrored in standalone `.py` scripts so you can run end‑to‑end on your own machine.  
  - Requires installing Python dependencies (see Section 3).  
  - Supports GPU if you have CUDA‑enabled hardware; otherwise runs on CPU (expect longer training times).

---

## Getting Started

### 1. **Prepare the Data**  
#### Clinical data for RNN  
   - Already included: `Clinical_and_Other_Features.xlsx`.  
#### MRI scans & masks for CNN
   - **Not included** (too large).  
   - Download the Duke Breast Cancer MRI dataset from TCIA:  
     https://wiki.cancerimagingarchive.net/display/Public/Duke-Breast-Cancer-MRI  
   - Organize as:
     ```
     data/images/<patient_id>/*.dcm
     data/masks/<patient_id>/Segmentation_<patient_id>_Breast.seg.nrrd
     ```
#### Clinical labels for CNN  
   - Place a CSV (`clinical.csv`) with columns `Name,Recurrence` under `data/clinical.csv`.

### 2. **Run in Google Colab** 
#### Upload or mount your `data/` folder to Colab.  
#### Open and run:
   - `CNN_with_pickle.ipynb` → produces `cnn_features.pkl`  
   - `Clinical_Data_RNN.ipynb` → produces `rnn_features.pkl`  
   - `Fusion_layer.ipynb` → trains & evaluates the fused model  
#### Download the generated `.pkl` files and final metrics.

### 3. **Run Locally via Python Scripts**  
#### CNN feature extraction
`python cnn_with_pickle.py \`

  --images_dir data/images/ \
  --masks_dir  data/masks/ \
  --clinical_csv data/clinical.csv \
  --output    cnn_features.pkl

#### RNN feature extraction
`python clinical_data_rnn.py \`

  --input     Clinical_and_Other_Features.xlsx \
  --output    rnn_features.pkl

#### Fusion model training & evaluation
`python fusion_layer.py \`

  --cnn_features rnn_features.pkl \
  --rnn_features rnn_features.pkl

- Inspect console output for confusion matrices, ROC AUC, and precision‑recall metrics.
- Visual artifacts (plots) are saved in the working directory.

## Data Attribution
This project uses the Duke Breast Cancer MRI dataset, including the clinical and other features, which is licensed under Creative Commons (CC BY-NC 4.0). The dataset is provided by The Cancer Imaging Archive (TCIA). For more details and to access the dataset, please visit: https://www.cancerimagingarchive.net/collection/duke-breast-cancer-mri DOI: 10.7937/TCIA.e3sv-re93

## Contributors
- Mason — Advanced CNN model, feature extraction, & serialization
- Alex — Presentation, sprint planning, baseline RNN model & baseline fusion layer
- Austine— Clinical data encoding, advanced RNN model & advanced fusion layer
