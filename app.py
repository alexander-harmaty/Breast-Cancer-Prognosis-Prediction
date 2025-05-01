import streamlit as st
import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import pydicom
import SimpleITK as sitk
import torch
from sklearn.preprocessing import LabelEncoder
import pandas.api.types as ptypes

# Expected features from training (96 total, in model input order)
EXPECTED_FEATURES = [
    'Age at diagnosis', 'Tumor size', 'Lymph nodes examined positive', 'Mutation count',
    'Nottingham prognostic index', 'Oncotree code', 'TMB (nonsynonymous)', 'Mutation load',
    'Fraction genome altered', 'HRD score', 'Neoplasm disease stage', 'AJCC stage',
    'Hormone Therapy', 'Radiation Therapy', 'Chemotherapy', 'Targeted Therapy',
    'Estrogen Receptor Status', 'Progesterone Receptor Status', 'HER2 Status', 'PAM50 Subtype',
    'Recurrence site', 'Histological Type', 'Tumor Grade', 'Laterality', 'Menopause Status',
    'Comorbidity Index', 'BMI Category', 'Smoking Status', 'Alcohol Use', 'Ethnicity',
    'Genetic Testing', 'Gene Panel Used', 'Family History of Cancer', 'Breastfeeding History',
    'Age at Menarche', 'Age at First Live Birth', 'Age at Menopause', 'HRT Use',
    'Oral Contraceptive Use', 'Parity', 'BRCA1 Status', 'BRCA2 Status', 'PALB2 Status',
    'ATM Status', 'CHEK2 Status', 'PTEN Status', 'TP53 Status', 'CDH1 Status',
    'Other Gene Mutations', 'Reconstruction Surgery', 'Sentinel Node Biopsy', 'Margins Status',
    'DCIS Presence', 'LCIS Presence', 'Inflammatory Breast Cancer', 'Multifocality',
    'Lymphovascular Invasion', 'Tumor-infiltrating Lymphocytes', 'Ki-67 Index', 'p53 Status',
    'Topoisomerase II alpha', 'Bcl-2 Expression', 'E-cadherin Expression', 'Cyclin D1 Expression',
    'EGFR Status', 'MGMT Methylation', 'PIK3CA Mutation', 'PTEN Expression', 'mTOR Activation',
    'AKT Activation', 'MAPK Activation', 'CDK4/6 Inhibitor Use', 'Immune Checkpoint Inhibitor',
    'Neoadjuvant Therapy', 'Adjuvant Therapy', 'Endocrine Therapy', 'HER2-targeted Therapy',
    'CDK4/6 Inhibitor Therapy', 'PD-L1 Expression', 'TILs Grade', 'MSI Status', 'Tumor Mutational Burden',
    'Genomic Grade Index', 'Gene Expression Signature', 'Circulating Tumor Cells',
    'Circulating Tumor DNA', 'Radiomics Features', 'MRI Enhancement Pattern', 'Mammographic Density',
    'Ultrasound Features', 'PET SUV Max', 'Bone Scan Result', 'Liver Function Test', 'Renal Function Test',
    'Performance Status', 'Comorbidity Score', 'Physical Activity Level', 'Dietary Score'
]

# Helper functions

def merge_headers(col_tuple):
    first, second = col_tuple
    first = first.replace('\n', ' ').strip() if isinstance(first, str) else ''
    second = second.replace('\n', ' ').strip() if isinstance(second, str) else ''
    return f"{first} - {second}" if second and 'Unnamed' not in second else first

def load_dicom_series(series_dir, target_size=(224, 224)):
    slices = [
        pydicom.dcmread(os.path.join(series_dir, file)).pixel_array.astype(np.float32)
        for file in sorted(os.listdir(series_dir)) if file.endswith('.dcm')
    ]
    volume = np.stack(slices, axis=0)
    image = np.mean(volume, axis=0)
    image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-5)
    image = torch.tensor(image).unsqueeze(0)
    image = torch.nn.functional.interpolate(image.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False)
    return image.squeeze(0)

def load_nrrd_mask(nrrd_path):
    if not os.path.exists(nrrd_path): return None
    image = sitk.ReadImage(nrrd_path)
    array = sitk.GetArrayFromImage(image)
    mask = (array > 0).astype(np.float32)
    mask = np.mean(mask, axis=0) if mask.shape[0] > 1 else mask[0]
    return torch.tensor(mask).unsqueeze(0)

def encode_clinical_data(uploaded_file):
    df = pd.read_excel(uploaded_file, header=[1, 2])
    df.columns = [merge_headers(col) for col in df.columns]

    if df.shape[0] >= 3 and any('=' in str(x) for x in df.iloc[0].values):
        df = df.iloc[3:].reset_index(drop=True)

    if 'Patient ID' in df.columns:
        df = df.drop(columns=['Patient ID'])

    target_cols = [col for col in df.columns if 'Recurrence event' in col]
    if target_cols:
        df = df.drop(columns=target_cols)

    for col in list(df.columns):
        try:
            if ptypes.is_object_dtype(df[col]) or ptypes.is_categorical_dtype(df[col]):
                df[col] = df[col].fillna("MISSING").astype(str)
                df[col] = pd.Categorical(df[col]).codes
            else:
                df[col] = df[col].fillna(0)
        except Exception as e:
            st.warning(f"Skipped column '{col}' due to error: {e}")

    for feature in EXPECTED_FEATURES:
        if feature not in df.columns:
            df[feature] = 0
    df = df[EXPECTED_FEATURES]

    clinical_array = np.expand_dims(np.expand_dims(df.values.astype(np.float32).flatten(), axis=0), axis=0)
    st.write("Shape of clinical_array before prediction:", clinical_array.shape)
    return clinical_array, df

# Model paths
model_dir = 'models'
rnn_model = load_model(os.path.join(model_dir, 'rnn_model.h5'))
fusion_model = load_model(os.path.join(model_dir, 'fusion_model_masked.h5'))
cnn_model_path = os.path.join(model_dir, 'cnn_model_masked.h5')
cnn_model_available = os.path.exists(cnn_model_path)
cnn_model = load_model(cnn_model_path) if cnn_model_available else None

st.title('Breast Cancer Recurrence Prediction')

clinical_file = st.file_uploader("Upload Clinical Data (Excel file)", type=['xlsx'])
mri_file = st.file_uploader("Upload MRI Image (DICOM)", type=['dcm'])

if clinical_file:
    st.write("Clinical data uploaded:")
    clinical_array, preview_df = encode_clinical_data(clinical_file)
    st.dataframe(preview_df.T)

if mri_file:
    st.write("MRI image uploaded:")
    dcm = pydicom.dcmread(mri_file)
    st.image(dcm.pixel_array, caption="Uploaded MRI Image", use_column_width=True)

if st.button("Predict Recurrence"):
    if clinical_file and mri_file:
        if cnn_model_available:
            mri_data = load_dicom_series(os.path.dirname(mri_file.name))
            clinical_pred = rnn_model.predict(clinical_array)
            mri_pred = cnn_model.predict(np.array([mri_data]))
            combined_pred = fusion_model.predict(np.concatenate([clinical_pred, mri_pred], axis=1))
            prediction = 'Recurrence' if combined_pred[0][0] > 0.5 else 'No Recurrence'
            st.write(f"Fusion Model Prediction: {prediction}")
        else:
            st.error("CNN model is not yet available for MRI predictions. Please upload only clinical data or wait for the model release.")

    elif clinical_file:
        clinical_pred = rnn_model.predict(clinical_array)
        prediction = 'Recurrence' if clinical_pred[0][0] > 0.5 else 'No Recurrence'
        st.write(f"RNN Model Prediction: {prediction}")

    elif mri_file:
        if cnn_model_available:
            mri_data = load_dicom_series(os.path.dirname(mri_file.name))
            mri_pred = cnn_model.predict(np.array([mri_data]))
            prediction = 'Recurrence' if mri_pred[0][0] > 0.5 else 'No Recurrence'
            st.write(f"CNN Model Prediction: {prediction}")
        else:
            st.warning("CNN model predictions are not yet available. Please use clinical data or both modalities.")
    else:
        st.error("Please upload at least one file (clinical data or MRI image) to make a prediction.")
