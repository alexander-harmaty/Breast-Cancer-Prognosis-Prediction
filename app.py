# General imports
import os
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model

# Clinical data processing
#from data_processing.clinical_data_rnn import encode_clinical_data
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# MRI image processing
#from data_processing.mri_images_cnn import load_dicom_series, load_nrrd_mask
import pydicom
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F



# Bypass mri_images_cnn and clinical_data_rnn imports for now
def merge_headers(col_tuple):
    first, second = col_tuple
    if isinstance(first, str):
        first = first.replace('\n', ' ').strip()
    if isinstance(second, str):
        second = second.replace('\n', ' ').strip()
    if not second or 'Unnamed' in second:
        return first
    else:
        return f"{first} - {second}"

def encode_clinical_data(df):
    encoded_df = df.copy()
    patient_ids = encoded_df['Patient ID'].copy().reset_index(drop=True)
    encoded_df = encoded_df.drop(columns=['Patient ID'])

    if encoded_df.shape[0] >= 4:
        sample_col = encoded_df.columns[0]
        first_rows = encoded_df.loc[0:3, sample_col].tolist()

        if any(isinstance(val, str) and '=' in str(val) for val in first_rows):
            encoded_df = encoded_df.iloc[3:].reset_index(drop=True)

    target_col = None
    for col in encoded_df.columns:
        if "Recurrence event" in col:
            target_col = col
            target_values = encoded_df[target_col].copy()
            break

    all_columns = encoded_df.columns.tolist()

    for col in all_columns:
        if col == target_col:
            continue
        try:
            if encoded_df[col].dtype == 'object':
                encoded_df[col] = encoded_df[col].fillna("MISSING")
                encoded_df[col] = encoded_df[col].astype(str)
                le = LabelEncoder()
                encoded_df[col] = le.fit_transform(encoded_df[col])
                print(f"  Encoded {len(le.classes_)} unique values")

            else:
                if encoded_df[col].isna().any():
                    if encoded_df[col].isna().all():
                        encoded_df[col] = 0
                    else:
                        median = encoded_df[col].median()
                        encoded_df[col] = encoded_df[col].fillna(median)

                if encoded_df[col].std() > 0:
                    mean_val = encoded_df[col].mean()
                    std_val = encoded_df[col].std()
                    encoded_df[col] = (encoded_df[col] - mean_val) / std_val

        except Exception as e:
            try:
                encoded_df[col] = encoded_df[col].fillna("MISSING")
                encoded_df[col] = encoded_df[col].astype(str)
                le = LabelEncoder()
                encoded_df[col] = le.fit_transform(encoded_df[col])

            except Exception as e2:
                encoded_df[col] = 0

    if target_col and 'target_values' in locals():
        encoded_df[target_col] = target_values

    if encoded_df.isna().any().any():
        nan_cols = encoded_df.columns[encoded_df.isna().any()].tolist()
        encoded_df = encoded_df.fillna(0)

    encoded_df['Patient ID'] = patient_ids

    return encoded_df

def load_dicom_series(series_dir, target_size=(224, 224)):
    slices = []
    for file in sorted(os.listdir(series_dir)):
        if file.endswith('.dcm'):
            dcm = pydicom.dcmread(os.path.join(series_dir, file))
            slices.append(dcm.pixel_array.astype(np.float32))
    volume = np.stack(slices, axis=0)

    image = np.mean(volume, axis=0)
    image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-5)

    image = torch.tensor(image).unsqueeze(0)  # (1, H, W)
    image = torch.nn.functional.interpolate(image.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False)
    return image.squeeze(0)  # (1, H, W)

def load_nrrd_mask(nrrd_path):
    if not os.path.exists(nrrd_path):
        return None
    image = sitk.ReadImage(nrrd_path)
    array = sitk.GetArrayFromImage(image)  # shape: (Z, H, W)
    mask = (array > 0).astype(np.float32)
    if mask.shape[0] > 1:
        mask = np.mean(mask, axis=0)
    else:
        mask = mask[0]
    return torch.tensor(mask).unsqueeze(0)  # shape: (1, H, W)



# Load all models
rnn_model              = load_model(f"models/rnn_model.h5")
#cnn_model_masked       = load_model(f"models/cnn_model_masked.h5")
#cnn_model_nomask       = load_model(f"models/cnn_model_nomask.h5")
fusion_model_masked    = load_model(f"models/fusion_model_masked.h5")
#fusion_model_nomask    = load_model(f"models/fusion_model_nomask.h5")

st.title("Breast Cancer Recurrence Predictor")

# Single-file uploaders
clin_file = st.file_uploader("Upload clinical data (XLSX)", type=["xlsx"], accept_multiple_files=False)
mri_file  = st.file_uploader("Upload MRI scan (DICOM or image)", type=["dcm","png","jpg"], accept_multiple_files=False)

mask_toggle = st.checkbox("Use mask for CNN/Fusion", value=False)

# Display uploaded data
if clin_file:
    # Read clinical Excel, merge headers, encode
    df = pd.read_excel(clin_file, header=[1,2])
    df.columns = [col[0] if 'Unnamed' in col[1] else f"{col[0]} - {col[1]}" for col in df.columns]
    encoded = encode_clinical_data(df)
    st.subheader("Clinical Data Preview")
    st.dataframe(encoded)

if mri_file:
    st.subheader("MRI Preview")
    # If DICOM: use load_dicom_series, else treat as image
    try:
        with open("_temp_mri.dcm", "wb") as f:
            f.write(mri_file.getbuffer())
        image_tensor = load_dicom_series("_temp_mri.dcm")
        st.image(image_tensor.numpy().transpose(1,2,0), caption="MRI Slice", use_column_width=True)
    except Exception:
        st.image(mri_file, caption="Uploaded Image", use_column_width=True)

# Run prediction
if st.button("Predict Recurrence"):
    use_clin = clin_file is not None
    use_img  = mri_file is not None
    
    # Prepare inputs
    clin_X, img_X = None, None
    if use_clin:
        clin_X = encoded.drop(columns=[col for col in encoded.columns if 'Patient ID' in col], errors='ignore').values.reshape(1,-1)
    if use_img:
        img_X = image_tensor.unsqueeze(0).numpy()
    
    # Model selection based on uploaded modalities and mask toggle
    if use_clin and use_img:
        model = fusion_model_masked if mask_toggle else fusion_model_nomask
        pred = model.predict([img_X, clin_X])
    elif use_clin:
        model = rnn_model
        pred = model.predict(clin_X)
    elif use_img:
        model = cnn_model_masked if mask_toggle else cnn_model_nomask
        pred = model.predict(img_X)
    else:
        st.error("Please upload at least one modality.")
        st.stop()
    
    # Display result
    label = 'Recurrence' if pred[0][0] >= 0.5 else 'No Recurrence'
    st.success(f"Prediction: **{label}** (prob: {pred[0][0]:.2f})")
