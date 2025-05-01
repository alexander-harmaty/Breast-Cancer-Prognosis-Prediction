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

# Helper functions copied from previous scripts
def merge_headers(col_tuple):
    # Unpack the tuple: first-level and second-level names
    first, second = col_tuple

    # If newline characters exist, remove and replace with space
    if isinstance(first, str):
        first = first.replace('\n', ' ').strip()
    if isinstance(second, str):
        second = second.replace('\n', ' ').strip()

    # If blank second-headers exist, return first-header only
    if not second or 'Unnamed' in second:
        return first
    # Otherwise, return merged header
    else:
        return f"{first} - {second}"

def load_dicom_series(series_dir, target_size=(224, 224)):
    slices = []
    for file in sorted(os.listdir(series_dir)):
        if file.endswith('.dcm'):
            dcm = pydicom.dcmread(os.path.join(series_dir, file))
            slices.append(dcm.pixel_array.astype(np.float32))
    volume = np.stack(slices, axis=0)
    image = np.mean(volume, axis=0)
    image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-5)
    image = torch.tensor(image).unsqueeze(0)
    image = torch.nn.functional.interpolate(image.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False)
    return image.squeeze(0)

def load_nrrd_mask(nrrd_path):
    if not os.path.exists(nrrd_path):
        return None
    image = sitk.ReadImage(nrrd_path)
    array = sitk.GetArrayFromImage(image)
    mask = (array > 0).astype(np.float32)
    if mask.shape[0] > 1:
        mask = np.mean(mask, axis=0)
    else:
        mask = mask[0]
    return torch.tensor(mask).unsqueeze(0)

def encode_clinical_data(uploaded_file):
    df = pd.read_excel(uploaded_file)

    # Merge multi-index headers if they exist
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [merge_headers(col) for col in df.columns]

    # Remove metadata rows if needed
    if df.shape[0] >= 4:
        sample_col = df.columns[0]
        if any(isinstance(val, str) and '=' in str(val) for val in df.loc[0:3, sample_col]):
            df = df.iloc[3:].reset_index(drop=True)

    # Drop Patient ID and target column if they exist
    if 'Patient ID' in df.columns:
        df = df.drop(columns=['Patient ID'])

    target_col = None
    for col in df.columns:
        if "Recurrence event" in col:
            target_col = col
            df = df.drop(columns=[target_col])
            break

    # Encode categorical and fill missing values
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna("MISSING").astype(str)
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
        else:
            if df[col].isna().any():
                if df[col].isna().all():
                    df[col] = 0
                else:
                    df[col] = df[col].fillna(df[col].median())
            if df[col].std() > 0:
                df[col] = (df[col] - df[col].mean()) / df[col].std()

    # Ensure shape is correct for RNN: (1, 1, 96)
    clinical_array = df.select_dtypes(include=[np.number]).values.astype('float32')
    clinical_array = np.expand_dims(clinical_array, axis=(0, 1))  # shape: (1, 1, 96)

    return clinical_array, df  # return both the array and the raw df for preview

# Model paths
model_dir = 'models'
rnn_model = load_model(os.path.join(model_dir, 'rnn_model.h5'))
fusion_model = load_model(os.path.join(model_dir, 'fusion_model_masked.h5'))
cnn_model_path = os.path.join(model_dir, 'cnn_model_masked.h5')
cnn_model_available = os.path.exists(cnn_model_path)
cnn_model = load_model(cnn_model_path) if cnn_model_available else None

st.title('Breast Cancer Recurrence Prediction')

# File uploader widgets
clinical_file = st.file_uploader("Upload Clinical Data (Excel file)", type=['xlsx'])
mri_file = st.file_uploader("Upload MRI Image (DICOM)", type=['dcm'])

# Display uploaded files
if clinical_file:
    st.write("Clinical data uploaded:")
    clinical_array, preview_df = encode_clinical_data(clinical_file)
    st.dataframe(preview_df)

if mri_file:
    st.write("MRI image uploaded:")
    dcm = pydicom.dcmread(mri_file)
    st.image(dcm.pixel_array, caption="Uploaded MRI Image", use_column_width=True)

# Prediction logic
if st.button("Predict Recurrence"):
    if clinical_file and mri_file:
        if cnn_model_available:
            mri_data = load_dicom_series(os.path.dirname(mri_file.name))
            reshaped_data = clinical_array.reshape((1, 1, -1))
            clinical_pred = rnn_model.predict(clinical_array)
            mri_pred = cnn_model.predict(np.array([mri_data]))
            combined_pred = fusion_model.predict(np.concatenate([clinical_pred, mri_pred], axis=1))
            prediction = 'Recurrence' if combined_pred[0][0] > 0.5 else 'No Recurrence'
            st.write(f"Fusion Model Prediction: {prediction}")
        else:
            st.error("CNN model is not yet available for MRI predictions. Please upload only clinical data or wait for the model release.")

    elif clinical_file:
        reshaped_data = clinical_array.reshape((1, 1, -1))
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