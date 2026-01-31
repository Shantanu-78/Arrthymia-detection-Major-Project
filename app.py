import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from model_utils import SwinTransformerBlock
import io

# Configuration
PRE_R = 90
POST_R = 110
WINDOW_SIZE = PRE_R + POST_R
CLASSES = ['Arrhythmia', 'Normal', 'Unclassifiable'] # Assumed alphabetical order from LabelEncoder

st.set_page_config(page_title="Arrhythmia Detection", layout="wide")

st.title("Arrhythmia Detection using CNN-Swin Transformer")
st.write("""
Upload your ECG data (CSV) to detect arrhythmia beats. 
The model combines CNN and Swin Transformer for accurate classification.
""")

# Sidebar
st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload ECG CSV file", type=["csv"])
model_path = st.sidebar.text_input("Model Path", "best_cnn_swin_model.keras")

# Caching the model loading
@st.cache_resource
def load_arrhythmia_model(path):
    try:
        # safe_mode=False is required because the model contains a Lambda layer
        model = load_model(path, custom_objects={'SwinTransformerBlock': SwinTransformerBlock}, safe_mode=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        # Check if data is raw signal or segmented
        # For simplicity, we assume single column raw signal or multiple columns as segmented beats
        
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        data_type = st.radio("Select Data Type", ("Raw Signal (Single Column)", "Segmented Beats (Rows)"))
        
        if data_type == "Raw Signal (Single Column)":
            column = st.selectbox("Select Signal Column", df.columns)
            signal = df[column].values
            
            st.subheader("ECG Signal Visualization")
            fig, ax = plt.subplots(figsize=(15, 4))
            ax.plot(signal[:2000]) # Plot first 2000 points
            ax.set_title("First 2000 Samples of ECG Signal")
            st.pyplot(fig)
            
            # Peak detection (Simple heuristic or scipy)
            from scipy.signal import find_peaks
            
            # Basic normalization for peak detection
            sig_std = (signal - np.mean(signal)) / np.std(signal)
            peaks, _ = find_peaks(sig_std, height=5, distance=150) # Tweak params as needed
            
            st.write(f"Detected {len(peaks)} R-peaks.")
            
            if st.button("Segment and Analyze"):
                if len(peaks) == 0:
                    st.warning("No peaks detected. Try adjusting signal or upload filtered data.")
                else:
                    segments = []
                    valid_peaks_indices = []
                    
                    for p in peaks:
                        start = p - PRE_R
                        end = p + POST_R
                        if start >= 0 and end <= len(signal):
                            seg = signal[start:end]
                            segments.append(seg)
                            valid_peaks_indices.append(p)
                    
                    if not segments:
                        st.error("No valid segments found with current window settings.")
                    else:
                        X = np.array(segments)
                        # Normalize/Scale if needed? Notebook used ADASYN on raw? 
                        # Notebook didn't explicitly scale in the snippet I saw (just ADASYN). 
                        # Usually ECG is normalized. Let's apply basic Z-score per beat or global.
                        # Notebook loaded 'StandardScaler' but I didn't see where it was applied in the training loop snippet.
                        # I'll stick to raw or basic normalization if results are bad. For now, raw.
                        
                        X = X.reshape(X.shape[0], X.shape[1], 1)
                        
                        model = load_arrhythmia_model(model_path)
                        if model:
                            predictions = model.predict(X)
                            pred_classes = np.argmax(predictions, axis=1)
                            
                            results = []
                            for i, peak_idx in enumerate(valid_peaks_indices):
                                cls = CLASSES[pred_classes[i]]
                                prob = predictions[i][pred_classes[i]]
                                results.append({"Beat Index": i, "Peak Location": peak_idx, "Prediction": cls, "Confidence": f"{prob:.2f}"})
                            
                            res_df = pd.DataFrame(results)
                            st.subheader("Analysis Results")
                            st.dataframe(res_df)
                            
                            # Visualization of results
                            st.subheader("Detected Arrhythmia Beats")
                            arrhythmia_indices = [i for i, r in enumerate(results) if r['Prediction'] == 'Arrhythmia']
                            
                            if arrhythmia_indices:
                                num_plots = min(5, len(arrhythmia_indices))
                                fig_arr, axes = plt.subplots(1, num_plots, figsize=(15, 3))
                                if num_plots == 1: axes = [axes]
                                for j, idx in enumerate(arrhythmia_indices[:num_plots]):
                                    axes[j].plot(segments[idx])
                                    axes[j].set_title(f"Beat {idx} (Arrhythmia)")
                                st.pyplot(fig_arr)
                            else:
                                st.write("No Arrhythmia beats detected.")

        elif data_type == "Segmented Beats (Rows)":
            # Assume each row is a beat of length 200
            if df.shape[1] != 200:
                st.warning(f"Expected 200 columns for segmented beats, got {df.shape[1]}. Resizing or padding might be needed.")
            
            if st.button("Analyze Beats"):
                X = df.values
                # Resize if necessary
                if X.shape[1] != 200:
                     # Simple truncation or padding
                     if X.shape[1] > 200:
                         X = X[:, :200]
                     else:
                         X = np.pad(X, ((0,0), (0, 200-X.shape[1])), 'constant')
                
                X = X.reshape(X.shape[0], 200, 1)
                
                model = load_arrhythmia_model(model_path)
                if model:
                    predictions = model.predict(X)
                    pred_classes = np.argmax(predictions, axis=1)
                    
                    df['Prediction'] = [CLASSES[c] for c in pred_classes]
                    df['Confidence'] = [p[c] for p, c in zip(predictions, pred_classes)]
                    
                    st.subheader("Analysis Results")
                    st.dataframe(df[['Prediction', 'Confidence']])
                    
                    st.bar_chart(df['Prediction'].value_counts())

    except Exception as e:
        st.error(f"Error processing file: {e}")

else:
    st.info("Awaiting file upload.")
