# Arrhythmia Detection using CNN-Swin Transformer

This project implements an ECG Arrhythmia Detection system using a hybrid Deep Learning model combining **1D CNNs** and **Swin Transformer** blocks. It includes a training pipeline using the MIT-BIH Arrhythmia Database and a Streamlit web application for real-time inference.

## Project Structure

- **`app.py`**: The Streamlit web application for user interaction.
- **`train_model.py`**: Script to download data, preprocess it, train the model, and save the best weights.
- **`model_utils.py`**: Contains the model definition (`create_model`) and the custom `SwinTransformerBlock` layer.
- **`best_cnn_swin_model.keras`**: The trained model file used by the app.
- **`incart/`, `mit-bih-arrhythmia-database-1.0.0/`**: Data directories.

## How to Run

### 1. Prerequisites
Ensure you have Python installed. Install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Training the Model (Optional)
If you need to retrain the model or don't have the `.keras` file yet:

```bash
python train_model.py
```
*This will automatically download the MIT-BIH database if missing, process the data, and train the model.*

### 3. Running the Web App
To start the user interface:

```bash
streamlit run app.py
```
The app will open in your default web browser.

## Input & Output

### Input
The application accepts **CSV files** via the sidebar uploader. 

**Format Options:**
1.  **Raw Signal (Single Column)**:
    - A CSV with a single column of continuous ECG signal values.
    - The app will automatically detect R-peaks, segment the beats, and classify them.
2.  **Segmented Beats (Rows)**:
    - A CSV where each row represents a single heartbeat segment.
    - Expected length: **200 samples** per row.

### Output
The application provides:
- **Visualizations**: Plots of the input raw signal and identified arrhythmia beats.
- **Predictions**: A table showing the classification for each detected beat:
    - `Normal`
    - `Arrhythmia`
    - `Unclassifiable`
- **Confidence**: The probability score for the prediction.

## Model Architecture
- **Input**: 200 raw ECG samples per beat.
- **CNN Layers**: Two 1D Convolutional layers for local feature extraction.
- **Swin Transformer**: A custom Transformer block with distinct windowed attention for global context.
- **Classification Head**: Dense layers with Dropout leading to a Softmax output.
