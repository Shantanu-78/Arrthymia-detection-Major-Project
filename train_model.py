import os
import numpy as np
import wfdb
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import ADASYN
from model_utils import create_model
import collections

# Configuration
# Path to the database. Adjust if necessary to point to the actual data directory.
# Data is in "d:/project/major project/mit-bih-arrhythmia-database-1.0.0/"
DB_DIR = "d:/project/major project/mit-bih-arrhythmia-database-1.0.0/"
BATCH_SIZE = 32
EPOCHS = 10 # Reduced from 20 for faster execution in this demo, adjust as needed
PRE_R = 90
POST_R = 110


# Records to process
RECORDS = [
    '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
    '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
    '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
    '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
    '222', '223', '228', '230', '231', '232', '233', '234'
]

ARRHYTHMIA_MAP = {
    'N': 'Normal',
    'L': 'Arrhythmia', 'R': 'Arrhythmia', 'B': 'Arrhythmia', 'A': 'Arrhythmia',
    'a': 'Arrhythmia', 'J': 'Arrhythmia', 'S': 'Arrhythmia', 'V': 'Arrhythmia',
    'r': 'Arrhythmia', 'F': 'Arrhythmia', 'e': 'Arrhythmia', 'j': 'Arrhythmia',
    'n': 'Arrhythmia', 'E': 'Arrhythmia', '/': 'Arrhythmia',
    'Q': 'Unclassifiable', 'f': 'Unclassifiable'
}

def download_data_if_needed(db_dir, records):
    os.makedirs(db_dir, exist_ok=True)
    missing_records = []
    for rec in records:
        if not (os.path.exists(os.path.join(db_dir, f"{rec}.dat")) and
                os.path.exists(os.path.join(db_dir, f"{rec}.hea")) and
                os.path.exists(os.path.join(db_dir, f"{rec}.atr"))):
            missing_records.append(rec)
    
    if missing_records:
        print(f"Downloading missing records: {missing_records}")
        try:
            wfdb.dl_database('mitdb', records=missing_records, dl_dir=db_dir)
            print("Download complete.")
        except Exception as e:
            print(f"Error downloading data: {e}")
            # Fallback or exit if critical
            pass
    else:
        print("All records present.")

def load_and_preprocess_data(db_dir, records):
    segmented_beats = []
    beat_labels = []
    
    print("Processing records...")
    for record_name in records:
        try:
            record_path = os.path.join(db_dir, record_name)
            if not os.path.exists(record_path + ".dat"):
                print(f"Skipping {record_name}, file not found.")
                continue
                
            record = wfdb.rdrecord(record_path)
            annotation = wfdb.rdann(record_path, 'atr')
            
            if 'MLII' in record.sig_name:
                signal = record.p_signal[:, record.sig_name.index('MLII')]
            elif 'II' in record.sig_name:
                signal = record.p_signal[:, record.sig_name.index('II')]
            else:
                signal = record.p_signal[:, 0]
            
            r_peaks = []
            beat_types = []
            for i, symbol in enumerate(annotation.symbol):
                if symbol in ARRHYTHMIA_MAP:
                    r_peaks.append(annotation.sample[i])
                    beat_types.append(symbol)
            
            for i, r_peak_idx in enumerate(r_peaks):
                start_idx = r_peak_idx - PRE_R
                end_idx = r_peak_idx + POST_R
                
                if start_idx < 0: start_idx = 0
                if end_idx > len(signal): end_idx = len(signal)
                
                segment = signal[start_idx:end_idx]
                
                expected_len = PRE_R + POST_R
                if len(segment) < expected_len:
                    padding = expected_len - len(segment)
                    if start_idx == 0:
                        segment = np.pad(segment, (0, padding), 'constant')
                    else:
                        segment = np.pad(segment, (padding, 0), 'constant')
                
                if len(segment) == expected_len:
                    segmented_beats.append(segment)
                    beat_labels.append(ARRHYTHMIA_MAP[beat_types[i]])
                    
        except Exception as e:
            print(f"Error processing record {record_name}: {e}")
            continue

    print(f"Total beats: {len(segmented_beats)}")
    return np.array(segmented_beats), np.array(beat_labels)

def main():
    # 1. Prepare Data
    download_data_if_needed(DB_DIR, RECORDS)
    
    X, y = load_and_preprocess_data(DB_DIR, RECORDS)
    
    if len(X) == 0:
        print("No data loaded. Exiting.")
        return

    print("Class distribution before resampling:", collections.Counter(y))
    
    # 2. ADASYN Resampling
    print("Applying ADASYN...")
    try:
        adasyn = ADASYN(random_state=42)
        X_resampled, y_resampled = adasyn.fit_resample(X, y)
        print("Class distribution after resampling:", collections.Counter(y_resampled))
    except ValueError as e:
        print(f"ADASYN failed (possibly rare classes): {e}. Proceeding with original data.")
        X_resampled, y_resampled = X, y

    # 3. Encoding
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_resampled)
    y_one_hot = to_categorical(y_encoded)
    
    classes = le.classes_
    print(f"Classes: {classes}")
    
    # Reshape for CNN (Batch, Steps, Channels)
    X_reshaped = X_resampled.reshape(X_resampled.shape[0], X_resampled.shape[1], 1)
    
    # 4. Split
    X_train, X_temp, y_train, y_temp = train_test_split(X_reshaped, y_one_hot, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # 5. Create and Train Model
    input_shape = (X_train.shape[1], 1)
    num_classes = y_one_hot.shape[1]
    
    model = create_model(input_shape, num_classes)
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Starting training...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint('best_cnn_swin_model.keras', save_best_only=True)
        ]
    )
    
    # 6. Evaluation
    print("Evaluating on test set...")
    loss, acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {acc:.4f}")
    
    print("Saving final model...")
    model.save('final_cnn_swin_model.keras')
    print("Done.")

if __name__ == "__main__":
    main()
