# Batch Feature Extraction with Directory Check
import os
import numpy as np
import librosa
import pandas as pd

DATA_DIR = 'ravdess/'  # Use RAVDESS dataset folder
features = []
labels = []

# Ensure the audio directory exists (auto-create if missing)
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    print(f"Directory '{DATA_DIR}' was missing and has been created.")
    print("Please add .wav or .mp3 files to this folder and rerun the script.")
else:
    audio_files = []
for root, dirs, files in os.walk(DATA_DIR):
    for file in files:
        if file.endswith('.wav') or file.endswith('.mp3'):
            audio_files.append(os.path.join(root, file))

if not audio_files:
    print(f"No .wav or .mp3 files found in '{DATA_DIR}'. Please add audio files to proceed.")
else:
    processed_files = []
    error_count = 0
    first_error = None
    for file in audio_files:
        try:
            # Use the full path for librosa
            y, sr = librosa.load(file, sr=None)
            mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
            chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
            contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
            feature_vec = np.hstack([mfccs, chroma, contrast])
            features.append(feature_vec)
            labels.append(os.path.basename(file).split('_')[0])
            processed_files.append(file)
        except Exception as e:
            error_count += 1
            if first_error is None:
                first_error = (file, str(e))
    if len(features) == len(labels) == len(processed_files) and len(features) > 0:
        X = np.array(features)
        y = np.array(labels)
        np.save('features.npy', X)
        np.save('labels.npy', y)
        pd.DataFrame({'filename': processed_files, 'label': y}).to_csv('labels.csv', index=False)
        print(f"Extracted features from {len(processed_files)} files. Saved as features.npy, labels.npy, labels.csv.")
        print(f"First 5 processed files: {processed_files[:5]}")
        print(f"First 5 labels: {y[:5]}")
        if error_count > 0:
            print(f"Encountered {error_count} errors during processing.")
            print(f"First error: File: {first_error[0]} | Error: {first_error[1]}")
    else:
        print(f"No valid audio files were processed. Please check your dataset.")
        if error_count > 0 and first_error is not None:
            print(f"First error: File: {first_error[0]} | Error: {first_error[1]}")
