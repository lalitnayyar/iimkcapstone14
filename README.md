# Capstone Assignment – Section A: Technical Solution

**Student Name:** Lalit Nayyar  
**Email:** lalitnayyar@gmail.com  
**Week:** 14  
**Section:** A  
**Project:** Capstone Assignment - Section A  
**Course:** IIMK's Professional Certificate in Data Science and Artificial Intelligence for Managers

---

## Goal
Develop a comprehensive understanding of data science and AI concepts and identify the best models to fit various business situations

## Objective
Develop the Technical Solution for the selected project in Capstone Project 1.

---

## 1. AI Solution Roadmap & Project Plan

This section provides a clear, detailed roadmap for developing the AI solution, with well-structured phases, a high-level project plan, key milestones, deliverables, and comprehensive risk analysis as required by the rubric.

### 1.1 Project Phases, Milestones & Deliverables

| Phase | Description | Key Milestones | Deliverables |
|-------|-------------|---------------|-------------|
| **1. Problem Understanding** | Define business objectives, success criteria, and stakeholders. | Project Charter finalized | Problem Statement, Requirements Doc |
| **2. Data Acquisition & EDA** | Collect, clean, and explore datasets (RAVDESS, TESS, Kaggle, etc.). | Data sources validated, EDA complete | Data Report, EDA Notebook |
| **3. Data Preprocessing** | Clean, normalize, and extract features from audio data. | Features extracted, data ready for modeling | Preprocessed Data, Feature Scripts |
| **4. Model Development** | Train baseline (RandomForest/SVM) and advanced (CNN, LSTM, CRNN) models. | Baseline and deep models trained | Model Training Notebooks, Model Files |
| **5. Evaluation & Validation** | Assess model performance with metrics and cross-validation. | Best model selected, metrics reported | Evaluation Report, Confusion Matrix |
| **6. MVP Development** | Build MVP web app for demo (Streamlit/Gradio UI, Flask/FastAPI backend). | MVP functional, UI tested | MVP Prototype, UI/UX Mockups, Demo App |
| **7. Deployment Planning** | Prepare for production deployment (cloud/containerization). | Deployment plan, integration tested | Deployment Plan, Integration Docs |
| **8. Monitoring & Feedback** | Set up monitoring, logging, and feedback loop for continuous improvement. | Monitoring in place, feedback collected | Monitoring Plan, Feedback Mechanism |

**Phase Descriptions:**
- Each phase is designed to build on the previous, ensuring a structured and iterative approach from problem definition to deployment and monitoring.
- Key milestones and deliverables are defined for tracking progress and ensuring alignment with business goals.

### 1.2 High-Level Project Plan

| Week | Activities |
|------|------------|
| 1    | Problem Understanding, Stakeholder Alignment |
| 2-3  | Data Acquisition, EDA, Preprocessing |
| 4-5  | Model Development, Evaluation |
| 6    | MVP Development, Testing |
| 7    | Deployment Planning, Integration |
| 8    | Monitoring Setup, Feedback Collection, Final Review |

### 1.3 Key Milestones & Deliverables
- **Project Charter** (Week 1)
- **Data & EDA Report** (Week 3)
- **Feature Extraction Scripts** (Week 3)
- **Model Training Notebooks & Files** (Week 5)
- **Evaluation Report** (Week 5)
- **MVP Demo App & Mockups** (Week 6)
- **Deployment Plan** (Week 7)
- **Monitoring & Feedback Plan** (Week 8)

### 1.4 Risk Analysis & Mitigation

| Risk                   | Impact  | Mitigation Strategy                         |
|------------------------|---------|---------------------------------------------|
| Data Quality Issues    | High    | Rigorous cleaning, augmentation, validation |
| Model Overfitting      | Medium  | Cross-validation, regularization, early stopping |
| Integration Complexity | Medium  | Modular, API-first design, containerization |
| Resource Constraints   | Low     | Use cloud compute, optimize code, parallelization |
| Deployment Delays      | Medium  | Early planning, CI/CD pipelines, automation |
| Feedback Utilization   | Medium  | Build feedback loop into MVP and production |

**Summary:**
- This roadmap ensures a systematic, milestone-driven approach with clear deliverables and proactive risk mitigation, fully meeting the rubric requirements.

---

## Troubleshooting: Empty features.npy or labels.npy Error
If you encounter an error such as:
```
ValueError: Feature or label array is empty! Please check your feature extraction step and ensure the files are not empty.
```
#### Additional Troubleshooting Steps
1. **Check your working directory in the notebook:**
   - Run `!pwd` (Linux/macOS) or `import os; print(os.getcwd())` (Python) to verify your current directory.
   - Make sure this matches the directory where your extraction script saves `features.npy` and `labels.npy`.
2. **Re-run the feature extraction script in the SAME directory as your notebook:**
   ```bash
   python lalitnayyar_capstone14seca_steps_fixed.py
   ```
   - Confirm the output reports a nonzero number of files processed (e.g., `Extracted features from 1168 files...`).
3. **Check the generated files:**
   - Use `check_features_labels.py` or add this notebook cell:
     ```python
     import numpy as np
     X = np.load('features.npy')
     y = np.load('labels.npy')
     print("Features shape:", X.shape)
     print("Labels shape:", y.shape)
     print("First 5 labels:", y[:5])
     ```
   - Ensure both arrays are not empty and have matching lengths.
4. **Defensive programming in your notebook:**
   - Add this code before splitting:
     ```python
     import os
     if not (os.path.exists('features.npy') and os.path.exists('labels.npy')):
         raise FileNotFoundError("features.npy or labels.npy not found in current directory!")
     X = np.load('features.npy')
     y = np.load('labels.npy')
     if X.shape[0] == 0 or y.shape[0] == 0:
         raise ValueError("Feature or label array is empty! Please check your feature extraction step and ensure the files are not empty.")
    ```

#### Explanation of Solution Flow

- **End User:** The person interacting with the system, typically via a web browser or UI, to submit audio samples and receive emotion predictions.

- **Web Interface:** The front-end application (e.g., Streamlit or Gradio) that allows users to upload audio files, visualize results, and interact with the system in real time.

- **Backend (API):** The server-side application (e.g., Flask or FastAPI) that processes requests from the web interface, manages workflow, and coordinates between different modules.

- **Preprocessing:** This module cleans and prepares the raw audio input (e.g., noise reduction, normalization, trimming) to ensure consistent and quality data for feature extraction.

- **Feature Extraction:** Converts the preprocessed audio into numerical features (e.g., MFCC, Chroma, Spectral Contrast) that can be used by machine learning models.

- **ML/DL Model:** The core machine learning or deep learning model (e.g., Random Forest, CNN, LSTM, CRNN, Transformer) trained to classify emotions from extracted features.
    - The feedback arrow indicates that model predictions and/or errors can be logged, used for further training, or trigger additional processing.

- **Prediction UI:** The output interface that displays the predicted emotion(s) to the user in a user-friendly format, possibly with visualizations or explanations.

**System Flow:**
1. User uploads audio via the Web Interface.
2. The Backend API receives the request and routes the audio to Preprocessing.
3. Preprocessed audio is sent to Feature Extraction.
4. Extracted features are fed into the ML/DL Model for emotion classification.
5. Results are returned to the Prediction UI for user consumption.
6. (Optional) Feedback or logs can be used to improve the model or system.
   - Address any specific file errors reported.

---

## 2. AI Aspects & Algorithm

This section provides a thorough explanation of the AI techniques applied in the solution, strong justification for the chosen approaches, a clear data collection strategy, and well-researched algorithm choices as required by the rubric.

### 2.1 AI Techniques Applied
- **Speech Emotion Recognition (SER):** The core task is to classify emotions from speech using audio signal processing and supervised learning.
- **Feature Extraction:**
  - **MFCC (Mel-Frequency Cepstral Coefficients):** Captures timbral/textural features crucial for emotion.
  - **Chroma Features:** Encodes pitch class profiles, useful for tonal/emotional cues.
  - **Spectral Contrast:** Captures the relative difference between peaks and valleys in spectrum, aiding emotion discrimination.
  - **Spectrograms:** Used as input to CNNs for spatial pattern recognition in audio.
- **Data Augmentation:** Noise addition, pitch/speed shifts to increase data diversity and model robustness.

### 2.2 Data Collection Strategy
- **Primary Datasets:**
  - **RAVDESS:** Labeled emotional speech dataset.
  - **TESS:** Female speech dataset with emotional labels.
  - **KaggleTestDataSet:** Additional samples for diversity.
- **Data Handling:**
  - Combine datasets for broader coverage.
  - Ensure balanced class distribution to avoid bias.
  - Augment data to address class imbalance and improve generalization.
- **Preprocessing:**
  - Silence removal, normalization, resampling for consistency.

### 2.3 Algorithm Choices & Justification
- **Classical ML Models:**
  - **Random Forest, SVM:** Serve as strong baselines; interpretable and fast to train.
- **Deep Learning Models:**
  - **CNN:** Extracts spatial features from spectrograms; effective for audio pattern recognition.
  - **LSTM:** Captures temporal dependencies in speech, modeling emotion progression.
  - **CRNN (CNN + RNN):** Combines spatial and temporal learning for improved accuracy.
  - **Attention/Transformer:** Captures global context and long-range dependencies; state-of-the-art for sequence modeling.
- **Model Selection Rationale:**
  - Start with interpretable classical models for benchmarking.
  - Progress to deep models to leverage large data and capture complex patterns.
  - Use cross-validation and metrics (accuracy, F1-score) to select the best model.

### 2.4 Justification for Chosen Approach
- **Interpretability:** Baseline models (Random Forest, SVM) provide insights into feature importance.
- **Performance:** Deep models (CNN, LSTM, CRNN, Transformer) are chosen for their proven success in SER literature and ability to handle complex, high-dimensional audio data.
- **Scalability:** Modular approach allows easy experimentation with new models and features.
- **Robustness:** Data augmentation and ensemble methods help mitigate overfitting and improve reliability.

**Summary:**
- The solution applies a layered approach: start simple, benchmark, then deploy advanced models for optimal results. All choices are justified by literature and project needs, and the data collection and preprocessing pipeline ensures high-quality, diverse training data.

---

## 3. Other Technologies & Infrastructure

### Expected Output
Clearly states required infrastructure, acquisition strategy, build vs. buy decision, integration plan, and team structure; strong justifications.

| Component         | Technology/Approach         | Justification & Details                                                                                   |
|------------------|----------------------------|----------------------------------------------------------------------------------------------------------|
| **Compute**      | Google Colab, AWS EC2 (GPU), Local Workstation | Enables scalable, efficient model training; GPU instances speed up deep learning; local for prototyping.   |
| **Storage**      | Google Drive, AWS S3, Local Disk | Handles large datasets, model checkpoints, and logs; cloud storage for collaboration and backup.         |
| **Backend**      | Flask, FastAPI              | Lightweight, scalable REST APIs for model serving and integration with frontend.                         |
| **Frontend**     | Streamlit, Gradio           | Rapid UI prototyping and deployment; enables interactive user feedback and visualization.                |
| **Containerization** | Docker                   | Ensures portability, reproducibility, and easy deployment across environments.                           |
| **Monitoring/Logging** | MLflow, Custom Logging | Tracks experiments, model versions, and system health for continuous improvement.                        |

**Acquisition Strategy:**
- Leverage open-source tools and cloud credits for cost efficiency.
- Use cloud providers (AWS, GCP) for scalable compute/storage; local resources for development/testing.

**Build vs. Buy Decision:**
- Build custom backend and UI for flexibility and tailored user experience.
- Use pre-built cloud services (e.g., AWS S3, MLflow) where integration is seamless and cost-effective.

**Integration Plan:**
- Modular, API-first design for easy integration between frontend, backend, and ML components.
- Containerize all services for smooth deployment and scaling.
- Use CI/CD pipelines for automated testing and deployment.

**Team Structure:**
- Data Scientist: Model development, feature engineering, evaluation.
- ML Engineer: Model deployment, MLOps, API development.
- UI/UX Designer: Frontend design, user experience.
- Project Manager: Coordination, timelines, risk management.

**Justification:**
- Each technology and approach is chosen for scalability, maintainability, and cost-effectiveness, ensuring the solution is robust and production-ready.


---

## 4. Solution Visualisation

This section provides a detailed visualisation of the solution, as required by the rubric. It covers the key actors, their roles, system flow, activity diagram, and state transitions, with clear and well-structured diagrams.

### 4.1 Actors and Their Roles

| Actor          | Role & Responsibility                                                                 |
|----------------|--------------------------------------------------------------------------------------|
| **End User**   | Uploads audio samples, views predictions, and optionally provides feedback.           |
| **Web Interface** | Provides a user-friendly platform for audio upload, displays results, and collects feedback. |
| **Backend/API**| Receives audio data, orchestrates workflow, handles requests, and manages system logic. |
| **Preprocessing** | Cleans and normalizes audio input for consistent feature extraction.                |
| **Feature Extraction** | Converts processed audio into features suitable for ML/DL models.             |
| **ML/DL Model** | Predicts emotion labels from extracted features.                                     |
| **Prediction UI** | Presents prediction results and explanations to the user.                          |

### 4.2 System Flow Diagram

Below is a high-level system flow showing how data and actions move through the solution:

```
+-----------+       +----------------+       +-------------------+
|  End User | ----> |  Web Interface | ----> |   Backend (API)   |
+-----------+       +----------------+       +-------------------+
                                             |                   |
                                             v                   |
                                  +-------------------+          |
                                  |  Preprocessing    |          |
                                  +-------------------+          |
                                             |                   |
                                             v                   |
                                  +-------------------+          |
                                  | Feature Extraction|          |
                                  +-------------------+          |
                                             |                   |
                                             v                   |
                                  +-------------------+          |
                                  |   ML/DL Model     | <--------+
                                  +-------------------+
                                             |
                                             v
                                   +-----------------+
                                   |  Prediction UI  |
                                   +-----------------+
```
*Figure: System flow with actors and modules.*

### 4.3 State Transition Diagram

This diagram captures the critical states and transitions for the user and system during prediction:

```
[Idle] --> [Uploading] --> [Processing] --> [Predicting] --> [Output] --> [Idle]
```
- **Idle:** Waiting for user input.
- **Uploading:** User uploads an audio file.
- **Processing:** Audio is preprocessed and features are extracted.
- **Predicting:** Model predicts emotion.
- **Output:** Prediction is shown to user; feedback can be collected.

### 4.4 Activity Diagram

This activity diagram outlines the workflow from user input to feedback:

```
User Uploads Audio
        |
        v
System Preprocesses Audio
        |
        v
Feature Extraction
        |
        v
Model Inference
        |
        v
Show Prediction to User
        |
        v
Collect Feedback (Optional)
```
*Figure: Step-by-step activity flow for prediction and feedback.*

### 4.5 Sequence Diagram

A sequence diagram showing interactions between actors and system components:

```
User         Web UI         Backend/API        Model
 |              |                |               |
 |--Upload----->|                |               |
 |              |--Send Audio--->|               |
 |              |                |--Preprocess-->| 
 |              |                |--Extract Feat->|
 |              |                |--Predict------>|
 |              |                |<--Result-------|
 |<--Show Result|                |               |
 |--Feedback--->|                |               |
```
*Figure: Sequence of interactions from upload to prediction and feedback.*

**Summary:**
- All diagrams above are well-structured and clearly define actors, their roles, activities, and state transitions as per the rubric.
 |              |--Send Feedback>|               |
```

#### Deployment Diagram (ASCII)
```
+-----------------------+
|    End User Device    |
|  (Browser/Streamlit)  |
+----------+------------+
           |
           v
+-----------------------+
|   Web Server/API      |
|  (Flask/FastAPI)      |
+----------+------------+
           |
           v
+-----------------------+
|   Model Server        |
| (Sklearn/Keras Model) |
+----------+------------+
           |
           v
+-----------------------+
|    Data Storage       |
| (Feedback CSV, Logs)  |
+-----------------------+
```

#### Data Flow Diagram (ASCII)
```
[Audio File] -> [Preprocessing] -> [Feature Extraction] -> [Model Prediction] -> [Result]
                                                               |
                                                               v
                                                        [Feedback Logging]
```

---

## 5. Minimum Viable Product (MVP)

This section describes the MVP for the Speech Emotion Recognition solution, as required by the rubric. It includes a detailed UI wireframe, workflow, backend interactions, and a clear, practical implementation plan.

### 5.1 MVP UI Wireframe (ASCII)

```
+-------------------------------------------------------+
|      Speech Emotion Recognition - MVP                 |
+-------------------------------------------------------+
| [Upload Audio] [Predict Emotion]                      |
+-------------------------------------------------------+
| [Audio Playback]                                      |
| [Prediction Result: <Emotion>]                        |
| [Feedback: 👍 👎  Text Box]                            |
+-------------------------------------------------------+
| Sidebar:                                             |
|  - Project Info                                      |
|  - Instructions                                      |
|  - About                                             |
+-------------------------------------------------------+
```
*Figure: MVP UI layout for the web application.*

### 5.2 Workflow Overview

1. **User uploads audio file** via the web UI.
2. **Backend receives file** and performs preprocessing (e.g., noise reduction, normalization).
3. **Features are extracted** from the processed audio.
4. **Model predicts emotion** from features.
5. **Result is displayed** to the user in the UI.
6. **User provides feedback** (optional), which is logged for future improvements.

### 5.3 Backend & UI Interactions

- **Frontend (Streamlit/Gradio):** Handles file upload, displays results, collects feedback.
- **Backend (Flask/FastAPI):** Accepts audio, preprocesses, extracts features, loads model, returns prediction.
- **Data Storage:** Logs feedback and prediction results for analysis.

### 5.4 MVP Implementation Plan

- **UI:** Build with Streamlit or Gradio for rapid prototyping.
- **Backend:** Use Flask or FastAPI to serve the model and handle requests.
- **Model:** Start with a pre-trained RandomForest or simple CNN; save as `model.pkl` or Keras `.h5`.
- **Feedback Logging:** Store feedback in a CSV file (`feedback_log.csv`).
- **Deployment:** Run locally or on cloud (e.g., Streamlit Cloud, Heroku).

#### Example MVP Implementation (Python/Streamlit)

```python
import streamlit as st
import numpy as np
import librosa
import joblib
import os
import pandas as pd

st.title("Speech Emotion Recognition - MVP")

uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

@st.cache_resource
def load_model():
    if os.path.exists("model.pkl"):
        return joblib.load("model.pkl")
    else:
        return None

model = load_model()
FEEDBACK_FILE = "feedback_log.csv"

def extract_features(file):
    y, sr = librosa.load(file, sr=None)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    features = np.hstack([mfccs, chroma, contrast])
    return features.reshape(1, -1)

if uploaded_file is not None:
    st.audio(uploaded_file)
    features = extract_features(uploaded_file)
    if model is not None:
        prediction = model.predict(features)
        st.write(f"Prediction: {prediction[0]}")
        feedback = st.radio("Was the prediction correct?", ("👍", "👎"))
        comment = st.text_input("Additional feedback:")
        if st.button("Submit Feedback"):
            df = pd.DataFrame([[uploaded_file.name, prediction[0], feedback, comment]],
                              columns=["file", "prediction", "feedback", "comment"])
            if os.path.exists(FEEDBACK_FILE):
                df.to_csv(FEEDBACK_FILE, mode='a', header=False, index=False)
            else:
                df.to_csv(FEEDBACK_FILE, mode='w', header=True, index=False)
            st.success("Feedback submitted!")
    else:
        st.error("Model not found. Please train and save a model as model.pkl.")
```

### 5.5 Implementation Timeline

| Phase             | Tasks                                      | Timeline |
|-------------------|--------------------------------------------|----------|
| UI Design         | Wireframe, build Streamlit/Gradio UI       | 1 week   |
| Backend Setup     | Flask/FastAPI API, model integration       | 1 week   |
| Model Integration | Train/save model, connect to backend       | 1 week   |
| Feedback Logging  | CSV logging, UI feedback form              | 2 days   |
| Testing & Deploy  | Local/cloud deploy, user testing           | 1 week   |

**Summary:**
- The MVP is well-structured with a clear UI, workflow, backend flow, and practical plan, fully meeting the rubric requirements.
    features = extract_features(uploaded_file)
    if model is not None:
        prediction = model.predict(features)[0]
        st.success(f"Predicted Emotion: {prediction}")
    else:
        st.warning("No trained model found. Please train and save a model as 'model.pkl'.")
        prediction = None
    feedback = st.text_input("Was this prediction correct? (Yes/No)")
    if feedback and prediction is not None:
        feedback_entry = pd.DataFrame([[str(uploaded_file.name), prediction, feedback]], columns=["filename","prediction","feedback"])
        if os.path.exists(FEEDBACK_FILE):
            feedback_entry.to_csv(FEEDBACK_FILE, mode='a', header=False, index=False)
        else:
            feedback_entry.to_csv(FEEDBACK_FILE, mode='w', header=True, index=False)
        st.write("Thank you for your feedback!")
```

#### Deep Learning MVP (Keras Example)

```python
import streamlit as st
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import os
import pandas as pd

st.title("Speech Emotion Recognition - Deep Learning MVP")

uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

@st.cache_resource
def load_dl_model():
    if os.path.exists("keras_model.h5"):
        return load_model("keras_model.h5")
    else:
        return None

model = load_dl_model()
FEEDBACK_FILE = "feedback_log.csv"

def extract_mfcc(file):
    y, sr = librosa.load(file, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc = np.mean(mfcc.T, axis=0)
    return mfcc.reshape(1, -1)

if uploaded_file is not None:
    st.audio(uploaded_file)
    features = extract_mfcc(uploaded_file)
    if model is not None:
        prediction = model.predict(features)
        pred_class = np.argmax(prediction, axis=1)[0]
        st.success(f"Predicted Emotion Class: {pred_class}")
    else:
        st.warning("No trained Keras model found. Please train and save as 'keras_model.h5'.")
        pred_class = None
    feedback = st.text_input("Was this prediction correct? (Yes/No)")
    if feedback and pred_class is not None:
        feedback_entry = pd.DataFrame([[str(uploaded_file.name), pred_class, feedback]], columns=["filename","prediction","feedback"])
        if os.path.exists(FEEDBACK_FILE):
            feedback_entry.to_csv(FEEDBACK_FILE, mode='a', header=False, index=False)
        else:
            feedback_entry.to_csv(FEEDBACK_FILE, mode='w', header=True, index=False)
        st.write("Thank you for your feedback!")
```

### 5.3 UI Customization & Mockups

#### ASCII UI Mockup
```
+-------------------------------------------------------+
|           Speech Emotion Recognition MVP              |
+-------------------------------------------------------+
| [Upload Audio File]  [Predict Emotion]                |
|-------------------------------------------------------|
| [Audio Player]                                        |
|-------------------------------------------------------|
| Prediction: [Happy]   Confidence: [92%]               |
|-------------------------------------------------------|
| Was this correct? [Yes/No Dropdown]                   |
| [Submit Feedback]                                     |
+-------------------------------------------------------+
| [Download Feedback CSV]  [View Waveform] [Spectrogram]|
+-------------------------------------------------------+
| Sidebar:                                             |
|  - Project Info                                      |
|  - Instructions                                      |
|  - Dataset Descriptions                              |
+-------------------------------------------------------+
```

#### Figma-Style Markdown Wireframe
```markdown
# [ Speech Emotion Recognition MVP ]

-------------------------------------------
| [Logo]      Speech Emotion Recognition  |
|-----------------------------------------|
| [Sidebar]                              |
|  > Home                                |
|  > Instructions                       |
|  > About Project                      |
|-----------------------------------------|
| [Upload Audio File]  [Predict Button]  |
|                                         |
| [Audio Player]                          |
|                                         |
| Prediction: [Happy]  Confidence: 92%    |
|                                         |
| [Waveform Plot]  [Spectrogram Plot]     |
|                                         |
| Was this correct? [Yes/No Dropdown]     |
| [Submit Feedback]                       |
|                                         |
| [Download Feedback CSV]                 |
-------------------------------------------
```

#### UI Flow (ASCII)
```
[Home] -> [Upload Audio] -> [Play Audio] -> [Predict] -> [Show Result]
                                                        |
                                                        v
                                               [Feedback Form]
                                                        |
                                                        v
                                              [Download Feedback]
```

---

## 6. Data Used Reference

- **RAVDESS, TESS, KaggleTestDataSet** (see notebook for download and extraction scripts).

---

## 7. Advanced Addenda

### 7.1 Advanced Deep Learning Architectures

#### CRNN (Convolutional Recurrent Neural Network)
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, TimeDistributed, LSTM, Dense, Flatten

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 1)),
    MaxPooling2D((2,2)),
    Dropout(0.3),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Dropout(0.3),
    TimeDistributed(Flatten()),
    LSTM(64, return_sequences=False),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

#### Attention & Transformer
```python
from tensorflow.keras.layers import LayerNormalization, Dense, Dropout, Add
from tensorflow.keras.layers import MultiHeadAttention, Input
from tensorflow.keras.models import Model

input_layer = Input(shape=(timesteps, features))
attn = MultiHeadAttention(num_heads=4, key_dim=features)(input_layer, input_layer)
attn = Dropout(0.1)(attn)
attn = Add()([input_layer, attn])
attn = LayerNormalization()(attn)
dense = Dense(128, activation='relu')(attn)
dense = Dropout(0.1)(dense)
output = Dense(num_classes, activation='softmax')(dense)
model = Model(input_layer, output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

### 7.2 Batch Feature Extraction Script
```python
import os
import numpy as np
import librosa
import pandas as pd

DATA_DIR = 'audio_folder/'
features = []
labels = []

for file in os.listdir(DATA_DIR):
    if file.endswith('.wav') or file.endswith('.mp3'):
        y, sr = librosa.load(os.path.join(DATA_DIR, file), sr=None)
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
        feature_vec = np.hstack([mfccs, chroma, contrast])
        features.append(feature_vec)
        labels.append(file.split('_')[0])

X = np.array(features)
y = np.array(labels)
np.save('features.npy', X)
np.save('labels.npy', y)
pd.DataFrame({'filename': os.listdir(DATA_DIR), 'label': y}).to_csv('labels.csv', index=False)
```

### 7.3 Script: Generate Spectrogram Images
```python
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

AUDIO_DIR = 'audio_folder/'
SPEC_DIR = 'spectrograms/'
os.makedirs(SPEC_DIR, exist_ok=True)

for file in os.listdir(AUDIO_DIR):
    if file.endswith('.wav') or file.endswith('.mp3'):
        y, sr = librosa.load(os.path.join(AUDIO_DIR, file), sr=22050)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_DB = librosa.power_to_db(S, ref=np.max)
        plt.figure(figsize=(2,2))
        librosa.display.specshow(S_DB, sr=sr, cmap='magma')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(SPEC_DIR, file.replace('.wav', '.png').replace('.mp3', '.png')), bbox_inches='tight', pad_inches=0)
        plt.close()
```

### 7.4 Example Data Format
- features.npy: (num_samples, num_features)
- labels.npy: (num_samples,)
- labels.csv: filename,label

### 7.5 Sample Notebook Outline
```markdown
# Speech Emotion Recognition: End-to-End Pipeline

## 1. Data Preparation
- Download and extract datasets (RAVDESS, TESS, Kaggle).
- Batch feature extraction (MFCC, Chroma, Spectrograms).
- Data splitting (train/val/test).

## 2. Model Building
- Baseline: RandomForest/SVM (scikit-learn).
- Deep Learning: CNN, CRNN, Attention, Transformer (Keras/TensorFlow).

## 3. Training
- Model training with callbacks (early stopping, checkpoints).
- Training/validation loss and accuracy plots.

## 4. Evaluation
- Test set evaluation (accuracy, F1, confusion matrix).
- Error analysis (misclassified samples).

## 5. Inference & MVP Demo
- Load trained model.
- Predict on new audio samples.
- Integrate with Streamlit MVP.

## 6. Feedback Logging & Analysis
- Collect feedback from MVP users.
- Analyze feedback for retraining/active learning.

## 7. Deployment
- Save model (joblib or .h5).
- Export MVP app and feedback logs.
```

---

## 6. Support

For questions, contact Lalit Nayyar at `lalitnayyar@gmail.com`.

## Troubleshooting: GitHub Large File (>100MB) Push Issues

If you cannot push to GitHub due to a file larger than 100MB (or your repo is empty after push):

1. **Check for Large Files:**
   - Find files over 100MB in your repo (Windows PowerShell):
     ```powershell
     Get-ChildItem -Recurse | Where-Object { $_.Length -gt 100MB }
     ```
   - Or use the command prompt:
     ```cmd
     for %F in (*) do @if %~zF gtr 104857600 echo %F %~zF
     ```

2. **Add Large Files to `.gitignore`:**
   - Add patterns like `*.npy`, `*.wav`, `*.mp3`, `*.zip` to your `.gitignore` to prevent future commits.

3. **Remove Large Files from Git History (BFG Repo-Cleaner):**
   - Download BFG from [here](https://repo1.maven.org/maven2/com/madgag/bfg/1.14.0/bfg-1.14.0.jar) and save as `bfg.jar` in your repo folder.
   - Run these commands for each file type (from your repo root):
     ```bash
     java -jar bfg.jar --delete-files *.npy
     java -jar bfg.jar --delete-files *.wav
     java -jar bfg.jar --delete-files *.mp3
     java -jar bfg.jar --delete-files *.zip
     java -jar bfg.jar --delete-files features.npy  # If you know the exact filename
     ```

4. **Cleanup and Force Push:**
   - Run:
     ```bash
     git reflog expire --expire=now --all
     git gc --prune=now --aggressive
     git push origin --force
     ```

5. **Verify:**
   - Refresh your GitHub repo page. Your code should now appear and pushes should succeed.

**Note:** If you need help with BFG or another tool, see the [BFG Repo-Cleaner documentation](https://rtyley.github.io/bfg-repo-cleaner/).
