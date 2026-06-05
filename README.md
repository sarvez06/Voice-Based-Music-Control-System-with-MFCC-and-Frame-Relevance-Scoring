# Voice-Based Music Control System Using Non-Linguistic Sound Recognition with MFCC and Frame Relevance Scoring

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-SVM-green)
![Signal Processing](https://img.shields.io/badge/Signal%20Processing-MFCC-orange)
![Audio Classification](https://img.shields.io/badge/Audio%20Classification-FRS-red)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## Overview

This project presents a **Voice-Based Music Control System** that uses **non-linguistic vocal sounds** instead of traditional speech commands for hands-free music player interaction.This was made with an intention that it will help the speech impaired person to interact so that the communication is effective.

Unlike conventional voice assistants that rely on language-dependent speech recognition, this system recognizes six distinct vocal sounds:

* Faaah
* Click
* Whistle
* Pop
* Hiss
* Hum

Each sound is mapped to a music control action such as play, pause, volume adjustment, and track navigation.

The system combines:

* **Mel Frequency Cepstral Coefficients (MFCC)**
* **Frame Relevance Scoring (FRS)**
* **Support Vector Machine (SVM) Classification**

to achieve robust real-time audio recognition.

---

## **Key Features**

* Language-independent control

* Hands-free music playback management

* MFCC-based feature extraction

* Frame Relevance Scoring (FRS)

* SVM classification model

* Real-time command recognition

* Interactive music player integration

* Visualization and analysis tools

---

## **System Architecture**

```text
┌──────────────────┐
│ Vocal Sound Input│
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Audio Recording  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Data Cleaning    │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ MFCC Extraction  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Frame Relevance  │
│ Scoring (FRS)    │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Feature Vector   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ SVM Classifier   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Music Control    │
│ Actions          │
└──────────────────┘
```

---

## **Sound-to-Action Mapping**

| **Vocal Sound** | **Music Action**   |
| ----------- | -------------- |
| Faaah       | Pause          |
| Hum         | Play / Resume  |
| Hiss        | Previous Track |
| Pop         | Next Track     |
| Click       | Volume Down    |
| Whistle     | Volume Up      |

---

## **Project Structure**

```text
Voice-Based-Music-Control-System/
│
├── README.md
│
├── data_collection.py
├── data_cleanup.py
├── feature_extraction.py
├── model_training.py
├── command_recognition_music.py
├── music_player.py
├── mfcc_visualisation.py
│
├── screenshots/
    ├── confusion_matrix.png
    ├── mfcc_scatter_plot.png
    ├── frs_comparison.png
    └── mfcc_heatmap.png
```

---

## **Required Libraries**

```text
numpy
scipy
librosa
sounddevice
pygame
scikit-learn
matplotlib
seaborn
joblib
```

---

## **Dataset Collection**

Run:

```bash
python data_collection.py
```

The script records:

* Training samples
* Test samples

for all six sound classes.

---

## **Dataset Cleaning**

```bash
python data_cleanup.py
```

Quality checks:

* Low volume detection
* Short recordings
* Clipping detection
* Silent audio removal

---

## **Feature Extraction**

MFCC Parameters:

| **Parameter**         | **Value**   |
| ----------------- | -------- |
| Sampling Rate     | 22050 Hz |
| MFCC Coefficients | 13       |
| FFT Window        | 2048     |
| Hop Length        | 512      |

---

## **Frame Relevance Scoring (FRS)**

The Frame Relevance Scoring algorithm removes unimportant frames and retains only acoustically significant information.

### **FRS Equation**

[
e(t) = [ Σ (k=1 to K) (M[k,t])² ] / max_{t'} [ Σ (k=1 to K) 
(M[k,t'])² ]  
]

Frames satisfying:

[
e(t) > 0.6
]

are retained.

---

## **Model Training**

Train the classifier:

```bash
python model_training.py
```

Classifier:

```text
Support Vector Machine (SVM)
Kernel: Linear
Probability Estimation: Enabled
```

Model Output:

```text
vocal_command_model.pkl
```

---

## **Experimental Results**

### Classification Accuracy

By giving more and more samples,the performance will eventually increase.

| **Dataset Size** | **Training Accuracy** | **Test Accuracy** |
| ------------ | ----------------- | ------------- |
| 30 Samples   | 88%               | 39%           |
| 120 Samples  | 60%               | 57%           |
| 180 Samples  | 80%               | 72%           |

---

## **Performance Summary**

By giving more and more samples,the performance will eventually increase.

| **Metric**        | **Value**      |
| ------------- | ---------- |
| Classes       | 6          |
| Features      | 13 MFCC    |
| Classifier    | Linear SVM |
| Test Accuracy | 72%        |
| Latency       | 150–300 ms |
| Sampling Rate | 22050 Hz   |

---

## **Screenshots**

### **Confusion Matrix**

<img width="1600" height="1405" alt="image" src="https://github.com/user-attachments/assets/50735828-4219-4296-9fac-4c8674d57ddc" />

### **PCA Feature Distribution**

<img width="1600" height="1250" alt="image" src="https://github.com/user-attachments/assets/ccb98584-7f3f-4cef-b705-5a4b72ace4ff" />

### **FRS Before vs After**

<img width="1600" height="651" alt="image" src="https://github.com/user-attachments/assets/229e1df9-c2b1-47c4-8bfe-c8eae2af12cf" />

### **Mean MFCC Heatmap**

<img width="1600" height="952" alt="image" src="https://github.com/user-attachments/assets/6d9355ff-7ecd-41e3-a81f-c8e904115331" />

---

## **Running the Music Player**

```bash
python music_player.py
```

Example:

```text
Listening for voice command...

Detected: whistle

Action:
Volume Up
```

---

## **Future Improvements**

* CNN-based classification
* LSTM temporal modeling
* Multi-speaker dataset
* Noise augmentation
* Embedded deployment on IoT devices
* Edge AI optimization
* Mobile application integration

---

## **License**

This project is released under the MIT License.

Feel free to use, modify, and distribute with proper attribution.
