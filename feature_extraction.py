import librosa 
import numpy as np 
import os 
import matplotlib.pyplot as plt 
 
# ========================= 
# FRS FUNCTION 
# ========================= 
def apply_frs(mfcc, threshold=0.5): 
    """ 
    Frame Relevance Score (FRS) 
    mfcc shape: (n_mfcc, frames) 
    """ 
    mfcc = mfcc.T  # (frames, features) 
 
    # Energy-based scoring 
    energy = np.sum(mfcc**2, axis=1) 
 
    if np.max(energy) > 0: 
        energy = energy / np.max(energy) 
 
    # Select important frames 
    idx = energy > threshold 
    selected = mfcc[idx] 
 
    # Safety fallback 
    if selected.shape[0] == 0: 
        selected = mfcc 
 
    return selected 
 
# ========================= 
# FEATURE EXTRACTION (WITH FRS) 
# ========================= 
import librosa 
import numpy as np 
 
def extract_features(audio_file): 
    try: 
        y, sr = librosa.load(audio_file, sr=22050, duration=2.0) 
 
        # Trim silence 
        y, _ = librosa.effects.trim(y, top_db=25) 
 
        # Normalize 
        if np.max(np.abs(y)) > 0: 
            y = y / np.max(np.abs(y)) 
 
        # ===== MFCC ===== 
        mfcc = librosa.feature.mfcc( 
            y=y, 
            sr=sr, 
            n_mfcc=13, 
            n_fft=2048, 
            hop_length=512 
        ) 
 
        # =====      FRS STEP ===== 
        # Frame energy (importance) 
        energy = np.sum(mfcc**2, axis=0) 
 
        # Normalize energy 
        energy = energy / np.max(energy) 
 
        # Select important frames 
        threshold = 0.3 
        important_frames = mfcc[:, energy > threshold] 
 
        # If too few frames, fallback 
        if important_frames.shape[1] < 5: 
            important_frames = mfcc 
 
        # Final feature = mean of selected frames 
        feature_vector = np.mean(important_frames, axis=1) 
 
#      FRS SCORE (REAL-TIME VALUE) 
        frs_score = np.mean(energy[energy > threshold]) if np.any(energy > 
threshold) else np.mean(energy) 
 
        return feature_vector, frs_score 
 
    except Exception as e: 
        print(f"Error: {e}") 
        return None 
 
# ========================= 
# DATASET CREATION 
# ========================= 
def create_dataset(data_folder): 
    features = [] 
    labels = [] 
 
    commands = ['faaah', 'click', 'whistle', 'pop', 'hiss', 'hum'] 
 
    for label, command in enumerate(commands): 
        command_folder = os.path.join(data_folder, command) 
 
        if not os.path.exists(command_folder): 
            print(f"Missing folder: {command_folder}") 
            continue 
 
        print(f"Processing {command}...") 
 
        for filename in os.listdir(command_folder): 
            if filename.endswith('.wav'): 
                filepath = os.path.join(command_folder, filename) 
 
                result = extract_features(filepath) 
 
                if result is not None: 
                    feature_vector, frs_score = result 
                    features.append(feature_vector) 
                    labels.append(label) 
 
                    # Optional: print FRS for debugging 
                    print(f"   {filename} -> FRS: {frs_score:.3f}") 
    return np.array(features), np.array(labels) 
 
# ========================= 
# SIMPLE CLASSIFIER (KNN) 
# ========================= 
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score, confusion_matrix, 
ConfusionMatrixDisplay 
 
def train_and_evaluate(features, labels): 
    X_train, X_test, y_train, y_test = train_test_split( 
        features, labels, test_size=0.2, random_state=42 
    ) 
 
    model = KNeighborsClassifier(n_neighbors=5) 
    model.fit(X_train, y_train) 
 
    y_pred = model.predict(X_test) 
 
    acc = accuracy_score(y_test, y_pred) 
    print(f"\n      Accuracy: {acc:.4f}") 
 
    # Confusion Matrix 
    cm = confusion_matrix(y_test, y_pred, labels=[0,1,2,3,4,5]) 
 
    disp = ConfusionMatrixDisplay( 
    confusion_matrix=cm, 
    display_labels=['faaah','click','whistle','pop','hiss','hum'] 
    ) 
 
    disp.plot(cmap='Blues') 
    plt.title("Confusion Matrix (With FRS)") 
    plt.show() 
 
    return model 
 
# ========================= 
# MAIN 
# ========================= 
if __name__ == "__main__": 
    print("Extracting features with FRS...") 
 
    base_dir = os.path.dirname(os.path.abspath(__file__)) 
    data_path = os.path.join(base_dir, "sounds", "train") 
 
    features, labels = create_dataset(data_path) 
 
    print(f"Dataset shape: {features.shape}") 
 
    model = train_and_evaluate(features, labels) 