from sklearn.svm import SVC 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, classification_report, 
confusion_matrix 
import joblib 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from feature_extraction import create_dataset 
 
def plot_confusion_matrix(y_true, y_pred, classes): 
    """Plot confusion matrix""" 
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes))) 
    plt.figure(figsize=(10, 8)) 
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',  
                xticklabels=classes, yticklabels=classes, 
                cbar_kws={'shrink': 0.8}) 
    plt.xlabel('Predicted') 
    plt.ylabel('Actual') 
    plt.title('Confusion Matrix - 6 Vocal Commands') 
    plt.tight_layout() 
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight') 
    plt.show() 
 
def main(): 
    # Updated commands list 
    commands = ['faaah', 'click', 'whistle', 'pop', 'hiss', 'hum'] 
     
    # Load the dataset 
    print("Loading training dataset...") 
    features, labels = create_dataset("C:/Users/Aneesh/Downloads/Non
Linguistic-Vocal-Command-Recognition-System-main (2)/Non-Linguistic
Vocal-Command-Recognition-System-main/Non-Linguistic-Vocal-Command
Recognition-System-main/sounds/train") 
     
    if len(features) == 0: 
        print("Error: No training data found! Please run data_collection.py first.") 
        return 
     
    print(f"Training samples: {features.shape[0]}") 
    print(f"Features per sample: {features.shape[1]}") 
     
    # Split into training and validation sets 
    X_train, X_val, y_train, y_val = train_test_split( 
        features, labels, test_size=0.25, random_state=42, stratify=None 
    ) 
 
    print(f"Training samples: {X_train.shape[0]}") 
    print(f"Validation samples: {X_val.shape[0]}") 
 
    # Train Support Vector Machine classifier 
    print("\nTraining SVM classifier...") 
    model = SVC(kernel='linear', probability=True, random_state=42) 
    model.fit(X_train, y_train) 
 
    # Evaluate the model on validation set 
    y_pred = model.predict(X_val) 
    accuracy = accuracy_score(y_val, y_pred) 
 
    print(f"\nModel accuracy: {accuracy:.2f}") 
     
    # Detailed classification report 
    print("\nClassification Report:") 
    print(classification_report( 
    y_val, 
    y_pred, 
    labels=range(len(commands)),   #      FIX 
    target_names=commands, 
    zero_division=0               # avoids warnings 
)) 
     
    # Plot confusion matrix 
    plot_confusion_matrix(y_val, y_pred, commands) 
     
    # Test on separate test data 
    print("\n=== TESTING ON SEPARATE TEST DATA ===") 
    test_features, test_labels = create_dataset("sounds/test") 
     
    if len(test_features) > 0: 
        test_pred = model.predict(test_features) 
        test_accuracy = accuracy_score(test_labels, test_pred) 
        print(f"Test accuracy: {test_accuracy:.2f}") 
        print("\nTest Classification Report:") 
        print(classification_report( 
    y_val,  
    y_pred,  
    labels=[0,1,2,3,4,5],  
    target_names=commands, 
    zero_division=0 
)) 
    else: 
        print("No test data found. Skipping test evaluation.") 
 
    # Save the model 
    joblib.dump(model, 'vocal_command_model.pkl') 
    print(f"\nModel saved as 'vocal_command_model.pkl'") 
     
    # Print some predictions 
    print("\n=== SAMPLE PREDICTIONS ===") 
    for i in range(min(8, len(X_val))): 
        actual = commands[y_val[i]] 
        predicted = commands[y_pred[i]] 
        confidence = np.max(model.predict_proba([X_val[i]])) 
        status = "✓" if actual == predicted else "✗" 
        print(f"{status} Sample {i+1}: Actual={actual}, Predicted={predicted}, 
Confidence={confidence:.2f}") 
 
if __name__ == "__main__": 
    main()