import os
import pandas as pd
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

def train_rf_model(X_train, y_train, n_estimators=100, random_state=42, class_weight="balanced"):
    """
    Train a RandomForestClassifier with the given training data.
    """
    model = RandomForestClassifier(n_estimators=n_estimators, 
                                   random_state=random_state, 
                                   class_weight=class_weight)
    model.fit(X_train, y_train)
    return model

def evaluate_rf_model(model, X_valid, y_valid):
    """
    Evaluate the RandomForestClassifier on validation data, print a classification report,
    and plot the confusion matrix.
    """
    y_pred = model.predict(X_valid)
    print("Validation Classification Report:")
    print(classification_report(y_valid, y_pred))
    
    cm = confusion_matrix(y_valid, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Random Forest Confusion Matrix (Validation)")
    plt.show()

def main():
    train_df = pd.read_csv("data/mitbih_train.csv", header=None)
    print("Training data shape:", train_df.shape)
    
    X = train_df.iloc[:, :-1]
    y = train_df.iloc[:, -1]
    print("Original training label distribution:", Counter(y))
    
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Training split label distribution:", Counter(y_train))
    print("Validation split label distribution:", Counter(y_valid))
    
    # Train 
    model = train_rf_model(X_train, y_train)
    
    # Evaluate on the validation set
    evaluate_rf_model(model, X_valid, y_valid)
    
    # Save the trained model to disk
    if not os.path.exists("model"):
        os.makedirs("model")
    joblib.dump(model, "model/rf_model.pkl")
    print("Random Forest model saved at model/rf_model.pkl")

if __name__ == "__main__":
    main()
