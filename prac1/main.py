import os
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Utility functions
from utils import plot_label_distribution, balance_dataset, split_test_data, plot_confusion_matrix

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from CNN_model import SimpleCNN

# 1. Load the Data


train_df = pd.read_csv("data/mitbih_train.csv", header=None)
test_df  = pd.read_csv("data/mitbih_test.csv", header=None)

print("Original training data shape:", train_df.shape)
print("Original test data shape:", test_df.shape)

plot_label_distribution(train_df.iloc[:, -1], 
                        title="Original Train Label Distribution", 
                        filename="figures/original_train_distribution.png",
                        show=True)

# 2. Balance the Training Data

balanced_train_df = balance_dataset(train_df, target_samples=17500)
print("Balanced training data shape:", balanced_train_df.shape)
print("Balanced training label distribution:", Counter(balanced_train_df.iloc[:, -1]))

plot_label_distribution(balanced_train_df.iloc[:, -1], 
                        title="Balanced Train Label Distribution", 
                        filename="figures/balanced_train_distribution.png",
                        show=True)

# 3. Split the Test Data into Validation and Test Sets

valid_df, final_test_df = split_test_data(test_df)
print("Validation set shape:", valid_df.shape)
print("Test set shape:", final_test_df.shape)

plot_label_distribution(valid_df.iloc[:, -1], 
                        title="Validation Label Distribution", 
                        filename="figures/validation_distribution.png",
                        show=True)

plot_label_distribution(final_test_df.iloc[:, -1], 
                        title="Test Label Distribution", 
                        filename="figures/test_distribution.png",
                        show=True)

# 4. Model 1: Random Forest 

X_train = balanced_train_df.iloc[:, :-1]
y_train = balanced_train_df.iloc[:, -1]
X_valid = valid_df.iloc[:, :-1]
y_valid = valid_df.iloc[:, -1]
X_test  = final_test_df.iloc[:, :-1]
y_test  = final_test_df.iloc[:, -1]

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
rf_model.fit(X_train, y_train)
y_pred_valid = rf_model.predict(X_valid)
print("Random Forest Validation Classification Report:")
print(classification_report(y_valid, y_pred_valid))
plot_confusion_matrix(y_valid, y_pred_valid, "Random Forest Confusion Matrix (Validation)")

# 5. Model 2: CNN

class ECGDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe.reset_index(drop=True)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sample = row.iloc[:-1].values.astype(np.float32)
        label = int(row.iloc[-1])
        return sample, label


train_dataset = ECGDataset(balanced_train_df)
valid_dataset = ECGDataset(valid_df)
test_dataset  = ECGDataset(final_test_df)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_model = SimpleCNN(num_classes=5).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

num_epochs = 20
train_losses = []
valid_losses = []

print("Training CNN model...")
for epoch in range(num_epochs):
    cnn_model.train()
    running_loss = 0.0
    for samples, labels in train_loader:
        samples = samples.unsqueeze(1).to(device)
        labels = torch.tensor(labels, dtype=torch.long).to(device)
        optimizer.zero_grad()
        outputs = cnn_model(samples)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    cnn_model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for samples, labels in valid_loader:
            samples = samples.unsqueeze(1).to(device)
            labels = torch.tensor(labels, dtype=torch.long).to(device)
            outputs = cnn_model(samples)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
    avg_valid_loss = running_loss / len(valid_loader)
    valid_losses.append(avg_valid_loss)
    
    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}")

# Plot loss curves
plt.figure(figsize=(10,5))
plt.plot(range(1, num_epochs+1), train_losses, label="Train Loss")
plt.plot(range(1, num_epochs+1), valid_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CNN Training and Validation Loss")
plt.legend()
plt.show()

# Evaluate CNN model on the test set
cnn_model.eval()
all_preds = []
all_true = []
with torch.no_grad():
    for samples, labels in test_loader:
        samples = samples.unsqueeze(1).to(device)
        outputs = cnn_model(samples)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_true.extend(labels)
print("CNN Test Classification Report:")
print(classification_report(all_true, all_preds))
plot_confusion_matrix(all_true, all_preds, "CNN Confusion Matrix (Test)")

# Save the CNN model checkpoint
if not os.path.exists("checkpoints"):
    os.makedirs("checkpoints")
torch.save(cnn_model.state_dict(), "checkpoints/cnn_model.pth")
print("CNN model saved at checkpoints/cnn_model.pth")
