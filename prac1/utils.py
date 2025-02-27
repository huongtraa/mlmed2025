import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def plot_label_distribution(y, title, filename, show=True):
    plt.figure(figsize=(10,5))
    counts, bins, _ = plt.hist(y, color="skyblue", edgecolor="black")
    for count, bin_edge in zip(counts, bins[:-1]):
        plt.text(bin_edge + (bins[1]-bins[0])/2, count, str(int(count)), 
                 ha='center', va='bottom', fontsize=10)
    plt.xticks([0,1,2,3,4])
    plt.title(title)
    plt.xlabel("Class Label")
    plt.ylabel("Count")
    if show:
        plt.show()
    plt.savefig(filename)
    plt.close()
    print("Figure saved at", filename)

def balance_dataset(df, target_samples=17500):
    # Assumes the label is in the last column.
    balanced_df = pd.DataFrame()
    for label, group in df.groupby(df.columns[-1]):
        if len(group) < target_samples:
            group_sampled = group.sample(n=target_samples, replace=True, random_state=42)
        else:
            group_sampled = group.sample(n=target_samples, replace=False, random_state=42)
        balanced_df = pd.concat([balanced_df, group_sampled], axis=0)
    return balanced_df

def split_test_data(df):
    from sklearn.model_selection import train_test_split
    valid_df, test_df = train_test_split(df, test_size=0.5, stratify=df.iloc[:, -1], random_state=42)
    return valid_df, test_df

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.show()
