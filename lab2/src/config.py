import os
import torch

try:
    print("Starting script")
    DATASET_PATH = "/Users/apple/Downloads/lab2/1327317"
    TRAINING_DIR = os.path.join(DATASET_PATH, "training_set")
    TEST_DIR = os.path.join(DATASET_PATH, "test_set")
    TRAINING_CSV = os.path.join(DATASET_PATH, "training_set_pixel_size_and_HC.csv")
    TEST_CSV = os.path.join(DATASET_PATH, "test_set_pixel_size.csv")
    MODEL_DIR = "trained_model"

    IMG_SIZE = (256, 256)
    BATCH_SIZE = 4
    EPOCHS = 10
    LR = 1e-4

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("setting finish")

except Exception as e:
    print(f"An error occurred: {str(e)}")