# pinn/config.py

# Neural network architecture
LAYERS = [3, 20, 20, 20, 20, 20, 20, 20, 20, 2]

# Training parameters
LEARNING_RATE = 0.001
nIter = 100

run_ID = '001'

# File paths
DATA_PATH = r"C:\Users\fredd\Desktop\Individual Project\Code\data/input_data.npz"
MODEL_SAVE_PATH = r"C:\Users\fredd\Desktop\Individual Project\Code\models/pinn_model.npz"

# Random seed for reproducibility
SEED = 1234

# TensorFlow settings
import tensorflow as tf
import numpy as np

tf.random.set_seed(SEED)
np.random.seed(SEED)

print("Configuration loaded.")