# pinn/config.py

# Neural network architecture
LAYERS = [3, 50, 50, 50, 2]

# Training parameters
LEARNING_RATE = 0.001
EPOCHS = 10000
BATCH_SIZE = 32

# File paths
run_ID = '001'
DATA_PATH = "data/input_data.npz"
MODEL_SAVE_PATH = "models/pinn_model.npz"

# Random seed for reproducibility
SEED = 1234
# TensorFlow settings
import tensorflow as tf
import numpy as np

tf.random.set_seed(SEED)
np.random.seed(SEED)

print("Configuration loaded.")