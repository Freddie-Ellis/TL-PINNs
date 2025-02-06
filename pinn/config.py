# pinn/config.py

import os
import tensorflow as tf
import numpy as np
import warnings
import logging

# Set TensorFlow logging level to suppress detailed messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all messages, 1 = ignore INFO, 2 = ignore INFO and WARN, 3 = ignore all

# Verify if GPU is detected but suppress warnings
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("GPU detected and configured.")
else:
    print("Running on CPU.")

# Suppress specific TensorFlow warnings
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")

# Reduce logging verbosity
tf.get_logger().setLevel(logging.ERROR)

# Neural network architecture
LAYERS = [3, 20, 20, 20, 20, 20, 20, 20, 20, 2]

# Training parameters
LEARNING_RATE = 0.001
Re = 100    # Set Reynolds number 
nIter = 10000     # Number of training iterations
run_ID = f'base_{Re}'      #Unique identifier for the run

# Random seed for reproducibility
SEED = 1234
tf.random.set_seed(SEED)
np.random.seed(SEED)

print("Configuration loaded.")
