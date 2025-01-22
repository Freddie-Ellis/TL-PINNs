import os
from pinn.model import PhysicsInformedNN
from pinn.config import DATA_PATH
from pinn.utils import save_model, plot_loss_history
import numpy as np
from pinn.config import run_ID, nIter
from pinn.pre_processing import process_reynolds_data

# Set parameters
Re = 100  # Set Reynolds number 
nIter = 100  # Number of training iterations
run_ID = '001'  # Unique identifier for the run

# Process the Reynolds data
data = process_reynolds_data(Re)

# Extract processed training data
x_train = data['x_train']
y_train = data['y_train']
t_train = data['t_train']
u_train = data['u_train']
v_train = data['v_train']

# Define neural network layers
layers = [3, 20, 20, 20, 20, 20, 20, 20, 20, 2]  # Input (x, y, t) and output (u, v)

# Train the model
print('Training Model...')
model = PhysicsInformedNN(x_train, y_train, t_train, u_train, v_train, layers)
model.train(nIter)

# Plot and save the loss history
model.plot_loss_history(save_path=f'{plot_dir}/specmodel_{run_ID}_loss_history.png')
print(f'Model Training Complete for {nIter} iterations')

# Saving the trained model
save_dir = r'C:\Users\fredd\Desktop\Individual Project\Code\TL-PINNs\models'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Save the trained model weights
save_model(model, os.path.join(save_dir, f'specmodel_weights_{run_ID}.npz'))
print(f'Model weights saved to {save_dir}/specmodel_weights_{run_ID}.npz')