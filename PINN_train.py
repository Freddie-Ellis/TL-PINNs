import os
from pinn.model import PhysicsInformedNN
from pinn.utils import save_model, plot_loss_history
import numpy as np
from pinn.config import run_ID, nIter, LAYERS, Re
from pinn.pre_processing import process_reynolds_data


# Process the Reynolds data
data = process_reynolds_data(Re)

# Extract processed training data
x_train = data['x_train']
y_train = data['y_train']
t_train = data['t_train']
u_train = data['u_train']
v_train = data['v_train']

# Train the model
print('Training Model...')
model = PhysicsInformedNN(x_train, y_train, t_train, u_train, v_train, LAYERS)
model.train(nIter)

# Plot and save the loss history
model.plot_loss_history(save_path=f'plots/specmodel_{run_ID}_loss_history.png')
print(f'Model Training Complete for {nIter} iterations')

# Save the trained model weights
save_model(model, f'models/specmodel_weights_{run_ID}.npz')
print(f"Model weights saved to 'models/specmodel_weights_{run_ID}.npz'")