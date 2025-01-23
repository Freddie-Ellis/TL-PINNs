import os
from pinn.model import PhysicsInformedNN
from pinn.utils import save_model, plot_loss_history
import numpy as np
from pinn.config import run_ID, nIter, LAYERS, Re
from pinn.pre_processing import process_reynolds_data


# Load or process data
data = process_reynolds_data(100)

# Extract processed training data
x_train = data['x_train']
y_train = data['y_train']
t_train = data['t_train']
u_train = data['u_train']
v_train = data['v_train']

# Define model
model = PhysicsInformedNN(x_train, y_train, t_train, u_train, v_train, LAYERS)

def train_model():
    
    print('Training Model...')
    model.train(nIter)

    # Plot and save the loss history
    model.plot_loss_history(save_path=f'plots/{run_ID}_loss_history.png')
    print(f'Model Training Complete for {nIter} iterations')

    # Save the trained model weights
    save_model(model, f'models/{run_ID}_weights.npz')
    print(f"Model weights saved to 'models/{run_ID}_weights.npz'")

# Run training only if executed directly
if __name__ == "__main__":
    train_model()