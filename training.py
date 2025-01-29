import os
from pinn.model import PhysicsInformedNN
from pinn.utils import save_model
from pinn.config import run_ID, nIter, LAYERS, Re
from pinn.pre_processing import process_reynolds_data, visualize_velocity_contours
from pinn.model_wcp import PhysicsInformedNN_wcp

save_dir = f'models/{run_ID}/'
os.makedirs(save_dir, exist_ok=True)

plots_dir = f'plots/{run_ID}/'
os.makedirs(plots_dir, exist_ok=True)

# Load or process data
data = process_reynolds_data(Re)
'''visualize_velocity_contours(data, save_path=f'plots/{run_ID}/{run_ID}_training_data.gif')'''

# Extract processed training data
x_train = data['x_train']
y_train = data['y_train']
t_train = data['t_train']
u_train = data['u_train']
v_train = data['v_train']

# Define model
'''model = PhysicsInformedNN(x_train, y_train, t_train, u_train, v_train, LAYERS)'''
model = PhysicsInformedNN_wcp(x_train, y_train, t_train, u_train, v_train, LAYERS)

def train_model():
    
    print('Training Model...')
    model.train(nIter)

    save_dir = f'plots/{run_ID}/'
    os.makedirs(save_dir, exist_ok=True)

    # Plot and save the loss history
    model.plot_loss_history(save_path=f'plots/{run_ID}/{run_ID}_loss_hist.png')
    model.plot_lambda_history(save_path=f'plots/{run_ID}/{run_ID}_lambda_hist.png')
    print(f'Model Training Complete for {nIter} iterations')

    # Save the trained model weights
    save_model(model, f'models/{run_ID}/{run_ID}_{nIter}.npz')

# Run training only if executed directly
if __name__ == "__main__":
    train_model()