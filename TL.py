from pinn.post_processing import evaluate_model
from pinn.config import run_ID
from pinn.config import Re, nIter, LAYERS
import os
from pinn.pre_processing import process_reynolds_data, visualize_velocity_contours
from pinn.model import PhysicsInformedNN
from pinn.utils import save_model

# Define the path to the directory containing the models
model_dir = f'models/{run_ID}/'
model_files = sorted([f for f in os.listdir(model_dir) if f.endswith(".npz")])

run_ID = f"{run_ID}_TL"
Re_TL = 150

save_dir = f'models/{run_ID}{Re_TL}/'
os.makedirs(save_dir, exist_ok=True)
plots_dir = f'plots/{run_ID}{Re_TL}/'
os.makedirs(plots_dir, exist_ok=True)

# Load or process data
data = process_reynolds_data(Re_TL)
visualize_velocity_contours(data, save_path=f'plots/{run_ID}{Re_TL}/{run_ID}{Re_TL}_training_data.gif')

# Extract processed training data
x_train = data['x_train']
y_train = data['y_train']
t_train = data['t_train']
u_train = data['u_train']
v_train = data['v_train']

# Check if any models exist
if not model_files:
    print("No models found in the directory.")
else:
    print(f"Found {len(model_files)} models.")

# Iterate through each model
for model_file in model_files:
    model_path = os.path.join(model_dir, model_file)
    print(f"Processing model: {model_path}")

    # Define model
    model = PhysicsInformedNN(x_train, y_train, t_train, u_train, v_train, LAYERS, model_path, layers_to_freeze=[1, 2])

    model_file_no_ext, _ = os.path.splitext(model_file)

    print('Training Model...')
    model.train(nIter)

    # Plot and save the loss history
    model.plot_loss_history(save_path=f'plots/{run_ID}{Re_TL}/{model_file_no_ext}_TL{Re_TL}_loss_hist.png')
    model.plot_lambda_history(save_path=f'plots/{run_ID}{Re_TL}/{run_ID}{Re_TL}_lambda_hist.png')
    print(f'Model Training Complete for {nIter} iterations')

    # Save the trained model weights
    save_model(model, f'models/{run_ID}{Re_TL}/{model_file_no_ext}_TL{Re_TL}_{nIter}.npz')

    # Save results or analyze predictions
    print(f"Model {model_file} processed successfully.")

