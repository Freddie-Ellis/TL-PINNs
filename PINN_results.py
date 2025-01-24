from pinn.post_processing import evaluate_model
from pinn.config import run_ID
from pinn.config import Re, nIter
import os

# Define the path to the directory containing the models
model_dir = f'models/{run_ID}/'

# List all model files in the directory
model_files = sorted([f for f in os.listdir(model_dir) if f.endswith(".npz")])

# Check if any models exist
if not model_files:
    print("No models found in the directory.")
else:
    print(f"Found {len(model_files)} models.")

# Iterate through each model
for model_file in model_files:
    model_path = os.path.join(model_dir, model_file)
    print(f"Processing model: {model_path}")
        
    model_file_no_ext, _ = os.path.splitext(model_file)

    # Pass the modified filename for saving plots
    results = evaluate_model(
        Re, 
        0, 
        f'models/{run_ID}/{model_file}', 
        f'plots/{run_ID}/{model_file_no_ext}/'
    )
    # Save results or analyze predictions
    print(f"Model {model_file} processed successfully.")

