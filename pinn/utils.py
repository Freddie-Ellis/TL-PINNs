# pinn/utils.py

import numpy as np

def save_model(model, path):
    save_data = {}
    
    # Save weights and biases
    for i, (w, b) in enumerate(zip(model.weights, model.biases)):
        save_data[f'weight_{i}'] = w.numpy()
        save_data[f'bias_{i}'] = b.numpy()
    
    # Save lambda values
    save_data['lambda_1'] = model.lambda_1.numpy()
    save_data['lambda_2'] = model.lambda_2.numpy()
    
    # Save to file
    np.savez(path, **save_data)
    print(f"Model saved to {path}")
