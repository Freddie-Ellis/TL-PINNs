# pinn/post_processing.py

import os
import numpy as np
import scipy.io
from scipy.interpolate import griddata
from pinn.config import run_ID, nIter
from training import model
import matplotlib.pyplot as plt
import matplotlib
from pinn.FFT import FFT
from matplotlib.animation import FuncAnimation
import pandas as pd

matplotlib.use('Agg')  # For non-interactive backend

def calculate_mse(true, pred):
    return np.mean((true - pred) ** 2)

def relative_l2_error(true, pred):
    return np.linalg.norm(true - pred) / np.linalg.norm(true)

def evaluate_model(Re, snap, model_path, save_dir, x_start=1, x_end=8, y_start=-2, y_end=2):
    # Load the model
    model.load_model(model_path)

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Load data
    data_path = f'../data/Cyl{Re}/'
    vel_data = scipy.io.loadmat(f'{data_path}ustar')['ustar']  # N x 2 x T
    coord_data = scipy.io.loadmat(f'{data_path}xstar')['xstar']  # N x 2
    t_data = scipy.io.loadmat(f'{data_path}tstar')['tstar']    # T x 1

    # Prepare test data
    x_test = coord_data[:, 0:1]
    y_test = coord_data[:, 1:2]
    u_test = vel_data[:, 0, snap]
    v_test = vel_data[:, 1, snap]

    # Apply spatial filtering
    mask = ((x_test >= x_start) & (x_test <= x_end)) & ((y_test >= y_start) & (y_test <= y_end))
    x_test_filtered = x_test[mask].reshape(-1, 1)
    y_test_filtered = y_test[mask].reshape(-1, 1)
    u_test_filtered = u_test[mask.flatten()].reshape(-1, 1)
    v_test_filtered = v_test[mask.flatten()].reshape(-1, 1)

    # Create grid for plotting
    grid_x = np.linspace(x_start, x_end, 100)
    grid_y = np.linspace(y_start, y_end, 100)
    X_grid, Y_grid = np.meshgrid(grid_x, grid_y)

    # Predict
    t_pred = np.full((x_test_filtered.shape[0], 1), snap)
    u_pred, v_pred, p_pred, f_u_pred, f_v_pred = model.predict(x_test_filtered, y_test_filtered, t_pred)

    # Interpolate predictions and true values onto the grid
    u_grid = griddata((x_test_filtered.flatten(), y_test_filtered.flatten()), 
                      u_test_filtered.flatten(), (X_grid, Y_grid), method='cubic')
    v_grid = griddata((x_test_filtered.flatten(), y_test_filtered.flatten()), 
                      v_test_filtered.flatten(), (X_grid, Y_grid), method='cubic')

    # Evaluate metrics
    mse_u = calculate_mse(u_test_filtered, u_pred)
    mse_v = calculate_mse(v_test_filtered, v_pred)
    rel_error_u = relative_l2_error(u_test_filtered, u_pred) * 100  
    rel_error_v = relative_l2_error(v_test_filtered, v_pred) * 100  
    residual_u = np.mean(np.abs(f_u_pred)) 
    residual_v = np.mean(np.abs(f_v_pred))  

    # Get final lambda1 and lambda2 values
    lambda1 = model.lambda_1.numpy() 
    lambda2 = model.lambda_2.numpy() 
    error_lambda1 = np.abs(lambda1 - 1.0) * 100
    error_lambda2 = np.abs(lambda2 - (1/(Re))) / (1/(Re)) * 100

    # Format metrics for better readability
    metrics = {
        "Model Path": model_path,
        "MSE_u": f"{mse_u:.6e}",
        "MSE_v": f"{mse_v:.6e}",
        "Relative Error_u (%)": f"{rel_error_u:.2f}%",  # Percentage with 2 decimal places
        "Relative Error_v (%)": f"{rel_error_v:.2f}%",
        "Residual_u (%)": f"{residual_u:.6e}",
        "Residual_v (%)": f"{residual_v:.6e}",
        "Lambda1": f"{lambda1:.6f}",
        "Lambda2": f"{lambda2:.6f}",
        "Error Lambda1": f"{error_lambda1:.2f}%",
        "Error Lambda2": f"{error_lambda2:.2f}%"
    }

    # Check if the CSV already exists
    file_exists = os.path.isfile(f'plots/{run_ID}/metrics.csv')

    # Append metrics to the CSV file
    with open(f'plots/{run_ID}/metrics.csv', mode='a', newline='') as file:
        writer = pd.DataFrame([metrics])
        if not file_exists:
            writer.to_csv(file, index=False)  # Write headers if the file doesn't exist
        else:
            writer.to_csv(file, index=False, header=False)  # Append without headers

    # Plotting
    levels = 15

    def plot_field(field, title, label, filename, cmap="viridis"):
        plt.figure(figsize=(10, 6))
        plt.contourf(X_grid, Y_grid, field, levels=levels, cmap=cmap)
        plt.colorbar(label=label)
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.title(title)
        plt.savefig(f"{save_dir}/{filename}", dpi=300)
        plt.close()

    '''plot_field(u_grid, "True u Field", "True Velocity u", "true_u_field.png")
    plot_field(v_grid, "True v Field", "True Velocity v", "true_v_field.png")
    plot_field(griddata((x_test_filtered.flatten(), y_test_filtered.flatten()), 
                        u_pred.flatten(), (X_grid, Y_grid), method='cubic'), 
               "Predicted u Field", "Predicted Velocity u", "predicted_u_field.png")
    plot_field(griddata((x_test_filtered.flatten(), y_test_filtered.flatten()), 
                        v_pred.flatten(), (X_grid, Y_grid), method='cubic'), 
               "Predicted v Field", "Predicted Velocity v", "predicted_v_field.png")
'''
    # Relative errors
    error_u = np.abs(u_test_filtered - u_pred) / (np.abs(u_test_filtered) + 1e-6)
    error_v = np.abs(v_test_filtered - v_pred) / (np.abs(v_test_filtered) + 1e-6)

    plot_field(griddata((x_test_filtered.flatten(), y_test_filtered.flatten()), 
                        error_u.flatten(), (X_grid, Y_grid), method='cubic'), 
               "Relative Error in u Field", "Relative Error in u", "relative_error_u.png", cmap="coolwarm")
    plot_field(griddata((x_test_filtered.flatten(), y_test_filtered.flatten()), 
                        error_v.flatten(), (X_grid, Y_grid), method='cubic'), 
               "Relative Error in v Field", "Relative Error in v", "relative_error_v.png", cmap="coolwarm")

    # Animation for vorticity predictions
    def animate_vorticity_predictions(t_values):
        fig, ax = plt.subplots(figsize=(10, 6))
        levels = 50

        def update(frame):
            t_input = np.full_like(x_test_filtered, t_values[frame])
            u_pred, v_pred, _, _, _ = model.predict(x_test_filtered, y_test_filtered, t_input)
            
            # Interpolate u_pred and v_pred to the grid
            u_pred_grid = griddata(
                (x_test_filtered.flatten(), y_test_filtered.flatten()), 
                u_pred.flatten(), 
                (X_grid, Y_grid), 
                method='cubic'
            )
            v_pred_grid = griddata(
                (x_test_filtered.flatten(), y_test_filtered.flatten()), 
                v_pred.flatten(), 
                (X_grid, Y_grid), 
                method='cubic'
            )
            
            # Calculate gradients for vorticity
            u_y, u_x = np.gradient(u_pred_grid, grid_y, grid_x)
            v_y, v_x = np.gradient(v_pred_grid, grid_y, grid_x)

            # Compute vorticity: ω = ∂v/∂x - ∂u/∂y
            vorticity = v_x - u_y

            # Clear the axes and plot the vorticity field
            ax.clear()
            contour = ax.contourf(X_grid, Y_grid, vorticity, levels=levels, cmap="coolwarm")
            ax.set_title(f"Predicted Vorticity Field at Time t = {t_values[frame]:.2f}")
            return contour

        # Create the animation
        anim = FuncAnimation(fig, update, frames=len(t_values), interval=200)
        anim.save(f"{save_dir}/vorticity_animation.gif", writer="pillow", fps=5)
        plt.close(fig)

    #animate_vorticity_predictions(np.linspace(0, 20, 100)) #Comment this out to avoid long compilations if animation is already made
    
    '''Once complete add a script to delete the temp data files to prevent using up loads of storage on completed models.'''

    # Animation for u predictions
    def animate_u_predictions(t_values):
        fig, ax = plt.subplots(figsize=(10, 6))
        levels = 50

        def update(frame):
            t_input = np.full_like(x_test_filtered, t_values[frame])
            u_pred, _, _, _, _ = model.predict(x_test_filtered, y_test_filtered, t_input)
            u_pred_grid = griddata(
                (x_test_filtered.flatten(), y_test_filtered.flatten()), 
                u_pred.flatten(), (X_grid, Y_grid), method='cubic'
            )
            ax.clear()
            contour = ax.contourf(X_grid, Y_grid, u_pred_grid, levels=levels, cmap="viridis")
            ax.set_title(f"Predicted u Field at Time t = {t_values[frame]:.2f}")
            return contour

        anim = FuncAnimation(fig, update, frames=len(t_values), interval=200)
        anim.save(f"{save_dir}/animation.gif", writer="pillow", fps=5)
        plt.close(fig)
    animate_u_predictions(np.linspace(0, 20, 100)) #Comment this out to avoid long compilations if animation is already made
    