# pinn/post_processing.py

import os
import numpy as np
import scipy.io
from scipy.interpolate import griddata
from pinn.config import run_ID
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
    #run_ID = f'{run_ID}_TLnyq{Re}_[1, 2]'
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
    grid_x = np.linspace(x_start, x_end, 500)
    grid_y = np.linspace(y_start, y_end, 500)
    X_grid, Y_grid = np.meshgrid(grid_x, grid_y)

    # Predict
    t_pred = np.full((x_test_filtered.shape[0], 1), snap * 0.08)
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

    # Define the metrics file path
    metrics_file = f'plots/Re150bad/metrics.csv'
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    # Convert dictionary to DataFrame
    df_metrics = pd.DataFrame([metrics])

    # Check if the file exists
    file_exists = os.path.isfile(metrics_file)

    # Save metrics to CSV correctly
    df_metrics.to_csv(metrics_file, mode='a', index=False, header=not file_exists)  # Write headers only if the file doesn't exist
    # Plotting
    levels = 15

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    def plot_field(field, title, label, filename, cmap="bwr"):
        fig, ax = plt.subplots(figsize=(10, 6))

        # Main contour plot
        contourf_plot = ax.contourf(X_grid, Y_grid, field, levels=10, cmap=cmap)
        contour_lines = ax.contour(X_grid, Y_grid, field, levels=10, colors='k', linewidths=0.5)

        # Create a divider for the existing axes instance
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)  # 5% width, 0.1 padding

        # Place colorbar in cax
        cbar = fig.colorbar(contourf_plot, cax=cax, label=label)

        # Axis styling
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal")

        plt.tight_layout()
        plt.savefig(f"{save_dir}/{filename}", dpi=300, bbox_inches='tight')
        plt.close(fig)

    plot_field(u_grid, "True u Field at Time t = 0.00", "True Velocity u", "true_u_field.png")
    plot_field(v_grid, "True v Field at Time t = 0.00", "True Velocity v", "true_v_field.png")
    plot_field(griddata((x_test_filtered.flatten(), y_test_filtered.flatten()), 
                        u_pred.flatten(), (X_grid, Y_grid), method='cubic'), 
               "Predicted u Field", "Predicted Velocity u", "predicted_u_field.png")
    plot_field(griddata((x_test_filtered.flatten(), y_test_filtered.flatten()), 
                        v_pred.flatten(), (X_grid, Y_grid), method='cubic'), 
               "Predicted v Field", "Predicted Velocity v", "predicted_v_field.png")

    # Absolute errors
    error_u = np.abs(u_test_filtered - u_pred) 
    error_v = np.abs(v_test_filtered - v_pred) 

    plot_field(griddata((x_test_filtered.flatten(), y_test_filtered.flatten()), 
                        error_u.flatten(), (X_grid, Y_grid), method='cubic'), 
               "Relative Error in u Field", "Relative Error in u", "relative_error_u.png", cmap="coolwarm")
    plot_field(griddata((x_test_filtered.flatten(), y_test_filtered.flatten()), 
                        error_v.flatten(), (X_grid, Y_grid), method='cubic'), 
               "Relative Error in v Field", "Relative Error in v", "relative_error_v.png", cmap="coolwarm")

    # Animation for vorticity predictions
    def animate_vorticity_predictions(t_values):
        fig, ax = plt.subplots(figsize=(10, 6))

        # Precompute vorticity for all time steps to get global min/max
        all_vorticities = []

        for t in t_values:
            t_input = np.full_like(x_test_filtered, t)
            u_pred, v_pred, _, _, _ = model.predict(x_test_filtered, y_test_filtered, t_input)

            u_pred_grid = griddata((x_test_filtered.flatten(), y_test_filtered.flatten()), u_pred.flatten(), (X_grid, Y_grid), method='cubic')
            v_pred_grid = griddata((x_test_filtered.flatten(), y_test_filtered.flatten()), v_pred.flatten(), (X_grid, Y_grid), method='cubic')

            u_y, u_x = np.gradient(u_pred_grid, grid_y, grid_x)
            v_y, v_x = np.gradient(v_pred_grid, grid_y, grid_x)
            vorticity = v_x - u_y

            vorticity = np.nan_to_num(vorticity, nan=0.0, posinf=0.0, neginf=0.0)
            all_vorticities.append(vorticity)

        # Convert list to 3D array and get global min/max
        levels = np.linspace(-3, 3, 11)

        contourf_plot = ax.contourf(X_grid, Y_grid, vorticity, levels=levels, cmap="rainbow")
        contour_lines = ax.contour(X_grid, Y_grid, vorticity, levels=levels, colors='k', linewidths=0.5)
        cbar = fig.colorbar(contourf_plot, ax=ax, label="Vorticity (ω)")

        def update(frame):
            t_input = np.full_like(x_test_filtered, t_values[frame])
            u_pred, v_pred, _, _, _ = model.predict(x_test_filtered, y_test_filtered, t_input)
            u_pred_grid = griddata((x_test_filtered.flatten(), y_test_filtered.flatten()), u_pred.flatten(), (X_grid, Y_grid), method='cubic')
            v_pred_grid = griddata((x_test_filtered.flatten(), y_test_filtered.flatten()), v_pred.flatten(), (X_grid, Y_grid), method='cubic')
            u_y, u_x = np.gradient(u_pred_grid, grid_y, grid_x)
            v_y, v_x = np.gradient(v_pred_grid, grid_y, grid_x)
            vorticity = v_x - u_y

            ax.collections.clear()
            ax.contourf(X_grid, Y_grid, vorticity, levels=levels, cmap="rainbow")
            ax.contour(X_grid, Y_grid, vorticity, levels=levels, colors='k', linewidths=0.5)
            ax.set_title(f"Predicted Vorticity Field at Time t = {t_values[frame]:.2f}")
            return ax.collections

        anim = FuncAnimation(fig, update, frames=len(t_values), interval=200)
        anim.save(f"{save_dir}/vorticity_animation.gif", writer="pillow", fps=5, dpi=200)
        plt.close(fig)

    #animate_vorticity_predictions(np.linspace(0, 20, 100)) #Comment this out to avoid long compilations if animation is already made

    # Animation for u predictions
    def animate_u_predictions(t_values):
        fig, ax = plt.subplots(figsize=(10, 6))

        # Initial frame for colorbar and plot
        t_input = np.full_like(x_test_filtered, t_values[0])
        u_pred, _, _, _, _ = model.predict(x_test_filtered, y_test_filtered, t_input)
        u_pred_grid = griddata(
            (x_test_filtered.flatten(), y_test_filtered.flatten()), 
            u_pred.flatten(), (X_grid, Y_grid), method='cubic'
        )
        u_pred_min = np.min(u_pred)
        u_pred_max = np.max(u_pred)
        levels = np.linspace(u_pred_min, u_pred_max, 11)
        # Create initial filled contour and colorbar
        contourf_plot = ax.contourf(X_grid, Y_grid, u_pred_grid, levels=levels, cmap="bwr")
        contour_lines = ax.contour(X_grid, Y_grid, u_pred_grid, levels=levels, colors='k', linewidths=0.5)
        cbar = fig.colorbar(contourf_plot, ax=ax, label="Velocity (u)")

        def update(frame):
            t_input = np.full_like(x_test_filtered, t_values[frame])
            u_pred, _, _, _, _ = model.predict(x_test_filtered, y_test_filtered, t_input)
            u_pred_grid = griddata(
                (x_test_filtered.flatten(), y_test_filtered.flatten()), 
                u_pred.flatten(), (X_grid, Y_grid), method='cubic'
            )

            ax.collections.clear()  # Clear both filled and line contours
            ax.contourf(X_grid, Y_grid, u_pred_grid, levels=levels, cmap="bwr")
            contour_lines = ax.contour(X_grid, Y_grid, u_pred_grid, levels=levels, colors='k', linewidths=0.5)

            ax.set_title(f"Predicted u Field at Time t = {t_values[frame]:.2f}")
            if frame < 5:
                plt.savefig(f'frames/res_frame{frame}.png', dpi=200, bbox_inches='tight')
            return contour_lines.collections

        anim = FuncAnimation(fig, update, frames=len(t_values), interval=200)
        anim.save(f"{save_dir}/animation.gif", writer="pillow", fps=5, dpi=200)
        plt.close(fig)

    animate_u_predictions(np.linspace(0, 20, 100))

    # Animation for pressure predictions
    def animate_p_predictions(t_values):
        fig, ax = plt.subplots(figsize=(10, 6))

        # Initial frame for colorbar and plot
        t_input = np.full_like(x_test_filtered, t_values[0])
        _, _, p_pred, _, _ = model.predict(x_test_filtered, y_test_filtered, t_input)
        p_pred_grid = griddata(
            (x_test_filtered.flatten(), y_test_filtered.flatten()), 
            p_pred.flatten(), (X_grid, Y_grid), method='cubic'
        )
        p_pred_min = np.min(p_pred)
        p_pred_max = np.max(p_pred)
        levels = np.linspace(p_pred_min, p_pred_max, 11)

        # Create initial filled contour and colorbar
        contourf_plot = ax.contourf(X_grid, Y_grid, p_pred_grid, levels=levels, cmap="bwr")
        contour_lines = ax.contour(X_grid, Y_grid, p_pred_grid, levels=levels, colors='k', linewidths=0.5)
        cbar = fig.colorbar(contourf_plot, ax=ax, label="Pressure (p)")

        def update(frame):
            t_input = np.full_like(x_test_filtered, t_values[frame])
            _, _, p_pred, _, _ = model.predict(x_test_filtered, y_test_filtered, t_input)
            p_pred_grid = griddata(
                (x_test_filtered.flatten(), y_test_filtered.flatten()), 
                p_pred.flatten(), (X_grid, Y_grid), method='cubic'
            )

            ax.collections.clear()  # Clear both filled and line contours
            ax.contourf(X_grid, Y_grid, p_pred_grid, levels=levels, cmap="bwr")
            contour_lines = ax.contour(X_grid, Y_grid, p_pred_grid, levels=levels, colors='k', linewidths=0.5)

            ax.set_title(f"Predicted Pressure Field at Time t = {t_values[frame]:.2f}")
            if frame < 5:
                plt.savefig(f'frames/result/pres_frame{frame}.png', dpi=200, bbox_inches='tight')
            return contour_lines.collections

        anim = FuncAnimation(fig, update, frames=len(t_values), interval=200)
        anim.save(f"{save_dir}/pressure_animation.gif", writer="pillow", fps=5, dpi=200)
        plt.close(fig)

    #animate_p_predictions(np.linspace(0, 20, 100))


    def animate_v_predictions(t_values):
        fig, ax = plt.subplots(figsize=(10, 6))

        # Initial frame to define color limits
        t_input = np.full_like(x_test_filtered, t_values[0])
        _, v_pred, _, _, _ = model.predict(x_test_filtered, y_test_filtered, t_input)
        v_pred_grid = griddata((x_test_filtered.flatten(), y_test_filtered.flatten()), v_pred.flatten(), (X_grid, Y_grid), method='cubic')
        vmin, vmax = np.min(v_pred), np.max(v_pred)
        levels = np.linspace(vmin, vmax, 11)

        contourf_plot = ax.contourf(X_grid, Y_grid, v_pred_grid, levels=levels, cmap="bwr")
        contour_lines = ax.contour(X_grid, Y_grid, v_pred_grid, levels=levels, colors='k', linewidths=0.5)
        cbar = fig.colorbar(contourf_plot, ax=ax, label="Velocity (v)")

        def update(frame):
            t_input = np.full_like(x_test_filtered, t_values[frame])
            _, v_pred, _, _, _ = model.predict(x_test_filtered, y_test_filtered, t_input)
            v_pred_grid = griddata((x_test_filtered.flatten(), y_test_filtered.flatten()), v_pred.flatten(), (X_grid, Y_grid), method='cubic')

            ax.collections.clear()
            ax.contourf(X_grid, Y_grid, v_pred_grid, levels=levels, cmap="bwr")
            ax.contour(X_grid, Y_grid, v_pred_grid, levels=levels, colors='k', linewidths=0.5)
            ax.set_title(f"Predicted v Field at Time t = {t_values[frame]:.2f}")
            if frame < 5:
                plt.savefig(f'frames/result/v_frame{frame}.png', dpi=200, bbox_inches='tight')
            return ax.collections

        anim = FuncAnimation(fig, update, frames=len(t_values), interval=200)
        anim.save(f"{save_dir}/v_animation.gif", writer="pillow", fps=5, dpi=200)
        plt.close(fig)
        
   #animate_v_predictions(np.linspace(0, 20, 100))

def evaluate_time_averaged_error(Re, model_path, save_dir, x_start=1, x_end=8, y_start=-2, y_end=2, time_steps=21):
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
    u_test = vel_data[:, 0, :]  # Full time series for u
    v_test = vel_data[:, 1, :]  # Full time series for v

    # Apply spatial filtering
    mask = ((x_test >= x_start) & (x_test <= x_end)) & ((y_test >= y_start) & (y_test <= y_end))
    x_test_filtered = x_test[mask].reshape(-1, 1)
    y_test_filtered = y_test[mask].reshape(-1, 1)
    u_test_filtered = u_test[mask.flatten(), :]
    v_test_filtered = v_test[mask.flatten(), :]

    # Create grid for plotting
    grid_x = np.linspace(x_start, x_end, 500)
    grid_y = np.linspace(y_start, y_end, 500)
    X_grid, Y_grid = np.meshgrid(grid_x, grid_y)

    # Initialize arrays to accumulate errors over time
    total_error_u = np.zeros_like(X_grid)
    total_error_v = np.zeros_like(X_grid)
    total_points = np.zeros_like(X_grid)

    # Make predictions every second for 20 seconds (time_steps = 20)
    for t in range(time_steps):
        t_pred = np.full((x_test_filtered.shape[0], 1), t)  # Time steps in seconds

        # Get predictions from the model
        u_pred, v_pred, p_pred, f_u_pred, f_v_pred = model.predict(x_test_filtered, y_test_filtered, t_pred)

        # Interpolate predictions onto the grid
        u_grid = griddata((x_test_filtered.flatten(), y_test_filtered.flatten()), 
                          u_pred.flatten(), (X_grid, Y_grid), method='cubic')
        v_grid = griddata((x_test_filtered.flatten(), y_test_filtered.flatten()), 
                          v_pred.flatten(), (X_grid, Y_grid), method='cubic')

        # Interpolate true values onto the grid
        u_true_grid = griddata((x_test_filtered.flatten(), y_test_filtered.flatten()), 
                               u_test_filtered[:, int(t / 0.08)], (X_grid, Y_grid), method='cubic')  # Access time step t correctly
        v_true_grid = griddata((x_test_filtered.flatten(), y_test_filtered.flatten()), 
                               v_test_filtered[:, int(t / 0.08)], (X_grid, Y_grid), method='cubic')  # Access time step t correctly

        # Calculate errors at this time step
        error_u = np.abs(u_true_grid - u_grid)
        error_v = np.abs(v_true_grid - v_grid)

        '''# Calculate relative errors as a percentage for u and v
        rel_error_u = np.abs(u_true_grid - u_grid) / np.abs(u_true_grid) * 100
        rel_error_v = np.abs(v_true_grid - v_grid) / np.abs(v_true_grid) * 100'''

        # Accumulate errors
        total_error_u += error_u
        total_error_v += error_v
        total_points += np.ones_like(error_u)

    # Compute average errors over the entire time domain
    avg_error_u = total_error_u / total_points
    avg_error_v = total_error_v / total_points

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    def plot_field(field, title, label, filename, cmap="bwr"):
        fig, ax = plt.subplots(figsize=(10, 6))

        # Main contour plot
        contourf_plot = ax.contourf(X_grid, Y_grid, field, levels=10, cmap=cmap)
        contour_lines = ax.contour(X_grid, Y_grid, field, levels=10, colors='k', linewidths=0.5)

        # Create a divider for the existing axes instance
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)  # 5% width, 0.1 padding

        # Place colorbar in cax
        cbar = fig.colorbar(contourf_plot, cax=cax, label=label)

        # Axis styling
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal")

        plt.tight_layout()
        plt.savefig(f"{save_dir}/{filename}", dpi=300, bbox_inches='tight')
        plt.close(fig)

    # Plot time-averaged error fields
    plot_field(avg_error_u, "Time-Averaged Error in u Field", "Error in u", "avg_error_u_field.png", cmap="coolwarm")
    plot_field(avg_error_v, "Time-Averaged Error in v Field", "Error in v", "avg_error_v_field.png", cmap="coolwarm")