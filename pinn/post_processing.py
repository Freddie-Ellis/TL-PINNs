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

matplotlib.use('Agg')

def evaluate_model(Re, snap, model_path, save_dir, x_start=1, x_end=8, y_start=-2, y_end=2):
    model.load_model(model_path)

    os.makedirs(save_dir, exist_ok=True)
    
    data_path = f'../data/Cyl{Re}/'
    vel_data = scipy.io.loadmat(f'{data_path}ustar')['ustar']  # N x 2 x T
    t_data = scipy.io.loadmat(f'{data_path}tstar')['tstar']    # T x 1
    coord_data = scipy.io.loadmat(f'{data_path}xstar')['xstar']  # N x 2
    P_data = scipy.io.loadmat(f'{data_path}pstar')['pstar']

    '''Fit true DNS data to the training domain'''
    x_test = coord_data[:, 0:1]
    y_test = coord_data[:, 1:2]
    u_test = vel_data[:, 0, snap]
    v_test = vel_data[:, 1, snap]
    '''p_test = P_data[:, snap]'''

    # Create boolean mask for points within the desired range
    mask_x = (x_test >= x_start) & (x_test <= x_end)
    mask_y = (y_test >= y_start) & (y_test <= y_end)

    # Combine masks to filter only points within both ranges
    mask = mask_x & mask_y

    # Apply mask to filter data
    x_test_filtered = x_test[mask].reshape(-1, 1)
    y_test_filtered = y_test[mask].reshape(-1, 1)
    u_test_filtered = u_test[mask.flatten()].reshape(-1, 1)
    v_test_filtered = v_test[mask.flatten()].reshape(-1, 1)
    '''p_test_filtered = p_test[mask.flatten()]'''

    grid_x = np.linspace(x_start, x_end, 100) 
    grid_y = np.linspace(y_start, y_end, 100)
    X_grid, Y_grid = np.meshgrid(grid_x, grid_y)

    '''Make predictions on the test points'''
    
    # Recreate grid again for plotting
    coord_grid_pred = np.hstack((x_test_filtered, y_test_filtered))
    t_pred = np.full((coord_grid_pred.shape[0], 1), snap)

    u_pred, v_pred, p_pred, f_u_pred, f_v_pred = model.predict(x_test_filtered, y_test_filtered, t_pred)

    u_grid = griddata((x_test_filtered.flatten(), y_test_filtered.flatten()), 
                    u_test_filtered.flatten(), (X_grid, Y_grid), method='cubic')

    v_grid = griddata((x_test_filtered.flatten(), y_test_filtered.flatten()), 
                    v_test_filtered.flatten(), (X_grid, Y_grid), method='cubic')
    
    def calculate_mse(true, pred):
        mse = np.mean((true - pred)**2)
        return mse

    mse_u = calculate_mse(u_test_filtered, u_pred)
    mse_v = calculate_mse(v_test_filtered, v_pred)
    '''mse_p = calculate_mse(p_test_filtered, p_pred)'''

    print(f"MSE for u: {mse_u:.6e}, v: {mse_v:.6e}") #p: {mse_p:.6e}

    def relative_l2_error(true, pred):
        return np.linalg.norm(true - pred) / np.linalg.norm(true)

    rel_error_u = relative_l2_error(u_test_filtered, u_pred)
    rel_error_v = relative_l2_error(v_test_filtered, v_pred)
    '''rel_error_p = relative_l2_error(p_test_filtered, p_pred)'''

    print(f"Relative error for u: {rel_error_u:.6e}, v: {rel_error_v:.6e}") #p: {rel_error_p:.6e}

    error_u = np.abs(u_test_filtered - u_pred) / (np.abs(u_test_filtered) + 1e-6)
    error_v = np.abs(v_test_filtered - v_pred) / (np.abs(v_test_filtered) + 1e-6)

    levels = 15

    # Plot relative error for u component
    plt.figure(figsize=(10, 6))
    plt.contourf(X_grid, Y_grid, 
                griddata((x_test_filtered.flatten(), y_test_filtered.flatten()), 
                        error_u.flatten(), (X_grid, Y_grid), method='cubic'), 
                levels=levels, cmap="coolwarm")
    plt.colorbar(label="Relative Error in u")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Relative Error in u Field")
    plt.savefig(f'{save_dir}relative_error_u.png', dpi=300)

    # Plot true u field
    plt.figure(figsize=(10, 6))
    plt.contourf(X_grid, Y_grid, u_grid, levels=levels, cmap="viridis")
    plt.colorbar(label="True Velocity u")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("True u Field")
    plt.savefig(f'{save_dir}true_u_field.png', dpi=300)

    # Plot predicted u field
    plt.figure(figsize=(10, 6))
    plt.contourf(X_grid, Y_grid, 
                griddata((x_test_filtered.flatten(), y_test_filtered.flatten()), 
                        u_pred.flatten(), (X_grid, Y_grid), method='cubic'), 
                levels=levels, cmap="viridis")
    plt.colorbar(label="Predicted Velocity u")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Predicted u Field")
    plt.savefig(f'{save_dir}predicted_u_field.png', dpi=300)

    # --- V component plots ---

    # Plot relative error for v component
    plt.figure(figsize=(10, 6))
    plt.contourf(X_grid, Y_grid, 
                griddata((x_test_filtered.flatten(), y_test_filtered.flatten()), 
                        error_v.flatten(), (X_grid, Y_grid), method='cubic'), 
                levels=levels, cmap="coolwarm")
    plt.colorbar(label="Relative Error in v")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Relative Error in v Field")
    plt.savefig(f'{save_dir}relative_error_v.png', dpi=300)

    # Plot true v field
    plt.figure(figsize=(10, 6))
    plt.contourf(X_grid, Y_grid, v_grid, levels=levels, cmap="viridis")
    plt.colorbar(label="True Velocity v")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("True v Field")
    plt.savefig(f'{save_dir}true_v_field.png', dpi=300)

    # Plot predicted v field
    plt.figure(figsize=(10, 6))
    plt.contourf(X_grid, Y_grid, 
                griddata((x_test_filtered.flatten(), y_test_filtered.flatten()), 
                        v_pred.flatten(), (X_grid, Y_grid), method='cubic'), 
                levels=levels, cmap="viridis")
    plt.colorbar(label="Predicted Velocity v")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Predicted v Field")
    plt.savefig(f'{save_dir}predicted_v_field.png', dpi=300)

    # Evaluate the residuals
    residual_u = np.mean(np.abs(f_u_pred))
    residual_v = np.mean(np.abs(f_v_pred))

    print(f"Mean Residual of u-equation: {residual_u:.6e}")
    print(f"Mean Residual of v-equation: {residual_v:.6e}")

    def animate_u_predictions(model, x_test_filtered, y_test_filtered, X_grid, Y_grid, t_values, save_dir):

        # Define contour levels for consistent visualization
        levels = 50

        # Setup figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Initial prediction for the first time step
        t_input = np.full_like(x_test_filtered, t_values[0])
        u_pred, _, _, _, _ = model.predict(x_test_filtered, y_test_filtered, t_input)
        
        # Interpolate u_pred to grid
        u_pred_grid = griddata(
            (x_test_filtered.flatten(), y_test_filtered.flatten()), 
            u_pred.flatten(), 
            (X_grid, Y_grid), 
            method='cubic'
        )

        # Initial contour plot
        contour = ax.contourf(X_grid, Y_grid, u_pred_grid, levels=levels, cmap="viridis")
        colorbar = plt.colorbar(contour, ax=ax)
        colorbar.set_label("Predicted Velocity u")

        # Title, labels, and formatting
        ax.set_title(f"Predicted u Field at Time t = {t_values[0]:.2f}")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.axis('equal')

        # Update function for animation
        def update(frame):
            ax.clear()
            t_input = np.full_like(x_test_filtered, t_values[frame])

            # Predict u_pred for the current time step
            u_pred, _, _, _, _ = model.predict(x_test_filtered, y_test_filtered, t_input)

            # Interpolate u_pred values to the structured grid
            u_pred_grid = griddata(
                (x_test_filtered.flatten(), y_test_filtered.flatten()), 
                u_pred.flatten(), 
                (X_grid, Y_grid), 
                method='cubic'
            )

            # Update contour plot
            contour = ax.contourf(X_grid, Y_grid, u_pred_grid, levels=levels, cmap="viridis")
            ax.set_title(f"Predicted u Field at Time t = {t_values[frame]:.2f}")
            ax.set_xlabel("X Coordinate")
            ax.set_ylabel("Y Coordinate")
            ax.axis('equal')

            return contour

        # Create the animation
        anim = FuncAnimation(fig, update, frames=len(t_values), interval=200)

        # Save the animation as a GIF
        anim.save(f'{save_dir}/animation.gif', writer='pillow', fps=5)
        plt.close(fig)
        print(f"Animation saved")

    animate_u_predictions(model, x_test_filtered, y_test_filtered, X_grid, Y_grid, np.linspace(0, 20, 100), save_dir)
    '''Once complete add a script to delete the temp data files to prevent using up loads of storage on completed models.'''