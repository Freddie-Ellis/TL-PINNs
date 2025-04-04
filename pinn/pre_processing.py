# pinn/pre_processing.py

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import scipy.io
from scipy.interpolate import griddata
from tqdm import tqdm
import os

def process_reynolds_data(Re, time_start=0, time_end=250, num_time_points=5,
                          x_start=1, x_end=8, y_start=-2, y_end=2,
                          num_points_x=25, num_points_y=10, cylinder_radius=0.5,
                          save_path='temp/'):

    # Ensure the save directory exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Define the cache file path with correct naming convention
    save_file = os.path.join(save_path, f'data_re{Re}_{num_points_x}x{num_points_y}x{num_time_points}.npz')

    # Check if processed data already exists
    if os.path.exists(save_file):
        print(f"Loading cached data from {save_file}")
        return load_data(save_file)

    print(f"Data processing began for Reynolds number: {Re}")

    # File paths based on the given Reynolds number
    data_path = f'../data/Cyl{Re}/'


    vel_data = scipy.io.loadmat(f'{data_path}ustar')['ustar']  # N x 2 x T
    t_data = scipy.io.loadmat(f'{data_path}tstar')['tstar']    # T x 1
    coord_data = scipy.io.loadmat(f'{data_path}xstar')['xstar']  # N x 2
    P_data = scipy.io.loadmat(f'{data_path}pstar')['pstar']


    # Get shape parameters
    N = coord_data.shape[0]
    T = t_data.shape[0]

    # Tile time data across spatial points
    TT = np.tile(t_data, (1, N)).T
    PP = P_data  # N x T

    # Select specific time points
    selected_time_indices = np.linspace(time_start, time_end, num_time_points).astype(int)
    print(selected_time_indices)
    print(t_data.shape)
    t_data_spec = t_data[:,selected_time_indices]
    #t_data_spec = t_data[selected_time_indices]

    # Create a linear space for the spatial domain
    x_values = np.linspace(x_start, x_end, num_points_x)
    y_values = np.linspace(y_start, y_end, num_points_y)

    # Create a meshgrid for the new linearly spaced x and y values
    X_spec, Y_spec = np.meshgrid(x_values, y_values)

    # Combine X and Y coordinates into a single array
    coord_grid_spec = np.column_stack((X_spec.ravel(), Y_spec.ravel()))

    # Compute distances from the origin (0,0)
    distances_from_origin = np.sqrt(coord_grid_spec[:, 0]**2 + coord_grid_spec[:, 1]**2)

    # Filter out points within the cylinder of the specified radius
    valid_points_mask = distances_from_origin > cylinder_radius

    # Keep only the points outside the cylinder
    coord_grid_spec_filtered = coord_grid_spec[valid_points_mask]

    # Interpolated data arrays for the given Reynolds number
    UU_spec_filtered = np.zeros((coord_grid_spec_filtered.shape[0], num_time_points))
    VV_spec_filtered = np.zeros((coord_grid_spec_filtered.shape[0], num_time_points))

    # Perform interpolation for each selected time point with progress bar
    for i, t_idx in tqdm(enumerate(selected_time_indices), total=num_time_points, desc="Processing time steps"):
        UU_spec_filtered[:, i] = griddata(coord_data, vel_data[:, 0, t_idx], coord_grid_spec_filtered, method='cubic')
        VV_spec_filtered[:, i] = griddata(coord_data, vel_data[:, 1, t_idx], coord_grid_spec_filtered, method='cubic')

    # Initialize the grid with NaNs and place the valid values
    UU_data = np.full((num_points_y * num_points_x, num_time_points), np.nan)
    VV_data = np.full((num_points_y * num_points_x, num_time_points), np.nan)

    # Place the filtered/interpolated data back into the grid, skipping invalid points
    UU_data[valid_points_mask, :] = UU_spec_filtered
    VV_data[valid_points_mask, :] = VV_spec_filtered

    # Reshape to original grid shape minus invalid points (filtered points remain as NaN)
    UU_data = UU_data.reshape((num_points_y, num_points_x, num_time_points))
    VV_data = VV_data.reshape((num_points_y, num_points_x, num_time_points))

    # Prepare training data
    x_train = np.tile(X_spec.flatten()[valid_points_mask, None], (num_time_points, 1))
    y_train = np.tile(Y_spec.flatten()[valid_points_mask, None], (num_time_points, 1))
    t_train = np.repeat(t_data_spec.flatten(), np.sum(valid_points_mask))[:, None]
    u_train = UU_spec_filtered.T.reshape(-1, 1)
    v_train = VV_spec_filtered.T.reshape(-1, 1)

    print(f"Data processing completed for Reynolds number: {Re}")

    # Data dictionary
    processed_data = {
        'x_train': x_train,
        'y_train': y_train,
        't_train': t_train,
        'u_train': u_train,
        'v_train': v_train,
        'UU_data': UU_data,
        'VV_data': VV_data,
        'coord_grid': coord_grid_spec_filtered,
        'time_indices': selected_time_indices,
        'coord_data': coord_data,
        'vel_data': vel_data,
        'P_data': P_data,
        'TT': TT
    }

    # Save processed data for future use
    save_data(processed_data, save_file)

    return processed_data


def save_data(data, filename):
    """Saves processed data to an .npz file with structured arrays."""
    np.savez_compressed(filename, **data)
    print(f"Processed data saved to {filename}")


def load_data(filename):
    """Loads processed data from an .npz file."""
    data = np.load(filename, allow_pickle=True)
    print(f"Data loaded from {filename}")
    return {key: data[key] for key in data.files}

def visualize_velocity_contours(processed_data, save_path='plots/velocity_magnitude_animation.gif'):
    # Extract processed data
    UU_data = processed_data['UU_data']
    VV_data = processed_data['VV_data']
    X_spec = processed_data['coord_grid'][:, 0].reshape(-1, processed_data['UU_data'].shape[1])
    Y_spec = processed_data['coord_grid'][:, 1].reshape(-1, processed_data['UU_data'].shape[0])
    t_data_spec = processed_data['time_indices']
    num_time_points = UU_data.shape[2]

    # Compute velocity magnitude for contour plots
    velocity_magnitude = np.sqrt(UU_data**2 + VV_data**2)

    # Extract x and y values for contour plotting
    x_values = np.linspace(X_spec.min(), X_spec.max(), UU_data.shape[1])  # Ensure correct x size
    y_values = np.linspace(Y_spec.min(), Y_spec.max(), UU_data.shape[0])  # Ensure correct y size

    # Setup figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Initial contour plot
    contour = ax.contourf(x_values, y_values, velocity_magnitude[:, :, 0], cmap='bwr')
    colorbar = plt.colorbar(contour, ax=ax, pad=0.02)  # Reduce padding between colorbar and plot
    colorbar.set_label("Velocity Magnitude")

    # Title, labels, and grid setup
    ax.set_title(f"Velocity Magnitude Contour at Time t = {t_data_spec[0]:.2f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(1, 8)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    # Update function for animation
    def update(frame):
        ax.collections.clear()  # Clears only the contours without resetting the entire figure
        contour = ax.contourf(x_values, y_values, velocity_magnitude[:, :, frame], cmap='bwr')
        ax.set_title(f"Velocity Magnitude Contour at Time t = {t_data_spec[frame] * 0.08:.2f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(1, 8)
        ax.set_ylim(-2, 2)
        ax.set_aspect('equal')
        plt.savefig(f'frames/pre/frame{frame}')
        return contour.collections  # Return collections for animation

    # Create animation
    anim = FuncAnimation(fig, update, frames=num_time_points, interval=100)

    # Save animation as a GIF (ensure Pillow is installed)
    anim.save(save_path, writer='pillow', fps=1)
    print(f"Animation saved as {save_path}")

'''# Create animation with 1 frame every 2 seconds
anim = FuncAnimation(fig, update, frames=num_time_points, interval=2000)

# Save animation (no need for fps, interval controls timing)
anim.save(save_path, writer='pillow')
print(f"Animation saved as {save_path}")
!SLOWER ANIMATION FOR TRUE TIME VISULISATION!'''