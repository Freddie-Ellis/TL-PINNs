from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import scipy.io
from scipy.interpolate import griddata
from tqdm import tqdm

def process_reynolds_data(Re, time_start=0, time_end=250, num_time_points=10,
                          x_start=1, x_end=8, y_start=-2, y_end=2,
                          num_points_x=35, num_points_y=15, cylinder_radius=0.5):

    # File paths based on the given Reynolds number
    data_path = fr'C:\Users\fredd\Desktop\Individual Project\Code\data/Cyl{Re}/'
    print(f"Data processing began for Reynolds number: {Re}")

    # Load the data for the given Reynolds number
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
    t_data_spec = t_data[selected_time_indices]

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
    x_train_spec = np.tile(X_spec.flatten()[valid_points_mask, None], (num_time_points, 1))
    y_train_spec = np.tile(Y_spec.flatten()[valid_points_mask, None], (num_time_points, 1))
    t_train_spec = np.repeat(t_data_spec.flatten(), np.sum(valid_points_mask))[:, None]
    u_train_spec = UU_spec_filtered.T.reshape(-1, 1)
    v_train_spec = VV_spec_filtered.T.reshape(-1, 1)

    print(f"Data processing completed for Reynolds number: {Re}")
    
    return {
        'x_train': x_train_spec,
        'y_train': y_train_spec,
        't_train': t_train_spec,
        'u_train': u_train_spec,
        'v_train': v_train_spec,
        'UU_data': UU_data,
        'VV_data': VV_data,
        'coord_grid': coord_grid_spec_filtered,
        'time_indices': selected_time_indices
    }


def visualize_velocity_contours(processed_data, save_path='plots/velocity_magnitude_animation.gif'):
    """
    Visualize the velocity magnitude contours from processed PINN data.

    Parameters:
    - processed_data: dict, output from `process_reynolds_data` function
    - save_path: str, path to save the animation GIF
    """

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
    contour = ax.contourf(x_values, y_values, velocity_magnitude[:, :, 0], levels=50, cmap='viridis')
    colorbar = plt.colorbar(contour, ax=ax)
    colorbar.set_label("Velocity Magnitude")

    # Title, labels, and grid setup
    ax.set_title(f"Velocity Magnitude Contour at Time t = {t_data_spec[0]:.2f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis('equal')
    ax.grid(True)

    # Update function for animation
    def update(frame):
        ax.clear()  # Clear the axes for new data
        contour = ax.contourf(x_values, y_values, velocity_magnitude[:, :, frame], levels=50, cmap='viridis')
        ax.set_title(f"Velocity Magnitude Contour at Time t = {t_data_spec[frame] * 0.08}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.axis('equal')
        ax.grid(True)
        return contour

    # Create animation
    anim = FuncAnimation(fig, update, frames=num_time_points, interval=100)

    # Save animation as a GIF (ensure Pillow is installed)
    anim.save(save_path, writer='pillow', fps=5)
    print(f"Animation saved as {save_path}")

