'''import matplotlib.pyplot as plt
import numpy as np

# Set random seeds for reproducibility
np.random.seed(1234)

# Define lower and upper bounds for x and y (ignoring t)
lb = [1, -2, 0]
ub = [8, 2, 20]

# Generate 100 collocation points in the given range for x and y
num_collocation = 4500
x_f = np.random.uniform(lb[0], ub[0], (num_collocation, 1))
y_f = np.random.uniform(lb[1], ub[1], (num_collocation, 1))
t_f = np.random.uniform(lb[2], ub[2], (num_collocation, 1))
# Generate training points on a 30x15 grid
x_train = np.linspace(lb[0], ub[0], 30)
y_train = np.linspace(lb[1], ub[1], 15)

# Create a meshgrid for the training points
X_train, Y_train = np.meshgrid(x_train, y_train, indexing='ij')

# Flatten the grid for plotting
X_train_flat = X_train.flatten()
Y_train_flat = Y_train.flatten()

# Create a 2D scatter plot of the collocation and training points
plt.figure(figsize=(8, 6))

# Plot training grid points in blue with small 'x' markers
plt.scatter(X_train_flat, Y_train_flat, c='b', marker='x', label='Training Grid', s=20)

# Plot collocation points in red with larger circles and some transparency
plt.scatter(x_f, y_f, c='r', marker='o', label='Collocation Points', s=50, alpha=0.7, edgecolors='k')

# Labels and title
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.savefig(f'training points for {num_collocation} collocation points')
plt.show()

t = np.linspace(0,20,11)
y, t = np.meshgrid(y_train, t)
plt.scatter(t, y, c='b', marker='x', label='Training Grid', s=20)
plt.scatter(t_f, y_f, c='r', marker='o', label='Collocation Points', s=50, alpha=0.7, edgecolors='k')
plt.xlabel('t')
plt.ylabel('y')
plt.savefig('other side')
plt.show()'''

# TRUE FIELD PLOTTING
import scipy.io
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# Load data
snap = 0
data_path = f'../data/Cyl{100}/'
vel_data = scipy.io.loadmat(f'{data_path}ustar')['ustar']  # N x 2 x T
coord_data = scipy.io.loadmat(f'{data_path}xstar')['xstar']  # N x 2
t_data = scipy.io.loadmat(f'{data_path}tstar')['tstar']    # T x 1
xp_data = scipy.io.loadmat(f'{data_path}xpstar')['xpstar']
p_data = scipy.io.loadmat(f'{data_path}pstar')['pstar']

# Prepare test data
x_test = coord_data[:, 0:1]
y_test = coord_data[:, 1:2]
u_test = vel_data[:, 0, snap]
v_test = vel_data[:, 1, snap]

# Create regular grid
x = x_test.flatten()
y = y_test.flatten()
u = u_test.flatten()

xi = np.linspace(np.min(x), np.max(x), 200)
yi = np.linspace(np.min(y), np.max(y), 200)
X_grid, Y_grid = np.meshgrid(xi, yi)
U_grid = griddata((x, y), u, (X_grid, Y_grid), method='cubic')

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
contour = ax.contourf(X_grid, Y_grid, U_grid, levels=50, cmap='bwr')
cbar = fig.colorbar(contour, ax=ax, label="u velocity")

# Add rectangular box (x=0 to x=8, y=-2 to y=2)
rect = patches.Rectangle((1, -2), 8, 4, linewidth=2, edgecolor='k', facecolor='none')
ax.add_patch(rect)

# Add a circle at origin with diameter 1 (i.e., radius 0.5)
circle = patches.Circle((0, 0), 0.5, edgecolor='k', facecolor='k')
ax.add_patch(circle)

# Formatting
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Streamwise Velocity (u) at t = 0")
ax.set_aspect("equal")
plt.tight_layout()
plt.savefig('report_plots/u_example0', dpi=300)
plt.show()

# Extract coordinates and pressure
x_p = xp_data[:, 0]
y_p = xp_data[:, 1]
p = p_data[:, snap]

# Create interpolation grid
xi = np.linspace(np.min(x_p), np.max(x_p), 200)
yi = np.linspace(np.min(y_p), np.max(y_p), 200)
X_grid, Y_grid = np.meshgrid(xi, yi)
P_grid = griddata((x_p, y_p), p, (X_grid, Y_grid), method='cubic')

# Plot pressure field
fig, ax = plt.subplots(figsize=(8, 6))
contour = ax.contourf(X_grid, Y_grid, P_grid, levels=50, cmap='bwr')
cbar = fig.colorbar(contour, ax=ax, label="Pressure")

# Optional: add cylinder and bounding box again
circle = patches.Circle((0, 0), 0.5, linewidth=2, edgecolor='k', facecolor='k')
rect = patches.Rectangle((1, -2), 8, 4, linewidth=2, edgecolor='k', facecolor='none')
ax.add_patch(circle)
ax.add_patch(rect)

# Formatting
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Pressure Field at t = 0")
ax.set_aspect("equal")
plt.tight_layout()
plt.savefig("report_plots/p_example0", dpi=300)
plt.show()

# Define bounding box limits
x_min, x_max = 1, 8
y_min, y_max = -2, 2

# Filter points inside the box
mask = (x_p >= x_min) & (x_p <= x_max) & (y_p >= y_min) & (y_p <= y_max)

x_p_box = x_p[mask]
y_p_box = y_p[mask]
p_box = p[mask]

# Create regular grid inside the box
xi = np.linspace(x_min, x_max, 200)
yi = np.linspace(y_min, y_max, 200)
X_grid, Y_grid = np.meshgrid(xi, yi)
P_grid = griddata((x_p_box, y_p_box), p_box, (X_grid, Y_grid), method='cubic')

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))

# Contour and filled contour
levels = 10
contourf_plot = ax.contourf(X_grid, Y_grid, P_grid, levels=levels, cmap='bwr')
contour_lines = ax.contour(X_grid, Y_grid, P_grid, levels=levels, colors='k', linewidths=0.5)

# Colorbar
cbar = fig.colorbar(contourf_plot, ax=ax, label="Pressure")

# Optional: overlay box and cylinder
rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='k', facecolor='none')
ax.add_patch(rect)

# Axis and formatting
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("True p Field")
ax.set_aspect("equal")

plt.tight_layout()
plt.savefig("report_plots/p_box_example0", dpi=300, bbox_inches='tight')
plt.show()