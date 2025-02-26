import matplotlib.pyplot as plt
import numpy as np

# Set random seeds for reproducibility
np.random.seed(1234)

# Define lower and upper bounds for x and y (ignoring t)
lb = [1, -2, 0]
ub = [8, 2, 20]

# Generate 100 collocation points in the given range for x and y
num_collocation = 500
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
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create grid in t, x, y
t = np.linspace(0, 10, 100)
x = np.linspace(-5, 5, 50)
y = np.linspace(-5, 5, 50)

# Generate meshgrid
T, X, Y = np.meshgrid(t, x, y)

# Define the function u(t, x, y)
u = np.sin(T) * np.cos(X) * np.sin(Y)

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Wireframe plot (we need to slice u for a 2D plot)
ax.plot_wireframe(T[:, :, 0], X[:, :, 0], u[:, :, 0], color='b', alpha=0.3)

# Labels
ax.set_xlabel('y')
ax.set_ylabel('x')
ax.set_zlabel('u(t,x,y)')

plt.show()