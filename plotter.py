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

import matplotlib.pyplot as plt
import networkx as nx

def draw_neural_network(input_nodes=3, hidden_layers=8, hidden_nodes=20, output_nodes=2):
    G = nx.DiGraph()

    layer_sizes = [input_nodes] + [hidden_nodes] * hidden_layers + [output_nodes]
    positions = {}
    node_count = 0

    # Define positions for each layer
    for layer_idx, layer_size in enumerate(layer_sizes):
        x = layer_idx  # Each layer is placed on a new x-coordinate
        y_positions = list(range(layer_size))  # Spread out nodes vertically
        y_start = -(layer_size - 1) / 2  # Center nodes around y=0

        for i in range(layer_size):
            positions[node_count] = (x, y_start + i)
            node_count += 1

    # Create edges between layers
    previous_layer_nodes = list(range(layer_sizes[0]))
    node_counter = layer_sizes[0]

    for layer_size in layer_sizes[1:]:
        current_layer_nodes = list(range(node_counter, node_counter + layer_size))
        for prev in previous_layer_nodes:
            for curr in current_layer_nodes:
                G.add_edge(prev, curr)
        previous_layer_nodes = current_layer_nodes
        node_counter += layer_size

    # Plot the network
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos=positions, with_labels=False, node_size=300, edge_color="gray", node_color="blue")
    
    # Add labels for layers
    layer_labels = ["Input Layer"] + [f"Hidden {i+1}" for i in range(hidden_layers)] + ["Output Layer"]
    for i, label in enumerate(layer_labels):
        plt.text(i, max(layer_sizes) / 1.8, label, ha="center", fontsize=12, fontweight="bold")

    plt.title("Neural Network Architecture", fontsize=14)
    plt.show()

# Call the function to draw the neural network
draw_neural_network()