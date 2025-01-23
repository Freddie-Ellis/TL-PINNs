# pinn/utils.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import tensorflow as tf

def save_model(model, path):
    save_dict = {}
    for i, w in enumerate(model.weights):
        save_dict[f"weight_{i}"] = w.numpy()
    for i, b in enumerate(model.biases):
        save_dict[f"bias_{i}"] = b.numpy()
    np.savez(path, **save_dict)
    print(f"Model saved to {path}")

def plot_loss_history(loss_history, save_path="plots/loss_plot.png"):
    plt.plot(loss_history)
    plt.yscale("log")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Training Loss History")
    plt.savefig(save_path)
    print(f"Loss history saved to {save_path}")
    
def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/4
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)
        

def newfig(width, height):
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_subplot(111)
    return fig, ax
    
def plot_solution(X_star, u_star, index, title):
    
    lb = X_star.min(0)
    ub = X_star.max(0)
    nn = 200
    x = np.linspace(lb[0], ub[0], nn)
    y = np.linspace(lb[1], ub[1], nn)
    X, Y = np.meshgrid(x,y)
    
    U_star = griddata(X_star, u_star.flatten(), (X, Y), method='cubic')
    
    plt.figure(index)
    plt.pcolor(X,Y,U_star, cmap = 'jet')
    plt.title(title)
    plt.colorbar()
