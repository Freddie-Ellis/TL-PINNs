# pinn/post_processing.py

import numpy as np
import scipy.io
from scipy.interpolate import griddata
from pinn.config import Re

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

def evaluate_model(Re, snap):
    snap = np.array([0])
    x_test = coord_data[:, 0:1]
    y_test = coord_data[:, 1:2]
    t_test = TT[:, snap]
    u_test = vel_data[:, 0, snap]
    v_test = vel_data[:, 1, snap]
    p_test = P_data[:, snap]
    print(snap)
