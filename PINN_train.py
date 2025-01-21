import os
from pinn.model import PhysicsInformedNN
from pinn.config import DATA_PATH
from pinn.utils import save_model, plot_loss_history
import numpy as np
from pinn.config import run_ID, nIter

# Load data
data = np.load(DATA_PATH)
x, y, t, u, v = data['x'], data['y'], data['t'], data['u'], data['v']

# Initialize and train the model
model = PhysicsInformedNN(x, y, t, u, v)
model.train(nIter)

save_dir = 'models'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

model.save_model(os.path.join(save_dir, f'specmodel_weights_{run_ID}.npz'))