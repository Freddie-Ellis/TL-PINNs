from pinn.post_processing import evaluate_model
from pinn.config import run_ID
from pinn.config import Re, nIter

results = evaluate_model(Re, 0, f'models/{run_ID}/{run_ID}_{nIter}.npz')
