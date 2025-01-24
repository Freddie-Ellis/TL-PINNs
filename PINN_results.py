from pinn.post_processing import evaluate_model
from pinn.config import run_ID
from pinn.config import Re

results = evaluate_model(Re, 0, f'models/{run_ID}_weights.npz')
