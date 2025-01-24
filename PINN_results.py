from pinn.post_processing import evaluate_model
from pinn.config import run_ID

results = evaluate_model(0, f'models/{run_ID}_weights.npz')
