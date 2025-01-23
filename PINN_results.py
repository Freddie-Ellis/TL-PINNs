from pinn.post_processing import evaluate_model
from pinn.utils import plot_solution
from pinn.config import run_ID

"""results = evaluate_model(
    model = model,
    coord_data = data['coord_data'],
    vel_data = data['vel_data'],
    P_data = data['P_data'],
    TT = data['TT'],
    Re = 100,
    model_path = f'models/{run_ID}_weights.npz',
    plot_solution = plot_solution
)"""
results = evaluate_model(100, 0)
