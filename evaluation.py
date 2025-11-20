import numpy as np
import torch
import joblib
import os
import math
import argparse
from src.model import DynamicsModel # Assuming model.py is accessible as 'model'
from src.config import Config

# --- 1. Leslie Model Definition (Copied from make_data.py) ---
class LeslieModel:
    def __init__(self, th1=23.5, th2=23.5, lower_bounds=[0, 0], upper_bounds=[90, 70]):
        self.th1 = th1
        self.th2 = th2
        self.lower_bounds = np.array(lower_bounds)
        self.upper_bounds = np.array(upper_bounds)

    def f(self, x):
        # x is a numpy array of shape (N, 2)
        # x[:, 0] is x0, x[:, 1] is x1
        x0 = x[:, 0]
        x1 = x[:, 1]
        
        # Calculate the new state components
        y0 = (self.th1 * x0 + self.th2 * x1) * np.exp(-0.1 * (x0 + x1))
        y1 = 0.7 * x0
        
        # Stack them back to (N, 2)
        return np.stack([y0, y1], axis=1)

def evaluate_sup_norm(config, grid_resolution=500):
    """
    Evaluates the sup-norm (L-infinity) error between the Leslie model and the NN.

    Args:
        num_pts_data (int): The number of data points used to train the NN (used for path).
        nn_config (dict): Configuration for loading the model.
        grid_resolution (int): Number of points along each dimension for the evaluation grid.
    """
    th1 = 23.5
    th2 = 23.5
    
    # Instantiate the Ground Truth Leslie Model
    leslie_model = LeslieModel(th1=th1, th2=th2)
    lower_bounds = config.lower_bounds
    upper_bounds = config.upper_bounds

    model_dir = config.model_dir
    model_path = os.path.join(model_dir, 'dynamics.pt')

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model}")
        return
    
    scaler_dir = config.scaler_dir 

    x_scaler_path = os.path.join(scaler_dir, 'x_scaler.gz')
    y_scaler_path = os.path.join(scaler_dir, 'y_scaler.gz')


    print(f"--- Starting Sup-Norm Evaluation on {grid_resolution}x{grid_resolution} grid ---")
    print(f"Model path: {model_path}")

    x0_lin = np.linspace(lower_bounds[0], upper_bounds[0], grid_resolution)
    x1_lin = np.linspace(lower_bounds[1], upper_bounds[1], grid_resolution)
    
    x0_grid, x1_grid = np.meshgrid(x0_lin, x1_lin)
    grid_points = np.stack([x0_grid.flatten(), x1_grid.flatten()], axis=1)
    
    print(f"Generated a total of {grid_points.shape[0]} test points.")

    # --- 3. Load Scalers and Trained Model ---
    try:
        x_scaler = joblib.load(x_scaler_path)
        y_scaler = joblib.load(y_scaler_path)
        print("Scalers loaded successfully.")
    except Exception as e:
        print(f"Error loading scalers: {e}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DynamicsModel(
        num_layers=config.num_layers, 
        hidden_shape=config.hidden_shape, 
        non_linearity=config.non_linearity
    )
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded and set to {device}.")

    y_true = leslie_model.f(grid_points)

    x_scaled = x_scaler.transform(grid_points)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        y_scaled_pred_tensor = model(x_tensor)
    
    y_scaled_pred = y_scaled_pred_tensor.cpu().numpy()

    y_pred = y_scaler.inverse_transform(y_scaled_pred)

    errors = np.abs(y_true - y_pred)
    
    difference_vectors = y_true - y_pred
    pointwise_error = np.linalg.norm(difference_vectors, axis=1)
    sup_norm_error = np.max(pointwise_error)
    
    mean_error = np.mean(pointwise_error)
    std_error = np.std(pointwise_error)

    max_error_index = np.argmax(pointwise_error)
    x_max_error = grid_points[max_error_index]
    y_true_max = y_true[max_error_index]
    y_pred_max = y_pred[max_error_index]

    np.set_printoptions(formatter={'float_kind':lambda x: f"{x:.8f}"})

    print("\n--- Evaluation Results ---")
    print(f"Grid Resolution: {grid_resolution}x{grid_resolution}")
    print(f"Total Points Evaluated: {grid_points.shape[0]}")
    print(f"Sup-Norm (L_inf) Error: {sup_norm_error:.6f}")
    print(f"Mean Pointwise Error (L_inf): {mean_error:.6f}")
    print(f"Std Dev of Pointwise Error: {std_error:.6f}")
    print("\n--- Maximum Error Details ---")
    print(f"Input Point (x0, x1): {x_max_error}")
    print(f"True Output (y0, y1): {y_true_max}")
    print(f"NN Prediction (y0, y1): {y_pred_max}")
    print(f"Component-wise Abs Error: {errors[max_error_index]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate NN sup-norm error against Leslie Model ground truth.")
    parser.add_argument('--resolution', type=int, default=100, help='Number of grid points per dimension (e.g., 500 means 500x500 grid).')
    parser.add_argument('--config_dir',help='Directory of config files',type=str,default='base_config/')
    parser.add_argument('--config',help='Config file inside config_dir',type=str,default='Leslie_with_zero.txt')
    
    args = parser.parse_args()

    config_fname = args.config_dir + args.config
    config = Config(config_fname)
    
    # Example Usage: python evaluate_nn.py --num_pts 2560 --resolution 500
    evaluate_sup_norm(config=config, grid_resolution=args.resolution)