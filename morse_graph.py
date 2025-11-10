import torch
import CMGDB
from functools import partial
import os
from src.model import DynamicsModel
import numpy
import joblib  # Import joblib
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pickle
from src.config import Config
import argparse

@torch.no_grad()
def g_base(x, dynamics_model, device, x_scaler, y_scaler):
    x_scaled = x_scaler.transform(np.asarray(x).reshape(1, -1))
    
    x_tensor = torch.as_tensor(x_scaled, dtype=torch.float32, device=device)
    output_tensor_scaled = dynamics_model(x_tensor)
    
    g_x_scaled = output_tensor_scaled.cpu().numpy()
    
    g_x_original = y_scaler.inverse_transform(g_x_scaled)[0]

    return g_x_original


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir',help='Directory of config files',type=str,default='base_config/')
    parser.add_argument('--config',help='Config file inside config_dir',type=str,default='Leslie.txt')

    args = parser.parse_args()
    config_fname = args.config_dir + args.config

    config = Config(config_fname)
    
    num_pts = config.num_pts
    ex_index = config.ex_index
    base_output_dir = config.base_output_dir 
    output_dir = config.output_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #  base_output_dir = f'output/Leslie/23.5_23.5/{num_pts}'
   # model_dir = os.path.join(output_dir, 'models')
    model_dir = config.model_dir
    model_path = os.path.join(model_dir, 'dynamics.pt')
    scaler_dir = config.scaler_dir 
    # x_scaler_path = os.path.join(base_output_dir, 'scalers/x_scaler.gz')
    # y_scaler_path = os.path.join(base_output_dir, 'scalers/y_scaler.gz')
    x_scaler_path = os.path.join(scaler_dir, 'x_scaler.gz')
    y_scaler_path = os.path.join(scaler_dir, 'y_scaler.gz')

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    dynamics_model = DynamicsModel(num_layers=config.num_layers, hidden_shape=config.hidden_shape, non_linearity=config.non_linearity)
    dynamics_model.load_state_dict(state_dict)
    dynamics_model.to(device)
    dynamics_model.eval()

    x_scaler = joblib.load(x_scaler_path)
    y_scaler = joblib.load(y_scaler_path)
    
    g = partial(g_base, dynamics_model=dynamics_model, device=device, x_scaler=x_scaler, y_scaler=y_scaler)

    def G(rect):
        return CMGDB.BoxMap(g, rect, padding=True)

    lower_bounds = config.lower_bounds
    upper_bounds = config.upper_bounds
    
    subdiv_min = config.subdiv_min
    subdiv_max = config.subdiv_max
    subdiv_init = config.subdiv_init
    subdiv_limit = config.subdiv_limit

    model = CMGDB.Model(subdiv_min, subdiv_max, subdiv_init, subdiv_limit, lower_bounds, upper_bounds, G)

    morse_graph, map_graph = CMGDB.ComputeConleyMorseGraph(model)
    
    MG_dir = os.path.join(output_dir, 'MG')

    # with open(os.path.join(MG_dir, 'morse_graph.pkl'), 'wb') as f:
    #     pickle.dump(morse_graph, f)

    morse_graph_plot = CMGDB.PlotMorseGraph(morse_graph)
    morse_graph_plot.render(os.path.join(MG_dir, 'morse_graph'), format='png', view=False, cleanup=True)

    morse_sets_plot = CMGDB.PlotMorseSets(morse_graph, xlim=[lower_bounds[0], upper_bounds[0]], ylim=[lower_bounds[1], upper_bounds[1]], fig_fname=os.path.join(MG_dir, 'morse_sets'))