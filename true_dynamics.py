import CMGDB
import math
import os
import ast
import argparse

def f(x):
  th1 = 23.5
  th2 = 23.5
  return [(th1 * x[0] + th2 * x[1]) * math.exp(-0.1 * (x[0] + x[1])), 0.7 * x[0]]

def F(rect):
    return CMGDB.BoxMap(f, rect, padding=True)

class Config:
   def __init__(self, config_fname):
    with open(config_fname) as f:
        config = ast.literal_eval(f.read())
    
        self.output_dir = config['output_dir']
        self.subdiv_min = config['subdiv_min']
        self.subdiv_max = config['subdiv_max']
        self.subdiv_init = config['subdiv_init']
        self.subdiv_limit = config['subdiv_limit']

        lower_x = config['lower_x']
        upper_x = config['upper_x']
        lower_y = config['lower_y']
        upper_y = config['upper_y']
        self.lower_bounds = [lower_x, lower_y]
        self.upper_bounds = [upper_x, upper_y]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir',help='Directory of config files',type=str,default='true_dynamics_configs/')
    parser.add_argument('--config',help='Config file inside config_dir',type=str,default='config_0.txt')

    args = parser.parse_args()
    config_fname = args.config_dir + args.config

    print('config_fname: ', config_fname)

    config = Config(config_fname)

    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    model = CMGDB.Model(config.subdiv_min, config.subdiv_max, config.subdiv_init, config.subdiv_limit, config.lower_bounds, config.upper_bounds, F)

    morse_graph, map_graph = CMGDB.ComputeConleyMorseGraph(model)
        
    morse_graph_plot = CMGDB.PlotMorseGraph(morse_graph)
    morse_graph_plot.render(os.path.join(config.output_dir, 'morse_graph'), format='png', view=False, cleanup=True)

    morse_sets_plot = CMGDB.PlotMorseSets(morse_graph, xlim=[config.lower_bounds[0], config.upper_bounds[0]], ylim=[config.lower_bounds[1], config.upper_bounds[1]], fig_fname=os.path.join(config.output_dir, 'morse_sets'))

    filename = os.path.join(config.output_dir, 'computation_log.txt')
    with open(filename, "w") as f:
        f.write("--- Computation Parameters ---\n")
        f.write(f"Lower bounds: {config.lower_bounds}\n")
        f.write(f"Upper bounds: {config.upper_bounds}\n")
        f.write("------------------------------\n")
        f.write(f"Subdivision init: {config.subdiv_init}\n")
        f.write(f"Subdivision min: {config.subdiv_min}\n")
        f.write(f"Subdivision max: {config.subdiv_max}\n")
        f.write(f"Subdivision limit: {config.subdiv_limit}")