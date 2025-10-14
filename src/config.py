import ast

class Config:

    def __init__(self, config_fname):
        with open(config_fname) as f:
            config = ast.literal_eval(f.read())
        self.num_pts = config['num_pts']
        self.ex_index = config['ex_index']
        self.num_layers = config['num_layers']
        self.base_output_dir = config['base_output_dir'] + f'{self.num_pts}'
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
        self.num_layers = config['num_layers']
        self.hidden_shape = config['hidden_shape']
        self.non_linearity = config['non_linearity']