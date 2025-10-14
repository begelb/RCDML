from torch import nn

class DynamicsModel(nn.Module):
    def __init__(self, num_layers=5, hidden_shape=128, non_linearity='ReLU', lower_shape=2):
        super(DynamicsModel, self).__init__()

        self.dynamics = nn.Sequential()
        for i in range(num_layers):
            if i == 0:
                self.dynamics.add_module(f"linear_{i}", nn.Linear(lower_shape, hidden_shape))
            else:
                self.dynamics.add_module(f"linear_{i}", nn.Linear(hidden_shape, hidden_shape))
            if non_linearity == 'ReLU':
                self.dynamics.add_module(f"relu_{i}", nn.ReLU(True))
            elif non_linearity == 'Hardtanh':
                self.dynamics.add_module(f"Htanh_{i}", nn.Hardtanh())
            elif non_linearity == 'Sigmoid':
                self.dynamics.add_module(f"sigmoid_{i}", nn.Sigmoid())
        self.dynamics.add_module(f"linear_{num_layers}", nn.Linear(hidden_shape, lower_shape))
    
    def forward(self, x):
        x = self.dynamics(x)
        return x
    
    