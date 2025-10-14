import torch
from torch import nn
import os
import pickle
from tqdm import tqdm
import numpy as np
from .model import DynamicsModel

class Trainer:
    def __init__(self, train_loader, test_loader, num_layers=3, hidden_shape=32, non_linearity='ReLU', lr=0.0001, model_dir='output/Leslie/models', log_dir='output/Leslie/logs', verbose=True):
        self.dynamics = DynamicsModel(num_layers=num_layers, hidden_shape=hidden_shape, non_linearity=non_linearity)
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.lr = lr

        self.model_dir = model_dir
        self.log_dir = log_dir

        self.verbose = bool(verbose)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device: ", self.device)

        self.dynamics.to(self.device)

        self.reset_losses()

        self.dynamics_criterion = nn.MSELoss(reduction='mean')

    def save_models(self, subfolder='', suffix=''):
        save_path = os.path.join(self.model_dir, subfolder)
        os.makedirs(save_path, exist_ok=True)
       # torch.save(self.dynamics, os.path.join(save_path, 'dynamics' + suffix + '.pt'))
        torch.save(self.dynamics.state_dict(), os.path.join(save_path, 'dynamics' + suffix + '.pt'))
    
    def load_models(self):
        self.dynamics = torch.load(os.path.join(self.model_dir, 'dynamics.pt'))
    
    def save_logs(self, suffix=''):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        with open(os.path.join(self.log_dir, 'train_losses_' + suffix + '.pkl'), 'wb') as f:
            pickle.dump(self.train_losses, f)
        
        with open(os.path.join(self.log_dir, 'test_losses_' + suffix + '.pkl'), 'wb') as f:
            pickle.dump(self.test_losses, f)
    
    def reset_losses(self):
        self.train_losses = {'loss_ae1': [], 'loss_ae2': [], 'loss_dyn': [], 'loss_contrastive': [], 'loss_topo': [], 'loss_total': []}
        self.test_losses = {'loss_ae1': [], 'loss_ae2': [], 'loss_dyn': [], 'loss_contrastive': [], 'loss_topo': [], 'loss_total': []}
    
    def forward(self, z_t, z_tau):
        # z_t = E(x_t)
        # z_tau = E(x_tau)

        # z_tau_pred = latent_dynamics(E(x_t))
        z_tau_pred = self.dynamics(z_t)

        return (z_tau, z_tau_pred)

    def dynamics_losses(self, forward_pass):
        z_tau, z_tau_pred = forward_pass
        loss_dyn = self.dynamics_criterion(z_tau_pred, z_tau)
        return loss_dyn
    
    def train(self, epochs=100, patience=20):

        list_parameters = list(self.dynamics.parameters())
        optimizer = torch.optim.Adam(list_parameters, lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
     #   scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epochs, eta_min=0.0, last_epoch=-1)
        for epoch in tqdm(range(epochs)):

            epoch_train_loss = 0
            epoch_test_loss  = 0

            self.dynamics.train()

            num_batches = len(self.train_loader)
            for (x_t, x_tau) in self.train_loader:
                x_t = x_t.to(self.device)
                x_tau = x_tau.to(self.device)
                optimizer.zero_grad()

                # Forward pass (apply all models)
                forward_pass = self.forward(x_t, x_tau)
                # Compute losses
                loss_dyn = self.dynamics_losses(forward_pass)

                # Backward pass
                loss_dyn.backward()
                optimizer.step()

                epoch_train_loss += loss_dyn.item()

            epoch_train_loss /= num_batches

            self.train_losses['loss_total'].append(epoch_train_loss)

            self.dynamics.eval()
            with torch.no_grad():

                num_batches = len(self.test_loader)
                for (x_t, x_tau) in self.test_loader:
                    optimizer.zero_grad()

                    # Forward pass (apply all models)
                    forward_pass = self.forward(x_t, x_tau)
                    # Compute losses
                    loss_dyn = self.dynamics_losses(forward_pass)

                    epoch_test_loss += loss_dyn.item()

                epoch_test_loss /= num_batches

                self.test_losses['loss_total'].append(epoch_test_loss)

            scheduler.step(epoch_test_loss)
            
            if epoch >= patience:
                if np.mean(self.test_losses['loss_total'][-patience:]) > np.mean(self.test_losses['loss_total'][-patience-1:-1]):
                    if self.verbose:
                        print("Early stopping")
                    break
            
            if self.verbose:
                print('Epoch [{}/{}], Train Loss: {:.4f}, Test Loss: {:.4f}'.format(epoch + 1, epochs, epoch_train_loss, epoch_test_loss))