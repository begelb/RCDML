from src.training import Trainer
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch
import joblib
import os
from src.config import Config
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir',help='Directory of config files',type=str,default='base_config/')
    parser.add_argument('--config',help='Config file inside config_dir',type=str,default='Leslie.txt')

    args = parser.parse_args()
    config_fname = args.config_dir + args.config

    config = Config(config_fname)

    num_pts = config.num_pts
    ex_index = config.ex_index
    base_data_dir = f'data/new/Leslie/23.5_23.5/{num_pts}'
    train_data_path = os.path.join(base_data_dir, 'train.csv')
    test_data_path = os.path.join(base_data_dir, 'test.csv')
    train_data = np.loadtxt(train_data_path, delimiter=',', skiprows=1)
    test_data = np.loadtxt(test_data_path, delimiter=',', skiprows=1)

    base_output_dir = config.base_output_dir 
    x_scaler_path = os.path.join(base_output_dir, 'scalers/x_scaler.gz')
    y_scaler_path = os.path.join(base_output_dir, 'scalers/y_scaler.gz')

    x_scaler = joblib.load(x_scaler_path)
    y_scaler = joblib.load(y_scaler_path)

    x_train = train_data[:, :2]
    y_train = train_data[:, 2:]
    x_test = test_data[:, :2]
    y_test = test_data[:, 2:]

    x_train_scaled = x_scaler.transform(x_train)
    y_train_scaled = y_scaler.transform(y_train)
    x_test_scaled = x_scaler.transform(x_test)
    y_test_scaled = y_scaler.transform(y_test)

    train_dataset = TensorDataset(torch.tensor(x_train_scaled, dtype=torch.float32), torch.tensor(y_train_scaled, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(x_test_scaled, dtype=torch.float32), torch.tensor(y_test_scaled, dtype=torch.float32))

    if num_pts < 32:
        batch_size = 16
    else:
        batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    output_dir = config.output_dir
    model_dir = os.path.join(output_dir, 'models')
    log_dir = os.path.join(output_dir, 'logs')

    num_layers = config.num_layers
    hidden_shape = config.hidden_shape
    non_linearity = config.non_linearity

    trainer = Trainer(train_loader, test_loader, num_layers=num_layers, hidden_shape=hidden_shape, non_linearity=non_linearity, lr=0.0001, model_dir=model_dir, log_dir=log_dir, verbose=True)

    trainer.train(epochs=1000)
    trainer.save_logs()
    trainer.reset_losses()
    trainer.save_models()