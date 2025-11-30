import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib # Import joblib
import os

if __name__ == "__main__":
    num_pt_list = []
    for j in range(5,9):
        num_pts = (2**j) * 10
        num_pt_list.append(num_pts)

    for num_pts in num_pt_list:
    
        train_data = np.loadtxt(f'data/new/Leslie/23.5_23.5/20_iterations/{num_pts}_20_iterations/train.csv', delimiter=',', skiprows=1)
        test_data = np.loadtxt(f'data/new/Leslie/23.5_23.5/20_iterations/{num_pts}_20_iterations/test.csv', delimiter=',', skiprows=1)
        # train_data = np.loadtxt(f'data/new/Leslie/23.5_23.5/{num_pts}/train.csv', delimiter=',', skiprows=1)
        # test_data = np.loadtxt(f'data/new/Leslie/23.5_23.5/{num_pts}/test.csv', delimiter=',', skiprows=1)

        x_train = train_data[:, :2]
        y_train = train_data[:, 2:]
        x_test = test_data[:, :2]
        y_test = test_data[:, 2:]

        x_scaler = StandardScaler()
        y_scaler = StandardScaler()

        x_train_scaled = x_scaler.fit_transform(x_train)
        y_train_scaled = y_scaler.fit_transform(y_train)

        x_test_scaled = x_scaler.transform(x_test)
        y_test_scaled = y_scaler.transform(y_test)

        output_dir = f'output/Leslie/23.5_23.5/20_iterations/{num_pts}/scalers/'
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(x_scaler, os.path.join(output_dir, 'x_scaler.gz'))
        joblib.dump(y_scaler, os.path.join(output_dir, 'y_scaler.gz'))