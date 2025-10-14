import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib # Import joblib
import os

if __name__ == "__main__":
    for num_pts in [20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240]:
    
        train_data = np.loadtxt(f'data/new/Leslie/23.5_23.5/{num_pts}/train.csv', delimiter=',', skiprows=1)
        test_data = np.loadtxt(f'data/new/Leslie/23.5_23.5/{num_pts}/test.csv', delimiter=',', skiprows=1)

        # Separate inputs (x) and outputs (y)
        x_train = train_data[:, :2]
        y_train = train_data[:, 2:]
        x_test = test_data[:, :2]
        y_test = test_data[:, 2:]

        # --- START: Normalization ---
        # Initialize scalers
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()

        # Fit scalers ONLY on the training data
        x_train_scaled = x_scaler.fit_transform(x_train)
        y_train_scaled = y_scaler.fit_transform(y_train)

        # Apply the same transformation to the test data
        x_test_scaled = x_scaler.transform(x_test)
        y_test_scaled = y_scaler.transform(y_test)

        # --- END: Normalization ---
        output_dir = f'output/Leslie/23.5_23.5/{num_pts}/scalers/'
        os.makedirs(output_dir, exist_ok=True) # Ensure the directory exists
        joblib.dump(x_scaler, os.path.join(output_dir, 'x_scaler.gz'))
        joblib.dump(y_scaler, os.path.join(output_dir, 'y_scaler.gz'))