import matplotlib.pyplot as plt
import numpy as np


num_pts = 20
train_data = np.loadtxt(f'data/new/Leslie/23.5_23.5/{num_pts}/train.csv', delimiter=',', skiprows=1)
test_data = np.loadtxt(f'data/new/Leslie/23.5_23.5/{num_pts}/test.csv', delimiter=',', skiprows=1)

# Separate inputs (x) and outputs (y)
x_train = train_data[:, :2]
y_train = train_data[:, 2:]

# scatter plot x_train

plt.figure(figsize=(8, 6))
#plt.scatter(x_train[:, 0], x_train[:, 1], c='blue', label='Train Data', alpha=0.5)
plt.scatter(y_train[:, 0], y_train[:, 1], c='red', label='Train Output', alpha=0.5)
plt.title('Scatter Plot of Training Data Inputs')
plt.xlabel('x0')
plt.ylabel('x1')
plt.grid(True)

plt.show()