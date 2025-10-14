import math
import numpy as np
import os
import matplotlib.pyplot as plt

class LeslieModel:
    def __init__(self, th1=23.5, th2=23.5, lower_bounds=[0.001, 0.001], upper_bounds=[90, 70]):
        self.th1 = th1
        self.th2 = th2
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

    def f(self, x):
        return [(self.th1 * x[0] + self.th2 * x[1]) * math.exp(-0.1 * (x[0] + x[1])), 0.7 * x[0]]
    
    
def sample_random_pts(lower_bounds, upper_bounds, n):
    return np.random.uniform(np.asarray(lower_bounds), np.asarray(upper_bounds), (n, len(lower_bounds)))

if __name__ == "__main__":
    leslie_model = LeslieModel(th1=23.5, th2=23.5)
    print("Leslie Model Parameters:")
    print(f"th1: {leslie_model.th1}, th2: {leslie_model.th2}")
    
    for k in range(1, 11):
        n_iterations = 10
        n_samples = 2**k #int(total_pts/n_iterations)
        total_pts = n_iterations * n_samples

        if not os.path.exists(f"data/new/Leslie/{leslie_model.th1}_{leslie_model.th2}/{total_pts}"):
            os.makedirs(f"data/new/Leslie/{leslie_model.th1}_{leslie_model.th2}/{total_pts}")

        for str in ['train', 'test']:
            initial_conditions = sample_random_pts(leslie_model.lower_bounds, leslie_model.upper_bounds, n_samples)
            
            X = []
            Y = []
            Z = []
            all_pts = []

            for point in initial_conditions:
                for iteration in range(n_iterations):
                    result = leslie_model.f(point)
                    X.append(point)
                    Y.append(result)
                    point = result
                    if iteration > 8:
                        Z.append(point)

            data = np.hstack((np.asarray(X), np.asarray(Y)))

    #     plt.figure(figsize=(8, 6))
    #     plt.scatter(data[:, 0], data[:, 1], c='blue', label='Input Points', alpha=0.5)
    #     plt.scatter(data[:, 2], data[:, 3], c='red', label='Output Points', alpha=0.5)
        #    plt.scatter(np.asarray(Z)[:, 0], np.asarray(Z)[:, 1], c='green', label='Later Iterations', alpha=0.5)
    #     plt.show()

            np.savetxt(f"tmp.csv", data, delimiter=",", header = "x0,x1,y0,y1", comments="", fmt="%.16f")

            train_data = np.loadtxt(f'tmp.csv', delimiter=',', skiprows=1)
            # Separate inputs (x) and outputs (y)
            x_train = train_data[:, :2]
            y_train = train_data[:, 2:]

            # scatter plot x_train

            plt.figure(figsize=(8, 6))
            #plt.scatter(x_train[:, 0], x_train[:, 1], c='blue', label='Train Data', alpha=0.5)
            plt.scatter(y_train[:, 0], y_train[:, 1], c='red', label='Train Output', alpha=0.5)

            plt.title(f'Scatter Plot of {str} Data Inputs')
            plt.xlabel('x0')
            plt.ylabel('x1')
            plt.grid(True)
            plt.show()

            

            
            np.savetxt(f"data/new/Leslie/{leslie_model.th1}_{leslie_model.th2}/{total_pts}/{str}.csv", data, delimiter=",", header = "x0,x1,y0,y1", comments="", fmt="%.16f")
        
