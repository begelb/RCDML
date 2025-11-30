import math
import numpy as np
import os
import matplotlib.pyplot as plt
from src.leslie_model import LeslieModel
    
def sample_random_pts(lower_bounds, upper_bounds, n):
    return np.random.uniform(np.asarray(lower_bounds), np.asarray(upper_bounds), (n, len(lower_bounds)))

if __name__ == "__main__":
    leslie_model = LeslieModel(th1=23.5, th2=23.5)
    print("Leslie Model Parameters:")
    print(f"th1: {leslie_model.th1}, th2: {leslie_model.th2}")
    
    for k in range(4,9):
        n_iterations = 20 #10
        n_samples = 2**k
        total_pts = n_iterations * n_samples

        print('total pts : ', total_pts)

        if not os.path.exists(f"data/new/Leslie/{leslie_model.th1}_{leslie_model.th2}/20_iterations/{total_pts}_20_iterations"):
            os.makedirs(f"data/new/Leslie/{leslie_model.th1}_{leslie_model.th2}/20_iterations/{total_pts}_20_iterations")

        for str in ['train', 'test']:
            initial_conditions = sample_random_pts(leslie_model.lower_bounds, leslie_model.upper_bounds, n_samples)
            
            X = []
            Y = []

            for point in initial_conditions:
                for iteration in range(n_iterations):
                    result = leslie_model.f(point)
                    X.append(point)
                    Y.append(result)
                    point = result

            data = np.hstack((np.asarray(X), np.asarray(Y)))
            np.savetxt(f"data/new/Leslie/{leslie_model.th1}_{leslie_model.th2}/20_iterations/{total_pts}_20_iterations/{str}.csv", data, delimiter=",", header = "x0,x1,y0,y1", comments="", fmt="%.16f")
        
