import math

class LeslieModel:
    def __init__(self, th1=23.5, th2=23.5, lower_bounds=[0.001, 0.001], upper_bounds=[90, 70]):
        self.th1 = th1
        self.th2 = th2
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

    def f(self, x):
        return [(self.th1 * x[0] + self.th2 * x[1]) * math.exp(-0.1 * (x[0] + x[1])), 0.7 * x[0]]