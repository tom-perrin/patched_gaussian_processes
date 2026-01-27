import numpy as np

def analytic_field(X : np.ndarray):
    '''
    Field on first two coordinates
    '''
    x = X[:, 0]
    y = X[:, 1]
    return np.sin(x) * np.cos(y)