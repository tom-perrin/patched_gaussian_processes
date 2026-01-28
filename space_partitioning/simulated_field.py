import numpy as np

def analytic_field(X : np.ndarray):
    '''
    Field on first two coordinates
    '''
    if X.shape[1] >= 2:
        x = X[:, 0]
        y = X[:, 1]
        return np.sin(x) * np.cos(y)

    return np.sin(X + 67)
