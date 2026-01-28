import numpy as np

def analytic_field(X: np.ndarray, sigma: float = 0.0):
    '''
    Analytic field on first two coordinates with optional Gaussian white noise

    Parameters
    ----------
    X : np.ndarray, shape (N, d)
        Input points
    sigma : float, optional
        Standard deviation of Gaussian white noise (default: 0.0)

    Returns
    -------
    np.ndarray, shape (N,)
        Field values
    '''
    x = X[:, 0]
    y = X[:, 1]

    field = np.sin(x) * np.cos(y)

    if sigma > 0.0:
        noise = np.random.normal(loc=0.0, scale=sigma, size=field.shape)
        field = field + noise

    return field