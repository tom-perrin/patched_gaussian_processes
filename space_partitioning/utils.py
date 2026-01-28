import numpy as np

from patchwork_kriging import PatchworkKriging

def predict_single_point(
        pk: PatchworkKriging,
        p : list[float]
):
    X_pred = np.array(p)

    Y_pred, V, region = pk.predict(X_pred)

    return X_pred, Y_pred, V, region



def predict_along_segment(
        pk: PatchworkKriging, 
        p1: list[float], 
        p2: list[float],
        n_points: int
    ):
    """
    Prédiction le long d'un segment [p1, p2]

    Parameters
    ----------
    pk : PatchworkKriging
    p1, p2 : array-like, shape (2,)
        Points extrémités du segment
    n_points : int
        Nombre de points d'échantillonnage

    Returns
    -------
    t : (n_points,)
        Paramètre le long de la ligne (0 → 1)
    X_pred : (n_points, 2)
        Points prédits
    Y_pred : (n_points,)
        Prédictions
    var_pred : (n_points,)
        Variances associées
    regions : (n_points,)
        Indices des régions
    """
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)

    t = np.linspace(0.0, 1.0, n_points)
    X_pred = (1 - t)[:, None] * p1 + t[:, None] * p2

    Y_pred = np.zeros(n_points)
    var_pred = np.zeros(n_points)
    regions = np.zeros(n_points, dtype=int)

    for i in range(n_points):
        y, v, r = pk.predict(X_pred[i:i+1])
        Y_pred[i] = y.item()
        var_pred[i] = v.item()
        regions[i] = int(r)

    return t, X_pred, Y_pred, var_pred, regions

def subsample_field(X_field, Y_field, ratio, seed=None):
    """
    ratio ∈ (0, 1] : fraction de points conservés
    """
    assert 0 < ratio <= 1.0

    N = X_field.shape[0]

    rng = np.random.default_rng(seed)
    idx = rng.choice(N, size=int(ratio * N), replace=False)

    return X_field[idx], Y_field[idx]