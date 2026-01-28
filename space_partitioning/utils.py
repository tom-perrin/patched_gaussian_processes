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


def predict_on_grid_2d(
    pk: PatchworkKriging,
    x_bounds,
    y_bounds,
    nx,
    ny,
    invalid_region=-1
):
    x = np.linspace(x_bounds[0], x_bounds[1], nx)
    y = np.linspace(y_bounds[0], y_bounds[1], ny)

    Xg, Yg = np.meshgrid(x, y)

    Z_pred = np.full_like(Xg, np.nan, dtype=float)
    Z_var = np.full_like(Xg, np.nan, dtype=float)
    Z_region = np.full_like(Xg, invalid_region, dtype=int)
    valid_mask = np.zeros_like(Xg, dtype=bool)

    for j in range(ny):
        for i in range(nx):
            Xp = np.array([[Xg[j, i], Yg[j, i]]])
            try:
                y_pred, v_pred, region = pk.predict(Xp)
                yv = y_pred.item()
                vv = v_pred.item()

                if not np.isfinite(yv) or not np.isfinite(vv):
                    continue

                if not(-2 < yv < 2):
                    continue

                Z_pred[j, i] = yv
                Z_var[j, i] = vv
                Z_region[j, i] = int(region)
                valid_mask[j, i] = True

            except Exception:
                continue

    return Xg, Yg, Z_pred, Z_var, Z_region, valid_mask




def subsample_field(X_field, Y_field, ratio, seed=None):
    """
    ratio ∈ (0, 1] : fraction de points conservés
    """
    assert 0 < ratio <= 1.0

    N = X_field.shape[0]

    rng = np.random.default_rng(seed)
    idx = rng.choice(N, size=int(ratio * N), replace=False)

    return X_field[idx], Y_field[idx]