from pykrige.ok import OrdinaryKriging
import numpy as np

def kriging(data, points ):

    OK = OrdinaryKriging(
        data[:, 0], data[:, 1], data[:, 2], 
        variogram_model='linear'
    )

    # Exécution sur une grille
    z, ss = OK.execute('points', points[:, 0], points[:, 1])
    return np.array(np.concat((points, z.reshape(-1, 1)), axis=1))

def local_kriging(data, point, m):

    z = np.zeros((len(point), 1))
    for i in range(len(point)):
        # find the closest m points to the given point
        distances = np.linalg.norm(data[:, :2] - point[i], axis=1)
        mask = np.argsort(distances)[:m]
        data_subset = data[mask]

        OK = OrdinaryKriging(

            data_subset[:, 0], data_subset[:, 1], data_subset[:, 2], 
            variogram_model='linear'
        )

        # Exécution pour un point spécifique
        z[i], ss = OK.execute('points', [point[i][0]], [point[i][1]])
    return np.array(np.concat((point, z.reshape(-1, 1)), axis=1))