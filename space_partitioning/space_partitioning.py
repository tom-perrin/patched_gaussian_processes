import numpy as np

###
# http://stackoverflow.com/questions/20507646/how-is-the-complexity-of-pca-ominp3-n3
# Covariance matrix computation is O(p^2*n); 
# its eigen-value decomposition is O(p^3). 
# So, the complexity of PCA is O(p^2*n+p^3).

def principal_direction(X: np.ndarray) -> np.ndarray:
    """
    Calcule le vecteur principal (première composante principale) d'un jeu de données.
    (sans utiliser sci kit learn)
    
    Parameters
    ----------
    X : np.ndarray
        Tableau de forme (n, d) contenant n vecteurs en dimension d.

    Returns
    -------
    np.ndarray
        Vecteur de taille (d,) correspondant à la direction principale.
    """
    # Centrage des données
    X_centered = X - np.mean(X, axis=0)

    # Matrice de covariance (d x d)
    cov = np.cov(X_centered, rowvar=False)

    # Décomposition spectrale (matrice symétrique → eigh)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Vecteur propre associé à la plus grande valeur propre
    principal_vector = eigenvectors[:, np.argmax(eigenvalues)]

    return principal_vector

def split_by_projection(X: np.ndarray, v: np.ndarray):
    """
    Sépare un jeu de données en deux sous-jeux de même taille
    selon la projection sur un vecteur donné.

    Parameters
    ----------
    X : np.ndarray
        Tableau de forme (n, d)
    v : np.ndarray
        Vecteur de direction de forme (d,)

    Returns
    -------
    X_low : np.ndarray
        Sous-jeu correspondant aux projections les plus faibles
    X_high : np.ndarray
        Sous-jeu correspondant aux projections les plus élevées
    """
    # Produits scalaires (n,)
    projections = X @ v

    # Indices de tri croissant
    sorted_idx = np.argsort(projections)

    half = X.shape[0] // 2

    # Découpage
    X_low = X[sorted_idx[:half]]
    X_high = X[sorted_idx[half:]]

    return [X_low, X_high]

def recursive_partition(
        depth: int, 
        point_groups: list[np.ndarray]
    ) -> list[np.ndarray]:

    if depth <= 0:
        return point_groups

    new_list = []

    for points in point_groups:
        v = principal_direction(points)
        sub_groups = split_by_projection(points,v)
        new_list.extend(recursive_partition(depth - 1, sub_groups))
    
    return new_list

def space_tree_partionning(
        points: np.ndarray, 
        depth: int,
    ) -> list[np.ndarray]:
    """
        points : shape (n, d), n points of dimension d.
        depth : depth of tree, cloud will be partitionned in 2^depth groups.
    """

    return recursive_partition(depth, [points])
