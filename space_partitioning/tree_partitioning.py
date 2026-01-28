import numpy as np

# --------------------------------------------------

def principal_direction(X: np.ndarray) -> np.ndarray:
    '''
    Computes the principal vector (first principal component) of a dataset (without scikit-learn)
    
    Parameters
    ----------
    X : np.ndarray
        Array of shape (n, d) containing n vectors of dimension d.

    Returns
    -------
    np.ndarray
        Vector of shape (d,) corresponding to the principal direction.
    '''
    # Centering data
    X_centered = X - np.mean(X, axis=0)

    n, d = X_centered.shape

    # Covariance matrix
    if d == 1:
        # For 1D, np.cov returns a scalar; convert to 2D
        cov = np.array([[np.var(X_centered, ddof=1)]])
    else:
        cov = np.cov(X_centered, rowvar=False)

    # Spectral decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Highest eigenvalue eigenvector = principal vector
    principal_vector = eigenvectors[:, np.argmax(eigenvalues)]

    # Ensure shape (d,)
    principal_vector = principal_vector.ravel()

    return principal_vector


class PartitioningTree:
    '''
    Binary tree representing the space partitioning along preferential directions.
    Each node contains the subset of data X it includes and its preferential direction.

    Parameters
    ----------
    X : np.ndarray
        Associated initial dataset.
    depth : int
        Amount of times the dataset will be recursively split when using attribute fully_extend().

    Attributes
    ----------
    X : Dataset of locations.
    Y : Dataset of values.
    depth : Amount of recursive subdivisions of the region.
    pdir : Preferential direction v of the dataset X.
    threshold : Value c of X@v at which the dataset is splitted in 2.
    left : Partitioning tree corresponding to the region where X@v <= c.
    right : Partitioning tree corresponding to the region where X@v > c.
    parent : If it exists, partitioning tree corresponding to the region from which this sub-region originates.

    Methods
    ----------
    extend : Splits into two sub partitioning trees. This defines pdir, threshold, left and right.
    fully_extend : Fully extends the partitioning tree into 2^depth leaves at final layer.
    get_leaves : Returns the list of leaves (partitioning trees).
    get_leaves_data : Returns the list of datasets X for each leave.
    '''
    def __init__(
            self,
            X: np.ndarray,
            depth: int,
            Y = None,
            pdir = None,
            threshold = None,
            left = None,
            right = None,
            parent = None
            ):
        self.X = X
        self.Y = Y
        self.depth = depth
        self.pdir = pdir # Principal vector v
        self.threshold = threshold # Threshold c used to split the dataset via PCA
        self.left = left # Sub-dataset X@v < c
        self.right = right # Sub-dataset X@v > c
        self.parent = parent
    
        # --- Bounding box attributes ---
        self.bb_min = None  # min sur chaque dimension
        self.bb_max = None  # max sur chaque dimension
        self._compute_bounding_box()  # calcul automatique à l'initialisation


    def _compute_bounding_box(self):
        '''
        Computes the bounding box of the current node's dataset X.
        Stores mins and maxs along each dimension in self.bb_min and self.bb_max.
        '''
        if self.X is not None and len(self.X) > 0:
            self.bb_min = np.min(self.X, axis=0)
            self.bb_max = np.max(self.X, axis=0)
        else:
            self.bb_min = None
            self.bb_max = None


    def extend(self):
        '''
        Splits into two subsets via PCA
        '''
        v = principal_direction(self.X)
        self.pdir = v

        # Computes threshold to split the data
        projections = self.X @ v
        self.threshold = np.median(projections) 
        
        # Splits the data into 2 subgroups (lesser and greater than the threshold)
        mask = projections <= self.threshold

        self.left = PartitioningTree(
            self.X[mask], 
            self.depth - 1,
            Y=self.Y[mask] if self.Y is not None else None,
            parent=self)
        
        self.right = PartitioningTree(
            self.X[~mask],
            self.depth - 1,
            Y=self.Y[~mask] if self.Y is not None else None,
            parent=self)
    

    def fully_extend(self):
        '''
        Fully extends the partitioning tree into 2^depth leaves at final layer
        '''
        # Final layer hit
        if self.depth <= 0:
            return
        
        # Extends current node
        self.extend()

        # Extends left and right sub-node if they exist
        if self.left:
            self.left.fully_extend()
        if self.right:
            self.right.fully_extend()
    

    def get_leaves(self):
        '''
        Returns the list of leaves (partitioning trees)
        '''
        if self.left is None and self.right is None:
            return [self]
        return self.left.get_leaves() + self.right.get_leaves()