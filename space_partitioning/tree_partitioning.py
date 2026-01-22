import numpy as np
import matplotlib.pyplot as plt

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

    # Covariance matrix
    cov = np.cov(X_centered, rowvar=False)

    # Spectral decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Highest eigenvalue eigenvector = principal vector
    principal_vector = eigenvectors[:, np.argmax(eigenvalues)]

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
    '''
    def __init__(
            self,
            X: np.ndarray,
            depth: int,
            pdir = None,
            threshold = None,
            left = None,
            right = None,
            parent = None
            ):
        self.X = X
        self.depth = depth
        self.pdir = pdir # Principal vector v
        self.threshold = threshold # Threshold c used to split the dataset via PCA
        self.left = left # Sub-dataset X@v < c
        self.right = right # Sub-dataset X@v > c
        self.parent = parent
    
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
        self.left = PartitioningTree(self.X[mask], self.depth - 1, parent=self)
        self.right = PartitioningTree(self.X[~mask], self.depth - 1, parent=self)
    
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

    def get_leaves_data(self) -> list[np.ndarray]:
        '''
        Returns the list of datasets X for each leave
        (Only used to display)
        '''
        if self.left is None and self.right is None:
            return [self.X]
        return self.left.get_leaves_data() + self.right.get_leaves_data()