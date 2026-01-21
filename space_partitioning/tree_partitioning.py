import numpy as np
import matplotlib.pyplot as plt
from space_partitioning import principal_direction

class PartitioningTree:
    '''
    Binary tree representing the space partitioning along preferential directions.
    Each node contains the subset of data X it includes and its preferential direction. 
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
        self.pdir = pdir
        self.threshold = threshold
        self.left = left
        self.right = right
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
        Fully extends the partitioning tree
        '''
        if self.depth <= 0:
            return
        
        self.extend()

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
        '''
        if self.left is None and self.right is None:
            return [self.X]
        return self.left.get_leaves_data() + self.right.get_leaves_data()