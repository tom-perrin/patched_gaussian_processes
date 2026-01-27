from abc import ABC, abstractmethod

import numpy as np

class Partionning:
    '''
    Abstract class that represent a generic partionning
    May be
        - Space Tree Partionning
        - Uniform Grid Partitionning

    In either case, a partionning should be able to deliver the
    following attributes.
    '''


    @abstractmethod
    def get_adjacency_matrix(self) -> np.ndarray:
        '''
        Get the matrix of adjency (boolean) between nodes of the partitionning
        
        :return: numpy array of shape (nb_nodes, nb_nodes) 
        :rtype: numpy.ndarray
        '''
        pass

    @abstractmethod
    def get_frontier_matrix(self) -> list[list[np.ndarray | None]]:
        '''
        Get the fontier matrix
        
        :param self: Description
        :return: Description
        :rtype: ndarray
        '''
        pass

    @abstractmethod
    def find_location(self, x_star: np.ndarray) -> int:
        '''
        Find the node where x_star is located
        
        :param x_star: vector to be tested
        :return: the index of the region where x_start is
        :rtype: numpy.ndarray
        '''

