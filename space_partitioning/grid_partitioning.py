import numpy as np
import networkx as nx

# --------------------------------------------------

class PartitioningGrid:
    '''
    Representation of a uniform grid partition of a dataset.

    Parameters
    ----------
    X : np.ndarray
        Associated initial dataset.
    n_div : int
        Amount of subdivisions of the x-axis.
    p_div : int
        Amount of subdivisions of the y-axis.

    Attributes
    ----------
    X : Dataset of locations.
    Y : Dataset of values.
    n : Amount of subdivisions of the x-axis.
    p : Amount of subdivisions of the y-axis.
    x_min, x_max, y_min, y_max : Bounds of the region.
    dx, dy : Sides of sub-regions.
    nodes : List of all sub datasets.
    mapping : Mapping of the region indexes.
    G : Adjacency graph of the regions.

    Methods
    ----------
    frontier : Computes the 2 end points of the frontier between regions i and j (if they are neighbors).
    adjacency_matrix : Computes the adjacency matrix of the regions.
    frontier_matrix : Computes the matrix of all frontiers between regions (both end points if they exist).
    '''
    def __init__(
            self,
            X: np.ndarray,
            n_div: int,
            p_div: int,
            Y = None
            ):
        self.X = X
        self.n = n_div
        self.p = p_div

        self.x_min, self.y_min = X.min(axis=0)
        self.x_max, self.y_max = X.max(axis=0)

        self.dx = (self.x_max - self.x_min) / n_div
        self.dy = (self.y_max - self.y_min) / p_div

        # Redefines mapping of regions : (i,j) --> j*p_div + i
        self.nodes_X = []
        self.nodes_Y = [] if Y is not None else None

        mapping = {}
        counter = 0
        for j in reversed(range(self.p)):
            for i in range(self.n):
                # Calculate bounds for this specific cell
                x_s = self.x_min + i * self.dx
                x_e = self.x_min + (i + 1) * self.dx
                y_s = self.y_min + j * self.dy
                y_e = self.y_min + (j + 1) * self.dy
                
                # Store the point subset for this index
                is_last_col = (i == self.n - 1) # Check if last column : if so, include last frontier 
                is_last_row = (j == 0) # Check if last row : if so, include last frontier

                mask_x = (X[:, 0] >= x_s) & (X[:, 0] <= x_e if is_last_col else X[:, 0] < x_e)
                mask_y = (X[:, 1] >= y_s) & (X[:, 1] <= y_e if is_last_row else X[:, 1] < y_e)
                mask = mask_x & mask_y
                self.nodes_X.append(X[mask])
                if Y is not None:
                    self.nodes_Y.append(Y[mask])
                
                # Create mapping for the graph
                mapping[(i, j)] = counter
                counter += 1
        
        G_raw = nx.grid_2d_graph(n_div, p_div)
        self.G = nx.relabel_nodes(G_raw, mapping)
    

    def frontier(self, i, j):
        '''
        Computes the 2 end points of the frontier between regions i and j (if they are neighbors)
        '''
        if not self.G.has_edge(i,j):
            return None
        
        # Get coordinates of sub regions
        def get_cords(x):
            return x % self.n, (self.p - 1) - x // self.n
        i_col, i_row = get_cords(i)
        j_col, j_row = get_cords(j)

        # Check vertical frontier
        if i_row == j_row:
            x_shared = self.x_min + max(i_col, j_col) * self.dx
            y_bottom = self.y_min + i_row * self.dy
            y_top = y_bottom + self.dy
            return np.array([x_shared, y_bottom]), np.array([x_shared, y_top])
        
        # Check horizontal frontier
        if i_col == j_col:
            y_shared = self.y_min + max(i_row, j_row) * self.dy
            x_left = self.x_min + i_col * self.dx
            x_right = x_left + self.dx
            return np.array([x_left, y_shared]), np.array([x_right, y_shared])
    
        return None
    

    @property
    def adjacency_matrix(self):
        '''
        Computes the adjacency matrix of the regions
        '''
        return nx.to_numpy_array(self.G, nodelist=range(len(self.nodes_X)))

    
    @property
    def frontier_matrix(self):
        '''
        Computes the matrix of all frontiers between regions (both end points if they exist)
        '''
        if not hasattr(self, '_cached_f_matrix'):
            N = len(self.nodes)
            self._cached_f_matrix = [[self.frontier(i, j) for j in range(N)] for i in range(N)]
        return self._cached_f_matrix


    def find_location(self, x_star):
        '''
        Finds the index of the region where a new location x_star is located
        '''
        pass