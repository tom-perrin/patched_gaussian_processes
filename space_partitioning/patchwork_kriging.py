import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.spatial.distance import cdist

from graph_partitioning import PartitioningGraph
from grid_partitioning import PartitioningGrid

# --------------------------------------------------

def rbf_kernel(X1, X2, length_scale=1.0, variance=1.0):
    pass

def matern_kernel_32(X1, X2, length_scale=1.0, variance=1.0):
    pass

def matern_kernel_52(X1, X2, length_scale=1.0, variance=1.0):
    pass


class PatchworkKriging:
    '''
    Implements patchwork kriging.

    Parameters
    ----------
    partitioning : PartitioningGraph | PartitioningTree
        Represents the different regions
    kernel :
        Covariance function
    B : int
        Amount of pseudo-points generated at the frontier
    '''
    def __init__(self, 
                 partitioning, 
                 kernel, 
                 B=10,
                 noise_std=1e-6,
                 pseudo_noise=1e-6
                 ):
        self.partitioning = partitioning
        self.X = partitioning.nodes_X
        self.Y = partitioning.nodes_Y
        self.kernel = kernel
        self.B = B
        self.noise = noise_std
        self.pseudo_noise = pseudo_noise

        # Store frontiers and associated pseudo-points
        self.edges = []
        self.pseudo_coords = []
        self._generate_pseudo_points()

    def _generate_pseudo_points(self):
        f_matrix = self.partitioning.frontier_matrix
        N = len(f_matrix)
        for i in range(N):
            for j in range(i+1, N):
                segment = f_matrix[i][j]
                if segment is not None:
                    p1, p2 = segment

                    # Linear repartition of pseudo-points along the frontier
                    t = np.linspace(0.1, 0.9, self.B)
                    pts = np.array([(1-ti)*p1 + ti*p2 for ti in t])
                    self.edges.append((i,j))
                    self.pseudo_coords.append(pts)
            
            if self.pseudo_coords:
                self.all_Z = np.vstack(self.pseudo_coords)
            else:
                self.all_Z = np.array([]).reshape(0, 2)


    def precompute(self):
        '''
        Evaluates Q, L and v (cf. Algorithm 1)
        '''
        n_regions = len(self.X)
        n_edges = len(self.edges)
        n_pseudo = n_edges * self.B

        # Precompute Cholesky factors for C_kk = K(X_k, X_k) + sigma**2 * Id
        self.L_kk = []
        self.alpha_k = [] # C_kk**-1 * Y_k
        for x_k, y_k in zip(self.X, self.Y):
            C_kk = self.kernel(x_k, x_k) + np.eye(len(x_k)) * self.noise**2
            L_k = cholesky(C_kk, lower=True)
            self.L_kk.append(L_k)

            sol = solve_triangular(L_k, y_k, lower=True)
            self.alpha_k.append(solve_triangular(L_k.T, sol, lower=False))
        
        # Compute C_D_Delta : covariance between local data and boundary differences at pseudo-points
        C_D_Delta = []
        for k in range(n_regions):
            X_k = self.X[k]
            row_blocks = []
            for edge_idx, (i,j) in enumerate(self.edges):
                Z_edge = self.pseudo_coords[edge_idx]
                if k == i:
                    row_blocks.append(self.kernel(X_k, Z_edge))
                elif k == j:
                    row_blocks.append(-self.kernel(X_k, Z_edge))
                else:
                    row_blocks.append(np.zeros(len(X_k), self.B))
            C_D_Delta.append(np.hstack(row_blocks))
        C_D_Delta = np.vstack(C_D_Delta)

        # Compute C_Delta_Delta : covariance of differences at pseudo-points
        C_Delta_Delta = np.zeros((n_pseudo, n_pseudo))
        for idx1, (i, j) in enumerate(self.edges):
            for idx2, (k, l) in enumerate(self.edges):
                Z1 = self.pseudo_coords[idx1]
                Z2 = self.pseudo_coords[idx2]
                K_Z1Z2 = self.kernel(Z1, Z2)

                # Use formula Cov(fi-fj, fk-fl) = K(zi, zk) - K(zi, zl) - K(zj, zk) + K(zj, zl)
                coeff = 0
                if i == k:
                    coeff += 1
                if i == l:
                    coeff -= 1
                if j == k:
                    coeff -= 1
                if j == l:
                    coeff += 1
                
                start_row, start_col = idx1 * self.B, idx2 * self.B
                C_Delta_Delta[start_row:start_row+self.B, start_col:start_col+self.B] = coeff * K_Z1Z2
        
        # Add noise for numerical stability
        C_Delta_Delta += np.eye(n_pseudo) * self.pseudo_noise**2

        # Compute M = C_Delta_Delta - C_D_Delta.T @ C_D_D**-1 @ C_D_Delta
        invCDD_CD_Delta = []
        current_row = 0
        for k in range(n_regions):
            nk = len(self.X[k])
            block = C_D_Delta[current_row : current_row + nk, :]
            sol = solve_triangular(self.L_kk[k], block, lower=True)
            invCDD_CD_Delta.append(solve_triangular(self.L_kk[k].T, sol, lower=False))
            current_row += nk
        invCDD_CD_Delta = np.vstack(invCDD_CD_Delta)

        M = C_Delta_Delta - C_D_Delta.T @ invCDD_CD_Delta
        self.L_M = cholesky(M, lower=True)

    
    def predict(self, X_star):
        '''
        Predicts mean and variance at locations X_star
        ''' 
        pass
