from typing import Callable
import numpy as np

from scipy.linalg import cholesky, solve_triangular, block_diag
from scipy.spatial.distance import cdist

from graph_partitioning import PartitioningGraph
from grid_partitioning import PartitioningGrid

# --------------------------------------------------
def linear_kernel(X, Y): 
    '''
    For testing purposes only (do not use)
    ''' 
    raise Exception("Do not use linear kernel")
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    return X @ Y.T

def rbf_kernel(X, Y, length_scale=1.0, variance=1.0):
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)

    # ||x - y||^2 = x^2 + y^2 - 2xy
    X2 = np.sum(X**2, axis=1)[:, None]   # (N, 1)
    Y2 = np.sum(Y**2, axis=1)[None, :]   # (1, M)

    sqdist = X2 + Y2 - 2 * X @ Y.T
    return variance * np.exp(-0.5 * sqdist / length_scale**2)

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
        Covariance function (problem is assumed to be stationary)
    B : int
        Amount of pseudo-points generated at each frontier
    '''
    def __init__(self, 
                 partitioning: PartitioningGraph, 
                 kernel, 
                 B = 10,
                 noise_std=1e-6,
                 pseudo_noise=1e-6  # For numerical stability
                 ):
        
        self.partitioning : PartitioningGraph = partitioning
        self.X : list[np.ndarray] = partitioning.nodes_X
        self.Y : list[float] = partitioning.nodes_Y
        self.kernel : Callable = kernel
        self.B : int = B
        self.noise : float = noise_std
        self.pseudo_noise : float = pseudo_noise

        # Store frontiers and associated pseudo-points
        self.edges = []
        self.pseudo_coords : list[np.ndarray] = []
        self._generate_pseudo_points()

    def _generate_pseudo_points(self):
        f_matrix = self.partitioning.frontier_matrix
        N = len(f_matrix)

        for i in range(N):
            for j in range(i+1, N):
                segment = f_matrix[i][j]
                if segment is not None:
                    p1, p2 = segment

                    # Pseudo observations are sampled with a uniform law
                    t = np.random.uniform(0.1, 0.9, size=self.B)
                    pts = (1 - t)[:, None] * p1 + t[:, None] * p2 # shape = (B,d)

                    self.edges.append((i, j))
                    self.pseudo_coords.append(pts)
            
        if self.pseudo_coords:
            self.all_Z = np.vstack(self.pseudo_coords)
        else:
            self.all_Z = np.array([]).reshape(0, 2)

    def precompute(self):
        '''
        Evaluates Q, L and v (cf. Algorithm 1)
        '''
        n_regions = len(self.X)         # K
        n_edges = len(self.edges)       # non empty edges
        n_pseudo = n_edges * self.B     #

        # Precompute Cholesky factors for C_DD = K(X_k, X_k) + sigma**2 * Id
        # C_DD is block diagonal, we're computing the blocks. 
        self.L_kk = []
        self.alpha_k = [] # C_kk**-1 * Y_k
        CDD_blocks = []

        for x_k, y_k in zip(self.X, self.Y):
            C_kk = self.kernel(x_k, x_k) + np.eye(len(x_k)) * self.noise**2
            CDD_blocks.append(C_kk)


            L_k = cholesky(C_kk, lower=True)
            self.L_kk.append(L_k)

            sol = solve_triangular(L_k, y_k, lower=True)
            self.alpha_k.append(solve_triangular(L_k.T, sol, lower=False))
        
        # print(
        #     f"{n_regions=}",
        #     f"{n_edges=}",
        #     f"{n_pseudo=}",
        #     sep="\n"
        # )
        # raise SystemExit

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
                    row_blocks.append(np.zeros((len(X_k), self.B)))
            C_D_Delta.append(np.hstack(row_blocks))
        C_D_Delta = np.vstack(C_D_Delta)

        # Compute C_Delta_Delta : covariance of differences at pseudo-points
        C_Delta_Delta = np.zeros((n_pseudo, n_pseudo))
        for idx1, (k, l) in enumerate(self.edges):
            for idx2, (u, v) in enumerate(self.edges):
                Z1 = self.pseudo_coords[idx1]
                Z2 = self.pseudo_coords[idx2]

                # Use formula (4) of paper. kinda heavy but clear
                if (
                    (k == u and l != v) or 
                    (l == v and k != u)
                ):
                    K_Z1Z2 = self.kernel(Z1, Z2)
                
                elif (
                    (k == v and l != u) or 
                    (l == u and k != v)    
                ):
                    K_Z1Z2 = - self.kernel(Z1, Z2)
                
                elif k==u and l==v:
                    K_Z1Z2 = 2 * self.kernel(Z1, Z2)
                
                else:
                    K_Z1Z2 = np.zeros((self.B, self.B))

                
                start_row, start_col = idx1 * self.B, idx2 * self.B
                C_Delta_Delta[start_row:start_row+self.B, start_col:start_col+self.B] = K_Z1Z2
        
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

        CDD_inv = np.linalg.inv(block_diag(*CDD_blocks))

        self.Q = np.linalg.inv(
            block_diag(*CDD_blocks) - C_D_Delta @ np.linalg.inv(C_Delta_Delta) @ C_D_Delta.T
        )

        # self.Q = CDD_inv + invCDD_CD_Delta @ M @ C_D_Delta.T @ CDD_inv
        L = cholesky(C_Delta_Delta, lower=True)
        self.L_inv = np.linalg.inv(L)

        self.v = self.L_inv @ C_D_Delta.T

        

        # M_inv = np.linalg.inv(M)

        # self.Q = (
        #     CDD_inv
        #     - CDD_inv @ C_D_Delta @ M_inv @ C_D_Delta.T @ CDD_inv
        # )

        # L = cholesky(M, lower=True)
        # self.L_inv = solve_triangular(L, np.eye(L.shape[0]), lower=True)

        # self.v = self.L_inv @ C_D_Delta.T


    def predict(self, X_star, region_idx: int):
        '''
        Predicts mean and variance at locations X_star
        ''' 
        region_idx = self.partitioning.find_location(X_star)

        if not region_idx:
            raise Exception(f"Couldn't find region for point {X_star}")

        # Compute of covariance between X_star and observations D and Delta
        # c_star_Delta
        covs = []
        for idx, (i, j) in enumerate(self.edges):
            obs = self.pseudo_coords[idx]               # (B, d)

            if region_idx == i:
                # K retourne (1, B)
                cov = self.kernel(X_star, obs)
            
            elif region_idx == j:
                cov = - self.kernel(X_star, obs)

            else:
                cov = np.zeros((1, self.B))

            covs.append(cov)

        c_star_Delta = np.hstack(covs) # taille (1, B * L)

        # c_star_D
        covs = []

        for i, X_i in enumerate(self.X):
            # X_i : (N_i, d)

            if i == region_idx:
                # covariance avec les observations du domaine region_idx
                cov = self.kernel(X_star, X_i)        # (1, N_i)
            else:
                # zéros pour les autres domaines (indépendance supposée)
                cov = np.zeros((1, X_i.shape[0]))

            covs.append(cov)

        c_star_D = np.hstack(covs)

        # c_star_star
        c_star_star = self.kernel(X_star, X_star)

        # Compute of w_star
        w_star = self.L_inv @ c_star_Delta.T

        # Expectation and variance
        mid_term = (c_star_D - w_star.T @ self.v)

        E_star = mid_term @ self.Q @ np.hstack(self.Y)
        V_star = c_star_star - w_star.T @ w_star - mid_term @ self.Q @ mid_term.T

        return E_star, V_star
