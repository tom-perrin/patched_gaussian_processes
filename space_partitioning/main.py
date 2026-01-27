from point_cloud import cloud_from_file
from display import plot_tree, plot_grid
from plot3d import plot_field_3d_plotly

from tree_partitioning import PartitioningTree
from graph_partitioning import PartitioningGraph
from grid_partitioning import PartitioningGrid
from simulated_field import analytic_field

from patchwork_kriging import PatchworkKriging, rbf_kernel

# --------------------------------------------------

CLOUD_FOLDER = "clouds/"

CLOUD_FILE = "gaussian_cloud.json"
# CLOUD_FILE = "uniform_cloud.json"
SEED = 67
DEPTH = 3

N_DIV, P_DIV = 3, 3

def subsample_field(X_field, Y_field, ratio, seed=None):
    """
    ratio ∈ (0, 1] : fraction de points conservés
    """
    assert 0 < ratio <= 1.0

    N = X_field.shape[0]

    rng = np.random.default_rng(seed)
    idx = rng.choice(N, size=int(ratio * N), replace=False)

    return X_field[idx], Y_field[idx]

if __name__ == "__main__":
    # Testing
    X_data = cloud_from_file(CLOUD_FOLDER+CLOUD_FILE, SEED)
    Y_data = analytic_field(X_data)

    tree = PartitioningTree(X_data, DEPTH, Y_data)
    tree.fully_extend()
    plot_tree(tree)

    graph = PartitioningGraph(tree)

    pk = PatchworkKriging(
        graph,
        rbf_kernel,
        30,
        pseudo_noise=1e-3
    )
    
    pk.precompute()

    import numpy as np
    X_pred = np.array([[0.2, 0.8]])

    Y_pred, V = pk.predict(X_pred, 1)

    print(
        f"{Y_pred=}",
        f"{V=}",
        sep='\n'
    )

    # grid = PartitioningGrid(X_data, N_DIV, P_DIV, Y_data)
    # plot_grid(grid)

    # Prédictions
    
    X, Y = subsample_field(X_data, Y_data, 0.05)

    plot_field_3d_plotly(
        X_data, Y_data,
        X_obs=None, Y_obs=None,
        X_pred=X_pred, Y_pred=Y_pred,
        title="Champ analytique, observations et prédiction"
    )