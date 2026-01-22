from point_cloud import cloud_from_file
from display import plot_tree, plot_grid

from tree_partitioning import PartitioningTree
from graph_partitioning import PartitioningGraph
from grid_partitioning import PartitioningGrid

# --------------------------------------------------

CLOUD_FOLDER = "clouds/"

CLOUD_FILE = "gaussian_cloud.json"
# CLOUD_FILE = "uniform_cloud.json"
SEED = 67
DEPTH = 3

N_DIV, P_DIV = 3, 3


def get_frontier_matrix(partitioning):
    '''
    Get the matrix of frontiers
    partitioning can be either a PartitioningGraph or a PartitioningGrid
    '''
    return partitioning.frontier_matrix


X_data = cloud_from_file(CLOUD_FOLDER+CLOUD_FILE, SEED)

tree = PartitioningTree(X_data, DEPTH)
tree.fully_extend()
plot_tree(tree)

grid = PartitioningGrid(X_data, N_DIV, P_DIV)
plot_grid(grid)