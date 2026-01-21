from point_cloud import cloud_from_file
from display import plot_tree

from tree_partitioning import PartitioningTree

CLOUD_FOLDER = "clouds/"

CLOUD_FILE = "gaussian_cloud.json"
# CLOUD_FILE = "uniform_cloud.json"
SEED = 67
DEPTH = 3

""" if __name__ == "__main__":
    # Generate points (n,d)
    points = cloud_from_file(
        CLOUD_FOLDER + CLOUD_FILE,
        SEED
    )

    point_groups = space_tree_partionning(
        points,
        DEPTH
    )

    display_groups(point_groups) """

X_data = cloud_from_file(CLOUD_FOLDER+CLOUD_FILE, SEED)

tree = PartitioningTree(X_data, DEPTH)
tree.fully_extend()

plot_tree(tree)