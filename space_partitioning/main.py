from point_cloud import cloud_from_file
from space_partitioning import space_tree_partionning
from display import(
    display_groups
)

CLOUD_FOLDER = "clouds/"

CLOUD_FILE = "gaussian_cloud.json"
# CLOUD_FILE = "uniform_cloud.json"
SEED = 67
DEPTH = 3

if __name__ == "__main__":
    # Generate points (n,d)
    points = cloud_from_file(
        CLOUD_FOLDER + CLOUD_FILE,
        SEED
    )

    point_groups = space_tree_partionning(
        points,
        DEPTH
    )

    display_groups(point_groups)
