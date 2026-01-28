from point_cloud import cloud_from_file
from display import plot_tree, plot_grid, plot_prediction_along_line
from plot3d import plot_field_3d_plotly,plot_predicted_field_3d_plotly, plot_regions_2d_plotly
from utils import predict_along_segment, predict_single_point, predict_on_grid_2d

from tree_partitioning import PartitioningTree
from graph_partitioning import PartitioningGraph
from grid_partitioning import PartitioningGrid
from simulated_field import analytic_field

from patchwork_kriging import PatchworkKriging, rbf_kernel

# --------------------------------------------------

CLOUD_FOLDER = "clouds/"

# CLOUD_FILE = "gaussian_cloud.json"
CLOUD_FILE = "uniform_cloud.json"
SEED = 67
DEPTH = 3


if __name__ == "__main__":
    # Testing
    X_data = cloud_from_file(CLOUD_FOLDER+CLOUD_FILE, SEED)
    Y_data = analytic_field(X_data, sigma = 0.05)

    tree = PartitioningTree(X_data, DEPTH, Y_data)
    tree.fully_extend()
    # plot_tree(tree)

    graph = PartitioningGraph(tree)

    pk = PatchworkKriging(
        graph,
        rbf_kernel,
        1,
        pseudo_noise=1e-3,
        noise_std=1e-3,
    )
    
    pk.precompute()


    # # single point prediciton; 3D viewing
    # X_pred, Y_pred, V_pred, region = predict_single_point(pk, [[1,1]])

    # plot_field_3d_plotly(
    #     X_data, Y_data,
    #     X_obs=None, Y_obs=None,
    #     X_pred=X_pred, Y_pred=Y_pred,
    #     title="Champ analytique, observations et prédiction"
    # )


    # # prediction along line; 2d representation
    # p1 = [1.9, -1.5]
    # p2 = [1.9, 1.5]

    # t, X_pred, Y_pred, var_pred, regions = predict_along_segment(
    #     pk,
    #     p1,
    #     p2,
    #     n_points=100
    # )

    # plot_prediction_along_line(
    #     t,
    #     Y_pred,
    #     var_pred,
    #     regions,
    #     show_std=True
    # )


    # prediction on a plane
    Xg, Yg, Z_pred, Z_var, Z_region, mask = predict_on_grid_2d(
        pk,
        x_bounds=(-2.0, 2.0),
        y_bounds=(-2.0, 2.0),
        nx=80,
        ny=80
    )

    plot_predicted_field_3d_plotly(
        Xg, Yg,
        Z_pred,
        Z_region,
        mask,
        title="Patchwork Kriging – prédiction 3D"
    )