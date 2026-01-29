from point_cloud import cloud_from_file
from display import display_global, animate_nearest_neighbors
from krigeage import kriging, local_kriging
import numpy as np
# --------------------------------------------------

CLOUD_FOLDER = "clouds/"

#CLOUD_FILE = "gaussian_cloud.json"
CLOUD_FILE = "uniform_cloud.json"
SEED = 67

if __name__ == "__main__":
    print("Le script est lancé directement.")
    X_data = cloud_from_file(CLOUD_FOLDER+CLOUD_FILE, SEED)
    X_pred = np.array([[0,0.5],[0.5,0.5],[0.5,0],[0.25,0.25],[0.75,0.75]])
    new_data = local_kriging(X_data, X_pred, 20)
    animate_nearest_neighbors( new_data, X_data, 20)