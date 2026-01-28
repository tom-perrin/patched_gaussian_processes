from typing import NamedTuple
import json

import numpy as np

#Cloud types
GAUSSIAN_PROCESS = "GaussianProcesses"
UNIFORM = "Uniform" 

class GaussianProcess(NamedTuple):  
    mean : np.ndarray
    covariance : np.ndarray

def generate_gaussian_clouds(
        points_distribution: list[tuple[GaussianProcess, int]], 
        seed = None
    ) -> np.ndarray:

    rng = np.random.default_rng(seed)

    all_points = []

    for (gp, n_samples) in points_distribution:
        pts = rng.multivariate_normal(gp.mean, gp.covariance, n_samples)
        all_points.append(pts)

    # Stack into an array of shape (total_points, (dimension,))
    return np.vstack(all_points)

def generate_uniform_cloud(
        dimension : int,
        nb_sample: int,
        low: int,
        high: int,
        seed = None
    ) -> np.ndarray:

    rng = np.random.default_rng(seed)

    # Uniform distribution in [0, 1) for each dimension
    return rng.uniform(low=low, high=high, size=(nb_sample, dimension))


def cloud_from_file(
        json_path: str, 
        seed = None
    ) -> np.ndarray:

    with open(json_path, 'r', encoding='utf-8') as f:
        content_dict: dict = json.load(f)
    
    cloud_type : dict = content_dict["Type"]
    process: dict = content_dict[cloud_type]

    if cloud_type == GAUSSIAN_PROCESS:
        point_distributions = []

        for _,v in process.items():
            gp = GaussianProcess(
                mean=v["mean"],
                covariance=v["covariance"]
            )
            point_distributions.append((gp, v["nb_samples"]))
        
        return generate_gaussian_clouds(point_distributions, seed)
    

    elif cloud_type == UNIFORM:
        low, high = process.get("boundaries", None) or [0.0, 1.1]    
        return generate_uniform_cloud(
            process["dimension"], 
            process["nb_samples"],
            low,
            high,
            seed)
    
    else:
        raise NotImplementedError()