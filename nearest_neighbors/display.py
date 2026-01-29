import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation

# --------------------------------------------------

def display_global( X_data : np.ndarray, new_data : np.ndarray) -> None:
    """Display a 3D point in 2D cloud with a colorbar as third dimension."""

    x = X_data[:, 0]
    y = X_data[:, 1]
    z = X_data[:, 2]

    x0 = new_data[:, 0]
    y0 = new_data[:, 1]
    z0 = new_data[:, 2]

    plt.figure(figsize=(8,6))
    scatter = plt.scatter(x, y, c=z, cmap='viridis',marker='o', label='Data', alpha=0.6)
    scatter2 =plt.scatter(x0, y0, cmap='viridis', marker='x', label='Kriging Points')
    plt.colorbar(scatter, label='Z value')
    plt.title('Global Kriging')
    plt.grid(True)
    plt.show()
    return None


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial import KDTree

def animate_nearest_neighbors(
    points1: np.ndarray,
    points2: np.ndarray,
    m: int,
):
    v_min, v_max = points2[:, 2].min(), points2[:, 2].max()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal', adjustable='box')
    # Initialisation des plots
    # Scatter pour le point actuel de points1
    scatter_p1 = ax.scatter(points1[0, 0], points1[0, 1], c=points1[0, 2], marker='+',
                            vmin=v_min, vmax=v_max, cmap='viridis', s=100, label='Target Point ')
    
    # Scatter pour les autres points de points2 (transparents)
    scatter_p2_others = ax.scatter(
        points2[:, 0], points2[:, 1], marker='o',
        color='lightgray',
    )
    distances = np.linalg.norm(points2[:, :2] - points1[0,:2], axis=1)
    mask = np.argsort(distances)[:m]
    data_subset = points2[mask]
    
    # Scatter pour les m plus proches voisins de points2
    scatter_p2_neighbors = ax.scatter(data_subset[:, 0], data_subset[:, 1], c=data_subset[:, 2], cmap='viridis',
                                      vmin=v_min, vmax=v_max, label=' Nearest Neighbors (P2)')
    fig.colorbar(scatter_p2_neighbors, label='Z value')
    circle = plt.Circle((0, 0), 0, color='grey', fill=False, 
                            linestyle='--', alpha=0.5, linewidth=1.5)
    ax.add_patch(circle)


    def update(frame):
        """
        Fonction appelée pour chaque frame du GIF.
        """
        current_point = points1[frame]
        
        # Trouver les 'm' plus proches voisins de current_point dans points2
        distances = np.linalg.norm(points2[:, :2] - points1[frame,:2], axis=1)
        mask = np.argsort(distances)[:m]
        data_subset = points2[mask]
        radius = distances[mask][-1]  
        # Mettre à jour le point actuel de points1
        scatter_p1.set_offsets(current_point[:2].reshape(1, 2))
        scatter_p1.set_array([current_point[2]])

        # Mettre à jour les plus proches voisins
        scatter_p2_neighbors.set_offsets(data_subset[:, :2])
        scatter_p2_neighbors.set_array(data_subset[:, 2])


        # 2. Mise à jour des voisins
        scatter_p2_neighbors.set_offsets(data_subset[:, :2])
        
        
        # 3. Mise à jour du cercle
        circle.set_center((current_point[0], current_point[1]))
        circle.set_radius(radius)

        
        return [scatter_p1, scatter_p2_neighbors, circle] # Retourner les objets modifiés pour l'animation

    ani = FuncAnimation(
        fig,
        update,
        frames=len(points1),
        interval=2 * 1000, # Durée entre les frames en millisecondes
        blit=True, # Optimisation pour ne redessiner que les éléments qui changent
        repeat=True
    )

    ani.save('anim.gif', writer='pillow', dpi=100)
    return None
