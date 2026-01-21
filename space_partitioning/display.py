import numpy as np
import matplotlib.pyplot as plt

def _prepare():
    plt.figure(figsize=(6, 6))

def _show():
    plt.axhline(0, color="gray", linewidth=0.5)
    plt.axvline(0, color="gray", linewidth=0.5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.show()

def display_cloud(points: np.ndarray) -> None:
    _prepare()

    plt.scatter(points[:, 0], points[:, 1], s=8, alpha=0.6)
    
    _show()

def display_groups(groups: list[np.ndarray]) -> None:
    _prepare()

    for points in groups:
        plt.scatter(
            points[:, 0], 
            points[:, 1],
            s=20, 
            alpha=0.6
        )

    _show()