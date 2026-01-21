import numpy as np
import matplotlib.pyplot as plt
from tree_partitioning import PartitioningTree
from graph_partitioning import get_clipped_frontier 

def _prepare():
    plt.figure()

def _show():
    plt.axhline(0, color="gray", linewidth=0.5)
    plt.axvline(0, color="gray", linewidth=0.5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.show()

def display_groups(groups: list[np.ndarray], size=10, alpha=0.7) -> None:
    for points in groups:
        plt.scatter(
            points[:, 0], 
            points[:, 1],
            s=size, 
            alpha=alpha
        )

def plot_tree_boundaries(node: PartitioningTree):
    '''
    !!! LEGACY !!!
    '''
    if node.left is None or node.right is None:
        return

    v = node.pdir
    c = node.threshold
    
    # Calculates the line segment for the frontier
    x_min, x_max = node.X[:, 0].min(), node.X[:, 0].max()
    y_min, y_max = node.X[:, 1].min(), node.X[:, 1].max()

    if abs(v[1]) > 1e-5: # Avoid division by zero for vertical lines
        f_x = np.array([x_min, x_max])
        f_y = (c - v[0] * f_x) / v[1]
    else:
        f_y = np.array([y_min, y_max])
        f_x = np.array([c / v[0], c / v[0]])

    # Plots frontier line
    plt.plot(f_x, f_y, 'k--', lw=1, alpha=0.9)

    # Recursion
    plot_tree_boundaries(node.left)
    plot_tree_boundaries(node.right)

def plot_tree_boundaries_v2(node: PartitioningTree):
    if node is None or (node.left is None and node.right is None):
        return
    
    # Plot frontier line
    segment = get_clipped_frontier(node)
    if segment is not None:
        p1, p2 = segment
        plt.plot(
            [p1[0], p2[0]], [p1[1], p2[1]],
            'k--',
            lw = 1,
            alpha = 0.9,
            zorder = 10,
        )
    
    # Recursion
    plot_tree_boundaries_v2(node.left)
    plot_tree_boundaries_v2(node.right)


def plot_tree(tree: PartitioningTree, size=10, alpha=0.7):
    '''
    Prints datasets of all leaves and boundaries of the partitioning tree.
    '''
    _prepare()
    leaves_data = tree.get_leaves_data()
    display_groups(leaves_data)
    plot_tree_boundaries_v2(tree)
    _show()