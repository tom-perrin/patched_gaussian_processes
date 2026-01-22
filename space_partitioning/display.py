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
    plot_tree_boundaries(node.left)
    plot_tree_boundaries(node.right)


def plot_tree(tree: PartitioningTree, size=10, alpha=0.7):
    '''
    Prints datasets of all leaves and boundaries of the partitioning tree.
    '''
    _prepare()
    leaves_data = tree.get_leaves_data()
    display_groups(leaves_data)
    plot_tree_boundaries(tree)
    _show()