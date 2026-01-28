import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from tree_partitioning import PartitioningTree
from graph_partitioning import PartitioningGraph, get_clipped_frontier 
from grid_partitioning import PartitioningGrid

# --------------------------------------------------

def _prepare():
    plt.figure()

def _show():
    plt.axhline(0, color="gray", linewidth=0.5)
    plt.axvline(0, color="gray", linewidth=0.5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.show()

def _draw_boundaries(node: PartitioningTree, ax: plt.Axes):
        if node.left is None: return
        segment = get_clipped_frontier(node)
        if segment is not None:
            p1, p2 = segment
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k--', lw=1, alpha=0.9)
        _draw_boundaries(node.left, ax)
        _draw_boundaries(node.right, ax)


def display_groups(groups: list[np.ndarray], ax: plt.Axes, size=10, alpha=0.7) -> None:
    for i, points in enumerate(groups):
        # Skip points if empty group
        if points.shape[0] == 0:
            continue

        ax.scatter(
            points[:, 0], 
            points[:, 1],
            s=size, 
            alpha=alpha,
            label=f'Group {i}'
        )

        # Center the label
        center = np.mean(points, axis=0)
        ax.text(
            center[0], center[1], str(i),
            fontsize=12,
            fontweight='bold',
            ha='center', va='center',
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='grey', boxstyle='round,pad=0.4')
        )


def plot_tree(tree: PartitioningTree, size=10, alpha=0.7):
    '''
    Prints datasets and boundaries & adjacency graph of the partition side by side
    '''
    # Define graph and figure
    graph = PartitioningGraph(tree)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    ### --- Left Side : Partitioned Space ---
    ax1.set_title('Partitioned Space')

    leaves_loc = [leaf.X for leaf in tree.get_leaves()]

    display_groups(leaves_loc, ax1)
    _draw_boundaries(tree, ax1)
    ax1.axis('equal')
    ax1.grid(True, linestyle=':', alpha=0.4)

    ### --- Right Side : Adjacency Graph ---
    ax2.set_title('Region Adjacency Graph')
    pos = nx.spring_layout(graph.G, seed=67)
    
    nx.draw(
        graph.G, pos, ax=ax2, 
        with_labels=True, 
        node_color='mediumpurple', 
        node_size=600, 
        font_weight='bold',
        edge_color='gray'
    )

    # Show figure
    plt.tight_layout()
    plt.show()


def plot_grid(grid: PartitioningGrid, size=10, alpha=0.7):
    '''
    Docstring for plot_grid
    '''
    # Define graph and figure
    X = grid.X
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # --- Left Side : Partitioned Space ---
    ax1.set_title('Partitioned Space')
    display_groups(grid.nodes_X, ax1, size=size, alpha=alpha)

    # Grid lines
    for i in range(grid.n + 1): 
        ax1.axvline(grid.x_min + i*grid.dx, color='k', linestyle='--', lw=1, alpha=0.9)
    for j in range(grid.p + 1): 
        ax1.axhline(grid.y_min + j*grid.dy, color='k', linestyle='--', lw=1, alpha=0.9)

    ax1.axis('equal')
    ax1.grid(True, linestyle=':', alpha=0.4)

    # --- Right Side : Adjacency Graph ---
    ax2.set_title('Region Adjacency Graph')
    
    # Coordinate-based node layout
    pos = {}
    counter = 0
    for j in reversed(range(grid.p)):
        for i in range(grid.n):
            # pos[index] = (column_index, row_index)
            pos[counter] = np.array([i, j])
            counter += 1

    nx.draw(
        grid.G, pos, ax=ax2, 
        with_labels=True, 
        node_color='mediumpurple', 
        node_size=600, 
        font_weight='bold',
        edge_color='gray'
    )

    # Show figure
    plt.tight_layout()
    plt.show()


def _draw_1d_boundaries(node: PartitioningTree, upper: float):
        if node.left is None: return
        segment = get_clipped_frontier(node)
        if segment is not None:
            val = segment[0][0]        
            plt.plot([val, val], [0, upper], 'k--', lw=1, alpha=0.9)
        _draw_1d_boundaries(node.left, upper)
        _draw_1d_boundaries(node.right, upper)

def plot_1D_tree(tree: PartitioningTree, size=10, alpha=0.7):
    _prepare()

    plt.scatter(
            tree.X, 
            tree.Y,
            s=size, 
            alpha=alpha,
        )
    
    _draw_1d_boundaries(tree, np.max(tree.Y))

    _show()