import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from tree_partitioning import PartitioningTree

# --------------------------------------------------

def get_constraints(node: PartitioningTree):
    '''
    Returns a list of all geometric constraints (v, threshold, sign) for a given node (partitioning tree).
    '''
    # Initialization
    constraints = []
    curr = node

    # Browsing parents recursively to get all anterior constraints
    while curr.parent is not None:
        p = curr.parent
        sign = 1 if curr == p.left else -1
        constraints.append((p.pdir, p.threshold, sign))
        curr = p
    
    return constraints


def get_clipped_frontier(
        node: PartitioningTree, 
        external_constraints: list = None
        ):
    '''
    Computes the 2 end points of the frontier segment of a node's split, clipped by previous frontiers
    and restricted to the node's bounding box.
    '''
    if node.pdir is None or node.bb_min is None or node.bb_max is None:
        return None
    
    # Frontier segment equation
    v = node.pdir
    c = node.threshold
    v_orth = np.array([-v[1], v[0]])
    anchor = v * c  # reference point on the segment

    # --- Parametrization of the end points using the bounding box ---
    # Project all 4 corners of the bounding box onto v_orth to find min/max t
    corners = np.array([
        [node.bb_min[0], node.bb_min[1]],
        [node.bb_min[0], node.bb_max[1]],
        [node.bb_max[0], node.bb_min[1]],
        [node.bb_max[0], node.bb_max[1]]
    ])
    t_values = [(corner - anchor) @ v_orth for corner in corners]
    t_min, t_max = min(t_values), max(t_values)

    # Checks for external constraints already defined
    if external_constraints is not None:
        constraints = external_constraints
    else:
        constraints = get_constraints(node)

    # Clip segment based on constraints
    for v_i, c_i, sign_i in constraints:
        dot_orth = v_orth @ v_i
        dot_anchor = anchor @ v_i

        lhs = sign_i * dot_orth
        rhs = sign_i * (c_i - dot_anchor)

        if abs(lhs) > 1e-9:  # line is not vertical
            val = rhs / lhs
            if lhs > 0:
                t_max = min(t_max, val)
            else:
                t_min = max(t_min, val)
    
    if t_min > t_max:  # no frontier
        return None
    return anchor + t_min * v_orth, anchor + t_max * v_orth



def get_shared_frontier(
        leaf_a: PartitioningTree,
        leaf_b: PartitioningTree,
        ):
    '''
    Computes the 2 end points of the shared frontier between two regions (if it exists),
    ensuring the points lie within the bounding box covering both regions.
    '''
    # --- Find path to leaf_a from tree root ---
    path_a = []
    curr = leaf_a
    while curr:
        path_a.append(curr)
        curr = curr.parent

    # --- Find lowest common ancestor of leaf_a and leaf_b ---
    lca = None
    curr = leaf_b
    while curr:
        if curr in path_a:
            lca = curr
            break
        curr = curr.parent
    
    if lca is None or lca.pdir is None:  # no common ancestor
        return None

    # --- Get constraints from both leaves ---
    combined_constraints = get_constraints(leaf_a) + get_constraints(leaf_b)
    segment = get_clipped_frontier(lca, external_constraints=combined_constraints)

    if segment is None:
        return None

    # --- Compute bounding box covering both regions ---
    bb_min = np.minimum(leaf_a.bb_min, leaf_b.bb_min)
    bb_max = np.maximum(leaf_a.bb_max, leaf_b.bb_max)

    # --- Clip segment points to stay within bounding box ---
    p1, p2 = segment
    p1_clipped = np.minimum(np.maximum(p1, bb_min), bb_max)
    p2_clipped = np.minimum(np.maximum(p2, bb_min), bb_max)

    if np.linalg.norm(p1_clipped - p2_clipped) > 1e-5:
        return p1_clipped, p2_clipped
    return None



def are_neighbors(
        leaf_a: PartitioningTree,
        leaf_b: PartitioningTree
        ):
    '''
    Tells if the regions corresponding to two leaves are neighbors i.e. share a common frontier
    '''
    return get_shared_frontier(leaf_a, leaf_b) is not None


class PartitioningGraph:
    '''
    Leaves adjacency graph for a fully extended partitioning tree.

    Parameters
    ----------
    tree : PartitioningTree
        Corresponding partitioning tree.

    Attributes
    ----------
    nodes : list[PartitioningTree]
        List of all leaves of the initial tree.
    G : networkx.Graph
        Adjacency graph of the leaves.
    
    Methods
    ----------
    frontier : Computes the 2 end points of the frontier between i and j (if they are neighbors).
    adjacency_matrix : Computes the adjacency matrix of the regions.
    frontier_matrix : Computes the matrix of all frontiers between regions (both end points if they exist).
    '''
    def __init__(
            self, 
            tree: PartitioningTree
            ):
        # Fully extends tree if not done already
        tree.fully_extend()
        
        # Creates nodes list and empty graph
        self.nodes = tree.get_leaves()
        self.nodes_X = [leaf.X for leaf in self.nodes]
        self.nodes_Y = [leaf.Y for leaf in self.nodes]

        self.G = nx.Graph()

        # Finds all neighbors = edges for each node 
        N = len(self.nodes)
        for i in range(N):
            self.G.add_node(i)
            for j in range(i+1, N):
                segment = get_shared_frontier(self.nodes[i], self.nodes[j])
                if segment is not None:
                    self.G.add_edge(i, j, geometry=segment) # end points of frontier segment stored in G[i][j]['geometry']
    

    def frontier(self, i:int, j:int):
        '''
        Computes the 2 end points of the frontier between i and j (if they are neighbors)
        '''
        if self.G.has_edge(i, j):
            return self.G[i][j]['geometry']
        return None
    
    @property
    def adjacency_matrix(self):
        '''
        Computes the adjacency matrix of the regions
        '''
        return nx.to_numpy_array(self.G)
    

    @property
    def frontier_matrix(self):
        '''
        Computes the matrix of all frontiers between regions (both end points if they exist)
        '''
        if not hasattr(self, '_cached_f_matrix'):
            N = len(self.nodes)
            matrix = [
                [self.G[i][j]['geometry'] if self.G.has_edge(i, j) else None 
                for j in range(N)] 
                for i in range(N)]
            self._cached_f_matrix = matrix
        return self._cached_f_matrix
    

    def find_location(self, x_star):
        '''
        Finds the index of the region (leaf) where a new point x_star is located.
        
        Parameters
        ----------
        x_star : array-like
            Coordinates of the point to locate.
        
        Returns
        -------
        int or None
            Index of the leaf containing x_star, or None if not found.
        '''
        x_star = np.asarray(x_star).ravel()  # convert to 1D array

        for idx, leaf in enumerate(self.nodes):
            constraints = get_constraints(leaf)
            inside = True
            for v, c, sign in constraints:
                if sign * (np.dot(v, x_star) - c) > 1e-9:
                    inside = False
                    break
            if inside:
                return idx
        return None
